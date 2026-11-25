/* The thread count of one triangulation's CPU work (PS deposit, standard '_a' average, point evaluation) in ONE place:
   every core when alone, the cores left per member inside a partition loop's team, or a positive 'envOverride' value.
   The run mode's core budget (run_mode.h) is the OpenMP maximum here. Below: the core pool and the range helpers. */
#ifndef DTFE_THREAD_COUNT_H
#define DTFE_THREAD_COUNT_H

#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>
#include <vector>
#ifdef OPEN_MP
#include <omp.h>
#endif

#include "env_knobs.h"
#include "run_mode.h"

// the PS partition loop's partitions in flight under its dynamic concurrency (DTFE.cpp; 0 otherwise): the core pool's
// holders and the per-partition teams count these, not the loop's threads (some of which wait in the memory gate)
extern std::atomic<int> dtfePartitionsInFlight;

inline int coresPerTeam(char const *envOverride)
{
    if ( long const v = Env::integer( envOverride, 0 ); v > 0 )
        return int( v );
    int cores = dtfeHardwareThreads();
#ifdef OPEN_MP
    cores = dtfeOmpMax();
    if ( omp_in_parallel() )
    {
        int const live = dtfePartitionsInFlight.load( std::memory_order_relaxed );   // (dynamic concurrency)
        return std::max( 1, cores / std::max( 1, live > 0 ? live : omp_get_num_threads() ) );
    }
#endif
    return std::max( 1, cores );
}

/* THE CORE POOL for PS-DTFE CPU work that can follow the free cores: free = total() - holders() - extra(). total() is
   the OpenMP maximum; holders() the threads with a core of their own (a loop's live partitions, else the caller's
   team, else the caller); extra() the helpers running beside their callers (the pool's own take/give, and the teams
   that do not ask it, each counted by a Hold). A counted team does not wait for free cores: pool users yield to it.
   A claim never changes a result: only work independent of the thread count (fixed-point sums, a merge) asks. */
extern std::atomic<int> dtfeTriangulationConcurrency;
extern std::atomic<int> dtfeTriangulationsRemaining;


namespace CorePool
{
inline int total()
{
#ifdef OPEN_MP
    return dtfeOmpMax();
#else
    return dtfeHardwareThreads();
#endif
}

inline std::atomic<int> &extra()
{
    static std::atomic<int> n{ 0 };
    return n;
}

inline int holders()
{
    if ( int const live = dtfePartitionsInFlight.load( std::memory_order_relaxed ); live > 0 )
        return live;   // the dynamic loop's partitions in flight (DTFE.cpp)
    int const planned = dtfeTriangulationConcurrency.load( std::memory_order_relaxed );
    if ( planned > 0 )
        return std::max( 1, std::min( planned, dtfeTriangulationsRemaining.load( std::memory_order_relaxed ) ) );
#ifdef OPEN_MP
    if ( omp_in_parallel() ) return std::max( 1, omp_get_num_threads() );
#endif
    return 1;
}

// claims up to 'want' helper cores among the free ones; returns how many it got (give them back with give)
inline int take(int const want)
{
    int got = 0;
    while ( got < want )
    {
        int const ex = extra().load( std::memory_order_relaxed );
        if ( total() - holders() - ex <= 0 ) break;
        int e = ex;
        if ( extra().compare_exchange_weak( e, ex + 1, std::memory_order_relaxed ) ) ++got;
    }
    return got;
}

inline void give(int const n) { if ( n > 0 ) extra().fetch_sub( n, std::memory_order_relaxed ); }

// A team that does not ask the pool counts its other 'n' threads in extra() while it runs (constructed before its
// first thread starts, destroyed after the last joins). A miscount only makes pool users take fewer helpers.
struct Hold
{
    explicit Hold(int const n) : held( std::max( 0, n ) )
    { if ( held > 0 ) extra().fetch_add( held, std::memory_order_relaxed ); }
    ~Hold() { give( held ); }
    Hold(Hold const &) = delete;
    Hold &operator=(Hold const &) = delete;
private:
    int const held;
};
}

// f(lo, hi) over [0, n) in 'threads' contiguous ranges, the caller running the first; every range runs, and the
// first exception is rethrown after all have joined.
template <class F>
inline void parallelRanges(size_t const n, int const threads, F const &f)
{
    size_t const nT = size_t( std::max<long long>( 1, std::min<long long>( threads, (long long)( n > 0 ? n : 1 ) ) ) );
    if ( nT <= 1 ) { f( size_t(0), n ); return; }
    std::vector<std::exception_ptr> errors( nT );
    auto part = [&](size_t const t)
    {
        try { f( n * t / nT, n * (t + 1) / nT ); }
        catch (...) { errors[t] = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    pool.reserve( nT - 1 );
    for (size_t t = 1; t < nT; ++t) pool.emplace_back( part, t );
    part( 0 );
    for (std::thread &x : pool) x.join();
    for (std::exception_ptr const &e : errors)
        if ( e ) std::rethrow_exception( e );
}

// parallelRanges on the caller plus the pool's free cores: at most 'cap' threads in all, and only as many as give each
// at least 'minPerThread' items; the helpers are claimed from the pool for the call and given back after it.
template <class F>
inline void poolRanges(size_t const n, size_t const minPerThread, int const cap, F const &f)
{
    long long const byWork = (long long)( n / std::max<size_t>( 1, minPerThread ) );
    int const want = int( std::max<long long>( 0, std::min<long long>( cap, byWork ) - 1 ) );
    int const got = want > 0 ? CorePool::take( want ) : 0;
    try { parallelRanges( n, 1 + got, f ); }
    catch (...) { CorePool::give( got ); throw; }
    CorePool::give( got );
}

#endif
