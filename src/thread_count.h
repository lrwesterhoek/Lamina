/* The thread count of one triangulation's CPU work -- the PS-DTFE CPU deposit (ps_interpolation.cc
   psDepositThreads), the standard binary's '_a' cell average, exact and Monte-Carlo (averaged_interpolation_1.cc
   exactAverageThreads), and the point evaluation of both binaries (ps_point_eval.cc pointEvalThreads) -- in ONE place:
   - every core when this is the only triangulation being processed (a single-domain run);
   - inside a partition loop's OpenMP team of concurrent partitions (or DTFE_parallel's team of sub-domains), the
     cores left per team member, so the two levels never oversubscribe;
   - a positive value of the environment variable 'envOverride' pins it (benchmarks, tests): DTFE_DEPOSIT_THREADS
     for the two deposits, DTFE_PTS_THREADS for the point evaluation.
   The callers run their threads as std::thread pools, not as a nested OpenMP region (disabled by default). The
   core budget of the run mode (run_mode.h) is the OpenMP maximum here: dtfeOmpMax() and dtfeHardwareThreads().
   DTFE.cpp's dtfeParallelThreads is another rule (the OpenMP maximum capped by '--max-concurrent', with no
   override) and stays out of this one. Below it: the process-wide core pool (CorePool) of the work that may follow
   the free cores, and the range helpers that run on them (parallelRanges, poolRanges). */
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

/* THE CORE POOL of the PS-DTFE CPU work that can follow the free cores (the CPU deposit's dynamic team and a pass's
   cell loops, ps_interpolation.cc; the parallel sub-grid merge and the normalization, quantities.cc). Process-wide:
   - total(): the cores the process may use (the OpenMP maximum, as coresPerTeam counts them);
   - holders(): the threads that hold a core of their own -- inside a partition loop its LIVE partitions (DTFE.cpp's
     planned concurrency, at most the partitions not yet finished: triangulation.cpp's two counters, the bound its
     parallel insertion takes too), else the OpenMP team the caller runs in, else the caller alone;
   - extra(): the helper threads running beside their callers now: the pool's own (each claimed by take and given
     back by give when it ends) and those of the teams that do not ask it, each counted by a Hold for its lifetime --
     the PS-DTFE grid pass's static teams (ps_interpolation.cc: the copy- and slab-mode CPU deposit, a pinned
     dynamic team, the slab binning, the copies' merge, the fixed-point bound walk) and the parallel insertion's TBB
     arena (triangulation.cpp). Not counted: the point evaluation's team (ps_point_eval.cc), the standard binary's
     '_a' average (averaged_interpolation_1.cc) and a partition loop's look-ahead thread (DTFE.cpp; a parallel
     insertion it runs is counted).
   free = total - holders - extra: what a helper may take (take) without oversubscribing the cores; a partition that
   finishes frees its core here. A counted team does not wait for free cores (its size is its own rule's): the pool's
   users yield to it, not it to them. A claim never changes a result: only work whose outcome is independent of how
   many threads share it (and which takes which block) asks the pool -- the integer fixed-point sums, the counts and
   flags of the default single build, and a merge that adds every cell once. */
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

// A team that does not ask the pool (a fixed size, beside its caller, a holder already) counts its other 'n' threads
// in extra() while it runs: constructed before its first thread starts, destroyed after the last has joined. The
// pool's users then see those cores taken. Counting a thread that failed to start, or a moment before or after the
// threads run, only has them take fewer helpers (take stops at no free core); it never changes a result.
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
