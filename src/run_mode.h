/* The process-wide RUN MODE (--run-mode gentle | balanced | performance, or DTFE_RUN_MODE), set at the end of
   readOptions before any thread exists: thread budget, GPU deposit buffer target, PS-DTFE exact item cap. The thread
   helpers return the unlimited value unless the mode LOWERED the budget; defaults are performance's. Plain C++17. */
#ifndef DTFE_RUN_MODE_H
#define DTFE_RUN_MODE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#ifdef OPEN_MP
#include <omp.h>
#endif

enum class RunMode : int { Performance = 0, Balanced = 1, Gentle = 2 };

inline char const *runModeName(RunMode m)
{
    return m == RunMode::Gentle ? "gentle" : m == RunMode::Balanced ? "balanced" : "performance";
}

struct RunModeState
{
    RunMode mode = RunMode::Performance;
    int ompDefault = 0;              // N0: the OpenMP default at start-up (hardware_concurrency without OpenMP)
    int coreBudget = 0;              // the threads the mode allows
    bool budgetLowered = false;      // coreBudget < ompDefault: only then do the helpers below differ from the unlimited values
    double gpuTargetSec = 0.25;      // the GPU deposit's buffer target (gpu_metal_retry.h gpuPace)
    size_t psItemCap = 4000000;      // PS-DTFE's exact items per GPU buffer (ps_metal_host.cc)
};

inline RunModeState &runModeState()
{
    static RunModeState s;
    return s;
}

// std::thread::hardware_concurrency(), at least 1, capped by a lowered budget
inline int dtfeHardwareThreads()
{
    int const hw = std::max( 1, int( std::thread::hardware_concurrency() ) );
    RunModeState const &r = runModeState();
    return r.budgetLowered ? std::max( 1, std::min( hw, r.coreBudget ) ) : hw;
}

// omp_get_max_threads(), at least 1, or a lowered budget (dtfeHardwareThreads without OpenMP)
inline int dtfeOmpMax()
{
#ifdef OPEN_MP
    RunModeState const &r = runModeState();
    return r.budgetLowered ? std::max( 1, r.coreBudget ) : std::max( 1, omp_get_max_threads() );
#else
    return dtfeHardwareThreads();
#endif
}

// for a std::thread that runs OpenMP regions (it starts from the runtime's default, not the main thread's): the
// lowered budget (DTFE.cpp's look-ahead thread calls it first)
inline void dtfeAdoptCoreBudget()
{
#ifdef OPEN_MP
    if ( runModeState().budgetLowered ) omp_set_num_threads( runModeState().coreBudget );
#endif
}

// the mode's thread budget from N0 and the performance cores (0: unknown): gentle half of base, balanced base less
// a quarter (1 to 4 threads), performance N0; base = min(N0, perfCores) where the performance cores are known
inline int runModeCoreBudget(RunMode m, int N0, int perfCores)
{
    N0 = std::max( 1, N0 );
    int const base = perfCores > 0 ? std::min( N0, perfCores ) : N0;
    if ( m == RunMode::Gentle ) return std::max( 1, base / 2 );
    if ( m == RunMode::Balanced ) return std::max( 1, base - std::clamp( int( std::lround( base / 4.0 ) ), 1, 4 ) );
    return N0;
}

#endif
