/* The process-wide RUN MODE: how hard a run may lean on the machine (--run-mode gentle | balanced | performance, or
   DTFE_RUN_MODE; User_options::resolveRunMode and applyRunMode set it at the end of readOptions, before any thread
   exists). The mode sets the thread budget, the GPU deposit's buffer target and PS-DTFE's exact item cap; the thread
   helpers below return exactly today's value unless the mode LOWERED the budget (budgetLowered), so performance (the
   binaries' default) and every run with OMP_NUM_THREADS given keep today's thread counts. The state is ONE object
   across all translation units, through the inline function's static; its defaults are performance's values, so the
   library and any caller that skips readOptions run as today. runModeCoreBudget is the budget's formula, a pure
   function (tests/run_mode_check.sh recomputes it). Plain C++17 with no other header of the project, so any
   translation unit can include it. */
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
    bool budgetLowered = false;      // coreBudget < ompDefault: only then do the helpers below differ from today's
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
