/* The run's progress sidecar '<output>.runlog.progress': a small JSON file ({pid, phase, done, total, elapsed_s, eta_s,
   rss_gb, gpu_retries, gpu_retries_by_cause, predicted_gb, peak_gb, updated}) rewritten at every phase change and
   partition so the launcher can follow a run. Memory is in GB of 1e9 bytes, -1 = unknown. */

/* Written atomically ('.tmp.<pid>' then rename). Removed on success and by the atexit handler on an error (throwError
   ends in exit(), which skips main()'s destructors); a leftover means a live or killed run, or phase 'rescued' (exit 3:
   outputs not where the launcher looks). Off for '--serve', '--auto-tune-report' and DTFE_PROGRESS=0. Header-only. */
#ifndef RUN_PROGRESS_H
#define RUN_PROGRESS_H

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <unistd.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "user_options.h"
#include "env_knobs.h"

#ifdef PHASE_SPACE
// failed GPU command buffers (3 failures of one chunk send the partition to the CPU), counted by the GPU host
// (ps_metal_host.cc), defined in ps_interpolation.cc
extern std::atomic<int> psGpuRetries;
extern std::atomic<int> psGpuRetriesByCause[4];   // innocent, timeout, oom, other
#else
// the standard binary's, same policy (dtfe_metal_host.cc), defined in averaged_interpolation_1.cc; zero in a CPU-only build
extern std::atomic<int> dtfeGpuRetries;
extern std::atomic<int> dtfeGpuRetriesByCause[4];   // innocent, timeout, oom, other
// the sub-domain deposits since the last reset that were meant for the GPU (asked or picked) and ran on the CPU instead
int dtfeGpuFallbacksTaken(bool reset = false);
#endif

struct RunProgress
{
    bool on = false;
    std::string path;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::string phase = "starting";
    long done = 0, total = 0;
    double eta = -1.;
    double predicted = -1.;   // the auto-tuner's predicted peak, bytes (-1: none)
    double loopPeak = -1.;    // the peak when the post-processing began (the partition loop's), bytes (-1: none)
    double startPeak = -1.;   // ... and once the partition loop's full grids were sized, before any partition
    double preGridsPeak = -1.;  // ... and just before they were sized
    std::mutex m;
};

inline RunProgress &runProgressState()
{
    static RunProgress p;
    return p;
}

// the resident memory of this process now (not the peak), in bytes; -1 when unknown
inline double runProgressRss()
{
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count ) == KERN_SUCCESS )
        return double( info.resident_size );
    return -1.;
#else
    double pages = -1.;
    if ( FILE *f = std::fopen( "/proc/self/statm", "r" ) )
    {
        double size = 0.;
        if ( std::fscanf( f, "%lf %lf", &size, &pages ) != 2 ) pages = -1.;
        std::fclose( f );
    }
    return pages < 0. ? -1. : pages * double( sysconf( _SC_PAGESIZE ) );
#endif
}

// the peak memory of this process so far, in bytes; -1 when unknown. macOS: the peak physical footprint (the
// ledger /usr/bin/time -l reports as 'peak memory footprint'; it includes the GPU's shared buffers), else the
// maximum RSS. *maxRss (optional) gets the maximum RSS in bytes either way (ru_maxrss: bytes on macOS, KB on Linux)
inline double runProgressPeak(double *maxRss = nullptr)
{
    double rss = -1.;
    struct rusage ru;
    if ( getrusage( RUSAGE_SELF, &ru ) == 0 )
#ifdef __APPLE__
        rss = double( ru.ru_maxrss );
#else
        rss = double( ru.ru_maxrss ) * 1024.;
#endif
    if ( maxRss ) *maxRss = rss;
#ifdef __APPLE__
    task_vm_info_data_t vm;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if ( task_info( mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &count ) == KERN_SUCCESS
         and count >= TASK_VM_INFO_REV3_COUNT and vm.ledger_phys_footprint_peak > 0 )
        return double( vm.ledger_phys_footprint_peak );
#endif
    return rss;
}

// the auto-tuner's predicted peak (bytes), for the sidecar and the SUMMARY's predicted-vs-measured line
inline void runProgressSetPredicted(double bytes)
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    p.predicted = bytes > 0. ? bytes : -1.;
}

inline double runProgressPredicted()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    return p.predicted;
}

// the peak so far when the post-processing begins (DTFE.cpp): the SUMMARY prints it beside the run's peak, so a
// log says which phase set the peak (the auto-tuner books both). Read once, at that moment.
inline void runProgressNoteLoopPeak()
{
    double const peak = runProgressPeak();
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    p.loopPeak = peak;
}

// the peak once the partition loop has sized its full grids, before its first partition: under --scratch-dir on macOS
// each scratch-backed grid is zero-filled in anonymous memory first (scratch_alloc.h ZeroFillScope), which can be the
// run's peak. 'before' is the peak read just before the sizing, so the SUMMARY can tell the two apart. Read once.
inline void runProgressNoteStartPeak(double const before)
{
    double const peak = runProgressPeak();
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    p.startPeak = peak;
    p.preGridsPeak = before;
}

inline double runProgressPreGridsPeak()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    return p.preGridsPeak;
}

inline double runProgressStartPeak()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    return p.startPeak;
}

inline double runProgressLoopPeak()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    return p.loopPeak;
}

inline int runProgressGpuRetries()
{
#ifdef PHASE_SPACE
    return psGpuRetries.load();
#else
    return dtfeGpuRetries.load();
#endif
}

// the failed attempts of one cause: 0 innocent victim, 1 timeout, 2 out of memory, 3 other
inline int runProgressGpuRetries(int cause)
{
#ifdef PHASE_SPACE
    return ( cause >= 0 and cause < 4 ) ? psGpuRetriesByCause[cause].load() : 0;
#else
    return ( cause >= 0 and cause < 4 ) ? dtfeGpuRetriesByCause[cause].load() : 0;
#endif
}

// (re)writes the sidecar from the current state; the caller holds the mutex
inline void runProgressWriteLocked(RunProgress &p)
{
    if ( not p.on ) return;
    double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - p.start ).count();
    double const rss = runProgressRss();
    double const peak = runProgressPeak();
    // the four causes loaded ONCE and the total written as THEIR sum: a host bumps psGpuRetries and
    // the cause's counter as two increments, and this write may run while another partition's deposit fails
    // (--max-concurrent >= 2), so five separate loads could print a total its causes do not add up to
    int byCause[4];
    for ( int k = 0; k < 4; ++k ) byCause[k] = runProgressGpuRetries( k );
    int const retries = byCause[0] + byCause[1] + byCause[2] + byCause[3];
    std::string const tmp = p.path + ".tmp." + std::to_string( long( ::getpid() ) );
    FILE *f = std::fopen( tmp.c_str(), "w" );
    if ( not f ) return;   // the output folder is not writable now: the progress is a convenience, never fatal
    std::fprintf( f, "{\"pid\": %ld, \"phase\": \"%s\", \"done\": %ld, \"total\": %ld, \"elapsed_s\": %.1f, "
                     "\"eta_s\": %.1f, \"rss_gb\": %.3f, \"gpu_retries\": %d, \"gpu_retries_by_cause\": "
                     "{\"innocent\": %d, \"timeout\": %d, \"oom\": %d, \"other\": %d}, \"predicted_gb\": %.3f, "
                     "\"peak_gb\": %.3f, \"updated\": %ld}\n",
                  long( ::getpid() ), p.phase.c_str(), p.done, p.total, elapsed, p.eta,
                  rss < 0. ? -1. : rss / 1.e9, retries, byCause[0], byCause[1], byCause[2], byCause[3],
                  p.predicted < 0. ? -1. : p.predicted / 1.e9, peak < 0. ? -1. : peak / 1.e9,
                  long( std::time( nullptr ) ) );
    bool const ok = std::fflush( f ) == 0;
    std::fclose( f );
    if ( not ok or std::rename( tmp.c_str(), p.path.c_str() ) != 0 )
        std::remove( tmp.c_str() );
}

// exit() of a run still armed (an error: throwError): the sidecar goes, so the launcher reads the failure from the
// run log. The lock is only tried: a worker's throwError may call exit() mid-write of another thread; a write in
// flight can still put the file back in that window (accepted). No logging here: exit() may have cut a message.
inline void runProgressAtExit()
{
    RunProgress &p = runProgressState();
    std::unique_lock<std::mutex> lock( p.m, std::try_to_lock );
    if ( not p.on or p.path.empty() ) return;
    std::remove( p.path.c_str() );
    std::remove( ( p.path + ".tmp." + std::to_string( long( ::getpid() ) ) ).c_str() );
    if ( lock.owns_lock() ) p.on = false;   // no later write by a running thread
}

// arms the sidecar for this run (main.cpp, right after the options are read)
inline void runProgressInit(User_options const &u)
{
    RunProgress &p = runProgressState();   // constructed BEFORE the handler is registered, so destroyed after it runs
    std::lock_guard<std::mutex> lock( p.m );
    p.on = not u.outputFilename.empty() and not u.psServe and not u.autoTuneReport
           and Env::flag( "DTFE_PROGRESS", true );   // DTFE_PROGRESS=0 writes none
    p.path = u.outputFilename + ".runlog.progress";
    p.start = std::chrono::steady_clock::now();
    static bool registered = false;
    if ( p.on and not registered )
    {
        registered = true;
        std::atexit( runProgressAtExit );
    }
    runProgressWriteLocked( p );
}

// a new phase ('reading', 'triangulation', 'partitions', 'post-processing', 'writing'); done/total reset
// unless given
inline void runProgressPhase(char const *phase, long done = 0, long total = 0, double eta = -1.)
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    p.phase = phase;
    p.done = done;
    p.total = total;
    p.eta = eta;
    runProgressWriteLocked( p );
}

// the run finished and its outputs are in place: the sidecar goes
inline void runProgressFinish()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    if ( p.on ) std::remove( p.path.c_str() );
    p.on = false;
}

// the outputs are in the rescue folder or could not be put in place (exit 3): the 'rescued' sidecar stays and nothing writes over it
inline void runProgressDisarm()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    p.on = false;
}

#endif
