/* Auto-selection of --partition and --max-concurrent from the simulation actually being
   processed (particle count, grid, requested fields) and the machine (physical RAM, cores).
   Called from DTFE_setup once the input is loaded and the options are finalized; it only
   fills in values the user did NOT give -- explicit flags always win.

   The peak-memory model is calibrated against measured runs (TNG50-4 512^3 = 45 GB measured
   vs ~50 GB modeled; TNG300-3 5x5x5/mc2 target ~40 GB vs ~36 GB modeled) and deliberately
   errs HIGH so the tuner never picks a configuration that measured tighter than predicted.
   Model constants (bytes/vertex etc.) were pinned with a compiled sizeof probe; see the
   README 'Memory' section for the empirical anchors.

   Memory budget (autoTuneBudget): 60% of physical RAM, and never more than what the OTHER
   programs leave free at start-up minus 4 GB of headroom -- the machine stays usable while a
   run goes (80% of RAM alone makes a 64 GB Mac swap with a browser, the GUI and the desktop
   open next to a 50 GB run).

   Environment overrides (for tests and unusual machines):
     DTFE_RAM_GB        physical-RAM override in GB (default: queried from the OS); simulates a
                        machine, so the live free-memory term is skipped (tests stay deterministic)
     DTFE_MEM_FRACTION  share of RAM the budget may take (default 0.60; 0.1 .. 0.95)
     DTFE_MEM_BUDGET_GB the budget itself, in GB -- overrides both rules above, but is still capped by what
                        the other programs leave free minus the headroom (unless DTFE_RAM_GB simulates a
                        machine): a budget above the free memory gets a run SIGKILLed
     DTFE_FREE_GB       test hook: the memory the other programs leave free, in GB (default: queried); the
                        partition memory gate and the budget cap read it instead of the OS
     DTFE_AUTO_MINN     minimum particle count before auto-partitioning engages (default 2e6;
                        below it a single domain always fits and partition seams are not worth it) */

/* This header is the tuner's interface: the machine queries (RAM, free memory, the budget, the cores), the report,
   the partition memory gate and the entry points. The model itself -- the bytes per cell and per vertex, the
   planners, the '--serve' split and their constants -- is in auto_tune.cc, an object of its own, so an edit of the
   model rebuilds that file only. A comment elsewhere that points at 'auto_tune.h <name>' means the pair. */

#ifndef AUTO_TUNE_HEADER
#define AUTO_TUNE_HEADER

// (what this interface itself uses; the model's own includes are in auto_tune.cc)
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <iterator>    // std::distance: the memory gate's blockers
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <set>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#else
#include <unistd.h>
#endif
#ifdef OPEN_MP
#include <omp.h>
#endif

#include "user_options.h"
#include "particle_data.h"
#include "env_knobs.h"


// The free memory the tuner leaves untouched: the budget is capped at what the other programs leave free minus this,
// and the partition memory gate waits while the free memory minus this is below one partition's need
constexpr double kAutoTuneHeadroom = 4.e9;

// Physical RAM in bytes (DTFE_RAM_GB env overrides; 8 GB fallback if the query fails).
inline double autoTunePhysicalRAM()
{
    if ( double const v = Env::real( "DTFE_RAM_GB", 0. ); v > 0. )
        return v * 1.e9;
#ifdef __APPLE__
    int64_t mem = 0;
    size_t len = sizeof(mem);
    if ( sysctlbyname("hw.memsize", &mem, &len, NULL, 0)==0 and mem>0 )
        return double(mem);
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long pageSize = sysconf(_SC_PAGE_SIZE);
    if ( pages>0 and pageSize>0 )
        return double(pages) * double(pageSize);
#endif
    return 8.e9;
}

// The memory the PS-DTFE CPU deposit threads' private accumulator copies may take per partition:
// 20% of physical RAM (at least 2 GB), or DTFE_DEPOSIT_MEM_GB. ps_interpolation.cc lowers its
// thread count to fit (a 256^3 grid with density, velocity and dispersion is ~1 GB per float copy);
// the partition model below budgets the same amount. With the fixed-point grids and the shared counts
// and flags the single build copies nothing by default: the budget binds only the float sums (double
// build, PS_GPU_FLOAT_*).
inline double psDepositThreadBytes()
{
    if ( double const v = Env::real( "DTFE_DEPOSIT_MEM_GB", -1. ); v >= 0. )
        return v * 1.e9;
    return std::max( 2.e9, 0.2 * autoTunePhysicalRAM() );
}

// Bytes free for NEW allocations right now (what every program, this one included, leaves free);
// < 0 when the OS cannot tell. DTFE_FREE_GB overrides it (tests).
//   macOS: RAM minus Activity Monitor's "Memory Used" (app memory = anonymous - purgeable,
//          + wired + compressed); Linux: MemAvailable. File cache counts as free: it is dropped.
inline double autoTuneSystemFree(double *ownBytes = nullptr)
{
    if ( double const v = Env::real( "DTFE_FREE_GB", 0. ); v > 0. )
    {
        if ( ownBytes ) *ownBytes = 0.;
        return v * 1.e9;
    }
#ifdef __APPLE__
    int64_t mem = 0;
    size_t len = sizeof(mem);
    vm_size_t page = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if ( sysctlbyname("hw.memsize", &mem, &len, NULL, 0)!=0 or mem<=0
         or host_page_size(mach_host_self(), &page)!=KERN_SUCCESS
         or host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count)!=KERN_SUCCESS )
        return -1.;
    double const used = ( double(vm.internal_page_count) - double(vm.purgeable_count)
                          + double(vm.wire_count) + double(vm.compressor_page_count) ) * double(page);
    double own = 0.;
    task_vm_info_data_t ti;
    mach_msg_type_number_t tcount = TASK_VM_INFO_COUNT;
    if ( task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &tcount)==KERN_SUCCESS )
        own = double(ti.phys_footprint);
    if ( ownBytes ) *ownBytes = own;
    return double(mem) - used;
#else
    double availKB = -1., ownKB = 0.;
    char key[64];
    double value;
    if ( FILE *f = std::fopen("/proc/meminfo", "r") )
    {
        while ( std::fscanf(f, "%63s %lf kB\n", key, &value)==2 )
            if ( std::string(key)=="MemAvailable:" ) { availKB = value; break; }
        std::fclose(f);
    }
    if ( FILE *f = std::fopen("/proc/self/status", "r") )
    {
        char line[256];
        while ( std::fgets(line, sizeof(line), f) )
            if ( std::sscanf(line, "VmRSS: %lf kB", &value)==1 ) { ownKB = value; break; }
        std::fclose(f);
    }
    if ( ownBytes ) *ownBytes = ownKB * 1024.;
    return availKB < 0. ? -1. : availKB * 1024.;
#endif
}

// Bytes the OTHER programs leave free right now, plus what this process already holds (its
// particles etc., which the run's model counts itself); < 0 when the OS cannot tell.
inline double autoTuneFreeForRun()
{
    double own = 0.;
    double const free = autoTuneSystemFree( &own );
    return free < 0. ? -1. : free + own;
}

// The memory budget the tuner plans against, and a one-line account of how it was set.
inline double autoTuneBudget(double ramBytes, std::string &why)
{
    char buf[256];
    bool const simulated = Env::real( "DTFE_RAM_GB", 0. ) > 0.;
    double const headroom = kAutoTuneHeadroom;
    {
        double const v = Env::real( "DTFE_MEM_BUDGET_GB", 0. );
        if ( v > 0. )
        {
            snprintf( buf, sizeof(buf), "%.3g GB set by DTFE_MEM_BUDGET_GB", v );
            why = buf;
            // ... but never more than the other programs leave free minus the headroom: the budget is a
            // plan, the machine's free memory the fact. Only a cut of more than 0.5 GB is one: 44 GB free less 4
            // against 40 asked is the same budget, and the reason line must not say "asks 40 GB, but ..." for it
            double const freeForRun = simulated ? -1. : autoTuneFreeForRun();
            if ( freeForRun > 0. and freeForRun - headroom < v * 1.e9 - 0.5e9 )
            {
                double const capped = std::max( freeForRun - headroom, std::min( v * 1.e9, 1.e9 ) );
                snprintf( buf, sizeof(buf), "%.3g GB: DTFE_MEM_BUDGET_GB asks %.3g GB, but the other programs leave "
                          "%.3g GB free, minus %.3g GB headroom", capped/1.e9, v, freeForRun/1.e9, headroom/1.e9 );
                why = buf;
                return capped;
            }
            return v * 1.e9;
        }
    }
    double frac = 0.60;
    if ( double const v = Env::real( "DTFE_MEM_FRACTION", frac ); v >= 0.1 and v <= 0.95 )
        frac = v;
    double budget = frac * ramBytes;
    snprintf( buf, sizeof(buf), "%.3g GB = %.0f%% of %.3g GB RAM", budget/1.e9, 100.*frac, ramBytes/1.e9 );
    why = buf;
    double const freeForRun = simulated ? -1. : autoTuneFreeForRun();
    if ( freeForRun > 0. and freeForRun - headroom < budget )
    {
        budget = std::max( freeForRun - headroom, 1.e9 );
        snprintf( buf, sizeof(buf), "%.3g GB: other programs leave %.3g GB free, minus %.3g GB headroom "
                  "(%.0f%% of RAM would be %.3g GB)", budget/1.e9, freeForRun/1.e9, headroom/1.e9,
                  100.*frac, frac*ramBytes/1.e9 );
        why = buf;
    }
    return budget;
}

inline int autoTuneCores()
{
#ifdef OPEN_MP
    return std::max( omp_get_max_threads(), 1 );
#else
    return 1;
#endif
}

// What the tuner decided and predicted, for '--auto-tune-report' (one machine-readable line; the
// GUI's memory check parses it). valid = false in the modes the tuner does not handle.
struct AutoTuneReport
{
    bool   valid = false;
    int    partition = 1, maxConcurrent = 1;
    double predicted = 0., fixed = 0., budget = 0., pts = 0.;   // bytes
    double perPartition = 0.;   // bytes ONE partition in flight adds (the run's memory gate waits for this much)
    double lookAhead = 0.;      // bytes the partition look-ahead adds to 'predicted' (0: none runs)
    double gpuSnapshot = 0.;    // bytes of the GPU deposit's host snapshot (with --ps-window: and of the deposit's window
                                // grids), held by ONE partition at a time (in 'predicted')
    double streamsEstimate = -1., streamsBudgeted = 0.;
    std::string line() const
    {
        if ( not valid ) return "AUTO-TUNE-REPORT unsupported";
        // (the GPU snapshot and the look-ahead are in 'predicted' too, listed so a check can see each term; the
        // parsers read key=value pairs, and over_budget stays last)
        char b[480];
        snprintf( b, sizeof(b), "AUTO-TUNE-REPORT partition=%d mc=%d predicted_gb=%.2f budget_gb=%.2f "
                  "fixed_gb=%.2f pts_gb=%.3f streams_est=%.3f streams_budget=%.3f snapshot_gb=%.3f lookahead_gb=%.3f "
                  "over_budget=%d",
                  partition, maxConcurrent, predicted/1.e9, budget/1.e9, fixed/1.e9, pts/1.e9,
                  streamsEstimate, streamsBudgeted, gpuSnapshot/1.e9, lookAhead/1.e9, predicted > budget ? 1 : 0 );
        return b;
    }
};

// Chooses --partition / --max-concurrent when the user did not give them (auto_tune.cc). 'metalActive' =
// the GPU deposit will actually run (flag given AND compiled in). Mutates userOptions.
AutoTuneReport autoTunePartitioning(User_options &u,
                                    size_t const nParticles,
                                    bool const userSampling,
                                    bool const metalActive,
                                    std::vector<Particle_data> const *particles = nullptr);

/* '--serve' on data too large for ONE tessellation: turn it into a composite server (ps_point_eval.cc)
   of n^3 partitions -- when '--tessellation-cache' lets the partitions live on disk. Without the
   cache a composite holds every partition in memory, which is MORE than one tessellation, so the
   tuner then only warns. Model (auto_tune.cc): a tessellation costs kPSVertexBytes / kStdVertexBytes per vertex
   plus ~160 B/vertex for the server's cell index; the particle array (freed once every partition is built) coexists with the
   concurrent builds. Picks the smallest n (2..8) that leaves room for two partitions next to the
   particles, then as many concurrent builds as fit. A user-given --partition is never touched. */
void autoTuneServe(User_options &u, size_t const nParticles);

#ifndef PHASE_SPACE
/* What the standard partition loop's LOOK-AHEAD holds for one partition (DTFE.cpp): its padded particle
   copy and, built from it on a team of their own, the tessellations of its thread sub-domains (each padded again on
   its own sub-box). The UNIFORM count: 'nOwn' particles in the partition (the split balances them), its box lengths
   'len' and per-side padding 'pad' (the mean of the two sides), split 'tGrid' threads per axis. The plan (auto_tune.cc)
   counts the same from the particles (its look-ahead term); the loop checks the free memory against this before each
   one. */
double stdLookAheadBytes(User_options const &u, double const nOwn, double const len[NO_DIM],
                         double const pad[NO_DIM], size_t const tGrid[NO_DIM]);
#endif

/* The partition MEMORY GATE. The auto-tuner plans a peak from a model; the machine's free memory
   is the fact, and other programs change it during a run (a browser and the launcher opened beside a run
   planned within its budget can still get it SIGKILLed by macOS). Before a partition starts, it waits
   -- in 2 s steps -- while the memory the system leaves free, minus the headroom, is below what one partition
   adds (AutoTuneReport::perPartition) AND another partition is in flight (whose finishing frees memory). With
   none in flight it always starts, saying so: the gate can delay, never deadlock. A skipped partition and the
   speed split's ordered merge never pass it; a cache-only one does (its loaded tessellation holds the same
   memory). need = 0 (no auto-tune estimate: a user-given --partition) disables it.
   ORDERED (the default): a finished partition can wait for every
   lower-numbered one to merge while it still holds its memory (only when the staging room is full:
   DTFE.cpp's staged ordered merge), so a partition waiting here for a HIGHER-numbered one to free memory could wait
   forever; in that mode a partition waits only while a lower-numbered one is in flight (those never wait for it).
   The partition entering is always the highest index handed out so far (the loop's dynamic schedule hands them
   out in order), so every partition in flight is lower-numbered and the two modes wait alike. A STAGED partition
   (finished, its sub-grids parked until its turn) has left the gate: its memory shows in the free-memory reading,
   and it was staged only while that reading left room for one more partition. That holds for the
   partitions the gate guards too: one whose sub-grids the free memory cannot hold beside one more partition (need)
   is SPILLED to disk (DTFE.cpp, ps_merge_spill.h) rather than parked in RAM, so parked sub-grids do not starve
   the gate down to one partition at a time (nor, with one in flight regardless, push the machine into swap). */
struct PartitionMemoryGate
{
    double need = 0., headroom = kAutoTuneHeadroom;
    bool ordered = false;
    std::mutex m;
    std::condition_variable cv;
    int inFlight = 0, held = 0;
    std::multiset<int> flying;   // the in-flight partition indices (the ordered mode's blockers)
    bool saidAlone = false;
    explicit PartitionMemoryGate(double needBytes, bool orderedMerge = false) : need(needBytes), ordered(orderedMerge) {}

    // the in-flight partitions whose finishing this one may wait for
    int blockers(int partition) const
    {
        if ( not ordered ) return inFlight;
        return int( std::distance( flying.begin(), flying.lower_bound( partition ) ) );
    }

    void enter(int partition, int verbose)
    {
        std::unique_lock<std::mutex> lock( m );
        if ( need > 0. )
        {
            bool waited = false;
            double free = autoTuneSystemFree();
            while ( free >= 0. and free - headroom < need and blockers( partition ) > 0 )
            {
                if ( not waited )
                {
                    ++held;
                    if ( verbose > 0 )
                        std::fprintf( stderr, "PS-DTFE memory gate: partition %d waits (%.1f GB free, minus %.0f GB "
                                      "headroom, < the %.1f GB it adds; %d in flight)\n", partition, free/1.e9,
                                      headroom/1.e9, need/1.e9, inFlight );
                    waited = true;
                }
                cv.wait_for( lock, std::chrono::seconds(2) );
                free = autoTuneSystemFree();
            }
            if ( free >= 0. and free - headroom < need and blockers( partition ) == 0 and not saidAlone )
            {
                saidAlone = true;
                if ( verbose > 0 )
                    std::fprintf( stderr, "PS-DTFE memory gate: %.1f GB free, minus %.0f GB headroom, is below the "
                                  "%.1f GB a partition adds -- running one partition at a time from now on\n",
                                  free/1.e9, headroom/1.e9, need/1.e9 );
            }
        }
        ++inFlight;
        flying.insert( partition );
    }
    void leave(int partition)
    {
        {
            std::lock_guard<std::mutex> lock( m );
            --inFlight;
            auto const it = flying.find( partition );
            if ( it != flying.end() ) flying.erase( it );
        }
        cv.notify_all();
    }
};

#endif
