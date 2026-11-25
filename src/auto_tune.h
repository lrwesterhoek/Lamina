/* Auto-selection of --partition and --max-concurrent from the simulation (particle count, grid, fields) and the
   machine (RAM, cores); it fills only values the user did NOT give. The peak-memory model is calibrated on measured
   runs and deliberately errs HIGH. Budget (autoTuneBudget): 60% of RAM, capped at what other programs leave free less 4 GB. */

/* Environment overrides (tests, unusual machines): DTFE_RAM_GB physical RAM in GB (skips the live free-memory term);
   DTFE_MEM_FRACTION share of RAM for the budget (default 0.60; 0.1..0.95); DTFE_MEM_BUDGET_GB the budget itself, still
   capped by the free memory less the headroom unless DTFE_RAM_GB simulates a machine; DTFE_FREE_GB the free memory the
   other programs leave (the gate and the cap read it instead of the OS); DTFE_AUTO_MINN minimum particle count before
   auto-partitioning (default 2e6). */

/* The tuner's interface: machine queries (RAM, free memory, budget, cores), the report, the partition memory gate and
   the entry points. The model (bytes per cell and vertex, planners, constants) is in auto_tune.cc, so a model edit
   rebuilds that file only. */

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
#include <atomic>      // the memory gate's live partition count (dynamic concurrency)

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
#include "run_mode.h"      // the run mode's core budget (autoTuneCores) and threads (autoTuneGpuShare)


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

// The memory the PS-DTFE CPU deposit threads' private accumulator copies may take per partition: 20% of physical RAM,
// at least 2 GB. ps_interpolation.cc lowers its thread count to fit; the partition model budgets the same amount (the
// single build copies nothing, so it binds only the double build's Real sums).
inline double psDepositThreadBytes()
{
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

// This process's own memory now, as the budget counts it: the physical footprint on macOS (the ledger the SUMMARY
// and autoTuneSystemFree read), the resident set on Linux; < 0 when unknown. The dynamic memory gate's ledger.
inline double autoTuneOwnFootprint()
{
#ifdef __APPLE__
    task_vm_info_data_t ti;
    mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    return task_info( mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &c ) == KERN_SUCCESS ? double(ti.phys_footprint) : -1.;
#else
    double kb = -1., value;
    if ( FILE *f = std::fopen( "/proc/self/status", "r" ) )
    {
        char line[256];
        while ( std::fgets( line, sizeof(line), f ) )
            if ( std::sscanf( line, "VmRSS: %lf kB", &value ) == 1 ) { kb = value; break; }
        std::fclose( f );
    }
    return kb < 0. ? -1. : kb * 1024.;
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

// The live free-memory figure behind a budget it did not cut: '' on a simulated machine (DTFE_RAM_GB,
// the tests: their lines stay fixed), else what the other programs leave free and the headroom, or that the OS
// did not say. A budget the free memory cuts says so itself.
inline std::string autoTuneFreeNote(bool simulated, double freeForRun, double headroom)
{
    if ( simulated ) return "";
    if ( freeForRun < 0. ) return " (the free memory is unknown: the OS did not say)";
    char buf[160];
    snprintf( buf, sizeof(buf), " (the other programs leave %.3g GB free: %.3g GB after the %.3g GB headroom)",
              freeForRun/1.e9, (freeForRun - headroom)/1.e9, headroom/1.e9 );
    return buf;
}

// The memory budget the tuner plans against, and a one-line account of how it was set: the knob or the RAM share,
// and the live free memory the run started with.
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
            // ... but never more than the other programs leave free minus the headroom. Only a cut of more than 0.5 GB
            // counts (44 GB free less 4 against 40 asked is the same budget)
            double const freeForRun = simulated ? -1. : autoTuneFreeForRun();
            if ( freeForRun > 0. and freeForRun - headroom < v * 1.e9 - 0.5e9 )
            {
                double const capped = std::max( freeForRun - headroom, std::min( v * 1.e9, 1.e9 ) );
                snprintf( buf, sizeof(buf), "%.3g GB: DTFE_MEM_BUDGET_GB asks %.3g GB, but the other programs leave "
                          "%.3g GB free, minus %.3g GB headroom", capped/1.e9, v, freeForRun/1.e9, headroom/1.e9 );
                why = buf;
                return capped;
            }
            why += autoTuneFreeNote( simulated, freeForRun, headroom );
            return v * 1.e9;
        }
    }
    double frac = 0.60;
    bool fracSet = false;
    if ( double const v = Env::real( "DTFE_MEM_FRACTION", frac ); v >= 0.1 and v <= 0.95 )
        { fracSet = v != frac; frac = v; }
    double budget = frac * ramBytes;
    snprintf( buf, sizeof(buf), "%.3g GB = %.0f%% of %.3g GB RAM%s", budget/1.e9, 100.*frac, ramBytes/1.e9,
              fracSet ? " (DTFE_MEM_FRACTION)" : "" );
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
    else
        why += autoTuneFreeNote( simulated, freeForRun, headroom );
    return budget;
}

// the threads the tuner plans for: the run mode's core budget (run_mode.h), the OpenMP maximum unless it was lowered
inline int autoTuneCores()
{
#ifdef OPEN_MP
    return dtfeOmpMax();
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
    double fixedLoop = 0.;      // standard DTFE: 'fixed' while the partition loop runs (its grids only; 0: as fixed)
    double perPartition = 0.;   // bytes each partition in flight adds; under PS-DTFE's lane share the first adds laneOnce
                                // more (the run's memory gate waits for perPartition + laneOnce + gpuSnapshot)
    double lookAhead = 0.;      // bytes the partition look-ahead adds to 'predicted' (0: none runs)
    double gpuSnapshot = 0.;    // bytes of the GPU deposit's host snapshot (with --grid-window: and of the deposit's window
                                // grids), held by ONE partition at a time (in 'predicted')
    bool   mixedLanes = false;  // PS-DTFE: a mixed run's split ('--gpu-share' < 1): gpuSnapshot is also the GPU lock
                                // holder's device and host grids (booked once: onceBytes)
    double laneOnce = 0.;       // PS-DTFE lane share (auto_tune.cc laneShareCut): what the first partition in flight holds
                                // beyond each other one, in 'predicted' once (0: every lane booked whole)
    double onceBytes = 0.;      // PS-DTFE: what 'predicted' books once beside its lanes, whatever the count in flight:
                                // gpuSnapshot (one deposit holds the GPU at a time) and laneOnce (0 on the CPU). The
                                // dynamic concurrency's ledger books it once and each lane as 'predicted' does
                                // (DTFE.cpp psOnceBytes; PartitionMemoryGate::once)
    double predictedWholeLanes = 0.;   // ... 'predicted' with every lane whole: the staged merge's park room
    int    inFlightCap = 0;     // ... the most partitions in flight that booking holds for (0: no limit)
    double streamsEstimate = -1., streamsBudgeted = 0.;
    // PS-DTFE, for the partition loop's dynamic concurrency: whether the tuner chose the concurrency (a
    // user's --max-concurrent is kept as given), and what the run still allocates once its partitions are in flight --
    // after the loop (the derived grids, the finalized point outputs) and as the partitions merge (the point stream
    // records, budgeted: their unmerged share is still to come)
    bool   concChosen = false;
    double reserveEnd = 0., reserveFirstMerge = 0., reserveRecords = 0.;
    // PS-DTFE under --scratch-dir on macOS: the start-up peak of
    // psReserveGrids, which zero-fills each scratch-backed full grid in anonymous memory before its file is mapped
    // over it (ScratchAlloc::ZeroFillScope): the particles + the grids on the heap + the largest pre-sized grid.
    // 'predicted' is at least this (0: no scratch-backed grid, or Linux, where the scope is a no-op)
    double reservePeak = 0.;
    // PS-DTFE with the face walk beside the GPU deposit (ps_interpolation.cc PSEarlyFaceWalk): the face sums' words
    // one partition holds through its device call, booked beside every lane's hold (in 'predicted'; 0: no such walk)
    double faceWalk = 0.;
    std::string line() const
    {
        if ( not valid ) return "AUTO-TUNE-REPORT unsupported";
        // (the GPU snapshot and the look-ahead are in 'predicted' too, listed so a check can see each term, and so
        // are the face walk's words, only where they are booked, so every other line stays as it was; the parsers
        // read key=value pairs, and over_budget stays last)
        char walk[64] = "";
        if ( faceWalk > 0. ) snprintf( walk, sizeof(walk), "facewalk_gb=%.3f ", faceWalk/1.e9 );
        char b[480];
        snprintf( b, sizeof(b), "AUTO-TUNE-REPORT partition=%d mc=%d predicted_gb=%.2f budget_gb=%.2f "
                  "fixed_gb=%.2f pts_gb=%.3f streams_est=%.3f streams_budget=%.3f snapshot_gb=%.3f lookahead_gb=%.3f "
                  "%sover_budget=%d",
                  partition, maxConcurrent, predicted/1.e9, budget/1.e9, fixed/1.e9, pts/1.e9,
                  streamsEstimate, streamsBudgeted, gpuSnapshot/1.e9, lookAhead/1.e9, walk,
                  predicted > budget ? 1 : 0 );
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

/* '--gpu-share auto' (PS-DTFE; DTFE.cpp psResolveGpuShare, before the plan): the per mille of a partitioned '--gpu' run's
   exact deposits that run on the GPU, the rest on the free cores. In performance (single build, no DTFE_MASS_AUDIT,
   from DTFE_AUTO_MINN particles, >= 4 start-up threads N0): 50*lround(20/(1 + rho (N0 - 1.8))) clamped to [500, 1000]
   (rho 0.09, fitted on the 10-core M1 Max); else 1000. everyMode ('--gpu-share mixed') applies the model in gentle and
   balanced too, on the mode's own threads. 'why' says which rule decided, beginning with the run mode. */
long autoTuneGpuShare(User_options const &u, double nParticles, std::string &why, bool everyMode);

/* '--serve' on data too large for ONE tessellation: turn it into a composite server (ps_point_eval.cc) of n^3
   partitions, when '--tessellation-cache' lets them live on disk (without it a composite holds more than one
   tessellation, so the tuner only warns). Model (auto_tune.cc): kPSVertexBytes / kStdVertexBytes per vertex plus ~160 B
   per vertex for the cell index. Picks the smallest n (2..8) leaving room for two partitions beside the particles,
   then as many concurrent builds as fit. A user-given --partition is never touched. */
void autoTuneServe(User_options &u, size_t const nParticles);

#ifndef PHASE_SPACE
/* What the standard partition loop's LOOK-AHEAD holds for one partition (DTFE.cpp): the builds of its thread
   sub-domains (each padded again on its own sub-box) and, for a team, its padded particle copy. The UNIFORM count:
   'nOwn' particles in the partition, box lengths 'len', per-side padding 'pad', split 'tGrid' threads per axis. The
   plan (auto_tune.cc) counts the same; the loop checks the free memory against this before each one. */
double stdLookAheadBytes(User_options const &u, double const nOwn, double const len[NO_DIM],
                         double const pad[NO_DIM], size_t const tGrid[NO_DIM]);
#endif

/* The partition MEMORY GATE: a partition waits (2 s steps) while the free memory minus the headroom is below what one
   partition adds (AutoTuneReport::perPartition) AND another is in flight; with none in flight it always starts, so
   the gate delays, never deadlocks. need = 0 (a user-given --partition) disables it. ORDERED (the default): a finished
   partition may wait to merge in order while holding memory, so a partition waits only for LOWER-numbered ones in
   flight. A STAGED partition has left the gate; one whose sub-grids the free memory cannot hold is SPILLED (ps_merge_spill.h). */
struct PartitionMemoryGate
{
    double need = 0., headroom = kAutoTuneHeadroom;
    bool ordered = false;
    std::mutex m;
    std::condition_variable cv;
    int inFlight = 0, held = 0;
    /* DYNAMIC CONCURRENCY (DTFE.cpp starts more loop threads than the plan's mc; single build only). Up to 'floor'
       partitions pass by the rule above; one more only while the LEDGER holds it: the free memory at loop start less
       the headroom, the reserve (what the run still allocates: reserveEnd, the first merge's grids, the unmerged
       point records) and the run's growth beside its partitions holds need - once for every partition in flight plus
       this one, within the budget, and the live free memory still holds one 'need'. 'live': partitions in flight. */
    bool dynamic = false;
    int floor = 1, total = 1, finished = 0, maxInFlight = 0;
    double reserveEnd = 0., reserveFirstMerge = 0., reserveRecords = 0., budget = 0.;
    double startFree = -1., startOwn = -1.;
    /* 'once' is the part of 'need' that one partition at a time holds (AutoTuneReport::onceBytes): the GPU deposit's host
       snapshot (with the lock holder's device and host grids in a MIXED split) and, under the lane share, what the
       first partition holds beyond the others (laneOnce). The ledger books it once and need - once per partition in
       flight; the live check keeps the whole 'need'. 0: every partition books 'need' in full (a CPU run). */
    double once = 0.;
    double (*footprint)() = nullptr;
    std::atomic<int> *live = nullptr;
    double reserve() const
    {
        double const left = total > 0 ? double( std::max( 0, total - finished ) ) / double(total) : 0.;
        return reserveEnd + ( finished == 0 ? reserveFirstMerge : 0. ) + reserveRecords * left;
    }
    bool roomForOneMore(double const free) const
    {
        if ( not ( free >= 0. ) or free - headroom < need ) return false;       // the machine now (other programs)
        double const own = footprint != nullptr ? footprint() : -1.;
        if ( startFree < 0. or startOwn < 0. or own < 0. ) return false;
        double const lane = need - once;
        double const booked = double(inFlight) * lane + ( inFlight > 0 ? once : 0. );
        double const other = std::max( 0., ( own - startOwn ) - booked );     // the run's growth beside its partitions
        double const all = double( inFlight + 1 ) * lane + once;
        if ( startFree - headroom - reserve() - other < all ) return false;
        if ( budget > 0. and startOwn + other + all + reserve() > budget ) return false;
        return true;
    }
    std::multiset<int> waiting;  // the partitions waiting in enter(): admitted strictly in index order
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
            waiting.insert( partition );
            // the rule a partition waits by: behind a lower-numbered one waiting here (strict index order, so every
            // partition in flight is lower-numbered and blockers() counts them all); then the floor's (above), or
            // beyond the floor the dynamic ledger
            auto mustWait = [&]() {
                if ( *waiting.begin() < partition ) return true;
                if ( blockers( partition ) == 0 ) return false;
                if ( dynamic and inFlight >= floor ) return not roomForOneMore( free );
                return free >= 0. and free - headroom < need;
            };
            while ( mustWait() )
            {
                if ( not waited and *waiting.begin() == partition )
                {
                    ++held;
                    if ( verbose > 0 and not ( dynamic and inFlight >= floor ) )
                        std::fprintf( stderr, "PS-DTFE memory gate: partition %d waits (%.1f GB free, minus %.0f GB "
                                      "headroom, < the %.1f GB it adds; %d in flight)\n", partition, free/1.e9,
                                      headroom/1.e9, need/1.e9, inFlight );
                    waited = true;
                }
                cv.wait_for( lock, std::chrono::seconds(2) );
                free = autoTuneSystemFree();
            }
            waiting.erase( waiting.find( partition ) );
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
        maxInFlight = std::max( maxInFlight, inFlight );
        if ( dynamic and live != nullptr ) live->store( std::max( 1, inFlight ), std::memory_order_relaxed );
        flying.insert( partition );
        if ( not waiting.empty() )
        {
            lock.unlock();
            cv.notify_all();   // the next one in index order may check now
        }
    }
    void leave(int partition)
    {
        {
            std::lock_guard<std::mutex> lock( m );
            --inFlight;
            ++finished;
            if ( dynamic and live != nullptr ) live->store( std::max( 1, inFlight ), std::memory_order_relaxed );
            auto const it = flying.find( partition );
            if ( it != flying.end() ) flying.erase( it );
        }
        cv.notify_all();
    }
};

#endif
