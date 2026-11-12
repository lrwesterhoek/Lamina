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
                        below it a single domain always fits and partition seams are not worth it)
     DTFE_AUTO_SPAN     0 = PS-DTFE books a partition's sub-grid as G/n + 40 cells per axis (the model before
                        2026-11-07) instead of each partition's Eulerian span measured on the particles (default 1)
     DTFE_DYNAMIC_CONCURRENCY  0 = PS-DTFE's partition loop runs the plan's number of partitions at once; by default,
                        over a concurrency the tuner chose, more while the live memory holds them (PartitionMemoryGate) */

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
#include "run_mode.h"      // the run mode's core budget (autoTuneCores) and start-up threads (autoTuneGpuShare)


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
// build, DTFE_GPU_FLOAT_*).
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

// The live free-memory figure behind a budget it did not cut (plan P8.7): '' on a simulated machine (DTFE_RAM_GB,
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
    // PS-DTFE, for the partition loop's dynamic concurrency (plan P8.10): whether the tuner chose the concurrency (a
    // user's --max-concurrent is kept as given), and what the run still allocates once its partitions are in flight --
    // after the loop (the derived grids, the finalized point outputs) and as the partitions merge (the point stream
    // records, budgeted: their unmerged share is still to come)
    bool   concChosen = false;
    double reserveEnd = 0., reserveFirstMerge = 0., reserveRecords = 0.;
    // PS-DTFE under --scratch-dir on macOS (plan P8.2 as rescoped in phase R, 2026-11-09): the start-up peak of
    // psReserveGrids, which zero-fills each scratch-backed full grid in anonymous memory before its file is mapped
    // over it (ScratchAlloc::ZeroFillScope): the particles + the read blocks kept + the largest pre-sized grid.
    // 'predicted' is at least this (0: no scratch-backed grid, or Linux, where the scope is a no-op)
    double reservePeak = 0.;
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

/* '--gpu-share auto' (PS-DTFE; DTFE.cpp psResolveGpuShare, before the plan): the per mille of a partitioned '--gpu'
   run's exact deposits that run on the GPU, the others on the free cores at once (auto_tune.cc; the run modes'
   decision D11, docs/cleanup-plan.md). In performance, for the single build without the float sums, from
   DTFE_AUTO_MINN particles and at 4 or more start-up threads N0, it is 50*lround(20/(1 + rho (N0 - 1.8))) clamped to
   [500, 1000] (600 on the 10-core M1 Max; rho 0.09, DTFE_GPU_SHARE_RHO); else 1000. 'why' says which rule decided and
   its inputs, e.g. 'performance, 10 threads, rho 0.09' or 'balanced: every partition on the GPU'. */
long autoTuneGpuShare(User_options const &u, double nParticles, std::string &why);

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
    /* DYNAMIC CONCURRENCY (plan P8.10; DTFE.cpp starts more loop threads than the plan's mc, single build only). Up to
       'floor' (the plan's mc) partitions pass by the rule above. One more only while the LEDGER holds it: the free
       memory read when the loop started (startFree), less the headroom, the reserve (what the run has still to allocate:
       reserveEnd, the grids the first merge sizes until it has run, the unmerged share of the budgeted point records)
       and the run's own growth beside its partitions in flight (its footprint since the start less what the partitions
       in flight book, need - once each and the once part once: parked sub-grids, written records), holds that for
       every partition in flight plus this one -- so a burst of waiting threads gets in only as far as that room goes,
       whatever the live readings have not seen yet -- and the same total stays within the budget; and the live free
       memory still holds one 'need' (other programs). 'live' (when given) is the partitions in flight, for the deposit
       teams' core pool (thread_count.h). */
    bool dynamic = false;
    int floor = 1, total = 1, finished = 0, maxInFlight = 0;
    double reserveEnd = 0., reserveFirstMerge = 0., reserveRecords = 0., budget = 0.;
    double startFree = -1., startOwn = -1.;
    /* 'once' is the part of 'need' that one partition at a time holds (AutoTuneReport::onceBytes): the GPU deposit's
       host snapshot -- in a MIXED run's split ('--gpu-share' < 1) with the lock holder's device and host grids -- and,
       under the lane share, what the first partition holds beyond the others (laneOnce). So the ledger books it once
       and need - once for each partition in flight, as the plan's prediction does (a lane the dynamic concurrency
       adds included); the live check keeps the whole 'need' (the GPU may be free when this partition reaches it).
       0: every partition books 'need' in full (a CPU run). Like a lane not yet at its peak, the once part booked
       while no deposit holds the GPU can hide up to that much of the run's other growth; it nets out what the
       footprint keeps of the last holder's grids. */
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
