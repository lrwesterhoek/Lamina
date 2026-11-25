/* The auto-tuner's model (interface and machine queries: auto_tune.h): the memory model of a PS-DTFE and a standard-DTFE
   run, the planners choosing --partition / --max-concurrent, the '--serve' split and the automatic GPU share. Seen
   outside: autoTunePartitioning, autoTuneServe, stdLookAheadBytes, autoTuneGpuShare. The parts in auto_tune/ are included below. */

#include "auto_tune.h"

#include <cstdint>     // UINT32_MAX: the GPU's sub-grid cell cap
#include <cstdlib>
#include <cmath>
#include <limits>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <fstream>
#include <dirent.h>    // autoTuneServe: the cached partition splits of a snapshot
#include <map>
#include <functional>  // the standard-DTFE split replay (stdTuneSplit)
#include <sstream>
#include <optional>    // the plan phases that may decide the split (AutoTunePSPlanner::choose)
#include "cache_key_lines.h"
#include "ps_point_eval.h"   // psEstimateMeanStreams: the sample-point stream count, from the data
#include "message.h"
// for PSCausticClass::CausticMask, so the --ps-caustics per-cell accumulator is budgeted at its
// real width rather than a literal that would drift the next time a bit is added
#include "CGAL_triangulation/ps_caustic_class.h"

#ifndef PHASE_SPACE
// the run's own split rules, defined in subpartition.h (compiled into DTFE.cpp, the only caller of the tuner)
void optimalSplit(std::vector<size_t> &counts, size_t const noPartitions, size_t *partitionIndices);
void parallelGrid(size_t const noAvailableProcessors, User_options const &userOptions, size_t *grid);
#endif

#include "auto_tune/model.cc"   // the memory model: its constants, the bytes per vertex, cell and grid

void autoTuneServe(User_options &u, size_t const nParticles)
{
    if ( u.partitionOn or u.partitionGiven or u.partNo >= 0 )
        return;
    double const N = double(nParticles);
    std::string budgetWhy;
    double const budget = autoTuneBudget( autoTunePhysicalRAM(), budgetWhy );
    double const partBytes = double(sizeof(Particle_data));
    // transient of the build: the particle copy that coexists with the growing tessellation, or
    // (TBB parallel insertion, PS-DTFE) the sorted point+payload copy plus CGAL's own copy of it
    // while the particles are already freed -- triangulation.cpp
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    double const kBuild = 330.;
#else
    double const kBuild = 160.;
#endif
#ifdef PHASE_SPACE
    double const kDT = kPSVertexBytes + kBuild;
    double const singlePadFloor = 0.10;             // PS: Lagrangian padding floor (DTFE.cpp)
#else
    double const kDT = kStdVertexBytes + kBuild;
    double const singlePadFloor = 0.;
#endif
    double padFrac = 0.;
    for (int d=0; d<NO_DIM; ++d)
    {
        double const boxLen = double(u.boxCoordinates[2*d+1] - u.boxCoordinates[2*d]);
        if ( boxLen > 0. )
            padFrac = std::max( padFrac, double(u.paddingLength[2*d] + u.paddingLength[2*d+1]) / (2.*boxLen) );
    }
    double padAxes[NO_DIM];                         // the widest axis's padding, on every axis
    for (int d=0; d<NO_DIM; ++d) padAxes[d] = padFrac;
    auto tessBytes = [&](int n) -> double
    {
        double const f = autoTunePadFactor( u, padAxes, n, N, (n == 1) ? singlePadFloor : kPartitionPadFloor );
        return N / std::pow( double(n), NO_DIM ) * f * kDT;
    };
    double const single = N*partBytes + tessBytes(1);
    if ( single <= budget )
        return;

    MESSAGE::Message message( u.verboseLevel );
    char buf[640];
    if ( u.tessellationCacheDir.empty() )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE (serve): one tessellation of these " << nParticles << " particles needs ~"
                << single/1.e9 << " GB, over the memory budget of " << budget/1.e9 << " GB (" << budgetWhy
                << "). Pass '--tessellation-cache DIR' (a local directory) and the server becomes a composite of "
                << "partition tessellations of which only a few are held in memory at once; continuing with "
                << "one tessellation.\n" << MESSAGE::EndWarning;
        return;
    }
    // the coarsest split that leaves room for two partitions next to the particles while they are built
    int nMin = 2;
    while ( nMin < 8 and 2.*tessBytes(nMin) > budget - N*partBytes ) ++nMin;
    /* Which split. A split this snapshot already has in the cache is reused -- the finest complete one, else
       the one with the most cached partitions -- since any other means building (and storing) every
       partition again. With nothing cached, the split that holds ~8 partitions in memory at once, the number a
       zoom touches. */
    std::map<int,int> const cached = cachedServeSplits( u );
    int n = -1;
    char const *why = "";
    for (auto it = cached.rbegin(); it != cached.rend() and n < 0; ++it)
        if ( it->first >= nMin and it->second >= int( std::pow( double(it->first), NO_DIM ) ) )
        { n = it->first; why = "the finest split fully in the tessellation cache"; }
    if ( n < 0 )
    {
        int best = 0;
        for (auto const &kv : cached)
            if ( kv.first >= nMin and kv.second > best ) { best = kv.second; n = kv.first; why = "the split with the most cached partitions"; }
    }
    if ( n < 0 )
    {
        // once built the particles are gone, so the budget holds partitions only; stop where a finer split
        // no longer shrinks them (on a small set the padding shells dominate, and 8^3 would only cost)
        n = nMin;
        while ( n < 8 and 8.*tessBytes(n) > budget and tessBytes(n+1) < 0.75 * tessBytes(n) ) ++n;
        why = cached.empty() ? "nothing cached yet: about 8 partitions held in memory, as many as a zoom touches"
                             : "the cached splits are too coarse for the memory budget: about 8 partitions held in memory";
    }
    double const room = std::max( 0., budget - N*partBytes );
    int conc = int( std::floor( room / std::max( tessBytes(n), 1. ) ) );
    conc = std::max( 1, std::min( conc, autoTuneCores() ) );
    u.partition = std::vector<size_t>( NO_DIM, size_t(n) );
    u.partitionOn = true;
    if ( not u.maxConcurrentOn )
        u.maxConcurrent = conc;
    snprintf( buf, sizeof(buf),
              "AUTO-TUNE (serve): one tessellation would need ~%.1f GB, over the %.1f GB budget -> serving a "
              "composite of %d^3 partitions (%s; ~%.1f GB each, %d built at a time); the resident set is sized "
              "once they are built\n", single/1.e9, budget/1.e9, n, why, tessBytes(n)/1.e9,
              u.maxConcurrentOn ? u.maxConcurrent : conc );
    message << buf << MESSAGE::Flush;
}

#include "auto_tune/std_plan.cc"   // standard DTFE's planner (AutoTuneStdPlanner)

#ifdef PHASE_SPACE
namespace {

#include "auto_tune/ps_inputs.cc"    // the PS planner's inputs (AutoTunePSInputs)
#include "auto_tune/ps_planner.cc"   // the PS planner (AutoTunePSPlanner)

}  // namespace
#endif

/* '--gpu-share auto' (auto_tune.h): the per mille of a partitioned PS-DTFE GPU run's exact deposits that run on the GPU, the
   rest on the free cores (DTFE.cpp psResolveGpuShare). Only performance mixes the devices, unless everyMode ('mixed'); the
   double build, DTFE_MASS_AUDIT, a small set (below DTFE_AUTO_MINN) and too few cores keep every partition on the GPU. Else
   the GPU takes 1/(1 + rho c), rounded to 0.05, in [0.5, 1]: c = N - 1.8 threads beside the GPU lanes (N = the start-up
   threads N0 in performance, else the mode's own, at most N0), rho = 0.09 (CPU/GPU deposit rate, fitted on the M1 Max). */
long autoTuneGpuShare(User_options const &u, double const nParticles, std::string &why, bool const everyMode)
{
    char buf[160];
    std::string const mode = runModeName( u.runMode );
    bool const performance = u.runMode == RunMode::Performance;
    if ( not performance and not everyMode )
    {
        why = mode + ": every partition on the GPU";
        return 1000;
    }
    if ( sizeof(Real) != 4 )
    {
        why = mode + ", the double build: every partition on the GPU";
        return 1000;
    }
    if ( Env::flag( "DTFE_MASS_AUDIT" ) )
    {
        why = mode + ", DTFE_MASS_AUDIT: every partition on the GPU";
        return 1000;
    }
    // the tuner's small-set threshold, read as autoTunePartitioning reads it
    double minN = kAutoMinN;
    if ( double const v = Env::real( "DTFE_AUTO_MINN", 0. ); v > 0. )
        minN = v;
    if ( nParticles < minN )
    {
        snprintf( buf, sizeof(buf), "%s, %.3g particles (below %.3g): every partition on the GPU",
                  mode.c_str(), nParticles, minN );
        why = buf;
        return 1000;
    }
    // the run's threads: performance's start-up threads, else the mode's own (an OMP_NUM_THREADS kept counts), at
    // most the start-up threads -- in either case never a '--max-concurrent' raise, so it moves no share
    int const n0 = runModeState().ompDefault;
    int const threads = performance ? n0 : std::min( n0, u.coreBudgetOwn );
    double const c = double(threads) - 1.8;
    if ( c < 2. )
    {
        snprintf( buf, sizeof(buf), "%s, %d threads: every partition on the GPU", mode.c_str(), threads );
        why = buf;
        return 1000;
    }
    constexpr double rho = 0.09;
    double const x = 1. / ( 1. + rho * c );
    long const share = std::max( 500L, std::min( 1000L, 50L * std::lround( 20. * x ) ) );
    snprintf( buf, sizeof(buf), "%s, %d threads, rho %g", mode.c_str(), threads, rho );
    why = buf;
    return share;
}

AutoTuneReport autoTunePartitioning(User_options &u,
                                    size_t const nParticles,
                                    bool const userSampling,
                                    bool const metalActive,
                                    std::vector<Particle_data> const *particles)
{
    if ( u.psServe )
    {
        autoTuneServe( u, nParticles );   // the grid/partition model below does not apply to a server
        return AutoTuneReport();
    }
    // partitionGiven, not only partitionOn: '--partition 1 1 1' asks for one triangulation, which
    // leaves partitionOn false -- reading that as "not given" would split the data anyway
    bool const autoPart = not (u.partitionOn or u.partitionGiven);
    // ... and a user's split that is not one domain on every axis ('--partition 1 2 2': partition[0] is 1, but it
    // runs the partition path): the model's n == 1 is the single-domain path only without it (partitionBytes)
    bool const userPartitionOn = u.partitionOn;
    bool const autoConc = not u.maxConcurrentOn;
    AutoTuneReport report;
    // a plan given in full still gets its prediction: the run's memory gate, RAM parking room and look-ahead room
    // come from this report; the planners change only the knobs left to them
    if ( u.partNo>=0 or u.redshiftConeOn or userSampling or u.psServe )
        return report;  // a mode where partitioning is unavailable
                        // (--serve keeps its single triangulation alive for every request)

    double minN = kAutoMinN;
    if ( double const v = Env::real( "DTFE_AUTO_MINN", 0. ); v > 0. )
        minN = v;

    double const N = double(nParticles);
    double const ramBytes = autoTunePhysicalRAM();
    std::string budgetWhy;
    double const budget = autoTuneBudget( ramBytes, budgetWhy );   // see the header comment
    int const cores = autoTuneCores();

    // the cells actually allocated and written: with --grid-window its cells, not the (virtual) full grid
    size_t outGrid[NO_DIM];
    u.outputGridSize( outGrid );
    double gridTotal = 1.;
    for (int d=0; d<NO_DIM; ++d)
        gridTotal *= double(outGrid[d]);

    AutoTunePS ps;
#ifdef PHASE_SPACE
    ps.phaseSpace     = true;
    ps.exactDeposit   = u.psExactDeposit;   // resolved (readOptions: '--exact', '--sampled', the pin or the default)
    ps.caustics       = u.psCaustics;
    ps.haloRelease    = u.psHaloRelease > Real(0.);
#endif
    ps.faceGradient   = u.faceGradient;
    ps.meanFlowField  = &u.aField;
    // uField and aField each allocate their OWN full grids and coexist (two reserveMemory calls in DTFE.cpp), so
    // summing both is right. The deposit and post phases are disjoint peaks -> max, not sum.
    double const bCellDeposit = autoTuneBytesPerCell(u.uField, ps, 0) + autoTuneBytesPerCell(u.aField, ps, 0);
    double const bCellPost    = autoTuneBytesPerCell(u.uField, ps, 1) + autoTuneBytesPerCell(u.aField, ps, 1);
    double const bCell        = std::max( bCellDeposit, bCellPost );

    // padding fraction of the box per axis (paddingLength was resolved to Mpc by updateEntries)
    double padFrac[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        double const boxLen = double(u.boxCoordinates[2*d+1] - u.boxCoordinates[2*d]);
        padFrac[d] = boxLen>0. ? double(u.paddingLength[2*d] + u.paddingLength[2*d+1]) / (2.*boxLen) : 0.;
    }

    MESSAGE::Message message( u.verboseLevel );

#ifdef PHASE_SPACE
    // the largest pre-sized full grid, for the --scratch-dir start-up peak (AutoTunePSPlanner::reserveZeroFillPeak)
    double const largestGridBytes = std::max( autoTuneLargestGridBytesPerCell(u.uField, ps),
                                              autoTuneLargestGridBytesPerCell(u.aField, ps) ) * gridTotal;
    // ---- PS-DTFE: Lagrangian partitions run in PARALLEL; concurrency is the throughput knob
    // and each concurrent partition holds a full padded triangulation (the dominant cost). ----
    // per padded vertex: the tessellation (kPSVertexBytes) and, beside it while it grows, the particle copy
    double const partBytes = double(sizeof(Particle_data));
    // '--parallel-triangulation' (TBB; PS-DTFE's default): the sorted point+payload copy and CGAL's own copy outlive the
    // particle copy during the insert (kParInsertBytes per vertex in flight; autoTuneServe's kBuild 330 holds it there).
    // It runs in parallel only where cores / concurrency leave >= 2 threads (triangulation.cpp), so it is booked per count
    // in flight (parInsertBytes: the split search, laneSharePeak, finishReport), not per vertex.
    double const kParInsert = 0.;
    // --scratch-dir (already armed by DTFE_setup when set): the full-grid accumulators live in
    // mmap'ed files, so they leave the RAM model -- the kernel pages them against the disk and
    // RSS stays bounded. The RAM check then covers only the particles + per-partition terms,
    // and the grid term is checked against the scratch volume's FREE SPACE instead.
    bool const scratchOn = not u.scratchDir.empty();

    // ---- '--sample-points' point evaluation (ps_point_eval.cc) ---------------------------
    // Also IRREDUCIBLE: the query points and their per-point stream records are GLOBAL
    // structures shared by every partition (each partition appends its hits), so --partition
    // does not shrink them. Without this a points-only run (tiny --grid, so the grid term
    // vanishes) could pick a coarse split against a budget that ignores tens of GB.
    size_t nPts = autoTuneSamplePointCount( u.psSamplePointsFile );
    if ( double const v = Env::real( "DTFE_AUTO_PTS_N", -1. ); v >= 0. )      // a plane that does not exist yet
        nPts = size_t(v);
    bool const ptsVelGrad = u.psPtsVelGrad or Env::flag( "DTFE_AUTO_PTS_VEL_GRAD" );
    // Mean streams (containing tetrahedra) per point. No fixed number fits: it grows with the mass resolution. So it is
    // ESTIMATED from the particles themselves (psEstimateMeanStreams: Eulerian over Lagrangian volume of the
    // tessellation, on random Lagrangian patches) and budgeted above the estimate (autoTunePSPointStreams): the
    // estimate is the box mean, while one plane varies around it.
    double ptsMeanStreams = 2.0;
    char ptsStreamsWhy[200] = "2 streams/point assumed";
    autoTunePSPointStreams( u, N, nPts, particles, report, ptsMeanStreams, ptsStreamsWhy );
    double ptsRecordBytes = 0., ptsOutputBytes = 0.;
    double const ptsBytes = autoTunePSPointBytes( u, nPts, ptsVelGrad, ptsMeanStreams, &ptsRecordBytes, &ptsOutputBytes );

    // IRREDUCIBLE in RAM unless scratch-backed: originals + the full-grid accumulators
    // + the point-evaluation structures.
    // --partition does NOT reduce this -- partitions sum into the SHARED full grid.
    // a --grid-window run that started without its particles (ps_globals_record.h) holds none; a late read adds them back
    // (at most the particles' bytes over budget)
    double const particleBytes = u.psParticlesDeferred ? 0. : partBytes*N;
    // (what the input reader frees goes back to the system: main.cpp dtfeAllocatorReexec)
    double const fixed = particleBytes + (scratchOn ? 0. : gridTotal*bCell) + ptsBytes;
    report.fixed = fixed;
    report.budget = budget;
    report.pts = ptsBytes;
    report.streamsBudgeted = nPts > 0 ? ptsMeanStreams : 0.;
    // NOTE: the over-budget check (auto_tune/ps_planner.cc irreducibleCase) must run BEFORE the small-N early return --
    // the grid term does not depend on N, so a small snapshot on a huge --grid is exactly the case that must not slip
    // through to a silent 60+ GB allocation.

    // the regular part's twins and each face pass's stream-average divergence: made
    // beside the deposit phase's grids in the normalization (whose mass flux grid bCellDeposit books already:
    // autoTuneFaceOutputBytes), so the scratch volume holds bCellDeposit + normR there (the larger of that and bCell)
    double const normR = autoTuneStreamsTwinBytes(u.uField, ps) + autoTuneStreamsTwinBytes(u.aField, ps)
                       + autoTuneFaceOutputBytes(u.uField, ps, false) + autoTuneFaceOutputBytes(u.aField, ps, false);
    autoTuneScratchSpaceCheck( u, scratchOn, gridTotal, std::max( bCell, bCellDeposit + normR ) );

    // ---- Is the IRREDUCIBLE term alone over budget? ---- If so ('fixed > budget', ps_planner.cc irreducibleCase), no
    // --partition value can help: partitions sum into the shared full grid, and the search would end at its first
    // candidate (the COARSEST split), blaming the split.

    // The per-tet/per-cell GPU-deposit pieces this field selection needs (unselected moment grids are not allocated; tet
    // velocities are extracted only for a velocity-derived grid). The velocity derivatives fold into the gradient after
    // auto-tune runs, so they count here as gradient requests.

    // the dispersion implies no velocity/mom grid (it carries its own mass-weighted mean beside the volume-weighted
    // velocity moments); the '_a' pass of a '--face-gradient' run deposits the mean velocity, selected or not
    // (autoTuneFaceVelocity)
    bool const psVel  = u.uField.velocity or u.aField.velocity
                        or autoTuneFaceVelocity( u.uField, ps ) or autoTuneFaceVelocity( u.aField, ps );
    bool const psDisp = u.uField.velocity_dispersion or u.aField.velocity_dispersion;
    bool const psGrad = u.uField.velocity_gradient or u.uField.selectedVelocityDerivatives()
                        or u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives();
    bool const psScal  = u.uField.scalar or u.aField.scalar;                       // (gpu_host.h out.scal, out.sgrad)
    bool const psSGrad = u.uField.scalar_gradient or u.aField.scalar_gradient;
    // flags gating the remaining GPU buffers (ps_metal_host.cc)
    bool const psCaust = ps.caustics;
    bool const psExact = ps.exactDeposit;

    // Every buffer in PSGpuGrids, per cell and copy (the GPU deposit's host and device grids): mass 4 + streams 4 = the 8;
    // the momw, caustic and streamvol terms count too -- 20 B/cell/copy = 40 after the x2 in partitionBytes(). The moment
    // grids are 64-bit fixed-point sums (ps_metal_host.cc, fxUnits): 8 B per component on the device and in the host's
    // copy (the 2 x); dispw is not deposited (the mass grid stands for it).
    double const gridB  = 8. + 2. * ( (psVel ? 12. : 0.) + (psDisp ? 24. : 0.) + (psGrad ? 36. : 0.)
                                    + 4.                                  // momw
                                    + (psExact ? 4. : 0.)                 // streamvol
                                    + (psDisp ? 12. : 0.)                 // dispvel
                                    + (psScal ? 4.*noScalarComp : 0.)     // scal
                                    + (psSGrad ? 4.*noScalarGradComp : 0.) )   // sgrad
                        + (psCaust ? 4. : 0.);                            // caustic

    // The single build's fixed-point MASS grid (ps_interpolation.cc 'massFx'): a pass with density selected accumulates 8 B
    // words per sub-grid cell in place of the 4 B float density (atomics, no private copies) and converts once after the
    // loop, when both coexist. The double build keeps its double sums. fxCell = the extra bytes per sub-grid cell while
    // the words live (the passes run one after the other: one grid).
    bool const psDensity = u.uField.density or u.uField.velocity_tweb or u.aField.density or u.aField.velocity_tweb;
    double const fxCell  = ( ps.phaseSpace and sizeof(Real) == 4 and psDensity ) ? 4. : 0.;
    // ... and the moment grids the same way: fxMom = the extra bytes per sub-grid cell while one pass's
    // words live (as many again as its Real moment grids: the larger pass), which also leave the private copies
    double const fxMom   = ( ps.phaseSpace and sizeof(Real) == 4 )
                         ? std::max( autoTuneFxMomentBytes(u.uField, ps), autoTuneFxMomentBytes(u.aField, ps) ) : 0.;
    // ... and the face sums ('--face-gradient'), int64 words in EVERY build: faceR = their Real grids per sub-grid
    // cell (the larger pass; in bCellDeposit), fxFace = what the words take beyond those (as many again in the
    // single build, nothing in the double build, whose grids are as wide as the words)
    double const faceR   = std::max( autoTuneFaceBytes(u.uField, ps), autoTuneFaceBytes(u.aField, ps) );
    double const fxFace  = faceR * ( 8./double(sizeof(Real)) - 1. );
    // ... and where the face walk runs beside the GPU deposit (ps_interpolation.cc PSEarlyFaceWalk), its words (psFaceUnits:
    // faceS 2 x NO_DIM x 3 int64 per sub-grid cell, faceN and faceP 2 x NO_DIM each; 240 B in 3D) exist from BEFORE the
    // pass's device call until their conversion or a failed call. Only under earlyWalk's conditions (single build, GPU
    // deposit, face pass, no NaN test hook); 0 otherwise.
    double const earlyFace = ( NO_DIM == 3 and metalActive and faceR > 0. and sizeof(Real) == 4
                               and Env::integer( "DTFE_TEST_NAN_SHARE", 0 ) <= 0 ) ? faceR + fxFace : 0.;
    // One private copy of the CPU deposit's copy mode, per sub-grid cell: the Real accumulators of ONE pass (the passes
    // run one after the other, each copying only its grids that are not fixed-point sums: ps_interpolation.cc accBytes).
    // Subtracting the fixed-point and shared terms from both passes would be wrong both ways with u and a selected, and
    // predict SLAB mode where the run uses the copy mode.
    double copyCell = autoTunePSCopyCell( u, ps );

    // ... and each deposit thread's per-tetrahedron sample lists (DepositThread insideFlat/insideRel, 16 B per sample of
    // the largest tetrahedron it meets), visible on a very fine grid over few particles: up to ~50x the samples of a mean
    // owned tetrahedron (grid x nSub^3 / 6 N) on crossed waves. Booked at 128x that mean; negligible at production densities.
    int const nSubScratch = std::max( 1, psExact ? 3 : int(u.psAvgSubsamples) );
    double const sampleCube = ( u.aField.selected() or psExact ) ? std::pow( double(nSubScratch), double(NO_DIM) ) : 1.;
    double const threadScratch = double(cores) * 16. * 128. * gridTotal * sampleCube / ( 6. * std::max( 1., N ) );

    AutoTunePSInputs in;
    in.metalActive     = metalActive;
    in.mixed           = metalActive and u.psGpuShareMille < 1000;   // (DTFE.cpp psResolveGpuShare, before the plan)
    in.autoPart        = autoPart;
    in.userPartitionOn = userPartitionOn;
    in.autoConc        = autoConc;
    in.minN            = minN;
    in.N               = N;
    in.ramBytes        = ramBytes;
    in.budgetWhy       = budgetWhy;
    in.budget          = budget;
    in.cores           = cores;
    for (int d=0; d<NO_DIM; ++d)
    {
        in.outGrid[d] = outGrid[d];
        in.padFrac[d] = padFrac[d];
    }
    in.gridTotal       = gridTotal;
    if ( not autoPart and not u.partition.empty() )
    {
        for (int d=0; d<NO_DIM; ++d) in.userK[d] = int( u.partition[ size_t(d) < u.partition.size() ? size_t(d) : 0 ] );
        for (int d=1; d<NO_DIM; ++d) in.userNonCubic = in.userNonCubic or in.userK[d] != in.userK[0];
    }
    if ( particles != nullptr and not u.psWindowOn )
    {
        auto const t0 = std::chrono::steady_clock::now();
        psPartitionSpans( *particles, u, padFrac, in.userNonCubic ? in.userK : nullptr, in.spanCells, in.spanUser,
                          in.spanMean, in.spanMeanUser );
        if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )
        {
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS sub-grid spans (%.3f s), largest partition's cells per n^3 split "
                          "(measured / G/n+40):", std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
            for (int n=1; n<=6; ++n)
            {
                double c = 1.;
                for (int d=0; d<NO_DIM; ++d) c *= std::min( double(outGrid[d]), double(u.gridSize[d])/double(n) + 40. );
                std::fprintf( stderr, " %d: %.4g / %.4g", n, in.spanCells[n], c );
            }
            if ( in.userNonCubic ) std::fprintf( stderr, " user split: %.4g", in.spanUser );
            std::fprintf( stderr, "\n" );
            // the lane share's floor (laneSpread)
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS span spread (mean over largest) per n^3 split:" );
            for (int n=2; n<=6; ++n)
                std::fprintf( stderr, " %d: %.3f", n, in.spanCells[n] > 0. ? in.spanMean[n] / in.spanCells[n] : 1. );
            if ( in.userNonCubic and in.spanUser > 0. )
                std::fprintf( stderr, " user split: %.3f", in.spanMeanUser / in.spanUser );
            std::fprintf( stderr, "\n" );
        }
    }
    in.ps              = ps;
    in.bCellDeposit    = bCellDeposit;
    in.bCell           = bCell;
    in.partBytes       = partBytes;
    in.kParInsert      = kParInsert;
    in.scratchOn       = scratchOn;
    in.largestGridBytes = largestGridBytes;
    in.scratchMinBytes = u.scratchThresholdBytes();   // the route's own (user_options.cc)
    in.nPts            = nPts;
    in.ptsStreamsWhy   = ptsStreamsWhy;
    in.ptsBytes        = ptsBytes;
    in.ptsRecordBytes  = ptsRecordBytes;
    in.ptsOutputBytes  = ptsOutputBytes;
    in.particleBytes   = particleBytes;
    in.fixed           = fixed;
    in.psVel           = psVel;
    in.psDisp          = psDisp;
    in.psGrad          = psGrad;
    in.psExact         = psExact;
    in.gridB           = gridB;
    in.psDensity       = psDensity;
    in.fxCell          = fxCell;
    in.fxMom           = fxMom;
    in.faceR           = faceR;
    in.fxFace          = fxFace;
    in.earlyFace       = earlyFace;
    in.normR           = normR;
    in.copyCell        = copyCell;
    in.threadScratch   = threadScratch;
    AutoTunePSPlanner const planner{ in, u, report, message };
    return planner.choose();

#else
    // ---- standard DTFE: partitions are processed SERIALLY (memory knob); inside one partition
    // the box is split across up to --max-concurrent threads (throughput knob). ----
    // A '--region' run is planned as what it triangulates: DTFE_setup crops first and passes options whose box IS the
    // region, N = the particles inside it, so 'padFrac' is the padding over the region's length. The padded copies the
    // crop holds are left out of 'fixed': the triangulation term counts them (36 B each next to ~700 B per vertex).
    double const partBytes = double(sizeof(Particle_data));

    /* '--sample-points' (the PS model's term, for the standard point evaluation): the query points and their records are
       GLOBAL (every partition appends its answers until psPointEvalFinalize), so a split does not shrink them: 'fixed'. Standard
       DTFE answers each point ONCE. The STORED record is packed (RecLayout): both densities + the velocity in doubles + the 2-B
       caustic mask, + the optional gradients/scalars (--pts-den-grad / --pts-vel-grad / --pts-scalar), + a point index and a
       locator; plus per point its position, bucket entry, offsets and the finalized outputs. KEEP IN SYNC with ps_point_eval.cc. */
    size_t nPts = autoTuneSamplePointCount( u.psSamplePointsFile );
    if ( double const v = Env::real( "DTFE_AUTO_PTS_N", -1. ); v >= 0. )
        nPts = size_t( v );
    bool const ptsVelGrad = u.psPtsVelGrad or Env::flag( "DTFE_AUTO_PTS_VEL_GRAD" );
    double bPtsVar = ( u.psPtsDenGrad ? 3. * double(sizeof(double)) : 0. ) + ( ptsVelGrad ? 9. * double(sizeof(double)) : 0. )
                   + ( u.psPtsScalar ? double(noScalarComp) * double(sizeof(double)) : 0. );
    double const bStreamRec = 5. * double(sizeof(double)) + 2. + bPtsVar + 4. + 8.;
    double const bPerPoint  = 3. * double(sizeof(double)) + 4. + 2. * 8.;
    double const bPtsOut    = double(sizeof(double)) + 3. * double(sizeof(double)) + 6. * double(sizeof(double)) + 4. + bPtsVar;
    double const ptsBytes   = double(nPts) * ( bPerPoint + bPtsOut + bStreamRec );
    // ... and each point-walk thread fills one 16 MB block of records before it hands it over (HitSink), per partition
    double const ptsWalkBytes = nPts > 0 ? double(dtfeHardwareThreads()) * kHitSinkBlockBytes : 0.;

    /* --scratch-dir: an allocation of at least DTFE_SCRATCH_MIN_GB (1 GB) is an mmap'ed file (scratch_alloc.cc), so full
       grids leave the RAM model one by one: a smaller grid stays on the heap (heapGrids), the largest file-backed one
       sets the start-up peak (scratchZeroFill), and after the loop the derived grids under it join them (heapPost). A
       partition's own blocks stay booked in full, files or not: they are its working memory, in the page cache (fitted on
       a 256^3 Zel'dovich box at 512^3, 2^3 partitions, and in one domain on a team of 4). The process itself is left out. */
    double const scratchMinBytes = u.scratchThresholdBytes();   // (0: no --scratch-dir)
    bool const scratchOn = scratchMinBytes > 0.;
    double heapGrids = 0., heapPost = 0., largestScratchGrid = 0.;
    if ( scratchOn )
    {
        for (Field *f : { &u.uField, &u.aField })
            for (double const perCell : autoTuneStdGridsPerCell( *f ))
            {
                double const bytes = perCell * gridTotal;
                if ( bytes < scratchMinBytes ) heapGrids += bytes;
                else largestScratchGrid = std::max( largestScratchGrid, bytes );
            }
        heapPost = heapGrids;
        for (Field *f : { &u.uField, &u.aField })
            for (double const perCell : autoTuneStdDerivedPerCell( *f ))
                if ( perCell * gridTotal < scratchMinBytes ) heapPost += perCell * gridTotal;
    }
    double const fixed = partBytes*N + (scratchOn ? heapPost : gridTotal*bCell) + ptsBytes;
    // ... and while the partition loop runs: its own grids only, the derived ones come after it (stdPeak)
    double const fixedLoop = partBytes*N + (scratchOn ? heapGrids : gridTotal*bCellDeposit) + ptsBytes;
    /* The GPU deposit of the '_a' pass (averaged_interpolation_1.cc -> dtfe_metal_host.cc), sampled and exact average alike.
       Per tetrahedron it extracts (std_plan.cc stdTetsOfBox) the flat arrays -- vertices 48 B, densities 16, vertex velocities 48
       where a velocity grid is deposited, the scalar 16, volume, sample count and flat index 4 each, the base cell 12 -- and the
       Morton order holds (key, index) pairs (8) and one gathered array copy (<= 48). Per sub-grid cell: the host grid (bGpuCell)
       and the device grids in words of 2 or 4 floats (deviceWordFactor: sampled, exact). The face pass supplies the gradient. */
    bool gpuVel = false, gpuScal = false;
    double bGpuCell = 0.;
    bool const gpuDeposit = metalActive and u.aField.selected();
    if ( gpuDeposit )
    {
        bool const den  = u.aField.density or u.aField.velocity_tweb;
        bool const grad = u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives();
        bool const faceSupplies = NO_DIM == 3 and grad and u.faceGradient and not Env::flag( "DTFE_FACE_CHECK" );
        bool const gradDeposited = grad and not faceSupplies;
        gpuVel  = u.aField.velocity or gradDeposited;
        gpuScal = u.aField.scalar or u.aField.scalar_gradient;
        bGpuCell = ( den ? 4. : 0. ) + ( u.aField.velocity ? 12. : 0. ) + ( gradDeposited ? 36. : 0. )
                 + ( u.aField.scalar ? 4. : 0. ) + ( u.aField.scalar_gradient ? 12. : 0. );
    }
    double const perTet = gpuDeposit ? 48. + 16. + ( gpuVel ? 48. : 0. ) + ( gpuScal ? 16. : 0. ) + 4. + 4. + 4. + 12.
                                       + 8. + 48. : 0.;
    /* The GPU deposit's host SNAPSHOT (PS's snapshotBytes, ps_planner.cc): the standard deposit resumes a failed command buffer
       from a host copy of the batch's device grids (dtfe_metal_host.cc runBatch, GpuSnapshot), in 8-B fixed-point words per
       4-B component, of ONE thread sub-domain at a time: one copy of the largest sub-grid, added once to the busiest
       partition's bytes (StdPlan::snapshot). An exact run's batches sum in 128-bit words (dtfe_gpu_fixed.h): 4 x bGpuCell per
       cell; planOf books two batches of the largest sub-grid for the device grids (deviceWordFactor). */
    bool const runWideWords = u.exactAverage;
    double const snapWordFactor = runWideWords ? 4. : 2.;
    double const deviceWordFactor = snapWordFactor;
    // the look-ahead (DTFE.cpp) runs with the GPU deposit when more than one partition is computed
    bool const lookAheadPossible = gpuDeposit and u.DTFE and u.method == 1
                                   and not u.stdNoOverlap and not u.redshiftConeOn;

    // the count grid of the particles (StdTuneDensity: one pass); without particles in memory, the uniform count
    auto const tuneStart = std::chrono::steady_clock::now();
    StdTuneDensity dens;
    // (DTFE_TEST_TUNE_UNIFORM=1, tests: the uniform count, as for a call without the particles, which the binaries
    // never make: tests/dtfe_auto_tune_check.sh M)
    if ( particles != nullptr and not particles->empty() and not Env::flag( "DTFE_TEST_TUNE_UNIFORM" ) )
        dens.build( *particles, u );
    double const countSec = std::chrono::duration<double>( std::chrono::steady_clock::now() - tuneStart ).count();
    // '--sample-points' evaluates ONE tessellation per partition (DTFE_parallel runs no thread split for it)
    bool const oneTessellation = not u.psSamplePointsFile.empty();
    double origin[NO_DIM], dxGrid[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        origin[d] = double( u.boxCoordinates[2*d] );
        dxGrid[d] = ( double( u.boxCoordinates[2*d+1] ) - origin[d] ) / double( u.gridSize[d] );
    }
    std::vector<size_t> const userSplit = ( not autoPart and u.partition.size() == size_t(NO_DIM) )
                                        ? u.partition : std::vector<size_t>( NO_DIM, 1 );

    AutoTuneStdInputs in;
    in.metalActive       = metalActive;
    in.autoPart          = autoPart;
    in.autoConc          = autoConc;
    in.N                 = N;
    in.ramBytes          = ramBytes;
    in.budgetWhy         = budgetWhy;
    in.budget            = budget;
    in.cores             = cores;
    in.gridTotal         = gridTotal;
    in.bCellDeposit      = bCellDeposit;
    for (int d=0; d<NO_DIM; ++d)
    {
        in.padFrac[d] = padFrac[d];
        in.origin[d]  = origin[d];
        in.dxGrid[d]  = dxGrid[d];
    }
    in.partBytes         = partBytes;
    in.nPts              = nPts;
    in.ptsBytes          = ptsBytes;
    in.ptsWalkBytes      = ptsWalkBytes;
    in.fixed             = fixed;
    in.fixedLoop         = fixedLoop;
    in.gpuDeposit        = gpuDeposit;
    in.bGpuCell          = bGpuCell;
    // the face pass (averaged_interpolation_1.cc avg1FacePass): its accumulator over the cells of the sub-domain the
    // '_a' interpolation runs on, one per concurrent sub-domain call, which together cover the partition
    in.faceCell = ( NO_DIM == 3 and u.faceGradient and u.aField.selected()
                    and ( u.aField.velocity_gradient or u.aField.velocity_divergence or u.aField.velocity_shear
                          or u.aField.velocity_vorticity or u.aField.velocity_vweb ) ) ? 9. * double(sizeof(double)) : 0.;
    /* The CPU exact average's run-wide words (averaged_interpolation_1.cc Avg1Words): 8 B per component of each grid its loop
       deposits, over the cells of the sub-domain, beside the Real grids until the loop ends; one set per concurrent sub-domain
       call (together covering the partition, as faceCell). Only for a grid with a run-wide unit (RunUnits::has: not the scalar
       and its gradient under '--approx-phase-space-density'), and only where the CPU deposits the '_a' grids (a GPU fall-back is
       booked by planOf at 2 x bGpuCell); the gradient only where the face pass does not supply it. 0 in double and MY_SCALAR builds. */
#ifndef MY_SCALAR
    bool const cpuWords = not metalActive and u.DTFE and u.method == 1 and u.exactAverage and sizeof(Real) == 4
                          and u.aField.selected();
    // (the gradient's words under '--per-tet-gradient', for the warning's way out below: the loop's own test)
    bool const cpuGradWords = cpuWords and u.runUnits.has( RU_GRADIENT );
    if ( cpuWords )
    {
        RunUnits const &ru = u.runUnits;
        bool const den = ( u.aField.density or u.aField.velocity_tweb ) and ru.has( RU_DENSITY );
        bool const vel = u.aField.velocity and ru.has( RU_VELOCITY );
        bool const grad = ( u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives() )
                          and not ( NO_DIM == 3 and u.faceGradient and not Env::flag( "DTFE_FACE_CHECK" ) )
                          and ru.has( RU_GRADIENT );
        bool const scal = u.aField.scalar and ru.has( RU_SCALAR );
        bool const sgrad = u.aField.scalar_gradient and ru.has( RU_SCALAR_GRADIENT );
        in.bCpuWordCell = 8. * ( ( den ? 1. : 0. ) + ( vel ? double(noVelComp) : 0. )
                                 + ( grad ? double(noGradComp) : 0. ) + ( scal ? double(noScalarComp) : 0. )
                                 + ( sgrad ? double(noScalarGradComp) : 0. ) );
    }
#else
    bool const cpuGradWords = false;
#endif
    // ... and what '--per-tet-gradient' would book less per sub-domain cell, for the 'no split fits' warning's way out
    // (text only): the face sums, less the gradient words that route deposits in their place (the face default without
    // DTFE_FACE_CHECK=1 skips them). None on the GPU deposit: without the face pass it deposits the gradient (host and
    // device grids, snapshot) and extracts the velocities for it, which books more
    in.faceWayOutCell = gpuDeposit ? 0. : in.faceCell - ( ( in.faceCell > 0. and cpuGradWords
                                                            and not Env::flag( "DTFE_FACE_CHECK" ) )
                                                          ? 8. * double(noGradComp) : 0. );
    in.perTet            = perTet;
    in.snapWordFactor    = snapWordFactor;
    in.deviceWordFactor  = deviceWordFactor;
    in.lookAheadPossible = lookAheadPossible;
    /* The start-up peak under --scratch-dir (macOS): Quantities::reserveMemory sizes each full grid at the loop's start
       and ScratchAlloc::zeroResize zero-fills a scratch-backed one in anonymous memory before its file is mapped over it
       (ZeroFillScope), one grid at a time: the particles, the grids left on the heap and the largest file-backed grid
       (the gradient's 36 B a cell where a velocity derivative is asked) are held at once, before any partition. Checked
       on TNG100-3-Dark z=0 at 1024^3 and a 256^3 Zel'dovich box at 512^3. Linux: none. PS-DTFE books the same. */
#ifdef __APPLE__
    if ( largestScratchGrid > 0. )   // (beside the heap grids, which are sized as well)
        in.scratchZeroFill = partBytes * N + heapGrids + largestScratchGrid;
#endif
    in.tuneStart         = tuneStart;
    in.countSec          = countSec;
    in.oneTessellation   = oneTessellation;
    in.userSplit         = userSplit;
    User_options pgOptions = u;                     // parallelGrid reads the partition's own grid from it
    pgOptions.verboseLevel = 0;                     // (and would say so for every plan)
    AutoTuneStdPlanner const planner{ in, u, report, message, dens, {}, std::move( pgOptions ) };
    return planner.chooseSplit();
#endif
}
