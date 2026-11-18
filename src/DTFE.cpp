/*
 *  Copyright (c) 2011       Marius Cautun
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* DTFE driver: data setup, padding, partitioning (standard and PS-DTFE), and velocity-derivative/web post-processing.
   The derived velocity fields and the V-web / T-web classification themselves are in web_fields.cc.
   The unit is split: the parts in the folder dtfe/ are included below in order and compiled only as part of this
   one object. */

#include <vector>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>   // partition-loop progress + ETA
#include <thread>      // the partition look-ahead
#include <exception>
#include <memory>
#include <map>       // the staged ordered merge's parked sub-grids
#include <functional>   // the standard look-ahead's memory estimate
#ifdef OPEN_MP
    #include <omp.h>
#endif
#include <boost/math/special_functions/fpclassify.hpp>


#include "define.h"
#include "particle_data.h"
#include "user_options.h"
#include "quantities.h"
#include "subpartition.h"
#include "miscellaneous.h"
#include "phase_totals.h"   // the run's per-stage totals (dtfeFinishFields prints them)
#include <limits>

#include "message.h"
#include "env_knobs.h"
#include "run_mode.h"   // the run mode's core budget (dtfeOmpMax) and the look-ahead thread's (dtfeAdoptCoreBudget)
#include "thread_count.h"   // CorePool::Hold: the look-ahead thread's core counted in the core pool
#include "auto_tune.h"
#include "scratch_alloc.h"
#include "ps_point_eval.h"
#include "ps_globals_record.h"
#include "fixed_units.h"   // the run-wide fixed-point units (runUnitsSetup)
#include "run_progress.h"   // the '<output>.runlog.progress' sidecar
#include "partition_checkpoint.h"   // checkpoint + '--resume' of a partitioned run (both binaries)
#include "web_fields.h"   // postProcessWebFields: the derived velocity fields and the V-web / T-web (web_fields.cc)
#ifdef PHASE_SPACE
#include "CGAL_triangulation/ps_mass_audit.h"   // DTFE_MASS_AUDIT=1: the run's mass closure after the merge
#include "ps_merge_spill.h"   // the staged ordered merge's spill of parked sub-grids to disk
#endif
#include "io/io.h"         // readInputData: the late read of a --grid-window run that started without its particles
#include "CGAL_triangulation/ps_deposit_params.h"   // fixedShareDrops: the non-finite guard's counts


using namespace std;

#include "interpolations.h"



// Interpolates the fields, splitting the work across OpenMP threads when enabled.
void DTFE_parallel(vector<Particle_data> *allParticles,
                   vector<Sample_point> &samples,
                   User_options & userOptions,
                   Quantities *uQuantities,
                   Quantities *aQuantities);
#ifdef PHASE_SPACE
// the partition look-ahead's two halves (triangulation.cpp): the tessellation + vertex densities of a
// partition built ahead (an opaque handle: this file does not see CGAL), and its grid passes later
std::shared_ptr<void> psPrebuildTessellation(vector<Particle_data> *p, User_options &userOptions);
void psInterpolatePrebuilt(std::shared_ptr<void> &tessellation, vector<Sample_point> &samples,
                           User_options &userOptions, Quantities *uQuantities, Quantities *aQuantities);
#else
// the standard partition loop's look-ahead (triangulation.cpp): one thread sub-domain's tessellation +
// vertex densities built ahead (an opaque handle), and its interpolation later
std::shared_ptr<void> dtfePrebuildTessellation(vector<Particle_data> *p, User_options &userOptions);
void dtfeInterpolatePrebuilt(std::shared_ptr<void> &tessellation, vector<Sample_point> &samples,
                             User_options &userOptions, Quantities *uQuantities, Quantities *aQuantities);
#endif
// One standard partition's sub-domains as the look-ahead prepared them: DTFE_parallel's thread split of
// the partition (or its single tessellation) and every sub-domain's tessellation, built; DTFE_parallelImpl then only
// interpolates them. (Declared in both builds, used by the standard partition loop only.)
struct DTFEPrepared
{
    bool single = true;                                // DTFE_parallel's single-tessellation branch
    std::vector<size_t> pGrid;                         // else its thread grid, its split and the sub-domains' tessellations
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    std::vector< std::shared_ptr<void> > tessellation; // [thread number]: the sub-domain DTFE_parallel's thread i takes
    std::vector<size_t> nParticles;                    // [thread number]: its padded particles (the thread statistics)
    size_t noTotalParticles = 0;
};
static void DTFE_parallelImpl(vector<Particle_data> *allParticles, vector<Sample_point> &samples,
                              User_options &userOptions, Quantities *uQuantities, Quantities *aQuantities,
                              DTFEPrepared *prepared);

// Builds the velocity-space DTFE density g_i used by the approximate phase-space density (--approx-phase-space-density).
#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
extern void computeVelocitySpaceDensity(vector<Particle_data> &particles,
                                         User_options &userOptions);
#endif


// A partition loop announces how many tessellations it builds at once and how many it has left
// (triangulation.cpp divides the cores of its parallel insertion by the smaller of the two);
// each partition counts itself off when it is done with its build and deposit.
extern std::atomic<int> dtfeTriangulationConcurrency;
extern std::atomic<int> dtfeTriangulationsRemaining;
extern std::atomic<int> dtfePartitionsInFlight;   // (thread_count.h: the dynamic PS loop's partitions in flight)
struct ConcurrentTriangulationPlan
{
    ConcurrentTriangulationPlan(int const concurrency, int const total)
    { dtfeTriangulationConcurrency = concurrency; dtfeTriangulationsRemaining = total; }
    ~ConcurrentTriangulationPlan() { dtfeTriangulationConcurrency = 0; dtfeTriangulationsRemaining = 0; }
};
struct ConcurrentTriangulationGuard
{
    ~ConcurrentTriangulationGuard() { --dtfeTriangulationsRemaining; }
};

/* THE LOOK-AHEAD of both partition loops (psRunPartitionLoop, dtfePartitionedRun; their comments say why). The
   partitions in 'order' are deposited in that order on this thread (deposit(i, P)), each prepared before
   (prepare(i, build, P)): the first in turn, every next one with its tessellation built (build = true) on a thread of
   its own while the current one deposits -- when the free memory less 4 GB of headroom holds the needBytes(next) it
   takes and canPrefetch(next) -- else in turn after the deposit (build = false). The preparation alone overlaps, so the
   outputs are those of the loop without it. An exception of either stage is rethrown once the thread has joined.
   Scope: constructed at each iteration's start and destroyed at its end, after the next partition is prepared (the PS
   loop counts its triangulations off with it). The 'prepared in' line names the next partition by its index plus
   label.shownBase and ends 'during <label.during>'; it is not said for a partition P.emptyPartition marks. */
struct LookAheadLabel
{
    int shownBase;          // added to a partition's index on the 'prepared in' line (PS-DTFE 0, standard DTFE 1)
    char const *during;     // what the preparation overlapped ("this deposit", "this one")
};
struct NoLookAheadScope {};
template <class Scope, class Prepared, class NeedBytes, class Prepare, class Deposit, class CanPrefetch>
static void runLookAhead(std::vector<int> const &order, NeedBytes &&needBytes, Prepare &prepare, Deposit &&deposit,
                         CanPrefetch &&canPrefetch, LookAheadLabel const &label, int const verboseLevel)
{
    Prepared cur;
    prepare( order[0], false, cur );
    bool saidNoRoom = false;
    for (size_t k=0; k<order.size(); ++k)
    {
        Scope scope;
        (void)scope;
        int const i = order[k];
        Prepared next;
        std::thread ahead;
        std::exception_ptr aheadError;
        bool aheadRan = false;
        if ( k + 1 < order.size() )
        {
            int const nextI = order[k+1];
            double const need = needBytes( nextI );
            double const freeNow = autoTuneSystemFree();
            bool const room = freeNow < 0. or freeNow - 4.e9 >= need;
            if ( room and canPrefetch( nextI ) )
            {
                aheadRan = true;
                ahead = std::thread( [&, nextI]()
                {
                    // this thread's core counted in the core pool while it prepares: partition k's pooled loops and a
                    // CPU-share deposit's team beside it then do not take it as free (thread_count.h CorePool::Hold)
                    CorePool::Hold const hold( 1 );
                    dtfeAdoptCoreBudget();   // a fresh thread has the OpenMP default: the run mode's budget too
                    try { prepare( nextI, true, next ); }
                    catch ( ... ) { aheadError = std::current_exception(); }
                } );
            }
            else if ( not room and not saidNoRoom )
            {
                saidNoRoom = true;
                MESSAGE::Message m( verboseLevel );
                m << MESSAGE::cDim() << "  look-ahead paused: " << freeNow/1.e9 << " GB free, minus 4 GB headroom, "
                  << "is below the ~" << need/1.e9 << " GB it holds; partitions are prepared in turn "
                  << "until memory frees up\n" << MESSAGE::cReset() << MESSAGE::Flush;
            }
        }
        try
        {
            deposit( i, cur );
        }
        catch ( ... )
        {
            if ( ahead.joinable() ) ahead.join();
            throw;
        }
        if ( ahead.joinable() ) ahead.join();
        if ( aheadError ) std::rethrow_exception( aheadError );
        if ( aheadRan and verboseLevel > 0 and not next.emptyPartition )
        {
            MESSAGE::Message m( verboseLevel );
            m << MESSAGE::cDim() << "  look-ahead: partition " << order[k+1] + label.shownBase << " prepared in "
              << MESSAGE::formatDuration( next.seconds ) << " during " << label.during << "\n" << MESSAGE::cReset()
              << MESSAGE::Flush;
        }
        if ( k + 1 < order.size() )
        {
            if ( not aheadRan ) prepare( order[k+1], false, next );
            cur = std::move( next );
        }
    }
}

// The '[partitions k/N | p%]  elapsed ..., ETA ~...' line of both partition loops. The caller rounds the percentage:
// the two loops form it in different orders (100*(k/N) and (100*k)/N), which round apart at exact halves (23 of 40),
// and each keeps its own.
static void partitionProgress(int const verboseLevel, int const done, int const total, int const percent,
                              double const elapsed, double const eta)
{
    MESSAGE::Message prog( verboseLevel );
    prog << MESSAGE::cGreen() << "  [partitions " << done << "/" << total
         << " | " << percent << "%]" << MESSAGE::cReset()
         << MESSAGE::cDim() << "  elapsed " << MESSAGE::formatDuration(elapsed)
         << ( done<total ? ", ETA ~" + MESSAGE::formatDuration(eta) : ", done" )
         << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
}

// Particle data plus the resolved per-run options produced by the common DTFE setup.
struct DTFE_State {
    vector<Particle_data> particles;
    User_options options;
};

#include "dtfe/ps_partition_plan.cc"   // PS: the Lagrangian box, the partition plan and costs, the GPU share
#include "dtfe/setup.cc"               // the run's setup: units, DTFE_setup, padding, the thread sub-domains

#ifdef PHASE_SPACE
#include "dtfe/ps_partition_setup.cc"   // PS: normalization, periodic setup, window, serve, grids, deferred read
#include "dtfe/ps_partition_loop.cc"    // PS: the partition loop and its merge staging
#include "dtfe/ps_partitioned_run.cc"   // PS: closing the moments, the report, psPartitionedRun
#endif // PHASE_SPACE

#include "dtfe/std_partitioned_run.cc"   // standard: the composite serve and dtfePartitionedRun

// After the interpolation: the point records reduced, the masks finalized, the web fields made, the counts reported.
static void dtfeFinishFields(DTFE_State &state, User_options &userOptions, Quantities *uQuantities,
                             Quantities *aQuantities)
{
    // all partitions (or the single triangulation) have contributed their streams: reduce the
    // per-point records to the final density/velocity/dispersion/stream-count outputs
    if ( psPointEvalActive() )
        psPointEvalFinalize( state.options );

#ifdef PHASE_SPACE
    // --ps-caustics: by now every partition has OR-merged its orientation bits -- a fold straddling a
    // partition boundary gets its two orientations from different partitions, so collapsing the bits any
    // earlier would lose it. The mask is deliberately NOT collapsed here: it stays full until writeOutputData
    // (input_output.cc), which writes '.causticClass' from it and only then collapses it in place to the 0/1
    // fold flag for '.caustic' (bits==3, both parities overlap -> 1, else 0) -- one grid instead of two, saving
    // 4 B/cell per Quantities across the post-processing window where peak RSS is set. The write order there is
    // load-bearing. The collapse is not idempotent (1 -> 0 on a second pass) and runs once per invocation
    // (--interlace is rejected with this flag at option parsing).
    // '.hidden_streams' bit 1: only now, after the merge, with '.streams' summed over every partition
    uQuantities->finalizeHiddenStreams();
    aQuantities->finalizeHiddenStreams();
#endif

    // post-processing: derive velocity divergence/shear/vorticity and cosmic-web labels from the gradient
    runProgressNoteLoopPeak();   // the SUMMARY's 'before post-proc' figure: which phase set the peak
    runProgressPhase( "post-processing" );
    postProcessWebFields( userOptions, uQuantities, aQuantities );

    // the run's phase totals (phase_totals.h): every stage printComputationTime timed, summed over all threads and
    // sub-domains -- what each phase cost the machine
    PhaseTotals::report( userOptions.verboseLevel );

#ifndef PHASE_SPACE
    // a GPU run: the sub-domain deposits that fell back to the CPU (as the PS binary's partition count
    // after its loop): each prints its reason as a warning of its own thread, and this line says how many there
    // were. Once per DTFE() call.
    if ( int const nFallback = dtfeGpuFallbacksTaken( true ) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << nFallback << " sub-domain deposit(s) were meant for the GPU and ran on the CPU instead (the reasons "
                << "are above); the grids are complete, the run only took longer.\n" << MESSAGE::EndWarning;
    }
#endif

    // the failed GPU buffers / kernels of the run by cause, one line the launcher's Runs tab
    // reads (the progress sidecar that carries the same counts is removed when the run succeeds). The counts
    // are cumulative over the process, so '--interlace' (two DTFE() calls) prints the line twice and the last
    // one holds the run's total. Both binaries (the standard GPU hosts count by the same causes); nothing is
    // printed without a failed buffer, so a CPU run's log has no such line.
    if ( int const nRetry = runProgressGpuRetries() )
    {
        MESSAGE::Message retries( userOptions.verboseLevel );
        retries << "GPU deposit retries: " << nRetry << " (innocent " << runProgressGpuRetries( 0 ) << ", timeout "
                << runProgressGpuRetries( 1 ) << ", oom " << runProgressGpuRetries( 2 ) << ", other "
                << runProgressGpuRetries( 3 ) << ")\n" << MESSAGE::Flush;
    }

    // the shares the non-finite guard of the exact sums dropped in this DTFE() call, by grid
    fixedShareDropsReact( userOptions );
}

// Top-level DTFE interpolation onto a grid; dispatches to the standard, partitioned, or PS-DTFE path. Clears allParticles.
void DTFE(vector<Particle_data> *allParticles,
          vector<Sample_point> &samples,
          User_options & userOptions,
          Quantities *uQuantities,
          Quantities *aQuantities)
{
    DTFE_State state = DTFE_setup( allParticles, samples, userOptions );

    // approxPSD: compute velocity-space density g_i for all particles before
    // partitioning/padding so the velocity tessellation uses the global distribution
#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
    if ( state.options.approxPSD && !state.particles.empty() )
        computeVelocitySpaceDensity( state.particles, state.options );
#endif

#ifdef PHASE_SPACE
    // PS-DTFE periodic padding: triangulation is built in Lagrangian space, so periodic
    // copies use shifted Lagrangian (not Eulerian) positions.
    Box lagBoxGlobal;  // saved for later use by partitioning
    lagBoxGlobal.assign(Real(0.));
    bool hasLagrangianPeriodicCopies = false;
    psCloudNormalization( state, userOptions );
    psLagrangianPeriodicSetup( state, userOptions, lagBoxGlobal, hasLagrangianPeriodicCopies );
#endif

    // arbitrary-point evaluation (--sample-points), both binaries: load the query points and
    // build their bucket index once, now that region/periodic/averageDensity are final. Each
    // triangulation then contributes its streams; psPointEvalFinalize() reduces them at the end.
    // --serve arms the same context without points; each request supplies its own.
    if ( not state.options.psSamplePointsFile.empty() or state.options.psServe )
        psPointEvalInit( state.options );

#ifdef PHASE_SPACE
    psResolveWindow( state, userOptions );
#endif

#ifdef PHASE_SPACE
    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        // PS-DTFE partitions in Lagrangian space (Eulerian partitioning fails because Lagrangian cells
        // map to arbitrary Eulerian locations). Each partition writes the full grid; results are summed.
        psPartitionedRun( state, samples, userOptions, uQuantities, aQuantities, lagBoxGlobal,
                          hasLagrangianPeriodicCopies );
    }
    else
#endif // PHASE_SPACE

    // standard DTFE: process the spatial partitions one at a time to bound peak memory on large data
    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        dtfePartitionedRun( state, samples, userOptions, uQuantities, aQuantities );
    }
    else
    {
        // single triangulation over the whole (padded) box
        DTFE_preparePadding( state, userOptions );
        DTFE_parallel( &state.particles, samples, state.options, uQuantities, aQuantities );
    }

    dtfeFinishFields( state, userOptions, uQuantities, aQuantities );
}



// Builds the Delaunay triangulation and interpolates the fields onto the grid (DTFE method).
extern void DTFE_interpolation(vector<Particle_data> *p,
                               vector<Sample_point> &samples,
                               User_options &userOptions,
                               Quantities *uQuantities,
                               Quantities *aQuantities);

// Dispatches to the interpolation method selected in userOptions (DTFE, NGP, CIC, TSC, PCS, SPH).
void interpolate(vector<Particle_data> *allParticles,
                 vector<Sample_point> &samples,
                 User_options & userOptions,
                 Quantities *uQuantities,
                 Quantities *aQuantities)
{
    if ( userOptions.DTFE )
        DTFE_interpolation( allParticles, samples, userOptions, uQuantities, aQuantities );
    else if ( userOptions.NGP )
        NGP_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.CIC )
        CIC_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.TSC )
        TSC_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.PCS )
        PCS_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.SPH )
        SPH_interpolation( allParticles, samples, userOptions, aQuantities );
    else
        throwError( "Unknown interpolation method in function 'interpolate'." );
}




// Splits the data into spatial partitions computed in parallel via OpenMP.
void DTFE_parallel(vector<Particle_data> *allParticles,
                   vector<Sample_point> &samples,
                   User_options & userOptions,
                   Quantities *uQuantities,
                   Quantities *aQuantities)
{
    DTFE_parallelImpl( allParticles, samples, userOptions, uQuantities, aQuantities, nullptr );
}

// ... the body; 'prepared' (the standard partition loop's look-ahead) holds the split and the sub-domains'
// tessellations already built from these options (dtfePrepareSubdomains), so only their interpolation is left
static void DTFE_parallelImpl(vector<Particle_data> *allParticles,
                              vector<Sample_point> &samples,
                              User_options & userOptions,
                              Quantities *uQuantities,
                              Quantities *aQuantities,
                              DTFEPrepared *prepared)
{
#ifndef PHASE_SPACE
    // a prepared single tessellation (as the branches below, which interpolate() it)
    auto interpolatePrepared = [&](size_t const sub, User_options &o, Quantities *u, Quantities *a)
    {
        if ( not userOptions.DTFE )
            throwError( "internal: a look-ahead tessellation reached a non-DTFE interpolation." );
        dtfeInterpolatePrebuilt( prepared->tessellation[sub], samples, o, u, a );
    };
#else
    if ( prepared ) throwError( "internal: the standard look-ahead reached PS-DTFE's DTFE_parallel." );
    auto interpolatePrepared = [&](size_t, User_options &, Quantities *, Quantities *) {};
#endif
#ifndef OPEN_MP
    if ( prepared ) interpolatePrepared( 0, userOptions, uQuantities, aQuantities );
    else interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );  // deletes allParticles
    return;
#else
    // cap concurrent sub-triangulations to bound peak memory (--max-concurrent); 0 = all threads
    int const noAvailableProcessors = dtfeParallelThreads( userOptions );

    // fall back to a single triangulation when spatial partitioning does not apply (dtfeParallelSingle: 1 thread,
    // user sampling, redshift cone, --sample-points point evaluation, PS-DTFE); a prepared partition decided it
    if ( prepared ? prepared->single : dtfeParallelSingle( userOptions, samples, noAvailableProcessors ) )
    {
        if ( prepared ) interpolatePrepared( 0, userOptions, uQuantities, aQuantities );
        else interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );
        return;
    }


    // more than 1 thread: split the data into spatial partitions, one per thread
    std::vector<size_t> pGrid(NO_DIM,0); // the parallel grid
    if ( prepared ) pGrid = prepared->pGrid;
    else parallelGrid( noAvailableProcessors, userOptions, &(pGrid[0]) );
    int noProcessors = 1;
    for (int d=0; d<NO_DIM; ++d) noProcessors *= pGrid[d];    //number of processors actually used (may differ from 'noAvailableProcessors')

    uQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.uField );
    aQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.aField );
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "From now on only the master thread will show messages on how the computation is going. Not all threads take the same execution time, so there may be a discrepancy between the messages displayed to the user and the computations across all threads.\n\n" << MESSAGE::Flush;
    std::vector<size_t> processorParticles(noProcessors);    // number of particles associated to each processor
    std::vector<Real> processorTime(noProcessors);           // the actual runtime of each thread
    size_t const noTotalParticles = prepared ? prepared->noTotalParticles : allParticles->size();
    

    // choose the partition geometry that balances the particle load across threads
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    if ( prepared )
    {
        subgridList = prepared->subgridList;
        subgridCoords = prepared->subgridCoords;
    }
    else
        optimalPartitionSplit( *allParticles, userOptions, pGrid, &subgridList, &subgridCoords );


    // the team's concurrency, for the parallel insertion's thread budget (triangulation.cpp divides the cores by it,
    // as in the PS partition loops; without it each of the noProcessors sub-boxes would insert on 6 threads at once)
    ConcurrentTriangulationPlan triangulationPlan( noProcessors, noProcessors );
#pragma omp parallel num_threads( noProcessors )
    {
        int const threadNo = omp_get_thread_num();
        ConcurrentTriangulationGuard concurrentTriangulation;
        // region allocated to this thread (dtfeSubdomainOptions: its sub-grid, box and padded box)
        User_options tempOptions = dtfeSubdomainOptions( userOptions, threadNo, noProcessors, pGrid, subgridList, subgridCoords );


        Quantities temp_uQuantities, temp_aQuantities;
        if ( prepared )   // the look-ahead built this sub-domain's tessellation from the same options and particles
        {
            processorParticles[ threadNo ] = prepared->nParticles[ size_t(threadNo) ];
            interpolatePrepared( size_t(threadNo), tempOptions, &temp_uQuantities, &temp_aQuantities );
        }
        else
        {
            vector<Particle_data> particles;
            findParticlesInBox( *allParticles, &particles, tempOptions );
            processorParticles[ threadNo ] = particles.size();
            interpolate( &particles, samples, tempOptions, &temp_uQuantities, &temp_aQuantities ); //this function deletes the vector 'particles'
        }
        processorTime[ threadNo ] = tempOptions.totalTime;


        // copy this partition's results to the main grid
        copySubgridResultsToMain( temp_uQuantities, userOptions.gridSize, tempOptions.uField, tempOptions, subgridList, uQuantities );
        copySubgridResultsToMain( temp_aQuantities, userOptions.gridSize, tempOptions.aField, tempOptions, subgridList, aQuantities );
        if ( threadNo==0 )
            message << "\nWaiting for all threads to finish the computations ... " << MESSAGE::Flush;
    }
    message << "Done.\n";
    allParticles->clear();


    // total runtime is the slowest thread's CPU time, since the run finishes only when all threads do
    userOptions.totalTime += maximum( processorTime.data(), noProcessors );
    approximativeThreadTime( processorTime.data(), noProcessors );
    message << "Statistics of the execution across the " << noProcessors << " threads:\n";
    for (int i=0; i<noProcessors; ++i)
        message << "\t Thread " << i << " had " << processorParticles[i] << " particles (which represent " << setprecision(4) <<  Real(processorParticles[i])/noTotalParticles*100. << "\%) and took " << processorTime[i] << " sec. \n";
    message << MESSAGE::Flush;
    
#endif
}






// shared zero placeholders for the velocity/scalar members that are compiled out
#ifndef VELOCITY
Pvector<Real,noVelComp> Data_structure::_velocity = Pvector<Real,noVelComp>::zero();
#endif
#ifndef SCALAR
Pvector<Real,noScalarComp> Data_structure::_scalar = Pvector<Real,noScalarComp>::zero();
#endif






#ifdef TRIANGULATION
#include "DTFE.h"

// Builds the triangulation and interpolates to grid, also returning the Delaunay triangulation.
extern void DTFE_interpolation(vector<Particle_data> *p,
                               vector<Sample_point> &samples,
                               User_options &userOptions,
                               Quantities *uQuantities,
                               Quantities *aQuantities,
                               DT &delaunay_triangulation);

// DTFE interpolation that also hands back the Delaunay triangulation (single-triangulation only). Clears allParticles.
void DTFE(vector<Particle_data> *allParticles,
          vector<Sample_point> &samples,
          User_options & userOptions,
          Quantities *uQuantities,
          Quantities *aQuantities,
          DT &delaunay_triangulation)
{
    DTFE_State state = DTFE_setup( allParticles, samples, userOptions );

    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        throwError( "There is no implementation of the option '--partition' in the absence of option '--partition-index' when requering that the 'DTFE' function returns the Delaunay triangulation." );
    }
    else
    {
        DTFE_preparePadding( state, userOptions );
        DTFE_interpolation( &state.particles, samples, state.options, uQuantities, aQuantities, delaunay_triangulation );
    }

    postProcessWebFields( userOptions, uQuantities, aQuantities );
    fixedShareDropsReact( userOptions );   // the non-finite guard's dropped shares, as at the end of DTFE() above
}
#endif

