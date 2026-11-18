// Part of src/DTFE.cpp: included once, in order, by that file and compiled only as part of its object, never alone.
// The run's setup: standard DTFE's run-wide units (runUnitsSetup), DTFE_setup, the padding, the thread
// sub-domains (dtfeSubdomainOptions, dtfePrepareSubdomains), psUnwrapEulerian and fixedShareDropsReact.

/* The run-wide fixed-point units (fixed_units.h RunUnits) of standard DTFE: one unit per accumulator for the whole
   run, from the grid, the mean density and one pass over the particles the run uses (after '--poisson' and
   '--random-sample'). Set here, before the planner, the partition loop and any thread sub-domain, so every split holds
   the same units; the copies of the options carry them. The cell sides are the grid box over '--grid' -- 'region':
   the box, or the '--region' box DTFE_setup crops to below -- as the grid frame's (user_options.h gridFrame). A run
   without a grid (sample points only) has none. One run-log line, from global quantities only, so it is the same at
   every split. PS-DTFE sets none: its passes take units of their own. */
#ifndef PHASE_SPACE
static void runUnitsSetup(User_options &u, vector<Particle_data> &particles)
{
    u.runUnits = RunUnits();
    if ( u.gridSize.size() != size_t(NO_DIM) ) return;
    MESSAGE::Message message( u.verboseLevel );
    double side[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
        side[d] = ( double( u.region.coords[2*d+1] ) - double( u.region.coords[2*d] ) ) / double( u.gridSize[d] );
    RunMoments const moments = RunUnits::measure( particles );
    u.runUnits.fill( side, double( u.averageDensity ), moments );
    message << u.runUnits.text() << "\n" << MESSAGE::Flush;
    std::string const out = u.runUnits.outOfRange();
    if ( not out.empty() )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "fixed-point units (run-wide): the unit of " << out << " lies outside 2^-120 .. 2^120; that grid "
                << "keeps its per-batch or per-pass unit.\n" << MESSAGE::EndWarning;
    }
#ifdef SCALAR
    // '--approx-phase-space-density' replaces each particle's scalar by f = rho g after this point (DTFE()'s
    // computeVelocitySpaceDensity, then the vertex loop after the tessellation, triangulation.cpp), so smax measured
    // the file's scalar, not the f the scalar grids sum: those two take no run-wide unit (has() false) and keep the
    // GPU batch's own unit and the CPU's float sums, whose last bits follow the split
    if ( u.approxPSD )
    {
        u.runUnits.ok[RU_SCALAR] = u.runUnits.ok[RU_SCALAR_GRADIENT] = false;
        message << "fixed-point units (run-wide): the scalar and scalar gradient grids keep the GPU batch's own unit "
                << "and the CPU's float sums ('--approx-phase-space-density' sets the scalar to f = rho g later).\n"
                << MESSAGE::Flush;
    }
#endif
}
#endif


// Common DTFE setup: optional random/subsampled particles, consistency checks, velocity-derivative
// selection, and region/partition extraction. Mutates userOptions (averageDensity, region).
DTFE_State DTFE_setup(vector<Particle_data> *allParticles,
                      vector<Sample_point> &samples,
                      User_options &userOptions)
{
    DTFE_State state;

    if ( userOptions.poisson!=0 )
        randomParticles( allParticles, &userOptions );

    // optionally replace the data with a random subsample of it
    vector<Particle_data> *particlePointer = allParticles;
    vector<Particle_data> particlesRandomSubsample;
    if (userOptions.randomSample>=Real(0.) )
    {
        randomSample( *particlePointer, &particlesRandomSubsample, userOptions );
        particlePointer->clear();
        particlePointer = &particlesRandomSubsample;
    }

    // run consistency checks and finalize derived options; default the density normalization to the box average
    // (a run that started without its particles takes their count and mean density from the globals record)
    size_t const nParticles = userOptions.psParticlesDeferred ? userOptions.psPresetN : particlePointer->size();
    userOptions.updateEntries( nParticles, not samples.empty() );
    if ( userOptions.averageDensity<0. )
    {
        userOptions.averageDensity = averageDensity( *particlePointer, userOptions );
        userOptions.psAverageDensityFromParticles = true;
    }
#ifndef PHASE_SPACE
    runUnitsSetup( userOptions, *particlePointer );
#endif

    // --scratch-dir: arm the out-of-core allocator BEFORE any full-grid allocation and before
    // auto-tune (which excludes the disk-backed grid term from its RAM model when armed). The
    // path itself was validated at option parsing; the arming can still fail in the library
    // build, whose scratch_alloc STUB never hijacks the host program's allocator.
    if ( not userOptions.scratchDir.empty() )
    {
        double thresholdGB = 1.0;
        if ( double const v = Env::real( "DTFE_SCRATCH_MIN_GB", 0. ); v > 0. )
            thresholdGB = v;
        if ( ScratchAlloc::scratchArm( userOptions.scratchDir.c_str(), size_t(thresholdGB*1.e9) ) )
        {
            MESSAGE::Message message( userOptions.verboseLevel );
            message << MESSAGE::cBold() << "SCRATCH:" << MESSAGE::cReset() << " allocations >= "
                    << thresholdGB << " GB (the full-grid accumulators) are backed by mmap'ed files in '"
                    << userOptions.scratchDir << "' -- unlinked at creation, so they free themselves on any exit.\n"
                    << MESSAGE::Flush;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "'--scratch-dir' has no effect in the library build (libDTFE must not replace the host "
                    << "program's global allocator); the full grids stay in RAM.\n" << MESSAGE::EndWarning;
            userOptions.scratchDir.clear();   // keep auto-tune's RAM model honest
        }
    }

    // --grid-window: the auto-tuner sizes the grids by the cells actually allocated -- the window's --
    // so its cell range must be known first (the header's box is final here; DTFE() computes it again,
    // identically). Without this it would see the whole virtual grid of a zoom,
    // call the run impossible and fall back to one partition at a time.
    if ( userOptions.psWindowOn )
        userOptions.computePsWindow();

    // '--resume' without '--partition': the split is the checkpoint's. The auto-tuner would pick it
    // from the free memory of THIS moment, which need not be the killed run's choice, and a different split
    // cannot continue the merged partitions; with the split given it still tunes '--max-concurrent'.
    // Both binaries (run_dtfe.sh, like run_ps_dtfe.sh, leaves the split to the auto-tuner).
    if ( userOptions.psResume and not userOptions.partitionGiven and userOptions.partNo < 0 )
    {
        PartitionCheckpoint::Manifest m;
        std::string why;
        std::string const dir = PartitionCheckpoint::directoryFor( userOptions );
        if ( not PartitionCheckpoint::readManifest( dir, m, why ) )
            throwError( "'--resume': nothing to resume -- ", why, ". Run without '--resume' to start the run." );
        if ( m.partition.size() != size_t(NO_DIM) )
            throwError( "'--resume': the checkpoint in '", dir, "' names no partition split of ", NO_DIM, " numbers." );
        userOptions.partition = m.partition;
        size_t n = 1;
        for ( size_t p : m.partition ) n *= p;
        userOptions.partitionOn = n > 1;
        userOptions.partitionGiven = true;
        MESSAGE::Message message( userOptions.verboseLevel );
        message << MESSAGE::cBold() << PartitionCheckpoint::NAME << ":" << MESSAGE::cReset() << " --resume: the partition split ["
                << MESSAGE::printElements( m.partition, "," ) << "] is the checkpoint's.\n" << MESSAGE::Flush;
    }

    /* '--region' / '--region-mpc' (standard DTFE; PS-DTFE rejects them at option parsing): the run triangulates only
       the region plus its padding, so the auto-tuner must plan THAT, not the whole snapshot: planned from the
       snapshot's particle count and box, a launcher exact zoom of a small region would be split into partitions
       and threads -- sub-triangulations each padded again around its tiny sub-box -- far slower than one domain.
       So the region is cropped HERE, before the tuner, and the
       tuner is handed a copy of the options whose box is the region, with the particles inside the UNPADDED
       region as its count. That is the same pairing as a whole-box run (the box's particles; the model adds the
       padding on top as a fraction of the box), so one domain predicts one padded region. Only the split and the
       concurrency it chooses are copied back: nothing else of the standard branch (or of autoTuneServe) is read
       later. */
    vector<Particle_data> particlesRegion;   // the padded region's particles, for the crop below
    bool regionCropped = false;
    User_options tuneOptions;                // the options the tuner plans a region run with
    size_t nTune = nParticles;
#ifndef PHASE_SPACE
    if ( userOptions.regionOn )
    {
        tuneOptions = userOptions;
        tuneOptions.region.validSubBox( tuneOptions.boxCoordinates, tuneOptions.periodic );
        tuneOptions.paddedBox = tuneOptions.region;
        tuneOptions.paddedBox.addPadding( tuneOptions.paddingLength );
        findParticlesInBox( *particlePointer, &particlesRegion, tuneOptions );   // periodic images as the crop takes them
        regionCropped = true;
        // the selected copies sit in the padded box's frame (images shifted in), so the unpadded region is a
        // plain box test here, periodic or not
        Box const &r = tuneOptions.region;
        nTune = size_t( std::count_if( particlesRegion.begin(), particlesRegion.end(),
                                       [&r](Particle_data &p) { return r.isParticleInBox( p ); } ) );
        tuneOptions.updateFullBox( tuneOptions.region );   // the box the model takes its padding fraction from
        tuneOptions.periodic = false;                       // the cropped run is a standalone non-periodic box
    }
#endif

    // auto-select --partition / --max-concurrent from the data and machine when not user-given
    {
        bool gpuActive = false;
#if defined(PS_GPU)
        gpuActive = userOptions.psUseMetal;
#elif defined(DTFE_GPU)
        gpuActive = userOptions.useMetal;
#endif
#ifdef PHASE_SPACE
        // the GPU share first: the tuner books a mixed run's partitions by it (auto_tune.cc mixedLane)
        psResolveGpuShare( userOptions, nParticles );
#endif
        // the particles the run triangulates: a region run's padded crop (in the region's frame, periodic images
        // shifted in), whose counts the standard tuner replays its split on (auto_tune.h StdTuneDensity)
        AutoTuneReport const report = autoTunePartitioning( regionCropped ? tuneOptions : userOptions, nTune,
                                                            not samples.empty(), gpuActive,
                                                            regionCropped ? &particlesRegion : particlePointer );
        if ( regionCropped )
        {
            userOptions.partition      = tuneOptions.partition;
            userOptions.partitionOn    = tuneOptions.partitionOn;
            userOptions.maxConcurrent  = tuneOptions.maxConcurrent;
#ifndef PHASE_SPACE
            userOptions.stdLookAheadOffBudget = tuneOptions.stdLookAheadOffBudget;
            userOptions.stdLookAheadCounted   = tuneOptions.stdLookAheadCounted;
#endif
        }
        if ( report.valid ) runProgressSetPredicted( report.predicted );   // the SUMMARY's predicted-vs-measured line
#ifndef PHASE_SPACE
        // what the budget leaves for the partition loop's look-ahead beside the busiest partition (as PS's)
        if ( report.valid )
            userOptions.stdLookAheadRoom = report.budget - ( report.fixedLoop > 0. ? report.fixedLoop : report.fixed )
                                         - report.gpuSnapshot - report.perPartition;   // the loop's room (stdPeak)
#endif
#ifdef PHASE_SPACE
        if ( report.valid )
        {
            // the partition memory gate (auto_tune.h). It compares ONE partition's need with the free memory, so
            // the GPU deposit's host snapshot -- one at a time, allocated during a deposit, not yet in the free
            // memory a partition sees when it starts -- is in each check once. A whole lane: under the lane share
            // (auto_tune.cc laneShareCut) the first partition's laneOnce is part of it
            userOptions.psPartitionBytes = report.perPartition + report.laneOnce + report.gpuSnapshot;
            // A MIXED run's split ('--gpu-share' < 1): that term is the GPU lock holder's device and host grids too,
            // larger than a lane's own arrays, and still held by one partition whatever the number in flight. The
            // dynamic concurrency's ledger, which counts every partition in flight, books what the plan books once
            // (onceBytes) -- that term (in an all-GPU run the snapshot alone) and the lane share's laneOnce -- once,
            // and every partition at the plan's lane (perPartition)
            userOptions.psOnceBytes = report.onceBytes;
            // what the budget leaves for the look-ahead beside one partition (checked against the free memory
            // only, a budget set low on purpose would be passed by one tessellation)
            userOptions.psLookAheadRoom = report.budget - report.fixed - report.gpuSnapshot - report.perPartition
                                        - report.laneOnce;
            // what the budget leaves beside the predicted peak -- with every lane booked whole (predictedWholeLanes:
            // the measured runs that set the lane share spilled under that booking) -- for partitions parked by the
            // staged ordered merge: the prediction does not count them, and on macOS the free-memory reading alone
            // does not stop them (the compressor and swap keep 'free' above the spill threshold); a lane less for
            // each partition the dynamic concurrency runs beyond the plan's (parkWhere). DTFE_TEST_PARK_ROOM_GB sets
            // it (tests).
            userOptions.psParkRoom = report.budget - report.predictedWholeLanes;
            // the partition loop's dynamic concurrency: only over a concurrency the tuner chose
            userOptions.psDynamicConcurrency = report.concChosen;
            userOptions.psInFlightCap = report.inFlightCap;   // ... and never past the lane share's cap (0: none)
            userOptions.psReserveEnd = report.reserveEnd;
            userOptions.psReserveRecords = report.reserveRecords;
            userOptions.psReserveFirstMerge = report.reserveFirstMerge;
            userOptions.psBudgetBytes = report.budget;
            // test hook: DTFE_TEST_PLAN_MC=k stands for a plan that chose k partitions at once (the dynamic loop's floor)
            if ( report.concChosen and Env::integer( "DTFE_TEST_PLAN_MC", 0 ) > 0 )
                userOptions.maxConcurrent = int( Env::integer( "DTFE_TEST_PLAN_MC", 0 ) );
        }
        if ( Env::isSet( "DTFE_TEST_PARK_ROOM_GB" ) )
            userOptions.psParkRoom = std::max( 0., Env::real( "DTFE_TEST_PARK_ROOM_GB", 0. ) ) * 1.e9;
        // one domain has no partition loop, and so no GPU share: known only now, from the plan (psResolveGpuShare).
        // Its report is the all-GPU one already (the tuner's mixed lanes are for a split), so the plan stands
        if ( not userOptions.partitionOn )
        {
            if ( userOptions.psGpuShareMille < 1000
                 and userOptions.psGpuShareSpec >= 0 )   // (a number: 'auto' and 'mixed' say nothing here)
                psGpuShareNote( userOptions, "one domain (no partition loop)", "the GPU" );
            userOptions.psGpuShareMille = 1000;
        }
#endif
        if ( userOptions.autoTuneReport )   // the memory check: the decision and prediction, nothing else
        {
            std::printf( "%s\n", report.line().c_str() );
            std::fflush( stdout );
            std::exit( 0 );
        }
    }

    // only the partition loop checkpoints, so only a partitioned run has anything to resume (both binaries)
    if ( userOptions.psResume and not ( userOptions.partitionOn and userOptions.partNo < 0 ) )
        throwError( "'--resume' continues a partitioned run ('--partition' with more than one partition); this run has "
                    "one triangulation. Run it without '--resume'." );

    // velocity derivatives are computed from the gradient afterwards, so request the gradient now
    // and defer the derived quantities to the post-processing step
    state.options = userOptions;
    if ( userOptions.uField.selectedVelocityDerivatives() )
    {
        state.options.uField.velocity_gradient = true;
        state.options.uField.deselectVelocityDerivatives();
    }
    if ( userOptions.aField.selectedVelocityDerivatives() )
    {
        state.options.aField.velocity_gradient = true;
        state.options.aField.deselectVelocityDerivatives();
    }

#ifdef PHASE_SPACE
    // --face-gradient: the mean-flow gradient (S-flux - <v> N-flux) / sum V_s needs the cell mean velocity <v>, which is
    // then deposited whether or not 'velocity_a' is written (the writer goes by the user's own fields)
    if ( userOptions.faceGradient and state.options.aField.velocity_gradient ) state.options.aField.velocity = true;
    // ... and the regular part's own gradient, shear and vorticity are kept beside the mean flow's where the user's
    // '_a' fields write the mean flow's: the normalization reads this mask, since the copy
    // above keeps only the gradient
    state.options.psStreamsTwins = userOptions.faceGradient
        ? ( ( userOptions.aField.velocity_gradient ? 1u : 0u ) | ( userOptions.aField.velocity_shear ? 2u : 0u )
            | ( userOptions.aField.velocity_vorticity ? 4u : 0u ) )
        : 0u;
#endif
    // the T-web is computed from the density grid (tidal tensor): make sure density is interpolated.
    // Drop the tweb flag itself from the internal copy -- the interpolation never fills the T-web
    // grids, so keeping it only makes reserveMemory pre-allocate them (16 B/cell) during the
    // partition loop; computeTidalWebClassification() re-creates them from the summed density.
    if ( userOptions.uField.velocity_tweb ) { state.options.uField.density = true; state.options.uField.velocity_tweb = false; }
    if ( userOptions.aField.velocity_tweb ) { state.options.aField.density = true; state.options.aField.velocity_tweb = false; }

    // restrict the data to the user-defined region (plus padding) and treat it as a standalone non-periodic box
    if ( userOptions.regionOn )
    {
        state.options.region.validSubBox( state.options.boxCoordinates, state.options.periodic );
        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );

        if ( not regionCropped )   // (the standard build selected them before the auto-tuner, above)
            findParticlesInBox( *particlePointer, &particlesRegion, state.options );
        state.options.periodic = false;
        state.options.updateFullBox( state.options.region );

        // FREE the snapshot's particles, not just clear() them: clear() keeps the whole array's capacity for the
        // rest of the run -- 36 B per particle of the whole snapshot under a triangulation of a few
        // million, which the auto-tuner (planning the region) does not count. Callers get the vector back
        // empty either way ('Clears allParticles'); main's '--interlace' keeps its own copy for the second pass.
        std::vector<Particle_data>().swap( *particlePointer );
        state.particles = std::move(particlesRegion);
        particlePointer = &state.particles;
    }

    // single-partition mode (--partition-index): keep only this partition's particles and grid sub-box
    if ( userOptions.partitionOn and userOptions.partNo>=0 )
    {
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "The program will interpolate the fields in partition number " << userOptions.partNo << " of partition grid [" << MESSAGE::printElements( userOptions.partition, "," ) << "].\n" << MESSAGE::Flush;

        std::vector< std::vector<size_t> > subgridList;
        std::vector< Box > subgridCoords;
        optimalPartitionSplit( *particlePointer, state.options, state.options.partition, &subgridList, &subgridCoords );
        copySubgridInformation( &state.options, subgridList, subgridCoords );
        userOptions.region = state.options.region;

        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );

        vector<Particle_data> particlesPartition;
        findParticlesInBox( *particlePointer, &particlesPartition, state.options );
        state.options.periodic = false;

        particlePointer->clear();
        state.particles = std::move(particlesPartition);
        particlePointer = &state.particles;

        subgrid( userOptions, subgridList );
    }

    // ensure state.particles owns the data when no region/partition move already populated it
    if ( particlePointer != &state.particles )
        state.particles = std::move(*particlePointer);

    return state;
}


// Sets up the box, padded box, and (for standard DTFE) periodic copies for the whole-box, non-partition case.
void DTFE_preparePadding(DTFE_State &state, User_options const &userOptions)
{
    if ( not userOptions.regionOn and not userOptions.partitionOn )
    {
        state.options.region = state.options.boxCoordinates;
        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );

#ifndef PHASE_SPACE
        // standard DTFE: add periodic copies based on Eulerian positions
        if ( userOptions.periodic )
        {
            vector<Particle_data> tempPart;
            findParticlesInBox( state.particles, &tempPart, state.options );
            state.particles = std::move(tempPart);
        }
#endif
        // PS-DTFE Lagrangian periodic padding is handled earlier in DTFE()
    }
#ifdef PHASE_SPACE
    // PS-DTFE keeps periodic=true only for grid-index wrapping; data periodicity comes
    // from the Lagrangian copies and Eulerian unwrapping
#else
    state.options.periodic = false;
#endif
}


// Standard binary, PARTITIONED point evaluation: the Eulerian box whose tetrahedra partition i of a
// split owns -- 'range' its grid-index range (optimalPartitionSplit's subgridList[i]) in the split's 'grid', and
// 'region' its sub-box. A face is on the run's outer boundary when its INDEX is 0 or the grid's end (the float cut
// region[0] + dx*n can miss the box face by an ulp); psPointEvalOwnBox resolves those faces (ps_point_eval.cc).
// Shared by the composite '--serve' and the batch '--sample-points --partition' loop, so both own alike. (Compiled
// in PS-DTFE too, whose build keeps the standard partition branch as unreachable code.)
static void standardPartitionOwnBox(std::vector<size_t> const &range, std::vector<size_t> const &grid,
                                    Box const &region, double ownLo[], double ownHi[])
{
    double lo[NO_DIM], hi[NO_DIM];
    bool outerLo[NO_DIM], outerHi[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        lo[d] = double( region.coords[2*d] );
        hi[d] = double( region.coords[2*d+1] );
        outerLo[d] = ( range[2*d] == 0 );
        outerHi[d] = ( range[2*d+1] >= grid[d] );
    }
    psPointEvalOwnBox( lo, hi, outerLo, outerHi, ownLo, ownHi );
}


/* DTFE_parallel's thread split, in pieces the standard partition loop's look-ahead shares: it prepares a
   partition's sub-domains EXACTLY as DTFE_parallel would (the same threads, the same split, the same per-thread
   options and particles), so these are the one definition of each. */
// the threads DTFE_parallel may split one call over: every OpenMP thread of the run mode's budget (run_mode.h), at most
// '--max-concurrent'
static int dtfeParallelThreads(User_options const &userOptions)
{
#ifdef OPEN_MP
    int n = dtfeOmpMax();
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < n )
        n = userOptions.maxConcurrent;
    return n;
#else
    (void)userOptions;
    return 1;
#endif
}
// ... and whether it runs ONE tessellation instead: 1 thread, user sampling, redshift cone, --sample-points point
// evaluation (the standard worker evaluates ONE Eulerian tessellation; overlapping padded sub-triangulations would
// double-count boundary tets), or PS-DTFE (which partitions in Lagrangian space)
static bool dtfeParallelSingle(User_options const &userOptions, vector<Sample_point> const &samples, int const threads)
{
#ifdef PHASE_SPACE
    (void)userOptions; (void)samples; (void)threads;
    return true;   // PS-DTFE: bypass spatial partitioning, use single triangulation
#else
    return threads==1 or not samples.empty() or userOptions.redshiftConeOn or psPointEvalActive();
#endif
}
// one thread's sub-domain of the split: its options (sub-grid, region, padded box), 'userOptions' being the call's
static User_options dtfeSubdomainOptions(User_options const &userOptions, int const threadNo, int const noProcessors,
                                         std::vector<size_t> const &pGrid,
                                         std::vector< std::vector<size_t> > &subgridList, std::vector< Box > &subgridCoords)
{
    User_options tempOptions = userOptions;
    tempOptions.noProcessors = noProcessors;
    tempOptions.threadId = threadNo;
    tempOptions.verboseLevel = (userOptions.verboseLevel>0) ? 1 : userOptions.verboseLevel;   // slaves show only errors/warnings; master shows all

    if ( threadNo==0 )
        tempOptions.verboseLevel = userOptions.verboseLevel;
    tempOptions.partNo = threadNo;
    tempOptions.partition.clear();
    for (int i=0; i<NO_DIM; ++i)
        tempOptions.partition.push_back( pGrid[i] );
    // the call's region becomes the sub-domain's full box: for a partition's sub-domain that is the PARTITION's region,
    // which so names the partition in the tessellation-cache key (its 'box=' line, tessellation_cache.h makeDescriptor)
    Box parentRegion = userOptions.region;   // (updateFullBox takes a non-const box)
    tempOptions.updateFullBox( parentRegion );


    // region allocated to this thread: writes subgrid size to tempOptions.gridSize and box to tempOptions.region
    copySubgridInformation( &tempOptions, subgridList, subgridCoords );
    tempOptions.paddedBox = tempOptions.region;
    tempOptions.paddedBox.addPadding( tempOptions.paddingLength );
    return tempOptions;
}

#ifndef PHASE_SPACE
/* The standard look-ahead's PREPARE stage: partition k+1's sub-domains, on the look-ahead thread while
   partition k deposits. The thread count is the CALLER's (dtfeParallelThreads on the loop's thread: a fresh thread's
   OpenMP defaults need not be the loop's), the rest is DTFE_parallel's own code above: the single-tessellation choice,
   parallelGrid, optimalPartitionSplit on the partition's particles, each sub-domain's options and particles, and
   buildTessellation (dtfePrebuildTessellation) -- with the cache when '--tessellation-cache' is on. The sub-domains are
   built on a team of their own (an OpenMP team started from this thread), as many at once as DTFE_parallel runs, and
   every one silently ('verbose 0': the log's progress line reports the look-ahead). Frees the partition's particles. */
static void dtfePrepareSubdomains(vector<Particle_data> *partParticles, vector<Sample_point> &samples,
                                  User_options const &partOptions, int const threads, DTFEPrepared &P)
{
    User_options quiet = partOptions;
    quiet.verboseLevel = 0;
    P.noTotalParticles = partParticles->size();
    P.single = dtfeParallelSingle( quiet, samples, threads );
    if ( P.single )
    {
        P.nParticles.assign( 1, partParticles->size() );
        P.tessellation.assign( 1, std::shared_ptr<void>() );
        P.tessellation[0] = dtfePrebuildTessellation( partParticles, quiet );   // frees the particles
        return;
    }
    P.pGrid.assign( NO_DIM, 0 );
    parallelGrid( threads, quiet, &(P.pGrid[0]) );
    int noProcessors = 1;
    for (int d=0; d<NO_DIM; ++d) noProcessors *= int( P.pGrid[d] );
    optimalPartitionSplit( *partParticles, quiet, P.pGrid, &P.subgridList, &P.subgridCoords );
    P.tessellation.assign( size_t(noProcessors), std::shared_ptr<void>() );
    P.nParticles.assign( size_t(noProcessors), 0 );
    std::vector<std::exception_ptr> errors( static_cast<size_t>( noProcessors ), std::exception_ptr() );
    // over the sub-domain numbers, not the thread numbers: each is built whatever team the runtime gives
#ifdef OPEN_MP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(noProcessors)
#endif
    for (int t=0; t<noProcessors; ++t)
    {
        try
        {
            User_options o = dtfeSubdomainOptions( quiet, t, noProcessors, P.pGrid, P.subgridList, P.subgridCoords );
            vector<Particle_data> particles;
            findParticlesInBox( *partParticles, &particles, o );
            P.nParticles[size_t(t)] = particles.size();
            P.tessellation[size_t(t)] = dtfePrebuildTessellation( &particles, o );
        }
        catch ( ... ) { errors[size_t(t)] = std::current_exception(); }
    }
    vector<Particle_data>().swap( *partParticles );
    for (std::exception_ptr const &e : errors)
        if ( e ) std::rethrow_exception( e );
}
#endif


#ifdef PHASE_SPACE
// PS-DTFE, periodic: unwrap the Eulerian positions so each displacement s = pos - lagPos lies in
// [-L/2, L/2]; otherwise boundary particles get displacements near L and huge spurious simplices.
static void psUnwrapEulerian(std::vector<Particle_data> &particles, User_options &options)
{
    Real eulerLen[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
        eulerLen[d] = options.boxCoordinates[2*d+1] - options.boxCoordinates[2*d];
    for (size_t i = 0; i < particles.size(); ++i)
        for (int d = 0; d < NO_DIM; ++d)
        {
            Real s = particles[i].pos[d] - particles[i].lagPos[d];
            s = std::fmod(s + eulerLen[d] * Real(0.5), eulerLen[d]);
            if (s < Real(0.)) s += eulerLen[d];
            s -= eulerLen[d] * Real(0.5);
            particles[i].pos[d] = particles[i].lagPos[d] + s;
        }
}
#endif

// The host's reaction to shares the non-finite guard of the exact fixed-point sums dropped (ps_deposit_params.h
// psFixedShareOk: a NaN, an infinity or a product beyond 2^63 -- in the run-wide 128-bit words, at or above the
// share cap, fixed_units.h RunUnits -- adds nothing and is counted, on the CPU and in every GPU kernel). THE one
// place that decides what such a run does: a warning that names each grid and its count,
// once per DTFE() call (the counts are taken and reset here), and the run goes on -- the grids are written,
// without those shares (a float sum would turn their cells NaN). To fail the run instead, throw here
// (throwError). Nothing is printed when nothing was dropped.
static void fixedShareDropsReact(User_options const &userOptions)
{
#ifdef PHASE_SPACE
    // the slots: PSFxGrid, then FX_DROP_MASS / FX_DROP_SUMW (ps_deposit_params.h)
    static char const *const names[FX_DROP_SLOTS] = {
        "velocity moment", "dispersion second moment", "velocity-gradient moment", "moment weight",
        "exact multiplicity", "dispersion mean", "dispersion weight", "scalar moment", "scalar-gradient moment",
        "released multiplicity", "mass (density)", "exact-deposit weight totals", "face velocity flux (--face-gradient)",
        "face stream flux (--face-gradient)", "face mass flux (--face-gradient)", "" };
#else
    // the standard deposits' grids (the GPU's, dtfe_gpu_fixed.h; the CPU exact average's run-wide words,
    // averaged_interpolation_1.cc Avg1Words), then the face pass's words (avg1FacePass)
    static char const *const names[FX_DROP_SLOTS] = {
        "density", "velocity", "velocity gradient", "scalar", "scalar gradient", "face velocity flux (--face-gradient)",
        "", "", "", "", "", "", "", "", "", "" };
#endif
    unsigned long long total = 0;
    std::string what;
    for (int k = 0; k < FX_DROP_SLOTS; ++k)
    {
        unsigned long long const n = fixedShareDrops[k].exchange( 0, std::memory_order_relaxed );
        if ( n == 0 ) continue;
        total += n;
        what += ( what.empty() ? "" : ", " ) + std::string( names[k][0] ? names[k] : "slot " + std::to_string( k ) )
              + " grid " + std::to_string( n );
    }
    if ( total == 0 ) return;
    MESSAGE::Warning warning( userOptions.verboseLevel );
    warning << total << " share(s) of the exact fixed-point sums were not finite or beyond their words' range and "
            << "were left out (" << what << "). The grids are written without them; a non-finite input value (a NaN "
            << "or an infinite position, velocity or mass) is the usual cause.\n" << MESSAGE::EndWarning;
}
