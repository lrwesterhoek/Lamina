// Part of src/DTFE.cpp: included once, in order, by that file and compiled only as part of its object, never alone.
// Standard DTFE's partitioned run: the composite serve of '--serve --partition' (its own guard) and
// dtfePartitionedRun.

#if NO_DIM==3 && !defined(PHASE_SPACE)
// '--serve --partition', standard DTFE: every Eulerian partition to the composite server, which serves; no return.
static void dtfeServeComposite(DTFE_State &state, User_options &userOptions, int &totalPartitions)
{
    // the load-balancing split works on particle counts per grid cell; a server has no
    // grid of its own (2^3), so the split gets a private 64^3 one
    User_options splitOpt = state.options;
    for (int d=0; d<NO_DIM; ++d)
        splitOpt.gridSize[d] = std::max<size_t>( 64, state.options.partition[d] );
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    optimalPartitionSplit( state.particles, splitOpt, state.options.partition, &subgridList, &subgridCoords );

    int concurrency = totalPartitions;
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < concurrency )
        concurrency = userOptions.maxConcurrent;
#ifdef OPEN_MP
    concurrency = std::max( 1, std::min( concurrency, dtfeOmpMax() ) );
#endif
    psServeCompositeBegin( state.options, totalPartitions );
#ifdef OPEN_MP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(concurrency)
#endif
    for (int i=0; i<totalPartitions; ++i)
    {
        User_options tempOpt = splitOpt;
        tempOpt.partNo = i;
        copySubgridInformation( &tempOpt, subgridList, subgridCoords );
        tempOpt.paddedBox = tempOpt.region;
        tempOpt.paddedBox.addPadding( tempOpt.paddingLength );
        User_options selectOpt = tempOpt;      // periodic as given: findParticlesInBox adds the images
        tempOpt.periodic = false;
        int tnum = 0;
#ifdef OPEN_MP
        tnum = omp_get_thread_num();
#endif
        tempOpt.verboseLevel = (tnum == 0) ? std::min( userOptions.verboseLevel, 1 ) : 0;

        double ownLo[NO_DIM], ownHi[NO_DIM];
        standardPartitionOwnBox( subgridList[size_t(i)], splitOpt.gridSize, tempOpt.region, ownLo, ownHi );
        psServeCompositeAddPartition( i, tempOpt,
            [&](std::vector<Particle_data> &out) { findParticlesInBox( state.particles, &out, selectOpt ); },
            ownLo, ownHi );
    }
    std::vector<Particle_data>().swap( state.particles );
    psServeCompositeRun( state.options );   // answers requests until stdin closes; never returns
}
#endif

// Standard DTFE --partition: the spatial partitions interpolated in turn (with the look-ahead) into the full grids.
static void dtfePartitionedRun(DTFE_State &state, vector<Sample_point> &samples, User_options &userOptions,
                               Quantities *uQuantities, Quantities *aQuantities)
{
    int totalPartitions = 1;
    for (int d=0; d<NO_DIM; ++d)
        totalPartitions *= state.options.partition[d];
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "The program will interpolate the fields in the region of interest using " << totalPartitions << " partitions defined via the grid [" << MESSAGE::printElements( userOptions.partition, "," ) << "].\n" << MESSAGE::Flush;

#if NO_DIM==3 && !defined(PHASE_SPACE)
    // '--serve --partition': a composite server of Eulerian partition tessellations. Each
    // partition owns the tetrahedra whose centroid lies in its (unpadded) region -- the outer
    // faces of a non-periodic box open to infinity, so hull tetrahedra keep an owner.
    if ( state.options.psServe )
        dtfeServeComposite( state, userOptions, totalPartitions );   // never returns
#endif

    double const peakBeforeGrids = runProgressPeak();
    uQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.uField );
    aQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.aField );
    runProgressNoteStartPeak( peakBeforeGrids );   // the SUMMARY's attribution: the full grids' sizing (run_progress.h)

    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    optimalPartitionSplit( state.particles, state.options, state.options.partition, &subgridList, &subgridCoords );

    /* Crash resume, as PS-DTFE's and through the same code (partition_checkpoint.h):
       after a partition is copied into the full grids they are checkpointed (unless the last write took more than
       5% of the time since it) with the list of the partitions in them, to '<output>.runlog.checkpoint' or under
       '--scratch-dir'; '--resume' loads them and computes only the partitions the list lacks. That is the whole
       state: each partition ASSIGNS its own disjoint sub-grid (copySubgridResultsToMain), the split is a function
       of the particles and the options (both in the settings key), nothing else carries from one partition to the
       next, and the T-web / V-web are made after the loop from the density / gradient grids saved here. So a
       resumed CPU run's outputs are byte-identical to an uninterrupted run's at the same thread count; '--gpu'
       ones too (its sums are exact). The
       progress counts the restored partitions as done (the sidecar's done/total and the '[partitions k/N]' line:
       N is every partition) and the ETA from this process's own rate. */
    PartitionCheckpoint::Checkpointer &checkpoint = PartitionCheckpoint::state();
    int const nRestored = PartitionCheckpoint::beginLoop( state.options, totalPartitions, *uQuantities, *aQuantities,
                                                          psPointEvalActive(), userOptions.verboseLevel );
    runProgressPhase( "partitions", nRestored, totalPartitions );
    int nComputed = 0;   // partitions computed by this process

    // '--sample-points': each partition answers the points inside
    // the tetrahedra it OWNS (Eulerian centroid in its sub-box, standardPartitionOwnBox), so every point is answered
    // once, by the tetrahedron the single tessellation would use. DTFE_parallel then builds ONE triangulation per
    // partition (no thread sub-boxes, whose padded overlaps would answer a point twice), as a single-domain run does
    if ( psPointEvalActive() )
        message << MESSAGE::cBold() << "DTFE:" << MESSAGE::cReset() << " point evaluation in " << totalPartitions
                << " partitions: each triangulates its padded sub-box once and answers the points in the "
                << "tetrahedra whose centroid lies in its sub-box.\n" << MESSAGE::Flush;

    auto const partLoopStart = std::chrono::steady_clock::now();
    // what the copies into the full grids and their checkpoints took (the 'merge time:' line after the loop; the
    // deposit stage below runs on this thread only)
    double copySec = 0., copyCheckpointSec = 0.;

    // One partition's work in two stages (as the PS loop's): PREPARE selects its particles (its padded
    // box, periodic images included) and, for the look-ahead, splits it into DTFE_parallel's thread sub-domains and
    // builds their tessellations; DEPOSIT interpolates (building the tessellations in place when nothing was
    // prepared), copies the partition's sub-grid into the full grids, checkpoints and reports. Without the
    // look-ahead each partition runs prepare + deposit in turn.
    struct StdPrepared
    {
        User_options opt;              // the partition's options (its sub-grid, region and padded box)
        vector<Particle_data> part;    // its particles (none once the look-ahead has built from them)
        bool built = false;            // the look-ahead built its sub-domains (sub)
        DTFEPrepared sub;
        double seconds = 0.;           // the look-ahead's time for it
        bool emptyPartition = false;   // (runLookAhead's: never set here, every 'prepared in' line is said)
    };
    int const parallelThreads = dtfeParallelThreads( state.options );   // DTFE_parallel's, read on THIS thread
    auto prepare = [&](int const i, bool const build, StdPrepared &P)
    {
        auto const t0 = std::chrono::steady_clock::now();
        P.opt = state.options;
        P.opt.partNo = i;
        copySubgridInformation( &P.opt, subgridList, subgridCoords );

        P.opt.paddedBox = P.opt.region;
        P.opt.paddedBox.addPadding( P.opt.paddingLength );

        findParticlesInBox( state.particles, &P.part, P.opt );
        P.opt.periodic = false;
#ifndef PHASE_SPACE
        if ( build )
        {
            dtfePrepareSubdomains( &P.part, samples, P.opt, parallelThreads, P.sub );
            P.built = true;
        }
#else
        (void)build;
#endif
        P.seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    };
    auto deposit = [&](int const i, StdPrepared &P)
    {
        message << MESSAGE::cBold() << "\n<<< Interpolating the fields for partition " << i+1 << "/" << totalPartitions << " ...\n" << MESSAGE::cReset();
        User_options &tempOpt = P.opt;

        if ( psPointEvalActive() )   // this partition's ownership (above); the loop is serial
        {
            double ownLo[NO_DIM], ownHi[NO_DIM];
            standardPartitionOwnBox( subgridList[size_t(i)], state.options.gridSize, tempOpt.region, ownLo, ownHi );
            psPointEvalSetOwnership( ownLo, ownHi );
        }
        Quantities temp_uQuantities, temp_aQuantities;
        // '--sample-points': tag this partition's point records, as the PS loop does, so the checkpoint
        // saves them per partition ('points.<i>', partition_checkpoint.h) and '--resume' reads them back. The tag is
        // thread-local; under point evaluation DTFE_parallel triangulates the partition once ON THIS THREAD (its
        // single-triangulation branch), and the point walk copies the tag into its sinks before it starts its own
        // threads (ps_point_eval.cc forEachCellParallel). A no-op without '--sample-points'. (The look-ahead only
        // BUILDS on its own thread; the interpolation, and so the walk, stays on this one.)
        psPointEvalSetPartition( i );
        DTFE_parallelImpl( &P.part, samples, tempOpt, &temp_uQuantities, &temp_aQuantities, P.built ? &P.sub : nullptr );
        psPointEvalSetPartition( -1 );
#ifndef PHASE_SPACE
        // this partition's tessellations are done with: released, and the TBB allocator's kept blocks handed back, so
        // the next partition starts from the live memory the auto-tuner books (triangulation.cpp)
        P.sub = DTFEPrepared();
        dtfeReleaseTessellationMemory();
#endif

        // map this partition's sub-grid results back into the full output grids
        auto const tCopy = std::chrono::steady_clock::now();
        copySubgridResultsToMain( temp_uQuantities, state.options.gridSize, tempOpt.uField, tempOpt, subgridList, uQuantities );
        copySubgridResultsToMain( temp_aQuantities, state.options.gridSize, tempOpt.aField, tempOpt, subgridList, aQuantities );
        auto const tCheckpoint = std::chrono::steady_clock::now();
        // the checkpoint (partition_checkpoint.h): the full grids now hold this partition too
        PartitionCheckpoint::afterMerge( i, false, *uQuantities, *aQuantities, userOptions.verboseLevel );
        copySec += std::chrono::duration<double>( tCheckpoint - tCopy ).count();
        copyCheckpointSec += std::chrono::duration<double>( std::chrono::steady_clock::now() - tCheckpoint ).count();

        // progress + ETA over the serial partition loop
        double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - partLoopStart ).count();
        ++nComputed;
        int    const done    = nRestored + nComputed;
        // (the split balances the particle counts, so counting partitions is the cost weighting here; the rate
        // is this process's own, the restored partitions cost it nothing)
        double const eta     = elapsed * ( double(totalPartitions - done) / double(nComputed) );
        runProgressPhase( "partitions", done, totalPartitions, eta );
        partitionProgress( userOptions.verboseLevel, done, totalPartitions, int(100.*done/double(totalPartitions) + 0.5),
                           elapsed, eta );
    };

    // the partitions this process computes, in index order (a resumed run's restored ones are in the grids already)
    std::vector<int> toCompute;
    for (int i=0; i<totalPartitions; ++i)
        if ( not checkpoint.merged[size_t(i)] ) toCompute.push_back( i );

    /* The LOOK-AHEAD (the PS loop's, dtfe/ps_partition_loop.cc, for the standard loop). Without it, with '--gpu'
       every partition's volume-averaged pass waits for the partition's particle selection and the Delaunay
       triangulations + vertex densities of all its thread sub-boxes, while the GPU idles. So partition k+1 is
       prepared -- selected, split into DTFE_parallel's sub-domains and triangulated (or loaded from
       '--tessellation-cache'), on a team of CPU threads of its own -- while partition k is interpolated, at most
       one ahead. Every sub-domain is built from the same particles with the same options as without the
       look-ahead (dtfePrepareSubdomains: DTFE_parallel's own split and options, the sequential insert), the
       interpolation is unchanged, and each partition still ASSIGNS its own part of the grids in index order, so
       every output is byte-identical (tests/dtfe_resume_check.sh D14). It holds the next partition's particle copy
       and tessellations; before each look-ahead the free memory, less 4 GB of headroom, is checked against that
       (the auto-tuner's booked figure: its build of the partition that needs the most, counted from the particles;
       without a plan auto_tune.h stdLookAheadBytes, the uniform count of one partition), and the partition is
       prepared in turn when it does not fit; the
       auto-tuner counts it in its prediction and switches it off for a run whose budget cannot hold it
       (stdLookAheadOffBudget). '--no-overlap' switches it off; DTFE_OVERLAP=1 forces it on a CPU run too (the
       tests' byte-identity check on any build), DTFE_OVERLAP=0 switches it off. A '--resume' looks ahead to the
       next partition it computes, not to the next index. */
    bool gpuDeposit = false;
#if defined(DTFE_GPU) && NO_DIM==3 && !defined(PHASE_SPACE)
    gpuDeposit = state.options.useMetal and state.options.method==1 and state.options.aField.selected()
                 and not state.options.redshiftConeOn and not state.options.userDefinedSampling;
#endif
    bool overlap = false;
    double aheadBytesMax = 0.;
    std::function<double(int)> aheadBytes = [](int) { return 0.; };
#ifndef PHASE_SPACE
    {
        int const overlapEnv = Env::tristate( "DTFE_OVERLAP" );   // 1 forces the look-ahead, 0 bans it
        bool const overlapForced = overlapEnv == 1;
        bool const overlapBanned = overlapEnv == 0;
        overlap = toCompute.size() > 1 and state.options.DTFE and not state.options.stdNoOverlap
                  and not state.options.stdLookAheadOffBudget and not overlapBanned and ( gpuDeposit or overlapForced );
    }
    // what one partition's look-ahead holds: the builds of its sub-domains and a team's padded particle copy
    // (auto_tune.h, the model's items). Where the auto-tuner planned, its booked figure: the partition that needs the
    // most, counted from the particles (auto_tune.cc reportSplit). Else the uniform count of the partition's geometry
    // (the split balances the particles), which misjudges a clustered run both ways: a dense partition's padding
    // shells hold more than it, and a small partition's box is padded far beyond its particles' share (TNG50-3 at
    // 512^3 in 4^3 partitions: 20.86 GB against the 9.36 GB the auto-tuner counted).
    aheadBytes = [&](int const i) -> double
    {
        if ( state.options.stdLookAheadCounted > 0. )
            return state.options.stdLookAheadCounted;
        User_options o = state.options;
        o.partNo = i;
        copySubgridInformation( &o, subgridList, subgridCoords );
        o.verboseLevel = 0;
        size_t tGrid[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) tGrid[d] = 1;
        if ( not dtfeParallelSingle( o, samples, parallelThreads ) )
            parallelGrid( parallelThreads, o, tGrid );
        double len[NO_DIM], pad[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            len[d] = double( o.region[2*d+1] - o.region[2*d] );
            pad[d] = double( o.paddingLength[2*d] + o.paddingLength[2*d+1] ) / 2.;
        }
        return stdLookAheadBytes( state.options, double( state.particles.size() ) / double( totalPartitions ),
                                  len, pad, tGrid );
    };
    if ( overlap )
    {
        for (int i : toCompute) aheadBytesMax = std::max( aheadBytesMax, aheadBytes( i ) );
        // the memory budget too, not only the free memory: what it leaves beside the grids, the particles and one
        // partition must hold the look-ahead, or the run has none (the same for every partition, decided once)
        if ( aheadBytesMax > state.options.stdLookAheadRoom )
        {
            overlap = false;
            message << MESSAGE::cBold() << "DTFE:" << MESSAGE::cReset() << " look-ahead off: the ~"
                    << std::round( aheadBytesMax/1.e7 )/100. << " GB it holds is more than the ~"
                    << std::round( std::max( 0., state.options.stdLookAheadRoom )/1.e7 )/100. << " GB the memory "
                    << "budget leaves beside one partition; partitions are prepared in turn.\n" << MESSAGE::Flush;
        }
    }
#endif
    if ( overlap )
    {
        message << MESSAGE::cBold() << "DTFE:" << MESSAGE::cReset() << " look-ahead on: partition k+1 is "
                << "triangulated on CPU threads while partition k is interpolated"
                << ( gpuDeposit ? " on the GPU" : " (DTFE_OVERLAP=1)" ) << " (~" << std::round( aheadBytesMax/1.e7 )/100.
                << " GB more in memory; '--no-overlap' switches it off).\n" << MESSAGE::Flush;

        // the partitions this process computes, each next one prepared ahead when the memory holds its own count;
        // the line names it by its 1-based number ('during this one')
        runLookAhead<NoLookAheadScope, StdPrepared>( toCompute, aheadBytes, prepare, deposit,
                                                    [](int) { return true; }, LookAheadLabel{ 1, "this one" },
                                                    userOptions.verboseLevel );
    }
    else
        for (int i : toCompute)
        {
            StdPrepared P;
            prepare( i, false, P );
            deposit( i, P );
        }
    PartitionCheckpoint::reportWrites( std::chrono::duration<double>( std::chrono::steady_clock::now() - partLoopStart ).count(),
                                       userOptions.verboseLevel );
    {   // the serial merge: each partition assigns its own sub-grid, in index order (the PS loop's line has its waits)
        char t[160];
        std::snprintf( t, sizeof(t), " partition(s) copied into the full grids in %.2f s, their checkpoints %.2f s "
                       "(one at a time, in index order)", copySec, copyCheckpointSec );
        message << MESSAGE::cDim() << "  merge time: " << nComputed << t << "\n" << MESSAGE::cReset() << MESSAGE::Flush;
    }

    // each partition copied its own particles; the global array is dead weight from here on
    std::vector<Particle_data>().swap( state.particles );
    if ( psPointEvalActive() )
        psPointEvalSetOwnership( nullptr, nullptr );   // back to the single-triangulation default
}
