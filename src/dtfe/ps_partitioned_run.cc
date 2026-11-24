// Part of src/DTFE.cpp, inside its #ifdef PHASE_SPACE: included once, in order, by that file and compiled only as
// part of its object, never alone.
// The PS partitioned run: the close-out of the summed moments (psCloseOutMoments, psNormalizeMoments), the
// aggregate report (psAggregateReport) and psPartitionedRun.

// THE CLOSE-OUT of the summed moments, once over all partitions. Returns the cell volume.
static Real psCloseOutMoments(DTFE_State &state, Quantities *uQuantities, Quantities *aQuantities)
{
    // fields were summed as weighted moments; normalize once now so cells drawing
    // streams from several partitions (multi-stream regions) come out correct. The scale turns the summed
    // rho/rho_bar back into the per-cell mass: the weight normalizePhaseSpace takes where no mass_weight was handed.
    Real cellVolume = Real(1.);
    for (int d = 0; d < NO_DIM; ++d)
        cellVolume *= (state.options.region[2*d+1] - state.options.region[2*d]) / Real(state.options.gridSize[d]);
    Real const weightScale = cellVolume * state.options.averageDensity;
    // --face-gradient: the spacing the merged face sums' gradient divides by
    for (int d = 0; d < NO_DIM; ++d)
        aQuantities->psFaceDx[d] = uQuantities->psFaceDx[d] =
            (state.options.region[2*d+1] - state.options.region[2*d]) / Real(state.options.gridSize[d]);
    uQuantities->normalizePhaseSpace( state.options.uField, weightScale );
    // (the regular part's twins: the '_a' grids only, as the writer writes them -- DTFE_setup's psStreamsTwins)
    aQuantities->normalizePhaseSpace( state.options.aField, weightScale, state.options.psStreamsTwins );
    return cellVolume;
}

// Normalizes the summed moments once, over all partitions; DTFE_MASS_AUDIT=1 reports the run's mass closure.
static void psNormalizeMoments(DTFE_State &state, Quantities *uQuantities, Quantities *aQuantities,
                               double &auditInputMass)
{
    Real const cellVolume = psCloseOutMoments( state, uQuantities, aQuantities );

    // DTFE_MASS_AUDIT=1, (h): the merged grid against the passes' sums and the input mass, per pass kind
    if ( PSMassAudit::on() )
        for (int k = 0; k < 2; ++k)
        {
            PSMassAudit::Totals const &T = PSMassAudit::totals[k];
            std::vector<Real> const &den = ( k ? aQuantities : uQuantities )->density;
            if ( T.passes == 0 or den.empty() ) continue;
            double sum = 0.;
            for (Real const v : den) sum += double(v);
            double const merged = sum * double(cellVolume) * double(state.options.averageDensity);
            auto rel = [](double x, double ref) { return ref != 0. ? x / ref - 1. : 0.; };
            std::fprintf( stderr, "PS mass audit: run (%s pass, %zu partition passes): input %.12g, sum (a) owned %.12g, "
                          "sum (b) tets %.12g, sum (c) deg0 %.12g, sum (d) dropped %.12g, sum (g) grids %.12g (post %.12g), "
                          "merged %.12g, grid mean - 1 = %.6e | owned/input-1 %.3e, (tets+deg0)/input-1 %.3e, "
                          "grids/tets-1 %.3e, merged/grids(post)-1 %.3e, merged/input-1 %.3e\n",
                          k ? "averaged" : "unaveraged", T.passes, auditInputMass, T.owned, T.tets, T.deg0, T.dropped,
                          T.gridPre, T.gridPost, merged, sum / double(den.size()) - 1.,
                          rel(T.owned, auditInputMass), rel(T.tets + T.deg0, auditInputMass), rel(T.gridPre, T.tets),
                          rel(merged, T.gridPost), rel(merged, auditInputMass) );
        }
}

// Reports the coverage and multi-stream statistics of the summed grid over all partitions.
static void psAggregateReport(Quantities *uQuantities, Quantities *aQuantities, int &totalPartitions,
                              MESSAGE::Message &message)
{
    // aggregate coverage / stream statistics over all partitions from the summed grid
    // (per-partition figures were suppressed as misleading); prefer the unaveraged
    // stream_count, falling back to the averaged one if only '_a' fields exist
    std::vector<Real> const *sc = nullptr;
    if      ( not uQuantities->stream_count.empty() ) sc = &uQuantities->stream_count;
    else if ( not aQuantities->stream_count.empty() ) sc = &aQuantities->stream_count;
    if ( sc != nullptr )
    {
        size_t const ncell = sc->size();
        size_t covered = 0, multi = 0;
        Real maxStreams = Real(0.);
        for (size_t i = 0; i < ncell; ++i)
        {
            Real const v = (*sc)[i];
            if (v > Real(0.)) ++covered;
            // multiplicity is a float sum under every deposit -- single-stream cells land
            // on 1.0 +/- eps, so a bare 'v > 1' counts float noise as multi-stream.
            // See PS_STREAM_TOL in quantities.h.
            if (v > Real(1.) + PS_STREAM_TOL) ++multi;
            if (v > maxStreams) maxStreams = v;
        }
        message << "\n" << MESSAGE::cCyan() << "PS-DTFE: aggregate over " << totalPartitions << " partitions"
                << " -- max streams " << maxStreams
                << ", multi-stream grid points " << multi
                << " (" << (100.*multi/ncell) << "\%), grid coverage "
                << covered << "/" << ncell
                << " (" << (100.*covered/ncell) << "\%)" << MESSAGE::cReset()
                << ( covered < ncell ?
                     "  -- uncovered cells may indicate insufficient padding, a grid finer than the tessellation, or legitimate empty/expanding regions" : "" )
                << "\n" << MESSAGE::Flush;
    }
}

// PS-DTFE --partition: the Lagrangian partitions deposited, merged in order, normalized, reported.
static void psPartitionedRun(DTFE_State &state, vector<Sample_point> &samples, User_options &userOptions,
                             Quantities *uQuantities, Quantities *aQuantities, Box &lagBoxGlobal,
                             bool &hasLagrangianPeriodicCopies)
{
    int totalPartitions = 1;
    for (int d=0; d<NO_DIM; ++d)
        totalPartitions *= state.options.partition[d];
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
            << " Lagrangian-space partitioning with " << MESSAGE::cMagenta() << totalPartitions
            << " partitions [" << MESSAGE::printElements( userOptions.partition, "," ) << "]"
            << MESSAGE::cReset() << ".\n" << MESSAGE::Flush;

    // the partition geometry (bounding box, lengths, padding), shared with the composite server
    PSPartitionPlan plan;
    plan.init( state, lagBoxGlobal, hasLagrangianPeriodicCopies, message );

    // DTFE_MASS_AUDIT=1: the input mass (the global array holds the originals only here; it is freed
    // before the merge is normalized) and fresh run totals (ps_mass_audit.h)
    double auditInputMass = 0.;
    if ( PSMassAudit::on() )
    {
        for (Particle_data &p : state.particles) auditInputMass += double( p.weight() );
        std::lock_guard<std::mutex> lock( PSMassAudit::mutex );
        PSMassAudit::totals[0] = PSMassAudit::totals[1] = PSMassAudit::Totals();
    }

    // Eulerian region/padding for the full box (needed by DTFE_interpolation; also part of every
    // partition's tessellation-cache identity, so it is set before the composite server too)
    if ( not userOptions.regionOn and not (userOptions.partitionOn and userOptions.partNo>=0) )
    {
        state.options.region = state.options.boxCoordinates;
        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );
    }

    // a partition's options: its ownership region and the partition-path switches. They are part of
    // its tessellation-cache identity, so the composite server, the --grid-window skip and the batch
    // loop must form them identically -- hence one definition
    auto partitionOptions = [&](Box const &lagRegion) -> User_options
    {
        User_options o = state.options;
        o.lagrangianRegion = lagRegion;    // for cell ownership check
        o.psSuppressGridStats = true;      // per-partition stats misleading; aggregate printed after the loop
        o.psDeferNormalization = true;     // keep fields as moments; normalize once after summing
        o.psUseSubgrid = true;             // store only this partition's Eulerian bounding box
        return o;
    };

    // '--serve --partition': a composite server of the partition tessellations instead of grids
    if ( state.options.psServe )
        psServeComposite( state, userOptions, plan, partitionOptions );   // never returns

    double const peakBeforeGrids = runProgressPeak();
    psReserveGrids( state, uQuantities, aQuantities );
    runProgressNoteStartPeak( peakBeforeGrids );   // the SUMMARY's attribution: the full grids' sizing (run_progress.h)

    // periodic=true here only drives grid-level index wrapping; data periodicity comes
    // from the Lagrangian copies and Eulerian unwrapping

    // each thread builds, interpolates, and accumulates its own triangulation into the
    // shared grids inside a critical section; peak memory ~ (concurrent partitions) x full grid

    // cap concurrent partition triangulations (the dominant memory cost); --max-concurrent trades cores for RAM, 0 = all threads
    int psConcurrency = totalPartitions;
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < psConcurrency )
        psConcurrency = userOptions.maxConcurrent;
#ifdef OPEN_MP
    // never more than the threads, like the serve paths and DTFE_parallel: '--max-concurrent 0' with a
    // hand-set '--partition 5 5 5' would start all 125 partitions at once, with no memory gate
    psConcurrency = std::min( psConcurrency, dtfeOmpMax() );
#endif
    if ( psConcurrency < 1 ) psConcurrency = 1;
    /* DYNAMIC CONCURRENCY. The plan fixes the split and what one partition adds, but its concurrency is
       a prediction, and a cautious one can leave most of the budget unused. When the tuner chose
       the concurrency, the loop starts up to one thread per core instead, and the memory gate (auto_tune.h) admits a
       partition beyond the plan's mc only while the live free memory, less the headroom and what the run has still to
       allocate, holds what it adds, and the run's own footprint plus that stays within the budget. The ordered merge
       sums in index order, so the outputs do not depend on it -- in the single build, whose deposit sums are integer
       words whatever the team size; the double build sums per-thread copies in thread order, so its grids would
       follow the in-flight count, and so would DTFE_MASS_AUDIT's per-thread tallies: they keep the plan's.
       Not for a user's --max-concurrent, the speed split (no gate), or a start whose free memory (records aside: the
       gate books their unmerged share as it goes) holds no partition beyond the plan's (that run keeps its look-ahead),
       and never past the lane share's cap (psInFlightCap: the memory model was measured up to there). */
    int const psFloor = psConcurrency;   // the plan's mc
    bool dynamicConc = false;
    char const *dynamicWhyNot = nullptr;
    double dynStartFree = -1., dynStartOwn = -1.;
#ifdef OPEN_MP
    if ( userOptions.psDynamicConcurrency and state.options.psPartitionBytes > 0. and not state.options.psSpeedSplit )
    {
        bool const orderFree = std::is_same<Real, float>::value and not Env::flag( "DTFE_MASS_AUDIT" );
        int const capShare = state.options.psInFlightCap;   // the lane share's limit (0: none)
        int const most = std::min( totalPartitions, capShare > 0 ? std::min( capShare, dtfeOmpMax() ) : dtfeOmpMax() );
        // one more than the plan's: each its own lane, and what one partition at a time holds once (psOnceBytes: the GPU
        // snapshot and the lane share's laneOnce; 0 on the CPU, where every partition books psPartitionBytes in full)
        double const once = std::min( state.options.psOnceBytes, state.options.psPartitionBytes );
        double const need = double( psFloor + 1 ) * ( state.options.psPartitionBytes - once ) + once;
        double const reserve = state.options.psReserveEnd + state.options.psReserveFirstMerge;
        dynStartFree = autoTuneSystemFree();
        dynStartOwn = autoTuneOwnFootprint();
        bool const room = dynStartFree >= 0. and dynStartOwn >= 0.
                          and dynStartFree - kAutoTuneHeadroom - reserve >= need
                          and ( state.options.psBudgetBytes <= 0.
                                or dynStartOwn + need + reserve <= state.options.psBudgetBytes );
        if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // the start check's terms (tests/run_mode_check.sh reads them)
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS dynamic start: partition %.0f B, once %.0f B, reserve %.0f B, "
                          "free %.3f GB, own %.3f GB, budget %.3f GB\n", state.options.psPartitionBytes, once, reserve,
                          dynStartFree/1.e9, dynStartOwn/1.e9, state.options.psBudgetBytes/1.e9 );
        if ( not orderFree ) dynamicWhyNot = "the double build's sums or DTFE_MASS_AUDIT's per-thread tallies would follow the count";
        else if ( most <= psFloor )
            dynamicWhyNot = ( capShare > 0 and capShare < std::min( totalPartitions, dtfeOmpMax() ) )
                            ? "the memory model's lane share holds for no more partitions in flight than the plan's"
                            : "the plan already runs one partition per core or partition";
        else if ( not room ) dynamicWhyNot = "the free memory or the budget at the start holds no partition beyond it";
        else
        {
            dynamicConc = true;
            psConcurrency = most;
        }
    }
#endif
    message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " building up to "
            << MESSAGE::cMagenta() << psConcurrency << MESSAGE::cReset() << " partition triangulation(s) concurrently";
    if ( dynamicConc )
        message << " (dynamic: at least the plan's " << psFloor << ", more while the free memory and the budget hold "
                << "them; '--max-concurrent' sets a fixed number)";
    else if ( dynamicWhyNot != nullptr )
        message << " (the plan's; dynamic concurrency not started: " << dynamicWhyNot << ")";
    else if ( userOptions.maxConcurrent > 0 and not userOptions.psDynamicConcurrency )
        message << " (--max-concurrent)";
    message << ".\n" << MESSAGE::Flush;

    // --grid-window: a partition whose occupancy map (beside its cached tessellation: written by a
    // composite server or an earlier batch run) has no bucket in the window deposits nothing
    // there -- skip it whole: no particle selection, no tessellation load or build. A partition
    // without a usable map is processed (and leaves its map for the next run).
    std::vector<char> skipPartition( size_t(totalPartitions), 0 );
    int nSkipped = 0;
    psWindowSkips( state, plan, partitionOptions, totalPartitions, message, skipPartition, nSkipped );
    psReadDeferredParticles( state, userOptions, plan, partitionOptions, totalPartitions, message, skipPartition );

    // The checkpoint of the merged grids and '--resume' (partition_checkpoint.h beginLoop / afterMerge / reportWrites,
    // which the standard loop of dtfe/std_partitioned_run.cc runs too). On resume the checkpoint's grids are loaded
    // into the pre-sized accumulators and its merged partitions are skipped like a --grid-window skip (no particles, no
    // tessellation, still a turn in the ordered merge).
    PartitionCheckpoint::Checkpointer &checkpoint = PartitionCheckpoint::state();
    PartitionCheckpoint::beginLoop( state.options, totalPartitions, *uQuantities, *aQuantities, psPointEvalActive(),
                                    userOptions.verboseLevel );
    if ( checkpoint.resumed )
        for (int pi=0; pi<totalPartitions; ++pi)
            if ( checkpoint.merged[size_t(pi)] and not skipPartition[size_t(pi)] ) { skipPartition[size_t(pi)] = 1; ++nSkipped; }
    // progress + ETA; updated only under mergeMutex (dtfe/ps_partition_loop.cc) so it is race-free. The ETA weights
    // each partition by a predicted cost instead of counting partitions: the deposit time per partition is heavy-tailed
    // and the costly cluster partitions sit together in index order, so elapsed x remaining/done can be far off.
    PSProgress progress;
    progress.toProcess = std::max( 1, totalPartitions - nSkipped );
    progress.loopStart = std::chrono::steady_clock::now();
    progress.partCost = psPartitionCosts( state.particles, plan, state.options );
    for (int pi=0; pi<totalPartitions; ++pi)
        if ( not skipPartition[size_t(pi)] ) progress.costTotal += progress.partCost[size_t(pi)];
    runProgressPhase( "partitions", 0, progress.toProcess );
    // THE ORDERED MERGE (the default of every split; the auto-tuner's speed split always): the full grids receive
    // partition 0, then 1, 2 ..., exactly the adds of a '--max-concurrent 1' run, so the merged grids are
    // byte-identical to those. '--ps-merge completion-order' adds partitions as they finish instead, so the last bits
    // vary run to run (in the double build and for the GPU mass grid too; tests/gpu_mass_check.sh pins
    // --max-concurrent 1).
    // STAGING: a partition that finishes before its turn moves its two sub-grid Quantities into 'staged' (no copy;
    // its particle copy is freed) and its thread takes the next one instead of idling behind a slow partition.
    // Whoever merges the partition whose turn it is merges every staged one after it, in index order
    // under the same lock, writing each checkpoint (afterMerge) in that order: the checkpoint, its 'points.<i>'
    // files and the test hook see an unstaged merge's sequence. Under the MEMORY GATE (an auto-tuned plan with
    // partitions in flight: psPartitionBytes > 0, not the speed split) a finished partition always parks, since
    // the gate admits by the free memory, parked bytes included; without it, only while the free memory less 4 GB
    // holds it, else it waits (counted on the 'ordered merge:' line). DTFE_MERGE_STAGE_GB=X caps the parked bytes
    // at X GB instead (0 = never park). No deadlock: the partition whose turn it is never waits or parks, each
    // partition merges once, the dynamic schedule hands them out in index order, and the gate waits only behind
    // lower-numbered partitions in flight (parked ones have left it).
    // THE SPILL: the gate must admit a partition with no lower one in flight whatever the free memory, so parking
    // only in RAM could exceed the machine. A partition that must park goes
    //   1. to RAM while the free memory less 4 GB (under the gate also less one psPartitionBytes) holds it
    //      (DTFE_MERGE_STAGE_GB=X: while the parked RAM stays under X GB); the free-memory reading already
    //      includes the sub-grids, a margin of one sub-grid kept on purpose;
    //   2. else to DISK (ps_merge_spill.h): one unlinked file in --scratch-dir (SCRATCH_DIR), else $TMPDIR, else
    //      /tmp, while that volume keeps 2 GB free (scratch_alloc.cc's margin) beside the spills being written and
    //      the --scratch-dir mappings' full size; at its turn it is read back into the same Quantities and merged
    //      by the same mergeOne (byte-identical to an in-RAM park). Written outside the merge lock; a partition
    //      whose turn comes during its write merges from RAM and drops the file;
    //   3. else, under the gate without a DTFE_MERGE_STAGE_GB cap, to RAM anyway (the same memory as waiting);
    //      otherwise it waits for its turn holding its thread.
    // DTFE_MERGE_SPILL=0 switches the spill off (the gate then parks in RAM always), =1 spills EVERY parked
    // partition with sub-grids (the tests' hook); DTFE_MERGE_STAGE_GB=0 parks nothing, RAM or disk. The 'ordered
    // merge:' line counts the spills (number, most at once, GB written). A spilled partition has left the gate
    // like a parked one; the tuner's model does not count parked sub-grids (the gate keeps them honest).
    // (the state of all this: PSMergeStaging in dtfe/ps_partition_loop.cc, with parkWhere)
    PSMergeStaging merge( state.options, psConcurrency );
    if ( dynamicConc ) merge.parkFloor = psFloor;   // (parkWhere: the partitions beyond the plan's shrink its RAM room)
    // test hook: DTFE_TEST_DELAY_PARTITION=i:s holds partition i for s seconds before its merge, so the partitions
    // after it finish first and the staging is certain to happen (tests/ps_runtime_check.sh R7)
    int delayPartition = -1;
    double delaySeconds = 0.;
    if ( char const *hook = Env::value( "DTFE_TEST_DELAY_PARTITION" ) )
        if ( std::sscanf( hook, "%d:%lf", &delayPartition, &delaySeconds ) != 2 ) delayPartition = -1;
    PSLoopCtx loop{ state, userOptions, uQuantities, aQuantities, message, totalPartitions, plan, psConcurrency,
                    skipPartition, progress, merge };
    psReportMergeMode( loop );

    // (dynamic concurrency: the core pool and the teams count the partitions in flight, dtfePartitionsInFlight, which
    // the gate keeps; this plan stays the loop's threads, the parallel insert's share)
    ConcurrentTriangulationPlan triangulationPlan( psConcurrency, totalPartitions );
    psGpuFallbacksTaken( true );   // counted from here; reported once after the loop
    {   // ... and the face walks beside the GPU deposit (the default, ps_interpolation.cc PSEarlyFaceWalk)
        long long kept, discarded;
        double walkSec, waitSec;
        psFaceWalksEarlyTaken( kept, discarded, walkSec, waitSec, true );
    }
    // a partition starts only when the machine has the memory for it (or nothing else is in flight);
    // under the ordered merge it waits only behind lower-numbered partitions (auto_tune.h)
    // (in completion order orderedMerge is false -- no partition waits for its turn, so the gate
    // needs no ordered mode: it waits while any partition is in flight, which the dynamic schedule makes the same
    // set, every one in flight being lower-numbered)
    PartitionMemoryGate memoryGate( psConcurrency > 1 ? state.options.psPartitionBytes : 0., merge.orderedMerge );
    if ( dynamicConc )
    {
        memoryGate.dynamic = true;
        memoryGate.floor = psFloor;
        memoryGate.once = std::min( state.options.psOnceBytes, state.options.psPartitionBytes );
        memoryGate.total = std::max( 1, progress.toProcess );
        memoryGate.reserveEnd = state.options.psReserveEnd;
        memoryGate.reserveRecords = state.options.psReserveRecords;
        memoryGate.reserveFirstMerge = state.options.psReserveFirstMerge;
        memoryGate.budget = state.options.psBudgetBytes;
        memoryGate.startFree = dynStartFree;
        memoryGate.startOwn = dynStartOwn;
        memoryGate.footprint = &autoTuneOwnFootprint;
        memoryGate.live = &dtfePartitionsInFlight;
        dtfePartitionsInFlight.store( 1, std::memory_order_relaxed );
    }

    // One partition's work, in two stages: PREPARE (select its particles; with 'build', also its tessellation and
    // vertex densities) and DEPOSIT (the grid passes, then the merge). The look-ahead (psRunPartitionLoop,
    // dtfe/ps_partition_loop.cc) runs the prepare stage of partition k+1 on another thread while partition k deposits.
    // THE GPU SHARE ('--gpu-share', resolved before the plan: psResolveGpuShare) = the per mille of a GPU run's
    // exact deposits that run on the GPU; 1000 is all of them ('auto', the default, outside performance). The other
    // partitions deposit on the CPU at the same time, so that both devices work at once: a GPU run keeps ~2 cores busy
    // beside a GPU that every partition queues for, and a CPU run leaves the GPU idle. Which partition goes where is a
    // rule of its index alone (the GPU's share spread evenly over the indices), never of the timing, the concurrency or
    // the window and resume skips, so a rerun or a resume deposits every partition on the same device. A CPU partition
    // takes the path of the GPU fallback (ps_interpolation.cc: the same fixed-point units as a GPU deposit of the pass,
    // so the merge sees no difference) without counting as one; its grids differ from a GPU deposit's at the float
    // rounding of the two exact integrations, so a mixed run equals neither the GPU run nor the CPU run to the last
    // bit. The resolved share below 1000 is in the checkpoint key (partition_checkpoint.h settingsKey): a resume with
    // another is refused.
    long const psGpuShare = state.options.psGpuShareMille;
    auto depositsOnCpu = [psGpuShare](int const pi) -> bool
    { return ( long(pi) + 1 ) * psGpuShare / 1000 == long(pi) * psGpuShare / 1000; };
#if defined(PS_GPU) && NO_DIM==3   // (a build without the GPU deposit has no partition on the GPU to report)
    if ( state.options.psUseMetal and state.options.psExactDeposit )
    {
        if ( psGpuShare < 1000 )
        {
            int onGpu = 0, toDeposit = 0;   // (outside the grid window or restored from a checkpoint = no deposit)
            for (int pi = 0; pi < totalPartitions; ++pi)
            {
                if ( skipPartition[size_t(pi)] ) continue;
                ++toDeposit;
                onGpu += depositsOnCpu( pi ) ? 0 : 1;
            }
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " GPU share "
                    << psGpuShareFraction( psGpuShare ) << " (" << state.options.psGpuShareSource << "): " << onGpu
                    << " of the " << toDeposit << " partitions to deposit do so on the GPU, the other "
                    << toDeposit - onGpu << " on the CPU with r3d clipping (a mixed run's grids differ from an all-GPU "
                    << "run's at float rounding).\n" << MESSAGE::Flush;
        }
        else
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " GPU share 1 ("
                    << state.options.psGpuShareSource << "): every partition deposits on the GPU.\n" << MESSAGE::Flush;
    }
#endif

    auto prepare = [&](int const pi, bool const build, PSPrepared &P)
    {
        auto const t0 = std::chrono::steady_clock::now();
        // unpadded Lagrangian region for this partition (ownership) and the padded selection box
        Box lagRegion, lagPadded;
        plan.regions( pi, lagRegion, lagPadded );

        // select particles by Lagrangian position; when periodic, this partition's images
        // are generated on the fly so the global array stays originals-only
        P.skipped = skipPartition[size_t(pi)] != 0;   // --grid-window: never reaches the window
        // a partition whose cached tessellation matches its descriptor is loaded, not built: its
        // particles are never needed, so their selection (a scan of every particle, plus a copy of the
        // padded share) is skipped -- the load must then succeed (psCacheOnly)
        P.opt = partitionOptions( lagRegion );
        if ( P.opt.psUseMetal and depositsOnCpu( pi ) ) P.opt.psUseMetal = false;   // (--gpu-share, above)
        P.cacheOnly = not P.skipped and not state.options.tessellationCacheDir.empty()
                      and psTessellationCached( P.opt );
        if ( not P.skipped and not P.cacheOnly and state.options.psParticlesDeferred )
            throwError( "internal: a partition needs its particles, but they were not read (the globals record path)." );
        if ( not P.skipped and not P.cacheOnly )
            findParticlesInBoxLagrangianPeriodic( state.particles, &P.part, lagPadded,
                                                  plan.eulerLen, plan.periodicCopies, 0, plan.selectionExtent() );
        // an empty (or skipped) partition builds nothing but still takes its turn in the ordered merge
        P.emptyPartition = P.skipped or ( not P.cacheOnly and P.part.empty() );
        P.opt.psCacheOnly = P.cacheOnly;
        if ( build and not P.emptyPartition )
        {
            P.opt.verboseLevel = 0;   // the look-ahead thread is silent; the progress line reports it
            P.pre = psPrebuildTessellation( &P.part, P.opt );
        }
        P.seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    };
    auto deposit = [&](int const pi, PSPrepared &P, int const verbose)
    {
        Quantities temp_uQuantities, temp_aQuantities;
        if ( not P.emptyPartition )
        {
            // per-partition options use the full Eulerian grid; only the master thread logs
            P.opt.verboseLevel = verbose;
            PSMassAudit::partition = pi;   // names the audit lines (this thread runs the passes)
            psPointEvalSetPartition( pi ); // tags the '--sample-points' records (the checkpoint saves them per partition)
            if ( P.pre )
                psInterpolatePrebuilt( P.pre, samples, P.opt, &temp_uQuantities, &temp_aQuantities );
            else   // PS-DTFE: DTFE_parallel calls serial interpolation directly (no nested OpenMP region)
                DTFE_parallel( &P.part, samples, P.opt, &temp_uQuantities, &temp_aQuantities );
            PSMassAudit::partition = -1;
            psPointEvalSetPartition( -1 );
        }

        if ( pi == delayPartition and delaySeconds > 0. )   // the test hook above
            std::this_thread::sleep_for( std::chrono::duration<double>( delaySeconds ) );
        // accumulate into the shared grids: in partition order (the staged ordered merge above, the default),
        // or as they finish under '--ps-merge completion-order'
        psMergeInTurn( loop, pi, P, temp_uQuantities, temp_aQuantities );
    };

    psRunPartitionLoop( loop, memoryGate, prepare, deposit );
    if ( dynamicConc )
    {
        dtfePartitionsInFlight.store( 0, std::memory_order_relaxed );
        int most = 0;
        {
            std::lock_guard<std::mutex> lock( memoryGate.m );
            most = memoryGate.maxInFlight;
        }
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " dynamic concurrency: at most "
                << MESSAGE::cMagenta() << most << MESSAGE::cReset() << " partitions in flight at once "
                << "(the plan's " << psFloor << ").\n" << MESSAGE::Flush;
        // the SUMMARY's predicted-vs-measured line: the plan's prediction with the partitions that ran beyond it
        if ( most > psFloor and runProgressPredicted() > 0. )
            runProgressSetPredicted( runProgressPredicted() + double( most - psFloor )
                                     * ( state.options.psPartitionBytes - memoryGate.once ) );   // (once: in the plan's)
    }
    psReportLoopEnd( loop );

    // every partition copied its own particles (tempPart); the global array is not needed by
    // the normalization, post-processing or write phases -- free its ~48 B/particle now
    std::vector<Particle_data>().swap( state.particles );

    psNormalizeMoments( state, uQuantities, aQuantities, auditInputMass );

    psAggregateReport( uQuantities, aQuantities, totalPartitions, message );
}
