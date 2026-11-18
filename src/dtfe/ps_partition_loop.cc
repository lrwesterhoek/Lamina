// Part of src/DTFE.cpp, inside its #ifdef PHASE_SPACE: included once, in order, by that file and compiled only as
// part of its object, never alone.
// The PS partition loop: the staged ordered merge (StagedPartition, PSMergeStaging), the loop's context and
// progress, psMergeInTurn, psRunPartitionLoop and psReportLoopEnd.

// A partition the staged ordered merge parks until its turn (psPartitionedRun).
struct StagedPartition
{
    Quantities u, a;
    bool emptyPartition = true, skipped = false;
    double bytes = 0.;          // the RAM it holds (0 when spilled)
    PSMergeSpill::File spill;   // spilled: u and a are in this file, their vectors empty
    std::chrono::steady_clock::time_point parkedAt;   // when it parked (the 'merge time:' line's parked seconds)
};
// where a finished partition parks (psPartitionedRun's parkWhere)
enum { PARK_WAIT = 0, PARK_RAM = 1, PARK_DISK = 2 };
// One partition between its two stages, prepare and deposit (psPartitionedRun).
struct PSPrepared
{
    bool skipped = false, cacheOnly = false, emptyPartition = true;
    User_options opt;
    vector<Particle_data> part;
    std::shared_ptr<void> pre;   // the look-ahead's tessellation (psPrebuildTessellation); none: built in the deposit
    double seconds = 0.;         // the look-ahead's build time
};
// Leaves the memory gate when a gated partition's iteration ends (psRunPartitionLoop).
struct GateRelease { PartitionMemoryGate *g; int pi; ~GateRelease() { if ( g ) g->leave( pi ); } };

// The staged ordered merge (psPartitionedRun's comment): its settings, fixed before the loop, and its state -- the
// turn, the parked partitions, the spill and their counts -- shared by the partition threads under mergeMutex.
struct PSMergeStaging
{
    User_options const &options;   // the run's: the scratch folder, the memory gate's partition bytes, the park room
    bool orderedMerge = false;
    std::mutex mergeMutex;
    std::condition_variable mergeCv;
    int nextMerge = 0;
    std::map<int, StagedPartition> staged;      // finished partitions waiting for their turn (under mergeMutex)
    double stagedBytes = 0., stagedPeakBytes = 0.;
    size_t stagedPeakCount = 0, stagedRamCount = 0;   // parked in RAM: at most / now
    int nStaged = 0, nWaitedTurn = 0;                  // parked (RAM or disk); waited for their turn instead
    double stageCapBytes = -1.;                 // DTFE_MERGE_STAGE_GB: a fixed cap; < 0 = the free-memory rule
    bool staging = false;
    bool gateGuards = false;                    // the memory gate guards every start (parkWhere)
    int spillMode = 0;                          // DTFE_MERGE_SPILL: -1 off, 1 every parked partition, 0 the rule
    std::string spillWhy;
    std::string spillDir;
    bool spillUsable = false;
    int nSpilled = 0, nSpillTurn = 0;
    size_t spilledCount = 0, spilledPeakCount = 0;
    double spilledBytes = 0., spilledPeakBytes = 0., spillWritten = 0.;
    // the spills being written now (outside the lock): their bytes are not on the disk yet, so the room each new
    // one is checked against must leave them out (k writers would pass one reading together)
    double spillPendingBytes = 0.;
    std::string parkRoomText;   // the RAM parking's budget room (psParkRoom), when one is known
    int parkFloor = 0;          // the plan's mc when the dynamic concurrency runs more partitions (0: it does not)
    // what the merge took over the run (psReportLoopEnd's 'merge time:' line), summed under mergeMutex: the adds into
    // the full grids, their checkpoints (afterMerge) and the spilled sub-grids read back -- the serial work under the
    // lock -- and the waits before it: a parked partition's from its park to its merge (its thread went on), and a
    // thread's blocked on the lock or for its turn
    int nMergedHere = 0;        // partitions this process merged (not the skipped ones: restored or outside the window)
    double addSec = 0., checkpointSec = 0., spillReadSec = 0., parkedSec = 0., blockedSec = 0.;

    PSMergeStaging(User_options const &o, int const psConcurrency) : options( o )
    {
        orderedMerge = o.psOrderedMerge;
        if ( Env::isSet( "DTFE_MERGE_STAGE_GB" ) )
            stageCapBytes = std::max( 0., Env::real( "DTFE_MERGE_STAGE_GB", 0. ) ) * 1.e9;
        staging = orderedMerge and psConcurrency > 1 and stageCapBytes != 0.;
        // may a partition of 'bytes' be staged now? (under mergeMutex) The gate guards every start when it has an
        // estimate and the partitions are gated (the omp loop: not the speed split, unless asked for)
        gateGuards = psConcurrency > 1 and o.psPartitionBytes > 0.
                     and ( not o.psSpeedSplit or o.psOrderedMergeFlag );
        // the spill (psPartitionedRun's comment): DTFE_MERGE_SPILL=0 off, =1 every parked partition, unset the rule
        int const spillEnv = Env::tristate( "DTFE_MERGE_SPILL" );
        spillMode = spillEnv == 0 ? -1 : spillEnv == 1 ? 1 : 0;
        spillDir = PSMergeSpill::folder( o.scratchDir, spillWhy );
        spillUsable = staging and spillMode >= 0 and spillWhy.empty();
        if ( spillMode < 0 ) spillWhy = "DTFE_MERGE_SPILL=0";
    }

    // the bytes a partition's sub-grids hold (every accumulator, as the checkpoint counts them)
    static double subgridBytes(Quantities &q)
    {
        double b = 0.;
        PartitionCheckpoint::forEachAccumulator( q, [&](char const *, auto &v) { b += double( v.capacity() ) * double( sizeof( v[0] ) ); } );
        return b;
    }

    // where a partition of 'bytes' parks now (under mergeMutex): 1 RAM, 2 disk, 0 nowhere (it waits its turn)
    int parkWhere(double const bytes) const
    {
        if ( not staging ) return PARK_WAIT;
        // spoken-for room on the spill disk: the writes in progress, and with '--scratch-dir' (the spill folder
        // then) every scratch mapping in full -- a sparse file whose unwritten pages statvfs reports free
        // (over-counting the pages already written only spills less: the fallback is the park or the wait)
        double const reserved = spillPendingBytes
                                + ( options.scratchDir.empty() ? 0. : double( ScratchAlloc::scratchLiveBytes() ) );
        bool const disk = spillUsable and bytes > 0. and PSMergeSpill::room( spillDir, bytes, reserved,
                                                                                PSMergeSpill::margin( options.scratchDir ) );
        if ( spillMode > 0 and disk ) return PARK_DISK;   // the test hook
        bool ram;
        if ( stageCapBytes > 0. ) ram = stagedBytes + bytes <= stageCapBytes;
        else
        {
            double const freeNow = autoTuneSystemFree();
            if ( freeNow < 0. ) ram = stagedBytes + bytes <= 2.e9;   // free memory unknown: a small fixed room
            else ram = freeNow - 4.e9 - ( gateGuards ? options.psPartitionBytes : 0. ) >= bytes;
            // and within what the memory budget leaves beside the plan's predicted peak (psParkRoom says
            // why). stagedBytes counts RAM-parked ones only.
            // That prediction has the plan's partitions in flight: each one the dynamic concurrency runs beyond them,
            // besides this one (its lane is what it parks), takes a lane off the room -- the lane the ledger admitted
            // it with and the SUMMARY's prediction adds for it -- or the parked ones could take the run past the budget
            if ( ram and std::isfinite( options.psParkRoom ) )
            {
                int const beyond = parkFloor > 0
                    ? std::max( 0, dtfePartitionsInFlight.load( std::memory_order_relaxed ) - 1 - parkFloor ) : 0;
                double const lane = options.psPartitionBytes
                                    - std::min( options.psOnceBytes, options.psPartitionBytes );
                ram = stagedBytes + bytes <= std::max( 0., options.psParkRoom - double(beyond) * lane );
            }
        }
        if ( ram ) return PARK_RAM;
        if ( disk ) return PARK_DISK;
        return ( gateGuards and stageCapBytes < 0. ) ? PARK_RAM : PARK_WAIT;
    }
};

// The partition loop's progress + ETA (psPartitionedRun's comment), updated under the merge lock.
struct PSProgress
{
    int toProcess = 1;                                  // the partitions this process computes
    std::chrono::steady_clock::time_point loopStart;
    int done = 0;
    std::vector<double> partCost;                       // each partition's predicted cost (psPartitionCosts)
    double costTotal = 0., costDone = 0.;
};

// The partition loop's shared state, references to psPartitionedRun's: the run, the partitions, the merge, progress.
struct PSLoopCtx
{
    DTFE_State &state;
    User_options &userOptions;
    Quantities *uQuantities;
    Quantities *aQuantities;
    MESSAGE::Message &message;
    int &totalPartitions;
    PSPartitionPlan &plan;
    int &psConcurrency;
    std::vector<char> &skipPartition;
    PSProgress &progress;
    PSMergeStaging &merge;
};

// Says how the partitions merge: in index order, with their parking and spill.
static void psReportMergeMode(PSLoopCtx &c)
{
    auto &state = c.state;
    auto &message = c.message;
    auto &psConcurrency = c.psConcurrency;
    auto &orderedMerge = c.merge.orderedMerge;
    auto &stageCapBytes = c.merge.stageCapBytes;
    auto &staging = c.merge.staging;
    auto &gateGuards = c.merge.gateGuards;
    auto &spillMode = c.merge.spillMode;
    auto &spillWhy = c.merge.spillWhy;
    auto &spillDir = c.merge.spillDir;
    auto &spillUsable = c.merge.spillUsable;
    auto &parkRoomText = c.merge.parkRoomText;
    auto &parkFloor = c.merge.parkFloor;
    if ( std::isfinite( state.options.psParkRoom ) )
    {
        char rb[96];
        std::snprintf( rb, sizeof(rb), ", in RAM up to the %.1f GB the memory budget leaves beside the plan",
                       std::max( 0., state.options.psParkRoom ) / 1.e9 );
        parkRoomText = rb;
        if ( parkFloor > 0 )   // the dynamic concurrency: the room shrinks beyond the plan's partitions (parkWhere)
        {
            double const lane = state.options.psPartitionBytes
                                - std::min( state.options.psOnceBytes, state.options.psPartitionBytes );
            std::snprintf( rb, sizeof(rb), ", less a lane (%.2f GB) for each partition in flight beyond the plan's %d",
                           lane / 1.e9, parkFloor );
            parkRoomText += rb;
        }
    }
    if ( orderedMerge and psConcurrency > 1 )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " partitions merge in index order "
                << "(bit-reproducible sums); "
                << ( staging ? ( stageCapBytes > 0. ? "one finished before its turn waits in memory, up to "
                                 + std::to_string( int( std::round( stageCapBytes / 1.e9 ) ) ) + " GB in all "
                                 "(DTFE_MERGE_STAGE_GB), while its thread goes on"
                                 : gateGuards ? std::string( "one finished before its turn waits in memory and its "
                                 "thread goes on (the memory gate admits the next partition)" )
                                   + parkRoomText
                                 : std::string( "one finished before its turn waits in memory, while the free memory "
                                 "allows, and its thread goes on" ) )
                             : std::string( "one finished before its turn waits for it (DTFE_MERGE_STAGE_GB=0)" ) )
                << ( not staging ? std::string()
                     : spillUsable ? std::string( spillMode > 0 ? "; every one with sub-grids is spilled to disk"
                                                                : "; when the memory cannot hold it, spilled to disk" )
                                     + " in '" + spillDir + "'" + ( spillMode > 0 ? " (DTFE_MERGE_SPILL=1)" : "" )
                     : "; no spill to disk (" + spillWhy + ")" )
                << ( state.options.psCompletionOrderMergeFlag ? "; --resume overrides --ps-merge completion-order" : "" )
                << ".\n" << MESSAGE::Flush;
}

// A finished partition, under the merge lock: merged in its turn (then the parked ones due), else parked or waits.
static void psMergeInTurn(PSLoopCtx &c, int const pi, PSPrepared &P, Quantities &temp_uQuantities,
                          Quantities &temp_aQuantities)
{
    auto &state = c.state;
    auto &userOptions = c.userOptions;
    auto &uQuantities = c.uQuantities;
    auto &aQuantities = c.aQuantities;
    PSMergeStaging &m = c.merge;
    PSProgress &progress = c.progress;
    auto &orderedMerge = m.orderedMerge;
    auto &mergeMutex = m.mergeMutex;
    auto &mergeCv = m.mergeCv;
    auto &nextMerge = m.nextMerge;
    auto &staged = m.staged;
    auto &stagedBytes = m.stagedBytes;
    auto &stagedPeakBytes = m.stagedPeakBytes;
    auto &stagedPeakCount = m.stagedPeakCount;
    auto &stagedRamCount = m.stagedRamCount;
    auto &nStaged = m.nStaged;
    auto &nWaitedTurn = m.nWaitedTurn;
    auto &spillWhy = m.spillWhy;
    auto &spillDir = m.spillDir;
    auto &spillUsable = m.spillUsable;
    auto &nSpilled = m.nSpilled;
    auto &nSpillTurn = m.nSpillTurn;
    auto &spilledCount = m.spilledCount;
    auto &spilledPeakCount = m.spilledPeakCount;
    auto &spilledBytes = m.spilledBytes;
    auto &spilledPeakBytes = m.spilledPeakBytes;
    auto &spillWritten = m.spillWritten;
    auto &spillPendingBytes = m.spillPendingBytes;
    // the merge's times (PSMergeStaging, the 'merge time:' line)
    using Clock = std::chrono::steady_clock;
    auto const since = [](Clock::time_point const t) { return std::chrono::duration<double>( Clock::now() - t ).count(); };
    auto const tArrive = Clock::now();
    std::unique_lock<std::mutex> lock( mergeMutex );
    m.blockedSec += since( tArrive );
    // one partition's grids into the full ones, then its checkpoint (partition_checkpoint.h), while the grids
    // hold still under the merge lock: after this merge unless the last write took more than 5% of the time
    // since it
    auto mergeOne = [&](int const pj, bool const emptyPartition, bool const skipped, Quantities &tu, Quantities &ta)
    {
        auto const tAdd = Clock::now();
        if ( not emptyPartition )
        {   // with --grid-window every partition's sub-grid IS the window: the merge is window-local
            size_t const *wo = state.options.psWindowOn ? state.options.psWindowLo   : nullptr;
            size_t const *wd = state.options.psWindowOn ? state.options.psWindowDims : nullptr;
            uQuantities->addFromSubgrid( tu, &(state.options.gridSize[0]), wo, wd );
            aQuantities->addFromSubgrid( ta, &(state.options.gridSize[0]), wo, wd );
        }
        m.addSec += since( tAdd );
        if ( not skipped ) ++m.nMergedHere;
        auto const tCheckpoint = Clock::now();
        PartitionCheckpoint::afterMerge( pj, skipped, *uQuantities, *aQuantities, userOptions.verboseLevel );
        m.checkpointSec += since( tCheckpoint );
    };
    bool stagedHere = false;
    if ( orderedMerge and nextMerge != pi )
    {
        // not its turn: park its sub-grids -- in RAM, or spilled to disk -- and free the thread,
        // else wait (parkWhere above)
        double const bytes = P.emptyPartition ? 0. : PSMergeStaging::subgridBytes( temp_uQuantities )
                                                     + PSMergeStaging::subgridBytes( temp_aQuantities );
        int where = m.parkWhere( bytes );
        if ( where == PARK_DISK )
        {
            // the write runs outside the merge lock (the merges of the others go on); the sub-grids stay in
            // memory until the file is complete, so a failed write loses nothing
            PSMergeSpill::File file;
            std::string why;
            spillPendingBytes += bytes;   // counted against the room of the spills that start meanwhile
            lock.unlock();
            bool const wrote = PSMergeSpill::write( temp_uQuantities, temp_aQuantities, spillDir, file, why );
            auto const tRelock = Clock::now();
            lock.lock();
            m.blockedSec += since( tRelock );
            spillPendingBytes -= bytes;
            if ( nextMerge == pi )
            {   // its turn came during the write: the drain before it found nothing parked here and stopped,
                // so it merges now, from memory, and the file goes (a late park would never be merged)
                ++nSpillTurn;
                where = -1;
            }
            else if ( wrote )
            {
                PSMergeSpill::release( temp_uQuantities );
                PSMergeSpill::release( temp_aQuantities );
                StagedPartition &S = staged[pi];
                S.u = std::move( temp_uQuantities );   // empty vectors; the sub-grid placement stays with them
                S.a = std::move( temp_aQuantities );
                S.emptyPartition = P.emptyPartition;
                S.skipped = P.skipped;
                S.bytes = 0.;
                ++nStaged;
                ++nSpilled;
                ++spilledCount;
                spilledBytes += file.bytes;
                spillWritten += file.bytes;
                spilledPeakCount = std::max( spilledPeakCount, spilledCount );
                spilledPeakBytes = std::max( spilledPeakBytes, spilledBytes );
                S.spill = std::move( file );
                S.parkedAt = Clock::now();
                stagedHere = true;
            }
            else
            {   // the disk refused (full, unwritable): no more spills this run; parked in RAM or waiting, as
                // without the spill
                if ( spillUsable )
                {
                    MESSAGE::Warning warning( userOptions.verboseLevel );
                    warning << "the staged ordered merge cannot spill to disk (" << why << "); from now on a "
                            << "partition that finishes before its turn waits in memory or for its turn.\n"
                            << MESSAGE::EndWarning;
                    spillUsable = false;
                    spillWhy = why;
                }
                where = m.parkWhere( bytes );
            }
        }
        if ( where == PARK_RAM )
        {
            StagedPartition &S = staged[pi];
            S.u = std::move( temp_uQuantities );
            S.a = std::move( temp_aQuantities );
            S.emptyPartition = P.emptyPartition;
            S.skipped = P.skipped;
            S.bytes = bytes;
            stagedBytes += bytes;
            ++nStaged;
            ++stagedRamCount;
            stagedPeakBytes = std::max( stagedPeakBytes, stagedBytes );
            stagedPeakCount = std::max( stagedPeakCount, stagedRamCount );
            S.parkedAt = Clock::now();
            stagedHere = true;
        }
        else if ( where == PARK_WAIT )
        {
            if ( not P.skipped ) ++nWaitedTurn;
            auto const tWait = Clock::now();
            mergeCv.wait( lock, [&]{ return nextMerge == pi; } );
            m.blockedSec += since( tWait );
        }
    }
    if ( not stagedHere )
    {
        mergeOne( pi, P.emptyPartition, P.skipped, temp_uQuantities, temp_aQuantities );
        if ( orderedMerge )
        {
            nextMerge = pi + 1;
            // the staged partitions whose turn has now come, in index order; a spilled one is read back first
            // (under the lock, beside its merge: one partition's sub-grids at a time)
            for ( auto it = staged.find( nextMerge ); it != staged.end(); it = staged.find( nextMerge ) )
            {
                StagedPartition S = std::move( it->second );
                staged.erase( it );
                m.parkedSec += since( S.parkedAt );
                if ( S.spill.fd >= 0 )
                {
                    spilledBytes -= S.spill.bytes;
                    --spilledCount;
                    std::string why;
                    auto const tRead = Clock::now();
                    if ( not PSMergeSpill::read( S.spill, S.u, S.a, why ) )
                        throwError( "the staged ordered merge: partition ", nextMerge, " was spilled to '" + spillDir
                                    + "' and cannot be merged: " + why );
                    m.spillReadSec += since( tRead );
                }
                else
                {
                    stagedBytes -= S.bytes;
                    --stagedRamCount;
                }
                mergeOne( nextMerge, S.emptyPartition, S.skipped, S.u, S.a );
                ++nextMerge;
            }
            mergeCv.notify_all();
        }
    }

    // progress over the partitions actually processed (the skipped ones cost nothing); a staged partition is
    // done: its work is finished, only its add waits
    if ( not P.skipped )
    {
        int &psDone = progress.done;
        int const toProcess = progress.toProcess;
        ++psDone;
        progress.costDone += progress.partCost[size_t(pi)];
        double const costDone = progress.costDone;
        double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - progress.loopStart ).count();
        double const frac    = double(psDone) / double(toProcess);
        // the time per unit of predicted cost so far, times the predicted cost still to come
        double const eta     = ( costDone > 0. ) ? elapsed * std::max( 0., progress.costTotal - costDone ) / costDone : 0.;
        partitionProgress( userOptions.verboseLevel, psDone, toProcess, int(100.*frac + 0.5), elapsed, eta );
        runProgressPhase( "partitions", psDone, toProcess, psDone < toProcess ? eta : 0. );
    }
}

// Runs the partitions: in turn with the look-ahead (k+1 prepared while k deposits), else in OpenMP behind the gate.
template <class Prepare, class Deposit>
static void psRunPartitionLoop(PSLoopCtx &c, PartitionMemoryGate &memoryGate, Prepare &prepare, Deposit &deposit)
{
    auto &state = c.state;
    auto &userOptions = c.userOptions;
    auto &message = c.message;
    auto &totalPartitions = c.totalPartitions;
    auto &plan = c.plan;
    auto &psConcurrency = c.psConcurrency;
    auto &skipPartition = c.skipPartition;
    int const toProcess = c.progress.toProcess;
    // The LOOK-AHEAD. At --max-concurrent 1 with the GPU deposit the cores would idle while the
    // GPU works (the deposit is most of a run), and each next partition's selection +
    // triangulation + vertex densities would wait behind it. So partition k+1 is prepared on a CPU thread
    // while partition k deposits (runLookAhead, DTFE.cpp), as without the look-ahead (same particles, same sequential
    // insert), and the merge stays in index order, so the outputs are byte-identical (tests/ps_runtime_check.sh). It
    // holds one more partition's particles + tessellation; before each look-ahead the free memory is checked against
    // that (the auto-tuner's prediction counts it too) and the partition is prepared in turn when it does not fit.
    // '--no-overlap' switches it off; DTFE_OVERLAP=1 forces it on a CPU deposit too (the test suite's byte-identity
    // check on a build without a GPU), DTFE_OVERLAP=0 switches it off.
    bool gpuDeposit = false;
#if defined(PS_GPU) && NO_DIM==3
    gpuDeposit = state.options.psUseMetal;
#endif
    int const overlapEnv = Env::tristate( "DTFE_OVERLAP" );   // 1 forces the look-ahead, 0 bans it
    bool const overlapForced = overlapEnv == 1;
    bool const overlapBanned = overlapEnv == 0;
    bool overlap = psConcurrency == 1 and toProcess > 1 and not state.options.psNoOverlap
                   and not state.options.psLookAheadOffBudget   // the auto-tuner's: it does not fit the budget
                   and not overlapBanned and ( gpuDeposit or overlapForced );
    // what the look-ahead holds: one partition's padded particle copy + its tessellation (auto_tune.h's
    // model: Particle_data + 666 B per vertex, padding included)
    double aheadBytes = 0.;
    if ( overlap )
    {
        double padVol = 1., ownVol = 1.;
        {
            Box r, rp;
            plan.regions( 0, r, rp );
            for (int d=0; d<NO_DIM; ++d) { ownVol *= double(r[2*d+1]-r[2*d]); padVol *= double(rp[2*d+1]-rp[2*d]); }
        }
        double const nAll = state.options.psParticlesDeferred ? double(userOptions.psPresetN) : double(state.particles.size());
        aheadBytes = nAll / double(totalPartitions) * ( padVol / std::max( ownVol, 1e-300 ) )
                     * ( double(sizeof(Particle_data)) + 666. );
        // the memory budget too, not only the free memory: what it leaves beside the grids,
        // particles and one partition must hold the look-ahead, or the run has none (the same for every
        // partition, so decided once)
        if ( aheadBytes > state.options.psLookAheadRoom )
        {
            overlap = false;
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " look-ahead off: the ~"
                    << std::round( aheadBytes/1.e7 )/100. << " GB it holds is more than the ~"
                    << std::round( std::max( 0., state.options.psLookAheadRoom )/1.e7 )/100. << " GB the memory "
                    << "budget leaves beside one partition; partitions are prepared in turn.\n" << MESSAGE::Flush;
        }
    }
    if ( overlap )
    {
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " look-ahead on: partition k+1 is "
                << "tessellated on a CPU thread while partition k deposits"
                << ( gpuDeposit ? " on the GPU" : " (DTFE_OVERLAP=1)" )
                << ( state.options.psGpuShareMille < 1000
                     ? " (and on the CPU for the GPU share's other partitions)" : "" )
                << " (~" << std::round( aheadBytes/1.e7 )/100.
                << " GB more in memory; '--no-overlap' switches it off).\n" << MESSAGE::Flush;

        // every partition in index order (a skipped one too: it takes its turn in the ordered merge), the next one
        // prepared ahead unless it is skipped; each iteration counts its triangulation off (the guard). The line
        // names the next partition by its 0-based index ('during this deposit')
        std::vector<int> order( static_cast<size_t>( totalPartitions ) );
        for (int pi=0; pi<totalPartitions; ++pi) order[size_t(pi)] = pi;
        runLookAhead<ConcurrentTriangulationGuard, PSPrepared>( order,
            [&](int) { return aheadBytes; }, prepare,
            [&](int const pi, PSPrepared &P) { deposit( pi, P, userOptions.verboseLevel ); },
            [&](int const next) { return not skipPartition[size_t(next)]; },
            LookAheadLabel{ 0, "this deposit" }, userOptions.verboseLevel );
    }
    else
    {
    // MONOTONIC: the partitions start in index order, which the ordered merge and the memory gate rely on. A plain
    // schedule(dynamic, 1) is nonmonotonic since OpenMP 5.0, and LLVM's libomp runs it as a static steal: thread t
    // works through the t-th contiguous block, so the later blocks finish while thread 0 is still on the first and
    // nearly every partition parks or spills
#ifdef OPEN_MP
    #pragma omp parallel for schedule(monotonic: dynamic, 1) num_threads(psConcurrency)
#endif
    for (int pi=0; pi<totalPartitions; ++pi)
    {
        ConcurrentTriangulationGuard concurrentTriangulation;
        // the memory gate, BEFORE the particle copy (part of a partition's memory): a partition that builds or
        // loads waits while the free memory cannot hold one more and another (under the ordered merge: a
        // lower-numbered one) is in flight; a skipped one passes, and so does the auto-tuner's speed split
        // (tiny uniform partitions). Released at the iteration's end.
        // (the speed split has its own flag, psSpeedSplit, since every split merges in order)
        bool const gated = not skipPartition[size_t(pi)] and ( not state.options.psSpeedSplit or state.options.psOrderedMergeFlag );
        if ( gated )
            memoryGate.enter( pi, userOptions.verboseLevel );
        GateRelease const gateRelease{ gated ? &memoryGate : nullptr, pi };
        PSPrepared P;
        prepare( pi, false, P );
        int tnum = 0;
#ifdef OPEN_MP
        tnum = omp_get_thread_num();
#endif
        deposit( pi, P, (tnum == 0) ? userOptions.verboseLevel : 0 );
    }
    }
}

// After the loop: the GPU fallbacks, the checkpoint writes, the ordered merge's line.
static void psReportLoopEnd(PSLoopCtx &c)
{
    auto &userOptions = c.userOptions;
    auto &message = c.message;
    auto &psConcurrency = c.psConcurrency;
    auto &psLoopStart = c.progress.loopStart;
    auto &orderedMerge = c.merge.orderedMerge;
    auto &staged = c.merge.staged;
    auto &stagedPeakBytes = c.merge.stagedPeakBytes;
    auto &stagedPeakCount = c.merge.stagedPeakCount;
    auto &nStaged = c.merge.nStaged;
    auto &nWaitedTurn = c.merge.nWaitedTurn;
    auto &staging = c.merge.staging;
    auto &spillWhy = c.merge.spillWhy;
    auto &spillUsable = c.merge.spillUsable;
    auto &nSpilled = c.merge.nSpilled;
    auto &nSpillTurn = c.merge.nSpillTurn;
    auto &spilledPeakCount = c.merge.spilledPeakCount;
    auto &spillWritten = c.merge.spillWritten;
    // A GPU deposit that fell back to the CPU warns from the master thread only (the others run at
    // verbose 0), so with partitions in flight on several threads most fallbacks would be silent: say how
    // many there were
    if ( int const nFallback = psGpuFallbacksTaken( true ) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << nFallback << " partition deposit pass(es) were meant for the GPU and ran on the CPU instead. Only "
                << "the master thread's partitions print the reason (above, when one of them fell back); the "
                << "grids are complete, the run only took longer.\n" << MESSAGE::EndWarning;
    }
    // What the face walks beside the GPU deposit (the default, ps_interpolation.cc PSEarlyFaceWalk) came to, once for
    // the loop (a pass says so itself only where it prints its grid lines, which the partition loop suppresses). The
    // seconds are the kept walks'
    {
        long long kept = 0, discarded = 0;
        double walkSec = 0., waitSec = 0.;
        psFaceWalksEarlyTaken( kept, discarded, walkSec, waitSec, true );
        if ( kept + discarded > 0 )
        {
            MESSAGE::Message early( userOptions.verboseLevel );
            early << "PS-DTFE: the face sums (--face-gradient) of " << kept << " pass(es) were walked beside the GPU "
                  << "deposit and kept; those walks took " << walkSec << " s in all, " << waitSec
                  << " s of it after the device call had returned; " << discarded
                  << " walk(s) thrown away after a failed GPU deposit (the CPU deposit made those face sums).\n"
                  << MESSAGE::Flush;
        }
    }

    PartitionCheckpoint::reportWrites( std::chrono::duration<double>( std::chrono::steady_clock::now() - psLoopStart ).count(),
                                       userOptions.verboseLevel );
    // what the staged ordered merge did: how many partitions finished before their turn and waited in
    // memory, at most how many / how much at once, and how many waited for their turn holding their thread
    if ( orderedMerge and psConcurrency > 1 )
    {
        if ( not staged.empty() )   // cannot happen: every partition merges once, the last drains the rest
            throwError( "internal: ", staged.size(), " staged partition(s) were never merged" );
        // (of those, how many were spilled to disk, at most how many at once, and the GB written)
        auto gbText = [](double const b) { char t[32]; std::snprintf( t, sizeof(t), "%.2f", b / 1.e9 ); return std::string( t ); };
        message << MESSAGE::cDim() << "  ordered merge: " << nStaged << " partition(s) finished before their turn: "
                << ( nStaged - nSpilled ) << " waited in memory (at most " << stagedPeakCount << " at once, "
                << std::round( stagedPeakBytes / 1.e7 ) / 100. << " GB), " << nSpilled << " spilled to disk"
                << ( nSpilled > 0 ? " (at most " + std::to_string( spilledPeakCount ) + " at once, "
                                    + gbText( spillWritten ) + " GB written)"
                                  : spillUsable or not staging ? std::string() : " (" + spillWhy + ")" )
                << ( nSpillTurn > 0 ? " (+" + std::to_string( nSpillTurn ) + " whose turn came while writing: merged "
                                      "from memory)" : "" )
                << "; " << nWaitedTurn << " waited for their turn "
                << "holding their thread" << ( staging and nWaitedTurn > 0 ? " (no room to stage)" : "" ) << "\n"
                << MESSAGE::cReset() << MESSAGE::Flush;
    }
    // what the merge took (PSMergeStaging's times): the serial work under the merge lock -- the adds into the full
    // grids, their checkpoints, the spilled sub-grids read back -- and the waits before it. In index order a
    // partition waits for its turn parked (its thread goes on) or holding its thread (the lock, or no room to
    // park); in completion order only for the lock
    {
        PSMergeStaging const &m = c.merge;
        char t[160];
        std::snprintf( t, sizeof(t), " partition(s) added to the full grids in %.2f s, their checkpoints %.2f s",
                       m.addSec, m.checkpointSec );
        std::string text = "  merge time: " + std::to_string( m.nMergedHere ) + t;
        if ( m.nSpilled > 0 )
        {
            std::snprintf( t, sizeof(t), ", the spilled ones read back in %.2f s", m.spillReadSec );
            text += t;
        }
        text += " (one at a time, under the merge lock); ";
        if ( orderedMerge )
            std::snprintf( t, sizeof(t), "waiting for their turn: %.2f s parked, %.2f s holding a thread",
                           m.parkedSec, m.blockedSec );
        else
            std::snprintf( t, sizeof(t), "waiting for the merge lock: %.2f s", m.blockedSec );
        message << MESSAGE::cDim() << text << t << "\n" << MESSAGE::cReset() << MESSAGE::Flush;
    }
}
