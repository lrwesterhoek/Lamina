// Part of src/auto_tune.cc, inside its #ifdef PHASE_SPACE and its anonymous namespace: included once, in order, by
// that file and compiled only as part of its object, never alone.
// The PS planner (AutoTunePSPlanner): its memory model of a split, its report, and the phases that choose the
// split (choose, irreducibleCase, smallSetCase, overBudgetCase, memorySplit).

/* The PS-DTFE plan once its inputs are known (autoTunePartitioning): the memory model of a split (partitionBytes,
   oneDomainPeak, snapshotBytes, lookAheadBytes, loneLaneSubGrids), the report of a choice (finishReport) and the
   phases that choose the split (choose). It reads the inputs by name (its base) and writes only through the three
   references. */
struct AutoTunePSPlanner : AutoTunePSInputs
{
    User_options &u;                 // the options the plan sets: --partition, --max-concurrent, the look-ahead switch
    AutoTuneReport &report;          // the report it completes (fixed, budget and the points' terms already set)
    MESSAGE::Message &message;       // the AUTO-TUNE lines

    // ONE DOMAIN on the CPU ('--partition 1 1 1', or one domain picked here: partitionOn false): the triangulation
    // deposits straight into the run's own grids (DTFE_interpolation: no reserveMemory, sub-grid or addFromSubgrid),
    // so the grids 'fixed' books are the deposit's own; partitionBytes(1) would book them twice (copyOne) and every
    // grid's conversion at once. Modelled per pass (u, then a), in ps_interpolation.cc's order: during the loop the
    // fixed-point words (the density's 8 B, each moment grid's 2 x its Real bytes), the int stream count, the flags
    // and the Real accumulators that are not fixed-point sums; after it each moment grid's Real grid beside its
    // words, the words freed grid by grid (psFxToReal), then the face sums' the same way (words in every build:
    // autoTuneFaceBytes), then the density from the mass words; then the export grids ('.streams' or '.tetTouch',
    // '.hidden_streams', the caustic bits, the volume-weighted dispersion's density copy).
    // A freed block under kAllocKeepBytes stays in the footprint, so the export grids pile on top of freed words
    // smaller than that; the next pass reuses what the previous one freed, and the derived fields come last. Bytes,
    // for 'cells' sub-grid cells.
    double oneDomainPeak(double const cells) const
    {
        double const B4 = kAllocKeepBytes;      // the allocator returns a freed block of this size or more
        double const R = double(sizeof(Real));
        bool const fxOn = ps.phaseSpace and sizeof(Real) == 4;
        double live = 0., kept = 0., peak = 0.;
        auto alloc = [&](double const b) { live += b*cells; peak = std::max( peak, live + kept ); };
        auto release = [&](double const b) { live -= b*cells; if ( b*cells < B4 ) kept += b*cells; };
        for (Field *f : { &u.uField, &u.aField })
        {
            if ( not f->selected() ) continue;
            bool const massFx = ps.phaseSpace and sizeof(Real) == 4 and ( f->density or f->velocity_tweb );   // fxCell
            std::vector<double> const G = fxOn ? autoTuneFxGrids(*f, ps) : std::vector<double>();
            double sumG = 0.;
            for (double const g : G) sumG += g;
            double const faceRf = autoTuneFaceBytes(*f, ps), faceW = faceRf * 8. / R;   // the face sums' grids, words
            double const mask = ps.caustics ? double(sizeof(PSCausticClass::CausticMask)) : 0.;
            double const E = R + R + ( ps.caustics ? R : 0. )                                    // the export grids
                           + ( ( f->velocity_dispersion and massFx ) ? R : 0. );
            // the loop's temporaries: the pass's slots, less the grids made after it, plus the words and the int count
            double const pre = autoTuneBytesPerCell(*f, ps, 0) - E - sumG - ( massFx ? R : 0. ) - faceRf
                             + 2.*sumG + ( massFx ? 8. : 0. ) + faceW + double(sizeof(int));
            double const reuse = std::min( kept, pre*cells );   // ... in what the previous pass freed
            kept -= reuse;
            live += pre*cells;
            peak = std::max( peak, live + kept );
            for (double const g : G) { alloc( g ); release( 2.*g ); }
            if ( faceRf > 0. )   // fx.faceS, faceN, faceP: each one's Real grid beside its words, the words then freed
                for (double const c : { double(noVelComp), 1., 1. }) { alloc( 2.*NO_DIM*c*R ); release( 2.*NO_DIM*c*8. ); }
            if ( massFx ) { alloc( R ); release( 8. ); }
            // a face pass's stream-average divergence, made in its normalization and kept to the output (its mass
            // flux grid is in 'pre' already: autoTuneFaceOutputBytes)
            alloc( autoTuneFaceOutputBytes( *f, ps, false ) );
            alloc( E );
            release( double(sizeof(int)) + 2. + mask );   // the int count, the flags, the caustic mask (pass end)
        }
        // the regular part's twins (made in the normalization, beside the face sums this model still counts), the
        // derived fields and the T-web stage come on top of what the passes left
        alloc( autoTuneStreamsTwinBytes(u.uField, ps) + autoTuneStreamsTwinBytes(u.aField, ps)
               + autoTuneDerivedBytes(u.uField) + autoTuneDerivedBytes(u.aField) );
        return peak;
    }

    // per-partition bytes for an n^3 Lagrangian split
    double partitionBytes(int n) const
    {
        double const nOwn = N / partsOf( n );
        double const fPad = padFactorOf( n );   // k spacings, or the Eulerian rule's floor of 0.3 partition widths
        double const nPad = nOwn * fPad;
        double m = nPad * (partBytes + kPSVertexBytes + kParInsert);   // particle copy coexists with the growing triangulation
        // '--sample-points': each worker fills a LOCAL hits buffer (pair<uint32,StreamRec>,
        // padded to 176 B) and merges it under the mutex at the end, so this part DOES scale
        // down with a finer split -- roughly this partition's share of the total records.
        if ( nPts > 0 )
            m += double(dtfeHardwareThreads()) * kHitSinkBlockBytes;   // one HitSink block per thread
        // Eulerian sub-grid this partition allocates (subGridCells below)
        double const subCells = subGridCells( n );
        // a mixed run's split: the larger of a CPU lane's deposit and a GPU lane's arrays before the lock with the
        // grids it keeps after it (mixedLane, gpuAfterLockCellBytes)
        if ( mixed and n >= 2 )
        {
            double cpu = 0., gpuPre = 0.;
            mixedLane( n, cpu, gpuPre );
            return m + std::max( cpu, gpuPre + gpuAfterLockCellBytes( n )*subCells );
        }
        if ( metalActive )
        {
            // flat tet arrays (the 16-float record 64 + masses 4 + the finite-cell ordinal 4 B/tet,
            // + vels 48 when velocity-derived grids are requested; the cost-sort permutes IN PLACE,
            // so no gather copy) + the PSGpuGrids host and device copies of the selected grids
            // (gpuCellBytes; under --grid-window held once, snapshotBytes) -- or what the lane holds once its
            // deposit has let go of the device, where a face pass makes that the larger (gpuAfterHoldCellBytes).
            // With the face walk beside the deposit (earlyFace) a face pass holds its face sums' words beside the
            // hold, and every lane holds them from before its device call, its wait for the device included: under
            // --grid-window those words alone (snapshotBytes takes them out of the deposit it books once)
            double const perTet = (kGpuTetBytes + (psVel or psDisp or psGrad ? kGpuTetVelBytes : 0.))
                                * kGpuTetSlack;
            m += perTet*kTetsPerVertex*nOwn
               + ( u.psWindowOn ? earlyFace*subCells
                                : std::max( gpuCellBytes() + earlyFace, gpuAfterHoldCellBytes( n ) )*subCells );
            // the exact deposit (--exact) runs depositExactItems (ps_metal_host.cc): a per-tet weight total
            // (8 B) + the item list, 8 B per 16 cells of a tet's bbox window -- ~20x the tet's
            // volume in cells on the crossed-waves benchmark, at least one item -- capped at 32M items
            if ( psExact )
            {
                double const nTet = kTetsPerVertex*nOwn;
                // --grid-window: items cover only the window's cells (the hull of each tet's bbox with
                // the window), so the window's cell count sets the items, not the virtual full grid
                double gridCells = 1.;
                for (int d=0; d<NO_DIM; ++d) gridCells *= double(outGrid[d]);
                double const cellsPerTet = gridCells / (6.*double(N));
                // (the weight totals are two 4 B words per tet: an exact per-tet sum)
                m += 8.*nTet + 8.*std::min( 32.e6, nTet*(1. + 20.*cellsPerTet/16.) );
            }
        }
        else
        {
            // CPU deposit sub-grids + the deposit threads' private copies, as many as the copy budget allows, unless
            // that leaves fewer than half the threads: the deposit then runs in SLAB mode (ps_interpolation.cc, the
            // same rule) with no copies, holding a cell handle, a first-slab byte and ~2 slab-list entries per tet
            // (+ the binning arrays). The fixed-point words (fxCell, fxMom) count during the deposit, and again at
            // their conversion after the loop only where that exceeds the copies or slab lists, freed by then. A
            // private copy holds neither the words nor the shared counts, flags and masks (autoTunePSCountBytes), so
            // the default single build has copyPriv = 0 and every core in the copy mode, as ps_interpolation.cc's
            // accBytes; only the double build's Real sums book copies. The face sums are
            // words in every build: fxFace beyond their Real grids during the deposit, and those grids (faceR)
            // beside the words at the conversion.
            double const copyOne = bCellDeposit*subCells;
            double const copyPriv = std::max( 0., copyCell )*subCells;   // (auto_tune/ps_inputs.cc copyCell: one pass's copy)
            int const copyThreads = copyPriv > 0.
                ? std::min( cores, 1 + int( std::min( double(cores), psDepositThreadBytes() / copyPriv ) ) ) : cores;
            double const slabBytes = kTetsPerVertex * nPad * ( 8. + 1. + 2.*4. + 20. );
            double const helpers = 2 * copyThreads < cores ? slabBytes : double(copyThreads - 1) * copyPriv;
            // One domain (oneDomainPeak above): its grids are 'fixed''s, so only what the deposit holds beyond them
            // counts here, with the copies or slab lists and the threads' sample lists, all kept by the allocator
            // once freed. A user's split such as '--partition 1 2 2' runs the partition path (userPartitionOn), and
            // with --scratch-dir 'fixed' has no grid term: both keep the partition formula.
            if ( n == 1 and not userPartitionOn and not scratchOn )
                m += helpers + threadScratch + std::max( 0., oneDomainPeak( subCells ) - bCell*subCells );
            else
                m += copyOne + (fxCell + fxMom + fxFace)*subCells
                   + std::max( helpers, (fxCell + fxMom + faceR)*subCells );
        }
        return m;
    }

    /* A MIXED run ('--gpu-share' below 1, DTFE.cpp psResolveGpuShare): the share's partitions deposit on the GPU, the
       others on the CPU, at the same time. The GPU deposit's device grids and their host copies exist only while a
       deposit holds the GPU (ps_metal_host.cc: the buffers are made once ctxMutex is taken, the output grids allocated
       only then), so one lane at a time holds them: snapshotBytes adds them once. A lane waiting for the GPU holds
       only what is built before the lock -- the flat tet arrays and the exact items -- and a CPU lane its deposit's
       sub-grids with the copies or slab lists; a GPU lane after its hold keeps the host grids the copy-back made, and
       with a face pass the face sums too (gpuAfterLockCellBytes, beside its arrays: booked, not left to the CPU lane
       being the larger). So a lane books the larger of the two (partitionBytes), each with its branch's expressions
       above, the GPU's without the lock holder's grids (gpuCellBytes). For a split only (n >= 2): one domain has no
       partition loop, and so no share. */
    void mixedLane(int n, double &cpu, double &gpuPre) const
    {
        double const nOwn = N / partsOf( n );
        double const nPad = nOwn * padFactorOf( n );
        double const subCells = subGridCells( n );
        // the GPU lane before the lock: the flat tet arrays, + the exact deposit's weight totals and item list
        double const perTet = (kGpuTetBytes + (psVel or psDisp or psGrad ? kGpuTetVelBytes : 0.))
                            * kGpuTetSlack;
        gpuPre = perTet*kTetsPerVertex*nOwn;
        if ( psExact )
        {
            double const nTet = kTetsPerVertex*nOwn;
            double gridCells = 1.;
            for (int d=0; d<NO_DIM; ++d) gridCells *= double(outGrid[d]);
            double const cellsPerTet = gridCells / (6.*double(N));
            gpuPre += 8.*nTet + 8.*std::min( 32.e6, nTet*(1. + 20.*cellsPerTet/16.) );
        }
        // the CPU lane: the CPU branch's partition formula
        double const copyOne = bCellDeposit*subCells;
        double const copyPriv = std::max( 0., copyCell )*subCells;
        int const copyThreads = copyPriv > 0.
            ? std::min( cores, 1 + int( std::min( double(cores), psDepositThreadBytes() / copyPriv ) ) ) : cores;
        double const slabBytes = kTetsPerVertex * nPad * ( 8. + 1. + 2.*4. + 20. );
        double const helpers = 2 * copyThreads < cores ? slabBytes : double(copyThreads - 1) * copyPriv;
        cpu = copyOne + (fxCell + fxMom + fxFace)*subCells
            + std::max( helpers, (fxCell + fxMom + faceR)*subCells );
    }

    /* THE LANE SHARE. c partitions in flight are never all at their own peak when the run peaks: the GPU
       lanes queue for one device (ctxMutex), a CPU lane holds its sub-grids only around its deposit, and the
       partitions' sub-grids differ in size. So a plan made by memorySplit books the first lane whole (partitionBytes)
       and each other one laneShareCut less, for 2 to laneShareCap partitions in flight, where the measured runs were. */
    bool laneMixed(int n) const { return mixed and n >= 2; }            // partitionBytes' mixed branch
    // the split's mean sub-grid over its largest (psPartitionSpans): the share's floor. 1 when not known
    double laneSpread(int n) const
    {
        double most = 0., mean = 0.;
        if ( isUserSplit( n ) ) { most = spanUser; mean = spanMeanUser; }
        else if ( n >= 2 and n <= 6 ) { most = spanCells[n]; mean = spanMean[n]; }
        return ( most > 0. and mean > 0. ) ? std::min( 1., mean / most ) : 1.;
    }
    // where the share holds: the runs it was fitted on (the GPU deposit of the 3-D single build with fixed-point
    // moments, sample points, a split, no window; a mixed run with half its partitions or more on the GPU, no face pass)
    bool laneShareOn(int n) const
    {
        return NO_DIM == 3 and metalActive and sizeof(Real) == 4
               and not Env::flag( "DTFE_MASS_AUDIT" ) and nPts > 0 and partsOf( n ) > 1. and not u.psWindowOn
               and gpuSubCells( n ) <= kGpuCellCap   // (a larger sub-grid is deposited on the CPU: choose()'s warning)
               and not ( mixed and ( n < 2 or faceR > 0. or u.psGpuShareMille < 500 ) );
    }
    int laneShareCap(int n) const { return laneMixed( n ) ? kPSLaneShareMax : kPSLaneShareMaxGpu; }
    // what each lane beside the first does not hold at the run's peak, bytes: partitionBytes' own-state term less the
    // same term at the share (0 <= cut <= that term; the tessellation, copies, HitSink blocks and GPU arrays stay whole)
    double laneShareCut(int n) const
    {
        double const subCells = subGridCells( n );
        double const f = std::max( kPSLaneShare, laneSpread( n ) );
        double const after = gpuAfterHoldCellBytes( n );                 // 0 without a face pass
        if ( laneMixed( n ) )
        {
            double cpu = 0., gpuPre = 0.;
            mixedLane( n, cpu, gpuPre );
            double const cpuShare = 1. - double( u.psGpuShareMille ) / 1000.;
            // the whole lane as partitionBytes books it, less the shared one as it was fitted (the host grids after
            // the lock are the first lane's: each other lane stays at its fitted share)
            return std::max( cpu, gpuPre + gpuAfterLockCellBytes( n )*subCells )
                   - std::max( f*cpuShare*cpu, gpuPre + f*after*subCells );
        }
        // (the face walk beside the deposit: each lane holds its face sums' words whole from before its device
        // call, so the first lane books them beside the hold, as partitionBytes, and each other one beside the hold's
        // share; the share after the hold holds them already. earlyFace is 0 without the walk, and a mixed run's face
        // pass takes no share: laneShareOn)
        double const hold = gpuCellBytes();
        return ( std::max( hold + earlyFace, after ) - std::max( kPSLaneShareHold*hold + earlyFace, f*after ) )
               * subCells;
    }
    // the full grids --scratch-dir leaves in RAM ('fixed' books none), less the point outputs 'fixed' books but the
    // loop never holds (allocated after it): what the share must still book, once
    double laneShareCover() const
    {
        if ( not scratchOn ) return 0.;
        return std::max( 0., std::min( gridTotal*bCellDeposit, kPSScratchRamGrids*scratchMinBytes ) - ptsOutputBytes );
    }
    // the TBB insert's copies of one partition, bytes, at inFlight partitions in flight: where the insert runs in
    // parallel (autoTuneParallelInsert: >= 2 threads per partition and a partition of at least the default rule's
    // parallelInsertMinPoints padded points, or forced by the flag or by DTFE_TRIANGULATION_THREADS). 0 elsewhere
    double parInsertBytes(int n, int inFlight) const
    {
        double const padded = N / partsOf( n ) * padFactorOf( n );
        return autoTuneParallelInsert( u, padded, inFlight ) ? padded * kParInsertBytes : 0.;
    }
    // the loop's prediction at c in flight under the share: finishReport's figure, the TBB copies included (as in the
    // split search's count: they can only lower the count the share adds)
    double laneSharePeak(int n, int c) const
    {
        return fixed + snapshotBytes( n ) + double(c)*( partitionBytes( n ) + parInsertBytes( n, c ) )
               - std::max( 0., double(c - 1)*laneShareCut( n ) - laneShareCover() );
    }

    // The GPU deposit's host snapshot (ps_metal_host.cc GpuSnapshot): to resume a failed command buffer the deposit
    // keeps a host copy of every buffer its kernel writes -- the device grids (gridB, with the 8 B fixed-point mass for
    // the 4 B float one: +4 B/cell), the per-tet masses (4 B/tet) and the exact items' weight totals (4 B/tet) -- taken
    // at each pass's start and every ~10 s of GPU work. It can be larger than the gate's 4 GB headroom. Not in
    // partitionBytes(): the deposits run one at a time (ctxMutex), so whatever the concurrency there is ONE such copy,
    // added once to every plan below.
    // --grid-window: every partition's sub-grid is the whole window (subGridCells), and the grids of the deposit
    // running are held once in the same way -- the device and host copies (gpuCellBytes) and, beside them, the
    // partition's own sub-grids (bCellDeposit: its density, counts and flags; fxCell + fxMom: the fixed-point words
    // while they are converted; fxFace: a face pass's words, or all it holds after the hold where that is more),
    // all freed and kept by the allocator for the next partition's deposit. In flight and waiting, a partition
    // holds only its tessellation (with the window's cells counted per partition, three in flight would book them
    // three times).
    // A mixed run's split (mixedLane) holds the device and host copies (gpuCellBytes) the same way: the deposit holding
    // the GPU has them, one at a time, so they are added here once and not per partition.
    // With the face walk beside the deposit (earlyFace) the window's face sums' words are not the deposit's alone: a
    // partition makes them before its device call and holds them through its wait for the device, so partitionBytes
    // books them for every partition in flight and they leave the term booked once here (bCellDeposit holds their
    // Real grids, the max their fxFace): one partition in flight books what it did without the walk. (A mixed run's
    // lane books them after its lock already, gpuAfterLockCellBytes, so its window term keeps them.)
    double snapshotBytes(int n) const
    {
        if ( not metalActive ) return 0.;
        double const subCells = subGridCells( n );   // the same sub-grid as partitionBytes()
        double const nTet = kTetsPerVertex * N / partsOf( n );
        double const walkOut = ( mixed and n >= 2 ) ? 0. : earlyFace;   // (partitionBytes' GPU branch books them)
        double const window = u.psWindowOn
            ? ( bCellDeposit + std::max( gpuCellBytes() + fxCell + fxMom + fxFace, gridB + 4. + fxFace + faceR )
                - walkOut )*subCells
            : 0.;
        double r = 0.;
        if ( mixed and n >= 2 and not u.psWindowOn )   // + the lock holder's device and host grids, once
            r = (gridB + 4.)*subCells + (psExact ? 12. : 4.)*nTet + window
                + gpuCellBytes()*subCells;
        else
            r = (gridB + 4.)*subCells + (psExact ? 12. : 4.)*nTet + window;
        return r;
    }

    // the GPU deposit's grids per sub-grid cell: gridB (auto_tune/ps_inputs.cc) -- every buffer in PSGpuGrids --
    // mirrored host and device (the x2), + 4 B/cell: the device's mass grid is a 64-bit fixed-point sum
    // (ps_metal_host.cc), + 4 B/cell more on the host, which receives those 8 B words instead of a 4 B float grid
    // (PSGpuGrids::massFx) -- unless a density is selected: the caller then allocates its density sub-grid (4 B; 8 in
    // the double build) only after the call, from the words, so the peak (the copy-back) is no higher than a float
    // grid's
    double gpuCellBytes() const
    {
        double const hostFx = ( ps.phaseSpace and not psDensity ) ? 4. : 0.;
        return 2.*gridB + 4. + hostFx;
    }

    // A GPU lane AFTER its hold of the device, per sub-grid cell, when its pass makes the face sums ('--face-gradient';
    // 0 otherwise). The device grids are gone by then; the host keeps what the deposit handed back (gridB + 4: the
    // mass and moment words), and only now come the face sums' words (ps_interpolation.cc psCpuFixedUnits, filled by
    // psFaceWalk on the CPU; fxFace + faceR). Then all of them are converted grid by grid into the partition's
    // sub-grids (psFixedToReal: a grid's Real values allocated, its words freed), the allocator keeping the freed
    // words. The sub-grids are the partition's own (bCellDeposit; one domain: the run's grids, in 'fixed', as in the
    // CPU branch), less what the conversion takes from freed words instead (afterHoldReuse). Every lane in flight
    // can be in that state at once -- the walk is CPU work, nothing serializes it -- so it is booked per lane, in
    // place of the hold where it is the larger. Without the face sums the lane books the hold
    // alone, and a lane with one partition in flight books its own sub-grids beside it once (loneLaneSubGrids): this
    // term is the face pass's alone.
    double gpuAfterHoldCellBytes(int n) const
    {
        if ( faceR <= 0. ) return 0.;
        double const own = ( n == 1 and not userPartitionOn and not scratchOn ) ? 0. : bCellDeposit;
        return own + gridB + 4. + fxFace + faceR - afterHoldReuse( subGridCells( n ) );
    }

    // A GPU lane of a MIXED run after its hold, per sub-grid cell (partitionBytes' mixed branch). The device grids are
    // gone; the host keeps the words the copy-back made (ps_metal_host.cc psHostCopyBack: the mass as 8 B words, the
    // moment words, the stream counts) until the pass has turned them into its sub-grids: gridB + 4, as
    // gpuAfterHoldCellBytes books the same words. An all-GPU lane books the whole hold, which holds them; a mixed lane
    // books the hold once (snapshotBytes), so it books them here, beside its arrays -- or a face pass's state behind
    // the hold, which holds the same words and is the larger (gpuAfterHoldCellBytes). --grid-window: snapshotBytes
    // holds them once (gridB + 4 in its window term).
    double gpuAfterLockCellBytes(int n) const
    {
        double const host = u.psWindowOn ? 0. : gridB + 4.;
        return std::max( host, gpuAfterHoldCellBytes( n ) );
    }

    // The Real bytes per sub-grid cell that the conversion after a face pass's GPU deposit takes from freed words and
    // not from new memory, for sub-grids of 'cells' cells: the single build with fixed-point moments, where the
    // device's words are the host's (swapped in) and psFixedToReal converts them one grid at a time -- the moment
    // grids in autoTuneFxGrids' order, the three face sums, the mass last. A freed block under kAllocKeepBytes goes
    // to a later request of about half its size or more, so the dispersion's Real grid goes into the velocity's
    // freed words, the gradient's into the dispersion's, the face velocities' into the gradient's. Counted here
    // only for a request of MORE than half the block (the smallest such block): a request of exactly half was
    // served at one sub-grid size and not at another. The smaller of the passes' sums (the face words are booked
    // for the larger pass); 0 in the double build, whose copy-back fills Real grids beside all the words.
    double afterHoldReuse(double const cells) const
    {
        if ( not ( ps.phaseSpace and sizeof(Real) == 4 ) ) return 0.;
        double const B4 = kAllocKeepBytes;      // the allocator returns a freed block of this size or more
        double const R = double(sizeof(Real));
        double least = -1.;
        for (Field *f : { &u.uField, &u.aField })
        {
            if ( autoTuneFaceBytes( *f, ps ) <= 0. ) continue;
            std::vector<double> kept;           // the freed words not yet taken, bytes per cell each
            double reused = 0.;
            auto convert = [&](double const real, double const words)
            {
                int best = -1;
                for (size_t i = 0; i < kept.size(); ++i)
                    if ( kept[i] >= real and kept[i] < 2.*real and ( best < 0 or kept[i] < kept[size_t(best)] ) )
                        best = int(i);
                if ( best >= 0 ) { reused += real; kept.erase( kept.begin() + best ); }
                if ( words*cells < B4 ) kept.push_back( words );
            };
            for (double const g : autoTuneFxGrids( *f, ps )) convert( g, 2.*g );
            for (double const c : { double(noVelComp), 1., 1. }) convert( 2.*NO_DIM*c*R, 2.*NO_DIM*c*8. );
            if ( f->density or f->velocity_tweb ) convert( R, 8. );
            least = least < 0. ? reused : std::min( least, reused );
        }
        return std::max( 0., least );
    }

    // The partition look-ahead (DTFE.cpp): at one partition in flight with the GPU deposit, the NEXT
    // partition's particle copy + tessellation are built during the deposit -- one more partition's
    // triangulation stage on top of the peak (the run skips a look-ahead the free memory cannot hold)
    double lookAheadBytes(int n, int c) const
    {
        if ( not metalActive or u.psNoOverlap or u.psLookAheadOffBudget or c != 1 or partsOf( n ) <= 1. ) return 0.;
        double const fPad = padFactorOf( n );
        return N / partsOf( n ) * fPad * (partBytes + kPSVertexBytes + kParInsert);
    }

    // ONE partition in flight on the GPU (partitionBytes' GPU branch, no face pass, no window): beside the device's
    // hold the partition also holds its own sub-grids (bCellDeposit), which the CPU branch (copyOne), the window path
    // (snapshotBytes) and a face pass (gpuAfterHoldCellBytes) book and this lane did not. With two or more in flight
    // the other lanes' bookings cover them.
    double loneLaneSubGrids(int n) const
    {
        if ( not metalActive or ( mixed and n >= 2 ) or u.psWindowOn or faceR > 0. or partsOf( n ) <= 1. ) return 0.;
        return bCellDeposit * subGridCells( n );
    }

    // the start-up peak of psReserveGrids under --scratch-dir (finishReport): every
    // scratch-backed full grid is zero-filled in anonymous memory before its file is mapped over it (ZeroFillScope),
    // the largest one on top of the particles and the read blocks the allocator still keeps. 0 when no grid is
    // scratch-backed, and on Linux, where the scope is a no-op (scratch_alloc.cc).
    double reserveZeroFillPeak() const
    {
#ifdef __APPLE__
        if ( not scratchOn or scratchMinBytes <= 0. or largestGridBytes < scratchMinBytes ) return 0.;
        return particleBytes + readKeptAll + largestGridBytes;
#else
        return 0.;
#endif
    }

    // what a given split and concurrency cost; the concurrency the run WILL use (0 = all cores)
    AutoTuneReport finishReport(int n, int c, bool const share = false) const
    {
        report.valid = true;
        report.partition = std::max( n, 1 );
        report.maxConcurrent = std::max( c, 1 );
        // partitions in flight: never more than there are (one domain x 10 threads is ONE domain, not ten)
        int const inFlight = int( std::min( double(report.maxConcurrent), partsOf( report.partition ) ) );
        if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // what the plan books per partition sub-grid
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS booked sub-grid: %.0f cells (split %d, %.0f partitions)\n",
                          subGridCells( report.partition ), report.partition, partsOf( report.partition ) );
        if ( mixed and report.partition >= 2 and Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // a mixed lane's terms
        {
            double cpu = 0., gpuPre = 0.;
            mixedLane( report.partition, cpu, gpuPre );
            double const once = u.psWindowOn ? 0. : gpuCellBytes()*subGridCells( report.partition );
            double const afterLock = gpuAfterLockCellBytes( report.partition )*subGridCells( report.partition );
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS mixed lane: cpu %.3f GB, gpu pre-lock %.3f GB, once %.3f GB, "
                          "after the lock %.3f GB\n", cpu/1.e9, gpuPre/1.e9, once/1.e9, afterLock/1.e9 );
        }
        if ( metalActive and faceR > 0. and Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // a face pass's GPU lane
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS GPU lane grids: hold %.0f B/cell, after the hold %.0f B/cell\n",
                          gpuCellBytes(), gpuAfterHoldCellBytes( report.partition ) );
        if ( earlyFace > 0. and Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // the face walk beside the GPU deposit
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS face walk beside the GPU deposit: face sums %.0f B/cell held "
                          "through the device call\n", earlyFace );
        // ... and those words of one partition, on the report line where they are booked (beside every lane's hold)
        report.faceWalk = earlyFace * subGridCells( report.partition );
        report.perPartition = partitionBytes( report.partition );
        // the lane share (memorySplit's plans, 'share'): the first lane whole, each other one laneShareCut less, so the
        // cut is booked once (laneOnce); every lane whole at one partition in flight and above the cap
        report.laneOnce = 0.;
        report.inFlightCap = 0;
        double laneCover = 0.;
        if ( share and inFlight >= 2 and laneShareOn( report.partition ) and inFlight <= laneShareCap( report.partition ) )
        {
            report.laneOnce = laneShareCut( report.partition );
            report.perPartition -= report.laneOnce;
            report.inFlightCap = laneShareCap( report.partition );
            laneCover = std::min( laneShareCover(), double(inFlight - 1) * report.laneOnce );   // never above whole lanes
        }
        if ( share and laneShareOn( report.partition ) and Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // memorySplit's plans
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS lane share: first lane %.3f GB, each other %.3f GB, spread %.3f, "
                          "share %.3f, scratch grids %.3f GB, used for 2 to %d in flight (%d planned)\n",
                          partitionBytes( report.partition )/1.e9,
                          ( partitionBytes( report.partition ) - laneShareCut( report.partition ) )/1.e9,
                          laneSpread( report.partition ), std::max( kPSLaneShare, laneSpread( report.partition ) ),
                          laneShareCover()/1.e9, laneShareCap( report.partition ), inFlight );
        // the TBB insert's copies, where the insert runs in parallel (parInsertBytes)
        report.perPartition += parInsertBytes( report.partition, inFlight );
        report.gpuSnapshot = snapshotBytes( report.partition );   // once: the GPU deposits run one at a time
        report.mixedLanes = mixed and report.partition >= 2;      // (gpuSnapshot is then the lock holder's grids too)
        // what 'predicted' (below) books once beside its lanes, in every run with the GPU deposit: the snapshot and the
        // lane share's laneOnce. The dynamic concurrency's ledger books the same (DTFE.cpp psOnceBytes): the first lane
        // whole, each other one -- one it adds beyond the plan's included, up to inFlightCap -- at perPartition
        // (not with '--scratch-dir' grids below the threshold in an all-GPU run, laneCover: the ledger has no term for
        // them, so every partition in flight books its whole lane there)
        report.onceBytes = ( laneCover > 0. and not report.mixedLanes ) ? 0. : report.gpuSnapshot + report.laneOnce;
        // the loop's dynamic concurrency (DTFE.cpp): what the run allocates after its partitions are in flight
        report.concChosen = autoConc;
        // (the regular part's twins and the stream average's divergence are made beside the deposit phase's grids, in
        // the normalization: bCellDeposit + normR there)
        report.reserveEnd = ( scratchOn ? 0. : gridTotal * std::max( 0., std::max( bCell, bCellDeposit + normR )
                                                                         - bCellDeposit ) )
                            + ptsOutputBytes;
        report.reserveFirstMerge = scratchOn ? 0. : gridTotal * ( autoTuneMergeGrownBytes( u.uField, ps )
                                                                 + autoTuneMergeGrownBytes( u.aField, ps ) );
        report.reserveRecords = ptsRecordBytes;
        // one partition in flight on the GPU: its own sub-grids beside the device's hold (loneLaneSubGrids)
        double const lone = inFlight == 1 ? loneLaneSubGrids( report.partition ) : 0.;
        if ( lone > 0. and Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS one partition in flight: its own sub-grids beside the device's "
                          "hold, %.0f B/cell x %.0f cells = %.3f GB\n", bCellDeposit,
                          subGridCells( report.partition ),
                          lone/1.e9 );
        double const base = fixed + report.gpuSnapshot + report.laneOnce + double(inFlight) * report.perPartition
                            + laneCover + lone;
        report.lookAhead = lookAheadBytes( report.partition, inFlight );
        // the split above is chosen without the look-ahead; adding it here anyway would make a plan that fits
        // report over_budget=1 (the launcher's memory check would halve a plane that fits) and run past a
        // budget set on purpose. Every return passes here, so this is where the
        // plan, its prediction and the run are made to agree: a look-ahead that does not fit beside one
        // partition is switched off for this run (DTFE.cpp reads psLookAheadOffBudget), and the prediction
        // is then the grids + particles + one partition. Over budget anyway, the look-ahead only adds to it.
        // (readKeptLoop, below, is in the predictions only: a plan may keep its look-ahead and report over budget by it)
        if ( report.lookAhead > 0. and base + report.lookAhead > budget )
        {
            message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " look-ahead off for this run: one "
                    << "partition (~" << (report.perPartition + lone)/1.e9 << " GB) plus the next one's tessellation (~"
                    << report.lookAhead/1.e9 << " GB) does not fit the " << (budget - fixed)/1.e9 << " GB the "
                    << "memory budget leaves beside the grids and particles; partitions are prepared in turn.\n"
                    << MESSAGE::Flush;
            u.psLookAheadOffBudget = true;
            report.lookAhead = 0.;
        }
        // the start-up peak of psReserveGrids under --scratch-dir (reserveZeroFillPeak): the loop term above still
        // chooses the split; this lifts the prediction, and so the over-budget check, where that transient is the
        // run's peak -- the 1024^3 class with few partitions in flight
        report.reservePeak = reserveZeroFillPeak();
        // the normalization: the regular part's twins and a face pass's stream-average
        // divergence beside the deposit phase's full grids (normR), no partition in flight; the loop term still
        // chooses the split, this lifts the prediction where it is the peak ('fixed' books bCell's grids: only what
        // normR adds beyond that counts)
        double const normPeak = normR > 0.
            ? fixed + ( scratchOn ? 0. : gridTotal * std::max( 0., bCellDeposit + normR - bCell ) ) : 0.;
        if ( normR > 0. and Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS normalization: the regular part's twins and the stream "
                          "average's divergence %.0f B/cell beside the deposit phase's %.0f (bCell %.0f), %.3f GB in "
                          "all\n", normR, bCellDeposit, bCell, normPeak/1.e9 );
        // the input reader's freed buffers that macOS's allocator keeps (autoTuneReadKeptBytes: readKeptAll), beside a
        // loop of one partition at a time (a split: one domain has no loop) on the GPU under --scratch-dir: the full
        // grids the scratch files hold take
        // none of those blocks, and no second lane's share covers them -- with two or more in flight the lane share's
        // fitted booking of the other lanes does (it was fitted on runs that held them). Booked whole: a measured
        // run held about as much in empty blocks near its peak. In the predictions only (with them the staged merge's
        // park room, predictedWholeLanes), as the start-up and normalization peaks are: the split, the count and the
        // look-ahead stay the loop's. (--grid-window books them in 'fixed': readKept)
        report.readKeptLoop = ( inFlight == 1 and partsOf( report.partition ) > 1. and metalActive
                                and not report.mixedLanes and scratchOn and not u.psWindowOn ) ? readKeptAll : 0.;
        if ( report.readKeptLoop > 0. and Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS one partition in flight under --scratch-dir: the read buffers "
                          "the allocator keeps beside the loop, %.3f GB\n", report.readKeptLoop/1.e9 );
        double const loopPeak = base + report.lookAhead + report.readKeptLoop;
        report.predicted = std::max( { loopPeak, report.reservePeak, normPeak } );
        // ... and with every lane whole (the staged merge's park room, DTFE.cpp): 'predicted' itself without the share
        report.predictedWholeLanes = std::max( { loopPeak - laneCover + double(inFlight - 1) * report.laneOnce,
                                                 report.reservePeak, normPeak } );
        if ( report.reservePeak > loopPeak )
            message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " the predicted peak is the start-up "
                    << "one: the particles, the read buffers kept and the zero fill of the largest scratch-backed grid "
                    << "(~" << report.reservePeak/1.e9 << " GB in all) exceed the partition loop's ~"
                    << loopPeak/1.e9 << " GB; the plan is the loop's"
                    << ( report.reservePeak > budget ? " -- and no split lowers it: it is over the budget whatever the"
                                                        " plan (a smaller grid, fewer fields, a larger DTFE_MEM_BUDGET_GB,"
                                                        " or Linux, where the zero fill goes to the file)" : "" )
                    << ".\n" << MESSAGE::Flush;
        return report;
    }

    // the cells of a partition's sub-grid in an n^3 split (psUseSubgrid): a Lagrangian sub-box maps to a DISPLACED,
    // distorted Eulerian region, so the crop is the touched span + slop, never above the axis -- valid on both
    // periodic and non-periodic runs since ps_interpolation.cc crops a periodic axis to its WRAPPED span instead of
    // keeping the whole axis
    double gpuSubCells(int n) const
    {
        // measured on the particles: a split's partitions crop; one domain never does (psUseSubgrid is the
        // partition path's), so n = 1 books the whole grid below whatever the particles span
        if ( isUserSplit( n ) and spanUser > 0. ) return spanUser;
        if ( not isUserSplit( n ) and n >= 2 and n <= 6 and spanCells[n] > 0. ) return spanCells[n];
        double c = 1.;
        for (int d=0; d<NO_DIM; ++d)
            c *= std::min( double(outGrid[d]), double(u.gridSize[d])/double( isUserSplit( n ) ? userK[d] : n ) + 40. );
        return c;
    }

    // the user's own split when it is not cubic: the planner names a split by n = u.partition[0], and these give that
    // split's partitions and padding per axis instead of n^NO_DIM (every other split is n per axis)
    bool isUserSplit(int n) const { return userNonCubic and not autoPart and n == userK[0]; }
    double partsOf(int n) const
    {
        double c = 1.;
        for (int d=0; d<NO_DIM; ++d) c *= double( isUserSplit( n ) ? userK[d] : n );
        return c;
    }
    double padFactorOf(int n) const
    {
        if ( not isUserSplit( n ) ) return padFactor( n );
        double f = 1.;
        for (int d=0; d<NO_DIM; ++d) f *= 1. + 2.*autoTunePadWidths( u, padFrac[d], userK[d], N, kPartitionPadFloor );
        return f;
    }

    // ... the cells it allocates: under --grid-window the WHOLE window, whatever the split (psSubgridBox,
    // ps_interpolation.cc: every partition's sub-grid is the window, merged window-local, DTFE.cpp) -- the crop
    // above is per axis, so a window spanning x and y of a 4096^2 slice would book 723^2 cells at 6^3, not 4096^2
    double subGridCells(int n) const { return u.psWindowOn ? gridTotal : gpuSubCells( n ); }

    // one partition's padding factor in an n^3 split (k spacings, or the Eulerian rule's floor)
    double padFactor(int n) const { return autoTunePadFactor( u, padFrac, n, N, kPartitionPadFloor ); }

    // the AUTO-TUNE line's device: ', GPU deposit', with the share a mixed run deposits on the GPU when the plan is
    // partitioned (one domain has no share: DTFE.cpp sets it to 1 after the plan); '' without the GPU deposit
    std::string deviceTag() const
    {
        if ( not metalActive ) return "";
        if ( not ( mixed and u.partitionOn ) ) return ", GPU deposit";
        char b[64];
        snprintf( b, sizeof(b), ", GPU deposit with the free cores (GPU share %.2f)", double(u.psGpuShareMille)/1000. );
        return b;
    }

    // The count the memory model gives the split n (memorySplit's search): as many partitions in flight as the budget
    // holds whole lanes for, at most targetC and the split's partitions; 0 where even one does not fit
    int countAt(int const n, int const targetC) const
    {
        double const m = partitionBytes(n);
        int cMem = int( std::floor( (budget - fixed - snapshotBytes(n)) / m ) );
        int c = std::min( { targetC, n*n*n, cMem } );
        // each partition's TBB insert copies where they run (parInsertBytes, which finishReport books at the partitions
        // in flight, never more than the split has): they end where a partition has fewer than 2 threads, so each count
        // is checked with its own, and only a count that does not fit with them is taken away
        auto const fitsWithCopies = [&](int const cc) {
            int const f = std::min( cc, int( partsOf( n ) ) );
            return fixed + snapshotBytes( n ) + double(f)*( m + parInsertBytes( n, f ) ) <= budget; };
        while ( c > 1 and not fitsWithCopies( c ) )
            --c;
        // one partition in flight (cMem's, or what the copies left) books finishReport's figure there (less the read
        // buffers kept, readKeptLoop, which are in the predictions only): its insert's
        // copies where the insert runs in parallel (a tessellation of parallelInsertMinPoints points or more with
        // 2 or more threads for it, or forced: parInsertBytes) and, on the GPU, its own sub-grids beside the hold. A
        // split whose one partition fits only without them is not taken -- the next finer split is tried -- unless the
        // lane share plans two in flight there
        if ( c == 1 and fixed + snapshotBytes(n) + m + parInsertBytes( n, 1 ) + loneLaneSubGrids(n) > budget
             and not ( autoConc and targetC >= 2 and laneShareOn( n ) and partsOf( n ) >= 2.
                       and laneShareCap( n ) >= 2 and laneSharePeak( n, 2 ) <= budget ) )
            c = 0;
        return c;
    }
    // ... and the lane share's raise of a split's count c (finishReport): more partitions in flight where only the
    // first is booked whole, up to the cap: the most in flight whose own prediction fits (laneSharePeak, the TBB
    // copies included, as in the split search's own count: neither leaves a count above one in flight whose loop
    // exceeds the budget). The copies end where a partition has fewer than 2 threads, so a count that does not fit
    // can lie below one that does
    int shareRaise(int const n, int const c, int const targetC) const
    {
        int raised = c;
        int const cMost = std::min( { targetC, int( partsOf( n ) ), laneShareCap( n ) } );
        for (int k = c + 1; k <= cMost; ++k)
            if ( laneSharePeak( n, k ) <= budget ) raised = k;
        return raised;
    }

    // The over-budget warnings' way out of the face gradient: where a pass makes the face
    // sums, what that route adds to the full-resolution grids per cell -- bCell less the same fields' without it --
    // and the switches that leave it out; "" where no pass makes them. Text only: no plan reads it.
    std::string faceGradientWayOut() const
    {
        if ( not ( faceR > 0. ) ) return "";
        AutoTunePS perTet = ps;
        perTet.faceGradient = false;
        double const perTetCell = std::max(
            autoTuneBytesPerCell( u.uField, perTet, 0 ) + autoTuneBytesPerCell( u.aField, perTet, 0 ),
            autoTuneBytesPerCell( u.uField, perTet, 1 ) + autoTuneBytesPerCell( u.aField, perTet, 1 ) );
        double const extra = std::max( 0., bCell - perTetCell );
        char b[96];
        snprintf( b, sizeof(b), "%.0f B/cell (%.3g GB)", extra, extra*gridTotal/1.e9 );
        return "the face gradient (" + u.faceGradientSource + ") adds " + b + " to the full-resolution grids and its "
               "face sums to every partition: '--per-tet-gradient' or DTFE_FACE_GRADIENT=0 leaves them out (the "
               "velocity gradient is then the stream average of each stream's own gradient)";
    }

    AutoTuneReport choose() const;
    std::optional<AutoTuneReport> irreducibleCase(int const userC, int const userN, int const nGpuMin,
                                                  bool &overBudget) const;
    std::optional<AutoTuneReport> smallSetCase(int const userC, int const userN, int const nGpuMin) const;
    std::optional<AutoTuneReport> overBudgetCase(int const userC, int const nGpuMin, bool const overBudget) const;
    AutoTuneReport memorySplit(int const userC, int const nGpuMin) const;
};

// Chooses the split and the concurrency: the GPU's sub-grid cap, then the first case that decides -- the irreducible
// term over budget, a small particle set, over budget at any split, else the memory-driven split.
inline AutoTuneReport AutoTunePSPlanner::choose() const
{
    int const userC = u.maxConcurrent > 0 ? u.maxConcurrent : cores;
    int const userN = u.partition.empty() ? 1 : int(u.partition[0]);

    // The GPU kernels number a partition's sub-grid cells with 32 bits: a sub-grid above 2^32 cells
    // (1626^3 unsplit) falls back to the CPU deposit (ps_interpolation.cc, averaged_interpolation_1.cc).
    // With the GPU on, every split chosen below keeps the sub-grid under that cap -- the same crop +
    // slop estimate as partitionBytes() -- so a very fine grid keeps its GPU deposit; a user's own
    // split that does not is told which one would (the pass itself warns too: psGpuEligible).
    double const gpuCellCap = kGpuCellCap;            // UINT32_MAX
    int nGpuMin = 1;
    if ( metalActive )
        while ( nGpuMin < 64 and gpuSubCells(nGpuMin) > gpuCellCap ) ++nGpuMin;
    if ( metalActive and not autoPart and gpuSubCells(userN) > gpuCellCap )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: with --partition " << userN << " each partition's sub-grid holds ~" << gpuSubCells(userN)
                << " cells, above the 2^32 the GPU kernels number: the deposit will run on the CPU. "
                << "--partition " << nGpuMin << " (or finer) keeps it on the GPU.\n" << MESSAGE::EndWarning;
    }

    bool overBudget = false;   // set below; the split is then chosen once partitionBytes() exists
    if ( auto const r = irreducibleCase( userC, userN, nGpuMin, overBudget ) ) return *r;
    if ( auto const r = smallSetCase( userC, userN, nGpuMin ) ) return *r;
    if ( auto const r = overBudgetCase( userC, nGpuMin, overBudget ) ) return *r;
    return memorySplit( userC, nGpuMin );
}

// The irreducible term (grids + particles + points) over budget: warns, and decides for a small set or a user's split.
inline std::optional<AutoTuneReport> AutoTunePSPlanner::irreducibleCase(int const userC, int const userN,
                                                                         int const nGpuMin, bool &overBudget) const
{
    if ( fixed > budget )
    {
        double const gridBytes = scratchOn ? 0. : gridTotal*bCell;
        // largest cubic grid whose accumulators still leave room for the particles and points
        double const room = budget - particleBytes - ptsBytes;
        long   const maxGrid = room > 0. ? long( std::cbrt( room / bCell ) ) : 0L;
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: this run does not fit in memory, and --partition cannot fix it.\n  ";
        if ( not scratchOn )
            warning << "The full-resolution output grids need " << gridBytes/1.e9 << " GB ("
                    << MESSAGE::printElements( u.gridSize, "x" ) << " cells x " << bCell << " B/cell for the "
                    << "selected fields), plus ";
        else
            warning << "With the grids on scratch disk, ";
        warning << (partBytes*N)/1.e9 << " GB of particles";
        if ( readKept > 0. )
            warning << ", " << readKept/1.e9 << " GB of read buffers the allocator keeps (--grid-window)";
        if ( nPts > 0 )
            warning << " and " << ptsBytes/1.e9 << " GB for the " << double(nPts) << " --sample-points ("
                    << ptsStreamsWhy << ")";
        warning << " = " << fixed/1.e9 << " GB, against a memory budget of " << budgetWhy << ".\n"
                << "  --partition splits the TRIANGULATION only: every Lagrangian partition deposits into "
                << "the same shared full-resolution grid and point set, so that part is irreducible.\n"
                << "  Options: ";
        if ( not scratchOn )
            warning << "(a) '--scratch-dir <local dir>' backs the grids with disk instead of RAM "
                    << "(needs ~" << gridBytes/1.e9 << " GB free on a local, non-synced volume, e.g. "
                    << "/private/tmp/dtfe-scratch); ";
        if ( maxGrid > 0 )
            warning << "(b) a smaller --grid -- in RAM, the largest that fits at these fields is about "
                    << maxGrid << "^3; ";
        else
            warning << "(b) a smaller --grid cannot help: the particles" << (nPts > 0 ? " and sample points" : "")
                    << " alone exceed the budget; ";
        warning << "(c) fewer --field entries (the gradient family costs 36 B/cell, "
                << "dispersion 24 B/cell); ";
        if ( nPts > 0 )
            warning << "(d) fewer --sample-points (a smaller image plane) or no --pts-vel-grad; ";
        warning << "(e) more RAM";
        // ... and the face gradient's grids, where the grids are in RAM
        if ( std::string const face = scratchOn ? std::string() : faceGradientWayOut(); not face.empty() )
            warning << "; (f) the per-tetrahedron gradient: " << face;
        warning << ". NOTE: '--partition-index'/'--region' are NOT a way out -- they crop particles by "
                << "Eulerian position and are rejected by this binary because the PS tessellation is Lagrangian.\n"
                << "  Running anyway with --max-concurrent 1 and the split with the smallest per-partition "
                << "triangulation; expect swapping.\n" << MESSAGE::EndWarning;
        if ( autoConc ) u.maxConcurrent = 1;
        // A small particle set triangulates cheaply in one piece; otherwise fall through to the
        // split choice below -- one UNPARTITIONED triangulation would put the whole
        // tessellation on top of an already over-budget run.
        if ( N < minN or not autoPart )
        {
            int const nOne = autoPart ? nGpuMin : userN;      // one domain, or the GPU cap's split
            if ( autoPart ) { u.partition.assign( NO_DIM, size_t(nOne) ); u.partitionOn = ( nOne > 1 ); }
            return finishReport( nOne, autoConc ? 1 : userC );
        }
        overBudget = true;
    }
    return std::nullopt;
}

// A small particle set that fits: the user's split, one domain, or a split for speed from the time model.
inline std::optional<AutoTuneReport> AutoTunePSPlanner::smallSetCase(int const userC, int const userN,
                                                                      int const nGpuMin) const
{
    char buf[512];
    // A small particle set triangulates cheaply in one piece -- unless that piece's DEPOSIT does not fit:
    // on a very fine grid (1700^3 from 0.26M particles) one domain's sub-grid accumulators and the
    // deposit threads' private copies are tens of GB, and only a split shrinks them. Such a run takes
    // the memory-driven split below instead.
    // (with the TBB insert's copies of the one in flight: one domain of 1e6 particles or more inserts in parallel)
    bool const smallFits = not autoPart or fixed + snapshotBytes( std::max( 1, nGpuMin ) )
                                           + partitionBytes( std::max( 1, nGpuMin ) )
                                           + parInsertBytes( std::max( 1, nGpuMin ), 1 ) <= budget;
    if ( N < minN and smallFits )     // small particle set AND the grid fits: one domain, or a split for SPEED
    {
        if ( not autoPart )
        {   // the user's split at the concurrency the report states: left unset, a hand-set '--partition 4 4 4'
            // would start all 64 partitions on a 10-core machine while the report says mc=10
            if ( autoConc ) u.maxConcurrent = cores;
            return finishReport( userN, autoConc ? cores : userC );
        }
        // The one stage a split parallelises is the sequential triangulation (+ vertex densities):
        // the deposit and the point evaluation already use every core inside one domain. A
        // triangulation-bound run (a slice of a small box) therefore gains from
        // 2^3 partitions, each triangulating a padded eighth on its own core; a deposit-bound one
        // (a 256^3 grid at 0.26M particles) does not, and loses the padding.
        // Predicted wall time per split, from per-unit costs measured on a 10-core Apple Silicon
        // (single-thread seconds):
        constexpr double tVertex = 3.6e-6, tSample = 3.5e-7, tPoint = 7.5e-7;
        double fCopy = 1.;                              // the single domain's periodic copies (none without --periodic)
        if ( u.periodic )
            for (int d=0; d<NO_DIM; ++d) fCopy *= 1. + 2.*padFrac[d];
        double const nSubCube = ( u.aField.selected() or psExact )
                              ? std::pow( double(std::max( 1, psExact ? 3 : int(u.psAvgSubsamples) )), double(NO_DIM) ) : 1.;
        double const depositWork = gridTotal * nSubCube * tSample + double(nPts) * tPoint;
        auto predictedWall = [&](int n, int c) -> double
        {
            double const parts = partsOf( n );
            double const fPad = padFactorOf( n );
            double const rounds = std::ceil( parts / double(c) );
            double const triangulate = ( n == 1 ? fCopy * N : fPad * N / parts ) * tVertex;
            double const threads = ( n == 1 ) ? double(cores) : double(std::max( 1, cores / c ));
            // a GPU deposit serialises on the one command queue: the split leaves it unchanged
            // and adds each partition's buffer setup, extraction and leftover passes
            double const deposit = metalActive ? depositWork / double(cores) : depositWork / parts / threads;
            double const overhead = ( n == 1 ) ? 0. : parts * ( N * 5.e-9 + 0.02 + ( metalActive ? 0.1 : 0. ) );
            return rounds * ( triangulate + deposit ) + overhead;
        };
        int bestN = 1, bestC = autoConc ? cores : userC;
        double const single = predictedWall( 1, cores );
        double bestT = single;
        if ( nGpuMin > 1 )                  // the GPU's sub-grid cap rules out one domain
        {
            bestN = nGpuMin;
            bestC = std::max( 1, std::min( { nGpuMin*nGpuMin*nGpuMin, cores,
                                               int( std::floor( (budget - fixed - snapshotBytes(nGpuMin)) / partitionBytes(nGpuMin) ) ) } ) );
            if ( not autoConc ) bestC = std::min( bestC, userC > 0 ? userC : cores );
            bestT = predictedWall( nGpuMin, bestC );
        }
        for (int n = std::max( 2, nGpuMin ); n <= std::max( 3, nGpuMin ); ++n)
        {
            int c = std::min( n*n*n, cores );
            int const cMem = int( std::floor( (budget - fixed - snapshotBytes(n)) / partitionBytes(n) ) );
            if ( cMem < 1 ) continue;
            c = std::min( c, cMem );
            if ( not autoConc ) c = std::min( c, userC > 0 ? userC : cores );
            double const t = predictedWall( n, c );
            if ( t < kSpeedSplitGain * bestT or ( bestN < nGpuMin ) ) { bestN = n; bestC = c; bestT = t; }   // a clear gain only
        }
        if ( bestN > 1 )
        {
            u.partition.assign( NO_DIM, size_t(bestN) );
            u.partitionOn = true;
            u.psOrderedMerge = true;   // uniform small partitions: merge in order, bit-reproducible at no cost
            u.psSpeedSplit = true;     // ... and no memory gate (DTFE.cpp; its own flag, the ordered merge
                                       // being every split's default)
            if ( autoConc ) u.maxConcurrent = bestC;
            if ( bestN == nGpuMin and nGpuMin > 1 )
                snprintf( buf, sizeof(buf),
                          "%s grid on the GPU: --partition %d %d %d --max-concurrent %d keeps each partition's "
                          "sub-grid under the 2^32 cells the GPU kernels number (~%.3g cells; predicted peak ~%.3g GB)",
                          MESSAGE::printElements( u.gridSize, "x" ).c_str(), bestN, bestN, bestN, u.maxConcurrent,
                          gpuSubCells(bestN), (fixed + snapshotBytes(bestN) + double(bestC)*partitionBytes(bestN))/1.e9 );
            else
                snprintf( buf, sizeof(buf),
                          "%.3g particles: triangulation-bound, so --partition %d %d %d --max-concurrent %d "
                          "(predicted %.2g s against %.2g s for one triangulation; predicted peak ~%.3g GB)",
                          N, bestN, bestN, bestN, u.maxConcurrent, bestT, single,
                          (fixed + snapshotBytes(bestN) + double(bestC)*partitionBytes(bestN))/1.e9 );
            message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
                    << ". Pass --partition 1 1 1 for one triangulation.\n" << MESSAGE::Flush;
            return finishReport( bestN, autoConc ? bestC : userC );
        }
        return finishReport( 1, autoConc ? cores : userC );
    }
    return std::nullopt;
}

// Over budget at any split: the split whose partition needs the least, one partition at a time.
inline std::optional<AutoTuneReport> AutoTunePSPlanner::overBudgetCase(int const userC, int const nGpuMin,
                                                                        bool const overBudget) const
{
    char buf[1024];
    if ( overBudget )
    {
        // over budget whatever we do (warned above): one partition at a time, split so that the
        // partition in flight is as small as the padding allows -- partitionBytes() stops falling
        // once the padding shells dominate, so take its minimum over the validated range
        int bestN = std::max( 2, nGpuMin );
        for (int n = bestN + 1; n <= std::max( 6, nGpuMin ); ++n)
            if ( partitionBytes(n) < partitionBytes(bestN) ) bestN = n;
        u.partition.assign( NO_DIM, size_t(bestN) );
        u.partitionOn = true;
        snprintf( buf, sizeof(buf),
                  "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> OVER BUDGET: --partition %d %d %d "
                  "--max-concurrent %d (smallest partition, ~%.3g GB; predicted peak ~%.3g GB%s)",
                  N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
                  bestN, bestN, bestN, u.maxConcurrent, partitionBytes(bestN)/1.e9,
                  (fixed + snapshotBytes(bestN) + partitionBytes(bestN))/1.e9, deviceTag().c_str() );
        message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
                << ". Pass --partition/--max-concurrent to override.\n           memory budget: "
                << budgetWhy << "\n" << MESSAGE::Flush;
        return finishReport( bestN, autoConc ? 1 : userC );
    }
    return std::nullopt;
}

// The memory-driven split: the coarsest split whose concurrency the budget allows, and the AUTO-TUNE line stating it.
inline AutoTuneReport AutoTunePSPlanner::memorySplit(int const userC, int const nGpuMin) const
{
    char buf[1024];
    // Concurrency target = the core count, GPU or not. The GPU deposits do serialize on one
    // queue (ps_metal_host.cc ctxMutex), but the dominant per-partition cost is the CPU work
    // AROUND them -- Lagrangian triangulation + vertex densities, most of a partition's time --
    // which runs fully in parallel: a cap of 3 would run 8 partitions in three waves, slower than
    // the CPU deposit alone. Memory still bounds it:
    // partitionBytes() counts the GPU buffers a waiting partition holds, and cMem caps c, with the TBB insert's copies
    // where c leaves 2 threads a partition (parInsertBytes; at one in flight too, where a split that fits only without
    // them gives way to a finer one). Where only the first lane is booked whole (laneShareCut), the concurrency rises
    // past cMem up to laneShareCap.
    int const targetC = cores;

    int bestN = -1, bestC = 0;
    int const nLo = autoPart ? std::max( 2, nGpuMin ) : int(u.partition[0]);
    int const nHi = autoPart ? std::max( 6, nGpuMin ) : int(u.partition[0]);   // <=6^3 keeps auto within validated territory (unless the GPU cap needs more)
    for (int n=nLo; n<=nHi; ++n)
    {
        int const c = countAt( n, targetC );
        if ( c > bestC or bestN<0 )
        {
            bestN = n;
            bestC = c;
        }
        if ( c >= std::min(targetC, n*n*n) )
            break;      // memory is not the binding constraint; smallest such n wins (fewest seams)
    }
    if ( bestC < 1 )
    {
        // 'fixed' fits (checked above), so this is the triangulation genuinely not fitting in
        // what is left -- here the split IS the right lever, and the finest one is the best try.
        bestC = 1;
        bestN = nHi;
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: even a single " << bestN << "^3-split partition's triangulation is predicted to "
                << "exceed the " << (budget - fixed)/1.e9 << " GB left after the full-grid accumulators ("
                << (gridTotal*bCell)/1.e9 << " GB) and particles (" << (partBytes*N)/1.e9 << " GB); running with "
                << "--max-concurrent 1 at the finest split. Expect swapping; a finer --partition than "
                << nHi << "^3 may still help";
        // ... and so may the per-tetrahedron gradient, which leaves out the face sums
        if ( std::string const face = faceGradientWayOut(); not face.empty() )
            warning << ", and so may the per-tetrahedron gradient: " << face;
        warning << ".\n" << MESSAGE::EndWarning;
    }
    else if ( autoConc and laneShareOn( bestN ) )
        bestC = shareRaise( bestN, bestC, targetC );   // the lane share (finishReport): more in flight, up to its cap

    if ( autoPart )
    {
        u.partition.assign( NO_DIM, size_t(bestN) );
        u.partitionOn = (bestN > 1);
    }
    if ( autoConc )
        u.maxConcurrent = bestC;

    // the concurrency the run WILL use: the tuner's, or the user's own (0 = all cores)
    int const effC = autoConc ? bestC : userC;
    // the report first: it decides whether the look-ahead fits the budget, and the message states its prediction
    AutoTuneReport const decided = finishReport( bestN, effC, true );
    double const predicted = decided.predicted;
    char scratchNote[256] = "";
    if ( scratchOn )
        snprintf( scratchNote, sizeof(scratchNote), "; grid accumulators (%.3g GB) on scratch disk",
                  gridTotal*bCell/1.e9 );
    char ptsNote[320] = "";
    if ( nPts > 0 )
        snprintf( ptsNote, sizeof(ptsNote), "; %.3g sample points (%.3g GB: %s)",
                  double(nPts), ptsBytes/1.e9, ptsStreamsWhy.c_str() );
    char walkNote[160] = "";   // (the face walk beside the GPU deposit: only where its words are booked)
    if ( decided.faceWalk > 0. )
        snprintf( walkNote, sizeof(walkNote), "; face sums held through each device call (their walk runs beside "
                  "it): %.3g GB per partition", decided.faceWalk/1.e9 );
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %s --max-concurrent %d "
              "(predicted peak ~%.3g GB%s%s%s%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              MESSAGE::printElements( u.partition, " " ).c_str(), u.maxConcurrent,     // NO_DIM numbers
              predicted/1.e9, deviceTag().c_str(), scratchNote, ptsNote, walkNote );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
            << autoTuneOverrideNote( autoPart, autoConc ) << "\n           memory budget: "
            << budgetWhy << "\n" << MESSAGE::Flush;
    return decided;
}
