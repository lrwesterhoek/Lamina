// Part of src/DTFE.cpp, inside its #ifdef PHASE_SPACE: included once, in order, by that file and compiled only as
// part of its object, never alone.
// The PS partitioned run's setup: the cloud's normalization, the Lagrangian periodic setup, the window, the
// composite serve, the grids, the window's skips and the deferred particle read (psReadDeferredParticles).

// PS-DTFE, neither periodic nor --region: the cloud's Lagrangian mean density and the --ps-alpha-shape radius.
static void psCloudNormalization(DTFE_State &state, User_options &userOptions)
{
    if ( not userOptions.periodic and not userOptions.regionOn )
    {
        // non-periodic isolated cloud: normalize to the cloud Lagrangian mean N*m/V_lagBox
        // (box mean N*m/L^3 undercounts a cloud filling part of the box; --region keeps the box mean)
        if ( not state.particles.empty() )
        {
            Box lagBoxNP;
            for (int d = 0; d < NO_DIM; ++d)
            { lagBoxNP[2*d] = state.particles[0].lagPos[d]; lagBoxNP[2*d+1] = state.particles[0].lagPos[d]; }
            double totalMass = 0.;
            for (size_t i = 0; i < state.particles.size(); ++i)
            {
                totalMass += double(state.particles[i].weight());
                for (int d = 0; d < NO_DIM; ++d)
                {
                    Real q = state.particles[i].lagPos[d];
                    if (q < lagBoxNP[2*d])   lagBoxNP[2*d]   = q;
                    if (q > lagBoxNP[2*d+1]) lagBoxNP[2*d+1] = q;
                }
            }
            double vLag = 1.;
            for (int d = 0; d < NO_DIM; ++d) vLag *= double(lagBoxNP[2*d+1] - lagBoxNP[2*d]);
            if ( vLag > 0. )
            {
                state.options.averageDensity = Real( totalMass / vLag );
                MESSAGE::Message msg( userOptions.verboseLevel );
                msg << MESSAGE::cBold() << "PS-DTFE (non-periodic):" << MESSAGE::cReset()
                    << " density normalized to the Lagrangian cloud density N*m/V_lagBox = "
                    << MESSAGE::cMagenta() << state.options.averageDensity << MESSAGE::cReset() << ".\n" << MESSAGE::Flush;
                // --ps-alpha-shape: the Lagrangian domain is the cloud's alpha shape (psFilterCell)
                if ( state.options.psAlphaShape > Real(0.) )
                {
                    double const spacing = std::cbrt( vLag / double(state.particles.size()) );
                    state.options.psLagAlphaRadius = double(state.options.psAlphaShape) * spacing;
                    msg << MESSAGE::cBold() << "PS-DTFE (non-periodic):" << MESSAGE::cReset()
                        << " Lagrangian domain = alpha shape, tetrahedra with circumradius <= "
                        << MESSAGE::cMagenta() << state.options.psLagAlphaRadius << MESSAGE::cReset() << " ("
                        << state.options.psAlphaShape << " mean spacings of " << spacing << "; --ps-alpha-shape).\n" << MESSAGE::Flush;
                }
            }
        }
    }
}

// PS-DTFE, periodic: unwraps, finds the Lagrangian box and seam, and adds the periodic copies (or defers them).
static void psLagrangianPeriodicSetup(DTFE_State &state, User_options &userOptions, Box &lagBoxGlobal,
                                      bool &hasLagrangianPeriodicCopies)
{
    if ( userOptions.periodic && !state.particles.empty() )
    {
        MESSAGE::Message msg( userOptions.verboseLevel );

        computeLagrangianBoundingBox( state.particles, lagBoxGlobal );

        // Eulerian box length per axis, used both for unwrapping and for shifting periodic copies
        Real eulerLen[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            eulerLen[d] = state.options.boxCoordinates[2*d+1] - state.options.boxCoordinates[2*d];

        // Lagrangian padding: same fraction as the Eulerian padding, with safety floors
        Real lagLength[NO_DIM], lagPad[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            lagLength[d] = lagBoxGlobal[2*d+1] - lagBoxGlobal[2*d];
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLen[d]);
            lagPad[d] = padFrac * lagLength[d];
            // floor at 10% of the box so the tessellation still tiles when the Eulerian padding is tiny
            if (lagPad[d] < lagLength[d] * Real(0.10))
                lagPad[d] = lagLength[d] * Real(0.10);
            // global copies must reach the per-partition padding (0.30 of a cell) so edge partitions stay covered
            if (userOptions.partitionOn && userOptions.partNo < 0)
            {
                Real partPad = lagLength[d] / Real(userOptions.partition[d]) * Real(0.30);
                if (lagPad[d] < partPad)
                    lagPad[d] = partPad;
            }
        }

        // unwrap Eulerian positions so each displacement s = pos - lagPos lies in [-L/2, L/2];
        // otherwise boundary particles get displacements near L and huge spurious simplices
        psUnwrapEulerian( state.particles, state.options );

        // the globals a later --grid-window run can start from without reading the particles
        // (ps_globals_record.h): written by every partitioned run with a grid and a tessellation cache whose
        // particles were read as they are and the mean density computed from them (a run that started without
        // its particles has neither: its mean density is the record's)
        if ( not state.options.tessellationCacheDir.empty() and userOptions.partitionOn and userOptions.partNo < 0
             and userOptions.psAverageDensityFromParticles and not userOptions.regionOn and userOptions.poisson == 0
             and userOptions.randomSample < Real(0.) and state.options.gridSize.size() == size_t(NO_DIM) )
        {
            PSGlobals::Record rec;
            rec.n = state.particles.size();
            rec.averageDensity = double( state.options.averageDensity );
            for (int i = 0; i < 2*NO_DIM; ++i)
            {
                rec.box[i] = double( state.options.boxCoordinates.coords[i] );
                rec.lagBox[i] = double( lagBoxGlobal.coords[i] );
            }
            rec.hubble = double( state.options.hubbleParam );
            rec.scaleFactor = double( state.options.scaleFactor );
            PSGlobals::write( state.options, rec );
        }

        msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
            << " Unwrapped Eulerian positions (displacements within [-L/2, L/2]).\n" << MESSAGE::Flush;

        hasLagrangianPeriodicCopies = true;

        // non-partition path builds one triangulation, so it needs all periodic images up front;
        // the partition path generates images per-partition and defers this whole-box copy array
        // to keep peak memory low
        bool const psPartitioned = ( userOptions.partitionOn and userOptions.partNo < 0 );
        // the one triangulation's ownership period per axis, [seam - L, seam): where its padding allows, the
        // seam whose periodic copies are all EXACT shifts of their originals (subpartition.h psPeriodicSeam --
        // the tessellation around the two edges of the period is then the same up to L, and the centroid
        // ownership tiles it exactly); else the fallback anchor, the top particle plane. The
        // partition path tiles its own (PSPartitionPlan, with the partitions' padding)
        Real seamG[NO_DIM];
        bool seamExactG[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            seamExactG[d] = psPeriodicSeam( lagBoxGlobal[2*d], lagBoxGlobal[2*d+1], eulerLen[d], lagPad[d], seamG[d] );
        if ( not psPartitioned )
        {
            // the padded period: [seam - L - pad, seam + pad] -- the originals and the -L copies of those above
            // seam - pad, no +L copy (the copy loop below makes the copies that fall inside it); the fallback seam
            // keeps the bounding box +- pad
            Box lagPaddedGlobal = lagBoxGlobal;
            for (int d=0; d<NO_DIM; ++d)
            {
                if ( seamExactG[d] )
                {
                    lagPaddedGlobal[2*d]   = seamG[d] - eulerLen[d] - lagPad[d];
                    lagPaddedGlobal[2*d+1] = seamG[d] + lagPad[d];
                    continue;
                }
                lagPaddedGlobal[2*d]   -= lagPad[d];
                lagPaddedGlobal[2*d+1] += lagPad[d];
            }
            psReportSeam( msg, seamG, seamExactG, eulerLen, lagPad, state.particles );

            // add periodic copies from all 26 (3D) or 8 (2D) images; Lagrangian and Eulerian
            // positions shift by the same box-length offset for correct simplex geometry
            size_t const noOriginal = state.particles.size();
            size_t const noOffsets = (NO_DIM == 2 ? 9 : 27);
            size_t const centerIdx = (NO_DIM == 2 ? 4 : 13);

            for (size_t n = 0; n < noOffsets; ++n)
            {
                if (n == centerIdx) continue;  // skip zero offset

                Real lagOffset[NO_DIM], eulerOffset[NO_DIM];
                size_t rem = n;
                for (int d = NO_DIM - 1; d >= 0; --d)
                {
                    int sign = int(rem % 3) - 1;  // -1, 0, or +1
                    // both use the period eulerLen; a bounding-box width would under-shift
                    // copies onto boundary originals and corrupt the tiling
                    lagOffset[d]   = Real(sign) * eulerLen[d];
                    eulerOffset[d] = Real(sign) * eulerLen[d];
                    rem /= 3;
                }

                for (size_t i = 0; i < noOriginal; ++i)
                {
                    bool inside = true;
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        Real shiftedLag = state.particles[i].lagPos[d] + lagOffset[d];
                        if (shiftedLag < lagPaddedGlobal[2*d] || shiftedLag > lagPaddedGlobal[2*d+1])
                        { inside = false; break; }
                    }
                    if (inside)
                    {
                        Particle_data copy = state.particles[i];
                        for (int d = 0; d < NO_DIM; ++d)
                        {
                            copy.lagPos[d] += lagOffset[d];
                            copy.pos[d]    += eulerOffset[d];
                        }
                        state.particles.push_back(copy);
                    }
                }
            }

            msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
                << MESSAGE::cMagenta() << lagBoxGlobal.print() << MESSAGE::cReset() << "\n"
                << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Added "
                << MESSAGE::cMagenta() << (state.particles.size() - noOriginal) << MESSAGE::cReset()
                << " Lagrangian periodic copies (" << MESSAGE::cMagenta() << noOriginal << MESSAGE::cReset()
                << " original particles).\n" << MESSAGE::Flush;
        }
        else
        {
            msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
                << MESSAGE::cMagenta() << lagBoxGlobal.print() << MESSAGE::cReset() << "\n"
                << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                << " periodic copies are generated per-partition (deferred) to keep peak memory low.\n" << MESSAGE::Flush;
        }

        // primary periodic box: the centroid ownership check keeps one image of each cell and tiles the box;
        // overridden per partition in the partition path. It is ONE PERIOD, [seam - L, seam), not the snapshot box
        // [0, L): on a lattice the box edge lies halfway between particle planes, so a straddling tet's centroid is
        // ON the edge and its float-rounded copies can put both images outside (mass lost on a single-precision
        // lattice; tests/tweb_check.sh). The period ends at the exact seam (subpartition.h psPeriodicSeam)
        // where the padding allows, at the top particle plane otherwise.
        for (int d=0; d<NO_DIM; ++d)
        {
            state.options.lagrangianRegion[2*d]   = seamG[d] - eulerLen[d];
            state.options.lagrangianRegion[2*d+1] = seamG[d];
        }
    }
    else if ( userOptions.periodic && userOptions.psParticlesDeferred )
    {
        // started without the particles (ps_globals_record.h): the recorded globals stand in for the
        // block above -- the bounding box of the initial positions, and (partitioned runs only) the
        // periodic copies generated per partition when a partition's particles are selected
        lagBoxGlobal = userOptions.psPresetLagBox;
        hasLagrangianPeriodicCopies = true;
        state.options.lagrangianRegion = state.options.boxCoordinates;
        MESSAGE::Message msg( userOptions.verboseLevel );
        msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
            << MESSAGE::cMagenta() << lagBoxGlobal.print() << MESSAGE::cReset() << " (recorded; the particles are not read yet).\n"
            << MESSAGE::Flush;
    }
}

// --grid-window: computes the window's cell range from the final box and grid, for both copies of the options.
static void psResolveWindow(DTFE_State &state, User_options &userOptions)
{
    // --grid-window: its cell range needs the final box and grid (the header's box is known only now).
    // Both the partition path (state.options) and the single triangulation (state.options via
    // DTFE_parallel) read it from state.options; the writers read the caller's copy, so both get it
    if ( state.options.psWindowOn )
    {
        state.options.computePsWindow();
        for (int d = 0; d < NO_DIM; ++d)
        {
            userOptions.psWindowLo[d]   = state.options.psWindowLo[d];
            userOptions.psWindowDims[d] = state.options.psWindowDims[d];
        }
        // NO_DIM entries: a fixed 3 would read past the end of the 2D build's 2-entry arrays
        std::string dims, lo;
        for (int d = 0; d < NO_DIM; ++d)
        {
            dims += ( d ? "x" : "" ) + std::to_string( state.options.psWindowDims[d] );
            lo   += ( d ? ", " : "" ) + std::to_string( state.options.psWindowLo[d] );
        }
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "--grid-window: the output grid is the window of " << dims << " cells at (" << lo
                << ") of the " << MESSAGE::printElements( state.options.gridSize, "x" ) << " grid.\n" << MESSAGE::Flush;
    }
}

// '--serve --partition', PS-DTFE: every Lagrangian partition to the composite server, which serves; no return.
template <class PartitionOptions>
static void psServeComposite(DTFE_State &state, User_options &userOptions, PSPartitionPlan &plan,
                             PartitionOptions &partitionOptions)
{
    int concurrency = plan.total;
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < concurrency )
        concurrency = userOptions.maxConcurrent;
#ifdef OPEN_MP
    concurrency = std::max( 1, std::min( concurrency, dtfeOmpMax() ) );
#endif
    psServeCompositeBegin( state.options, plan.total );
    ConcurrentTriangulationPlan triangulationPlan( concurrency, plan.total );
#ifdef OPEN_MP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(concurrency)
#endif
    for (int pi=0; pi<plan.total; ++pi)
    {
    ConcurrentTriangulationGuard concurrentTriangulation;
        Box lagRegion, lagPadded;
        plan.regions( pi, lagRegion, lagPadded );
        // the batch loop's per-partition options (dtfe/ps_partitioned_run.cc), so the cache files are shared
        User_options tempOpt = partitionOptions( lagRegion );
        int tnum = 0;
#ifdef OPEN_MP
        tnum = omp_get_thread_num();
#endif
        tempOpt.verboseLevel = (tnum == 0) ? std::min( userOptions.verboseLevel, 1 ) : 0;
        psServeCompositeAddPartition( pi, tempOpt,
            [&](std::vector<Particle_data> &out)
            {
                findParticlesInBoxLagrangianPeriodic( state.particles, &out, lagPadded,
                                                      plan.eulerLen, plan.periodicCopies, 0, plan.selectionExtent() );
            }, nullptr, nullptr );
    }
    std::vector<Particle_data>().swap( state.particles );
    psServeCompositeRun( state.options );   // answers requests until stdin closes; never returns
}

// Sizes the main grids: the selected fields and the PS accumulators the options ask for (streams, hidden streams,
// caustic bits, exact touches), one pass over the field list per side (Quantities::reserveMemory).
static void psReserveGrids(DTFE_State &state, Quantities *uQuantities, Quantities *aQuantities)
{
    size_t mainGrid[NO_DIM];                       // the main grid: the --grid-window cells, else the full grid
    state.options.outputGridSize( mainGrid );
    // pre-sized NOW, before the partition loop, so they are allocated when its memory gate reads the free memory
    // (and with '--scratch-dir' counted on the line below). Not for the merge's sake: it grows a grid it finds
    // unsized at the first values it adds (quantities.cc addFromSubgrid), which is how the moment weights come
    uQuantities->reserveMemory( mainGrid, state.options.uField, &state.options );
    aQuantities->reserveMemory( mainGrid, state.options.aField, &state.options );
    if ( not state.options.scratchDir.empty() )
    {
        // tests/ps_scratch_check.sh (section A) parses "<N> zero-filled in memory" from this line: keep the wording
        MESSAGE::Message message( state.options.verboseLevel );
        message << "SCRATCH: " << ScratchAlloc::scratchTotalAllocs() << " grid(s), "
                << ScratchAlloc::scratchLiveBytes() / 1.e9 << " GB, on scratch files; "
                << ScratchAlloc::scratchZeroFillMoves() << " zero-filled in memory and moved onto their file unwritten.\n"
                << MESSAGE::Flush;
    }
}

// --grid-window with a tessellation cache: skips the partitions whose occupancy maps never reach the window.
template <class PartitionOptions>
static void psWindowSkips(DTFE_State &state, PSPartitionPlan &plan, PartitionOptions &partitionOptions,
                          int &totalPartitions, MESSAGE::Message &message, std::vector<char> &skipPartition,
                          int &nSkipped)
{
    if ( state.options.psWindowOn and not state.options.tessellationCacheDir.empty() )
    {
        int nUnknown = 0;
        for (int pi=0; pi<totalPartitions; ++pi)
        {
            Box lagRegion, lagPadded;
            plan.regions( pi, lagRegion, lagPadded );
            int const t = psOccupancyTouchesWindow( partitionOptions( lagRegion ), state.options );
            if ( t == 0 ) { skipPartition[size_t(pi)] = 1; ++nSkipped; }
            if ( t < 0 ) ++nUnknown;
        }
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " --grid-window: "
                << MESSAGE::cMagenta() << nSkipped << MESSAGE::cReset() << " of " << totalPartitions
                << " partitions never reach the window (their occupancy maps in the tessellation cache) and are skipped; "
                << (totalPartitions - nSkipped) << " to process"
                << ( nUnknown > 0 ? " (" + std::to_string(nUnknown) + " without a map yet)" : std::string() )
                << ".\n" << MESSAGE::Flush;
    }
}

// A run started without its particles: reads them when a partition the window needs is not cached.
template <class PartitionOptions>
static void psReadDeferredParticles(DTFE_State &state, User_options &userOptions, PSPartitionPlan &plan,
                                    PartitionOptions &partitionOptions, int &totalPartitions,
                                    MESSAGE::Message &message, std::vector<char> &skipPartition)
{
    // started without the particles: every partition the window needs must be cached, or they are
    // read now (the run is then the ordinary one) -- verified against the record before use
    if ( state.options.psParticlesDeferred )
    {
        int nMissing = 0;
        for (int pi=0; pi<totalPartitions; ++pi)
        {
            if ( skipPartition[size_t(pi)] ) continue;
            Box lagRegion, lagPadded;
            plan.regions( pi, lagRegion, lagPadded );
            if ( not psTessellationCached( partitionOptions( lagRegion ) ) ) ++nMissing;
        }
        if ( nMissing == 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " every partition the window needs is "
                    << "cached: the snapshot is not read (its globals come from the record).\n" << MESSAGE::Flush;
        else
        {
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " " << nMissing
                    << " partition(s) the window needs are not cached: reading the particles after all.\n" << MESSAGE::Flush;
            User_options readOpts = userOptions;
            readOpts.psParticlesDeferred = false;
            std::vector<Sample_point> noSamples;
            readInputData( &state.particles, &noSamples, &readOpts );
            bool same = state.particles.size() == userOptions.psPresetN;
            for (int i = 0; i < 2*NO_DIM and same; ++i)
                same = readOpts.boxCoordinates.coords[i] == state.options.boxCoordinates.coords[i];
            if ( not same )
                throwError( "the globals record '", PSGlobals::path( userOptions ), "' does not match the snapshot "
                            "it names (another particle count or box); delete it -- the next run rewrites it." );
            psUnwrapEulerian( state.particles, state.options );
            state.options.psParticlesDeferred = false;
        }
    }
}
