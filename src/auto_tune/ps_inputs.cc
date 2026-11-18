// Part of src/auto_tune.cc, inside its #ifdef PHASE_SPACE and its anonymous namespace: included once, in order, by
// that file and compiled only as part of its object, never alone.
// The PS planner's inputs: the partition spans, the sample points' streams and bytes, the scratch space check,
// one thread's copy per cell, and AutoTunePSInputs.

/* The cells of the largest partition sub-grid of each n^NO_DIM split, n = 1 .. 6 (cells[n]), and of the user's own
   split k[] when it is not cubic (cellsUser), from the particles (0: not known), and the mean over the
   split's partitions (mean[n], meanUser): the lane share's floor (laneSpread). The binary crops a
   partition's grids to the Eulerian span of its kept tetrahedra per axis (psSubgridBox, ps_interpolation.cc), and that
   span grows as structure forms. (A single domain never crops: the planner books the whole grid there,
   gpuSubCells.)
   The partitions are equal slices of the Lagrangian box -- on a periodic run of the period [seam - L, seam), the seam
   placed by the partition padding (DTFE.cpp regions(), lagPadding; subpartition.h psPeriodicSeam, replayed below
   without its exactness test) -- owned by the tetrahedra's Lagrangian centroids. So per split: the particles binned on
   a B^NO_DIM grid over that Lagrangian domain (a partition's edge rounded to the nearest bin), each bin's Eulerian
   extent per axis in grid cells (periodic: each particle the minimal image of the bin's first one, as the deposit
   wraps a tetrahedron about its first vertex); a partition's touched planes = its bins' extents widened below for the
   tetrahedra; then psSubgridBox's rule (their [lo, hi], or on a periodic axis touched at both ends the complement of
   the largest untouched run), at most the axis. */
inline void psPartitionSpans(std::vector<Particle_data> const &p, User_options &u, double const padFrac[NO_DIM],
                             int const *userK, double (&cells)[7], double &cellsUser, double (&mean)[7],
                             double &meanUser)
{
    for (double &c : cells) c = 0.;
    cellsUser = 0.;
    for (double &c : mean) c = 0.;
    meanUser = 0.;
#ifdef PHASE_SPACE
    if ( p.empty() ) return;
    double const N = double( p.size() );
    // bins per axis: about a lattice cell's diagonal of particles each, at most 60
    int const B = std::max( 6, std::min( 60, int( std::pow( N, 1. / double(NO_DIM) ) / std::sqrt( double(NO_DIM) ) ) ) );
    double bbLo[NO_DIM], bbHi[NO_DIM], boxLo[NO_DIM], boxLen[NO_DIM], dx[NO_DIM];
    int G[NO_DIM];
    double vol = 1.;
    for (int d=0; d<NO_DIM; ++d)
    {
        double lo = double( p[0].lagPos[d] ), hi = lo;
        for (Particle_data const &q : p) { double const v = double( q.lagPos[d] ); lo = std::min( lo, v ); hi = std::max( hi, v ); }
        double const eps = ( hi - lo ) * 1.e-6;   // computeLagrangianBoundingBox's margin
        bbLo[d] = lo - eps;
        bbHi[d] = hi + eps;
        boxLo[d] = double( u.boxCoordinates[2*d] );
        boxLen[d] = double( u.boxCoordinates[2*d+1] ) - boxLo[d];
        G[d] = int( u.gridSize[d] );
        dx[d] = boxLen[d] / double( G[d] );
        if ( not ( dx[d] > 0. ) or G[d] < 1 or not ( bbHi[d] > bbLo[d] ) ) return;
        vol *= u.periodic ? boxLen[d] : bbHi[d] - bbLo[d];
    }
    // the partition padding of a split with k partitions on axis d (DTFE.cpp): --ps-lagrangian-padding k spacings (capped at a
    // period), else the Eulerian --padding fraction floored at 0.3 of a partition; >= 4.2 alpha cuts
    double const lagSpacing = u.psLagPadding > Real(0.) ? std::pow( vol / N, 1. / double(NO_DIM) ) : 0.;
    auto padOf = [&](int d, int k) {
        double const len = bbHi[d] - bbLo[d];
        double pad = std::max( padFrac[d] * len, len / double(k) * 0.3 );
        if ( lagSpacing > 0. )
        {
            pad = double( u.psLagPadding ) * lagSpacing;
            if ( u.periodic and pad > boxLen[d] ) pad = boxLen[d];
        }
        if ( u.psLagAlphaRadius > 0. ) pad = std::max( pad, 4.2 * double( u.psLagAlphaRadius ) );
        return pad;
    };
    // the Lagrangian domain the partitions tile, per axis: the period below the seam, or the bounding box
    auto domainOf = [&](int d, int k, double &lo0, double &len0) {
        if ( not u.periodic ) { lo0 = bbLo[d]; len0 = bbHi[d] - bbLo[d]; return; }
        double const L = boxLen[d], pad = padOf( d, k );
        double const m = std::max( 4.e-6 * L, 8. * std::ldexp( 1., std::ilogb( std::max( std::fabs( bbLo[d] ), std::fabs( bbHi[d] ) ) + L ) - 23 ) );
        double const sd = bbLo[d] + L - pad - m;
        double const seam = ( sd > bbHi[d] - L ) ? sd : bbHi[d];   // psPeriodicSeam, or its fallback anchor
        lo0 = seam - L;
        len0 = L;
    };
    size_t nb = 1;
    for (int d=0; d<NO_DIM; ++d) nb *= size_t(B);
    double const unset = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> ref;
    std::vector<float> lo, hi;
    double binnedLo[NO_DIM];
    bool binned = false;
    // per bin and axis: the Eulerian extent in grid cells over the Lagrangian domain starting at dLo (length dLen)
    auto binAll = [&](double const (&dLo)[NO_DIM], double const (&dLen)[NO_DIM]) {
        ref.assign( nb*NO_DIM, unset );
        lo.assign( nb*NO_DIM, 0.f );
        hi.assign( nb*NO_DIM, 0.f );
        for (Particle_data const &q : p)
        {
            size_t b = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                double t = ( double(q.lagPos[d]) - dLo[d] ) / dLen[d];
                if ( u.periodic ) t -= std::floor( t );   // above the seam: its -L copy
                int const i = std::min( B - 1, std::max( 0, int( t * B ) ) );
                b = b * size_t(B) + size_t(i);
            }
            for (int d=0; d<NO_DIM; ++d)
            {
                size_t const k = b*NO_DIM + size_t(d);
                double x = double( q.pos[d] );
                if ( std::isnan( ref[k] ) )
                {   // the bin's first particle: its reference and extent
                    ref[k] = x;
                    lo[k] = hi[k] = float( ( x - boxLo[d] ) / dx[d] );
                    continue;
                }
                if ( u.periodic ) x = ref[k] + ( x - ref[k] ) - boxLen[d] * std::round( ( x - ref[k] ) / boxLen[d] );
                float const c = float( ( x - boxLo[d] ) / dx[d] );
                lo[k] = std::min( lo[k], c );
                hi[k] = std::max( hi[k], c );
            }
        }
    };
    // a tetrahedron reaches about one particle spacing past the particles it joins: per axis, each bin's extent is
    // widened by a spacing at the bin's own stretch (its Eulerian extent over its Lagrangian width, at least 1) and a
    // cell -- inside the partition and into the padding alike. Measured against psSubgridBox on pancakes and crossed
    // waves: 1.00 - 2.09x the largest sub-grid, never below
    double const spacingBins = double(B) / std::pow( N, 1. / double(NO_DIM) );   // a particle spacing in bin widths
    std::vector<int> diff[NO_DIM];
    // the largest sub-grid of the split k[] (partitions per axis), and the mean over its partitions (avg)
    auto spanOf = [&](int const (&k)[NO_DIM], double &avg) -> double {
        double dLo[NO_DIM], dLen[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) domainOf( d, k[d], dLo[d], dLen[d] );
        bool same = binned;
        for (int d=0; d<NO_DIM; ++d) same = same and dLo[d] == binnedLo[d];
        if ( not same )
        {
            binAll( dLo, dLen );
            for (int d=0; d<NO_DIM; ++d) binnedLo[d] = dLo[d];
            binned = true;
        }
        size_t nParts = 1;
        for (int d=0; d<NO_DIM; ++d) nParts *= size_t( k[d] );
        double most = 0., sum = 0.;
        for (size_t pi=0; pi<nParts; ++pi)
        {
            int idx[NO_DIM], b0[NO_DIM], b1[NO_DIM];
            size_t rem = pi;
            for (int d=NO_DIM-1; d>=0; --d) { idx[d] = int( rem % size_t(k[d]) ); rem /= size_t(k[d]); }
            for (int d=0; d<NO_DIM; ++d)
            {   // partition idx's first bin and the next partition's, per axis
                b0[d] = int( std::floor( double(idx[d]) * double(B) / double(k[d]) + 0.5 ) );
                b1[d] = std::max( b0[d] + 1, int( std::floor( double(idx[d] + 1) * double(B) / double(k[d]) + 0.5 ) ) );
            }
            bool any[NO_DIM], full[NO_DIM];
            for (int d=0; d<NO_DIM; ++d) { diff[d].assign( size_t( G[d] ) + 1, 0 ); any[d] = full[d] = false; }
            auto mark = [&](int d, double a, double z) {   // planes floor(a) - 1 .. floor(z) + 1 on axis d (the
                                                           // sampled deposit's +-1 cell, psSubgridBox)
                long lo0 = long( std::floor( a ) ) - 1, hi0 = long( std::floor( z ) ) + 1;
                any[d] = true;
                if ( full[d] ) return;
                if ( hi0 - lo0 + 1 >= G[d] ) { full[d] = true; return; }
                if ( u.periodic )
                {
                    long const a0 = ( lo0 % G[d] + G[d] ) % G[d], z0 = a0 + ( hi0 - lo0 );
                    if ( z0 < G[d] ) { ++diff[d][size_t(a0)]; --diff[d][size_t(z0) + 1]; }
                    else { ++diff[d][size_t(a0)]; --diff[d][size_t(G[d])]; ++diff[d][0]; --diff[d][size_t(z0 - G[d]) + 1]; }
                }
                else
                {
                    lo0 = std::max( 0L, lo0 ); hi0 = std::min( long(G[d]) - 1, hi0 );
                    if ( lo0 <= hi0 ) { ++diff[d][size_t(lo0)]; --diff[d][size_t(hi0) + 1]; }
                }
            };
            int j[NO_DIM];
            for (int e=0; e<NO_DIM; ++e) j[e] = b0[e];
            while ( true )
            {
                size_t b = 0;
                for (int e=0; e<NO_DIM; ++e) b = b * size_t(B) + size_t(std::min( j[e], B - 1 ));
                if ( not std::isnan( ref[b*NO_DIM] ) )
                    for (int d=0; d<NO_DIM; ++d)
                    {
                        size_t const kb = b*NO_DIM + size_t(d);
                        double const binCells = dLen[d] / double(B) / dx[d];
                        double const reach = std::max( double( hi[kb] - lo[kb] ), binCells ) * spacingBins + 1.;
                        mark( d, double( lo[kb] ) - reach, double( hi[kb] ) + reach );
                    }
                int e = NO_DIM - 1;
                while ( e >= 0 and ++j[e] >= b1[e] ) { j[e] = b0[e]; --e; }
                if ( e < 0 ) break;
            }
            double c = 1.;
            for (int d=0; d<NO_DIM; ++d)
            {
                double span = G[d];
                if ( any[d] and not full[d] )
                {
                    int run = 0, lo0 = -1, hi0 = -1, gap = 0, cur = 0;
                    for (int g=0; g<G[d]; ++g)
                    {
                        run += diff[d][size_t(g)];
                        if ( run > 0 ) { if ( lo0 < 0 ) lo0 = g; hi0 = g; gap = std::max( gap, cur ); cur = 0; }
                        else ++cur;
                    }
                    if ( lo0 >= 0 )
                        span = ( u.periodic and lo0 == 0 and hi0 == G[d] - 1 ) ? double( G[d] - gap ) : double( hi0 - lo0 + 1 );
                }
                c *= span;
            }
            most = std::max( most, c );
            sum += c;
        }
        avg = sum / double(nParts);
        return most;
    };
    for (int n=1; n<=6; ++n)
    {
        int k[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) k[d] = n;
        cells[n] = spanOf( k, mean[n] );
    }
    if ( userK != nullptr )
    {
        int k[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) k[d] = std::max( 1, userK[d] );
        cellsUser = spanOf( k, meanUser );
    }
#else
    (void)p; (void)u; (void)padFrac; (void)userK; (void)mean; (void)meanUser;
#endif
}

// The mean streams per sample point the plan budgets, and why: estimated from the particles.
inline void autoTunePSPointStreams(User_options &u, double const N, size_t const nPts,
                                   std::vector<Particle_data> const *particles, AutoTuneReport &report,
                                   double &ptsMeanStreams, char (&ptsStreamsWhy)[200])
{
    if ( nPts > 0 and particles != nullptr )
    {
        double boxLo[3], boxLen[3];
        for (int d=0; d<3; ++d)
        {
            boxLo[d]  = double( u.boxCoordinates[2*d] );
            boxLen[d] = double( u.boxCoordinates[2*d+1] ) - boxLo[d];
        }
        int patches = 0;
        // the estimate is made on the points' own slab -- their bounding box, widened by one mean particle
        // spacing per side -- not the whole box: the heavy tail sits in a few haloes a plane misses. A points set
        // that fills the box gets the box mean. DTFE_AUTO_PTS_BOXMEAN=1 forces the box-wide estimate.
        double rLo[3], rHi[3];
        bool haveRegion = false;
        if ( not Env::flag( "DTFE_AUTO_PTS_BOXMEAN" ) and NO_DIM == 3 )
        {
            double pLo[NO_DIM], pHi[NO_DIM], bLo[NO_DIM], bLen[NO_DIM];
            for (int d=0; d<NO_DIM; ++d) { bLo[d] = boxLo[d]; bLen[d] = boxLen[d]; }
            if ( autoTuneSamplePointBounds( u.psSamplePointsFile, bLo, bLen, u.periodic, pLo, pHi ) )
            {
                double const spacing = std::cbrt( boxLen[0]*boxLen[1]*boxLen[2] / std::max( 1., N ) );
                double vol = 1.;
                for (int d=0; d<NO_DIM; ++d)
                {
                    rLo[d] = pLo[d] - spacing;
                    rHi[d] = pHi[d] + spacing;
                    if ( u.periodic and rHi[d] - rLo[d] >= boxLen[d] ) { rLo[d] = boxLo[d]; rHi[d] = boxLo[d] + boxLen[d]; }
                    vol *= (rHi[d] - rLo[d]) / boxLen[d];
                }
                haveRegion = vol < 0.999;   // a points set that fills the box: the box mean is its slab's
            }
        }
        double const est = psEstimateMeanStreams( *particles, boxLo, boxLen, u.periodic, &patches,
                                                  haveRegion ? rLo : nullptr, haveRegion ? rHi : nullptr );
        if ( est > 0. )
        {
            // 1 + 1.4 (est - 1): at least 5% above the measured stream count of each calibration set (TNG100-3-Dark
            // z=0 and z=1, TNG50-3-Dark z=0; 1.3 would leave 1% at z=1)
            ptsMeanStreams = std::max( 1., 1. + 1.4*(est - 1.) );
            report.streamsEstimate = est;
            snprintf( ptsStreamsWhy, sizeof(ptsStreamsWhy),
                      "%.2f streams/point estimated from %d Lagrangian patches%s, budgeted as %.2f", est, patches,
                      haveRegion ? " on the points' slab" : "", ptsMeanStreams );
        }
    }
}

// The bytes the '--sample-points' structures hold: per point its position and outputs, and its per-stream records.
inline double autoTunePSPointBytes(User_options &u, size_t const nPts, bool const ptsVelGrad,
                                   double const ptsMeanStreams, double *recordBytes = nullptr,
                                   double *outputBytes = nullptr)
{
    // The STORED per-stream record is packed (ps_point_eval.cc RecLayout): denGeo + denDtfe +
    // vel[3] doubles + the 2-B caustic mask = 42 B, plus grad[3] / vgrad[9] doubles / ids[4]
    // uint64 only with --pts-den-grad / --pts-vel-grad / --ps-per-stream-ids. KEEP IN SYNC.
    double bStreamRec = 5. * double(sizeof(double)) + 2.;
    if ( u.psPtsDenGrad )   bStreamRec += 3. * double(sizeof(double));
    if ( ptsVelGrad )       bStreamRec += 9. * double(sizeof(double));
    if ( u.psPerStreamIds ) bStreamRec += 4. * 8.;
    if ( u.psPtsScalar )    bStreamRec += double(noScalarComp) * double(sizeof(double));
    if ( u.psPtsLagrangian ) bStreamRec += 3. * double(sizeof(double));
    // ... stored as uint32 point index + the packed record in blocks, and grouped at finalize by
    // one uint64 locator per record (ps_point_eval.cc HitSink / psPointEvalFinalize)
    bStreamRec += 4. + 8.;
    // per-point: pos (3 doubles) + bucketPts (uint32) + finalize's offsets and cursor (2 uint64)
    double const bPerPoint = 3. * double(sizeof(double)) + 4. + 2. * 8.;
    // finalized outputs, allocated while the records are still being drained
    double bPtsOut = double(sizeof(double))            // outDen
                   + 3. * double(sizeof(double))       // outVel
                   + 6. * double(sizeof(double))       // outDisp
                   + 4.;                               // outStreams (int32)
    if ( u.psPtsDenGrad ) bPtsOut += 3. * double(sizeof(double));
    if ( ptsVelGrad ) bPtsOut += 9. * double(sizeof(double));
    if ( u.psPerStream )  bPtsOut += ptsMeanStreams * 4. * double(sizeof(double));   // ragged records
    if ( u.psPtsScalar )  bPtsOut += double(noScalarComp) * double(sizeof(double))
                                     * (1. + (u.psPerStream ? ptsMeanStreams : 0.));   // point + per-stream scalars
    if ( u.psPtsLagrangian ) bPtsOut += ptsMeanStreams * 3. * double(sizeof(double));  // per-stream q(x)
    // for the loop's dynamic concurrency: the records as the partitions merge them (without their finalize locator),
    // and what the finalize allocates after the loop (the outputs, the per-point offsets and the per-record locators)
    if ( recordBytes ) *recordBytes = double(nPts) * ptsMeanStreams * ( bStreamRec - 8. );
    if ( outputBytes ) *outputBytes = double(nPts) * ( bPtsOut + 2. * 8. + ptsMeanStreams * 8. );
    return double(nPts) * (bPerPoint + bPtsOut + ptsMeanStreams * bStreamRec);
}

// Warns when the '--scratch-dir' volume cannot hold the full-grid accumulators.
inline void autoTuneScratchSpaceCheck(User_options &u, bool const scratchOn, double const gridTotal, double const bCell)
{
    if ( scratchOn )
    {
        struct statvfs vfs;
        if ( statvfs(u.scratchDir.c_str(), &vfs) == 0 )
        {
            double const freeBytes = double(vfs.f_bavail) * double(vfs.f_frsize);
            double const need = gridTotal*bCell;
            if ( need * 1.05 + 2.e9 > freeBytes )
            {
                MESSAGE::Warning warning( u.verboseLevel );
                warning << "AUTO-TUNE: the scratch volume cannot hold this run's grids: '"
                        << u.scratchDir << "' has " << freeBytes/1.e9 << " GB free but the "
                        << "full-resolution accumulators need ~" << need/1.e9 << " GB ("
                        << MESSAGE::printElements( u.gridSize, "x" ) << " cells x " << bCell
                        << " B/cell). The allocator will fall back to the heap when the disk "
                        << "runs out -- expect swapping. Free up space, use a bigger volume, or "
                        << "reduce --grid / --field.\n" << MESSAGE::EndWarning;
            }
        }
    }
}

// The bytes per sub-grid cell of one CPU deposit thread's private copy: the larger pass's grids that are not
// fixed-point sums. Never the face sums (words the threads share, in every build: ps_interpolation.cc accBytes
// counts neither them nor the mass flux's grid, which the normalization makes).
inline double autoTunePSCopyCell(User_options &u, AutoTunePS &ps)
{
    double copyCell = 0.;
    for (Field *f : { &u.uField, &u.aField })
    {
        if ( not f->selected() ) continue;
        double const R = double(sizeof(Real));
        bool const massFx = ps.phaseSpace and sizeof(Real) == 4 and ( f->density or f->velocity_tweb );   // as fxCell
        double const fxPass = ( ps.phaseSpace and sizeof(Real) == 4 ) ? autoTuneFxMomentBytes(*f, ps) : 0.;
        double const facePass = autoTuneFaceOn(*f, ps) ? autoTuneFaceBytes(*f, ps) + R : 0.;   // (+ the mass flux's grid)
        // the exact multiplicity is a Real sum every thread copies where the moments are Real sums, the double build
        // (exactMult in accBytes; it becomes the stream count, the slot the count bytes take out, only at the export)
        double const exactCopy = ( ps.phaseSpace and ps.exactDeposit and not ( sizeof(Real) == 4 ) ) ? R : 0.;
        copyCell = std::max( copyCell, autoTuneBytesPerCell(*f, ps, 0) - autoTunePSCountBytes(*f, ps, 0)
                                       - ( massFx ? R : 0. ) - fxPass - facePass + exactCopy );
    }
    return copyCell;
}

/* What the PS-DTFE plan reads: its inputs, resolved once by autoTunePartitioning from the options, the data and the
   machine. Values, set by name there; the planner (auto_tune/ps_planner.cc) only reads them. */
struct AutoTunePSInputs
{
    bool metalActive = false;        // the GPU deposit will run (flag given AND compiled in)
    bool mixed = false;              // ... on the GPU share's partitions only, the rest on the CPU ('--gpu-share' < 1)
    bool autoPart = false;           // --partition left to the tuner
    bool userPartitionOn = false;    // a user's split that runs the partition path ('--partition 1 2 2' too)
    bool autoConc = false;           // --max-concurrent left to the tuner
    double minN = 0.;                // the particle count from which a split is chosen for memory (kAutoMinN)
    double N = 0.;                   // the particles
    double ramBytes = 0.;            // physical RAM
    std::string budgetWhy;           // how the memory budget was set (the AUTO-TUNE line)
    double budget = 0.;              // the memory budget
    int cores = 1;                   // the threads
    size_t outGrid[NO_DIM] = {};     // the cells allocated and written (--grid-window: the window's)
    double gridTotal = 0.;           // ... their product
    double spanCells[7] = {};        // the largest partition sub-grid of an n^3 split, n <= 6 (psPartitionSpans; 0: unknown)
    double spanMean[7] = {};         // ... and the mean of the split's partition sub-grids (psPartitionSpans; 0: unknown)
    int userK[NO_DIM] = {};          // the user's --partition per axis ...
    bool userNonCubic = false;       // ... when its counts differ (the planner's n is then u.partition[0]: partsOf)
    double spanUser = 0.;            // ... and its largest sub-grid (psPartitionSpans; 0: unknown)
    double spanMeanUser = 0.;        // ... and the mean of its sub-grids
    AutoTunePS ps;                   // the PS flags
    double bCellDeposit = 0.;        // full-grid bytes per cell during the deposit (both passes)
    double bCell = 0.;               // ... the larger of the deposit and the post phase
    double padFrac[NO_DIM] = {};     // the Eulerian padding over the box length, per axis
    double partBytes = 0.;           // bytes per particle (sizeof(Particle_data))
    double kParInsert = 0.;          // the TBB insert's bytes per vertex (kParInsertBytes, 0 without it)
    bool scratchOn = false;          // --scratch-dir: the full grids are not in RAM
    size_t nPts = 0;                 // the '--sample-points'
    std::string ptsStreamsWhy;       // ... the streams per point budgeted, and why
    double ptsBytes = 0.;            // ... their structures
    double ptsRecordBytes = 0.;      // ... of which the stream records (written as the partitions merge)
    double ptsOutputBytes = 0.;      // ... and the finalized outputs (allocated at the end)
    double particleBytes = 0.;       // the particles held (0 with deferred particles)
    double readKept = 0.;            // the read buffers the allocator keeps (--grid-window)
    double readKeptAll = 0.;         // ... what every run still holds right after the read (the reserve peak:
                                     // auto_tune/ps_planner.cc reserveZeroFillPeak)
    double largestGridBytes = 0.;    // the largest pre-sized full grid (autoTuneLargestGridBytesPerCell x the cells)
    double scratchMinBytes = 0.;     // --scratch-dir's threshold: grids of at least this are scratch-backed (0: no scratch)
    double fixed = 0.;               // the irreducible term: particles + read buffers + grids + points
    bool psVel = false, psDisp = false, psGrad = false, psExact = false;   // the GPU deposit's grid selection
    double gridB = 0.;               // the GPU grids' bytes per cell and copy
    bool psDensity = false;          // a density grid is selected
    double fxCell = 0.;              // the fixed-point mass words' extra bytes per sub-grid cell
    double fxMom = 0.;               // ... the moment words' (the larger pass; without the face sums)
    double faceR = 0.;               // '--face-gradient': the face sums' Real grids per sub-grid cell (the larger pass)
    double fxFace = 0.;              // ... and what their words, int64 in EVERY build, take beyond those
    double earlyFace = 0.;           // ... those words (faceR + fxFace) where the face walk runs beside the GPU
                                     // deposit (ps_interpolation.cc PSEarlyFaceWalk): held through the device call;
                                     // 0 else
    double normR = 0.;               // ... what the normalization makes beside the deposit phase's grids, per
                                     // full-grid cell: the regular part's twins (autoTuneStreamsTwinBytes) and the
                                     // stream average's divergence (autoTuneFaceOutputBytes)
    double copyCell = 0.;            // one CPU deposit thread's private copy per sub-grid cell
    double threadScratch = 0.;       // the deposit threads' per-tetrahedron sample lists, in all
};
