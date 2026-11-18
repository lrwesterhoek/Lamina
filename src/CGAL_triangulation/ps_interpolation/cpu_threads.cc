// Part of src/CGAL_triangulation/ps_interpolation.cc, inside its #ifdef PHASE_SPACE: included once, in order, by
// that file and compiled only as part of its object, never alone.
// The CPU deposit's threads: their count and mode (psDepositThreadMode), the slab binning (psSlabBinning) and
// each thread's start (psDepositThreadsInit).

// The CPU deposit's thread count and mode (private copies, or slabs of the x axis), and the copy mode's line.
static inline void psDepositThreadMode(PSPassCtx const &passCtx, PSLayoutCtx const &layoutCtx, bool const metalDeposited,
                                       double const accBytes, bool const massAudit, std::vector<uint32_t> &deferredOrd)
{
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
    size_t const noTotalCells = passCtx.noTotalCells;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    int &nDepThreads = layoutCtx.nDepThreads;
    bool &slabMode = layoutCtx.slabMode;
    bool &dynamicTeam = layoutCtx.dynamicTeam;
    bool &teamGrows = layoutCtx.teamGrows;
    if ( !metalDeposited && !userOptions.psCausticCusps )
    {
        int const wantThreads = psDepositThreads();
        nDepThreads = wantThreads;
        double const extra = psDepositThreadBytes();
        if ( accBytes > 0. )   // (the quotient capped before the int conversion: a few cells' copy is a few bytes)
            nDepThreads = std::max( 1, std::min( nDepThreads, 1 + int( std::min( double(nDepThreads), extra / accBytes ) ) ) );
        size_t const blocks = noTotalCells / 256 + 1;                      // at least a few blocks per thread (DEP_BLOCK)
        int const blockCap = std::max( 1, int( blocks / 4 ) + 1 );
        nDepThreads = std::max( 1, std::min( nDepThreads, blockCap ) );
        int const slabThreads = std::max( 1, std::min( wantThreads, blockCap ) );
        int const slabSetting = Env::tristate( "DTFE_DEPOSIT_SLABS" );     // 1 force, 0 never, unset automatic
        // automatic: only when the copies leave fewer than HALF the threads -- a slab run classifies a
        // straddling tet once per slab (~1.6x the work), so the copies win while they keep half the threads.
        // The fixed-point grids, the counts and the flags are shared (psAddCount), so the single build copies
        // nothing and this rule only meets the float sums (the double build) on fine grids.
        if ( slabSetting != 0 and slabThreads > 1 and subDims[0] >= 2
             and ( slabSetting == 1 or 2 * nDepThreads < slabThreads ) and noTotalCells < size_t(UINT32_MAX) )
        {
            slabMode = true;
            nDepThreads = slabThreads;
        }
        // THE DYNAMIC TEAM (PSBlockClaim): the copy mode with nothing copied -- the single build, where every
        // accumulator is a shared integer word, count, flag or mask (accBytes == 0) -- claims its blocks from one
        // counter, and unless DTFE_DEPOSIT_THREADS pins the team it follows the free cores (CorePool), up to every core
        // (and the blocks' cap). The double build and a pass with a float grid (a bound without a unit) keep the
        // static rule: their per-thread copies are summed in thread order, so their sums depend on which
        // thread took which block. So does DTFE_MASS_AUDIT's per-thread double tallies (its line would vary).
        if ( !slabMode && accBytes == 0. && std::is_same<Real, float>::value && !massAudit )
        {
            bool const pinned = Env::integer( "DTFE_DEPOSIT_THREADS", 0 ) > 0;
            int const most = pinned ? nDepThreads : std::max( 1, std::min( CorePool::total(), blockCap ) );
            if ( most > 1 )
            {
                dynamicTeam = true;
                teamGrows = !pinned;
                nDepThreads = most;
            }
        }
    }
    // The GPU-DEFERRED tetrahedra run in slab mode too, on the deferred cells only: one thread walking every
    // finite cell could outlast the GPU deposit itself (a deferred tet is typically a large one whose samples all
    // take the exact double test). Slab mode writes the real arrays in the single-threaded loop's order, so the grids
    // stay bit-identical to the one-thread loop (DTFE_DEPOSIT_SLABS=0 keeps that loop). A few hundred deferred tets are
    // not worth the binning: below DEFERRED_SLAB_MIN the loop stays serial.
    size_t const DEFERRED_SLAB_MIN = 2048;
    if ( metalDeposited && deferredOrd.size() >= DEFERRED_SLAB_MIN && !userOptions.psCausticCusps
         && subDims[0] >= 2 && noTotalCells < size_t(UINT32_MAX) )
    {
        int const slabThreads = std::max( 1, std::min( psDepositThreads(), int( deferredOrd.size() / 1024 ) + 1 ) );
        if ( Env::tristate( "DTFE_DEPOSIT_SLABS" ) != 0 && slabThreads > 1 )
        {
            slabMode = true;
            nDepThreads = slabThreads;
        }
    }
    if ( !metalDeposited && !slabMode && !dynamicTeam && ( not userOptions.psSuppressGridStats or userOptions.verboseLevel >= 3 ) )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " CPU deposit on " << nDepThreads
                << ( nDepThreads == 1 ? " thread" : " threads" ) << " (copy mode: " << accBytes / 1.e6
                << " MB of private accumulators per extra thread, budget " << psDepositThreadBytes() / 1.e9
                << " GB; the fixed-point grids, counts and flags are shared).\n" << MESSAGE::Flush;
}

// Slab mode: bins the tetrahedra into slabs of local x planes cut at equal work, and prints the slab line.
static inline void psSlabBinning(PSPassCtx const &passCtx, PSLayoutCtx const &layoutCtx, bool const metalDeposited,
                                 std::vector<uint32_t> &deferredOrd)
{
    DT &dt = passCtx.dt;   // (PS_FINITE_CELLS_BEGIN / END)
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    Real (&dx)[NO_DIM] = passCtx.dx;
    size_t const noTotalCells = passCtx.noTotalCells;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    bool const slabMode = layoutCtx.slabMode;
    int const nDepThreads = layoutCtx.nDepThreads;
    int &nSlabs = layoutCtx.nSlabs;
    std::vector<long> &slabXLo = layoutCtx.slabXLo, &slabXHi = layoutCtx.slabXHi;
    std::vector< std::vector<uint32_t> > &slabTetLists = layoutCtx.slabTetLists;
    std::vector<Cell_handle> &slabTets = layoutCtx.slabTets;
    std::vector<uint8_t> &slabFirst = layoutCtx.slabFirst;
    if ( slabMode )
    {
        // binning: each tetrahedron's x range of global cells -- the bounding box of its Eulerian
        // vertices, minimum-image wrapped EXACTLY as psFilterCell wraps them (same Real operations, so the
        // same positions), widened by two cells: a superset of every cell the deposit writes (the sample
        // window is the bounding box itself, the centroid fallback lies inside it). No
        // filter here: running it once more per tetrahedron would cost a full extra pass (a large part of the
        // slab run on a clustered box, where a tet covers few cells); the deposit filters, and counts a
        // rejection in the tet's first slab. After a GPU deposit: only the deferred cells, in loop order.
        if ( metalDeposited )
        {
            slabTets.reserve( deferredOrd.size() );
            size_t ord = 0, next = 0;
            for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END && next < deferredOrd.size(); ++it, ++ord)
                if ( size_t(deferredOrd[next]) == ord ) { slabTets.push_back( it ); ++next; }
        }
        else
        {
            slabTets.reserve( noTotalCells );
            for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END; ++it) slabTets.push_back( it );
        }
        size_t const nTets = slabTets.size();
        std::vector<int32_t> gLo( nTets ), gHi( nTets );         // its global x cells [gLo, gHi)
        std::vector<float> cost( nTets, 0.f );                    // its work: bbox cells + a per-tet overhead
        std::vector<double> extentSum( size_t(nDepThreads), 0. );
        std::vector<size_t> extentN( size_t(nDepThreads), 0 );
        auto bin = [&](size_t th)
        {
            size_t const lo = nTets * th / size_t(nDepThreads), hi = nTets * (th + 1) / size_t(nDepThreads);
            for (size_t i = lo; i < hi; ++i)
            {
                Cell_handle cell = slabTets[i];
                Real ep[NO_DIM+1][NO_DIM];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        ep[v][d] = cell->vertex(v)->info().eulerianPosition(d);
                if ( userOptions.periodic )
                    for (int v = 1; v <= NO_DIM; ++v)
                        for (int d = 0; d < NO_DIM; ++d)
                        {
                            Real const boxLen = boxCoordinates[2*d+1] - boxCoordinates[2*d];
                            Real const diff = ep[v][d] - ep[0][d];
                            if (diff >  boxLen * Real(0.5)) ep[v][d] -= boxLen;
                            if (diff < -boxLen * Real(0.5)) ep[v][d] += boxLen;
                        }
                double cells = 1.;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    double eMin = double(ep[0][d]), eMax = eMin;
                    for (int v = 1; v <= NO_DIM; ++v)
                    {
                        eMin = std::min( eMin, double(ep[v][d]) );
                        eMax = std::max( eMax, double(ep[v][d]) );
                    }
                    double const x0 = double(boxCoordinates[2*d]), dxd = double(dx[d]);
                    double const lo = std::floor((eMin - x0) / dxd), hi = std::floor((eMax - x0) / dxd) + 1.;
                    cells *= hi - lo;
                    if ( d == 0 )
                    {
                        gLo[i] = int32_t( lo ) - 2;
                        gHi[i] = int32_t( hi ) + 2;
                        extentSum[th] += hi - lo;
                        extentN[th] += 1;
                    }
                }
                cost[i] = float( 2. + cells );
            }
        };
        {
            CorePool::Hold const hold( nDepThreads - 1 );   // (the slab binning's static teams: counted in the pool)
            std::vector<std::thread> pool;
            for (int th = 1; th < nDepThreads; ++th) pool.emplace_back( bin, size_t(th) );
            bin( 0 );
            for (std::thread &x : pool) x.join();
        }
        double eSum = 0.; size_t eN = 0;
        for (int th = 0; th < nDepThreads; ++th)
        { eSum += extentSum[size_t(th)]; eN += extentN[size_t(th)]; }
        double const meanExtent = eN ? eSum / double(eN) : 1.;
        long const nx = long(subDims[0]);
        bool const periodicX = userOptions.periodic;
        long const nG = long(nGrid[0]), sO = long(subOrigin[0]);
        // a tetrahedron's global x cells, mapped to local planes exactly as the scatter maps them
        auto forEachPlane = [&](size_t i, auto const &f)
        {
            long a = gLo[i], b = gHi[i];
            if ( periodicX and b - a > nG ) { a = 0; b = nG; }       // spans the whole period
            for (long g = a; g < b; ++g)
            {
                long w = g;
                if ( periodicX ) w = ((g % nG) + nG) % nG;
                else if ( g < 0 || g >= nG ) continue;
                long loc = w - sO;
                if ( loc < 0 ) loc += nG;
                if ( loc < 0 || loc >= nx ) continue;
                f( loc );
            }
        };
        // The slab boundaries balance the WORK, not the width: halos put most tetrahedra into a few
        // planes of a clustered box, where equal-width slabs would leave the thread holding them
        // alone. Each tet's cost (bbox cells + overhead) is spread over its planes, the slabs cut at
        // equal cumulative cost. Their number is a multiple of the threads (k slabs each, picked up
        // dynamically), k ~ the planes per thread over four mean tetrahedron extents (a straddler
        // then costs ~25% extra), at least 2, at most 64 slabs.
        std::vector<double> planeCost( size_t(nx), 0. );
        {
            std::vector< std::vector<double> > part( size_t(nDepThreads), std::vector<double>( size_t(nx), 0. ) );
            auto hist = [&](size_t th)
            {
                size_t const lo = nTets * th / size_t(nDepThreads), hi = nTets * (th + 1) / size_t(nDepThreads);
                std::vector<double> &h = part[th];
                for (size_t i = lo; i < hi; ++i)
                {
                    double const share = double(cost[i]) / double( std::max<long>( 1, gHi[i] - gLo[i] ) );
                    forEachPlane( i, [&](long loc) { h[size_t(loc)] += share; } );
                }
            };
            CorePool::Hold const hold( nDepThreads - 1 );
            std::vector<std::thread> pool;
            for (int th = 1; th < nDepThreads; ++th) pool.emplace_back( hist, size_t(th) );
            hist( 0 );
            for (std::thread &x : pool) x.join();
            for (int th = 0; th < nDepThreads; ++th)        // summed in thread order: the same cuts every run
                for (long x = 0; x < nx; ++x) planeCost[size_t(x)] += part[size_t(th)][size_t(x)];
        }
        std::vector<float>().swap( cost );
        int const perThread = std::max( 2, std::min( 64 / std::max( 1, nDepThreads ),
                              int( std::lround( double(nx) / ( 4. * std::max( 1., meanExtent ) * double(nDepThreads) ) ) ) ) );
        nSlabs = int( std::min<long>( nx, std::min( 64L, long(perThread) * long(nDepThreads) ) ) );
        if ( long const v = Env::integer( "DTFE_DEPOSIT_NSLABS", 0 ); v > 0 )
            nSlabs = int( std::min( long(nx), std::min( 64L, v ) ) );
        slabXLo.assign( size_t(nSlabs), 0 ); slabXHi.assign( size_t(nSlabs), 0 );
        std::vector<int> slabOfX( static_cast<size_t>(nx), 0 );
        {
            double total = 0.;
            for (long x = 0; x < nx; ++x) total += planeCost[size_t(x)];
            long x = 0;
            double acc = 0.;
            for (int sl = 0; sl < nSlabs; ++sl)
            {
                slabXLo[size_t(sl)] = x;
                // the last slab takes the rest; every slab keeps at least one plane, and leaves one for each after it
                long const maxHi = nx - long(nSlabs - 1 - sl);
                double const target = total * double(sl + 1) / double(nSlabs);
                long hi = x + 1;
                acc += planeCost[size_t(x)];
                while ( hi < maxHi and ( sl == nSlabs - 1 or acc + 0.5 * planeCost[size_t(hi)] <= target ) )
                { acc += planeCost[size_t(hi)]; ++hi; }
                if ( sl == nSlabs - 1 ) hi = nx;
                slabXHi[size_t(sl)] = hi;
                for (long y = x; y < hi; ++y) slabOfX[size_t(y)] = sl;
                x = hi;
            }
        }
        std::vector<uint64_t> mask( nTets, 0 );
        auto toSlabs = [&](size_t th)
        {
            size_t const lo = nTets * th / size_t(nDepThreads), hi = nTets * (th + 1) / size_t(nDepThreads);
            for (size_t i = lo; i < hi; ++i)
            {
                uint64_t m = 0;
                forEachPlane( i, [&](long loc) { m |= uint64_t(1) << slabOfX[size_t(loc)]; } );
                // reaching no plane (e.g. the padding around a --box region): it deposits nothing, but is
                // still filtered and counted once -- spread over the slabs, not all on slab 0, which the work
                // balance does not see
                mask[i] = m ? m : uint64_t(1) << ( i % size_t(nSlabs) );
            }
        };
        {
            CorePool::Hold const hold( nDepThreads - 1 );
            std::vector<std::thread> pool;
            for (int th = 1; th < nDepThreads; ++th) pool.emplace_back( toSlabs, size_t(th) );
            toSlabs( 0 );
            for (std::thread &x : pool) x.join();
        }
        std::vector<int32_t>().swap( gLo ); std::vector<int32_t>().swap( gHi );
        slabTetLists.resize( size_t(nSlabs) );
        slabFirst.assign( nTets, 0 );
        {
            std::vector<size_t> count( size_t(nSlabs), 0 );
            for (size_t i = 0; i < nTets; ++i)
                for (uint64_t m = mask[i]; m; m &= m - 1) ++count[size_t(__builtin_ctzll(m))];
            for (int sl = 0; sl < nSlabs; ++sl) slabTetLists[size_t(sl)].reserve( count[size_t(sl)] );
        }
        size_t entries = 0;
        for (size_t i = 0; i < nTets; ++i)
        {
            if ( mask[i] ) slabFirst[i] = uint8_t( __builtin_ctzll(mask[i]) );
            for (uint64_t m = mask[i]; m; m &= m - 1)
            { slabTetLists[size_t(__builtin_ctzll(m))].push_back( uint32_t(i) ); ++entries; }
        }
        size_t kept = 0;
        for (size_t i = 0; i < nTets; ++i) kept += mask[i] ? 1 : 0;
        if ( not userOptions.psSuppressGridStats or userOptions.verboseLevel >= 3 )   // (a partition's at 3)
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                    << ( metalDeposited ? " GPU-deferred tetrahedra on the CPU in SLAB mode: " : " CPU deposit in SLAB mode: " )
                    << nDepThreads << " threads over " << nSlabs << " slabs of the x axis"
                    << ( metalDeposited ? ", " : " (the accumulator copies would not fit), " )
                    << ( kept ? double(entries) / double(kept) : 0. )
                    << " slabs per tetrahedron (mean extent " << meanExtent << " cells).\n" << MESSAGE::Flush;
    }
}

// The deposit threads' accumulators: the real arrays for thread 0 (all threads in slab mode), else zeroed copies.
template <class DepositThread>
static inline void psDepositThreadsInit(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx,
                                        std::vector<DepositThread> &depT, int const nDepThreads, bool const slabMode)
{
    Quantities *quantities = passCtx.quantities;
    std::vector<int> &streamCount = gridsCtx.streamCount;
    std::vector<Real> &releasedMult = gridsCtx.releasedMult;
    std::vector<unsigned char> &subSampleMassFlag = gridsCtx.subSampleMassFlag;
    std::vector<unsigned char> &flippedFlag = gridsCtx.flippedFlag;
    std::vector<Real> &exactMult = gridsCtx.exactMult;
    std::vector<PSCausticClass::CausticMask> &orientBits = gridsCtx.orientBits;
    std::vector<Real> &massWeight = gridsCtx.massWeight;
    std::vector<Real> &dispWeight = gridsCtx.dispWeight;
    std::vector< Pvector<Real,noVelComp> > &dispVel = gridsCtx.dispVel;
    PSFxWords &massFx = gridsCtx.massFx;
    bool &massFxOn = gridsCtx.massFxOn;
    std::vector<size_t> &insideFlat = gridsCtx.insideFlat;
    std::vector<Real> &insideRel = gridsCtx.insideRel;
    std::vector<double> &exMom = gridsCtx.exMom, &exW = gridsCtx.exW;
    for (int th = 0; th < nDepThreads; ++th)
    {
        DepositThread &D = depT[size_t(th)];
        D.massFx = massFxOn ? massFx.w.data() : nullptr;   // shared by all (its own density copy stays EMPTY)
        if ( th == 0 or slabMode )      // slab mode: every thread writes its own cells of the real arrays
        {
            D.q = quantities; D.streamCount = &streamCount; D.releasedMult = &releasedMult;
            D.exactMult = &exactMult; D.dispWeight = &dispWeight; D.dispVel = &dispVel; D.massWeight = &massWeight;
            D.orientBits = &orientBits; D.subSampleMassFlag = &subSampleMassFlag; D.flippedFlag = &flippedFlag;
            D.insideFlat.swap( insideFlat ); D.insideRel.swap( insideRel );
            D.exMom.swap( exMom ); D.exW.swap( exW );
            continue;
        }
        auto zeroLike = [](auto &own, auto const &real) { own.assign( real.size(), typename std::decay_t<decltype(real)>::value_type() ); };
        zeroLike( D.ownQ.density, quantities->density );
        zeroLike( D.ownQ.velocity, quantities->velocity );
        zeroLike( D.ownQ.velocity_gradient, quantities->velocity_gradient );
        zeroLike( D.ownQ.velocity_dispersion, quantities->velocity_dispersion );
#ifdef SCALAR
        zeroLike( D.ownQ.scalar, quantities->scalar );
        zeroLike( D.ownQ.scalar_gradient, quantities->scalar_gradient );
#endif
        zeroLike( D.ownReleasedMult, releasedMult );       zeroLike( D.ownExactMult, exactMult );
        zeroLike( D.ownDispWeight, dispWeight );           zeroLike( D.ownDispVel, dispVel );
        zeroLike( D.ownMassWeight, massWeight );
        D.q = &D.ownQ; D.releasedMult = &D.ownReleasedMult;
        D.exactMult = &D.ownExactMult; D.dispWeight = &D.ownDispWeight; D.dispVel = &D.ownDispVel;
        D.massWeight = &D.ownMassWeight;
        // shared by every thread, written with relaxed atomics (countAtomicL in depositLoop)
        D.streamCount = &streamCount;
        D.orientBits = &orientBits; D.subSampleMassFlag = &subSampleMassFlag; D.flippedFlag = &flippedFlag;
    }
}
