// Part of ps_interpolation.cc (included in order inside its #ifdef PHASE_SPACE, never compiled alone). GPU 3D builds only:
// the GPU part of the pass (PSGpuCtx), the face walk beside the GPU deposit (PSEarlyFaceWalk) and psGpuExtractAndDeposit.

#if defined(PS_GPU) && NO_DIM==3
// The GPU deposit's part of the pass (interpolateGrid_phaseSpace): what the GPU walk needs beyond the pass and its grids.
struct PSGpuCtx
{
    PSPassCtx const &passCtx;
    PSGridsCtx const &gridsCtx;
    PSFxState &fx;
    psExactInside::Frame const &exactFrame;
    double const (&gridLenD)[NO_DIM];
    size_t psKeptCells, subTotal;
    bool degreesLazy, psVertexMass, massAudit;
    double psHaloRelease;
    Real cellVolume;
    size_t &nReleased, &nDegenerateInverse;
    bool &metalDeposited;
    std::vector<uint32_t> &deferredOrd;
};

// The face walk beside the GPU deposit: psFaceWalk on a thread of its own while the pass's thread waits in the device call,
// joined right after it and before the copy-back. It reads the tessellation, box and degrees and writes the face words only,
// its dropped shares counted apart ('drops'); its sums are the pass's only once the device call succeeded (else the caller
// throws both away and the CPU deposit makes the face sums). It counts as one core of the pool while it runs; the destructor
// joins, so an exception on the pass's thread cannot leave the walk running.
struct PSEarlyFaceWalk
{
    std::thread th;
    std::exception_ptr error;
    double sec = 0.;
    unsigned long long drops[FX_DROP_SLOTS] = {};
    bool started = false;
    PSEarlyFaceWalk() = default;
    PSEarlyFaceWalk(PSEarlyFaceWalk const &) = delete;
    PSEarlyFaceWalk &operator=(PSEarlyFaceWalk const &) = delete;
    ~PSEarlyFaceWalk() { if ( th.joinable() ) th.join(); }
    // (a thread the system refuses, or one without the memory to start, leaves 'started' false: the walk then runs
    // after the deposit)
    void start(PSPassCtx const &passCtx, PSFxState &fx, bool const psVertexMass, int const threads)
    {
        try
        {
            th = std::thread( [this, &passCtx, &fx, psVertexMass, threads]
            {
                try
                {
                    CorePool::Hold const hold( 1 );
                    auto const t0 = std::chrono::steady_clock::now();
                    psFaceWalk( passCtx, fx, psVertexMass, threads, drops );
                    sec = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
                }
                catch (...) { error = std::current_exception(); }
            } );
            started = true;
        }
        catch (std::system_error const &) { started = false; }
        catch (std::bad_alloc const &) { started = false; }   // (no memory for the thread's state: the same fallback)
    }
    // waits for the walk; an exception it met is thrown here, on the pass's thread -- unless the walk's sums are
    // thrown away anyway (keep false: the device call failed, and the CPU deposit makes the face sums as it does when
    // the walk runs after the deposit)
    void join(bool const keep = true)
    {
        if ( th.joinable() ) th.join();
        if ( error && keep ) std::rethrow_exception( error );
    }
};
// ... and the run's count of them (psFaceWalksEarlyTaken)
static inline void psFaceWalkEarlyCount(bool const kept, double const walkSec, double const waitSec)
{
    ( kept ? psFaceEarlyKept : psFaceEarlyDiscarded ).fetch_add( 1 );
    if ( !kept ) return;   // (a walk thrown away: the closing line's seconds are the kept walks')
    psFaceEarlyWalkUs.fetch_add( (long long)( walkSec * 1e6 ) );
    psFaceEarlyWaitUs.fetch_add( (long long)( waitSec * 1e6 ) );
}

// GPU deposit (--gpu): the extraction walk (the CPU loop's filters, one loop), the cost sort, the call, and the GPU's grids
// taken over as the pass's (released tets added on the host); metalDeposited on success, else the CPU deposits the pass.
template <class MassAudit, class Lap, class GridSum>
static void psGpuExtractAndDeposit(PSGpuCtx const &ctx, MassAudit &audit, Lap &lap, GridSum const &gridSum)
{
    PSPassCtx const &passCtx = ctx.passCtx;
    PSGridsCtx const &gridsCtx = ctx.gridsCtx;
    PSFxState &fx = ctx.fx;
    DT &dt = passCtx.dt;
    User_options &userOptions = passCtx.userOptions;
    Quantities *quantities = passCtx.quantities;
    Field &field = passCtx.field;
    MESSAGE::Message &message = passCtx.message;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    Real (&dx)[NO_DIM] = passCtx.dx;
    int const nSub = passCtx.nSub;
    size_t const noTotalCells = passCtx.noTotalCells;
    bool const windowMode = passCtx.windowMode;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    size_t const totalGrid = gridsCtx.totalGrid;
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
    float &massFxScale = gridsCtx.massFxScale;
    bool &massFxOn = gridsCtx.massFxOn;
    bool const (&fxSel)[PS_FX_CPU_N] = gridsCtx.fxSel;
    PSFxWords (&fxg)[PS_FX_CPU_N] = fx.fxg;
    bool const massFxWanted = gridsCtx.massFxWanted, fxMomWanted = gridsCtx.fxMomWanted, haveVel = gridsCtx.haveVel;
    bool const needWeight = gridsCtx.needWeight;
    bool const dispNeedsOwnWeight = gridsCtx.dispNeedsOwnWeight, psExact = gridsCtx.psExact, psCaustics = gridsCtx.psCaustics;
    psExactInside::Frame const &exactFrame = ctx.exactFrame;
    double const (&gridLenD)[NO_DIM] = ctx.gridLenD;
    size_t const psKeptCells = ctx.psKeptCells, subTotal = ctx.subTotal;
    bool const degreesLazy = ctx.degreesLazy, psVertexMass = ctx.psVertexMass, massAudit = ctx.massAudit;
    double const psHaloRelease = ctx.psHaloRelease;
    Real const cellVolume = ctx.cellVolume;
    size_t &nReleased = ctx.nReleased, &nDegenerateInverse = ctx.nDegenerateInverse;
    bool &metalDeposited = ctx.metalDeposited;
    std::vector<uint32_t> &deferredOrd = ctx.deferredOrd;
    auto windowTouched = [&passCtx](Real const (&ep)[NO_DIM+1][NO_DIM]) { return psWindowTouched( passCtx, ep ); };
    auto centroidCell = [&passCtx](Real const (&ep)[NO_DIM+1][NO_DIM], Real (&centroid)[NO_DIM], size_t &flat) -> bool
    { return psCentroidCell( passCtx, ep, centroid, flat ); };
    auto markFlipped = [&passCtx](PSCellGeometry const &geo, std::vector<unsigned char> &flags)
    { psMarkFlipped( passCtx, geo, flags, 0, LONG_MAX, false ); };
    // --grid-window on the GPU: the vertex degrees computed lazily (see degreesLazy in interpolateGrid_phaseSpace)
    std::vector<Cell_handle> degIncident;     // scratch of lazyDegrees (single-threaded use only)
    auto lazyDegrees = [&](Cell_handle const &cell)
    {
        for (int v = 0; v <= NO_DIM; ++v)
        {
            Vertex_handle const vh = cell->vertex(v);
            if ( vh->info().psDegree() >= 0 ) continue;
            degIncident.clear();
            psFiniteIncidentCells( dt, vh, degIncident );
            int32_t n = 0;
            size_t dropped = 0;
            for (Cell_handle c : degIncident)
            {
                PSCellGeometry g;
                if ( psFilterCell(c, userOptions, boxCoordinates, true, &dropped, g, /*skipOwnership=*/true) ) ++n;
            }
            vh->info().psDegree() = n;
        }
    };
    bool tryMetal = userOptions.psUseMetal;
    psGpuEligible( passCtx, subTotal, tryMetal );
    if ( tryMetal )
    {
        // which moment grids this run needs: unselected ones are neither extracted, allocated
        // (host or GPU; m2 is 24 B/cell, grad 36 B/cell) nor copied back
        bool const fVel  = haveVel;
        bool const fDisp = field.velocity_dispersion;
        bool const fGrad = field.velocity_gradient;
        bool const needsVelArrays = fVel || fDisp || fGrad;
        // the scalar field (one component): the kernel deposits it like the velocity moment
        bool fScal = false, fSGrad = false;
#ifdef SCALAR
        fScal  = field.scalar;
        fSGrad = field.scalar_gradient;
#endif
        bool const needsScalArrays = fScal || fSGrad;

        // reserve for the cells that survive the filters (counted by the subgrid pass), not all
        // finite cells -- the padded-partition majority is filtered out
        std::vector<float> tetVerts, tetVels, tetMasses, tetDens, tetScal;
        tetVerts.reserve( psKeptCells*PS_TET_STRIDE );
        if ( needsVelArrays )
            tetVels.reserve( psKeptCells*12 );
        if ( needsScalArrays )
            tetScal.reserve( psKeptCells*4 );
        tetMasses.reserve( psKeptCells );
        if ( psCaustics )
            tetDens.reserve( psKeptCells*4 );
        // --ps-halo-release: released tets never reach the GPU arrays; their single-cell centroid deposits (the CPU fallback's
        // arithmetic) are collected here and applied on the host AFTER the GPU grids are copied back (the copy ASSIGNS), and
        // discarded if the GPU dispatch fails (the CPU loop then re-handles every cell). momw = moment weight (V_eul).
        struct ReleasedTet { size_t flat; Real mass; Real momw; Real volFrac; PSCausticClass::CausticMask orient; Real vel[noVelComp]; Real grad[noGradComp]; Real scal; Real sgrad[NO_DIM]; };
        std::vector<ReleasedTet> releasedTets;
        // per GPU tet, its finite-cell ordinal (kept in lockstep through the cost sort): maps the
        // kernel's deferred tets back to the cells the CPU loop must deposit
        std::vector<uint32_t> tetOrd;
        tetOrd.reserve( psKeptCells );
        uint32_t cellOrdCounter = 0;
        // the bounds of the moment grids' fixed-point units (fxg): every tet that passes the filter (handed to the GPU, deferred
        // ones included, the released ones and those of no mass), summed per block of 256 finite cells and then block by block,
        // EXACTLY as the CPU deposit's bound walk (psFxBoundWalk), so a GPU and a CPU deposit of one pass take the same units.
        // Under --grid-window both count only the tets that reach the window, and the window's MASS unit is taken from these
        // bounds too (B[DW], PSFixedUnits.massTotal). Every pass sums them (fxBoundsWanted is always true).
        bool const fxBoundsWanted = true;
        std::vector<PSFixedBounds> gpuBoundPart( noTotalCells / 256 + 1 );
        // grid and canonical frame of the per-tet records (the CPU exact test's own definitions)
        double  recLo[3], recLen[3], recShift[3], recWrap[3];
        int32_t recN[3];
        for (int d = 0; d < 3; ++d)
        {
            recLo[d]    = double( boxCoordinates[2*d] );
            recLen[d]   = gridLenD[d];
            recN[d]     = int32_t( nGrid[d] );
            recShift[d] = exactFrame.periodic ? exactFrame.canonLd[d] : 0.;
            recWrap[d]  = exactFrame.periodic ? exactFrame.canonLd[d] : 0.;
        }
        lap( PS_PH_SETUP );
        PS_FOREACH_FINITE_CELL
        {
            uint32_t const cellOrd = cellOrdCounter++;   // counts EVERY finite cell, like the CPU loop
            Cell_handle cell = itC;
            PSCellGeometry geo;   // the CPU loop's filters, including the zero-inverse check
            // --grid-window: a tet that misses the window is dropped on its Eulerian positions, before the
            // filter's determinant and inverse -- the same set as testing after the filter, at a fraction of
            // the cost (a partition holds far more tets than reach a small window)
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &nDegenerateInverse, geo, false,
                               [&](Real const (&ep)[NO_DIM+1][NO_DIM]) { return windowMode && !windowTouched(ep); }) ) continue;
            markFlipped(geo, flippedFlag);     // '.hidden_streams' bit 1: geometric, so the host does it for the GPU path
            Real (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
            double Lag[NO_DIM][NO_DIM];
            for (int v=0;v<NO_DIM;++v) for (int i=0;i<NO_DIM;++i) Lag[v][i]=double(cell->vertex(v+1)->point()[i])-double(cell->vertex(0)->point()[i]);
            // absDetLag always computed: the --ps-halo-release classification below stays
            // geometric (V_lag/V_eul) so kept/released sets are identical under either mass convention
            double const absDetLag = std::fabs(determinant(Lag));
            PSCausticClass::CausticMask const tetBits = psCaustics
                ? (PSCausticClass::CausticMask)( (geo.cellDet > 0. ? PSCausticClass::BIT_PARITY_POS
                                                                   : PSCausticClass::BIT_PARITY_NEG)
                                   | PSCausticClass::classifyTet( geo.Ax, Lag ) )
                : (PSCausticClass::CausticMask)0;
            if ( degreesLazy ) lazyDegrees(cell);   // --grid-window: this tet's vertices only
            float tm = psVertexMass ? float( psVertexShareMass(cell) )
                                    : float( double(userOptions.averageDensity)*absDetLag/factorial(NO_DIM) );
            if ( fixedShareTestNanEvery() && fixedShareTestNan( double(cell->vertex(0)->point()[0]), double(cell->vertex(0)->point()[1]),
                                                                double(cell->vertex(0)->point()[NO_DIM-1]) ) )
                tm = std::numeric_limits<float>::quiet_NaN();   // DTFE_TEST_NAN_SHARE (ps_deposit_params.h)
            if ( fxBoundsWanted ) fx.tetBound( cell, geo, double( tm > 0.f ? tm : 0.f ), gpuBoundPart[cellOrd / 256] );
            if (tm<=0.f) { if ( massAudit ) { audit.after += double(tm); ++audit.nAfter; } continue; }
            if ( psHaloRelease > 0. && absDetLag > psHaloRelease * geo.cellAbsDet )
            {
                // identical classification to the CPU loop (double |det Lag| > D*|det Ax|);
                // centroid cell + centroid-evaluated velocity mirror the CPU fallback exactly
                Real centroid[NO_DIM];
                size_t f = 0;
                if ( !centroidCell(ep, centroid, f) )   // outside this grid/partition region: drop (as the CPU fallback does)
                { if ( massAudit ) { audit.after += double(tm); ++audit.nAfter; } continue; }
                if ( massAudit ) audit.released += double(tm);
                ReleasedTet rt;
                rt.flat = f;
                rt.mass = Real(tm);
                rt.momw = Real( geo.cellAbsDet / factorial(NO_DIM) );
                rt.volFrac = Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolume) );
                rt.orient = psCaustics ? tetBits : (PSCausticClass::CausticMask)(geo.cellDet > 0. ? 1 : 2);
                if ( needsVelArrays )
                {
                    Real velGrad[NO_DIM][noVelComp], temp[NO_DIM][noVelComp];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (size_t j = 0; j < noVelComp; ++j)
                            temp[v][j] = cell->vertex(v+1)->info().velocity(j) - cell->vertex(0)->info().velocity(j);
                    matrixMultiplication<noVelComp>(geo.posMatInv, temp, velGrad);
                    for (size_t j = 0; j < noVelComp; ++j)
                    {
                        rt.vel[j] = cell->vertex(0)->info().velocity(j);
                        for (int i = 0; i < NO_DIM; ++i)
                            rt.vel[j] += velGrad[i][j] * (centroid[i] - ep[0][i]);
                    }
                    if ( fGrad )
                        for (size_t j = 0; j < noVelComp; ++j)
                            for (int i = 0; i < NO_DIM; ++i)
                                rt.grad[j*NO_DIM+i] = velGrad[i][j];
                }
                rt.scal = Real(0.);
                for (int i = 0; i < NO_DIM; ++i) rt.sgrad[i] = Real(0.);
#ifdef SCALAR
                if ( needsScalArrays )
                {   // the linear scalar at the centroid and its constant gradient, as the CPU fallback deposits them
                    Real sg[NO_DIM][noScalarComp], temp[NO_DIM][noScalarComp];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (size_t j = 0; j < noScalarComp; ++j)
                            temp[v][j] = cell->vertex(v+1)->info().myScalar()[j] - cell->vertex(0)->info().myScalar()[j];
                    matrixMultiplication<noScalarComp>(geo.posMatInv, temp, sg);
                    rt.scal = cell->vertex(0)->info().myScalar()[0];
                    for (int i = 0; i < NO_DIM; ++i)
                    {
                        rt.scal += sg[i][0] * (centroid[i] - ep[0][i]);
                        rt.sgrad[i] = sg[i][0];
                    }
                }
#endif
                releasedTets.push_back(rt);
                ++nReleased;
                continue;
            }
            {
                // the record (ps_deposit_params.h): canonical vertices -> vertex-0 cell + fraction and
                // edges, plus the centroid-fallback cell exactly as the CPU fallback picks it. The vertices stay
                // Real: the double build's exact test compares DOUBLE positions (psFillTetRecord says why)
                Real cv[NO_DIM+1][NO_DIM];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        cv[v][d] = psExactInside::canonicalCoord( exactFrame, cell->vertex(v)->info().eulerianPosition(d), d );
                Real centroid[NO_DIM];
                size_t cflat = 0;
                uint32_t const cf = centroidCell(ep, centroid, cflat) ? uint32_t(cflat) : 0xFFFFFFFFu;
                size_t const at = tetVerts.size();
                tetVerts.resize( at + PS_TET_STRIDE );
                psFillTetRecord( &tetVerts[at], cv, recLo, recLen, recN, recShift, recWrap, cf );
            }
            if (needsVelArrays)
                for (int v=0;v<=NO_DIM;++v) for (size_t j=0;j<noVelComp;++j) tetVels.push_back(float(cell->vertex(v)->info().velocity(j)));
#ifdef SCALAR
            if (needsScalArrays)
                for (int v=0;v<=NO_DIM;++v) tetScal.push_back(float(cell->vertex(v)->info().myScalar()[0]));
#endif
            tetMasses.push_back(tm);
            tetOrd.push_back(cellOrd);
            // --ps-caustics: the 'dens' slots, which carry no density, hold the caustic mask (slot 0) so the
            // kernel gets the full stratification, not just the parity.
            if ( psCaustics )
            {
                tetDens.push_back( float(tetBits) );
                for (int v=1; v<=NO_DIM; ++v) tetDens.push_back( 0.f );
            }

        }

        lap( PS_PH_EXTRACT );
        // Sort tetrahedra by grid-footprint cost before dispatch. A GPU chunk finishes when its SLOWEST thread finishes: one
        // stretched void tet among cheap halo tets leaves most of the GPU idle. Cost-ordering makes every chunk's threads
        // uniformly sized, so the adaptive chunk-size controller tracks a smooth monotone cost curve.
        auto windowCells = [&](size_t const t) -> float
        {
            float const* rec = &tetVerts[t*PS_TET_STRIDE];   // edges at [6..14]
            float cells = 1.f;
            for (int d = 0; d < 3; ++d)
            {
                float lo = 0.f, hi = 0.f;
                for (int e = 0; e < 3; ++e)
                {
                    float const cc = rec[6 + e*3 + d];
                    if (cc < lo) lo = cc;
                    if (cc > hi) hi = cc;
                }
                cells *= (hi-lo) / float(dx[d]) + 1.f;
            }
            return cells;
        };
        {
            size_t const nT = tetMasses.size();
            // Counting sort on the log2 cost, quantized to 2048 steps per doubling: linear time, and balancing the chunks needs
            // no finer order (6 B/tet).
            std::vector<uint16_t> costKey(nT);
            for (size_t t = 0; t < nT; ++t)
            {
                float const cells = windowCells( t );
                // cells >= 1 (every factor is), so the key is >= 0; 30 doublings fill 61440 < 65536
                costKey[t] = uint16_t( std::min( 65535.f, std::log2(cells) * 2048.f ) );
            }
            std::vector<uint32_t> order(nT);   // order[i] = the tet that goes to position i
            {
                std::vector<size_t> slot(size_t(65536) + 1, 0);
                for (size_t t = 0; t < nT; ++t) ++slot[size_t(costKey[t]) + 1];
                for (size_t k = 0; k < size_t(65536); ++k) slot[k+1] += slot[k];
                for (size_t t = 0; t < nT; ++t) order[ slot[costKey[t]]++ ] = uint32_t(t);
            }
            std::vector<uint16_t>().swap(costKey);
            // apply the permutation IN PLACE by cycle-walking all arrays in lockstep: a gather into a fresh array would
            // transiently hold a second copy; this needs one element of scratch plus an nT-byte mask
            {
                std::vector<char> visited(nT, 0);
                bool const haveU = not tetVels.empty();
                bool const haveD = not tetDens.empty();
                bool const haveS = not tetScal.empty();
                float tmpV[PS_TET_STRIDE], tmpU[12], tmpD[4], tmpS[4];
                for (size_t start = 0; start < nT; ++start)
                {
                    if (visited[start]) continue;
                    std::memcpy(tmpV, &tetVerts[start*PS_TET_STRIDE], PS_TET_STRIDE*sizeof(float));
                    if (haveU) std::memcpy(tmpU, &tetVels[start*12], 12*sizeof(float));
                    if (haveD) std::memcpy(tmpD, &tetDens[start*4], 4*sizeof(float));
                    if (haveS) std::memcpy(tmpS, &tetScal[start*4], 4*sizeof(float));
                    float const tmpM = tetMasses[start];
                    uint32_t const tmpO = tetOrd[start];
                    size_t i = start;
                    while (true)
                    {
                        size_t const src = order[i];
                        visited[i] = 1;
                        if (src == start)
                        {
                            std::memcpy(&tetVerts[i*PS_TET_STRIDE], tmpV, PS_TET_STRIDE*sizeof(float));
                            if (haveU) std::memcpy(&tetVels[i*12], tmpU, 12*sizeof(float));
                            if (haveD) std::memcpy(&tetDens[i*4], tmpD, 4*sizeof(float));
                            if (haveS) std::memcpy(&tetScal[i*4], tmpS, 4*sizeof(float));
                            tetMasses[i] = tmpM;
                            tetOrd[i] = tmpO;
                            break;
                        }
                        std::memcpy(&tetVerts[i*PS_TET_STRIDE], &tetVerts[src*PS_TET_STRIDE], PS_TET_STRIDE*sizeof(float));
                        if (haveU) std::memcpy(&tetVels[i*12], &tetVels[src*12], 12*sizeof(float));
                        if (haveD) std::memcpy(&tetDens[i*4], &tetDens[src*4], 4*sizeof(float));
                        if (haveS) std::memcpy(&tetScal[i*4], &tetScal[src*4], 4*sizeof(float));
                        tetMasses[i] = tetMasses[src];
                        tetOrd[i] = tetOrd[src];
                        i = src;
                    }
                }
            }
        }

        double bl[3]  = { double(boxCoordinates[0]), double(boxCoordinates[2]), double(boxCoordinates[4]) };
        double dxv[3] = { double(dx[0]), double(dx[1]), double(dx[2]) };
        size_t const nTetsDeposited = tetMasses.size();   // psGpuDepositFields consumes (frees) the arrays
        if ( massAudit )
        {
            audit.gpuHanded = 0.;
            for (float const m : tetMasses) audit.gpuHanded += double(m);
        }
        PSGpuGrids gpuOut;
        std::vector<uint32_t> deferredTets;
        std::string metalErr;
        // the released tets' mass: the host adds it into the GPU's fixed-point grid below, so the unit covers it
        double releasedMass = 0.;
        for (ReleasedTet const &rt : releasedTets) releasedMass += double(rt.mass);
        double const hostMass = releasedMass;   // the mass the host adds into the GPU's words
        // the moment grids' units (fxg): one per grid the kernel deposits, from the bounds above, so the
        // host's later integer adds (the deferred and released tets) stay within them too; dispw's unit only says
        // "the mass grid is it" (the host then skips it when its mass grid is fixed).
        PSFixedUnits fxUnits;
        PSFixedBounds gpuBounds;
        for (PSFixedBounds const &b : gpuBoundPart) gpuBounds.add( b );   // in block order (see gpuBoundPart)
        std::vector<PSFixedBounds>().swap( gpuBoundPart );
        // kept for the pass: the face sums' units are taken from these after the deposit (psCpuFixedUnits)
        fx.gpuBounds = gpuBounds;
        fx.gpuBoundsValid = true;
        bool const gpuSel[PS_FX_N] = { fVel, fDisp, fGrad, true, psExact, fDisp, fDisp, fScal, fSGrad };
        for (int g = 0; g < PS_FX_N; ++g)
            if ( gpuSel[g] )
                fxUnits.scale[g] = g == PS_FX_DW ? 1.f : fx.unit( g, gpuBounds.B[g] );
        fxUnits.sumW = true;
        // --grid-window: the mass grid's unit from the window's own total, as the CPU window deposit
        // takes it (fxBoundWalk), not from the host's sum of the masses it is handed
        if ( windowMode ) fxUnits.massTotal = gpuBounds.B[PS_FX_DW];
        lap( PS_PH_SORT );
        // The face walk (--face-gradient) starts here, beside the device call, when the walk after the deposit would take its
        // units from these same bounds (psFxBoundsFromGpu: a single build without the NaN test hook); its words are that walk's.
        // They are held through the device call, which the planner books (AutoTunePSInputs::earlyFace). In the double build,
        // under the NaN test hook, when there is no room for the face words (std::bad_alloc) or the system refuses the thread
        // (PSEarlyFaceWalk::start), the walk runs after the deposit (psFaceWalk below the CPU loop).
        PSEarlyFaceWalk earlyFace;
        bool earlyWalk = false;
        try
        {
            earlyWalk = fx.face && psFxBoundsFromGpu( true, fx ) && psFaceUnits( fx, fx.gpuBounds, totalGrid );
        }
        catch (std::bad_alloc const &)   // (no room for the face sums before the call: the walk after it makes them)
        {
            fx.faceRelease();
        }
        if ( earlyWalk )
        {
            earlyFace.start( passCtx, fx, ctx.psVertexMass, psDepositThreads() );
            if ( !earlyFace.started ) fx.faceRelease();
        }
        // (the GPU call's wall time includes any wait for the device: concurrent partitions queue on it; the two
        // constants are the kernel's linear deposit, off, and its volume-weighted moments, on)
        bool gpuOk = psGpuDepositFields(tetVerts, tetVels, tetMasses, tetDens, tetScal, bl, dxv,
                                  nGrid, subOrigin, subDims, nSub,
                                  userOptions.periodic, fVel, fDisp, fGrad, false,
                                  true, psCaustics, psExact, fScal, fSGrad, windowMode, hostMass, fxUnits,
                                  gpuOut, deferredTets, metalErr);
        // A host that chose the fixed-point unit hands back the words and leaves its float grid empty, so words
        // of another size would send the decode below into that empty grid: a failed deposit instead (the CPU
        // deposits the partition). Neither host produces it; this keeps it from reading
        // out of bounds if one ever does.
        if ( gpuOk && gpuOut.massScale > 0.f && gpuOut.massFx.size() != totalGrid )
        {
            gpuOk = false;
            metalErr = "the GPU returned a mass grid of the wrong size";
        }
        for (int g = 0; gpuOk && g < PS_FX_N; ++g)   // the same guard for the moment grids' words
            if ( gpuOut.fxScale[g] > 0.f && gpuOut.fx[g].size() != totalGrid * fx.comps(g) )
            {
                gpuOk = false;
                metalErr = "the GPU returned a moment grid of the wrong size";
            }
        lap( PS_PH_GPU );
        if ( earlyFace.started )
        {   // before the copy-back; what is left of the walk counts as CPU deposit time, like the later walk's
            auto const tJoin0 = std::chrono::steady_clock::now();
            earlyFace.join( gpuOk );   // (a failed call throws the walk away, and an error of the walk with it)
            double const waited = std::chrono::duration<double>( std::chrono::steady_clock::now() - tJoin0 ).count();
            lap( PS_PH_CPU );
            psFaceWalkEarlyCount( gpuOk, earlyFace.sec, waited );
            if ( gpuOk )
            {
                fx.faceWalked = true;
                fx.faceWalkSec = earlyFace.sec;
                for (int k = 0; k < FX_DROP_SLOTS; ++k)
                    if ( earlyFace.drops[k] ) fixedShareDropped( k, earlyFace.drops[k] );
            }
            else fx.faceRelease();   // (the CPU deposit makes the face sums, and counts their dropped shares, itself)
        }
        if ( gpuOk )
        {
            fx.logUnits( "GPU", gpuOut.massScale, gpuOut.fxScale );   // (--verbose 3)
            // the deferred tets' cells, ascending: the CPU loop (ps_interpolation.cc) deposits exactly these
            deferredOrd.reserve( deferredTets.size() );
            for (uint32_t const t : deferredTets) deferredOrd.push_back( tetOrd[t] );
            size_t const nGpuDeferred = deferredOrd.size();
            std::sort( deferredOrd.begin(), deferredOrd.end() );
            if ( not userOptions.psSuppressGridStats )
                message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " " << gpuBackendName() << " GPU deposit ("
                        << MESSAGE::cMagenta() << gpuDeviceName() << MESSAGE::cReset() << "), "
                        << MESSAGE::cMagenta() << nTetsDeposited << MESSAGE::cReset() << " tetrahedra ("
                        << MESSAGE::cMagenta() << nGpuDeferred << MESSAGE::cReset()
                        << ( userOptions.psExactDeposit
                             ? " deposited by the CPU in double: degenerate in float, or a dispersion float32 cannot resolve)"
                             : " with a sample on a face, deposited by the CPU's exact test)" );
            if ( not userOptions.psSuppressGridStats )
                message << ".\n" << MESSAGE::Flush;
            // The mass grid (see massFx): the GPU's fixed-point words become this pass's mass grid as they are -- the
            // released tets are added into them below, the deferred ones in the CPU loop (ps_interpolation.cc), and
            // they are converted to the float density after that loop. Otherwise (the double build, or a mass without a
            // unit) the density is set here, from the words converted straight to Real or from the float grid.
            bool const gpuFixed = gpuOut.massScale > 0.f && gpuOut.massFx.size() == totalGrid;
            if ( field.density && massFxWanted && gpuFixed )
            {
                massFx.w.swap( gpuOut.massFx );
                massFxScale = gpuOut.massScale;
                massFxOn = true;
            }
            else if ( field.density )
                quantities->density.assign( totalGrid, Real(0.) );   // (set from the GPU's grid just below)
            std::vector<int64_t> const &gpuFx = massFxOn ? massFx.w : gpuOut.massFx;
            auto gpuMass = [&](size_t const i) -> double   // the GPU's mass of cell i, whichever form it came in
            {
                return gpuFixed ? psMassFixedMass( gpuFx[i], gpuOut.massScale ) : double( gpuOut.mass[i] );
            };
            // The moment grids (fxg): a grid the GPU returns as words becomes this pass's fixed-point grid as it is (single
            // build; deferred and released tets are added into it and it is converted after the CPU loop, like the mass);
            // otherwise its Real grid is set here from the GPU's floats, or (double build) from its words. MOMW is the GPU's
            // volume-share grid; DW is the mass grid when the kernel skipped it (dispwIsMass), and once the mass grid is this
            // pass's own words (massFxOn) DW is not accumulated at all: it becomes the density after the loop.
            std::vector<float> const *const gpuFloat[PS_FX_N] = { &gpuOut.mom, &gpuOut.m2, &gpuOut.grad, &gpuOut.momw,
                &gpuOut.streamvol, &gpuOut.dispvel, &gpuOut.dispw, &gpuOut.scal, &gpuOut.sgrad };
            bool const dwIsMass = gpuOut.dispwIsMass;
            bool fxFill[PS_FX_N] = {};    // grid g's Real grid is filled from the GPU in the cell loop below
            for (int g = 0; g < PS_FX_N; ++g)
            {
                if ( !fxSel[g] ) continue;
                if ( g == PS_FX_DW && massFxOn ) continue;            // (the density after the loop)
                bool const fromMass = ( g == PS_FX_DW && dwIsMass );
                if ( fxMomWanted && fromMass && gpuFixed && !massFxOn )
                {   // (copied while the density below still reads them)
                    fxg[g].scale = gpuOut.massScale;
                    if ( field.density ) fxg[g].w = gpuOut.massFx;
                    else fxg[g].w.swap( gpuOut.massFx );
                }
                else if ( fxMomWanted && !fromMass && gpuOut.fxScale[g] > 0.f )
                {
                    fxg[g].scale = gpuOut.fxScale[g];
                    fxg[g].w.swap( gpuOut.fx[g] );
                }
                else
                {
                    fx.ensureReal( g );
                    fxFill[g] = true;
                }
            }
            // the released tets' multiplicity (sampled deposit + --ps-halo-release): the host's own sum, in the unit a
            // CPU deposit of this pass would take (the same bounds), so the two write the same '.streams'
            if ( fxMomWanted && fxSel[PS_FX_RM] )
            {
                float const scale = fx.unit( PS_FX_RM, gpuBounds.B[PS_FX_RM] );
                if ( scale > 0.f ) fx.start( PS_FX_RM, scale );
            }
            // grid g's GPU value at component index 'at' (cell x components + component), whichever form it came in
            auto gpuVal = [&](int const g, size_t const at) -> double
            {
                if ( g == PS_FX_DW && dwIsMass ) return gpuMass(at);
                if ( gpuOut.fxScale[g] > 0.f ) return psMassFixedMass( gpuOut.fx[g][at], gpuOut.fxScale[g] );
                return double( (*gpuFloat[g])[at] );
            };
            for (size_t i = 0; i < totalGrid; ++i)
            {
                if (field.density && !massFxOn) quantities->density[i] = Real( gpuMass(i) );
                // moment normalizer: the volume-share sum (in the single build the exact words' value, rounded
                // once)
                if (fxFill[PS_FX_MOMW]) massWeight[i] = Real( gpuVal(PS_FX_MOMW, i) );
                if (fxFill[PS_FX_MOM])
                    for (size_t j = 0; j < noVelComp; ++j)
                        quantities->velocity[i][j] = Real( gpuVal(PS_FX_MOM, i*3+j) );
                if (fxFill[PS_FX_GRAD])
                    for (size_t q = 0; q < noGradComp; ++q)
                        quantities->velocity_gradient[i][q] = Real( gpuVal(PS_FX_GRAD, i*9+q) );
                if (fxFill[PS_FX_M2])
                    for (size_t cc = 0; cc < noDispComp; ++cc)
                        quantities->velocity_dispersion[i][cc] = Real( gpuVal(PS_FX_M2, i*6+cc) );
#ifdef SCALAR
                if (fxFill[PS_FX_SCAL]) quantities->scalar[i][0] = Real( gpuVal(PS_FX_SCAL, i) );
                if (fxFill[PS_FX_SG]) for (int q = 0; q < NO_DIM; ++q) quantities->scalar_gradient[i][q] = Real( gpuVal(PS_FX_SG, i*3+q) );
#endif
                // the dispersion's own mass-weighted mean/normalizer (kernel buffers 13/14)
                if (fxFill[PS_FX_DW]) dispWeight[i] = Real( gpuVal(PS_FX_DW, i) );
                if (fxFill[PS_FX_DV])
                    for (size_t j = 0; j < noVelComp; ++j)
                        dispVel[i][j] = Real( gpuVal(PS_FX_DV, i*3+j) );
                streamCount[i] = int(gpuOut.streams[i] & 0x7FFFFFFFu);   // bit 31 = the sub-sample mass flag ('.hidden_streams' bit 2)
                if ( gpuOut.streams[i] >> 31 ) subSampleMassFlag[i] = 1;
                if (fxFill[PS_FX_SV])
                    exactMult[i] = Real( gpuVal(PS_FX_SV, i) );   // exact cell-mean multiplicity
                // The mask rides in 'dens' and the kernel ORs the host-built bits verbatim, so a GPU
                // deposit gives the fold flag EXACTLY (asserted by ps_caustic_class_check.sh) and the
                // stratification bits the GPU carries (BITS_GPU_MAX).
                if (psCaustics)
                    orientBits[i] = (PSCausticClass::CausticMask)
                        ( gpuOut.caustic[i] & (unsigned)PSCausticClass::BITS_GPU_MAX );
            }
            if ( massAudit )
            {
                audit.gpuDeferred = gpuOut.auditDeferredMass;
                audit.gpuFixed = gpuOut.auditFixedTotal;
                if ( field.density ) audit.gpuFloat = gridSum();
            }
            // --ps-halo-release: host-side monolithic centroid deposits of the released tets
            // (same accumulation sites and moments as the CPU fallback's single sample)
            // (each moment add: into the grid's fixed-point words when it has them, else its Real grid --
            // the same float product either way; fxHost)
            auto fxHost = [&](int const g, size_t const at, Real const v) -> bool
            {
                if ( fxg[g].w.empty() ) return false;
                psFxAdd( fxg[g].sink( 0 ), at, double(v), g, false );
                return true;
            };
            for (ReleasedTet const &rt : releasedTets)
            {
                if (field.density)
                {
                    if ( massFxOn )
                        psFxAdd( massFx.sink( 0 ), rt.flat, double(rt.mass), FX_DROP_MASS, false );   // exact
                    else quantities->density[rt.flat] += rt.mass;
                }
                if (needWeight && !fxHost(PS_FX_MOMW, rt.flat, rt.momw)) massWeight[rt.flat] += rt.momw;
                if (haveVel)
                    for (size_t j = 0; j < noVelComp; ++j)
                        if ( !fxHost(PS_FX_MOM, rt.flat*noVelComp + j, rt.vel[j] * rt.momw) )
                            quantities->velocity[rt.flat][j] += rt.vel[j] * rt.momw;
                if (field.velocity_gradient)
                    for (size_t q = 0; q < noGradComp; ++q)
                        if ( !fxHost(PS_FX_GRAD, rt.flat*noGradComp + q, rt.grad[q] * rt.momw) )
                            quantities->velocity_gradient[rt.flat][q] += rt.grad[q] * rt.momw;
#ifdef SCALAR
                if (fScal && !fxHost(PS_FX_SCAL, rt.flat, rt.scal * rt.momw)) quantities->scalar[rt.flat][0] += rt.scal * rt.momw;
                if (fSGrad)
                    for (int q = 0; q < NO_DIM; ++q)
                        if ( !fxHost(PS_FX_SG, rt.flat*NO_DIM + size_t(q), rt.sgrad[q] * rt.momw) )
                            quantities->scalar_gradient[rt.flat][q] += rt.sgrad[q] * rt.momw;
#endif
                if (field.velocity_dispersion)
                {
                    size_t c = 0;
                    for (int a = 0; a < NO_DIM; ++a)
                        for (int b = a; b < NO_DIM; ++b, ++c)
                            if ( !fxHost(PS_FX_M2, rt.flat*noDispComp + c, rt.mass * rt.vel[a] * rt.vel[b]) )
                                quantities->velocity_dispersion[rt.flat][c] += rt.mass * rt.vel[a] * rt.vel[b];
                    if ( dispNeedsOwnWeight )
                    {
                        if ( !massFxOn && !fxHost(PS_FX_DW, rt.flat, rt.mass) )   // (massFxOn: DW is the density)
                            dispWeight[rt.flat] += rt.mass;
                        for (size_t j = 0; j < noVelComp; ++j)
                            if ( !fxHost(PS_FX_DV, rt.flat*noVelComp + j, rt.vel[j] * rt.mass) )
                                dispVel[rt.flat][j] += rt.vel[j] * rt.mass;
                    }
                }
                subSampleMassFlag[rt.flat] = 1;            // mass from a never-sampled tet
                if (psExact)
                {
                    streamCount[rt.flat]++;             // raw tet-touch count ('.tetTouch')
                    if ( !fxHost(PS_FX_SV, rt.flat, rt.volFrac) )
                        exactMult[rt.flat] += rt.volFrac;   // V_tet / V_cell, as the CPU fallback deposits
                }
                else if ( !fxHost(PS_FX_RM, rt.flat, rt.volFrac) )
                    releasedMult[rt.flat] += rt.volFrac;   // never sampled: its volume fraction
                if (psCaustics)
                    orientBits[rt.flat] |= rt.orient;
            }
            if ( massAudit && field.density && !releasedTets.empty() ) audit.releasedGain = gridSum() - audit.gpuFloat;
            metalDeposited = true;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "PS-DTFE GPU deposit unavailable (" << metalErr << "); using the CPU deposit.\n" << MESSAGE::EndWarning;
            ++psGpuFallbacks;
            nReleased = 0;   // releasedTets are discarded; the CPU loop (ps_interpolation.cc) recounts them
            if ( massAudit ) { audit.after = audit.released = 0.; audit.nAfter = 0; audit.gpuHanded = -1.; }
        }
    }
}
#endif // PS_GPU
