// Part of ps_interpolation.cc (included in order inside its #ifdef PHASE_SPACE, never compiled alone). The cell-loop macros
// (never #undef'd: later parts use them), the pass's contexts (PSPassCtx, PSGridsCtx, PSLayoutCtx, PSBlockClaim), its tet tests.

// The cell iteration of the pass and its phases: the finite faces (2D) / cells (3D) of the triangulation 'dt'.
#if NO_DIM==2
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
    #define PS_FINITE_CELLS_BEGIN dt.finite_faces_begin()
    #define PS_FINITE_CELLS_END dt.finite_faces_end()
#elif NO_DIM==3
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
    #define PS_FINITE_CELLS_BEGIN dt.finite_cells_begin()
    #define PS_FINITE_CELLS_END dt.finite_cells_end()
#endif

// The pass's triangulation, options, fields and grid geometry, as references, for the phases around its deposit.
struct PSPassCtx
{
    DT &dt;
    User_options &userOptions;
    Quantities *quantities;
    Field &field;
    MESSAGE::Message &message;
    Box &boxCoordinates;
    size_t const *nGrid;
    Real (&dx)[NO_DIM];
    int nSub;
    size_t noTotalCells;
    bool windowMode;
    size_t (&subOrigin)[NO_DIM];
    size_t (&subDims)[NO_DIM];
};

// The pass's accumulators, its per-tetrahedron scratch and the flags that select them, as references.
struct PSGridsCtx
{
    size_t totalGrid;
    std::vector<int> &streamCount;
    std::vector<Real> &releasedMult;
    std::vector<unsigned char> &subSampleMassFlag;
    std::vector<unsigned char> &flippedFlag;
    std::vector<Real> &exactMult;
    std::vector<PSCausticClass::CausticMask> &orientBits;
    std::vector<Real> &massWeight;
    std::vector<Real> &dispWeight;
    std::vector< Pvector<Real,noVelComp> > &dispVel;
    PSFxWords &massFx;
    float &massFxScale;   // (massFx.scale)
    bool &massFxOn;
    std::vector<size_t> &insideFlat;
    std::vector<Real> &insideRel;
    std::vector<double> &exMom, &exW;
    bool const (&fxSel)[PS_FX_CPU_N];
    bool massFxWanted, fxMomWanted, haveVel, needWeight, dispNeedsOwnWeight, deferNorm, psExact, psCaustics;
};

// The CPU deposit's thread layout: the thread count, the slab mode and the slabs' tetrahedra, the dynamic team, as
// references.
struct PSLayoutCtx
{
    int &nDepThreads;
    bool &slabMode;
    bool &dynamicTeam;   // blocks claimed from one counter (PSBlockClaim); nDepThreads is then the team's most threads
    bool &teamGrows;     // ... and the team follows the free cores (else pinned by DTFE_DEPOSIT_THREADS)
    int &nSlabs;
    std::vector<long> &slabXLo, &slabXHi;
    std::vector< std::vector<uint32_t> > &slabTetLists;
    std::vector<Cell_handle> &slabTets;
    std::vector<uint8_t> &slabFirst;
};

// THE DYNAMIC TEAM of the CPU deposit (psDepositThreadMode says when): the finite cells in blocks of 'blockLen' (DEP_BLOCK),
// each claimed whole from one shared counter by whichever thread is free, the team grown by its first thread whenever the core
// pool (CorePool) has a free core. Only where every accumulator is shared and order-free (integer fixed-point words, counts,
// flags, masks; relaxed atomics, no private copies): the sums do not depend on which thread deposits which block, so the grids
// are those of DTFE_DEPOSIT_THREADS=1 byte for byte. The block index holds each block's first cell; one walk builds it.
template <class CellIt>
struct PSBlockClaim
{
    std::vector<CellIt> blockStart;            // block b's first finite cell, in the loop's order
    size_t nCells = 0, blockLen = 256;
    std::atomic<size_t> next{ 0 };
    std::function<void()> recruit;             // (the first thread, per claim) one more helper if a core is free

    // block b for thread th, with its first cell and its number of cells (>= 1); false when every block is taken
    bool claim(size_t const th, CellIt &first, size_t &cells)
    {
        if ( th == 0 && recruit ) recruit();
        size_t const b = next.fetch_add( 1, std::memory_order_relaxed );
        if ( b >= blockStart.size() ) return false;
        first = blockStart[b];
        cells = std::min( blockLen, nCells - b * blockLen );
        return true;
    }
    // ... and the block's index b, for a caller that keeps a result per block (psFxBoundWalk's per-block sums)
    bool claim(size_t const th, CellIt &first, size_t &cells, size_t &block)
    {
        if ( th == 0 && recruit ) recruit();
        size_t const b = next.fetch_add( 1, std::memory_order_relaxed );
        if ( b >= blockStart.size() ) return false;
        first = blockStart[b];
        cells = std::min( blockLen, nCells - b * blockLen );
        block = b;
        return true;
    }
};

// ---- the pass's per-tetrahedron tests ----

// --grid-window: a tetrahedron whose Eulerian footprint misses the window on some axis deposits
// nothing: skipped before extraction / sampling. The one-cell ring kept here for nSub > 1 is only
// generous (the deposits have no sample margin: no sample lies in it)
static inline bool psWindowTouched(PSPassCtx const &passCtx, Real const (&ep)[NO_DIM+1][NO_DIM])
{
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    Real (&dx)[NO_DIM] = passCtx.dx;
    int const nSub = passCtx.nSub;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    for (int d = 0; d < NO_DIM; ++d)
    {
        Real eLo = ep[0][d], eHi = ep[0][d];
        for (int v = 1; v <= NO_DIM; ++v) { eLo = std::min(eLo, ep[v][d]); eHi = std::max(eHi, ep[v][d]); }
        long lo = long(std::floor(double(eLo - boxCoordinates[2*d]) / double(dx[d])));
        long hi = long(std::floor(double(eHi - boxCoordinates[2*d]) / double(dx[d]))) + 1;
        if ( nSub > 1 && !userOptions.psExactDeposit ) { lo -= 1; hi += 1; }
        if ( !psSpanTouches(lo, hi, long(nGrid[d]), userOptions.periodic, long(subOrigin[d]), long(subDims[d])) )
            return false;
    }
    return true;
}

// The exact CPU deposit loop's own test of whether a tetrahedron reaches the grid (and, under --grid-window, the window), on
// its wrapped positions: the loop's bounding-box cell span formed in Real exactly as its "iMin / iMax" block (KEEP THE TWO IN
// STEP). psFaceWalk rejects on this test, not psWindowTouched: that one divides in double, and the loop's float quotient can
// round onto a grid plane where the double one does not, and it has no rule for a tet wholly outside a non-periodic grid.
// So the face sums beside a GPU deposit come from exactly the tets the CPU loop deposits faces for.
static inline bool psLoopSpanReaches(Real const (&ep)[NO_DIM+1][NO_DIM], Real const *boxL, Real const *dxL,
                                     size_t const *nGridL, bool const periodic, bool const windowMode,
                                     size_t const *subOriginL, size_t const *subDimsL)
{
    if ( periodic && !windowMode ) return true;   // (a periodic grid without a window: every kept tetrahedron reaches it)
    int iMinU[NO_DIM], iMaxU[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        Real eMin = ep[0][d], eMax = ep[0][d];
        for (int v = 1; v <= NO_DIM; ++v)
        {
            if (ep[v][d] < eMin) eMin = ep[v][d];
            if (ep[v][d] > eMax) eMax = ep[v][d];
        }
        iMinU[d] = int(floor((eMin - boxL[2*d]) / dxL[d]));
        iMaxU[d] = int(floor((eMax - boxL[2*d]) / dxL[d])) + 1;
        // the loop's "outsideGrid" (it tests the clamped span, which says the same)
        if ( !periodic && ( iMinU[d] >= (int)nGridL[d] || iMaxU[d] <= 0 ) ) return false;
    }
    if ( windowMode )
        for (int d = 0; d < NO_DIM; ++d)
            if ( !psSpanTouches(iMinU[d], iMaxU[d], long(nGridL[d]), periodic, long(subOriginL[d]), long(subDimsL[d])) )
                return false;
    return true;
}

// --face-gradient in a non-periodic run without a window: does a tet the span test puts OUTSIDE the grid own the grid's section in
// an UPPER boundary plane (the upper face of the last layer, which no tet inside owns: face_sweep.h's half-open rule)? True when,
// on every axis where the span starts at the grid's end, the lowest vertex lies in or below the plane (in double, against the same
// rounded nGrid * dx as psDepositFaces' first plane test) and on every other axis the span reaches the grid. Not under
// --grid-window. KEEP IN STEP with the loop's span block and psLoopSpanReaches.
static inline bool psOwnsUpperBoundarySection(Real const (&ep)[NO_DIM+1][NO_DIM], Real const *boxL, Real const *dxL,
                                              size_t const *nGridL, bool const periodic, bool const windowMode)
{
    if ( periodic || windowMode ) return false;   // (no boundary plane; a window: see above)
    bool atEnd = false;
    for (int d = 0; d < NO_DIM; ++d)
    {
        Real eMin = ep[0][d], eMax = ep[0][d];
        for (int v = 1; v <= NO_DIM; ++v)
        {
            if (ep[v][d] < eMin) eMin = ep[v][d];
            if (ep[v][d] > eMax) eMax = ep[v][d];
        }
        int const iMinU = int(floor((eMin - boxL[2*d]) / dxL[d]));
        int const iMaxU = int(floor((eMax - boxL[2*d]) / dxL[d])) + 1;
        if ( iMaxU <= 0 || iMinU > (int)nGridL[d] ) return false;   // below the grid, or a whole cell beyond its end
        if ( iMinU < (int)nGridL[d] ) continue;                       // reaches the grid along this axis
        // at the end: psDepositFaces' test (its first plane, ceil(min / dx), is nGrid exactly when min <= nGrid dx,
        // the same rounded double product as here)
        if ( double(eMin) - double(boxL[2*d]) > double(nGridL[d]) * double(dxL[d]) ) return false;
        atEnd = true;
    }
    return atEnd;
}

// The centroid-fallback cell of a tetrahedron: its Eulerian centroid (Real) and that centroid's flat sub-grid index; false when
// outside this partition region. ONE definition for the CPU fallback, the released tets and the GPU record: all three must put
// the mass in the same cell. The cell is found in double from the vertices' differences from vertex 0 and its offset from the box
// corner, reduced by the Real box length where periodic, so it is the same in every periodic image (a Real sum of the four
// positions rounds at the image's own ulp and could move a centroid near a plane). The returned centroid stays the Real one.
static inline bool psCentroidCell(PSPassCtx const &passCtx, Real const (&ep)[NO_DIM+1][NO_DIM], Real (&centroid)[NO_DIM],
                                  size_t &flat)
{
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    for (int d = 0; d < NO_DIM; ++d)
    {
        centroid[d] = Real(0.);
        for (int v = 0; v <= NO_DIM; ++v) centroid[d] += ep[v][d];
        centroid[d] /= Real(NO_DIM + 1);
    }
    flat = 0;
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const lo = double( boxCoordinates[2*d] );
        double const len = double( Real( boxCoordinates[2*d+1] - boxCoordinates[2*d] ) );   // the copies' period
        double rel = 0.;   // the centroid's offset from vertex 0
        for (int v = 1; v <= NO_DIM; ++v) rel += double( ep[v][d] ) - double( ep[0][d] );
        rel /= double( NO_DIM + 1 );
        double a0 = double( ep[0][d] ) - lo;   // vertex 0's offset from the box corner
        if ( userOptions.periodic ) a0 -= len * std::floor( a0 / len );
        int raw = int( std::floor( ( a0 + rel ) * double( nGrid[d] ) / len ) );
        int w = userOptions.periodic ? ((raw % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw;
        long loc = (long)w - (long)subOrigin[d];
        if (loc < 0) loc += (long)nGrid[d];        // sub-box may wrap a periodic axis
        if (w < 0 || w >= (int)nGrid[d] || loc < 0 || loc >= (long)subDims[d]) return false;
        flat = flat * subDims[d] + (size_t)loc;
    }
    return true;
}

// '.hidden_streams' bit 1 (flippedFlag in interpolateGrid_phaseSpace): a folded tetrahedron's overlap with the cells.
// xLo/xHi: only cells whose local x index lies in [xLo, xHi) -- the slab deposit's thread owns them;
// shared: the copy mode's threads write the same flags (relaxed atomics, psSetFlag; a flag only goes 0 -> 1)
static inline void psMarkFlipped(PSPassCtx const &passCtx, PSCellGeometry const &geo, std::vector<unsigned char> &flags,
                                 long const xLo, long const xHi, bool const shared)
{
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    Real (&dx)[NO_DIM] = passCtx.dx;
    bool const windowMode = passCtx.windowMode;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    auto isSet = [&flags, shared](size_t const f) -> bool
    { return shared ? __atomic_load_n( &flags[f], __ATOMIC_RELAXED ) != 0 : flags[f] != 0; };
#if NO_DIM==3
    if ( !(geo.cellDet < 0.) ) return;
    Real const (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
    double v[4][3], half[3];
    int wLo[3], wHi[3];
    size_t cells = 1;
    for (int d = 0; d < 3; ++d)
    {
        double eLo = double(ep[0][d]), eHi = eLo;
        for (int k = 0; k < 4; ++k)
        {
            v[k][d] = double(ep[k][d]) - double(ep[0][d]);
            eLo = std::min(eLo, double(ep[k][d]));
            eHi = std::max(eHi, double(ep[k][d]));
        }
        half[d] = 0.5 * double(dx[d]);
        wLo[d] = int(std::floor((eLo - double(boxCoordinates[2*d])) / double(dx[d])));
        wHi[d] = int(std::floor((eHi - double(boxCoordinates[2*d])) / double(dx[d]))) + 1;
        if ( !userOptions.periodic )
        {
            wLo[d] = std::max(wLo[d], 0);
            wHi[d] = std::min(wHi[d], (int)nGrid[d]);
            if ( wLo[d] >= wHi[d] ) return;
        }
        cells *= size_t(wHi[d] - wLo[d]);   // the whole bbox: 'cells == 1' below means the tet fits one cell
    }
    // --grid-window: only the window's cells are stored, so visit only the hull of the bbox with the
    // window's images (as the exact deposit does) -- on a zoom's fine virtual grid a folded void
    // tet's bbox holds millions of cells, and the whole walk would dominate the pass
    int lLo[3] = { wLo[0], wLo[1], wLo[2] }, lHi[3] = { wHi[0], wHi[1], wHi[2] };
    if ( windowMode )
        for (int d = 0; d < 3; ++d)
        {
            long hl = 0, hh = 0;
            if ( !psWindowHull(wLo[d], wHi[d], long(nGrid[d]), userOptions.periodic,
                               long(subOrigin[d]), long(subDims[d]), hl, hh) ) return;
            lLo[d] = int(hl); lHi[d] = int(hh);
        }
    TetBoxSAT const sat(v, half);
    for (int gi = lLo[0]; gi < lHi[0]; ++gi)
    for (int gj = lLo[1]; gj < lHi[1]; ++gj)
    for (int gk = lLo[2]; gk < lHi[2]; ++gk)
    {
        int const raw[3] = { gi, gj, gk };
        size_t flat = 0;
        bool inSub = true;
        for (int d = 0; d < 3; ++d)
        {
            int const w = userOptions.periodic ? ((raw[d] % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw[d];
            long loc = (long)w - (long)subOrigin[d];
            if (loc < 0) loc += (long)nGrid[d];        // sub-box may wrap a periodic axis
            if (loc < 0 || loc >= (long)subDims[d]) { inSub = false; break; }
            if (d == 0 && (loc < xLo || loc >= xHi)) { inSub = false; break; }   // another slab's cell
            flat = flat * subDims[d] + (size_t)loc;
        }
        if ( !inSub || isSet(flat) ) continue;
        if ( cells == 1 ) { psSetFlag( flags[flat], shared ); continue; }   // the whole (non-degenerate) tet is in this cell
        double c[3];
        for (int d = 0; d < 3; ++d)
            c[d] = double(boxCoordinates[2*d]) + (raw[d] + 0.5) * double(dx[d]) - double(ep[0][d]);
        if ( sat.overlaps(c) ) psSetFlag( flags[flat], shared );
    }
#else
    // 2D: a folded triangle overlapping the cell with positive area (ps_clip2d.h); under
    // --grid-window only the hull of the bbox with the window's images, as in 3D
    if ( !(geo.cellDet < 0.) ) return;
    Real const (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
    double tri[NO_DIM+1][NO_DIM];
    int wLo[NO_DIM], wHi[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double eLo = double(ep[0][d]), eHi = eLo;
        for (int k = 0; k <= NO_DIM; ++k)
        {
            tri[k][d] = double(ep[k][d]) - double(ep[0][d]);
            eLo = std::min(eLo, double(ep[k][d]));
            eHi = std::max(eHi, double(ep[k][d]));
        }
        wLo[d] = int(std::floor((eLo - double(boxCoordinates[2*d])) / double(dx[d])));
        wHi[d] = int(std::floor((eHi - double(boxCoordinates[2*d])) / double(dx[d]))) + 1;
        if ( !userOptions.periodic )
        {
            wLo[d] = std::max(wLo[d], 0);
            wHi[d] = std::min(wHi[d], (int)nGrid[d]);
            if ( wLo[d] >= wHi[d] ) return;
        }
    }
    for (int d = 0; d < NO_DIM; ++d) std::swap( tri[1][d], tri[2][d] );   // folded: counter-clockwise now
    if ( windowMode )
        for (int d = 0; d < NO_DIM; ++d)
        {
            long hl = 0, hh = 0;
            if ( !psWindowHull(wLo[d], wHi[d], long(nGrid[d]), userOptions.periodic,
                               long(subOrigin[d]), long(subDims[d]), hl, hh) ) return;
            wLo[d] = int(hl); wHi[d] = int(hh);
        }
    for (int gi = wLo[0]; gi < wHi[0]; ++gi)
    for (int gj = wLo[1]; gj < wHi[1]; ++gj)
    {
        int const raw[NO_DIM] = { gi, gj };
        size_t flat = 0;
        bool inSub = true;
        for (int d = 0; d < NO_DIM; ++d)
        {
            int const w = userOptions.periodic ? ((raw[d] % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw[d];
            long loc = (long)w - (long)subOrigin[d];
            if (loc < 0) loc += (long)nGrid[d];
            if (loc < 0 || loc >= (long)subDims[d]) { inSub = false; break; }
            if (d == 0 && (loc < xLo || loc >= xHi)) { inSub = false; break; }   // another slab's cell
            flat = flat * subDims[d] + (size_t)loc;
        }
        if ( !inSub || isSet(flat) ) continue;
        double lo[NO_DIM], hi[NO_DIM], mom[psClip2d::MOMENTS];
        for (int d = 0; d < NO_DIM; ++d)
        {
            lo[d] = double(boxCoordinates[2*d]) + raw[d] * double(dx[d]) - double(ep[0][d]);
            hi[d] = lo[d] + double(dx[d]);
        }
        if ( psClip2d::clipTriangleToCell( tri, lo, hi, mom ) ) psSetFlag( flags[flat], shared );
    }
#endif
}
