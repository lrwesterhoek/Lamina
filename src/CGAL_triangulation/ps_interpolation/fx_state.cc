// Part of ps_interpolation.cc (included in order inside its #ifdef PHASE_SPACE, never compiled alone). The pass's
// fixed-point moment state (PSFxState), the walk that bounds its words (psFxBoundWalk) and the face walk (psFaceWalk).

// The pass's moment grids as fixed-point sums (fxg in interpolateGrid_phaseSpace): which grids it accumulates, their words,
// their Real accumulators and their bounds and units. Used per pass, never per cell.
struct PSFxState
{
    PSFxWords (&fxg)[PS_FX_CPU_N];
    bool const (&fxSel)[PS_FX_CPU_N];
    User_options &userOptions;
    Quantities *quantities;
    Field &field;
    size_t totalGrid;
    std::vector<Real> &massWeight, &exactMult, &dispWeight, &releasedMult;
    std::vector< Pvector<Real,noVelComp> > &dispVel;
    bool fxVelBound, fxScalSel, fxSGradSel;
    Real cellVolume;
    // --face-gradient: this pass accumulates the face sums (the exact CPU deposit): words faceS (9 per cell), faceN (NO_DIM),
    // outside the PSFxGrid set so the GPU's grids and counters are untouched, and the grid spacing their bound counts planes with
    bool face = false;
    PSFxWords faceS, faceN, faceP;   // (faceP: the normal momentum through each face, the mass flux)
    Real const *gridDx = nullptr;
    // the bounds the GPU extraction summed for this pass (same tets, same block order), kept for the face sums' units so no
    // second bound walk runs after a GPU deposit (psFxBoundsFromGpu says when they stand in for the walk's)
    PSFixedBounds gpuBounds;
    bool gpuBoundsValid = false;
    // the face walk ran beside the GPU deposit (the default, PSEarlyFaceWalk) and the device call succeeded:
    // the face words are filled and their units set, so psCpuFixedUnits and the walk after the deposit leave them
    // alone; and that walk's wall time
    bool faceWalked = false;
    double faceWalkSec = 0.;
    // forgets the face words and their units (a walk beside the GPU deposit that is thrown away)
    void faceRelease()
    {
        for (PSFxWords *const p : { &faceS, &faceN, &faceP }) p->release();
    }

    // the Real accumulator of grid g, allocated (zeroed) if the pass has none yet: a grid that is not a fixed-point
    // sum after all (no unit, or the GPU's float grid)
    void ensureReal(int const g)
    {
        if ( !fxSel[g] ) return;
        switch ( g )
        {
        case PS_FX_MOM:  if ( quantities->velocity.empty() ) quantities->velocity.assign(totalGrid, Pvector<Real,noVelComp>::zero()); break;
        case PS_FX_M2:   if ( quantities->velocity_dispersion.empty() ) quantities->velocity_dispersion.assign(totalGrid, Pvector<Real,noDispComp>::zero()); break;
        case PS_FX_GRAD: if ( quantities->velocity_gradient.empty() ) quantities->velocity_gradient.assign(totalGrid, Pvector<Real,noGradComp>::zero()); break;
        case PS_FX_MOMW: if ( massWeight.empty() ) massWeight.assign(totalGrid, Real(0.)); break;
        case PS_FX_SV:   if ( exactMult.empty() ) exactMult.assign(totalGrid, Real(0.)); break;
        case PS_FX_DV:   if ( dispVel.empty() ) dispVel.assign(totalGrid, Pvector<Real,noVelComp>::zero()); break;
        case PS_FX_DW:   if ( dispWeight.empty() ) dispWeight.assign(totalGrid, Real(0.)); break;
#ifdef SCALAR
        case PS_FX_SCAL: if ( quantities->scalar.empty() ) quantities->scalar.assign(totalGrid, Pvector<Real,noScalarComp>::zero()); break;
        case PS_FX_SG:   if ( quantities->scalar_gradient.empty() ) quantities->scalar_gradient.assign(totalGrid, Pvector<Real,noScalarGradComp>::zero()); break;
#endif
        case PS_FX_RM:   if ( releasedMult.empty() ) releasedMult.assign(totalGrid, Real(0.)); break;
        default: break;
        }
    }
    // components per cell of grid g (the words' layout)
    size_t comps(int const g) const
    {
        switch ( g )
        {
        case PS_FX_MOM: case PS_FX_DV: return noVelComp;
        case PS_FX_M2:   return noDispComp;
        case PS_FX_GRAD: return noGradComp;
#ifdef SCALAR
        case PS_FX_SCAL: return noScalarComp;
        case PS_FX_SG:   return noScalarGradComp;
#endif
        default: return 1;
        }
    }
    // grid g becomes a fixed-point sum in units of 1/scale (its 64-bit words zeroed, its Real accumulator released)
    void start(int const g, float const scale)
    {
        fxg[g].scale = scale;
        fxg[g].w.assign( totalGrid * comps(g), 0 );
        switch ( g )
        {
        case PS_FX_MOM:  std::vector< Pvector<Real,noVelComp> >().swap( quantities->velocity ); break;
        case PS_FX_M2:   std::vector< Pvector<Real,noDispComp> >().swap( quantities->velocity_dispersion ); break;
        case PS_FX_GRAD: std::vector< Pvector<Real,noGradComp> >().swap( quantities->velocity_gradient ); break;
        case PS_FX_MOMW: std::vector<Real>().swap( massWeight ); break;
        case PS_FX_SV:   std::vector<Real>().swap( exactMult ); break;
        case PS_FX_DV:   std::vector< Pvector<Real,noVelComp> >().swap( dispVel ); break;
        case PS_FX_DW:   std::vector<Real>().swap( dispWeight ); break;
#ifdef SCALAR
        case PS_FX_SCAL: std::vector< Pvector<Real,noScalarComp> >().swap( quantities->scalar ); break;
        case PS_FX_SG:   std::vector< Pvector<Real,noScalarGradComp> >().swap( quantities->scalar_gradient ); break;
#endif
        case PS_FX_RM:   std::vector<Real>().swap( releasedMult ); break;
        default: break;
        }
    }
    // grid g's words become its Real grid, once (after the deposit); a no-op for a grid without words
    void toReal(int const g)
    {
        PSFxWords &F = fxg[g];
        if ( F.w.empty() ) return;
        switch ( g )
        {
        case PS_FX_MOM:  psFxToReal( F, quantities->velocity, totalGrid ); break;
        case PS_FX_M2:   psFxToReal( F, quantities->velocity_dispersion, totalGrid ); break;
        case PS_FX_GRAD: psFxToReal( F, quantities->velocity_gradient, totalGrid ); break;
        case PS_FX_MOMW: psFxToReal( F, massWeight, totalGrid ); break;
        case PS_FX_SV:   psFxToReal( F, exactMult, totalGrid ); break;
        case PS_FX_DV:   psFxToReal( F, dispVel, totalGrid ); break;
        case PS_FX_DW:   psFxToReal( F, dispWeight, totalGrid ); break;
#ifdef SCALAR
        case PS_FX_SCAL: psFxToReal( F, quantities->scalar, totalGrid ); break;
        case PS_FX_SG:   psFxToReal( F, quantities->scalar_gradient, totalGrid ); break;
#endif
        case PS_FX_RM:   psFxToReal( F, releasedMult, totalGrid ); break;
        default: break;
        }
    }
    // One kept tet's terms of the bounds (PSFixedBounds::addTet): the largest |value| it can deposit per grid, from the vertex
    // values and the gradients the CPU deposit forms (geo.posMatInv, Real). The GPU kernel's float gradients may exceed these
    // near the degeneracy cut; the factor-4 headroom in the unit absorbs it (psMomentFixedScale).
    void tetBound(Cell_handle const &cell, PSCellGeometry const &geo, double const tetMass, PSFixedBounds &fb) const
    {
        double const V = geo.cellAbsDet / double(factorial(NO_DIM));
        double const wm = V;
        double vMax = 0., gMax = 0., sMax = 0., sgMax = 0.;
        if ( fxVelBound )
        {
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noVelComp; ++j)
                    vMax = std::max( vMax, std::fabs( double( cell->vertex(v)->info().velocity(j) ) ) );
            if ( field.velocity_gradient )
            {
                Real temp[NO_DIM][noVelComp], vg[NO_DIM][noVelComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t j = 0; j < noVelComp; ++j)
                        temp[v][j] = cell->vertex(v+1)->info().velocity(j) - cell->vertex(0)->info().velocity(j);
                matrixMultiplication<noVelComp>( const_cast<Real (&)[NO_DIM][NO_DIM]>(geo.posMatInv), temp, vg );
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noVelComp; ++j) gMax = std::max( gMax, std::fabs( double(vg[i][j]) ) );
            }
        }
#ifdef SCALAR
        if ( fxScalSel || fxSGradSel )
        {
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noScalarComp; ++j)
                    sMax = std::max( sMax, std::fabs( double( cell->vertex(v)->info().myScalar()[j] ) ) );
            if ( fxSGradSel )
            {
                Real temp[NO_DIM][noScalarComp], sg[NO_DIM][noScalarComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t j = 0; j < noScalarComp; ++j)
                        temp[v][j] = cell->vertex(v+1)->info().myScalar()[j] - cell->vertex(0)->info().myScalar()[j];
                matrixMultiplication<noScalarComp>( const_cast<Real (&)[NO_DIM][NO_DIM]>(geo.posMatInv), temp, sg );
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noScalarComp; ++j) sgMax = std::max( sgMax, std::fabs( double(sg[i][j]) ) );
            }
        }
#endif
        fb.addTet( tetMass, V, wm, vMax, gMax, sMax, sgMax, double(cellVolume) );
#if NO_DIM==3
        // the face sums: the tet's sections, summed over the planes it crosses, cover at most (planes) x (its bounding
        // box's cross-section) per axis; a face share is at most vMax times a piece of that area (the profile is a
        // convex combination of the vertex values)
        if ( face && gridDx )
        {
            double vM = vMax;
            if ( !fxVelBound )
                for (int v = 0; v <= NO_DIM; ++v)
                    for (size_t j = 0; j < noVelComp; ++j)
                        vM = std::max( vM, std::fabs( double( cell->vertex(v)->info().velocity(j) ) ) );
            double ext[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
            {
                double lo = double( geo.eulerPos[0][d] ), hi = lo;
                for (int v = 1; v <= NO_DIM; ++v)
                {
                    lo = std::min( lo, double( geo.eulerPos[v][d] ) );
                    hi = std::max( hi, double( geo.eulerPos[v][d] ) );
                }
                ext[d] = hi - lo;
            }
            double S = 0.;
            for (int a = 0; a < NO_DIM; ++a)
            {
                int u, w;
                faceSweep::planeAxes( a, u, w );
                S += ( std::floor( ext[a] / double( gridDx[a] ) ) + 2. ) * ext[u] * ext[w];
            }
            double const rhoTet = ( V > 0. && double(userOptions.averageDensity) > 0. )
                                  ? tetMass / ( V * double(userOptions.averageDensity) ) : 0.;
            fb.addFace( vM, S, rhoTet );
        }
#endif
    }
    // the unit of grid g from its bound; 0 (and a warning naming the grid) when the bound has none: the grid then
    // keeps the float sums.
    float unit(int const g, double const bound) const
    {
        static char const *const name[PS_FX_CPU_N] = { "velocity moment", "dispersion second moment", "velocity-gradient moment",
            "moment weight", "exact multiplicity", "dispersion mean", "dispersion weight", "scalar moment",
            "scalar-gradient moment", "released multiplicity" };
        float scale = 0.f;
        if ( psMomentFixedScale( bound, scale ) ) return scale;
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "PS-DTFE: the " << name[g] << " grid's bound (" << bound << ") gives no fixed-point unit; this pass "
                << "sums it in floating point (not bit-reproducible).\n" << MESSAGE::EndWarning;
        return 0.f;
    }
    // --verbose 3: the pass's fixed-point units, one line -- the mass grid's and each moment grid's
    // 2^k (DW, the mass grid's twin, left out) -- the same for a GPU and a CPU deposit of one pass
    // (tests/ps_window_check.sh compares a GPU and a CPU --grid-window run's lines).
    void logUnits(char const *path, float const massScale, float const (&scale)[PS_FX_N]) const
    {
        if ( userOptions.verboseLevel < 3 ) return;
        std::ostringstream line;
        line << "PS-DTFE fixed-point units (" << path << "): mass ";
        if ( massScale > 0.f ) line << "2^" << std::ilogb( massScale ); else line << "float";
        for (int g = 0; g < PS_FX_N; ++g)
            if ( g != PS_FX_DW && scale[g] > 0.f ) line << ", [" << g << "] 2^" << std::ilogb( scale[g] );
        std::fprintf( stderr, "%s\n", line.str().c_str() );
    }
};

// The bounds of the pass's fixed-point units: one walk over the cells with the deposit's own filter and tet mass, summing
// PSFixedBounds per tet. Cells go in blocks of 256 claimed by the team's threads (PSBlockClaim); each block's sums are made by
// the one thread that claimed it, from zero, in the cells' order, then added in block order: the 13 doubles are the same at every
// thread count and claim order, and the same a GPU deposit takes (gpuBoundPart). --grid-window: only tets that reach the window
// count (windowTouched, as the GPU extraction), so GPU and CPU window deposits take the same units.
static PSFixedBounds psFxBoundWalk(PSPassCtx const &passCtx, PSFxState const &fx, bool const psVertexMass)
{
    DT &dt = passCtx.dt;   // (PS_FINITE_CELLS_BEGIN / END)
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const noTotalCells = passCtx.noTotalCells;
    bool const windowMode = passCtx.windowMode;
    auto windowTouched = [&passCtx](Real const (&ep)[NO_DIM+1][NO_DIM]) { return psWindowTouched( passCtx, ep ); };
    auto const outsideWindow = [&](Real const (&ep)[NO_DIM+1][NO_DIM]) { return windowMode && !windowTouched(ep); };
    size_t const BLK = 256;
    size_t const nBlocks = noTotalCells / BLK + 1;
    std::vector<PSFixedBounds> part( nBlocks );
    size_t const nT = size_t( std::max( 1, std::min( psDepositThreads(), int( std::min<size_t>( nBlocks, 1024 ) ) ) ) );
    // the block index: each block's first cell, from one walk over the iterator (the blocks of the static rule: cell
    // ordinals [256 b, 256 b + 256)); no recruit hook, the team is the nT threads below
    using BoundCellIt = decltype( PS_FINITE_CELLS_BEGIN );
    PSBlockClaim<BoundCellIt> claim;
    claim.blockLen = BLK;
    claim.blockStart.reserve( nBlocks );
    {
        size_t k = 0;
        for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END; ++it, ++k)
            if ( k % BLK == 0 ) claim.blockStart.push_back( it );
        claim.nCells = k;
    }
    auto boundRun = [&](size_t const th)
    {
        auto it = PS_FINITE_CELLS_BEGIN;
        size_t cells = 0, blk = 0;
        while ( claim.claim( th, it, cells, blk ) )   // a whole block, its cells in the loop's order
            for (size_t c = 0; c < cells; ++c, ++it)
            {
                Cell_handle cell = it;
                PSCellGeometry geo;
                if ( !psFilterCell( cell, userOptions, boxCoordinates, true, NULL, geo, false, outsideWindow ) )
                    continue;
                double tetMass;
                if ( psVertexMass ) tetMass = double( Real( psVertexShareMass(cell) ) );
                else
                {
                    double Lag[NO_DIM][NO_DIM];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (int i = 0; i < NO_DIM; ++i)
                            Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
                    tetMass = double( Real( double(userOptions.averageDensity) * std::fabs(determinant(Lag))
                                            / factorial(NO_DIM) ) );
                }
                if ( !(tetMass > 0.) ) tetMass = 0.;   // (the deposit clamps a negative mass to 0)
                fx.tetBound( cell, geo, tetMass, part[blk] );
            }
    };
    {
        CorePool::Hold const hold( int(nT) - 1 );   // (a fixed team: its cores counted in the pool, thread_count.h)
        std::vector<std::thread> pool;
        for (size_t th = 1; th < nT; ++th) pool.emplace_back( boundRun, th );
        boundRun( 0 );
        for (std::thread &x : pool) x.join();
    }
    PSFixedBounds fb;
    for (PSFixedBounds const &p : part) fb.add( p );   // in block order: thread-count and claim-order independent
    // DTFE_DEBUG_FX_BOUNDS: the walk's 13 sums exactly (hex), with its team and blocks, one line per walk: the same
    // sums at every thread count (tests/ps_fixed_moments_check.sh F6)
    if ( Env::flag( "DTFE_DEBUG_FX_BOUNDS" ) )
    {
        std::ostringstream line;
        line << "DTFE_DEBUG_FX_BOUNDS: bound walk on " << nT << " thread(s), " << claim.blockStart.size()
             << " blocks; sums";
        char word[48];
        for (int g = 0; g < PS_FX_CPU_N; ++g)
        {
            std::snprintf( word, sizeof(word), " %a", fb.B[g] );
            line << word;
        }
        for (double const v : { fb.faceS, fb.faceN, fb.faceP })
        {
            std::snprintf( word, sizeof(word), " %a", v );
            line << word;
        }
        std::fprintf( stderr, "%s\n", line.str().c_str() );
    }
    return fb;
}

#if NO_DIM==3
// --face-gradient beside a GPU deposit: the face sums (psDepositFaces) of every kept tetrahedron from ONE CPU walk, on the bound
// walk's skeleton (same filter, mass, gradient and block order as the deposit loop, GPU-deferred tets included), then the
// deposit loop's span test or face-only test (psOwnsUpperBoundarySection). It writes the face words only, shared by its threads
// (relaxed atomics, order-free). 'threads': the walk's team (it runs outside the partition loop's team). 'dropsOut': where
// dropped shares are counted (FX_DROP_SLOTS counters, or null: the run's), for a walk whose sums may be thrown away.
static void psFaceWalk(PSPassCtx const &passCtx, PSFxState &fx, bool const psVertexMass, int const threads,
                       unsigned long long *const dropsOut)
{
    DT &dt = passCtx.dt;   // (PS_FINITE_CELLS_BEGIN / END)
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const noTotalCells = passCtx.noTotalCells;
    bool const windowMode = passCtx.windowMode;
    Real boxL[2*NO_DIM], dxL[NO_DIM];
    size_t nGridL[NO_DIM], subOriginL[NO_DIM], subDimsL[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        boxL[2*d] = boxCoordinates[2*d]; boxL[2*d+1] = boxCoordinates[2*d+1];
        dxL[d] = passCtx.dx[d]; nGridL[d] = passCtx.nGrid[d];
        subOriginL[d] = passCtx.subOrigin[d]; subDimsL[d] = passCtx.subDims[d];
    }
    bool const periodic = userOptions.periodic;
    double const avgDensity = double( userOptions.averageDensity );
    size_t const BLK = 256;
    size_t const nBlocks = noTotalCells / BLK + 1;
    size_t const nT = size_t( std::max( 1, std::min( threads, int( std::min<size_t>( nBlocks, 1024 ) ) ) ) );
    bool const atomic = nT > 1;
    std::vector<unsigned long long> dropPart( dropsOut ? nT * FX_DROP_SLOTS : 0, 0ull );   // (per walk thread)
    auto faceRun = [&](size_t const th)
    {
        FixedShareDropSinkScope const sink( dropsOut ? &dropPart[th * FX_DROP_SLOTS] : nullptr );
        PSFxSink const S = fx.faceS.sink( th ), N = fx.faceN.sink( th ), P = fx.faceP.sink( th );
        size_t idx = 0;
        for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END; ++it, ++idx)
        {
            if ( ( idx / BLK ) % nT != th ) continue;
            Cell_handle cell = it;
            PSCellGeometry geo;
            // (the span test as the filter's early reject: on the wrapped positions the loop tests, before the inverse;
            // with the loop's face-only case, a tetrahedron outside the grid that owns an upper boundary section)
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, NULL, geo, false,
                               [&](Real const (&ep)[NO_DIM+1][NO_DIM])
                               { return !psLoopSpanReaches( ep, boxL, dxL, nGridL, periodic, windowMode, subOriginL, subDimsL )
                                        && !psOwnsUpperBoundarySection( ep, boxL, dxL, nGridL, periodic,
                                                                        windowMode ); }) )
                continue;
            double tetMass;   // (the deposit loop's: psVertexShareMass, or rho_bar times the Lagrangian volume, in Real)
            if ( psVertexMass ) tetMass = double( Real( psVertexShareMass(cell) ) );
            else
            {
                double Lag[NO_DIM][NO_DIM];
                for (int v = 0; v < NO_DIM; ++v)
                    for (int i = 0; i < NO_DIM; ++i)
                        Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
                tetMass = double( Real( avgDensity * std::fabs(determinant(Lag)) / factorial(NO_DIM) ) );
            }
            // the deposit loop's clamp: a negative mass to 0, a NaN kept, so that its momentum shares (P) are left
            // out and counted here as the loop counts them (a zero share would add the same nothing, uncounted)
            if ( tetMass < 0. ) tetMass = 0.;
            // DTFE_TEST_NAN_SHARE (ps_deposit_params.h), after the clamp as in the deposit loop: the hooked tetrahedron's
            // density is NaN there, so its momentum shares (P) are dropped and counted; the same here
            if ( fixedShareTestNanEvery() && fixedShareTestNan( double(cell->vertex(0)->point()[0]), double(cell->vertex(0)->point()[1]),
                                                                double(cell->vertex(0)->point()[NO_DIM-1]) ) )
                tetMass = std::numeric_limits<double>::quiet_NaN();
            // the constant velocity gradient, as the deposit loop forms it (geo.posMatInv, Real)
            Vertex_handle base = cell->vertex(0);
            Real velGrad[NO_DIM][noVelComp], temp[NO_DIM][noVelComp];
            for (int v = 0; v < NO_DIM; ++v)
                for (size_t i = 0; i < noVelComp; ++i)
                    temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
            matrixMultiplication<noVelComp>(geo.posMatInv, temp, velGrad);
            double v0[noVelComp];
            for (size_t j = 0; j < noVelComp; ++j) v0[j] = double( base->info().velocity(j) );
            double const vTet = double(geo.cellAbsDet) / double(factorial(NO_DIM));
            double const rhoTet = ( vTet > 0. && avgDensity > 0. ) ? tetMass / ( vTet * avgDensity ) : 0.;
            psDepositFaces( geo.eulerPos, boxL, dxL, nGridL, periodic, subOriginL, subDimsL, 0L, LONG_MAX,
                            velGrad, v0, S, N, P, rhoTet, atomic );
        }
    };
    {
        CorePool::Hold const hold( int(nT) - 1 );   // (a static team: its cores counted in the pool, thread_count.h)
        std::vector<std::thread> pool;
        for (size_t th = 1; th < nT; ++th) pool.emplace_back( faceRun, th );
        faceRun( 0 );
        for (std::thread &x : pool) x.join();
    }
    if ( dropsOut )
        for (size_t th = 0; th < nT; ++th)
            for (int k = 0; k < FX_DROP_SLOTS; ++k) dropsOut[k] += dropPart[th * FX_DROP_SLOTS + k];
}
#endif
