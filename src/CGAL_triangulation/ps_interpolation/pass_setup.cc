// Part of ps_interpolation.cc (included in order inside its #ifdef PHASE_SPACE, never compiled alone). The pass's setup:
// sub-grid box, sample positions, audit count, moment grids, GPU eligibility, debug self-checks, fixed-point units.

// The pass's sub-grid box: the --grid-window box, or (psUseSubgrid) the Eulerian bbox of the partition's kept cells.
static inline void psSubgridBox(PSPassCtx const &passCtx, size_t &psKeptCells)
{
    DT &dt = passCtx.dt;   // (PS_FOREACH_FINITE_CELL)
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    Real (&dx)[NO_DIM] = passCtx.dx;
    int const nSub = passCtx.nSub;
    bool const windowMode = passCtx.windowMode;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    if ( userOptions.psUseSubgrid && !windowMode )
    {
        psKeptCells = 0;
        std::vector<char> touched[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d) touched[d].assign(nGrid[d], char(0));
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry geo;   // dummy/ownership/hull/degeneracy filters (no inverse check here)
            if ( !psFilterCell(cell, userOptions, boxCoordinates, false, NULL, geo) ) continue;
            Real (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
            ++psKeptCells;
            Real eLo[NO_DIM],eHi[NO_DIM];
            for (int d=0;d<NO_DIM;++d){eLo[d]=ep[0][d];eHi[d]=ep[0][d];}
            for (int v=1;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d){ if(ep[v][d]<eLo[d])eLo[d]=ep[v][d]; if(ep[v][d]>eHi[d])eHi[d]=ep[v][d]; }
            for (int d=0;d<NO_DIM;++d)
            {
                int iLo=int(floor((eLo[d]-boxCoordinates[2*d])/dx[d]));
                int iHi=int(floor((eHi[d]-boxCoordinates[2*d])/dx[d]))+1;
                if (nSub>1){iLo-=1;iHi+=1;}
                if (!userOptions.periodic){ if(iLo<0)iLo=0; if(iHi>(int)nGrid[d])iHi=(int)nGrid[d]; }
                for (int g=iLo; g<iHi; ++g){ int w = userOptions.periodic ? ((g%(int)nGrid[d]+(int)nGrid[d])%(int)nGrid[d]) : g; if (w>=0 && w<(int)nGrid[d]) touched[d][w]=char(1); }
            }
        }
        for (int d = 0; d < NO_DIM; ++d)
        {
            int lo=-1, hi=-1;
            for (int g=0; g<(int)nGrid[d]; ++g) if (touched[d][g]){ if(lo<0)lo=g; hi=g; }
            if (lo<0) { subOrigin[d]=0; subDims[d]=nGrid[d]; }                                 // no cells touched: keep full axis (harmless)
            else if (touched[d][0] && touched[d][nGrid[d]-1])
            {
                // Touched planes reach both ends: the NORMAL case under --periodic for a partition straddling the box seam, and
                // keeping the full axis would make every such slab allocate the whole grid. The footprint is a WRAPPED arc, so crop
                // to it: its complement is the largest run of untouched planes, strictly interior, found by a linear scan.
                int bestLen=0, bestStart=-1, curStart=-1;
                for (int g=0; g<(int)nGrid[d]; ++g)
                {
                    if (!touched[d][g]) { if (curStart<0) curStart=g; }
                    else if (curStart>=0)
                    {
                        int const len = g - curStart;
                        if (len > bestLen) { bestLen=len; bestStart=curStart; }
                        curStart = -1;
                    }
                }
                if (bestLen<=0) { subOrigin[d]=0; subDims[d]=nGrid[d]; }                       // every plane touched: genuinely full
                else { subOrigin[d]=(size_t)((bestStart+bestLen)%(int)nGrid[d]); subDims[d]=(size_t)((int)nGrid[d]-bestLen); }
            }
            else { subOrigin[d]=(size_t)lo; subDims[d]=(size_t)(hi-lo+1); }                    // crop to the touched [lo,hi] span
        }
    }
    if ( windowMode )
        for (int d = 0; d < NO_DIM; ++d) { subOrigin[d] = userOptions.psWindowLo[d]; subDims[d] = userOptions.psWindowDims[d]; }
}

// The sample positions per axis, in double (samplePosAxis[d][g * nSub + k]), and the grid's lengths.
static inline void psSamplePositions(PSPassCtx const &passCtx, double (&gridLenD)[NO_DIM],
                                     std::vector<double> (&samplePosAxis)[NO_DIM])
{
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    int const nSub = passCtx.nSub;
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const lo = double( boxCoordinates[2*d] );
        gridLenD[d] = double( boxCoordinates[2*d+1] ) - lo;
        samplePosAxis[d].resize( nGrid[d] * size_t(nSub) );
        for (size_t g = 0; g < nGrid[d]; ++g)
            for (int k = 0; k < nSub; ++k)
            {
                double const frac = (nSub == 1) ? 0.5 : (double(k) + 0.5) / double(nSub);
                samplePosAxis[d][g * size_t(nSub) + size_t(k)] = lo + ((double(g) + frac) * gridLenD[d]) / double(nGrid[d]);
            }
    }
}

// DTFE_MASS_AUDIT: counts what the pass should deposit, before it deposits (a template: the audit is the pass's type).
template <class MassAudit, class FullDegreePass>
static inline void psMassAuditCount(PSPassCtx const &passCtx, bool const massAudit, bool const psVertexMass,
                                    bool &degreesLazy, FullDegreePass const &fullDegreePass, MassAudit &audit)
{
    DT &dt = passCtx.dt;
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    if ( massAudit )
    {
        if ( degreesLazy ) { fullDegreePass(); degreesLazy = false; }   // the same counts (DTFE_DEBUG_DEGREES)
        Box &own = userOptions.lagrangianRegion;
        bool const ownsAll = own.isNullBox();
        auto owned = [&](double const (&p)[NO_DIM]) -> bool   // psFilterCell's half-open ownership test
        {
            if ( ownsAll ) return true;
            for (int d = 0; d < NO_DIM; ++d)
                if ( p[d] < own[2*d] || p[d] >= own[2*d+1] ) return false;
            return true;
        };
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
        {
#ifdef TEST_PADDING
            if ( vit->info().isDummy() ) continue;
#endif
            double p[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) p[d] = double(vit->point()[d]);
            if ( !owned(p) ) continue;
            double const w = double(vit->info().weight());
            audit.owned += w; ++audit.nOwned;
            if ( psVertexMass && vit->info().psDegree() == 0 ) { audit.deg0 += w; ++audit.nDeg0; }
        }
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry g;
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            double const lagMass = double(userOptions.averageDensity) * std::fabs(determinant(Lag)) / factorial(NO_DIM);
            if ( psFilterCell(cell, userOptions, boxCoordinates, true, NULL, g) )
            {
                audit.tets += psVertexMass ? psVertexShareMass(cell) : lagMass;
                ++audit.nTets;
                if ( psVertexMass )
                    for (int v = 0; v <= NO_DIM; ++v)
                        if ( cell->vertex(v)->info().psDegree() == 0 ) { ++audit.nTetsDeg0; break; }   // must stay 0
                continue;
            }
            bool dummy = false;
#ifdef TEST_PADDING
            for (int v = 0; v <= NO_DIM; ++v) dummy = dummy || cell->vertex(v)->info().isDummy();
#endif
            double cen[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
            {
                cen[d] = 0.;
                for (int v = 0; v <= NO_DIM; ++v) cen[d] += double(cell->vertex(v)->point()[d]);
                cen[d] /= double(NO_DIM + 1);
            }
            if ( !dummy && owned(cen) ) { audit.dropped += lagMass; ++audit.nDropped; }
        }
        // DTFE_MASS_AUDIT=2 on one domain: the per-particle closure -- a particle's shares over the kept tets
        // of ALL its copies (periodic images) must add up to its weight; the worst offenders are listed
        if ( PSMassAudit::level() >= 2 && PSMassAudit::partition < 0 && psVertexMass )
        {
            std::unordered_map<uint64_t, double> share;
            std::unordered_map<uint64_t, std::vector<Vertex_handle> > copies;
            for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            {
#ifdef TEST_PADDING
                if ( vit->info().isDummy() ) continue;
#endif
                copies[ vit->info().particleID() ].push_back( vit );
            }
            PS_FOREACH_FINITE_CELL
            {
                Cell_handle cell = itC;
                PSCellGeometry g;
                if ( !psFilterCell(cell, userOptions, boxCoordinates, true, NULL, g) ) continue;
                for (int v = 0; v <= NO_DIM; ++v)
                {
                    int32_t const deg = cell->vertex(v)->info().psDegree();
                    if ( deg > 0 ) share[ cell->vertex(v)->info().particleID() ] += double(cell->vertex(v)->info().weight()) / double(deg);
                }
            }
            std::vector< std::pair<double, uint64_t> > bad;
            double net = 0.;
            for (auto const &c : copies)
            {
                double const w = double( c.second[0]->info().weight() );
                double const dev = share[c.first] - w;
                if ( std::fabs(dev) > 1e-9 * w ) { bad.emplace_back( dev, c.first ); net += dev; }
            }
            std::sort( bad.begin(), bad.end(), [](auto const &a, auto const &b) { return std::fabs(a.first) > std::fabs(b.first); } );
            std::ostringstream s;
            s << std::setprecision(9) << "PS mass audit: per-particle closure: " << bad.size() << " of " << copies.size()
              << " particles off (net " << net << ")";
            for (size_t i = 0; i < bad.size() && i < 6; ++i)
            {
                s << "\n    id " << bad[i].second << " off " << bad[i].first << ", copies (Lagrangian position: degree):";
                for (Vertex_handle const &vh : copies[bad[i].second])
                {
                    s << " (";
                    for (int d = 0; d < NO_DIM; ++d) s << vh->point()[d] << (d + 1 < NO_DIM ? " " : ": ");
                    s << vh->info().psDegree() << ")";
                }
            }
            std::fprintf( stderr, "%s\n", s.str().c_str() );
        }
    }
}

// Allocates the moment grids' Real accumulators of a float-sum pass, or releases them for the fixed-point words.
static inline void psAllocMoments(PSPassCtx const &passCtx, bool const haveVel, bool const fxMomWanted, size_t const totalGrid)
{
    Quantities *quantities = passCtx.quantities;
    Field &field = passCtx.field;
    // (fxMomWanted: allocated when the pass knows whether each one is a fixed-point sum -- fx.ensureReal)
    if ( haveVel && !fxMomWanted )
        quantities->velocity.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    if ( field.velocity_gradient && !fxMomWanted )
        quantities->velocity_gradient.assign(totalGrid, Pvector<Real,noGradComp>::zero());
    if ( field.velocity_dispersion && !fxMomWanted )
        quantities->velocity_dispersion.assign(totalGrid, Pvector<Real,noDispComp>::zero());
#ifdef SCALAR
    if ( field.scalar && !fxMomWanted )
        quantities->scalar.assign(totalGrid, Pvector<Real,noScalarComp>::zero());
    if ( field.scalar_gradient && !fxMomWanted )
        quantities->scalar_gradient.assign(totalGrid, Pvector<Real,noScalarGradComp>::zero());
#endif
    if ( fxMomWanted )
    {   // a grid left over from an earlier use of 'quantities' is released here, as the density is by the caller
        // (interpolateGrid_phaseSpace, ps_interpolation.cc)
        std::vector< Pvector<Real,noVelComp> >().swap( quantities->velocity );
        std::vector< Pvector<Real,noGradComp> >().swap( quantities->velocity_gradient );
        std::vector< Pvector<Real,noDispComp> >().swap( quantities->velocity_dispersion );
#ifdef SCALAR
        std::vector< Pvector<Real,noScalarComp> >().swap( quantities->scalar );
        std::vector< Pvector<Real,noScalarGradComp> >().swap( quantities->scalar_gradient );
#endif
    }
}

#if defined(PS_GPU) && NO_DIM==3
// Clears tryMetal (with a warning) for a pass the GPU cannot deposit: cusps, a multi-component scalar, 2^32 cells. The
// program's pick of the device never picks the GPU where the options already tell; what only the pass can tell (its cells,
// the sub-grid it deposits) takes the CPU with the warning and a counted fall-back, for a picked GPU as for an asked one.
static inline void psGpuEligible(PSPassCtx const &passCtx, size_t const subTotal, bool &tryMetal)
{
    User_options &userOptions = passCtx.userOptions;
#ifdef SCALAR
    Field &field = passCtx.field;
#endif
    size_t const noTotalCells = passCtx.noTotalCells;
    // The A3/A4 indicators of --ps-caustic-cusps fit each near-fold tet over its vertex ring (a CGAL neighbour gather), which
    // only the CPU deposit has: a GPU deposit would leave bits 7 and 8 unset. So such a run deposits on the CPU.
    if ( tryMetal && userOptions.psCausticCusps )
    {
        static std::atomic<bool> told{ false };
        if ( !told.exchange( true ) )
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "--ps-caustic-cusps needs the CPU deposit (its stencil gathers the triangulation's neighbours); "
                    << userOptions.gpuRequest() << " is not used for the deposit of this run.\n" << MESSAGE::EndWarning;
        }
        tryMetal = false;
    }
#ifdef SCALAR
    if ( tryMetal && (field.scalar || field.scalar_gradient) && noScalarComp != 1 )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "the GPU deposit supports ONE scalar component (this build has NO_SCALARS=" << noScalarComp << "); using the CPU deposit for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
        ++psGpuFallbacks;
    }
#endif
    // The kernels index cells with a 32-bit 'uint flat' (metal/ps_deposit.metal). Above 2^32 cells it wraps, and as the buffers
    // are LARGER than 2^32 the wrapped index stays in bounds: atomics would land in the wrong cells silently. Fall back to the
    // CPU instead (as the standard-DTFE guard in averaged_interpolation_1.cc). Checked on subTotal, the PARTITION's sub-box
    // (a finer --partition can bring a big grid back under it); the cap is gpuCellCap (gpu_host.h: 2^32; the test hook
    // DTFE_TEST_GPU_CELL_CAP lowers it).
    if ( tryMetal and noTotalCells > size_t(UINT32_MAX) )
    {   // the deferred tets are mapped back to their cells through 32-bit finite-cell ordinals
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "the GPU deposit does not support 2^32 or more Delaunay cells per triangulation (this one has "
                << noTotalCells << "); using the CPU deposit for this pass. A finer --partition splits them.\n"
                << MESSAGE::EndWarning;
        tryMetal = false;
        ++psGpuFallbacks;
    }
    if ( tryMetal and subTotal >= gpuCellCap() )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "the GPU deposit does not support 2^32 or more cells per partition sub-grid (this one has "
                << subTotal << "); using the CPU deposit for this pass. "
                << "A finer --partition would bring the sub-grid back under the cap.\n" << MESSAGE::EndWarning;
        tryMetal = false;
        ++psGpuFallbacks;
    }
}
#endif

// DTFE_DEBUG_DEGREES: checks the lazily computed vertex degrees of a GPU --grid-window pass against the full pass.
static inline void psDebugDegreesCheck(PSPassCtx const &passCtx, bool const degreesLazy, bool const metalDeposited)
{
    DT &dt = passCtx.dt;
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    if ( degreesLazy && metalDeposited && Env::flag( "DTFE_DEBUG_DEGREES" ) )
    {   // self-check of the lazy degrees: every vertex the extraction computed must carry the full pass's count
        std::unordered_map<void const*, int32_t> full;
        size_t degDropped = 0;
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry dgeo;
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degDropped, dgeo, true) ) continue;
            for (int v = 0; v <= NO_DIM; ++v) full[ &*cell->vertex(v) ] += 1;
        }
        size_t checked = 0, bad = 0;
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
        {
            int32_t const lazy = vit->info().psDegree();
            if ( lazy < 0 ) continue;
            ++checked;
            auto const it = full.find( &*vit );
            if ( lazy != ( it == full.end() ? 0 : it->second ) ) ++bad;
        }
        std::fprintf( stderr, "DTFE_DEBUG_DEGREES: %zu lazy vertex degrees checked against the full pass, %zu differ\n", checked, bad );
        if ( bad ) throwError( "DTFE_DEBUG_DEGREES: the lazy vertex degrees differ from the full pass." );
    }
}

// Whether the bounds of the GPU extraction (fx.gpuBounds) stand in for the bound walk's after a GPU deposit: only where the two
// are the same words. The extraction rounds the tet mass to float and the walk to Real, so a single build only; and under the
// NaN test hook (DTFE_TEST_NAN_SHARE) the extraction's bound leaves the hooked tets' mass out while the walk keeps it.
static inline bool psFxBoundsFromGpu(bool const metalDeposited, PSFxState const &fx)
{
    return metalDeposited && fx.gpuBoundsValid && std::is_same<Real, float>::value && fixedShareTestNanEvery() == 0;
}

// DTFE_DEBUG_FX_BOUNDS: checks the bounds taken over from the GPU extraction (psFxBoundsFromGpu) against the bound
// walk's, word for word (the moment grids' bounds and the 3 face bounds), and stops the run on a difference. Only
// where they are used: a pass with face sums (--face-gradient) after a GPU deposit.
static inline void psDebugFxBoundsCheck(PSPassCtx const &passCtx, PSFxState const &fx, bool const metalDeposited,
                                        bool const psVertexMass)
{
    if ( !fx.face || !psFxBoundsFromGpu( metalDeposited, fx ) || !Env::flag( "DTFE_DEBUG_FX_BOUNDS" ) ) return;
    PSFixedBounds const walk = psFxBoundWalk( passCtx, fx, psVertexMass );
    PSFixedBounds const &gpu = fx.gpuBounds;
    int bad = 0;
    for (int g = 0; g < PS_FX_CPU_N; ++g) bad += std::memcmp( &walk.B[g], &gpu.B[g], sizeof(double) ) != 0;
    bad += std::memcmp( &walk.faceS, &gpu.faceS, sizeof(double) ) != 0;
    bad += std::memcmp( &walk.faceN, &gpu.faceN, sizeof(double) ) != 0;
    bad += std::memcmp( &walk.faceP, &gpu.faceP, sizeof(double) ) != 0;
    std::fprintf( stderr, "DTFE_DEBUG_FX_BOUNDS: %d bounds of the GPU extraction checked against the bound walk, %d differ\n",
                  PS_FX_CPU_N + 3, bad );
    if ( bad ) throwError( "DTFE_DEBUG_FX_BOUNDS: the GPU extraction's bounds differ from the bound walk's." );
}

// The three units alone (S, N, P), the rule psFaceUnits applies: false when a bound gives none.
static inline bool psFaceScales(PSFixedBounds const &fb, float (&sc)[3])
{
    float sS = 0.f, sN = 0.f, sP = 0.f;
    bool const units = psMomentFixedScale( fb.faceS, sS ) && psMomentFixedScale( fb.faceN, sN )
                       && psMomentFixedScale( fb.faceP, sP ) && sS > 0.f && sN > 0.f && sP > 0.f;
    if ( !units ) return false;
    sc[0] = sS;
    sc[1] = sN;
    sc[2] = sP;
    return true;
}
// The face sums' units (--face-gradient) from a pass's bounds, and their zeroed words (int64: the walk's or the
// dynamic team's threads add in any order) for every cell of the pass. False, with nothing changed, when a bound gives
// no unit.
static inline bool psFaceUnits(PSFxState &fx, PSFixedBounds const &fb, size_t const totalGrid)
{
    float sc[3];
    if ( !psFaceScales( fb, sc ) ) return false;
    float const sS = sc[0], sN = sc[1], sP = sc[2];
    fx.faceS.scale = sS;
    fx.faceS.w.assign( totalGrid * 2 * NO_DIM * noVelComp, 0 );
    fx.faceN.scale = sN;
    fx.faceN.w.assign( totalGrid * 2 * NO_DIM, 0 );
    fx.faceP.scale = sP;
    fx.faceP.w.assign( totalGrid * 2 * NO_DIM, 0 );
    return true;
}

// The CPU deposit's fixed-point units: the mass accumulator and the moment grids' units.
static inline void psCpuFixedUnits(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, bool const metalDeposited,
                                   PSFxState &fx, bool const psVertexMass)
{
    PSFxWords (&fxg)[PS_FX_CPU_N] = fx.fxg;
    auto fxBoundWalk = [&]() { return psFxBoundWalk( passCtx, fx, psVertexMass ); };
    DT &dt = passCtx.dt;
    User_options &userOptions = passCtx.userOptions;
    Quantities *quantities = passCtx.quantities;
    Field &field = passCtx.field;
    bool const windowMode = passCtx.windowMode;
    size_t const totalGrid = gridsCtx.totalGrid;
    PSFxWords &massFx = gridsCtx.massFx;
    float &massFxScale = gridsCtx.massFxScale;
    bool &massFxOn = gridsCtx.massFxOn;
    bool const (&fxSel)[PS_FX_CPU_N] = gridsCtx.fxSel;
    bool const massFxWanted = gridsCtx.massFxWanted, fxMomWanted = gridsCtx.fxMomWanted;
    PSFixedBounds walkBounds;
    bool walked = false;
    // the pass's bounds, once per pass whoever asks first (mass grid under --grid-window, moment grids, face sums): the bound
    // walk, or after a GPU deposit the sums its extraction made over the same tets (fx.gpuBounds; only the face sums ask then)
    bool const boundsFromGpu = psFxBoundsFromGpu( metalDeposited, fx );
    auto bounds = [&]() -> PSFixedBounds const &
    {
        if ( boundsFromGpu ) return fx.gpuBounds;
        if ( !walked ) { walkBounds = fxBoundWalk(); walked = true; }
        return walkBounds;
    };

    // The CPU deposits the whole pass (no GPU, or declined or gave up): its mass accumulator is allocated here, the density (double
    // build) or the fixed-point grid from zero (massFx). Its unit must hold the pass's mass before the loop sees a tet, so it
    // comes from an upper bound found in one walk over the vertices: under --ps-vertex-mass the vertices' total weight, else
    // rho_bar times the vertices' Lagrangian bounding box; the larger of the two, deterministic. --grid-window: the window's own
    // total, B[DW] of the bound walk, which a GPU window deposit takes too (PSFixedUnits.massTotal): both deposit the same integers.
    if ( !metalDeposited && field.density && !massFxWanted )
        quantities->density.assign( totalGrid, Real(0.) );   // the float (double build: double) accumulator
    else if ( !metalDeposited && field.density )
    {
        double bound = 0.;
        if ( windowMode )
        {
            bound = bounds().B[PS_FX_DW];
        }
        else
        {
            double wSum = 0., lagLo[NO_DIM], lagHi[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) { lagLo[d] = std::numeric_limits<double>::max(); lagHi[d] = -lagLo[d]; }
            for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            {
#ifdef TEST_PADDING
                if ( vit->info().isDummy() ) continue;   // the filters drop every tet that touches one
#endif
                wSum += std::fabs( double(vit->info().weight()) );
                for (int d = 0; d < NO_DIM; ++d)
                {
                    double const p = double( vit->point()[d] );
                    lagLo[d] = std::min( lagLo[d], p );
                    lagHi[d] = std::max( lagHi[d], p );
                }
            }
            double lagVolume = 1.;
            for (int d = 0; d < NO_DIM; ++d) lagVolume *= std::max( 0., lagHi[d] - lagLo[d] );
            bound = std::max( wSum, std::fabs( double(userOptions.averageDensity) ) * lagVolume );
        }
        float scale = 0.f;
        if ( psMassFixedScale( bound, scale ) )
        {
            massFx.w.assign( totalGrid, 0 );
            massFxScale = scale;
            massFxOn = true;
        }
        else
            quantities->density.assign( totalGrid, Real(0.) );   // (no vertices / no mass: the float grid)
    }

    // The CPU deposits the whole pass and the moment grids are to be fixed-point sums (fxg): their units need bounds before the
    // loop deposits. No cheap vertex walk bounds the gradients or volumes, so they come from the bound walk (fxBoundWalk; under
    // --grid-window the window's tets only). DW is not accumulated when the mass grid is this pass's own words.
    if ( !metalDeposited && fxMomWanted )
    {
        bool any = false;
        for (int g = 0; g < PS_FX_CPU_N; ++g) any = any || ( fxSel[g] && !( g == PS_FX_DW && massFxOn ) );
        if ( any )
        {
            PSFixedBounds const &fb = bounds();
            for (int g = 0; g < PS_FX_CPU_N; ++g)
            {
                if ( !fxSel[g] || ( g == PS_FX_DW && massFxOn ) ) continue;
                float const scale = fx.unit( g, fb.B[g] );
                if ( scale > 0.f ) fx.start( g, scale );
                else fx.ensureReal( g );
            }
        }
        else
            for (int g = 0; g < PS_FX_CPU_N; ++g)
                if ( !( g == PS_FX_DW && massFxOn ) ) fx.ensureReal( g );   // (nothing selected: a no-op)
    }
    // --face-gradient: the face sums' units, from the same bounds; their int64 words for every cell of the pass. No unit (a
    // bound that is not finite): the face gradient is off for the pass. Under a GPU deposit the words are psFaceWalk's and the
    // units come from the bounds the extraction summed (same sums as the walk's); where those do not stand in (double build,
    // NaN test hook) the bound walk runs once more. (fx.faceWalked: the walk beside the GPU deposit already filled the words.)
    if ( fx.face && !fx.faceWalked )
    {
        bool const units = psFaceUnits( fx, bounds(), totalGrid );
        if ( !units )
        {
            PSFixedBounds const &fb = bounds();
            fx.face = false;
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "PS-DTFE: the face sums' bounds (" << fb.faceS << ", " << fb.faceN << ") give no fixed-point unit; "
                    << "'--face-gradient' is off for this pass (its gradient stays the stream average).\n" << MESSAGE::EndWarning;
        }
    }
    if ( !metalDeposited )
    {   // (--verbose 3) the CPU deposit's units, as the GPU's are logged after its deposit
        float sc[PS_FX_N];
        for (int g = 0; g < PS_FX_N; ++g) sc[g] = fxg[g].w.empty() ? 0.f : fxg[g].scale;
        fx.logUnits( "CPU", massFxOn ? massFxScale : 0.f, sc );
    }
}
