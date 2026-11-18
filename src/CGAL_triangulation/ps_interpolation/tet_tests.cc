// Part of src/CGAL_triangulation/ps_interpolation.cc, inside its #ifdef PHASE_SPACE: included once, in order, by
// that file and compiled only as part of its object, never alone.
// The tetrahedron-box overlap test (TetBoxSAT), the CPU deposit's thread count (psDepositThreads) and the count
// of a straddling tetrahedron's samples beyond a non-periodic grid (psCountBeyondGrid).

namespace {
/* Does a tetrahedron overlap an axis-aligned box with POSITIVE volume? The separating-axis test
   over the 3 box normals, the 4 face normals and the 18 (edge x box axis) products, which is
   complete for two convex polyhedra. Contact of zero volume (a shared face, edge or vertex) counts
   as separated. Double throughout, in a frame relative to the tetrahedron's vertex 0; the tet's
   projections and the box's projected half-widths depend only on the tet (all boxes are one grid
   cell), so they are set once and each box costs one dot product per axis. Used by the '.hidden_streams'
   multi-stream flag (see flippedFlag). */
struct TetBoxSAT
{
    int n = 0;
    double a[25][3], lo[25], hi[25], rad[25];

    void add(double const v[4][3], double const half[3], double x, double y, double z)
    {
        if ( !(x*x + y*y + z*z > 0.) ) return;        // an edge parallel to a box axis: no new axis
        a[n][0] = x; a[n][1] = y; a[n][2] = z;
        double pmin = x*v[0][0] + y*v[0][1] + z*v[0][2], pmax = pmin;
        for (int k = 1; k < 4; ++k)
        {
            double const p = x*v[k][0] + y*v[k][1] + z*v[k][2];
            pmin = std::min(pmin, p);
            pmax = std::max(pmax, p);
        }
        lo[n] = pmin;
        hi[n] = pmax;
        rad[n] = half[0]*std::fabs(x) + half[1]*std::fabs(y) + half[2]*std::fabs(z);
        ++n;
    }

    TetBoxSAT(double const v[4][3], double const half[3])
    {
        add(v, half, 1., 0., 0.);
        add(v, half, 0., 1., 0.);
        add(v, half, 0., 0., 1.);
        static int const F[4][3] = { {1,2,3}, {0,2,3}, {0,1,3}, {0,1,2} };
        for (int f = 0; f < 4; ++f)
        {
            double e1[3], e2[3];
            for (int d = 0; d < 3; ++d)
            {
                e1[d] = v[F[f][1]][d] - v[F[f][0]][d];
                e2[d] = v[F[f][2]][d] - v[F[f][0]][d];
            }
            add(v, half, e1[1]*e2[2] - e1[2]*e2[1], e1[2]*e2[0] - e1[0]*e2[2], e1[0]*e2[1] - e1[1]*e2[0]);
        }
        static int const E[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
        for (int e = 0; e < 6; ++e)
        {
            double const dx = v[E[e][1]][0] - v[E[e][0]][0];
            double const dy = v[E[e][1]][1] - v[E[e][0]][1];
            double const dz = v[E[e][1]][2] - v[E[e][0]][2];
            add(v, half, 0., dz, -dy);          // edge x (1,0,0)
            add(v, half, -dz, 0., dx);          // edge x (0,1,0)
            add(v, half, dy, -dx, 0.);          // edge x (0,0,1)
        }
    }

    bool overlaps(double const c[3]) const     // c = the box centre, in the tet's frame
    {
        for (int i = 0; i < n; ++i)
        {
            double const bc = a[i][0]*c[0] + a[i][1]*c[1] + a[i][2]*c[2];
            if ( hi[i] <= bc - rad[i] || lo[i] >= bc + rad[i] ) return false;
        }
        return true;
    }
};
} // namespace


// Threads for one partition's CPU deposit: every core when this is the only triangulation being
// processed; inside the partitioned loop (an OpenMP team of concurrent partitions) the cores left
// per partition, so the two levels never oversubscribe (thread_count.h). DTFE_DEPOSIT_THREADS pins it.
static int psDepositThreads()
{
    return coresPerTeam( "DTFE_DEPOSIT_THREADS" );
}

// The samples of a straddling tetrahedron that lie BEYOND a non-periodic grid, counted for the sampled
// deposit's share m / (in + beyond). Kept OUT of the deposit loop's lambda, not inlined, and handed
// COPIES of the tetrahedron's state, on purpose: written inline, its nested lambdas would capture the
// loop's hoisted locals by reference; as a function taking the loop's invD / vertex positions by
// reference, their addresses would escape and every in-grid sample test -- every tetrahedron's hot path
// -- would reload them from memory. Either costs the deposit several times what this counting takes. The
// function keeps its own copy of the exact-test cache (rawPos / exactTet): at most one extra setup per
// tetrahedron.
struct PSBeyondGrid
{
    size_t nGrid[NO_DIM];
    int    nSub;
    double gridLen[NO_DIM];
    Real   boxLo[NO_DIM];
    Real   dx[NO_DIM];
};
struct PSBeyondTet
{
    int    iMinU[NO_DIM], iMaxU[NO_DIM];   // the unclamped window
    Real   eulerPos[NO_DIM+1][NO_DIM];
    double invD[NO_DIM][NO_DIM];
    double band1;
    Cell_handle cell;
};
struct PSBeyondCount { size_t n = 0; };

__attribute__((noinline))
static PSBeyondCount psCountBeyondGrid(PSBeyondGrid const &grid, PSBeyondTet const &T,
                                       psExactInside::Frame const &exactFrame)
{
#if NO_DIM==3
    int const *iMinU = T.iMinU, *iMaxU = T.iMaxU;
    Real const (&eulerPos)[NO_DIM+1][NO_DIM] = T.eulerPos;
    double const (&invD)[NO_DIM][NO_DIM] = T.invD;
    double const band1 = T.band1;
    Cell_handle const &cell = T.cell;
    Real rawPos[NO_DIM+1][NO_DIM];
    bool rawPosReady = false;
    psExactInside::TetTest exactTet;
    size_t nOutside = 0;
    // The samples beyond the grid are only COUNTED, so they are taken one sample COLUMN at a time, along
    // the tet's LONGEST window axis (fewest columns): along a column every barycentric is affine in the
    // position, so the samples that clear tier 1's band by a margin on all four faces form one run,
    // counted in bulk, and so do those that cannot be inside. Only the few in neither -- within the band
    // of a face -- take the per-sample exact test below, so the count is exactly the per-sample one,
    // whichever axis the columns run along. Only columns that HAVE samples beyond the grid and lie
    // inside the tet's projection are visited. Short columns take the per-sample test.
    int ax[NO_DIM] = { 0, 1, 2 };   // ax[2]: the column axis; ax[0] rows, ax[1] the inner loop
    {
        int best = 0;
        for (int d = 1; d < NO_DIM; ++d) if ( iMaxU[d] - iMinU[d] > iMaxU[best] - iMinU[best] ) best = d;
        int k = 0;
        for (int d = 0; d < NO_DIM; ++d) if ( d != best ) ax[k++] = d;
        ax[2] = best;
    }
    int const a0 = ax[0], a1 = ax[1], ac = ax[2];
    long const nS = long(grid.nSub);
    long const S0c = long(iMinU[ac]) * nS, S1c = long(iMaxU[ac]) * nS;
    long const gridC = long(grid.nGrid[ac]) * nS;
    double const e0[NO_DIM] = { double(eulerPos[0][0]), double(eulerPos[0][1]), double(eulerPos[0][2]) };
    auto floorDiv = [nS](long s) -> long { return s >= 0 ? s / nS : -((-s + nS - 1) / nS); };
    // sample s along axis d -> its position, lo + (s + 1/2) h: the samplePosAxis expression up to rounding,
    // with one multiply-add instead of a floor division and a division (the positions only need to be the
    // sample lattice; each sample's inside/outside decision below is exact at the position used)
    double h[NO_DIM], invH[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        h[d] = grid.gridLen[d] / (double(grid.nGrid[d]) * double(grid.nSub));
        invH[d] = 1. / h[d];
    }
    auto samplePos = [&](int d, long s) -> double { return double(grid.boxLo[d]) + (double(s) + 0.5) * h[d]; };
    // the per-sample exact test of the in-grid loop (tier 1, then the canonical tiers)
    auto exactInside = [&](double const pos[NO_DIM]) -> bool
    {
        double relD[NO_DIM], baryD[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d) relD[d] = pos[d] - e0[d];
        int const fine = psExactInside::fineClassify( invD, relD, band1, baryD );
        if ( fine != 0 ) return fine > 0;
        if ( not rawPosReady )
        {
            for (int v = 0; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    rawPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);
            rawPosReady = true;
        }
        double pc[NO_DIM];
        psExactInside::canonicalQuery( exactFrame, pos, pc );
        return psExactInside::canonicalInside( exactFrame, rawPos, exactTet, pc );
    };
    // tier 1's band plus a margin far above the rounding difference between the affine
    // evaluation below and fineClassify's (|bary| terms stay << 1e6 for any kept tet)
    double const tBand = band1 + 1.e-9;
    bool const bulkOK = band1 < 1.e300;   // no inverse: every sample takes the exact tiers
    // only the columns that HAVE samples beyond the grid: when the window stays inside the grid
    // along the column axis, a column inside it along a0 needs just its a1-parts beyond the grid
    bool const cStraddle = S0c < 0 || S1c > gridC;
    long const S10 = long(iMinU[a1]) * nS, S11 = long(iMaxU[a1]) * nS, grid1 = long(grid.nGrid[a1]) * nS;
    // ... and only the columns inside the tet's projection on the (a0, a1) plane: along a row that is
    // one interval, the extent of the six projected edges crossing it, padded by a sample each side
    double V[NO_DIM+1][2];
    for (int v = 0; v <= NO_DIM; ++v)
    {
        V[v][0] = double(eulerPos[v][a0]) - e0[a0];
        V[v][1] = double(eulerPos[v][a1]) - e0[a1];
    }
    double const rowPad = 1.e-7 * double(grid.dx[a0]);
    double const s1Scale = invH[a1];
    double const s1Shift = (e0[a1] - double(grid.boxLo[a1])) * s1Scale - 0.5;
    double const scScale = invH[ac];
    double const scShift = (e0[ac] - double(grid.boxLo[ac])) * scScale - 0.5;
    // the faces' slopes along the column axis are the same for every column: invert them once
    double Gc[4], invGc[4];
    for (int v = 0; v < NO_DIM; ++v) Gc[v] = invD[ac][v];
    Gc[3] = -(Gc[0] + Gc[1] + Gc[2]);
    for (int v = 0; v < 4; ++v) invGc[v] = Gc[v] != 0. ? 1. / Gc[v] : 0.;
    for (long s0 = long(iMinU[a0]) * nS; s0 < long(iMaxU[a0]) * nS; ++s0)
    {
        long const k0 = floorDiv(s0);
        bool const in0 = k0 >= 0 && k0 < long(grid.nGrid[a0]);
        double const rel0 = samplePos(a0, s0) - e0[a0];
        double lo1 = std::numeric_limits<double>::infinity(), hi1 = -lo1;
        for (int i = 0; i < NO_DIM; ++i)
            for (int j = i + 1; j <= NO_DIM; ++j)
            {
                double const xi = V[i][0], xj = V[j][0];
                if ( rel0 < std::min(xi, xj) - rowPad || rel0 > std::max(xi, xj) + rowPad ) continue;
                double y0 = V[i][1], y1 = V[j][1];
                if ( xj != xi )
                {
                    double const t = std::min(1., std::max(0., (rel0 - xi) / (xj - xi)));
                    y0 = y1 = V[i][1] + t * (V[j][1] - V[i][1]);
                }
                lo1 = std::min(lo1, std::min(y0, y1));
                hi1 = std::max(hi1, std::max(y0, y1));
            }
        if ( !(lo1 <= hi1) ) continue;   // this row misses the tet
        long const pLo = std::max(S10, long(std::floor(lo1 * s1Scale + s1Shift)) - 1);
        long const pHi = std::min(S11, long(std::ceil(hi1 * s1Scale + s1Shift)) + 2);
        if ( pLo >= pHi ) continue;
        long rLo[2] = { pLo, 0 }, rHi[2] = { pHi, 0 };
        int nR = 1;
        if ( in0 && !cStraddle )
        {
            nR = 0;
            if ( pLo < 0 )     { rLo[nR] = pLo; rHi[nR] = std::min(pHi, 0L); ++nR; }
            if ( pHi > grid1 ) { rLo[nR] = std::max(pLo, grid1); rHi[nR] = pHi; ++nR; }
        }
        for (int ir = 0; ir < nR; ++ir)
        for (long s1 = rLo[ir]; s1 < rHi[ir]; ++s1)
        {
            long const k1 = floorDiv(s1);
            bool const colIn = in0 && k1 >= 0 && k1 < long(grid.nGrid[a1]);
            // this column's samples beyond the grid: all of it, or its parts below / above the grid
            long outLo[2], outHi[2];
            int nOut = 0;
            if ( !colIn ) { outLo[0] = S0c; outHi[0] = S1c; nOut = 1; }
            else
            {
                if ( S0c < 0 )     { outLo[nOut] = S0c; outHi[nOut] = std::min(S1c, 0L); ++nOut; }
                if ( S1c > gridC ) { outLo[nOut] = std::max(S0c, gridC); outHi[nOut] = S1c; ++nOut; }
            }
            if ( nOut == 0 ) continue;
            double pos[NO_DIM];
            pos[a0] = samplePos(a0, s0);
            pos[a1] = samplePos(a1, s1);
            auto perSample = [&](long u, long v)
            {
                for (long sc = u; sc < v; ++sc)
                {
                    pos[ac] = samplePos(ac, sc);
                    if ( !exactInside(pos) ) continue;
                    ++nOutside;
                }
            };
            long outLen = 0;
            for (int i = 0; i < nOut; ++i) outLen += outHi[i] - outLo[i];
            if ( !bulkOK || outLen <= 4 )
            {
                for (int i = 0; i < nOut; ++i) perSample(outLo[i], outHi[i]);
                continue;
            }
            // bary_v = A_v + G_v r_c (v = 0..2) and l0 = A_3 + G_3 r_c, each ONE affine expression of
            // the column offset r_c (which rises with the sample index), so each is monotone along it
            double const r0 = pos[a0] - e0[a0], r1 = pos[a1] - e0[a1];
            double A[4];
            for (int v = 0; v < NO_DIM; ++v) A[v] = invD[a0][v] * r0 + invD[a1][v] * r1;
            A[3] = 1. - (A[0] + A[1] + A[2]);
            double const * const G = Gc;
            auto value = [&](int v, long sc) { return A[v] + G[v] * (samplePos(ac, sc) - e0[ac]); };
            // first index of [lo, hi) where a monotone false->true predicate holds (hi if none): two
            // probes at the analytic crossing 'guess' settle it (it is off by rounding only), bisection
            // covers whatever they leave
            auto transition = [](long lo, long hi, double guess, auto pred) -> long
            {
                long L = lo, H = hi;
                long g = (guess > double(lo)) ? ( guess < double(hi) ? long(std::ceil(guess)) : hi ) : lo;
                if ( g > L && g - 1 < H ) { if ( pred(g - 1) ) H = g - 1; else L = g; }
                if ( g >= L && g < H )    { if ( pred(g) ) H = g; else L = g + 1; }
                while ( L < H ) { long const mid = L + (H - L) / 2; if ( pred(mid) ) H = mid; else L = mid + 1; }
                return L;
            };
            // [a, b) = the samples of [S0c, S1c) with value > T on all four faces (one run)
            auto run = [&](double T, long &a, long &b)
            {
                a = S0c; b = S1c;
                for (int v = 0; v < 4 && a < b; ++v)
                {
                    double const guess = G[v] != 0. ? (T - A[v]) * invGc[v] * scScale + scShift : 0.;
                    if ( G[v] > 0. )        // rises along the column: the first sample above T
                        a = transition( a, b, guess, [&](long sc) { return value(v, sc) > T; } );
                    else if ( G[v] < 0. )   // falls: one past the last sample above T
                        b = transition( a, b, guess, [&](long sc) { return !(value(v, sc) > T); } );
                    else if ( !(A[v] > T) ) b = a;
                }
            };
            long aIn, bIn, aPos, bPos;
            run( tBand, aIn, bIn );     // tier 1 decides these inside
            run( -tBand, aPos, bPos );  // the rest lie beyond -band1 on some face: outside
            if ( aIn >= bIn ) aIn = bIn = aPos;   // nothing certain: all of [aPos, bPos) is tested
            auto clip = [](long a, long b, long u, long v, long &c, long &d) { c = std::max(a, u); d = std::min(b, v); return c < d; };
            for (int i = 0; i < nOut; ++i)
            {
                long c, d;
                if ( clip(aIn, bIn, outLo[i], outHi[i], c, d) )
                {
                    nOutside += size_t(d - c);
                }
                if ( clip(aPos, aIn, outLo[i], outHi[i], c, d) ) perSample(c, d);
                if ( clip(bIn, bPos, outLo[i], outHi[i], c, d) ) perSample(c, d);
            }
        }
    }
    PSBeyondCount out;
    out.n = nOutside;
    return out;
#else
    // 2D: every sample of the unclamped window beyond the grid, one exact test each (the 3D column
    // scheme above is an optimization of exactly this count)
    PSBeyondCount out;
    long const nS = long(grid.nSub);
    double h[NO_DIM];
    long S0[NO_DIM], S1[NO_DIM], G[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        h[d] = grid.gridLen[d] / (double(grid.nGrid[d]) * double(grid.nSub));
        S0[d] = long(T.iMinU[d]) * nS; S1[d] = long(T.iMaxU[d]) * nS; G[d] = long(grid.nGrid[d]) * nS;
    }
    double const e0[NO_DIM] = { double(T.eulerPos[0][0]), double(T.eulerPos[0][1]) };
    Real rawPos[NO_DIM+1][NO_DIM];
    bool rawPosReady = false;
    psExactInside::TetTest exactTet;
    for (long s0 = S0[0]; s0 < S1[0]; ++s0)
        for (long s1 = S0[1]; s1 < S1[1]; ++s1)
        {
            if ( s0 >= 0 && s0 < G[0] && s1 >= 0 && s1 < G[1] ) continue;   // the in-grid loop's sample
            double const pos[NO_DIM] = { double(grid.boxLo[0]) + (double(s0) + 0.5) * h[0],
                                         double(grid.boxLo[1]) + (double(s1) + 0.5) * h[1] };
            double relD[NO_DIM], baryD[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) relD[d] = pos[d] - e0[d];
            int const fine = psExactInside::fineClassify( T.invD, relD, T.band1, baryD );
            bool inside = fine > 0;
            if ( fine == 0 )
            {
                if ( not rawPosReady )
                {
                    for (int v = 0; v <= NO_DIM; ++v)
                        for (int d = 0; d < NO_DIM; ++d)
                            rawPos[v][d] = T.cell->vertex(v)->info().eulerianPosition(d);
                    rawPosReady = true;
                }
                double pc[NO_DIM];
                psExactInside::canonicalQuery( exactFrame, pos, pc );
                inside = psExactInside::canonicalInside( exactFrame, rawPos, exactTet, pc );
            }
            if ( not inside ) continue;
            ++out.n;
        }
    return out;
#endif
}
