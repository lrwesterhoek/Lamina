// Part of src/CGAL_triangulation/ps_interpolation.cc, inside its #ifdef PHASE_SPACE: included once, in order, by
// that file and compiled only as part of its object, never alone.
// The CPU deposit's fixed-point sums: the mass add, the moment sinks and words (PSFxSink, PSFxWords), the atomic
// count / flag / mask adds, the words to Real (psFxToReal), and in 3D the face sums' deposit (psDepositFaces).

// One share into the fixed-point MASS grid (see 'massFx' in interpolateGrid_phaseSpace): a plain add
// where one thread owns the cell (the serial loop, a slab), a relaxed atomic add where the threads of the copy mode
// share the grid. Integer adds commute, so the grid is the same exact sum whatever the interleaving.
static inline void psAddMassFixed(int64_t *grid, size_t const flat, int64_t const q, bool const atomic)
{
    if ( q == 0 ) return;
    if ( atomic ) __atomic_fetch_add( grid + flat, q, __ATOMIC_RELAXED );
    else grid[flat] += q;
}
// One thread's handle on a grid's words for its adds (PSFxWords::sink): the words (null: the grid has none), the unit
// 2^K as a double, the share cap (0: the pass's 64-bit words; > 0: the run-wide words, src/fixed_units.h), the bound
// of the run-wide words' 64-bit add (min(cap, 2^63); 0: none) and this thread's carry list (the run-wide words only)
struct PSFxSink
{
    int64_t *w = nullptr;
    double scale = 0.;
    double cap = 0.;
    double fast = 0.;
    FxCarryList *list = nullptr;
};
// The run-wide words' rare cases, kept out of line so that the add below stays small at every deposit site: a
// crossing of the int64 range into the thread's list; and a share of 2^63 units or more, or one with no integer at
// all (a NaN, an infinity, at or beyond the cap), which takes the 128-bit statements (psFixedShareOk's count in
// 'slot', then fxAddOwned / fxAddShared)
__attribute__((noinline, cold)) static void psFxCross(FxCarryList *const list, size_t const at, int64_t const c)
{
    list->push_back( FxCarry( uint64_t( at ), c ) );
}
__attribute__((noinline)) static void psFxAddWide(int64_t *const w, FxCarryList *const list, double const cap,
                                                  size_t const at, double const product, int const slot,
                                                  bool const atomic)
{
    __int128 q;
    if ( !psFixedShareOk( product, cap, slot, q ) || q == 0 ) return;
    if ( atomic ) fxAddShared( w + at, q, uint64_t( at ), *list );
    else fxAddOwned( w[at], q, uint64_t( at ), *list );
}
// One share into a grid's words, the one add of the CPU deposit: the share times the unit, its integer by the guard
// (ps_deposit_params.h psFixedShareOk: none is counted in 'slot'), added plainly where this thread owns the word and
// atomically where the threads share it. The 64-bit words take psAddMassFixed's add; the run-wide words add below
// the share cap and record each crossing of the int64 range in the thread's list (fxAddOwned / fxAddShared), so the
// sum is exact at any magnitude and in any order. A share below 2^63 units (nearly every one) is added here in 64
// bits: llrint gives the integer nearbyint gives (one rounding mode, and it fits), the share has no part beyond the
// word, and the add and its crossing test are fxAddOwned's / fxAddShared's own -- the same word and the same list
// entries as the 128-bit statements, which take the rest (psFxAddWide). Always inlined: one multiply, one rounding
// and one add at each deposit site.
static inline __attribute__((always_inline))
void psFxAdd(PSFxSink const &s, size_t const at, double const share, int const slot, bool const atomic)
{
    double const product = share * s.scale;
    if ( s.cap == 0. )
    {
        int64_t q;
        psFixedShareOk( product, slot, q );
        psAddMassFixed( s.w, at, q, atomic );
        return;
    }
    if ( std::fabs( product ) < s.fast )   // (false for a NaN too)
    {
        int64_t const q = int64_t( std::llrint( product ) );
        if ( q == 0 ) return;
        if ( atomic )
        {
            int64_t const old = __atomic_fetch_add( s.w + at, q, __ATOMIC_RELAXED );
            int64_t const after = int64_t( uint64_t( old ) + uint64_t( q ) );
            if ( q > 0 ? after < old : after > old ) psFxCross( s.list, at, q > 0 ? 1 : -1 );
        }
        else
        {
            int64_t r;
            if ( __builtin_add_overflow( s.w[at], q, &r ) ) psFxCross( s.list, at, q > 0 ? 1 : -1 );
            s.w[at] = r;
        }
        return;
    }
    psFxAddWide( s.w, s.list, s.cap, at, product, slot, atomic );
}
// A grid's fixed-point words (the moment grids' fxg, the mass grid's massFx and the face sums in
// interpolateGrid_phaseSpace): the pass's own int64 words, in units of 1/scale (a power of two). EMPTY w: the grid has
// no words (its Real accumulator is in use).
struct PSFxWords
{
    std::vector<int64_t> w;
    float scale = 0.f;
    // forgets the words and the unit
    void release()
    {
        std::vector<int64_t>().swap( w );
        scale = 0.f;
    }
    // a thread's handle (null words when there are none), the same for every thread: the words and the unit; the
    // handle's cap, fast and list keep their own initial 0, 0 and null (the 64-bit words: psFxAdd's plain add)
    PSFxSink sink(size_t)
    {
        PSFxSink s;
        if ( w.empty() ) return s;
        s.w = w.data();
        s.scale = double( scale );
        return s;
    }
};
// The deposit's order-free COUNTS and FLAGS, by the same rule: the int stream count (+1), the byte flags
// subSampleMassFlag / flippedFlag (set to 1) and the caustic mask (OR). The copy mode's threads share them as they share
// the fixed-point words -- +1, set and OR commute, so every thread count gives the same arrays and none needs a copy.
static inline void psAddCount(int &c, bool const atomic)
{
    if ( atomic ) __atomic_fetch_add( &c, 1, __ATOMIC_RELAXED );
    else ++c;
}
static inline void psSetFlag(unsigned char &f, bool const atomic)
{
    if ( atomic ) __atomic_store_n( &f, (unsigned char)1, __ATOMIC_RELAXED );
    else f = 1;
}
template <class Mask> static inline void psOrMask(Mask &m, Mask const bits, bool const atomic)
{
    if ( !atomic ) m |= bits;
    else if ( ( __atomic_load_n( &m, __ATOMIC_RELAXED ) & bits ) != bits )   // (mostly set already: skip the RMW)
        __atomic_fetch_or( &m, bits, __ATOMIC_RELAXED );
}

// The moment grids' fixed-point words ('fxg' in interpolateGrid_phaseSpace) and their Real grids: a
// cell element is a Pvector (component c) or a plain Real, and the words lie in the same layout, cell by cell.
template <class Elem> inline Real &psFxComp(Elem &e, size_t const c) { return e[c]; }
inline Real &psFxComp(Real &e, size_t) { return e; }
// A pass's cell-by-cell loop (a conversion, a scaling, the stream statistics) in cell ranges over this thread and the
// free cores (thread_count.h poolRanges): every cell is computed on its own, so the grids are the same at any split.
template <class F>
inline void psCellRanges(size_t const nCell, F const &f)
{
    poolRanges( nCell, size_t(1) << 16, CorePool::total(), f );
}
// The words of a grid become its Real grid, each component rounded once (the mass grid's conversion), and are freed.
template <class Elem>
inline void psFxToReal(std::vector<int64_t> &w, float const scale, std::vector<Elem> &grid, size_t const nCell)
{
    size_t const comps = sizeof(Elem) / sizeof(Real);
    grid.assign( nCell, Elem() );
    psCellRanges( nCell, [&](size_t const lo, size_t const hi)   // (cell by cell: over the free cores)
    {
        for (size_t i = lo; i < hi; ++i)
            for (size_t c = 0; c < comps; ++c)
                psFxComp( grid[i], c ) = Real( psMassFixedMass( w[i*comps + c], scale ) );
    } );
    std::vector<int64_t>().swap( w );
}
// ... from a PSFxWords, whose words are then released
template <class Elem>
inline void psFxToReal(PSFxWords &F, std::vector<Elem> &grid, size_t const nCell)
{
    psFxToReal( F.w, F.scale, grid, nCell );
    F.release();
}

#if NO_DIM==3
/* --face-gradient: one kept tetrahedron's FACE sums. Every grid plane it owns a section of
   (face_sweep.h's half-open rule, decided on region-frame coordinates every tetrahedron sharing a vertex computes
   alike) is cut into the face squares it covers; a piece's integral of the stream's linear velocity is its centroid
   value times its area. The piece's integrals go to the UPPER face of the cell below the plane and the LOWER face of
   the cell above (Quantities::face_velocity / face_coverage / face_momentum's slots): S = the sum of the streams'
   velocities, S.w[cell*18 + (side*3 + a)*3 + j], N = their count, N.w[cell*6 + side*3 + a] (the covered area), and P
   = the sum of rho_s v_a, the momentum through the face (rhoTet: the tet's uniform density, rho/rho_bar), P.w like
   N.w. The frame, the raw grid indices and the sub-box map are the exact deposit's (its r3d sweep's bounds and local
   index); a cell outside the sub-box (or, in slab mode, outside [slabLo, slabHi) along local x) gets nothing here: its
   own partition or slab adds it. Fixed point (psFxAdd into the calling thread's sinks, the non-finite guard's slots
   FX_DROP_FACE_*): order-free. */
static void psDepositFaces( Real const (&eulerPos)[NO_DIM+1][NO_DIM], Real const *boxL, Real const *dxL,
                            size_t const *nGridL, bool const periodic, size_t const *subOrigin, size_t const *subDims,
                            long const slabLo, long const slabHi, Real const (&velGrad)[NO_DIM][noVelComp],
                            double const (&v0)[noVelComp], PSFxSink const &S, PSFxSink const &N, PSFxSink const &P,
                            double const rhoTet, bool const atomic )
{
    double p[4][3], X[4][3];
    for (int v = 0; v < 4; ++v)
        for (int d = 0; d < 3; ++d)
        {
            X[v][d] = double( eulerPos[v][d] ) - double( boxL[2*d] );
            p[v][d] = double( eulerPos[v][d] ) - double( eulerPos[0][d] );
        }
    // the local index of a raw grid index along axis d (wrapped on a periodic axis), or -1 outside the sub-box
    auto local = [&](int const d, long const raw) -> long
    {
        long const n = long( nGridL[d] );
        long const wg = periodic ? ( ( raw % n ) + n ) % n : raw;
        if ( !periodic && ( wg < 0 || wg >= n ) ) return -1;
        long loc = wg - long( subOrigin[d] );
        if ( loc < 0 ) loc += n;
        return ( loc < 0 || loc >= long( subDims[d] ) ) ? -1 : loc;
    };
    for (int a = 0; a < 3; ++a)
    {
        int u, w;
        faceSweep::planeAxes( a, u, w );
        double const h = double( dxL[a] );
        double mn = X[0][a], mx = X[0][a];
        for (int v = 1; v < 4; ++v) { mn = std::min( mn, X[v][a] ); mx = std::max( mx, X[v][a] ); }
        long pFirst = long( std::floor( mn / h ) );
        if ( double( pFirst ) * h < mn ) ++pFirst;
        long pLast = long( std::floor( mx / h ) );
        if ( double( pLast ) * h >= mx ) --pLast;
        if ( !periodic ) { pFirst = std::max( pFirst, 0L ); pLast = std::min( pLast, long( nGridL[a] ) ); }
        if ( pFirst > pLast ) continue;
        int lo2[2], hi2[2];
        for (int k = 0; k < 2; ++k)
        {
            int const ax = k == 0 ? u : w;
            double lo = X[0][ax], hi = X[0][ax];
            for (int v = 1; v < 4; ++v) { lo = std::min( lo, X[v][ax] ); hi = std::max( hi, X[v][ax] ); }
            lo2[k] = int( std::floor( lo / double( dxL[ax] ) ) );
            hi2[k] = int( std::floor( hi / double( dxL[ax] ) ) ) + 1;
            if ( !periodic ) { lo2[k] = std::max( lo2[k], 0 ); hi2[k] = std::min( hi2[k], int( nGridL[ax] ) ); }
        }
        for (long pl = pFirst; pl <= pLast; ++pl)
        {
            long const below = local( a, pl - 1 ), above = local( a, pl );
            if ( below < 0 && above < 0 ) continue;
            double dist[4];
            for (int v = 0; v < 4; ++v) dist[v] = X[v][a] - double( pl ) * h;
            faceSweep::Polygon sec;
            if ( !faceSweep::section( p, a, dist, sec ) ) continue;
            double const c = double( boxL[2*a] ) + double( pl ) * h - double( eulerPos[0][a] );   // the plane, vertex-0 frame
            faceSweep::sweep( sec, lo2, hi2,
                [&](int const k, int const raw, double &blo, double &bhi)
                {
                    int const ax = k == 0 ? u : w;
                    blo = double( boxL[2*ax] ) + raw * double( dxL[ax] ) - double( eulerPos[0][ax] );
                    bhi = blo + double( dxL[ax] );
                },
                [&](int const k, int const raw) -> long { return local( k == 0 ? u : w, raw ); },
                [&](int const *, long const *loc, double const m[3])
                {
                    double q[3];
                    q[a] = c; q[u] = m[1] / m[0]; q[w] = m[2] / m[0];
                    double F[noVelComp];
                    for (size_t j = 0; j < noVelComp; ++j)
                    {
                        double vj = v0[j];
                        for (int i = 0; i < 3; ++i) vj += double( velGrad[i][j] ) * q[i];
                        F[j] = vj * m[0];
                    }
                    long cl[3];
                    cl[u] = loc[0]; cl[w] = loc[1];
                    for (int c2 = 0; c2 < 2; ++c2)   // the cell below the plane (its UPPER face), the cell above (its LOWER)
                    {
                        cl[a] = c2 == 0 ? below : above;
                        if ( cl[a] < 0 || cl[0] < slabLo || cl[0] >= slabHi ) continue;
                        size_t const flat = ( size_t( cl[0] ) * subDims[1] + size_t( cl[1] ) ) * subDims[2] + size_t( cl[2] );
                        size_t const slot = size_t( ( c2 == 0 ? 1 : 0 ) * NO_DIM + a );   // side 1 = upper, 0 = lower
                        for (size_t j = 0; j < noVelComp; ++j)
                            psFxAdd( S, flat * ( 2 * NO_DIM * noVelComp ) + slot * noVelComp + j, F[j],
                                     FX_DROP_FACE_S, atomic );
                        psFxAdd( N, flat * ( 2 * NO_DIM ) + slot, m[0], FX_DROP_FACE_N, atomic );
                        // the mass flux: the stream's uniform density times its velocity's NORMAL component
                        psFxAdd( P, flat * ( 2 * NO_DIM ) + slot, rhoTet * F[a], FX_DROP_FACE_P, atomic );
                    }
                } );
        }
    }
}
#endif
