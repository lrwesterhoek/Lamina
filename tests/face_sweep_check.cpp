// tests/face_sweep_check.sh compiles and runs this: the face sweep's geometry (src/CGAL_triangulation/face_sweep.h)
// against exact references, with no binary and no data.
//   F1  section(): the plane sections of a tetrahedron, integrated along the plane's axis, give the tetrahedron's
//       VOLUME and FIRST MOMENTS. A(c) is quadratic and c-weighted first moments cubic between two vertex levels, so
//       2-point Gauss-Legendre per interval is exact: relative error <= 1e-12, 2000 random tetrahedra x 3 axes.
//   F2  sweep(): the pieces of a section over a 2D grid add up to the section's area and moments (<= 1e-9: a
//       section near a vertex is tiny against its pieces' coordinates, so its relative rounding is larger; measured
//       3e-11 with fused multiply-adds (arm64) and 1.4e-10 without them (CI's x86 runner)).
//   F3  Gauss on a tessellation: a box of Kuhn-split cubes (6 tetrahedra each) carries ONE linear field
//       v(q) = a + B q. For every grid cell inside the box, the face integrals (each face piece added to the cell
//       below and subtracted from the cell above) equal B times the cell volume (<= 1e-11 of |B| V). Grid planes
//       are placed both off and ON the lattice's vertex levels (the half-open rule: faces lying in a grid plane
//       counted once) and the grid is rotated against nothing (axis-aligned cubes) and shifted.
// The canonical float sweep (src/CGAL_triangulation/face_sweep_f32.h: PS-DTFE's face sums under DTFE_FACE_F32=1
// and DTFE_GPU_FACES=1, the arithmetic the GPU face kernel shares with the CPU). Its tolerances are PROVISIONAL,
// about ten times the worst value of a float32 emulation of the header (python, outside the repository): TO MEASURE
// from this program's own PASS lines, then set to about ten times the value printed there.
//   F4  against the double sweep, as psDepositFaces forms the face words (S, N and P per side cell and slot), on
//       2100 tetrahedra of seven kinds (random, tiny (h/1000) and small (h/20) ones at a grid plane, big ones over
//       50 cells, vertices on grid planes, slivers, lattice cells), periodic with a sub-box and not: <= 1e-4 of the
//       piece's scale (the smaller of the cell and the tetrahedron, squared, times its largest velocity; emulated
//       8e-6, the big ones; the others 3e-7). A sweep with no early exit, wider index ranges and the sub-box test
//       after the clips (a kernel's loop) gives the same pieces bit for bit.
//   F5  F3's Gauss check with the canonical sweep and float inputs: <= 6e-5 of |B| V (emulated 6e-6)
//   F6  exact properties: a strip's vertices lie within its two bounds and neighbouring strips share their bound bit
//       for bit, and the pieces of a section add up to it (<= 5e-5 of its bounding box; emulated 3.5e-6); a
//       tetrahedron face lying in a grid plane is owned by the tetrahedron above it and not by the one below, its
//       section that face; the four corners of a 2-2 split in the order (p,r), (p,s), (q,s), (q,r) give the
//       atan2-ordered double section's area, and the 24 vertex orders of a tetrahedron own the same planes with the
//       same areas (<= 1e-6 of the tetrahedron's extent squared, and 5e-5 relative for a section above 1e-2 of it;
//       emulated 6e-8 / 2.5e-6 and 1.2e-7 / 4.8e-6); a vertex in a plane is a corner bit for bit, and alone in the
//       plane above the rest no section; the local index wraps as psDepositFaces' does, and a tetrahedron across
//       the periodic seam deposits its wrapped pieces as the double sweep does.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <utility>
#include <vector>
#include "face_sweep.h"
#include "face_sweep_f32.h"

static int failures = 0;
static void check(bool ok, char const *what, double err)
{
    std::printf("   %s  %s (worst %.3g)\n", ok ? "PASS" : "FAIL", what, err);
    if ( !ok ) ++failures;
}

// --- the canonical float sweep (face_sweep_f32.h) against the double one: face words keyed by (side cell, slot) ---
using Acc = std::map<std::pair<long, int>, std::array<double, 5>>;   // [S0, S1, S2, N, P]

// a piece's shares into the words of the cell below the plane (side 1, its upper face) and the cell above (side 0)
struct AccSink
{
    FsGrid const &g;
    Acc &acc;
    void piece(int const a, int const below, int const above, int const lu, int const lw, float const s0, float const s1,
               float const s2, float const n, float const p)
    {
        int u, w;
        fsPlaneAxes( a, u, w );
        int cl[3];
        cl[u] = lu;
        cl[w] = lw;
        for (int c2 = 0; c2 < 2; ++c2)
        {
            cl[a] = c2 == 0 ? below : above;
            if ( cl[a] < 0 ) continue;
            long const flat = ( long( cl[0] ) * g.subDims[1] + cl[1] ) * g.subDims[2] + cl[2];
            std::pair<long, int> const key( flat, ( c2 == 0 ? 1 : 0 ) * 3 + a );
            std::array<double, 5> &v = acc[key];
            v[0] += s0; v[1] += s1; v[2] += s2; v[3] += n; v[4] += p;
        }
    }
};

// every piece in the order it comes, its cell and its five shares (the bit-for-bit comparisons)
using Pieces = std::vector<std::array<float, 10>>;
struct RecordSink
{
    Pieces &out;
    void piece(int const a, int const below, int const above, int const lu, int const lw, float const s0, float const s1,
               float const s2, float const n, float const p)
    {
        out.push_back( { float( a ), float( below ), float( above ), float( lu ), float( lw ), s0, s1, s2, n, p } );
    }
};

// the double sweep of psDepositFaces (ps_interpolation.cc, its default path) on the same float inputs
static void refDeposit(float const E[4][3], FsGrid const &g, float const vG[3][3], float const v0[3], float const rho,
                       Acc &acc)
{
    double p[4][3], X[4][3];
    for (int v = 0; v < 4; ++v)
        for (int d = 0; d < 3; ++d)
        {
            X[v][d] = double( E[v][d] ) - double( g.lo[d] );
            p[v][d] = double( E[v][d] ) - double( E[0][d] );
        }
    bool const periodic = g.periodic != 0;
    auto local = [&](int const d, long const raw) -> long
    {
        long const n = g.n[d];
        long const wg = periodic ? ( ( raw % n ) + n ) % n : raw;
        if ( !periodic && ( wg < 0 || wg >= n ) ) return -1;
        long loc = wg - g.subOrigin[d];
        if ( loc < 0 ) loc += n;
        return ( loc < 0 || loc >= g.subDims[d] ) ? -1 : loc;
    };
    for (int a = 0; a < 3; ++a)
    {
        int u, w;
        faceSweep::planeAxes( a, u, w );
        double const h = double( g.h[a] );
        double mn = X[0][a], mx = X[0][a];
        for (int v = 1; v < 4; ++v) { mn = std::min( mn, X[v][a] ); mx = std::max( mx, X[v][a] ); }
        long pFirst = long( std::floor( mn / h ) );
        if ( double( pFirst ) * h < mn ) ++pFirst;
        long pLast = long( std::floor( mx / h ) );
        if ( double( pLast ) * h >= mx ) --pLast;
        if ( !periodic ) { pFirst = std::max( pFirst, 0L ); pLast = std::min( pLast, long( g.n[a] ) ); }
        if ( pFirst > pLast ) continue;
        int lo2[2], hi2[2];
        for (int k = 0; k < 2; ++k)
        {
            int const ax = k == 0 ? u : w;
            double lo = X[0][ax], hi = X[0][ax];
            for (int v = 1; v < 4; ++v) { lo = std::min( lo, X[v][ax] ); hi = std::max( hi, X[v][ax] ); }
            lo2[k] = int( std::floor( lo / double( g.h[ax] ) ) );
            hi2[k] = int( std::floor( hi / double( g.h[ax] ) ) ) + 1;
            if ( !periodic ) { lo2[k] = std::max( lo2[k], 0 ); hi2[k] = std::min( hi2[k], g.n[ax] ); }
        }
        for (long pl = pFirst; pl <= pLast; ++pl)
        {
            long const below = local( a, pl - 1 ), above = local( a, pl );
            if ( below < 0 && above < 0 ) continue;
            double dist[4];
            for (int v = 0; v < 4; ++v) dist[v] = X[v][a] - double( pl ) * h;
            faceSweep::Polygon sec;
            if ( !faceSweep::section( p, a, dist, sec ) ) continue;
            double const c = double( g.lo[a] ) + double( pl ) * h - double( E[0][a] );
            faceSweep::sweep( sec, lo2, hi2,
                [&](int const k, int const raw, double &blo, double &bhi)
                {
                    int const ax = k == 0 ? u : w;
                    blo = double( g.lo[ax] ) + raw * double( g.h[ax] ) - double( E[0][ax] );
                    bhi = blo + double( g.h[ax] );
                },
                [&](int const k, int const raw) -> long { return local( k == 0 ? u : w, raw ); },
                [&](int const *, long const *loc, double const m[3])
                {
                    double q[3];
                    q[a] = c; q[u] = m[1] / m[0]; q[w] = m[2] / m[0];
                    double F[3];
                    for (int j = 0; j < 3; ++j)
                    {
                        double vj = double( v0[j] );
                        for (int i = 0; i < 3; ++i) vj += double( vG[i][j] ) * q[i];
                        F[j] = vj * m[0];
                    }
                    long cl[3];
                    cl[u] = loc[0]; cl[w] = loc[1];
                    for (int c2 = 0; c2 < 2; ++c2)
                    {
                        cl[a] = c2 == 0 ? below : above;
                        if ( cl[a] < 0 ) continue;
                        long const flat = ( cl[0] * g.subDims[1] + cl[1] ) * g.subDims[2] + cl[2];
                        std::pair<long, int> const key( flat, ( c2 == 0 ? 1 : 0 ) * 3 + a );
                        std::array<double, 5> &acv = acc[key];
                        for (int j = 0; j < 3; ++j) acv[size_t( j )] += F[j];
                        acv[3] += m[0];
                        acv[4] += double( rho ) * F[a];
                    }
                } );
        }
    }
}

// a kernel's loop over the same building blocks: no early exit, the index ranges two wider on each side, the section
// taken before the sub-box test of its plane and every clip before the sub-box test of its strip and cell
static void kernelLikeDeposit(FsTet const &t, FsGrid const &g, Pieces &out)
{
    for (int a = 0; a < 3; ++a)
    {
        int k0;
        int const nk = fsPlanes( t, g, a, k0 );
        for (int k = k0; k < k0 + nk; ++k)
        {
            FsPoly sec;
            float c;
            if ( !fsSection( t, g, a, k, sec, c ) ) continue;
            int const below = fsLocal( g, a, k - 1 ), above = fsLocal( g, a, k );
            if ( below < 0 && above < 0 ) continue;
            int u, w;
            fsPlaneAxes( a, u, w );
            int i0, i1, j0, j1;
            if ( !fsInPlaneRange( t, g, u, i0, i1 ) || !fsInPlaneRange( t, g, w, j0, j1 ) ) continue;
            i0 -= 2; i1 += 2; j0 -= 2; j1 += 2;
            if ( g.periodic == 0 )
            {
                i0 = std::max( i0, 0 ); i1 = std::min( i1, g.n[u] );
                j0 = std::max( j0, 0 ); j1 = std::min( j1, g.n[w] );
            }
            for (int i = i0; i < i1; ++i)
            {
                FsPoly strip;
                fsSlab( sec, 0, fsBound( i, g.h[u], t.R[0][u] ), fsBound( i + 1, g.h[u], t.R[0][u] ), strip );
                for (int j = j0; j < j1; ++j)
                {
                    FsPoly cell;
                    fsSlab( strip, 1, fsBound( j, g.h[w], t.R[0][w] ), fsBound( j + 1, g.h[w], t.R[0][w] ), cell );
                    float S[3], N, Pn;
                    if ( !fsPieceShares( t, a, u, w, c, cell, S, N, Pn ) ) continue;
                    int const lu = fsLocal( g, u, i ), lw = fsLocal( g, w, j );
                    if ( lu < 0 || lw < 0 ) continue;
                    out.push_back( { float( a ), float( below ), float( above ), float( lu ), float( lw ), S[0], S[1], S[2],
                                     N, Pn } );
                }
            }
        }
    }
}

// a grid of n cells per axis, spacing h, from 0, with its sub-box
static FsGrid fsGridOf(float const h, int const n, bool const periodic, int const *sub0, int const *subd)
{
    FsGrid g;
    for (int d = 0; d < 3; ++d)
    {
        g.lo[d] = 0.f;
        g.h[d] = h;
        g.ih[d] = 1.f / h;
        g.n[d] = n;
        g.subOrigin[d] = sub0 ? sub0[d] : 0;
        g.subDims[d] = subd ? subd[d] : n;
    }
    g.periodic = periodic ? 1 : 0;
    return g;
}

// the largest extent of a tetrahedron along an axis
static double extentOf(float const E[4][3])
{
    double L = 0.;
    for (int d = 0; d < 3; ++d)
    {
        double lo = E[0][d], hi = E[0][d];
        for (int v = 1; v < 4; ++v) { lo = std::min( lo, double( E[v][d] ) ); hi = std::max( hi, double( E[v][d] ) ); }
        L = std::max( L, hi - lo );
    }
    return L;
}

// F4's comparison of one tetrahedron's words: the worst |difference| of S, N and P, each over its scale
static void compareWords(Acc const &f, Acc const &r, double const s2, double const vmax, double const rho, double worst[3],
                         long &words)
{
    Acc all = f;
    for (auto const &e : r) (void)all[e.first];   // (the union of the keys; a missing word is zero)
    for (auto const &e : all)
    {
        auto const fi = f.find( e.first ), ri = r.find( e.first );
        std::array<double, 5> const z = { 0., 0., 0., 0., 0. };
        std::array<double, 5> const &a = fi == f.end() ? z : fi->second, &b = ri == r.end() ? z : ri->second;
        for (int j = 0; j < 3; ++j) worst[0] = std::max( worst[0], std::fabs( a[size_t( j )] - b[size_t( j )] ) / ( s2 * vmax ) );
        worst[1] = std::max( worst[1], std::fabs( a[3] - b[3] ) / s2 );
        worst[2] = std::max( worst[2], std::fabs( a[4] - b[4] ) / ( rho * s2 * vmax ) );
        ++words;
    }
}

static int const kKuhn[6][3] = { {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0} };

static double tetVolume(double const p[4][3], double centroid[3])
{
    double a[3], b[3], c[3];
    for (int k = 0; k < 3; ++k) { a[k] = p[1][k]-p[0][k]; b[k] = p[2][k]-p[0][k]; c[k] = p[3][k]-p[0][k]; }
    double const det = a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]);
    for (int k = 0; k < 3; ++k) centroid[k] = 0.25*(p[0][k]+p[1][k]+p[2][k]+p[3][k]);
    return std::fabs(det) / 6.;
}

int main()
{
    std::mt19937_64 rng(20261107);
    std::uniform_real_distribution<double> U(-1., 1.);

    // F1 + F2
    double worstF1 = 0., worstF2 = 0.;
    for (int t = 0; t < 2000; ++t)
    {
        double p[4][3];
        for (int v = 0; v < 4; ++v) for (int k = 0; k < 3; ++k) p[v][k] = U(rng);
        if ( t % 5 == 0 ) p[3][0] = p[1][0];           // vertices sharing a level (ties)
        if ( t % 7 == 0 ) { p[2][1] = p[0][1]; p[3][1] = p[0][1]; }
        double cen[3];
        double const V = tetVolume(p, cen);
        if ( V < 1e-6 ) continue;
        for (int axis = 0; axis < 3; ++axis)
        {
            int u, w;
            faceSweep::planeAxes(axis, u, w);
            double lv[4];
            for (int v = 0; v < 4; ++v) lv[v] = p[v][axis];
            std::sort(lv, lv + 4);
            double vol = 0., mu = 0., mw = 0., ma = 0.;
            double const g = 0.5 / std::sqrt(3.);
            for (int i = 0; i < 3; ++i)
            {
                double const h = lv[i+1] - lv[i];
                if ( h <= 0. ) continue;
                for (double s : { 0.5 - g, 0.5 + g })
                {
                    double const c = lv[i] + s*h;
                    faceSweep::Polygon sec;
                    if ( !faceSweep::section(p, axis, c, sec) ) continue;
                    double m[3];
                    faceSweep::moments1(sec, m);
                    vol += 0.5*h*m[0]; mu += 0.5*h*m[1]; mw += 0.5*h*m[2]; ma += 0.5*h*m[0]*c;
                    // F2: the section over a random 2D grid
                    double const dxg = 0.05 + 0.3*std::fabs(U(rng)), off = U(rng);
                    int lo[2] = { -100, -100 }, hi[2] = { 100, 100 };
                    double s0 = 0., s1 = 0., s2 = 0.;
                    faceSweep::sweep(sec, lo, hi,
                        [&](int, int idx, double &a, double &b) { a = off + idx*dxg; b = a + dxg; },
                        [&](int, int idx) -> long { return idx + 1000; },     // (a negative local index skips)
                        [&](int const *, long const *, double const mm[3]) { s0 += mm[0]; s1 += mm[1]; s2 += mm[2]; });
                    double const e2 = std::fabs(s0-m[0]) / m[0] + std::fabs(s1-m[1]) / (m[0] + std::fabs(m[1]))
                                    + std::fabs(s2-m[2]) / (m[0] + std::fabs(m[2]));
                    if ( e2 > worstF2 ) worstF2 = e2;
                }
            }
            double const e1 = std::fabs(vol-V)/V + std::fabs(mu-V*cen[u])/(V*(1.+std::fabs(cen[u])))
                            + std::fabs(mw-V*cen[w])/(V*(1.+std::fabs(cen[w])))
                            + std::fabs(ma-V*cen[axis])/(V*(1.+std::fabs(cen[axis])));
            if ( e1 > worstF1 ) worstF1 = e1;
        }
    }
    check(worstF1 <= 1e-12, "F1 sections integrate to the tetrahedron's volume and first moments", worstF1);
    check(worstF2 <= 1e-9, "F2 the pieces of a section over a 2D grid add up to the section", worstF2);

    // F3: Kuhn-split lattice of L^3 unit cubes, a global linear field, grid cells inside the box
    double worstF3 = 0.;
    int const L = 6;
    for (int trial = 0; trial < 3; ++trial)
    {
        double av[3], B[3][3];                         // v_j(q) = a_j + sum_i B[i][j] q_i
        for (int j = 0; j < 3; ++j) { av[j] = 300.*U(rng); for (int i = 0; i < 3; ++i) B[i][j] = 100.*U(rng); }
        // trial 0: grid planes on the lattice levels (dx = 1, offset 0: every tetrahedron face in a grid plane)
        // trial 1: dx = 0.5 on the levels; trial 2: an incommensurate grid, shifted
        double const dxg = trial == 0 ? 1. : (trial == 1 ? 0.5 : 0.37);
        double const off = trial == 2 ? 0.123 : 0.;
        int const N = int(std::floor((L - off) / dxg)) - 1;          // cells fully inside [off, off + N dxg]
        std::vector<double> acc(size_t(N)*N*N*9, 0.);
        static int const kuhn[6][3] = { {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0} };
        for (int ci = 0; ci < L; ++ci) for (int cj = 0; cj < L; ++cj) for (int ck = 0; ck < L; ++ck)
            for (int s = 0; s < 6; ++s)
            {
                double p[4][3];
                double q[3] = { double(ci), double(cj), double(ck) };
                for (int k = 0; k < 3; ++k) p[0][k] = q[k];
                for (int v = 1; v < 4; ++v)
                {
                    q[kuhn[s][v-1]] += 1.;
                    for (int k = 0; k < 3; ++k) p[v][k] = q[k];
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    int u, w;
                    faceSweep::planeAxes(axis, u, w);
                    for (int pl = 0; pl <= N; ++pl)
                    {
                        double const c = off + pl*dxg;
                        faceSweep::Polygon sec;
                        if ( !faceSweep::section(p, axis, c, sec) ) continue;
                        int lo[2] = { 0, 0 }, hi[2] = { N, N };
                        faceSweep::sweep(sec, lo, hi,
                            [&](int, int idx, double &a, double &b) { a = off + idx*dxg; b = a + dxg; },
                            [&](int, int idx) -> long { return idx; },
                            [&](int const *ix, long const *, double const m[3])
                            {
                                double pt[3];
                                pt[axis] = c; pt[u] = m[1]/m[0]; pt[w] = m[2]/m[0];
                                double F[3];
                                for (int j = 0; j < 3; ++j)
                                {
                                    F[j] = av[j];
                                    for (int i = 0; i < 3; ++i) F[j] += B[i][j]*pt[i];
                                    F[j] *= m[0];
                                }
                                int cellIdx[3];
                                cellIdx[u] = ix[0]; cellIdx[w] = ix[1];
                                for (int side = 0; side < 2; ++side)    // the cell below (+F), the cell above (-F)
                                {
                                    cellIdx[axis] = side == 0 ? pl - 1 : pl;
                                    if ( cellIdx[axis] < 0 || cellIdx[axis] >= N ) continue;
                                    size_t const flat = (size_t(cellIdx[0])*N + size_t(cellIdx[1]))*N + size_t(cellIdx[2]);
                                    for (int j = 0; j < 3; ++j)
                                        acc[flat*9 + size_t(j*3 + axis)] += side == 0 ? F[j] : -F[j];
                                }
                            });
                    }
                }
            }
        double const Vc = dxg*dxg*dxg;
        double bmax = 0.;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) bmax = std::max(bmax, std::fabs(B[i][j]));
        for (int c = 0; c < N*N*N; ++c)
            for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i)
            {
                double const e = std::fabs(acc[size_t(c)*9 + size_t(j*3 + i)] - B[i][j]*Vc) / (bmax*Vc);
                if ( e > worstF3 ) worstF3 = e;
            }
    }
    check(worstF3 <= 1e-11, "F3 face integrals of a linear field on a Kuhn tessellation give its gradient (grid on, at half and off the lattice levels)", worstF3);

    // ---- the canonical float sweep (face_sweep_f32.h) ----
    std::uniform_real_distribution<double> U01(0., 1.);
    auto uni = [&](double const a, double const b) { return a + (b - a) * U01(rng); };
    auto pick = [&](int const n) { return int(std::floor(U01(rng) * n)) % n; };   // 0 .. n-1
    float const zeroG[3][3] = { {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f} };
    float const zeroV[3] = { 0.f, 0.f, 0.f };

    // F4: against the double sweep; and a kernel's loop, bit for bit
    {
        char const *const kindName[7] = { "random", "tiny", "small", "big", "on-plane", "sliver", "lattice" };
        double worst[3] = { 0., 0., 0. }, worstAll = 0.;
        int worstKind = 0;
        long words = 0, pieces = 0, loopDiffer = 0;
        for (int it = 0; it < 2100; ++it)
        {
            int const kind = it % 7;
            // (on-plane and lattice kinds: h = m/64, so that k h is exact and their vertices lie IN the planes)
            float const h = ( kind == 4 || kind == 6 ) ? float(4 + pick(19)) / 64.f : float(uni(0.05, 0.35));
            double const hd = double(h);
            bool const periodic = U01(rng) < 0.4;
            int const n = 16;
            int sub0[3], subd[3];
            for (int d = 0; d < 3; ++d)
            {
                sub0[d] = pick(n / 2);
                subd[d] = periodic ? 4 + pick(n - 3) : n - sub0[d] - pick(4);
            }
            FsGrid const g = fsGridOf(h, n, periodic, sub0, subd);
            double const L = hd * ( kind == 0 ? 2. : kind == 1 ? 1e-3 : kind == 2 ? 0.05 : kind == 3 ? 25. : kind == 6 ? 1. : 2. );
            double c[3];
            for (int d = 0; d < 3; ++d)
            {
                c[d] = uni(-3. * hd, 3. * hd);
                if ( kind == 1 || kind == 2 ) c[d] = std::round(c[d] / hd) * hd + uni(-0.3, 0.3) * L;   // at a plane
            }
            double const shift = periodic ? 0. : 8. * hd;   // (non-periodic: inside the grid's span, mostly)
            float E[4][3];
            if ( kind == 6 )
            {   // a Kuhn cell of the lattice of the grid's own planes
                int const s = pick(6);
                double q[3];
                for (int d = 0; d < 3; ++d) q[d] = double(pick(5) - 2) * hd + shift;
                for (int d = 0; d < 3; ++d) E[0][d] = float(q[d]);
                for (int v = 1; v < 4; ++v)
                {
                    q[kKuhn[s][v-1]] += hd;
                    for (int d = 0; d < 3; ++d) E[v][d] = float(q[d]);
                }
            }
            else
            {
                for (int v = 0; v < 4; ++v)
                    for (int d = 0; d < 3; ++d) E[v][d] = float(c[d] + uni(-L, L) + shift);
                if ( kind == 4 )   // vertex coordinates moved onto grid planes
                    for (int v = 0; v < 4; ++v)
                        for (int d = 0; d < 3; ++d)
                            if ( U01(rng) < 0.4 ) E[v][d] = float(std::round(double(E[v][d]) / hd)) * h;
                if ( kind == 5 )   // vertex 3 within 1e-4 of the tetrahedron's size of the plane of the other three
                {
                    double wt[3], ws = 0.;
                    for (int i = 0; i < 3; ++i) { wt[i] = uni(0.1, 1.); ws += wt[i]; }
                    for (int d = 0; d < 3; ++d)
                        E[3][d] = float((wt[0] * E[0][d] + wt[1] * E[1][d] + wt[2] * E[2][d]) / ws + uni(-1e-4, 1e-4) * L);
                }
            }
            float vG[3][3], v0[3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) vG[i][j] = float(30. / L * uni(-1., 1.));
            for (int j = 0; j < 3; ++j) v0[j] = float(uni(-300., 300.));
            float const rho = float(uni(0.1, 10.));
            Acc accF, accD;
            FsTet t;
            fsTetSet(E, g, vG, v0, rho, t);
            AccSink sinkF{ g, accF };
            fsDepositTet(t, g, sinkF);
            refDeposit(E, g, vG, v0, rho, accD);
            double vmax = 0.;
            for (int v = 0; v < 4; ++v)
                for (int j = 0; j < 3; ++j)
                {
                    double vv = double(v0[j]);
                    for (int i = 0; i < 3; ++i) vv += double(vG[i][j]) * ( double(E[v][i]) - double(E[0][i]) );
                    vmax = std::max(vmax, std::fabs(vv));
                }
            double const s = std::min(hd, extentOf(E));
            double w[3] = { 0., 0., 0. };
            compareWords(accF, accD, s * s, vmax, double(rho), w, words);
            for (int q = 0; q < 3; ++q)
            {
                worst[q] = std::max(worst[q], w[q]);
                if ( w[q] > worstAll ) { worstAll = w[q]; worstKind = kind; }
            }
            // the kernel's loop
            Pieces rec, ker;
            RecordSink sinkR{ rec };
            fsDepositTet(t, g, sinkR);
            kernelLikeDeposit(t, g, ker);
            pieces += long(rec.size());
            if ( rec != ker ) ++loopDiffer;
        }
        std::printf("   (F4: %ld words of 2100 tetrahedra; worst S %.3g, N %.3g, P %.3g of the piece's scale, the worst "
                    "kind '%s'; %ld pieces)\n", words, worst[0], worst[1], worst[2], kindName[worstKind], pieces);
        check(words > 20000 && worstAll <= 1e-4, "F4 the canonical float sweep against the double sweep (provisional "
              "tolerance 1e-4 of the piece's scale, TO MEASURE)", worstAll);
        check(pieces > 20000 && loopDiffer == 0, "F4 a kernel's loop (no early exit, wider ranges, the sub-box test after "
              "the clips) gives the same pieces bit for bit (tetrahedra that differ)", double(loopDiffer));
    }

    // F5: Gauss on the Kuhn tessellation with the canonical sweep (F3 in float)
    {
        struct GaussSink
        {
            int N;
            std::vector<double> &acc;
            void piece(int const a, int const below, int const above, int const lu, int const lw, float const s0,
                       float const s1, float const s2, float, float)
            {
                int u, w;
                fsPlaneAxes(a, u, w);
                float const S[3] = { s0, s1, s2 };
                int cl[3];
                cl[u] = lu; cl[w] = lw;
                for (int side = 0; side < 2; ++side)   // the cell below the plane (+S: its upper face), the cell above (-S)
                {
                    cl[a] = side == 0 ? below : above;
                    if ( cl[a] < 0 ) continue;
                    size_t const flat = (size_t(cl[0])*size_t(N) + size_t(cl[1]))*size_t(N) + size_t(cl[2]);
                    for (int j = 0; j < 3; ++j) acc[flat*9 + size_t(j*3 + a)] += side == 0 ? double(S[j]) : -double(S[j]);
                }
            }
        };
        double worstF5 = 0.;
        int const L = 6;
        for (int trial = 0; trial < 3; ++trial)
        {
            float av[3], B[3][3];
            for (int j = 0; j < 3; ++j) { av[j] = float(300.*U(rng)); for (int i = 0; i < 3; ++i) B[i][j] = float(100.*U(rng)); }
            double const dxg = trial == 0 ? 1. : (trial == 1 ? 0.5 : 0.37);
            double const off = trial == 2 ? 0.123 : 0.;
            int const N = int(std::floor((L - off) / dxg)) - 1;
            FsGrid g = fsGridOf(float(dxg), N, false, nullptr, nullptr);
            for (int d = 0; d < 3; ++d) g.lo[d] = float(off);
            std::vector<double> acc(size_t(N)*N*N*9, 0.);
            GaussSink sink{ N, acc };
            for (int ci = 0; ci < L; ++ci) for (int cj = 0; cj < L; ++cj) for (int ck = 0; ck < L; ++ck)
                for (int s = 0; s < 6; ++s)
                {
                    float E[4][3];
                    float q[3] = { float(ci), float(cj), float(ck) };
                    for (int k = 0; k < 3; ++k) E[0][k] = q[k];
                    for (int v = 1; v < 4; ++v)
                    {
                        q[kKuhn[s][v-1]] += 1.f;
                        for (int k = 0; k < 3; ++k) E[v][k] = q[k];
                    }
                    float v0[3];
                    for (int j = 0; j < 3; ++j)
                    {
                        double vj = double(av[j]);
                        for (int i = 0; i < 3; ++i) vj += double(B[i][j]) * double(E[0][i]);
                        v0[j] = float(vj);
                    }
                    FsTet t;
                    fsTetSet(E, g, B, v0, 1.f, t);
                    fsDepositTet(t, g, sink);
                }
            double const Vc = double(g.h[0]) * double(g.h[0]) * double(g.h[0]);
            double bmax = 0.;
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) bmax = std::max(bmax, std::fabs(double(B[i][j])));
            for (int c = 0; c < N*N*N; ++c)
                for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i)
                {
                    double const e = std::fabs(acc[size_t(c)*9 + size_t(j*3 + i)] - double(B[i][j])*Vc) / (bmax*Vc);
                    if ( e > worstF5 ) worstF5 = e;
                }
        }
        check(worstF5 <= 6e-5, "F5 the canonical float sweep's face integrals of a linear field on the Kuhn tessellation "
              "give its gradient (provisional tolerance 6e-5 of |B| V, TO MEASURE)", worstF5);
    }

    // F6a: a strip's vertices within its bounds, neighbouring strips sharing their bound bit for bit, a cell's within
    // its own along w; the pieces of a section add up to it
    {
        long violations = 0, shared = 0;
        double worstSum = 0.;
        for (int it = 0; it < 1200; ++it)
        {
            float const h = float(uni(0.05, 0.35));
            FsGrid const g = fsGridOf(h, 64, true, nullptr, nullptr);
            double const scales[4] = { 0.02, 0.5, 2., 8. };
            double const Lt = double(h) * scales[it % 4];
            double c[3];
            for (int d = 0; d < 3; ++d) c[d] = uni(-2., 2.);
            float E[4][3];
            for (int v = 0; v < 4; ++v) for (int d = 0; d < 3; ++d) E[v][d] = float(c[d] + uni(-Lt, Lt));
            FsTet t;
            fsTetSet(E, g, zeroG, zeroV, 1.f, t);
            for (int a = 0; a < 3; ++a)
            {
                int u, w;
                fsPlaneAxes(a, u, w);
                int k0;
                int const nk = fsPlanes(t, g, a, k0);
                for (int k = k0; k < k0 + nk; ++k)
                {
                    FsPoly sec;
                    float cpl;
                    if ( !fsSection(t, g, a, k, sec, cpl) ) continue;
                    float A2s, m1, m2;
                    fsMoments6(sec, A2s, m1, m2);
                    double const As = std::fabs(double(A2s)) / 2.;
                    float sxMin = sec.x[0], sxMax = sec.x[0], syMin = sec.y[0], syMax = sec.y[0];
                    for (int m = 1; m < sec.n; ++m)
                    {
                        sxMin = std::min(sxMin, sec.x[m]); sxMax = std::max(sxMax, sec.x[m]);
                        syMin = std::min(syMin, sec.y[m]); syMax = std::max(syMax, sec.y[m]);
                    }
                    double const bbox = ( double(sxMax) - double(sxMin) ) * ( double(syMax) - double(syMin) );
                    int i0, i1, j0, j1;
                    if ( !fsInPlaneRange(t, g, u, i0, i1) || !fsInPlaneRange(t, g, w, j0, j1) ) { ++violations; continue; }
                    double tot = 0.;
                    bool havePrev = false;
                    float prevMax = 0.f;
                    for (int i = i0; i < i1; ++i)
                    {
                        float const blo = fsBound(i, g.h[u], t.R[0][u]), bhi = fsBound(i + 1, g.h[u], t.R[0][u]);
                        FsPoly strip;
                        fsSlab(sec, 0, blo, bhi, strip);
                        bool nonEmpty = false;
                        if ( strip.n >= 3 )
                        {
                            float sa, sb, sc;
                            fsMoments6(strip, sa, sb, sc);
                            nonEmpty = sa > 0.f || sa < 0.f;
                        }
                        if ( nonEmpty )
                        {
                            float mnx = strip.x[0], mxx = strip.x[0];
                            for (int m = 0; m < strip.n; ++m)
                            {
                                if ( !( blo <= strip.x[m] && strip.x[m] <= bhi ) ) ++violations;
                                mnx = std::min(mnx, strip.x[m]); mxx = std::max(mxx, strip.x[m]);
                            }
                            if ( havePrev ) { ++shared; if ( prevMax != mnx ) ++violations; }
                            if ( sxMax > bhi && mxx != bhi ) ++violations;
                            havePrev = true;
                            prevMax = mxx;
                        }
                        else havePrev = false;
                        for (int j = j0; j < j1; ++j)
                        {
                            float const clo = fsBound(j, g.h[w], t.R[0][w]), chi = fsBound(j + 1, g.h[w], t.R[0][w]);
                            FsPoly cell;
                            fsSlab(strip, 1, clo, chi, cell);
                            if ( cell.n < 3 ) continue;
                            for (int m = 0; m < cell.n; ++m)
                                if ( !( clo <= cell.y[m] && cell.y[m] <= chi ) ) ++violations;
                            float ca, cb, cc;
                            fsMoments6(cell, ca, cb, cc);
                            tot += std::fabs(double(ca)) / 2.;
                        }
                    }
                    if ( bbox > 0. ) worstSum = std::max(worstSum, std::fabs(tot - As) / bbox);
                }
            }
        }
        std::printf("   (F6: %ld pairs of neighbouring strips)\n", shared);
        check(violations == 0 && shared > 1000, "F6 a strip's vertices lie within its bounds and neighbouring strips "
              "share their bound bit for bit, a cell's along w (violations)", double(violations));
        check(worstSum <= 5e-5, "F6 the pieces of a section add up to it (of its bounding box; provisional tolerance 5e-5, "
              "TO MEASURE)", worstSum);
    }

    // F6b: the half-open rule on a tetrahedron face lying in a grid plane
    {
        long bad = 0, cases = 0;
        double worstB = 0.;
        for (int it = 0; it < 500; ++it)
        {
            int const m = 4 + pick(19);
            float const h = float(m) / 64.f;                       // (k h exact)
            FsGrid const g = fsGridOf(h, 64, true, nullptr, nullptr);
            int const k = pick(21) - 10;
            float const xp = float(k) * h;
            float face[3][3];
            for (int v = 0; v < 3; ++v) { face[v][0] = xp; face[v][1] = float(uni(-1., 1.)); face[v][2] = float(uni(-1., 1.)); }
            double const fa = 0.5 * std::fabs( ( double(face[1][1]) - face[0][1] ) * ( double(face[2][2]) - face[0][2] )
                                             - ( double(face[2][1]) - face[0][1] ) * ( double(face[1][2]) - face[0][2] ) );
            double ylo = face[0][1], yhi = face[0][1], zlo = face[0][2], zhi = face[0][2];
            for (int v = 1; v < 3; ++v)
            {
                ylo = std::min(ylo, double(face[v][1])); yhi = std::max(yhi, double(face[v][1]));
                zlo = std::min(zlo, double(face[v][2])); zhi = std::max(zhi, double(face[v][2]));
            }
            double const fbox = ( yhi - ylo ) * ( zhi - zlo );
            if ( !( fa > 1e-3 * fbox ) ) continue;                  // (a near-degenerate face: no case)
            for (int below = 0; below < 2; ++below)
                for (int apexFirst = 0; apexFirst < 2; ++apexFirst)
                {
                    float apex[3] = { below ? xp - float(uni(0.01, 1.)) : xp + float(uni(0.01, 1.)),
                                      float(uni(-1., 1.)), float(uni(-1., 1.)) };
                    float E[4][3];
                    for (int v = 0; v < 4; ++v)
                    {   // the apex first or last, the face's vertices in their order
                        bool const isApex = apexFirst ? v == 0 : v == 3;
                        int const fv = isApex ? 0 : ( apexFirst ? v - 1 : v );
                        for (int d = 0; d < 3; ++d) E[v][d] = isApex ? apex[d] : face[fv][d];
                    }
                    FsTet t;
                    fsTetSet(E, g, zeroG, zeroV, 1.f, t);
                    int k0;
                    int const nk = fsPlanes(t, g, 0, k0);
                    bool const owns = k >= k0 && k < k0 + nk;
                    FsPoly s;
                    float cpl;
                    bool const got = fsSection(t, g, 0, k, s, cpl);
                    if ( below ) { if ( owns || got ) ++bad; }
                    else if ( !owns || !got || s.n != 3 ) ++bad;
                    else
                    {
                        float A2, m1, m2;
                        fsMoments6(s, A2, m1, m2);
                        worstB = std::max(worstB, std::fabs(std::fabs(double(A2)) / 2. - fa) / fbox);
                    }
                    ++cases;
                }
        }
        check(bad == 0 && cases > 1500, "F6 a tetrahedron face in a grid plane: the tetrahedron above owns its section, the "
              "one below does not (cases wrong)", double(bad));
        check(worstB <= 1e-5, "F6 ... and that section is the face (of its bounding box)", worstB);
    }

    // F6c: the combinatorial corner order against the atan2 order of the double section; the 24 vertex orders
    {
        long quads = 0, badOrders = 0;
        double worstQ = 0., relQ = 0., worstP = 0., relP = 0.;
        for (int it = 0; it < 300; ++it)
        {
            float const h = float(uni(0.05, 0.35));
            FsGrid const g = fsGridOf(h, 64, true, nullptr, nullptr);
            float E0[4][3];
            for (int v = 0; v < 4; ++v) for (int d = 0; d < 3; ++d) E0[v][d] = float(uni(-1., 1.));
            double const Lt = extentOf(E0);
            std::map<std::pair<int, int>, std::vector<double>> areas;
            int perm[4] = { 0, 1, 2, 3 };
            bool first = true;
            do
            {
                float E[4][3];
                for (int v = 0; v < 4; ++v) for (int d = 0; d < 3; ++d) E[v][d] = E0[perm[v]][d];
                FsTet t;
                fsTetSet(E, g, zeroG, zeroV, 1.f, t);
                for (int a = 0; a < 3; ++a)
                {
                    int k0;
                    int const nk = fsPlanes(t, g, a, k0);
                    for (int k = k0; k < k0 + nk; ++k)
                    {
                        FsPoly s;
                        float cpl;
                        if ( !fsSection(t, g, a, k, s, cpl) ) continue;
                        float A2, m1, m2;
                        fsMoments6(s, A2, m1, m2);
                        double const area = std::fabs(double(A2)) / 2.;
                        areas[std::make_pair(a, k)].push_back(area);
                        if ( !first ) continue;
                        // the double section from the same frame and sides, its corners ordered by angle
                        double pp[4][3], dd[4];
                        for (int v = 0; v < 4; ++v)
                        {
                            for (int d = 0; d < 3; ++d) pp[v][d] = double(t.P[v][d]);
                            dd[v] = double(FS_FMA(-float(k), g.h[a], t.R[v][a]));
                        }
                        faceSweep::Polygon ds;
                        if ( !faceSweep::section(pp, a, dd, ds) ) { ++badOrders; continue; }
                        double md[3];
                        faceSweep::moments1(ds, md);
                        if ( s.n == 4 ) ++quads;
                        worstQ = std::max(worstQ, std::fabs(area - md[0]) / (Lt * Lt));
                        if ( md[0] > 1e-2 * Lt * Lt ) relQ = std::max(relQ, std::fabs(area - md[0]) / md[0]);
                    }
                }
                first = false;
            } while ( std::next_permutation(perm, perm + 4) );
            for (auto const &e : areas)
            {
                std::vector<double> const &v = e.second;
                if ( v.size() != 24 ) { ++badOrders; continue; }
                double const lo = *std::min_element(v.begin(), v.end()), hi = *std::max_element(v.begin(), v.end());
                worstP = std::max(worstP, (hi - lo) / (Lt * Lt));
                if ( hi > 1e-2 * Lt * Lt ) relP = std::max(relP, (hi - lo) / hi);
            }
        }
        std::printf("   (F6: %ld four-corner sections; relative %.3g and %.3g where above 1e-2 of the extent squared)\n",
                    quads, relQ, relP);
        check(badOrders == 0 && quads > 100 && worstQ <= 1e-6 && relQ <= 5e-5, "F6 the corners in the combinatorial order "
              "(four: (p,r), (p,s), (q,s), (q,r)) give the atan2-ordered double section's area (of the extent squared; "
              "provisional, TO MEASURE)", worstQ);
        check(badOrders == 0 && worstP <= 1e-6 && relP <= 5e-5, "F6 the 24 vertex orders of a tetrahedron own the same "
              "planes, with the same section areas (of the extent squared; provisional, TO MEASURE)", worstP);
    }

    // F6d: a vertex lying in a grid plane
    {
        long bad = 0, cases = 0;
        for (int it = 0; it < 400; ++it)
        {
            float const h = float(4 + pick(19)) / 64.f;
            FsGrid const g = fsGridOf(h, 64, true, nullptr, nullptr);
            int const a = it % 3, vin = (it / 3) % 4;
            bool const alone = ( it / 12 ) % 2 == 1;   // the others all above the plane: no section
            int const k = pick(21) - 10;
            float const xp = float(k) * h;
            float E[4][3];
            int others = 0;
            for (int v = 0; v < 4; ++v)
            {
                for (int d = 0; d < 3; ++d) E[v][d] = float(uni(-1., 1.));
                if ( v == vin ) { E[v][a] = xp; continue; }
                bool const up = alone || others > 0;   // one below and two above, or all three above
                E[v][a] = up ? xp + float(uni(0.01, 1.)) : xp - float(uni(0.01, 1.));
                ++others;
            }
            FsTet t;
            fsTetSet(E, g, zeroG, zeroV, 1.f, t);
            int u, w;
            fsPlaneAxes(a, u, w);
            FsPoly s;
            float cpl;
            bool const got = fsSection(t, g, a, k, s, cpl);
            if ( alone ) { if ( got ) ++bad; }
            else if ( !got || s.n != 3 || s.x[0] != t.P[vin][u] || s.y[0] != t.P[vin][w] ) ++bad;
            ++cases;
        }
        check(bad == 0 && cases == 400, "F6 a vertex in a grid plane: a corner bit for bit (one vertex below, two above); "
              "no section with the other three above (cases wrong)", double(bad));
    }

    // F6: the constant of step 7 (face_sweep_f32.h FS_K6) is the float nearest 1/6, checked at run time
    {
        float const six = float(U01(rng) < 2.) * 6.f;   // (6, from a value the compiler cannot fold)
        check(FS_K6 == 1.f / six, "F6 FS_K6 is the float nearest 1/6 (its difference from 1/6 in double)",
              std::fabs(double(FS_K6) - 1. / 6.));
    }

    // F6e: the local index and the periodic seam
    {
        long bad = 0;
        for (int n = 1; n <= 9; ++n)
            for (int sub0 = 0; sub0 < n; ++sub0)
                for (int subd = 1; subd <= n; ++subd)
                    for (int per = 0; per < 2; ++per)
                    {
                        FsGrid g = fsGridOf(1.f, n, per != 0, nullptr, nullptr);
                        g.subOrigin[0] = sub0;
                        g.subDims[0] = subd;
                        for (long raw = -3L * n; raw <= 3L * n; ++raw)
                        {   // psDepositFaces' 'local'
                            long const wg = per ? ( ( raw % n ) + n ) % n : raw;
                            long ref;
                            if ( !per && ( wg < 0 || wg >= n ) ) ref = -1;
                            else
                            {
                                long loc = wg - sub0;
                                if ( loc < 0 ) loc += n;
                                ref = ( loc < 0 || loc >= subd ) ? -1 : loc;
                            }
                            if ( long(fsLocal(g, 0, int(raw))) != ref ) ++bad;
                        }
                    }
        check(bad == 0, "F6 the local index of a raw grid index is psDepositFaces' (wrapped, the sub-box; mismatches)", double(bad));
        double worstSeam = 0.;
        long words = 0, wrapped = 0;
        for (int it = 0; it < 300; ++it)
        {
            float const h = float(uni(0.05, 0.35));
            int const n = 8;
            FsGrid const g = fsGridOf(h, n, true, nullptr, nullptr);
            float E[4][3];
            for (int v = 0; v < 4; ++v)
                for (int d = 0; d < 3; ++d) E[v][d] = float(uni(-0.8, 0.8) * double(h));   // across the planes 0 (raw -1 | 0)
            float vG[3][3], v0[3];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) vG[i][j] = float(uni(-100., 100.));
            for (int j = 0; j < 3; ++j) v0[j] = float(uni(-300., 300.));
            float const rho = float(uni(0.1, 10.));
            Acc accF, accD;
            FsTet t;
            fsTetSet(E, g, vG, v0, rho, t);
            AccSink sinkF{ g, accF };
            fsDepositTet(t, g, sinkF);
            refDeposit(E, g, vG, v0, rho, accD);
            for (auto const &e : accF)
            {
                long const flat = e.first.first;
                if ( flat < 0 || flat >= long(n) * n * n ) ++bad;
                if ( flat / ( n * n ) == n - 1 || ( flat / n ) % n == n - 1 || flat % n == n - 1 ) ++wrapped;
            }
            double vmax = 0.;
            for (int v = 0; v < 4; ++v)
                for (int j = 0; j < 3; ++j)
                {
                    double vv = double(v0[j]);
                    for (int i = 0; i < 3; ++i) vv += double(vG[i][j]) * ( double(E[v][i]) - double(E[0][i]) );
                    vmax = std::max(vmax, std::fabs(vv));
                }
            double const s = std::min(double(h), extentOf(E));
            double w[3] = { 0., 0., 0. };
            compareWords(accF, accD, s * s, vmax, double(rho), w, words);
            worstSeam = std::max(worstSeam, std::max(w[0], std::max(w[1], w[2])));
        }
        check(bad == 0 && wrapped > 100 && worstSeam <= 1e-4, "F6 across the periodic seam: the wrapped pieces, as the "
              "double sweep deposits them (of the piece's scale; provisional, TO MEASURE)", worstSeam);
    }

    if ( failures ) { std::printf("RESULT: %d FAILED\n", failures); return 1; }
    std::printf("RESULT: PASS\n");
    return 0;
}
