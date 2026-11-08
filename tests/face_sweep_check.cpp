// tests/face_sweep_check.sh compiles and runs this: the face sweep's geometry (src/CGAL_triangulation/face_sweep.h)
// against exact references, with no binary and no data.
//   F1  section(): the plane sections of a tetrahedron, integrated along the plane's axis, give the tetrahedron's
//       VOLUME and FIRST MOMENTS. A(c) is quadratic and c-weighted first moments cubic between two vertex levels, so
//       2-point Gauss-Legendre per interval is exact: relative error <= 1e-12, 2000 random tetrahedra x 3 axes.
//   F2  sweep(): the pieces of a section over a 2D grid add up to the section's area and moments (<= 1e-10: a
//       section near a vertex is tiny against its pieces' coordinates, so its relative rounding is larger; measured 3e-11).
//   F3  Gauss on a tessellation: a box of Kuhn-split cubes (6 tetrahedra each) carries ONE linear field
//       v(q) = a + B q. For every grid cell inside the box, the face integrals (each face piece added to the cell
//       below and subtracted from the cell above) equal B times the cell volume (<= 1e-11 of |B| V). Grid planes
//       are placed both off and ON the lattice's vertex levels (the half-open rule: faces lying in a grid plane
//       counted once) and the grid is rotated against nothing (axis-aligned cubes) and shifted.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "face_sweep.h"

static int failures = 0;
static void check(bool ok, char const *what, double err)
{
    std::printf("   %s  %s (worst %.3g)\n", ok ? "PASS" : "FAIL", what, err);
    if ( !ok ) ++failures;
}

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
    check(worstF2 <= 1e-10, "F2 the pieces of a section over a 2D grid add up to the section", worstF2);

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
    if ( failures ) { std::printf("RESULT: %d FAILED\n", failures); return 1; }
    std::printf("RESULT: PASS\n");
    return 0;
}
