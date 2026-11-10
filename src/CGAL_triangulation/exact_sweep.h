/* The CPU exact deposits' tetrahedron-cell clipping, done once per slab and column instead of once
   per cell: '--exact' in both binaries (ps_interpolation.cc and averaged_interpolation_1.cc).

   Both used to clip the whole tetrahedron against the six planes of EVERY cell of its bounding box
   (r3d_clip(&piece, planes, 6), planes in the order x-lo, x-hi, y-lo, y-hi, z-lo, z-hi). r3d_clip
   applies its planes one at a time, each to the polyhedron the previous ones left, so that clip is
   the tetrahedron clipped by the cell's two x planes (its SLAB), that polyhedron clipped by the two y
   planes (its COLUMN), and that one clipped by the two z planes. sweep() keeps the slab and the column
   and reuses them for every cell they hold: each cell's piece comes from the same r3d_clip calls on the
   same vertices as before, so its moments are bit-identical. The work per cell drops from six planes
   on the tetrahedron to two on the column, and the cells, columns and slabs outside their parent
   piece's extent are skipped without clipping (Powell & Abel 2015 voxelize by splitting along the grid
   planes, which is cheaper still but interpolates each vertex through a different chain of planes:
   the same pieces to rounding, not to the bit).

   The skip is r3d_clip's own decision. A plane n.x + d >= 0 with n = +-e_axis gives a vertex the signed
   distance x - lo (or hi - x) rounded once, so its sign is exact; r3d_clip leaves the polyhedron alone
   when no distance is negative and empties it when, besides, none is positive. So the lo plane empties
   it exactly when min x < lo and max x <= lo, and -- the lo plane then clipping nothing -- the hi plane
   exactly when min x >= hi and max x > hi. (A polyhedron flat in a cell plane, every distance 0, is
   kept, as r3d keeps it.) The cell bounds grow with the index, so the first cell past the extent ends
   the axis. */

#ifndef EXACT_SWEEP_HEADER
#define EXACT_SWEEP_HEADER

#include <cstddef>
#include <cstring>
#include "../../third_party/r3d/r3d.h"

namespace exactSweep {

// The live vertices only: an r3d_poly holds R3D_MAX_VERTS of them (4.6 KB at 128), and copying it whole for
// every cell cost more than clipping the cell.
inline void copyPoly( r3d_poly &dst, r3d_poly const &src )
{
    dst.nverts = src.nverts;
    if ( src.nverts > 0 ) std::memcpy( dst.verts, src.verts, size_t(src.nverts) * sizeof(r3d_vertex) );
}

// r3d_clip by the planes lo <= x <= hi of an axis would empty a polyhedron of extent [mn, mx] along it
// (see the top): beyond the cell's lo side, and beyond its hi side
inline bool beyondLo( double const mn, double const mx, double const lo ) { return mn < lo && mx <= lo; }
inline bool beyondHi( double const mn, double const mx, double const hi ) { return mn >= hi && mx > hi; }

inline void extent( r3d_poly const &p, int const axis, double &lo, double &hi )
{
    lo = hi = p.verts[0].pos.xyz[axis];
    for (int v = 1; v < p.nverts; ++v)
    {
        double const x = p.verts[v].pos.xyz[axis];
        if ( x < lo ) lo = x;
        if ( x > hi ) hi = x;
    }
}

// dst = src clipped to lo <= x_axis <= hi: the per-cell clip's two planes of this axis, in its order.
// False when nothing is left.
inline bool clipAxis( r3d_poly &dst, r3d_poly const &src, int const axis, double const lo, double const hi )
{
    r3d_plane planes[2];
    for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 3; ++k) planes[q].n.xyz[k] = 0.;
    planes[0].n.xyz[axis] =  1.; planes[0].d = -lo;
    planes[1].n.xyz[axis] = -1.; planes[1].d =  hi;
    copyPoly( dst, src );
    r3d_clip( &dst, planes, 2 );
    return dst.nverts > 0;
}

// r3d_reduce at polyorder 1 or 2 (moments [1, x, y, z] or [1, x, y, z, xx, xy, xz, yy, yz, zz]),
// unrolled: half the CPU deposit's time went into r3d_reduce's general-order recursion, which per
// triangle loops over its trinomial pyramid in small run-time-indexed arrays. This is that recursion
// written out for the two orders, with every floating-point operation r3d does, in its order and
// grouping -- the same face walk, a += b*c where r3d has one (accumulators starting from 0), the same
// constants and divisors -- so the moments are bit-identical where the compiler fuses a*b + c the same
// way in both. Clang fuses within an expression only (-ffp-contract=on, its default for r3d.c and
// here), so the same expressions fuse alike; GCC fuses across statements by default in r3d.c (and not
// at all in this code: the Makefile's -ffp-contract=off), so GCC builds keep calling r3d_reduce.
template <int ORDER>
inline void reduce( r3d_poly &poly, r3d_real *moments )
{
    static_assert( ORDER == 1 || ORDER == 2, "orders 1 and 2 only" );
#if !defined(__clang__)
    r3d_reduce( &poly, moments, ORDER );
#else
    int const NM = ORDER == 1 ? 4 : 10;
    for (int m = 0; m < NM; ++m) moments[m] = 0.0;
    int const nv = poly.nverts;
    if ( nv <= 0 ) return;
    r3d_vertex const *vb = poly.verts;
    unsigned char emarks[R3D_MAX_VERTS][3];
    std::memset( emarks, 0, size_t(nv) * sizeof emarks[0] );
    r3d_real const sixth = 0.16666666666666666666666666666666666666666666666666666667;   // r3d's ONE_SIXTH
    for (int vstart = 0; vstart < nv; ++vstart)
        for (int pstart = 0; pstart < 3; ++pstart)
        {
            if ( emarks[vstart][pstart] ) continue;
            // the face walk of r3d_reduce: a triangle fan from the face's first vertex
            int pnext = pstart, vcur = vstart, np;
            emarks[vcur][pnext] = 1;
            int vnext = vb[vcur].pnbrs[pnext];
            r3d_rvec3 const v0 = vb[vcur].pos;
            for (np = 0; np < 3; ++np)
                if ( vb[vnext].pnbrs[np] == vcur ) break;
            vcur = vnext;
            pnext = (np + 1) % 3;
            emarks[vcur][pnext] = 1;
            vnext = vb[vcur].pnbrs[pnext];
            while ( vnext != vstart )
            {
                r3d_rvec3 const v2 = vb[vcur].pos;
                r3d_rvec3 const v1 = vb[vnext].pos;
                r3d_real const sixv = (-v2.x * v1.y * v0.z + v1.x * v2.y * v0.z + v2.x * v0.y * v1.z -
                                       v0.x * v2.y * v1.z - v1.x * v0.y * v2.z + v0.x * v1.y * v2.z);
                moments[0] += sixth * sixv;
                // order 1: C (v2), D (v1), S (v0) of the pyramid's (1,0,0), (0,1,0), (0,0,1) entries, from 1
                r3d_real const one = 1.0;
                r3d_real cx = 0., dx = 0., sx = 0.;
                cx += v2.x * one; dx += v1.x * one; sx += v0.x * one;
                dx += cx; sx += dx;
                moments[1] += sixv * sx;
                r3d_real cy = 0., dy = 0., sy = 0.;
                cy += v2.y * one; dy += v1.y * one; sy += v0.y * one;
                dy += cy; sy += dy;
                moments[2] += sixv * sy;
                r3d_real cz = 0., dz = 0., sz = 0.;
                cz += v2.z * one; dz += v1.z * one; sz += v0.z * one;
                dz += cz; sz += dz;
                moments[3] += sixv * sz;
                if ( ORDER == 2 )
                {
                    // order 2, r3d's (i, j, k) loop order: (2,0,0) (1,1,0) (1,0,1) (0,2,0) (0,1,1) (0,0,2),
                    // each entry from the order-1 ones as r3d's if (i > 0) / (j > 0) / (k > 0) adds them
                    auto entry = [&](int const i, int const j, int const k, int const m)
                    {
                        r3d_real c = 0., d = 0., s = 0.;
                        if ( i > 0 )
                        {
                            r3d_real const pc = j > 0 ? cy : (k > 0 ? cz : cx);   // C[i-1][j]: (0,1) (0,0)k (1,0)
                            r3d_real const pd = j > 0 ? dy : (k > 0 ? dz : dx);
                            r3d_real const ps = j > 0 ? sy : (k > 0 ? sz : sx);
                            c += v2.x * pc; d += v1.x * pd; s += v0.x * ps;
                        }
                        if ( j > 0 )
                        {
                            r3d_real const pc = i > 0 ? cx : (k > 0 ? cz : cy);   // C[i][j-1]
                            r3d_real const pd = i > 0 ? dx : (k > 0 ? dz : dy);
                            r3d_real const ps = i > 0 ? sx : (k > 0 ? sz : sy);
                            c += v2.y * pc; d += v1.y * pd; s += v0.y * ps;
                        }
                        if ( k > 0 )
                        {
                            r3d_real const pc = i > 0 ? cx : (j > 0 ? cy : cz);   // C[i][j]
                            r3d_real const pd = i > 0 ? dx : (j > 0 ? dy : dz);
                            r3d_real const ps = i > 0 ? sx : (j > 0 ? sy : sz);
                            c += v2.z * pc; d += v1.z * pd; s += v0.z * ps;
                        }
                        d += c; s += d;
                        moments[m] += sixv * s;
                    };
                    entry( 2, 0, 0, 4 ); entry( 1, 1, 0, 5 ); entry( 1, 0, 1, 6 );
                    entry( 0, 2, 0, 7 ); entry( 0, 1, 1, 8 ); entry( 0, 0, 2, 9 );
                }
                for (np = 0; np < 3; ++np)
                    if ( vb[vnext].pnbrs[np] == vcur ) break;
                vcur = vnext;
                pnext = (np + 1) % 3;
                emarks[vcur][pnext] = 1;
                vnext = vb[vcur].pnbrs[pnext];
            }
        }
    // r3d's divisors: the leading multinomial coefficient times (n+1)(n+2)(n+3)
    for (int m = 1; m <= 3; ++m) moments[m] /= 24.;
    if ( ORDER == 2 )
    {
        moments[4] /= 60.; moments[5] /= 120.; moments[6] /= 120.;
        moments[7] /= 60.; moments[8] /= 120.; moments[9] /= 60.;
    }
#endif
}

// Every cell of the window [cLo, cHi) whose piece of 'tet' is not empty, in gi-gj-gk order (the old
// triple loop's order, so the callers' sums add in the same order).
//   bounds(axis, idx, lo, hi): the cell planes along an axis in the tetrahedron's frame (the caller's
//                              per-cell expressions: they must not depend on the other two indices)
//   local(axis, idx) -> long:  the index's position in the caller's grid, or < 0 to skip it (another
//                              thread's plane, outside the sub-box); also asked before any clipping
//   visit(idx[3], loc[3], piece): the cell's piece, never empty
template <class Bounds, class Local, class Visit>
inline void sweep( r3d_poly const &tet, int const cLo[3], int const cHi[3], Bounds &&bounds, Local &&local, Visit &&visit )
{
    if ( tet.nverts <= 0 ) return;
    // three polyhedra of 4.6 KB each: per thread, not on the deposit threads' stacks
    static thread_local r3d_poly slab, column, piece;
    int idx[3];
    long loc[3];
    double ext[3][2];
    extent( tet, 0, ext[0][0], ext[0][1] );
    for (idx[0] = cLo[0]; idx[0] < cHi[0]; ++idx[0])
    {
        double lo, hi;
        bounds( 0, idx[0], lo, hi );
        if ( beyondLo( ext[0][0], ext[0][1], lo ) ) break;
        if ( beyondHi( ext[0][0], ext[0][1], hi ) ) continue;
        if ( (loc[0] = local( 0, idx[0] )) < 0 ) continue;
        if ( !clipAxis( slab, tet, 0, lo, hi ) ) continue;
        extent( slab, 1, ext[1][0], ext[1][1] );
        for (idx[1] = cLo[1]; idx[1] < cHi[1]; ++idx[1])
        {
            bounds( 1, idx[1], lo, hi );
            if ( beyondLo( ext[1][0], ext[1][1], lo ) ) break;
            if ( beyondHi( ext[1][0], ext[1][1], hi ) ) continue;
            if ( (loc[1] = local( 1, idx[1] )) < 0 ) continue;
            if ( !clipAxis( column, slab, 1, lo, hi ) ) continue;
            extent( column, 2, ext[2][0], ext[2][1] );
            for (idx[2] = cLo[2]; idx[2] < cHi[2]; ++idx[2])
            {
                bounds( 2, idx[2], lo, hi );
                if ( beyondLo( ext[2][0], ext[2][1], lo ) ) break;
                if ( beyondHi( ext[2][0], ext[2][1], hi ) ) continue;
                if ( (loc[2] = local( 2, idx[2] )) < 0 ) continue;
                if ( !clipAxis( piece, column, 2, lo, hi ) ) continue;
                visit( idx, loc, piece );
            }
        }
    }
}

}   // namespace exactSweep

#endif
