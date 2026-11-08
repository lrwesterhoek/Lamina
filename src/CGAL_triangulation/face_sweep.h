/* The face sweep: exact integrals of a linear field over the grid's FACES, for the face method of the velocity
   gradient (docs/cleanup-plan.md P10; '--face-gradient').

   By Gauss, the average over a grid cell of the gradient of a field v is (1/V) times the integral of v n over the
   cell's six faces. For standard DTFE's interpolant -- linear in every tetrahedron and continuous across them --
   that is exactly the cell average of the per-tetrahedron gradient the deposit adds up today; for PS-DTFE, where
   the streams overlap and the mean velocity jumps at the caustics, it is the finite-volume gradient of the
   cell-averaged mean flow (P10.3). Both need the same geometry, here:

   - section(): the plane x_axis = c cuts a tetrahedron in a convex polygon (a triangle or a quadrilateral), given in
     the plane's own coordinates (u, w), the other two axes in increasing order (x: (y, z), y: (x, z), z: (x, y)).
     HALF-OPEN RULE: a tetrahedron owns its section when min_axis <= c < max_axis. The plane cuts the tessellation
     into the tetrahedra's sections, which then tile it; a tetrahedron face lying IN the plane belongs to the
     tetrahedron on its + side only (the one with min_axis = c), so no area is counted twice or lost.
   - sweep(): the section clipped to every face square of the plane's 2D grid it overlaps, strip by strip along u
     and cell by cell along w (the 2D counterpart of exact_sweep.h's slab/column/cell walk, with the same early
     exits), and the area and first moments of each piece (Green's theorem over its boundary, as ps_clip2d.h). For a
     linear field the integral over a piece is its value at the piece's centroid times its area.

   Everything is in double, in the caller's frame (the deposits' base-vertex-relative coordinates). */

#ifndef FACE_SWEEP_HEADER
#define FACE_SWEEP_HEADER

#include <cmath>
#include "ps_clip2d.h"

namespace faceSweep {

using Polygon = psClip2d::Polygon;

// the in-plane axes (u, w) of the plane normal to 'axis', in increasing order
inline void planeAxes( int const axis, int &u, int &w )
{
    u = ( axis == 0 ) ? 1 : 0;
    w = ( axis == 2 ) ? 1 : 2;
}

// Area and first moments [A, A*u_c, A*w_c] of a polygon, counter-clockwise or not (the sign is taken off, so the
// centroid m[1]/m[0], m[2]/m[0] holds either way). Green's theorem over the boundary, as ps_clip2d.h.
inline void moments1( Polygon const &p, double m[3] )
{
    m[0] = m[1] = m[2] = 0.;
    for (int i = 0; i < p.n; ++i)
    {
        int const j = ( i + 1 ) % p.n;
        double const c = p.x[i] * p.y[j] - p.x[j] * p.y[i];
        m[0] += c;
        m[1] += ( p.x[i] + p.x[j] ) * c;
        m[2] += ( p.y[i] + p.y[j] ) * c;
    }
    m[0] /= 2.; m[1] /= 6.; m[2] /= 6.;
    if ( m[0] < 0. ) { m[0] = -m[0]; m[1] = -m[1]; m[2] = -m[2]; }
}

/* The section of the tetrahedron with vertices p[4][3] by the plane x_axis = c, as a counter-clockwise convex polygon
   in the plane's (u, w). d[4]: the vertices' SIGNED DISTANCES to the plane, which decide the sides and the half-open
   rule (the top): the tetrahedron owns a section when min d <= 0 < max d. The caller computes them from coordinates
   every tetrahedron sharing a vertex computes alike (the deposits' absolute region-frame positions), so two
   tetrahedra never decide a shared face lying in the plane differently, whatever the rounding of their own frames;
   p only places the corners. False when the tetrahedron owns no section of positive area. A vertex with d = 0 is a
   corner; an edge whose ends lie strictly on opposite sides gives the corner where it crosses. At most four. */
inline bool section( double const p[4][3], int const axis, double const d[4], Polygon &out )
{
    out.n = 0;
    double mn = d[0], mx = d[0];
    for (int v = 1; v < 4; ++v)
    {
        if ( d[v] < mn ) mn = d[v];
        if ( d[v] > mx ) mx = d[v];
    }
    if ( !( mn <= 0. && 0. < mx ) ) return false;
    int u, w;
    planeAxes( axis, u, w );
    double cu[6], cw[6];
    int n = 0;
    for (int v = 0; v < 4; ++v)
        if ( d[v] == 0. ) { cu[n] = p[v][u]; cw[n] = p[v][w]; ++n; }
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
            if ( ( d[a] < 0. && d[b] > 0. ) || ( d[a] > 0. && d[b] < 0. ) )
            {
                double const t = d[a] / ( d[a] - d[b] );
                cu[n] = p[a][u] + t * ( p[b][u] - p[a][u] );
                cw[n] = p[a][w] + t * ( p[b][w] - p[a][w] );
                ++n;
            }
    if ( n < 3 ) return false;
    // the corners of a convex polygon, counter-clockwise: by angle around their mean (at most four)
    double mu = 0., mw = 0.;
    for (int i = 0; i < n; ++i) { mu += cu[i]; mw += cw[i]; }
    mu /= n; mw /= n;
    double ang[6];
    int order[6];
    for (int i = 0; i < n; ++i) { ang[i] = std::atan2( cw[i] - mw, cu[i] - mu ); order[i] = i; }
    for (int i = 1; i < n; ++i)              // insertion sort of at most four
        for (int j = i; j > 0 && ang[order[j]] < ang[order[j-1]]; --j)
        { int const t = order[j]; order[j] = order[j-1]; order[j-1] = t; }
    for (int i = 0; i < n; ++i) { out.x[i] = cu[order[i]]; out.y[i] = cw[order[i]]; }
    out.n = n;
    double m[3];
    moments1( out, m );
    if ( !( m[0] > 0. ) ) { out.n = 0; return false; }
    return true;
}

// The same with the distances taken in p's own frame (d = p[v][axis] - c): one tetrahedron, or a frame all share.
inline bool section( double const p[4][3], int const axis, double const c, Polygon &out )
{
    double d[4];
    for (int v = 0; v < 4; ++v) d[v] = p[v][axis] - c;
    return section( p, axis, d, out );
}

/* Every face square of the window [lo, hi) (indices along u, then w) whose piece of the section 'sec' has positive
   area, in iu-iw order:
     bounds(k, idx, lo, hi): the face square's planes along in-plane axis k (0: u, 1: w) in the caller's frame
     local(k, idx) -> long:  the index's position in the caller's grid, or < 0 to skip it (another thread's strip)
     visit(idx[2], loc[2], m[3]): the piece's area and first moments [A, A*u_c, A*w_c] (A > 0)
   A strip or cell beyond the section's extent ends its axis, as in exact_sweep.h. */
template <class Bounds, class Local, class Visit>
inline void sweep( Polygon const &sec, int const lo[2], int const hi[2], Bounds &&bounds, Local &&local, Visit &&visit )
{
    if ( sec.n < 3 ) return;
    double uMin = sec.x[0], uMax = sec.x[0];
    for (int i = 1; i < sec.n; ++i)
    {
        if ( sec.x[i] < uMin ) uMin = sec.x[i];
        if ( sec.x[i] > uMax ) uMax = sec.x[i];
    }
    Polygon a, strip, cell;
    int idx[2];
    long loc[2];
    for (idx[0] = lo[0]; idx[0] < hi[0]; ++idx[0])
    {
        double blo, bhi;
        bounds( 0, idx[0], blo, bhi );
        if ( uMax <= blo ) break;                    // the strips only move further away
        if ( uMin >= bhi ) continue;
        if ( (loc[0] = local( 0, idx[0] )) < 0 ) continue;
        psClip2d::clipHalfPlane( sec, 0, blo,  1., a );
        psClip2d::clipHalfPlane( a,   0, bhi, -1., strip );
        if ( strip.n < 3 ) continue;
        double wMin = strip.y[0], wMax = strip.y[0];
        for (int i = 1; i < strip.n; ++i)
        {
            if ( strip.y[i] < wMin ) wMin = strip.y[i];
            if ( strip.y[i] > wMax ) wMax = strip.y[i];
        }
        for (idx[1] = lo[1]; idx[1] < hi[1]; ++idx[1])
        {
            double clo, chi;
            bounds( 1, idx[1], clo, chi );
            if ( wMax <= clo ) break;
            if ( wMin >= chi ) continue;
            if ( (loc[1] = local( 1, idx[1] )) < 0 ) continue;
            psClip2d::clipHalfPlane( strip, 1, clo,  1., a );
            psClip2d::clipHalfPlane( a,     1, chi, -1., cell );
            if ( cell.n < 3 ) continue;
            double m[3];
            moments1( cell, m );
            if ( m[0] > 0. ) visit( idx, loc, m );
        }
    }
}

}   // namespace faceSweep

#endif
