/* The face sweep: exact integrals of a linear field over the grid's faces (the face method of the velocity
   gradient, '--face-gradient'). A tetrahedron owns its plane section when min_axis <= c < max_axis (half-open),
   so a face lying in the plane is counted once. Everything is in double, in the caller's frame. */

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

// Area and first moments [A, A*u_c, A*w_c] of a polygon (sign taken off); Green's theorem, as ps_clip2d.h.
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

/* The section of the tetrahedron p[4][3] by the plane x_axis = c, counter-clockwise in (u, w). d[4]: the vertices'
   signed distances to the plane, computed by the caller from coordinates all tetrahedra sharing a vertex agree
   on, so a shared face in the plane is never decided two ways. False when the section has no positive area. */
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

/* Every face square of the window [lo, hi) whose piece of the section 'sec' has positive area, in iu-iw order.
   bounds(k, idx, lo, hi): the square's planes along in-plane axis k; local(k, idx): grid position, < 0 to skip;
   visit(idx[2], loc[2], m[3]): the piece's area and first moments. Early exits as exact_sweep.h. */
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
