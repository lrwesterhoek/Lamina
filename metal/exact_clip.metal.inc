//  The GPU exact tetrahedron-cell intersection: the volume and the first and second moments of one
//  tetrahedron intersected with one axis-aligned grid cell, in float32. Shared by the phase-space
//  deposit (ps_deposit.metal: --exact) and the standard-DTFE exact volume average
//  (dtfe_deposit.metal: --exact); the Makefile PREPENDS this file to both kernel sources
//  when it embeds them (and metal/validate_deposit.cpp does the same when it loads the source).
//  Keep it self-contained: it is compiled before everything else in each kernel source.
#include <metal_stdlib>
using namespace metal;

// The tet's four face planes, outward (n.x <= off inside), in the vertex-0 frame: a cell lying
// wholly beyond one of them cannot intersect the tet, so it is skipped before any clipping. A
// stretched (folded, sheet-like) tet's axis-aligned window holds mostly such cells -- a diagonal
// 16x16x0.1 Mpc sheet spans ~40^3 cells of 0.4 Mpc and touches ~40^2 -- and clipping every one of
// them twice made the exact deposit crawl when cells are much smaller than tets (0.26M particles
// on 256^3: >17 min). Skipped cells clip to nothing anyway: same output, to float rounding.
struct TetPlanes { float3 n[4]; float off[4]; };

static inline TetPlanes tet_planes(float3 t1, float3 t2, float3 t3) {
    float3 const P[4] = { float3(0.0f), t1, t2, t3 };
    TetPlanes F;
    for (int k=0; k<4; ++k) {
        float3 const A = P[(k+1)&3], B = P[(k+2)&3], C = P[(k+3)&3];
        float3 n = cross(B - A, C - A);
        float off = dot(n, A);
        if (dot(n, P[k]) - off > 0.0f) { n = -n; off = -off; }   // vertex k on the inner side
        F.n[k] = n; F.off[k] = off;
    }
    return F;
}

// true when the box (centre c, half-widths h) lies strictly outside the tet; the margin keeps a
// box that touches or overlaps it within float rounding
static inline bool box_outside_tet(thread const TetPlanes& F, float3 c, float3 h) {
    for (int k=0; k<4; ++k) {
        float3 const n = F.n[k];
        float const r = h.x*fabs(n.x) + h.y*fabs(n.y) + h.z*fabs(n.z);
        float const margin = 1.0e-4f * (fabs(n.x)*(fabs(c.x)+h.x) + fabs(n.y)*(fabs(c.y)+h.y)
                                      + fabs(n.z)*(fabs(c.z)+h.z) + fabs(F.off[k]));
        if (dot(n, c) - F.off[k] > r + margin) return true;
    }
    return false;
}

// ================================================================================================
// sweep_moments: the moments [V, x, y, z, xx, xy, xz, yy, yz, zz] of a tetrahedron T intersected
// with an axis-aligned box B, as true integrals (V = the volume, x = the integral of x over T∩B,
// ...), about the frame origin. order2=false fills mom[0..3] only.
//
// The method is a sweep in z. The horizontal cross-section of T at height z is a convex polygon
// of 3 or 4 vertices, each on a tet edge; with the vertices sorted by z they sit in 4 fixed SLOTS
// (s0 on edge 03; s1 on 01 or 13; s2 on 01, 12 or 23; s3 on 02 or 23; one vertex twice where the
// section is a triangle), counter-clockwise for a positive tet and reversed otherwise. Its
// intersection with the cell's rectangle R changes shape only at BREAKPOINTS: the z of the tet's
// middle vertices, the z where a tet edge crosses one of R's four side planes within R's range
// along that side (with a margin of 1e-4 of the cell: an extra breakpoint costs one interval, a
// missed one the accuracy), and the z where one of R's four vertical edges enters or leaves T --
// at most SW_NCAND = 34. Between two breakpoints every vertex of the clipped polygon moves
// linearly in z, so its area and 2D moments are polynomials in z (the area of degree 2, the
// second moments of degree 4, and the z-weighted moments at most 4), and the 3-point
// Gauss-Legendre rule (exact to degree 5) integrates each interval exactly up to float rounding.
//
// The 2D moments at a node come from Green's theorem along a closed chain: the polygon's 4 edges,
// each Liang-Barsky clipped to R, plus the chords along R's 4 sides between the outermost
// crossings of that side. Every (slot, side) pair has ONE signed distance and every (edge, side)
// pair ONE crossing, used by both the edge and the chord, so the chain closes in every degeneracy
// (a vertex on a side, an edge along it, a collapsed slot). Each node's 2D moments are taken about
// the centre of the section's bounding box intersected with R and shifted back, so they do not
// cancel when the section is small against the frame.
//
// Everything lives in registers. Apple GPUs cannot index registers at run time: a thread array
// indexed by a run-time value sits in memory. The r3d port this replaced (a vertex graph of up to
// 48 vertices with neighbour lists, clipped plane by plane and reduced by a walk over its faces)
// spilled 832 B/thread and was latency-bound. Here every array access has a compile-time index
// after unrolling (the candidate list, the 4 slots, the 4 sides); the two run-time loops -- the
// walk over the intervals and the 3 Gauss nodes, which stays rolled because unrolling it made the
// sweep 0.6x as fast -- index nothing.
//
// Two numerical rules, both found by validation and both required:
//  - The Green cross term of a segment P->Q is written c = px*(qy - py) - py*(qx - px), not
//    px*qy - qx*py. Both are the same number in exact arithmetic, but a clipped-away edge is a
//    COLLAPSED segment (P == Q) and fast math may contract the product difference into an FMA;
//    the difference form is then still exactly 0, the product form leaves the rounding of one
//    product (~eps |P|^2) per dead segment. That noise broke tiny pieces (whose moments it
//    exceeded), the order-1 against order-2 agreement and validate_deposit.
//  - No breakpoint merging: the next breakpoint is the smallest candidate STRICTLY above the
//    current one, however close. Merging close candidates integrated a polynomial across a kink
//    and broke thin tilted slivers. The Gauss nodes lie strictly inside each interval and never on
//    its ends, where the section changes shape and the slot choice (z <= v1.z, z <= v2.z) is a tie:
//    each node sees the one polygon valid on the open interval.
//
// The order-1 path (order2=false) does the same arithmetic for V and the first moments as the
// order-2 one and skips the rest, so the exact deposit's pass 0 (weights) and pass 1 (deposit)
// see identical volumes. Only the volumes: compiled as two pipelines, Metal's fast math may contract
// the first moments differently (measured: ~17% of pairs off, by at most 3.7e-8 of the cell),
// which would reach a deposit only through --ps-linear-deposit's weight (about 1e-8 of a tet's
// mass): that deposit runs both passes at order 2 in one pipeline (ps_deposit.metal,
// depositExactItems' kItemsPhase 2), so both passes form the same weights and its total carries
// only the rounding of each item's float sum, as the uniform deposit's does.
//
// The frame. The exact deposits clip in a frame whose origin is the centre of the OVERLAP of the
// tet's bounding box with the cell, so the clipped piece -- which lies in that overlap -- has
// coordinates of its own size and its volume and moments do not cancel. The r3d port took the
// piece's volume from origin-apex simplices, so a piece far from the origin cancelled: in the
// vertex-0 frame a piece a few Mpc from vertex 0 lost its volume once cells were small (a zoom's
// virtual grid: 1.6% per cell at 12 kpc, 20% at 3 kpc), and in the cell-centre frame a small tet
// in a large cell did the same. The overlap is near the piece in both regimes, and the sweep keeps
// it: its z moments and every second moment are taken about the frame origin, and the callers'
// centroids and covariances subtract the frame's offset back out.
//
// Metal has no double, so the moments match the CPU exact deposits (double r3d) to float
// rounding. Against the CPU on a 0.26M-particle box at 256^3 the phase-space a_den differs by
// 1.2e-3 of its rms at most (as with the r3d port) and a_velDisp by 2.5e-3 (the r3d port: 3.0e-2);
// causticClass is identical. One known residual: a tet face lying exactly in a cell plane (a
// perfect lattice) leaves a slab <= 1e-7 cell thick in the cell it only touches, which then
// counts as a tet touch; its mass is ~1e-7 of the tet's. Speed (M1 Max, one partition, the whole
// --exact): 18.1 -> 8.7 s for 0.26M particles on 256^3, 10.8 -> 6.2 s for 2.1M on
// 128^3, 35.7 -> 18.1 s for 2.1M on 256^3, against the r3d port.
// ================================================================================================
#define SW_NCAND 34
#define SW_BIG 3.0e38f   // a finite sentinel: fast math may assume no infinities

struct SwM2 { float a, sx, sy, sxx, sxy, syy; };

// one segment P->Q of the closed Green chain (the normalizations are applied once, at the end)
static inline void sw_green(thread SwM2& m, float px, float py, float qx, float qy, bool order2) {
    // the cross term from the segment's own difference: EXACTLY 0 for a collapsed (dead) segment even when the
    // compiler contracts it into an FMA (px*qy - qx*py leaves the rounding of one product: absolute noise ~eps |P|^2)
    float const c = px*(qy - py) - py*(qx - px);
    m.a  += c;
    m.sx += (px + qx)*c;  m.sy += (py + qy)*c;
    if (order2) {
        m.sxx += (px*px + px*qx + qx*qx)*c;
        m.syy += (py*py + py*qy + qy*qy)*c;
        m.sxy += (2.0f*(px*py + qx*qy) + px*qy + qx*py)*c;
    }
}

// 2D moments of the convex ccw 4-slot polygon (q0..q3, duplicates allowed) ∩ [xl,xh]x[yl,yh].
// Closed chain: the 4 edges Liang-Barsky clipped to R + the chords of R's 4 sides, one sign per
// (slot, side) and one crossing per (edge, side) shared by both -> consistent in every degeneracy.
static inline SwM2 sw_poly_rect(float4 qx, float4 qy, float xl, float xh, float yl, float yh, bool order2) {
    SwM2 m = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float4 const s0 = qx - xl, s1 = xh - qx, s2 = qy - yl, s3 = yh - qy;   // per side, per slot (>= 0 inside)
    float4 cmin = float4(SW_BIG), cmax = float4(-SW_BIG);              // chord extents, per side
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int const k1 = (k + 1) & 3;                                          // compile-time after unrolling
        float const ax = qx[k], ay = qy[k], dxk = qx[k1] - qx[k], dyk = qy[k1] - qy[k];
        float4 const a = float4(s0[k], s1[k], s2[k], s3[k]);
        float4 const b = float4(s0[k1], s1[k1], s2[k1], s3[k1]);
        bool4 const ao = a < 0.0f, bo = b < 0.0f;
        bool4 const cr = ao != bo;
        // a - b is -dx, +dx, -dy, +dy of the edge: one reciprocal per axis
        float const rx = 1.0f / (dxk != 0.0f ? dxk : 1.0f), ry = 1.0f / (dyk != 0.0f ? dyk : 1.0f);
        float4 const t = select(float4(0.0f), clamp(a * float4(-rx, rx, -ry, ry), 0.0f, 1.0f), cr);
        float const tin  = max(max(select(0.0f, t.x, ao.x && !bo.x), select(0.0f, t.y, ao.y && !bo.y)),
                               max(select(0.0f, t.z, ao.z && !bo.z), select(0.0f, t.w, ao.w && !bo.w)));
        float const tout = min(min(select(1.0f, t.x, !ao.x && bo.x), select(1.0f, t.y, !ao.y && bo.y)),
                               min(select(1.0f, t.z, !ao.z && bo.z), select(1.0f, t.w, !ao.w && bo.w)));
        // crossings along the sides (y for the x-sides, x for the y-sides)
        float4 const cc = float4(ay + t.x*dyk, ay + t.y*dyk, ax + t.z*dxk, ax + t.w*dxk);
        cmin = select(cmin, min(cmin, cc), cr);
        cmax = select(cmax, max(cmax, cc), cr);
        bool const live = !any(ao && bo) && (tin < tout);
        float const t0 = live ? tin : 0.0f, t1 = live ? tout : 0.0f;          // dead edge: zero-length segment
        float const px = clamp(ax + t0*dxk, xl, xh), py = clamp(ay + t0*dyk, yl, yh);
        float const ex = clamp(ax + t1*dxk, xl, xh), ey = clamp(ay + t1*dyk, yl, yh);
        sw_green(m, px, py, ex, ey, order2);
    }
    bool4 const has = cmin <= cmax;
    float4 const lo4 = float4(yl, yl, xl, xl), hi4 = float4(yh, yh, xh, xh);
    float4 const ca = select(lo4, clamp(cmin, lo4, hi4), has);
    float4 const cb = select(lo4, clamp(cmax, lo4, hi4), has);
    sw_green(m, xl, cb.x, xl, ca.x, order2);   // x = xl, downward
    sw_green(m, xh, ca.y, xh, cb.y, order2);   // x = xh, upward
    sw_green(m, ca.z, yl, cb.z, yl, order2);   // y = yl, rightward
    sw_green(m, cb.w, yh, ca.w, yh, order2);   // y = yh, leftward
    m.a *= 0.5f; m.sx *= (1.0f/6.0f); m.sy *= (1.0f/6.0f);
    m.sxx *= (1.0f/12.0f); m.syy *= (1.0f/12.0f); m.sxy *= (1.0f/24.0f);
    return m;
}

// compare-exchange by z (the vertex sort's network)
static inline void sw_cx(thread float3& a, thread float3& b) {
    bool const s = b.z < a.z;
    float3 const lo = select(a, b, s), hi = select(b, a, s);
    a = lo; b = hi;
}

// one candidate: z where tet edge (a,b) crosses the side plane coord[ax] = pc, kept only when the crossing
// lies within the other side range (+margin): a possible vertex of T∩B
static inline float sw_edge_cand(float3 a, float3 b, int ax, float pc, float olo, float ohi) {
    float const sa = (ax == 0 ? a.x : a.y) - pc, sb = (ax == 0 ? b.x : b.y) - pc;
    bool const ok = (sa < 0.0f) != (sb < 0.0f);
    float const t = ok ? sa / (ok ? sa - sb : 1.0f) : 0.0f;
    float const o = (ax == 0) ? a.y + t*(b.y - a.y) : a.x + t*(b.x - a.x);
    float const z = a.z + t*(b.z - a.z);
    return (ok && o >= olo && o <= ohi) ? z : SW_BIG;
}

// The tet (p0..p3, either orientation) ∩ the box [lo, hi], all in the caller's frame; see the header above.
static inline void sweep_moments(float3 p0, float3 p1, float3 p2, float3 p3, float3 lo, float3 hi,
                                 bool order2, thread float* mom)
{
    for (int m = 0; m < 10; ++m) mom[m] = 0.0f;
    float3 v0 = p0, v1 = p1, v2 = p2, v3 = p3;
    sw_cx(v0, v1); sw_cx(v2, v3); sw_cx(v0, v2); sw_cx(v1, v3); sw_cx(v1, v2);
    float const Zlo = max(lo.z, v0.z), Zhi = min(hi.z, v3.z);
    if (!(Zlo < Zhi)) return;
    float const D6 = dot(v1 - v0, cross(v2 - v0, v3 - v0));
    bool const flip = !(D6 > 0.0f);

    // breakpoint candidates, in registers
    float cand[SW_NCAND];
    cand[0] = v1.z; cand[1] = v2.z;
    float const mxm = 1.0e-4f*(hi.x - lo.x), mym = 1.0e-4f*(hi.y - lo.y);
    float const yl = lo.y - mym, yh = hi.y + mym, xl = lo.x - mxm, xh = hi.x + mxm;
    #define SW_E(i, A, B) \
        cand[i+0] = sw_edge_cand(A, B, 0, lo.x, yl, yh); cand[i+1] = sw_edge_cand(A, B, 0, hi.x, yl, yh); \
        cand[i+2] = sw_edge_cand(A, B, 1, lo.y, xl, xh); cand[i+3] = sw_edge_cand(A, B, 1, hi.y, xl, xh);
    SW_E(2, v0, v1) SW_E(6, v0, v2) SW_E(10, v0, v3) SW_E(14, v1, v2) SW_E(18, v1, v3) SW_E(22, v2, v3)
    #undef SW_E
    {   // the cell's 4 vertical edges: their entry / exit z through the tet (face planes n.x <= d)
        float3 fn[4]; float fd[4];
        float3 const P[4] = { v0, v1, v2, v3 };
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float3 const A = P[(k+1)&3], B = P[(k+2)&3], C = P[(k+3)&3];
            float3 n = cross(B - A, C - A); float d = dot(n, A);
            bool const neg = dot(n, P[k]) - d > 0.0f;
            fn[k] = neg ? -n : n; fd[k] = neg ? -d : d;
        }
        #pragma unroll
        for (int cr = 0; cr < 4; ++cr) {
            float const xc = (cr & 1) ? hi.x : lo.x, yc = (cr & 2) ? hi.y : lo.y;
            float zin = -SW_BIG, zout = SW_BIG;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float const r = fd[k] - fn[k].x*xc - fn[k].y*yc, nz = fn[k].z;
                float const q = r / (nz != 0.0f ? nz : 1.0f);
                zout = (nz > 0.0f) ? min(zout, q) : zout;
                zin  = (nz < 0.0f) ? max(zin, q) : ((nz == 0.0f && r < 0.0f) ? SW_BIG : zin);
            }
            bool const ok = zin <= zout;
            cand[26 + 2*cr] = ok ? zin : SW_BIG;
            cand[27 + 2*cr] = ok ? zout : SW_BIG;
        }
    }

    // the walk over the intervals [cur, nxt]: the next breakpoint is the smallest candidate strictly
    // above cur (no merging, see the header)
    float cur = Zlo;
    for (int it = 0; it < SW_NCAND + 1 && cur < Zhi; ++it) {     // run-time trip count, no indexing by it
        float nxt = Zhi;
        #pragma unroll
        for (int k = 0; k < SW_NCAND; ++k) { float const c = cand[k]; nxt = (c > cur && c < nxt) ? c : nxt; }
        float const h = 0.5f*(nxt - cur), zm = 0.5f*(nxt + cur);
        // 3-point Gauss-Legendre, nodes strictly inside the interval; rolled (unrolled it was 0.6x as fast)
        #pragma clang loop unroll(disable)
        for (int g = 0; g < 3; ++g) {
            float const xi = (g == 0) ? -0.7745966692414834f : ((g == 1) ? 0.0f : 0.7745966692414834f);
            float const w  = h * ((g == 1) ? (8.0f/9.0f) : (5.0f/9.0f));
            float const z  = zm + h*xi;
            bool const b1 = z <= v1.z, b2 = z <= v2.z;
            // slots: s0 on 03; s1 on 13|01; s2 on 12|01|23; s3 on 02|23 (ccw for a positive sorted tet)
            float3 const A1 = select(v1, v0, b1),                  B1 = select(v3, v1, b1);
            float3 const A2 = select(select(v2, v1, b2), v0, b1),  B2 = select(select(v3, v2, b2), v1, b1);
            float3 const A3 = select(v2, v0, b2),                  B3 = select(v3, v2, b2);
            float4 const Az = float4(v0.z, A1.z, A2.z, A3.z), Bz = float4(v3.z, B1.z, B2.z, B3.z);
            float4 const Ax = float4(v0.x, A1.x, A2.x, A3.x), Bx = float4(v3.x, B1.x, B2.x, B3.x);
            float4 const Ay = float4(v0.y, A1.y, A2.y, A3.y), By = float4(v3.y, B1.y, B2.y, B3.y);
            float4 const dz = Bz - Az;
            float4 const t = clamp(select(float4(0.5f), (z - Az) / select(float4(1.0f), dz, dz > 0.0f), dz > 0.0f), 0.0f, 1.0f);
            float4 qx = Ax + t*(Bx - Ax), qy = Ay + t*(By - Ay);
            qx = flip ? qx.xwzy : qx;  qy = flip ? qy.xwzy : qy;
            // per-z 2D origin: centre of bbox(q) ∩ R
            float const ox = clamp(0.5f*(max(min(min(qx.x, qx.y), min(qx.z, qx.w)), lo.x) + min(max(max(qx.x, qx.y), max(qx.z, qx.w)), hi.x)), lo.x, hi.x);
            float const oy = clamp(0.5f*(max(min(min(qy.x, qy.y), min(qy.z, qy.w)), lo.y) + min(max(max(qy.x, qy.y), max(qy.z, qy.w)), hi.y)), lo.y, hi.y);
            SwM2 q = sw_poly_rect(qx - ox, qy - oy, lo.x - ox, hi.x - ox, lo.y - oy, hi.y - oy, order2);
            float const wa = w*q.a;
            mom[0] += wa;
            mom[1] += w*(q.sx + ox*q.a);
            mom[2] += w*(q.sy + oy*q.a);
            mom[3] += wa*z;
            if (order2) {
                mom[4] += w*(q.sxx + 2.0f*ox*q.sx + ox*ox*q.a);
                mom[5] += w*(q.sxy + ox*q.sy + oy*q.sx + ox*oy*q.a);
                mom[6] += w*z*(q.sx + ox*q.a);
                mom[7] += w*(q.syy + 2.0f*oy*q.sy + oy*oy*q.a);
                mom[8] += w*z*(q.sy + oy*q.a);
                mom[9] += wa*z*z;
            }
        }
        cur = nxt;
    }
}
