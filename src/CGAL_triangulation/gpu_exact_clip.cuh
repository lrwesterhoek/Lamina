/* The exact tetrahedron-cell overlap of the CUDA/HIP backend: the moments of one tetrahedron
   intersected with one axis-aligned grid cell (sweep_moments, order 1 and order 2), plus the small V3
   vector type and the 3x3 helpers both deposits use. Shared by ps_gpu_cuda.cu (--exact-average)
   and dtfe_gpu_cuda.cu (--exact-average). Device code only; included inside the backends' anonymous
   namespace.

   sweep_moments is a line-by-line twin of the Metal kernels' sweep_moments (metal/, the exact-clip
   code the Makefile prepends to ps_deposit.metal and dtfe_deposit.metal): the same operations in the
   same order, V3 and explicit components in place of float3/float4, swizzles and select(). The
   method is documented there: the tet is swept along z between its breakpoints (vertex heights,
   tet-edge crossings of the cell's side planes, the cell's vertical edges entering/leaving the tet);
   between two breakpoints the clipped cross-section's area and moments are polynomials in z, which
   3-point Gauss-Legendre integrates exactly; each cross-section (a convex 4-slot polygon) is clipped
   to the cell's rectangle and its 2D moments come from a closed Green chain. No thread array is
   indexed by a run-time value (registers only). It replaces the float32 r3d port (Powell & Abel
   2015) these backends used before; the CPU path keeps the double-precision r3d as the reference.

   Two numerical rules the method rests on -- keep them in any edit:
   - the Green cross term is written c = px*(qy - py) - py*(qx - px): EXACTLY 0 for a collapsed
     (dead) segment even when the compiler contracts it into an FMA. The textbook px*qy - qx*py
     leaves the rounding of one product (~eps |P|^2 per dead segment) and broke tiny pieces, the
     pass-0 / pass-1 consistency of the PS deposit and the validation;
   - no breakpoint merging: the next breakpoint is the smallest candidate c > cur, however close.
     Merging close breakpoints integrated across kinks and broke thin tilted slivers. */

#ifndef GPU_EXACT_CLIP_CUH
#define GPU_EXACT_CLIP_CUH

typedef unsigned long long ull;

// a 12-byte vector with explicit arithmetic (no operators on the runtime's float3: HIP
// defines its own and they would clash)
struct V3 { float x, y, z; };
__device__ __forceinline__ V3 v3(float x, float y, float z) { V3 r; r.x = x; r.y = y; r.z = z; return r; }
__device__ __forceinline__ V3 add3(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ V3 sub3(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ V3 neg3(V3 a) { return v3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ V3 scale3(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ V3 cross3(V3 a, V3 b)
{ return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

#define GPU_INF __int_as_float(0x7f800000)

__device__ __forceinline__ float det3(const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Product of the row lengths (Hadamard's bound on |det|).
__device__ __forceinline__ float rowNormProduct3(const float A[3][3]) {
    float p = 1.0f;
    for (int i=0; i<3; ++i) p *= sqrtf(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
    return p;
}

// Inverse of 3x3; false if |det| <= 1e-6 x the product of the row lengths -- the CPU
// matrixInverse() sentinel (math_functions.h, RELATIVE and unit-free).
__device__ __forceinline__ bool inverse3(const float A[3][3], float inv[3][3]) {
    float d = det3(A);
    if (!(fabsf(d) > 1.0e-6f * rowNormProduct3(A))) return false;
    float invd = 1.0f / d;
    inv[0][0] =  (A[1][1]*A[2][2]-A[1][2]*A[2][1])*invd;
    inv[1][0] = -(A[1][0]*A[2][2]-A[1][2]*A[2][0])*invd;
    inv[2][0] =  (A[1][0]*A[2][1]-A[1][1]*A[2][0])*invd;
    inv[0][1] = -(A[0][1]*A[2][2]-A[0][2]*A[2][1])*invd;
    inv[1][1] =  (A[0][0]*A[2][2]-A[0][2]*A[2][0])*invd;
    inv[2][1] = -(A[0][0]*A[2][1]-A[0][1]*A[2][0])*invd;
    inv[0][2] =  (A[0][1]*A[1][2]-A[0][2]*A[1][1])*invd;
    inv[1][2] = -(A[0][0]*A[1][2]-A[0][2]*A[1][0])*invd;
    inv[2][2] =  (A[0][0]*A[1][1]-A[0][1]*A[1][0])*invd;
    return true;
}

__device__ __forceinline__ int wrapIdx(int g, int n) { return ((g % n) + n) % n; }

// --ps-linear-deposit sample weight: linear density at offset rel from vertex 0, clamped >= 0
__device__ __forceinline__ float linearWeight(const float rel[3], float d0, const float dG[3]) {
    float w = d0 + dG[0]*rel[0] + dG[1]*rel[1] + dG[2]*rel[2];
    return w < 0.0f ? 0.0f : w;
}

// ================================================================================================
// The exact deposits: tet planes (a cell wholly outside a face plane holds no piece), the overlap
// frame, and the z-sweep (sweep_moments).
// ================================================================================================
// The tet's four face planes, outward (n.x <= off inside), in the vertex-0 frame.
struct TetPlanes { V3 n[4]; float off[4]; };

__device__ __forceinline__ TetPlanes tet_planes(V3 t1, V3 t2, V3 t3) {
    V3 const P[4] = { v3(0.0f, 0.0f, 0.0f), t1, t2, t3 };
    TetPlanes F;
    for (int k=0; k<4; ++k) {
        V3 const A = P[(k+1)&3], B = P[(k+2)&3], C = P[(k+3)&3];
        V3 n = cross3(sub3(B, A), sub3(C, A));
        float off = dot3(n, A);
        if (dot3(n, P[k]) - off > 0.0f) { n = neg3(n); off = -off; }   // vertex k on the inner side
        F.n[k] = n; F.off[k] = off;
    }
    return F;
}

// true when the box (centre c, half-widths h) lies strictly outside the tet
__device__ __forceinline__ bool box_outside_tet(const TetPlanes& F, V3 c, V3 h) {
    for (int k=0; k<4; ++k) {
        V3 const n = F.n[k];
        float const r = h.x*fabsf(n.x) + h.y*fabsf(n.y) + h.z*fabsf(n.z);
        float const margin = 1.0e-4f * (fabsf(n.x)*(fabsf(c.x)+h.x) + fabsf(n.y)*(fabsf(c.y)+h.y)
                                      + fabsf(n.z)*(fabsf(c.z)+h.z) + fabsf(F.off[k]));
        if (dot3(n, c) - F.off[k] > r + margin) return true;
    }
    return false;
}

// The exact deposits integrate in a frame whose origin is the centre of the OVERLAP of the tet's
// bounding box with the cell, so the piece -- which lies in that overlap -- has coordinates of its own
// size: the sweep's rounding noise is absolute (~eps x the coordinates), and a frame far from the
// piece (the vertex-0 frame once cells are small, the cell-centre frame for a small tet in a large
// cell) would bury a small piece's volume and moments in it.
// the overlap frame's origin (vertex-0 frame): the centre of the tet bbox's overlap with the cell
// (cell centre c, half-widths h; tet vertices 0, t1, t2, t3)
__device__ __forceinline__ V3 overlap_origin(V3 c, V3 h, V3 t1, V3 t2, V3 t3) {
    float const cs[3] = { c.x, c.y, c.z }, hs[3] = { h.x, h.y, h.z };
    float const a[3] = { t1.x, t1.y, t1.z }, b[3] = { t2.x, t2.y, t2.z }, e[3] = { t3.x, t3.y, t3.z };
    float o[3];
    for (int d = 0; d < 3; ++d) {
        float const tlo = fminf(fminf(0.0f, a[d]), fminf(b[d], e[d]));
        float const thi = fmaxf(fmaxf(0.0f, a[d]), fmaxf(b[d], e[d]));
        o[d] = 0.5f * (fmaxf(cs[d] - hs[d], tlo) + fminf(cs[d] + hs[d], thi));
    }
    return v3(o[0], o[1], o[2]);
}

// ------------------------------------------------------------------------------------------------
// sweep_moments: register-resident exact tet ∩ axis-aligned box moments (z-sweep + Gauss-Legendre +
// 2D Green chain), the twin of the Metal sweep_moments. Every loop that touches an array has a
// compile-time trip count and is unrolled; the only run-time loops (the interval walk and the rolled
// Gauss loop) index no array by a run-time value.
// Output: [V, x, y, z, xx, xy, xz, yy, yz, zz] about the frame origin (the overlap frame) -- true
// integrals, as r3d's reduce gave. order2=false fills mom[0..3] only and zeroes mom[0..9] (the caller
// passes 10 floats): the same nodes and expressions, so the order-1 values are bit-identical to the
// order-2 ones (the PS deposit's pass 0 and pass 1 agree exactly).
// ------------------------------------------------------------------------------------------------
#define SW_NCAND 34
#define SW_BIG 3.0e38f   // a finite sentinel: fast math may assume no infinities

struct SwM2 { float a, sx, sy, sxx, sxy, syy; };

__device__ __forceinline__ float sw_clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

__device__ __forceinline__ void sw_green(SwM2& m, float px, float py, float qx, float qy, bool order2) {
    // the cross term from the segment's own difference: EXACTLY 0 for a collapsed (dead) segment even when
    // the compiler contracts it into an FMA (px*qy - qx*py leaves the rounding of one product: ~eps |P|^2)
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
// Per side j: 0 = x >= xl, 1 = x <= xh, 2 = y >= yl, 3 = y <= yh (the Metal float4 lanes x, y, z, w).
__device__ __forceinline__ SwM2 sw_poly_rect(const float qx[4], const float qy[4],
                                             float xl, float xh, float yl, float yh, bool order2) {
    SwM2 m = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float s0[4], s1[4], s2[4], s3[4];                                    // per side, per slot (>= 0 inside)
    #pragma unroll
    for (int k = 0; k < 4; ++k) { s0[k] = qx[k] - xl; s1[k] = xh - qx[k]; s2[k] = qy[k] - yl; s3[k] = yh - qy[k]; }
    float cmin[4] = { SW_BIG, SW_BIG, SW_BIG, SW_BIG }, cmax[4] = { -SW_BIG, -SW_BIG, -SW_BIG, -SW_BIG };   // chord extents, per side
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int const k1 = (k + 1) & 3;                                          // compile-time after unrolling
        float const ax = qx[k], ay = qy[k], dxk = qx[k1] - qx[k], dyk = qy[k1] - qy[k];
        float const a[4] = { s0[k],  s1[k],  s2[k],  s3[k]  };
        float const b[4] = { s0[k1], s1[k1], s2[k1], s3[k1] };
        // a - b is -dx, +dx, -dy, +dy of the edge: one reciprocal per axis
        float const rx = 1.0f / (dxk != 0.0f ? dxk : 1.0f), ry = 1.0f / (dyk != 0.0f ? dyk : 1.0f);
        float const rs[4] = { -rx, rx, -ry, ry };
        bool ao[4], bo[4], cr[4];
        float t[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            ao[j] = a[j] < 0.0f; bo[j] = b[j] < 0.0f; cr[j] = ao[j] != bo[j];
            t[j] = cr[j] ? sw_clamp(a[j] * rs[j], 0.0f, 1.0f) : 0.0f;
        }
        float const tin  = fmaxf(fmaxf((ao[0] && !bo[0]) ? t[0] : 0.0f, (ao[1] && !bo[1]) ? t[1] : 0.0f),
                                 fmaxf((ao[2] && !bo[2]) ? t[2] : 0.0f, (ao[3] && !bo[3]) ? t[3] : 0.0f));
        float const tout = fminf(fminf((!ao[0] && bo[0]) ? t[0] : 1.0f, (!ao[1] && bo[1]) ? t[1] : 1.0f),
                                 fminf((!ao[2] && bo[2]) ? t[2] : 1.0f, (!ao[3] && bo[3]) ? t[3] : 1.0f));
        // crossings along the sides (y for the x-sides, x for the y-sides)
        float const cc[4] = { ay + t[0]*dyk, ay + t[1]*dyk, ax + t[2]*dxk, ax + t[3]*dxk };
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            cmin[j] = cr[j] ? fminf(cmin[j], cc[j]) : cmin[j];
            cmax[j] = cr[j] ? fmaxf(cmax[j], cc[j]) : cmax[j];
        }
        bool const live = !((ao[0] && bo[0]) || (ao[1] && bo[1]) || (ao[2] && bo[2]) || (ao[3] && bo[3])) && (tin < tout);
        float const t0 = live ? tin : 0.0f, t1 = live ? tout : 0.0f;         // dead edge: zero-length segment
        float const px = sw_clamp(ax + t0*dxk, xl, xh), py = sw_clamp(ay + t0*dyk, yl, yh);
        float const ex = sw_clamp(ax + t1*dxk, xl, xh), ey = sw_clamp(ay + t1*dyk, yl, yh);
        sw_green(m, px, py, ex, ey, order2);
    }
    float const lo4[4] = { yl, yl, xl, xl }, hi4[4] = { yh, yh, xh, xh };
    float ca[4], cb[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        bool const has = cmin[j] <= cmax[j];
        ca[j] = has ? sw_clamp(cmin[j], lo4[j], hi4[j]) : lo4[j];
        cb[j] = has ? sw_clamp(cmax[j], lo4[j], hi4[j]) : lo4[j];
    }
    sw_green(m, xl, cb[0], xl, ca[0], order2);   // x = xl, downward
    sw_green(m, xh, ca[1], xh, cb[1], order2);   // x = xh, upward
    sw_green(m, ca[2], yl, cb[2], yl, order2);   // y = yl, rightward
    sw_green(m, cb[3], yh, ca[3], yh, order2);   // y = yh, leftward
    m.a *= 0.5f; m.sx *= (1.0f/6.0f); m.sy *= (1.0f/6.0f);
    m.sxx *= (1.0f/12.0f); m.syy *= (1.0f/12.0f); m.sxy *= (1.0f/24.0f);
    return m;
}

// compare-exchange by z (the sorting network of sweep_moments)
__device__ __forceinline__ void sw_cx(V3& a, V3& b) {
    bool const s = b.z < a.z;
    V3 const lo = s ? b : a, hi = s ? a : b;
    a = lo; b = hi;
}

// one candidate: z where tet edge (a,b) crosses the side plane coord[ax] = pc, kept only when the crossing
// lies within the other side range (+margin): a possible vertex of T∩B
__device__ __forceinline__ float sw_edge_cand(V3 a, V3 b, int ax, float pc, float olo, float ohi) {
    float const sa = (ax == 0 ? a.x : a.y) - pc, sb = (ax == 0 ? b.x : b.y) - pc;
    bool const ok = (sa < 0.0f) != (sb < 0.0f);
    float const t = ok ? sa / (ok ? sa - sb : 1.0f) : 0.0f;
    float const o = (ax == 0) ? a.y + t*(b.y - a.y) : a.x + t*(b.x - a.x);
    float const z = a.z + t*(b.z - a.z);
    return (ok && o >= olo && o <= ohi) ? z : SW_BIG;
}

// The moments of tet (p0..p3, either orientation) ∩ box [lo, hi], all in the overlap frame.
__device__ void sweep_moments(V3 p0, V3 p1, V3 p2, V3 p3, V3 lo, V3 hi, bool order2, float* mom)
{
    #pragma unroll
    for (int m = 0; m < 10; ++m) mom[m] = 0.0f;
    V3 v0 = p0, v1 = p1, v2 = p2, v3_ = p3;
    sw_cx(v0, v1); sw_cx(v2, v3_); sw_cx(v0, v2); sw_cx(v1, v3_); sw_cx(v1, v2);
    float const Zlo = fmaxf(lo.z, v0.z), Zhi = fminf(hi.z, v3_.z);
    if (!(Zlo < Zhi)) return;
    float const D6 = dot3(sub3(v1, v0), cross3(sub3(v2, v0), sub3(v3_, v0)));
    bool const flip = !(D6 > 0.0f);

    // breakpoint candidates, in registers
    float cand[SW_NCAND];
    cand[0] = v1.z; cand[1] = v2.z;
    float const mxm = 1.0e-4f*(hi.x - lo.x), mym = 1.0e-4f*(hi.y - lo.y);
    float const yl = lo.y - mym, yh = hi.y + mym, xl = lo.x - mxm, xh = hi.x + mxm;
    #define SW_E(i, A, B) \
        cand[i+0] = sw_edge_cand(A, B, 0, lo.x, yl, yh); cand[i+1] = sw_edge_cand(A, B, 0, hi.x, yl, yh); \
        cand[i+2] = sw_edge_cand(A, B, 1, lo.y, xl, xh); cand[i+3] = sw_edge_cand(A, B, 1, hi.y, xl, xh);
    SW_E(2, v0, v1) SW_E(6, v0, v2) SW_E(10, v0, v3_) SW_E(14, v1, v2) SW_E(18, v1, v3_) SW_E(22, v2, v3_)
    #undef SW_E
    {   // the cell's 4 vertical edges: their entry / exit z through the tet (face planes n.x <= d)
        V3 fn[4]; float fd[4];
        V3 const P[4] = { v0, v1, v2, v3_ };
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            V3 const A = P[(k+1)&3], B = P[(k+2)&3], C = P[(k+3)&3];
            V3 const n = cross3(sub3(B, A), sub3(C, A)); float const d = dot3(n, A);
            bool const neg = dot3(n, P[k]) - d > 0.0f;
            fn[k] = neg ? neg3(n) : n; fd[k] = neg ? -d : d;
        }
        #pragma unroll
        for (int cr = 0; cr < 4; ++cr) {
            float const xc = (cr & 1) ? hi.x : lo.x, yc = (cr & 2) ? hi.y : lo.y;
            float zin = -SW_BIG, zout = SW_BIG;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float const r = fd[k] - fn[k].x*xc - fn[k].y*yc, nz = fn[k].z;
                float const q = r / (nz != 0.0f ? nz : 1.0f);
                zout = (nz > 0.0f) ? fminf(zout, q) : zout;
                zin  = (nz < 0.0f) ? fmaxf(zin, q) : ((nz == 0.0f && r < 0.0f) ? SW_BIG : zin);
            }
            bool const ok = zin <= zout;
            cand[26 + 2*cr] = ok ? zin : SW_BIG;
            cand[27 + 2*cr] = ok ? zout : SW_BIG;
        }
    }

    float cur = Zlo;
    for (int it = 0; it < SW_NCAND + 1 && cur < Zhi; ++it) {     // run-time trip count, no indexing by it
        // the next breakpoint: the smallest candidate beyond cur, however close (no merging)
        float nxt = Zhi;
        #pragma unroll
        for (int k = 0; k < SW_NCAND; ++k) { float const c = cand[k]; nxt = (c > cur && c < nxt) ? c : nxt; }
        float const h = 0.5f*(nxt - cur), zm = 0.5f*(nxt + cur);
        #pragma unroll 1
        for (int g = 0; g < 3; ++g) {                               // rolled: unrolling made the sweep 0.6x on Metal
            float const xi = (g == 0) ? -0.7745966692414834f : ((g == 1) ? 0.0f : 0.7745966692414834f);
            float const w  = h * ((g == 1) ? (8.0f/9.0f) : (5.0f/9.0f));
            float const z  = zm + h*xi;
            bool const b1 = z <= v1.z, b2 = z <= v2.z;
            // slots: s0 on 03; s1 on 13|01; s2 on 12|01|23; s3 on 02|23 (ccw for a positive sorted tet)
            V3 const A1 = b1 ? v0 : v1,                  B1 = b1 ? v1 : v3_;
            V3 const A2 = b1 ? v0 : (b2 ? v1 : v2),      B2 = b1 ? v1 : (b2 ? v2 : v3_);
            V3 const A3 = b2 ? v0 : v2,                  B3 = b2 ? v2 : v3_;
            float const Az[4] = { v0.z, A1.z, A2.z, A3.z }, Bz[4] = { v3_.z, B1.z, B2.z, B3.z };
            float const Ax[4] = { v0.x, A1.x, A2.x, A3.x }, Bx[4] = { v3_.x, B1.x, B2.x, B3.x };
            float const Ay[4] = { v0.y, A1.y, A2.y, A3.y }, By[4] = { v3_.y, B1.y, B2.y, B3.y };
            float qx[4], qy[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float const dz = Bz[j] - Az[j];
                float const t = sw_clamp((dz > 0.0f) ? (z - Az[j]) / ((dz > 0.0f) ? dz : 1.0f) : 0.5f, 0.0f, 1.0f);
                qx[j] = Ax[j] + t*(Bx[j] - Ax[j]); qy[j] = Ay[j] + t*(By[j] - Ay[j]);
            }
            {   // a negative tet: the slots run cw -> swap slots 1 and 3 (Metal's .xwzy)
                float const x1 = qx[1], y1 = qy[1];
                qx[1] = flip ? qx[3] : qx[1]; qx[3] = flip ? x1 : qx[3];
                qy[1] = flip ? qy[3] : qy[1]; qy[3] = flip ? y1 : qy[3];
            }
            // per-z 2D origin: centre of bbox(q) ∩ R
            float const ox = sw_clamp(0.5f*(fmaxf(fminf(fminf(qx[0], qx[1]), fminf(qx[2], qx[3])), lo.x)
                                          + fminf(fmaxf(fmaxf(qx[0], qx[1]), fmaxf(qx[2], qx[3])), hi.x)), lo.x, hi.x);
            float const oy = sw_clamp(0.5f*(fmaxf(fminf(fminf(qy[0], qy[1]), fminf(qy[2], qy[3])), lo.y)
                                          + fminf(fmaxf(fmaxf(qy[0], qy[1]), fmaxf(qy[2], qy[3])), hi.y)), lo.y, hi.y);
            float const rx[4] = { qx[0] - ox, qx[1] - ox, qx[2] - ox, qx[3] - ox };
            float const ry[4] = { qy[0] - oy, qy[1] - oy, qy[2] - oy, qy[3] - oy };
            SwM2 const q = sw_poly_rect(rx, ry, lo.x - ox, hi.x - ox, lo.y - oy, hi.y - oy, order2);
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

#undef SW_NCAND
#undef SW_BIG

#endif
