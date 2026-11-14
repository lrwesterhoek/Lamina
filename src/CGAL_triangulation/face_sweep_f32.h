/* The face sweep in ONE float arithmetic for the CPU and the GPU (docs/cleanup-plan.md P11: the GPU kernel for
   PS-DTFE's face sweep, step 2 of its design): PS-DTFE's face sums (--face-gradient) formed so that the CPU and a
   Metal kernel give the same float shares bit for bit, and so the same fixed-point face words. face_sweep.h is the
   double sweep of the default; this header is the opt-in one (DTFE_FACE_F32=1, DTFE_GPU_FACES=1: ps_interpolation.cc
   psFaceF32).

   Why a second sweep: the default forms every share in double, and Metal has no double. A float evaluation of the
   same formulas differs at ~1e-7 relative, which moves the integer of a share by 2^1 to 2^11 units. Identity needs
   one arithmetic both devices run operation for operation:
   - every operation a correctly rounded IEEE single operation, nothing reassociated: the Metal library that includes
     this is compiled in the safe math mode; every product that feeds a sum is an explicit FS_FMA, and no expression
     here has the form a*b + c (so there is nothing for a compiler to contract, on top of the pragma below);
   - flush-to-zero, the device's subnormal rule: an Apple GPU takes a subnormal operand for a zero and gives a zero
     for a result below FLT_MIN, so the CPU runs this sweep under the same rule (fsFlushed below);
   - no transcendental function: the corners of a section are ordered combinatorially, not by atan2;
   - every input formed once, on the host (the velocity gradient, the density: psFaceVelGrad, psFaceRho), never
     recomputed on a device.
   The fixed-point step after the float share (share x 2^k, the guard, rint) is NOT here: the CPU's psFxAdd(double(
   share)) and the kernels' fixedShareOk / addFixedQ / addShare give the same integer for every float share (the
   product with 2^k is exact in float and in double; a NaN, an infinity or a product of 2^63 units or more is left
   out and counted on both).

   The common subset of C++17 and the Metal Shading Language: no std:: (FS_FMA / FS_FLOOR), no lambdas, no double,
   no long (64-bit in MSL), fixed-size arrays (FsPoly: at most FS_MAXV = 8 vertices -- a section has at most 4, and
   each of the four half-plane clips of a piece adds at most one), loops bounded by a constant or by the data,
   FS_THREAD before every reference and array parameter (MSL's thread address space). Under MSL the kernel source
   must have <metal_stdlib> and 'using namespace metal' before this file (metal/fixed_words.metal.inc, which the
   Makefile prepends to metal/ps_face.metal before this file, gives them). The sweep of one plane with a sink
   (fsSweepPlane) is the CPU's and the kernel's alike, each with its own sink (ps_interpolation.cc PSFaceF32Sink, the
   kernel's FaceSink); the loop over a tetrahedron's planes (fsDepositTet) is the CPU's, a kernel item being one
   (axis, plane) of it.
   The GPU's registers (the kernel's design, K7): an Apple GPU keeps a thread array in memory once a run-time value
   indexes it (metal/exact_clip.metal.inc says why that is slow). With "#define FS_REGISTERS 1" in front of the kernel
   source, every write of a polygon vertex is predicated over the FS_MAXV slots (fsPut), every read by a vertex number
   a select chain (FS_AT4, FS_AT4C) and the loops over vertices and slots are unrolled, so the indices are known at
   compile time; the slots are zeroed first (fsPolyClear), so no select reads an undefined value. The same values land
   in the same slots, so the words do not move. The CPU, and the kernel without FS_REGISTERS, index the arrays
   directly. Which form is faster on the device is TO MEASURE (the kernel harness, BENCH=1).

   The arithmetic, in order (u, w: the plane's in-plane axes in increasing order, as face_sweep.h; all float):
   1. Frames (fsTetSet): R[v] = E[v] - lo, the region frame, which decides the sides (a function of the vertex's own
      wrapped position, so every tetrahedron sharing a vertex decides alike); P[v] = E[v] - E[0], the vertex-0
      frame, which places the corners.
   2. Plane candidates per axis (fsPlanes): floor(min R ih) - 1 .. floor(max R ih) + 1, ih = 1/h formed once on the
      host; clamped to [0, n] when not periodic. Only candidates: step 3 decides.
   3. Sides: d[v] = fma(-k, h, R[v][a]) -- the exact R - k h, rounded once, so its sign is exact. Owned iff
      min d <= 0 < max d (face_sweep.h's half-open rule). The plane in the vertex-0 frame: c = fma(k, h, -R[0][a]).
   4. Corners (fsSection): a vertex with d == 0 is a corner (P[v][u], P[v][w]); an edge (i < j) whose ends lie
      strictly on opposite sides gives t = d[i] / (d[i] - d[j]) and the corner fma(t, P[j] - P[i], P[i]). The order:
      three corners are a cycle in any order (vertex corners in vertex order, then edges in (i, j) order); four come
      only from a 2-2 split, below {p < q}, above {r < s}, as the cycle of edges (p,r), (p,s), (q,s), (q,r). Green's
      sum takes the orientation's sign off.
   5. Section check: the doubled area by step 7's sum (about the section's own corner 0) neither > 0 nor < 0: none.
   6. Strips and cells (fsBound, fsClip, fsSlab): strip i along u has the bounds fsBound(i) = fma(i, h_u, -R[0][u])
      and fsBound(i + 1), the next strip's own lower bound, so neighbouring pieces share their bound bit for bit (the
      double sweep's bhi = blo + dx rounds apart); cells along w alike. Each clip is Sutherland-Hodgman as
      psClip2d::clipHalfPlane: the side value x - bound (or bound - x), a crossing t = v_i / (v_i - v_j), its other
      coordinate fma(t, o_j - o_i, o_i) and its clipped coordinate the bound exactly. A strip is clipped from the
      section (lower bound, then upper), a cell from its strip (lower, then upper).
   7. A piece's moments (fsPieceShares) about ITS OWN vertex 0 (x0, y0): x' = x - x0, y' = y - y0, and over the edges
      (i, i+1 mod n): cr = fma(x'_i, y'_{i+1}, -(x'_{i+1} y'_i)); A2 += cr; Mu6 = fma(x'_i + x'_{i+1}, cr, Mu6); Mw6 =
      fma(y'_i + y'_{i+1}, cr, Mw6). All three negated if A2 < 0; no piece unless A2 > 0. A = A2 * 0.5, Mu = Mu6 * K6,
      Mw = Mw6 * K6 (K6 = the float nearest 1/6, so no division).
   8. Shares: the velocity at the piece's vertex 0, vS[j] = fma(vG[w][j], y0, fma(vG[u][j], x0, fma(vG[a][j], c,
      v0[j]))); S[j] = fma(vG[w][j], Mw, fma(vG[u][j], Mu, A vS[j])) (exact algebra for a linear velocity); N = A;
      P = rho S[a].
   Steps 7 and 8 differ from the kernel's design, which took the moments about the face square's corner (blo, clo)
   and the velocity there: a piece much smaller than its face square then cancels -- both the cross products of
   coordinates ~h for an area ~L^2 and the velocity extrapolated ~h from a tetrahedron of size L (its gradient ~dv/L)
   -- and in a float32 emulation the share's error grew as (h/L)^2, to 7e-2 of the piece's own scale at L = h/100,
   where halo tetrahedra sit. About the piece's own vertex both stay at float rounding (~1e-7 of the piece's scale at
   every size, the same emulation). A side effect: a clipped piece whose points all lie on one strip or cell bound
   has A2 == 0 exactly, so a sweep that tests more strips or cells than another (no early exit, a wider candidate
   range, no sub-box test before the clip) deposits the same pieces.

   What a caller does per tetrahedron (fsDepositTet below, the CPU's; a kernel item does one (axis, plane) of it):
     fsTetSet(E, g, vG, v0, rho, t);
     for each axis a: n = fsPlanes(t, g, a, k0); for k in [k0, k0 + n): fsSweepPlane(t, g, a, k, sink)
   and fsSweepPlane: below = fsLocal(g, a, k - 1), above = fsLocal(g, a, k) (none: next plane); fsSection; for every
   strip i of fsInPlaneRange(u) with a non-negative fsLocal(g, u, i): fsSlab(sec, 0, fsBound(i), fsBound(i + 1));
   for every cell j of fsInPlaneRange(w) with a non-negative fsLocal(g, w, j): fsSlab(strip, 1, ...);
   fsPieceShares; the piece's shares to the cell below the plane (its UPPER face, side 1) and the cell above (its
   LOWER face, side 0), each where its local index is >= 0. The early exits of fsSweepPlane are optional (above). */

#ifndef FACE_SWEEP_F32_HEADER
#define FACE_SWEEP_F32_HEADER

#ifdef __METAL_VERSION__
#define FS_FMA(a, b, c) metal::fma( (a), (b), (c) )
#define FS_FLOOR(x)     metal::floor( (x) )
#define FS_THREAD       thread
#define FS_NO_CONTRACT
#else
#include <cmath>
// the float overloads, by type: an int or a double argument converts to float here instead of choosing the double fma
static inline float fsFmaF(float const a, float const b, float const c) { return std::fma( a, b, c ); }
static inline float fsFloorF(float const x) { return std::floor( x ); }
#define FS_FMA(a, b, c) fsFmaF( (a), (b), (c) )
#define FS_FLOOR(x)     fsFloorF( (x) )
#define FS_THREAD
// clang contracts a*b + c inside one expression by default; no expression here has that form, and this keeps it so
// in every function body (block scope: nothing leaks into the including file). GCC builds take -ffp-contract=off.
#if defined(__clang__)
#define FS_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define FS_NO_CONTRACT
#endif
/* The device's subnormal rule on the CPU (fsFlushed). An Apple GPU computes float in flush-to-zero mode: a subnormal
   operand counts as a zero of its sign, and a result whose exact value lies below FLT_MIN is a zero of its sign (the
   face kernel's device probe measured it: metal/validate_deposit.cpp T16a). AArch64's FPCR.FZ (bit 24) is that rule,
   so every CPU evaluation that must give the face kernel's bits runs under it:
       fsFlushed( [&] { FsTet t; fsTetSet( ... ); fsDepositTet( t, g, sink ); } );
   fsFlushed is never inlined and its two writes of the register are barriers for memory, so no float operation of
   the caller runs under the rule and none of f outside it: f takes its inputs and leaves its results through memory
   (capture by reference; form the inputs before the call). The register is the calling thread's own and is put back
   as it was, also when f throws. The rule moves a value only where an operand or an exact result is below 1.18e-38
   in size, so the canonical arithmetic is: correctly rounded IEEE single operations, flush-to-zero. Without AArch64
   (no face kernel runs there) f runs as it is. */
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
#define FS_FLUSH_RULE 1
static inline unsigned long long fsFpcr()
{
    unsigned long long v;
    __asm__ __volatile__( "mrs %0, fpcr" : "=r"( v ) );
    return v;
}
static inline void fsFpcrSet(unsigned long long const v)
{
    __asm__ __volatile__( "msr fpcr, %0" : : "r"( v ) : "memory" );
}
struct FsFpcrKeep
{
    unsigned long long saved;
    ~FsFpcrKeep() { fsFpcrSet( saved ); }
};
#else
#define FS_FLUSH_RULE 0
#endif
template <class F>
#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline))
#endif
void fsFlushed(F const &f)
{
#if FS_FLUSH_RULE
    FsFpcrKeep const keep{ fsFpcr() };
    fsFpcrSet( keep.saved | ( 1ull << 24 ) );
#endif
    f();
}
#endif
// a correctly rounded quotient on both devices (the safe math mode's division on the GPU, which the kernel's device
// probe is to confirm). Defined here for both sides at once: never override it on one side only.
#define FS_DIV(a, b) ( (a) / (b) )
#define FS_MAXV 8
// the float nearest 1/6 (0x1.555556p-3), written as its exact decimal (no constant-expression check here: the
// production flags carry -frounding-math, under which a compiler need not fold 1.0f / 6.0f;
// tests/face_sweep_check.cpp checks it at run time)
#define FS_K6 0.16666667163372039794921875f
// The register form of the kernel (FS_REGISTERS, above): FS_PREDICATED 1 there; 0 on the CPU and in a kernel
// without it
#if defined(__METAL_VERSION__) && defined(FS_REGISTERS) && FS_REGISTERS
#define FS_PREDICATED 1
#define FS_UNROLL _Pragma("unroll")
// element v (0..3) of a 4-array, and row v of a [4][3] array at column c, for a run-time v: a select chain
#define FS_AT4(arr, v)   ( (v) == 0 ? (arr)[0] : (v) == 1 ? (arr)[1] : (v) == 2 ? (arr)[2] : (arr)[3] )
#define FS_AT4C(m, v, c) ( (v) == 0 ? (m)[0][c] : (v) == 1 ? (m)[1][c] : (v) == 2 ? (m)[2][c] : (m)[3][c] )
#else
#define FS_PREDICATED 0
#define FS_UNROLL
#define FS_AT4(arr, v)   ( (arr)[v] )
#define FS_AT4C(m, v, c) ( (m)[v][c] )
#endif

// a convex polygon in a plane's (u, w), as x = u, y = w; n vertices
struct FsPoly
{
    float x[FS_MAXV];
    float y[FS_MAXV];
    int n;
};

// an empty polygon (the register form zeroes every slot too: see FS_REGISTERS above)
static inline void fsPolyClear(FS_THREAD FsPoly &p)
{
    p.n = 0;
#if FS_PREDICATED
    FS_UNROLL
    for (int s = 0; s < FS_MAXV; ++s)
    {
        p.x[s] = 0.f;
        p.y[s] = 0.f;
    }
#endif
}

// the vertex (x, y) appended to p, unless p holds FS_MAXV vertices already: a ninth, which only a non-convex rounding
// sliver can make, is dropped, on both devices alike. The register form writes slot p.n by a select in every slot.
static inline void fsPut(FS_THREAD FsPoly &p, float const x, float const y)
{
    if ( p.n >= FS_MAXV ) return;
#if FS_PREDICATED
    FS_UNROLL
    for (int s = 0; s < FS_MAXV; ++s)
    {
        p.x[s] = s == p.n ? x : p.x[s];
        p.y[s] = s == p.n ? y : p.y[s];
    }
#else
    p.x[p.n] = x;
    p.y[p.n] = y;
#endif
    ++p.n;
}

// the grid of a pass: its region's lower corner lo, the spacing h and 1/h (formed once on the host), the cells per
// axis n, the sub-box (subOrigin, subDims: the pass's sub-grid, ps_interpolation.cc's local index) and the wrap
struct FsGrid
{
    float lo[3];
    float h[3];
    float ih[3];
    int n[3];
    int subOrigin[3];
    int subDims[3];
    int periodic;   // 0 / 1
};

// one tetrahedron's canonical inputs: its frames (step 1), the velocity gradient vG[i][j] = d v_j / d x_i, the
// velocity v0 at vertex 0 and its uniform density rho (rho / rho_bar), the last three as the host formed them
struct FsTet
{
    float R[4][3];
    float P[4][3];
    float vG[3][3];
    float v0[3];
    float rho;
};

// the in-plane axes (u, w) of the plane normal to axis a, in increasing order (face_sweep.h planeAxes)
static inline void fsPlaneAxes(int const a, FS_THREAD int &u, FS_THREAD int &w)
{
    u = ( a == 0 ) ? 1 : 0;
    w = ( a == 2 ) ? 1 : 2;
}

// step 1: the frames from the wrapped Eulerian vertices E (float), and the host's vG, v0 and rho
static inline void fsTetSet(FS_THREAD const float E[4][3], FS_THREAD const FsGrid &g, FS_THREAD const float vG[3][3],
                            FS_THREAD const float v0[3], float const rho, FS_THREAD FsTet &t)
{
    FS_NO_CONTRACT
    FS_UNROLL
    for (int v = 0; v < 4; ++v)
    {
        FS_UNROLL
        for (int d = 0; d < 3; ++d)
        {
            t.R[v][d] = E[v][d] - g.lo[d];
            t.P[v][d] = E[v][d] - E[0][d];
        }
    }
    FS_UNROLL
    for (int i = 0; i < 3; ++i)
    {
        FS_UNROLL
        for (int j = 0; j < 3; ++j) t.vG[i][j] = vG[i][j];
    }
    FS_UNROLL
    for (int j = 0; j < 3; ++j) t.v0[j] = v0[j];
    t.rho = rho;
}

// the local index along axis d of the raw grid index 'raw' (wrapped on a periodic axis), or -1 outside the sub-box:
// ps_interpolation.cc psDepositFaces' 'local', in int
static inline int fsLocal(FS_THREAD const FsGrid &g, int const d, int const raw)
{
    int const n = g.n[d];
    int wg = raw;
    if ( g.periodic != 0 )
    {
        wg = raw % n;
        if ( wg < 0 ) wg += n;
    }
    else if ( wg < 0 || wg >= n ) return -1;
    int loc = wg - g.subOrigin[d];
    if ( loc < 0 ) loc += n;
    return ( loc < 0 || loc >= g.subDims[d] ) ? -1 : loc;
}

// the tetrahedron's extent along axis d in the region frame
static inline void fsExtent(FS_THREAD const FsTet &t, int const d, FS_THREAD float &mn, FS_THREAD float &mx)
{
    mn = t.R[0][d];
    mx = t.R[0][d];
    FS_UNROLL
    for (int v = 1; v < 4; ++v)
    {
        mn = t.R[v][d] < mn ? t.R[v][d] : mn;
        mx = t.R[v][d] > mx ? t.R[v][d] : mx;
    }
}

// the grid index of a region-frame coordinate times 1/h, floored; false where it has no int (not finite, or 1e9
// cells or more away: no grid has that many, so that tetrahedron owns nothing here)
static inline bool fsFloorIndex(float const xih, FS_THREAD int &k)
{
    if ( !( xih > -1.0e9f && xih < 1.0e9f ) ) return false;
    k = int( FS_FLOOR( xih ) );
    return true;
}

// step 3: does a region-frame coordinate lie above the plane k h, d = fma(-k, h, r) > 0 (exact sign)? With r the
// tetrahedron's minimum along the axis: true when it owns no section in that plane (ps_interpolation.cc
// psOwnsUpperBoundarySection's canonical test). r = e - lo is formed in float, as fsTetSet forms R.
static inline bool fsAbovePlane(float const e, float const lo, int const k, float const h)
{
    FS_NO_CONTRACT
    float const r = e - lo;
    return FS_FMA( -float( k ), h, r ) > 0.f;
}

// steps 2 and 3: the planes x_a = k h_a the tetrahedron owns a section of (the half-open rule on its region-frame
// extent, min R <= k h < max R, exact): [k0, k0 + count) -- contiguous, so the count and the first are all a kernel
// needs to turn an item's ordinal into its plane. The candidates reach one plane beyond floor(R ih) on either side,
// so the rounded 1/h cannot drop an owned plane; clamped to [0, n] when not periodic (plane n: the upper boundary).
static inline int fsPlanes(FS_THREAD const FsTet &t, FS_THREAD const FsGrid &g, int const a, FS_THREAD int &k0)
{
    FS_NO_CONTRACT
    k0 = 0;
    float mn, mx;
    fsExtent( t, a, mn, mx );
    float const lo = mn * g.ih[a], hi = mx * g.ih[a];
    int c0, c1;
    if ( !fsFloorIndex( lo, c0 ) || !fsFloorIndex( hi, c1 ) ) return 0;
    c0 -= 1;
    c1 += 1;
    if ( g.periodic == 0 )
    {
        c0 = c0 < 0 ? 0 : c0;
        c1 = c1 > g.n[a] ? g.n[a] : c1;
    }
    int kf = c0;
    while ( kf <= c1 && !( FS_FMA( -float( kf ), g.h[a], mn ) <= 0.f ) ) ++kf;   // the first with min R <= k h
    int kl = c1;
    while ( kl >= kf && !( 0.f < FS_FMA( -float( kl ), g.h[a], mx ) ) ) --kl;    // the last with k h < max R
    k0 = kf;
    return kl >= kf ? kl - kf + 1 : 0;
}

// the doubled area and the six-fold first moments of a polygon about its own vertex 0 (steps 5 and 7: Green's theorem
// over the edges (i, i+1 mod n); the edges at vertex 0 add exact zeros)
static inline void fsMoments6(FS_THREAD const FsPoly &p, FS_THREAD float &A2, FS_THREAD float &Mu6,
                              FS_THREAD float &Mw6)
{
    FS_NO_CONTRACT
    A2 = 0.f;
    Mu6 = 0.f;
    Mw6 = 0.f;
    float const x0 = p.x[0], y0 = p.y[0];
    FS_UNROLL
    for (int i = 0; i < FS_MAXV; ++i)
    {
        if ( i < p.n )
        {   // the edge from vertex i to vertex i + 1, or to vertex 0 from the last (i1: a slot index in every case)
            int const i1 = i + 1 < FS_MAXV ? i + 1 : 0;
            bool const last = !( i + 1 < p.n );
            float const xi = p.x[i] - x0, yi = p.y[i] - y0;
            float const xj = ( last ? p.x[0] : p.x[i1] ) - x0, yj = ( last ? p.y[0] : p.y[i1] ) - y0;
            float const xjyi = xj * yi;
            float const cr = FS_FMA( xi, yj, -xjyi );
            A2 = A2 + cr;
            Mu6 = FS_FMA( xi + xj, cr, Mu6 );
            Mw6 = FS_FMA( yi + yj, cr, Mw6 );
        }
    }
}

// step 4: the corner on edge (i, j) of the tetrahedron, the lower vertex first, appended to sec
static inline void fsEdgeCorner(FS_THREAD const FsTet &t, FS_THREAD const float d[4], int const i, int const j,
                                int const u, int const w, FS_THREAD FsPoly &sec)
{
    FS_NO_CONTRACT
    int const p = i < j ? i : j, q = i < j ? j : i;
    float const dp = FS_AT4( d, p ), dq = FS_AT4( d, q );
    float const pu = FS_AT4C( t.P, p, u ), qu = FS_AT4C( t.P, q, u );
    float const pw = FS_AT4C( t.P, p, w ), qw = FS_AT4C( t.P, q, w );
    float const tt = FS_DIV( dp, dp - dq );
    fsPut( sec, FS_FMA( tt, qu - pu, pu ), FS_FMA( tt, qw - pw, pw ) );
}

// steps 3 to 5: the section of the tetrahedron by plane k on axis a, as a polygon in the plane's (u, w) in the
// vertex-0 frame, and the plane's position c in that frame; false (sec.n = 0) when it owns no section of non-zero
// area
static inline bool fsSection(FS_THREAD const FsTet &t, FS_THREAD const FsGrid &g, int const a, int const k,
                             FS_THREAD FsPoly &sec, FS_THREAD float &c)
{
    FS_NO_CONTRACT
    fsPolyClear( sec );
    c = 0.f;
    int u, w;
    fsPlaneAxes( a, u, w );
    float const kf = float( k );
    float d[4];
    FS_UNROLL
    for (int v = 0; v < 4; ++v) d[v] = FS_FMA( -kf, g.h[a], t.R[v][a] );
    float mn = d[0], mx = d[0];
    int nNeg = 0, nPos = 0;
    FS_UNROLL
    for (int v = 0; v < 4; ++v)
    {
        mn = d[v] < mn ? d[v] : mn;
        mx = d[v] > mx ? d[v] : mx;
        nNeg += d[v] < 0.f ? 1 : 0;
        nPos += d[v] > 0.f ? 1 : 0;
    }
    if ( !( mn <= 0.f && 0.f < mx ) ) return false;
    c = FS_FMA( kf, g.h[a], -t.R[0][a] );
    if ( nNeg == 2 && nPos == 2 )
    {   // the 2-2 split: below {p, q}, above {r, s}, both ascending; the cycle (p,r), (p,s), (q,s), (q,r)
        int p = 0, q = 0, r = 0, s = 0;
        bool haveP = false, haveR = false;
        FS_UNROLL
        for (int v = 0; v < 4; ++v)
        {
            if ( d[v] < 0.f ) { if ( !haveP ) { p = v; haveP = true; } else q = v; }
            else { if ( !haveR ) { r = v; haveR = true; } else s = v; }
        }
        fsEdgeCorner( t, d, p, r, u, w, sec );
        fsEdgeCorner( t, d, p, s, u, w, sec );
        fsEdgeCorner( t, d, q, s, u, w, sec );
        fsEdgeCorner( t, d, q, r, u, w, sec );
    }
    else
    {   // three corners (or fewer: then no section): the vertices in the plane, then the crossing edges
        FS_UNROLL
        for (int v = 0; v < 4; ++v)
            if ( d[v] == 0.f ) fsPut( sec, t.P[v][u], t.P[v][w] );
        FS_UNROLL
        for (int i = 0; i < 4; ++i)
        {
            FS_UNROLL
            for (int j = i + 1; j < 4; ++j)
                if ( ( d[i] < 0.f && d[j] > 0.f ) || ( d[i] > 0.f && d[j] < 0.f ) )
                    fsEdgeCorner( t, d, i, j, u, w, sec );
        }
    }
    if ( sec.n < 3 )
    {
        sec.n = 0;
        return false;
    }
    float A2, Mu6, Mw6;
    fsMoments6( sec, A2, Mu6, Mw6 );
    if ( !( A2 > 0.f || A2 < 0.f ) )
    {
        sec.n = 0;
        return false;
    }
    return true;
}

// step 6: the bound of strip (cell) i along an in-plane axis, in the vertex-0 frame: fma(i, h, -r0), r0 = R[0] along
// that axis. The only way a bound is formed: strip i's upper bound is fsBound(i + 1), strip i + 1's lower one.
static inline float fsBound(int const i, float const h, float const r0)
{
    FS_NO_CONTRACT
    return FS_FMA( float( i ), h, -r0 );
}

// one Sutherland-Hodgman clip: the part of 'src' with coordinate k (0: x = u, 1: y = w) >= bound (keepAbove) or
// <= bound
static inline void fsClip(FS_THREAD const FsPoly &src, int const k, float const bound, bool const keepAbove,
                          FS_THREAD FsPoly &out)
{
    FS_NO_CONTRACT
    fsPolyClear( out );
    FS_UNROLL
    for (int i = 0; i < FS_MAXV; ++i)
    {
        if ( i < src.n )
        {   // the edge from vertex i to vertex i + 1, or to vertex 0 from the last (i1: a slot index in every case)
            int const i1 = i + 1 < FS_MAXV ? i + 1 : 0;
            bool const last = !( i + 1 < src.n );
            float const xi = src.x[i], yi = src.y[i];
            float const xj = last ? src.x[0] : src.x[i1], yj = last ? src.y[0] : src.y[i1];
            float const ci = k == 0 ? xi : yi, cj = k == 0 ? xj : yj;
            float const oi = k == 0 ? yi : xi, oj = k == 0 ? yj : xj;
            // (bound - x is exactly -(x - bound): psClip2d's sign * (x - bound) with sign -1)
            float const vi = keepAbove ? ci - bound : bound - ci;
            float const vj = keepAbove ? cj - bound : bound - cj;
            bool const inI = vi >= 0.f, inJ = vj >= 0.f;
            if ( inI ) fsPut( out, xi, yi );
            if ( inI != inJ )
            {
                float const tt = FS_DIV( vi, vi - vj );
                float const o = FS_FMA( tt, oj - oi, oi );
                fsPut( out, k == 0 ? bound : o, k == 0 ? o : bound );
            }
        }
    }
}

// step 6: the slab [lo, hi] of 'src' along coordinate k: clipped at lo, then at hi (a strip from the section along u,
// k = 0; a cell from its strip along w, k = 1)
static inline void fsSlab(FS_THREAD const FsPoly &src, int const k, float const lo, float const hi,
                          FS_THREAD FsPoly &out)
{
    FsPoly tmp;
    fsClip( src, k, lo, true, tmp );
    fsClip( tmp, k, hi, false, out );
}

// the candidate strips (cells) along in-plane axis d: [i0, i1), one beyond floor(R ih) of the tetrahedron's extent on
// either side, clamped to [0, n) when not periodic. Any wider range deposits the same pieces (an empty or one-line
// piece has no area); false when the extent has no index (fsFloorIndex)
static inline bool fsInPlaneRange(FS_THREAD const FsTet &t, FS_THREAD const FsGrid &g, int const d, FS_THREAD int &i0,
                                  FS_THREAD int &i1)
{
    FS_NO_CONTRACT
    float mn, mx;
    fsExtent( t, d, mn, mx );
    float const lo = mn * g.ih[d], hi = mx * g.ih[d];
    i0 = 0;
    i1 = 0;
    if ( !fsFloorIndex( lo, i0 ) || !fsFloorIndex( hi, i1 ) ) { i0 = i1 = 0; return false; }
    i0 -= 1;
    i1 += 2;
    if ( g.periodic == 0 )
    {
        i0 = i0 < 0 ? 0 : i0;
        i1 = i1 > g.n[d] ? g.n[d] : i1;
    }
    return true;
}

// steps 7 and 8: a piece's three shares, S[j] (the integral of the stream's velocity component j over the piece), N
// (its area) and Pn (the normal momentum rho S[a]); false when the piece has no positive area. (u, w): a's in-plane
// axes; c: the plane in the vertex-0 frame (fsSection).
static inline bool fsPieceShares(FS_THREAD const FsTet &t, int const a, int const u, int const w, float const c,
                                 FS_THREAD const FsPoly &piece, FS_THREAD float S[3], FS_THREAD float &N,
                                 FS_THREAD float &Pn)
{
    FS_NO_CONTRACT
    if ( piece.n < 3 ) return false;
    float A2, Mu6, Mw6;
    fsMoments6( piece, A2, Mu6, Mw6 );
    if ( A2 < 0.f )
    {
        A2 = -A2;
        Mu6 = -Mu6;
        Mw6 = -Mw6;
    }
    if ( !( A2 > 0.f ) ) return false;
    float const A = A2 * 0.5f, Mu = Mu6 * FS_K6, Mw = Mw6 * FS_K6;
    float const x0 = piece.x[0], y0 = piece.y[0];
    FS_UNROLL
    for (int j = 0; j < 3; ++j)
    {
        float const vS = FS_FMA( t.vG[w][j], y0, FS_FMA( t.vG[u][j], x0, FS_FMA( t.vG[a][j], c, t.v0[j] ) ) );
        float const Av = A * vS;
        S[j] = FS_FMA( t.vG[w][j], Mw, FS_FMA( t.vG[u][j], Mu, Av ) );
    }
    N = A;
    Pn = t.rho * S[a];
    return true;
}

/* The sweep of one owned plane, the CPU's (fsDepositTet) and the GPU face kernel's (metal/ps_face.metal: one item)
   alike. The sink takes each piece:
     sink.piece(a, below, above, lu, lw, S0, S1, S2, N, P)
   a: the axis; below / above: the local index along a of the cell below the plane (its UPPER face takes the piece,
   side 1) and of the cell above (its LOWER face, side 0), -1 where that cell is not in the sub-box; lu / lw: the
   piece's local indices along u and w (both >= 0). */
template <class Sink>
inline void fsSweepPlane(FS_THREAD const FsTet &t, FS_THREAD const FsGrid &g, int const a, int const k,
                         FS_THREAD Sink &sink)
{
    int const below = fsLocal( g, a, k - 1 ), above = fsLocal( g, a, k );
    if ( below < 0 && above < 0 ) return;
    FsPoly sec;
    float c;
    if ( !fsSection( t, g, a, k, sec, c ) ) return;
    int u, w;
    fsPlaneAxes( a, u, w );
    int i0, i1, j0, j1;
    if ( !fsInPlaneRange( t, g, u, i0, i1 ) || !fsInPlaneRange( t, g, w, j0, j1 ) ) return;
    float uMin = sec.x[0], uMax = sec.x[0];
    FS_UNROLL
    for (int m = 1; m < FS_MAXV; ++m)
        if ( m < sec.n )
        {
            uMin = sec.x[m] < uMin ? sec.x[m] : uMin;
            uMax = sec.x[m] > uMax ? sec.x[m] : uMax;
        }
    for (int i = i0; i < i1; ++i)
    {
        float const blo = fsBound( i, g.h[u], t.R[0][u] ), bhi = fsBound( i + 1, g.h[u], t.R[0][u] );
        if ( uMax <= blo ) break;                    // the strips only move further away
        if ( uMin >= bhi ) continue;
        int const lu = fsLocal( g, u, i );
        if ( lu < 0 ) continue;
        FsPoly strip;
        fsSlab( sec, 0, blo, bhi, strip );
        if ( strip.n < 3 ) continue;
        float wMin = strip.y[0], wMax = strip.y[0];
        FS_UNROLL
        for (int m = 1; m < FS_MAXV; ++m)
            if ( m < strip.n )
            {
                wMin = strip.y[m] < wMin ? strip.y[m] : wMin;
                wMax = strip.y[m] > wMax ? strip.y[m] : wMax;
            }
        for (int j = j0; j < j1; ++j)
        {
            float const clo = fsBound( j, g.h[w], t.R[0][w] ), chi = fsBound( j + 1, g.h[w], t.R[0][w] );
            if ( wMax <= clo ) break;
            if ( wMin >= chi ) continue;
            int const lw = fsLocal( g, w, j );
            if ( lw < 0 ) continue;
            FsPoly cell;
            fsSlab( strip, 1, clo, chi, cell );
            float S[3], N, Pn;
            if ( !fsPieceShares( t, a, u, w, c, cell, S, N, Pn ) ) continue;
            sink.piece( a, below, above, lu, lw, S[0], S[1], S[2], N, Pn );
        }
    }
}

#ifndef __METAL_VERSION__
// every owned plane of one tetrahedron, axis by axis (psDepositFaces under DTFE_FACE_F32 / DTFE_GPU_FACES)
template <class Sink>
inline void fsDepositTet(FsTet const &t, FsGrid const &g, Sink &sink)
{
    for (int a = 0; a < 3; ++a)
    {
        int k0;
        int const n = fsPlanes( t, g, a, k0 );
        for (int k = k0; k < k0 + n; ++k) fsSweepPlane( t, g, a, k, sink );
    }
}
#endif

#endif
