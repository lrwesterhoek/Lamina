//  PS-DTFE mass-conserving deposit -- Metal compute kernel.
//
//  One GPU thread per tetrahedron. Mirrors the CPU deposit in ps_interpolation.cc:
//  each Lagrangian flow element carries a constant mass m = rho_bar * V_lag; the thread
//  splits m equally among the Eulerian sub-sample points inside the simplex and scatters
//  the shares into the (mass) grid with atomic adds -- an exact fixed-point sum, see
//  DepositParams.massFixed; every moment grid too, see DepositParams.fxMask. The words are 64-bit words in the units
//  of the pass, or under DTFE_PS_RUN_UNITS=1 the run-wide 128-bit limbs of metal/fixed_words.metal.inc in the run's
//  units (DepositParams.fxWords 2; the Makefile prepends that file to this source). Total deposited == m exactly,
//  so the field is mass-conserving at any nSub and for any V_eul (no caustic spikes).
//
//  Which samples a tetrahedron contains is decided EXACTLY, i.e. identically to the CPU
//  deposit's test (src/CGAL_triangulation/ps_exact_inside.h): the host hands every tet over in
//  the canonical frame of that test (TET_STRIDE record below), and the kernel classifies each
//  sample in float with a RIGOROUS error bound (classifySample). Outside the bound the float
//  decision provably equals the exact one. A tet with any sample inside the bound -- a sample
//  within float rounding of a face, a few per million -- is not deposited here at all: the
//  kernel DEFERS it by negating its mass, and the host deposits it with the CPU's exact test.

#include <metal_stdlib>
using namespace metal;

// The parameter block (buffer 8): PSDepositParams in src/CGAL_triangulation/ps_kernel_params.h, the hosts' own
// struct, which the Makefile prepends to this source after exact_clip.metal.inc and fixed_words.metal.inc, followed
// by metal/ps_common.metal.inc (metal/validate_deposit.cpp and metal/build_prototype.sh do the same) -- one
// definition, so the kernels and the hosts cannot drift apart. Its fields are documented there.
typedef PSDepositParams DepositParams;

// the moment grids' indices (ps_deposit_params.h PSFxGrid -- keep in step) and the buffer slots of their words
constant uint FX_MOM = 0, FX_M2 = 1, FX_GRAD = 2, FX_MOMW = 3, FX_SV = 4, FX_DV = 5, FX_DW = 6, FX_SCAL = 7, FX_SG = 8;
constant uint FX_SUMW_BIT = 1u << 16;
// words of MOM, M2, GRAD, MOMW, SV, DV, SCAL, SG at buffers 22..29 (DW has none), the items' sumW words at 30

static inline float det3(thread const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Product of the row lengths (Hadamard's bound on |det|).
static inline float rowNormProduct3(thread const float A[3][3]) {
    float p = 1.0f;
    for (int i=0; i<3; ++i) p *= sqrt(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
    return p;
}

// Inverse of 3x3; false if |det| <= 1e-6 x the product of the row lengths -- the CPU
// matrixInverse() sentinel (math_functions.h, RELATIVE and unit-free), so a collapsed simplex
// is dropped rather than bounding-box filled while a merely SMALL one is kept.
static inline bool inverse3(thread const float A[3][3], thread float inv[3][3]) {
    float d = det3(A);
    if (!(fabs(d) > 1.0e-6f * rowNormProduct3(A))) return false;
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

static inline int wrapIdx(int g, int n) { return ((g % n) + n) % n; }

// --ps-linear-deposit sample weight: linear density at offset rel from vertex 0, clamped >= 0
// (a sample on a face can graze negative in float).
static inline float linearWeight(thread const float rel[3], float d0, thread const float dG[3]) {
    float w = d0 + dG[0]*rel[0] + dG[1]*rel[1] + dG[2]*rel[2];
    return w < 0.0f ? 0.0f : w;
}

// ---- the per-tetrahedron record (host: ps_interpolation.cc, harness: validate_deposit.cpp) ----
// TET_STRIDE floats per tet in the 'verts' buffer, in the CANONICAL frame of the exact inside
// test (ps_exact_inside.h: every periodic image of a particle mapped to one position, a float in
// the single build and a double in the double build):
//   [0..2]  ic0 -- vertex 0's grid-cell index in the canonical index frame (int bits)
//   [3..5]  f0  -- vertex 0's offset inside that cell, in cell units, in [0,1)
//   [6..14] E   -- the edges vertex_k - vertex_0 (k = 1..3, one row each), Mpc; differences of
//                  the canonical positions (exact for floats), rounded once to float
//   [15]    the centroid-fallback cell as a flat sub-grid index, computed by the host exactly as
//           the CPU fallback does (uint bits; 0xFFFFFFFF = outside this partition: dropped)
// Positions relative to vertex 0 are then formed from SMALL numbers -- sample offset in cells
// (g - ic0) + frac - f0 -- which is what makes a tight rounding bound possible.
constant int TET_STRIDE = 16;
constant float U_F = 5.9604645e-8f;   // 2^-24, the float unit roundoff

// Band-certified classification of one sample, at offset r (Mpc) from vertex 0. With C the
// cofactor matrix of the edge matrix E (rows = edges), the barycentric numerators are
// n_a = sum_i C[a][i] r_i (vertex a+1, Cramer's rule) and n_z = det - sum_a n_a (vertex 0); the
// sample is inside iff all four have the sign of det. B[] bound |computed - exact| for each
// (see the per-tet setup), so a numerator beyond its bound has its exact sign. Returns +1
// inside, -1 outside (both exact), 0 when some numerator lies within its bound -- then only
// the exact predicate can decide, and the caller defers the tetrahedron to the host.
// Two stages: B[] holds the per-tet bounds, valid for the worst sample of the window; a sample
// they cannot decide gets bounds built from ITS OWN offset t (cells) -- the same analysis, with
// |t_i| in place of the window maximum T_i -- which are several times tighter for the samples
// that matter (those near the tet, well inside a window padded by a cell on each side).
static inline int classifySample(thread const float C[3][3], thread const float M[3][3], float det, float sg,
                                 thread const float B[4], float Bdet, thread const float r[3],
                                 thread const float t[3], constant DepositParams& P)
{
    float n[4];
    n[0] = C[0][0]*r[0] + C[0][1]*r[1] + C[0][2]*r[2];
    n[1] = C[1][0]*r[0] + C[1][1]*r[1] + C[1][2]*r[2];
    n[2] = C[2][0]*r[0] + C[2][1]*r[1] + C[2][2]*r[2];
    n[3] = det - (n[0] + n[1] + n[2]);
    bool sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -B[k]) return -1;      // exactly outside, whatever the other three are
        if (x <= B[k]) sure = false;
    }
    if (sure) return 1;

    // stage 2: this sample's own bounds (7u|r_i| + rho_i = dx_i (u (6 + 12|t_i|) + 1e-9))
    float w[3];
    for (int i = 0; i < 3; ++i) w[i] = P.dx[i] * (U_F * (6.0f + 12.0f*fabs(t[i])) + 1.0e-9f);
    float Bs[4];
    float nMag = 0.0f;
    for (int a = 0; a < 3; ++a) {
        Bs[a] = 2.0f * (M[a][0]*w[0] + M[a][1]*w[1] + M[a][2]*w[2]);
        nMag += M[a][0]*fabs(r[0]) + M[a][1]*fabs(r[1]) + M[a][2]*fabs(r[2]);
    }
    Bs[3] = Bdet + Bs[0] + Bs[1] + Bs[2] + 2.0f * 3.0f * U_F * (fabs(det) + nMag);
    sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -Bs[k]) return -1;
        if (x <= Bs[k]) sure = false;
    }
    return sure ? 1 : 0;
}

// The exact clip helpers (tet_planes, box_outside_tet, sweep_moments) live in
// metal/exact_clip.metal.inc, which the Makefile prepends to this source when it embeds it
// (dtfe_deposit.metal shares them).

// ------------------------------------------------------------------------------------------------
// depositFields: the deposit. Each interior sample receives the mass share and, per the field
// flags, also the mass-weighted velocity moment sum(w v_j), second moment
// sum(w v_i v_j) (upper triangle xx,xy,xz,yy,yz,zz), velocity-gradient moment sum(w dv_i/dx_j)
// (layout j*3+i, matching the CPU Pvector), and increments the per-sample stream counter.
// The host normalizes: vbar = mom/W, sigma_ij = m2/W - vbar_i vbar_j, density = mass/cellVol,
// streams = count/nSub^3 -- identical to ps_interpolation.cc.
// ------------------------------------------------------------------------------------------------
struct FieldGrids {
    device atomic_float* mass;
    device atomic_float* mom;     // nCell*3   [flat*3+j]
    device atomic_float* m2;      // nCell*6   [flat*6+c]
    device atomic_float* grad;    // nCell*9   [flat*9+j*3+i]
    device atomic_uint*  streams;
    device atomic_float* momw;    // nCell     moment-weight normalizer (fVolW; else dummy)
    device atomic_uint*  caustic; // nCell     orientation bits, atomic OR (fCaustic; else dummy)
    device atomic_float* sv;      // nCell     exact multiplicity sum(V_int)/V_cell (fExact; else dummy)
    device atomic_float* dispvel; // nCell*3   sum(m_s v_s), the dispersion's own MASS-weighted mean (fVolW+fDisp; else dummy)
    device atomic_float* dispw;   // nCell     sum(m_s), its normalizer (fVolW+fDisp; else dummy)
    device atomic_float* scal;    // nCell     scalar moment sum(wm s) (fScal; else dummy)
    device atomic_float* sgrad;   // nCell*3   scalar-gradient moment sum(wm ds/dx_i) (fSGrad; else dummy)
    device atomic_uint*  massFx;  // nCell*2   the mass as a 64-bit fixed-point sum [flat*2] low, [flat*2+1] high (massFixed; else dummy)
                                  //           (fxWords 2: nCell*4, the four limbs [flat*4 .. flat*4+3])
    float massScale;              // its unit: 1/massScale mass
    bool  massFixed;
    // the moment grids' fixed-point views -- the same buffers as the float pointers above, read as two
    // uint words per component (four limbs under fxWords 2) -- for the grids whose fxMask bit is set
    // (DepositParams.fxMask)
    device atomic_uint*  momX;
    device atomic_uint*  m2X;
    device atomic_uint*  gradX;
    device atomic_uint*  momwX;
    device atomic_uint*  svX;
    device atomic_uint*  dvX;
    device atomic_uint*  scalX;
    device atomic_uint*  sgX;
    uint  fxMask;
    constant float* fxScale;      // DepositParams.fxScale
    // the non-finite guard's counters, one word per slot (FX_DROP_*): depositFields' own buffer 30,
    // the exact items' tail of the sumW buffer (ExactItemParams.badAt); the host zeroes them before each command
    // buffer and adds them up after one that succeeded
    device atomic_uint*  bad;
    // DepositParams.fxWords and fxCapBits: 2 = the mass and fxMask grids in the run-wide 128-bit limbs, below the
    // share cap capBits (addShare, addLimbsN); 0 = the 64-bit words below. The adds read the size from the
    // compile-time PS_FX_WORDS (FX_WIDE, below), which the host keeps equal to fxWords
    uint  fxWords;
    uint  capBits;
};

// The non-finite guard of a fixed-point share (fixedShareOk) and the 64-bit words' add (addFixedQ): in
// metal/ps_common.metal.inc, which the Makefile prepends to this source and to the face kernel's
// (metal/ps_face.metal). The guard's counters, one word per slot:
constant uint FX_DROP_MASS = 10u, FX_DROP_SUMW = 11u;   // ps_deposit_params.h FixedShareSlot (keep in step)

// The word size is a compile-time constant: the host compiles one library per size, with "#define PS_FX_WORDS 0"
// (the 64-bit words) or "#define PS_FX_WORDS 2" (the run-wide 128-bit limbs) in front of this source
// (ps_metal_host.cc pipes(); DepositParams.fxWords holds the same number). So each library carries one add path,
// and the 64-bit one compiles the adds as they stood before the wide words.
#if !defined(PS_FX_WORDS) || (PS_FX_WORDS != 0 && PS_FX_WORDS != 2)
#error "PS_FX_WORDS must be 0 or 2 (the host compiles one library per word size)"
#endif
#define FX_WIDE (PS_FX_WORDS == 2)

// One share v of a moment grid (component index 'at' = flat * comps + c): the fixed-point words when its fxMask
// bit is set -- rint(v * scale), scale = 2^k so the float product is exact, the integer the host adds for the
// same float (ps_deposit_params.h psFixedShare) -- else a float atomic. fxWords 2: the four limbs in place of the
// two words (fixed_words.metal.inc addShare: below 2^63 units the same integer, as one signed 64-bit add), a share
// at or above the cap counted.
static inline void addMoment(device atomic_float* f, device atomic_uint* x, thread const FieldGrids& G, uint g,
                             ulong at, float v)
{
    if ((G.fxMask >> g) & 1u) {
        if (FX_WIDE) {
            if (!addShare(x, at, v, G.fxScale[g], G.capBits))
                atomic_fetch_add_explicit(&G.bad[g], 1u, memory_order_relaxed);
            return;
        }
        float const p = v * G.fxScale[g];
        if (fixedShareOk(p)) addFixedQ(x, at, as_type<ulong>(long(rint(p))));
        else atomic_fetch_add_explicit(&G.bad[g], 1u, memory_order_relaxed);   // (the guard above)
    }
    else atomic_fetch_add_explicit(&f[at], v, memory_order_relaxed);
}

// A mass share into the mass grid: the exact fixed-point sum (massFixed), else a float atomic.
// fxWords 2: the run-wide 128-bit limbs (addShare), below the share cap. The 64-bit words: two's-complement words,
// the low word's carry read off the value its atomic returns (the add wrapped
// iff the result is below it), and the high words add modulo 2^32, so once every thread is done the two
// words hold the exact integer total, whatever the interleaving. The host keeps the partition's whole
// mass below 2^62 units there, so no share and no cell total overflows.
static inline void depositMass(thread const FieldGrids& G, uint flat, float w)
{
    if (!G.massFixed) {
        atomic_fetch_add_explicit(&G.mass[flat], w, memory_order_relaxed);
        return;
    }
    if (FX_WIDE) {
        if (!addShare(G.massFx, ulong(flat), w, G.massScale, G.capBits))
            atomic_fetch_add_explicit(&G.bad[FX_DROP_MASS], 1u, memory_order_relaxed);
        return;
    }
    // (64-bit cell offsets inside addFixedQ: a sub-grid may hold up to 2^32 cells)
    float const p = w * G.massScale;
    if (fixedShareOk(p)) addFixedQ(G.massFx, ulong(flat), as_type<ulong>(long(rint(p))));
    else atomic_fetch_add_explicit(&G.bad[FX_DROP_MASS], 1u, memory_order_relaxed);   // (the non-finite guard)
}

// n equal shares v of moment grid g at component 'at', as ONE add of n * q (q = the integer addMoment adds for v):
// the fixed-point grids only (fxMask bit set). Integer adds are exact and the words add modulo 2^64 whatever the
// interleaving, so the words end where n adds of q leave them; a share the non-finite guard drops counts n times.
// fxWords 2: n times the limbs (addLimbsN; the cap holds the n <= nSub^3 equal shares of one cell's samples).
static inline void addMomentN(device atomic_uint* x, thread const FieldGrids& G, uint g, ulong at, float v, uint n)
{
    if (FX_WIDE) {
        uint a[4]; bool neg = false;
        if (shareLimbs(v, G.fxScale[g], G.capBits, a, neg)) addLimbsN(x, at, a, neg, n);
        else atomic_fetch_add_explicit(&G.bad[g], n, memory_order_relaxed);
        return;
    }
    float const p = v * G.fxScale[g];
    if (fixedShareOk(p)) addFixedQ(x, at, as_type<ulong>(long(rint(p))) * ulong(n));
    else atomic_fetch_add_explicit(&G.bad[g], n, memory_order_relaxed);   // (the guard above)
}

// n equal mass shares w as one add of n * q (the fixed-point mass grid only: massFixed), as addMomentN
static inline void depositMassN(thread const FieldGrids& G, uint flat, float w, uint n)
{
    if (FX_WIDE) {
        uint a[4]; bool neg = false;
        if (shareLimbs(w, G.massScale, G.capBits, a, neg)) addLimbsN(G.massFx, ulong(flat), a, neg, n);
        else atomic_fetch_add_explicit(&G.bad[FX_DROP_MASS], n, memory_order_relaxed);
        return;
    }
    float const p = w * G.massScale;
    if (fixedShareOk(p)) addFixedQ(G.massFx, ulong(flat), as_type<ulong>(long(rint(p))) * ulong(n));
    else atomic_fetch_add_explicit(&G.bad[FX_DROP_MASS], n, memory_order_relaxed);   // (the non-finite guard)
}

// w = mass share (mass grid), wm = moment weight (== w by default; the V_eul share under
// fVolW), obits = the tet's orientation bits for the caustic OR (fCaustic), svShare = this
// sample's exact-multiplicity share (fExact fall-through path only).
// batched: the sampled deposit of a tet without --ps-linear-deposit, whose shares w, wm and svShare are the same for
// every sample: the parts that do not depend on the sample's position and go into an exact integer grid -- the
// fixed-point mass, gradient, scalar-gradient, moment-weight and multiplicity shares, the stream count and the
// caustic OR -- are left to depositSampleBatch, once per (tet, cell) for the cell's n samples.
// unresolved: this is a centroid-fallback deposit (sets bit 31, see the end).
// countStream: whether this deposit adds to the sample-count multiplicity '.streams'. True for
// every interior SAMPLE; false for the centroid fallback of a tet that contains no sample, which
// moves mass but is no sample of the stream field (counting it +1 would inflate '.streams' by one
// whole sample per sub-resolution tetrahedron -- up to +839 in collapsed cells). Under fExact
// the counter is the raw tet-touch count instead, so the fallback counts there.
static inline void depositSample(thread const FieldGrids& G, uint flat,
                                 thread const float rel[3], float w, float wm, uint obits,
                                 thread const float u0[3], thread const float vG[3][3],
                                 float s0, thread const float sG[3],
                                 bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                 bool fExact, float svShare, bool countStream, bool unresolved,
                                 bool fScal, bool fSGrad, bool batched)
{
    // Moment-grid offsets are 64-bit BECAUSE flat*9 WOULD overflow a 32-bit uint above ~782^3
    // and silently scatter atomics to low addresses.
    // The surviving 32-bit quantity is 'flat' ITSELF (built by the callers), which caps a
    // sub-grid at 2^32 cells (~1625^3); ps_interpolation.cc guards that with a CPU fallback.
    ulong base = ulong(flat);
    if (!(batched && G.massFixed)) depositMass(G, flat, w);
    if (fVel || fDisp) {
        float vv[3];
        for (int j=0; j<3; ++j) {
            vv[j]=u0[j];
            for (int i=0;i<3;++i) vv[j]+=vG[i][j]*rel[i];
            if (fVel)
                addMoment(G.mom, G.momX, G, FX_MOM, base*3ul+ulong(j), vv[j]*wm);
        }
        if (fDisp) {
            // sigma_ij is a moment of f -> always MASS-weighted (w), even when the velocity
            // moments carry volume shares (wm) under fVolW.
            ulong c=0;
            for (int i=0;i<3;++i)
                for (int j=i;j<3;++j)
                    addMoment(G.m2, G.m2X, G, FX_M2, base*6ul + c++, w*vv[i]*vv[j]);
            if (fVolW) {   // its own mass-weighted mean + normalizer (the mom grid is volume-weighted)
                for (int j=0;j<3;++j)
                    addMoment(G.dispvel, G.dvX, G, FX_DV, base*3ul+ulong(j), vv[j]*w);
                if (((G.fxMask >> FX_DW) & 1u) == 0u)   // (else the mass grid is it)
                    atomic_fetch_add_explicit(&G.dispw[flat], w, memory_order_relaxed);
            }
        }
    }
    if (fGrad && !(batched && ((G.fxMask >> FX_GRAD) & 1u)))
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                addMoment(G.grad, G.gradX, G, FX_GRAD, base*9ul + ulong(j*3+i), vG[i][j]*wm);
    if (fScal) {   // the linear scalar profile at the sample, moment-weighted like the velocity
        float sv = s0;
        for (int i=0;i<3;++i) sv += sG[i]*rel[i];
        addMoment(G.scal, G.scalX, G, FX_SCAL, base, sv*wm);
    }
    if (fSGrad && !(batched && ((G.fxMask >> FX_SG) & 1u)))
        for (int i=0;i<3;++i)
            addMoment(G.sgrad, G.sgX, G, FX_SG, base*3ul + ulong(i), sG[i]*wm);
    if (fVolW && !(batched && ((G.fxMask >> FX_MOMW) & 1u)))
        addMoment(G.momw, G.momwX, G, FX_MOMW, base, wm);
    if (fCaustic && !batched)
        atomic_fetch_or_explicit(&G.caustic[flat], obits, memory_order_relaxed);
    if (fExact && !(batched && ((G.fxMask >> FX_SV) & 1u)))
        addMoment(G.sv, G.svX, G, FX_SV, base, svShare);
    if (countStream && !batched)
        atomic_fetch_add_explicit(&G.streams[flat], 1u, memory_order_relaxed);
    // '.hidden_streams' bit 2: mass from a tet none of whose samples lies in this cell (the centroid
    // fallback). Bit 31 of the same counter -- the sample count never reaches 2^31 -- so no
    // extra buffer is bound; the host splits the two (ps_interpolation.cc).
    if (unresolved)
        atomic_fetch_or_explicit(&G.streams[flat], 0x80000000u, memory_order_relaxed);
}

// The parts of n batched samples' depositSample that do not depend on the sample's position (see there), for the n
// interior samples one tet has in cell 'flat': one add of n * q into each exact integer grid (addMomentN), n
// stream counts and one caustic OR (OR is idempotent). The fixed-point words, the counts and the bits end exactly
// where n per-sample atomics leave them: a fewer-atomics form of the same deposit, byte for byte.
static inline void depositSampleBatch(thread const FieldGrids& G, uint flat, uint n, float w, float wm, uint obits,
                                      thread const float vG[3][3], thread const float sG[3],
                                      bool fGrad, bool fVolW, bool fCaustic, bool fExact, float svShare, bool fSGrad)
{
    ulong base = ulong(flat);
    if (G.massFixed) depositMassN(G, flat, w, n);
    if (fGrad && ((G.fxMask >> FX_GRAD) & 1u))
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                addMomentN(G.gradX, G, FX_GRAD, base*9ul + ulong(j*3+i), vG[i][j]*wm, n);
    if (fSGrad && ((G.fxMask >> FX_SG) & 1u))
        for (int i=0;i<3;++i)
            addMomentN(G.sgX, G, FX_SG, base*3ul + ulong(i), sG[i]*wm, n);
    if (fVolW && ((G.fxMask >> FX_MOMW) & 1u))
        addMomentN(G.momwX, G, FX_MOMW, base, wm, n);
    if (fCaustic)
        atomic_fetch_or_explicit(&G.caustic[flat], obits, memory_order_relaxed);
    if (fExact && ((G.fxMask >> FX_SV) & 1u))
        addMomentN(G.svX, G, FX_SV, base, svShare, n);
    atomic_fetch_add_explicit(&G.streams[flat], n, memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------
// Exact deposit (--exact), one window cell at a time. Shared by depositExactItems (the
// production path: many threads per large tet, below) and depositFields (one thread per tet: the
// few tets the items pass leaves over).
// ------------------------------------------------------------------------------------------------

// A RAW window index triple -> its flat sub-grid index: the periodic wrap, then the partition
// sub-box, which may itself wrap a periodic axis (see ps_interpolation.cc). 0 when the cell lies
// outside the grid, 1 when inside the grid but outside the sub-box (under windowMode the SAMPLED
// deposit still counts such a cell's samples for the tet's normalization), 2 when inside the sub-box.
static inline int window_flat(thread const int raw[3], constant DepositParams& P, thread uint& flat) {
    int l[3];
    bool inSub = true;
    for (int dd=0; dd<3; ++dd) {
        int const wg = P.periodic ? wrapIdx(raw[dd], P.nGrid[dd]) : raw[dd];
        if (wg < 0 || wg >= P.nGrid[dd]) return 0;
        l[dd] = wg - P.subOrigin[dd];
        if (l[dd] < 0) l[dd] += P.nGrid[dd];
        if (l[dd] < 0 || l[dd] >= P.subDims[dd]) inSub = false;
    }
    if (!inSub) return 1;
    flat = (uint(l[0])*uint(P.subDims[1]) + uint(l[1]))*uint(P.subDims[2]) + uint(l[2]);
    return 2;
}

// --grid-window, exact deposit: restrict a tet's raw span [lo, hi) along one axis to the hull of the
// window's periodic images it meets (gpu_host.h psWindowHull, which the host's item bound uses --
// keep in sync). False when the span misses the window.
static inline bool window_hull(thread int& lo, thread int& hi, int n, bool periodic, int o, int m) {
    if (hi <= lo || m <= 0) return false;
    if (!periodic) { lo = max(lo, 0); hi = min(hi, n); if (hi <= lo) return false; }
    bool any = false;
    int hlo = 0, hhi = 0;
    int const kMax = periodic ? 2 : 0;
    for (int k = -kMax; k <= kMax; ++k) {
        int const a = max(lo, o + k*n), b = min(hi, o + m + k*n);
        if (b <= a) continue;
        if (!any) { hlo = a; hhi = b; any = true; }
        else { hlo = min(hlo, a); hhi = max(hhi, b); }
    }
    if (!any) return false;
    lo = hlo; hi = hhi;
    return true;
}

struct ExactTet {
    float3 t1, t2, t3;       // edges from vertex 0, positively oriented (folded tet: 1 and 2 swapped)
    TetPlanes faces;
    float3 hcell;            // half a cell
    int    ic0[3];
    float  f0[3];
    float  u0[3], vG[3][3];  // the linear velocity (zero when no velocity grid is requested)
    float  vc[3], vh[3];     // the centre and half the range of each velocity component over the 4 vertices: every
                             // piece's mean velocity and velocity covariance must lie within them (exact_cell_deposit)
    float  s0, sG[3];        // the linear scalar (fScal / fSGrad; else zero)
    float  d0, dG[3];        // the linear density (fLinear)
    uint   obits;
    float  invCellVol;
    // a tet that reaches beyond a non-periodic grid is normalized by its WHOLE weight wFull (not
    // the sum over its in-grid pieces), so the grid keeps exactly the share of its mass inside it
    bool   straddle;
    float  wFull;
    // the unit of this tet's weight total in the items' fixed-point sumW (FX_SUMW_BIT): 2^k with the
    // bound V (|the linear density| at most dMax under fLinear: V dMax) at [2^58, 2^59) units, so the sum of its
    // pieces' weights -- which is that total to float rounding -- stays far below 2^63 (sumw_scale)
    float  wScale;
};

// ExactTet.wScale of a tet whose weight bound is wb0 (|det|/6), times the largest |vertex density| under fLinear:
// wb in [2^(e-1), 2^e) puts it at [2^58, 2^59) units. Under fxWords 2 every cell's weight becomes its integer in this
// unit before the tet's total is summed (both exact kernels), so the total is the same however the cells are cut
// into items; each weight is >= 0 and they sum to at most wb (to float rounding), so the total stays below 2^60.
static inline float sumw_scale(float wb0, bool fLinear, device const float* dens, ulong tet)
{
    float wb = wb0;
    if (fLinear) {
        float dMax = 0.0f;
        for (int v=0; v<4; ++v) dMax = max(dMax, fabs(dens[tet*4ul+ulong(v)]));
        wb *= dMax;
    }
    int e = 0;
    (void)frexp(wb, e);                        // wb < 2^e
    return (wb > 0.0f && isfinite(wb)) ? ldexp(1.0f, clamp(59 - e, -126, 126)) : 1.0f;
}

// The float nearest to hi 2^32 + lo (ties to even), rounded ONCE: the two words shifted so the top set bit is bit 31
// of one word, every bit below that word folded into its lowest bit (a sticky bit, 7 places below float's 24), and
// the power of two put back, which is exact. float(hi) 2^32 + float(lo) rounds twice and drops lo wherever hi needs
// more than 24 bits: a weight total at [2^58, 2^59) units (sumw_scale) then reads low by 2^31 units on average, 5e-9
// of it, and every share m / sumW of the tet high by as much -- on every tet, so the mass grid's sum with it.
static inline float words_to_float(uint hi, uint lo)
{
    if (hi == 0u) return float(lo);
    uint const s = clz(hi);                    // 0..31: hi != 0
    uint top = hi, rest = lo;
    if (s != 0u) { top = (hi << s) | (lo >> (32u - s)); rest = lo << s; }
    if (rest != 0u) top |= 1u;
    return ldexp(float(top), 32 - int(s));
}

// The exact dispersion's float32 limit. Each piece adds the variance of the tet's linear velocity over it,
// G^T Cov G, with Cov from the piece's second position moments; those carry an absolute float error of ~eps ell^2
// (ell the piece's extent, at most the cell's diagonal: the sweep works about origins inside the piece's overlap
// box), and a thin tet with a steep velocity gradient multiplies it by |G|^2 ~ (dv / thickness)^2. On TNG100-3-Dark
// z = 0 at 512^3 that put spurious dispersions of up to 6e10 (km/s)^2 into ~1% of the cells (the CPU, in double: at
// most 1.75e6). Per velocity component j, kappa_j^2 = |G_.j|^2 ell^2 / range_j^2 (range_j over the 4 vertices) sets
// the relative error of S_jj, ~eps kappa_j^2; above EX_DISP_KAPPA2 for any component the tet goes to the CPU's
// double-precision exact deposit (deferred, as a float-degenerate tet is). The tet's own extent L enters only through
// the edge interpolation (an error ~eps L / thickness, linear): EX_DISP_KAPPA2_L guards that on |G_.j|^2 L^2. Below
// both, the clamp in exact_cell_deposit bounds the magnitude of what is left.
#ifndef EX_DISP_KAPPA2
#define EX_DISP_KAPPA2 1.0e4f     // relative error ~6e-4
#endif
#ifndef EX_DISP_KAPPA2_L
#define EX_DISP_KAPPA2_L 1.0e10f  // relative error ~eps * 1e5 = 6e-3 from the edge interpolation
#endif
// the range of the tet's linear velocity over its 4 vertices, per component: its centre vc and half-width vh
static inline void exact_vel_range(thread const float Ax[3][3], thread const float vG[3][3], thread const float u0[3],
                                   thread float vc[3], thread float vh[3])
{
    for (int j=0; j<3; ++j) {
        float lo = 0.0f, hi = 0.0f;   // vertex 0's offset is 0
        for (int e=0; e<3; ++e) {
            float const dv = Ax[e][0]*vG[0][j] + Ax[e][1]*vG[1][j] + Ax[e][2]*vG[2][j];
            lo = min(lo, dv); hi = max(hi, dv);
        }
        vc[j] = u0[j] + 0.5f*(lo + hi);
        vh[j] = 0.5f*(hi - lo);
    }
}
static inline bool exact_disp_illcond(thread const float Ax[3][3], thread const float vG[3][3], thread const float vh[3],
                                      float3 dx)
{
    float L = 0.0f;
    for (int d=0; d<3; ++d) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = min(lo, Ax[e][d]); hi = max(hi, Ax[e][d]); }
        L = max(L, hi - lo);
    }
    float const ell = min(L, length(dx));
    for (int j=0; j<3; ++j) {   // (a component with no range has no gradient either: 0 > 0 is false)
        float const g2 = vG[0][j]*vG[0][j] + vG[1][j]*vG[1][j] + vG[2][j]*vG[2][j];
        float const r2 = 4.0f*vh[j]*vh[j];
        if (g2*ell*ell > EX_DISP_KAPPA2 * r2 || g2*L*L > EX_DISP_KAPPA2_L * r2) return true;
    }
    return false;
}

// The tet's whole deposit weight: its volume, or under fLinear the integral of the linear density
// d0 + dG.x over it = V (d0 + dG.centroid), the centroid (t1+t2+t3)/4 in the vertex-0 frame.
static inline float exact_full_weight(float vol, bool fLinear, float d0, thread const float dG[3],
                                      float3 t1, float3 t2, float3 t3)
{
    if (!fLinear) return vol;
    float3 const c = 0.25f * (t1 + t2 + t3);
    return vol * (d0 + dG[0]*c.x + dG[1]*c.y + dG[2]*c.z);
}

// Clips the tet against the window cell 'raw'. False when they do not overlap; otherwise mo holds
// the piece's moments (volume and first moments only unless 'full': pass 0 discards the rest) and
// w its deposit weight -- V_int, or the exact linear-profile integral under fLinear (clamped at 0).
static inline bool exact_cell_clip(thread const ExactTet& E, thread const int raw[3],
                                   constant DepositParams& P, bool full, bool fLinear,
                                   thread float mo[10], thread float& w, thread float3& cc)
{
    // a cell wholly outside a face plane cannot hold any of the tet
    float3 const ccen = float3((float(raw[0] - E.ic0[0]) - E.f0[0] + 0.5f) * P.dx[0],
                               (float(raw[1] - E.ic0[1]) - E.f0[1] + 0.5f) * P.dx[1],
                               (float(raw[2] - E.ic0[2]) - E.f0[2] + 0.5f) * P.dx[2]);
    if (box_outside_tet(E.faces, ccen, E.hcell)) return false;

    // clip the tet against this cell in the OVERLAP frame (exact_clip.metal.inc: origin at the centre
    // of the tet bbox's overlap with the cell, where the piece lies): the moments come out about that
    // origin 'cc' (vertex-0 frame), which the caller adds back to the centroid
    float3 const clo = ccen - E.hcell, chi = ccen + E.hcell;
    float3 const tlo = min(min(float3(0.0f), E.t1), min(E.t2, E.t3));
    float3 const thi = max(max(float3(0.0f), E.t1), max(E.t2, E.t3));
    float3 const org = 0.5f * (max(clo, tlo) + min(chi, thi));
    // order 1 in pass 0: the same arithmetic for V and the first moments as order 2, so the
    // passes' volumes agree to the bit (the first moments need not: --ps-linear-deposit, whose weight
    // uses them, clips at order 2 in both passes, depositExactItems' kItemsPhase 2)
    sweep_moments(-org, E.t1 - org, E.t2 - org, E.t3 - org, clo - org, chi - org, full, mo);
    if (!(mo[0] > 0.0f)) return false;

    cc = org;
    w = mo[0];
    if (fLinear) {   // the linear density integrated over the piece: (d0 + dG.cc) V + dG.(first moments about cc)
        w = (E.d0 + E.dG[0]*org.x + E.dG[1]*org.y + E.dG[2]*org.z)*mo[0]
          + E.dG[0]*mo[1] + E.dG[1]*mo[2] + E.dG[2]*mo[3];
        if (w < 0.0f) w = 0.0f;
    }
    return true;
}

// Deposits one clipped piece (moments mo, from exact_cell_clip with full=true): its mass share,
// the exact first and second velocity moments of the linear profile over the piece, the
// gradient, the caustic bits, the exact multiplicity share V_int/V_cell and the tet-touch count.
static inline void exact_cell_deposit(thread const FieldGrids& G, uint flat, thread const float mo[10],
                                      float massShare, thread const ExactTet& E,
                                      bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                      bool fScal, bool fSGrad, float3 cc)
{
    float const wmEx = fVolW ? mo[0] : massShare;
    ulong const base = ulong(flat);
    depositMass(G, flat, massShare);
    float const invV = 1.0f / mo[0];
    // the piece's centroid about the clip origin cc (the moments' frame), and in the vertex-0 frame
    float const loc[3] = { mo[1]*invV, mo[2]*invV, mo[3]*invV };
    float const cen[3] = { cc.x + loc[0], cc.y + loc[1], cc.z + loc[2] };
    float vb[3];
    if (fVel || fDisp) {
        for (int j=0; j<3; ++j) {
            vb[j] = E.u0[j];
            for (int i=0; i<3; ++i) vb[j] += E.vG[i][j]*cen[i];
            // the mean of a linear field over a part of the tet lies within its range over the vertices: a centroid
            // off by float rounding cannot move it outside (a steep gradient G turns eps into a huge velocity)
            vb[j] = clamp(vb[j], E.vc[j] - E.vh[j], E.vc[j] + E.vh[j]);
            if (fVel)
                addMoment(G.mom, G.momX, G, FX_MOM, base*3ul+ulong(j), vb[j]*wmEx);
        }
        if (fDisp) {
            // exact second moment of the linear profile: <v_a v_b> over the piece
            // = vb_a vb_b + (G^T Cov G)_ab, Cov from the order-2 position moments
            float cov[3][3];
            // (translation-invariant: taken about the clip origin, near the piece, where nothing cancels)
            cov[0][0] = mo[4]*invV - loc[0]*loc[0];
            cov[0][1] = cov[1][0] = mo[5]*invV - loc[0]*loc[1];
            cov[0][2] = cov[2][0] = mo[6]*invV - loc[0]*loc[2];
            cov[1][1] = mo[7]*invV - loc[1]*loc[1];
            cov[1][2] = cov[2][1] = mo[8]*invV - loc[1]*loc[2];
            cov[2][2] = mo[9]*invV - loc[2]*loc[2];
            // the velocity covariance over the piece, S = G^T Cov G, clamped to what a linear field on a part of the
            // tet can have: a component spanning a range r = 2 vh has a variance of at most r^2/4 = vh^2, so
            // |S_ab| <= sqrt(S_aa S_bb) <= vh_a vh_b. A correct value always satisfies both (the bound is not taken
            // from the float diagonals, which may be clamped themselves); float noise beyond them is cut.
            float S[3][3];
            for (int a=0; a<3; ++a)
                for (int b=a; b<3; ++b) {
                    float gcg = 0.0f;
                    for (int i=0; i<3; ++i)
                        for (int k=0; k<3; ++k)
                            gcg += E.vG[i][a]*cov[i][k]*E.vG[k][b];
                    S[a][b] = gcg;
                }
            for (int a=0; a<3; ++a) {
                S[a][a] = clamp(S[a][a], 0.0f, E.vh[a]*E.vh[a]);
                for (int b=a+1; b<3; ++b) S[a][b] = clamp(S[a][b], -E.vh[a]*E.vh[b], E.vh[a]*E.vh[b]);
            }
            ulong c2 = 0;
            for (int a=0; a<3; ++a)
                for (int b=a; b<3; ++b)
                    addMoment(G.m2, G.m2X, G, FX_M2, base*6ul + c2++, massShare*(vb[a]*vb[b] + S[a][b]));
            if (fVolW) {
                for (int j=0;j<3;++j)
                    addMoment(G.dispvel, G.dvX, G, FX_DV, base*3ul+ulong(j), vb[j]*massShare);
                if (((G.fxMask >> FX_DW) & 1u) == 0u)   // (else the mass grid is it)
                    atomic_fetch_add_explicit(&G.dispw[flat], massShare, memory_order_relaxed);
            }
        }
    }
    if (fGrad)
        for (int j=0; j<3; ++j)
            for (int i=0; i<3; ++i)
                addMoment(G.grad, G.gradX, G, FX_GRAD, base*9ul + ulong(j*3+i), E.vG[i][j]*wmEx);
    if (fScal) {   // exact mean of the linear scalar profile over the piece (the velocity's construction)
        float sb = E.s0;
        for (int i=0; i<3; ++i) sb += E.sG[i]*cen[i];
        addMoment(G.scal, G.scalX, G, FX_SCAL, base, sb*wmEx);
    }
    if (fSGrad)
        for (int i=0; i<3; ++i)
            addMoment(G.sgrad, G.sgX, G, FX_SG, base*3ul + ulong(i), E.sG[i]*wmEx);
    if (fVolW)
        addMoment(G.momw, G.momwX, G, FX_MOMW, base, wmEx);
    if (fCaustic)
        atomic_fetch_or_explicit(&G.caustic[flat], E.obits, memory_order_relaxed);
    // exact cell-mean multiplicity share V_int/V_cell, and the raw tet-touch count
    addMoment(G.sv, G.svX, G, FX_SV, base, mo[0]*E.invCellVol);
    atomic_fetch_add_explicit(&G.streams[flat], 1u, memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------
// Counting a straddling tet's samples BEYOND a non-periodic grid (pass 0 of the sampled deposit).
// They are only counted (and, under fLinear, their weights summed), so they are taken one sample
// COLUMN (along z) at a time: along a column each numerator of classifySample is affine in the z
// offset, n_a = P_a + Q_a r_z, and re-evaluated in that grouping it differs from classifySample's
// own value by far less than its bound B. So the samples whose numerators all clear 2B are inside
// (classifySample would say +1), those below -2B on some face are outside (it would say -1), and
// each of the two sets is one run along the column, found by binary search. Only the samples in
// neither -- within ~B of a face -- are classified one by one, and an undecidable one defers the
// tet exactly as before, so the count and the deferral decisions are the per-sample ones. Cost: the
// tet's cross-section beyond the grid instead of its volume. Short columns are classified directly.
// ------------------------------------------------------------------------------------------------

// sample index s along one axis -> its offset from vertex 0 in cells (the sampled loop's expression)
static inline float sample_tc(int s, int nSub, float invNSub, int ic0d, float f0d) {
    int const k = (s >= 0) ? s / nSub : -((-s + nSub - 1) / nSub);
    int const sub = s - k * nSub;
    float const fr = (nSub == 1) ? 0.5f : (float(sub) + 0.5f) * invNSub;
    return (float(k - ic0d) - f0d) + fr;
}

struct BeyondColumn {
    float P[4], Q[4];      // sg * numerator_v = sg * (P_v + Q_v r_z), n_z included as ONE affine form
    int   ic0z; float f0z; float dxz; int nSub; float invNSub;
};

static inline float column_value(thread const BeyondColumn& c, int v, int s) {
    float const rz = sample_tc(s, c.nSub, c.invNSub, c.ic0z, c.f0z) * c.dxz;
    return c.P[v] + c.Q[v] * rz;
}

// First index of [lo, hi) where "value(v) > T" becomes true (rise) or false (fall along the column),
// hi if never. Two probes at the analytic crossing 'guess' settle it (it is off by rounding only);
// bisection covers whatever they leave -- plain bisection alone made this the deposit's hot spot.
static inline int column_transition(thread const BeyondColumn& c, int v, float T, bool rise,
                                    int lo, int hi, float guess) {
    int L = lo, H = hi;
    int const g = (guess > float(lo)) ? ((guess < float(hi)) ? int(ceil(guess)) : hi) : lo;
    if (g > L && g - 1 < H) { if ((column_value(c, v, g - 1) > T) == rise) H = g - 1; else L = g; }
    if (g >= L && g < H)    { if ((column_value(c, v, g) > T) == rise) H = g; else L = g + 1; }
    while (L < H) { int const mid = L + (H - L) / 2; if ((column_value(c, v, mid) > T) == rise) H = mid; else L = mid + 1; }
    return L;
}

// [a, b) = the samples of [S0, S1) whose four values all exceed T[v] (one run: each value is monotone)
static inline void column_run(thread const BeyondColumn& c, int S0, int S1, thread const float T[4],
                              thread int& a, thread int& b) {
    a = S0; b = S1;
    for (int v = 0; v < 4 && a < b; ++v) {
        // the sample index where P + Q r_z crosses T: r_z = tc dx, tc = (s + 0.5)/nSub - ic0 - f0
        float const guess = (c.Q[v] != 0.0f)
                          ? ((T[v] - c.P[v]) / c.Q[v] / c.dxz + float(c.ic0z) + c.f0z) * float(c.nSub) - 0.5f : 0.0f;
        if (c.Q[v] > 0.0f)      a = column_transition(c, v, T[v], true,  a, b, guess);   // rises: first above T
        else if (c.Q[v] < 0.0f) b = column_transition(c, v, T[v], false, a, b, guess);   // falls: one past the last
        else if (!(c.P[v] > T[v])) b = a;
    }
}

// Counts the tet's samples beyond the grid into Nout (and their linear weights into sumW).
// Returns false when one of them is undecidable: the caller defers the tet.
static inline bool count_beyond_grid(thread const float C[3][3], thread const float M[3][3],
                                     thread const float Ax[3][3],
                                     float detC, float sg, thread const float B[4], float Bdet,
                                     thread const int iMin[3], thread const int iMax[3],
                                     thread const int ic0[3], thread const float f0[3],
                                     int nSub, float invNSub, bool fLinear, float d0,
                                     thread const float dG[3], constant DepositParams& P,
                                     thread uint& Nout, thread float& sumW)
{
    int const S0z = iMin[2] * nSub, S1z = iMax[2] * nSub, gridZ = P.nGrid[2] * nSub;
    float const Tin[4]  = { 2.0f*B[0], 2.0f*B[1], 2.0f*B[2], 2.0f*B[3] };
    float const Tpos[4] = { -2.0f*B[0], -2.0f*B[1], -2.0f*B[2], -2.0f*B[3] };
    BeyondColumn c;
    c.ic0z = ic0[2]; c.f0z = f0[2]; c.dxz = P.dx[2]; c.nSub = nSub; c.invNSub = invNSub;
    // only the columns that HAVE samples beyond the grid: when the window stays inside the grid along
    // z, a column inside it along x needs just its y-parts beyond the grid
    bool const zStraddle = S0z < 0 || S1z > gridZ;
    int const Sy0 = iMin[1] * nSub, Sy1 = iMax[1] * nSub, gridY = P.nGrid[1] * nSub;
    // ... and only the columns inside the tet's xy projection: along a row (fixed x) one y-interval,
    // the extent of the six projected edges crossing x, padded by two samples on each side
    float const Vx[4] = { 0.0f, Ax[0][0], Ax[1][0], Ax[2][0] };
    float const Vy[4] = { 0.0f, Ax[0][1], Ax[1][1], Ax[2][1] };
    float const xPad = 1.0e-4f * P.dx[0];
    for (int sx = iMin[0] * nSub; sx < iMax[0] * nSub; ++sx) {
    int const kx = (sx >= 0) ? sx / nSub : -((-sx + nSub - 1) / nSub);
    bool const xInCol = kx >= 0 && kx < P.nGrid[0];
    float const relx = sample_tc(sx, nSub, invNSub, ic0[0], f0[0]) * P.dx[0];
    float yMin = INFINITY, yMax = -INFINITY;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 4; ++j) {
            float const xi = Vx[i], xj = Vx[j];
            if (relx < min(xi, xj) - xPad || relx > max(xi, xj) + xPad) continue;
            float y0 = Vy[i], y1 = Vy[j];
            if (xj != xi) {
                float const t = clamp((relx - xi) / (xj - xi), 0.0f, 1.0f);
                y0 = y1 = Vy[i] + t * (Vy[j] - Vy[i]);
            }
            yMin = min(yMin, min(y0, y1));
            yMax = max(yMax, max(y0, y1));
        }
    if (!(yMin <= yMax)) continue;       // this row misses the tet
    int const pLo = max(Sy0, int(floor((yMin / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) - 2);
    int const pHi = min(Sy1, int(ceil((yMax / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) + 3);
    if (pLo >= pHi) continue;
    int yLo[2] = { pLo, 0 }, yHi[2] = { pHi, 0 }, nY = 1;
    if (xInCol && !zStraddle) {
        nY = 0;
        if (pLo < 0)     { yLo[nY] = pLo; yHi[nY] = min(pHi, 0); ++nY; }
        if (pHi > gridY) { yLo[nY] = max(pLo, gridY); yHi[nY] = pHi; ++nY; }
    }
    for (int iy = 0; iy < nY; ++iy)
    for (int sy = yLo[iy]; sy < yHi[iy]; ++sy) {
        int const ky = (sy >= 0) ? sy / nSub : -((-sy + nSub - 1) / nSub);
        bool const colIn = kx >= 0 && kx < P.nGrid[0] && ky >= 0 && ky < P.nGrid[1];
        int outLo[2], outHi[2], nOut = 0;
        if (!colIn) { outLo[0] = S0z; outHi[0] = S1z; nOut = 1; }
        else {
            if (S0z < 0)     { outLo[nOut] = S0z; outHi[nOut] = min(S1z, 0); ++nOut; }
            if (S1z > gridZ) { outLo[nOut] = max(S0z, gridZ); outHi[nOut] = S1z; ++nOut; }
        }
        if (nOut == 0) continue;
        float const tx = sample_tc(sx, nSub, invNSub, ic0[0], f0[0]);
        float const ty = sample_tc(sy, nSub, invNSub, ic0[1], f0[1]);
        float const rx = tx * P.dx[0], ry = ty * P.dx[1];
        int outLen = 0;
        for (int i = 0; i < nOut; ++i) outLen += outHi[i] - outLo[i];
        // per-sample classification of [u, v): short columns, and the samples near a face
        // (written out twice below rather than as a lambda: Metal has none)
        int aIn = 0, bIn = 0, aPos = 0, bPos = 0;
        bool bulk = outLen > 16;
        if (bulk) {
            for (int a = 0; a < 3; ++a) {
                c.P[a] = sg * (C[a][0]*rx + C[a][1]*ry);
                c.Q[a] = sg * C[a][2];
            }
            c.P[3] = sg * (detC - (C[0][0]*rx + C[0][1]*ry + C[1][0]*rx + C[1][1]*ry + C[2][0]*rx + C[2][1]*ry));
            c.Q[3] = -sg * (C[0][2] + C[1][2] + C[2][2]);
            column_run(c, S0z, S1z, Tin, aIn, bIn);    // classifySample says +1
            column_run(c, S0z, S1z, Tpos, aPos, bPos); // the rest are below -B on some face: -1
            if (aIn >= bIn) { aIn = aPos; bIn = aPos; }
        }
        for (int i = 0; i < nOut; ++i) {
            // bulk: the certain run; the samples of [aPos, bPos) outside it are classified below
            int lo1 = outLo[i], hi1 = outHi[i], lo2 = 0, hi2 = 0;
            if (bulk) {
                int const c0 = max(aIn, outLo[i]), c1 = min(bIn, outHi[i]);
                if (c0 < c1) {
                    Nout += uint(c1 - c0);
                    if (fLinear) {
                        // sum of d0 + dG.r over the run: r_z = tc_z dx, tc_z = (s + 0.5)/nSub - ic0 - f0
                        float const n = float(c1 - c0);
                        float const mz = ((float(c0) + float(c1)) * 0.5f * invNSub - float(ic0[2]) - f0[2]) * P.dx[2];
                        sumW += n * (d0 + dG[0]*rx + dG[1]*ry + dG[2]*mz);
                    }
                }
                lo1 = max(aPos, outLo[i]); hi1 = min(aIn, outHi[i]);
                lo2 = max(bIn, outLo[i]);  hi2 = min(bPos, outHi[i]);
            }
            for (int pass2 = 0; pass2 < 2; ++pass2) {
                int const u = pass2 == 0 ? lo1 : lo2, w = pass2 == 0 ? hi1 : hi2;
                for (int sz = u; sz < w; ++sz) {
                    float const tz = sample_tc(sz, nSub, invNSub, ic0[2], f0[2]);
                    float const t[3] = { tx, ty, tz };
                    float rel[3] = { rx, ry, tz * P.dx[2] };
                    int const cls = classifySample(C, M, detC, sg, B, Bdet, rel, t, P);
                    if (cls == 0) return false;
                    if (cls > 0) { Nout++; if (fLinear) sumW += linearWeight(rel, d0, dG); }
                }
            }
        }
    }
    }
    return true;
}

kernel void depositFields(
    device const float*        verts    [[buffer(0)]],  // nTet * TET_STRIDE (the record above)
    device const float*        vels     [[buffer(1)]],  // nTet * 12 (4 verts * vxyz)
    device float*              masses   [[buffer(2)]],  // nTet; a DEFERRED tet's mass is negated here
    device atomic_float*       massGrid [[buffer(3)]],
    device atomic_float*       momGrid  [[buffer(4)]],
    device atomic_float*       m2Grid   [[buffer(5)]],
    device atomic_float*       gradGrid [[buffer(6)]],
    device atomic_uint*        strGrid  [[buffer(7)]],
    constant DepositParams&    P        [[buffer(8)]],
    device const float*        dens     [[buffer(9)]],  // nTet * 4 vertex densities (fLinear; else dummy)
    device atomic_float*       momwGrid [[buffer(10)]], // nCell moment-weight normalizer (fVolW; else dummy)
    device atomic_uint*        caustGrid[[buffer(11)]], // nCell orientation-bit OR grid (fCaustic; else dummy)
    device atomic_float*       svGrid   [[buffer(12)]], // nCell exact multiplicity sum(V_int)/V_cell (fExact; else dummy)
    device atomic_float*       dvGrid   [[buffer(13)]], // nCell*3 dispersion's mass-weighted mean (fVolW+fDisp; else dummy)
    device atomic_float*       dwGrid   [[buffer(14)]], // nCell   its normalizer (fVolW+fDisp; else dummy)
    device const float*        scal     [[buffer(18)]], // nTet*4 vertex scalars (fScal / fSGrad; else dummy)
    device atomic_float*       scalGrid [[buffer(19)]], // nCell   scalar moment (fScal; else dummy)
    device atomic_float*       sgGrid   [[buffer(20)]], // nCell*3 scalar-gradient moment (fSGrad; else dummy)
    device atomic_uint*        massFx   [[buffer(21)]], // nCell*2 the fixed-point mass (massFixed; else dummy -- then buffer 3 is the mass)
    //                                                    (fxWords 2: nCell*4, the four limbs of each cell)
    // the moment grids' fixed-point words (fxMask), the same buffers as 4, 5, 6, 10, 12, 13, 19, 20
    device atomic_uint*        momX     [[buffer(22)]],
    device atomic_uint*        m2X      [[buffer(23)]],
    device atomic_uint*        gradX    [[buffer(24)]],
    device atomic_uint*        momwX    [[buffer(25)]],
    device atomic_uint*        svX      [[buffer(26)]],
    device atomic_uint*        dvX      [[buffer(27)]],
    device atomic_uint*        scalX    [[buffer(28)]],
    device atomic_uint*        sgX      [[buffer(29)]],
    device atomic_uint*        bad      [[buffer(30)]], // the non-finite guard's counters (FieldGrids.bad)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= P.nTet) return;

    float  m  = masses[tid];
    if (m <= 0.0f) return;          // (a tet deferred by an earlier, retried attempt stays deferred)

    device const float* R = verts + ulong(tid) * ulong(TET_STRIDE);
    int   const ic0[3] = { as_type<int>(R[0]), as_type<int>(R[1]), as_type<int>(R[2]) };
    float const f0[3]  = { R[3], R[4], R[5] };
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    uint  const centroidFlat = as_type<uint>(R[15]);

    // The host kept this tet (psFilterCell, double precision); should the float recheck call it
    // degenerate, defer it rather than drop it, so the host deposits it exactly and no mass is lost.
    float d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabs(d) < 1.0e-6f * avgEdge2 * sqrt(avgEdge2) || !inverse3(Ax, posInv)) { masses[tid] = -m; return; }

    // constant per-tet velocity gradient vG = posInv * (u_{k+1}-u_0); skipped entirely (and
    // the vels buffer never read -- it may be a 4-byte dummy) when no velocity-derived grid
    // is requested (fVel, fDisp and fGrad all 0, i.e. a density-only run)
    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const needsVel = fVel || fDisp || fGrad;
    float u0[3] = { 0.0f, 0.0f, 0.0f };
    float vG[3][3] = { {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f} };
    if (needsVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        float dV[3][3];
        for (int e=0;e<3;++e)
            for (int j=0;j<3;++j)
                dV[e][j] = vels[tid*12+(e+1)*3+j] - u0[j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j];
                vG[i][j]=s;
            }
    }

    // the linear scalar profile (its gradient is constant over the tet, like the velocity's)
    bool const fScal = P.fScal != 0, fSGrad = P.fSGrad != 0;
    float s0 = 0.0f, sG[3] = { 0.0f, 0.0f, 0.0f };
    if (fScal || fSGrad) {
        s0 = scal[tid*4+0];
        float ds[3];
        for (int e=0;e<3;++e) ds[e] = scal[tid*4+e+1] - s0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*ds[k]; sG[i]=s; }
    }

    FieldGrids G { massGrid, momGrid, m2Grid, gradGrid, strGrid, momwGrid, caustGrid, svGrid, dvGrid, dwGrid, scalGrid, sgGrid,
                   massFx, P.massScale, P.massFixed != 0, momX, m2X, gradX, momwX, svX, dvX, scalX, sgX,
                   P.fxMask, P.fxScale, bad, P.fxWords, P.fxCapBits };

    // --ps-volume-weighted: the tet's total moment weight is its Eulerian volume |det|/6
    // (matching the CPU volumeShare); default: the moment weight equals the mass share.
    // --ps-caustics: orientation bits from the sign of det(Ax) (the Lagrangian->Eulerian
    // parity; CGAL cells are positively oriented in Lagrangian space).
    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fExact = P.fExact != 0;
    float const mw = fVolW ? fabs(d)/6.0f : m;
    /* Per-tet caustic mask. With fCaustic and NOT fLinear the 'dens' buffer is otherwise unused, so
       the host parks the mask (built in DOUBLE precision from the same deformation tensor the CPU
       deposit uses) in slot 0 of each tet's four floats. That carries the whole stratification --
       collapse multiplicity and the umbilic indicator, not just the parity -- through an EXISTING
       buffer index; adding a sixteenth buffer for it measurably corrupts this grid, whereas
       reusing buffer 9 does not. Under
       fLinear the slot really holds a density, so the GPU falls back to the parity it can derive
       from 'd' and '.causticClass' is left to the CPU deposit. */
    uint  const obits = (fCaustic && P.fLinear == 0) ? uint(dens[tid*4+0]) : (d > 0.0f ? 1u : 2u);
    float const invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    float const tetVol = fabs(d)/6.0f;   // exact-multiplicity share of a monolithic deposit

    // --ps-linear-deposit: constant density gradient across the tet (same affine convention
    // as vG above); the vertex densities arrive per tet in the dens buffer.
    bool const fLinear = P.fLinear != 0;
    float d0 = 0.0f, dG[3] = {0.0f, 0.0f, 0.0f};
    if (fLinear) {
        d0 = dens[tid*4+0];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[tid*4+e+1] - d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; dG[i]=s; }
    }

    // Grid window of the tet's bounding box, in the canonical index frame (vertex extent relative
    // to vertex 0, in cells; periodic indices are wrapped per cell below). On a non-periodic grid
    // iMin/iMax (the sampled loop) stay UNCLAMPED -- a tet cut by the grid face (a --box region of
    // a larger cloud) must count its samples beyond it too, see the sampled path -- and iMinC/
    // iMaxC are the clamped window the exact path clips. A tet wholly beyond the grid deposits
    // nothing (the CPU loop's outsideGrid).
    int iMin[3], iMax[3], iMinC[3], iMaxC[3];
    bool straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = min(lo, Ax[e][dd]); hi = max(hi, Ax[e][dd]); }
        iMin[dd] = ic0[dd] + int(floor(f0[dd] + lo / P.dx[dd]));
        iMax[dd] = ic0[dd] + int(floor(f0[dd] + hi / P.dx[dd])) + 1;
        // straddle is decided on the tet's own bbox (a tet whose bbox stays inside the grid has no
        // sample beyond it)
        if (!P.periodic && (iMin[dd] < 0 || iMax[dd] > P.nGrid[dd])) straddle = true;
        // No one-cell sample margin around this window (CPU and Metal alike): a
        // sub-sample sits at (k + 0.5)/nSub of its cell, >= 1/(2 nSub) inside it, so every sample in
        // the bbox lies in a cell of [floor(lo), floor(hi)] -- exactly this window. The ring's
        // samples would all be outside (each classified, 2 passes x nSub^3 per cell: 1.8-2.7x the
        // window cells of real TNG50 z=0 tets), and the classification is window-independent (the
        // per-sample stage-2 bounds of classifySample never exceed the window's B), so a margin would
        // keep the same samples and deferrals.
        iMinC[dd] = iMin[dd]; iMaxC[dd] = iMax[dd];
        if (!P.periodic) {
            if (iMinC[dd] < 0)           iMinC[dd] = 0;
            if (iMaxC[dd] > P.nGrid[dd]) iMaxC[dd] = P.nGrid[dd];
            if (iMinC[dd] >= P.nGrid[dd] || iMaxC[dd] <= 0) return;   // wholly beyond the grid
        }
    }
    int nSub = P.nSub;

    /* Error bounds of classifySample, per tet. Every sample offset in this window is formed as
       r_i = ((g_i - ic0_i) + frac - f0_i) * dx_i from numbers of order T_i (cells), so it is within
       rho_i = dx_i (u (6 + 5 T_i) + 1e-9) of the exact offset the CPU test uses (the 1e-9 dwarfs the
       double roundings that define that offset). A cofactor C = ab - cd, from edges each rounded
       once to float, is within 4u M of exact, M = |ab| + |cd|; a numerator sum_i C_i r_i is then
       within sum_i M_i (7u R_i + rho_i), R_i = T_i dx_i; det = sum_i E_0i C_0i within
       8u sum_i |E_0i| M_0i; n_z = det - sum_a n_a within the sum of those plus its own 3u rounding.
       Valid under any evaluation order or fma contraction (the bounds use sums of magnitudes);
       doubled for safety. u = 2^-24. */
    float C[3][3], M[3][3];
    {
        float const A00=Ax[0][0], A01=Ax[0][1], A02=Ax[0][2];
        float const A10=Ax[1][0], A11=Ax[1][1], A12=Ax[1][2];
        float const A20=Ax[2][0], A21=Ax[2][1], A22=Ax[2][2];
        C[0][0]=A11*A22-A12*A21; M[0][0]=fabs(A11*A22)+fabs(A12*A21);
        C[0][1]=A12*A20-A10*A22; M[0][1]=fabs(A12*A20)+fabs(A10*A22);
        C[0][2]=A10*A21-A11*A20; M[0][2]=fabs(A10*A21)+fabs(A11*A20);
        C[1][0]=A02*A21-A01*A22; M[1][0]=fabs(A02*A21)+fabs(A01*A22);
        C[1][1]=A00*A22-A02*A20; M[1][1]=fabs(A00*A22)+fabs(A02*A20);
        C[1][2]=A01*A20-A00*A21; M[1][2]=fabs(A01*A20)+fabs(A00*A21);
        C[2][0]=A01*A12-A02*A11; M[2][0]=fabs(A01*A12)+fabs(A02*A11);
        C[2][1]=A02*A10-A00*A12; M[2][1]=fabs(A02*A10)+fabs(A00*A12);
        C[2][2]=A00*A11-A01*A10; M[2][2]=fabs(A00*A11)+fabs(A01*A10);
    }
    float const detC = Ax[0][0]*C[0][0] + Ax[0][1]*C[0][1] + Ax[0][2]*C[0][2];
    float Rw[3], rho[3];
    for (int i=0; i<3; ++i) {
        float const T = max(fabs(float(iMin[i] - ic0[i]) - f0[i]), fabs(float(iMax[i] - ic0[i]) - f0[i]));
        Rw[i]  = T * P.dx[i];
        rho[i] = P.dx[i] * (U_F * (6.0f + 5.0f*T) + 1.0e-9f);
    }
    float B[4];
    float nMag = 0.0f;   // bound on sum_a |n_a| over the window
    for (int a=0; a<3; ++a) {
        float b = 0.0f;
        for (int i=0; i<3; ++i) { b += M[a][i] * (7.0f*U_F*Rw[i] + rho[i]); nMag += M[a][i] * Rw[i]; }
        B[a] = 2.0f * b;
    }
    float const Bdet = 2.0f * 8.0f * U_F * (fabs(Ax[0][0])*M[0][0] + fabs(Ax[0][1])*M[0][1] + fabs(Ax[0][2])*M[0][2]);
    B[3] = Bdet + B[0] + B[1] + B[2] + 2.0f * 3.0f * U_F * (fabs(detC) + nMag);
    // the orientation itself must be certain: a det within its bound leaves every sign open
    bool const orientSure = fabs(detC) > Bdet;
    float const sg = detC > 0.0f ? 1.0f : -1.0f;
    float const invNSub = 1.0f / float(nSub);

    // ===================== exact conservative deposit (--exact) =====================
    // Mirrors the CPU exact path: PASS 0 sums the per-cell weights (V_int, or the exact linear-
    // profile integral under fLinear), PASS 1 renormalizes the shares to the tet mass and
    // deposits the exact moments. A tet whose whole window clips empty (sliver below float
    // resolution) falls through to the sampled two-pass below, whose empty window then drives
    // the centroid fallback -- exactly the CPU fall-through.
    if (fExact) {
        // positively oriented, as the CPU hands it to r3d: a folded (det<0) tet has vertices 1,2
        // swapped; vertex 0 stays the frame origin (the sweep and tet_planes take either orientation)
        ExactTet E;
        E.t1 = float3(Ax[0][0], Ax[0][1], Ax[0][2]);
        E.t2 = float3(Ax[1][0], Ax[1][1], Ax[1][2]);
        E.t3 = float3(Ax[2][0], Ax[2][1], Ax[2][2]);
        if (d < 0.0f) { float3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
        E.faces = tet_planes(E.t1, E.t2, E.t3);
        E.hcell = float3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
        for (int i=0; i<3; ++i) {
            E.ic0[i] = ic0[i];  E.f0[i] = f0[i];  E.u0[i] = u0[i];  E.dG[i] = dG[i];  E.sG[i] = sG[i];
            for (int j=0; j<3; ++j) E.vG[i][j] = vG[i][j];
        }
        for (int j=0; j<3; ++j) { E.vc[j] = 0.0f; E.vh[j] = 0.0f; }
        if (fVel || fDisp) exact_vel_range(Ax, vG, u0, E.vc, E.vh);
        E.d0 = d0;  E.s0 = s0;  E.obits = obits;  E.invCellVol = invCellVol;
        E.straddle = straddle;
        E.wFull = exact_full_weight(tetVol, fLinear, d0, dG, E.t1, E.t2, E.t3);

        // --grid-window: only the window's cells are clipped and the tet is normalized by its whole
        // weight (as a straddling tet is); an empty window then means "no volume inside the
        // window", not a degenerate tet -- nothing to deposit, no sampled fallback
        if (P.windowMode != 0)
            for (int dd=0; dd<3; ++dd)
                if (!window_hull(iMinC[dd], iMaxC[dd], P.nGrid[dd], P.periodic != 0, P.subOrigin[dd], P.subDims[dd])) return;
        // a dispersion float32 cannot resolve: the CPU's double deposit takes the tet (EX_DISP_KAPPA2)
        if (fDisp && exact_disp_illcond(Ax, vG, E.vh, float3(P.dx[0], P.dx[1], P.dx[2]))) { masses[tid] = -m; return; }
        float sumW = 0.0f;
        float shareFac = 0.0f;
        bool  deposited = false;
        // fxWords 2 with the items' sumW words (FX_SUMW_BIT): pass 0 sums each cell's weight as its integer in the
        // tet's unit (sumw_scale) and pass 1 reads the total as the items kernel does (words_to_float), so both
        // kernels give one tet the same total; else the thread's own float sum
        ulong sumQ = 0ul;
        uint  sumBad = 0u;
        bool const sumWInt = FX_WIDE && (P.fxMask & FX_SUMW_BIT) != 0u;
        if (sumWInt) E.wScale = sumw_scale(tetVol, fLinear, dens, ulong(tid));
        for (int pass=0; pass<2; ++pass) {
            if (pass==1) {
                if (sumWInt) {
                    if (sumBad != 0u) atomic_fetch_add_explicit(&bad[FX_DROP_SUMW], sumBad, memory_order_relaxed);
                    sumW = sumQ == 0ul ? 0.0f
                                       : words_to_float(uint(sumQ >> 32), uint(sumQ)) / E.wScale;
                }
                if (!(sumW > 0.0f)) { if (P.windowMode != 0) return; break; }   // empty window: sampled fallback below
                shareFac = m / ((E.straddle || P.windowMode != 0) ? max(sumW, E.wFull) : sumW);
                deposited = true;
            }
            for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
            for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
            for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
                int const raw[3]={gi,gj,gk};
                uint flat;
                if (window_flat(raw, P, flat) != 2) continue;   // outside the sub-box / window
                float mo[10], w;
                float3 cc;
                if (!exact_cell_clip(E, raw, P, pass==1, fLinear, mo, w, cc)) continue;
                if (pass==0) {
                    if (sumWInt) {
                        float const p = w * E.wScale;
                        if (fixedShareOk(p)) sumQ += ulong(long(rint(p)));
                        else ++sumBad;                              // (the non-finite guard, per cell)
                    }
                    else sumW += w;
                    continue;
                }
                exact_cell_deposit(G, flat, mo, w * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic, fScal, fSGrad, cc);
            }
        }
        if (deposited) return;   // exact deposit complete; skip the sampled path
    }

    // Two identical passes over (cell, sample): pass 0 counts N (fLinear: sums the sample weights), pass 1 deposits
    // m/N (fLinear: m*w/sumW). The cell's wrap + sub-grid mapping is resolved before the sample loop (like the CPU
    // inSub guard), so N counts exactly the samples pass 1 deposits: no mass is lost to invalid cells. Pass 0 DEFERS
    // the tet at its first sample the float classification cannot decide; pass 1 meets the same numbers.
    // A straddling tet (cut by a non-periodic grid's face) also counts its samples BEYOND the grid (Nout, and their
    // linear weights) with the same certified test, so each sample carries m / (N + Nout) and the faces of a --box
    // region get no pile-up of the outside share.
    if (!orientSure) { masses[tid] = -m; return; }
    // Without --ps-linear-deposit every sample of the tet carries the same shares, so the position-independent
    // ones of a cell's samples go into the integer grids as one atomic each (depositSampleBatch); under fLinear the
    // mass and moment weights vary per sample and every sample deposits alone
    bool const batched = !fLinear;
    uint N = 0, Nout = 0;
    float sumW = 0.0f;
    float share = 0.0f;
    for (int pass=0; pass<2; ++pass) {
        if (pass==1) {
            // a straddling tet's samples beyond the grid (they share its mass, but are not deposited)
            // (skipped when nothing lies in the grid and the centroid is beyond it: nothing is deposited then)
            if (straddle && (N > 0u || centroidFlat != 0xFFFFFFFFu)
                && !count_beyond_grid(C, M, Ax, detC, sg, B, Bdet, iMin, iMax, ic0, f0, nSub, invNSub,
                                               fLinear, d0, dG, P, Nout, sumW)) { masses[tid] = -m; return; }
            if (N==0u) break;      // nothing in the grid: centroid fallback (or nothing) below
            share = m / float(N + Nout);
        }
        // the in-grid cells only (the clamped window): the samples beyond the grid are counted by
        // count_beyond_grid, and walking a large straddling tet's whole window here cost up to ~10^6
        // empty iterations per thread. The bounds B stay those of the unclamped window.
        for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
        for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
        for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
            int raw[3]={gi,gj,gk};
            int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi;
            int wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj;
            int wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
            bool const inGrid = !(wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2]);
            if (!inGrid) continue;   // beyond the grid: counted column by column (count_beyond_grid)
            uint flat = 0u;
            bool inSub = true;
            {
                int l0=wg0-P.subOrigin[0], l1=wg1-P.subOrigin[1], l2=wg2-P.subOrigin[2];
                // the sub-box may wrap a periodic axis (ps_interpolation.cc): without this every
                // cell on the far side of the seam was skipped -- in a periodic --partition run
                // that is most of each partition's cells (their mass went to centroid fallbacks)
                if (l0<0) l0+=P.nGrid[0];
                if (l1<0) l1+=P.nGrid[1];
                if (l2<0) l2+=P.nGrid[2];
                if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) inSub = false;
                else flat=(uint(l0)*uint(P.subDims[1])+uint(l1))*uint(P.subDims[2])+uint(l2);
            }
            // outside the sub-box: a partition's box holds every cell its tets touch, so skip; a WINDOW
            // still counts the cell's samples in pass 0 (the full run's normalization) and skips its deposit
            if (!inSub && !(P.windowMode != 0 && pass == 0)) continue;
            // this cell's offset from vertex 0 in cells, before the sub-sample fraction
            float const cb[3] = { float(raw[0] - ic0[0]) - f0[0],
                                  float(raw[1] - ic0[1]) - f0[1],
                                  float(raw[2] - ic0[2]) - f0[2] };
            uint nIn = 0u;   // pass 1: the cell's interior samples, whose position-independent shares go in at once below
            for (int k2=0; k2<nSub; ++k2)
            for (int k1=0; k1<nSub; ++k1)
            for (int k0=0; k0<nSub; ++k0) {
                float const fr0 = (nSub==1) ? 0.5f : (float(k0) + 0.5f) * invNSub;
                float const fr1 = (nSub==1) ? 0.5f : (float(k1) + 0.5f) * invNSub;
                float const fr2 = (nSub==1) ? 0.5f : (float(k2) + 0.5f) * invNSub;
                float const tc[3] = { cb[0] + fr0, cb[1] + fr1, cb[2] + fr2 };   // offset in cells
                float rel[3] = { tc[0] * P.dx[0], tc[1] * P.dx[1], tc[2] * P.dx[2] };
                int const cls = classifySample(C, M, detC, sg, B, Bdet, rel, tc, P);
                if (cls == 0 && pass == 0) { masses[tid] = -m; return; }   // nothing deposited yet
                if (cls > 0) {
                    if (pass==0) {
                        N++;
                        if (fLinear) sumW += linearWeight(rel, d0, dG);
                    } else {
                        float w = share;
                        if (fLinear && sumW > 0.0f)
                            w = m * linearWeight(rel, d0, dG) / sumW;
                        // moment weight: equal V_eul share under fVolW (fLinear is rejected
                        // with fVolW at option parsing), else the mass share w
                        float wm = fVolW ? mw / float(N + Nout) : w;
                        // --exact fall-through: share the tet volume over its samples
                        float sv = fExact ? (tetVol*invCellVol)/float(N + Nout) : 0.0f;
                        if (P.debugNoAtomic == 0u)
                            depositSample(G, flat, rel, w, wm, obits, u0, vG, s0, sG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, sv, true, false, fScal, fSGrad, batched);
                        ++nIn;
                    }
                }
            }
            if (batched && nIn > 0u && P.debugNoAtomic == 0u) {
                // the shares of every sample above (w = share and wm, sv as computed there: fLinear is off)
                float const wm = fVolW ? mw / float(N + Nout) : share;
                float const sv = fExact ? (tetVol*invCellVol)/float(N + Nout) : 0.0f;
                depositSampleBatch(G, flat, nIn, share, wm, obits, vG, sG, fGrad, fVolW, fCaustic, fExact, sv, fSGrad);
            }
        }
    }

    if (N==0u && Nout > 0u) return;   // every sample lies beyond the grid: not the grid's mass
    if (N==0u) {   // centroid fallback (mass + fields at the centroid position)
        // the target cell comes from the host, computed exactly as the CPU fallback computes it,
        // so both deposits put this mass (and the sub-sample mass flag) in the same cell
        if (centroidFlat == 0xFFFFFFFFu) return;   // centroid outside this partition's box: drop
        float crel[3];
        for (int i=0; i<3; ++i) crel[i] = (Ax[0][i] + Ax[1][i] + Ax[2][i]) * 0.25f;
        if (P.debugNoAtomic == 0u)
            depositSample(G, centroidFlat, crel, m, mw, obits, u0, vG, s0, sG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, tetVol*invCellVol, fExact, true, fScal, fSGrad, false);
    }
}

// ------------------------------------------------------------------------------------------------
// depositExactItems: the exact deposit with the work split BELOW the tetrahedron. One thread per
// ITEM = (tet, block of IP.blockCells consecutive cells of the tet's window, in depositFields'
// gi-gj-gk order). With one thread per tet the cost per thread spans five orders of magnitude when
// cells are much smaller than tets (a stretched void tet clips ~10^4 cells, a halo tet one), so
// each buffer waits on its few giants and the dispatch controller shrinks buffers to 50 tets
// (0.26M particles on 256^3: >17 min). Two dispatches over ALL items:
//   phase 0: each item adds its cells' weights into sumW[tet] (atomic);
//   phase 1: each item deposits its cells with the share m / sumW[tet], so the tet still deposits
//            exactly its mass (to float rounding), as in the one-thread path.
// A tet whose sumW stays 0 -- float-degenerate, or a window that clips empty -- deposits nothing
// here; the host runs depositFields on those few, which falls through to the sampled path and the
// centroid fallback, or defers to the CPU, exactly as in the one-thread path. Nothing here writes 'masses'.
// Under --grid-window the items cover only the window's cells and the share is m / wFull (the whole
// weight): a tet with no volume in the window stays at sumW 0 and depositFields drops it.
// ------------------------------------------------------------------------------------------------
// The per-dispatch parameters (buffer 17): PSExactItemParams in ps_kernel_params.h (prepended, as DepositParams).
// badAt: the word of the sumW buffer (bound at 16 and 30) where the non-finite guard's counters start,
// past the per-tet totals -- every one of the 31 buffer slots is taken here, so they ride in that buffer's tail
typedef PSExactItemParams ExactItemParams;

// An item's (tet, block) from the items' map: psItemOf in metal/ps_common.metal.inc (prepended; the face kernel
// reads its own items' map with it too).

// The pass a depositExactItems pipeline runs (== ExactItemParams.phase): the host builds one pipeline per value,
// so the summing pass carries none of the deposit's registers (an M1 Max then fits 512 threads per threadgroup
// instead of 384: phase 0 ran 12-23% faster, the deposit up to 8%).
// 2 = both passes in ONE pipeline, the pass read from ExactItemParams.phase at run time: the host takes it under
// --ps-linear-deposit (P.fLinear). That weight is built from the clip's first moments, and two pipelines may contract
// them differently under fast math (docs/cleanup-plan.md P4.4: about 17% of the pieces, by up to 3.7e-8 of a cell),
// so phase 0's total missed the weights phase 1 deposits and a tet deposited m (sum w1 / sum w0), about 1e-8 off its
// mass. In one pipeline both passes run the same compiled clip (order 2, the deposit's), so both form the same
// weights; the total carries only the per-item float sum's rounding (phase 0 sums an item's weights in float and
// adds that sum to the tet's total once; under fxWords 2 each cell's integer, no float sum), as the uniform
// deposit's does. Values 0 and 1 fold to the constants they always were (deposit = phase 1, the clip's order 2 only
// there), so the uniform deposit's two pipelines are the code they were.
constant uint kItemsPhase [[function_constant(0)]];

// The per-tet numbers depositFields derives in its preamble, for the exact path (keep in sync).
// False when the float recheck calls the tet degenerate.
// decide: also defer an ill-conditioned dispersion (EX_DISP_KAPPA2) by returning false -- phase 0 only. Phase 1 must not
// decide again: its pipeline is compiled apart (kItemsPhase), and a borderline tet it called ill-conditioned after
// phase 0 summed its weights would be lost (sumW > 0: no leftover). A tet phase 0 left out has sumW 0 there anyway.
static inline bool exact_tet_setup(uint tet, device const float* verts, device const float* vels,
                                   device const float* dens, device const float* scal, constant DepositParams& P,
                                   thread ExactTet& E, thread int iMin[3], thread int iMax[3], bool decide)
{
    device const float* R = verts + ulong(tet) * ulong(TET_STRIDE);
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    for (int i=0; i<3; ++i) { E.ic0[i] = as_type<int>(R[i]); E.f0[i] = R[3+i]; }

    float const d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabs(d) < 1.0e-6f * avgEdge2 * sqrt(avgEdge2) || !inverse3(Ax, posInv)) return false;

    for (int i=0; i<3; ++i) { E.u0[i] = 0.0f; for (int j=0; j<3; ++j) E.vG[i][j] = 0.0f; }
    if (P.fVel != 0 || P.fDisp != 0 || P.fGrad != 0) {
        ulong const v0 = ulong(tet) * 12ul;
        for (int j=0;j<3;++j) E.u0[j] = vels[v0+ulong(j)];
        float dV[3][3];
        for (int e=0;e<3;++e)
            for (int j=0;j<3;++j)
                dV[e][j] = vels[v0+ulong((e+1)*3+j)] - E.u0[j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j];
                E.vG[i][j]=s;
            }
    }
    for (int j=0; j<3; ++j) { E.vc[j] = 0.0f; E.vh[j] = 0.0f; }
    if (P.fVel != 0 || P.fDisp != 0) exact_vel_range(Ax, E.vG, E.u0, E.vc, E.vh);
    // ill-conditioned: left to depositFields, which defers it to the CPU (EX_DISP_KAPPA2)
    if (P.fDisp != 0 && decide && exact_disp_illcond(Ax, E.vG, E.vh, float3(P.dx[0], P.dx[1], P.dx[2]))) return false;
    E.s0 = 0.0f;
    for (int i=0; i<3; ++i) E.sG[i] = 0.0f;
    if (P.fScal != 0 || P.fSGrad != 0) {
        E.s0 = scal[ulong(tet)*4ul];
        float ds[3];
        for (int e=0;e<3;++e) ds[e] = scal[ulong(tet)*4ul+ulong(e+1)] - E.s0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*ds[k]; E.sG[i]=s; }
    }
    E.obits = (P.fCaustic != 0 && P.fLinear == 0) ? uint(dens[ulong(tet)*4ul]) : (d > 0.0f ? 1u : 2u);
    E.invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    E.d0 = 0.0f;
    for (int i=0; i<3; ++i) E.dG[i] = 0.0f;
    if (P.fLinear != 0) {
        E.d0 = dens[ulong(tet)*4ul];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[ulong(tet)*4ul+ulong(e+1)] - E.d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; E.dG[i]=s; }
    }

    // the bbox window, exactly as depositFields forms it on its exact path (no sample margin)
    E.straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = min(lo, Ax[e][dd]); hi = max(hi, Ax[e][dd]); }
        iMin[dd] = E.ic0[dd] + int(floor(E.f0[dd] + lo / P.dx[dd]));
        iMax[dd] = E.ic0[dd] + int(floor(E.f0[dd] + hi / P.dx[dd])) + 1;
        if (!P.periodic) {
            if (iMin[dd] < 0)           { iMin[dd] = 0;           E.straddle = true; }
            if (iMax[dd] > P.nGrid[dd]) { iMax[dd] = P.nGrid[dd]; E.straddle = true; }
        }
    }

    // --grid-window: only the window's cells (the hull of its images along each axis) are items;
    // a tet missing the window gets an empty span (no items idle on it: the host bound has the
    // same hull)
    if (P.windowMode != 0)
        for (int dd=0; dd<3; ++dd)
            if (!window_hull(iMin[dd], iMax[dd], P.nGrid[dd], P.periodic != 0, P.subOrigin[dd], P.subDims[dd])) { iMax[dd] = iMin[dd]; }

    E.t1 = float3(Ax[0][0], Ax[0][1], Ax[0][2]);
    E.t2 = float3(Ax[1][0], Ax[1][1], Ax[1][2]);
    E.t3 = float3(Ax[2][0], Ax[2][1], Ax[2][2]);
    if (d < 0.0f) { float3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
    E.faces = tet_planes(E.t1, E.t2, E.t3);
    E.hcell = float3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
    E.wFull = exact_full_weight(fabs(d)/6.0f, P.fLinear != 0, E.d0, E.dG, E.t1, E.t2, E.t3);
    E.wScale = 1.0f;
    if ((P.fxMask & FX_SUMW_BIT) != 0u) E.wScale = sumw_scale(fabs(d)/6.0f, P.fLinear != 0, dens, ulong(tet));
    return true;
}

kernel void depositExactItems(
    device const float*        verts    [[buffer(0)]],
    device const float*        vels     [[buffer(1)]],
    device const float*        masses   [[buffer(2)]],  // read only: deferral is depositFields' job
    device atomic_float*       massGrid [[buffer(3)]],
    device atomic_float*       momGrid  [[buffer(4)]],
    device atomic_float*       m2Grid   [[buffer(5)]],
    device atomic_float*       gradGrid [[buffer(6)]],
    device atomic_uint*        strGrid  [[buffer(7)]],
    constant DepositParams&    P        [[buffer(8)]],
    device const float*        dens     [[buffer(9)]],
    device atomic_float*       momwGrid [[buffer(10)]],
    device atomic_uint*        caustGrid[[buffer(11)]],
    device atomic_float*       svGrid   [[buffer(12)]],
    device atomic_float*       dvGrid   [[buffer(13)]],
    device atomic_float*       dwGrid   [[buffer(14)]],
    device const uint*         itemMap  [[buffer(15)]], // the items' map (psItemOf), tets indexing buffers 0-2 and 9 unoffset
    device atomic_float*       sumW     [[buffer(16)]], // nTet: per-tet weight total (phase 0 writes, 1 reads)
    constant ExactItemParams&  IP       [[buffer(17)]],
    device const float*        scal     [[buffer(18)]],
    device atomic_float*       scalGrid [[buffer(19)]],
    device atomic_float*       sgGrid   [[buffer(20)]],
    device atomic_uint*        massFx   [[buffer(21)]],
    device atomic_uint*        momX     [[buffer(22)]],
    device atomic_uint*        m2X      [[buffer(23)]],
    device atomic_uint*        gradX    [[buffer(24)]],
    device atomic_uint*        momwX    [[buffer(25)]],
    device atomic_uint*        svX      [[buffer(26)]],
    device atomic_uint*        dvX      [[buffer(27)]],
    device atomic_uint*        scalX    [[buffer(28)]],
    device atomic_uint*        sgX      [[buffer(29)]],
    // FX_SUMW_BIT: sumW as two words per tet, the same buffer as 16 -- phase 0 adds each item's
    // weight as an integer in the tet's own unit (ExactTet.wScale; under fxWords 2 the sum of its cells' integers),
    // so the total does not depend on the order the items land in (a float atomic's would, and every share
    // m / sumW with it: the mass grid too)
    device atomic_uint*        sumWX    [[buffer(30)]],
    uint lid [[thread_position_in_grid]])
{
    if (lid >= IP.nItems) return;
    device atomic_uint* const bad = sumWX + ulong(IP.badAt);   // the guard's counters (ExactItemParams.badAt)
    uint2 const it = psItemOf(itemMap, itemMap + (ulong(IP.nTet) + 1ul), IP.nTet, IP.itemBase + lid);   // (tet, block)
    uint  const tet = it.x;
    // (kItemsPhase 2: the pass from IP.phase, the clip at order 2 in both; see kItemsPhase)
    bool const deposit = kItemsPhase == 2u ? IP.phase == 1u : kItemsPhase == 1u;
    bool const fullMoments = kItemsPhase != 0u;
    float const m = masses[tet];
    if (m <= 0.0f) return;           // deferred by an earlier, retried attempt
    bool const sumWFixed = (P.fxMask & FX_SUMW_BIT) != 0u;
    float s = 0.0f;
    if (deposit && !sumWFixed) {
        s = atomic_load_explicit(&sumW[tet], memory_order_relaxed);
        if (!(s > 0.0f)) return;     // left to depositFields (host)
    }

    ExactTet E;
    int iMin[3], iMax[3];
    if (!exact_tet_setup(tet, verts, vels, dens, scal, P, E, iMin, iMax, !deposit)) return;
    if (deposit && sumWFixed) {      // the exact total, converted to float: the same float whatever the order
        // (from the two words separately: a 64-bit integer built from two atomic loads crashed the Metal compiler
        // service when it built this pipeline -- XPC_ERROR_CONNECTION_INTERRUPTED, macOS 27 -- as the
        // standard deposit found for its threadgroup table)
        uint const lo = atomic_load_explicit(&sumWX[ulong(tet)*2ul], memory_order_relaxed);
        int const hi = as_type<int>(atomic_load_explicit(&sumWX[ulong(tet)*2ul+1ul], memory_order_relaxed));
        if (hi < 0 || (hi == 0 && lo == 0u)) return;   // left to depositFields (host)
        // (the wide words: in one rounding, words_to_float; else the two-step sum of the releases before it, as it
        // was)
#if PS_FX_WORDS == 2
        s = words_to_float(uint(hi), lo) / E.wScale;
#else
        s = (float(hi) * 4294967296.0f + float(lo)) / E.wScale;
#endif
    }
    // a tet cut by the grid face keeps only its inside share: normalized by its whole weight --
    // as is every tet of a --grid-window run (only its window pieces are clipped)
    float const shareFac = deposit ? m / ((E.straddle || P.windowMode != 0) ? max(s, E.wFull) : s) : 0.0f;
    ulong const n1 = ulong(max(iMax[1] - iMin[1], 0));
    ulong const n2 = ulong(max(iMax[2] - iMin[2], 0));
    ulong const n12 = n1 * n2;
    ulong const nW = ulong(max(iMax[0] - iMin[0], 0)) * n12;
    ulong w = ulong(it.y) * ulong(IP.blockCells);
    if (w >= nW) return;             // the host's window bound has a margin: surplus blocks idle
    ulong const wEnd = (nW - w > ulong(IP.blockCells)) ? w + ulong(IP.blockCells) : nW;
    int a0 = int(w / n12);
    ulong const rem = w - ulong(a0) * n12;
    int a1 = int(rem / n2);
    int a2 = int(rem - ulong(a1) * n2);

    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fLinear = P.fLinear != 0;
    bool const fScal = P.fScal != 0, fSGrad = P.fSGrad != 0;
    FieldGrids G { massGrid, momGrid, m2Grid, gradGrid, strGrid, momwGrid, caustGrid, svGrid, dvGrid, dwGrid, scalGrid, sgGrid,
                   massFx, P.massScale, P.massFixed != 0, momX, m2X, gradX, momwX, svX, dvX, scalX, sgX,
                   P.fxMask, P.fxScale, bad, P.fxWords, P.fxCapBits };
    // phase 0's sum of the item's cell weights: under fxWords 2 each cell's own integer in the tet's unit
    // (sumw_scale), so the tet's total does not depend on the cells per item; else one float per item, as an integer
    // once
    bool const sumWInt = FX_WIDE && sumWFixed;
    ulong accQ = 0ul;
    uint  accBad = 0u;
    float acc = 0.0f;
    for (; w < wEnd; ++w) {
        int const raw[3] = { iMin[0] + a0, iMin[1] + a1, iMin[2] + a2 };
        if (ulong(++a2) == n2) { a2 = 0; if (ulong(++a1) == n1) { a1 = 0; ++a0; } }
        uint flat;
        if (window_flat(raw, P, flat) != 2) continue;   // outside the sub-box / window
        float mo[10], wt;
        float3 cc;
        if (!exact_cell_clip(E, raw, P, fullMoments, fLinear, mo, wt, cc)) continue;
        if (!deposit) {
            if (sumWInt) {
                float const p = wt * E.wScale;
                if (fixedShareOk(p)) accQ += ulong(long(rint(p)));
                else ++accBad;                                      // (the non-finite guard, per cell)
            }
            else acc += wt;
        }
        else exact_cell_deposit(G, flat, mo, wt * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic, fScal, fSGrad, cc);
    }
    if (!deposit && sumWInt) {
        if (accQ != 0ul) addFixedQ(sumWX, ulong(tet), accQ);
        if (accBad != 0u) atomic_fetch_add_explicit(&bad[FX_DROP_SUMW], accBad, memory_order_relaxed);
    }
    else if (!deposit && acc > 0.0f) {
        if (sumWFixed) {
            float const p = acc * E.wScale;
            if (fixedShareOk(p)) addFixedQ(sumWX, ulong(tet), as_type<ulong>(long(rint(p))));
            else atomic_fetch_add_explicit(&bad[FX_DROP_SUMW], 1u, memory_order_relaxed);   // (the non-finite guard)
        }
        else atomic_fetch_add_explicit(&sumW[tet], acc, memory_order_relaxed);
    }
}
