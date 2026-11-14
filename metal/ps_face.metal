//  PS-DTFE's face sums (--face-gradient) on the GPU: the Metal face kernel of DTFE_GPU_FACES=1 (docs/cleanup-plan.md
//  P11, the GPU kernel for the face sweep, step 3 of its design).
//
//  The CPU's face walk (ps_interpolation.cc psFaceWalk -> psDepositFaces) cuts the section of every kept tetrahedron
//  with each grid plane it owns into the face squares it covers and adds, per piece, the integral of the stream's
//  linear velocity (S, three components), the covered area (N) and the normal momentum (P = rho S_a) to the face
//  words of the cell below the plane (its UPPER face, side 1) and of the cell above (its LOWER face, side 0). This
//  kernel makes the same words, one thread per ITEM = (record, axis, plane): the host's face record of the
//  tetrahedron (ps_kernel_params.h PS_FACE_REC_*), the item's plane from the items' map (psItemOf) and the plane
//  counts, and that plane swept by face_sweep_f32.h's fsSweepPlane -- the canonical float arithmetic the CPU runs
//  under DTFE_FACE_F32=1 and DTFE_GPU_FACES=1, operation for operation. Every float operation there is a correctly
//  rounded IEEE single operation (this library is compiled in Metal's safe math mode, with precise functions: the
//  host's MTLCompileOptions; the deposit's library keeps the default fast math), every product that feeds a sum an
//  explicit fma, and no function is transcendental; so the float shares are the CPU's bit for bit, their integers
//  rint(share * 2^K) too (the unit is a power of two: the product is exact in float and in double), and integer
//  adds land in any order. The device probe faceProbe (below) checks those operations under this library's options.
//
//  The words: per sub-grid cell 18 + 6 + 6 components (S: [flat*18 + (side*3 + a)*3 + j], N and P: [flat*6 +
//  side*3 + a], flat = (l0 * subDims[1] + l1) * subDims[2] + l2), the host's own layout (ps_interpolation.cc
//  PSFaceF32Sink). Under PS_FX_WORDS 0 each component is the host's int64 word as two uint (low, high), added with
//  addFixedQ after the non-finite guard (fixedShareOk); under PS_FX_WORDS 2 four 32-bit limbs (fixed_words.metal.inc
//  addShare below the share cap), which the host folds into its run-wide words. A share the guard or the cap drops
//  adds nothing and is counted (PS_FACE_CNT_S / N / P), as the CPU counts FX_DROP_FACE_S / N / P.
//
//  The Makefile prepends, in this order: metal/fixed_words.metal.inc (<metal_stdlib> and the run-wide words' adds),
//  src/CGAL_triangulation/ps_kernel_params.h (PSFaceParams, the record layout, the counter slots),
//  metal/ps_common.metal.inc (fixedShareOk, addFixedQ, psItemOf: the deposit kernel's own) and
//  src/CGAL_triangulation/face_sweep_f32.h; the host puts "#define PS_FX_WORDS 0" (or 2) in front of it all, as for
//  the deposit's library, and "#define FS_REGISTERS 1" for the register form of the sweep (face_sweep_f32.h; which of
//  the two forms is faster is TO MEASURE).

#include <metal_stdlib>
using namespace metal;

#if !defined(PS_FX_WORDS) || (PS_FX_WORDS != 0 && PS_FX_WORDS != 2)
#error "PS_FX_WORDS must be 0 or 2 (the host compiles one face library per word size)"
#endif
#define FX_WIDE (PS_FX_WORDS == 2)

typedef PSFaceParams FaceParams;

// One share v of a face accumulator into its component 'at' of x, the CPU's psFxAdd on the same float: under
// PS_FX_WORDS 0 the integer rint(v * scale) into the two-word cell (the guard as ps_deposit.metal's addMoment), under
// 2 the four limbs below the share cap (addShare). A share that has no integer there adds nothing and is counted.
static inline void faceAdd(device atomic_uint* x, ulong at, float v, float scale, uint capBits,
                           device atomic_uint* cnt, uint slot)
{
    if (FX_WIDE) {
        if (!addShare(x, at, v, scale, capBits))
            atomic_fetch_add_explicit(&cnt[slot], 1u, memory_order_relaxed);
        return;
    }
    float const p = v * scale;
    if (fixedShareOk(p)) addFixedQ(x, at, as_type<ulong>(long(rint(p))));
    else atomic_fetch_add_explicit(&cnt[slot], 1u, memory_order_relaxed);   // (the non-finite guard)
}

// fsSweepPlane's sink on the device: a piece's five shares into the words of the cell below the plane (side 1, its
// upper face) and of the cell above (side 0, its lower face), each where it lies in the sub-box (fsLocal >= 0) --
// ps_interpolation.cc PSFaceF32Sink's adds, with no slab bound (the face walk passes none either)
struct FaceSink {
    device atomic_uint* wS;
    device atomic_uint* wN;
    device atomic_uint* wP;
    device atomic_uint* cnt;
    float sS, sN, sP;          // the units (FaceParams.scale)
    uint  capS, capN, capP;    // the share caps' bits (FaceParams.capBits; PS_FX_WORDS 2 only)
    ulong dim1, dim2;          // subDims[1], subDims[2]
    void piece(int a, int below, int above, int lu, int lw, float s0, float s1, float s2, float n, float p) const {
        // the cell's local indices: the side's index along a, lu along u and lw along w (u < w, the two other axes)
        for (int side = 1; side >= 0; --side) {
            int const ca = side == 1 ? below : above;
            if (ca < 0) continue;
            int const l0 = a == 0 ? ca : lu;
            int const l1 = a == 1 ? ca : (a == 0 ? lu : lw);
            int const l2 = a == 2 ? ca : lw;
            ulong const flat = (ulong(l0) * dim1 + ulong(l1)) * dim2 + ulong(l2);
            ulong const slot = ulong(side * 3 + a);
            faceAdd(wS, flat * 18ul + slot * 3ul + 0ul, s0, sS, capS, cnt, PS_FACE_CNT_S);
            faceAdd(wS, flat * 18ul + slot * 3ul + 1ul, s1, sS, capS, cnt, PS_FACE_CNT_S);
            faceAdd(wS, flat * 18ul + slot * 3ul + 2ul, s2, sS, capS, cnt, PS_FACE_CNT_S);
            faceAdd(wN, flat * 6ul + slot, n, sN, capN, cnt, PS_FACE_CNT_N);
            faceAdd(wP, flat * 6ul + slot, p, sP, capP, cnt, PS_FACE_CNT_P);
        }
    }
};

// One item = one plane k of axis a of one record. Thread i of a chunk runs item itemBase + i: psItemOf gives its
// record and its number among the record's items, which run over the owned planes of axis 0, then 1, then 2 (the
// record's three counts). The device counts the planes of the item's axis again (fsPlanes, on the record's own
// floats) and adds nothing, counting a mismatch, where its count is not the host's -- so a device that rounded
// differently from the CPU shows in the counter instead of in the words. The register form (FS_PREDICATED) runs each
// axis as its own inlined copy, so the axis indices are known at compile time.
kernel void faceSweepItems(
    device const float*   recs    [[buffer(0)]],   // nRec records, PS_FACE_REC_FLOATS floats each
    device const uint*    itemMap [[buffer(1)]],   // the items' map: itemStart[nRec + 1], then blockTet (psItemOf)
    device atomic_uint*   wS      [[buffer(2)]],   // S words: 18 components per sub-grid cell
    device atomic_uint*   wN      [[buffer(3)]],   // N words: 6 per cell
    device atomic_uint*   wP      [[buffer(4)]],   // P words: 6 per cell
    device atomic_uint*   cnt     [[buffer(5)]],   // PS_FACE_CNT_WORDS counters
    constant FaceParams&  FP      [[buffer(6)]],
    uint lid [[thread_position_in_grid]])
{
    if (lid >= FP.nItems) return;
    if (FP.fxWords != uint(PS_FX_WORDS)) {   // a library of the other word size: its words would be misread
        atomic_fetch_add_explicit(&cnt[PS_FACE_CNT_MISMATCH], 1u, memory_order_relaxed);
        return;
    }
    uint2 const it = psItemOf(itemMap, itemMap + (ulong(FP.nRec) + 1ul), FP.nRec, FP.itemBase + lid);
    device const float* const R = recs + ulong(it.x) * ulong(PS_FACE_REC_FLOATS);

    // the pass's grid and the record, into thread memory (face_sweep_f32.h takes thread references)
    FsGrid g;
    for (int d = 0; d < 3; ++d) {
        g.lo[d] = FP.lo[d];
        g.h[d] = FP.h[d];
        g.ih[d] = FP.ih[d];
        g.n[d] = FP.nGrid[d];
        g.subOrigin[d] = FP.subOrigin[d];
        g.subDims[d] = FP.subDims[d];
    }
    g.periodic = FP.periodic;
    float E[4][3], vG[3][3], v0[3];
    for (int v = 0; v < 4; ++v)
        for (int d = 0; d < 3; ++d) E[v][d] = R[PS_FACE_REC_E + 3 * v + d];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) vG[i][j] = R[PS_FACE_REC_VG + 3 * i + j];
    for (int j = 0; j < 3; ++j) v0[j] = R[PS_FACE_REC_V0 + j];
    FsTet t;
    fsTetSet(E, g, vG, v0, R[PS_FACE_REC_RHO], t);

    // the item's axis and its number among that axis' owned planes
    uint const c0 = as_type<uint>(R[PS_FACE_REC_COUNT + 0]);
    uint const c1 = as_type<uint>(R[PS_FACE_REC_COUNT + 1]);
    uint const c2 = as_type<uint>(R[PS_FACE_REC_COUNT + 2]);
    uint const item = it.y;
    int a = 0;
    uint ja = item, ca = c0;
    if (item >= c0) {
        if (item - c0 < c1) { a = 1; ja = item - c0; ca = c1; }
        else { a = 2; ja = item - c0 - c1; ca = c2; }
    }
    int k0 = 0, nDev = 0;
#if FS_PREDICATED
    if (a == 0) nDev = fsPlanes(t, g, 0, k0);
    else if (a == 1) nDev = fsPlanes(t, g, 1, k0);
    else nDev = fsPlanes(t, g, 2, k0);
#else
    nDev = fsPlanes(t, g, a, k0);
#endif
    if (!(ja < ca) || uint(nDev) != ca) {     // the map, the record and the device disagree: nothing added
        atomic_fetch_add_explicit(&cnt[PS_FACE_CNT_MISMATCH], 1u, memory_order_relaxed);
        return;
    }
    int const k = k0 + int(ja);

    FaceSink sink;
    sink.wS = wS;
    sink.wN = wN;
    sink.wP = wP;
    sink.cnt = cnt;
    sink.sS = FP.scale[0];
    sink.sN = FP.scale[1];
    sink.sP = FP.scale[2];
    sink.capS = FP.capBits[0];
    sink.capN = FP.capBits[1];
    sink.capP = FP.capBits[2];
    sink.dim1 = ulong(FP.subDims[1]);
    sink.dim2 = ulong(FP.subDims[2]);
#if FS_PREDICATED
    if (a == 0) fsSweepPlane<FaceSink>(t, g, 0, k, sink);
    else if (a == 1) fsSweepPlane<FaceSink>(t, g, 1, k, sink);
    else fsSweepPlane<FaceSink>(t, g, 2, k, sink);
#else
    fsSweepPlane<FaceSink>(t, g, a, k, sink);
#endif
}

// The device probe of the face library's arithmetic (the kernel's design K11): per case the inputs (a, b, c, d) =
// cases[PS_FACE_PROBE_IN * id ..], the outputs' float bits (uint) at bits[PS_FACE_PROBE_OUT * id ..], each to compare
// bit for bit with the CPU's operation on the same floats (C++ float arithmetic with the contraction off):
//    0  a / b                        FS_DIV: correctly rounded in the safe math mode?
//    1  fma(a, b, c)                 FS_FMA
//    2  a * b                        a product (subnormal results: the device flushes them to zero)
//    3  t = a * d; t + c             two statements: the fused value fma(a, d, c) here means the library contracts
//                                    across statements, which face_sweep_f32.h assumes it does not
//    4  a + c,  5  a - c             sums and differences (subnormal inputs and results)
//    6  t = c * d; fma(a, b, -t)     fsMoments6's cross term
//    7  d + (6)                      ... and its area sum
//    8  rint(a),  9  floor(a)
//   10  fsFloorIndex(a): the int's bits, or 0x80000000 where it has none
//   11  p = a * c: fixedShareOk(p) (1 / 0)        -- with c a power of two, as every unit is, the CPU's reference
//   12  low and 13 high word of long(rint(p)),       is its psFxAdd: |double(a) * double(c)| < 2^63 and
//       0 where the guard said no                    llrint(double(a) * double(c)); for any other c the device
//                                                    rounds p to float first: the same on double(float(a * c))
kernel void faceProbe(
    device const float*  cases [[buffer(0)]],   // nCase * PS_FACE_PROBE_IN floats
    device uint*         bits  [[buffer(1)]],   // nCase * PS_FACE_PROBE_OUT words
    constant uint&       nCase [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= nCase) return;
    device const float* const q = cases + ulong(id) * ulong(PS_FACE_PROBE_IN);
    device uint* const o = bits + ulong(id) * ulong(PS_FACE_PROBE_OUT);
    float const a = q[0], b = q[1], c = q[2], d = q[3];
    o[0] = as_type<uint>(FS_DIV(a, b));
    o[1] = as_type<uint>(FS_FMA(a, b, c));
    o[2] = as_type<uint>(a * b);
    float const ad = a * d;
    float const adc = ad + c;
    o[3] = as_type<uint>(adc);
    o[4] = as_type<uint>(a + c);
    o[5] = as_type<uint>(a - c);
    float const cd = c * d;
    float const cr = FS_FMA(a, b, -cd);
    o[6] = as_type<uint>(cr);
    float const area = d + cr;
    o[7] = as_type<uint>(area);
    o[8] = as_type<uint>(rint(a));
    o[9] = as_type<uint>(FS_FLOOR(a));
    int k = 0;
    o[10] = fsFloorIndex(a, k) ? as_type<uint>(k) : 0x80000000u;
    float const p = a * c;
    bool const ok = fixedShareOk(p);
    ulong const r = ok ? as_type<ulong>(long(rint(p))) : 0ul;
    o[11] = ok ? 1u : 0u;
    o[12] = uint(r);
    o[13] = uint(r >> 32);
}
