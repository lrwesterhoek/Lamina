//  Standard-DTFE volume-averaged (method 1) grid interpolation, Metal compute kernels mirroring averaged_interpolation_1.cc:
//  depositAveraged1 (a thread per tet), depositAveraged1Items (a thread per work item, with a cell cache) and
//  depositExactAverage (--exact). Positions are relative to the region's lower corner; periodicity is pre-baked.

#include <metal_stdlib>
using namespace metal;

// Which fields the item kernels deposit, fixed per pipeline: the compiler drops the unused velocity and gradient code
// and registers (the item kernel is register-bound), which buys occupancy.
constant bool FC_DEN  [[function_constant(0)]];
constant bool FC_VEL  [[function_constant(1)]];
constant bool FC_GRAD [[function_constant(2)]];
// the scalar field (ONE component, NO_SCALARS=1 builds) and its gradient, deposited like the density and the velocity
// gradient: buffer 13 holds the four vertex values; grids: buffer 14 (nCell) and 15 (nCell*3, [flat*3+i] = ds/dx_i).
constant bool FC_SCAL  [[function_constant(3)]];
constant bool FC_SGRAD [[function_constant(4)]];
// The exact sums (P.fixedSum): the item pipelines are specialized on the mode too (one kernel with both paths would take
// registers from the float one). Fixed-point sums make the grids independent of the order the atomics land in; the item
// kernel's float pre-sums (SIMD reduction, threadgroup table) stay fixed because the host starts every chunk at a
// multiple of 256 items (dtfe_metal_host.cc runBatch) and every value is converted where it leaves a thread or SIMD
// group. Scales: src/CGAL_triangulation/dtfe_gpu_fixed.h batchFixedScales; P.fixedSum 2: RunUnits (src/fixed_units.h).
constant bool FC_FIXED [[function_constant(5)]];

// The parameter block (buffer 10), one batch = one sub-domain's tets: DTFEDepositParams (dtfe_deposit_params.h), the
// hosts' own struct, prepended to this source by the Makefile so kernels and hosts cannot drift apart.
typedef DTFEDepositParams DepositParams;

// The accumulated grids of one dispatch, float and fixed-point views of the same buffers (one of the two
// is used, by P.fixedSum). Component c of flat cell f is at [f*comps + c]; a fixed-point component holds
// two uint words, [2*(f*comps + c)] low and [+1] high (P.fixedSum 1), or four limbs, [4*(f*comps + c)] lowest to
// [+3] highest (P.fixedSum 2, fixed_words.metal.inc).
struct DGrids {
    device atomic_float* den;  device atomic_float* vel;  device atomic_float* grad;
    device atomic_float* scal; device atomic_float* sg;
    device atomic_uint*  denX; device atomic_uint*  velX; device atomic_uint*  gradX;
    device atomic_uint*  scalX; device atomic_uint* sgX;
    device atomic_uint*  bad;   // the non-finite guard's counters, one word per grid (buffer 21, fxq)
};

// A share in fixed-point units: rint(v * scale), exact since scale is 2^k. A product not below 2^63 in magnitude (NaN,
// inf, overflow) has no int64: it gives 0, which every add skips, and is counted in its grid's word of 'bad'
// (ps_common.metal.inc fixedShareOk is the same bit test, which fast math cannot fold away).
static inline ulong fxq(float v, float scale, device atomic_uint* bad, uint slot) {
    float const p = v * scale;
    if ((as_type<uint>(p) & 0x7fffffffu) < 0x5f000000u) return as_type<ulong>(long(rint(p)));
    atomic_fetch_add_explicit(&bad[slot], 1u, memory_order_relaxed);
    return 0ul;
}

// Adds a 64-bit two's-complement integer, given as (low, high) words, to the two-word cell 'at': the low word's carry
// is read off the value its atomic returns, the high words add modulo 2^32, so the sum is exact in any order. The
// words stay separate: rebuilding a ulong from two threadgroup atomic loads crashed the Metal compiler service.
static inline void addWordsLH(device atomic_uint* g, ulong at, uint lo, uint hi) {
    if (lo == 0u && hi == 0u) return;
    uint const old = atomic_fetch_add_explicit(&g[at * 2ul], lo, memory_order_relaxed);
    uint const up = hi + ((old + lo < old) ? 1u : 0u);
    if (up != 0u)
        atomic_fetch_add_explicit(&g[at * 2ul + 1ul], up, memory_order_relaxed);
}
static inline void addWords(device atomic_uint* g, ulong at, ulong q) {
    addWordsLH(g, at, uint(q), uint(q >> 32));
}
// ... and to a cell of the threadgroup table (tableAdd)
static inline void addWordsTg(threadgroup atomic_uint* g, uint at, ulong q) {
    if (q == 0ul) return;
    uint const lo = uint(q), hi = uint(q >> 32);
    uint const old = atomic_fetch_add_explicit(&g[at * 2u], lo, memory_order_relaxed);
    uint const up = hi + ((old + lo < old) ? 1u : 0u);
    if (up != 0u)
        atomic_fetch_add_explicit(&g[at * 2u + 1u], up, memory_order_relaxed);
}

// One share of a grid component: the fixed-point words when fx (P.fixedSum; FC_FIXED in the item pipelines),
// else the float atomic.
// at = flat * comps + component.
static inline void addDen(thread const DGrids& G, constant DepositParams& P, bool fx, ulong at, float v) {
    if (fx) addWords(G.denX, at, fxq(v, P.fxScale[0], G.bad, 0u));
    else atomic_fetch_add_explicit(&G.den[at], v, memory_order_relaxed);
}
static inline void addVel(thread const DGrids& G, constant DepositParams& P, bool fx, ulong at, float v) {
    if (fx) addWords(G.velX, at, fxq(v, P.fxScale[1], G.bad, 1u));
    else atomic_fetch_add_explicit(&G.vel[at], v, memory_order_relaxed);
}
static inline void addGrad(thread const DGrids& G, constant DepositParams& P, bool fx, ulong at, float v) {
    if (fx) addWords(G.gradX, at, fxq(v, P.fxScale[2], G.bad, 2u));
    else atomic_fetch_add_explicit(&G.grad[at], v, memory_order_relaxed);
}
static inline void addScal(thread const DGrids& G, constant DepositParams& P, bool fx, ulong at, float v) {
    if (fx) addWords(G.scalX, at, fxq(v, P.fxScale[3], G.bad, 3u));
    else atomic_fetch_add_explicit(&G.scal[at], v, memory_order_relaxed);
}
static inline void addSG(thread const DGrids& G, constant DepositParams& P, bool fx, ulong at, float v) {
    if (fx) addWords(G.sgX, at, fxq(v, P.fxScale[4], G.bad, 4u));
    else atomic_fetch_add_explicit(&G.sg[at], v, memory_order_relaxed);
}

// One share of grid slot g into the run-wide 128-bit words (P.fixedSum == 2): its integer in the run's unit
// 1/P.fxScale[g] (shareLimbs), added limb by limb at component 'at'. A share with no integer below the grid's cap
// (P.fxCapBits[g]) adds nothing and is counted in 'bad'; below the cap no interleaving can wrap a cell.
static inline void addRun(device atomic_uint* x, device atomic_uint* bad, constant DepositParams& P, uint g,
                          ulong at, float v) {
    uint a[4];
    bool neg = false;
    if (shareLimbs(v, P.fxScale[g], P.fxCapBits[g], a, neg)) addLimbs(x, at, a, neg);
    else atomic_fetch_add_explicit(&bad[g], 1u, memory_order_relaxed);
}

// full-grid flat index of a sub-grid cell (the host guarantees fullGrid cells < 2^32)
static inline uint fullFlat(constant DepositParams& P, thread const int pos[3]) {
    return (uint(pos[0] + P.subOff[0]) * uint(P.fullGrid[1]) + uint(pos[1] + P.subOff[1])) * uint(P.fullGrid[2])
         + uint(pos[2] + P.subOff[2]);
}

static inline float det3(thread const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Inverse of 3x3, mirroring the CPU matrixInverse(): |det| <= 1e-6 x the product of the row
// lengths (RELATIVE, unit-free -- see math_functions.h) -> ZERO matrix. The CPU then
// interpolates a constant field from the base vertex; it does NOT drop the tetrahedron, so
// neither do we (unlike the PS deposit, which conserves mass and drops).
static inline void inverse3zero(thread const float A[3][3], thread float inv[3][3]) {
    float d = det3(A);
    float rn = 1.0f;
    for (int i=0; i<3; ++i) rn *= sqrt(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
    if (!(fabs(d) > 1.0e-6f * rn)) {
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) inv[i][j]=0.0f;
        return;
    }
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
}

// One deposit of (den, vel, gradFlat) * w into grid cell 'flat'. Offsets are 64-bit BECAUSE
// flat*9 WOULD overflow a 32-bit uint above ~782^3 (as in the PS deposit). The 2^32-cell cap
// on 'flat' is guarded in averaged_interpolation_1.cc. P.fixedSum == 2 (uniform per dispatch: the exact batches)
// takes the run-wide 128-bit words, else the 64-bit words (1) or the float atomics (0).
static inline void depositCell(thread const DGrids& G,
                               constant DepositParams& P,
                               ulong flat, float den,
                               thread const float vel[3],
                               thread const float gradFlat[9],
                               float sc, thread const float sG[3], float w)
{
    if (P.fixedSum == 2u) {
        if (P.fDen)
            addRun(G.denX, G.bad, P, 0u, flat, den*w);
        if (P.fVel)
            for (int j=0;j<3;++j)
                addRun(G.velX, G.bad, P, 1u, flat*3ul+ulong(j), vel[j]*w);
        if (P.fGrad)
            for (int q=0;q<9;++q)
                addRun(G.gradX, G.bad, P, 2u, flat*9ul+ulong(q), gradFlat[q]*w);
        if (P.fScal)
            addRun(G.scalX, G.bad, P, 3u, flat, sc*w);
        if (P.fSGrad)
            for (int i=0;i<3;++i)
                addRun(G.sgX, G.bad, P, 4u, flat*3ul+ulong(i), sG[i]*w);
        return;
    }
    if (P.fDen)
        addDen(G, P, P.fixedSum != 0u, flat, den*w);
    if (P.fVel)
        for (int j=0;j<3;++j)
            addVel(G, P, P.fixedSum != 0u, flat*3ul+ulong(j), vel[j]*w);
    if (P.fGrad)
        for (int q=0;q<9;++q)
            addGrad(G, P, P.fixedSum != 0u, flat*9ul+ulong(q), gradFlat[q]*w);
    if (P.fScal)
        addScal(G, P, P.fixedSum != 0u, flat, sc*w);
    if (P.fSGrad)
        for (int i=0;i<3;++i)
            addSG(G, P, P.fixedSum != 0u, flat*3ul+ulong(i), sG[i]*w);
}

kernel void depositAveraged1(
    device const float*     verts    [[buffer(0)]],   // nTet*12: 4 vertices, relative to the base cell's corner (cells) or the region's
    device const float*     dens     [[buffer(1)]],   // nTet*4:  vertex densities
    device const float*     vels     [[buffer(2)]],   // nTet*12: vertex velocities (dummy when fVel==fGrad==0)
    device const float*     vols     [[buffer(3)]],   // nTet:    tetrahedron volume (CPU value)
    device const uint*      cnts     [[buffer(4)]],   // nTet:    sample count; 0 = single-grid-cell fast path
    device const uint*      flats    [[buffer(5)]],   // nTet:    FULL-grid flat index for the fast path (subOff is 0: the grid's own index)
    device const int*       cells    [[buffer(22)]],  // nTet*3:  each tetrahedron's base cell (sub-grid indices; its positions are relative to that cell's corner): 0 under the region frame
    device const float4*    sobol    [[buffer(6)]],   // maxNN: shared Sobol barycentric table (xyz, pad)
    device atomic_float*    denGrid  [[buffer(7)]],
    device atomic_float*    velGrid  [[buffer(8)]],
    device atomic_float*    gradGrid [[buffer(9)]],
    constant DepositParams& P        [[buffer(10)]],
    device const float*     scal     [[buffer(13)]],   // nTet*4: vertex scalars (dummy when fScal==fSGrad==0)
    device atomic_float*    scalGrid [[buffer(14)]],
    device atomic_float*    sgGrid   [[buffer(15)]],
    device atomic_uint*     denX     [[buffer(16)]],   // the same grids as fixed-point words (P.fixedSum)
    device atomic_uint*     velX     [[buffer(17)]],
    device atomic_uint*     gradX    [[buffer(18)]],
    device atomic_uint*     scalX    [[buffer(19)]],
    device atomic_uint*     sgX      [[buffer(20)]],
    device atomic_uint*     badX     [[buffer(21)]],   // the non-finite guard's counters (DGrids.bad)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= P.nTet) return;
    DGrids const G = { denGrid, velGrid, gradGrid, scalGrid, sgGrid, denX, velX, gradX, scalX, sgX, badX };

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];   // vertex-difference matrix, rows = vertex k+1 - vertex 0
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];

    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = (P.fVel != 0) || (P.fGrad != 0);
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3];                       // vG[i][j] = d(v_j)/d(x_i), like the CPU velocityGrad
    float gradFlat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
        for (int j=0;j<3;++j)             // CPU velocityGradient() layout: out[j*3+i] = vG[i][j]
            for (int i=0;i<3;++i)
                gradFlat[j*3+i] = vG[i][j];
    }
    // the scalar: linear over the tetrahedron, its gradient constant (the CPU scalarGrad())
    bool const needScal = (P.fScal != 0) || (P.fSGrad != 0);
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (needScal) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    float const vol = vols[tid];
    uint  const n   = cnts[tid];

    if (n == 0u) {   // whole tetrahedron inside one (in-range) grid cell: centroid value * volume
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (P.fVel) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (P.fScal) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        depositCell(G, P, ulong(flats[tid]), denC, velC, gradFlat, scC, sG, vol);
        return;
    }

    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    float const factor = vol / float(n);
    for (uint s=0; s<n; ++s) {
        float4 const q = sobol[s];
        float const q0 = q.x, q1 = q.y, q2 = q.z;
        float p[3];   // sample point relative to the base vertex (CPU quasiRandomPointsInCell)
        for (int d=0;d<3;++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0;d<3;++d) {
            pos[d] = cells[tid*3+d] + int(floor((base[d]+p[d])*P.invDx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;   // sample outside this batch's sub-grid
        ulong flat = ulong(fullFlat(P, pos));
        float denS = den0;
        for (int i=0;i<3;++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*p[i];
            }
        float scS = s0;
        if (P.fScal) for (int i=0;i<3;++i) scS += p[i]*sG[i];
        depositCell(G, P, flat, denS, velS, gradFlat, scS, sG, factor);
    }
}


// Work-item kernel: thread = (tetrahedron, samples [s0, min(n, s0+itemSamples))), with a per-thread cache of K slots
// (flat cell, sums, count), each flushed with one atomic per field. The slot arrays are indexed only inside fully
// unrolled loops with a predicate, so they stay in registers. Sums are weighted by V_tet/n once, at the flush; the
// velocity gradient (constant over the tet) gets vG * (count * factor).
template <int K>
struct CellCache {
    uint  flat[K];
    float den[K];
    float vx[K], vy[K], vz[K];
    float sc[K];        // the scalar's sum (FC_SCAL)
    float cnt[K];
};

// vG[i][j] = d(v_j)/d(x_i); the CPU velocityGradient() layout is out[j*3+i] = vG[i][j]
template <int K>
static inline void flushSlot(thread const DGrids& G, constant DepositParams& P,
                             thread const CellCache<K>& C, int k, float factor,
                             thread const float vG[3][3], thread const float sG[3])
{
    // gather slot k with predicated selects (keeps the cache in registers)
    uint  flat = 0u; float den = 0.0f, vx = 0.0f, vy = 0.0f, vz = 0.0f, sc = 0.0f, cnt = 0.0f;
    for (int j = 0; j < K; ++j) {
        bool const m = (j == k);
        flat = m ? C.flat[j] : flat;
        den  = m ? C.den[j]  : den;
        vx   = m ? C.vx[j]   : vx;
        vy   = m ? C.vy[j]   : vy;
        vz   = m ? C.vz[j]   : vz;
        sc   = m ? C.sc[j]   : sc;
        cnt  = m ? C.cnt[j]  : cnt;
    }
    // (the same products, in the same float operations, as the table path's tableAdd: an eviction adds the
    // same integers whichever of the two takes it, so the exact sums do not depend on the table's fill)
    ulong const f = ulong(flat);
    if (P.fDen)
        addDen(G, P, FC_FIXED, f, den * factor);
    if (P.fVel) {
        addVel(G, P, FC_FIXED, f*3ul+0ul, vx * factor);
        addVel(G, P, FC_FIXED, f*3ul+1ul, vy * factor);
        addVel(G, P, FC_FIXED, f*3ul+2ul, vz * factor);
    }
    if (FC_SCAL)
        addScal(G, P, FC_FIXED, f, sc * factor);
    if (P.fGrad) {
        float const w = cnt * factor;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                addGrad(G, P, FC_FIXED, f*9ul+ulong(j*3+i), vG[i][j] * w);
    }
    if (FC_SGRAD) {
        float const w = cnt * factor;
        for (int i = 0; i < 3; ++i)
            addSG(G, P, FC_FIXED, f*3ul+ulong(i), sG[i] * w);
    }
}

// Work items are implicit (no per-item buffer). itemStart[t] is the first item of tet t (a prefix sum of
// max(1, ceil(cnt/itemSamples))); blockTet[iid >> ITEM_BLOCK_SHIFT] the tet holding that block's first item. An item's
// tet is the last t in [blockTet[blk], blockTet[blk+1]] with itemStart[t] <= iid (binary search). Any tet order works;
// the host uses a Morton curve so neighbouring threads deposit into neighbouring cells.
#define ITEM_BLOCK_SHIFT 5
// -> (tetrahedron, index of this item among the tetrahedron's items, items of the tetrahedron)
static inline uint3 itemOf(device const uint* itemStart, device const uint* blockTet, uint nTet, uint iid) {
    uint const blk = iid >> ITEM_BLOCK_SHIFT;
    uint lo = blockTet[blk];
    uint hi = min(blockTet[blk + 1u], nTet - 1u);
    while (hi > lo) {
        uint const mid = (lo + hi + 1u) >> 1;
        if (itemStart[mid] <= iid) lo = mid; else hi = mid - 1u;
    }
    uint const first = itemStart[lo];
    return uint3(lo, iid - first, itemStart[lo + 1u] - first);
}

// Threadgroup cell table: where the per-thread cache's EVICTIONS go. A void tet's samples rarely repeat a cell within
// one item but do across its items, which share a threadgroup: the table (open addressing, 32-bit keys claimed by
// compare-exchange) collects them, the threadgroup flushes it once; a full probe falls back to flushSlot's atomics.
// Entry: key, den, vx, vy, vz, [g0..g8]. Under FC_FIXED the values are two-word integers (tValsX: value v at words
// [2v], [2v+1]; tVals a dummy), so the exact grids do not inherit the order a float table sums the evictions in.
#define TABLE_EMPTY 0xffffffffu
#define TABLE_PROBES 8u

// floats per entry: den, vx, vy, vz, [scalar], [9 gradient], [3 scalar gradient]
static inline uint tableOffScal()  { return 4u; }
static inline uint tableOffGrad()  { return 4u + (FC_SCAL ? 1u : 0u); }
static inline uint tableOffSGrad() { return tableOffGrad() + (FC_GRAD ? 9u : 0u); }
static inline uint tableStride()   { return tableOffSGrad() + (FC_SGRAD ? 3u : 0u); }

static inline bool tableAdd(threadgroup atomic_uint* tKeys, threadgroup atomic_float* tVals,
                            threadgroup atomic_uint* tValsX, uint E, constant DepositParams& P, device atomic_uint* bad,
                            uint key, float den, float vx, float vy, float vz, float sc, float w,
                            thread const float vG[3][3], thread const float sG[3])
{
    uint h = (key * 2654435761u) >> 7;
    h &= (E - 1u);
    for (uint probe = 0u; probe < TABLE_PROBES; ++probe) {
        uint k = atomic_load_explicit(&tKeys[h], memory_order_relaxed);
        if (k == TABLE_EMPTY) {
            uint expected = TABLE_EMPTY;
            if (!atomic_compare_exchange_weak_explicit(&tKeys[h], &expected, key, memory_order_relaxed, memory_order_relaxed))
                k = expected;         // somebody else claimed it meanwhile: 'expected' holds their key
            else
                k = key;
        }
        if (k == key) {
            uint const base = h * tableStride();
            if (FC_FIXED) {   // the integers flushSlot would add (the same float products, converted here)
                if (FC_DEN) addWordsTg(tValsX, base + 0u, fxq(den, P.fxScale[0], bad, 0u));
                if (FC_VEL) {
                    addWordsTg(tValsX, base + 1u, fxq(vx, P.fxScale[1], bad, 1u));
                    addWordsTg(tValsX, base + 2u, fxq(vy, P.fxScale[1], bad, 1u));
                    addWordsTg(tValsX, base + 3u, fxq(vz, P.fxScale[1], bad, 1u));
                }
                if (FC_SCAL) addWordsTg(tValsX, base + tableOffScal(), fxq(sc, P.fxScale[3], bad, 3u));
                if (FC_GRAD)
                    for (int j = 0; j < 3; ++j)
                        for (int i = 0; i < 3; ++i)
                            addWordsTg(tValsX, base + tableOffGrad() + uint(j*3+i), fxq(vG[i][j] * w, P.fxScale[2], bad, 2u));
                if (FC_SGRAD)
                    for (int i = 0; i < 3; ++i)
                        addWordsTg(tValsX, base + tableOffSGrad() + uint(i), fxq(sG[i] * w, P.fxScale[4], bad, 4u));
                return true;
            }
            if (FC_DEN) atomic_fetch_add_explicit(&tVals[base + 0u], den, memory_order_relaxed);
            if (FC_VEL) {
                atomic_fetch_add_explicit(&tVals[base + 1u], vx, memory_order_relaxed);
                atomic_fetch_add_explicit(&tVals[base + 2u], vy, memory_order_relaxed);
                atomic_fetch_add_explicit(&tVals[base + 3u], vz, memory_order_relaxed);
            }
            if (FC_SCAL) atomic_fetch_add_explicit(&tVals[base + tableOffScal()], sc, memory_order_relaxed);
            if (FC_GRAD)
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i)
                        atomic_fetch_add_explicit(&tVals[base + tableOffGrad() + uint(j*3+i)], vG[i][j] * w, memory_order_relaxed);
            if (FC_SGRAD)
                for (int i = 0; i < 3; ++i)
                    atomic_fetch_add_explicit(&tVals[base + tableOffSGrad() + uint(i)], sG[i] * w, memory_order_relaxed);
            return true;
        }
        h = (h + 1u) & (E - 1u);
    }
    return false;
}

// The item kernel's body, templated on the cache size K (a function constant cannot size an array); the kernel
// instantiates K = 2, as it is register-bound.
template <int K>
static inline void depositItem(
    device const float*     verts, device const float* dens, device const float* vels,
    device const float*     vols,  device const uint*  cnts, device const uint*  flats,
    device const int*       cells, device const float4* sobol,
    thread const DGrids&    G,
    device const float*     scal,
    constant DepositParams& P, uint3 const it, uint const lane,
    threadgroup atomic_uint* tKeys, threadgroup atomic_float* tVals, threadgroup atomic_uint* tValsX)
{
    uint  const tid    = it.x;
    uint  const sub    = it.y;      // this item's index among the tetrahedron's items
    uint  const perTet = it.z;      // the tetrahedron's items: this one takes samples sub, sub+perTet, ...

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];

    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = FC_VEL || FC_GRAD;
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3] = {{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f}};   // vG[i][j] = d(v_j)/d(x_i)
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
    }

    // the scalar: linear over the tetrahedron, its gradient constant
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (FC_SCAL || FC_SGRAD) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    float const vol = vols[tid];
    uint  const n   = cnts[tid];

    CellCache<K> C;
    for (int k = 0; k < K; ++k) { C.flat[k] = 0u; C.den[k] = 0.0f; C.vx[k] = 0.0f; C.vy[k] = 0.0f; C.vz[k] = 0.0f; C.sc[k] = 0.0f; C.cnt[k] = 0.0f; }
    int used = 0;      // slots holding a cell
    int victim = 0;    // round-robin replacement pointer
    float factor;      // weight of the slot sums: V/n per sample, or V for the single-cell centroid

    if (n == 0u) {   // single-cell tetrahedron: its only item deposits the centroid value * volume
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (FC_VEL) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (FC_SCAL) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        C.flat[0] = flats[tid]; C.den[0] = denC; C.vx[0] = velC[0]; C.vy[0] = velC[1]; C.vz[0] = velC[2]; C.sc[0] = scC; C.cnt[0] = 1.0f;
        used = 1;
        factor = vol;
    } else {
    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    factor = vol / float(n);

    // Samples are dealt to a tet's items round-robin, so consecutive lanes read consecutive table entries.
    for (uint s = sub; s < n; s += perTet) {
        float4 const q = sobol[s];
        float const q0 = q.x, q1 = q.y, q2 = q.z;
        float p[3];
        for (int d=0;d<3;++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0;d<3;++d) {
            pos[d] = cells[tid*3+d] + int(floor((base[d]+p[d])*P.invDx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;
        uint const flat = fullFlat(P, pos);
        float denS = den0;
        for (int i=0;i<3;++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (FC_VEL)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*p[i];
            }
        float scS = s0;
        if (FC_SCAL) for (int i=0;i<3;++i) scS += p[i]*sG[i];

        int hit = -1;
        for (int k = 0; k < K; ++k)
            hit = (k < used && C.flat[k] == flat) ? k : hit;
        if (hit < 0) {
            if (used < K) { hit = used; ++used; }
            else {
                hit = victim;
                victim = (victim + 1 == K) ? 0 : victim + 1;
                // gather the victim slot (predicated, keeps the cache in registers)
                uint  vKey = 0u; float vDen = 0.0f, vVx = 0.0f, vVy = 0.0f, vVz = 0.0f, vSc = 0.0f, vCnt = 0.0f;
                for (int j = 0; j < K; ++j) {
                    bool const mm = (j == hit);
                    vKey = mm ? C.flat[j] : vKey; vDen = mm ? C.den[j] : vDen; vVx = mm ? C.vx[j] : vVx;
                    vVy = mm ? C.vy[j] : vVy; vVz = mm ? C.vz[j] : vVz; vSc = mm ? C.sc[j] : vSc; vCnt = mm ? C.cnt[j] : vCnt;
                }
                if (P.tableEntries == 0u ||
                    !tableAdd(tKeys, tVals, tValsX, P.tableEntries, P, G.bad, vKey, vDen * factor, vVx * factor, vVy * factor,
                              vVz * factor, vSc * factor, vCnt * factor, vG, sG))
                    flushSlot<K>(G, P, C, hit, factor, vG, sG);
            }
            for (int k = 0; k < K; ++k) {       // (re)initialise slot 'hit'
                bool const m = (k == hit);
                C.flat[k] = m ? flat : C.flat[k];
                C.den[k]  = m ? 0.0f : C.den[k];
                C.vx[k]   = m ? 0.0f : C.vx[k];
                C.vy[k]   = m ? 0.0f : C.vy[k];
                C.vz[k]   = m ? 0.0f : C.vz[k];
                C.sc[k]   = m ? 0.0f : C.sc[k];
                C.cnt[k]  = m ? 0.0f : C.cnt[k];
            }
        }
        for (int k = 0; k < K; ++k) {           // accumulate into slot 'hit'
            bool const m = (k == hit);
            C.den[k] = m ? C.den[k] + denS    : C.den[k];
            if (FC_VEL) {
                C.vx[k]  = m ? C.vx[k]  + velS[0] : C.vx[k];
                C.vy[k]  = m ? C.vy[k]  + velS[1] : C.vy[k];
                C.vz[k]  = m ? C.vz[k]  + velS[2] : C.vz[k];
            }
            if (FC_SCAL)
                C.sc[k]  = m ? C.sc[k]  + scS     : C.sc[k];
            C.cnt[k] = m ? C.cnt[k] + 1.0f    : C.cnt[k];
        }
    }
    }   // sampled path

    // ---- flush with a SIMD-group reduction by cell: per slot, elect the smallest pending key, simd_sum the pre-weighted
    // values of every lane holding it and let one lane add them. Control flow is uniform (simd_any); lanes that exited
    // early are inactive. Under FC_FIXED the leader converts its SIMD sums to integers (a SIMD group is always the same
    // 32 items: the host aligns the chunks).
    for (int k = 0; k < K; ++k) {
        bool  pend = k < used;
        uint  key  = pend ? C.flat[k] : 0xffffffffu;
        float const dW = C.den[k] * factor;
        float const xW = C.vx[k] * factor, yW = C.vy[k] * factor, zW = C.vz[k] * factor;
        float const cW = C.sc[k] * factor;
        float const gW = C.cnt[k] * factor;           // the gradients' weight: they are constant over the tetrahedron
        while (simd_any(pend)) {
            uint const leaderKey = simd_min(pend ? key : 0xffffffffu);
            bool const m = pend && key == leaderKey;
            uint const leader = simd_min(m ? lane : 32u);
            float const sd = FC_DEN ? simd_sum(m ? dW : 0.0f) : 0.0f;
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
            if (FC_VEL) {
                sx = simd_sum(m ? xW : 0.0f);
                sy = simd_sum(m ? yW : 0.0f);
                sz = simd_sum(m ? zW : 0.0f);
            }
            float const ss = FC_SCAL ? simd_sum(m ? cW : 0.0f) : 0.0f;
            if (lane == leader) {
                ulong const f = ulong(leaderKey);
                if (FC_DEN)
                    addDen(G, P, FC_FIXED, f, sd);
                if (FC_VEL) {
                    addVel(G, P, FC_FIXED, f*3ul+0ul, sx);
                    addVel(G, P, FC_FIXED, f*3ul+1ul, sy);
                    addVel(G, P, FC_FIXED, f*3ul+2ul, sz);
                }
                if (FC_SCAL)
                    addScal(G, P, FC_FIXED, f, ss);
            }
            if (FC_GRAD || FC_SGRAD) {
                // The gradient is a per-TETRAHEDRON constant times the cell weight, so it is reduced per
                // (cell, tetrahedron): one weight sum per tetrahedron sharing the cell (usually one or
                // two in a SIMD group) instead of nine component sums, and no nine-float array alive.
                bool mg = m;
                while (simd_any(mg)) {
                    uint const leaderTet = simd_min(mg ? tid : 0xffffffffu);
                    bool const mt = mg && tid == leaderTet;
                    uint const leaderG = simd_min(mt ? lane : 32u);
                    float const sw = simd_sum(mt ? gW : 0.0f);
                    if (lane == leaderG) {
                        ulong const f = ulong(leaderKey);
                        if (FC_GRAD)
                            for (int j = 0; j < 3; ++j)
                                for (int i = 0; i < 3; ++i)
                                    addGrad(G, P, FC_FIXED, f*9ul+ulong(j*3+i), vG[i][j] * sw);
                        if (FC_SGRAD)
                            for (int i = 0; i < 3; ++i)
                                addSG(G, P, FC_FIXED, f*3ul+ulong(i), sG[i] * sw);
                    }
                    mg = mg && !mt;
                }
            }
            pend = pend && !m;
        }
    }
}

// Adds value t of the fixed-point table (two words) into the device cell 'at', word by word (addWordsLH).
static inline void flushWords(device atomic_uint* g, ulong at, threadgroup atomic_uint* tValsX, uint t) {
    uint const lo = atomic_load_explicit(&tValsX[t * 2u], memory_order_relaxed);
    uint const hi = atomic_load_explicit(&tValsX[t * 2u + 1u], memory_order_relaxed);
    addWordsLH(g, at, lo, hi);
}

// Flushes the threadgroup table into the grids: one device atomic per field per used entry (two words each,
// integer to integer, under FC_FIXED).
static inline void tableFlush(threadgroup atomic_uint* tKeys, threadgroup atomic_float* tVals,
                              threadgroup atomic_uint* tValsX, uint E, uint ltid, uint tgSize,
                              thread const DGrids& G)
{
    for (uint e = ltid; e < E; e += tgSize) {
        uint const key = atomic_load_explicit(&tKeys[e], memory_order_relaxed);
        if (key == TABLE_EMPTY) continue;
        ulong const f = ulong(key);
        uint const base = e * tableStride();
        if (FC_FIXED) {
            if (FC_DEN)
                flushWords(G.denX, f, tValsX, base + 0u);
            if (FC_VEL)
                for (uint c = 0u; c < 3u; ++c)
                    flushWords(G.velX, f*3ul+ulong(c), tValsX, base + 1u + c);
            if (FC_SCAL)
                flushWords(G.scalX, f, tValsX, base + tableOffScal());
            if (FC_GRAD)
                for (uint q = 0u; q < 9u; ++q)
                    flushWords(G.gradX, f*9ul+ulong(q), tValsX, base + tableOffGrad() + q);
            if (FC_SGRAD)
                for (uint i = 0u; i < 3u; ++i)
                    flushWords(G.sgX, f*3ul+ulong(i), tValsX, base + tableOffSGrad() + i);
            continue;
        }
        if (FC_DEN)
            atomic_fetch_add_explicit(&G.den[f], atomic_load_explicit(&tVals[base + 0u], memory_order_relaxed), memory_order_relaxed);
        if (FC_VEL)
            for (uint c = 0u; c < 3u; ++c)
                atomic_fetch_add_explicit(&G.vel[f*3ul+ulong(c)], atomic_load_explicit(&tVals[base + 1u + c], memory_order_relaxed), memory_order_relaxed);
        if (FC_SCAL)
            atomic_fetch_add_explicit(&G.scal[f], atomic_load_explicit(&tVals[base + tableOffScal()], memory_order_relaxed), memory_order_relaxed);
        if (FC_GRAD)
            for (uint q = 0u; q < 9u; ++q)
                atomic_fetch_add_explicit(&G.grad[f*9ul+ulong(q)], atomic_load_explicit(&tVals[base + tableOffGrad() + q], memory_order_relaxed), memory_order_relaxed);
        if (FC_SGRAD)
            for (uint i = 0u; i < 3u; ++i)
                atomic_fetch_add_explicit(&G.sg[f*3ul+ulong(i)], atomic_load_explicit(&tVals[base + tableOffSGrad() + i], memory_order_relaxed), memory_order_relaxed);
    }
}

#define DTFE_ITEM_KERNEL(NAME, K)                                                           \
kernel void NAME(                                                                           \
    device const float*     verts    [[buffer(0)]],                                         \
    device const float*     dens     [[buffer(1)]],                                         \
    device const float*     vels     [[buffer(2)]],                                         \
    device const float*     vols     [[buffer(3)]],                                         \
    device const uint*      cnts     [[buffer(4)]],                                         \
    device const uint*      flats    [[buffer(5)]],                                         \
    device const int*       cells    [[buffer(22)]], /* each tetrahedron's base cell */     \
    device const float4*    sobol    [[buffer(6)]],                                         \
    device atomic_float*    denGrid  [[buffer(7)]],                                         \
    device atomic_float*    velGrid  [[buffer(8)]],                                         \
    device atomic_float*    gradGrid [[buffer(9)]],                                         \
    constant DepositParams& P        [[buffer(10)]],                                        \
    device const uint*      itemStart [[buffer(11)]], /* first item of each tetrahedron */  \
    device const uint*      blockTet [[buffer(12)]],  /* tetrahedron of each block's first */ \
    device const float*     scal     [[buffer(13)]],  /* nTet*4 vertex scalars (dummy when unused) */ \
    device atomic_float*    scalGrid [[buffer(14)]],                                        \
    device atomic_float*    sgGrid   [[buffer(15)]],                                        \
    device atomic_uint*     denX     [[buffer(16)]],  /* the grids' fixed-point words */    \
    device atomic_uint*     velX     [[buffer(17)]],                                        \
    device atomic_uint*     gradX    [[buffer(18)]],                                        \
    device atomic_uint*     scalX    [[buffer(19)]],                                        \
    device atomic_uint*     sgX      [[buffer(20)]],                                        \
    device atomic_uint*     badX     [[buffer(21)]],  /* the non-finite guard's counters */ \
    threadgroup atomic_uint*  tKeys  [[threadgroup(0)]],  /* cell table keys   (host-sized) */ \
    threadgroup atomic_float* tVals  [[threadgroup(1)]],  /* cell table values (host-sized) */ \
    threadgroup atomic_uint*  tValsX [[threadgroup(2)]],  /* ... as two-word integers (FC_FIXED) */ \
    uint iid [[thread_position_in_grid]],                                                   \
    uint lane [[thread_index_in_simdgroup]],                                                \
    uint ltid [[thread_position_in_threadgroup]],                                           \
    uint tgSize [[threads_per_threadgroup]])                                                \
{                                                                                           \
    uint const E = P.tableEntries;                                                          \
    DGrids const G = { denGrid, velGrid, gradGrid, scalGrid, sgGrid, denX, velX, gradX, scalX, sgX, badX }; \
    for (uint e = ltid; e < E; e += tgSize) {                                               \
        atomic_store_explicit(&tKeys[e], TABLE_EMPTY, memory_order_relaxed);                \
        if (FC_FIXED)                                                                       \
            for (uint c = 0u; c < 2u * tableStride(); ++c)                                  \
                atomic_store_explicit(&tValsX[e * 2u * tableStride() + c], 0u, memory_order_relaxed); \
        else                                                                                \
            for (uint c = 0u; c < tableStride(); ++c)                                       \
                atomic_store_explicit(&tVals[e * tableStride() + c], 0.0f, memory_order_relaxed); \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    if (iid < P.nTet)                                                                       \
        depositItem<K>(verts, dens, vels, vols, cnts, flats, cells, sobol, G, scal,         \
                       P, itemOf(itemStart, blockTet, P.nTetTotal, P.itemBase + iid), lane, \
                       tKeys, tVals, tValsX);                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    tableFlush(tKeys, tVals, tValsX, E, ltid, tgSize, G);                                   \
}

DTFE_ITEM_KERNEL(depositAveraged1Items2, 2)


// depositExactAverage: --exact on the GPU. One thread per ITEM = (tet, block of P.itemSamples consecutive cells of its
// bounding-box window), as the phase-space depositExactItems: each piece of the tet inside a cell deposits its centroid
// value times the piece volume (order-1 moments of sweep_moments), the gradients times the piece volume, as the CPU
// loop. No per-tet normalization, so one dispatch; a single-cell tet (cnts == 0) deposits centroid value * volume. The
// host bounds every window from its double-precision vertices; a surplus block idles. No cell cache.
kernel void depositExactAverage(
    device const float*     verts    [[buffer(0)]],
    device const float*     dens     [[buffer(1)]],
    device const float*     vels     [[buffer(2)]],
    device const float*     vols     [[buffer(3)]],
    device const uint*      cnts     [[buffer(4)]],
    device const uint*      flats    [[buffer(5)]],
    device atomic_float*    denGrid  [[buffer(7)]],
    device atomic_float*    velGrid  [[buffer(8)]],
    device atomic_float*    gradGrid [[buffer(9)]],
    constant DepositParams& P        [[buffer(10)]],
    device const uint*      itemStart [[buffer(11)]],
    device const uint*      blockTet [[buffer(12)]],
    device const float*     scal     [[buffer(13)]],
    device atomic_float*    scalGrid [[buffer(14)]],
    device atomic_float*    sgGrid   [[buffer(15)]],
    device atomic_uint*     denX     [[buffer(16)]],
    device atomic_uint*     velX     [[buffer(17)]],
    device atomic_uint*     gradX    [[buffer(18)]],
    device atomic_uint*     scalX    [[buffer(19)]],
    device atomic_uint*     sgX      [[buffer(20)]],
    device atomic_uint*     badX     [[buffer(21)]],
    device const int*       cells    [[buffer(22)]],   // nTet*3: each tetrahedron's base cell (sub-grid indices)
    uint iid [[thread_position_in_grid]])
{
    if (iid >= P.nTet) return;
    DGrids const G = { denGrid, velGrid, gradGrid, scalGrid, sgGrid, denX, velX, gradX, scalX, sgX, badX };
    uint3 const it = itemOf(itemStart, blockTet, P.nTetTotal, P.itemBase + iid);
    uint const tid = it.x, sub = it.y;

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];
    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = (P.fVel != 0) || (P.fGrad != 0);
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3] = {{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f}};
    float gradFlat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                gradFlat[j*3+i] = vG[i][j];
    }
    bool const needScal = (P.fScal != 0) || (P.fSGrad != 0);
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (needScal) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    if (cnts[tid] == 0u) {   // single-cell tetrahedron: the fast path, by its only item
        if (sub != 0u) return;
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (P.fVel) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (P.fScal) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        depositCell(G, P, ulong(flats[tid]), denC, velC, gradFlat, scC, sG, vols[tid]);
        return;
    }

    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    // the window of grid cells overlapped by the tetrahedron's bounding box, as LOCAL cell offsets c from its base
    // cell cb (the host's frame: the positions are relative to that cell's corner, so this arithmetic is the same
    // in every sub-domain), clamped so that the sub-grid index cb + c lies in the sub-grid (the CPU's iLo/iHi)
    int const cb[3] = { cells[tid*3+0], cells[tid*3+1], cells[tid*3+2] };
    int iLo[3], iHi[3];
    for (int d=0; d<3; ++d) {
        float lo = 0.0f, hi = 0.0f;
        for (int v=0; v<3; ++v) { lo = min(lo, Ax[v][d]); hi = max(hi, Ax[v][d]); }
        iLo[d] = int(floor((base[d] + lo) * P.invDx[d]));
        iHi[d] = int(floor((base[d] + hi) * P.invDx[d])) + 1;
        if (iLo[d] < -cb[d]) iLo[d] = -cb[d];
        if (iHi[d] > P.nGrid[d] - cb[d]) iHi[d] = P.nGrid[d] - cb[d];
    }
    ulong const n1 = ulong(max(iHi[1] - iLo[1], 0));
    ulong const n2 = ulong(max(iHi[2] - iLo[2], 0));
    ulong const n12 = n1 * n2;
    ulong const nW = ulong(max(iHi[0] - iLo[0], 0)) * n12;
    ulong w = ulong(sub) * ulong(P.itemSamples);
    if (w >= nW) return;             // the host's window bound has a margin: surplus blocks idle
    ulong const wEnd = (nW - w > ulong(P.itemSamples)) ? w + ulong(P.itemSamples) : nW;
    int a0 = int(w / n12);
    ulong const rem = w - ulong(a0) * n12;
    int a1 = int(rem / n2);
    int a2 = int(rem - ulong(a1) * n2);

    // the tetrahedron in its vertex-0 frame, positively oriented as the CPU hands it to r3d (a
    // negative one has vertices 1 and 2 swapped; the sweep and tet_planes take either orientation)
    float3 t1 = float3(Ax[0][0], Ax[0][1], Ax[0][2]);
    float3 t2 = float3(Ax[1][0], Ax[1][1], Ax[1][2]);
    float3 t3 = float3(Ax[2][0], Ax[2][1], Ax[2][2]);
    if (det3(Ax) < 0.0f) { float3 tt = t1; t1 = t2; t2 = tt; }
    TetPlanes const faces = tet_planes(t1, t2, t3);
    float3 const hcell = float3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
    float3 const tlo = min(min(float3(0.0f), t1), min(t2, t3));   // the tet's bbox (vertex-0 frame)
    float3 const thi = max(max(float3(0.0f), t1), max(t2, t3));

    // The item's cells in runs of 32: first the cheap test of every cell of the run (not wholly outside a face plane)
    // into a bit mask, then the clip of the cells that passed, in the same ascending order. A SIMD group then waits
    // for the clips of its busiest lane, not for every cell at which some lane clips.
    while (w < wEnd) {
        uint const nRun = uint(min(wEnd - w, 32ul));
        int const r0 = a0, r1 = a1, r2 = a2;   // the run's first cell
        uint hits = 0u;
        for (uint j = 0u; j < nRun; ++j) {
            int const c[3] = { iLo[0] + a0, iLo[1] + a1, iLo[2] + a2 };        // local cell offsets
            if (ulong(++a2) == n2) { a2 = 0; if (ulong(++a1) == n1) { a1 = 0; ++a0; } }
            // the cell in the vertex-0 frame: a cell wholly outside a face plane holds no piece
            float3 const ccen = float3((float(c[0]) + 0.5f) * P.dx[0] - base[0],
                                       (float(c[1]) + 0.5f) * P.dx[1] - base[1],
                                       (float(c[2]) + 0.5f) * P.dx[2] - base[2]);
            if (!box_outside_tet(faces, ccen, hcell)) hits |= 1u << j;
        }
        w += ulong(nRun);
        while (hits != 0u) {
            uint const j = ctz(hits);
            hits &= hits - 1u;
            // cell j of the run: the first cell stepped j times along the innermost axis, with its carries
            uint const s2 = uint(r2) + j;
            uint const c1 = uint(r1) + s2 / uint(n2);
            int const c[3] = { iLo[0] + r0 + int(c1 / uint(n1)), iLo[1] + int(c1 % uint(n1)), iLo[2] + int(s2 % uint(n2)) };
            int const pos[3] = { cb[0] + c[0], cb[1] + c[1], cb[2] + c[2] };          // sub-grid indices
            float3 const ccen = float3((float(c[0]) + 0.5f) * P.dx[0] - base[0],
                                       (float(c[1]) + 0.5f) * P.dx[1] - base[1],
                                       (float(c[2]) + 0.5f) * P.dx[2] - base[2]);
            // clipped in the OVERLAP frame (exact_clip.metal.inc: origin at the centre of the tet bbox's
            // overlap with the cell, where the piece lies -- no frame far from the piece cancels)
            float3 const clo = ccen - hcell, chi = ccen + hcell;
            float3 const org = 0.5f * (max(clo, tlo) + min(chi, thi));
            float mo[10];   // sweep_moments clears all 10; order 1 fills mo[0..3]
            sweep_moments(-org, t1 - org, t2 - org, t3 - org, clo - org, chi - org, false, mo);
            if (!(mo[0] > 0.0f)) continue;
            float const dV = mo[0];
            float const cen[3] = { org.x + mo[1]/mo[0], org.y + mo[2]/mo[0], org.z + mo[3]/mo[0] };   // the piece's centroid, vertex-0 frame
            float denS = den0;
            for (int i=0;i<3;++i) denS += cen[i]*densGrad[i];
            float velS[3] = {0.0f,0.0f,0.0f};
            if (P.fVel)
                for (int j=0;j<3;++j) {
                    velS[j]=u0[j];
                    for (int i=0;i<3;++i) velS[j] += vG[i][j]*cen[i];
                }
            float scS = s0;
            if (P.fScal) for (int i=0;i<3;++i) scS += cen[i]*sG[i];
            depositCell(G, P, ulong(fullFlat(P, pos)), denS, velS, gradFlat, scS, sG, dV);
        }
    }
}
