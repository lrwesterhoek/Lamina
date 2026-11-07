/* CUDA/HIP backend of the PS-DTFE GPU deposit (implements gpu_host.h). Compiled only in
   CUDA=1 (nvcc) or HIP=1 (hipcc, '-x hip') builds, which define PS_GPU for the callers.

   The kernels are a line-by-line port of metal/ps_deposit.metal -- depositFields (one thread
   per tetrahedron: the certified float inside test, the sampled two-pass deposit, the beyond-grid
   column counting, the centroid fallback and the one-thread exact deposit) and depositExactItems
   (the exact deposit split below the tetrahedron) with every helper, same float32 arithmetic,
   same deferral rule -- and the host mirrors ps_metal_host.cc: the same chunked dispatch
   controller, the same items pass + leftovers + deferrals, the same retry discipline. Where the
   Metal host reads its unified-memory buffers in place (the items' window bounds, the leftovers,
   the deferrals), this host keeps a host mirror of the masses and copies explicitly.

   The CPU deposit remains the double-precision reference; this backend carries the same
   parity contract as the Metal one (float rounding, atomic summation order). The MASS grid is,
   like Metal's, an exact 64-bit fixed-point sum (depositMass): it loses no small shares in busy
   cells and is bit-identical run to run -- and so is every moment grid (addMoment,
   DepositParams.fxMask: metal/ps_deposit.metal says why). It has been compiled and its CPU
   fallback exercised on Linux, but NOT yet validated on NVIDIA/AMD hardware -- see
   metal/README.md ("CUDA / HIP"). */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
extern std::atomic<int> psGpuRetries;   // ps_interpolation.cc (run_progress.h)
extern std::atomic<int> psGpuRetriesByCause[4];   // ... by cause: innocent, timeout, oom, other
#include <unistd.h>

#include "gpu_host.h"
#include "../message.h"       // ANSI colour helpers only (the retry warnings)
#include "gpu_cuda_compat.h"
#include "ps_deposit_params.h"
#include "ps_mass_audit.h"      // PS_MASS_AUDIT=1: the fixed-point grid's exact total (as ps_metal_host.cc)

namespace {

using DepositParams = PSDepositParams;   // the shared host struct (ps_deposit_params.h)

#include "gpu_exact_clip.cuh"   // V3, det3/inverse3, the exact tet-cell sweep (shared with dtfe_gpu_cuda.cu)

// ---- the per-tetrahedron record: PS_TET_STRIDE floats per tet, see ps_deposit_params.h /
// metal/ps_deposit.metal (ic0 as int bits, f0, the three edges, the centroid cell as uint bits)
#define TET_STRIDE PS_TET_STRIDE
#define U_F 5.9604645e-8f   // 2^-24, the float unit roundoff

// Band-certified classification of one sample (see metal/ps_deposit.metal for the derivation):
// +1 inside, -1 outside (both exact), 0 undecidable in float -> the caller defers the tet.
__device__ __forceinline__ int classifySample(const float C[3][3], const float M[3][3], float det, float sg,
                                              const float B[4], float Bdet, const float r[3],
                                              const float t[3], const DepositParams& P)
{
    float n[4];
    n[0] = C[0][0]*r[0] + C[0][1]*r[1] + C[0][2]*r[2];
    n[1] = C[1][0]*r[0] + C[1][1]*r[1] + C[1][2]*r[2];
    n[2] = C[2][0]*r[0] + C[2][1]*r[1] + C[2][2]*r[2];
    n[3] = det - (n[0] + n[1] + n[2]);
    bool sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -B[k]) return -1;
        if (x <= B[k]) sure = false;
    }
    if (sure) return 1;

    // stage 2: this sample's own bounds (7u|r_i| + rho_i = dx_i (u (6 + 12|t_i|) + 1e-9))
    float w[3];
    for (int i = 0; i < 3; ++i) w[i] = P.dx[i] * (U_F * (6.0f + 12.0f*fabsf(t[i])) + 1.0e-9f);
    float Bs[4];
    float nMag = 0.0f;
    for (int a = 0; a < 3; ++a) {
        Bs[a] = 2.0f * (M[a][0]*w[0] + M[a][1]*w[1] + M[a][2]*w[2]);
        nMag += M[a][0]*fabsf(r[0]) + M[a][1]*fabsf(r[1]) + M[a][2]*fabsf(r[2]);
    }
    Bs[3] = Bdet + Bs[0] + Bs[1] + Bs[2] + 2.0f * 3.0f * U_F * (fabsf(det) + nMag);
    sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -Bs[k]) return -1;
        if (x <= Bs[k]) sure = false;
    }
    return sure ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------
// The grids (see metal/ps_deposit.metal FieldGrids) and the per-sample deposit.
// ------------------------------------------------------------------------------------------------
struct FieldGrids {
    float*    mass;
    float*    mom;     // nCell*3   [flat*3+j]
    float*    m2;      // nCell*6   [flat*6+c]
    float*    grad;    // nCell*9   [flat*9+j*3+i]
    unsigned* streams;
    float*    momw;    // nCell     moment-weight normalizer (fVolW; else dummy)
    unsigned* caustic; // nCell     orientation bits, atomic OR (fCaustic; else dummy)
    float*    sv;      // nCell     exact multiplicity sum(V_int)/V_cell (fExact; else dummy)
    float*    dispvel; // nCell*3   dispersion's own mass-weighted mean (fVolW+fDisp; else dummy)
    float*    dispw;   // nCell     its normalizer (fVolW+fDisp; else dummy)
    float*    scal;    // nCell     scalar moment sum(wm s) (fScal; else dummy)
    float*    sgrad;   // nCell*3   scalar-gradient moment (fSGrad; else dummy)
    unsigned* massFx;  // nCell*2   the mass as a 64-bit fixed-point sum [flat*2] low, [flat*2+1] high (massFixed; else dummy)
    float     massScale; // its unit: 1/massScale mass
    bool      massFixed;
    // the moment grids as exact fixed-point sums (metal/ps_deposit.metal DepositParams.fxMask): bit g
    // = grid g's buffer holds two unsigned words per component (the float pointer above, reinterpreted -- the
    // Metal kernel takes the same buffer at a second index), in units of 1/fxScale[g]
    unsigned  fxMask;
    float     fxScale[PS_FX_N];
    // the non-finite guard's counters, one word per slot (FX_DROP_*; metal/ps_deposit.metal FieldGrids.bad)
    unsigned* bad;
};

// The non-finite guard of a fixed-point share (metal/ps_deposit.metal fixedShareOk, ps_deposit_params.h
// psFixedShareOk): a product not below 2^63 in magnitude -- NaN, inf, overflow -- has no int64 and adds nothing; it
// is counted in its slot. The float's bits are tested, as in Metal (immune to fast math).
__device__ __forceinline__ bool fixedShareOk(float p) { return (__float_as_uint(p) & 0x7fffffffu) < 0x5f000000u; }

// One 64-bit two's-complement integer q into the two-word cell 'at' of x (metal/ps_deposit.metal addFixedQ): the
// low word's carry is read off the value its atomic returns, the high words add modulo 2^32, so the words end at
// the exact integer total whatever the interleaving.
__device__ __forceinline__ void addFixedQ(unsigned* x, ull at, ull q)
{
    if (q == 0ull) return;                          // below half a unit: nothing to add
    unsigned const lo = unsigned(q), hi = unsigned(q >> 32);
    unsigned const old = atomicAdd(&x[at * 2ull], lo);
    unsigned const up = hi + ((old + lo < old) ? 1u : 0u);
    if (up != 0u)
        atomicAdd(&x[at * 2ull + 1ull], up);
}

// One share v of moment grid g at component index 'at' (metal/ps_deposit.metal addMoment): rint(v * scale) into
// the words when the grid is fixed-point, else a float atomic.
__device__ __forceinline__ void addMoment(float* f, const FieldGrids& G, unsigned g, ull at, float v)
{
    if ((G.fxMask >> g) & 1u) {
        float const p = v * G.fxScale[g];
        if (fixedShareOk(p)) addFixedQ(reinterpret_cast<unsigned*>(f), at, ull((long long)(rintf(p))));
        else atomicAdd(&G.bad[g], 1u);   // (the guard above)
    }
    else atomicAdd(&f[at], v);
}

// the FieldGrids' fixed-point fields from the parameters (both kernels)
__device__ __forceinline__ void setFixed(FieldGrids& G, const DepositParams& P)
{
    G.fxMask = P.fxMask;
    for (int g = 0; g < PS_FX_N; ++g) G.fxScale[g] = P.fxScale[g];
}

// A mass share into the mass grid (metal/ps_deposit.metal depositMass; DepositParams.massFixed
// there says why): the exact 64-bit fixed-point sum (massFixed), else a float atomic. Float atomics
// drop every share below half an ulp of a busy cell's running sum, always downward, and their result
// depends on the order the atomics land in. Two's-complement words with the carry done by hand, as in
// Metal (a native 64-bit atomicAdd would give the same bits; the two words keep the kernels line by line
// and the host decode, psMassFixedValue, shared): the low word's carry is read off the value its atomic
// returns (the add wrapped iff the result is below it), and the high words add modulo 2^32, so once every
// thread is done the two words hold the exact integer total whatever the interleaving. The host keeps the
// partition's whole mass below 2^62 units, so no share and no cell total overflows. massScale is 2^k, so
// the product is exact and rintf (round to nearest even, as Metal's rint) gives Metal's integer.
__device__ __forceinline__ void depositMass(const FieldGrids& G, unsigned flat, float w)
{
    if (!G.massFixed) {
        atomicAdd(&G.mass[flat], w);
        return;
    }
    // (64-bit cell offsets inside addFixedQ: a sub-grid may hold up to 2^32 cells)
    float const p = w * G.massScale;
    if (fixedShareOk(p)) addFixedQ(G.massFx, ull(flat), ull((long long)(rintf(p))));
    else atomicAdd(&G.bad[FX_DROP_MASS], 1u);   // (the non-finite guard)
}

// n equal shares v of moment grid g as ONE add of n * q (metal/ps_deposit.metal addMomentN): fixed-point grids only
__device__ __forceinline__ void addMomentN(float* f, const FieldGrids& G, unsigned g, ull at, float v, unsigned n)
{
    float const p = v * G.fxScale[g];
    if (fixedShareOk(p)) addFixedQ(reinterpret_cast<unsigned*>(f), at, ull((long long)(rintf(p))) * ull(n));
    else atomicAdd(&G.bad[g], n);   // (the guard above)
}

// n equal mass shares w as one add of n * q (metal/ps_deposit.metal depositMassN): massFixed only
__device__ __forceinline__ void depositMassN(const FieldGrids& G, unsigned flat, float w, unsigned n)
{
    float const p = w * G.massScale;
    if (fixedShareOk(p)) addFixedQ(G.massFx, ull(flat), ull((long long)(rintf(p))) * ull(n));
    else atomicAdd(&G.bad[FX_DROP_MASS], n);   // (the non-finite guard)
}

// batched: metal/ps_deposit.metal depositSample says which parts depositSampleBatch then adds once per (tet, cell)
__device__ __forceinline__ void depositSample(const FieldGrids& G, unsigned flat,
                                              const float rel[3], float w, float wm, unsigned obits,
                                              const float u0[3], const float vG[3][3],
                                              float s0, const float sG[3],
                                              bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                              bool fExact, float svShare, bool countStream, bool unresolved,
                                              bool fScal, bool fSGrad, bool batched)
{
    // Moment-grid offsets are 64-bit BECAUSE flat*9 WOULD overflow a 32-bit uint above ~782^3.
    ull base = ull(flat);
    if (!(batched && G.massFixed)) depositMass(G, flat, w);
    if (fVel || fDisp) {
        float vv[3];
        for (int j=0; j<3; ++j) {
            vv[j]=u0[j];
            for (int i=0;i<3;++i) vv[j]+=vG[i][j]*rel[i];
            if (fVel)
                addMoment(G.mom, G, PS_FX_MOM, base*3ull+ull(j), vv[j]*wm);
        }
        if (fDisp) {
            ull c=0;
            for (int i=0;i<3;++i)
                for (int j=i;j<3;++j)
                    addMoment(G.m2, G, PS_FX_M2, base*6ull + c++, w*vv[i]*vv[j]);
            if (fVolW) {
                for (int j=0;j<3;++j)
                    addMoment(G.dispvel, G, PS_FX_DV, base*3ull+ull(j), vv[j]*w);
                if (((G.fxMask >> PS_FX_DW) & 1u) == 0u)   // (else the mass grid is it)
                    atomicAdd(&G.dispw[flat], w);
            }
        }
    }
    if (fGrad && !(batched && ((G.fxMask >> PS_FX_GRAD) & 1u)))
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                addMoment(G.grad, G, PS_FX_GRAD, base*9ull + ull(j*3+i), vG[i][j]*wm);
    if (fScal) {   // the linear scalar profile at the sample, moment-weighted like the velocity
        float sv = s0;
        for (int i=0;i<3;++i) sv += sG[i]*rel[i];
        addMoment(G.scal, G, PS_FX_SCAL, base, sv*wm);
    }
    if (fSGrad && !(batched && ((G.fxMask >> PS_FX_SG) & 1u)))
        for (int i=0;i<3;++i)
            addMoment(G.sgrad, G, PS_FX_SG, base*3ull + ull(i), sG[i]*wm);
    if (fVolW && !(batched && ((G.fxMask >> PS_FX_MOMW) & 1u)))
        addMoment(G.momw, G, PS_FX_MOMW, base, wm);
    if (fCaustic && !batched)
        atomicOr(&G.caustic[flat], obits);
    if (fExact && !(batched && ((G.fxMask >> PS_FX_SV) & 1u)))
        addMoment(G.sv, G, PS_FX_SV, base, svShare);
    if (countStream && !batched)
        atomicAdd(&G.streams[flat], 1u);
    if (unresolved)
        atomicOr(&G.streams[flat], 0x80000000u);
}

// The position-independent parts of n batched samples in cell 'flat' (metal/ps_deposit.metal depositSampleBatch):
// one add of n * q per exact integer grid, n stream counts and one caustic OR -- the same words, counts and bits
__device__ __forceinline__ void depositSampleBatch(const FieldGrids& G, unsigned flat, unsigned n, float w, float wm,
                                                   unsigned obits, const float vG[3][3], const float sG[3],
                                                   bool fGrad, bool fVolW, bool fCaustic, bool fExact, float svShare,
                                                   bool fSGrad)
{
    ull base = ull(flat);
    if (G.massFixed) depositMassN(G, flat, w, n);
    if (fGrad && ((G.fxMask >> PS_FX_GRAD) & 1u))
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                addMomentN(G.grad, G, PS_FX_GRAD, base*9ull + ull(j*3+i), vG[i][j]*wm, n);
    if (fSGrad && ((G.fxMask >> PS_FX_SG) & 1u))
        for (int i=0;i<3;++i)
            addMomentN(G.sgrad, G, PS_FX_SG, base*3ull + ull(i), sG[i]*wm, n);
    if (fVolW && ((G.fxMask >> PS_FX_MOMW) & 1u))
        addMomentN(G.momw, G, PS_FX_MOMW, base, wm, n);
    if (fCaustic)
        atomicOr(&G.caustic[flat], obits);
    if (fExact && ((G.fxMask >> PS_FX_SV) & 1u))
        addMomentN(G.sv, G, PS_FX_SV, base, svShare, n);
    atomicAdd(&G.streams[flat], n);
}

// A RAW window index triple -> its flat sub-grid index (periodic wrap, then the partition
// sub-box, which may itself wrap a periodic axis). False when outside the grid or the sub-box.
// 0: outside the grid; 1: inside the grid but outside the sub-box (the SAMPLED deposit counts such a
// cell's samples under windowMode); 2: inside
__device__ __forceinline__ int window_flat(const int raw[3], const DepositParams& P, unsigned& flat) {
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
    flat = (unsigned(l[0])*unsigned(P.subDims[1]) + unsigned(l[1]))*unsigned(P.subDims[2]) + unsigned(l[2]);
    return 2;
}

// --ps-window, exact deposit: restrict a tet's raw span [lo, hi) along one axis to the hull of the
// window's periodic images it meets (gpu_host.h psWindowHull, which the host's item bound uses --
// keep in sync). False when the span misses the window.
__device__ __forceinline__ bool window_hull(int& lo, int& hi, int n, bool periodic, int o, int m) {
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
    V3     t1, t2, t3;       // edges from vertex 0, positively oriented (folded tet: 1 and 2 swapped)
    TetPlanes faces;
    V3     hcell;            // half a cell
    int    ic0[3];
    float  f0[3];
    float  u0[3], vG[3][3];
    float  vc[3], vh[3];     // the centre and half the range of each velocity component over the 4 vertices
                             // (metal/ps_deposit.metal ExactTet: the bounds exact_cell_deposit clamps to)
    float  s0, sG[3];        // the linear scalar (fScal / fSGrad; else zero)
    float  d0, dG[3];
    unsigned obits;
    float  invCellVol;
    bool   straddle;
    float  wFull;
    float  wScale;           // the unit of its fixed-point sumW (PS_FX_SUMW_BIT; metal/ps_deposit.metal ExactTet)
};

// The exact dispersion's float32 limit (metal/ps_deposit.metal, where the derivation is): a tet whose velocity
// gradient makes G^T Cov G cancel beyond EX_DISP_KAPPA2 for any component goes to the CPU's double deposit.
#ifndef EX_DISP_KAPPA2
#define EX_DISP_KAPPA2 1.0e4f
#endif
#ifndef EX_DISP_KAPPA2_L
#define EX_DISP_KAPPA2_L 1.0e10f
#endif
__device__ __forceinline__ void exact_vel_range(const float Ax[3][3], const float vG[3][3], const float u0[3],
                                                float vc[3], float vh[3])
{
    for (int j=0; j<3; ++j) {
        float lo = 0.0f, hi = 0.0f;   // vertex 0's offset is 0
        for (int e=0; e<3; ++e) {
            float const dv = Ax[e][0]*vG[0][j] + Ax[e][1]*vG[1][j] + Ax[e][2]*vG[2][j];
            lo = fminf(lo, dv); hi = fmaxf(hi, dv);
        }
        vc[j] = u0[j] + 0.5f*(lo + hi);
        vh[j] = 0.5f*(hi - lo);
    }
}
__device__ __forceinline__ bool exact_disp_illcond(const float Ax[3][3], const float vG[3][3], const float vh[3],
                                                   const float dx[3])
{
    float L = 0.0f;
    for (int d=0; d<3; ++d) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = fminf(lo, Ax[e][d]); hi = fmaxf(hi, Ax[e][d]); }
        L = fmaxf(L, hi - lo);
    }
    float const ell = fminf(L, sqrtf(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]));
    for (int j=0; j<3; ++j) {
        float const g2 = vG[0][j]*vG[0][j] + vG[1][j]*vG[1][j] + vG[2][j]*vG[2][j];
        float const r2 = 4.0f*vh[j]*vh[j];
        if (g2*ell*ell > EX_DISP_KAPPA2 * r2 || g2*L*L > EX_DISP_KAPPA2_L * r2) return true;
    }
    return false;
}

__device__ __forceinline__ float exact_full_weight(float vol, bool fLinear, float d0, const float dG[3],
                                                   V3 t1, V3 t2, V3 t3)
{
    if (!fLinear) return vol;
    V3 const c = scale3(add3(add3(t1, t2), t3), 0.25f);
    return vol * (d0 + dG[0]*c.x + dG[1]*c.y + dG[2]*c.z);
}

__device__ bool exact_cell_clip(const ExactTet& E, const int raw[3], const DepositParams& P,
                                bool full, bool fLinear, float mo[10], float& w, V3& cc)
{
    V3 const ccen = v3((float(raw[0] - E.ic0[0]) - E.f0[0] + 0.5f) * P.dx[0],
                       (float(raw[1] - E.ic0[1]) - E.f0[1] + 0.5f) * P.dx[1],
                       (float(raw[2] - E.ic0[2]) - E.f0[2] + 0.5f) * P.dx[2]);
    if (box_outside_tet(E.faces, ccen, E.hcell)) return false;

    // the tet ∩ cell moments in the OVERLAP frame (gpu_exact_clip.cuh: origin at the centre of the tet
    // bbox's overlap with the cell, where the piece lies), by the z-sweep (sweep_moments; order 2 only
    // when 'full' -- pass 0 needs V and the first moments, bit-identical to pass 1's); the moments come
    // out about that origin cc, which the caller adds back to the centroid
    V3 const clo = sub3(ccen, E.hcell), chi = add3(ccen, E.hcell);
    V3 const org = overlap_origin(ccen, E.hcell, E.t1, E.t2, E.t3);
    sweep_moments(neg3(org), sub3(E.t1, org), sub3(E.t2, org), sub3(E.t3, org), sub3(clo, org), sub3(chi, org), full, mo);
    if (!(mo[0] > 0.0f)) return false;

    cc = org;
    w = mo[0];
    if (fLinear) {
        w = (E.d0 + E.dG[0]*org.x + E.dG[1]*org.y + E.dG[2]*org.z)*mo[0]
          + E.dG[0]*mo[1] + E.dG[1]*mo[2] + E.dG[2]*mo[3];
        if (w < 0.0f) w = 0.0f;
    }
    return true;
}

__device__ void exact_cell_deposit(const FieldGrids& G, unsigned flat, const float mo[10],
                                   float massShare, const ExactTet& E,
                                   bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                   bool fScal, bool fSGrad, V3 cc)
{
    float const wmEx = fVolW ? mo[0] : massShare;
    ull const base = ull(flat);
    depositMass(G, flat, massShare);
    float const invV = 1.0f / mo[0];
    float const loc[3] = { mo[1]*invV, mo[2]*invV, mo[3]*invV };            // about the clip origin
    float const cen[3] = { cc.x + loc[0], cc.y + loc[1], cc.z + loc[2] };   // vertex-0 frame
    float vb[3];
    if (fVel || fDisp) {
        for (int j=0; j<3; ++j) {
            vb[j] = E.u0[j];
            for (int i=0; i<3; ++i) vb[j] += E.vG[i][j]*cen[i];
            vb[j] = fminf(fmaxf(vb[j], E.vc[j] - E.vh[j]), E.vc[j] + E.vh[j]);   // within the vertices' range
            if (fVel)
                addMoment(G.mom, G, PS_FX_MOM, base*3ull+ull(j), vb[j]*wmEx);
        }
        if (fDisp) {
            float cov[3][3];
            // (translation-invariant: taken about the clip origin, near the piece, where nothing cancels)
            cov[0][0] = mo[4]*invV - loc[0]*loc[0];
            cov[0][1] = cov[1][0] = mo[5]*invV - loc[0]*loc[1];
            cov[0][2] = cov[2][0] = mo[6]*invV - loc[0]*loc[2];
            cov[1][1] = mo[7]*invV - loc[1]*loc[1];
            cov[1][2] = cov[2][1] = mo[8]*invV - loc[1]*loc[2];
            cov[2][2] = mo[9]*invV - loc[2]*loc[2];
            // S = G^T Cov G clamped to what a linear field on a part of the tet can have (metal/ps_deposit.metal)
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
                S[a][a] = fminf(fmaxf(S[a][a], 0.0f), E.vh[a]*E.vh[a]);
                for (int b=a+1; b<3; ++b) S[a][b] = fminf(fmaxf(S[a][b], -E.vh[a]*E.vh[b]), E.vh[a]*E.vh[b]);
            }
            ull c2 = 0;
            for (int a=0; a<3; ++a)
                for (int b=a; b<3; ++b)
                    addMoment(G.m2, G, PS_FX_M2, base*6ull + c2++, massShare*(vb[a]*vb[b] + S[a][b]));
            if (fVolW) {
                for (int j=0;j<3;++j)
                    addMoment(G.dispvel, G, PS_FX_DV, base*3ull+ull(j), vb[j]*massShare);
                if (((G.fxMask >> PS_FX_DW) & 1u) == 0u)   // (else the mass grid is it)
                    atomicAdd(&G.dispw[flat], massShare);
            }
        }
    }
    if (fGrad)
        for (int j=0; j<3; ++j)
            for (int i=0; i<3; ++i)
                addMoment(G.grad, G, PS_FX_GRAD, base*9ull + ull(j*3+i), E.vG[i][j]*wmEx);
    if (fScal) {   // exact mean of the linear scalar profile over the piece (the velocity's construction)
        float sb = E.s0;
        for (int i=0; i<3; ++i) sb += E.sG[i]*cen[i];
        addMoment(G.scal, G, PS_FX_SCAL, base, sb*wmEx);
    }
    if (fSGrad)
        for (int i=0; i<3; ++i)
            addMoment(G.sgrad, G, PS_FX_SG, base*3ull + ull(i), E.sG[i]*wmEx);
    if (fVolW)
        addMoment(G.momw, G, PS_FX_MOMW, base, wmEx);
    if (fCaustic)
        atomicOr(&G.caustic[flat], E.obits);
    addMoment(G.sv, G, PS_FX_SV, base, mo[0]*E.invCellVol);
    atomicAdd(&G.streams[flat], 1u);
}

// ------------------------------------------------------------------------------------------------
// Counting a straddling tet's samples BEYOND a non-periodic grid (pass 0 of the sampled deposit),
// one sample column (along z) at a time -- see metal/ps_deposit.metal for the derivation.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float sample_tc(int s, int nSub, float invNSub, int ic0d, float f0d) {
    int const k = (s >= 0) ? s / nSub : -((-s + nSub - 1) / nSub);
    int const sub = s - k * nSub;
    float const fr = (nSub == 1) ? 0.5f : (float(sub) + 0.5f) * invNSub;
    return (float(k - ic0d) - f0d) + fr;
}

struct BeyondColumn {
    float P[4], Q[4];
    int   ic0z; float f0z; float dxz; int nSub; float invNSub;
};

__device__ __forceinline__ float column_value(const BeyondColumn& c, int v, int s) {
    float const rz = sample_tc(s, c.nSub, c.invNSub, c.ic0z, c.f0z) * c.dxz;
    return c.P[v] + c.Q[v] * rz;
}

__device__ __forceinline__ int column_transition(const BeyondColumn& c, int v, float T, bool rise,
                                                 int lo, int hi, float guess) {
    int L = lo, H = hi;
    int const g = (guess > float(lo)) ? ((guess < float(hi)) ? int(ceilf(guess)) : hi) : lo;
    if (g > L && g - 1 < H) { if ((column_value(c, v, g - 1) > T) == rise) H = g - 1; else L = g; }
    if (g >= L && g < H)    { if ((column_value(c, v, g) > T) == rise) H = g; else L = g + 1; }
    while (L < H) { int const mid = L + (H - L) / 2; if ((column_value(c, v, mid) > T) == rise) H = mid; else L = mid + 1; }
    return L;
}

__device__ __forceinline__ void column_run(const BeyondColumn& c, int S0, int S1, const float T[4],
                                           int& a, int& b) {
    a = S0; b = S1;
    for (int v = 0; v < 4 && a < b; ++v) {
        float const guess = (c.Q[v] != 0.0f)
                          ? ((T[v] - c.P[v]) / c.Q[v] / c.dxz + float(c.ic0z) + c.f0z) * float(c.nSub) - 0.5f : 0.0f;
        if (c.Q[v] > 0.0f)      a = column_transition(c, v, T[v], true,  a, b, guess);
        else if (c.Q[v] < 0.0f) b = column_transition(c, v, T[v], false, a, b, guess);
        else if (!(c.P[v] > T[v])) b = a;
    }
}

__device__ bool count_beyond_grid(const float C[3][3], const float M[3][3], const float Ax[3][3],
                                  float detC, float sg, const float B[4], float Bdet,
                                  const int iMin[3], const int iMax[3],
                                  const int ic0[3], const float f0[3],
                                  int nSub, float invNSub, bool fLinear, float d0,
                                  const float dG[3], const DepositParams& P,
                                  unsigned& Nout, float& sumW)
{
    int const S0z = iMin[2] * nSub, S1z = iMax[2] * nSub, gridZ = P.nGrid[2] * nSub;
    float const Tin[4]  = { 2.0f*B[0], 2.0f*B[1], 2.0f*B[2], 2.0f*B[3] };
    float const Tpos[4] = { -2.0f*B[0], -2.0f*B[1], -2.0f*B[2], -2.0f*B[3] };
    BeyondColumn c;
    c.ic0z = ic0[2]; c.f0z = f0[2]; c.dxz = P.dx[2]; c.nSub = nSub; c.invNSub = invNSub;
    bool const zStraddle = S0z < 0 || S1z > gridZ;
    int const Sy0 = iMin[1] * nSub, Sy1 = iMax[1] * nSub, gridY = P.nGrid[1] * nSub;
    float const Vx[4] = { 0.0f, Ax[0][0], Ax[1][0], Ax[2][0] };
    float const Vy[4] = { 0.0f, Ax[0][1], Ax[1][1], Ax[2][1] };
    float const xPad = 1.0e-4f * P.dx[0];
    for (int sx = iMin[0] * nSub; sx < iMax[0] * nSub; ++sx) {
    int const kx = (sx >= 0) ? sx / nSub : -((-sx + nSub - 1) / nSub);
    bool const xInCol = kx >= 0 && kx < P.nGrid[0];
    float const relx = sample_tc(sx, nSub, invNSub, ic0[0], f0[0]) * P.dx[0];
    float yMin = GPU_INF, yMax = -GPU_INF;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 4; ++j) {
            float const xi = Vx[i], xj = Vx[j];
            if (relx < fminf(xi, xj) - xPad || relx > fmaxf(xi, xj) + xPad) continue;
            float y0 = Vy[i], y1 = Vy[j];
            if (xj != xi) {
                float const t = fminf(fmaxf((relx - xi) / (xj - xi), 0.0f), 1.0f);
                y0 = y1 = Vy[i] + t * (Vy[j] - Vy[i]);
            }
            yMin = fminf(yMin, fminf(y0, y1));
            yMax = fmaxf(yMax, fmaxf(y0, y1));
        }
    if (!(yMin <= yMax)) continue;
    int const pLo = max(Sy0, int(floorf((yMin / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) - 2);
    int const pHi = min(Sy1, int(ceilf((yMax / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) + 3);
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
        int aIn = 0, bIn = 0, aPos = 0, bPos = 0;
        bool bulk = outLen > 16;
        if (bulk) {
            for (int a = 0; a < 3; ++a) {
                c.P[a] = sg * (C[a][0]*rx + C[a][1]*ry);
                c.Q[a] = sg * C[a][2];
            }
            c.P[3] = sg * (detC - (C[0][0]*rx + C[0][1]*ry + C[1][0]*rx + C[1][1]*ry + C[2][0]*rx + C[2][1]*ry));
            c.Q[3] = -sg * (C[0][2] + C[1][2] + C[2][2]);
            column_run(c, S0z, S1z, Tin, aIn, bIn);
            column_run(c, S0z, S1z, Tpos, aPos, bPos);
            if (aIn >= bIn) { aIn = aPos; bIn = aPos; }
        }
        for (int i = 0; i < nOut; ++i) {
            int lo1 = outLo[i], hi1 = outHi[i], lo2 = 0, hi2 = 0;
            if (bulk) {
                int const c0 = max(aIn, outLo[i]), c1 = min(bIn, outHi[i]);
                if (c0 < c1) {
                    Nout += unsigned(c1 - c0);
                    if (fLinear) {
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

// ================================================================================================
// depositFields: one thread per tetrahedron (metal/ps_deposit.metal::depositFields).
// ================================================================================================
__global__ void __launch_bounds__(256)
depositFields(const float* __restrict__ verts, const float* __restrict__ vels, float* masses,
              float* massGrid, float* momGrid, float* m2Grid, float* gradGrid, unsigned* strGrid,
              const DepositParams P, const float* __restrict__ dens,
              float* momwGrid, unsigned* caustGrid, float* svGrid, float* dvGrid, float* dwGrid,
              const float* __restrict__ scal, float* scalGrid, float* sgGrid, unsigned* massFxGrid, unsigned* bad)
{
    unsigned const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= P.nTet) return;

    float  m  = masses[tid];
    if (m <= 0.0f) return;

    const float* R = verts + ull(tid) * ull(TET_STRIDE);
    int   const ic0[3] = { __float_as_int(R[0]), __float_as_int(R[1]), __float_as_int(R[2]) };
    float const f0[3]  = { R[3], R[4], R[5] };
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    unsigned const centroidFlat = __float_as_uint(R[15]);

    float d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabsf(d) < 1.0e-6f * avgEdge2 * sqrtf(avgEdge2) || !inverse3(Ax, posInv)) { masses[tid] = -m; return; }

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

    FieldGrids G;
    G.mass = massGrid; G.mom = momGrid; G.m2 = m2Grid; G.grad = gradGrid; G.streams = strGrid;
    G.momw = momwGrid; G.caustic = caustGrid; G.sv = svGrid; G.dispvel = dvGrid; G.dispw = dwGrid;
    G.scal = scalGrid; G.sgrad = sgGrid;
    G.massFx = massFxGrid; G.massScale = P.massScale; G.massFixed = P.massFixed != 0;
    setFixed(G, P);
    G.bad = bad;

    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fExact = P.fExact != 0;
    float const mw = fVolW ? fabsf(d)/6.0f : m;
    unsigned const obits = (fCaustic && P.fLinear == 0) ? unsigned(dens[tid*4+0]) : (d > 0.0f ? 1u : 2u);
    float const invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    float const tetVol = fabsf(d)/6.0f;

    bool const fLinear = P.fLinear != 0;
    float d0 = 0.0f, dG[3] = {0.0f, 0.0f, 0.0f};
    if (fLinear) {
        d0 = dens[tid*4+0];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[tid*4+e+1] - d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; dG[i]=s; }
    }

    int iMin[3], iMax[3], iMinC[3], iMaxC[3];
    bool straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = fminf(lo, Ax[e][dd]); hi = fmaxf(hi, Ax[e][dd]); }
        iMin[dd] = ic0[dd] + int(floorf(f0[dd] + lo / P.dx[dd]));
        iMax[dd] = ic0[dd] + int(floorf(f0[dd] + hi / P.dx[dd])) + 1;
        if (!P.periodic && (iMin[dd] < 0 || iMax[dd] > P.nGrid[dd])) straddle = true;
        // no one-cell sample margin (as metal/ps_deposit.metal): every sub-sample lies
        // >= 1/(2 nSub) inside its cell, so the ring's samples would all be outside the tet
        iMinC[dd] = iMin[dd]; iMaxC[dd] = iMax[dd];
        if (!P.periodic) {
            if (iMinC[dd] < 0)           iMinC[dd] = 0;
            if (iMaxC[dd] > P.nGrid[dd]) iMaxC[dd] = P.nGrid[dd];
            if (iMinC[dd] >= P.nGrid[dd] || iMaxC[dd] <= 0) return;
        }
    }
    int nSub = P.nSub;

    float C[3][3], M[3][3];
    {
        float const A00=Ax[0][0], A01=Ax[0][1], A02=Ax[0][2];
        float const A10=Ax[1][0], A11=Ax[1][1], A12=Ax[1][2];
        float const A20=Ax[2][0], A21=Ax[2][1], A22=Ax[2][2];
        C[0][0]=A11*A22-A12*A21; M[0][0]=fabsf(A11*A22)+fabsf(A12*A21);
        C[0][1]=A12*A20-A10*A22; M[0][1]=fabsf(A12*A20)+fabsf(A10*A22);
        C[0][2]=A10*A21-A11*A20; M[0][2]=fabsf(A10*A21)+fabsf(A11*A20);
        C[1][0]=A02*A21-A01*A22; M[1][0]=fabsf(A02*A21)+fabsf(A01*A22);
        C[1][1]=A00*A22-A02*A20; M[1][1]=fabsf(A00*A22)+fabsf(A02*A20);
        C[1][2]=A01*A20-A00*A21; M[1][2]=fabsf(A01*A20)+fabsf(A00*A21);
        C[2][0]=A01*A12-A02*A11; M[2][0]=fabsf(A01*A12)+fabsf(A02*A11);
        C[2][1]=A02*A10-A00*A12; M[2][1]=fabsf(A02*A10)+fabsf(A00*A12);
        C[2][2]=A00*A11-A01*A10; M[2][2]=fabsf(A00*A11)+fabsf(A01*A10);
    }
    float const detC = Ax[0][0]*C[0][0] + Ax[0][1]*C[0][1] + Ax[0][2]*C[0][2];
    float Rw[3], rho[3];
    for (int i=0; i<3; ++i) {
        float const T = fmaxf(fabsf(float(iMin[i] - ic0[i]) - f0[i]), fabsf(float(iMax[i] - ic0[i]) - f0[i]));
        Rw[i]  = T * P.dx[i];
        rho[i] = P.dx[i] * (U_F * (6.0f + 5.0f*T) + 1.0e-9f);
    }
    float B[4];
    float nMag = 0.0f;
    for (int a=0; a<3; ++a) {
        float b = 0.0f;
        for (int i=0; i<3; ++i) { b += M[a][i] * (7.0f*U_F*Rw[i] + rho[i]); nMag += M[a][i] * Rw[i]; }
        B[a] = 2.0f * b;
    }
    float const Bdet = 2.0f * 8.0f * U_F * (fabsf(Ax[0][0])*M[0][0] + fabsf(Ax[0][1])*M[0][1] + fabsf(Ax[0][2])*M[0][2]);
    B[3] = Bdet + B[0] + B[1] + B[2] + 2.0f * 3.0f * U_F * (fabsf(detC) + nMag);
    bool const orientSure = fabsf(detC) > Bdet;
    float const sg = detC > 0.0f ? 1.0f : -1.0f;
    float const invNSub = 1.0f / float(nSub);

    // ===================== exact conservative deposit (--exact-average) =====================
    if (fExact) {
        ExactTet E;
        E.t1 = v3(Ax[0][0], Ax[0][1], Ax[0][2]);
        E.t2 = v3(Ax[1][0], Ax[1][1], Ax[1][2]);
        E.t3 = v3(Ax[2][0], Ax[2][1], Ax[2][2]);
        if (d < 0.0f) { V3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
        E.faces = tet_planes(E.t1, E.t2, E.t3);
        E.hcell = v3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
        for (int i=0; i<3; ++i) {
            E.ic0[i] = ic0[i];  E.f0[i] = f0[i];  E.u0[i] = u0[i];  E.dG[i] = dG[i];  E.sG[i] = sG[i];
            for (int j=0; j<3; ++j) E.vG[i][j] = vG[i][j];
        }
        for (int j=0; j<3; ++j) { E.vc[j] = 0.0f; E.vh[j] = 0.0f; }
        if (fVel || fDisp) exact_vel_range(Ax, vG, u0, E.vc, E.vh);
        E.d0 = d0;  E.s0 = s0;  E.obits = obits;  E.invCellVol = invCellVol;
        E.straddle = straddle;
        E.wFull = exact_full_weight(tetVol, fLinear, d0, dG, E.t1, E.t2, E.t3);

        // --ps-window: only the window's cells are clipped and the tet is normalized by its whole
        // weight (as a straddling tet is); an empty window then means "no volume inside the
        // window", not a degenerate tet -- nothing to deposit, no sampled fallback
        if (P.windowMode != 0)
            for (int dd=0; dd<3; ++dd)
                if (!window_hull(iMinC[dd], iMaxC[dd], P.nGrid[dd], P.periodic != 0, P.subOrigin[dd], P.subDims[dd])) return;
        // a dispersion float32 cannot resolve: the CPU's double deposit takes the tet (EX_DISP_KAPPA2)
        if (fDisp && exact_disp_illcond(Ax, vG, E.vh, P.dx)) { masses[tid] = -m; return; }
        float sumW = 0.0f;
        float shareFac = 0.0f;
        bool  deposited = false;
        for (int pass=0; pass<2; ++pass) {
            if (pass==1) {
                if (!(sumW > 0.0f)) { if (P.windowMode != 0) return; break; }
                shareFac = m / ((E.straddle || P.windowMode != 0) ? fmaxf(sumW, E.wFull) : sumW);
                deposited = true;
            }
            for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
            for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
            for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
                int const raw[3]={gi,gj,gk};
                unsigned flat;
                if (window_flat(raw, P, flat) != 2) continue;   // outside the sub-box / window
                float mo[10], w;
                V3 cc;
                if (!exact_cell_clip(E, raw, P, pass==1, fLinear, mo, w, cc)) continue;
                if (pass==0) { sumW += w; continue; }
                exact_cell_deposit(G, flat, mo, w * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic, fScal, fSGrad, cc);
            }
        }
        if (deposited) return;
    }

    // ===================== the sampled two-pass deposit =====================
    if (!orientSure) { masses[tid] = -m; return; }
    // without --ps-linear-deposit every sample carries the same shares: depositSampleBatch (metal/ps_deposit.metal)
    bool const batched = !fLinear;
    unsigned N = 0, Nout = 0;
    float sumW = 0.0f;
    float share = 0.0f;
    for (int pass=0; pass<2; ++pass) {
        if (pass==1) {
            if (straddle && (N > 0u || centroidFlat != 0xFFFFFFFFu)
                && !count_beyond_grid(C, M, Ax, detC, sg, B, Bdet, iMin, iMax, ic0, f0, nSub, invNSub,
                                      fLinear, d0, dG, P, Nout, sumW)) { masses[tid] = -m; return; }
            if (N==0u) break;
            share = m / float(N + Nout);
        }
        for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
        for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
        for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
            int raw[3]={gi,gj,gk};
            int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi;
            int wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj;
            int wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
            bool const inGrid = !(wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2]);
            if (!inGrid) continue;
            unsigned flat = 0u;
            bool inSub = true;
            {
                int l0=wg0-P.subOrigin[0], l1=wg1-P.subOrigin[1], l2=wg2-P.subOrigin[2];
                if (l0<0) l0+=P.nGrid[0];
                if (l1<0) l1+=P.nGrid[1];
                if (l2<0) l2+=P.nGrid[2];
                if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) inSub = false;
                else flat=(unsigned(l0)*unsigned(P.subDims[1])+unsigned(l1))*unsigned(P.subDims[2])+unsigned(l2);
            }
            // a WINDOW still counts the cell's samples in pass 0 (the full run's normalization), deposits none
            if (!inSub && !(P.windowMode != 0 && pass == 0)) continue;
            float const cb[3] = { float(raw[0] - ic0[0]) - f0[0],
                                  float(raw[1] - ic0[1]) - f0[1],
                                  float(raw[2] - ic0[2]) - f0[2] };
            unsigned nIn = 0u;   // pass 1: the cell's interior samples (their position-independent shares: below)
            for (int k2=0; k2<nSub; ++k2)
            for (int k1=0; k1<nSub; ++k1)
            for (int k0=0; k0<nSub; ++k0) {
                float const fr0 = (nSub==1) ? 0.5f : (float(k0) + 0.5f) * invNSub;
                float const fr1 = (nSub==1) ? 0.5f : (float(k1) + 0.5f) * invNSub;
                float const fr2 = (nSub==1) ? 0.5f : (float(k2) + 0.5f) * invNSub;
                float const tc[3] = { cb[0] + fr0, cb[1] + fr1, cb[2] + fr2 };
                float rel[3] = { tc[0] * P.dx[0], tc[1] * P.dx[1], tc[2] * P.dx[2] };
                int const cls = classifySample(C, M, detC, sg, B, Bdet, rel, tc, P);
                if (cls == 0 && pass == 0) { masses[tid] = -m; return; }
                if (cls > 0) {
                    if (pass==0) {
                        N++;
                        if (fLinear) sumW += linearWeight(rel, d0, dG);
                    } else {
                        float w = share;
                        if (fLinear && sumW > 0.0f)
                            w = m * linearWeight(rel, d0, dG) / sumW;
                        float wm = fVolW ? mw / float(N + Nout) : w;
                        float sv = fExact ? (tetVol*invCellVol)/float(N + Nout) : 0.0f;
                        if (P.debugNoAtomic == 0u)   // (PS_GPU_DEBUG_NOATOMIC: metal/ps_deposit.metal DepositParams)
                            depositSample(G, flat, rel, w, wm, obits, u0, vG, s0, sG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, sv, true, false, fScal, fSGrad, batched);
                        ++nIn;
                    }
                }
            }
            if (batched && nIn > 0u && P.debugNoAtomic == 0u) {
                float const wm = fVolW ? mw / float(N + Nout) : share;
                float const sv = fExact ? (tetVol*invCellVol)/float(N + Nout) : 0.0f;
                depositSampleBatch(G, flat, nIn, share, wm, obits, vG, sG, fGrad, fVolW, fCaustic, fExact, sv, fSGrad);
            }
        }
    }

    if (N==0u && Nout > 0u) return;
    if (N==0u) {   // centroid fallback
        if (centroidFlat == 0xFFFFFFFFu) return;
        float crel[3];
        for (int i=0; i<3; ++i) crel[i] = (Ax[0][i] + Ax[1][i] + Ax[2][i]) * 0.25f;
        if (P.debugNoAtomic == 0u)
            depositSample(G, centroidFlat, crel, m, mw, obits, u0, vG, s0, sG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, tetVol*invCellVol, fExact, true, fScal, fSGrad, false);
    }
}

// ================================================================================================
// depositExactItems: the exact deposit with the work split BELOW the tetrahedron, one thread per
// (tet, block of IP.blockCells window cells); phase 0 sums the weights, phase 1 deposits.
// ================================================================================================
// (ps_kernel_params.h, the Metal kernel's struct; badAt stays 0 here: the guard counters are a buffer of their own)
using ExactItemParams = PSExactItemParams;

// decide: also defer an ill-conditioned dispersion (EX_DISP_KAPPA2) by returning false -- phase 0 only, as in
// metal/ps_deposit.metal (a borderline tet phase 1 re-decided after phase 0 summed its weights would be lost)
__device__ bool exact_tet_setup(unsigned tet, const float* verts, const float* vels,
                                const float* dens, const float* scal, const DepositParams& P,
                                ExactTet& E, int iMin[3], int iMax[3], bool decide)
{
    const float* R = verts + ull(tet) * ull(TET_STRIDE);
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    for (int i=0; i<3; ++i) { E.ic0[i] = __float_as_int(R[i]); E.f0[i] = R[3+i]; }

    float const d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabsf(d) < 1.0e-6f * avgEdge2 * sqrtf(avgEdge2) || !inverse3(Ax, posInv)) return false;

    for (int i=0; i<3; ++i) { E.u0[i] = 0.0f; for (int j=0; j<3; ++j) E.vG[i][j] = 0.0f; }
    if (P.fVel != 0 || P.fDisp != 0 || P.fGrad != 0) {
        ull const v0 = ull(tet) * 12ull;
        for (int j=0;j<3;++j) E.u0[j] = vels[v0+ull(j)];
        float dV[3][3];
        for (int e=0;e<3;++e)
            for (int j=0;j<3;++j)
                dV[e][j] = vels[v0+ull((e+1)*3+j)] - E.u0[j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j];
                E.vG[i][j]=s;
            }
    }
    for (int j=0; j<3; ++j) { E.vc[j] = 0.0f; E.vh[j] = 0.0f; }
    if (P.fVel != 0 || P.fDisp != 0) exact_vel_range(Ax, E.vG, E.u0, E.vc, E.vh);
    // ill-conditioned: left to depositFields, which defers it to the CPU (EX_DISP_KAPPA2)
    if (P.fDisp != 0 && decide && exact_disp_illcond(Ax, E.vG, E.vh, P.dx)) return false;
    E.s0 = 0.0f;
    for (int i=0; i<3; ++i) E.sG[i] = 0.0f;
    if (P.fScal != 0 || P.fSGrad != 0) {
        E.s0 = scal[ull(tet)*4ull];
        float ds[3];
        for (int e=0;e<3;++e) ds[e] = scal[ull(tet)*4ull+ull(e+1)] - E.s0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*ds[k]; E.sG[i]=s; }
    }
    E.obits = (P.fCaustic != 0 && P.fLinear == 0) ? unsigned(dens[ull(tet)*4ull]) : (d > 0.0f ? 1u : 2u);
    E.invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    E.d0 = 0.0f;
    for (int i=0; i<3; ++i) E.dG[i] = 0.0f;
    if (P.fLinear != 0) {
        E.d0 = dens[ull(tet)*4ull];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[ull(tet)*4ull+ull(e+1)] - E.d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; E.dG[i]=s; }
    }

    E.straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = fminf(lo, Ax[e][dd]); hi = fmaxf(hi, Ax[e][dd]); }
        iMin[dd] = E.ic0[dd] + int(floorf(E.f0[dd] + lo / P.dx[dd]));
        iMax[dd] = E.ic0[dd] + int(floorf(E.f0[dd] + hi / P.dx[dd])) + 1;
        if (!P.periodic) {
            if (iMin[dd] < 0)           { iMin[dd] = 0;           E.straddle = true; }
            if (iMax[dd] > P.nGrid[dd]) { iMax[dd] = P.nGrid[dd]; E.straddle = true; }
        }
    }

    // --ps-window: only the window's cells (the hull of its images along each axis) are items
    if (P.windowMode != 0)
        for (int dd=0; dd<3; ++dd)
            if (!window_hull(iMin[dd], iMax[dd], P.nGrid[dd], P.periodic != 0, P.subOrigin[dd], P.subDims[dd])) { iMax[dd] = iMin[dd]; }

    E.t1 = v3(Ax[0][0], Ax[0][1], Ax[0][2]);
    E.t2 = v3(Ax[1][0], Ax[1][1], Ax[1][2]);
    E.t3 = v3(Ax[2][0], Ax[2][1], Ax[2][2]);
    if (d < 0.0f) { V3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
    E.faces = tet_planes(E.t1, E.t2, E.t3);
    E.hcell = v3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
    E.wFull = exact_full_weight(fabsf(d)/6.0f, P.fLinear != 0, E.d0, E.dG, E.t1, E.t2, E.t3);
    E.wScale = 1.0f;
    if ((P.fxMask & PS_FX_SUMW_BIT) != 0u) {   // (metal/ps_deposit.metal exact_tet_setup)
        float wb = fabsf(d)/6.0f;
        if (P.fLinear != 0) {
            float dMax = 0.0f;
            for (int v=0; v<4; ++v) dMax = fmaxf(dMax, fabsf(dens[ull(tet)*4ull+ull(v)]));
            wb *= dMax;
        }
        int e = 0;
        (void)frexpf(wb, &e);                      // wb < 2^e
        // wb <= FLT_MAX is isfinite(wb) for the wb > 0 that passed: the unqualified isfinite is no
        // name under gcc's <cmath>, so the CUDA emulation (CUDAEMU=1, CI on Linux) would not compile
        E.wScale = (wb > 0.0f && wb <= 3.40282347e+38f) ? ldexpf(1.0f, min(max(59 - e, -126), 126)) : 1.0f;
    }
    return true;
}

// An item's (tet, block) from the items' map (gpu_host.h psBuildExactItemMap): metal/ps_deposit.metal psItemOf, the
// last t in [blockTet[blk], blockTet[blk+1]] with itemStart[t] <= iid, and the block iid - itemStart[t]
__device__ __forceinline__ uint2 psItemOf(const unsigned* itemStart, const unsigned* blockTet, unsigned nTet, unsigned iid) {
    unsigned const blk = iid >> PS_ITEM_BLOCK_SHIFT;
    unsigned lo = blockTet[blk];
    unsigned hi = min(blockTet[blk + 1u], nTet - 1u);
    while (hi > lo) {
        unsigned const mid = (lo + hi + 1u) >> 1;
        if (itemStart[mid] <= iid) lo = mid; else hi = mid - 1u;
    }
    uint2 it;   // (field by field: the CPU emulation's uint2 has no make_uint2)
    it.x = lo;
    it.y = iid - itemStart[lo];
    return it;
}

// PHASE: the pass (== IP.phase), a template parameter as Metal's kItemsPhase is a function constant: the summing
// pass then carries none of the deposit's registers
template <int PHASE>
__global__ void __launch_bounds__(256)
depositExactItems(const float* __restrict__ verts, const float* __restrict__ vels, const float* __restrict__ masses,
                  float* massGrid, float* momGrid, float* m2Grid, float* gradGrid, unsigned* strGrid,
                  const DepositParams P, const float* __restrict__ dens,
                  float* momwGrid, unsigned* caustGrid, float* svGrid, float* dvGrid, float* dwGrid,
                  const unsigned* __restrict__ itemMap, float* sumW, const ExactItemParams IP,
                  const float* __restrict__ scal, float* scalGrid, float* sgGrid, unsigned* massFxGrid, unsigned* bad)
{
    unsigned const lid = blockIdx.x * blockDim.x + threadIdx.x;
    if (lid >= IP.nItems) return;
    uint2 const it = psItemOf(itemMap, itemMap + (ull(IP.nTet) + 1ull), IP.nTet, IP.itemBase + lid);   // (tet, block)
    unsigned const tet = it.x;
    float const m = masses[tet];
    if (m <= 0.0f) return;
    bool const deposit = PHASE == 1;
    bool const sumWFixed = (P.fxMask & PS_FX_SUMW_BIT) != 0u;
    float s = 0.0f;
    if (deposit && !sumWFixed) {
        s = sumW[tet];               // written by phase 0's kernel, complete before this launch
        if (!(s > 0.0f)) return;
    }

    ExactTet E;
    int iMin[3], iMax[3];
    if (!exact_tet_setup(tet, verts, vels, dens, scal, P, E, iMin, iMax, !deposit)) return;
    if (deposit && sumWFixed) {      // the exact total in the tet's own unit (two words), as the Metal kernel forms it
        unsigned const* x = reinterpret_cast<unsigned const*>(sumW);
        unsigned const lo = x[ull(tet)*2ull];
        int const hi = int(x[ull(tet)*2ull+1ull]);
        if (hi < 0 || (hi == 0 && lo == 0u)) return;
        s = (float(hi) * 4294967296.0f + float(lo)) / E.wScale;
    }
    float const shareFac = deposit ? m / ((E.straddle || P.windowMode != 0) ? fmaxf(s, E.wFull) : s) : 0.0f;
    ull const n1 = ull(max(iMax[1] - iMin[1], 0));
    ull const n2 = ull(max(iMax[2] - iMin[2], 0));
    ull const n12 = n1 * n2;
    ull const nW = ull(max(iMax[0] - iMin[0], 0)) * n12;
    ull w = ull(it.y) * ull(IP.blockCells);
    if (w >= nW) return;
    ull const wEnd = (nW - w > ull(IP.blockCells)) ? w + ull(IP.blockCells) : nW;
    int a0 = int(w / n12);
    ull const rem = w - ull(a0) * n12;
    int a1 = int(rem / n2);
    int a2 = int(rem - ull(a1) * n2);

    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fLinear = P.fLinear != 0;
    bool const fScal = P.fScal != 0, fSGrad = P.fSGrad != 0;
    FieldGrids G;
    G.mass = massGrid; G.mom = momGrid; G.m2 = m2Grid; G.grad = gradGrid; G.streams = strGrid;
    G.momw = momwGrid; G.caustic = caustGrid; G.sv = svGrid; G.dispvel = dvGrid; G.dispw = dwGrid;
    G.scal = scalGrid; G.sgrad = sgGrid;
    G.massFx = massFxGrid; G.massScale = P.massScale; G.massFixed = P.massFixed != 0;
    setFixed(G, P);
    G.bad = bad;
    float acc = 0.0f;
    for (; w < wEnd; ++w) {
        int const raw[3] = { iMin[0] + a0, iMin[1] + a1, iMin[2] + a2 };
        if (ull(++a2) == n2) { a2 = 0; if (ull(++a1) == n1) { a1 = 0; ++a0; } }
        unsigned flat;
        if (window_flat(raw, P, flat) != 2) continue;   // outside the sub-box / window
        float mo[10], wt;
        V3 cc;
        if (!exact_cell_clip(E, raw, P, deposit, fLinear, mo, wt, cc)) continue;
        if (!deposit) acc += wt;
        else exact_cell_deposit(G, flat, mo, wt * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic, fScal, fSGrad, cc);
    }
    if (!deposit && acc > 0.0f) {
        if (sumWFixed) {
            float const p = acc * E.wScale;
            if (fixedShareOk(p)) addFixedQ(reinterpret_cast<unsigned*>(sumW), ull(tet), ull((long long)(rintf(p))));
            else atomicAdd(&bad[FX_DROP_SUMW], 1u);   // (the non-finite guard)
        }
        else atomicAdd(&sumW[tet], acc);
    }
}

// gathers n records of 'stride' floats: dst[k] = src[idx[k]] (the compact leftovers)
__global__ void gatherRecords(const float* __restrict__ src, const unsigned* __restrict__ idx,
                              unsigned n, unsigned stride, float* dst)
{
    unsigned const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float* s = src + ull(idx[k]) * ull(stride);
    float* d = dst + ull(k) * ull(stride);
    for (unsigned c = 0; c < stride; ++c) d[c] = s[c];
}


// ================================================================================================
// host
// ================================================================================================
struct Ctx
{
    bool        ready = false;
    std::string err;
    std::string name;
};

// serializes the dispatches (one device): a psGpuDepositFields call holds it from its uploads to the end of its passes
std::mutex& ctxMutex() { static std::mutex m; return m; }

// The device query: ctx() runs this once.
void ctxInit(Ctx& c)
{
    int n = 0;
    if (gpuGetDeviceCount(&n) != gpuSuccess || n < 1) { c.err = "no " GPU_BACKEND_STRING " device"; return; }
    gpuDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    if (gpuGetDeviceProperties(&prop, 0) != gpuSuccess) { c.err = "device query failed"; return; }
    c.name = prop.name;
    c.ready = true;
}

// The process's device context, built by the first caller (std::call_once: the partitions prepare their deposits
// before they take ctxMutex, so several may ask at once; as ps_metal_host.cc)
Ctx& ctx()
{
    static Ctx c;
    static std::once_flag once;
    std::call_once(once, ctxInit, std::ref(c));
    return c;
}

// a device allocation that is zeroed; null (and 'err' set) on failure
float* devAlloc(size_t bytes, const char* what, std::string& err)
{
    void* p = nullptr;
    if (!gpuCheck(gpuMalloc(&p, bytes ? bytes : 4), what, err)) return nullptr;
    if (!gpuCheck(gpuMemset(p, 0, bytes ? bytes : 4), what, err)) { gpuFree(p); return nullptr; }
    return static_cast<float*>(p);
}

// uploads a host array (a 4-byte zeroed dummy when empty); null on failure
float* devUpload(const void* src, size_t bytes, const char* what, std::string& err)
{
    float* p = devAlloc(bytes, what, err);
    if (!p || !bytes) return p;
    if (!gpuCheck(gpuMemcpy(p, src, bytes, gpuMemcpyHostToDevice), what, err)) { gpuFree(p); return nullptr; }
    return p;
}

inline unsigned blocksFor(size_t n, int tpb) { return unsigned((n + size_t(tpb) - 1) / size_t(tpb)); }


// ---- chunk-level checkpoints and the cause-aware retry policy (ported from ps_metal_host.cc) ----
// A failed kernel does not redo the WHOLE partition deposit from re-zeroed grids: every
// pass keeps a host-side snapshot of the device buffers its kernel writes (the accumulator grids, the exact
// items' sumW, the per-tet masses the kernel negates when it defers a tet), taken at the pass's start and
// every DTFE_GPU_CHECKPOINT_CHUNKS kernels; a failed kernel (which may have deposited part or all of its
// atomics) restores the last snapshot and continues from its chunk. The snapshot is a device-to-host copy
// (the Metal host memcpy's its unified memory), so each checkpoint costs a PCIe round of the grids. The
// policy's pieces (GpuFault, the error codes, the pace, the fault hook, GpuSnapshot) live in gpu_cuda_retry.cuh,
// shared with the standard-DTFE host (dtfe_gpu_cuda.cu); the pass loop is psGpuDepositFields' runChunked below.
#include "gpu_cuda_retry.cuh"

// the run's retry counters (ps_interpolation.cc's), by gpuRetryCause's index
void countGpuRetry(GpuFault f, gpuError_t code)
{
    int const k = gpuRetryCause(f, code);
    ++psGpuRetries;
    ++psGpuRetriesByCause[k];
}

} // namespace


std::string gpuBackendName() { return GPU_BACKEND_STRING; }

std::string gpuDeviceName()
{
    Ctx& c = ctx();   // (thread-safe: std::call_once)
    return c.ready ? c.name : std::string();
}


namespace {

// The arguments of psHostUploadBuffers: psGpuDepositFields' locals (the record arrays it consumes, the host mirror of
// the masses, and the locals it sets, by reference).
struct PSHostUploadArgs
{
    std::vector<float>     &verts, &vels, &hostMass, &dens, &scal;
    PSGpuPlan const&       pl;
    size_t                 nCell, nTetTotal, sumWBytes;
    bool&                  useItems;
    std::vector<uint32_t>& items;
    std::string&           err;
};

// A call's device buffers as the upload leaves them (null: allocation failed; the exact items' two also when off).
struct PSHostBuffers
{
    float    *dV, *dU, *dM, *dD, *dSc, *dMass, *dMassFx, *dMom, *dM2, *dGrad, *dStr, *dMomW, *dCaust, *dSV, *dDV, *dDW,
             *dScal, *dSG;
    unsigned *dBad;
    float    *dItems, *dSumW;
};

// Uploads the per-tet records and the exact items, freeing each host array at once, and allocates the zeroed grids.
PSHostBuffers psHostUploadBuffers(PSHostUploadArgs const& u)
{
    std::vector<float> &verts = u.verts, &vels = u.vels, &hostMass = u.hostMass, &dens = u.dens, &scal = u.scal;
    PSGpuPlan const& pl = u.pl;
    size_t const nCell = u.nCell, nTetTotal = u.nTetTotal, sumWBytes = u.sumWBytes;
    bool& useItems = u.useItems;
    std::vector<uint32_t>& items = u.items;
    std::string& err = u.err;
    float* dV = devUpload(verts.data(), verts.size() * sizeof(float), "upload tet records", err);
    std::vector<float>().swap(verts);
    float* dU = dV ? devUpload(vels.data(), vels.size() * sizeof(float), "upload velocities", err) : nullptr;
    std::vector<float>().swap(vels);
    float* dM = dU ? devUpload(hostMass.data(), hostMass.size() * sizeof(float), "upload masses", err) : nullptr;
    float* dD = dM ? devUpload(dens.data(), dens.size() * sizeof(float), "upload densities", err) : nullptr;
    std::vector<float>().swap(dens);
    float* dSc = dD ? devUpload(scal.data(), scal.size() * sizeof(float), "upload scalars", err) : nullptr;
    std::vector<float>().swap(scal);
    float* dMass = dSc ? devAlloc(pl.massBytes, "mass grid", err) : nullptr;
    float* dMassFx = dMass ? devAlloc(pl.massFxBytes, "fixed-point mass grid", err) : nullptr;
    float* dMom  = dMassFx ? devAlloc(pl.momBytes, "moment grid", err) : nullptr;
    float* dM2   = dMom ? devAlloc(pl.m2Bytes, "second-moment grid", err) : nullptr;
    float* dGrad = dM2 ? devAlloc(pl.gradBytes, "gradient grid", err) : nullptr;
    float* dStr  = dGrad ? devAlloc(nCell * 4, "stream grid", err) : nullptr;
    float* dMomW = dStr ? devAlloc(pl.momwBytes, "moment-weight grid", err) : nullptr;
    float* dCaust = dMomW ? devAlloc(pl.caustBytes, "caustic grid", err) : nullptr;
    float* dSV   = dCaust ? devAlloc(pl.svBytes, "stream-volume grid", err) : nullptr;
    float* dDV   = dSV ? devAlloc(pl.dvBytes, "dispersion-mean grid", err) : nullptr;
    float* dDW   = dDV ? devAlloc(pl.dwBytes, "dispersion-weight grid", err) : nullptr;
    float* dScal = dDW ? devAlloc(pl.scalBytes, "scalar grid", err) : nullptr;
    float* dSG   = dScal ? devAlloc(pl.sgBytes, "scalar-gradient grid", err) : nullptr;
    // the non-finite guard's counters (FieldGrids.bad), one word per slot, for both kernels
    unsigned* dBad = dSG ? reinterpret_cast<unsigned*>(devAlloc(FX_DROP_SLOTS * 4, "guard counters", err)) : nullptr;
    if (!dBad && dSG) { gpuFree(dSG); dSG = nullptr; }
    float* dItems = nullptr;
    float* dSumW  = nullptr;
    if (dSG && useItems)
    {
        dItems = devUpload(items.data(), items.size() * sizeof(uint32_t), "upload exact items", err);
        dSumW  = dItems ? devAlloc(std::max<size_t>(nTetTotal, 1) * sumWBytes, "per-tet weight totals", err) : nullptr;
        if (!dItems || !dSumW)
        {   // too big for the device: the one-thread-per-tet kernel still works
            if (dItems) gpuFree(dItems);
            if (dSumW)  gpuFree(dSumW);
            dItems = dSumW = nullptr;
            useItems = false;
            err.clear();
        }
        std::vector<uint32_t>().swap(items);
    }
    return PSHostBuffers{ dV, dU, dM, dD, dSc, dMass, dMassFx, dMom, dM2, dGrad, dStr, dMomW, dCaust, dSV, dDV, dDW,
                          dScal, dSG, dBad, dItems, dSumW };
}

// The arguments of psHostExactItems: psGpuDepositFields' locals (by reference those it sets).
struct PSHostItemsArgs
{
    std::vector<float>&    verts;
    DepositParams const&   P;
    const size_t*          nGrid;
    size_t                 nTetTotal;
    bool                   periodic, windowMode, timing;
    bool&                  useItems;
    uint32_t&              blockCells;
    std::vector<uint32_t>& items;
    size_t&                nItems;
    PSGpuPhaseTimes&       ph;
};

// Before the hold: bounds the tets' windows from the host records and builds the exact items' map
// (psBuildExactItemMap), uploaded with the records; the items are off if their numbers do not fit 32 bits.
void psHostExactItems(PSHostItemsArgs const& x)
{
    std::vector<float>& verts = x.verts;
    DepositParams const& P = x.P;
    const size_t* nGrid = x.nGrid;
    size_t const nTetTotal = x.nTetTotal;
    bool const periodic = x.periodic, windowMode = x.windowMode, timing = x.timing;
    bool& useItems = x.useItems;
    uint32_t& blockCells = x.blockCells;
    std::vector<uint32_t>& items = x.items;
    size_t& nItems = x.nItems;
    if (useItems)
    {
        auto tPh = std::chrono::steady_clock::now();
        std::vector<uint64_t> bound;
        double const total = psExactItemBounds(verts.data(), nTetTotal, P, nGrid, periodic, windowMode, bound,
                                               blockCells, nItems);
        if (timing)
            fprintf(stderr, "PS-DTFE %s: exact items: %zu tets, %zu items of %u cells, window bound %.3g cells\n",
                    GPU_BACKEND_STRING, nTetTotal, nItems, blockCells, total);
        x.ph.bounds += PSGpuPhaseTimes::lap(tPh);
        if (psExactItemMapFits(nTetTotal, nItems))
        {
            items.resize(psExactItemMapWords(nTetTotal, nItems));
            psBuildExactItemMap(bound, blockCells, nItems, items.data());
        }
        else useItems = false;   // (32-bit item numbers: the one-thread-per-tet kernel still works)
        x.ph.fill += PSGpuPhaseTimes::lap(tPh);
    }
}

// The arguments of psHostCopyBack: psGpuDepositFields' locals (by reference the host mirror and the call's outputs).
struct PSHostCopyArgs
{
    PSHostBuffers const&   B;
    PSGpuPlan const&       pl;
    size_t                 nCell, nTetTotal;
    bool                   fCaustic;
    std::vector<float>&    hostMass;
    PSGpuGrids&            out;
    std::vector<uint32_t>& deferred;
    std::string&           err;
};

// Lists the deferred tets (their negated masses) and copies the kept grids into 'out'; false when a copy fails.
bool psHostCopyBack(PSHostCopyArgs const& k)
{
    PSHostBuffers const& B = k.B;
    float *dM = B.dM, *dMass = B.dMass, *dMassFx = B.dMassFx, *dMom = B.dMom, *dM2 = B.dM2, *dGrad = B.dGrad,
          *dStr = B.dStr, *dMomW = B.dMomW, *dCaust = B.dCaust, *dSV = B.dSV, *dDV = B.dDV, *dDW = B.dDW,
          *dScal = B.dScal, *dSG = B.dSG;
    PSGpuPlan const& pl = k.pl;
    size_t const nCell = k.nCell, nTetTotal = k.nTetTotal;
    bool const fCaustic = k.fCaustic;
    std::vector<float>& hostMass = k.hostMass;
    PSGpuGrids& out = k.out;
    std::vector<uint32_t>& deferred = k.deferred;
    std::string& err = k.err;
    // deferred tets: the kernel negated their masses (per-tet path: on the device; leftovers:
    // written back into both copies)
    bool ok = gpuCheck(gpuMemcpy(hostMass.data(), dM, nTetTotal * sizeof(float), gpuMemcpyDeviceToHost), "read masses back", err);
    bool const massAudit = PSMassAudit::on();   // PS_MASS_AUDIT=1: the deferred mass and the fixed-point total, as the Metal host
    double deferredMass = 0.;
    for (size_t t = 0; ok && t < nTetTotal; ++t)
        if (hostMass[t] < 0.f) { deferred.push_back(uint32_t(t)); deferredMass -= double(hostMass[t]); }
    if (massAudit) out.auditDeferredMass = deferredMass;

    if (ok && pl.massFixed)
    {   // the words themselves: the caller adds the
        // deferred and released tets into the same integers and converts once (ps_metal_host.cc says why).
        // Low word first: on a little-endian host each pair is the int64 (ps_deposit_params.h).
        out.massFx.resize(nCell);
        ok = gpuCheck(gpuMemcpy(out.massFx.data(), dMassFx, nCell * 8, gpuMemcpyDeviceToHost),
                      "copy fixed-point mass back", err);
        if (ok && massAudit)
        {
            int64_t q = 0;
            for (size_t i = 0; i < nCell; ++i) q += out.massFx[i];
            out.auditFixedTotal = psMassFixedMass(q, pl.massScale);
        }
    }
    else if (ok)
        ok = gpuCheck(gpuMemcpy(out.mass.data(), dMass, nCell * 4, gpuMemcpyDeviceToHost), "copy mass back", err);
    {   // each moment grid: its words (fixed point; little-endian pairs = the int64) or its floats
        float* const fxDev[PS_FX_N] = { dMom, dM2, dGrad, dMomW, dSV, dDV, nullptr, dScal, dSG };
        std::vector<float>* const flt[PS_FX_N] = { &out.mom, &out.m2, &out.grad, &out.momw, &out.streamvol, &out.dispvel,
                                                   nullptr, &out.scal, &out.sgrad };
        for (int g = 0; ok && g < PS_FX_N; ++g)
        {
            if (!pl.sel[g] || !fxDev[g]) continue;
            size_t const n = nCell * PS_FX_COMPS[g];
            if (pl.fxOn[g])
            {
                out.fx[g].resize(n);
                ok = gpuCheck(gpuMemcpy(out.fx[g].data(), fxDev[g], n * 8, gpuMemcpyDeviceToHost), "copy fixed-point moments back", err);
            }
            else ok = gpuCheck(gpuMemcpy(flt[g]->data(), fxDev[g], n * 4, gpuMemcpyDeviceToHost), "copy moments back", err);
        }
    }
    ok = ok && gpuCheck(gpuMemcpy(out.streams.data(), dStr,  nCell * 4,  gpuMemcpyDeviceToHost), "copy streams back", err)
            && (!fCaustic || gpuCheck(gpuMemcpy(out.caustic.data(), dCaust, nCell * 4,  gpuMemcpyDeviceToHost), "copy caustic back", err))
            && (!pl.fDispOwn || pl.dispwIsMass || gpuCheck(gpuMemcpy(out.dispw.data(), dDW, nCell * 4, gpuMemcpyDeviceToHost), "copy dispw back", err));
    return ok;
}

} // namespace


bool psGpuDepositFields(std::vector<float>& verts,
                        std::vector<float>& vels,
                        std::vector<float>& masses,
                        std::vector<float>& dens,
                        std::vector<float>& scal,
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        bool fVel, bool fDisp, bool fGrad, bool fLinear,
                        bool fVolW, bool fCaustic, bool fExact, bool fScal, bool fSGrad, bool windowMode,
                        double massExtra, PSFixedUnits const& fxUnits,
                        PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err)
{
    // PS_GPU_TIMING=1: one line per kernel run, and per partition where its call went,
    // before and in its hold of the device (PSGpuPhaseTimes: timed always, printed only then)
    bool const timing = Env::flag({"PS_GPU_TIMING"});
    PSGpuPhaseTimes ph;
    auto tPh = std::chrono::steady_clock::now();
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }
    ph.device += PSGpuPhaseTimes::lap(tPh);

    // Before the hold (ctxMutex below): the plan, the parameter block and the exact items' map, from the host
    // records -- work that needs no device, done while another partition's passes hold it (as ps_metal_host.cc).
    // The plan (the selected grids, their fixed-point form and device sizes; gpu_host.h, shared with the Metal
    // host). PS_GPU_FLOAT_MASS=1 keeps float mass atomics, for comparison.
    bool const floatMass = Env::flag({"PS_GPU_FLOAT_MASS"});
    PSGpuPlan const pl = psGpuPlan(subDims, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, fScal, fSGrad, masses,
                                   massExtra, fxUnits, floatMass);
    size_t const nCell = pl.nCell;
    size_t const sumWBytes = pl.sumWFixed ? 8 : sizeof(float);   // per tet
    deferred.clear();
    if (masses.empty())                 // nothing to deposit (the caller may still add released tets)
    {
        psGpuInitGrids(out, pl, fxUnits);
        psGpuZeroWords(out, pl);
        return true;
    }
    if (verts.size() != masses.size() * size_t(PS_TET_STRIDE))
    {
        err = "tet record size mismatch (verts must hold PS_TET_STRIDE floats per tet)";
        return false;
    }

    DepositParams P = psGpuParams(pl, fxUnits, boxLo, dx, nGrid, subOrigin, subDims, nSub, periodic, fLinear,
                                  windowMode, !dens.empty(), !scal.empty(), masses.size());
    size_t const nTetTotal = masses.size();
    ph.plan += PSGpuPhaseTimes::lap(tPh);

    // --exact-average items (psExactItemBounds in gpu_host.h, as the Metal host): the per-tet window bound comes
    // from the records, read here from the HOST copy before it is uploaded and freed
    bool useItems = fExact;
    useItems = useItems && Env::flag("PS_EXACT_ITEMS", true);
    uint32_t blockCells = 16;
    if (long const v = Env::integer("PS_EXACT_ITEM_CELLS", 0); v > 0) blockCells = uint32_t(v);
    std::vector<uint32_t> items;      // the items' map (psBuildExactItemMap)
    size_t nItems = 0;
    psHostExactItems(PSHostItemsArgs{ verts, P, nGrid, nTetTotal, periodic, windowMode, timing, useItems, blockCells,
                                      items, nItems, ph });

    auto const tAsk = std::chrono::steady_clock::now();   // PS_GPU_TIMING: the wait for the device, and its hold
    std::unique_lock<std::mutex> lock(ctxMutex());   // serialize dispatches (one device); let go after the frees
    auto const tGot = std::chrono::steady_clock::now();

    // Upload the per-tet arrays and free each host array IMMEDIATELY (at most one double-held at
    // a time). The masses keep a host mirror (4 B per tet): the leftovers and the deferrals are
    // read from it, in place of the Metal host's unified-memory reads.
    bool const haveVels = !vels.empty();
    bool const haveDens = !dens.empty();
    bool const haveScal = !scal.empty();
    std::vector<float> hostMass;
    hostMass.swap(masses);
    tPh = std::chrono::steady_clock::now();
    PSHostBuffers const B = psHostUploadBuffers(PSHostUploadArgs{ verts, vels, hostMass, dens, scal, pl, nCell, nTetTotal,
                                                                  sumWBytes, useItems, items, err });
    float *dV = B.dV, *dU = B.dU, *dM = B.dM, *dD = B.dD, *dSc = B.dSc, *dMass = B.dMass, *dMassFx = B.dMassFx,
          *dMom = B.dMom, *dM2 = B.dM2, *dGrad = B.dGrad, *dStr = B.dStr, *dMomW = B.dMomW, *dCaust = B.dCaust,
          *dSV = B.dSV, *dDV = B.dDV, *dDW = B.dDW, *dScal = B.dScal, *dSG = B.dSG;
    unsigned* dBad = B.dBad;
    float* dItems = B.dItems;
    float* dSumW  = B.dSumW;
    ph.uploads += PSGpuPhaseTimes::lap(tPh);
    auto freeAll = [&]()
    {
        for (float* p : { dV, dU, dM, dD, dSc, dMass, dMassFx, dMom, dM2, dGrad, dStr, dMomW, dCaust, dSV, dDV, dDW, dScal, dSG, dItems, dSumW })
            if (p) gpuFree(p);
        if (dBad) gpuFree(dBad);
    };
    if (!dSG)
    {
        if (err.empty()) err = GPU_BACKEND_STRING " allocation failed (out of device memory?)";
        freeAll();
        return false;
    }

    // Dispatch in CHUNKS, one short kernel each, serialized: the Metal host's chunk controller, which bounds a
    // kernel's duration (a display-attached Linux GPU kills kernels after a few seconds). A killed kernel may
    // have deposited part of its atomics, so a failure resumes from the pass's last snapshot (runChunked below);
    // the caller deposits the partition on the CPU only after a sticky error (the context is lost: CUDA's
    // launch timeout is one), 3 failures of one chunk or 16 in the deposit, or when no snapshot can be held.
    // The gap between kernels is zero except while a failure has escalated the pace (GpuPace).
    // PS_GPU_CHUNK=<n> sets the per-tet kernel's starting chunk size.
    size_t const MIN_CHUNK = fExact ? 50 : 2000;
    size_t const MAX_CHUNK = 500000;
    GpuPace& pace = gpuPace();
    size_t baseChunk = psHostFirstChunk(fExact, pace.targetSec, MIN_CHUNK);
    std::string lastErr;
    int const tpb = 256;

    PSHostCkEnv const ck = psHostCheckpointEnv();
    long ckChunks = ck.ckChunks;
    double paceRecoverSec = ck.paceRecoverSec;
    std::vector<DevBuf> const gridBufs = {
        { dMass, pl.massBytes }, { dMassFx, pl.massFxBytes }, { dMom, pl.momBytes }, { dM2, pl.m2Bytes }, { dGrad, pl.gradBytes },
        { dStr, nCell * 4 }, { dMomW, pl.momwBytes }, { dCaust, pl.caustBytes }, { dSV, pl.svBytes }, { dDV, pl.dvBytes },
        { dDW, pl.dwBytes }, { dScal, pl.scalBytes }, { dSG, pl.sgBytes } };
    GpuSnapshot snap;
    int nFailed = 0, nResumedCk = 0;   // failed kernels of this deposit, and how many resumed from a periodic checkpoint
    // The shares the non-finite guard dropped in this deposit, by slot (ps_metal_host.cc badRun): dBad is
    // zeroed before each kernel and read after one that succeeded, and a restore rolls these sums back to the held
    // snapshot's, so a failed kernel and the redone ones count once. Handed to the run's counters only when the
    // deposit is kept.
    unsigned long long badRun[FX_DROP_SLOTS] = {};
    int const MAX_FAILS_SAME_CHUNK = 3, MAX_FAILS_TOTAL = 16;

    // One paced, chunked run of a kernel over nThreads threads; launch(start, n) launches the chunk
    // [start, start+n). 'extras' are the non-grid buffers the kernel WRITES (sumW, the per-tet masses
    // it negates); gridsZero: the grids are all zero at the pass's start (snapshotted as such, without
    // a copy); gridsKeepZero: the pass never writes them (the exact items' phase 0), so its periodic
    // checkpoints skip them too. False (lastErr set) when the pass gave up (the caller falls back).
    auto runChunked = [&](const char* name, size_t nThreads, size_t firstChunk, size_t minChunk,
                          std::vector<DevBuf> const& extras, bool gridsZero, bool gridsKeepZero, auto&& launch) -> bool
    {
        size_t chunk = std::max(firstChunk, minChunk);
        size_t start = 0, nBuf = 0, sinceCk = 0;
        double busy = 0.;
        auto const tRun = std::chrono::steady_clock::now();
        std::vector<DevBuf> ckBufs = gridBufs;
        ckBufs.insert(ckBufs.end(), extras.begin(), extras.end());
        std::vector<bool> zeroMask(ckBufs.size(), false);
        if (gridsZero) std::fill(zeroMask.begin(), zeroMask.begin() + gridBufs.size(), true);
        GpuSnapshot::Took const took = snap.take(ckBufs, zeroMask, 0, lastErr);
        if (took == GpuSnapshot::DeviceError) return false;
        bool ckOn = took == GpuSnapshot::Ok;
        bool ckPeriodic = false;           // the held snapshot is a periodic checkpoint (not the pass start)
        unsigned long long badCk[FX_DROP_SLOTS];   // badRun at the held snapshot
        std::memcpy(badCk, badRun, sizeof badCk);
        // failures of the 'same chunk': a failure that comes before the redo got past the previous failed
        // kernel's end (chunk boundaries move with the timing controller, so a start offset is no identity)
        int  failsHere = 0;
        size_t failEnd = 0;
        while (start < nThreads)
        {
            uint32_t const n = uint32_t(std::min(chunk, nThreads - start));
            auto t0 = std::chrono::steady_clock::now();
            if (!gpuCheck(gpuMemset(dBad, 0, FX_DROP_SLOTS * 4), "zero the guard counters", lastErr)) return false;
            launch(start, n);
            gpuError_t code = gpuGetLastError();
            if (code == gpuSuccess) code = gpuDeviceSynchronize();
            double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            std::string desc;
            GpuFault cause = GpuFault::Resource;
            bool const realError = code != gpuSuccess;
            bool failed = realError;
            if (realError)
            {
                desc = gpuGetErrorString(code);
                cause = code == kGpuErrTimeout ? GpuFault::Watchdog : GpuFault::Resource;
            }
            else failed = gpuFaultInjected(desc, cause, code);   // the test hook (the kernel DID deposit)

            if (failed)
            {
                ++nFailed;
                countGpuRetry(cause, code);   // the run's counters count every failed kernel (run_progress.h)
                lastErr = std::string(GPU_BACKEND_STRING " kernel ") + name + " failed: " + desc + " [" + gpuFaultName(cause) + "]";
                failsHere = (failsHere > 0 && start < failEnd) ? failsHere + 1 : 1;
                failEnd = std::max(failEnd, start + n);
                if (realError)
                {   // a real error: a sticky one has lost the context, the probe returns it again
                    (void)gpuGetLastError();
                    if (gpuDeviceSynchronize() != gpuSuccess)
                    {
                        cause = GpuFault::Fatal;
                        lastErr += " (sticky: the device context is lost)";
                    }
                }
                if (cause == GpuFault::Fatal || !ckOn || failsHere >= MAX_FAILS_SAME_CHUNK || nFailed >= MAX_FAILS_TOTAL)
                {
                    if (!ckOn) lastErr += " (no checkpoint held)";
                    else if (failsHere >= MAX_FAILS_SAME_CHUNK) lastErr += " (" + std::to_string(failsHere) + " failures of the same chunk)";
                    else if (nFailed >= MAX_FAILS_TOTAL) lastErr += " (" + std::to_string(nFailed) + " failed kernels in this deposit)";
                    return false;
                }
                if (cause != GpuFault::Victim)
                {   // escalate (process-wide; GpuPace eases it back): 4x shorter kernels, gaps of 8-50 ms, smaller chunk
                    pace.onFault();
                    chunk = std::max(chunk / 4, minChunk);
                    baseChunk = std::max(baseChunk / 2, MIN_CHUNK);
                }
                std::string const why = lastErr;
                if (!snap.restore(lastErr)) { lastErr = why + " -- " + lastErr; return false; }
                std::memcpy(badRun, badCk, sizeof badCk);
                if (ckPeriodic) ++nResumedCk;
                fprintf(stderr, "%sPS-DTFE %s: %s -- resuming from the %s at thread %zu of %zu "
                                "(redoing %zu; failure %d of this chunk, target %.0f ms kernels, %.0f ms gaps)%s\n",
                        MESSAGE::cYellow(), GPU_BACKEND_STRING, lastErr.c_str(), ckPeriodic ? "checkpoint" : "pass start",
                        snap.start, nThreads, start + n - snap.start, failsHere,
                        pace.targetSec * 1e3, pace.gapUs / 1e3, MESSAGE::cReset());
                start = snap.start;
                sinceCk = 0;
                // an innocent victim retries at once; our own kernels' failures let the device settle first
                auto tGap = std::chrono::steady_clock::now();
                if (cause == GpuFault::Resource) sleep(2);
                else if (cause == GpuFault::Watchdog) sleep(1);
                ph.gaps += PSGpuPhaseTimes::lap(tGap);
                continue;
            }

            start += n;
            ++nBuf;
            ++sinceCk;
            busy += el;
            {
                unsigned words[FX_DROP_SLOTS];
                if (!gpuCheck(gpuMemcpy(words, dBad, sizeof words, gpuMemcpyDeviceToHost), "read the guard counters", lastErr))
                    return false;
                for (int k = 0; k < FX_DROP_SLOTS; ++k) badRun[k] += words[k];
            }
            if (el > 1e-4)   // retarget the next chunk toward targetSec (bounded rescale)
            {
                double scale = pace.targetSec / el;
                if (scale < 0.5) scale = 0.5;
                if (scale > 4.0) scale = 4.0;
                chunk = std::min(std::max(size_t(double(chunk) * scale), minChunk), MAX_CHUNK);
            }
            double clean = 0.;
            if (pace.onClean(el, paceRecoverSec, clean))   // an escalated pace stepped back after a clean stretch (GpuPace)
                fprintf(stderr, "PS-DTFE %s: %.0f s of GPU work without a fault -- pace eased to %.0f ms "
                                "kernels, %.0f ms gaps%s\n", GPU_BACKEND_STRING, clean, pace.targetSec * 1e3,
                        pace.gapUs / 1e3,
                        pace.targetSec >= GPU_PACE_TARGET0 && pace.gapUs <= GPU_PACE_GAP0 ? " (the default)" : "");
            long const every = ckChunks >= 0 ? ckChunks : std::max(1L, long(std::lround(10.0 / pace.targetSec)));
            if (ckOn && every > 0 && long(sinceCk) >= every && start < nThreads)
            {   // synchronized above: the buffers hold exactly [0, start)
                GpuSnapshot::Took const t = snap.take(ckBufs, gridsKeepZero ? zeroMask : std::vector<bool>(), start, lastErr);
                if (t == GpuSnapshot::DeviceError) return false;
                if (t == GpuSnapshot::Ok) { ++snap.taken; ckPeriodic = true; sinceCk = 0; std::memcpy(badCk, badRun, sizeof badCk); }
                // else: no host memory for the spare copy; the last snapshot stays valid
            }
            auto tGap = std::chrono::steady_clock::now();
            if (pace.gapUs) usleep(pace.gapUs);
            ph.gaps += PSGpuPhaseTimes::lap(tGap);
        }
        ph.kernels += busy;
        if (timing)
            fprintf(stderr, "PS-DTFE %s: %-17s %10zu threads  %6zu kernels  %8.3f s busy  %8.3f s wall\n",
                    GPU_BACKEND_STRING, name, nThreads, nBuf, busy,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tRun).count());
        return true;
    };

    // depositFields over the tets [0,nT) of the given records (per-tet buffers at the chunk's offset).
    // The kernel negates the masses of the tets it defers, so mB is part of the snapshot: a redo then
    // runs exactly the same code path.
    auto runPerTet = [&](const float* vB, const float* uB, float* mB, const float* dB, const float* sB, size_t nT,
                         bool gridsZero) -> bool
    {
        return runChunked("depositFields", nT, baseChunk, MIN_CHUNK, { { mB, nT * sizeof(float) } }, gridsZero, false,
                          [&](size_t start, uint32_t n)
        {
            DepositParams Pc = P;
            Pc.nTet = n;
            GPU_LAUNCH(depositFields, blocksFor(n, tpb), tpb, 0,
                vB + start * PS_TET_STRIDE, haveVels ? uB + start * 12 : uB, mB + start,
                dMass, dMom, dM2, dGrad, reinterpret_cast<unsigned*>(dStr), Pc,
                haveDens ? dB + start * 4 : dB,
                dMomW, reinterpret_cast<unsigned*>(dCaust), dSV, dDV, dDW,
                haveScal ? sB + start * 4 : sB, dScal, dSG, reinterpret_cast<unsigned*>(dMassFx), dBad);
        });
    };

    auto runItems = [&](uint32_t phase) -> bool
    {
        // phase 0 writes only sumW (the grids are still zero), phase 1 only the grids (sumW is read)
        std::vector<DevBuf> extras;
        if (phase == 0) extras.push_back({ dSumW, nTetTotal * sumWBytes });
        return runChunked("depositExactItems", nItems, 8192, 1024, extras, true, phase == 0,
                          [&](size_t start, uint32_t n)
        {
            ExactItemParams const IP{ n, phase, blockCells, 0u, uint32_t(start), uint32_t(nTetTotal) };
            if (phase == 0)
                GPU_LAUNCH(depositExactItems<0>, blocksFor(n, tpb), tpb, 0,
                    dV, dU, dM, dMass, dMom, dM2, dGrad, reinterpret_cast<unsigned*>(dStr), P, dD,
                    dMomW, reinterpret_cast<unsigned*>(dCaust), dSV, dDV, dDW,
                    reinterpret_cast<const unsigned*>(dItems), dSumW, IP, dSc, dScal, dSG,
                    reinterpret_cast<unsigned*>(dMassFx), dBad);
            else
                GPU_LAUNCH(depositExactItems<1>, blocksFor(n, tpb), tpb, 0,
                    dV, dU, dM, dMass, dMom, dM2, dGrad, reinterpret_cast<unsigned*>(dStr), P, dD,
                    dMomW, reinterpret_cast<unsigned*>(dCaust), dSV, dDV, dDW,
                    reinterpret_cast<const unsigned*>(dItems), dSumW, IP, dSc, dScal, dSG,
                    reinterpret_cast<unsigned*>(dMassFx), dBad);
        });
    };

    // The tets the items pass leaves over (sumW still 0) run through depositFields; their
    // deferrals are written back into the masses (host mirror AND device copy).
    auto runLeftovers = [&]() -> bool
    {
        auto tL = std::chrono::steady_clock::now();   // (PS_GPU_TIMING: this lambda's own host work, not its kernels)
        // the per-tet totals: floats, or (sumWFixed) the int64 words -- 8 B each, read through a double-sized float
        // vector so the one buffer serves both
        std::vector<float> sw(nTetTotal * (pl.sumWFixed ? 2 : 1));
        if (!gpuCheck(gpuMemcpy(sw.data(), dSumW, nTetTotal * sumWBytes, gpuMemcpyDeviceToHost), "read weight totals", lastErr))
            return false;
        std::vector<uint32_t> left;
        for (size_t t = 0; t < nTetTotal; ++t)
        {
            bool has = false;
            if (pl.sumWFixed) { int64_t q; std::memcpy(&q, &sw[2 * t], 8); has = q > 0; }
            else has = sw[t] > 0.f;
            if (hostMass[t] > 0.f && !has) left.push_back(uint32_t(t));
        }
        if (left.empty()) { ph.leftovers += PSGpuPhaseTimes::lap(tL); return true; }
        size_t const nL = left.size();
        bool const compact = nL <= nTetTotal / 16;
        float *lV = dV, *lU = dU, *lD = dD, *lS = dSc, *lM = nullptr, *dLeft = nullptr;
        bool ok = true;
        if (compact)
        {
            dLeft = devUpload(left.data(), nL * sizeof(uint32_t), "upload leftover indices", lastErr);
            auto gather = [&](const float* src, unsigned stride) -> float* {
                float* b = devAlloc(nL * stride * sizeof(float), "leftover records", lastErr);
                if (!b) return b;
                GPU_LAUNCH(gatherRecords, blocksFor(nL, tpb), tpb, 0, src, reinterpret_cast<const unsigned*>(dLeft), unsigned(nL), stride, b);
                if (!gpuCheck(gpuGetLastError(), "gather launch", lastErr) || !gpuCheck(gpuDeviceSynchronize(), "gather", lastErr))
                { gpuFree(b); return nullptr; }
                return b;
            };
            ok = dLeft != nullptr;
            if (ok) { lV = gather(dV, PS_TET_STRIDE); ok = lV != nullptr; }
            if (ok) { lU = haveVels ? gather(dU, 12) : devAlloc(4, "dummy", lastErr); ok = lU != nullptr; }
            if (ok) { lM = gather(dM, 1); ok = lM != nullptr; }
            if (ok) { lD = haveDens ? gather(dD, 4) : devAlloc(4, "dummy", lastErr); ok = lD != nullptr; }
            if (ok) { lS = haveScal ? gather(dSc, 4) : devAlloc(4, "dummy", lastErr); ok = lS != nullptr; }
        }
        else
        {
            std::vector<float> lm(nTetTotal, 0.f);
            for (uint32_t t : left) lm[t] = hostMass[t];
            lM = devUpload(lm.data(), nTetTotal * sizeof(float), "leftover mass mask", lastErr);
            ok = lM != nullptr;
        }
        if (!ok && lastErr.empty()) lastErr = GPU_BACKEND_STRING " allocation failed (exact-deposit leftovers)";
        ph.leftovers += PSGpuPhaseTimes::lap(tL);
        if (ok) ok = runPerTet(lV, lU, lM, lD, lS, compact ? nL : nTetTotal, false);
        tL = std::chrono::steady_clock::now();
        if (ok)
        {
            std::vector<float> lm(compact ? nL : nTetTotal);
            ok = gpuCheck(gpuMemcpy(lm.data(), lM, lm.size() * sizeof(float), gpuMemcpyDeviceToHost), "read leftover masses", lastErr);
            if (ok)
            {
                for (size_t k = 0; k < nL; ++k)
                {
                    float const v = lm[compact ? k : size_t(left[k])];
                    if (v < 0.f) hostMass[left[k]] = v;
                }
                ok = gpuCheck(gpuMemcpy(dM, hostMass.data(), nTetTotal * sizeof(float), gpuMemcpyHostToDevice), "write back masses", lastErr);
            }
        }
        if (lM) gpuFree(lM);
        if (dLeft) gpuFree(dLeft);
        if (compact) for (float* b : {lV, lU, lD, lS}) if (b) gpuFree(b);
        ph.leftovers += PSGpuPhaseTimes::lap(tL);
        return ok;
    };

    // one pass sequence on the zeroed grids (devAlloc); each pass heals its own failed kernels from its
    // snapshots (runChunked), so a false here means the GPU gave up on this partition and the caller
    // deposits it on the CPU
    bool success;
    if (useItems)
    {
        tPh = std::chrono::steady_clock::now();
        success = gpuCheck(gpuMemset(dSumW, 0, nTetTotal * sumWBytes), "zero weight totals", lastErr);
        ph.uploads += PSGpuPhaseTimes::lap(tPh);
        success = success && runItems(0) && runItems(1) && runLeftovers();
    }
    else
        success = runPerTet(dV, dU, dM, dD, dSc, nTetTotal, true);
    psHostReportPasses(GPU_BACKEND_STRING, "kernel", success, nFailed, nResumedCk, snap, timing);
    // the hold ends with the frees (as ps_metal_host.cc): the next partition allocates its whole device set as soon
    // as it takes the device, and this host has no free-memory gate, so this call's buffers must be gone by then
    // (two partitions' sets side by side can overflow VRAM: an allocation failure, a CPU fallback). The device is
    // synchronized first, so a kernel's late error fails the passes rather than the copy-back.
    if (success && !gpuCheck(gpuDeviceSynchronize(), "synchronize after the passes", lastErr)) success = false;
    if (!success)
    {
        err = lastErr;
        freeAll();
        return false;
    }

    tPh = std::chrono::steady_clock::now();
    psGpuInitGrids(out, pl, fxUnits);   // (the output grids, allocated only now: a partition waiting for the GPU holds none)
    bool ok = psHostCopyBack(PSHostCopyArgs{ B, pl, nCell, nTetTotal, fCaustic, hostMass, out, deferred, err });
    ph.copyBack += PSGpuPhaseTimes::lap(tPh);
    freeAll();
    ph.releases += PSGpuPhaseTimes::lap(tPh);
    auto const tFree = tPh;
    lock.unlock();
    // the deposit is kept (its grids are back): its dropped shares join the run's counts (reported once at the end,
    // DTFE.cpp); a failed copy-back leaves them to the CPU re-deposit, which counts its own
    if (ok)
        for (int k = 0; k < FX_DROP_SLOTS; ++k)
            if (badRun[k]) fixedShareDropped(k, badRun[k]);
    if (timing && ok)   // the hold covers the uploads, the passes, the copy-back and the frees (as ps_metal_host.cc)
    {
        double const held = std::chrono::duration<double>(tFree - tGot).count();
        psGpuReportPhases(GPU_BACKEND_STRING, PSMassAudit::partition, ph, snap.sec, held);
        fprintf(stderr, "PS-DTFE %s: partition %d waited %.3f s for the GPU and held it %.3f s\n",
                GPU_BACKEND_STRING, PSMassAudit::partition, std::chrono::duration<double>(tGot - tAsk).count(), held);
    }
    return ok;
}
