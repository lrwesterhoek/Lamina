/* Backend-neutral host interface to the GPU deposit kernels, implemented by the Metal
   backend (ps_metal_host.cc / dtfe_metal_host.cc, 'make <target> METAL=1' on macOS/Apple
   Silicon) and by the CUDA/HIP backend (ps_gpu_cuda.cu / dtfe_gpu_cuda.cu, 'CUDA=1' or
   'HIP=1' on Linux; single-source ports of the Metal kernels). The callers never see which.

   A GPU build defines PS_GPU (PS-DTFE) / DTFE_GPU (standard); the callers in
   ps_interpolation.cc and averaged_interpolation_1.cc guard every use on those and fall
   back to the CPU path on any failure. The GPU results match the CPU path to float
   rounding (atomic summation order) -- the parity contract is validated by
   tests/dtfe_metal_check.sh and metal/validate_deposit.cpp (Metal; the CUDA/HIP port is
   compiled and fallback-tested on Linux, not yet validated on hardware).

   This header is self-contained (no DTFE/CGAL/backend includes; ps_deposit_params.h is plain C++, and
   ../message.h gives the shared host steps at the end their colours). */

#ifndef GPU_HOST_HEADER
#define GPU_HOST_HEADER

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ps_deposit_params.h"   // PSFxGrid / PSFixedUnits: the moment grids' fixed-point words
#include "../message.h"          // ANSI colour helpers only (psHostReportPasses)

// Name of the backend compiled into this binary ("Metal", "CUDA" or "HIP").
std::string gpuBackendName();

// Name of the GPU device (empty if unavailable); initializes the device on first call.
std::string gpuDeviceName();


// ---------------------------------------------------------------- PS-DTFE deposit
// Output grids, all sized to the partition sub-grid (subDims product = subTotal). Only the
// grids selected by the fVel/fDisp/fGrad flags are allocated (mom/m2/grad stay EMPTY when
// their flag is off -- the m2 and grad grids alone are 24+36 B per sub-grid cell); mass and
// streams are always produced.
struct PSGpuGrids
{
    // The MASS grid (density = mass / cellVolume), in one of two forms:
    //   massScale > 0  'massFx' holds each cell's exact 64-bit fixed-point sum in units of 1/massScale (the
    //                  kernels' own words, handed over unconverted) and 'mass' stays EMPTY. The caller adds the
    //                  GPU-deferred and released tetrahedra into the same integers and converts to float once
    //                  (ps_interpolation.cc), so no deposit after the kernel rounds a share away.
    //   massScale = 0  the float grid in 'mass' (DTFE_GPU_FLOAT_MASS=1, kept for A/B, or a
    //                  total mass that gives no usable unit) and 'massFx' stays EMPTY.
    std::vector<float>    mass;     // subTotal      accumulated MASS (float form; see above)
    std::vector<int64_t>  massFx;   // subTotal      accumulated MASS (fixed-point form; see above)
    float                 massScale = 0.f;
    std::vector<float>    mom;      // subTotal*3    weighted velocity moment  [i*3+j]   (fVel)
    std::vector<float>    m2;       // subTotal*6    weighted 2nd moment, upper triangle [i*6+c] (fDisp)
    std::vector<float>    grad;     // subTotal*9    weighted velocity-gradient moment [i*9+j*3+i'] (fGrad)
    std::vector<uint32_t> streams;  // subTotal      interior-sample count (streams = count / nSub^3)
    std::vector<float>    momw;     // subTotal      moment-weight normalizer sum (fVolW; else EMPTY -- 'mass' is the normalizer)
    std::vector<uint32_t> caustic;  // subTotal      orientation bits, OR-accumulated (fCaustic; else EMPTY)
    std::vector<float>    dispvel;  // subTotal*3    dispersion's own mass-weighted mean sum(m_s v_s) (fVolW+fDisp; else EMPTY)
    std::vector<float>    dispw;    // subTotal      its normalizer sum(m_s)                (fVolW+fDisp; else EMPTY)
    std::vector<float>    streamvol;// subTotal      exact cell-mean multiplicity sum(V_int)/V_cell (fExact; else EMPTY). 'streams' then carries the raw integer tet-touch count.
    std::vector<float>    scal;     // subTotal      scalar moment sum(wm s), one component (fScal; else EMPTY)
    std::vector<float>    sgrad;    // subTotal*3    scalar-gradient moment sum(wm ds/dx_i) [i*3+d] (fSGrad; else EMPTY)
    // The moment grids as exact fixed-point sums (PSFxGrid in ps_deposit_params.h): fxScale[g] > 0 =
    // grid g came back as the kernels' 64-bit words in fx[g] (subTotal x its components, the float vector's
    // layout), in units of 1/fxScale[g], and its float vector above stays EMPTY -- the caller adds the deferred
    // and released tetrahedra into the same integers and converts once, as for the mass. 0 = the float vector.
    // dispwIsMass: the kernel did not deposit 'dispw' (the fixed mass grid sums the same shares): read massFx.
    std::vector<int64_t>  fx[PS_FX_N];
    float                 fxScale[PS_FX_N] = {};
    bool                  dispwIsMass = false;
    // DTFE_MASS_AUDIT=1 (ps_mass_audit.h) only, else -1: the fixed-point mass grid's exact total and the
    // (float) masses of the tets the kernel deferred, summed by the host before it converts/frees them
    double auditFixedTotal = -1., auditDeferredMass = -1.;
};

// Runs the mass-conserving tetrahedral deposit on the default GPU device. 'verts' holds one
// PS_TET_STRIDE-float record per tet (ps_deposit_params.h psFillTetRecord: vertex 0 as cell
// index + fraction and the edges, in the canonical frame of the exact inside test).
// Thread-safe: the calls prepare their work and copy their grids back concurrently, and serialize their device
// work (uploads and passes) on one device queue/stream.
// The kernel decides which samples each tet contains exactly as the CPU deposit does
// (ps_exact_inside.h): in float, with a rigorous error bound. A tet with a sample inside that
// bound is NOT deposited; its index (into the arrays as passed) is returned in 'deferred', and
// the caller must deposit it with the CPU's exact test.
// fVel = velocity moments needed (velocity or dispersion selected), fDisp = second moments
// (dispersion), fGrad = velocity-gradient moments; unselected grids are neither allocated on
// the device (4-byte dummies) nor on the host. 'vels' may be EMPTY when all three are off.
// fLinear (--ps-linear-deposit) weights the interior samples by the linear density built
// from the per-tet vertex densities in 'dens' (nTet*4; may be EMPTY when fLinear is off),
// renormalized per tet so the deposited total still equals the tet mass exactly.
// fVolW (--ps-volume-weighted) switches the mom/m2/grad moment weights to equal Eulerian-
// volume shares (|det|/6 per tet, computed in-kernel) and fills out.momw with their
// normalizer; the mass grid is unaffected. Mutually exclusive with fLinear (rejected at
// option parsing). fCaustic (--ps-caustics) ORs each tet's orientation bits (1 = det>0,
// 2 = det<0) into out.caustic at every deposited sample -- OR is order-invariant, so the
// result matches the CPU deposit's bits (float-vs-double det sign only differs below the
// degeneracy cut, where cells are dropped).
// fExact (--exact-average) replaces the nSub^3 sampling by the kernels' float32 exact clipping
// (sweep_moments in metal/exact_clip.metal.inc: a sweep in z, analytic tet-cell volume + order-2 moments); nSub is then
// ignored. The CPU path stays
// the DOUBLE-precision reference -- the GPU exact deposit agrees to float rounding, like
// every other GPU/CPU field pair.
// fScal / fSGrad: the scalar field (ONE component, NO_SCALARS=1 builds; the caller declines the GPU
// otherwise): 'scal' holds the four vertex scalars per tet (nTet*4; EMPTY when both are off), and
// the kernel deposits the linear profile's value (and the constant gradient) times the moment
// weight, exactly as the velocity moment, into out.scal / out.sgrad.
// windowMode (--grid-window): the sub-box is a window of the full grid, not a partition's own box --
// every sample / clipped piece of a tet is counted for its normalization as in the full run, only
// those inside the sub-box are deposited.
// massExtra: the mass the caller adds to the mass grid itself on top of 'masses' (the
// --ps-halo-release centroid deposits); the fixed-point unit (psMassFixedScale) is set from sum(masses) +
// massExtra, so the caller's own integer adds cannot overflow a cell. The deferred tets are part of 'masses'.
// fxUnits: the units of the moment grids, chosen by the caller from bounds that cover its own adds
// too (the deferred and released tets): a grid with scale > 0 is deposited as an exact fixed-point sum and
// handed back as words (out.fx), the rest as float atomics; sumW makes the exact items' weight totals exact.
// Returns false with 'err' set if there is no device, setup fails or the GPU gives up on the deposit; 'out' is then
// untouched (its grids are shaped and filled only after a deposit the GPU kept; a CUDA/HIP copy-back that fails
// returns false with them partly filled -- the caller reads 'out' only after true).
// CONSUMES verts/vels/masses/dens/scal (several GB per partition at scale): the backend frees each one
// right after copying it to the device, so they may be empty when the call returns --
// the CPU fallback deposits from the triangulation, not from these arrays.
bool psGpuDepositFields(std::vector<float>& verts,   // nTet*PS_TET_STRIDE: the per-tet records
                        std::vector<float>& vels,    // nTet*12: 4 vertex velocities (empty if unused)
                        std::vector<float>& masses,  // nTet: tet mass (rho_bar * V_lag, or the --ps-vertex-mass shares)
                        std::vector<float>& dens,    // nTet*4: vertex densities (empty unless fLinear)
                        std::vector<float>& scal,    // nTet*4: vertex scalars (empty unless fScal / fSGrad)
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        bool fVel, bool fDisp, bool fGrad, bool fLinear,
                        bool fVolW, bool fCaustic, bool fExact, bool fScal, bool fSGrad, bool windowMode,
                        double massExtra, PSFixedUnits const& fxUnits,
                        PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err);


// ---------------------------------------------------------------- standard-DTFE method-1 deposit
// Output grids, sized to the interpolation grid (nGrid product = nCell). Each holds the
// accumulated value*volume contributions; the caller divides by the grid-cell volume
// (identical to the CPU loop). Only the requested fields are allocated; the rest stay empty.
struct DTFEGpuGrids
{
    std::vector<float> density;   // nCell      accumulated density * volume
    std::vector<float> velocity;  // nCell*3    accumulated velocity * volume     [i*3+j]
    std::vector<float> grad;      // nCell*9    accumulated velocity gradient * volume [i*9 + j*3 + i'],
                                  //            layout matching the CPU velocityGradient() Pvector
    std::vector<float> scalar;    // nCell      accumulated scalar * volume (one component; fScal)
    std::vector<float> scalar_grad; // nCell*3  accumulated scalar gradient * volume [i*3 + d] (fSGrad)
};

// Runs the method-1 volume-averaged deposit on the default GPU device. Vertex positions
// must be relative to the region's lower corner; vols/cnts/flats carry the CPU-computed
// tetrahedron volume, sample count (0 = single-grid-cell fast path), and fast-path flat
// grid index. 'sobol' is the shared quasi-random barycentric table (maxNN*3, every count
// is <= maxNN). Thread-safe (serialized).
// Returns false with 'err' set on any failure; outputs untouched in the "no device" case.
// CONSUMES verts/dens/vels/vols/cnts/flats (see psGpuDepositFields); the CPU fallback
// interpolates from the triangulation, not from these arrays.
// fScal / fSGrad: the scalar field (ONE component, NO_SCALARS=1 builds; the caller declines the GPU
// otherwise): 'scal' holds the four vertex scalars per tet (empty when both are off); the kernels
// deposit it like the density (linear over the tet) and its constant gradient like the velocity
// gradient.
// exact (--exact-average): the analytic volume average instead of the samples -- every tetrahedron
// is clipped against each grid cell of its bounding box (the kernels' float32 z-sweep, sweep_moments) and
// deposits the linear interpolant's integral over each piece; 'cnts' then only marks the
// single-cell tetrahedra (0), which take the centroid fast path as on the CPU. Matches the CPU's
// double-precision r3d to float rounding.
// The device sums every grid as an exact 64-bit fixed-point integer per cell (dtfe_gpu_fixed.h
// sets the units from a bound of the batch's shares; metal/dtfe_deposit.metal's header says how the kernels keep
// the sum independent of the order their threads add in) and the host converts each cell to these floats once,
// so two runs give byte-identical grids (DTFE_GPU_FLOAT_GRIDS=1 keeps the float atomics
// for A/B). A failed command buffer / kernel resumes from the batch's last checkpoint, the PS deposit's policy
// (gpu_metal_retry.h / gpu_cuda_retry.cuh); a false return after the upload means that policy gave up.
bool dtfeGpuDepositAveraged1(std::vector<float>& verts,      // nTet*12: 4 vertices, region-relative
                             std::vector<float>& dens,       // nTet*4:  vertex densities
                             std::vector<float>& vels,       // nTet*12: 4 vertex velocities (empty if unused)
                             std::vector<float>& scal,       // nTet*4:  vertex scalars (empty if unused)
                             std::vector<float>& vols,       // nTet:    tetrahedron volumes
                             std::vector<uint32_t>& cnts,    // nTet:    sample counts (0 = fast path)
                             std::vector<uint32_t>& flats,   // nTet:    fast-path flat grid indices
                             const std::vector<float>& sobol,// maxNN*3 barycentric table (not consumed)
                             const double dx[3], const size_t nGrid[3],
                             bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, bool exact,
                             DTFEGpuGrids& out, std::string& err);

// ---- shared accumulator: the sub-domains of one DTFE_parallel call deposit into ONE full grid.
// Begin (main thread, before the parallel region) allocates and zeroes the full grids on the
// GPU -- false when memory would not allow it or there is no device, and then every thread
// takes the per-call path above. Each thread then extracts its tetrahedra exactly
// as for dtfeGpuDepositAveraged1 (positions and fast-path flat indices relative to ITS
// sub-grid) and hands them over with the sub-grid's origin 'subOff' in the full grid; the
// deposit of samples inside the sub-grid lands at full-grid cells subOff + cell, so every
// cell has one owner and the batches never overlap. Dispatches are serialized on the device
// queue (the chunk controller stays warm across batches); uploads run concurrently. A failed
// batch leaves its owner box zeroed and returns false: the caller deposits that sub-domain on
// the CPU (or via the per-call path) into its own sub-grid, and the final merge ADDS the GPU
// grids into the main grid -- a cell is non-zero in exactly one of the two.
// Result gives read pointers (accumulated value*volume, full-grid layout; a pointer is null for
// a field not requested) that stay valid until End; End frees the GPU grids. The shared grid keeps the float
// atomic sums (its one unit would have to suit every batch, which no batch's bound sets), and a
// failed buffer restarts its batch from the zeroed owner box, without periodic checkpoints.
bool dtfeGpuSharedBegin(const size_t fullGrid[3], bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, std::string& err);
bool dtfeGpuSharedActive();
bool dtfeGpuSharedDeposit(std::vector<float>& verts, std::vector<float>& dens, std::vector<float>& vels,
                          std::vector<float>& scal,
                          std::vector<float>& vols, std::vector<uint32_t>& cnts, std::vector<uint32_t>& flats,
                          const std::vector<float>& sobol, const double dx[3], const size_t nGrid[3],
                          const size_t subOff[3], std::string& err);
bool dtfeGpuSharedResult(const float** den, const float** vel, const float** grad, const float** scal, const float** sgrad);
void dtfeGpuSharedEnd();

// --grid-window, the exact deposit (CPU loop and both GPU hosts' item bounds; the kernels carry a twin):
// the contiguous hull, in RAW (unwrapped) cell indices, of the part of a tetrahedron's span [lo, hi)
// that lies in the window [o, o+m) or one of its periodic images (o+m may exceed n: a window wrapping
// the seam). False when the span misses the window. A tetrahedron is shorter than the box, so at
// most two images meet its span; their hull can hold a gap of in-grid cells outside the window,
// which the per-cell window test skips cheaply. A non-periodic span is clamped to the grid first.
inline bool psWindowHull(long lo, long hi, long n, bool periodic, long o, long m, long& outLo, long& outHi)
{
    if (hi <= lo || m <= 0 || n <= 0) return false;
    if (!periodic)
    {
        lo = lo < 0 ? 0 : lo;  hi = hi > n ? n : hi;
        if (hi <= lo) return false;
    }
    bool any = false;
    long hlo = 0, hhi = 0;
    int const kMax = periodic ? 2 : 0;
    for (int k = -kMax; k <= kMax; ++k)
    {
        long const a = lo > o + k*n ? lo : o + k*n;
        long const b = hi < o + m + k*n ? hi : o + m + k*n;
        if (b <= a) continue;
        if (!any) { hlo = a; hhi = b; any = true; }
        else { if (a < hlo) hlo = a; if (b > hhi) hhi = b; }
    }
    if (!any) return false;
    outLo = hlo; outHi = hhi;
    return true;
}


// ---------------------------------------------------------------- the PS-DTFE GPU hosts' common steps
// ps_metal_host.cc and ps_gpu_cuda.cu prepare a psGpuDepositFields call the same way: the plan (which grids, which
// of them as fixed-point words, their device sizes, the mass unit), the output grids' initial shapes, the kernels'
// parameter block and the exact items' window bounds.

// components per cell of the moment grids, by PSFxGrid index
inline constexpr size_t PS_FX_COMPS[PS_FX_N] = { 3, 6, 9, 1, 1, 3, 1, 1, 3 };

// One call's plan. Only the requested moment grids are allocated, host- and device-side: the m2 (24 B/cell) and
// grad (36 B/cell) grids dominate the deposit's footprint and most runs need neither. A grid the caller gave a
// fixed-point unit (fxUnits) is a 64-bit integer sum: 8 B per component in place of the float's 4, bound at its
// float index AND at its word index (22 + slot, ps_deposit.metal). An unselected grid's bytes are 4 (a dummy).
struct PSGpuPlan
{
    bool     fVel = false, fDisp = false, fGrad = false, fVolW = false, fCaustic = false, fExact = false,
             fScal = false, fSGrad = false;   // the call's flags
    size_t   nCell = 0;
    bool     fDispOwn = false;      // fVolW && fDisp: the dispersion keeps its own mass-weighted mean+normalizer
    bool     sel[PS_FX_N] = {};     // the moment grids the flags select (PSFxGrid order)
    bool     fxOn[PS_FX_N] = {};    // ... deposited as fixed-point words
    uint32_t fxMask = 0;            // PSDepositParams.fxMask
    bool     massFixed = false;     // the mass grid as an exact 64-bit fixed-point sum in units of 1/massScale
    float    massScale = 1.f;
    bool     dispwIsMass = false;   // dispw is not deposited: the fixed mass grid sums its shares (PSGpuGrids)
    bool     sumWFixed = false;     // the exact items' per-tet weight totals as fixed-point sums
    size_t   momBytes = 0, m2Bytes = 0, gradBytes = 0, momwBytes = 0, caustBytes = 0, svBytes = 0, dvBytes = 0,
             dwBytes = 0, scalBytes = 0, sgBytes = 0, massBytes = 0, massFxBytes = 0;
};

// The plan of a call (its arguments as psGpuDepositFields takes them). The mass grid is an exact 64-bit fixed-point
// sum (DepositParams.massFixed in the kernel says why) unless floatMass, the host's A/B switch to float atomics
// (DTFE_GPU_FLOAT_MASS=1). Its unit covers the caller's own
// deposits too (massExtra): the caller adds the deferred and released tets into these words.
inline PSGpuPlan psGpuPlan(const size_t subDims[3], bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                           bool fExact, bool fScal, bool fSGrad, std::vector<float> const& masses, double massExtra,
                           PSFixedUnits const& fxUnits, bool floatMass)
{
    PSGpuPlan p;
    p.fVel = fVel; p.fDisp = fDisp; p.fGrad = fGrad; p.fVolW = fVolW; p.fCaustic = fCaustic; p.fExact = fExact;
    p.fScal = fScal; p.fSGrad = fSGrad;
    size_t const nCell = p.nCell = subDims[0] * subDims[1] * subDims[2];
    p.fDispOwn = fVolW && fDisp;
    bool const sel[PS_FX_N] = { fVel, fDisp, fGrad, fVolW, fExact, p.fDispOwn, false, fScal, fSGrad };
    for (int g = 0; g < PS_FX_N; ++g)
    {
        p.sel[g] = sel[g];
        p.fxOn[g] = sel[g] && fxUnits.scale[g] > 0.f;
        if (p.fxOn[g]) p.fxMask |= 1u << g;
    }
    auto gridBytes4 = [&](int g, size_t comps) -> size_t { return sel[g] ? nCell * comps * (p.fxOn[g] ? 8 : 4) : 4; };
    p.momBytes   = gridBytes4(PS_FX_MOM, 3);
    p.m2Bytes    = gridBytes4(PS_FX_M2, 6);
    p.gradBytes  = gridBytes4(PS_FX_GRAD, 9);
    p.momwBytes  = gridBytes4(PS_FX_MOMW, 1);
    p.caustBytes = fCaustic ? nCell * 4 : 4;
    p.svBytes    = gridBytes4(PS_FX_SV, 1);
    p.dvBytes    = gridBytes4(PS_FX_DV, 3);
    p.scalBytes  = gridBytes4(PS_FX_SCAL, 1);
    p.sgBytes    = gridBytes4(PS_FX_SG, 3);
    {
        double totalMass = massExtra > 0. ? massExtra : 0.;
        // (a non-finite mass -- bad input, or the DTFE_TEST_NAN_SHARE hook -- is left out of the unit: its shares are
        // the non-finite guard's to drop and count, not a reason to lose the whole grid's fixed point)
        for (float const mt : masses) if (std::isfinite(mt)) totalMass += double(mt);
        if (fxUnits.massTotal > 0.) totalMass = fxUnits.massTotal;   // (a --grid-window pass: the caller's total)
        p.massFixed = !floatMass && psMassFixedScale(totalMass, p.massScale);
    }
    // the dispersion's own normalizer sums exactly the mass grid's shares: with both exact it is not deposited
    // at all (the caller reads the mass words; PSGpuGrids::dispwIsMass), else float atomics
    p.dispwIsMass = p.fDispOwn && p.massFixed && fxUnits.scale[PS_FX_DW] > 0.f;
    if (p.dispwIsMass) p.fxMask |= 1u << PS_FX_DW;
    p.dwBytes = (p.fDispOwn && !p.dispwIsMass) ? nCell * 4 : 4;
    p.sumWFixed = fExact && fxUnits.sumW;
    if (p.sumWFixed) p.fxMask |= PS_FX_SUMW_BIT;
    p.massBytes   = p.massFixed ? 4 : nCell * 4;   // the float grid (Metal buffer 3), else a dummy
    p.massFxBytes = p.massFixed ? nCell * 8 : 4;   // the fixed-point grid (Metal buffer 21), else a dummy
    return p;
}

// The output grids' initial shapes: each selected float grid zeroed; a fixed-point grid comes back as words
// (out.fx[g], out.massFx: filled from the device grid at the end) and leaves its float vector EMPTY.
inline void psGpuInitGrids(PSGpuGrids& out, PSGpuPlan const& p, PSFixedUnits const& fxUnits)
{
    size_t const nCell = p.nCell;
    out.dispwIsMass = p.dispwIsMass;
    for (int g = 0; g < PS_FX_N; ++g) { out.fx[g].clear(); out.fxScale[g] = p.fxOn[g] ? fxUnits.scale[g] : 0.f; }
    out.mass.assign(p.massFixed ? 0 : nCell, 0.f);
    out.massFx.clear();
    out.massScale = p.massFixed ? p.massScale : 0.f;
    out.mom.assign(p.fVel && !p.fxOn[PS_FX_MOM] ? nCell * 3 : 0, 0.f);
    out.m2.assign(p.fDisp && !p.fxOn[PS_FX_M2] ? nCell * 6 : 0, 0.f);
    out.grad.assign(p.fGrad && !p.fxOn[PS_FX_GRAD] ? nCell * 9 : 0, 0.f);
    out.streams.assign(nCell, 0u);
    out.momw.assign(p.fVolW && !p.fxOn[PS_FX_MOMW] ? nCell : 0, 0.f);
    out.caustic.assign(p.fCaustic ? nCell : 0, 0u);
    out.streamvol.assign(p.fExact && !p.fxOn[PS_FX_SV] ? nCell : 0, 0.f);
    out.dispvel.assign(p.fDispOwn && !p.fxOn[PS_FX_DV] ? nCell * 3 : 0, 0.f);
    out.dispw.assign(p.fDispOwn && !p.dispwIsMass ? nCell : 0, 0.f);
    out.scal.assign(p.fScal && !p.fxOn[PS_FX_SCAL] ? nCell : 0, 0.f);
    out.sgrad.assign(p.fSGrad && !p.fxOn[PS_FX_SG] ? nCell * 3 : 0, 0.f);
}

// The result of a call with no tetrahedra (the caller may still add released ones): its fixed-point grids as zeros.
inline void psGpuZeroWords(PSGpuGrids& out, PSGpuPlan const& p)
{
    if (p.massFixed) out.massFx.assign(p.nCell, 0);
    for (int g = 0; g < PS_FX_N; ++g) if (p.fxOn[g]) out.fx[g].assign(p.nCell * PS_FX_COMPS[g], 0);
}

// The kernels' parameter block of a plan; haveDens / haveScal: the per-tet vertex densities / scalars were handed over.
inline PSDepositParams psGpuParams(PSGpuPlan const& p, PSFixedUnits const& fxUnits, const double boxLo[3],
                                   const double dx[3], const size_t nGrid[3], const size_t subOrigin[3],
                                   const size_t subDims[3], int nSub, bool periodic, bool fLinear, bool windowMode,
                                   bool haveDens, bool haveScal, size_t nTet)
{
    PSDepositParams P;
    std::memset(&P, 0, sizeof(P));
    for (int d = 0; d < 3; ++d)
    {
        P.boxLo[d]     = float(boxLo[d]);
        P.dx[d]        = float(dx[d]);
        P.nGrid[d]     = int32_t(nGrid[d]);
        P.subOrigin[d] = int32_t(subOrigin[d]);
        P.subDims[d]   = int32_t(subDims[d]);
    }
    P.nSub     = nSub < 1 ? 1 : nSub;
    P.periodic = periodic ? 1 : 0;
    P.fVel     = p.fVel  ? 1 : 0;
    P.fDisp    = p.fDisp ? 1 : 0;
    P.fGrad    = p.fGrad ? 1 : 0;
    P.fLinear  = (fLinear && haveDens) ? 1 : 0;
    P.fVolW    = p.fVolW    ? 1 : 0;
    P.fCaustic = p.fCaustic ? 1 : 0;
    P.fExact   = p.fExact   ? 1 : 0;
    P.nTet     = uint32_t(nTet);
    P.fScal    = (p.fScal  && haveScal) ? 1 : 0;
    P.fSGrad   = (p.fSGrad && haveScal) ? 1 : 0;
    P.windowMode = windowMode ? 1 : 0;
    P.massFixed = p.massFixed ? 1 : 0;
    P.massScale = p.massScale;
    P.fxMask = p.fxMask;               // the moment grids' fixed-point sums (the caller's units)
    for (int g = 0; g < PS_FX_N; ++g) P.fxScale[g] = p.fxOn[g] ? fxUnits.scale[g] : 0.f;
    // PS_GPU_DEBUG_NOATOMIC=1: a timing experiment (the atomics' share of the sampled kernel, the counterpart of
    // DTFE_GPU_DEBUG_NOATOMIC) -- the sampled path does all its work but deposits nothing, so the grids come out empty
    P.debugNoAtomic = Env::flag("PS_GPU_DEBUG_NOATOMIC") ? 1u : 0u;
    return P;
}

// --exact-average's work items: an item is (tet, block of blockCells cells of its bbox window); the host only
// needs an UPPER bound of each window -- the kernel re-derives it in float and idles surplus blocks. The float and
// double floors can differ only when the value is within rounding of an integer, so only such a bound is widened
// (by eps, far above the float error): padding every axis by a cell instead would inflate a 2-3 cell window up to
// 8x, and each idle item still pays the setup. Under windowMode the kernel's items cover the window's hull only
// (window_hull), so the bound does too. recs = the nTet records (PS_TET_STRIDE floats each); bound[t] = tet t's
// window cells, 0 for a window wholly outside the grid (no item: a leftover); blockCells grows so that at most
// ~32M items result (16-32 cells run within 10%; the cap keeps the item counts, and so the sums, of the
// list of (tet, block) pairs this map replaced); nItems = the item count. Returns the summed bound.
inline double psExactItemBounds(float const* recs, size_t nTet, PSDepositParams const& P, const size_t nGrid[3],
                                bool periodic, bool windowMode, std::vector<uint64_t>& bound, uint32_t& blockCells,
                                size_t& nItems)
{
    bound.assign(nTet, 0);
    double total = 0.;
    for (size_t t = 0; t < nTet; ++t)
    {
        float const* R = recs + t * PS_TET_STRIDE;
        uint64_t nw = 1;
        for (int dd = 0; dd < 3 && nw; ++dd)
        {
            int32_t ic0; std::memcpy(&ic0, &R[dd], sizeof ic0);
            double lo = 0., hi = 0.;
            for (int e = 0; e < 3; ++e) { double const a = R[6 + e*3 + dd]; lo = std::min(lo, a); hi = std::max(hi, a); }
            double const f0 = R[3 + dd], dxd = double(P.dx[dd]);
            double const vlo = f0 + lo / dxd, vhi = f0 + hi / dxd;
            double const flo = std::floor(vlo - (1.e-3 + 1.e-5 * std::fabs(vlo)));
            double const fhi = std::floor(vhi + (1.e-3 + 1.e-5 * std::fabs(vhi)));
            if (!std::isfinite(flo) || !std::isfinite(fhi) || fhi - flo > 1.e7) { nw = 0; break; }
            int64_t iLo = int64_t(ic0) + int64_t(flo);
            int64_t iHi = int64_t(ic0) + int64_t(fhi) + 1;
            if (!periodic) { iLo = std::max<int64_t>(iLo, 0); iHi = std::min<int64_t>(iHi, int64_t(nGrid[dd])); }
            if (windowMode)
            {
                long hl = 0, hh = 0;
                if (!psWindowHull(long(iLo), long(iHi), long(nGrid[dd]), periodic, long(P.subOrigin[dd]), long(P.subDims[dd]), hl, hh)) { nw = 0; break; }
                iLo = hl; iHi = hh;
            }
            nw = iHi > iLo ? nw * uint64_t(iHi - iLo) : 0;
        }
        bound[t] = nw;
        total += double(nw);
    }
    double const maxItems = 32.e6;
    if (total / blockCells > maxItems) blockCells = uint32_t(std::ceil(total / maxItems));
    nItems = 0;
    for (size_t t = 0; t < nTet; ++t)
        nItems += (bound[t] + blockCells - 1) / blockCells;
    return total;
}

// The exact items as a compact map from psExactItemBounds' per-tet bounds, the standard-DTFE kernel's form
// (dtfe_gpu_fixed.h dtfeItemMap): tet t's items are its blocks k = 0 .. ceil(bound[t] / blockCells) - 1, item
// numbers itemStart[t] + k, in tet order -- the same (tet, block) pairs in the same order as a list of pairs, at
// 4 B per tet plus 4 B per 2^PS_ITEM_BLOCK_SHIFT items instead of 8 B per item. The kernels decode an item with a
// short binary search (psItemOf in metal/ps_deposit.metal and ps_gpu_cuda.cu). 'map' holds psExactItemMapWords
// words: itemStart[0 .. nTet] (itemStart[nTet] = nItems), then blockTet[b] = the tet holding item b << SHIFT, for
// b = 0 .. (nItems >> SHIFT) + 1 (the last ones past the items hold nTet - 1). A tet with no items (bound 0: a
// leftover) has itemStart[t] == itemStart[t + 1] and no block points at it. The Metal host fills its GPU buffer
// in place, the CUDA host a host vector it then uploads. Item numbers are 32-bit: psExactItemMapFits first.
inline bool psExactItemMapFits(size_t nTet, size_t nItems) { return nTet < 0xFFFFFFFFu && nItems < 0xFFFFFFFFu; }
inline size_t psExactItemMapWords(size_t nTet, size_t nItems) { return (nTet + 1) + (nItems >> PS_ITEM_BLOCK_SHIFT) + 2; }
inline void psBuildExactItemMap(std::vector<uint64_t> const& bound, uint32_t const blockCells, size_t const nItems,
                                uint32_t* map)
{
    size_t const nTet = bound.size();
    uint32_t* const itemStart = map;
    uint32_t* const blockTet = map + nTet + 1;
    size_t n = 0;
    for (size_t t = 0; t < nTet; ++t)
    {
        itemStart[t] = uint32_t(n);
        n += (bound[t] + blockCells - 1) / blockCells;
    }
    itemStart[nTet] = uint32_t(n);
    size_t const nBlocks = (nItems >> PS_ITEM_BLOCK_SHIFT) + 2;
    std::fill(blockTet, blockTet + nBlocks, uint32_t(nTet ? nTet - 1 : 0));
    for (size_t t = 0, blk = 0; t < nTet && blk < nBlocks; ++t)
        while (blk < nBlocks && (blk << PS_ITEM_BLOCK_SHIFT) < size_t(itemStart[t + 1])) blockTet[blk++] = uint32_t(t);
}

// The per-tet kernel's opening chunk: by deposit kind or PS_GPU_CHUNK, scaled down under an
// escalated pace (paceTargetSec: the backend's GpuPace::targetSec).
inline size_t psHostFirstChunk(bool const fExact, double const paceTargetSec, size_t const MIN_CHUNK)
{
    // --exact-average costs ~1-2 orders of magnitude more per tet (analytic clipping of
    // every cell in the window vs a barycentric test per sample), so start far smaller: the
    // adaptive controller only retargets AFTER the first chunk, and one oversized opening
    // buffer is exactly what the watchdog kills.
    size_t baseChunk = fExact ? 500 : 25000;
    if (long const v = Env::integer({"PS_GPU_CHUNK"}, 0); v > 0) baseChunk = size_t(v);
    // an escalated pace (an earlier partition's failures) opens with a proportionally smaller chunk
    if (paceTargetSec < 0.25)
        baseChunk = std::max(size_t(double(baseChunk) * paceTargetSec / 0.25), MIN_CHUNK);
    return baseChunk;
}

// The chunk checkpoints' settings (DTFE_GPU_CHECKPOINT_CHUNKS, DTFE_GPU_PACE_RECOVER_SEC).
struct PSHostCkEnv
{
    long   ckChunks;
    double paceRecoverSec;
};

// Reads how often a pass checkpoints and after how much clean GPU work an escalated pace steps back.
inline PSHostCkEnv psHostCheckpointEnv()
{
    // DTFE_GPU_CHECKPOINT_CHUNKS=<n>: a checkpoint every n command buffers / kernels (0 = only the pass-start
    // snapshot). Unset: every ~10 s of GPU work (n = 10 s / the current buffer target, so an escalated pace of
    // 0.02 s buffers does not checkpoint 12x as often). A checkpoint copies the accumulator grids (~24 B per cell
    // for density + velocity: ~0.4 GB at 256^3, ~0.1 s; the CUDA/HIP host device-to-host) into ONE spare host
    // copy, allocated at the first checkpoint and held until the partition's deposit returns.
    long ckChunks = -1;
    if (Env::isSet("DTFE_GPU_CHECKPOINT_CHUNKS")) ckChunks = std::max(0L, Env::integer("DTFE_GPU_CHECKPOINT_CHUNKS", 0));
    // DTFE_GPU_PACE_RECOVER_SEC=<s>: the clean GPU work after which an escalated pace steps back (GpuPace)
    double paceRecoverSec = 90.;
    if (double const v = Env::real("DTFE_GPU_PACE_RECOVER_SEC", -1.); v >= 0.) paceRecoverSec = v;
    return PSHostCkEnv{ ckChunks, paceRecoverSec };
}

// Reports the passes' failed command buffers / kernels and the checkpoints (after any failure, else under
// DTFE_GPU_TIMING). backend: "Metal", "CUDA" or "HIP"; unit: what failed, "buffer" (Metal) or "kernel" (CUDA/HIP);
// snap: the backend's GpuSnapshot (gpu_metal_retry.h / gpu_cuda_retry.cuh).
template <class Snapshot>
inline void psHostReportPasses(char const* backend, char const* unit, bool const success, int const nFailed,
                               int const nResumedCk, Snapshot const& snap, bool const timing)
{
    if (nFailed > 0)
        fprintf(stderr, "%sPS-DTFE %s: deposit %s after %d failed %s(s) (%d resumed from a periodic checkpoint); "
                        "%d checkpoint(s), %.0f MB held, %.3f s copying%s\n",
                success ? MESSAGE::cYellow() : MESSAGE::cRed(), backend, success ? "recovered on the GPU" : "gave up",
                nFailed, unit, nResumedCk, snap.taken, snap.bytes / 1.e6, snap.sec, MESSAGE::cReset());
    else if (timing)
        fprintf(stderr, "PS-DTFE %s: checkpoints: %d, %.0f MB held, %.3f s copying\n", backend, snap.taken,
                snap.bytes / 1.e6, snap.sec);
}

// DTFE_GPU_TIMING=1: where a psGpuDepositFields call spends its time, phase by phase, in seconds (both hosts). The
// hold of the GPU runs from taking the host's ctxMutex to the release of the call's buffers: the GPU idles in all
// of it but 'kernels'. The call prepares its work before it takes the lock, so the partitions that queue for the
// GPU do that work while another one holds it; it copies its grids back and frees its buffers before it lets go,
// so one deposit's buffers exist at a time: the auto-tuner books them once (Metal: unified memory), and on CUDA/HIP,
// with no free-memory gate, a second set beside the first could overflow the device (an allocation failure: the
// CPU fallback).
//   before the hold:
//   device     the device and its kernels' first use (ctx(): the runtime compile of the Metal source, once)
//   plan       the plan and the parameter block (psGpuPlan, psGpuParams)
//   bounds     the exact items' window bounds (psExactItemBounds)
//   fill       the exact items' map (psBuildExactItemMap; Metal: into its GPU buffer)
//   in the hold:
//   uploads    the per-tet records to the device, the zeroed grids and the exact items' per-tet totals
//              (psHostUploadBuffers; CUDA: and the items' map)
//   kernels    the passes' command buffers / kernels that succeeded (their 'busy' seconds)
//   gaps       the idle gaps between them (the pace) and the settling sleeps after a failure
//   leftovers  the exact items' leftovers around their own kernels: the scan, the gathers, the write-back
//   copyBack   the output grids' shapes (psGpuInitGrids), the deferred tets' scan and the grids back to the host
//              (psHostCopyBack)
//   releases   the device buffers' release
// The checkpoints' copies are GpuSnapshot::sec; what is left of the hold (encoding, the guard counters' reads, the
// failed buffers' own time) is reported as 'other'.
struct PSGpuPhaseTimes
{
    double device = 0., plan = 0., uploads = 0., bounds = 0., fill = 0., kernels = 0., gaps = 0., leftovers = 0.,
           copyBack = 0., releases = 0.;
    // the seconds since t, and t restarted: consecutive phases chain on one clock
    static double lap(std::chrono::steady_clock::time_point& t)
    {
        auto const now = std::chrono::steady_clock::now();
        double const s = std::chrono::duration<double>(now - t).count();
        t = now;
        return s;
    }
};

// One line per partition (DTFE_GPU_TIMING): the phases before and in a hold of 'held' seconds, with snapSec = the
// checkpoints' copies (in the hold).
inline void psGpuReportPhases(char const* backend, int const partition, PSGpuPhaseTimes const& t, double const snapSec,
                              double const held)
{
    double const other = std::max(0., held - (t.uploads + t.kernels + t.gaps + snapSec + t.leftovers + t.copyBack
                                              + t.releases));   // (disjoint spans)
    fprintf(stderr, "PS-DTFE %s: partition %d before the hold: device %.3f s, plan %.3f s, item bounds %.3f s, "
                    "item fill %.3f s; hold by phase: uploads %.3f s, kernels %.3f s, pacing gaps %.3f s, "
                    "snapshots %.3f s, leftovers %.3f s, copy-back %.3f s, releases %.3f s, other %.3f s\n",
            backend, partition, t.device, t.plan, t.bounds, t.fill, t.uploads, t.kernels, t.gaps, snapSec, t.leftovers,
            t.copyBack, t.releases, other);
}

#endif
