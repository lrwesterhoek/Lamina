/* Host interface to the GPU deposit kernels, implemented by the Metal backend (ps_metal_host.cc, dtfe_metal_host.cc).
   Callers see only these declarations and the guards (PS_GPU / DTFE_GPU), never a Metal type, and fall back to the CPU
   path on any failure; results match it to float rounding. Self-contained: no DTFE/CGAL/backend includes. */

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
#include <utility>
#include <vector>

#include "ps_deposit_params.h"   // PSFxGrid / PSFixedUnits: the moment grids' fixed-point words
#include "../message.h"          // ANSI colour helpers only (psHostReportPasses)

// Name of the backend compiled into this binary ("Metal").
std::string gpuBackendName();

// Name of the GPU device (empty if unavailable); initializes the device on first call.
std::string gpuDeviceName();

// The cells per pass the GPU kernels number: 2^32 (32-bit cell indices). A pass whose own grid holds this many cells
// or more deposits on the CPU with a warning and a counted fall-back (ps_interpolation.cc psGpuEligible,
// averaged_interpolation_1.cc avg1GpuUsable); the planner's own cap is auto_tune.cc kGpuCellCap. The test hook
// DTFE_TEST_GPU_CELL_CAP=<n> lowers it to n cells (1..2^32), read once, so a small grid reaches both guards
// (tests/defaults_check.sh G); the warnings still name 2^32.
inline size_t gpuCellCap()
{
    static size_t const cap = [] {
        long const v = Env::integer("DTFE_TEST_GPU_CELL_CAP", 0);
        return v >= 1 && int64_t(v) <= (int64_t(1) << 32) ? size_t(v) : size_t(1) << 32;
    }();
    return cap;
}


// ---------------------------------------------------------------- PS-DTFE deposit
// Output grids, sized to the partition sub-grid (subTotal cells). Grids not selected by fVel/fDisp/fGrad stay EMPTY;
// mass and streams are always produced.
struct PSGpuGrids
{
    // The MASS grid (density = mass / cellVolume): massScale > 0: 'massFx' holds each cell's exact fixed-point sum in
    // units of 1/massScale (the kernels' own 64-bit words) and 'mass' stays EMPTY; the caller adds the deferred and
    // released tets into the same integers and converts once, so no deposit rounds a share away. massScale = 0: the
    // float grid in 'mass' (a total mass that gives no usable unit).
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
    // The moment grids as exact fixed-point sums (PSFxGrid): fxScale[g] > 0 = grid g came back as words in fx[g] (the
    // float vector's layout) in units of 1/fxScale[g] and its float vector stays EMPTY; the caller adds the deferred
    // and released tets into the same integers and converts once, as for the mass. 0 = the float vector.
    // dispwIsMass: the kernel did not deposit 'dispw' (the fixed mass grid sums the same shares): read massFx.
    std::vector<int64_t>  fx[PS_FX_N];
    float                 fxScale[PS_FX_N] = {};
    bool                  dispwIsMass = false;
    // DTFE_MASS_AUDIT=1 (ps_mass_audit.h) only, else -1: the fixed-point mass grid's exact total and the
    // (float) masses of the tets the kernel deferred, summed by the host before it converts/frees them
    double auditFixedTotal = -1., auditDeferredMass = -1.;
};

// Runs the mass-conserving tetrahedral deposit on the default GPU device (thread-safe: calls prepare and copy back
// concurrently, device work is serialized). The kernel decides which samples a tet contains exactly as the CPU
// deposit does (ps_exact_inside.h, rigorous float bound); a tet with a sample inside the bound is NOT deposited: its
// index goes to 'deferred' for the CPU's exact test. Returns false with 'err' set if the GPU gives up ('out' then
// untouched). CONSUMES verts/vels/masses/dens/scal: freed right after upload (the CPU fallback uses the triangulation).
bool psGpuDepositFields(std::vector<float>& verts,   // nTet*PS_TET_STRIDE: the per-tet records
                        std::vector<float>& vels,    // nTet*12: 4 vertex velocities (empty if unused)
                        std::vector<float>& masses,  // nTet: tet mass (rho_bar * V_lag, or the --ps-vertex-mass shares)
                        std::vector<float>& dens,    // nTet*4: vertex densities (empty unless fLinear)
                        std::vector<float>& scal,    // nTet*4: vertex scalars (empty unless fScal / fSGrad)
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        bool fVel, bool fDisp, bool fGrad, bool fLinear,
                        // fVolW: moment weights = equal Eulerian-volume shares (out.momw their normalizer); fCaustic: ORs
                        // orientation bits (order-invariant, so equal to the CPU's); fExact (--exact): float32 exact
                        // clipping, nSub read only by a tet with an empty window; fScal / fSGrad: ONE scalar component
                        // ('scal' = four vertex scalars per tet); windowMode (--grid-window): the sub-box is a window of
                        // the full grid, normalized as in the full run
                        bool fVolW, bool fCaustic, bool fExact, bool fScal, bool fSGrad, bool windowMode,
                        // massExtra: mass the caller adds itself; the fixed unit covers sum(masses) + massExtra. fxUnits:
                        // the moment grids' units (cover the caller's own adds); scale > 0: handed back as words in out.fx
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

// The run-wide fixed-point units of the five grids (fixed_units.h RunUnits; order of DTFEDepositParams.fxScale: density,
// velocity, velocity gradient, scalar, its gradient). One unit 2^-K[g] per grid for the whole run, so every split
// rounds every share of an exact batch alike. on false: a run without them (batches keep their own units); has[g]
// false: K lies outside [-120, 120], or a scalar grid of '--approx-phase-space-density' (DTFE.cpp runUnitsSetup).
struct DTFEGpuRunUnits
{
    bool     on = false;
    bool     has[5] = {};
    int      K[5] = {};
    uint32_t capBits[5] = {};   // the float bits of each grid's share cap (RunUnits::capBits; the kernels' compare)
};

// Runs the method-1 volume-averaged deposit on the default GPU device (thread-safe, serialized). Positions are relative
// to the base cell / region corner; cnts = sample count (0 = single-cell fast path); exact (--exact): analytic volume
// average, 'cnts' only marks single-cell tets. Every grid is an exact fixed-point sum per cell (metal/dtfe_deposit.metal
// header), converted to float once, so two runs give byte-identical grids: 128-bit words in the run-wide 'units' for
// an exact batch, else 64-bit words (dtfe_gpu_fixed.h). False with 'err' set on failure; CONSUMES the host arrays.
bool dtfeGpuDepositAveraged1(std::vector<float>& verts,      // nTet*12: 4 vertices, relative to the base cell's corner (cells) or the region's
                             std::vector<float>& dens,       // nTet*4:  vertex densities
                             std::vector<float>& vels,       // nTet*12: 4 vertex velocities (empty if unused)
                             std::vector<float>& scal,       // nTet*4:  vertex scalars (empty if unused)
                             std::vector<float>& vols,       // nTet:    tetrahedron volumes
                             std::vector<uint32_t>& cnts,    // nTet:    sample counts (0 = fast path)
                             std::vector<uint32_t>& flats,   // nTet:    fast-path flat grid indices
                             std::vector<int32_t>& cells,    // nTet*3:  each tetrahedron's base cell (sub-grid indices, may lie outside it): the frame of its positions; 0 = the region's corner
                             const std::vector<float>& sobol,// maxNN*3 barycentric table (not consumed)
                             const double dx[3], const size_t nGrid[3],
                             bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, bool exact,
                             DTFEGpuRunUnits const& units,   // the run-wide units (exact batches)
                             DTFEGpuGrids& out, std::string& err);

// --grid-window, the exact deposit (CPU loop and the PS GPU host's item bounds; the kernel carries a twin): the
// contiguous hull, in RAW (unwrapped) cell indices, of the part of a tetrahedron's span [lo, hi) inside the window
// [o, o+m) or one of its periodic images (o+m may exceed n). False when the span misses the window. At most two
// images meet a span; their hull can hold a gap of cells outside the window, which the per-cell test skips. A
// non-periodic span is clamped to the grid first.
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
// The steps of ps_metal_host.cc's psGpuDepositFields that are plain C++: the plan (which grids, which of them as
// fixed-point words, their device sizes, the mass unit), the output grids' initial shapes, the kernels' parameter
// block and the exact items' window bounds.

// components per cell of the moment grids, by PSFxGrid index
inline constexpr size_t PS_FX_COMPS[PS_FX_N] = { 3, 6, 9, 1, 1, 3, 1, 1, 3 };

// One call's plan. Only the requested moment grids are allocated, host- and device-side. A grid with a fixed-point
// unit (fxUnits) is an integer sum: 8 B per component (64-bit words) in place of 4, bound at its float index AND at
// its word index (22 + slot, ps_deposit.metal). An unselected grid's bytes are 4 (a dummy).
struct PSGpuPlan
{
    bool     fVel = false, fDisp = false, fGrad = false, fVolW = false, fCaustic = false, fExact = false,
             fScal = false, fSGrad = false;   // the call's flags
    size_t   nCell = 0;
    bool     fDispOwn = false;      // fVolW && fDisp: the dispersion keeps its own mass-weighted mean+normalizer
    bool     sel[PS_FX_N] = {};     // the moment grids the flags select (PSFxGrid order)
    bool     fxOn[PS_FX_N] = {};    // ... deposited as fixed-point words
    uint32_t fxMask = 0;            // PSDepositParams.fxMask
    bool     massFixed = false;     // the mass grid as an exact fixed-point sum in units of 1/massScale
    float    massScale = 1.f;
    bool     dispwIsMass = false;   // dispw is not deposited: the fixed mass grid sums its shares (PSGpuGrids)
    bool     sumWFixed = false;     // the exact items' per-tet weight totals as fixed-point sums
    size_t   momBytes = 0, m2Bytes = 0, gradBytes = 0, momwBytes = 0, caustBytes = 0, svBytes = 0, dvBytes = 0,
             dwBytes = 0, scalBytes = 0, sgBytes = 0, massBytes = 0, massFxBytes = 0;
};

// The plan of a call (its arguments as psGpuDepositFields takes them). The mass grid is an exact fixed-point
// sum (DepositParams.massFixed in the kernel says why), or float atomics when its total mass gives no usable unit.
// Its unit covers the caller's own deposits too (massExtra): the caller adds the deferred and released tets into
// these words.
inline PSGpuPlan psGpuPlan(const size_t subDims[3], bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                           bool fExact, bool fScal, bool fSGrad, std::vector<float> const& masses, double massExtra,
                           PSFixedUnits const& fxUnits)
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
    auto gridBytes4 = [&](int g, size_t comps) -> size_t {
        return sel[g] ? nCell * comps * (p.fxOn[g] ? 8 : 4) : 4; };
    p.momBytes   = gridBytes4(PS_FX_MOM, 3);
    p.m2Bytes    = gridBytes4(PS_FX_M2, 6);
    p.gradBytes  = gridBytes4(PS_FX_GRAD, 9);
    p.momwBytes  = gridBytes4(PS_FX_MOMW, 1);
    p.caustBytes = fCaustic ? nCell * 4 : 4;
    p.svBytes    = gridBytes4(PS_FX_SV, 1);
    p.dvBytes    = gridBytes4(PS_FX_DV, 3);
    p.scalBytes  = gridBytes4(PS_FX_SCAL, 1);
    p.sgBytes    = gridBytes4(PS_FX_SG, 3);
    double totalMass = massExtra > 0. ? massExtra : 0.;
    // (a non-finite mass -- bad input, or the DTFE_TEST_NAN_SHARE hook -- is left out of the unit: its shares are
    // the non-finite guard's to drop and count, not a reason to lose the whole grid's fixed point)
    for (float const mt : masses) if (std::isfinite(mt)) totalMass += double(mt);
    if (fxUnits.massTotal > 0.) totalMass = fxUnits.massTotal;   // (a --grid-window pass: the caller's total)
    p.massFixed = psMassFixedScale(totalMass, p.massScale);
    // the dispersion's own normalizer sums exactly the mass grid's shares: with both exact it is not deposited
    // at all (the caller reads the mass words; PSGpuGrids::dispwIsMass), else float atomics
    p.dispwIsMass = p.fDispOwn && p.massFixed && fxUnits.scale[PS_FX_DW] > 0.f;
    if (p.dispwIsMass) p.fxMask |= 1u << PS_FX_DW;
    p.dwBytes = (p.fDispOwn && !p.dispwIsMass) ? nCell * 4 : 4;
    p.sumWFixed = fExact && fxUnits.sumW;
    if (p.sumWFixed) p.fxMask |= PS_FX_SUMW_BIT;
    p.massBytes   = p.massFixed ? 4 : nCell * 4;   // the float grid (Metal buffer 3), else a dummy
    // the fixed-point grid (Metal buffer 21: 8 B per cell), else a dummy
    p.massFxBytes = p.massFixed ? nCell * 8 : 4;
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
    P.fxWords = 0u;                    // the device words: the 64-bit ones (the library's PS_FX_WORDS 0)
    P.fxCapBits = 0u;                  // (no share cap: the 64-bit words' guard, at 2^63)
    return P;
}

// --exact's work items: an item is (tet, block of blockCells cells of its bbox window); the host needs only an UPPER
// bound of each window (the kernel re-derives it in float and idles surplus blocks), widened by eps only where the
// float and double floors can differ; under windowMode it covers the window's hull. bound[t] = tet t's window cells
// (0: no item, a leftover); blockCells grows so that at most ~32M items result. Returns the summed
// bound. Cap and blockCells fix sumW (depositExactItems rounds each item's float sum once): another value changes it.
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

// The exact items as a compact map from psExactItemBounds' bounds (dtfe_gpu_fixed.h dtfeItemMap's form): tet t's
// items are its blocks k = 0 .. ceil(bound[t] / blockCells) - 1, numbered itemStart[t] + k in tet order. 'map' holds
// psExactItemMapWords words: itemStart[0 .. nTet] (itemStart[nTet] = nItems), then blockTet[b] = the tet holding item
// b << SHIFT, b = 0 .. (nItems >> SHIFT) + 1 (those past the items hold nTet - 1); the kernels decode an item by a
// short binary search (psItemOf). Item numbers are 32-bit: psExactItemMapFits first.
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
    // --exact costs far more per tet (analytic clipping of every cell in the window), so start far smaller: the
    // controller only retargets AFTER the first chunk, and one oversized opening buffer is what the watchdog kills.
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
    // DTFE_GPU_CHECKPOINT_CHUNKS=<n>: a checkpoint every n command buffers (0 = only the pass-start snapshot). Unset:
    // every ~10 s of GPU work (n = 10 s / the current buffer target). A checkpoint copies the accumulator grids into
    // ONE spare host copy, allocated at the first checkpoint and freed before the deposit lets go of the GPU.
    long ckChunks = -1;
    if (Env::isSet("DTFE_GPU_CHECKPOINT_CHUNKS")) ckChunks = std::max(0L, Env::integer("DTFE_GPU_CHECKPOINT_CHUNKS", 0));
    // DTFE_GPU_PACE_RECOVER_SEC=<s>: the clean GPU work after which an escalated pace steps back (GpuPace)
    double paceRecoverSec = 90.;
    if (double const v = Env::real("DTFE_GPU_PACE_RECOVER_SEC", -1.); v >= 0.) paceRecoverSec = v;
    return PSHostCkEnv{ ckChunks, paceRecoverSec };
}

// Reports the passes' failed command buffers and the checkpoints (after any failure, else under
// DTFE_GPU_TIMING). backend: "Metal"; unit: what failed, "buffer"; snap: the host's GpuSnapshot (gpu_metal_retry.h).
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
// call prepares before it takes the host's ctxMutex (device, plan, bounds, fill), then holds the GPU (uploads, kernels,
// gaps, leftovers, copyBack, releases); the GPU idles in all of the hold but 'kernels'. The checkpoints' copies are
// GpuSnapshot::sec; the rest of the hold (encoding, guard-counter reads, failed buffers) is reported as 'other'.
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
