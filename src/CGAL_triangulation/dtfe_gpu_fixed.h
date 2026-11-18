/* The standard-DTFE GPU deposit's exact sums: the units of its fixed-point grids, for
   the GPU host of the standard binary (dtfe_metal_host.cc). Host-side only, plain C++ (no
   backend includes). An exact batch takes the run-wide units and 128-bit words (src/fixed_units.h RunUnits,
   dtfeBatchParams below); a sampled batch takes the 64-bit words in units from its own bound (batchFixedScales),
   whose rule is the phase-space mass grid's (ps_deposit_params.h: psMassFixedScale picks the unit, psMassFixedMass
   converts a cell back). Why the grids are integer sums, and how the kernels keep them independent of the order
   their atomics land in: metal/dtfe_deposit.metal's header.
   Below batchFixedScales: the other host-side steps that need no Metal type (the knobs, a batch's parameter block
   and item map, the timing line). */
#ifndef DTFE_GPU_FIXED_H
#define DTFE_GPU_FIXED_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu_host.h"            // DTFEGpuGrids, DTFEGpuRunUnits
#include "ps_deposit_params.h"
#include "../env_knobs.h"

// The per-batch units (a sampled batch's 64-bit grids, an exact batch's of a run without the units, and an exact
// batch's grid without a run-wide unit: dtfeBatchParams; the kernel's header says why the grids are integer sums):
// grid g holds integers in units of 1/scale[g], scale = 2^k, set per BATCH from a bound B of the sum of |share| over
// everything the batch can deposit -- no cell, holding part of that sum, can exceed it. psMassFixedScale (the PS mass
// grid's rule) puts 2B at 2^61..2^62 units, so B itself sits at 2^60..2^61: a factor 4 below the 2^63 of the signed
// words for the kernel's float rounding of a sample value (the inverse edge matrix of a tetrahedron near the
// degeneracy cut amplifies it; a few per cent at worst). B, per grid, sums V_tet * max|value| over the batch's
// tetrahedra: a sample, a clipped piece's centroid and the single-cell centroid all lie in the tetrahedron, where the
// linear interpolant is a convex combination of the vertex values as uploaded (the Monte-Carlo centring shift is in
// them already). The two gradients are per-tetrahedron constants: computed here in double from the kernel's own
// float edges and differences, with the degeneracy cut relaxed to half the kernel's (inverse3zero's 1e-6 of the
// row-length product), so a tetrahedron the float kernel keeps is never bounded as zero. The resolution this
// leaves: a 512^3 sub-domain of 1.5e7 cells puts a cell holding 1e-3 of the mean share at ~2^26 units -- 2^-26 of
// its own value, finer than the float (2^-24) the grid is handed over in. False (the batch keeps the float atomics) when a bound is not
// finite or has no usable power of two; an all-zero grid gets unit 1.
inline bool batchFixedScales(const std::vector<float>& verts, const std::vector<float>& dens, const std::vector<float>& vels,
                             const std::vector<float>& scal, const std::vector<float>& vols,
                             bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, float scale[5])
{
    size_t const nTet = vols.size();
    double B[5] = {0., 0., 0., 0., 0.};
    bool const needInv = fGrad || fSGrad;
    for (size_t t = 0; t < nTet; ++t)
    {
        double const V = std::fabs(double(vols[t]));
        if (fDen)
        {
            double m = 0.;
            for (int v = 0; v < 4; ++v) m = std::max(m, std::fabs(double(dens[t*4 + v])));
            B[0] += V * m;
        }
        if (fVel)
        {
            double m = 0.;
            for (int k = 0; k < 12; ++k) m = std::max(m, std::fabs(double(vels[t*12 + k])));
            B[1] += V * m;
        }
        if (fScal)
        {
            double m = 0.;
            for (int v = 0; v < 4; ++v) m = std::max(m, std::fabs(double(scal[t*4 + v])));
            B[3] += V * m;
        }
        if (!needInv) continue;
        // the kernel's edge matrix (float differences, as it forms them), its cofactors and determinant in double;
        // the gradient is cof^T df / det, so its largest component is max|sum_k cof[k][i] df[k]| / |det|. The cut
        // compares squares (no square roots: this loop runs over every tetrahedron of the batch)
        double A[3][3];
        for (int v = 0; v < 3; ++v)
            for (int d = 0; d < 3; ++d)
                A[v][d] = double(float(verts[t*12 + (v+1)*3 + d] - verts[t*12 + d]));
        double const C[3][3] = {   // C[i][k] = the inverse's [i][k] times det (inverse3zero's formulas)
            {  A[1][1]*A[2][2]-A[1][2]*A[2][1], -(A[0][1]*A[2][2]-A[0][2]*A[2][1]),  A[0][1]*A[1][2]-A[0][2]*A[1][1] },
            { -(A[1][0]*A[2][2]-A[1][2]*A[2][0]),  A[0][0]*A[2][2]-A[0][2]*A[2][0], -(A[0][0]*A[1][2]-A[0][2]*A[1][0]) },
            {  A[1][0]*A[2][1]-A[1][1]*A[2][0], -(A[0][0]*A[2][1]-A[0][1]*A[2][0]),  A[0][0]*A[1][1]-A[0][1]*A[1][0] } };
        double const det = A[0][0]*C[0][0] + A[0][1]*C[1][0] + A[0][2]*C[2][0];
        double rn2 = 1.;
        for (int i = 0; i < 3; ++i) rn2 *= A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2];
        if (!(det * det > 0.25e-12 * rn2)) continue;      // |det| <= 0.5e-6 rn: the kernel zeroes this one's gradients too
        double const Vd = V / std::fabs(det);
        // max over i of |sum_k C[i][k] * df[k]| for the component differences df (the kernel's float ones)
        auto gradMax = [&](const float* f0, size_t stride, int comps) {
            double m = 0.;
            for (int j = 0; j < comps; ++j)
            {
                double df[3];
                for (int k = 0; k < 3; ++k) df[k] = double(float(f0[(k+1)*stride + j] - f0[j]));
                for (int i = 0; i < 3; ++i)
                    m = std::max(m, std::fabs(C[i][0]*df[0] + C[i][1]*df[1] + C[i][2]*df[2]));
            }
            return m;
        };
        if (fGrad)  B[2] += Vd * gradMax(&vels[t*12], 3, 3);
        if (fSGrad) B[4] += Vd * gradMax(&scal[t*4], 1, 1);
    }
    bool const on[5] = { fDen, fVel, fGrad, fScal, fSGrad };
    for (int g = 0; g < 5; ++g)
    {
        scale[g] = 1.f;
        if (!on[g] || B[g] == 0.) continue;
        if (!psMassFixedScale(2. * B[g], scale[g])) return false;
    }
    return true;
}


// ---------------------------------------------------------------- knobs
// Work items (default): every tetrahedron becomes max(1, ceil(cnt / itemSamples)) threads, so no
// thread loops over more than itemSamples samples -- the sample count per tetrahedron reaches
// maxNN = 10000 on fine grids, which makes the per-tet kernel's chunks wildly non-uniform and
// trips the GPU watchdog -- and the kernel's per-thread cell cache turns hundreds of per-sample
// atomics into one set per distinct cell (see the kernel header). DTFE_GPU_ITEMS=0 runs the
// per-tet kernel for A/B comparisons. An item runs at most itemSamples = 48 samples; the cell cache
// has 2 slots (the kernel depositAveraged1Items2). Longer items ran faster up to about 48 samples and
// gained little beyond, and dropping the atomics altogether saved little: the kernel is compute/occupancy-bound.
struct DTFEGpuKnobs
{
    bool     useItems    = true;
    uint32_t itemSamples = 48;   // samples per work item
};

// The knobs from the environment; itemsOk: the host can run the item kernels at all.
inline DTFEGpuKnobs dtfeGpuKnobs(bool itemsOk)
{
    DTFEGpuKnobs k;
    k.useItems = itemsOk;
    k.useItems = k.useItems && Env::flag("DTFE_GPU_ITEMS", true);
    return k;
}


// ---------------------------------------------------------------- one sub-domain's batch
// The batch's parameter block, in the host's DepositParams (zeroed by the host; the fields named here are
// the same in both): the exact sums' mode and units, before the upload consumes the batch's values, the grids'
// geometry, the field flags and the item length.
//   The exact sums. An EXACT batch (--exact) sums in the run-wide 128-bit words (P.fixedSum = 2): every grid in the
// run's unit 2^-units.K[g] and below its share cap (fixed_units.h RunUnits: one unit per grid for the whole run, from
// global quantities, so every partition and thread split rounds every share alike; no bound pass runs). A grid
// without a run-wide unit (units.has false: its K outside [-120, 120], or a scalar grid of
// '--approx-phase-space-density') takes this batch's own unit (batchFixedScales) in the same words, which hold either.
// A SAMPLED batch, and every batch of a run without the units (units.on false), sums in the 64-bit words in units
// from the batch's own bound (P.fixedSum = 1), or in float atomics (0) when a bound is not finite.
// DTFE_TEST_RUN_UNIT_SHIFT=n makes the run-wide units 2^n finer (RunUnits::fill); the batch's own units take no shift.
// Returns the exact kernel's item, a block of exactCells = 16 cells of the tetrahedron's window, which is then
// P.itemSamples.
template <class Params>
inline uint32_t dtfeBatchParams(Params& P, DTFEGpuKnobs const& k, const std::vector<float>& verts,
                                const std::vector<float>& dens, const std::vector<float>& vels,
                                const std::vector<float>& scal, const std::vector<float>& vols, const double dx[3],
                                const size_t nGrid[3], const size_t subOff[3], const size_t fullGrid[3], bool fDen,
                                bool fVel, bool fGrad, bool fScal, bool fSGrad, bool exact, DTFEGpuRunUnits const& units)
{
    P.fixedSum = 0;
    for (int g = 0; g < 5; ++g) P.fxScale[g] = 1.f;
    for (int g = 0; g < 5; ++g) P.fxCapBits[g] = 0u;
    bool const runWide = exact && units.on;
    if (runWide)
    {   // the run-wide 128-bit words; a grid without a run-wide unit takes the batch's own (none: no bound pass)
        bool const on[5] = { fDen, fVel, fGrad, fScal, fSGrad };
        bool own[5];
        bool anyOwn = false;
        for (int g = 0; g < 5; ++g) { own[g] = on[g] && !units.has[g]; anyOwn = anyOwn || own[g]; }
        float ownScale[5] = { 1.f, 1.f, 1.f, 1.f, 1.f };
        if (!anyOwn
            || batchFixedScales(verts, dens, vels, scal, vols, own[0], own[1], own[2], own[3], own[4], ownScale))
        {
            P.fixedSum = 2u;
            for (int g = 0; g < 5; ++g)
            {
                P.fxScale[g] = own[g] ? ownScale[g] : units.has[g] ? std::ldexp(1.f, units.K[g]) : 1.f;
                P.fxCapBits[g] = units.capBits[g];
            }
        }
    }
    else
        P.fixedSum = batchFixedScales(verts, dens, vels, scal, vols, fDen, fVel, fGrad, fScal, fSGrad, P.fxScale) ? 1u : 0u;
    for (int d = 0; d < 3; ++d)
    {
        P.dx[d]       = float(dx[d]);
        P.invDx[d]    = 1.0f / float(dx[d]);
        P.nGrid[d]    = int32_t(nGrid[d]);
        P.subOff[d]   = int32_t(subOff[d]);
        P.fullGrid[d] = int32_t(fullGrid[d]);
    }
    P.fDen  = fDen ? 1 : 0;
    P.fVel  = fVel ? 1 : 0;
    P.fGrad = fGrad ? 1 : 0;
    P.fScal = fScal ? 1 : 0;
    P.fSGrad = fSGrad ? 1 : 0;
    P.itemSamples = k.itemSamples;
    constexpr uint32_t exactCells = 16;   // cells per item of the exact kernel
    if (exact) P.itemSamples = exactCells;
    return exactCells;
}

// The item map (see the kernel's itemOf): itemStart[t] = first item of tetrahedron t (nTet+1 entries) and, per
// block of 2^blockShift items, the tetrahedron holding the block's first. A tetrahedron of n samples takes
// ceil(n / itemSamples) items, a single-cell one (n = 0) one; under 'exact' a sampled one takes
// ceil(window / exactCells) items of an UPPER bound of the kernel's cell window (its float floor can differ from
// this double one only within rounding of an integer, which the margin covers; surplus blocks idle). Without
// items only the counts are taken. Adds the batch's items, samples and single-cell tetrahedra to nItems /
// nSamples / nSingle; false (err set) beyond 2^32 items.
inline bool dtfeItemMap(std::vector<float> const& verts, std::vector<uint32_t> const& cnts,
                        std::vector<int32_t> const& cells, const double dx[3],
                        const size_t nGrid[3], bool useItems, bool exact, uint32_t itemSamples, uint32_t exactCells,
                        int blockShift, std::vector<uint32_t>& itemStart, std::vector<uint32_t>& blockTet,
                        size_t& nItems, size_t& nSamples, size_t& nSingle, std::string& err)
{
    size_t const nTet = cnts.size();
    if (!useItems)
    {
        for (size_t t = 0; t < nTet; ++t) { nSamples += cnts[t]; nSingle += cnts[t] == 0; }
        return true;
    }
    itemStart.resize(nTet + 1);
    for (size_t t = 0; t < nTet; ++t)
    {
        uint32_t const n = cnts[t];
        itemStart[t] = uint32_t(nItems);
        size_t items = n ? (n + itemSamples - 1) / itemSamples : 1u;
        if (exact && n)
        {
            uint64_t nw = 1;
            for (int d = 0; d < 3 && nw; ++d)
            {
                double const base = verts[t*12 + d];
                double lo = 0., hi = 0.;
                for (int v = 1; v < 4; ++v)
                {
                    double const a = double(verts[t*12 + v*3 + d]) - base;
                    lo = std::min(lo, a); hi = std::max(hi, a);
                }
                double const vlo = (base + lo) / dx[d], vhi = (base + hi) / dx[d];
                double const flo = std::floor(vlo - (1.e-3 + 1.e-5 * std::fabs(vlo)));
                double const fhi = std::floor(vhi + (1.e-3 + 1.e-5 * std::fabs(vhi)));
                // the positions are relative to the tetrahedron's base cell cb (sub-grid indices): the kernel's frame
                int64_t const cb = cells.empty() ? 0 : int64_t(cells[t*3 + d]);
                int64_t iLo = cb + (std::isfinite(flo) ? int64_t(flo) : 0), iHi = cb + (std::isfinite(fhi) ? int64_t(fhi) + 1 : 0);
                iLo = std::max<int64_t>(iLo, 0); iHi = std::min<int64_t>(iHi, int64_t(nGrid[d]));
                nw = iHi > iLo ? nw * uint64_t(iHi - iLo) : 0;
            }
            items = nw ? size_t((nw + exactCells - 1) / exactCells) : 1u;
        }
        nItems += items;
        nSamples += exact ? 0 : n;
        nSingle += n == 0;
        if (nItems > size_t(UINT32_MAX)) { err = "too many work items for one batch (> 2^32)"; return false; }
    }
    itemStart[nTet] = uint32_t(nItems);
    size_t const nBlocks = (nItems >> blockShift) + 2;
    blockTet.assign(nBlocks, uint32_t(nTet ? nTet - 1 : 0));
    for (size_t t = 0, blk = 0; t < nTet && blk < nBlocks; ++t)
        while (blk < nBlocks && (blk << blockShift) < size_t(itemStart[t + 1])) blockTet[blk++] = uint32_t(t);
    return true;
}

// The sample table as (x, y, z, 0) entries: one 16-byte load per sample
inline std::vector<float> dtfeSobol4(std::vector<float> const& sobol)
{
    std::vector<float> sobol4(sobol.size() / 3 * 4, 0.f);
    for (size_t i = 0; i < sobol.size() / 3; ++i)
        for (int d = 0; d < 3; ++d) sobol4[i * 4 + d] = sobol[i * 3 + d];
    return sobol4;
}

// The per-call path's output grids, zeroed at the requested fields' sizes (the rest EMPTY)
inline void dtfeGpuInitGrids(DTFEGpuGrids& out, size_t nCell, bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad)
{
    out.density.assign(fDen ? nCell : 0, 0.f);
    out.velocity.assign(fVel ? nCell * 3 : 0, 0.f);
    out.grad.assign(fGrad ? nCell * 9 : 0, 0.f);
    out.scalar.assign(fScal ? nCell : 0, 0.f);
    out.scalar_grad.assign(fSGrad ? nCell * 3 : 0, 0.f);
}

// The batch's timing line (DTFE_GPU_TIMING), from the host's Batch and Stats.
template <class Batch, class Stats>
inline void dtfeLogBatch(const char* backend, const Batch& b, const Stats& st, double runSec)
{
    std::fprintf(stderr, "DTFE %s timing: %s kernel, %zu tetrahedra (%zu single-cell), %zu items of <= %u %s, "
                         "%zu samples, table %u, %s sums, %d chunks, %d retries, %.2f s (%.0f M samples/s)\n",
            backend,
            b.exact ? "exact" : b.k.useItems ? "item/K2" : "per-tet",
            b.nTet, b.nSingle, b.k.useItems ? b.nItems : size_t(0), b.P.itemSamples, b.exact ? "cells" : "samples", b.nSamples, b.P.tableEntries,
            b.P.fixedSum == 2u ? "run-wide 128-bit fixed-point" : b.P.fixedSum ? "fixed-point" : "float",
            st.chunks, st.retries, runSec, runSec > 0 ? b.nSamples / runSec / 1e6 : 0.0);
}

#endif
