/* The standard-DTFE GPU deposit's exact sums: the units of its fixed-point grids, for the GPU host of the standard
   binary (dtfe_metal_host.cc), plus the host-side steps that need no Metal type (knobs, batch parameter block, item
   map, timing line). Why the grids are integer sums: metal/dtfe_deposit.metal's header. Host-side plain C++. */
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

// Per-batch units: grid g holds integers in units of 1/scale[g], scale = 2^k, from a bound B of sum |share| over
// everything the batch can deposit (V_tet * max|value| per tetrahedron; the interpolant is a convex combination of
// vertex values). psMassFixedScale puts B at 2^60..2^61: a factor 4 below the signed 2^63 for the kernel's float
// rounding of a sample value. The gradient bound uses the kernel's float edges with the degeneracy cut relaxed to
// half the kernel's. False (float atomics) when a bound is not finite or has no power of two; all-zero grid: unit 1.
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
// Work items (default): every tetrahedron becomes max(1, ceil(cnt / itemSamples)) threads, so none loops over more
// than itemSamples = 48 samples (up to maxNN = 10000 would trip the GPU watchdog), and the per-thread cell cache
// (2 slots, kernel depositAveraged1Items2) merges the per-sample atomics per distinct cell.
// DTFE_GPU_ITEMS=0 runs the per-tet kernel for A/B comparisons.
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
// The batch's parameter block (zeroed by the host). An EXACT batch (--exact) sums in the run-wide 128-bit words
// (P.fixedSum = 2), each grid in the run's unit 2^-units.K[g] below its share cap (fixed_units.h RunUnits: global,
// so every split rounds every share alike); a grid without one takes batchFixedScales' own unit. A sampled batch,
// or a run without units, uses 64-bit words from its own bound (1), or float atomics (0). Returns the exact kernel's
// item size (exactCells = 16). DTFE_TEST_RUN_UNIT_SHIFT=n makes the run-wide units 2^n finer.
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

// The item map (kernel's itemOf): itemStart[t] = first item of tetrahedron t, and per block of 2^blockShift items the
// tetrahedron holding the first. n samples take ceil(n / itemSamples) items, a single-cell tetrahedron one; under
// 'exact' ceil(window / exactCells) of an UPPER bound of the kernel's cell window (its float floor differs from the
// double one only within rounding of an integer, which the margin covers). Adds to nItems / nSamples / nSingle;
// false (err set) beyond 2^32 items.
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
