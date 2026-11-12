/* The PS-DTFE GPU kernels' parameter blocks: ONE definition for the hosts and the Metal kernels.
   The hosts (ps_metal_host.cc, metal/validate_deposit.cpp) include it through ps_deposit_params.h
   (PSDepositParams); the Makefile PREPENDS this file to metal/ps_deposit.metal, after exact_clip.metal.inc, when it
   embeds the kernel source (metal/validate_deposit.cpp and metal/build_prototype.sh do the same), and the kernels
   name the blocks DepositParams and ExactItemParams. So the struct the kernels read cannot drift from the one the
   hosts fill. The file is plain C++ that is also valid MSL: fixed-width scalar types (MSL has int32_t and
   uint32_t), arrays and no vector types (a float3 would be padded to 16 bytes), no includes under MSL. Its
   static_asserts run in both compilers and pin the layout the field offsets below imply.
   Each field's buffer numbers are those of metal/ps_deposit.metal. */
#ifndef PS_KERNEL_PARAMS_H
#define PS_KERNEL_PARAMS_H

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

// depositFields' and depositExactItems' parameters (Metal buffer 8)
struct PSDepositParams
{
    float    boxLo[3];
    float    dx[3];
    int32_t  nGrid[3];
    int32_t  nSub;
    int32_t  periodic;     // 0 / 1
    // Partition sub-grid (production PS-DTFE stores only the Eulerian bbox each Lagrangian
    // partition touches): output arrays are subDims-sized; a wrapped cell w maps to
    // loc = (w - subOrigin + nGrid) % nGrid -- the box may wrap a periodic axis -- and cells
    // outside [0,subDims) are skipped BEFORE sample counting, exactly like the inSub guard in
    // ps_interpolation.cc. Full grid: subOrigin=0, subDims=nGrid.
    int32_t  subOrigin[3];
    int32_t  subDims[3];
    // Field flags: which moment grids this run actually needs. When a flag is 0 the host binds
    // a 4-byte dummy buffer for that grid and the kernel never touches it (mass and streams are
    // always deposited). fVel = velocity or dispersion selected (mom), fDisp = dispersion (m2,
    // 24 B per cell), fGrad = velocity gradient (grad, 36 B per cell).
    int32_t  fVel;
    int32_t  fDisp;
    int32_t  fGrad;
    // --ps-linear-deposit: weight the interior samples by the DTFE-interpolated linear density
    // (from the per-tet vertex densities in buffer 9), renormalized per tet so the deposited
    // total still equals the tet mass exactly. 0 = uniform equal shares (buffer 9 is a dummy).
    int32_t  fLinear;
    // --ps-volume-weighted: the velocity/dispersion/gradient moments carry equal EULERIAN-VOLUME
    // shares |det(Ax)|/6 / N instead of the mass shares (the mass grid always keeps mass), and
    // their normalizer is deposited into the momw grid (buffer 10; dummy when 0). Mutually
    // exclusive with fLinear (rejected at option parsing).
    int32_t  fVolW;
    // --ps-caustics: atomic-OR the tet's orientation bits (1 = det(Ax)>0, 2 = det<0) into the
    // caustic grid (buffer 11; dummy when 0) at every deposited sample. OR is commutative and
    // idempotent, so the result is chunk-, thread- and partition-order invariant.
    int32_t  fCaustic;
    // --exact: replace the nSub^3 sub-sampled deposit by analytic tet-cell clipping
    // (sweep_moments in exact_clip.metal.inc, see exact_cell_clip in metal/ps_deposit.metal). Ignores nSub; a tet
    // whose whole clip window comes up empty (sliver below float resolution) falls through to
    // the sampled path, whose empty window then drives the centroid fallback -- the same
    // fall-through the CPU exact deposit uses.
    int32_t  fExact;
    uint32_t nTet;
    // The scalar field (builds with ONE scalar component, NO_SCALARS=1; the callers decline the GPU
    // otherwise): the per-tet vertex scalars arrive in buffer 18 (nTet*4; a dummy when both flags are
    // 0) and are deposited exactly like the velocity moment -- the linear profile's value at the
    // sample (or at the piece centroid) times the moment weight -- into buffer 19 (nCell), and the
    // tet's constant scalar gradient times the moment weight into buffer 20 (nCell*3, [flat*3+i] =
    // ds/dx_i, the CPU's scalar_gradient layout for one component).
    int32_t  fScal;
    int32_t  fSGrad;
    // --grid-window: the sub-box (subOrigin/subDims) is a WINDOW of the full grid, not a partition's own
    // box: a tet straddling its edge counts every in-grid sample (pass 0 / phase 0 sumW) exactly as in
    // the full run and deposits only the samples / pieces inside, so the window equals the full run.
    int32_t  windowMode;
    // The mass grid's accumulator. massFixed = 1: every mass share is added as the 64-bit
    // integer rint(share * massScale) to massFx (buffer 21: two uint words per cell, low then high, the
    // carry done by hand) -- exact, and independent of the order the atomics land in. Float atomics into
    // 'mass' (buffer 3) drop every share below half an ulp of a busy cell's running sum, always
    // downward: 0.06% of the box mass at TNG50-3-Dark z=0, percents in halo-core cells with 1e7 deposits.
    // massFixed = 0 keeps that float grid (DTFE_GPU_FLOAT_MASS=1, for comparison); the unused one is a
    // 4-byte dummy. (The moment grids: fxMask below.)
    int32_t  massFixed;
    float    massScale;    // 2^k (ps_deposit_params.h psMassFixedScale), so the product is exact
    // The MOMENT grids (ps_deposit_params.h PSFxGrid) as exact 64-bit fixed-point sums: rint(value * fxScale[g]) adds
    // to the same words in any atomic order, as the mass does. fxMask bit g = grid g is fixed-point: its words (two
    // uint per component, low then high) sit in the buffer of its float index (8 B per component, as in
    // metal/dtfe_deposit.metal), in units of 1/fxScale[g], a power of two the host bounds so no cell overflows.
    // Bit FX_DW = dispw is not deposited (it equals the mass grid's shares; the host reads the mass words).
    // FX_SUMW_BIT = the exact items' per-tet weight totals are fixed-point too (depositExactItems).
    // 0 = every moment grid a float sum (DTFE_GPU_FLOAT_MOMENTS=1 on the host).
    uint32_t fxMask;
    float    fxScale[9];   // [g], g = PS_FX_MOM ... PS_FX_SG: the PS_FX_N grids (PS_FX_DW unused)
    // PS_GPU_DEBUG_NOATOMIC (a timing experiment, the atomics' share of the sampled kernel): 1 = the sampled path
    // classifies and counts every sample as always but deposits none -- no grid atomics, the grids stay zero
    uint32_t debugNoAtomic;
};
static_assert(sizeof(PSDepositParams) == 41 * 4, "PSDepositParams: 41 four-byte fields, no padding (host == kernel)");

// depositExactItems' per-dispatch parameters (Metal buffer 17, set per command buffer): the items of this chunk,
// the pass (== the pipeline's kItemsPhase, 0 or 1; the kernel reads it only in the kItemsPhase-2 pipeline, which runs
// both passes of --ps-linear-deposit), the cells per item, and badAt = the word of the sumW buffer where the
// non-finite guard's counters start, past the per-tet totals (Metal: every one of the 31 buffer slots is taken, so
// they ride in that buffer's tail).
// itemBase = the chunk's first item and nTet = the tets of the items' map (gpu_host.h psBuildExactItemMap): the
// chunk's thread k runs item itemBase + k, which the kernel decodes from the map (psItemOf)
struct PSExactItemParams
{
    uint32_t nItems;
    uint32_t phase;
    uint32_t blockCells;
    uint32_t badAt;
    uint32_t itemBase;
    uint32_t nTet;
};
static_assert(sizeof(PSExactItemParams) == 24, "PSExactItemParams: six 32-bit words (host == kernel)");

// The exact items' map (gpu_host.h psBuildExactItemMap; the kernels' psItemOf): one block-table entry per
// 2^PS_ITEM_BLOCK_SHIFT items, as the standard-DTFE kernel's ITEM_BLOCK_SHIFT
#define PS_ITEM_BLOCK_SHIFT 5

#endif
