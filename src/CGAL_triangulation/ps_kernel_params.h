/* The PS-DTFE GPU kernels' parameter blocks: ONE definition for the hosts and the Metal kernels (the Makefile prepends this
   file to metal/ps_deposit.metal). Plain C++ that is also valid MSL: fixed-width scalars, arrays, no vector types (float3
   pads to 16 bytes), no includes under MSL; its static_asserts pin the layout. Buffer numbers are those of ps_deposit.metal. */
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
    // Partition sub-grid: output arrays are subDims-sized; a wrapped cell w maps to loc = (w - subOrigin + nGrid) % nGrid and
    // cells outside [0,subDims) are skipped BEFORE sample counting, like the inSub guard in ps_interpolation.cc.
    int32_t  subOrigin[3];
    int32_t  subDims[3];
    // Field flags: which moment grids are needed; a 0 flag binds a 4-byte dummy buffer the kernel never touches (mass and
    // streams are always deposited). fVel = velocity or dispersion (mom), fDisp = dispersion (m2), fGrad = gradient (grad).
    int32_t  fVel;
    int32_t  fDisp;
    int32_t  fGrad;
    // --ps-linear-deposit: weight samples by the DTFE-interpolated linear density (buffer 9), renormalized per tet so the
    // deposited total still equals the tet mass exactly. 0 = uniform shares.
    int32_t  fLinear;
    // --ps-volume-weighted: the moments carry equal EULERIAN-VOLUME shares instead of the mass shares (the mass grid keeps
    // mass); the normalizer goes into the momw grid (buffer 10). Exclusive with fLinear.
    int32_t  fVolW;
    // --ps-caustics: atomic-OR the tet's orientation bits (1 = det>0, 2 = det<0) into the caustic grid (buffer 11); OR is
    // commutative and idempotent, so the result is chunk-, thread- and partition-order invariant.
    int32_t  fCaustic;
    // --exact: analytic tet-cell clipping instead of the nSub^3 samples. A tet whose clip window comes up empty (a sliver
    // below float resolution) falls through to the sampled path and its centroid fallback, as in the CPU exact deposit.
    int32_t  fExact;
    uint32_t nTet;
    // The scalar field (ONE scalar component, NO_SCALARS=1; callers decline the GPU otherwise): per-tet vertex scalars in
    // buffer 18, deposited like the velocity moment into buffer 19, the tet's scalar gradient into buffer 20 ([flat*3+i]).
    int32_t  fScal;
    int32_t  fSGrad;
    // --grid-window: the sub-box is a WINDOW of the full grid: a tet straddling its edge counts every in-grid sample
    // exactly as in the full run and deposits only those inside, so the window equals the full run.
    int32_t  windowMode;
    // The mass grid's accumulator. massFixed = 1: every share is added as the integer rint(share * massScale) to massFx
    // (buffer 21: two uint words per cell, low then high, carry by hand; four limbs under fxWords 2): exact and independent
    // of the order the atomics land in. Float atomics drop every share below half an ulp of a busy cell's running sum,
    // always downward. massFixed = 0 keeps that float grid (DTFE_GPU_FLOAT_MASS=1); the unused one is a 4-byte dummy.
    int32_t  massFixed;
    float    massScale;    // 2^k (the run's unit, or ps_deposit_params.h psMassFixedScale), so the product is exact
    // The MOMENT grids (PSFxGrid) as exact fixed-point sums: rint(value * fxScale[g]) adds to the same words in any atomic
    // order. fxMask bit g = grid g is fixed-point: words (two uint per component, low then high; four limbs under fxWords 2)
    // in units of 1/fxScale[g], a power of two. Bit FX_DW = dispw is not deposited (it equals the mass shares);
    // FX_SUMW_BIT = the exact items' per-tet weight totals are fixed-point too. 0 = every moment grid a float sum
    // (DTFE_GPU_FLOAT_MOMENTS=1).
    uint32_t fxMask;
    float    fxScale[9];   // [g], g = PS_FX_MOM ... PS_FX_SG: the PS_FX_N grids (PS_FX_DW unused)
    // Always 0: no switch sets it (gpu_host.h psGpuParams). Kept, and still tested by the sampled path, so the kernel
    // code stays the same.
    uint32_t debugNoAtomic;
    // The words of the mass grid (buffer 21) and of every fxMask grid (buffers 22..29). 0 = 64-bit words, two uint per
    // component. 2 = the run-wide 128-bit words (DTFE_PS_RUN_UNITS=1): four uint limbs, little-endian
    // (metal/fixed_words.metal.inc), in the run's units (src/fixed_units.h RunUnits). The sumW words (FX_SUMW_BIT) stay two
    // uint per tet in both. The kernels take the size at compile time (PS_FX_WORDS: one library per size).
    uint32_t fxWords;
    // fxWords 2: the float bits of the share cap 2^capLog2, one for every PS-DTFE grid (RunUnits::capBits): a share
    // at or above it adds nothing and is counted (shareLimbs); 0 otherwise
    uint32_t fxCapBits;
};
static_assert(sizeof(PSDepositParams) == 43 * 4, "PSDepositParams: 43 four-byte fields, no padding (host == kernel)");

// depositExactItems' per-dispatch parameters (Metal buffer 17): the chunk's items, the pass (the kernel reads it only in the
// kItemsPhase-2 pipeline), the cells per item, badAt = the sumW buffer word where the non-finite guard's counters start
// (every one of the 31 buffer slots is taken, so they ride in that buffer's tail); itemBase = the chunk's first item and
// nTet = the tets of the items' map (psBuildExactItemMap): thread k runs item itemBase + k (decoded by psItemOf).
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

// The exact items' map (psBuildExactItemMap; the kernels' psItemOf): one block-table entry per 2^PS_ITEM_BLOCK_SHIFT items
#define PS_ITEM_BLOCK_SHIFT 5

#endif
