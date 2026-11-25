/* The standard-DTFE GPU kernels' parameter block: ONE definition for the hosts and the Metal kernels.
   The Makefile prepends it to metal/dtfe_deposit.metal (the kernels name it DepositParams), so host and kernel
   layouts cannot drift. Plain C++ that is also valid MSL: fixed-width types, arrays, no vector types, no includes. */
#ifndef DTFE_DEPOSIT_PARAMS_H
#define DTFE_DEPOSIT_PARAMS_H

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

struct DTFEDepositParams
{
    float    dx[3];
    int32_t  nGrid[3];       // the batch's sub-grid: samples outside [0, nGrid) are not this batch's
    uint32_t nTet;           // threads in this dispatch: tetrahedra (per-tet kernel) or items (item kernel)
    int32_t  fDen;           // deposit density
    int32_t  fVel;           // deposit velocity
    int32_t  fGrad;          // deposit velocity gradient
    uint32_t itemSamples;    // item kernel: samples per work item (every tetrahedron is split so)
    uint32_t nTetTotal;      // item kernel: tetrahedra in the batch (bounds the block search, see itemOf)
    int32_t  subOff[3];      // the sub-grid's origin in the accumulated grid
    uint32_t itemBase;       // item kernel: the first item of this dispatch (items are numbered per batch)
    int32_t  fullGrid[3];    // the accumulated grid's dimensions (flat index = full-grid index)
    float    invDx[3];       // 1/dx: the cell of a sample is floor((base + p) * invDx) in both kernels
    uint32_t tableEntries;   // item kernel: entries of the threadgroup cell table (a power of two; 0 = none)
    int32_t  fScal;          // deposit the scalar, ONE component (per-tet kernel; Metal specializes the item
    int32_t  fSGrad;         // pipelines on FC_SCAL / FC_SGRAD instead)
    // fixedSum 2 (exact batches): shares go to each grid's four-limb 128-bit buffer (16..20) in units of
    // 1/fxScale[g] (fixed_units.h RunUnits), below the cap fxCapBits[g]; 1: two-word 64-bit buffer, units set
    // from the batch's own bound; 0: float atomics into buffers 7/8/9/14/15. The host binds ONE buffer per
    // grid at both indices.
    uint32_t fixedSum;
    float    fxScale[5];     // 2^k per grid: [0] density, [1] velocity, [2] velocity gradient, [3] scalar, [4] its gradient
    uint32_t fxCapBits[5];   // fixedSum 2: the float bits of each grid's share cap (fixed_words.metal.inc shareLimbs)
};
static_assert(sizeof(DTFEDepositParams) == 36 * 4,
              "DTFEDepositParams: 36 four-byte fields, no padding (host == kernel)");

#endif
