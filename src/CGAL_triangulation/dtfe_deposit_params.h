/* The standard-DTFE GPU kernels' parameter block: ONE definition for the hosts and the Metal kernels.
   dtfe_metal_host.cc uses it as is and dtfe_gpu_cuda.cu derives its own DepositParams from it (one more field,
   the guard counters' pointer, which Metal binds as a buffer of its own); the Makefile PREPENDS this file to
   metal/dtfe_deposit.metal, after exact_clip.metal.inc, when it embeds the kernel source, and the kernels name the
   block DepositParams. So the struct the kernels read cannot drift from the one the hosts fill
   (dtfe_gpu_fixed.h dtfeBatchParams). The file is plain C++ that is also valid MSL: fixed-width scalar types (MSL
   has int32_t and uint32_t), arrays and no vector types (a float3 would be padded to 16 bytes), no includes under
   MSL. Its static_assert runs in both compilers and pins the layout.

   A batch = one sub-domain's tetrahedra. Positions are relative to the SUB-DOMAIN's lower corner and nGrid is its
   sub-grid (so the cell classification is the CPU loop's, bit for bit); a sample inside the sub-grid is owned by
   this batch and lands at full-grid cell pos + subOff. With subOff = 0 and fullGrid = nGrid this is the plain
   single-grid deposit. Buffer numbers are those of metal/dtfe_deposit.metal. */
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
    uint32_t debugNoAtomic;  // DTFE_GPU_DEBUG_NOATOMIC, a timing experiment: compute everything, issue fewer or no
                             // atomics (the grids come out wrong; dtfe_gpu_fixed.h dtfeBatchParams has the levels)
    float    invDx[3];       // 1/dx: the cell of a sample is floor((base + p) * invDx) in both kernels
    uint32_t tableEntries;   // item kernel: entries of the threadgroup cell table (a power of two; 0 = none)
    int32_t  fScal;          // deposit the scalar, ONE component (per-tet kernel; Metal specializes the item
    int32_t  fSGrad;         // pipelines on FC_SCAL / FC_SGRAD instead, CUDA/HIP reads these in both kernels)
    // the grids as exact 64-bit fixed-point sums (metal/dtfe_deposit.metal's header). fixedSum = 1: every grid's
    // shares go to its two-word integer buffer (16..20) in units of 1/fxScale[g]; 0: the float atomics into
    // buffers 7/8/9/14/15. The host binds ONE buffer per grid at both indices (sized for the mode in use).
    uint32_t fixedSum;
    float    fxScale[5];     // 2^k per grid: [0] density, [1] velocity, [2] velocity gradient, [3] scalar, [4] its gradient
};
static_assert(sizeof(DTFEDepositParams) == 32 * 4, "DTFEDepositParams: 32 four-byte fields, no padding (host == kernel)");

#endif
