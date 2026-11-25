//  Shared pieces of the PS-DTFE kernels: the non-finite guard of a fixed-point share (fixedShareOk), the 64-bit
//  words' add (addFixedQ) and the reader of the items' map (psItemOf). Prepended to ps_deposit.metal by the Makefile,
//  after ps_kernel_params.h (psItemOf reads PS_ITEM_BLOCK_SHIFT).
#include <metal_stdlib>
using namespace metal;

// Guard of a fixed-point share (ps_deposit_params.h psFixedShareOk is the host's twin): a product p = share * 2^k not
// below 2^63 in magnitude (NaN, infinity, overflow) adds NOTHING and is counted in its slot. The test reads the
// float's bits, so fast math cannot fold it away. The 128-bit limbs use the run's share cap 2^(94 - ceil(log2 nSub^3))
// units instead (shareLimbs, RunUnits::capBits), so no partial sum of at most 2^32 nSub^3 shares reaches 2^126.
static inline bool fixedShareOk(float p) { return (as_type<uint>(p) & 0x7fffffffu) < 0x5f000000u; }

// One 64-bit two's-complement integer q into the two-word cell 'at' of x: the low word's carry is read off the old
// value its atomic returns, the high word adds modulo 2^32, so the total is exact whatever the interleaving.
static inline void addFixedQ(device atomic_uint* x, ulong at, ulong q)
{
    if (q == 0ul) return;                           // below half a unit: nothing to add
    uint const lo = uint(q), hi = uint(q >> 32);
    uint const old = atomic_fetch_add_explicit(&x[at * 2ul], lo, memory_order_relaxed);
    uint const up = hi + ((old + lo < old) ? 1u : 0u);
    if (up != 0u)
        atomic_fetch_add_explicit(&x[at * 2ul + 1ul], up, memory_order_relaxed);
}

// An item's (tet, block) from the items' map (gpu_host.h psBuildExactItemMap): the tet is the last t in
// [blockTet[blk], blockTet[blk+1]] with itemStart[t] <= iid (binary search), the block is iid - itemStart[t]. The face
// kernel's map has blockCells 1: one item per (record, axis, plane).
static inline uint2 psItemOf(device const uint* itemStart, device const uint* blockTet, uint nTet, uint iid) {
    uint const blk = iid >> PS_ITEM_BLOCK_SHIFT;
    uint lo = blockTet[blk];
    uint hi = min(blockTet[blk + 1u], nTet - 1u);
    while (hi > lo) {
        uint const mid = (lo + hi + 1u) >> 1;
        if (itemStart[mid] <= iid) lo = mid; else hi = mid - 1u;
    }
    return uint2(lo, iid - itemStart[lo]);
}
