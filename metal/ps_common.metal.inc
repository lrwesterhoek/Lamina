//  The PS-DTFE kernels' shared pieces: the non-finite guard of a fixed-point share (fixedShareOk), the 64-bit words'
//  add (addFixedQ) and the reader of an items' map (psItemOf). Prepended by the Makefile after ps_kernel_params.h
//  (psItemOf reads PS_ITEM_BLOCK_SHIFT) to metal/ps_deposit.metal and to metal/ps_face.metal, so both kernels add and
//  decode alike (metal/validate_deposit.cpp and metal/build_prototype.sh do the same). Moved here unchanged from
//  ps_deposit.metal for the face kernel (docs/cleanup-plan.md P11, the GPU kernel for the face sweep, step 3).
#include <metal_stdlib>
using namespace metal;

// The non-finite guard of a fixed-point share (ps_deposit_params.h psFixedShareOk is the host's twin and
// says why): a product p = share * 2^k that is not below 2^63 in magnitude -- a NaN, an infinity, an overflow -- has
// no int64 (the float->long conversion is unspecified there) and adds NOTHING; it is counted in its slot instead.
// The test reads the float's bits (|p| < 2^63 <=> (bits & 0x7fffffff) < 0x5f000000, NaN and inf above it), so the
// default fast math cannot fold it away. One compare; a finite in-range share adds its integer unchanged.
// The 64-bit words take it (and the items' sumW words in both modes); the 128-bit limbs take the run's share cap
// 2^(94 - ceil(log2 nSub^3)) units in its place (shareLimbs: RunUnits::capBits, the same bit test), so every partial
// sum of a component's at most 2^32 nSub^3 shares stays below 2^126 and the four limbs never wrap.
static inline bool fixedShareOk(float p) { return (as_type<uint>(p) & 0x7fffffffu) < 0x5f000000u; }

// One 64-bit two's-complement integer q into the two-word cell 'at' of x: the low word's carry is read off the
// value its atomic returns (the add wrapped iff the result is below it), and the high words add modulo 2^32, so
// once every thread is done the two words hold the exact integer total, whatever the interleaving.
static inline void addFixedQ(device atomic_uint* x, ulong at, ulong q)
{
    if (q == 0ul) return;                           // below half a unit: nothing to add
    uint const lo = uint(q), hi = uint(q >> 32);
    uint const old = atomic_fetch_add_explicit(&x[at * 2ul], lo, memory_order_relaxed);
    uint const up = hi + ((old + lo < old) ? 1u : 0u);
    if (up != 0u)
        atomic_fetch_add_explicit(&x[at * 2ul + 1ul], up, memory_order_relaxed);
}

// An item's (tet, block) from the items' map (gpu_host.h psBuildExactItemMap: itemStart[nTet+1], then blockTet), the
// standard-DTFE kernel's itemOf: the item's tet is the last t in [blockTet[blk], blockTet[blk+1]] with
// itemStart[t] <= iid -- a binary search over at most one block's worth of tets (2^PS_ITEM_BLOCK_SHIFT + 1 entries,
// more only across tets with no items, which the "last t" skips) -- and its block is iid - itemStart[t]. (The face
// kernel's map: one item per (record, axis, plane), blockCells 1, so the "block" is the item's number in its record.)
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
