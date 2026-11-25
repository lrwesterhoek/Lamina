//  The run-wide fixed-point words of the GPU deposits (src/fixed_units.h): a signed 128-bit integer per grid
//  component, four 32-bit atomic limbs, little-endian at [4*at .. 4*at+3]; the host folds them into 64-bit words.
//  Prepended to both kernel sources by the Makefile, after exact_clip.metal.inc. Self-contained.
#include <metal_stdlib>
using namespace metal;

// The magnitude of rint(v * scale) as limbs a[0..3] and its sign, or false: no integer below the cap, nothing is
// added and the caller counts it. scale = 2^K, so the product is exact; capBits = float bits of the cap 2^capLog2
// (RunUnits::capBits, capLog2 <= 118). The bits are tested, so fast math cannot fold the compare away.
static inline bool shareLimbs(float v, float scale, uint capBits, thread uint a[4], thread bool& neg) {
    float const p = v * scale;
    uint const bits = as_type<uint>(p) & 0x7fffffffu;
    if (!(bits < capBits)) return false;
    neg = as_type<int>(p) < 0;
    a[0] = 0u; a[1] = 0u; a[2] = 0u; a[3] = 0u;
    if (bits < 0x5f000000u) {                            // |p| < 2^63: the integer the 64-bit grids add, in magnitude
        long const r = long(rint(p));
        ulong const q = ulong(r < 0 ? -r : r);
        a[0] = uint(q); a[1] = uint(q >> 32);
    } else {                                             // 2^63 <= |p| < cap: p is an integer, its mantissa at bit e-23
        uint const sh = (bits >> 23) - 127u - 23u;       // e - 23, at least 40 and below 95 under a cap of 2^118
        ulong const t = ulong((bits & 0x7fffffu) | 0x800000u) << (sh & 31u);
        a[sh >> 5] = uint(t); a[(sh >> 5) + 1u] = uint(t >> 32);   // sh >> 5 is 1 or 2
    }
    return true;
}

// The magnitude a[0..3] added to (neg: subtracted from) the four-limb word at 'at' of x (sign-magnitude). Each
// limb's carry (borrow) is read off the old value its atomic returns and passes to the next limb, so the limbs hold
// the exact sum modulo 2^128 whatever the order. Stops at the first limb with no carry and no higher magnitude limb.
static inline void addLimbs(device atomic_uint* x, ulong at, thread const uint a[4], bool neg) {
    uint c = 0u;
    for (uint i = 0u; i < 4u; ++i) {
        uint const d = a[i] + c;
        uint const wrapIn = (d < a[i]) ? 1u : 0u;        // a[i] = 0xffffffff and c = 1: d = 0, the carry moves on
        if (d == 0u) {
            c = wrapIn;
        } else {
            uint const old = neg ? atomic_fetch_sub_explicit(&x[at * 4ul + ulong(i)], d, memory_order_relaxed)
                                 : atomic_fetch_add_explicit(&x[at * 4ul + ulong(i)], d, memory_order_relaxed);
            c = neg ? ((old < d) ? 1u : 0u) : ((old + d < old) ? 1u : 0u);
        }
        if (c == 0u && (i == 3u || (a[i + 1u] | (i < 2u ? a[i + 2u] : 0u) | (i < 1u ? a[i + 3u] : 0u)) == 0u)) break;
    }
}

// A signed 64-bit integer r added to the four-limb word at 'at' of x in two's complement (r's sign extension into
// limbs 2 and 3), carries as in addLimbs. The word ends on the bits addLimbs leaves for the same integer, so the two
// may meet on one word. Untouched limbs: addend zero, or r >= 0 with no carry left.
static inline void addLimbs63(device atomic_uint* x, ulong at, long r) {
    if (r == 0l) return;                                 // below half a unit: nothing to add
    ulong const q = as_type<ulong>(r);
    uint const lo = uint(q), hi = uint(q >> 32);
    uint const ext = (r < 0l) ? 0xffffffffu : 0u;        // limbs 2 and 3 of r: its sign extension
    device atomic_uint* const w = x + at * 4ul;
    uint c = 0u;
    if (lo != 0u) {
        uint const old = atomic_fetch_add_explicit(&w[0], lo, memory_order_relaxed);
        c = (old + lo < old) ? 1u : 0u;
    }
    uint d = hi + c;
    c = (d < c) ? 1u : 0u;                               // hi = 0xffffffff and c = 1: d = 0, the carry moves on
    if (d != 0u) {
        uint const old = atomic_fetch_add_explicit(&w[1], d, memory_order_relaxed);
        c = (old + d < old) ? 1u : 0u;
    }
    if ((ext | c) == 0u) return;                         // r >= 0 and no carry: limbs 2 and 3 take nothing
    d = ext + c;
    c = (d < c) ? 1u : 0u;
    if (d != 0u) {
        uint const old = atomic_fetch_add_explicit(&w[2], d, memory_order_relaxed);
        c = (old + d < old) ? 1u : 0u;
    }
    d = ext + c;                                         // limb 3 adds modulo 2^32: nothing leaves the word
    if (d != 0u) atomic_fetch_add_explicit(&w[3], d, memory_order_relaxed);
}

// One share rint(v * scale) into the four-limb word at 'at' of x, or false (above the cap, counted by the caller).
// Below 2^63 units, nearly every share, it goes in as one 64-bit add (addLimbs63), above as limbs. A cap below 2^63
// units still binds: both tests must hold.
static inline bool addShare(device atomic_uint* x, ulong at, float v, float scale, uint capBits) {
    float const p = v * scale;
    uint const bits = as_type<uint>(p) & 0x7fffffffu;
    if (bits < 0x5f000000u && bits < capBits) {
        addLimbs63(x, at, long(rint(p)));
        return true;
    }
    uint a[4]; bool neg = false;
    if (!shareLimbs(v, scale, capBits, a, neg)) return false;
    addLimbs(x, at, a, neg);
    return true;
}

// n times the magnitude a[0..3] into b[0..3], limb by limb. The caps keep n * |q| below 2^94 (the PS-DTFE cap holds
// the ceil(log2 nSub^3) bits of a batched add), so nothing leaves limb 3.
static inline void mulLimbs(thread const uint a[4], uint n, thread uint b[4]) {
    uint carry = 0u;
    for (uint i = 0u; i < 4u; ++i) {
        uint const lo = a[i] * n;
        uint const hi = mulhi(a[i], n);                  // at most 2^32 - 2: hi + 1 does not wrap
        uint const s = lo + carry;
        b[i] = s;
        carry = hi + ((s < lo) ? 1u : 0u);
    }
}

// n equal shares as ONE add of n * q (the batched adds of the sampled path): integer adds are exact. A share
// shareLimbs refuses counts n times in its slot.
static inline void addLimbsN(device atomic_uint* x, ulong at, thread const uint a[4], bool neg, uint n) {
    if (n == 0u) return;
    uint b[4];
    mulLimbs(a, n, b);
    addLimbs(x, at, b, neg);
}
