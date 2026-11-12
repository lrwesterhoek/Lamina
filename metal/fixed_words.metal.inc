//  The run-wide fixed-point words of the GPU deposits (src/fixed_units.h says why, and why nothing wraps): a signed
//  128-bit integer per grid component, held as four 32-bit atomic limbs, little-endian at [4*at .. 4*at+3] (16 B per
//  component; the host folds them into its 64-bit words with an exact carry, fixed_units.h fxFoldLimbs). Only 32-bit
//  atomics: each limb's carry (or borrow) is read off the old value its own atomic returns and goes to the next limb as
//  one more add (subtract), so once every thread is done the four limbs hold the exact sum modulo 2^128, whatever the
//  interleaving -- and the host's caps keep every partial sum below 2^126 in magnitude (RunUnits::capLog2). The limbs
//  stay separate words, as the two-word adds of the 64-bit grids do (dtfe_deposit.metal addWordsLH).
//  The Makefile prepends this file to both kernel sources, after exact_clip.metal.inc. Self-contained.
#include <metal_stdlib>
using namespace metal;

// The magnitude of rint(v * scale) as four limbs a[0..3] and its sign, or false: the share has no integer below the
// cap and adds nothing (the caller counts it in its slot, as the 64-bit grids' guard does). scale = 2^K, so the float
// product is exact; capBits = the float bits of the cap 2^capLog2 (RunUnits::capBits; capLog2 at most 118): the
// product is below the cap iff its bits without the sign are below capBits, and a NaN or an infinity lies above. The
// float's bits are tested, as the 64-bit guard's are (fast math cannot fold an integer compare away).
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

// The magnitude a[0..3] added to (neg: subtracted from) the four-limb word at 'at' of x (sign-magnitude, so a small
// negative share touches one or two limbs, not four). Each limb's carry (borrow) is read off the old value its own
// atomic returns and passes to the next limb: whatever order the atomics land in, limb i ends at its start plus its
// deltas modulo 2^32 and passes on exactly the carries those deltas made, so the four limbs hold the exact
// two's-complement sum modulo 2^128. It stops at the first limb with no carry and no higher magnitude limb.
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

// n times the magnitude a[0..3] into b[0..3], limb by limb: the low 32 bits of a[i] * n, plus the carry from the limb
// below, and the high 32 bits (mulhi) carried up. The caps keep n * |q| below 2^94 (the PS-DTFE cap holds the
// ceil(log2 nSub^3) bits a batched add of n <= nSub^3 equal shares takes), so nothing leaves limb 3.
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

// n equal shares as ONE add of n * q (the batched adds of the sampled path, addMomentN / depositMassN): integer adds
// are exact, so the word ends where n adds of q leave it. A share shareLimbs refuses counts n times in its slot.
static inline void addLimbsN(device atomic_uint* x, ulong at, thread const uint a[4], bool neg, uint n) {
    if (n == 0u) return;
    uint b[4];
    mulLimbs(a, n, b);
    addLimbs(x, at, b, neg);
}
