#ifndef PS_DEPOSIT_PARAMS_H
#define PS_DEPOSIT_PARAMS_H

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../env_knobs.h"

// GPU per-tet record: PS_TET_STRIDE floats per tet; the layout is documented at TET_STRIDE in metal/ps_deposit.metal.
constexpr int PS_TET_STRIDE = 16;

// Fills one record from cv (the tet's vertices in the canonical frame of the exact inside test, in the build's own
// Real) and the grid, shift and wrap parameters; vertex 0 = cell index + fraction, edges = double differences rounded
// once to float. The double build passes DOUBLE vertices: float-rounding them could flip a sub-sample vs the CPU test.
template <typename V>
inline void psFillTetRecord(float *rec, V const cv[4][3], double const gridLo[3],
                            double const gridLen[3], int32_t const nGrid[3],
                            double const canonShift[3], double const wrapL[3],
                            uint32_t const centroidFlat)
{
    for (int d = 0; d < 3; ++d)
    {
        double const a0 = (double(cv[0][d]) - (gridLo[d] + canonShift[d])) * double(nGrid[d]) / gridLen[d];
        double const ic = std::floor(a0);
        int32_t const ic0 = int32_t(ic);
        std::memcpy(&rec[d], &ic0, sizeof(ic0));
        rec[3 + d] = float(a0 - ic);
    }
    for (int e = 0; e < 3; ++e)
        for (int d = 0; d < 3; ++d)
        {
            double r = double(cv[e+1][d]) - double(cv[0][d]);   // exact for float vertices
            if (wrapL[d] > 0.)
            {
                if (r >  0.5 * wrapL[d]) r -= wrapL[d];
                else if (r < -0.5 * wrapL[d]) r += wrapL[d];
            }
            rec[6 + e*3 + d] = float(r);
        }
    std::memcpy(&rec[15], &centroidFlat, sizeof(centroidFlat));
}

// The MOMENT grids are exact 64-bit fixed-point sums like the mass grid: float sums would depend on the GPU atomics'
// order and the CPU thread count. Each share becomes rint(share * scale[g]) (scale = 2^k, exact product, both sides
// round alike: psFixedShareOk) and the integers add in any order. g indexes PSDepositParams.fxScale, PSGpuGrids::fx.
enum PSFxGrid { PS_FX_MOM = 0, PS_FX_M2, PS_FX_GRAD, PS_FX_MOMW, PS_FX_SV, PS_FX_DV, PS_FX_DW, PS_FX_SCAL, PS_FX_SG,
                PS_FX_N,                 // the grids a GPU deposit can hold as words (PSDepositParams.fxScale)
                PS_FX_RM = PS_FX_N,      // + the CPU-only one
                PS_FX_CPU_N };
// MOM sum(wm v_j) (3); M2 sum(w v_i v_j) (6); GRAD sum(wm dv_i/dx_j) (9); MOMW sum(wm), the normalizer; SV sum(V_int)/V_cell;
// DV sum(w v_j) (3); DW sum(w), not deposited on the GPU when the mass grid is fixed (the mass words serve); SCAL sum(wm s);
// SG sum(wm ds/dx_i) (3); RM sum(V_tet)/V_cell (CPU only, --ps-halo-release). w = mass share, wm = moment weight (volume share, or w).
// PSDepositParams.fxMask: bit g = grid g is fixed-point (words at buffer 22 + slot), except bit PS_FX_DW = "dispw is the mass
// grid"; PS_FX_SUMW_BIT = the exact items' per-tet weight totals are fixed-point sums too.
static uint32_t const PS_FX_SUMW_BIT = 1u << 16;

// The unit of each moment grid: scale[g] = 2^k from a bound B of the sum of |share| into grid g (PSFixedBounds::addTet);
// 2B sits at 2^61..2^62 units (psMassFixedScale), a factor 4 below 2^63 for the GPU's float rounding of a value.
// False for a bound that is not finite or gives no power of two; an empty bound gets unit 1.
inline bool psMassFixedScale(double totalMass, float &scale);   // (below)
inline bool psMomentFixedScale(double bound, float &scale)
{
    scale = 1.f;
    if ( !std::isfinite( bound ) || bound < 0. ) return false;
    if ( bound == 0. ) return true;
    return psMassFixedScale( 2. * bound, scale );
}

// The bounds B[g] of one pass, summed per tetrahedron: w = its mass, V = Eulerian volume, wm = V; vMax, gMax, sMax, sgMax =
// largest |v|, |dv/dx|, |s|, |ds/dx| there (a sample or centroid lies in the tet, so the linear profile is bounded by the
// vertex values); cellVol = the grid cell's volume.
struct PSFixedBounds
{
    double B[PS_FX_CPU_N] = {};
    void addTet(double w, double V, double wm, double vMax, double gMax, double sMax, double sgMax, double cellVol)
    {
        B[PS_FX_MOM]  += wm * vMax;
        B[PS_FX_M2]   += w * vMax * vMax;
        B[PS_FX_GRAD] += wm * gMax;
        B[PS_FX_MOMW] += wm;
        B[PS_FX_SV]   += V / cellVol;
        B[PS_FX_DV]   += w * vMax;
        B[PS_FX_DW]   += w;
        B[PS_FX_SCAL] += wm * sMax;
        B[PS_FX_SG]   += wm * sgMax;
        B[PS_FX_RM]   += V / cellVol;
    }
    // --face-gradient (CPU only; not a PSFxGrid): face sums of the streams' velocity and covered area. S = an upper bound
    // on the tet's section area over the grid planes it crosses.
    double faceS = 0., faceN = 0., faceP = 0.;
    // rho = the tet's uniform density in rho/rho_bar (its mass over its Eulerian volume): the normal momentum's bound
    void addFace(double vMax, double S, double rho) { faceS += vMax * S; faceN += S; faceP += rho * vMax * S; }
    void add(PSFixedBounds const &o)
    {
        for (int g = 0; g < PS_FX_CPU_N; ++g) B[g] += o.B[g];
        faceS += o.faceS; faceN += o.faceN; faceP += o.faceP;
    }
};

// The units a pass's caller hands the GPU host: scale[g] > 0 = grid g as a fixed-point sum in units of 1/scale[g], 0 = float
// atomics; sumW = the exact items' per-tet weight totals as fixed-point sums too. Plain, so gpu_host.h stays free of DTFE headers.
struct PSFixedUnits
{
    float scale[PS_FX_N] = {};
    bool  sumW = false;
    // the total the mass unit is taken from when > 0 (a --grid-window pass: GPU and CPU take the same unit); 0 = the host's own sum
    double massTotal = 0.;
};

// Non-finite guard of the fixed-point shares: rint(share * scale) -> int64 is UNSPECIFIED for a NaN, an infinity or a product
// beyond 2^63 (arm64 and x86-64 differ). Policy, the same on CPU and in Metal: a share whose product is not below 2^63 in
// magnitude adds NOTHING and is COUNTED per slot (reported once, fixedShareDropsReact in DTFE.cpp). The kernels test the float
// product's bits (fast math cannot fold that away), the CPU the double product, the same value; in-range shares are unaffected.
// The 128-bit run-wide words (src/fixed_units.h) use their share cap 2^capLog2 in place of 2^63. Slots: g = PSFxGrid, the mass grid, the per-tet weight totals; standard DTFE 0..4 = its grids, 5 = face pass.
enum FixedShareSlot { FX_DROP_MASS = 10, FX_DROP_SUMW = 11, FX_DROP_FACE_S = 12, FX_DROP_FACE_N = 13, FX_DROP_FACE_P = 14,
                      FX_DROP_SLOTS = 16 };
static_assert( int(PS_FX_CPU_N) <= int(FX_DROP_MASS), "a moment grid's slot would be the mass grid's" );
// The run's dropped shares by slot: every pass (CPU, or a GPU host after a deposit it kept) adds its counts here
inline std::atomic<unsigned long long> fixedShareDrops[FX_DROP_SLOTS] = {};
// A thread whose counts are not the run's yet points this at its own FX_DROP_SLOTS counters (null: the run's); the face walk
// beside the GPU deposit does: its sums are thrown away if the device call fails and the CPU then counts the same shares.
inline thread_local unsigned long long *fixedShareDropSink = nullptr;
struct FixedShareDropSinkScope   // sets the sink for a scope of the calling thread
{
    explicit FixedShareDropSinkScope(unsigned long long *const sink) : before( fixedShareDropSink )
    {
        fixedShareDropSink = sink;
    }
    ~FixedShareDropSinkScope() { fixedShareDropSink = before; }
    FixedShareDropSinkScope(FixedShareDropSinkScope const &) = delete;
    FixedShareDropSinkScope &operator=(FixedShareDropSinkScope const &) = delete;
private:
    unsigned long long *const before;
};
// Counts n dropped shares of a slot (the cold path of the guard; also the GPU hosts' per-pass totals)
inline void fixedShareDropped(int const slot, unsigned long long const n = 1)
{
    if ( fixedShareDropSink ) { fixedShareDropSink[slot] += n; return; }
    fixedShareDrops[slot].fetch_add( n, std::memory_order_relaxed );
}
// The guard: the share's integer in q, or false (q = 0, counted in 'slot'). The share times the 2^k unit is exact in float and
// double alike, so rint gives the integer the kernels add (metal/ps_deposit.metal addMoment, depositMass): a share adds the same
// units whichever side deposits it, and the grid is ONE exact integer sum independent of order and thread count.
inline bool psFixedShareOk(double const product, int const slot, int64_t &q)
{
    if ( !( std::fabs( product ) < 9223372036854775808.0 ) )   // 2^63; false for a NaN too
    {
        fixedShareDropped( slot );
        q = 0;
        return false;
    }
    q = int64_t( std::llrint( product ) );
    return true;
}
// ... for the run-wide 128-bit words: below cap = 2^capLog2 the share's integer (as src/fixed_units.h fxShare), else false
// (q = 0, counted in 'slot'). At cap = 2^63 it gives the integer of the guard above.
inline bool psFixedShareOk(double const product, double const cap, int const slot, __int128 &q)
{
    if ( !( std::fabs( product ) < cap ) )   // false for a NaN too
    {
        fixedShareDropped( slot );
        q = 0;
        return false;
    }
    q = static_cast<__int128>( std::nearbyint( product ) );
    return true;
}

// Test hook DTFE_TEST_NAN_SHARE=<k>: the deposit's input is NaN for about one tet in k (picked by a hash of its first vertex,
// so every path agrees). Units and bounds leave those out, so the rest stays exact and the guard drops and counts them. 0 = off.
inline unsigned long fixedShareTestNanEvery()
{
    static unsigned long const k = [] { long const v = Env::integer( "DTFE_TEST_NAN_SHARE", 0 );
                                        return v > 0 ? (unsigned long)v : 0ul; }();
    return k;
}
inline bool fixedShareTestNan(double const x, double const y, double const z)
{
    unsigned long const k = fixedShareTestNanEvery();
    if ( k == 0 ) return false;
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (double const c : { x, y, z })
    {
        uint64_t b;
        std::memcpy( &b, &c, sizeof b );
        h = ( h ^ b ) * 0xbf58476d1ce4e5b9ull;
        h ^= h >> 31;
    }
    return h % k == 0;
}

// The parameter block of the PS-DTFE GPU deposit (PSDepositParams, PSExactItemParams): ONE definition in ps_kernel_params.h,
// shared by the hosts and the Metal kernels, to whose source the Makefile prepends it.
#include "ps_kernel_params.h"
static_assert( int(PS_FX_N) == 9, "PSDepositParams.fxScale[9] holds one unit per PSFxGrid (ps_kernel_params.h)" );

// The unit of the mass grid's fixed-point sum: scale = 2^k with the partition's mass at 2^61..2^62 units, the finest unit
// at which no cell overflows the signed 64-bit sum. It must be fine: zoom cells hold tiny pieces of a tet. False when
// the mass gives no usable power of two (the caller keeps the float grid).
inline bool psMassFixedScale(double totalMass, float &scale)
{
    if ( !(totalMass > 0.) || !std::isfinite( totalMass ) ) return false;
    int eTotal = 0;
    std::frexp( totalMass, &eTotal );                   // total < 2^eTotal
    int const k = 62 - eTotal;
    if ( k < -120 || k > 120 ) return false;           // far outside any mass unit: float range
    scale = std::ldexp( 1.f, k );
    return true;
}

// The 64-bit integer of a cell from its two fixed-point words (low, high; two's complement).
inline int64_t psMassFixedWord(uint32_t lo, uint32_t hi)
{
    return int64_t( ( uint64_t(hi) << 32 ) | uint64_t(lo) );
}

// The mass of one cell from its fixed-point sum.
inline double psMassFixedMass(int64_t q, float scale)
{
    return double(q) / double(scale);
}

// The mass of one cell from its two fixed-point words.
inline double psMassFixedValue(uint32_t lo, uint32_t hi, float scale)
{
    return psMassFixedMass( psMassFixedWord(lo, hi), scale );
}

// The GPU hosts hand the kernels' (low, high) uint32 word pairs over as int64_t by copying the bytes
// (ps_metal_host.cc): that IS the integer on a little-endian host, which every Metal (Apple Silicon) host is.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the GPU hosts copy the fixed-point mass words as little-endian int64_t (ps_deposit_params.h)"
#endif

#endif
