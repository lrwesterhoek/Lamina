/* THE RUN-WIDE FIXED-POINT UNITS AND WORDS (both binaries): one unit 2^-K per accumulator for the whole run, from global
   quantities only (DTFE.cpp runUnitsSetup, before the planner and any partition), so every split rounds every share to
   the same integer; the integers add exactly in any order. Plain C++17 (and __int128): no Metal or Objective-C. */

// THE RULE. K_g = N_g - e_g, with the accumulator's per-cell scale S_g in [2^(e_g - 1), 2^e_g), from the grid's cell
// sides, rho_bar and one serial pass over the particles (vmax, smax, total mass, count). N_g: 48 for the density, the
// velocity, the per-tet gradient and the scalar; 61 for the standard face pass; 72 for the vertex star sums. A scale
// not positive and finite falls back to its weight's scale, else 1; K outside [-120, 120] leaves the accumulator on its
// per-batch rule (has() false), as do the standard scalar grids under '--approx-phase-space-density'.

// THE CAP. One share's integer stays below 2^(126 - c) units, c = log2 of the most shares one cell component gets in
// one batch (32), so every partial sum of a 128-bit word stays below 2^126 in any order; a share at or above the cap
// adds nothing and is counted (psFixedShareOk). The face pass is bounded by 2 vmax A (< 2^61): its cap guards NaN only.

// THE WORDS. GPU: four 32-bit atomic limbs per component (fixed_words.metal.inc). Host (FxWords): an int64 per component
// plus an exact sparse carry list, value = w[i] + 2^64 carry_i in any add order (fxAddOwned, fxAddShared, fxCarryFinish).
// fxToReal rounds the 128-bit integer once, then scales exactly (a double then a float conversion would round twice).

// DTFE_TEST_RUN_UNIT_SHIFT=n makes every unit 2^n finer but the vertex star sums'; the checkpoint key names it.
#ifndef DTFE_FIXED_UNITS_H
#define DTFE_FIXED_UNITS_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "define.h"
#include "env_knobs.h"
#include "message.h"

// DTFE_TEST_RUN_UNIT_SHIFT=n: every run-wide unit 2^n finer (the overflow and split tests); 0 unset or out of range
inline int runUnitShift()
{
    static int const n = [] { long const v = Env::integer( "DTFE_TEST_RUN_UNIT_SHIFT", 0 );
                              return v >= -200 and v <= 200 ? int( v ) : 0; }();
    return n;
}

// The checkpoint settings key's line (partition_checkpoint.h settingsKey; standard DTFE's key only): the rule, its N
// and the test shift -- what the environment changes without changing the command line. The units themselves follow
// from the input files, the options and the binary, which the key's other lines name.
inline std::string runUnitsKeyLine()
{
    return std::string( "units=run-wide N=48/61/72 shift=" ) + std::to_string( runUnitShift() );
}

// The accumulators a run-wide unit is set for (the header comment's table). [RU_DENSITY .. RU_SCALAR_GRADIENT] in the
// order of the standard GPU's fxScale (dtfe_deposit_params.h).
enum RunUnitGrid
{
    RU_DENSITY = 0, RU_VELOCITY, RU_GRADIENT, RU_SCALAR, RU_SCALAR_GRADIENT,   // the standard deposits
    RU_FACE,                                                                  // the standard face pass
    RU_VERTEX_VOLUME,                                                         // the vertex star sums
    RU_N
};

// The particles' global moments (RunUnits::measure)
struct RunMoments
{
    double totalMass = 0.;   // M = sum m
    double vmax = 0.;        // the largest |velocity component|
    double smax = 0.;        // the largest |scalar component|
    size_t n = 0;            // the particles
};

struct RunUnits
{
    bool   filled = false;         // fill() ran: K[] hold this run's units (else none is set: has() false)
    double cellVolume = 0., dxMin = 0., faceArea = 0.;   // V_cell, dx_min, A = V_cell / dx_min
    double rhoBar = 0.;            // rho_bar
    RunMoments moments;
    int    shift = 0;              // DTFE_TEST_RUN_UNIT_SHIFT
    double S[RU_N] = {};           // the natural per-cell scales
    int    N[RU_N] = {};           // where the unit puts them: S_g in [2^(N_g - 1), 2^N_g) units
    int    K[RU_N] = {};           // the units 2^-K
    bool   ok[RU_N] = {};          // K in [-120, 120] (runUnitsSetup clears the scalar grids' under approx PSD)
    int    capLog2[RU_N] = {};     // one share's integer stays below 2^capLog2 units

    // One serial pass over the particles in array order: the total mass and the largest |velocity component| and
    // |scalar component|. The sum runs in one order, the maxima are exact: a function of the array alone.
    template<class Particles>
    static RunMoments measure(Particles &particles)
    {
        RunMoments m;
        for (auto &p : particles)
        {
            m.totalMass += double( p.weight() );
            for (size_t j = 0; j < noVelComp; ++j)
            {
                double const v = double( p.velocity( int(j) ) );
                m.vmax = std::max( m.vmax, std::fabs( v ) );
            }
#ifdef SCALAR
            for (size_t j = 0; j < noScalarComp; ++j)
                m.smax = std::max( m.smax, std::fabs( double( p.scalar( int(j) ) ) ) );
#endif
            ++m.n;
        }
        return m;
    }

    // The table (the header comment): side = the grid's cell sides, meanDensity = rho_bar
    void fill(double const side[NO_DIM], double const meanDensity, RunMoments const &m)
    {
        *this = RunUnits();
        filled = true;
        moments = m;
        cellVolume = 1.;
        dxMin = HUGE_VAL;
        for (int d = 0; d < NO_DIM; ++d)
        {
            cellVolume *= side[d];
            dxMin = std::min( dxMin, side[d] );
        }
        faceArea = cellVolume / dxMin;
        rhoBar = meanDensity;
        shift = runUnitShift();
        auto pick = [](double const s, double const fallback)
        {
            if ( s > 0. and std::isfinite( s ) ) return s;
            return ( fallback > 0. and std::isfinite( fallback ) ) ? fallback : 1.;
        };
        double const V = cellVolume, A = faceArea, vmax = m.vmax, smax = m.smax;
        S[RU_DENSITY]                 = pick( V, 1. );
        S[RU_VELOCITY]                = pick( V * vmax, V );
        S[RU_GRADIENT]                = pick( V * vmax / dxMin, V );
        S[RU_SCALAR]                  = pick( V * smax, V );
        S[RU_SCALAR_GRADIENT]         = pick( V * smax / dxMin, V );
        S[RU_FACE]                    = pick( 2. * vmax * A, 2. * A );
        double const meanMass = m.n > 0 ? m.totalMass / double( m.n ) : 0.;   // M / n
        S[RU_VERTEX_VOLUME]           = pick( double( NO_DIM + 1 ) * meanMass / rhoBar, 1. );
        // c: 32; the standard face pass's per-cell bound holds its sums, so its cap is the limbs' largest, 2^118 (the
        // header comment's cap)
        for (int g = 0; g < RU_N; ++g)
        {
            N[g] = g == RU_FACE ? 61 : g == RU_VERTEX_VOLUME ? 72 : 48;
            int e = 0;
            std::frexp( S[g], &e );   // S in [2^(e-1), 2^e)
            // (the vertex star sums take no test shift: the header comment's hook)
            K[g] = N[g] + ( g == RU_VERTEX_VOLUME ? 0 : shift ) - e;
            ok[g] = K[g] >= -120 and K[g] <= 120;
            capLog2[g] = g == RU_FACE ? 118 : 126 - 32;
        }
    }

    // grid g has a run-wide unit (else its deposit keeps its per-batch or per-pass rule)
    bool has(int const g) const { return filled and ok[g]; }

// Whether vertexDensity sums the vertex stars exactly (triangulation.cpp vertexDensityStarSums) in a tessellation of
// mean density meanDensity: the standard binary's single build with the unit set and the run's own mean density (unit
// W_bar = (NO_DIM+1)(M / n)/rho_bar); PS-DTFE, the double build and another mean density ('--approx-phase-space-density')
// keep the double sums. The tessellation cache's key names it (tessellation_cache.h makeDescriptor).
    bool starSums(double const meanDensity) const
    {
        return sizeof(Real) == 4 and has( RU_VERTEX_VOLUME ) and meanDensity == rhoBar;
    }
    // its unit as the kernels' float factor 2^K, and as a double
    float  scale(int const g) const { return std::ldexp( 1.f, K[g] ); }
    double scaleD(int const g) const { return std::ldexp( 1., K[g] ); }
    // its cap: in units (the host's compare, fxShare), and as the float bits of 2^capLog2 (the kernels' shareLimbs: a
    // float product p is below the cap iff (bits(p) & 0x7fffffff) < capBits; NaN and infinity lie above it)
    double   cap(int const g) const { return std::ldexp( 1., capLog2[g] ); }
    uint32_t capBits(int const g) const { return uint32_t( 127 + capLog2[g] ) << 23; }

    // the accumulators the run log names: standard DTFE's and the vertex star sums
    static std::vector<int> listed()
    {
        return { RU_DENSITY, RU_VELOCITY, RU_GRADIENT, RU_SCALAR, RU_SCALAR_GRADIENT, RU_FACE, RU_VERTEX_VOLUME };
    }
    static char const *name(int const g)
    {
        static char const *const names[RU_N] = {
            "density", "velocity", "gradient", "scalar", "scalar gradient", "face", "vertex volume" };
        return g >= 0 and g < RU_N ? names[g] : "?";
    }

    // the listed accumulators whose K falls outside [-120, 120], comma-separated (empty: none)
    std::string outOfRange() const
    {
        std::string t;
        if ( not filled ) return t;
        for (int const g : listed())
            if ( not ok[g] )
                t += std::string( t.empty() ? "" : ", " ) + name( g ) + " (K " + std::to_string( K[g] ) + ")";
        return t;
    }

    // The run log's line: the inputs and every listed accumulator's K (its unit 2^-K), then the caps and the test
    // shift. Global quantities only, so the line is the same at every split (the split tests compare it).
    std::string text() const
    {
        if ( not filled ) return "fixed-point units (run-wide): none";
        char b[400];
        std::snprintf( b, sizeof b, "fixed-point units (run-wide): cell volume %.6g, mean density %.6g, max |velocity "
                       "component| %.6g, max |scalar| %.6g over %zu particles; 2^-K:", cellVolume, rhoBar,
                       moments.vmax, moments.smax, moments.n );
        std::string t = b;
        bool first = true;
        for (int const g : listed())
        {
            t += std::string( first ? " " : ", " ) + name( g ) + " " + std::to_string( K[g] );
            if ( not ok[g] ) t += " (out of range: its per-batch or per-pass unit)";
            first = false;
        }
        std::snprintf( b, sizeof b, "; share caps 2^%d units", capLog2[RU_DENSITY] );
        t += b;
        if ( shift != 0 ) t += ", DTFE_TEST_RUN_UNIT_SHIFT=" + std::to_string( shift );
        return t;
    }
};

// One crossing of the int64 range, or the high part of a share of 2^63 units or more: (component index, carry)
typedef std::pair<uint64_t, int64_t> FxCarry;
typedef std::vector<FxCarry> FxCarryList;   // one per thread while a pass adds; sorted and summed by fxCarryFinish

// One grid's exact sums: an int64 per component, plus the exact sparse carry. The value of component i is
// int128(w[i]) + 2^64 * carry_i, whatever order the adds came in. carry holds only the components whose adds
// crossed the int64 range on balance, sorted by index.
struct FxWords
{
    std::vector<int64_t> w;
    FxCarryList carry;   // (component index, carry), sorted, non-zero only
    int K = 0;           // the unit 2^-K
};

// The share's integer (the host's twin of the kernels' shareLimbs): product = the share times 2^K, exact (a power of
// two), rounded to nearest even -- the integer llrint gives below 2^63 -- or false (q = 0) when it has none below the
// cap (RunUnits::cap): a NaN, an infinity, |product| >= cap. The caller counts a false in its slot (ps_deposit_params.h
// fixedShareDropped). A double holding an integer below 2^127 converts to __int128 exactly.
inline bool fxShare(double const product, double const cap, __int128 &q)
{
    if ( !( std::fabs( product ) < cap ) )
    {
        q = 0;
        return false;
    }
    q = static_cast<__int128>( std::nearbyint( product ) );
    return true;
}

// q (below the cap) into the word of component 'key' that this thread owns: its low 64 bits into the word with
// __builtin_add_overflow (which stores the wrapped sum), a crossing of the int64 range and the rest of q
// (q - low, a multiple of 2^64) into this thread's list
inline void fxAddOwned(int64_t &w, __int128 const q, uint64_t const key, FxCarryList &list)
{
    int64_t const lo = int64_t( uint64_t( q ) );
    int64_t const hi = int64_t( ( q - __int128( lo ) ) >> 64 );
    if ( hi != 0 ) list.push_back( FxCarry( key, hi ) );
    int64_t r;
    if ( __builtin_add_overflow( w, lo, &r ) ) list.push_back( FxCarry( key, lo > 0 ? 1 : -1 ) );
    w = r;
}

// The 64-bit route through fxShare + fxAddOwned, which nearly every share takes: a product below fxFast64(cap) =
// min(cap, 2^63) units has its integer in an int64 (llrint rounds as nearbyint does) and no part beyond the low word, so
// fxAddOwned64 of llrint(product) adds what fxShare + fxAddOwned add: the same word and carry entry. Any other product
// takes fxShare + fxAddOwned. (On arm64 the 128-bit route costs a library call per share.)
inline double fxFast64(double const cap) { return std::min( cap, 0x1p63 ); }
inline void fxAddOwned64(int64_t &w, int64_t const q, uint64_t const key, FxCarryList &list)
{
    int64_t r;
    if ( __builtin_add_overflow( w, q, &r ) ) list.push_back( FxCarry( key, q > 0 ? 1 : -1 ) );
    w = r;
}

// ... into a word other threads add to as well: an atomic add, whose returned old value tells the crossing (the word
// after the add holds old + low modulo 2^64, formed in unsigned arithmetic; it crossed iff it moved the other way);
// the atomics of one word form one total order, so every crossing is recorded once, whatever the interleaving
inline void fxAddShared(int64_t *p, __int128 const q, uint64_t const key, FxCarryList &list)
{
    int64_t const lo = int64_t( uint64_t( q ) );
    int64_t const hi = int64_t( ( q - __int128( lo ) ) >> 64 );
    if ( hi != 0 ) list.push_back( FxCarry( key, hi ) );
    if ( lo == 0 ) return;
    int64_t const old = __atomic_fetch_add( p, lo, __ATOMIC_RELAXED );
    int64_t const after = int64_t( uint64_t( old ) + uint64_t( lo ) );
    if ( lo > 0 ? after < old : after > old ) list.push_back( FxCarry( key, lo > 0 ? 1 : -1 ) );
}

// After a pass: the threads' lists and the carry the words hold already into one carry, sorted by component and
// summed (in __int128, so the order of the entries cannot matter), zeros dropped; the lists are freed. A component's
// carry beyond the int64 range (a cell of 2^127 units, far beyond every cap) stops the run, naming 'what'.
inline void fxCarryFinish(FxWords &f, std::vector<FxCarryList> &lists, char const *what)
{
    size_t n = f.carry.size();
    for (FxCarryList const &l : lists) n += l.size();
    if ( n == 0 ) return;
    FxCarryList all;
    all.reserve( n );
    all.insert( all.end(), f.carry.begin(), f.carry.end() );
    for (FxCarryList &l : lists)
    {
        all.insert( all.end(), l.begin(), l.end() );
        FxCarryList().swap( l );
    }
    std::sort( all.begin(), all.end(), [](FxCarry const &a, FxCarry const &b) { return a.first < b.first; } );
    FxCarryList out;
    for (size_t i = 0; i < all.size(); )
    {
        uint64_t const key = all[i].first;
        __int128 s = 0;
        for ( ; i < all.size() and all[i].first == key; ++i ) s += all[i].second;
        if ( s == 0 ) continue;
        if ( s > __int128( INT64_MAX ) or s < __int128( INT64_MIN ) )
            throwError( "the fixed-point sums of ", what, ": a cell's carry left the int64 range (the cell would hold "
                        "2^127 units or more). The run stops: a wrapped cell could not be repaired." );
        out.push_back( FxCarry( key, int64_t( s ) ) );
    }
    f.carry.swap( out );
}

// One component's four 32-bit device limbs (little-endian, fixed_words.metal.inc) as its 128-bit integer
inline __int128 fxLimbsValue(uint32_t const *l)
{
    unsigned __int128 const u = (unsigned __int128)l[0] | ( (unsigned __int128)l[1] << 32 )
                              | ( (unsigned __int128)l[2] << 64 ) | ( (unsigned __int128)l[3] << 96 );
    return (__int128)u;   // two's complement
}

// Component i's exact integer: its word plus 2^64 times its carry
inline __int128 fxValue(FxWords const &f, size_t const i)
{
    __int128 v = __int128( f.w[i] );
    auto const it = std::lower_bound( f.carry.begin(), f.carry.end(), uint64_t( i ),
                                      [](FxCarry const &c, uint64_t const k) { return c.first < k; } );
    if ( it != f.carry.end() and it->first == uint64_t( i ) ) v += __int128( it->second ) * ( __int128( 1 ) << 64 );
    return v;
}

// THE conversion (the header comment): one rounding of the integer, then an exact scaling by 2^-K. T = Real, or
// float where the grids are handed over as floats in every build (the standard GPU deposit's DTFEGpuGrids,
// dtfe_metal_host.cc): a double rounding and then a float one would round twice in the double build.
template<class T = Real>
inline T fxToReal(__int128 const X, int const K)
{
    return std::ldexp( static_cast<T>( X ), -K );
}

// A grid's n components into out: a dense pass over the words, then each carry entry overwrites its component
inline void fxToRealGrid(FxWords const &f, Real *out, size_t const n)
{
    size_t const m = std::min( n, f.w.size() );
    for (size_t i = 0; i < m; ++i) out[i] = fxToReal( __int128( f.w[i] ), f.K );
    for (FxCarry const &c : f.carry)
        if ( c.first < m )
            out[c.first] = fxToReal( __int128( f.w[c.first] ) + __int128( c.second ) * ( __int128( 1 ) << 64 ), f.K );
}

#endif
