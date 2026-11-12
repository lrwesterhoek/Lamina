/* THE RUN-WIDE FIXED-POINT UNITS AND WORDS (both binaries).

   A grid summed in fixed point counts in a unit 2^-K: a share s becomes the integer rint(s * 2^K) (an exact product,
   a power of two, rounded half to even), and the integers add exactly in any order. The unit decides a share's
   integer, so a unit taken per GPU batch or per PS-DTFE pass from that batch's own bound rounds the same share to
   another integer in another partition split. RunUnits holds ONE unit per accumulator for the whole run, from global
   quantities only, fixed before the planner and before any partition or thread sub-domain exists (DTFE.cpp
   runUnitsSetup, right after the mean density): every split rounds every share alike.

   THE RULE. K_g = N_g - e_g, where the accumulator's natural per-cell scale S_g lies in [2^(e_g - 1), 2^e_g), so S_g
   sits at [2^(N_g - 1), 2^N_g) units. The inputs: the grid's cell sides dx_a (its box over its cells, as the grid
   frame's dxG: user_options.h gridFrame), V_cell, dx_min and the largest cell face A = V_cell / dx_min; rho_bar (the
   run's mean density), cellMass = rho_bar V_cell, Sw = V_cell under '--ps-volume-weighted', else cellMass; and one
   serial pass over the particles the run uses (measure): vmax = the largest |velocity component|, smax = the largest
   |scalar component|, M = the total mass, <v^2>_m = sum m |v|^2 / (NO_DIM M) and their count n. The maxima are exact
   and the sums run over one array in one order, so every split gets the same bits.
       accumulator                             one share             S_g                                N_g
       standard density                        (rho/rho_bar) dV      V_cell                             48
       standard velocity (each component)      v dV                  V_cell vmax                        48
       standard per-tet gradient               dv/dx dV              V_cell vmax / dx_min               48
       standard scalar, its gradient           s dV, ds/dx dV        V_cell smax, V_cell smax / dx_min  48
       standard face pass (9 words per cell)   v_j A (a double)      2 vmax A                           61
       PS-DTFE mass, DW                        w                     cellMass                           48
       PS-DTFE MOM, DV                         wm v_j, w v_j         Sw vmax, cellMass vmax             48
       PS-DTFE M2                              w v_a v_b             cellMass <v^2>_m                   48
       PS-DTFE GRAD                            wm dv/dx              Sw vmax / dx_min                   48
       PS-DTFE MOMW                            wm                    Sw                                 48
       PS-DTFE SV, RM                          V / V_cell            1                                  48
       PS-DTFE SCAL, SG                        wm s, wm ds/dx        Sw smax, Sw smax / dx_min          48
       PS-DTFE face S, P                       v_j A, rho v_a A      vmax A                             48
       PS-DTFE face N                          A                     A                                  48
       vertex star sums                        a cell's volume       (NO_DIM+1) (M / n) / rho_bar       72
   The dispersion moment takes <v^2>_m, not vmax^2, so one fast particle cannot push the cold cells of M2 toward their
   float resolution. A scale that is not positive and finite falls back to its weight's scale, else 1 (the fixed
   merge's 'pick', ps_fixed_merge.h). K outside [-120, 120] (psMassFixedScale's range, ps_deposit_params.h): that
   accumulator has no run-wide unit (has() false) and keeps its per-batch or per-pass rule; the units line names it
   and the run warns once. The standard scalar grids under '--approx-phase-space-density' have none either: their
   scalar is f = rho g, formed after the units are set, so smax does not measure it (DTFE.cpp runUnitsSetup clears
   their ok[]).

   THE CAP. One share's integer must stay below 2^capLog2 units, capLog2 = 126 - c, c = log2 of the most shares one cell
   component can receive in one batch or pass: 32 for the standard deposits and the vertex star sums (at most 2^32
   tetrahedra in one batch, pass or vertex star), and 32 + ceil(log2 nSub^NO_DIM) for every PS-DTFE pass (nSub =
   '--subsamples'; an exact pass's leftovers fall through to the sampled path, and a batched add of n equal shares adds
   n * q at once). The standard face pass needs no count: its per-cell bound (below) holds every partial sum, so its
   cap only guards against a NaN or an infinity, at 2^118, the largest a share's device limbs take
   (fixed_words.metal.inc shareLimbs); that leaves the test shift its room (N = 61 lies 33 bits below 2^94). A share at
   or above the cap -- a NaN or an infinity too -- adds nothing and is counted, as the 2^63 guard of the 64-bit words
   does (ps_deposit_params.h psFixedShareOk). At N = 48 the cap is about 2^46.5 S_g (standard) and 2^41.5 S_g (PS-DTFE,
   nSub 3).

   WHY NOTHING WRAPS. A 128-bit word holds |x| < 2^127. Every integer added is below 2^(126 - c) and one cell component
   receives at most 2^c of them, so every partial sum -- a subset sum, in whatever order the adds land -- stays below
   2^126. The device's four limbs hold the sum modulo 2^128 exactly (fixed_words.metal.inc addLimbs), and so the sum
   itself. FxWords hold word + 2^64 carry exactly for any |value| below 2^127. The standard face pass: one tessellation
   tiles each grid plane once and every vertex velocity is a particle's, so a cell's face word lies within 2 vmax A,
   below 2^61 units -- an int64 holds it with room for a float excess of a share up to a factor just under 4, and the
   carry holds anything beyond. The vertex star sums: each volume's integer is below 2^94 and a vertex has fewer than
   2^32 tetrahedra, so a sum stays below 2^126.

   THE WORDS. On the GPU (metal/fixed_words.metal.inc): four 32-bit atomic limbs per component, 16 B. On the host
   (FxWords): an int64 per component plus an exact sparse carry, so the host grids stay at 8 B per component: the
   value of component i is w[i] + 2^64 carry_i whatever order the adds came in; the carry holds only the components
   whose adds crossed the int64 range on balance, sorted by index. A thread that owns its cells adds with
   __builtin_add_overflow (fxAddOwned); threads that share cells add atomically and read a crossing off the old value
   their own atomic returns (fxAddShared): the atomics of one word form one total order, so every crossing is
   recorded once. Each thread keeps its own crossing list; fxCarryFinish sorts and sums them. A share of 2^63 units
   or more (below the cap) goes in as its low 64 bits plus a carry entry for the rest. The device's limbs fold into
   FxWords at the copy-back (fxFoldLimbs).

   THE CONVERSION. fxToReal(X, K) = ldexp(Real(X), -K): one rounding of the 128-bit integer to nearest even (the
   compiler's __int128 conversion), then a scaling that is exact while the result is a normal float -- the fixed merge's
   'finish' (ps_fixed_merge.h). A double conversion and then a float one would round twice.

   THE SWITCH AND THE HOOK. DTFE_RUN_UNITS_LEGACY=1 (runUnitsLegacy, read once): a deposit that takes these units runs
   its per-batch or per-pass path instead, with its 64-bit words and its float or double sums, for A/B comparisons; the
   checkpoint key names the switch (runUnitsKeyLine), so a resume across it is refused. DTFE_TEST_RUN_UNIT_SHIFT=n
   (runUnitShift, read once) makes every N_g larger by n, every unit 2^n finer: -20 makes nearly every share round,
   +40 pushes cells into limbs 2 and 3 and into the host carries. A deposit on its legacy path adds n to its own unit's
   exponent likewise, so the split tests can show that the legacy path fails them. The vertex star sums take no
   shift: their shares are double volumes, which round at any unit, and their cap lies only 2^22 W_bar above a cell's
   integer, so a finer unit would only send their cells to the double sums of their fallback (triangulation.cpp
   vertexDensityStarSums) and a coarser one round small stars away.

   Plain C++17 (and the GCC / Clang __int128): no Metal or Objective-C, so every build includes it (user_options.h). */
#ifndef DTFE_FIXED_UNITS_H
#define DTFE_FIXED_UNITS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "define.h"
#include "env_knobs.h"
#include "message.h"

// DTFE_RUN_UNITS_LEGACY=1: a deposit that takes the run-wide units runs its per-batch or per-pass path instead
// (the header comment)
inline bool runUnitsLegacy()
{
    static bool const on = Env::flag( "DTFE_RUN_UNITS_LEGACY" );
    return on;
}

// DTFE_TEST_RUN_UNIT_SHIFT=n: every run-wide unit 2^n finer (the overflow and split tests); 0 unset or out of range
inline int runUnitShift()
{
    static int const n = [] { long const v = Env::integer( "DTFE_TEST_RUN_UNIT_SHIFT", 0 );
                              return v >= -200 and v <= 200 ? int( v ) : 0; }();
    return n;
}

// The checkpoint settings key's line (partition_checkpoint.h settingsKey): the rule, its N and the test shift -- what
// the environment changes without changing the command line. The units themselves follow from the input files, the
// options and the binary, which the key's other lines name.
inline std::string runUnitsKeyLine()
{
    return std::string( "units=" ) + ( runUnitsLegacy() ? "legacy" : "run-wide" ) + " N=48/61/72 shift="
           + std::to_string( runUnitShift() );
}

// The accumulators a run-wide unit is set for (the header comment's table). [RU_DENSITY .. RU_SCALAR_GRADIENT] in the
// order of the standard GPU's fxScale (dtfe_deposit_params.h); [RU_PS_FX + g] in the order of PSFxGrid
// (ps_deposit_params.h: MOM, M2, GRAD, MOMW, SV, DV, DW, SCAL, SG, RM -- keep in step).
enum RunUnitGrid
{
    RU_DENSITY = 0, RU_VELOCITY, RU_GRADIENT, RU_SCALAR, RU_SCALAR_GRADIENT,   // the standard deposits
    RU_FACE,                                                                  // the standard face pass
    RU_PS_MASS,                                                               // the PS-DTFE mass grid
    RU_PS_FX,                                                                 // + PSFxGrid g, g < RU_PS_FX_COUNT
    RU_PS_FACE_S = RU_PS_FX + 10, RU_PS_FACE_N, RU_PS_FACE_P,                 // the PS-DTFE face sums
    RU_VERTEX_VOLUME,                                                         // the vertex star sums
    RU_N
};
static int const RU_PS_FX_COUNT = 10;   // PS_FX_CPU_N (ps_deposit_params.h)

// The particles' global moments (RunUnits::measure)
struct RunMoments
{
    double totalMass = 0.;   // M = sum m
    double vmax = 0.;        // the largest |velocity component|
    double smax = 0.;        // the largest |scalar component|
    double v2m = 0.;         // <v^2>_m = sum m |v|^2 / (NO_DIM M): the mass-weighted mean square per component
    size_t n = 0;            // the particles
};

struct RunUnits
{
    bool   filled = false;         // fill() ran: K[] hold this run's units (else none is set: has() false)
    double cellVolume = 0., dxMin = 0., faceArea = 0.;   // V_cell, dx_min, A = V_cell / dx_min
    double rhoBar = 0., cellMass = 0., momentWeight = 0.;   // rho_bar, rho_bar V_cell, Sw
    RunMoments moments;
    int    shift = 0;              // DTFE_TEST_RUN_UNIT_SHIFT
    int    nSub = 1;               // PS-DTFE '--subsamples' (the PS caps' term)
    double S[RU_N] = {};           // the natural per-cell scales
    int    N[RU_N] = {};           // where the unit puts them: S_g in [2^(N_g - 1), 2^N_g) units
    int    K[RU_N] = {};           // the units 2^-K
    bool   ok[RU_N] = {};          // K in [-120, 120] (runUnitsSetup clears the scalar grids' under approx PSD)
    int    capLog2[RU_N] = {};     // one share's integer stays below 2^capLog2 units

    // One serial pass over the particles in array order: the total mass, the largest |velocity component| and
    // |scalar component|, and the mass-weighted mean square velocity. The sums run in one order, the maxima are
    // exact: a function of the array alone. (PSFixedMerge::Merge::setup takes its total mass, vmax and smax from here.)
    template<class Particles>
    static RunMoments measure(Particles &particles)
    {
        RunMoments m;
        double v2 = 0.;
        for (auto &p : particles)
        {
            double const w = double( p.weight() );
            m.totalMass += w;
            double s2 = 0.;
            for (size_t j = 0; j < noVelComp; ++j)
            {
                double const v = double( p.velocity( int(j) ) );
                m.vmax = std::max( m.vmax, std::fabs( v ) );
                s2 += v * v;
            }
            v2 += w * s2;
#ifdef SCALAR
            for (size_t j = 0; j < noScalarComp; ++j)
                m.smax = std::max( m.smax, std::fabs( double( p.scalar( int(j) ) ) ) );
#endif
            ++m.n;
        }
        m.v2m = m.totalMass > 0. ? v2 / ( double( NO_DIM ) * m.totalMass ) : 0.;
        return m;
    }

    // The table (the header comment): side = the grid's cell sides, meanDensity = rho_bar, volumeWeighted =
    // '--ps-volume-weighted', subsamples = PS-DTFE's '--subsamples' (1 in the standard binary)
    void fill(double const side[NO_DIM], double const meanDensity, bool const volumeWeighted, int const subsamples,
              RunMoments const &m)
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
        cellMass = rhoBar * cellVolume;
        momentWeight = volumeWeighted ? cellVolume : cellMass;
        shift = runUnitShift();
        nSub = std::max( subsamples, 1 );
        auto pick = [](double const s, double const fallback)
        {
            if ( s > 0. and std::isfinite( s ) ) return s;
            return ( fallback > 0. and std::isfinite( fallback ) ) ? fallback : 1.;
        };
        double const V = cellVolume, A = faceArea, sw = momentWeight, vmax = m.vmax, smax = m.smax;
        S[RU_DENSITY]                 = pick( V, 1. );
        S[RU_VELOCITY]                = pick( V * vmax, V );
        S[RU_GRADIENT]                = pick( V * vmax / dxMin, V );
        S[RU_SCALAR]                  = pick( V * smax, V );
        S[RU_SCALAR_GRADIENT]         = pick( V * smax / dxMin, V );
        S[RU_FACE]                    = pick( 2. * vmax * A, 2. * A );
        S[RU_PS_MASS]                 = pick( cellMass, 1. );
        S[RU_PS_FX + 0]               = pick( sw * vmax, sw );               // MOM
        S[RU_PS_FX + 1]               = pick( cellMass * m.v2m, cellMass );  // M2
        S[RU_PS_FX + 2]               = pick( sw * vmax / dxMin, sw );       // GRAD
        S[RU_PS_FX + 3]               = pick( sw, 1. );                      // MOMW
        S[RU_PS_FX + 4]               = 1.;                                  // SV
        S[RU_PS_FX + 5]               = pick( cellMass * vmax, cellMass );   // DV
        S[RU_PS_FX + 6]               = pick( cellMass, 1. );                // DW
        S[RU_PS_FX + 7]               = pick( sw * smax, sw );               // SCAL
        S[RU_PS_FX + 8]               = pick( sw * smax / dxMin, sw );       // SG
        S[RU_PS_FX + 9]               = 1.;                                  // RM
        S[RU_PS_FACE_S]               = pick( vmax * A, A );
        S[RU_PS_FACE_N]               = pick( A, 1. );
        S[RU_PS_FACE_P]               = pick( vmax * A, A );
        double const meanMass = m.n > 0 ? m.totalMass / double( m.n ) : 0.;   // M / n
        S[RU_VERTEX_VOLUME]           = pick( double( NO_DIM + 1 ) * meanMass / rhoBar, 1. );
        // c: 32, plus ceil(log2 nSub^NO_DIM) for every PS-DTFE accumulator; the standard face pass's per-cell bound
        // holds its sums, so its cap is the limbs' largest, 2^118 (the header comment's cap)
        double const nSubD = std::pow( double( nSub ), double( NO_DIM ) );
        int psBits = 0;
        while ( std::ldexp( 1., psBits ) < nSubD ) ++psBits;
        for (int g = 0; g < RU_N; ++g)
        {
            N[g] = g == RU_FACE ? 61 : g == RU_VERTEX_VOLUME ? 72 : 48;
            int e = 0;
            std::frexp( S[g], &e );   // S in [2^(e-1), 2^e)
            // (the vertex star sums take no test shift: the header comment's hook)
            K[g] = N[g] + ( g == RU_VERTEX_VOLUME ? 0 : shift ) - e;
            ok[g] = K[g] >= -120 and K[g] <= 120;
            bool const ps = g >= RU_PS_MASS and g <= RU_PS_FACE_P;
            capLog2[g] = g == RU_FACE ? 118 : 126 - 32 - ( ps ? psBits : 0 );
        }
    }

    // grid g has a run-wide unit (else its deposit keeps its per-batch or per-pass rule)
    bool has(int const g) const { return filled and ok[g]; }

    // Whether vertexDensity sums the vertex stars exactly (triangulation.cpp vertexDensityStarSums) in a tessellation
    // of mean density meanDensity: the standard binary's single build (PS-DTFE's V_lag and V_eul and the double build
    // keep the double sums), the unit set, and the run's own mean density -- the unit is W_bar = (NO_DIM+1) (M / n) /
    // rho_bar, so a tessellation of another mean density (the velocity-space one of '--approx-phase-space-density')
    // keeps the double sums -- and not DTFE_RUN_UNITS_LEGACY=1. The tessellation cache's key names it
    // (tessellation_cache.h makeDescriptor): the vertex densities a cache file holds differ between the two.
    bool starSums(double const meanDensity) const
    {
#ifdef PHASE_SPACE
        (void)meanDensity;
        return false;
#else
        return sizeof(Real) == 4 and has( RU_VERTEX_VOLUME ) and meanDensity == rhoBar and not runUnitsLegacy();
#endif
    }
    // its unit as the kernels' float factor 2^K, and as a double
    float  scale(int const g) const { return std::ldexp( 1.f, K[g] ); }
    double scaleD(int const g) const { return std::ldexp( 1., K[g] ); }
    // its cap: in units (the host's compare, fxShare), and as the float bits of 2^capLog2 (the kernels' shareLimbs: a
    // float product p is below the cap iff (bits(p) & 0x7fffffff) < capBits; NaN and infinity lie above it)
    double   cap(int const g) const { return std::ldexp( 1., capLog2[g] ); }
    uint32_t capBits(int const g) const { return uint32_t( 127 + capLog2[g] ) << 23; }

    // the accumulators the run log names: this binary's own and the vertex star sums
    static std::vector<int> listed()
    {
#ifdef PHASE_SPACE
        std::vector<int> l = { RU_PS_MASS };
        for (int g = 0; g < RU_PS_FX_COUNT; ++g) l.push_back( RU_PS_FX + g );
        l.push_back( RU_PS_FACE_S );
        l.push_back( RU_PS_FACE_N );
        l.push_back( RU_PS_FACE_P );
#else
        std::vector<int> l = { RU_DENSITY, RU_VELOCITY, RU_GRADIENT, RU_SCALAR, RU_SCALAR_GRADIENT, RU_FACE };
#endif
        l.push_back( RU_VERTEX_VOLUME );
        return l;
    }
    static char const *name(int const g)
    {
        static char const *const names[RU_N] = {
            "density", "velocity", "gradient", "scalar", "scalar gradient", "face", "mass",
            "MOM", "M2", "GRAD", "MOMW", "SV", "DV", "DW", "SCAL", "SG", "RM",
            "face S", "face N", "face P", "vertex volume" };
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

    // The run log's line: the inputs and every listed accumulator's K (its unit 2^-K), then the caps. Global
    // quantities only, so the line is the same at every split (the split tests compare it).
    std::string text() const
    {
        if ( not filled ) return "fixed-point units (run-wide): none";
        char b[400];
        std::snprintf( b, sizeof b, "fixed-point units (run-wide): cell volume %.6g, mean density %.6g, max |velocity "
                       "component| %.6g, max |scalar| %.6g, <v^2>_m %.6g over %zu particles; 2^-K:", cellVolume, rhoBar,
                       moments.vmax, moments.smax, moments.v2m, moments.n );
        std::string t = b;
        bool first = true;
        for (int const g : listed())
        {
            t += std::string( first ? " " : ", " ) + name( g ) + " " + std::to_string( K[g] );
            if ( not ok[g] ) t += " (out of range: its per-batch or per-pass unit)";
            first = false;
        }
#ifdef PHASE_SPACE
        std::snprintf( b, sizeof b, "; share caps 2^%d units (2^%d for the vertex volumes)", capLog2[RU_PS_MASS],
                       capLog2[RU_VERTEX_VOLUME] );
#else
        std::snprintf( b, sizeof b, "; share caps 2^%d units", capLog2[RU_DENSITY] );
#endif
        t += b;
        if ( shift != 0 ) t += ", DTFE_TEST_RUN_UNIT_SHIFT=" + std::to_string( shift );
        if ( runUnitsLegacy() ) t += ", DTFE_RUN_UNITS_LEGACY=1";
        return t;
    }
};

// One crossing of the int64 range, or the high part of a share of 2^63 units or more: (component index, carry)
typedef std::pair<uint64_t, int64_t> FxCarry;
typedef std::vector<FxCarry> FxCarryList;   // one per thread while a pass adds; sorted and summed by fxCarryFinish

// One grid's exact sums: an int64 per component, plus the exact sparse carry. The value of component i is
// int128(w[i]) + 2^64 * carry_i, whatever order the adds came in. carry holds only the components whose adds
// crossed the int64 range on balance, sorted by index. (Under '--scratch-dir' an allocation of >= 1 GB is backed by
// scratch_alloc.cc, as every grid's is.)
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

// n components of device limbs into f (the copy-back): each component's low 64 bits into its word, the rest
// (E - low, a multiple of 2^64) into the carry, which comes out sorted
inline void fxFoldLimbs(uint32_t const *limbs, size_t const n, FxWords &f)
{
    f.w.resize( n );
    f.carry.clear();
    for (size_t i = 0; i < n; ++i)
    {
        __int128 const E = fxLimbsValue( limbs + 4 * i );
        int64_t const lo = int64_t( uint64_t( E ) );
        int64_t const hi = int64_t( ( E - __int128( lo ) ) >> 64 );
        f.w[i] = lo;
        if ( hi != 0 ) f.carry.push_back( FxCarry( uint64_t( i ), hi ) );
    }
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
