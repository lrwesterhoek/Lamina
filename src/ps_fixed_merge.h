/* THE FIXED-POINT MERGE of the PS-DTFE partition loop ('--ps-merge fixed-point', DTFE.cpp; OPT-IN).

   The default merge (the staged ordered merge, DTFE.cpp) adds each partition's sub-grid -- Real values, each an exact
   fixed-point sum of the deposit rounded to Real once, so a partition's values do not depend on the run -- into float
   full grids in partition INDEX order: float adds do not commute, so only one order gives the same bits, and a
   partition that finishes early must park (in RAM, or spilled to disk) until its turn. This mode adds every partition
   AS IT FINISHES into exact INTEGER full-grid accumulators instead; integer adds commute, so the result cannot depend
   on the completion order, the concurrency or the thread count, and nothing parks. It is opt-in, for comparing its
   memory and speed against the default merge.

   THE ACCUMULATORS. Every accumulator addFromSubgrid SUMS (forEachSummed below: its '+=' list; caustic_bits and
   hidden_streams are OR-merged, order-free already, and stay Real grids, as do the T-web / V-web labels, which no
   partition fills) becomes, per component and cell, an int64 WORD plus an int8 CARRY in a global unit 2^-K of that
   accumulator: 9 B where the float took 4. A partition value v becomes q = llrint(v * 2^K) -- v * 2^K is exact (a
   power of two), llrint rounds half to even, so q is a property of the partition alone -- and is added to the word
   with __builtin_add_overflow; on a signed overflow the carry takes the sign of q. A cell is then carry * 2^64 + word,
   an exact 72-bit integer whatever the order of the adds. The integer arrays replace the pre-sized Real grids when
   the merge is set up, before the first partition starts (so the memory gate's free-memory reading holds them, as
   the default's holds its Real grids), except those of the grids no partition fills (derived
   after the merge); an accumulator the default grows at its first values (the moment weights) gets its arrays at a
   partition's first value of it (the units are fixed before, so the allocation order changes nothing); the Real full
   grids do not exist during the loop (the pre-sized ones are freed when the merge is set up). Allocations of >= 1 GB
   go through scratch_alloc.cc like the float grids, so '--scratch-dir' backs these too.

   THE CONVERSION, once, after the loop, accumulator by accumulator (its Real grid allocated, its integer arrays freed
   right after): the 72-bit integer is converted to the nearest float ONCE (round half to even: the compiler's
   __int128 -> float conversion), then scaled by 2^-K (ldexp: exact while the result is a normal float). Deterministic
   per cell. The result differs from the ordered merge's float sums at about 1e-7 relative: each partition value is
   rounded to the unit (below), and the float sums of the default round at every add.

   THE UNITS are fixed BEFORE the first merge from GLOBAL quantities only -- never from the partitions that finish
   first, which would make them order-dependent. Per accumulator a natural per-cell scale S (cellMass = the particles'
   total mass / the grid's cells, cellVol = the grid cell's volume, Sw = the moment weight's scale: cellVol under
   '--ps-volume-weighted', else cellMass; vmax = the particles' largest |velocity component|, smax = their largest
   |scalar component|, dx = the smallest cell side; one O(N) pass over the particles):
       density                                  1             (the sub-grid holds rho/rho_bar already)
       mass_weight                              Sw            (the mass shares, or the volume shares)
       disp_weight                              cellMass
       velocity, velocity_std                   Sw   * vmax
       disp_velocity                            cellMass * vmax
       velocity_dispersion                      cellMass * vmax^2   (the second moment is always mass-weighted)
       velocity_gradient, _divergence, _shear,  Sw   * vmax / dx
         _vorticity, velocity_vweb_eigenvalues
       scalar                                   Sw   * smax
       scalar_gradient                          Sw   * smax / dx
       stream_count, tet_touch                  1             (counts: sample-mean multiplicity, tet touches)
       velocity_tweb_eigenvalues                1
   (the derived fields and the eigenvalues are in addFromSubgrid's list, but no partition fills them: they are made
   after the merge.) A scale that is not positive and finite (vmax = 0: no velocities) falls back to its weight's, or
   1. K is chosen so that S maps to 2^36..2^37 units (S in [2^(e-1), 2^e): K = 37 - e): a value >= ~1e-4 S keeps its
   float's 24 bits, a single value may reach 2^26 S (2^63 units), and a sum ~2^34 S thanks to the carry.
   DTFE_TEST_FXMERGE_SHIFT=n makes every unit 2^n finer (the test suite's overflow hook; a negative n coarser).

   OVERFLOW, order-free and FATAL (a wrapped cell cannot be repaired): (a) a value that is not finite or has
   |v * 2^K| >= 2^63 stops the run naming the accumulator, the partition and the value's ratio to S -- a property of
   one partition, so it fires whatever the order; (b) per accumulator the sum over the merged partitions of each one's
   largest |q| is kept (an unsigned __int128); it bounds every cell's |sum|, and it must stay below 2^71 - 2^63 (the
   int8 carry with a signed word holds [-2^71 - 2^63, 2^71 - 2^63)). It only grows, so checking it at every merge
   fails a run exactly when its total over all partitions reaches the limit, whatever the order; the message names
   the accumulator and the bound, not the partition at which it crossed (that one is order-dependent). Never the
   running sum's own overflow (order-dependent).
   Whether a run stops is a property of its partitions; which partition (a) names when several hold such values is
   the first to finish among them. DTFE_TEST_FXMERGE_SUM_LIMIT=n lowers the limit of (b) to 2^n (its test).

   CHECKPOINT ('--resume', partition_checkpoint.h): the words and carries go into the checkpoint as raw blobs
   ('x.fxw.<side>.<name>', 'x.fxc.<side>.<name>') with the per-accumulator state ('x.fxmerge.state': K, the overflow
   bound (b), the largest |q| and the cells), and the settings key carries 'merge=fixed-point', so a checkpoint of the
   other merge mode is refused with a clear message. A resumed run's integers are the killed run's plus the rest,
   so its outputs are byte-identical to an uninterrupted run (tests/ps_fixed_merge_check.sh). The units are compared
   on resume (a changed DTFE_TEST_FXMERGE_SHIFT, or another binary, is refused).

   THE WORD MERGE (fix 3 S6): all of the above is the merge of the passes' ROUNDED Real grids, which is what the
   option does in the double and the 2-D builds, and under DTFE_RUN_UNITS_LEGACY=1. In the single 3-D build the option
   switches the run-wide units on (src/fixed_units.h runUnitsPsFixedMerge) and the merge adds WORDS instead
   (User_options::psWordMerge, set by DTFE.cpp runUnitsSetup):
   - every pass deposits in the run-wide units (RunUnits: one unit per accumulator for the whole run) and hands its
     exact sums over unconverted (ps_interpolation.cc psHandOverWords: quantities.h PSWordGrids) -- per accumulator
     its int64 words, their sparse carry and K, and its int stream counts;
   - addWords adds them into the full grid's: owned adds (__builtin_add_overflow, each crossing of the int64 range
     recorded), the partition's carries added at their full-grid components, every carry summed by fxCarryFinish
     (which stops the run, naming the accumulator, should a cell's carry ever leave the int64 range: a cell of 2^127
     units). Integer adds commute: nothing is ordered, parked or spilled, and the merged words are the exact sum of
     every share whatever the split, the order and the concurrency. The masks still OR into the Real full grids;
   - the merge's unit IS the pass's: the run's units, no table of its own (no measure, no shift, no headroom checks
     (a) and (b): a pass's word is a sum below 2^126 by the share cap, the merged one is exact by the carry);
   - finish hands the merged words to the full Quantities (psWords), and ONE close-out converts them, each component
     rounded once, and normalizes: DTFE.cpp psCloseOutMoments (Quantities::psWordsClose, then normalizePhaseSpace),
     which a one-domain run under the option runs on its own pass's words too;
   - the face sums of '--face-gradient' (fix 3 S7) are held ONCE PER FACE in the full grid's words: a cell's NO_DIM
     upper faces, its lower ones being the upper faces of the cells below, and on a grid that is not periodic one
     boundary plane per axis for the lower faces of its layer 0 (quantities.h PSWordGrids, faceOnce). A pass keeps its
     own layout (both faces per cell); the merge adds every upper face, skips a lower face whose twin -- the same
     share, written by psDepositFaces to the cell below in the same pass -- it adds already, and routes a lower face
     without one to the cell below's upper face or the boundary plane (addFaceSlot), so every share counts once
     whatever the split. A one-domain run's face sums go through the same rule (oneDomainFacesOnce). The close-out
     keeps them as words and the gradient reads them so (Quantities::psFaceGradientCellWords), with no Real face
     grid; DTFE_DEBUG_FACE_TWINS=1 checks every twin pair word for word and prints one line per side.
   Memory: 8 B per summed component and full-grid cell (the int stream counts 4 B; the face sums NO_DIM x (noVelComp
   + 2) components per cell, plus (cells / n_a) x (noVelComp + 2) per non-periodic axis a for the boundary planes),
   plus 16 B per carried component; the words of the accumulators whose Real grids the loop pre-sized are allocated
   at setup, as the integer grids above, the others at a partition's first words. The checkpoint holds the words
   ('x.fxw.<side>.<accumulator>', a boundary plane 'x.fxw.<side>.<face sum>.plane<a>'), the carries ('x.fxk...', 16 B
   entries), the counts ('x.fxn.<side>') and their K and sizes ('x.fxwords.state'); restore compares the K with this
   run's and the sizes with the once layout's. */
#ifndef PS_FIXED_MERGE_H
#define PS_FIXED_MERGE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "define.h"
#include "message.h"
#include "env_knobs.h"
#include "fixed_units.h"   // RunUnits::measure: the particles' total mass, vmax and smax
#include "particle_data.h"
#include "quantities.h"
#include "user_options.h"
#include "partition_checkpoint.h"   // the checkpoint's blob list (Checkpointer::blobs)
#include "scratch_alloc.h"   // zeroAssign: the integer grids' zero fills under '--scratch-dir'
#include "thread_count.h"    // poolRanges: the word merge's adds over the free cores

namespace PSFixedMerge
{

// the scale kinds (the table in the header comment): each summed grid's tag in the field list (quantities.h)
enum Kind { K_ONE = Quantities::SCALE_ONE, K_WEIGHT = Quantities::SCALE_WEIGHT, K_MASS = Quantities::SCALE_MASS,
            K_VEL_W = Quantities::SCALE_VEL_W, K_VEL_M = Quantities::SCALE_VEL_M, K_VEL2_M = Quantities::SCALE_VEL2_M,
            K_GRAD_W = Quantities::SCALE_GRAD_W, K_SCAL_W = Quantities::SCALE_SCAL_W, K_SGRAD_W = Quantities::SCALE_SGRAD_W };

// every accumulator addFromSubgrid SUMS ('+='), f(info, grid): the field list's FIELD_SUM grids (quantities.h
// Quantities::forEachField), in its order, with their names, scale kinds (info.scale) and whether a partition fills
// them (info.madeAfterMerge) -- the one list, so a new summed grid is merged here too
template<class F>
inline void forEachSummed(Quantities &q, F &&f)
{
    Quantities::forEachField( [&](Quantities::FieldInfo const &fi, auto merge, auto member)
    {
        if constexpr ( decltype( merge )::value == Quantities::FIELD_SUM )
            f( fi, q.*member );
    } );
}

// the sum bound (b): an int8 carry over a signed int64 word holds [-2^71 - 2^63, 2^71 - 2^63)
static unsigned __int128 const SUM_LIMIT = ( (unsigned __int128)1 << 71 ) - ( (unsigned __int128)1 << 63 );
static double const VALUE_LIMIT = 9223372036854775808.;   // 2^63: one value's |q| must stay below it

// one summed accumulator of u or a
struct Slot
{
    std::string name;              // "u.density"
    int kind = K_ONE;
    size_t comps = 1;              // Real components per cell
    double S = 1.;                 // its natural per-cell scale
    int K = 0;                     // its unit is 2^-K
    size_t presized = 0;           // cells of the Real grid before the loop (freed at setup; 0: it had none)
    size_t cells = 0;              // cells of the integer arrays (0: not allocated yet)
    std::vector<int64_t> w;        // the words, comps per cell
    std::vector<int8_t> c;         // the carries
    unsigned __int128 bound = 0;   // (b): the sum over the merged partitions of each one's largest |q|
    uint64_t maxQ = 0;             // the largest |q| of one value merged
};

// the saved state of one slot in the checkpoint ('x.fxmerge.state')
struct SavedSlot
{
    int64_t K = 0;
    uint64_t boundLo = 0, boundHi = 0, maxQ = 0, cells = 0;
};
// the word merge's saved state of one accumulator of one side, or of its counts ('x.fxwords.state'): K, and the
// words (counts) the checkpoint holds
struct SavedWords
{
    int64_t K = 0;
    uint64_t n = 0;
};

inline double log2u128(unsigned __int128 v) { return v == 0 ? -1.e300 : std::log2( double( v ) ); }

/* THE FACE SUMS ONCE PER FACE (fix 3 S7; the header comment's last section). A pass's face words hold both faces of
   every axis per cell (quantities.h PSWordGrids: side 1 the upper face, side 0 the lower), and psDepositFaces
   (ps_interpolation.cc) writes each share of a grid plane to the upper face of the cell below it AND the lower face
   of the cell above it, wherever its local() rule finds that cell in the pass's sub-grid. So a lower face whose cell
   below is in the pass too -- its TWIN -- holds the very words of that cell's upper face, and the merged grid keeps
   only the upper faces: addFaceSlot adds every upper face at its destination cell, skips every lower face with a
   twin, and routes a lower face without one (the sub-grid's lowest layer of an axis that does not wrap all round in
   the pass) to the upper face of the destination's cell below, wrapped on a periodic grid, or for the grid's layer 0
   of an axis that does not wrap to that axis' boundary plane. Every share is then added once, whatever the split. */

// psDepositFaces' local() (ps_interpolation.cc -- a faithful copy, KEEP THE TWO IN STEP): the local index of raw grid
// index 'raw' along an axis of n cells (wrapped when periodic), or -1 outside the sub-box [origin, origin + dims)
inline long psFaceLocal(long const raw, long const n, bool const periodic, long const origin, long const dims)
{
    long const wg = periodic ? ( ( raw % n ) + n ) % n : raw;
    if ( !periodic && ( wg < 0 || wg >= n ) ) return -1;
    long loc = wg - origin;
    if ( loc < 0 ) loc += n;
    return ( loc < 0 || loc >= dims ) ? -1 : loc;
}

// One pass's sub-grid as the face rule sees it: its cells per axis and row-major strides, and per axis a and local
// layer c whether that layer's lower faces have their twin in the pass (twin[a][c]), and the twin's layer
// (below[a][c])
struct FaceGeometry
{
    size_t m[NO_DIM] = {}, stride[NO_DIM] = {};
    std::vector<char> twin[NO_DIM];
    std::vector<size_t> below[NO_DIM];
};

// ... from the pass's sub-box (dims m, origin o) in the grid of n cells per axis (the pass's own: the full grid's).
// Layer c's lower faces lie on the plane whose local() is c -- raw c + o, or on an axis that does not wrap c + o - n,
// local()'s '+= n' --, and their twin is the cell local() finds below that plane (raw - 1). A layer local() never
// reaches (only when the sub-box is wider than the grid) holds no face sums; its lower faces count as twinless, so
// even then every word is added somewhere.
inline FaceGeometry faceGeometry(size_t const *m, size_t const *o, size_t const *n, bool const periodic)
{
    FaceGeometry G;
    size_t s = 1;
    for (int d = NO_DIM - 1; d >= 0; --d)
    {
        G.m[d] = m[d];
        G.stride[d] = s;
        s *= m[d];
    }
    for (int a = 0; a < NO_DIM; ++a)
    {
        long const na = long( n[a] ), oa = long( o[a] ), ma = long( m[a] );
        G.twin[a].assign( m[a], char( 0 ) );
        G.below[a].assign( m[a], size_t( 0 ) );
        for (size_t c = 0; c < m[a]; ++c)
        {
            long raw = long( c ) + oa;
            if ( not periodic and raw >= na ) raw -= na;
            if ( psFaceLocal( raw, na, periodic, oa, ma ) != long( c ) ) continue;
            long const lb = psFaceLocal( raw - 1, na, periodic, oa, ma );
            if ( lb < 0 ) continue;
            G.twin[a][c] = char( 1 );
            G.below[a][c] = size_t( lb );
        }
    }
    return G;
}

// DTFE_DEBUG_FACE_TWINS=1 (a developer check, read once per merge or one-domain close-out): per side, over the passes
// merged, the twin pairs compared word for word (their 128-bit values), how many differ (0 expected), the twinless
// lower words routed and how many of them are non-zero, and of those routed the ones that went to a boundary plane
// and how many of these are non-zero (a non-periodic grid's boundary faces: non-zero where streams cross them; the
// non-zero ones routed to a cell below, nonZero - planeNonZero, are shares the merge of before missed in one cell)
struct FaceTwinStats
{
    unsigned long long pairs = 0, mismatches = 0, routed = 0, nonZero = 0, toPlane = 0, planeNonZero = 0;
    int passes = 0;
};

// the summary line (fixed words: tests grep it), on stderr as the other DTFE_DEBUG_ checks
inline void faceTwinsLine(FaceTwinStats const &t, int const side)
{
    std::fprintf( stderr, "DTFE_DEBUG_FACE_TWINS: face twins (side %s): %llu pairs, %llu mismatches; %llu words "
                  "routed, %llu non-zero; %llu to a boundary plane, %llu non-zero (%d pass(es))\n", side ? "a" : "u",
                  t.pairs, t.mismatches, t.routed, t.nonZero, t.toPlane, t.planeNonZero, t.passes );
}

// the once layout of a full grid's words (the merged accumulator, or a one-domain run's): the face sums once per
// face on the grid of '--grid', periodic or not as the run (the passes' grid-index wrap: User_options::periodic)
inline void onceLayout(PSWordGrids &W, User_options const &opt)
{
    W.faceOnce = true;
    W.periodic = opt.periodic;
    for (int d = 0; d < NO_DIM; ++d) W.dims[d] = opt.gridSize[d];
}

struct Merge
{
    bool on = false;
    std::vector<Slot> slots;              // u's, then a's, in forEachSummed order
    double cellMass = 0., cellVol = 0., vmax = 0., smax = 0., dxMin = 0.;
    int shift = 0;                        // DTFE_TEST_FXMERGE_SHIFT
    unsigned __int128 sumLimit = SUM_LIMIT;   // (b); DTFE_TEST_FXMERGE_SUM_LIMIT=n lowers it to 2^n (the test of (b))
    int sumLimitLog2 = -1;                    // n, or -1
    double peakBytes = 0., liveBytes = 0.;
    long zeroFillMoves0 = 0;              // ScratchAlloc::scratchZeroFillMoves() when setup began (scratchText)
    std::vector<SavedSlot> loadedState;   // filled by the checkpoint load (restore() checks it)
    // the word merge (the header comment's last section; User_options::psWordMerge): the full grid's words of each
    // side (0 u, 1 a), the run's units (the passes'), and the checkpoint's loaded state (restoreWords checks it)
    bool words = false;
    PSWordGrids full[2];
    RunUnits units;
    std::vector<SavedWords> loadedWords;
    size_t carried = 0;                   // the components carried beyond int64 when the loop ended (reportText)
    bool debugTwins = false;              // DTFE_DEBUG_FACE_TWINS=1 (setupWords): the face rule's check ...
    FaceTwinStats twins[2];               // ... and its counts per side (printed by finishWords)

    // the slot of the i-th summed accumulator of side 0 (u) / 1 (a)
    size_t perSide() const { return slots.size() / 2; }

    // the units, from the global quantities; replaces the pre-sized Real grids of the summed accumulators by their
    // integer arrays (freed only, for the grids no partition fills). The word merge: setupWords.
    void setup(Quantities &u, Quantities &a, User_options const &opt, std::vector<Particle_data> &particles)
    {
        if ( opt.psWordMerge )
        {
            setupWords( u, a, opt );
            return;
        }
        on = true;
        // the one O(N) pass of the run-wide units (fixed_units.h RunUnits::measure), over this array in its order
        RunMoments const moments = RunUnits::measure( particles );
        double const totalMass = moments.totalMass;
        vmax = moments.vmax;
        smax = moments.smax;
        double fullCells = 1.;
        cellVol = 1.;
        dxMin = HUGE_VAL;
        for (int d = 0; d < NO_DIM; ++d)
        {
            double const dx = double( opt.region.coords[2*d+1] - opt.region.coords[2*d] ) / double( opt.gridSize[d] );
            cellVol *= dx;
            dxMin = std::min( dxMin, dx );
            fullCells *= double( opt.gridSize[d] );
        }
        cellMass = totalMass / fullCells;
        shift = int( Env::integer( "DTFE_TEST_FXMERGE_SHIFT", shift ) );
        long const limitLog2 = Env::integer( "DTFE_TEST_FXMERGE_SUM_LIMIT", 0 );
        if ( limitLog2 > 0 and limitLog2 < 71 )
        {
            sumLimitLog2 = int( limitLog2 );
            sumLimit = (unsigned __int128)1 << sumLimitLog2;
        }
        double const sw = opt.psVolumeWeighted ? cellVol : cellMass;
        auto scaleOf = [&](int const kind) -> double
        {
            auto pick = [](double const s, double const fallback)
            {
                if ( s > 0. and std::isfinite( s ) ) return s;
                return ( fallback > 0. and std::isfinite( fallback ) ) ? fallback : 1.;
            };
            switch ( kind )
            {
            case K_WEIGHT:  return pick( sw, 1. );
            case K_MASS:    return pick( cellMass, 1. );
            case K_VEL_W:   return pick( sw * vmax, sw );
            case K_VEL_M:   return pick( cellMass * vmax, cellMass );
            case K_VEL2_M:  return pick( cellMass * vmax * vmax, cellMass );
            case K_GRAD_W:  return pick( sw * vmax / dxMin, sw );
            case K_SCAL_W:  return pick( sw * smax, sw );
            case K_SGRAD_W: return pick( sw * smax / dxMin, sw );
            default:        return 1.;
            }
        };
        slots.clear();
        zeroFillMoves0 = ScratchAlloc::scratchZeroFillMoves();
        for (int side = 0; side < 2; ++side)
            forEachSummed( side ? a : u, [&](Quantities::FieldInfo const &fi, auto &v)
            {
                using V = typename std::decay<decltype( v )>::type;
                using T = typename V::value_type;
                Slot s;
                s.name = std::string( side ? "a." : "u." ) + fi.name;
                s.kind = fi.scale;
                s.comps = sizeof( T ) / sizeof( Real );
                s.S = scaleOf( fi.scale );
                int e = 0;
                std::frexp( s.S, &e );   // S in [2^(e-1), 2^e)
                s.K = 37 - e + shift;    // S * 2^K in [2^36, 2^37)
                s.presized = v.size();
                V().swap( v );           // the Real grid: none during the loop
                // its integer twin NOW, before any partition starts: allocated at the first merge instead, the
                // memory gate would admit the first partitions with a free-memory reading that does not hold the
                // 9 B grids yet (the default's Real grids are pre-sized before the loop, so its gate sees them) --
                // a hand-set --max-concurrent could then overcommit. Not for the grids no partition
                // fills (derived after the merge: info.madeAfterMerge); those not pre-sized (the moment weights)
                // still come with their first values, as the default's do. Zero fills through
                // ScratchAlloc::zeroAssign: under '--scratch-dir' these are scratch mappings, 9 B per component, and
                // their zeros would otherwise be written back to the disk before any work (scratch_alloc.h)
                if ( s.presized > 0 and not fi.madeAfterMerge )
                {
                    ScratchAlloc::zeroAssign( s.w, s.presized * s.comps, int64_t( 0 ) );
                    ScratchAlloc::zeroAssign( s.c, s.presized * s.comps, int8_t( 0 ) );
                    s.cells = s.presized;
                }
                slots.push_back( std::move( s ) );
            } );
        account();
    }

    // The word merge's accumulator that fills a pre-sized summed Real grid (its words are allocated at setup): the
    // mass for the density, MOM, M2, GRAD, SCAL, SG for their grids, SV for the exact deposit's stream count; -1 for
    // the others (the counts below, the grids made after the merge, and those no pass fills)
    static int wordSlotOf(char const *name, bool const exact)
    {
        std::string const n = name;
        if ( n == "density" ) return RU_PS_MASS;
        if ( n == "velocity" ) return RU_PS_FX + 0;
        if ( n == "velocity_dispersion" ) return RU_PS_FX + 1;
        if ( n == "velocity_gradient" ) return RU_PS_FX + 2;
        if ( n == "scalar" ) return RU_PS_FX + 7;
        if ( n == "scalar_gradient" ) return RU_PS_FX + 8;
        if ( n == "stream_count" and exact ) return RU_PS_FX + 4;
        return -1;
    }
    // ... and whether the int counts fill it: the sampled stream count, the exact deposit's tet touches
    static bool countsFill(char const *name, bool const exact)
    {
        std::string const n = name;
        return ( n == "stream_count" and not exact ) or n == "tet_touch";
    }

    // THE WORD MERGE's setup (the header comment's last section): the units are the run's (the passes'), the
    // pre-sized summed Real grids are freed (none during the loop: the close-out makes them; their cells kept, so
    // that one no word fills comes back as zeros), and the words of the accumulators that fill them are allocated
    // now, as the integer grids above (the memory gate's reading holds them); the others come with a partition's
    // first words.
    // Each side's close-out settings follow from the options, as each pass forms them: the deposit, the sampled
    // count's divisor (the '_u' pass samples once, the '_a' pass nSub^NO_DIM times) and whether DW is the mass grid's
    // sums. The full grid's words take the once layout of the face sums (onceLayout: fix 3 S7).
    void setupWords(Quantities &u, Quantities &a, User_options const &opt)
    {
        on = true;
        words = true;
        units = opt.runUnits;
        debugTwins = Env::flag( "DTFE_DEBUG_FACE_TWINS" );
        twins[0] = twins[1] = FaceTwinStats();
        slots.clear();
        size_t cells = 1;
        for (int d = 0; d < NO_DIM; ++d) cells *= opt.gridSize[d];
        size_t samples = 1;
        for (int d = 0; d < NO_DIM; ++d) samples *= size_t( std::max( opt.psAvgSubsamples, 1 ) );
        zeroFillMoves0 = ScratchAlloc::scratchZeroFillMoves();
        for (int side = 0; side < 2; ++side)
        {
            Quantities &q = side ? a : u;
            Field const &field = side ? opt.aField : opt.uField;
            PSWordGrids &W = full[side];
            W.release();
            W.on = true;
            W.exact = opt.psExactDeposit;
            W.samplesPerCell = side ? samples : 1;
            W.cells = cells;
            W.dwIsMass = opt.psVolumeWeighted and field.velocity_dispersion and field.density;
            onceLayout( W, opt );
            forEachSummed( q, [&](Quantities::FieldInfo const &fi, auto &v)
            {
                using V = typename std::decay<decltype( v )>::type;
                size_t const pre = v.size();
                W.presized.push_back( pre );
                V().swap( v );   // the Real grid: none during the loop
                if ( pre == 0 or fi.madeAfterMerge ) return;
                if ( pre != cells )
                    throwError( "internal: the word merge's pre-sized grid ", fi.name, " holds "
                                + std::to_string( pre ) + " cells, the grid " + std::to_string( cells ) + "." );
                int const slot = wordSlotOf( fi.name, W.exact );
                if ( slot >= 0 and W.at( slot ).w.empty() )
                {   // (zeroAssign: as at the setup above)
                    ScratchAlloc::zeroAssign( W.at( slot ).w, cells * W.cellComps( slot ), int64_t( 0 ) );
                    W.at( slot ).K = units.K[slot];
                }
                if ( countsFill( fi.name, W.exact ) and W.count.empty() )
                    ScratchAlloc::zeroAssign( W.count, cells, int( 0 ) );
            } );
        }
        account();
    }

    // A pass's cell map into the full grid (subgridTarget, as addOne maps; never a window: addWordsSide): dst[l] =
    // the full-grid cell of the pass's cell l; oneToOne = no two of its cells share one (every axis of the sub-grid
    // within the full grid's); m / o = its sub-grid's cells and origin per axis (the full grid's, origin 0, when
    // 'sub' spans it)
    static void cellMap(PSWordGrids const &S, Quantities const &sub, size_t const *fullGrid, int const partition,
                        std::vector<size_t> &dst, bool &oneToOne, size_t (&m)[NO_DIM], size_t (&o)[NO_DIM])
    {
        bool const isSub = sub.ps_subDims[0] != 0;   // else 'sub' spans the destination, cell for cell
        size_t const subTotal = S.cells;
        dst.assign( subTotal, size_t( 0 ) );
        oneToOne = true;
        size_t cells = 1;
        for (int d = 0; d < NO_DIM; ++d)
        {
            m[d] = isSub ? sub.ps_subDims[d] : fullGrid[d];
            o[d] = isSub ? sub.ps_subOrigin[d] : 0;
            cells *= m[d];
            oneToOne = oneToOne and m[d] <= fullGrid[d];
        }
        if ( cells != subTotal )
            throwError( "internal: the word merge: partition ", partition, "'s words cover ", subTotal,
                        isSub ? " cells, its sub-grid " + std::to_string( cells ) + "."
                              : " cells, the full grid " + std::to_string( cells ) + "." );
        if ( not isSub )
        {
            for (size_t l = 0; l < subTotal; ++l) dst[l] = l;
            return;
        }
        for (size_t l = 0; l < subTotal; ++l)
        {
            size_t rem = l, cc[NO_DIM], g = 0;
            for (int d = NO_DIM - 1; d >= 0; --d) { cc[d] = rem % m[d]; rem /= m[d]; }
            subgridTarget( cc, o, fullGrid, nullptr, nullptr, g );   // (no window: always true)
            dst[l] = g;
        }
    }

    // The full grid's words of accumulator 'slot' in unit 2^-K: allocated zero at its first words (zeroAssign: as at
    // setup), cells x cellComps (the once layout of a face sum), with a face sum's boundary planes where the grid is
    // not periodic; else checked against the unit and the sizes
    FxWords &wordsFor(PSWordGrids &W, int const slot, int const K, std::string const &name)
    {
        FxWords &to = W.at( slot );
        bool const face = PSWordGrids::isFace( slot ), planes = face and not W.periodic;
        size_t const per = face ? PSWordGrids::facePer( slot ) : 0;
        if ( to.w.empty() )
        {
            ScratchAlloc::zeroAssign( to.w, W.cells * W.cellComps( slot ), int64_t( 0 ) );
            to.K = K;
            if ( planes )
                for (int a = 0; a < NO_DIM; ++a)
                {
                    FxWords &B = W.boundary( slot, a );
                    ScratchAlloc::zeroAssign( B.w, W.planeCells( a ) * per, int64_t( 0 ) );
                    B.K = K;
                }
            account();
            return to;
        }
        bool ok = to.K == K and to.w.size() == W.cells * W.cellComps( slot );
        if ( face )
            for (int a = 0; a < NO_DIM; ++a)
            {
                FxWords const &B = W.boundary( slot, a );
                ok = ok and B.w.size() == ( planes ? W.planeCells( a ) * per : size_t( 0 ) )
                     and ( B.w.empty() or B.K == K );
            }
        if ( not ok )
            throwError( "internal: the word merge's full-grid words of ", name, " are not the partition's unit "
                        "or size." );
        return to;
    }

    // THE FACE RULE (FaceGeometry above; fix 3 S7): one face sum of one pass (src, the pass's layout: cell l, side,
    // axis a, component j at l * 2*NO_DIM*per + (side*NO_DIM + a)*per + j) into the full grid's once layout
    // (W.at(slot): cell g, axis a, j at g * NO_DIM*per + a*per + j; W.boundary(slot, a): plane cell p, j at
    // p*per + j).
    //   1. every UPPER face at its destination cell dst[l]: over the free cores when oneToOne (each destination cell
    //      then takes one pass cell, as the other accumulators), each range's crossings in its own list;
    //   2. a LOWER face with its twin in the pass is skipped (the twin's upper face holds the same words, added in
    //      1); one without it -- the layers G.twin marks 0, a thin set: the sub-grid's lowest layer of an axis it
    //      does not span round -- goes, in a serial sweep after 1 has joined, to the upper face of the full grid's
    //      cell below dst[l] (wrapped on a periodic grid), or for layer 0 of an axis of a grid that is not to that
    //      axis' boundary plane. No two sub-grid cells route to one word (oneToOne), and the sweep is serial anyway;
    //   3. the pass's carries by the same rule: an upper component's at its once index, a lower one's dropped (twin)
    //      or routed with its word; fxCarryFinish sums each target's lists (the plane's on their own).
    // Integer adds commute, so the words are the exact sum of every share added once, in any order of the adds, and
    // their carries the same at any thread count. 'stats' (DTFE_DEBUG_FACE_TWINS: else null, and nothing of it runs)
    // gets the twin pairs compared word for word and the routed words counted.
    void addFaceSlot(PSWordGrids &W, FxWords const &src, int const slot, std::vector<size_t> const &dst,
                     bool const oneToOne, FaceGeometry const &G, std::string const &name, FaceTwinStats *const stats)
    {
        size_t const per = PSWordGrids::facePer( slot );   // noVelComp (S), 1 (N, P)
        size_t const half = size_t( NO_DIM ) * per;          // one side's components = the once layout's per cell
        size_t const pcomps = 2 * half;                      // the pass's per cell
        size_t const subTotal = dst.size();
        size_t const skip = size_t( -1 );
        FxWords &to = W.at( slot );
        int64_t *const tw = to.w.data();
        int64_t const *const sw = src.w.data();
        size_t stride[NO_DIM];   // the full grid's (row-major, the last axis fastest)
        {
            size_t s = 1;
            for (int d = NO_DIM - 1; d >= 0; --d) { stride[d] = s; s *= W.dims[d]; }
        }
        // the developer check first (it reads the pass's words only): every twin pair, word for word
        if ( stats )
        {
            std::mutex statsMutex;
            poolRanges( subTotal, size_t(1) << 15, CorePool::total(), [&](size_t const lo, size_t const hi)
            {
                unsigned long long pairs = 0, bad = 0;
                for (size_t l = lo; l < hi; ++l)
                {
                    if ( dst[l] == skip ) continue;
                    for (int a = 0; a < NO_DIM; ++a)
                    {
                        size_t const c = ( l / G.stride[a] ) % G.m[a];
                        if ( not G.twin[a][c] ) continue;
                        size_t const lb = l - c * G.stride[a] + G.below[a][c] * G.stride[a];
                        for (size_t j = 0; j < per; ++j)
                        {
                            ++pairs;
                            if ( fxValue( src, l * pcomps + size_t( a ) * per + j )
                                 != fxValue( src, lb * pcomps + half + size_t( a ) * per + j ) ) ++bad;
                        }
                    }
                }
                std::lock_guard<std::mutex> lock( statsMutex );
                stats->pairs += pairs;
                stats->mismatches += bad;
            } );
        }
        // 1. the upper faces
        std::vector<FxCarryList> lists;
        std::mutex listsMutex;
        auto addUpper = [&](size_t const lo, size_t const hi)
        {
            FxCarryList own;
            for (size_t l = lo; l < hi; ++l)
            {
                size_t const g = dst[l];
                if ( g == skip ) continue;
                for (size_t c = 0; c < half; ++c)
                {
                    size_t const i = g * half + c;
                    int64_t const q = sw[l * pcomps + half + c];
                    int64_t r;
                    if ( __builtin_add_overflow( tw[i], q, &r ) )
                        own.push_back( FxCarry( uint64_t( i ), q > 0 ? 1 : -1 ) );
                    tw[i] = r;
                }
            }
            if ( not own.empty() )
            {
                std::lock_guard<std::mutex> lock( listsMutex );
                lists.push_back( std::move( own ) );
            }
        };
        if ( oneToOne ) poolRanges( subTotal, size_t(1) << 15, 8, addUpper );
        else addUpper( 0, subTotal );   // (two sub-grid cells may share a destination: one thread)
        // where the lower face (axis a, component j) of full-grid cell g goes: the upper face of the cell below
        // (wrapped where periodic) in W.at(slot), else (layer 0, not periodic) its cell of axis a's boundary plane
        auto target = [&](size_t const g, int const a, size_t const j, size_t &idx) -> FxWords &
        {
            size_t const ga = ( g / stride[a] ) % W.dims[a];
            if ( ga > 0 or W.periodic )
            {
                size_t const gb = ga > 0 ? g - stride[a] : g + ( W.dims[a] - 1 ) * stride[a];
                idx = gb * half + size_t( a ) * per + j;
                return to;
            }
            idx = ( ( g / ( W.dims[a] * stride[a] ) ) * stride[a] + g % stride[a] ) * per + j;
            return W.boundary( slot, a );
        };
        FxCarryList routed, planeList[NO_DIM];   // the crossings (and below the carries) of the routed words
        auto listOf = [&](FxWords const &F, int const a) -> FxCarryList &
        {
            return &F == &to ? routed : planeList[a];
        };
        // 2. the lower faces without a twin (serial, after 1)
        for (int a = 0; a < NO_DIM; ++a)
        {
            size_t const ma = G.m[a], sa = G.stride[a], outer = subTotal / ( ma * sa );
            for (size_t c = 0; c < ma; ++c)
            {
                if ( G.twin[a][c] ) continue;
                for (size_t h = 0; h < outer; ++h)
                    for (size_t r = 0; r < sa; ++r)
                    {
                        size_t const l = ( h * ma + c ) * sa + r;
                        size_t const g = dst[l];
                        if ( g == skip ) continue;
                        for (size_t j = 0; j < per; ++j)
                        {
                            size_t const k = l * pcomps + size_t( a ) * per + j;
                            size_t idx = 0;
                            FxWords &F = target( g, a, j, idx );
                            if ( stats )
                            {
                                bool const nonZero = fxValue( src, k ) != 0, plane = &F != &to;
                                ++stats->routed;
                                if ( nonZero ) ++stats->nonZero;
                                if ( plane ) ++stats->toPlane;
                                if ( plane and nonZero ) ++stats->planeNonZero;
                            }
                            int64_t const q = sw[k];
                            int64_t sum;
                            if ( __builtin_add_overflow( F.w[idx], q, &sum ) )
                                listOf( F, a ).push_back( FxCarry( uint64_t( idx ), q > 0 ? 1 : -1 ) );
                            F.w[idx] = sum;
                        }
                    }
            }
        }
        // 3. the pass's own carries
        for (FxCarry const &e : src.carry)
        {
            size_t const l = size_t( e.first ) / pcomps, c = size_t( e.first ) % pcomps;
            if ( l >= subTotal or dst[l] == skip ) continue;
            size_t const g = dst[l];
            if ( c >= half )
            {   // an upper face's: at its once index
                routed.push_back( FxCarry( uint64_t( g * half + ( c - half ) ), e.second ) );
                continue;
            }
            int const a = int( c / per );
            size_t const j = c % per;
            if ( G.twin[a][( l / G.stride[a] ) % G.m[a]] ) continue;   // (its twin's carry: added at the twin's)
            size_t idx = 0;
            FxWords &F = target( g, a, j, idx );
            listOf( F, a ).push_back( FxCarry( uint64_t( idx ), e.second ) );
        }
        if ( not routed.empty() ) lists.push_back( std::move( routed ) );
        if ( not lists.empty() ) fxCarryFinish( to, lists, name.c_str() );
        for (int a = 0; a < NO_DIM; ++a)
            if ( not planeList[a].empty() )
            {
                std::vector<FxCarryList> one( 1 );
                one[0].swap( planeList[a] );
                fxCarryFinish( W.boundary( slot, a ), one, ( name + " (boundary plane)" ).c_str() );
            }
    }

    // The word merge: one side of one partition (S, its sub-grid Quantities' words) into the full grid's (W). Each
    // sub-grid cell's destination once (subgridTarget, as addOne maps), then per accumulator the owned adds of its
    // words -- over the free cores when no two sub-grid cells share a destination (every axis of the sub-grid within
    // the full grid's, as addFromSubgrid's rows), each range's crossings in a list of its own -- and its carries at
    // their full-grid components; fxCarryFinish sums the lists into the full grid's carry (sorted, so the order of
    // the lists does not matter). The face sums by the face rule instead (addFaceSlot: once per face, fix 3 S7). The
    // counts likewise, plain int adds (a sum of 2^31 is refused, never wrapped).
    // The destination is the full grid, never a window: '--grid-window' is refused beside '--ps-merge fixed-point'
    // (user_options.cc readPartitionOptions), so mainOrigin is null and no sub-grid cell is skipped (subgridTarget).
    void addWordsSide(PSWordGrids &W, PSWordGrids &S, Quantities const &sub, size_t const *fullGrid,
                      size_t const *mainOrigin, size_t const *mainDims, int const partition, int const side)
    {
        char const *const sideName = side ? "a" : "u";
        if ( S.exact != W.exact or S.samplesPerCell != W.samplesPerCell or S.dwIsMass != W.dwIsMass )
            throwError( "internal: the word merge: partition ", partition, " handed over words of another deposit, "
                        "sub-sample count or dispersion weight than the run's (side ", sideName, ")." );
        if ( mainOrigin != nullptr or not W.faceOnce or S.faceOnce )
            throwError( "internal: the word merge adds a pass's words (one value per cell and face side) into the "
                        "full grid's (each face once), never into a window: partition ", partition, " (side ",
                        sideName, ")." );
        size_t dstTotal = 1;
        for (int d = 0; d < NO_DIM; ++d) dstTotal *= mainOrigin ? mainDims[d] : fullGrid[d];
        if ( W.cells != dstTotal )
            throwError( "internal: the word merge's full grid holds ", W.cells, " cells, the destination ", dstTotal,
                        "." );
        std::vector<size_t> dst;
        bool oneToOne = true;
        size_t m[NO_DIM], o[NO_DIM];
        cellMap( S, sub, fullGrid, partition, dst, oneToOne, m, o );
        size_t const subTotal = S.cells;
        size_t const skip = size_t( -1 );
        // the face rule's geometry of this pass, when it holds face sums (its sub-box in the passes' own grid, which
        // wraps as the run's: the full grid's words take their periodicity from the same options, onceLayout)
        bool faces = false;
        for (int slot = RU_PS_FACE_S; slot <= RU_PS_FACE_P; ++slot) faces = faces or not S.at( slot ).w.empty();
        FaceGeometry const geo = faces ? faceGeometry( m, o, fullGrid, W.periodic ) : FaceGeometry();
        if ( faces and debugTwins ) ++twins[side].passes;
        for (int k = 0; k < PSWordGrids::N; ++k)
        {
            int const slot = RU_PS_MASS + k;
            FxWords &src = S.g[k];
            if ( src.w.empty() ) continue;
            size_t const comps = S.cellComps( slot );   // (the pass's layout: per cell, both faces of a face sum)
            std::string const name = std::string( sideName ) + "." + PSWordGrids::key( slot );
            if ( src.w.size() != subTotal * comps or not units.has( slot ) or src.K != units.K[slot] )
                throwError( "internal: the word merge: partition ", partition, "'s words of ", name,
                            " are not the run-wide unit's or not its sub-grid's." );
            FxWords &to = wordsFor( W, slot, src.K, name );
            if ( PSWordGrids::isFace( slot ) )
            {
                addFaceSlot( W, src, slot, dst, oneToOne, geo, name, debugTwins ? &twins[side] : nullptr );
                src = FxWords();   // (freed)
                continue;
            }
            int64_t *const tw = to.w.data();
            int64_t const *const sw = src.w.data();
            std::vector<FxCarryList> lists;
            std::mutex listsMutex;
            auto addRange = [&](size_t const lo, size_t const hi)
            {
                FxCarryList own;
                for (size_t l = lo; l < hi; ++l)
                {
                    size_t const g = dst[l];
                    if ( g == skip ) continue;
                    for (size_t c = 0; c < comps; ++c)
                    {
                        size_t const i = g * comps + c;
                        int64_t const q = sw[l * comps + c];
                        int64_t r;
                        if ( __builtin_add_overflow( tw[i], q, &r ) )
                            own.push_back( FxCarry( uint64_t( i ), q > 0 ? 1 : -1 ) );
                        tw[i] = r;
                    }
                }
                if ( not own.empty() )
                {
                    std::lock_guard<std::mutex> lock( listsMutex );
                    lists.push_back( std::move( own ) );
                }
            };
            if ( oneToOne ) poolRanges( subTotal, size_t(1) << 15, 8, addRange );
            else addRange( 0, subTotal );   // (two sub-grid cells may share a destination: one thread)
            FxCarryList moved;   // the partition's own carries, at their full-grid components
            for (FxCarry const &e : src.carry)
            {
                size_t const l = size_t( e.first ) / comps, c = size_t( e.first ) % comps;
                if ( l < subTotal and dst[l] != skip )
                    moved.push_back( FxCarry( uint64_t( dst[l] * comps + c ), e.second ) );
            }
            if ( not moved.empty() ) lists.push_back( std::move( moved ) );
            if ( not lists.empty() ) fxCarryFinish( to, lists, name.c_str() );
            src = FxWords();   // (freed)
        }
        if ( not S.count.empty() )
        {
            if ( S.count.size() != subTotal )
                throwError( "internal: the word merge: partition ", partition, "'s stream counts cover ",
                            S.count.size(), " cells, its words " + std::to_string( subTotal ) + "." );
            if ( W.count.empty() )
            {
                ScratchAlloc::zeroAssign( W.count, dstTotal, int( 0 ) );
                account();
            }
            for (size_t l = 0; l < subTotal; ++l)
            {
                size_t const g = dst[l];
                if ( g == skip ) continue;
                int r;
                if ( __builtin_add_overflow( W.count[g], S.count[l], &r ) )
                    throwError( "--ps-merge fixed-point: a cell's stream count left the int range (2^31) in the "
                                "merge of partition ", partition, " (side ", sideName, "). The run stops." );
                W.count[g] = r;
            }
        }
    }

    // The word merge: one partition's two sub-grid Quantities -- their words into the full grid's (addWordsSide),
    // freed, and their OR-merged masks into the full Real grids by addFromSubgrid (every summed vector being empty,
    // it does only those; a summed one that is not, a Real sum the pass should have handed over as words, stops the
    // run)
    void addWords(int const partition, Quantities &tu, Quantities &ta, Quantities &fu, Quantities &fa,
                  size_t const *fullGrid, size_t const *mainOrigin, size_t const *mainDims)
    {
        for (int side = 0; side < 2; ++side)
        {
            Quantities &t = side ? ta : tu;
            forEachSummed( t, [&](Quantities::FieldInfo const &fi, auto &v)
            {
                if ( not v.empty() )
                    throwError( "internal: the word merge: partition ", partition, " handed over its summed grid ",
                                fi.name, " as Real values, not as words." );
            } );
            if ( t.psWords.on )
                addWordsSide( full[side], t.psWords, t, fullGrid, mainOrigin, mainDims, partition, side );
            t.psWords.release();
            ( side ? fa : fu ).addFromSubgrid( t, fullGrid, mainOrigin, mainDims );
        }
        account();
    }

    // The word merge's checkpoint state per side (fxwords.state): the accumulators' K and word counts, then the face
    // sums' boundary planes' (savedPlane: face sum f, axis a), then the stream counts' size
    static size_t savedPer() { return size_t( PSWordGrids::N ) + size_t( PSWordGrids::NF ) * size_t( NO_DIM ) + 1; }
    static size_t savedPlane(int const f, int const a)
    {
        return size_t( PSWordGrids::N ) + size_t( f ) * size_t( NO_DIM ) + size_t( a );
    }

    // The word merge's checkpoint blobs (the header comment): per side and accumulator its words and its carry (the
    // face sums in their once layout, fix 3 S7), per face sum and axis its boundary plane's words and carry
    // ('fxw.<side>.faceS.plane0', 'fxk...': none on a periodic grid), per side the counts, and the state (K and the
    // sizes); registered before beginLoop, a resume loads into them
    std::vector<PartitionCheckpoint::Blob> wordBlobs()
    {
        std::vector<PartitionCheckpoint::Blob> b;
        // one FxWords' two blobs: its words (in unit 2^-K) and its carry
        auto wordsBlobs = [&](FxWords *const F, int const K, std::string const &key)
        {
            b.push_back( { "fxw." + key, sizeof( int64_t ),
                           [F]() { return F->w.size(); },
                           [F]() { return static_cast<void const *>( F->w.data() ); },
                           [this, F, K](size_t const n)
                           {
                               F->w.assign( n, 0 );
                               F->K = K;
                               account();
                               return static_cast<void *>( F->w.data() );
                           } } );
            b.push_back( { "fxk." + key, sizeof( FxCarry ),
                           [F]() { return F->carry.size(); },
                           [F]() { return static_cast<void const *>( F->carry.data() ); },
                           [this, F](size_t const n)
                           {
                               F->carry.assign( n, FxCarry() );
                               account();
                               return static_cast<void *>( F->carry.data() );
                           } } );
        };
        for (int side = 0; side < 2; ++side)
        {
            std::string const s = side ? "a." : "u.";
            for (int k = 0; k < PSWordGrids::N; ++k)
                wordsBlobs( &full[side].g[k], units.K[RU_PS_MASS + k], s + PSWordGrids::key( RU_PS_MASS + k ) );
            for (int f = 0; f < PSWordGrids::NF; ++f)
                for (int a = 0; a < NO_DIM; ++a)
                    wordsBlobs( &full[side].plane[f][a], units.K[RU_PS_FACE_S + f],
                                s + PSWordGrids::key( RU_PS_FACE_S + f ) + ".plane" + std::to_string( a ) );
            std::vector<int> *C = &full[side].count;
            b.push_back( { "fxn." + s.substr( 0, 1 ), sizeof( int ),
                           [C]() { return C->size(); },
                           [C]() { return static_cast<void const *>( C->data() ); },
                           [this, C](size_t const n)
                           {
                               C->assign( n, 0 );
                               account();
                               return static_cast<void *>( C->data() );
                           } } );
        }
        size_t const per = savedPer();   // the accumulators, the boundary planes and the counts, per side
        b.push_back( { "fxwords.state", sizeof( SavedWords ),
                       [per]() { return 2 * per; },
                       [this, per]()
                       {
                           loadedWords.assign( 2 * per, SavedWords() );
                           for (int side = 0; side < 2; ++side)
                           {
                               for (int k = 0; k < PSWordGrids::N; ++k)
                               {
                                   SavedWords &t = loadedWords[size_t( side ) * per + size_t( k )];
                                   t.K = full[side].g[k].K;
                                   t.n = full[side].g[k].w.size();
                               }
                               for (int f = 0; f < PSWordGrids::NF; ++f)
                                   for (int a = 0; a < NO_DIM; ++a)
                                   {
                                       SavedWords &t = loadedWords[size_t( side ) * per + savedPlane( f, a )];
                                       t.K = full[side].plane[f][a].K;
                                       t.n = full[side].plane[f][a].w.size();
                                   }
                               loadedWords[size_t( side ) * per + per - 1].n = full[side].count.size();
                           }
                           return static_cast<void const *>( loadedWords.data() );
                       },
                       [this](size_t const n)
                       {
                           loadedWords.assign( n, SavedWords() );
                           return static_cast<void *>( loadedWords.data() );
                       } } );
        return b;
    }

    // after a checkpoint was loaded ('--resume') under the word merge: the saved units must be this run's (the words
    // of another unit cannot be added to this run's) and the words and counts as many as the state says -- the face
    // sums in the once layout with their boundary planes (fix 3 S7: a checkpoint of another layout is refused here,
    // by its state's size or its words' sizes)
    void restoreWords()
    {
        size_t const per = savedPer();
        if ( loadedWords.size() != 2 * per )
            throwError( "'--resume': the checkpoint holds no word-merge state for this run (", loadedWords.size(),
                        " entries saved, " + std::to_string( 2 * per ) + " here). Run without '--resume' to start "
                        "over (run_ps_dtfe.sh: DTFE_RESUME=0)." );
        for (int side = 0; side < 2; ++side)
        {
            PSWordGrids const &W = full[side];
            // one FxWords against its saved state: as many words as saved, 'want' of them (or none), its unit this
            // run's, its carry sorted
            auto check = [&](SavedWords const &t, FxWords const &F, size_t const want, int const K,
                             std::string const &name)
            {
                if ( t.n != F.w.size() or ( t.n > 0 and F.w.size() != want ) )
                    throwError( "'--resume': the checkpoint's words of ", name, " do not match its state (cut or "
                                "damaged, or of another layout). Run without '--resume' to start over." );
                if ( t.n > 0 and t.K != int64_t( K ) )
                    throwError( "'--resume': the checkpoint's unit of ", name, " is 2^-" + std::to_string( t.K )
                                + ", this run's 2^-" + std::to_string( K ) + " (DTFE_TEST_RUN_UNIT_SHIFT "
                                "or the particles changed): its words cannot be added to this run's. Run without "
                                "'--resume' to start over." );
                for (size_t j = 1; j < F.carry.size(); ++j)
                    if ( not ( F.carry[j - 1].first < F.carry[j].first ) )
                        throwError( "'--resume': the checkpoint's carry of ", name, " is not sorted (damaged). Run "
                                    "without '--resume' to start over." );
            };
            for (int k = 0; k < PSWordGrids::N; ++k)
            {
                int const slot = RU_PS_MASS + k;
                std::string const name = std::string( side ? "a." : "u." ) + PSWordGrids::key( slot );
                check( loadedWords[size_t( side ) * per + size_t( k )], W.g[k], W.cells * W.cellComps( slot ),
                       units.K[slot], name );
            }
            // a face sum's boundary planes: there exactly when its words are and the grid is not periodic
            for (int f = 0; f < PSWordGrids::NF; ++f)
                for (int a = 0; a < NO_DIM; ++a)
                {
                    int const slot = RU_PS_FACE_S + f;
                    std::string const name = std::string( side ? "a." : "u." ) + PSWordGrids::key( slot ) + ".plane"
                                             + std::to_string( a );
                    FxWords const &B = W.plane[f][a];
                    size_t const want = W.planeCells( a ) * PSWordGrids::facePer( slot );
                    check( loadedWords[size_t( side ) * per + savedPlane( f, a )], B, want, units.K[slot], name );
                    if ( B.w.empty() != ( W.at( slot ).w.empty() or W.periodic ) )
                        throwError( "'--resume': the checkpoint's boundary plane ", name, " does not match its face "
                                    "sums (cut or damaged). Run without '--resume' to start over." );
                }
            if ( loadedWords[size_t( side ) * per + per - 1].n != full[side].count.size()
                 or ( not full[side].count.empty() and full[side].count.size() != full[side].cells ) )
                throwError( "'--resume': the checkpoint's stream counts (side ", side ? "a" : "u", ") do not match "
                            "its state (cut or damaged). Run without '--resume' to start over." );
        }
        std::vector<SavedWords>().swap( loadedWords );
    }

    // after the loop, the word merge: the full grid's words go to the full Quantities (psWords: the face sums in
    // their once layout, with their boundary planes, dims and periodicity), where ONE close-out converts and
    // normalizes them (DTFE.cpp psCloseOutMoments); DTFE_DEBUG_FACE_TWINS=1: its line per side
    void finishWords(Quantities &u, Quantities &a)
    {
        carried = 0;
        for (int side = 0; side < 2; ++side)
        {
            for (int k = 0; k < PSWordGrids::N; ++k) carried += full[side].g[k].carry.size();
            for (int f = 0; f < PSWordGrids::NF; ++f)
                for (int d = 0; d < NO_DIM; ++d) carried += full[side].plane[f][d].carry.size();
        }
        if ( debugTwins )
            for (int side = 0; side < 2; ++side) faceTwinsLine( twins[side], side );
        account();
        u.psWords = std::move( full[0] );
        a.psWords = std::move( full[1] );
        full[0].release();
        full[1].release();
        account();
    }

    // the checkpoint's blobs: every slot's words and carries, and the state (registered before beginLoop: a resume
    // loads into them). The word merge: wordBlobs.
    std::vector<PartitionCheckpoint::Blob> blobs()
    {
        if ( words ) return wordBlobs();
        std::vector<PartitionCheckpoint::Blob> b;
        for (size_t i = 0; i < slots.size(); ++i)
        {
            Slot *s = &slots[i];
            b.push_back( { "fxw." + s->name, sizeof( int64_t ),
                           [s]() { return s->w.size(); },
                           [s]() { return static_cast<void const *>( s->w.data() ); },
                           [this, s](size_t const n) { s->w.assign( n, 0 ); s->cells = n / s->comps; account(); return static_cast<void *>( s->w.data() ); } } );
            b.push_back( { "fxc." + s->name, sizeof( int8_t ),
                           [s]() { return s->c.size(); },
                           [s]() { return static_cast<void const *>( s->c.data() ); },
                           [this, s](size_t const n) { s->c.assign( n, 0 ); account(); return static_cast<void *>( s->c.data() ); } } );
        }
        b.push_back( { "fxmerge.state", sizeof( SavedSlot ),
                       [this]() { return slots.size(); },
                       [this]()
                       {
                           loadedState.assign( slots.size(), SavedSlot() );
                           for (size_t i = 0; i < slots.size(); ++i)
                           {
                               SavedSlot &t = loadedState[i];
                               t.K = slots[i].K;
                               t.boundLo = uint64_t( slots[i].bound );
                               t.boundHi = uint64_t( slots[i].bound >> 64 );
                               t.maxQ = slots[i].maxQ;
                               t.cells = slots[i].cells;
                           }
                           return static_cast<void const *>( loadedState.data() );
                       },
                       [this](size_t const n) { loadedState.assign( n, SavedSlot() ); return static_cast<void *>( loadedState.data() ); } } );
        return b;
    }

    // after a checkpoint was loaded ('--resume'): the saved units must be this run's; restores the bound (b) and the
    // largest |q| (they cannot be recomputed from the summed words). The word merge: restoreWords.
    void restore()
    {
        if ( words )
        {
            restoreWords();
            return;
        }
        if ( loadedState.size() != slots.size() )
            throwError( "'--resume': the checkpoint holds no fixed-point merge state for this run's accumulators (",
                        loadedState.size(), " saved, " + std::to_string( slots.size() ) + " here). Run without '--resume' "
                        "to start over (run_ps_dtfe.sh: DTFE_RESUME=0)." );
        for (size_t i = 0; i < slots.size(); ++i)
        {
            Slot &s = slots[i];
            SavedSlot const &t = loadedState[i];
            if ( t.K != int64_t( s.K ) )
                throwError( "'--resume': the checkpoint's fixed-point unit of ", s.name, " is 2^-" + std::to_string( t.K )
                            + ", this run's 2^-" + std::to_string( s.K ) + " (DTFE_TEST_FXMERGE_SHIFT or the particles "
                            "changed): its integers cannot be added to this run's. Run without '--resume' to start over." );
            if ( ( t.cells == 0 ) != s.w.empty() or ( t.cells != 0 and ( s.w.size() != t.cells * s.comps
                                                                         or s.c.size() != s.w.size() ) ) )
                throwError( "'--resume': the checkpoint's fixed-point words of ", s.name, " do not match its state (cut or "
                            "damaged). Run without '--resume' to start over." );
            s.bound = ( (unsigned __int128)t.boundHi << 64 ) | (unsigned __int128)t.boundLo;
            s.maxQ = t.maxQ;
            s.cells = size_t( t.cells );
        }
        std::vector<SavedSlot>().swap( loadedState );
    }

    void account()
    {
        double b = 0.;
        for (Slot const &s : slots) b += double( s.w.capacity() ) * 8. + double( s.c.capacity() );
        auto bytesOf = [](FxWords const &F)
        {
            return double( F.w.capacity() ) * 8. + double( F.carry.capacity() ) * double( sizeof( FxCarry ) );
        };
        for (PSWordGrids const &W : full)   // (the word merge's; empty otherwise)
        {
            for (FxWords const &F : W.g) b += bytesOf( F );   // (the face sums once per face: cellComps)
            for (int f = 0; f < PSWordGrids::NF; ++f)
                for (int d = 0; d < NO_DIM; ++d) b += bytesOf( W.plane[f][d] );   // their boundary planes
            b += double( W.count.capacity() ) * double( sizeof( int ) );
        }
        liveBytes = b;
        peakBytes = std::max( peakBytes, b );
    }

    // one accumulator of one partition into its slot (under the merge lock)
    template<class T>
    void addOne(Slot &s, std::vector<T> const &src, Quantities const &sub, size_t const *fullGrid,
                size_t const *mainOrigin, size_t const *mainDims, int const partition)
    {
        if ( src.empty() ) return;
        size_t dstTotal = 1;
        for (int d = 0; d < NO_DIM; ++d) dstTotal *= mainOrigin ? mainDims[d] : fullGrid[d];
        bool const isSub = sub.ps_subDims[0] != 0;   // else 'sub' spans the main grid, cell for cell (addFrom)
        if ( not isSub ) dstTotal = std::max( dstTotal, src.size() );
        if ( s.cells == 0 )
        {   // (zeroAssign: as at setup)
            ScratchAlloc::zeroAssign( s.w, dstTotal * s.comps, int64_t( 0 ) );
            ScratchAlloc::zeroAssign( s.c, dstTotal * s.comps, int8_t( 0 ) );
            s.cells = dstTotal;
            account();
        }
        if ( s.cells < dstTotal )
            throwError( "internal: the fixed-point merge's ", s.name, " holds " + std::to_string( s.cells ) + " cells, a "
                        "partition delivers " + std::to_string( dstTotal ) );
        size_t const comps = s.comps;
        Real const *v = reinterpret_cast<Real const *>( src.data() );
        size_t subTotal = src.size();
        size_t const *o = sub.ps_subOrigin, *m = sub.ps_subDims;
        uint64_t pmax = 0;
        int64_t *w = s.w.data();
        int8_t *c = s.c.data();
        double const unit = std::ldexp( 1., s.K );
        for (size_t l = 0; l < subTotal; ++l)
        {
            size_t g = l;
            if ( isSub )
            {
                size_t rem = l, cc[NO_DIM];
                for (int d = NO_DIM - 1; d >= 0; --d) { cc[d] = rem % m[d]; rem /= m[d]; }   // local coords
                if ( not subgridTarget( cc, o, fullGrid, mainOrigin, mainDims, g ) ) continue;
            }
            for (size_t k = 0; k < comps; ++k)
            {
                double const x = double( v[l*comps + k] ) * unit;   // exact: a power of two
                if ( not ( std::fabs( x ) < VALUE_LIMIT ) )   // NaN and inf too
                    valueError( s, partition, double( v[l*comps + k] ) );
                int64_t const q = int64_t( std::llrint( x ) );
                uint64_t const aq = q < 0 ? uint64_t( 0 ) - uint64_t( q ) : uint64_t( q );
                pmax = std::max( pmax, aq );
                size_t const i = g * comps + k;
                int64_t r;
                if ( __builtin_add_overflow( w[i], q, &r ) ) c[i] = int8_t( c[i] + ( q > 0 ? 1 : -1 ) );
                w[i] = r;
            }
        }
        s.bound += pmax;
        s.maxQ = std::max( s.maxQ, pmax );
        if ( s.bound >= sumLimit )
            throwError( "--ps-merge fixed-point: the merged partitions' largest values of ", s.name, " sum to 2^"
                        + fmt( log2u128( s.bound ), 4 ) + " units (" + fmt( std::ldexp( double( s.bound ), -s.K ) / s.S, 3 )
                        + " times its scale S = " + fmt( s.S, 4 ) + "), the limit " + ( sumLimitLog2 > 0 ? "2^" + std::to_string(
                        sumLimitLog2 ) + " (DTFE_TEST_FXMERGE_SUM_LIMIT)" : std::string( "2^71 - 2^63 of its 72-bit sums" ) ) + ": a cell "
                        "could overflow, which could not be repaired. The run stops. Run with the default (ordered) "
                        "merge, which has no such limit.", "" );
    }

    static std::string fmt(double const x, int const prec)
    {
        char b[64];
        std::snprintf( b, sizeof b, "%.*g", prec, x );
        return b;
    }

    [[noreturn]] void valueError(Slot const &s, int const partition, double const v)
    {
        if ( not std::isfinite( v ) )
            throwError( "--ps-merge fixed-point: partition ", partition, " holds a non-finite value (" + fmt( v, 6 ) + ") in "
                        + s.name + ", which no integer unit can hold. The run stops." );
        throwError( "--ps-merge fixed-point: partition ", partition, " holds a value of " + s.name + " of " + fmt( v, 6 )
                    + " = " + fmt( v / s.S, 4 ) + " times its scale S = " + fmt( s.S, 4 ) + "; its unit 2^-"
                    + std::to_string( s.K ) + " holds |values| below " + fmt( std::ldexp( 1., 63 - s.K ) / s.S, 4 )
                    + " S (2^63 units). A wrapped cell could not be repaired: the run stops. Run with the default "
                    "(ordered) merge, which has no such limit.", "" );
        std::abort();   // throwError does not return
    }

    // one partition's two sub-grid Quantities into the integer grids (their summed vectors are released), and their
    // OR-merged masks into the full Real grids by addFromSubgrid (the summed vectors being empty, it does only
    // those). The word merge: addWords.
    void add(int const partition, Quantities &tu, Quantities &ta, Quantities &fu, Quantities &fa,
             size_t const *fullGrid, size_t const *mainOrigin, size_t const *mainDims)
    {
        if ( words )
        {
            addWords( partition, tu, ta, fu, fa, fullGrid, mainOrigin, mainDims );
            return;
        }
        for (int side = 0; side < 2; ++side)
        {
            Quantities &t = side ? ta : tu;
            size_t i = side * perSide();
            forEachSummed( t, [&](Quantities::FieldInfo const &, auto &v)
            {
                using V = typename std::decay<decltype( v )>::type;
                addOne( slots[i++], v, t, fullGrid, mainOrigin, mainDims, partition );
                V().swap( v );
            } );
            ( side ? fa : fu ).addFromSubgrid( t, fullGrid, mainOrigin, mainDims );
        }
    }

    // after the loop: every slot to its Real grid, one at a time (its integers freed right after); a pre-sized grid no
    // partition filled comes back as zeros, as the default merge leaves it. The word merge: finishWords (the
    // close-out converts).
    void finish(Quantities &u, Quantities &a)
    {
        if ( words )
        {
            finishWords( u, a );
            return;
        }
        for (int side = 0; side < 2; ++side)
        {
            size_t i = side * perSide();
            forEachSummed( side ? a : u, [&](Quantities::FieldInfo const &, auto &v)
            {
                using V = typename std::decay<decltype( v )>::type;
                using T = typename V::value_type;
                Slot &s = slots[i++];
                if ( s.cells > 0 )
                {
                    v.assign( s.cells, T() );
                    Real *out = reinterpret_cast<Real *>( v.data() );
                    size_t const n = s.cells * s.comps;
                    for (size_t j = 0; j < n; ++j)
                    {
                        __int128 const t = (__int128)s.c[j] * ( (__int128)1 << 64 ) + (__int128)s.w[j];
                        out[j] = std::ldexp( static_cast<Real>( t ), -s.K );   // one rounding, then an exact scale
                    }
                    std::vector<int64_t>().swap( s.w );
                    std::vector<int8_t>().swap( s.c );
                }
                else if ( s.presized > 0 )
                {
                    v.assign( s.presized, T() );
                    std::memset( static_cast<void *>( v.data() ), 0, s.presized * sizeof( T ) );
                }
            } );
        }
        account();
    }

    // the units, said before the loop: the allocated slots are not known yet, so every selected kind once
    std::string unitsText() const
    {
        char b[200];
        if ( words )   // (the passes' units: the run's line names every accumulator's)
        {
            std::snprintf( b, sizeof b, "the run-wide ones of the 'fixed-point units (run-wide)' line (mass 2^-%d, "
                           "MOM 2^-%d, GRAD 2^-%d, ...)", units.K[RU_PS_MASS], units.K[RU_PS_FX + 0],
                           units.K[RU_PS_FX + 2] );
            return b;
        }
        std::snprintf( b, sizeof b, "cell mass %.4g, cell volume %.4g, max |velocity component| %.4g%s",
                       cellMass, cellVol, vmax, shift ? ( ", DTFE_TEST_FXMERGE_SHIFT=" + std::to_string( shift ) ).c_str() : "" );
        return b;
    }

    // '--scratch-dir', after the loop: the scratch mappings so far, and how many grids were zero-filled in memory and
    // moved onto their file unwritten -- in all, and since this merge's setup (its integer grids: the psReserveGrids
    // line, DTFE.cpp, prints before the setup; tests/ps_fixed_merge_check.sh X3 reads both counts)
    std::string scratchText() const
    {
        char b[300];
        long const moves = ScratchAlloc::scratchZeroFillMoves();
        std::snprintf( b, sizeof b, "SCRATCH: %ld grid(s) on scratch files in all, %.4g GB now; %ld zero-filled in memory "
                       "and moved onto their file unwritten, %ld of them since the fixed-point merge's setup (its integer "
                       "grids).", ScratchAlloc::scratchTotalAllocs(), double( ScratchAlloc::scratchLiveBytes() ) / 1.e9,
                       moves, moves - zeroFillMoves0 );
        return b;
    }

    // the report after the loop: the bytes, and per filled accumulator its unit and its headroom (one value's largest
    // |q| below 2^63, the sum bound (b) below 2^71)
    std::string reportText() const
    {
        std::string t;
        char b[400];
        if ( words )   // (no headroom to report: the words are exact at any magnitude a share cap admits)
        {
            std::snprintf( b, sizeof b, "fixed-point merge: %.2f GB of integer grids at most (the word merge: 8 B "
                           "per summed component and cell, the face sums once per face, 4 B per count, 16 B per "
                           "carried component); %zu component(s) carried beyond 64 bits; nothing parked or spilled",
                           peakBytes / 1.e9, carried );
            return b;
        }
        std::snprintf( b, sizeof b, "fixed-point merge: %.2f GB of integer grids at most (9 B per summed component and "
                       "cell); nothing parked or spilled", peakBytes / 1.e9 );
        t = b;
        for (Slot const &s : slots)
        {
            if ( s.maxQ == 0 and s.bound == 0 ) continue;
            std::snprintf( b, sizeof b, "; %s: unit 2^-%d (S %.3g), headroom 2^%.1f per value, 2^%.1f per sum",
                           s.name.c_str(), s.K, s.S, 63. - std::log2( double( std::max<uint64_t>( s.maxQ, 1 ) ) ),
                           71. - std::max( 0., log2u128( s.bound ) ) );
            t += b;
        }
        return t;
    }
};

// A one-domain run under the word merge (DTFE.cpp psOneDomainCloseOut, before the close-out; fix 3 S7): its pass
// handed its face sums over in the pass's layout (per cell, both faces of every axis). They go into the once layout
// of the merged full grid (onceLayout) by the merge's own face rule (Merge::addFaceSlot, with the pass's cell map:
// its sub-grid is the whole grid, so every cell maps to itself and only the lower faces of a non-periodic grid's
// layer 0 lack a twin), so the close-out reads the words a partitioned run's merge leaves and the bytes cannot depend
// on the route. The other accumulators stay as they are: an identity merge would add the same words. A no-op without
// face words (and for words already once per face); DTFE_DEBUG_FACE_TWINS=1: the side's line, as finishWords prints.
inline void oneDomainFacesOnce(Quantities &q, User_options const &opt, int const side)
{
    PSWordGrids &S = q.psWords;
    if ( not S.on or S.faceOnce ) return;
    bool faces = false;
    for (int slot = RU_PS_FACE_S; slot <= RU_PS_FACE_P; ++slot) faces = faces or not S.at( slot ).w.empty();
    if ( not faces ) return;
    char const *const sideName = side ? "a" : "u";
    if ( opt.gridSize.size() != size_t( NO_DIM ) )
        throwError( "internal: the word merge: the one domain's face sums without a grid (side ", sideName, ")." );
    size_t const *const grid = opt.gridSize.data();
    size_t cells = 1;
    bool whole = true;   // the pass's sub-grid record: the whole grid (or none)
    for (int d = 0; d < NO_DIM; ++d)
    {
        cells *= grid[d];
        whole = whole and q.ps_subOrigin[d] == 0 and ( q.ps_subDims[0] == 0 or q.ps_subDims[d] == grid[d] );
    }
    if ( S.cells != cells or not whole )
        throwError( "internal: the word merge: the one domain's pass holds ", S.cells, " cells, not its grid's "
                    "whole ", cells, " (side " + std::string( sideName ) + ")." );
    Merge m;   // (its face rule, in the run's units)
    m.words = true;
    m.units = opt.runUnits;
    bool const debug = Env::flag( "DTFE_DEBUG_FACE_TWINS" );
    FaceTwinStats stats;
    stats.passes = 1;
    std::vector<size_t> dst;
    bool oneToOne = true;
    size_t mm[NO_DIM], o[NO_DIM];
    Merge::cellMap( S, q, grid, 0, dst, oneToOne, mm, o );
    PSWordGrids D;   // the once layout's face sums, then moved into the pass's words
    D.on = true;
    D.cells = cells;
    onceLayout( D, opt );
    FaceGeometry const geo = faceGeometry( mm, o, grid, D.periodic );
    for (int slot = RU_PS_FACE_S; slot <= RU_PS_FACE_P; ++slot)
    {
        FxWords &src = S.at( slot );
        if ( src.w.empty() ) continue;
        std::string const name = std::string( sideName ) + "." + PSWordGrids::key( slot );
        if ( src.w.size() != cells * PSWordGrids::comps( slot ) or not m.units.has( slot )
             or src.K != m.units.K[slot] )
            throwError( "internal: the word merge: the one domain's words of ", name, " are not the run-wide unit's "
                        "or not its grid's." );
        m.wordsFor( D, slot, src.K, name );
        m.addFaceSlot( D, src, slot, dst, oneToOne, geo, name, debug ? &stats : nullptr );
        src = std::move( D.at( slot ) );   // (the pass's per-cell words freed)
        for (int a = 0; a < NO_DIM; ++a) S.boundary( slot, a ) = std::move( D.boundary( slot, a ) );
    }
    S.faceOnce = true;
    S.periodic = D.periodic;
    for (int d = 0; d < NO_DIM; ++d) S.dims[d] = D.dims[d];
    if ( debug ) faceTwinsLine( stats, side );
}

}   // namespace PSFixedMerge

#endif
