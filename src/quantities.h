/*
 *  Copyright (c) 2011       Marius Cautun
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */



/*
  Holds every quantity the DTFE grid interpolation can compute, each in its own vector so memory
  is allocated only for requested quantities. Pvector ("Pvector.h"); component counts in "define.h".
*/



#ifndef QUANTITIES_HEADER
#define QUANTITIES_HEADER

#include <type_traits>
#include <vector>

#include "define.h"
#include "particle_data.h"
#include "user_options.h"


#ifdef PHASE_SPACE
// Tolerance for testing the stream multiplicity ('stream_count', written as '.streams') against
// an integer. It is a FLOAT sum under every deposit, so single-stream cells land on 1.0 +/- eps
// and an exact '== 1' / '> 1' test is always wrong: on the 48^3 Zel'dovich pancake '> 1.0f'
// reports 28.33% of cells as multi-stream where the true slab is 16.67%. Mirrors
// dtfelib.STREAM_TOL on the Python side -- keep the two in sync. The plateau is sharp (every
// single-stream cell measures inside 1e-5 of 1.0), so 1e-3 admits noise but no real structure.
Real const PS_STREAM_TOL = Real(1.e-3);

// A partition sub-grid cell (local coords c; sub-grid dims m, global origin o) -> its index in the full grid 'full'
// (used by quantities.cc SubgridRowMap/mergeSubgridCells and the fixed-point merge, src/ps_fixed_merge.h). Each cell
// maps to its global row-major index; o, m, full are in grid cells.
// The destination may itself be a WINDOW of the full grid (--grid-window: mainOrigin/mainDims given):
// a source cell is then mapped to its window-local index and skipped when it lies outside.
inline bool subgridTarget(size_t const *c, size_t const *o, size_t const *full,
                          size_t const *mainOrigin, size_t const *mainDims, size_t &g)
{
    g = 0;
    for (int d = 0; d < NO_DIM; ++d)
    {
        // '% full[d]': the sub-box may WRAP a periodic axis (o[d]+m[d] > full[d]), which is the
        // normal case for a partition straddling the box seam. No-op for an unwrapped box.
        size_t gl = (c[d] + o[d]) % full[d];              // global index
        if (mainOrigin)
        {
            long ml = long(gl) - long(mainOrigin[d]);
            if (ml < 0) ml += long(full[d]);              // the window may wrap too
            if (ml < 0 || ml >= long(mainDims[d])) return false;
            g = g * mainDims[d] + size_t(ml);
        }
        else
            g = g * full[d] + gl;                         // global row-major flat
    }
    return true;
}

/* THE WORDS OF THE WORD MERGE ('--ps-merge fixed-point' in the single 3-D build: User_options::psWordMerge; fix 3
   S6). A pass hands its exact sums over UNCONVERTED, in the run-wide units (src/fixed_units.h RunUnits): per PS-DTFE
   accumulator its int64 words with their sparse carry and K (FxWords), indexed by the accumulator's RunUnitGrid
   slot (at: RU_PS_MASS, RU_PS_FX + PSFxGrid g -- MOM, M2, GRAD, MOMW, SV, DV, DW, SCAL, SG, RM --, RU_PS_FACE_S / N /
   P), and its int stream counts. The merge (src/ps_fixed_merge.h) adds a partition's into the full grid's -- integer
   adds, so the result does not depend on the split, the order or the device's arithmetic of the adds --, and ONE
   close-out (Quantities::psWordsClose, then normalizePhaseSpace) turns the full grid's into the Real grids: the
   merged words of a partitioned run, a one-domain run's own. EMPTY (on false, every vector empty) without the word
   merge: the struct then costs its own few hundred bytes per Quantities and nothing else. Which Real grid each
   accumulator becomes: psWordsClose.
   THE FACE SUMS (--face-gradient: RU_PS_FACE_S / N / P; fix 3 S7). A pass's words hold them PER CELL, both faces of
   every axis (comps: S.w[cell*2*NO_DIM*noVelComp + (side*NO_DIM + a)*noVelComp + j], N and P [cell*2*NO_DIM +
   side*NO_DIM + a], side 1 the upper face, 0 the lower), so a face inside the pass's sub-grid is held twice, by the
   cell below (its upper) and the cell above (its lower). The merged full grid's words -- and a one-domain run's after
   its close-out has routed them alike (ps_fixed_merge.h oneDomainFacesOnce) -- hold every face ONCE (faceOnce):
   each cell its NO_DIM UPPER faces (cellComps: S.w[(cell*NO_DIM + a)*noVelComp + j], N and P [cell*NO_DIM + a]);
   a cell's lower face is the upper face of the cell below it (wrapped where the grid is periodic), and on a grid
   that is not, the lower faces of layer 0 of axis a are that axis' BOUNDARY PLANE (plane[f][a], (cells /
   dims[a]) * facePer cells' components, indexed row-major over the other axes). The merge's rule:
   ps_fixed_merge.h addFaceSlot; the gradient reads them as words: Quantities::psFaceGradientCellWords. */
struct PSWordGrids
{
    static int const N = RU_PS_FACE_P - RU_PS_MASS + 1;   // the PS-DTFE accumulators, from RU_PS_MASS
    static int const NF = RU_PS_FACE_P - RU_PS_FACE_S + 1;   // the face sums among them (S, N, P)
    bool on = false;                // words handed over (else nothing below is in use)
    bool exact = false;             // the exact deposit: 'count' is the tet touches, SV the stream count
    bool dwIsMass = false;          // DW is the mass grid's sums, not handed over: the close-out copies the mass
    size_t samplesPerCell = 1;      // the pass's nSub^NO_DIM: the sampled stream count's divisor
    size_t cells = 0;               // the grid's cells (the pass's sub-grid, or the full grid)
    FxWords g[N];                   // [slot - RU_PS_MASS]: EMPTY w = this accumulator has no words
    std::vector<int> count;         // the int stream counts: the sampled in-simplex samples, or the exact tet touches
    std::vector<size_t> presized;   // the merge: each summed Real grid's cells before the loop (forEachField order)
    // the once layout of the face sums (the header comment): the merged full grid's, never a pass's
    bool faceOnce = false;          // the face slots hold each face once (else per cell, as a pass hands them over)
    bool periodic = false;          // faceOnce: the grid wraps on every axis (layer 0's lower face: the top layer's)
    size_t dims[NO_DIM] = {};       // faceOnce: the grid's cells per axis (row-major, the last axis fastest)
    FxWords plane[NF][NO_DIM];      // faceOnce, not periodic: [f][a] axis a's boundary plane of face sum f
    bool faceDirect[NF] = {};       // the close-out (psWordsClose): face sum f's unit is 2^-K with |K| <= 120 ...
    Real faceInv[NF] = {};          // ... and 2^-K then (psFaceGradientCellWords converts as psWordsToGrid)
    FxWords &at(int const slot) { return g[slot - RU_PS_MASS]; }
    FxWords const &at(int const slot) const { return g[slot - RU_PS_MASS]; }
    static bool isFace(int const slot) { return slot >= RU_PS_FACE_S && slot <= RU_PS_FACE_P; }
    // a face sum's components per face: S noVelComp, N and P 1
    static size_t facePer(int const slot) { return comps( slot ) / size_t( 2 * NO_DIM ); }
    // the components per cell of accumulator 'slot' in THIS grid's layout: comps, or for a face sum in the once
    // layout its upper faces only (half of comps). Every size or index of a merged full grid's words takes this.
    size_t cellComps(int const slot) const { return faceOnce && isFace( slot ) ? comps( slot ) / 2 : comps( slot ); }
    // the cells of axis a's boundary plane (the grid's layer 0 on a): cells / dims[a]
    size_t planeCells(int const a) const { return dims[a] > 0 ? cells / dims[a] : 0; }
    FxWords &boundary(int const slot, int const a) { return plane[slot - RU_PS_FACE_S][a]; }
    FxWords const &boundary(int const slot, int const a) const { return plane[slot - RU_PS_FACE_S][a]; }
    // the components per cell of accumulator 'slot' in a PASS's layout (the layout of its words: cell by cell, as its
    // Real grid; the face sums both faces of every axis)
    static size_t comps(int const slot)
    {
        switch ( slot )
        {
        case RU_PS_FX + 0: case RU_PS_FX + 5: return noVelComp;   // MOM, DV
        case RU_PS_FX + 1: return noDispComp;                     // M2
        case RU_PS_FX + 2: return noGradComp;                     // GRAD
        case RU_PS_FX + 7: return noScalarComp;                   // SCAL
        case RU_PS_FX + 8: return noScalarGradComp;               // SG
        case RU_PS_FACE_S: return size_t( 2 * NO_DIM ) * noVelComp;
        case RU_PS_FACE_N: case RU_PS_FACE_P: return size_t( 2 * NO_DIM );
        default: return 1;                                        // the mass, MOMW, SV, DW, RM
        }
    }
    // a name without spaces (the checkpoint's blobs: ps_fixed_merge.h)
    static char const *key(int const slot)
    {
        static char const *const k[N] = { "mass", "MOM", "M2", "GRAD", "MOMW", "SV", "DV", "DW", "SCAL", "SG", "RM",
                                          "faceS", "faceN", "faceP" };
        return slot >= RU_PS_MASS && slot <= RU_PS_FACE_P ? k[slot - RU_PS_MASS] : "?";
    }
    void release() { *this = PSWordGrids(); }   // (the vectors' storage freed)
};
#endif


// Container for every interpolated grid field; each vector is allocated only when its field is requested.
struct Quantities
{
    std::vector<Real>                         density;
    std::vector< Pvector<Real,noVelComp> >    velocity;
    std::vector< Pvector<Real,noGradComp> >   velocity_gradient;
    std::vector<Real>                         velocity_divergence;
    std::vector< Pvector<Real,noShearComp> >  velocity_shear;
    std::vector< Pvector<Real,noVortComp> >   velocity_vorticity;
    std::vector<Real>                         velocity_std;         // velocity standard deviation
    std::vector< Pvector<Real,noDispComp> >   velocity_dispersion;  // dispersion tensor sigma_ij (symmetric, upper-triangle row-major; trace = dispersion scalar)
    std::vector< Pvector<Real,noScalarComp> > scalar;
    std::vector< Pvector<Real,noScalarGradComp> > scalar_gradient;
    std::vector<Real>                         velocity_tweb;        // T-web classification label (0=void, 1=wall, 2=filament, 3=node)
    std::vector< Pvector<Real,NO_DIM> >       velocity_tweb_eigenvalues; // T-web eigenvalues (sorted descending)
    std::vector<Real>                         velocity_vweb;        // V-web classification label (0=void, 1=wall, 2=filament, 3=node)
    std::vector< Pvector<Real,NO_DIM> >       velocity_vweb_eigenvalues; // V-web eigenvalues (sorted descending)
#ifdef PHASE_SPACE
    std::vector<Real>                         stream_count;       // stream multiplicity at each grid point: the sampled deposit's sample-mean, or --exact's analytic volume-weighted (1/V_cell)sum(V_int). A FLOAT under both -- compare against integers with PS_STREAM_TOL, never with ==/> 1 exactly (see below)
    std::vector<Real>                         tet_touch;          // --exact only: raw INTEGER count of tetrahedra with a nonzero intersection with the cell (incl. corner grazes), written as '.tetTouch'. A geometric multiplicity, NOT a stream count -- 'stream_count' carries the volume-weighted (physical) multiplicity under every deposit.
    std::vector<Real>                         mass_weight;        // per-cell summed moment weight (mass shares, or the EULERIAN-VOLUME shares under --ps-volume-weighted); normalizes velocity/gradient/scalar. Internal to normalizePhaseSpace, never written out
    // --ps-volume-weighted + dispersion only: the dispersion stays MASS-weighted (sigma_ij is a
    // moment of the phase-space distribution f, so it is f- i.e. mass-weighted by definition),
    // while velocity/gradient go volume-weighted. sigma_ij = <v_i v_j>_m - <v_i>_m <v_j>_m then
    // needs its OWN mass-weighted mean and normalizer, since 'velocity' now holds the volume-
    // weighted one. Both are internal to normalizePhaseSpace and never written out.
    std::vector<Real>                         disp_weight;        // sum(m_s)
    std::vector< Pvector<Real,noVelComp> >    disp_velocity;      // sum(m_s v_s)
    std::vector<Real>                         caustic_bits;       // --ps-caustics: per-cell caustic mask, small exact ints in float (bit0 = a det(Ax)>0 stream overlaps the cell, bit1 = det<0, higher bits = the stratification of CGAL_triangulation/ps_caustic_class.h). OR-merged across partitions (addFrom/addFromSubgrid) and kept FULL right through post-processing: writeOutputData emits '.causticClass' from it and only then binarizes it in place to the 0/1 fold flag for '.caustic'. There is deliberately no second grid -- see the note there
    // --face-gradient (docs/cleanup-plan.md P10.3): each cell's FACE sums of the streams, its lower and its upper face
    // per axis (side 0 / 1): face_velocity[(side*NO_DIM + a)*3 + j] = the integral over that face of S_j, S = the sum of
    // the streams' velocities; face_coverage[side*NO_DIM + a] = the integral of N, their count (the covered area). A
    // face's two cells hold the same sums. normalizePhaseSpace makes the gradient of the face-averaged MEAN velocity
    // from them, (S_up/N_up - S_lo/N_lo) / dx: conservative (it telescopes: a periodic box's mean divergence is 0), the
    // stream average itself where the stream count is constant through the cell, exact for a planar flow. Never written.
    // (The word merge never makes these three: its merged face sums stay words, once per face, until the gradient:
    // psFaceWordsHeld below.)
    std::vector< Pvector<Real,2*NO_DIM*noVelComp> > face_velocity;
    std::vector< Pvector<Real,2*NO_DIM> >          face_coverage;
    // ... and the momentum through each face (P10.4): face_momentum[side*NO_DIM + a] = the integral of sum_s rho_s v_s,a
    // (rho/rho_bar times velocity, times area); its cell difference over the volume is the mass flux divergence
    std::vector< Pvector<Real,2*NO_DIM> >          face_momentum;
    Real psFaceDx[NO_DIM] = {};   // the grid spacing the gradient divides by (set before the normalization)
    // --face-gradient: div(sum_s rho_s v_s) per cell, rho/rho_bar times velocity per length: by continuity -a d(1+delta)/dt
    // (the rate a region gains or loses mass), written as '.a_massFluxDiv'
    std::vector<Real>                         mass_flux_divergence;
    // --face-gradient: the divergence of the stream average of each stream's own gradient (the deposit's regular part,
    // what 'velocity_gradient' held before the mean flow's replaced it), written as '.a_velDivStreams'
    std::vector<Real>                         velocity_divergence_streams;
    // ... and that regular part's own gradient, shear and vorticity (docs/cleanup-plan.md D12), each made only where
    // the run writes its mean-flow twin ('gradient_a', 'shear_a', 'vorticity_a': User_options::psStreamsTwins), by the
    // derivations web_fields.cc applies to a stored gradient -- so their bytes are what a run without the face
    // gradient stores as '.a_velGrad', '.a_velShear', '.a_velVort'. Written as '.a_velGradStreams',
    // '.a_velShearStreams', '.a_velVortStreams'
    std::vector< Pvector<Real,noGradComp> >   velocity_gradient_streams;
    std::vector< Pvector<Real,noShearComp> >  velocity_shear_streams;
    std::vector< Pvector<Real,noVortComp> >   velocity_vorticity_streams;
    bool psStreamsTwinsOn = false;            // psFaceGradientBegin made one of them
    void psStreamsTwinsCell(size_t const i, Pvector<Real,noGradComp> g);   // quantities.cc
    // --face-gradient, cell i once its velocity_gradient is normalized: keep the regular part's divergence, then make
    // the gradient the face-averaged mean flow's along every axis both of whose faces are covered. A no-op without the
    // face sums.
    inline void psFaceGradientCell(size_t const i)
    {
        if ( psWords.faceOnce ) { psFaceGradientCellWords( i ); return; }   // the word merge: its words (below)
        if ( face_velocity.empty() ) return;
        if ( not face_momentum.empty() )
        {   // the mass flux: (the momentum out through the upper faces - in through the lower) / the cell volume
            Real vol = Real(1.), flux = Real(0.);
            for (int a = 0; a < NO_DIM; ++a)
            {
                vol *= psFaceDx[a];
                flux += face_momentum[i][NO_DIM + a] - face_momentum[i][a];
            }
            mass_flux_divergence[i] = vol > Real(0.) ? flux / vol : Real(0.);
        }
        Pvector<Real,noGradComp> &g = velocity_gradient[i];
        velocity_divergence_streams[i] = g[0] + g[NO_DIM+1] + g[2*NO_DIM+2];   // [j*NO_DIM+i]: the trace
        if ( psStreamsTwinsOn ) psStreamsTwinsCell( i, g );   // (a copy: the mean flow's overwrites g below)
        for (int a = 0; a < NO_DIM; ++a)
        {
            Real const nLo = face_coverage[i][a], nUp = face_coverage[i][NO_DIM + a];
            if ( !( nLo > Real(0.) && nUp > Real(0.) ) || !( psFaceDx[a] > Real(0.) ) ) continue;
            for (size_t j = 0; j < noVelComp; ++j)
                g[j*NO_DIM + size_t(a)] = ( face_velocity[i][size_t(NO_DIM + a)*noVelComp + j] / nUp
                                            - face_velocity[i][size_t(a)*noVelComp + j] / nLo ) / psFaceDx[a];
        }
    }
    // before / after the cells: the regular part's divergence grid sized -- and its gradient, shear and vorticity as
    // 'twins' asks (bits 0, 1, 2: User_options::psStreamsTwins) --, the face sums released (the word merge's words
    // too: psWords)
    inline void psFaceGradientBegin(size_t const n, unsigned const twins = 0u)
    {
        if ( face_velocity.empty() and not psFaceWordsHeld() ) return;
        velocity_divergence_streams.assign( n, Real(0.) );
        if ( not face_momentum.empty() or ( psFaceWordsHeld() and not psWords.at( RU_PS_FACE_P ).w.empty() ) )
            mass_flux_divergence.assign( n, Real(0.) );
        if ( twins & 1u ) velocity_gradient_streams.assign( n, Pvector<Real,noGradComp>::zero() );
        if ( twins & 2u ) velocity_shear_streams.assign( n, Pvector<Real,noShearComp>::zero() );
        if ( twins & 4u ) velocity_vorticity_streams.assign( n, Pvector<Real,noVortComp>::zero() );
        psStreamsTwinsOn = ( twins & 7u ) != 0u;
    }
    inline void psFaceGradientEnd()
    {
        std::vector< Pvector<Real,2*NO_DIM*noVelComp> >().swap( face_velocity );
        std::vector< Pvector<Real,2*NO_DIM> >().swap( face_coverage );
        std::vector< Pvector<Real,2*NO_DIM> >().swap( face_momentum );
        psFaceWordsRelease();
    }
    // the word merge (fix 3 S7): the merged face sums stay words, once per face (PSWordGrids' header comment), from
    // the close-out's first half (psWordsClose, which converts every other accumulator and frees it) until the
    // gradient is made from them -- held: both S and N have words (P, the mass flux, may be absent)
    bool psFaceWordsHeld() const
    {
        return psWords.faceOnce and not psWords.at( RU_PS_FACE_S ).w.empty()
               and not psWords.at( RU_PS_FACE_N ).w.empty();
    }
    // ... cell i's gradient and mass flux from them (quantities.cc): psFaceGradientCell's arithmetic, each face sum
    // converted as psWordsToGrid converts, a cell's lower faces read as the cell below's upper ones (or the boundary
    // plane's)
    void psFaceGradientCellWords(size_t const i);
    // ... and their release (psFaceGradientEnd, and normalizePhaseSpace on every way out: a side whose gradient is
    // not made keeps no words); a no-op without them
    void psFaceWordsRelease() { if ( psWords.faceOnce ) psWords.release(); }
    std::vector<Real>                         hidden_streams;     // PS-DTFE: a small bitmask in Real, written as '.hidden_streams'. BIT 1 (value 1) = the cell holds multi-stream volume its '.streams' does not show: a negatively oriented (folded) tetrahedron overlaps it with positive volume while '.streams' reads single-stream -- a fold thinner than the sub-sample spacing. Exact in a periodic box (degree-1 map: a point is multi-stream iff a flipped tet covers it), so the single-stream mask is '.streams == 1 (to PS_STREAM_TOL) and not (hidden_streams & 1)': every point of such a cell lies on one stream. BIT 2 (value 2) = the cell received MASS from a tetrahedron none of whose sub-samples lies in it (the centroid fallback, or a --ps-halo-release tet): the cell mean includes a sub-resolution piece, possibly of the SAME stream -- a quadrature note, not a stream count (non-periodic runs also set bit 1 there, conservatively). Legacy '.unresolved' files hold only 0/1 (the mass flag), which reads as bit 1 and so stays conservative. OR-merged like caustic_bits; finalizeHiddenStreams() clears bit 1 where '.streams' already reads multi-stream
    // PS-DTFE partition sub-grid in global grid-cell units, so addFromSubgrid() can map cells back.
    size_t ps_subOrigin[NO_DIM] = {0};
    size_t ps_subDims[NO_DIM]   = {0};   // [0]==0 means "not a sub-grid" (vectors span the full grid)
    // the word merge (PSWordGrids above): the pass's words handed over unconverted, or the full grid's merged ones
    // until the close-out (their face sums until the gradient: psFaceWordsHeld). Outside the field list on purpose:
    // no checkpoint, spill or Real merge walks it (the merge keeps its own words in the checkpoint: ps_fixed_merge.h)
    PSWordGrids psWords;
    // THE CLOSE-OUT's first half (quantities.cc): the words to their Real grids, each component rounded once, the
    // density to rho/rho_bar (times invRhoBar / cellVolume, the pass's expression), the stream counts; a no-op
    // without words. The face sums stay words, once per face (psFaceWordsHeld). normalizePhaseSpace follows
    // (DTFE.cpp psCloseOutMoments).
    void psWordsClose(Real const invRhoBar, Real const cellVolume);
#endif

    /* THE FIELD LIST. forEachField(f) calls f(info, merge, member) for every grid above, in this order, which is also
       the checkpoint's field order (partition_checkpoint.h forEachAccumulator), the merge spill's
       (ps_merge_spill.h) and the fixed-point merge's slots (ps_fixed_merge.h forEachSummed). Everything that walks
       the grids goes through it: size, copyFromSubgrid, addFrom, addFromSubgrid and reserveMemory (quantities.cc),
       and those two lists. A new grid is added HERE, with its tags; besides, only what is particular to it: its
       output file (input_output.cc writeOutputData) and, when it is made from other grids, its derivation
       (web_fields.cc).
         info.name            the stable name (the checkpoint's)
         info.flag            the Field flag that asks for it (nullptr: a PS-DTFE accumulator the options size)
         info.sizing          which grids reserveMemory sizes before a partition loop (FieldSizing)
         info.scale           a summed grid's natural per-cell scale (FieldScale: the fixed-point merge's unit)
         info.madeAfterMerge  summed, but no PS-DTFE partition fills it: made after the merge from the others
         merge                MergeTag<FIELD_SUM> ('+=' across partitions), <FIELD_OR> (a bit mask: OR) or
                              <FIELD_DERIVED> (never merged: the web labels, made after the merge) -- a type, so a
                              merge can pick its code with 'if constexpr'
         member               &Quantities::<grid>, so one walk can visit the same grid of two Quantities */
    enum FieldMerge { FIELD_SUM, FIELD_OR, FIELD_DERIVED };
    template <int M> using MergeTag = std::integral_constant<int, M>;
    enum FieldSizing
    {
        SIZE_FIELD,         // when its Field flag is set
        SIZE_VELOCITY,      // the velocity: its flag, or the dispersion's (which needs <v> too)
        SIZE_PS_ANY,        // PS-DTFE: whenever any field of that side is selected
        SIZE_PS_CAUSTICS,   // PS-DTFE: '--ps-caustics', when any field of that side is selected
        SIZE_PS_EXACT,      // PS-DTFE: '--exact', when any field of that side is selected
        SIZE_AT_MERGE       // never pre-sized: the merge grows it at a partition's first values (the moment weights)
    };
    enum FieldScale { SCALE_ONE = 0, SCALE_WEIGHT, SCALE_MASS, SCALE_VEL_W, SCALE_VEL_M, SCALE_VEL2_M, SCALE_GRAD_W,
                      SCALE_SCAL_W, SCALE_SGRAD_W };   // the table in ps_fixed_merge.h's header comment
    struct FieldInfo
    {
        char const *name;
        bool Field::*flag;
        int sizing;
        int scale;
        bool madeAfterMerge;
    };
    template <class F>
    static void forEachField(F &&f)
    {
        using Q = Quantities;
        MergeTag<FIELD_SUM> const sum{};
        MergeTag<FIELD_OR> const orMask{};
        MergeTag<FIELD_DERIVED> const derived{};
        f( FieldInfo{ "density", &Field::density, SIZE_FIELD, SCALE_ONE, false }, sum, &Q::density );
        f( FieldInfo{ "velocity", &Field::velocity, SIZE_VELOCITY, SCALE_VEL_W, false }, sum, &Q::velocity );
        f( FieldInfo{ "velocity_gradient", &Field::velocity_gradient, SIZE_FIELD, SCALE_GRAD_W, false }, sum, &Q::velocity_gradient );
        f( FieldInfo{ "velocity_divergence", &Field::velocity_divergence, SIZE_FIELD, SCALE_GRAD_W, true }, sum, &Q::velocity_divergence );
        f( FieldInfo{ "velocity_shear", &Field::velocity_shear, SIZE_FIELD, SCALE_GRAD_W, true }, sum, &Q::velocity_shear );
        f( FieldInfo{ "velocity_vorticity", &Field::velocity_vorticity, SIZE_FIELD, SCALE_GRAD_W, true }, sum, &Q::velocity_vorticity );
        f( FieldInfo{ "velocity_std", &Field::velocity_std, SIZE_FIELD, SCALE_VEL_W, true }, sum, &Q::velocity_std );
        f( FieldInfo{ "velocity_dispersion", &Field::velocity_dispersion, SIZE_FIELD, SCALE_VEL2_M, false }, sum, &Q::velocity_dispersion );
        f( FieldInfo{ "scalar", &Field::scalar, SIZE_FIELD, SCALE_SCAL_W, false }, sum, &Q::scalar );
        f( FieldInfo{ "scalar_gradient", &Field::scalar_gradient, SIZE_FIELD, SCALE_SGRAD_W, false }, sum, &Q::scalar_gradient );
        f( FieldInfo{ "velocity_tweb", &Field::velocity_tweb, SIZE_FIELD, SCALE_ONE, false }, derived, &Q::velocity_tweb );
        f( FieldInfo{ "velocity_tweb_eigenvalues", &Field::velocity_tweb, SIZE_FIELD, SCALE_ONE, true }, sum, &Q::velocity_tweb_eigenvalues );
        f( FieldInfo{ "velocity_vweb", &Field::velocity_vweb, SIZE_FIELD, SCALE_ONE, false }, derived, &Q::velocity_vweb );
        f( FieldInfo{ "velocity_vweb_eigenvalues", &Field::velocity_vweb, SIZE_FIELD, SCALE_GRAD_W, true }, sum, &Q::velocity_vweb_eigenvalues );
#ifdef PHASE_SPACE
        f( FieldInfo{ "stream_count", nullptr, SIZE_PS_ANY, SCALE_ONE, false }, sum, &Q::stream_count );
        f( FieldInfo{ "tet_touch", nullptr, SIZE_PS_EXACT, SCALE_ONE, false }, sum, &Q::tet_touch );
        f( FieldInfo{ "mass_weight", nullptr, SIZE_AT_MERGE, SCALE_WEIGHT, false }, sum, &Q::mass_weight );
        f( FieldInfo{ "disp_weight", nullptr, SIZE_AT_MERGE, SCALE_MASS, false }, sum, &Q::disp_weight );
        f( FieldInfo{ "disp_velocity", nullptr, SIZE_AT_MERGE, SCALE_VEL_M, false }, sum, &Q::disp_velocity );
        // (the face sums: S's scale is the gradient's -- vmax * dx^2 under '--ps-volume-weighted', which --face-gradient
        //  requires -- N's the weight's, a cell volume: a covered face area over dx is a few of them)
        f( FieldInfo{ "face_velocity", nullptr, SIZE_AT_MERGE, SCALE_GRAD_W, false }, sum, &Q::face_velocity );
        f( FieldInfo{ "face_coverage", nullptr, SIZE_AT_MERGE, SCALE_WEIGHT, false }, sum, &Q::face_coverage );
        f( FieldInfo{ "face_momentum", nullptr, SIZE_AT_MERGE, SCALE_GRAD_W, false }, sum, &Q::face_momentum );
        f( FieldInfo{ "caustic_bits", nullptr, SIZE_PS_CAUSTICS, SCALE_ONE, false }, orMask, &Q::caustic_bits );
        f( FieldInfo{ "hidden_streams", nullptr, SIZE_PS_ANY, SCALE_ONE, false }, orMask, &Q::hidden_streams );
#else
        (void)orMask;
#endif
    }

    void copyFromSubgrid(Quantities const &subgridResults,
                        Field const &field,
                        std::vector<size_t> const &mainGrid,
                        std::vector<size_t> const &subgrid,
                        std::vector<size_t> const &subgridOffset); // copy subgrid ('partition') results into the full grid
    size_t size() const;	// size of any non-empty member vector
    // sizes and zero-fills the main grids of a partition loop ('partition' option) that 'field' asks for (sizing
    // SIZE_FIELD / SIZE_VELOCITY); with psRun (the PS-DTFE loop's options) also the PS accumulators those options
    // ask for (SIZE_PS_*), so the memory gate sees them before the loop starts
    void reserveMemory(size_t *gridSize, Field &field, User_options const *psRun = nullptr);
    void addFrom(Quantities const &other); // element-wise accumulate from another Quantities (PS-DTFE partitioning)
#ifdef PHASE_SPACE
    // Turn summed density-weighted moments into mass-weighted means (call once, after all
    // partitions added). When the partitions aliased the weight to the density grid (density
    // field selected -> no mass_weight was filled), pass weightFromDensityScale =
    // cellVolume * averageDensity so the per-cell mass is reconstructed from the summed
    // (rho/rho_bar) density; 0 normalizes by the mass_weight grid only.
    // streamsTwins: the regular part's twins of a face pass (psFaceGradientBegin's 'twins')
    void normalizePhaseSpace(Field const &field, Real const weightFromDensityScale = Real(0.),
                             unsigned const streamsTwins = 0u);
    void addFromSubgrid(Quantities const &other, size_t const *fullGrid, size_t const *mainOrigin = nullptr, size_t const *mainDims = nullptr);   // mainOrigin/mainDims: this grid is itself a window of the full grid (--grid-window) // like addFrom but 'other' stores only its Eulerian box; maps each cell into the full grid (dims fullGrid)
    void finalizeHiddenStreams();   // after the merge: clear '.hidden_streams' bit 1 where '.streams' reads multi-stream (idempotent)
#endif
};


#endif
