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
// Tolerance for testing the stream multiplicity ('stream_count', '.streams') against an integer: it is a FLOAT sum under
// every deposit, so an exact '== 1' / '> 1' test is wrong. Mirrors dtfelib.STREAM_TOL on the Python side: keep in sync.
Real const PS_STREAM_TOL = Real(1.e-3);

// A partition sub-grid cell (local coords c; sub-grid dims m, global origin o) -> its index in the full grid 'full'
// (used by quantities.cc SubgridRowMap/mergeSubgridCells). Each cell maps to its global row-major index; o, m, full
// are in grid cells.
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
    std::vector<Real>                         stream_count;       // stream multiplicity: the sampled deposit's sample-mean, or --exact's volume-weighted value; a FLOAT: compare with PS_STREAM_TOL, never ==/> 1
    std::vector<Real>                         tet_touch;          // --exact only: raw INTEGER count of tetrahedra touching the cell ('.tetTouch'); a geometric multiplicity, not a stream count
    std::vector<Real>                         mass_weight;        // per-cell summed moment weight (the Eulerian-volume shares); normalizes velocity/gradient/scalar; internal, never written
    // dispersion only: it stays MASS-weighted (sigma_ij is a moment of f) while velocity/gradient are volume-weighted,
    // so it needs its own mass-weighted mean and normalizer; both are internal to normalizePhaseSpace.
    std::vector<Real>                         disp_weight;        // sum(m_s)
    std::vector< Pvector<Real,noVelComp> >    disp_velocity;      // sum(m_s v_s)
    std::vector<Real>                         caustic_bits;       // --ps-caustics: per-cell caustic mask, small exact ints in float (bit0 = a det(Ax)>0 stream overlaps the cell, bit1 = det<0, higher bits = ps_caustic_class.h stratification); OR-merged, kept FULL until writeOutputData emits '.causticClass', then binarized in place for '.caustic'
    // --face-gradient: each cell's FACE sums of the streams, its lower and upper face per axis (side 0 / 1):
    // face_velocity[(side*NO_DIM + a)*3 + j] = the integral over that face of S_j (S = the sum of the streams'
    // velocities); face_coverage[side*NO_DIM + a] = the integral of N, their count. A face's two cells hold the same
    // sums. normalizePhaseSpace makes the gradient of the face-averaged mean velocity from them, (S_up/N_up -
    // S_lo/N_lo) / dx: conservative, exact for a planar flow. Never written.
    std::vector< Pvector<Real,2*NO_DIM*noVelComp> > face_velocity;
    std::vector< Pvector<Real,2*NO_DIM> >          face_coverage;
    // ... and the momentum through each face: face_momentum[side*NO_DIM + a] = the integral of sum_s rho_s v_s,a
    // (rho/rho_bar times velocity, times area); its cell difference over the volume is the mass flux divergence
    std::vector< Pvector<Real,2*NO_DIM> >          face_momentum;
    Real psFaceDx[NO_DIM] = {};   // the grid spacing the gradient divides by (set before the normalization)
    // --face-gradient: div(sum_s rho_s v_s) per cell, rho/rho_bar times velocity per length: by continuity -a d(1+delta)/dt
    // (the rate a region gains or loses mass), written as '.a_massFluxDiv'
    std::vector<Real>                         mass_flux_divergence;
    // --face-gradient: the divergence of the stream average of each stream's own gradient (the deposit's regular part),
    // written as '.a_velDivStreams'
    std::vector<Real>                         velocity_divergence_streams;
    // ... and that regular part's own gradient, shear and vorticity, each made only where
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
    // 'twins' asks (bits 0, 1, 2: User_options::psStreamsTwins) --, the face sums released
    inline void psFaceGradientBegin(size_t const n, unsigned const twins = 0u)
    {
        if ( face_velocity.empty() ) return;
        velocity_divergence_streams.assign( n, Real(0.) );
        if ( not face_momentum.empty() )
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
    }
    // BIT 1 = multi-stream volume that '.streams' does not show (a folded tetrahedron overlaps the cell while '.streams' reads
    // single-stream; exact in a periodic box). BIT 2 = the cell got MASS from a tetrahedron with no sub-sample in it
    // (centroid fallback or --ps-halo-release): a quadrature note, not a stream count. Legacy '.unresolved' 0/1 reads
    // as bit 1. finalizeHiddenStreams() clears bit 1 where '.streams' already reads multi-stream.
    std::vector<Real>                         hidden_streams;     // PS-DTFE bitmask in Real, written as '.hidden_streams' (bits above); OR-merged
    // PS-DTFE partition sub-grid in global grid-cell units, so addFromSubgrid() can map cells back.
    size_t ps_subOrigin[NO_DIM] = {0};
    size_t ps_subDims[NO_DIM]   = {0};   // [0]==0 means "not a sub-grid" (vectors span the full grid)
#endif

    /* THE FIELD LIST. forEachField(f) calls f(info, merge, member) for every grid above, in this order, which is also the
       checkpoint's field order and the merge spill's. Everything that walks the grids goes through it; a new grid is added HERE,
       plus its output file (input_output.cc writeOutputData) and, if derived, web_fields.cc. info.name: stable name; info.flag:
       the Field flag asking for it (nullptr: a PS accumulator the options size); info.sizing: FieldSizing tag; merge:
       MergeTag<FIELD_SUM | FIELD_OR | FIELD_DERIVED> (for 'if constexpr'); member: &Quantities::<grid>. */
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
    struct FieldInfo
    {
        char const *name;
        bool Field::*flag;
        int sizing;
    };
    template <class F>
    static void forEachField(F &&f)
    {
        using Q = Quantities;
        MergeTag<FIELD_SUM> const sum{};
        MergeTag<FIELD_OR> const orMask{};
        MergeTag<FIELD_DERIVED> const derived{};
        f( FieldInfo{ "density", &Field::density, SIZE_FIELD }, sum, &Q::density );
        f( FieldInfo{ "velocity", &Field::velocity, SIZE_VELOCITY }, sum, &Q::velocity );
        f( FieldInfo{ "velocity_gradient", &Field::velocity_gradient, SIZE_FIELD }, sum, &Q::velocity_gradient );
        f( FieldInfo{ "velocity_divergence", &Field::velocity_divergence, SIZE_FIELD }, sum, &Q::velocity_divergence );
        f( FieldInfo{ "velocity_shear", &Field::velocity_shear, SIZE_FIELD }, sum, &Q::velocity_shear );
        f( FieldInfo{ "velocity_vorticity", &Field::velocity_vorticity, SIZE_FIELD }, sum, &Q::velocity_vorticity );
        f( FieldInfo{ "velocity_std", &Field::velocity_std, SIZE_FIELD }, sum, &Q::velocity_std );
        f( FieldInfo{ "velocity_dispersion", &Field::velocity_dispersion, SIZE_FIELD }, sum, &Q::velocity_dispersion );
        f( FieldInfo{ "scalar", &Field::scalar, SIZE_FIELD }, sum, &Q::scalar );
        f( FieldInfo{ "scalar_gradient", &Field::scalar_gradient, SIZE_FIELD }, sum, &Q::scalar_gradient );
        f( FieldInfo{ "velocity_tweb", &Field::velocity_tweb, SIZE_FIELD }, derived, &Q::velocity_tweb );
        f( FieldInfo{ "velocity_tweb_eigenvalues", &Field::velocity_tweb, SIZE_FIELD }, sum, &Q::velocity_tweb_eigenvalues );
        f( FieldInfo{ "velocity_vweb", &Field::velocity_vweb, SIZE_FIELD }, derived, &Q::velocity_vweb );
        f( FieldInfo{ "velocity_vweb_eigenvalues", &Field::velocity_vweb, SIZE_FIELD }, sum, &Q::velocity_vweb_eigenvalues );
#ifdef PHASE_SPACE
        f( FieldInfo{ "stream_count", nullptr, SIZE_PS_ANY }, sum, &Q::stream_count );
        f( FieldInfo{ "tet_touch", nullptr, SIZE_PS_EXACT }, sum, &Q::tet_touch );
        f( FieldInfo{ "mass_weight", nullptr, SIZE_AT_MERGE }, sum, &Q::mass_weight );
        f( FieldInfo{ "disp_weight", nullptr, SIZE_AT_MERGE }, sum, &Q::disp_weight );
        f( FieldInfo{ "disp_velocity", nullptr, SIZE_AT_MERGE }, sum, &Q::disp_velocity );
        // (the face sums: S's scale is the gradient's -- vmax * dx^2 with the volume-weighted moments of every PS-DTFE
        //  run -- N's the weight's, a cell volume: a covered face area over dx is a few of them)
        f( FieldInfo{ "face_velocity", nullptr, SIZE_AT_MERGE }, sum, &Q::face_velocity );
        f( FieldInfo{ "face_coverage", nullptr, SIZE_AT_MERGE }, sum, &Q::face_coverage );
        f( FieldInfo{ "face_momentum", nullptr, SIZE_AT_MERGE }, sum, &Q::face_momentum );
        f( FieldInfo{ "caustic_bits", nullptr, SIZE_PS_CAUSTICS }, orMask, &Q::caustic_bits );
        f( FieldInfo{ "hidden_streams", nullptr, SIZE_PS_ANY }, orMask, &Q::hidden_streams );
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
    // Turn the summed moments into means (once, after all partitions): the velocity family by its moment weight, the
    // dispersion by its own mass normalizer where disp_weight holds it. If the partitions aliased the weight to the
    // density grid (no mass_weight filled), pass weightFromDensityScale = cellVolume * averageDensity to rebuild the
    // per-cell mass; 0 normalizes by the mass_weight grid only. streamsTwins: the regular part's twins of a face pass.
    void normalizePhaseSpace(Field const &field, Real const weightFromDensityScale = Real(0.),
                             unsigned const streamsTwins = 0u);
    void addFromSubgrid(Quantities const &other, size_t const *fullGrid, size_t const *mainOrigin = nullptr, size_t const *mainDims = nullptr);   // mainOrigin/mainDims: this grid is itself a window of the full grid (--grid-window); maps each cell of 'other' (its Eulerian box only) into the full grid
    void finalizeHiddenStreams();   // after the merge: clear '.hidden_streams' bit 1 where '.streams' reads multi-stream (idempotent)
#endif
};


#endif
