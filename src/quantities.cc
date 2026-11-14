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


/* Quantities member functions: copy/accumulate field results across partitions and (PS-DTFE) turn
   summed density-weighted moments into mass-weighted means. */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "quantities.h"
#include "scratch_alloc.h"
#include "message.h"
#ifdef PHASE_SPACE
#include "thread_count.h"   // poolRanges: the parallel merge and normalization
#include "web_fields.h"     // velocityShear / velocityVorticity: the regular part's twins, the stored grids' arithmetic
#endif




// Copies one Quantities field from the subgrid results into the matching block of the main grid.
template<typename T>
void copyField(T const &subgridResults,
                T *mainGridResults,
                bool toCopy,
                std::vector<size_t> const &mainGrid,
                std::vector<size_t> const &subgrid,
                std::vector<size_t> const &subgridOffset)
{
    if ( not toCopy )
        return;
    
    size_t start[NO_DIM], end[NO_DIM];
    for (size_t i=0; i<NO_DIM; ++i)
    {
        start[i] = subgridOffset[i];
        end[i] = subgridOffset[i] + subgrid[i];
    }
    
    
    size_t totalSubgrid = 1;
    for (size_t d=0; d<NO_DIM; ++d) totalSubgrid *= subgrid[d];
    // a selected field that the interpolation did not compute leaves an empty grid, which the copy would read past
    if ( subgridResults.size() != totalSubgrid )
        throwError( "A selected field holds ", subgridResults.size(), " values for a sub-grid of ", totalSubgrid,
                    " cells: the chosen interpolation method does not compute it (see '--field' in the help)." );

    for (size_t flatIdx=0; flatIdx<totalSubgrid; ++flatIdx)
    {
        size_t subIdx[NO_DIM], rem = flatIdx;
        for (int d=NO_DIM-1; d>=0; --d)
        {
            subIdx[d] = rem % subgrid[d];
            rem /= subgrid[d];
        }
        size_t indexMain = 0, indexSub = 0;
        for (int d=0; d<NO_DIM; ++d)
        {
            if (d > 0) { indexMain *= mainGrid[d]; indexSub *= subgrid[d]; }
            indexMain += subIdx[d] + start[d];
            indexSub += subIdx[d];
        }
        (*mainGridResults)[indexMain] = subgridResults[indexSub];
    }
}

// Copies '--partition' subgrid results into the full-grid results ('subgrid', 'subgridOffset' from subpartition.h).
void Quantities::copyFromSubgrid(Quantities const &subgridResults,
                                 Field const &field,
                                 std::vector<size_t> const &mainGrid,
                                 std::vector<size_t> const &subgrid,
                                 std::vector<size_t> const &subgridOffset)
{
    // every grid a Field flag asks for (the PS-DTFE accumulators have none: a standard partition holds none of them)
    forEachField( [&](FieldInfo const &fi, auto, auto member)
    {
        if ( fi.flag )
            copyField( subgridResults.*member, &(this->*member), field.*(fi.flag), mainGrid, subgrid, subgridOffset );
    } );
}



// Errors out if 'object' is non-empty and its size differs from the running expected size; updates *expectedSize.
template< typename T>
void fieldSize(T const & object,
                size_t *expectedSize)
{
    if( not object.empty() )
    {
        if ( (*expectedSize)!=0 and (*expectedSize)!=object.size() )
            throwError( "Two or more objects of class 'Quantities' have different sizes. All objects in this class should be empty or have the same size." );
        else
            (*expectedSize) = object.size();
    }
}

// Returns the size of any non-empty member (all non-empty members share the same size).
size_t Quantities::size() const
{
    size_t temp = 0;
    forEachField( [&](FieldInfo const &, auto, auto member) { fieldSize( this->*member, &temp ); } );
    return temp;
}


// Element-wise adds 'src' into 'dst'; used to sum PS-DTFE partition results across partitions.
template<typename T>
void addField(std::vector<T> const &src, std::vector<T> *dst)
{
    if (src.empty()) return;
    // Grow an unsized dst (zero-filled) so accumulation starts from zero.
    if (dst->size() < src.size())
        ScratchAlloc::zeroResize(*dst, src.size(), T());   // (scratch: an empty dst's zeros write nothing)
    for (size_t i = 0; i < src.size(); ++i)
        (*dst)[i] += src[i];
}

// Bitwise-OR accumulate for bit-mask fields stored as Real (small exact integers, e.g. the
// caustic orientation bits). OR is commutative AND idempotent, so -- unlike '+=' -- the merge
// is invariant under the partitions' completion order and under a stream being seen by the
// deposit any number of times.
void orField(std::vector<Real> const &src, std::vector<Real> *dst)
{
    if (src.empty()) return;
    if (dst->size() < src.size())
        ScratchAlloc::zeroResize(*dst, src.size(), Real(0.));
    for (size_t i = 0; i < src.size(); ++i)
        (*dst)[i] = Real( int((*dst)[i]) | int(src[i]) );
}

// Accumulates every field of 'other' into this object (PS-DTFE: one partition's results summed into the whole): the
// summed grids '+=', the bit masks OR (never '+='); the web labels are made after the merge, only their eigenvalues sum.
void Quantities::addFrom(Quantities const &other)
{
    forEachField( [&](FieldInfo const &, auto merge, auto member)
    {
        if constexpr ( decltype( merge )::value == FIELD_SUM )
            addField( other.*member, &(this->*member) );
        else if constexpr ( decltype( merge )::value == FIELD_OR )
            orField( other.*member, &(this->*member) );
    } );
}

#ifdef PHASE_SPACE
// '.hidden_streams' bit 1 is marked per partition from the geometry alone (a flipped tet overlaps the
// cell); whether the SAMPLES missed it is only known once every partition's samples are summed
// into stream_count. Where '.streams' already reads multi-stream the samples saw it, so the bit
// is cleared: bit 1 then means exactly "multi-stream volume that '.streams' does not show".
void Quantities::finalizeHiddenStreams()
{
    if ( hidden_streams.empty() || stream_count.size() != hidden_streams.size() ) return;
    for (size_t i = 0; i < hidden_streams.size(); ++i)
    {
        int const bits = int(hidden_streams[i]);
        if ( (bits & 1) && stream_count[i] > Real(1.) + PS_STREAM_TOL )
            hidden_streams[i] = Real(bits & ~1);
    }
}
#endif


#ifdef PHASE_SPACE
// PS-DTFE partition path: partitions accumulate density-weighted moments sum(rho_s f_s) and per-cell
// mass sum(rho_s); divide once here after all partitions are summed, since averaging is non-linear.
// The weight is either the explicit mass_weight grid, or -- when the density field was selected and
// the partitions aliased the weight to it -- reconstructed as density * weightFromDensityScale
// (= density * cellVolume * averageDensity, undoing the rho/rho_bar conversion; saves 4 B/cell).
void Quantities::normalizePhaseSpace(Field const &field, Real const weightFromDensityScale,
                                     unsigned const streamsTwins)
{
    bool const fromDensity = this->mass_weight.empty();
    if ( fromDensity and (weightFromDensityScale <= Real(0.) or this->density.empty()) )
    {
        psFaceWordsRelease();   // (the word merge's face sums, if any: no gradient is made from them)
        return;   // no mass-weighted field was deferred (or nothing to reconstruct it from)
    }
    size_t const n = fromDensity ? this->density.size() : this->mass_weight.size();
    // velocity holds sum(rho v). A dispersion-only VOLUME-WEIGHTED run never allocates it --
    // the dispersion carries its own mean in disp_velocity (nonempty disp_weight is that
    // configuration's signature), so infer the gate from the data actually present.
    bool const haveVel = field.velocity
                         || (field.velocity_dispersion and this->disp_weight.empty());
    if ( haveVel and this->velocity.empty() )
    {
        psFaceWordsRelease();
        return;   // no weighted moments were accumulated (e.g. density-only field selection)
    }
    // --face-gradient: the gradient of the mean flow from the face sums (psFaceGradientCell), with the volume
    // weights; under the word merge from its face words, once per face (psFaceWordsHeld: no Real face grids there)
    bool const face = ( not this->face_velocity.empty() or psFaceWordsHeld() ) and field.velocity_gradient;
    if ( face ) psFaceGradientBegin( n, streamsTwins );
    // every cell on its own (the same arithmetic per cell): split over the free cores (thread_count.h poolRanges)
    poolRanges( n, size_t(1) << 16, CorePool::total(), [&](size_t const lo, size_t const hi)
    {
    for (size_t i = lo; i < hi; ++i)
    {
        Real const w = fromDensity ? this->density[i] * weightFromDensityScale : this->mass_weight[i];
        if ( w <= Real(0.) ) continue;
        Real const inv = Real(1.) / w;
        if ( haveVel )                 this->velocity[i]          *= inv;   // <v> = sum(rho v)/sum(rho)
        if ( field.velocity_gradient ) this->velocity_gradient[i] *= inv;
        if ( face )                    psFaceGradientCell( i );
        if ( field.scalar )            this->scalar[i]            *= inv;
        if ( field.scalar_gradient )   this->scalar_gradient[i]   *= inv;
        if ( field.velocity_dispersion and this->disp_weight.empty() )
        {
            // sigma_ij = <v_i v_j> - <v_i><v_j>, using the now-normalized <v>. Same weighting
            // for both moments here, so 'velocity' IS the right mean.
            Pvector<Real,noVelComp> const &vbar = this->velocity[i];
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                {
                    Real s = this->velocity_dispersion[i][c] * inv - vbar[a]*vbar[b];
                    if ( a == b and s < Real(0.) ) s = Real(0.);   // variance: clamp FP-noise negatives
                    this->velocity_dispersion[i][c] = s;
                    ++c;
                }
        }
    }
    } );

    // --ps-volume-weighted + dispersion: the dispersion carries MASS-weighted moments while
    // 'velocity' above was normalized by the VOLUME weight, so it must be closed out with its
    // own mass-weighted mean -- a separate pass over its own normalizer.
    if ( field.velocity_dispersion and not this->disp_weight.empty() )
    {
        size_t const nd = this->disp_weight.size();
        poolRanges( nd, size_t(1) << 16, CorePool::total(), [&](size_t const lo, size_t const hi)
        {
        for (size_t i = lo; i < hi; ++i)
        {
            Real const wm = this->disp_weight[i];
            if ( wm <= Real(0.) ) continue;
            Real const invm = Real(1.) / wm;
            Pvector<Real,noVelComp> const vbar = this->disp_velocity[i] * invm;   // <v>_mass
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                {
                    Real s = this->velocity_dispersion[i][c] * invm - vbar[a]*vbar[b];
                    if ( a == b and s < Real(0.) ) s = Real(0.);   // variance: clamp FP-noise negatives
                    this->velocity_dispersion[i][c] = s;
                    ++c;
                }
        }
        } );
    }

    if ( face ) psFaceGradientEnd();
    psFaceWordsRelease();   // (the word merge's face sums of a side whose gradient was not made: freed too)
    // these three are internal to this normalization and never written out -- release for real
    std::vector<Real>().swap( this->mass_weight );
    std::vector<Real>().swap( this->disp_weight );
    std::vector< Pvector<Real,noVelComp> >().swap( this->disp_velocity );
}


// The word merge's conversion of one accumulator (PSWordGrids, quantities.h): its words become its Real grid of nCell
// cells, each component's integer rounded ONCE to Real at the unit 2^-K -- the pass's own conversion
// (ps_interpolation.cc psFxToReal): the int64 word rounded to Real, times the exact 2^-K where |K| <= 120 (no
// result is then subnormal: a non-zero word is at least 1 unit, 2^-120 > FLT_MIN), else src/fixed_units.h fxToReal;
// then each carried component from its full 128-bit value -- and the words are freed. Over the free cores (every
// component on its own, so the same at any thread count). False (nothing done) for an accumulator without words.
// The scalar conversion is ONE pair of helpers (psWordToReal, psCarriedWordToReal), which the face sums' close-out
// (psFaceGradientCellWords) reads its words through too, so both give the same Real for the same word.
static inline bool psWordDirect(int const K) { return K >= -120 && K <= 120; }
static inline Real psWordInv(int const K) { return psWordDirect( K ) ? std::ldexp( Real( 1 ), -K ) : Real( 0 ); }
// one component's word w at the unit 2^-K: direct (psWordDirect) Real(w) times inv (= psWordInv), else fxToReal
static inline Real psWordToReal(int64_t const w, int const K, bool const direct, Real const inv)
{
    return direct ? Real( w ) * inv : fxToReal( __int128( w ), K );
}
// ... a carried component, from its full 128-bit value w + 2^64 carry
static inline Real psCarriedWordToReal(int64_t const w, int64_t const carry, int const K)
{
    return fxToReal( __int128( w ) + __int128( carry ) * ( __int128( 1 ) << 64 ), K );
}
#if NO_DIM == 3
// ... and component k of F, its carry looked up (only when F has one): the value psWordsToGrid leaves at out[k] (the
// face sums' close-out: 3-D only, as psFaceGradientCellWords)
static inline Real psWordComponent(FxWords const &F, size_t const k, bool const direct, Real const inv)
{
    if ( not F.carry.empty() )
    {
        auto const it = std::lower_bound( F.carry.begin(), F.carry.end(), uint64_t( k ),
                                          [](FxCarry const &c, uint64_t const key) { return c.first < key; } );
        if ( it != F.carry.end() and it->first == uint64_t( k ) )
            return psCarriedWordToReal( F.w[k], it->second, F.K );
    }
    return psWordToReal( F.w[k], F.K, direct, inv );
}
#endif
template <class Elem>
static bool psWordsToGrid(FxWords &F, std::vector<Elem> &grid, size_t const nCell)
{
    if ( F.w.empty() ) return false;
    size_t const comps = sizeof(Elem) / sizeof(Real);
    if ( F.w.size() != nCell * comps )
        throwError( "internal: the word merge's close-out has ", F.w.size(), " words for a grid of "
                    + std::to_string( nCell ) + " cells of " + std::to_string( comps ) + " components." );
    grid.assign( nCell, Elem() );
    Real *const out = reinterpret_cast<Real *>( grid.data() );
    int const K = F.K;
    bool const direct = psWordDirect( K );
    Real const inv = psWordInv( K );
    size_t const n = nCell * comps;
    poolRanges( n, size_t(1) << 16, CorePool::total(), [&](size_t const lo, size_t const hi)
    {
        if ( direct )
            for (size_t i = lo; i < hi; ++i) out[i] = psWordToReal( F.w[i], K, true, inv );
        else
            for (size_t i = lo; i < hi; ++i) out[i] = psWordToReal( F.w[i], K, false, inv );
    } );
    for (FxCarry const &e : F.carry)
        if ( e.first < n )
            out[e.first] = psCarriedWordToReal( F.w[e.first], e.second, K );
    F = FxWords();
    return true;
}

// THE CLOSE-OUT of the word merge, its first half (quantities.h): the words of this Quantities -- the merged full
// grid's, or a one-domain run's own -- become the Real grids the passes of the other merges hand over, in the order
// the pass forms them (ps_interpolation.cc, the pass end: psFixedToReal, then the density scaling and
// psExportStreams):
//   the mass words -> the density, as mass; where DW is the mass grid's sums (dwIsMass) disp_weight = that density,
//   copied before it is scaled (psFixedToReal's 'dispWeight = quantities->density'); then the density times
//   (invRhoBar / cellVolume), the pass's expression;
//   MOM -> velocity, M2 -> velocity_dispersion, GRAD -> velocity_gradient, MOMW -> mass_weight, DV -> disp_velocity,
//   DW -> disp_weight, SCAL -> scalar, SG -> scalar_gradient;
//   the stream counts: the exact deposit's SV words -> stream_count and its tet touches -> tet_touch (Real(count));
//   the sampled one's stream_count = Real(count) * (1 / nSub^NO_DIM), plus the released tets' multiplicity (RM).
// Then every summed grid the loop had pre-sized and no word filled comes back as zeros (as the other merges leave
// it), and the words are freed -- but the face sums S / N / P (fix 3 S7): they stay WORDS, once per face
// (quantities.h PSWordGrids, faceOnce), with their unit's factors, until normalizePhaseSpace makes the gradient from
// them (psFaceGradientCellWords) and frees them (psFaceGradientEnd); no Real face grid is made. normalizePhaseSpace
// follows (DTFE.cpp psCloseOutMoments), as after the other merges.
void Quantities::psWordsClose(Real const invRhoBar, Real const cellVolume)
{
    PSWordGrids &W = psWords;
    if ( not W.on ) return;
    size_t const n = W.cells;
    // the face sums: in the once layout (the merge's, or a one-domain run's routed alike: ps_fixed_merge.h), S and N
    // together, P only beside them, each with its boundary planes where the grid is not periodic
    bool const faceS = not W.at( RU_PS_FACE_S ).w.empty(), faceN = not W.at( RU_PS_FACE_N ).w.empty();
    bool const faceP = not W.at( RU_PS_FACE_P ).w.empty();
    if ( faceS or faceN or faceP )
    {
        size_t dimsCells = 1;
        for (int d = 0; d < NO_DIM; ++d) dimsCells *= W.dims[d];
        if ( not W.faceOnce or faceS != faceN or ( faceP and not faceS ) or dimsCells != n )
            throwError( "internal: the word merge's close-out found face sums that are not once per face, or not S "
                        "and N together, or not on its grid of ", n, " cells." );
        for (int slot = RU_PS_FACE_S; slot <= RU_PS_FACE_P; ++slot)
        {
            FxWords const &F = W.at( slot );
            if ( F.w.empty() ) continue;
            bool ok = F.w.size() == n * W.cellComps( slot );
            for (int a = 0; a < NO_DIM; ++a)
            {
                FxWords const &B = W.boundary( slot, a );
                size_t const want = W.periodic ? size_t( 0 ) : W.planeCells( a ) * PSWordGrids::facePer( slot );
                ok = ok and B.w.size() == want and ( B.w.empty() or B.K == F.K );
            }
            if ( not ok )
                throwError( "internal: the word merge's close-out found the face sums of ", RunUnits::name( slot ),
                            " (or their boundary planes) of another size or unit than their grid's." );
            W.faceDirect[slot - RU_PS_FACE_S] = psWordDirect( F.K );
            W.faceInv[slot - RU_PS_FACE_S] = psWordInv( F.K );
        }
    }
    if ( psWordsToGrid( W.at( RU_PS_MASS ), density, n ) )
    {
        if ( W.dwIsMass and W.at( RU_PS_FX + 6 ).w.empty() ) disp_weight = density;   // (DW: the mass, unscaled)
        poolRanges( n, size_t(1) << 16, CorePool::total(), [&](size_t const lo, size_t const hi)
        {
            for (size_t i = lo; i < hi; ++i)
                density[i] *= invRhoBar / cellVolume;
        } );
    }
    psWordsToGrid( W.at( RU_PS_FX + 0 ), velocity, n );
    psWordsToGrid( W.at( RU_PS_FX + 1 ), velocity_dispersion, n );
    psWordsToGrid( W.at( RU_PS_FX + 2 ), velocity_gradient, n );
    psWordsToGrid( W.at( RU_PS_FX + 3 ), mass_weight, n );
    psWordsToGrid( W.at( RU_PS_FX + 5 ), disp_velocity, n );
    psWordsToGrid( W.at( RU_PS_FX + 6 ), disp_weight, n );
    psWordsToGrid( W.at( RU_PS_FX + 7 ), scalar, n );
    psWordsToGrid( W.at( RU_PS_FX + 8 ), scalar_gradient, n );
    // (the face sums S / N / P: no Real grids -- they stay words, below)
    if ( W.exact )
    {   // the analytic multiplicity, and the raw tet touches
        psWordsToGrid( W.at( RU_PS_FX + 4 ), stream_count, n );
        if ( not W.count.empty() )
        {
            tet_touch.resize( n );
            poolRanges( n, size_t(1) << 16, CorePool::total(), [&](size_t const lo, size_t const hi)
            {
                for (size_t i = lo; i < hi; ++i) tet_touch[i] = Real( W.count[i] );
            } );
        }
    }
    else
    {   // the sample-mean multiplicity, then the released (never sampled) tets' volume fractions
        std::vector<Real> released;
        psWordsToGrid( W.at( RU_PS_FX + 9 ), released, n );
        if ( not W.count.empty() )
        {
            Real const invSamples = Real(1.) / Real(W.samplesPerCell);
            stream_count.resize( n );
            poolRanges( n, size_t(1) << 16, CorePool::total(), [&](size_t const lo, size_t const hi)
            {
                for (size_t i = lo; i < hi; ++i)
                    stream_count[i] = Real(W.count[i]) * invSamples;
                if ( not released.empty() )
                    for (size_t i = lo; i < hi; ++i)
                        stream_count[i] += released[i];
            } );
        }
    }
    // the summed grids pre-sized before the loop that no word filled: zeros, as the other merges leave them
    if ( not W.presized.empty() )
    {
        size_t k = 0;
        forEachField( [&](FieldInfo const &, auto merge, auto member)
        {
            if constexpr ( decltype( merge )::value == FIELD_SUM )
            {
                auto &v = this->*member;
                using T = typename std::decay<decltype( v )>::type::value_type;
                size_t const pre = k < W.presized.size() ? W.presized[k] : 0;
                ++k;
                if ( v.empty() and pre > 0 )
                {
                    v.assign( pre, T() );
                    std::memset( static_cast<void *>( v.data() ), 0, pre * sizeof( T ) );
                }
            }
        } );
    }
    // every word but the face sums' is converted and freed (psWordsToGrid); the counts and the sizes go too, and
    // 'on' (the words are closed out: a second call does nothing). The face sums, once per face, stay until the
    // gradient (psFaceWordsHeld; normalizePhaseSpace frees them on every way out)
    std::vector<int>().swap( W.count );
    std::vector<size_t>().swap( W.presized );
    W.on = false;
    if ( not psFaceWordsHeld() ) W.release();
}


// --face-gradient under the word merge (fix 3 S7), cell i once its velocity_gradient is normalized:
// psFaceGradientCell's arithmetic (quantities.h -- KEEP THE TWO IN STEP) on the face sums' words, once per face
// (PSWordGrids' header comment), each converted exactly as psWordsToGrid converts it (psWordComponent, with the
// unit's factors psWordsClose set), into a named Real before the expression: the values the Real face grids of the
// merge of before held. A cell's upper face is its own slot; its lower face is the cell below's upper slot on that
// axis, wrapped on a periodic grid, and for layer 0 of a grid that is not, that axis' boundary plane (row-major over
// the other axes). So where every pass held both shares of each face (the normal case) the bytes are those of the
// face grids converted per cell. 3-D only, as the face sums (ps_interpolation.cc psDepositFaces) and the word merge
// (fixed_units.h runUnitsWordMergeBuild) are: the 2-D build never holds face words.
void Quantities::psFaceGradientCellWords(size_t const i)
{
#if NO_DIM == 3
    if ( not psFaceWordsHeld() ) return;
    PSWordGrids const &W = psWords;
    size_t c[NO_DIM], stride[NO_DIM];   // the cell's coordinates, the grid's strides (row-major, last axis fastest)
    {
        size_t rem = i, s = 1;
        for (int d = NO_DIM - 1; d >= 0; --d)
        {
            c[d] = rem % W.dims[d];
            rem /= W.dims[d];
            stride[d] = s;
            s *= W.dims[d];
        }
    }
    // face sum f (0 S, 1 N, 2 P) on axis a, component j: the upper face's value, and the lower face's
    auto up = [&](int const f, int const a, size_t const j) -> Real
    {
        size_t const per = PSWordGrids::facePer( RU_PS_FACE_S + f );
        return psWordComponent( W.at( RU_PS_FACE_S + f ), ( i * size_t( NO_DIM ) + size_t( a ) ) * per + j,
                                W.faceDirect[f], W.faceInv[f] );
    };
    auto lo = [&](int const f, int const a, size_t const j) -> Real
    {
        size_t const per = PSWordGrids::facePer( RU_PS_FACE_S + f );
        if ( c[a] > 0 or W.periodic )
        {
            size_t const b = c[a] > 0 ? i - stride[a] : i + ( W.dims[a] - 1 ) * stride[a];   // the cell below
            return psWordComponent( W.at( RU_PS_FACE_S + f ), ( b * size_t( NO_DIM ) + size_t( a ) ) * per + j,
                                    W.faceDirect[f], W.faceInv[f] );
        }
        size_t const p = ( i / ( W.dims[a] * stride[a] ) ) * stride[a] + i % stride[a];   // its cell in the plane
        return psWordComponent( W.boundary( RU_PS_FACE_S + f, a ), p * per + j, W.faceDirect[f], W.faceInv[f] );
    };
    if ( not W.at( RU_PS_FACE_P ).w.empty() )
    {   // the mass flux: (the momentum out through the upper faces - in through the lower) / the cell volume
        Real vol = Real(1.), flux = Real(0.);
        for (int a = 0; a < NO_DIM; ++a)
        {
            vol *= psFaceDx[a];
            Real const pUp = up( 2, a, 0 ), pLo = lo( 2, a, 0 );
            flux += pUp - pLo;
        }
        mass_flux_divergence[i] = vol > Real(0.) ? flux / vol : Real(0.);
    }
    Pvector<Real,noGradComp> &g = velocity_gradient[i];
    velocity_divergence_streams[i] = g[0] + g[NO_DIM+1] + g[2*NO_DIM+2];   // [j*NO_DIM+i]: the trace
    if ( psStreamsTwinsOn ) psStreamsTwinsCell( i, g );   // (a copy: the mean flow's overwrites g below)
    for (int a = 0; a < NO_DIM; ++a)
    {
        Real const nLo = lo( 1, a, 0 ), nUp = up( 1, a, 0 );
        if ( !( nLo > Real(0.) && nUp > Real(0.) ) || !( psFaceDx[a] > Real(0.) ) ) continue;
        for (size_t j = 0; j < noVelComp; ++j)
        {
            Real const sUp = up( 0, a, j ), sLo = lo( 0, a, j );
            g[j*NO_DIM + size_t(a)] = ( sUp / nUp - sLo / nLo ) / psFaceDx[a];
        }
    }
#else
    (void)i;
#endif
}


// --face-gradient, cell i: the regular part's gradient g (normalized, before the mean flow's replaces it) and the shear
// and vorticity web_fields.cc derives from a stored gradient -- the same functions on the same values, so the bytes are
// a run's without the face gradient (tests/ps_face_gradient_check.sh P15)
void Quantities::psStreamsTwinsCell(size_t const i, Pvector<Real,noGradComp> g)
{
    if ( not velocity_gradient_streams.empty() )  velocity_gradient_streams[i]  = g;
    if ( not velocity_shear_streams.empty() )     velocity_shear_streams[i]     = velocityShear( g );
    if ( not velocity_vorticity_streams.empty() ) velocity_vorticity_streams[i] = velocityVorticity( g );
}


// The cell map of a sub-grid merge (addFromSubgrid): 'src' is a sub-grid (dims m, global origin o) of the full grid
// 'full', or -- mainOrigin/mainDims given -- of this grid's own window of it (--grid-window). Each local cell goes where
// subgridTarget (quantities.h; the fixed-point merge, src/ps_fixed_merge.h, maps cells by it too) sends it, axis by
// axis, so the map is one table per axis: the destination coordinate of every local index, or -1 outside the window.
// The merge then runs ROW by row (the last axis: contiguous in both grids), each row's destination start computed
// once and the row cut into runs only where the periodic wrap (or the window) breaks it -- no per-cell divisions and
// modulos -- and, since every destination cell receives at most one source cell, the rows split over threads: each
// element still gets the one '+=' (or OR) it got, so the sums are the same bit for bit.
struct SubgridRowMap
{
    std::vector<long> axis[NO_DIM];
    size_t m[NO_DIM], dims[NO_DIM];   // the source's dims, the destination's (full or window)
    size_t nRows = 1, rowLen = 1, dstTotal = 1;
    bool oneToOne = true;             // no two source cells share a destination cell (every sub-grid axis <= full)
    struct Run { size_t c0, len; long g0; };   // local [c0, c0+len) of a row -> destination [g0, g0+len) of its row
    std::vector<Run> runs;

    SubgridRowMap(size_t const *o, size_t const *mm, size_t const *full, size_t const *mainOrigin, size_t const *mainDims)
    {
        for (int d = 0; d < NO_DIM; ++d)
        {
            m[d] = mm[d];
            dims[d] = mainOrigin ? mainDims[d] : full[d];
            dstTotal *= dims[d];
            oneToOne = oneToOne && m[d] <= full[d];
            axis[d].resize( m[d] );
            for (size_t c = 0; c < m[d]; ++c)
            {
                size_t const gl = (c + o[d]) % full[d];   // the global index, as subgridTarget wraps it
                long g = long(gl);
                if ( mainOrigin )
                {
                    long ml = long(gl) - long(mainOrigin[d]);
                    if ( ml < 0 ) ml += long(full[d]);
                    g = ( ml < 0 || ml >= long(mainDims[d]) ) ? -1 : ml;
                }
                axis[d][c] = g;
            }
        }
        for (int d = 0; d < NO_DIM - 1; ++d) nRows *= m[d];
        rowLen = m[NO_DIM-1];
        std::vector<long> const &a = axis[NO_DIM-1];
        for (size_t c = 0; c < rowLen; )
        {
            if ( a[c] < 0 ) { ++c; continue; }
            size_t e = c + 1;
            while ( e < rowLen && a[e] == a[e-1] + 1 ) ++e;
            runs.push_back( Run{ c, e - c, a[c] } );
            c = e;
        }
    }

    // row r's first destination index (the flat index of its row with last coordinate 0); false outside the window
    bool rowStart(size_t const r, size_t &g) const
    {
        size_t c[NO_DIM], rem = r;
        for (int d = NO_DIM - 2; d >= 0; --d) { c[d] = rem % m[d]; rem /= m[d]; }
        g = 0;
        for (int d = 0; d < NO_DIM - 1; ++d)
        {
            long const x = axis[d][c[d]];
            if ( x < 0 ) return false;
            g = g * dims[d] + size_t(x);
        }
        g *= dims[NO_DIM-1];
        return true;
    }
};

// rows [r0, r1) of 'src' into 'dst' with op(dst element, src element): '+=' for the summed grids, OR for the masks
template <typename T, class Op>
void mergeSubgridRows(std::vector<T> const &src, std::vector<T> &dst, SubgridRowMap const &map,
                      size_t const r0, size_t const r1, Op const &op)
{
    for (size_t r = r0; r < r1; ++r)
    {
        size_t g;
        if ( !map.rowStart( r, g ) ) continue;
        T const *s = src.data() + r * map.rowLen;
        T *dRow = dst.data() + g;
        for (SubgridRowMap::Run const &run : map.runs)
        {
            T const *sp = s + run.c0;
            T *dp = dRow + run.g0;
            for (size_t k = 0; k < run.len; ++k) op( dp[k], sp[k] );
        }
    }
}

// the same, cell by cell in local order (the merge's former loop): a sub-grid wider than the full grid on some axis
// sends two of its cells to one destination cell, whose adds must then keep their order
template <typename T, class Op>
void mergeSubgridCells(std::vector<T> const &src, std::vector<T> &dst, size_t const *o, size_t const *m,
                       size_t const *full, size_t const *mainOrigin, size_t const *mainDims, Op const &op)
{
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= m[d];
    for (size_t l = 0; l < subTotal; ++l)
    {
        size_t rem = l, c[NO_DIM];
        for (int d = NO_DIM - 1; d >= 0; --d) { c[d] = rem % m[d]; rem /= m[d]; }   // local coords
        size_t g;
        if (!subgridTarget(c, o, full, mainOrigin, mainDims, g)) continue;
        op( dst[g], src[l] );
    }
}

// Accumulates 'other' (which holds only its Eulerian sub-box) into the full grid, mapping each cell by
// global index -- or into THIS grid's own window of it (mainOrigin/mainDims, --grid-window).
void Quantities::addFromSubgrid(Quantities const &other, size_t const *fullGrid, size_t const *mainOrigin, size_t const *mainDims)
{
    if (other.ps_subDims[0] == 0) { this->addFrom(other); return; }   // 'other' spans the full grid
    size_t const *o = other.ps_subOrigin;
    size_t const *m = other.ps_subDims;
    // the main grid is a window only when both are given: every field then goes through the window-aware mapping
    bool const window = mainOrigin && mainDims;
    size_t const *wo = window ? mainOrigin : nullptr;
    size_t const *wd = window ? mainDims : nullptr;
    SubgridRowMap const map( o, m, fullGrid, wo, wd );
    // Grow every unsized destination (zero-filled, so accumulation starts from zero) HERE, on the merging thread and
    // before any row is merged: each block is fully grown, its zero-fill scope (ScratchAlloc) closed, before a thread
    // writes into it. (scratch: an empty dst's zeros write nothing)
    forEachField( [&](FieldInfo const &, auto merge, auto member)
    {
        if constexpr ( decltype( merge )::value == FIELD_SUM || decltype( merge )::value == FIELD_OR )
        {
            auto &dst = this->*member;
            using T = typename std::decay<decltype( dst )>::type::value_type;
            if ( !( other.*member ).empty() && dst.size() < map.dstTotal ) ScratchAlloc::zeroResize( dst, map.dstTotal, T() );
        }
    } );
    // the summed grids '+=', the masks OR (bit masks stored as Real: small exact integers, see orField)
    auto add = [](auto &d, auto const &v) { d += v; };
    auto bitOr = [](Real &d, Real const v) { d = Real( int(d) | int(v) ); };
    auto mergeAll = [&](size_t const r0, size_t const r1)
    {
        forEachField( [&](FieldInfo const &, auto merge, auto member)
        {
            auto const &src = other.*member;
            if ( src.empty() ) return;
            if constexpr ( decltype( merge )::value == FIELD_SUM )
                mergeSubgridRows( src, this->*member, map, r0, r1, add );
            else if constexpr ( decltype( merge )::value == FIELD_OR )
                mergeSubgridRows( src, this->*member, map, r0, r1, bitOr );
        } );
    };
    if ( !map.oneToOne )
    {   // (not a sub-box of the full grid: the former cell-by-cell order, on this thread)
        forEachField( [&](FieldInfo const &, auto merge, auto member)
        {
            auto const &src = other.*member;
            if ( src.empty() ) return;
            if constexpr ( decltype( merge )::value == FIELD_SUM )
                mergeSubgridCells( src, this->*member, o, m, fullGrid, wo, wd, add );
            else if constexpr ( decltype( merge )::value == FIELD_OR )
                mergeSubgridCells( src, this->*member, o, m, fullGrid, wo, wd, bitOr );
        } );
        return;
    }
    // the rows over this thread and the free cores (at most 8: the merge is bound by memory bandwidth), a few
    // thousand cells per thread at least
    poolRanges( map.nRows, std::max<size_t>( 1, ( size_t(1) << 15 ) / std::max<size_t>( 1, map.rowLen ) ), 8, mergeAll );
}
#endif


// Reserves and zero-fills main-grid memory for each requested quantity ('--partition' option): one pass over the
// field list by each grid's sizing tag (quantities.h), one zero fill.
void Quantities::reserveMemory(size_t *gridSize, Field &field, User_options const *psRun)
{
    size_t totalSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        totalSize *= gridSize[i];
    bool const anyField = field.selected();
    (void)psRun;
    (void)anyField;

    forEachField( [&](FieldInfo const &fi, auto, auto member)
    {
        bool want = false;
        switch ( fi.sizing )
        {
        case SIZE_FIELD:       want = field.*(fi.flag); break;
        case SIZE_VELOCITY:    want = field.*(fi.flag) || field.velocity_dispersion; break;   // dispersion needs <v> too
#ifdef PHASE_SPACE
        case SIZE_PS_ANY:      want = psRun and anyField; break;
        case SIZE_PS_CAUSTICS: want = psRun and psRun->psCaustics and anyField; break;
        case SIZE_PS_EXACT:    want = psRun and psRun->psExactDeposit and anyField; break;
#endif
        default:               break;   // SIZE_AT_MERGE: the merge grows it
        }
        if ( want )
        {
            auto &v = this->*member;
            using T = typename std::decay<decltype( v )>::type::value_type;
            ScratchAlloc::zeroResize( v, totalSize, T() );   // (scratch: an empty vector's zeros write nothing)
        }
    } );
}


