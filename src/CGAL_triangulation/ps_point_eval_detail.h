/* The point evaluation's private header: what ps_point_eval.cc (point evaluation, '--sample-points') and
   ps_serve.cc ('--serve', the composite server and the occupancy maps) share -- the stream records and their packed
   layout, the evaluation context and its one instance (g_pe), the bucket walk, and the two per-cell steps a server
   repeats per tessellation (the PS vertex-mass pass, the standard cell filter). Include it from those two files only;
   the public interface is ../ps_point_eval.h. */

#ifndef PS_POINT_EVAL_DETAIL_HEADER
#define PS_POINT_EVAL_DETAIL_HEADER

#include "../define.h"

#include "../ps_point_eval.h"

#include "triangulation_common.h"
#include "ps_caustic_class.h"   // --ps-caustics: per-tetrahedron caustic stratification bits
#include "ps_exact_inside.h"    // exact, tie-consistent containment (shared with the grid deposit)

// the triangulation's finite simplices: CGAL calls them faces in 2D, cells in 3D
#if NO_DIM == 2
typedef DT::Finite_faces_iterator PSFiniteCellIterator;
#define PS_PE_FINITE_BEGIN(t) (t).finite_faces_begin()
#define PS_PE_FINITE_END(t) (t).finite_faces_end()
#else
typedef DT::Finite_cells_iterator PSFiniteCellIterator;
#define PS_PE_FINITE_BEGIN(t) (t).finite_cells_begin()
#define PS_PE_FINITE_END(t) (t).finite_cells_end()
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdarg>   // --serve-progress: serveProgress(fmt, ...)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <condition_variable>   // composite --serve: parallel partition loads
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>   // --serve: dup/dup2/read/write on the protocol channel
#include <sstream>    // --resume: the point-record files' text headers
#include <dirent.h>   // --resume: an earlier run's point-record files
#include <sys/stat.h> // --resume: a point-record file's size
#ifdef OPEN_MP
#include <omp.h>
#endif

// terminal-message prefix only -- the machinery is shared by both binaries and the PS
// output must stay byte-identical (including its log lines)
#ifdef PHASE_SPACE
#define PTS_MSG_BINARY "PS-DTFE:"
#else
#define PTS_MSG_BINARY "DTFE:"
#endif

namespace psPointEvalDetail {

using namespace psExactInside;

// One stream (containing tetrahedron) at a query point. Both density variants are kept so
// the estimator choice only affects the reduction, never the collected geometry.
struct StreamRec
{
    double denGeo;              // 'geometric': rho_bar * V_lag / V_eul, constant per tet
    double denDtfe;             // 'dtfe': vertex densities linearly interpolated at the point
    double vel[noVelComp];      // velocity linearly interpolated at the point
    double grad[NO_DIM];        // constant 'dtfe' density-profile gradient of the tet (--pts-den-grad; 0 otherwise)
    double vgrad[NO_DIM*NO_DIM];// constant velocity gradient of the tet, [d*3+j] = dv_j/dx_d (--pts-vel-grad; 0 otherwise)
    uint64_t ids[NO_DIM+1];     // sorted Lagrangian-vertex ParticleIDs (--ps-per-stream-ids; all 0 otherwise)
    PSCausticClass::CausticMask caustic;   // --ps-caustics: this tetrahedron's fold parity + stratification bits (ps_caustic_class.h; 0 otherwise). Point evaluation runs the derivative-free classifyTet only, so bits 7..8 (the A3/A4 indicators, which need the vertex-incident stencil) are never set here -- the type matches the grid deposit's so that adding them later cannot silently truncate
    double scal[noScalarComp];  // --pts-scalar: the stream's linear interpolant of its vertices' scalar values (0 otherwise)
    double lag[NO_DIM];         // --ps-per-stream-lagrangian (PS-DTFE): the stream's Lagrangian coordinate q(x), wrapped into the box when periodic (0 otherwise)
};

// The STORED form of a StreamRec: only the members this run's options fill. The core (both
// densities, the velocity, the caustic mask) always; the density gradient, velocity gradient and
// vertex IDs only with --pts-den-grad / --pts-vel-grad / --ps-per-stream-ids. The records ARE the
// memory of a large --sample-points run (an 8192^2 plane x ~4 streams at z=0 held ~48 GB of
// 176-B StreamRecs), and packed they take 42 B without those options, 114 B with --pts-vel-grad.
// unpack() zeroes the members that were not stored -- exactly what the record builders put there
// when the option is off -- so finalize() sorts and reduces the SAME records: bit-identical output.
struct RecLayout
{
    bool   grad = false, vgrad = false, ids = false, scal = false, lag = false;
    size_t bytes = 0;
    void init(bool withGrad, bool withVGrad, bool withIds, bool withScal, bool withLag)
    {
        grad = withGrad; vgrad = withVGrad; ids = withIds; scal = withScal; lag = withLag;
        bytes = 2*sizeof(double) + noVelComp*sizeof(double) + sizeof(PSCausticClass::CausticMask)
              + (grad  ? NO_DIM*sizeof(double)        : 0)
              + (vgrad ? NO_DIM*NO_DIM*sizeof(double) : 0)
              + (ids   ? (NO_DIM+1)*sizeof(uint64_t)  : 0)
              + (scal  ? noScalarComp*sizeof(double)  : 0)
              + (lag   ? NO_DIM*sizeof(double)        : 0);
    }
    void pack(std::vector<unsigned char> &out, StreamRec const &r) const
    {
        size_t const at = out.size();
        out.resize( at + bytes );
        unsigned char *p = out.data() + at;
        auto put = [&p](void const *src, size_t n) { std::memcpy( p, src, n ); p += n; };
        put( &r.denGeo, sizeof(r.denGeo) );
        put( &r.denDtfe, sizeof(r.denDtfe) );
        put( r.vel, sizeof(r.vel) );
        put( &r.caustic, sizeof(r.caustic) );
        if ( grad )  put( r.grad, sizeof(r.grad) );
        if ( vgrad ) put( r.vgrad, sizeof(r.vgrad) );
        if ( ids )   put( r.ids, sizeof(r.ids) );
        if ( scal )  put( r.scal, sizeof(r.scal) );
        if ( lag )   put( r.lag, sizeof(r.lag) );
    }
    void unpackOne(unsigned char const *p, StreamRec &r) const
    {
        r = StreamRec();                    // value-initialized: every unstored member is 0
        auto get = [&p](void *dst, size_t k) { std::memcpy( dst, p, k ); p += k; };
        get( &r.denGeo, sizeof(r.denGeo) );
        get( &r.denDtfe, sizeof(r.denDtfe) );
        get( r.vel, sizeof(r.vel) );
        get( &r.caustic, sizeof(r.caustic) );
        if ( grad )  get( r.grad, sizeof(r.grad) );
        if ( vgrad ) get( r.vgrad, sizeof(r.vgrad) );
        if ( ids )   get( r.ids, sizeof(r.ids) );
        if ( scal )  get( r.scal, sizeof(r.scal) );
        if ( lag )   get( r.lag, sizeof(r.lag) );
    }
};

// An Eulerian ownership box (see Context::own). 'on' = false keeps every tetrahedron.
struct OwnBox
{
    bool   on = false;
    double lo[NO_DIM], hi[NO_DIM];
    bool contains(double const c[NO_DIM]) const
    {
        for (int d = 0; d < NO_DIM; ++d)
            if ( c[d] < lo[d] || c[d] >= hi[d] ) return false;
        return true;
    }
};

struct Context
{
    bool   active = false;
    bool   finalized = false;
    bool   perStream = false;
    bool   perStreamIds = false;
    bool   ptsDenGrad = false;
    bool   ptsVelGrad = false;
    bool   psCaustics = false;      // --ps-caustics: also emit the per-point caustic mask
    bool   ptsScalar = false;       // --pts-scalar: also interpolate the vertex scalars per stream
    bool   ptsLagrangian = false;   // --ps-per-stream-lagrangian (PS): also record each stream's Lagrangian coordinate
    bool   useGeometric = false;    // variant used for totals, weights and per-stream output
    // the grid deposit's conventions, followed by the points ('--ps-point-conventions grid', the
    // default; 'legacy' clears both): see the notes at interpolatePoints_phaseSpace and psPointEvalFinalize
    bool   vertexMass = false;      // --ps-vertex-mass: the grid's tetrahedron masses in both per-stream densities
    bool   volumeWeighted = false;  // --ps-volume-weighted: equal-volume stream means (velocity, its gradient, scalars)
    bool   periodic = false;
    double rhoBar = 1.;             // output densities are rho/rhoBar (grid convention)
    double boxLo[NO_DIM], boxLen[NO_DIM];

    // exact containment (see the file header): the canonical-image frame (ps_exact_inside.h)
    Frame  canon;

    // Standard binary: which tetrahedra this triangulation OWNS -- those whose Eulerian centroid
    // (vertices min-image wrapped to vertex 0) lies in [lo, hi). A periodic single triangulation
    // holds boundary tetrahedra twice (original + image in the padding) and owns the primary-box
    // one; a composite server's partition owns its own region. The PS worker's ownership is the
    // Lagrangian-centroid test inside psFilterCell (the partition's lagrangianRegion) instead.
    OwnBox own;
    // a batch '--sample-points --partition' run set a partition's box here (psPointEvalSetOwnership):
    // finalize() then words its overlap warning for partitions
    bool   ownPartitioned = false;

    size_t nPoints = 0;
    std::vector<double> pos;        // 3*nPoints, wrapped into the box when periodic

    // uniform bucket grid over the query points (CSR layout)
    int    nB[NO_DIM];
    double bInvW[NO_DIM];
    std::vector<size_t>   bucketStart;   // nB^3 + 1 offsets
    std::vector<uint32_t> bucketPts;     // point indices grouped by bucket

    // The stream records of every point, appended in BLOCKS by (possibly concurrent) workers.
    // An entry is the uint32 point index followed by the PACKED record (RecLayout: only the
    // members the options fill); finalize() groups the entries by point with a counting sort.
    // A worker holds one block at most: one std::vector per point (a heap block each), with every
    // worker holding its partition's hits UNPACKED (~200 B each) until one merge at the end, would
    // peak at 28 GB for 16.8M points at 2.1M particles.
    std::vector< std::vector<unsigned char> > recBlocks;
    // the partition each block's records belong to (psPointEvalSetPartition; -1 outside a partition loop), one
    // per block: a worker's sink serves one triangulation, so a block never mixes partitions. The '--resume'
    // checkpoint saves a merged partition's blocks
    std::vector<int> recBlockPart;
    size_t nRecs = 0;
    RecLayout layout;

    // finalized outputs
    std::vector<double>   outDen;        // N
    std::vector<double>   outVel;        // 3N
    std::vector<double>   outDisp;       // 6N, upper triangle row-major: xx xy xz yy yz zz
    std::vector<int32_t>  outStreams;    // N
    std::vector<uint64_t> outOffsets;    // N+1 (only with --ps-per-stream)
    std::vector<double>   outRecords;    // 4 per stream: density, vx, vy, vz
    std::vector<uint64_t> outIds;        // NO_DIM+1 per stream (only with --ps-per-stream-ids)
    std::vector<double>   outDenGrad;    // 3N (only with --pts-den-grad)
    std::vector<double>   outVelGrad;    // 9N (only with --pts-vel-grad)
    std::vector<double>   outRecGrad;    // 3 per stream (only with --pts-den-grad + --ps-per-stream)
    std::vector<int32_t>  outCaustic;    // N (only with --ps-caustics): OR of the containing tetrahedra's masks
    std::vector<double>   outScalar;     // noScalarComp per point (only with --pts-scalar): density-weighted mean over the streams
    std::vector<double>   outRecScalar;  // noScalarComp per stream (--pts-scalar + --ps-per-stream)
    std::vector<double>   outRecLag;     // 3 per stream (only with --ps-per-stream-lagrangian)
};

// The state the point evaluation and the servers share, ONE instance (g_pe, defined in ps_point_eval.cc). The partition
// tag of the calling thread (g_curPartition, ps_point_eval.cc) is thread_local and so stays a variable of its own; the
// servers' own state (the protocol channel, the composite server) is ps_serve.cc's g_serve.
struct PointEvalState
{
    Context ctx;                                          // points, records, outputs (psPointEvalInit re-arms it)
    std::mutex mergeMutex;                                // guards ctx.recBlocks while workers hand over their blocks
    std::vector<Cell_handle> const *cellList = nullptr;   // candidate cells of one '--serve' request (a server's
                                                          // cell index), or null: every finite cell
};
extern PointEvalState g_pe;

// Installs a query point set (wrapped, bucketed, fresh records); buckets the installed points again (ps_point_eval.cc).
void setQueryPoints(std::vector<double> &&pts, int const bucketFloor);
void bucketQueryPoints(int const bucketFloor);

// The flat index of every bucket of [lo, hi] per axis (inclusive; n[d] buckets on axis d, periodic indices
// wrapped), the last axis fastest: the order of the nested 3D loops it replaced, in any NO_DIM.
template <class F>
inline void forEachBucketIn(int const lo[NO_DIM], int const hi[NO_DIM], int const n[NO_DIM], bool const periodic,
                            F const &f)
{
    int c[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        if ( lo[d] > hi[d] ) return;
        c[d] = lo[d];
    }
    for (;;)
    {
        size_t b = 0;
        for (int d = 0; d < NO_DIM; ++d)
        {
            int const w = periodic ? ((c[d] % n[d]) + n[d]) % n[d] : c[d];
            b = b * size_t(n[d]) + size_t(w);
        }
        f( b );
        int d = NO_DIM - 1;
        while ( d >= 0 and ++c[d] > hi[d] ) { c[d] = lo[d]; --d; }
        if ( d < 0 ) return;
    }
}

} // namespace psPointEvalDetail

#ifdef PHASE_SPACE

namespace psPointEvalDetail {

/* The grid deposit's conventions at the points ('--ps-point-conventions grid', the default).

   Without them the planes and grids of the same run measure different things: the grids use --ps-vertex-mass
   (tetrahedron masses from the particles, m_tet = sum_v m_v/deg_v) and --ps-volume-weighted (volume-weighted
   velocity moments), the planes rho_bar*V_lag masses and density-weighted stream means. With TNG's evolved
   z = 127 positions as the Lagrangian input rho_bar*V_lag carries the initial contrast, so every density mode of
   a plane comes out filtered by 1 - D(z_ic)/D(z): at z = 20 a 4096^2 plane's contrast is 0.81-0.82 times the
   grid's (rms 0.211 against 0.280), and its multi-stream velocity divergence 2-4x smaller.

   Under --ps-vertex-mass both point estimators carry the grid's tetrahedron masses, through ONE number per
   vertex: the ratio of the two mass models over the vertex's star of depositing tetrahedra,

       c_v = sum_{t in star(v)} m_tet / sum_{t in star(v)} rho_bar V_lag(t)      (the grid's / rho_bar V_lag),

   i.e. the Lagrangian density of the grid's masses around v in units of rho_bar (the IC contrast the default
   masses miss, plus their vertex-share noise). Then
     'dtfe'       rho_v -> rho_v c_v: the vertex density rho_bar V_lag(star)/V_eul(star) becomes
                  M(star)/V_eul(star), the grid's mass around v over its Eulerian volume (exactly so where
                  no cell of the star is dropped: vertexDensity sums every cell, c_v the depositing ones);
     'geometric'  rho_tet -> rho_tet <c_v>, the mean over the tetrahedron's vertices: (V_lag/V_eul)(t) times the
                  local Lagrangian density. Over the depositing cells this conserves the grid's mass exactly,
                  sum_t V_lag(t) <c_v> rho_bar = sum_v M(star_v)/(d+1) = sum_t m_tet.
   With the default masses c_v = 1 and both are the plain estimators. Why not m_tet/V_eul(t) itself, the grid's own
   per-tetrahedron density: near-lattice initial conditions (TNG's) tile with near-flat slivers whose equal vertex
   shares give them a normal mass in almost no volume -- a cell average of the grid holds that mass, a POINT is
   almost never inside such a sliver: on a 32^3 crossed-wave box (jitter 0.05) a 64^3 point set's mean density
   came out 0.89 instead of 1. And not standard DTFE's (d+1) m_v/V_eul(star) either: on a near-lattice the star
   volumes vary with the tetrahedralization, which M(star) averages over the star (a scipy prototype on a uniform
   24^3 lattice, jitter 0.01: vertex std 0.19 against 0.08). The non-periodic hull cells keep their constant density,
   corrected the same way; the density gradient (--pts-den-grad) belongs to the corrected 'dtfe' profile. Under
   --ps-volume-weighted the per-point velocity, its gradient and the scalar are equal-weight stream means
   (psPointEvalFinalize). '--ps-point-conventions legacy' keeps the plain estimators; without the two grid options both
   choices run the same code, byte for byte.

   c_v is computed by prepareVertexMass over the whole tessellation (the batch worker per partition, before the
   grid deposit, which recounts its degrees after a reset; a server once per tessellation it builds or loads) and
   left in the vertex's volumeAccumulator() -- vertexDensity's scratch, unused once the vertex densities are set,
   and not cached with the tessellation. Its two passes over the depositing cells (psForEachDepositingCell, the
   grid's own walk) keep their per-vertex sums in a side table indexed through psDegree() (20 B per vertex, freed
   at the end), which then gets the vertex's true degree back. */
inline void prepareVertexMass(DT &dt, User_options &userOptions)
{
    Box boxCoordinates = userOptions.region;
    size_t const nV = dt.number_of_vertices();
    if ( nV > size_t( std::numeric_limits<int32_t>::max() ) )
        throwError( "--ps-vertex-mass point evaluation: more than 2^31 vertices in one tessellation." );
    // (0) index the vertices through psDegree
    {
        int32_t k = 0;
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            vit->info().psDegree() = k++;
    }
    std::vector<int32_t> deg( nV, 0 );
    std::vector<double> lagStar( nV, 0. ), massStar( nV, 0. );
    double const invFact = 1. / factorial( NO_DIM );
    auto lagVolume = [invFact](Cell_handle &cell)
    {
        double Lag[NO_DIM][NO_DIM];
        for (int v = 0; v < NO_DIM; ++v)
            for (int i = 0; i < NO_DIM; ++i)
                Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
        return std::fabs( determinant(Lag) ) * invFact;
    };
    // (1) the degrees (the grid's) and the Lagrangian star volumes
    psForEachDepositingCell( dt, userOptions, boxCoordinates, [&](Cell_handle &cell, PSCellGeometry const &)
    {
        double const vl = lagVolume( cell );
        for (int v = 0; v <= NO_DIM; ++v)
        {
            size_t const k = size_t( cell->vertex(v)->info().psDegree() );
            deg[k] += 1;
            lagStar[k] += vl;
        }
    } );
    // (2) the grid's tetrahedron masses (psVertexShareMass's sum, from the table's degrees) summed per star
    psForEachDepositingCell( dt, userOptions, boxCoordinates, [&](Cell_handle &cell, PSCellGeometry const &)
    {
        double m = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
        {
            int32_t const d = deg[ size_t( cell->vertex(v)->info().psDegree() ) ];
            if ( d > 0 ) m += double( cell->vertex(v)->info().weight() ) / double( d );
        }
        for (int v = 0; v <= NO_DIM; ++v)
            massStar[ size_t( cell->vertex(v)->info().psDegree() ) ] += m;
    } );
    // (3) c_v into volumeAccumulator, the true degree back into psDegree
    double const rhoBar = double( userOptions.averageDensity );
    for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
    {
        size_t const k = size_t( vit->info().psDegree() );
        vit->info().volumeAccumulator() = lagStar[k] > 0. ? massStar[k] / ( rhoBar * lagStar[k] ) : 0.;
        vit->info().psDegree() = deg[k];
    }
}

} // namespace psPointEvalDetail

#else // !PHASE_SPACE

namespace psPointEvalDetail {

// The standard worker's per-cell filter, shared with the composite server's occupancy pass:
// the vertex densities/velocities (false when one is not positive -- a degenerate hull
// configuration), the Eulerian vertex positions min-image wrapped to vertex 0 (eulerPos) and as
// stored (rawPos, for the exact test's canonical frame), and ownership: a periodic run holds a
// boundary tetrahedron once per image, a composite partition also its neighbours' tetrahedra in
// its padding, and exactly ONE copy -- the one whose centroid lies in 'own' -- is kept, so a point
// near a boundary receives exactly one record from each physical tetrahedron.
inline bool standardCell(Cell_handle &cell, OwnBox const &own, double rho[NO_DIM+1],
                         double vel[NO_DIM+1][noVelComp], Real eulerPos[NO_DIM+1][NO_DIM],
                         Real rawPos[NO_DIM+1][NO_DIM])
{
#ifdef TEST_PADDING
    // skip cells touching a dummy padding-test vertex
    for (int v = 0; v <= NO_DIM; ++v)
        if (cell->vertex(v)->info().isDummy()) return false;
#endif

    // vertex DTFE densities must be positive to interpolate the linear profile
    for (int v = 0; v <= NO_DIM; ++v)
    {
        rho[v] = double( cell->vertex(v)->info().density() );
        if ( rho[v] <= 0. ) return false;
        for (size_t j = 0; j < noVelComp; ++j)
            vel[v][j] = double( cell->vertex(v)->info().velocity(j) );
    }

    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            rawPos[v][d] = eulerPos[v][d] = Real( cell->vertex(v)->point()[d] );
    if ( g_pe.ctx.periodic )
        for (int v = 1; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real const diff = eulerPos[v][d] - eulerPos[0][d];
                if (diff >  Real(0.5) * Real(g_pe.ctx.boxLen[d])) eulerPos[v][d] -= Real(g_pe.ctx.boxLen[d]);
                if (diff < -Real(0.5) * Real(g_pe.ctx.boxLen[d])) eulerPos[v][d] += Real(g_pe.ctx.boxLen[d]);
            }
    if ( own.on )
    {
        double cen[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
        {
            cen[d] = 0.;
            for (int v = 0; v <= NO_DIM; ++v) cen[d] += double(eulerPos[v][d]);
            cen[d] /= double(NO_DIM + 1);
        }
        if ( not own.contains(cen) ) return false;
    }
    return true;
}

} // namespace psPointEvalDetail

#endif // PHASE_SPACE

// The per-triangulation workers (ps_point_eval.cc), which the servers call per request and partition.
#ifdef PHASE_SPACE
void interpolatePoints_phaseSpace(DT &dt, User_options &userOptions);
#else
void interpolatePoints_standard(DT &dt, User_options &userOptions);
void interpolatePoints_standardOwn(DT &dt, User_options &userOptions, psPointEvalDetail::OwnBox const &own);
#endif

#endif
