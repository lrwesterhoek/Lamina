/* Arbitrary-point evaluation (--sample-points), shared by BOTH binaries.

   PS-DTFE: evaluates the multi-stream phase-space fields at user-supplied Eulerian points
   instead of (only) on a regular grid. The triangulation is built in Lagrangian space;
   every Delaunay cell maps to a (possibly folded) Eulerian tetrahedron, and a query point
   receives one stream per tetrahedron that contains it. Per point we report the total
   density, the density-weighted mean velocity, the velocity dispersion tensor and the
   stream count; --ps-per-stream additionally records every stream's own density and velocity,
   and --ps-per-stream-ids each stream's identity (the sorted Lagrangian-vertex ParticleIDs).
   --pts-den-grad adds the density gradient: per stream the constant gradient of the linear
   'dtfe' profile (regardless of --ps-stream-density -- the geometric density is piecewise
   constant, so the dtfe-profile gradient is the only well-defined one), per point the sum
   over its streams.

   Standard DTFE (no PHASE_SPACE): the same interface and .pts_* file formats over the
   EULERIAN tessellation -- exactly one containing tetrahedron per point, so the stream
   count is the 0/1 coverage flag, the density is the plain linear DTFE interpolant and the
   dispersion is identically 0. The context, bucket index, reduction and writers below are
   shared verbatim; only the per-triangulation worker differs (interpolatePoints_standard).
   --ps-per-stream/--ps-per-stream-ids and --ps-stream-density are PS-only (rejected at option
   parsing); the Sample_point mechanism is a separate path.

   Per-stream density comes in two variants (--ps-stream-density):
     'dtfe'      (default) linear interpolation of the Lagrangian-vertex DTFE densities
                 inside the tetrahedron -- matches the PhaseSpaceDTFE class of the method
                 authors' reference implementation (github.com/jfeldbrugge/PS-DTFE,
                 Python/density.py). Continuous within a stream, but not mass-conserving.
     'geometric' the constant per-tetrahedron density m_tet / V_eul (= rho_bar V_lag/V_eul)
                 -- the density whose cell-averaged, quantized rendering the mass-conserving
                 grid deposit produces. Discontinuous across tetrahedra, conserves mass.
   Both follow the run's grid conventions ('--ps-point-conventions grid', the default):
   under --ps-vertex-mass they carry the grid's tetrahedron masses through a per-vertex mass-model
   ratio, under --ps-volume-weighted the per-point velocity, its gradient and the scalar are
   equal-volume stream means (see 'The grid deposit's conventions at the points' below);
   '--ps-point-conventions legacy' keeps rho_bar V_lag and density weighting. A partitioned run's stream
   records are checkpointed per partition for '--resume' ('--resume with --sample-points' below).

   The candidate search buckets the (static, read-only) QUERY POINTS on a uniform grid and
   streams the partition's tetrahedra through it: each kept tetrahedron looks up only the
   buckets its Eulerian bounding box overlaps. This is the same O(T + P + hits) complexity
   as a hierarchy over the tetrahedron boxes, with the smaller (point) set indexed.

   Partition protocol: the cell filters below are the CPU grid deposit's filters VERBATIM
   (ps_interpolation.cc) -- dummy-vertex skip, Lagrangian-centroid ownership, bad-density
   drop under periodic, minimum-image wrap of vertices 1..NO_DIM to vertex 0, relative-
   determinant degeneracy and the zero-inverse check -- so a stream is contributed by
   exactly one Lagrangian partition and the union over partitions is exactly the serial
   stream set. Per-point stream records accumulate across partitions (the analogue of
   psDeferNormalization: moments are formed only once, in psPointEvalFinalize, from the
   deterministically sorted records, so results do not depend on partition/thread order).

   Containment is EXACT and tie-consistent (pointInTet; the machinery lives in
   ps_exact_inside.h, which the CPU grid deposit shares for its sub-samples): a point is counted by
   exactly one tetrahedron of every stream, including points ON a shared face, edge or
   vertex -- e.g. a query at a particle position, which a tolerance-based test counts once
   per incident tetrahedron (~25 fake streams). Two ingredients make that possible:
     * every periodic image of a particle is mapped to one canonical coordinate
       (canonicalCoord), so tetrahedra sharing a face see bit-identical vertices even when
       one holds the original particle and the other its (float-shifted) periodic copy;
     * the decision uses exact orientation predicates (CGAL, filtered) with a symbolic
       perturbation of the query point (simulation of simplicity) that breaks every tie
       the same way in all tetrahedra.
   A cheap double-precision classification decides every point that is not within a
   rigorous per-tetrahedron uncertainty band of a face; only the band goes to the exact
   predicates, so the common path costs what a tolerance test would.

   Each triangulation's cells are processed by a small thread pool (pointEvalThreads): all
   cores for a single-triangulation run, cores / concurrent partitions inside the
   partitioned PS loop, so the partition-level parallelism is never oversubscribed.

   The same machinery answers '--serve' requests: ps_serve.cc holds the servers (the single and the
   composite one; the protocol is documented at the top of that file) and the partitions' occupancy maps,
   ps_mean_streams.cc the auto-tuner's stream-count estimate. ps_point_eval_detail.h is what they share. */

#include "ps_point_eval_detail.h"   // the context, records and helpers shared with ps_serve.cc

#include "../output_staging.h"   // the .pts_* files are staged with the grids (crash-safe outputs)
#include "../env_knobs.h"
#include "../thread_count.h" // coresPerTeam: the point evaluation's threads; CorePool::Hold: their cores in the pool

namespace psPointEvalDetail {

// The state shared with the servers (ps_point_eval_detail.h).
PointEvalState g_pe;

// the partition the calling thread deposits (psPointEvalSetPartition): read where a triangulation's cell walk
// starts, and carried by its sinks onto their worker threads
static thread_local int g_curPartition = -1;

// A worker thread's hits: packed at once (point index + RecLayout bytes) and handed to the shared
// store whenever the block reaches FLUSH_BYTES -- a thread never holds more than one block.
struct HitSink
{
    static constexpr size_t FLUSH_BYTES = size_t(16) << 20;
    std::vector<unsigned char> buf;
    size_t n = 0;
    int part = -1;                  // the partition whose triangulation this sink walks (g_pe.ctx.recBlockPart)

    void push_back(std::pair<uint32_t, StreamRec> const &h)
    {
        if ( buf.capacity() == 0 ) buf.reserve( FLUSH_BYTES + 1024 );
        size_t const at = buf.size();
        buf.resize( at + sizeof(uint32_t) );
        std::memcpy( buf.data() + at, &h.first, sizeof(uint32_t) );
        g_pe.ctx.layout.pack( buf, h.second );
        ++n;
        if ( buf.size() >= FLUSH_BYTES ) flush();
    }
    void flush()
    {
        if ( n == 0 ) return;
        buf.shrink_to_fit();
        std::lock_guard<std::mutex> lock( g_pe.mergeMutex );
        g_pe.ctx.recBlocks.push_back( std::move(buf) );
        g_pe.ctx.recBlockPart.push_back( part );
        g_pe.ctx.nRecs += n;
        buf = std::vector<unsigned char>();
        n = 0;
    }
};


// 2x2 / 3x3 double inverse by the adjugate (math_functions.h adjugateInverse) WITHOUT inverseNd's relative
// singularity guard: the standard worker drops no tetrahedron for being small or flat (interpolatePoints_standardOwn
// says why). Returns det, or 0 when det is exactly 0 (2D: or not finite); the inverse is valid when det is finite
// and nonzero.
inline double inverse3x3Unguarded(double const m[NO_DIM][NO_DIM], double inv[NO_DIM][NO_DIM])
{
    double det;
#if NO_DIM == 2
    auto const singular = [](double const d) { return d == 0. || not std::isfinite(d); };
#else
    auto const singular = [](double const d) { return d == 0.; };
#endif
    return adjugateInverse( m, inv, det, singular ) ? det : 0.;
}


// Eulerian bounding box of a wrapped tetrahedron -> point-bucket range and rel-space bbox
// pre-test bounds. The slack covers the canonical-vs-wrapped geometry discrepancy (epsAbs),
// so no point the exact test could accept is ever culled here. Shared by the PS and standard
// workers so the candidate search is identical in both binaries. Returns false when the
// (non-periodic) range misses every bucket.
inline bool tetBucketRange(Real const eulerPos[NO_DIM+1][NO_DIM],
                           double relLo[NO_DIM], double relHi[NO_DIM],
                           int bLo[NO_DIM], int bHi[NO_DIM])
{
    for (int d = 0; d < NO_DIM; ++d)
    {
        double eMin = double(eulerPos[0][d]);
        double eMax = eMin;
        for (int v = 1; v <= NO_DIM; ++v)
        {
            double const c = double(eulerPos[v][d]);
            if (c < eMin) eMin = c;
            if (c > eMax) eMax = c;
        }
        double const margin = 1.e-5 * (eMax - eMin) + 1.e-12 + g_pe.ctx.canon.epsAbs;
        relLo[d] = eMin - double(eulerPos[0][d]) - margin;
        relHi[d] = eMax - double(eulerPos[0][d]) + margin;

        bLo[d] = int( std::floor( (eMin - margin - g_pe.ctx.boxLo[d]) * g_pe.ctx.bInvW[d] ) );
        bHi[d] = int( std::floor( (eMax + margin - g_pe.ctx.boxLo[d]) * g_pe.ctx.bInvW[d] ) );
        if ( g_pe.ctx.periodic )
        {
            if ( bHi[d] - bLo[d] + 1 >= g_pe.ctx.nB[d] ) { bLo[d] = 0; bHi[d] = g_pe.ctx.nB[d] - 1; }  // spans every bucket: visit each once
        }
        else
        {
            if ( bLo[d] < 0 ) bLo[d] = 0;
            if ( bHi[d] >= g_pe.ctx.nB[d] ) bHi[d] = g_pe.ctx.nB[d] - 1;
            if ( bLo[d] > bHi[d] ) return false;
        }
    }
    return true;
}


// ---- exact, tie-consistent containment (ps_exact_inside.h) --------------------------------

// Query point p in the canonical frame (its stored position is already wrapped into the box).
inline void canonicalQuery(uint32_t p, double pc[NO_DIM])
{
    canonicalQuery( g_pe.ctx.canon, &g_pe.ctx.pos[size_t(p)*NO_DIM], pc );
}


// Minimum-image displacement of query point p to (wrapped) vertex 0 and the rel-space bbox
// pre-test, then the containment decision in three tiers, each deciding only what it provably
// can: (1) the usual barycentrics on the wrapped geometry, outside the tetrahedron's fineBand;
// (2) a double-precision classification in the canonical frame, outside its own (much
// narrower) band; (3) the exact predicate. Every tier-1/2 decision equals the exact one, so
// the whole test IS the exact, tie-consistent one -- at the cost of a plain tolerance test
// for all but the points hugging a face. The returned bary[] are the usual interpolation
// weights on the wrapped geometry (Ax stores edges as rows, so bary = inv^T * rel -- note the
// [i][v] transpose). Shared by the PS and standard workers.
inline bool pointInTet(uint32_t p, Real const eulerPos[NO_DIM+1][NO_DIM],
                       Real const rawPos[NO_DIM+1][NO_DIM], TetTest &T, double const band1,
                       double const inv[NO_DIM][NO_DIM],
                       double const relLo[NO_DIM], double const relHi[NO_DIM],
                       double bary[NO_DIM])
{
    double rel[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double r = g_pe.ctx.pos[size_t(p)*NO_DIM+d] - double(eulerPos[0][d]);
        if ( g_pe.ctx.periodic )
        {
            if (r >  0.5 * g_pe.ctx.boxLen[d]) r -= g_pe.ctx.boxLen[d];
            if (r < -0.5 * g_pe.ctx.boxLen[d]) r += g_pe.ctx.boxLen[d];
        }
        if ( r < relLo[d] || r > relHi[d] ) return false;
        rel[d] = r;
    }

    // tier 1: the wrapped-geometry barycentrics (also the interpolation weights)
    int const fine = fineClassify( inv, rel, band1, bary );
    if ( fine != 0 ) return fine > 0;

    // tiers 2 and 3 in the canonical frame (bary[] already holds the tier-1 weights)
    double pc[NO_DIM];
    canonicalQuery( p, pc );
    return canonicalInside( g_pe.ctx.canon, rawPos, T, pc );
}


// Threads for one triangulation's point evaluation: every core when this is the only
// triangulation being processed; inside the partitioned PS loop (an OpenMP team of
// concurrent partitions), the cores left per partition -- so the two levels never
// oversubscribe (thread_count.h). std::thread rather than a nested OpenMP region, which is disabled by
// default. DTFE_PTS_THREADS pins it (benchmarks, tests).
inline int pointEvalThreads()
{
    return coresPerTeam( "DTFE_PTS_THREADS" );
}

// Runs work(cell, hits) over every finite cell of dt -- or only over g_pe.cellList when the server
// has narrowed a request to the cells near its points -- with nThreads threads. Walking dt, each
// thread takes every nThreads-th block of 256 cells, so no index array is materialized; each
// thread's HitSink hands its packed records to the shared store in blocks (record order is
// irrelevant: finalize sorts every point's records, so any order gives bit-identical outputs).
template <typename Work>
void forEachCellParallel(DT &dt, int nThreads, Work work)
{
    size_t const BLOCK = 256;
    if ( g_pe.cellList )   // a small candidate list is not worth a thread per block
        nThreads = int( std::min<size_t>( size_t(std::max(1, nThreads)), g_pe.cellList->size() / (8 * BLOCK) + 1 ) );
    size_t const nT = size_t( std::max(1, nThreads) );
    std::vector<HitSink> hits( nT );
    for (HitSink &h : hits) h.part = g_curPartition;   // this thread's partition, for the worker threads below
    std::vector<std::exception_ptr> errors( nT );

    auto run = [&](int t)
    {
        try
        {
            if ( g_pe.cellList )
            {
                std::vector<Cell_handle> const &list = *g_pe.cellList;
                for (size_t b = size_t(t) * BLOCK; b < list.size(); b += nT * BLOCK)
                    for (size_t i = b; i < std::min(b + BLOCK, list.size()); ++i)
                    {
                        Cell_handle cell = list[i];
                        work( cell, hits[size_t(t)] );
                    }
                return;
            }
            size_t idx = 0;
            for (PSFiniteCellIterator itC = PS_PE_FINITE_BEGIN(dt); itC != PS_PE_FINITE_END(dt); ++itC, ++idx)
            {
                if ( ( (idx / BLOCK) % nT ) != size_t(t) ) continue;
                Cell_handle cell = itC;
                work( cell, hits[size_t(t)] );
            }
        }
        catch (...) { errors[size_t(t)] = std::current_exception(); }
    };

    if ( nT <= 1 )
        run(0);
    else
    {   // a fixed team beside the partition loop's other work: its cores counted in the core pool while it runs, so the
        // pooled loops of the other partitions in flight do not take them as free (thread_count.h CorePool::Hold)
        CorePool::Hold const hold( int(nT) - 1 );
        std::vector<std::thread> pool;
        pool.reserve( nT - 1 );
        for (size_t t = 1; t < nT; ++t)
            pool.emplace_back( run, int(t) );
        run(0);
        for (std::thread &th : pool) th.join();
    }
    for (std::exception_ptr const &e : errors)
        if ( e ) std::rethrow_exception( e );

    for (HitSink &h : hits)                 // the last, partial blocks
        h.flush();
}


// Reads the sample-point file: text ("x y z" per line) or raw binary (float64, N x 3,
// row-major, native little-endian, no header), auto-detected from the leading bytes.
static void readSamplePoints(std::string const &filename, std::vector<double> &points)
{
    std::ifstream file( filename.c_str(), std::ios::binary );
    if ( not file )
        throwError( "Cannot open the '--sample-points' file '", filename, "'." );
    file.seekg( 0, std::ios::end );
    size_t const fileSize = size_t( file.tellg() );
    file.seekg( 0, std::ios::beg );
    if ( fileSize == 0 )
        throwError( "The '--sample-points' file '", filename, "' is empty." );

    char head[4096];
    size_t const headSize = std::min( fileSize, sizeof(head) );
    file.read( head, headSize );
    bool isText = true;
    for (size_t i = 0; i < headSize; ++i)
    {
        char const c = head[i];
        bool const numeric = (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.'
                             || c == 'e' || c == 'E' || c == ' ' || c == '\t'
                             || c == '\n' || c == '\r';
        if ( not numeric ) { isText = false; break; }
    }

    if ( isText )
    {
        file.clear();
        file.seekg( 0, std::ios::beg );
        double v[NO_DIM];
        while ( file >> v[0] )
        {
            for (int d = 1; d < NO_DIM; ++d)
                if ( not (file >> v[d]) )
                    throwError( "The '--sample-points' text file '", filename,
                                NO_DIM == 2 ? "' ends mid-point; every line must hold 'x y'."
                                            : "' ends mid-point; every line must hold 'x y z'." );
            for (int d = 0; d < NO_DIM; ++d) points.push_back( v[d] );
        }
    }
    else
    {
        if ( fileSize % (NO_DIM * sizeof(double)) != 0 )
            throwError( "The '--sample-points' file '", filename,
                        NO_DIM == 2 ? "' is not text and its size is not a multiple of 16 bytes -- expected raw float64 x/y pairs (N x 2, no header)."
                                    : "' is not text and its size is not a multiple of 24 bytes -- expected raw float64 x/y/z triplets (N x 3, no header)." );
        points.resize( fileSize / sizeof(double) );
        file.clear();
        file.seekg( 0, std::ios::beg );
        file.read( reinterpret_cast<char *>(points.data()), fileSize );
        if ( not file )
            throwError( "Failed reading the binary '--sample-points' file '", filename, "'." );
    }

    if ( points.empty() )
        throwError( "No sample points found in '", filename, "'." );
}

void setQueryRecords();                                                 // defined below

} // namespace psPointEvalDetail


using namespace psPointEvalDetail;


bool psPointEvalActive()
{
    return g_pe.ctx.active;
}


void psPointEvalInit(User_options &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );

    g_pe.ctx = Context();  // re-arm cleanly (defensive; DTFE() runs once per process)
    g_pe.ctx.perStream    = userOptions.psPerStream;
    g_pe.ctx.perStreamIds = userOptions.psPerStreamIds;
    g_pe.ctx.ptsDenGrad   = userOptions.psPtsDenGrad;
    g_pe.ctx.ptsVelGrad   = userOptions.psPtsVelGrad;
#ifdef PHASE_SPACE
    g_pe.ctx.psCaustics   = userOptions.psCaustics;
    g_pe.ctx.ptsLagrangian = userOptions.psPtsLagrangian;
#endif
    g_pe.ctx.ptsScalar    = userOptions.psPtsScalar;
    g_pe.ctx.useGeometric = userOptions.psStreamDensityGeometric;
    g_pe.ctx.periodic     = userOptions.periodic;
#ifdef PHASE_SPACE
    // the grid deposit's mass model and velocity weighting, unless '--ps-point-conventions legacy'
    g_pe.ctx.vertexMass     = userOptions.psVertexMass and not userOptions.psPtsLegacyConventions;
    g_pe.ctx.volumeWeighted = userOptions.psVolumeWeighted and not userOptions.psPtsLegacyConventions;
    g_pe.ctx.rhoBar       = double( userOptions.averageDensity );   // PS vertex densities are physical
#else
    g_pe.ctx.rhoBar       = 1.;   // standard vertexDensity() already normalizes by averageDensity
                               // (triangulation.cpp: factor = (NO_DIM+1)/averageDensity), so the
                               // interpolated values are rho/rho_bar as-is
#endif
    for (int d = 0; d < NO_DIM; ++d)
    {
        g_pe.ctx.boxLo[d]  = double( userOptions.region[2*d] );
        g_pe.ctx.boxLen[d] = double( userOptions.region[2*d+1] ) - double( userOptions.region[2*d] );
    }

    // Canonical-image frame of the exact containment test (ps_exact_inside.h): the box length
    // is formed with the SAME Real expression as every periodic-copy generator, so
    // canonicalCoord replays their exact float additions.
    g_pe.ctx.canon.init( userOptions );

    // standard binary, single triangulation: a periodic run owns the primary-box image of each
    // boundary tetrahedron (a composite server sets each partition's own box per request)
    g_pe.ctx.own.on = g_pe.ctx.periodic;
    for (int d = 0; d < NO_DIM; ++d)
    {
        g_pe.ctx.own.lo[d] = g_pe.ctx.boxLo[d];
        g_pe.ctx.own.hi[d] = g_pe.ctx.boxLo[d] + g_pe.ctx.boxLen[d];
    }

    g_pe.ctx.active = true;
    if ( userOptions.psServe )
    {
        message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
                << " point evaluation server armed (--serve): points arrive per request.\n" << MESSAGE::Flush;
        return;     // setQueryPoints() runs per request (psServe)
    }

    std::vector<double> pts;
    readSamplePoints( userOptions.psSamplePointsFile, pts );
    setQueryPoints( std::move(pts), 1 );

    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
            << " point evaluation at " << MESSAGE::cMagenta() << g_pe.ctx.nPoints << MESSAGE::cReset()
            << " sample points from '" << MESSAGE::cBlue() << userOptions.psSamplePointsFile
            << MESSAGE::cReset() << "' (per-stream density: "
            << MESSAGE::cMagenta() << (g_pe.ctx.useGeometric ? "geometric" : "dtfe") << MESSAGE::cReset()
            << (g_pe.ctx.vertexMass ? " with the grids' vertex-share masses" : "")
            << (g_pe.ctx.volumeWeighted ? ", volume-weighted stream means" : "")
            << (g_pe.ctx.perStream ? ", writing per-stream records" : "")
            << (g_pe.ctx.perStreamIds ? " + stream ids" : "")
            << (g_pe.ctx.ptsDenGrad ? ", writing density gradients" : "")
            << (g_pe.ctx.ptsVelGrad ? ", writing velocity gradients" : "")
            << (g_pe.ctx.ptsScalar ? ", writing scalar values" : "")
            << (g_pe.ctx.ptsLagrangian ? ", writing Lagrangian positions" : "") << ").\n" << MESSAGE::Flush;
}


/* Partitioned point evaluation in the standard binary (ps_point_eval.h). The Eulerian tessellation of
   the whole box is cut into the partitions' sub-boxes, each triangulated with its padding, and a tetrahedron is
   OWNED by the partition whose sub-box holds its centroid (vertices min-image wrapped to vertex 0, standardCell).
   The sub-boxes tile the box -- neighbours share the SAME float cut (optimalPartitionSplit forms both sides from one
   expression), half-open [lo, hi) -- so every tetrahedron of the single tessellation has exactly one owner, and the
   exact, tie-consistent containment (pointInTet) hands every point to exactly one of them: the same tetrahedron
   the single-domain run finds, wherever the owner's padding reproduces its neighbourhood (the vertex densities come
   from the owner's own triangulation, so the values agree to float rounding). Only the box's outer faces need care:
   on a periodic box they are the single run's primary-box faces, bit for bit, so the two images of a tetrahedron
   that straddles the seam are told apart exactly as in the single run; on a non-periodic one they open to
   infinity, as the single run keeps every tetrahedron. */
void psPointEvalOwnBox(double const lo[], double const hi[], bool const outerLo[], bool const outerHi[],
                       double ownLo[], double ownHi[])
{
    double const inf = std::numeric_limits<double>::infinity();
    for (int d = 0; d < NO_DIM; ++d)
    {
        ownLo[d] = lo[d];
        ownHi[d] = hi[d];
        if ( outerLo[d] ) ownLo[d] = g_pe.ctx.periodic ? g_pe.ctx.boxLo[d] : -inf;
        // the expression psPointEvalInit gives the single triangulation's own.hi
        if ( outerHi[d] ) ownHi[d] = g_pe.ctx.periodic ? g_pe.ctx.boxLo[d] + g_pe.ctx.boxLen[d] : inf;
    }
}

void psPointEvalSetOwnership(double const *ownLo, double const *ownHi)
{
#ifndef PHASE_SPACE
    if ( ownLo != nullptr and ownHi != nullptr )
    {
        g_pe.ctx.own.on = true;
        for (int d = 0; d < NO_DIM; ++d)
        {
            g_pe.ctx.own.lo[d] = ownLo[d];
            g_pe.ctx.own.hi[d] = ownHi[d];
        }
        g_pe.ctx.ownPartitioned = true;
        return;
    }
    // the single-triangulation default (psPointEvalInit)
    g_pe.ctx.own.on = g_pe.ctx.periodic;
    for (int d = 0; d < NO_DIM; ++d)
    {
        g_pe.ctx.own.lo[d] = g_pe.ctx.boxLo[d];
        g_pe.ctx.own.hi[d] = g_pe.ctx.boxLo[d] + g_pe.ctx.boxLen[d];
    }
#else
    (void)ownLo; (void)ownHi;   // PS-DTFE owns by the Lagrangian centroid (psFilterCell)
#endif
}


namespace psPointEvalDetail {

// Installs a query point set: wraps it into the periodic box, buckets it and (re)arms the
// per-point records and outputs. 'bucketFloor' is the minimum buckets per axis -- the
// sparse-request case (--serve) wants buckets about tetrahedron-sized, so that tetrahedra far
// from every point are dropped by the cheap occupancy test instead of the full per-tet work.
void setQueryPoints(std::vector<double> &&pts, int const bucketFloor)
{
    g_pe.ctx.pos = std::move(pts);
    g_pe.ctx.nPoints = g_pe.ctx.pos.size() / NO_DIM;
    if ( g_pe.ctx.nPoints > size_t(0xFFFFFFFFu) )
        throwError( "Point evaluation supports at most 2^32-1 points per request." );

    // periodic box: wrap the query points inside; the per-tetrahedron test below uses the
    // minimum-image displacement to vertex 0, mirroring the grid deposit's index wrapping
    if ( g_pe.ctx.periodic )
        for (size_t i = 0; i < g_pe.ctx.nPoints; ++i)
            for (int d = 0; d < NO_DIM; ++d)
            {
                double v = std::fmod( g_pe.ctx.pos[i*NO_DIM+d] - g_pe.ctx.boxLo[d], g_pe.ctx.boxLen[d] );
                if ( v < 0. ) v += g_pe.ctx.boxLen[d];
                g_pe.ctx.pos[i*NO_DIM+d] = g_pe.ctx.boxLo[d] + v;
            }

    bucketQueryPoints( bucketFloor );
    setQueryRecords();
}

// Buckets the installed points on a uniform grid (~4 points per bucket, at least 'bucketFloor' per
// axis, capped so empty-bucket overhead stays negligible for small point sets). A composite request
// re-buckets once when a partition walks its cells although the request was bucketed for the index.
void bucketQueryPoints(int const bucketFloor)
{
    {
        int n = int( std::pow( double(g_pe.ctx.nPoints) / 4., 1. / double(NO_DIM) ) ) + 1;   // ~4 points per bucket
        if ( n < bucketFloor ) n = bucketFloor;
        if ( n < 1 )   n = 1;
        if ( n > 256 ) n = 256;
        size_t nTotal = 1;
        for (int d = 0; d < NO_DIM; ++d)
        {
            g_pe.ctx.nB[d]    = n;
            g_pe.ctx.bInvW[d] = double(n) / g_pe.ctx.boxLen[d];
            nTotal *= size_t(n);
        }

        std::vector<size_t> count( nTotal, 0 );
        auto bucketOf = [&](size_t i) -> size_t
        {
            size_t b = 0;
            for (int d = 0; d < NO_DIM; ++d)
            {
                int c = int( std::floor( (g_pe.ctx.pos[i*NO_DIM+d] - g_pe.ctx.boxLo[d]) * g_pe.ctx.bInvW[d] ) );
                if ( c < 0 ) c = 0;                             // non-periodic points may lie outside the box
                if ( c >= g_pe.ctx.nB[d] ) c = g_pe.ctx.nB[d] - 1;
                b = b * size_t(g_pe.ctx.nB[d]) + size_t(c);
            }
            return b;
        };
        for (size_t i = 0; i < g_pe.ctx.nPoints; ++i)
            count[ bucketOf(i) ]++;
        g_pe.ctx.bucketStart.assign( nTotal + 1, 0 );
        for (size_t b = 0; b < nTotal; ++b)
            g_pe.ctx.bucketStart[b+1] = g_pe.ctx.bucketStart[b] + count[b];
        g_pe.ctx.bucketPts.resize( g_pe.ctx.nPoints );
        std::vector<size_t> fill( g_pe.ctx.bucketStart.begin(), g_pe.ctx.bucketStart.end() - 1 );
        for (size_t i = 0; i < g_pe.ctx.nPoints; ++i)
            g_pe.ctx.bucketPts[ fill[ bucketOf(i) ]++ ] = uint32_t(i);
    }
}

// fresh per-point records and outputs for the installed points
void setQueryRecords()
{
    // fresh records and outputs (finalize fills the out* arrays; the ragged ones append)
    std::vector< std::vector<unsigned char> >().swap( g_pe.ctx.recBlocks );
    std::vector<int>().swap( g_pe.ctx.recBlockPart );
    g_pe.ctx.nRecs = 0;
    if ( Env::flag( "DTFE_PTS_FULL_RECORDS" ) )     // test switch: store every member, unpacked
        g_pe.ctx.layout.init( true, true, true, true, true );   // packing (tests/ps_point_eval_check.sh compares)
    else
        g_pe.ctx.layout.init( g_pe.ctx.ptsDenGrad, g_pe.ctx.ptsVelGrad, g_pe.ctx.perStreamIds, g_pe.ctx.ptsScalar, g_pe.ctx.ptsLagrangian );
    g_pe.ctx.outRecords.clear();
    g_pe.ctx.outIds.clear();
    g_pe.ctx.outRecGrad.clear();
    g_pe.ctx.outRecScalar.clear();
    g_pe.ctx.outRecLag.clear();
    g_pe.ctx.outOffsets.clear();
    g_pe.ctx.finalized = false;
}

// True when any query point lies in the bucket range [bLo, bHi] (periodic ranges wrap). The
// cheap test that lets a tetrahedron with no point anywhere near it skip all per-tet work.
// The buckets of a range [bLo, bHi] per axis (inclusive), in NO_DIM dimensions: the last axis runs
// fastest -- the order of the nested 3D loops these replaced -- and a periodic index wraps.
inline bool firstBucket(int bc[NO_DIM], int const bLo[NO_DIM], int const bHi[NO_DIM])
{
    for (int d = 0; d < NO_DIM; ++d)
    {
        if ( bLo[d] > bHi[d] ) return false;
        bc[d] = bLo[d];
    }
    return true;
}
inline bool nextBucket(int bc[NO_DIM], int const bLo[NO_DIM], int const bHi[NO_DIM])
{
    int d = NO_DIM - 1;
    while ( d >= 0 and ++bc[d] > bHi[d] ) { bc[d] = bLo[d]; --d; }
    return d >= 0;
}
inline size_t bucketIndex(int const bc[NO_DIM])
{
    size_t b = 0;
    for (int d = 0; d < NO_DIM; ++d)
    {
        int const w = g_pe.ctx.periodic ? ((bc[d] % g_pe.ctx.nB[d] + g_pe.ctx.nB[d]) % g_pe.ctx.nB[d]) : bc[d];
        b = b * size_t(g_pe.ctx.nB[d]) + size_t(w);
    }
    return b;
}


inline bool anyPointInBuckets(int const bLo[NO_DIM], int const bHi[NO_DIM])
{
    int bc[NO_DIM];
    for (bool more = firstBucket(bc, bLo, bHi); more; more = nextBucket(bc, bLo, bHi))
    {
        size_t const b = bucketIndex(bc);
        if ( g_pe.ctx.bucketStart[b+1] > g_pe.ctx.bucketStart[b] ) return true;
    }
    return false;
}

} // namespace psPointEvalDetail


#ifdef PHASE_SPACE

/* Collects this partition's stream contributions for every query point. Called once per
   Lagrangian partition (and once for a single-triangulation run) after vertexDensity().
   The tessellation is read-only here (apart from the vertex-mass pass above); the cells are split over
   pointEvalThreads() threads, and concurrent workers only synchronize on the short record merge at the end. */
void interpolatePoints_phaseSpace(DT &dt, User_options &userOptions)
{
    if ( not g_pe.ctx.active || dt.number_of_vertices() < NO_DIM + 1 )
        return;

    Box boxCoordinates = userOptions.region;
    double const avgDens = double( userOptions.averageDensity );
    // --ps-vertex-mass under the grid conventions: the vertices' mass-model ratios. A server ran the pass when it
    // built or loaded the tessellation (a request must not repeat a whole-tessellation pass); a batch partition
    // runs it when its walk first meets a tetrahedron with query points in reach, so the many partitions a plane
    // never reaches skip it. The pass runs once and alone (call_once); the walk reads no field it writes before.
    bool const vertexMass = g_pe.ctx.vertexMass;
    bool const vertexMassLazy = vertexMass and not userOptions.psServe;
    std::once_flag vertexMassOnce;

    forEachCellParallel( dt, pointEvalThreads(),
                         [&](Cell_handle &cell, HitSink &hits)
    {
        // The CPU grid deposit's filter chain, shared via ps_cell_filter.h (Real/double
        // arithmetic identical to the grid path, so borderline cells classify identically).
        PSCellGeometry geo;
        if ( !psFilterCell(cell, userOptions, boxCoordinates, true, NULL, geo) )
            return;
        bool const useVolumeRatioDensity = geo.useVolumeRatioDensity;
        Real (&eulerPos)[NO_DIM+1][NO_DIM] = geo.eulerPos;
        double (&Ax)[NO_DIM][NO_DIM] = geo.Ax;
        double const cellAbsDet = geo.cellAbsDet;

        // Eulerian bounding box (+ exact-test slack) -> bucket range (shared helper); a
        // tetrahedron with no query point in reach skips everything below
        double relLo[NO_DIM], relHi[NO_DIM];
        int bLo[NO_DIM], bHi[NO_DIM];
        if ( not tetBucketRange(eulerPos, relLo, relHi, bLo, bHi) ) { return; }
        if ( not anyPointInBuckets(bLo, bHi) ) { return; }
        if ( vertexMassLazy )
            std::call_once( vertexMassOnce, [&]() { prepareVertexMass( dt, userOptions ); } );

        // double-precision inverse for the evaluation (same zero criterion as above)
        double inv[NO_DIM][NO_DIM];
        if ( not inverseNd(Ax, inv) ) { return; }

        // 'geometric' density: rho_bar * V_lag / V_eul (= m_tet * d! / |det Ax|), constant per tet
        double denGeo;
        // --ps-caustics: the same per-tetrahedron mask the grid deposit accumulates (fold parity
        // plus the collapse multiplicity from the constant deformation tensor), carried per stream
        // so the reduction can OR the masks of every tetrahedron containing the point.
        PSCausticClass::CausticMask tetCaustic = 0;
        {
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            denGeo = avgDens * std::fabs(determinant(Lag)) / cellAbsDet;
            // --ps-vertex-mass under the grid conventions: times the mean mass-model ratio of the vertices (above)
            if ( vertexMass )
            {
                double c = 0.;
                for (int v = 0; v <= NO_DIM; ++v)
                    c += cell->vertex(v)->info().volumeAccumulator();
                denGeo *= c / double( NO_DIM + 1 );
            }
            if (denGeo < 0.) denGeo = 0.;
            if ( g_pe.ctx.psCaustics )
                tetCaustic = (PSCausticClass::CausticMask)
                             ( (geo.cellDet > 0. ? PSCausticClass::BIT_PARITY_POS
                                                 : PSCausticClass::BIT_PARITY_NEG)
                               | PSCausticClass::classifyTet( geo.Ax, Lag ) );
        }

        double rho[NO_DIM+1], vel[NO_DIM+1][noVelComp];
        for (int v = 0; v <= NO_DIM; ++v)
        {
            rho[v] = double( cell->vertex(v)->info().density() );
            for (size_t j = 0; j < noVelComp; ++j)
                vel[v][j] = double( cell->vertex(v)->info().velocity(j) );
        }
        // the 'dtfe' vertex densities with the grid's masses: rho_v c_v = M(star)/V_eul(star) (the note above)
        if ( vertexMass )
            for (int v = 0; v <= NO_DIM; ++v)
                rho[v] *= cell->vertex(v)->info().volumeAccumulator();
        // --pts-scalar: the vertices' scalar values; --ps-per-stream-lagrangian: their Lagrangian positions
        // (the triangulation's own points). A tetrahedron is compact in Lagrangian space, so the
        // differences need no wrap; only the interpolated q is wrapped into the box.
        double scal[NO_DIM+1][noScalarComp], lagV[NO_DIM+1][NO_DIM];
        if ( g_pe.ctx.ptsScalar )
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noScalarComp; ++j)
                    scal[v][j] = double( cell->vertex(v)->info().scalar(int(j)) );
        if ( g_pe.ctx.ptsLagrangian )
            for (int v = 0; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    lagV[v][d] = double( cell->vertex(v)->point()[d] );

        // stored (unwrapped) Eulerian vertex positions: the exact containment test canonicalizes
        // these, so a periodic copy and its original resolve to the same vertex
        Real rawPos[NO_DIM+1][NO_DIM];
        for (int v = 0; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
                rawPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);
        TetTest tt;
        double const band1 = fineBand( g_pe.ctx.canon, Ax, cellAbsDet );

        StreamRec rec;
        rec.denGeo = denGeo;
        rec.caustic = tetCaustic;
        for (size_t j = 0; j < noScalarComp; ++j)
            rec.scal[j] = 0.;
        for (int d = 0; d < NO_DIM; ++d)
            rec.lag[d] = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
            rec.ids[v] = 0;
        if ( g_pe.ctx.perStreamIds )
        {
            // sorted ascending: the quadruple is orientation-independent, and identical for the
            // original cell and any periodic image (copies keep the original's ParticleID)
            for (int v = 0; v <= NO_DIM; ++v)
                rec.ids[v] = cell->vertex(v)->info().particleID();
            std::sort( rec.ids, rec.ids + NO_DIM + 1 );
        }
        for (int i = 0; i < NO_DIM; ++i)
            rec.grad[i] = 0.;
        if ( g_pe.ctx.ptsDenGrad && !useVolumeRatioDensity )
        {
            // constant 'dtfe' gradient of this stream: rho(p) = rho_0 + grad . (p - x_0) with
            // bary = (Ax^-1)^T rel  =>  grad_i = sum_v inv[i][v] (rho_{v+1} - rho_0) -- the same
            // affine convention as the linear deposit's denGrad (ps_interpolation.cc). Hull
            // cells (volume-ratio density) have a constant profile: gradient stays 0.
            for (int i = 0; i < NO_DIM; ++i)
                for (int v = 0; v < NO_DIM; ++v)
                    rec.grad[i] += inv[i][v] * (rho[v+1] - rho[0]);
        }
        for (int q = 0; q < NO_DIM*NO_DIM; ++q)
            rec.vgrad[q] = 0.;
        if ( g_pe.ctx.ptsVelGrad )
        {
            // constant velocity gradient of this stream: dv_j/dx_d = sum_v inv[d][v]
            // (vel_{v+1,j} - vel_{0,j}), same affine convention as the density gradient and
            // the same matrix as the grid deposit's velGrad (ps_interpolation.cc
            // matrixMultiplication(posMatInv, dvel)), stored TRANSPOSED: the grid writes it
            // [j*3+d] (field_computation.h velocityGradient), a point [d*3+j] (measured on a rigid
            // rotation). The velocity profile is linear on every
            // kept cell -- hull cells included -- so there is no volume-ratio gate here.
            for (int d = 0; d < NO_DIM; ++d)
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int v = 0; v < NO_DIM; ++v)
                        rec.vgrad[d*NO_DIM + j] += inv[d][v] * (vel[v+1][j] - vel[0][j]);
        }

        int bc[NO_DIM];
        for (bool more = firstBucket(bc, bLo, bHi); more; more = nextBucket(bc, bLo, bHi))
        {
            size_t const b = bucketIndex(bc);

            for (size_t s = g_pe.ctx.bucketStart[b]; s < g_pe.ctx.bucketStart[b+1]; ++s)
            {
                uint32_t const p = g_pe.ctx.bucketPts[s];

                // min-image displacement + bbox pre-test + exact containment (shared helper)
                double bary[NO_DIM];
                if ( not pointInTet(p, eulerPos, rawPos, tt, band1, inv, relLo, relHi, bary) ) continue;

                if ( useVolumeRatioDensity )
                    rec.denDtfe = denGeo;   // hull cell: vertex densities are undefined (0)
                else
                {
                    double dd = rho[0];
                    for (int v = 0; v < NO_DIM; ++v)
                        dd += bary[v] * (rho[v+1] - rho[0]);
                    rec.denDtfe = (dd > 0.) ? dd : 0.;   // a point ON a face can graze negative by rounding
                }
                for (size_t j = 0; j < noVelComp; ++j)
                {
                    double vv = vel[0][j];
                    for (int v = 0; v < NO_DIM; ++v)
                        vv += bary[v] * (vel[v+1][j] - vel[0][j]);
                    rec.vel[j] = vv;
                }
                // the scalar and the Lagrangian coordinate follow the SAME affine map as the
                // velocity: q(x) = q_0 + sum_v bary_v (q_{v+1} - q_0) inverts x(q) on this stream
                if ( g_pe.ctx.ptsScalar )
                    for (size_t j = 0; j < noScalarComp; ++j)
                    {
                        double ss = scal[0][j];
                        for (int v = 0; v < NO_DIM; ++v)
                            ss += bary[v] * (scal[v+1][j] - scal[0][j]);
                        rec.scal[j] = ss;
                    }
                if ( g_pe.ctx.ptsLagrangian )
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        double qq = lagV[0][d];
                        for (int v = 0; v < NO_DIM; ++v)
                            qq += bary[v] * (lagV[v+1][d] - lagV[0][d]);
                        if ( g_pe.ctx.periodic )
                        {
                            double w = std::fmod( qq - g_pe.ctx.boxLo[d], g_pe.ctx.boxLen[d] );
                            if ( w < 0. ) w += g_pe.ctx.boxLen[d];
                            if ( w >= g_pe.ctx.boxLen[d] ) w -= g_pe.ctx.boxLen[d];
                            qq = g_pe.ctx.boxLo[d] + w;
                        }
                        rec.lag[d] = qq;
                    }

                hits.push_back( std::make_pair( p, rec ) );
            }
        }
    } );   // end cell loop
}

#else // !PHASE_SPACE

/* Standard-DTFE worker: the tessellation itself is Eulerian, so a query point has exactly
   one containing tetrahedron (stream count = 0/1 coverage; the exact containment test makes
   that hold for points ON shared faces too, and psPointEvalFinalize keeps a 1-record cap
   as a safety net). Reuses the PS candidate search verbatim (bucketed points,
   tetBucketRange/pointInTet);
   the estimator differences:
     (a) vertex positions come from the Delaunay points themselves, min-image-wrapped to
         vertex 0 under --periodic exactly like the PS Eulerian wrap (ps_cell_filter.h);
     (b) --periodic pads the box with particle COPIES, so a boundary tetrahedron appears as
         several images: exactly ONE is kept (Eulerian centroid inside the primary box) --
         the standard-build analogue of the PS Lagrangian-centroid partition ownership;
     (c) there is no geometric (V_lag/V_eul) density variant: denGeo aliases the 'dtfe'
         value, so the shared sort/reduction machinery behaves identically. */

// (standardCell, the per-cell filter it shares with the composite server: ps_point_eval_detail.h)
void interpolatePoints_standardOwn(DT &dt, User_options &userOptions, OwnBox const &own)
{
    if ( not g_pe.ctx.active || dt.number_of_vertices() < NO_DIM + 1 )
        return;

    (void)userOptions;   // box geometry/periodicity already captured in g_pe.ctx at init

    forEachCellParallel( dt, pointEvalThreads(),
                         [&](Cell_handle &cell, HitSink &hits)
    {
        double rho[NO_DIM+1], vel[NO_DIM+1][noVelComp];
        Real eulerPos[NO_DIM+1][NO_DIM], rawPos[NO_DIM+1][NO_DIM];
        if ( not standardCell( cell, own, rho, vel, eulerPos, rawPos ) ) { return; }

        // Eulerian edge matrix. Unlike the PS worker (whose stream set must equal the grid
        // deposit's, degeneracy drops included), NO tetrahedron is dropped for being small or
        // flat: the Eulerian tessellation tiles the box, so every drop is a hole -- a point in
        // it, or a query AT a vertex whose tie-break lands in it, would come out uncovered.
        // Containment is decided exactly regardless of shape (fineBand grows as the tet
        // flattens, handing the decision to the exact tiers), and a guard-free inverse gives
        // the interpolation weights; only an exactly zero-volume cell holds no points.
        double relLo[NO_DIM], relHi[NO_DIM];
        int bLo[NO_DIM], bHi[NO_DIM];
        if ( not tetBucketRange(eulerPos, relLo, relHi, bLo, bHi) ) { return; }
        if ( not anyPointInBuckets(bLo, bHi) ) { return; }   // no query point in reach

        double Ax[NO_DIM][NO_DIM];
        for (int v = 0; v < NO_DIM; ++v)
            for (int i = 0; i < NO_DIM; ++i)
                Ax[v][i] = double(eulerPos[v+1][i]) - double(eulerPos[0][i]);
        double inv[NO_DIM][NO_DIM];
        double const absDet = std::fabs( inverse3x3Unguarded(Ax, inv) );
        if ( not (absDet > 0.) || not std::isfinite(absDet) ) { return; }
        TetTest tt;
        double const band1 = fineBand( g_pe.ctx.canon, Ax, absDet );

        StreamRec rec;
        rec.caustic = 0;                    // --ps-caustics is PS-only (no fold parity to report here)
        for (int v = 0; v <= NO_DIM; ++v)
            rec.ids[v] = 0;                 // --ps-per-stream-ids is PS-only (rejected at parsing)
        for (int d = 0; d < NO_DIM; ++d)
            rec.lag[d] = 0.;                // --ps-per-stream-lagrangian is PS-only
        for (size_t j = 0; j < noScalarComp; ++j)
            rec.scal[j] = 0.;
        double scal[NO_DIM+1][noScalarComp];
        if ( g_pe.ctx.ptsScalar )
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noScalarComp; ++j)
                    scal[v][j] = double( cell->vertex(v)->info().scalar(int(j)) );
        for (int i = 0; i < NO_DIM; ++i)
            rec.grad[i] = 0.;
        if ( g_pe.ctx.ptsDenGrad )
        {
            // constant gradient of the linear DTFE profile, same affine convention as PS:
            // rho(p) = rho_0 + grad . (p - x_0), grad_i = sum_v inv[i][v] (rho_{v+1} - rho_0)
            for (int i = 0; i < NO_DIM; ++i)
                for (int v = 0; v < NO_DIM; ++v)
                    rec.grad[i] += inv[i][v] * (rho[v+1] - rho[0]);
        }
        for (int q = 0; q < NO_DIM*NO_DIM; ++q)
            rec.vgrad[q] = 0.;
        if ( g_pe.ctx.ptsVelGrad )
        {
            // the containing tetrahedron's constant velocity gradient, same [d*3+j] layout
            // as the PS worker (exactly one stream per point here, so the 'mean' is it)
            for (int d = 0; d < NO_DIM; ++d)
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int v = 0; v < NO_DIM; ++v)
                        rec.vgrad[d*NO_DIM + j] += inv[d][v] * (vel[v+1][j] - vel[0][j]);
        }

        int bc[NO_DIM];
        for (bool more = firstBucket(bc, bLo, bHi); more; more = nextBucket(bc, bLo, bHi))
        {
            size_t const b = bucketIndex(bc);

            for (size_t s = g_pe.ctx.bucketStart[b]; s < g_pe.ctx.bucketStart[b+1]; ++s)
            {
                uint32_t const p = g_pe.ctx.bucketPts[s];

                double bary[NO_DIM];
                if ( not pointInTet(p, eulerPos, rawPos, tt, band1, inv, relLo, relHi, bary) ) continue;

                double dd = rho[0];
                for (int v = 0; v < NO_DIM; ++v)
                    dd += bary[v] * (rho[v+1] - rho[0]);
                rec.denDtfe = (dd > 0.) ? dd : 0.;   // a point ON a face can graze negative by rounding
                rec.denGeo  = rec.denDtfe;           // no geometric variant on the Eulerian tessellation
                for (size_t j = 0; j < noVelComp; ++j)
                {
                    double vv = vel[0][j];
                    for (int v = 0; v < NO_DIM; ++v)
                        vv += bary[v] * (vel[v+1][j] - vel[0][j]);
                    rec.vel[j] = vv;
                }
                if ( g_pe.ctx.ptsScalar )
                    for (size_t j = 0; j < noScalarComp; ++j)
                    {
                        double ss = scal[0][j];
                        for (int v = 0; v < NO_DIM; ++v)
                            ss += bary[v] * (scal[v+1][j] - scal[0][j]);
                        rec.scal[j] = ss;
                    }

                hits.push_back( std::make_pair( p, rec ) );
            }
        }
    } );   // end cell loop
}

void interpolatePoints_standard(DT &dt, User_options &userOptions)
{
    interpolatePoints_standardOwn( dt, userOptions, g_pe.ctx.own );
}

#endif // PHASE_SPACE


namespace psPointEvalDetail {

// psPointEvalFinalize's counts, for its report
struct FinalizeCounts
{
    size_t covered = 0, multi = 0;
    size_t maxStreams = 0;
    size_t overlapped = 0;                  // standard binary: points inside more than one owned tetrahedron
};

// Finalize, phase 1: the stored entries grouped by point (first: N+1 offsets, where: one locator per record).
static void finalizeGroupRecords(size_t const N, size_t const E, std::vector<uint64_t> &first, std::vector<uint64_t> &where)
{
    for (std::vector<unsigned char> const &blk : g_pe.ctx.recBlocks)
        for (size_t k = 0; k < blk.size() / E; ++k)
        {
            uint32_t pt;
            std::memcpy( &pt, blk.data() + k * E, sizeof(uint32_t) );
            ++first[ size_t(pt) + 1 ];
        }
    for (size_t i = 0; i < N; ++i) first[i+1] += first[i];
    {
        std::vector<uint64_t> next( first.begin(), first.end() - 1 );
        for (size_t b = 0; b < g_pe.ctx.recBlocks.size(); ++b)
        {
            std::vector<unsigned char> const &blk = g_pe.ctx.recBlocks[b];
            for (size_t k = 0; k < blk.size() / E; ++k)
            {
                uint32_t pt;
                std::memcpy( &pt, blk.data() + k * E, sizeof(uint32_t) );
                where[ next[pt]++ ] = ( uint64_t(b) << 32 ) | uint64_t(k);
            }
        }
    }
}

// Finalize, phase 2: point i's records, unpacked into recs (the buffer is reused) and sorted into the strict order.
static void finalizeLoadPoint(size_t const i, std::vector<uint64_t> const &first, std::vector<uint64_t> const &where,
                              size_t const E, bool const geo, std::vector<StreamRec> &recs)
{
    // strict total order (selected density descending, full tie-break) -> the outputs are
    // deterministic and independent of the partition split and thread completion order
    auto recLess = [geo](StreamRec const &a, StreamRec const &b) -> bool
    {
        double const a1 = geo ? a.denGeo : a.denDtfe;
        double const b1 = geo ? b.denGeo : b.denDtfe;
        if (a1 != b1) return a1 > b1;
        if (a.denGeo  != b.denGeo)  return a.denGeo  > b.denGeo;
        if (a.denDtfe != b.denDtfe) return a.denDtfe > b.denDtfe;
        for (size_t j = 0; j < noVelComp; ++j)
            if (a.vel[j] != b.vel[j]) return a.vel[j] < b.vel[j];
        for (int i = 0; i < NO_DIM; ++i)        // all 0 unless --pts-den-grad
            if (a.grad[i] != b.grad[i]) return a.grad[i] < b.grad[i];
        for (int q = 0; q < NO_DIM*NO_DIM; ++q) // all 0 unless --pts-vel-grad
            if (a.vgrad[q] != b.vgrad[q]) return a.vgrad[q] < b.vgrad[q];
        for (int v = 0; v <= NO_DIM; ++v)       // all 0 unless --ps-per-stream-ids; only breaks exact
            if (a.ids[v] != b.ids[v]) return a.ids[v] < b.ids[v];   // den+vel ties, so the ids file
                                                                    // is partition-invariant too
        for (size_t j = 0; j < noScalarComp; ++j)   // all 0 unless --pts-scalar
            if (a.scal[j] != b.scal[j]) return a.scal[j] < b.scal[j];
        for (int d = 0; d < NO_DIM; ++d)             // all 0 unless --ps-per-stream-lagrangian
            if (a.lag[d] != b.lag[d]) return a.lag[d] < b.lag[d];
        return false;
    };

    recs.resize( size_t( first[i+1] - first[i] ) );
    for (uint64_t r = first[i]; r < first[i+1]; ++r)
    {
        uint64_t const loc = where[r];
        g_pe.ctx.layout.unpackOne( g_pe.ctx.recBlocks[ size_t(loc >> 32) ].data() + (loc & 0xFFFFFFFFu) * E
                                + sizeof(uint32_t), recs[ size_t(r - first[i]) ] );
    }
    std::sort( recs.begin(), recs.end(), recLess );
}

// Finalize, phase 3: point i's outputs from its sorted records -- the stream count, the caustic mask (OR of the
// streams' masks), the moments and the per-stream records.
static void finalizeReducePoint(size_t const i, std::vector<StreamRec> &recs, bool const geo, double const invRhoBar,
                                FinalizeCounts &counts)
{
#ifndef PHASE_SPACE
    // Eulerian tessellation: exactly one containing tet per point, which the exact
    // containment test guarantees even ON shared faces. Kept as a safety net: should a
    // second record ever appear, both interpolate the same continuous field, so keep the
    // deterministic sorted-first one and the coverage count stays an honest 0/1.
    if ( recs.size() > 1 )
    {
        recs.resize( 1 );
        ++counts.overlapped;                // said in the report (phase 4)
    }
#endif

    size_t const nS = recs.size();
    g_pe.ctx.outStreams[i] = int32_t( nS );
    // --ps-caustics: OR the masks of every tetrahedron covering this point. OR is commutative
    // and idempotent, so the result is independent of the partition and thread schedule -- the
    // same guarantee the grid '.caustic' has, and the reason a point ON a fold reports both
    // parities exactly as the enclosing grid cell does.
    if ( g_pe.ctx.psCaustics )
    {
        int mask = 0;
        for (size_t s = 0; s < nS; ++s) mask |= int( recs[s].caustic );
        g_pe.ctx.outCaustic[i] = int32_t( mask );
    }
    if (nS > 0) ++counts.covered;
    if (nS > 1) ++counts.multi;
    if (nS > counts.maxStreams) counts.maxStreams = nS;

    // moments in sorted order (the psDeferNormalization pattern: sums are linear across
    // partitions; the density-weighted means/dispersion are formed exactly once, here)
    double W = 0., mom1[noVelComp] = {0.}, mom2[noDispComp] = {0.};
    double momG[NO_DIM*NO_DIM] = {0.};
    double momS[noScalarComp] = {0.};
    for (size_t s = 0; s < nS; ++s)
    {
        double const den = geo ? recs[s].denGeo : recs[s].denDtfe;
        W += den;
        for (size_t j = 0; j < noVelComp; ++j)
            mom1[j] += den * recs[s].vel[j];
        if ( g_pe.ctx.ptsScalar )
            for (size_t j = 0; j < noScalarComp; ++j)
                momS[j] += den * recs[s].scal[j];
        if ( g_pe.ctx.ptsVelGrad )
            for (int q = 0; q < NO_DIM*NO_DIM; ++q)
                momG[q] += den * recs[s].vgrad[q];
        size_t c = 0;
        for (int a = 0; a < NO_DIM; ++a)
            for (int b = a; b < NO_DIM; ++b)
                mom2[c++] += den * recs[s].vel[a] * recs[s].vel[b];
    }
    g_pe.ctx.outDen[i] = W * invRhoBar;
    if ( g_pe.ctx.ptsDenGrad )
    {
        // gradient of the total ('dtfe') density: sum of the streams' constant gradients,
        // accumulated in the same sorted order as the moments (deterministic across splits)
        double gsum[NO_DIM] = {0.};
        for (size_t s = 0; s < nS; ++s)
            for (int d = 0; d < NO_DIM; ++d)
                gsum[d] += recs[s].grad[d];
        for (int d = 0; d < NO_DIM; ++d)
            g_pe.ctx.outDenGrad[i*NO_DIM + d] = gsum[d] * invRhoBar;
    }
    if ( g_pe.ctx.volumeWeighted and nS > 0 )
    {
        // --ps-volume-weighted under the grid conventions: the grid weights each stream's
        // velocity, gradient and scalar by the Eulerian volume it fills in the cell, and at a point every one
        // of its streams fills the point -- equal shares, so the plain mean over the streams (the sorted
        // order, as above). The DISPERSION stays what the grid keeps it: the mass-weighted second moment about
        // the mass-weighted mean (the f-weighted CBE moment; see '--ps-volume-weighted').
        double const invN = 1. / double( nS );
        double sum1[noVelComp] = {0.}, sumG[NO_DIM*NO_DIM] = {0.}, sumS[noScalarComp] = {0.};
        for (size_t s = 0; s < nS; ++s)
        {
            for (size_t j = 0; j < noVelComp; ++j)
                sum1[j] += recs[s].vel[j];
            if ( g_pe.ctx.ptsVelGrad )
                for (int q = 0; q < NO_DIM*NO_DIM; ++q)
                    sumG[q] += recs[s].vgrad[q];
            if ( g_pe.ctx.ptsScalar )
                for (size_t j = 0; j < noScalarComp; ++j)
                    sumS[j] += recs[s].scal[j];
        }
        for (size_t j = 0; j < noVelComp; ++j)
            g_pe.ctx.outVel[i*noVelComp+j] = sum1[j] * invN;
        if ( g_pe.ctx.ptsVelGrad )
            for (int q = 0; q < NO_DIM*NO_DIM; ++q)
                g_pe.ctx.outVelGrad[i*NO_DIM*NO_DIM + q] = sumG[q] * invN;
        if ( g_pe.ctx.ptsScalar )
            for (size_t j = 0; j < noScalarComp; ++j)
                g_pe.ctx.outScalar[i*noScalarComp + j] = sumS[j] * invN;
        if ( nS > 1 and W > 0. )
        {
            double const invW = 1. / W;
            double vm[noVelComp];
            for (size_t j = 0; j < noVelComp; ++j)
                vm[j] = mom1[j] * invW;
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                {
                    double s2 = mom2[c] * invW - vm[a] * vm[b];
                    if (a == b && s2 < 0.) s2 = 0.;   // variance: clamp FP-noise negatives
                    g_pe.ctx.outDisp[i*noDispComp + c] = s2;
                    ++c;
                }
        }
    }
    else if ( W > 0. )
    {
        double const invW = 1. / W;
        for (size_t j = 0; j < noVelComp; ++j)
            g_pe.ctx.outVel[i*noVelComp+j] = mom1[j] * invW;
        if ( g_pe.ctx.ptsVelGrad )
            for (int q = 0; q < NO_DIM*NO_DIM; ++q)
                g_pe.ctx.outVelGrad[i*NO_DIM*NO_DIM + q] = momG[q] * invW;
        // the scalar: the same density-weighted stream mean as the velocity
        if ( g_pe.ctx.ptsScalar )
            for (size_t j = 0; j < noScalarComp; ++j)
                g_pe.ctx.outScalar[i*noScalarComp + j] = momS[j] * invW;
        // a single stream has zero dispersion by definition; skip the moment difference
        // (it would only return ~1e-9 cancellation noise)
        if ( nS > 1 )
        {
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                {
                    double s2 = mom2[c] * invW
                              - g_pe.ctx.outVel[i*noVelComp+a] * g_pe.ctx.outVel[i*noVelComp+b];
                    if (a == b && s2 < 0.) s2 = 0.;   // variance: clamp FP-noise negatives
                    g_pe.ctx.outDisp[i*noDispComp + c] = s2;
                    ++c;
                }
        }
    }

    if ( g_pe.ctx.perStream )
    {
        for (size_t s = 0; s < nS; ++s)
        {
            g_pe.ctx.outRecords.push_back( (geo ? recs[s].denGeo : recs[s].denDtfe) * invRhoBar );
            for (size_t j = 0; j < noVelComp; ++j)
                g_pe.ctx.outRecords.push_back( recs[s].vel[j] );
            if ( g_pe.ctx.perStreamIds )
                for (int v = 0; v <= NO_DIM; ++v)
                    g_pe.ctx.outIds.push_back( recs[s].ids[v] );
            if ( g_pe.ctx.ptsDenGrad )
                for (int d = 0; d < NO_DIM; ++d)
                    g_pe.ctx.outRecGrad.push_back( recs[s].grad[d] * invRhoBar );
            if ( g_pe.ctx.ptsScalar )
                for (size_t j = 0; j < noScalarComp; ++j)
                    g_pe.ctx.outRecScalar.push_back( recs[s].scal[j] );
            if ( g_pe.ctx.ptsLagrangian )
                for (int d = 0; d < NO_DIM; ++d)
                    g_pe.ctx.outRecLag.push_back( recs[s].lag[d] );
        }
        g_pe.ctx.outOffsets[i+1] = uint64_t( g_pe.ctx.outOffsets[i] ) + uint64_t( nS );
    }
}

// Finalize, phase 4: the report -- the coverage line and, in the standard binary, the ownership warning.
static void finalizeReport(User_options const &userOptions, size_t const N, FinalizeCounts const &counts)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " point evaluation -- "
            << MESSAGE::cMagenta() << counts.covered << "/" << N << MESSAGE::cReset() << " points covered, "
            << MESSAGE::cMagenta() << counts.multi << MESSAGE::cReset() << " multi-stream, max streams "
            << MESSAGE::cMagenta() << counts.maxStreams << MESSAGE::cReset() << "."
            << ( counts.covered < N ? "  (uncovered points lie outside the tessellation or in empty regions)" : "" )
            << "\n" << MESSAGE::Flush;
#ifndef PHASE_SPACE
    // The safety net of phase 3 warns: the Eulerian tessellation gives every point exactly one
    // containing tetrahedron (exact containment), and on a periodic box every point has one. A second record, or
    // none on a periodic box, means that a tetrahedron was OWNED twice or not at all: by partitions (or a composite
    // server's) whose padding does not reproduce the tessellation near their faces, or -- the same in a single run --
    // by the two images of a tetrahedron across a periodic box face, whose centroids come from periodic copies that
    // the box shift rounds (measured on a box at 131020-131120 Mpc: 6 doubles and 15 holes in 17600 points, the
    // partitioned run identical; on a box starting at 0 the -L copies are exact and the +L ones round at ~ulp(L)).
    bool const holes = g_pe.ctx.periodic and counts.covered < N;
    if ( counts.overlapped > 0 or holes )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        if ( counts.overlapped > 0 )
            warning << counts.overlapped << " of " << N << " points lie inside more than one owned tetrahedron (each keeps the "
                    << "first in the deterministic order). ";
        if ( holes )
            warning << ( N - counts.covered ) << " of " << N << " points are covered by none, although the box is periodic. ";
        warning << "Either " << ( g_pe.ctx.ownPartitioned ? "the partitions' '--padding'" : "'--padding'" ) << " is too small "
                << "(increase it), or, for a few points next to a periodic box face, the rounding of the periodic copies made a "
                << "tetrahedron across the face owned by both of its images or by neither.\n" << MESSAGE::EndWarning;
    }
#endif
}

} // namespace psPointEvalDetail


void psPointEvalFinalize(User_options const &userOptions)
{
    if ( not g_pe.ctx.active || g_pe.ctx.finalized )
        return;

    size_t const N = g_pe.ctx.nPoints;
    bool const geo = g_pe.ctx.useGeometric;
    double const invRhoBar = g_pe.ctx.rhoBar > 0. ? 1. / g_pe.ctx.rhoBar : 1.;

    // the outputs, the records grouped by point (phase 1), the per-stream outputs -- in this order (the peak memory)
    g_pe.ctx.outDen.assign( N, 0. );
    g_pe.ctx.outVel.assign( N * noVelComp, 0. );
    g_pe.ctx.outDisp.assign( N * noDispComp, 0. );
    g_pe.ctx.outStreams.assign( N, 0 );
    if ( g_pe.ctx.psCaustics )
        g_pe.ctx.outCaustic.assign( N, 0 );
    size_t const totalStreams = g_pe.ctx.nRecs;

    // Group the stored entries by point: a counting sort into 'first' (N+1 offsets) and 'where'
    // (one locator per record: block << 32 | entry within the block). The order of a point's
    // records here is the arrival order; the strict sort of phase 2 makes the output independent of it.
    size_t const E = sizeof(uint32_t) + g_pe.ctx.layout.bytes;
    std::vector<uint64_t> first( N + 1, 0 ), where( totalStreams );
    finalizeGroupRecords( N, E, first, where );
    if ( g_pe.ctx.perStream )
    {
        g_pe.ctx.outOffsets.assign( N + 1, 0 );
        g_pe.ctx.outRecords.reserve( totalStreams * (1 + noVelComp) );
        if ( g_pe.ctx.perStreamIds )
            g_pe.ctx.outIds.reserve( totalStreams * (NO_DIM + 1) );
        if ( g_pe.ctx.ptsDenGrad )
            g_pe.ctx.outRecGrad.reserve( totalStreams * NO_DIM );
        if ( g_pe.ctx.ptsScalar )
            g_pe.ctx.outRecScalar.reserve( totalStreams * noScalarComp );
        if ( g_pe.ctx.ptsLagrangian )
            g_pe.ctx.outRecLag.reserve( totalStreams * NO_DIM );
    }
    if ( g_pe.ctx.ptsScalar )
        g_pe.ctx.outScalar.assign( N * noScalarComp, 0. );
    if ( g_pe.ctx.ptsDenGrad )
        g_pe.ctx.outDenGrad.assign( N * NO_DIM, 0. );
    if ( g_pe.ctx.ptsVelGrad )
        g_pe.ctx.outVelGrad.assign( N * NO_DIM*NO_DIM, 0. );

    // every point: its sorted records (phase 2) reduced to its outputs (phase 3)
    FinalizeCounts counts;
    std::vector<StreamRec> recs;            // one point's records, unpacked (the buffer is reused)
    for (size_t i = 0; i < N; ++i)
    {
        finalizeLoadPoint( i, first, where, E, geo, recs );
        finalizeReducePoint( i, recs, geo, invRhoBar, counts );
    }
    std::vector< std::vector<unsigned char> >().swap( g_pe.ctx.recBlocks );
    std::vector<int>().swap( g_pe.ctx.recBlockPart );
    g_pe.ctx.nRecs = 0;
    g_pe.ctx.finalized = true;

    finalizeReport( userOptions, N, counts );   // phase 4
}


void psPointEvalWriteOutputs(User_options const &userOptions)
{
    if ( not g_pe.ctx.active || not g_pe.ctx.finalized )
        return;

    MESSAGE::Message message( userOptions.verboseLevel );

    auto writeRaw = [&userOptions](std::string const &ext, void const *data, size_t bytes,
                                   char const *what)
    {
        std::string const name = userOptions.outputFilename + ext;
        // staged as '<file>.part.<pid>' (or rescued) with the grids, renamed by main.cpp (output_staging.h)
        std::string path = outputStagePath( name );
        std::ofstream f( path.c_str(), std::ios::binary | std::ios::trunc );
        if ( not f )
        {
            path = outputStageRetry( name );
            if ( not path.empty() ) f.open( path.c_str(), std::ios::binary | std::ios::trunc );
        }
        if ( not f.is_open() )
            throwError( "Cannot open the point-evaluation output file '", name, "'." );
        f.write( reinterpret_cast<char const *>(data), bytes );
        if ( not f )
            throwError( "Failed writing the point-evaluation output file '", name, "'." );
        MESSAGE::Message m( userOptions.verboseLevel );
        m << "Writing the " << what << " to the file '" << MESSAGE::cBlue() << name
          << MESSAGE::cReset() << "' ... Done.\n" << MESSAGE::Flush;
    };

    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
            << " writing the point-evaluation outputs (raw binary, see README):\n" << MESSAGE::Flush;
    writeRaw( ".pts_den",     g_pe.ctx.outDen.data(),     g_pe.ctx.outDen.size()     * sizeof(double),
              "point densities (float64, rho/rho_bar)" );
    writeRaw( ".pts_vel",     g_pe.ctx.outVel.data(),     g_pe.ctx.outVel.size()     * sizeof(double),
              NO_DIM == 2 ? "point mean velocities (float64 x 2)" : "point mean velocities (float64 x 3)" );
    writeRaw( ".pts_velDisp", g_pe.ctx.outDisp.data(),    g_pe.ctx.outDisp.size()    * sizeof(double),
              NO_DIM == 2 ? "point velocity dispersion tensors (float64 x 3, xx xy yy)"
                          : "point velocity dispersion tensors (float64 x 6, xx xy xz yy yz zz)" );
    writeRaw( ".pts_streams", g_pe.ctx.outStreams.data(), g_pe.ctx.outStreams.size() * sizeof(int32_t),
              "point stream counts (int32)" );
    if ( g_pe.ctx.psCaustics )
        writeRaw( ".pts_caustic", g_pe.ctx.outCaustic.data(), g_pe.ctx.outCaustic.size() * sizeof(int32_t),
                  "point caustic masks (int32; bits as in '.causticClass', both parity bits = fold)" );
    if ( g_pe.ctx.ptsDenGrad )
        writeRaw( ".pts_denGrad", g_pe.ctx.outDenGrad.data(), g_pe.ctx.outDenGrad.size() * sizeof(double),
                  NO_DIM == 2 ? "point density gradients (float64 x 2, d(rho/rho_bar)/dx_i, 'dtfe' profile)"
                              : "point density gradients (float64 x 3, d(rho/rho_bar)/dx_i, 'dtfe' profile)" );
    if ( g_pe.ctx.ptsVelGrad )
        writeRaw( ".pts_velGrad", g_pe.ctx.outVelGrad.data(), g_pe.ctx.outVelGrad.size() * sizeof(double),
                  g_pe.ctx.volumeWeighted
                  ? ( NO_DIM == 2 ? "point velocity gradients (float64 x 4, [d*2+j] = dv_j/dx_d, equal-volume stream mean)"
                                  : "point velocity gradients (float64 x 9, [d*3+j] = dv_j/dx_d, equal-volume stream mean)" )
                  : ( NO_DIM == 2 ? "point velocity gradients (float64 x 4, [d*2+j] = dv_j/dx_d, density-weighted stream mean)"
                                  : "point velocity gradients (float64 x 9, [d*3+j] = dv_j/dx_d, density-weighted stream mean)" ) );
    if ( g_pe.ctx.ptsScalar )
        writeRaw( ".pts_scalar", g_pe.ctx.outScalar.data(), g_pe.ctx.outScalar.size() * sizeof(double),
                  g_pe.ctx.volumeWeighted ? "point scalar values (float64, equal-volume mean over the streams)"
                                       : "point scalar values (float64, density-weighted mean over the streams)" );
    if ( g_pe.ctx.perStream )
    {
        writeRaw( ".pts_stream_offsets", g_pe.ctx.outOffsets.data(),
                  g_pe.ctx.outOffsets.size() * sizeof(uint64_t),
                  "per-stream record offsets (uint64, N+1)" );
        writeRaw( ".pts_stream_records", g_pe.ctx.outRecords.data(),
                  g_pe.ctx.outRecords.size() * sizeof(double),
                  NO_DIM == 2 ? "per-stream records (float64 x 3: density, vx, vy; density-descending)"
                              : "per-stream records (float64 x 4: density, vx, vy, vz; density-descending)" );
        if ( g_pe.ctx.perStreamIds )
            writeRaw( ".pts_stream_ids", g_pe.ctx.outIds.data(),
                      g_pe.ctx.outIds.size() * sizeof(uint64_t),
                      NO_DIM == 2 ? "per-stream identities (uint64 x 3: sorted Lagrangian-vertex ParticleIDs)"
                                  : "per-stream identities (uint64 x 4: sorted Lagrangian-vertex ParticleIDs)" );
        if ( g_pe.ctx.ptsDenGrad )
            writeRaw( ".pts_stream_dengrad", g_pe.ctx.outRecGrad.data(),
                      g_pe.ctx.outRecGrad.size() * sizeof(double),
                      NO_DIM == 2 ? "per-stream density gradients (float64 x 2, 'dtfe' profile)"
                                  : "per-stream density gradients (float64 x 3, 'dtfe' profile)" );
        if ( g_pe.ctx.ptsScalar )
            writeRaw( ".pts_stream_scalar", g_pe.ctx.outRecScalar.data(),
                      g_pe.ctx.outRecScalar.size() * sizeof(double),
                      "per-stream scalar values (float64)" );
        if ( g_pe.ctx.ptsLagrangian )
            writeRaw( ".pts_stream_lagpos", g_pe.ctx.outRecLag.data(),
                      g_pe.ctx.outRecLag.size() * sizeof(double),
                      NO_DIM == 2 ? "per-stream Lagrangian coordinates (float64 x 2, the initial position of each stream's matter)"
                                  : "per-stream Lagrangian coordinates (float64 x 3, the initial position of each stream's matter)" );
    }
}


/* ---- '--resume' with '--sample-points' ---------------------------------------------------------------------

   The planes are made by the grid runs themselves, so without checkpoints a killed run (TNG50-3-Dark z = 0:
   2.5 h, partitions 5-10 in flight) would restart from zero. The point evaluation's only accumulators are the
   stream records, and each record belongs to the one partition that owns its tetrahedron; one pool filled by
   every partition in flight could not say which records go with the partitions a checkpoint lists, so every
   block carries its partition (a sink serves one triangulation, so a block never mixes two), and a checkpoint
   (partition_checkpoint.h) writes, BEFORE its manifest, one file per merged partition not saved yet:

       <checkpoint dir>/points.<i>   a text header -- the magic line, 'hash' (the checkpoint key's), 'partition',
                                     'points' (the point count), 'entry_bytes' (uint32 point index + the packed
                                     record, RecLayout), 'records', 'header-end' -- then the entries as stored

   written once each (a temp file, fsync, rename). '--resume' reads the files of the partitions its manifest lists
   back into the store and skips those partitions. psPointEvalFinalize sorts each point's records into a strict
   order before any sum, so it does not matter which records were read back and which computed: the '.pts_*'
   outputs are byte-identical to an uninterrupted run's (tests/ps_resume_points_check.sh). The files hold the
   bytes the run keeps in memory anyway, once: a 4096^2 plane with --pts-vel-grad at z = 0 (~4 streams per point,
   118 B per entry) writes ~8 GB over the whole run, inside the checkpoint's 5% time throttle and room rule.
   The STANDARD binary's partition loop tags its records the same way (DTFE.cpp; one record per
   point there, from the partition owning its tetrahedron), so a standard '--sample-points --partition' run is
   checkpointed and resumed by the same files ('DTFE-PS-POINTS 1' names the format, shared by both binaries; the
   key hash in each header keeps a standard run's files from being read into a PS run and the other way round). */

namespace psPointEvalDetail {

static char const *const POINTS_MAGIC = "DTFE-PS-POINTS 1";

static std::string pointsPath(std::string const &dir, size_t pi)
{
    return dir + "/points." + std::to_string( pi );
}

// the (data, bytes) of every stored block of each partition to save, taken under the store's lock: another
// partition's flush may move the block VECTORS (the outer vector grows), never their buffers, and a saved
// partition's blocks are complete (it merged after its cell walk flushed every sink)
static std::vector< std::vector< std::pair<unsigned char const *, size_t> > >
blocksToSave(std::vector<char> const &merged, std::vector<char> const &saved)
{
    std::vector< std::vector< std::pair<unsigned char const *, size_t> > > parts( merged.size() );
    std::lock_guard<std::mutex> lock( g_pe.mergeMutex );
    for (size_t k = 0; k < g_pe.ctx.recBlocks.size(); ++k)
    {
        int const p = g_pe.ctx.recBlockPart[k];
        if ( p < 0 or size_t(p) >= merged.size() or not merged[size_t(p)] or saved[size_t(p)] ) continue;
        parts[size_t(p)].push_back( std::make_pair( g_pe.ctx.recBlocks[k].data(), g_pe.ctx.recBlocks[k].size() ) );
    }
    return parts;
}

} // namespace psPointEvalDetail


void psPointEvalSetPartition(int const partition)
{
    g_curPartition = partition;
}

double psPointEvalUnsavedBytes(std::vector<char> const &merged, std::vector<char> const &saved)
{
    if ( not g_pe.ctx.active ) return 0.;
    double bytes = 0.;
    for (auto const &part : blocksToSave( merged, saved ))
        for (auto const &b : part) bytes += double( b.second );
    return bytes;
}

bool psPointEvalSavePartitions(std::string const &dir, std::string const &keyHash, std::vector<char> const &merged,
                               std::vector<char> &saved, std::string &why)
{
    if ( not g_pe.ctx.active ) return true;
    size_t const E = sizeof(uint32_t) + g_pe.ctx.layout.bytes;
    auto const parts = blocksToSave( merged, saved );
    for (size_t pi = 0; pi < merged.size(); ++pi)
    {
        if ( not merged[pi] or saved[pi] ) continue;
        size_t bytes = 0;
        for (auto const &b : parts[pi]) bytes += b.second;
        std::ostringstream h;
        h << POINTS_MAGIC << "\nhash " << keyHash << "\npartition " << pi << "\npoints " << g_pe.ctx.nPoints
          << "\nentry_bytes " << E << "\nrecords " << bytes / E << "\nheader-end\n";
        std::string const header = h.str();
        std::string const path = pointsPath( dir, pi );
        std::string const tmp = path + ".tmp." + std::to_string( long( ::getpid() ) );
        FILE *f = std::fopen( tmp.c_str(), "wb" );
        if ( not f ) { why = "cannot write '" + tmp + "': " + std::strerror( errno ); return false; }
        bool ok = std::fwrite( header.data(), 1, header.size(), f ) == header.size();
        for (auto const &b : parts[pi])
            if ( ok ) ok = std::fwrite( b.first, 1, b.second, f ) == b.second;
        ok = ok and std::fflush( f ) == 0 and ::fsync( ::fileno( f ) ) == 0;
        int const err = errno;
        ok = ( std::fclose( f ) == 0 ) and ok;
        if ( not ok or std::rename( tmp.c_str(), path.c_str() ) != 0 )
        {
            why = "writing the point records '" + tmp + "' failed: " + std::strerror( ok ? errno : err );
            std::remove( tmp.c_str() );
            return false;
        }
        saved[pi] = 1;
    }
    return true;
}

bool psPointEvalLoadPartitions(std::string const &dir, std::string const &keyHash, std::vector<char> const &merged,
                               std::string &why)
{
    if ( not g_pe.ctx.active ) return true;
    size_t const E = sizeof(uint32_t) + g_pe.ctx.layout.bytes;
    size_t const CHUNK = std::max<size_t>( 1, HitSink::FLUSH_BYTES / E ) * E;   // blocks the size the sinks hand over
    for (size_t pi = 0; pi < merged.size(); ++pi)
    {
        if ( not merged[pi] ) continue;
        std::string const path = pointsPath( dir, pi );
        std::ifstream in( path, std::ios::binary );
        if ( not in ) { why = "the point records of partition " + std::to_string( pi ) + " ('" + path + "') are missing"; return false; }
        std::string line;
        std::string hash;
        long long part = -1, points = -1, entry = -1, records = -1;
        bool ended = false;
        if ( not std::getline( in, line ) or line != POINTS_MAGIC ) { why = "'" + path + "' is not a point-record file"; return false; }
        while ( std::getline( in, line ) )
        {
            if ( line == "header-end" ) { ended = true; break; }
            std::istringstream ls( line );
            std::string tag;
            ls >> tag;
            if ( tag == "hash" ) ls >> hash;
            else if ( tag == "partition" ) ls >> part;
            else if ( tag == "points" ) ls >> points;
            else if ( tag == "entry_bytes" ) ls >> entry;
            else if ( tag == "records" ) ls >> records;
        }
        if ( not ended ) { why = "the header of '" + path + "' is incomplete"; return false; }
        if ( hash != keyHash or part != (long long)pi )
        { why = "'" + path + "' belongs to another run or partition (key hash " + hash + ", this run " + keyHash + ")"; return false; }
        if ( points != (long long)g_pe.ctx.nPoints or entry != (long long)E or records < 0 )
        {
            why = "'" + path + "' holds " + std::to_string( points ) + " points with " + std::to_string( entry ) + "-byte "
                  "entries, this run " + std::to_string( g_pe.ctx.nPoints ) + " and " + std::to_string( E ) + " (another sample-point "
                  "file or record layout, e.g. DTFE_PTS_FULL_RECORDS)";
            return false;
        }
        long long const start = (long long)in.tellg();
        struct stat st;
        if ( ::stat( path.c_str(), &st ) != 0 or (unsigned long long)st.st_size != (unsigned long long)start + (unsigned long long)records * E )
        { why = "'" + path + "' is not the size its header lists (cut or damaged)"; return false; }
        size_t left = size_t( records ) * E;
        while ( left > 0 )
        {
            size_t const n = std::min( left, CHUNK );
            std::vector<unsigned char> blk( n );
            if ( not in.read( reinterpret_cast<char *>( blk.data() ), std::streamsize( n ) ) )
            { why = "reading '" + path + "' failed"; return false; }
            for (size_t k = 0; k < n; k += E)
            {
                uint32_t pt;
                std::memcpy( &pt, blk.data() + k, sizeof(uint32_t) );
                if ( size_t( pt ) >= g_pe.ctx.nPoints ) { why = "'" + path + "' names a point beyond this run's"; return false; }
            }
            std::lock_guard<std::mutex> lock( g_pe.mergeMutex );
            g_pe.ctx.recBlocks.push_back( std::move( blk ) );
            g_pe.ctx.recBlockPart.push_back( int( pi ) );
            g_pe.ctx.nRecs += n / E;
            left -= n;
        }
    }
    return true;
}

void psPointEvalRemoveUnsaved(std::string const &dir, std::vector<char> const &saved)
{
    DIR *d = ::opendir( dir.c_str() );
    if ( not d ) return;
    std::vector<std::string> stale;
    while ( dirent *e = ::readdir( d ) )
    {
        std::string const n = e->d_name;
        if ( n.rfind( "points.", 0 ) != 0 or n.find( ".tmp." ) != std::string::npos ) continue;
        char *end = nullptr;
        unsigned long long const pi = std::strtoull( n.c_str() + 7, &end, 10 );
        if ( end == n.c_str() + 7 or *end != '\0' ) continue;
        if ( pi >= saved.size() or not saved[size_t(pi)] ) stale.push_back( dir + "/" + n );
    }
    ::closedir( d );
    for (auto const &s : stale) std::remove( s.c_str() );
}
