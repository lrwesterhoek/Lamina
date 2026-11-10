/* '--serve' (both binaries): the point-evaluation server over stdin/stdout -- one tessellation, or a COMPOSITE
   of partition tessellations (--serve + --partition, or when one tessellation would not fit) -- and the partitions'
   occupancy maps, which a batch '--grid-window' run reads too. Every request is answered by the '--sample-points'
   machinery of ps_point_eval.cc (ps_point_eval_detail.h), so an answer is bit-identical to a batch run's. The
   protocol is documented below. */

#include "ps_point_eval_detail.h"   // the point evaluation's context, records and workers

#include "tessellation_cache.h" // composite --serve: partitions reloaded from '--tessellation-cache'
#include "../auto_tune.h"       // composite --serve: the memory budget that sizes the resident set
#include "../env_knobs.h"
#include "../run_mode.h"        // dtfeHardwareThreads: the run mode's thread budget (the loaders)


// '--tessellation-cache' load or build of a tessellation plus its vertex densities (triangulation.cpp,
// the first half of DTFE_interpolation); true when it came from the cache. Composite --serve only.
bool buildTessellation(std::vector<Particle_data> *p, User_options &userOptions, DT &dt);


/* ---- --serve: interactive point evaluation over stdin/stdout ------------------------------

   The tessellation is built once; every request is then answered from it by exactly the
   --sample-points machinery (same exact containment, same threads, same reduction), so a
   request's answer is bit-identical to a '--sample-points' run on the same points. Client:
   python/dtfelib/estimator.py. All integers little-endian, doubles native (IEEE-754):

   server -> client, once, when the tessellation is ready:
       char[8]  "DTFESRV2"
       uint32   flags: bit0 PS-DTFE build, bit1 --ps-per-stream, bit2 --ps-per-stream-ids,
                       bit3 --pts-den-grad, bit4 --pts-vel-grad, bit5 --ps-caustics,
                       bit6 --ps-stream-density geometric, bit7 periodic,
                       bit8 --pts-scalar, bit9 --ps-per-stream-lagrangian, bit10 composite (partitioned)
       uint32   S           scalar components per value (--pts-scalar; 0 otherwise)
       uint64   tessellation vertex count (a composite: summed over its partitions)
       double   region[6]   xlo xhi ylo yhi zlo zhi (the box the points are wrapped into)
       uint32   partitions  (1 for a single tessellation)
       uint32   resident    at most this many partitions are held in memory at once
   client -> server, per request:
       uint64   n           number of points; 0 (or closing stdin) ends the session
       double   xyz[3n]
   server -> client, per request -- the '.pts_*' arrays of a --sample-points run, in order:
       char[8]  "DTFEPTS1",  uint64 n,  uint64 R (per-stream records; 0 without --ps-per-stream)
       double den[n], vel[3n], velDisp[6n];  int32 streams[n]
       int32  caustic[n]        (--ps-caustics)       double denGrad[3n]  (--pts-den-grad)
       double velGrad[9n]       (--pts-vel-grad)      uint64 offsets[n+1] (--ps-per-stream)
       double records[4R]       (--ps-per-stream)        uint64 ids[4R]      (--ps-per-stream-ids)
       double recGrad[3R]       (--ps-per-stream + --pts-den-grad)
       double scalar[S n]       (--pts-scalar)        double recScalar[S R] (--pts-scalar + --ps-per-stream)
       double recLag[3R]        (--ps-per-stream-lagrangian)
   a request the server rejects (non-finite coordinates, too many points, a partition whose
   cached tessellation vanished) is answered with
       char[8]  "DTFEERR1",  uint64 length,  char message[length]
   and the session continues.

   Version 2 has the scalar/Lagrangian arrays and the partition counts that version 1 lacks; its
   header differs too, so a version-1 client stops at the handshake instead of misreading. */

namespace psPointEvalDetail {

// The server's cell index: every finite cell registered in the uniform buckets its Eulerian
// bounding box overlaps -- min-image wrapped and padded exactly like tetBucketRange, so a cell
// containing a point is always registered in that point's bucket. Built once; a small request
// then lists only the cells of its points' buckets (deduplicated by a per-request stamp) and
// the workers run on that list instead of the whole tessellation. Identical per-cell work on
// a superset of the containing cells, so the answers stay bit-identical to a full walk.
struct ServeIndex
{
    int    n[NO_DIM];
    double invW[NO_DIM];
    std::vector<Cell_handle> cells;     // every finite cell
    std::vector<size_t>   start;        // CSR over buckets: members[start[b] .. start[b+1])
    std::vector<uint32_t> members;      // indices into 'cells'
    std::vector<uint32_t> stamp;        // per cell: the last request that listed it
    uint32_t request = 0;
};

/* ---- composite '--serve' (--serve + --partition) -------------------------------------------

   CosmoDTFE's CompositeEstimator routes each query to one of several local estimators; here the
   same idea serves boxes that do not fit in memory as ONE tessellation. The partitions are the
   run's usual ones -- Lagrangian sub-boxes in PS-DTFE (a stream is owned by exactly one partition
   via the Lagrangian-centroid test in psFilterCell), Eulerian sub-boxes in the standard binary
   (a tetrahedron is owned by the partition whose region holds its centroid, OwnBox) -- so the
   union of the partitions' owned tetrahedra is exactly the single tessellation's, and a request
   answered partition by partition is bit-identical to one answered by a single server (the
   records are sorted before any reduction).

   Routing: every partition keeps an OCCUPANCY map, one bit per bucket of a coarse OCC_N^NO_DIM grid over
   the box, set wherever one of its owned tetrahedra's Eulerian bounding boxes reaches (min-image
   wrapped and padded like tetBucketRange). A tetrahedron containing a point always covers that
   point's bucket, so a request only visits the partitions whose map holds one of its points.

   Out of core: with '--tessellation-cache' each partition's tessellation lives on disk (the batch
   runs' own per-partition cache files: the descriptors match, so a batch run with the cache warms
   the server and vice versa) and its occupancy map beside it ('<cache file>.occ', stamped with the
   same descriptor). At most 'resident' partitions are held in memory; the least recently used one
   is dropped when another must be loaded. A session whose inputs are unchanged registers every
   partition from these files without building or even loading anything. Without the cache all
   partitions stay in memory (an in-memory composite: faster to build, since the partitions are
   triangulated in parallel, but not smaller than one tessellation). */

static const int OCC_N = 64;                 // occupancy buckets per axis (32 KB per partition)
// A partition's cell index is built only once the partition has answered this many requests by
// walking its cells while in memory. Measured on TNG100-3-Dark's 14.3M-vertex partitions: the index
// takes 18-22 s to build, a walk for a 512^2 zoom (or a click) 1.1-1.4 s, a request through the index
// 0.2-0.9 s -- so building it on the first request would make the first zoom or click in a region 3-15x
// slower, and a partition evicted again (a region needing more partitions than fit) would rebuild it every time.
static const uint64_t INDEX_AFTER_WALKS = 8;
// Memory of a resident partition, for the automatic resident set: its tessellation per vertex (the
// server's RSS with 8 resident 3.65M-vertex partitions: 664 B/vertex; the same figures as
// auto_tune.h autoTuneServe) and, before one is built, its cell index per vertex (the index's own
// arrays are counted once it exists).
#ifdef PHASE_SPACE
static const double SERVE_TRI_BYTES_PER_VERTEX = 666.;
#else
static const double SERVE_TRI_BYTES_PER_VERTEX = 648.;
#endif
static const double SERVE_INDEX_BYTES_PER_VERTEX = 192.;
static const size_t OCC_WORDS = ( NO_DIM == 2 ? size_t(OCC_N) * OCC_N : size_t(OCC_N) * OCC_N * OCC_N ) / 64;
static const char OCC_MAGIC[8] = { 'D','T','F','E','O','C','C','1' };
static const char OCC_ALPHA_TAG[4] = { 'O','C','C','A' };

struct ServePart
{
    User_options opts;                       // this partition's options: its ownership + cache identity
    OwnBox own;                              // standard binary: the Eulerian box it owns
    std::vector<uint64_t> occ;               // occupancy bits (OCC_WORDS words); empty = owns nothing
    std::unique_ptr<DT> dt;                  // resident tessellation, or null (in the cache only)
    std::unique_ptr<ServeIndex> index;       // its cell index (see INDEX_AFTER_WALKS)
    uint64_t walks = 0;                      // requests answered by walking it since it was loaded
    double indexBytes = 0.;                  // its cell index's arrays, while it has one
    uint64_t lastUse = 0;
    uint64_t nVertices = 0;
    bool cached = false;                     // a valid cache file exists: may be dropped and reloaded
};

struct Composite
{
    std::vector<ServePart> parts;
    int verbose = 1;
    std::mutex logMutex;
    int done = 0;
    uint64_t clock = 0;
    size_t maxResident = 1;                  // cap on resident CACHED partitions (uncached ones are pinned)
    bool byBytes = false;                    // automatic: as many as fit in budgetBytes (else maxResident)
    double budgetBytes = 0.;
    double indexPerVertex = SERVE_INDEX_BYTES_PER_VERTEX;   // the last built index's, per vertex
};

// The servers' state, ONE instance (g_serve): the protocol channel, '--serve-progress' and the composite server.
// (What they share with the point evaluation is ps_point_eval_detail.h's g_pe.)
struct ServeState
{
    int       fd = -1;               // the protocol channel: a private duplicate of the original stdout
    bool      progress = false;      // '--serve-progress' (serveProgress)
    Composite comp;                  // the composite server's partitions and resident set
};
static ServeState g_serve;

} // namespace psPointEvalDetail


using namespace psPointEvalDetail;

void psServeRedirectStdout()
{
    std::cout.flush();
    std::fflush( stdout );
    int const fd = ::dup( STDOUT_FILENO );
    if ( fd < 0 || ::dup2( STDERR_FILENO, STDOUT_FILENO ) < 0 )
    {
        std::fprintf( stderr, "--serve: cannot set up the protocol channel (dup/dup2 failed).\n" );
        std::exit( 1 );
    }
    g_serve.fd = fd;
}

namespace psPointEvalDetail {

// A vanished client (EPIPE) ends the server quietly: nobody is left to report to.
static void serveWrite(void const *data, size_t bytes)
{
    char const *p = static_cast<char const *>(data);
    while ( bytes > 0 )
    {
        ssize_t const w = ::write( g_serve.fd, p, bytes );
        if ( w < 0 )
        {
            if ( errno == EINTR ) continue;
            std::_Exit( 0 );
        }
        p += w;
        bytes -= size_t(w);
    }
}

// false on end of input (the client closed stdin), which ends the session
static bool serveRead(void *data, size_t bytes)
{
    char *p = static_cast<char *>(data);
    while ( bytes > 0 )
    {
        ssize_t const r = ::read( STDIN_FILENO, p, bytes );
        if ( r < 0 && errno == EINTR ) continue;
        if ( r <= 0 ) return false;
        p += r;
        bytes -= size_t(r);
    }
    return true;
}

template <typename T>
static void serveWriteVec(std::vector<T> const &v)
{
    if ( not v.empty() ) serveWrite( v.data(), v.size() * sizeof(T) );
}

static void serveError(std::string const &msg)
{
    uint64_t const len = msg.size();
    serveWrite( "DTFEERR1", 8 );
    serveWrite( &len, sizeof(len) );
    serveWrite( msg.data(), msg.size() );
}

inline Real storedEulerian(Cell_handle const &c, int v, int d)
{
#ifdef PHASE_SPACE
    return c->vertex(v)->info().eulerianPosition(d);
#else
    return Real( c->vertex(v)->point()[d] );
#endif
}

inline int serveBucketOf(ServeIndex const &I, double x, int d)
{
    int b = int( std::floor( (x - g_pe.ctx.boxLo[d]) * I.invW[d] ) );
    if ( b < 0 ) b = 0;                      // wrapped periodic points never are; others clamp
    if ( b >= I.n[d] ) b = I.n[d] - 1;
    return b;
}

// the cell's bucket range (unwrapped for periodic axes; callers wrap); false = no overlap
static bool serveRange(ServeIndex const &I, Cell_handle const &c, int lo[NO_DIM], int hi[NO_DIM])
{
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const x0 = double( storedEulerian(c, 0, d) );
        double eMin = x0, eMax = x0;
        for (int v = 1; v <= NO_DIM; ++v)
        {
            double x = double( storedEulerian(c, v, d) );
            if ( g_pe.ctx.periodic )
            {
                double const r = x - x0;
                if ( r >  0.5 * g_pe.ctx.boxLen[d] ) x -= g_pe.ctx.boxLen[d];
                if ( r < -0.5 * g_pe.ctx.boxLen[d] ) x += g_pe.ctx.boxLen[d];
            }
            eMin = std::min( eMin, x );
            eMax = std::max( eMax, x );
        }
        double const margin = 1.e-5 * (eMax - eMin) + 1.e-12 + 2. * g_pe.ctx.canon.epsAbs;
        lo[d] = int( std::floor( (eMin - margin - g_pe.ctx.boxLo[d]) * I.invW[d] ) );
        hi[d] = int( std::floor( (eMax + margin - g_pe.ctx.boxLo[d]) * I.invW[d] ) );
        if ( g_pe.ctx.periodic )
        {
            if ( hi[d] - lo[d] + 1 >= I.n[d] ) { lo[d] = 0; hi[d] = I.n[d] - 1; }
        }
        else
        {
            if ( lo[d] < 0 ) lo[d] = 0;
            if ( hi[d] >= I.n[d] ) hi[d] = I.n[d] - 1;
            if ( lo[d] > hi[d] ) return false;
        }
    }
    return true;
}

static void buildServeIndex(DT &dt, ServeIndex &I)
{
    for (PSFiniteCellIterator it = PS_PE_FINITE_BEGIN(dt); it != PS_PE_FINITE_END(dt); ++it)
        I.cells.push_back( it );
    size_t const T = I.cells.size();
    int nb = int( std::pow( double(T) / 8., 1. / double(NO_DIM) ) );    // buckets about a few tetrahedra wide
    nb = std::max( 1, std::min( nb, 256 ) );
    size_t nTot = 1;
    for (int d = 0; d < NO_DIM; ++d)
    {
        I.n[d] = nb;
        I.invW[d] = double(nb) / g_pe.ctx.boxLen[d];
        nTot *= size_t(nb);
    }

    // two passes (count, fill) over the same per-cell bucket ranges
    I.start.assign( nTot + 1, 0 );
    std::vector<size_t> fill;
    for (int pass = 0; pass < 2; ++pass)
    {
        for (size_t ci = 0; ci < T; ++ci)
        {
            int lo[NO_DIM], hi[NO_DIM];
            if ( not serveRange( I, I.cells[ci], lo, hi ) ) continue;
            forEachBucketIn( lo, hi, I.n, g_pe.ctx.periodic, [&](size_t b) {
                if ( pass == 0 ) ++I.start[b+1];
                else             I.members[ fill[b]++ ] = uint32_t(ci);
            } );
        }
        if ( pass == 0 )
        {
            for (size_t b = 0; b < nTot; ++b) I.start[b+1] += I.start[b];
            I.members.resize( I.start[nTot] );
            fill.assign( I.start.begin(), I.start.end() - 1 );
        }
    }
    I.stamp.assign( T, 0u );
}

// the deduplicated cells registered in the buckets of the current request's points
static void serveCandidates(ServeIndex &I, std::vector<Cell_handle> &list)
{
    if ( ++I.request == 0 )     // stamp wrap-around: reset and start over
    {
        std::fill( I.stamp.begin(), I.stamp.end(), 0u );
        I.request = 1;
    }
    list.clear();
    for (size_t i = 0; i < g_pe.ctx.nPoints; ++i)
    {
        size_t b = 0;
        for (int d = 0; d < NO_DIM; ++d)
            b = b * size_t(I.n[d]) + size_t( serveBucketOf( I, g_pe.ctx.pos[i*NO_DIM+d], d ) );
        for (size_t m = I.start[b]; m < I.start[b+1]; ++m)
        {
            uint32_t const ci = I.members[m];
            if ( I.stamp[ci] == I.request ) continue;
            I.stamp[ci] = I.request;
            list.push_back( I.cells[ci] );
        }
    }
}

} // namespace psPointEvalDetail

namespace psPointEvalDetail {

// The handshake (protocol above): written once the server can answer.
static void serveHandshake(uint64_t const nVert, uint32_t const nPartitions, uint32_t const nResident)
{
    uint32_t flags = 0;
#ifdef PHASE_SPACE
    flags |= 1u;
#endif
    if ( g_pe.ctx.perStream )     flags |= 1u << 1;
    if ( g_pe.ctx.perStreamIds )  flags |= 1u << 2;
    if ( g_pe.ctx.ptsDenGrad )    flags |= 1u << 3;
    if ( g_pe.ctx.ptsVelGrad )    flags |= 1u << 4;
    if ( g_pe.ctx.psCaustics )    flags |= 1u << 5;
    if ( g_pe.ctx.useGeometric )  flags |= 1u << 6;
    if ( g_pe.ctx.periodic )      flags |= 1u << 7;
    if ( g_pe.ctx.ptsScalar )     flags |= 1u << 8;
    if ( g_pe.ctx.ptsLagrangian ) flags |= 1u << 9;
    if ( nPartitions > 1 )     flags |= 1u << 10;
    if ( NO_DIM == 2 )         flags |= 1u << 11;   // a 2D server: points, the region and every vector carry 2 components
    uint32_t const nScalar = g_pe.ctx.ptsScalar ? uint32_t(noScalarComp) : 0u;
    double region[2*NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        region[2*d]   = g_pe.ctx.boxLo[d];
        region[2*d+1] = g_pe.ctx.boxLo[d] + g_pe.ctx.boxLen[d];
    }
    serveWrite( "DTFESRV2", 8 );
    serveWrite( &flags, sizeof(flags) );
    serveWrite( &nScalar, sizeof(nScalar) );
    serveWrite( &nVert, sizeof(nVert) );
    serveWrite( region, sizeof(region) );
    serveWrite( &nPartitions, sizeof(nPartitions) );
    serveWrite( &nResident, sizeof(nResident) );
}

// '--serve-progress': plain progress lines on stderr (the server's log), whatever '--verbose' says --
// MESSAGE::Message prints nothing at the launcher's verbose 1, which left its progress bar blind
static void serveProgress(char const *fmt, ...) __attribute__((format(printf, 1, 2)));
static void serveProgress(char const *fmt, ...)
{
    if ( not g_serve.progress ) return;
    va_list ap;
    va_start( ap, fmt );
    std::vfprintf( stderr, fmt, ap );
    va_end( ap );
    std::fflush( stderr );
}

/* The request loop shared by the single and the composite server. 'evaluate' collects the streams
   of the installed query points (the workers append to g_pe.ctx's records) and returns an empty
   string, or an error message -- the request is then refused and the session continues.
   'indexCells' = the cell count a request is compared against: fewer points than 1/64 of it go
   through the servers' cell indices ('useIndex'), more walk the tessellation(s) with buckets about
   'bucketFloor' per axis, so cells far from every point still fail the cheap occupancy test. */
template <typename Evaluate>
[[noreturn]] void serveLoop(User_options &userOptions, size_t const indexCells, int const bucketFloor,
                            Evaluate evaluate)
{
    size_t nRequests = 0;
    for (;;)
    {
        uint64_t n = 0;
        if ( not serveRead( &n, sizeof(n) ) || n == 0 )
            break;
        if ( n > uint64_t(0xFFFFFFFFu) )
        {
            // drain the payload so the stream stays in sync, then refuse
            std::vector<char> sink( size_t(1) << 20 );
            uint64_t left = n * NO_DIM * sizeof(double);
            bool ok = true;
            while ( left > 0 && ok )
            {
                size_t const chunk = size_t( std::min<uint64_t>( left, sink.size() ) );
                ok = serveRead( sink.data(), chunk );
                left -= chunk;
            }
            if ( not ok ) break;
            serveError( "a request holds at most 2^32-1 points; split it" );
            continue;
        }

        std::vector<double> pts( size_t(n) * NO_DIM );
        if ( not serveRead( pts.data(), pts.size() * sizeof(double) ) )
            break;
        bool finite = true;
        for (double const x : pts)
            if ( not std::isfinite(x) ) { finite = false; break; }
        if ( not finite )
        {
            serveError( "non-finite point coordinate (NaN or inf) in the request" );
            continue;
        }

        bool useIndex = n * 64 < uint64_t(indexCells);
        if ( int const forced = Env::tristate( "DTFE_SERVE_INDEX" ); forced >= 0 )   // 0 / 1: force the walk / the cell index
            useIndex = forced == 1;
        setQueryPoints( std::move(pts), useIndex ? 1 : bucketFloor );
        std::string const failure = evaluate( useIndex );
        g_pe.cellList = nullptr;
        if ( not failure.empty() )
        {
            serveError( failure );
            continue;
        }
        psPointEvalFinalize( userOptions );

        uint64_t const R = g_pe.ctx.perStream ? uint64_t( g_pe.ctx.outRecords.size() / (1 + noVelComp) ) : 0;
        serveWrite( "DTFEPTS1", 8 );
        serveWrite( &n, sizeof(n) );
        serveWrite( &R, sizeof(R) );
        serveWriteVec( g_pe.ctx.outDen );
        serveWriteVec( g_pe.ctx.outVel );
        serveWriteVec( g_pe.ctx.outDisp );
        serveWriteVec( g_pe.ctx.outStreams );
        if ( g_pe.ctx.psCaustics )   serveWriteVec( g_pe.ctx.outCaustic );
        if ( g_pe.ctx.ptsDenGrad )   serveWriteVec( g_pe.ctx.outDenGrad );
        if ( g_pe.ctx.ptsVelGrad )   serveWriteVec( g_pe.ctx.outVelGrad );
        if ( g_pe.ctx.perStream )
        {
            serveWriteVec( g_pe.ctx.outOffsets );
            serveWriteVec( g_pe.ctx.outRecords );
            if ( g_pe.ctx.perStreamIds ) serveWriteVec( g_pe.ctx.outIds );
            if ( g_pe.ctx.ptsDenGrad )   serveWriteVec( g_pe.ctx.outRecGrad );
        }
        if ( g_pe.ctx.ptsScalar )
        {
            serveWriteVec( g_pe.ctx.outScalar );
            if ( g_pe.ctx.perStream ) serveWriteVec( g_pe.ctx.outRecScalar );
        }
        if ( g_pe.ctx.ptsLagrangian ) serveWriteVec( g_pe.ctx.outRecLag );
        ++nRequests;
    }

    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " server done ("
            << nRequests << " requests answered).\n" << MESSAGE::Flush;
    std::cout.flush();
    std::fflush( stdout );
    std::fflush( stderr );
    ::close( g_serve.fd );
    // _Exit: nothing is left to write, and tearing down a large triangulation (plus the
    // static state of the libraries) would only delay the client's clean shutdown
    std::_Exit( 0 );
}

} // namespace psPointEvalDetail

[[noreturn]] void psServe(DT &dt, User_options &userOptions)
{
    if ( g_serve.fd < 0 )
        throwError( "'--serve': the protocol channel is not set up (main() must call psServeRedirectStdout() before any output)." );
    if ( not g_pe.ctx.active )
        throwError( "'--serve': the point-evaluation context was not initialized." );

    // the cell index answers small requests without touching the rest of the tessellation;
    // large ones walk it all (with buckets about a few tetrahedra wide, so cells far from
    // every point still fail the cheap occupancy test)
    ServeIndex index;
    {
        Timer t;
        t.start();
        buildServeIndex( dt, index );
        printComputationTime( &t, &userOptions, "point-evaluation server cell index" );
    }
    size_t const nCells = index.cells.size();
    int const bucketFloor = int( std::pow( double(nCells) / 16., 1. / double(NO_DIM) ) );
    std::vector<Cell_handle> candidates;
#ifdef PHASE_SPACE
    // --ps-vertex-mass under the grid conventions: the vertex degrees and star volumes, once for every request
    if ( g_pe.ctx.vertexMass )
        prepareVertexMass( dt, userOptions );
#endif

    serveHandshake( uint64_t( dt.number_of_vertices() ), 1u, 1u );
    {
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
                << " serving point-evaluation requests (" << dt.number_of_vertices()
                << " vertices); close stdin to stop.\n" << MESSAGE::Flush;
    }

    g_serve.progress = userOptions.psServeProgress;
    serveLoop( userOptions, nCells, bucketFloor, [&](bool const useIndex) -> std::string
    {
        auto const t0 = std::chrono::steady_clock::now();
        serveProgress( "[request 1/1] one tessellation: in memory\n" );
        if ( useIndex )
        {
            serveCandidates( index, candidates );
            g_pe.cellList = &candidates;
        }
#ifdef PHASE_SPACE
        interpolatePoints_phaseSpace( dt, userOptions );
#else
        interpolatePoints_standard( dt, userOptions );
#endif
        serveProgress( "[request done] %zu points, 1 of 1 partitions, 0 loaded from the cache in 0.0 s, %.2f s in all\n",
                       g_pe.ctx.nPoints, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
        return std::string();
    } );
}


namespace psPointEvalDetail {

inline int occBucket(double x, int d)
{
    double const w = double(OCC_N) / g_pe.ctx.boxLen[d];
    int b = int( std::floor( (x - g_pe.ctx.boxLo[d]) * w ) );
    if ( g_pe.ctx.periodic )
    {
        b %= OCC_N;
        if ( b < 0 ) b += OCC_N;
    }
    else
    {
        if ( b < 0 ) b = 0;
        if ( b >= OCC_N ) b = OCC_N - 1;
    }
    return b;
}

// The bucket frame of an occupancy map: the box it covers (the run's region), its periodicity and
// the canonical-frame slack. The server takes it from its point-evaluation context; a batch
// partition run (its map writer, and the --grid-window partition skip) builds the same from the
// partition's options -- the region is the box in both, from the moment the header is read.
struct OccFrame
{
    double boxLo[NO_DIM], boxLen[NO_DIM];
    bool   periodic = false;
    double epsAbs = 0.;
};

static OccFrame occFrameFromCtx()
{
    OccFrame F;
    for (int d = 0; d < NO_DIM; ++d) { F.boxLo[d] = g_pe.ctx.boxLo[d]; F.boxLen[d] = g_pe.ctx.boxLen[d]; }
    F.periodic = g_pe.ctx.periodic;
    F.epsAbs = g_pe.ctx.canon.epsAbs;
    return F;
}

static OccFrame occFrameFrom(User_options const &o)
{
    OccFrame F;
    for (int d = 0; d < NO_DIM; ++d)
    {
        F.boxLo[d]  = double( o.region.coords[2*d] );
        F.boxLen[d] = double( o.region.coords[2*d+1] ) - double( o.region.coords[2*d] );
    }
    F.periodic = o.periodic;
    User_options c = o;                // Frame::init takes a non-const reference
    Frame canon;
    canon.init( c );
    F.epsAbs = canon.epsAbs;
    return F;
}

// Marks the occupancy buckets of one tetrahedron's (wrapped) Eulerian vertices -- the same bbox
// and slack as tetBucketRange, so no bucket the exact test could accept a point in is missed.
static void markOccupancy(std::vector<uint64_t> &occ, Real const eulerPos[NO_DIM+1][NO_DIM], OccFrame const &F)
{
    int lo[NO_DIM], hi[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double eMin = double(eulerPos[0][d]), eMax = eMin;
        for (int v = 1; v <= NO_DIM; ++v)
        {
            double const c = double(eulerPos[v][d]);
            eMin = std::min( eMin, c );
            eMax = std::max( eMax, c );
        }
        double const margin = 1.e-5 * (eMax - eMin) + 1.e-12 + 2. * F.epsAbs;
        double const w = double(OCC_N) / F.boxLen[d];
        lo[d] = int( std::floor( (eMin - margin - F.boxLo[d]) * w ) );
        hi[d] = int( std::floor( (eMax + margin - F.boxLo[d]) * w ) );
        if ( F.periodic )
        {
            if ( hi[d] - lo[d] + 1 >= OCC_N ) { lo[d] = 0; hi[d] = OCC_N - 1; }
        }
        else
        {
            lo[d] = std::max( 0, std::min( lo[d], OCC_N - 1 ) );
            hi[d] = std::max( 0, std::min( hi[d], OCC_N - 1 ) );
        }
    }
    int const nOcc[NO_DIM] = { OCC_N, OCC_N
#if NO_DIM == 3
                               , OCC_N
#endif
                             };
    forEachBucketIn( lo, hi, nOcc, true, [&](size_t b) { occ[b >> 6] |= uint64_t(1) << (b & 63); } );
}

// The partition's occupancy map from its OWNED tetrahedra (the workers' own filters).
static void computeOccupancy(DT &dt, ServePart &P, OccFrame const &F)
{
    P.occ.assign( OCC_WORDS, 0 );
    for (PSFiniteCellIterator it = PS_PE_FINITE_BEGIN(dt); it != PS_PE_FINITE_END(dt); ++it)
    {
        Cell_handle cell = it;
#ifdef PHASE_SPACE
        PSCellGeometry geo;
        Box boxCoordinates = P.opts.region;
        if ( not psFilterCell( cell, P.opts, boxCoordinates, true, NULL, geo ) ) continue;
        markOccupancy( P.occ, geo.eulerPos, F );
#else
        double rho[NO_DIM+1], vel[NO_DIM+1][noVelComp];
        Real eulerPos[NO_DIM+1][NO_DIM], rawPos[NO_DIM+1][NO_DIM];
        if ( not standardCell( cell, P.own, rho, vel, eulerPos, rawPos ) ) continue;
        markOccupancy( P.occ, eulerPos, F );
#endif
    }
}

// '<cache file>.occ': the occupancy map, stamped with the tessellation's own descriptor.
static std::string occPath(User_options const &o) { return TessellationCache::cachePath( o ) + ".occ"; }

static void writeOccupancy(ServePart const &P)
{
    std::string const desc = TessellationCache::makeDescriptor( P.opts );
    // the temp name is this writer's own: the composite server and a batch run share the cache
    // folder, and two writers of one map would truncate one shared '.tmp' under each other
    std::string const path = occPath( P.opts ), tmp = path + ".tmp." + std::to_string( long( ::getpid() ) );
    std::ofstream f( tmp.c_str(), std::ios::binary | std::ios::trunc );
    if ( not f ) return;
    uint32_t const len = uint32_t( desc.size() ), n = uint32_t( OCC_N );
    uint64_t const words = uint64_t( P.occ.size() );
    f.write( OCC_MAGIC, 8 );
    f.write( reinterpret_cast<char const*>(&len), sizeof len );
    f.write( desc.data(), std::streamsize(len) );
    f.write( reinterpret_cast<char const*>(&n), sizeof n );
    f.write( reinterpret_cast<char const*>(&P.nVertices), sizeof P.nVertices );
    f.write( reinterpret_cast<char const*>(&words), sizeof words );
    f.write( reinterpret_cast<char const*>(P.occ.data()), std::streamsize( words * sizeof(uint64_t) ) );
    // trailer: the one cell filter the descriptor does not pin -- the non-periodic alpha shape's
    // radius (0 = the whole hull). A --grid-window run trusts a map for skipping only when it kept at
    // least the tetrahedra the run keeps. Readers that stop after the words (the server) ignore it.
#ifdef PHASE_SPACE
    double const alpha = double( P.opts.psLagAlphaRadius );
#else
    double const alpha = 0.;                        // the standard binary has no alpha shape
#endif
    f.write( OCC_ALPHA_TAG, 4 );
    f.write( reinterpret_cast<char const*>(&alpha), sizeof alpha );
    f.close();
    if ( f ) std::rename( tmp.c_str(), path.c_str() );
    else     std::remove( tmp.c_str() );
}

static bool readOccupancy(ServePart &P)
{
    std::ifstream f( occPath( P.opts ).c_str(), std::ios::binary );
    if ( not f ) return false;
    char magic[8] = {0};
    uint32_t len = 0, n = 0;
    uint64_t nVert = 0, words = 0;
    f.read( magic, 8 );
    f.read( reinterpret_cast<char*>(&len), sizeof len );
    if ( not f or std::memcmp( magic, OCC_MAGIC, 8 ) != 0 or len == 0 or len > (1u<<20) ) return false;
    std::string stored( size_t(len), '\0' );
    f.read( &stored[0], std::streamsize(len) );
    f.read( reinterpret_cast<char*>(&n), sizeof n );
    f.read( reinterpret_cast<char*>(&nVert), sizeof nVert );
    f.read( reinterpret_cast<char*>(&words), sizeof words );
    if ( not f or stored != TessellationCache::makeDescriptor( P.opts ) or n != uint32_t(OCC_N)
         or (words != OCC_WORDS and words != 0) ) return false;
    std::vector<uint64_t> occ( static_cast<size_t>(words) );
    f.read( reinterpret_cast<char*>(occ.data()), std::streamsize( words * sizeof(uint64_t) ) );
    if ( not f ) return false;
    P.occ.swap( occ );
    P.nVertices = nVert;
    return true;
}

static size_t residentCached()
{
    size_t k = 0;
    for (ServePart const &P : g_serve.comp.parts)
        if ( P.dt and P.cached ) ++k;
    return k;
}

inline double serveTriBytes(ServePart const &P)
{
    return SERVE_TRI_BYTES_PER_VERTEX * double( std::max<uint64_t>( P.nVertices, 1 ) );
}

// the memory of the resident CACHED partitions (uncached ones are pinned and not counted)
static double residentBytes()
{
    double b = 0.;
    for (ServePart const &P : g_serve.comp.parts)
        if ( P.dt and P.cached ) b += serveTriBytes( P ) + ( P.index ? P.indexBytes : 0. );
    return b;
}

// may partition P be loaded next to the resident ones? (by bytes when automatic, else by count)
static bool roomToLoad(ServePart const &P)
{
    if ( g_serve.comp.byBytes ) return residentBytes() + serveTriBytes( P ) <= g_serve.comp.budgetBytes;
    return residentCached() < g_serve.comp.maxResident;
}

static void dropResident(ServePart &P)
{
    P.index.reset();                         // its Cell_handles point into the tessellation
    P.indexBytes = 0.;
    P.dt.reset();
    P.walks = 0;
}

// Loads partition p (dropping the least recently used cached ones beyond the cap). False when its
// cached tessellation cannot be read back.
static bool ensureResident(size_t const p, double &loadSeconds)
{
    ServePart &P = g_serve.comp.parts[p];
    if ( P.dt ) return true;
    while ( not roomToLoad( P ) )
    {
        size_t victim = g_serve.comp.parts.size();
        for (size_t q = 0; q < g_serve.comp.parts.size(); ++q)
        {
            ServePart const &Q = g_serve.comp.parts[q];
            if ( q == p or not Q.dt or not Q.cached ) continue;
            if ( victim == g_serve.comp.parts.size() or Q.lastUse < g_serve.comp.parts[victim].lastUse ) victim = q;
        }
        if ( victim == g_serve.comp.parts.size() ) break;
        dropResident( g_serve.comp.parts[victim] );
    }
    auto const t0 = std::chrono::steady_clock::now();
    std::unique_ptr<DT> dt( new DT );
    if ( not TessellationCache::tryLoad( *dt, P.opts ) ) return false;
#ifdef PHASE_SPACE
    if ( g_pe.ctx.vertexMass )   // the grid conventions' vertex-mass pass, once per load (not per request)
        prepareVertexMass( *dt, P.opts );
#endif
    loadSeconds += std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    P.dt = std::move( dt );
    return true;
}

} // namespace psPointEvalDetail


void psServeCompositeBegin(User_options const &userOptions, int const totalPartitions)
{
    if ( g_serve.fd < 0 )
        throwError( "'--serve': the protocol channel is not set up (main() must call psServeRedirectStdout() before any output)." );
    if ( not g_pe.ctx.active )
        throwError( "'--serve': the point-evaluation context was not initialized." );
    g_serve.comp.parts.clear();
    g_serve.comp.parts.resize( size_t( std::max( 1, totalPartitions ) ) );
    g_serve.comp.verbose = userOptions.verboseLevel;
    g_serve.progress = userOptions.psServeProgress;
    g_serve.comp.done = 0;
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " composite server over "
            << MESSAGE::cMagenta() << totalPartitions << MESSAGE::cReset() << " partition tessellations"
            << ( userOptions.tessellationCacheDir.empty()
                 ? " (no '--tessellation-cache': every partition stays in memory)"
                 : " (out of core: partitions live in '" + userOptions.tessellationCacheDir + "')" )
            << ".\n" << MESSAGE::Flush;
}


void psServeCompositeAddPartition(int const index, User_options &partOptions,
                                  std::function<void(std::vector<Particle_data>&)> const &selectParticles,
                                  double const *ownLo, double const *ownHi)
{
    ServePart &P = g_serve.comp.parts.at( size_t(index) );
    P.opts = partOptions;
    P.own.on = ( ownLo != nullptr );
    for (int d = 0; d < NO_DIM; ++d)
    {
        P.own.lo[d] = ownLo ? ownLo[d] : 0.;
        P.own.hi[d] = ownHi ? ownHi[d] : 0.;
    }
    auto const t0 = std::chrono::steady_clock::now();
    bool const caching = not P.opts.tessellationCacheDir.empty();
    char const *how = "built";

    // an earlier session (or a batch run with the cache) left this partition's tessellation AND
    // its map: register it without loading anything
    if ( caching and TessellationCache::headerMatches( P.opts ) and readOccupancy( P ) )
    {
        P.cached = true;
        how = "cached";
    }
    else
    {
        // the tessellation may be cached without its map (a batch run wrote it): load it; else
        // select the particles and build (and, with the cache, save) it
        std::unique_ptr<DT> dt( new DT );
        bool have = caching and TessellationCache::headerMatches( P.opts )
                    and TessellationCache::tryLoad( *dt, P.opts );
        if ( have )
            how = "loaded";
        else
        {
            std::vector<Particle_data> particles;
            selectParticles( particles );
            if ( not particles.empty() )
            {
                buildTessellation( &particles, P.opts, *dt );
                have = true;
            }
        }
        if ( not have )
        {
            P.occ.clear();              // no particles: owns nothing
            how = "empty";
        }
        else
        {
            P.nVertices = uint64_t( dt->number_of_vertices() );
            computeOccupancy( *dt, P, occFrameFromCtx() );
            if ( caching and TessellationCache::headerMatches( P.opts ) )
            {
                writeOccupancy( P );
                P.cached = true;
                dt.reset();             // out of core: reloaded when a request needs it
            }
            else
            {
#ifdef PHASE_SPACE
                if ( g_pe.ctx.vertexMass )   // the grid conventions' vertex-mass pass, once (not per request)
                    prepareVertexMass( *dt, P.opts );
#endif
                P.dt = std::move( dt ); // no cache (or it could not be written): stays in memory
            }
        }
    }

    double const sec = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    std::lock_guard<std::mutex> lock( g_serve.comp.logMutex );
    ++g_serve.comp.done;
    serveProgress( "[partitions %d/%zu] partition %d: %s, %llu vertices, %.1f s\n", g_serve.comp.done, g_serve.comp.parts.size(),
                   index, how, (unsigned long long)P.nVertices, sec );
    MESSAGE::Message prog( g_serve.comp.verbose );
    prog << MESSAGE::cGreen() << "  [partitions " << g_serve.comp.done << "/" << g_serve.comp.parts.size() << "]"
         << MESSAGE::cReset() << MESSAGE::cDim() << "  partition " << index << ": " << how << ", "
         << P.nVertices << " vertices, " << MESSAGE::formatDuration(sec) << MESSAGE::cReset() << "\n"
         << MESSAGE::Flush;
}


namespace psPointEvalDetail {

// Composite run, phase 1: the resident set.
static void compositeResidentSet(User_options const &userOptions, size_t const nParts, uint64_t const nVertMax,
                                 size_t const nCached)
{
    // which cached partitions may be in memory at once: --serve-resident N of them, else as many as
    // fit in the auto-tune memory budget -- their tessellations (SERVE_TRI_BYTES_PER_VERTEX) plus the
    // cell indexes that exist (built only when they fit too, see the request loop). Counting an index
    // for every partition (850 B/vertex) would keep TNG100-3-Dark's server at 3 resident
    // 14.3M-vertex partitions where 4 fit, so a zoom needing 4 would reload one every time.
    // DTFE_SERVE_BUDGET_GB overrides the budget (tests).
    g_serve.comp.byBytes = false;
    if ( userOptions.serveResident > 0 )
        g_serve.comp.maxResident = size_t( userOptions.serveResident );
    else if ( nCached > 0 )
    {
        std::string why;
        double budget = autoTuneBudget( autoTunePhysicalRAM(), why );
        if ( double const v = Env::real( "DTFE_SERVE_BUDGET_GB", 0. ); v > 0. )
            budget = v * 1.e9;
        g_serve.comp.byBytes = true;
        g_serve.comp.budgetBytes = budget;
        double const perPart = SERVE_TRI_BYTES_PER_VERTEX * double( std::max<uint64_t>( nVertMax, 1 ) );
        g_serve.comp.maxResident = size_t( std::max( 1., std::floor( budget / perPart ) ) );   // reported only
    }
    else
        g_serve.comp.maxResident = nParts;
    g_serve.comp.maxResident = std::max<size_t>( 1, std::min( g_serve.comp.maxResident, nParts ) );
}

// Composite request, phase 1: the partitions whose occupancy map holds one of the installed points.
static std::vector<size_t> compositeNeeded(size_t const nParts)
{
    // the buckets holding a requested point
    std::vector<uint64_t> want( OCC_WORDS, 0 );
    for (size_t i = 0; i < g_pe.ctx.nPoints; ++i)
    {
        size_t b = 0;
        for (int d = 0; d < NO_DIM; ++d)
            b = b * size_t(OCC_N) + size_t( occBucket( g_pe.ctx.pos[i*NO_DIM+d], d ) );
        want[b >> 6] |= uint64_t(1) << (b & 63);
    }
    std::vector<size_t> need;
    for (size_t p = 0; p < nParts; ++p)
    {
        std::vector<uint64_t> const &occ = g_serve.comp.parts[p].occ;
        if ( occ.empty() ) continue;
        for (size_t w = 0; w < OCC_WORDS; ++w)
            if ( occ[w] & want[w] ) { need.push_back( p ); break; }
    }
    // the resident ones first, so a load never evicts a partition this request still needs
    // while an already-loaded one waits
    std::stable_partition( need.begin(), need.end(), [](size_t p) { return bool( g_serve.comp.parts[p].dt ); } );
    return need;
}

// Composite request, phase 2: the parallel loads (compositeStartLoads) and their state. The destructor waits for the
// loads in flight, so every exit path of a request does.
struct CompositeLoads
{
    struct Prefetch { std::unique_ptr<DT> dt; bool ok = false, done = false; double sec = 0.; };
    std::vector<size_t> toLoad;
    std::vector<Prefetch> pre;
    std::vector<long> preIndex;
    std::mutex preMutex;
    std::condition_variable preCv;
    std::atomic<size_t> preNext{ 0 };
    std::vector<std::thread> loaders;
    ~CompositeLoads() { for (std::thread &t : loaders) if ( t.joinable() ) t.join(); }
};

// PARALLEL LOADS: the needed partitions that are not in memory are read back from the cache by
// background threads -- as many as fit under the resident cap once the partitions this request
// does not need are dropped -- while the resident ones are evaluated. A load (inflate + CGAL
// deserialization, ~8 s for a 6M-vertex TNG100-3 partition) is independent of every other, so
// k loads take about one load's time on k cores. The evaluation ORDER does not depend on the
// loads (resident first, then the others in 'need' order), so no answer does. Partitions beyond the
// cap are loaded one at a time afterwards (compositeEvaluatePartition: ensureResident).
static void compositeStartLoads(std::vector<size_t> const &need, size_t const nParts, CompositeLoads &L)
{
    std::vector<char> inNeed( nParts, 0 );
    for (size_t const p : need)
    {
        inNeed[p] = 1;
        if ( not g_serve.comp.parts[p].dt ) L.toLoad.push_back( p );
    }
    // how many of toLoad (in order) fit next to the resident ones
    auto fittingLoads = [&]() -> size_t
    {
        if ( not g_serve.comp.byBytes )
        {
            size_t const r = residentCached();
            return g_serve.comp.maxResident > r ? std::min( g_serve.comp.maxResident - r, L.toLoad.size() ) : 0;
        }
        double free = g_serve.comp.budgetBytes - residentBytes();
        size_t k = 0;
        while ( k < L.toLoad.size() and serveTriBytes( g_serve.comp.parts[L.toLoad[k]] ) <= free )
            free -= serveTriBytes( g_serve.comp.parts[L.toLoad[k++]] );
        return k;
    };
    size_t room = fittingLoads();
    while ( room < L.toLoad.size() )
    {
        size_t victim = nParts;
        for (size_t q = 0; q < nParts; ++q)
        {
            ServePart const &Q = g_serve.comp.parts[q];
            if ( inNeed[q] or not Q.dt or not Q.cached ) continue;
            if ( victim == nParts or Q.lastUse < g_serve.comp.parts[victim].lastUse ) victim = q;
        }
        if ( victim == nParts ) break;
        dropResident( g_serve.comp.parts[victim] );
        room = fittingLoads();
    }
    size_t const nPre = std::min( room, L.toLoad.size() );
    L.pre.resize( nPre );
    L.preIndex.assign( nParts, -1 );
    for (size_t i = 0; i < nPre; ++i) L.preIndex[L.toLoad[i]] = long(i);
    if ( nPre > 0 )
    {
        unsigned const hw = unsigned( dtfeHardwareThreads() );
        size_t const nThreads = std::min<size_t>( nPre, std::max<size_t>( 1, std::min( 8u, std::max( 1u, hw / 2u ) ) ) );
        for (size_t t = 0; t < nThreads; ++t)
            L.loaders.emplace_back( [&L, nPre]()   // (L outlives the threads: its destructor joins them)
            {
                for (;;)
                {
                    size_t const i = L.preNext.fetch_add( 1 );
                    if ( i >= nPre ) return;
                    auto const t0 = std::chrono::steady_clock::now();
                    std::unique_ptr<DT> dt( new DT );
                    bool const ok = TessellationCache::tryLoad( *dt, g_serve.comp.parts[L.toLoad[i]].opts );
                    double const sec = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
                    std::lock_guard<std::mutex> lk( L.preMutex );
                    L.pre[i].dt = std::move( dt );
                    L.pre[i].ok = ok;
                    L.pre[i].sec = sec;
                    L.pre[i].done = true;
                    L.preCv.notify_all();
                }
            } );
    }
}

// Composite request, phase 3: partition p (the k-th of the nNeed this request visits) loaded if need be and evaluated;
// an error message when its cached tessellation cannot be read back, else empty.
static std::string compositeEvaluatePartition(size_t const p, size_t const k, size_t const nNeed, bool const useIndex,
                                              int const bucketFloor, bool &walkBucketed, std::vector<Cell_handle> &candidates,
                                              CompositeLoads &L, double &loadSeconds, size_t &loaded)
{
    ServePart &P = g_serve.comp.parts[p];
    bool const wasResident = bool( P.dt );
    serveProgress( "[request %zu/%zu] partition %zu: %s\n", k, nNeed, p,
                   wasResident ? "in memory" : "loading from the cache" );
    auto const tPart = std::chrono::steady_clock::now();
    double const loadBefore = loadSeconds;
    double indexSeconds = 0.;
    std::string const unreadable = "partition " + std::to_string(p) + "'s cached tessellation could not be read back from '"
        + P.opts.tessellationCacheDir + "' (deleted, or the inputs changed since the server started); restart the server";
    if ( not wasResident )
    {
        long const i = L.preIndex[p];
        if ( i >= 0 )
        {
            std::unique_lock<std::mutex> lk( L.preMutex );
            L.preCv.wait( lk, [&]{ return L.pre[size_t(i)].done; } );
            if ( not L.pre[size_t(i)].ok ) return unreadable;
            P.dt = std::move( L.pre[size_t(i)].dt );
            loadSeconds += L.pre[size_t(i)].sec;
        }
        else if ( not ensureResident( p, loadSeconds ) )
            return unreadable;
        ++loaded;
    }
    g_pe.cellList = nullptr;
    // the index for a small request only once this partition has earned it (INDEX_AFTER_WALKS),
    // or when it exists already; else the walk (DTFE_SERVE_INDEX=1 forces the index)
    bool const forceIndex = Env::tristate( "DTFE_SERVE_INDEX" ) == 1;
    // ... and, under the automatic resident set, only when the index fits next to the resident
    // partitions WITHOUT evicting one (a reload costs more than the index saves); else walk
    bool const indexFits = P.index or forceIndex or not g_serve.comp.byBytes or not P.cached
        or residentBytes() + g_serve.comp.indexPerVertex * double( std::max<uint64_t>( P.nVertices, 1 ) ) <= g_serve.comp.budgetBytes;
    bool const partIndex = useIndex and indexFits and ( P.index or P.walks >= INDEX_AFTER_WALKS or forceIndex );
    if ( useIndex and not indexFits and P.walks >= INDEX_AFTER_WALKS )
        serveProgress( "[request %zu/%zu] partition %zu: its cell index would not fit in memory, walking\n",
                       k, nNeed, p );
    if ( not partIndex ) ++P.walks;
    if ( useIndex and not partIndex and not walkBucketed )
    {   // the request was bucketed for the index (1 bucket floor): a walk wants the walk's buckets
        bucketQueryPoints( bucketFloor );
        walkBucketed = true;
    }
    if ( partIndex )
    {
        if ( not P.index )
        {
            serveProgress( "[request %zu/%zu] partition %zu: building its cell index\n", k, nNeed, p );
            auto const tIdx = std::chrono::steady_clock::now();
            P.index.reset( new ServeIndex );
            buildServeIndex( *P.dt, *P.index );
            indexSeconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - tIdx ).count();
            ServeIndex const &I = *P.index;
            P.indexBytes = double( I.cells.capacity() ) * sizeof(Cell_handle) + double( I.start.capacity() ) * sizeof(size_t)
                         + double( I.members.capacity() + I.stamp.capacity() ) * sizeof(uint32_t);
            g_serve.comp.indexPerVertex = P.indexBytes / double( std::max<uint64_t>( P.nVertices, 1 ) );
        }
        serveCandidates( *P.index, candidates );
        g_pe.cellList = &candidates;
    }
#ifdef PHASE_SPACE
    interpolatePoints_phaseSpace( *P.dt, P.opts );
#else
    interpolatePoints_standardOwn( *P.dt, P.opts, P.own );
#endif
    g_pe.cellList = nullptr;
    P.lastUse = ++g_serve.comp.clock;
    double const partSeconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - tPart ).count();
    serveProgress( "[request %zu/%zu] partition %zu done in %.2f s (load %.2f s, index %.2f s)\n", k, nNeed, p,
                   partSeconds, loadSeconds - loadBefore, indexSeconds );
    return std::string();
}

// Composite request, phase 4: the request's progress line and log line.
static void compositeRequestReport(User_options const &userOptions, size_t const nNeed, size_t const nParts, size_t const loaded,
                                   double const loadSeconds, std::chrono::steady_clock::time_point const tReq)
{
    serveProgress( "[request done] %zu points, %zu of %zu partitions, %zu loaded from the cache in %.1f s, %.2f s in all\n",
                   g_pe.ctx.nPoints, nNeed, nParts, loaded, loadSeconds,
                   std::chrono::duration<double>( std::chrono::steady_clock::now() - tReq ).count() );
    if ( userOptions.verboseLevel >= 2 or loaded > 0 )
    {
        MESSAGE::Message m( userOptions.verboseLevel );
        m << MESSAGE::cDim() << "  request of " << g_pe.ctx.nPoints << " points: " << nNeed
          << " of " << nParts << " partitions (" << loaded << " loaded from the cache in "
          << MESSAGE::formatDuration(loadSeconds) << ")" << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
    }
}

} // namespace psPointEvalDetail


[[noreturn]] void psServeCompositeRun(User_options &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    size_t const nParts = g_serve.comp.parts.size();
    uint64_t nVertTotal = 0, nVertMax = 0;
    size_t nCached = 0;
    for (ServePart const &P : g_serve.comp.parts)
    {
        nVertTotal += P.nVertices;
        nVertMax = std::max( nVertMax, P.nVertices );
        if ( P.cached ) ++nCached;
    }

    compositeResidentSet( userOptions, nParts, nVertMax, nCached );
    char budgetText[32];
    std::snprintf( budgetText, sizeof budgetText, "%.1f GB", g_serve.comp.budgetBytes / 1.e9 );

    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " composite server ready: "
            << nParts << " partitions, " << nVertTotal << " vertices in total; "
            << ( nCached == 0 ? std::string( "all held in memory" )
                 : g_serve.comp.byBytes ? "as many held in memory at once as fit in " + std::string( budgetText )
                                    + " (" + std::to_string( g_serve.comp.maxResident ) + " of the largest), the others reloaded from the cache on demand"
                 : std::to_string( g_serve.comp.maxResident ) + " held in memory at once (--serve-resident), the others reloaded from the cache on demand" )
            << "; close stdin to stop.\n" << MESSAGE::Flush;

    size_t const indexCells = size_t( 6.8 * double( std::max<uint64_t>( nVertMax, 1 ) ) );
    int const bucketFloor = int( std::pow( double(indexCells) / 16., 1. / double(NO_DIM) ) );
    std::vector<Cell_handle> candidates;
    serveHandshake( nVertTotal, uint32_t(nParts), uint32_t(g_serve.comp.maxResident) );

    serveLoop( userOptions, indexCells, bucketFloor, [&](bool const useIndex) -> std::string
    {
        std::vector<size_t> const need = compositeNeeded( nParts );
        size_t loaded = 0;
        double loadSeconds = 0.;
        auto const tReq = std::chrono::steady_clock::now();

        CompositeLoads loads;
        compositeStartLoads( need, nParts, loads );

        size_t k = 0;
        bool walkBucketed = not useIndex;
        for (size_t const p : need)
        {
            std::string const failure = compositeEvaluatePartition( p, ++k, need.size(), useIndex, bucketFloor, walkBucketed,
                                                                    candidates, loads, loadSeconds, loaded );
            if ( not failure.empty() ) return failure;
        }
        compositeRequestReport( userOptions, need.size(), nParts, loaded, loadSeconds, tReq );
        return std::string();
    } );
}


// ======================================================================================
// Batch partition runs and the occupancy maps (DTFE.cpp's Lagrangian partition loop).
// --------------------------------------------------------------------------------------

// --grid-window: can partition 'partOpts' deposit into the window of 'mainOpts'? 1 = its occupancy
// map has a bucket inside the window, 0 = none (the partition's streams never reach the window, so
// it is skipped: no particle selection, no tessellation load or build), -1 = no usable map (process
// it). The map is the one a composite server or an earlier batch run wrote beside the partition's
// cached tessellation, validated by the same full descriptor. The window's bucket range is widened
// by one bucket on each side, so float rounding at its edges can never drop a partition.
int psOccupancyTouchesWindow(User_options const &partOpts, User_options const &mainOpts)
{
    using namespace psPointEvalDetail;
    if ( not mainOpts.psWindowOn or partOpts.tessellationCacheDir.empty() ) return -1;
    std::ifstream f( occPath( partOpts ).c_str(), std::ios::binary );
    if ( not f ) return -1;
    char magic[8] = {0};
    uint32_t len = 0, n = 0;
    uint64_t nVert = 0, words = 0;
    f.read( magic, 8 );
    f.read( reinterpret_cast<char*>(&len), sizeof len );
    if ( not f or std::memcmp( magic, OCC_MAGIC, 8 ) != 0 or len == 0 or len > (1u<<20) ) return -1;
    std::string stored( size_t(len), '\0' );
    f.read( &stored[0], std::streamsize(len) );
    f.read( reinterpret_cast<char*>(&n), sizeof n );
    f.read( reinterpret_cast<char*>(&nVert), sizeof nVert );
    f.read( reinterpret_cast<char*>(&words), sizeof words );
    if ( not f or stored != TessellationCache::makeDescriptor( partOpts ) or n != uint32_t(OCC_N)
         or (words != OCC_WORDS and words != 0) ) return -1;
    std::vector<uint64_t> occ( static_cast<size_t>(words) );
    f.read( reinterpret_cast<char*>(occ.data()), std::streamsize( words * sizeof(uint64_t) ) );
    if ( not f ) return -1;
    char tag[4] = {0};
    double alphaMap = -1.;
    bool alphaKnown = false;
    if ( f.read( tag, 4 ) and std::memcmp( tag, OCC_ALPHA_TAG, 4 ) == 0
         and f.read( reinterpret_cast<char*>(&alphaMap), sizeof alphaMap ) )
        alphaKnown = true;

    // the map holds the tetrahedra the server's cell filter kept; the descriptor pins all of that
    // filter except the non-periodic alpha shape (a periodic box never has one): trust the map only
    // when it kept at least the tetrahedra this run keeps
#ifdef PHASE_SPACE
    double const az = double( partOpts.psLagAlphaRadius );
#else
    double const az = 0.;
#endif
    bool filterOk;
    if ( partOpts.periodic and az <= 0. ) filterOk = true;
    else if ( not alphaKnown )            filterOk = false;
    else if ( az <= 0. )                  filterOk = ( alphaMap <= 0. );
    else                                  filterOk = ( alphaMap <= 0. or alphaMap >= az * (1. - 1.e-12) );
    if ( not filterOk ) return -1;
    if ( words == 0 ) return 0;            // the partition owns no tetrahedron at all

    OccFrame const F = occFrameFrom( partOpts );
    Box const &b = mainOpts.regionOn ? mainOpts.region : mainOpts.boxCoordinates;
    std::vector<int> bk[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const lo = double( b.coords[2*d] );
        double const dx = ( double( b.coords[2*d+1] ) - lo ) / double( mainOpts.gridSize[d] );
        double const x0 = lo + double( mainOpts.psWindowLo[d] ) * dx;
        double const x1 = lo + double( mainOpts.psWindowLo[d] + mainOpts.psWindowDims[d] ) * dx;
        double const w = double(OCC_N) / F.boxLen[d];
        long b0 = long( std::floor( (x0 - F.boxLo[d]) * w ) ) - 1;
        long b1 = long( std::floor( (x1 - F.boxLo[d]) * w ) ) + 1;
        if ( F.periodic )
        {
            if ( b1 - b0 + 1 >= OCC_N ) { b0 = 0; b1 = OCC_N - 1; }
            for (long q = b0; q <= b1; ++q) bk[d].push_back( int( ((q % OCC_N) + OCC_N) % OCC_N ) );
        }
        else
        {
            b0 = std::max( 0L, std::min( b0, long(OCC_N) - 1 ) );
            b1 = std::max( 0L, std::min( b1, long(OCC_N) - 1 ) );
            for (long q = b0; q <= b1; ++q) bk[d].push_back( int(q) );
        }
    }
    // every combination of the per-axis bucket lists (an odometer over them, the last axis fastest)
    size_t idx[NO_DIM] = {};
    for (int d = 0; d < NO_DIM; ++d) if ( bk[d].empty() ) return 0;
    for (;;)
    {
        size_t q = 0;
        for (int d = 0; d < NO_DIM; ++d) q = q * OCC_N + size_t( bk[d][idx[d]] );
        if ( (occ[q >> 6] >> (q & 63)) & 1u ) return 1;
        int d = NO_DIM - 1;
        while ( d >= 0 and ++idx[d] == bk[d].size() ) { idx[d] = 0; --d; }
        if ( d < 0 ) break;
    }
    return 0;
}

bool psTessellationCached(User_options const &opts)
{
    return not opts.tessellationCacheDir.empty() and TessellationCache::headerMatches( opts );
}

// A batch partition run with '--tessellation-cache' leaves the partition's occupancy map beside its
// cached tessellation when none is there yet (the server writes its own): a later --grid-window run
// (the launcher's exact zoom) then skips the partitions that never reach its window, and a
// composite server registers the partition without loading it. One pass of the cell filter over
// the tessellation, seconds per partition, paid once.
void psWriteOccupancyIfMissing(DT &dt, User_options &partOpts)
{
#ifdef PHASE_SPACE
    using namespace psPointEvalDetail;
    if ( partOpts.tessellationCacheDir.empty() or not partOpts.psUseSubgrid ) return;
    if ( not TessellationCache::headerMatches( partOpts ) ) return;    // not cached: no map either
    ServePart P;
    P.opts = partOpts;
    if ( readOccupancy( P ) ) return;                                    // already there
    P.nVertices = uint64_t( dt.number_of_vertices() );
    computeOccupancy( dt, P, occFrameFrom( partOpts ) );
    writeOccupancy( P );
#else
    (void)dt; (void)partOpts;
#endif
}
