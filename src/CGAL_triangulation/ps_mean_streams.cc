/* The auto-tuner's estimate of the mean stream count (psEstimateMeanStreams, ps_point_eval.h), from the
   particles before any triangulation of the run exists. Only auto_tune.h uses it. */

#include "../define.h"

#include "../ps_point_eval.h"

#include "triangulation_common.h"


// ======================================================================================
// Auto-tune support: the mean number of streams at a random point of the box, estimated from
// the particles before any triangulation exists (declared in ps_point_eval.h).
//
// Every tetrahedron of the Lagrangian tessellation covers its Eulerian volume exactly once, so
// the mean stream count at a uniformly random point is sum|V_Euler| / sum V_Lagrange over all
// tetrahedra. That ratio is estimated on up to 256 random Lagrangian patches of 20 mean spacings
// on a side: Delaunay-triangulate each patch's Lagrangian positions (the tessellation PS-DTFE
// itself uses), keep the tetrahedra whose Lagrangian centroid lies 3 spacings inside the patch
// (hull tetrahedra are artefacts of the cut), and sum both volumes. MEASURED against the
// '.pts_streams' of a full-box 8192^2 plane, TNG100-3-Dark: estimated 1.00/0.99/1.13/1.6/2.3/
// 3.4-3.9 vs measured 1.00/1.02/1.21/2.30/2.65/4.25 at z = 20/5/2/1/0.5/0; the rest of the gap
// is the PLANE, not the estimate (16 strips of the z=0 plane range 1.45 .. 22.8), which is why
// the tuner budgets twice the multi-stream excess.
// ======================================================================================
#if NO_DIM == 3 && defined(PHASE_SPACE)      // Lagrangian positions exist only in the PS build
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <random>

double psEstimateMeanStreams(std::vector<Particle_data> const &particles, double const boxLoIn[3],
                             double const boxLenIn[3], bool const periodic, int *patchesUsed,
                             double const *regionLo, double const *regionHi)
{
    typedef CGAL::Exact_predicates_inexact_constructions_kernel EK;
    typedef CGAL::Triangulation_vertex_base_with_info_3<uint32_t, EK> EVb;
    typedef CGAL::Triangulation_data_structure_3<EVb> ETds;
    typedef CGAL::Delaunay_triangulation_3<EK, ETds> EDT;

    if ( patchesUsed ) *patchesUsed = 0;
    size_t const N = particles.size();
    if ( N < 2000 or N > size_t(4294967295u) ) return -1.;
    // The Lagrangian domain the patches tile. Periodic: the box. Non-periodic: the particles' OWN
    // Lagrangian bounding box -- the output region ('--box') can be smaller than the cloud, and
    // wrapping initial positions into it folded particles from far apart into one patch, whose
    // Delaunay tetrahedra then spanned the whole cloud in Eulerian space: 8.1 million "streams"
    // per point on a TNG100-3 Lagrangian sub-cube with a 31 Mpc region (true mean ~1-2).
    double boxLo[3], boxLen[3];
    for (int d=0; d<3; ++d) { boxLo[d] = boxLoIn[d]; boxLen[d] = boxLenIn[d]; }
    if ( not periodic )
    {
        double lo[3], hi[3];
        for (int d=0; d<3; ++d) { lo[d] = double(particles[0].lagPos[d]); hi[d] = lo[d]; }
        for (size_t i=1; i<N; ++i)
            for (int d=0; d<3; ++d)
            {
                double const q = double(particles[i].lagPos[d]);
                lo[d] = std::min(lo[d], q);
                hi[d] = std::max(hi[d], q);
            }
        for (int d=0; d<3; ++d)
        {
            if ( not (hi[d] > lo[d]) ) return -1.;
            boxLo[d] = lo[d];
            boxLen[d] = (hi[d] - lo[d]) * (1. + 1e-9);          // the maximum stays inside the last cell
        }
    }
    double const spacing = std::cbrt( boxLen[0]*boxLen[1]*boxLen[2] / double(N) );
    int G[3];
    for (int d=0; d<3; ++d)
        G[d] = std::max( 2, int(boxLen[d] / (20.*spacing)) );
    size_t const nCells = size_t(G[0]) * size_t(G[1]) * size_t(G[2]);
    auto wrapped = [&](Real v, int d) {                        // (non-periodic: the bounding box holds every position)
        double x = double(v) - boxLo[d];
        if ( periodic ) x -= boxLen[d] * std::floor( x / boxLen[d] );
        return x;
    };
    auto cellOf = [&](Particle_data const &part) {
        size_t c = 0;
        for (int d=0; d<3; ++d)
        {
            int g = int( wrapped(part.lagPos[d], d) / boxLen[d] * G[d] );
            c = c * size_t(G[d]) + size_t( std::min( std::max(g, 0), G[d]-1 ) );
        }
        return c;
    };
    // The length of [lo,hi] inside the region on axis d, over the periodic images
    auto overlap = [&](double lo, double hi, int d) {
        double sum = 0.;
        for (int k = (periodic ? -1 : 0); k <= (periodic ? 1 : 0); ++k)
        {
            double const sh = k * boxLenIn[d];
            sum += std::max( 0., std::min( hi + sh, regionHi[d] ) - std::max( lo + sh, regionLo[d] ) );
        }
        return sum;
    };

    // '--sample-points' region: the candidate patches are the Lagrangian cells whose Eulerian image
    // meets the region -- the image's bounding box from one pass over the particles, each position taken at its
    // minimum image from the cell's Lagrangian centre (displacements stay below half a box), so a cell across the
    // box edge is not stretched over the whole axis
    bool useRegion = regionLo != nullptr and regionHi != nullptr;
    double regionVol = 1.;
    std::vector<size_t> order;
    if ( useRegion )
    {
        for (int d=0; d<3; ++d) regionVol *= std::max( 0., regionHi[d] - regionLo[d] );
        std::vector<double> eLo( 3*nCells, std::numeric_limits<double>::max() ), eHi( 3*nCells, -std::numeric_limits<double>::max() );
        for (size_t i=0; i<N; ++i)
        {
            size_t const c = cellOf( particles[i] );
            size_t r = c;
            for (int d=2; d>=0; --d)
            {
                int const g = int( r % size_t(G[d]) );
                r /= size_t(G[d]);
                double x = double( particles[i].pos[d] );
                if ( periodic )
                {
                    double const qc = boxLo[d] + (g + 0.5) * boxLen[d] / G[d];
                    x -= boxLenIn[d] * std::nearbyint( (x - qc) / boxLenIn[d] );
                }
                eLo[3*c+d] = std::min( eLo[3*c+d], x );
                eHi[3*c+d] = std::max( eHi[3*c+d], x );
            }
        }
        for (size_t c=0; c<nCells; ++c)
        {
            bool meets = eHi[3*c] >= eLo[3*c];
            for (int d=0; d<3 and meets; ++d)
                meets = ( periodic and eHi[3*c+d] - eLo[3*c+d] >= boxLenIn[d] )
                        or overlap( eLo[3*c+d], eHi[3*c+d], d ) > 0.
                        or ( eHi[3*c+d] == eLo[3*c+d] and eLo[3*c+d] >= regionLo[d] and eLo[3*c+d] <= regionHi[d] );
            if ( meets ) order.push_back( c );
        }
        if ( order.size() < 16 or not (regionVol > 0.) ) { useRegion = false; order.clear(); }
    }
    size_t const nCand = useRegion ? order.size() : nCells;
    if ( not useRegion )
    {
        order.resize( nCells );
        for (size_t c=0; c<nCells; ++c) order[c] = c;
    }
    size_t const K = std::min( nCand, size_t(256) );
    std::mt19937_64 rng( 20260929 );                          // fixed: the same data, the same estimate
    std::shuffle( order.begin(), order.end(), rng );
    std::vector<int> patchOf( nCells, -1 );
    for (size_t k=0; k<K; ++k) patchOf[ order[k] ] = int(k);

    // one pass: the particles of the chosen patches, by their Lagrangian position (wrapped into
    // the box: initial positions can stray a little outside it)
    std::vector< std::vector<uint32_t> > members( K );
    double const margin = 3.*spacing;
    for (size_t i=0; i<N; ++i)
    {
        if ( not useRegion )
        {
            size_t const c = cellOf( particles[i] );
            if ( patchOf[c] >= 0 ) members[ patchOf[c] ].push_back( uint32_t(i) );
            continue;
        }
        // with a region the counted tetrahedra must TILE the Lagrangian space (a patch shrunk by its margin would
        // count a slab that falls in the margins as empty: 0.7 instead of 1 stream on a 1-D pancake), so a patch
        // counts the tetrahedra with their centroid in its WHOLE cell and also holds the particles within the
        // margin around it -- a particle near a cell face belongs to up to 8 patches
        int gs[3][2], ng[3];
        for (int d=0; d<3; ++d)
        {
            double const w = boxLen[d] / G[d];
            double const xq = wrapped( particles[i].lagPos[d], d );
            int const g = std::min( std::max( int( xq / w ), 0 ), G[d]-1 );
            ng[d] = 0;
            gs[d][ng[d]++] = g;
            double const f = xq - g * w;
            int const gn = f < margin ? g - 1 : ( w - f < margin ? g + 1 : g );
            if ( gn != g )
            {
                if ( gn >= 0 and gn < G[d] ) gs[d][ng[d]++] = gn;
                else if ( periodic ) gs[d][ng[d]++] = ( gn + G[d] ) % G[d];
            }
        }
        for (int a=0; a<ng[0]; ++a) for (int b=0; b<ng[1]; ++b) for (int e=0; e<ng[2]; ++e)
        {
            size_t const c = ( size_t(gs[0][a]) * size_t(G[1]) + size_t(gs[1][b]) ) * size_t(G[2]) + size_t(gs[2][e]);
            if ( patchOf[c] >= 0 ) members[ patchOf[c] ].push_back( uint32_t(i) );
        }
    }

    double sumEuler = 0., sumLag = 0.;
    int used = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:sumEuler,sumLag,used)
    for (long k=0; k<long(K); ++k)
    {
        std::vector<uint32_t> const &idx = members[k];
        if ( idx.size() < 64 ) continue;
        size_t cell = order[size_t(k)];
        double lo[3], hi[3];                                  // the patch's inner region (Lagrangian)
        for (int d=2; d>=0; --d)
        {
            int const g = int( cell % size_t(G[d]) );
            cell /= size_t(G[d]);
            double const w = boxLen[d] / G[d];
            lo[d] = useRegion ? g * w : g * w + margin;       // region: the whole cell (the members reach past it)
            hi[d] = useRegion ? (g + 1) * w : (g + 1) * w - margin;
        }
        // a region patch's members can come from across the periodic boundary: their Lagrangian position is taken
        // at its minimum image from the cell's centre
        auto lagAt = [&](uint32_t i, int d) {
            double qv = wrapped( particles[i].lagPos[d], d );
            if ( useRegion and periodic )
            {
                double const cen = 0.5 * (lo[d] + hi[d]);
                qv -= boxLen[d] * std::nearbyint( (qv - cen) / boxLen[d] );
            }
            return qv;
        };
        std::vector< std::pair<EK::Point_3, uint32_t> > pts;
        pts.reserve( idx.size() );
        for (uint32_t i : idx)
            pts.emplace_back( EK::Point_3( lagAt(i, 0), lagAt(i, 1), lagAt(i, 2) ), i );
        EDT dt( pts.begin(), pts.end() );
        for (auto c = dt.finite_cells_begin(); c != dt.finite_cells_end(); ++c)
        {
            double q[4][3], x[4][3];
            bool inside = true;
            for (int v=0; v<4; ++v)
            {
                EK::Point_3 const &p = c->vertex(v)->point();
                q[v][0] = p.x(); q[v][1] = p.y(); q[v][2] = p.z();
                Particle_data const &part = particles[ c->vertex(v)->info() ];
                for (int d=0; d<3; ++d) x[v][d] = double( part.pos[d] );
            }
            for (int d=0; d<3 and inside; ++d)
            {
                double const cen = 0.25 * (q[0][d] + q[1][d] + q[2][d] + q[3][d]);
                inside = cen >= lo[d] and cen < hi[d];
            }
            if ( not inside ) continue;
            double eq[3][3], ex[3][3];
            for (int v=1; v<4; ++v)
                for (int d=0; d<3; ++d)
                {
                    eq[v-1][d] = q[v][d] - q[0][d];
                    double e = x[v][d] - x[0][d];
                    if ( periodic ) e -= boxLen[d] * std::nearbyint( e / boxLen[d] );   // minimum image
                    ex[v-1][d] = e;
                }
            auto det = [](double const a[3][3]) {
                return a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
                     - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
                     + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]);
            };
            sumLag   += std::fabs( det(eq) );
            double vE = std::fabs( det(ex) );
            if ( useRegion )      // the share of its Eulerian extent inside the region, axis by axis
                for (int d=0; d<3; ++d)
                {
                    double lo = 0., hi = 0.;
                    for (int v=0; v<3; ++v) { lo = std::min( lo, ex[v][d] ); hi = std::max( hi, ex[v][d] ); }
                    lo += x[0][d];
                    hi += x[0][d];
                    double const len = hi - lo;
                    vE *= len > 1e-12 * boxLenIn[d] ? std::min( 1., overlap( lo, hi, d ) / len )
                                                    : ( overlap( lo - 0.5e-12 * boxLenIn[d], lo + 0.5e-12 * boxLenIn[d], d ) > 0. ? 1. : 0. );
                }
            sumEuler += vE;
        }
        ++used;
    }
    if ( patchesUsed ) *patchesUsed = used;
    if ( not (sumLag > 0.) ) return -1.;
    if ( not useRegion ) return sumEuler / sumLag;
    // Eulerian volume inside the region per Lagrangian volume, times the Lagrangian volume of every cell that
    // meets it (the others put nothing there), over the region's volume
    double const cellVol = boxLen[0]/G[0] * boxLen[1]/G[1] * boxLen[2]/G[2];
    return sumEuler / sumLag * double(nCand) * cellVol / regionVol;
}
#else
double psEstimateMeanStreams(std::vector<Particle_data> const &, double const *, double const *, bool, int *patchesUsed,
                             double const *, double const *)
{
    if ( patchesUsed ) *patchesUsed = 0;
    return -1.;
}
#endif
