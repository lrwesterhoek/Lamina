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


/* Volume-averaged grid interpolation, method 1 (= sample points chosen inside the
   Delaunay cell, proportional to cell volume). */
#include "triangulation_common.h"
#include "face_sweep.h"          // --face-gradient: exact face integrals (avg1FacePass)
#include "../env_knobs.h"
#include "../thread_count.h"   // coresPerTeam: the exact average's CPU threads
#include <typeinfo>
#include <stdio.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#include <gsl/gsl_qrng.h>
#ifdef OPEN_MP
#include <omp.h>
#endif

#ifndef PHASE_SPACE
// The standard binary's GPU deposit counters (the PS binary's are psGpuRetries & co. in
// ps_interpolation.cc): the failed GPU command buffers of the run by cause -- [0] innocent victim,
// [1] timeout (a watchdog kill), [2] out of memory, [3] any other; they sum to dtfeGpuRetries -- counted by the
// GPU host (dtfe_metal_host.cc), read by run_progress.h for the progress sidecar's gpu_retries /
// gpu_retries_by_cause and DTFE.cpp's closing 'GPU deposit retries:' line; and the sub-domain deposits that asked
// for the GPU and ran on the CPU instead (DTFE.cpp's closing warning: the threads other than the master print
// their reason only as a warning line of their own). Defined in every standard build, GPU or not.
std::atomic<int> dtfeGpuRetries{ 0 };
std::atomic<int> dtfeGpuRetriesByCause[4] = { { 0 }, { 0 }, { 0 }, { 0 } };
static std::atomic<int> dtfeGpuFallbacks{ 0 };
int dtfeGpuFallbacksTaken(bool reset)
{
    return reset ? dtfeGpuFallbacks.exchange( 0 ) : dtfeGpuFallbacks.load();
}
#endif

// GPU deposit (METAL=1 build, 3D only); the CPU loop below remains the fallback. MY_SCALAR
// evaluates arbitrary user C++ per sample, which cannot run on the GPU.
#if defined(DTFE_GPU) && NO_DIM==3 && !defined(MY_SCALAR)
#define DTFE_GPU_ACTIVE
#include <cstdint>
#include <algorithm>
#include <utility>
#include "gpu_host.h"
#endif

// --exact-average: analytic cell∩tet moments (vendored r3d, Powell & Abel 2015, in 3D; the
// triangle clipper ps_clip2d.h in 2D; MY_SCALAR is not linear inside a tet, so it keeps the MC
// loop -- rejected at parsing). r3d.h carries its own extern "C" guard.
#if NO_DIM==3 && !defined(MY_SCALAR)
#include "../../third_party/r3d/r3d.h"
#include "exact_sweep.h"    // the per-slab/column clipping of the exact average
#elif NO_DIM==2 && !defined(MY_SCALAR)
#include "ps_clip2d.h"
#endif



// Fills 'arraySize' quasi-random (Sobol) barycentric coordinates inside the unit simplex; samples in
// the unit cube are rejected unless their coordinates sum to <= 1, keeping them within the simplex.
void quasiRandomSequence(Real quasiRandomNumbers[][NO_DIM],
                         size_t const arraySize)
{
    gsl_qrng * q = gsl_qrng_alloc( gsl_qrng_sobol, NO_DIM );

    double randomRoot[NO_DIM];
    for (size_t i=0; i<arraySize; )
    {
        gsl_qrng_get (q, randomRoot);
        Real sum = NO_DIM==2 ? randomRoot[0]+randomRoot[1] : randomRoot[0]+randomRoot[1]+randomRoot[2];
        if ( sum<=1. )
        {
            for (int j=0; j<NO_DIM; ++j)
                quasiRandomNumbers[i][j] = Real( randomRoot[j] );
            ++i;
        }
    }
    gsl_qrng_free (q);
}
// Maps unit-simplex barycentric samples into the Delaunay cell via its vertex-difference matrix 'vM'
// (triangle in 2D, tetrahedron in 3D); writes the resulting Cartesian points to 'randomPoints'.
void quasiRandomPointsInCell(double vM[][NO_DIM],
                             size_t noRandomPoints,
                             Real qR[][NO_DIM],
                             Point *randomPoints)
{
#if NO_DIM==2
    for (size_t i=0; i<noRandomPoints; ++i)
        randomPoints[i] = Point( qR[i][0]*vM[0][0]+qR[i][1]*vM[1][0], qR[i][0]*vM[0][1]+qR[i][1]*vM[1][1] );
#elif NO_DIM==3
    for (size_t i=0; i<noRandomPoints; ++i)
        randomPoints[i] = Point( qR[i][0]*vM[0][0]+qR[i][1]*vM[1][0]+qR[i][2]*vM[2][0], qR[i][0]*vM[0][1]+qR[i][1]*vM[1][1]+qR[i][2]*vM[2][1], qR[i][0]*vM[0][2]+qR[i][1]*vM[1][2]+qR[i][2]*vM[2][2] );
#endif
}


// A Delaunay cell with its vertices in the order the Monte-Carlo average uses (vertex 0 = the base the
// barycentric samples are mapped from). It stands in for the CGAL cell in the field helpers
// (field_computation.h: 'cell->vertex(i)'), which are templates on the cell type.
struct McCell
{
    Vertex_handle v[NO_DIM+1];
    inline Vertex_handle vertex(int const i) const { return v[i]; }
    inline McCell const* operator->() const { return this; }
};

// splitmix64 finalizer: a well-mixed 64-bit hash step.
inline uint64_t mcMix(uint64_t z)
{
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
// Hash of a vertex's ABSOLUTE coordinates. Every partition, thread and padding choice of one run sees the same
// coordinates for the same vertex (a periodic image is x -/+ L, computed the same way everywhere), so the hash
// is a property of the vertex, not of the triangulation that holds it. The coordinates are hashed as FLOATS:
// the single and the double build then share the realization (a kpc snapshot scaled to Mpc gives float- and
// double-rounded coordinates whose double bit patterns differ in every vertex, which would make the two builds'
// grids differ by 8% rms per cell; the float patterns agree but for rare double roundings). Two vertices of one
// cell that round to the same float hash alike and are ordered by their exact coordinates (mcVertexOrder).
inline uint64_t mcVertexHash(Vertex_handle const &v)
{
    uint64_t h = 0x6a09e667f3bcc909ULL;
    for (int d=0; d<NO_DIM; ++d)
    {
        float const x = float( v->point()[d] );
        uint32_t u; std::memcpy( &u, &x, sizeof(u) );
        h = mcMix( h ^ uint64_t(u) );
    }
    return h;
}

// The vertex order of the Monte-Carlo average. 'legacy' keeps CGAL's own order (vertex(0) as the base), which
// the --exact-average path uses and DTFE_MC_LEGACY=1 selects for the sampled path (A/B). Otherwise the
// NO_DIM+1 vertices are sorted by a key that depends only on the cell's own vertex SET: each vertex's
// coordinate hash mixed with the (order-free) sum of the cell's hashes, ties (never seen; 64-bit keys) broken
// by the coordinates.
//   Why: the fixed Sobol prefix of n samples does not average to the centroid (barycentric slot means
//   0.160/0.262/0.275/0.302 at n = 17, the minimum count), so each slot carries a fixed weight error, and
//   CGAL's labelling correlates with density (slot densities 1.023/1.012/1.006/0.959 at TNG100 z=1). The
//   product is a systematic mass loss of 0.17-0.29% in an '_a' grid. A pseudo-random,
//   order-free slot assignment decorrelates the slots from the density AND makes the realization of each
//   tetrahedron independent of the insertion order, the thread count and the partition split.
inline McCell mcVertexOrder(Finite_cells_iterator &cell,
                            bool const legacy)
{
    McCell c;
    for (int i=0; i<NO_DIM+1; ++i)
        c.v[i] = cell->vertex(i);
    if ( legacy )
        return c;
    uint64_t hv[NO_DIM+1], sum = 0;
    for (int i=0; i<NO_DIM+1; ++i)
    {
        hv[i] = mcVertexHash( c.v[i] );
        sum += hv[i];
    }
    uint64_t key[NO_DIM+1];
    for (int i=0; i<NO_DIM+1; ++i)
        key[i] = mcMix( hv[i] ^ sum );
    auto before = [&](int const a, int const b) {   // strict order on (key, coordinates)
        if ( key[a] != key[b] ) return key[a] < key[b];
        for (int d=0; d<NO_DIM; ++d)
        {
            double const xa = double( c.v[a]->point()[d] ), xb = double( c.v[b]->point()[d] );
            if ( xa != xb ) return xa < xb;
        }
        return false; };
    int idx[NO_DIM+1];
    for (int i=0; i<NO_DIM+1; ++i) idx[i] = i;
    for (int i=1; i<NO_DIM+1; ++i)                    // insertion sort of NO_DIM+1 entries
        for (int j=i; j>0 and before( idx[j], idx[j-1] ); --j)
            std::swap( idx[j], idx[j-1] );
    McCell s;
    for (int i=0; i<NO_DIM+1; ++i)
        s.v[i] = c.v[ idx[i] ];
    return s;
}



// Returns true if the Delaunay cell's bounding box lies entirely outside the region of interest.
inline bool cellOutsideRegion(DT & dt,
                              Finite_cells_iterator &cell,
                              Bbox &fullBox)
{
#if NO_DIM==2
    return not do_overlap( dt.triangle( cell ).bbox(), fullBox );
#elif NO_DIM==3
    return not do_overlap( dt.tetrahedron( cell ).bbox(), fullBox );
#endif
}

static int const SINGLE_GRID_CELL = 0;
static int const MULTIPLE_GRID_CELLS = 1;
// Classifies whether a Delaunay cell fits in one grid cell or spans several, and fills 'vertexMatrix'
// with each non-base vertex's offset from the base vertex (used to map barycentric samples). 'Cell' is the
// CGAL cell iterator or an McCell (the Monte-Carlo vertex order; the arithmetic is the same for both).
template <typename Cell>
int checkCellPosition(Cell &cell,
                      Box &regionBox,
                      Real *dx,
                      double vertexMatrix[][NO_DIM],
                      int *baseGridCell,
                      Real *basePosition)
{
    Point base = cell->vertex(0)->point();
    for (int i=0; i<NO_DIM; ++i)
    {
        basePosition[i] = base[i] - regionBox[2*i]; // base vertex position
        baseGridCell[i] = int( floor( (basePosition[i])/dx[i] ) ); // grid cell containing the base point
    }
    Box gridCellBox;    // bounds of the grid cell that contains 'base'
    for (int i=0; i<NO_DIM; ++i)
    {
        gridCellBox[2*i] = baseGridCell[i] * dx[i] + regionBox[2*i];
        gridCellBox[2*i+1] = gridCellBox[2*i] + dx[i];
    }
    bool insideOneCell = true;
    for (int i=1; i<NO_DIM+1; ++i)
        if ( not gridCellBox.isPointInBox( cell->vertex(i)->point() ) )
        {
            insideOneCell = false;
            break;
        }


    // store each non-base vertex position relative to 'base'
    for (int v = 0; v<NO_DIM; ++v)
        for (int i=0; i<NO_DIM; ++i)
            vertexMatrix[v][i] = double(cell->vertex(v+1)->point()[i]) - double(base[i]);

    if ( insideOneCell )
        return SINGLE_GRID_CELL;
    return MULTIPLE_GRID_CELLS;
}

/* The exact average's GRID FRAME (plan P10.8). Its deposits -- the CPU's r3d / ps_clip2d clipping, the GPU kernel
   and the face pass -- form a tetrahedron's geometry relative to the corner of the FULL grid's cell holding its base
   vertex: origin + g * dxG in double, with g the cell's global index (user_options.h gridFrame: the run's grid box and
   full grid, this sub-grid's origin in it). The vertices relative to that corner are exact in double and narrow to a
   float whose rounding does not depend on the sub-domain, and the cells are named by their global index minus the
   sub-grid's origin, so every sub-domain computes the same shares for the same tetrahedron. (The sub-domain frame,
   checkCellPosition's, rounded the GPU deposit's float vertices by up to 2^-24 of the sub-domain's extent: 1.9e-6 Mpc,
   a strip of cells off by up to 1e-2 of the gradient's rms in TNG100's 3^3 plan, the P10 status. The Monte-Carlo
   average keeps that frame and its bytes.) */
struct GridFrame
{
    double origin[NO_DIM], dxG[NO_DIM];
    long   offset[NO_DIM];     // this sub-grid's origin in the full grid (cells)
    explicit GridFrame(User_options const &o) { o.gridFrame( origin, dxG, offset ); }
    // the global cell holding coordinate x on axis d
    inline long cellOf(double const x, int const d) const { return long( std::floor( ( x - origin[d] ) / dxG[d] ) ); }
    // the position of the global plane g on axis d: ONE expression for every tetrahedron and sub-domain, so the
    // face pass decides a plane's side alike for a vertex shared by two tetrahedra
    inline double plane(long const g, int const d) const { return origin[d] + double(g) * dxG[d]; }
};
// One tetrahedron in the frame: the base vertex's cell (sub-grid indices, outside [0, nGrid) for a padding
// tetrahedron), that cell's corner, the base vertex relative to it (in [0, dxG) up to rounding), and whether every
// vertex lies in that one cell. vertexMatrix: the other vertices relative to the base, exact double differences.
struct TetFrame
{
    int    cell[NO_DIM];
    double corner[NO_DIM], base[NO_DIM];
    bool   single;
};
template <typename Cell>
inline TetFrame tetFrame(Cell &c, GridFrame const &F, double vertexMatrix[][NO_DIM])
{
    TetFrame t;
    t.single = true;
    for (int d=0; d<NO_DIM; ++d)
    {
        double const x0 = double( c->vertex(0)->point()[d] );
        long const g = F.cellOf( x0, d );
        t.cell[d] = int( g - F.offset[d] );
        t.corner[d] = F.plane( g, d );
        t.base[d] = x0 - t.corner[d];
        for (int v=1; v<NO_DIM+1; ++v)
        {
            double const xv = double( c->vertex(v)->point()[d] );
            vertexMatrix[v-1][d] = xv - x0;
            if ( F.cellOf( xv, d ) != g ) t.single = false;
        }
    }
    return t;
}



// Returns true if 'gridCell' is in range, writing its flat array index to 'cellIndex'.
inline bool isGridCellIndex(int *gridCell,
                            size_t *totalGridSize,
                            size_t *cellIndex)
{
    for (int i=0; i<NO_DIM; ++i)
        if ( gridCell[i]<0 or gridCell[i]>=int(totalGridSize[i]) )
            return false;

    *cellIndex = gridCellIndex( gridCell, totalGridSize );
    return true;
}
// The grid-cell coordinate along axis i of a sample point (relative to the base vertex at 'basePosition'): ONE
// expression, used by isGridCellIndex and by the threaded Monte-Carlo average's slab test (avg1McTet), so the two
// can never put a sample in different x planes.
inline int sampleGridCoord(Real const *dx,
                           Point const &samplePoint,
                           Real const *basePosition,
                           int const i)
{
    return int(floor( (basePosition[i] + samplePoint[i])/dx[i] ));
}
// Locates the grid cell containing a sample point; returns true and writes 'cellIndex' if in range.
inline bool isGridCellIndex(size_t *totalGridSize,
                            Real *dx,
                            Point &samplePoint,
                            Real *basePosition,
                            size_t *cellIndex)
{
    int pos[NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
    {
        pos[i] = sampleGridCoord( dx, samplePoint, basePosition, i );
        if ( pos[i]<0 or pos[i]>=int(totalGridSize[i]) )
            return false;
    }

    *cellIndex = gridCellIndex( pos, totalGridSize );
    return true;
}



// Returns the density averaged over the cell's vertices (= its value at the cell centroid).
template <typename Cell>
inline Real averageDensity(Cell &cell)
{
    Real temp = 0.;
    for (int i=0; i<NO_DIM+1; ++i)
        temp += cell->vertex(i)->info().density();
    return temp / Real(NO_DIM+1);
}
// Returns the velocity averaged over the cell's vertices (= its value at the cell centroid).
template <typename Cell>
inline Pvector<Real,noVelComp> averageVelocity(Cell &cell)
{
    Pvector<Real,noVelComp> temp = Pvector<Real,noVelComp>::zero();
    for (int i=0; i<NO_DIM+1; ++i)
        temp += cell->vertex(i)->info().velocity();
    return temp / Real(NO_DIM+1);
}
// Returns the scalar averaged over the cell's vertices (= its value at the cell centroid).
template <typename Cell>
inline Pvector<Real,noScalarComp> averageScalar(Cell &cell)
{
    Pvector<Real,noScalarComp> temp = Pvector<Real,noScalarComp>::zero();
    for (int i=0; i<NO_DIM+1; ++i)
        temp += cell->vertex(i)->info().myScalar();
    return temp / Real(NO_DIM+1);
}





// The threads of the CPU average, exact and Monte-Carlo alike (the CPU loop of interpolateGrid_averaged_1 says how
// they split the work): every core when this triangulation is the only one being processed -- a single-domain run, or each
// partition of a '--sample-points' run (DTFE_parallel's single-triangulation branch) -- and, inside DTFE_parallel's
// OpenMP team of sub-domains, the cores left per sub-domain, so the two levels never oversubscribe (PS-DTFE's
// psDepositThreads rule, thread_count.h). DTFE_DEPOSIT_THREADS pins it, as for the PS deposit.
static int exactAverageThreads()
{
    return coresPerTeam( "DTFE_DEPOSIT_THREADS" );
}

// The output grids of the '_a' fields (pointers into Quantities), for the phases that zero, fill and normalize them.
struct Avg1Grids
{
    std::vector<Real>                             *density;
    std::vector< Pvector<Real,noVelComp> >        *velocity;
    std::vector< Pvector<Real,noGradComp> >       *velocity_gradient;
    std::vector< Pvector<Real,noScalarComp> >     *scalar;
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient;
};

// Prints the opening line: the grid, the region and how the volume average is done (Monte Carlo or exact).
static inline void avg1Announce(MESSAGE::Message &message, size_t *nGrid, Box &boxCoordinates, size_t const NN,
                                bool const exactAvg)
{
    if ( exactAvg )
        message << "\nComputing interpolation of the fields volume averaged over the sampling cell on a regular "
                << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region "
                << boxCoordinates.print()
                << ". The volume average integrates the linear interpolant EXACTLY over every grid-cell/tetrahedron intersection (--exact-average: r3d on the CPU, a z-sweep on the GPU; '--subsamples' is ignored).\n"
                << "\t Done: " << MESSAGE::Flush;
    else
        message << "\nComputing interpolation of the fields volume averaged over the sampling cell on a regular "
            << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region "
            << boxCoordinates.print()
            << ". The volume average is done using Monte Carlo sampling in each of the triangles/tetrahedra of the Delaunay triangulation. "
            << "There are " << NN << " random sampling points on average in each grid cell of the output fields (the random samples in the Delaunay triangulation's triangles/tetrahedra are proportional to the area/volume of the cell).\n"
            << "\t Done: " << MESSAGE::Flush;
}

// Zeroes the selected output grids; warns that method 1 does not give the velocity standard deviation.
static inline void avg1ZeroGrids(User_options &userOptions, Field &field, size_t const gridSize, Avg1Grids const &grids)
{
    std::vector<Real>                         *density = grids.density;
    std::vector< Pvector<Real,noVelComp> >    *velocity = grids.velocity;
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = grids.velocity_gradient;
    std::vector< Pvector<Real,noScalarComp> > *scalar = grids.scalar;
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = grids.scalar_gradient;
    if ( field.density ) assingZeroValues<Real>( density, gridSize );
    if ( field.velocity ) assingZeroValues< Pvector<Real,noVelComp> >( velocity, gridSize );
    if ( field.velocity_gradient ) assingZeroValues< Pvector<Real,noGradComp> >( velocity_gradient, gridSize );
    if ( field.scalar ) assingZeroValues< Pvector<Real,noScalarComp> >( scalar, gridSize );
    if ( field.scalar_gradient ) assingZeroValues< Pvector<Real,noScalarGradComp> >( scalar_gradient, gridSize );
    if ( field.velocity_std )
    {
        MESSAGE::Warning warning(userOptions.verboseLevel);
        warning << "You cannot use the averaging method 1 (Monte Carlo sampling inside the Delaunay cell) to compute the velocity standard deviation inside the grid cell. Please use the averaging method 2 '--average-method 2' (Monte Carlo sampling inside the grid cell) to compute the velocity standard deviation.\n" << MESSAGE::EndWarning;
    }
}

#ifdef DTFE_GPU_ACTIVE
// Whether this pass may use the GPU deposit: --gpu, at most one scalar component and at most 2^32 grid cells.
static inline bool avg1GpuUsable(User_options &userOptions, Field &field, size_t const gridSize)
{
    bool tryMetal = userOptions.useMetal;   // (--exact-average runs on the GPU too: depositExactAverage)
#ifdef SCALAR
    if ( tryMetal and (field.scalar or field.scalar_gradient) and noScalarComp != 1 )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--gpu supports ONE scalar component (this build has NO_SCALARS=" << noScalarComp << "); using the CPU interpolation for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
        ++dtfeGpuFallbacks;
    }
#endif
    // the single-cell fast path ships 32-bit flat grid indices to the GPU
    if ( tryMetal and gridSize > size_t(UINT32_MAX) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--gpu does not support more than 2^32 grid cells; using the CPU interpolation.\n" << MESSAGE::EndWarning;
        tryMetal = false;
        ++dtfeGpuFallbacks;
    }
    return tryMetal;
}

// The GPU pass's options, grid geometry, field flags and per-tetrahedron arrays, as references, for the phases
// after the extraction walk.
struct Avg1GpuCtx
{
    User_options &userOptions;
    MESSAGE::Message &message;
    size_t *nGrid;
    Real (&dx)[NO_DIM];
    size_t gridSize;
    bool exactAvg, fDen, fVel, fGrad, fScal, fSGrad, needVel, needScal;
    int mortonShift;
    std::vector<float> &tetVerts, &tetDens, &tetVels, &tetScal, &tetVols;
    std::vector<uint32_t> &tetCnts, &tetFlats, &tetKeys;
    std::vector<int32_t> &tetCells;    // each tetrahedron's base cell, the frame of its positions (GridFrame; 0 = the region's corner)
    double dxDeposit[3];               // the kernel's cell size: the grid frame's for the exact average, the sub-domain's for Monte Carlo
};

// Orders the tetrahedra along a Morton curve of their base cell before dispatch, so the
// threads running together deposit into neighbouring cells: the deposit is bound by its
// atomics, and random cells across a 2 GB grid cost 3x the throughput of local ones (TLB
// and cache reach), measured on the shared full grid of a 512^3 run. No cost sort (by
// sample count, so that a chunk's threads are uniformly sized) is needed: the
// work items bound every thread to itemSamples samples (the per-tet A/B kernel,
// DTFE_GPU_ITEMS=0, pays for that with its imbalance). The permutation is applied one
// array at a time so only a single extra copy is alive at once.
static inline void avg1MortonOrder(Avg1GpuCtx &c)
{
    std::vector<float> &tetVerts = c.tetVerts, &tetDens = c.tetDens, &tetVels = c.tetVels, &tetScal = c.tetScal,
                       &tetVols = c.tetVols;
    std::vector<uint32_t> &tetCnts = c.tetCnts, &tetFlats = c.tetFlats, &tetKeys = c.tetKeys;
    std::vector<int32_t> &tetCells = c.tetCells;
    bool const needVel = c.needVel, needScal = c.needScal;
    int const mortonShift = c.mortonShift;
    size_t const nT = tetCnts.size();
    std::vector< std::pair<uint32_t,uint32_t> > order(nT);
    for (size_t t=0; t<nT; ++t)
        order[t] = std::make_pair( tetKeys[t], uint32_t(t) );
    std::vector<uint32_t>().swap( tetKeys );
    std::sort( order.begin(), order.end() );
    if ( Env::flag( { "DTFE_GPU_TIMING" } ) )
    {   // how many distinct base cells the 32 consecutive work items of a SIMD group touch
        // (16-sample items; a lower bound on the cell sharing a SIMD-group reduction could use)
        size_t lane = 0, distinct = 0, groups = 0, sumDistinct = 0; uint32_t last = ~0u;
        for (size_t i=0; i<nT; ++i)
        {
            uint32_t const n = tetCnts[order[i].second];
            size_t const items = n ? (n + 15) / 16 : 1;
            for (size_t k=0; k<items; ++k)
            {
                if ( lane==0 ) { distinct = 0; last = ~0u; }
                if ( order[i].first != last ) { ++distinct; last = order[i].first; }
                if ( ++lane == 32 ) { lane = 0; ++groups; sumDistinct += distinct; }
            }
        }
        fprintf( stderr, "DTFE Metal order: %zu tetrahedra, %zu SIMD groups of 32 items, %.1f distinct base cells per group (Morton shift %d)\n",
                 nT, groups, groups ? double(sumDistinct)/double(groups) : 0., mortonShift );
    }
    {
        std::vector<float> sv(nT*12);
        for (size_t i=0; i<nT; ++i)
        { size_t const src = order[i].second; for (int k=0; k<12; ++k) sv[i*12+k] = tetVerts[src*12+k]; }
        tetVerts.swap(sv);
    }
    {
        std::vector<float> sd(nT*4);
        for (size_t i=0; i<nT; ++i)
        { size_t const src = order[i].second; for (int k=0; k<4; ++k) sd[i*4+k] = tetDens[src*4+k]; }
        tetDens.swap(sd);
    }
    if ( needVel )
    {
        std::vector<float> su(nT*12);
        for (size_t i=0; i<nT; ++i)
        { size_t const src = order[i].second; for (int k=0; k<12; ++k) su[i*12+k] = tetVels[src*12+k]; }
        tetVels.swap(su);
    }
    if ( needScal )
    {
        std::vector<float> ss(nT*4);
        for (size_t i=0; i<nT; ++i)
        { size_t const src = order[i].second; for (int k=0; k<4; ++k) ss[i*4+k] = tetScal[src*4+k]; }
        tetScal.swap(ss);
    }
    {
        std::vector<float> so(nT);
        for (size_t i=0; i<nT; ++i) so[i] = tetVols[order[i].second];
        tetVols.swap(so);
    }
    {
        std::vector<uint32_t> sc(nT);
        for (size_t i=0; i<nT; ++i) sc[i] = tetCnts[order[i].second];
        tetCnts.swap(sc);
    }
    {
        std::vector<uint32_t> sf(nT);
        for (size_t i=0; i<nT; ++i) sf[i] = tetFlats[order[i].second];
        tetFlats.swap(sf);
    }
    {
        std::vector<int32_t> sl(nT*3);
        for (size_t i=0; i<nT; ++i)
        { size_t const src = order[i].second; for (int k=0; k<3; ++k) sl[i*3+k] = tetCells[src*3+k]; }
        tetCells.swap(sl);
    }
}

// Deposits the extracted tetrahedra on the GPU: into the shared full grid of a DTFE_parallel call, else into a grid
// of its own that is copied into the output. False: the CPU loop interpolates this pass.
static inline bool avg1GpuDispatch(Avg1GpuCtx &c, Avg1Grids const &grids, std::vector<Real> &quasiRandomNumbers_buf)
{
    User_options &userOptions = c.userOptions;
    MESSAGE::Message &message = c.message;
    size_t *nGrid = c.nGrid;
    size_t const gridSize = c.gridSize;
    bool const exactAvg = c.exactAvg, fDen = c.fDen, fVel = c.fVel, fGrad = c.fGrad, fScal = c.fScal, fSGrad = c.fSGrad;
    std::vector<float> &tetVerts = c.tetVerts, &tetDens = c.tetDens, &tetVels = c.tetVels, &tetScal = c.tetScal,
                       &tetVols = c.tetVols;
    std::vector<uint32_t> &tetCnts = c.tetCnts, &tetFlats = c.tetFlats;
    std::vector<int32_t> &tetCells = c.tetCells;
    std::vector<Real>                         *density = grids.density;
    std::vector< Pvector<Real,noVelComp> >    *velocity = grids.velocity;
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = grids.velocity_gradient;
    std::vector< Pvector<Real,noScalarComp> > *scalar = grids.scalar;
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = grids.scalar_gradient;
    bool metalDeposited = false;
    std::vector<float> sobolTable( quasiRandomNumbers_buf.begin(), quasiRandomNumbers_buf.end() );
    double dxv[3] = { c.dxDeposit[0], c.dxDeposit[1], c.dxDeposit[2] };   // (the grid frame's for the exact average)
    size_t const nTetsDeposited = tetCnts.size();   // the deposit call consumes (frees) the arrays
    DTFEGpuGrids gpuOut;
    std::string metalErr;
    message << gpuBackendName() << " GPU (" << gpuDeviceName() << ", " << nTetsDeposited << " tetrahedra) ... " << MESSAGE::Flush;

    // The shared full grid of this DTFE_parallel call (dtfeGpuSharedBegin in DTFE.cpp): this
    // sub-domain's samples land at full-grid cells subgridOffset + cell, and the main thread adds
    // the grid into the output afterwards -- so this thread's own sub-grid stays ZERO. A batch the
    // device hands back (allocation, watchdog) leaves its owner box zeroed and falls through to the
    // per-call path below, then to the CPU loop.
    bool sharedDone = false, sharedFailed = false;
    if ( dtfeGpuSharedActive() and userOptions.subgridOffset.size()==size_t(NO_DIM) )
    {
        size_t subOff[3] = { userOptions.subgridOffset[0], userOptions.subgridOffset[1], userOptions.subgridOffset[2] };
        if ( dtfeGpuSharedDeposit( tetVerts, tetDens, tetVels, tetScal, tetVols, tetCnts, tetFlats, tetCells,
                                   sobolTable, dxv, nGrid, subOff, metalErr ) )
            sharedDone = true;
        else if ( not tetCnts.empty() )     // the arrays survive only when the upload itself failed
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "DTFE " << gpuBackendName() << " shared grid handed this sub-domain back (" << metalErr << "); depositing it on its own.\n" << MESSAGE::EndWarning;
        }
        else if ( nTetsDeposited > 0 )
        {   // consumed by the upload but the dispatch failed: nothing left for the per-call path
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "DTFE " << gpuBackendName() << " shared grid failed for this sub-domain (" << metalErr << "); using the CPU interpolation.\n" << MESSAGE::EndWarning;
            ++dtfeGpuFallbacks;
            sharedFailed = true;
        }
    }
    // The per-call path. A deposit that fails AFTER its upload consumed the arrays (the retry policy gave up)
    // warns too, so a runtime fallback to the CPU loop below is never silent -- the suites' 'using the CPU
    // interpolation' check relies on it.
    bool perCallOk = false, perCallFailed = false;
    if ( not sharedDone and not sharedFailed and not tetCnts.empty() )
    {
        perCallOk = dtfeGpuDepositAveraged1( tetVerts, tetDens, tetVels, tetScal, tetVols, tetCnts, tetFlats, tetCells,
                                             sobolTable, dxv, nGrid, fDen, fVel, fGrad, fScal, fSGrad, exactAvg, gpuOut, metalErr );
        perCallFailed = not perCallOk;
    }
    if ( sharedDone )
        metalDeposited = true;
    else if ( perCallOk )
    {
        for (size_t i=0; i<gridSize; ++i)
        {
            if (fDen) (*density)[i] = Real(gpuOut.density[i]);
#ifdef VELOCITY
            if (fVel)
                for (size_t j=0; j<noVelComp; ++j)
                    (*velocity)[i][j] = Real(gpuOut.velocity[i*3+j]);
            if (fGrad)
                for (size_t q=0; q<noGradComp; ++q)
                    (*velocity_gradient)[i][q] = Real(gpuOut.grad[i*9+q]);
#endif
#ifdef SCALAR
            if (fScal)  (*scalar)[i][0] = Real(gpuOut.scalar[i]);
            if (fSGrad) for (int q=0; q<NO_DIM; ++q) (*scalar_gradient)[i][q] = Real(gpuOut.scalar_grad[i*3+q]);
#endif
        }
        metalDeposited = true;
    }
    else if ( perCallFailed )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "DTFE " << gpuBackendName() << " deposit unavailable (" << metalErr << "); using the CPU interpolation.\n" << MESSAGE::EndWarning;
        ++dtfeGpuFallbacks;
    }
    else if ( nTetsDeposited == 0 )
        metalDeposited = true;              // no tetrahedra in the region: nothing to deposit either way
    return metalDeposited;
}
#else
// Warns once that --gpu needs a GPU build of this binary; the pass runs on the CPU.
static inline void avg1NoGpuWarning(User_options &userOptions)
{
    if ( userOptions.useMetal )
    {
#ifndef PHASE_SPACE
        ++dtfeGpuFallbacks;
#endif
        static bool warned = false;
        if ( !warned )
        {
            warned = true;
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "--gpu requested but this binary was built without GPU support (rebuild with METAL=1 on a Mac; or NO_DIM!=3 / MY_SCALAR); using the CPU interpolation.\n" << MESSAGE::EndWarning;
        }
    }
}
#endif // DTFE_GPU_ACTIVE

// Divides the accumulated contributions by the grid-cell volume, which turns them into volume averages.
static inline void avg1Normalize(Field &field, Avg1Grids const &grids, Real const gridCellVolume)
{
    std::vector<Real>                         *density = grids.density;
    std::vector< Pvector<Real,noVelComp> >    *velocity = grids.velocity;
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = grids.velocity_gradient;
    std::vector< Pvector<Real,noScalarComp> > *scalar = grids.scalar;
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = grids.scalar_gradient;
    if (field.density)
        for (vector<Real>::iterator it=density->begin(); it!=density->end(); ++it)
            *it /= gridCellVolume;
#ifdef VELOCITY
    if (field.velocity)
        for (vector< Pvector<Real,noVelComp> >::iterator it=velocity->begin(); it!=velocity->end(); ++it)
            *it /= gridCellVolume;
    if (field.velocity_gradient)
        for (vector< Pvector<Real,noGradComp> >::iterator it=velocity_gradient->begin(); it!=velocity_gradient->end(); ++it)
            *it /= gridCellVolume;
#endif
#ifdef SCALAR
    if (field.scalar)
        for (vector< Pvector<Real,noScalarComp> >::iterator it=scalar->begin(); it!=scalar->end(); ++it)
            *it /= gridCellVolume;
    if (field.scalar_gradient)
        for (vector< Pvector<Real,noScalarGradComp> >::iterator it=scalar_gradient->begin(); it!=scalar_gradient->end(); ++it)
            *it /= gridCellVolume;
#endif
}

// The CPU loop's run-wide inputs (interpolateGrid_averaged_1 sets them once; its threads share them): the fields and
// their grids, the grid geometry, the Monte-Carlo average's Sobol table, and the threads' x-plane slabs. The
// helpers below bind its members by REFERENCE ('* const &'), so their loops read them from memory as the lambda's
// captures did: copies kept in registers made the Monte-Carlo sample loop ~4% slower (register spills).
struct Avg1CpuCtx
{
    Field const      &field;
    Avg1Grids const  &grids;
    size_t           *nGrid;
    Real             *dx;
    Real            (*quasiRandomNumbers)[NO_DIM];
    bool              centreMC;       // the Monte-Carlo samples are centred (see mcLegacy)
    int               nT, slabW;      // threads, and the width of the x-plane slabs they deal out
    // does thread th deposit into x plane gi? (the slabs of slabW planes go to the threads in turn)
    bool owns(int const th, int const gi) const { return nT == 1 or ( gi / slabW ) % nT == th; }
};

// TEST_PADDING: one cell's dummy flags and the calling thread's lists of the grid cells they flag (empty without it)
struct Avg1Dummy
{
#ifdef TEST_PADDING
    bool dummyNeighbors, dummyVertices;
    std::vector<size_t> *incompleteCells_d, *incompleteCells;
#endif
};

#ifndef MY_SCALAR
// A cell inside ONE grid cell (index): its centroid value times its volume, which is exact for the linear interpolant
// (shared by the sampled and the exact average).
static inline void avg1SingleCell(Avg1CpuCtx const &C, McCell &mc, Real posMatrixInverse[][NO_DIM], Real const cellVolume,
                                  size_t const index, Avg1Dummy const &dummy)
{
    Field const &field = C.field;
    std::vector<Real>                             * const &density = C.grids.density;
    if (field.density) (*density)[index] += averageDensity(mc) * cellVolume;
#ifdef VELOCITY
    std::vector< Pvector<Real,noVelComp> >        * const &velocity = C.grids.velocity;
    std::vector< Pvector<Real,noGradComp> >       * const &velocity_gradient = C.grids.velocity_gradient;
    if (field.velocity) (*velocity)[index] += averageVelocity(mc) * cellVolume;
    if (field.velocity_gradient)
    {
        Real velGrad[NO_DIM][noVelComp];
        velocityGrad( mc, posMatrixInverse, velGrad );
        (*velocity_gradient)[index] += velocityGradient(velGrad) * cellVolume;
    }
#endif
#ifdef SCALAR
    std::vector< Pvector<Real,noScalarComp> >     * const &scalar = C.grids.scalar;
    std::vector< Pvector<Real,noScalarGradComp> > * const &scalar_gradient = C.grids.scalar_gradient;
    if (field.scalar) (*scalar)[index] += averageScalar(mc) * cellVolume;
    if (field.scalar_gradient)
    {
        Real sGrad[NO_DIM][noScalarComp];
        scalarGrad( mc, posMatrixInverse, sGrad );
        (*scalar_gradient)[index] += scalarGradient(sGrad) * cellVolume;
    }
#endif
#ifdef TEST_PADDING
    if ( dummy.dummyNeighbors ) updateDummyGridCells( index, dummy.incompleteCells_d );
    if ( dummy.dummyVertices ) updateDummyGridCells( index, dummy.incompleteCells );
#endif
}
// The same for the exact average (plan P10.8): its gradients in double (posInvD, matrixInverseDouble's), each share
// narrowed once.
static inline void avg1SingleCellExact(Avg1CpuCtx const &C, McCell &mc, double posInvD[][NO_DIM], Real const cellVolume,
                                       size_t const index, Avg1Dummy const &dummy)
{
    (void)posInvD;
    Field const &field = C.field;
    std::vector<Real>                             * const &density = C.grids.density;
    if (field.density) (*density)[index] += averageDensity(mc) * cellVolume;
#ifdef VELOCITY
    std::vector< Pvector<Real,noVelComp> >        * const &velocity = C.grids.velocity;
    std::vector< Pvector<Real,noGradComp> >       * const &velocity_gradient = C.grids.velocity_gradient;
    if (field.velocity) (*velocity)[index] += averageVelocity(mc) * cellVolume;
    if (field.velocity_gradient)
    {
        double velGrad[NO_DIM][noVelComp];
        velocityGradDouble( mc, posInvD, velGrad );
        Pvector<Real,noGradComp> pg;
        for (size_t j = 0; j < noVelComp; ++j)
            for (size_t i = 0; i < NO_DIM; ++i)
                pg[j*NO_DIM+i] = Real( velGrad[i][j] * double(cellVolume) );
        (*velocity_gradient)[index] += pg;
    }
#endif
#ifdef SCALAR
    std::vector< Pvector<Real,noScalarComp> >     * const &scalar = C.grids.scalar;
    std::vector< Pvector<Real,noScalarGradComp> > * const &scalar_gradient = C.grids.scalar_gradient;
    if (field.scalar) (*scalar)[index] += averageScalar(mc) * cellVolume;
    if (field.scalar_gradient)
    {
        double sGrad[NO_DIM][noScalarComp];
        scalarGradDouble( mc, posInvD, sGrad );
        Pvector<Real,noScalarGradComp> pg;
        for (size_t j = 0; j < noScalarComp; ++j)
            for (size_t i = 0; i < NO_DIM; ++i)
                pg[j*NO_DIM+i] = Real( sGrad[i][j] * double(cellVolume) );
        (*scalar_gradient)[index] += pg;
    }
#endif
#ifdef TEST_PADDING
    if ( dummy.dummyNeighbors ) updateDummyGridCells( index, dummy.incompleteCells_d );
    if ( dummy.dummyVertices ) updateDummyGridCells( index, dummy.incompleteCells );
#endif
}
#endif


#if NO_DIM==3 && !defined(MY_SCALAR)
// ----- exact volume average (--exact-average): analytic cell∩tet integration -----
// The interpolant is LINEAR inside the tet, so its integral over each intersection
// piece is the centroid value times the piece volume -- order-1 r3d moments suffice
// (Powell & Abel 2015). The geometry is the grid frame's (GridFrame / TetFrame, plan P10.8): the tet in its vertex-0
// frame (vertexMatrix, exact double differences), the cells as offsets from its base cell, whose corner its base
// position is relative to -- the same arithmetic in every sub-domain. The gradients and the values are formed in
// double (posInvD: matrixInverseDouble) and each share is narrowed once. Out-of-region window cells are skipped
// exactly like out-of-region samples. The single-grid-cell fast path is avg1SingleCellExact.
// Thread th clips only the cells of its own x planes.
static inline void avg1ExactTet(Avg1CpuCtx const &C, int const th, McCell &mc, double vertexMatrix[][NO_DIM],
                                TetFrame const &tf, GridFrame const &F, double posInvD[][NO_DIM], Avg1Dummy const &dummy)
{
    Field const &field = C.field;
    size_t * const &nGrid = C.nGrid;
    int const nT = C.nT;
    std::vector<Real>                             * const &density = C.grids.density;
#ifdef VELOCITY
    std::vector< Pvector<Real,noVelComp> >        * const &velocity = C.grids.velocity;
    std::vector< Pvector<Real,noGradComp> >       * const &velocity_gradient = C.grids.velocity_gradient;
#endif
#ifdef SCALAR
    std::vector< Pvector<Real,noScalarComp> >     * const &scalar = C.grids.scalar;
    std::vector< Pvector<Real,noScalarGradComp> > * const &scalar_gradient = C.grids.scalar_gradient;
#endif
    size_t index;

    double densGrad[NO_DIM];
    densityGradDouble( mc, posInvD, densGrad );
    double const dens0 = double( mc.v[0]->info().density() );
#ifdef VELOCITY
    double velGrad[NO_DIM][noVelComp], vel0[noVelComp];
    velocityGradDouble( mc, posInvD, velGrad );
    for (size_t j = 0; j < noVelComp; ++j) vel0[j] = double( mc.v[0]->info().velocity(j) );
#endif
#ifdef SCALAR
    double sGrad[NO_DIM][noScalarComp], sc0[noScalarComp];
    scalarGradDouble( mc, posInvD, sGrad );
    for (size_t j = 0; j < noScalarComp; ++j) sc0[j] = double( mc.v[0]->info().myScalar()[j] );
#endif

    r3d_rvec3 tv[NO_DIM+1];
    for (int d = 0; d < NO_DIM; ++d) tv[0].xyz[d] = 0.;
    for (int v = 1; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            tv[v].xyz[d] = vertexMatrix[v-1][d];
    if ( determinant(vertexMatrix) < 0. )   // r3d wants positive orientation
    { r3d_rvec3 const tmp = tv[1]; tv[1] = tv[2]; tv[2] = tmp; }
    r3d_poly tetPoly;
    r3d_init_tet(&tetPoly, tv);

    // window of grid cells overlapped by the tet bbox: sub-grid indices from the offsets to the base cell, clamped
    int iLo[NO_DIM], iHi[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double lo = 0., hi = 0.;
        for (int v = 0; v < NO_DIM; ++v)
        {
            if (vertexMatrix[v][d] < lo) lo = vertexMatrix[v][d];
            if (vertexMatrix[v][d] > hi) hi = vertexMatrix[v][d];
        }
        iLo[d] = tf.cell[d] + int( std::floor( (tf.base[d] + lo) / F.dxG[d] ) );
        iHi[d] = tf.cell[d] + int( std::floor( (tf.base[d] + hi) / F.dxG[d] ) ) + 1;
        if (iLo[d] < 0) iLo[d] = 0;
        if (iHi[d] > int(nGrid[d])) iHi[d] = int(nGrid[d]);
    }

    // the window's cells, each clipped by its 6 half-spaces (exact_sweep.h: once per slab and
    // column); another thread's x planes are not clipped
    exactSweep::sweep( tetPoly, iLo, iHi,
        [&](int const d, int const g, double &lo, double &hi)
        {
            lo = double( g - tf.cell[d] ) * F.dxG[d] - tf.base[d];
            hi = lo + F.dxG[d];
        },
        [&](int const d, int const g) -> long { return ( d == 0 and nT > 1 and not C.owns( th, g ) ) ? -1 : g; },
        [&](int const *g, long const *, r3d_poly &piece)
    {
        r3d_real mom[4];
        exactSweep::reduce<1>( piece, mom );
        if ( !(mom[0] > 0.) ) return;

        index = ( size_t(g[0])*nGrid[1] + size_t(g[1]) )*nGrid[2] + size_t(g[2]);
        double const dV = mom[0];
        double const cen[3] = { mom[1]/mom[0], mom[2]/mom[0], mom[3]/mom[0] };   // the piece's centroid, vertex-0 frame
        if (field.density)
        {
            double v = dens0;
            for (int i = 0; i < NO_DIM; ++i) v += densGrad[i] * cen[i];
            (*density)[index] += Real( v * dV );
        }
#ifdef VELOCITY
        if (field.velocity)
        {
            Pvector<Real,noVelComp> pv;
            for (size_t j = 0; j < noVelComp; ++j)
            {
                double v = vel0[j];
                for (int i = 0; i < NO_DIM; ++i) v += velGrad[i][j] * cen[i];
                pv[j] = Real( v * dV );
            }
            (*velocity)[index] += pv;
        }
        if (field.velocity_gradient)
        {
            Pvector<Real,noGradComp> pg;
            for (size_t j = 0; j < noVelComp; ++j)
                for (size_t i = 0; i < NO_DIM; ++i)
                    pg[j*NO_DIM+i] = Real( velGrad[i][j] * dV );
            (*velocity_gradient)[index] += pg;
        }
#endif
#ifdef SCALAR
        if (field.scalar)
        {
            Pvector<Real,noScalarComp> ps;
            for (size_t j = 0; j < noScalarComp; ++j)
            {
                double v = sc0[j];
                for (int i = 0; i < NO_DIM; ++i) v += sGrad[i][j] * cen[i];
                ps[j] = Real( v * dV );
            }
            (*scalar)[index] += ps;
        }
        if (field.scalar_gradient)
        {
            Pvector<Real,noScalarGradComp> pg;
            for (size_t j = 0; j < noScalarComp; ++j)
                for (size_t i = 0; i < NO_DIM; ++i)
                    pg[j*NO_DIM+i] = Real( sGrad[i][j] * dV );
            (*scalar_gradient)[index] += pg;
        }
#endif
#ifdef TEST_PADDING
        if ( dummy.dummyNeighbors ) updateDummyGridCells( index, dummy.incompleteCells_d );
        if ( dummy.dummyVertices ) updateDummyGridCells( index, dummy.incompleteCells );
#endif
    } );
}
#elif NO_DIM==2 && !defined(MY_SCALAR)
// ----- exact area average (--exact-average), 2D: the same integral with the triangle clipped
// against each cell by ps_clip2d.h (area and first moments: the centroid of each piece), in the grid frame
// (TetFrame) with the gradients in double, as the 3D path -----
// Thread th clips only the cells of its own x planes.
static inline void avg1ExactTet(Avg1CpuCtx const &C, int const th, McCell &mc, double vertexMatrix[][NO_DIM],
                                TetFrame const &tf, GridFrame const &F, double posInvD[][NO_DIM], Avg1Dummy const &dummy)
{
    Field const &field = C.field;
    size_t * const &nGrid = C.nGrid;
    int const nT = C.nT;
    std::vector<Real>                             * const &density = C.grids.density;
#ifdef VELOCITY
    std::vector< Pvector<Real,noVelComp> >        * const &velocity = C.grids.velocity;
    std::vector< Pvector<Real,noGradComp> >       * const &velocity_gradient = C.grids.velocity_gradient;
#endif
#ifdef SCALAR
    std::vector< Pvector<Real,noScalarComp> >     * const &scalar = C.grids.scalar;
    std::vector< Pvector<Real,noScalarGradComp> > * const &scalar_gradient = C.grids.scalar_gradient;
#endif
    size_t index;

    double densGrad[NO_DIM];
    densityGradDouble( mc, posInvD, densGrad );
    double const dens0 = double( mc.v[0]->info().density() );
#ifdef VELOCITY
    double velGrad[NO_DIM][noVelComp], vel0[noVelComp];
    velocityGradDouble( mc, posInvD, velGrad );
    for (size_t j = 0; j < noVelComp; ++j) vel0[j] = double( mc.v[0]->info().velocity(j) );
#endif
#ifdef SCALAR
    double sGrad[NO_DIM][noScalarComp], sc0[noScalarComp];
    scalarGradDouble( mc, posInvD, sGrad );
    for (size_t j = 0; j < noScalarComp; ++j) sc0[j] = double( mc.v[0]->info().myScalar()[j] );
#endif
    double tri[3][2] = { {0., 0.},
                         {vertexMatrix[0][0], vertexMatrix[0][1]},
                         {vertexMatrix[1][0], vertexMatrix[1][1]} };
    if ( determinant(vertexMatrix) < 0. )   // the clipper wants a counter-clockwise triangle
        for (int d = 0; d < 2; ++d) std::swap( tri[1][d], tri[2][d] );

    int iLo[NO_DIM], iHi[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double lo = 0., hi = 0.;
        for (int v = 0; v < NO_DIM; ++v)
        {
            if (vertexMatrix[v][d] < lo) lo = vertexMatrix[v][d];
            if (vertexMatrix[v][d] > hi) hi = vertexMatrix[v][d];
        }
        iLo[d] = tf.cell[d] + int( std::floor( (tf.base[d] + lo) / F.dxG[d] ) );
        iHi[d] = tf.cell[d] + int( std::floor( (tf.base[d] + hi) / F.dxG[d] ) ) + 1;
        if (iLo[d] < 0) iLo[d] = 0;
        if (iHi[d] > int(nGrid[d])) iHi[d] = int(nGrid[d]);
    }

    for (int gi = iLo[0]; gi < iHi[0]; ++gi)
    for (int gj = iLo[1]; gj < iHi[1]; ++gj)
    {
        if ( nT > 1 and not C.owns( th, gi ) ) continue;   // another thread's x plane
        int const gij[2] = {gi, gj};
        double lo[2], hi[2], mom[psClip2d::MOMENTS];
        for (int d = 0; d < 2; ++d)
        {
            lo[d] = double( gij[d] - tf.cell[d] ) * F.dxG[d] - tf.base[d];
            hi[d] = lo[d] + F.dxG[d];
        }
        if ( not psClip2d::clipTriangleToCell( tri, lo, hi, mom ) ) continue;

        index = size_t(gi)*nGrid[1] + size_t(gj);
        double const dV = mom[0];
        double const cen[2] = { mom[1]/mom[0], mom[2]/mom[0] };
        if (field.density)
        {
            double v = dens0;
            for (int i = 0; i < NO_DIM; ++i) v += densGrad[i] * cen[i];
            (*density)[index] += Real( v * dV );
        }
#ifdef VELOCITY
        if (field.velocity)
        {
            Pvector<Real,noVelComp> pv;
            for (size_t j = 0; j < noVelComp; ++j)
            {
                double v = vel0[j];
                for (int i = 0; i < NO_DIM; ++i) v += velGrad[i][j] * cen[i];
                pv[j] = Real( v * dV );
            }
            (*velocity)[index] += pv;
        }
        if (field.velocity_gradient)
        {
            Pvector<Real,noGradComp> pg;
            for (size_t j = 0; j < noVelComp; ++j)
                for (size_t i = 0; i < NO_DIM; ++i)
                    pg[j*NO_DIM+i] = Real( velGrad[i][j] * dV );
            (*velocity_gradient)[index] += pg;
        }
#endif
#ifdef SCALAR
        if (field.scalar)
        {
            Pvector<Real,noScalarComp> ps;
            for (size_t j = 0; j < noScalarComp; ++j)
            {
                double v = sc0[j];
                for (int i = 0; i < NO_DIM; ++i) v += sGrad[i][j] * cen[i];
                ps[j] = Real( v * dV );
            }
            (*scalar)[index] += ps;
        }
        if (field.scalar_gradient)
        {
            Pvector<Real,noScalarGradComp> pg;
            for (size_t j = 0; j < noScalarComp; ++j)
                for (size_t i = 0; i < NO_DIM; ++i)
                    pg[j*NO_DIM+i] = Real( sGrad[i][j] * dV );
            (*scalar_gradient)[index] += pg;
        }
#endif
#ifdef TEST_PADDING
        if ( dummy.dummyNeighbors ) updateDummyGridCells( index, dummy.incompleteCells_d );
        if ( dummy.dummyVertices ) updateDummyGridCells( index, dummy.incompleteCells );
#endif
    }
}
#endif  // exact average (3D r3d / 2D ps_clip2d)


// A cell over several grid cells, Monte-Carlo sampled: its first noRandomPoints Sobol samples, each deposited in its
// own grid cell with an equal share of the cell's volume. A sample at p (relative to the base) is EVALUATED at
// p + delta (the centring, see mcLegacy in interpolateGrid_averaged_1; zero without it). randomPoints is the calling
// thread's sample buffer, grown to the largest count so far and reused: no heap block per cell.
// Slabs (the threaded CPU loop, C.nT > 1): thread th deposits only the samples whose x plane is one of its slabs. It
// finds them first, from the samples' positions alone (own: its per-sample flags, reused like randomPoints), and only
// for a cell with one does it invert the vertex matrix into posMatrixInverse and evaluate the gradients and the
// samples; a cell without one returns at once. Each sample is so deposited by exactly one thread, and every grid cell
// gets its contributions in the single-threaded order (tetrahedron order, then sample order): bit-identical at any
// thread count. Without Slabs the caller has inverted the matrix, and th and own are not used.
template <bool Slabs>
static inline void avg1McTet(Avg1CpuCtx const &C, McCell &mc, double vertexMatrix[][NO_DIM], Real basePosition[NO_DIM],
                             Real posMatrixInverse[][NO_DIM], Real const cellVolume, size_t const noRandomPoints,
                             Real const delta[NO_DIM], std::vector<Point> &randomPoints, Avg1Dummy const &dummy,
                             int const th, std::vector<unsigned char> &own)
{
    Field const &field = C.field;
    size_t * const &nGrid = C.nGrid;
    Real * const &dx = C.dx;
    std::vector<Real>                             * const &density = C.grids.density;
#ifdef VELOCITY
    std::vector< Pvector<Real,noVelComp> >        * const &velocity = C.grids.velocity;
    std::vector< Pvector<Real,noGradComp> >       * const &velocity_gradient = C.grids.velocity_gradient;
#endif
#ifdef SCALAR
    std::vector< Pvector<Real,noScalarComp> >     * const &scalar = C.grids.scalar;
#ifndef MY_SCALAR
    std::vector< Pvector<Real,noScalarGradComp> > * const &scalar_gradient = C.grids.scalar_gradient;
#endif
#endif
    size_t index;

    if ( randomPoints.size() < noRandomPoints )
        randomPoints.resize( noRandomPoints );
    quasiRandomPointsInCell( vertexMatrix, noRandomPoints, C.quasiRandomNumbers, randomPoints.data() );
    if constexpr ( Slabs )
    {
        // this thread's samples: x plane in the grid and in one of its slabs (isGridCellIndex's own x coordinate)
        if ( own.size() < noRandomPoints )
            own.resize( noRandomPoints );
        int const nx = int( nGrid[0] );
        bool any = false;
        for (size_t i=0; i<noRandomPoints; ++i)
        {
            int const gi = sampleGridCoord( dx, randomPoints[i], basePosition, 0 );
            own[i] = gi >= 0 and gi < nx and C.owns( th, gi );
            any = any or own[i];
        }
        if ( not any )
            return;   // every sample is another thread's (or outside the grid)
        matrixInverse( vertexMatrix, posMatrixInverse );
    }
    else
    {
        (void)th; (void)own;
    }
    Real factor = cellVolume / noRandomPoints;  // volume per sample point
    Vertex_handle base = mc.v[0];

    Real densGrad[NO_DIM];
    densityGrad( mc, posMatrixInverse, densGrad );
#ifdef VELOCITY
    Real velGrad[NO_DIM][noVelComp];
    velocityGrad( mc, posMatrixInverse, velGrad );
#endif
#ifdef SCALAR
    Real sGrad[NO_DIM][noScalarComp];
    scalarGrad( mc, posMatrixInverse, sGrad );
#endif
    for (size_t i=0; i<noRandomPoints; ++i) // scatter each sample point's field value into its grid cell
    {
        if constexpr ( Slabs )
            if ( not own[i] )
                continue;   // another thread's x plane
        if ( not isGridCellIndex( nGrid, dx, randomPoints[i], basePosition, &index ) )
            continue;   // sample outside the region of interest
        Point evalPoint = randomPoints[i];   // where the (linear) fields are evaluated
        if ( C.centreMC )
        {
#if NO_DIM==2
            evalPoint = Point( randomPoints[i][0]+delta[0], randomPoints[i][1]+delta[1] );
#else
            evalPoint = Point( randomPoints[i][0]+delta[0], randomPoints[i][1]+delta[1], randomPoints[i][2]+delta[2] );
#endif
        }

        if (field.density) (*density)[index] += densityValue(densGrad,base,evalPoint) * factor;
#ifdef VELOCITY
        if (field.velocity) (*velocity)[index] += velocityValue(velGrad,base,evalPoint) * factor;
        if (field.velocity_gradient) (*velocity_gradient)[index] += velocityGradient(velGrad) * factor;
#endif
#ifdef SCALAR
#ifdef MY_SCALAR
        // the user's function (any C++, not linear) sees the sample itself; only linear fields are centred
        (*scalar)[index] +=  customScalar(mc, posMatrixInverse, base, randomPoints[i]) * factor;
#else
        if (field.scalar) (*scalar)[index] += scalarValue(sGrad,base,evalPoint) * factor;
        if (field.scalar_gradient) (*scalar_gradient)[index] += scalarGradient(sGrad) * factor;
#endif
#endif
#ifdef TEST_PADDING
        if ( dummy.dummyNeighbors ) updateDummyGridCells( index, dummy.incompleteCells_d );
        if ( dummy.dummyVertices ) updateDummyGridCells( index, dummy.incompleteCells );
#endif
    }
}

#if NO_DIM==3 && defined(VELOCITY) && !defined(MY_SCALAR)
/* '--face-gradient' (docs/cleanup-plan.md P10.2): the cell average of the velocity gradient from the FACES. By Gauss,
   (1/V) times the integral of v n over a grid cell's six faces is the cell average of grad v; the DTFE interpolant is
   linear in every tetrahedron and continuous across them, so that is the per-tetrahedron gradient's cell average the
   deposit adds up -- reached here by a second, independent route (face_sweep.h: every grid plane's section through
   every tetrahedron, clipped to the face squares; a face piece's integral is its centroid value times its area, added
   to the cell below the plane and subtracted from the cell above). It is exact whatever the '_a' deposit was (the
   exact average, Monte Carlo, the GPU), and the same machinery gives PS-DTFE its mean-flow gradient (P10.3). Its
   geometry is the grid frame's (GridFrame, plan P10.8): the planes' sides are decided on absolute double positions
   against planes named by their global index, and the gradient is formed in double from the hashed vertex order, so
   the result is the same in every split.
   acc: gridSize x 9 doubles, the integral of grad v over each cell in velocityGradient()'s layout ([j*3+i] =
   dv_j/dx_i). The x planes are dealt out as in the deposit (thread th adds only to the cells of its own slabs, every
   thread walking every tetrahedron in order), so the sums are bit-identical at any thread count. degenerate (when
   not null) collects the cells a tetrahedron with a zeroed inverse can reach: its velocity is constant there, so its
   faces carry jumps that the per-tetrahedron gradient (0) does not see -- the DTFE_FACE_CHECK comparison skips them. */
static void avg1FacePass( DT &dt, GridFrame const &F, Bbox &fullBox, size_t *nGrid, int const nT,
                          int const slabW, bool const legacyOrder, std::vector<double> &acc,
                          std::vector<unsigned char> *degenerate )
{
    size_t const gridSize = nGrid[0]*nGrid[1]*nGrid[2];
    acc.assign( gridSize * noGradComp, 0. );
    if ( degenerate ) degenerate->assign( gridSize, 0 );
    auto owns = [&](int const th, int const gi) { return nT == 1 or ( gi / slabW ) % nT == th; };
    auto flat = [&](int const i, int const j, int const k) {
        return ( size_t(i)*nGrid[1] + size_t(j) )*nGrid[2] + size_t(k); };
    auto pass = [&](int const th)
    {
        for (Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
        {
            if ( cellOutsideRegion( dt, itC, fullBox ) )
                continue;
            McCell mc = mcVertexOrder( itC, legacyOrder );   // the deposits' vertex order (hashed; CGAL's under DTFE_MC_LEGACY=1), base = vertex 0
            double vertexMatrix[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    vertexMatrix[v][d] = double( mc.v[v+1]->point()[d] ) - double( mc.v[0]->point()[d] );
            // X: the vertices' ABSOLUTE positions, exact in double. The planes' sides are decided on these against
            // F.plane(), ONE expression of the global plane index, so every tetrahedron sharing a vertex, in any
            // sub-domain, computes the same signed distance, and two tetrahedra never decide a shared face lying in a
            // grid plane differently (face_sweep.h). p: the base-relative frame, X[v] - X[0] exactly.
            double X[4][3], p[4][3];
            for (int v = 0; v < 4; ++v)
                for (int d = 0; d < 3; ++d)
                {
                    X[v][d] = double( mc.v[v]->point()[d] );
                    p[v][d] = v == 0 ? 0. : vertexMatrix[v-1][d];
                }
            // the grid planes the tetrahedron may own a section of (min <= plane < max; sub-grid indices), and its
            // window of grid cells
            int pFirst[3], pLast[3], iLo[3], iHi[3];
            bool crosses = false;
            for (int d = 0; d < 3; ++d)
            {
                double mn = X[0][d], mx = X[0][d];
                for (int v = 1; v < 4; ++v) { mn = std::min( mn, X[v][d] ); mx = std::max( mx, X[v][d] ); }
                long g0 = F.cellOf( mn, d );                  // the first plane at or above mn
                while ( F.plane( g0, d ) < mn ) ++g0;
                while ( F.plane( g0 - 1, d ) >= mn ) --g0;
                long g1 = F.cellOf( mx, d );                  // the last plane below mx
                while ( F.plane( g1, d ) >= mx ) --g1;
                while ( F.plane( g1 + 1, d ) < mx ) ++g1;
                pFirst[d] = int( std::max( 0L, g0 - F.offset[d] ) );
                pLast[d] = int( std::min( long(nGrid[d]), g1 - F.offset[d] ) );
                crosses = crosses or pFirst[d] <= pLast[d];
                iLo[d] = int( F.cellOf( mn, d ) - F.offset[d] );
                iHi[d] = int( F.cellOf( mx, d ) - F.offset[d] ) + 1;
            }
            // a tetrahedron owning no section adds nothing to any face (its cell's faces lie outside it)
            if ( not crosses ) continue;
            // this thread's? (an x range of cells -- one more on each side, the x planes' two cells -- of its own)
            if ( nT > 1 )
            {
                int const a = std::max( 0, iLo[0] - 1 ), b = std::min( int(nGrid[0]), iHi[0] + 1 );
                bool mine = false;
                for (int gi = a; gi < b and not mine; ++gi) mine = owns( th, gi );
                if ( not mine ) continue;
            }
            // the gradient in double, from the double inverse (plan P10.8: not the float rounding of the labelling)
            double posInv[NO_DIM][NO_DIM];
            matrixInverseDouble( vertexMatrix, posInv );
            double velGrad[NO_DIM][noVelComp];
            velocityGradDouble( mc, posInv, velGrad );
            double vb[noVelComp];
            for (size_t j = 0; j < noVelComp; ++j) vb[j] = double( mc.v[0]->info().velocity(j) );
            if ( degenerate )
            {
                bool zero = true;
                for (int i = 0; i < 3 and zero; ++i)
                    for (int j = 0; j < 3 and zero; ++j) zero = posInv[i][j] == 0.;
                if ( zero )
                    for (int i = std::max( 0, iLo[0] - 1 ); i < std::min( int(nGrid[0]), iHi[0] + 1 ); ++i)
                        if ( owns( th, i ) )
                            for (int j = std::max( 0, iLo[1] - 1 ); j < std::min( int(nGrid[1]), iHi[1] + 1 ); ++j)
                                for (int k = std::max( 0, iLo[2] - 1 ); k < std::min( int(nGrid[2]), iHi[2] + 1 ); ++k)
                                    (*degenerate)[flat( i, j, k )] = 1;
            }
            for (int a = 0; a < 3; ++a)
            {
                int u, w;
                faceSweep::planeAxes( a, u, w );
                int win0[2] = { std::max( 0, iLo[u] ), std::max( 0, iLo[w] ) };
                int win1[2] = { std::min( int(nGrid[u]), iHi[u] ), std::min( int(nGrid[w]), iHi[w] ) };
                for (int pl = pFirst[a]; pl <= pLast[a]; ++pl)
                {
                    // an x plane: its two cells' slabs; a y or z plane: its strips' (in-plane u = x, below)
                    if ( a == 0 and nT > 1 and not ( ( pl > 0 and owns( th, pl - 1 ) ) or ( pl < int(nGrid[0]) and owns( th, pl ) ) ) )
                        continue;
                    double const P = F.plane( long(pl) + F.offset[a], a );
                    double dist[4];
                    for (int v = 0; v < 4; ++v) dist[v] = X[v][a] - P;
                    faceSweep::Polygon sec;
                    if ( not faceSweep::section( p, a, dist, sec ) ) continue;
                    double const c = P - X[0][a];                    // the plane in the base-relative frame
                    faceSweep::sweep( sec, win0, win1,
                        [&](int const k, int const g, double &blo, double &bhi)
                        {
                            int const ax = k == 0 ? u : w;
                            blo = F.plane( long(g) + F.offset[ax], ax ) - X[0][ax];
                            bhi = F.plane( long(g) + 1 + F.offset[ax], ax ) - X[0][ax];
                        },
                        [&](int const k, int const g) -> long
                        { return ( k == 0 and u == 0 and nT > 1 and not owns( th, g ) ) ? -1 : g; },
                        [&](int const *g, long const *, double const m[3])
                        {
                            double q[3];
                            q[a] = c; q[u] = m[1] / m[0]; q[w] = m[2] / m[0];
                            double Fj[noVelComp];   // the piece's flux of each velocity component
                            for (size_t j = 0; j < noVelComp; ++j)
                            {
                                double vj = vb[j];
                                for (int i = 0; i < 3; ++i) vj += velGrad[i][j] * q[i];
                                Fj[j] = vj * m[0];
                            }
                            int cell[3];
                            cell[u] = g[0]; cell[w] = g[1];
                            for (int side = 0; side < 2; ++side)   // the cell below the plane (+), the cell above (-)
                            {
                                cell[a] = side == 0 ? pl - 1 : pl;
                                if ( cell[a] < 0 or cell[a] >= int(nGrid[a]) ) continue;
                                if ( nT > 1 and not owns( th, cell[0] ) ) continue;
                                double *out = &acc[flat( cell[0], cell[1], cell[2] ) * noGradComp];
                                for (size_t j = 0; j < noVelComp; ++j)
                                    out[j*NO_DIM + size_t(a)] += side == 0 ? Fj[j] : -Fj[j];
                            }
                        } );
                }
            }
        }
    };
    if ( nT == 1 )
        pass( 0 );
    else
    {
        std::vector<std::thread> pool;
        for (int th = 1; th < nT; ++th) pool.emplace_back( pass, th );
        pass( 0 );
        for (std::thread &x : pool) x.join();
    }
}
#endif

// Volume-averaged grid interpolation (method 1): quasi-Monte-Carlo sampling inside each Delaunay cell,
// with the sample count proportional to cell volume; scatters each sample's field value into its grid cell.
// --exact-average replaces the MC inner loop by analytic cell∩tet integration (r3d; NO_DIM==3 only).
void interpolateGrid_averaged_1(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    size_t const NN = userOptions.noPoints;     // average random points per grid cell
    // --exact-average: analytic cell∩tet integration of the linear interpolant (3D CPU only)
    bool const exactAvg = userOptions.exactAverage;
    size_t const minNN = NN/6 + 1;              // minimum sample points per Delaunay cell
    Real const minRatio = 1./6.;                // vol(Delaunay)/vol(grid) below which minNN is used
    size_t const maxNN = 100*NN;                // maximum sample points per Delaunay cell
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );
#if NO_DIM==2
    Bbox fullBox( boxCoordinates[0], boxCoordinates[2], boxCoordinates[1], boxCoordinates[3] );
#elif NO_DIM==3
    Bbox fullBox( boxCoordinates[0], boxCoordinates[2], boxCoordinates[4], boxCoordinates[1], boxCoordinates[3], boxCoordinates[5] );
#endif
    // The exact average's grid frame (GridFrame, plan P10.8) and its prefilter box: the region widened by 1e-6 of its
    // length on each side, so that a tetrahedron reaching into the region's first or last cell by less than the float
    // rounding of the region's corner is not dropped before the frame's integer cells decide (a tetrahedron outside
    // the sub-grid deposits nothing).
    GridFrame const frame( userOptions );
    if ( exactAvg )   // the frame must be the grid this call fills: its sub-grid's planes at the region's faces
        for (int d=0; d<NO_DIM; ++d)
        {
            double const lo = frame.plane( frame.offset[d], d ), hi = frame.plane( frame.offset[d] + long(nGrid[d]), d );
            if ( std::fabs( lo - double(boxCoordinates[2*d]) ) > 1e-3 * frame.dxG[d]
                 or std::fabs( hi - double(boxCoordinates[2*d+1]) ) > 1e-3 * frame.dxG[d] )
            {
                MESSAGE::Error error;
                error << "The exact average's grid frame does not match this region: axis " << d << " spans [" << lo
                      << ", " << hi << "] in the frame but [" << boxCoordinates[2*d] << ", " << boxCoordinates[2*d+1]
                      << "] in the region (user_options.h gridFrame, subpartition.h copySubgridInformation)." << MESSAGE::EndError;
            }
        }
    Bbox fullBoxExact = fullBox;
    {
        // the margin scales with the coordinates' magnitude (the float rounding of a corner does), not only the length
        double m[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            m[d] = 1e-6 * std::max( double( boxLength[d] ), std::max( std::fabs( double(boxCoordinates[2*d]) ), std::fabs( double(boxCoordinates[2*d+1]) ) ) );
#if NO_DIM==2
        fullBoxExact = Bbox( fullBox.xmin() - m[0], fullBox.ymin() - m[1], fullBox.xmax() + m[0], fullBox.ymax() + m[1] );
#elif NO_DIM==3
        fullBoxExact = Bbox( fullBox.xmin() - m[0], fullBox.ymin() - m[1], fullBox.zmin() - m[2],
                             fullBox.xmax() + m[0], fullBox.ymax() + m[1], fullBox.zmax() + m[2] );
#endif
    }
    Bbox depositBox = exactAvg ? fullBoxExact : fullBox;   // the deposits' prefilter (cellOutsideRegion)
    
    
    // output field vectors
    Field field = userOptions.aField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);
    Avg1Grids const grids{ density, velocity, velocity_gradient, scalar, scalar_gradient };


    MESSAGE::Message message( userOptions.verboseLevel );
    avg1Announce( message, nGrid, boxCoordinates, NN, exactAvg );
    
    // zero-initialize all grid fields
    size_t const gridSize = (NO_DIM==2) ? nGrid[0]*nGrid[1] : nGrid[0]*nGrid[1]*nGrid[2];
    avg1ZeroGrids( userOptions, field, gridSize, grids );
    // without cells there is nothing to interpolate
    if ( dt.number_of_vertices()<NO_DIM+1 )
    {
        MESSAGE::Warning warning(1);
        warning << "Because there are less than " << NO_DIM+1 << " vertices in the Delaunay triangulation there is no cell and hence there is no information that can be used to interpolate the fields to a grid. All field values will be initialized to 0.\n" << MESSAGE::EndWarning;
        return;
    }
#if NO_DIM==3 && defined(VELOCITY) && !defined(MY_SCALAR)
    // --face-gradient (the default since 2026-11-09): the face pass at the end supplies the velocity gradient and
    // overwrites the deposit's, so the deposits -- the CPU loops and the GPU kernels, which read this 'field' --
    // skip their per-tetrahedron gradient (9 of the 13 components of a velocity run; the gradient's share of
    // the deposit). DTFE_FACE_CHECK=1 keeps it: the check compares the two. The grids were zeroed above with
    // the full field; the flag is restored before the face pass, whose output avg1Normalize then divides.
    bool const faceSuppliesGradient = userOptions.faceGradient and field.velocity_gradient and not Env::flag( "DTFE_FACE_CHECK" );
    if ( faceSuppliesGradient ) field.velocity_gradient = false;
#endif



    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];
    Real gridCellVolume = (NO_DIM==2) ? (dx[0]*dx[1]) : (dx[0]*dx[1]*dx[2]);
    if ( exactAvg )   // the grid frame's cell volume: the same in every sub-domain (plan P10.8)
    {
        double v = 1.;
        for (int d=0; d<NO_DIM; ++d) v *= frame.dxG[d];
        gridCellVolume = Real(v);
    }
#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;// grid cells with a dummy-point error in the density field
    vector<size_t> incompleteCells;  // grid cells with a dummy-point error in the other fields
#endif


    // The Monte-Carlo average: every tetrahedron maps the first n entries of ONE shared Sobol table into itself.
    // Mapped from CGAL's vertex(0) that is a BIAS (0.17-0.29% of an '_a' grid's mass; mcVertexOrder's comment says
    // why), so the CPU loop and the GPU extraction both apply two measures:
    //   - vertex order: mcVertexOrder() sorts the vertices by a hash of the cell's vertex SET, so the slots carry no
    //     density correlation and are the same in every partition / thread / padding split (the grids agree to float
    //     summation order, but for a sample lying EXACTLY on a cell face, which two splits may round apart);
    //   - centring: every sample value is shifted by grad f . delta, delta = sum_j c_j (v_j - v_0), c_j = 1/(D+1) -
    //     mean_j(n) over the table prefix used. For the linear interpolant the n samples then sum to n f(centroid)
    //     exactly, so a tetrahedron's mass and momentum are exact and only their split over the grid cells is
    //     sampled. The shifted weights lambda_ij + c_j stay >= 0 for n <= 34; beyond, the most negative is -0.013
    //     (n = 88), which gives a negative sample only at a 70x density contrast across one tetrahedron.
    // tests/dtfe_mc_average_check.sh checks the mass, the momentum and the split invariance. --exact-average takes
    // the same vertex order (plan P10.8): its per-tetrahedron gradient is formed from vertex 0, in float on the GPU,
    // and CGAL's order -- which differs between the triangulations of two splits -- moved the exact grids by 2.7e-5
    // of their rms between a 3^3 and a 6^3 plan (the P10 status). DTFE_MC_LEGACY=1 selects CGAL's order for both
    // (and the uncentred Monte-Carlo realization: A/B, and to reproduce grids made with it).
    bool const mcLegacy = Env::flag( "DTFE_MC_LEGACY" );
    bool const legacyOrder = mcLegacy;
    bool const centreMC = not exactAvg and not mcLegacy;
    if ( mcLegacy and not exactAvg )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "DTFE_MC_LEGACY=1: the Monte-Carlo '_a' average uses the uncentred CGAL-order realization (vertex(0) as the base), which loses ~0.2-0.3% of the mass and depends on the partition/thread split.\n" << MESSAGE::EndWarning;
    }
    // samples in a multi-cell tetrahedron: proportional to its volume, at least minNN and at most maxNN
    auto sampleCount = [&](Real const cellVolume) -> size_t {
        size_t const tempInt = size_t(NN*cellVolume/gridCellVolume) + 1;
        return (cellVolume/gridCellVolume>minRatio) ? (tempInt>maxNN ? maxNN:tempInt) : minNN; };

    // quasi-random (Sobol) sequence shared by all cells
    std::vector<Real> quasiRandomNumbers_buf(maxNN * NO_DIM);
    Real (*quasiRandomNumbers)[NO_DIM] = reinterpret_cast<Real(*)[NO_DIM]>(quasiRandomNumbers_buf.data());
    quasiRandomSequence( quasiRandomNumbers, maxNN );
    // centring: prefix sums (double) of the table's barycentric coordinates AS STORED (Real), so the
    // correction matches the samples actually placed; c_j(n) = 1/(D+1) - prefix_j(n)/n
    std::vector<double> prefixSum;
    if ( centreMC )
    {
        prefixSum.assign( (maxNN+1)*NO_DIM, 0. );
        for (size_t i=0; i<maxNN; ++i)
            for (int j=0; j<NO_DIM; ++j)
                prefixSum[(i+1)*NO_DIM+j] = prefixSum[i*NO_DIM+j] + double(quasiRandomNumbers[i][j]);
    }
    auto centreWeights = [&](size_t const n, double c[NO_DIM]) {
        for (int j=0; j<NO_DIM; ++j)
            c[j] = 1./double(NO_DIM+1) - prefixSum[n*NO_DIM+j] / double(n); };


    // GPU deposit (--gpu, GPU build): extract flat per-tetrahedron arrays using the CPU
    // loop's own helpers (classification, sample counts, volumes -- so both paths make identical
    // decisions), dispatch metal/dtfe_deposit.metal::depositAveraged1, and copy the grids back.
    // Any failure (no device, kernel compile, buffer alloc) falls back to the CPU loop below.
    bool metalDeposited = false;
#ifdef DTFE_GPU_ACTIVE
    bool tryMetal = avg1GpuUsable( userOptions, field, gridSize );
    if ( tryMetal )
    {
        bool const fDen = field.density;
        bool fVel = false, fGrad = false;
#ifdef VELOCITY
        fVel  = field.velocity;
        fGrad = field.velocity_gradient;
#endif
        bool const needVel = fVel or fGrad;
        bool fScal = false, fSGrad = false;     // the scalar (one component) and its gradient
#ifdef SCALAR
        fScal  = field.scalar;
        fSGrad = field.scalar_gradient;
#endif
        bool const needScal = fScal or fSGrad;

        // reserve for all finite cells (upper bound; a counting pre-pass would cost a full extra
        // CGAL cell walk, which measures comparably to the whole extraction -- the padded fraction
        // outside the region is small for standard-DTFE partitions, unlike the PS Lagrangian ones)
        size_t const nKept = dt.number_of_finite_cells();

        std::vector<float>    tetVerts, tetDens, tetVels, tetScal, tetVols;
        std::vector<uint32_t> tetCnts, tetFlats, tetKeys;
        std::vector<int32_t>  tetCells;
        tetVerts.reserve( nKept*12 );
        tetKeys.reserve( nKept );
        // Morton key of a grid cell (sub-grid coordinates clamped into the grid; axes coarsened to
        // <= 1024 cells so 3 x 10 bits suffice)
        int mortonShift = 0;
        for (int d=0; d<NO_DIM; ++d)
            while ( (nGrid[d] >> mortonShift) > 1024 ) ++mortonShift;
        auto spread = [](uint32_t v) {
            v &= 0x3ff; v = (v | (v << 16)) & 0x030000ff; v = (v | (v << 8)) & 0x0300f00f;
            v = (v | (v << 4)) & 0x030c30c3; v = (v | (v << 2)) & 0x09249249; return v; };
        auto mortonKey = [&](int const cell[NO_DIM]) {
            uint32_t key = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                long c = cell[d]; if ( c < 0 ) c = 0; if ( c >= long(nGrid[d]) ) c = long(nGrid[d]) - 1;
                key |= spread( uint32_t(c >> mortonShift) ) << d;
            }
            return key; };
        tetDens.reserve( nKept*4 );
        if ( needVel ) tetVels.reserve( nKept*12 );
        if ( needScal ) tetScal.reserve( nKept*4 );
        tetVols.reserve( nKept );
        tetCnts.reserve( nKept );
        tetFlats.reserve( nKept );
        tetCells.reserve( nKept*NO_DIM );

        for (Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
        {
            if ( cellOutsideRegion(dt, itC, depositBox) )
                continue;

            // the CPU loop's vertex order (mcVertexOrder: the hashed order; DTFE_MC_LEGACY=1 keeps CGAL's); the
            // kernels take vertex 0 as the base, in the order given
            McCell mc = mcVertexOrder( itC, legacyOrder );
            double vertexMatrix[NO_DIM][NO_DIM];
            int baseGridCell[NO_DIM];
            Real basePosition[NO_DIM];
            TetFrame tf{};
            bool single;
            if ( exactAvg )
            {   // the grid frame (plan P10.8): the base cell's corner is this tetrahedron's origin
                tf = tetFrame( mc, frame, vertexMatrix );
                single = tf.single;
                for (int d=0; d<NO_DIM; ++d)
                {
                    baseGridCell[d] = tf.cell[d];
                    basePosition[d] = Real( double(mc.v[0]->point()[d]) - double(boxCoordinates[2*d]) );   // (TEST_PADDING's sampling below)
                }
            }
            else
                single = checkCellPosition( mc, boxCoordinates, dx, vertexMatrix, baseGridCell, basePosition ) == SINGLE_GRID_CELL;
            (void)basePosition;
            Real cellVolume = volume( dt, itC );

            size_t index;
            uint32_t cnt = 0, flat = 0;
            if ( single and isGridCellIndex( baseGridCell, nGrid, &index ) )
                flat = uint32_t(index);   // cnt 0 = deposit the centroid value at 'flat'
            else
                cnt = uint32_t( sampleCount( cellVolume ) );

            // Centring on the GPU without a kernel change: the kernels evaluate a sample as
            // f_0 + grad f . p with grad f built from the vertex DIFFERENCES f_j - f_0, so adding the same
            // constant C = sum_j c_j (f_j - f_0) (= grad f . delta of the CPU loop) to all NO_DIM+1 vertex
            // values shifts every sample value by C and leaves the gradient alone. Only for sampled tets:
            // the single-cell fast path (cnt 0) deposits the vertex mean, which is exact already.
            bool const centreThis = centreMC and cnt > 0;
            double cw[NO_DIM] = {};
            if ( centreThis ) centreWeights( cnt, cw );
            auto shiftOf = [&](double const f[NO_DIM+1]) {
                double c = 0.;
                for (int j=0; j<NO_DIM; ++j) c += cw[j] * (f[j+1] - f[0]);
                return c; };

            // vertex positions: the exact average's relative to its base cell's corner (tetFrame, in double, so the
            // narrowed floats are the same in every sub-domain, plan P10.8), with that cell beside them; the Monte-Carlo
            // average's relative to the region's lower corner (subtracted in double, so the narrowed floats match the
            // CPU's Real basePosition exactly), cell 0
            for (int v=0; v<NO_DIM+1; ++v)
                for (int d=0; d<NO_DIM; ++d)
                    tetVerts.push_back( float( double(mc.v[v]->point()[d]) - ( exactAvg ? tf.corner[d] : double(boxCoordinates[2*d]) ) ) );
            for (int d=0; d<NO_DIM; ++d)
                tetCells.push_back( exactAvg ? int32_t(tf.cell[d]) : 0 );
            if ( centreThis )
            {
                double f[NO_DIM+1];
                for (int v=0; v<NO_DIM+1; ++v) f[v] = double( mc.v[v]->info().density() );
                double const c = shiftOf( f );
                for (int v=0; v<NO_DIM+1; ++v) tetDens.push_back( float(f[v] + c) );
            }
            else
                for (int v=0; v<NO_DIM+1; ++v)
                    tetDens.push_back( float(mc.v[v]->info().density()) );
            // DTFE_TEST_NAN_SHARE (ps_deposit_params.h): this tetrahedron's vertex densities NaN, so the
            // kernel's density shares of it are NaN and the non-finite guard drops and counts them
            if ( fixedShareTestNanEvery() and fixedShareTestNan( double(mc.v[0]->point()[0]), double(mc.v[0]->point()[1]),
                                                                 double(mc.v[0]->point()[2]) ) )
                for (int v=0; v<NO_DIM+1; ++v) tetDens[tetDens.size() - 1 - size_t(v)] = std::numeric_limits<float>::quiet_NaN();
#ifdef VELOCITY
            if ( needVel )
            {
                double c[noVelComp] = {};
                if ( centreThis )
                    for (size_t j=0; j<noVelComp; ++j)
                    {
                        double f[NO_DIM+1];
                        for (int v=0; v<NO_DIM+1; ++v) f[v] = double( mc.v[v]->info().velocity(j) );
                        c[j] = shiftOf( f );
                    }
                for (int v=0; v<NO_DIM+1; ++v)
                    for (size_t j=0; j<noVelComp; ++j)
                        tetVels.push_back( centreThis ? float( double(mc.v[v]->info().velocity(j)) + c[j] )
                                                      : float( mc.v[v]->info().velocity(j) ) );
            }
#endif
#ifdef SCALAR
            if ( needScal )
            {
                double f[NO_DIM+1];
                for (int v=0; v<NO_DIM+1; ++v) f[v] = double( mc.v[v]->info().myScalar()[0] );
                double const c = centreThis ? shiftOf( f ) : 0.;
                for (int v=0; v<NO_DIM+1; ++v)
                    tetScal.push_back( centreThis ? float(f[v] + c) : float(mc.v[v]->info().myScalar()[0]) );
            }
#endif
            tetVols.push_back( float(cellVolume) );
            tetCnts.push_back( cnt );
            tetFlats.push_back( flat );
            tetKeys.push_back( mortonKey( baseGridCell ) );

#ifdef TEST_PADDING
            // the GPU deposits this tetrahedron's field values; replicate only the dummy-cell
            // REPORTING here (dummy tetrahedra live near the padded boundary, so this is cheap)
            bool dummyNeighbors = hasDummyNeighbor( itC );
            bool dummyVertices = hasDummyVertex( itC );
            if ( dummyNeighbors or dummyVertices )
            {
                if ( cnt==0 )
                {
                    if ( dummyNeighbors ) updateDummyGridCells( size_t(flat), &incompleteCells_d );
                    if ( dummyVertices ) updateDummyGridCells( size_t(flat), &incompleteCells );
                }
                else
                {
                    std::vector<Point> randomPoints(cnt);
                    quasiRandomPointsInCell( vertexMatrix, cnt, quasiRandomNumbers, randomPoints.data() );
                    for (size_t i=0; i<cnt; ++i)
                        if ( isGridCellIndex( nGrid, dx, randomPoints[i], basePosition, &index ) )
                        {
                            if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d );
                            if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells );
                        }
                }
            }
#endif
        }

        Avg1GpuCtx gpu{ userOptions, message, nGrid, dx, gridSize, exactAvg, fDen, fVel, fGrad, fScal, fSGrad, needVel,
                        needScal, mortonShift, tetVerts, tetDens, tetVels, tetScal, tetVols, tetCnts, tetFlats, tetKeys,
                        tetCells, { exactAvg ? frame.dxG[0] : double(dx[0]), exactAvg ? frame.dxG[1] : double(dx[1]),
                                    exactAvg ? frame.dxG[2] : double(dx[2]) } };
        avg1MortonOrder( gpu );
        Timer tGpu; tGpu.start();
        metalDeposited = avg1GpuDispatch( gpu, grids, quasiRandomNumbers_buf );
        printComputationTime( &tGpu, &userOptions, "GPU deposit (upload, dispatch, read-back)" );   // (phase_totals.h)
    }
#else
    avg1NoGpuWarning( userOptions );
#endif // DTFE_GPU_ACTIVE


    if ( not metalDeposited )
    {
    // The CPU loop runs on nT = exactAverageThreads() threads, for the exact and the Monte-Carlo average alike, also
    // where this function sees a whole triangulation (a single-domain run; every partition of a '--sample-points' run,
    // DTFE_parallel's single-triangulation branch, including a GPU run's fallback). The x planes are dealt out in
    // interleaved slabs of slabW planes; EVERY thread walks every Delaunay cell in order but deposits only into its own
    // slabs, so each grid cell gets its contributions in the single-threaded order: bit-identical at any thread count
    // (as PS-DTFE's slab deposit). A thread skips a cell outside its slabs before the inverse and the r3d / ps_clip2d
    // clipping or the sampling (the cost). A Monte-Carlo cell's samples are dealt out one by one: each thread maps the
    // cell's samples and deposits those in its own x planes, evaluating nothing for a cell with none (avg1McTet).
    // MY_SCALAR keeps one thread: the user's scalar function is arbitrary C++, not assumed to be thread-safe.
#ifdef MY_SCALAR
    bool constexpr userScalar = true;
#else
    bool constexpr userScalar = false;
#endif
    int const nT = userScalar ? 1 : exactAverageThreads();
    int const slabW = std::max( 1, int( nGrid[0] / size_t( 8 * nT ) ) );   // ~8 slabs per thread: the load balances
    if ( nT > 1 )
        message << ( exactAvg ? "(exact average on " : "(Monte-Carlo average on " ) << nT << " CPU threads, slabs of "
                << slabW << " x planes) " << MESSAGE::Flush;
    std::vector< std::vector<size_t> > incD( static_cast<size_t>( nT ) ), incF( static_cast<size_t>( nT ) );   // TEST_PADDING's flagged cells, per thread
    Avg1CpuCtx const cpu{ field, grids, nGrid, dx, quasiRandomNumbers, centreMC, nT, slabW };
    auto cpuLoop = [&](int const th, std::vector<size_t> &incompleteCells_d, std::vector<size_t> &incompleteCells)
    {
    (void)incompleteCells_d; (void)incompleteCells;
    std::vector<Point> randomPoints;   // this thread's Monte-Carlo samples of one cell (avg1McTet), reused
    std::vector<unsigned char> ownSamples;   // which of them are this thread's (nT > 1: avg1McTet<true>), reused
    size_t prev = 0, amount100 = 0, count = 0;
#if NO_DIM==2
    size_t noTotalCells = dt.number_of_faces();
    for(Finite_cells_iterator itC = dt.finite_faces_begin(); itC!= dt.finite_faces_end(); ++itC)
#elif NO_DIM==3
    size_t noTotalCells = dt.number_of_finite_cells();
    for(Finite_cells_iterator itC = dt.finite_cells_begin(); itC!= dt.finite_cells_end(); ++itC)
#endif
    {
        amount100 = (100 * count++)/ noTotalCells;
        if (th == 0 and prev < amount100)   // (the progress: thread 0's walk, which every thread repeats)
            message.updateProgress( ++prev );


        if ( cellOutsideRegion( dt, itC, depositBox) )
            continue;

        // threads (above), the Monte-Carlo average: a first test on the cell's own vertices, before the vertex order
        // (a hash) and the classification. A cell none of whose x planes -- one more on each side, for the float
        // rounding of the base-relative frame below (2^-24 of the position: a few 1e-4 of a plane at 4096 cells per
        // axis) -- is this thread's is another thread's.
        if ( nT > 1 and not exactAvg )
        {
            double xlo = double( itC->vertex(0)->point()[0] ), xhi = xlo;
            for (int v = 1; v < NO_DIM+1; ++v)
            {
                double const x = double( itC->vertex(v)->point()[0] );
                if ( x < xlo ) xlo = x;
                if ( x > xhi ) xhi = x;
            }
            int a = int( std::floor( ( xlo - double(boxCoordinates[0]) ) / double(dx[0]) ) ) - 1;
            int b = int( std::floor( ( xhi - double(boxCoordinates[0]) ) / double(dx[0]) ) ) + 2;
            if (a < 0) a = 0;
            if (b > int(nGrid[0])) b = int(nGrid[0]);
            bool mine = false;
            for (int gi = a; gi < b and not mine; ++gi) mine = cpu.owns( th, gi );
            if ( not mine ) continue;
        }


#ifdef TEST_PADDING
        Avg1Dummy const dummy{ hasDummyNeighbor( itC ), hasDummyVertex( itC ), &incompleteCells_d, &incompleteCells };
#else
        Avg1Dummy const dummy{};
#endif


        McCell mc = mcVertexOrder( itC, legacyOrder );   // vertex order (base = mc.v[0]): the hashed one; CGAL's under DTFE_MC_LEGACY=1
        double vertexMatrix[NO_DIM][NO_DIM]; // vertex differences of the Delaunay cell
        int baseGridCell[NO_DIM];  // grid cell containing the base point
        Real basePosition[NO_DIM]; // base point position relative to the box's lower-left corner (the Monte-Carlo average)
        TetFrame tf{};             // the exact average's frame (plan P10.8): the base cell and the position in it
        bool single;
        if ( exactAvg )
        {
            tf = tetFrame( mc, frame, vertexMatrix );
            single = tf.single;
            for (int d=0; d<NO_DIM; ++d) baseGridCell[d] = tf.cell[d];
        }
        else
            single = checkCellPosition( mc, boxCoordinates, dx, vertexMatrix, baseGridCell, basePosition ) == SINGLE_GRID_CELL;
        size_t index;

        // threads (above): a cell none of whose x planes this thread owns -- the single grid cell it falls in, or
        // the x range of its clip window (the exact path's own expression, avg1ExactTet) -- is another thread's.
        // A Monte-Carlo cell's samples are convex combinations of the vertex differences (Sobol weights >= 0 whose
        // sum the float table may round above 1 by 2^-23), so a sample's x lies in [lo, hi] up to 1.3e-7 of hi - lo;
        // the window is widened by 1e-6 of it. avg1McTet's per-sample test is the one that decides.
        bool mcSlabs = false;   // a Monte-Carlo cell of the threaded loop (avg1McTet<true>)
        if ( nT > 1 )
        {
            if ( single and isGridCellIndex( baseGridCell, nGrid, &index ) )
            {
                if ( not cpu.owns( th, baseGridCell[0] ) ) continue;
            }
            else
            {
                double lo = 0., hi = 0.;
                for (int v = 0; v < NO_DIM; ++v)
                {
                    if (vertexMatrix[v][0] < lo) lo = vertexMatrix[v][0];
                    if (vertexMatrix[v][0] > hi) hi = vertexMatrix[v][0];
                }
                mcSlabs = not exactAvg;
                if ( mcSlabs )
                {
                    double const eps = 1e-6 * ( hi - lo );
                    lo -= eps;
                    hi += eps;
                }
                int a, b;
                if ( exactAvg )
                {   // the exact path's own window expression (avg1ExactTet), in the grid frame
                    a = tf.cell[0] + int( std::floor( (tf.base[0] + lo) / frame.dxG[0] ) );
                    b = tf.cell[0] + int( std::floor( (tf.base[0] + hi) / frame.dxG[0] ) ) + 1;
                }
                else
                {
                    a = int( std::floor( (double(basePosition[0]) + lo) / double(dx[0]) ) );
                    b = int( std::floor( (double(basePosition[0]) + hi) / double(dx[0]) ) ) + 1;
                }
                if (a < 0) a = 0;
                if (b > int(nGrid[0])) b = int(nGrid[0]);
                bool mine = false;
                for (int gi = a; gi < b and not mine; ++gi) mine = cpu.owns( th, gi );
                if ( not mine ) continue;
            }
        }

        Real cellVolume = volume( dt, itC );
#ifndef MY_SCALAR
        // --exact-average: the inverse and the gradients in double (plan P10.8) and the grid frame's windows. A cell
        // inside one grid cell adds its centroid value times its volume (exact for the linear interpolant); the others
        // the analytic cell∩tet (3D r3d) / cell∩triangle (2D ps_clip2d) integration.
        if ( exactAvg )
        {
            double posInvD[NO_DIM][NO_DIM];
            matrixInverseDouble( vertexMatrix, posInvD );
            if ( single and isGridCellIndex( baseGridCell, nGrid, &index ) )
                avg1SingleCellExact( cpu, mc, posInvD, cellVolume, index, dummy );
            else
                avg1ExactTet( cpu, th, mc, vertexMatrix, tf, frame, posInvD, dummy );
            continue;
        }
#endif
        Real posMatrixInverse[NO_DIM][NO_DIM];  // inverse vertex position-difference matrix
        if ( not mcSlabs )   // (avg1McTet<true> inverts it for a cell with a sample of this thread's only)
            matrixInverse( vertexMatrix, posMatrixInverse );

        // cell fits in one grid cell -> add its centroid value times volume, no MC sampling needed
#ifndef MY_SCALAR
        if ( single and isGridCellIndex( baseGridCell, nGrid, &index ) )
        {
            avg1SingleCell( cpu, mc, posMatrixInverse, cellVolume, index, dummy );
            continue;
        }
#endif


        // cell spans multiple grid cells -> MC-sample inside it
        size_t const noRandomPoints = sampleCount( cellVolume );
        // centring (see mcLegacy above): a sample at p (relative to the base) is DEPOSITED in p's grid cell
        // but EVALUATED at p + delta, delta = sum_j c_j (v_j - v_0) = centroid - mean of the n samples, so
        // the n values of a linear field sum to n f(centroid)
        Real delta[NO_DIM] = {};
        if ( centreMC )
        {
            double cw[NO_DIM];
            centreWeights( noRandomPoints, cw );
            for (int d=0; d<NO_DIM; ++d)
            {
                double t = 0.;
                for (int j=0; j<NO_DIM; ++j) t += cw[j] * vertexMatrix[j][d];
                delta[d] = Real(t);
            }
        }
        if ( mcSlabs )
            avg1McTet<true>( cpu, mc, vertexMatrix, basePosition, posMatrixInverse, cellVolume, noRandomPoints, delta,
                             randomPoints, dummy, th, ownSamples );
        else
            avg1McTet<false>( cpu, mc, vertexMatrix, basePosition, posMatrixInverse, cellVolume, noRandomPoints, delta,
                              randomPoints, dummy, th, ownSamples );
    }
    };   // cpuLoop
    if ( nT == 1 )
        cpuLoop( 0, incD[0], incF[0] );
    else
    {
        std::vector<std::thread> pool;
        for (int th = 1; th < nT; ++th) pool.emplace_back( cpuLoop, th, std::ref( incD[size_t(th)] ), std::ref( incF[size_t(th)] ) );
        cpuLoop( 0, incD[0], incF[0] );
        for (std::thread &x : pool) x.join();
    }
#ifdef TEST_PADDING
    for (int th = 0; th < nT; ++th)   // (showCellsContainingDummyPoints sorts and de-duplicates them)
    {
        incompleteCells_d.insert( incompleteCells_d.end(), incD[size_t(th)].begin(), incD[size_t(th)].end() );
        incompleteCells.insert( incompleteCells.end(), incF[size_t(th)].begin(), incF[size_t(th)].end() );
    }
#endif
    }   // end of the CPU interpolation (skipped when the Metal deposit succeeded)


#if NO_DIM==3 && defined(VELOCITY) && !defined(MY_SCALAR)
    // --face-gradient: the velocity gradient's cell integrals from the faces (avg1FacePass) replace the deposit's;
    // DTFE_FACE_CHECK=1 first compares them (one 'FACE-CHECK' line on stderr per call, i.e. per sub-domain)
    if ( faceSuppliesGradient ) field.velocity_gradient = true;   // (the deposits skipped it; see the top)
    if ( userOptions.faceGradient and field.velocity_gradient )
    {
        int const nTf = exactAverageThreads();
        int const slabWf = std::max( 1, int( nGrid[0] / size_t( 8 * nTf ) ) );
        bool const check = Env::flag( "DTFE_FACE_CHECK" );
        std::vector<double> acc;
        std::vector<unsigned char> degen;
        Timer tFace; tFace.start();
        avg1FacePass( dt, frame, fullBoxExact, nGrid, nTf, slabWf, legacyOrder, acc, check ? &degen : nullptr );
        printComputationTime( &tFace, &userOptions, "face gradient pass (--face-gradient)" );   // (phase_totals.h)
        if ( check )
        {
            // relative to the rms of the deposit's integrals over all cells and components
            double ss = 0.;
            for (size_t c = 0; c < gridSize; ++c)
                for (size_t q = 0; q < noGradComp; ++q) ss += double( (*velocity_gradient)[c][q] ) * double( (*velocity_gradient)[c][q] );
            double const scale = std::sqrt( ss / double( gridSize * noGradComp ) );
            double worst = 0., sum2 = 0.;
            size_t n = 0, above = 0, nDegen = 0;
            for (size_t c = 0; c < gridSize; ++c)
            {
                if ( degen[c] ) { ++nDegen; continue; }
                double cw = 0.;
                for (size_t q = 0; q < noGradComp; ++q)
                {
                    double const e = std::fabs( acc[c*noGradComp + q] - double( (*velocity_gradient)[c][q] ) ) / ( scale > 0. ? scale : 1. );
                    cw = std::max( cw, e );
                    sum2 += e * e;
                }
                worst = std::max( worst, cw );
                if ( cw > 1e-4 ) ++above;
                ++n;
            }
            std::fprintf( stderr, "FACE-CHECK cells %zu compared %zu degenerate %zu max_rel %.3g rms_rel %.3g above_1e-4 %zu "
                          "scale %.6g\n", gridSize, n, nDegen, worst, n ? std::sqrt( sum2 / double( n * noGradComp ) ) : 0.,
                          above, scale );
        }
        for (size_t c = 0; c < gridSize; ++c)
            for (size_t q = 0; q < noGradComp; ++q) (*velocity_gradient)[c][q] = Real( acc[c*noGradComp + q] );
    }
#endif

    avg1Normalize( field, grids, gridCellVolume );
    
    message << "100%\n" << MESSAGE::Flush;
    
    
#ifdef TEST_PADDING
    if (field.density)
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // report cells whose density may be wrong (incomplete tessellation)
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // report cells whose other fields may be wrong (incomplete tessellation)
#endif
}



