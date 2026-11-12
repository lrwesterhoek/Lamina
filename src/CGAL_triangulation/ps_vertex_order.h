/* PS-DTFE: the canonical vertex order of the tessellation's cells (docs/cleanup-plan.md D19).

   CGAL numbers a cell's vertices in the order its insertion history leaves them. That history differs between the
   sequential and the parallel insert, between two parallel inserts (their threads' timing), with the input's particle
   order and with the partition split. Every per-tetrahedron quantity of PS-DTFE is formed in the cell's vertex-0
   frame -- the Eulerian edge matrix and its inverse, the velocity and scalar gradients, the Lagrangian volume (the
   mass), the periodic wrap to vertex 0's image, the centroid, the exact deposit's clipping planes, the face sections,
   the GPU record (vertex 0's cell and fraction, the edges from it) -- so in CGAL's numbering their rounding follows the
   history: under the parallel insert two identical runs differ in the last bits of the velocity moments (5% of the
   values, at most 3.7e-7 of the rms; plan P11.3), although every grid of the single build is an order-free
   fixed-point sum.

   psCanonicalVertexOrder renumbers every finite cell ONCE, right after the tessellation exists (buildTessellation in
   triangulation.cpp: after the insert, before the vertex densities and the cache save, and after a cache load; the
   composite '--serve' after each of its own cache loads, ps_serve.cc), so every reader -- the CPU deposit and its
   walks, the GPU extraction and kernel, the vertex densities, the point evaluation, '--serve' -- sees one numbering
   whatever the insertion order:
   - the vertices sorted by their Lagrangian position (CGAL's own points; x, then y, then z), which a tessellation
     holds once each, so the order is strict;
   - where that sort is an ODD permutation of CGAL's numbering, the last two swapped back. CGAL numbers every finite
     cell with positive orientation, and the deposits read the sign of the Eulerian edge matrix's determinant as the
     stream's parity (the folded flag, the caustic parity bits, r3d's orientation swap, the kernels' fold test); an
     even permutation keeps that sign. Two CGAL numberings of one cell differ by an even permutation, so the result
     depends on the vertex set alone;
   - each neighbour moves with its vertex (neighbor(i) stays opposite vertex(i)), and every vertex's cell pointer
     stays valid (the cell still holds the vertex). No cell is created, freed or moved, so the iteration orders and
     every handle stay as they are. PS-DTFE inserts, removes or locates nothing afterwards, and the tessellation
     cache stores the numbering as it finds it.
   A Lagrangian periodic copy is an exact shift of its original where the seam allows one (subpartition.h
   psPeriodicSeam), which keeps the order, so a copy of a tetrahedron takes its original's frame; on the fallback
   seam a copy can round off its original, but identical runs make the same copies and so still repeat. The
   deposits' samples are grid points and their containment test is exact and labelling-free (ps_exact_inside.h): the
   order changes no realization, only the rounding.
   One thread per tessellation on purpose: CGAL keeps the compact container's bookkeeping in a cell's neighbour slot
   (Triangulation_ds_cell_base_3::for_compact_container, N[0]), so renumbering a cell while another thread's iterator
   walks past it would be a data race. DTFE_PS_VERTEX_ORDER_LEGACY=1 keeps CGAL's numbering and the cusp stencil's
   address order (ps_interpolation.cc), for an A/B run against grids made without the order; a tessellation loaded
   from a cache file keeps the file's numbering under it, and a partition checkpoint names the switch in its settings
   key (partition_checkpoint.h settingsKey), so '--resume' refuses to mix the two numberings. */

#ifndef PS_VERTEX_ORDER_HEADER
#define PS_VERTEX_ORDER_HEADER

#ifdef PHASE_SPACE

#include <atomic>
#include <cstddef>
#include <utility>   // std::swap

#include "triangulation_common.h"   // DT, Point, Cell_handle, Timer, printComputationTime, MESSAGE
#include "../env_knobs.h"


// DTFE_PS_VERTEX_ORDER_LEGACY=1: CGAL's numbering (read once per process)
inline bool psVertexOrderLegacy()
{
    static bool const legacy = Env::flag( "DTFE_PS_VERTEX_ORDER_LEGACY" );
    return legacy;
}

// strict lexicographic order of two Lagrangian points (a tessellation holds no point twice)
inline bool psLagLess(Point const &a, Point const &b)
{
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const x = double( a[d] ), y = double( b[d] );
        if ( x != y ) return x < y;
    }
    return false;
}

// One finite cell into the canonical order; true when its numbering changes.
inline bool psCanonicalCell(Cell_handle const &c)
{
    int idx[NO_DIM+1];
    for (int i = 0; i <= NO_DIM; ++i) idx[i] = i;
    int swaps = 0;                                   // the permutation's parity: the insertion sort's swaps
    for (int i = 1; i <= NO_DIM; ++i)
        for (int j = i; j > 0 && psLagLess( c->vertex( idx[j] )->point(), c->vertex( idx[j-1] )->point() ); --j)
        {
            std::swap( idx[j], idx[j-1] );
            ++swaps;
        }
    if ( swaps & 1 ) std::swap( idx[NO_DIM-1], idx[NO_DIM] );   // odd: the last two back, CGAL's orientation kept
    bool same = true;
    for (int i = 0; i <= NO_DIM; ++i) same = same && idx[i] == i;
    if ( same ) return false;
    Vertex_handle v[NO_DIM+1];
    Cell_handle n[NO_DIM+1];
    for (int i = 0; i <= NO_DIM; ++i)
    {
        v[i] = c->vertex( idx[i] );
        n[i] = c->neighbor( idx[i] );                // the neighbour opposite a vertex moves with it
    }
#if NO_DIM==3
    c->set_vertices( v[0], v[1], v[2], v[3] );
    c->set_neighbors( n[0], n[1], n[2], n[3] );
#else
    c->set_vertices( v[0], v[1], v[2] );             // 2D: an even permutation of a face is a rotation, so the
    c->set_neighbors( n[0], n[1], n[2] );            // counter-clockwise order holds
#endif
    return true;
}

// The pass over the finite cells; returns how many it renumbers (none for a tessellation of lower dimension).
inline size_t psCanonicalVertexOrder(DT &dt)
{
    if ( dt.dimension() != NO_DIM ) return 0;
    size_t changed = 0;
#if NO_DIM==2
    for (DT::Finite_faces_iterator it = dt.finite_faces_begin(); it != dt.finite_faces_end(); ++it)
#else
    for (DT::Finite_cells_iterator it = dt.finite_cells_begin(); it != dt.finite_cells_end(); ++it)
#endif
    {
        Cell_handle const c = it;
        if ( psCanonicalCell( c ) ) ++changed;
    }
    return changed;
}

// The pass where a tessellation is built or loaded (triangulation.cpp buildTessellation), timed as its own phase
// ("canonical vertex order" in the phase totals). Under DTFE_PS_VERTEX_ORDER_LEGACY=1 the cells keep CGAL's
// numbering, and the first caller that prints warnings says so once per process.
inline void psVertexOrderPass(DT &dt, User_options &userOptions)
{
    if ( psVertexOrderLegacy() )
    {
        static std::atomic<bool> told{ false };
        if ( userOptions.verboseLevel >= 1 and not told.exchange( true ) )
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "DTFE_PS_VERTEX_ORDER_LEGACY=1: the cells keep CGAL's vertex numbering, so the deposits' last "
                    << "bits follow the insertion order (two runs with the parallel insert then differ).\n"
                    << MESSAGE::EndWarning;
        }
        return;
    }
    Timer t;
    t.start();
    size_t const renumbered = psCanonicalVertexOrder( dt );
    printComputationTime( &t, &userOptions, "canonical vertex order" );
    if ( userOptions.verboseLevel >= 3 )
    {
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "(the canonical vertex order renumbered " << renumbered << " cells)\n" << MESSAGE::Flush;
    }
}

// The same without any output, for the tessellations the composite '--serve' loads from the cache itself
// (ps_serve.cc; its loader threads write no lines of their own)
inline void psVertexOrderQuiet(DT &dt)
{
    if ( not psVertexOrderLegacy() ) psCanonicalVertexOrder( dt );
}

#endif  // PHASE_SPACE

#endif
