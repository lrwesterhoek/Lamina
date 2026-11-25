/* PS-DTFE: the canonical vertex order of the tessellation's cells: every per-tetrahedron quantity (formed in the
   vertex-0 frame) then rounds the same whatever CGAL's insertion history was. Vertices sort by Lagrangian position;
   an odd permutation gets the last two swapped back, so CGAL's positive orientation (the stream parity) is kept. */

#ifndef PS_VERTEX_ORDER_HEADER
#define PS_VERTEX_ORDER_HEADER

#ifdef PHASE_SPACE

#include <atomic>
#include <cstddef>
#include <utility>   // std::swap

#include "triangulation_common.h"   // DT, Point, Cell_handle, Timer, printComputationTime, MESSAGE


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
// One thread per tessellation: CGAL keeps the compact container's bookkeeping in a cell's neighbour slot, so
// renumbering a cell while another thread's iterator passes it would be a data race.
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

// The pass after a tessellation is built; a tessellation loaded from the cache holds the order already.
inline void psVertexOrderPass(DT &dt, User_options &userOptions)
{
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

#endif  // PHASE_SPACE

#endif
