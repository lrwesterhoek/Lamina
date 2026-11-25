/* Shared PS-DTFE per-cell filter chain (dummy skip, partition ownership, hull handling, wrap, degeneracy test). Every
   consumer (grid deposit, GPU extraction, point evaluation) must classify borderline cells IDENTICALLY, so the chain
   lives here once and runs the same arithmetic in the same precision. */

#ifndef PS_CELL_FILTER_HEADER
#define PS_CELL_FILTER_HEADER

#ifdef PHASE_SPACE   // both consumers (ps_interpolation.cc, ps_point_eval.cc) are PS-only

#include <algorithm>   // std::sort / std::max in the labelling-independent degeneracy scale

// Geometry of a cell that passed psFilterCell().
struct PSCellGeometry
{
    Real   eulerPos[NO_DIM+1][NO_DIM]; // Eulerian vertex positions, minimum-image wrapped
    double Ax[NO_DIM][NO_DIM];         // Eulerian edge matrix (rows = vertices 1..NO_DIM relative to vertex 0)
    double cellDet;                    // determinant(Ax)
    double cellAbsDet;                 // |cellDet|
    Real   posMatInv[NO_DIM][NO_DIM];  // matrixInverse(Ax); filled only when checkSingularInverse is true
    bool   useVolumeRatioDensity;      // zero/negative-density hull vertex found (kept: non-periodic cloud surface)
};

// Runs the filter chain on one cell; false = skip it, else 'g' holds the wrapped geometry. checkSingularInverse also
// drops a zero matrixInverse() (it reads as "inside" for the whole bbox; the Metal kernel mirrors it, keep the two in step).
// skipOwnership keeps every chart-independent filter but not the ownership tiling (vertex degrees count globally).
// No early rejection (the default of psFilterCell's last argument).
struct PSNoEarlyReject { bool operator()(Real const (&)[NO_DIM+1][NO_DIM]) const { return false; } };

// earlyReject(eulerPos) drops a cell on its wrapped positions alone, before the edge matrix (default: none).
template <class EarlyReject = PSNoEarlyReject>
inline bool psFilterCell(Cell_handle &cell,
                         User_options &userOptions,
                         Box &boxCoordinates,
                         bool const checkSingularInverse,
                         size_t *nDegenerateInverse,
                         PSCellGeometry &g,
                         bool const skipOwnership = false,
                         EarlyReject const &earlyReject = EarlyReject())
{
#ifdef TEST_PADDING
    // Skip cells touching a dummy padding vertex.
    {
        bool hasDummy = false;
        for (int v = 0; v <= NO_DIM; ++v)
            if (cell->vertex(v)->info().isDummy()) { hasDummy = true; break; }
        if (hasDummy) { return false; }
    }
#endif

    // Partition ownership: a cell is kept only if its Lagrangian centroid lies in the primary box, decided per edge as
    // sum_v (x_v - lo) >= 0 and sum_v (x_v - hi) < 0. Each x_v - edge is exact in double, and so is the sum where the
    // answer is close, in any vertex order: a tetrahedron and its periodic copy (exact shift by L) get complementary
    // answers bit for bit. (Summing the raw coordinates first can round: both or neither owned.)
    if ( !skipOwnership && !userOptions.lagrangianRegion.isNullBox() )
    {
        bool owned = true;
        for (int d = 0; d < NO_DIM && owned; ++d)
        {
            double const lo = double(userOptions.lagrangianRegion[2*d]);
            double const hi = double(userOptions.lagrangianRegion[2*d+1]);
            double sLo = 0., sHi = 0.;
            for (int v = 0; v <= NO_DIM; ++v)
            {
                double const x = double(cell->vertex(v)->point()[d]);
                sLo += x - lo;
                sHi += x - hi;
            }
            if ( sLo < 0. || sHi >= 0. ) owned = false;
        }
        if (!owned) { return false; }
    }

    // Non-periodic clouds (--ps-alpha-shape): keep only tetrahedra whose Lagrangian circumradius is at most
    // psLagAlphaRadius, dropping hull slivers. A function of the four vertices only, and a tetrahedron within the cut
    // is in a partition's triangulation exactly when in the global one (the plan pads by >= 4 cuts). Applies with
    // skipOwnership too.
    if ( userOptions.psLagAlphaRadius > 0. )
    {
#if NO_DIM==3
        double e[3][3];
        for (int v = 0; v < 3; ++v)
            for (int d = 0; d < 3; ++d)
                e[v][d] = double(cell->vertex(v+1)->point()[d]) - double(cell->vertex(0)->point()[d]);
        auto cross = [](double const *a, double const *b, double *c)
        { c[0] = a[1]*b[2] - a[2]*b[1]; c[1] = a[2]*b[0] - a[0]*b[2]; c[2] = a[0]*b[1] - a[1]*b[0]; };
        double bc[3], ca[3], ab[3];
        cross(e[1], e[2], bc); cross(e[2], e[0], ca); cross(e[0], e[1], ab);
        double const den = 2. * (e[0][0]*bc[0] + e[0][1]*bc[1] + e[0][2]*bc[2]);
        double n2[3];
        for (int v = 0; v < 3; ++v) n2[v] = e[v][0]*e[v][0] + e[v][1]*e[v][1] + e[v][2]*e[v][2];
        double num2 = 0.;
        for (int d = 0; d < 3; ++d)
        {
            double const c = n2[0]*bc[d] + n2[1]*ca[d] + n2[2]*ab[d];
            num2 += c*c;
        }
        // circumradius = |num| / |den|; a flat tetrahedron (den -> 0) is beyond any cut
        double const a = userOptions.psLagAlphaRadius;
        if ( !( num2 <= a*a * den*den ) || den == 0. ) { return false; }
#else
        double const p0x = cell->vertex(0)->point()[0], p0y = cell->vertex(0)->point()[1];
        double const ax = cell->vertex(1)->point()[0] - p0x, ay = cell->vertex(1)->point()[1] - p0y;
        double const bx = cell->vertex(2)->point()[0] - p0x, by = cell->vertex(2)->point()[1] - p0y;
        double const den = 2. * (ax*by - ay*bx);
        double const ux = by*(ax*ax + ay*ay) - ay*(bx*bx + by*by), uy = ax*(bx*bx + by*by) - bx*(ax*ax + ay*ay);
        double const a = userOptions.psLagAlphaRadius;
        if ( !( ux*ux + uy*uy <= a*a * den*den ) || den == 0. ) { return false; }
#endif
    }

    // zero/negative-density vertex sits on the Lagrangian convex hull. Periodic: artefact, drop the
    // cell. Non-periodic: real cloud surface, give it a constant volume-ratio density (avgDensity * V_Lag/V_Eul).
    g.useVolumeRatioDensity = false;
    {
        bool hasBadVertex = false;
        for (int v = 0; v <= NO_DIM; ++v)
            if (cell->vertex(v)->info().density() <= Real(0.)) { hasBadVertex = true; break; }
        if (hasBadVertex)
        {
            if ( userOptions.periodic ) { return false; }
            g.useVolumeRatioDensity = true;
        }
    }

    // Gather the Eulerian positions; the minimum-image wrap below keeps boundary cells from spanning the periodic box.
    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            g.eulerPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);

    if ( userOptions.periodic )
    {
        Real boxLen[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
            boxLen[d] = boxCoordinates[2*d+1] - boxCoordinates[2*d];

        // Wrap vertices 1..NO_DIM to their nearest image of vertex 0.
        for (int v = 1; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real diff = g.eulerPos[v][d] - g.eulerPos[0][d];
                if (diff >  boxLen[d] * Real(0.5)) g.eulerPos[v][d] -= boxLen[d];
                if (diff < -boxLen[d] * Real(0.5)) g.eulerPos[v][d] += boxLen[d];
            }
    }

    if ( earlyReject(g.eulerPos) ) { return false; }

    // Eulerian edge matrix (rows = vertices 1..NO_DIM relative to vertex 0, possibly wrapped).
    for (int v = 0; v < NO_DIM; ++v)
        for (int i = 0; i < NO_DIM; ++i)
            g.Ax[v][i] = double(g.eulerPos[v+1][i]) - double(g.eulerPos[0][i]);

    g.cellDet = determinant(g.Ax);
    g.cellAbsDet = std::fabs(g.cellDet);

    // Drop degenerate cells: |det| < tol * edge^3. The edge scale is the LARGEST over the vertices of the mean squared
    // edge length there, a function of the tetrahedron and not of CGAL's labelling (vertex 0 follows the insertion
    // history), so a tet is kept or dropped identically in every run. It bounds every labelling's vertex-0 scale, so a
    // kept tet also passes matrixInverse()'s relative test (AM-GM) and the GPU kernels' float re-check.
    double const DEGENERATE_DET_TOL = 1.e-6;
    {
        // squared edge lengths, each unordered pair once (wrapped positions, the frame of Ax)
        double e2[NO_DIM+1][NO_DIM+1];
        for (int v = 0; v <= NO_DIM; ++v)
        {
            e2[v][v] = 0.;
            for (int w = v + 1; w <= NO_DIM; ++w)
            {
                double len2 = 0.;
                for (int i = 0; i < NO_DIM; ++i)
                {
                    double const d = double(g.eulerPos[w][i]) - double(g.eulerPos[v][i]);
                    len2 += d * d;
                }
                e2[v][w] = e2[w][v] = len2;
            }
        }
        double maxAvgEdge2 = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
        {
            // the star's lengths summed in ascending order: the same bits whatever the labelling
            double star[NO_DIM];
            int n = 0;
            for (int w = 0; w <= NO_DIM; ++w)
                if (w != v) star[n++] = e2[v][w];
            std::sort(star, star + NO_DIM);
            double avgEdge2 = 0.;
            for (int k = 0; k < NO_DIM; ++k) avgEdge2 += star[k];
            avgEdge2 /= NO_DIM;
            maxAvgEdge2 = std::max(maxAvgEdge2, avgEdge2);
        }
        double edgeScale = maxAvgEdge2 * std::sqrt(maxAvgEdge2); // (max over vertices of the mean edge)^3
        if (g.cellAbsDet < DEGENERATE_DET_TOL * edgeScale) { return false; }
    }

    if ( checkSingularInverse )
    {
        matrixInverse(g.Ax, g.posMatInv);
        bool singularInverse = true;
        for (int a = 0; a < NO_DIM && singularInverse; ++a)
            for (int b = 0; b < NO_DIM; ++b)
                if (g.posMatInv[a][b] != Real(0.)) { singularInverse = false; break; }
        if (singularInverse)
        {
            if (nDegenerateInverse) ++(*nDegenerateInverse);
            return false;
        }
    }

    return true;
}


/* --ps-vertex-mass, shared by the grid deposit and the point evaluation so both read one definition.
   psForEachDepositingCell calls f(cell, geometry) for every cell that deposits (all drop filters, ownership skipped).
   psVertexDegreePass counts those cells per vertex after a reset, so repeated passes give the same counts; a
   tetrahedron's mass is the sum of its vertices' weight/degree shares, so the masses add up to the particles' exactly. */
template <class F>
inline void psForEachDepositingCell(DT &dt, User_options &userOptions, Box &boxCoordinates, F &&f)
{
    size_t degDropped = 0;   // local counter: the deposit loop reports its own
#if NO_DIM==2
    for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
#else
    for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
#endif
    {
        Cell_handle cell = itC;
        PSCellGeometry dgeo;
        if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degDropped, dgeo,
                           /*skipOwnership=*/true) )
            continue;
        f( cell, dgeo );
    }
}

inline void psVertexDegreePass(DT &dt, User_options &userOptions, Box &boxCoordinates)
{
    for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
        vit->info().psDegree() = 0;
    psForEachDepositingCell( dt, userOptions, boxCoordinates, [](Cell_handle &cell, PSCellGeometry const &)
    {
        for (int v = 0; v <= NO_DIM; ++v)
            cell->vertex(v)->info().psDegree() += 1;
    } );
}

// per-tet vertex-share mass: sum over the cell's vertices of weight/degree (a non-positive degree contributes nothing)
inline double psVertexShareMass(Cell_handle const &cell)
{
    double m = 0.;
    for (int v = 0; v <= NO_DIM; ++v)
    {
        int32_t const deg = cell->vertex(v)->info().psDegree();
        if (deg > 0)
            m += double(cell->vertex(v)->info().weight()) / double(deg);
    }
    return m;
}

#endif  // PHASE_SPACE

#endif
