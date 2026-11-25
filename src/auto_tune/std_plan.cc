// Part of src/auto_tune.cc, included once by that file (never compiled alone). Standard DTFE only: the density replay,
// the look-ahead's bytes (stdLookAheadBytes, declared in auto_tune.h) and the planner (AutoTuneStdPlanner).

#ifndef PHASE_SPACE
namespace {

/* ---- Standard DTFE: what one partition's thread sub-triangulations hold, from the data ---------------------------
   The plan replays the run's own split on the particles (the box-average count per sub-box is wrong on clustered data: the
   padding is a fixed length): a count grid with a summed-area table gives the particles in any box; the partition split and
   each thread grid are replayed with the run's functions and every padded thread box is counted. The helpers below itemize a
   partition's bytes; the plan is the busiest partition's sum on top of the particles and the loop's grids ('fixedLoop'). */

// The particles in any box, from a coarse count grid and its summed-area table: the density is taken as constant
// inside a coarse cell, so the cumulative count is multilinear in each cell and a box costs 2^NO_DIM interpolations.
// Periodic: the grid covers the box and wraps (a box reaching past a face counts the periodic images, as
// findParticlesInBox selects them); otherwise it covers the box plus its padding (a '--region' crop's particles).
struct StdTuneDensity
{
    bool   ok = false, periodic = false;
    size_t m[NO_DIM] = {};               // cells per axis
    double lo[NO_DIM] = {}, cell[NO_DIM] = {};
    std::vector<double> sat;             // (m0+1) x (m1+1) [x (m2+1)]: the particles in [0,i) x [0,j) [x [0,k)]

    void build(std::vector<Particle_data> const &p, User_options const &u)
    {
        periodic = u.periodic;
        // 128 cells across the box per axis (1024 in 2D); without wrap whole cells of the same size continue over the
        // padding, so a box's cells are the same with or without it (a '--region' that is the whole box counts like
        // the periodic box). A padding that would take the grid past maxCells widens every cell instead.
        size_t const perAxis = NO_DIM == 3 ? 128 : 1024;
        double const maxCells = NO_DIM == 3 ? 4.2e6 : 2.1e6;
        double boxLo[NO_DIM], boxLen[NO_DIM], padLo[NO_DIM], padHi[NO_DIM], scale = 1.;
        for (int pass=0; pass<2; ++pass)
        {
            double total = 1.;
            for (int d=0; d<NO_DIM; ++d)
            {
                boxLo[d]  = double(u.boxCoordinates.coords[2*d]);
                boxLen[d] = double(u.boxCoordinates.coords[2*d+1]) - boxLo[d];
                if ( not ( boxLen[d] > 0. ) ) return;
                padLo[d] = periodic ? 0. : std::max( 0., double(u.paddingLength.coords[2*d]) );
                padHi[d] = periodic ? 0. : std::max( 0., double(u.paddingLength.coords[2*d+1]) );
                cell[d] = scale * boxLen[d] / double(perAxis);
                size_t const below = size_t( std::ceil( padLo[d] / cell[d] ) ), above = size_t( std::ceil( padHi[d] / cell[d] ) );
                m[d] = size_t( std::ceil( boxLen[d] / cell[d] - 1.e-9 ) ) + below + above;
                lo[d] = boxLo[d] - double(below) * cell[d];
                total *= double(m[d]);
            }
            if ( total <= maxCells ) break;
            scale *= std::pow( total / maxCells, 1. / double(NO_DIM) ) * 1.01;
        }
        size_t cells = 1, satCells = 1;
        for (int d=0; d<NO_DIM; ++d)
        {
            cells *= m[d];
            satCells *= m[d] + 1;
        }
        std::vector<uint32_t> count( cells, 0u );
        long const n = long( p.size() );
#ifdef OPEN_MP
        #pragma omp parallel for schedule(static)
#endif
        for (long i=0; i<n; ++i)
        {
            size_t k = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                double t = ( double(p[size_t(i)].pos[d]) - lo[d] ) / cell[d];
                if ( periodic ) t -= double(m[d]) * std::floor( t / double(m[d]) );
                long c = std::isfinite( t ) ? long( std::floor( t ) ) : 0L;
                c = std::max( 0L, std::min( long(m[d]) - 1, c ) );
                k = k * m[d] + size_t(c);
            }
#ifdef OPEN_MP
            #pragma omp atomic
#endif
            ++count[k];
        }
        // the summed-area table: the counts placed at index + 1, then prefix sums along every axis
        sat.assign( satCells, 0. );
        for (size_t k=0; k<cells; ++k)
        {
            size_t r = k, s = 0, stride = 1;
            for (int d=NO_DIM-1; d>=0; --d)
            {
                s += ( r % m[d] + 1 ) * stride;
                r /= m[d];
                stride *= m[d] + 1;
            }
            sat[s] = double( count[k] );
        }
        size_t stride = 1;
        for (int d=NO_DIM-1; d>=0; --d)
        {
            size_t const len = m[d] + 1;
            for (size_t k=0; k<satCells; ++k)
                if ( ( k / stride ) % len != 0 )
                    sat[k] += sat[k - stride];
            stride *= len;
        }
        ok = true;
    }

    // the summed-area table at a fractional cell position (each coordinate in [0, m])
    double table(double const t[NO_DIM]) const
    {
        size_t i0[NO_DIM];
        double f[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            double const c = std::max( 0., std::min( double(m[d]), t[d] ) );
            i0[d] = std::min( m[d] - 1, size_t( c ) );
            f[d] = c - double(i0[d]);
        }
        double v = 0.;
        for (int corner=0; corner<(1<<NO_DIM); ++corner)
        {
            double w = 1.;
            size_t s = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                bool const up = ( corner >> d ) & 1;
                w *= up ? f[d] : 1. - f[d];
                s = s * ( m[d] + 1 ) + i0[d] + ( up ? 1 : 0 );
            }
            if ( w != 0. ) v += w * sat[s];
        }
        return v;
    }

    // the particles with every coordinate below x (periodic: counting the images below the box, negatively, as well)
    double cumulative(double const x[NO_DIM]) const
    {
        double whole[NO_DIM], part[NO_DIM];   // per axis: periods below x, and the position within the period
        for (int d=0; d<NO_DIM; ++d)
        {
            double const t = ( x[d] - lo[d] ) / cell[d];
            whole[d] = periodic ? std::floor( t / double(m[d]) ) : 0.;
            part[d]  = t - whole[d] * double(m[d]);
        }
        double v = 0.;
        for (int combo=0; combo<(1<<NO_DIM); ++combo)
        {
            double w = 1., t[NO_DIM];
            for (int d=0; d<NO_DIM and w != 0.; ++d)
            {
                bool const full = ( combo >> d ) & 1;   // the whole periods below, or the part of the last one
                w *= full ? whole[d] : 1.;
                t[d] = full ? double(m[d]) : part[d];
            }
            if ( w != 0. ) v += w * table( t );
        }
        return v;
    }

    // the particles in the box [a, b)
    double count(double const a[NO_DIM], double const b[NO_DIM]) const
    {
        double v = 0.;
        for (int corner=0; corner<(1<<NO_DIM); ++corner)
        {
            double x[NO_DIM];
            int low = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                bool const up = ( corner >> d ) & 1;
                x[d] = up ? b[d] : a[d];
                low += up ? 0 : 1;
            }
            v += ( low % 2 ? -1. : 1. ) * cumulative( x );
        }
        return std::max( 0., v );
    }
};

// optimalPartitionSplit's split of a box of the run's grid -- x first, then y within each x slab, then z within each
// column, every cut where optimalSplit puts it -- replayed on the count grid. Ranges are in grid cells relative to
// 'origin' (the box's lower corner) with cell size dx; counts are per cell of the run's grid (the count grid interpolates
// inside its coarse cells). An axis with fewer cells than pieces (the run refuses that split) is cut into equal pieces.
inline std::vector< std::vector<double> > stdTuneSplit(StdTuneDensity const &dens, double const origin[NO_DIM],
                                                       double const dx[NO_DIM], std::vector<double> const &range,
                                                       size_t const parts[NO_DIM])
{
    std::vector< std::vector<double> > out;
    std::vector<double> cur( range );
    std::function<void(int)> recurse = [&](int d)
    {
        if ( d == NO_DIM ) { out.push_back( cur ); return; }
        double const a = range[2*d], b = range[2*d+1];
        size_t const cells = size_t( std::llround( b - a ) );
        std::vector<double> edges( parts[d] + 1, a );
        if ( parts[d] <= 1 )
            edges[1] = b;
        else if ( cells < parts[d] )
            for (size_t k=0; k<=parts[d]; ++k) edges[k] = a + ( b - a ) * double(k) / double(parts[d]);
        else
        {
            std::vector<size_t> counts( cells ), idx( parts[d] + 1, 0 );
            for (size_t i=0; i<cells; ++i)
            {
                double lo[NO_DIM], hi[NO_DIM];
                for (int e=0; e<NO_DIM; ++e)
                {
                    double const ra = ( e == d ) ? a + double(i) : cur[2*e];
                    double const rb = ( e == d ) ? a + double(i + 1) : cur[2*e+1];
                    lo[e] = origin[e] + ra * dx[e];
                    hi[e] = origin[e] + rb * dx[e];
                }
                counts[i] = size_t( std::llround( dens.count( lo, hi ) ) );
            }
            optimalSplit( counts, parts[d], idx.data() );
            for (size_t k=0; k<=parts[d]; ++k) edges[k] = a + double( std::min( idx[k], cells ) );
            edges[0] = a; edges[parts[d]] = b;
        }
        for (size_t k=0; k<parts[d]; ++k)
        {
            cur[2*d] = edges[k];
            cur[2*d+1] = edges[k+1];
            recurse( d + 1 );
        }
        cur[2*d] = a; cur[2*d+1] = b;
    };
    recurse( 0 );
    return out;
}

/* The itemized bytes of a standard tessellation, its build and its deposit. Fitted on the partitions of TNG100-3-Dark and
   TNG50-3-Dark z=0 at 512^3 (Exact on the GPU, the production fields) with freed memory handed back (main.cpp
   dtfeAllocatorReexec, triangulation.cpp dtfeReleaseTessellationMemory, dtfe_metal_host.cc releaseBuffer), so that a footprint is live bytes. */
// a vertex's vertexData (vertexData.h): Data_structure (weight, density, the velocity, the scalar: 6 Reals), the
// padding test's two flags (8 with padding) and the star sum's two 8-B limbs
inline double stdVertexDataBytes() { return 6. * double(sizeof(Real)) + 8. + 16.; }
// ... a finished tessellation per vertex: kStdVertexBytes, its vertexData widened in the double build
inline double stdBytesPerTessVertex() { return kStdVertexBytes + ( stdVertexDataBytes() - 48. ); }
// ... of V padded points with the padding test's dummy points, 6 V^(2/3) on the padded box's faces (triangulation.cpp)
inline double stdTessBytes(double const V)
{
    return V > 0. ? stdBytesPerTessVertex() * ( V + 6. * std::cbrt( V * V ) ) : 0.;
}
/* ... and the insert's transient beside it at its end, the build's peak: the parallel insert's point+payload copy
   (Point_3<Epick> 24 + vertexData) and CGAL's own index, point and info vectors (8 + 24 + vertexData), grown by
   push_back to the next power of two (g in [1, 2)); the sequential insert's particle copy, freed when it is done */
inline double stdInsertBytes(double const V, bool const parallel)
{
    if ( not ( V > 0. ) ) return 0.;
    if ( not parallel ) return V * double(sizeof(Particle_data));
    double const g = V > 1. ? std::exp2( std::ceil( std::log2( V ) ) ) / V : 1.;
    return V * ( ( 24. + stdVertexDataBytes() ) + ( 8. + 24. + stdVertexDataBytes() ) * g );
}
// The GPU deposit's extracted tetrahedra of a box with 'own' particles: every finite cell whose bounding box meets the box
// (averaged_interpolation_1.cc cellOutsideRegion), kTetsPerVertex per particle and the ones crossing its faces, a shell about
// one mean spacing deep on its 6 faces, booked kStdFaceShell times (fitted on the 27 and 64 partitions of TNG100-3 and TNG50-3)
inline double stdTetsOfBox(double const own)
{
    return own > 0. ? kTetsPerVertex * own * ( 1. + kStdFaceShell * 6. / std::cbrt( own ) ) : 0.;
}

}  // namespace

// (auto_tune.h)
double stdLookAheadBytes(User_options const &u, double const nOwn, double const len[NO_DIM],
                         double const pad[NO_DIM], size_t const tGrid[NO_DIM])
{
    double fPart = 1., fThreads = 1., team = 1.;
    for (int d=0; d<NO_DIM; ++d)
    {
        double const l = std::max( len[d], 1e-300 );
        double const t = double( std::max<size_t>( tGrid[d], 1 ) );
        fPart    *= 1. + 2.*pad[d] / l;
        fThreads *= 1. + 2.*pad[d]*t / l;
        team     *= t;
    }
    // every box's build (its tessellation and the insert's transient) and, for a team, the partition's particle copy
    double const padded = nOwn * fThreads / team;
    return team * ( stdTessBytes( padded ) + stdInsertBytes( padded, autoTuneParallelInsert( u, padded, int( team ) ) ) )
           + ( team > 1. ? nOwn * fPart * double(sizeof(Particle_data)) : 0. );
}

namespace {

// The busiest partition of a split (P per axis, or the user's own) with up to T threads in each: its bytes on top of 'fixed',
// the padded vertices its sub-triangulations hold, and those per thread (DTFE_AUTO_TUNE_DEBUG) (+ the GPU snapshot of its
// largest thread sub-grid, and the look-ahead's booked bytes: the tessellation stage of the partition that needs the most)
// (+ bytesNoAhead: the busiest partition's bytes where no look-ahead runs after all: stdLonePhases)
struct StdPlan { double bytes = 0., vertices = 0., snapshot = 0., ahead = 0., bytesNoAhead = 0.; int part = -1;
                 size_t parts = 1; std::vector<double> perThread; };

/* One tessellation per partition (a partition loop without DTFE_parallel's thread split) where no look-ahead runs:
   the partition's padded particle copy is the tessellation's input -- the sequential insert frees it once the
   tessellation is built, the parallel one as soon as its own copies of the points exist (triangulation.cpp) -- so
   beside its tessellation (tess) the partition holds the insert's transient (insert, stdInsertBytes) or, once the
   insert is over, the deposit's arrays and grids (deposit), not both: its peak is the larger phase. */
inline double stdLonePhases(double const tess, double const deposit, double const insert)
{
    return tess + std::max( deposit, insert );
}

// DTFE_parallel's team at up to t threads: the largest 2^a 3^b <= t (subpartition.h parallelGrid; its grid-shape step
// leaves fewer only on a grid too small to split)
inline int stdTeamSize(int const t)
{
    int best = 1;
    for (long p2 = 1; p2 <= long(t); p2 *= 2)
        for (long p = p2; p <= long(t); p *= 3)
            best = std::max( best, int( p ) );
    return best;
}

/* What the standard-DTFE plan reads: its inputs, resolved once by autoTunePartitioning from the options, the data and
   the machine. Values, set by name there; the planner below only reads them. */
struct AutoTuneStdInputs
{
    bool metalActive = false;        // the GPU deposit will run (flag given AND compiled in)
    bool autoPart = false;           // --partition left to the tuner
    bool autoConc = false;           // --max-concurrent left to the tuner
    double N = 0.;                   // the particles (a '--region' run: the region's, unpadded)
    double ramBytes = 0.;            // physical RAM
    std::string budgetWhy;           // how the memory budget was set (the AUTO-TUNE line)
    double budget = 0.;              // the memory budget
    int cores = 1;                   // the threads
    double gridTotal = 0.;           // the grid's cells
    double bCellDeposit = 0.;        // full-grid bytes per cell during the deposit
    double padFrac[NO_DIM] = {};     // the padding over the box length, per axis
    double partBytes = 0.;           // bytes per particle (sizeof(Particle_data))
    size_t nPts = 0;                 // the '--sample-points'
    double ptsBytes = 0.;            // ... their structures (in 'fixed')
    double ptsWalkBytes = 0.;        // ... the point-walk threads' record blocks, per partition
    double fixed = 0.;               // the irreducible term: particles + grids + points
    double fixedLoop = 0.;           // ... while the partition loop runs: its grids only (bCellDeposit)
    bool gpuDeposit = false;         // the '_a' grids deposit on the GPU
    double bGpuCell = 0.;            // ... its bytes per sub-grid cell of the grids it deposits, as 4-B floats
    double faceCell = 0.;            // --face-gradient: avg1FacePass's 9 sums of 8 B per sub-domain cell
    double bCpuWordCell = 0.;        // the CPU exact average's run-wide words per cell of each sub-domain (Avg1Words)
    double faceWayOutCell = 0.;      // what '--per-tet-gradient' books less per sub-domain cell on the CPU deposit
                                     // (text only: the 'no split fits' warning's way out; none on the GPU deposit)
    double perTet = 0.;              // ... the GPU deposit's bytes per extracted tetrahedron
    double snapWordFactor = 0.;      // the GPU snapshot's bytes per 4 B grid component (2: 64-bit words; 4: 128-bit)
    double deviceWordFactor = 0.;    // ... and the device grids' (the same words)
    bool lookAheadPossible = false;  // the partition look-ahead runs (GPU deposit, more than one partition)
    double scratchZeroFill = 0.;     // the --scratch-dir start-up peak, macOS (auto_tune.cc: 0 where there is none)
    std::chrono::steady_clock::time_point tuneStart;   // when the tuner started (DTFE_AUTO_TUNE_DEBUG)
    double countSec = 0.;            // ... and the seconds its count grid took
    bool oneTessellation = false;    // '--sample-points': one tessellation per partition (no thread split)
    double origin[NO_DIM] = {};      // the grid's lower corner
    double dxGrid[NO_DIM] = {};      // ... and its cell size
    std::vector<size_t> userSplit;   // the user's --partition (all 1 when the tuner chooses)
};

/* The standard-DTFE plan once its inputs are known (autoTunePartitioning): the memoised plan of a split (planOf) and
   the phases that choose the split (chooseSplit) and state it (reportSplit). It reads the inputs by name (its base),
   the particles' count grid through 'dens', and writes only through the references and its own memo. */
struct AutoTuneStdPlanner : AutoTuneStdInputs
{
    User_options &u;                 // the options the plan sets: --partition, --max-concurrent, the look-ahead
    AutoTuneReport &report;          // the report it fills
    MESSAGE::Message &message;       // the AUTO-TUNE lines
    StdTuneDensity const &dens;      // the particles' count grid (not ok: the uniform count)
    mutable std::map< std::pair<int,int>, StdPlan > plans;   // planOf's memo, per (P, T)
    mutable User_options pgOptions;  // parallelGrid reads a partition's own grid from this copy of the options

    // the plan of P partitions per axis (or the user's split) with up to T threads each, computed once per pair
    StdPlan const &planOf(int P, int T) const
    {
        int const threads = ( oneTessellation or T <= 1 ) ? 1 : std::min( T, cores );
        std::pair<int,int> const key( P, threads );
        auto const found = plans.find( key );
        if ( found != plans.end() ) return found->second;
        StdPlan plan;
        size_t partsP[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) partsP[d] = autoPart ? size_t(P) : userSplit[d];
        double nParts = 1.;
        for (int d=0; d<NO_DIM; ++d) nParts *= double( partsP[d] );
        plan.parts = size_t( nParts );
        // a partition's grids: its own (DTFE_parallel's, when split), the threads' sub-grid copies, each sub-domain
        // call's host grid of the GPU deposit (bGpuCell: allocated when the call starts, so every sub-domain of a team
        // holds its own), the face pass's sums and the CPU exact average's words
        auto gridBytes = [&](double cells, size_t nThreads) {
            return ( nParts > 1. ? cells * bCellDeposit : 0. ) + ( nThreads > 1 ? cells * bCellDeposit : 0. )
                   + cells * ( bGpuCell + faceCell + bCpuWordCell ); };
        /* One partition from its thread boxes (own particles, padded particles, sub-grid cells each; one box: one tessellation,
           several: DTFE_parallel's team), its padded particle copy and its cells. Every box holds its tessellation (stdTessBytes)
           and, while it is built, the insert's transient (stdInsertBytes) or, while it deposits, its extracted tets (stdTetsOfBox x
           perTet); the device grids are those of the batch holding the device and of the one converting its words, of the largest box. */
        struct PartBytes { double bytes = 0., noAhead = 0., ahead = 0., vertices = 0., maxCells = 0.; };
        auto partitionBytes = [&](std::vector< std::array<double,3> > const &boxes, double const partPadded,
                                  double const cells) -> PartBytes
        {
            size_t const nThreads = boxes.size();
            bool const lone = nThreads == 1 and nParts > 1.;   // one tessellation per partition
            PartBytes r;
            double tess = 0., build = 0., worst = 0., arrays = 0.;
            for (std::array<double,3> const &b : boxes)
            {
                double const tessB = stdTessBytes( b[1] );
                double const insertB = stdInsertBytes( b[1], autoTuneParallelInsert( u, b[1], int( nThreads ) ) );
                double const arraysB = stdTetsOfBox( b[0] ) * perTet;
                tess += tessB;
                build += tessB + insertB;
                worst += tessB + std::max( insertB, arraysB );   // a team's box: building or depositing
                arrays += arraysB;
                r.vertices += b[1];
                r.maxCells = std::max( r.maxCells, b[2] );
            }
            double const device = deviceWordFactor * bGpuCell * r.maxCells * double( std::min<size_t>( 2, nThreads ) );
            double const grids = gridBytes( cells, nThreads ) + ( nPts > 0 ? ptsWalkBytes : 0. ) + device;
            if ( lone )
            {   // the partition's copy is the tessellation's input, freed by the insert: the insert's transient or the
                // deposit beside the tessellation, not both (stdLonePhases). With the look-ahead the next partition's
                // build is booked apart (StdPlan::ahead) while this one deposits.
                r.noAhead = stdLonePhases( tess, arrays + grids, build - tess );
                r.bytes = lookAheadPossible ? tess + arrays + grids : r.noAhead;
                r.ahead = build;
            }
            else
            {   // a team (or one domain): its boxes, the partition's particle copy (DTFE_parallel keeps its capacity to
                // the partition's end) or one domain's periodic copies / a '--region' crop's padding beyond the N in
                // 'fixed', and the grids; a look-ahead builds every box of the next partition beside its copy
                r.bytes = r.noAhead = worst + ( nParts > 1. ? partPadded : std::max( 0., partPadded - N ) ) * partBytes
                                      + grids;
                r.ahead = build + partPadded * partBytes;
            }
            return r;
        };
        if ( not dens.ok )
        {   // every sub-box holds the box average (the uniform count), on the team the run forms: DTFE_parallel's
            // thread grid of a partition's grid (parallelGrid: 2^a 3^b threads, 4 of 5), each sub-box padded on its own
            // faces -- as the counted branch below, and as the loop's own uniform look-ahead count (stdLookAheadBytes)
            size_t g[NO_DIM];
            for (int d=0; d<NO_DIM; ++d)
            {
                g[d] = 1;
                pgOptions.gridSize[d] = size_t( std::max( 1LL, (long long)std::llround( double( u.gridSize[d] )
                                                                                        / double( partsP[d] ) ) ) );
            }
            if ( threads > 1 )
                parallelGrid( size_t(threads), pgOptions, g );
            size_t team = 1;
            for (int d=0; d<NO_DIM; ++d) team *= g[d];
            double f = 1., fPart = 1.;
            for (int d=0; d<NO_DIM; ++d)
            {
                f     *= 1. + 2.*padFrac[d]*double(partsP[d])*double(g[d]);
                fPart *= 1. + 2.*padFrac[d]*double(partsP[d]);
            }
            double const own = N / nParts;
            std::vector< std::array<double,3> > const boxes( team, std::array<double,3>{ {
                own / double( team ), own * f / double( team ), gridTotal / nParts / double( team ) } } );
            PartBytes const r = partitionBytes( boxes, own * fPart, gridTotal / nParts );
            plan.part = 0;
            plan.vertices = r.vertices;
            plan.bytes = r.bytes;
            plan.bytesNoAhead = r.noAhead;
            plan.ahead = nParts > 1. ? r.ahead : 0.;
            plan.snapshot = snapWordFactor * bGpuCell * r.maxCells;
        }
        else
        {
            std::vector<double> whole( 2*NO_DIM, 0. );
            for (int d=0; d<NO_DIM; ++d) whole[2*d+1] = double( u.gridSize[d] );
            auto const partitions = stdTuneSplit( dens, origin, dxGrid, whole, partsP );
            auto boxCount = [&](std::vector<double> const &b, bool const padded) {
                double lo[NO_DIM], hi[NO_DIM];
                for (int d=0; d<NO_DIM; ++d)
                {
                    lo[d] = origin[d] + b[2*d]*dxGrid[d]   - ( padded ? double( u.paddingLength[2*d] ) : 0. );
                    hi[d] = origin[d] + b[2*d+1]*dxGrid[d] + ( padded ? double( u.paddingLength[2*d+1] ) : 0. );
                }
                return dens.count( lo, hi ); };
            auto boxCells = [&](std::vector<double> const &b) {
                double c = 1.;
                for (int d=0; d<NO_DIM; ++d) c *= std::max( 1., b[2*d+1] - b[2*d] );
                return c; };
            for (size_t ip=0; ip<partitions.size(); ++ip)
            {
                std::vector<double> const &pr = partitions[ip];
                size_t pGrid[NO_DIM];
                for (int d=0; d<NO_DIM; ++d)
                {
                    pGrid[d] = 1;
                    pgOptions.gridSize[d] = size_t( std::max( 1LL, (long long)std::llround( pr[2*d+1] - pr[2*d] ) ) );
                }
                if ( threads > 1 )
                    parallelGrid( size_t(threads), pgOptions, pGrid );   // DTFE_parallel's thread grid (2^a 3^b threads)
                size_t nThreads = 1;
                for (int d=0; d<NO_DIM; ++d) nThreads *= pGrid[d];
                std::vector< std::vector<double> > const boxes = nThreads > 1
                    ? stdTuneSplit( dens, origin, dxGrid, pr, pGrid ) : std::vector< std::vector<double> >( 1, pr );
                std::vector< std::array<double,3> > counted;
                std::vector<double> perThread;
                for (std::vector<double> const &b : boxes)
                {
                    counted.push_back( std::array<double,3>{ { boxCount( b, false ), boxCount( b, true ), boxCells( b ) } } );
                    perThread.push_back( counted.back()[1] );
                }
                PartBytes const r = partitionBytes( counted, boxCount( pr, true ), boxCells( pr ) );
                // the run's largest thread sub-grid, whichever partition holds it (one snapshot is held at a time)
                plan.snapshot = std::max( plan.snapshot, snapWordFactor * bGpuCell * r.maxCells );
                if ( nParts > 1. )
                    plan.ahead = std::max( plan.ahead, r.ahead );
                plan.bytesNoAhead = std::max( plan.bytesNoAhead, r.noAhead );
                if ( r.bytes > plan.bytes )
                {
                    plan.bytes = r.bytes;
                    plan.vertices = r.vertices;
                    plan.part = int(ip);
                    plan.perThread = perThread;
                }
            }
        }
        return plans.emplace( key, plan ).first->second;
    }

    /* The run's peak with 'loop' bytes (the busiest partition, its GPU snapshot and look-ahead) in the partition loop of a plan of
       'parts' partitions. The loop holds the particles and its own grids (bCellDeposit; DTFE.cpp swaps the derived fields for the
       gradient while it runs): 'fixedLoop'; the derived grids are allocated after it ('fixed'), when the partitions are gone and, in
       a partitioned run, the particles too (std_partitioned_run.cc). The larger phase is the peak; the allocators hand the freed
       partitions back (main.cpp dtfeAllocatorReexec, dtfeReleaseTessellationMemory, dtfe_metal_host.cc releaseBuffer). */
    double stdPeak(double loop, size_t const parts) const
    {
        return std::max( fixedLoop + loop, fixed - ( parts > 1 ? partBytes * N : 0. ) );
    }

    /* '--run-mode balanced': a chosen team of T threads whose busiest partition leaves no room for the look-ahead (reportSplit would
       switch it off) gives way to the largest smaller team that has room (DTFE_parallel's teams, 2^a 3^b threads, down to 2): it
       pads fewer sub-boxes, so it holds fewer padded vertices in the busiest partition and in the next one's look-ahead build; its
       larger GPU snapshot and the parallel insert's copies count against it (planOf). Not one tessellation (chooseSplit took it
       already). Only where the tuner chose the threads, under balanced. 'tried': a smaller team was looked for. Returns the team, or T. */
    int lookAheadTeam(int const P, int const T, bool &tried) const
    {
        tried = false;
        auto const aheadFits = [&](int const t) {
            StdPlan const &pl = planOf( P, t );
            double const ahead = ( lookAheadPossible and pl.parts > 1 ) ? pl.ahead : 0.;
            return stdPeak( pl.bytes + pl.snapshot + ahead, pl.parts ) <= budget; };
        if ( not ( autoConc and lookAheadPossible and not oneTessellation
                   and runModeState().mode == RunMode::Balanced )
             or aheadFits( T ) )
            return T;
        if ( stdTeamSize( T ) <= 2 )   // no smaller team of 2 or more threads exists: nothing to try
            return T;
        tried = true;
        for (int t = stdTeamSize( T ) - 1; t >= 2; --t)
            if ( stdTeamSize( t ) == t and aheadFits( t ) )
                return t;
        return T;
    }

    AutoTuneReport chooseSplit() const;
    AutoTuneReport reportSplit(int const bestP, int const chosenT, bool const nothingFits) const;
};

/* Searches the splits for the coarsest one that fits (or, when none fits, the least demanding). Its threads: with the GPU deposit
   one tessellation per partition where that fits (more than one partition), else the most threads that fit; without it the most
   threads, the split then the coarsest that holds them. One tessellation per partition inserts on TBB and walks its tets on every
   core (averaged_interpolation_1.cc) and needs less memory than a team of the same split, whose sub-domains are each padded. The
   CPU deposit keeps the most threads. Chosen on Exact GPU runs of TNG100-3, TNG50-3 z=0 and a 256^3 Zel'dovich box at 512^3. */
inline AutoTuneReport AutoTuneStdPlanner::chooseSplit() const
{
    int const pLo = autoPart ? 1 : int(userSplit[0]);
    int const pHi = autoPart ? 6 : int(userSplit[0]);
    // the threads a partition may use: every core, or the user's --max-concurrent (0 = all threads)
    int const userT = u.maxConcurrent > 0 ? std::min( u.maxConcurrent, cores ) : cores;
    int const tMax = autoConc ? cores : userT;
    /* Only a split that FITS competes: a T=1 split that does not fit must not win the 'T==bestT and P<bestP' tie, or
       whenever one thread is all any split allows the COARSEST split, the one needing the most memory, would be
       returned. A T>1 split fits by construction, and among fitting T=1 splits the coarsest wins; when no split fits
       at all, the one predicted to need the least memory is taken and a warning says so. A user's --max-concurrent is
       the thread count a split is checked at. */
    int bestP = pHi, bestT = 0;
    int leastP = pHi;                      // the split with the smallest predicted peak, for the nothing-fits case
    int const tLeast = autoConc ? 1 : tMax;
    // a plan's bytes: the busiest partition without a look-ahead and the GPU snapshot (need), at the run's peak
    // (stdPeak: the loop or the post-processing, whichever holds more); the look-ahead is added once the split is
    // chosen, as PS-DTFE's, and switched off when it does not fit (below)
    auto need = [&](int P, int T) { StdPlan const &pl = planOf( P, T ); return pl.bytesNoAhead + pl.snapshot; };
    auto peak = [&](int P, int T) { return stdPeak( need( P, T ), planOf( P, T ).parts ); };
    bool const lonePreferred = autoConc and gpuDeposit;   // one tessellation per partition first (above)
    for (int P=pLo; P<=pHi; ++P)
    {
        if ( need(P,tLeast) < need(leastP,tLeast) ) leastP = P;
        if ( lonePreferred and planOf( P, 1 ).parts > 1 and peak( P, 1 ) <= budget )
        {
            bestP = P;
            bestT = 1;
            break;      // one tessellation per partition at the coarsest split that holds it: done
        }
        int T = tMax;
        while ( autoConc and T>1 and peak( P, T ) > budget )
            --T;
        if ( peak( P, T ) > budget )
            continue;   // not even one thread fits at this split
        if ( lonePreferred or T > bestT or (T==bestT and P<bestP) )
        {
            bestP = P;
            bestT = T;
        }
        if ( lonePreferred or T == tMax )
            break;      // the coarsest split that fits (the GPU deposit), or the full thread count fits: done
    }
    bool const nothingFits = ( bestT == 0 );
    if ( nothingFits )
    {
        bestP = leastP;
        bestT = tLeast;
    }
    return reportSplit( bestP, bestT, nothingFits );
}

// Fills the report of the chosen split, keeps the look-ahead with a smaller team under '--run-mode balanced'
// (lookAheadTeam) or turns off a look-ahead that does not fit, and states the choice.
inline AutoTuneReport AutoTuneStdPlanner::reportSplit(int const bestP, int const chosenT, bool const nothingFits) const
{
    char buf[512];
    // the team: the chosen one, or under balanced a smaller one that keeps the look-ahead
    bool teamTried = false;
    int const bestT = nothingFits ? chosenT : lookAheadTeam( bestP, chosenT, teamTried );
    /* The report: the same line PS prints, so the
       launcher's 'Check memory', run_dtfe.sh AUTO_TUNE_REPORT=1 and the SUMMARY's predicted-vs-measured line work for
       standard runs too. mc = the threads of a partition (its concurrent sub-triangulations); the partitions run one
       after the other. perPartition = the busiest partition's bytes on top of 'fixed'. */
    StdPlan const &chosen = planOf( bestP, bestT );
    report.valid = true;
    report.partition = bestP;
    report.maxConcurrent = bestT;
    report.fixed = fixed;
    report.budget = budget;
    report.pts = ptsBytes;
    report.streamsBudgeted = nPts > 0 ? 1. : 0.;
    report.perPartition = chosen.bytes;
    report.gpuSnapshot = chosen.snapshot;
    // the look-ahead (DTFE.cpp): with the GPU deposit and more than one partition, the next partition's
    // particle copy and tessellations on top of the peak, its builds item by item (StdPlan::ahead):
    // the gate below, the report, the prediction and the run (stdLookAheadCounted) read this one figure. A plan that fits without it but not with it runs without it
    // (stdLookAheadOffBudget, as PS-DTFE's psLookAheadOffBudget), and the prediction is then the plan's alone.
    report.lookAhead = ( lookAheadPossible and chosen.parts > 1 ) ? chosen.ahead : 0.;
    u.stdLookAheadCounted = report.lookAhead;   // the loop checks the free memory against it (DTFE.cpp)
    report.fixedLoop = fixedLoop;
    if ( bestT < chosenT )   // (balanced: the chosen team had no room for it, this one has: lookAheadTeam)
    {
        StdPlan const &full = planOf( bestP, chosenT );
        message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " look-ahead kept for this run with "
                << bestT << " threads per partition instead of " << chosenT << " (--run-mode balanced): the busiest "
                << "partition (~" << chosen.bytes/1.e9 << " GB) plus the next one's look-ahead build (~"
                << report.lookAhead/1.e9 << " GB) fit the " << (budget - fixedLoop - chosen.snapshot)/1.e9
                << " GB the memory budget leaves beside the partition loop's grids and particles, where with "
                << chosenT << " threads they would not (~" << full.bytes/1.e9 << " GB plus ~"
                << full.ahead/1.e9 << " GB).\n" << MESSAGE::Flush;
    }
    if ( report.lookAhead > 0. and stdPeak( chosen.bytes + chosen.snapshot + report.lookAhead, chosen.parts ) > budget )
    {
        message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " look-ahead off for this run: the busiest "
                << "partition (~" << chosen.bytes/1.e9 << " GB) plus the next one's look-ahead build (~"
                << report.lookAhead/1.e9 << " GB) do not fit the " << (budget - fixedLoop - chosen.snapshot)/1.e9
                << " GB the memory budget leaves beside the partition loop's grids and particles; partitions are "
                << "prepared in turn" << ( teamTried ? " (--run-mode balanced: no smaller team of 2 or more threads "
                                                       "has room for it either)" : "" ) << ".\n" << MESSAGE::Flush;
        u.stdLookAheadOffBudget = true;
        report.lookAhead = 0.;
        // ... and one tessellation per partition is booked by its phases (stdLonePhases)
        report.perPartition = chosen.bytesNoAhead;
    }
    double const loopPeak = stdPeak( report.perPartition + chosen.snapshot + report.lookAhead, chosen.parts );
    // the start-up peak under --scratch-dir (scratchZeroFill): no split lowers it, so the loop still chooses the plan, and this
    // lifts the prediction and the over-budget check where it is the run's peak. Only a run that sizes its full grids before the
    // loop has it: a partitioned one, or one domain on a team (one tessellation of the whole box sizes none ahead: dtfeParallelSingle)
    bool const gridsSizedAhead = chosen.parts > 1 or ( bestT > 1 and not oneTessellation and not u.redshiftConeOn
                                                       and not u.userDefinedSampling );
    double const startUp = gridsSizedAhead ? scratchZeroFill : 0.;
    report.reservePeak = startUp;
    double const predicted = std::max( loopPeak, startUp );
    report.predicted = predicted;
    if ( startUp > loopPeak )
        message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " the predicted peak is the start-up one: "
                << "the particles, the grids left on the heap and the zero fill of the largest scratch-backed grid (~"
                << startUp/1.e9
                << " GB in all) exceed the partition loop's ~" << loopPeak/1.e9 << " GB; the plan is the loop's"
                << ( startUp > budget ? " -- and no split lowers it: it is over the budget whatever the plan "
                                                "(a smaller grid, fewer fields, a larger DTFE_MEM_BUDGET_GB, or Linux, "
                                                "where the zero fill goes to the file)" : "" )
                << ".\n" << MESSAGE::Flush;
    if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )
    {
        std::fprintf( stderr, "AUTO-TUNE-DEBUG standard: %s (count grid %.3f s, %zu plans %.3f s); busiest partition "
                      "%d of %zu: %.0f padded vertices, %.3f GB on top of %.3f GB; per thread:",
                      dens.ok ? "counted from the particles" : "uniform", countSec, plans.size(),
                      std::chrono::duration<double>( std::chrono::steady_clock::now() - tuneStart ).count() - countSec,
                      chosen.part + 1, chosen.parts, chosen.vertices, report.perPartition/1.e9, fixed/1.e9 );
        for (double v : chosen.perThread) std::fprintf( stderr, " %.0f", v );
        std::fprintf( stderr, "\n" );
    }

    // single domain with all threads fits: keep the defaults (and stay silent)
    if ( bestP==1 and bestT==cores )
        return report;

    if ( autoPart )
    {
        u.partition.assign( NO_DIM, size_t(bestP) );
        u.partitionOn = (bestP > 1);
    }
    if ( autoConc and bestT < cores )
        u.maxConcurrent = bestT;

    std::vector<size_t> splitShown = u.partition;
    if ( splitShown.size() != size_t(NO_DIM) ) splitShown.assign( NO_DIM, size_t(bestP) );
    char vertNote[160] = "";
    if ( dens.ok )
        snprintf( vertNote, sizeof(vertNote), "; the busiest partition triangulates ~%.3g M padded vertices at once",
                  chosen.vertices/1.e6 );
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %s, %d thread(s) "
              "(predicted peak ~%.3g GB%s%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              MESSAGE::printElements( splitShown, " " ).c_str(),
              bestT, predicted/1.e9, metalActive ? ", GPU interpolation" : "", vertNote );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
            << autoTuneOverrideNote( autoPart, autoConc ) << "\n           memory budget: "
            << budgetWhy << "\n" << MESSAGE::Flush;
    if ( nothingFits )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: no split fits the memory budget: even the least demanding one (--partition "
                << MESSAGE::printElements( splitShown, " " )
                << " with " << bestT << " thread(s)) is predicted to need "
                << predicted/1.e9 << " GB against a " << budget/1.e9 << " GB budget. The run may swap or be "
                << "killed: free memory, lower --padding (the dummy-point test reports whether it still suffices) "
                << "or --grid, or pass a finer --partition";
        // ... or the per-tetrahedron gradient, where it books less than the face pass
        if ( faceWayOutCell > 0. )
            warning << ", or leave out the face gradient's sums (" << u.faceGradientSource << "; " << faceWayOutCell
                    << " B per cell of every sub-domain) with '--per-tet-gradient' or DTFE_FACE_GRADIENT=0";
        warning << ".\n" << MESSAGE::EndWarning;
    }
    return report;
}

}  // namespace
#endif
