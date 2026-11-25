/* Phase-space DTFE grid interpolation: the triangulation is built in Lagrangian space and Eulerian simplices can overlap,
   so every cell's contribution goes to every grid point it contains. The parts in ps_interpolation/ are #included below
   in order and compiled as part of this one object. */

#ifdef PHASE_SPACE

/* 2D phase space (DIM=2 build: PS-DTFE-2d). The Gadget-HDF5 reader reads N*NO_DIM values, so a 2D build must be fed 2D data
 * (tests/generate_ps_test_data.py --dim 2). The 2D path rests on the exact point-in-TRIANGLE test (ps_exact_inside.h), CGAL's
 * 2D face circulator for the degrees and ps_clip2d.h for the exact deposit; CGAL's 2D binary triangulation I/O does not read
 * back (tessellation_cache.h uses ASCII). 3D only: the GPU deposit, point evaluation (--sample-points, --serve),
 * --grid-window and --ps-caustic-cusps. In 2D --ps-caustics gives parity + collapse multiplicity k = 0..2. */

#include "triangulation_common.h"

#include <algorithm> // sort/unique for the deduplicated nearest-N cusp stencil
#include <atomic>    // the slab deposit's dynamic slab pickup
#include <chrono>    // timing of the GPU-deferred CPU deposits
#include <climits>   // LONG_MAX (markFlipped's default slab bounds)
#include <cstring>   // memcpy in the in-place cost-sort permutation
#include <exception> // the deposit threads hand their errors back
#include <functional> // the dynamic team's recruit hook (PSBlockClaim)
#include <iomanip>   // the phase-timing line
#include <mutex>     // the stream statistics over cell ranges (psStreamStats)
#include <new>       // std::bad_alloc: the face walk beside the GPU deposit falls back when its words do not fit
#include <sstream>   // the phase-timing line
#include <system_error> // a thread the system refuses (PSEarlyFaceWalk)
#include <thread>    // the CPU deposit's threads (psDepositThreads)
#include <type_traits>
#include <unordered_map>   // DTFE_DEBUG_DEGREES self-check
#ifdef OPEN_MP
#include <omp.h>
#endif

// gpu_host.h declares the GPU deposit API (the CPU deposit is the fallback) and defines psWindowHull, which the CPU loop uses
// in every build, so it is included unconditionally.
#include "gpu_host.h"
#include "ps_clip2d.h"      // the 2D exact deposit's clipper (DIM=2 builds)
#include "../thread_count.h" // coresPerTeam: the CPU deposit's threads (psDepositThreads)

// The deposit passes meant for the GPU, asked or picked, that ran on the CPU (ps_point_eval.h psGpuFallbacksTaken).
static std::atomic<int> psGpuFallbacks{ 0 };
int psGpuFallbacksTaken(bool reset)
{
    return reset ? psGpuFallbacks.exchange( 0 ) : psGpuFallbacks.load();
}

// The face walks beside the GPU deposit (the default, PSEarlyFaceWalk; ps_point_eval.h psFaceWalksEarlyTaken):
// the passes whose walk was kept, the walks thrown away after a failed device call, the kept walks' summed wall time
// and the part of it their passes still waited for after the device call had returned (microseconds).
static std::atomic<long long> psFaceEarlyKept{ 0 }, psFaceEarlyDiscarded{ 0 };
static std::atomic<long long> psFaceEarlyWalkUs{ 0 }, psFaceEarlyWaitUs{ 0 };
void psFaceWalksEarlyTaken(long long &kept, long long &discarded, double &walkSec, double &waitSec, bool reset)
{
    kept      = reset ? psFaceEarlyKept.exchange( 0 )      : psFaceEarlyKept.load();
    discarded = reset ? psFaceEarlyDiscarded.exchange( 0 ) : psFaceEarlyDiscarded.load();
    walkSec = double( reset ? psFaceEarlyWalkUs.exchange( 0 ) : psFaceEarlyWalkUs.load() ) * 1e-6;
    waitSec = double( reset ? psFaceEarlyWaitUs.exchange( 0 ) : psFaceEarlyWaitUs.load() ) * 1e-6;
}

// the TBB allocator's kept tessellation blocks back to the system (triangulation.cpp)
void dtfeReleaseTessellationMemory();

// Failed GPU deposit attempts of this run (run_progress.h: the progress sidecar's gpu_retries), counted by
// the GPU host (ps_metal_host.cc).
std::atomic<int> psGpuRetries{ 0 };
// ... and the same attempts by cause (the sidecar's gpu_retries_by_cause and the run's closing
// 'GPU deposit retries:' line): [0] innocent victim, [1] timeout (a watchdog kill), [2] out of memory, [3] any
// other cause; they sum to psGpuRetries (each host bumps both for every failed buffer / kernel).
std::atomic<int> psGpuRetriesByCause[4] = { { 0 }, { 0 }, { 0 }, { 0 } };

// Wall time of the phases inside 'field interpolation' (extraction walk, cost sort, GPU call, GPU-deferred tets,
// normalization), in microseconds over every pass of the run; each pass prints its own split and the run's totals so far.
namespace {
enum PSPhase { PS_PH_SETUP, PS_PH_EXTRACT, PS_PH_SORT, PS_PH_GPU, PS_PH_COPY, PS_PH_DEFERRED, PS_PH_CPU, PS_PH_NORM, PS_PH_N };
char const* const psPhaseName[PS_PH_N] = { "setup", "extraction walk", "cost sort", "GPU deposit", "copy-back",
                                           "deferred tets on the CPU", "CPU deposit", "normalization" };
std::atomic<long long> psPhaseTotalUs[PS_PH_N];
std::atomic<long long> psPhaseTotalDeferred{ 0 };
std::atomic<int>       psPhasePasses{ 0 };
}

// The finite cells (3D) / faces (2D) around a vertex, appended to 'out'.
template <class Tr, class V>
static inline void psFiniteIncidentCells(Tr const &dt, V const &vh, std::vector<Cell_handle> &out)
{
#if NO_DIM==3
    dt.finite_incident_cells( vh, std::back_inserter(out) );
#else
    auto fc = dt.incident_faces( vh ), done = fc;
    if ( fc == nullptr ) return;
    do { if ( not dt.is_infinite( fc ) ) out.push_back( fc ); } while ( ++fc != done );
#endif
}

// --grid-window pre-filter: does the raw cell span [lo, hi) of a tetrahedron (the periodic image it lies in) touch the
// sub-box [sub0, sub0+subN), which may itself wrap the axis? A tet touching no axis deposits nothing and is skipped.
static inline bool psSpanTouches(long lo, long hi, long n, bool periodic, long sub0, long subN)
{
    if (hi <= lo) return false;
    if (!periodic)
    {
        lo = std::max(lo, 0L); hi = std::min(hi, n);
        if (hi <= lo) return false;
    }
    else if (hi - lo >= n) return true;
    long A[2][2], B[2][2];
    auto pieces = [n](long a, long b, long (&out)[2][2]) -> int   // [a,b) as residues mod n: one or two runs
    {
        long const len = b - a;
        a = ((a % n) + n) % n; b = a + len;
        if (b <= n) { out[0][0] = a; out[0][1] = b; return 1; }
        out[0][0] = a; out[0][1] = n; out[1][0] = 0; out[1][1] = b - n; return 2;
    };
    int const na = pieces(lo, hi, A), nb = pieces(sub0, sub0 + subN, B);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j)
            if (A[i][0] < B[j][1] && B[j][0] < A[i][1]) return true;
    return false;
}
// a cell index that marks "counted for the tet's normalization, outside the window: not deposited"
static size_t const PS_NOT_IN_WINDOW = size_t(-1);
// the GPU per-tet record (psFillTetRecord), and the fixed-point mass unit and the shares' guard (psMassFixedScale,
// psFixedShareOk) that the CPU deposit uses in every build, GPU or not
#include "ps_deposit_params.h"
#include "face_sweep.h"   // --face-gradient: the face sections (psDepositFaces)

#include "ps_interpolation/fixed_point.cc"   // the CPU deposit's fixed-point sums, the face sums' deposit

// --ps-caustics: per-tetrahedron caustic stratification bits. Included UNCONDITIONALLY -- the
// classification is part of the CPU deposit and exists in 2D as well, so it must not sit behind
// a GPU or 3D guard (doing so breaks the CPU-only and the 2D builds).
#include "ps_caustic_class.h"

// exact, tie-consistent sample-in-tetrahedron test (shared with the point evaluation)
#include "ps_exact_inside.h"

// psDepositThreadBytes: the memory the deposit threads' private copies may take (budgeted there too)
#include "../auto_tune.h"

// DTFE_MASS_AUDIT=1: the deposit's mass bookkeeping (opt-in diagnostic)
#include "ps_mass_audit.h"

#include "../env_knobs.h"

#if NO_DIM==3
// --exact: analytic tetrahedron-cell intersection moments (Powell & Abel 2015).
// Vendored C library; r3d.h carries its own extern "C" guard. 3D only.
#include "../../third_party/r3d/r3d.h"
#include "exact_sweep.h"    // the per-slab/column clipping of the exact deposit
#endif

#include "ps_interpolation/tet_tests.cc"      // TetBoxSAT, psDepositThreads, the beyond-grid count
#include "ps_interpolation/pass_context.cc"   // the cell-loop macros, the pass contexts, the per-tetrahedron tests
#include "ps_interpolation/fx_state.cc"       // PSFxState, the bound walk, the face walk
#include "ps_interpolation/pass_setup.cc"     // the pass's setup: sub-grid box to the CPU fixed-point units
#include "ps_interpolation/cpu_threads.cc"    // the CPU deposit's thread mode, slab binning, thread start
#include "ps_interpolation/pass_close.cc"     // the pass's close: fixed point to Real to the phase report
#include "ps_interpolation/gpu_deposit.cc"    // the GPU extraction and deposit (psGpuExtractAndDeposit)

/* Interpolates fields onto a regular grid via PS-DTFE: scatters each finite cell's contribution to the grid points it
   overlaps. nSub = sub-samples per axis for volume-averaged fields (1 = unaveraged). mayClearDT = the last use of an
   internally-owned dt: clear it right after the deposit's last read. */
void interpolateGrid_phaseSpace(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities,
                                Field &field,
                                int nSub,
                                bool mayClearDT)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    size_t const *nGrid = &(userOptions.gridSize[0]);
    Box boxCoordinates = userOptions.region;

    // the phase split of this pass (psPhaseTotalUs above): lap(k) books the time since the last lap to phase k
    double phaseSec[PS_PH_N] = { 0. };
    auto phaseMark = std::chrono::steady_clock::now();
    auto lap = [&](int k)
    {
        auto const now = std::chrono::steady_clock::now();
        phaseSec[k] += std::chrono::duration<double>( now - phaseMark ).count();
        phaseMark = now;
    };

    // Averaged fields sample each grid cell on an nSub^NO_DIM sub-grid (nSub==1 = cell centre only).
    if ( nSub < 1 ) nSub = 1;
    size_t nSamplesPerCell = 1;
    for (int d = 0; d < NO_DIM; ++d) nSamplesPerCell *= size_t(nSub);

    Real dx[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
        dx[d] = (boxCoordinates[2*d+1] - boxCoordinates[2*d]) / nGrid[d];
    Real cellVolume = Real(1.);
    for (int d = 0; d < NO_DIM; ++d) cellVolume *= dx[d];

    size_t totalGrid = 1;
    for (int d = 0; d < NO_DIM; ++d) totalGrid *= nGrid[d];

    // Every pass below walks the finite cells DIRECTLY via the CGAL iterator (all consuming loops are serial and the
    // iteration order is deterministic for a fixed triangulation), so no cellHandles vector is materialized.
    size_t noTotalCells = 0;
#if NO_DIM==2
    noTotalCells = dt.number_of_faces();
#elif NO_DIM==3
    noTotalCells = dt.number_of_finite_cells();
#endif

    // psUseSubgrid: store only the Eulerian bbox of cells this partition touches (the same filter as the scatter). The box may
    // WRAP a periodic axis: subOrigin+subDims can exceed nGrid and the local index is then (g - subOrigin + nGrid) % nGrid.
    // Every consumer must apply that same wrap (the CPU scatter loops, the GPU kernels, Quantities::addFromSubgrid); it is a
    // no-op for an unwrapped box.
    size_t subOrigin[NO_DIM], subDims[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d) { subOrigin[d] = 0; subDims[d] = nGrid[d]; }
    // cells surviving the dummy/ownership/degeneracy filters, counted by the subgrid pass below, to right-size the Metal tet
    // arrays (an upper bound when the pass does not run).
    size_t psKeptCells = noTotalCells;
    bool const windowMode = userOptions.psWindowOn;   // --grid-window: the sub-box is a window of the full grid
    PSPassCtx const passCtx{ dt, userOptions, quantities, field, message, boxCoordinates, nGrid, dx, nSub, noTotalCells,
                             windowMode, subOrigin, subDims };
    psSubgridBox( passCtx, psKeptCells );
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= subDims[d];
    if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // the sub-grid the planner's span estimate books
    {
        std::fprintf( stderr, "AUTO-TUNE-DEBUG sub-grid:" );
        for (int d = 0; d < NO_DIM; ++d) std::fprintf( stderr, " %zu", subDims[d] );
        std::fprintf( stderr, " = %zu cells\n", subTotal );
    }
    totalGrid = subTotal;   // all allocations and loops below use the sub-grid size

    // The centroid-fallback cell of a tetrahedron (psCentroidCell says what and why). The closure holds a COPY of the
    // pass context (a struct of references), so a call through it passes the closure's address as the context.
    auto centroidCell = [passCtx](Real const (&ep)[NO_DIM+1][NO_DIM], Real (&centroid)[NO_DIM], size_t &flat) -> bool
    { return psCentroidCell( passCtx, ep, centroid, flat ); };

    // Exact containment of the sub-samples (ps_exact_inside.h): a float tolerance would count a sample near a shared face in
    // BOTH tets depending on each run's periodic-copy rounding (serial and --partition runs would disagree). Every sample is
    // counted by exactly one tetrahedron of each stream; the GPU kernel reaches the same decisions and defers the rest here.
    psExactInside::Frame exactFrame;
    exactFrame.init( userOptions );
    // Sample positions in DOUBLE per axis: samplePosAxis[d][g * nSub + k] = lo + ((g + (k+0.5)/nSub) * L) / n for the WRAPPED
    // cell index g, so every tetrahedron (original or periodic image, any partition) sees the same point; at nSub=1 it is
    // (i+0.5)*L/n bit for bit, the expression the tests hand to --sample-points.
    double gridLenD[NO_DIM];
    std::vector<double> samplePosAxis[NO_DIM];
    psSamplePositions( passCtx, gridLenD, samplePosAxis );


    // --ps-vertex-mass: chart-independent masses. Count, per vertex, the incident cells that DEPOSIT; a tet's mass is the sum of
    // its vertices' weight/degree shares (psVertexShareMass), so the deposited masses add up to sum_v m_v exactly. The degree
    // applies the loop's chart-independent drop filters with ownership skipped (padded vertices see their full neighbourhood,
    // so per-partition degrees match the global ones), counted on THIS triangulation after a reset (both passes share one dt).
    bool const psVertexMass = userOptions.psVertexMass;
    // the full degree pass: every finite cell once, its surviving vertices counted (ps_cell_filter.h: the
    // point evaluation runs the same pass)
    auto fullDegreePass = [&]()
    {
        psVertexDegreePass( dt, userOptions, boxCoordinates );
    };
    // --grid-window on the GPU: only the vertices of the tetrahedra that reach the window need their degree, computed LAZILY
    // from the vertex's own incident cells (same filter, ownership skipped: the same count as the full pass); -1 = not computed.
    // The threaded CPU loop keeps the full pass, run before it whenever the GPU does not deposit (degreesLazy cleared).
    bool degreesLazy = false;
#if defined(PS_GPU) && NO_DIM==3
    degreesLazy = psVertexMass && windowMode && userOptions.psUseMetal;
#endif
    if ( degreesLazy )
    {
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            vit->info().psDegree() = -1;
    }
    else if ( psVertexMass )
        fullDegreePass();
    // (the lazy degrees themselves: lazyDegrees in psGpuExtractAndDeposit)

    // The MASS grid is an exact 64-bit fixed-point sum on the GPU (metal/ps_deposit.metal depositMass) and the CPU alike: a
    // float '+=' drops shares below half an ulp of a busy cell's sum and depends on the thread count. After a GPU deposit the
    // host keeps its words (PSGpuGrids::massFx) and released/deferred tets add rint(share * scale) into them (psFxAdd); a CPU
    // deposit starts from zero with psMassFixedScale's unit. Integer adds commute: the grid is independent of thread count,
    // slab split and atomics' order. Single-precision build only (a 2^-62 unit is coarser than a double sum in a light cell).
    bool const massFxWanted = std::is_same<Real, float>::value && field.density;
    PSFxWords massFx;                    // totalGrid fixed-point words while massFxOn, else EMPTY
    float &massFxScale = massFx.scale;   // their unit: 1/massFxScale mass (a power of two)
    bool massFxOn = false;

    // The MOMENT grids (PSFxGrid) are exact 64-bit fixed-point sums too: shares added as rint(v*w * scale). A GPU deposit returns
    // the words of every grid it got a unit for (fxUnits); a CPU deposit takes its units from the same bounds, summed in one
    // filter pass in fixed blocks of cells (thread-count independent, fxBoundWalk); all threads add into ONE set of words,
    // converted once after the loop (fx.toReal). A grid's Real accumulator stays EMPTY while its words live. Single-precision
    // build only, like the mass; a GPU deposit sums its grids and exact weight totals in fixed point in both builds.
    bool const fxMomWanted = std::is_same<Real, float>::value;
    PSFxWords fxg[PS_FX_CPU_N];    // grid g's words (EMPTY = its Real accumulator is in use) and their unit

    // DTFE_MASS_AUDIT=1 (ps_mass_audit.h): what this pass should deposit, counted before it in one extra filter pass: (a) owned
    // vertices, (c) degree 0, (b) tets passing the filter with ownership and their mass, (d) owned tets the drop filters reject.
    bool const massAudit = PSMassAudit::on();
    struct MassAudit
    {
        double owned = 0., deg0 = 0., tets = 0., dropped = 0., after = 0.;
        size_t nOwned = 0, nDeg0 = 0, nTets = 0, nTetsDeg0 = 0, nDropped = 0, nAfter = 0;
        double gpuHanded = -1., gpuDeferred = -1., gpuFixed = -1., gpuFloat = -1., released = 0., releasedGain = 0.;
        double cpuTets = 0., cpuShares = 0., cpuGain = -1.;
        size_t nCpuTets = 0;
    } audit;
    auto gridSum = [&]() -> double   // the mass grid in mass units: the fixed-point words while they live (exact)
    {
        if ( massFxOn )
        {
            int64_t q = 0;
            for (int64_t const w : massFx.w) q += w;
            return psMassFixedMass( q, massFxScale );
        }
        double s = 0.;
        for (Real const v : quantities->density) s += double(v);
        return s;
    };
    psMassAuditCount( passCtx, massAudit, psVertexMass, degreesLazy, fullDegreePass, audit );

    // Allocate the requested output fields, zeroed since the cell loop accumulates into them. The density comes later: from the
    // GPU's mass words at the copy, or (CPU) just before the loop, as the fixed-point mass grid that stands in for it.
    if ( field.density )
        std::vector<Real>().swap( quantities->density );
    // The velocity grid only when the velocity is asked: the dispersion carries its OWN mass-weighted mean
    // (disp_velocity/disp_weight below), so a dispersion-only run allocates, deposits and GPU-round-trips no
    // 12 B/cell velocity grid.
    bool const haveVel = field.velocity;
    // the dispersion always needs the per-sample velocity VALUES (for sum m v_i v_j and its
    // own mean), even when the velocity GRID above is gated away -- keep the two distinct
    bool const needVelValues = haveVel || field.velocity_dispersion;
    psAllocMoments( passCtx, haveVel, fxMomWanted, totalGrid );

    // Per-grid-point count of overlapping streams (diagnostic; stream_count). Sampled deposit: +1 per in-simplex sample, divided by
    // nSub^NO_DIM at the end; a tet containing NO sample deposits through the centroid fallback but adds nothing here (it would
    // bias '.streams' upward in collapsed cells). Exact deposit: the TETRAHEDRA TOUCHING the cell, a geometric multiplicity that is
    // NOT a stream count, so it is exported as '.tetTouch' and the physical multiplicity is accumulated in exactMult below.
    std::vector<int> streamCount(totalGrid, 0);
    // --ps-halo-release: a released tet is never sampled, so it adds its volume fraction V_tet/V_cell instead, its exact share
    // of the cell-mean multiplicity (as the exact deposit does). Allocated only when tets can be released.
    std::vector<Real> releasedMult;
    if ( userOptions.psHaloRelease > Real(0.) && !userOptions.psExactDeposit )
        releasedMult.assign(totalGrid, Real(0.));
    // 1 where a tetrahedron deposited MASS here without any of its samples lying here (the centroid fallback), invisible in the
    // sample count; exported as '.hidden_streams' to keep single-stream masks clean (quantities.h). The GPU kernel reports it in
    // bit 31 of its stream counter.
    std::vector<unsigned char> subSampleMassFlag(totalGrid, 0);
    // The EXACT companion ('.hidden_streams' bit 1): 1 where a negatively oriented (folded) tet overlaps the cell with positive
    // volume. In a periodic box the map has degree 1, so the stream count is 1 + 2 #negative and a point is multi-stream EXACTLY
    // where a flipped tet covers it. Marked per tet in both deposit paths (markFlipped), OR-merged across partitions;
    // DTFE() keeps it only where '.streams' reads single-stream (finalizeHiddenStreams). Also '--ps-caustics' parity bit 1.
    std::vector<unsigned char> flippedFlag(totalGrid, 0);
    // psMarkFlipped (xLo/xHi and 'shared' as there); the closure copies the pass context, as centroidCell's does
    auto markFlipped = [passCtx](PSCellGeometry const &geo, std::vector<unsigned char> &flags,
                                 long const xLo = 0, long const xHi = LONG_MAX, bool const shared = false)
    { psMarkFlipped( passCtx, geo, flags, xLo, xHi, shared ); };
    // --exact: exact cell-mean stream multiplicity = (1/V_cell) * sum_tets V_int, the analytic nSub->infinity limit of the
    // sampled '.streams', so single-stream masking works under either deposit, but ONLY to a tolerance (|s-1| < 1e-3,
    // dtfelib.STREAM_TOL): a float sum, so single-stream cells land on 1.0 +/- eps. Grazing slivers contribute ~0.
    std::vector<Real> exactMult;
    if ( userOptions.psExactDeposit && !fxMomWanted ) exactMult.assign(totalGrid, Real(0.));   // (fxMomWanted: fx.ensureReal)

    // --ps-caustics: per-cell orientation mask (bit0 = a det(Ax)>0 stream overlaps the cell, bit1 = det<0); the sign flips at
    // every fold, so both set = a fold caustic crosses the cell. Exported as small exact ints in Real. Accumulated per
    // deposited SAMPLE; the export ORs the geometric flippedFlag into bit 1, so bit 1 is the true overlap.
    bool const psCaustics = userOptions.psCaustics;
    // PSCausticClass::CausticMask, not unsigned char: the A4 indicator lives in bit 8, so a byte
    // accumulator would drop it here without any compiler diagnostic.
    std::vector<PSCausticClass::CausticMask> orientBits;
    if ( psCaustics )
        orientBits.assign(totalGrid, 0);

    // --ps-halo-release: threshold D on the geometric stream density rho_geo/rho_bar = |det Lag| / |det Ax|. Tets beyond D
    // sit in virialized, mixed regions where the sheet estimate is unconverged (Stuecker et al. 2021): they take the
    // mass-conserving centroid fallback. Classified in double in BOTH deposits (CPU and GPU must release identical tets). 0 = off.
    double const psHaloRelease = double(userOptions.psHaloRelease);
    size_t nReleased = 0;

    // --exact: analytic per-cell intersection moments instead of nSub^3 sampling. It ignores nSub where it clips, but a tet
    // left with no piece takes the sampled path at nSub. '.streams' carries the volume-weighted multiplicity (exactMult): a
    // FLOAT, so the partition merge is NOT bit-exact (the volume shares sum in thread order); the raw integer tet-touch count
    // merges bit-exactly and is exported as '.tetTouch'.
    bool const psExact = userOptions.psExactDeposit;

    // Count of cells dropped because their Eulerian simplex is non-invertible (see the check below).
    size_t nDegenerateInverse = 0;

    // multi-stream velocity/scalar = volume-weighted mean sum(V_s f_s)/sum(V_s): the moments and their normalizer carry
    // EULERIAN-VOLUME shares (the standard-DTFE '_a' convention); density, stream count and caustics keep the mass deposit.
    // psDeferNormalization leaves the moments un-normalized so they sum linearly across partitions (the mean is non-linear).
    bool const deferNorm = userOptions.psDeferNormalization;
    bool const needWeight = field.velocity || field.velocity_gradient || field.velocity_dispersion
#ifdef SCALAR
                            || field.scalar || field.scalar_gradient
#endif
                            ;
    // the moment normalizer sum(V_s): a grid of its own
    std::vector<Real> massWeight;
    if ( needWeight && !fxMomWanted ) massWeight.assign(totalGrid, Real(0.));   // (fxMomWanted: fx.ensureReal)
    // The DISPERSION stays mass-weighted (sigma_ij = <v_i v_j>_m - <v_i>_m<v_j>_m) while velocity/gradient are volume-weighted,
    // so it needs its own mass-weighted mean and normalizer.
    bool const dispNeedsOwnWeight = field.velocity_dispersion;
    std::vector<Real> dispWeight;                      // sum(m_s)
    std::vector< Pvector<Real,noVelComp> > dispVel;    // sum(m_s v_s)
    if ( dispNeedsOwnWeight && !fxMomWanted )   // (fxMomWanted: fx.ensureReal)
    {
        dispWeight.assign(totalGrid, Real(0.));
        dispVel.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    }

    // ---- the moment grids as fixed-point sums (see fxg above) ----
    // which grids this pass accumulates (ps_deposit_params.h PSFxGrid): MOMW is massWeight (the volume shares), DW
    // dispWeight, RM releasedMult
#ifdef SCALAR
    bool const fxScalSel = field.scalar, fxSGradSel = field.scalar_gradient;
#else
    bool const fxScalSel = false, fxSGradSel = false;
#endif
    bool const fxSel[PS_FX_CPU_N] = { haveVel, bool(field.velocity_dispersion), bool(field.velocity_gradient),
                                      needWeight, psExact, dispNeedsOwnWeight, dispNeedsOwnWeight,
                                      fxScalSel, fxSGradSel, !releasedMult.empty() };
    // the moment grids' fixed-point state: allocation, units, bounds and the conversion of grid g (PSFxState)
    PSFxState fx{ fxg, fxSel, userOptions, quantities, field, totalGrid, massWeight, exactMult, dispWeight, releasedMult,
                  dispVel, needVelValues || field.velocity_gradient, fxScalSel, fxSGradSel, cellVolume };
    fx.face = userOptions.faceGradient && psExact && field.velocity_gradient && NO_DIM == 3;
    fx.gridDx = dx;
    for (int d = 0; d < NO_DIM; ++d) quantities->psFaceDx[d] = dx[d];   // (the face gradient's spacing: normalization)

    // Reusable per-tet scratch of the mass-conserving deposit: flat grid index and Eulerian offset of every in-simplex sample.
    std::vector<size_t> insideFlat;
    std::vector<Real>   insideRel;   // insideFlat.size() * NO_DIM, row-major per sample
    // --exact per-tet scratch: 10 order-2 moments and the mass weight per hit cell
    std::vector<double> exMom, exW;

    if ( not userOptions.psSuppressGridStats )
        message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                << " Interpolating fields to grid by iterating over all Delaunay cells ...\n" << MESSAGE::Flush;

    // GPU deposit (--gpu): extract flat per-tet arrays with EXACTLY the CPU loop's filters, dispatch depositFields (mirrors the
    // CPU scatter incl. the partition sub-grid) and copy the grids back. Any failure falls back to the CPU loop with a warning.
    bool metalDeposited = false;
    // Ordinals of the tets the GPU DEFERRED (a sample within float rounding of a face): the CPU loop below deposits these.
    std::vector<uint32_t> deferredOrd;
    PSGridsCtx const gridsCtx{ totalGrid, streamCount, releasedMult, subSampleMassFlag, flippedFlag, exactMult, orientBits,
                               massWeight, dispWeight, dispVel, massFx, massFxScale, massFxOn, insideFlat, insideRel,
                               exMom, exW, fxSel, massFxWanted, fxMomWanted, haveVel, needWeight,
                               dispNeedsOwnWeight, deferNorm, psExact, psCaustics };
#if defined(PS_GPU) && NO_DIM==3
    PSGpuCtx const gpuCtx{ passCtx, gridsCtx, fx, exactFrame, gridLenD, psKeptCells, subTotal, degreesLazy, psVertexMass,
                           massAudit, psHaloRelease, cellVolume, nReleased,
                           nDegenerateInverse, metalDeposited, deferredOrd };
    psGpuExtractAndDeposit( gpuCtx, audit, lap, gridSum );
#endif // PS_GPU
    psDebugDegreesCheck( passCtx, degreesLazy, metalDeposited );
    if ( degreesLazy && not metalDeposited )
    {   // the CPU loop deposits every cell: it needs every degree
        fullDegreePass();
        degreesLazy = false;
    }
#if defined(PS_GPU) && NO_DIM==3
#else
    if ( userOptions.psUseMetal )
    {
        ++psGpuFallbacks;
        static bool warned = false;
        if ( !warned )
        {
            warned = true;
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << userOptions.gpuRequest() << " requested but this binary was built without GPU support (rebuild with METAL=1 on a Mac; or NO_DIM!=3); using the CPU deposit.\n" << MESSAGE::EndWarning;
        }
    }
#endif // PS_GPU

    psCpuFixedUnits( passCtx, gridsCtx, metalDeposited, fx, psVertexMass );
    psDebugFxBoundsCheck( passCtx, fx, metalDeposited, psVertexMass );

    // The per-cell scatter. The CPU deposit runs on nDepThreads threads (capped by the memory of their private accumulators):
    // thread t takes every T-th block of 256 cells into its own copies, summed in thread order (deterministic for a given thread
    // count; DTFE_DEPOSIT_THREADS pins it). Shared by all threads: the single build's fixed-point mass and moment words, and in
    // every build the stream count, flags and caustic mask (relaxed atomics). GPU-deferred cells and --ps-caustic-cusps (CGAL's
    // incident_cells marks visited cells: not re-entrant) stay on one thread; after a GPU deposit only deferredOrd runs here.
    struct DepositThread
    {
        Quantities *q = nullptr;
        std::vector<int> *streamCount = nullptr;
        std::vector<Real> *releasedMult = nullptr, *exactMult = nullptr, *dispWeight = nullptr, *massWeight = nullptr;
        std::vector< Pvector<Real,noVelComp> > *dispVel = nullptr;
        std::vector<PSCausticClass::CausticMask> *orientBits = nullptr;
        std::vector<unsigned char> *subSampleMassFlag = nullptr, *flippedFlag = nullptr;
        // the fixed-point mass grid (massFxOn; else null): ONE grid for every thread, never copied
        int64_t *massFx = nullptr;
        // the private copies of threads > 0 (thread 0 points at the real arrays): the Real accumulators only -- the
        // stream count, the two flags and the caustic mask are the real arrays for every thread (psAddCount)
        Quantities ownQ;
        std::vector<Real> ownReleasedMult, ownExactMult, ownDispWeight, ownMassWeight;
        std::vector< Pvector<Real,noVelComp> > ownDispVel;
        // per-tetrahedron scratch, reused
        std::vector<size_t> insideFlat;
        std::vector<Real>   insideRel;
        std::vector<double> exMom, exW;
        size_t nDegenerateInverse = 0, nReleased = 0;
        // DTFE_MASS_AUDIT=1: the mass of the tets this thread deposits (once per tet) and of the shares it writes
        double auditTets = 0., auditShares = 0.;
        size_t auditNTets = 0;
    };
    // bytes of one private copy of every Real accumulator (grids standing in as shared fixed-point words are EMPTY): the
    // default single build copies NOTHING; only float sums (the double build, a grid without a unit) cost a copy per thread.
    auto vecBytes = [](auto const &v) -> double { return double(v.size()) * double(sizeof(v[0])); };
    double accBytes = vecBytes(releasedMult) + vecBytes(exactMult) + vecBytes(dispWeight) + vecBytes(massWeight)
                    + vecBytes(dispVel) + vecBytes(quantities->density) + vecBytes(quantities->velocity)
                    + vecBytes(quantities->velocity_gradient) + vecBytes(quantities->velocity_dispersion);
#ifdef SCALAR
    accBytes += vecBytes(quantities->scalar) + vecBytes(quantities->scalar_gradient);
#endif
    // (the thread setup below -- slab binning, private copies -- is booked to the deposit it serves)
    lap( metalDeposited ? PS_PH_COPY : PS_PH_SETUP );
    int nDepThreads = 1;
    // SLAB MODE. When private copies would cut the threads below what the cores allow, each thread instead OWNS a slab of local
    // x planes and writes only its cells straight into the real arrays, walking the slab's tets in the loop's order: every
    // cell gets its contributions in the single-threaded order, bit-identical to DTFE_DEPOSIT_THREADS=1 at ANY thread count.
    // A tet straddling k slabs is classified k times, so slabs are several tet extents wide. DTFE_DEPOSIT_SLABS=1 forces the
    // mode (the tests), 0 forbids it.
    bool slabMode = false;
    int nSlabs = 0;
    std::vector<long> slabXLo, slabXHi;                     // each slab's local x range [lo, hi)
    std::vector< std::vector<uint32_t> > slabTetLists;      // per slab: its tetrahedra (ordinals, loop order)
    std::vector<Cell_handle> slabTets;                      // ordinal -> cell
    std::vector<uint8_t> slabFirst;                         // ordinal -> its first slab (per-tet tallies)
    bool dynamicTeam = false, teamGrows = false;   // the dynamic team (PSBlockClaim; psDepositThreadMode says when)
    PSLayoutCtx const layoutCtx{ nDepThreads, slabMode, dynamicTeam, teamGrows, nSlabs, slabXLo, slabXHi, slabTetLists,
                                 slabTets, slabFirst };
    psDepositThreadMode( passCtx, layoutCtx, metalDeposited, accBytes, massAudit, deferredOrd );
    psSlabBinning( passCtx, layoutCtx, metalDeposited, deferredOrd );
    std::vector<DepositThread> depT( static_cast<size_t>(nDepThreads) );
    psDepositThreadsInit( passCtx, gridsCtx, depT, nDepThreads, slabMode );

    // One thread's share of the deposit: the loop over the cells, whose body is the serial loop's (its accumulators and scratch
    // are this thread's under the SAME names; the do/while(false) keeps 'continue' meaning "skip the rest of this tet").
    // Thread th of nT takes every nT-th block of 256 cells, or the deferred ordinals after a GPU deposit. The loop sits INSIDE
    // the lambda: a call per tetrahedron would slow the single-thread path.
    size_t const DEP_BLOCK = 256;
    using PSCellIt = decltype( PS_FINITE_CELLS_BEGIN );
    PSBlockClaim<PSCellIt> *blockClaim = nullptr;   // the dynamic team's block counter, while it deposits
    auto depositLoop = [&](DepositThread &TH, size_t th, size_t nT, bool deferred, int const slab)
    {
        Quantities *quantities = TH.q;
        std::vector<int> &streamCount = *TH.streamCount;
        std::vector<Real> &releasedMult = *TH.releasedMult;
        std::vector<Real> &exactMult = *TH.exactMult;
        std::vector<Real> &dispWeight = *TH.dispWeight;
        std::vector<Real> &massWeight = *TH.massWeight;
        std::vector< Pvector<Real,noVelComp> > &dispVel = *TH.dispVel;
        std::vector<PSCausticClass::CausticMask> &orientBits = *TH.orientBits;
        std::vector<unsigned char> &subSampleMassFlag = *TH.subSampleMassFlag;
        std::vector<unsigned char> &flippedFlag = *TH.flippedFlag;
        std::vector<size_t> &insideFlat = TH.insideFlat;
        std::vector<Real> &insideRel = TH.insideRel;
        std::vector<double> &exMom = TH.exMom;
        std::vector<double> &exW = TH.exW;
        size_t &nDegenerateInverse = TH.nDegenerateInverse;
        size_t &nReleased = TH.nReleased;
        // The per-sample loops' constants as TRUE locals: read through the lambda's captured
        // references they would be reloaded after every push_back/accumulate that might alias them
        // (size_t / Real stores), which makes the wrapped loop slower than the inline one.
        bool const periodicL = userOptions.periodic;
        bool const windowModeL = windowMode;
        int const nSubL = nSub;
        size_t const nSamplesL = nSamplesPerCell;
        Real const cellVolumeL = cellVolume;
        size_t nGridL[NO_DIM], subOriginL[NO_DIM], subDimsL[NO_DIM];
        double gridLenL[NO_DIM];
        double const *samplePosL[NO_DIM];
        Real dxL[NO_DIM], boxL[2*NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
        {
            nGridL[d] = nGrid[d]; subOriginL[d] = subOrigin[d]; subDimsL[d] = subDims[d];
            gridLenL[d] = gridLenD[d]; samplePosL[d] = samplePosAxis[d].data(); dxL[d] = dx[d];
            boxL[2*d] = boxCoordinates[2*d]; boxL[2*d+1] = boxCoordinates[2*d+1];
        }
        PSBeyondGrid beyondGrid;   // (a copy: passing the hoisted locals themselves would escape them)
        beyondGrid.nSub = nSub;
        for (int d = 0; d < NO_DIM; ++d)
        {
            beyondGrid.nGrid[d] = nGrid[d]; beyondGrid.gridLen[d] = gridLenD[d];
            beyondGrid.boxLo[d] = boxCoordinates[2*d]; beyondGrid.dx[d] = dx[d];
        }
        bool const dispNeedsOwnWeightL = dispNeedsOwnWeight;
        bool const haveVelL = haveVel;
        bool const needVelValuesL = needVelValues;
        bool const needWeightL = needWeight;
        bool const psCausticsL = psCaustics;
        bool const psExactL = psExact;
        bool const psVertexMassL = psVertexMass;
        bool const fVGradL = field.velocity_gradient;
        bool const fVDispL = field.velocity_dispersion;
        bool const fSGradL = field.scalar_gradient;
        bool const fDenL = field.density;
        bool const fVelL = field.velocity;
        bool const fScalL = field.scalar;
        bool const auditL = massAudit;
        // the fixed-point mass grid (null: the float density) and how this thread adds to it: atomically when
        // the copy mode's threads share it, plainly when this thread owns its cells (one thread, or a slab); its
        // handle (psFxAdd: the unit, the cap and this thread's carry list)
        int64_t *const massFxL = TH.massFx;
        PSFxSink const massSinkL = massFx.sink( th );
        bool const massFxAtomicL = nT > 1 && slab < 0;
        // the stream count, the two flags and the caustic mask: the same rule (psAddCount, psSetFlag, psOrMask)
        bool const countAtomicL = massFxAtomicL;
        // the moment grids' fixed-point words (fxg; null: the grid's Real accumulator), ONE set for every
        // thread, added exactly as the mass words are (the same atomic rule); each share is the float product the
        // Real grid would have added. DW is not accumulated while the mass words are this pass's (massFxOn): it
        // becomes the density after the loop.
        int64_t *fxWL[PS_FX_CPU_N];
        PSFxSink fxSinkL[PS_FX_CPU_N];
        for (int g = 0; g < PS_FX_CPU_N; ++g) { fxSinkL[g] = fxg[g].sink( th ); fxWL[g] = fxSinkL[g].w; }
        bool const dwOffL = massFxOn;
        auto fxAddL = [&](int const g, size_t const at, Real const v)
        { psFxAdd( fxSinkL[g], at, double(v), g, massFxAtomicL ); };
        // --face-gradient: the face sums' words (null: off), shared by every thread like the moments' (psDepositFaces).
        // Not under a GPU deposit: psFaceWalk then sums every kept tetrahedron's faces, the deferred ones this loop
        // deposits included
        bool const faceHereL = fx.face && !metalDeposited;
        PSFxSink const faceSL = faceHereL ? fx.faceS.sink( th ) : PSFxSink();
        PSFxSink const faceNL = faceHereL ? fx.faceN.sink( th ) : PSFxSink();
        PSFxSink const facePL = faceHereL ? fx.faceP.sink( th ) : PSFxSink();
        int64_t *const faceSWL = faceSL.w;
        double const avgDensityL = double( userOptions.averageDensity );
        (void)faceSL; (void)faceNL; (void)facePL; (void)faceSWL; (void)avgDensityL;
        // slab mode (slab >= 0): this slab's tetrahedra, in the loop's order, and the local x planes it
        // owns -- every write below skips the cells of other planes
        std::vector<uint32_t> const *slabList = slab >= 0 ? &slabTetLists[size_t(slab)] : nullptr;
        long const slabLoL = slab >= 0 ? slabXLo[size_t(slab)] : 0L;
        long const slabHiL = slab >= 0 ? slabXHi[size_t(slab)] : LONG_MAX;
        size_t planeL = 1;                                   // cells per local x plane
        for (int d = 1; d < NO_DIM; ++d) planeL *= subDims[d];
        size_t slabK = 0, curOrd = 0;
        size_t idx = 0, cellOrdCounter = 0, nextDeferred = 0;
        auto itC = PS_FINITE_CELLS_BEGIN;
        auto const itEnd = PS_FINITE_CELLS_END;
        PSBlockClaim<PSCellIt> *const claimL = deferred ? nullptr : blockClaim;   // the dynamic team: claimed blocks
        size_t inBlockL = 0;                                                       // cells left in the claimed block
        for (;;)
        {
            Cell_handle cell;
            if ( slabList )
            {
                if ( slabK == slabList->size() ) break;
                curOrd = (*slabList)[slabK++];
                cell = slabTets[curOrd];
            }
            else if ( claimL )
            {
                if ( inBlockL == 0 && !claimL->claim( th, itC, inBlockL ) ) break;   // every block taken
                cell = itC;
                ++itC;
                --inBlockL;
            }
            else
            {
                if ( itC == itEnd ) break;
                cell = itC;
                ++itC;
                if ( deferred )
                {
                    size_t const cellOrd = cellOrdCounter++;
                    if ( nextDeferred == deferredOrd.size() ) break;   // every deferred cell done
                    if ( size_t(deferredOrd[nextDeferred]) != cellOrd ) continue;
                    ++nextDeferred;
                }
                else if ( ( (idx++ / DEP_BLOCK) % nT ) != th )
                    continue;
            }
        do {

            // Shared filter chain (ps_cell_filter.h): dummy-vertex skip, Lagrangian-centroid
            // ownership, hull handling, minimum-image wrap, degeneracy + zero-inverse checks.
            PSCellGeometry geo;
            size_t degenerateHere = 0;
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degenerateHere, geo) )
            {
                if ( !slabList || int(slabFirst[curOrd]) == slab ) nDegenerateInverse += degenerateHere;   // once per tet
                continue;
            }
            if ( !metalDeposited ) markFlipped(geo, flippedFlag, slabLoL, slabHiL, countAtomicL);   // (GPU-deferred tets: the host loop marked them)
            Real (&eulerPos)[NO_DIM+1][NO_DIM] = geo.eulerPos;
            Real (&posMatInv)[NO_DIM][NO_DIM] = geo.posMatInv;

            // PS-DTFE per-tetrahedron MASS: each Lagrangian flow element carries a constant mass m = averageDensity * V_lag,
            // conserved as it maps to its Eulerian simplex. We deposit this MASS onto the grid (see the scatter below) rather
            // than point-sample rho = m / V_eul, which at a caustic (V_eul -> 0) either misses the thin simplex or splats a huge
            // value over a whole cell. Depositing m is exact at any nSubL.
            Real tetMass;
            double absDetLag;
            // --ps-caustics: the tetrahedron's caustic bits (collapse multiplicity, umbilic indicator), computed HERE where both
            // edge matrices exist (J = Ax * Lag^-1 is constant inside the tet) and OR-ed into every cell it deposits into.
            PSCausticClass::CausticMask tetCausticClass = 0;
            {
                double Lag[NO_DIM][NO_DIM];
                for (int v = 0; v < NO_DIM; ++v)
                    for (int i = 0; i < NO_DIM; ++i)
                        Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
                // absDetLag always computed: the --ps-halo-release classification stays geometric
                // (V_lag/V_eul) so kept/released sets are identical under either mass convention
                absDetLag = std::fabs(determinant(Lag));
                tetMass = psVertexMassL ? Real( psVertexShareMass(cell) )
                                       : Real( double(userOptions.averageDensity) * absDetLag / factorial(NO_DIM) );  // = rho_bar * V_lag
                if (tetMass < Real(0.)) tetMass = Real(0.);
                if ( fixedShareTestNanEvery() && fixedShareTestNan( double(cell->vertex(0)->point()[0]), double(cell->vertex(0)->point()[1]),
                                                                    double(cell->vertex(0)->point()[NO_DIM-1]) ) )
                    tetMass = std::numeric_limits<Real>::quiet_NaN();   // DTFE_TEST_NAN_SHARE (ps_deposit_params.h)
                if ( auditL && ( !slabList || int(slabFirst[curOrd]) == slab ) )   // once per tet
                { TH.auditTets += double(tetMass); ++TH.auditNTets; }
                if ( psCausticsL )
                {
                    PSCausticClass::TetDeformation const self = PSCausticClass::analyzeTet( geo.Ax, Lag );
                    tetCausticClass = self.bits;

                    /* A3 (cusp) indicator. grad(lambda_c) does not exist inside this tetrahedron (J is constant), so it is
                       estimated from neighbours' critical eigenvalues against their Lagrangian centroid offsets. Only tets
                       already near their fold do this work. The neighbours go through psFilterCell with ownership skipped, so
                       their edge matrices use the deposit's min-image convention and an outside-partition neighbour contributes. */
                    // (2D too: the triangles around the three vertices; no A4 bit in 2D)
                    if ( userOptions.psCausticCusps and self.valid and self.spread > 0.
                         and std::fabs(self.critical) <= PSCausticClass::FOLD_BAND * self.spread )
                    {
                        /* STENCIL: every tetrahedron sharing a VERTEX with this one, not just the four sharing a face: four
                           samples barely determine a 3-D gradient (in one slab the fit is ill-conditioned along exactly the
                           direction the cusp test interrogates, and transverse noise reads as tangency). The vertex-incident
                           ring gives ~20-40 samples in every direction. */
                        static int const MAX_STENCIL = 64;
                        double dq[MAX_STENCIL][NO_DIM], dLam[MAX_STENCIL];
                        int nb = 0;
                        double c0[NO_DIM] = {0.};
                        for (int v = 0; v <= NO_DIM; ++v)
                            for (int d = 0; d < NO_DIM; ++d)
                                c0[d] += double(cell->vertex(v)->point()[d]) / double(NO_DIM+1);

                        std::vector<Cell_handle> ring;
                        ring.reserve( 48 );
                        for (int v = 0; v <= NO_DIM; ++v)
                        {
#if NO_DIM==3
                            dt.incident_cells( cell->vertex(v), std::back_inserter(ring) );
#else
                            auto fc = dt.incident_faces( cell->vertex(v) ), done = fc;
                            if ( fc != nullptr )
                                do { ring.push_back( fc ); } while ( ++fc != done );
#endif
                        }

                        /* Deduplicate, then keep the MAX_STENCIL nearest cells. The raw ring lists a face neighbour once per
                           shared vertex; repeats would re-weight the fit and fill the size cap, dropping the last vertex's
                           cells and biasing the tangency. Selection happens before psFilterCell/analyzeTet (the centroid needs
                           only vertex coordinates), so the expensive analysis runs on 64 cells. */
                        std::sort( ring.begin(), ring.end(),
                                   [](Cell_handle a, Cell_handle b){ return &*a < &*b; } );
                        ring.erase( std::unique( ring.begin(), ring.end() ), ring.end() );

                        struct RingCand { double d2; Cell_handle cell; double nc[NO_DIM]; };
                        std::vector<RingCand> cand;
                        cand.reserve( ring.size() );
                        for (size_t ri = 0; ri < ring.size(); ++ri)
                        {
                            Cell_handle nbCell = ring[ri];
                            if ( nbCell == cell or dt.is_infinite(nbCell) ) continue;
                            RingCand rc;
                            rc.cell = nbCell;
                            double d2 = 0.;
                            for (int d = 0; d < NO_DIM; ++d) rc.nc[d] = 0.;
                            for (int v = 0; v <= NO_DIM; ++v)
                                for (int d = 0; d < NO_DIM; ++d)
                                    rc.nc[d] += double(nbCell->vertex(v)->point()[d]) / double(NO_DIM+1);
                            for (int d = 0; d < NO_DIM; ++d)
                            {
                                double const e = rc.nc[d] - c0[d];
                                d2 += e*e;
                            }
                            rc.d2 = d2;
                            cand.push_back( rc );
                        }
                        /* The nearest MAX_STENCIL in a STRICT order: distance, then Lagrangian centroid, then the cell's
                           vertices. The cells' addresses and the ring's walk follow the insertion history, so neither may pick a
                           tie or order the fit's sums; the order is canonical because the cells' numbering is
                           (ps_vertex_order.h) and two distinct cells differ in their vertex sequence. */
                        std::sort( cand.begin(), cand.end(), [](RingCand const &a, RingCand const &b)
                        {
                            if ( a.d2 != b.d2 ) return a.d2 < b.d2;
                            for (int d = 0; d < NO_DIM; ++d)
                                if ( a.nc[d] != b.nc[d] ) return a.nc[d] < b.nc[d];
                            for (int v = 0; v <= NO_DIM; ++v)
                                for (int d = 0; d < NO_DIM; ++d)
                                {
                                    double const x = double( a.cell->vertex(v)->point()[d] );
                                    double const y = double( b.cell->vertex(v)->point()[d] );
                                    if ( x != y ) return x < y;
                                }
                            return false;
                        } );
                        if ( cand.size() > size_t(MAX_STENCIL) ) cand.resize( MAX_STENCIL );

                        for (size_t ci = 0; ci < cand.size() and nb < MAX_STENCIL; ++ci)
                        {
                            Cell_handle nbCell = cand[ci].cell;
                            PSCellGeometry ngeo;
                            if ( !psFilterCell(nbCell, userOptions, boxCoordinates, true, NULL, ngeo, true) )
                                continue;
                            double nLag[NO_DIM][NO_DIM];
                            for (int v = 0; v < NO_DIM; ++v)
                                for (int i = 0; i < NO_DIM; ++i)
                                    nLag[v][i] = double(nbCell->vertex(v+1)->point()[i])
                                               - double(nbCell->vertex(0)->point()[i]);
                            PSCausticClass::TetDeformation const other =
                                PSCausticClass::analyzeTet( ngeo.Ax, nLag );
                            if ( not other.valid ) continue;
                            for (int d = 0; d < NO_DIM; ++d) dq[nb][d] = cand[ci].nc[d] - c0[d];
                            dLam[nb] = other.critical - self.critical;
                            ++nb;
                        }
                        if ( PSCausticClass::cuspIndicator( self, dq, dLam, nb ) )
                            tetCausticClass |= PSCausticClass::BIT_CUSP;
                        /* A4 from the SAME ring: one extra 5x5 solve. NOT nested inside the A3 branch: cuspIndicator's linear fit
                           reads a true swallowtail's cubic as a slope and says "no cusp" (see foldDerivatives). */
                        if ( PSCausticClass::swallowtailIndicator( self, dq, dLam, nb ) )
                            tetCausticClass |= PSCausticClass::BIT_SWALLOWTAIL;
                    }
                }
            }

            // --ps-halo-release: released tets skip PASS 1 below (zero-trip window), so the
            // empty sample set drives the mass-conserving centroid fallback -- a monolithic
            // single-cell deposit with the centroid-evaluated velocity.
            bool const released = psHaloRelease > 0. && absDetLag > psHaloRelease * geo.cellAbsDet;
            if (released && (!slabList || int(slabFirst[curOrd]) == slab)) ++nReleased;   // once per tet

            // Eulerian bounding box of this cell (from the wrapped positions). psLoopSpanReaches repeats this block's
            // span, "outsideGrid" and window test for psFaceWalk, and psOwnsUpperBoundarySection its face-only case:
            // KEEP THEM IN STEP.
            Real eMin[NO_DIM], eMax[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
            {
                eMin[d] = eulerPos[0][d];
                eMax[d] = eulerPos[0][d];
            }
            for (int v = 1; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                {
                    if (eulerPos[v][d] < eMin[d]) eMin[d] = eulerPos[v][d];
                    if (eulerPos[v][d] > eMax[d]) eMax[d] = eulerPos[v][d];
                }

            // Grid index range overlapping the bounding box (periodic indices are wrapped below). A non-periodic grid clamps it;
            // iMinU/iMaxU keep the unclamped extent and 'straddle' marks a tet reaching beyond the grid (a cut --box region).
            int iMin[NO_DIM], iMax[NO_DIM], iMinU[NO_DIM], iMaxU[NO_DIM];
            bool outsideGrid = false, straddle = false;
            for (int d = 0; d < NO_DIM; ++d)
            {
                iMin[d] = int(floor((eMin[d] - boxL[2*d]) / dxL[d]));
                iMax[d] = int(floor((eMax[d] - boxL[2*d]) / dxL[d])) + 1;
                // straddle is decided on the tet's own bbox: a tet whose bbox stays inside the grid has
                // no sample beyond it (a one-cell margin would flag every tet within a cell of a region
                // face, all for nothing)
                if (!periodicL && (iMin[d] < 0 || iMax[d] > (int)nGridL[d])) straddle = true;
                // No one-cell sample margin (as in the GPU kernels): a sub-sample sits >= 1/(2 nSub) inside its cell, so every
                // sample in the bbox lies in a cell of this window; a ring's samples would all be outside the tet.
                iMinU[d] = iMin[d]; iMaxU[d] = iMax[d];
                if (!periodicL)
                {
                    if (iMin[d] < 0) iMin[d] = 0;
                    if (iMax[d] > (int)nGridL[d]) iMax[d] = (int)nGridL[d];
                    if (iMin[d] >= (int)nGridL[d] || iMax[d] <= 0) { outsideGrid = true; break; }
                }
            }
            if (outsideGrid)
            {
    #if NO_DIM==3
                // --face-gradient: outside the grid but owning its section in an upper boundary plane (the upper face of the last
                // layer, which no tetrahedron inside owns: psOwnsUpperBoundarySection, as psFaceWalk asks): that section only, no
                // volume, stream or moment share.
                if ( faceSWL
                     && psOwnsUpperBoundarySection( eulerPos, boxL, dxL, nGridL, periodicL, windowModeL ) )
                {
                    Vertex_handle base = cell->vertex(0);
                    Real faceGrad[NO_DIM][noVelComp], temp[NO_DIM][noVelComp];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (size_t i = 0; i < noVelComp; ++i)
                            temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
                    matrixMultiplication<noVelComp>(posMatInv, temp, faceGrad);
                    double v0[noVelComp];
                    for (size_t j = 0; j < noVelComp; ++j) v0[j] = double( base->info().velocity(j) );
                    double const vTet = double(geo.cellAbsDet) / double(factorial(NO_DIM));
                    double const rhoTet = ( vTet > 0. && avgDensityL > 0. ) ? double(tetMass) / ( vTet * avgDensityL ) : 0.;
                    psDepositFaces( geo.eulerPos, boxL, dxL, nGridL, periodicL, subOriginL, subDimsL, slabLoL, slabHiL,
                                    faceGrad, v0, faceSL, faceNL, facePL, rhoTet, massFxAtomicL );
                }
    #endif
                continue;
            }
            if ( windowModeL )
            {   // --grid-window: a tetrahedron that misses the window on some axis deposits nothing
                bool touches = true;
                for (int d = 0; d < NO_DIM && touches; ++d)
                    touches = psSpanTouches(iMinU[d], iMaxU[d], long(nGridL[d]), periodicL, long(subOriginL[d]), long(subDimsL[d]));
                if ( !touches ) continue;
            }
            if (released)
                for (int d = 0; d < NO_DIM; ++d) iMax[d] = iMin[d];   // zero-trip PASS 1 -> centroid fallback

            // (density is deposited as the per-tetrahedron mass tetMass computed above; no per-vertex
            //  density gradient is needed for the scatter.)

            // Constant velocity gradient (also needed to evaluate the per-stream velocity for the dispersion).
            Real velGrad[NO_DIM][noVelComp];
            if (fVelL || fVGradL || fVDispL)
            {
                Vertex_handle base = cell->vertex(0);
                Real temp[NO_DIM][noVelComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t i = 0; i < noVelComp; ++i)
                        temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
                matrixMultiplication<noVelComp>(posMatInv, temp, velGrad);
            }

            // Constant scalar gradient across this cell.
    #ifdef SCALAR
            Real sGrad[NO_DIM][noScalarComp];
            if (fScalL || fSGradL)
            {
                Real temp[NO_DIM][noScalarComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t i = 0; i < noScalarComp; ++i)
                        temp[v][i] = cell->vertex(v+1)->info().myScalar()[i] - cell->vertex(0)->info().myScalar()[i];
                matrixMultiplication<noScalarComp>(posMatInv, temp, sGrad);
            }
    #endif

            // ===================== exact conservative deposit (--exact) =====================
            // r3d (Powell & Abel 2015): clip the Eulerian tetrahedron against every grid cell in its bbox window and integrate
            // the moments analytically, in coordinates RELATIVE to (wrapped) vertex 0 (the frame of velGrad). Released tets
            // (--ps-halo-release) fall through to the sampled path, whose zero-tripped window drives the centroid fallback.
            if ( psExactL && !released )
            {
                // the moments of a piece: [1, x.., then the second moments a <= b row by row] --
                // 10 in 3D (r3d's order-2 layout), 6 in 2D (ps_clip2d.h)
                static int const NMOM = 1 + NO_DIM + NO_DIM*(NO_DIM+1)/2;
    #if NO_DIM==3
                r3d_rvec3 tv[NO_DIM+1];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        tv[v].xyz[d] = double(eulerPos[v][d]) - double(eulerPos[0][d]);
                // r3d_init_tet wants positive orientation; a folded (negative-parity) tet is
                // handed over with vertices 1,2 swapped -- vertex 0 stays the frame origin
                if ( geo.cellDet < 0. )
                {
                    r3d_rvec3 const tmp = tv[1]; tv[1] = tv[2]; tv[2] = tmp;
                }
                r3d_poly tetPoly;
                r3d_init_tet(&tetPoly, tv);
    #else
                // 2D: the triangle in the vertex-0 frame, counter-clockwise (a folded one swapped)
                double tri2[NO_DIM+1][NO_DIM];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        tri2[v][d] = double(eulerPos[v][d]) - double(eulerPos[0][d]);
                if ( geo.cellDet < 0. )
                    for (int d = 0; d < NO_DIM; ++d) std::swap( tri2[1][d], tri2[2][d] );
    #endif

                // PASS 1 (exact): moments of tet ∩ cell for every window cell with nonzero volume, order 2, in the vertex-0 frame.
                // --grid-window clips only the cells of the window: the tet is normalized by its WHOLE weight below, so the
                // pieces outside the window are never needed.
                int cLo[NO_DIM], cHi[NO_DIM];
                bool missesWindow = false;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    cLo[d] = iMin[d]; cHi[d] = iMax[d];
                    if ( !windowModeL ) continue;
                    long hl = 0, hh = 0;
                    if ( !psWindowHull(iMin[d], iMax[d], long(nGridL[d]), periodicL, long(subOriginL[d]), long(subDimsL[d]), hl, hh) )
                    { missesWindow = true; break; }
                    cLo[d] = int(hl); cHi[d] = int(hh);
                }
                if ( missesWindow ) continue;   // next tetrahedron: nothing of it lies in the window
                insideFlat.clear();          // reused scratch: flat sub-grid index per hit cell
                exMom.clear();
                exW.clear();
                double sumW = 0.;            // per-tet weight normalizer (the volume)
                // a hit cell's moments and its mass weight: the exact volume
                auto addPiece = [&](size_t const flatIdx, double const *mom)
                {
                    insideFlat.push_back(flatIdx);
                    for (int m = 0; m < NMOM; ++m) exMom.push_back(mom[m]);
                    double w = mom[0];
                    exW.push_back(w);
                    sumW += w;
                };
    #if NO_DIM==3
                // the window's cells, each clipped by its 6 planes (exact_sweep.h: once per slab and column).
                // The planes are in the vertex-0 frame at the RAW indices (the periodic image the wrapped tet
                // actually intersects). A cell outside the sub-box is not clipped: a partition's box holds every
                // cell its tets touch, and a window's hull may hold cells outside the window.
                exactSweep::sweep( tetPoly, cLo, cHi,
                    [&](int const d, int const raw, double &lo, double &hi)
                    {
                        lo = double(boxL[2*d]) + raw * double(dxL[d]) - double(eulerPos[0][d]);
                        hi = lo + double(dxL[d]);
                    },
                    [&](int const d, int const raw) -> long
                    {
                        int const wg = periodicL ? ((raw % (int)nGridL[d] + (int)nGridL[d]) % (int)nGridL[d]) : raw;
                        long loc = (long)wg - (long)subOriginL[d];
                        if (loc < 0) loc += (long)nGridL[d];        // sub-box may wrap a periodic axis
                        return (loc < 0 || loc >= (long)subDimsL[d]) ? -1 : loc;
                    },
                    [&](int const *, long const *loc, r3d_poly &piece)
                    {
                        r3d_real mom[10];
                        exactSweep::reduce<2>( piece, mom );
                        if ( !(mom[0] > 0.) ) return;
                        addPiece( (size_t(loc[0]) * subDimsL[1] + size_t(loc[1])) * subDimsL[2] + size_t(loc[2]), mom );
                    } );
    #else
                for (int gi = cLo[0]; gi < cHi[0]; ++gi)
                for (int gj = cLo[1]; gj < cHi[1]; ++gj)
                {
                    int const wgi = periodicL ? ((gi % (int)nGridL[0] + (int)nGridL[0]) % (int)nGridL[0]) : gi;
                    int const wgj = periodicL ? ((gj % (int)nGridL[1] + (int)nGridL[1]) % (int)nGridL[1]) : gj;
                    int const gridIdx[2] = {wgi, wgj};
                    int const rawIdx[2]  = {gi, gj};
                    size_t flatIdx = 0;
                    bool inSub = true;
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        long loc = (long)gridIdx[d] - (long)subOriginL[d];
                        if (loc < 0) loc += (long)nGridL[d];        // sub-box may wrap a periodic axis
                        if (loc < 0 || loc >= (long)subDimsL[d]) { inSub = false; break; }
                        flatIdx = flatIdx * subDimsL[d] + (size_t)loc;
                    }
                    // outside the sub-box: a partition's box holds every cell its tets touch, and a
                    // window's hull may hold cells outside the window -- neither is clipped
                    if (!inSub) continue;

                    // cell bounds in the vertex-0 frame (RAW indices: the periodic image the
                    // wrapped tet actually intersects)
                    double lo2[NO_DIM], hi2[NO_DIM];
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        lo2[d] = double(boxL[2*d]) + rawIdx[d] * double(dxL[d]) - double(eulerPos[0][d]);
                        hi2[d] = lo2[d] + double(dxL[d]);
                    }
                    double mom[psClip2d::MOMENTS];
                    if ( not psClip2d::clipTriangleToCell( tri2, lo2, hi2, mom ) ) continue;
                    addPiece( flatIdx, mom );
                }
    #endif

    #if NO_DIM==3
                // --face-gradient: this tetrahedron's face sums (psDepositFaces). Ahead of the volume gate below, not
                // under it: psFaceWalk forms the same sums beside a GPU deposit and knows no clipped volume, so both take
                // every tetrahedron that passed the filter and the span test above (and the span block's face-only
                // case, outside the grid with an upper boundary section). (A face share needs the tetrahedron
                // on both sides of its plane, so the gate would only withhold the sections of a sliver r3d loses.)
                if ( faceSWL )
                {
                    double v0[noVelComp];
                    for (size_t j = 0; j < noVelComp; ++j) v0[j] = double( cell->vertex(0)->info().velocity(j) );
                    double const vTet = double(geo.cellAbsDet) / double(factorial(NO_DIM));
                    double const rhoTet = ( vTet > 0. && avgDensityL > 0. ) ? double(tetMass) / ( vTet * avgDensityL ) : 0.;
                    psDepositFaces( geo.eulerPos, boxL, dxL, nGridL, periodicL, subOriginL, subDimsL, slabLoL, slabHiL,
                                    velGrad, v0, faceSL, faceNL, facePL, rhoTet, massFxAtomicL );
                }
    #endif

                // PASS 2 (exact): renormalize the shares to tetMass, exact per tet. A tet that leaves a non-periodic grid is
                // normalized by its WHOLE weight instead (the analytic volume; max() guards rounding), so the grid keeps exactly
                // its inside share. A --grid-window run normalizes EVERY tet this way. An all-empty window (a sliver below r3d's
                // resolution, or a tet wholly beyond the grid) falls through to the sampled path, except under --grid-window
                // (no volume inside the window: the tet deposits nothing here).
                if ( sumW > 0. )
                {
                    double norm = sumW;
                    if ( straddle || windowModeL )
                    {
                        double const vol = double(geo.cellAbsDet) / double(factorial(NO_DIM));
                        double wFull = vol;
                        norm = std::max( sumW, wFull );
                    }
                    double const shareFac = double(tetMass) / norm;
                    for (size_t c = 0; c < insideFlat.size(); ++c)
                    {
                        size_t const flatIdx = insideFlat[c];
                        if ( slabList ) { long const xl = long(flatIdx / planeL); if ( xl < slabLoL || xl >= slabHiL ) continue; }
                        double const *m = &exMom[c*NMOM];
                        Real const massShare = Real( exW[c] * shareFac );
                        // moment weight: the analytic intersection volume V_int (the exact continuum volume share)
                        Real const momShare = Real( m[0] );
                        if (fDenL)
                        {
                            if ( massFxL )
                                psFxAdd( massSinkL, flatIdx, double(massShare), FX_DROP_MASS, massFxAtomicL );
                            else quantities->density[flatIdx] += massShare;
                        }
                        if (auditL) TH.auditShares += double(massShare);
                        if (needWeightL)
                        {
                            if ( fxWL[PS_FX_MOMW] ) fxAddL( PS_FX_MOMW, flatIdx, momShare );
                            else massWeight[flatIdx] += momShare;
                        }

                        // exact mean of the linear velocity profile over this intersection
                        Pvector<Real,noVelComp> velBar;
                        if (needVelValuesL)
                        {
                            double const invV = 1. / m[0];
                            double cen[NO_DIM];
                            for (int d = 0; d < NO_DIM; ++d) cen[d] = m[1+d]*invV;
                            for (size_t j = 0; j < noVelComp; ++j)
                            {
                                double vb = double( cell->vertex(0)->info().velocity(j) );
                                for (int i = 0; i < NO_DIM; ++i)
                                    vb += double(velGrad[i][j]) * cen[i];
                                velBar[j] = Real(vb);
                            }
                            if (haveVelL)
                            {
                                if ( fxWL[PS_FX_MOM] )
                                    for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_MOM, flatIdx*noVelComp + j, velBar[j] * momShare );
                                else quantities->velocity[flatIdx] += velBar * momShare;
                            }
                        }
                        if (fVGradL)
                        {
                            Pvector<Real,noGradComp> grad;
                            for (size_t j = 0; j < noVelComp; ++j)
                                for (int i = 0; i < NO_DIM; ++i)
                                    grad[j*NO_DIM+i] = velGrad[i][j];
                            if ( fxWL[PS_FX_GRAD] )
                                for (size_t q = 0; q < noGradComp; ++q) fxAddL( PS_FX_GRAD, flatIdx*noGradComp + q, grad[q] * momShare );
                            else quantities->velocity_gradient[flatIdx] += grad * momShare;
                        }
                        if (fVDispL)
                        {
                            // exact second moment of the linear profile: <v_i v_j> over the piece
                            // = vbar_i vbar_j + (G^T Cov G)_ij with Cov the piece's position
                            // covariance from the order-2 moments (M2 index map: xx xy xz yy yz zz)
                            double const invV = 1. / m[0];
                            double cen[NO_DIM];
                            for (int d = 0; d < NO_DIM; ++d) cen[d] = m[1+d]*invV;
                            double cov[NO_DIM][NO_DIM];
                            {   // second moments a <= b row by row (3D: xx xy xz yy yz zz, 2D: xx xy yy)
                                int k = 1 + NO_DIM;
                                for (int a = 0; a < NO_DIM; ++a)
                                    for (int b = a; b < NO_DIM; ++b, ++k)
                                        cov[a][b] = cov[b][a] = m[k]*invV - cen[a]*cen[b];
                            }
                            size_t c2 = 0;
                            for (int a = 0; a < NO_DIM; ++a)
                                for (int b = a; b < NO_DIM; ++b)
                                {
                                    double gcg = 0.;
                                    for (int i = 0; i < NO_DIM; ++i)
                                        for (int k = 0; k < NO_DIM; ++k)
                                            gcg += double(velGrad[i][a]) * cov[i][k] * double(velGrad[k][b]);
                                    // (the Real grid's add keeps its one expression: a contracted multiply-add there)
                                    if ( fxWL[PS_FX_M2] ) fxAddL( PS_FX_M2, flatIdx*noDispComp + c2, massShare * ( velBar[a] * velBar[b] + Real(gcg) ) );
                                    else quantities->velocity_dispersion[flatIdx][c2] += massShare * ( velBar[a] * velBar[b] + Real(gcg) );
                                    ++c2;
                                }
                            if ( dispNeedsOwnWeightL )
                            {
                                if ( fxWL[PS_FX_DW] ) fxAddL( PS_FX_DW, flatIdx, massShare );
                                else if ( !dwOffL ) dispWeight[flatIdx] += massShare;
                                if ( fxWL[PS_FX_DV] )
                                    for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_DV, flatIdx*noVelComp + j, velBar[j] * massShare );
                                else dispVel[flatIdx] += velBar * massShare;
                            }
                        }
    #ifdef SCALAR
                        if (fScalL)
                        {
                            // exact mean of the linear scalar profile (same construction as velBar)
                            double const invV = 1. / m[0];
                            double cen[NO_DIM];
                            for (int d = 0; d < NO_DIM; ++d) cen[d] = m[1+d]*invV;
                            Pvector<Real,noScalarComp> scalarVal;
                            for (size_t j = 0; j < noScalarComp; ++j)
                            {
                                double sb = double( cell->vertex(0)->info().myScalar()[j] );
                                for (int i = 0; i < NO_DIM; ++i)
                                    sb += double(sGrad[i][j]) * cen[i];
                                scalarVal[j] = Real(sb);
                            }
                            if ( fxWL[PS_FX_SCAL] )
                                for (size_t j = 0; j < noScalarComp; ++j) fxAddL( PS_FX_SCAL, flatIdx*noScalarComp + j, scalarVal[j] * momShare );
                            else quantities->scalar[flatIdx] += scalarVal * momShare;
                        }
                        if (fSGradL)
                        {
                            Pvector<Real,noScalarGradComp> sgrad;
                            for (size_t j = 0; j < noScalarComp; ++j)
                                for (int i = 0; i < NO_DIM; ++i)
                                    sgrad[j*NO_DIM+i] = sGrad[i][j];
                            if ( fxWL[PS_FX_SG] )
                                for (size_t q = 0; q < noScalarGradComp; ++q) fxAddL( PS_FX_SG, flatIdx*noScalarGradComp + q, sgrad[q] * momShare );
                            else quantities->scalar_gradient[flatIdx] += sgrad * momShare;
                        }
    #endif
                        psAddCount( streamCount[flatIdx], countAtomicL );   // raw tet-touch count -> '.tetTouch'
                        if ( fxWL[PS_FX_SV] ) fxAddL( PS_FX_SV, flatIdx, Real( m[0] / double(cellVolumeL) ) );
                        else exactMult[flatIdx] += Real( m[0] / double(cellVolumeL) );   // exact multiplicity share
                        if ( psCausticsL )
                            psOrMask( orientBits[flatIdx], (PSCausticClass::CausticMask)((geo.cellDet > 0. ? 1 : 2) | tetCausticClass), countAtomicL );
                    }
                    continue;   // next tetrahedron (the sampled PASS 1/2 below is skipped)
                }
                if ( windowModeL ) continue;   // no volume in the window: nothing to deposit (see above)
            }

            // ===================== Mass-conserving deposit of this tetrahedron =====================
            // PASS 1: gather every sub-sample point (nSubL^NO_DIM per grid cell in the Eulerian bbox) that
            // lies inside the Eulerian simplex, recording its grid cell (flatIdx) and offset rel from
            // vertex 0. Sub-sampling probes the simplex volume uniformly.
            insideFlat.clear();
            insideRel.clear();

            // exact-containment state of this tetrahedron: double inverse + tier-1 band on the
            // wrapped geometry; the canonical tiers' state (from the stored, unwrapped vertices)
            // is built only if some sample lands within band1 of a face
            double invD[NO_DIM][NO_DIM];
            double band1 = std::numeric_limits<double>::infinity();   // no inverse: all to the exact tiers
            if ( psExactInside::inverseNd( geo.Ax, invD ) )
                band1 = psExactInside::fineBand( exactFrame, geo.Ax, geo.cellAbsDet );
            else
                for (int a = 0; a < NO_DIM; ++a)
                    for (int b = 0; b < NO_DIM; ++b) invD[a][b] = 0.;
            Real rawPos[NO_DIM+1][NO_DIM];
            bool rawPosReady = false;
            psExactInside::TetTest exactTet;
    #if NO_DIM==2
            for (int gi = iMin[0]; gi < iMax[0]; ++gi)
            for (int gj = iMin[1]; gj < iMax[1]; ++gj)
            {
                int wgi = periodicL ? ((gi % (int)nGridL[0] + (int)nGridL[0]) % (int)nGridL[0]) : gi;
                int wgj = periodicL ? ((gj % (int)nGridL[1] + (int)nGridL[1]) % (int)nGridL[1]) : gj;
                int gridIdx[2] = {wgi, wgj};
                int rawIdx[2]  = {gi, gj};
    #elif NO_DIM==3
            for (int gi = iMin[0]; gi < iMax[0]; ++gi)
            for (int gj = iMin[1]; gj < iMax[1]; ++gj)
            for (int gk = iMin[2]; gk < iMax[2]; ++gk)
            {
                int wgi = periodicL ? ((gi % (int)nGridL[0] + (int)nGridL[0]) % (int)nGridL[0]) : gi;
                int wgj = periodicL ? ((gj % (int)nGridL[1] + (int)nGridL[1]) % (int)nGridL[1]) : gj;
                int wgk = periodicL ? ((gk % (int)nGridL[2] + (int)nGridL[2]) % (int)nGridL[2]) : gk;
                int gridIdx[3] = {wgi, wgj, wgk};
                int rawIdx[3]  = {gi, gj, gk};
    #endif
                // flat index within this partition's sub-grid; inSub guard skips cells outside it --
                // except under --grid-window, where the cell's samples still COUNT (the tet's full-run
                // normalization) and are marked so pass 2 deposits none of them
                size_t flatIdx = 0;
                bool inSub = true;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    long loc = (long)gridIdx[d] - (long)subOriginL[d];
                    if (loc < 0) loc += (long)nGridL[d];            // sub-box may wrap a periodic axis
                    if (loc < 0 || loc >= (long)subDimsL[d]) { inSub = false; break; }
                    flatIdx = flatIdx * subDimsL[d] + (size_t)loc;
                }
                if (!inSub) { if (!windowModeL) continue; flatIdx = PS_NOT_IN_WINDOW; }

                for (size_t sIdx = 0; sIdx < nSamplesL; ++sIdx)
                {
                    // Exact point-in-Eulerian-simplex test (ps_exact_inside.h) at the sample's double
                    // position: tier 1 = barycentrics of its minimum-image offset from (wrapped)
                    // vertex 0, decisive outside band1; the rest go to the canonical tiers.
                    {
                        double pos[NO_DIM], relD[NO_DIM], baryD[NO_DIM];
                        size_t remD = sIdx;
                        for (int d = 0; d < NO_DIM; ++d)
                        {
                            pos[d] = samplePosL[d][ size_t(gridIdx[d]) * size_t(nSubL) + remD % size_t(nSubL) ];
                            remD /= size_t(nSubL);
                            double r = pos[d] - double(eulerPos[0][d]);
                            if ( periodicL )
                            {
                                if (r >  0.5 * gridLenL[d]) r -= gridLenL[d];
                                if (r < -0.5 * gridLenL[d]) r += gridLenL[d];
                            }
                            relD[d] = r;
                        }
                        int const fine = psExactInside::fineClassify( invD, relD, band1, baryD );
                        if ( fine < 0 ) continue;
                        if ( fine == 0 )
                        {
                            if ( not rawPosReady )
                            {
                                for (int v = 0; v <= NO_DIM; ++v)
                                    for (int d = 0; d < NO_DIM; ++d)
                                        rawPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);
                                rawPosReady = true;
                            }
                            double pc[NO_DIM];
                            psExactInside::canonicalQuery( exactFrame, pos, pc );
                            if ( not psExactInside::canonicalInside( exactFrame, rawPos, exactTet, pc ) ) continue;
                        }
                    }

                    // Offset from (wrapped) vertex 0 for the deposited moments: the plain float
                    // expression -- the exact test decides only WHICH samples a tetrahedron keeps,
                    // not the bits a kept sample deposits.
                    Real rel[NO_DIM];
                    size_t rem = sIdx;
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        Real frac = (nSubL == 1) ? Real(0.5)
                                  : ( Real(int(rem % size_t(nSubL))) + Real(0.5) ) / Real(nSubL);
                        rem /= size_t(nSubL);
                        Real const gridPos = boxL[2*d] + (rawIdx[d] + frac) * dxL[d];
                        rel[d] = gridPos - eulerPos[0][d];
                    }

                    insideFlat.push_back(flatIdx);
                    for (int d = 0; d < NO_DIM; ++d) insideRel.push_back(rel[d]);
                }
            }

            // A tet that leaves a non-periodic grid (straddle) also counts its samples BEYOND the grid, with the same exact test:
            // every sample then carries tetMass / (all its samples), so the grid keeps only the mass share that lies inside it
            // (not piling the outside share onto a --box region's faces). Out-of-grid sample positions use the expression that
            // builds samplePosAxis, so an in-grid index would give the table's value bit for bit.
            size_t nOutside = 0;
    #if NO_DIM==3
            // a straddling tet with no sample inside the grid and its centroid beyond it deposits nothing,
            // whatever it holds beyond the grid: its count is skipped
            bool needBeyond = straddle && !released;
            if ( needBeyond && insideFlat.empty() )
            {
                Real cBeyond[NO_DIM];
                size_t fBeyond = 0;
                needBeyond = centroidCell(eulerPos, cBeyond, fBeyond);
            }
            if ( needBeyond )
            {
                PSBeyondTet bt;   // copies: nothing of this loop's state may escape (see psCountBeyondGrid)
                for (int d = 0; d < NO_DIM; ++d)
                {
                    bt.iMinU[d] = iMinU[d]; bt.iMaxU[d] = iMaxU[d];
                    for (int v = 0; v <= NO_DIM; ++v) bt.eulerPos[v][d] = eulerPos[v][d];
                    for (int e = 0; e < NO_DIM; ++e) bt.invD[d][e] = invD[d][e];
                }
                bt.band1 = band1; bt.cell = cell;
                PSBeyondCount const bc = psCountBeyondGrid( beyondGrid, bt, exactFrame );
                nOutside = bc.n;
            }
    #endif
            // every sample of this tet lies beyond the grid: its mass is not the region's
            if ( insideFlat.empty() && nOutside > 0 ) continue;

            // Fallback: a simplex smaller than the sub-sample spacing may catch no interior point. Deposit
            // its whole mass at the centroid's cell (the centroid is always inside) so no mass is ever lost
            // -- this is what makes even nSubL==1 mass-conserving (no thin-stream undercount).
            bool useCentroid = false;
            size_t centroidFlat = 0;
            Real   centroidRel[NO_DIM];
            if (insideFlat.empty())
            {
                Real centroid[NO_DIM];
                size_t f = 0;
                if ( !centroidCell(eulerPos, centroid, f) ) { continue; }   // simplex centroid outside this grid/partition region: drop it
                useCentroid = true;
                centroidFlat = f;
                for (int d = 0; d < NO_DIM; ++d) centroidRel[d] = centroid[d] - eulerPos[0][d];
            }

            // PASS 2: split tetMass among the interior samples and deposit. Equal shares reproduce the mass
            // distribution across the cells the simplex covers, and the total is exactly tetMass (at any
            // nSubL and for any V_eul) -- no cell can receive more than the tetrahedron's mass.
            size_t const nInside = useCentroid ? size_t(1) : insideFlat.size();
            size_t const nShare  = nInside + nOutside;   // the samples the mass is shared among
            Real   const uniformShare = tetMass / Real(nShare);
            // the velocity moments' weight: an equal Eulerian-volume share per sample (the centroid
            // fallback carries the whole tet volume); density keeps mass
            Real   const volumeShare  = Real( geo.cellAbsDet / factorial(NO_DIM) ) / Real(nShare);
            for (size_t s = 0; s < nInside; ++s)
            {
                size_t flatIdx;
                Real rel[NO_DIM];
                if (useCentroid)
                {
                    flatIdx = centroidFlat;
                    for (int d = 0; d < NO_DIM; ++d) rel[d] = centroidRel[d];
                }
                else
                {
                    flatIdx = insideFlat[s];
                    if (flatIdx == PS_NOT_IN_WINDOW) continue;   // counted in nShare, not the window's
                    for (int d = 0; d < NO_DIM; ++d) rel[d] = insideRel[s*NO_DIM + d];
                }
                if ( slabList ) { long const xl = long(flatIdx / planeL); if ( xl < slabLoL || xl >= slabHiL ) continue; }
                Real const massShare = uniformShare;
                // moment weight: the equal volume share (volume-average means); density always deposits MASS
                Real const momShare = volumeShare;

                // density accumulates MASS here; it is divided by the cell volume once at the end. Single build:
                // into the fixed-point grid (massFx), exact; converted to the float density once after the loop.
                if (fDenL)
                {
                    if ( massFxL ) psFxAdd( massSinkL, flatIdx, double(massShare), FX_DROP_MASS, massFxAtomicL );
                    else quantities->density[flatIdx] += massShare;
                }
                if (auditL) TH.auditShares += double(massShare);
                if (needWeightL)
                {
                    if ( fxWL[PS_FX_MOMW] ) fxAddL( PS_FX_MOMW, flatIdx, momShare );
                    else massWeight[flatIdx] += momShare;
                }

                // Weighted velocity moment sum(w_s v_s); also needed for the dispersion (uses <v>).
                Pvector<Real,noVelComp> velVal;
                if (needVelValuesL)
                {
                    velVal = cell->vertex(0)->info().velocity();
                    for (int i = 0; i < NO_DIM; ++i)
                        for (size_t j = 0; j < noVelComp; ++j)
                            velVal[j] += velGrad[i][j] * rel[i];
                    if (haveVelL)
                    {
                        if ( fxWL[PS_FX_MOM] )
                            for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_MOM, flatIdx*noVelComp + j, velVal[j] * momShare );
                        else quantities->velocity[flatIdx] += velVal * momShare;
                    }
                }

                if (fVGradL)
                {
                    Pvector<Real,noGradComp> grad;
                    for (size_t j = 0; j < noVelComp; ++j)
                        for (int i = 0; i < NO_DIM; ++i)
                            grad[j*NO_DIM+i] = velGrad[i][j];
                    if ( fxWL[PS_FX_GRAD] )
                        for (size_t q = 0; q < noGradComp; ++q) fxAddL( PS_FX_GRAD, flatIdx*noGradComp + q, grad[q] * momShare );
                    else quantities->velocity_gradient[flatIdx] += grad * momShare;
                }

                // velocity dispersion: weighted second moment sum(w_s v_i v_j) (upper triangle).
                // After normalization -> sigma_ij = <v_i v_j> - <v_i><v_j>.
                if (fVDispL)
                {
                    size_t c = 0;
                    for (int i = 0; i < NO_DIM; ++i)
                        for (int j = i; j < NO_DIM; ++j, ++c)
                        {
                            if ( fxWL[PS_FX_M2] ) fxAddL( PS_FX_M2, flatIdx*noDispComp + c, massShare * velVal[i] * velVal[j] );
                            else quantities->velocity_dispersion[flatIdx][c] += massShare * velVal[i] * velVal[j];
                        }
                    if ( dispNeedsOwnWeightL )
                    {
                        if ( fxWL[PS_FX_DW] ) fxAddL( PS_FX_DW, flatIdx, massShare );
                        else if ( !dwOffL ) dispWeight[flatIdx] += massShare;
                        if ( fxWL[PS_FX_DV] )
                            for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_DV, flatIdx*noVelComp + j, velVal[j] * massShare );
                        else dispVel[flatIdx] += velVal * massShare;
                    }
                }

    #ifdef SCALAR
                if (fScalL)
                {
                    Pvector<Real,noScalarComp> scalarVal = cell->vertex(0)->info().myScalar();
                    for (int i = 0; i < NO_DIM; ++i)
                        for (size_t j = 0; j < noScalarComp; ++j)
                            scalarVal[j] += sGrad[i][j] * rel[i];
                    if ( fxWL[PS_FX_SCAL] )
                        for (size_t j = 0; j < noScalarComp; ++j) fxAddL( PS_FX_SCAL, flatIdx*noScalarComp + j, scalarVal[j] * momShare );
                    else quantities->scalar[flatIdx] += scalarVal * momShare;
                }
                if (fSGradL)
                {
                    Pvector<Real,noScalarGradComp> sgrad;
                    for (size_t j = 0; j < noScalarComp; ++j)
                        for (int i = 0; i < NO_DIM; ++i)
                            sgrad[j*NO_DIM+i] = sGrad[i][j];
                    if ( fxWL[PS_FX_SG] )
                        for (size_t q = 0; q < noScalarGradComp; ++q) fxAddL( PS_FX_SG, flatIdx*noScalarGradComp + q, sgrad[q] * momShare );
                    else quantities->scalar_gradient[flatIdx] += sgrad * momShare;
                }
    #endif

                // the stream multiplicity (see streamCount): one count per real SAMPLE; a released
                // tet (never sampled) adds its volume fraction; a sampled tet that caught no sample
                // (useCentroid) moves mass only. Under psExactL the counter is the tet-touch count.
                if ( useCentroid )
                    psSetFlag( subSampleMassFlag[flatIdx], countAtomicL );   // mass from a tet with no sample here
                if ( psExactL )
                    psAddCount( streamCount[flatIdx], countAtomicL );
                else if ( released )
                {
                    if ( fxWL[PS_FX_RM] ) fxAddL( PS_FX_RM, flatIdx, Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) ) );
                    else releasedMult[flatIdx] += Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) );
                }
                else if ( !useCentroid )
                    psAddCount( streamCount[flatIdx], countAtomicL );
                // --exact fall-through (empty clip window, released or sub-resolution
                // tets): the tet's volume is shared equally among its nInside samples, so each
                // carries V_tet/nInside -- for the monolithic centroid fallback (nInside==1) that
                // is exactly V_tet/V_cell, the same volume fraction the exact path deposits.
                if ( psExactL )
                {
                    if ( fxWL[PS_FX_SV] ) fxAddL( PS_FX_SV, flatIdx, Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) ) / Real(nShare) );
                    else exactMult[flatIdx] += Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) )
                                               / Real(nShare);
                }
                if ( psCausticsL )
                    // fold parity (sign of det(Ax)) plus this tetrahedron's stratification bits
                    psOrMask( orientBits[flatIdx], (PSCausticClass::CausticMask)((geo.cellDet > 0. ? 1 : 2) | tetCausticClass), countAtomicL );
            }
        } while (false);
        }   // end cell loop
    };

    double const auditGridBefore = ( massAudit && field.density ) ? gridSum() : 0.;   // the CPU loop's starting grid
    auto const tCpuLoop0 = std::chrono::steady_clock::now();
    if ( metalDeposited && !deferredOrd.empty() && !slabMode )
        depositLoop( depT[0], 0, 1, true, -1 );
    else if ( dynamicTeam )
    {   // the dynamic team (PSBlockClaim): depT holds a thread's state for every thread it may grow to
        size_t const nT = size_t(nDepThreads);
        PSBlockClaim<PSCellIt> claim;
        claim.blockLen = DEP_BLOCK;
        claim.blockStart.reserve( noTotalCells / DEP_BLOCK + 1 );
        {
            size_t k = 0;
            for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END; ++it, ++k)
                if ( k % DEP_BLOCK == 0 ) claim.blockStart.push_back( it );
            claim.nCells = k;
        }
        std::vector<std::exception_ptr> errors( nT );
        auto run = [&](size_t const th)
        {
            try { depositLoop( depT[th], th, nT, false, -1 ); }
            catch (...) { errors[th] = std::current_exception(); }
        };
        // a helper's core comes from the pool (unless pinned) and goes back the moment it runs out of blocks
        auto helper = [&](size_t const th) { run( th ); if ( teamGrows ) CorePool::give( 1 ); };
        CorePool::Hold const pinnedHold( teamGrows ? 0 : int(nT) - 1 );   // (a pinned team does not ask the pool)
        std::vector<std::thread> team;
        team.reserve( nT );
        blockClaim = &claim;  // (before the first helper starts: every thread of the team must see the claim)
        size_t started = 1;   // threads started (this one included); thread th takes depT[th]
        auto start = [&]() -> bool
        {
            try { team.emplace_back( helper, started ); }
            catch (...) { return false; }   // (no thread: the team stays as it is)
            ++started;
            return true;
        };
        if ( teamGrows )
        {
            for (int k = CorePool::take( int(nT) - 1 ); k > 0; --k)
                if ( !start() ) { CorePool::give( k ); break; }
            claim.recruit = [&]()   // (not for the last couple of blocks: a helper would find none)
            {
                if ( started < nT && claim.next.load( std::memory_order_relaxed ) + 2 < claim.blockStart.size()
                     && CorePool::take( 1 ) == 1 && !start() ) CorePool::give( 1 );
            };
        }
        else
            while ( started < nT && start() ) {}
        if ( not userOptions.psSuppressGridStats or userOptions.verboseLevel >= 3 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " CPU deposit on " << started
                    << ( started == 1 ? " thread" : " threads" ) << " (copy mode: 0 MB of private accumulators per extra "
                    << "thread, budget " << psDepositThreadBytes() / 1.e9 << " GB; the fixed-point grids, counts and flags "
                    << "are shared), a dynamic team: blocks of " << DEP_BLOCK << " cells claimed from one counter"
                    << ( teamGrows ? ", the team grown to at most " : " by the " ) << nT
                    << ( teamGrows ? " threads as cores come free.\n" : " pinned threads.\n" ) << MESSAGE::Flush;
        run( 0 );
        claim.recruit = nullptr;
        for (std::thread &x : team) x.join();
        blockClaim = nullptr;
        nDepThreads = int( started );   // (the threads it grew to: depT beyond them is untouched)
        for (std::exception_ptr const &e : errors)
            if ( e ) std::rethrow_exception( e );
    }
    else if ( !metalDeposited || slabMode )   // (after a GPU deposit, slab mode means its deferred cells)
    {
        size_t const nT = size_t(nDepThreads);
        std::vector<std::exception_ptr> errors( nT );
        std::atomic<int> nextSlab( 0 );
        auto run = [&](size_t th)
        {
            try
            {
                if ( slabMode )      // whole slabs, picked up as threads come free
                    for (int sl = nextSlab++; sl < nSlabs; sl = nextSlab++)
                        depositLoop( depT[th], th, nT, false, sl );
                else
                    depositLoop( depT[th], th, nT, false, -1 );
            }
            catch (...) { errors[th] = std::current_exception(); }
        };
        if ( nT <= 1 )
            run( 0 );
        else
        {   // the static rule's team: its cores counted in the pool for its lifetime (thread_count.h CorePool::Hold)
            CorePool::Hold const hold( int(nT) - 1 );
            std::vector<std::thread> pool;
            for (size_t th = 1; th < nT; ++th) pool.emplace_back( run, th );
            run( 0 );
            for (std::thread &x : pool) x.join();
        }
        for (std::exception_ptr const &e : errors)
            if ( e ) std::rethrow_exception( e );
        // sum the private copies into the real arrays in thread order, cell ranges in parallel (slab mode has none:
        // every thread wrote its own cells; nor has a copy mode whose accumulators are all shared, accBytes == 0)
        if ( nT > 1 && !slabMode && accBytes > 0. )
        {
            auto merge = [&](size_t part)
            {
                size_t const lo = totalGrid * part / nT, hi = totalGrid * (part + 1) / nT;
                auto addInto = [lo, hi](auto &real, auto const &own) { if (!own.empty()) for (size_t c = lo; c < hi; ++c) real[c] += own[c]; };
                for (size_t th = 1; th < nT; ++th)
                {
                    DepositThread const &D = depT[th];
                    addInto( quantities->density, D.ownQ.density );
                    addInto( quantities->velocity, D.ownQ.velocity );
                    addInto( quantities->velocity_gradient, D.ownQ.velocity_gradient );
                    addInto( quantities->velocity_dispersion, D.ownQ.velocity_dispersion );
#ifdef SCALAR
                    addInto( quantities->scalar, D.ownQ.scalar );
                    addInto( quantities->scalar_gradient, D.ownQ.scalar_gradient );
#endif
                    addInto( releasedMult, D.ownReleasedMult ); addInto( exactMult, D.ownExactMult );
                    addInto( dispWeight, D.ownDispWeight );     addInto( dispVel, D.ownDispVel );
                    addInto( massWeight, D.ownMassWeight );
                }
            };
            CorePool::Hold const hold( int(nT) - 1 );
            std::vector<std::thread> pool;
            for (size_t part = 1; part < nT; ++part) pool.emplace_back( merge, part );
            merge( 0 );
            for (std::thread &x : pool) x.join();
        }
    }
    for (DepositThread const &D : depT)
    {
        nDegenerateInverse += D.nDegenerateInverse;
        nReleased += D.nReleased;
        audit.cpuTets += D.auditTets; audit.cpuShares += D.auditShares; audit.nCpuTets += D.auditNTets;
    }
    lap( metalDeposited ? PS_PH_DEFERRED : PS_PH_CPU );
    if ( massAudit && field.density ) audit.cpuGain = gridSum() - auditGridBefore;
    std::vector<DepositThread>().swap( depT );   // the private copies
    std::vector<Cell_handle>().swap( slabTets );
    std::vector< std::vector<uint32_t> >().swap( slabTetLists );
    std::vector<uint8_t>().swap( slabFirst );
    if ( metalDeposited && !deferredOrd.empty() && not userOptions.psSuppressGridStats )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " deposited the "
                << MESSAGE::cMagenta() << deferredOrd.size() << MESSAGE::cReset() << " GPU-deferred tetrahedra on the CPU in "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - tCpuLoop0).count() << " s ("
                << nDepThreads << ( nDepThreads == 1 ? " thread" : " threads" ) << ").\n" << MESSAGE::Flush;

#if NO_DIM==3
    // --face-gradient beside the GPU deposit: the face sums of every kept tetrahedron from one CPU walk -- here,
    // after the deposit, or (the default: fx.faceWalked) already made while the device call ran
    if ( metalDeposited && fx.face && !fx.faceS.w.empty() )
    {
        double faceSec = fx.faceWalkSec;
        if ( !fx.faceWalked )
        {
            auto const tFace0 = std::chrono::steady_clock::now();
            psFaceWalk( passCtx, fx, psVertexMass, psDepositThreads(), nullptr );
            lap( PS_PH_CPU );
            faceSec = std::chrono::duration<double>( std::chrono::steady_clock::now() - tFace0 ).count();
        }
        if ( not userOptions.psSuppressGridStats )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                    << " face sums (--face-gradient) of the GPU-deposited tetrahedra on the CPU in "
                    << faceSec << " s" << ( fx.faceWalked ? " (beside the GPU deposit)" : "" ) << ".\n"
                    << MESSAGE::Flush;
    }
#endif

    // Whichever deposit ran, its last read of dt is behind us: for the final call on an internally-owned dt, free the
    // triangulation NOW (during the normalization and the caller's merge) and hand the allocator's kept blocks back to the system
    // (dtfeReleaseTessellationMemory).
    if ( mayClearDT )
    {
        dt.clear();
        dtfeReleaseTessellationMemory();
    }

    psFixedToReal( passCtx, gridsCtx, fx );

    psPassMessages( passCtx, nDegenerateInverse, nReleased );

    // turn the summed moments into means: the velocity family's sum(V_s f_s)/sum(V_s) by the Eulerian-volume
    // shares, the dispersion's against its own mass normalizer. Serial: here, BEFORE the density grid is
    // converted to rho/rho_bar. deferNorm: hand un-normalized moments to the
    // caller, which sums across partitions and normalizes once (the only order correct for cells
    // spanning partitions); the weight travels as mass_weight.
    Real const invSamples = Real(1.) / Real(nSamplesPerCell);
    psNormalizeMoments( passCtx, gridsCtx );

    // The density grid holds MASS (the per-tet shares): divide by the cell volume and by averageDensity, so the written field is
    // rho/rho_bar (as standard DTFE). The accumulators differ in exactness: the single build's fixed-point grid is rounded once
    // per cell to float, the double build's sums are good to double rounding, a float grid (a mass without a unit) loses shares
    // below half an ulp. Tetrahedra the cell filters drop take their mass along in every build.
    double const auditGridPre = ( massAudit && field.density ) ? gridSum() : -1.;
    if ( field.density )
    {
        Real const invRhoBar = userOptions.averageDensity > Real(0.)
                             ? Real(1.) / userOptions.averageDensity : Real(1.);
        psCellRanges( totalGrid, [&](size_t const lo, size_t const hi)   // (cell by cell: over the free cores)
        {
            for (size_t i = lo; i < hi; ++i)
                quantities->density[i] *= invRhoBar / cellVolume;
        } );
    }
    psMassAuditLine( passCtx, massAudit, audit, gridSum, cellVolume, auditGridPre );

    psStreamStats( passCtx, gridsCtx, invSamples, nSamplesPerCell );

    psExportStreams( passCtx, gridsCtx, invSamples );

    // Record the sub-grid box so the caller (addFromSubgrid) can map results back into the full shared grid.
    for (int d = 0; d < NO_DIM; ++d)
    {
        quantities->ps_subOrigin[d] = subOrigin[d];
        quantities->ps_subDims[d]   = subDims[d];
    }

    // this pass's phase split and the run's totals so far (psPhaseTotalUs)
    lap( PS_PH_NORM );
    psPhaseReport( passCtx, phaseSec, metalDeposited, deferredOrd );
}

#endif // PHASE_SPACE
