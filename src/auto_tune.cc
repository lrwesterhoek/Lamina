/* The auto-tuner's model (auto_tune.h is its interface and holds the machine queries): the memory model of a
   PS-DTFE and of a standard-DTFE run, the planners that choose --partition / --max-concurrent from it, and the
   '--serve' split. Only autoTunePartitioning, autoTuneServe and stdLookAheadBytes are seen outside this file. */

#include "auto_tune.h"

#include <cstdint>     // UINT32_MAX: the GPU's sub-grid cell cap
#include <cstdlib>
#include <cmath>
#include <limits>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <fstream>
#include <thread>      // hardware_concurrency: the point-evaluation HitSink blocks, one per thread
#include <dirent.h>    // autoTuneServe: the cached partition splits of a snapshot
#include <map>
#include <functional>  // the standard-DTFE split replay (stdTuneSplit)
#include <sstream>
#include <optional>    // the plan phases that may decide the split (AutoTunePSPlanner::choose)
#include "cache_key_lines.h"
#include "ps_point_eval.h"   // psEstimateMeanStreams: the sample-point stream count, from the data
#include "message.h"
// for PSCausticClass::CausticMask, so the --ps-caustics per-cell accumulator is budgeted at its
// real width rather than a literal that would drift the next time a bit is added
#include "CGAL_triangulation/ps_caustic_class.h"

#ifndef PHASE_SPACE
// the run's own split rules, defined in subpartition.h (compiled into DTFE.cpp, the only caller of the tuner)
void optimalSplit(std::vector<size_t> &counts, size_t const noPartitions, size_t *partitionIndices);
void parallelGrid(size_t const noAvailableProcessors, User_options const &userOptions, size_t *grid);
#endif

namespace {

// ---- The model's constants: one name and one source each ---------------------------------------------------------
// a finished PS-DTFE tessellation, per vertex: the 112 B vertex (with the uint64 ParticleID and the double volume
// accumulator) + 6.77 cells x 72 B (CGAL's cell) + allocator slack; a compiled sizeof probe and measured runs
constexpr double kPSVertexBytes = 666.;
// ... and a standard-DTFE one (no ParticleID; stdBytesPerPaddedVertex adds the double build's wider vertex data)
constexpr double kStdVertexBytes = 648.;
// '--parallel-triangulation' (TBB): the sorted point+payload copy and CGAL's own copy of it, per vertex in flight
// (203 B measured on PS-DTFE)
constexpr double kParInsertBytes = 210.;
// Delaunay tetrahedra per vertex of a 3D tessellation (the Poisson mean, which the measured runs follow)
constexpr double kTetsPerVertex = 6.77;
// the PS GPU deposit's flat arrays per tetrahedron: the 16-float record (64 B), the mass and the finite-cell ordinal
// (4 B each)
constexpr double kGpuTetBytes = 72.;
// ... + the tetrahedron's vertex velocities, extracted only when a velocity-derived grid is selected
constexpr double kGpuTetVelBytes = 48.;
// ... and the allocator's slack on those arrays
constexpr double kGpuTetSlack = 1.3;
// the standard GPU deposit's tetrahedra per particle of a thread's own box: x this for the ones crossing its faces
constexpr double kStdFaceTets = 1.3;
// standard DTFE: the share of the busiest partition's bytes still in the footprint once the partition loop is over and
// the derived grids are allocated (stdPeak). Fitted on a replay of the 28 logged standard runs (plan P8.9 step 1:
// lowest -1%, mean +13%; std100_099 +43% -> +14%; there 0.5 GB of an 11.8 GB partition stayed) ...
constexpr double kStdPostKeep = 0.35;
// ... but a small partition stays whole: up to this many bytes are booked in full (tests/dtfe_auto_tune_check.sh E,
// 128^3: all of a 0.42 GB partition stayed, the derived grids came on top). Both refitted in phase R from the
// SUMMARY's 'before post-proc' line
constexpr double kStdPostKeepFloor = 1.e9;
// a split for speed must predict less than this share of the best wall time so far: a clear gain only
constexpr double kSpeedSplitGain = 0.85;
// the '--sample-points' HitSink block one point-walk thread fills before handing it over (ps_point_eval.cc's size)
constexpr double kHitSinkBlockBytes = double(16 << 20);
// below this many particles a split is chosen for speed only (DTFE_AUTO_MINN overrides; see the header comment)
constexpr double kAutoMinN = 2.e6;
// macOS's allocator returns a freed block of this size (4 GiB) or more to the system; a smaller one stays in the
// footprint and serves later requests of at least half its size
constexpr double kAllocKeepBytes = 4294967296.;
// the GPU kernels number a partition's sub-grid cells with 32 bits: a larger sub-grid deposits on the CPU
constexpr double kGpuCellCap = double(UINT32_MAX);
// the Eulerian padding rule's floor, in partition widths (DTFE.cpp's PSPartitionPlan; autoTunePadWidths)
constexpr double kPartitionPadFloor = 0.3;

// Full-grid bytes per cell for one field selection, for ONE of the two phases:
//
//   phase 0 = DEPOSIT. What is resident while the partition triangulations run, i.e. the term
//             that coexists with partitionBytes(). Includes the PS weight grids.
//   phase 1 = POST. What is resident afterwards, when the derived fields are computed. The
//             weights have been freed by normalizePhaseSpace (swap-with-empty, quantities.cc)
//             in psNormalizeMoments (DTFE.cpp) BEFORE postProcessWebFields (web_fields.cc) allocates
//             div/shear/vort.
//
// The two peaks are disjoint sets, so the caller takes max(deposit, post) rather than summing.
// A single blended number would bound the deposit only by luck: for a density+velocity+
// dispersion volume-weighted run it gives 44 B/cell where the deposit really needs 64.
//
// 'ps' carries the PS flags: they gate real full-grid accumulators and are all finalized before
// the DTFE.cpp call site. Byte counts verified against quantities.cc / ps_interpolation.cc with
// a compiled sizeof probe (Real=4, Pvector<Real,n> = 4n exactly, no padding).
// Number of query points a '--sample-points' file holds, WITHOUT reading it (auto-tune runs
// long before psPointEvalInit). Raw binary is N x 3 float64 = 24 B/point; a text file is
// probed the same way readSamplePoints does and estimated from its average line length.
// Returns 0 when the file is absent/unreadable, which switches the model off.
inline size_t autoTuneSamplePointCount(std::string const &file)
{
    if ( file.empty() ) return 0;
    struct stat st;
    if ( stat( file.c_str(), &st ) != 0 || st.st_size <= 0 ) return 0;
    double const bytes = double(st.st_size);

    std::ifstream f( file.c_str(), std::ios::binary );
    if ( not f ) return 0;
    char head[4096];
    f.read( head, sizeof(head) );
    std::streamsize const got = f.gcount();
    bool isText = true;
    for (std::streamsize i = 0; i < got; ++i)
    {
        char const c = head[i];
        bool const numeric = (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.'
                             || c == 'e' || c == 'E' || c == ' ' || c == '\t'
                             || c == '\n' || c == '\r';
        if ( not numeric ) { isText = false; break; }
    }
    if ( not isText )
        return size_t( bytes / (3. * sizeof(double)) );

    // text: count the newlines in the probe to get an average bytes-per-triplet
    long lines = 0;
    for (std::streamsize i = 0; i < got; ++i) if ( head[i] == '\n' ) ++lines;
    double const perLine = lines > 0 ? double(got) / double(lines) : 30.;
    return size_t( bytes / perLine );
}

// The Eulerian bounding box of the '--sample-points', for the stream estimate on the points' own slab
// (psEstimateMeanStreams' region): binary files are sampled at up to 65536 evenly strided points (a plane's bounds
// to a pixel), text files read whole. Periodic: wrapped into the box first, as the evaluation does. False when the
// file cannot be read.
inline bool autoTuneSamplePointBounds(std::string const &file, double const boxLo[NO_DIM], double const boxLen[NO_DIM],
                                      bool const periodic, double lo[NO_DIM], double hi[NO_DIM])
{
    size_t const n = autoTuneSamplePointCount( file );
    if ( n == 0 ) return false;
    std::ifstream f( file.c_str(), std::ios::binary );
    if ( not f ) return false;
    for (int d=0; d<NO_DIM; ++d) { lo[d] = std::numeric_limits<double>::max(); hi[d] = -std::numeric_limits<double>::max(); }
    auto add = [&](double const *v) {
        for (int d=0; d<NO_DIM; ++d)
        {
            double x = v[d];
            if ( periodic and boxLen[d] > 0. ) x -= boxLen[d] * std::floor( (x - boxLo[d]) / boxLen[d] );
            lo[d] = std::min( lo[d], x );
            hi[d] = std::max( hi[d], x );
        }
    };
    char head[4096];
    f.read( head, sizeof(head) );
    std::streamsize const got = f.gcount();
    bool isText = true;
    for (std::streamsize i = 0; i < got and isText; ++i)
    {
        char const c = head[i];
        isText = (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E' || c == ' '
                 || c == '\t' || c == '\n' || c == '\r';
    }
    f.clear();
    size_t seen = 0;
    double v[NO_DIM];
    if ( isText )
    {
        f.seekg( 0, std::ios::beg );
        while ( f >> v[0] )
        {
            bool full = true;
            for (int d=1; d<NO_DIM and full; ++d) full = bool( f >> v[d] );
            if ( not full ) break;
            add( v );
            ++seen;
        }
    }
    else
    {
        size_t const step = std::max( size_t(1), n / 65536 );
        for (size_t i = 0; i < n; i += step)
        {
            f.seekg( std::streamoff( i * NO_DIM * sizeof(double) ), std::ios::beg );
            if ( not f.read( reinterpret_cast<char*>(v), sizeof(v) ) ) break;
            add( v );
            ++seen;
        }
        f.clear();   // ... and the last point, so a strided sample cannot miss the far edge
        f.seekg( std::streamoff( (n - 1) * NO_DIM * sizeof(double) ), std::ios::beg );
        if ( f.read( reinterpret_cast<char*>(v), sizeof(v) ) ) add( v );
    }
    if ( seen == 0 ) return false;
    for (int d=0; d<NO_DIM; ++d) if ( not std::isfinite( lo[d] ) or not std::isfinite( hi[d] ) ) return false;
    return true;
}


struct AutoTunePS
{
    bool phaseSpace   = false;
    bool volumeWeighted = false;   // --ps-volume-weighted
    bool exactDeposit = false;     // --exact-average
    bool caustics     = false;     // --ps-caustics
    bool haloRelease  = false;     // --ps-halo-release (sampled deposit: the released-tet multiplicity grid)
    bool fixedMerge   = false;     // --ps-merge fixed-point: the summed full grids are 9 B integer words + carries
    bool faceGradient = false;     // --face-gradient (docs/cleanup-plan.md P10): the face sums (both binaries)
};

// The PS deposit's MOMENT grids as 64-bit fixed-point sums (ps_deposit_params.h PSFxGrid): on by
// default on the GPU (both builds) and on the CPU of the single build; DTFE_GPU_FLOAT_MOMENTS=1 selects the float
// sums (ps_interpolation.cc fxGpuWanted / fxMomWanted). Read here so the model follows an A/B run too.
inline bool autoTunePSMomentsFixed()
{
    return not Env::flag( { "DTFE_GPU_FLOAT_MOMENTS" } );
}

// One pass's MOMENT accumulators in the PS deposit, in Real bytes per sub-grid cell: the grids
// ps_interpolation.cc holds as fixed-point words while the pass deposits ('fxg' -- velocity, gradient,
// dispersion, scalar and its gradient, the moment weight, the dispersion's own mean and weight, the exact and the
// released multiplicities), with the same selection rules as autoTuneBytesPerCell's deposit phase below. In the
// single build the words take 8 B per component where these floats take 4, and the deposit threads share ONE set
// of them (no private copies). The dispersion weight is the density there when one is selected (not accumulated).
inline double autoTuneFxMomentBytes(Field &f, AutoTunePS const &ps)
{
    if ( not f.selected() or not ps.phaseSpace ) return 0.;
    double const R = double(sizeof(Real));
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    bool const densityGrid = f.density or f.velocity_tweb;
    bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                            or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                            or f.velocity_vweb or f.scalar or f.scalar_gradient;
    double b = 0.;
    if ( f.velocity or (f.velocity_dispersion and not ps.volumeWeighted) ) b += 3.*R;
    if ( gradient ) b += 9.*R;
    if ( f.velocity_dispersion ) b += 6.*R;
    if ( f.scalar ) b += noScalarComp*R;
    if ( f.scalar_gradient ) b += noScalarGradComp*R;
    if ( needWeight and not (densityGrid and not ps.volumeWeighted) ) b += R;                 // mass_weight
    if ( ps.volumeWeighted and f.velocity_dispersion ) b += 3.*R + (densityGrid ? 0. : R);   // disp_velocity (+ disp_weight)
    if ( ps.exactDeposit ) b += R;                                                           // the exact multiplicity
    if ( ps.haloRelease and not ps.exactDeposit ) b += R;                                    // the released multiplicity
    // --face-gradient: the face sums' int64 words (ps_interpolation.cc fx.faceS / faceN: 2 x NO_DIM x (3 + 1) per cell,
    // in EVERY build), the same "as many again" as the Real grids they become
    if ( ps.faceGradient and ps.exactDeposit and ps.volumeWeighted and gradient ) b += 2.*NO_DIM*(noVelComp + 2.)*R;
    return b;
}

// The same grids ONE BY ONE, in the order the deposit converts them after its loop (ps_deposit_params.h PSFxGrid:
// fxToReal walks g = 0, 1, ...), as Real bytes per sub-grid cell each (their words take twice that, and
// the entries add up to autoTuneFxMomentBytes): the one-domain model below follows that conversion grid by grid.
inline std::vector<double> autoTuneFxGrids(Field &f, AutoTunePS const &ps)
{
    std::vector<double> g;
    if ( not f.selected() or not ps.phaseSpace ) return g;
    double const R = double(sizeof(Real));
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    bool const densityGrid = f.density or f.velocity_tweb;
    bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                            or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                            or f.velocity_vweb or f.scalar or f.scalar_gradient;
    if ( f.velocity or (f.velocity_dispersion and not ps.volumeWeighted) ) g.push_back( 3.*R );   // PS_FX_MOM
    if ( f.velocity_dispersion ) g.push_back( 6.*R );                                         // PS_FX_M2
    if ( gradient ) g.push_back( 9.*R );                                                      // PS_FX_GRAD
    if ( needWeight and not (densityGrid and not ps.volumeWeighted) ) g.push_back( R );       // PS_FX_MOMW
    if ( ps.exactDeposit ) g.push_back( R );                                                  // PS_FX_SV
    if ( ps.volumeWeighted and f.velocity_dispersion )
    {
        g.push_back( 3.*R );                                                                  // PS_FX_DV
        if ( not densityGrid ) g.push_back( R );                                              // PS_FX_DW
    }
    if ( f.scalar ) g.push_back( noScalarComp*R );                                            // PS_FX_SCAL
    if ( f.scalar_gradient ) g.push_back( noScalarGradComp*R );                               // PS_FX_SG
    if ( ps.haloRelease and not ps.exactDeposit ) g.push_back( R );                           // PS_FX_RM
    // --face-gradient: the face sums' words, converted after the moment grids (ps_interpolation.cc fx.faceS, then
    // fx.faceN; autoTuneFxMomentBytes counts the same)
    if ( ps.faceGradient and ps.exactDeposit and ps.volumeWeighted and gradient )
    {
        g.push_back( 2.*NO_DIM*noVelComp*R );
        g.push_back( 2.*NO_DIM*R );
        g.push_back( 2.*NO_DIM*R );   // fx.faceP (the mass flux)
    }
    return g;
}

// What the input reader frees and macOS's allocator keeps, in bytes: the reader's flat arrays (gadget_reader_HDF5.cc:
// the positions, masses and velocities 'readParticleData' asks for, the scalars, the Lagrangian positions and the
// IDs), freed one by one as Read_data::transferData builds the particles, and with '--ps-lagrangian-input' the IC file's
// coordinates and IDs and the by-ID table (input_output.cc; a sort index of 8 B instead of the table's 12 when the
// IDs are not contiguous). libmalloc keeps a freed block under 4 GiB in the footprint and serves only a request of
// about half its size or more from it, so these stay until something that large is allocated. A run with full-size
// grids allocates them right away; a --grid-window run does not -- its window grids, the partitions' particle copies,
// the tessellations (TBB's own pools) and the GPU's buffers are all far smaller or elsewhere -- so the whole residue
// is still there when the first wave of partitions tessellates: TNG100-3-Dark z=0 (94M particles, positions from a
// '--ps-lagrangian-input' file) reads to 13.1 GB, 57 B/particle live and 80 kept, and peaks with the first six
// tessellations on top (19.6 GB measured at '--partition 6 6 6 --max-concurrent 6'). Later partitions' large
// allocations take the blocks over, so the residue is gone a few partitions in.
inline double autoTuneReadKeptBytes(User_options const &u, double const N)
{
    double const B4 = kAllocKeepBytes;   // a freed block of this size or more goes back to the system
    double const R = double(sizeof(Real));
    auto kept = [&](double const perParticle) { return perParticle*N < B4 ? perParticle*N : 0.; };
    auto reads = [&](size_t const i) { return u.readParticleData.size() > i and u.readParticleData[i] != 0; };
    double b = 0.;
    if ( reads(0) ) b += kept( NO_DIM*R );                    // positions
    if ( reads(1) ) b += kept( R );                           // masses
#ifdef VELOCITY
    if ( reads(2) ) b += kept( noVelComp*R );                 // velocities
#endif
#ifdef SCALAR
    for (size_t i = 3; i < u.readParticleData.size(); ++i)
        if ( reads(i) ) { b += kept( R ); break; }            // the scalar
#endif
#ifdef PHASE_SPACE
    b += kept( NO_DIM*R ) + kept( 8. );                       // the Lagrangian positions, the IDs
    if ( not u.lagrangianInputFilename.empty() )
        b += kept( NO_DIM*4. ) + kept( 8. ) + kept( NO_DIM*4. );   // the IC file's coordinates and IDs, the by-ID table
#endif
    return b;
}

// One Quantities' PS COUNT, FLAG and MASK grids in bytes per sub-grid cell (part of autoTuneBytesPerCell): none of
// them is a private copy of the deposit's copy mode -- the deposit's own int stream count, its two byte flags and
// its caustic mask are shared by the threads (ps_interpolation.cc psAddCount), and the others are
// DTFE.cpp's output grids -- so the partition model's copy term leaves them out.
inline double autoTunePSCountBytes(Field &f, AutoTunePS const &ps, int const phase)
{
    if ( not f.selected() or not ps.phaseSpace ) return 0.;
    double const R = double(sizeof(Real));
    double b = 0.;
    b += R;                                                    // stream_count (both phases)
    b += R;                                                    // hidden_streams (both phases, DTFE.cpp)
    if ( phase == 0 )      b += 2.;                            // subSampleMassFlag + flippedFlag, the deposit's byte accumulators
    if ( ps.caustics )     b += R;                             // caustic_bits (psReserveGrids, DTFE.cpp)
    if ( ps.exactDeposit ) b += R;                             // tet_touch    (psReserveGrids, DTFE.cpp)
    /* orientBits, the deposit's per-cell accumulator (ps_interpolation.cc): a CausticMask,
       not a byte, since the A4 indicator needs bit 8. Deposit phase only -- freed before the
       derived fields are allocated. Counted once per selected Quantities, like caustic_bits;
       when only one of the u/a pair is deposited that over-counts slightly, deliberately,
       since for a memory budget the safe rounding direction is up. */
    if ( ps.caustics and phase == 0 ) b += double(sizeof(PSCausticClass::CausticMask));
    return b;
}

// The deposit-phase full grids the first merge sizes (Quantities SIZE_AT_MERGE: the moment weights, the face sums), in
// bytes per cell -- the partition loop's dynamic concurrency reserves them until then (autoTuneBytesPerCell counts
// them in the deposit phase; the predicates are its)
inline double autoTuneMergeGrownBytes(Field &f, AutoTunePS const &ps)
{
    if ( not f.selected() or not ps.phaseSpace ) return 0.;
    double const R = double(sizeof(Real));
    bool const densityGrid = f.density or f.velocity_tweb;
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                            or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                            or f.velocity_vweb or f.scalar or f.scalar_gradient;
    double b = 0.;
    if ( needWeight and not (densityGrid and not ps.volumeWeighted) ) b += R;   // mass_weight
    if ( ps.volumeWeighted and f.velocity_dispersion ) b += R + 3.*R;           // disp_weight + disp_velocity
    if ( ps.faceGradient and gradient and ps.exactDeposit and ps.volumeWeighted )
        b += 2.*NO_DIM*(noVelComp + 2.)*R;                                        // face_velocity, _coverage, _momentum
    if ( ps.fixedMerge ) b = b / R * 9.;                                          // int64 word + int8 carry each
    return b;
}

// The grids only the POST phase allocates (autoTuneBytesPerCell's phase 1, below), in bytes per cell: the derived
// fields and the T-web tidal stage, both made after the deposit loop by postProcessWebFields (web_fields.cc).
inline double autoTuneDerivedBytes(Field &f)
{
    if ( not f.selected() ) return 0.;
    double const R = double(sizeof(Real));
    double b = 0.;
    if ( f.velocity_divergence ) b += R;
    if ( f.velocity_shear ) b += 5.*R;
    if ( f.velocity_vorticity ) b += 3.*R;
    if ( f.velocity_tweb ) b += R + 3.*R;                      // web label + eigenvalues
    if ( f.velocity_vweb ) b += R + 3.*R;
    // T-web tidal stage (computeTidalWebClassification, web_fields.cc): all live at
    // once -- work(N) 8 B + deltaK/tensK fftw_complex ~16.03 B + tens[0..5] 6*R. 48 B/cell
    // on top of every resident grid; invisible at the 256-512 sizes this was calibrated on
    // (6.5 GB at 512^3) but 51.6 GB at 1024^3.
    if ( f.velocity_tweb ) b += 8. + 16.032 + 6.*R;
    return b;
}

inline double autoTuneBytesPerCell(Field &f, AutoTunePS const &ps, int const phase)
{
    if ( not f.selected() ) return 0.;
    double const R = double(sizeof(Real));
    double b = 0.;
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    // 'density||tweb' mirrors the tweb->density folding in DTFE_setup (DTFE.cpp), which happens AFTER
    // autoTunePartitioning is called there -- so use this predicate anywhere the deposit sees
    // field.density, not the raw f.density.
    bool const densityGrid = f.density or f.velocity_tweb;
    if ( densityGrid ) b += R;
    // velocityStd_a: sized before the loop (reserveMemory) and filled by the deposit (averaged_interpolation_2.cc;
    // deselectVelocityDerivatives keeps it), so both phases hold it
    if ( f.velocity_std ) b += R;
    // dispersion allocates the velocity grid too -- EXCEPT volume-weighted, where it carries
    // its own disp_velocity mean instead (counted in the deposit-phase PS block below)
    if ( f.velocity or (f.velocity_dispersion and not (ps.phaseSpace and ps.volumeWeighted)) )
        b += 3.*R;
    if ( gradient ) b += 9.*R;
    if ( f.velocity_dispersion ) b += 6.*R;
    if ( f.scalar ) b += noScalarComp*R;
    if ( f.scalar_gradient ) b += noScalarGradComp*R;
    // --face-gradient (docs/cleanup-plan.md P10), the deposit phase: PS-DTFE's merged face sums (Quantities::face_velocity
    // + face_coverage, 2 x NO_DIM x (3 + 1) Real per cell, freed in the normalization) and the cell mean velocity the
    // mean-flow gradient needs, written or not. (Standard DTFE's face pass holds 9 doubles per cell of the sub-domain it
    // interpolates, not of the full grid: the standard planner books it per sub-domain, AutoTuneStdInputs::faceCell.)
    if ( ps.faceGradient and gradient and phase == 0 and ps.phaseSpace and ps.exactDeposit and ps.volumeWeighted )
        b += 2.*NO_DIM*(noVelComp + 2.)*R + ( f.velocity ? 0. : 3.*R ) + R;   // (+ the mass flux's output grid)

    if ( phase == 1 )
        b += autoTuneDerivedBytes( f );   // derived fields: allocated only after the deposit loop (above)
    else if ( ps.phaseSpace )
    {
        // ---- deposit-phase PS weight grids (freed in psNormalizeMoments, before the derived fields) ----
        bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                                or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                                or f.velocity_vweb or f.scalar or f.scalar_gradient;
        // ps_interpolation.cc: weightIsDensity = needWeight && field.density && !psVolWeighted.
        // --ps-volume-weighted therefore BREAKS the aliasing: the weight carries Eulerian-volume
        // shares, which are not the density's mass sums, so it needs its own grid even when
        // density is selected. That is the production default, not a corner case.
        if ( needWeight and not (densityGrid and not ps.volumeWeighted) ) b += R;   // mass_weight
        // dispersion stays MASS-weighted under --ps-volume-weighted, so it carries its OWN
        // mean+normalizer (ps_interpolation.cc: dispNeedsOwnWeight)
        if ( ps.volumeWeighted and f.velocity_dispersion ) b += R + 3.*R;           // disp_weight + disp_velocity
    }

    // '--ps-merge fixed-point' (ps_fixed_merge.h): during the loop every SUMMED full-grid component (counted
    // above at R: density, velocity, gradient, dispersion, scalar(s), mass_weight, disp_weight, disp_velocity; below:
    // stream_count, tet_touch) is an int64 word + an int8 carry, 9 B, and its Real grid does not exist; the masks
    // (hidden_streams, caustic_bits) stay Real. After the loop the grids convert one at a time, the largest one's Real
    // grid briefly beside the integers. Nothing is staged or spilled in this mode (the model counts neither).
    if ( ps.phaseSpace and ps.fixedMerge and phase == 0 )
    {
        double const counts = 1. + ( ps.exactDeposit ? 1. : 0. );   // stream_count, tet_touch: R each below, + 9 - R here
        double const largest = gradient ? 9. : f.velocity_dispersion ? 6. : ( f.velocity or f.velocity_dispersion ) ? 3. : 1.;
        b = b / R * 9. + counts * ( 9. - R ) + largest * R;
    }
    if ( ps.phaseSpace )
    {
        b += autoTunePSCountBytes(f, ps, phase);                   // the counts, flags and masks (above)
        // --ps-halo-release on the sampled deposit: released tets add their volume fraction to
        // a per-cell Real accumulator (releasedMult, ps_interpolation.cc). Deposit phase only.
        if ( ps.haloRelease and not ps.exactDeposit and phase == 0 ) b += R;
    }
    return b;
}

/* A Lagrangian partition's padding on one axis, in units of the partition's own width, for an n^3 split of N
   particles: with '--ps-lagrangian-padding <k>' (the default, k = 8) k mean spacings -- k n /
   N^(1/NO_DIM) partition widths, at most one period (n widths) on a periodic run -- and at least 4.2
   '--ps-alpha-shape' cuts on a non-periodic one, with no floor; with '--ps-lagrangian-padding 0' the Eulerian '--padding'
   fraction floored at 'floorFrac' of a partition (DTFE.cpp's PSPartitionPlan). One definition for every model below. */
inline double autoTunePadWidths(User_options const &u, double const padFracAxis, int const n, double const N,
                                double const floorFrac)
{
#ifdef PHASE_SPACE
    if ( u.psLagPadding > Real(0.) and n > 1 and N > 0. )
    {
        double k = double(u.psLagPadding);
        if ( not u.periodic and u.psAlphaShape > Real(0.) ) k = std::max( k, 4.2 * double(u.psAlphaShape) );
        double const w = k * double(n) / std::pow( N, 1. / double(NO_DIM) );
        return u.periodic ? std::min( w, double(n) ) : w;     // DTFE.cpp caps a periodic padding at one period
    }
#endif
    return std::max( floorFrac, padFracAxis * double(n) );
}

/* The partition splits of THIS snapshot already in the tessellation cache folder: n -> the number of
   cached n^3 partition files. Reads only each file's uncompressed header (magic, length, descriptor) and
   compares the lines that name the snapshot and the build -- exactly the text tessellation_cache.h writes
   (cache_key_lines.h) -- plus the periodicity and length unit; the rest of a partition's key follows from
   the split. Read-only. */
inline std::map<int,int> cachedServeSplits(User_options const &u)
{
    std::map<int,int> out;
    if ( u.tessellationCacheDir.empty() ) return out;
    std::string const wantInput = cacheFileLine( "input", u.inputFilename ) + '\n';
#ifdef PHASE_SPACE
    std::string const wantLag = cacheFileLine( "lagrange", u.lagrangianInputFilename ) + '\n';
    // tessellation_cache.h's 'lagPadding=<k>' line, written by every partition of a k > 0 run
    std::string wantPad;
    if ( u.psLagPadding > Real(0.) )
    {
        std::ostringstream pad;
        pad << "\nlagPadding=" << u.psLagPadding << '\n';
        wantPad = pad.str();
    }
#endif
    std::string const wantBuild = cacheBuildLines();
    std::ostringstream other;
    other << "MpcUnit=" << u.MpcValue << '\n' << "periodic=" << (u.periodicInput ? 1 : 0) << '\n';
    std::string const wantOther = other.str();
    DIR *dir = ::opendir( u.tessellationCacheDir.c_str() );
    if ( dir == nullptr ) return out;
    while ( struct dirent *e = ::readdir( dir ) )
    {
        std::string const name = e->d_name;
        if ( name.size() < 11 or name.compare( 0, 5, "tess_" ) != 0 or name.compare( name.size() - 5, 5, ".tess" ) != 0 )
            continue;
        std::ifstream f( ( u.tessellationCacheDir + "/" + name ).c_str(), std::ios::binary );
        char magic[8];
        uint32_t len = 0;
        if ( not f.read( magic, 8 ) or std::string( magic, 8 ) != "DTFETESS" ) continue;
        if ( not f.read( reinterpret_cast<char*>(&len), 4 ) or len == 0 or len > (1u << 20) ) continue;
        std::string desc( len, '\0' );
        if ( not f.read( &desc[0], len ) ) continue;
        if ( desc.find( wantBuild ) == std::string::npos or desc.find( wantInput ) == std::string::npos
             or desc.find( wantOther ) == std::string::npos )
            continue;
#ifdef PHASE_SPACE
        if ( desc.find( wantLag ) == std::string::npos ) continue;
        // the partition padding: a split cached under another padding (the Eulerian rule writes no line)
        // would miss in the run, so it must not steer the split either
        if ( wantPad.empty() ? desc.find( "\nlagPadding=" ) != std::string::npos : desc.find( wantPad ) == std::string::npos )
            continue;
#endif
        static char const SPLIT[] = "\npartition=1 ";
        size_t const p = desc.find( SPLIT );
        if ( p == std::string::npos ) continue;
        int a = 0, b = 0, c = 0;
        if ( std::sscanf( desc.c_str() + p + sizeof(SPLIT) - 1, "%d,%d,%d", &a, &b, &c ) == 3 and a == b and ( NO_DIM == 2 or b == c ) and a > 1 )
            ++out[a];
    }
    ::closedir( dir );
    return out;
}

// One partition's padding factor in an n^3 split: the padded particles over the owned ones, the product over the axes
// of 1 + 2 x its padding in partition widths (autoTunePadWidths). One loop for every model below.
inline double autoTunePadFactor(User_options const &u, double const padFrac[NO_DIM], int const n, double const N,
                                double const floorFrac)
{
    double f = 1.;
    for (int d=0; d<NO_DIM; ++d)
        f *= 1. + 2.*autoTunePadWidths( u, padFrac[d], n, N, floorFrac );
    return f;
}

// the end of the AUTO-TUNE line: how to override the choice, or (both knobs given) what the line is for
inline char const *autoTuneOverrideNote(bool const autoPart, bool const autoConc)
{
    return ( autoPart or autoConc ) ? ". Pass --partition/--max-concurrent to override."
                                    : ". Both are given: this is the memory model's prediction for them (the run's "
                                      "memory limits come from it).";
}

}  // namespace

void autoTuneServe(User_options &u, size_t const nParticles)
{
    if ( u.partitionOn or u.partitionGiven or u.partNo >= 0 )
        return;
    double const N = double(nParticles);
    std::string budgetWhy;
    double const budget = autoTuneBudget( autoTunePhysicalRAM(), budgetWhy );
    double const partBytes = double(sizeof(Particle_data));
    // transient of the build: the particle copy that coexists with the growing tessellation, or
    // (TBB parallel insertion, PS-DTFE) the sorted point+payload copy plus CGAL's own copy of it
    // while the particles are already freed -- triangulation.cpp
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    double const kBuild = 330.;
#else
    double const kBuild = 160.;
#endif
#ifdef PHASE_SPACE
    double const kDT = kPSVertexBytes + kBuild;
    double const singlePadFloor = 0.10;             // PS: Lagrangian padding floor (DTFE.cpp)
#else
    double const kDT = kStdVertexBytes + kBuild;
    double const singlePadFloor = 0.;
#endif
    double padFrac = 0.;
    for (int d=0; d<NO_DIM; ++d)
    {
        double const boxLen = double(u.boxCoordinates[2*d+1] - u.boxCoordinates[2*d]);
        if ( boxLen > 0. )
            padFrac = std::max( padFrac, double(u.paddingLength[2*d] + u.paddingLength[2*d+1]) / (2.*boxLen) );
    }
    double padAxes[NO_DIM];                         // the widest axis's padding, on every axis
    for (int d=0; d<NO_DIM; ++d) padAxes[d] = padFrac;
    auto tessBytes = [&](int n) -> double
    {
        double const f = autoTunePadFactor( u, padAxes, n, N, (n == 1) ? singlePadFloor : kPartitionPadFloor );
        return N / std::pow( double(n), NO_DIM ) * f * kDT;
    };
    double const single = N*partBytes + tessBytes(1);
    if ( single <= budget )
        return;

    MESSAGE::Message message( u.verboseLevel );
    char buf[640];
    if ( u.tessellationCacheDir.empty() )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE (serve): one tessellation of these " << nParticles << " particles needs ~"
                << single/1.e9 << " GB, over the memory budget of " << budget/1.e9 << " GB (" << budgetWhy
                << "). Pass '--tessellation-cache DIR' (a local directory) and the server becomes a composite of "
                << "partition tessellations of which only a few are held in memory at once; continuing with "
                << "one tessellation.\n" << MESSAGE::EndWarning;
        return;
    }
    // the coarsest split that leaves room for two partitions next to the particles while they are built
    int nMin = 2;
    while ( nMin < 8 and 2.*tessBytes(nMin) > budget - N*partBytes ) ++nMin;
    /* Which split. A split this snapshot already has in the cache is reused -- the finest complete one, else
       the one with the most cached partitions -- since any other means building (and storing) every
       partition again (TNG100-3-Dark's 4^3 set: 469 s and 52 GB). With nothing cached, the split that
       holds ~8 partitions in memory at once, the number a zoom touches (TNG100-3-Dark z=0: a cold 512^2 zoom
       takes 19.7 s at 3^3, which holds 4, and 9.8 s at 4^3, which holds 10). */
    std::map<int,int> const cached = cachedServeSplits( u );
    int n = -1;
    char const *why = "";
    for (auto it = cached.rbegin(); it != cached.rend() and n < 0; ++it)
        if ( it->first >= nMin and it->second >= int( std::pow( double(it->first), NO_DIM ) ) )
        { n = it->first; why = "the finest split fully in the tessellation cache"; }
    if ( n < 0 )
    {
        int best = 0;
        for (auto const &kv : cached)
            if ( kv.first >= nMin and kv.second > best ) { best = kv.second; n = kv.first; why = "the split with the most cached partitions"; }
    }
    if ( n < 0 )
    {
        // once built the particles are gone, so the budget holds partitions only; stop where a finer split
        // no longer shrinks them (on a small set the padding shells dominate, and 8^3 would only cost)
        n = nMin;
        while ( n < 8 and 8.*tessBytes(n) > budget and tessBytes(n+1) < 0.75 * tessBytes(n) ) ++n;
        why = cached.empty() ? "nothing cached yet: about 8 partitions held in memory, as many as a zoom touches"
                             : "the cached splits are too coarse for the memory budget: about 8 partitions held in memory";
    }
    double const room = std::max( 0., budget - N*partBytes );
    int conc = int( std::floor( room / std::max( tessBytes(n), 1. ) ) );
    conc = std::max( 1, std::min( conc, autoTuneCores() ) );
    u.partition = std::vector<size_t>( NO_DIM, size_t(n) );
    u.partitionOn = true;
    if ( not u.maxConcurrentOn )
        u.maxConcurrent = conc;
    snprintf( buf, sizeof(buf),
              "AUTO-TUNE (serve): one tessellation would need ~%.1f GB, over the %.1f GB budget -> serving a "
              "composite of %d^3 partitions (%s; ~%.1f GB each, %d built at a time); the resident set is sized "
              "once they are built\n", single/1.e9, budget/1.e9, n, why, tessBytes(n)/1.e9,
              u.maxConcurrentOn ? u.maxConcurrent : conc );
    message << buf << MESSAGE::Flush;
}

#ifndef PHASE_SPACE
namespace {

/* ---- Standard DTFE: what one partition's thread sub-triangulations hold, from the data --------------------------
   The standard '--exact-average' run peaks 25-30% above the sampled one in footprint (TNG100-3-Dark z=1, 3^3,
   9 threads: 37.4 vs 30.0 GB), yet what the code allocates has no exact-average term:
     - CPU (averaged_interpolation_1.cc): the exact loop clips each tetrahedron against its cell window with r3d on the
       stack -- no per-tetrahedron or per-cell storage that the Monte-Carlo loop does not have;
     - GPU (dtfe_metal_host.cc / dtfe_gpu_cuda.cu): BOTH averages extract the same flat arrays of the tetrahedra that
       touch the thread's region (vertices 48 B, densities 16, velocities 48, volume, count, flat index, Morton key
       4 each), sort them in place, upload them into buffers of the same size, and deposit into the same per-call
       sub-grid (device + host copy). The exact kernel's only extra is its work-item table (4 B per tetrahedron and
       4 B per 32 items), which the sampled item kernel builds as well.
   What changes is WHEN memory is held. The exact kernel is several times slower, and the thread sub-domains of a
   partition queue on the ONE device (ctxMutex): every thread then waits with its triangulation and its uploaded
   batch alive, so all of a partition's sub-triangulations coexist -- the worst case this model assumes. In a
   sampled run the GPU drains a batch in about a second, the small sub-domains finish and free theirs before the
   largest one is built, and the peak stays below the worst case. So the model needs the worst case to be RIGHT,
   and the box-average particle count per sub-box, (N / P^3) x the padding factor, is not: the
   load-balanced split (optimalPartitionSplit) gives a dense partition a small VOLUME, while the padding is a fixed
   length (k mean spacings), so on clustered data the padding shells hold most of the vertices (TNG50 z=0: one
   partition's 4 threads triangulate 35.4M particles, 10.6M in one sub-box, where the box average gives 7.8M; the
   busiest partition holds more as structure grows).
   So the plan replays the run's own split on the particles: a coarse count grid of the box (or, without periodic
   wrap, of the box and its padding) with a summed-area table gives the particles in any box; the partition split
   (optimalSplit along x, then y per slab, then z per column -- optimalPartitionSplit's order, on bins of the run's
   grid cells), the thread grid of each partition (parallelGrid, as DTFE_parallel) and its split are replayed with
   the run's functions, and every padded thread box is counted, periodic images included. A partition costs:
     - per padded vertex (+6% for the padding test's dummy points): its particle copy, which coexists with the
       growing triangulation, + kDT (+ the TBB insert's copies with '--parallel-triangulation');
     - on the GPU, per tetrahedron of the thread's own box (6.77 per particle, x1.3 for the ones crossing its faces):
       the flat arrays, one sort copy and the item table, which turn into the uploaded batch one array at a time;
     - its sub-grids: the threads' copies, the partition's own grids, and the GPU deposit's host and device grids;
     - its padded particle copy (a partition), or the periodic copies of the whole box (one domain);
   and the plan is the busiest partition's sum, on top of the particles and the full-resolution grids ('fixed').
   Replayed on the logs of 38 production runs (TNG50/100/300, z = 127 .. 0, sampled and exact, from their 'Thread
   i had' lines) it bounds every measured peak footprint, by 19-90%. It stays high because 'fixed' holds the
   post-processing grids (168 B/cell for the production fields) next to the triangulations, and the footprints
   need that sum: the peak is in the post-processing, where macOS still counts much of the freed triangulation
   memory. On sub-GB runs the same retention can put the footprint up to ~30% above the plan (a 0.9M-particle
   crossed-wave set at 4^3 partitions: 0.71 GB against 0.54 predicted): the footprint ratchets up over the
   partitions. The counted padded vertices agree with the run's own to ~5% on clustered test
   sets (tests/dtfe_auto_tune_check.sh). The count grid is one pass over the particles (128^NO_DIM cells: 0.07 s for
   1.7e7 particles); a full search is <= 60 plans (~0.5 s, independent of N); a run without particles in memory keeps
   the uniform count. DTFE_AUTO_TUNE_DEBUG=1 prints the busiest partition's predicted padded count per thread (for
   comparison with the run's own 'Thread i had' lines). */

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
// 'origin' (the box's lower corner) with cell size dx; counts are per bin of up to 128 bins per axis (the coarse grid
// resolves no finer). An axis with fewer cells than pieces (the run refuses that split) is cut into equal pieces.
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
            size_t const nb = std::min( cells, size_t(128) );
            std::vector<size_t> binEdge( nb + 1 ), counts( nb ), idx( parts[d] + 1, 0 );
            for (size_t i=0; i<=nb; ++i) binEdge[i] = ( i * cells ) / nb;
            for (size_t i=0; i<nb; ++i)
            {
                double lo[NO_DIM], hi[NO_DIM];
                for (int e=0; e<NO_DIM; ++e)
                {
                    double const ra = ( e == d ) ? a + double(binEdge[i]) : cur[2*e];
                    double const rb = ( e == d ) ? a + double(binEdge[i+1]) : cur[2*e+1];
                    lo[e] = origin[e] + ra * dx[e];
                    hi[e] = origin[e] + rb * dx[e];
                }
                counts[i] = size_t( std::llround( dens.count( lo, hi ) ) );
            }
            optimalSplit( counts, parts[d], idx.data() );
            for (size_t k=0; k<=parts[d]; ++k) edges[k] = a + double( binEdge[ std::min( idx[k], nb ) ] );
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

// Bytes per padded vertex of a standard sub-domain while its tessellation is built: the thread's particle copy (alive
// while the triangulation grows) + the triangulation (96 B incl. the double volume accumulator + 6.77 cells x 72 B +
// slack; the double build's vertex data in 8 B instead of 4: +50 B/vertex) + the TBB insert's sorted copies with
// '--parallel-triangulation' (+203 B per vertex, measured on PS), and 6% for the dummy points the padding test adds on
// the padded box's faces (3-5% of a sub-box at TNG scale). One definition for the plan below and the partition loop's
// look-ahead.
inline double stdBytesPerPaddedVertex(User_options const &u)
{
    double const kDT = kStdVertexBytes + 10. * ( double(sizeof(Real)) - 4. );   // the double build: +10 B per 4 B of Real
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    double const kParInsert = u.psParallelTriangulation ? kParInsertBytes : 0.;
#else
    double const kParInsert = 0.;
    (void)u;
#endif
    return 1.06 * ( double(sizeof(Particle_data)) + kDT + kParInsert );
}

}  // namespace

// (auto_tune.h)
double stdLookAheadBytes(User_options const &u, double const nOwn, double const len[NO_DIM],
                         double const pad[NO_DIM], size_t const tGrid[NO_DIM])
{
    double fPart = 1., fThreads = 1.;
    for (int d=0; d<NO_DIM; ++d)
    {
        double const l = std::max( len[d], 1e-300 );
        fPart    *= 1. + 2.*pad[d] / l;
        fThreads *= 1. + 2.*pad[d]*double( std::max<size_t>( tGrid[d], 1 ) ) / l;
    }
    return nOwn * fThreads * stdBytesPerPaddedVertex( u ) + nOwn * fPart * double(sizeof(Particle_data));
}

namespace {

// The busiest partition of a split (P per axis, or the user's own) with up to T threads in each: its bytes on top
// of 'fixed', the padded vertices its sub-triangulations hold, and those per thread (DTFE_AUTO_TUNE_DEBUG)
// (+ the GPU snapshot of its largest thread sub-grid, and the look-ahead's bytes: the tessellation stage
// of the partition that needs the most, which may be any partition's turn to be prepared ahead)
struct StdPlan { double bytes = 0., vertices = 0., snapshot = 0., aheadMax = 0.; int part = -1; size_t parts = 1;
                 std::vector<double> perThread; };

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
    double fixedLoop = 0.;           // ... while the partition loop runs: its grids only (bCellDeposit; plan P8.9)
    double perVertex = 0.;           // bytes per padded vertex while a tessellation is built (stdBytesPerPaddedVertex)
    double bGpuCell = 0.;            // the GPU deposit's bytes per sub-grid cell, per copy (0: no GPU deposit)
    double faceCell = 0.;            // --face-gradient: avg1FacePass's 9 doubles per cell of each sub-domain it runs on
    double perTet = 0.;              // ... its bytes per extracted tetrahedron
    double tetsPerOwn = 0.;          // ... the tetrahedra extracted per particle of a thread's own box
    double snapWordFactor = 0.;      // the GPU snapshot's bytes per 4 B grid component (2: fixed-point words)
    bool lookAheadPossible = false;  // the partition look-ahead runs (GPU deposit, more than one partition)
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
        // a partition's grids: its own (DTFE_parallel's, when split), the threads' sub-grid copies, the GPU's two
        auto gridBytes = [&](double cells, size_t nThreads) {
            return ( nParts > 1. ? cells * bCellDeposit : 0. ) + ( nThreads > 1 ? cells * bCellDeposit : 0. )
                   + 2. * cells * bGpuCell + cells * faceCell; };
        if ( not dens.ok )
        {   // every sub-box holds the box average (the uniform count)
            double const tSplit = std::pow( double(threads), 1. / double(NO_DIM) );
            double f = 1.;
            for (int d=0; d<NO_DIM; ++d) f *= 1. + 2.*padFrac[d]*double(partsP[d])*tSplit;
            double const own = N / nParts;
            plan.vertices = own * f;
            plan.bytes = plan.vertices * perVertex + own * tetsPerOwn * perTet
                       + gridBytes( gridTotal / nParts, size_t(threads) ) + ( nPts > 0 ? ptsWalkBytes : 0. );
            plan.part = 0;
            plan.snapshot = snapWordFactor * bGpuCell * gridTotal / nParts / double( threads );
            double fPart = 1.;
            for (int d=0; d<NO_DIM; ++d) fPart *= 1. + 2.*padFrac[d]*double(partsP[d]);
            plan.aheadMax = plan.vertices * perVertex + ( nParts > 1. ? own * fPart * partBytes : 0. );
        }
        else
        {
            std::vector<double> whole( 2*NO_DIM, 0. );
            for (int d=0; d<NO_DIM; ++d) whole[2*d+1] = double( u.gridSize[d] );
            auto const partitions = stdTuneSplit( dens, origin, dxGrid, whole, partsP );
            for (size_t ip=0; ip<partitions.size(); ++ip)
            {
                std::vector<double> const &pr = partitions[ip];
                size_t pGrid[NO_DIM];
                double cells = 1.;
                for (int d=0; d<NO_DIM; ++d)
                {
                    pGrid[d] = 1;
                    cells *= std::max( 1., pr[2*d+1] - pr[2*d] );
                    pgOptions.gridSize[d] = size_t( std::max( 1LL, (long long)std::llround( pr[2*d+1] - pr[2*d] ) ) );
                }
                if ( threads > 1 )
                    parallelGrid( size_t(threads), pgOptions, pGrid );   // DTFE_parallel's thread grid (2^a 3^b threads)
                size_t nThreads = 1;
                for (int d=0; d<NO_DIM; ++d) nThreads *= pGrid[d];
                std::vector< std::vector<double> > const boxes = nThreads > 1
                    ? stdTuneSplit( dens, origin, dxGrid, pr, pGrid ) : std::vector< std::vector<double> >( 1, pr );
                double bytes = 0., vertices = 0., maxThreadCells = 0.;
                std::vector<double> perThread;
                auto boxCount = [&](std::vector<double> const &b, bool const padded) {
                    double lo[NO_DIM], hi[NO_DIM];
                    for (int d=0; d<NO_DIM; ++d)
                    {
                        lo[d] = origin[d] + b[2*d]*dxGrid[d]   - ( padded ? double( u.paddingLength[2*d] ) : 0. );
                        hi[d] = origin[d] + b[2*d+1]*dxGrid[d] + ( padded ? double( u.paddingLength[2*d+1] ) : 0. );
                    }
                    return dens.count( lo, hi ); };
                for (std::vector<double> const &b : boxes)
                {
                    double const padded = boxCount( b, true );
                    perThread.push_back( padded );
                    vertices += padded;
                    bytes += padded * perVertex + boxCount( b, false ) * tetsPerOwn * perTet;
                    double c = 1.;
                    for (int d=0; d<NO_DIM; ++d) c *= std::max( 1., b[2*d+1] - b[2*d] );
                    maxThreadCells = std::max( maxThreadCells, c );
                }
                // a partition's padded particle copy (the partition loop's), or one domain's periodic copies
                // (DTFE_preparePadding) / a '--region' crop's padding beyond the N counted in 'fixed'
                double const partPadded = boxCount( pr, true );
                bytes += ( nParts > 1. ? partPadded : std::max( 0., partPadded - N ) ) * partBytes;
                bytes += gridBytes( cells, nThreads ) + ( nPts > 0 ? ptsWalkBytes : 0. );
                // the run's largest thread sub-grid, whichever partition holds it (the snapshot is held by one at a time)
                plan.snapshot = std::max( plan.snapshot, snapWordFactor * bGpuCell * maxThreadCells );
                // this partition prepared ahead: its particle copy and its sub-domains' tessellations
                if ( nParts > 1. )
                    plan.aheadMax = std::max( plan.aheadMax, vertices * perVertex + partPadded * partBytes );
                if ( bytes > plan.bytes )
                {
                    plan.bytes = bytes;
                    plan.vertices = vertices;
                    plan.part = int(ip);
                    plan.perThread = perThread;
                }
            }
        }
        return plans.emplace( key, plan ).first->second;
    }

    /* The run's peak with 'loop' bytes (the busiest partition, its GPU snapshot and look-ahead) in the partition loop
       (plan P8.9 step 1). The loop holds only its own grids -- density, velocity and the gradient (bCellDeposit;
       DTFE.cpp swaps the derived fields for the gradient while it runs) -- and the derived grids are allocated after
       it (postProcessWebFields), when the partitions are gone: but the allocator keeps freed blocks in the footprint,
       so part of the busiest partition is booked beside them: all of it up to kStdPostKeepFloor, a share kStdPostKeep
       of a larger one. The larger of the two phases is the peak. Never above the sum of both (the model before
       2026-11-07), and equal to it with '--scratch-dir' or without derived fields (fixedLoop == fixed). */
    double stdPeak(double loop) const
    {
        double const kept = std::min( loop, std::max( kStdPostKeep * loop, kStdPostKeepFloor ) );
        return std::max( fixedLoop + loop, fixed + kept );
    }

    AutoTuneReport chooseSplit() const;
    AutoTuneReport reportSplit(int const bestP, int const bestT, bool const nothingFits) const;
};

// Searches the splits for the coarsest one that fits with the most threads (or, when none fits, the least demanding).
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
    // a plan's bytes: the busiest partition and the GPU snapshot (need), at the run's peak (stdPeak: the loop or the
    // post-processing, whichever holds more); the look-ahead is added once the split is chosen, as PS-DTFE's, and
    // switched off when it does not fit (below)
    auto need = [&](int P, int T) { StdPlan const &pl = planOf( P, T ); return pl.bytes + pl.snapshot; };
    for (int P=pLo; P<=pHi; ++P)
    {
        if ( need(P,tLeast) < need(leastP,tLeast) ) leastP = P;
        int T = tMax;
        while ( autoConc and T>1 and stdPeak( need(P,T) ) > budget )
            --T;
        if ( stdPeak( need(P,T) ) > budget )
            continue;   // not even one thread fits at this split
        if ( T > bestT or (T==bestT and P<bestP) )
        {
            bestP = P;
            bestT = T;
        }
        if ( T == tMax )
        {
            bestP = P;
            bestT = T;
            break;      // full thread count fits at the coarsest split so far: done
        }
    }
    bool const nothingFits = ( bestT == 0 );
    if ( nothingFits )
    {
        bestP = leastP;
        bestT = tLeast;
    }
    return reportSplit( bestP, bestT, nothingFits );
}

// Fills the report of the chosen split, turns off a look-ahead that does not fit, and states the choice.
inline AutoTuneReport AutoTuneStdPlanner::reportSplit(int const bestP, int const bestT, bool const nothingFits) const
{
    char buf[512];
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
    // particle copy and tessellations on top of the peak. A plan that fits without it but not with it runs without it
    // (stdLookAheadOffBudget, as PS-DTFE's psLookAheadOffBudget), and the prediction is then the plan's alone.
    report.lookAhead = ( lookAheadPossible and chosen.parts > 1 ) ? chosen.aheadMax : 0.;
    u.stdLookAheadCounted = report.lookAhead;   // the loop checks the free memory against it (DTFE.cpp)
    report.fixedLoop = fixedLoop;
    if ( report.lookAhead > 0. and stdPeak( chosen.bytes + chosen.snapshot + report.lookAhead ) > budget )
    {
        message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " look-ahead off for this run: the busiest "
                << "partition (~" << chosen.bytes/1.e9 << " GB) plus the next one's tessellations (~"
                << report.lookAhead/1.e9 << " GB) do not fit the " << (budget - fixedLoop - chosen.snapshot)/1.e9
                << " GB the memory budget leaves beside the partition loop's grids and particles; partitions are "
                << "prepared in turn.\n" << MESSAGE::Flush;
        u.stdLookAheadOffBudget = true;
        report.lookAhead = 0.;
    }
    double const predicted = stdPeak( chosen.bytes + chosen.snapshot + report.lookAhead );
    report.predicted = predicted;
    if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )
    {
        std::fprintf( stderr, "AUTO-TUNE-DEBUG standard: %s (count grid %.3f s, %zu plans %.3f s); busiest partition "
                      "%d of %zu: %.0f padded vertices, %.3f GB on top of %.3f GB; per thread:",
                      dens.ok ? "counted from the particles" : "uniform", countSec, plans.size(),
                      std::chrono::duration<double>( std::chrono::steady_clock::now() - tuneStart ).count() - countSec,
                      chosen.part + 1, chosen.parts, chosen.vertices, chosen.bytes/1.e9, fixed/1.e9 );
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
                << "or --grid, or pass a finer --partition.\n" << MESSAGE::EndWarning;
    }
    return report;
}

}  // namespace
#endif

#ifdef PHASE_SPACE
namespace {

/* The cells of the largest partition sub-grid of each n^NO_DIM split, n = 1 .. 6 (cells[n]), and of the user's own
   split k[] when it is not cubic (cellsUser), from the particles (plan P8.2; 0: not known). The binary crops a
   partition's grids to the Eulerian span of its kept tetrahedra per axis (psSubgridBox, ps_interpolation.cc), and that
   span grows as structure forms: the logged 5^3 runs park ~4x the bytes per modelled cell at z = 0 as at z = 127,
   where the model booked G/n + 40 cells per axis whatever the snapshot. (A single domain never crops: the planner
   books the whole grid there, gpuSubCells.)
   The partitions are equal slices of the Lagrangian box -- on a periodic run of the period [seam - L, seam), the seam
   placed by the partition padding (DTFE.cpp regions(), lagPadding; subpartition.h psPeriodicSeam, replayed below
   without its exactness test) -- owned by the tetrahedra's Lagrangian centroids. So per split: the particles binned on
   a B^NO_DIM grid over that Lagrangian domain (a partition's edge rounded to the nearest bin), each bin's Eulerian
   extent per axis in grid cells (periodic: each particle the minimal image of the bin's first one, as the deposit
   wraps a tetrahedron about its first vertex); a partition's touched planes = its bins' extents widened below for the
   tetrahedra; then psSubgridBox's rule (their [lo, hi], or on a periodic axis touched at both ends the complement of
   the largest untouched run), at most the axis. DTFE_AUTO_SPAN=0: the G/n + 40 rule. */
inline void psPartitionSpans(std::vector<Particle_data> const &p, User_options &u, double const padFrac[NO_DIM],
                             int const *userK, double (&cells)[7], double &cellsUser)
{
    for (double &c : cells) c = 0.;
    cellsUser = 0.;
#ifdef PHASE_SPACE
    if ( p.empty() ) return;
    double const N = double( p.size() );
    // bins per axis: about a lattice cell's diagonal of particles each, at most 60
    int const B = std::max( 6, std::min( 60, int( std::pow( N, 1. / double(NO_DIM) ) / std::sqrt( double(NO_DIM) ) ) ) );
    double bbLo[NO_DIM], bbHi[NO_DIM], boxLo[NO_DIM], boxLen[NO_DIM], dx[NO_DIM];
    int G[NO_DIM];
    double vol = 1.;
    for (int d=0; d<NO_DIM; ++d)
    {
        double lo = double( p[0].lagPos[d] ), hi = lo;
        for (Particle_data const &q : p) { double const v = double( q.lagPos[d] ); lo = std::min( lo, v ); hi = std::max( hi, v ); }
        double const eps = ( hi - lo ) * 1.e-6;   // computeLagrangianBoundingBox's margin
        bbLo[d] = lo - eps;
        bbHi[d] = hi + eps;
        boxLo[d] = double( u.boxCoordinates[2*d] );
        boxLen[d] = double( u.boxCoordinates[2*d+1] ) - boxLo[d];
        G[d] = int( u.gridSize[d] );
        dx[d] = boxLen[d] / double( G[d] );
        if ( not ( dx[d] > 0. ) or G[d] < 1 or not ( bbHi[d] > bbLo[d] ) ) return;
        vol *= u.periodic ? boxLen[d] : bbHi[d] - bbLo[d];
    }
    // the partition padding of a split with k partitions on axis d (DTFE.cpp): --ps-lagrangian-padding k spacings (capped at a
    // period), else the Eulerian --padding fraction floored at 0.3 of a partition; >= 4.2 alpha cuts
    double const lagSpacing = u.psLagPadding > Real(0.) ? std::pow( vol / N, 1. / double(NO_DIM) ) : 0.;
    auto padOf = [&](int d, int k) {
        double const len = bbHi[d] - bbLo[d];
        double pad = std::max( padFrac[d] * len, len / double(k) * 0.3 );
        if ( lagSpacing > 0. )
        {
            pad = double( u.psLagPadding ) * lagSpacing;
            if ( u.periodic and pad > boxLen[d] ) pad = boxLen[d];
        }
        if ( u.psLagAlphaRadius > 0. ) pad = std::max( pad, 4.2 * double( u.psLagAlphaRadius ) );
        return pad;
    };
    // the Lagrangian domain the partitions tile, per axis: the period below the seam, or the bounding box
    auto domainOf = [&](int d, int k, double &lo0, double &len0) {
        if ( not u.periodic ) { lo0 = bbLo[d]; len0 = bbHi[d] - bbLo[d]; return; }
        double const L = boxLen[d], pad = padOf( d, k );
        double const m = std::max( 4.e-6 * L, 8. * std::ldexp( 1., std::ilogb( std::max( std::fabs( bbLo[d] ), std::fabs( bbHi[d] ) ) + L ) - 23 ) );
        double const sd = bbLo[d] + L - pad - m;
        double const seam = ( sd > bbHi[d] - L ) ? sd : bbHi[d];   // psPeriodicSeam, or its fallback anchor
        lo0 = seam - L;
        len0 = L;
    };
    size_t nb = 1;
    for (int d=0; d<NO_DIM; ++d) nb *= size_t(B);
    double const unset = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> ref;
    std::vector<float> lo, hi;
    double binnedLo[NO_DIM];
    bool binned = false;
    // per bin and axis: the Eulerian extent in grid cells over the Lagrangian domain starting at dLo (length dLen)
    auto binAll = [&](double const (&dLo)[NO_DIM], double const (&dLen)[NO_DIM]) {
        ref.assign( nb*NO_DIM, unset );
        lo.assign( nb*NO_DIM, 0.f );
        hi.assign( nb*NO_DIM, 0.f );
        for (Particle_data const &q : p)
        {
            size_t b = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                double t = ( double(q.lagPos[d]) - dLo[d] ) / dLen[d];
                if ( u.periodic ) t -= std::floor( t );   // above the seam: its -L copy
                int const i = std::min( B - 1, std::max( 0, int( t * B ) ) );
                b = b * size_t(B) + size_t(i);
            }
            for (int d=0; d<NO_DIM; ++d)
            {
                size_t const k = b*NO_DIM + size_t(d);
                double x = double( q.pos[d] );
                if ( std::isnan( ref[k] ) )
                {   // the bin's first particle: its reference and extent
                    ref[k] = x;
                    lo[k] = hi[k] = float( ( x - boxLo[d] ) / dx[d] );
                    continue;
                }
                if ( u.periodic ) x = ref[k] + ( x - ref[k] ) - boxLen[d] * std::round( ( x - ref[k] ) / boxLen[d] );
                float const c = float( ( x - boxLo[d] ) / dx[d] );
                lo[k] = std::min( lo[k], c );
                hi[k] = std::max( hi[k], c );
            }
        }
    };
    // a tetrahedron reaches about one particle spacing past the particles it joins: per axis, each bin's extent is
    // widened by a spacing at the bin's own stretch (its Eulerian extent over its Lagrangian width, at least 1) and a
    // cell -- inside the partition and into the padding alike. Measured against psSubgridBox on pancakes and crossed
    // waves (32^3 - 128^3 particles, 128^3 and 384^3 grids, 2^3 and 4^3 splits): 1.00 - 2.09x the largest sub-grid,
    // geometric mean 1.24x, never below; the G/n + 40 rule was 0.05 - 4.3x
    double const spacingBins = double(B) / std::pow( N, 1. / double(NO_DIM) );   // a particle spacing in bin widths
    std::vector<int> diff[NO_DIM];
    // the largest sub-grid of the split k[] (partitions per axis)
    auto spanOf = [&](int const (&k)[NO_DIM]) -> double {
        double dLo[NO_DIM], dLen[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) domainOf( d, k[d], dLo[d], dLen[d] );
        bool same = binned;
        for (int d=0; d<NO_DIM; ++d) same = same and dLo[d] == binnedLo[d];
        if ( not same )
        {
            binAll( dLo, dLen );
            for (int d=0; d<NO_DIM; ++d) binnedLo[d] = dLo[d];
            binned = true;
        }
        size_t nParts = 1;
        for (int d=0; d<NO_DIM; ++d) nParts *= size_t( k[d] );
        double most = 0.;
        for (size_t pi=0; pi<nParts; ++pi)
        {
            int idx[NO_DIM], b0[NO_DIM], b1[NO_DIM];
            size_t rem = pi;
            for (int d=NO_DIM-1; d>=0; --d) { idx[d] = int( rem % size_t(k[d]) ); rem /= size_t(k[d]); }
            for (int d=0; d<NO_DIM; ++d)
            {   // partition idx's first bin and the next partition's, per axis
                b0[d] = int( std::floor( double(idx[d]) * double(B) / double(k[d]) + 0.5 ) );
                b1[d] = std::max( b0[d] + 1, int( std::floor( double(idx[d] + 1) * double(B) / double(k[d]) + 0.5 ) ) );
            }
            bool any[NO_DIM], full[NO_DIM];
            for (int d=0; d<NO_DIM; ++d) { diff[d].assign( size_t( G[d] ) + 1, 0 ); any[d] = full[d] = false; }
            auto mark = [&](int d, double a, double z) {   // planes floor(a) - 1 .. floor(z) + 1 on axis d (the
                                                           // sampled deposit's +-1 cell, psSubgridBox)
                long lo0 = long( std::floor( a ) ) - 1, hi0 = long( std::floor( z ) ) + 1;
                any[d] = true;
                if ( full[d] ) return;
                if ( hi0 - lo0 + 1 >= G[d] ) { full[d] = true; return; }
                if ( u.periodic )
                {
                    long const a0 = ( lo0 % G[d] + G[d] ) % G[d], z0 = a0 + ( hi0 - lo0 );
                    if ( z0 < G[d] ) { ++diff[d][size_t(a0)]; --diff[d][size_t(z0) + 1]; }
                    else { ++diff[d][size_t(a0)]; --diff[d][size_t(G[d])]; ++diff[d][0]; --diff[d][size_t(z0 - G[d]) + 1]; }
                }
                else
                {
                    lo0 = std::max( 0L, lo0 ); hi0 = std::min( long(G[d]) - 1, hi0 );
                    if ( lo0 <= hi0 ) { ++diff[d][size_t(lo0)]; --diff[d][size_t(hi0) + 1]; }
                }
            };
            int j[NO_DIM];
            for (int e=0; e<NO_DIM; ++e) j[e] = b0[e];
            while ( true )
            {
                size_t b = 0;
                for (int e=0; e<NO_DIM; ++e) b = b * size_t(B) + size_t(std::min( j[e], B - 1 ));
                if ( not std::isnan( ref[b*NO_DIM] ) )
                    for (int d=0; d<NO_DIM; ++d)
                    {
                        size_t const kb = b*NO_DIM + size_t(d);
                        double const binCells = dLen[d] / double(B) / dx[d];
                        double const reach = std::max( double( hi[kb] - lo[kb] ), binCells ) * spacingBins + 1.;
                        mark( d, double( lo[kb] ) - reach, double( hi[kb] ) + reach );
                    }
                int e = NO_DIM - 1;
                while ( e >= 0 and ++j[e] >= b1[e] ) { j[e] = b0[e]; --e; }
                if ( e < 0 ) break;
            }
            double c = 1.;
            for (int d=0; d<NO_DIM; ++d)
            {
                double span = G[d];
                if ( any[d] and not full[d] )
                {
                    int run = 0, lo0 = -1, hi0 = -1, gap = 0, cur = 0;
                    for (int g=0; g<G[d]; ++g)
                    {
                        run += diff[d][size_t(g)];
                        if ( run > 0 ) { if ( lo0 < 0 ) lo0 = g; hi0 = g; gap = std::max( gap, cur ); cur = 0; }
                        else ++cur;
                    }
                    if ( lo0 >= 0 )
                        span = ( u.periodic and lo0 == 0 and hi0 == G[d] - 1 ) ? double( G[d] - gap ) : double( hi0 - lo0 + 1 );
                }
                c *= span;
            }
            most = std::max( most, c );
        }
        return most;
    };
    for (int n=1; n<=6; ++n)
    {
        int k[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) k[d] = n;
        cells[n] = spanOf( k );
    }
    if ( userK != nullptr )
    {
        int k[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) k[d] = std::max( 1, userK[d] );
        cellsUser = spanOf( k );
    }
#else
    (void)p; (void)u; (void)padFrac; (void)userK;
#endif
}

// The mean streams per sample point the plan budgets, and why: DTFE_AUTO_PTS_STREAMS, or estimated from the particles.
inline void autoTunePSPointStreams(User_options &u, double const N, size_t const nPts,
                                   std::vector<Particle_data> const *particles, AutoTuneReport &report,
                                   double &ptsMeanStreams, char (&ptsStreamsWhy)[200])
{
    if ( Env::isSet( "DTFE_AUTO_PTS_STREAMS" ) )
    {
        double const v = Env::real( "DTFE_AUTO_PTS_STREAMS", 0. );
        if ( v >= 1. )
        {
            ptsMeanStreams = v;
            snprintf( ptsStreamsWhy, sizeof(ptsStreamsWhy), "%.3g streams/point from DTFE_AUTO_PTS_STREAMS", v );
        }
    }
    else if ( nPts > 0 and particles != nullptr )
    {
        double boxLo[3], boxLen[3];
        for (int d=0; d<3; ++d)
        {
            boxLo[d]  = double( u.boxCoordinates[2*d] );
            boxLen[d] = double( u.boxCoordinates[2*d+1] ) - boxLo[d];
        }
        int patches = 0;
        // the estimate is made on the points' own slab -- their bounding box, widened by one mean particle
        // spacing per side -- not the whole box: the heavy tail sits in a few haloes a plane misses (TNG50-3-Dark
        // z=0: box mean 17.98, a z2048 plane's '.pts_streams' 5.38). A points set that fills the box gets the box
        // mean. DTFE_AUTO_PTS_BOXMEAN=1 forces the box-wide estimate.
        double rLo[3], rHi[3];
        bool haveRegion = false;
        if ( not Env::flag( "DTFE_AUTO_PTS_BOXMEAN" ) and NO_DIM == 3 )
        {
            double pLo[NO_DIM], pHi[NO_DIM], bLo[NO_DIM], bLen[NO_DIM];
            for (int d=0; d<NO_DIM; ++d) { bLo[d] = boxLo[d]; bLen[d] = boxLen[d]; }
            if ( autoTuneSamplePointBounds( u.psSamplePointsFile, bLo, bLen, u.periodic, pLo, pHi ) )
            {
                double const spacing = std::cbrt( boxLen[0]*boxLen[1]*boxLen[2] / std::max( 1., N ) );
                double vol = 1.;
                for (int d=0; d<NO_DIM; ++d)
                {
                    rLo[d] = pLo[d] - spacing;
                    rHi[d] = pHi[d] + spacing;
                    if ( u.periodic and rHi[d] - rLo[d] >= boxLen[d] ) { rLo[d] = boxLo[d]; rHi[d] = boxLo[d] + boxLen[d]; }
                    vol *= (rHi[d] - rLo[d]) / boxLen[d];
                }
                haveRegion = vol < 0.999;   // a points set that fills the box: the box mean is its slab's
            }
        }
        double const est = psEstimateMeanStreams( *particles, boxLo, boxLen, u.periodic, &patches,
                                                  haveRegion ? rLo : nullptr, haveRegion ? rHi : nullptr );
        if ( est > 0. )
        {
            ptsMeanStreams = std::max( 1., 1. + 2.*(est - 1.) );
            report.streamsEstimate = est;
            snprintf( ptsStreamsWhy, sizeof(ptsStreamsWhy),
                      "%.2f streams/point estimated from %d Lagrangian patches%s, budgeted as %.2f", est, patches,
                      haveRegion ? " on the points' slab" : "", ptsMeanStreams );
        }
    }
}

// The bytes the '--sample-points' structures hold: per point its position and outputs, and its per-stream records.
inline double autoTunePSPointBytes(User_options &u, size_t const nPts, bool const ptsVelGrad,
                                   double const ptsMeanStreams, double *recordBytes = nullptr,
                                   double *outputBytes = nullptr)
{
    // The STORED per-stream record is packed (ps_point_eval.cc RecLayout): denGeo + denDtfe +
    // vel[3] doubles + the 2-B caustic mask = 42 B, plus grad[3] / vgrad[9] doubles / ids[4]
    // uint64 only with --pts-den-grad / --pts-vel-grad / --ps-per-stream-ids. KEEP IN SYNC.
    double bStreamRec = 5. * double(sizeof(double)) + 2.;
    if ( u.psPtsDenGrad )   bStreamRec += 3. * double(sizeof(double));
    if ( ptsVelGrad )       bStreamRec += 9. * double(sizeof(double));
    if ( u.psPerStreamIds ) bStreamRec += 4. * 8.;
    if ( u.psPtsScalar )    bStreamRec += double(noScalarComp) * double(sizeof(double));
    if ( u.psPtsLagrangian ) bStreamRec += 3. * double(sizeof(double));
    // ... stored as uint32 point index + the packed record in blocks, and grouped at finalize by
    // one uint64 locator per record (ps_point_eval.cc HitSink / psPointEvalFinalize)
    bStreamRec += 4. + 8.;
    // per-point: pos (3 doubles) + bucketPts (uint32) + finalize's offsets and cursor (2 uint64)
    double const bPerPoint = 3. * double(sizeof(double)) + 4. + 2. * 8.;
    // finalized outputs, allocated while the records are still being drained
    double bPtsOut = double(sizeof(double))            // outDen
                   + 3. * double(sizeof(double))       // outVel
                   + 6. * double(sizeof(double))       // outDisp
                   + 4.;                               // outStreams (int32)
    if ( u.psPtsDenGrad ) bPtsOut += 3. * double(sizeof(double));
    if ( ptsVelGrad ) bPtsOut += 9. * double(sizeof(double));
    if ( u.psPerStream )  bPtsOut += ptsMeanStreams * 4. * double(sizeof(double));   // ragged records
    if ( u.psPtsScalar )  bPtsOut += double(noScalarComp) * double(sizeof(double))
                                     * (1. + (u.psPerStream ? ptsMeanStreams : 0.));   // point + per-stream scalars
    if ( u.psPtsLagrangian ) bPtsOut += ptsMeanStreams * 3. * double(sizeof(double));  // per-stream q(x)
    // for the loop's dynamic concurrency: the records as the partitions merge them (without their finalize locator),
    // and what the finalize allocates after the loop (the outputs, the per-point offsets and the per-record locators)
    if ( recordBytes ) *recordBytes = double(nPts) * ptsMeanStreams * ( bStreamRec - 8. );
    if ( outputBytes ) *outputBytes = double(nPts) * ( bPtsOut + 2. * 8. + ptsMeanStreams * 8. );
    return double(nPts) * (bPerPoint + bPtsOut + ptsMeanStreams * bStreamRec);
}

// Warns when the '--scratch-dir' volume cannot hold the full-grid accumulators.
inline void autoTuneScratchSpaceCheck(User_options &u, bool const scratchOn, double const gridTotal, double const bCell)
{
    if ( scratchOn )
    {
        struct statvfs vfs;
        if ( statvfs(u.scratchDir.c_str(), &vfs) == 0 )
        {
            double const freeBytes = double(vfs.f_bavail) * double(vfs.f_frsize);
            double const need = gridTotal*bCell;
            if ( need * 1.05 + 2.e9 > freeBytes )
            {
                MESSAGE::Warning warning( u.verboseLevel );
                warning << "AUTO-TUNE: the scratch volume cannot hold this run's grids: '"
                        << u.scratchDir << "' has " << freeBytes/1.e9 << " GB free but the "
                        << "full-resolution accumulators need ~" << need/1.e9 << " GB ("
                        << MESSAGE::printElements( u.gridSize, "x" ) << " cells x " << bCell
                        << " B/cell). The allocator will fall back to the heap when the disk "
                        << "runs out -- expect swapping. Free up space, use a bigger volume, or "
                        << "reduce --grid / --field.\n" << MESSAGE::EndWarning;
            }
        }
    }
}

// The bytes per sub-grid cell of one CPU deposit thread's private copy: the larger pass's grids that are not
// fixed-point sums.
inline double autoTunePSCopyCell(User_options &u, AutoTunePS &ps, bool const momFx)
{
    double copyCell = 0.;
    for (Field *f : { &u.uField, &u.aField })
    {
        if ( not f->selected() ) continue;
        double const R = double(sizeof(Real));
        bool const massFx = ps.phaseSpace and sizeof(Real) == 4 and ( f->density or f->velocity_tweb );   // as fxCell
        double const fxPass = ( ps.phaseSpace and sizeof(Real) == 4 and momFx ) ? autoTuneFxMomentBytes(*f, ps) : 0.;
        copyCell = std::max( copyCell, autoTuneBytesPerCell(*f, ps, 0) - autoTunePSCountBytes(*f, ps, 0)
                                       - ( massFx ? R : 0. ) - fxPass );
    }
    return copyCell;
}

/* What the PS-DTFE plan reads: its inputs, resolved once by autoTunePartitioning from the options, the data and the
   machine. Values, set by name there; the planner below only reads them. */
struct AutoTunePSInputs
{
    bool metalActive = false;        // the GPU deposit will run (flag given AND compiled in)
    bool autoPart = false;           // --partition left to the tuner
    bool userPartitionOn = false;    // a user's split that runs the partition path ('--partition 1 2 2' too)
    bool autoConc = false;           // --max-concurrent left to the tuner
    double minN = 0.;                // the particle count from which a split is chosen for memory (kAutoMinN)
    double N = 0.;                   // the particles
    double ramBytes = 0.;            // physical RAM
    std::string budgetWhy;           // how the memory budget was set (the AUTO-TUNE line)
    double budget = 0.;              // the memory budget
    int cores = 1;                   // the threads
    size_t outGrid[NO_DIM] = {};     // the cells allocated and written (--grid-window: the window's)
    double gridTotal = 0.;           // ... their product
    double spanCells[7] = {};        // the largest partition sub-grid of an n^3 split, n <= 6 (psPartitionSpans; 0: unknown)
    int userK[NO_DIM] = {};          // the user's --partition per axis ...
    bool userNonCubic = false;       // ... when its counts differ (the planner's n is then u.partition[0]: partsOf)
    double spanUser = 0.;            // ... and its largest sub-grid (psPartitionSpans; 0: unknown)
    AutoTunePS ps;                   // the PS flags
    double bCellDeposit = 0.;        // full-grid bytes per cell during the deposit (both passes)
    double bCell = 0.;               // ... the larger of the deposit and the post phase
    std::string fxNote;              // the AUTO-TUNE line's --ps-merge fixed-point note
    double padFrac[NO_DIM] = {};     // the Eulerian padding over the box length, per axis
    double partBytes = 0.;           // bytes per particle (sizeof(Particle_data))
    double kParInsert = 0.;          // the TBB insert's bytes per vertex (kParInsertBytes, 0 without it)
    bool scratchOn = false;          // --scratch-dir: the full grids are not in RAM
    size_t nPts = 0;                 // the '--sample-points'
    std::string ptsStreamsWhy;       // ... the streams per point budgeted, and why
    double ptsBytes = 0.;            // ... their structures
    double ptsRecordBytes = 0.;      // ... of which the stream records (written as the partitions merge)
    double ptsOutputBytes = 0.;      // ... and the finalized outputs (allocated at the end)
    double particleBytes = 0.;       // the particles held (0 with deferred particles)
    double readKept = 0.;            // the read buffers the allocator keeps (--grid-window)
    double fixed = 0.;               // the irreducible term: particles + read buffers + grids + points
    bool psVel = false, psDisp = false, psGrad = false, psExact = false;   // the GPU deposit's grid selection
    bool momFx = false;              // the moment grids are fixed-point sums
    double gridB = 0.;               // the GPU grids' bytes per cell and copy
    bool psDensity = false;          // a density grid is selected
    double fxCell = 0.;              // the fixed-point mass words' extra bytes per sub-grid cell
    double fxMom = 0.;               // ... the moment words' (the larger pass)
    double copyCell = 0.;            // one CPU deposit thread's private copy per sub-grid cell
    double threadScratch = 0.;       // the deposit threads' per-tetrahedron sample lists, in all
};

/* The PS-DTFE plan once its inputs are known (autoTunePartitioning): the memory model of a split (partitionBytes,
   oneDomainPeak, snapshotBytes, lookAheadBytes), the report of a choice (finishReport) and the phases that choose the
   split (choose). It reads the inputs by name (its base) and writes only through the three references. */
struct AutoTunePSPlanner : AutoTunePSInputs
{
    User_options &u;                 // the options the plan sets: --partition, --max-concurrent, the look-ahead switch
    AutoTuneReport &report;          // the report it completes (fixed, budget and the points' terms already set)
    MESSAGE::Message &message;       // the AUTO-TUNE lines

    // ONE DOMAIN on the CPU ('--partition 1 1 1', or one domain picked here: partitionOn false): the triangulation
    // deposits straight into the run's own grids (DTFE_interpolation: no reserveMemory, sub-grid or addFromSubgrid),
    // so the grids 'fixed' books are the deposit's own; partitionBytes(1) would book them twice (copyOne) and every
    // grid's conversion at once (38.8 GB for a 768^3 run that peaks at 22.9). Modelled per pass (u, then a), in
    // ps_interpolation.cc's order: during the loop the fixed-point words (the density's 8 B, each moment grid's 2 x
    // its Real bytes), the int stream count, the flags and the Real accumulators that are not fixed-point sums; after
    // it each moment grid's Real grid beside its words, the words freed grid by grid (psFxToReal), then the density
    // from the mass words; then the export grids ('.streams' or '.tetTouch', '.hidden_streams', the caustic bits, the
    // volume-weighted dispersion's density copy).
    // A freed block under 4 GiB stays in the footprint (macOS's allocator keeps it, serving later requests of at
    // least half its size), so the export grids pile on top of freed words smaller than that; the next pass reuses
    // what the previous one freed, and the derived fields come last. Bytes, for 'cells' sub-grid cells.
    double oneDomainPeak(double const cells) const
    {
        double const B4 = kAllocKeepBytes;      // the allocator returns a freed block of this size or more
        double const R = double(sizeof(Real));
        bool const fxOn = ps.phaseSpace and sizeof(Real) == 4 and momFx;
        double live = 0., kept = 0., peak = 0.;
        auto alloc = [&](double const b) { live += b*cells; peak = std::max( peak, live + kept ); };
        auto release = [&](double const b) { live -= b*cells; if ( b*cells < B4 ) kept += b*cells; };
        for (Field *f : { &u.uField, &u.aField })
        {
            if ( not f->selected() ) continue;
            bool const massFx = ps.phaseSpace and sizeof(Real) == 4 and ( f->density or f->velocity_tweb );   // fxCell
            std::vector<double> const G = fxOn ? autoTuneFxGrids(*f, ps) : std::vector<double>();
            double sumG = 0.;
            for (double const g : G) sumG += g;
            double const mask = ps.caustics ? double(sizeof(PSCausticClass::CausticMask)) : 0.;
            double const E = R + R + ( ps.caustics ? R : 0. )                                    // the export grids
                           + ( ( ps.volumeWeighted and f->velocity_dispersion and massFx ) ? R : 0. );
            // the loop's temporaries: the pass's slots, less the grids made after it, plus the words and the int count
            double const pre = autoTuneBytesPerCell(*f, ps, 0) - E - sumG - ( massFx ? R : 0. )
                             + 2.*sumG + ( massFx ? 8. : 0. ) + double(sizeof(int));
            double const reuse = std::min( kept, pre*cells );   // ... in what the previous pass freed
            kept -= reuse;
            live += pre*cells;
            peak = std::max( peak, live + kept );
            for (double const g : G) { alloc( g ); release( 2.*g ); }
            if ( massFx ) { alloc( R ); release( 8. ); }
            alloc( E );
            release( double(sizeof(int)) + 2. + mask );   // the int count, the flags, the caustic mask (pass end)
        }
        // the derived fields and the T-web stage come on top of what the passes left
        alloc( autoTuneDerivedBytes(u.uField) + autoTuneDerivedBytes(u.aField) );
        return peak;
    }

    // per-partition bytes for an n^3 Lagrangian split
    double partitionBytes(int n) const
    {
        double const nOwn = N / partsOf( n );
        double const fPad = padFactorOf( n );   // k spacings, or the Eulerian rule's floor of 0.3 partition widths
        double const nPad = nOwn * fPad;
        double m = nPad * (partBytes + kPSVertexBytes + kParInsert);   // particle copy coexists with the growing triangulation
        // '--sample-points': each worker fills a LOCAL hits buffer (pair<uint32,StreamRec>,
        // padded to 176 B) and merges it under the mutex at the end, so this part DOES scale
        // down with a finer split -- roughly this partition's share of the total records.
        if ( nPts > 0 )
            m += double(std::thread::hardware_concurrency()) * kHitSinkBlockBytes;   // one HitSink block per thread
        // Eulerian sub-grid this partition allocates (subGridCells below)
        double const subCells = subGridCells( n );
        if ( metalActive )
        {
            // flat tet arrays (the 16-float record 64 + masses 4 + the finite-cell ordinal 4 B/tet,
            // + vels 48 when velocity-derived grids are requested; the cost-sort permutes IN PLACE,
            // so no gather copy) + the PSGpuGrids host and device copies of the selected grids
            // (gpuCellBytes; under --grid-window held once, snapshotBytes)
            double const perTet = (kGpuTetBytes + (psVel or psDisp or psGrad ? kGpuTetVelBytes : 0.)) * kGpuTetSlack;
            m += perTet*kTetsPerVertex*nOwn + ( u.psWindowOn ? 0. : gpuCellBytes()*subCells );
            // --exact-average runs depositExactItems (ps_metal_host.cc): a per-tet weight total
            // (4 B) + the item list, 8 B per 16 cells of a tet's bbox window -- ~20x the tet's
            // volume in cells on the crossed-waves benchmark, at least one item -- capped at 32M items
            if ( psExact )
            {
                double const nTet = kTetsPerVertex*nOwn;
                // --grid-window: items cover only the window's cells (the hull of each tet's bbox with
                // the window), so the window's cell count sets the items, not the virtual full grid
                double gridCells = 1.;
                for (int d=0; d<NO_DIM; ++d) gridCells *= double(outGrid[d]);
                double const cellsPerTet = gridCells / (6.*double(N));
                // (the weight totals are two 4 B words per tet: an exact per-tet sum)
                m += (momFx ? 8. : 4.)*nTet + 8.*std::min( 32.e6, nTet*(1. + 20.*cellsPerTet/16.) );
            }
        }
        else
        {
            // CPU deposit sub-grids + the deposit threads' private copies, as many as the copy budget allows, unless
            // that leaves fewer than half the threads: the deposit then runs in SLAB mode (ps_interpolation.cc, the
            // same rule) with no copies, holding a cell handle, a first-slab byte and ~2 slab-list entries per tet
            // (+ the binning arrays). The fixed-point words (fxCell, fxMom) count during the deposit, and again at
            // their conversion after the loop only where that exceeds the copies or slab lists, freed by then. A
            // private copy holds neither the words nor the shared counts, flags and masks (autoTunePSCountBytes), so
            // the default single build has copyPriv = 0 and every core in the copy mode, as ps_interpolation.cc's
            // accBytes; only the float sums (double build, DTFE_GPU_FLOAT_MOMENTS=1) book copies.
            double const copyOne = bCellDeposit*subCells;
            double const copyPriv = std::max( 0., copyCell )*subCells;   // (copyCell above: one pass's copy)
            int const copyThreads = copyPriv > 0.
                ? std::min( cores, 1 + int( std::min( double(cores), psDepositThreadBytes() / copyPriv ) ) ) : cores;
            double const slabBytes = kTetsPerVertex * nPad * ( 8. + 1. + 2.*4. + 20. );
            double const helpers = 2 * copyThreads < cores ? slabBytes : double(copyThreads - 1) * copyPriv;
            // One domain (oneDomainPeak above): its grids are 'fixed''s, so only what the deposit holds beyond them
            // counts here, with the copies or slab lists and the threads' sample lists, all kept by the allocator
            // once freed. A user's split such as '--partition 1 2 2' runs the partition path (userPartitionOn), and
            // with --scratch-dir 'fixed' has no grid term: both keep the partition formula.
            if ( n == 1 and not userPartitionOn and not scratchOn )
                m += helpers + threadScratch + std::max( 0., oneDomainPeak( subCells ) - bCell*subCells );
            else
                m += copyOne + (fxCell + fxMom)*subCells + std::max( helpers, (fxCell + fxMom)*subCells );
        }
        return m;
    }

    // The GPU deposit's host snapshot (ps_metal_host.cc GpuSnapshot): to resume a failed command buffer
    // the deposit keeps a host copy of every buffer its kernel writes -- the device grids (gridB, with the 8 B
    // fixed-point mass for the 4 B float one: +4 B/cell), the per-tet masses (4 B/tet) and the exact items' weight
    // totals (4 B/tet) -- taken at each pass's start and every ~10 s of GPU work. It is ~4.7 GB per
    // partition at 1024^3 split 2^3 (volume-weighted velocity), past the gate's 4 GB headroom. Not in
    // partitionBytes(): the deposits run one at a time (ctxMutex), so whatever the concurrency there is ONE such
    // copy, added once to every plan below. (The CUDA port keeps none yet: there it is slack; the model errs high.)
    // --grid-window: every partition's sub-grid is the whole window (subGridCells), and the grids of the deposit
    // running are held once in the same way -- the device and host copies (gpuCellBytes) and, beside them, the
    // partition's own sub-grids (bCellDeposit: its density, counts and flags; fxCell + fxMom: the fixed-point words
    // while they are converted), all freed and kept by the allocator for the next partition's deposit. In flight
    // and waiting, a partition holds only its tessellation (with the window's cells counted per partition, three
    // in flight would book them three times: +20% at 3^3 over 2.1M particles on a 4096^2 window, measured).
    double snapshotBytes(int n) const
    {
        if ( not metalActive ) return 0.;
        double const subCells = subGridCells( n );   // the same sub-grid as partitionBytes()
        double const nTet = kTetsPerVertex * N / partsOf( n );
        double const window = u.psWindowOn ? ( gpuCellBytes() + bCellDeposit + fxCell + fxMom )*subCells : 0.;
        return (gridB + 4.)*subCells + (psExact ? (momFx ? 12. : 8.) : 4.)*nTet + window;
    }

    // the GPU deposit's grids per sub-grid cell: gridB (above) -- every buffer in PSGpuGrids -- mirrored host and
    // device (the x2), + 4 B/cell: the device's mass grid is a 64-bit fixed-point sum (ps_metal_host.cc), + 4 B/cell
    // more on the host, which receives those 8 B words instead of a 4 B float grid (PSGpuGrids::massFx) -- unless a
    // density is selected: the caller then allocates its density sub-grid (4 B; 8 in the double build) only after
    // the call, from the words, so the peak (the copy-back) is no higher than a float grid's
    double gpuCellBytes() const
    {
        double const hostFx = ( ps.phaseSpace and not psDensity ) ? 4. : 0.;
        return 2.*gridB + 4. + hostFx;
    }

    // The partition look-ahead (DTFE.cpp): at one partition in flight with the GPU deposit, the NEXT
    // partition's particle copy + tessellation are built during the deposit -- one more partition's
    // triangulation stage on top of the peak (the run skips a look-ahead the free memory cannot hold)
    double lookAheadBytes(int n, int c) const
    {
        if ( not metalActive or u.psNoOverlap or u.psLookAheadOffBudget or c != 1 or partsOf( n ) <= 1. ) return 0.;
        double const fPad = padFactorOf( n );
        return N / partsOf( n ) * fPad * (partBytes + kPSVertexBytes + kParInsert);
    }

    // what a given split and concurrency cost; the concurrency the run WILL use (0 = all cores)
    AutoTuneReport finishReport(int n, int c) const
    {
        report.valid = true;
        report.partition = std::max( n, 1 );
        report.maxConcurrent = std::max( c, 1 );
        // partitions in flight: never more than there are (one domain x 10 threads is ONE domain, not ten)
        int const inFlight = int( std::min( double(report.maxConcurrent), partsOf( report.partition ) ) );
        if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )   // what the plan books per partition sub-grid (P8.2's check)
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS booked sub-grid: %.0f cells (split %d, %.0f partitions)\n",
                          subGridCells( report.partition ), report.partition, partsOf( report.partition ) );
        report.perPartition = partitionBytes( report.partition );
        report.gpuSnapshot = snapshotBytes( report.partition );   // once: the GPU deposits run one at a time
        // the loop's dynamic concurrency (P8.10, DTFE.cpp): what the run allocates after its partitions are in flight
        report.concChosen = autoConc;
        report.reserveEnd = ( scratchOn ? 0. : gridTotal * std::max( 0., bCell - bCellDeposit ) ) + ptsOutputBytes;
        report.reserveFirstMerge = scratchOn ? 0. : gridTotal * ( autoTuneMergeGrownBytes( u.uField, ps )
                                                                 + autoTuneMergeGrownBytes( u.aField, ps ) );
        report.reserveRecords = ptsRecordBytes;
        double const base = fixed + report.gpuSnapshot + double(inFlight) * report.perPartition;
        report.lookAhead = lookAheadBytes( report.partition, inFlight );
        // the split above is chosen without the look-ahead; adding it here anyway would make a plan that fits
        // report over_budget=1 (the launcher's memory check would halve a plane that fits) and run past a
        // budget set on purpose. Every return passes here, so this is where the
        // plan, its prediction and the run are made to agree: a look-ahead that does not fit beside one
        // partition is switched off for this run (DTFE.cpp reads psLookAheadOffBudget), and the prediction
        // is then the grids + particles + one partition. Over budget anyway, the look-ahead only adds to it.
        if ( report.lookAhead > 0. and base + report.lookAhead > budget )
        {
            message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " look-ahead off for this run: one "
                    << "partition (~" << report.perPartition/1.e9 << " GB) plus the next one's tessellation (~"
                    << report.lookAhead/1.e9 << " GB) does not fit the " << (budget - fixed)/1.e9 << " GB the "
                    << "memory budget leaves beside the grids and particles; partitions are prepared in turn.\n"
                    << MESSAGE::Flush;
            u.psLookAheadOffBudget = true;
            report.lookAhead = 0.;
        }
        report.predicted = base + report.lookAhead;
        return report;
    }

    // the cells of a partition's sub-grid in an n^3 split (psUseSubgrid): a Lagrangian sub-box maps to a DISPLACED,
    // distorted Eulerian region, so the crop is the touched span + slop, never above the axis -- valid on both
    // periodic and non-periodic runs since ps_interpolation.cc crops a periodic axis to its WRAPPED span instead of
    // keeping the whole axis
    double gpuSubCells(int n) const
    {
        // measured on the particles (P8.2): a split's partitions crop; one domain never does (psUseSubgrid is the
        // partition path's), so n = 1 books the whole grid below whatever the particles span
        if ( isUserSplit( n ) and spanUser > 0. ) return spanUser;
        if ( not isUserSplit( n ) and n >= 2 and n <= 6 and spanCells[n] > 0. ) return spanCells[n];
        double c = 1.;
        for (int d=0; d<NO_DIM; ++d)
            c *= std::min( double(outGrid[d]), double(u.gridSize[d])/double( isUserSplit( n ) ? userK[d] : n ) + 40. );
        return c;
    }

    // the user's own split when it is not cubic: the planner names a split by n = u.partition[0], and these give that
    // split's partitions and padding per axis instead of n^NO_DIM (every other split is n per axis)
    bool isUserSplit(int n) const { return userNonCubic and not autoPart and n == userK[0]; }
    double partsOf(int n) const
    {
        double c = 1.;
        for (int d=0; d<NO_DIM; ++d) c *= double( isUserSplit( n ) ? userK[d] : n );
        return c;
    }
    double padFactorOf(int n) const
    {
        if ( not isUserSplit( n ) ) return padFactor( n );
        double f = 1.;
        for (int d=0; d<NO_DIM; ++d) f *= 1. + 2.*autoTunePadWidths( u, padFrac[d], userK[d], N, kPartitionPadFloor );
        return f;
    }

    // ... the cells it allocates: under --grid-window the WHOLE window, whatever the split (psSubgridBox,
    // ps_interpolation.cc: every partition's sub-grid is the window, merged window-local, DTFE.cpp) -- the crop
    // above is per axis, so a window spanning x and y of a 4096^2 slice would book 723^2 cells at 6^3, not 4096^2
    double subGridCells(int n) const { return u.psWindowOn ? gridTotal : gpuSubCells( n ); }

    // one partition's padding factor in an n^3 split (k spacings, or the Eulerian rule's floor)
    double padFactor(int n) const { return autoTunePadFactor( u, padFrac, n, N, kPartitionPadFloor ); }

    AutoTuneReport choose() const;
    std::optional<AutoTuneReport> irreducibleCase(int const userC, int const userN, int const nGpuMin,
                                                  bool &overBudget) const;
    std::optional<AutoTuneReport> smallSetCase(int const userC, int const userN, int const nGpuMin) const;
    std::optional<AutoTuneReport> overBudgetCase(int const userC, int const nGpuMin, bool const overBudget) const;
    AutoTuneReport memorySplit(int const userC, int const nGpuMin) const;
};

// Chooses the split and the concurrency: the GPU's sub-grid cap, then the first case that decides -- the irreducible
// term over budget, a small particle set, over budget at any split, else the memory-driven split.
inline AutoTuneReport AutoTunePSPlanner::choose() const
{
    int const userC = u.maxConcurrent > 0 ? u.maxConcurrent : cores;
    int const userN = u.partition.empty() ? 1 : int(u.partition[0]);

    // The GPU kernels number a partition's sub-grid cells with 32 bits: a sub-grid above 2^32 cells
    // (1626^3 unsplit) falls back to the CPU deposit (ps_interpolation.cc, averaged_interpolation_1.cc).
    // With the GPU on, every split chosen below keeps the sub-grid under that cap -- the same crop +
    // slop estimate as partitionBytes() -- so a very fine grid keeps its GPU deposit; a user's own
    // split that does not is told which one would.
    double const gpuCellCap = kGpuCellCap;            // UINT32_MAX
    int nGpuMin = 1;
    if ( metalActive )
        while ( nGpuMin < 64 and gpuSubCells(nGpuMin) > gpuCellCap ) ++nGpuMin;
    if ( metalActive and not autoPart and gpuSubCells(userN) > gpuCellCap )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: with --partition " << userN << " each partition's sub-grid holds ~" << gpuSubCells(userN)
                << " cells, above the 2^32 the GPU kernels number: the deposit will run on the CPU. "
                << "--partition " << nGpuMin << " (or finer) keeps it on the GPU.\n" << MESSAGE::EndWarning;
    }

    bool overBudget = false;   // set below; the split is then chosen once partitionBytes() exists
    if ( auto const r = irreducibleCase( userC, userN, nGpuMin, overBudget ) ) return *r;
    if ( auto const r = smallSetCase( userC, userN, nGpuMin ) ) return *r;
    if ( auto const r = overBudgetCase( userC, nGpuMin, overBudget ) ) return *r;
    return memorySplit( userC, nGpuMin );
}

// The irreducible term (grids + particles + points) over budget: warns, and decides for a small set or a user's split.
inline std::optional<AutoTuneReport> AutoTunePSPlanner::irreducibleCase(int const userC, int const userN,
                                                                         int const nGpuMin, bool &overBudget) const
{
    if ( fixed > budget )
    {
        double const gridBytes = scratchOn ? 0. : gridTotal*bCell;
        // largest cubic grid whose accumulators still leave room for the particles and points
        double const room = budget - particleBytes - ptsBytes;
        long   const maxGrid = room > 0. ? long( std::cbrt( room / bCell ) ) : 0L;
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: this run does not fit in memory, and --partition cannot fix it.\n  ";
        if ( not scratchOn )
            warning << "The full-resolution output grids need " << gridBytes/1.e9 << " GB ("
                    << MESSAGE::printElements( u.gridSize, "x" ) << " cells x " << bCell << " B/cell for the "
                    << "selected fields), plus ";
        else
            warning << "With the grids on scratch disk, ";
        warning << (partBytes*N)/1.e9 << " GB of particles";
        if ( readKept > 0. )
            warning << ", " << readKept/1.e9 << " GB of read buffers the allocator keeps (--grid-window)";
        if ( nPts > 0 )
            warning << " and " << ptsBytes/1.e9 << " GB for the " << double(nPts) << " --sample-points ("
                    << ptsStreamsWhy << "; DTFE_AUTO_PTS_STREAMS overrides)";
        warning << " = " << fixed/1.e9 << " GB, against a memory budget of " << budgetWhy << ".\n"
                << "  --partition splits the TRIANGULATION only: every Lagrangian partition deposits into "
                << "the same shared full-resolution grid and point set, so that part is irreducible.\n"
                << "  Options: ";
        if ( not scratchOn )
            warning << "(a) '--scratch-dir <local dir>' backs the grids with disk instead of RAM "
                    << "(needs ~" << gridBytes/1.e9 << " GB free on a local, non-synced volume, e.g. "
                    << "/private/tmp/dtfe-scratch); ";
        if ( maxGrid > 0 )
            warning << "(b) a smaller --grid -- in RAM, the largest that fits at these fields is about "
                    << maxGrid << "^3; ";
        else
            warning << "(b) a smaller --grid cannot help: the particles" << (nPts > 0 ? " and sample points" : "")
                    << " alone exceed the budget; ";
        warning << "(c) fewer --field entries (the gradient family costs 36 B/cell, "
                << "dispersion 24 B/cell); ";
        if ( nPts > 0 )
            warning << "(d) fewer --sample-points (a smaller image plane) or no --pts-vel-grad; ";
        warning << "(e) more RAM. NOTE: '--partition-index'/'--region' are NOT a way out -- they crop particles by "
                << "Eulerian position and are rejected by this binary because the PS tessellation is Lagrangian.\n"
                << "  Running anyway with --max-concurrent 1 and the split with the smallest per-partition "
                << "triangulation; expect swapping.\n" << MESSAGE::EndWarning;
        if ( autoConc ) u.maxConcurrent = 1;
        // A small particle set triangulates cheaply in one piece; otherwise fall through to the
        // split choice below -- one UNPARTITIONED triangulation would put the whole
        // tessellation (~100 GB for 94M particles) on top of an already over-budget run.
        if ( N < minN or not autoPart )
        {
            int const nOne = autoPart ? nGpuMin : userN;      // one domain, or the GPU cap's split
            if ( autoPart ) { u.partition.assign( NO_DIM, size_t(nOne) ); u.partitionOn = ( nOne > 1 ); }
            return finishReport( nOne, autoConc ? 1 : userC );
        }
        overBudget = true;
    }
    return std::nullopt;
}

// A small particle set that fits: the user's split, one domain, or a split for speed from the time model.
inline std::optional<AutoTuneReport> AutoTunePSPlanner::smallSetCase(int const userC, int const userN,
                                                                      int const nGpuMin) const
{
    char buf[512];
    // A small particle set triangulates cheaply in one piece -- unless that piece's DEPOSIT does not fit:
    // on a very fine grid (1700^3 from 0.26M particles) one domain's sub-grid accumulators and the
    // deposit threads' private copies are tens of GB, and only a split shrinks them. Such a run takes
    // the memory-driven split below instead (unsplit it is predicted at 83 GB, over budget).
    bool const smallFits = not autoPart or fixed + snapshotBytes( std::max( 1, nGpuMin ) )
                                           + partitionBytes( std::max( 1, nGpuMin ) ) <= budget;
    if ( N < minN and smallFits )     // small particle set AND the grid fits: one domain, or a split for SPEED
    {
        if ( not autoPart )
        {   // the user's split at the concurrency the report states: left unset, a hand-set '--partition 4 4 4'
            // would start all 64 partitions on a 10-core machine while the report says mc=10
            if ( autoConc ) u.maxConcurrent = cores;
            return finishReport( userN, autoConc ? cores : userC );
        }
        // The one stage a split parallelises is the sequential triangulation (+ vertex densities):
        // the deposit and the point evaluation already use every core inside one domain. A
        // triangulation-bound run (a slice of a small box: 1.5 s of 3 s) therefore gains from
        // 2^3 partitions, each triangulating a padded eighth on its own core; a deposit-bound one
        // (a 256^3 grid at 0.26M particles: 1.5 s of 18 s) does not, and loses the padding.
        // Predicted wall time per split, from per-unit costs measured on a 10-core Apple Silicon
        // (single-thread seconds; DTFE_AUTO_T_VERTEX/_SAMPLE/_POINT override):
        double tVertex = 3.6e-6, tSample = 3.5e-7, tPoint = 7.5e-7;
        if ( double const v = Env::real( "DTFE_AUTO_T_VERTEX", 0. ); v > 0. ) tVertex = v;
        if ( double const v = Env::real( "DTFE_AUTO_T_SAMPLE", 0. ); v > 0. ) tSample = v;
        if ( double const v = Env::real( "DTFE_AUTO_T_POINT", 0. );  v > 0. ) tPoint = v;
        double fCopy = 1.;                              // the single domain's periodic copies (none without --periodic)
        if ( u.periodic )
            for (int d=0; d<NO_DIM; ++d) fCopy *= 1. + 2.*padFrac[d];
        double const nSubCube = ( u.aField.selected() or psExact )
                              ? std::pow( double(std::max( 1, psExact ? 3 : int(u.psAvgSubsamples) )), double(NO_DIM) ) : 1.;
        double const depositWork = gridTotal * nSubCube * tSample + double(nPts) * tPoint;
        auto predictedWall = [&](int n, int c) -> double
        {
            double const parts = partsOf( n );
            double const fPad = padFactorOf( n );
            double const rounds = std::ceil( parts / double(c) );
            double const triangulate = ( n == 1 ? fCopy * N : fPad * N / parts ) * tVertex;
            double const threads = ( n == 1 ) ? double(cores) : double(std::max( 1, cores / c ));
            // a GPU deposit serialises on the one command queue: the split leaves it unchanged
            // and adds each partition's buffer setup, extraction and leftover passes
            double const deposit = metalActive ? depositWork / double(cores) : depositWork / parts / threads;
            double const overhead = ( n == 1 ) ? 0. : parts * ( N * 5.e-9 + 0.02 + ( metalActive ? 0.1 : 0. ) );
            return rounds * ( triangulate + deposit ) + overhead;
        };
        int bestN = 1, bestC = autoConc ? cores : userC;
        double const single = predictedWall( 1, cores );
        double bestT = single;
        if ( nGpuMin > 1 )                  // the GPU's sub-grid cap rules out one domain
        {
            bestN = nGpuMin;
            bestC = std::max( 1, std::min( { nGpuMin*nGpuMin*nGpuMin, cores,
                                               int( std::floor( (budget - fixed - snapshotBytes(nGpuMin)) / partitionBytes(nGpuMin) ) ) } ) );
            if ( not autoConc ) bestC = std::min( bestC, userC > 0 ? userC : cores );
            bestT = predictedWall( nGpuMin, bestC );
        }
        for (int n = std::max( 2, nGpuMin ); n <= std::max( 3, nGpuMin ); ++n)
        {
            int c = std::min( n*n*n, cores );
            int const cMem = int( std::floor( (budget - fixed - snapshotBytes(n)) / partitionBytes(n) ) );
            if ( cMem < 1 ) continue;
            c = std::min( c, cMem );
            if ( not autoConc ) c = std::min( c, userC > 0 ? userC : cores );
            double const t = predictedWall( n, c );
            if ( t < kSpeedSplitGain * bestT or ( bestN < nGpuMin ) ) { bestN = n; bestC = c; bestT = t; }   // a clear gain only
        }
        if ( bestN > 1 )
        {
            u.partition.assign( NO_DIM, size_t(bestN) );
            u.partitionOn = true;
            u.psOrderedMerge = true;   // uniform small partitions: merge in order, bit-reproducible at no cost
            u.psSpeedSplit = true;     // ... and no memory gate (DTFE.cpp; its own flag, the ordered merge
                                       // being every split's default)
            if ( autoConc ) u.maxConcurrent = bestC;
            if ( bestN == nGpuMin and nGpuMin > 1 )
                snprintf( buf, sizeof(buf),
                          "%s grid on the GPU: --partition %d %d %d --max-concurrent %d keeps each partition's "
                          "sub-grid under the 2^32 cells the GPU kernels number (~%.3g cells; predicted peak ~%.3g GB)",
                          MESSAGE::printElements( u.gridSize, "x" ).c_str(), bestN, bestN, bestN, u.maxConcurrent,
                          gpuSubCells(bestN), (fixed + snapshotBytes(bestN) + double(bestC)*partitionBytes(bestN))/1.e9 );
            else
                snprintf( buf, sizeof(buf),
                          "%.3g particles: triangulation-bound, so --partition %d %d %d --max-concurrent %d "
                          "(predicted %.2g s against %.2g s for one triangulation; predicted peak ~%.3g GB)",
                          N, bestN, bestN, bestN, u.maxConcurrent, bestT, single,
                          (fixed + snapshotBytes(bestN) + double(bestC)*partitionBytes(bestN))/1.e9 );
            message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf << fxNote
                    << ". Pass --partition 1 1 1 for one triangulation.\n" << MESSAGE::Flush;
            return finishReport( bestN, autoConc ? bestC : userC );
        }
        return finishReport( 1, autoConc ? cores : userC );
    }
    return std::nullopt;
}

// Over budget at any split: the split whose partition needs the least, one partition at a time.
inline std::optional<AutoTuneReport> AutoTunePSPlanner::overBudgetCase(int const userC, int const nGpuMin,
                                                                        bool const overBudget) const
{
    char buf[512];
    if ( overBudget )
    {
        // over budget whatever we do (warned above): one partition at a time, split so that the
        // partition in flight is as small as the padding allows -- partitionBytes() stops falling
        // once the padding shells dominate, so take its minimum over the validated range
        int bestN = std::max( 2, nGpuMin );
        for (int n = bestN + 1; n <= std::max( 6, nGpuMin ); ++n)
            if ( partitionBytes(n) < partitionBytes(bestN) ) bestN = n;
        u.partition.assign( NO_DIM, size_t(bestN) );
        u.partitionOn = true;
        snprintf( buf, sizeof(buf),
                  "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> OVER BUDGET: --partition %d %d %d "
                  "--max-concurrent %d (smallest partition, ~%.3g GB; predicted peak ~%.3g GB%s)",
                  N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
                  bestN, bestN, bestN, u.maxConcurrent, partitionBytes(bestN)/1.e9,
                  (fixed + snapshotBytes(bestN) + partitionBytes(bestN))/1.e9, metalActive ? ", GPU deposit" : "" );
        message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf << fxNote
                << ". Pass --partition/--max-concurrent to override.\n           memory budget: "
                << budgetWhy << "\n" << MESSAGE::Flush;
        return finishReport( bestN, autoConc ? 1 : userC );
    }
    return std::nullopt;
}

// The memory-driven split: the coarsest split whose concurrency the budget allows, and the AUTO-TUNE line stating it.
inline AutoTuneReport AutoTunePSPlanner::memorySplit(int const userC, int const nGpuMin) const
{
    char buf[512];
    // Concurrency target = the core count, GPU or not. The GPU deposits do serialize on one
    // queue (ps_metal_host.cc ctxMutex), but the dominant per-partition cost is the CPU work
    // AROUND them -- Lagrangian triangulation + vertex densities, ~7 s of an ~8 s partition at
    // 2.1M particles / 256^3 -- which runs fully in parallel. A cap of 3 would run 8
    // partitions in three waves: 21 s on the GPU vs 9.5 s on the CPU. Memory still bounds it:
    // partitionBytes() counts the GPU buffers a waiting partition holds, and cMem caps c.
    int const targetC = cores;

    int bestN = -1, bestC = 0;
    int const nLo = autoPart ? std::max( 2, nGpuMin ) : int(u.partition[0]);
    int const nHi = autoPart ? std::max( 6, nGpuMin ) : int(u.partition[0]);   // <=6^3 keeps auto within validated territory (unless the GPU cap needs more)
    for (int n=nLo; n<=nHi; ++n)
    {
        double const m = partitionBytes(n);
        int cMem = int( std::floor( (budget - fixed - snapshotBytes(n)) / m ) );
        int c = std::min( { targetC, n*n*n, cMem } );
        if ( c > bestC or bestN<0 )
        {
            bestN = n;
            bestC = c;
        }
        if ( c >= std::min(targetC, n*n*n) )
            break;      // memory is not the binding constraint; smallest such n wins (fewest seams)
    }
    if ( bestC < 1 )
    {
        // 'fixed' fits (checked above), so this is the triangulation genuinely not fitting in
        // what is left -- here the split IS the right lever, and the finest one is the best try.
        bestC = 1;
        bestN = nHi;
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: even a single " << bestN << "^3-split partition's triangulation is predicted to "
                << "exceed the " << (budget - fixed)/1.e9 << " GB left after the full-grid accumulators ("
                << (gridTotal*bCell)/1.e9 << " GB) and particles (" << (partBytes*N)/1.e9 << " GB); running with "
                << "--max-concurrent 1 at the finest split. Expect swapping; a finer --partition than "
                << nHi << "^3 may still help.\n" << MESSAGE::EndWarning;
    }

    if ( autoPart )
    {
        u.partition.assign( NO_DIM, size_t(bestN) );
        u.partitionOn = (bestN > 1);
    }
    if ( autoConc )
        u.maxConcurrent = bestC;

    // the concurrency the run WILL use: the tuner's, or the user's own (0 = all cores)
    int const effC = autoConc ? bestC : userC;
    // the report first: it decides whether the look-ahead fits the budget, and the message states its prediction
    AutoTuneReport const decided = finishReport( bestN, effC );
    double const predicted = decided.predicted;
    char scratchNote[256] = "";
    if ( scratchOn )
        snprintf( scratchNote, sizeof(scratchNote), "; grid accumulators (%.3g GB) on scratch disk",
                  gridTotal*bCell/1.e9 );
    char ptsNote[320] = "";
    if ( nPts > 0 )
        snprintf( ptsNote, sizeof(ptsNote), "; %.3g sample points (%.3g GB: %s)",
                  double(nPts), ptsBytes/1.e9, ptsStreamsWhy.c_str() );
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %s --max-concurrent %d "
              "(predicted peak ~%.3g GB%s%s%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              MESSAGE::printElements( u.partition, " " ).c_str(), u.maxConcurrent,     // NO_DIM numbers
              predicted/1.e9, metalActive ? ", GPU deposit" : "", scratchNote, ptsNote );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf << fxNote
            << autoTuneOverrideNote( autoPart, autoConc ) << "\n           memory budget: "
            << budgetWhy << "\n" << MESSAGE::Flush;
    return decided;
}

}  // namespace
#endif

AutoTuneReport autoTunePartitioning(User_options &u,
                                    size_t const nParticles,
                                    bool const userSampling,
                                    bool const metalActive,
                                    std::vector<Particle_data> const *particles)
{
    if ( u.psServe )
    {
        autoTuneServe( u, nParticles );   // the grid/partition model below does not apply to a server
        return AutoTuneReport();
    }
    // partitionGiven, not only partitionOn: '--partition 1 1 1' asks for one triangulation, which
    // leaves partitionOn false -- reading that as "not given" would split the data anyway
    bool const autoPart = not (u.partitionOn or u.partitionGiven);
    // ... and a user's split that is not one domain on every axis ('--partition 1 2 2': partition[0] is 1, but it
    // runs the partition path): the model's n == 1 is the single-domain path only without it (partitionBytes)
    bool const userPartitionOn = u.partitionOn;
    bool const autoConc = not u.maxConcurrentOn;
    AutoTuneReport report;
    // a plan given in full still gets its prediction: the run's memory gate, RAM parking room and look-ahead room
    // come from this report (without it a pinned TNG100 z=1 '--partition 5 5 5 --max-concurrent 10' parked 43 GB in
    // RAM on a 64 GB machine); the planners change only the knobs left to them
    if ( u.partNo>=0 or u.redshiftConeOn or userSampling or u.psServe )
        return report;  // a mode where partitioning is unavailable
                        // (--serve keeps its single triangulation alive for every request)

    double minN = kAutoMinN;
    if ( double const v = Env::real( "DTFE_AUTO_MINN", 0. ); v > 0. )
        minN = v;

    double const N = double(nParticles);
    double const ramBytes = autoTunePhysicalRAM();
    std::string budgetWhy;
    double const budget = autoTuneBudget( ramBytes, budgetWhy );   // see the header comment
    int const cores = autoTuneCores();

    // the cells actually allocated and written: with --grid-window its cells, not the (virtual) full grid
    size_t outGrid[NO_DIM];
    u.outputGridSize( outGrid );
    double gridTotal = 1.;
    for (int d=0; d<NO_DIM; ++d)
        gridTotal *= double(outGrid[d]);

    AutoTunePS ps;
#ifdef PHASE_SPACE
    ps.phaseSpace     = true;
    ps.volumeWeighted = u.psVolumeWeighted;
    ps.exactDeposit   = u.psExactDeposit;
    ps.caustics       = u.psCaustics;
    ps.haloRelease    = u.psHaloRelease > Real(0.);
    ps.fixedMerge     = u.psFixedPointMerge;
#endif
    ps.faceGradient   = u.faceGradient;
    // uField and aField each allocate their OWN full grids and coexist (two reserveMemory calls in DTFE.cpp), so
    // summing both is right. The deposit and post phases are disjoint peaks -> max, not sum.
    double const bCellDeposit = autoTuneBytesPerCell(u.uField, ps, 0) + autoTuneBytesPerCell(u.aField, ps, 0);
    double const bCellPost    = autoTuneBytesPerCell(u.uField, ps, 1) + autoTuneBytesPerCell(u.aField, ps, 1);
    double const bCell        = std::max( bCellDeposit, bCellPost );
    // said on the AUTO-TUNE line: this run's merge keeps integer grids during the loop (ps_fixed_merge.h)
    std::string const fxNote = ps.fixedMerge ? " [--ps-merge fixed-point: the summed full grids counted at 9 B per component "
                                               "and cell during the loop; nothing staged or spilled]" : "";

    // padding fraction of the box per axis (paddingLength was resolved to Mpc by updateEntries)
    double padFrac[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        double const boxLen = double(u.boxCoordinates[2*d+1] - u.boxCoordinates[2*d]);
        padFrac[d] = boxLen>0. ? double(u.paddingLength[2*d] + u.paddingLength[2*d+1]) / (2.*boxLen) : 0.;
    }

    MESSAGE::Message message( u.verboseLevel );

#ifdef PHASE_SPACE
    // ---- PS-DTFE: Lagrangian partitions run in PARALLEL; concurrency is the throughput knob
    // and each concurrent partition holds a full padded triangulation (the dominant cost). ----
    // per padded vertex: the tessellation (kPSVertexBytes) and, beside it while it grows, the particle copy
    double const partBytes = double(sizeof(Particle_data));
    // '--parallel-triangulation' (TBB): the sorted point+payload copy and CGAL's own copy of it outlive the particle
    // copy during the insert: +203 B per vertex in flight, measured (autoTuneServe's kBuild 330 holds it there).
    // Counted whenever asked for: with fewer than 2 threads per partition the insert runs sequentially and this
    // errs high.
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    double const kParInsert = u.psParallelTriangulation ? kParInsertBytes : 0.;
#else
    double const kParInsert = 0.;
#endif
    // --scratch-dir (already armed by DTFE_setup when set): the full-grid accumulators live in
    // mmap'ed files, so they leave the RAM model -- the kernel pages them against the disk and
    // RSS stays bounded. The RAM check then covers only the particles + per-partition terms,
    // and the grid term is checked against the scratch volume's FREE SPACE instead.
    bool const scratchOn = not u.scratchDir.empty();

    // ---- '--sample-points' point evaluation (ps_point_eval.cc) ---------------------------
    // Also IRREDUCIBLE: the query points and their per-point stream records are GLOBAL
    // structures shared by every partition (each partition appends its hits), so --partition
    // does not shrink them. Without this a points-only run (tiny --grid, so the grid term
    // vanishes) could pick a coarse split against a budget that ignores tens of GB.
    size_t nPts = autoTuneSamplePointCount( u.psSamplePointsFile );
    if ( double const v = Env::real( "DTFE_AUTO_PTS_N", -1. ); v >= 0. )      // a plane that does not exist yet
        nPts = size_t(v);
    bool const ptsVelGrad = u.psPtsVelGrad or Env::flag( "DTFE_AUTO_PTS_VEL_GRAD" );
    // Mean streams (containing tetrahedra) per point. No fixed number fits: it grows with the mass
    // resolution (TNG300-3-Dark 1.899 at z=0, TNG100-3-Dark 4.25 at z=0 and 2.30 at z=1). So it is
    // ESTIMATED from the particles themselves (psEstimateMeanStreams: Eulerian over Lagrangian
    // volume of the tessellation, on random Lagrangian patches) and budgeted at TWICE the
    // multi-stream excess, 1 + 2 (est - 1): the estimate is the box mean, while one plane varies
    // around it (the z=0 plane's strips range 1.45 .. 22.8). DTFE_AUTO_PTS_STREAMS overrides.
    double ptsMeanStreams = 2.0;
    char ptsStreamsWhy[200] = "2 streams/point assumed";
    autoTunePSPointStreams( u, N, nPts, particles, report, ptsMeanStreams, ptsStreamsWhy );
    double ptsRecordBytes = 0., ptsOutputBytes = 0.;
    double const ptsBytes = autoTunePSPointBytes( u, nPts, ptsVelGrad, ptsMeanStreams, &ptsRecordBytes, &ptsOutputBytes );

    // IRREDUCIBLE in RAM unless scratch-backed: originals + the full-grid accumulators
    // + the point-evaluation structures.
    // --partition does NOT reduce this -- partitions sum into the SHARED full grid.
    // a --grid-window run that started without its particles (ps_globals_record.h) holds none: only
    // cached partitions are loaded, so their memory goes to more partitions at once. (A late read, when
    // a needed partition turns out uncached, adds them back: at most the particles' bytes over budget.)
    double const particleBytes = u.psParticlesDeferred ? 0. : partBytes*N;
    // ... and a --grid-window run keeps what reading them freed (autoTuneReadKeptBytes: nothing it allocates is large
    // enough to take those blocks over before the first partitions peak)
    double const readKept = ( u.psWindowOn and not u.psParticlesDeferred ) ? autoTuneReadKeptBytes( u, N ) : 0.;
    double const fixed = particleBytes + readKept + (scratchOn ? 0. : gridTotal*bCell) + ptsBytes;
    report.fixed = fixed;
    report.budget = budget;
    report.pts = ptsBytes;
    report.streamsBudgeted = nPts > 0 ? ptsMeanStreams : 0.;
    // NOTE: the over-budget check below must run BEFORE the small-N early return -- the grid
    // term does not depend on N, so a small snapshot on a huge --grid is exactly the case
    // that must not slip through to a silent 60+ GB allocation.

    autoTuneScratchSpaceCheck( u, scratchOn, gridTotal, bCell );

    // ---- Is the IRREDUCIBLE term alone over budget? --------------------------------------
    // If so ('fixed > budget' below), no --partition value can help: partitions are summed into the shared full
    // grid, and the search would end at its first candidate (2, the COARSEST split), blaming the split.
    // The per-tet/per-cell GPU-deposit pieces this field selection needs (unselected moment grids are not
    // allocated; the tet velocities are extracted only for a velocity-derived grid). The velocity derivatives are
    // folded into the gradient after auto-tune runs, so they count here as gradient requests.
    // dispersion implies the velocity/mom grids only when NOT volume-weighted (5d gating)
    bool const psVel  = u.uField.velocity or u.aField.velocity
                        or ((u.uField.velocity_dispersion or u.aField.velocity_dispersion)
                            and not ps.volumeWeighted);
    bool const psDisp = u.uField.velocity_dispersion or u.aField.velocity_dispersion;
    bool const psGrad = u.uField.velocity_gradient or u.uField.selectedVelocityDerivatives()
                        or u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives();
    // flags gating the remaining GPU buffers (ps_metal_host.cc)
    bool const psVolW  = ps.volumeWeighted;
    bool const psCaust = ps.caustics;
    bool const psExact = ps.exactDeposit;

    // Every buffer in PSGpuGrids, per cell and copy (the GPU deposit's host and device grids). mass 4 + streams 4
    // = the 8; the fVolW/fCaustic/fExact terms count too, and fVolW is the PRODUCTION default -- 20 B/cell/copy =
    // 40 after the x2 in partitionBytes(), i.e. 6.7 GB per partition at 1024^3/n=2.
    // The moment grids are 64-bit fixed-point sums (ps_metal_host.cc, fxUnits): 8 B per component
    // on the device and in the host's copy where a float takes 4 (mB = 2), and dispw is not deposited (the mass
    // grid stands for it); DTFE_GPU_FLOAT_MOMENTS=1 keeps the floats (mB = 1).
    bool const momFx = autoTunePSMomentsFixed();
    double const mB = momFx ? 2. : 1.;
    double const gridB  = 8. + mB * ( (psVel ? 12. : 0.) + (psDisp ? 24. : 0.) + (psGrad ? 36. : 0.)
                                    + (psVolW ? 4. : 0.)                  // momw
                                    + (psExact ? 4. : 0.)                 // streamvol
                                    + ((psVolW and psDisp) ? 12. : 0.) )  // dispvel
                        + (psCaust ? 4. : 0.)                             // caustic
                        + ((psVolW and psDisp and not momFx) ? 4. : 0.);  // dispw

    // The single build's fixed-point MASS grid (ps_interpolation.cc 'massFx'): a pass whose density is
    // selected accumulates its mass as 8 B words per sub-grid cell in place of the 4 B float density, shared by the
    // deposit threads (atomics; no private density copies), and converts them to the float density once after the
    // loop -- when the 8 B words and the 4 B grid coexist for a moment. The double build keeps its double sums.
    // fxCell = the extra bytes per sub-grid cell while the words live (the passes run one after the other: one grid).
    bool const psDensity = u.uField.density or u.uField.velocity_tweb or u.aField.density or u.aField.velocity_tweb;
    double const fxCell  = ( ps.phaseSpace and sizeof(Real) == 4 and psDensity ) ? 4. : 0.;
    // ... and the moment grids the same way: fxMom = the extra bytes per sub-grid cell while one pass's
    // words live (as many again as its Real moment grids: the larger pass), which also leave the private copies
    double const fxMom   = ( ps.phaseSpace and sizeof(Real) == 4 and momFx )
                         ? std::max( autoTuneFxMomentBytes(u.uField, ps), autoTuneFxMomentBytes(u.aField, ps) ) : 0.;
    // One private copy of the CPU deposit's copy mode, per sub-grid cell: the Real accumulators of ONE
    // pass -- the passes deposit one after the other, each copying only its own grids that are not fixed-point
    // sums (ps_interpolation.cc accBytes, of the 'quantities' it fills). Both passes' grids less one pass's words
    // and the shared counts, flags and masks (bCellDeposit - fxCell - fxMom - autoTunePSCountBytes of u and a) would
    // be wrong both ways with u and a fields selected: 16 B/cell of copies in the single build, which copies nothing,
    // and twice the double build's 32 B/cell, so few threads that the model would predict SLAB mode where the run
    // uses the copy mode.
    double copyCell = autoTunePSCopyCell( u, ps, momFx );

    // ... and each deposit thread's per-tetrahedron sample lists (DepositThread insideFlat/insideRel/insideW, 16 B
    // per sample of the largest tetrahedron it meets), which a very fine grid over few particles makes visible:
    // ~33 MB per thread at 512^3 over 8^3 particles: ~50x the samples of a mean owned tetrahedron of a single-stream
    // sheet, grid x nSub^3 / 6 N (crossed waves, 256^3 .. 512^3; which tetrahedra a thread meets depends on the
    // scheduling). Booked at 128x that mean; negligible at production densities (768^3 over 0.26M: 6 MB in all).
    int const nSubScratch = std::max( 1, psExact ? 3 : int(u.psAvgSubsamples) );
    double const sampleCube = ( u.aField.selected() or psExact ) ? std::pow( double(nSubScratch), double(NO_DIM) ) : 1.;
    double const threadScratch = double(cores) * 16. * 128. * gridTotal * sampleCube / ( 6. * std::max( 1., N ) );

    AutoTunePSInputs in;
    in.metalActive     = metalActive;
    in.autoPart        = autoPart;
    in.userPartitionOn = userPartitionOn;
    in.autoConc        = autoConc;
    in.minN            = minN;
    in.N               = N;
    in.ramBytes        = ramBytes;
    in.budgetWhy       = budgetWhy;
    in.budget          = budget;
    in.cores           = cores;
    for (int d=0; d<NO_DIM; ++d)
    {
        in.outGrid[d] = outGrid[d];
        in.padFrac[d] = padFrac[d];
    }
    in.gridTotal       = gridTotal;
    if ( not autoPart and not u.partition.empty() )
    {
        for (int d=0; d<NO_DIM; ++d) in.userK[d] = int( u.partition[ size_t(d) < u.partition.size() ? size_t(d) : 0 ] );
        for (int d=1; d<NO_DIM; ++d) in.userNonCubic = in.userNonCubic or in.userK[d] != in.userK[0];
    }
    if ( particles != nullptr and not u.psWindowOn and Env::flag( "DTFE_AUTO_SPAN", true ) )
    {
        auto const t0 = std::chrono::steady_clock::now();
        psPartitionSpans( *particles, u, padFrac, in.userNonCubic ? in.userK : nullptr, in.spanCells, in.spanUser );
        if ( Env::flag( "DTFE_AUTO_TUNE_DEBUG" ) )
        {
            std::fprintf( stderr, "AUTO-TUNE-DEBUG PS sub-grid spans (%.3f s), largest partition's cells per n^3 split "
                          "(measured / G/n+40):", std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
            for (int n=1; n<=6; ++n)
            {
                double c = 1.;
                for (int d=0; d<NO_DIM; ++d) c *= std::min( double(outGrid[d]), double(u.gridSize[d])/double(n) + 40. );
                std::fprintf( stderr, " %d: %.4g / %.4g", n, in.spanCells[n], c );
            }
            if ( in.userNonCubic ) std::fprintf( stderr, " user split: %.4g", in.spanUser );
            std::fprintf( stderr, "\n" );
        }
    }
    in.ps              = ps;
    in.bCellDeposit    = bCellDeposit;
    in.bCell           = bCell;
    in.fxNote          = fxNote;
    in.partBytes       = partBytes;
    in.kParInsert      = kParInsert;
    in.scratchOn       = scratchOn;
    in.nPts            = nPts;
    in.ptsStreamsWhy   = ptsStreamsWhy;
    in.ptsBytes        = ptsBytes;
    in.ptsRecordBytes  = ptsRecordBytes;
    in.ptsOutputBytes  = ptsOutputBytes;
    in.particleBytes   = particleBytes;
    in.readKept        = readKept;
    in.fixed           = fixed;
    in.psVel           = psVel;
    in.psDisp          = psDisp;
    in.psGrad          = psGrad;
    in.psExact         = psExact;
    in.momFx           = momFx;
    in.gridB           = gridB;
    in.psDensity       = psDensity;
    in.fxCell          = fxCell;
    in.fxMom           = fxMom;
    in.copyCell        = copyCell;
    in.threadScratch   = threadScratch;
    AutoTunePSPlanner const planner{ in, u, report, message };
    return planner.choose();

#else
    // ---- standard DTFE: partitions are processed SERIALLY (memory knob); inside one partition
    // the box is split across up to --max-concurrent threads (throughput knob). ----
    // A '--region'/'--region-mpc' run is planned as what it triangulates: DTFE_setup crops the region
    // first and passes a copy of the options whose box IS the region, with N = the particles inside it (unpadded),
    // so 'padFrac' above is the padding over the region's length -- the pairing a whole-box run has. The padded
    // copies the crop holds are left out of 'fixed': 36 B each next to the ~700 B per vertex of the triangulation
    // term, which counts them.
    double const partBytes = double(sizeof(Particle_data));

    /* '--sample-points' (the PS model's term above, for the standard point evaluation): the query points
       and their records are GLOBAL -- every partition appends the answers of the points in the tetrahedra it owns
       (ps_point_eval.cc), and a '--partition' run keeps them all until psPointEvalFinalize, the checkpoint's
       'points.<i>' files being copies on disk -- so a split does not shrink them: they are 'fixed'. Standard DTFE
       answers each point ONCE (its one tetrahedron of the Eulerian tessellation, 0/1 coverage), so one record per point
       where PS budgets its streams. The STORED record is packed (RecLayout): both densities + the velocity in doubles +
       the 2-B caustic mask (40 + 2), + the density gradient (24) / velocity gradient (72) / scalars with
       --pts-den-grad / --pts-vel-grad / --pts-scalar, + its uint32 point index and the uint64 locator finalize() groups
       it by; per point its position (3 doubles), its bucket entry and finalize's offset and cursor; and the finalized
       outputs (density, velocity, dispersion, stream count, + the gradients and scalars), allocated while the records
       are still held. KEEP IN SYNC with ps_point_eval.cc and the PS term. DTFE_AUTO_PTS_N / DTFE_AUTO_PTS_VEL_GRAD=1
       size a plane that does not exist yet (the launcher's memory check). A 4096^2 plane with '--pts-vel-grad'
       holds ~4.7 GB of it. */
    size_t nPts = autoTuneSamplePointCount( u.psSamplePointsFile );
    if ( double const v = Env::real( "DTFE_AUTO_PTS_N", -1. ); v >= 0. )
        nPts = size_t( v );
    bool const ptsVelGrad = u.psPtsVelGrad or Env::flag( "DTFE_AUTO_PTS_VEL_GRAD" );
    double bPtsVar = ( u.psPtsDenGrad ? 3. * double(sizeof(double)) : 0. ) + ( ptsVelGrad ? 9. * double(sizeof(double)) : 0. )
                   + ( u.psPtsScalar ? double(noScalarComp) * double(sizeof(double)) : 0. );
    double const bStreamRec = 5. * double(sizeof(double)) + 2. + bPtsVar + 4. + 8.;
    double const bPerPoint  = 3. * double(sizeof(double)) + 4. + 2. * 8.;
    double const bPtsOut    = double(sizeof(double)) + 3. * double(sizeof(double)) + 6. * double(sizeof(double)) + 4. + bPtsVar;
    double const ptsBytes   = double(nPts) * ( bPerPoint + bPtsOut + bStreamRec );
    // ... and each point-walk thread fills one 16 MB block of records before it hands it over (HitSink), per partition
    double const ptsWalkBytes = nPts > 0 ? double(std::thread::hardware_concurrency()) * kHitSinkBlockBytes : 0.;

    // --scratch-dir: the full-grid accumulators are mmap-backed and leave the RAM model. (The process itself -- code,
    // libraries, the GPU runtime: 66 MB of footprint, 84 MB with --gpu -- is left out, as in the PS model: it matters on
    // sub-GB runs only, where it would make the tests' simulated small machines plan around it.)
    double const fixed = partBytes*N + (u.scratchDir.empty() ? gridTotal*bCell : 0.) + ptsBytes;
    // ... and while the partition loop runs: its own grids only, the derived ones come after it (stdPeak)
    double const fixedLoop = partBytes*N + (u.scratchDir.empty() ? gridTotal*bCellDeposit : 0.) + ptsBytes;
    // per padded vertex: the thread's particle copy (alive while its triangulation grows) + the triangulation, and 6%
    // for the dummy points the padding test adds (stdBytesPerPaddedVertex, shared with the partition loop's look-ahead)
    double const perVertex = stdBytesPerPaddedVertex( u );

    /* The GPU deposit of the '_a' pass (averaged_interpolation_1.cc -> dtfe_metal_host.cc / dtfe_gpu_cuda.cu), the
       same for the sampled and the exact average (see StdTuneDensity's comment): per tetrahedron of a thread's own box
       the flat arrays (vertices 48, densities 16, velocities 48 when a velocity-derived '_a' grid is asked, the scalar
       16, volume, sample count, flat index and Morton key 4 each), the Morton order pairs (8), one array's sort copy
       (<= 48) and the work-item table (4) -- the arrays become the uploaded batch one at a time -- and per sub-grid
       cell the per-call deposit's device grid and its host copy (bGpuCell each). Charging 6.77 x 120 B per PADDED
       vertex would count every tetrahedron of the padding as extracted: only the ones touching the thread's own
       box are. */
    bool gpuVel = false, gpuScal = false;
    double bGpuCell = 0.;
    if ( metalActive and u.aField.selected() )
    {
        bool const den  = u.aField.density or u.aField.velocity_tweb;
        bool const grad = u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives();
        gpuVel  = u.aField.velocity or grad;
        gpuScal = u.aField.scalar or u.aField.scalar_gradient;
        bGpuCell = ( den ? 4. : 0. ) + ( u.aField.velocity ? 12. : 0. ) + ( grad ? 36. : 0. )
                 + ( u.aField.scalar ? 4. : 0. ) + ( u.aField.scalar_gradient ? 12. : 0. );
    }
    double const perTet = bGpuCell > 0. ? 80. + ( gpuVel ? 48. : 0. ) + ( gpuScal ? 16. : 0. ) + 8. + 48. + 4. : 0.;
    double const tetsPerOwn = kTetsPerVertex * kStdFaceTets;   // per particle of the thread's own box, + the ones crossing its faces
    /* The GPU deposit's host SNAPSHOT (PS's snapshotBytes above). The standard deposit
       resumes a failed command buffer from a checkpoint (dtfe_metal_host.cc runBatch, gpu_metal_retry.h GpuSnapshot):
       it keeps a host copy of the batch's device grids, taken every ~10 s of GPU work.
       Those grids are one thread sub-domain's (the per-call path: allocated under the device mutex, one batch at a
       time), in 8-B fixed-point words per 4-B component (4 B under
       DTFE_GPU_FLOAT_GRIDS=1), so the run holds ONE such copy, of the largest sub-grid:
       added once to the busiest partition's bytes (StdPlan::snapshot). (The partition CHECKPOINT itself,
       partition_checkpoint.h, writes the full grids from where they are, allocating nothing; the partition's own
       sub-grids are still alive then and already in its bytes.) */
    double const snapWordFactor = Env::flag( { "DTFE_GPU_FLOAT_GRIDS" } ) ? 1. : 2.;
    // the look-ahead (DTFE.cpp) runs with the GPU deposit when more than one partition is computed
    bool const lookAheadPossible = metalActive and bGpuCell > 0. and u.DTFE and u.method == 1
                                   and not u.stdNoOverlap and not u.redshiftConeOn;

    // the count grid of the particles (StdTuneDensity: one pass); without particles in memory, the uniform count
    auto const tuneStart = std::chrono::steady_clock::now();
    StdTuneDensity dens;
    if ( particles != nullptr and not particles->empty() )
        dens.build( *particles, u );
    double const countSec = std::chrono::duration<double>( std::chrono::steady_clock::now() - tuneStart ).count();
    // '--sample-points' evaluates ONE tessellation per partition (DTFE_parallel runs no thread split for it)
    bool const oneTessellation = not u.psSamplePointsFile.empty();
    double origin[NO_DIM], dxGrid[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        origin[d] = double( u.boxCoordinates[2*d] );
        dxGrid[d] = ( double( u.boxCoordinates[2*d+1] ) - origin[d] ) / double( u.gridSize[d] );
    }
    std::vector<size_t> const userSplit = ( not autoPart and u.partition.size() == size_t(NO_DIM) )
                                        ? u.partition : std::vector<size_t>( NO_DIM, 1 );

    AutoTuneStdInputs in;
    in.metalActive       = metalActive;
    in.autoPart          = autoPart;
    in.autoConc          = autoConc;
    in.N                 = N;
    in.ramBytes          = ramBytes;
    in.budgetWhy         = budgetWhy;
    in.budget            = budget;
    in.cores             = cores;
    in.gridTotal         = gridTotal;
    in.bCellDeposit      = bCellDeposit;
    for (int d=0; d<NO_DIM; ++d)
    {
        in.padFrac[d] = padFrac[d];
        in.origin[d]  = origin[d];
        in.dxGrid[d]  = dxGrid[d];
    }
    in.partBytes         = partBytes;
    in.nPts              = nPts;
    in.ptsBytes          = ptsBytes;
    in.ptsWalkBytes      = ptsWalkBytes;
    in.fixed             = fixed;
    in.fixedLoop         = fixedLoop;
    in.perVertex         = perVertex;
    in.bGpuCell          = bGpuCell;
    // the face pass (averaged_interpolation_1.cc avg1FacePass): its accumulator over the cells of the sub-domain the
    // '_a' interpolation runs on, one per concurrent sub-domain call, which together cover the partition
    in.faceCell = ( NO_DIM == 3 and u.faceGradient and u.aField.selected()
                    and ( u.aField.velocity_gradient or u.aField.velocity_divergence or u.aField.velocity_shear
                          or u.aField.velocity_vorticity or u.aField.velocity_vweb ) ) ? 9. * double(sizeof(double)) : 0.;
    in.perTet            = perTet;
    in.tetsPerOwn        = tetsPerOwn;
    in.snapWordFactor    = snapWordFactor;
    in.lookAheadPossible = lookAheadPossible;
    in.tuneStart         = tuneStart;
    in.countSec          = countSec;
    in.oneTessellation   = oneTessellation;
    in.userSplit         = userSplit;
    User_options pgOptions = u;                     // parallelGrid reads the partition's own grid from it
    pgOptions.verboseLevel = 0;                     // (and would say so for every plan)
    AutoTuneStdPlanner const planner{ in, u, report, message, dens, {}, std::move( pgOptions ) };
    return planner.chooseSplit();
#endif
}
