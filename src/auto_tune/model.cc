// Part of src/auto_tune.cc: included once, in order, by that file and compiled only as part of its object, never
// alone.
// The memory model (its own anonymous namespace): the model's constants, the sample-point file's count and
// bounds, and the bytes per vertex, cell and grid both planners share, up to autoTuneParallelInsert.

namespace {

// ---- The model's constants: one name and one source each ---------------------------------------------------------
// a finished PS-DTFE tessellation, per vertex: the 112 B vertex (with the uint64 ParticleID and the double volume
// accumulator) + 6.77 cells x 72 B (CGAL's cell) + allocator slack; a compiled sizeof probe and measured runs
constexpr double kPSVertexBytes = 666.;
// ... and a standard-DTFE one (no ParticleID; stdBytesPerPaddedVertex adds the double build's wider vertex data): the
// measured 648 B and the 8 B of the exact star sum's high limb (vertexData::_volAccHi, the standard binary only, in
// every build: the field is compiled in)
constexpr double kStdVertexBytes = 648. + 8.;
// '--parallel-triangulation' (TBB): the sorted point+payload copy and CGAL's own copy of it, per vertex in flight.
// Fitted: +4.4 GB max RSS for one partition of ~5.7M padded vertices inserted on 6 threads (TNG100 z=0, 512^3).
constexpr double kParInsertBytes = 800.;
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
// the derived grids are allocated (stdPeak). Fitted with kStdPostKeepGrids and kStdLookAheadFactor on a record of 27
// logged standard runs: at least -4.7%, mean +8.7%; the 6^3 face-gradient runs come out up to 41% over, accepted (see
// kStdPostKeepGrids) ...
constexpr double kStdPostKeep = 0.35;
// ... but a small partition stays whole: up to this many bytes are booked in full (tests/dtfe_auto_tune_check.sh E,
// 128^3: all of a 0.42 GB partition stayed, the derived grids came on top). Both fitted from the SUMMARY's 'before
// post-proc' line. The absolute floor: kStdPostKeepGrids raises it where the post-processing allocates more than 2 GB
constexpr double kStdPostKeepFloor = 1.e9;
// ... and the floor of what stays after the loop, as a share of the grids only the post-processing allocates
// (fixed - fixedLoop: the derived fields and the T-web stage; 0 with '--scratch-dir' or without derived fields).
// FITTED on ONE run (a 6^3 split with '--per-tet-gradient'): it needs >= 0.476 for -5%, and the 4^3 timing plan stays
// chosen up to 0.511. Its face-gradient twins peaked below 'fixed': which state a run gets depends on the machine, not
// on the gradient route
constexpr double kStdPostKeepGrids = 0.5;
// standard DTFE: what a look-ahead build holds beyond the tessellations and particle copy it is counted with
// (StdPlan::aheadMax). Not itemized, FITTED on the same 27-run record: the build's transients (at one thread per
// partition the TBB insert's sorted copies by the default rule, triangulation.cpp) and the count grid's under-count of
// the busiest partition at 4-6 threads. The record needs >= 1.258 for -5%; a run with sample points keeps its
// look-ahead under its budget up to 1.294
constexpr double kStdLookAheadFactor = 1.275;
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
// PS-DTFE LANE SHARE (memorySplit's plans only, see AutoTunePSPlanner::laneShareOn). Beside the first
// partition in flight, booked whole (partitionBytes), each other lane books this share of its own sub-grid state: a CPU
// deposit's sub-grids and words (times the CPU share of a mixed run's partitions), a face pass's state after the GPU
// hold. FITTED, not derived, on 9 TNG100-3-Dark z=0 runs at 512^3 (6^3, exact, sample points; mixed and all-GPU, 3 to
// 6 in flight): the all-GPU face run sets the ceiling, 0.562 at +15%; a small face run without points needs 0.489 or
// more. Never below the split's mean sub-grid over its largest (laneSpread).
constexpr double kPSLaneShare = 0.55;
// ... and the share of the GPU hold grids (gpuCellBytes) each other all-GPU lane books: one deposit holds the device at
// a time (ctxMutex), and a lane after its hold keeps its own sub-grids (bCellDeposit).
// FITTED on the 7 all-GPU TNG runs with points of the same record, whose ceiling is 0.36 (one at +15%); a flat 4-lane
// run outside the share needs 0.052 or more.
constexpr double kPSLaneShareHold = 0.30;
// the most partitions in flight the shares hold for: what was measured (mixed runs up to 6, all-GPU runs 4). Above it
// every lane is booked whole, and the dynamic concurrency stops there (DTFE.cpp psInFlightCap)
constexpr int kPSLaneShareMax = 6;
constexpr int kPSLaneShareMaxGpu = 4;
// under --scratch-dir the full grids below its threshold stay in RAM and 'fixed' books none of them: 3.2-3.8 GB at
// 512^3 in the production selection, here in units of the threshold (scratchMinBytes, 1 GB
// by default). The lane share books them (laneShareCover); with every lane booked whole that booking hid them
constexpr double kPSScratchRamGrids = 3.8;

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
    bool exactDeposit = false;     // the exact deposit (--exact; resolved by readOptions, the default with an '_a' field)
    bool caustics     = false;     // --ps-caustics
    bool haloRelease  = false;     // --ps-halo-release (sampled deposit: the released-tet multiplicity grid)
    bool faceGradient = false;     // --face-gradient: the face sums (both binaries)
    Field const *meanFlowField = nullptr;   // ... and the pass that then deposits the cell mean velocity unselected
                                            // (the '_a' field: autoTuneFaceVelocity; not set = either pass)
};

// '--face-gradient' in PS-DTFE: the pass deposits the face sums -- ps_interpolation.cc
// fx.faceS, faceN and faceP, 2 x NO_DIM x (3 + 1 + 1) components per sub-grid cell. Their accumulators are int64
// words in EVERY build, shared by the deposit threads (no private copies), made only after a GPU lane's hold of the
// device (psCpuFixedUnits; psFaceWalk fills them on the CPU) and converted to their Real grids
// (Quantities::face_velocity, face_coverage, face_momentum) after the moment grids. So the planner books them on
// their own (AutoTunePSInputs::faceR and fxFace) and not with the moment words, which only the single build holds
// as words.
inline bool autoTuneFaceOn(Field &f, AutoTunePS const &ps)
{
    return f.selected() and ps.phaseSpace and ps.faceGradient and ps.exactDeposit
           and ( f.velocity_gradient or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                 or f.velocity_vweb );
}

// ... and whether such a pass deposits the cell mean velocity, selected or not: the '_a' pass does (DTFE.cpp
// implies 'velocity' for aField, after the tuner has run); a face pass of the other field runs without it
// (ps_interpolation.cc haveVel goes by the field's own flag, and the face gradient reads the face sums only).
inline bool autoTuneFaceVelocity(Field &f, AutoTunePS const &ps)
{
    return autoTuneFaceOn( f, ps ) and ( ps.meanFlowField == nullptr or ps.meanFlowField == &f );
}

// ... and the face sums' Real grids, in bytes per cell (their words: 8 B per component, x 8 / sizeof(Real))
inline double autoTuneFaceBytes(Field &f, AutoTunePS const &ps)
{
    return autoTuneFaceOn( f, ps ) ? 2.*NO_DIM*(noVelComp + 2.)*double(sizeof(Real)) : 0.;
}

// ... and the regular part's twins: the stream average's own gradient, shear and
// vorticity a face pass keeps beside the mean flow's, each where the '_a' pass writes its mean-flow twin
// (Quantities::psStreamsTwinsCell; DTFE_setup sets User_options::psStreamsTwins from these flags), in Real bytes per
// cell. Made in the normalization, after the partition loop, beside the face sums; kept to the output. (The tuner runs
// before DTFE_setup folds the derived fields into the gradient, so the flags here are the user's.)
inline double autoTuneStreamsTwinBytes(Field &f, AutoTunePS const &ps)
{
    if ( not autoTuneFaceVelocity( f, ps ) ) return 0.;   // (the '_a' pass, the writer's; either pass when not set)
    double const R = double(sizeof(Real));
    return ( ( f.velocity_gradient ? double(noGradComp) : 0. ) + ( f.velocity_shear ? double(noShearComp) : 0. )
             + ( f.velocity_vorticity ? double(noVortComp) : 0. ) ) * R;
}

// ... and the two grids every face pass's normalization makes and keeps to the output
// (Quantities::psFaceGradientBegin, one Real per cell each): the stream average's divergence
// (velocity_divergence_streams, written as '.a_velDivStreams') and the mass flux divergence (mass_flux_divergence,
// '.a_massFluxDiv'). The '_u' pass makes them too, though the writer writes the '_a' pass's only. In the post phase
// both (autoTuneBytesPerCell's phase 1); at the normalization the divergence alone (postPhase false), since the
// deposit phase books the mass flux's grid already (its '+ R' below, which the deposit-phase figures --
// bCellDeposit, a partition's own sub-grids -- keep as they were).
inline double autoTuneFaceOutputBytes(Field &f, AutoTunePS const &ps, bool const postPhase)
{
    if ( not autoTuneFaceOn( f, ps ) ) return 0.;
    double const R = double(sizeof(Real));
    return postPhase ? 2.*R : R;
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
    if ( f.velocity or autoTuneFaceVelocity( f, ps ) ) b += 3.*R;
    if ( gradient ) b += 9.*R;
    if ( f.velocity_dispersion ) b += 6.*R;
    if ( f.scalar ) b += noScalarComp*R;
    if ( f.scalar_gradient ) b += noScalarGradComp*R;
    if ( needWeight ) b += R;                                                                // mass_weight
    if ( f.velocity_dispersion ) b += 3.*R + (densityGrid ? 0. : R);                         // disp_velocity (+ disp_weight)
    if ( ps.exactDeposit ) b += R;                                                           // the exact multiplicity
    if ( ps.haloRelease and not ps.exactDeposit ) b += R;                                    // the released multiplicity
    // (not the face sums: words in every build, booked on their own -- autoTuneFaceBytes above)
    return b;
}

// The same grids ONE BY ONE, in the order the deposit converts them after its loop (ps_deposit_params.h PSFxGrid:
// fxToReal walks g = 0, 1, ...), as Real bytes per sub-grid cell each (their words take twice that, and the entries add
// up to autoTuneFxMomentBytes): the one-domain model (auto_tune/ps_planner.cc oneDomainPeak) follows that conversion
// grid by grid.
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
    if ( f.velocity or autoTuneFaceVelocity( f, ps ) )
        g.push_back( 3.*R );                                                                  // PS_FX_MOM
    if ( f.velocity_dispersion ) g.push_back( 6.*R );                                         // PS_FX_M2
    if ( gradient ) g.push_back( 9.*R );                                                      // PS_FX_GRAD
    if ( needWeight ) g.push_back( R );                                                       // PS_FX_MOMW
    if ( ps.exactDeposit ) g.push_back( R );                                                  // PS_FX_SV
    if ( f.velocity_dispersion )
    {
        g.push_back( 3.*R );                                                                  // PS_FX_DV
        if ( not densityGrid ) g.push_back( R );                                              // PS_FX_DW
    }
    if ( f.scalar ) g.push_back( noScalarComp*R );                                            // PS_FX_SCAL
    if ( f.scalar_gradient ) g.push_back( noScalarGradComp*R );                               // PS_FX_SG
    if ( ps.haloRelease and not ps.exactDeposit ) g.push_back( R );                           // PS_FX_RM
    // (the face sums' words are converted after these, in every build: oneDomainPeak walks them itself)
    return g;
}

// What the input reader frees and macOS's allocator keeps, in bytes: the reader's flat arrays (gadget_reader_HDF5.cc:
// the positions, masses and velocities 'readParticleData' asks for, the scalars, the Lagrangian positions and the
// IDs), freed one by one as Read_data::transferData builds the particles, and with '--ps-lagrangian-input' the IC file's
// coordinates and IDs and the by-ID table (input_output.cc; a sort index of 8 B instead of the table's 12 when the
// IDs are not contiguous). Each block under kAllocKeepBytes stays in the footprint until a request of about half its
// size or more takes it over. A run with full-size grids allocates them right away; a --grid-window run does not --
// its window grids, the partitions' particle copies, the tessellations (TBB's own pools) and the GPU's buffers are all
// far smaller or elsewhere -- so the whole residue is still there when the first wave of partitions tessellates. Later
// partitions' large allocations take the blocks over, so the residue is gone a few partitions in.
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
    bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                            or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                            or f.velocity_vweb or f.scalar or f.scalar_gradient;
    double b = 0.;
    if ( needWeight ) b += R;                                                   // mass_weight
    if ( f.velocity_dispersion ) b += R + 3.*R;                                 // disp_weight + disp_velocity
    b += autoTuneFaceBytes( f, ps );                                            // face_velocity, _coverage, _momentum
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
    // on top of every resident grid.
    if ( f.velocity_tweb ) b += 8. + 16.032 + 6.*R;
    return b;
}

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
    // dispersion allocates the velocity grid too -- EXCEPT in PS-DTFE (volume-weighted velocity moments), where it
    // carries its own disp_velocity mean instead (counted in the deposit-phase PS block below)
    if ( f.velocity or (f.velocity_dispersion and not ps.phaseSpace) )
        b += 3.*R;
    if ( gradient ) b += 9.*R;
    if ( f.velocity_dispersion ) b += 6.*R;
    if ( f.scalar ) b += noScalarComp*R;
    if ( f.scalar_gradient ) b += noScalarGradComp*R;
    // --face-gradient, the deposit phase: PS-DTFE's merged face sums (Quantities::face_velocity,
    // face_coverage and face_momentum, 2 x NO_DIM x (3 + 1 + 1) Real per cell, freed in the normalization:
    // autoTuneFaceBytes) and the cell mean velocity the mean-flow gradient needs, written or not. (Standard DTFE's face
    // pass holds 9 doubles per cell of the sub-domain it interpolates, not of the full grid: the standard planner books
    // it per sub-domain, AutoTuneStdInputs::faceCell.)
    if ( phase == 0 and autoTuneFaceOn( f, ps ) )
        b += autoTuneFaceBytes( f, ps ) + ( ( f.velocity or not autoTuneFaceVelocity( f, ps ) ) ? 0. : 3.*R )
             + R;                                                           // (+ the mass flux's output grid)

    if ( phase == 1 )   // the derived fields, the regular part's twins and a face pass's two output grids: allocated
                        // only after the deposit loop
        b += autoTuneDerivedBytes( f ) + autoTuneStreamsTwinBytes( f, ps ) + autoTuneFaceOutputBytes( f, ps, true );
    else if ( ps.phaseSpace )
    {
        // ---- deposit-phase PS weight grids (freed in psNormalizeMoments, before the derived fields) ----
        bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                                or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                                or f.velocity_vweb or f.scalar or f.scalar_gradient;
        // The weight carries Eulerian-volume shares (the volume-weighted velocity moments), which are not the
        // density's mass sums, so it has its own grid even when density is selected.
        if ( needWeight ) b += R;                                                   // mass_weight
        // dispersion stays MASS-weighted beside the volume-weighted velocity, so it carries its OWN
        // mean+normalizer (ps_interpolation.cc: dispNeedsOwnWeight)
        if ( f.velocity_dispersion ) b += R + 3.*R;                                 // disp_weight + disp_velocity
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

// The largest single pre-sized full grid of a field set, in bytes per cell: the deposit-phase grids psReserveGrids
// sizes before the partition loop (the merge-grown weights and face sums of autoTuneMergeGrownBytes come at the first
// merge, the derived fields after the loop). Under --scratch-dir ScratchAlloc::zeroResize fills each of them with zeros
// in anonymous memory before its file is mapped over it (ZeroFillScope, macOS), so the largest one is a start-up peak
// of the run: particles + the read blocks kept + this grid (AutoTunePSPlanner::reserveZeroFillPeak).
inline double autoTuneLargestGridBytesPerCell(Field &f, AutoTunePS const &ps)
{
    if ( not f.selected() ) return 0.;
    double const R = double(sizeof(Real));
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    double comps = 0.;
    if ( f.density or f.velocity_tweb or f.velocity_std ) comps = 1.;
    if ( f.velocity or (f.velocity_dispersion and not ps.phaseSpace) ) comps = std::max( comps, 3. );
    if ( gradient ) comps = std::max( comps, 9. );
    if ( f.velocity_dispersion ) comps = std::max( comps, 6. );
    if ( f.scalar ) comps = std::max( comps, double(noScalarComp) );
    if ( f.scalar_gradient ) comps = std::max( comps, double(noScalarGradComp) );
    if ( ps.phaseSpace ) comps = std::max( comps, 1. );          // the counts, flags and masks (autoTunePSCountBytes)
    return comps * R;
}

/* A Lagrangian partition's padding on one axis, in units of the partition's own width, for an n^3 split of N
   particles: with '--ps-lagrangian-padding <k>' (the default, k = 8) k mean spacings -- k n /
   N^(1/NO_DIM) partition widths, at most one period (n widths) on a periodic run -- and at least 4.2
   '--ps-alpha-shape' cuts on a non-periodic one, with no floor; with '--ps-lagrangian-padding 0' the Eulerian '--padding'
   fraction floored at 'floorFrac' of a partition (DTFE.cpp's PSPartitionPlan). One definition for the models of
   auto_tune.cc and auto_tune/ps_planner.cc. */
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
   cached n^3 partition files. Reads only each file's uncompressed header (magic, length, descriptor) and the
   chunked body's marker after it, and compares the format line and the lines that name the snapshot and the
   build -- exactly the text tessellation_cache.h writes (FORMAT_VERSION and CHUNK_MARKER there, KEEP IN STEP;
   cache_key_lines.h) -- plus the periodicity and length unit; the rest of a partition's key follows from the
   split. A file of another format misses in the run, so it does not steer the split either. Read-only. */
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
        char marker[8];
        if ( not f.read( &desc[0], len ) or not f.read( marker, 8 ) or std::string( marker, 8 ) != "DTFECHK1" )
            continue;
        // the descriptor's first line (tessellation_cache.h makeDescriptor)
        static char const FORMAT[] = "format=3\n";
        if ( desc.compare( 0, sizeof(FORMAT) - 1, FORMAT ) != 0 ) continue;
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
// of 1 + 2 x its padding in partition widths (autoTunePadWidths). One loop for the models of auto_tune.cc and
// auto_tune/ps_planner.cc.
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

// Whether a tessellation of 'padded' points is inserted in parallel (CGAL's TBB insert, whose sorted copies the
// planners book at kParInsertBytes a point) while 'builds' tessellations are built at once: triangulation.cpp's rule
// (delaunayTriangulation) at the planned count. On the default rule (at least User_options::parallelInsertMinPoints),
// with '--parallel-triangulation' or with DTFE_TRIANGULATION_THREADS set (both: any size; a count of 1 keeps the insert
// sequential, and a count beats a run mode's lowered thread budget), where the machine's threads divided by the builds
// leave 2 or more. The OpenMP cap is not followed: at 1 thread it books copies the run does not make (the safe side).
inline bool autoTuneParallelInsert(User_options const &u, double const padded, int const builds)
{
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    long const pinned = Env::integer( "DTFE_TRIANGULATION_THREADS", 0 );
    bool const pinSet = Env::isSet( "DTFE_TRIANGULATION_THREADS" );
    if ( not ( u.psParallelTriangulation or pinSet ) or pinned == 1 ) return false;
    int const hardware = ( pinned > 0 and runModeState().budgetLowered )
                         ? std::max( 1, int( std::thread::hardware_concurrency() ) ) : dtfeHardwareThreads();
    return hardware / std::max( 1, builds ) >= 2
           and ( u.parallelTriangulationForced or pinSet
                 or padded >= double( User_options::parallelInsertMinPoints ) );
#else
    (void)u; (void)padded; (void)builds;
    return false;
#endif
}

}  // namespace
