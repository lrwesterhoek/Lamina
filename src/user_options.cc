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

/* User_options implementation: defaults, command-line/config parsing, validation, and the run-configuration printout. */


#include <algorithm>   // std::max
#include <fstream>
#include <cstdio>
#include <cstring>     // std::strchr, std::strerror
#include <cmath>       // HUGE_VAL
#include <cerrno>      // errno (applyRunMode: gentle's nice)
#include <sys/stat.h>
#include <sys/resource.h>   // getpriority / setpriority (applyRunMode)
#ifdef __APPLE__
#include <sys/sysctl.h>     // hw.perflevel0.logicalcpu (applyRunMode)
#endif
#include "user_options.h"
#include "env_knobs.h"    // DTFE_FACE_GRADIENT (the default gradient route), DTFE_EXACT_AVERAGE (the '_a' average's pin), DTFE_GPU (the device's)
#include "run_mode.h"     // the run mode's process-wide state (applyRunMode)

using namespace std;

// The builds whose grid work has a GPU path: the guards the deposits compile their GPU code under
// (averaged_interpolation_1.cc DTFE_GPU_ACTIVE: standard DTFE's GPU macro, 3D, not MY_SCALAR; ps_interpolation.cc:
// PS-DTFE's GPU macro, 3D). The program's pick of the device (gpuPathRuns) and the header's device and GPU share
// lines read it, so a change of either guard belongs here too.
#if ( defined(PHASE_SPACE) && defined(PS_GPU) && NO_DIM==3 ) \
    || ( !defined(PHASE_SPACE) && defined(DTFE_GPU) && NO_DIM==3 && !defined(MY_SCALAR) )
#define GPU_PATH_BUILT
#endif


#ifndef INPUT_FILE_DEFAULT
#define INPUT_FILE_DEFAULT 101
#endif
#ifndef OUTPUT_FILE_DEFAULT
#define OUTPUT_FILE_DEFAULT 101
#endif
#ifndef MPC_UNIT
#define MPC_UNIT 1.
#endif

// Initializes every option to its default before the command line/config overrides it.
User_options::User_options()
{
    periodic = false;
    periodicInput = false;
    boxCoordinates.assign(0.);
    userGivenBoxCoordinates = false;
    inputFileType = INPUT_FILE_DEFAULT;     // default input file type (see Makefile)
    outputFileType = OUTPUT_FILE_DEFAULT;   // default output file type (see Makefile)
    readParticleData.assign( 10, 0 );
    readParticleSpecies.assign( 10, 0 );
    readParticleData[0] = 1; readParticleData[1] = 1; readParticleData[2] = 1;
    readParticleSpecies[1] = 1;
    
    regionOn = false;
    regionMpcOn = false;
    psWindowOn = false;
    for (int d = 0; d < NO_DIM; ++d) { psWindowLo[d] = 0; psWindowDims[d] = 0; }
    
    partitionOn = false;
    partitionGiven = false;
    partNo = -1;
    maxConcurrent = 0;    // 0 = build as many concurrent triangulations as threads (no memory cap)
    psParallelTriangulation = true;       // the parallel insert for tessellations of at least parallelInsertMinPoints (1e6) points where more than one core is free; --sequential-triangulation turns it off, --parallel-triangulation forces it
    parallelTriangulationForced = false;  // --parallel-triangulation
    sequentialTriangulation = false;
    autoTuneReport = false;
    maxConcurrentOn = false;  // when neither --partition nor --max-concurrent is given, the auto-tuner picks them (auto_tune.h)
    runMode = RunMode::Balanced;      // resolveRunMode: --run-mode, else DTFE_RUN_MODE, else the binaries' default (balanced; performance inside a batch job)
    runModeSource = "";
    coreBudget = 0;           // set by applyRunMode (the mode's threads)
    coreBudgetOwn = 0;        // ... and before a '--max-concurrent' raise (autoTuneGpuShare's threads under 'mixed')
    coreBudgetNote = "";
    priorityNote = "";
    psGpuShareSpec = -1;      // --gpu-share (PS-DTFE): -1 = auto (the default), -2 = mixed, else per mille as given
    psGpuShareText = "auto";
    psGpuShareMille = 1000;   // resolved by DTFE.cpp psResolveGpuShare before the auto-tuner
    psGpuShareSource = "";

    paddingOn = false;
    paddingMpcOn = false;
#ifdef PHASE_SPACE
    paddingParticles = Real(5.);      // PS-DTFE: 5 mean spacings (its periodic Lagrangian copies, the k-spacings rule of its partitions)
#else
    paddingParticles = Real(15.);     // standard DTFE: 15 mean spacings, complete in the emptiest voids at z = 0 (5 is not: the '--padding' help)
#endif
    testPaddedBoundaries = true;
    
    // density options
    method = 1;
    noPoints = 100;
    noPointsOn = false;
    useMetal = false;     // the CPU; checkGpuOptions resolves the device ('--gpu', DTFE_GPU, the program's pick), so a library caller that skips readOptions keeps the CPU
    gpuAlias = false;
    deviceSource = "";
    gpuPicked = false;
    deviceLine = "";
    exactAverage = false; // standard DTFE: readOptions resolves the '_a' average (resolveAverageMode: exact by default); a library caller that skips readOptions keeps the sampled one
    exactFlag = false;    // --exact
    sampledFlag = false;  // --sampled
    averageModeSource = "";
    faceGradient = false; // the velocity gradient: the face gradient is both binaries' default where it applies
                          // (checkGpuOptions), unless --per-tet-gradient
    perTetGradient = false;
    faceGradientSource = "";
    psSamplePointsFile = "";          // point-evaluation mode off unless --sample-points is given (both binaries)
    psServe = false;                  // interactive point-evaluation server off unless --serve is given
    psPerStream = false;
    psPerStreamIds = false;
    psPtsDenGrad = false;
    psPtsVelGrad = false;
    psPtsScalar = false;
    psPtsLagrangian = false;
    scalarDataset = "";               // the reader's own scalar (gas temperature etc.), if any
    serveResident = 0;                // composite --serve: resident partitions chosen automatically
    psServeProgress = false;          // --serve-progress
#ifdef PHASE_SPACE
    psAvgSubsamples = 1;  // nSub for PS-DTFE's sampled '_a' deposit: one point per cell (an exact run without a count keeps kPsExactSliverSubsamples: resolveAverageMode)
    psUseMetal = false;   // the CPU; checkGpuOptions resolves the device, so a library caller that skips readOptions keeps the CPU
    psCaustics = false;               // no '.caustic' fold-flag grid unless --ps-caustics
    psCausticCusps = false;   // --ps-caustic-cusps: A3 indicator, opt-in (finite-difference estimator)
    psHaloRelease = Real(0.);         // 0 = deposit every tetrahedron by bbox rasterization (no release)
    psAlphaShape = Real(3.);          // non-periodic: alpha shape of 3 mean spacings (interior tets stay below ~0.9)
    psLagPadding = Real(8.);          // --ps-lagrangian-padding: partitions padded by 8 mean spacings; 0 = the Eulerian --padding fraction
    psExactDeposit = false;           // resolved by readOptions (resolveAverageMode: the exact deposit by default for a run with an '_a' field); the nSub^3 sub-sampled deposit without it
    psVertexMass = false;             // rho_bar * V_lag tet masses unless --ps-vertex-mass
#endif
    averageDensity = Real(-1.);

    redshiftConeOn = false;

    // additional options
    DTFE = true;
    NGP = false;
    CIC = false;
    TSC = false;
    PCS = false;
    SPH = false;
    Voronoi = false;
    interlace = false;
    MpcValue = Real(MPC_UNIT);
    extensive = false;
    approxPSD = false;
    lambda_th = Real(0.3);   // '--lambda-th' (its default_value, which po::notify writes, is the same)
    hubbleParam = Real(-1.);
    scaleFactor = Real(-1.);
    verboseLevel = 3;
    randomSample = Real(-1.);
    poisson = 0;
    
    
    // internal variables
    paddedBox.assign(0.);
#ifdef PHASE_SPACE
    // these three are only set true for per-partition sub-calls in the PS-DTFE partition path
    psSuppressGridStats = false;
    psDeferNormalization = false;
    psLagAlphaRadius = 0.;   // set from --ps-alpha-shape once the Lagrangian cloud is known (DTFE.cpp)
    psOrderedMerge = true;   // the staged ordered merge, the default (--ps-merge completion-order: off)
    psOrderedMergeFlag = false;   // a given --ps-merge index-order, or --resume (folded into psOrderedMerge after parsing)
    psCompletionOrderMergeFlag = false;   // --ps-merge completion-order
    psSpeedSplit = false;    // set by the auto-tuner's speed split (auto_tune.h)
    psNoOverlap = false;     // --no-overlap
    psLookAheadOffBudget = false;   // set by the auto-tuner when the look-ahead does not fit the budget
    psDynamicConcurrency = false;   // set from the auto-tuner's report (DTFE.cpp)
    psReserveEnd = 0.;
    psReserveRecords = 0.;
    psReserveFirstMerge = 0.;
    psBudgetBytes = 0.;
    psParkRoom = HUGE_VAL;          // set from the auto-tuner's report (DTFE.cpp): no budget known = the free-memory rule alone
    psLookAheadRoom = HUGE_VAL;     // set from the auto-tuner's report (DTFE.cpp): no budget known = no budget limit
    psPartitionBytes = 0.;   // set from the auto-tuner's report (DTFE.cpp): the partition memory gate
    psOnceBytes = 0.;        // ... the part of it that one partition at a time holds
    psInFlightCap = 0;       // ... the lane share's limit on the partitions in flight (0: none)
    psUseSubgrid = false;
    psTessellationPrebuilt = false;   // set only by psInterpolatePrebuilt (the partition look-ahead)
    psMayClearDT = false;   // set only by the DTFE_interpolation overload that owns its DT
#endif
#ifndef PHASE_SPACE
    stdTessellationPrebuilt = false;   // set only by dtfeInterpolatePrebuilt (the standard look-ahead)
    stdNoOverlap = false;              // --no-overlap
    stdLookAheadOffBudget = false;     // set by the auto-tuner when the look-ahead does not fit the budget
    stdLookAheadRoom = HUGE_VAL;       // set from the auto-tuner's report (DTFE.cpp): no budget known = no budget limit
    stdLookAheadCounted = 0.;          // set by the auto-tuner (its counted look-ahead)
#endif
    psResume = false;        // --resume (both binaries)
    psNoCheckpoint = false;  // set under --interlace (below)
    userDefinedSampling = false;
    noProcessors = 1;
    totalTime = Real(0.);
    programOptions = "";
}



// The checkpoint key's command line (argv[1..]), the binaries' short letters in their long names, so one build gives
// one key for both spellings: '-i 105' = '--input-format 105', '-g64' = '--grid 64'. Any other token unchanged.
static std::vector<std::string> canonicalCommandTokens(int argc, char *argv[])
{
    auto const digits = [](std::string const &v) { return not v.empty() and v.find_first_not_of( "0123456789" ) == std::string::npos; };
    static char const *const longName[][2] = { { "i", "input-format" }, { "g", "grid" }, { "o", "output-format" },
                                               { "f", "field" }, { "c", "config" }, { "v", "verbose" },
                                               { "p", "periodic" } };
    std::vector<std::string> out;
    for (int i = 1; i < argc; ++i)
    {
        std::string const t = argv[i];
        // a letter alone, or with a number attached ('-g64', '-i105', '-o101', '-v2')
        if ( t.size() >= 2 and t[0] == '-' and t[1] != '\0' and std::strchr( "igofcvp", t[1] ) != nullptr
             and ( t.size() == 2 or ( std::strchr( "igov", t[1] ) != nullptr and digits( t.substr( 2 ) ) ) ) )
        {
            for ( auto const &l : longName )
                if ( t[1] == l[0][0] ) out.push_back( std::string( "--" ) + l[1] );
            if ( t.size() > 2 ) out.push_back( t.substr( 2 ) );
        }
        else out.push_back( t );
    }
    return out;
}


// Register every program option. Each is declared ONCE, in one group below; every group goes into 'allOptions' (what the
// command line and a '--config' file are parsed against: a missing one is an unknown option) and into 'visibleOptions'
// ('--full-help', through helpInformation). Groups in printed order: Input; Grid and fields; Cell average; Device and
// machine use; Split, memory and resume; Region; Points and server; Phase space (PS-DTFE only); Other estimators; Other;
// Developer options; Help. 'hidden' holds the two positional file names (never shown).
void User_options::addOptions(po::options_description &allOptions,
                              po::options_description &visibleOptions,
                              po::positional_options_description &p)
{
    po::options_description inputOptions("Input");
    inputOptions.add_options()
            ("input-format,i", po::value< std::vector<int> >()->multitoken(), "give the type of the input file -- the file itself is the first positional argument (101=binary gadget snapshot in one file or several, 105=gadget HDF5 file, see documentation for more options). If present, a 2nd argument gives the data to be read from file (1=positions, 2=weights (the masses), 4=velocities, ..., 2^n=the n+1 data) - e.g. to read positions, masses and velocities insert 1+2+4=7. If present, a 3rd argument gives the particle species to be read from file (1=1st species, 2=2nd species, ..., 2^n=the n+1 species) - e.g. to read the data of species 2,3 and 4 insert 2+4+8=14.")
            ("mpc-unit", po::value<Real>(&(this->MpcValue)), "specify the value of 1Mpc in units of the input particle position data. [DEFAULT value is the one given in the 'MPC_UNIT' variable in the Makefile.]")
            ("periodic,p", "particle data is in a periodic box with box coordinates given by option '--box' or read from input file.")
            ("box", po::value< std::vector<Real> >()->multitoken(), "the coordinates of the box encompassing all the particles. It needs 6 arguments for 3D (4 for 2D) which give 'x_left', 'x_right', 'y_left', 'y_right', etc... (where 'x_left' is the left box coordinates along x-direction). For example '--box 0 1 0.5 1.5 0 10'.")
#ifdef PHASE_SPACE
            ("ps-lagrangian-input", po::value<std::string>(&(this->lagrangianInputFilename)), "specify the HDF5 snapshot (one file, or for a snapshot split over several files a name with '%i' for the file number) holding the Lagrangian (initial condition) positions for PS-DTFE. It is read only when the main input file has no 'InitialCoordinates' dataset: a main input file that has one uses its own, whether this option is given or not. The 'Coordinates' of the species read are taken from this file and matched to the main input file's particles by 'ParticleIDs', so their order does not matter; both files must hold the same number of particles of those species, and '--mpc-unit' converts both. Either way the main input must be a Gadget HDF5 snapshot read as type 105: no other reader gives 'InitialCoordinates' or the 'ParticleIDs' the match needs. Without this file and without 'InitialCoordinates' the run stops with an error.")
            ("ps-vertex-mass", po::bool_switch(&(this->psVertexMass)), "PS-DTFE only: assign each tetrahedron its CHART-INDEPENDENT mass m_tet = sum over its 4 vertices of (particle mass)/(number of finite tetrahedra incident to that vertex), instead of the default m_tet = rho_bar * V_lag. The default is exact when the Lagrangian positions are the UNPERTURBED lattice, but when '--ps-lagrangian-input' holds evolved IC positions (e.g. TNG's z=127 snapshot) the IC configuration already carries the density contrast delta_ic = D(z_ic)/D(z) * delta(z): tetrahedra in future-overdense regions are pre-compressed and rho_bar*V_lag under-weights them, filtering EVERY density mode by 1 - D(z_ic)/D(z) (measured -16.5\% at z=20, -3\% at z=2 for TNG's z_ic=127; velocity fields are unaffected, but the divergence-density slope steepens by the inverse factor). Splitting each particle's mass equally among its incident tetrahedra makes the mass follow the particles, removing the bias at any z_ic; total mass is conserved exactly, at the cost of small-scale degree noise (per-tet masses vary with the local Delaunay connectivity). CPU, GPU and the exact deposit ('--exact'); '--sample-points' and '--serve' follow it too. The '--ps-halo-release' kept/released classification stays geometric (V_lag/V_eul), so it is identical with and without this flag.")
#endif
            ("padding", po::value< std::vector<Real> >()->multitoken(), "give the size of the padding need to make sure that the Delaunay triangulation fully covers the region of interest. There are two ways to give the padding size:\n"
                    "  1) \t by giving one value which is the average number of particles that will be copied along each face of the region of interest. The actual computation uses all the particle that are within 'padding number' * 'particle grid spacing' distance from the region of interest. For example '--padding 5' will add an average of 5 particles on both the left and right sides for each dimension.\n"
                    "  2) \t by giving the size of the padding for each face of the box of interest. This size is given with respect to the box length along each coordinate (i.e. '-padding 0.1 0.2 0.5 0.5 0.1 0.1' means that box will be padded with '0.1*x box length' on the left of the x-coordinate and by '0.2*x box length' on the right of the x-coordinate, similar for the y and z dimensions).\n"
                    "DEFAULT: 15 mean particle spacings in standard DTFE, 5 in PS-DTFE. Standard DTFE: 5 is too little for the emptiest voids of a cosmological box: grid cells beyond the tessellation are reported (dummy-point test, '*.badGridPoints*' files), but point values ('--sample-points', '--serve') within a few spacings of a face come from the wrong tetrahedra while their coverage still reads 1 (up to 4% off on demo/z0_64 at z = 0). 15 is complete at z = 0 on TNG50/100/300-3-Dark (10 is not); scripts/run_dtfe.sh, dtfelib's Estimator (standard server) and the launcher's standard exact zoom pass it too. Each partition and each thread's sub-box of a standard run carries its own padding, so the cost grows with their number. PS-DTFE pads in Lagrangian space by the same fraction of the box (at least 10% of it), and its partitions by '--ps-lagrangian-padding'.")
            ("padding-mpc", po::value< std::vector<Real> >()->multitoken(), "give the size of the padding need to make sure that the Delaunay triangulation fully covers the region of interest. Similar to option 'padding' choice '2)' with the difference that the padding size is given in Mpc and not box lengths.")
            ("output-format,o", po::value< int >(&(this->outputFileType)), "give the type of the output file -- the output root is the second positional argument (101=binary file, 111=text file, see documentation for more options).")
            ("reader-options", po::value< std::vector<std::string> >( &(this->additionalOptions) )->multitoken(), "strings for the input readers (the binary reader's mass, value and threshold columns; the Gadget HDF5 reader's HI fraction files), supplied in a very simple way. Each additional option will be stored as a string in 'User_options.additionalOptions' (this variable is a vector of strings).")
            ("particle-fraction", po::value<Real>(&(this->randomSample)), "generates a random subsample of the input data. The size of the subsample is given by value supplied to the option (with values from 0. to 1.). Only this random subsample of the full data set will be used in any further computations. For example '--particle-fraction 0.1' will keep only 10\% of the data set for further computations.")
            ("poisson", po::value<size_t>(&(this->poisson)), "generate the particle positions randomly. The argument gives the root 3 in 3D (and root 2 in 2D) of the random number of particles. The particles have the same weight and are in a box of size unity. For example '-poisson 256' will generate 256^3 particles in 3D, while only 256^2 particles in 2D.")
            ("scalar-dataset", po::value<std::string>(&(this->scalarDataset)), "Gadget HDF5 input (types 105/106) only: read the per-particle dataset '/PartTypeN/<NAME>' (one value per particle, used as is) as the scalar field -- e.g. 'Potential', a halo membership flag, or 'Values' written by dtfelib.Estimator.from_arrays. It is interpolated by '--pts-scalar' at points and by the 'scalar' fields on the grid. Replaces the reader's default scalar (the gas temperature).")
            ;
    
    
    po::options_description gridFieldOptions("Grid and fields");
    gridFieldOptions.add_options()
            ("grid,g", po::value< std::vector<size_t> >(&(this->gridSize))->multitoken(), "choose grid size along each direction (e.g. '-g 256' for a 256^3 grid; '-g 256 128 512' for different gridsize along each direction).")
            ("field,f", po::value<std::vector<std::string> >()->multitoken(), "specify which field is to be interpolated to grid - e.g. '-f density'. You can specify multiple fields at a time. You can choose to output the field value at the sampling point or the volume averaged field value inside the grid cell associated to the sampling point (field names end with '_a' with 'a' standing for averaged). Available options are:\n"
                    "  density = \tcompute the density at the sampling point position.\n"
                    "  density_a = \tcompute the volume averaged density inside the sampling cell associated to the sampling point [DEFAULT choice if none provided]. Each of the options below have also a '*_a' version which will be left out to minimize the help messages.\n"
#ifdef VELOCITY
                    "  velocity = \tcompute the velocity at the sampling point position (use 'velocity_a' to get the averaged velocity inside the sampling cell).\n"
                    "  gradient = \tcompute the velocity gradient at the sampling point position (use 'gradient_a' to get the averaged velocity gradient inside the sampling cell).\n"
                    "  divergence = \tcompute velocity divergence at the sampling point position (use 'divergence_a' to get the averaged velocity divergence inside the sampling cell).\n"
                    "  shear = \tcompute velocity shear at the sampling point position (use 'shear_a' to get the averaged velocity shear inside the sampling cell).\n"
                    "  vorticity = \tcompute velocity vorticity at the sampling point position (use 'vorticity_a' to get the averaged velocity vorticity inside the sampling cell).\n"
                    "  velocityStd_a = \tcompute velocity standard deviation inside the sampling cell (NOTE: there is no 'velocityStd' of this option and this option works only with standard DTFE's averaging method 2 '--average-method 2'; PS-DTFE's spread between streams is 'dispersion').\n"
#ifdef PHASE_SPACE
                    "  dispersion = \t[PS-DTFE only] mass-weighted multi-stream velocity dispersion. Writes the trace sigma^2 = sum(rho_s |v_s-<v>|^2)/sum(rho_s) to '.velDisp' and the full symmetric tensor sigma_ij to '.velDispTensor'. ~0 in single-stream regions (cold void interiors), large in multi-stream regions (walls/filaments). Use 'dispersion_a' for the volume-averaged version.\n"
                    "  (PS-DTFE: the velocity moments -- the velocity, the gradient and the fields built on it, the "
                    "scalars -- are volume-weighted: each stream counts with the Eulerian volume it fills in the "
                    "cell, so a cell holds the volume average of the field, the standard-DTFE '_a' convention. The "
                    "dispersion stays mass-weighted: it is a second moment of the phase-space distribution. "
                    "'gradient' and the 'divergence', 'shear', 'vorticity' and 'vweb' built on it "
                    "are, by default with the exact deposit, the gradient of the volume-weighted MEAN velocity from "
                    "exact face sums -- "
                    "the quantity standard DTFE gives (see '--face-gradient'). Otherwise ('--per-tet-gradient', the "
                    "sampled deposit, '--ps-halo-release', 2D) they are, in a multi-stream "
                    "cell, the average over the streams of each stream's OWN velocity gradient, weighted like the "
                    "velocity, NOT the gradient of the mean velocity. The two agree in "
                    "single-stream cells; in folds they differ (an inverted stream inside a wall diverges while the "
                    "mean flow converges), so the box mean of that divergence is not 0 and its V-web finds few "
                    "filaments and nodes (TNG100-3-Dark z = 1: 1.78% and 0.005%, against 6.62% and 0.20% for "
                    "standard DTFE). dtfelib's FieldSet.load(name, mean_field=True) gives the gradient of the mean "
                    "velocity either way: the stored grids of a face-gradient run, else the FFT derivatives of "
                    "'.a_vel'.)\n"
#endif
#endif
#ifdef SCALAR
                    "  scalar = \tcompute scalar quantities at the sampling point position (use 'scalar_a' to get the averaged field components inside the sampling cell).\n"
                    "  scalarGradient = \tcompute the gradient of the scalar quantities at the sampling point position (use 'scalarGradient_a' to get the averaged field gradient inside the sampling cell).\n"
#endif
#ifdef VELOCITY
                    "  tweb = \tT-web cosmic web classification (0=void, 1=wall, 2=filament, 3=node) from the eigenvalues of the tidal tensor, FFT-Poisson-solved from the DENSITY grid (use 'tweb_a' for the volume-averaged density). Also outputs the eigenvalues. Requires the density field (enabled automatically) and a periodic box; 2D builds too (two eigenvalues: 0=void, 1=filament, 2=node). Computed from the RAW density grid, with no smoothing, so the classes depend on the cell size and on the estimator's noise at that scale: on TNG100-3-Dark at z = 1 (512^3) the raw T-web finds 37.6% void with standard DTFE and 33.8% with PS-DTFE, both 46.8% after a Gaussian smoothing of one cell. Quote web fractions at a stated smoothing length, recomputed from the density grid; no option smooths it here, and plot_PS_DTFE.py smooths only the eigenvalue maps it draws, not the classes. See also --lambda-th.\n"
                    "  vweb = \tV-web cosmic web classification (0=void, 1=wall, 2=filament, 3=node) from velocity shear tensor eigenvalues (use 'vweb_a' for volume averaged). Also outputs eigenvalues.\n"
#endif
            )
            ("lambda-th", po::value<Real>(&(this->lambda_th))->default_value(Real(0.3), "0.3"), "eigenvalue threshold for T-web/V-web cosmic web classification. Grid cells with eigenvalues above this threshold are classified as collapsing along that axis. [DEFAULT: 0.3, the production value, which scripts/run_dtfe.sh and run_ps_dtfe.sh pass too; '--lambda-th 0' classifies by the eigenvalues' signs alone]")
            ("mean-density", po::value<Real>(&averageDensity), "supply a value to be used to scale the density. If none is supplied, the average density will be used for this task.")
            ("hubble", po::value<Real>(&(this->hubbleParam)), "Hubble parameter h for T-web/V-web normalization (H=100 h km/s/Mpc). If not specified, the value is read from the simulation file header.")
            ("scale-factor", po::value<Real>(&(this->scaleFactor)), "scale factor a of the snapshot, used to convert Gadget u-velocities (u = v_pec/sqrt(a)) to peculiar km/s in the V-web normalization. If not specified, the value is read from the simulation file header ('Time'); pass it explicitly for inputs whose header lacks it.")
            ("scalars-extensive", "specify that all the fields under 'scalar fields' are extensive quantities. If this option is missing than the code treats the variables as intensive fields. This option is important only when using the TSC or SPH interpolation methods applied to the scalar variable.")
#ifndef PHASE_SPACE
            ("approx-phase-space-density", "standard DTFE (unrelated to PS-DTFE, the phase-space estimator): compute approximate phase-space density f = rho * g, where rho is the spatial DTFE density and g is the velocity-space DTFE density. Builds a second Delaunay tessellation in velocity space. The result is stored in the scalar output field.")
#endif
            ;
    
    
    po::options_description cellAverageOptions("Cell average (the two modes)");
    cellAverageOptions.add_options()
#ifdef PHASE_SPACE
            ("exact", po::bool_switch(&(this->exactFlag)), "PS-DTFE (2D and 3D builds): replace the nSub^3 sub-sampled grid deposit by the EXACT conservative voxelization of Powell & Abel 2015 (vendored r3d library, third_party/r3d; a 2D build clips triangles itself, ps_clip2d.h): every kept tetrahedron is analytically clipped against each grid cell in its bounding box and deposits mass * V_intersection/V_tetrahedron per cell -- no sampling noise, mass conservation to double precision. THE DEFAULT for a run with an '_a' field: such a run deposits exactly unless '--sampled', DTFE_EXACT_AVERAGE=0 (=1 is '--exact'; a flag beats it) or a '--subsamples' count is given, or the method is not PS-DTFE's; a run without '_a' fields keeps the sampled deposit (its grids hold the point values) unless '--exact' is given, which then drives every grid pass. The run log's 'grid deposit' line says which deposit ran and why. The kept-cell classification, periodic minimum-image convention and per-tet renormalization are IDENTICAL to the standard deposit, so '.streams'-defining overlaps, '--ps-halo-release' and '--ps-caustics' compose unchanged. Velocity moments integrate the LINEAR velocity profile exactly (order-1 moments); the dispersion additionally carries each cell-intersection's exact velocity covariance (order-2 moments). Beside '--exact', the count nSub ('--subsamples') reaches only a tetrahedron too thin to clip (sampled at that count) and a partition's sub-grid ring in the '_a' pass: 3 when none is given; a given count is kept. The exact deposit is the nSub->infinity limit, so the unaveraged and '_a' grids coincide where every tetrahedron clips (the unaveraged pass samples one too thin to clip at one point). Runs on the CPU (double precision, the reference: r3d, each tetrahedron clipped once per slab and column of its cells, exact_sweep.h) and on the GPU with '--gpu' (a different exact method: a z-sweep with exact Gauss-Legendre quadrature in float32, metal/exact_clip.metal.inc, agreeing with the CPU to float rounding as every GPU field does but the unaveraged velocity derivatives: see '--gpu'). The GPU hands every tetrahedron whose velocity dispersion float32 cannot resolve (a thin tetrahedron with a steep velocity gradient) to the CPU's double deposit, as it does degenerate ones. A GPU build runs the exact deposit on the GPU without '--gpu' (the program's pick, as for every deposit; DTFE_GPU=0 keeps the CPU), in performance together with the free cores ('--gpu-share auto', the default; '--gpu-share mixed' in every mode). Point evaluation ('--sample-points') is unaffected. (The same name as standard DTFE's exact cell average.)")
            ("sampled", po::bool_switch(&(this->sampledFlag)), "PS-DTFE: the nSub^3 SUB-SAMPLED grid deposit (nSub^2 in a 2D build; '--subsamples nSub', 1 by default: one point per cell, at its centre -- quick, with noisy per-cell values, so smooth the grids before use; more points per cell give a smoother cell average and take more time) instead of the exact deposit of '--exact' (the reference), the default for a run with an '_a' field. A run deposits sampled by itself when it selects no '_a' field (its grids then hold the point values, nSub = 1, as before), when a '--subsamples' count is given without '--exact', under DTFE_EXACT_AVERAGE=0 (a flag beats it), and with a method other than PS-DTFE's. The older PS-DTFE grids made without asking for the exact one hold the sampled deposit; the face gradient needs the exact one: a sampled run keeps the stream average of each stream's own gradient ('--face-gradient' beside '--sampled' is refused). '--exact' and '--sampled' exclude each other.")
#else
            ("exact", po::bool_switch(&(this->exactFlag)), "standard DTFE (2D and 3D builds; 2D clips triangles itself): compute the volume-averaged ('_a') fields by integrating the LINEAR DTFE interpolant EXACTLY over every grid-cell/tetrahedron intersection (vendored r3d library, Powell & Abel 2015) instead of Monte-Carlo sampling. THE DEFAULT: a run that passes neither '--exact' nor '--sampled' integrates its '_a' fields exactly, unless DTFE_EXACT_AVERAGE=0 is set (=1 is '--exact'; a flag beats it), a sampling count or method is given ('--subsamples' or '--average-method', any value: the sampled average), or the exact average does not apply ('velocityStd_a', a '--redshift-cone' grid, a MY_SCALAR build, a method other than DTFE): such a run samples by itself, and the run log's 'volume averaging' line says which average ran and why. Asked for explicitly ('--exact' or DTFE_EXACT_AVERAGE=1) it is refused where it cannot run (MY_SCALAR, another method, 'velocityStd_a', '--average-method 2'), and on a redshift cone grid it warns and samples. For a linear field the integral over each intersection is its centroid value times its volume, so order-1 moments suffice -- no sampling noise, and on a perfect particle lattice the averaged density is exactly 1 to rounding. Runs on the method-1 per-tetrahedron scatter topology with the same cell classification (the single-grid-cell fast path is exact as well); beside '--exact', '--subsamples' is ignored and '--average-method' must be 1. Runs on the GPU with '--gpu', by a different exact method there (a z-sweep with exact Gauss-Legendre quadrature in float32, metal/exact_clip.metal.inc), agreeing with the CPU to float rounding. The CPU clips each tetrahedron once per slab and column of its cells (exact_sweep.h). The exact average's box mean is 1 to ~3e-8 (scripts/run_dtfe.sh passes '--exact' or '--sampled' explicitly). Its deposits form every tetrahedron's geometry in the GRID FRAME -- relative to the corner of the full grid's cell holding its base vertex, the vertices in the hashed order the Monte-Carlo average uses, the gradients in double -- so a '--partition' or thread split does not change what a tetrahedron deposits, and the shares are summed exactly, as integers in one fixed-point unit per grid for the whole run (set before the first batch from the cell volume, the mean density and the particles' largest velocity component; the GPU in 128-bit words, the CPU and the face pass in 64-bit words with an exact carry), each cell rounded to float once: in the single build an all-GPU or an all-CPU run writes the same bytes at any partition split and thread count (tests/split_bytes_check.sh: the 64^3 demo at 1, 2^3 and 3^3 partitions). (Older binaries formed the frame per sub-domain in float32, so a split moved the bytes: where a sub-domain corner lay in a finer binade than the sub-domain's extent the vertices rounded by up to 2e-6 Mpc, and a 3^3 and a 6^3 split of TNG100-3-Dark z = 0 at 512^3 differed by 2.5e-5 of the density rms, up to 7% in single cells.) (PS-DTFE's '--exact' is its exact deposit.)")
            ("sampled", po::bool_switch(&(this->sampledFlag)), "standard DTFE: compute the volume-averaged ('_a') fields by Monte-Carlo SAMPLING ('--average-method', method 1 by default; '--subsamples k', k^3 points per grid cell, k^2 in 2D) instead of the exact integration of '--exact', the default. A run samples by itself where the exact average does not apply ('velocityStd_a', which takes '--average-method 2'; a '--redshift-cone' grid or user-given sampling coordinates, which take method 2; a MY_SCALAR build; a method other than DTFE), when '--subsamples' or '--average-method' is given without '--exact', and under DTFE_EXACT_AVERAGE=0 (a flag beats it). The older '_a' grids made without asking for the exact one hold the sampled average (the run log's 'volume averaging' line says which ran). '--exact' and '--sampled' exclude each other.")
#endif
            ("seed", po::value<size_t>(&(this->randomSeed)), "integer seed of the random generator of standard DTFE's '--average-method 2' (random points in each grid cell), '--particle-fraction' and '--poisson'. The exact average ('--exact', the default) and method 1 (the sampled average's default) are deterministic and ignore it. Drawn from the clock when not supplied, so two '--average-method 2' runs then differ; method 2 also adds 1000 x the number of the sub-domain (partition or thread), so its realization depends on the split.")
#ifndef PHASE_SPACE
            ("average-method", po::value<int>(&(this->method))->default_value(this->method), "standard DTFE: choose the method of the SAMPLED volume average ('--sampled'; only for fields inserted using the '_a' ending). Giving it (any value) without '--exact' selects the sampled average; beside '--exact' only 1 is accepted:\n"
                    "  1 = \tvolume average the fields using a Monte Carlo method inside the Delaunay cell: each tetrahedron is sampled at the first n points of one fixed quasi-random (Sobol) table, n in proportion to its volume, so the result is deterministic. The samples are centred on the tetrahedron's centroid (its mass and momentum are exact; only their split over the cells is sampled) and laid out from a vertex order hashed from the vertex positions, so the realization is the same for any '--partition', thread count or padding. '--exact' (the default) removes the sampling altogether.\n"
                    "  2 = \tvolume average the fields using the Monte Carlo method inside the grid cell.")
#endif
            ;
    
    
    po::options_description deviceOptions("Device and machine use");
    deviceOptions.add_options()
#ifdef PHASE_SPACE
            ("gpu", po::bool_switch(&(this->gpuAlias)), "PS-DTFE: run the grid deposit on the GPU. Requires a GPU build ('make PS-DTFE METAL=1' on macOS/Apple Silicon); otherwise the option is ignored with a warning and the CPU deposit is used. Without it the program picks the device: the GPU wherever the deposit runs there, for averaged ('_a') and unaveraged fields, exact or sampled (a 3D GPU build, the PS-DTFE method, at most one scalar component, no '--ps-caustic-cusps', not '--serve'), else the CPU, without a warning. Asked or picked, a pass whose sub-grid holds 2^32 cells or more (the GPU kernels number fewer: one domain's whole grid, a coarse '--partition', a '--grid-window' that large) deposits on the CPU and says so (the auto-tuner aims the sub-grids of a split it chooses under it). DTFE_GPU in the environment pins the device: 1 (any value but the off ones) asks for the GPU exactly as this flag does, 0 (false, no, off) keeps the CPU, unset, empty or 'auto' leaves it to the program; the flag beats it. The run log's 'PS deposit' line says which device and why. The cell averages (the '_a' grids), the densities, the stream counts and the flags match the CPU deposit to float rounding (atomic summation order): the GPU's mass grid is an exact fixed-point sum (64-bit words in a unit per pass), and the stream counts and caustic flags are identical in the single-precision build (the double build's GPU works on float copies of its tetrahedra and may differ in up to 1e-4 of the cells). In the unaveraged grids (the values at the cell centres) the velocity-derivative grids can differ from the CPU's by more than rounding in single cells, the vorticity most: the kernel evaluates them in float where the CPU works in double. (Standard DTFE's '--gpu' runs its volume-averaged interpolation on the GPU.)")
#else
            ("gpu", po::bool_switch(&(this->gpuAlias)), "standard DTFE: run the volume-averaged ('_a', method 1) grid interpolation on the GPU. Requires a GPU build ('make DTFE METAL=1' on macOS/Apple Silicon); otherwise the option is ignored with a warning and the CPU interpolation is used. Without it the program picks the device: the GPU for the exact average ('--exact', the default) wherever it runs there (a 3D GPU build, the DTFE method, '--average-method 1', a regular grid, at most one scalar component, not '--serve'), the CPU for the sampled average ('--sampled', in every run mode) and everywhere else, without a warning. Asked or picked, a sub-domain (one thread's box of a partition or of the whole grid) of 2^32 cells or more (the GPU kernels number fewer) is averaged on the CPU and says so. DTFE_GPU in the environment pins the device: 1 (any value but the off ones) asks for the GPU exactly as this flag does, 0 (false, no, off) keeps the CPU, unset, empty or 'auto' leaves it to the program; the flag beats it. The run log's ''_a' interpolation' line says which device and why. With '--sampled' the GPU takes the same samples as the CPU, and its cells match the CPU's to float rounding except in about 0.1-0.2% of them, where a sample within float rounding of a cell face lands in the neighbouring cell and moves both by up to a few percent; the box sums agree to rounding. Its sums are exact fixed-point words, converted to float once, so two GPU runs write byte-identical grids: with '--exact' 128-bit words in one unit per grid for the whole run (the same at every partition and thread split), with '--sampled' 64-bit words in a unit per batch. A failed command buffer is retried with PS-DTFE's policy, from the batch's last checkpoint (DTFE_GPU_CHECKPOINT_CHUNKS) at a pace set by the cause; after repeated failures the batch runs on the CPU, and the run ends with the 'GPU deposit retries' line. With '--exact' (the default) the GPU agrees with the CPU to float rounding. Unaveraged fields and method 2 always use the CPU. (PS-DTFE's '--gpu' runs its grid deposit on the GPU.)")
#endif
            ("run-mode", po::value<std::string>(), "How hard the run may lean on the machine: performance uses every core and long GPU buffers; balanced (the default) the performance cores less a quarter and short GPU buffers (~50 ms), so the display never waits on the GPU; gentle half the performance cores and the short buffers, and on Linux a lower priority (nice 10), so the machine stays usable. Inside a batch job (SLURM_JOB_ID or PBS_JOBID set and not empty: the job's cores are the run's) the default is performance. DTFE_RUN_MODE is the same choice (any case; unset or 'auto' = the default), and the flag beats it. OMP_NUM_THREADS and the DTFE_*_THREADS pins are kept, and a '--max-concurrent' above the mode's threads raises them. The mode changes no PS-DTFE single-precision byte but through the GPU share below, nor a byte of standard DTFE's single-precision exact '_a' grids; its other CPU grids (the Monte-Carlo average of '--sampled', the '_u' fields, the double build) follow the thread count at float rounding, as with OMP_NUM_THREADS. With '--gpu-share auto' (the default), performance also deposits part of a partitioned PS-DTFE GPU run's exact deposits on the free cores (a mixed run's grids differ from an all-GPU run's at float rounding); with '--gpu-share mixed' every mode does, at the share of the mode's threads. The run log's 'run mode' line says what was set and why.")
            ;
    
    
    po::options_description splitOptions("Split, memory and resume");
    splitOptions.add_options()
            ("partition", po::value< std::vector<size_t> >(&(this->partition))->multitoken(), "choose this option if the particle data is too large to compute the Delaunay triangulation for the full data at once. Specify here in how many parts to split the box along each direction (e.g. '--partition 3 3 3' splits the data in 27 chunks; '--partition 1' keeps ONE triangulation). If NOT given, the program AUTO-SELECTS the split from the particle count, grid, requested fields and the machine's RAM/cores once the input is read (DTFE_RAM_GB env overrides the detected RAM)."
#ifdef PHASE_SPACE
             " In PS-DTFE this also drives the OpenMP parallelism: partitions run in parallel (one triangulation per thread), so speedup scales up to the number of partitions. NOTE on memory: each Lagrangian partition writes the WHOLE Eulerian grid, so peak memory ~ (partitions running concurrently) x full grid x number of fields -- choose the partition count from your core budget AND available RAM."
#endif
             )
            ("max-concurrent", po::value<int>(&(this->maxConcurrent))->default_value(0), "caps how many Delaunay triangulations are built AT ONCE, to bound peak memory on machines with less RAM (trades CPU cores for memory). Each concurrently-built triangulation is the dominant memory cost, so peak RAM ~ (concurrent triangulations) x per-triangulation size. '0' uses all the threads of the run mode ('--run-mode': every thread in performance). If NOT given, the program AUTO-SELECTS the cap together with the partition split (see '--partition')."
#ifdef PHASE_SPACE
             " In PS-DTFE this limits how many Lagrangian partitions run in parallel (e.g. '--partition 4 4 4 --max-concurrent 4' builds at most 4 of the 64 partition triangulations simultaneously). Combine a FINER --partition (smaller per-triangulation size) with a SMALLER --max-concurrent to fit large grids in limited RAM."
#else
             " In standard DTFE this limits the concurrent spatial sub-triangulations inside the parallel interpolation; combine with --partition for serial, low-memory processing of very large data. The partitions themselves always run one after another; the auto-tuner plans its split with a given value, and with '--sample-points' every partition is a single triangulation, so the cap has no effect there."
#endif
             )
            ("scratch-dir", po::value<std::string>(&(this->scratchDir)), "back every allocation of >= 1 GB -- in practice, the full-resolution output grids, which '--partition' can NOT reduce (each partition deposits into the same shared grid) -- with memory-mapped files in this directory instead of RAM. Lets a run whose grids exceed the RAM (e.g. a 1024^3 grid with the full field set, ~146 GB of accumulators) run against local disk; results are bit-identical, the kernel pages the grids in and out as needed, and the scratch files are unlinked at creation so they vanish automatically even if the run crashes. The directory must exist on a LOCAL, non-synced volume (e.g. /private/tmp/dtfe-scratch) with enough free space for the grids; iCloud paths are rejected. The 1 GB threshold can be changed with the DTFE_SCRATCH_MIN_GB environment variable. PS-DTFE, partitioned runs: a partition that finishes before its turn in the ordered merge and does not fit in memory is spilled to an unlinked file here (without this option: in the system temporary folder; see '--ps-merge').")
            ("auto-tune-report", po::bool_switch(&(this->autoTuneReport)), "both binaries (in standard DTFE mc is the number of threads each partition uses, the partitions running one after another): read the input, run the memory auto-tuner exactly as a real run would (the same flags, including a given --partition/--max-concurrent), print ONE line -- AUTO-TUNE-REPORT partition=.. mc=.. predicted_gb=.. budget_gb=.. ... over_budget=0|1 -- and exit without computing anything. For checking whether a configuration fits before running it (the GUI does). DTFE_AUTO_PTS_N and DTFE_AUTO_PTS_VEL_GRAD=1 size a --sample-points plane that does not exist yet (both binaries; the standard plan counts a plane's records, one per point). The line also gives pts_gb (the plane), snapshot_gb (the GPU deposit's host copy of its grids, kept to resume a failed command buffer) and lookahead_gb (the next partition's tessellations, prepared during a GPU deposit), and, only where a plan books them, readkept_gb (PS-DTFE: the read buffers kept beside one partition at a time under '--scratch-dir') and facewalk_gb (PS-DTFE: the face walk's sums beside the deposit), each already in predicted_gb.")
#ifdef PHASE_SPACE
            ("resume", po::bool_switch(&(this->psResume)), "PS-DTFE, partitioned runs: continue a run that was killed, from its checkpoint. A partitioned run saves its merged grids and the list of merged partitions in '<output>.runlog.checkpoint' (with '--scratch-dir': in the scratch directory, named from the whole output path) after a partition merges, unless the last save took more than 5% of the time since it; the folder is removed once the outputs are in place. '--resume' loads it, skips the partitions it lists and computes the rest; it refuses, naming the differing settings, when the command line (apart from --resume, --ps-merge, --no-overlap, --verbose, --max-concurrent, --run-mode, --gpu-share, --scratch-dir and --tessellation-cache; a short letter counts as its long name), the resolved cell average (exact or sampled) and gradient route (face or per-tetrahedron; both follow the flags, and DTFE_EXACT_AVERAGE and DTFE_FACE_GRADIENT can change them), the resolved device (GPU or CPU: '--gpu', DTFE_GPU or the program's pick), the build or the binary itself (a rebuilt binary does not resume), the input files or the partition split differ. The resolved GPU share is part of the key; --resume with --gpu-share auto (the default) or mixed keeps the checkpoint's. Without '--partition' the checkpoint's split is used. It merges in index order (the default; it overrides a '--ps-merge completion-order'): when the killed run merged in index order too, the grids are byte-identical to an uninterrupted run. DTFE_CHECKPOINT=0 writes no checkpoints; runs with '--interlace' are not checkpointed. A '--sample-points' run is checkpointed too: each merged partition's stream records are saved once as 'points.<i>' beside the checkpoint, and the sample-point file (size + mtime) is part of the key.")
            ("ps-merge", po::value<std::string>(), "PS-DTFE, partitioned runs: how the partitions' grids are added into the result, one of:\n"
                    "  index-order = \tadd the partitions' grids into the result in partition INDEX order, whatever order they finish in, so two identical runs give bit-identical grids (the merge's float adds then always happen in the same order; with '--gpu' the deposit's own atomic sums must be order-free too, which every grid of the deposit is: exact 64-bit fixed-point sums). The ordered merge is STAGED: a partition that finishes before its turn hands its sub-grids to a staging area in memory and its thread takes the next partition, and the partition whose turn it is merges every staged one after it in index order -- so no thread idles behind a slow partition. Staging moves the sub-grids (nothing is allocated) and frees the partition's particles; where the memory gate guards each partition start (an auto-tuned plan with several in flight) a partition is always staged, the gate then admitting the next one only while the free memory holds it; otherwise (a hand-set split) only while the free memory less 4 GB still holds the sub-grids, else it waits for its turn holding its thread. DTFE_MERGE_STAGE_GB=X caps the staged bytes at X GB instead, 0 = never stage. A partition whose sub-grids the memory cannot hold (under the gate: beside one more partition) is SPILLED instead: written to an unlinked file in '--scratch-dir', else in the system temporary folder, freed, and read back at its turn -- the merged grids stay byte-identical; DTFE_MERGE_SPILL=0 switches that off, DTFE_MERGE_SPILL=1 spills every staged partition. The run log's 'ordered merge:' line says how many were staged, how much memory they held and how many were spilled. The memory gate still applies; a partition waits for memory only while a LOWER-numbered one is in flight, so no wait can deadlock. At '--max-concurrent 1' the order is the index order anyway, and the auto-tuner's speed split of a small set always merges in order. Given explicitly, the index order also keeps the partition memory gate on under the auto-tuner's speed split of a small set (which merges in order and otherwise drops the gate).\n"
                    "  completion-order = \tadd each partition's grids into the result as soon as it finishes, instead of the default ordered merge. With '--max-concurrent' above 1 the float adds then follow the completion order, so two identical runs differ in the last bits (~1e-7 relative in float); it holds no staged sub-grids. '--resume' merges in index order whatever this says, and the auto-tuner's speed split of a small set always does.\n"
                    "DEFAULT: index-order. '--resume' overrides completion-order (it merges in index order). scripts/run_ps_dtfe.sh: PS_MERGE=index-order (its default, passed explicitly) or completion-order.")
#else
            ("resume", po::bool_switch(&(this->psResume)), "standard DTFE, partitioned runs (PS-DTFE's '--resume', through the same code): continue a run that was killed, from its checkpoint. The partition loop saves the full grids of the partitions finished so far, and their list, in '<output>.runlog.checkpoint' (with '--scratch-dir': in the scratch directory, named from the whole output path) after a partition, unless the last save took more than 5% of the time since it; the folder is removed once the outputs are in place. '--resume' loads it, skips the partitions it lists and computes the rest; it refuses, naming the differing settings, when the command line (apart from --resume, --no-overlap, --verbose, --max-concurrent, --run-mode, --scratch-dir and --tessellation-cache; a short letter counts as its long name), the resolved cell average (exact or sampled) and gradient route (face or per-tetrahedron; both follow the flags, and DTFE_EXACT_AVERAGE and DTFE_FACE_GRADIENT can change them), the resolved device (GPU or CPU: '--gpu', DTFE_GPU or the program's pick), the build or the binary itself (a rebuilt binary does not resume), the input files or the partition split differ. Without '--partition' the checkpoint's split is used. On the CPU the outputs are byte-identical to an uninterrupted run's at the same thread count ('--max-concurrent', OMP_NUM_THREADS: the threads split each partition into sub-boxes of their own, so another count moves the Monte-Carlo average's grids, the '_u' fields and the double build's, by float rounding where the padding is ample; the single build's exact '_a' grids are the same at any count, their sums being exact in one unit for the whole run); with '--gpu' too (its sums are exact). scripts/run_dtfe.sh passes it when a checkpoint exists (DTFE_RESUME=0 starts over). DTFE_CHECKPOINT=0 writes no checkpoints; '--interlace' runs are not checkpointed. A '--sample-points' run is checkpointed too, as in PS-DTFE: each finished partition's point answers are saved once as 'points.<i>' beside the checkpoint, and the sample-point file (size + mtime) is part of the key, so a resumed plane is byte-identical to an uninterrupted one.")
#endif
            ("partition-index", po::value<int>(&(this->partNo)), "choose to compute the density only for this partition number (from 0 to 'maximum partitions'-1). This options is usefull if you would like to compute the density for a large particle number on several different machines at the same time - need to run the program on each machine independently.")
            ("tessellation-cache", po::value<std::string>(&(this->tessellationCacheDir)), "reuse the Delaunay tessellation across runs: after it is built (and its vertex densities computed) the whole tessellation is written to this directory, and any later run whose inputs are IDENTICAL loads it instead of rebuilding. Building the tessellation is the dominant fixed cost of a run, so re-querying one snapshot with a different '--sample-points' set, grid or field selection gets much cheaper (CGAL's binary structure I/O restores the tessellation faster than re-inserting the points, and the vertex-density pass is skipped entirely). The identity of every input that can change the tessellation or its per-vertex payload (the input files' size and modification time, the box, region, padding, partition, sub-sampling and the compile-time configuration) is stored in the file and compared before it is used, so a stale cache can only ever be a MISS, never a wrong result -- a miss silently rebuilds, and '--verbose 3' names the value that differed. The file is named after that identity, NOT after the output, so a second run writing different outputs reuses it. A PARTITIONED phase-space run caches every partition separately -- one file each, keyed by the partition's own Lagrangian region, since 'partNo' is -1 for all of them. A partitioned STANDARD run caches every tessellation it builds: one file per thread sub-box of each partition (keyed by the sub-box and its partition's region), or one per partition under '--sample-points'; its split follows the '--grid', so another grid is another set of files. COSTS DISK: the file is compressed (deflate), which puts it at about 0.15 KB per tessellation vertex -- 225 MB for 1.5e6 vertices instead of 735 MB raw -- and the Lagrangian padding multiplies the particle count by ~4x. Reading is no slower than an uncompressed cache (the smaller read pays for the inflation); writing costs a little more. The run refuses to write (with a warning, without failing) when the volume would not hold it. The directory must exist on a LOCAL, non-synced volume; iCloud paths are rejected.")
            ;
    
    
    po::options_description regionOptions("Region");
    regionOptions.add_options()
            ("region", po::value< std::vector<Real> >()->multitoken(), "choose this option to compute the field interpolation only in a given part of the full box. Use this option to specify the region of interest in terms of fractions of box length (i.e. '--region 0.4 0.6 0.3 0.7 0.45 0.55' computes the density for the box that extends from 0.4 to 0.6 of the box length along direction x, 0.3 to 0.7 along direction y and 0.45 to 0.55 along direction z).")
            ("region-mpc", po::value< std::vector<Real> >()->multitoken(), "choose this option to compute the field interpolation only in a given part of the box. Specify the limits of that region in Mpc units; see previous option for additional information. (A region lays the '--grid' over the region and tessellates the particles around it; PS-DTFE's '--grid-window' instead keeps the cells of the full run's '--grid' that lie inside a box.)")
            ("redshift-cone", po::value< std::vector<Real> >()->multitoken(), "specify to interpolate the fields on a redshift cone grid (i.e. on spherical coordinates). Must give 6 arguments in 3D (4 in 2D) which give 'r_min', 'r_max', 'theta_min', 'theta_max', 'psi_min' and 'psi_max' (where 'r' is the distance in Mpc and 'theta' and 'psi' the two angles expressed in degrees). For example '--redshift-cone 1 10 0 90 0 360' gives you a half a sphere shell." )
            ("cone-origin", po::value< std::vector<Real> >( &(this->originPosition) )->multitoken(), "specify the origin of the spherical coordinate system used in option '--redshift-cone'. It gives the x, y and z values of the origin point.")
#ifdef PHASE_SPACE
            ("grid-window", po::value< std::vector<Real> >()->multitoken(), "PS-DTFE only (2D and 3D builds): deposit, allocate and write ONLY the grid cells of the full '--grid' that lie inside the given box ('x0 x1 y0 y1 z0 z1' in the OUTPUT coordinates, i.e. Mpc after the '--mpc-unit' conversion; the box is snapped outward to cell edges). The tessellation is the full run's, periodic or not, with or without '--partition', and so is every tetrahedron's normalization, so the window run EQUALS the full run's window cells -- unlike '--box', which changes the grid. The sampled deposit counts every sub-sample of a tetrahedron straddling the window edge (over its whole bounding box) and deposits only the part inside: the full run's cells on the CPU (counts and masks bit for bit, the sums to the last fixed-point unit, a window pass taking the units of the tetrahedra that reach the window, as the GPU does), to float rounding on the GPU. The exact deposit ('--exact') clips ONLY the window's cells and normalizes each tetrahedron by its whole (analytic) weight, which equals the full run's sum over all its pieces to double rounding: the full run's cells to float rounding, at a cost that scales with the window, not with the grid -- the choice for a fine grid, e.g. the launcher's exact zoom (a 4096^3 virtual grid windowed to one slice). The output grids hold the window's cells only (their header carries the window's box), so a fine look at one region of a large box costs the tessellation plus a small grid: a 4096^3 grid over a 100 Mpc box, windowed to 512^2 cells of one slice, holds 256k cells. With '--tessellation-cache' and '--partition', the cached partition tessellations are loaded instead of rebuilt, and every partition whose occupancy map (written beside its tessellation by a composite '--serve' or an earlier partitioned run) shows that its streams never reach the window is skipped outright -- the launcher's exact zoom of a 94-million-particle box then processes ~12 of 64 partitions. Not combinable with the T-web / V-web classes (they need the full periodic grid), '--region' or '--partition-index'.")
#endif
            ;
    
    
    po::options_description pointOptions("Points and server");
    pointOptions.add_options()
            ("sample-points", po::value<std::string>(&(this->psSamplePointsFile)), "evaluate the fields at arbitrary Eulerian points read from the given file, in ADDITION to the regular grid interpolation (pass a small '--grid', e.g. '-g 16', when only point values are wanted). PS-DTFE: the multi-stream phase-space fields, one stream per folded tetrahedron containing the point, deterministic under any '--partition' split. Point location is exact: a point on a shared face, edge or vertex (e.g. a particle position) is counted once per stream, by exact predicates with a symbolic tie-break. Standard DTFE: the plain Eulerian DTFE interpolant -- exactly one containing tetrahedron, '.pts_streams' becomes the 0/1 coverage flag, '.pts_velDisp' is identically 0, and with '--partition' each partition answers the points inside the tetrahedra whose Eulerian centroid lies in its sub-box: one triangulation per partition (no thread sub-boxes, so '--max-concurrent' has no effect), equal to a single triangulation to float rounding; '--partition-index' is refused. Point coordinates are in Mpc, as the particle positions are after the '--mpc-unit' conversion (not in the input file's length unit, which '--box' takes). Accepted formats, auto-detected: a text file with one 'x y z' triplet per line, or a raw binary file of float64 triplets (N x 3, row-major, native/little-endian, no header). Writes '<output>.pts_den' (float64, rho/rho_bar), '.pts_vel' (float64 x3, the plain mean over the streams: equal shares, the grid's volume weighting at a point), '.pts_velDisp' (float64 x6 dispersion tensor, xx xy xz yy yz zz), '.pts_streams' (int32); all CPU-only (multi-threaded; DTFE_PTS_THREADS pins the count), double precision. For repeated queries without rebuilding see '--serve'.")
            ("pts-den-grad", po::bool_switch(&(this->psPtsDenGrad)), "with '--sample-points', also write the spatial gradient of the total point density to '<output>.pts_denGrad' (float64 x3 per point, d(rho/rho_bar)/dx_i in 1/Mpc). PS-DTFE: the sum over the point's streams of each stream's constant linear 'dtfe' density-profile gradient; with '--ps-per-stream', additionally writes '<output>.pts_stream_dengrad' (float64 x3 per record, same ragged order as '.pts_stream_records'). Hull cells with the constant volume-ratio density contribute 0. Standard DTFE: the containing tetrahedron's constant DTFE density gradient.")
            ("pts-vel-grad", po::bool_switch(&(this->psPtsVelGrad)), "with '--sample-points', also write the velocity gradient of the point's multi-stream flow to '<output>.pts_velGrad' (float64 x9 per point, row-major [d*3+j] = d v_j / d x_d in velocity units per Mpc -- the TRANSPOSE of the '.velGrad' grid output, whose [j*3+d] = d v_j / d x_d). PS-DTFE: the plain mean over the point's streams of each stream's constant linear-velocity-profile gradient (the pointwise analogue of the grid deposit's own gradient, so the stream average of the gradients, as in the PS grids without the face gradient ('--per-tet-gradient'), not the gradient of the mean velocity; the same equal shares as for '.pts_vel'). Divergence, shear and vorticity follow algebraically from this tensor, so they need no files of their own. Standard DTFE: the containing tetrahedron's constant velocity gradient.")
            ("pts-scalar", po::bool_switch(&(this->psPtsScalar)), "with '--sample-points' or '--serve', also interpolate the particles' SCALAR value at every point -- any per-particle quantity, read with '--scalar-dataset' (or a reader's own scalar, e.g. the gas temperature). PS-DTFE: each stream carries the linear interpolant of its tetrahedron's vertex values (exactly how the velocity is interpolated; CosmoDTFE's PhaseSpaceEstimator values), '<output>.pts_scalar' holds the plain mean over the point's streams and, with '--ps-per-stream', '<output>.pts_stream_scalar' every stream's own value (float64, same ragged order as '.pts_stream_records'). Standard DTFE: the plain linear DTFE interpolant of the scalar.")
            ("serve", po::bool_switch(&(this->psServe)), "both binaries (2D and 3D builds; the handshake says which): build the tessellation ONCE, keep it in memory and answer point-evaluation requests over stdin/stdout until stdin closes -- the interactive mode behind the Python 'dtfelib.Estimator' (est(points) -> fields), so a notebook can query any set of points repeatedly without rebuilding. Every '--sample-points' output option applies to the answers ('--ps-per-stream', '--ps-per-stream-ids', '--pts-den-grad', '--pts-vel-grad', '--ps-caustics'); no grid or '.pts_*' file is written. Normal log messages go to stderr; stdout carries only the binary protocol (documented in src/CGAL_triangulation/ps_serve.cc). With '--partition' (or when the auto-tuner finds that one triangulation would not fit in memory) the server becomes a COMPOSITE of partition tessellations, the out-of-core analogue of CosmoDTFE's CompositeEstimator: each partition is built once (in parallel, '--max-concurrent'), a coarse map of the Eulerian buckets its tetrahedra cover is kept, and a request visits only the partitions whose map holds one of its points. With '--tessellation-cache' the partitions are written to disk and at most '--serve-resident' of them stay in memory (least recently used ones are dropped and reloaded on demand), so a box far larger than RAM can be queried interactively; a later session whose inputs are unchanged starts without building anything. Answers are bit-identical to '--sample-points' with the same '--partition' (PS-DTFE; the standard server splits on its own 64-per-axis grid and a batch run on its '--grid', so there they agree to float rounding); against an unpartitioned server the stream counts are identical (exact, partition-invariant counting) and the values agree to float rounding (a partition's periodic copies are rounded differently). The standard binary's partitions are Eulerian sub-boxes (a tetrahedron belongs to the one holding its centroid); on a NON-periodic box, points within a few tetrahedra of where the particles' convex hull meets a partition cut can differ slightly more. Cannot be combined with '--sample-points'.")
#ifdef PHASE_SPACE
            ("ps-per-stream", po::bool_switch(&(this->psPerStream)), "PS-DTFE only: with '--sample-points', additionally write each stream's own density and velocity per point in a ragged layout: '<output>.pts_stream_offsets' (uint64, N+1; point i owns records [off[i], off[i+1])) and '<output>.pts_stream_records' (float64 x4 per record: density, vx, vy, vz; sorted by density, descending).")
            ("ps-per-stream-ids", po::bool_switch(&(this->psPerStreamIds)), "PS-DTFE only: with '--sample-points', also write each stream's identity -- the ParticleIDs of the 4 Lagrangian vertices of its tetrahedron -- to '<output>.pts_stream_ids' (uint64 x4 per record, same ragged order as '.pts_stream_records'; each quadruple sorted ascending, so it is orientation-independent and identical under any '--partition' split). Implies '--ps-per-stream'. Needs an input reader that provides ParticleIDs (Gadget HDF5); otherwise the IDs are written as 0.")
            ("ps-per-stream-lagrangian", po::bool_switch(&(this->psPtsLagrangian)), "PS-DTFE only: with '--sample-points' or '--serve', also write each stream's LAGRANGIAN coordinate q(x) -- the initial position of the matter that stream brings to the point, i.e. the inverse of the (folded) Lagrangian-to-Eulerian map -- to '<output>.pts_stream_lagpos' (float64 x3 per stream, the '.pts_stream_records' order; wrapped into the box when periodic). Where the matter in a void, wall or filament came from, stream by stream. Implies '--ps-per-stream'.")
#endif
            ;
    
    
#ifdef PHASE_SPACE
    po::options_description phaseSpaceOptions("Phase space");
    phaseSpaceOptions.add_options()
            ("ps-caustics", po::bool_switch(&(this->psCaustics)), "PS-DTFE only (2D and 3D builds): flag fold-caustic grid cells during the grid deposit and write '<output>.caustic' (float32, same N^3 layout as '.streams'; 1 = the cell is overlapped by tetrahedra of BOTH orientations, 0 otherwise). The orientation of a stream is the sign of det(Ax), the parity of the Lagrangian->Eulerian map, which flips at every fold caustic -- so a cell containing both parities is crossed by a fold surface (the caustic skeleton of Feldbrugge & van de Weygaert). The two per-cell orientation bits are merged with a bitwise OR across '--partition' splits, making the output partition- and thread-order invariant (byte-identical for any split, per backend). Computed by both the CPU and GPU deposits (the GPU ORs the bits atomically, so its output is equally order-invariant; CPU-vs-GPU grids can differ only for cells whose float-vs-double determinant sign straddles the degeneracy cut, and in the double build in up to 1e-4 of the cells, where a sample lies within the float rounding of the double vertices). The negative-orientation bit is geometric: every cell a folded (det<0) tetrahedron overlaps, the exact test behind bit 1 of '.hidden_streams', so '.caustic' equals ('.streams' > 1 or '.hidden_streams' bit 1) and does not depend on '--subsamples' or the exact deposit. (A bit set only where a folded tetrahedron deposits a sample misses every fold thinner than the sampling: on TNG100-3-Dark at z = 1 it flags 21.0% of the cells instead of 33.6%; dtfelib's FieldSet.fold_mask() rebuilds the exact flag from grids written that way.) '<output>.causticClass' carries the same two bits plus the collapse multiplicity (bits 2-5), the umbilic indicator (bit 6) and with '--ps-caustic-cusps' bits 7-8; those are recorded at the deposited samples, so they depend on the sampling, a fold cell can show k = 0, and the cusp bits can differ between splits in a few cells. Point evaluation ('--sample-points') is unaffected by this flag.")
            ("ps-caustic-cusps", po::bool_switch(&(this->psCausticCusps)), "PS-DTFE only, CPU deposit only (2D and 3D builds; 2D sets no swallowtail bit): with '--ps-caustics', additionally estimate the A3 (CUSP) indicator and report it as bit 7 of '.causticClass'. A fold is the surface where the critical eigenvalue lambda_c of the deformation tensor crosses zero; it is an ordinary A2 fold where lambda_c varies at first order along its own null direction, and an A3 cusp where that variation stops (grad(lambda_c) . v_c = 0). Because the deformation tensor is CONSTANT inside a tetrahedron, the gradient does not exist there and is estimated by least squares from the face neighbours -- so this is an INDICATOR at the tessellation's resolution, not a pointwise identification of the A3 stratum. The gradient is fitted over the tetrahedra sharing a VERTEX with each near-fold tetrahedron, DEDUPLICATED and cut to the 64 nearest (the raw ring lists a face neighbour once per shared vertex, and those repeats would both re-weight the fit and fill its size cap, truncating the gather), not just the four sharing a face: four samples barely determine a 3-D gradient, and when they cluster the transverse fit noise reads as tangency. MEASURED on the 1-D Zel'dovich wave, whose caustic is two parallel planes: at A*k = 1.8, which has NO cusps anywhere, the wide stencil flags 0.01% of the fold cells (a four-neighbour stencil flags 7%); at A*k = 1.0, where the fold is exactly tangent (lambda_c = 1 + cos(kq) is stationary at kq = pi), it lights up the tangency plane. OFF by default because the wider stencil makes the deposit's caustic pass slower. The SAME stencil also yields the A4 (SWALLOWTAIL) indicator in bit 8, so this one flag emits both: A4 additionally requires the SECOND directional derivative to vanish, which is tested by fitting a cubic along the null direction and demanding that the expansion of lambda_c start at cubic order. MEASURED on waves with known strata: an exact A4 lights 74.5% of the cusp cells against 0.5% for a pure A3 and 0% for a generic fold -- a factor ~146, so bit 8 RANKS candidates rather than certifying them. Note A4 is codimension 3, i.e. its points are ISOLATED and generically inside no tetrahedron, so bit 8 marks proximity to a swallowtail along the cusp curve, over a band whose width is a fixed fraction of the stencil radius. Bit 8 is NOT a subset of bit 7: near a true swallowtail lambda_c is cubic along the null direction and bit 7's linear fit reads that as a non-zero slope, so it stays off there.")
            ("ps-halo-release", po::value<Real>(&(this->psHaloRelease)), "PS-DTFE only: release halo-interior streams from the grid deposit's bbox rasterization. A tetrahedron compressed beyond the given threshold D -- geometric stream density rho_geo = rho_bar*V_lag/V_eul > D, in rho/rho_bar units -- is deposited MONOLITHICALLY at its Eulerian centroid cell (the same mass-conserving fallback used for sub-sample-spacing tetrahedra), with its velocity/dispersion moments carrying the centroid-evaluated velocity. Rationale (Stuecker et al. 2021, arXiv:2109.09760): inside virialized halos the dark-matter sheet is mixed, fold counts explode (millions of streams per cell) and the sheet estimate is not converged anyway -- releasing those tetrahedra cuts the dominant tet-cell overlap cost while conserving mass EXACTLY and leaving single-stream regions (void interiors) bit-identical. Applies to the CPU and GPU deposits (identical kept/released classification, double-precision |det Lag| > D*|det Ax| in both). Point evaluation ('--sample-points') is NOT affected by this flag. Suggested range 100-1000.")
            ("ps-alpha-shape", po::value<Real>(&(this->psAlphaShape)), "PS-DTFE only, NON-periodic runs: keep only the Lagrangian tetrahedra whose circumradius is at most the given number of mean particle spacings (mean spacing = (V_lagBox/N)^(1/3)) -- the cloud's ALPHA SHAPE instead of its convex hull. Default 3. Interior tetrahedra of lattice-like initial conditions stay below ~0.9 spacings (jittered lattices and TNG100-3 alike), so the cut removes only the boundary: the flat slivers on the hull, which join particles up to a whole cloud face apart and whose stretched Eulerian images put spurious mass and streams across the region, and the tetrahedra that bridge concavities (e.g. particles missing from a cut region). It also makes '--partition' runs agree with a single triangulation: the hull of a nearly flat face is a global object that each partition builds differently, while every tetrahedron within the cut is local (the partitions are padded to at least 4 cuts). Costs the boundary layer beyond the cut (about 0.5% of an isolated 24^3 clump's mass). 0 = keep the whole convex hull (partitioned runs then differ at the hull). Periodic runs have no hull and ignore it.")
            ;
#endif
    
    
    po::options_description estimatorOptions("Other estimators");
    estimatorOptions.add_options()
            ("ngp", "choose the NGP (nearest grid point) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("cic", "choose the CIC (Cloud In Cell) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("tsc", "choose the TSC (Triangular Shape Cloud) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("pcs", "choose the PCS (Piecewise Cubic Spline) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("sph", po::value<int>(&(this->SPH_neighbors)), "choose the SPH (Smoothed Particle Hydrodynamics) as the grid interpolation method instead of DTFE. This method is available only for: density, velocity and scalar fields.")
            ("voronoi", "use Voronoi volume density estimation with NGP grid assignment. Builds the Delaunay triangulation, computes vertex densities (1/Voronoi volume), and assigns them to the grid using nearest grid point. Gives a piecewise-constant density field. Only outputs density.")
            ("interlace", "enable interlacing to reduce aliasing in the density field. Runs the grid interpolation twice, the second time on the grid moved by half a cell, and averages the two DENSITY grids in Fourier space. Works with any interpolation method, PS-DTFE included: a periodic run moves the particles by half a cell for the second pass; without '--periodic' a '--region' or '--region-mpc' at least half a cell inside the data box is required, and the second pass grids that region moved by half a cell. The PS-DTFE binary refuses '--region' and '--region-mpc', so there '--interlace' needs '--periodic'. Only the density is interlaced: the other fields, and the T-web and V-web, which each pass derives inside its own interpolation, come from the first, un-interlaced pass.")
            ;
    
    
    po::options_description otherOptions("Other");
    otherOptions.add_options()
            ("config,c", po::value<std::string>(&(this->configFilename)), "supply all/part of the program options in a configuration file. The option syntax is the same as at the command line. Can insert comments using the '#' symbol (everything after this symbol until the end of the line will be considered a comment)." )
            ("verbose,v", po::value<int>(&(this->verboseLevel))->default_value(verboseLevel), "choose the verbosity level of the program (a value from 0 to 3). See the documentation for additional help.")
#ifdef REDSHIFT_SPACE
            ("redshiftSpace", po::value< std::vector<Real> >()->multitoken(), "NOT IMPLEMENTED in this version: the run is refused with an error. Apply the transform 'redshift-space = position-space + (d1,d2,d3)*velocity / H' to the positions before running instead." )
#endif
            ;
    
    
    po::options_description developerOptions("Developer options (for tests and comparisons; the two modes set them)");
    developerOptions.add_options()
#ifdef PHASE_SPACE
            ("subsamples", po::value<int>(&(this->psAvgSubsamples))->default_value(1), "PS-DTFE: linear sub-sample count nSub (per axis) for the volume-averaged ('_a') fields of the SAMPLED deposit ('--sampled'). Giving it without '--exact' selects the sampled deposit. The exact deposit ('--exact', the default for '_a' fields and the reference) clips instead: it reads the count only for a tetrahedron too thin to clip (sampled at that count) and a partition's sub-grid ring in the '_a' pass -- 3 when none is given; a given count is kept. A cell's sub-points lie on a regular nSub^3 sub-grid (nSub^2 in a 2D build), at (k + 0.5)/nSub of the cell along each axis; every tetrahedron shares its mass among the sub-points that lie inside it (in equal shares), and one that holds none puts its whole mass in the cell of its centroid, so the mass is conserved at any count. 1 (the default) = ONE point per cell, at its centre -- the deposit a sampled run's unaveraged grids get: quick, with noisy per-cell values, so smooth the grids before use (void catalogues at the usual smoothing come out nearly the same as with the exact deposit). More points per cell give a smoother cell average and take more time.")
#else
            ("subsamples", po::value<int>(), "standard DTFE: k sampling points per axis, k^3 per grid cell (k^2 in 2D), when the SAMPLED volume average ('--sampled') averages the fields inside each grid cell (e.g. '--subsamples 3': 27 points; the same unit as PS-DTFE's '--subsamples'). Giving it without '--exact' selects the sampled average; beside '--exact' it is ignored. Method 2 takes k^3 random points, method 1 an average of k^3 per grid cell. DEFAULT if not given:\n"
                    "  1st method: \tan average of 100 sample points per grid cell.\n"
                    "  2nd method: \t20 random points per grid cell.")
#endif
            ("face-gradient", po::bool_switch(&(this->faceGradient)), "both binaries (3D): compute the volume-averaged velocity gradient ('gradient_a' and the 'divergence_a', 'shear_a', 'vorticity_a' and 'vweb_a' built on it) from EXACT FACE INTEGRALS of the interpolant: by Gauss, (1/V) times the integral of v n over a grid cell's six faces is the cell average of grad v. Every grid plane's section through every tetrahedron is clipped to the face squares and integrated exactly (src/CGAL_triangulation/face_sweep.h). THE DEFAULT of both binaries wherever it applies: '--per-tet-gradient' or DTFE_FACE_GRADIENT=0 keep the deposit's own gradient, this flag or DTFE_FACE_GRADIENT=1 ask for the face route and are refused where it cannot run, and a run where the default does not apply keeps the deposit's own gradient and says why in its header's 'velocity gradient' line. STANDARD DTFE: Its interpolant is linear in each tetrahedron and continuous across them, so this is the per-tetrahedron gradient's exact cell average, the value '--exact' gives, whatever the deposit (Monte Carlo, exact, GPU): the same grids for every deposit and device, formed in double (the deposit's gradient is float32 on the GPU: at TNG100 z = 0 and 512^3 the two differ by 5e-7 of the gradient rms per sub-domain, in 294 of 134 M cells by more than 1e-4), the same definition PS-DTFE's mean-flow gradient uses, and the fluxes close cell by cell. DTFE_FACE_CHECK=1 compares the two routes and prints one 'FACE-CHECK' line per sub-domain. 3D only (the 2D build keeps the deposit's gradient). PS-DTFE (where the run has the exact deposit, '--exact', the default for '_a' fields, and no '--ps-halo-release'; the face sums are formed on the CPU, beside the CPU deposit or the GPU deposit): the gradient of the volume-weighted MEAN velocity of the streams, the quantity standard DTFE stores, with the jumps at the caustics entering as exact flux differences -- instead of the stream average of each stream's own gradient, which is then written beside it: its divergence as '.a_velDivStreams', and for each of 'gradient_a', 'shear_a' and 'vorticity_a' asked its gradient, shear and vorticity as '.a_velGradStreams', '.a_velShearStreams' and '.a_velVortStreams' (the bytes a '--per-tet-gradient' run writes under the plain names). Costs one more walk over the tetrahedra that cross a grid plane and 30 more numbers per cell in the merged grids, which the auto-tuner books.")
#ifdef PHASE_SPACE
            ("per-tet-gradient", po::bool_switch(&(this->perTetGradient)),
             "PS-DTFE: the velocity gradient and the fields built on it (divergence, shear, vorticity, V-web) as the "
             "stream average of each stream's own gradient -- the route of the older PS-DTFE grids made "
             "without '--face-gradient' -- instead of the face gradient of the volume-weighted MEAN velocity, the "
             "default wherever it applies (the exact deposit, no "
             "'--ps-halo-release', 3D). Elsewhere a run takes this route by itself, and its "
             "header's 'velocity gradient' line says why. DTFE_FACE_GRADIENT=0 in the environment is the same switch; "
             "the two flags exclude each other.")
#else
            ("per-tet-gradient", po::bool_switch(&(this->perTetGradient)), "standard DTFE (3D): the velocity gradient and the fields built on it (divergence, shear, vorticity, V-web) from each tetrahedron's own gradient added up by the deposit, instead of the exact face integrals ('--face-gradient', the default). The deposit's gradient is float32 on the GPU and differs from the face value by 5e-7 of the gradient rms per sub-domain at 512^3 (294 of 134 M cells by more than 1e-4); the older production grids hold it, and it is the 2D build's only route. DTFE_FACE_GRADIENT=0 in the environment is the same switch; the two flags exclude each other.")
#endif
#ifdef PHASE_SPACE
            ("no-overlap", po::bool_switch(&(this->psNoOverlap)), "PS-DTFE, partitioned GPU runs at '--max-concurrent 1': switch off the look-ahead that builds the NEXT partition's tessellation (particle selection, Delaunay triangulation, vertex densities) on a CPU thread while the current one deposits on the GPU. The look-ahead changes no output byte (each tessellation is built exactly as without it and the merge stays in index order); it holds one more partition's tessellation in memory, which the memory gate and the auto-tuner count, and it is skipped for a partition when the free memory cannot hold it. (Standard DTFE's '--no-overlap' does the same for its GPU interpolation.)")
#else
            ("no-overlap", po::bool_switch(&(this->stdNoOverlap)), "standard DTFE, partitioned runs with '--gpu' (the same switch as PS-DTFE's): switch off the look-ahead that prepares the NEXT partition (its particle selection, its thread sub-boxes, their Delaunay triangulations and vertex densities, or their loads from '--tessellation-cache') on CPU threads while the current partition's volume-averaged fields are computed on the GPU. The look-ahead changes no output byte (each sub-box tessellation is built exactly as without it, and every partition still fills its own part of the grids in turn); it holds the next partition's tessellations in memory, which the auto-tuner counts, and it is skipped for a partition when the free memory, less 4 GB, cannot hold them. DTFE_OVERLAP=0 switches it off as well, DTFE_OVERLAP=1 forces it on a CPU run too (the tests' byte-identity check).")
#endif
            ("parallel-triangulation", po::bool_switch(&(this->parallelTriangulationForced)), "both binaries, builds with the tbb library (TBB=auto/1): FORCE the parallel insert of the particles into the Delaunay triangulation, whatever the tessellation's size. THE DEFAULT RULE: the parallel insert runs for a tessellation of at least 1000000 points when more than one core is free for it -- the performance cores (at most six; DTFE_TRIANGULATION_THREADS=N sets the count and also forces the insert) are divided among the tessellations built at once (a partition loop's concurrency, the standard sub-domain team), so with every core busy the insert stays sequential and a run with few tessellations at once gains. The parallel insert holds about 4.4 GB more per partition in flight on TNG100-3-Dark z = 0 at 512^3 (the auto-tuner budgets it); below the threshold the sequential insert is no slower. The parallel insert's vertex and cell order differ run to run, so what is summed in that order moves at rounding between two identical runs: the standard Monte-Carlo average's CPU float sums, the point sums (~1e-15), and the double builds' CPU sums (the exact average, the face pass, the vertex densities); unchanged: in the single build, standard DTFE's exact '_a' grids on either device, its face pass and its vertex densities (integer sums in one unit for the whole run), and every grid of the single-precision PS-DTFE, whose cells are renumbered in a canonical vertex order before any use. '--sequential-triangulation' keeps the bit-reproducible insert.")
            ("sequential-triangulation", po::bool_switch(&(this->sequentialTriangulation)), "both binaries: insert the particles one by one, in spatial-sort order -- bit-reproducible run to run -- instead of the default rule (the parallel insert for tessellations of at least 1000000 points where cores are free; see '--parallel-triangulation'). DTFE_TRIANGULATION_THREADS=1 is the same switch.")
            ("serve-progress", po::bool_switch(&(this->psServeProgress)), "with '--serve': write a plain progress line to stderr for every partition the server builds or loads ('[partitions k/N] partition p: built|loaded|cached, V vertices, T s'), for every partition a request visits ('[request k/P] partition p: in memory|loading from the cache') and at the end of every request ('[request done] N points, P of M partitions, L loaded from the cache in T s, T s in all'), whatever '--verbose' says. The launcher's progress bar and zoom timings read them; the protocol on stdout is unchanged.")
            ("serve-resident", po::value<int>(&(this->serveResident))->default_value(0), "with '--serve --partition --tessellation-cache': the maximum number of partition tessellations held in memory at once; the others live in the cache and are reloaded when a request needs them (least recently used first out). 0 (default) = as many as fit in the auto-tune memory budget, at least 1, counting each tessellation (~666 B/vertex) and the cell indexes built (one is built only when it fits). Without '--tessellation-cache' every partition stays resident.")
#ifdef TEST_PADDING
            ("no-padding-test", "do not test for the efficiency of the padding when computing the Delaunay tessellation. The default is to use dummy particles positioned at the boundary of the extended padded box to test if the Delaunay tessellation fully covers the region of interest (i.e. the unpadded box).")
#endif
#ifdef PHASE_SPACE
            ("gpu-share", po::value<std::string>(), "PS-DTFE, a partitioned GPU run's exact deposit: the share of the partitions deposited on the GPU, the others on the free CPU cores at the same time (chosen by partition index alone, so a rerun repeats it). auto, the default: the run mode's share (performance from its thread count; gentle and balanced 1). mixed: that model in every run mode, from the mode's threads. A fraction from 0 to 1: that share (1: every partition on the GPU). A mixed run's grids differ from an all-GPU or all-CPU run at ~1e-6 of their rms, and the mass stays exact. The resolved share is in the checkpoint key; '--resume' with auto or mixed keeps the checkpoint's. It is ignored, with a note for a number, on a run on the CPU, for the sampled deposit, in a CPU-only build and for one domain.")
            ("ps-lagrangian-padding", po::value<Real>(&(this->psLagPadding)), "PS-DTFE only, '--partition' runs: pad each Lagrangian partition by this many mean particle spacings (and at least 4.2 '--ps-alpha-shape' cuts) instead of the Eulerian '--padding' fraction, which is floored at 0.3 of a partition and makes each TNG partition triangulate 7-9x the particles it owns. A partition's owned tetrahedra, their vertex densities and the '--ps-vertex-mass' degrees reach at most ~6 circumradii (<= 0.9 spacings on lattice-like initial conditions, at every redshift: the tessellation is Lagrangian), so 8 is a safe margin. Default 8 (far fewer particles per TNG partition than the Eulerian rule; the degeneracy cut does not depend on CGAL's vertex labelling, so the grids equal the Eulerian rule's to float rounding, ~3e-15 in the double build). 0 = the Eulerian rule (it also reuses tessellation caches written under that rule).")
#endif
            ;
    
    
    po::options_description helpOptions("Help");
    helpOptions.add_options()
            ("help,h", "produce summary of help message. For more detailed help use the '--full-help' option.")
            ("full-help", "produce detailed help message. For more detailed help information consult the documentation.")
            ;
    
    
    po::options_description hidden("Hidden options");
    hidden.add_options()
            ("input-file", po::value<std::string>(&(this->inputFilename)), "name of the input position file")
            ("output-file", po::value<std::string>(&(this->outputFilename)), "root name of the output file/files")
            ;
    
    
    // every group into 'visibleOptions' ('--full-help') and into 'allOptions' (the parser).
    visibleOptions.add(inputOptions);
    visibleOptions.add(gridFieldOptions);
    visibleOptions.add(cellAverageOptions);
    visibleOptions.add(deviceOptions);
    visibleOptions.add(splitOptions);
    visibleOptions.add(regionOptions);
    visibleOptions.add(pointOptions);
#ifdef PHASE_SPACE
    visibleOptions.add(phaseSpaceOptions);
#endif
    visibleOptions.add(estimatorOptions);
    visibleOptions.add(otherOptions);
    visibleOptions.add(developerOptions);
    visibleOptions.add(helpOptions);
    
    
    allOptions.add(inputOptions);
    allOptions.add(gridFieldOptions);
    allOptions.add(cellAverageOptions);
    allOptions.add(deviceOptions);
    allOptions.add(splitOptions);
    allOptions.add(regionOptions);
    allOptions.add(pointOptions);
#ifdef PHASE_SPACE
    allOptions.add(phaseSpaceOptions);
#endif
    allOptions.add(estimatorOptions);
    allOptions.add(otherOptions);
    allOptions.add(developerOptions);
    allOptions.add(helpOptions);
    allOptions.add(hidden);
    
    // hidden options as positional arguments
    p.add( "input-file", 1 );
    p.add( "output-file", 1 );
}


// Prints the detailed help (full option descriptions) and exits.
void User_options::helpInformation( po::options_description &visibleOptions, char *progName )
{
    MESSAGE::Message message( 3 );
    message << "Use this program to interpolate fields on a grid using the DTFE method - there are multiple options which allow for increased flexibility of the program. The computations are done using the Delaunay Triangulation module from the CGAL library.\n";
    message << "Usage:    " << progName << "  name_position_file  output(root)_file  'options - see below' \n";
    message << "On top of the above, the user can add any of the following options:\n";
    message << visibleOptions << "\n" << MESSAGE::Flush;
    exit( EXIT_SUCCESS );
}
// Prints the short help ('-h') and exits: the usage line, the two modes and a short entry per option of the same-named
// '--full-help' groups (PS-DTFE and standard lines differ under '#ifdef PHASE_SPACE'). A display-only second list: each
// declaration must keep the short letter and 'arg' of its real declaration in addOptions, and a '--full-help' name
// (tests/docs_drift_check.py rule G). The lines keep boost's format ('  -g [ --grid ] arg   text') at 80 columns:
// tests/run_scripts_check.sh, tests/docs_drift_check.py (help_names) and python/gui/presets.py (parse_help) read it.
void User_options::shortHelp( char *progName )
{
    // (po::value<std::string>() without a bind target: the line shows 'arg', nothing is stored)
    po::options_description inputOptions("Input");
    inputOptions.add_options()
#ifdef PHASE_SPACE
            ("input-format,i", po::value<std::string>(), "the input's file type: 105 Gadget HDF5 (the default), 101 binary Gadget (one file or several); PS-DTFE needs 105 for the initial positions.")
            ("mpc-unit", po::value<std::string>(), "1 Mpc in the input's length unit (the build's default: 1000, kpc).")
            ("periodic,p", "the particles fill a periodic box (from --box or the file's header).")
            ("box", po::value<std::string>(), "the box that holds every particle: x_min x_max y_min y_max, and z_min z_max in 3D (else from the file).")
            ("ps-lagrangian-input", po::value<std::string>(), "an HDF5 snapshot of the initial positions, matched by 'ParticleIDs'; read only when the snapshot has no 'InitialCoordinates'.")
            ("ps-vertex-mass", "tetrahedron masses from the particles' masses, for perturbed initial positions such as TNG's (off here; the drivers and the launcher's TNG jobs pass it).")
#else
            ("input-format,i", po::value<std::string>(), "the input's file type: 105 Gadget HDF5 (the default) or 101 binary Gadget, one file or several.")
            ("mpc-unit", po::value<std::string>(), "1 Mpc in the input's length unit (the build's default: 1000, kpc).")
            ("periodic,p", "the particles fill a periodic box (from --box or the file's header).")
            ("box", po::value<std::string>(), "the box that holds every particle: x_min x_max y_min y_max z_min z_max (4 numbers in 2D; else from the file).")
            ("padding", po::value<std::string>(), "the particles copied around the region, in mean spacings (15 by default, as the drivers and the launcher pass: 5 is too thin near a face in a void-rich box).")
#endif
            ;
    po::options_description gridFieldOptions("Grid and fields");
    gridFieldOptions.add_options()
#ifdef PHASE_SPACE
            ("grid,g", po::value<std::string>(), "grid cells per axis: one number for all axes, or one per axis.")
            ("field,f", po::value<std::string>(), "the fields: density_a (the default), velocity_a, dispersion_a, gradient_a, divergence_a, shear_a, vorticity_a, tweb_a, vweb_a; without '_a': the deposit at one point per cell (exact in an exact run).")
            ("lambda-th", po::value<std::string>(), "the T-web / V-web eigenvalue threshold (0.3 by default, as the drivers pass).")
#else
            ("grid,g", po::value<std::string>(), "grid cells per axis: one number for every axis, or one per axis.")
            ("field,f", po::value<std::string>(), "the fields: density_a (the default), velocity_a, gradient_a, divergence_a, shear_a, vorticity_a, tweb_a, vweb_a; without '_a': the value at the grid point.")
            ("lambda-th", po::value<std::string>(), "the T-web / V-web eigenvalue threshold (0.3 by default, as the drivers pass).")
#endif
            ;
    po::options_description cellAverageOptions("Cell average (the two modes)");
    cellAverageOptions.add_options()
#ifdef PHASE_SPACE
            ("exact", "Exact: each tetrahedron's exact share of every cell (the default for '_a' fields); in 3D also the exact face gradient of the mean flow.")
            ("sampled", "Fast: the deposit at one point per cell axis, noisy per-cell values (smooth before use); the derivatives are the stream average.")
#else
            ("exact", "Exact: the exact cell average of the linear interpolant (the default).")
            ("sampled", "Fast: a Monte-Carlo cell average; in 3D the velocity derivatives stay exact.")
#endif
            ;
    po::options_description deviceOptions("Device and machine use");
    deviceOptions.add_options()
#ifdef PHASE_SPACE
            ("gpu", "the grid deposit on the GPU (a 3D METAL=1 build; else a warning and the CPU). Without it the program picks: the GPU where the deposit runs there, else the CPU; DTFE_GPU=1 is --gpu, DTFE_GPU=0 keeps the CPU.")
#else
            ("gpu", "the cell average on the GPU (a 3D METAL=1 build; else a warning and the CPU). Without it the program picks: the exact average on the GPU where it runs there, the sampled one on the CPU; DTFE_GPU=1 is --gpu, DTFE_GPU=0 keeps the CPU.")
#endif
            ("run-mode", po::value<std::string>(), "how hard the run leans on the machine: balanced (the default; performance inside a batch job: SLURM_JOB_ID or PBS_JOBID set), performance or gentle; DTFE_RUN_MODE is the same.")
            ;
    po::options_description splitOptions("Split, memory and resume");
    splitOptions.add_options()
#ifdef PHASE_SPACE
            ("partition", po::value<std::string>(), "split into this many partitions per axis (default: the auto-tuner plans the split for 60% of the RAM or DTFE_MEM_BUDGET_GB, capped by the free memory).")
            ("max-concurrent", po::value<std::string>(), "at most this many partitions in flight (default: the auto-tuner's).")
            ("scratch-dir", po::value<std::string>(), "keep the full-size grids in memory-mapped files in this local folder.")
            ("auto-tune-report", "print the auto-tuner's plan as one 'AUTO-TUNE-REPORT' line and exit.")
            ("resume", "continue a killed partitioned run from its checkpoint ('<output>.runlog.checkpoint'; with --scratch-dir, in that folder).")
            ("ps-merge", po::value<std::string>(), "how a split run's partitions merge: index-order (the default; the same order every run) or completion-order.")
#else
            ("partition", po::value<std::string>(), "split the box into this many parts per axis (default: the auto-tuner plans the split for its memory budget, which DTFE_MEM_BUDGET_GB can set).")
            ("max-concurrent", po::value<std::string>(), "at most this many triangulations at once (default: the auto-tuner's).")
            ("scratch-dir", po::value<std::string>(), "keep the full-size grids in memory-mapped files in this local folder.")
            ("auto-tune-report", "print the auto-tuner's plan as one 'AUTO-TUNE-REPORT' line and exit.")
            ("resume", "continue a killed partitioned run from its checkpoint ('<output>.runlog.checkpoint'; with --scratch-dir, in that folder).")
#endif
            ;
    po::options_description estimatorOptions("Other estimators");
    estimatorOptions.add_options()
            ("ngp", "choose NGP (Nearest Grid Point) for grid interpolation.")
            ("cic", "choose CIC (Cloud In Cell) for grid interpolation.")
            ("tsc", "choose TSC (Triangular Shape Cloud) for grid interpolation.")
            ("pcs", "choose PCS (Piecewise Cubic Spline) for grid interpolation.")
            ("sph", po::value<std::string>(), "choose SPH for grid interpolation (argument = number nearest neighbors).")
            ("voronoi", "use Voronoi volume density with NGP grid assignment (density only).")
            ("interlace", "enable interlacing to reduce aliasing (Fourier-space averaging of half-cell offset density grids; the T-web comes from the first pass).")
            ;
    po::options_description helpOptions("Help");
    helpOptions.add_options()
            ("help,h", "this list.")
            ("full-help", "every option in detail; the developer options in a group of their own.")
            ;
    
    po::options_description shortList;   // no caption, boost's default 80 columns: one description column
    shortList.add(inputOptions);
    shortList.add(gridFieldOptions);
    shortList.add(cellAverageOptions);
    shortList.add(deviceOptions);
    shortList.add(splitOptions);
    shortList.add(estimatorOptions);
    shortList.add(helpOptions);
    
    MESSAGE::Message message( 3 );
#ifdef PHASE_SPACE
    message << "Usage: " << progName << " <snapshot> <output root> [options]\n";
    message << "Two modes: --exact (the default for '_a' fields) and --sampled.\n--full-help lists every option.\n";
#else
    message << "Usage: " << progName << " <input file> <output root> [options]\n";
    message << "Two modes: --exact (the default) and --sampled.\n--full-help lists every option.\n";
#endif
    message << shortList << "\n" << MESSAGE::Flush;
    exit( EXIT_SUCCESS );
}




// Print the options the program will use.
void User_options::printOptions()
{
    std::string uField;
    if ( this->uField.density ) uField = " density,";
    if ( this->uField.velocity ) uField += " velocity,";
    if ( this->uField.velocity_gradient ) uField += " velocity gradient,";
    if ( this->uField.velocity_divergence ) uField += " velocity divergence,";
    if ( this->uField.velocity_shear ) uField += " velocity shear,";
    if ( this->uField.velocity_vorticity ) uField += " velocity vorticity,";
    if ( this->uField.scalar ) uField += " scalar,";
    if ( this->uField.scalar_gradient ) uField += " scalar gradient,";
    if ( this->uField.triangulation ) uField += " triangulation,";
    if ( not this->uField.selected() and this->NGP ) uField = "none since NGP interpolation";
    else if ( not this->uField.selected() and this->CIC ) uField = "none since CIC interpolation";
    else if ( not this->uField.selected() and this->TSC ) uField = "none since TSC interpolation";
    else if ( not this->uField.selected() and this->SPH ) uField = "none since SPH interpolation";
    else if ( not this->uField.selected() ) uField = "none";
    
    std::string aField;
    if ( this->aField.density ) aField = " density,";
    if ( this->aField.velocity ) aField += " velocity,";
    if ( this->aField.velocity_gradient ) aField += " velocity gradient,";
    if ( this->aField.velocity_divergence ) aField += " velocity divergence,";
    if ( this->aField.velocity_shear ) aField += " velocity shear,";
    if ( this->aField.velocity_vorticity ) aField += " velocity vorticity,";
    if ( this->aField.velocity_std ) aField += " velocity standard deviation,";
    if ( this->aField.scalar ) aField += " scalar,";
    if ( this->aField.scalar_gradient ) aField += " scalar gradient,";
    if ( not this->aField.selected() ) aField = "none";
    
    std::string interpolationMethod = "DTFE";
    if ( this->CIC ) interpolationMethod = "CIC";
    else if ( this->NGP ) interpolationMethod = "NGP";
    else if ( this->TSC ) interpolationMethod = "TSC";
    else if ( this->PCS ) interpolationMethod = "PCS";
    else if ( this->Voronoi ) interpolationMethod = "Voronoi (NGP)";
    else if ( this->SPH )
    {
        char temp[100];
        snprintf( temp, sizeof(temp), "%d neighbors", this->SPH_neighbors );
        interpolationMethod = "SPH ";
        interpolationMethod += temp;
    }
    
    std::string inFileType = "unknown";
    if ( this->inputFileType==101 ) inFileType = "Gadget multiple files";
    else if ( this->inputFileType==105 ) inFileType = "Gadget HDF5 file/files";
    else if ( this->inputFileType==111 ) inFileType = "text file with positions (first 3 columns and weights in 4th column)";
    
    std::string outFileType = "unknown";
    if ( this->outputFileType==101 ) outFileType = "binary file";
    else if ( this->outputFileType==100 ) outFileType = "density binary file";
    else if ( this->outputFileType==110 ) outFileType = "text file";
    
    
    MESSAGE::Message message( verboseLevel );
    message << MESSAGE::banner("RUN CONFIGURATION");
    message << MESSAGE::cBold() << "RUNNING: " << this->programOptions << MESSAGE::cReset() << "\n\n";
    message << MESSAGE::cCyan() << "The program will interpolate to grid the chosen field using the DTFE method with the following input parameters:" << MESSAGE::cReset() << "\n"
            << "\t unaveraged field(s)    : " << uField << "\n"
            << "\t averaged field(s)      : " << aField << "\n"
            << "\t interpolation method   : " << interpolationMethod << "\n"
            << "\t input data file        : " << this->inputFilename << "\n"
            << "\t input data file type   : " << this->inputFileType << " - " << inFileType << "\n"
            << "\t input data blocks      : " << MESSAGE::printElements( this->readParticleData, "  " ) << "\n"
            << "\t input particle species : " << MESSAGE::printElements( this->readParticleSpecies, "  " ) << "\n"
            << "\t output file            : " << this->outputFilename << "\n"
            << "\t output file type       : " << this->outputFileType << " - " << outFileType << "\n";
    if ( not this->gridSize.empty() )
        message << "\t grid size              : " << MESSAGE::printElements( this->gridSize, "  " ) << (this->regionOn ? "   for the box region selected by the user\n" : "   for the full particle box\n" );
    else
        message << "\t grid size              : none specified at the moment";
    if ( not boxCoordinates.isNullBox() )
        message << "\t box coordinates        : [" << MESSAGE::printElements( boxCoordinates, ", " ) << "]\n";
    if ( this->periodic )
        message << "\t computing the grid interpolation in a PERIODIC box\n";
    
    
    if ( this->regionOn )
    {
        char const axisName[] = "xyz";
        message << "\t computing the grid interpolation only for the user specified region of coordinates:\n";
        for (int d=0; d<NO_DIM; ++d)
            message << "\t\t " << axisName[d] << " extension : " << region[2*d] << "   " << region[2*d+1] << (regionMpcOn?"  Mpc":"  box length") << "\n";
    }
    
    
    if ( this->partitionOn )
    {
        message << "\t splitting the data in  : " << MESSAGE::printElements( partition, "  ") << "   separate data sets on which to apply the Delaunay triangulation.\n";
        if ( this->partNo>=0 )
            message << "\t computing grid interpolation ONLY for data set " << partNo << " of the partitioned data\n";
    }
    if ( this->maxConcurrent>0 )
        message << "\t max concurrent triangs : " << this->maxConcurrent << "   (capping concurrent Delaunay triangulations to bound peak memory)\n";
#ifdef PHASE_SPACE
    // the merge order (index order, staged, is the default): said when the run is split, or asked for
    if ( this->partitionOn or this->psOrderedMergeFlag or this->psCompletionOrderMergeFlag )
    {
        if ( this->psOrderedMerge )
            message << "\t partition merge order  : partition index (bit-reproducible sums; staged, the default"
                    << ( this->psResume and this->psCompletionOrderMergeFlag ? "; --resume overrides --ps-merge completion-order" : "" )
                    << ")\n";
        else
            message << "\t partition merge order  : completion order (--ps-merge completion-order; not bit-reproducible)\n";
    }
    if ( this->psNoOverlap )
        message << "\t look-ahead tessellation: off (--no-overlap)\n";
    else if ( this->psLookAheadOffBudget )
        message << "\t look-ahead tessellation: off (one partition plus the look-ahead does not fit the memory budget)\n";
#endif
#ifndef PHASE_SPACE
    if ( this->stdNoOverlap )
        message << "\t look-ahead tessellation: off (--no-overlap)\n";
    else if ( this->stdLookAheadOffBudget )
        message << "\t look-ahead tessellation: off (the busiest partition plus the look-ahead does not fit the memory budget)\n";
#endif
    if ( this->partitionGiven and not this->partitionOn )
        message << "\t splitting the data in  : 1   (--partition 1: a single triangulation, no auto-selected split)\n";
    if ( not this->partitionOn and not this->partitionGiven and not this->maxConcurrentOn and this->partNo<0 and not this->redshiftConeOn )
        message << "\t partition/concurrency  : auto (chosen from the data size, grid, fields and available RAM once the input is read; pass --partition / --max-concurrent to override)\n";
    
    
    if ( not this->paddingLength.isNullBox() )
    {
        char const padAxisName[] = "xyz";
        message << "\t padding length:\n";
        for (int d=0; d<NO_DIM; ++d)
            message << "\t\t " << padAxisName[d] << " axis : " << paddingLength[2*d] << "   " << paddingLength[2*d+1] << (paddingMpcOn?"  Mpc":"  box length") << "\n";
    }
    else
        message << "\t number padding particles: " << this->paddingParticles << "\n";
#ifndef PHASE_SPACE
    // suppressed in PS-DTFE: no dummy-boundary-test particles, and it uses nSub^NO_DIM
    // sub-grid averaging instead of the Monte-Carlo grid-cell averaging reported below
    if ( this->DTFE and this->testPaddedBoundaries )
        message << "\t the DTFE computation will add DUMMY TEST particles to test the efficiency of the padding\n";


    if ( this->aField.selected() and this->DTFE )
    {
        std::string averagingMethod;
        if ( this->method==1 ) averagingMethod = "Monte Carlo sampling using quasi-random points inside the Delaunay cells";
        else if ( this->method==2 ) averagingMethod = "Monte Carlo sampling using random points inside the sampling cells";
        else averagingMethod = "unknown";
        message << "\t volume averaging method: " << this->method << " - " << averagingMethod << "\n"
                << "\t number sampling points : " << this->noPoints << "\n";
        if ( this->method==2 )
            message << "\t random generator seed  : " << this->randomSeed << "\n";
        if ( averageDensity>Real(0.) )
            message << "\t density scaling value  : " << averageDensity << "\n";
        // the device of the '_a' work and why (checkGpuOptions), on the GPU and on the CPU alike (empty only for a
        // caller that prints without readOptions)
        if ( not this->deviceLine.empty() )
            message << "\t '_a' interpolation     : " << this->deviceLine << "\n";
    }
#endif
    
    
    if ( redshiftConeOn )
        message << "\t Computing the grid interpolation to a redshift cone grid of coordinates " << (NO_DIM==2? "[r_min, r_max, psi_min, psi_max]" : "[r_min, r_max, theta_min, theta_max, psi_min, psi_max]") << " = [" << MESSAGE::printElements( redshiftCone, ", " ) << "]\n"
            << "\t The origin of the light cone is (x,y,z) : (" << MESSAGE::printElements( originPosition, ", " ) << ")\n";
    
#ifdef PHASE_SPACE
    message << "\t PS-DTFE mode           : enabled (triangulation in Lagrangian space)\n";
    if ( this->aField.selected() )
    {
        int nSubPts = 1;
        for (int d=0; d<NO_DIM; ++d) nSubPts *= (this->psAvgSubsamples<1 ? 1 : this->psAvgSubsamples);
        if ( not this->psExactDeposit )   // the exact deposit clips: its 'grid deposit' line names where it still reads nSub
            message << "\t avg sub-samples (nSub) : " << this->psAvgSubsamples << "   ("
                    << ( nSubPts == 1 ? std::string( "one point per '_a' cell, at its centre: noisy per-cell values, smooth before use" )
                                      : "each '_a' cell volume-averaged over nSub^" + std::to_string( NO_DIM ) + " = "
                                        + std::to_string( nSubPts ) + " points: more points, a smoother cell average and more time" )
                    << ")\n";
    }
    // the device of the deposit and why (checkGpuOptions), on the GPU and on the CPU alike, for every run whose
    // deposit makes a field, averaged or not (empty only for a caller that prints without readOptions)
    if ( ( this->aField.selected() or this->uField.selected() ) and not this->deviceLine.empty() )
        message << "\t PS deposit             : " << this->deviceLine << "\n";
    // with the option only the rule is known here: the read decides, and readInputData says when the file is not read
    if ( not this->lagrangianInputFilename.empty() )
        message << "\t Lagrangian positions   : the snapshot's 'InitialCoordinates' when it has them, else '"
                << this->lagrangianInputFilename << "' (--ps-lagrangian-input, matched by ParticleIDs)\n";
    else
        message << "\t Lagrangian positions   : read from 'InitialCoordinates' dataset in main input file\n";
    if ( this->psServe )
        message << "\t point evaluation       : interactive server on stdin/stdout (--serve; per-stream density: linear"
                << ( this->psVertexMass ? ", vertex-share masses" : "" )
                << ", volume-weighted stream means"
                << (this->partitionGiven ? ", composite of partition tessellations" : "")
                << (this->psPtsScalar ? ", scalar values" : "")
                << (this->psPtsLagrangian ? ", Lagrangian positions" : "") << ")\n";
    if ( not this->psSamplePointsFile.empty() )
        message << "\t point evaluation       : at the sample points in '" << this->psSamplePointsFile
                << "' (per-stream density: linear"
                << ( this->psVertexMass ? ", vertex-share masses" : "" )
                << ", volume-weighted stream means"
                << (this->psPerStream ? ", writing per-stream records" : "")
                << (this->psPerStreamIds ? " + stream ids" : "")
                << (this->psPtsDenGrad ? ", writing density gradients" : "")
                << (this->psPtsVelGrad ? ", writing velocity gradients" : "")
                << (this->psPtsScalar ? ", writing scalar values" : "")
                << (this->psPtsLagrangian ? ", writing Lagrangian positions" : "") << ")\n";
    if ( this->psCaustics )
        message << "\t caustic flagging       : writing the fold-caustic cell flag to '.caustic' (--ps-caustics)\n";
    if ( not this->periodic )
        message << "\t Lagrangian domain      : "
                << ( this->psAlphaShape > Real(0.) ? "alpha shape, tetrahedra with circumradius <= " : "the whole convex hull (--ps-alpha-shape 0)" );
    if ( not this->periodic and this->psAlphaShape > Real(0.) )
        message << this->psAlphaShape << " mean particle spacings (--ps-alpha-shape)";
    if ( not this->periodic )
        message << "\n";
    if ( this->psHaloRelease > Real(0.) )
        message << "\t halo-interior release  : tetrahedra with rho_geo/rho_bar > " << this->psHaloRelease
                << " deposit monolithically at their centroid cell (--ps-halo-release; mass-conserving)\n";
    // which deposit runs, and why (resolveAverageMode): the exact one drives every grid pass, so it is named whenever
    // it runs; the sampled one where an '_a' field of the PS-DTFE method asks for it
    if ( this->psExactDeposit )
        message << "\t grid deposit           : EXACT tetrahedron-cell intersection moments (" << this->averageModeSource << "; "
                << ( this->psUseMetal ? "z-sweep on the GPU" : "r3d clipping" ) << "; count "
                << ( this->psAvgSubsamples < 1 ? 1 : this->psAvgSubsamples )
                << ( this->aField.selected()
                     ? " only for a tetrahedron too thin to clip and the sub-grid ring in the '_a' pass)\n"
                     : " not used: the unaveraged pass samples a tetrahedron too thin to clip at one point)\n" );
    else if ( this->aField.selected() and this->DTFE and not this->Voronoi and not this->averageModeSource.empty() )
        message << "\t grid deposit           : SAMPLED on nSub^" << NO_DIM << " sub-points (" << this->averageModeSource << ")\n";
    // the velocity moments' weighting, in every PS-DTFE run: the readers of a run log (python/dtfelib/io.py
    // run_velocity_weighting, python/gui/results.py) take this line as the record that its grids' velocity moments
    // are volume-weighted
    message << "\t velocity moments       : volume-weighted (the dispersion mass-weighted)\n";
    // the velocity gradient's route (checkGpuOptions), where a field built on the gradient is computed: the mean flow's
    // from the face sums, or the stream average of each stream's own gradient and why ('.a_velDivStreams' comes with an
    // '_a' field of that family only)
    bool const aGradient = this->aField.velocity_gradient or this->aField.selectedVelocityDerivatives();
    if ( aGradient or this->uField.velocity_gradient or this->uField.selectedVelocityDerivatives() )
    {
        if ( this->faceGradient )
            message << "\t velocity gradient      : from EXACT FACE sums, the gradient of the volume-weighted MEAN "
                       "velocity (" << this->faceGradientSource
                    << ( aGradient ? "; '.a_velDivStreams' holds the stream average's divergence" : "" )
                    << "; '--per-tet-gradient' keeps the stream average)\n";
        else
            message << "\t velocity gradient      : the stream average of each stream's own gradient ("
                    << this->faceGradientSource << ")\n";
    }
#else
    if ( this->psServe )
        message << "\t point evaluation       : interactive server on stdin/stdout (--serve; standard DTFE interpolant"
                << (this->partitionGiven ? ", composite of partition tessellations" : "")
                << (this->psPtsScalar ? ", scalar values" : "") << ")\n";
    if ( not this->psSamplePointsFile.empty() )
        message << "\t point evaluation       : at the sample points in '" << this->psSamplePointsFile
                << "' (standard DTFE interpolant, 0/1 coverage"
                << (this->psPtsDenGrad ? ", writing density gradients" : "")
                << (this->psPtsVelGrad ? ", writing velocity gradients" : "")
                << (this->psPtsScalar ? ", writing scalar values" : "") << ")\n";
    // which '_a' average runs, and why (resolveAverageMode): for the DTFE method's '_a' fields
    if ( this->aField.selected() and this->DTFE and not this->Voronoi and not this->averageModeSource.empty() )
    {
        if ( this->exactAverage )
            message << "\t volume averaging       : EXACT cell-tetrahedron integration of the linear interpolant (" << this->averageModeSource << ( this->gpuPicked ? "; r3d, or a z-sweep on the GPU; --subsamples ignored)\n" : "; r3d, or a z-sweep on the GPU with --gpu; --subsamples ignored)\n" );
        else
            message << "\t volume averaging       : SAMPLED (" << this->averageModeSource << ")\n";
    }
    if ( this->faceGradient )   // the source is empty only for a library caller that sets faceGradient itself
        message << "\t velocity gradient      : from EXACT FACE integrals of the interpolant (Gauss; "
                << ( this->faceGradientSource.empty() ? std::string() : this->faceGradientSource + "; " )
                << "'--per-tet-gradient' for the deposit's own)\n";
    else if ( this->perTetGradient or this->uField.velocity_gradient or this->uField.selectedVelocityDerivatives()
              or this->aField.velocity_gradient or this->aField.selectedVelocityDerivatives() )
        message << "\t velocity gradient      : each tetrahedron's own, added up by the deposit ("
                << this->faceGradientSource << ")\n";
#endif
    // the triangulation's insert (both binaries)
    if ( this->psParallelTriangulation )
#if NO_DIM==3
    {
        if ( this->parallelTriangulationForced )
            message << "\t triangulation          : parallel insertion for every tessellation (--parallel-triangulation; the sums in walk order are then not bit-reproducible run to run)\n";
        else
            message << "\t triangulation          : parallel insertion for tessellations of at least " << parallelInsertMinPoints << " points where more than one core is free (the default; --sequential-triangulation keeps the bit-reproducible insert)\n";
    }
#else
        message << "\t triangulation          : sequential (the parallel insertion is 3D only)\n";
#endif
    else
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
        message << "\t triangulation          : sequential insertion (bit-reproducible; --parallel-triangulation builds on several cores where they are free)\n";
#else
        message << "\t triangulation          : sequential insertion (this build has no parallel insert: TBB=auto/1 or TBB_STD=1 with the tbb library)\n";
#endif
    // the run mode (resolveRunMode, applyRunMode): what it set and why, on every run -- the threads (an
    // OMP_NUM_THREADS kept, a '--max-concurrent' that raised them), the GPU buffers of a GPU run, gentle's nice
    {
        RunModeState const &rm = runModeState();
        message << "\t run mode               : " << runModeName( this->runMode ) << " (" << this->runModeSource << "; ";
        if ( this->coreBudgetNote.rfind( "kept:", 0 ) == 0 )
            message << rm.ompDefault << " threads (" << this->coreBudgetNote << ")";
        else
            message << this->coreBudget << " of " << rm.ompDefault << " threads"
                    << ( this->coreBudgetNote.empty() ? std::string() : " (" + this->coreBudgetNote + ")" );
#if defined(PS_GPU)
        bool const gpuBuffers = this->psUseMetal;
#elif defined(DTFE_GPU)
        bool const gpuBuffers = this->useMetal;
#else
        bool const gpuBuffers = false;   // (this build has no GPU path)
#endif
        if ( gpuBuffers )
            message << "; GPU buffers up to " << std::lround( rm.gpuTargetSec * 1000. ) << " ms";
        if ( not this->priorityNote.empty() )
            message << "; " << this->priorityNote;
        message << ")\n";
    }
#ifdef PHASE_SPACE
    // the GPU share of a GPU run: 'auto', the default, where it is resolved (DTFE.cpp psResolveGpuShare: the exact
    // deposit of a build with the 3D GPU deposit); 'mixed' or a number given on any GPU run
#ifdef GPU_PATH_BUILT
    bool const autoShareResolved = this->psExactDeposit;
#else
    bool const autoShareResolved = false;
#endif
    if ( this->psUseMetal and ( this->psGpuShareSpec != -1 or autoShareResolved ) )
        message << "\t GPU share              : " << this->psGpuShareText << " ("
                << ( this->psGpuShareSpec == -2 ? "the GPU and the free cores in every mode, resolved with the plan"
                     : this->psGpuShareSpec < 0 ? "the run mode's share, resolved with the plan" : "--gpu-share" )
                << ")\n";
#endif
    message << "\t 1Mpc = " << MpcValue << " in units of input data.\n";
    if ( poisson>0 )
        message << "\t Particle positions will be generated randomly. Particle number = " << (NO_DIM==2? poisson*poisson : poisson*poisson*poisson) << ".\n";

#ifdef REDSHIFT_SPACE
    if ( transformToRedshiftSpaceOn )
        message << "\t Transforming from position-space to redshift space using the velocity along the direction :  ( " << MESSAGE::printElements( transformToRedshiftSpace, ", " ) << " )\n";
#endif
    
    
    if ( this->gridSize.empty() )
        message << "\n\n~~~WARNING~~~ No grid size specified. Unless the input data file specifies the grid size for the output result, the program will end with and error message!\n\n";
    
    message << "\n" << MESSAGE::Flush;
}




// Parses the command line and, with '--config', the configuration file into 'vm' (the command line wins: boost keeps
// the value stored first), and only then notifies: every bound member is written from its key.
void User_options::parseOptions(int argc, char *argv[], po::options_description &allOptions,
                                po::positional_options_description &p, po::variables_map &vm,
                                MESSAGE::Message &showMessage)
{
    try
    {
        po::store( po::command_line_parser(argc, argv).options(allOptions).positional(p).run(), vm );
    }
    catch (exception& e)
    {
        throwError( "When reading the command line program options:\n\t\"", e.what(), "\"\n" );
    }
    catch (...)
    {
        throwError( "Unknown error when reading the command line program options!" );
    }
    
    
    
    if ( vm.count("config") )
    {
        std::string const file = vm["config"].as<std::string>();   // (configFilename is written by the notify below)
        ifstream ifs( file.c_str() );
        if (!ifs)
            showMessage << "~~~ERROR~~~ Can not open the configuration file '" << file << "'.\n";
        else
        {
            try
            {
                store( parse_config_file( ifs, allOptions ), vm );
            }
            catch (exception& e)
            {
                throwError( "When reading the configuration file program options:\n\t\"", e.what(), "\"\n" );
            }
            catch (...)
            {
                throwError( "Unknown error when reading the configuration file program options!" );
            }
        }
    }
    
    try
    {
        po::notify(vm);
    }
    catch (exception& e)
    {
        throwError( "When reading the program options:\n\t\"", e.what(), "\"\n" );
    }
}


// Stops when the input or output file is missing, else prints the help '--help' / '--full-help' ask for.
void User_options::checkFileNameOptions(char *argv[], bool getFileNames, po::options_description &visibleOptions,
                                        po::variables_map &vm, MESSAGE::Message &showMessage)
{
    bool const fullHelp = vm.count("full-help");
    if ( not (vm.count("help") or fullHelp) and not vm.count("input-file") and getFileNames )
        showMessage << "~~~ERROR~~~ No input file detected.\n";
    if ( not (vm.count("help") or fullHelp) and not vm.count("output-file") and getFileNames )
        showMessage << "~~~ERROR~~~ No output file detected.\n";
    if ( not (vm.count("help") or fullHelp) and (not vm.count("input-file") or not vm.count("output-file")) and getFileNames )
        exit( EXIT_SUCCESS );
    
    if ( vm.count("help") )
        this->shortHelp( argv[0] );
    else if ( fullHelp )
        this->helpInformation( visibleOptions, argv[0] );
}


// Refuses mutually exclusive options and options given without the option they depend on.
static void checkConflictingOptions(po::variables_map &vm)
{
    conflicting_options(vm, "sph", "tsc");
    conflicting_options(vm, "sph", "cic");
    conflicting_options(vm, "sph", "pcs");
    conflicting_options(vm, "cic", "tsc");
    conflicting_options(vm, "cic", "pcs");
    conflicting_options(vm, "tsc", "pcs");
    conflicting_options(vm, "ngp", "cic");
    conflicting_options(vm, "ngp", "tsc");
    conflicting_options(vm, "ngp", "pcs");
    conflicting_options(vm, "ngp", "sph");
    conflicting_options(vm, "voronoi", "ngp");
    conflicting_options(vm, "voronoi", "cic");
    conflicting_options(vm, "voronoi", "tsc");
    conflicting_options(vm, "voronoi", "pcs");
    conflicting_options(vm, "voronoi", "sph");
    conflicting_options(vm, "region", "region-mpc");
    conflicting_options(vm, "padding", "padding-mpc");
    
    option_dependency(vm, "partition-index", "partition");
    option_dependency(vm, "redshift-cone", "cone-origin");
}


// Reads the main options: '--grid', '--box', '--input-format' and '--periodic'.
void User_options::readMainOptions(po::variables_map &vm)
{
    if ( vm.count("grid") )
    {
        int const temp = this->gridSize.size();
        if ( temp==1 ) {
            this->gridSize = std::vector<size_t>(NO_DIM, this->gridSize.at(0));
        } else if ( temp!=NO_DIM ) {
            throwError( "You can only insert 1 or ", NO_DIM, " values for the '-g [ --grid ]' option (e.g. '-g 256' or '-g 128 256 256')." );
        }
        for (size_t i=0; i<this->gridSize.size(); ++i) {
            lowerBoundCheck( this->gridSize[i], size_t(1), "the values of option '-g [ --grid ]'" );
        }
    }
    else if ( this->psServe )
        this->gridSize = std::vector<size_t>( NO_DIM, size_t(2) );   // --serve writes no grid: skip the default-size grid allocations
    if ( vm.count("box") )      // read the box coordinates
    {
        int const temp = vm["box"].as< std::vector<Real> >().size();
        if ( temp!=2*NO_DIM ) throwError( "You need to specify ", 2*NO_DIM, " arguments with the option '--box'." );
        
        boxCoordinates.coords = vm["box"].as< std::vector<Real> >();
        for (int i=0; i<NO_DIM; ++i)
            if( boxCoordinates[2*i+1]<boxCoordinates[2*i]) throwError( "The right coordinate of the box encompassing the data along axis " , i+1, " (1=x, 2=y, 3=z) must be larger than the left coordinate of the box. This is not the case in the arguments supplied to '--box' option." );
        userGivenBoxCoordinates = true;
    }
    if ( vm.count("input-format") )    // file type, data blocks to read, and particle species to read
    {
        std::vector<int> temp = vm["input-format"].as< std::vector<int> >();
        inputFileType = temp[0];
        if ( temp.size()>=2 )   // data blocks
            for (size_t i=0; i<readParticleData.size(); ++i)
            {
                readParticleData[i] = temp[1] % 2;
                temp[1] /= 2;
            }
        if ( temp.size()>=3 )   // particle species
            for (size_t i=0; i<readParticleSpecies.size(); ++i)
            {
                readParticleSpecies[i] = temp[2] % 2;
                temp[2] /= 2;
            }
    }
    if ( vm.count("periodic") )
        this->periodic = this->periodicInput = true;
}


// Reads the '--field' choices (density_a when none is given).
void User_options::readFieldOptions(po::variables_map &vm)
{
    if ( vm.count("field") )
    {
        for (size_t i=0; i<vm["field"].as< std::vector<std::string> >().size(); ++i)
        {
            std::string field = vm["field"].as< std::vector<std::string> >().at(i);
            bool uFieldOption = this->uField.updateChoices( field, "triangulation", "density", "velocity", "gradient", "divergence", "shear", "vorticity", "", "scalar", "scalarGradient", "tweb", "vweb", "dispersion" );
            bool aFieldOption = this->aField.updateChoices( field, "", "density_a", "velocity_a", "gradient_a", "divergence_a", "shear_a", "vorticity_a",  "velocityStd_a", "scalar_a", "scalarGradient_a", "tweb_a", "vweb_a", "dispersion_a" );
            if ( not(uFieldOption or aFieldOption) ) throwError( "Unknown value '" + field + "' for the option '--field'." );
        }
    }
    else
        this->aField.density = true;
#ifndef PHASE_SPACE
    // the dispersion between streams exists only in PS-DTFE: this binary would leave its grids empty, which the
    // sub-domain merge then reads (an internal error in 3D, a segfault in 2D)
    if ( this->uField.velocity_dispersion or this->aField.velocity_dispersion )
        throwError( "'dispersion' and 'dispersion_a' (the mass-weighted velocity dispersion between streams) are PS-DTFE fields: the standard DTFE interpolant has one velocity at each point, so it has no dispersion to compute. Run PS-DTFE for them." );
#endif
}


// Reads and checks '--grid-window', '--region' and '--region-mpc'.
void User_options::readRegionOptions(po::variables_map &vm)
{
    if ( vm.count("grid-window") )
    {
        this->psWindow = vm["grid-window"].as< std::vector<Real> >();
        if ( this->psWindow.size() != size_t(2*NO_DIM) )
            throwError( "'--grid-window' needs 2*NO_DIM values: x0 x1 y0 y1 z0 z1 (the output coordinates)." );
        for (int d = 0; d < NO_DIM; ++d)
            if ( not (this->psWindow[2*d] < this->psWindow[2*d+1]) )
                throwError( "'--grid-window': each axis needs lower < upper." );
        this->psWindowOn = true;
#ifndef PHASE_SPACE
        throwError( "'--grid-window' is a PS-DTFE option (the standard binary crops a region with '--region-mpc')." );
#endif
        if ( this->uField.velocity_tweb or this->aField.velocity_tweb or this->uField.velocity_vweb or this->aField.velocity_vweb )
            throwError( "'--grid-window' cannot be combined with the T-web / V-web classes: they need the full periodic grid." );
        if ( vm.count("region") or vm.count("region-mpc") or vm.count("partition-index") )
            throwError( "'--grid-window' cannot be combined with '--region', '--region-mpc' or '--partition-index'." );
    }
    if ( vm.count("region-mpc") )
        this->regionMpcOn = true;
    if ( vm.count("region-mpc") or vm.count("region") )
    {
        this->regionOn = true;
        size_t temp = 0;
        if (this->regionMpcOn)
        {
            temp = vm["region-mpc"].as< std::vector<Real> >().size();
            this->region.coords = vm["region-mpc"].as< std::vector<Real> >();
        }
        else
        {
            temp = vm["region"].as< std::vector<Real> >().size();
            this->region.coords = vm["region"].as< std::vector<Real> >();
        }
        if ( temp!=2*NO_DIM ) throwError( "You have to insert ", 2*NO_DIM, " values for the '--region' or '--region-mpc' option (e.g. '--region 0.4 0.6 0.3 0.7 0.45 0.55')." );
        for (int i=0; i<NO_DIM; ++i)
            if( region[2*i+1]<region[2*i]) throwError( "The right coordinate of the region of interest along axis ", i+1, " (1=x, 2=y, 3=z) must be larger than the left coordinate. This is not the case in the arguments supplied to '--region' option." );
    }
}


#ifdef PHASE_SPACE
// The count an EXACT PS-DTFE run without '--subsamples' keeps (resolveAverageMode). The exact deposit ignores it where it
// clips, but still reads it for the slivers r3d leaves with no piece (they fall through to the sampled path,
// ps_interpolation.cc) and for a partition sub-grid box's one-cell ring when above 1 (psSubgridBox). The sampled default is 1;
// 3 keeps an exact run's outputs byte for byte; a run with no '_a' field keeps it too (one point per grid pass).
static constexpr int kPsExactSliverSubsamples = 3;
#endif

// The '_a' average, EXACT (standard: cell integrals of the interpolant; PS-DTFE: the exact deposit) or SAMPLED, into
// exactAverage / psExactDeposit, with the reason averageModeSource gives the run log. First matching rule wins: both
// '--exact' and '--sampled': refused; '--sampled'; '--exact'; DTFE_EXACT_AVERAGE=1 as '--exact', an off value as
// '--sampled' (a flag beats the pin); a sampling count or method given: sampled; else exact where it applies, else sampled
// silently. The exact default never turns a command line into an error (an explicit '--exact' refused elsewhere samples here).
void User_options::resolveAverageMode(po::variables_map &vm)
{
    if ( this->exactFlag and this->sampledFlag )
        throwError( "'--exact' and '--sampled' exclude each other: the '_a' cell averages are integrated exactly or sampled." );
    int const pin = Env::tristate( "DTFE_EXACT_AVERAGE" );   // 1 = as '--exact', 0 = as '--sampled', -1 = unset or auto
    bool const otherMethod = vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph") or vm.count("voronoi");
    Field a = this->aField;   // the '_a' fields the run computes
#ifndef VELOCITY
    a.deselectVelocity();
#endif
#ifndef SCALAR
    a.deselectScalar();
#endif
    bool exact = false;
    std::string source;       // empty: no header line (another method; PS-DTFE: no '_a' field)
#ifdef PHASE_SPACE
    // '--subsamples' has a default_value (1), so only .defaulted() tells a given one
    bool const countGiven = vm.count("subsamples") and not vm["subsamples"].defaulted();
    if ( this->sampledFlag ) source = "--sampled";
    else if ( this->exactFlag ) { exact = true; source = "--exact"; }
    else if ( pin == 1 ) { exact = true; source = "--exact from DTFE_EXACT_AVERAGE=1"; }
    else if ( pin == 0 ) source = "--sampled from DTFE_EXACT_AVERAGE=0";
    else if ( countGiven ) source = "a --subsamples count was given; --exact deposits every tetrahedron exactly";
    else if ( not otherMethod and a.selected() ) { exact = true; source = "--exact, the default"; }
    // else another method, or no '_a' field: the sampled deposit (an unaveraged grid keeps its point values)
    this->psExactDeposit = exact;
    // the count's default (1) is the sampled '_a' deposit's alone: every other run without a count (an exact one, one
    // with no '_a' field, another method's) keeps kPsExactSliverSubsamples where it still reads the count
    bool const sampledAverage = not exact and not otherMethod and a.selected();
    if ( not countGiven and not sampledAverage ) this->psAvgSubsamples = kPsExactSliverSubsamples;
#else
    // .defaulted(): '--average-method' carries a default_value
    bool const methodGiven = vm.count("average-method") and not vm["average-method"].defaulted();
    bool const countGiven = vm.count("subsamples") or methodGiven;
    bool const cone = vm.count("redshift-cone") > 0;
    // where the default cannot be exact, the reason (the explicit '--exact' is refused there, or samples on a cone)
    char const *noExact = a.velocity_std ? "velocityStd_a has no exact form"
                        : cone ? "a redshift cone grid has no exact form" : nullptr;
#ifdef MY_SCALAR
    if ( noExact == nullptr ) noExact = "the MY_SCALAR build has no exact form";
#endif
    if ( this->sampledFlag ) source = "--sampled";
    else if ( this->exactFlag or pin == 1 )
    {
        exact = true;
        source = this->exactFlag ? "--exact" : "--exact from DTFE_EXACT_AVERAGE=1";
        // a redshift cone grid samples its '_a' fields (method 2, updateEntries): a warning and the sampled average,
        // not a refusal -- unless the request is refused anyway (MY_SCALAR, another method, '--average-method 2',
        // velocityStd_a: checkGpuOptions and readAveragingOptions then say why)
        bool refused = otherMethod or a.velocity_std or ( methodGiven and this->method != 1 );
#ifdef MY_SCALAR
        refused = true;
#endif
        if ( cone and not refused )
        {
            MESSAGE::Warning warning( this->verboseLevel );
            warning << "'--exact' does not apply to a redshift cone grid: the '_a' fields are sampled there (volume averaging method 2)." << MESSAGE::EndWarning;
            exact = false;
            source = "a redshift cone grid has no exact form";
        }
    }
    else if ( pin == 0 ) source = "--sampled from DTFE_EXACT_AVERAGE=0";
    else if ( countGiven ) source = "a sampling count or method was given; --exact integrates the cells exactly";
    else if ( not otherMethod and noExact == nullptr ) { exact = true; source = "--exact, the default"; }
    else if ( not otherMethod ) source = noExact;
    // else another method: sampled, no header line
    this->exactAverage = exact;
#endif
    this->averageModeSource = source;
}


// The run mode (run_mode.h) into runMode, with the reason the run log's 'run mode' line gives (runModeSource). First given
// wins: '--run-mode MODE' (exact lower case), DTFE_RUN_MODE (any case; unset, empty or 'auto' is not given), else the
// default: performance inside a batch job (SLURM_JOB_ID or PBS_JOBID set and not empty), else balanced; any other word errors.
// PS-DTFE's '--gpu-share': 'auto' (default: the run mode's share, resolved with the plan), 'mixed' (auto's model in every run
// mode) or a fraction 0..1 kept in per mille; DTFE.cpp psResolveGpuShare decides where it applies. applyRunMode sets the levers.
void User_options::resolveRunMode(po::variables_map &vm)
{
    auto const modeOf = [](std::string const &w, RunMode &m)
    {
        if ( w == "gentle" ) m = RunMode::Gentle;
        else if ( w == "balanced" ) m = RunMode::Balanced;
        else if ( w == "performance" ) m = RunMode::Performance;
        else return false;
        return true;
    };
    if ( vm.count("run-mode") )
    {
        std::string const v = vm["run-mode"].as<std::string>();
        if ( not modeOf( v, this->runMode ) )
            throwError( "Unknown value '", v, "' for '--run-mode'; use 'gentle', 'balanced' (the default) or 'performance'." );
        this->runModeSource = "--run-mode " + v;
    }
    else
    {
        std::string const raw = Env::text( "DTFE_RUN_MODE" );
        std::string const lo = Env::lower( raw.c_str() );
        if ( not lo.empty() and lo != "auto" )
        {
            if ( not modeOf( lo, this->runMode ) )
                throwError( "Unknown value '", raw, "' for DTFE_RUN_MODE; use gentle, balanced or performance (in any case), or leave it unset (or 'auto') for the default." );
            this->runModeSource = "--run-mode " + lo + " from DTFE_RUN_MODE=" + raw;
        }
        else if ( Env::isSet( "SLURM_JOB_ID" ) )
        {
            this->runMode = RunMode::Performance;
            this->runModeSource = "--run-mode performance, the default inside a batch job (SLURM_JOB_ID)";
        }
        else if ( Env::isSet( "PBS_JOBID" ) )
        {
            this->runMode = RunMode::Performance;
            this->runModeSource = "--run-mode performance, the default inside a batch job (PBS_JOBID)";
        }
        else
        {
            this->runMode = RunMode::Balanced;
            this->runModeSource = "--run-mode balanced, the default";
        }
    }
#ifdef PHASE_SPACE
    if ( vm.count("gpu-share") )
    {
        std::string const v = vm["gpu-share"].as<std::string>();
        this->psGpuShareText = v;
        if ( v == "auto" )
            this->psGpuShareSpec = -1;
        else if ( v == "mixed" )
            this->psGpuShareSpec = -2;
        else
        {
            // a fraction, the whole text a number from 0 to 1
            double f = -1.;
            size_t used = 0;
            try { f = std::stod( v, &used ); }
            catch (exception const &) { used = 0; }
            if ( used == 0 or used != v.size() or not std::isfinite( f ) or f < 0. or f > 1. )
                throwError( "Unknown value '", v, "' for '--gpu-share'; use a fraction from 0 to 1 (0.6: 60% of the partitions deposit on the GPU), 'auto' (the run mode's share) or 'mixed' (the GPU and the free cores in every run mode)." );
            this->psGpuShareSpec = std::lround( f * 1000. );
        }
    }
#endif
}


#ifdef PHASE_SPACE
// PS-DTFE's default velocity-gradient route: true = the face gradient wherever the run
// qualifies (faceGradientBlocker), as in standard DTFE; false = the stream average of each stream's own gradient unless
// '--face-gradient' or DTFE_FACE_GRADIENT=1 asks for the face route. The drivers follow it (an empty FACE_GRADIENT
// passes neither flag); the launcher spells the route out on every job (python/gui/runspec.py face_gradient_env).
static constexpr bool kPsFaceGradientDefault = true;
#endif


// Why the face gradient (the velocity gradient from exact face integrals) is not this run's default route, or "" where it is:
// it is 3D, on the DTFE interpolant, not in the MY_SCALAR build; PS-DTFE's face sums are integrated beside the exact
// deposit, give the gradient of the VOLUME-weighted mean velocity and need every tetrahedron's faces (no halo release), and
// are its default only while kPsFaceGradientDefault says so. checkGpuOptions asks only when neither flag nor
// DTFE_FACE_GRADIENT decides; the run then keeps the deposit's own gradient and the header's line gives this reason.
std::string User_options::faceGradientBlocker(po::variables_map const &vm) const
{
    std::string why;
    bool const otherMethod = vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph")
                             or vm.count("voronoi");
    if ( NO_DIM != 3 )
        why = "the face gradient is 3D only";
    else if ( otherMethod )
        why = "the face gradient needs the DTFE interpolation method";
#ifdef MY_SCALAR
    if ( why.empty() )
        why = "the MY_SCALAR build has no face gradient";
#endif
#ifdef PHASE_SPACE
    if ( why.empty() )
    {
        if ( not kPsFaceGradientDefault )
            why = "the face gradient is not PS-DTFE's default: '--face-gradient' asks for it";
        else if ( not this->psExactDeposit )
            why = "the face gradient needs the exact deposit; this run deposits sampled ("
                  + ( this->averageModeSource.empty() ? std::string( "no '_a' field" ) : this->averageModeSource )
                  + ")";
        else if ( this->psHaloRelease > Real(0.) )
            why = "the face gradient cannot run with '--ps-halo-release' (a released tetrahedron has no faces)";
    }
#endif
    return why;
}


// Whether this run's grid work has a GPU path, as far as the options tell: a GPU build; a (PS-)DTFE field on a regular
// '--grid' (no '--redshift-cone', no '--serve'); at most one scalar component; averaging method 1; in PS-DTFE no
// '--ps-caustic-cusps'. A pass can still take the CPU (the 2^32-cell limit: avg1GpuUsable, psGpuEligible), with a warning
// and a counted fall-back. Keep in step with those guards.
bool User_options::gpuPathRuns(po::variables_map const &vm) const
{
#ifdef GPU_PATH_BUILT
    Field a = this->aField;   // the '_a' fields this build computes (as resolveAverageMode)
#ifdef PHASE_SPACE
    Field u = this->uField;   // PS-DTFE's unaveraged fields: the same deposit makes them, on the same device
#else
    Field u;                  // (standard DTFE's GPU path averages: its unaveraged fields are the CPU's)
#endif
#ifndef VELOCITY
    a.deselectVelocity();
    u.deselectVelocity();
#endif
#ifndef SCALAR
    a.deselectScalar();
    u.deselectScalar();
#endif
    bool const otherMethod = vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph")
                             or vm.count("voronoi");
    if ( not ( a.selected() or u.selected() ) or otherMethod or this->gridSize.empty() or vm.count("redshift-cone")
         or this->psServe )
        return false;
#ifdef SCALAR
    if ( ( a.scalar or a.scalar_gradient ) and noScalarComp != 1 )
        return false;
#ifdef PHASE_SPACE
    // (PS-DTFE's unaveraged pass deposits on the same device, and psGpuEligible tests its fields too)
    if ( ( this->uField.scalar or this->uField.scalar_gradient ) and noScalarComp != 1 )
        return false;
#endif
#endif
#ifdef PHASE_SPACE
    return not this->psCausticCusps;
#else
    return this->method == 1;
#endif
#else
    (void)vm;
    return false;
#endif
}


// Resolves the device into the Metal member, the triangulation's insert and the gradient route, and checks the
// '--exact' combinations (exactAverage is resolved already: only an explicit '--exact' or DTFE_EXACT_AVERAGE=1 reaches the
// refusals).
void User_options::checkGpuOptions(po::variables_map &vm)
{
    // the triangulation's insert: the default rule (psParallelTriangulation: the parallel insert above the size threshold),
    // '--parallel-triangulation' forces it, '--sequential-triangulation' turns it off; both at once is an error
    if ( this->sequentialTriangulation and this->parallelTriangulationForced )
        throwError( "'--parallel-triangulation' and '--sequential-triangulation' exclude each other." );
    if ( this->sequentialTriangulation )
        this->psParallelTriangulation = false;
    // THE DEVICE of the grid work (useMetal / psUseMetal), once, after the average and the run mode. First given wins:
    // '--gpu' asks for the GPU; DTFE_GPU (Env::tristate); else the program picks the GPU where the grid work has a GPU
    // path (gpuPathRuns) and, in standard DTFE, the average is exact. An asked GPU gets the warnings and the fall-back, a
    // picked one none. The launcher reads a deviceLine starting with 'CPU' as the CPU, else one holding ' GPU' as the GPU.
    {
        int const pin = Env::tristate( "DTFE_GPU" );   // 1 = as '--gpu', 0 = the CPU, -1 = unset or auto
        std::string const pinSource = "DTFE_GPU=" + Env::text( "DTFE_GPU" );
        bool const canRun = this->gpuPathRuns( vm );
        bool gpu = false;
        this->deviceSource = "";
        this->gpuPicked = false;
        if ( this->gpuAlias )
        {
            gpu = true;
            this->deviceSource = "--gpu";
        }
        else if ( pin == 1 )
        {
            gpu = true;
            this->deviceSource = pinSource;
        }
        else if ( pin == 0 )
            this->deviceSource = pinSource;
        else
        {
#ifdef PHASE_SPACE
            gpu = canRun;
#else
            gpu = canRun and this->exactAverage;
#endif
            this->gpuPicked = gpu;
        }
#ifdef PHASE_SPACE
        this->psUseMetal = gpu;
#else
        this->useMetal = gpu;
#endif
#ifdef GPU_PATH_BUILT
        if ( gpu and not this->gpuPicked )
            this->deviceLine = std::string( GPU_BACKEND_NAME " GPU (" ) + this->deviceSource + ")";
        else if ( gpu )
#ifdef PHASE_SPACE
            this->deviceLine = GPU_BACKEND_NAME " GPU (the default; DTFE_GPU=0 keeps the run on the CPU)";
#else
            this->deviceLine = GPU_BACKEND_NAME " GPU (the default for the exact average; DTFE_GPU=0 keeps the run on the CPU)";
#endif
        else if ( not this->deviceSource.empty() )
            this->deviceLine = "CPU (" + this->deviceSource + ")";
#ifndef PHASE_SPACE
        else if ( canRun )
            this->deviceLine = "CPU (the default for the sampled average; --gpu runs it on the GPU)";
#endif
        else
#ifdef PHASE_SPACE
            this->deviceLine = "CPU (the default: this deposit has no GPU path)";
#else
            this->deviceLine = "CPU (the default: this average has no GPU path)";
#endif
#else
        // a build without a GPU path: an asked GPU falls back with the deposits' warning, as '--gpu' does
        if ( gpu )
            this->deviceLine = "CPU (" + this->deviceSource + " given but this binary was built without GPU support; falling back)";
        else if ( not this->deviceSource.empty() )
            this->deviceLine = "CPU (" + this->deviceSource + ")";
        else
            this->deviceLine = "CPU (this binary was built without GPU support)";
#endif
    }

    // The velocity gradient's route (both binaries): the face gradient is the DEFAULT where faceGradientBlocker finds nothing
    // against it, unless '--per-tet-gradient' or DTFE_FACE_GRADIENT=0 keeps the deposit's per-tetrahedron gradient (PS-DTFE:
    // the stream average of each stream's own); '--face-gradient' or DTFE_FACE_GRADIENT=1 ask for the face route explicitly
    // and are checked (PS-DTFE: checkPsDepositOptions). Where the default does not apply the run keeps the deposit's
    // gradient; faceGradientSource says why.
    if ( this->faceGradient and this->perTetGradient )
        throwError( "'--face-gradient' and '--per-tet-gradient' exclude each other: the velocity gradient comes from the exact face integrals or from the tetrahedra's own gradients." );
    {
        int const faceEnv = Env::tristate( "DTFE_FACE_GRADIENT" );   // 1 forces the face route, 0 the deposit's, else unset
        bool const faceGiven = vm.count("face-gradient") and not vm["face-gradient"].defaulted();
        std::string const blocker = this->faceGradientBlocker( vm );   // "" where the face route is the default
        if ( this->perTetGradient )                 // the flag: the deposit's route
        {
            this->faceGradient = false;
            this->faceGradientSource = "--per-tet-gradient";
        }
        else if ( faceGiven )                       // the flag: the face route
        {
            this->faceGradient = true;
            this->faceGradientSource = "--face-gradient";
        }
        else if ( faceEnv == 0 )                    // the environment: the deposit's route
        {
            this->perTetGradient = true;
            this->faceGradientSource = "DTFE_FACE_GRADIENT=0";
        }
        else if ( faceEnv == 1 )                    // the environment: the face route, asked for (checked as the flag)
        {
            this->faceGradient = true;
            this->faceGradientSource = "DTFE_FACE_GRADIENT=1";
        }
        else if ( blocker.empty() )                 // the default where the face route applies
        {
            this->faceGradient = true;
            this->faceGradientSource = "the default where it applies";
        }
        else                                        // the default elsewhere: the deposit's route, and why
            this->faceGradientSource = blocker;
    }
#ifndef PHASE_SPACE
    // --face-gradient: the face sweep is 3D (a 2D build would integrate along grid lines), on the DTFE interpolant
    if ( this->faceGradient )
    {
        if ( NO_DIM != 3 )
            throwError( "'--face-gradient' is 3D only for now; drop it in the 2D build." );
        if ( vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph") or vm.count("voronoi") )
            throwError( "'--face-gradient' needs the DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
#ifdef MY_SCALAR
        throwError( "'--face-gradient' is not built with MY_SCALAR (the user's scalar keeps one thread and its own path)." );
#endif
    }
    // exact '_a' averaging (--exact) options, standard binary only: the resolved default is never refused here
    // (resolveAverageMode makes it exact only where none of these applies)
    if ( this->exactAverage )
    {
#ifdef MY_SCALAR
        throwError( "'--exact' cannot integrate the user-defined MY_SCALAR function (it is not linear inside a tetrahedron); rebuild without MY_SCALAR or use the sampled averaging." );
#endif
        // .defaulted(): '--average-method' carries a default_value, so vm.count() alone is always 1
        if ( vm.count("average-method") and not vm["average-method"].defaulted() and this->method != 1 )
            throwError( "'--exact' replaces the sampled averaging and runs on the method-1 scatter topology; drop '--average-method' or pass '--average-method 1'." );
        if ( vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph") or vm.count("voronoi") )
            throwError( "'--exact' needs the DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }
#endif
}


// Checks the '--scratch-dir' and '--tessellation-cache' directories.
void User_options::checkDirectoryOptions()
{
    // --scratch-dir (both binaries): out-of-core backing for the full-grid accumulators.
    // Validate the path HERE, before any big allocation could route through it.
    if ( not this->scratchDir.empty() )
    {
        // iCloud/cloud-synced volumes are refused outright: the scratch files can hold hundreds of GB and a sync daemon
        // that raced the unlink would try to upload them (a subdirectory beside outputs in iCloud Drive is exactly wrong).
        if ( this->scratchDir.find("Mobile Documents") != std::string::npos
             or this->scratchDir.find("com~apple~CloudDocs") != std::string::npos )
            throwError( "'--scratch-dir ", this->scratchDir, "' points into iCloud Drive. The scratch files hold up to hundreds of GB of grid data and must live on a LOCAL, non-synced volume -- use e.g. '/private/tmp/dtfe-scratch' (create it first)." );
        struct stat st;
        if ( ::stat(this->scratchDir.c_str(), &st) != 0 or not S_ISDIR(st.st_mode) )
            throwError( "'--scratch-dir ", this->scratchDir, "' is not an existing directory. Create it first (it must be on a local, non-synced volume with enough free space for the full-resolution grids)." );
    }

    // --tessellation-cache (both binaries): reusable on-disk tessellation. Same volume rules as
    // --scratch-dir -- the files run to tens of GB, so an iCloud path would sync-upload them.
    if ( not this->tessellationCacheDir.empty() )
    {
        if ( this->tessellationCacheDir.find("Mobile Documents") != std::string::npos
             or this->tessellationCacheDir.find("com~apple~CloudDocs") != std::string::npos )
            throwError( "'--tessellation-cache ", this->tessellationCacheDir, "' points into iCloud Drive. The cached tessellation runs to tens of GB and must live on a LOCAL, non-synced volume -- use e.g. '/private/tmp/dtfe-tess' (create it first)." );
        struct stat st;
        if ( ::stat(this->tessellationCacheDir.c_str(), &st) != 0 or not S_ISDIR(st.st_mode) )
            throwError( "'--tessellation-cache ", this->tessellationCacheDir, "' is not an existing directory. Create it first (it must be on a local, non-synced volume with room for ~0.5 KB per tessellation vertex)." );
        if ( this->poisson != 0 )
            throwError( "'--tessellation-cache' cannot be used with '--poisson': the particle positions are generated randomly each run, so a cached tessellation would not correspond to them." );
    }
}


// Checks the point-evaluation options (--sample-points, --serve, companions) and PS-DTFE's --partition-index/--region refusal.
void User_options::checkPointEvalOptions(po::variables_map &vm)
{
    // point evaluation (--sample-points) options, both binaries
#ifdef PHASE_SPACE
    // '--partition-index' and '--region' crop the PARTICLES to an Eulerian box (findParticlesInBox reads p.pos), sound for
    // standard DTFE (Eulerian tessellation). PS-DTFE triangulates in LAGRANGIAN space, where the tessellation of an
    // Eulerian-selected subset is NOT the subset of the tessellation, so the result would be silently wrong. The right PS
    // path is DTFE.cpp's Lagrangian partition loop (gated on 'partNo<0'): its partitions are SUMMED into the shared full grid.
    if ( vm.count("partition-index") )
        throwError( "'--partition-index' is not supported by PS-DTFE. It selects this partition's particles by their EULERIAN position, but PS-DTFE builds the tessellation in LAGRANGIAN space, so the result is silently wrong (measured: 626% error, 3.5% of cells uncovered) rather than merely partitioned. Use '--partition' WITHOUT '--partition-index': PS-DTFE partitions in Lagrangian space internally and sums the partitions itself." );
    if ( this->regionOn or this->regionMpcOn )
        throwError( "'--region'/'--region-mpc' are not supported by PS-DTFE: they crop the particles to an EULERIAN box, but the tessellation is built in LAGRANGIAN space, so the interpolated field is silently wrong (measured: 99.7% of cells). Run the full box; to bound memory use '--partition' (Lagrangian, correct) instead." );
#else
    // vm.count, not partitionOn: the partition options are folded into members further down. '--partition' is fine
    // (each partition answers the points of the tetrahedra it owns, DTFE.cpp); '--partition-index' computes
    // one partition only, whose points would silently be a fraction of the file's
    if ( not this->psSamplePointsFile.empty() and vm.count("partition-index") )
        throwError( "In the standard DTFE binary '--sample-points' cannot be combined with '--partition-index': one partition answers only the points in its own sub-box. Use '--partition' without '--partition-index' (every partition then answers its share of the points)." );
#endif
    if ( this->psPerStreamIds or this->psPtsLagrangian )
        this->psPerStream = true;   // the ids / Lagrangian-position files index into the per-stream ragged layout
    if ( this->psServe )
    {
        if ( not this->psSamplePointsFile.empty() )
            throwError( "'--serve' answers point requests interactively; it cannot be combined with '--sample-points' (send those points to the server instead)." );
        if ( vm.count("partition-index") )
            throwError( "'--serve' answers for the whole box; it cannot be combined with '--partition-index' (use '--partition' for a composite server of partition tessellations)." );
    }
    if ( this->serveResident < 0 )
        throwError( "'--serve-resident' must be 0 (automatic) or a positive number of partitions." );
    if ( this->psServeProgress and not this->psServe )
        throwError( "'--serve-progress' only applies to '--serve'." );
    if ( not vm["serve-resident"].defaulted() and not this->psServe )
        throwError( "'--serve-resident' only applies to '--serve'." );
    if ( not this->scalarDataset.empty() )
    {
        if ( this->scalarDataset.find('/') != std::string::npos )
            throwError( "'--scalar-dataset ", this->scalarDataset, "' must be a dataset NAME inside each '/PartTypeN' group (e.g. 'Potential'), not a path." );
        if ( this->readParticleData.size() > 3 )
            this->readParticleData[3] = 1;   // allocate the scalar block (the HDF5 reader fills it from the dataset)
    }
    if ( this->psPtsScalar and this->scalarDataset.empty()
         and not ( this->readParticleData.size() > 3 and this->readParticleData[3] ) )
        throwError( "'--pts-scalar' interpolates the particles' scalar value, but none is read: name the per-particle dataset with '--scalar-dataset NAME' (Gadget HDF5 input)." );
    if ( not this->psSamplePointsFile.empty() or this->psServe )
    {
        char const *mode = this->psServe ? "'--serve'" : "'--sample-points'";
        if ( vm.count("interlace") )
            throwError( mode, " cannot be combined with '--interlace' (interlacing runs the interpolation twice, which would double-count the point streams)." );
        if ( vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph") or vm.count("voronoi") )
            throwError( mode, " needs the (PS-)DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }
    else if ( this->psPtsLagrangian )
        throwError( "'--ps-per-stream-lagrangian' requires '--sample-points' or '--serve'." );
    else if ( this->psPerStreamIds )
        throwError( "'--ps-per-stream-ids' requires '--sample-points' or '--serve'." );
    else if ( this->psPerStream )
        throwError( "'--ps-per-stream' requires '--sample-points' or '--serve'." );
    else if ( this->psPtsDenGrad )
        throwError( "'--pts-den-grad' requires '--sample-points' or '--serve'." );
    else if ( this->psPtsVelGrad )
        throwError( "'--pts-vel-grad' requires '--sample-points' or '--serve'." );
    else if ( this->psPtsScalar )
        throwError( "'--pts-scalar' requires '--sample-points' or '--serve'." );
}


#ifdef PHASE_SPACE
// Checks PS-DTFE's caustic, alpha-shape, Lagrangian-padding, halo-release, exact-deposit and face-gradient options.
void User_options::checkPsDepositOptions(po::variables_map &vm)
{
    // caustic flagging (--ps-caustics) options
    if ( this->psCaustics )
    {
        // 2D (PS-DTFE-2d): the parity bits, the collapse multiplicity k = 0..2 and, with
        // --ps-caustic-cusps, the cusp bit (no swallowtail bit: see ps_caustic_class.h)
        if ( vm.count("interlace") )
            throwError( "'--ps-caustics' cannot be combined with '--interlace' (interlacing runs the interpolation twice, which would corrupt the per-cell orientation bits)." );
        if ( vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph") or vm.count("voronoi") )
            throwError( "'--ps-caustics' needs the PS-DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }
    else if ( this->psCausticCusps )
        throwError( "'--ps-caustic-cusps' refines the caustic mask, so it requires '--ps-caustics'." );

    if ( this->psAlphaShape < Real(0.) )
        throwError( "'--ps-alpha-shape' takes a number of mean particle spacings >= 0 (0 keeps the whole convex hull)." );
    // a negative padding would shrink each partition below its own ownership box (owned tetrahedra lost)
    if ( this->psLagPadding < Real(0.) )
        throwError( "'--ps-lagrangian-padding' takes a number of mean particle spacings >= 0 (0 = the Eulerian '--padding' rule)." );

    // halo-interior release (--ps-halo-release) options
    if ( vm.count("ps-halo-release") and this->psHaloRelease <= Real(0.) )
        throwError( "'--ps-halo-release' needs a positive threshold D (geometric stream density in rho/rho_bar units, e.g. '--ps-halo-release 300')." );

    // exact conservative deposit (--exact) options: resolved already, the default only with the PS-DTFE method, so
    // only an explicit '--exact' or DTFE_EXACT_AVERAGE=1 reaches the refusal
    if ( this->psExactDeposit )
    {
        // 2D (PS-DTFE-2d) clips triangles against the cells itself (ps_clip2d.h); 3D uses r3d
        if ( vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph") or vm.count("voronoi") )
            throwError( "'--exact' (the exact deposit) needs the PS-DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }

    // the mean-flow gradient from face integrals (--face-gradient): the exact deposit (on the
    // CPU, or on the GPU with the face sums from a CPU walk beside it), the volume-weighted mean (every PS-DTFE
    // run's; the quotient form (dS - <v> dN) / sum V_s is that mean's), 3D
    if ( this->faceGradient )
    {
        if ( NO_DIM != 3 )
            throwError( "'--face-gradient' is 3D only for now; drop it in the 2D build." );
        if ( not this->psExactDeposit )
            throwError( "'--face-gradient' integrates the faces in PS-DTFE's exact deposit: add '--exact' (or drop '--sampled')." );
        if ( this->psHaloRelease > Real(0.) )
            throwError( "'--face-gradient' cannot be combined with '--ps-halo-release': a released tetrahedron deposits as a point and has no faces." );
    }
}
#endif


// Reads the partition options, with PS-DTFE's merge mode first.
void User_options::readPartitionOptions(po::variables_map &vm)
{
#ifdef PHASE_SPACE
    // '--ps-merge MODE' (one value: boost refuses a second) sets its member: a GIVEN index order also keeps the memory
    // gate on under the speed split (DTFE.cpp). Any other value is refused
    if ( vm.count("ps-merge") )
    {
        std::string const mode = vm["ps-merge"].as<std::string>();
        if ( mode == "index-order" ) this->psOrderedMergeFlag = true;
        else if ( mode == "completion-order" ) this->psCompletionOrderMergeFlag = true;
        else throwError( "Unknown value '", mode, "' for '--ps-merge'; use 'index-order' (the default) or "
                         "'completion-order'." );
    }
    // '--resume' implies the ordered merge: resumed partitions add in index order, so a run killed under it resumes to
    // byte-identical grids. The FLAG too: it keeps the memory gate on (DTFE.cpp). It wins over a '--ps-merge
    // completion-order' silently: run_ps_dtfe.sh adds --resume on its own when a checkpoint exists, and a refusal there
    // would stop an unattended run.
    if ( this->psResume )
        this->psOrderedMergeFlag = true;
    // the (staged) index-order merge is the default; '--ps-merge completion-order' asks for completion order
    this->psOrderedMerge = this->psOrderedMergeFlag or not this->psCompletionOrderMergeFlag;   // the speed split sets it too
#endif
    if ( vm.count("max-concurrent") and not vm["max-concurrent"].defaulted() )
        this->maxConcurrentOn = true;   // user gave it (possibly 0 = uncapped); auto-tuner must not touch it
    if ( vm.count("partition") )
    {
        this->partitionOn = true;
        this->partitionGiven = true;   // survives the '1 1 1 = no actual partition' reset below
        int const temp = this->partition.size();
        // build a new vector rather than self-assign: assign(n, partition.at(0)) reads the fill
        // value through a reference into the vector being reallocated (UB -> garbage partition
        // counts -> the partition loop silently runs zero times and writes all-zero grids)
        if ( temp==1 ) this->partition = std::vector<size_t>( NO_DIM, this->partition.at(0) );
        else if ( temp!=NO_DIM ) throwError( "You can only insert 1 or ", NO_DIM, " values for the '--partition' option (e.g. '--partition 3' or '--partition 2 3 3')." );
        
        for (size_t i=0; i<this->partition.size(); ++i)
            lowerBoundCheck( this->partition[i], size_t(1), "the values for the option '--partition'" );
        
        size_t temp2 = 1;
        for (size_t i=0; i<this->partition.size(); ++i)
            temp2 *= this->partition[i];
        if ( vm.count("partition-index") ) intervalCheck( this->partNo, 0, int(temp2-1), "'--partition-index' program option" );
        
        if ( temp2==1 ) // no actual partition
            this->partitionOn = false;
    }
}


// Reads the padding options and turns the padding test off where it does not apply.
void User_options::readPaddingOptions(po::variables_map &vm)
{
    if ( vm.count("padding-mpc") )
        this->paddingMpcOn = true;
    if ( vm.count("padding-mpc") or vm.count("padding") )
    {
        this->paddingOn = true;
        size_t temp = 0;
        if (this->paddingMpcOn)
        {
            temp = vm["padding-mpc"].as< std::vector<Real> >().size();
            this->paddingLength.coords = vm["padding-mpc"].as< std::vector<Real> >();
        }
        else
        {
            temp = vm["padding"].as< std::vector<Real> >().size();
            if ( temp==1 )
            {
                paddingParticles = vm["padding"].as< std::vector<Real> >().at(0);
                lowerBoundCheck( paddingParticles, Real(0.), "'--padding' program option" );
                temp = 2*NO_DIM;
            }
            else
                paddingLength.coords = vm["padding"].as< std::vector<Real> >();
        }
        if ( temp!=2*NO_DIM )
            throwError( "You have to insert 1 or ", 2*NO_DIM, " values for the '--padding' or '--padding-mpc' option (e.g. '--padding 0.1 0.2 0.1 0.1 0.3 0.5')." );
    }
    if ( vm.count("no-padding-test") ) this->testPaddedBoundaries = false;
#ifndef PHASE_SPACE
    // Point evaluation (--sample-points / --serve) needs the PARTICLES' own tessellation. The padding test's dummy points are
    // inserted INTO it; on a non-periodic box they sit against the convex hull and change the vertex densities there
    // (differently for every composite partition). Periodic runs are unaffected while the padding is complete (the dummies
    // lie beyond the image copies).
    if ( this->psServe or not this->psSamplePointsFile.empty() )
        this->testPaddedBoundaries = false;
#endif
}


// Reads the averaging options: '--average-method', '--subsamples', '--mean-density' and '--seed'.
void User_options::readAveragingOptions(po::variables_map &vm)
{
    intervalCheck( this->method, 1, 2, "'--average-method' can have only 2 values (1 or 2) since there are implemented only 2 methods for field averaging inside the sampling cell (see '--help' for additional details)" );
#ifndef PHASE_SPACE
    // velocityStd_a: only the DTFE method's averaging method 2 computes it; every other path leaves its grid empty,
    // which the sub-domain merge then reads (a segfault). PS-DTFE's deposit computes it under every method.
    if ( this->aField.velocity_std )
    {
        if ( vm.count("ngp") or vm.count("cic") or vm.count("tsc") or vm.count("pcs") or vm.count("sph") or vm.count("voronoi") )
            throwError( "'velocityStd_a' (the velocity standard deviation inside the grid cell) needs the DTFE interpolation method with the averaging method 2 '--average-method 2'; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
        if ( this->exactAverage )   // resolved: only an explicit '--exact' or DTFE_EXACT_AVERAGE=1 (the default samples here)
            throwError( "'--exact' cannot compute 'velocityStd_a' (it integrates the mean velocity, not its spread inside the grid cell); drop '--exact' and use the averaging method 2 '--average-method 2' (Monte Carlo sampling inside the grid cell) to compute the velocity standard deviation." );
        if ( this->method != 2 )
            throwError( "You cannot use the averaging method ", this->method,
                        " (Monte Carlo sampling inside the Delaunay cell) to compute the velocity standard deviation inside the grid cell. Please use the averaging method 2 '--average-method 2' (Monte Carlo sampling inside the grid cell) to compute the velocity standard deviation." );
    }
#endif
    if ( this->useMetal and this->method!=1 )   // (an asked GPU only: the program picks it for method 1 alone)
    {
        MESSAGE::Warning warning( this->verboseLevel );
        warning << this->gpuRequest() << " accelerates only the method-1 ('--average-method 1') volume-averaged interpolation; method " << this->method << " will run on the CPU.\n" << MESSAGE::EndWarning;
    }
#ifndef PHASE_SPACE
    if ( vm.count("subsamples") )
    {
        // k per axis, k^NO_DIM points per cell (method 2: k^NO_DIM random points)
        int const k = vm["subsamples"].as<int>();
        lowerBoundCheck( k, 1, "value of the '--subsamples' option (sampling points per axis)" );
        this->noPoints = 1;
        for (int d = 0; d < NO_DIM; ++d) this->noPoints *= k;
        this->noPointsOn = true;
    }
    else
#endif
    if ( this->method==2 )
        this->noPoints = 20;  // default for method 2
    if ( vm.count("mean-density") )
        lowerBoundCheck( averageDensity, Real(0.), "the value of '--mean-density'" );
    if ( vm.count("seed") )
        lowerBoundCheck( this->randomSeed, size_t(0), "value supplied with program option '--seed'" );
    else
    {
        std::srand( (unsigned)time(0) );
        this->randomSeed = std::rand();
    }
}


// Reads and checks '--redshift-cone' and '--cone-origin'.
void User_options::readRedshiftConeOptions(po::variables_map &vm)
{
    if ( vm.count("redshift-cone") )
    {
        redshiftConeOn = true;
        size_t temp = vm["redshift-cone"].as< std::vector<Real> >().size();
        if ( temp!=2*NO_DIM ) throwError( "The '--redshift-cone' option must be followed by ", 2*NO_DIM, " values which give the extension ", (NO_DIM==2 ? "(r_min, r_max, psi_min, psi_max)" : "(r_min, r_max, theta_min, theta_max, psi_min, psi_max)" ), " for the spherical coordinates region used to interpolate to grid."  );
        redshiftCone.coords = vm["redshift-cone"].as< std::vector<Real> >();
        
        for (int i=0; i<NO_DIM; ++i)
            if ( redshiftCone[2*i]>=redshiftCone[2*i+1] ) throwError( "When inserting the lower and upper values for option '--redshift-cone'. A lower value is higher than an upper value. If error was due to an angular interval value, increase the upper bound by 360 degrees." );
        if (NO_DIM==3)
        {
            intervalCheck( redshiftCone[2], Real(0.), Real(180.), "3rd value of option '--redshift-cone'" );
            intervalCheck( redshiftCone[3], Real(0.), Real(180.), "4th value of option '--redshift-cone'" );
        }
        Real tempRes = redshiftCone[2*(NO_DIM-1)+1] - redshiftCone[2*(NO_DIM-1)]; // psi_max - psi_min
        if ( tempRes>=Real(360.) ) throwError( "The 'psi' angle interval can stretch at most 360 degrees." );
    }
    if ( vm.count("cone-origin") )
        if ( originPosition.size()!=NO_DIM ) throwError( "The option '--cone-origin' must be followed by ", NO_DIM, " values." );
}


// Reads the additional options: NGP/CIC/TSC/PCS/SPH/Voronoi, '--interlace', units, verbosity and '--redshiftSpace'.
void User_options::readAdditionalOptions(po::variables_map &vm)
{
    if ( vm.count("ngp") )
    {
        this->NGP = true;
        this->DTFE = false;
    }
    if ( vm.count("cic") )
    {
        this->CIC = true;
        this->DTFE = false;
    }
    if ( vm.count("tsc") )
    {
        this->TSC = true;
        this->DTFE = false;
    }
    if ( vm.count("pcs") )
    {
        this->PCS = true;
        this->DTFE = false;
    }
    if ( vm.count("sph") )
    {
        this->SPH = true;
        this->DTFE = false;
    }
    if ( vm.count("voronoi") )
    {
        this->Voronoi = true;
        // Voronoi uses the DTFE pipeline (needs triangulation) so keep DTFE=true, density-only
        // output. The NGP assignment fills whichever density field was selected; if the user
        // selected none, default to the unaveraged one. Every OTHER field must be rejected: the
        // Voronoi path returns before the DTFE interpolation, so a selected-but-unfilled field
        // would be written as a 0-byte file (single domain) or fail the '--partition' size check.
        if ( not this->uField.density and not this->aField.density )
            this->uField.density = true;
        if ( this->uField.selectedVelocity() or this->uField.selectedScalar()
             or this->aField.selectedVelocity() or this->aField.selectedScalar() )
            throwError( "'--voronoi' outputs the density only (it NGP-assigns the 1/Voronoi-volume vertex densities); drop the other fields from '--field'." );
    }
    if ( vm.count("interlace") )
        this->interlace = true;
    // '--interlace' runs the partition loop twice, the second time on a half-cell-shifted grid (main.cpp). Both
    // passes would checkpoint into ONE folder under ONE settings key (the shift is not on the command line), so a
    // resume would load one pass's grids as the other's: an interlaced run is not checkpointed, and
    // '--resume' refuses it. Both binaries (main.cpp runs standard DTFE twice the same way).
    if ( this->interlace )
    {
        if ( this->psResume )
            throwError( "'--resume' cannot be combined with '--interlace': an interlaced run (two passes, the second on "
                        "a half-cell-shifted grid) writes no checkpoint. Run it without '--resume'." );
        this->psNoCheckpoint = true;
    }
    if ( vm.count("mpc-unit") )
        lowerBoundCheck( MpcValue, Real(0.), "value of option '--mpc-unit'" );
    if ( vm.count("scalars-extensive") )
        this->extensive = true;
#ifndef PHASE_SPACE
    if ( vm.count("approx-phase-space-density") )
    {
        this->approxPSD = true;
        // auto-enable scalar field output
        if ( not this->uField.scalar )
            this->uField.scalar = true;
    }
#endif
    if ( vm.count("verbose") )
        intervalCheck( verboseLevel, 0, 3, "value of option '--verbose'" );
    if ( vm.count("particle-fraction") )
        intervalCheck( randomSample, Real(0.), Real(1.), "value of option '--particle-fraction'" );
    if ( vm.count("poisson") )
        lowerBoundCheck( poisson, size_t(1), "value of option '--poisson'" );
#ifdef REDSHIFT_SPACE
    if ( vm.count("redshiftSpace") )
    {
        // the transform itself is not implemented -- nothing under src/ reads these two members except the
        // tessellation-cache / globals-record keys -- so accepting the option would make it a SILENT no-op.
        // Refuse loudly until the shift is implemented (x += (d.v) d / H with H = 100 h km/s/Mpc, Gadget velocities
        // times sqrt(a), periodic wrap, applied after the read and before the triangulation).
        throwError( "The '--redshiftSpace' transform is not implemented in this version: the positions would not be shifted. Remove the option, or shift the positions before running." );
    }
    else
#endif
    {
        transformToRedshiftSpaceOn = false;
        transformToRedshiftSpace.assign( NO_DIM, Real(0.) );
    }
}


// Drops what this build compiled out and fits the field choice to the NGP/CIC/TSC/PCS/SPH methods.
void User_options::applyMethodFieldRules()
{
    // disable options that were turned off at compile time
#ifndef TEST_PADDING
    this->testPaddedBoundaries = false;
#endif
#ifndef VELOCITY
    this->uField.deselectVelocity();
    this->aField.deselectVelocity();
#endif
#ifndef SCALAR
    this->uField.deselectScalar();
    this->aField.deselectScalar();
#endif
    
    // special settings for the NGP/CIC/TSC/PCS/SPH methods
    if ( this->NGP or this->CIC or this->TSC or this->PCS or this->SPH )
    {
        // these methods produce only cell-averaged fields, so promote each unaveraged request to its '_a' form
        if ( this->uField.density ) this->aField.density = true;
        if ( this->uField.velocity ) this->aField.velocity = true;
        if ( this->uField.velocity_gradient ) this->aField.velocity_gradient = true;
        if ( this->uField.velocity_divergence ) this->aField.velocity_divergence = true;
        if ( this->uField.velocity_shear ) this->aField.velocity_shear = true;
        if ( this->uField.velocity_vorticity ) this->aField.velocity_vorticity = true;
        if ( this->uField.scalar ) this->aField.scalar = true;
        if ( this->uField.scalar_gradient ) this->aField.scalar_gradient = true;
        this->uField = Field();
        
        // these methods cannot compute the velocity gradient
        if ( this->aField.velocity_gradient or this->aField.selectedVelocityDerivatives() )
        {
            MESSAGE::Warning warning( verboseLevel );
            warning << "The NGP, CIC, TSC, PCS or SPH grid interpolation methods do not have implemented a method for computing the velocity gradient. The velocity gradient will not be computed!" << MESSAGE::EndWarning;
            this->aField.velocity_gradient = false;
            this->aField.deselectVelocityDerivatives(); // also turns off divergence, shear, vorticity
        }
        if ( this->aField.scalar_gradient )
        {
            MESSAGE::Warning warning( verboseLevel );
            warning << "The NGP, CIC, TSC, PCS or SPH grid interpolation methods do not have implemented a method for computing the velocity gradient. The velocity gradient will not be computed!" << MESSAGE::EndWarning;
            this->aField.scalar_gradient = false;
        }
        if ( (this->NGP or this->CIC or this->TSC or this->PCS) and this->aField.scalar )
        {
            MESSAGE::Warning warning( verboseLevel );
            warning << "The CIC and TSC grid interpolation method does not have implemented a method for computing the scalar fields on the grid. The scalar field cannot be computed!" << MESSAGE::EndWarning;
            this->aField.scalar = false;
        }
    }
}



// The run mode's levers (run_mode.h), once the options are read and before any thread exists. Thread budget: from N0, the
// OpenMP default at start-up, by the mode's formula (runModeCoreBudget), unless an explicit lever beats it: a non-empty
// OMP_NUM_THREADS keeps its count, a '--max-concurrent' above the mode's threads raises them. It goes into the process-wide
// state the thread counts read and, when lowered, into this thread's OpenMP default (dtfeAdoptCoreBudget for later threads).

// GPU buffer target / PS exact item cap: performance 0.25 s / 4M items, gentle and balanced 0.05 s / 500k. Gentle on Linux
// also lowers the priority to nice 10.
void User_options::applyRunMode()
{
#ifdef OPEN_MP
    int const N0 = std::max( 1, omp_get_max_threads() );
#else
    int const N0 = std::max( 1, int( std::thread::hardware_concurrency() ) );
#endif
    int perf = 0;   // the performance cores (triangulation.cpp parallelInsertCores reads them the same way)
#ifdef __APPLE__
    size_t sz = sizeof(perf);
    if ( sysctlbyname( "hw.perflevel0.logicalcpu", &perf, &sz, nullptr, 0 ) != 0 or perf < 0 )
        perf = 0;
#endif
    int budget = N0;
    this->coreBudgetNote = "";
    this->coreBudgetOwn = N0;
    std::string const ompThreads = Env::text( "OMP_NUM_THREADS" );
    if ( not ompThreads.empty() )
        this->coreBudgetNote = "kept: OMP_NUM_THREADS=" + ompThreads + " was given";
    else
    {
        budget = runModeCoreBudget( this->runMode, N0, perf );
        this->coreBudgetOwn = budget;   // (the share of '--gpu-share mixed' follows it, never the raise below)
        if ( this->maxConcurrentOn and this->maxConcurrent > budget )
        {
            budget = this->maxConcurrent;
            this->coreBudgetNote = "raised: --max-concurrent " + std::to_string( this->maxConcurrent ) + " was given";
        }
    }
    bool const performance = this->runMode == RunMode::Performance;
    RunModeState &state = runModeState();
    state.mode = this->runMode;
    state.ompDefault = N0;
    state.coreBudget = budget;
    state.budgetLowered = budget < N0;
    state.gpuTargetSec = performance ? 0.25 : 0.05;
    state.psItemCap = performance ? size_t(4000000) : size_t(500000);
    this->coreBudget = budget;
#ifdef OPEN_MP
    if ( state.budgetLowered )
        omp_set_num_threads( budget );
#endif
    this->priorityNote = "";
#ifdef __linux__
    // gentle: nice 10, never a lower nice than the process has (getpriority itself failing, which it does not for the
    // process itself, leaves the priority as it is, unsaid)
    if ( this->runMode == RunMode::Gentle )
    {
        errno = 0;
        int const cur = getpriority( PRIO_PROCESS, 0 );
        int const err = errno;
        if ( err == 0 and cur >= 10 )
            this->priorityNote = "nice " + std::to_string( cur ) + " kept";
        else if ( err == 0 )
        {
            if ( setpriority( PRIO_PROCESS, 0, 10 ) == 0 )
                this->priorityNote = "nice 10";
            else
            {
                int const why = errno;
                this->priorityNote = std::string( "priority unchanged (setpriority: " ) + std::strerror( why ) + ")";
            }
        }
    }
#endif
}


// Read the user-supplied options and check they satisfy the restrictions.
void User_options::readOptions(int argc, char *argv[], bool getFileNames, bool showOptions)
{
    po::options_description visibleOptions("Allowed options"), allOptions("All options");
    po::positional_options_description p;

    this->addOptions( allOptions, visibleOptions, p );
    
    po::variables_map vm;
    MESSAGE::Message showMessage(3);
    this->parseOptions( argc, argv, allOptions, p, vm, showMessage );
    this->checkFileNameOptions( argv, getFileNames, visibleOptions, vm, showMessage );
    checkConflictingOptions( vm );
    this->readMainOptions( vm );
    this->readFieldOptions( vm );
    this->readRegionOptions( vm );
    this->resolveAverageMode( vm );     // before every reader of exactAverage / psExactDeposit
    this->resolveRunMode( vm );
    this->checkGpuOptions( vm );
    this->checkDirectoryOptions();
    this->checkPointEvalOptions( vm );
#ifdef PHASE_SPACE
    this->checkPsDepositOptions( vm );
#endif
    this->readPartitionOptions( vm );
    this->readPaddingOptions( vm );
    this->readAveragingOptions( vm );
    this->readRedshiftConeOptions( vm );
    this->readAdditionalOptions( vm );
    
    // set values to userOptions->programOptions
    for (int i=0; i<argc; ++i)
        this->programOptions += string( argv[i] ) + " ";
    // the checkpoint's settings key, short letters in their long names: a run with '-g 64' resumes under '--grid 64'
    this->commandTokens = canonicalCommandTokens( argc, argv );
    
    this->applyMethodFieldRules();
    
    // check to see that there is at least one task selected
    if ( not (this->uField.selected() or this->uField.triangulation or this->aField.selected() or this->aField.triangulation) )
        throwError( "No valid field interpolation quantities were selected for the computation. The program cannot continue since it does not compute anything." );
    
    this->applyRunMode();   // the run mode's levers, before any thread exists (after readPartitionOptions: maxConcurrentOn)
    
    if (showOptions)
        this->printOptions();
}


// Sets 'boxCoordinates' to newBox and recomputes the per-axis 'fullBoxOffset' and 'fullBoxLength'.
void User_options::updateFullBox(Box &newBox)
{
    boxCoordinates = newBox;
    fullBoxOffset.clear();
    fullBoxOffset.reserve(NO_DIM);
    fullBoxLength.clear();
    fullBoxLength.reserve(NO_DIM);
    for (size_t i=0; i<NO_DIM; ++i)
    {
        fullBoxOffset.push_back( boxCoordinates[2*i] );
        fullBoxLength.push_back( boxCoordinates[2*i+1] - boxCoordinates[2*i] );
    }
}




// After input is read: resolves box/grid/region/padding, applies method overrides, and error-checks the options.
void User_options::updateEntries(size_t const noTotalParticles,
                                 bool userSampling)
{
    // check full box size and grid values
    if ( this->boxCoordinates.size()!=2*NO_DIM )
        throwError( "Failed a consistency check. The box encompassing the data should have ", 2*NO_DIM, " coordinates, but it has ", this->boxCoordinates.size(), " coordinates. Check again the values supplied as the coordinates of the full data box." );
    if ( this->boxCoordinates.volume()==0. )
        throwError( "Failed a consistency check. The box encompassing the data has 0. volume. Probably you forget to initialize the coordinates of the full particle data box." );
    
    if ( (this->gridSize.empty() and not userSampling) or this->gridSize.size()!=NO_DIM )
        throwError( "Failed a consistency check. The array storing the interpolation grid should have ", NO_DIM, " values, but it has ", this->gridSize.empty()?0:this->gridSize.size(), " values. Check again the values supplied as the size of the interpolation grid." );
    if ( not gridSize.empty() )
        for (size_t i=0; i<this->gridSize.size(); ++i)
            lowerBoundCheck( this->gridSize[i], size_t(1), "the values of option '--grid'" );
    
    
    // update paddedBox, fullBoxOffset, fullBoxLength
    paddedBox = boxCoordinates;
    this->updateFullBox( boxCoordinates );  // computes 'fullBoxOffset' and 'fullBoxLength'

    updatePadding( noTotalParticles );


    // set the 'region' variable. Entry i of the box is axis i/2 (xmin,xmax,ymin,...)
    if ( regionOn and not regionMpcOn )
    {
        for (size_t i=0; i<region.size(); ++i )
            region[i] = fullBoxOffset[i/2] + region[i] * fullBoxLength[i/2];
        regionMpcOn = true;
    }
    else if ( not regionOn )
        region = boxCoordinates;
    
    
    // update the user sampling points
    userDefinedSampling = userSampling;
    if ( userSampling )
    {
        redshiftConeOn = false;
        if ( method!=2 and aField.selected() )
        {
            bool const wasExact = exactAverage;
            method = 2;
            exactAverage = false;   // method 2 samples: the exact average (resolveAverageMode) has no form here
            if ( not noPointsOn ) noPoints = 20;
            MESSAGE::Warning warning( verboseLevel );
            warning << "When computing the volume average of the fields on a user defined grid only volume averaging method 2 is available. The program will use volume averaging method '2 - Monte Carlo sampling using random points inside the grid cells' using '"<< noPoints <<"' random samples in each user given grid cell." << MESSAGE::EndWarning;
            if ( wasExact )   // the header's EXACT line was printed before the sampling points were known
            {
                averageModeSource = "user-given sampling points have no exact form";
                MESSAGE::Message corrected( verboseLevel );
                corrected << "\t volume averaging       : SAMPLED (" << averageModeSource
                          << "; this replaces the header's EXACT line)\n" << MESSAGE::Flush;
            }
        }
    }
    else if ( redshiftConeOn )
    {
        if ( method!=2 and aField.selected() )
        {
            method = 2;
            exactAverage = false;   // (resolveAverageMode samples a cone already; a library caller may have set it)
            if ( not noPointsOn ) noPoints = 20;
            MESSAGE::Warning warning( verboseLevel );
            warning << "When computing the volume average of the fields on a redshift cone grid only volume averaging method 2 is available. The program will use volume averaging method '2 - Monte Carlo sampling using random points inside the grid cells' using '"<< noPoints <<"' random samples in each grid cell." << MESSAGE::EndWarning;
        }
    }
#ifndef PHASE_SPACE
    // a GPU the program picked (checkGpuOptions) runs method 1 alone: where user-given sampling points or a redshift
    // cone grid took method 2 above (a cone given on the command line never meets the pick: gpuPathRuns), the pick
    // takes the CPU without a warning, and the header's device line is replaced as its EXACT line is (an asked GPU
    // stays as '--gpu': method 2 never reads it)
    if ( this->gpuPicked and this->method != 1 )
    {
        this->useMetal = false;
        this->gpuPicked = false;
        this->deviceLine = "CPU (the default: this average has no GPU path)";
        MESSAGE::Message corrected( verboseLevel );
        corrected << "\t '_a' interpolation     : CPU (the default: this average has no GPU path; this replaces "
                  << "the header's GPU line)\n" << MESSAGE::Flush;
    }
#endif
    
    
    // check that the option "--partition" is disabled when dealing with redshift cone or used defined sampling coordinates
    if ( partitionOn and (userSampling or redshiftConeOn) )
    {
        MESSAGE::Warning warning( verboseLevel );
        warning << "The option '--partition' is not available when interpolating the fields to a redshift cone grid or to user defined sampling points. This '--partition' option will be disabled in the rest of the program. If the computation is too large for the available RAM we advise that you manually split the computation in manageable data portions." << MESSAGE::EndWarning;
        partitionOn = false;
        for (int i=0; i<NO_DIM; ++i)
            partition[i] = 1;
        partNo = -1;
    }
    
    
    // additional error checking
    if ( redshiftConeOn )
    {
        if (NO_DIM==3)
        {
            intervalCheck( redshiftCone[2], Real(0.), Real(180.), "3rd value of option '--redshift-cone'" );
            intervalCheck( redshiftCone[3], Real(0.), Real(180.), "4th value of option '--redshift-cone'" );
        }
        Real tempRes = redshiftCone[2*(NO_DIM-1)+1] - redshiftCone[2*(NO_DIM-1)]; // psi_max - psi_min
        if ( tempRes>=Real(360.) ) throwError( "The 'psi' angle interval can stretch at most 360 degrees." );
        if ( originPosition.empty() or originPosition.size()!=NO_DIM ) throwError( "The vector 'User_options::originPosition' must have ", NO_DIM, " entries." );
    }
    if ( randomSample>=Real(0.) )
        intervalCheck( randomSample, Real(0.), Real(1.), "value of '--particle-fraction'" );


    // check the values of the program constants defined in "define.h"
    if ( noVelComp!=NO_DIM )
        throwError( "The program constant 'noVelComp' must have the same value as the number of spatial dimensions. Please check the value of the constant in file 'define.h'." );
    if ( noGradComp != NO_DIM*NO_DIM )
        throwError( "The number of components of the velocity gradient must have the value (number of spatial dimensions)^2. Please check the value of the constant in file 'define.h'." );
    if ( noShearComp != (NO_DIM*(NO_DIM+1))/2-1 )
        throwError( "The number of components of the velocity shear must be 2 in 2D and 5 in 3D. Please check the value of the constant in file 'define.h'." );
    if ( noVortComp != ((NO_DIM-1)*NO_DIM)/2 )
        throwError( "The number of components of the velocity vorticity must be 1 in 2D and 3 in 3D. Please check the value of the constant in file 'define.h'." );
    if ( noScalarGradComp != noScalarComp*NO_DIM )
        throwError( "The number of components of the scalar values gradient must be the number of dimensions times the number of scalar components (which is ", noScalarComp*NO_DIM, " for the given parameters). Please check the value of the constant in file 'define.h'." );
    
    
    // these methods cannot compute gradients or scalar fields
    if ( NGP or CIC or TSC or PCS or SPH )
    {
        if ( aField.velocity_gradient or aField.selectedVelocityDerivatives() )
            throwError( "The NGP, CIC, TSC, PCS and SPH grid interpolation methods do not have implemented a method for computing the velocity gradient. The velocity gradient cannot be computed!");
        if ( aField.scalar_gradient )
            throwError( "The NGP, CIC, TSC, PCS and SPH grid interpolation methods do not have implemented a method for computing the scalar fields gradients. The scalar gradient cannot be computed!");
        if ( (CIC or TSC or PCS) and aField.scalar )
            throwError( "The NGP, CIC, TSC and PCS grid interpolation methods do not have implemented a method for computing the scalar fields on the grid. The scalar field cannot be computed!");
    }
    
    
    // check that the options are disabled in the absence of the VELOCITY and SCALAR compiler directives
#ifndef VELOCITY
    if ( uField.selectedVelocity() or aField.selectedVelocity() )
        throwError( "Compiler directive 'VELOCITY' not detected. You cannot interpolate the velocity and/or velocity related quantities if the compiler directive 'VELOCITY' is not activated." );
#endif
#ifndef SCALAR
    if ( uField.selectedScalar() or aField.selectedScalar() )
        throwError( "Compiler directive 'SCALAR' not detected. You cannot interpolate the scalar data and/or scalar gradient if the compiler directive 'SCALAR' is not activated." );
#endif
#ifndef TEST_PADDING
    if ( testPaddedBoundaries )
        throwError( "Compiler directive 'TEST_PADDING' not detected. You cannot test the padding efficiency if the compiler directive 'TEST_PADDING' is not activated." );
#endif
}



// Resolves the per-axis padding length in Mpc from the particle count or a box-relative input.
void User_options::updatePadding(size_t const noTotalParticles)
{
    if ( paddingLength.isNullBox() )
    {
        if ( paddingParticles<=Real(0.) )   // no padding requested
            paddingLength.assign( Real(0.) );
        else        // derive the padding from the requested particle-count per face
        {
            // (entry i of the box is axis i/2)
            // DTFE/SPH: 'paddingParticles' grid spacings, scaled by the mean inter-particle distance
            if ( this->DTFE or this->SPH )
                for (int i=0; i<2*NO_DIM; ++i)
                    paddingLength[i] = paddingParticles / Real( pow(noTotalParticles,1./NO_DIM) ) * fullBoxLength[i/2];
            else if ( this->TSC or this->NGP or this->CIC )  // one grid cell suffices for these stencils (CIC:
                for (int i=0; i<2*NO_DIM; ++i)                  // half a cell)
                    paddingLength[i] = fullBoxLength[i/2] / gridSize[i/2];
            else if ( this->PCS )   // the cubic kernel reaches 2 cells (PCS_kernel::PAD_CELLS): with NO padding
                                    // a periodic run would lose the images at the box faces and every auto-tune
                                    // partition seam
                for (int i=0; i<2*NO_DIM; ++i)
                    paddingLength[i] = Real(2.) * fullBoxLength[i/2] / gridSize[i/2];
        }
        paddingOn = true;
        paddingMpcOn = true;
    }
    else if ( paddingOn and not paddingMpcOn )  // convert box-relative padding to Mpc
    {
        for (int i=0; i<2*NO_DIM; ++i)
            paddingLength[i] *= fullBoxLength[i/2];     // axis i/2
        paddingMpcOn = true;
    }
}




// --grid-window: the window's cell range in the full grid, from the grid box (the region for a
// partition/--region run, else the particle box) and the grid size, each edge snapped OUTWARD to
// the enclosing cell edge (a 1e-6-cell tolerance makes cell-aligned coordinates round-trip
// exactly). Call once the box is final (after the input file's header has set it).
void User_options::computePsWindow()
{
    if ( not this->psWindowOn ) return;
    Box const &b = this->regionOn ? this->region : this->boxCoordinates;
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const len = double(b.coords[2*d+1]) - double(b.coords[2*d]);
        double const n = double(this->gridSize[d]);
        if ( not (len > 0.) or not (n >= 1.) )
            throwError( "'--grid-window': the grid box or the grid size is not set." );
        double const dx = len / n;
        double const f0 = (double(this->psWindow[2*d]) - double(b.coords[2*d])) / dx;
        double const f1 = (double(this->psWindow[2*d+1]) - double(b.coords[2*d])) / dx;
        long lo = long(std::floor(f0 + 1.e-6)), hi = long(std::ceil(f1 - 1.e-6));
        long const n_ = long(this->gridSize[d]);
        if ( hi <= lo ) hi = lo + 1;
        if ( this->periodic and not this->regionOn )
        {
            // a periodic box: a window beyond the box edge WRAPS (lo+dims may exceed the grid, as for
            // a partition straddling the seam) -- the sub-grid mapping and the writers wrap it back
            if ( hi - lo > n_ ) hi = lo + n_;
            lo = ((lo % n_) + n_) % n_;
            this->psWindowLo[d] = size_t(lo);
            this->psWindowDims[d] = size_t(hi - lo);
        }
        else
        {
            if ( hi <= 0 or lo >= n_ )
                throwError( "'--grid-window' lies outside the grid box along axis " + std::to_string(d) + "." );
            if ( lo < 0 ) lo = 0;
            if ( hi > n_ ) hi = n_;
            this->psWindowLo[d] = size_t(lo);
            this->psWindowDims[d] = size_t(hi - lo);
        }
    }
}

void User_options::outputGridSize(size_t out[NO_DIM]) const
{
    for (int d = 0; d < NO_DIM; ++d)
        out[d] = (this->psWindowOn and this->psWindowDims[d] > 0) ? this->psWindowDims[d] : this->gridSize[d];
}

void User_options::outputBox(Box &out) const
{
    Box const &b = this->regionOn ? this->region : this->boxCoordinates;
    out = b;
    if ( not this->psWindowOn or this->psWindowDims[0] == 0 ) return;
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const dx = (double(b.coords[2*d+1]) - double(b.coords[2*d])) / double(this->gridSize[d]);
        out.coords[2*d]   = Real( double(b.coords[2*d]) + double(this->psWindowLo[d]) * dx );
        out.coords[2*d+1] = Real( double(b.coords[2*d]) + double(this->psWindowLo[d] + this->psWindowDims[d]) * dx );
    }
}


/* --scratch-dir's threshold (scratch_alloc.cc): an allocation of at least this many bytes through operator new is an
   mmap'ed file, out of the footprint -- DTFE_SCRATCH_MIN_GB in decimal GB, 1 by default; 0 without --scratch-dir. The
   route's arming (dtfe/setup.cc), both planners (auto_tune.cc) and the SUMMARY (main.cpp) read it here. */
double User_options::scratchThresholdBytes() const
{
    if ( scratchDir.empty() ) return 0.;
    double const gb = Env::real( "DTFE_SCRATCH_MIN_GB", 0. );
    return ( gb > 0. ? gb : 1. ) * 1.e9;
}
