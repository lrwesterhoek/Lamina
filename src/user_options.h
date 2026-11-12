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


/* Declares the Field selector and the User_options class that hold every command-line/config program option. */


#ifndef USER_OPTIONS_HEADER
#define USER_OPTIONS_HEADER

#include <string>
#include <vector>
#include <cstdlib>
#include <ctime> 

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "define.h"
#include "miscellaneous.h"
#include "box.h"
#include "message.h"
#include "run_mode.h"
#include "fixed_units.h"



// Boolean selectors for which fields to interpolate, with helpers to set/query them by name.
struct Field
{
    bool triangulation;
    bool density;
    bool velocity;
    bool velocity_gradient;
    bool velocity_divergence;
    bool velocity_shear;
    bool velocity_vorticity;
    bool velocity_std;
    bool velocity_dispersion;   // PS-DTFE: multi-stream velocity dispersion tensor (trace + full sigma_ij)
    bool scalar;
    bool scalar_gradient;
    bool velocity_tweb;
    bool velocity_vweb;

    Field()
    { triangulation = false; density = false; velocity = false; velocity_gradient = false; velocity_divergence = false; velocity_shear = false; velocity_vorticity = false; velocity_std = false; velocity_dispersion = false; scalar = false; scalar_gradient = false; velocity_tweb = false; velocity_vweb = false; }

    // Enables the field whose name matches 'choice'; returns false if 'choice' is none of the given names.
    bool updateChoices(std::string choice,
                       std::string str_triang, std::string str_den, std::string str_vel, std::string str_grad, std::string str_div, std::string str_shear, std::string str_vort, std::string str_velstd, std::string str_scalar, std::string str_scalarGrad, std::string str_tweb, std::string str_vweb, std::string str_veldisp="")
    {
        if ( choice.compare(str_triang)==0 ) triangulation = true;
        else if ( choice.compare(str_den)==0 ) density = true;
        else if ( choice.compare(str_vel)==0 ) velocity = true;
        else if ( choice.compare(str_grad)==0 ) velocity_gradient = true;
        else if ( choice.compare(str_div)==0 ) velocity_divergence = true;
        else if ( choice.compare(str_shear)==0 ) velocity_shear = true;
        else if ( choice.compare(str_vort)==0 ) velocity_vorticity = true;
        else if ( choice.compare(str_velstd)==0 ) velocity_std = true;
        else if ( not str_veldisp.empty() and choice.compare(str_veldisp)==0 ) velocity_dispersion = true;
        else if ( choice.compare(str_scalar)==0 ) scalar = true;
        else if ( choice.compare(str_scalarGrad)==0 ) scalar_gradient = true;
        else if ( choice.compare(str_tweb)==0 ) velocity_tweb = true;
        else if ( choice.compare(str_vweb)==0 ) velocity_vweb = true;
        else return false;
        return true;
    }

    // True if any field at all is selected.
    bool selected()
    { return ( density or velocity or velocity_gradient or velocity_divergence or velocity_shear or velocity_vorticity or velocity_std or velocity_dispersion or scalar or scalar_gradient or velocity_tweb or velocity_vweb ); }

    // True if any field derived from the velocity gradient (divergence/shear/vorticity/web) is selected.
    // NOTE: velocity_tweb is NOT here -- the T-web is computed from the DENSITY grid (tidal
    // tensor via FFT Poisson), not from the velocity gradient, so it must not force the gradient.
    bool selectedVelocityDerivatives()
    { return ( velocity_divergence or velocity_shear or velocity_vorticity or velocity_vweb ); }
    void deselectVelocityDerivatives()
    { velocity_divergence = false; velocity_shear = false; velocity_vorticity = false; velocity_vweb = false; }
    // True if any velocity-related field is selected.
    bool selectedVelocity()
    { return ( velocity or velocity_gradient or velocity_divergence or velocity_shear or velocity_vorticity or velocity_std or velocity_dispersion or velocity_tweb or velocity_vweb ); }
    void deselectVelocity()
    { velocity = false; velocity_gradient = false; velocity_divergence = false; velocity_shear = false; velocity_vorticity = false; velocity_std = false; velocity_dispersion = false; velocity_tweb = false; velocity_vweb = false; }

    // True if any scalar-related field is selected.
    bool selectedScalar()
    { return ( scalar or scalar_gradient ); }
    void deselectScalar()
    { scalar = false; scalar_gradient = false; }
};


// All user-specified program options plus a few values derived at runtime.
struct User_options
{
    std::string inputFilename; // input particle positions file
    std::string outputFilename;// output file/files (root) name
#ifdef PHASE_SPACE
    std::string lagrangianInputFilename; // file with Lagrangian (initial condition) positions for PS-DTFE
#endif
    std::string tessellationCacheDir;    // --tessellation-cache: directory holding the reusable on-disk tessellation (empty = off); BOTH binaries build a tessellation, so this is not PS-only. See CGAL_triangulation/tessellation_cache.h


    std::vector<size_t> gridSize;// grid size along each direction
    // the grid the output files hold: the --grid-window cells (PS-DTFE) or the full grid, and its box
    void   outputGridSize(size_t out[NO_DIM]) const;
    void   outputBox(Box &out) const;
    // --grid-window x0 x1 y0 y1 z0 z1 (the output coordinates, Mpc; PS-DTFE only, the standard binary refuses it
    // but carries the members so the writers compile once): deposit, allocate and write ONLY the cells of the
    // full '--grid' that lie in this box. The tessellation and every tetrahedron's normalization are those of
    // the full run, so a window run equals the full run's window cells: bit for bit for the sampled deposit
    // (a straddling tetrahedron counts all its samples, deposits the inside part), to float rounding for the
    // exact one (only the window's cells are clipped; the whole weight normalizes). For a fine look at a
    // region of a periodic box (the launcher's exact zoom), where the full fine grid would not fit.
    bool   psWindowOn;
    std::vector<Real> psWindow;        // the box as given
    size_t psWindowLo[NO_DIM];         // its cell range in the full grid, set by computePsWindow() once the box is final
    size_t psWindowDims[NO_DIM];
    void   computePsWindow();          // from psWindow, the grid box (region/boxCoordinates) and gridSize
    // (both binaries compile the setup that reads them; only PS-DTFE sets them)
    // ps_globals_record.h: a --grid-window run that started WITHOUT reading its particles (their globals come
    // from the record in the tessellation cache); they are read late if a needed partition is not cached
    std::string psGlobalsKey;           // its key, computed in main() BEFORE the input is read (the box as given)
    bool   psParticlesDeferred = false;
    size_t psPresetN = 0;               // the recorded particle count (padding, auto-tune)
    Box    psPresetLagBox;              // the recorded bounding box of the initial positions
    bool   psAverageDensityFromParticles = false;   // set when the run computed the mean density from its particles
    bool   periodic;      // true if the particle data is in a periodic box
    bool   periodicInput; // the user's '--periodic' as given -- never reset by the partition/padding code, which turns 'periodic' off once the copies exist (the tessellation cache keys on this: the copies change the tessellation)
    Box    boxCoordinates;// coordinates of the full particle data box
    bool   userGivenBoxCoordinates;// true if the user supplied the box coordinates
    int    inputFileType;   // input file type
    std::vector<int>   readParticleData;       // which particle properties to read
    std::vector<int>   readParticleSpecies;    // which particle species to read
    int    outputFileType;  // output file type


    Field  uField;        // unaveraged field: interpolated at the sampling points
    Field  aField;        // averaged field: volume average over the grid cell
    
    
    bool   partitionOn;   // true if the data IS split into several partitions (user- or auto-selected)
    bool   partitionGiven;// true if the user passed '--partition' at all -- including '1 1 1', which asks for ONE triangulation (partitionOn stays false) and must keep the auto-tuner from choosing a split
    std::vector<size_t> partition;// grid along which to split the particle data so the Delaunay triangulation runs on parts (time/memory)
    std::vector<size_t> subgridOffset; // set by copySubgridInformation(): this sub-grid's origin (cells) in the grid it was split from -- the GPU's shared deposit maps the sub-domain into that grid
    // The run's GRID FRAME, the same in every partition and thread sub-domain (plan P10.8): the full grid's size, this
    // sub-grid's origin in it (cumulative over the partition and the thread levels) and the grid box in double. Set by
    // copySubgridInformation() the first time it splits the grid (from gridSize/boxCoordinates, then the run's);
    // empty in a run that never splits, where gridSize/boxCoordinates are the frame. gridFrame() reads it either way.
    // The exact-average deposits form their geometry in this frame (averaged_interpolation_1.cc GridFrame), so a
    // tetrahedron's shares do not depend on the split.
    std::vector<size_t> fullGridSize;
    std::vector<long>   fullGridOffset;
    std::vector<double> fullGridBox;   // xMin, xMax, yMin, yMax, ...
    inline void gridFrame(double origin[], double dxGlobal[], long offset[]) const
    {
        bool const split = fullGridSize.size() == size_t(NO_DIM);
        for (int d = 0; d < NO_DIM; ++d)
        {
            double const lo = split ? fullGridBox[2*d] : double( boxCoordinates.coords[2*d] );
            double const hi = split ? fullGridBox[2*d+1] : double( boxCoordinates.coords[2*d+1] );
            origin[d]   = lo;
            dxGlobal[d] = ( hi - lo ) / double( split ? fullGridSize[d] : gridSize[d] );
            offset[d]   = split ? fullGridOffset[d] : 0;
        }
    }
    // The run-wide fixed-point units (fixed_units.h RunUnits): one unit per accumulator for the whole run, set by
    // DTFE_setup (DTFE.cpp runUnitsSetup) from the grid, the mean density and the particles before any split, and
    // carried by every copy of the options, so every partition and thread sub-domain holds the same ones
    RunUnits runUnits;
    int    partNo;        // which partition the program should output (index in 'fileGrid')
    int    maxConcurrent; // cap on concurrently-built Delaunay triangulations (peak RAM ~ concurrency x per-triangulation); 0 = all threads. PS-DTFE: concurrent Lagrangian partitions; standard DTFE: concurrent spatial sub-triangulations.
    bool   psParallelTriangulation;    // --parallel-triangulation (PS-DTFE's option; both binaries read DTFE_TRIANGULATION_THREADS): build each Delaunay triangulation on the performance cores with CGAL's TBB insert (a PARALLEL_TRIANGULATION build: TBB=auto/1 for PS-DTFE, TBB_STD=1 for the standard binaries). The default for both binaries since 2026-11-09 for tessellations of at least DTFE_TRIANGULATION_MIN_POINTS (1e6) points (the insert's vertex and cell order differ run to run, so sums in walk order differ at rounding); the auto-tuner budgets its extra bytes per vertex (auto_tune.h)
    bool   parallelTriangulationForced; // --parallel-triangulation: the parallel insert for every tessellation, whatever its size (the default rule has the threshold)
    bool   sequentialTriangulation;    // --sequential-triangulation: the sequential insert (bit-reproducible run to run) for every tessellation
    bool   autoTuneReport;  // --auto-tune-report: print the auto-tuner's decision and predicted peak memory as ONE machine-readable line after loading the input, then exit (the GUI's memory check)
    bool   maxConcurrentOn; // true if the user gave --max-concurrent explicitly (0 is both the default and a legal value, so presence must be tracked separately for the auto-tuner)
    // the run mode (run_mode.h; resolveRunMode, applyRunMode): how hard the run may lean on the machine, both binaries
    RunMode runMode;               // --run-mode gentle|balanced|performance, else DTFE_RUN_MODE (any case), else performance (the binaries' default)
    std::string runModeSource;     // why the mode is what it is: '--run-mode gentle', '--run-mode gentle from DTFE_RUN_MODE=GENTLE' or '--run-mode performance, the default' (the run log's 'run mode' line)
    int    coreBudget;             // the threads the mode allows (applyRunMode; runModeState().coreBudget): N0, the OpenMP default at start-up, or the mode's share of it; 0 before applyRunMode
    std::string coreBudgetNote;    // why the budget is not the mode's own: 'kept: OMP_NUM_THREADS=<v> was given' or 'raised: --max-concurrent <k> was given'; empty otherwise
    std::string priorityNote;      // gentle on Linux: 'nice 10', 'nice <n> kept' or 'priority unchanged (setpriority: <why>)'; empty otherwise
    bool   psGpuShareGiven;        // PS-DTFE --gpu-share was given (the standard binary has no such option)
    long   psGpuShareSpec;         // --gpu-share as given: per mille 0..1000 (1000 = every partition on the GPU, the default), -1 = auto (the run mode's share)
    std::string psGpuShareText;    // --gpu-share's value as typed ('0.6', 'auto'), for the run log
    long   psGpuShareMille;        // the RESOLVED share, per mille (DTFE.cpp psResolveGpuShare, before the auto-tuner): the partitions of a partitioned '--gpu' exact run that deposit on the GPU, the others on the CPU; 1000 = all of them
    std::string psGpuShareSource;  // why the share is what it is (psResolveGpuShare): '--gpu-share 0.6', '--gpu-share auto: ...', '--resume: the checkpoint's'
    
    
    bool   paddingOn;     // true if a padding value is set
    bool   paddingMpcOn;  // true if the padding is in Mpc, not relative to box length
    Real   paddingParticles;// number of particles to use for padding when partitioning
    Box    paddingLength; // padding length (Mpc or relative box length) per direction, both edges
    bool   testPaddedBoundaries; // insert dummy boundary points to check the padding fully covers the region of interest
    
    
    bool   regionOn;      // true if computing only in a user-defined region of the box
    bool   regionMpcOn;   // true if region coordinates are in Mpc, not a fraction of box length
    Box    region;        // coordinates of the user-defined region
    
    
    // averaging computation
    int    method;        // volume-averaging method (standard DTFE --average-method; 1 in PS-DTFE)
    int    noPoints;      // sample points per grid cell for the average
    bool   noPointsOn;    // true if the user specified the number of sampling points
    bool   useMetal;      // standard DTFE: run the method-1 '_a' interpolation on the GPU; needs a Metal build (METAL=1, else falls back to the CPU loop with a warning). Set by --gpu.
    bool   exactAverage;  // standard DTFE, the RESOLVED '_a' average (resolveAverageMode; --exact, the default since 2026-11-09, or --sampled): true = compute the '_a' fields by integrating the LINEAR DTFE interpolant exactly over every cell∩tet intersection (CPU: vendored r3d, order-1 moments) instead of Monte-Carlo sampling; --subsamples is ignored. Runs on the GPU too with --gpu (float32 z-sweep with exact Gauss-Legendre quadrature, metal/exact_clip.metal.inc) and in 2D (triangles clipped by hand); runs on the method-1 scatter topology. False in the constructor: a library caller that fills User_options without readOptions keeps the sampled average unless it sets this.
    bool   exactFlag;     // scratch for the --exact switch (both binaries; the deprecated --exact-average and PS-DTFE's --ps-exact-deposit bind it too): resolveAverageMode folds it into exactAverage / psExactDeposit
    bool   sampledFlag;   // scratch for the --sampled switch (both binaries), folded likewise
    std::string averageModeSource; // why the run averages as it does (resolveAverageMode): '--exact, the default', '--sampled', ... -- the reason the run log's 'volume averaging' / 'grid deposit' line gives; empty where no such line is printed
    bool   faceGradient;  // --face-gradient (docs/cleanup-plan.md P10): the velocity gradient and its derived fields
                          // from EXACT FACE integrals of the interpolant (Gauss), averaged_interpolation_1.cc
                          // avg1FacePass; both binaries' default where it applies (checkGpuOptions, D13), PS-DTFE's
                          // face sums in ps_interpolation.cc
    bool   perTetGradient;// --per-tet-gradient (both binaries, D13): keep the deposit's per-tetrahedron gradient
                          // (PS-DTFE: the stream average of each stream's own) where the face gradient is the default;
                          // DTFE_FACE_GRADIENT=0 does the same, =1 asks for the face gradient
    std::string faceGradientSource; // why the velocity gradient takes its route (checkGpuOptions): '--face-gradient',
                          // '--per-tet-gradient', 'DTFE_FACE_GRADIENT=0' or '=1', 'the default where it applies', or
                          // what keeps the default off (faceGradientBlocker) -- the reason the run log's 'velocity
                          // gradient' line gives
    unsigned psStreamsTwins = 0u; // PS-DTFE face gradient (docs/cleanup-plan.md D12): the regular part's twins the
                          // writer asks for, bit 0 'gradient_a', 1 'shear_a', 2 'vorticity_a'; set by DTFE_setup
                          // from the user's '_a' fields, never by an option (Quantities::psFaceGradientBegin)
    bool   gpuAlias;      // scratch for the backend-neutral --gpu switch (both binaries; PS-DTFE's deprecated --ps-gpu folds into it); OR-ed into useMetal/psUseMetal after parsing (the deprecated --ps-gpu binds it too: foldDeprecatedAliases leaves one of the two entries for po::notify)
    // --sample-points point evaluation, BOTH binaries (see ps_point_eval.h): PS-DTFE reports
    // one stream per folded tetrahedron; the standard binary the Eulerian DTFE interpolant
    // (0/1 coverage). --ps-per-stream/--ps-per-stream-ids parse in both but are rejected in the
    // standard binary (there is no stream decomposition to write).
    std::string psSamplePointsFile;    // --sample-points: file with arbitrary Eulerian evaluation points (empty = mode off)
    bool   psServe;                    // --serve: keep the (single) tessellation in memory and answer point-evaluation batches over stdin/stdout (the dtfelib.Estimator protocol, ps_point_eval.cc) instead of writing grids
    bool   psPerStream;                // --ps-per-stream (PS-DTFE only): also write every stream's density+velocity per sample point (ragged layout)
    bool   psPerStreamIds;             // --ps-per-stream-ids (PS-DTFE only): also write each stream's identity = its 4 Lagrangian-vertex ParticleIDs, sorted (implies --ps-per-stream)
    bool   psPtsDenGrad;               // --pts-den-grad: also write the 'dtfe'-profile density gradient at each sample point (float64 x3; + ragged per-stream file with --ps-per-stream)
    bool   psPtsVelGrad;               // --pts-vel-grad: also write the density-weighted mean velocity gradient at each sample point (float64 x9, [d*3+j] = dv_j/dx_d)
    bool   psPtsScalar;                // --pts-scalar: also interpolate the particles' scalar value (--scalar-dataset) per stream: '.pts_scalar' (density-weighted mean over the streams) + '.pts_stream_scalar' with --ps-per-stream
    bool   psPtsLagrangian;            // --ps-per-stream-lagrangian (PS-DTFE only): also write each stream's Lagrangian coordinate q(x) -- where the matter at the point started -- to '.pts_stream_lagpos' (float64 x3 per stream; implies --ps-per-stream)
    std::string scalarDataset;         // --scalar-dataset NAME: Gadget HDF5 input reads '/PartTypeN/NAME' (one value per particle, raw) as the scalar field every estimator interpolates (empty = the reader's own scalar, if any)
    bool   psServeProgress;            // --serve-progress: plain stderr progress lines for the build and every request, whatever --verbose says (the launcher's progress bar)
    int    serveResident;              // --serve-resident N: with '--serve --partition', how many partition tessellations stay in memory at once (the rest are reloaded from '--tessellation-cache' on demand); 0 = automatic
    bool   psStreamDensityGeometric;   // PS-DTFE --ps-stream-density: true = 'geometric' (m_tet/V_eul, matches the mass-conserving deposit), false = 'linear' (vertex-density interpolation, matches the Feldbrugge reference; DEFAULT; 'dtfe' is its deprecated name). Always false in the standard binary.
#ifdef PHASE_SPACE
    int    psAvgSubsamples; // PS-DTFE --subsamples: linear sub-sample count nSub for the sampled deposit's '_a' fields (nSub^NO_DIM sub-grid per cell, cost ~nSub^NO_DIM); 3 = 27 sub-points (default), 1 = cell-centre. The exact deposit ignores it.
    bool   psUseMetal;      // PS-DTFE: run the grid deposit on the GPU; needs a Metal build (METAL=1, else falls back to the CPU deposit with a warning). Set by --gpu.
    bool   psLinearDeposit;            // PS-DTFE --ps-linear-deposit: weight each tetrahedron's interior samples by the DTFE-interpolated linear density (renormalized per tet, so the deposited total still equals the tet mass EXACTLY) instead of equal shares
    bool   psCausticCusps;      // --ps-caustic-cusps: also estimate the A3 (cusp) indicator, bit 7 of the caustic mask. Opt-in: it costs a neighbour gather per near-fold tetrahedron and, being a finite-difference estimator on an irregular tessellation, carries a measurable false-positive rate (see ps_caustic_class.h)
    bool   psCaustics;                 // PS-DTFE --ps-caustics: record per grid cell whether tetrahedra of BOTH map orientations (sign of det(Ax) = Lagrangian->Eulerian parity, flips at each fold) overlap it; the 0/1 flag is written as '<output>.caustic'. CPU and GPU deposits; 3D only.
    Real   psAlphaShape;               // PS-DTFE --ps-alpha-shape <c>: a NON-periodic run keeps only the tetrahedra whose Lagrangian circumradius is at most c mean particle spacings -- the cloud's alpha shape instead of its convex hull (default 3; 0 = the whole hull)
    Real   psLagPadding;               // PS-DTFE --ps-lagrangian-padding <k>: pad each Lagrangian partition by k mean particle spacings (and >= 4.2 alpha cuts) instead of the Eulerian --padding fraction with its 0.3-partition floor; default 8, 0 = the Eulerian rule
    Real   psHaloRelease;              // PS-DTFE --ps-halo-release <D>: during the GRID deposit, a tetrahedron whose geometric stream density rho_geo/rho_bar = V_lag/V_eul exceeds D is deposited monolithically at its Eulerian centroid cell (the sub-sample-spacing fallback path) instead of being rasterized over its bbox. 0 (default) = off. Mass conservation is exact; point evaluation is unaffected.
    bool   psExactDeposit;             // PS-DTFE, the RESOLVED deposit (resolveAverageMode; --exact, the default since 2026-11-09 for a run with an '_a' field, or --sampled; deprecated: --exact-average, --ps-exact-deposit). False in the constructor, like exactAverage. True = replace the nSub^3 sub-sampled grid deposit by EXACT tetrahedron-cell intersection moments (vendored r3d, Powell & Abel 2015, on the CPU). CPU (double, reference) and GPU (float32 z-sweep, --gpu); 3D only; nSub is ignored (the exact deposit IS the nSub->infinity limit, so '.den' == '.a_den').
    bool   psVertexMass;               // PS-DTFE --ps-vertex-mass: per-tetrahedron mass = sum over its vertices of weight/degree (chart-independent) instead of rho_bar * V_lag. Removes the 1-D(z_ic)/D(z) density-contrast suppression when --ps-lagrangian-input holds PERTURBED IC positions (e.g. TNG z=127) rather than the unperturbed lattice.
    bool   psVolumeWeighted;           // PS-DTFE --ps-volume-weighted: weight the velocity/gradient/scalar moment deposits (NOT the dispersion -- sigma_ij is an f-weighted CBE moment and stays mass-weighted, bit-identical to a default run) by EULERIAN-VOLUME shares instead of mass shares -> the cell values become volume-weighted means (the standard-DTFE '_a' convention, what linear theory's -aHf delta refers to) instead of mass-weighted (momentum-like) means. Density/streams/caustics unchanged. CPU and GPU deposits (and the exact deposit, --exact); incompatible with --ps-linear-deposit.
    bool   psPtsLegacyConventions;     // PS-DTFE --ps-point-conventions legacy: the point evaluation keeps rho_bar*V_lag masses and density-weighted stream means whatever --ps-vertex-mass / --ps-volume-weighted say. Default false ('grid'): the points follow those two grid conventions.
#endif
    Real   averageDensity;// density normalization (computed as the box average if not given)
    size_t randomSeed;    // random seed for the Monte Carlo interpolation
    
    
    
    // light cone computations
    std::vector<Real> originPosition;   // light cone origin
    bool              redshiftConeOn;   // true if interpolating to a light cone grid
    Box               redshiftCone;     // light cone grid bounds (r_min, r_max, theta_min, theta_max, psi_min, psi_max)
    
    
    // additional options
    std::string configFilename;// config file from which program options were read
    std::string scratchDir;    // --scratch-dir: back >=1 GB allocations (the full-grid accumulators) with mmap'ed files in this LOCAL directory instead of RAM; empty = off. See scratch_alloc.h
    bool   NGP;           // true if to use NGP grid interpolation instead of DTFE
    bool   CIC;           // true if to use CIC grid interpolation instead of DTFE
    bool   TSC;           // true if to use TSC grid interpolation instead of DTFE
    bool   PCS;           // true if to use PCS grid interpolation instead of DTFE
    bool   SPH;           // true if to use SPH grid interpolation instead of DTFE
    bool   Voronoi;       // true if to use Voronoi volume density with NGP grid assignment
    bool   interlace;     // true if to use interlacing to reduce aliasing in density field
    int    SPH_neighbors; // keep track of the number of neighbors used for the SPH grid interpolation
    Real   MpcValue;      // the value of 1Mpc in input units
    bool   extensive;     // true if scalar fields are extensive (else intensive); matters for TSC/SPH
    bool   approxPSD;     // compute approximate phase-space density f = rho * g (velocity-space DTFE)
    Real   lambda_th;    // --lambda-th: eigenvalue threshold for T-web/V-web classification (default 0.0)
    Real   hubbleParam;  // Hubble parameter h for T-web/V-web normalization (-1 = read from header)
    Real   scaleFactor;  // scale factor a of the snapshot (-1 = read from header 'Time'). Gadget velocities are u = v_pec/sqrt(a), so the V-web gradient needs a sqrt(a) to reach peculiar km/s -- without it the eigenvalues are 1/sqrt(a) inflated at high z (4.6x at z=20)
    int    verboseLevel;  // verbose level (see 'message.h')
    Real   randomSample;  // size of the random subsample if using only a subset of the data
    size_t poisson;       // if !=0, generate this many random particles instead of reading positions
    std::vector<std::string> additionalOptions; // extra options the user can pass to the code
    bool   transformToRedshiftSpaceOn;          // true if transforming position to redshift space
    std::vector<Real> transformToRedshiftSpace; // velocity coords for redshift-space transform (H=100 h km/s/Mpc)
    
    
    
    // set during runtime, not from the command line
#ifdef PHASE_SPACE
    Box    lagrangianRegion; // unpadded Lagrangian partition region (PS-DTFE cell ownership check)
    double psLagAlphaRadius; // the --ps-alpha-shape cut in Lagrangian length (c x mean spacing, set for non-periodic runs; 0 = no cut)
    bool   psTessellationPrebuilt;   // set by psInterpolatePrebuilt (triangulation.cpp): the partition look-ahead built the tessellation already, so DTFE_interpolation skips buildTessellation
    bool   psMayClearDT;     // set by the DTFE_interpolation overload that OWNS its triangulation: the last interpolateGrid_phaseSpace call may clear the DT right after its deposit (frees ~650 B/vertex before the merge phase). Never set when the caller keeps the triangulation (TRIANGULATION library API).
    bool   psSuppressGridStats; // PS-DTFE partition path: silence per-partition coverage/stream stats (each covers ~1/Npart of the grid); aggregate reported once after the loop
    bool   psCacheOnly = false;  // PS-DTFE partition path: the partition's particles were NOT selected because its cached tessellation's header matched -- a load that then fails is an error, never an empty build (internal, no option)
    bool   psUseSubgrid;        // PS-DTFE partition path: allocate only each partition's Eulerian bounding box, mapped back via addFromSubgrid -> lower peak memory
    double psPartitionBytes;    // PS-DTFE partition path: what ONE partition in flight adds (the auto-tuner's whole lane, perPartition + laneOnce, + the GPU deposit's host snapshot once); the memory gate before each partition waits for this much free memory (0 = no estimate: no gate)
    double psOnceBytes;         // PS-DTFE partition path: the part of psPartitionBytes one partition at a time holds (AutoTuneReport::onceBytes: the GPU deposit's snapshot -- a mixed run's lock holder's grids with it -- and the lane share's laneOnce); the dynamic concurrency books it once, not per partition (0: a CPU run, every partition books psPartitionBytes in full)
    int psInFlightCap;          // PS-DTFE partition path: the most partitions in flight the auto-tuner's lane share holds for; the dynamic concurrency stops there (0: no limit)
    bool   psOrderedMerge;      // PS-DTFE partition path: merge the partitions' grids in index order (bit-reproducible sums). THE DEFAULT (the STAGED ordered merge, DTFE.cpp: a partition finished before its turn hands its sub-grids to a staging map, within the memory the machine has free, and its thread goes on, so threads do not idle behind a slow partition -- waiting behind one can cost 35%); '--ps-merge completion-order' (psCompletionOrderMergeFlag) adds in completion order instead (the grids then differ run to run in the last bits, even the exact Metal mass grid). The auto-tuner's SPEED split of a small set always merges in order (psSpeedSplit). '--ps-merge index-order' is accepted and changes nothing
    bool   psOrderedMergeFlag;  // the user gave --ps-merge index-order, or --resume (psOrderedMerge is then set too); unlike the speed split, such a run keeps the memory gate (DTFE.cpp)
    bool   psCompletionOrderMergeFlag; // --ps-merge completion-order: the partitions' grids are added as they finish (no staging, not bit-reproducible); --resume overrides it
    bool   psFixedPointMerge;   // --ps-merge fixed-point (opt-in): every partition's sub-grids are added AS IT FINISHES into exact integer full-grid accumulators (int64 words + int8 carries, src/ps_fixed_merge.h), so the sums cannot depend on the completion order or the concurrency and nothing parks or spills; differs from the ordered merge at ~1e-7. Exclusive with --ps-merge completion-order ('--ps-merge index-order', which changes nothing, is accepted beside it); refused in the double build and with --grid-window
    bool   psSpeedSplit;        // set by the auto-tuner: its SPEED split of a small set -- tiny uniform partitions, ordered merge, no memory gate
    bool   psNoOverlap;         // --no-overlap (deprecated: --ps-no-overlap): no look-ahead tessellation of partition k+1 while partition k deposits on the GPU (DTFE.cpp)
    bool   psLookAheadOffBudget; // set by the auto-tuner: one partition plus the look-ahead's tessellation does not fit the memory budget, so this run has no look-ahead (its predicted peak counts none)
    bool   psDynamicConcurrency;    // PS-DTFE: the auto-tuner chose the concurrency, so the partition loop may run more than its mc while the live memory holds them (DTFE.cpp, auto_tune.h PartitionMemoryGate; plan P8.10; DTFE_DYNAMIC_CONCURRENCY=0 keeps the plan's)
    double psReserveEnd;        // ... what the run allocates after the loop (derived grids, finalized point outputs), bytes
    double psReserveRecords;    // ... the point stream records it writes as the partitions merge (budgeted), bytes
    double psReserveFirstMerge; // ... the deposit grids the first merge sizes (moment weights, face sums), bytes
    double psBudgetBytes;       // ... the auto-tuner's memory budget (0: none known)
    double psParkRoom;          // what the memory budget leaves beside the auto-tuner's predicted peak with every partition lane whole (budget - predictedWholeLanes): the staged ordered merge parks finished partitions in RAM only within it and spills the rest (DTFE.cpp parkWhere; HUGE_VAL = no budget known)
    double psLookAheadRoom;     // what the memory budget leaves beside the grids, particles and ONE partition (the auto-tuner's budget - fixed - perPartition): the look-ahead runs only when it holds no more (DTFE.cpp; HUGE_VAL = no budget known)
    bool   psDeferNormalization; // PS-DTFE partition path: keep fields as density-weighted moments (no per-partition normalize) so they sum linearly and normalize once afterwards; correct for multi-stream cells spanning partitions
#endif
#ifndef PHASE_SPACE
    // the standard partition loop's LOOK-AHEAD (DTFE.cpp: PS-DTFE's look-ahead, for the standard loop)
    bool   stdTessellationPrebuilt; // set by dtfeInterpolatePrebuilt (triangulation.cpp): the look-ahead built this sub-domain's tessellation already, so DTFE_interpolation skips buildTessellation
    bool   stdNoOverlap;        // --no-overlap: no look-ahead tessellation of partition k+1 while partition k deposits on the GPU
    bool   stdLookAheadOffBudget; // set by the auto-tuner: the busiest partition plus the next one's tessellations do not fit the memory budget, so this run has no look-ahead (its predicted peak counts none)
    double stdLookAheadCounted; // the auto-tuner's booked look-ahead: kStdLookAheadFactor x its count of the partition that needs the most, from the particles (0 = no plan); the loop's free-memory check takes the larger of it and its own uniform count
    double stdLookAheadRoom;    // what the memory budget leaves beside the grids, particles, the GPU snapshot and ONE partition (the auto-tuner's budget - fixed - gpuSnapshot - perPartition); HUGE_VAL = no budget known
#endif
    // both binaries (standard DTFE resumes its partition loop through the same partition_checkpoint.h)
    bool   psResume;            // --resume: continue a killed partitioned run from its checkpoint ('<output>.runlog.checkpoint', partition_checkpoint.h); in PS-DTFE it turns psOrderedMerge + psOrderedMergeFlag on (over a --ps-merge completion-order)
    bool   psNoCheckpoint;      // no partition checkpoints: set under '--interlace', whose two passes (the second on a half-cell-shifted grid) would share one checkpoint and key; a copy of the options keeps it when main.cpp clears 'interlace' for the second pass
    Box    paddedBox;      // dimensions of the padded box
    std::vector<Real> fullBoxOffset; // full box left side per axis (redundant with 'boxCoordinates')
    std::vector<Real> fullBoxLength; // full box length per axis (redundant with 'boxCoordinates')

    bool   DTFE;           // true if DTFE interpolation method
    bool   userDefinedSampling; // true if the user supplied sampling coordinates
    int    noProcessors;   // number of threads running in parallel
    int    threadId;       // thread id for parallel processes
    Real   totalTime;      // total CPU time used by the program
    std::string programOptions; // all program options supplied by the user
    std::vector<std::string> commandTokens; // argv[1..] one token each, every deprecated option name in its canonical spelling: the settings key of a partition checkpoint (partition_checkpoint.h)
    
    
    
    
    // function bodies are in "user_options.cc"
    User_options();
    void readOptions(int argc, char *argv[], bool getFileNames=true, bool showOptions=true );   // parse the command line/config, validate, and store the options
    void addOptions(po::options_description &allOptions,
                    po::options_description &visibleOptions,
                    po::positional_options_description &p); // register every program option with Boost.Program_options
    void helpInformation( po::options_description &visibleOptions, char *progName ); // print the detailed help and exit
    void shortHelp( char *progName ); // print the summary help and exit
    void printOptions(); // print the resolved configuration the run will use
    void updateFullBox(Box &newBox); // set 'boxCoordinates' and recompute 'fullBoxOffset'/'fullBoxLength'
    void updateEntries(size_t const noTotalParticles,
                       bool userSampling);   // after reading input: update paddedBox/fullBoxOffset/fullBoxLength and error-check members
    void updatePadding(size_t const noParticles); // resolve the padding length (from particle count or box-relative input)

private:
    // the phases of readOptions(), in the order it calls them
    void parseOptions(int argc, char *argv[], po::options_description &allOptions,
                      po::positional_options_description &p, po::variables_map &vm,
                      MESSAGE::Message &showMessage); // the command line and the '--config' file into 'vm'
    void checkFileNameOptions(char *argv[], bool getFileNames, po::options_description &visibleOptions,
                              po::variables_map &vm, MESSAGE::Message &showMessage); // input/output file names, '--help'
    void readMainOptions(po::variables_map &vm);         // '--grid', '--box', '--input-format', '--periodic'
    void readFieldOptions(po::variables_map &vm);        // '--field'
    void readRegionOptions(po::variables_map &vm);       // '--grid-window', '--region', '--region-mpc'
    void foldDeprecatedAliases(po::variables_map &vm);   // the deprecated spellings onto the canonical options
    void resolveAverageMode(po::variables_map &vm);      // '--exact' / '--sampled': the '_a' average (exactAverage, psExactDeposit)
    void resolveRunMode(po::variables_map &vm);          // '--run-mode' / DTFE_RUN_MODE (runMode) and PS-DTFE's '--gpu-share' as given
    void checkGpuOptions(po::variables_map &vm);         // '--gpu', the triangulation's insert, the gradient route, '--exact'
    std::string faceGradientBlocker(po::variables_map const &vm) const;   // why the face gradient is not this run's
                                                                          // default route ("" where it is)
    void checkDirectoryOptions();                        // '--scratch-dir', '--tessellation-cache'
    void checkPointEvalOptions(po::variables_map &vm);   // '--sample-points', '--serve' and their companions
    void checkPsDepositOptions(po::variables_map &vm);   // PS-DTFE only: caustics, alpha shape, deposit variants
    void readPartitionOptions(po::variables_map &vm);    // '--partition', '--partition-index', the merge order
    void readPaddingOptions(po::variables_map &vm);      // '--padding', '--padding-mpc', '--no-padding-test'
    void readAveragingOptions(po::variables_map &vm);    // '--average-method', '--subsamples', '--mean-density', '--seed'
    void readRedshiftConeOptions(po::variables_map &vm); // '--redshift-cone', '--cone-origin'
    void readAdditionalOptions(po::variables_map &vm);   // the grid methods, '--interlace', units, '--redshiftSpace'
    void applyMethodFieldRules();                        // the field choice the build and the grid method allow
    void applyRunMode();                                 // the mode's levers: the thread budget, the GPU pace and item cap, gentle's nice (run_mode.h)
};





#endif


