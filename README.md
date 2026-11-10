# Lamina

**Lamina** (Linear Adaptive Multi-stream Interpolation for N-body Analysis) turns the particles of a cosmological
simulation into continuous density and velocity fields with two estimators built on Delaunay tessellations, in two and
three dimensions:

* **DTFE**, the Delaunay Tessellation Field Estimator ([Schaap & van de Weygaert 2000](https://ui.adsabs.harvard.edu/abs/2000A%26A...363L..29S/abstract)), run by the program
  `DTFE`: the particles are tessellated where they are and every field is linear inside each tetrahedron, so the
  resolution follows the particles from voids to clusters. Lamina extends the DTFE public software of
  [Cautun & van de Weygaert (2011)](https://ui.adsabs.harvard.edu/abs/2011arXiv1105.0370C/abstract).
* **PS-DTFE**, the phase-space DTFE ([Feldbrugge 2024](https://arxiv.org/abs/2402.16234)), run by the program `PS-DTFE`: the particles are
  tessellated at their initial (Lagrangian) positions, before any shell crossing, and each tetrahedron is followed to
  where the simulation moved it. Where the dark-matter sheet has folded, a point lies in several streams: PS-DTFE
  resolves each of them, where DTFE averages them into one.

A *lamina* is a thin sheet: the dark-matter sheet in phase space that PS-DTFE tessellates, and the walls of the cosmic
web that both estimators resolve without smoothing them away.

Around the two programs: GPU deposits (Metal on Apple Silicon), exact cell averages and
evaluation at any point, a domain decomposition chosen from the machine's memory, resumable runs, a Python package
(`dtfelib`), batch drivers for IllustrisTNG snapshots and a graphical launcher. The programs run in parallel on
shared-memory machines (OpenMP). Although they were written for cosmological simulations and galaxy surveys, they
interpolate any quantity known at a set of points onto a grid or at other points.

The [user guide](docs/DTFE_user_guide.pdf) of the DTFE public software documents the original program and its paper,
[Cautun & van de Weygaert (2011), arXiv:1105.0370](https://arxiv.org/abs/1105.0370), the method; this README and
each program's `--full-help` document the rest. A test dataset and analysis of the code output is given in the [demo directory](demo): `./DTFE demo/z0_64.gadget out --grid 64 --field density --periodic --padding 12` works out of the box (the box is periodic, and its voids are wider than the default padding of 5 particle spacings: with it about 1% of the cells would lie outside the tessellation, which the run reports). [demo/COMMANDS.md](demo/COMMANDS.md) is a cheat sheet of every script and binary with its useful flag combinations.

## The DTFE method
The Delaunay Tessellation Field Estimator (DTFE) method represents the natural way of going from discrete samples/measurements to values on a periodic grid and it is especially suitable for astronomical data due to the following reasons:
* Preserves the multi-scale character of the point distribution. This is the case in numerical simulations of large scale structure where the density varies over more than 6 orders of magnitude.
* Preserves the local geometry of the point distribution. This is important in recovering sharp features like the different components of the cosmic web (i.e. clusters, filaments, walls and voids).
* The method does not depend on user defined parameters or choices.
* The interpolated fields are volume weighted (versus mass weighted quantities in most other interpolation schemes). This can have a significant effect especially when comparing with analytical predictions which are volume weighted.

For detailed information about the DTFE method see [Schaap and van de Weygaert (2000)](https://ui.adsabs.harvard.edu/abs/2000A%26A...363L..29S/abstract), [van de Weygaert and Schaap (2009)](https://ui.adsabs.harvard.edu/abs/2009LNP...665..291V/abstract), and [Cautun et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011arXiv1105.0370C/abstract).

| <img src="figures/DTFE_filament.png" width="600" title="An illustration of the adaptive nature of the DTFE method."> |
|:------:|
| Figure 1: *An illustration of the 2D Delaunay tessellation of a set of particles from a cosmological simulation. Courtesy: Willem Schaap.* |

| <img src="figures/DTFE_paper_density.png" width="800" title="Examples of DTFE density fields."> |
|:------:|
| Figure 2: *An example of the DTFE density field form a cosmological simulation. The right panel shows the same result but now using the smoothed particle hydrodynamics (SPH) method.* |

| <img src="figures/DTFE_paper_velocity.png" width="800" title="Illustration of the DTFE velocity field."> |
|:------:|
| Figure 3: *A map of the DTFE computed velocity flow (left panel) and velocity divergence (right panel) corresponding to the density field shown in Figure 2.* |


## Summary of software features

* Works in both 2 and 3 spatial dimensions.
* Interpolates the fields to three different types of grids:
  + Regular rectangular and cuboid grid - useful for cosmological simulation.
  + Redshift cone (spherical coordinates) grid - useful for galaxy redshift survey or for mock observations.
  + User given sampling points - can describe any complex or non-regular sampling geometry
* Returns both the value at the centre of each cell of the interpolation grid as well as the value averaged over each cell.
* Interpolates the density, velocity, velocity gradient, divergence, shear and vorticity, any number of further fields and their gradients, and the T-web and V-web classes.
* Each sample point has a weight associated to it to represent multiple resolution N-body simulations and observational biases for galaxy redshift surveys.
* Periodic boundary conditions, zoom-in regions, and splitting the data into partitions processed in parallel on shared-memory machines.
* A phase-space (Lagrangian) variant, `PS-DTFE`, that recovers the multi-stream structure of the cosmic web: the multi-stream density, the stream count, the velocity dispersion and the caustic surfaces.
* Exact, sampling-free cell averages by default (Monte-Carlo or sub-sampled averages on request), and evaluation at arbitrary points with exact point location.
* An interactive mode: the tessellation is built once and queried from Python (`dtfelib.Estimator`) at any points; boxes too large for one tessellation are served as a composite of partitions kept on disk.
* Automatic choice of the domain decomposition from the data and the machine's memory, crash-safe outputs and resumable runs, and an out-of-core mode for grids larger than the memory.
* A tessellation cache on disk that later runs on the same data reuse.
* Optional GPU deposit on Apple Silicon (Metal); an optional double-precision build.
* For comparison: nearest grid point (NGP), cloud in cell (CIC), triangular shape cloud (TSC), piecewise cubic spline (PCS), SPH and Voronoi interpolation.
* Returns the Delaunay tessellation of the given point set, and can be used as an external library.
* Batch drivers for IllustrisTNG snapshots, a Python analysis stack (`python/dtfelib`, `python/plot`) and a graphical launcher.


## Installation and Building

### Supported Platforms
- **macOS** (Intel and Apple Silicon)
- **Linux** (Ubuntu, Debian, Fedora, CentOS, RHEL, and other distributions)

### Prerequisites

#### macOS
1. **Install Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install dependencies:**
   ```bash
   brew install gsl boost cgal mpfr gmp hdf5 fftw llvm libomp
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install libgsl-dev libboost-all-dev libcgal-dev libmpfr-dev libhdf5-dev libgmp-dev libfftw3-dev
```

#### Linux (Fedora/RHEL/CentOS)
```bash
# For Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel fftw-devel

# For older RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel fftw-devel
```

#### Linux (Arch/Manjaro)
```bash
sudo pacman -S base-devel gsl boost cgal mpfr hdf5 gmp fftw
```

Optional: `tbb` (the parallel Delaunay insert of both binaries) and `libdeflate` (faster tessellation-cache compression);
the Makefile uses them when it finds them.

### Quick start

`./scripts/install.sh` installs any missing dependencies, builds both binaries with the best backend for the machine
(Metal on Apple Silicon, else the CPU), rebuilds any double or 2D pair already
built, installs the Python package and sets up the graphical launcher. Flags: `--cpu`, `--no-deps`, `--jobs N`,
`--double` (also the double-precision pair), `--no-python`, `--no-gui` (e.g. on a cluster), `--no-app`, `--docker`
(the Linux container image instead; never Metal); `./scripts/install.sh --help` explains each. A failed compile
leaves the previous binaries in place and prints the end of its build log.

### Building by hand

```bash
make test-platform        # the detected platform, compiler and paths
make deps-check           # every required header; prints the install command for anything missing
make DTFE PS-DTFE         # the standard and the phase-space binary (add -j8)
make library              # the shared library libDTFE (.dylib / .so)
make clean                # every binary set, libDTFE and the objects
python3 -m pip install -e python   # dtfelib (editable: it finds ./DTFE and ./PS-DTFE; else set DTFE_BIN_DIR)
```
`./DTFE --version` / `./PS-DTFE --version` print the git revision and build time, which every run also writes into
its log (`Build:`).

Build variables (combine them on one `make` line):

| variable | effect |
|---|---|
| `METAL=1` | Apple Silicon: the standard binary's `_a` interpolation and the PS-DTFE deposit on the GPU (`--gpu` in both), with a CPU fallback |
| `DOUBLE=1` | the double-precision pair `DTFE-double` / `PS-DTFE-double` (see Precision) |
| `DIM=2` | the 2D pair `DTFE-2d` / `PS-DTFE-2d` (`-2d-double` with `DOUBLE=1`); CPU only |
| `TBB=auto` / `1` / `0` | the parallel Delaunay insert of both binaries (the default for tessellations of a million points or more where cores are free; `--parallel-triangulation` / `--sequential-triangulation`; `TBB_STD` sets the standard pair apart): when found / required / left out |
| `LIBDEFLATE=auto` / `1` / `0` | libdeflate instead of zlib for the tessellation cache (both write plain gzip) |
| `BUILD_MODE=debug` | debug symbols, no optimization, sanitizers |
| `EXTRA_FLAGS`, `CC_OVERRIDE`, `<LIB>_PATH_OVERRIDE` | extra compiler flags, another compiler, a library elsewhere (see the Makefile's header) |

The GPU code is Metal-only: on Linux and Intel Macs the programs run on the CPU. (An earlier CUDA/HIP port of the
kernels, never run on NVIDIA or AMD hardware, is kept on the branch `archive/cuda-hip-2026-11-08`; `CUDA=1`, `HIP=1` and
`CUDAEMU=1` now stop the build with a message saying so.) Each binary set has its own object directories (`o/`,
`o_ps/`; `o_d/`, `o_ps_d/`; `o_2d/`, `o_ps_2d/`), so the sets coexist. The build mode is stamped there
(`o/.build_mode`); a mode switch wipes the objects, and `make DTFE $(cat o/.build_mode)` rebuilds without changing the
mode. The run scripts find the GPU mode from the `.gpu_mode_metal` stamp.

**Compile-time options.** The Makefile's `OPTIONS_COMMON` block sets the compiled-in features and defaults:
`-DVELOCITY`, `-DSCALAR` with `-DNO_SCALARS=1` (one scalar component; the GPU deposits only that case),
`-DINPUT_FILE_DEFAULT=105` (Gadget HDF5), `-DMPC_UNIT=1000` (input lengths in kpc), `-DOUTPUT_FILE_DEFAULT=101`
(binary output), `-DOPEN_MP`, `-DTEST_PADDING`, `-DTRIANGULATION`. Edit it there; see the Makefile for the full list.

**Precision.** `DOUBLE=1` keeps every position, density, field and accumulator in double from the input read on
(the Gadget-HDF5 reader then keeps TNG's double coordinates, which the default build rounds on read) and writes float64
grids (`dtfelib` and the launcher read either), at about twice the memory. The run scripts select the pair with
`DTFE_PRECISION=double`, the launcher with its Precision setting. The GPU kernels are single precision: a double build
hands the GPU float copies of its tetrahedra, so its GPU stream counts and caustic flags may differ from its CPU's in up
to 1e-4 of the cells (identical in the single build). The PS-DTFE grids are exact 64-bit fixed-point sums on the GPU
and the CPU (single build: identical from run to run at any thread count; `DTFE_GPU_FLOAT_MASS=1` and
`DTFE_GPU_FLOAT_MOMENTS=1` bring back float sums for comparison).
A share without a 64-bit integer value (NaN, infinity, beyond 2^63 units) is left out and counted, and the run ends
with a warning naming the grid.

**2D.** `make DTFE PS-DTFE DIM=2` reads the same Gadget-HDF5 layout with two-column `Coordinates`, `Velocities` and
`InitialCoordinates` (`python3 tests/generate_ps_test_data.py --dim 2` writes such files). Every field, the T-web
(classes 0 void, 1 filament, 2 node), the exact and sampled averages (`--exact`, `--sampled`), `--ps-caustics` (no swallowtail
bit), `--grid-window`, `--partition`, the tessellation cache, `--sample-points` (`x y` points) and `--serve` work in 2D;
the GPU and `--parallel-triangulation` are 3D only. A snapshot whose coordinates have the wrong number of columns for
the binary is refused with the name of the binary to use.

### Tests

`tests/ci_suite.sh <stage>` is the one list of suites that CI (`.github/workflows/ci.yml`) and the Docker image run:

| stage | runs |
|---|---|
| `build` | builds the single, double and 2D pairs (CPU) |
| `core` | every CPU suite against `./DTFE` and `./PS-DTFE`, the auto-tune and `--scratch-dir` suites, the dtfelib and void suites, the docs check, `python/selftest.py` |
| `double` | `tests/ps_double_check.sh`, then the program suites against the double pair |
| `2d` | `tests/ps_2d_check.sh` (the 2D pair against analytic 2D flows) |
| `gui` | the launcher's job specifications; the end-to-end window test when `DTFE_GUI_PYTHON` names a Python with PySide6 |
| `oldpy` | every `.py` parses under Python 3.10 |
| `variants` | `TBB=0 LIBDEFLATE=0`, `make library` and `demo/examples` linked against it |
| `all` | `build core double 2d` |

CI runs three jobs: build-and-test (build, core, gui, double, 2d), old-python (every .py parses under Python 3.10, and
ruff's pyflakes rules pass) and compile-variants. The GPU suites need a GPU build and run
by hand: `tests/dtfe_metal_check.sh`, `tests/gpu_scalar_check.sh`, `tests/gpu_mass_check.sh`,
`tests/gpu_retry_check.sh`, `tests/dtfe_gpu_retry_check.sh`.
`DTFE_TEST_PRECISION=double tests/<suite>` runs a suite against the double pair. The suites run `PYTHON` when set, else the
first python3 that imports numpy and h5py (`scripts/h5py_python.sh`). The standard-DTFE regression
(`tests/run_tests.py`) compares a periodic 1000-particle run with `tests/reference/regression_periodic.txt` to 3e-4
relative; after an intended change of the estimator regenerate it with `python3 tests/run_tests.py --update-ref`.

The development tools in `tests/dev/` (not run by CI; macOS):

* `tests/dev/battery.sh <scratch dir>` runs everything above that a Mac can run, in 32 stages (about 1.5-2 h): the three
  builds, the `core`, `double`, `2d` and `gui` stages, the GPU suites in single and double precision, the Python
  suites under CI's package versions with no simulation data, and `deprecation_gate` (no unexpected 'is deprecated'
  line in the suites' temporary files). `tests/dev/make_ci_venv.sh` makes that Python (python
  3.11, numpy 1.26.4, scipy 1.11.4, matplotlib 3.6.3; by default in `~/.venvs/dtfe-ci`). `--list` names the stages,
  `--only` picks some, and `<scratch dir>/battery/SUMMARY.txt` records the commit and whether the tree changed
  during the run.
* `tests/dev/bytecmp.sh <reference binaries> <output dir>` checks that a change moves no output: 17 runs of both
  programs (single, double, 2D, CPU and GPU, one domain and partitioned, the CPU exact deposits) compared byte for byte
  with a copy of the six binaries taken before the change. `--gpu-exact` adds the GPU's exact deposits (6 runs),
  for a change that leaves the GPU kernels' source alone: they are float32 with fast math, so only two builds of the
  same kernel source give the same bytes.
* `tests/dev/gui_shots.py <out dir>` renders every tab of the launcher at two window sizes and the Explore tab's 3D
  view, on a synthetic data root, to see layout bugs the suites cannot (`--native`: in a real window, with the
  system style). It prints the width each tab's form needs against the width it gets, folded and with its folded
  groups open (`--open`: the shots taken with them open).
* `tests/dev/tune_report_set.sh <binaries> <out file>` writes the auto-tuner's plan for a fixed list of 91
  configurations (`--auto-tune-report`, with the machine's RAM simulated), one line each. Diff the sets of two
  builds: only the configurations a change targets may differ.


## Running the binaries

```bash
./DTFE    <input file> <output prefix> [options]      # standard DTFE (Eulerian tessellation)
./PS-DTFE <input file> <output prefix> [options]      # phase-space DTFE (Lagrangian tessellation)
./PS-DTFE --help          # the options in short;  --full-help: every option with its default and caveats
```
Both binaries spell a shared option the same way (`--gpu`, `--exact`, `--sampled`, `--run-mode`, `--no-overlap`,
`--subsamples`, `--mpc-unit`, `--input-format`, ...), in lower-case kebab-case, with a `ps-` prefix exactly for what
needs phase-space streams, the Lagrangian tessellation or caustics. The older spellings are still accepted for one release (on the
command line and in a `--config` file) with one `warning: --old is deprecated, use --new` line and the same outputs;
deprecated: `--MpcUnit`, `--input 105`, `-m 2`, `--avg-subsamples`, `--exact-average`, `--ps-window`, `--ordered-merge`, `--NGP`, ...
(change_log.txt has the full table of the deprecated names).
The `--full-help` texts are the reference for every option; [demo/COMMANDS.md](demo/COMMANDS.md) collects typical
command lines. The most used: `--grid N [NY NZ]`, `--periodic`, `--field ...`, `--input-format 101|105|106|111|112|121|122`
(binary Gadget, Gadget HDF5 (default), its HI data, text `x y z w`, text positions, two raw binary layouts),
`--mpc-unit` (input length units per Mpc; default 1000, i.e. kpc), `--box`, `--region` / `--region-mpc` (a zoom),
`--padding` (default 5 mean particle spacings), `--partition X Y Z`, `--max-concurrent N`, `--verbose 0..3`,
`--config file` (options from a file).

**Outputs.** Each field is written as `<prefix>.<name>` (the cell-centre value) or `<prefix>.a_<name>` (the `_a`
cell average): raw float32 (float64 in a double build), N^3 cells in C order times the components, no header
(`--output-format 111` writes text). The density is rho/rho_bar.

| `--field` | file | components |
|---|---|---|
| `density`, `density_a` | `.den`, `.a_den` | 1 |
| `velocity` | `.vel` | 3 (the input's units; dtfelib returns peculiar km/s) |
| `gradient` / `divergence` / `shear` / `vorticity` | `.velGrad` / `.velDiv` / `.velShear` / `.velVort` | 9 / 1 / 5 / 3 |
| `tweb`, `vweb` | `.tweb` + `.twebEig`, `.velVweb` + `.velVwebEig` | class (0 void, 1 wall, 2 filament, 3 node) + 3 eigenvalues |
| `dispersion` (PS-DTFE) | `.velDisp` + `.velDispTensor` | trace + 6 (xx xy xz yy yz zz) |
| `scalar`, `scalarGradient` | `.scalar`, `.scalarGrad` | the scalar dataset and its gradient |
| every grid pass (PS-DTFE) | `.streams` / `.a_streams`, `.hidden_streams` / `.a_hidden_streams` | stream count; bitmask of streams the count misses |
| the exact deposit (PS-DTFE `--exact`, the default with an `_a` field) | `.tetTouch` | tetrahedra touching the cell (not a stream count) |
| `--face-gradient` (PS-DTFE) | `.a_velDivStreams`, `.a_massFluxDiv` | the stream average's divergence (the velocity-gradient family is then the mean flow's); the mass flux divergence div(Σ ρ_s v_s) |
| `--ps-caustics` | `.caustic`, `.causticClass` | fold flag; caustic bitmask (one file, no `a_` twin) |
| `--sample-points` | `.pts_den`, `.pts_vel`, `.pts_streams`, ... | one record per point (`.pts_den` float64 rho/rho_bar) |

The T-web needs a periodic box; both web classes take `--lambda-th` (binary default 0; the run scripts pass 0.3).
Standard DTFE also writes `*.badGridPoints*` files when the padding left cells outside the tessellation. While a run
goes it keeps `<prefix>.runlog.progress` (a JSON file: phase, partitions done, ETA, memory, GPU retries;
`DTFE_PROGRESS=0` switches it off). Every output is written as `<file>.part.<pid>` and the set is renamed into
place after the last one; if the output folder is missing at that point (an unmounted disk) the binary waits for it
(`DTFE_OUTPUT_WAIT` seconds, default 300), then writes to `$TMPDIR/dtfe-rescue/...`, prints the `mv` command and
exits with status 3.

**Interlacing, mass assignment.** `--interlace` grids the box a second time moved by half a cell and averages the two
density grids in Fourier space (any method, PS-DTFE included); without `--periodic` it needs a `--region` at least half
a cell inside the box. Only the density is interlaced. The CIC/TSC/PCS/NGP kernels get periodic images of their reach
and are clipped at a face or partition seam.

### Environment variables of the binaries

`./DTFE` and `./PS-DTFE` read these besides their options (the drivers' knobs are under *Running on TNG snapshots*, the
GPU knobs in [metal/README.md](metal/README.md), *Knobs*). One rule holds for all of them: an empty value is the same as
an unset one; a switch is off when set to `0`, `false`, `no` or `off` (in any case) and on when set to anything else;
a three-way one (on forces, off forbids) is automatic while unset or set to `auto`. A GPU knob has one name,
`*_GPU_*` (the `*_METAL_*` aliases of earlier releases are no longer read). One name per concept in both binaries
(`DTFE_` for both, `PS` inside a name only for a stream concept). A renamed variable's older name is still read for
one release when the new one is unset, with one `warning: OLD is deprecated, use NEW` line at start-up:

| older name (deprecated) | new name |
|---|---|
| `DTFE_PS_OVERLAP`, `DTFE_PS_CHECKPOINT` (deprecated) | `DTFE_OVERLAP`, `DTFE_CHECKPOINT` (both binaries) |
| `DTFE_NO_PROGRESS=1` (deprecated) | `DTFE_PROGRESS=0` |
| `DTFE_AUTO_PTS_VELGRAD`, `DTFE_TBB_THREADS` (deprecated) | `DTFE_AUTO_PTS_VEL_GRAD`, `DTFE_TRIANGULATION_THREADS` |
| `DTFE_TESS_CACHE_THREADS`, `_ZLIB`, `_CGAL_IO`, `_FINGERPRINT`, `_LEGACY` (deprecated) | `DTFE_TESSELLATION_CACHE_THREADS`, ... |
| `PS_MASS_AUDIT`, `PS_DEBUG_DEGREES` (deprecated) | `DTFE_MASS_AUDIT`, `DTFE_DEBUG_DEGREES` |
| `PS_GPU_TIMING`, `PS_GPU_FLOAT_MASS`, `PS_GPU_FLOAT_MOMENTS` (deprecated) | `DTFE_GPU_TIMING`, `DTFE_GPU_FLOAT_MASS`, `DTFE_GPU_FLOAT_MOMENTS` (metal/README.md) |
| `PS_EXACT_ITEM_CELLS`, `PS_EXACT_ITEMS` (deprecated) | `DTFE_GPU_EXACT_CELLS`, `DTFE_GPU_ITEMS` (metal/README.md) |


| variable | effect | default |
|---|---|---|
| `DTFE_RAM_GB` | the physical RAM the auto-tuner plans with, in GB (also skips its free-memory cap) | the machine's |
| `DTFE_FREE_GB` | the memory the other programs leave free, in GB, as the auto-tuner and the memory gates see it | the system's figure |
| `DTFE_MEM_FRACTION` | the auto-tuner's budget as a fraction of the RAM (0.1-0.95) | 0.60 |
| `DTFE_MEM_BUDGET_GB` | the budget outright, in GB (still capped by the free memory less 4 GB) | unset: the fraction |
| `DTFE_AUTO_MINN` | particles below which a run stays in one domain unless a split is predicted clearly faster | 2e6 |
| `DTFE_DYNAMIC_CONCURRENCY` | 0 = PS-DTFE runs the auto-tuner's number of partitions at once; by default, when the tuner chose it, more start while the live free memory and the budget hold them (the grids are the same); a mixed run (`--gpu-share` below 1) books the GPU lock holder's grids once, not per partition | 1 |
| `DTFE_AUTO_SPAN` | 0 = PS-DTFE's planner books a partition's sub-grid as G/n + 40 cells per axis instead of measuring each partition's Eulerian span on the particles | 1 |
| `DTFE_AUTO_T_VERTEX`, `DTFE_AUTO_T_SAMPLE`, `DTFE_AUTO_T_POINT` | the speed split's single-thread seconds per vertex, per sample, per point | 3.6e-6, 3.5e-7, 7.5e-7 |
| `DTFE_AUTO_PTS_N` | the number of points the auto-tuner budgets (a `--sample-points` file not written yet) | the file's count |
| `DTFE_AUTO_PTS_STREAMS` | the mean streams per point it budgets (at least 1) | estimated from the particles |
| `DTFE_AUTO_PTS_VEL_GRAD` | budgets the points' velocity gradients, as `--pts-vel-grad` does | off |
| `DTFE_AUTO_PTS_BOXMEAN` | estimates the streams per point over the whole box instead of the points' own slab | off |
| `DTFE_DEPOSIT_MEM_GB` | PS-DTFE CPU deposit: the memory its threads' private accumulator copies may take, in GB | 20% of the RAM, at least 2 |
| `DTFE_DEPOSIT_THREADS` | threads of one partition's CPU deposit (PS-DTFE) and of standard DTFE's `_a` average (`--exact` and the Monte-Carlo one); `1` = one thread | the cores left per partition; the single-precision PS-DTFE deposit grows into the cores other partitions free |
| `DTFE_DEPOSIT_SLABS` | PS-DTFE CPU deposit in slabs along x: 1 forces, 0 forbids (the GPU-deferred tetrahedra's too) | automatic |
| `DTFE_DEPOSIT_NSLABS` | the number of slabs (at most 64) | from the cost per plane |
| `DTFE_PTS_THREADS` | threads of the point evaluation | the cores left per partition |
| `DTFE_TRIANGULATION_THREADS` | both binaries: forces the parallel insert (as `--parallel-triangulation`) with this many threads (capped by `OMP_NUM_THREADS` and by the machine's threads divided among the tessellations built at once, not by a gentler `--run-mode`); `DTFE_TRIANGULATION_MIN_POINTS` moves the default rule's size threshold | unset (the rule: tessellations of at least 1000000 points, the performance cores, at most 6, divided among the tessellations built at once) |
| `DTFE_TESSELLATION_CACHE_THREADS` | threads per tessellation-cache load or save | up to 4 |
| `DTFE_OVERLAP` | the look-ahead of a partitioned run (both binaries): 1 forces it (on the CPU too), 0 forbids it | with a GPU deposit |
| `DTFE_MERGE_STAGE_GB` | PS-DTFE: a fixed cap on the partitions parked for the ordered merge, in GB (0: none parked) | the free-memory rule |
| `DTFE_MERGE_SPILL` | PS-DTFE: 1 spills every parked partition to disk, 0 none | the free-memory rule |
| `DTFE_SPILL_MIN_GB` | the free space a spill leaves on its disk, in GB | 16 in `$TMPDIR`, 2 in `--scratch-dir` |
| `DTFE_SCRATCH_MIN_GB` | the smallest allocation `--scratch-dir` backs with a file, in GB | 1 |
| `DTFE_CHECKPOINT` | `0` writes no partition checkpoints (both binaries) | on |
| `DTFE_PROGRESS` | `0` writes no `<prefix>.runlog.progress` | on |
| `DTFE_OUTPUT_WAIT` | seconds to wait for a missing output folder before the rescue copy | 300 |
| `DTFE_SERVE_INDEX` | `--serve`: 1 forces the cell index, 0 the walk (a composite server ignores 0) | automatic |
| `DTFE_SERVE_BUDGET_GB` | composite `--serve`: the memory for the resident partitions, in GB | the auto-tuner's budget |
| `DTFE_EXACT_AVERAGE` | both binaries: pins the `_a` cell average, `1` (on) as `--exact`, `0` (off) as `--sampled`; a flag on the command line wins, and the run log's header names the source | unset (or `auto`): no pin -- a sampling count or method selects the sampled average, else exact where it applies (Field evaluation methods) |
| `DTFE_RUN_MODE` | both binaries: the run mode, as `--run-mode` (`gentle`, `balanced` or `performance`, in any case; *Run mode* under *Memory, partitions and long runs*); an empty value or `auto` is unset, any other word an error, a flag on the command line wins, and the run log's `run mode` line names the source | unset: performance |
| `DTFE_GPU_SHARE_RHO` | PS-DTFE, for development: the constant of `--gpu-share auto`'s model in performance, 1/(1 + rho (threads - 1.8)) (a stronger GPU than an M1 Max's wants a smaller one) | 0.09 |
| `DTFE_MC_LEGACY` | standard DTFE `--average-method 1`: the uncentred CGAL-order Monte-Carlo realization (A/B) | off |
| `DTFE_MASS_AUDIT` | PS-DTFE: 1 prints each pass's mass bookkeeping, 2 adds the per-particle closure (one domain) | 0 |
| `TMPDIR` | the rescue folder and the merge spills without `--scratch-dir` | `/tmp` |
| `NO_COLOR` | no colours, for any non-empty value (the no-color.org convention, so `0` too) | colours on a terminal only |
| `CLICOLOR_FORCE` | a switch: colours also through a pipe | off |

For testing and debugging only: `DTFE_TEST_CHECKPOINT_FREE_BYTES` (the free bytes the checkpoint's room rule sees),
`DTFE_TEST_EXIT_AFTER_PARTITIONS=k` (exits after the k-th merged partition: a killed run to resume),
`DTFE_TEST_DELAY_PARTITION=i:s` (partition i starts s seconds late, which forces the staging),
`DTFE_TEST_PARK_ROOM_GB` (the RAM parking's budget room), `DTFE_TEST_FXMERGE_SHIFT=n` (the `--ps-merge fixed-point`
units n bits finer) and `DTFE_TEST_FXMERGE_SUM_LIMIT=n` (its sums capped at 2^n), for the overflow checks,
`DTFE_TEST_NAN_SHARE=k` (every k-th share NaN: the exact sums' guard), `DTFE_TEST_PLAN_MC=k` (stands for an
auto-tuned plan of k partitions at once: the dynamic concurrency's floor), `DTFE_AUTO_TUNE_DEBUG` (prints the
standard auto-tuner's busiest partition per thread, PS-DTFE's partition sub-grids, measured, booked and real, and the
terms of its dynamic concurrency's start check), `DTFE_DEBUG_DEGREES` (checks the lazy vertex degrees after a GPU
deposit against a full pass), `DTFE_DEBUG_FX_BOUNDS` (PS-DTFE, single precision: checks the bounds a `--face-gradient`
pass takes over from the GPU extraction against a bound walk's), `DTFE_PTS_FULL_RECORDS` (keeps every point record
member, unpacked: the packing test),
`DTFE_TESSELLATION_CACHE_FINGERPRINT` (prints a loaded tessellation's structure fingerprint),
`DTFE_TESSELLATION_CACHE_LEGACY` (writes the single-stream cache format), `DTFE_TESSELLATION_CACHE_ZLIB` (zlib instead
of libdeflate) and `DTFE_TESSELLATION_CACHE_CGAL_IO`
(CGAL's own tessellation reader and writer), the last two for comparisons. The GPU's debugging knobs, among them the
timing experiments `DTFE_GPU_DEBUG_NOATOMIC` and `PS_GPU_DEBUG_NOATOMIC` (no grid atomics: the grids come out wrong),
are with the other GPU knobs in metal/README.md, *Knobs*.


## Phase-Space DTFE

Standard DTFE builds the Delaunay tessellation in Eulerian (present-day) space and returns a single-valued density at
each point. **Phase-Space DTFE** builds it in Lagrangian (initial-condition) space and follows it as it is deformed to
the present day. The deformed tessellation folds over onto itself, so the method recovers the *multi-stream*
structure of the cosmic web: the density at a point is the sum over every stream (folded tetrahedron) covering it, and
the number of streams is a field of its own. It follows the phase-space tessellation approach of Abel, Hahn & Kaehler
(2012) and Shandarin, Habib & Heitmann (2012).

It needs **two positions per particle**, the Eulerian and the Lagrangian one, from a Gadget-HDF5 file: an
`InitialCoordinates` dataset in the same file, or a separate initial-conditions snapshot (`--ps-lagrangian-input`):
```bash
make PS-DTFE METAL=1     # METAL=1 on Apple Silicon for --gpu
./PS-DTFE snapshot.hdf5 output_root --grid 256 --periodic --field density_a --ps-lagrangian-input ics.hdf5
```

The `_a` fields come from the exact deposit by default (`--exact`); `--sampled` (or a `--subsamples` count) averages
over an nSub^3 sub-grid instead (`--subsamples`, default 3), at a cost that grows as nSub^3.
The options that shape the result (the drivers' production settings are marked):

| option | what it does |
|---|---|
| `--gpu` | the deposit on the GPU (needs a GPU build; otherwise ignored with a warning); every grid field, the sampled and the exact deposit |
| `--ps-vertex-mass` (production) | each particle's mass shared over its tetrahedra: chart-independent masses, removing the bias rho_bar V_lag carries when TNG's z = 127 positions are the Lagrangian input |
| `--ps-volume-weighted` (production) | velocity moments weighted by Eulerian volume, the standard-DTFE convention; the dispersion stays mass-weighted |
| `--ps-caustics` (production) | the fold flag `.caustic` and the bitmask `.causticClass` |
| `--ps-caustic-cusps` | with `--ps-caustics`: the cusp (bit 7) and swallowtail (bit 8) indicators; about 1.8x the caustic pass |
| `--ps-halo-release D` | tetrahedra with a stream density above D (rho/rho_bar) are deposited whole at their centroid cell: the mixed halo interiors (Stücker et al. 2021) |
| `--exact` (production: the exact deposit, the default) or `--sampled`, `--ps-linear-deposit`, `--grid-window` | see Field evaluation methods |
| `--ps-alpha-shape` | non-periodic clouds: keep tetrahedra with Lagrangian circumradius up to this many mean spacings (default 3; 0 = the convex hull) |
| `--ps-lagrangian-padding` | the padding of each Lagrangian partition in mean spacings (default 8; 0 = the Eulerian `--padding` rule, under which a TNG partition triangulates 7-9x the particles it owns) |
| `--parallel-triangulation` | force the parallel Delaunay insert (TBB builds) for every tessellation; the default rule applies it to tessellations of at least a million points where more than one core is free (3.1x on the triangulation of TNG100 at 512^3 with one partition in flight; nothing at the auto plan's concurrency). Two identical runs then differ at rounding in what is summed in walk order (CPU float sums, the face pass's last bit, PS-DTFE's velocity moments); `--sequential-triangulation` keeps the bit-reproducible insert |

**Streams and hidden streams.** The stream count is measured at the sub-samples of each cell, so it cannot see a fold
thinner than the sampling. `.hidden_streams` bit 1 (value 1) marks exactly the cells a folded tetrahedron overlaps
while the count misses it; bit 2 marks cells that received the mass of a tetrahedron too small to hold a sample (its
mass goes to the cell of its centroid, so mass is conserved). A single-stream mask needs both: `.streams` equal to 1
and bit 1 clear (`FieldSet.velocity_single_stream()` / `single_stream_mask()` in dtfelib). Which tetrahedra contain a
sample is decided exactly, so a `--partition` split gives the same counts as one run and the GPU the same as the CPU.

**Caustics.** `.caustic` is geometric: a cell overlapped by tetrahedra of both orientations, equal to `.streams` > 1
or `.hidden_streams` bit 1, independent of `--subsamples`, the deposit, the split and the GPU. The other bits of
`.causticClass` (the collapse multiplicity k in bits 2-5: 1 wall, 2 filament, 3 node; the umbilic indicator in bit 6;
the cusp and swallowtail indicators in bits 7-8) are recorded at the deposited samples and depend on the sampling.
Grids whose `.caustic` holds the older per-sample flag (it marked 21.0% of the cells where folded tetrahedra overlap
33.6%, TNG100-3-Dark z = 1) need no rerun: `FieldSet.fold_mask()` rebuilds the exact flag from `.streams` and
`.hidden_streams`, and the caustic scripts and the launcher use it. The cusp and swallowtail bits are indicators
(an exact swallowtail lights 74.5% of the cusp cells, a pure cusp 0.5%), not classifications; decode them with
`dtfelib.io.caustic_collapse_max`, `caustic_is_fold`, `caustic_is_cusp`, `caustic_is_swallowtail`.

**Velocity gradients in multi-stream cells.** PS-DTFE's gradient, and the divergence, shear, vorticity and V-web
built on it, is ⟨∇v⟩, the average over the streams of each stream's own gradient; standard DTFE's is ∇⟨v⟩, the
gradient of the mean velocity. They agree in single-stream cells and differ in folds (an inverted stream inside a wall
diverges while the mean flow converges), so the box mean of the PS divergence is not zero and the PS V-web finds few
filaments and nodes. `FieldSet.load(name, mean_field=True)` returns ∇⟨v⟩ for a PS grid (FFT derivatives of `.a_vel`,
with the binary's V-web normalisation and the run's `--lambda-th`); plot_velDiv_den.py, plot_cosmic_web.py,
plot_web_streams.py, plot_shear_triaxial.py, plot_DTFE.py, plot_halo_tracking.py and the launcher's Explore tab use it
for PS grids by default (`--ps-gradient stored` restores the stored grids). At 512^3 the whole-cube computation takes
3-7.5 GB; `fields.mean_field_plane` computes one slice in ~100 MB. A plane's `.pts_velGrad` stays ⟨∇v⟩.

**T-web at the grid scale.** `tweb` / `tweb_a` classify the tidal tensor of the raw density grid, with no smoothing, so
the classes follow each estimator's cell-scale noise (TNG100-3-Dark z = 1: 37.6% void with standard DTFE, 33.8% with
PS-DTFE; 46.8% for both after a one-cell Gaussian). Quote web fractions at a stated smoothing, recomputed from `.a_den`.

**Periodic seam.** A periodic run tiles one Lagrangian period exactly (`tests/ps_periodic_seam_check.sh`). Grids made
before that fix deposit some tetrahedra twice or not at all within a few cells of the box faces (up to 394 cells per
TNG snapshot, a box-mass offset of 1-2e-7); a run whose padding exceeds about a quarter of the box keeps that seam and
says so.

**Boundaries and regions.** A `--box` region cut from a larger cloud keeps exactly the mass inside it (a tetrahedron
crossing the face deposits only its inside share). The flatness cut of degenerate tetrahedra is relative to each
tetrahedron's size, so it does not depend on the length unit.

**CPU deposit threads.** In the default single-precision build every accumulator is a shared integer sum (or a count,
flag or mask), so the threads claim blocks of cells from one counter and the team grows as cores come free (a finished
partition's), up to every core: the grids are byte-identical to one thread. The double build and the float sums
(`DTFE_GPU_FLOAT_MOMENTS=1`, `DTFE_GPU_FLOAT_MASS=1`) keep a fixed team per partition, each thread summing into its own copy
of the accumulators; when those copies would not fit (a fifth of the memory) the deposit switches to slabs along x,
bit-identical to one thread at any thread count (`DTFE_DEPOSIT_SLABS=1` forces, `0` forbids it). `DTFE_DEPOSIT_THREADS`
pins the count. `DTFE_MASS_AUDIT=1` prints each partition's mass balance.

**Phase-space density.** Standard DTFE's `--approx-phase-space-density` approximates the distribution function as f ≈ ρ(x)·g(v), a
second tessellation in velocity space, which breaks down where streams cross. PS-DTFE resolves the streams: the exact
distribution function is the per-stream decomposition of `--sample-points --ps-per-stream`, and Q = ρ/σ³ follows from its
density and dispersion grids (`dtfelib.fields.pseudo_phase_space_density`).


## Field evaluation methods

A cell's value is either the field at the cell centre (`density`, ...) or its average over the cell (`density_a`, ...).

**Exact or sampled.** Since 2026-11-09 both binaries integrate the `_a` averages exactly by default (`--exact`: the
exact average of standard DTFE, the exact deposit of PS-DTFE, below). `--sampled` selects the sampled average
(standard DTFE: the Monte Carlo average, with `--subsamples` and `--average-method`; PS-DTFE: the sampled deposit on
nSub^3 sub-points), and so does a sampling count given without `--exact` and without a pin (`--subsamples`; in
standard DTFE also `--average-method`, on the command line or in a `--config` file). The two flags exclude each other;
a count beside `--exact` is accepted and ignored. `DTFE_EXACT_AVERAGE=1` or `0` pins the choice for a run that gives
neither flag (a count does not override it).
Where the exact form does not apply -- `velocityStd_a`, a redshift cone grid, a `MY_SCALAR` build, the other
interpolation methods (`--ngp`, `--cic`, `--tsc`, `--pcs`, `--sph`, `--voronoi`) -- standard DTFE samples by itself;
an explicit `--exact` there is refused with an error (a redshift cone grid warns instead, and samples). A PS-DTFE run
without an `_a` field runs as before unless `--exact` is given. The run log's header says which average ran and why
(standard DTFE's `volume averaging` line, PS-DTFE's `grid deposit` line: `EXACT ...` or `SAMPLED (...)`).

* **Monte Carlo average** (standard DTFE `--sampled`, with `--average-method 1`, the sampled average's default;
  `--subsamples k`, k^3 points per cell, default an average of 100 per cell). A tetrahedron is sampled at the first n points of a shared Sobol table, centred on its centroid and in a
  vertex order hashed from its positions, so the average is unbiased (the demo's box mean 1.00002) and the
  realization does not depend on `--partition`, the thread count or the padding (cells agree to 4-6e-6). `DTFE_MC_LEGACY=1` selects the
  uncentred CGAL-order average, which loses 0.17-0.29% of the mass at z <= 2. `--seed` seeds method 2 (`--average-method 2`, random
  points per cell; the only standard-DTFE path that computes `velocityStd_a`, which every other method and an
  explicit `--exact` reject with an error), `--particle-fraction` and `--poisson`.
* **Exact average** (`--exact`, standard DTFE; the default since 2026-11-09) integrates the linear interpolant over
  every cell/tetrahedron intersection: no sampling noise, box mean 1 to 3e-8 on TNG100-3-Dark. The CPU clips with the
  vendored [r3d](third_party/r3d/README.md) library (Powell & Abel 2015), each tetrahedron once per slab and column of
  its cells; with `--gpu` the GPU integrates the same intersections by a z-sweep with exact Gauss-Legendre quadrature
  ([metal/README.md](metal/README.md)), equal to float rounding. On TNG100-3-Dark z = 0 at 512^3 (Apple M1 Max) a run
  takes 439 s on the GPU against 433 s with the sampled average, and 431 s on the CPU against 315 s. The drivers pass
  it unless `--sampled`, `DTFE_EXACT_AVERAGE=0` or a `SUBSAMPLES` count asks for the Monte Carlo average.
* **Face gradient** (`--face-gradient`, both binaries, 3D): the velocity gradient and the divergence, shear,
  vorticity and V-web built on it from EXACT FACE INTEGRALS of the interpolant (Gauss: the cell average of grad v is the
  integral of v n over the cell's faces divided by its volume). Standard DTFE: THE DEFAULT since 2026-11-09 -- the
  per-tetrahedron gradient's exact cell average, formed in double and the same for every deposit and device (the
  deposit's own gradient, float32 on the GPU, differs from it by 5e-7 of the gradient rms per sub-domain at 512^3);
  `--per-tet-gradient` (or `DTFE_FACE_GRADIENT=0`) keeps the deposit's own gradient, the route of grids made before
  that date and the 2D build's only one. PS-DTFE (with the
  exact deposit, on the CPU or -- since 2026-11-09, the face sums then from a CPU walk beside the device's deposit --
  the GPU, and `--ps-volume-weighted`): the gradient of the volume-weighted MEAN velocity, the quantity
  standard DTFE stores, as the difference of the face-averaged mean velocities -- conservative (a periodic box's mean
  divergence is 0), with the caustics' jumps as exact face differences -- where the default stores the stream average of
  each stream's own gradient (written beside it as `.a_velDivStreams`). The same face sums give the mass flux
  divergence div(Σ ρ_s v_s) (`.a_massFluxDiv`, ρ/ρ̄ times velocity per length): by continuity −a d(1+δ)/dt, the rate
  a cell gains or loses mass, caustic sheets included. dtfelib reads these grids for its mean-field quantities instead
  of differentiating `.a_vel` by FFT.
* **Sampled deposit** (PS-DTFE `--sampled`, or a `--subsamples` count; the default before 2026-11-09): each stream's
  mass is spread over the sub-samples inside it. `--ps-linear-deposit` weights the sub-samples by the linear density
  profile instead (still mass-conserving; not with `--ps-volume-weighted`).
* **Exact deposit** (`--exact`, PS-DTFE; the default since 2026-11-09 for a run with an `_a` field): each stream is
  integrated analytically over every grid cell (CPU: r3d clipping; GPU: the z-sweep). On TNG100-3-Dark z = 0 at
  512^3 (Apple M1 Max, a sample-point plane evaluated in the same run) it takes 1133 s on the CPU and 1156 s on the
  GPU, against 1260 s for the sampled deposit on the GPU and 1644 s on the CPU. The drivers' `DTFE_GPU=auto` and the
  launcher run it on the GPU of a GPU build, in performance together with the free cores (`--gpu-share auto`);
  `DTFE_GPU=0` keeps the CPU. The GPU hands the tetrahedra whose dispersion
  float32 cannot resolve to the CPU's double deposit (20.6M of them on TNG100-3-Dark z = 0 at 512^3).
* **Window** (`--grid-window x0 x1 y0 y1 z0 z1`, PS-DTFE, Mpc): only the cells of the full `--grid` inside that box are
  deposited and written, equal to the full run's cells (counts and masks bit for bit, sums to the last fixed-point
  unit). The cost follows the window, not the grid: the launcher's exact zoom is a window of a fine virtual grid. With
  `--partition` and `--tessellation-cache` it loads the cached partitions, skips those whose streams never reach the
  window, and reads no snapshot when every partition is cached. Not with the T-web/V-web, `--region`, `--partition-index` or
  `--ps-linear-deposit`.

### Point evaluation

Both binaries evaluate their fields at arbitrary points (`--sample-points FILE`: text `x y z` lines or raw float64
triples), one record per point: `.pts_den`, `.pts_vel`, `.pts_streams` (PS-DTFE: the stream count; standard DTFE: 0/1
coverage), and with options `.pts_velGrad` (`--pts-vel-grad`), the density gradient (`--pts-den-grad`), the caustic bits
`.pts_caustic` (`--ps-caustics`; the bits of `.causticClass`), and in PS-DTFE the per-stream records (`--ps-per-stream`,
`--ps-per-stream-ids`), each stream's Lagrangian origin (`--ps-per-stream-lagrangian`) and a per-particle dataset along it
(`--pts-scalar --scalar-dataset NAME`). Point location
is exact: a point on a shared face, edge or vertex is counted once per stream, also across the periodic boundary and
under any `--partition` split (each partition answers the points in the tetrahedra whose centroid it owns, so the
values equal one triangulation's to rounding). `--ps-stream-density linear|geometric` picks the per-stream estimator (`dtfe` is the older name of `linear`).
Point evaluation runs on the CPU (double precision, all cores; `DTFE_PTS_THREADS`).

**Image planes follow the grids' conventions** (`--ps-point-conventions grid`, the default): a PS plane takes the run's
`--ps-vertex-mass` and `--ps-volume-weighted`, so its density contrast matches the grid's (slope 0.97-0.99 against the
exact-deposit grid on a TNG-like synthetic box). `--ps-point-conventions legacy` keeps the reference implementation's
estimators; planes made with them hold 0.81-0.82 of the grid's density contrast at z = 20 and the launcher's Runs tab
marks them. The T-web and V-web of a plane are sampled from the grid at the points
(`dtfelib.io.PointPlane.sample_grid_field`).

### Interactive point evaluation

```python
from dtfelib import Estimator

with Estimator("snapshot.hdf5") as est:       # PS-DTFE; the tessellation is built here, once
    f = est(points)                           # (N, 3) points in Mpc
    f.density, f.velocity, f.streams, f.dispersion
    cube = est.grid(xs, ys, zs)               # the fields on the grid xs x ys x zs
est = Estimator.from_arrays(x, velocities=v, lagrangian=q, box=100.0)   # particles from numpy arrays
```
Behind it the binary runs with `--serve` (a binary protocol on stdin/stdout, documented at the top of
`src/CGAL_triangulation/ps_serve.cc`). Every answer is bit-identical to `--sample-points` for the same points;
`phase_space=False` gives the standard estimator, padded with 15 spacings (`Estimator(padding=...)`). Velocities come
back in peculiar km/s. With `tessellation_cache=dir` a later session starts from the cached tessellation. A box too
large for one tessellation is served as a composite of partitions (`partition=4`, or chosen automatically with a cache
directory when one tessellation would not fit): built once in parallel into the cache, only the partitions a request
reaches are visited, and at most `resident` (`--serve-resident`; default: as many as the memory budget holds) are kept
in memory. Answers equal `--sample-points` with the same split bit for bit in PS-DTFE, to rounding in standard DTFE.
`--serve-progress` writes progress lines for the launcher.


## Memory, partitions and long runs

**Auto-tuning.** Without `--partition` / `--max-concurrent` the binary chooses both from the particle count, the
grid, the fields and the machine, and prints the split and the predicted peak. The budget is 60% of the RAM
(`DTFE_MEM_FRACTION`), never more than the other programs leave free less 4 GB; `DTFE_MEM_BUDGET_GB` sets it outright
(still capped by the free memory). Explicit options always win; `--partition 1` keeps one tessellation. A plan given
in full (both options) is still predicted: the line states it, and the memory gate and the RAM a finished PS-DTFE
partition may park in come from it, as for a tuned plan. Below 2e6
particles (`DTFE_AUTO_MINN`) a run stays in one domain unless a split into 8 or 27 partitions is predicted to be
clearly faster; such a split merges in a fixed order (reproducible), but differs from one domain at float rounding.
`--auto-tune-report` prints the decision and the prediction as one line and stops (the drivers' `AUTO_TUNE_REPORT=1`,
the launcher's Check memory). A run with a prediction ends its SUMMARY with the predicted against the measured peak.
- In PS-DTFE `--max-concurrent` is the number of partitions in flight, each writing into the whole grid; while
  several run, a partition waits until the free memory would hold it.
- In standard DTFE the partitions run one after another and `--max-concurrent` is the number of threads in each, every
  thread triangulating its own padded sub-box. The plan replays the split on a count grid of the particles (19-90%
  above the measured peaks of 38 TNG runs). A `--region` run is planned as the region.
- `--scratch-dir DIR` backs every allocation of 1 GB or more (the full grids, the T-web's FFT arrays) with files on a
  local disk, trading disk for RAM with unchanged results. Grids beyond 2^32 cells work (the GPU splits until every
  sub-grid holds fewer).
- A point evaluation's memory grows with the stream count at its points, which the tuner estimates from the plane's
  own slab (42 B per stream at least, 114 with `--pts-vel-grad`).

**Run mode.** `--run-mode gentle|balanced|performance` (both binaries, every build; `DTFE_RUN_MODE`, in any case, is
the same choice, and the flag wins) sets how hard a run leans on the machine: `performance` takes every core and long
GPU buffers, `balanced` most of the performance cores and short buffers, so the display never waits on the GPU, and
`gentle` half of them, so the machine stays usable beside the run. The binaries default to `performance`, the
behaviour of earlier releases; the drivers and the launcher default to `balanced` (`performance` inside a SLURM or PBS
job and in the launcher's SLURM scripts) and pass the mode on every run. A mode fills in only what was not given: a
non-empty `OMP_NUM_THREADS` keeps its thread count, the `DTFE_*_THREADS` pins keep their teams, a `--max-concurrent`
above the mode's threads raises them to it, and `--partition`, `DTFE_GPU` and `--gpu-share` stay as given; the memory
budget, the estimator, the fields and the merge are not the mode's. The run log's header line `run mode : <mode>
(<source>; <threads>[; GPU buffers up to 250|50 ms][; nice 10])` says what was set and why. The mode changes no
PS-DTFE single-precision byte; standard DTFE's grids follow the thread count at float rounding, as with
`OMP_NUM_THREADS`. The checkpoint's settings key leaves the mode out, so a run killed in one mode resumes in another.

| what the mode sets | `gentle` | `balanced` | `performance` |
|---|---|---|---|
| threads, from the base: the performance cores on Apple silicon, else every core | half the base, at least 1 | the base less a quarter of it, holding back 1 to 4 | every core (the OpenMP default) |
| threads on 2 cores / 8 cores / an M1 Max (8 + 2) / 64 cores | 1 / 4 / 4 / 32 | 1 / 6 / 6 / 60 | 2 / 8 / 10 / 64 |
| priority | nice 10 on Linux (never lowered); unchanged on macOS | unchanged | unchanged |
| GPU command buffers (8 ms gaps in every mode) | ~50 ms | ~50 ms | ~250 ms |
| PS-DTFE exact deposit: items per GPU buffer | 500k | 500k | 4M |
| `--gpu-share auto` (PS-DTFE) | 1 | 1 | from the threads: 0.6 at 10, 0.65 at 8, 0.85 at 4 |
| the device the drivers and the launcher pick on a GPU build | the GPU | the GPU | the GPU; standard DTFE's sampled average the CPU |

The items per buffer change no byte. `--gpu-share auto|<fraction>` (PS-DTFE, the exact deposit of a partitioned
`--gpu` run) deposits that share of the partitions on the GPU and the others on the free CPU cores at the same time,
chosen by the partition index alone, so a rerun repeats it. The binary's default is 1, every partition on the GPU;
`auto` is the run mode's share: in performance 1/(1 + 0.09 (threads - 1.8)) in steps of 0.05, between 0.5 and 1
(`DTFE_GPU_SHARE_RHO` replaces the 0.09), and 1 in the double build, with `DTFE_GPU_FLOAT_MOMENTS`,
`DTFE_GPU_FLOAT_MASS` or `DTFE_MASS_AUDIT`, below `DTFE_AUTO_MINN` particles and under 4 threads; 1 in gentle and
balanced. The drivers (`GPU_SHARE`, default `auto`) and the launcher pass `--gpu-share auto` on every PS exact run on
the GPU, so their performance runs use both devices. A mixed run's grids differ from an all-GPU or an all-CPU run's at
float rounding, and the mass stays exact. The resolved share is in the checkpoint key (a `gpu-share=` line, only below
1), and `--resume` with `auto` keeps the checkpoint's. A number is ignored, with a note, without `--gpu`, for the
sampled deposit, in a CPU-only build and for one domain. The run log's `PS-DTFE: GPU share <share> (<source>): ...`
line says how many partitions deposit on each device, and a mixed plan's AUTO-TUNE line ends `GPU deposit with the
free cores (GPU share <share>)`.

Measured on TNG100-3-Dark z = 0 at 512³ (M1 Max, PS-DTFE's exact deposit): 1133 s on the CPU and 1156 s on the GPU at
500k items per buffer (the bench of `2f428bc`); on 2026-11-09, with `2f428bc` plus the selection skip and the core lend,
1025 s on the GPU at 4M items per buffer (performance's buffers) with the tuner's plan, and 712 s at a GPU share of 0.6
with `--partition 6 6 6 --max-concurrent 6` given by hand. The gaps between buffers take 18.6% of the time a run holds
the GPU at 50 ms buffers and 4.4% at 250 ms by the model (17.4% measured at 53 ms). A probe of the display path was
late by more than a 60 Hz frame in 0.0% of its samples at 53 ms buffers, and in 5.6% (up to 214 ms) at 155 ms.

The modes themselves were timed once each on that input in the night of 2026-11-10, while a screen saver drew all
night (with no run the GPU was 59% and the efficiency cores 67% busy, against 28% and 50% the evening before: the
binary and command of the 712 s run took 789 s that night, so these times compare with each other and not with the
ones above):

| run | `gentle` | `balanced` | `performance` |
|---|---|---|---|
| PS-DTFE, exact deposit, `--gpu --gpu-share auto` | 1408 s | 1444 s | 908 s on the tuner's plan (4 partitions at once, a fifth added at run time); 783 s with `--partition 6 6 6 --max-concurrent 6` given by hand |
| standard DTFE, exact average, face gradient, `--gpu` | 444 s | 447 s | 486 s |

PS-DTFE's gentle and balanced runs kept 1.4 cores busy on average and wrote the bytes of the `2f428bc` bench's GPU
run; performance kept 6 cores busy. The tuner's memory model books more than a mixed run takes (4 at once and a fifth
added at run time: predicted 41.3 GB, peak 29.8 GB; 6 at once by hand: predicted 44.7 GB, peak 31.4 GB), so
performance's own plan runs fewer partitions at once than the memory holds. Standard DTFE's 10-thread plan had no
memory left for the look-ahead on this input (the next partition triangulated during the GPU's work; so before the
modes too), which the 6- and 4-thread plans had: balanced and gentle were the faster runs here, at peaks of 41.3 and
40.7 GB against 33.1 GB. Not measured: nice 10, a CPU run at a lowered thread budget, and any mode on a machine at
rest.

**Padding.** Every partition (and in standard DTFE every thread's sub-box) carries its own padding. The binary's
default of 5 mean spacings is too short for the emptiest voids at z = 0: `run_dtfe.sh`, dtfelib's standard server and
the launcher pass 15, complete at z = 0 on TNG50/100/300-3-Dark (10 is not). An incomplete padding leaves
`*.badGridPoints*` files, never silently wrong cells (point values within a few spacings of a face can be off, though:
up to 4% at padding 5). PS-DTFE partitions are padded in Lagrangian space by `--ps-lagrangian-padding` (default 8).

**Partition merge (PS-DTFE).** By default the partitions' grids are added in partition index order whatever order they
finish in (`--ps-merge index-order`, the staged merge), so two identical runs give bit-identical grids. The partitions also
start in index order, so only the few in flight beside a slower one finish before their turn (TNG100-3-Dark z = 1 at
5^3: at most 5 at once, nothing spilled). A partition that
finishes early parks its sub-grids in memory, or, when the memory is short, spills them to a file in `--scratch-dir`
(else `$TMPDIR`) with identical results (`DTFE_MERGE_SPILL=0|1`, `DTFE_MERGE_STAGE_GB`, `DTFE_SPILL_MIN_GB`). The run
log's `ordered merge:` line reports it; its `merge time:` line (standard DTFE prints one too) gives the seconds of
the merge itself, which runs one partition at a time, and of its checkpoints, and how long the partitions waited for
their turn, parked or holding a thread. `--ps-merge completion-order` adds them as they finish (not bit-reproducible).
`--ps-merge fixed-point` (opt-in) adds each partition as it finishes into exact 72-bit integer grids, 9 B per cell
instead of 4: order-free, nothing parked, about 1e-7 of each grid's peak from the ordered merge; not in the double
build nor with `--grid-window`, and an overflow stops the run. A partition the GPU gives up on is deposited by the CPU,
which rounds differently, so such a run is reproducible only where the fallback recurs.

**Resume.** A partitioned run of either binary saves its merged grids and the list of finished partitions in
`<prefix>.runlog.checkpoint` (or in `--scratch-dir`). After a crash, `--resume` computes only the missing partitions,
and refuses when the build, the inputs, the split or the command line changed (`--verbose`, `--max-concurrent`,
`--run-mode`, `--scratch-dir`, `--tessellation-cache` and the overlap and merge-order flags may change, `--gpu-share`
(PS-DTFE) while the share it resolves to stays the same: *Run mode* above). A resumed PS run merged in
index order, and a resumed standard CPU or `--gpu` run at the same thread count, are byte-identical to an uninterrupted
one; `--sample-points` runs resume too. The drivers resume by themselves (`DTFE_RESUME=0` starts over, in both);
`DTFE_CHECKPOINT=0` writes no checkpoints; `--interlace` runs are not checkpointed. Two limits: a standard run that
draws random numbers (`--average-method 2`, `--particle-fraction`) without `--seed` is seeded from the clock, and the
seed is not in the checkpoint's settings key, so its resumed run cannot match any one uninterrupted run byte for byte
(give `--seed` for that); and the key holds each option as given, so a PS run that passed `--ps-stream-density linear`
(or the deprecated `dtfe`) explicitly does not resume under a command that leaves the option out, though both mean
the default: give it the same way both times. The old spellings of an option resume under the new ones (one key).
The key also holds the average and the gradient route the run resolved to (`average=exact|sampled`,
`gradient=face|per-tet`), so an environment pin (`DTFE_EXACT_AVERAGE`, `DTFE_FACE_GRADIENT`) is part of it; a
checkpoint written before these two lines existed (2026-11-09) does not match any more.

**Look-ahead.** A partitioned standard `--gpu` run triangulates the next partition while the GPU works on the current
one (`--no-overlap` switches it off); PS-DTFE does the same at `--max-concurrent 1` only (`--no-overlap` too). The
outputs are byte-identical either way.

**GPU failures.** The GPU deposit runs as short paced command buffers with a snapshot of the
grids every ~10 s of GPU work (`DTFE_GPU_CHECKPOINT_CHUNKS`). A killed buffer (the display watchdog, another program's
GPU hang, out of memory) restores the last snapshot and retries by cause; a watchdog or memory error slows the pace,
which eases back after a clean stretch (`DTFE_GPU_PACE_RECOVER_SEC`) to the run mode's base: ~250 ms buffers in
performance, ~50 ms in gentle and balanced, 8 ms gaps in every mode (the `pace eased` line then ends `(the default)`
or `(the run mode's)`). Three failures of one chunk, sixteen in a deposit
or a lost device send that partition (standard DTFE: that sub-domain) to the CPU. The run log ends with
`GPU deposit retries: N (innocent a, timeout b, oom c, other d)`. `DTFE_GPU_FAULT_AT=<k>[:innocent|timeout|oom|persist]`
is the test hook. Standard DTFE's GPU `_a` grids are exact 64-bit fixed-point sums, so two `--gpu` runs are
byte-identical (`DTFE_GPU_FLOAT_GRIDS=1`: float atomics, for comparison).

**Tessellation cache.** Building the tessellation and its vertex densities is the dominant fixed cost (81% of a run on
884736 particles). `--tessellation-cache DIR` (a local, non-synced folder) writes it once and later runs on the same
inputs load it, whatever their grid, fields or sample points; any change that would alter it (input files, box,
region, padding, periodicity, partition, build) is a miss and a rebuild, never a wrong result (`--verbose 3` says
why). The files are compressed in 16 MB chunks (about 0.15 KB per vertex). Partitioned runs cache every tessellation
they build: a PS partition per file, a standard run one file per thread sub-box (one per partition under
`--sample-points`; its split follows the `--grid`). `python/tools/upgrade_tess_cache.py <folder>` converts files of
the older single-stream format.


## Standard DTFE and PS-DTFE compared

| | standard DTFE (`./DTFE`) | PS-DTFE (`./PS-DTFE`) |
|---|---|---|
| Density | DTFE density at each particle, interpolated linearly | sum over the streams of tetrahedron mass over Eulerian volume |
| Velocity | volume average of the single-valued interpolant | mean over the streams (by volume under `--ps-volume-weighted`); dispersion by mass |
| Velocity derivatives | ∇⟨v⟩ | ⟨∇v⟩ (dtfelib's `mean_field=True` gives ∇⟨v⟩) |
| Cell average | exact (`--exact`, the default) or Monte Carlo (`--sampled`) | the exact deposit (`--exact`, the default) or sampled on nSub^3 (`--sampled`, nSub = 3) |
| GPU | `--gpu`: the `_a` averages | `--gpu`: every grid field, sampled and exact |
| Run-to-run identical | yes (exact GPU sums; CPU at a fixed thread count) | yes (exact sums, index-order merge) |
| `--max-concurrent` | threads per partition | partitions in flight |
| Padding | 15 spacings (drivers, dtfelib, launcher) | `--ps-lagrangian-padding 8` |
| Points | the interpolant; `.pts_streams` = 0/1 coverage | every stream (`--ps-per-stream`) |
| Streams, dispersion, caustics | none | yes |

In single-stream void interiors the standard density is 8-14% above PS-DTFE's: its tetrahedra bridge wall and void
particles.


## Running on TNG snapshots

The workflow has three steps, run from the repository root and driven by `DTFE_DATA_ROOT` (the folder that holds the
simulations) and `DTFE_SIM` (which one). Neither has a default: `DTFE_DATA_ROOT` must be set (e.g.
`export DTFE_DATA_ROOT=~/IllustrisTNG`; without it the drivers stop with "no data root"), and every run names the
simulation, with `-s` (the drivers), `--sim` (the plot scripts), `SIMS` (`run_ps_pipeline.sh`, `SIMS=all` for every
simulation on disk) or `DTFE_SIM` (without one a script stops with "no simulation given"). `scripts/config.sh` holds
the defaults and resolves either data layout: per family (`$DTFE_DATA_ROOT/TNG100/TNG100-3-Dark/snapdir_099/`, the
family being the name up to its first `-`) or flat (`$DTFE_DATA_ROOT/TNG100-3-Dark/snapdir_099/`); `dtfelib` and
`python/config.py` read the same variables. Each snapshot is one folder `snapdir_NNN` holding its merged, h-free
`combined_NNN.hdf5` and, after a run, `ps_output.*` (PS-DTFE) or `output.*` (standard DTFE) beside their `.runlog`.
The files and fields of the TNG data are described in the
[TNG data specifications](https://www.tng-project.org/data/docs/specifications/).

1. **Download and merge** (a TNG API key in `~/.tng_api_key`, `TNG_API_KEY` or `-k`):
   `DTFE_SIM=TNG100-3-Dark scripts/download_snapshots.sh 50 99` fetches the raw chunks (`-c` the group catalogues,
   `-t` the merger trees, `-i` the initial conditions; no numbers: the 19-snapshot ladder);
   `python3 python/tools/merge_HDF5.py -d <the simulation's folder> --snaps 50 99` merges them (the launcher's Data tab runs
   both); a combined file appears only when every chunk is there and the totals verify.
2. **Run an estimator**: `DTFE_SIM=TNG100-3-Dark scripts/run_ps_dtfe.sh 50 99` (phase-space) or
   `DTFE_SIM=TNG100-3-Dark scripts/run_dtfe.sh 50 99` (standard; with no numbers both run snapshot 99). Both default to the production
   settings, so `-g 512` and `-e` are not needed: since 2026-11-09 both drivers average exactly by default and pass
   `--exact` (`--sampled`, `DTFE_EXACT_AVERAGE=0` or a `SUBSAMPLES` count selects the sampled average), `DTFE_GPU`
   picks the device (`-m` forces the GPU), and the run mode is `balanced` (`performance` inside a SLURM or PBS job)
   unless `--run-mode` or `DTFE_RUN_MODE` says otherwise.
   `DRY_RUN=1` prints each snapshot's command and runs nothing; `--help` lists every knob.
   `SIMS=all scripts/run_ps_pipeline.sh` runs every simulation on disk, adds a full-box image plane per snapshot, and resumes
   (a snapshot is skipped when its outputs are newer than its inputs and its run log shows the same settings). Under the
   exact default a PS grid set made with the sampled deposit counts as stale and is recomputed;
   `DTFE_EXACT_AVERAGE=0` keeps such sets current.
3. **Analyse**: `DTFE_SIM=TNG100-3-Dark python3 python/analyze.py all` and the scripts in `python/plot/` (below).

**Run logs and safety.** A driver writes `<prefix>.runlog.partial` while it runs and renames it to `.runlog` on success
or `.runlog.failed` on failure, so `.runlog` always belongs to the grids on disk. After a successful rerun the files of
the earlier run that this one did not write move to `<prefix>.superseded/<date_time>/` (never deleted). A driver holds
`<prefix>.runlog.lock` while it runs a snapshot (a second driver is refused), stops with exit 2 when `DTFE_DATA_ROOT`
is unset or the data disk is missing, and skips a snapshot whose folder is not writable or lacks room
(`SPACE_CHECK=0` skips the space check).

| knob (environment) | meaning | default |
|---|---|---|
| `DTFE_DATA_ROOT` | the folder that holds the simulations | none: required (a driver's `-d` names a simulation's folder instead) |
| `DTFE_SIM` (or `-s`) | the simulation | none: required |
| `DTFE_FIGURES_ROOT` | where every figure goes (the plot scripts, `analyze.py`, the launcher) | `python/figures` (git-ignored) |
| `PYTHON` | the python the scripts (and the test suites) run for numpy and h5py; the launcher passes its own | the first of `python3` on `PATH`, Homebrew's and `/usr/local`'s that imports both (`scripts/h5py_python.sh`) |
| `DTFE_PY_RAM_GB`, `DTFE_PY_LOAD_FRAC` | dtfelib's `FieldSet.load`: the available RAM in GB, and the share of it one grid may take before `mode='auto'` memory-maps it instead of reading it | detected; 0.5 |
| `DTFE_SMOOTH_MPC` | the void catalogue's smoothing in Mpc, the same in every box (for the void scripts and `analyze.py export`) | unset: 10 cells |
| `GRID_SIZE` (or `-g`) | cells per axis of the grids | 512 (`config.sh`; the production grids of both estimators) |
| `PADDING` | the padding of each partition (and in standard DTFE of each thread's sub-box) in mean particle spacings | standard DTFE 15 (`run_dtfe.sh`), PS-DTFE 50 (`config.sh`; it matters only for an unsplit tessellation or `PS_LAGRANGIAN_PADDING=0`) |
| `DTFE_GPU` (`-m`, `--gpu`) | the deposit / interpolation on the GPU (1) or the CPU (0), both drivers | `auto` (by the run mode: the GPU when the binary was built with one, except standard DTFE's sampled average in performance, which runs faster on the CPU; else the CPU) |
| `DTFE_RUN_MODE` (`--run-mode`) | how hard a run leans on the machine (*Run mode* under *Memory, partitions and long runs*): `gentle`, `balanced` or `performance`, passed as `--run-mode` on every run, both drivers and the pipeline | `auto` (balanced, or performance inside a SLURM or PBS job) |
| `GPU_SHARE` (`--gpu-share`) | PS-DTFE: the share of a partitioned GPU run's partitions that deposit on the GPU, the others on the free cores at the same time; passed with the exact deposit on the GPU, and whenever it is not `auto` (so that the binary's note shows) | `auto` (the run mode's share) |
| `DTFE_EXACT_AVERAGE` (`-e`, `--exact`, `--sampled`) | 1 = the exact average (`--exact`: PS-DTFE's exact deposit, standard DTFE's exact cell averages), 0 = the sampled one (`--sampled`: PS-DTFE's nSub^3 deposit, standard DTFE's Monte-Carlo cell mean); the drivers pass one of the two flags on every run | exact in both drivers; while it is unset a `SUBSAMPLES` count selects sampled (`-e` or an explicit 1 beside the count stays exact) |
| `SUBSAMPLES` (`-n`, `--subsamples`) | k sub-samples per axis, k^3 per cell, for the sampled average: PS-DTFE's sampled deposit, standard DTFE's Monte-Carlo mean | unset: none passed (with `--sampled` the binary's own count, PS-DTFE 3) |
| `PARTITION`, `MAX_CONCURRENT` | the split of the box and how many parts run at once | auto-tuned |
| `SCRATCH_DIR` | the full grids as files outside RAM (large runs) | off |
| `FIELDS`, `OUTPUT_PREFIX`, `LAMBDA_TH` | the fields, the files' prefix, the web threshold | density, velocity, divergence, shear in both drivers (the gradient, vorticity, dispersion, T-web and V-web on request), `ps_output` (standard DTFE `output`; a `ps_*` prefix refused there), 0.3 |
| `PS_VERTEX_MASS`, `PS_VOLUME_WEIGHTED`, `PS_CAUSTICS` | PS-DTFE: vertex masses; volume-weighted velocity moments; the caustic classes | 1, 1, 1 |
| `FACE_GRADIENT` | 1 = `--face-gradient` (see Face gradient above): PS-DTFE the mean flow's velocity derivatives plus `.a_velDivStreams` and `.a_massFluxDiv` (needs the exact deposit, the default -- not `--sampled`, `DTFE_EXACT_AVERAGE=0` or a lone `SUBSAMPLES` count -- and `PS_VOLUME_WEIGHTED=1`; with the GPU deposit the face sums come from a CPU walk beside it); standard DTFE the same gradient by a second route, its default since 2026-11-09 (0 = `--per-tet-gradient`, the deposit's own; empty = the binary's default) | PS 0; standard empty |
| `SAMPLE_POINTS`, `PTS_VEL_GRAD` | a `--sample-points` file evaluated in the same run; with velocity gradients | off, 0 |
| `PS_POINT_CONVENTIONS` | PS-DTFE planes: `grid` (the grids' conventions) or `legacy` (`--ps-point-conventions legacy`) | `grid` |
| `TESSELLATION_CACHE` | a `--tessellation-cache` folder | off |
| `DTFE_PRECISION` | `double` runs the double-precision pair | `single` |
| `PS_MERGE` | PS-DTFE: `index-order` (bit-reproducible; given, it keeps the memory gate on under the auto-tuner's speed split), `completion-order` (as they finish), `fixed-point` (opt-in; not with `DTFE_PRECISION=double`): `--ps-merge MODE` | `index-order` (passed explicitly: the memory gate stays on under the speed split) |
| `DTFE_RESUME` | 0 = start a killed partitioned run over instead of resuming it (both drivers) | 1 |
| `AUTO_TUNE_REPORT` | 1 = only the auto-tuner's report per snapshot; nothing computed or written | 0 |
| `DRY_RUN` | 1 = print the commands, run nothing | 0 |
| `SPACE_CHECK` | 0 = skip the drivers' free-space check | 1 |

The rest (`THREADS`, `PS_LINEAR_DEPOSIT`, `PS_CAUSTIC_CUSPS`, `PARALLEL_TRIANGULATION`, `PS_LAGRANGIAN_PADDING`,
`PS_LAGRANGIAN_INPUT`, and the pipeline's `SIMS`, `SNAPS`, `NU`, `AXIS`, `CENTER`, `PLANES`, `THICKNESS`, `SUPERSAMPLE`,
`PLANE_WINDOW`, `FORCE`) is listed in each script's `--help`. A switch is off for `0`, `false`, `no` or `off` and on
for any other value (the binaries' rule). A knob's name is its flag's (`--face-gradient`, `FACE_GRADIENT`), except
`DTFE_EXACT_AVERAGE`, which keeps its 1.0.0 name and chooses between `--exact` and `--sampled`; the knobs both
drivers share keep the `DTFE_` prefix.

A knob's older name is still read for one release, with one `warning: OLD is deprecated, use NEW` line, when its new
name is unset (when both are set the new one wins, and the warning says so):

| older name (deprecated) | new name |
|---|---|
| `PS_GPU`, `PS_METAL`, `DTFE_METAL` (deprecated) | `DTFE_GPU` |
| `PS_EXACT`, `DTFE_EXACT` (deprecated) | `DTFE_EXACT_AVERAGE` (exact by default in both drivers since 2026-11-09) |
| `PS_RESUME`, `AVG_SUBSAMPLES` (deprecated) | `DTFE_RESUME`, `SUBSAMPLES` |
| `ORDERED_MERGE=1` / `ORDERED_MERGE=0`, `FIXED_POINT_MERGE=1` (deprecated) | `PS_MERGE=index-order` / `completion-order`, `fixed-point` (fixed-point wins over the pair, as before) |
| `PS_PARALLEL_TRI`, `PS_LAG_PADDING`, `LAGRANGIAN_INPUT` (deprecated) | `PARALLEL_TRIANGULATION`, `PS_LAGRANGIAN_PADDING`, `PS_LAGRANGIAN_INPUT` |
| `PTS_CONVENTIONS`, `TESS_CACHE` (deprecated) | `PS_POINT_CONVENTIONS`, `TESSELLATION_CACHE` |
| `DTFE_GRID_SIZE`, `DTFE_PADDING`, `DTFE_SPACE_CHECK` (deprecated) | `GRID_SIZE`, `PADDING`, `SPACE_CHECK` |
| the pipeline's `WINDOW`, `PY` (deprecated) | `PLANE_WINDOW`, `PYTHON` |
| python's `DTFE_SIGMA_MPC` (deprecated) | `DTFE_SMOOTH_MPC` |

Until they go, the launcher sets every knob under both names (the same value, so a stray older export in its
environment cannot stand beside a new one): a PS job's log carries about six of these warning lines, a standard one
two or three; its command line shows each knob once, by its new name. A launcher started from the Finder takes only
`DTFE_*`, `PYTHON` (and its older name `PY`), `TNG_API_KEY` and `PATH` from the login shell
(`scripts/install_gui_app.sh`), so a bare knob exported there (`SUBSAMPLES`, `PS_MERGE`, `THREADS`, ...) does not reach
its jobs, as before.

### Analysis

`python3 python/analyze.py [check|compute|plot|all|export|voids|cache]` (default `all`) runs the core standard-DTFE
analysis of `DTFE_SIM`: `compute` caches the smoothed density, Hessian and void catalogue, `plot` draws from the cache,
`export` writes the void catalogues as CSV + HDF5, `voids` runs the void scripts, `cache` lists the cache namespaces
(`--prune` lists what is unreachable, `--prune --yes` deletes it; `--move DIR [--yes]` moves the cache, e.g. out of a
cloud-synced folder, and records the folder in `~/.config/dtfe/cache_dir`; `DTFE_CACHE_DIR` overrides it). Every
figure goes under `DTFE_FIGURES_ROOT` (default `python/figures`).

The plot scripts share `--data-root` (else `DTFE_DATA_ROOT`), `--sim`, `--snap`, `--estimator auto|ps|dtfe`, `--prefix`
(another grid set), `--raw` (cell-centre grids) and `--smooth` (cells). Field maps: plot_PS_DTFE.py, plot_DTFE.py, plot_cosmic_web.py,
plot_web_streams.py, plot_velDiv_den.py, plot_PDF_CDF.py, plot_shear_triaxial.py, plot_eigenvalues.py,
plot_pk_compare.py; point-evaluated planes: plot_pointeval.py, plot_pointeval_panels.py; caustics:
plot_caustic_skeleton.py (needs `.causticClass`; `--prefix ps_caustic` for the separate caustic runs); haloes:
plot_halo_tracking.py; voids: below. A non-default option adds a tag to the output names, so the default figures are
never overwritten.

**Voids.** The catalogue's voids are the minima of the density contrast smoothed with a Gaussian of
`SMOOTHING_SIGMA_CELLS` = 10 cells (`python/config.py`), a different physical scale in every box (1.0, 2.2 and 5.9 Mpc in
TNG50-3, TNG100-3 and TNG300-3): compare boxes at one scale with `--smooth-mpc 2.1625` (`config.CROSSBOX_SMOOTHING_MPC`)
or `DTFE_SMOOTH_MPC=2.1625`, whose products get their own names (`analyze.py compute|plot|all` refuses to run under it).
Each void has two sizes: R_eff, the ellipsoid fit's (abc)^(1/3), which sets the resolution cut (largest semi-axis
within 10 smoothing lengths and half the box) and the shapes; and R_v, where the spherically averaged smoothed density
first rises back to the mean (about 0.57 R_eff, searched out to min(24 sigma, L/2 - sigma)). `catalog.R_V_DEFINITIONS`
also gives `delta08` (enclosed contrast -0.8) and `ridge` (the compensation ridge): `catalog.r_v_mpc(cat, 'delta08')`.
The scripts use R_v and the *distinct* voids by default (a void centred inside a deeper one is left out;
`--keep-overlaps` keeps them, plot_void_profiles.py `--radius r_eff` stacks in R_eff). `DTFE_VOID_GATE=none` drops the
ellipsoid cut from the sample (outputs tagged `gatenone`). The radius settings are part of a
catalogue's identity: a cache made with others is re-measured on first load (about 15 s per 512^3 snapshot).
Caveat: the -0.8 radius of a smoothed-field minimum follows the smoothing length, so a size function from it is not a
test of the void abundance theory; `--size-rv-definition delta08raw` measures it on the raw density instead.

| script | what | main options |
|---|---|---|
| plot_void_population.py | abundance, sizes, shapes; the size function against Sheth & van de Weygaert (Jennings et al. 2013; `dtfelib/theory.py`) | `--size-rv-definition`, `--size-smooth` (default 5 cells), `--size-fixed-r` (with `DTFE_VOID_GATE=none`) |
| plot_void_profiles.py | stacked density and velocity profiles in r/R_v, Hamaus et al. (2014) fits | `--edges`, `--radius` |
| plot_void_profile_evolution.py | the stacked profiles at every snapshot, v_r against linear theory | `--rv-definition`, `--replot` |
| plot_void_environment.py | void-in-cloud against void-in-void; the default criterion is net infall in the shell at 2-3 R_v | `--criterion velocity` / `density-smoothed` / `density-raw` |
| plot_void_fate.py | survive / merge / split / vanish census: the particles within R_v (at most 15 smoothing lengths, `lagrangian.FATE_MAX_SIGMA`) found again in the next snapshot; a void survives when half of them end in one descendant (`lagrangian.SURVIVE_FRAC`, `SPLIT_FRAC`) | `--exclusive` (each particle to one void), `--ref-snap` (default 33) |
| plot_void_tracking.py | each z = 0 void back in time | `--link particles`, `--exclusive`, `--n-voids` |
| plot_void_crossbox.py | the boxes side by side at one smoothing in Mpc | `--smooth-mpc` |
| plot_void_convergence.py | grid, particle-mass and estimator convergence of the void statistics | `--factors`, `--match-sigma` |
| plot_caustic_skeleton.py | wall metrics of the voids against random spheres | `--per-void`, `--null-draws`, `--all-caustic` |

The fate census reads every combined snapshot once (TNG100-3-Dark: 16 min, 12.5 GB); its
links are cached, so a rerun takes seconds (`DTFE_FATE_WORKERS`, default up to 4; `DTFE_FATE_CHUNK`, default 8e6
particles per read). `python/tools/void_outlines.py` writes a snapshot's void table as CSV.


## Graphical launcher

`scripts/gui.sh` opens a PySide6 window for the things normally typed in a terminal (`--setup` reopens the first-run
dialog, `--smoke` builds the window, prints what its jobs would use and quits). `scripts/install.sh` sets it up in its
own environment (`~/.venvs/dtfe-gui`, about 1 GB, seeing the main Python's numpy and h5py; `DTFE_GUI_VENV` /
`DTFE_GUI_PYTHON` override) and makes it launchable as an app (macOS: **Lamina** in Applications, which needs
Full Disk Access once for iCloud Drive and external disks; Linux: a menu entry). By hand:
```bash
python3 -m venv ~/.venvs/dtfe-gui --system-site-packages
~/.venvs/dtfe-gui/bin/pip install PySide6
```

The first start walks through the data folder (the folder that holds the simulations, which the launcher keeps and
hands to every job as `DTFE_DATA_ROOT`; it starts without one), building the programs, the TNG API key and a **demo**
(a synthetic crossed-wave simulation run in seconds and opened in Explore). The tabs:
- **Data**: what is on disk per snapshot; downloads (snapshots, group catalogues, merger trees, initial conditions) and
  merges them, with the size against the free disk.
- **Grids**: one run of either estimator (`run_ps_dtfe.sh` / `run_dtfe.sh`): deposit, grid, fields, phase-space
  options, a hi-res slice, memory and merge settings, precision, presets.
- **Custom snapshot**: either binary on any Gadget (HDF5 or binary) snapshot or text file, with its units, box,
  periodicity and initial positions stated; 2D files run on the 2D binaries. Its *Extra options* line passes any
  program option the tab has no control for (typed as on a command line, after the tab's own).
- **Pipeline**: `run_ps_pipeline.sh` for the chosen simulations, with the plane's geometry and the render options.
- **Plots**: the scripts of `python/plot` and `analyze.py`, each with its own options.
- **Explore**: maps of any run's grids (a slice read from disk), field, component, axis, scale and smoothing; a query
  server on the snapshot behind the map (a click lists every stream there; dragging a rectangle zooms by point
  evaluation); an **exact** zoom (a real run on the region: `--region-mpc` for standard DTFE, `--grid-window` for
  PS-DTFE); void outlines from the analysis catalogue (*Sizes* keeps a range of radii); *Compare with*, the figure grid across outputs and redshifts,
  and *Save image* with publication presets. PS velocity derivatives are shown as the mean field ∇⟨v⟩ by default
  (*PS derivatives* switches to the stored ⟨∇v⟩). **Show in 3D** puts the map's field on three planes through the box
  with the voids at their true size, as spheres of radius R_v or as the catalogue's fitted ellipsoids: drag to turn
  it, right-drag to move it, scroll to zoom; click a void to inspect it, double-click it to close up (the planes
  through its centre at full resolution).
- Beside the log: **Queue** (jobs run one after another, kept across sessions), **Figures**, **Runs** (every run log on
  disk: settings, time, peak memory; outputs made by a build before a fix that changed them, legacy planes and
  resumable runs are marked; load the settings back, open in Explore, move to the Trash, build the void catalogue) and
  **Running now** (every DTFE run on the machine, from its progress file, with a Stop button).

Every tab shows the exact commands its choices stand for, flags combinations that would be refused or would not fit
(**Check memory** asks the auto-tuner per snapshot), and can export them as a bash script or a SLURM job. A fresh job
ticks the GPU when the binary was built with one, except standard DTFE's sampled average under Performance.
**Machine use** (Gentle / Balanced / Performance: `--run-mode`, *Run mode* above; Balanced by default) sits right after
the GPU tick on the Grids, Pipeline and Custom tabs. Changing it re-applies the recommended GPU tick unless the tick was
set by hand; a generated SLURM script runs Performance (the choice is for this machine); presets and *Load settings*
leave it alone. The first start after this change ticks the GPU once on the saved PS-DTFE exact jobs, since the GPU
now runs that deposit in every mode.

The labels follow the option names: **Standard DTFE** / **PS-DTFE** (the estimator), **Cell average: Sampled /
Exact** for both (`--sampled` / `--exact`; Exact is ticked by default since 2026-11-09, a saved choice is kept), **Grid window (Mpc)** on the Custom tab (`--grid-window`) beside **Plane window
(Mpc)** on the Pipeline tab (`PLANE_WINDOW`), **Concurrent triangulations** (`--max-concurrent`, which caps the
triangulations built at once in both binaries). The merge tick on the Grids and Pipeline tabs passes `PS_MERGE=index-order`
(an explicit index order: the memory gate stays on under the auto-tuner's speed split); the Custom tab passes
`--ps-merge` only for a choice other than the default. Saved settings written with older option names (the Custom
tab's extra options, its stream density, a plot's criterion or panel projection) load under the new ones.

The code lives in `python/gui`: `app.py` (the window) with the mixins `app_tabs.py`, `app_specs.py`, `app_run.py`,
`app_menus.py`, `app_check.py` and `app_widgets.py`; `explore.py` with `explore_canvas.py`, `explore_grid.py`,
`explore_save.py`, `explore_server.py`, `explore_voids.py`, `explore_zoom.py` and `explore_3d.py` (its geometry in
`view3d.py`, no Qt); `runspec.py` (the job specifications
and their commands, no Qt) with `runspec_pipeline.py`, `runspec_plot.py`, `runspec_data.py` and `runspec_custom.py`;
and `grids.py`, `presets.py`, `results.py`, `runs_view.py`, `setup_wizard.py`, `void_cache.py`, `figsave.py`,
`make_icon.py` (the app icon).
Import `app`, `explore` or `runspec` before any of their mixin modules.


## Citing

If you use Lamina, cite the methods you use, [Schaap & van de Weygaert (2000)](https://ui.adsabs.harvard.edu/abs/2000A%26A...363L..29S/abstract) for DTFE and
[Feldbrugge (2024)](https://arxiv.org/abs/2402.16234) for PS-DTFE, and the DTFE public software it builds on,
[Cautun & van de Weygaert (2011)](https://ui.adsabs.harvard.edu/abs/2011arXiv1105.0370C/abstract). [CITATION.cff](CITATION.cff) lists them with Lamina itself in a
machine-readable form (GitHub's *Cite this repository*).

## Contributors
* **Marius Cautun (Kapteyn Astronomical Institute, Durham University, Leiden University)** - *code and documentation.*
* **Rien van de Weygaert (Kapteyn Astronomical Institute)** - *various discussions about the method and implementation.*
* **Luuk Westerhoek (Kapteyn Astronomical Institute, University of Groningen)** - *phase-space DTFE, the exact / point evaluation methods, automatic memory management, the Metal GPU backend and the analysis pipeline.*


## License

This project is licensed under GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.
