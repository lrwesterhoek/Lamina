# Lamina

**Lamina** (Linear Adaptive Multi-stream Interpolation for N-body Analysis) turns the particles of a cosmological
simulation into continuous density and velocity fields, on a grid or at any set of points. It has two estimators,
both built on Delaunay tessellations and both in two and three dimensions: the Delaunay Tessellation Field Estimator,
run by the program `DTFE`, and the phase-space DTFE, run by the program `PS-DTFE`. A *lamina* is a thin sheet: the
dark-matter sheet in phase space that PS-DTFE tessellates, and the walls of the cosmic web that both estimators
resolve without smoothing them away.

**DTFE** ([Schaap & van de Weygaert 2000][schaap2000]; [van de Weygaert & Schaap 2009][vdws2009]) tessellates the
particles where they are, estimates the density at each particle from the volume of the tetrahedra around it and
interpolates every field linearly inside each tetrahedron. The resolution follows the particles from voids to
clusters, with no smoothing length to choose. The fields are single-valued and volume-weighted, like analytical
predictions (most other schemes weight by mass): one density and one velocity, with its divergence, shear and
vorticity, at every point. Use it for any point set (a simulation, a galaxy survey with a weight per point, any
quantity known at a set of points); Lamina extends the DTFE public software of
[Cautun & van de Weygaert (2011)][cautun2011].

**PS-DTFE** ([Feldbrugge 2024][feldbrugge2024]), after the phase-space tessellations of Abel, Hahn & Kaehler (2012)
and Shandarin, Habib & Heitmann (2012), tessellates the particles at their initial (Lagrangian) positions and follows
each tetrahedron to where the simulation moved its four particles. Where the dark-matter sheet has folded, a point
lies inside several tetrahedra, one per stream, which PS-DTFE resolves where DTFE averages them into one. Besides the
density (the sum over the streams) and the mean velocity it gives the stream count, the velocity dispersion and the
caustics where the sheet folds. It needs every particle's initial position: use it for dark-matter N-body snapshots
that carry them or come with their initial conditions.

Around the two programs: the T-web and V-web, grids on a box, a zoom region or a redshift cone and values at any
points, exact cell averages, deposits on the GPU (Metal on Apple Silicon), a split of large boxes chosen from the
machine's memory, resumable runs, an out-of-core mode for grids larger than the memory, a tessellation cache, a point
server queried from Python, comparison estimators (NGP, CIC, TSC, PCS, SPH, Voronoi), batch drivers for IllustrisTNG
snapshots, the Python package `dtfelib`, a graphical launcher and a shared library. The programs run in parallel on
shared-memory machines (OpenMP).

![The 2D Delaunay tessellation of particles from a cosmological simulation](figures/DTFE_filament.png)

*The 2D Delaunay tessellation of a set of particles from a cosmological simulation. Courtesy: Willem Schaap.*

## Prerequisites

You need macOS (Apple Silicon or Intel) or Linux, a C++ compiler with OpenMP, and GSL, Boost, CGAL, MPFR, GMP, HDF5
and FFTW. The GPU path is written in Metal and needs Apple Silicon; elsewhere the programs run on the CPU.
`scripts/install.sh` (below) installs any missing library; to install them yourself:

```bash
# macOS
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"   # Homebrew
brew install gsl boost cgal mpfr gmp hdf5 fftw llvm libomp
```

```bash
# Ubuntu, Debian
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install libgsl-dev libboost-all-dev libcgal-dev libmpfr-dev libhdf5-dev libgmp-dev libfftw3-dev
```

```bash
# Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel fftw-devel
# older RHEL, CentOS
sudo yum groupinstall "Development Tools"
sudo yum install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel fftw-devel
```

```bash
# Arch, Manjaro
sudo pacman -S base-devel gsl boost cgal mpfr hdf5 gmp fftw
```

The Makefile also uses `tbb` (the parallel Delaunay insert) and `libdeflate` (faster tessellation-cache compression)
when it finds them. `dtfelib`, the plot scripts and the tests need Python 3.10 or newer with numpy, h5py, scipy and
matplotlib; the launcher also needs PySide6, which `scripts/install.sh` puts in an environment of its own.

## Building

The quick way is one script, run from the repository root:

```bash
./scripts/install.sh
```

It installs missing dependencies, builds both programs with the best backend for the machine (Metal on Apple
Silicon, else the CPU), rebuilds any double-precision or 2D pair already built, installs `dtfelib` and sets up the
launcher. A failed compile keeps the previous programs. Its flags (`--cpu`, `--double`, `--jobs N`, `--no-deps`,
`--no-python`, `--no-gui` for a cluster, `--no-app`, `--docker`) are explained by `./scripts/install.sh --help`.

To build by hand on Apple Silicon, with the GPU path:

```bash
make DTFE PS-DTFE METAL=1 -j8
```

On Linux and Intel Macs leave out `METAL=1`: the programs then run on the CPU. On a Metal build, give `METAL=1` to
every later `make DTFE`, `make PS-DTFE` and `make library` too, as a `make` without it rebuilds for the CPU. Then
install `dtfelib`; installed editable, it finds `./DTFE` and `./PS-DTFE` in the repository by itself:

```bash
python3 -m pip install -e python
```

`make deps-check` checks every required header and prints the install command for anything missing. `DOUBLE=1` on
the make line builds the double-precision pair, `./DTFE-double` and `./PS-DTFE-double`, and `DIM=2` the 2D pair,
`./DTFE-2d` and `./PS-DTFE-2d` (CPU only); each set builds in object folders of its own, so the sets coexist.
`TBB=0` and `LIBDEFLATE=0` build without those libraries, and `make library` builds the shared library that
`demo/examples` links against. `./DTFE --version` prints a program's revision and build time.

## A first run

**Standard DTFE.** `demo/z0_64.gadget` is a binary Gadget snapshot of a periodic box at z = 0, with lengths in kpc,
the programs' default unit. Its velocities are not usable, so ask it for the density only:

```bash
./DTFE demo/z0_64.gadget z0_64 --grid 64 --periodic --field density density_a
```

The program recognises the binary Gadget format by itself (`--input-format 101`; the default is Gadget HDF5) and
prints its log to the terminal. It writes `z0_64.den`, the density at each cell centre, and `z0_64.a_den`, its
average over each cell, both in units of the mean density. numpy reads one with
`np.fromfile("z0_64.a_den", dtype=np.float32).reshape(64, 64, 64)`.

**PS-DTFE** needs each particle's initial position, which the demo file lacks. `tests/generate_ps_test_data.py`
(numpy and h5py) writes a Gadget HDF5 snapshot that holds them: 32^3 particles in a periodic box of 100 Mpc,
displaced by crossed sine waves so that up to 27 streams overlap.

```bash
python3 tests/generate_ps_test_data.py --out waves.hdf5 --crossed-waves
```

```bash
./PS-DTFE waves.hdf5 waves --grid 64 --periodic --mpc-unit 1 --field density_a dispersion_a
```

The generator writes lengths in Mpc, hence `--mpc-unit 1`. `./PS-DTFE -h` lists what a user of the two modes needs,
and `--full-help` every option.

## Output files

Each field goes to `<output root>.<suffix>`: the value at the cell centre under the plain suffix, the cell average
under `a_<suffix>` (`.den` and `.a_den`). A grid file is raw float32 (float64 from the `-double` programs), N^3 cells
in C order, a vector field's components side by side in each cell, with no header (`--output-format` selects text or
a binary format with a header instead). The density is rho/rho_bar.

| `--field` | file | components |
|---|---|---|
| `density`, `density_a` | `.den`, `.a_den` | 1 |
| `velocity` | `.vel` | 3 |
| `gradient`, `divergence`, `shear`, `vorticity` | `.velGrad`, `.velDiv`, `.velShear`, `.velVort` | 9, 1, 5, 3 |
| `tweb`, `vweb` | `.tweb` + `.twebEig`, `.velVweb` + `.velVwebEig` | class (0 void, 1 wall, 2 filament, 3 node) + 3 eigenvalues |
| `dispersion` (PS-DTFE) | `.velDisp` + `.velDispTensor` | trace + 6 (xx xy xz yy yz zz) |
| `scalar`, `scalarGradient` | `.scalar`, `.scalarGrad` | the scalar and its gradient |

PS-DTFE also writes the stream count (`.streams`, `.a_streams`), a bitmask of streams the count misses
(`.hidden_streams`), with the exact deposit the tetrahedra touching each cell (`.tetTouch`), with the face gradient
the stream average's derivatives (`.a_velDivStreams`, ...) and with `--ps-caustics` the caustics (`.caustic`,
`.causticClass`). `--sample-points FILE` evaluates the fields at arbitrary points (`x y z` per line or raw float64
triples, in Mpc) and writes one record per point (`.pts_den`, ...); `--serve` keeps the tessellation and answers
point requests on stdin, which `dtfelib.Estimator` drives from Python.

Every output is written as `<file>.part.<pid>` and the set is renamed into place after the last file. If the output
folder is gone by then, the program waits `DTFE_OUTPUT_WAIT` seconds, writes the set to `$TMPDIR/dtfe-rescue/`,
prints the `mv` command that puts it in place and exits with status 3. While a run goes it keeps
`<output root>.runlog.progress`, a small JSON file with the phase, the time left and the memory.

## Two modes: Exact and Fast

Fields whose names end in `_a` (`density_a`, `velocity_a`, ...) are averages over each grid cell; the others are
values at the cell centre. Both programs compute the cell averages in one of two modes:

- **Exact** (`--exact`, the default): standard DTFE integrates its linear interpolant exactly over every cell;
  PS-DTFE gives every cell the exact share of each tetrahedron that overlaps it, and in 3D takes the velocity
  derivatives of the mean flow from exact integrals over the cell faces (`--face-gradient`; `--per-tet-gradient`
  takes each tetrahedron's own).
- **Fast** (`--sampled`): standard DTFE takes a Monte-Carlo average of each cell, and in 3D its velocity derivatives
  stay exact. PS-DTFE deposits at one point per cell: quick, with noisy values per cell, so smooth the grids before
  use; its velocity derivatives are then each stream's own, averaged over the streams.

**Device.** On a `METAL=1` build each program puts its grid work on the GPU where that work has a GPU path (standard
DTFE's Fast average and point evaluation stay on the CPU); `--gpu` asks for the GPU, `DTFE_GPU=0` keeps the CPU. In
an Exact PS-DTFE run split into partitions, `--gpu-share` sends a share of the partitions to the free CPU cores at
the same time (`auto`, the default, does so only in `performance`). The run log names the device and why.

**Run mode.** `--run-mode` sets how hard a run leans on the machine: `performance` takes every core and long GPU
buffers, `balanced` (the default; `performance` inside a SLURM or PBS job) most of the performance cores and short
buffers, so the display never waits on the GPU, and `gentle` half of them. The mode changes no byte of PS-DTFE's
single-precision grids (except through the GPU share) or of standard DTFE's single-precision Exact grids.

**How long it takes.** TNG100-3-Dark at z = 0 (94 million particles, 75 Mpc/h) on a 512^3 grid, on an M1 Max with
64 GB in the default run mode, each program choosing its own plan, measured in November 2026 (the PS-DTFE peaks
before it handed freed memory back, see below):

| Program | Mode | Device | Wall time | Peak memory |
|---|---|---|---|---|
| `PS-DTFE` | Exact | GPU | 25 min 50 s | 34.1 GB |
| `PS-DTFE` | Fast | GPU | 9 min 36 s | 26.2 GB |
| `DTFE` | Exact | GPU | 7 min 9 s | 23.8 GB |
| `DTFE` | Fast | CPU | 5 min 54 s | 30.4 GB |

## Memory and long runs

**The automatic split.** Without `--partition` and `--max-concurrent` the program chooses both from the particle
count, the grid, the fields and the machine, and prints its choice and the predicted peak in an `AUTO-TUNE:` line;
options you give always win, and `--partition 1 1 1` keeps one tessellation. The budget is 60% of the RAM
(`DTFE_MEM_FRACTION`), never more than the other programs leave free less 4 GB (`DTFE_MEM_BUDGET_GB` sets it
outright). `--auto-tune-report` prints the plan and exits. The `SUMMARY` at the end of a run compares the predicted
peak with the measured one.

**Freed memory on macOS.** Both programs start with the allocator's cache of freed blocks off (`MallocLargeCache=0`),
hand each partition's tessellation memory back after the partition and release the GPU's buffers at once, so their
peak is the memory they use; on TNG100-3-Dark at 512^3 this lowered PS-DTFE's Exact peak from 35.4 to 28.1 GB at
the same run time, with the same grids.

**Out of core.** `--scratch-dir DIR` backs every allocation of 1 GB or more, in practice the full-size grids, with
memory-mapped files on a local disk, so a grid larger than the RAM runs with bit-identical results. DIR must be on a
local, non-synced volume.

**Tessellation cache.** `--tessellation-cache DIR` writes the tessellation and its vertex densities once; later runs
on the same inputs load them, whatever their grid, fields or sample points. Any change that would alter them is a
miss and a rebuild, never a wrong result.

**Checkpoints and resume.** A partitioned run saves its merged grids and the finished partitions in
`<output root>.runlog.checkpoint`. After a crash, `--resume` computes only the missing partitions, and refuses,
naming what differs, when the build, the inputs, the split or the command line changed. A resumed run equals an
uninterrupted one byte for byte (PS-DTFE merged in index order, standard DTFE at the same thread count).

**Reproducibility.** The single-precision Exact grids are fixed-point sums that do not depend on the order of the
additions: the same at any thread count, on the CPU and the GPU. Fast grids, the unaveraged fields and the
`-double` programs follow the thread count at float rounding.

## Environment variables of the programs

An empty value is the same as an unset one; a switch is off for `0`, `false`, `no` or `off` (any case) and on for
anything else; a three-way switch is automatic while unset or `auto`. An option beats its variable.

| variable | effect | default |
|---|---|---|
| `DTFE_GPU` | the device: `1` as `--gpu`, `0` the CPU, `auto` as unset | the program's pick |
| `DTFE_RUN_MODE` | the run mode, as `--run-mode` | `balanced` |
| `DTFE_EXACT_AVERAGE` | `1` as `--exact`, `0` as `--sampled` | Exact |
| `DTFE_FACE_GRADIENT` | `1` as `--face-gradient`, `0` as `--per-tet-gradient` | the mode's |
| `DTFE_RAM_GB` | the RAM the auto-tuner plans with, in GB (it then skips the free-memory cap) | the machine's |
| `DTFE_FREE_GB` | the free memory the auto-tuner, the memory gate and the void tools see, in GB | the system's |
| `DTFE_MEM_FRACTION` | the auto-tuner's budget as a fraction of the RAM, 0.1 to 0.95 | 0.60 |
| `DTFE_MEM_BUDGET_GB` | the budget outright, in GB, still capped by the free memory less 4 GB | unset |
| `DTFE_AUTO_MINN` | PS-DTFE: below this many particles one domain unless a split is clearly faster | 2e6 |
| `DTFE_AUTO_PTS_N`, `DTFE_AUTO_PTS_VEL_GRAD`, `DTFE_AUTO_PTS_BOXMEAN` | the points the auto-tuner budgets for, their velocity gradients, their streams over the whole box | the file's |
| `DTFE_DEPOSIT_THREADS`, `DTFE_PTS_THREADS` | threads of a partition's CPU deposit, of the point evaluation | the cores left |
| `DTFE_DEPOSIT_SLABS`, `DTFE_DEPOSIT_NSLABS` | PS-DTFE's CPU deposit in slabs along x (`1` forces, `0` forbids); their number | automatic |
| `DTFE_TRIANGULATION_THREADS` | forces the parallel insert with this many threads (`1`: sequential) | the rule |
| `DTFE_OVERLAP` | the look-ahead: `1` forces it, `0` forbids it | with a GPU deposit |
| `DTFE_MERGE_STAGE_GB`, `DTFE_MERGE_SPILL` | PS-DTFE: a cap on the parked partitions in GB; `1` spills them all to disk, `0` none | the free-memory rule |
| `DTFE_SPILL_MIN_GB` | the free space a spill leaves on its disk, in GB | 16, 2 in a scratch dir |
| `DTFE_SCRATCH_MIN_GB` | the smallest allocation `--scratch-dir` backs with a file, in GB | 1 |
| `DTFE_CHECKPOINT`, `DTFE_PROGRESS` | `0` writes no partition checkpoints; no progress file | on |
| `DTFE_OUTPUT_WAIT` | seconds to wait for a missing output folder before the rescue copy | 300 |
| `DTFE_SERVE_BUDGET_GB` | composite `--serve`: the memory for the resident partitions, in GB | the auto-tuner's |
| `DTFE_GPU_ITEMS` | `0`: the GPU's per-tetrahedron kernels instead of the item kernels (for comparisons) | on |
| `DTFE_GPU_CHUNK`, `PS_GPU_CHUNK` | the GPU's opening chunk: standard DTFE in work items, PS-DTFE in tetrahedra | automatic |
| `DTFE_GPU_TIMING` | one timing line per GPU batch or kernel run | off |
| `DTFE_GPU_CHECKPOINT_CHUNKS`, `DTFE_GPU_PACE_RECOVER_SEC` | a grid snapshot every n GPU buffers; seconds of clean work before an escalated pace eases | ~10 s of work, 90 |
| `OMP_NUM_THREADS` | the thread count, which the run mode keeps | the run mode's |
| `MallocLargeCache` | macOS: `1` keeps the allocator's cache of freed blocks | off |
| `SLURM_JOB_ID`, `PBS_JOBID` | set: the run mode defaults to `performance` | unset |
| `TMPDIR` | the rescue folder, and the merge spills without `--scratch-dir` | `/tmp` |
| `NO_COLOR`, `CLICOLOR_FORCE` | no colours; colours through a pipe too | colours on a terminal |

## Batch drivers

For IllustrisTNG snapshots, the scripts work in the folder that holds your simulations, which you name once
(`export DTFE_DATA_ROOT=/path/to/simulations`, or `-d DIR` for one simulation's folder), and each run names its
simulation (`-s`, `--sim` or `DTFE_SIM`). A simulation lives in `$DTFE_DATA_ROOT/TNG100/TNG100-3-Dark/` or
`$DTFE_DATA_ROOT/TNG100-3-Dark/`; each snapshot is a folder `snapdir_NNN` with the merged `combined_NNN.hdf5` and,
after a run, the outputs `ps_output.*` (PS-DTFE) or `output.*` (standard DTFE) beside their `.runlog`.

**Download and merge.** `scripts/download_snapshots.sh` fetches the raw chunks with a TNG API key (`-k KEY`,
`TNG_API_KEY` or a file `~/.tng_api_key`); without snapshot numbers it fetches a ladder of 19 snapshots, and `-c`
(`--groupcats`), `-t` (`--trees`) and `-i` (`--ics`) fetch the group catalogues, the merger trees and the initial
conditions. `python/tools/merge_HDF5.py` merges the chunks into one h-free file per snapshot (`--ics` writes
`combined_ics.hdf5`, the initial positions PS-DTFE reads; `--add-fields` adds the per-particle datasets to an older
file; `--verify` compares a file with its chunks; `--delete-chunks` removes chunks only after they verify).

```bash
DTFE_SIM=TNG100-3-Dark scripts/download_snapshots.sh 50 99
```

```bash
python3 python/tools/merge_HDF5.py --sim-dir /path/to/data/TNG100/TNG100-3-Dark --snaps 50 99
```

**Run an estimator.** `scripts/run_ps_dtfe.sh` runs PS-DTFE, `scripts/run_dtfe.sh` standard DTFE; with no snapshot
numbers both run snapshot 99.

```bash
DTFE_SIM=TNG100-3-Dark scripts/run_ps_dtfe.sh 50 99
```

Both run Exact at 512^3 cells with the density, the velocity and its divergence and shear, `--periodic` and
`--lambda-th 0.3`; standard DTFE with `--padding 15`, PS-DTFE with `--ps-vertex-mass` and `--ps-caustics`. Their
options are `-d DIR`, `-s SIM`, `-g N` (`--grid`), `-n K` (`--subsamples`), `-m` (`--gpu`), `-e` (`--exact`) or
`--sampled`, `--run-mode MODE` and `-h` (`--help`). `SCALAR_DATASET=Potential` grids a per-particle dataset as the
scalar field (`<prefix>.a_scalar`); give such a run its own `OUTPUT_PREFIX` (for PS-DTFE starting with `ps_`).

**The pipeline.** `scripts/run_ps_pipeline.sh` runs `run_ps_dtfe.sh` over every snapshot of every simulation in
`SIMS` and evaluates a full-box image plane per snapshot. It skips a snapshot whose outputs are newer than its inputs
and whose run log shows the same settings; `FORCE=1` recomputes everything.

```bash
SIMS=all scripts/run_ps_pipeline.sh
```

**Run logs.** A driver writes `<prefix>.runlog.partial` while the program runs and renames it to `<prefix>.runlog`
on success or `<prefix>.runlog.failed` on a failure, so `.runlog` always belongs to the grids on disk. After a
successful rerun, the earlier run's files that this one did not write move to `<prefix>.superseded/<date_time>/`;
nothing is deleted. A driver holds `<prefix>.runlog.lock` while it runs a snapshot and refuses a second driver on it.

**Knobs.** The drivers read these environment variables; the command-line options win where both exist.

| knob (environment) | meaning | default |
|---|---|---|
| `DTFE_DATA_ROOT` | the folder that holds the simulations | none: required (or `-d`) |
| `DTFE_SIM` (`-s`) | the simulation | none: required |
| `GRID_SIZE` (`-g`) | cells per axis | 512 |
| `PADDING` | the padding of each partition or thread sub-box, in mean spacings | standard 15, PS-DTFE 50 |
| `FIELDS` | the fields | density, velocity, divergence and shear (`_a`) |
| `SCALAR_DATASET` | a per-particle dataset gridded as the scalar; adds `scalar_a` and `density_a` | unset |
| `OUTPUT_PREFIX` | the files' prefix; `run_dtfe.sh` refuses a `ps_*` prefix | `ps_output` (PS-DTFE), `output` |
| `LAMBDA_TH` | the T-web and V-web threshold (`--lambda-th`) | 0.3 |
| `DTFE_GPU` (`-m`) | the device: `1` the GPU, `0` the CPU, `auto` as unset; passed on | unset: the program's pick |
| `DTFE_RUN_MODE` (`--run-mode`) | `gentle`, `balanced` or `performance`, handed to the program | unset: its default |
| `DTFE_EXACT_AVERAGE` (`-e`, `--sampled`) | `1` Exact, `0` Fast | 1 |
| `SUBSAMPLES` (`-n`) | the Fast count k (k^3 points per cell); alone it selects Fast | unset: the program's own |
| `FACE_GRADIENT` | `1` passes `--face-gradient`, `0` `--per-tet-gradient`; empty: the program's route | empty |
| `GPU_SHARE` | PS-DTFE: `--gpu-share`, passed when not `auto` | `auto` |
| `PARTITION`, `MAX_CONCURRENT` | `--partition` and `--max-concurrent` | auto-tuned |
| `THREADS` | the thread count, exported as `OMP_NUM_THREADS` | unset: the run mode's |
| `SCRATCH_DIR` | `--scratch-dir`: the full grids as files on a local disk | off |
| `SAMPLE_POINTS`, `PTS_VEL_GRAD` | a `--sample-points` file for the same run; `1` adds `--pts-vel-grad` | off, 0 |
| `TESSELLATION_CACHE` | a `--tessellation-cache` folder, made if missing | off |
| `PS_VERTEX_MASS`, `PS_CAUSTICS` | PS-DTFE: `--ps-vertex-mass`; `--ps-caustics` | 1, 1 |
| `PS_CAUSTIC_CUSPS` | PS-DTFE: `--ps-caustic-cusps`, with `PS_CAUSTICS` (the deposit then runs on the CPU) | 0 |
| `PS_MERGE` | PS-DTFE: `--ps-merge`, `index-order` or `completion-order`, always passed | `index-order` |
| `PS_LAGRANGIAN_INPUT` | PS-DTFE: the initial-positions file, passed when it exists | `combined_ics.hdf5` of the sim |
| `PS_LAGRANGIAN_PADDING` | PS-DTFE: `--ps-lagrangian-padding` | unset: the program's 8 |
| `PARALLEL_TRIANGULATION` | PS-DTFE: `1` passes `--parallel-triangulation` | 0 |
| `DTFE_PRECISION` | `double` runs `DTFE-double` and `PS-DTFE-double` | `single` |
| `DTFE_RESUME` | `0` starts a killed partitioned run over instead of resuming it | 1 |
| `AUTO_TUNE_REPORT` | `1`: only the auto-tuner's report per snapshot | 0 |
| `DRY_RUN` | `1`: print the commands, run nothing | 0 |
| `SPACE_CHECK` | `0` skips the free-space check | 1 |
| `PYTHON` | the python with numpy and h5py that the scripts run | the first `python3` that imports both |

The pipeline adds its own knobs and hands the others on to `run_ps_dtfe.sh`:

| pipeline knob | meaning | default |
|---|---|---|
| `SIMS` | the simulations, or `all` | none: required |
| `SNAPS` | only these snapshot numbers | every snapshot on disk |
| `NU` | image-plane pixels across the box | 8192 |
| `AXIS`, `CENTER` | the plane's orientation and its position along the axis | z, the box centre |
| `PLANES`, `THICKNESS` | sampling planes across a slab of this thickness in Mpc | 1, 2 |
| `SUPERSAMPLE` | K x K sub-samples per pixel | 1 |
| `PLANE_WINDOW` | `u0 u1 v0 v1` in Mpc: image only that part of the plane | the whole plane |
| `PTS_VEL_GRAD` | `1`: the velocity gradient at every point too | 1 |
| `FORCE` | `1`: ignore the freshness checks | 0 |

## The launcher and the analysis

The launcher is a window for what the drivers and the programs do from a terminal:

```bash
scripts/gui.sh
```

It downloads and merges TNG data, runs either program on a TNG snapshot or on any snapshot file, runs the pipeline
and the plot scripts, and shows the grids as maps and in 3D with the voids of the analysis catalogue. Its first
start asks for the data folder, builds the programs if needed and offers a demo run. Each run tab asks for the
Quality (Exact or Fast) and the Machine use (Gentle, Balanced or Performance) and shows the commands its choices
stand for, which it can export as a bash script or a SLURM job.

`python/analyze.py` runs the core analysis of one simulation on its grids: `check` (which snapshots have grids),
`compute` (caches the smoothed density, Hessian and void catalogue), `plot` (the core figures), `all` (the default),
`export` (the void catalogues as CSV and HDF5) and `voids` (the void scripts). The figure scripts in `python/plot/`
share one parser (`--sim`, `--snap`, `--estimator auto|ps|dtfe`, `--prefix`, `--smooth`); each one's `--help` lists
its options.

```bash
DTFE_SIM=TNG100-3-Dark python3 python/analyze.py all
```

`dtfelib` is the package every script reads the grids through:

```python
from dtfelib import FieldSet

fs = FieldSet("/path/to/data/TNG100/TNG100-3-Dark/snapdir_099")   # the estimator found from the files
rho = fs.density(units="mean")                  # rho/rho_bar for both estimators
v = fs.load("velocity")                         # (N, N, N, 3) in km/s
div = fs.load("divergence", mean_field=True)    # div<v> for both estimators
```

| variable | meaning | default |
|---|---|---|
| `DTFE_FIGURES_ROOT` | where every figure goes | `python/figures` |
| `DTFE_CACHE_DIR` | the analysis cache's folder | the recorded folder, else `python/cache` |
| `DTFE_SMOOTH_MPC` | the void smoothing in Mpc | 10 cells |
| `DTFE_VOID_GATE` | `none` drops the cut's largest-axis clause from the void samples | `ellipsoid` |
| `DTFE_PY_RAM_GB`, `DTFE_PY_LOAD_FRAC` | the RAM `FieldSet.load` sees; the share one grid may take before it is memory-mapped | detected, 0.5 |
| `DTFE_FATE_WORKERS`, `DTFE_FATE_CHUNK` | `plot_void_fate.py`'s parallel chunks; its particles per read | up to 4, 8000000 |
| `DTFE_BIN_DIR` | the folder holding the programs, for `Estimator` | the repository root, then `PATH` |
| `DTFE_GUI_CUBE_GB` | the memory in GB Explore keeps whole grids in | a quarter of the memory, 1.5 GB to 16 GiB |

## Tests

After a build, a quick test of both programs (three suites, no simulation data; it builds nothing):

```bash
make check
```

`tests/ci_suite.sh <stage>` is the one list of suites that CI and the Docker image run: `build` (the single, double
and 2D pairs, CPU-only), `core` (every CPU suite on the single pair, the dtfelib and void suites, the documents
check), `double`, `2d`, `gui` (the launcher), `oldpy` (every file parses under Python 3.10) and `variants` (builds
without TBB and libdeflate, the library and its demo).

```bash
tests/ci_suite.sh core
```

The suites run `PYTHON` when it is set, else the first `python3` that imports numpy and h5py. On a Metal build
`tests/dtfe_metal_check.sh` compares the GPU with the CPU; do not run the `build` or `all` stage there, as it
rebuilds the programs for the CPU.

## For developers

- **Option names** are lower-case kebab-case, spelled the same in both programs; `ps-` marks exactly what needs
  phase-space streams, so standard DTFE has no `ps-` option.
- **Stored key text is not an option spelling.** The tessellation cache's descriptor, PS-DTFE's globals record and
  the checkpoint's settings key are compared verbatim by later runs: changing a label makes every cache miss and
  every checkpoint refuse.
- **A default run's outputs stay byte-identical** unless a change means to change results: run the old and the new
  binary with the same command and `cmp` every output, on the CPU and the GPU, single and double, one domain and
  split.
- **The deposit loops and the Metal kernels are timed before and after an edit**, old and new binary alternating on
  an idle machine (`DTFE_GPU_TIMING=1` on the GPU).
- **Kernels and hosts share their parameter blocks** (`ps_kernel_params.h`, `dtfe_deposit_params.h`): plain C++
  that is also valid Metal, the layout pinned by `static_assert`s.
- **Environment variables go through `src/env_knobs.h`**, one rule for every variable.
- **What this README names exists:** `tests/docs_drift_check.py` checks every option, variable and knob it names
  against the code, and the knob table's defaults against the scripts.

[docs/DTFE_user_guide.pdf](docs/DTFE_user_guide.pdf) is the guide of the DTFE public software that the standard
program extends; for the options of the programs here, use their `--full-help`.

## Citing

If you use Lamina, cite the methods you use, [Schaap & van de Weygaert (2000)][schaap2000] for DTFE and
[Feldbrugge (2024)][feldbrugge2024] for PS-DTFE, and the DTFE public software it builds on,
[Cautun & van de Weygaert (2011)][cautun2011]. [CITATION.cff](CITATION.cff) lists them with Lamina itself in a
machine-readable form (GitHub's *Cite this repository*).

## Contributors

* **Marius Cautun (Kapteyn Astronomical Institute, Durham University, Leiden University)** - *code and
  documentation.*
* **Rien van de Weygaert (Kapteyn Astronomical Institute)** - *various discussions about the method and
  implementation.*
* **Luuk Westerhoek (Kapteyn Astronomical Institute, University of Groningen)** - *phase-space DTFE, the exact /
  point evaluation methods, automatic memory management, the Metal GPU backend and the analysis pipeline.*

## Licence

Lamina is free software under the GNU General Public License, version 3 or (at your option) any later version: see
[LICENSE](LICENSE) and [COPYRIGHT](COPYRIGHT). The bundled k-d tree kdtree2 (`src/kdtree`) keeps its own
licence, the Academic Free License 1.1 with a request to cite its paper that is advisory beside the GPL: see
[src/kdtree/LICENSE](src/kdtree/LICENSE).

[schaap2000]: https://ui.adsabs.harvard.edu/abs/2000A%26A...363L..29S/abstract
[vdws2009]: https://ui.adsabs.harvard.edu/abs/2009LNP...665..291V/abstract
[cautun2011]: https://ui.adsabs.harvard.edu/abs/2011arXiv1105.0370C/abstract
[feldbrugge2024]: https://arxiv.org/abs/2402.16234
