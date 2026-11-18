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
`./DTFE --version` prints a program's revision and build time. The other make targets and build variables, the
Docker image and the shared library are in the [developer guide](docs/developer-guide.md#building).

## A first run

**Standard DTFE.** `demo/z0_64.gadget` is a binary Gadget snapshot of a periodic box at z = 0, with lengths in kpc,
the programs' default unit. Its velocities are not usable, so ask it for the density only:

```bash
./DTFE demo/z0_64.gadget z0_64 --grid 64 --periodic --field density density_a
```

The program recognises the binary Gadget format by itself (`--input-format 101`; the default is Gadget HDF5) and
prints its log to the terminal. It writes `z0_64.den`, the density at each cell centre, and `z0_64.a_den`, its
average over each cell, both in units of the mean density: raw float32 (float64 from the double-precision build)
with no header, 64^3 cells in C order, a vector field's components side by side in each cell. numpy reads one with
`np.fromfile("z0_64.a_den", dtype=np.float32).reshape(64, 64, 64)`.

**PS-DTFE** needs each particle's initial position, which the demo file lacks. `tests/generate_ps_test_data.py`
(numpy and h5py) writes a Gadget HDF5 snapshot that holds them, the kind of box the launcher's demo runs: 32^3
particles in a periodic box of 100 Mpc, displaced by crossed sine waves so that up to 27 streams overlap.

```bash
python3 tests/generate_ps_test_data.py --out waves.hdf5 --crossed-waves
```

```bash
./PS-DTFE waves.hdf5 waves --grid 64 --periodic --mpc-unit 1 --field density_a dispersion_a
```

The generator writes lengths in Mpc, hence `--mpc-unit 1`. The run writes `waves.a_den` (the density summed over
the streams), `waves.a_velDisp` and `waves.a_velDispTensor` (the velocity dispersion: its trace and its six
components) and `waves.a_streams` (the stream count), with a few companion files. `./PS-DTFE -h` lists what a user of
the two modes needs, `--full-help` every option, and the [user guide](docs/user-guide.md) every output file.

## Two modes: Exact and Fast

Fields whose names end in `_a` (`density_a`, `velocity_a`, ...) are averages over each grid cell; the others are
values at the cell centre. Both programs compute the cell averages in one of two modes:

- **Exact** (`--exact`, the default): standard DTFE integrates its linear interpolant exactly over every cell;
  PS-DTFE gives every cell the exact share of each tetrahedron that overlaps it, and in 3D takes the velocity
  derivatives of the mean flow from exact integrals over the cell faces.
- **Fast** (`--sampled`): standard DTFE takes a Monte-Carlo average of each cell, and in 3D its velocity derivatives
  stay exact. PS-DTFE deposits at one point per cell: quick, with noisy values per cell, so smooth the grids before
  use; its velocity derivatives are then each stream's own, averaged over the streams.

On a `METAL=1` build each program puts its grid work on the GPU where that work has a GPU path (standard DTFE's Fast
average stays on the CPU); `--gpu` asks for the GPU, `DTFE_GPU=0` in the environment keeps the CPU. `--run-mode`
sets how hard a run leans on the machine: `balanced` by default (`performance` inside a SLURM or PBS job),
`performance` or `gentle`. The user guide describes the [two modes](docs/user-guide.md#5-the-two-modes-exact-and-fast)
and the [device and run mode](docs/user-guide.md#10-device-run-mode-and-gpu-share) in full;
[docs/timings.md](docs/timings.md) gives the measured run time and memory of both modes on a full-size snapshot.

## Batch drivers

For IllustrisTNG snapshots, the scripts work in the folder that holds your simulations, which you name once
(`export DTFE_DATA_ROOT=/path/to/simulations`). `scripts/download_snapshots.sh` (with a TNG API key in
`~/.tng_api_key`) and `python/tools/merge_HDF5.py` fetch and merge the snapshots and their initial conditions, and
two drivers run an estimator over a series of snapshots:

```bash
scripts/run_ps_dtfe.sh -s TNG100-3-Dark 99
```

```bash
scripts/run_dtfe.sh -s TNG100-3-Dark 99
```

Both run Exact at 512^3 cells with the density, the velocity and its divergence and shear, and write the grids
beside the snapshot with a `.runlog`; `--sampled` selects Fast, `DRY_RUN=1` prints the commands and runs nothing,
and `--help` lists every knob. `scripts/run_ps_pipeline.sh` runs every snapshot of the simulations in `SIMS` and adds
a full-box image plane to each. The [user guide](docs/user-guide.md#14-the-batch-drivers) describes the data layout,
the downloads and every knob.

## The launcher

The launcher is a window for what the drivers and the programs do from a terminal:

```bash
scripts/gui.sh
```

It downloads and merges TNG data, runs either program on a TNG snapshot or on any snapshot file, runs the pipeline
and the plot scripts, and shows the grids as maps and in 3D with the voids of the analysis catalogue. Its first
start asks for the data folder, builds the programs if needed and offers a demo run. Each run tab asks for the
Quality (Exact or Fast) and the Machine use (Gentle, Balanced or Performance) and shows the commands its choices
stand for, which it can export as a bash script or a SLURM job. The program picks the device unless the launcher's
Advanced settings set the CPU, the GPU or both. `scripts/install.sh` sets it up;
[docs/launcher-and-analysis.md](docs/launcher-and-analysis.md) describes it with the figures, the plot scripts and
`dtfelib`.

## Tests

After a build, a quick test of both programs (three suites, no simulation data; it builds nothing):

```bash
make check
```

Every CPU suite against `./DTFE` and `./PS-DTFE`, as the continuous integration runs them (standard DTFE's
comparison with its stored reference included):

```bash
tests/ci_suite.sh core
```

The suites run `PYTHON` when it is set, else the first `python3` that imports numpy and h5py; the analysis suites of
the core stage also need scipy and matplotlib. On a Metal build `tests/dtfe_metal_check.sh` compares the GPU with
the CPU; do not run the `build` or `all` stage of `tests/ci_suite.sh` there, as it rebuilds the programs for the
CPU. The other stages and suites are in the [developer guide](docs/developer-guide.md#tests).

## Where everything else is

- [docs/user-guide.md](docs/user-guide.md): running the two programs and the batch drivers: inputs, fields and
  output files, the two modes, devices and run modes, memory and partitions, resumed runs, point evaluation, the
  drivers' knobs.
- [docs/launcher-and-analysis.md](docs/launcher-and-analysis.md): the launcher, the pipeline's figures, the plot
  scripts and tools, and the Python package `dtfelib`.
- [docs/developer-guide.md](docs/developer-guide.md): building in detail, the source layout, the test suites and
  the conventions of the code.
- [RELEASE_NOTES.md](RELEASE_NOTES.md): what changed for a user, with a table of old and new names.
- [docs/decisions.md](docs/decisions.md): the design decisions behind the defaults.
- [docs/timings.md](docs/timings.md): the measured run time and memory of the two modes on a full-size snapshot.
- [demo/COMMANDS.md](demo/COMMANDS.md): every script and program with its useful flag combinations.
- [metal/README.md](metal/README.md): the Metal GPU backend and its knobs.
- [docs/DTFE_user_guide.pdf](docs/DTFE_user_guide.pdf): the guide of the DTFE public software that the standard
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
[LICENSE.md](LICENSE.md) and [COPYRIGHT](COPYRIGHT). The bundled k-d tree kdtree2 (`src/kdtree`) keeps its own
licence, the Academic Free License 1.1 with a request to cite its paper that is advisory beside the GPL: see
[src/kdtree/LICENSE](src/kdtree/LICENSE).

[schaap2000]: https://ui.adsabs.harvard.edu/abs/2000A%26A...363L..29S/abstract
[vdws2009]: https://ui.adsabs.harvard.edu/abs/2009LNP...665..291V/abstract
[cautun2011]: https://ui.adsabs.harvard.edu/abs/2011arXiv1105.0370C/abstract
[feldbrugge2024]: https://arxiv.org/abs/2402.16234
