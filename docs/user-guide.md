# User guide

This guide explains how to run the two programs, `DTFE` (standard DTFE) and `PS-DTFE` (phase-space DTFE), and the
batch drivers in `scripts/`. Building is in the [README](../README.md) and the [developer guide](developer-guide.md);
the launcher, the figures and the Python package `dtfelib` are in [launcher and analysis](launcher-and-analysis.md).
Two references hold every detail left out here: each program's help (`-h` lists the options of the two modes,
`--full-help` every option with its full text) and each driver's `--help` (the comment at the top of the script).
[demo/COMMANDS.md](../demo/COMMANDS.md) collects typical command lines, and [timings](timings.md) gives the
measured run time and memory of a full-size run.

Contents: 1 Quick start; 2 Input; 3 Output; 4 Fields; 5 The two modes, Exact and Fast; 6 Other estimators;
7 Regions, windows and points; 8 Phase-space DTFE; 9 The two estimators compared; 10 Device, run mode and GPU share;
11 Memory and partitions; 12 Long runs and the run log; 13 Environment variables; 14 The batch drivers.

## 1 Quick start

Both programs take an input file, an output root and options:

```text
./DTFE    <input file> <output root> [options]
./PS-DTFE <snapshot>   <output root> [options]
```

A first run on the demo snapshot that ships with the code, a periodic box, writes the cell-averaged density
`/path/to/out/demo.a_den` on a 64^3 grid:

```bash
./DTFE demo/z0_64.gadget /path/to/out/demo --grid 64 --periodic --field density_a
```

A phase-space run needs each particle's initial position as well (section 8):

```bash
./PS-DTFE /path/to/snapshot.hdf5 /path/to/out/ps --grid 256 --periodic --field density_a --ps-vertex-mass
```

For a series of IllustrisTNG snapshots use the drivers (section 14):

```bash
DTFE_DATA_ROOT=/path/to/data DTFE_SIM=TNG100-3-Dark scripts/run_ps_dtfe.sh 99
```

`./DTFE --version` and `./PS-DTFE --version` print the program's revision and build time, which every run also
writes into its log (the `Build:` line).

## 2 Input

`--input-format` (`-i`) names the file type:

| `--input-format` | input |
|---|---|
| 101 | binary Gadget snapshot, one file or several |
| 105 | Gadget HDF5 snapshot (the default); a binary Gadget file given as 105 is read as 101 |
| 106 | the HI data of a Gadget HDF5 snapshot |
| 111 | text: the particle count, the box (`x_min x_max y_min y_max z_min z_max`), then `x y z w` per particle |
| 112 | text, positions only |
| 121, 122 | two raw binary layouts |

A second and a third number after the type select the data and the particle species to read (`--full-help`). The
Gadget HDF5 reader takes `Coordinates`, `Velocities`, `Masses` or the header's `MassTable`, `ParticleIDs` and, for
PS-DTFE, `InitialCoordinates`; the box, the scale factor and the Hubble parameter come from the header.

**Units.** `--mpc-unit` is 1 Mpc in the input's length unit: 1000 by default, so the input is read in kpc. `--box`
is in the input's length unit, as the positions and the header's box are; the lengths of `--region-mpc`,
`--grid-window`, `--padding-mpc` and the sample points are in Mpc. Velocities stay in the input's units; for the
V-web the programs convert Gadget velocities to peculiar km/s with `--scale-factor` and `--hubble`, which default to
the header's values.

**Box.** `--periodic` (`-p`) treats the particles as a periodic box, from `--box` or the file header;
`--box x_min x_max y_min y_max z_min z_max`, in the input's length unit, gives the box outright.

**Other input options.** `--scalar-dataset NAME` reads `/PartTypeN/NAME` of a Gadget HDF5 file as the scalar field;
`--particle-fraction f` keeps a random fraction of the particles; `--poisson n` generates n^3 random particles in a
unit box; `--reader-options` passes strings to the readers. `--config FILE` (`-c`) reads options from a file in the
command-line syntax, `#` starting a comment.

**2D.** The `-2d` programs (`DTFE-2d`, `PS-DTFE-2d`) read the same layout with two-column `Coordinates`,
`Velocities` and `InitialCoordinates`; a snapshot with the wrong number of columns is refused with the name of the
program to use. Every field, both modes, `--ps-caustics`, `--grid-window`, `--partition`, the tessellation cache,
`--sample-points` (`x y` points) and `--serve` work in 2D; the GPU, the face gradient, `--parallel-triangulation` and
the swallowtail bit of `--ps-caustic-cusps` are 3D only.

## 3 Output

`--grid` (`-g`) takes one cell count for every axis or one per axis (`-g 256 128 512`). Each field goes to its own file,
`<output root>.<suffix>`: the value at the grid point under the plain suffix, the cell average under `a_<suffix>`
(`.den` and `.a_den`). With the default `--output-format 101` a grid file is raw float32 (float64 from the `-double`
programs), N^3 cells in C order times the components, with no header. 111 writes text; 112, 113 and 114 write text lines
prefixed by the grid index, the sampling position or the cone position; 100 writes binary with a self-describing header.
The density is rho/rho_bar (`--mean-density` sets rho_bar outright).

| `--field` | file | components |
|---|---|---|
| `density`, `density_a` | `.den`, `.a_den` | 1 |
| `velocity` | `.vel` | 3 |
| `gradient`, `divergence`, `shear`, `vorticity` | `.velGrad`, `.velDiv`, `.velShear`, `.velVort` | 9, 1, 5, 3 |
| `tweb` | `.tweb` + `.twebEig` | class (0 void, 1 wall, 2 filament, 3 node) + 3 eigenvalues |
| `vweb` | `.velVweb` + `.velVwebEig` | class + 3 eigenvalues |
| `dispersion` (PS-DTFE) | `.velDisp` + `.velDispTensor` | trace + 6 (xx xy xz yy yz zz) |
| `velocityStd_a` (standard DTFE) | `.a_velStd` | 1 |
| `scalar`, `scalarGradient` | `.scalar`, `.scalarGrad` | the scalar and its gradient |

Each grid has its `_a` twin (`.a_vel`, `.a_velDiv`, ...). PS-DTFE also writes:

| written with | file | holds |
|---|---|---|
| every grid pass | `.streams`, `.a_streams` | the stream count |
| every grid pass | `.hidden_streams`, `.a_hidden_streams` | bitmask of streams the count misses (section 8) |
| the exact deposit | `.tetTouch` | tetrahedra touching the cell (not a stream count) |
| the face gradient | `.a_velDivStreams` | the stream average's divergence (section 5) |
| the face gradient | `.a_massFluxDiv` | the mass flux divergence div(sum rho_s v_s) |
| the face gradient, `gradient_a` | `.a_velGradStreams` | the stream average's gradient |
| the face gradient, `shear_a` | `.a_velShearStreams` | the stream average's shear |
| the face gradient, `vorticity_a` | `.a_velVortStreams` | the stream average's vorticity |
| `--ps-caustics` | `.caustic`, `.causticClass` | fold flag; caustic bitmask (no `a_` twin) |

Point evaluation writes one record per point (`.pts_den`, ...; section 7). Standard DTFE writes
`<output root>_density.badGridPoints` and `<output root>_fields.badGridPoints` (a thread number appended when several
threads run) when the padding left grid cells outside the tessellation (section 11).

**Safe writes.** Every output is written as `<file>.part.<pid>`, and the set is renamed into place after the last
file. If the output folder is missing at that point (an unmounted disk) the program waits for it (`DTFE_OUTPUT_WAIT`
seconds, 300 by default), then writes the set to `$TMPDIR/dtfe-rescue/<folder>_<name>`, prints an `OUTPUT RESCUE`
line with the `mv` command that puts it in place, and exits with status 3.

**Progress.** While a run goes it keeps `<output root>.runlog.progress`, a small JSON file with the phase, the
partitions done, the time left, the memory and the GPU retries; it is removed when the run ends normally.
`DTFE_PROGRESS=0` switches it off.

## 4 Fields

`--field` (`-f`) takes any number of field names; `density_a` is the default. A name ending in `_a` asks for the
average over each grid cell, a name without it for the value at the grid point (in PS-DTFE: the deposit at one point
per cell, exact in an Exact run). The velocity derivatives and the V-web are built on the velocity gradient, whose
route section 5 describes.

**T-web and V-web.** `tweb` classifies the tidal tensor, solved by FFT from the density grid, and needs a periodic
box; `vweb` classifies the velocity shear. Both use the eigenvalue threshold `--lambda-th` (default 0.3; 0 classifies
by the eigenvalues' signs alone). In 2D the classes are 0 void, 1 filament, 2 node. The T-web is computed from the raw
density grid, with no smoothing, so its classes follow each estimator's noise at the cell scale: quote web fractions
at a stated smoothing, recomputed from `.a_den`.

**Velocity moments in PS-DTFE.** The velocity, its gradient, the fields built on it and the scalars are
volume-weighted: each stream counts with the Eulerian volume it fills in the cell, as in standard DTFE. The
dispersion is mass-weighted, a second moment of the phase-space distribution. The run log's `velocity moments` line
says so.

`--hubble`, `--scale-factor` and `--mean-density` override the header values the fields are normalised with;
`--scalars-extensive` marks the scalars as extensive (it matters for the TSC and SPH methods).

## 5 The two modes, Exact and Fast

Each program computes its `_a` fields in one of two modes: `--exact` (Exact, the default) or `--sampled` (Fast). The
two flags exclude each other.

| | Exact (`--exact`) | Fast (`--sampled`) |
|---|---|---|
| standard DTFE | the exact average of the linear interpolant over each cell | a Monte-Carlo average |
| PS-DTFE | the exact deposit: each tetrahedron's exact share of each cell | the deposit at sub-points of each cell |

**Standard DTFE, Exact.** The linear interpolant is integrated over every cell-tetrahedron intersection: no sampling
noise. The CPU clips with the vendored r3d library (Powell & Abel 2015); the GPU integrates the same intersections by a
sweep along z with exact Gauss-Legendre quadrature, equal to float rounding. In the single-precision programs the shares
are summed as integers in one fixed-point unit per grid for the whole run, so a cell does not depend on the partition or
thread split. Beside `--exact`, `--subsamples` is ignored and `--average-method` must be 1.

**Standard DTFE, Fast.** Method 1 (`--average-method 1`, the default) samples each tetrahedron at the first points of
a fixed quasi-random table, in proportion to its volume and centred on its centroid: deterministic, and the same for
any split, thread count or padding. Method 2 samples random points in each grid cell, seeded by `--seed` (else by the
clock); it is the only route to `velocityStd_a`. `--subsamples k` asks for k^3 points per cell (k^2 in 2D); without
it method 1 averages 100 points per cell and method 2 takes 20. In 3D the velocity derivatives stay exact.

**PS-DTFE, Exact.** Every kept tetrahedron is clipped analytically against each cell it reaches and deposits
`mass * V_intersection / V_tetrahedron` there: no sampling noise, the mass conserved to double precision. The velocity
moments integrate the linear velocity profile exactly, and the dispersion carries each intersection's exact velocity
covariance. The CPU uses r3d in double precision, the reference; the GPU uses the z-sweep in float32 and hands the
tetrahedra whose dispersion float32 cannot resolve to the CPU. A tetrahedron too thin to clip, and a partition's
sub-grid ring in the `_a` pass, are sampled at the `--subsamples` count, 3 when none is given.

**PS-DTFE, Fast.** Each tetrahedron shares its mass equally among the sub-points inside it, on a regular nSub^3
sub-grid per cell (nSub^2 in 2D; `--subsamples nSub`, default 1: one point per cell, at its centre). A tetrahedron
that holds no sub-point puts its whole mass in the cell of its centroid, so the mass is conserved at any count. One
point per cell gives noisy per-cell values: smooth the grids before use. More points give a smoother cell average and
take more time.

**How a run picks its mode.** The flag first; then `DTFE_EXACT_AVERAGE` (`1` Exact, `0` Fast; unset or `auto` pins
nothing); then a sampling count given without `--exact` (`--subsamples`, and in standard DTFE `--average-method`),
which selects Fast; else Exact. The run log says which ran and why: standard DTFE's `volume averaging` line, PS-DTFE's
`grid deposit` line. Where Exact does not apply, standard DTFE samples by itself: `velocityStd_a`, a `--redshift-cone`
grid, a build with `MY_SCALAR` and the estimators of section 6; an explicit `--exact` there is refused (on a cone
grid it warns and samples). A PS-DTFE run without an `_a` field keeps the sampled deposit unless `--exact` is given.

**The velocity gradient.** In 3D both programs compute the gradient, and the divergence, shear, vorticity and V-web
built on it, from exact face integrals of the interpolant (Gauss: the cell average of the gradient is the integral of
v n over the cell's faces divided by its volume), wherever this face gradient applies:

- standard DTFE: in every 3D run of the DTFE method, whatever the mode or device, formed in double;
- PS-DTFE: in Exact runs without `--ps-halo-release`. It is the gradient of the volume-weighted mean velocity,
  ∇⟨v⟩, the quantity standard DTFE stores, with the jumps at caustics as exact face differences. The stream
  average ⟨∇v⟩ (each stream's own gradient, averaged) is written beside it in the `*Streams` files, and the same
  face sums give `.a_massFluxDiv`, by continuity the rate at which a cell gains or loses mass.

`--per-tet-gradient` (or `DTFE_FACE_GRADIENT=0`) keeps the deposit's own gradient, in PS-DTFE the stream average
⟨∇v⟩; `--face-gradient` (or `DTFE_FACE_GRADIENT=1`) asks for the face route and is refused where it cannot run. A
run where it does not apply (PS-DTFE Fast, `--ps-halo-release`, 2D) keeps the deposit's own and says why in its
`velocity gradient` line. ∇⟨v⟩ and ⟨∇v⟩ agree in single-stream cells and differ in folds, where an inverted
stream inside a wall diverges while the mean flow converges. dtfelib returns ∇⟨v⟩ for any PS-DTFE grid set
([launcher and analysis](launcher-and-analysis.md#dtfelib)).

The launcher offers the two modes as its Quality setting, Exact and Fast, for both programs. What each mode costs
in time and memory on a full-size snapshot is in [timings](timings.md).

## 6 Other estimators

Instead of DTFE, a grid can be filled by nearest grid point (`--ngp`), cloud in cell (`--cic`), triangular shaped
cloud (`--tsc`), piecewise cubic spline (`--pcs`), SPH (`--sph n`, n nearest neighbours) or the Voronoi density with
nearest-grid-point assignment (`--voronoi`, density only). NGP, CIC, TSC and PCS give the density and the velocity,
SPH also the scalars. The kernels see the periodic images within their reach and are clipped at a face or a partition
seam. Exact does not apply to these methods.

`--interlace` grids the box a second time, moved by half a cell, and averages the two density grids in Fourier space,
which reduces aliasing; it works with every method, PS-DTFE included. Without `--periodic` it needs a `--region` at
least half a cell inside the box, so in PS-DTFE, which refuses a region (section 7), it needs `--periodic`. Only the
density is interlaced: the other fields, the T-web and the V-web are the
first pass's, byte for byte what a run without `--interlace` writes.

Standard DTFE's `--approx-phase-space-density` approximates the phase-space density as f = rho g, with g the DTFE
density of a second tessellation in velocity space, written to the scalar output. It fails where streams cross;
PS-DTFE resolves the streams instead (section 8).

## 7 Regions, windows and points

**A region (standard DTFE).** `--region x0 x1 y0 y1 z0 z1` (fractions of the box) or `--region-mpc` (Mpc) lays the
`--grid` over that part of the box and tessellates the particles around it, padded by `--padding`.
`--partition-index i` computes only partition i of the `--partition` split, to spread one run over several machines.
PS-DTFE refuses all three: they select particles by their present position, and its tessellation is Lagrangian. In
PS-DTFE a `--box` smaller than the particle cloud lays the grid over that box and keeps exactly the mass inside it: a
tetrahedron that crosses a face deposits only its inside share.

**A window (PS-DTFE).** `--grid-window x0 x1 y0 y1 z0 z1` (Mpc, snapped outward to cell edges) deposits and writes
only the cells of the full `--grid` inside that box, from the full run's tessellation. Unlike a smaller `--box`, which
changes the grid, the window's cells equal the full run's: counts and masks bit for bit, Fast's sums to the last
fixed-point unit on the CPU, the rest to float rounding. The cost follows the window, not the grid, so a fine grid
over a small window is cheap (the launcher's exact zoom is such a window). With `--partition` and
`--tessellation-cache` the cached partitions are loaded, those whose streams never reach the window are skipped, and
no snapshot is read when every partition is cached. A window cannot be combined with the T-web or V-web.

**A redshift cone.** `--redshift-cone r_min r_max theta_min theta_max psi_min psi_max` (Mpc and degrees) interpolates
onto spherical coordinates around `--cone-origin x y z`.

**Point evaluation.** `--sample-points FILE` evaluates the fields at arbitrary points, in addition to the grid (give a
small `--grid`, such as `-g 16`, for the points alone). The file holds `x y z` per line (`x y` in 2D) or raw float64
triples, in Mpc. Both programs write one record per point:

| file | holds |
|---|---|
| `.pts_den` | float64 rho/rho_bar |
| `.pts_vel`, `.pts_velDisp` | float64 x3; float64 x6 dispersion tensor (zero in standard DTFE) |
| `.pts_streams` | int32: the stream count (standard DTFE: 0/1 coverage) |
| `.pts_velGrad` (`--pts-vel-grad`) | float64 x9, row-major d v_j / d x_d (the transpose of `.velGrad`) |
| `.pts_denGrad` (`--pts-den-grad`) | float64 x3, the density gradient in 1/Mpc |
| `.pts_scalar` (`--pts-scalar`) | the scalar, e.g. from `--scalar-dataset NAME` |
| `.pts_caustic` (`--ps-caustics`, PS-DTFE) | the caustic bits of `.causticClass` |

PS-DTFE adds per-stream records. `--ps-per-stream` writes `.pts_stream_offsets` (uint64, N+1: point i owns records
`off[i]` to `off[i+1]`) and `.pts_stream_records` (density, vx, vy, vz per stream); `--ps-per-stream-ids` adds
`.pts_stream_ids` (the ParticleIDs of the tetrahedron's four vertices); `--ps-per-stream-lagrangian` adds
`.pts_stream_lagpos` (each stream's Lagrangian origin); `--pts-den-grad` and `--pts-scalar` add
`.pts_stream_dengrad` and `.pts_stream_scalar`.

Point location is exact: a point on a shared face, edge or vertex counts once per stream, across the periodic boundary
and under any `--partition` split (each partition answers the points in the tetrahedra whose centroid it owns). In
PS-DTFE a stream's density at a point is the linear interpolation of its tetrahedron's vertex densities; the velocity,
its gradient and the scalar are the plain mean over the point's streams (the grids' volume weighting at a point); the
dispersion is the density-weighted second moment. A plane follows the run's `--ps-vertex-mass`, so its density matches
the grids'. Point evaluation runs on the CPU in double precision (`DTFE_PTS_THREADS` pins its threads).

**The point server.** `--serve` builds the tessellation once and answers point requests on stdin and stdout until stdin
closes; an answer equals a `--sample-points` run on the same points and split (a partitioned standard server's to float
rounding). The protocol is documented at the top of `src/CGAL_triangulation/ps_serve.cc`; `dtfelib.Estimator` is its
Python client ([launcher and analysis](launcher-and-analysis.md#dtfelib)). With `--partition`, or when one tessellation
would not fit in memory, the server is a composite of partition tessellations, each built once, and a request visits
only the partitions its points reach. With `--tessellation-cache` the partitions live on disk and at most
`--serve-resident` stay in memory (default 0: as many as the budget holds, `DTFE_SERVE_BUDGET_GB`). `--serve-progress`
prints a line per partition and request. `--serve` cannot be combined with `--sample-points`.

## 8 Phase-space DTFE

Standard DTFE tessellates the particles where they are now and gives one value per point. PS-DTFE tessellates them at
their initial (Lagrangian) positions and follows each tetrahedron to where the simulation moved it. Where the
dark-matter sheet has folded, several tetrahedra (streams) cover a point: the density is the sum over the streams,
and the stream count is a field of its own.

**The Lagrangian input.** PS-DTFE reads Gadget HDF5 (type 105) only and takes the snapshot's own
`InitialCoordinates` when it has them. Otherwise `--ps-lagrangian-input FILE` names an HDF5 snapshot of the initial
positions (`%i` in the name for one split over files), whose `Coordinates` are matched to the snapshot's particles by
`ParticleIDs`; `--mpc-unit` converts both. Without either the run stops with an error.

**Tetrahedron masses.** By default a tetrahedron carries rho_bar times its Lagrangian volume, exact for an unperturbed
lattice. Initial positions taken from an evolved initial-conditions snapshot (TNG's, for instance) already carry the
early density contrast, and that mass biases the density. `--ps-vertex-mass` shares each particle's mass equally
among its tetrahedra instead, which removes the bias and conserves the total mass exactly. The drivers and the
launcher pass it.

**Streams and hidden streams.** The stream count is measured at the sub-points of each cell, so a fold thinner than
the sampling can cross a cell unseen. `.hidden_streams` marks those cells: value 1 where a folded tetrahedron overlaps
the cell while the count misses it (an exact test), value 2 where a tetrahedron too small to hold a sub-point put its
mass in the cell of its centroid. A single-stream cell needs `.streams` equal to 1 and the value-1 bit clear. Which
tetrahedra contain a sub-point is decided exactly, so a split gives the same counts as one run, the GPU as the CPU.

**Caustics.** `--ps-caustics` writes `.caustic`, 1 where tetrahedra of both orientations overlap the cell: equal to
`.streams` > 1 or the value-1 bit of `.hidden_streams`, independent of the mode and the split. The CPU and the GPU
agree except in cells crossed by a nearly flat tetrahedron whose orientation the GPU's float arithmetic and the CPU's
double decide differently, and in the `-double` programs in a few cells more. `.causticClass` is a bitmask:

| bit (value) | meaning |
|---|---|
| 0, 1 (1, 2) | a tetrahedron of positive, of negative orientation overlaps the cell |
| 2 to 5 (4 to 32) | the collapse multiplicity k = 0, 1 (wall), 2 (filament), 3 (node) occurs in the cell |
| 6 (64) | umbilic indicator: two eigenvalues of the deformation vanish together |
| 7 (128) | cusp (A3) indicator, with `--ps-caustic-cusps` |
| 8 (256) | swallowtail (A4) indicator, with `--ps-caustic-cusps` (3D only) |

Bits 2 to 8 are recorded at the deposited samples, so they depend on the sampling. The cusp and swallowtail bits are
indicators that rank candidates, not classifications; `--ps-caustic-cusps` needs the CPU deposit and makes the caustic
pass slower. dtfelib decodes the bits ([launcher and analysis](launcher-and-analysis.md#dtfelib)).

**Halo interiors.** `--ps-halo-release D` deposits every tetrahedron compressed beyond a stream density D (in
rho/rho_bar) whole at its centroid cell. Inside haloes the sheet is mixed (Stücker et al. 2021) and the stream count
explodes; releasing those tetrahedra cuts the cost, conserves the mass exactly and leaves single-stream regions
bit-identical. The help suggests D between 100 and 1000. Point evaluation is not affected.

**Boundaries.** A periodic run tiles one Lagrangian period exactly; a run whose padding is too wide for an exact seam
says so. On a non-periodic cloud `--ps-alpha-shape a` keeps only the Lagrangian tetrahedra whose circumradius is at
most a mean spacings (default 3), which drops the flat slivers of the convex hull and makes a partitioned run agree
with one tessellation; 0 keeps the whole hull. Partitions are padded in Lagrangian space by `--ps-lagrangian-padding`
mean spacings (default 8); 0 falls back to the Eulerian `--padding` rule.

**Phase-space density.** `--sample-points` with `--ps-per-stream` gives the per-stream decomposition of the
distribution function at each point, and Q = rho / sigma^3 follows from the density and dispersion grids.

## 9 The two estimators compared

| | standard DTFE (`DTFE`) | PS-DTFE (`PS-DTFE`) |
|---|---|---|
| tessellation | present (Eulerian) positions | initial (Lagrangian) positions |
| density | DTFE density at the particles, interpolated linearly | sum over the streams of mass over volume |
| velocity | the single-valued interpolant | mean over the streams, by volume; dispersion by mass |
| velocity derivatives | ∇⟨v⟩ | ∇⟨v⟩ in Exact, with ⟨∇v⟩ beside it; ⟨∇v⟩ in Fast |
| Exact / Fast | exact average / Monte-Carlo average | exact deposit / sub-point deposit, nSub 1 |
| device of the `_a` work | GPU for Exact on a GPU build, CPU for Fast | GPU for both on a GPU build |
| `--max-concurrent` | threads per partition | partitions in flight |
| padding | `--padding` 15 mean spacings | `--ps-lagrangian-padding` 8 |
| points | the interpolant; `.pts_streams` = 0/1 coverage | every stream (`--ps-per-stream`) |
| streams, dispersion, caustics | no | yes |

In single-stream void interiors standard DTFE's density lies above PS-DTFE's: its tetrahedra bridge wall and void
particles.

## 10 Device, run mode and GPU share

**Device.** On Apple Silicon a `METAL=1` build runs the grid work on the GPU; elsewhere the programs run on the CPU.
`--gpu` asks for the GPU (in a build without one: a warning and the CPU). Without it the program picks:

- PS-DTFE: the GPU wherever its deposit runs there (a 3D GPU build, at most one scalar component, no
  `--ps-caustic-cusps`, not `--serve`), for both modes and for fields with or without `_a`;
- standard DTFE: the GPU for Exact wherever it runs there (a 3D GPU build, the DTFE method, `--average-method 1`, a
  regular grid, at most one scalar component, not `--serve`), the CPU for Fast and everywhere else.

`DTFE_GPU=1` asks for the GPU as `--gpu` does, `DTFE_GPU=0` keeps the CPU, and `DTFE_GPU=auto` leaves the pick to the
program, as an unset variable does; `--gpu` beats the variable. A pass whose own grid holds 2^32 cells or more (the
GPU kernels number fewer) runs on the CPU and says so; the auto-tuner keeps the sub-grids of a split it chooses below
that. Point evaluation always runs on the CPU. The run log names the device and why: standard DTFE's
`'_a' interpolation` line, PS-DTFE's `PS deposit` line. PS-DTFE's GPU deposit matches the CPU's to float rounding in
the cell averages (the `_a` grids), the densities, the stream counts and the flags (in the double build these two may
differ in a few cells); in the unaveraged grids its velocity-derivative grids can differ from the CPU's by more than
rounding in single cells, the vorticity most, as the kernel evaluates them in float where the CPU works in double. The
GPU knobs are in [metal/README.md](../metal/README.md), *Knobs*.

**Run mode.** `--run-mode` sets how hard a run leans on the machine: `performance` takes every core and long GPU
buffers; `balanced` most of the performance cores and short buffers, so the display never waits on the GPU; `gentle`
half of them, so the machine stays usable. The default is `balanced`, and `performance` inside a batch job
(`SLURM_JOB_ID` or `PBS_JOBID` set and not empty). `DTFE_RUN_MODE` is the same choice (any case; `auto` is unset); the
flag beats it. The run log's `run mode` line says what was set and why. The threads start from a base, the
performance cores on Apple Silicon and every core elsewhere:

| what the mode sets | `gentle` | `balanced` | `performance` |
|---|---|---|---|
| threads | half the base | the base less a quarter (1 to 4) | every core |
| threads on 2, 8, 8 performance + 2 efficiency, 64 cores | 1, 4, 4, 32 | 1, 6, 6, 60 | 2, 8, 10, 64 |
| priority | nice 10 on Linux | unchanged | unchanged |
| GPU command buffers (8 ms gaps in every mode) | about 50 ms | about 50 ms | about 250 ms |
| PS-DTFE Exact: items per GPU buffer | 500k | 500k | 4M |
| `--gpu-share auto` (PS-DTFE) | 1 | 1 | from the threads: 0.85 at 4, 0.65 at 8, 0.6 at 10 |

A mode fills in only what was not given: `OMP_NUM_THREADS` and the `DTFE_*_THREADS` pins keep their counts, a
`--max-concurrent` above the mode's threads raises them, and `--partition`, the device and `--gpu-share` stay as
given. The mode changes no byte of PS-DTFE's single-precision grids (but through the GPU share) and none of standard
DTFE's single-precision Exact grids; the other CPU grids (standard Fast, the unaveraged fields, the `-double`
programs) follow the thread count at float rounding.

**GPU share (PS-DTFE).** In an Exact run split into partitions on the GPU, `--gpu-share` deposits a share of the
partitions on the GPU and the others on the free CPU cores at the same time, chosen by partition index alone, so a
rerun repeats it:

- `auto` (the default): the run mode's share. In `performance` it is 1/(1 + 0.09 (threads - 1.8)), in steps of 0.05
  between 0.5 and 1; it is 1 under 4 threads, below `DTFE_AUTO_MINN` particles, in the `-double` programs, and in
  `gentle` and `balanced`;
- `mixed`: the same model in every mode, on the mode's own threads;
- a fraction from 0 to 1: that share (1: every partition on the GPU).

A mixed run's grids differ from an all-GPU or all-CPU run's at float rounding; the mass stays exact. A number is
ignored, with a note, on the CPU, in Fast, in a build without a GPU and for one domain. The run log's
`PS-DTFE: GPU share` line says how many partitions deposit on each device.

The launcher leaves the device to the program unless its Advanced settings set CPU, GPU or CPU + GPU: on Automatic it
starts every run with `DTFE_GPU=auto` and adds no `-m` or `--gpu` of its own, so the rule above decides, and it shows
the device the program reported after a memory check and in its Runs tab
([launcher and analysis](launcher-and-analysis.md#advanced-settings)).

## 11 Memory and partitions

**The automatic split.** Without `--partition` and `--max-concurrent` the program chooses both from the particle
count, the grid, the fields and the machine, and prints its choice and the predicted peak in an `AUTO-TUNE:` line;
options you give always win, and `--partition 1` keeps one tessellation. The budget is 60% of the RAM
(`DTFE_MEM_FRACTION`), never more than the other programs leave free less 4 GB; `DTFE_MEM_BUDGET_GB` sets it
outright, still capped by the free memory. In PS-DTFE, below 2e6 particles (`DTFE_AUTO_MINN`) a run stays in one
domain unless a split into 8 or 27 partitions is predicted clearly faster; such a split merges in a fixed order, so it
is reproducible, but differs from one domain at float rounding (section 12, *Reproducibility*). Standard DTFE takes
the coarsest split, one domain first, that fits the budget with every thread, else the one that fits with the most
threads. The `SUMMARY` at the end of a run compares the predicted peak with the measured one.

`--auto-tune-report` reads the input, plans exactly as a real run would, prints one `AUTO-TUNE-REPORT` line
(`partition=.. mc=.. predicted_gb=.. budget_gb=.. ... over_budget=0|1`) and exits without computing anything; the
drivers' `AUTO_TUNE_REPORT=1` and the launcher's memory check use it. `pts_gb` (the sample points), `snapshot_gb`
(the GPU deposit's host copy of its grids) and `lookahead_gb` (the next partition's tessellations) are parts of
`predicted_gb`, and so are two PS-DTFE keys the line prints only where they are booked: `facewalk_gb`, the face sums
held beside a GPU deposit, and `readkept_gb`, the input reader's freed buffers that macOS's allocator keeps, booked
for a plan of one partition at a time on the GPU under `--scratch-dir` (without `--grid-window` or a mixed GPU share).
`DTFE_AUTO_PTS_N` and `DTFE_AUTO_PTS_VEL_GRAD` size a `--sample-points` file that does not exist yet.

**What `--partition` and `--max-concurrent` mean.**

- PS-DTFE: `--partition X Y Z` splits the Lagrangian box, and `--max-concurrent` is the number of partitions in
  flight. Each partition writes into the whole grid, so the memory grows with the partitions at once times the full
  grids. While several run, a partition waits until the free memory would hold it (the memory gate).
- Standard DTFE: the partitions run one after another, and `--max-concurrent` is the number of threads in each, every
  thread tessellating its own padded sub-box. With `--sample-points` each partition is one tessellation.

When the auto-tuner chose PS-DTFE's concurrency, more partitions start while the live free memory and the budget
hold them; the grids are the same. `--max-concurrent` sets a fixed number.

**Out of core.** `--scratch-dir DIR` backs every allocation of 1 GB or more (`DTFE_SCRATCH_MIN_GB`), in practice the
full-size grids, with memory-mapped files in DIR, so a grid larger than the RAM runs against a local disk with
bit-identical results. DIR must exist on a local, non-synced volume; iCloud paths are refused. The files are unlinked
when made, so they vanish even if the run crashes.

**Padding.** Every partition, and in standard DTFE every thread's sub-box, carries its own padding. Standard DTFE's
`--padding` default is 15 mean spacings, which the drivers, dtfelib and the launcher pass too. An incomplete padding
leaves `*.badGridPoints*` files rather than silently wrong cells, though point values within a few spacings of a face
can be off. `--padding-mpc` gives the padding in Mpc per face. PS-DTFE's own `--padding` default is 5; its partitions
are padded by `--ps-lagrangian-padding` (section 8).

**Tessellation cache.** Building the tessellation and its vertex densities is the main fixed cost of a run.
`--tessellation-cache DIR` (a local, non-synced folder) writes it once, and later runs on the same inputs load it,
whatever their grid, fields or sample points. Any change that would alter it (the input files, box, region, padding,
periodicity, partition, build) is a miss and a rebuild, never a wrong result; `--verbose 3` says why. The files are
compressed. A partitioned run caches every tessellation it builds: one file per PS-DTFE
partition, one per thread sub-box of a standard run (one per partition under `--sample-points`). A file of another
cache format is a miss too: the run writes a new file beside it, and the old one can be deleted.

**Precision.** The `-double` programs (`DTFE-double`, `PS-DTFE-double`) keep every position, density, field and sum in
double from the input on (the HDF5 reader keeps double coordinates, which the single-precision programs round on read)
and write float64 grids, at about twice the memory; dtfelib and the launcher read either kind. The drivers run them with
`DTFE_PRECISION=double`. The GPU kernels are single precision, so a `-double` program's GPU stream counts and caustic
flags may differ from its CPU's in a few cells. Building the `-double` and `-2d` programs is in the
[developer guide](developer-guide.md#building).

## 12 Long runs and the run log

**Partition merge (PS-DTFE).** By default the partitions' grids are added in partition index order whatever order
they finish in (`--ps-merge index-order`), so two identical runs give bit-identical grids. A partition that finishes
before its turn parks its sub-grids in memory, or, when memory is short, spills them to a file in `--scratch-dir`
(else in `$TMPDIR`) with identical results, and its thread goes on. `DTFE_MERGE_SPILL`, `DTFE_MERGE_STAGE_GB` and
`DTFE_SPILL_MIN_GB` tune this. `--ps-merge completion-order` adds the grids as they finish (not bit-reproducible).
The run log's `ordered merge:` line reports the parked and spilled partitions, and its `merge time:` line (standard
DTFE prints one too) the seconds of the merge and how long partitions waited for their turn.

**Checkpoints and resume.** A partitioned run of either program saves its merged grids and the list of finished
partitions in `<output root>.runlog.checkpoint` (with `--scratch-dir`, in the scratch folder) and removes it once the
outputs are in place. After a crash, `--resume` computes only the missing partitions. It refuses, naming what
differs, when the build, the inputs, the split, the command line or the resolved mode, gradient route or device
changed; `--verbose`, `--max-concurrent`, `--run-mode`, `--scratch-dir`, `--tessellation-cache`, `--no-overlap` and
`--ps-merge` may change, and `--gpu-share` while the share it resolves to stays the same. A resumed PS-DTFE run
merged in index order, and a resumed standard run at the same thread count, equal an uninterrupted run byte for byte;
`--sample-points` runs resume too. Three things to know:

- the checkpoint holds each option as given: name an option the same way both times (a short letter counts as its
  long name, `-g 64` as `--grid 64`);
- a standard run that draws random numbers (`--average-method 2`, `--particle-fraction`) resumes to match an
  uninterrupted run only with `--seed`;
- `DTFE_CHECKPOINT=0` writes no checkpoints, and `--interlace` runs are never checkpointed.

The drivers resume by themselves (`DTFE_RESUME=0` starts over).

**Look-ahead.** A partitioned standard run on the GPU tessellates the next partition while the GPU works on the
current one; PS-DTFE does the same at `--max-concurrent 1`. The outputs are byte-identical either way. `--no-overlap`
(or `DTFE_OVERLAP=0`) switches it off; it is skipped where the free memory cannot hold it.

**GPU failures.** The GPU work runs as short command buffers, with a snapshot of the grids every ~10 s of GPU work. A
killed buffer (the display watchdog, another program's GPU hang, out of memory) restores the last snapshot and
retries; a watchdog or memory error slows the pace, which eases back after a clean stretch. Three failures of one
chunk, sixteen in one deposit or a lost device send that partition (standard DTFE: that sub-domain) to the CPU, which
rounds differently. The run log ends with `GPU deposit retries: N (innocent a, timeout b, oom c, other d)`. A deposit
meant for the GPU, asked for or picked, that a pass sent to the CPU (these failures, a pass of 2^32 cells (section 10),
a scalar of more than one component, a build without a GPU) is counted in a closing warning,
`N ... were meant for the GPU and ran on the CPU instead`: standard DTFE counts its sub-domain deposits there, a
partitioned PS-DTFE run its partitions' deposit passes. The grids are complete; the run only took longer.
[metal/README.md](../metal/README.md) describes `DTFE_GPU_CHECKPOINT_CHUNKS` and `DTFE_GPU_PACE_RECOVER_SEC`.

**Reproducibility.** In the single-precision programs, standard DTFE's Exact grids are integer sums in one unit for the
whole run, the same at any split and thread count on either device (the CPU and the GPU agree to float rounding); its
face gradient is the same for every mode and device; PS-DTFE's grids are exact fixed-point sums merged in index order.
Two identical runs differ at rounding only where something is summed in a run-dependent order: standard DTFE's Fast on
the CPU, the point sums, the `-double` programs' CPU sums under the parallel Delaunay insert, a mixed GPU share against
an all-GPU run, and a partition the GPU gave up on. A PS-DTFE run split into partitions and the same run in one
domain agree to float rounding, not byte for byte, but for the exact deposit's touch count `.tetTouch` (below): each
tetrahedron is deposited by the periodic image whose Lagrangian centroid lies in the ownership period, the padding
moves that period, and the rounding of a tetrahedron's values follows its image's coordinates. Which cells a
tetrahedron reaches does not depend on the image in the sampled deposit wherever the images are exact shifts of each
other, as they are in a box that starts at 0 (a snapshot's own box): its sample test is exact, and the cell of a
tetrahedron deposited whole at its centroid (one that catches no sample point, or one `--ps-halo-release` releases)
comes in double precision from the vertices' differences. The exact deposit on the CPU places its clip planes in the
depositing image's frame, so in single precision a split run and one domain can also differ by the float rounding of
the plane positions: the shares by about that much, and the touch count `.tetTouch` by whole touches where a
tetrahedron grazes a plane. The parallel insert runs for
tessellations of at least 1000000 points where more than one core is free (builds with the tbb library);
`--parallel-triangulation` forces it, `--sequential-triangulation` (or `DTFE_TRIANGULATION_THREADS=1`) keeps the
bit-reproducible insert. A share of the exact sums that is not finite or beyond their range (in practice a NaN or
infinite position, velocity or mass in the input) is left out and counted, and the run ends with a warning naming the
grids it is missing from.

**The run log.** A program prints its settings, its progress and a closing summary, as much as `--verbose` (`-v`,
0 to 3, default 3) asks for; the drivers keep that output as `<prefix>.runlog`. The lines to look for:

| line | says |
|---|---|
| `Build:` | the program, its revision and build time |
| `run mode` | the mode, its source, the threads, the GPU buffer length |
| `volume averaging` (standard), `grid deposit` (PS-DTFE) | which mode ran and why |
| `'_a' interpolation` (standard), `PS deposit` (PS-DTFE) | the device and why |
| `velocity gradient` | the gradient route and why |
| `velocity moments` (PS-DTFE) | volume-weighted, the dispersion mass-weighted |
| `triangulation` | the parallel or sequential insert |
| `AUTO-TUNE:` | the split, the partitions at once and the predicted peak |
| `PS-DTFE: GPU share` | the share and how many partitions deposit on each device |
| `ordered merge:`, `merge time:` | the parked and spilled partitions; the merge's seconds |
| `GPU deposit retries:` | the GPU retries by cause |
| `were meant for the GPU and ran on the CPU instead` | how many deposits fell back to the CPU |
| `SUMMARY` | the wall time and the peak memory, predicted against measured |
| `OUTPUT RESCUE` | where the outputs went when the output folder was missing |

## 13 Environment variables

The programs read these besides their options. An empty value is the same as an unset one; a switch is off for `0`,
`false`, `no` or `off` (any case) and on for anything else; a three-way switch is automatic while unset or `auto`. The
GPU knobs are in [metal/README.md](../metal/README.md), the variables for tests and debugging in the
[developer guide](developer-guide.md).

| variable | effect | default |
|---|---|---|
| `DTFE_GPU` | the device: `1` as `--gpu`, `0` the CPU, `auto` as unset; `--gpu` beats it | unset: the program's pick |
| `DTFE_RUN_MODE` | the run mode, as `--run-mode` (any case); the flag beats it | unset: `balanced` |
| `DTFE_EXACT_AVERAGE` | `1` as `--exact`, `0` as `--sampled`; a flag beats it | unset: section 5's rule |
| `DTFE_FACE_GRADIENT` | `1` as `--face-gradient`, `0` as `--per-tet-gradient`; a flag beats it | unset: section 5 |
| `DTFE_RAM_GB` | the RAM the auto-tuner plans with, in GB (it then skips the free-memory cap) | the machine's |
| `DTFE_FREE_GB` | the free memory the auto-tuner and the memory gate see, in GB | the system's figure |
| `DTFE_MEM_FRACTION` | the auto-tuner's budget as a fraction of the RAM, 0.1 to 0.95 | 0.60 |
| `DTFE_MEM_BUDGET_GB` | the budget outright, in GB, still capped by the free memory less 4 GB | unset |
| `DTFE_AUTO_MINN` | PS-DTFE: below this many particles one domain unless a split is clearly faster; GPU share 1 | 2e6 |
| `DTFE_AUTO_PTS_N` | the points the auto-tuner budgets for a `--sample-points` file to come | the file's count |
| `DTFE_AUTO_PTS_VEL_GRAD` | `1`: the auto-tuner budgets the points' velocity gradients | off |
| `DTFE_AUTO_PTS_BOXMEAN` | `1`: the auto-tuner estimates the streams per point over the whole box | off |
| `DTFE_DEPOSIT_THREADS` | threads of a partition's CPU deposit, and of standard DTFE's `_a` average | the cores left |
| `DTFE_DEPOSIT_SLABS` | PS-DTFE's CPU deposit in slabs along x: `1` forces, `0` forbids | automatic |
| `DTFE_DEPOSIT_NSLABS` | the number of those slabs, at most 64 | from the cost per plane |
| `DTFE_PTS_THREADS` | threads of the point evaluation | the cores left |
| `DTFE_TRIANGULATION_THREADS` | forces the parallel insert with this many threads (`1`: sequential) | the rule |
| `DTFE_OVERLAP` | the look-ahead: `1` forces it, on the CPU too; `0` forbids it | with a GPU deposit |
| `DTFE_MERGE_STAGE_GB` | PS-DTFE: a fixed cap on the parked partitions in GB (`0`: never park) | the free-memory rule |
| `DTFE_MERGE_SPILL` | PS-DTFE: `1` spills every parked partition to disk, `0` none | the free-memory rule |
| `DTFE_SPILL_MIN_GB` | the free space a spill leaves on its disk, in GB | 16 in `$TMPDIR`, 2 in a scratch dir |
| `DTFE_SCRATCH_MIN_GB` | the smallest allocation `--scratch-dir` backs with a file, in GB | 1 |
| `DTFE_CHECKPOINT` | `0` writes no partition checkpoints | on |
| `DTFE_PROGRESS` | `0` writes no `<output root>.runlog.progress` | on |
| `DTFE_OUTPUT_WAIT` | seconds to wait for a missing output folder before the rescue copy | 300 |
| `DTFE_SERVE_BUDGET_GB` | composite `--serve`: the memory for the resident partitions, in GB | the auto-tuner's |
| `OMP_NUM_THREADS` | the thread count, which the run mode keeps | the run mode's |
| `SLURM_JOB_ID`, `PBS_JOBID` | set and not empty: the run mode defaults to `performance` | unset |
| `TMPDIR` | the rescue folder, and the merge spills without `--scratch-dir` | `/tmp` |
| `NO_COLOR` | no colours, for any non-empty value | colours on a terminal |
| `CLICOLOR_FORCE` | a switch: colours through a pipe too | off |

## 14 The batch drivers

The drivers run the programs over IllustrisTNG snapshots, from the repository root. They need two settings, neither
with a default:

- `DTFE_DATA_ROOT`, the folder that holds the simulations (`export DTFE_DATA_ROOT=/path/to/data`); without it a
  driver stops with "no data root" (exit 2). A driver's `-d DIR` names one simulation's folder instead.
- the simulation: `-s` or `--sim` (the drivers), `SIMS` (`run_ps_pipeline.sh`; `SIMS=all` for every simulation on
  disk) or `DTFE_SIM`; without one a driver stops with "no simulation given".

`scripts/config.sh` holds the shared defaults and finds a simulation in either layout: per family
(`$DTFE_DATA_ROOT/TNG100/TNG100-3-Dark/`, the family being the name up to its first `-`) or flat
(`$DTFE_DATA_ROOT/TNG100-3-Dark/`). Each snapshot is a folder `snapdir_NNN` holding the merged `combined_NNN.hdf5`
and, after a run, the outputs `ps_output.*` (PS-DTFE) or `output.*` (standard DTFE) beside their `.runlog`. The
[TNG data specifications](https://www.tng-project.org/data/docs/specifications/) describe the files and fields.

**1. Download and merge.** `scripts/download_snapshots.sh` fetches the raw chunks with a TNG API key (`-k KEY`,
`TNG_API_KEY` or a file `~/.tng_api_key`). Without snapshot numbers it fetches a ladder of 19 snapshots; `-c`
(`--groupcats`) fetches the group catalogues, `-t` (`--trees`) the merger trees, `-i` (`--ics`) the initial
conditions.

```bash
DTFE_SIM=TNG100-3-Dark scripts/download_snapshots.sh 50 99
```

`python/tools/merge_HDF5.py` merges the chunks into one h-free file per snapshot; a combined file appears only when
every chunk is there and the totals verify. `--ics` writes `combined_ics.hdf5`, the initial positions PS-DTFE reads,
instead of the snapshots (`--snapshots-too` does both). The launcher's Data tab runs both steps.

```bash
python3 python/tools/merge_HDF5.py --sim-dir /path/to/data/TNG100/TNG100-3-Dark --snaps 50 99
```

A merged snapshot holds the dark-matter particles (`PartType1`) and no other type; the programs read their
positions, velocities and IDs. The file also carries every other per-particle dataset the chunks hold of them, in
the particles' order and h-free: for TNG's dark-matter runs `Potential`, `SubfindDMDensity`, `SubfindDensity`,
`SubfindHsml` and `SubfindVelDisp`, each with TNG's unit attributes and the power of h taken out in
`h_power_applied`. A merged snapshot keeps the positions in single precision (the chunks hold them in double).

`--add-fields` puts those datasets into a combined file merged before it carried them, from the chunks that are
still there; the disk needs room for a copy of the file. Positions, velocities, IDs and the file's time stamp stay
as they are, so no grid built from it looks out of date. The file is larger afterwards: the programs build their
cached tessellations of that snapshot anew once (the old ones can be deleted), and a killed run of it can no
longer `--resume`: run it while no run or query server uses the snapshot, and finish a killed run first.

`--verify` compares each combined file with its chunks, particle for particle and dataset for dataset, and changes
nothing; it exits 1 when a file does not hold everything its chunks do, and names a snapshot whose chunks are gone
as not compared. With `--delete-chunks`, `--add-fields` and `--verify` delete chunks only after that comparison: of
the snapshots that compare the same, and of no other. Chunks that also hold gas or stars do not compare the same
and are kept; a plain merge with `--delete-chunks` removes them once the dark-matter totals verify.

```bash
python3 python/tools/merge_HDF5.py --sim-dir /path/to/data/TNG100/TNG100-3-Dark --snaps 50 99 --add-fields
python3 python/tools/merge_HDF5.py --sim-dir /path/to/data/TNG100/TNG100-3-Dark --snaps 50 99 --verify
```

**2. Run an estimator.** `scripts/run_ps_dtfe.sh` runs PS-DTFE, `scripts/run_dtfe.sh` standard DTFE; with no
snapshot numbers both run snapshot 99.

```bash
DTFE_SIM=TNG100-3-Dark scripts/run_ps_dtfe.sh 50 99
```

```bash
DTFE_SIM=TNG100-3-Dark scripts/run_dtfe.sh 50 99
```

Both default to the production settings: a 512^3 grid, `--periodic`, Exact (they pass `--exact` or `--sampled` on
every run), the fields density, velocity, divergence and shear, `--lambda-th 0.3`; standard DTFE with `--padding 15`,
PS-DTFE with `--ps-vertex-mass` and `--ps-caustics`. The device, the run mode and the GPU share are the program's:
the drivers hand `DTFE_GPU` and `DTFE_RUN_MODE` on as you set them. Their options:

| option | does |
|---|---|
| `-d DIR` (`--sim-dir`) | the simulation's folder, instead of finding it under `DTFE_DATA_ROOT` |
| `-s SIM` (`--sim`) | the simulation |
| `-g N` (`--grid`) | cells per axis (`GRID_SIZE`) |
| `-n K` (`--subsamples`) | the Fast count (`SUBSAMPLES`) |
| `-m` (`--gpu`) | the GPU: starts the program with `DTFE_GPU=1` |
| `-e` (`--exact`), `--sampled` | Exact or Fast |
| `--run-mode MODE` | passed to the program as given; `auto` passes nothing |
| `-h` (`--help`), `--version` | the script's header; the revision |

`DRY_RUN=1` prints each snapshot's command and runs nothing; `AUTO_TUNE_REPORT=1` prints the auto-tuner's report per
snapshot and writes nothing.

**A particle field as a grid.** `SCALAR_DATASET=NAME` grids one of the snapshot's per-particle datasets, such as
`Potential`, as the scalar field (`<prefix>.a_scalar`): the driver passes `--scalar-dataset NAME` and adds `scalar_a`,
and `density_a` when it is missing, to `FIELDS`. Standard DTFE's cell value is the cell average of the linear
interpolant of the particles' values, which suits a single-valued field such as the potential; PS-DTFE's is the
streams' mean weighted by their volume in the cell. The values keep the dataset's units. Give the run a prefix of its
own, since a rerun into an existing prefix moves the earlier set's files it does not rewrite aside. A scalar field in
`FIELDS` without `SCALAR_DATASET` is refused: the program would read no value and write zeros.

To add the field to a set that exists, run it as above (at the set's grid size, with the same estimator) and move two
files beside that set: `<prefix>.a_scalar` as `<set>.a_scalar` and the run's `<prefix>.runlog` as
`<set>.a_scalar.log`. Explore then lists the field in that set and takes the dataset from that log; the potential
shows as "Gravitational potential". A later rerun into the set without `SCALAR_DATASET` moves both aside with the rest.

```bash
SCALAR_DATASET=Potential FIELDS=density_a OUTPUT_PREFIX=output1024_potential \
    DTFE_SIM=TNG100-3-Dark scripts/run_dtfe.sh -g 1024 99
```

For PS-DTFE keep the `ps_` prefix, such as `OUTPUT_PREFIX=ps_output1024_potential`: the launcher tells the two
estimators' outputs apart by it. The pipeline compares the dataset too: grids made for another `SCALAR_DATASET` are
recomputed.

**3. The pipeline.** `scripts/run_ps_pipeline.sh` runs `run_ps_dtfe.sh` over every snapshot of every simulation in
`SIMS` and evaluates a full-box image plane per snapshot in the same run (made by `python/tools/make_image_plane.py`).
It takes no arguments: its knobs are environment variables. It resumes: a snapshot is skipped when its outputs are
newer than its inputs and its run log shows the same settings (not the device, run mode, GPU share, split or build).
A run log without a `run mode` header line is not read, and its snapshot is recomputed; a snapshot without a run log
is judged by the file times alone. In an Exact pipeline a grid set made in Fast is stale; a Fast pipeline run keeps a
set current at its own count (`SUBSAMPLES=3` for a set whose run log names 3, `DTFE_EXACT_AVERAGE=0` without a count
for one at one point per cell). `FORCE=1` recomputes everything. The figures come afterwards, from the plot scripts
([launcher and analysis](launcher-and-analysis.md#the-pipelines-figures)).

```bash
SIMS=all scripts/run_ps_pipeline.sh
```

**Run logs and safety.** A driver writes `<prefix>.runlog.partial` while the program runs and renames it to
`<prefix>.runlog` on success or `<prefix>.runlog.failed` on a failure or a stop, so `.runlog` always belongs to the
grids on disk. After a successful rerun, the files of the earlier run that this one did not write move to
`<prefix>.superseded/<date_time>/` with the old run log; nothing is deleted. While it runs a snapshot, a driver holds
`<prefix>.runlog.lock` and refuses a second driver on it. A driver stops with exit 2 when the data disk is missing,
and fails a snapshot, without running it, when its folder is not writable or lacks room for the grids
(`SPACE_CHECK=0` skips the space check); the batch goes on and exits 1 at the end. Where `/usr/bin/time` exists, it
reports the run's peak memory.

**Knobs.** The drivers read these environment variables; the command-line options above win where both exist. A
switch is off for `0`, `false`, `no` or `off` and on for any other value, as in the programs.

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

For PS-DTFE, `PADDING` matters only for one tessellation or with `PS_LAGRANGIAN_PADDING=0`. While
`DTFE_EXACT_AVERAGE` is unset a `SUBSAMPLES` count selects Fast; beside Exact the count is ignored with a note.
`run_ps_dtfe.sh` refuses `FACE_GRADIENT=1` in Fast. Most knobs are named after their option (`FACE_GRADIENT` for
`--face-gradient`); `GRID_SIZE`, `FIELDS`, `THREADS` and `DTFE_EXACT_AVERAGE` (which chooses between `--exact` and
`--sampled`) are not, and `DTFE_SIM`, `DTFE_GPU`, `DTFE_RUN_MODE` and `DTFE_RESUME` carry the prefix `DTFE_`.

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

The image-plane files are large; they stay on disk as the input of every figure.
