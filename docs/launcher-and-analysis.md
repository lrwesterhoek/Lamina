# The launcher and the analysis tools

Lamina has a graphical launcher, **Lamina**, for everything you would otherwise type in a terminal, and a Python
side that turns the grids into figures and void catalogues. This page covers both. The programs and the batch
drivers are described in the [user guide](user-guide.md); building, the source layout and the tests in the
[developer guide](developer-guide.md); the measured run times of the two modes in [timings](timings.md).

## Starting the launcher

From the repository root:

```bash
scripts/gui.sh
```

`--setup` opens the first-run setup again, `--smoke` builds the window, prints what its jobs would use and quits,
and `--help` prints the usage. The launcher needs PySide6: `scripts/install.sh` sets it up in an environment of its
own, `~/.venvs/dtfe-gui`, which sees the numpy and h5py of your Python (`--no-gui` skips this step;
`DTFE_GUI_VENV` names another folder, `DTFE_GUI_PYTHON` another interpreter). To set it up by hand:

```bash
python3 -m venv ~/.venvs/dtfe-gui --system-site-packages
```

```bash
~/.venvs/dtfe-gui/bin/pip install PySide6
```

`scripts/install.sh` then runs `scripts/install_gui_app.sh` (`--no-app` skips it), which makes the launcher an
application: **Lamina** in `/Applications` and a Desktop shortcut on macOS, a menu entry and a Desktop copy on
Linux (its `--help` lists the options). The app starts `scripts/gui.sh` of the repository it was made from, so make
it again after moving the repository. On macOS its output goes to `~/Library/Logs/Lamina.log`, and an app started
from the Finder may not read iCloud Drive or an external disk until you give it Full Disk Access once (System
Settings > Privacy & Security > Full Disk Access); the launcher explains this and opens that pane when refused.

Started from the Finder, the macOS app takes `PATH`, `PYTHON`, `TNG_API_KEY` and every `DTFE_*` variable from your
login shell, so it finds the same `python3`, `wget` and data root as a terminal; other variables exported there do
not reach its jobs. On Linux the menu entry runs `scripts/gui.sh` with your desktop session's environment. Each run
job also sets its own switches explicitly (the device, `DTFE_GPU=auto` when the program picks it; the cell average,
the gradient route, the merge order), so a variable in your environment cannot change a job behind its command.

### First-run setup

The setup dialog opens on the first start, and from **Setup…** beside the data root:

1. **Data folder**: the folder that holds the simulations, with the grids beside their snapshots (default the
   configured folder, else `~/DTFE-data`). When the repository lies in iCloud Drive with the analysis caches
   inside it, the step offers to move the caches out (`analyze.py cache --move`, below).
2. **Programs**: what is built (GPU backend, TBB, the double-precision pair, the 2D programs) and whether the
   Python of the figure scripts has numpy, h5py, matplotlib and scipy. **Build the programs** runs `make`.
3. **TNG API key**, optional, for downloads: saved to `~/.tng_api_key`, readable only by you.
4. **Try it**: a synthetic crossed-wave demo of 32^3 particles, run at Exact into
   `~/DTFE-output/demo` and opened in Explore.

The **Data root** field at the top of the window is remembered between sessions, and every job gets it as
`DTFE_DATA_ROOT`. With no folder remembered yet it starts at `DTFE_DATA_ROOT` when that is set; otherwise every job
refuses to run until you choose a folder. Dropping a folder on the window makes it the data root, a snapshot
(`.hdf5`) opens in the Custom snapshot tab and a grid file in Explore.

## The window

The left pane has six tabs: Data, Grids, Custom snapshot, Pipeline and Plots each make a job, and Explore shows
grids. The right pane has **Run**, **Add to queue**, **Check memory** and **Stop**, the checks of the current job,
and the tabs Log, Queue, Figures, Runs and Running now.

Every job is a sequence of commands you could have typed: View > Show Commands shows them, and **Export…** saves
them as a bash script or a SLURM batch job. The checks list says before the run what the job cannot do (this
blocks Run), what it warns about and what a choice leaves out. **Check memory** asks the program's auto-tuner what
each snapshot needs on this machine, how it would split the work and on which device it would run, without
computing anything. A failed job gets a banner with the likely cause from its log and a button for the fix.
Tooltips come from the programs' own `--full-help`; Help > What's This? shows the whole text.

## The tabs

**Data.** Name the simulation (for example TNG100-3-Dark); the table shows its snapshots with z and whether the
particle files and group catalogues are raw chunks or merged. Tick what to download from the TNG API (snapshots
and group catalogues of the ticked snapshots, merger trees and initial conditions of the whole simulation) and
what to merge into the h-free combined files (`combined_NNN.hdf5` and the like), optionally deleting the chunks
after a verified merge. The job runs `scripts/download_snapshots.sh` and `python/tools/merge_HDF5.py`, and its
checks compare the size with the free space on the data disk.

**Grids.** One run of `scripts/run_ps_dtfe.sh` or `scripts/run_dtfe.sh` on the chosen snapshots (**Not yet run**
selects those without grids). Choose a preset, the estimator (**PS-DTFE** or **Standard DTFE**), the quality, grid
and machine use, the fields with the T-web / V-web threshold λ_th (default 0.3), the **Phase space** group
(caustics, and cusps on the CPU; PS-DTFE only) and a **Hi-res slice**: the fields point-evaluated on one of the
simulation's image planes, with the velocity gradient if ticked.

**Custom snapshot.** The program itself on any snapshot file outside the TNG layout. You state the file, its
format (Gadget HDF5, Gadget binary or text), its dimensions (2D runs on the CPU-only 2D programs), the length unit,
periodicity, an alpha shape for a non-periodic box (`--ps-alpha-shape`), the box when the header has none, and for
PS-DTFE the initial positions: perturbed initial conditions, whose tetrahedra take their masses from the particles
(`--ps-vertex-mass`), or an unperturbed lattice (the default). Further rows add a per-particle value
(`--scalar-dataset`), the halo release (`--ps-halo-release`), a `--sample-points` file, a **Grid window (Mpc)**
(`--grid-window`), the output folder (default `~/DTFE-output`) and, once the double-precision pair is built, a
**Precision** row. The settings are kept beside the outputs (`<name>.gui.json`) for Explore and Runs. **Extra
options**, under **More**, passes any other program option as typed. Extra options that name the device (`--gpu`,
and for PS-DTFE `--gpu-share`) take the place of the tab's own device options; with `--gpu-share` alone the
program picks the device.

**Pipeline.** `scripts/run_ps_pipeline.sh` for the ticked simulations, in the order you drag them to: grids and a
full-box point-evaluated image plane for every snapshot that is missing or out of date. You set the plane (pixels
across the box, default 8192; sub-samples, axis, position, slab, **Plane window (Mpc)**); **Adaptive** uses Check
memory to take the largest plane each snapshot fits. The tab counts what the script would compute: a slice that is
missing, older than its snapshot or plane, or of another plane, and a run log written by an older version of the
programs. **Run** can recompute everything (`FORCE`), show only the plan (`DRY_RUN`) and then render the figures.

**Plots.** The figure scripts of `python/plot` and `analyze.py` as figure sets with their own options; tick the
sets, pick the estimator, a grid prefix and a smoothing in cells. **Void catalogue build** fills the analysis
cache that Explore and the void figures read. The scripts are listed under [The plot scripts](#the-plot-scripts).

### What the run tabs share

**Quality** is one choice on the Grids, Custom snapshot and Pipeline tabs, with a grey line saying what it means
for the estimator; it stays when the estimator changes.

- **Exact** (the default, `--exact`): PS-DTFE deposits each tetrahedron's exact share of each cell and takes the
  exact derivatives of the mean flow from face integrals; standard DTFE takes the exact cell average.
- **Fast** (`--sampled`): PS-DTFE samples one point per cell, quick but with noisy per-cell values (smooth the
  grids before use), and its velocity derivatives are the stream average of each stream's own gradient. Standard
  DTFE takes a Monte-Carlo mean of each cell, and its derivatives stay exact.

**Machine use** (`--run-mode`): *Gentle* keeps the machine fully usable, *Balanced* (the default) takes most of it,
*Performance* takes every core. An exported SLURM job always runs Performance.

A grey **Runs on** line says where the job runs. Unless the Advanced settings name a device, the program picks it
by its own rule ([user guide](user-guide.md#10-device-run-mode-and-gpu-share)): the job passes `DTFE_GPU=auto` and
adds no `-m` or `--gpu` of its own, and the line reads
"Runs on: the device the program picks ('Check memory' names it)". After a memory check of the job's settings it
names the device the program reported: "Runs on: the GPU (as the memory check reported)", or the CPU. Where a PS-DTFE
Exact run in single precision on the GPU shares its partitions with the free cores (under Performance, or with the
Device CPU + GPU), the line's tooltip says that a run split into partitions deposits some of them on the CPU and that
a small or unsplit run deposits on the GPU alone. A device set in the Advanced settings is named "(set in Advanced)",
"the CPU and the GPU together" for CPU + GPU where the job mixes the two, else "the GPU", and `--gpu` among the extra
options "(set in the Extra options)"; a 2D job says "Runs on: the CPU (the 2D programs have no GPU)" and a program
built without the GPU "Runs on: the CPU (this build has no GPU)".

**Presets**: *Quick look* (Fast, 128^3, density and velocity), *Standard* (Exact, 512^3, density, velocity,
divergence and shear) and *Publication* (Standard plus the caustics and a hi-res slice on the largest image plane),
and your own (**Save as preset…**). A preset never sets the machine use nor the Custom tab's initial positions.

**More** folds the output prefix, **Start over** (do not resume a killed run's checkpoint), a scratch directory
and a tessellation cache folder (both on a local disk). The **time estimate** on the Grids tab, in the queue and in
the export dialog is the median of earlier successful runs with the same simulation, grid, estimator, slice,
precision, dimensions, machine use and quality, and the same device when the Advanced settings name one (the
program picks the same device for the same settings on the same build).

### Advanced settings

View > Advanced Settings (off by default, and remembered) adds an **Advanced** group to the three run tabs, for
tests and special cases. Each row starts on **Automatic**, which sets nothing beyond the job's quality and machine
use; a grey note says what Automatic gives the job now.

- **Device**: Automatic lets the program pick (`DTFE_GPU=auto`; the note reads "Automatic: the program picks" until
  a memory check of the job names the device), CPU, GPU (on a GPU build: every partition on the GPU), or for
  PS-DTFE in 3D and single precision **CPU + GPU**, which shares an Exact run's partitions between the cores and the
  GPU in every machine use (`--gpu-share mixed`; on Automatic a run the program puts on the GPU shares under
  Performance only).
- **Samples per cell axis** (1 to 6, Automatic 1): PS-DTFE Fast's points per cell axis.
- **Velocity derivatives**: exact face integrals or the stream average (PS-DTFE in 3D).
- **Partitions per axis** (1 to 8), **Concurrent triangulations** (1 to 16) and **Memory budget (GB)**:
  Automatic lets the program plan them.
- **Parallel triangulation** (PS-DTFE in 3D, with TBB), **Merge order** (completion order, PS-DTFE) and
  **Precision** (double, once the double-precision program is built).

A job with such a setting names it under its Runs on line ("Advanced: Device: CPU; Partitions: 4"); the checks
warn about one the job cannot use, which is then not passed. Switching the group off puts the three tabs' jobs
back to Automatic; a queued job keeps what it was queued with and is marked "advanced".

## The job queue

**Add to queue** freezes the current job with its settings; its commands are built when it starts. **Start
queue** runs the jobs one after another, and **Up**, **Down**, **Remove** and **Clear finished** edit the list.
Three ticks, on by default and also in the Queue menu: stop the queue when a job fails, wait before a grids,
pipeline or custom-snapshot job while another DTFE or PS-DTFE run is going, and notify when a job or the queue
finishes. The queue is kept across sessions.

## The Runs browser

The Runs tab lists every run on disk from its run log: each `<prefix>.runlog` under the data root and the Custom
tab's output folders, with any later attempt under the same name (`.runlog.failed`, `.runlog.partial`, and a
`.runlog.checkpoint` that makes a killed run resumable). From the log it reads the command, the header (run mode,
device, cell average, gradient route, a sampled run's count), the wall time and the peak memory. The columns give
the run's place, grid, options (Exact or Fast, a PS-DTFE Fast run with its samples per cell axis, and the main
switches), time, peak memory, size, whether its void catalogue is cached, stale or absent, and a status that marks
what to check before use: a later attempt (failed, killed, resumable or still going, and whether it rewrote the
outputs), PS-DTFE stream counts without `.hidden_streams`, a TNG PS-DTFE run without `--ps-vertex-mass`.

A log with a command but no header `run mode` line was written by an older version of the programs. Runs lists
such a run with its name, time, outcome and files, shows `older version` for its settings, which it does not read,
and warns once to recompute it before use; it offers no **Load settings** and leaves the run out of the time
estimates. A log without any command (the run died before the header) shows no settings either.

The buttons open the log, **Load settings** (into the Grids tab, a custom run's into the Custom snapshot tab; the
machine use stays), open the run in Explore, show its folder, move its outputs to the Trash and queue **Build void
catalogue**. Load settings of a TNG run puts the Advanced rows on Automatic; a custom run's brings back the
Advanced settings it ran with (which switches the Advanced settings on). Either way a sampled PS-DTFE run's count
other than Fast's becomes **Samples per cell axis**.

**Running now** lists every DTFE and PS-DTFE run on the machine, wherever it was started, with a Stop button;
**Figures** shows the figures of the last run, or all of them.

## Explore

Explore shows any run's grids as maps. Choose the output by **Simulation**, **Redshift** and **Output** (or
Explore > Open Output…), then the **Field**, **Component** (with in-plane **vectors**), **Slice**, **Colours**,
**Range** and **Smoothing**.

- **Start query server** runs a `dtfelib.Estimator` on the snapshot behind the map; a click then lists every
  stream at that point with its density, velocity and Lagrangian origin (PS-DTFE velocities are the plain mean
  over the streams).
- **Zoom**: drag a rectangle. *Point values (instant)*, the default, evaluates it through the query server; *Exact
  cell averages (recompute)* runs the program on the region (`--region-mpc` for standard DTFE, `--grid-window` for
  PS-DTFE).
- **Contours**: thin lines at round values of the colour scale, marked on the colour bar, so a line keeps its
  value as you scroll through the layers; a slider sets *fewer* or *more*. They are not offered on class maps.
- **PS derivatives**: a phase-space output's velocity derivatives and V-web are shown as the mean field ∇⟨v⟩ (the
  default: an Exact run's stored grids, else computed from the velocity grid) or as the stored grids, the stream
  average ⟨∇v⟩ of a run without the face gradient. An output with the face gradient also lists the stream
  average's `*Streams` fields and the mass flux divergence (`.a_massFluxDiv`).
- **Compare**: a figure grid of one field across redshifts, one snapshot across fields or across outputs.
- **Save image…** with the presets *Standard*, *Publication* (vector PDF, 300 dpi), *Bare image* (the map alone,
  one image pixel per data pixel) and *Slide* (larger fonts). The contour lines on screen are in every saved
  picture, the Bare image too; a PDF or SVG keeps them as lines up to 16000 pieces a colour and holds more as an
  image. The file name offered says what the picture shows: simulation, field, redshift,
  output prefix, grid, plane and slice position, smoothing, log or lin, and the overlays drawn, as in
  `TNG100-3-Dark_density_z0.00_ps_output1024_grid1024_xy-plane_z55.52Mpc_smooth0_log_contours.svg`.

**Voids** draws the catalogue's distinct voids (see [Voids](#voids)) over the map, the figure grid and the 3D
view, found at the production 10 cells or at the σ beside the tick (in Mpc, two cells or more; a search estimated
over two minutes asks first). Draw them *as ellipsoids* (the default), with semi-axes √(2 |δ_min| / λ) where the
Hessian's quadratic fit returns to the mean density, or *as spheres of R_v*, the other radius dashed if ticked; on
the map each is the ellipse where it cuts the slice. **Sizes** keeps a range of radii, a click opens the
inspector, and Save image also writes the outlines and a CSV. Tick **3D** beside Smoothing to see the field the
voids belong to.

**Show in 3D** draws the field as a translucent volume or on three planes, with the voids. Drag to turn it,
right-drag to move it, scroll to zoom; in perspective (the default) scrolling in moves the eye through the field
toward the cursor. The detail runs from 128^3 to 1024^3 cells (default 256^3). Click a void to inspect it, and
double-click it or press **Close up** to see its surroundings at full resolution (**Whole box** goes back). With
Contours ticked the volume shows the density's void-side levels as thin shells. The volume is drawn with OpenGL
where the machine has it; `DTFE_GUI_NO_GL` set to anything but 0 turns that off.

## Analysis

The analysis code is Python under `python/`: the driver `analyze.py`, the figure scripts in `python/plot`, the
tools in `python/tools`, the shared settings in `python/config.py` and the package `dtfelib`. The figure scripts
and `analyze.py` take their data root from `--data-root` or `DTFE_DATA_ROOT` and their simulation from `--sim` or
`DTFE_SIM`; neither has a default. `plot_pointeval.py` and `plot_void_crossbox.py` take a list of simulations
(`--sims`), and `plot_pointeval_panels.py` reads one output (`--root`). The tools take what their table below
names: `merge_HDF5.py` the simulation's folder (`--sim-dir`), `convert_ic_units.py` two paths, and the others a
simulation under the data root. Every figure goes under `DTFE_FIGURES_ROOT` (default `python/figures`).

### The pipeline's figures

`scripts/run_ps_pipeline.sh` computes grids and image planes and draws nothing (see the
[user guide](user-guide.md#14-the-batch-drivers)). Render the point-evaluated maps of every simulation and snapshot
afterwards:

```bash
python3 python/plot/plot_pointeval.py --sims all
```

The `.pts_*` files stay on disk, so you can restyle without recomputing; a figure newer than its data is skipped
unless you pass `--force`. `--project slab` averages every plane of a slab, `--smooth-px` smooths every map, and
each gradient map also gets a companion smoothed at `--smooth-derivatives-px` (default 10 pixels).

### The analysis driver

`analyze.py` runs the core analysis of one simulation on the grids each snapshot folder holds (PS-DTFE's when there
are any, else standard DTFE's; `check` says which):

```bash
DTFE_SIM=TNG100-3-Dark python3 python/analyze.py all
```

| mode | what it does |
|---|---|
| `check` | which snapshots have grids, of which estimator, and whether they are complete |
| `compute` | caches the smoothed density, Hessian and void catalogue of each snapshot |
| `plot` | the core figure set (below), drawn from the cache |
| `all` (default) | `compute` in its own process, then `plot` |
| `export` | the void catalogues as CSV and HDF5 (`--format`, `-o`) |
| `voids` | the void scripts for the simulation |
| `cache` | the cache's namespaces and sizes; `--prune` lists what no setting reaches, `--prune --yes` deletes it |

`--snaps` limits `compute` and `export` (default: the snapshots of the redshift ladder on disk) and `--only` runs
only the named plot scripts. The cache is the folder `DTFE_CACHE_DIR` names, else the one recorded in
`dtfe/cache_dir` under `$XDG_CONFIG_HOME` (default `~/.config`), else `python/cache`.
`analyze.py cache --move DIR --yes` moves the whole cache to `DIR` (copied and verified first) and records it;
without `--yes` it is a dry run, and it refuses while the cache may be in use.

### The plot scripts

All scripts but `plot_pointeval.py`, `plot_pointeval_panels.py` and `plot_void_crossbox.py` share one parser:
`--data-root`, `--sim`, `--snap` (default 99; a series script takes the whole series), `--estimator auto|ps|dtfe`,
`--prefix` (another grid set), `--raw` (the unaveraged grids) and `--smooth` (cells). The tracking scripts take
`--start-snap` in place of `--snap`. Scripts that read a PS-DTFE velocity gradient take `--ps-gradient mean-field`
(the default) or `stored`; the void scripts add `--smooth-mpc`. A non-default option adds a tag to the output
names, so the default figures are never overwritten. Each script's `--help` lists all of its options.

| script | what it makes | main options |
|---|---|---|
| `plot_PS_DTFE.py` | slices of every grid of one snapshot, and its point-evaluated maps | `--pointeval-only` |
| `plot_DTFE.py` | density, velocity, divergence and shear across snapshots | `--snaps`, `--ps-gradient` |
| `plot_cosmic_web.py` | T-web and V-web classes and eigenvalue maps | `--ps-gradient` |
| `plot_web_streams.py` | stream multiplicity per T-web and V-web class | `--web`, `--snaps` |
| `plot_velDiv_den.py` | velocity divergence against density contrast | `--ps-gradient` |
| `plot_shear_triaxial.py` | shear magnitude and triaxiality maps and statistics | `--ps-gradient` |
| `plot_pk_compare.py` | P(k) of the grids against linear theory | `--n-bins`, `--ref-snap` |
| `plot_caustic_skeleton.py` | walls, filaments and nodes of a `--ps-caustics` run | `--per-void`, `--null-draws` |
| `plot_halo_tracking.py` | subhalo mass, shape, mergers and environment back in time | `--subhalos`, `--n-halos` |
| `plot_pointeval.py` | the point-evaluated maps of the image planes | `--sims`, `--maps`, `--project` |
| `plot_pointeval_panels.py` | density, sheet view and stream count of one plane | `--root`, `--plane`, `--panels` |

`analyze.py plot` runs the core set with the shared options: `plot_DTFE.py`, `plot_velDiv_den.py`,
`plot_eigenvalues.py` (Hessian eigenvalues), `plot_shear_triaxial.py`, `plot_PDF_CDF.py` (critical points),
`plot_contour_filter.py` and `plot_shape_filter.py` (void shapes), `plot_ellipses_contour_3D.py` (deep minima and
their ellipses), `plot_marked_correlation_BBKS.py` (marked correlations), `plot_smoothing_comparison.py` (the
catalogue against the smoothing), `plot_phi_delta.py` (potential maxima against void minima) and
`plot_tidal_correlation.py` (void axes against the tidal field).

The void scripts (`analyze.py voids` runs the first six):

| script | what it makes | main options |
|---|---|---|
| `plot_void_population.py` | counts, sizes, shapes; the size function | `--size-rv-definition`, `--size-smooth` |
| `plot_void_profiles.py` | stacked density, velocity, single-stream fraction | `--radius`, `--edges` |
| `plot_void_profile_evolution.py` | the stacked profiles at every snapshot | `--rv-definition`, `--replot` |
| `plot_void_environment.py` | void-in-cloud against void-in-void | `--criterion` |
| `plot_void_fate.py` | survive, merge, split or vanish census | `--exclusive`, `--ref-snap` |
| `plot_void_convergence.py` | grid, particle-mass and estimator convergence | `--factors`, `--match-sigma` |
| `plot_void_tracking.py` | each void followed back from `--start-snap` | `--link`, `--n-voids` |
| `plot_void_crossbox.py` | several boxes side by side at one smoothing in Mpc | `--sims`, `--smooth-mpc` |

`plot_void_population.py` compares the size function with the model of Sheth & van de Weygaert (2004) in the
volume-conserving form of Jennings, Li & Hu (2013) (`dtfelib/theory.py`), at its own smoothing (`--size-smooth`,
default 5 cells). `plot_void_profiles.py` and `plot_void_profile_evolution.py` fit the stacked density with the
universal profile of Hamaus, Sutter & Wandelt (2014). `plot_void_environment.py` takes `--criterion velocity` (the
default: net infall in the shell at 2 to 3 R_v), `density-smoothed` or `density-raw`. `plot_void_fate.py` follows
the particles within R_v into the next snapshot; a void survives when half of them end in one descendant
(`lagrangian.SURVIVE_FRAC`). It caches its links; `--ref-snap` (default 33) also links that snapshot to the last.

| tool (`python/tools`) | what it makes | main options |
|---|---|---|
| `merge_HDF5.py` | h-free combined files from downloaded chunks | `--sim-dir`, `--snaps`, `--trees`, `--add-fields`, `--verify` |
| `convert_ic_units.py` | a TNG initial-conditions file in h-free units | input and output path |
| `make_image_plane.py` | a `--sample-points` image plane and its JSON sidecar | `--sim`, `--nu`, `--planes` |
| `compare_deposits.py` | a sampled grid set against an exact one | `--prefix-a`, `--prefix-b` |
| `void_outlines.py` | one snapshot's voids as JSON (Explore's outlines) | `--sim`, `--snap`, `--csv` |

### Voids

The void catalogue is built from the density contrast smoothed with a Gaussian of 10 cells
(`SMOOTHING_SIGMA_CELLS` in `python/config.py`), a different physical length in each box. Its voids are the local
minima, and the Hessian at each gives an ellipsoid. Each void has two sizes:

- R_eff = (abc)^(1/3) of the ellipsoid fit, which gives the shapes and the resolution cut: a void is well resolved
  when its largest semi-axis is within 10 smoothing lengths and half the box, its smallest is at least one cell and
  its axis ratio is at most 10 (`config.ELLIPSOID_CUT_RULES`);
- R_v, where the spherically averaged smoothed density first rises back to the mean, searched out to 24 smoothing
  lengths within half the box. `catalog.R_V_DEFINITIONS` also offers `delta08` (enclosed contrast -0.8) and
  `ridge` (the compensation ridge): `catalog.r_v_mpc(cat, "delta08")`.

The scripts use R_v and the distinct voids: a void whose centre lies inside a deeper void is left out
(`--keep-overlaps` keeps it). `DTFE_VOID_GATE=none` drops the cut's largest-axis clause from the samples and tags
the outputs `gatenone`. To compare boxes at one physical scale use `--smooth-mpc 2.1625`
(`config.CROSSBOX_SMOOTHING_MPC`) or `DTFE_SMOOTH_MPC=2.1625`; the products get their own names, and `analyze.py`
refuses `compute`, `plot` and `all` under it. A cached catalogue made with other radius settings is measured again
on first load. The -0.8 radius of a smoothed minimum follows the smoothing length;
`--size-rv-definition delta08raw` measures it on the raw density.

### dtfelib

`dtfelib` is the package every script reads the grids through. `FieldSet` (`io.py`) loads a snapshot's grid set of
either estimator, gives densities as ρ/ρ̄ and memory-maps grids too large to read; `Estimator` (`estimator.py`)
is the Python client of a program's `--serve` mode; `cli.py` holds the shared parser; `pipeline.py` and
`catalog.py` build the void catalogue, and the other modules its statistics, spectra, linear theory, group
catalogues and merger trees.

```python
from dtfelib import FieldSet

fs = FieldSet("/path/to/data/TNG100/TNG100-3-Dark/snapdir_099")   # the estimator found from the files
rho = fs.density(units="mean")                  # rho/rho_bar for both estimators
v = fs.load("velocity")                         # (N, N, N, 3) in km/s
div = fs.load("divergence", mean_field=True)    # div<v> for both estimators
if fs.has("streams"):
    streams = fs.load("streams")                # PS-DTFE only
print(fs)                                       # estimator, grid, box, z, rho_bar
```

`FieldSet(path, method="ps")` or `method="dtfe"` picks the estimator, and `prefix=` another grid set. For a
PS-DTFE grid set:

- `load(name, mean_field=True)` gives the derivatives of the mean velocity, ∇⟨v⟩, as standard DTFE stores them: the
  stored grids of a run with the face gradient (`fs.face_gradient` says whether it had one), else FFT derivatives of
  the velocity grid; `fields.mean_field_plane` computes them for one plane without the whole cube's FFTs.
  `load(name, stream_average=True)` gives the stream average ⟨∇v⟩ of either kind of run.
- `velocity_single_stream()` is the velocity with NaN outside the single-stream cells, and
  `dtfelib.io.single_stream_mask(streams, hidden)` the mask itself (`.streams` at 1 and bit 1 of `.hidden_streams`
  clear).
- `fold_mask()` gives the fold-caustic cells; `caustic_is_fold`, `caustic_is_cusp`, `caustic_is_swallowtail` and
  `caustic_collapse_max` in `dtfelib.io` decode the bits of a `caustic_class` grid.

An `Estimator` builds a snapshot's tessellation once and then evaluates the fields at any points; every answer
equals what `--sample-points` writes for the same points:

```python
from dtfelib import Estimator

with Estimator("/path/to/snapshot.hdf5") as est:    # PS-DTFE; the tessellation is built here, once
    f = est(points)                                 # points: (N, 3) array in Mpc
    print(f.density, f.velocity, f.streams)         # one value per point
    cube = est.grid(xs, ys, zs)                     # the fields on the grid xs x ys x zs
```

`phase_space=False` runs standard DTFE (`padding=` sets its `--padding`, 15 mean spacings by default),
`tessellation_cache=` keeps the tessellation in a folder for the next session, `partition=` serves a composite of
partition tessellations (with a cache, at most `resident=` of them in memory: `--serve-resident`), and
`Estimator.from_arrays(...)` takes the particles from numpy arrays. Velocities come back in peculiar km/s
(`peculiar=False` returns the program's raw units).

### The Python environment

`scripts/install.sh` installs `dtfelib` editable (`--no-python` skips it; the [README](../README.md#building) gives
the command by hand). It needs Python 3.10 or later with numpy, h5py, scipy and matplotlib. The plot scripts and
tools also run directly from any folder. The launcher runs them with `PYTHON` when it is set, else with the `python3`
on `PATH`; the scripts read `DTFE_DATA_ROOT` and `DTFE_SIM` as under [Analysis](#analysis), and these variables:

| variable | meaning | default |
|---|---|---|
| `DTFE_FIGURES_ROOT` | where every figure goes (the scripts, `analyze.py`, the launcher) | `python/figures` |
| `DTFE_CACHE_DIR` | the analysis cache's folder | the recorded folder, else `python/cache` |
| `DTFE_SMOOTH_MPC` | the void smoothing in Mpc (void scripts, `analyze.py export`) | unset: 10 cells |
| `DTFE_VOID_GATE` | `none` drops the cut's largest-axis clause from the void samples | `ellipsoid` |
| `DTFE_PY_RAM_GB` | the available RAM in GB for `FieldSet.load` | detected |
| `DTFE_PY_LOAD_FRAC` | the share of that RAM one grid may take before it is memory-mapped | 0.5 |
| `DTFE_FATE_WORKERS` | `plot_void_fate.py`'s parallel chunks (1 = serial) | up to 4 |
| `DTFE_FATE_CHUNK` | `plot_void_fate.py`'s particles per read | 8000000 |
| `DTFE_FREE_GB` | the free memory in GB for the void catalogue build and the fate census | detected |
| `DTFE_BIN_DIR` | the folder holding the programs, for `Estimator` | the repository root, then `PATH` |
| `DTFE_GUI_CUBE_GB` | the memory in GB Explore keeps whole grids in | a quarter of the memory, 1.5 to 16 |
