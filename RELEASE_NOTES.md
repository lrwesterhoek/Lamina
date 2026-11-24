# Release notes

What a user of an earlier version must know, newest release first, by topic. How to run the programs is in the
[user guide](docs/user-guide.md), the launcher and the analysis tools in
[launcher-and-analysis.md](docs/launcher-and-analysis.md), and the measured run times in
[timings.md](docs/timings.md).

## Changes since 1.0.0

### Before you upgrade

- Every option, variable and knob that was renamed is gone under its old name. An old option is an unknown option to
  the programs, the drivers and the python scripts; an old environment variable is not read at all, without a word.
  Check your command lines, `--config` files, job scripts and shell profiles against the table at the end.
- Rebuild the programs when you update the scripts: the drivers of this version need the programs of this version.
  `scripts/install.sh` rebuilds both with the best backend for the machine; the make targets are in the
  [developer guide](docs/developer-guide.md).
- Grid sets made by an older version are recomputed by the pipeline (*Existing data* below).

### Two modes per estimator: Exact and Fast

Each program computes its `_a` cell averages in one of two modes: `--exact`, the default, or `--sampled`.

| | Standard DTFE | PS-DTFE |
|---|---|---|
| Exact (`--exact`, the default) | the interpolant's exact cell average | each tetrahedron's exact share of a cell |
| Fast (`--sampled`) | a Monte-Carlo cell average | the deposit at one point per cell axis |

In 3D the velocity derivatives are exact face integrals in both modes of standard DTFE and in PS-DTFE's Exact (*the
face gradient*, below); PS-DTFE's Fast takes the stream average of each stream's own gradient, and its per-cell
values are noisy, so smooth its grids before use.

- `DTFE_EXACT_AVERAGE` pins the mode in the environment (`1` exact, `0` sampled); a flag on the command line wins. A
  sampling count without `--exact` (`--subsamples`, or standard DTFE's `--average-method`) selects the sampled mode.
- PS-DTFE's sampled deposit takes one point per cell axis by default (it took 3). `--subsamples N` sets N per axis, so
  `--sampled --subsamples 3` restores the earlier count. Standard DTFE's `--subsamples k` is also a count per axis:
  k^3 points per cell, k^2 in 2D.
- Where the exact average cannot run (`velocityStd_a`, a `--redshift-cone` grid, a MY_SCALAR build, the other
  estimators) standard DTFE samples by itself. The run log's header says which average ran and why (standard DTFE's
  `volume averaging` line, PS-DTFE's `grid deposit` line).
- The launcher: each run tab has one Quality choice, Exact (the default) or Fast. The controls it replaced are under
  View > Advanced Settings, off by default.

### The program picks its device, run mode and GPU share

- **Device.** `--gpu` asks for the GPU, `DTFE_GPU` in the environment pins the device (`0` the CPU, `1` as `--gpu`);
  without either, or with `DTFE_GPU=auto`, the program picks. PS-DTFE deposits on the GPU of a GPU build wherever its
  deposit runs there: averaged or not, exact or sampled (not with `--ps-caustic-cusps`, `--serve` or a scalar of more
  than one component). Standard DTFE runs its exact average on the GPU and its sampled average on the CPU. A pass
  whose sub-grid holds 2^32 cells or more deposits on the CPU and says so. The header's device line (PS-DTFE's
  `PS deposit`, standard DTFE's `'_a' interpolation`) names the device and the reason.
- **Run mode.** `--run-mode gentle|balanced|performance` (or `DTFE_RUN_MODE`) sets how hard a run leans on the
  machine: its threads, the length of its GPU command buffers and, in gentle on Linux, a lower priority. The default
  is balanced, and performance inside a batch job (`SLURM_JOB_ID` or `PBS_JOBID` set). The run log's `run mode` line
  says what was set and why.
- **GPU share.** PS-DTFE's `--gpu-share` (default `auto`) lets a partitioned GPU run of the exact deposit put part
  of its partitions on the free cores at the same time: `auto` does so in performance only, `mixed` in every mode, a
  fraction from 0 to 1 sets the share. A mixed run's grids differ from an all-GPU run's at float rounding;
  the mass stays exact.
- **The drivers** pass on what you set: `DTFE_GPU` and `DTFE_RUN_MODE` reach the program as you set them, `-m` sets
  `DTFE_GPU=1`, `--run-mode MODE` is passed when given, `GPU_SHARE` when it is not `auto`.
- **Metal is the only GPU backend.** The CUDA and HIP ports, which never ran on NVIDIA or AMD hardware, are removed:
  `make CUDA=1`, `HIP=1` or `CUDAEMU=1` stops with a message naming the branch that keeps them. Linux builds run on
  the CPU.
- The launcher: a Machine use choice (Gentle, Balanced, Performance; Balanced by default) on every run tab, and in
  the Advanced settings a Device row: Automatic, CPU, GPU or CPU + GPU.
- The launcher has no device rule of its own. On Automatic it starts the program with `DTFE_GPU=auto` and adds no
  `-m` or `--gpu` of its own, and the program picks, a small snapshot included: PS-DTFE's Fast on fewer than a
  million particles runs on the GPU of a GPU build. On a GPU build, before a memory check, the run tab says
  "Runs on: the device the program picks ('Check memory' names it)"; after one, and in the Runs tab, the launcher
  shows the device the program reported. Extra options that give `--gpu-share` without `--gpu` leave the device to
  the program too. The time estimate of an Automatic job takes earlier runs on either device.

### Velocity derivatives: the face gradient

- In 3D the velocity gradient, and the divergence, shear, vorticity and V-web built on it, come by default from exact
  integrals of the interpolant over the cell faces: the gradient of the mean velocity, conservative cell by cell.
  Standard DTFE takes it for every average and device; PS-DTFE with the exact deposit and without
  `--ps-halo-release`. The run log's `velocity gradient` line says which route ran and why.
- PS-DTFE's face runs also write `.a_velDivStreams` (the stream average's divergence) and `.a_massFluxDiv` (the
  divergence of the mass flux), and beside `gradient_a`, `shear_a` and `vorticity_a` the stream average's
  `.a_velGradStreams`, `.a_velShearStreams` and `.a_velVortStreams`. dtfelib reads either kind:
  `FieldSet.load(name, mean_field=True)` or `stream_average=True`.
- `--per-tet-gradient` (or `DTFE_FACE_GRADIENT=0`) keeps each tetrahedron's own gradient, the route of every grid
  made by 1.0.0 and of the 2D builds; in PS-DTFE it is the stream average of each stream's gradient.
- PS-DTFE weights its velocity moments by Eulerian volume in every run, the standard DTFE convention; the dispersion
  stays mass-weighted. In 1.0.0 this took `--ps-volume-weighted`, which the drivers passed; a bare run without it
  gave mass-weighted moments.

### Other defaults and results that changed

- `--lambda-th` is 0.3 without the option (it was 0): a T-web or V-web run without it writes other classes.
- Standard DTFE's `--padding` is 15 mean particle spacings without the option (it was 5); PS-DTFE keeps 5. A run in
  an open box changes near the hull of the particles.
- The drivers and the launcher write four fields by default: density, velocity, divergence and shear. `FIELDS`, or a
  tick in the launcher, adds the others.
- PS-DTFE renumbers every tessellation in a canonical vertex order, so its single-precision grids repeat to the byte
  under the parallel insert. Its grids moved once against 1.0.0's: in the last bits of most cells, by more in a few
  cells at caustics.
- PS-DTFE's exact deposit on the GPU tests a run of a tetrahedron's cells before it clips any, and spends about a
  third less GPU time in that deposit; a PS-DTFE Exact run of TNG100-3-Dark at 512^3 took 7% less time. Its exact
  GPU grids moved once more, in the last bits (a median of 1e-7 of a cell's density); standard DTFE's did not.
- Standard DTFE's single-precision exact grids no longer depend on the split, the thread count or the insertion
  order: every grid sums in one fixed-point unit for the whole run.
- Standard DTFE with the GPU deposit runs a split with one tessellation per partition where that fits, at the coarsest
  split that fits (it took the most threads per partition, the split then the coarsest that holds them). One
  tessellation now inserts in parallel and walks its tetrahedra on every core: the GPU deposit's arrays, their order
  and the face gradient's sums, which run beside the GPU deposit. TNG50-3-Dark at 512^3, Exact on the GPU: 3^3
  partitions in 13 min 8 s, where the old choice, 6^3 with a team of 5 threads, took 37 min 40 s; the grids do not
  depend on the split. Without the GPU deposit the choice is unchanged.
- Standard DTFE hands freed memory back on macOS (user guide, section 11): it turns the allocator's cache of freed
  blocks off when it starts, returns each partition's tessellation memory after the partition and releases the GPU's
  buffers at once, so a split run's peak no longer grows by blocks it cannot reuse (TNG50-3-Dark at 512^3 in 27
  partitions: 51.2 GB before, 29.6 GB now). `MallocLargeCache=1` keeps the system allocator's cache.
- Standard DTFE's memory model books the run item by item (the tessellation, the insert's copies, the GPU deposit's
  arrays and grids, the post-processing without the freed particles) where it had fitted constants: its predictions
  were 12-41% above the measured peaks with the memory handed back, and are 4-25% above them now (TNG50-3-Dark and
  TNG100-3-Dark at 512^3, a 256^3 Zel'dovich box). A split run's look-ahead is checked against the auto-tuner's
  count of the partition that needs the most, where it planned, and no longer against a uniform guess that took a
  small dense partition's padding for its particles (TNG50-3-Dark at 4^3 kept no look-ahead: 20.86 GB guessed, 9.36
  GB counted). Under `--scratch-dir` it books the full grids too small for a scratch file (under 1 GB) in memory, and
  on macOS its prediction includes the start-up zero fill of the largest file-backed grid beside them and the
  particles, as PS-DTFE's does (TNG100-3-Dark at 1024^3: 42.05 GB predicted, 42.08 measured, where it said 32.09); the
  `SUMMARY` says when the peak was set before the partition loop. After the loop it books the derived grids under 1
  GB beside them, which it left out (a 256^3 Zel'dovich box at 512^3, all fields in 4^3 partitions: 5.97 GB for the
  post-processing, 5.85 measured, where it booked 1.14). A partition's own blocks of 1 GB or more, which
  `--scratch-dir` puts in files too, stay booked as memory: they are its working set, in the page cache while it runs,
  and a plan that left them out chose one domain on a team of 4 for that box at 512^3 (39.64 GB and 2m10s, against
  11.65 GB and 1m46s for 2^3 partitions); the `SUMMARY` says they are in the prediction but not in the footprint.
- The standard auto-tuner replays the run's split cell by cell (it cut on bins of 4 cells at 512^3, and its count of
  the busiest partition was off by up to 21%).
- The parallel (TBB) Delaunay insert is in every binary built with TBB and engages by itself for tessellations of at
  least a million points where cores are free; `--sequential-triangulation` keeps the sequential insert.
- `--ps-lagrangian-input` needs an HDF5 Gadget snapshot (`--input-format 105`), the only reader that gives the
  particle IDs the initial positions are matched by; with another input type the program stops.
- `-h` is a short list of what the two modes need; `--full-help` lists every option in groups, the developer options
  in a group of their own.
- The closing warning that counts the deposits meant for the GPU that ran on the CPU reads "were meant for the GPU and
  ran on the CPU instead" (it said "asked for the GPU"), since the program may have picked the GPU itself.
- PS-DTFE's `--gpu` and `--exact` help say which grids match the CPU deposit to float rounding (the cell averages, the
  densities, the stream counts and the flags) and which do not: in the unaveraged grids the velocity-derivative grids
  differ in single cells, the vorticity most. The `--sample-points` help says the points are in Mpc, and PS-DTFE's
  `--interlace` help and its error without `--periodic` say that PS-DTFE needs `--periodic` for it, as it takes no
  `--region`.

### Fixes that change results

- PS-DTFE's exact deposit on the GPU wrote spurious velocity dispersions for thin tetrahedra with steep velocity
  gradients. Such tetrahedra now go to the CPU's double-precision deposit: recompute dispersion grids that 1.0.0
  deposited exactly on the GPU.
- The GPU exact deposits integrate each tetrahedron-cell piece with exact quadrature in place of a float32 clip, and
  agree more closely with the CPU reference.
- Standard DTFE's exact average forms each tetrahedron in the full grid's frame in double precision: the strips of
  rounding in the velocity gradient of split GPU runs are gone.
- PS-DTFE: a tetrahedron deposited whole in the cell of its centroid (one that catches no sample point, in the
  sampled deposit or too thin to clip in the exact one; with `--ps-halo-release`, a released one) finds that cell in
  double precision from its vertices' differences, the same cell in every periodic image wherever the images are exact
  shifts of each other, as they are in a box that starts at 0 (a snapshot's own box). The cell followed the image that
  deposits the tetrahedron, which the split and the padding choose, so a split run and one domain could put such a
  tetrahedron in neighbouring cells. The grids move in the cells where such a centroid lies within a rounding of a
  cell plane: the whole tetrahedron's mass and moments go to the neighbouring cell. In a Fast run of 94 million
  particles on a 512^3 grid that is 26,325 of the 134 million density cells, and the total mass stays the same to
  rounding. The tidal field's grids, computed from the whole density field, follow it everywhere: the eigenvalues
  move in nearly every cell, by up to 7.6e-5 of their peak, and the classification changes in 16,173 cells.
- PS-DTFE's memory prediction for a plan of one partition at a time on the GPU under `--scratch-dir` books the input
  reader's freed buffers that macOS's allocator keeps (`readkept_gb` on the `--auto-tune-report` line); it predicted
  less than such a run used. Such a plan now predicts more and reports over budget sooner, in the launcher's memory
  check too; the split, the partitions in flight and the look-ahead are chosen as before; the room the ordered merge
  parks finished partitions in shrinks by the same amount. For one such run, of 94 million particles on a 512^3
  grid, the prediction went from 28.5 to 36.0 GB, against a peak memory footprint of 29.2 GB.
- Standard DTFE's load-balanced split cut each column of partitions along z by the particles of another column (the
  last x slab's), so its partitions were uneven: on TNG50-3-Dark at 3^3 the busiest held 16.6 million padded particles
  where the even split gives 14.5 million. The exact grids do not depend on the split; a split run's Fast density,
  velocity and T-web eigenvalues move at float rounding (by at most 3e-6 of a grid's largest value in a test of 16.8
  million particles on a 128^3 grid in 3^3 partitions), its classes and velocity derivatives not.
- Standard DTFE: an unaveraged grid or redshift-cone row that left the convex hull came out zero for the rest of the
  row (non-periodic runs with `--no-padding-test`); `--average-method 2` drew different points from one `--seed`
  under GCC and Clang; a library sample outside `--region` was dropped, which shifted the entries after it.
- `scripts/download_snapshots.sh` fetches the TNG snapshot chunks again.

### Removed

- PS-DTFE's whole-number merge (`--ps-merge fixed-point`) and its run-wide units (`DTFE_PS_RUN_UNITS`, with the test
  hooks `DTFE_TEST_FXMERGE_SHIFT`, `DTFE_TEST_FXMERGE_SUM_LIMIT` and `DTFE_DEBUG_FACE_TWINS`). `--ps-merge` takes
  `index-order` (the default) or `completion-order`.
- The four PS-DTFE conventions that no longer chose anything: `--ps-volume-weighted`, `--ps-linear-deposit` (the
  linear-profile deposit), `--ps-stream-density` (the geometric stream density at points) and
  `--ps-point-conventions` (the legacy point values); the drivers' `PS_VOLUME_WEIGHTED`, `PS_LINEAR_DEPOSIT` and
  `PS_POINT_CONVENTIONS`; `dtfelib.Estimator`'s keyword `stream_density`.
- Standard DTFE's `--average-method 3`: methods 1 and 2 remain. [docs/DTFE_user_guide.pdf](docs/DTFE_user_guide.pdf),
  the original program's guide, still documents method 3 and the original option names.
- The float sums: `DTFE_GPU_FLOAT_MASS`, `DTFE_GPU_FLOAT_MOMENTS`, `DTFE_GPU_FLOAT_GRIDS`.
- The "as before" switches: `DTFE_MC_LEGACY`, `DTFE_PS_VERTEX_ORDER_LEGACY`, `DTFE_RUN_UNITS_LEGACY`,
  `DTFE_TESSELLATION_CACHE_LEGACY`, and `python/tools/upgrade_tess_cache.py`.
- The GPU face kernel, `DTFE_GPU_FACES` and `DTFE_FACE_F32`: the face sums are formed on the CPU beside the GPU
  deposit.
- The off switches of four defaults: `DTFE_FACE_WALK_EARLY`, `DTFE_LOOKAHEAD_TEAM`, `DTFE_AUTO_SPAN`,
  `DTFE_DYNAMIC_CONCURRENCY` (`--max-concurrent` sets a fixed number of partitions at once).
- Thirteen tunables, now fixed: `DTFE_AUTO_PTS_STREAMS`, `DTFE_AUTO_T_VERTEX`, `DTFE_AUTO_T_SAMPLE`,
  `DTFE_AUTO_T_POINT`, `DTFE_DEPOSIT_MEM_GB`, `DTFE_GPU_CACHE_K`, `DTFE_GPU_ITEM_SAMPLES`, `DTFE_GPU_TABLE`,
  `DTFE_GPU_EXACT_CELLS`, `DTFE_GPU_SHARE_RHO`, `DTFE_SERVE_INDEX`, `DTFE_TESSELLATION_CACHE_THREADS`,
  `DTFE_TRIANGULATION_MIN_POINTS`.
- Developer experiments: the planner's time objective (`DTFE_AUTO_TIME`, `DTFE_AUTO_TIME_GAIN`,
  `DTFE_AUTO_T_STD_PART`, `DTFE_AUTO_T_PS_PART`), standard DTFE's shared GPU grid (`DTFE_GPU_SHARED`,
  `DTFE_GPU_SHARED_GB`), the GPU timing switches (`DTFE_GPU_DEBUG_NOWRITE`, `DTFE_GPU_DEBUG_NOATOMIC`,
  `PS_GPU_DEBUG_NOATOMIC`), the CPU share of the GPU's tetrahedra (`PS_GPU_CPU_SHARE`, `PS_GPU_CPU_SHARE_END`) and
  the `*_METAL_*` spellings of the GPU knobs.
- The CUDA and HIP build: `CUDA`, `HIP`, `CUDAEMU`, `GPU_ARCH`, `CUDA_PATH`, `ROCM_PATH`, `DTFE_GPU_EMU_WARP`.
- Python: `merge_HDF5.py --fix-units`, `dtfelib.fields.pseudo_phase_space_density`, `PointPlane.sample_grid_field`
  and `python/dtfelib/runlog.py`.
- Every old spelling: the table below.

### The source tree

- The three largest source files, `src/CGAL_triangulation/ps_interpolation.cc`, `src/DTFE.cpp` and
  `src/auto_tune.cc`, are split into parts, each file's parts in a folder of their own; each file still compiles as
  one unit and includes its parts in order ([developer guide](docs/developer-guide.md#source-layout)).
- The branches that only the removed switches reached are gone from the CPU deposit loops and from standard DTFE's
  GPU kernel. PS-DTFE's kernel keeps them: without them the second pass of its exact deposit took about a tenth
  more device time ([docs/decisions.md](docs/decisions.md)).
- The working plan (`docs/cleanup-plan.md`) and the long change log (`change_log.txt`) are no longer in the tree.
  These notes say what changed for a user and [docs/decisions.md](docs/decisions.md) the reasons behind the defaults.

### Existing data, logs, caches and saved jobs

- **Run logs of an older version.** A run log without the header's `run mode` line was written by an older version
  of the programs, and its settings are not read. `scripts/run_ps_pipeline.sh` recomputes such a snapshot and says
  so ("written by an older version of the programs, recomputing"). The launcher's Runs tab lists such a run with
  "older version" for its settings, offers no Load settings for it, leaves it out of the time estimates and warns to
  recompute it before use; the Pipeline tab counts it as to compute. A snapshot without a run log is judged by its
  file times.
- dtfelib still reads a grid set's velocity weighting from its run log (the word `--ps-volume-weighted` in an older
  log means volume-weighted moments) and the older output name `.unresolved` as `.hidden_streams`.
- **Tessellation caches** of an older format are not read: the run builds the tessellation and writes a new file.
  Delete the old files yourself when you no longer need them.
- **Checkpoints** of an older program are never resumed: `--resume` refuses them. Run without it (the drivers:
  `DTFE_RESUME=0`) to start over.
- **Saved launcher jobs.** Settings saved before the two modes, and old option names in a saved Custom job or in
  saved figure options, are not converted: a setting the launcher does not know is ignored and the default stands. A
  job whose extra options hold an old name fails, and the launcher's advice says the option does not exist.
- **Shell profiles and job scripts.** An old variable such as `PS_GPU=0`, `DTFE_PS_CHECKPOINT=0`,
  `DTFE_NO_PROGRESS=1` or `PY=...` no longer acts, and nothing says so.

### Old names and what to type now

Every name in the left column is gone. The programs' options, both programs (`--input` and `--output` are now
refused as ambiguous):

| old | now |
|---|---|
| `--exact-average` | `--exact` |
| `--full_help` | `--full-help` |
| `--lambda_th` | `--lambda-th` |
| `--input` | `--input-format` (`-i` stays) |
| `--output` | `--output-format` (`-o` stays) |
| `--inputFile` | `--input-file` |
| `--outputFile` | `--output-file` |
| `--MpcUnit` | `--mpc-unit` |
| `--regionMpc` | `--region-mpc` |
| `--partNo` | `--partition-index` |
| `--paddingMpc` | `--padding-mpc` |
| `--noTest` | `--no-padding-test` |
| `--density0` | `--mean-density` |
| `--redshiftCone` | `--redshift-cone` |
| `--origin` | `--cone-origin` |
| `--extensive` | `--scalars-extensive` |
| `--randomSample` | `--particle-fraction` |
| `--options` | `--reader-options` |
| `--NGP`, `--CIC`, `--TSC`, `--PCS`, `--SPH`, `--Voronoi` | the same words in lower case |

Standard DTFE only:

| old | now |
|---|---|
| `-m`, `--method` | `--average-method` |
| `--approxPSD` | `--approx-phase-space-density` |
| `-s N`, `--samples N` | `--subsamples k` (per cell axis; k^3 = N, k^2 in 2D) |

PS-DTFE only:

| old | now |
|---|---|
| `--ps-gpu` | `--gpu` |
| `--ps-exact-deposit` | `--exact` |
| `--ps-no-overlap` | `--no-overlap` |
| `--avg-subsamples` | `--subsamples` |
| `--lagrangianInput` | `--ps-lagrangian-input` |
| `--ps-lag-padding` | `--ps-lagrangian-padding` |
| `--per-stream`, `--per-stream-ids` | `--ps-per-stream`, `--ps-per-stream-ids` |
| `--pts-lagrangian` | `--ps-per-stream-lagrangian` |
| `--ps-window` | `--grid-window` |
| `--ordered-merge` | `--ps-merge index-order` (the default) |
| `--completion-order-merge` | `--ps-merge completion-order` |
| `--fixed-point-merge`, `-m`, `--method`, `-s`, `--samples` | nothing: leave the word out |
| `--ps-volume-weighted`, `--ps-linear-deposit`, `--ps-stream-density` | nothing |
| `--ps-point-conventions`, `--pts-conventions` | nothing |

The programs' environment:

| old | now |
|---|---|
| `DTFE_PS_OVERLAP` | `DTFE_OVERLAP` |
| `DTFE_PS_CHECKPOINT` | `DTFE_CHECKPOINT` |
| `PS_GPU_TIMING` | `DTFE_GPU_TIMING` |
| `PS_MASS_AUDIT` | `DTFE_MASS_AUDIT` |
| `PS_DEBUG_DEGREES` | `DTFE_DEBUG_DEGREES` |
| `PS_EXACT_ITEMS` | `DTFE_GPU_ITEMS` |
| `DTFE_TBB_THREADS` | `DTFE_TRIANGULATION_THREADS` |
| `DTFE_AUTO_PTS_VELGRAD` | `DTFE_AUTO_PTS_VEL_GRAD` |
| `DTFE_TESS_CACHE_ZLIB`, `_CGAL_IO`, `_FINGERPRINT` | `DTFE_TESSELLATION_CACHE_ZLIB`, `_CGAL_IO`, `_FINGERPRINT` |
| `DTFE_NO_PROGRESS=1` | `DTFE_PROGRESS=0` |

The drivers' knobs and options:

| old | now |
|---|---|
| `DTFE_GRID_SIZE`, `DTFE_PADDING`, `DTFE_SPACE_CHECK` | `GRID_SIZE`, `PADDING`, `SPACE_CHECK` |
| `PY` | `PYTHON` |
| `DTFE_METAL`, `PS_GPU`, `PS_METAL` | `DTFE_GPU` |
| `DTFE_EXACT`, `PS_EXACT` | `DTFE_EXACT_AVERAGE` |
| `AVG_SUBSAMPLES` | `SUBSAMPLES` |
| `TESS_CACHE` | `TESSELLATION_CACHE` |
| `PS_RESUME` | `DTFE_RESUME` |
| `PS_PARALLEL_TRI` | `PARALLEL_TRIANGULATION` |
| `PS_LAG_PADDING` | `PS_LAGRANGIAN_PADDING` |
| `LAGRANGIAN_INPUT` | `PS_LAGRANGIAN_INPUT` |
| `WINDOW` | `PLANE_WINDOW` |
| `ORDERED_MERGE=1`, `ORDERED_MERGE=0` | `PS_MERGE=index-order` (the default), `PS_MERGE=completion-order` |
| `--exact-average` | `--exact` |
| `FIXED_POINT_MERGE`, `PTS_CONVENTIONS`, `PS_POINT_CONVENTIONS` | nothing |
| `PS_VOLUME_WEIGHTED`, `PS_LINEAR_DEPOSIT` | nothing |

The python scripts:

| old | now |
|---|---|
| `--method` | `--estimator` |
| `--output` | `--out` |
| `--r-v-definition` | `--rv-definition` |
| `--sigma-mpc` | `--smooth-mpc` (void_outlines.py) |
| `--snap` | `--start-snap` (plot_halo_tracking.py, plot_void_tracking.py) |
| `--voids` | `--per-void` (plot_caustic_skeleton.py) |
| `--subhalo`, `--top` | `--subhalos`, `--n-halos` (plot_halo_tracking.py) |
| `--nbins` | `--n-bins` (plot_pk_compare.py) |
| `--fields`, `--smooth` | `--maps`, `--smooth-px` (plot_pointeval.py) |
| `--smooth-derivatives` | `--smooth-derivatives-px` (plot_pointeval.py) |
| `--prefix` | `--root` (plot_pointeval_panels.py); its `--project mean` is `--project slab` |
| `--z` | `--redshifts` (plot_void_convergence.py, plot_void_crossbox.py) |
| `--size-rv` | `--size-rv-definition` (plot_void_population.py) |
| `--a`, `--b` | `--prefix-a`, `--prefix-b` (compare_deposits.py) |
| `--u0 --u1 --v0 --v1` | `--plane-window U0 U1 V0 V1` (make_image_plane.py) |
| `--data-dir` | `--sim-dir` (merge_HDF5.py) |
| a criterion with `_` | the same with `-` (plot_void_environment.py) |
| a comma-joined list | the values separated by blanks |
| positional snapshot numbers | `--snaps` (analyze.py, merge_HDF5.py) |
| `DTFE_SIGMA_MPC` | `DTFE_SMOOTH_MPC` |

## Earlier versions

**1.0.0 (Lamina, 2026-10-23).** The first release under the name Lamina: the standard program DTFE, the phase-space
program PS-DTFE, the python package dtfelib and the launcher. It brought the double-precision and 2D builds, the GPU
deposits, exact fixed-point sums, the auto-tuner, `--scratch-dir`, `--tessellation-cache`, `--resume`, `--serve`,
the drivers' run logs and the void toolkit.

**1.3.0 (never tagged).** The phase-space work Lamina grew from: PS-DTFE (a Lagrangian tessellation followed to the
present day, with stream counts, cell averages and partitions), the exact conservative deposit (Powell & Abel 2015)
and the exact cell average of standard DTFE, the halo release, the caustic flags, point evaluation in both programs,
and the scripts that download and merge TNG snapshots.

**1.2.2 and older: the DTFE of Cautun & van de Weygaert (2011).** 1.2.2 gave each input type its own reader under
`src/io` and added a reader for the FORGE HDF5 snapshots. 1.2.1 made `--CIC` work (it had done NGP) and added NGP.
1.2 added the HDF5 Gadget reader, the options that choose the input and output file types, and the redshift-space
option. 1.1.2 fixed `--partition` with 27 or more partitions.
