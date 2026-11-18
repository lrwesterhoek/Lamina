# Developer guide

This guide is for you if you change the code: the two estimators `DTFE` and `PS-DTFE`, their batch drivers, the
launcher and the Python library. It covers the build, where the code lives, how a run moves through it, the rules a
change must keep, the tests, and the steps to add an option, a field or a test.

The other documents: the [README](../README.md) (start here), the [user guide](user-guide.md) (running the programs
and the drivers, every option, the user's environment variables), [the launcher and the analysis
tools](launcher-and-analysis.md), [metal/README.md](../metal/README.md) (the GPU kernels and their knobs),
[demo/COMMANDS.md](../demo/COMMANDS.md) and [decisions](decisions.md) (the choices that shape the code, with their
reasons, and what was considered and not done).

## Building

Install the dependencies first (the [README](../README.md#prerequisites) says how). The Makefile works on macOS
(Homebrew paths) and Linux
(system paths):

| command | what it does |
|---|---|
| `make DTFE` | builds the standard estimator `./DTFE` |
| `make PS-DTFE` | builds the phase-space estimator `./PS-DTFE` |
| `make library` | builds `libDTFE.dylib` or `libDTFE.so` (standard DTFE), its headers in `DTFE_include/` |
| `make clean` | removes every binary set, the library and the objects |
| `make check` | a quick test of the built `./DTFE` and `./PS-DTFE` with no simulation data; builds nothing (*Tests*) |
| `make deps-check` | checks every required header and prints the install command for what is missing |
| `make test-platform` | prints the detected platform, compiler, paths and flags |

```bash
make DTFE PS-DTFE -j8
```

Each configuration has its own binary names and object folders, so all of them coexist:

| make variables | binaries | object folders |
|---|---|---|
| (none) | `DTFE`, `PS-DTFE` | `o/`, `o_ps/` |
| `DOUBLE=1` | `DTFE-double`, `PS-DTFE-double` | `o_d/`, `o_ps_d/` |
| `DIM=2` | `DTFE-2d`, `PS-DTFE-2d` | `o_2d/`, `o_ps_2d/` |
| `DIM=2 DOUBLE=1` | `DTFE-2d-double`, `PS-DTFE-2d-double` | `o_2d_d/`, `o_ps_2d_d/` |

`DOUBLE=1` makes the type `Real` (`src/define.h`) a `double`: every position, density, field and accumulator, from
the HDF5 read to the float64 output grids. `DIM=2` sets `-DNO_DIM=2` and is always CPU-only. The other variables:

| variable | effect |
|---|---|
| `METAL=1` | the Metal GPU backend (Apple Silicon); see *The GPU backend* |
| `TBB=auto` / `1` / `0` | the TBB-parallel Delaunay insert: compiled in when found (default), required, left out |
| `TBB_STD=0` / `1` | the same for the standard binaries and the library only (default: as `TBB`) |
| `LIBDEFLATE=auto` / `1` / `0` | libdeflate instead of zlib for the tessellation cache (both write plain gzip) |
| `BUILD_MODE=debug` | `-O0 -g3 -DDEBUG` with the address, undefined-behaviour and leak sanitizers |
| `ARCH_FLAGS=...` | CPU tuning only; default `-mcpu=native` on Apple Silicon, `-march=native` elsewhere |
| `EXTRA_FLAGS=...` | compiler flags added to every compile |
| `CC_OVERRIDE=...` | another compiler |
| `GSL_PATH_OVERRIDE=...` | GSL elsewhere; likewise `BOOST_`, `CGAL_`, `MPFR_`, `HDF5_`, `GMP_`, `FFTW_`, `TBB_` |
| `MACOS_SDK_OVERRIDE=...` | the macOS SDK passed as `-isysroot` (else the Makefile finds one that exists) |

The project's C++ objects are compiled with `-std=c++17` and `-frounding-math`; the vendored k-d tree (kdtree2 in
`src/kdtree/`, its licence in `src/kdtree/LICENSE`; the one object with `-ffast-math`) and r3d have flags of their
own. A GCC build adds `-ffp-contract=off`, so GCC fuses no multiply-add across statements and two code paths written
to do the same arithmetic round alike; clang fuses only within one expression. The Makefile has one compile
template, `cxx_rule`, that builds the project's sources for the standard programs, for PS-DTFE and for the library
alike; one link template, `program_rule`, for both programs; and one, `embed_kernel`, for the generated kernel
headers (*The GPU backend*). The vendored code and the Metal hosts have rules of their own. An object's
prerequisites are its `NAME.deps` list in the Makefile, its source first, and the compiler writes a `.d` file beside
each object, so a header edit rebuilds every object that includes it.

**Compile-time options.** The block `OPTIONS_COMMON` in the Makefile sets what both programs compile in:
`-DVELOCITY`, `-DSCALAR` with `-DNO_SCALARS=1` (one scalar component, the only case the GPU deposits),
`-DINPUT_FILE_DEFAULT=105` (Gadget HDF5), `-DMPC_UNIT=1000` (input lengths in kpc), `-DOUTPUT_FILE_DEFAULT=101` (raw
binary grids), `-DOPEN_MP`, `-DTEST_PADDING`, `-DREDSHIFT_SPACE` and `-DTRIANGULATION`. PS-DTFE adds `-DPHASE_SPACE`:
both programs are built from the same sources, and `#ifdef PHASE_SPACE` separates them.

### The GPU backend

`METAL=1` (macOS on Apple Silicon) is the only GPU backend. It defines `DTFE_GPU` (standard build) or `PS_GPU`
(PS-DTFE build), the guards of every GPU call site, compiles the host `src/CGAL_triangulation/dtfe_metal_host.cc` or
`ps_metal_host.cc`, and links the Metal and Foundation frameworks. The hosts use Apple's header-only metal-cpp,
vendored in `third_party/metal-cpp`. Without `METAL=1`, and in every `DIM=2` build, the programs are CPU-only and
`--gpu` falls back to the CPU with a warning.

The kernels become part of the binaries as text. For each program the Makefile's `embed_kernel` writes a header into
the object folder, `o/dtfe_deposit_msl.h` or `o_ps/ps_deposit_msl.h`, holding one raw string literal made of, in
this order:

- standard DTFE: `metal/exact_clip.metal.inc`, `metal/fixed_words.metal.inc`,
  `src/CGAL_triangulation/dtfe_deposit_params.h`, `metal/dtfe_deposit.metal`;
- PS-DTFE: `metal/exact_clip.metal.inc`, `src/CGAL_triangulation/ps_kernel_params.h`, `metal/ps_common.metal.inc`,
  `metal/ps_deposit.metal`.

The host compiles that source with Metal at run time, so the binary needs no file beside it, and a kernel that does
not compile shows up only when a run reaches the GPU, as a fallback to the CPU with a warning. After you edit the
standard kernel, run `tests/dtfe_metal_check.sh`; after you edit the PS-DTFE kernels, build and run the kernel
harness ([metal/README.md](../metal/README.md) describes its tests, the kernels, the exact sums and the retries):

```bash
bash metal/build_prototype.sh
```

```bash
./metal/validate_deposit
```

### TBB, libdeflate and the library

With TBB the programs can insert a large tessellation into CGAL's Delaunay structure in parallel
(`-DPARALLEL_TRIANGULATION`). The program decides per tessellation (`--full-help` gives the rule under
`--parallel-triangulation`); that option and `--sequential-triangulation` override it. The triangulation type
changes with the setting, so it is stamped like the GPU mode. libdeflate changes only the tessellation cache's codec:
every build reads every cache.

`make library` compiles the standard sources again with `-fPIC` (the `*_l.o` objects in `o/`). The library links
`src/scratch_alloc_stub.cc` instead of `src/scratch_alloc.cc` (which replaces the global `operator new` for
`--scratch-dir`) and writes no partition checkpoints (`dtfeIsProgram` in `src/build_info.cc` is true only in the
programs). The compile-time options and the TBB setting decide the data layout of its interface, so
`demo/examples/Makefile`, which builds a program against it, repeats the root Makefile's `OPTIONS_COMMON` and TBB
blocks and its macOS SDK pinning: keep the two the same.

### The build stamps

Every object folder carries stamp files of the configuration its objects were compiled for: `.gpu_mode_metal` or
`.gpu_mode_off`, `.dim_2` or `.dim_3`, `.prec_float` or `.prec_double`, `.tbb_0` or `.tbb_1`, `.deflate_0` or
`.deflate_1`, `.flags_<mode>_<checksum>` (of `BUILD_MODE`, `ARCH_FLAGS` and `EXTRA_FLAGS`) and, under GCC,
`.fpc_off`. When a make line asks for another configuration, the targets `dtfe_gpu_mode_check` and
`ps_gpu_mode_check` first wipe the folder's objects and generated kernel header, so a binary never mixes objects of
two configurations. `.build_mode` holds the make arguments of the folder's mode (for example `METAL=1`), so this
rebuilds without changing the mode; the test suites rebuild the same way (`build_target` in `tests/lib.sh`):

```bash
make DTFE $(cat o/.build_mode)
```

The binaries carry a build stamp too: `src/build_info.cc` is recompiled on every link with the git revision (marked
dirty when the tree has uncommitted changes) and the link time in UTC. `./DTFE --version` prints them, every run log
has them in its `Build:` line, and the checkpoint key holds them, so `--resume` refuses a rebuilt binary.

`scripts/install.sh` is the users' one-command build (`make clean`, then every binary set that was built); while you
work on the code, call make yourself.

## Source layout

| path | holds |
|---|---|
| `src/main.cpp` | the program: options, input, `DTFE()`, interlacing, outputs, the closing summary |
| `src/DTFE.cpp`, `DTFE.h` | `DTFE()` and `DTFE_parallel` (a domain split over the threads), the partition look-ahead, the end of a run (`dtfeFinishFields`); its parts in `src/dtfe/` (*The split units*) |
| `src/user_options.cc`, `.h` | every option (`addOptions`), the short help (`shortHelp`), parsing and resolution |
| `src/auto_tune.cc`, `.h` | the planner's entry points (`autoTunePartitioning`, `autoTuneServe`, `autoTuneGpuShare`), its parts in `src/auto_tune/`; the header: its interface, the memory gate (`PartitionMemoryGate`), the machine queries |
| `src/quantities.cc`, `.h` | the grids (`Quantities`) and the field table `forEachField` |
| `src/input_output.cc`, `src/io/` | the readers (Gadget binary and HDF5, text, raw binary) and the writers |
| `src/web_fields.cc` | divergence, shear, vorticity, the T-web and the V-web, from the complete grids |
| `src/env_knobs.h` | the one reader of environment variables |
| `src/fixed_units.h` | the run-wide fixed-point units and words of the exact sums |
| `src/partition_checkpoint.h` | the checkpoints of a partitioned run and `--resume` |
| `src/output_staging.h`, `run_progress.h` | crash-safe outputs (`.part.<pid>`, a rescue folder); the progress file |
| `src/subpartition.h` | the particles of a sub-box; the partition splits |
| `src/run_mode.h`, `thread_count.h` | the run mode's core budget; thread counts and the shared core pool |
| `src/ps_merge_spill.h`, `ps_globals_record.h` | PS-DTFE: the ordered merge's spill to disk; a snapshot's globals |
| `src/*_interpolation.cc`, `interlacing.cc` | the other estimators (NGP, CIC, TSC, PCS, SPH); interlacing |
| `src/CGAL_triangulation/` | the tessellation and everything that walks it; the files that matter follow |
| `.../triangulation.cpp` | the tessellation: cache load or CGAL insert, vertex densities, the grid passes' dispatch |
| `.../ps_interpolation.cc` | PS-DTFE's grid pass `interpolateGrid_phaseSpace`: the CPU deposit loops, exact and sampled; its parts in `ps_interpolation/` (*The split units*) |
| `.../averaged_interpolation_1.cc` | standard DTFE's `_a` average: exact and Monte Carlo, face gradient, GPU call |
| `.../averaged_interpolation_2.cc` | standard DTFE's averaging method 2 (redshift cone, user sampling points) |
| `.../unaveraged_interpolation.cc` | standard DTFE's point values at the cell centres |
| `.../ps_point_eval.cc`, `ps_serve.cc` | `--sample-points`; `--serve` with its protocol (both programs) |
| `.../tessellation_cache.h` | `--tessellation-cache`: the file format and the cache identity |
| `.../exact_sweep.h`, `face_sweep.h` | the CPU's clipping of tetrahedra against grid cells; the face integrals |
| `.../ps_metal_host.cc`, `dtfe_metal_host.cc` | the Metal hosts: device, pipelines, chunked dispatch, retries |
| `.../ps_kernel_params.h`, `dtfe_deposit_params.h` | the parameter blocks the kernels and the hosts share |
| `metal/` | the kernels (`*.metal`, `*.metal.inc`) and the kernel harness |
| `scripts/` | the drivers (`run_dtfe.sh`, `run_ps_dtfe.sh`, `run_ps_pipeline.sh`), `config.sh`, install, `gui.sh` |
| `python/dtfelib/` | the analysis library: grid loaders (`FieldSet`), the interactive `Estimator`, voids |
| `python/gui/` | the PySide6 launcher |
| `python/plot/`, `python/tools/` | the figure scripts; tools (merging, unit conversion, image planes, deposit A/B) |
| `python/analyze.py`, `config.py`, `selftest.py` | the analysis driver, its configuration, its smoke test |
| `tests/`, `tests/dev/` | the suites, shared helpers, references (`tests/reference/`); the local tools |
| `third_party/` | vendored: `r3d` (exact tetrahedron-cell clipping, C99) and `metal-cpp` |
| `demo/` | a small snapshot, `COMMANDS.md`, the library example |

**The split units.** `src/DTFE.cpp`, `src/auto_tune.cc` and `src/CGAL_triangulation/ps_interpolation.cc` are each
the root of one translation unit, the file the Makefile compiles, one object per build, and each `#include`s in order
the parts in the folder of its name. A part is a `.cc` file that is compiled only as part of its root's object, never
alone, and holds no `#include`: a new include goes into the root. A part sits inside the guards and namespaces that
enclose its `#include` line in the root, and its header comment names them. The Makefile finds each folder's parts
by a wildcard and lists them as prerequisites of the root's objects, so a new part needs only its line in the root;
an object list never comes from a glob over the sources. The parts, in each root's order (paths under `src/`):

| part | holds |
|---|---|
| `dtfe/ps_partition_plan.cc` | PS-DTFE: the Lagrangian bounding box, the seam report, the Lagrangian padding, the partition plan and its costs (`PSPartitionPlan`, `psPartitionCosts`), the GPU share (`psResolveGpuShare`) |
| `dtfe/setup.cc` | the run's setup: standard DTFE's run-wide units (`runUnitsSetup`), `DTFE_setup`, the padding, the thread sub-domains (`dtfePrepareSubdomains`), `psUnwrapEulerian` |
| `dtfe/ps_partition_setup.cc` | PS-DTFE's partitioned run up to its loop: the cloud's normalization, the periodic setup, the window, the composite serve, the grids, the deferred particle read |
| `dtfe/ps_partition_loop.cc` | PS-DTFE's partition loop (`psRunPartitionLoop`) and its staged ordered merge (`PSMergeStaging`, `psMergeInTurn`) |
| `dtfe/ps_partitioned_run.cc` | PS-DTFE: the close-out of the summed moments, the aggregate report, `psPartitionedRun` |
| `dtfe/std_partitioned_run.cc` | standard DTFE: the composite serve of `--serve --partition` and `dtfePartitionedRun` |
| `auto_tune/model.cc` | the memory model: its constants, the sample-point file's count and bounds, the bytes per vertex, cell and grid both planners share |
| `auto_tune/std_plan.cc` | standard DTFE: the density replay, the look-ahead's bytes (`stdLookAheadBytes`), the planner (`AutoTuneStdPlanner`) |
| `auto_tune/ps_inputs.cc` | PS-DTFE's planner inputs (`AutoTunePSInputs`): the partition spans, the sample points' streams and bytes, the scratch space check |
| `auto_tune/ps_planner.cc` | PS-DTFE's planner (`AutoTunePSPlanner`): the memory model of a split, the report (`finishReport`), the phases that choose the split |
| `CGAL_triangulation/ps_interpolation/fixed_point.cc` | the CPU deposit's fixed-point sums (`PSFxSink`, `PSFxWords`, `psFxToReal`) and, in 3D, the face sums' deposit (`psDepositFaces`) |
| `CGAL_triangulation/ps_interpolation/tet_tests.cc` | the tetrahedron-box overlap test (`TetBoxSAT`), the CPU deposit's thread count, the samples beyond a non-periodic grid |
| `CGAL_triangulation/ps_interpolation/pass_context.cc` | the cell-loop macros, the pass's contexts (`PSPassCtx`, `PSGridsCtx`, `PSLayoutCtx`), the per-tetrahedron tests, the centroid-fallback cell (`psCentroidCell`) |
| `CGAL_triangulation/ps_interpolation/fx_state.cc` | the pass's fixed-point moment state (`PSFxState`), the walk that bounds its words, in 3D the face walk |
| `CGAL_triangulation/ps_interpolation/pass_setup.cc` | a pass's setup: the sub-grid box, the sample positions, the moment grids, the GPU's eligibility, the fixed-point units |
| `CGAL_triangulation/ps_interpolation/cpu_threads.cc` | the CPU deposit's threads: their count and mode, the slab binning, each thread's start |
| `CGAL_triangulation/ps_interpolation/pass_close.cc` | a pass's close: the words to Real, the moments' normalization, the stream statistics, the phase report |
| `CGAL_triangulation/ps_interpolation/gpu_deposit.cc` | GPU 3D builds: the GPU extraction and deposit (`psGpuExtractAndDeposit`) and the face walk beside it |

The launcher in `python/gui`: `app.py` (the window) with its mixins `app_tabs.py`, `app_specs.py`, `app_run.py`,
`app_menus.py`, `app_check.py`, `app_advanced.py` (the Advanced settings) and `app_widgets.py`; `explore.py` with
`explore_canvas.py`, `explore_grid.py`, `explore_save.py`, `explore_server.py`, `explore_voids.py`, `explore_zoom.py`
and `explore_3d.py` (its geometry in `view3d.py` and its volume in `volume3d.py`, both without Qt; `volume_gl.py`
draws that volume with OpenGL); `contours.py` (the contour levels and lines, no Qt); `runspec.py` (the job
specifications and their commands, no Qt) with `runspec_pipeline.py`, `runspec_plot.py`, `runspec_data.py` and
`runspec_custom.py`; and `grids.py`, `presets.py`, `results.py`, `runs_view.py`, `setup_wizard.py`, `void_cache.py`,
`figsave.py` and `make_icon.py` (the app icon). Import `app`, `explore` or `runspec` before any of their mixin
modules: the mixins read their parent's names through it at call time.

## How a run flows

1. **Options.** `main()` (`src/main.cpp`) answers `--version` before anything else. `User_options::readOptions`
   (`src/user_options.cc`) parses the command line and a `--config` file against the groups of `addOptions`, refuses
   impossible combinations, and resolves what the program decides itself: the `_a` average (exact or sampled), the
   velocity gradient's route, the device and the run mode.
2. **Input.** `readInputData` (`src/input_output.cc`, `src/io/`) reads the particles, for PS-DTFE with their initial
   positions. A PS-DTFE `--grid-window` run whose snapshot globals are recorded beside its tessellation cache starts
   without them and reads them only if a partition it needs is not cached.
3. **Setup and plan.** `DTFE()` (`src/DTFE.cpp`) calls `DTFE_setup`: sub-sampling, `updateEntries` (the options
   that follow from the data), the mean density, standard DTFE's run-wide fixed-point units, then the planner
   `autoTunePartitioning` (`src/auto_tune.cc`). It fills in `--partition` and `--max-concurrent` when the user gave
   none, from the particle count, the grid, the fields and the machine's memory. `--auto-tune-report` stops here.
4. **Split.** PS-DTFE splits in Lagrangian space (`psPartitionedRun`): each partition takes its particles plus a
   padding, is triangulated and deposits into its own Eulerian sub-grid. Several run at once, admitted by a memory
   gate (`PartitionMemoryGate`), with a look-ahead preparing the next one; finished partitions merge into the full
   grids in index order (the default of `--ps-merge`), and the loop checkpoints after merges. Standard DTFE splits
   in Eulerian space (`dtfePartitionedRun`): one partition at a time, split again over the threads
   (`DTFE_parallel`), each sub-domain filling its own part of the grids. One domain goes straight to `DTFE_parallel`.
5. **One tessellation** (`DTFE_interpolation`, `src/CGAL_triangulation/triangulation.cpp`). `buildTessellation`
   loads it from `--tessellation-cache` or inserts the points with CGAL (sequentially or with TBB), puts PS-DTFE's
   cells into a canonical vertex order, computes the vertex densities and saves the cache. Then come `--serve`,
   `--sample-points` and the grid passes: `interpolateGrid_phaseSpace` for PS-DTFE; `interpolateGrid` (point
   values) and `interpolateGrid_averaged_1` (exact or Monte-Carlo average) for standard DTFE. A GPU pass hands the
   tetrahedra to the Metal host; PS-DTFE then deposits on the CPU only those the kernel deferred.
6. **Finish** (`dtfeFinishFields`). The point records are reduced, PS-DTFE's hidden-stream flags settled, and
   `postProcessWebFields` (`src/web_fields.cc`) derives the divergence, shear, vorticity and web classes from the
   full grids.
7. **Outputs.** With `--interlace`, `main()` runs `DTFE()` again on a half-cell-shifted grid and averages the two
   densities in Fourier space. `writeOutputData` writes every grid as `<prefix>.<suffix>`, as `.part.<pid>` files
   renamed into place after the last one. The summary sets the planner's predicted peak memory beside the measured.

## Conventions that must not be broken

**Option names.** An option name is lower-case kebab-case and spelled the same in both programs when both have it.
The prefix `ps-` marks exactly what needs phase-space streams, the Lagrangian tessellation or caustics, so standard
DTFE has no `ps-` option. `tests/docs_drift_check.py` (rule G) checks the built programs' help: no capital or
underscore, no `ps-` name in standard DTFE, every `-h` name also in `--full-help`, and a `-h` of 20 to 35 option
names with no date and no wall time. It exempts one name, `--redshiftSpace`, an option that only refuses the run.

**Stored key text is not an option spelling.** The tessellation cache's identity (`makeDescriptor` in
`tessellation_cache.h`), PS-DTFE's globals record (`ps_globals_record.h`) and the checkpoint's settings key
(`settingsKey` in `partition_checkpoint.h`) are text a later run compares verbatim. Their labels (`MpcUnit=`,
`redshiftSpace=`, `randomSample=`, `average=`, `device=`, ...) belong to those file formats: renaming an option does
not rename them, and changing a label makes every existing cache miss and every checkpoint refuse. A setting that
changes which particles are triangulated, their coordinates or the vertex payload needs a line in the descriptor, or a
cache goes silently stale; write it only when the setting is in force, so the caches of runs without it keep their
identity. The checkpoint key holds the command line (`commandKeyLines`) minus the options that cannot change a grid.

**A default run's outputs stay byte-identical.** A change that is not meant to change results must leave every
output of a default run the same, byte for byte. Keep the binaries from before the change, run the same command with
both, and compare every output with `cmp` (inside a suite, the `same` helper of `tests/lib.sh`): on the CPU and, on a
Metal build, on the GPU; in single and double precision; in one domain and split. Most grids of the single build are
exact fixed-point sums that do not depend on the order of the additions, so a difference is a real change; compare
at the same thread count all the same, since the Monte-Carlo average and the double build's sums depend on it. A
change meant to change results says so and updates the references it moves (*Stored references*).

**The deposit loops and the kernels are timed before and after an edit.** The CPU deposit loops
(`ps_interpolation.cc` and its parts, `averaged_interpolation_1.cc`, `exact_sweep.h`) and the Metal kernels take most
of a run's time, and an edit that looks neutral can slow them. Run the same command with the old and the new binary
on an otherwise idle machine, alternating them, and compare the wall times and the run logs' `phase totals` lines.
On the GPU, `DTFE_GPU_TIMING=1` adds a timing line per batch or kernel run, and the kernel harness takes `BENCH=1`.

**Kernels and hosts share their parameter blocks.** `src/CGAL_triangulation/ps_kernel_params.h` and
`dtfe_deposit_params.h` are the one definition of the structs the kernels read: the Makefile prepends them to the
kernel source and the hosts include them. Keep them plain C++ that is also valid Metal: fixed-width scalars and
arrays, no vector types (a `float3` is padded to 16 bytes), no includes under `__METAL_VERSION__`; their
`static_assert`s pin the layout in both compilers (`PSDepositParams` is 43 four-byte words, `DTFEDepositParams` 36).
A new kernel parameter goes there, never into a second struct in a `.metal` file. The PS-DTFE kernel keeps the
fields and branches of removed switches (`fLinear`, `fVolW`, `fxWords`, `debugNoAtomic`): without them its exact
deposit was slower (metal/README.md, *The branches of removed switches stay*). Two pairs
share no file and change together: the standard kernel's 128-bit fixed-point words (`metal/fixed_words.metal.inc`
and the host's in `src/fixed_units.h`) and the window hull (`window_hull` in the PS-DTFE kernel, `psWindowHull` on
the host).

**Environment variables go through `src/env_knobs.h`.** `Env::flag`, `Env::tristate`, `Env::integer`, `Env::real`
and `Env::text` give every variable one rule: an empty value is unset; `0`, `false`, `no` and `off` (in any case)
turn a switch off and anything else turns it on; `auto` leaves a three-way switch to the program. A name starts with
`DTFE_` and means the same in both programs; a GPU knob has `_GPU_` in it (metal/README.md, *Knobs*, names the one
PS-DTFE knob of its own).

**What a document names exists.** `tests/docs_drift_check.py` reads the documents (its `DOCS` list: the README, the
three guides and the timings page in `docs/`, `demo/COMMANDS.md` and `metal/README.md`; not the release notes, which
name removed options on purpose): every option written with two dashes must be declared (the programs' option table, a script's case
arm, a Python parser), every `DTFE_` variable and every variable a document sets must be read by the code or a test,
the user guide's knob table must give the scripts' defaults, and every driver knob must be documented, in a document
or in the script's `--help` header. Change the documents with the code.

## Tests

While you work, build what you changed and run the suites that touch it, with `--no-build`. Run the full round (the
CI stages, the battery, the Linux container) once a batch of changes is done.

**The quick test.** `make check` tests the built single-precision 3D pair with three suites and stops at the first
that fails: `tests/run_tests.py` (standard DTFE, its last test against the stored reference),
`tests/dtfe_mc_average_check.sh` (standard DTFE's `_a` average on the bundled `demo/z0_64.gadget`) and
`tests/ps_smoke_test.sh` (PS-DTFE end to end on a generated Zel'dovich pancake). It needs no simulation data and
builds nothing, so it never changes a Metal build's mode; without `./DTFE` or `./PS-DTFE` it stops and names the
make command. It runs the suites without the `DTFE_TEST_REFERENCE_DIR` and `DTFE_TEST_PRECISION` of your shell. The
runs use the CPU, but for the demo suite's `--gpu` part on a Metal build.

`tests/ci_suite.sh <stage>...` is the one list of suites that CI and the Docker image run; the first failing suite
stops it:

| stage | runs |
|---|---|
| `build` | the single, double and 2D pairs, CPU-only, with `ARCH_FLAGS=-mtune=generic` |
| `core` | every CPU suite on the single pair, both planners, the dtfelib and void suites, the documents check |
| `double` | `tests/ps_double_check.sh`, then the program suites again on the double pair |
| `2d` | `tests/ps_2d_check.sh`, the 2D pair against analytic 2D flows |
| `gui` | the launcher suites |
| `oldpy` | every `.py` file parses under the oldest supported Python, 3.10 |
| `variants` | a build with `TBB=0 LIBDEFLATE=0`, `make library`, and `demo/examples` linked against it |
| `all` | `build core double 2d` |

```bash
tests/ci_suite.sh build core
```

Do not run the `build` or `all` stage in a checkout with a Metal build: it runs `make` without `METAL=1`, so the
stamps wipe the Metal objects (`o/`, `o_ps/`, and the double pair's) and the programs come back CPU-only. On such a
checkout run the other stages against the programs you built.

CI (`.github/workflows/ci.yml`) runs on a push to any branch, on pull requests and by hand. It has three jobs on
Ubuntu: build-and-test (build, core, gui, double, 2d), old-python (oldpy under Python 3.10, and ruff's pyflakes
rules) and compile-variants (variants, without TBB and libdeflate installed). No CI job builds the GPU backend or
the Docker image.

**The Linux container check.** The Dockerfile runs the same stages (`build`, then `core double 2d`) in an Ubuntu
image, so a successful image build is CI's build-and-test job, less the launcher stage, on your machine, and the only
test of the Dockerfile and `.dockerignore`. It needs a Docker-compatible runtime (on macOS a Linux VM, which never
builds Metal). Run it before you push a change to the C++ code or the build: GCC on Linux rejects things clang on
macOS accepts. The image keeps the programs in
`/opt/dtfe` (the Dockerfile's header shows how to run them on mounted data); `scripts/install.sh --docker` builds
the same image. Its build context leaves out `docs/` and `demo/` (`.dockerignore`) but for the files the test
stages read: a document added to the `DOCS` of `tests/docs_drift_check.py` needs a line there too, and the check's
rule F fails without it.

```bash
docker build -t dtfe .
```

**The local battery.** `tests/dev/battery.sh <scratch dir>` (macOS on Apple Silicon) runs what CI runs and what it
cannot: it builds the three pairs (the 3D ones with `METAL=1`), runs the stages core, double, 2d and gui, the GPU
suites in single and double precision, and the Python suites under CI's package versions with no simulation data.
`--list` names its stages and `--only` picks some; `<scratch dir>/battery/SUMMARY.txt` records the commit and
whether the tree changed during the run (do not edit it while the battery runs). `tests/dev/make_ci_venv.sh` makes
the CI-like Python; `--ci-python` and `--gui-python` name other ones.

**GPU suites.** `tests/dtfe_metal_check.sh` (standard DTFE, CPU against GPU) builds `./DTFE` with `METAL=1` itself
and fails if a run fell back to the CPU; `tests/gpu_scalar_check.sh`, `tests/gpu_mass_check.sh`,
`tests/gpu_retry_check.sh` and `tests/dtfe_gpu_retry_check.sh` print a SKIP line without a Metal build. The PS-DTFE
suites listed in metal/README.md run a GPU part when `o_ps/` carries the Metal stamp.

**Launcher suites.** `tests/py_gui_runspec_test.py` (the commands the launcher builds; no Qt),
`tests/py_gui_app_test.py` (the window end to end, offscreen, through real runs on synthetic data),
`tests/py_gui_explore_test.py` and `tests/py_gui_app2_test.py` (a Qt tier with PySide6, a pure tier everywhere).
`tests/dev/gui_shots.py <out dir>` renders every tab to PNG files, to see layout faults no suite catches.

```bash
DTFE_GUI_PYTHON=/path/to/python-with-pyside6 tests/ci_suite.sh gui
```

**Documents check.** `python3 tests/docs_drift_check.py` applies the rules under *What a document names exists*,
checks the built programs' help (rule G) and feeds itself planted faults; it runs in the core stage.

**Stored references.** `tests/run_tests.py` compares a periodic standard-DTFE run with
`tests/reference/regression_periodic.txt` to 3e-4 relative; after an intended change of the estimator, run
`python3 tests/run_tests.py --update-ref`. `tests/ps_regression_test.py --update-baseline` and
`tests/dtfe_sampling_check.sh --update-reference` rewrite the other two files in `tests/reference/`. CI only
compares.

| variable | effect on the suites |
|---|---|
| `DTFE_TEST_PRECISION` | `single` (default) or `double`: run `./DTFE-double` and `./PS-DTFE-double`, build nothing |
| `DTFE_TEST_REQUIRED` | `1`: a suite that would SKIP for a missing binary or feature fails (the double and 2d stages) |
| `DTFE_TEST_TMP` | where the suites write (default `dtfe-tests` in `$TMPDIR`, else in `/tmp`) |
| `DTFE_TEST_REAL` | set by `tests/precision.sh`: the numpy type of the grids, `float32` or `float64` |
| `DTFE_TEST_REFERENCE_DIR` | another folder for the regression reference; one without a reference makes it a SKIP |
| `PYTHON` | the Python the suites run (default: the first `python3` that imports numpy and h5py) |
| `DTFE_GUI_PYTHON` | a Python with PySide6, for the gui stage |
| `DTFE_GUI_TEST_REQUIRED` | `1`: a missing PySide6 fails the end-to-end launcher test |
| `DTFE_CI_PYTHON`, `OLD_PYTHON` | the battery's CI-like Python; the oldpy stage's Python 3.10 |
| `DTFE_DATA_ROOT` | the simulation data the data-backed Python checks read; unset, they SKIP |

The shell suites pin `DTFE_RUN_MODE=performance` and `DTFE_GPU=0` and unset `DTFE_EXACT_AVERAGE` (`tests/lib.sh`),
so an export in your shell cannot steer them; `tests/ci_suite.sh` hands the same pins to every Python suite it
runs. A Python suite you start by hand (through `tests/ps_test_helpers.py`, or `tests/run_tests.py`) sets the two
pins only where they are unset and leaves `DTFE_EXACT_AVERAGE` alone, so unset all three first. A GPU part passes
`--gpu` itself, and a suite that tests the programs' defaults unsets the pins per run.

## Developer and test variables of the programs

Besides the user's variables (the user guide) and the GPU knobs (metal/README.md, *Knobs*), the programs read these
test hooks and diagnostics:

| variable | what it does |
|---|---|
| `DTFE_TEST_EXIT_AFTER_PARTITIONS=k` | writes a checkpoint after the k-th merged partition, then kills the process |
| `DTFE_TEST_CHECKPOINT_FREE_BYTES` | the free bytes the checkpoint's room rule sees |
| `DTFE_TEST_DELAY_PARTITION=i:s` | PS-DTFE: partition i waits s seconds before its merge, so later ones park |
| `DTFE_TEST_PARK_ROOM_GB` | PS-DTFE: the memory room for partitions parked by the ordered merge, in GB |
| `DTFE_TEST_PLAN_MC=k` | PS-DTFE: stands for a plan of k partitions at once (the dynamic concurrency's floor) |
| `DTFE_TEST_TUNE_UNIFORM` | the standard planner assumes uniformly spread particles, as without the particles |
| `DTFE_TEST_RUN_UNIT_SHIFT=n` | every run-wide fixed-point unit 2^n finer, the vertex star sums' excepted; in the key |
| `DTFE_TEST_NAN_SHARE=k` | every k-th share of the exact sums becomes NaN, to test their guard |
| `DTFE_TEST_GPU_CELL_CAP=n` | a GPU pass's limit of 2^32 cells becomes n, so a small pass falls back to the CPU; the planner keeps its own limit |
| `DTFE_AUTO_TUNE_DEBUG` | prints the planner's terms: busiest partition per thread, PS sub-grids, the start check |
| `DTFE_DEBUG_DEGREES` | PS-DTFE: checks the lazy vertex degrees after a GPU deposit against a full pass |
| `DTFE_DEBUG_FX_BOUNDS` | PS-DTFE: prints each fixed-point bound walk's sums in hex (the same at any thread count) |
| `DTFE_FACE_CHECK` | standard DTFE: keeps the per-tetrahedron gradient beside the face pass, prints `FACE-CHECK` |
| `DTFE_MASS_AUDIT` | PS-DTFE: `1` prints each pass's mass bookkeeping, `2` adds the per-particle closure |
| `DTFE_PTS_FULL_RECORDS` | keeps every member of a point record, unpacked (the packing test) |
| `DTFE_TESSELLATION_CACHE_FINGERPRINT` | prints a loaded tessellation's structure fingerprint |
| `DTFE_TESSELLATION_CACHE_ZLIB` | zlib instead of libdeflate for the cache, for comparisons |
| `DTFE_TESSELLATION_CACHE_CGAL_IO` | CGAL's own tessellation reader and writer, for comparisons |

For an A/B comparison, pin what the program otherwise picks with the user's variables `DTFE_DEPOSIT_THREADS`,
`DTFE_PTS_THREADS`, `DTFE_TRIANGULATION_THREADS`, `DTFE_DEPOSIT_SLABS`, `DTFE_OVERLAP`, `DTFE_RAM_GB` and
`DTFE_FREE_GB` (the user guide), PS-DTFE's partitions at once with `--max-concurrent`, and on the GPU with
`DTFE_GPU_ITEMS=0` (metal/README.md).

## Adding an option, a field or a test

**An option.**

1. Declare it once, in its group of `User_options::addOptions` (`src/user_options.cc`), bound to a member of
   `User_options` (`src/user_options.h`) whose default the constructor sets. Its help text is the reference:
   `--full-help` prints it, and the launcher shows it as a tooltip (`python/gui/presets.py` reads `--full-help`).
2. Check it in the matching `read...` or `check...` function of `src/user_options.cc`; refuse a combination that
   cannot work with `throwError`, naming the option.
3. If a user of the two modes needs it, add a line to `User_options::shortHelp` under the same caption, with the
   short letter and `arg` of the real declaration; `-h` stays between 20 and 35 names.
4. If it changes which particles are triangulated, their coordinates or the vertex payload, give it a line in
   `makeDescriptor`. If it cannot change a grid value, add it to the drop lists of `commandKeyLines` and to the list
   in `--resume`'s help, so a resumed run may change it.
5. If it changes the memory a run needs, model it in the planner (`src/auto_tune.cc` and its parts).
6. If the drivers pass it, add a knob `NAME="${NAME:-default}"` to the driver or `scripts/config.sh` and document it
   in the script's `--help` header. If the launcher sets it, add it to the job specification
   (`python/gui/runspec.py`, `runspec_custom.py`) and a label to `OPTION_TEXT` in `presets.py`.
7. Add a suite that runs it, and describe it in the user guide.

**A field.**

1. Add a flag to `struct Field` (`src/user_options.h`) and to the struct's own lists: the constructor, a parameter
   and a branch of `updateChoices`, `selected()` (a run whose only field is missing there skips the grid pass) and
   the groupings it belongs to (`selectedVelocity`, `deselectVelocity`, ...). Pass its names, the point-value one
   and the `_a` one, in the `updateChoices` calls of `User_options::readFieldOptions`.
2. Add the grid to `Quantities` (`src/quantities.h`) and a row to `Quantities::forEachField`: name, flag, sizing and
   how partitions merge it (`FIELD_SUM`, `FIELD_OR` or `FIELD_DERIVED`). The row drives the reservation, the
   partition merge and the checkpoint.
3. Compute it in the deposit (`ps_interpolation.cc` and its parts; standard DTFE: `averaged_interpolation_1.cc`,
   `unaveraged_interpolation.cc`) or, when it derives from complete grids, in `postProcessWebFields`.
4. Write it in `writeOutputData` (`src/input_output.cc`) as `<prefix>.<suffix>` and `<prefix>.a_<suffix>`.
5. Make sure a GPU run either deposits it (a flag in the parameter block, the kernel and the host) or stays on the
   CPU: `User_options::gpuPathRuns` decides whether a run has a GPU path.
6. Add its bytes per cell to the planner's memory model (in `src/auto_tune/`).
7. Add a row to `FIELDS` in `python/dtfelib/io.py` (name, file suffix, components, PS-DTFE only), and the field to
   the drivers' and the launcher's field lists where users should pick it.
8. Test it against a closed form, a split against one domain (as `tests/ps_split_merge_check.sh` does) and in double
   precision. A PS-DTFE split and its one domain agree to float rounding, not to the byte: each tetrahedron is
   deposited by the periodic image the padding chooses and rounds in that image's coordinates, and in single precision
   the exact deposit on the CPU also places its clip planes in that image's frame, which can move `.tetTouch` by whole
   touches ([user guide](user-guide.md#12-long-runs-and-the-run-log), *Reproducibility*).

**A test.**

1. A shell suite `tests/<name>_check.sh` sources `tests/precision.sh` (which sources `tests/lib.sh`), takes
   `--no-build`, runs `$PS_BIN` or `$DTFE_BIN`, writes into `$DTFE_TEST_TMP`, reads grids as `DTFE_TEST_REAL`, calls
   `skip_or_fail` when a binary or feature is missing, compares outputs with `same`, and exits non-zero on a
   failure. `tests/lib.sh` defines the helpers that several suites share, once, one name per contract (its comments
   give each one's arguments and output): the counted `check` with `check_result`, which prints the closing RESULT
   line and sets the exit status; `check_ok` (a 1 passes; a failure counts in `FAILS`) and `agrep` (`check_ok` of
   whether a file matches a pattern); `check_yes` (a 1 passes) and `check_rc` (a 0 passes), which count a failure in
   the suite's `fails`; `has` (whether a file holds a text) and `has01` (whether it matches a pattern, as 0 or 1 for
   `check_rc`); `pass` and `fail`; `same`; `skip_or_fail`; and the readers of a checkpoint's lists and key and of a
   run log's header lines (`key_of`, `mode_of`, `share_of`). Use them, and give a helper of a new contract a name of
   its own. `tests/generate_ps_test_data.py` writes synthetic snapshots (`--dim 2` for the 2D pair).
2. A Python suite `tests/py_<name>_test.py` uses `tests/_harness.py` (`setup`, `check` of a function, `check_ok` of a
   value, the closing RESULT line); one that runs the programs takes `binary` and `skip_or_fail` from
   `tests/ps_test_helpers.py`.
3. Register it: a `step` line in `run_program_suites` of `tests/ci_suite.sh` (it then runs again in the double stage)
   or in `stage_core` to run once. A GPU suite skips on a CPU build (`require_gpu_backend` in `tests/lib.sh`) and
   goes into `tests/dev/battery.sh`, as does a Python suite (its `PY_SUITES` list).
4. A reference the suite compares with lives in `tests/reference/` and is written by an update flag of the suite.
