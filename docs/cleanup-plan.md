# Lamina: cleanup and polish plan

Status: **plan only**. Nothing in it has been changed yet. It covers the C++ binaries (DTFE, PS-DTFE and their
double and 2D builds), the Metal and CUDA/HIP GPU code, the driver scripts, `python/dtfelib`, the plot scripts and
tools, and the PySide6 launcher in `python/gui`.

**Base.** The survey read commit `5ac1ce1` with three pending pieces of work applied on top (Phase 0 lands them):

- the faster exact deposits: the CPU r3d sweep `exact_sweep.h`, the GPU register z-sweep, the GPU exact-dispersion
  fix, and PS exact on the CPU by default;
- the `--ps-window` auto-tune fix (the read residue and window sub-grids);
- the `download_snapshots.sh` https fix.

Line numbers are from that patched state and will drift. Items therefore name `file:function` first and give line
numbers in brackets.

**Scope rules carried over from the earlier cleanup rounds:**

- The binaries keep the names DTFE / PS-DTFE and the Python package stays `dtfelib`. The project is called Lamina.
- No default simulation and no default data root anywhere.
- The launcher's QSettings key stays `("DTFE", "launcher")`.
- The launcher's 3D view (`python/gui/explore_3d.py`, `view3d.py`) is being redesigned separately and is out of scope.
- **Hot-loop rule:** the PS deposit-loop lambda and the GPU extraction walk stay inline unless the split is proven
  output-identical and timing-neutral.
- Upstream code from Cautun's DTFE (2011 headers) is touched only where this plan says so: `averaged_interpolation_2.cc`,
  `unaveraged_interpolation.cc`, the SPH/NGP interpolators, `src/kdtree`, and the `Particle_data`, `Box` and
  `miscellaneous.h` helpers. Changes there are marked **upstream**: they are the owner's call and low priority.

---

## How to read an item

Each item has an ID (e.g. `P3.2`) so that dependencies can refer to it. Every item gives:

- **What and where**: the change and the files or functions it touches.
- **Why**: the reason for the change.
- **Risk**: what could break.
- **Effort**:
  - S = up to half a day;
  - M = 1–2 days;
  - L = 3 days or more, including the gate runs.
- **Verify**: the gates below, plus any item-specific check.
- **Depends**: the items that must land first.

Estimated gains are marked *(est.)*. Only numbers labelled *measured* come from real runs.

### Gates

| Gate | What it runs | Pass |
|---|---|---|
| **G-build** | all six build modes: plain, `METAL=1`, `METAL=1 DOUBLE=1`, `DIM=2`, `make library`, `CUDAEMU=1` | 0 warnings, 0 errors |
| **G-byte** | `tests/dev/bytecmp.sh <ref0> <out>` against the reference binaries taken in P0.3 | every file identical (184 output files after P0) |
| **G-gpu** | the GPU-exact same-binary comparison added in P1.12, plus `metal/validate_deposit` | identical bytes before and after a host-only change; 31/31 |
| **G-emu** | `CUDAEMU=1` build + `tests/cudaemu_check.sh` at warps 32 and 64 | PASS |
| **G-battery** | `tests/dev/battery.sh` | 31/31, tree unchanged during the run |
| **G-linux** | CI's Linux stages in a local Ubuntu 24.04 container: `tests/ci_suite.sh build`, `core`, `double`, `2d`, `variants`, and `CUDAEMU` + cudaemu_check | PASS; required before every push |
| **G-gui** | the four GUI suites (`tests/py_gui_app_test.py`, `py_gui_app2_test.py`, `py_gui_runspec_test.py`, `py_gui_explore_test.py`), an offscreen render of every tab at two window sizes, and one look in a real macOS window | suites pass; no clipping or overlap in the renders |
| **G-docs** | `tests/docs_drift_check.py` | PASS |
| **G-tuner** | `tests/ps_auto_tune_check.sh` and `tests/dtfe_auto_tune_check.sh`. A fixed list of `--auto-tune-report` configurations must give the same lines, except the configurations the item targets | PASS; only the targeted lines change |
| **G-real** | reruns of the real simulations from P2.3 | predicted vs measured peak footprint within the item's band; wall time recorded |
| **G-move** | proofs for moved code: Python AST equality per moved function; for hot C++ functions, `-O3` disassembly of the symbol with addresses stripped | identical |

Reviews stay light. Each phase gets at most a couple of reader agents and one judge, and only HIGH or MEDIUM findings
are acted on. Each phase updates `change_log.txt` and the README section it touches.

---

## Phase overview

| Phase | Theme | Changes output bytes? | Main gates | Depends on | Effort |
|---|---|---|---|---|---|
| **P0** | Land the pending work, freeze reference binaries | as measured in its own sessions | G-battery, G-byte vs the pre-patch build, G-linux, CI | — | M (mostly machine time) |
| **P1** | Quick fixes and dead code (incl. two GUI layout bugs) | no | G-build, G-byte, G-gpu, G-gui | P0 | M |
| **P2** | Measure first: timers, a debug knob, real reruns | no (instrumentation only) | G-byte; the reruns are the product | P0 | M + machine hours |
| **P3** | C++ structure: shared helpers, splits, tuner to `.cc` | no | G-byte, G-move, G-tuner, G-battery | P1 | L |
| **P4** | GPU structure: shared host code, one params header, twin check | no | G-byte, G-gpu, G-emu, retry suites | P1 | L |
| **P5** | Python, plots, tools, bash drivers | no (same figures and data) | py suites, `run_scripts_check.sh`, G-docs, G-move | P0 | L |
| **P6** | One flag scheme with a deprecation path | no (names only) | resume suites, `run_scripts_check.sh`, runspec suite, G-docs, G-byte | P3 (P3.3) | M–L |
| **P7** | Launcher polish | no | G-gui | P1, P6 | M–L |
| **P8** | Auto-tuner calibration and a time objective | plans change; outputs do not (checked with the plan pinned) | G-tuner, G-real | P2, P3.8 | L |
| **P9** | CPU and GPU optimisations | no (each item is byte-identical by design) | G-byte, G-gpu, G-emu, G-real timing | P2, P4.1 | L |

**Parallel work.** P3, P4 and P5 touch disjoint files, so they can run as separate sessions in separate worktrees.
Some files are shared between phases and must land in one order:

| Shared file or group | Order |
|---|---|
| `ps_interpolation.cc` | P1.3 / P1.11 → P3.5 → P9.2 |
| `ps_metal_host.cc` / `ps_gpu_cuda.cu` | P4.1 → P9.5–P9.7 |
| `auto_tune.h` | P3.8 → P8 |
| `partition_checkpoint.h` | P3.3 → P6.2 |
| `python/gui/explore*.py` | P7 coordinates with the 3D-view branch |

---

## P0. Land the pending work and freeze a baseline

**P0.1 Regenerate the exact-deposit patch before landing it.**
- What and where: the saved exact-deposit patch is **incomplete**. It does not contain the `--scratch-dir` zero-fill
  fix (`ScratchAlloc::ZeroFillScope`), which exists only in the local worktree where the work was done:
  - `src/scratch_alloc.h`
  - `src/scratch_alloc.cc`
  - `src/scratch_alloc_stub.cc`
  - `src/quantities.cc`
  - possibly more of its 22 call sites.

  That worktree is based on `066c7ac`, not on `5ac1ce1`. Regenerate the patch from it against `5ac1ce1`, then check
  two things: that nothing `5ac1ce1` added is reverted (the Explore and 3D-view files, `auto_tune.h`,
  `ps_runtime_check.sh`), and that `grep ZeroFillScope` finds every call site.
- Why: applying the old patch would silently drop the fix that let 1536³ runs on an external scratch disk proceed.
- Risk: medium (a three-way merge). Effort: S.
- Verify:
  - `ps_scratch_check`;
  - G-byte against a `5ac1ce1` build (only GPU-exact files may differ; the CPU exact files stay identical);
  - G-battery;
  - G-gpu: `validate_deposit` 31/31;
  - G-emu.

**P0.2 Land the window-autotune fix and the download fix.**
- What and where: `src/auto_tune.h`, `tests/ps_auto_tune_check.sh` (window section W),
  `scripts/download_snapshots.sh` and `tests/run_scripts_check.sh`.
- Risk: low. Effort: S.
- Verify: `ps_auto_tune_check`, `ps_window_check`, `run_scripts_check`.
- Depends: may land together with P0.1, since the two patches touch no common file.

**P0.3 Freeze the baseline.**
- What:
  - Copy the six binaries of the landed tree into a reference folder (`ref0`). Every byte-identical phase compares
    against it.
  - Save one `--auto-tune-report` line set (P3.8 uses it).
  - Save the GUI renders (P1.12 adds the script that makes them).
- Gate for the whole phase: G-linux. Nobody has compiled the GCC branch of the unrolled r3d reduce yet, because that
  branch exists for GCC only. Then push and run CI.
- Decision before landing: **D1** (keep or delete `PS_GPU_CPU_SHARE`).

---

## P1. Quick fixes and dead code (byte-identical)

Every item here is small and mechanical. Land them as one or two commits.
Gate: G-build, G-byte, G-gpu for the GPU items, and G-gui for P1.1 and P1.2.

**P1.1 Pipeline tab: the grid row is drawn on top of the λ_th row** *(bug, seen in the offscreen render)*.
- What and where: `python/gui/app_tabs.py:_build_pipeline_tab` (about lines 898–923). It adds the Grid, Deposit and
  GPU rows to the `_fields_group()` grid at fixed rows 3–5. `_fields_group` already puts the T-web threshold row at
  `(len(PS_FIELDS)+2)//3` = 4, so two rows share row 4.
- Fix: append at `g.layout().rowCount()`. Better, move Grid, Deposit and GPU into their own "Estimator and grid" group,
  as the Grids tab already does.
- Risk: low; the tests find the widgets by attribute (`p_grid`, `p_deposit`, `p_gpu`). Effort: S.

**P1.2 The right edge of the forms is clipped on the Grids, Custom, Pipeline and Plots tabs** *(bug)*.
- What you see: the preset row loses its third button ("Delete"), and the Deposit, File, Format, Box and Pixels fields
  run off the edge.
- Cause, measured in the render: the scroll viewport is 666 px wide but the form needs 741–765 px, and
  `app_widgets._scroll` turns the horizontal scroll bar off. A combo box sizes to its longest item; the Deposit combo
  is the likely culprit.
- Fix:
  - In `app_widgets._grow_fields`, set every `QComboBox` to `AdjustToMinimumContentsLengthWithIcon` with about 18
    characters.
  - Let the preset combo stretch.
  - Shorten the longest checkbox texts.
  - Add a check in the GUI test that `inner.minimumSizeHint().width() <= viewport` for every tab.
- The offscreen render used Qt's generic style, not the macOS one, so confirm the widths in a real window.
- Risk: low. Effort: S. Keeps the 690 px minimum left pane width.

**P1.3 Tautological dimension guards.** `NO_DIM` can only be 2 or 3 (the Makefile and `define.h` reject anything
else), so `#if NO_DIM==3 || NO_DIM==2` is always true.
- Where:
  - `ps_point_eval.cc` (line 76), together with its never-compiled `#else` stubs (about lines 3207–3256) and an empty
    `#ifdef PHASE_SPACE` block;
  - `ps_exact_inside.h` (line 31);
  - `ps_interpolation.cc` (a 294-line block, about lines 3608–3902; only the guard lines go).
- Risk: none. Effort: S.

**P1.4 Functions with no caller** (checked by `grep -w` over src, metal, tests, python and scripts):
- `particle_data_traits.h`: `compareParticles`, `sameParticle`.
- `triangulation_miscellaneous.h`: `vertexPositionMatrix`.
- **upstream**:
  - `Box::isParticleLagrangianInBox`;
  - the `Particle_data` setters `setDelta`, `setLagrangianPosition`, `setScalar`, `setVelocity`, `setWeight`,
    `setPosition`;
  - in `miscellaneous.h`: `upperBoundCheck` and both `superfluous_options`;
  - in `src/kdtree`: `n_nearest_brute_force`, `r_count`, `r_count_around_point`, `r_nearest_around_point`.
- The upstream ones are optional, see D10.
- Risk: none. Effort: S. Verify: G-build, which must include the CUDA/HIP compile jobs on CI.

**P1.5 Declare `printElapsedTime` once.** `subpartition.h` defines it, and `SPH_interpolation.cc`,
`NGP_interpolation.cc` and `grid_deposit_engine.h` each declare it again. Effort: S.

**P1.6 Delete the `PS_COLUMNS` prototype** in `metal/ps_deposit.metal`: the doc comment and define (lines 20–29) and the
`#if PS_COLUMNS … #endif` block (about lines 1080–1184).
- Nothing defines `PS_COLUMNS` (no Makefile rule, no test), and the CUDA twin has no counterpart.
- Keep the `BeyondColumn` / `column_*` helpers: `count_beyond_grid` uses them.
- Risk: low. Kernel text that is never compiled changes nothing. Verify: G-gpu. Effort: S.

**P1.7 Stale comments after the GPU z-sweep replaced the GPU r3d port.**
- `gpu_host.h` (about lines 97, 160, 163) and `ps_deposit_params.h` (line 241) still say "float32 r3d port".
- `tests/ps_linear_deposit_check.sh` (lines 261–262) names kernel functions that no longer exist (`exact_init_tet`,
  `exact_clip_plane`, `exact_reduce2`).
- Check that the Makefile's GPU comment ("not yet validated on hardware") is still accurate.
- Effort: S.

**P1.8 Unused include:** `#include <deque>` in `ps_metal_host.cc`. Effort: S.

**P1.9 Dead Python names** (each appears only at its definition):
- `config.py`: `cosmic_time_gyr`, `CORRELATION_N_BINS`, `VIS_SMOOTHING_SIGMA`;
- `fields.py`: `ensure_output_dir`;
- `plot/plot_PDF_CDF.py`: `get_critical_point_densities`.

Some helpers are used only by tests: `catalog.read_hdf5`, `io.velocity_scale`, `theory.bbks_ep_pdf`,
`trees.descendant_branch`. Keep them as public API.
- Verify: ruff F, the py suites. Effort: S.

**P1.10 Unused environment knobs in the tests.** Seven test suites (`gpu_retry_check`, `ps_fixed_moments_check`,
`ps_mass_exact_check`, `gpu_mass_check`, `dtfe_gpu_retry_check`, `dtfe_metal_check`, `dtfe_auto_tune_check`) still
set the `*_METAL_*` spellings of GPU knobs.
- Move them to the canonical `*_GPU_*` names.
- Dropping the aliases themselves is D3.
- Effort: S.

**P1.11 `PS_GPU_CPU_SHARE`, if D1 says delete.**
- Where: `ps_interpolation.cc`, about 120 lines inside the GPU walk (about lines 2770–2890).
- Context: opt-in, the best measured gain was 4.5% on the 94M-particle run, and no test or doc covers it.
- Verify: G-byte, G-gpu (the default path, share 0, is byte-identical by construction). Effort: S.

**P1.12 Two verification tools the later phases need.**
- (a) Extend `tests/dev/bytecmp.sh` with an opt-in GPU-exact section: `--ps-exact-deposit --ps-gpu` and
  `--exact-average --gpu`. It compares two binaries built from the same kernel source. Today `bytecmp` leaves GPU
  exact out, because fast math makes it differ from the CPU. A host-only refactor must still leave it unchanged.
- (b) Add a `tests/dev/gui_shots.py` that renders every tab offscreen at 1500 and 1100 px, using `remember=False` so
  that saved settings are never written.
- Effort: S each.

---

## P2. Measure first (no output change)

The tuner and the optimisation phases need numbers that the existing run logs do not have. Everything here is
instrumentation or reruns.

**P2.1 GPU per-phase timers under `PS_GPU_TIMING=1`.**
- Where: `psGpuDepositFields` in `ps_metal_host.cc` and `ps_gpu_cuda.cu`. Today both take `ctxMutex` as their first
  statement and hold it to the end.
- What to time separately, inside the held window:
  - plan and grid init;
  - uploads;
  - the exact item bounds;
  - the item fill;
  - snapshots;
  - the leftovers scan;
  - the copy-back;
  - the releases;
  - the pacing gaps.
- Why: on the 94M-particle exact run the lock was held 1143 s, of which the kernels were busy 877 s (*measured*). The
  other 266 s (23%) is host work, and P9.5–P9.7 target it. A rough split from reading the code: bounds and item fill
  0.5–0.7 s per partition, uploads and first touch 0.4–0.6 s, copy-back and releases 0.2 s, pacing gaps about 0.2 s
  *(est.)*.
- Risk: none. Effort: S. Verify: one-partition run (multi-partition kernel sums swing up to 3× run to run).

**P2.2 A PS counterpart of `DTFE_GPU_DEBUG_NOATOMIC`.** It measures the atomics' share of the sampled kernel's
time before P9.8 is built. Effort: S.

**P2.3 Rerun the runs the tuner table flags (see P8), with the current binary.**
- GPU sampled with sample points, TNG100-3-Dark z = 1 and z = 0, and TNG50-3-Dark z = 0. The logs on disk come from a
  build that predates the fix to the partition hand-out order (`schedule(monotonic: dynamic, 1)`), so their
  −45…−69% errors are mostly parking that the fix removed. The TNG50 z = 0 run took 2.4 h at 13% CPU.
- PS exact at 1024³, with `ZeroFillScope` (the logged run predates it).
- **One auto-planned standard-DTFE run.** All 27 standard logs on disk used hand-set plans, so the standard planner's
  predictions have never been checked against a real run.
- A CPU `--ps-window` run at 2048² and at 4096². These are under-predicted by about 25–30% even after P0.2.
- Also record per-phase times: triangulation, vertex densities, deposit, merge, export.
- Effort: M (several hours of machine time; needs the simulation data).

**P2.4 Time the standard binary's dummy-point insertion** (the padding test).
- Where: `triangulation.cpp` (lines 133–139), on a lattice-like snapshot. It inserts about 6n² points that lie on
  lattice planes, which is likely to force CGAL's exact-predicate fallback.
- Feeds D5. Effort: S.

**P2.5 Merge share.** From the P2.3 logs, take the time spent in the serial ordered merge (`DTFE.cpp:mergeOne` →
`quantities.cc:addFieldSubgrid` / `orFieldSubgrid`). This decides P9.1. Effort: S.

---

## P3. C++ structure (byte-identical)

Gate for each item: G-byte, G-build, G-battery for the phase, and G-move wherever a hot function is touched.

**P3.1 One thread-count helper.**
- Four copies of the same function:
  - `ps_interpolation.cc:psDepositThreads` (line 282) and `averaged_interpolation_1.cc:exactAverageThreads`
    (line 332), which are line-identical;
  - probably also `ps_point_eval.cc:pointEvalThreads` (line 461) and `DTFE.cpp:dtfeParallelThreads` (line 751).
- Change: one `coresPerTeam()` in a small header next to `env_knobs.h`.
- Why: P9.2 changes this rule, and it should change in one place.
- Risk: low. Effort: S.

**P3.2 One 3×3 adjugate.**
- Three copies: `math_functions.h:matrixInverse`, `ps_exact_inside.h:inverse3x3d`, `ps_point_eval.cc:inverse3x3Unguarded`.
  They differ only in the singularity guard.
- Change: one inline `adjugate3x3(m, inv) -> det` core with the same operation order, and thin guards around it.
- Risk: medium. The operation order must stay exactly the same, because `dtfe_gpu_fixed.h` mirrors this arithmetic.
- Verify: G-byte, plus `-O0` IR floating-point operation counts. Effort: S.

**P3.3 One field list.**
- The list of `Quantities` grids is written out six times in `quantities.cc`, and again in:
  - `partition_checkpoint.h:forEachAccumulator`;
  - `ps_fixed_merge.h:forEachSummed`, whose comment says "if you add one there, add it here";
  - `input_output.cc`, twice;
  - `web_fields.cc`, twice.
- Change: one `Quantities::forEachField(f)` with a tag for summed, OR-merged and derived fields.
- Why: today a new field must be added in about 12 places, and forgetting one is silent.
- Risk: low to medium. Effort: M.
- Verify: G-byte, the resume and fixed-merge suites (`ps_resume_check`, `dtfe_resume_check`, `ps_fixed_merge_check`).

**P3.4 One look-ahead pipeline, one progress line.**
- Two copies of the look-ahead pipeline: `DTFE.cpp:psRunPartitionLoop` (lines 1685–1744) and
  `dtfePartitionedRun` (lines 2488–2560). Both have:
  - the same `std::thread` with `exception_ptr`;
  - the same room test, `autoTuneSystemFree() - 4e9 >= need`;
  - the same "look-ahead paused" and "prepared in" messages;
  - the same join and rethrow.
- Change: a `runLookAhead(order, needBytes, prepare, deposit, canPrefetch, label)` template. Also one
  `partitionProgress()` for the `[partitions k/N] elapsed ETA` line (lines 1617 and 2412).
- Then: make the merge-staging state in `psPartitionedRun` (about lines 2050–2160) a `struct PSMergeStaging`, with
  `parkWhere` and `subgridBytes` as methods. `PSLoopCtx` (40 reference members) then shrinks to a handful.
- Risk: low; ordering is unchanged. Effort: M.
- Verify: `ps_runtime_check` (its parking and look-ahead checks), `dtfe_resume_check`, G-byte.

**P3.5 The `interpolateGrid_phaseSpace` remainder (2574 lines) without touching the deposit loop.**
- Lift the `fx*` lambdas (about lines 2297–2460) into a `PSFxState` struct. They run per pass, not per cell; only
  `fxAddL` is used inside the loop.
- Lift `markFlipped` and `fxBoundWalk`.
- Lift the GPU block as `psGpuExtractAndDeposit(ctx)`, keeping the walk lambda inside it (hot-loop rule).
- `depositLoop` itself (about 1010 lines) stays; see D7.
- Risk: medium. Effort: M.
- Verify: G-move on `depositLoop` and the slab lambdas, G-byte, a timing check within ±1%.

**P3.6 Split `ps_point_eval.cc` (3535 lines) into its three modules** by pure moves:
- `ps_point_eval.cc`: point evaluation;
- `ps_serve.cc`: `--serve` and the composite server (about lines 2020–3200);
- `ps_mean_streams.cc`: `psEstimateMeanStreams`, 244 lines, which only `auto_tune.h` uses.

Also split by phase, with no change of logic:
- `psPointEvalFinalize` (299 lines): reduce, masks, write;
- `psServeCompositeRun` (243 lines).

Group the file's `g_`-prefixed globals into one context struct. Update the Makefile source lists and the `.cu`
dependency lists.
- Risk: low. Effort: M.
- Verify: G-byte (the point-evaluation and serve runs), `serve_split_check`, G-emu.

**P3.7 `interpolateGrid_averaged_1` (643 lines).**
- Split its `cpuLoop` lambda (about lines 958–1274) into `avg1SingleCell`, `avg1ExactTet` and `avg1McTet`.
- Remove the per-tetrahedron heap allocation `std::vector<Point> randomPoints(n)` (line 1210): reuse a thread-local
  buffer. That part is a small speed-up as well.
- Risk: medium. It is the CPU hot loop of standard DTFE.
- Verify: G-move on the lambda, G-byte, `dtfe_mc_average_check`. Effort: M.

**P3.8 `auto_tune.h` (2446 lines, a header) → `auto_tune.h` + `auto_tune.cc`.**
- Move the non-template, non-hot bodies into the `.cc`:
  - `autoTunePartitioning` (about 330 lines);
  - the `AutoTuneStdPlanner` bodies;
  - `stdTuneSplit`;
  - `StdTuneDensity::build`;
  - `autoTuneServe`;
  - `cachedServeSplits`.

  Today every edit rebuilds all of `DTFE.cpp`.
- Name the magic numbers, each with a one-line source comment:
  - `kDT = 666` and `kParInsert = 210`, with the standard side's duplicate `648`;
  - 6.77 tetrahedra per vertex (about 10 uses);
  - the 72/48 B per-tet sizes and their 1.3 slack;
  - the 4e9 headroom;
  - the 0.85 split gain;
  - `16 << 20` (the HitSink block);
  - `2e6` (`minN`);
  - the 4 GiB allocator rule;
  - `UINT32_MAX`.
- Share the padding-factor loop (`fPad`, repeated in `partitionBytes`, `lookAheadBytes` and `predictedWall`).
- Replace `AutoTunePSPlanner`'s 40 reference members with one struct of resolved inputs.
- Risk: low. Effort: M.
- Verify: G-tuner with **every** report line identical (the P0.3 line set), G-byte.

**P3.9 (optional, upstream) The sampling variants.**
- Seven functions repeat the per-point "locate, cached inverse, accumulate the fields" loop, about 40–55% identical
  lines:
  - `interpolateGrid_averaged_2`, `interpolateRedshiftCone_averaged_2`, `interpolateUserSampling_averaged_2`,
    `interpolateGrid_averaged_3`;
  - `interpolateGrid`, `interpolateRedshiftCone`, `interpolateUserSampling`.
- Methods 2 and 3, the redshift cone and user sampling have **no tests**. The first step is therefore golden outputs
  with a fixed `--seed`, which is worth doing on its own. The dedupe (an `evalAtPoint` helper plus a sample-source
  functor) is D10.
- Risk: medium. Effort: L.

---

## P4. GPU structure (byte-identical)

Gate for each item: G-byte, G-gpu, G-emu. The retry items add `tests/gpu_retry_check.sh`,
`tests/dtfe_gpu_retry_check.sh` and `DTFE_GPU_FAULT_AT` runs.

**P4.1 Move the remaining identical host helpers into `gpu_host.h`.**
- `psHostFirstChunk` and `psHostCheckpointEnv` are textually identical in `ps_metal_host.cc` and `ps_gpu_cuda.cu`.
- So is the exact item-list fill inside `psHostExactItems`. Make it one shared `psBuildExactItems(const float*, …)`.
  P9.6 needs that.
- `psHostReportPasses` differs only by the backend's name.
- Risk: low. Effort: S.

**P4.2 One parameter-struct header per estimator.**
- `DepositParams` exists in four places:
  - PS: `ps_deposit_params.h` plus its MSL copy in `metal/ps_deposit.metal`;
  - DTFE: `dtfe_metal_host.cc`, `dtfe_gpu_cuda.cu` (with an extra `bad` pointer) and `metal/dtfe_deposit.metal`.

  The only guard today is a comment, "must match byte-for-byte".
- Change: a `dtfe_deposit_params.h` with `static_assert(sizeof …)`, and `static_assert`s on `ExactItemParams`. The
  Makefile already prepends text to the MSL source, so the host header can be prepended the same way, with `int32_t`
  mapped to `int`.
- Risk: low. Effort: M.

**P4.3 One chunk-and-retry loop.**
- The loop (fault cause, pace escalate/ease, the 3-fail and 16-fail limits, checkpoint cadence, chunk rescale) exists
  four times:
  - `ps_metal_host.cc:runChunked`;
  - `dtfe_metal_host.cc:runBatch`;
  - `ps_gpu_cuda.cu:runChunked`;
  - `dtfe_gpu_cuda.cu:runBatch`.

  That is about 350 lines of duplication.
- Steps:
  1. `GpuPace::onFault` / `onClean`.
  2. A header-only `GpuChunkLoop<Backend>`, where the backend supplies `submit(start, n)`, the snapshot take/restore
     and its labels.
  3. Unify the `GpuFault` / `GpuSnapshot` twins in `gpu_metal_retry.h` and `gpu_cuda_retry.cuh`.
- This shortens `psGpuDepositFields` (426 / 399 lines) and `runBatch` (262 / 235) by about 110 lines each.
- Risk: medium. The retry paths are exercised only by the fault-injection suites.
- Effort: L.

**P4.4 A Metal↔CUDA twin check, before any kernel text changes.**
- Nothing compares the twin kernels function by function today: `validate_deposit` loads only the Metal source, and
  `cudaemu_check` compares whole binaries.
- Change: compile `gpu_exact_clip.cuh` (through `gpu_cuda_compat.h` in emulation) as plain C++, feed it the same
  random tetrahedra and cells as `validate_deposit`'s tests 7 and 14, and compare with the Metal output within float tolerance.
  Run it in CI's cudaemu job.
- Also add a "lockstep pairs" table (function, Metal location, CUDA location) to `metal/README.md`.
- Risk: none. Effort: M.

**P4.5 (optional, after P4.4) Shared scalar GPU math.**
- About 350–400 lines are near-identical between `metal/ps_deposit.metal` and `ps_gpu_cuda.cu`:
  - `classifySample`;
  - `window_hull` and `window_flat`;
  - `exact_vel_range` and `exact_disp_illcond`;
  - `count_beyond_grid`;
  - `det3` and `inverse3`;
  - `linearWeight`;
  - `fixedShareOk`;
  - `tet_planes` and `box_outside_tet`.
- Change: a `gpu_math_common.h` behind a small macro layer (`GP_FN`, `GP_THREAD`, `gp_min`). Leave
  `sweep_moments` out: the Metal version is vectorised and the CUDA one hand-scalarised.
- Risk: **high** for byte identity. Metal compiles with fast math, so a textual move can change how FMAs are
  contracted.
- Verify: G-gpu same-machine A/B, then G-emu, P4.4 and kernel timing. Effort: L.
- Also merge `ps_deposit.metal:exact_tet_setup` with the preamble of `depositFields`, which carries a "keep in sync"
  comment. A/B the sampled kernel's register use.

---

## P5. Python, plot scripts, tools and bash drivers (same figures, same data)

Gate for each item: the py suites, `tests/run_scripts_check.sh`, G-docs, G-move for moves, and a byte comparison of
the `.npz` / `.csv` products and the `--help` text of every script touched. P5 is independent of the C++ phases.

**P5.1 Snapshot helpers in `dtfelib/cli.py`.**
- About 11 places paste the same snapshot-ladder expression, `sorted({int(d.name.split("_")[1]) for d in
  sim_dir(...).glob("snapdir_*")})`:
  - `plot_void_convergence.py`, `plot_void_crossbox.py`, `plot_void_fate.py`, `plot_void_environment.py`,
    `plot_void_tracking.py`, `plot_void_population.py`, `plot_void_profile_evolution.py`, `plot_web_streams.py`,
    `plot_caustic_skeleton.py`;
  - `analyze.py:snapshots_on_disk`;
  - a variant in `plot_pointeval.py:discover_snaps`.
- About 15 more build `sim_dir(...) / f"snapdir_{n:03d}"` by hand.
- Change: add `snapdir_path(sim, snap, root=None)` and `snapshot_ladder(sim, root=None, require=None)`. Keep each
  site's choice of passing or not passing the data root.
- Risk: low. Effort: M.

**P5.2 One output-tag rule.**
- The rule `("" if grids == "ps" else f"_{grids}") + ("_overlaps" if keep_overlaps else "")` is copied in five
  places:
  - `plot_void_population.py`;
  - `plot_void_environment.py:name_tag`;
  - `plot_void_profile_evolution.py`;
  - `plot_void_profiles.py`;
  - `plot_caustic_skeleton.py`.
- Change: one `catalog.output_tag(...)`.
- Risk: **medium**. The rule decides figure and cache names, so a wrong tag orphans caches.
- Verify: a unit test of the helper against each old rule over (ps, dtfe, prefix, overlaps), and identical file names.
- Effort: S–M.

**P5.3 Figure saving through `figures.save_figure` everywhere.**
- Seven sites call `savefig` directly: `plot_DTFE.py`, `plot_cosmic_web.py`, `plot_halo_tracking.py`,
  `plot_void_tracking.py` (×2), `plot_pointeval_panels.py`, `pointeval.py`.
- Make the `dpi` default `None`, meaning `config.DPI`. That drops about 20 redundant `dpi=DPI` arguments.
- The two tracking plots use 200 dpi on purpose; name it `config.TRACK_DPI`.
- Build `OUTPUT_DIR` one way: `config.figures_path`, not `LOCAL_FIGURES_ROOT` or `FIGURE_ROOT`.
- Risk: low. Effort: S.

**P5.4 Small helpers copied between scripts** move into `figures.py` or `dtfelib`:
- `_title`, three copies;
- `plot_finite` and `unit_cos`, two copies each;
- `smooth_components`, two versions;
- the `scipy.fft` shim in `pipeline.py` and `spectra.py`, which becomes `dtfelib/_fft.py`.
- Risk: none. Effort: S.

**P5.5 Argument helpers.**
- `plot_eigenvalues.py`, `plot_shear_triaxial.py` and `plot_smoothing_comparison.py` reach into argparse's private
  `_option_string_actions`. Replace that with `make_parser(..., snap_default=...)`.
- Add `add_void_args`, `add_snaps_arg`, `add_out_arg` (a `Path`) and `add_rv_definition_arg`. Accept both spellings,
  `--r-v-definition` and `--rv-definition`.
- Tools: `compare_deposits.py`, `void_outlines.py` and `make_image_plane.py` should go through `make_parser` /
  `require_sim` (today some ignore `DTFE_SIM`), and share one `_bootstrap`.
- Make `--out` the name everywhere, with `--output` / `-o` as an alias.
- Never rename a flag the launcher emits; only add aliases.
- Risk: low. Effort: M.

**P5.6 A series driver.**
- Eight scripts repeat the same loop: snapshot list → mkdir → `process_snapshot` per snapshot → "Completed: n/m" /
  "Failed snapshots".
  - `plot_PDF_CDF.py`, `plot_eigenvalues.py`, `plot_contour_filter.py`, `plot_ellipses_contour_3D.py`,
    `plot_shape_filter.py`, `plot_velDiv_den.py`, `plot_shear_triaxial.py`, `plot_marked_correlation_BBKS.py`.
- Change: `cli.snapshot_series(args)` + `cli.run_series(items, fn, label)`.
- First grep `python/gui` for parsers of these lines. The launcher reads script stdout.
- Risk: low. Effort: M.

**P5.7 Long functions and mixed modules.**
- `plot_void_population.py:main`, 272 lines: split into setup, per-snapshot statistics, figures, tables.
- The cache code in `analyze.py` (about 420 lines) moves to `dtfelib/cache.py`. The `analyze.py cache` subcommand
  stays.
- `pipeline.py` (1051 lines) holds three subjects:
  - critical points → `critical.py`;
  - void-radius measurement → `void_radius.py`;
  - smoothing and the products cache, which stay.

  The moved parts are re-exported from `pipeline.py`.
- Risk: low. Effort: M. Verify: G-move.

**P5.8 Stale configuration copies.**
- About 10 plot scripts copy `config.SMOOTHING_SIGMA_CELLS`, `DPI` and `SNAPSHOT_TO_REDSHIFT` at import, while
  `pipeline.set_smoothing` rewrites `config` at run time.
- Read the values where they are used. Then add type hints and docstrings to the public functions of `pipeline.py`
  (1 of 39 hinted) and `figures.py` (0 of 24).
- Risk: low. Effort: M.

**P5.9 Bash strictness.**
- The drivers differ:
  - `run_ps_dtfe.sh` has **no** `set` line;
  - `run_dtfe.sh` and `download_snapshots.sh` use `set -uo pipefail`;
  - `run_ps_pipeline.sh` uses `set -u`.
- Change:
  1. Give `run_ps_dtfe.sh` `set -uo pipefail` after an audit of every `${VAR}`.
  2. Explain in `config.sh` why the batch drivers leave out `-e`: they continue after a failed snapshot.
  3. Move the repeated `-d`/`-s`/`-h` getopts handling and the data-root and simulation checks into one `config.sh`
     function.
- Keep `/bin/bash` 3.2 compatibility.
- Risk: medium. A missed unset variable would abort a batch.
- Verify: `run_scripts_check.sh`, `DRY_RUN=1` of every driver. Effort: M.

---

## P6. One flag scheme, with a deprecation path

**The scheme.**
- `ps-` marks only what is meaningful with phase-space streams, Lagrangian space or caustics.
- Every concept the two estimators share gets one name in both binaries.

**Shared concepts with two names today:**

| Concept | DTFE today | PS-DTFE today | Canonical (both) |
|---|---|---|---|
| deposit / interpolation on the GPU | `--gpu` | `--ps-gpu` | `--gpu` (both already bind the same member, `gpuAlias`) |
| exact cell averages | `--exact-average` | `--ps-exact-deposit` | `--exact-average` (see D2) |
| no look-ahead during a GPU deposit | `--no-overlap` | `--ps-no-overlap` | `--no-overlap` |
| spelling | `--full_help`, `--lambda_th` | same | `--full-help`, `--lambda-th` |

**Kept as they are:**
- the true PS concepts: `--ps-caustics`, `--ps-caustic-cusps`, `--ps-vertex-mass`, `--ps-volume-weighted`,
  `--ps-linear-deposit`, `--ps-halo-release`, `--ps-alpha-shape`, `--ps-lag-padding`, `--ps-stream-density`,
  `--ps-window`;
- the camelCase upstream flags (`--MpcUnit`, `--paddingMpc`, `--partNo`, `--regionMpc`, …);
- the PS-only flags without a prefix: `--avg-subsamples`, the merge flags, `--pts-*`, `--per-stream*`.

Renaming those costs more than it gains.

**Order: the deprecation path.** Each step lands on its own, in this order.

**P6.1 A reader normaliser first.**
- Every reader of command lines must map the old spellings to the canonical ones **before** anything emits the new
  ones:
  - `python/gui/results.py:has()` and its `_NSUB` parser;
  - `python/dtfelib/io.py` (it parses `--lambda_th`);
  - **`scripts/run_ps_pipeline.sh:run_settings`**.
- The last one is the important one. It compares the flags in a run log's first line with the new command to decide
  whether a snapshot is stale. If a driver emits `--gpu` while the logs on disk say `--ps-gpu`, every snapshot is
  recomputed: hours of work.
- Verify: a test that feeds an old-spelling log and a new-spelling command to `run_settings` and expects "unchanged".
- Risk: high if skipped. Effort: S–M.

**P6.2 Hidden aliases in the binaries.**
- Register the old names in the existing hidden `options_description`, each bound to a shadow member.
- Add one `foldDeprecatedAliases(vm)` step at the start of the `checkGpuOptions` phase in `user_options.cc`:
  - it ORs a switch or copies a value onto the canonical option;
  - it errors when both spellings are given with conflicting values;
  - it prints one `warning: --ps-gpu is deprecated, use --gpu` line.
- All validation stays on the canonical names. Config files (`parse_config_file`) accept hidden names automatically.
- **Canonicalise `commandTokens` before `partition_checkpoint.h:commandKeyLines`**, and add the new spellings to its
  drop lists. Otherwise `--gpu` and `--ps-gpu` runs get different checkpoint keys.
- Help texts: `--gpu` is no longer "standard DTFE only".
- Risk: medium (the checkpoint key and `--resume`). Effort: M.
- Verify:
  - `dtfe_resume_check`, `ps_resume_check`;
  - a new check that both spellings give the same key and the same outputs (G-byte);
  - the help assertions in `run_scripts_check.sh` (about lines 690–695, which assert that the standard `--help` lacks
    `--ps-gpu`; update them);
  - a config file that uses an old name.
- Depends: P6.1, P3.3.

**P6.3 Drivers and environment.**
- The drivers emit the canonical flags: `run_dtfe.sh`, `run_ps_dtfe.sh`, `run_ps_pipeline.sh`.
- The environment knobs become `DTFE_GPU` / `PS_GPU` and `DTFE_EXACT` / `PS_EXACT`. The old names are read after them:
  `${PS_GPU:-${PS_METAL:-auto}}`, `DTFE_METAL`, `DTFE_EXACT_AVERAGE`.
- Banner strings change in the same commit as the tests that grep them:
  - `GPU (--ps-gpu)` in `ps_auto_tune_check.sh`;
  - `GPU (--gpu)` in `run_scripts_check.sh`;
  - the `(--ps-exact-deposit; …)` message.
- Also fix a trap: the drivers' `-m` means GPU and `-s` means simulation, while the binaries' `-m` means method and
  `-s` means samples. Document it in the drivers' `--help` at least.
- Risk: medium. Effort: S–M. Verify: `run_scripts_check`, `ps_auto_tune_check`, G-docs.

**P6.4 Launcher.**
- Emit the canonical flags in:
  - `runspec_custom.py`;
  - `explore_zoom.py`;
  - `dtfelib/skeleton.py`;
  - the `OPTION_TEXT` keys in `presets.py`;
  - the environment exports in `runspec.py` and `runspec_pipeline.py`. Keep the old names exported as well for one
    release.
- `RunSpec.to_dict()` keys, which are the saved settings, do not change.
- `flag_help` parses `--full_help` by name, so its mapping must follow the rename.
- Verify: `py_gui_runspec_test` (every emitted flag exists in the binaries' help), the tooltip checks in
  `py_gui_app_test`.
- Effort: M.

**P6.5 Docs and tests.**
- Update the README flag table, `demo/COMMANDS.md`, `change_log.txt` (a "Deprecated" list with the old→new table)
  and `tests/run_tests.py` (it gates a feature on `--full_help` content).
- `docs_drift_check.py` part A must reject deprecated names in the docs. It currently validates against every
  registered name, hidden ones included.
- The ~40 test scripts may keep the old spellings; they then exercise the aliases.
- Effort: M.

**P6.6 One release later: remove the aliases.** The precedent is the removal of `--metal` / `--ps-metal`. Effort: S.

**Not adopted:** a `--deposit exact|sampled|linear` enum. It merges three switches that have different exclusion rules
(`user_options.cc`, the exact/linear checks), and it changes run-log text, the checkpoint key and the launcher mapping
for little gain. Revisit once the aliases have shipped.

---

## P7. Launcher polish (excluding the 3D view)

The two layout bugs are P1.1 and P1.2. Gate for every item: G-gui.

**P7.1 Style helpers in `app_widgets.py`.**
- Spacing constants: `GAP`, `GROUP_MARGIN`, `FORM_V_SPACING`, `COL_GAP`. They replace about nine scattered
  `setContentsMargins` / `setSpacing` calls.
- `group(title, *rows, collapsed=False)`, plus a collapsible `advanced_group(title)` that remembers its state under
  a new settings key.
- `labelled_row(label, widget, unit=None, tip=None)`: units as spin-box suffixes.
- `field_grid(names, columns=3)`.
- `grid_row()`: the "Grid ³ cells / sub-samples per axis" row, of which three copies exist today.
- `run_buttons()`.
- Effort: M.

**P7.2 Less on screen by default.**
- "Resources and output" takes 272–327 px on every job tab and is rarely changed. It holds partition, concurrent
  partitions, scratch dir, parallel triangulation, ordered merge and start over. Collapse it, with a one-line summary
  ("auto plan, 4 partitions in flight").
- Move the expert options into Advanced:
  - "Also mark cusps and swallowtails";
  - "Linear density inside each tetrahedron";
  - "Fixed density range";
  - "Re-render figures newer than their data";
  - the Custom tab's "stream by stream (ragged files)" and "each stream's tetrahedron IDs".
- On the Plots tab, the void figure sets repeat "Keep overlapping voids" and "each particle in one void". Show them as
  one shared "Void options" row.
- See D8. Effort: M.

**P7.3 A stable top panel.**
- The shared Data panel is tall on the Grids and Plots tabs and short on the others, so the tab bar jumps by about
  250 px between tabs.
- Put the tab bar above the panel, or give the panel a fixed height. Drop the redundant "Data" title, and move
  "Setup…" into the Data root row.
- Where: `app.py:__init__` (left-pane assembly).
- Risk: medium; the tests use tab indices and `_tab_view`. Effort: M.

**P7.4 One wording table.**
- One `LABELS` dict for the texts that differ between tabs today:
  - field names: "density" vs "density ρ/ρ̄";
  - group titles: "Grid and fields" vs "Estimator and grid" + "Fields" vs "Grids";
  - quick-selection buttons: "Not yet run" / "Not merged" / "With grids".
- Sentence case, nouns for groups, verbs for buttons, "…" only when a dialog opens.
- Tooltips at most two lines; the long explanations go to Help. The `gpu`, `lam` and Pipeline-fields tooltips are
  essays today.
- Risk: low (some tests match label text). Effort: S–M.

**P7.5 One estimator group builder.**
- About 60% of `_build_own_tab` (203 lines), `_build_grids_tab` and `_build_pipeline_tab` (191) is the same widget
  construction under different attribute names.
- Change: a table-driven `_estimator_group(prefix)` that removes about 300 lines and keeps the old attribute names
  (`o_*`, `p_*`, `field_checks`) as aliases, because the tests reach widgets by name.
- Risk: medium. Effort: M. Depends: P7.1.

**P7.6 Long UI builders split into private `_build_*` methods**, with attribute names kept:
- `explore.py:__init__`, 497 lines;
- `figsave.py:_dialog_class`, 344, with a 210-line nested `__init__`;
- `app.py:__init__`, 196;
- `setup_wizard.py:__init__`, 147;
- `app_menus.py:_build_menus`, 125.

Move the logic out of the UI modules: `explore_zoom.py:exact_zoom_plan` and `explore_grid.py:grid_figure` become
plain functions.
- Coordinate `explore*.py` with the 3D-view branch.
- Risk: low. Effort: M.

**P7.7 Explore details.**
- The "Voids" control appears twice.
- In the stream table, the value column is too wide and the "ρ/ρ̄" header is truncated.
- The save buttons sit in the right pane, away from the controls.
- The log strip's "!" button has no tooltip.
- Effort: S.

---

## P8. Auto-tuner

### What the real runs say

The 65 run logs on the data disk come from the development machine (64 GB, 10 cores, M1 Max). Each carries an
`AUTO-TUNE` line and a summary line comparing the predicted with the measured peak footprint. Error = (predicted −
measured) / measured; negative means the model **under**-predicts, which risks swapping.

| Runs | Build | Plan | Error vs peak footprint |
|---|---|---|---|
| PS, GPU sampled + 8192² sample points, TNG50/100/300-3-Dark, 27 snapshots | before the hand-out-order fix | auto (5³–6³, mc 6–10) | mean −36%; early snapshots −3…−22%, z = 0 −45…−69%. The error tracks the parked-partition set (up to 100 GB logged); TNG50 z = 0 took 2.4 h at 13% CPU |
| PS, CPU exact + sample points, TNG100-3-Dark, 9 snapshots | current, with the exact-deposit work | auto (5³–6³, mc 4–10) | +7…+31% (mean +16%), over-predicting |
| PS, CPU exact, no points, TNG100-3-Dark z = 0 at 1024³ with a scratch dir | before `ZeroFillScope` | auto 6³ mc 10 | −43% (29.6 GB predicted, 52.1 measured) |
| PS, GPU sampled, no points, TNG100-3-Dark z = 0 | older | auto 6³ mc 9 | −3% |
| Standard DTFE, GPU exact average, 27 snapshots | older | **all hand-set** (3³/4³/5³) | no prediction to compare; CPU use 20–45% |

Points read from the code (`auto_tune.h`, patched state):

- **The GPU sampled class is mostly history.** Those builds handed partitions out as a static steal, so nearly every
  partition parked behind the first thread's block. `schedule(monotonic: dynamic, 1)` (in `DTFE.cpp`
  `psRunPartitionLoop`) and the spill budget fixed that. The −45…−69% must be re-measured (P2.3) before the model
  changes.
- **The CPU exact deposit has no term of its own.** In `AutoTunePSPlanner::partitionBytes`, the exact deposit's
  per-tet weights and item list are booked only in the `metalActive` branch. The CPU branch books the sampled
  deposit's copies or slab lists only. *Verified by reading.*
- **The point-stream budget is about 2× generous.** The estimator (`psEstimateMeanStreams`) is good: estimated vs real
  mean streams per point at z = 0 are 4.00 vs 4.25, 5.79 vs 5.38 and 2.06 vs 1.90. The budget
  `1 + 2(est − 1)` then roughly doubles the excess. That slack is what hides the missing CPU exact term at 512³.
- **The budget reacts to free memory.** One exact snapshot got a 29.2 GB budget instead of 41.2 GB (free − 4 GB) and
  ran at mc 4 / 60% CPU. The log does not say why.
- `targetC = cores` for GPU runs too (around line 1987): the CPU work around the serialised GPU deposit runs in
  parallel, so the GPU planner does not cap concurrency at a small number.

### Items

**P8.1 Re-baseline the error table** from the P2.3 reruns, per class: standard / PS, CPU / GPU, exact / sampled,
with / without points, one domain / partitioned / window.
- Accept band: −5% to +15% of the peak footprint.
- Depends: P2.3. Effort: S.

**P8.2 A CPU exact term.**
- Where: the non-Metal branch of `partitionBytes()`, booking the r3d per-thread scratch and the per-tet weight totals
  of the CPU exact path.
- Calibrate from the P2.3 1024³ rerun and a 512³ exact run without points.
- Verify: G-tuner (only exact lines change), G-real. Effort: M.

**P8.3 Only if P8.1 still shows it: parked and spilled partitions in the prediction.**
- Book `min(park room, staged bound)` in `finishReport()`. The staged bound is the in-flight sub-grid bytes per waiting
  partition.
- Risk: lower concurrency, so slower runs. Effort: M.

**P8.4 Scratch-file pages.**
- With `--scratch-dir` the grid term leaves `fixed`, but pages the run dirties still count as resident.
- Add an allowance only if the post-`ZeroFillScope` rerun still shows the gap. Print both footprint and RSS (the
  summary already has both).
- Effort: S.

**P8.5 Tighten the point-stream budget** from `1 + 2(est − 1)` to about `1 + 1.3(est − 1)`.
- Only after P8.2: otherwise the 1024³-class runs lose their hidden slack.
- Verify: the three z = 0 point sets above. Effort: S.

**P8.6 CPU `--ps-window` runs.**
- They are still under-predicted: 0.33 / 0.98 GB predicted vs 0.44 / 1.43 GB measured at 2048² / 4096², because
  their window buffers are per partition and the allocator keeps the freed window blocks.
- Effort: S–M.

**P8.7 Log why the budget is what it is.** Print the free-memory figure that set it, and its source: the live free
memory or the `DTFE_MEM_*` knobs. Effort: S.

**P8.8 A time objective for N ≥ 2·10⁶.**
- Today the planner takes the coarsest split at which every core fits, and only N < 2·10⁶ has a time model.
- Proposed objective:
  `T(n, c) = ⌈n³/c⌉ · (t_tri + t_pts + t_vdens) + T_deposit + T_merge`.
  - `t_tri = f_pad · N/n³ · t_vertex · (1 + 0.04(c − 1))`. The contention term fits the logs: 3.4–4.9 µs per vertex
    as c grows.
  - `T_deposit` depends on the device:
    - GPU: `max(serialised GPU queue, CPU work)`;
    - CPU exact: `Σ tets · cells/tet · t_r3d / cores_eff(c)`.
  - `T_merge`: the staged bytes times the spill rate.
- Choose the plan with the smallest T that fits the memory budget.
- Calibrate per-unit costs from the P2.3 logs' phase lines.
- Depends: P8.2, and P9.2, which changes `cores_eff`. Effort: L.
- Verify: G-real, e.g. a 4³ / 5³ / 6³ × mc sweep on one snapshot, and CPU exact at mc 4 vs mc 10.

**P8.9 Standard DTFE: a prediction line and a time model.**
- All 27 standard production runs were hand-set and used 20–45% of the CPU. This is the largest untapped speed-up for
  the standard estimator.
- Steps:
  1. Check the standard planner's prediction on the P2.3 run.
  2. Give it the P8.8 objective, with the standard's serial partitions and threads per partition.
- Effort: L.

---

## P9. Optimisations

Each item is byte-identical by construction unless it says otherwise. Every claimed gain is to be measured: CPU on the
synthetic 2.1M-particle box at 256³ and on TNG100-3-Dark z = 0 at 512³; GPU with one partition and
`PS_GPU_TIMING=1`.

### CPU

**P9.1 A parallel, row-wise sub-grid merge.**
- Where:
  - `quantities.cc:addFieldSubgrid` and `orFieldSubgrid`. They run about 17 fields under `mergeMutex`, and for every
    cell they do NO_DIM 64-bit divisions/modulos plus a `subgridTarget` call.
  - `normalizePhaseSpace`.
  - The density-scale and stream-statistics loops in `ps_interpolation.cc`.

  None of these is threaded today.
- Change: copy row by row, with the destination row start computed once and a split only at a periodic wrap, then add
  `#pragma omp parallel for` over rows.
- Byte-identical: every element is still added once, in the same order per element.
- Gain: large on the merge itself *(est. ≥10×)*; on the wall, whatever P2.5 measures the merge share to be.
- Risk: low. Effort: S.
- Verify: G-byte, `ps_periodic_seam_check`, `ps_partition_padding_check`, `ps_window_check`.

**P9.2 Deposit threads that follow the free cores.**
- Today `psDepositThreads()` is fixed when a deposit starts, and the block-to-thread assignment is static
  (`idx++ / DEP_BLOCK % nT`). The cores of finished partitions sit idle.
  - *Measured*: 58% core use on a partitioned CPU run.
  - *Measured*: with two deposit threads per partition, 31.0 → 25.4 s, byte-identical.
- Change, in the default single build (integer fixed-point atomics, no private copies):
  - claim blocks from a shared atomic counter;
  - size each team from a process-wide pool of cores.
- Also stop each thread walking the whole CGAL cell iterator to skip the blocks it does not own: build the
  cell-handle vector once (8 B per cell, *est.* 3–8% of the deposit).
- The double build and the float-moment modes keep the static rule, because their sums depend on the assignment.
- Gain: +15–20% on CPU PS runs *(est., from the measurement)*. This matters more now that PS exact runs on the CPU by
  default.
- Risk: medium. Effort: M.
- Verify:
  - G-byte;
  - `DTFE_DEPOSIT_THREADS=1` vs the new default;
  - `ps_slab_deposit_check`, `ps_parallel_check`;
  - the double build kept on the static path.
- Depends: P3.1.

**P9.3 Threaded Monte-Carlo average for standard DTFE.**
- Where: `averaged_interpolation_1.cc`, around line 952: `nT = exactAvg ? exactAverageThreads() : 1`.
- Change: compute each sample's x index first and skip samples outside the thread's slab before evaluating anything.
  Each cell then still accumulates in tetrahedron order.
- Secondary, since the production default is the exact average.
- Risk: medium. Effort: M.
- Verify: `dtfe_mc_average_check`, G-byte at 1, 4 and 8 threads.

**P9.4 (upstream, D5) The padding test's dummy points.** Replace them by a per-cell circumsphere test, only if P2.4
shows a real cost. This changes the triangulation and the `.badGridPoints` sets, so it is a semantic change.

### GPU

**P9.5 Copy-back and releases after the lock.**
- In both `psGpuDepositFields`, turn the `lock_guard` into a `unique_lock` and unlock after `psHostReportPasses`.
  The copy-back, the deferred scan and the buffer releases are private to the call.
- On CUDA, synchronise the stream first.
- Gain: 10–25 s of the 266 s *(est.)*.
- Risk: low. Effort: S. Verify: G-byte, G-gpu, `gpu_retry_check`.

**P9.6 Build the exact items before taking the lock, in compact form.**
- Move plan, grid init, parameters, `psExactItemBounds` and the item fill above the lock. The CUDA host already builds
  its items from the host vertices before the upload.
- `ctx()` then needs `std::call_once`.
- Replace the 8 B-per-item list (up to 256 MB) with the `itemStart` + `blockTet` map that the standard-DTFE kernel
  already uses (`metal/dtfe_deposit.metal`, `dtfe_gpu_fixed.h`): about 24 MB, the same (tet, block) pairs in the same
  order. The compact map is also what keeps the waiting partitions' memory small.
- Gain: 60–90 s *(est.)*.
- Risk: medium; the kernel does an extra binary search, so benchmark.
- Verify: G-gpu, `validate_deposit` T14, G-emu.
- Depends: P4.1, P2.1.

**P9.7 Prefetch the next partition's uploads (decide from the P2.1 timers).**
- A second `stageMutex`: always take stage before ctx.
- It holds one more partition's device buffers (about 0.5 GB), to be added once to the tuner's `snapshotBytes`.
- Gain: up to about 60 s *(est.)*. Effort: M.

Together, P9.5–P9.7 *(est.)* save 150–190 s of the 266 s. That brings the 94M-particle GPU exact run from 17m39s to
about 15 min, near the CPU's 14m27s, so the CPU default for PS exact stays unless P2.3-style reruns show otherwise.

**P9.8 Fewer atomics in the sampled kernel.**
- The nSub³ = 27 samples of one tetrahedron that land in one cell share the same constant fixed-point increments:
  mass share, momentum weight, the gradient terms, the scalar gradient and the stream volume.
- Change: one atomic of n·q instead of n atomics of q. Integer adds are exact, so fixed-point output is
  byte-identical. Position-dependent grids keep their per-sample atomics.
- Gain: about 20–55% fewer atomics *(est.)*. This is the best lever for the sampled production deposit, which is
  kernel-bound (about 94% GPU busy).
- Check P2.2's atomic share first.
- Risk: medium. Effort: M. Verify: G-byte, `ps_fixed_moments_check`, `gpu_mass_check`, G-emu.

**P9.9 (Metal) Specialise the PS pipelines on the field flags** with function constants, as the DTFE host already does
(`FC_*`).
- Gain: occupancy, for runs that request fewer fields *(est. 5–15%)*.
- Risk: code generation may move low bits; verify with G-gpu. Effort: M.

---

## Decisions for the owner

Each decision has a recommended default, so the plan can proceed if no answer comes.

| ID | Decision | Recommended default | Alternative |
|---|---|---|---|
| D1 | `PS_GPU_CPU_SHARE` prototype | **delete** (P1.11): 4.5% at best, untested, 120 lines in the GPU walk | keep it opt-in and add a test plus a README line |
| D2 | canonical name for the exact deposit in both binaries | **`--exact-average`**: no change for standard DTFE's production command | `--exact` in both, with both old names as aliases |
| D3 | legacy switches | keep `DTFE_MC_LEGACY` and `DTFE_TESS_CACHE_LEGACY` while grids made the old way exist; **drop the `*_METAL_*` knob aliases** after P1.10 | keep all |
| D4 | always-on macros `VELOCITY`, `SCALAR`, `TEST_PADDING` and the `*_OPTIONS` help macros (**upstream**) | **leave them**; their off-states are not built anywhere, so do not advertise them | delete the off arms, or add a CI compile of each off-state |
| D5 | padding-test dummy points (**upstream**, semantic) | **keep**, unless P2.4 shows over 5% of a standard run | P9.4 |
| D6 | `--fixed-point-merge` as default | **leave it off** if the P2.3 reruns park at most a handful of partitions | make it the default for PS runs above some N |
| D7 | split `depositLoop` (1010 lines, hot) | **no**, unless P9.2 needs the restructuring; the proof would be bytes plus disassembly plus a ±1% timing check | split into a `DepositCtx` plus `psDepositExactTet` |
| D8 | launcher Advanced sections | **collapse** Resources and the expert PS options, with the state remembered | only shorten labels and tooltips |
| D9 | when to remove the flag aliases | **the release after** the one that introduces them | keep them indefinitely, hidden |
| D10 | **upstream** dedupe and deletions (P1.4's upstream half, P3.9) | **golden tests yes**; the dedupe and deletions **no**, since the repository is still a fork of upstream | do both |

## Not proposed, and why

| Not proposed | Why |
|---|---|
| A hybrid GPU+CPU deposit by tetrahedron class, or a CPU share of the GPU's tetrahedra | Measured at most about 1.2× and 4.5% on the real data. |
| Narrowing the GPU lock for **sampled** runs | Measured 0.2 s per deposit beyond the kernel. (Exact runs differ: see P9.5–P9.7.) |
| GPU exact ideas already measured and rejected | Column cache, threadgroup-memory polyhedra, one-pass piece storage, full SAT rejection, register r3d. |
| Skipping exact phase 0 by using the analytic tetrahedron volume | It changes bytes, leaves a ~1e-7 per-tet mass leak and removes the route for degenerate tetrahedra. |
| Shorter GPU pacing gaps | About 3% of kernel time, traded against the display watchdog. |
| `--deposit` enum, camelCase flag renames | See P6. |
| A shared helper for the heredoc Python blocks in the shell tests | Boilerplate only: about 150 lines saved for 38 suites to re-prove. |
| `logging` instead of `print` in `dtfelib` and the plot scripts | The launcher parses their stdout. |
| `-MMD` dependencies for the `.cu` rules | Cannot be tested without nvcc/hipcc. Keep the hand-listed headers. |
