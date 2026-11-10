# Lamina: cleanup and polish plan

Status (2026-11-09): P0, P1, P6 (P6.6, the alias removal, is for the next release), P7 and P10.1–P10.7 done; most of P2–P5 and P9
done (each phase's status line says what is open); phase R's calibration night ran 2026-11-08/09 (the R status; P10.8, decided and done on 2026-11-09 (D16), D15 settled the same evening by D20 (exact; P12, committed as `39583f0`); P13 (the run modes, added 2026-11-10: committed as `9bfa39a`); P11.7 items 1 (the PS tuner's face sums), 2, 4, 5 and 6 done 2026-11-10 (item 3, the pinned runs, open); the rest of R was not launched); P8
partly done (P8.1 for the classes phase R ran; P8.2 as the partition span and the start-up zero fill, P8.5, P8.7, P8.9 step 1 and P8.10 in code, gates green on 2026-11-09 (G-battery 32 of 32 over two runs, G-linux); P8.3 and P8.4 not needed; the rest open, see the P8 status). **Added 2026-11-07 (afternoon), owner's order:** P10 (shared exact velocity derivatives, the face method)
joins the plan, and **every code item is finished and checked on small data before any rerun**; the reruns then run
once, as one campaign (phase R, after P10). P0.1 and P0.2 landed in `60a072a`; P0.3's CI fixes and P0.4 landed in `70e153e` and `38eb4be`; P0 closed on
2026-11-08: CI green on `38eb4be` (all seven jobs, once the job that had timed out on a package mirror was rerun), `ref0`
frozen, the P0.4 sentence in change_log. Since 2026-11-08 CI runs three jobs (the CUDA/HIP jobs were removed on the
owner's instruction; D11). The first CI run of the face method (master fast-forwarded to `1377342`, 2026-11-08) failed
`face_sweep_check` F2 on the x86 runner by rounding alone (1.43e-10 against a 1e-10 bound; 3e-11 on arm64): the bound
is 1e-9 since the commit after it. D11 was decided the same evening: the CUDA/HIP port is removed, the tree is
Metal-only, and the port's last state is the branch `archive/cuda-hip-2026-11-08`. The plan covers the C++
binaries (DTFE, PS-DTFE and their double and 2D builds), the Metal GPU code (and, until D11, the CUDA/HIP port), the driver scripts,
`python/dtfelib`, the plot scripts and tools, and the PySide6 launcher in `python/gui`.

**Surveyed base: master `60a072a`** (the cleanup itself starts from `70e153e`, below). The survey read commit `5ac1ce1` with the work that `60a072a` then committed already
applied:

- the faster exact deposits: the CPU r3d sweep `exact_sweep.h`, the GPU register z-sweep, the GPU exact-dispersion
  fix, and PS exact on the CPU by default;
- the `--ps-window` auto-tune fix (the read residue and window sub-grids);
- an earlier version of the `download_snapshots.sh` https fix.

`60a072a` also brought three changes the survey did not see. The plan was re-checked against all three:

- the `--scratch-dir` zero-fill fix (`ScratchAlloc::ZeroFillScope`);
- the Explore 3D view as a translucent volume (`python/gui/volume3d.py` new; `explore_3d.py`, `view3d.py` rewritten);
- a reworked download fix (`download_snapshots.sh:fetch_listing`), which replaces the surveyed version.

Every other file is byte-identical to the surveyed tree.

**Re-checked against master `70e153e`.** It changes 9 files of `60a072a`:

- the CI fix: `src/main.cpp` line-buffers stderr, `tests/ps_mass_exact_check.sh` (X4) and `tests/cudaemu_check.sh`;
- the 3D volume's speed-ups and two-step settle (`python/gui/volume3d.py`, `explore_3d.py`);
- the OpenGL renderer for the 3D volume (`python/gui/volume_gl.py` new; `explore_3d.py`; GPU-vs-QPainter tests in
  `tests/py_gui_explore_test.py`);
- `change_log.txt` and `.gitignore`.

Every item that names one of these files was re-checked and its line numbers updated; every other item still holds
as written, since its files are unchanged. Line numbers are from `70e153e` for those files and from `60a072a`
elsewhere, and will drift. Items therefore name `file:function` first and give line numbers in brackets.

**Scope rules carried over from the earlier cleanup rounds:**

- The binaries keep the names DTFE / PS-DTFE and the Python package stays `dtfelib`. The project is called Lamina.
- No default simulation and no default data root anywhere.
- The launcher's QSettings key stays `("DTFE", "launcher")`.
- The launcher's 3D view (`python/gui/explore_3d.py`, `view3d.py`, `volume3d.py`, `volume_gl.py`) is in scope
  (P7.8–P7.12).
  - Two renderers. QPainter is the reference: it draws offscreen (the tests), for Save image, and wherever no OpenGL
    4.1 context can be made (or `DTFE_GUI_NO_GL=1`). On screen `volume_gl.py` draws the volume on the GPU. Any GL
    error falls back to QPainter for the session.
  - A 3D item must not change the rendered pixels. Part 11 of `tests/py_gui_explore_test.py` compares QPainter's
    16-bit compositing with `volume3d.composite`, and parts 10–11 must keep passing unchanged.
  - The GPU must stay within its tolerance of QPainter: `volume_gl_parity` in the same suite, run in a child process
    on the default Qt platform. It SKIPs without a display or OpenGL, so CI never runs it: run it on the Mac for
    any item that touches the volume's drawing.
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
| **G-build** | all five build modes: plain, `METAL=1`, `METAL=1 DOUBLE=1`, `DIM=2`, `make library` (and `CUDAEMU=1` until D11, 2026-11-08) | 0 warnings, 0 errors |
| **G-byte** | `tests/dev/bytecmp.sh <ref0> <out>` against the reference binaries taken in P0.3 | every file identical (184 output files after P0; 261 with `--gpu-exact`) |
| **G-gpu** | `tests/dev/bytecmp.sh --gpu-exact` (P1.12), plus `metal/validate_deposit` | identical bytes before and after a host-only change; 31/31 |
| **G-emu** | `CUDAEMU=1` build + `tests/cudaemu_check.sh` at warps 32 and 64; local only since 2026-11-08 (CI no longer builds the port: D11). **Retired 2026-11-08** with the port (D11: removed; the emulator and the suite are on `archive/cuda-hip-2026-11-08`) | PASS (warp 32 failed on `60a072a`; fixed in `70e153e`: P0.3) |
| **G-battery** | `tests/dev/battery.sh` | 31/31, tree unchanged during the run |
| **G-linux** | CI's Linux stages in a local Ubuntu 24.04 container: `tests/ci_suite.sh build`, `core`, `double`, `2d`, `variants`, the old-python stage (python 3.10) and ruff (CI's three jobs since 2026-11-08); `CUDAEMU=1` + cudaemu_check was an opt-in local stage until D11 | PASS; required before every push. glibc differs from macOS where the tests can see it: it writes a long unbuffered stderr line in 128-byte pieces (P0.3) |
| **G-gui** | the four GUI suites (`tests/py_gui_app_test.py`, `py_gui_app2_test.py`, `py_gui_runspec_test.py`, `py_gui_explore_test.py`), an offscreen render of every tab at two window sizes, and one look in a real macOS window | suites pass, with `volume_gl_parity` run (not SKIPped) on the Mac; no clipping or overlap in the renders |
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
| **P0** | Close the baseline: the work landed in `60a072a`; its red CI job fixed in `70e153e`; freeze reference binaries | no | CI green, G-battery, G-byte vs a `5ac1ce1` build | — | S–M (mostly machine time) |
| **P1** | Quick fixes and dead code (incl. two GUI layout bugs) | no | G-build, G-byte, G-gpu, G-gui | P0 | M |
| **P2** | Measure first: timers, a debug knob, real reruns | no (instrumentation only) | G-byte; the reruns are the product | P0 | M + machine hours |
| **P3** | C++ structure: shared helpers, splits, tuner to `.cc` | no | G-byte, G-move, G-tuner, G-battery | P1 | L |
| **P4** | GPU structure: shared host code, one params header, twin check | no | G-byte, G-gpu, retry suites (G-emu until D11) | P1 | L |
| **P5** | Python, plots, tools, bash drivers | no (same figures and data) | py suites, `run_scripts_check.sh`, G-docs, G-move | P0 | L |
| **P6** | One flag scheme with a deprecation path | no (names only) | resume suites, `run_scripts_check.sh`, runspec suite, G-docs, G-byte | P3 (P3.3) | M–L |
| **P7** | Launcher polish, including the 3D view | no | G-gui | P1, P6 | L |
| **P8** | Auto-tuner calibration and a time objective | plans change; outputs do not (checked with the plan pinned) | G-tuner, G-real | P2, P3.8 | L |
| **P9** | CPU and GPU optimisations | no (each item is byte-identical by design) | G-byte, G-gpu, G-real timing (G-emu until D11) | P2, P4.1 | L |
| **P10** | Shared exact velocity derivatives (the face method) for both estimators, PS mean-flow gradient and mass flux | standard: no (same gradient), except P10.8 (the standard exact-average grids move by rounding: 6e-8 of the rms on the CPU exact bytecmp runs, 2e-16 in the double build; the 3³ plan's strip by up to 1e-2 in single cells; the P10 status has the 512³ figures); PS: new and changed velocity-derivative grids | G-build, G-byte (standard), the P10 tests, G-battery, G-linux | P3, P9.2 | L |
| **R** | The rerun campaign, once, after every code item: P2.3's reruns, P8.1/P8.2's calibration, the PS exact reruns with the P10 grids, the science comparisons. **Calibration night done 2026-11-08/09** (the R status); the rest is not launched (D16 decided and P10.8 done on 2026-11-09, D15 settled by D20) | the products | G-real | all code items | machine nights |

**Order (owner, 2026-11-07).** All code first: the open code items of P3–P9 and all of P10, each checked on small
data (the synthetic and 64³–256³ test sets, `--auto-tune-report` plans). Then phase R runs the real simulations once.
Constants that only real runs can fit (P8.2's CPU exact term, P8.1's band check) are written in code with their
current best values and refitted in R; a refit is a constants-only change.

**Parallel work.** P3, P4 and P5 touch disjoint files, so they can run as separate sessions in separate worktrees.
Some files are shared between phases and must land in one order:

| Shared file or group | Order |
|---|---|
| `ps_interpolation.cc` | P1.3 / P1.11 → P3.5 → P9.2 |
| `ps_metal_host.cc` / `ps_gpu_cuda.cu` | P4.1 → P9.5–P9.7 |
| `auto_tune.h` | P3.8 → P8 |
| `partition_checkpoint.h` | P3.3 → P6.2 |
| `quantities.cc` / `DTFE.cpp:psReserveGrids` | P3.3 → P9.1 |

---

## P0. Close the baseline

**P0.1 The exact-deposit work and the zero-fill fix — done in `60a072a`.**
- Landed:
  - `exact_sweep.h`;
  - the GPU z-sweep;
  - the CPU default for PS exact;
  - the `--scratch-dir` zero-fill fix: `ScratchAlloc::ZeroFillScope`, `zeroAssign` and `zeroResize` in
    `scratch_alloc.h`, with the mechanism in `scratch_alloc.cc` (`scratchTryMap` [82], `settlePending` [175], the
    scope's constructor and destructor [261–269]).
- The fix has 26 call sites:
  - 8 `zeroAssign` in `DTFE.cpp:psReserveGrids` [1222–1242];
  - 18 `zeroResize` in `quantities.cc`: `addField`, `orField`, `addFieldSubgrid`, `orFieldSubgrid` and
    `reserveMemory`.
- Find the sites with `grep -n 'zeroAssign\|zeroResize'`. `ZeroFillScope` itself appears only inside those two helpers.
- Still to run on the landed tree: G-byte against a `5ac1ce1` build (only GPU-exact files may differ), G-gpu
  (`validate_deposit` 31/31).

**P0.2 The window auto-tune fix and the download fix — done in `60a072a`.**
- Window fix: `auto_tune.h:autoTuneReadKeptBytes` [428], used at [2172–2174], tested by section W of
  `ps_auto_tune_check.sh`.
- The download fix that landed differs from the one surveyed:
  - `download_snapshots.sh:fetch_listing` [128–142] reads the product's JSON listing (`?format=json`), upgrades its
    http:// chunk links to https, and fetches exactly those links with `wget -i -`, trying the list up to three times.
  - It replaces both recursive crawls.
  - The fake `wget` in `run_scripts_check.sh` and the one in `py_gui_app_test.py` follow the new protocol.
- CI on `60a072a`: the Linux `build-and-test` job passed. Its core stage runs `ps_auto_tune_check`,
  `ps_window_check`, `ps_scratch_check` and `run_scripts_check`.

**P0.3 Make CI green and freeze the baseline — done 2026-11-08.**
- CI on `60a072a`: six of seven jobs passed; `cudaemu (32)` failed in `ps_mass_exact_check`.
- Step 1 is **done in `70e153e`**. The failure was not a warp-32 bug:
  - glibc 2.37 and later writes an `fprintf` to an unbuffered stderr in 128-byte pieces, each its own `write()`;
  - so one partition thread's `PS mass audit` line was cut at byte 128 by another thread's progress line;
  - the suite's Python then crashed on that line before its RESULT line;
  - a timing race (1 line in 24): warp 64 and the earlier runs were lucky; macOS writes the whole line.
  - The fix: `main()` line-buffers stderr first thing. X4 reports a line it cannot parse as a FAIL instead of
    crashing, and `cudaemu_check.sh` shows the last lines of a suite that fails without a FAIL line.
  - Every output file is byte-identical to `60a072a`'s (184 of 184).
- CI on `70e153e`: the five other jobs passed. Both `cudaemu` jobs crashed with SIGILL (exit 132, even on
  `--full_help`) before testing anything. **Fixed in `38eb4be`** (`.github/workflows/ci.yml` only):
  - Cause: the job built with the Makefile's default `-march=native` through ccache. Its cache, restored by key
    prefix, held objects compiled on another runner's CPU. ccache matches `-march=native` by its spelling, not by
    the instructions it picks.
  - Fix: a job-level `ARCH_FLAGS: -mtune=generic`, as `tests/ci_suite.sh` builds. It is job-level because the
    Makefile's flags stamp hashes `ARCH_FLAGS` and `dtfe_metal_check.sh` runs its own `make`. Also a new cache key
    prefix, and a 10-minute timeout on the job's package install (a stuck mirror had held both jobs for an hour).
  - Results are unchanged: GCC builds use `-ffp-contract=off`, so FMA availability does not matter. Checked in a
    Linux container: six emulator suites pass at warps 32 and 64.
- Step 2 is **done 2026-11-08**: the six binaries of `70e153e` (byte-identical source to `38eb4be`, which changes only
  the workflow) are kept as `ref0`, with checksums, in a folder outside the repository. Every byte-identical phase
  compares against it. Frozen at 01:55, after CI on `38eb4be` went green (checksums 6 of 6; the folder's README says so).
- Step 3 is **done**: a fixed list of 91 `--auto-tune-report` configurations, kept with `ref0`, with the script that
  makes it.
  - It covers both binaries and their double builds: RAM × grid × fields × sampled/exact × CPU/GPU, plus a
    hand-set split, `--ps-window`, sample points and a larger N.
  - What could vary is pinned: seeded synthetic snapshots, a simulated RAM size (`DTFE_RAM_GB`: no live
    free-memory term), `OMP_NUM_THREADS` and `DTFE_AUTO_MINN`. Two runs were byte-identical.
  - Standard DTFE's `--exact-average` does not change its lines, by design: the exact average allocates nothing
    extra.
  - The script is now `tests/dev/tune_report_set.sh` (P1.12).
- Step 4: save the GUI renders (P1.12 adds the script that makes them; until then this step waits).
- Gate: CI green on all seven jobs (the run on `38eb4be`), G-battery. **Met 2026-11-08 01:49**: `cudaemu (32)` had failed
  once on its 10-minute package-install timeout (a stuck mirror; nothing ran) and passed when rerun. Later that night the
  `cudaemu (32)`, `cudaemu (64)`, `cuda-compile` and `hip-compile` jobs were removed from `.github/workflows/ci.yml` on
  the owner's instruction (the port is not in use and those jobs kept stalling on the runners' mirrors); CI now runs
  `build-and-test`, `old-python` and `compile-variants`. The CUDA/HIP code stayed until D11 was decided, the same
  evening: removed (see D11).
- **D1** (keep or delete `PS_GPU_CPU_SHARE`) is decided: delete (P1.11).

**P0.4 Release notes for what `60a072a` left out — done in `70e153e`; the last sentence fixed with P1.11.**
- `70e153e` added the three missing entries (the `--ps-window` auto-tune fix, the `--scratch-dir` zero-fill fix,
  the download fix) and one for the stderr fix.
- Left: the 3D speed-up entry says a cube of 384 cells or more "now settles in two steps". Since the OpenGL renderer
  landed with it, that holds only where QPainter draws the volume. Qualify the sentence: "So where QPainter draws
  the volume (without OpenGL: see the next entry) a cube of 384 cells or more ...".
- Effort: S. Verify: G-docs.

---

## P1. Quick fixes and dead code (byte-identical)

Every item here is small and mechanical. Land them as one or two commits.

**Status 2026-11-07.** All done (P1.1–P1.18).
Gate: G-build, G-byte, G-gpu for the GPU items, and G-gui for P1.1 and P1.2.

**P1.1 Pipeline tab: the grid row is drawn on top of the λ_th row** *(bug, seen in the offscreen render)* — *done 2026-11-06.*
Grid, Deposit and GPU are in a "Grid and deposit" group of their own. (With 9 fields the λ_th row is row 3, not 4:
the Grid row shared it.)
- What and where: `python/gui/app_tabs.py:_build_pipeline_tab` (about lines 898–923). It adds the Grid, Deposit and
  GPU rows to the `_fields_group()` grid at fixed rows 3–5. `_fields_group` already puts the T-web threshold row at
  `(len(PS_FIELDS)+2)//3` = 4, so two rows share row 4.
- Fix: append at `g.layout().rowCount()`. Better, move Grid, Deposit and GPU into their own "Estimator and grid" group,
  as the Grids tab already does.
- Risk: low; the tests find the widgets by attribute (`p_grid`, `p_deposit`, `p_gpu`). Effort: S.

**P1.2 The right edge of the forms is clipped on the Grids, Custom, Pipeline and Plots tabs** *(bug)* — *done 2026-11-06.*
Measured: offscreen the forms needed 726–783 px of 666, in a real macOS window 732–795 of 682. The wide parts were
the "Resources and output" group (a check box beside the form's label column: label 150 + text 552 px) and two Plots
pages (a 281 px label column beside a 416 px row). Fixed: `_grow_fields` lets a combo box wider than 300 px shrink to
24 characters; the Plots pages' check boxes span the row; three labels are shorter, the rest in the tool tip. Now
601–617 of 666 offscreen, 615–666 of 682 natively. `py_gui_app_test` checks every tab fits (and fails on the old code).
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

**P1.3 Tautological dimension guards** — *done 2026-11-06.* `NO_DIM` can only be 2 or 3 (the Makefile and `define.h` reject anything
else), so `#if NO_DIM==3 || NO_DIM==2` is always true.
- Where:
  - `ps_point_eval.cc` (line 76), together with its never-compiled `#else` stubs (about lines 3207–3256) and an empty
    `#ifdef PHASE_SPACE` block;
  - `ps_exact_inside.h` (line 31);
  - `ps_interpolation.cc` (a 294-line block, about lines 3608–3902; only the guard lines go).
- Risk: none. Effort: S.

**P1.4 Functions with no caller** (checked by `grep -w` over src, metal, tests, python and scripts) — *done 2026-11-06.* Our two are
gone; the upstream ones stay (D10).
- `particle_data_traits.h`: `compareParticles`, `sameParticle`.
- Not dead: `triangulation_miscellaneous.h:vertexPositionMatrix` has a caller in `demo/examples/main_particle_density.cpp`
  (the first grep left `demo/` out). It stays.
- **upstream**:
  - `Box::isParticleLagrangianInBox`;
  - the `Particle_data` setters `setDelta`, `setLagrangianPosition`, `setScalar`, `setVelocity`, `setWeight`,
    `setPosition`;
  - in `miscellaneous.h`: `upperBoundCheck` and both `superfluous_options`;
  - in `src/kdtree`: `n_nearest_brute_force`, `r_count`, `r_count_around_point`, `r_nearest_around_point`.
- The upstream ones are optional, see D10.
- Risk: none. Effort: S. Verify: G-build, which must include the CUDA/HIP compile jobs on CI (until D11).

**P1.5 Declare `printElapsedTime` once** — *done 2026-11-06.* `subpartition.h` defines it, and `SPH_interpolation.cc`,
`NGP_interpolation.cc` and `grid_deposit_engine.h` each declared it again. Now all four include `src/elapsed_time.h`.
Effort: S.

**P1.6 Delete the `PS_COLUMNS` prototype** — *done 2026-11-06.* In `metal/ps_deposit.metal`: the doc comment and define (lines 20–29) and the
`#if PS_COLUMNS … #endif` block (about lines 1080–1184).
- Nothing defines `PS_COLUMNS` (no Makefile rule, no test), and the CUDA twin has no counterpart.
- Keep the `BeyondColumn` / `column_*` helpers: `count_beyond_grid` uses them.
- Risk: low. Kernel text that is never compiled changes nothing. Verify: G-gpu. Effort: S.

**P1.7 Stale comments after the GPU z-sweep replaced the GPU r3d port** — *done 2026-11-06.* The Makefile's GPU comment now
says the CUDA/HIP ports run under the CPU emulation on Linux and have not run on NVIDIA/AMD hardware.
- `gpu_host.h` (about lines 97, 160, 163) and `ps_deposit_params.h` (line 241) still say "float32 r3d port".
- `tests/ps_linear_deposit_check.sh` (lines 261–262) names kernel functions that no longer exist (`exact_init_tet`,
  `exact_clip_plane`, `exact_reduce2`).
- Check that the Makefile's GPU comment ("not yet validated on hardware") is still accurate.
- Effort: S.

**P1.8 Unused include:** `#include <deque>` in `ps_metal_host.cc` — *done 2026-11-06.* Effort: S.

**P1.9 Dead Python names** (each appears only at its definition) — *done 2026-11-06.* `fields.py`'s `Path` import went with
`ensure_output_dir`; ruff F passes.
- `config.py`: `cosmic_time_gyr`, `CORRELATION_N_BINS`, `VIS_SMOOTHING_SIGMA`;
- `fields.py`: `ensure_output_dir`;
- `plot/plot_PDF_CDF.py`: `get_critical_point_densities`.

Some helpers are used only by tests: `catalog.read_hdf5`, `io.velocity_scale`, `theory.bbks_ep_pdf`,
`trees.descendant_branch`. Keep them as public API. Keep also `view3d.texture_map`, `view3d.section_points` and
`volume3d.composite`: they are the 3D tests' reference oracles.
- Verify: ruff F, the py suites. Effort: S.

**P1.10 Unused environment knobs in the tests** — *done 2026-11-06.* Seven test suites (`gpu_retry_check`, `ps_fixed_moments_check`,
`ps_mass_exact_check`, `gpu_mass_check`, `dtfe_gpu_retry_check`, `dtfe_metal_check`, `dtfe_auto_tune_check`) still
set the `*_METAL_*` spellings of GPU knobs.
- Move them to the canonical `*_GPU_*` names. Done: where a run set both spellings, only the GPU one is left. The
  `unset` lines and the two Python readers of the environment keep both, since a user's environment may still hold
  the alias.
- Dropping the aliases themselves is D3.
- Effort: S.

**P1.11 Delete `PS_GPU_CPU_SHARE`** (D1, decided 2026-11-06: delete) — *done 2026-11-06.* The block (about 45 lines,
not the 120 estimated) and its two
`metal/README.md` rows are gone. Verified: G-build 0 warnings in all six modes; G-byte 184/0 against `ref0`; the
GPU-exact outputs of `ref0`'s `PS-DTFE` and the new one byte-identical (16 files, one domain and 2×2×2, the
P1.12(a) check done by hand); `validate_deposit` 31/31 (after P1.17); the GPU walk's suites pass.
- Where: `ps_interpolation.cc`, inside the GPU walk.
- Context: opt-in, the best measured gain was 4.5% on the 94M-particle run, and no test or doc covers it.
- Verify: G-byte, G-gpu (the default path, share 0, is byte-identical by construction). Effort: S.

**P1.12 Two verification tools the later phases need** — *done 2026-11-06.*
- (b) Done as `tests/dev/gui_shots.py` (synthetic data root; `--native` for a real window).
- (a) Done as `bytecmp.sh --gpu-exact` (6 runs: DTFE and PS-DTFE, single and double, one domain and 2×2×2). With it,
  the auto-tune report script moved to `tests/dev/tune_report_set.sh`. The plan was: extend `tests/dev/bytecmp.sh` with
  an opt-in GPU-exact section: `--ps-exact-deposit --ps-gpu` and
  `--exact-average --gpu`. It compares two binaries built from the same kernel source. Today `bytecmp` leaves GPU
  exact out, because fast math makes it differ from the CPU. A host-only refactor must still leave it unchanged.
- (b) Add a `tests/dev/gui_shots.py` that renders every tab offscreen at 1500 and 1100 px, using `remember=False` so
  that saved settings are never written. Also render the Explore tab with "Show in 3D" on (volume, then planes), on
  the synthetic output that `py_gui_explore_test`'s `_vol_set` builds.
- Effort: S each.

**P1.13 Turn the three `TODO(human)` markers into design notes** — *done 2026-11-06.*
- Where: `volume3d.py:opacity_curve` [626], `view3d.py:close_up_window` [584] and, since `70e153e`,
  `explore_3d.py:next_turn_step` [241] (the GPU's turning-frame policy). All three mark working, tested defaults as
  "optional".
- Change: rewrite each as a plain design note and keep the measured numbers.
- Verify: `grep -rn 'TODO' python/gui` is empty. Effort: S.

**P1.14 Make `ps_scratch_check` prove the zero-fill route is taken** — *done 2026-11-06.* Section A reads the count off the
partitioned run's SCRATCH line. Built with the scope forced off, that check fails (0 grids) and every other check
still passes.
- Where: section A of `tests/ps_scratch_check.sh`. It should parse the "N zero-filled in memory" count from the
  partitioned run's log (`DTFE.cpp` prints it after `psReserveGrids`) and require N > 0. With
  `DTFE_SCRATCH_MIN_GB=0.0001` every grid there is a scratch mapping.
- The unpartitioned run prints no count, because its grids come from `reserveMemory`. Either print the same line there
  or leave that run out.
- Add a comment next to the message saying that a test parses it.
- Why: today a silently disabled scope would pass every check, and the zero write-back stall would come back.
- Risk: low. Effort: S.
- Verify: the check passes; with the scope forced off it fails.

**P1.15 Check `ZeroFillScope` on Linux** — *done 2026-11-06.* Measured in Ubuntu 24.04 (Colima), a 640³ run with
`--scratch-dir`: in the scope the peak anonymous RSS was 2.92 GB (the 3.1 GB velocity grid); without it 0.01 GB
(5.9 GB of page cache instead). With the container capped at 2 GB the scope's run was OOM-killed, the other ran. So
the scope is now macOS-only (`#if defined(__APPLE__)` in its constructor), and `ps_scratch_check` expects 0 moves
off macOS.
- How it works: inside a scope, `scratchTryMap` hands out anonymous memory, `std::vector` writes its zeros into it,
  and `settlePending` maps the scratch file over it when the scope closes. Each scope covers one grid.
- The risk: under the macOS memory compressor those zero pages cost almost nothing. On Linux every written zero page
  is resident until the remap, so the largest single grid may be held in RAM for a moment. That is the case
  `--scratch-dir` exists for. *(UNVERIFIED: the fix was measured on macOS only, and CI's tiny grids pass either way.)*
- Test: measure the peak RSS around `psReserveGrids` in the G-linux container, with one grid larger than the
  container's memory limit and a small `DTFE_SCRATCH_MIN_GB`.
- If RSS grows by the grid's size, make the scope a no-op off macOS (one `#if defined(__APPLE__)` in its
  constructor) and say so in the header comment.
- Risk: low; Linux would return to the previous path. Effort: S.
- Verify: G-linux, `ps_scratch_check` on Linux. Depends: P0.3.

**P1.16 Stray links after a build-mode switch.** *(found 2026-11-06, building the six modes at `70e153e`)* — *closed
2026-11-07: not reproducible; make's built-in rules cancelled.* In the four logs that show it, the stray links come
before the mode check's own message, so make ran them while it re-read the `.d` files, where a failed recipe is
ignored (make still exits 0). It did not come back in 14 later switches: plain, METAL, `library` → CUDAEMU and back,
with and without `-d`, in a path with spaces, and with two builds overlapping in one tree. Every object and binary has
an explicit rule, so the Makefile now cancels make's built-in rules (`.SUFFIXES:` and an empty `%: %.o`): the
`clang++ X.o -o X` link cannot run whatever triggers it. G-build passes in all six modes with them cancelled. The two
files a no-change rebuild recompiles are `build_info.o` (both binaries), by design: the link step stamps the build.
- What: the first `make DTFE PS-DTFE CUDAEMU=1 -j10` after a build in another mode also runs make's built-in
  `%: %.o` rule on some objects (`clang++ o/r3d.o -o o/r3d`, `o_ps/ps_point_eval.o -o o_ps/ps_point_eval`, ...). Each
  fails with `ld: library 'c++' not found` and is ignored. The count varies between runs (4 and 44 seen), so it is a
  race in the `-j` build. Make still exits 0 and the real binaries are right. A second CUDAEMU build is clean, and
  so are the other five modes.
- Where to look: what makes make want a target named like the object without its suffix just after the mode
  check (`Makefile` [350–362]) wipes the objects. Suspects are the `-include $(wildcard $(OBJ_DIR)/*.d)` lines
  [642–643], read while the wipe runs, or a prerequisite written without `$(OBJ_EXT)`. Compare the "Makefile
  metal-stamp race" fixed earlier.
- Also: a CUDAEMU rebuild with nothing changed still recompiles 2 files.
- Not every switch shows it: the P1 batch's G-build (2026-11-06, `make library` then `CUDAEMU=1`) had 0 stray lines.
- Why: G-build's "0 warnings, 0 errors" cannot be read off a first build after a switch: the stray lines add about
  one warning and one error each. On macOS 27, Homebrew LLVM 23 adds a missing-`MacOSX27.sdk` warning to each, since
  those lines lack the Makefile's `$(MACOS_ISYSROOT)`.
- Risk: low (build only). Effort: S–M. Verify: G-build from a mode switch, three times, 0 stray lines; G-byte.

**P1.17 The Metal harness did not build** — *found and fixed with P1.11, 2026-11-06.*
- `60a072a` moved the exact clip helpers (`tet_planes`, ...) into `metal/exact_clip.metal.inc`, which the Makefile
  prepends to the binaries' kernel source and the harness prepends when it compiles at run time.
- `metal/build_prototype.sh` still compiled `ps_deposit.metal` alone for the offline `.metallib`, so it stopped
  ("use of undeclared identifier 'tet_planes'") before building `validate_deposit`: G-gpu's 31/31 could not run.
- Fix: the offline compile takes `-include metal/exact_clip.metal.inc`. Verified: the harness builds and passes
  31/31.

**P1.18 CI installs only the Boost it uses** — *done 2026-11-06.*
- On `38eb4be`'s run, `cudaemu (32)` hit the new 10-minute install timeout: the runner's mirror took minutes per
  package. The three jobs installed `libboost-all-dev`, which also pulls Boost.MPI, OpenMPI and the ROCm/HIP
  libraries.
- The build links only `boost_thread`, `boost_filesystem` and `boost_program_options` (the Makefile and
  `demo/examples`); every header is in `libboost-dev`.
- Change: those four packages instead. On a clean Ubuntu 24.04: 226 packages instead of 363; the binaries,
  `make library` and `demo/examples` build with 0 warnings and run.

---

## P2. Measure first (no output change)

**Status 2026-11-07.** P2.1, P2.2, P2.4 and P2.5 done; P2.3 partly (three of its reruns).
- P2.1: `PSGpuPhaseTimes` / `psGpuReportPhases` in `gpu_host.h`. Since P9.6 the per-partition line under
  `PS_GPU_TIMING=1` reads *before the hold* (device, plan, item bounds, item fill) and *hold by phase* (uploads,
  kernels, pacing gaps, snapshots, leftovers, copy-back, releases); the hold's phases sum to it. (P9.5 had moved the
  copy-back and releases after the hold; it was reverted.)
- P2.2: `PS_GPU_DEBUG_NOATOMIC=1`. 2.1M-particle crossed waves at 256³, one partition, nSub 3, kernel busy time:
  density only 4.56 → 4.49 s (about 2%, noise); density + velocity + gradient 17.0 → 3.4 s, so the atomics take up
  to 80% (an upper bound: the knob skips all of `depositSample`). Input for P9.8.
- P2.4: no code change; `triangulation.cpp` already times the insertion. The dummy points take 1.2–3.2% of a
  standard run (2.1M near-lattice boxes, `--periodic --grid 256 --exact-average`, loaded machine): 3.1% for the
  exact distorted lattice on 10 sub-domains, 3.2% at 2% jitter, 1.2% on one domain. Below D5's 5%: D5 = keep, and
  P9.4 is not needed.
- P2.5: both partition loops print a `merge time:` line (adds into the full grids, checkpoints, spilled reads,
  parked time and lock waits; `--verbose` >= 2). 2.1M particles at 256³, 3×3×3, loaded machine, median of 3: adds
  0.24 s and checkpoints 0.12 s of a 38 s partition loop (about 1%); 313 s parked in all, no thread waited. Under
  `--fixed-point-merge` (still cell by cell): 1.33 s, 1.48 s and 0.80 s of lock wait in 42 s. The real-data share
  comes from the P2.3 logs, which now carry the line.
- P2.3, three reruns with the final tree (M1 Max, production flags, auto-planned, into a separate folder; the logged
  runs' folders were only read). Error as in P8 = (predicted − measured) / measured, of the peak footprint:

  | Run | Plan (budget) | Predicted → measured | Error | Wall | The logged run (build 49ccb9e) |
  |---|---|---|---|---|---|
  | Standard, TNG100-3-Dark z = 0, 512³, GPU exact average | 6³, 8 threads, look-ahead off (38 GB) | 37.8 → 26.5 GB | +43% | 1022 s | hand-set 3³ mc 10: 32.1 GB, 562 s |
  | PS, TNG100-3-Dark z = 1, GPU sampled + 4096² points | 6³ mc 10 (40.1 GB) | 34.8 → 24.6 GB | +41% | 507 s | 5³ mc 10 (50.3 GB): 45.1 → 82.9 GB (−46%), 647 s |
  | PS, TNG100-3-Dark z = 0, the same | 6³ mc 8 (41.2 GB) | 39.3 → 31.1 GB | +27% | 1379 s | 6³ mc 10 (52.0 GB): 43.2 → 123.2 GB (−65%), 1681 s |

  - The GPU sampled class's −45…−69% was the parking: the logged runs held up to 109 and 189 finished partitions in
    RAM (42 and 74 GB), the reruns at most 8 and 4 (1.9 GB), none spilled. Both classes now over-predict, by 27–43%:
    outside P8.1's band, on the safe side.
  - The standard planner's +43% made it pick 6³ (216 partitions, look-ahead off) where the hand-set 3³ ran in
    32 GB: triangulation 458 s summed over the partitions instead of 194 s, the run 1.8× the logged one (another
    build, so only roughly). Input for P8.
  - The budgets were 10–12 GB below the logged runs' (less free memory at the start): hence 6³ instead of 5³ at
    z = 1 and mc 8 at z = 0.
  - `merge time:` standard: 216 copies 1.5 s, checkpoints 47 s (5% of the run). PS z = 1: adds 20.5 s, checkpoints
    22.3 s, 348 s parked, 83 s holding a thread (the merge lock while another partition merges or drains; no
    partition waited for its turn); z = 0: 15.1 s, 67.1 s, 345 s, 25 s. The merge is 6–8% of a PS run, mostly
    checkpoints.
  - Per phase (PS: only the master thread's 22–27 partitions print times): PS triangulation 2.6–4.1 s, vertex
    densities ~0.3 s, deposit 5–12 s (z = 1) and 16–68 s (z = 0) a partition, T-web 14 s; standard, over all 216:
    triangulation 458 s, vertex densities 10 s, the GPU average 58 s, T-web 14 s. No GPU fallback in any of them.
  - Outputs: the z = 0 PS rerun (the logged run's 6³ split) is byte-identical to the logged run in all seven grids
    compared (density, velocity, dispersion, streams, caustics, the points' density and streams). The other two
    changed split (6³ vs 5³ and 3³): mean differences ~1e-5 of the mean, the same mass; the PS dispersion moves by
    more than 1% in 0.02% of the cells (most likely the tetrahedra at the partition seams; not checked), up to 30× in
    a few. Only the same-split run checks tonight's code against the logged one (the others also differ in build).
- Open: P2.3's TNG50-3-Dark z = 0 rerun (GPU sampled + points, ~2.4 h logged), the 1024³ CPU exact run (53 min
  logged) and the CPU `--ps-window` runs. **Moved to phase R** (after all code items, owner 2026-11-07).
  - A first TNG50 attempt (2026-11-07, auto budget 38.7 GB = 42.7 GB free − 4 GB) planned 6³ **mc 1**: the 25.7 GB
    of point records left room for one partition (~2.7 GB, *est.* from the two plans). It was stopped at 69 of 216
    partitions (3 h 7 min; 26.5 GB peak footprint against 38.4 GB predicted) to run overnight. Input for P8.5 and
    P8.10.
  - The rerun uses the logged run's own `DTFE_MEM_BUDGET_GB=52`, so its plan matches the logged one (6³ mc 6). The
    binary caps the knob at the free memory − 4 GB, so it needs ≥ 56 GB free at the start.

**Status 2026-11-09 (phase R).** **Operational rule (found 2026-11-09 14:10, P11):** the tuner plans from the free
memory it sees at start, so a resident Colima VM (the Linux check's, 16 GB) turned the standard TNG100 z = 0 plan
from 3³ into 4³ and the run from 435 s into 616 s; stop Colima (`colima stop`) before any production or timing run.
P2.3 done except the CPU `--ps-window` runs (not run). The night of 2026-11-08/09
reran six cases with the final binaries (`4e2b35b`, Metal-only) into the data disk's `p23-reruns/` (the production
folders only read); with the three of 2026-11-07 (`745daac`), all exited 0. TNG100-3-Dark z = 0 unless named. Error
as in P8 = (predicted − measured) / measured, of the peak footprint (the run logs print (measured − predicted) /
predicted: their −9% is +10.5% here); every standard run had the look-ahead off:

| Run | Deposit | Grid | Plan (budget) | Predicted → measured | Error | Wall | The logged run | Build |
|---|---|---|---|---|---|---|---|---|
| `std100_099_final` | standard, GPU exact average | 512³ | auto 3³ × 10 threads (41.2 GB) | 34.7 → 31.4 GB | +10.5%, in band | 421 s | hand-set 3³ mc 10, 562 s (49ccb9e) | 4e2b35b |
| `std100_099_face` | the same + `--face-gradient`, `DTFE_FACE_CHECK=1` | 512³ | auto 3³ × 10 (41.2) | 34.9 → 28.2 | +23.7%, over | 485 s | — | 4e2b35b |
| `ps100_099_final` | PS, GPU sampled + 4096² points | 512³ | auto 6³ mc 3 (41.2) | 38.3 → 27.2 | +40.6%, over | 1706 s | 6³ mc 10, 1681 s (49ccb9e) | 4e2b35b |
| `ps50_099_mc2` | the same, TNG50-3-Dark z = 0 | 512³ | hand 6³ mc 2 (50, `DTFE_MEM_BUDGET_GB`) | 51.3 → 34.9 | +46.8%, over | 11845 s | 6³ mc 6, 8606 s (49ccb9e) | 4e2b35b |
| `ps100_099_exact1024` | PS, CPU exact, no points, scratch dir | 1024³ | auto 6³ mc 2 (41.2) | 32.4 → 38.9 | **−16.9%, under** | 3972 s | 6³ mc 10, 3166 s (066c7ac) | 4e2b35b |
| `ps100_099_face512` | PS, CPU exact + `--face-gradient`, no points | 512³ | auto 6³ mc 5 (41.2) | 36.5 → 33.7 | +8.1%, in band | 1673 s | — (`ps_exact`: points, no face, 6³ mc 4, 1377 s, 066c7ac) | 4e2b35b |
| `std100_099` (11-07) | standard, GPU exact average | 512³ | auto 6³ × 8 threads (38 GB) | 37.8 → 26.5 | +42.7%, over | 1022 s | 562 s | 745daac |
| `ps100_099` (11-07) | PS, GPU sampled + 4096² points | 512³ | auto 6³ mc 8 (41.2) | 39.3 → 31.1 | +26.6%, over | 1379 s | 1681 s | 745daac |
| `ps100_050` (11-07) | the same, z = 1 | 512³ | auto 6³ mc 10 (40.1) | 34.8 → 24.6 | +41.4%, over | 507 s | 5³ mc 10, 647 s | 745daac |

- Outputs, read-only, in units of side B's field rms:

  | Pair | What differs |
  |---|---|
  | PS GPU sampled + points, `ps100_099_final` (mc 3) vs the 11-07 rerun (mc 8) and vs production (mc 10, 49ccb9e); all 6³ | nothing: all 22 files identical, both times |
  | PS CPU exact, `ps100_099_face512` (6³ mc 5) vs production `ps_exact` (6³ mc 4, 066c7ac) | nothing in the 11 non-derivative grids; the velocity derivatives by design (grad⟨v⟩ under `--face-gradient`) |
  | PS CPU exact 1024³, 6³ mc 2 vs production 6³ mc 10 (066c7ac) | nothing: all 9 grids identical |
  | PS GPU sampled TNG50, hand 6³ mc 2 (one pass ran on the CPU) vs production 6³ mc 6 | 0.02–0.5% of the words of density, velocity, dispersion, T-web eigenvalues and derivatives, the V-web in 2 cells (density rms 1.5e-9; derivatives rms ~6e-4, up to 13.6 in single cells); streams, caustics, T-web, all `pts_*` identical |
  | standard GPU exact average, `std100_099_final` (3³) vs the 11-07 6³ | density in 98% of the cells: rms 2.7e-5, max 0.095, 4345 cells > 1e-3, 89 > 1e-2, mass equal to 2e-8; gradient rms 2.8e-5; 1171 T-web cells |
  | `std100_099_split6` (the same binary at 6³) vs the 11-07 6³ | density in 96.7%: rms 1.7e-5, max 0.041, 2483 > 1e-3 |
  | `std100_099_split6` vs `std100_099_final` (3³) | density in 97.9%: rms 2.5e-5, max 0.066, 4187 > 1e-3, mass equal to 5e-8 |
  | `std100_099_split6_face` vs `std100_099_split6` (6³ with and without the face pass) | nothing in density, velocity and T-web; the velocity derivatives by design (the face pass's) |
  | face pass 6³ (`std100_099_split6_face`) vs face pass 3³ (`std100_099_face`), `a_velGrad` | 2.8% of the cells (3.72 M) above 1e-4 of the rms, rms 2.7e-5, max 7.6e-3, smooth in x, y and z: the deposit's split dependence is the face pass's too (the P10 status) |
  | **after P10.8** (2026-11-09 afternoon): `std100_099_face_p108` (3³) vs `std100_099_split6_face_p108` (6³) | density in 1886 cells (1.4e-5), velocity in 12,308 values (3.1e-5), gradient in 493, divergence in 5, shear in 69 — each by one float ulp (rms 6e-12 of the density rms); T-web and V-web byte-identical (the P10 status) |
  | `std100_099_face_p108` vs `std100_099_face` (the 3³ plan before and after P10.8) | density in 98%: rms 2.0e-5, max 0.047; velocity rms 1.3e-6; gradient rms 2.2e-5, max 5.7e-3: the old split scatter, removed |
  | `std100_099_final` vs production 3³ (49ccb9e) | density in 70.7%: rms 4.7e-6, max 0.014, 196 > 1e-3; gradient rms 3e-7, max 7e-4 |

  Reading: PS-DTFE's grids reproduce byte for byte at a given split, across concurrency and builds (the TNG50 run,
  whose one deposit pass ran on the CPU, differs in 0.02–0.5% of its words; across splits the 2026-11-07 z = 1
  rerun moved ~1e-5 of the mean, above). Standard DTFE's GPU exact average does not: one binary and
  split reproduce (the face run's density and velocity equal the final run's), but across splits and builds the grids
  agree only to ~1e-5 of the field rms (the velocity to ~1e-6), mass exact, the split moving them more than the
  build. Both trees passed bytecmp against `ref0` on small data. The cause shown for the worst case (the per-cell
  map) was the deposit's float32 frame (P10.8); the rest was in the face pass too, so not the frame: the
  per-tetrahedron gradient's float labelling dependence (confirmed by P10.8's step 0 and removed with its second
  part, the hashed order and the double gradients; the P10 status of 2026-11-09 afternoon has what is left).
- `merge time:` (P2.5's measure, shares of the partition loop): standard copies 1.3–1.5 s (≤ 0.36%), checkpoints
  17–47 s (4.6–5.4%); PS GPU sampled adds 11–21 s (0.15–4.8%), checkpoints 22–186 s (1.6–5.2%); PS CPU exact adds 84 s
  at 512³ with the face sums (5.3%) and 475 s at 1024³ (12.9%, checkpoints 321 s): that run held the merge lock 796 s,
  21.7% of its 3676 s loop, but lost only 281 s holding a thread (3.8% of loop × 2; elsewhere 8–93 s). Parked waits
  release the thread. Only at 1024³ is the merge a serial cost. For D6: the auto plans held at most 1–4 waiting
  partitions at once (1.8–2.7 GB; the 11-07 runs 4 and 8); the TNG50 hand plan, with no park room, spilled 58.
- Per-phase sums (PS `run total` lines: seconds over all passes and threads, not wall): GPU sampled TNG100 setup
  176 s, extraction walk 384 s, GPU deposit 1648 s, deferred tets on the CPU 926 s; TNG50 268 / 856 / 9326 / 10350 s;
  CPU exact, CPU deposit 4120 s (1024³) and 5439 s (face512). The other phases P2.3 names (triangulation, vertex
  densities, export) and the standard runs': in the run logs, not tabulated.
- Plans: the auto standard plan now picks the hand-set 3³ × 10 threads (0.75× the logged wall, 2.4× faster than the
  11-07 6³). PS GPU at mc 3 took 1.24× the 11-07 mc 8 for the same bytes (P8's over-booking). TNG50's 1.38× the logged
  wall is attributed to the hand plan's mc 2 (the logged run: mc 6, another build; no park room, so 58 partitions
  spilled, 38.35 GB), not to its GPU failures (below).
- GPU: no event in the standard runs; one hang, recovered, in `ps100_099_final`; TNG50: 27 retries (15 timeouts), one
  pass on the CPU after 5 failed buffers (the grids complete), about 4% of the wall, with the display on all night.
- Memory: lowest free 14.5 GB at `std100_099_final`'s last sample (18 s before its end), 27.7 GB in the TNG50 run;
  swap (sampled from the TNG50 run on) 0 until the 1024³ run, 0.28 GB 88 s into it (the window of its start-up zero
  fill, the P8 status), 0.73 GB by its end; every footprint ≤ 38.95 GB.

The tuner and the optimisation phases need numbers that the existing run logs do not have. Everything here is
instrumentation or reruns.

**P2.1 GPU per-phase timers under `PS_GPU_TIMING=1`.** — *done 2026-11-07.*
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

**P2.2 A PS counterpart of `DTFE_GPU_DEBUG_NOATOMIC`.** — *done 2026-11-07.* It measures the atomics' share of the sampled kernel's
time before P9.8 is built. Effort: S.

**P2.3 Rerun the runs the tuner table flags (see P8), with the current binary.** — *three of them done 2026-11-07, the rest on 2026-11-08/09 in phase R except the CPU `--ps-window` runs (see the status above).*
- GPU sampled with sample points, TNG100-3-Dark z = 1 and z = 0, and TNG50-3-Dark z = 0. The logs on disk come from a
  build that predates the fix to the partition hand-out order (`schedule(monotonic: dynamic, 1)`), so their
  −45…−69% errors are mostly parking that the fix removed. The TNG50 z = 0 run took 2.4 h at 13% CPU.
- PS exact at 1024³, with `ZeroFillScope` (the logged run predates it).
- **One auto-planned standard-DTFE run.** All 27 standard logs on disk used hand-set plans, so the standard planner's
  predictions have never been checked against a real run.
- A CPU `--ps-window` run at 2048² and at 4096². These are under-predicted by about 25–30% even after P0.2.
- Also record per-phase times: triangulation, vertex densities, deposit, merge, export.
- Effort: M (several hours of machine time; needs the simulation data).

**P2.4 Time the standard binary's dummy-point insertion** — *done 2026-11-07.* (the padding test).
- Where: `triangulation.cpp` (lines 133–139), on a lattice-like snapshot. It inserts about 6n² points that lie on
  lattice planes, which is likely to force CGAL's exact-predicate fallback.
- Feeds D5. Effort: S.

**P2.5 Merge share.** — *done 2026-11-07.*
- From the P2.3 logs, take the time spent in the serial ordered merge. The path is `DTFE.cpp:psMergeInTurn` [1444],
  lambda `mergeOne` [1485–1500] → `quantities.cc:Quantities::addFromSubgrid` [343] → `addFieldSubgrid` [300] /
  `orFieldSubgrid` [323]. Under `--fixed-point-merge` it goes through `fxMerge.add` instead.
- `PartitionCheckpoint::afterMerge` runs under the same lock; time it separately.
- This decides P9.1. Effort: S.

---

## P3. C++ structure (byte-identical)

**Status 2026-11-07.** All done except P3.9's dedupe (D10). **2026-11-08:** P3.9's dedupe and its three bugs done; P3 is
closed.
- P3.5: `interpolateGrid_phaseSpace` 2533 → 1686 lines (`PSFxState`, `psMarkFlipped`, `psCentroidCell`,
  `psWindowTouched`, `psFxBoundWalk`, `psGpuExtractAndDeposit`). `depositLoop` stays inline (D7) and compiles to the
  same 7370 instructions (-O3, METAL object); only the three calls into the lifted functions differ.
- P3.7: 643 → 433 lines; not instruction-identical (the body is split), byte-identical outputs, MC average x1.004 and
  exact average x0.999 of the base's time. The helpers' `* const &` bindings matter: copies cost about 4%.
- P3.8: the whole model is in `src/auto_tune.cc` (anonymous namespace); the header says a pointer to
  `auto_tune.h <name>` now means the pair. Its follow-ups are done too: the header includes only what the
  interface uses (every includer's object byte-identical), and the PS branch's unused `countCell` is gone.
- P3.9, first step: `tests/dtfe_sampling_check.sh` (methods 2 and 3, the redshift cone, the grid's point values,
  user points through libDTFE, stored goldens for the cone and the user points). Upstream bugs it found (fixed with
  the dedupe on 2026-11-08, see P3.9): (a) `averaged_interpolation_2.cc` builds method 2's sample as `Point(uni()*dx+x, uni()*dy+y, uni()*dz+z)`,
  unsequenced, so GCC and Clang draw different realizations from one `--seed`; (b) `interpolateRedshiftCone`'s
  infinite-cell `continue` skips `psi += dx[2]`, shifting the rest of the row; (c)
  `interpolateUserSampling_averaged_2` skips a sample outside the region without pushing a value, so the output is
  shorter than the sample list.
- P3.9, second step (2026-11-08): the seven functions share `sample_eval.h` (`unaveraged_interpolation.cc` 497 → 164
  lines, `averaged_interpolation_2.cc` 922 → 310); byte-identical on every production path (451 of 483 A/B files over
  periodic and non-periodic grids, methods 1/2/3 and the exact average, `--region`, `--padding`, `--partition`, 2D,
  single and double, 1 and 4 threads; the 32 others are the fixed cells, all exactly 0 before). The reader found (b) in
  `interpolateGrid` too and (c) in `interpolateUserSampling` too, plus a `Real` padding index (d): all fixed, with
  S7 grid goldens, S8 and S9 in `tests/dtfe_sampling_check.sh`, which fail on the old binary. Timing: grid point values
  0.99, methods 2 and 3 0.98 of the old time (single thread, a loaded machine; `always_inline` on the helpers was needed:
  the first build was 5–10% slower with them out of line).

Gate for each item: G-byte, G-build, G-battery for the phase, and G-move wherever a hot function is touched.

**P3.1 One thread-count helper.** — *done 2026-11-07.*
- Three copies of the same function: `ps_interpolation.cc:psDepositThreads` (line 282),
  `averaged_interpolation_1.cc:exactAverageThreads` (line 332) and `ps_point_eval.cc:pointEvalThreads` (line 461).
  - They differ only in the environment variable that overrides them: `DTFE_DEPOSIT_THREADS` twice,
    `DTFE_PTS_THREADS` once.
  - `DTFE.cpp:dtfeParallelThreads` (line 751) is a different rule (the OpenMP maximum capped by `--max-concurrent`,
    with no override) and stays out of the helper.
- Change: one `coresPerTeam(char const *envOverride)` in a small header next to `env_knobs.h`.
- Why: P9.2 changes this rule, and it should change in one place.
- Risk: low. Effort: S.

**P3.2 One 3×3 adjugate.** — *done 2026-11-07.*
- Three copies: `math_functions.h:matrixInverse`, `ps_exact_inside.h:inverse3x3d`, `ps_point_eval.cc:inverse3x3Unguarded`.
  They differ only in the singularity guard.
- Change: one inline `adjugate3x3(m, inv) -> det` core with the same operation order, and thin guards around it.
- Risk: medium. The operation order must stay exactly the same, because `dtfe_gpu_fixed.h` mirrors this arithmetic.
- Verify: G-byte, plus `-O0` IR floating-point operation counts. Effort: S.

**P3.3 One field list.** — *done 2026-11-07.*
- The list of `Quantities` grids is written out six times in `quantities.cc`, and again in:
  - `partition_checkpoint.h:forEachAccumulator`;
  - `ps_fixed_merge.h:forEachSummed`, whose comment says "if you add one there, add it here";
  - `input_output.cc`, twice;
  - `web_fields.cc`, twice;
  - `DTFE.cpp:psReserveGrids` [1209–1250]. It sizes the PS-only grids that `reserveMemory` leaves out
    (`stream_count`, `hidden_streams`, `caustic_bits`, `tet_touch`), with 8 `zeroAssign` lines.
- Change: one `Quantities::forEachField(f)` with a tag for summed, OR-merged and derived fields. `reserveMemory` and
  `psReserveGrids` then become one sizing pass over it, so the zero-fill form (`zeroResize` / `zeroAssign`) is written
  once.
- Also fix the stale comment in `psReserveGrids` [1216–1217]. It says the pre-sizing stops `addFromSubgrid` writing
  past an unsized vector, but the merge now grows an unsized grid itself (`quantities.cc` [307, 329]). The real reason
  for pre-sizing is that the grids are then allocated before the partition loop, where the memory gate sees them.
- Why: today a new field must be added in about 13 places, and forgetting one is silent.
- Risk: low to medium. Effort: M.
- Verify:
  - G-byte;
  - the resume and fixed-merge suites (`ps_resume_check`, `dtfe_resume_check`, `ps_fixed_merge_check`);
  - `ps_scratch_check`.

**P3.4 One look-ahead pipeline, one progress line.** — *done 2026-11-07.*
- Two copies of the look-ahead pipeline: in `DTFE.cpp:psRunPartitionLoop` (lines 1693–1750) and in
  `dtfePartitionedRun` (lines 2505–2564). Both have:
  - the same `std::thread` with `exception_ptr`;
  - the same room test, `autoTuneSystemFree() - 4e9 >= need`;
  - the same "look-ahead paused" and "prepared in" messages;
  - the same join and rethrow.
- Change: a `runLookAhead(order, needBytes, prepare, deposit, canPrefetch, label)` template. Also one
  `partitionProgress()` for the `[partitions k/N] elapsed ETA` line (lines 1625 and 2420).
- Then: make the merge-staging state in `psPartitionedRun` (about lines 2087–2160) a `struct PSMergeStaging`, with
  `parkWhere` and `subgridBytes` as methods. `PSLoopCtx` (40 reference members) then shrinks to a handful.
- Risk: low; ordering is unchanged. Effort: M.
- Verify: `ps_runtime_check` (its parking and look-ahead checks), `dtfe_resume_check`, G-byte.

**P3.5 The `interpolateGrid_phaseSpace` remainder (2574 lines) without touching the deposit loop.** — *done 2026-11-07.*
- Lift the `fx*` lambdas (about lines 2297–2460) into a `PSFxState` struct. They run per pass, not per cell; only
  `fxAddL` is used inside the loop.
- Lift `markFlipped` and `fxBoundWalk`.
- Lift the GPU block as `psGpuExtractAndDeposit(ctx)`, keeping the walk lambda inside it (hot-loop rule).
- `depositLoop` itself (about 1010 lines) stays; see D7.
- Risk: medium. Effort: M.
- Verify: G-move on `depositLoop` and the slab lambdas, G-byte, a timing check within ±1%.

**P3.6 Split `ps_point_eval.cc` (3535 lines) into its three modules** — *done 2026-11-07.* by pure moves:
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

**P3.7 `interpolateGrid_averaged_1` (643 lines).** — *done 2026-11-07.*
- Split its `cpuLoop` lambda (about lines 958–1274) into `avg1SingleCell`, `avg1ExactTet` and `avg1McTet`.
- Remove the per-tetrahedron heap allocation `std::vector<Point> randomPoints(n)` (line 1210): reuse a thread-local
  buffer. That part is a small speed-up as well.
- Risk: medium. It is the CPU hot loop of standard DTFE.
- Verify: G-move on the lambda, G-byte, `dtfe_mc_average_check`. Effort: M.

**P3.8 `auto_tune.h` (2446 lines, a header) → `auto_tune.h` + `auto_tune.cc`.** — *done 2026-11-07.*
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

**P3.9 (optional, upstream) The sampling variants.** — *done: the golden tests (D10) 2026-11-07; the dedupe and the three fixes 2026-11-08 (see the P3 status).*
- Seven functions repeat the per-point "locate, cached inverse, accumulate the fields" loop, about 40–55% identical
  lines:
  - `interpolateGrid_averaged_2`, `interpolateRedshiftCone_averaged_2`, `interpolateUserSampling_averaged_2`,
    `interpolateGrid_averaged_3`;
  - `interpolateGrid`, `interpolateRedshiftCone`, `interpolateUserSampling`.
- Methods 2 and 3, the redshift cone and user sampling have **no tests**. The first step is therefore golden outputs
  with a fixed `--seed`, which is worth doing on its own. The dedupe (an `evalAtPoint` helper plus a sample-source
  functor) is D10.
- Risk: medium. Effort: L.

**P3.10 The fixed-point merge's integer grids through `zeroAssign`.** — *done 2026-11-07.*
- The problem: `PSFixedMerge::Merge::setup` frees each pre-sized grid and fills its int64 and int8 twins with a plain
  `assign` (`ps_fixed_merge.h` [238–239]), and the first merge of an unsized slot does the same [340–341]. Under
  `--scratch-dir` these are scratch mappings of 9 B per component, so `--fixed-point-merge` brings back the zero
  write-back stall: only the float accumulators got the fix.
- Change: wrap those four lines in `ScratchAlloc::zeroAssign`. The restore paths [267, 271] are followed by real data
  and gain nothing.
- Report the count after setup or at the end of the run: the existing line prints before setup.
- Risk: low; the bytes are the same. Effort: S.
- Verify: `ps_fixed_merge_check` (its scratch section), `ps_scratch_check`, G-byte. Depends: P1.14.

---

## P4. GPU structure (byte-identical)

**Status 2026-11-07.** P4.1, P4.2 and P4.4 done; P4.3 partly; P4.5 not started.
- P4.3: step 1 done (`GpuPace::onFault` / `onClean` at all four loops); of step 3, `GpuFault`, `gpuFaultName` and
  `GpuFaultHook` are shared in `gpu_retry_common.h`, the `GpuSnapshot` twins not. Step 2 (`GpuChunkLoop<Backend>`) is
  open: the four loops differ in error handling, the shared-box path, `drainQueue`, chunk alignment and message text,
  and byte-identical messages across backends could only be gated through the emulator.
- P4.4: `tests/gpu_twin_check.sh` (CUDA clip vs double r3d anywhere, vs the Metal clip on a Mac; about 3.1M pairs in
  5 s; CI's cudaemu job). **Finding:** on Metal, with order 1 and order 2 compiled as two function-constant
  pipelines (as `depositExactItems`' phases 0 and 1 are), the volumes agree to the bit but the first moments differ
  in about 17% of the pairs (at most 3.7e-8 of the cell); the emulated CUDA twin agrees bitwise. It matters only
  under `--exact-average --ps-linear-deposit`, where the fLinear weight uses the first moments: a tet then deposits
  m·(Σw₁/Σw₀) instead of exactly m (about 1e-8 relative). A candidate item: take both passes' linear weights from
  one compiled order.

**Status 2026-11-08.** All done or closed. D11 (remove) took the CUDA/HIP port, its emulator and the twin check out of
the tree, so P4's "four loops" are two, both Metal:
- P4.3: step 2 done as `gpuRunPass` in `gpu_metal_retry.h` (the fault policy, the snapshot checkpoints, the 3-per-chunk
  and 16-per-deposit limits, the chunk rescale and the log lines; each host supplies its encode, restore and escalate
  steps: `ps_metal_host.cc:runChunked` per pass, `dtfe_metal_host.cc:runBatch` per batch, which keeps its `drainQueue`
  and shared-box path inside its restore). Step 3 closed by construction: `gpu_retry_common.h` is folded into
  `gpu_metal_retry.h`, its only includer once the CUDA twin was gone. Gates: `bytecmp --gpu-exact` 261 files / 0 differ,
  the retry suites PASS with the same log shapes, and an A/B against `f9f6e2f` under 16 fault-injection scenarios
  (innocent, timeout, oom, persist; checkpoint cadences 0/1/2; partitions; the shared box) 61 files / 0 differ.
- P4.4: the twin check went with the port. Its finding (the first moments under two function-constant pipelines, above)
  stays as the candidate item.
- P4.5: moot, one kernel text is left.

Gate for each item: G-byte, G-gpu (G-emu until D11). The retry items add `tests/gpu_retry_check.sh`,
`tests/dtfe_gpu_retry_check.sh` and `DTFE_GPU_FAULT_AT` runs.

**P4.1 Move the remaining identical host helpers into `gpu_host.h`.** — *done 2026-11-07.*
- `psHostFirstChunk` and `psHostCheckpointEnv` are textually identical in `ps_metal_host.cc` and `ps_gpu_cuda.cu`.
- So is the exact item-list fill inside `psHostExactItems`. Make it one shared `psBuildExactItems(const float*, …)`.
  P9.6 needs that.
- `psHostReportPasses` differs only by the backend's name.
- Risk: low. Effort: S.

**P4.2 One parameter-struct header per estimator.** — *done 2026-11-07.*
- `DepositParams` exists in four places:
  - PS: `ps_deposit_params.h` plus its MSL copy in `metal/ps_deposit.metal`;
  - DTFE: `dtfe_metal_host.cc`, `dtfe_gpu_cuda.cu` (with an extra `bad` pointer) and `metal/dtfe_deposit.metal`.

  The only guard today is a comment, "must match byte-for-byte".
- Change: a `dtfe_deposit_params.h` with `static_assert(sizeof …)`, and `static_assert`s on `ExactItemParams`. The
  Makefile already prepends text to the MSL source, so the host header can be prepended the same way, with `int32_t`
  mapped to `int`.
- Risk: low. Effort: M.

**P4.3 One chunk-and-retry loop.** — *done 2026-11-08 (step 1 and half of step 3 2026-11-07; D11 left two loops, now one `gpuRunPass`).*
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

**P4.4 A Metal↔CUDA twin check, before any kernel text changes.** — *done 2026-11-07; deleted with the port 2026-11-08 (D11).*
- Nothing compares the twin kernels function by function today: `validate_deposit` loads only the Metal source, and
  `cudaemu_check` compares whole binaries.
- Change: compile `gpu_exact_clip.cuh` (through `gpu_cuda_compat.h` in emulation) as plain C++, feed it the same
  random tetrahedra and cells as `validate_deposit`'s tests 7 and 14, and compare with the Metal output within float tolerance.
  Run it in CI's cudaemu job.
- Also add a "lockstep pairs" table (function, Metal location, CUDA location) to `metal/README.md`.
- Risk: none. Effort: M.

**P4.5 (optional, after P4.4) Shared scalar GPU math.** — *closed 2026-11-08: moot after D11 (one kernel text).*
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

**Status 2026-11-08.** All done.
- P5.5: `make_parser(snap_default=, snap_help=)` replaces the private argparse edits in six scripts; `add_sim_arg`,
  `add_snaps_arg`, `add_out_arg` (`--output` and `-o` too) and `add_rv_definition_arg`. `-o` is now a short option of
  `analyze.py` and nine plot scripts. The rest closed 2026-11-08 (the batch after d7626b6), mostly without code:
  `make_image_plane.py`'s hand-written `DTFE_SIM` fallback goes through `require_sim` (not `add_sim_arg`: its
  `DTFE_SIM` default would hide an explicit `--combined`, and the pipeline calls the tool with `--same-as` or
  `--combined` alone); `void_outlines.py` (and the launcher's `void_cache.py build`) take the simulation from `--sim`
  by setting the process's `DTFE_SIM` before `config` is imported, refusing only a `config` already imported for
  another simulation (`name_simulation`); the three tools stay on `add_sim_arg` / `require_sim` (`make_parser`'s six
  field flags are not theirs, and `--snap` is required in two of them); no `add_void_args` (only `--rv-definition` is
  shared, and it has its helper; `--rv-range`, `--radius`, `--keep-overlaps` differ in mechanism or help; the rest have
  one caller each); no tools bootstrap (the launcher's two path loaders, `explore.py:_void_tool` and
  `grids.available_gb`, would need `python/tools` on `sys.path` around `exec_module`; three `sys.path.insert` lines
  stay). `--out` / `--output` / `-o` and `--rv-definition` / `--r-v-definition` were already uniform; three plot scripts
  (`plot_void_environment`, `plot_void_population`, `plot_void_tracking`) write under `FIGURE_ROOT` with no `--out`.
- P5.9's two driver quirks are fixed too: `run_dtfe.sh --help` no longer prints its `set` line, and
  `run_ps_dtfe.sh` with `DRY_RUN=1 AUTO_TUNE_REPORT=1` no longer leaves an empty temp file.

Gate for each item: the py suites, `tests/run_scripts_check.sh`, G-docs, G-move for moves, and a byte comparison of
the `.npz` / `.csv` products and the `--help` text of every script touched. P5 is independent of the C++ phases.

**P5.1 Snapshot helpers in `dtfelib/cli.py`.** — *done 2026-11-07.*
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

**P5.2 One output-tag rule.** — *done 2026-11-07.*
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

**P5.3 Figure saving through `figures.save_figure` everywhere.** — *done 2026-11-07.*
- Seven sites call `savefig` directly: `plot_DTFE.py`, `plot_cosmic_web.py`, `plot_halo_tracking.py`,
  `plot_void_tracking.py` (×2), `plot_pointeval_panels.py`, `pointeval.py`.
- Make the `dpi` default `None`, meaning `config.DPI`. That drops about 20 redundant `dpi=DPI` arguments.
- The two tracking plots use 200 dpi on purpose; name it `config.TRACK_DPI`.
- Build `OUTPUT_DIR` one way: `config.figures_path`, not `LOCAL_FIGURES_ROOT` or `FIGURE_ROOT`.
- Risk: low. Effort: S.

**P5.4 Small helpers copied between scripts** — *done 2026-11-07.* move into `figures.py` or `dtfelib`:
- `_title`, three copies;
- `plot_finite` and `unit_cos`, two copies each;
- `smooth_components`, two versions;
- the `scipy.fft` shim in `pipeline.py` and `spectra.py`, which becomes `dtfelib/_fft.py`.
- Risk: none. Effort: S.

**P5.5 Argument helpers.** — *done 2026-11-08 (the rest closed without code; see the status).*
- `plot_eigenvalues.py`, `plot_shear_triaxial.py` and `plot_smoothing_comparison.py` reach into argparse's private
  `_option_string_actions`. Replace that with `make_parser(..., snap_default=...)`.
- Add `add_void_args`, `add_snaps_arg`, `add_out_arg` (a `Path`) and `add_rv_definition_arg`. Accept both spellings,
  `--r-v-definition` and `--rv-definition`.
- Tools: `compare_deposits.py`, `void_outlines.py` and `make_image_plane.py` should go through `make_parser` /
  `require_sim` (today some ignore `DTFE_SIM`), and share one `_bootstrap`.
- Make `--out` the name everywhere, with `--output` / `-o` as an alias.
- Never rename a flag the launcher emits; only add aliases.
- Risk: low. Effort: M.

**P5.6 A series driver.** — *done 2026-11-07.*
- Eight scripts repeat the same loop: snapshot list → mkdir → `process_snapshot` per snapshot → "Completed: n/m" /
  "Failed snapshots".
  - `plot_PDF_CDF.py`, `plot_eigenvalues.py`, `plot_contour_filter.py`, `plot_ellipses_contour_3D.py`,
    `plot_shape_filter.py`, `plot_velDiv_den.py`, `plot_shear_triaxial.py`, `plot_marked_correlation_BBKS.py`.
- Change: `cli.snapshot_series(args)` + `cli.run_series(items, fn, label)`.
- First grep `python/gui` for parsers of these lines. The launcher reads script stdout.
- Risk: low. Effort: M.

**P5.7 Long functions and mixed modules.** — *done 2026-11-07.*
- `plot_void_population.py:main`, 272 lines: split into setup, per-snapshot statistics, figures, tables.
- The cache code in `analyze.py` (about 420 lines) moves to `dtfelib/cache.py`. The `analyze.py cache` subcommand
  stays.
- `pipeline.py` (1051 lines) holds three subjects:
  - critical points → `critical.py`;
  - void-radius measurement → `void_radius.py`;
  - smoothing and the products cache, which stay.

  The moved parts are re-exported from `pipeline.py`.
- Risk: low. Effort: M. Verify: G-move.

**P5.8 Stale configuration copies.** — *done 2026-11-07.*
- About 10 plot scripts copy `config.SMOOTHING_SIGMA_CELLS`, `DPI` and `SNAPSHOT_TO_REDSHIFT` at import, while
  `pipeline.set_smoothing` rewrites `config` at run time.
- Read the values where they are used. Then add type hints and docstrings to the public functions of `pipeline.py`
  (1 of 39 hinted) and `figures.py` (0 of 24).
- Risk: low. Effort: M.

**P5.9 Bash strictness.** — *done 2026-11-07.*
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

**P5.10 Test the download retry.** — *done 2026-11-07.*
- `download_snapshots.sh:fetch_listing` retries the chunk list up to three times, but no check reaches that loop: the
  fake `wget` in `run_scripts_check.sh` fails its test snapshot already at the listing.
- Change: give the fake a per-run counter that fails the first `wget -i -` call and passes the second. Assert:
  - exit 0, "attempt 2 of 3" in the log and the chunk in place;
  - a chunk that keeps failing still exits non-zero.
- Keep the grep-based listing parser: the script must run without Python.
- Risk: low. Effort: S. Verify: `run_scripts_check`.

---

## P6. One flag scheme, with a deprecation path

**Status 2026-11-08.** P6.1–P6.5 done, with D2, D3 (the `*_METAL_*` knob aliases are no longer read) and D9, and the
second pass P6.7 (a-f, below; committed as `d7626b6`; every gate green on 2026-11-08: the battery twice, bytecmp 261/0, the tuner set 110/110); P6.6 (its removal list is below) is for the next release. Choices made where the plan left room: the deprecation line goes to stderr in lower
case (`warning: --old is deprecated, use --new`), so stdout and upper-case `WARNING` greps are unaffected; the
standard binary gets no `--ps-gpu` alias (it never had one); README and `demo/COMMANDS.md` may name an old spelling
only on a line that says "deprecated"; the help never names them; the launcher still exports `DTFE_METAL`,
`PS_METAL` and `DTFE_EXACT_AVERAGE` beside the new names for one release. Boost's prefix matching: `--lambda` and
`--full` are now ambiguous abbreviations (nothing uses them). For P6.6, remove: the hidden options and their shadow
members, `deprecatedOptions` and `canonicalCommandToken` (keep the canonicalisation only if old checkpoints matter),
the `--full_help` fallbacks in `checkFileNameOptions` and `presets.flag_help`, the DEPRECATED tables in `results.py`,
`run_ps_pipeline.sh` and `docs_drift_check.py`, the drivers' old-name reads and the launcher's old-name exports. The
`canonical_command` readers can stay for old run logs. Found by the night's review and fixed: a config file's
`lambda_th` and a command-line `--lambda-th` were refused as a disagreement (the command line now wins across the
spellings), and the Custom tab's repeat check missed an old spelling. The tests now use the canonical names except
where an old spelling is the point (alias, resume and run-log reader checks; bytecmp and the tuner set, which also
run the reference binaries).

**P6.7 (owner, 2026-11-07 evening): a second, complete flag pass.** "Clean up the flags as well. It is a big mess
right now; regular and PS DTFE use different flag names and the same things are called different across methods."
So the names P6 kept (the camelCase upstream flags, the PS-only names without a prefix, the drivers' PS_*/DTFE_* pairs,
the python CLIs' spellings) are in scope after all: one name per concept across both binaries, both drivers, the
launcher and the python scripts, one style, the old names as deprecated aliases for one release (P6's mechanism).
Like every code item, before phase R.
- Audit (2026-11-07, a workflow: three inventories of every name -- 96 binary options and field values, the drivers'
  and binaries' knobs, the python CLIs and launcher --, a design, a critique of the design against the code with 37
  corrections). Rules: kebab-case whole words; `ps-` exactly for stream / Lagrangian / caustic concepts; one name
  per concept across both binaries, both drivers, the launcher and python; `pts-` = writes `.pts_<x>`; a driver
  knob is its flag in UPPER_SNAKE; formats never move (output suffixes, cache-key labels, AUTO-TUNE-REPORT keys,
  QSettings and RunSpec keys).
- Owner's decisions: `--subsamples k` per axis in BOTH binaries (k^D points per cell; PS `--avg-subsamples` and
  standard `-s/--samples N` become aliases, the latter keeping its exact total); `--ps-merge index-order |
  completion-order | fixed-point` for the three merge switches; the knobs both drivers share keep the `DTFE_` prefix
  (`DTFE_GPU`, `DTFE_EXACT_AVERAGE`, `DTFE_RESUME`; `PS_GPU`, `PS_EXACT`, `PS_RESUME`, `DTFE_EXACT`, the `*_METAL`
  names become aliases); `--ps-window` becomes `--grid-window`; the drivers keep `-m`/`-e`.
- Main renames: `--MpcUnit --paddingMpc --regionMpc --partNo --redshiftCone --lagrangianInput --NGP --CIC --TSC --PCS
  --SPH --Voronoi --noTest --density0 --randomSample --approxPSD --origin --extensive --options` to kebab-case or
  clearer names; `-i/--input` and `-o/--output` (type codes) to `--input-format` / `--output-format`; the binaries'
  `-m/--method` to `--average-method` (drops the letter the drivers use for the GPU); `--per-stream*`,
  `--pts-lagrangian`, `--pts-conventions` to `--ps-*` (PS only); `--ps-lag-padding` to `--ps-lagrangian-padding`;
  the binaries' env knobs to one `DTFE_` name per concept (`DTFE_OVERLAP`, `DTFE_CHECKPOINT`, the GPU ones where the
  meanings match); python: `--method {ps,dtfe}` to `--estimator`, one smoothing name `--smooth-mpc` /
  `DTFE_SMOOTH_MPC`, lists as space-separated `nargs`, `-d/--sim-dir` vs `--data-root`.
- Old names stay as deprecated aliases with one warning line for the release that carries P6 and P6.7 (P6's
  mechanism, extended: `vm` keys rewritten in `foldDeprecatedAliases`; env aliases warned once at start-up); the
  run-log readers keep their old-to-new table for good (a new `dtfelib/runlog.py`); `--redshiftSpace` stays (its
  line is in every cache key).
- Steps, each gated on its own: P6.7a readers; P6.7b binaries with their emitters; P6.7c drivers; P6.7d python CLIs;
  P6.7e launcher; P6.7f docs and tests. Full design, critique and spec: the session notes of 2026-11-07
  (`p67-flag-design.md`, `p67-flag-critique.md`, `p67-spec.md`); the full rename table follows (copied in P6.7a).
- **P6.7a done 2026-11-07 (readers).** `python/dtfelib/runlog.py` holds the permanent table (`RENAMES`, `SWITCHES`,
  `VALUE_RENAMES`) and `canonical_command` with the value transforms; `gui/results.py` re-exports it
  (`DEPRECATED_FLAGS`, `canonical_flag`, `canonical_command`) and reads `--subsamples`, `--ps-merge`,
  `--ps-point-conventions`, `--ps-lagrangian-input`; `dtfelib/io.py`'s threshold and weighting reads go through it
  (`gui/grids.py` and `gui/runs_view.py` read command lines only through those two). `run_settings` carries the same
  rows as `old[]` / `sw[]`, compares the canonical names, and now also `--face-gradient`, `--ps-lagrangian-padding`,
  `--ps-alpha-shape`, `--ps-halo-release`; `docs_drift_check.py` (E) keeps runlog.py, results.py and `run_settings`
  equal and the binaries' alias rows inside it. `presets.flag_help` keeps each entry under the binary's own name and
  the canonical one. Nothing emits a new name yet.
- **P6.7b done 2026-11-07 (binaries and their emitters).** `user_options.cc` registers every binary rename of the
  table as the canonical option, with the old name a hidden alias bound to the same member and no default;
  `foldDeprecatedAliases` now runs at the end of `parseOptions` (after the command line and the `--config` file are
  stored, before the one `po::notify`) and moves each alias's `vm` entry onto its canonical key (a defaulted
  canonical replaced, both spellings with different values refused, a defaulted bool_switch alias dropped), so
  every `vm.count` / `vm[...]`, `conflicting_options` and `option_dependency` reads the canonical key only; the P6
  shadow members are gone. Kinds beside the plain renames: PS's merge switches stay `bool_switch`es on their own
  members (`deprecatedSwitches`) and `--ps-merge MODE` sets the same members, so "given vs defaulted", fixed-point
  beside index-order and `--resume` over completion-order keep their meaning; standard `-s/--samples N` keeps its
  total (`--subsamples k` sets k^D; both given must agree), `-m/--method` is a rename of `--average-method`;
  PS's `-m`/`-s` warn "no effect" and are dropped; `--ps-stream-density dtfe` warns and reads as the new default
  `linear`. `--ps-per-stream*` and `--parallel-triangulation` are PS-only in both help tables (the standard binary
  no longer registers the per-stream options). `canonicalCommandTokens` gives the checkpoint key the readers'
  spellings (and the long name of every short letter); `commandKeyLines` drops `--ps-merge` (the merge switches left
  its flag list). The binaries' env knobs: one `Env::renamed` table in `env_knobs.h` (a `Name` built from a new name
  finds its old one; `DTFE_NO_PROGRESS` an inverting alias of `DTFE_PROGRESS`), `Env::warnRenamed()` in `main`
  names each old one set, once, before any thread starts. `PS_GPU_CHUNK` and `PS_GPU_DEBUG_NOATOMIC` stay apart
  (standard's twins count work items and take a level). The emitters (`runspec_custom.py`, `explore_zoom.py`,
  `dtfelib/estimator.py`, `presets.OPTION_TEXT`; `skeleton.py` emits no binary flag) moved in the same step; the
  Custom tab passes `--ps-merge completion-order` only off the default, and `CustomSpec.stream_density`'s saved
  `'dtfe'` is emitted as nothing (the default `linear`). Driver-knob names in the help and the refusal texts
  (`ORDERED_MERGE`, `DTFE_EXACT`, `PS_RESUME`) wait for P6.7c; README / COMMANDS.md still name the old flags, so
  `docs_drift_check` part A is red until P6.7f.
- **P6.7c done 2026-11-08 (drivers).** `config.sh` holds the resolver: `knob [-e] NEW OLD...` (new wins; an old
  name that is set is copied with one `warning: OLD is deprecated, use NEW` line or, beside a different new value,
  said not read; no refusal during the window; the old name is unset afterwards and the new one exported when the
  old was, so the pipeline's driver neither warns twice nor misses it; `-e` counts an empty value as set, for
  `PS_LAGRANGIAN_INPUT`), `knob_merge` (`ORDERED_MERGE` / `FIXED_POINT_MERGE` -> `PS_MERGE`, fixed-point winning),
  and `knob_on` / `knob_bool` / `knob_gpu` (env_knobs.h's rule: 0/false/no/off off, anything else on, empty = unset;
  every driver switch normalised to 1/0, `gpu_auto_resolve` and the pipeline's exact test included). Each `knob`
  line sits before the knob's `VAR="${VAR:-default}"` line, which stays; `py_gui_runspec_test.py` reads the `knob*`
  lines for the older names. Each driver aliases only the names it read before (run_dtfe.sh: `DTFE_METAL`,
  `DTFE_EXACT`, `TESS_CACHE`; run_ps_dtfe.sh: `PS_GPU`, `PS_METAL`, `PS_EXACT`, `PS_RESUME`, `AVG_SUBSAMPLES`, the
  merge pair, `PS_PARALLEL_TRI`, `PS_LAG_PADDING`, `LAGRANGIAN_INPUT`, `PTS_CONVENTIONS`, `TESS_CACHE`; the
  pipeline: `PS_GPU`, `PS_METAL`, `PS_EXACT`, `PTS_CONVENTIONS`, `WINDOW`, `PY`; config.sh: `DTFE_GRID_SIZE`,
  `DTFE_PADDING`, `DTFE_SPACE_CHECK`; download_snapshots.sh: `PY`). `DTFE_EXACT_AVERAGE` now wins over `DTFE_EXACT`
  (the precedence flipped). `run_dtfe.sh` gains `-n` / `SUBSAMPLES` (passed only when set) and `THREADS`; both
  drivers take `--sim --sim-dir --grid --gpu --exact-average --subsamples` (a token-for-token pre-pass in
  `driver_options`, so download_snapshots.sh takes `--sim` / `--sim-dir` too). run_ps_dtfe.sh emits the canonical
  binary flags (`--input-format --mpc-unit --subsamples --ps-lagrangian-input --ps-lagrangian-padding
  --ps-point-conventions`) and `--ps-merge MODE` only when `PS_MERGE` is given (a given index-order too). So a plain
  driver run no longer passes `--ordered-merge`, and a small run under the tuner's speed split no longer keeps the
  memory gate unless `PS_MERGE=index-order` is given. The launcher still exports `ORDERED_MERGE=1`, so its runs keep
  the gate. The pipeline resolves `GRID_SIZE` through config.sh (critique 12), keeps the letters in its
  run_ps_dtfe.sh call and hands on `DTFE_GPU` / `PS_POINT_CONVENTIONS`. The binaries' refusal and help texts name
  `DTFE_RESUME=0` (both drivers), `PS_MERGE=...` and `DTFE_EXACT_AVERAGE=0`. `run_scripts_check` (T) asserts the
  alias matrix.
- **P6.7d done 2026-11-08 (python CLIs and their launcher emitters).** `dtfelib/cli.py` holds the machinery:
  `new_parser` (`allow_abbrev=False`; every python CLI, `make_parser` included), `deprecated_alias(parser, old, new)`
  (a hidden option with the new one's dest, nargs, type and choices that hands its value to the new option's own
  action and warns once on stderr, `warning: --old is deprecated, use --new`; a required new option becomes a
  required pair, so the old spelling alone still satisfies it), `deprecated_option` (an old option with no single new
  one: make_image_plane's `--u0..--v1`), `add_list_arg` (space-separated lists; a comma-joined value still split with
  one warning), `choice_type` (renamed values: `--criterion density_raw` -> `density-raw`, kept `density_raw` inside
  for the npz keys and figure tags; panels' `--project mean` -> `slab`), `smooth_mpc_env` and `SMOOTH_MPC_ENV_NAMES`
  (`DTFE_SMOOTH_MPC`, `DTFE_SIGMA_MPC` read with a warning; config.py, the cache inventory, analyze.py's refusal, the
  Explore void helper's and crossbox workers' environment strip both). The dests stay what the scripts read
  (`--estimator` is `args.method`, `--n-halos` `args.top`, `--per-void` `args.voids`, `--maps` `args.fields`, ...).
  Renamed: `--method` -> `--estimator` (make_parser, plot_pointeval, plot_void_crossbox and its worker and re-exec,
  void_outlines, gui/void_cache), `--sigma-mpc` -> `--smooth-mpc`, `--r-v-definition` -> `--rv-definition`,
  `--size-rv` -> `--size-rv-definition`, the tracking scripts' `--snap` -> `--start-snap` (confirmed: `--ref-snap`
  means two other things; make_parser's `snap_option`), `--top` -> `--n-halos` (`--n-voids` stays), `--subhalo` ->
  `--subhalos`, `--nbins` -> `--n-bins`, `--z` -> `--redshifts`, `--a/--b` -> `--prefix-a/--prefix-b`,
  `--voids` -> `--per-void`, plot_pointeval `--fields/--smooth/--smooth-derivatives` -> `--maps/--smooth-px/
  --smooth-derivatives-px`, plot_pointeval_panels `--prefix` -> `--root`, `-o/--output/--out` -> `-o/--out`
  (`add_out_arg`), merge_HDF5 `--data-dir` -> `--sim-dir` and its positional snapshots -> `--snaps`, analyze.py gains
  `--sim`, `--data-root` (set before config is imported) and `--snaps` (positionals warned), make_image_plane
  `--plane-window U0 U1 V0 V1` (the pipeline passes it), download_snapshots.sh `--groupcats/--trees/--ics/--api-key`
  (config.sh `driver_long_options`). The emitters (runspec_plot, runspec_pipeline's render step, runspec_data's merge
  steps, explore_voids) pass only the new spellings; `py_gui_runspec_test` checks every python flag the launcher
  emits against its script's `--help` as a whole word (a substring test passed `--size-rv` inside
  `--size-rv-definition`), and `tests/py_cli_alias_test.py` (CI) parses each old spelling against the new one. The
  launcher's knob exports (`PY`, `WINDOW`, `PS_*`) and its saved values (`project` 'mean', `criterion` with `_`) are
  P6.7e's (the emitters map them meanwhile); README / COMMANDS.md python spellings are P6.7f's.
- **P6.7e done 2026-11-08 (launcher remainder).** The jobs export every run-script knob under its new name AND its
  older one, the same value, until P6.6 (critique 7: a stray older export the launcher inherits, install_gui_app.sh
  importing every `DTFE_*`, is overwritten instead of standing beside the new one): `DTFE_GPU`, `DTFE_EXACT_AVERAGE`
  (Grids PS jobs now pin it too, so a login `DTFE_EXACT_AVERAGE=1` no longer turns a sampled PS job exact),
  `DTFE_RESUME`, `PARALLEL_TRIANGULATION`, `PS_POINT_CONVENTIONS`, `SUBSAMPLES`, `TESSELLATION_CACHE`,
  `PLANE_WINDOW`, `PYTHON` (`runspec.ps_run_env` / `ps_exact_env` / `ps_conventions_env`); the launcher reads `PYTHON`
  before `PY`; `PIPELINE_ONLY_KEYS` holds both spellings. The merge tick is `PS_MERGE=index-order` (explicit: the
  memory gate stays on under the tuner's speed split, as `ORDERED_MERGE=1` did), unticked `completion-order`. The
  drivers print one `warning: OLD is deprecated` line per older name (about six per PS job, two per standard one);
  the command shown and exported (`runspec.shell`, `OLDER_KNOB_NAMES`) names each knob once, by its new name.
  Saved state: `CustomSpec.from_dict` (QSettings, queued jobs, "Load settings" sidecars) runs `migrate_saved`: the
  extra options through `runlog.canonical_command` (standard DTFE: with `-s`/`-m`; PS-DTFE keeps those, they have no
  effect there: `averaging=False`) and `stream_density` `dtfe` -> `linear`; `PlotSpec.from_dict` migrates the
  criterion (`density_raw` -> `density-raw`) and the panels' `mean` -> `slab`, which the option choices now offer
  (idempotent, so no one-time flag; user presets hold no flag text). Labels: "Standard DTFE", "Cell average", "Grid
  and cell average", "Grid window (Mpc)" (Custom) vs "Plane window (Mpc)" (Pipeline), "Concurrent triangulations"
  (summary "N triangulations at once"); texts name `DTFE_RESUME=0`.
- **P6.7f done 2026-11-08 (docs and tests).** README.md, demo/COMMANDS.md and metal/README.md name only the new
  flags and knobs; each old name stands on a line that says so (the README's two "older name (deprecated) | new name"
  tables, the binaries' and the drivers', and metal/README.md's knob note); the README knob table names the new knobs
  only (`DTFE_EXACT_AVERAGE`'s per-driver default as prose: `script_defaults` merges the drivers). The resume notes say
  that a standard `--average-method 2` / `--particle-fraction` run without `--seed` is clock-seeded (not in the key:
  its resume matches no single run byte for byte; predates P6.7) and that an explicit `--ps-stream-density` and an
  omitted one have different keys (refused, by design). `src/config.cfg` uses `input-file`, `output-file`,
  `average-method`, `subsamples = 3`, `region-mpc` (the never-registered `sampling` is gone: `./DTFE -c src/config.cfg`
  parses again). change_log.txt: one entry for P6 and P6.7 with a "Renamed, old names accepted for one release" table
  per surface (binaries, their environment, the drivers, python, the labels) and the behaviour changes. The review
  items of a-c (critique addendum R1-R10): R1 the standard `samples` / `subsamples` pair across the command line and a
  `--config` file (`dropConfigAliasesOfCommandLine`: the command line's spelling wins, warned); R2
  `runlog.canonical_command(averaging=None)` reads the rule from the binary (`command_is_ps`: a PS-DTFE log keeps its
  no-effect `-s` / `-m`); R3 `--samples=27` keys as `--subsamples=3`, and `--samples 27 --subsamples 3` keys (and
  reads) as one `--subsamples 3`; R5 the stale comments (`auto_tune.h`, `config.sh`, the drivers' `-m` / `-s` lines,
  the hidden help's `--input-file`, `run_scripts_check`'s canonical-help regex); R6 the pipeline resolves
  run_ps_dtfe.sh's older knobs once (one warning per name per batch; `run_scripts_check` asserts it); R8 the
  `--ps-merge` runs in `ps_runtime_check` / `ps_resume_check` / `ps_resume_points_check` / `ps_fixed_merge_check`, an
  invalid-value line, `flag_spelling_check` F12 (`--output --extensive --SPH`, `--lagrangianInput`) and F13 (the config
  pair), F8's keys; R9 `docs_drift_check` (E) compares `deprecatedSwitches` with `runlog.SWITCHES`; R10 a dry run creates
  no cache folder. The binaries' `-m` / `-s` warnings say `-m/--method`, `-s/--samples` (`vm` cannot tell which was
  typed). **R7, decided 2026-11-08 (flipped back):** P6.7c first let a plain run_ps_dtfe.sh run (and the pipeline) pass
  no merge flag, so a small run under the tuner's speed split (N < 2e6) ran the index-order merge WITHOUT the memory
  gate, like a bare binary run. The owner chose the older behaviour: `PS_MERGE="${PS_MERGE:-index-order}"`, the flag
  always passed; with it the driver's comment and pass site, `config.sh`'s `knob_merge` note, the README knob table
  (rule C now compares the value), COMMANDS.md, the binary's `--ps-merge` help sentence and `run_scripts_check` R
  (the plain run asserts `--ps-merge index-order`; `FIXED_POINT_MERGE=0` alone leaves the default). Gates after it:
  `run_scripts_check` PASS, `docs_drift_check` 36/36, `flag_spelling_check` 56/56; `DTFE` and `PS-DTFE` rebuilt for the
  help text (grids untouched; the four real-binary suites that call the driver, `ps_partition_padding_check`,
  `ps_vertex_mass_check`, `ps_convergence_test` and `py_dtfelib_test`, PASS after it). The Custom tab's command still passes no `--ps-merge` at its default (a binary command:
  a log without the flag reads as the index order), so under the speed split only that path runs without the gate.
  `docs_drift_check.py` new rules: (A) also the merge switches, standard `--samples` and the python CLIs' old names
  (the `deprecated_alias` calls whose old name no parser registers any more; a per-script one such as `--prefix`,
  `--snap`, `--smooth` is not checked); (F) every `Env::renamed` old name, every driver `knob` / `knob_merge` old name
  and `DTFE_SIGMA_MPC` appears in the three docs only on a "deprecated" / "old(er) name" line, never in the README knob
  table; (G) from the built binaries' help (a SKIP line when they are not built): no visible name with a capital or `_`
  (`--redshiftSpace` exempt: its cache line), no `ps-` name in standard DTFE's help (so every `ps-` name is PS-only and
  every shared name lacks it), every `-h` name in the same binary's `--full-help`. Tests: every suite that is not about
  an old spelling runs the canonical flags and knob names (about 50 scripts); kept old on purpose: `tests/dev/bytecmp.sh` and
  `tests/dev/tune_report_set.sh` (they also run ref0; `cudaemu_check.sh` and `gpu_mass_check.sh` run only the checkout's
  binaries and were moved to the new names in the gate step, below), `flag_spelling_check`, `run_scripts_check`'s deprecation and run-log cases, `ps_resume_check` C8 (`--ps-gpu`) and
  C15, `dtfe_resume_check` D4 (`--lambda_th`), the run-log reader cases of the python suites, and one deliberate
  deprecated total, `--samples 20`, in `dtfe_sampling_check` (S3/S7 golden values) and `dtfe_mc_average_check` (A1):
  20 points per cell is no k^3, so P6.6 must either keep `--samples N` or move those two to a cube (and regenerate
  S7's golden file). A test that reads an environment variable of the binaries itself reads the old name as a fallback.
  Measured in this step (Mac): the three builds and `make library METAL=1`, 0 warnings, the build stamps unchanged;
  `docs_drift_check` 36/36 (A-G and the synthetic half); `flag_spelling_check` 56/56; the 54 suites the step touched
  (the py and GUI suites, run_tests.py, run_scripts_check, the resume, merge, window, 2D, auto-tune, cache, point,
  GPU and retry suites) all pass, `ps_auto_tune_check` Y5 on the second run (the first saw 5 partitions in flight
  against its bound of 4: the live gate's timing, no flag involved). Gates run after the step (2026-11-08 02:50–04:29): the full battery twice,
  31/31 with the tree unchanged (the second run after the gate step's own test edits); `bytecmp.sh --gpu-exact` against
  `ref0-70e153e` 261 files, 0 differ, no GPU fallback on either side; `tune_report_set.sh` 110 of 110 lines equal to the
  P8.10 baseline; no `warning: ... is deprecated` line in the suites' child logs (the stage logs never show the binaries'
  stderr, so the gate scanned the suites' temp folders) except in the suites whose point is an old spelling. The gate
  step's edits: `gpu_mass_check.sh` and `cudaemu_check.sh` to `--input-format` / `--mpc-unit` / `--subsamples`,
  `DTFE_GPU_FLOAT_MASS` and `DTFE_GPU_EMU_WARP` (their env reads keep the old name as a fallback); `ps_resume_check.sh`
  C10's array; `ps_point_conventions_check.sh` runs `--ps-stream-density linear` (labels `b_linear_*`).
  `cudaemu_check.sh`'s edit is checked by G-linux's cudaemu stage (CI no longer runs it). G-linux (Colima, Ubuntu
  24.04, 2026-11-08 04:35–05:35; the stages of CI's three jobs plus the warp-32 emulation): build, core, 2d, variants,
  old-python (python 3.10) and cudaemu 32 PASS; ruff found four dead imports and variables left by d and f
  (`plot_void_crossbox.py`'s `argparse`; two `subprocess as _sp` imports and a `repo` in `py_gui_runspec_test.py`:
  removed, ruff clean); the double stage's `ps_resume_check` C15 failed once with no checkpoint written at the kill:
  the VM's disk stood at 94%, and the checkpoint room rule (the grids' bytes + 1 GB free) refused the write, which the
  test does not report. On a fresh container: build and double PASS (22/22, C15 at 20 partitions as on the Mac).
- **Review of d–f (2026-11-08 04:40; 4 readers, an opus skeptic per finding, read-only):** 8 findings, 1 confirmed
  and fixed: the Custom tab's live repeat check mapped a merge switch in the extra options to the pair
  `--ps-merge MODE`, so it never matched the tab's own `--ps-merge` (merge tick off), while the migrated run copy (the
  memory check, the queue) did and the binary refused the doubled option; `names()` now takes the option word
  (`py_gui_runspec_test` covers the live spec and the copy, tick on and off). Fixed on the way, older than the change:
  README and COMMANDS.md still called `dtfe` the `--ps-stream-density` default (it is `linear`); `ps_point_eval_check`'s
  labels; `py_gui_app_test`'s `--maps` assertion compared a two-word prefix. Refuted as pre-existing or harmless:
  `command_is_ps` reads the command's first word (a binary path with a space reads as standard DTFE, as before);
  `command_dim` ignores a whole log's RUNNING line (no caller reads `-s` from a whole log); `gpu_mass_check`'s
  float-moments read takes either name (it then only skips one byte check). After the fixes: `py_gui_runspec_test`
  410/410, `py_gui_app_test` 388/388, `docs_drift_check` 36/36; ruff (CI's static check, run locally at 0.17.1) clean.
- **Left open after P6.7, all closed 2026-11-08 (the batch after d7626b6):** (1) R7, above: a plain
  `run_ps_dtfe.sh` run passes `--ps-merge index-order` again. (2) A run that fails with a message after its start no
  longer leaves `<output>.runlog.progress`: `run_progress.h` registers an exit handler that removes the sidecar while
  the run is still armed (`throwError` ends in `exit()`, which runs it); a killed run keeps its sidecar (no handler
  runs), an exit-3 'rescued' run disarms it first (`main.cpp`); `ps_runtime_check` R8 (a missing input in both
  binaries, the kill, the exit 3). Side effect worth knowing: a refused `--resume` removes the sidecar the killed run
  left under the same output name (the launcher then shows a failed attempt, not a death). (3) `ps_auto_tune_check`
  Y5's band is 2..6: the room holds about five partitions and the live gate's timing moves the count by one (4, now
  and then 5). (4) The launcher pins `SUBSAMPLES` and `THREADS` empty for standard Grids jobs (`runspec.py`;
  `py_gui_runspec_test`, `py_gui_app2_test`). (5) `dtfe_resume_check`'s shared array spells `--lambda-th`; only D2's
  and D4a's killed runs keep `--lambda_th` (`C_OLD`), and D2 asserts its warning. (6) `tests/dev/deprecation_gate.sh`,
  the battery's 32nd stage: the suites' temporary files (by name: `*.log`, `*.err`, `*.out`, `*.txt`, `*runlog*`, read
  as text, newer than the battery's start marker) hold no `warning: ... is deprecated` line outside an allow-list of
  the suites that test the old spellings; `--only deprecation_gate` alone reads the whole folder; the summary counts
  the files read. Known gap: five python suites (ps_3d_test, ps_standard_cross_check, ps_convergence_test,
  ps_nonperiodic_test, ps_regression_test) let the binaries' stderr through to the battery's stage logs, which the
  gate does not read. (7) Decided: the launcher exports `DTFE_AUTO_PTS_VEL_GRAD` only; the older spelling beside it
  would make the binary warn, and the alias goes with P6.6. (8) `ps_resume_check`'s kill cases print the binary's
  "checkpoint could not be written" warning and the free space when no checkpoint was left (`tests/lib.sh
  no_checkpoint_why`, C2 and C15). Review of the batch (one opus skeptic per group, one re-check of the gate): four
  real findings, three fixed (the gate's marker under `--only`, `grep -I` skipping NUL-bearing server logs, a
  symlinked temp folder read as empty) and one recorded as the known gap above; nits fixed (`void_outlines` exits in
  one line on an unknown simulation, a tautological launcher test, nested parentheses in `make_image_plane`'s usage
  error, `void_cache.py build` aligned, the Y5 comment, D2's assertion in D2's check). Battery after the batch
  (2026-11-08 14:16-14:44): 30 of 32 stages at first. `ci_core` failed `ps_tessellation_cache_check`'s standard-binary
  fingerprint check while a Colima image prune ran beside the battery; alone it passes, and that check keeps neither
  reader's output, so the cause is not known (an external-load flake, seen once). `ci_gui` failed `py_gui_app_test`'s
  Stop check, which sampled the process tree at a fixed 3 s after the exact 256^3 run had already ended (2.9 s now);
  it polls for the binary (up to 10 s) and the GUI stage passes (412/388/251/59). The `deprecation_gate` stage PASSED
  on the real battery folder (allow-listed: flag_spelling_check, ps_resume_check C8 and C15, ps_runtime_check R7).
  Left for a later batch: that fingerprint check discards both runs' output, so a repeat of the flake could not be
  diagnosed; keep the two logs in the suite's temp folder.

**The P6.7 rename table** (spec decisions applied; every old name accepted with one warning for the release that
carries P6 and P6.7, the readers' rows for good). Binaries (both unless "PS" / "std"):

| Old | New | Kind / note |
|---|---|---|
| `--MpcUnit` | `--mpc-unit` | rename; cache label `MpcUnit=` stays |
| `--paddingMpc`, `--regionMpc` | `--padding-mpc`, `--region-mpc` | rename |
| `--partNo` | `--partition-index` | rename; cache label `partNo=` stays |
| `--redshiftCone`, `--origin` | `--redshift-cone`, `--cone-origin` | rename (the dependency follows) |
| `--NGP --CIC --TSC --PCS --SPH --Voronoi` | `--ngp --cic --tsc --pcs --sph --voronoi` | rename |
| hidden `inputFile`, `outputFile` | `input-file`, `output-file` | rename (config files too) |
| `-i/--input`, `-o/--output` (type codes) | `-i/--input-format`, `-o/--output-format` | rename; the letters stay |
| std `-m/--method 1\|2\|3` | `--average-method` | rename; `-m` goes (the drivers' GPU letter); PS's ignored `-m` warns |
| PS `--avg-subsamples k` | `--subsamples k` | rename: k per axis, k^D points per cell |
| std `-s/--samples N` (a total) | `--subsamples k` (both binaries) | alias that keeps its exact total N; `--subsamples k` = k^D (method 3: k per axis, method 2: k^D random points); neither = today's defaults (20 / 27). Readers: `--subsamples k` when N = k^D, else `--samples N` |
| PS `--ordered-merge`, `--completion-order-merge`, `--fixed-point-merge` | `--ps-merge index-order\|completion-order\|fixed-point` | switch to value; given vs defaulted kept (an explicit index order keeps the memory gate on); `--resume` overrides completion order only; fixed-point beside ordered still accepted (fixed-point wins) |
| PS `--lagrangianInput`, `--ps-lag-padding` | `--ps-lagrangian-input`, `--ps-lagrangian-padding` | rename; cache label `lagPadding=` stays |
| `--per-stream`, `--per-stream-ids` | `--ps-per-stream`, `--ps-per-stream-ids` | rename, registered in PS only |
| PS `--pts-lagrangian` | `--ps-per-stream-lagrangian` | rename (writes `.pts_stream_lagpos`) |
| PS `--pts-conventions grid\|legacy` | `--ps-point-conventions` | rename |
| PS `--ps-window` | `--grid-window` | rename (it crops `--grid`; the pipeline's is `--plane-window`) |
| PS `--ps-stream-density dtfe` | `--ps-stream-density linear` | value (saved `CustomSpec.stream_density` / sidecars map it) |
| `--noTest`, `--density0`, `--randomSample` | `--no-padding-test`, `--mean-density`, `--particle-fraction` | rename; cache labels `density0=`, `randomSample=` stay |
| std `--approxPSD` | `--approx-phase-space-density` | rename; its help says it is unrelated to PS-DTFE |
| `--extensive` | `--scalars-extensive` | rename to a name the old one does not abbreviate |
| `--options` | `--reader-options` | rename |
| `--ps-gpu`, `--ps-exact-deposit`, `--ps-no-overlap`, `--full_help`, `--lambda_th` | `--gpu`, `--exact-average`, `--no-overlap`, `--full-help`, `--lambda-th` | P6, unchanged |
| `--redshiftSpace` | stays | its `redshiftSpace=` line is in every cache key and globals record |

Driver knobs and options (`config.sh`, `run_dtfe.sh`, `run_ps_dtfe.sh`, `run_ps_pipeline.sh`; new wins with a warning
during the window, no conflict exit; the `VAR="${VAR:-...}"` line shape kept for the knob-discovery checks):

| Old | New | Note |
|---|---|---|
| `PS_GPU`, `PS_METAL`, `DTFE_METAL` | `DTFE_GPU` (auto\|0\|1) | both drivers; PS auto under exact resolves to the CPU |
| `DTFE_EXACT` (P6.3, never shipped), `PS_EXACT` | `DTFE_EXACT_AVERAGE` (the shipped 1.0.0 name) | each driver keeps its default (std 1, PS 0) |
| `PS_RESUME` | `DTFE_RESUME` | also the refusal texts (`START_OVER`) |
| `AVG_SUBSAMPLES`, `-n k` | `SUBSAMPLES`, `-n k` / `--subsamples k` | per axis in both; the standard driver passes it only when set |
| `ORDERED_MERGE` + `FIXED_POINT_MERGE` | `PS_MERGE` | converted |
| `PS_PARALLEL_TRI`, `PS_LAG_PADDING`, `PTS_CONVENTIONS`, `TESS_CACHE` | `PARALLEL_TRIANGULATION`, `PS_LAGRANGIAN_PADDING`, `PS_POINT_CONVENTIONS`, `TESSELLATION_CACHE` | R9 |
| `LAGRANGIAN_INPUT` | `PS_LAGRANGIAN_INPUT` | `${OLD+x}`: an explicit empty value still means InitialCoordinates |
| `DTFE_GRID_SIZE`, `DTFE_PADDING`, `DTFE_SPACE_CHECK` | `GRID_SIZE`, `PADDING`, `SPACE_CHECK` | the pipeline's early `GRID_SIZE` capture goes through the resolver too |
| `WINDOW`, `PY` | `PLANE_WINDOW` (`make_image_plane.py --plane-window`), `PYTHON` | also `runspec.py` `PIPELINE_ONLY_KEYS`, `install_gui_app.sh` |
| (none) | `THREADS` in `run_dtfe.sh` too | `MPC_UNIT` stays a constant |
| `= "1"` tests | `knob_on` (1/0/true/false/yes/no/on/off) | everywhere at once, incl. `run_ps_pipeline.sh`'s exact test and `gpu_auto_resolve` |
| long options | `--sim`, `--sim-dir`, `--grid`, `--gpu`, `--exact-average`, `--subsamples` | `-m` / `-e` stay; the pipeline keeps the letters in its `run_ps_dtfe.sh` call (`runspec.py` `_PLAN`) |

Environment knobs of the binaries (`env_knobs.h` `Name{new, old}`; an alias warned once at start-up in `main`):

| Old | New | Note |
|---|---|---|
| `DTFE_PS_OVERLAP`, `DTFE_PS_CHECKPOINT` | `DTFE_OVERLAP`, `DTFE_CHECKPOINT` | one name in both binaries |
| `PS_GPU_TIMING`, `PS_GPU_FLOAT_MASS`, `PS_GPU_FLOAT_MOMENTS`, `PS_MASS_AUDIT`, `PS_DEBUG_DEGREES`, `GPU_EMU_WARP` | `DTFE_GPU_TIMING`, `DTFE_GPU_FLOAT_MASS`, `DTFE_GPU_FLOAT_MOMENTS`, `DTFE_MASS_AUDIT`, `DTFE_DEBUG_DEGREES`, `DTFE_GPU_EMU_WARP` | R10 |
| `PS_EXACT_ITEM_CELLS`, `PS_EXACT_ITEMS` | `DTFE_GPU_EXACT_CELLS`, `DTFE_GPU_ITEMS` (its standard twin, on/off) | |
| `PS_GPU_CHUNK`, `PS_GPU_DEBUG_NOATOMIC` | kept apart (P6.7b): standard's `DTFE_GPU_CHUNK` counts work items, its `DTFE_GPU_DEBUG_NOATOMIC` is a level 1-3 | PS counts tetrahedra, a switch |
| `DTFE_TBB_THREADS`, `DTFE_AUTO_PTS_VELGRAD`, `DTFE_TESS_CACHE_*` | `DTFE_TRIANGULATION_THREADS`, `DTFE_AUTO_PTS_VEL_GRAD`, `DTFE_TESSELLATION_CACHE_*` | |
| `DTFE_NO_PROGRESS=1` | `DTFE_PROGRESS=0` | an inverting alias |

Python CLIs and the launcher (`allow_abbrev=False`; one `deprecated_alias` helper in `dtfelib/cli.py`):

| Old | New | Note |
|---|---|---|
| `--method {auto,ps,dtfe}` | `--estimator` | `cli.make_parser`, `plot_pointeval`, `plot_void_crossbox` (its worker parser and re-exec too), `void_outlines`, `gui/void_cache.py` |
| `--sigma-mpc`, `DTFE_SIGMA_MPC` | `--smooth-mpc`, `DTFE_SMOOTH_MPC` | one helper; `explore_voids.py` strips both |
| `--r-v-definition`, `--size-rv` | `--rv-definition`, `--size-rv-definition` | |
| tracking scripts' `--snap` (the reference snapshot) | `--start-snap` | not `--ref-snap`, which means two other things (critique 23); confirmed in P6.7d |
| `--top`, `--a`/`--b`, `--nbins`, `--z`, `--subhalo` | `--n-halos`, `--prefix-a`/`--prefix-b`, `--n-bins`, `--redshifts`, `--subhalos` | |
| `merge_HDF5.py -d/--data-dir`, the drivers' `-d` | `-d/--sim-dir` | vs `--data-root` (the folder of simulations) |
| `plot_pointeval --smooth`, `--smooth-derivatives`, `--fields` | `--smooth-px`, `--smooth-derivatives-px`, `--maps` | |
| `plot_pointeval_panels --project mean`, `--prefix PATH` | `slab`, `--root PATH` | saved settings migrate |
| comma lists (`--sims`, `--snaps`, `--rv-range`, `--edges`, `--panels`) | space-separated `nargs` | commas still split with a warning for one release |
| `--criterion density_smoothed\|density_raw` | `density-smoothed\|density-raw` | converted at argparse only: npz keys and figure tags keep the internal names |
| `-o/--output/--out` | `-o/--out` | one helper (`dtfelib/cli.py`) |
| `plot_caustic_skeleton --voids` | `--per-void` | |
| `download_snapshots.sh -c/-t/-i`; `analyze.py`, `merge_HDF5.py` positionals | `--groupcats/--trees/--ics/--sim/--sim-dir/--api-key`; `--sim`, `--data-root`, `--snaps` | merge_HDF5.py: `--snaps` only (its `-d/--sim-dir` names the simulation's folder) |
| GUI labels | "Standard DTFE" / "PS-DTFE", "Cell average: Sampled / Exact", "Grid window (Mpc)" vs "Plane window (Mpc)", "Concurrent triangulations" | QSettings / RunSpec keys never change; `extra_args`, presets and "Load settings" sidecars migrate once |

Not moved: `-f/--field` and its values, output suffixes, the cache-key labels (`MpcUnit=`, `partNo=`, `lagPadding=`,
`scalarDataset=`, `region=`, `padding=`, `randomSample=`, `sampleSeed=`, `poisson=`, `density0=`, `redshiftSpace=`,
`type=`, `lagrange=`), AUTO-TUNE-REPORT keys, the numeric codes, `--max-concurrent`, `velocityStd_a` vs
`dispersion`, `Estimator(...)` keywords; `tests/generate_ps_test_data.py` is out of scope (a test tool); the
reference runs (`bytecmp.sh`, `tune_report_set.sh`, `cudaemu_check.sh`, `gpu_mass_check.sh`) keep the old flag and
env spellings for the whole window.

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

**Kept as they are** (after P6.7; P6 alone had also kept the camelCase upstream flags and the unprefixed PS-only
ones, which P6.7 renamed after all):
- the true PS concepts, `ps-` and PS-only: `--ps-caustics`, `--ps-caustic-cusps`, `--ps-vertex-mass`,
  `--ps-volume-weighted`, `--ps-linear-deposit`, `--ps-halo-release`, `--ps-alpha-shape`, `--ps-stream-density`, and
  P6.7's `--ps-merge`, `--ps-lagrangian-input`, `--ps-lagrangian-padding`, `--ps-per-stream*`, `--ps-point-conventions`;
- unprefixed although PS-only, being no stream concept: `--grid-window`, `--parallel-triangulation`;
- `--redshiftSpace` (camelCase; its `redshiftSpace=` line is in every cache key and globals record);
- `-f/--field` and its values (`density_a`, `velocityStd_a`, `scalarGradient`, ...), the output suffixes, the
  cache-key labels, the AUTO-TUNE-REPORT keys, the numeric codes, `--max-concurrent`, `velocityStd_a` vs
  `dispersion`, the `Estimator(...)` keywords, the QSettings and `RunSpec.to_dict()` keys;
- the drivers' `-m` / `-e` / `-n` / `-s` / `-d` / `-g` letters; the env knobs `PS_GPU_CHUNK` and
  `PS_GPU_DEBUG_NOATOMIC` (standard DTFE's twins mean something else).

**Order: the deprecation path.** Each step lands on its own, in this order.

**P6.1 A reader normaliser first.** — *done 2026-11-07.*
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

**P6.2 Hidden aliases in the binaries.** — *done 2026-11-07.*
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
  - the help assertions in `run_scripts_check.sh` (about lines 714–720, which assert that the standard `--help` lacks
    `--ps-gpu`; update them);
  - a config file that uses an old name.
- Depends: P6.1, P3.3.

**P6.3 Drivers and environment.** — *done 2026-11-07.*
- The drivers emit the canonical flags: `run_dtfe.sh`, `run_ps_dtfe.sh`, `run_ps_pipeline.sh`.
- The environment knobs become `DTFE_GPU` / `PS_GPU` and `DTFE_EXACT` / `PS_EXACT`. The old names are read after them:
  `${PS_GPU:-${PS_METAL:-auto}}`, `DTFE_METAL`, `DTFE_EXACT_AVERAGE`.
- The renamed knobs keep the exact-deposit rule: an unset or `auto` `PS_GPU` resolves to the CPU when `PS_EXACT=1`.
  - Today that is `config.sh:gpu_auto_resolve` with its `cpu_first` argument, which `run_ps_dtfe.sh` passes [267].
  - The old `PS_METAL` spelling must resolve the same way.
  - Add a `run_scripts_check` assertion for both spellings.
- Banner strings change in the same commit as the tests that grep them:
  - `GPU (--ps-gpu)` in `ps_auto_tune_check.sh` (lines 332, 430, 458, 621);
  - the `(--ps-exact-deposit; …)` message.
- The standard binary's `GPU (--gpu)` banner (`user_options.cc` [745], `run_dtfe.sh` [178]) already uses the
  canonical name and stays.
- Also fix a trap: the drivers' `-m` means GPU and `-s` means simulation, while the binaries' `-m` means method and
  `-s` means samples. Document it in the drivers' `--help` at least.
- Risk: medium. Effort: S–M. Verify: `run_scripts_check`, `ps_auto_tune_check`, G-docs.

**P6.4 Launcher.** — *done 2026-11-07.*
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

**P6.5 Docs and tests.** — *done 2026-11-07.*
- Update the README flag table, `demo/COMMANDS.md`, `change_log.txt` (a "Deprecated" list with the old→new table)
  and `tests/run_tests.py` (it gates a feature on `--full_help` content).
- `docs_drift_check.py` part A must reject deprecated names in the docs. It currently validates against every
  registered name, hidden ones included.
- The ~40 test scripts may keep the old spellings; they then exercise the aliases.
- Effort: M.

**P6.6 One release later: remove the aliases.** The precedent is the removal of `--metal` / `--ps-metal`. Effort: S.
The final alias tables (P6 and P6.7), all removed together:
- the binaries: `deprecatedOptions` (36 rows: the PS block, the standard `method` / `approxPSD` rows, the shared rest)
  with their hidden options, and `deprecatedSwitches` (PS-DTFE's 3 merge switches: `ordered-merge`,
  `completion-order-merge`, `fixed-point-merge`, their members `psOrderedMergeFlag` / `psCompletionOrderMergeFlag`
  then set by `--ps-merge` alone); the hidden `-m/--method` and `-s/--samples` of both binaries (standard: the total
  `noPoints` path in `readAveragingOptions` and `foldDeprecatedAliases`; decide first whether a total that is no k^3
  stays expressible: `dtfe_sampling_check` and `dtfe_mc_average_check` use `--samples 20`); the `--ps-stream-density
  dtfe` value; `spelled()`, the eq-form and de-duplication branches of `canonicalCommandTokens` (keep the
  canonicalisation only if old checkpoints matter) and the `samples` / merge arms of `dropConfigAliasesOfCommandLine`;
  the `--full_help` fallbacks in `checkFileNameOptions` and `presets.flag_help`;
- the binaries' environment: `Env::renamed` in `src/env_knobs.h` (18 rows, `DTFE_NO_PROGRESS` the inverting one) and
  `Env::warnRenamed`;
- the drivers: every `knob NEW OLD...` / `knob -e` / `knob_merge` line's old names (29 lines in `config.sh`,
  `run_dtfe.sh`, `run_ps_dtfe.sh`, `run_ps_pipeline.sh`, `download_snapshots.sh`; `knob_merge` itself), the headers'
  "Older names" paragraphs and the README / COMMANDS.md / metal/README.md "older name (deprecated)" tables;
- python: the 20 `deprecated_alias` and the `deprecated_option` calls, the `old_snaps` / `old_snapshots` positionals
  (analyze.py, merge_HDF5.py), the comma splitting in `_ListAction`, the `old=` maps of the `choice_type` calls,
  `DTFE_SIGMA_MPC` in `SMOOTH_MPC_ENV_NAMES`, the command builders' `mean`->`slab` and `_`->`-` value maps;
- the launcher: the older-name exports (`ps_run_env`, `ps_exact_env`, `ps_conventions_env`, `_run_option_env`'s
  `TESS_CACHE`, `runspec_pipeline.command`'s `AVG_SUBSAMPLES` / `WINDOW` / `PY`, `runspec_data`'s `PY`, the standard
  branch's `DTFE_METAL` / `DTFE_EXACT`), the old names in `PIPELINE_ONLY_KEYS`, then `OLDER_KNOB_NAMES` and the
  `shell()` filter, `runspec_custom`'s acceptance of `dtfe` beside `linear`; keep `migrate_saved` / `migrate_options`
  (old `.gui.json` sidecars stay on disk);
- the tests that exercise the aliases (`flag_spelling_check`, `py_cli_alias_test`'s old half, `run_scripts_check`'s (T)
  matrix and deprecation cases, `ps_resume_check` C8/C15, `dtfe_resume_check` D4) turn into "the old name is refused"
  checks; `docs_drift_check`'s `DEPRECATED` goes, its (F) old-name list then catches any doc still naming one.
Kept for good: the run-log readers' tables (`python/dtfelib/runlog.py` RENAMES / SWITCHES / VALUE_RENAMES,
`run_settings`' `old[]` / `sw[]`), and `tests/dev/bytecmp.sh` / `tune_report_set.sh` need ref0 re-frozen from a
post-rename build (or a per-binary spelling map) before the old spellings can go from them.

**Not adopted:** a `--deposit exact|sampled|linear` enum. It merges three switches that have different exclusion rules
(`user_options.cc`, the exact/linear checks), and it changes run-log text, the checkpoint key and the launcher mapping
for little gain. Revisit once the aliases have shipped.

---

## P7. Launcher polish

**Status 2026-11-07.** All done (P7.1–P7.12). A layout bug found on the way, fixed: with P7.2's folded Resources
group opened on a build without the TBB or GPU stamps, the build note on the parallel-triangulation tick made the
Grids, Custom snapshot and Pipeline forms 746 px wide in a 670 px pane (`presets.noted_label` now replaces the tick's
own explanation by the note; the app suite checks the widths with every group open and the notes shown).
- P7.1: the helpers are in `app_widgets.py` (spacing constants, `flush`, `group`, `Collapsible` /
  `advanced_group`, `labelled_row`, `field_grid`, `grid_row`, `button_row`, `tip` / `option_tip`, `ElidedLabel`) and
  used in the job tabs, `app.py`, `setup_wizard.py`, `explore_3d.py`'s 3D group and the run panel
  (`RunMixin._build_run_buttons`; no separate `run_buttons()` helper, it would have one caller).
- P7.4: `presets.LABELS`, field names from `grids.FIELD_LABELS`, `grids.colourbar_label` / `component_suffix`, and
  tooltips of at most two lines (an option's first sentence; the whole text under Help > What's This?), the
  Explore and 3D-view ones included (`tip_auto`; `py_gui_explore_test` part 12).
- P7.6: `app.py`, the save dialog, the setup wizard, the menus and `explore.py` are split into `_build_*` methods
  (attribute names, creation order and Tab order kept; `explore.py` and `explore_3d.py:_build_3d` checked by AST,
  then by a creation-order / Tab-order / geometry dump).
- P7.11: the on-screen volume and its GPU copy are kept at a job start (the view holds them).

The two layout bugs are P1.1 and P1.2.
- P7.1–P7.7 cover the launcher's tabs and Explore. P7.8–P7.12 cover the 3D view.
- Gate for every item: G-gui. The 3D items also run parts 10–11 of `py_gui_explore_test`, and `volume_gl_parity`
  on the Mac.

**P7.1 Style helpers in `app_widgets.py`.** — *done 2026-11-07.*
- Spacing constants: `GAP`, `GROUP_MARGIN`, `FORM_V_SPACING`, `COL_GAP`. They replace about nine scattered
  `setContentsMargins` / `setSpacing` calls.
- `group(title, *rows, collapsed=False)`, plus a collapsible `advanced_group(title)` that remembers its state under
  a new settings key.
- `labelled_row(label, widget, unit=None, tip=None)`: units as spin-box suffixes.
- `field_grid(names, columns=3)`.
- `grid_row()`: the "Grid ³ cells / sub-samples per axis" row, of which three copies exist today.
- `run_buttons()`.
- Apply the helpers to the 3D view's group (`explore_3d.py:_build_3d`) as well.
- Effort: M.

**P7.2 Less on screen by default.** — *done 2026-11-07.*
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

**P7.3 A stable top panel.** — *done 2026-11-07.*
- The shared Data panel is tall on the Grids and Plots tabs and short on the others, so the tab bar jumps by about
  250 px between tabs.
- Put the tab bar above the panel, or give the panel a fixed height. Drop the redundant "Data" title, and move
  "Setup…" into the Data root row.
- Where: `app.py:__init__` (left-pane assembly).
- Risk: medium; the tests use tab indices and `_tab_view`. Effort: M.

**P7.4 One wording table.** — *done 2026-11-07.*
- One `LABELS` dict for the texts that differ between tabs today:
  - field names: "density" vs "density ρ/ρ̄";
  - group titles: "Grid and fields" vs "Estimator and grid" + "Fields" vs "Grids";
  - quick-selection buttons: "Not yet run" / "Not merged" / "With grids".
- Sentence case, nouns for groups, verbs for buttons, "…" only when a dialog opens.
- Tooltips at most two lines; the long explanations go to Help. The `gpu`, `lam` and Pipeline-fields tooltips are
  essays today, and so are the 3D view's (`explore_3d.py`, lines 31–87; `DETAIL_TIP` grew with the GPU's timings).
- One `colourbar_label(label, req)` helper for the "log₁₀ " prefix and the component suffix. It replaces the copies at
  `explore.py` [1643], `explore_zoom.py` [743] and `explore_3d.py` [1487, 1664]. The existing title and
  colour-bar checks of `py_gui_explore_test` must pass unchanged.
- Risk: low (some tests match label text). Effort: S–M.

**P7.5 One estimator group builder.** — *done 2026-11-07.*
- About 60% of `_build_own_tab` (203 lines), `_build_grids_tab` and `_build_pipeline_tab` (191) is the same widget
  construction under different attribute names.
- Change: a table-driven `_estimator_group(prefix)` that removes about 300 lines and keeps the old attribute names
  (`o_*`, `p_*`, `field_checks`) as aliases, because the tests reach widgets by name.
- Risk: medium. Effort: M. Depends: P7.1.

**P7.6 Long UI builders split into private `_build_*` methods** — *done 2026-11-07.*, with attribute names kept:
- `explore.py:__init__`, 497 lines;
- `figsave.py:_dialog_class`, 344, with a 210-line nested `__init__`;
- `app.py:__init__`, 196;
- `setup_wizard.py:__init__`, 147;
- `explore_3d.py:_build_3d`, 151 (lines 1100–1251);
- `app_menus.py:_build_menus`, 125.

The 3D group is already a mixin in its own file (`View3DMixin`). Split `explore.py:__init__` the same way: one
`_build_*` per left-pane group, keeping the existing call `self._build_3d(col)` (explore.py:611).

Move the logic out of the UI modules: `explore_zoom.py:exact_zoom_plan` and `explore_grid.py:grid_figure` become
plain functions.
- Risk: low. Effort: M.

**P7.7 Explore details.** — *done 2026-11-07.*
- The "Voids" control appears twice.
- In the stream table, the value column is too wide and the "ρ/ρ̄" header is truncated.
- The save buttons sit in the right pane, away from the controls.
- The log strip's "!" button has no tooltip.
- Effort: S.

**P7.8 One rule for the values a map shows.** — *done 2026-11-07.*
- The duplication: `volume3d.py:field_values` [151–177] repeats the second half of `grids.py:OutputSet.slice` (from
  line 552): the vorticity-to-curl factors, the norm / trace / named component, and the velocity factor.
- Change: extract `OutputSet.component_values(name, arr, component)` in `grids.py` and call it from both. Keep the
  float64 order of the norm so that `slice()` stays byte-identical.
- Also give `grids.py` and `volume3d.py` public names for what the other modules reach by underscore:
  - in `volume3d.py`: `G._cube_key`, `G._cached_cube`, `G._CAPPED`, `o._velocity_factor`;
  - in `explore_3d.py`: `G._cube_key`, `VOL._file_shape`, `VOL._sources`, `VOL._prepared_bytes`, and since
    `70e153e` `VGL._STATE.get("renderer")` (in `View3D._gl_forget` and `gl_bytes`: the renderer if one was made,
    without making one). Give `volume_gl.py` a `current_renderer()` for it.
- Why: two copies of a physical rule diverge silently.
- Risk: medium; the map's path feeds saved figures. Effort: M.
- Verify:
  - G-gui;
  - part 11 compares the two paths for a vector component (add vorticity and a trace if absent);
  - saved map PNGs byte-identical before and after.

**P7.9 Move `_smoothed` out of `explore.py`.** — *done 2026-11-07.*
- `explore.py:_smoothed` [178–186] is Gaussian smoothing in cells, with periodic wrap; it is a no-op without scipy.
- Move it to `grids.py`, import it at the top of `explore.py` and `explore_3d.py`, and delete the function-local import
  at `explore_3d.py:112`, which works around an import cycle.
- Risk: low. Effort: S. Verify: G-gui; `explore_3d.planes3d` runs with `explore` not imported.

**P7.10 Remember the 3D view's choices across restarts.** — *done 2026-11-07.*
- `explore.py:ExploreControls.state()` and `restore_state()` keep only `void_shape` of the 3D controls.
- Add:
  - the style (`view3d_style`);
  - perspective (`view3d_persp`);
  - the planes' opacity (`view3d_opacity`);
  - the detail (`view3d_detail`);
  - the volume's opacity slider (`vol3d_opacity`).
- Restore them with signals blocked, as `void_shape` is, and use `d.get` with today's defaults for older saved
  states.
- Do not restore "Show in 3D" itself, nor a shown range: the range belongs to one volume.
- Risk: low. Effort: S.
- Verify: in `py_gui_explore_test`, set the widgets, take `state()`, restore it into a fresh `ExploreControls`, and
  compare. An old-format dict must leave the defaults.

**P7.11 One release for the launcher's cached cubes.** — *done 2026-11-07.*
- The job start (`app_run.py` [281–282]) and the memory report (`app_check.py` [62–63]) each call `G.drop_cubes()`
  and then `VOL.drop_volumes()`.
- Change: give `grids.py` a list of drop callbacks that `drop_cubes()` calls and sums. `volume3d`, which already
  imports `grids`, registers `drop_volumes` at import. The two sites then call `G.drop_cubes()` only, and a future
  cache cannot be missed.
- Decide, and write down, whether the volume on screen (`View3D.volume`: two bytes a cell, about 2.1 GB at 1024³) is
  released at a job start too. Since `70e153e` the same question covers the GPU's copy of it (`View3D.gl_bytes()`:
  1.1 GB at 1024³, freed by `_gl_forget`), which the memory check already counts.
- Test: add a check next to the `drop_cubes` checks in `py_gui_runspec_test`. Fill the 3D caches with a small volume,
  take the job-start path, and assert that `VOL.drop_volumes()` then returns 0.
- Risk: low. Effort: S. Verify: G-gui.

**P7.12 `volume3d.average_window` reuses `_run`.** — *done 2026-11-07.*
- Its submit / drain / cancel loop [318–330] repeats `volume3d._run` [83–106], with only a per-block "done" counter
  added. Give `_run` that counter (`progress['done']`) and call it. (`70e153e` rewrote the block sums inside it, not
  the loop.)
- Leave `volume_gl.py`'s two pool uses alone (the cube's half [467–493], the transparent-block rows [630–667]): they
  submit everything and are polled from the GL upload ticks, so they do not fit `_run`'s blocking shape.
- Store the worker count in a module constant instead of reading `ThreadPoolExecutor`'s private `_max_workers`
  [80, 94].
- Risk: low. Keep the "done" meaning and the cancel latency, both of which the user sees. Effort: S.
- Verify:
  - part 11's cancel and progress checks pass unchanged;
  - the averaged cube is byte-identical for a 20³ and a 64³ synthetic grid (`_block_mean_ref`).

---

## P8. Auto-tuner

**Status 2026-11-07 (evening).** P8.2 (as the partition span), P8.7, P8.9 step 1 and P8.10 done in code (below); the rest open. Gate changes first:
`tests/dev/bytecmp.sh` pins every run's plan (`--partition 1`, or `2 2 2 --max-concurrent 2`), so a tuner change
cannot move its grids; `tests/dev/tune_report_set.sh` appends 19 lines (110 in all) the P8 items move: standard
runs with the production fields (the old standard lines ask density and velocity only, where the loop and the
post-processing hold the same grids), PS CPU exact with points and in a window, and the face sums. With `ref0` the
first 91 lines are still the frozen set.
- P8.7: the budget's reason line now also names the live free memory when it does not cut the budget ("(the other
  programs leave 57.1 GB free: 53.1 GB after the 4 GB headroom)"), and `DTFE_MEM_FRACTION` when set; a simulated
  machine (`DTFE_RAM_GB`) prints as before. `ps_auto_tune_check.sh` J2c/J2d.
- P8.9 step 1: `AutoTuneStdPlanner::stdPeak` = max(loop, post-processing): the loop holds its own grids
  (`fixedLoop`, 52 B/cell) + the busiest partition; the post-processing all grids (`fixed`) + the share of the busiest
  partition the allocator keeps: all of it up to `kStdPostKeepFloor` = 1 GB, `kStdPostKeep` = 0.35 of a larger one.
  Never above the old sum; equal to it with `--scratch-dir`, without derived fields, or with partitions under 1 GB.
  The floor came from `dtfe_auto_tune_check.sh` E (128³, production fields): with 0.35 alone it predicted 0.53 GB
  against 0.84 GB measured (1.57×; the test's bound is 1.35×); the new SUMMARY line `before post-proc` (the
  peak when the post-processing began, `runProgressNoteLoopPeak`) showed 0.63 GB, so the whole 0.42 GB partition
  stayed and the derived grids came on top; std100_099 kept 0.5 GB of 11.8. Now 1.08× (GPU case 0.81×). Tuner set:
  5 standard production lines lower (256³: 2.99 → 2.59 GB), every other line identical. Phase R refits both
  constants from `before post-proc` on the real runs.
- Measured in passing (P10.5's booking): on the 64³-particle pancake at 256³ (production fields, CPU exact) the PS
  model over-predicts with and without the face sums alike (one domain +49% / +48%, 2³ × 8 +108% / +119%): the
  face booking adds no error of its own; the base model over-books small grid-dominated runs (P8.2's span, below).
- P8.2, rescoped by its own analysis (the session notes' P8.2 report): the CPU exact deposit's own working set is
  ~96 B per hit cell per deposit thread (`insideFlat`, `exMom`, `exW`; reused) and r3d's ~15 KB per thread, both
  negligible; the large error is the partition **sub-grid span**. The model booked `min(G, G/n + 40)` cells per
  axis whatever the snapshot, while the logged 5³ runs park ~4× the bytes per modelled cell at z = 0 as at z = 127,
  and the 1024³ exact run was −43%. Done: `psPartitionSpans` (`auto_tune.cc`) measures each split's largest
  sub-grid on the particles. They are binned on a Lagrangian grid (B = min(60, N^(1/3)/√3) per axis; periodic runs
  over the ownership period `[seam − L, seam)`, as `DTFE.cpp` tiles it); each bin's Eulerian extent per axis (minimal
  image of the bin's first particle, as the deposit wraps a tetrahedron); a partition's touched planes = its bins'
  extents widened by one particle spacing at the bin's own stretch and a cell; then `psSubgridBox`'s rule (the
  `[lo, hi]`, or the complement of the largest untouched run on a periodic axis). It replaces `G/n + 40` in
  `gpuSubCells` for n ≤ 6 (window runs and n > 6 keep the rule; `DTFE_AUTO_SPAN=0` restores it). Checked against
  the binary's own sub-grids (`DTFE_AUTO_TUNE_DEBUG=1` now prints each pass's `sub-grid: a b c`): pancakes and
  crossed waves, 32³–128³ particles, 128³ and 384³ grids, 2³ and 4³ splits: 1.00–2.09× the largest sub-grid
  (geometric mean 1.24×), never below; the old rule was 0.05–4.3× (20× low on the crossed waves at 384³). Cost
  0.01–0.5 s (one pass over the particles). Two wrong turns on the way, kept out: bins over the particles'
  Lagrangian box instead of the ownership period (the seam is `lagLo − padding`: a 12% shift on a 64³ set) read
  2× low; hulls with the 26 neighbour bins over-read by a bin width on fine grids. Tuner set: 66 of 110 lines
  change, every PS line with a split; standard lines identical. Phase R checks it on the real runs (the debug line
  gives the real sub-grids).
- Review of the evening's code (4 readers, a skeptic per finding; all 8 findings confirmed, all fixed):
  - P8.2: one domain never crops (`psUseSubgrid` is the partition path's), so `gpuSubCells(1)` books the whole grid
    again (the measured span under-booked a non-periodic clump 14×); the seam and the partition padding are replayed
    per split (`--ps-lag-padding 0`'s Eulerian rule depends on the split and moved the seam: 0.46× before); a split
    that is not cubic (`--partition 2 3 4`) is costed with its own counts, padding and measured span
    (`partsOf` / `padFactorOf`; the planner named it by `partition[0]` alone, `N/n³`, which also undercounted every
    2D split by n). `ps_auto_tune_check.sh` X: the booked sub-grid ≥ the largest real one on those cases (1.00–1.88×).
    I2 now checks the property (a split, sub-grids under 2³², the GPU kept): with the pancake's real 2³ sub-grid
    (~1.1e9 cells at 1700³, measured 760 × 428 × 428 at 850³) the budget takes the split to 5³.
  - P8.9: `velocityStd_a` is a loop grid (booked in both phases); the look-ahead gate uses `stdPeak`, and the loop's
    room `budget − fixedLoop − …` reaches the run (`AutoTuneReport::fixedLoop`); standard `--face-gradient`'s 9
    doubles per cell are per sub-domain (`faceCell` in `planOf`), not a full grid.
  - P10.4: the drift test covers crossed waves too (the y and z momentum words: 0.028 max / 7.1e-4 rms, ×4.0), and
    the thread, slab-mode and resume checks compare `.a_massFluxDiv`.
Earlier (2026-11-07): P2.3's first three reruns (the P2 status) give P8.1 two classes: the standard
GPU exact average +43% and the PS GPU sampled with points +27% and +41%, all over-predicting now that the parking is
bounded; the TNG50, 1024³ exact and window reruns are still to do. The standard planner's margin costs time: it chose
6³ where 3³ fit, and the run took 1.8× the hand-set one. Since P9.2, the single-precision build's CPU deposit follows
the core pool (`CorePool` in `src/thread_count.h`: holders = min(planned concurrency, partitions left)), which P8's
`cores_eff` should use. P8.10 (dynamic concurrency) proposed 2026-11-07, from the stopped TNG50 rerun. Order (owner, 2026-11-07): the code items come first, with constants from the existing logs; P8.1's re-baseline and P8.2's refit run in phase R.

**Status 2026-11-09 (phase R).** P8.1 re-baselined for the classes the night ran; P8.2 rescoped again (the start-up
zero fill) and P8.5 (1.4) in code, gates green (Gates, below); P8.3 and P8.4 not needed; P8.9 step 1's
check done, its refit open; P8.10 not measured. In code since `2f428bc`: `src/auto_tune.{h,cc}`.
- P8.1, the final model (`4e2b35b`) only; the 11-07 runs are an earlier model (the old `G/n + 40` span: a replica
  reproduces their predictions to 0.01 GB) and stay out of the fit:

  | Class | Run | Plan | Predicted → footprint (max RSS) | Error | Band |
  |---|---|---|---|---|---|
  | standard, GPU exact average, 512³ | `std100_099_final` | auto 3³ × 10 threads | 34.70 → 31.42 (41.22) GB | +10.5% | in |
  | the same + `--face-gradient` (`DTFE_FACE_CHECK=1`) | `std100_099_face` | auto 3³ × 10 | 34.91 → 28.22 (41.78) | +23.7% | over |
  | standard, GPU exact average, 512³, split by hand (a diagnostic) | `std100_099_split6` | 6³ × 10 | 31.70 → 35.11 (38.14) | **−9.7%** | **under** |
  | the same split + `--face-gradient` (`DTFE_FACE_CHECK=1`, a diagnostic; same plan and binary four hours apart: the footprint figure scatters by 11 GB while the max RSS agrees, so this class's error is bounded by that scatter) | `std100_099_split6_face` | 6³ × 10 | 31.71 → 24.00 (37.57) | **+32.1%** | **over** |
  | standard, CPU exact average + face gradient, 256³ (a diagnostic) | `std100_099_face_cpu256` | auto 2³ × 10 | 30.37 → 28.62 (30.75) | +6.1% | in |
  | PS GPU sampled + 4096² points, TNG100-3-Dark z = 0 | `ps100_099_final` | auto 6³ mc 3 | 38.26 → 27.21 (34.23) | +40.6% | over |
  | the same, TNG50-3-Dark z = 0 | `ps50_099_mc2` | hand 6³ mc 2, budget 50 | 51.26 → 34.93 (38.44) | +46.8% | over |
  | PS CPU exact + face sums, no points, 512³ | `ps100_099_face512` | auto 6³ mc 5 | 36.46 → 33.73 (51.18) | +8.1% | in |
  | PS CPU exact, no points, 1024³ | `ps100_099_exact1024` | auto 6³ mc 2 | 32.38 → 38.95 (55.99) | **−16.9%** | **under** |

- Standard: one snapshot and binary, three runs on two plans (3³ × 10, the face run adding `--face-gradient`; 6³ ×
  10), +10.5 / +23.7 / −9.7%; the two 3³ runs measured 31.42 and 28.22 GB (11% apart). P8.9 step 1's check (the
  standard planner on the P2.3 run): +10.5%, in band. Its refit of `kStdPostKeep` / `kStdPostKeepFloor` from `before
  post-proc` is open: the 6³ miss is the post-processing (the loop 26.45 GB; the run 35.11).
- P8.2, rescoped: the 1024³ miss is no per-partition CPU exact term (~96 B per hit cell per thread; a per-partition
  refit, +3.28 GB per partition in flight, would book a one-time event per partition and take face512 to +14.4%, or
  to +56.7% and mc 3). It fits the start-up zero fill: `psReserveGrids` zero-fills each scratch-backed grid in
  anonymous memory before mapping its file (`ZeroFillScope`), the largest (dispersion, 25.77 GB) on top of the
  particles (5.275) and the read blocks kept (7.536): 38.58 GB against 38.95 measured (−0.9%); the first merge's zero
  fills, in the same 88 s window, are not ruled out (open, below). In code (macOS with `--scratch-dir`; the scope is a
  no-op on Linux): `autoTuneLargestGridBytesPerCell`, `AutoTunePSPlanner::reserveZeroFillPeak`, predicted =
  max(loop + look-ahead, `reservePeak`), a report field and an AUTO-TUNE line when it sets the peak; the
  AUTO-TUNE-REPORT line unchanged; every 512³ prediction unchanged (the term ≤ 26.25 GB there). Its battery gate fails
  (`auto_tune_d` I1, Gates below). Open: reserve or first merge (a time-stamped footprint trace and
  `DTFE_AUTO_TUNE_DEBUG=1` on the next 1024³ run), and the 1024³ loop's own over-booking (32.38 GB against ~15–25 GB
  held). Alternative: remove the transient in `scratch_alloc`.
- P8.5, in code: `1 + 1.4(est − 1)`. Measured streams per point 4.25 / 5.38 / 2.30 (TNG100 z = 0, TNG50, TNG100
  z = 1; estimated 4.00 / 5.79 / 2.02); 1.4 books 5.20 / 7.71 / 2.43 (was 7.00 / 10.59 / 3.04), ≥ 5% above each; 1.3
  leaves 1% at z = 1. P8.5's own verify set adds a third z = 0 pair (2.06 estimated, 1.90 measured): 2.48, +31%. No
  rerun yet. The points were 53% / 67% of the sampled class's over-booking; the rest is the span terms (the snapshot
  + mc × the largest sub-grid, 1.7–1.9× what the runs held), which no item covers yet. P8.5's guard concerns CPU
  exact + points, not rerun: one `ps_exact` rerun settles it.
- P8.3 not needed: the 1024³ run parked one partition (2.03 GB) inside its 8.8 GB room. P8.4 no allowance: RSS is
  17.0 / 17.5 GB above the footprint (1024³ / face512), most likely the file-backed scratch pages (not verified;
  `std100_099_final`, without scratch, shows 9.8 GB): they are in RSS, not in the footprint the error is taken on.
- P8.10 not measured: the hand plan turns it off, and the three auto runs printed "dynamic concurrency not started:
  the free memory or the budget at the start holds no partition beyond it". The start check (`DTFE.cpp`) needs room
  for mc + 1 booked partitions (4.4 GB each for the TNG100 points run, 13.55 at 1024³) beside the reserve, in the live
  free memory and in the budget, from which the planner had already taken the largest mc it fits. It needs a start
  that passes: P8.10's TNG50 verify after P8.5, or a tighter span booking.
- Uncalibrated (no final-model run): PS CPU exact + points, GPU sampled without points, CPU `--ps-window`, one domain,
  PS GPU exact, TNG100 z = 1, any other 1024³ run.
- Gates: G-tuner 108 of 110 lines identical to `f9f6e2f`'s, the two '4M velgrad' point lines lower (6.31 → 6.18,
  6.82 → 6.69 GB: P8.5; the reserve term moves no line of the set). `--auto-tune-report` on the real snapshots: 1024³
  38.58 GB at the same 6³ mc 2; face512 unchanged (36.46 GB, mc 5); the TNG100 points run books 14.3 GB of points,
  not 18.2, and plans mc 4 at 38.85 GB, not mc 3 at 38.26. No rerun yet.
  - G-battery (32 stages, 2026-11-09 05:13–05:51): 29 of 32, then the three failing stages rerun alone on a quiet
    machine (06:42–06:59): 3 of 3. `auto_tune_d` I1 had failed on the reserve term: in the double build the 1700³
    test's largest scratch grid (~39.3 GB) puts the start-up peak (39.35 GB) over the test's 38.40 GB budget, so the
    report is over budget (6³ mc 4) where the test expected a split to fit. That is the term working (the production
    ps_exact1536 run of 2026-11-05 died on zero fills the planner did not see), so I1 now accepts an over-budget
    report when the start-up line is printed, and the AUTO-TUNE line says that no split lowers that peak. `ci_double`
    (the standard binary's tessellation-cache fingerprints, 8 sub-domains) and `ci_gui` (progress fragments in the
    log) had failed under load (a comparison job read 137 GB of grids during the battery) and pass unchanged.
  - G-linux (Colima, Ubuntu 24.04, 2026-11-09 06:59–07:35): build, core, double, 2d, variants, oldpy and ruff all
    passed on the tree with these changes.

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
  `psRunPartitionLoop`, line 1761) and the spill budget fixed that. The −45…−69% must be re-measured (P2.3) before the model
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
with / without points, one domain / partitioned / window. — *done 2026-11-09 for the classes phase R ran (the status above); the rest still uncalibrated.*
- Accept band: −5% to +15% of the peak footprint.
- Depends: P2.3. Effort: S.

**P8.2 A CPU exact term.** — *rescoped 2026-11-09 (phase R) and in code: no per-partition term; the 1024³ miss fits the start-up zero fill of the largest scratch grid; its battery gate fails (the status above).*
- Where: the non-Metal branch of `partitionBytes()`, booking the r3d per-thread scratch and the per-tet weight totals
  of the CPU exact path.
- Calibrate from the P2.3 1024³ rerun and a 512³ exact run without points.
- Verify: G-tuner (only exact lines change), G-real. Effort: M.

**P8.3 Only if P8.1 still shows it: parked and spilled partitions in the prediction.** — *not needed (2026-11-09): the 1024³ run parked one partition, 2.03 GB, inside its park room.*
- Book `min(park room, staged bound)` in `finishReport()`. The staged bound is the in-flight sub-grid bytes per waiting
  partition.
- Risk: lower concurrency, so slower runs. Effort: M.

**P8.4 Scratch-file pages.** — *no allowance (2026-11-09): the 17 GB of RSS above the footprint in the CPU exact runs are most likely file-backed scratch pages (not verified), outside the footprint.*
- With `--scratch-dir` the grid term leaves `fixed`, but pages the run dirties still count as resident.
- Add an allowance only if the post-`ZeroFillScope` rerun still shows the gap. Print both footprint and RSS (the
  summary already has both).
- Effort: S.

**P8.5 Tighten the point-stream budget** from `1 + 2(est − 1)` to about `1 + 1.3(est − 1)`. — *in code 2026-11-09 with 1.4 (the status above); no rerun yet.*
- Only after P8.2: otherwise the 1024³-class runs lose their hidden slack.
- Verify: the three z = 0 point sets above. Effort: S.

**P8.6 CPU `--ps-window` runs.**
- They are still under-predicted: 0.33 / 0.98 GB predicted vs 0.44 / 1.43 GB measured at 2048² / 4096², because
  their window buffers are per partition and the allocator keeps the freed window blocks.
- Effort: S–M.

- Measured 2026-11-07 on synthetic sets (64³ particles; a one-cell z window across 2048² and 4096²; production
  fields; CPU, auto-planned 2³): predicted vs measured peak footprint, pancake 13.3 / 11.6, 14.2 / 12.9 (exact),
  33.1 / 28.8, 35.3 / 34.7 GB; crossed waves 13.3 / 12.0, 14.2 / 13.7, **39.2 / 46.1** (4096² sampled: 18% under;
  its footprint 46 GB against 36 GB resident -- compressor pressure on the 64 GB machine), 35.3 / 34.7 GB. So the
  window model mostly holds on these sets; the under-prediction is the one case and the real 2048²/4096² planes
  (0.33 / 0.98 against 0.44 / 1.43 GB: small runs, retained window blocks). Left to phase R: the real planes give the
  constant; P8.10's dynamic gate does not help a window run (its gate books the same per-partition bytes).

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

**P8.10 Dynamic concurrency in the PS partition loop.** — *done in code 2026-11-07 (the owner: "continue with … the remaining plan items"); not yet measured: it did not start in phase R's runs (the P8 status).*
- Today the plan fixes `--max-concurrent` before the loop starts (`psConcurrency`, the loop's `num_threads`).
  `PartitionMemoryGate` reads the live free memory before each partition, but it can only hold a partition back below
  that number, never admit one more.
- The case (*measured*, the stopped TNG50-3-Dark z = 0 rerun, see the P2 status):
  - 6³ at a 38.7 GB auto budget got mc 1. The plan books all 25.7 GB of point stream records from the first
    partition (5.79 streams per point estimated, budgeted as 10.59).
  - The records grow as partitions merge: at 69 of 216 partitions the peak footprint was 26.5 GB against 38.4 GB
    predicted.
  - 163 s a partition, against 40 s for the logged mc 6 run. At mc 1 the deposit is serial: the GPU, then the
    GPU-deferred tetrahedra on the CPU, with only the next partition's triangulation (~10 s) beside it.
- Change:
  - The plan still chooses the split (it fixes the partitions' geometry) and the bytes one partition adds.
  - The loop starts `min(cores, partitions)` threads. The gate admits a partition while
    `free − headroom − reserve ≥ need`. `reserve` is what the run has still to allocate: the point stream records
    not yet written (their budgeted bytes × the share of partitions not merged), the finalized point outputs, the
    grids allocated after the loop.
  - The live readings exist: `autoTuneSystemFree` and the run's own footprint (`runProgressPeak`).
    `CorePool::holders()` already follows the live partitions.
  - The AUTO-TUNE line keeps the plan's mc as the floor; the SUMMARY prints the most partitions in flight.
- Outputs: unchanged. The ordered merge sums in index order (the TNG100-3-Dark z = 0 rerun at mc 8 was
  byte-identical to the logged mc 10 run).
- Does not fix:
  - the split: the standard planner's 6³ instead of 3³ is a split problem (P8.9);
  - the bytes one partition adds, still a prediction;
  - GPU memory, which the GPU hold serialises.
- Risk: a reserve that is too small squeezes the end of the run; the gate then runs one partition at a time (it
  delays, never deadlocks). macOS keeps freed blocks in the footprint, so the readings err on the safe side.
- Against P8.5: P8.5 fixes the same case at plan time and is smaller. This one also absorbs model error in
  everything already allocated, and other programs opening or closing during a run.
- The standard loop (serial partitions) could choose each partition's threads the same way. Not proposed before the
  P8.9 prediction is right.
- Verify: G-tuner (the plan lines do not change); bytecmp of a forced mc 1 run against a dynamic one; the
  TNG50-3-Dark z = 0 rerun at the auto budget: time per partition, the in-flight high-water mark, 0 swaps.
- Effort: M.
- Done (second version, after a review of the first: 4 opus readers and a skeptic per finding; 16 findings confirmed):
  - The gate (`auto_tune.h PartitionMemoryGate`): partitions enter strictly in index order (a waiting partition waits
    behind any lower-numbered one waiting), so every partition in flight is lower-numbered and the ordered mode's
    `blockers()` counts them all (the first version let a later waiter in with no memory check). Up to the plan's
    mc by the old rule; beyond it by a LEDGER: the free memory read when the loop started, less the headroom, the
    reserve and the run's own growth beside its partitions in flight (its footprint since the start less `need` per
    partition in flight: parked sub-grids, written records) must hold `need` for every partition in flight plus the
    new one, and that total stays within the budget; the live free memory must still hold one `need`. The first
    version checked the live readings for one partition only, which the partitions just admitted had not lowered
    yet: every waiting thread got in at once (10 against room for 2).
  - The reserve: `reserveEnd` (the derived grids, the point finalize's outputs, offsets and per-record locators),
    `reserveFirstMerge` (the grids the first merge sizes: the moment weights and face sums,
    `autoTuneMergeGrownBytes`) until a partition has finished, and the unmerged share of the budgeted point records.
    The start check leaves the records out (the gate books them as it goes; else the case P8.10 is for never starts).
  - Single build only: the double build and the float sums (`PS_GPU_FLOAT_MOMENTS/_MASS`, `PS_MASS_AUDIT`) sum
    per-thread copies in thread order, so their grids would follow the in-flight count; they keep the plan's mc and
    say why.
  - The teams: a separate counter `dtfePartitionsInFlight` (the gate's in-flight count) sizes the core pool's holders
    and `coresPerTeam` inside the loop, so the threads waiting in the gate do not cut every deposit team to one core;
    the parallel insert's share keeps the loop's thread count. The footprint is the physical footprint
    (`autoTuneOwnFootprint`, the budget's own measure), not the resident set.
  - Logs: "dynamic: at least the plan's m …", or "the plan's; dynamic concurrency not started: <why>"; after the loop
    "at most k partitions in flight"; the SUMMARY's prediction is restated with the partitions beyond the plan's.
  - Tests (`ps_auto_tune_check.sh` Y): a plan of 1 with memory for more runs 10 at once, the `--max-concurrent 1`
    grids byte for byte (Y1); no room for a second keeps 1 (Y2); the knob and a user's mc keep theirs (Y3); room for
    about four admits 2–4, not every thread (Y5: the first version's burst; measured 4); the double build keeps the
    plan's (Y6); killed after 3 and resumed: the same grids (Y4). The tuner's report lines do not change.
  - The strict order changed one test's premise: `ps_runtime_check.sh` R7 triggered the spill with no free memory,
    which worked only because the old gate let later partitions past a waiting one (several at once, unchecked) --
    against its own "running one partition at a time" line. R7 now takes no RAM parking room
    (`DTFE_TEST_PARK_ROOM_GB=0`) with free memory to spare: 25 partitions spilled to `--scratch-dir`, byte-identical.
  - Gates: three builds 0 warnings; tuner set identical; `ps_auto_tune_check` (single and double),
    `dtfe_auto_tune_check`, `ps_runtime_check`, `ps_fixed_merge_check`, `ps_scratch_check`, `dtfe_resume_check`,
    `ps_parallel_check`, `ps_slab_deposit_check`, `ps_partition_padding_check`, `dtfe_mc_average_check`, both face
    suites; bytecmp `--gpu-exact` 261 / 0.

---

## P9. Optimisations

**Status 2026-11-07.** P9.1–P9.3, P9.6 and P9.8 done; P9.5 tried and reverted; P9.4 not needed (P2.4); P9.7 and
P9.9 decided against. All byte-identical (bytecmp `--gpu-exact`, 261 files). The timings in the items below were
taken on a busy machine except P9.3's. The re-times on a quiet machine (old and new alternated, three rounds within
2%; base = the tree after wave 1) are in the change log:
2.1M particles at 256³, CPU 2×2×2 31.2 → 24.2 s and 3×3×3 27.9 → 24.7 s; GPU sampled kernels 6.3 → 3.9 s (density),
22.5 → 6.7 s (+ velocity, gradient), 17.6 → 11.3 s (+ velocity, dispersion); GPU exact kernels 16.2 → 12.8 s.
- P9.1: a per-axis row map (`SubgridRowMap`); rows split only at the periodic wrap or the window edge; up to 8
  pool threads. A 160³ sub-box into 256³ (84 B/cell): x7.5–9.5 with helpers, x2.7–3.7 on one thread. The opt-in
  `--fixed-point-merge` path still merges cell by cell. The non-one-to-one fallback (a sub-grid wider than the grid)
  is the old loop, and no test reaches it.
- P9.2: a dynamic team (`PSBlockClaim`, 256-cell blocks from one atomic counter) only in the single build without
  the copy mode, slabs, caustic cusps or `PS_MASS_AUDIT`; the others keep the static rule. One deviation: a
  block-start iterator index (8 B per 256 cells) instead of an 8 B/cell handle vector, so the tuner's model needs no
  new term. The pool also counts the static teams while they run (`CorePool::Hold`: copy/slab deposit, a pinned
  team, slab binning, the private-copies merge, `psFxBoundWalk`, the parallel insert's arena), found by the night's
  review; still uncounted: the point evaluation's team, standard DTFE's `_a` average team and the look-ahead thread.
  Follow-up: `psFxBoundWalk` could claim blocks too.
- P9.3: the MC average reuses the exact path's interleaved x-plane slabs; byte-identical at 1/4/8 threads in the
  single, double and 2D builds. It applies where one triangulation has the cores: a `--sample-points` run, 2×2×2
  partitions, 10 threads, quiet machine: `_a` interpolation 20.6 → 10.6 s. A run split into one sub-domain per thread is
  unchanged (4.7 s either way).
- P9.6: the plan, params, item bounds and the item map are built before the lock (`ctx()` via `std::call_once`).
  The item list became `itemStart[nTet+1]` + `blockTet[(nItems>>5)+2]` (37.3M items: 298 → 64 MB). A partition
  waiting for the GPU holds its item map and briefly its bounds, and no output grids.
- P9.5 was done, then reverted by the night's review: after the unlock the partition's whole device set (records,
  accumulators, item map) stayed allocated through the copy-back, so the next partition's set sat beside it -- on
  CUDA/HIP (no free-memory gate) an allocation failure and a CPU fallback at almost every hand-off of a big run, on
  Metal 262-553 MB per partition that the tuner does not book. The copy-back and releases took 4-57 ms a partition
  (2.1M particles, 256³, 2×2×2), at most 5% of a hold. Both backends now copy back and free before the unlock;
  `PS_GPU_TIMING` reports them in the hold again. The drop counters are added only after a successful copy-back.
- P9.7: uploads are 1–2% of a hold (0.26–0.35 s of 15–25 s); a prefetch needs a second mutex with a lock-order rule
  and another partition's device buffers (~0.5 GB at production scale) in the tuner's `snapshotBytes`.
- P9.8: without fLinear, a cell's constant fixed-point shares (mass, grad, sgrad, momw, sv, streams, the caustic
  OR) are added once as n·q after its sample loop. Kernel busy time (2.1M particles, 256³, loaded machine): density
  + velocity + gradient 22.5 → 9.7 s, density 4.6 → 3.9 s, density + velocity + dispersion 17.7 → 16.7 s.
- P9.9: specialising the pipelines recompiles the exact clip under fast math too, which can move low bits (the
  `--gpu-exact` byte gate could not be promised); after P9.8 the sampled kernel is bound by the classification.

Each item is byte-identical by construction unless it says otherwise. Every claimed gain is to be measured: CPU on the
synthetic 2.1M-particle box at 256³ and on TNG100-3-Dark z = 0 at 512³; GPU with one partition and
`PS_GPU_TIMING=1`.

### CPU

**P9.1 A parallel, row-wise sub-grid merge.** — *done 2026-11-07.*
- Where:
  - `quantities.cc:addFieldSubgrid` [300] and `orFieldSubgrid` [323]. They merge 17 summed fields and 2 OR-merged
    mask fields, in each of the u and a grids, under `mergeMutex`. For every cell they do NO_DIM 64-bit
    divisions/modulos plus a `subgridTarget` call.
  - `normalizePhaseSpace`.
  - The density-scale and stream-statistics loops in `ps_interpolation.cc`.

  None of these is threaded today.
- Change: copy row by row, with the destination row start computed once and a split only at a periodic wrap, then add
  `#pragma omp parallel for` over rows.
- Byte-identical: every element is still added once, in the same order per element.
- Keep the grow-to-zeros line (`if ( dst->size() < dstTotal ) ScratchAlloc::zeroResize(...)`, `quantities.cc`
  [307, 329]) on the merging thread, before the parallel loop. The block must be fully grown, and its zero-fill scope
  closed, before any thread writes into it.
- Gain: large on the merge itself *(est. ≥10×)*; on the wall, whatever P2.5 measures the merge share to be.
- Risk: low. Effort: S. Depends: P3.3.
- Verify: G-byte, `ps_periodic_seam_check`, `ps_partition_padding_check`, `ps_window_check`, `ps_scratch_check`.

**P9.2 Deposit threads that follow the free cores.** — *done 2026-11-07.*
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

**P9.3 Threaded Monte-Carlo average for standard DTFE.** — *done 2026-11-07.*
- Where: `averaged_interpolation_1.cc`, around line 952: `nT = exactAvg ? exactAverageThreads() : 1`.
- Change: compute each sample's x index first and skip samples outside the thread's slab before evaluating anything.
  Each cell then still accumulates in tetrahedron order.
- Secondary, since the production default is the exact average.
- Risk: medium. Effort: M.
- Verify: `dtfe_mc_average_check`, G-byte at 1, 4 and 8 threads.

**P9.4 (upstream, D5) The padding test's dummy points.** — *not needed (P2.4, D5) 2026-11-07.* Replace them by a per-cell circumsphere test, only if P2.4
shows a real cost. This changes the triangulation and the `.badGridPoints` sets, so it is a semantic change.

### GPU

**P9.5 Copy-back and releases after the lock.** — *tried and reverted 2026-11-07 (see the status above).*
- In both `psGpuDepositFields`, turn the `lock_guard` into a `unique_lock` and unlock after `psHostReportPasses`.
  The copy-back, the deferred scan and the buffer releases are private to the call.
- On CUDA, synchronise the stream first.
- Gain: 10–25 s of the 266 s *(est.)*.
- Risk: low. Effort: S. Verify: G-byte, G-gpu, `gpu_retry_check`.

**P9.6 Build the exact items before taking the lock, in compact form.** — *done 2026-11-07.*
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

**P9.7 Prefetch the next partition's uploads (decide from the P2.1 timers).** — *decided against 2026-11-07.*
- A second `stageMutex`: always take stage before ctx.
- It holds one more partition's device buffers (about 0.5 GB), to be added once to the tuner's `snapshotBytes`.
- Gain: up to about 60 s *(est.)*. Effort: M.

Together, P9.5–P9.7 *(est.)* save 150–190 s of the 266 s. That brings the 94M-particle GPU exact run from 17m39s to
about 15 min, near the CPU's 14m27s, so the CPU default for PS exact stays unless P2.3-style reruns show otherwise.

**P9.8 Fewer atomics in the sampled kernel.** — *done 2026-11-07.*
- The nSub³ = 27 samples of one tetrahedron that land in one cell share the same constant fixed-point increments:
  mass share, momentum weight, the gradient terms, the scalar gradient and the stream volume.
- Change: one atomic of n·q instead of n atomics of q. Integer adds are exact, so fixed-point output is
  byte-identical. Position-dependent grids keep their per-sample atomics.
- Gain: about 20–55% fewer atomics *(est.)*. This is the best lever for the sampled production deposit, which is
  kernel-bound (about 94% GPU busy).
- Check P2.2's atomic share first.
- Risk: medium. Effort: M. Verify: G-byte, `ps_fixed_moments_check`, `gpu_mass_check`, G-emu.

**P9.9 (Metal) Specialise the PS pipelines on the field flags** — *decided against 2026-11-07.* with function constants, as the DTFE host already does
(`FC_*`).
- Gain: occupancy, for runs that request fewer fields *(est. 5–15%)*.
- Risk: code generation may move low bits; verify with G-gpu. Effort: M.

---

## P10. Shared exact velocity derivatives (the face method)

**Status 2026-11-07.** P10.1–P10.7 done (CPU), with the drivers' `FACE_GRADIENT` knob; a launcher tick is optional (the reruns use the drivers).
- P10.1: `src/CGAL_triangulation/face_sweep.h` (sections with caller-given signed distances, so tetrahedra sharing a
  face lying in a grid plane decide it alike; the strip/cell clip). `tests/face_sweep_check.sh`: sections integrate
  to the tetrahedron's volume and first moments (3e-13), pieces add up to their section (3e-11), and Gauss on a Kuhn
  tessellation with a linear field, grid planes on and off the vertex levels (1e-12).
- P10.2: `--face-gradient` in the standard binary (`averaged_interpolation_1.cc:avg1FacePass`, after the deposit,
  replacing the gradient's cell integrals; `DTFE_FACE_CHECK=1` compares first). `tests/dtfe_face_gradient_check.sh`
  on the 64³ demo: equal to the per-tetrahedron gradient in every cell to 1.1e-5 of its rms (rms 1.2e-7); final
  divergence 4.4e-6, shear 9.0e-6; byte-identical at 1/3/10 deposit threads and after the Monte-Carlo, exact or
  GPU deposit; a 2³ partition split 1.8e-6 from one domain; periodic box mean 8e-10 of the rms. Cost: ~1 s on a
  6 s run (one thread). Both suites are in `ci_suite.sh`'s program stage.
- P10.3, changed from the first design: the per-cell FLUX sums gave the quotient form (dS − ⟨v⟩dN)/ΣV, which is not
  conservative in caustic cells (a periodic pancake's mean divergence +0.21 of its rms). Each cell now keeps its LOWER
  and UPPER face sums per axis (S and N: 24 values; int64 fixed point in the deposit, 192 B/cell of a sub-grid; 96 B/cell
  merged), and the gradient is (S_up/N_up − S_lo/N_lo)/dx: it telescopes (mean 5e-11 of the rms), equals the stream
  average where no caustic touches the cell (1.5e-6), and a one-cell `--ps-window` equals the full run bit for bit (the
  window cells hold their own upper faces). Fixed-point words outside `PSFxGrid` (the GPU's grids and FX_DROP slots
  10/11 untouched; the face sums use slots 12/13); bounds from the existing walk (now once per pass: `bounds()`); the
  grids `face_velocity` / `face_coverage` in `Quantities::forEachField`, so the merge, the checkpoint and the spill carry
  them. Refused without `--exact-average` or `--ps-volume-weighted`, with `--gpu`, with `--ps-halo-release`, in 2D.
  `tests/ps_face_gradient_check.sh` (12 checks, 9 s).
- P10.5: `auto_tune.cc` books the face sums: PS 30 Real per cell (S, N and, since P10.4, the momentum) in the merged grids (+ the velocity face mode forces)
  and their int64 words (autoTuneFxMomentBytes / autoTuneFxGrids, after the moment grids); standard DTFE the face pass's
  9 doubles per cell. Measured at 256³ (pancake / 64³ demo): PS one domain +31%, PS 2³ +87%, standard +43% (all on the
  safe side; P8 calibrates in phase R). Before the words were in the one-domain model PS was −18%. The 91-configuration
  tuner report is identical (no configuration uses the flag); bytecmp `--gpu-exact` vs `ref0-70e153e`: 261 files, 0 differ.
- P10.4: the mass flux divergence div(Σ ρ_s v_s) from per-face momentum sums (6 more int64 words per cell, FX_DROP
  slot 14; booked by the tuner with the other face sums), written as `.a_massFluxDiv` (dtfelib `mass_flux_divergence`).
  Checked by continuity on a drifted 3-stream pancake (double build, positions x + v t for t in [−T, T]): the cell mass
  change equals minus the time-integrated outflow, max 0.022 / rms 7e-4 of the rms with Simpson on 8 intervals,
  rms × 3.8 from 4 intervals (16: 0.005 / 1.6e-4), caustic cells included; cells three away from a fractional stream
  count agree by central differences to 3e-4 (O(δt²)). The time-derivative check converges slowly only where a caustic
  sheet crosses a face during the step. Periodic box mean telescopes to 0.
- P10.6: `FieldSet.face_gradient` (marker: `.a_velDivStreams`), `has/load(name, mean_field=True)` read the stored grids
  (both new grids take the velocity's √a scaling). `plot_DTFE.py`'s arrows are block means of quiver_step² cells at
  the block centres (`fields.block_mean_slice`; density-weighted for a mass-weighted PS grid, from the run log:
  `FieldSet.velocity_weighting`, the rule `convergence.velocity_weighting` already used, now one copy in `io.py`);
  `tests/py_convergence_test.py` checks it against `convergence.block_average(_velocity)`.
- Drivers: `FACE_GRADIENT=1` passes `--face-gradient` (both). `run_ps_dtfe.sh` refuses it, exit 2 before any
  snapshot, without `-e`, without `PS_VOLUME_WEIGHTED=1` or with the GPU deposit; `run_scripts_check.sh` (S) checks
  all five cases. README knob table, `demo/COMMANDS.md`. Default 0: phase R decides the production setting.
Background, measurements and the design review are in the session notes of
2026-11-07 (the velocity-method synthesis): standard DTFE's stored divergence/shear are the exact cell average of the
per-tetrahedron gradient; PS-DTFE's stored ones are the stream average ⟨∇v⟩ (the regular part of the gradient of the
volume-weighted mean velocity); dtfelib's FFT "mean field" adds the caustic sheets but smears them and rings
(TNG100-3-Dark z = 0 slice: correlation with standard 1.000 / 0.015 / 0.495 for standard stored / PS stored / PS FFT).

**Status 2026-11-09 (phase R).** P10.2's full-size check does not hold; the cause is the deposit's float32 region
frame (P10.8, new). D13 keeps the per-tetrahedron deposit.
- `std100_099_face` (TNG100-3-Dark z = 0, 512³, GPU exact average, auto 3³ × 10 threads; 485 s against 421 s
  without, +15%): of 243 sub-domain records covering all 134,217,728 cells, 133 fail the test's bound (max_rel ≤ 1e-4,
  none above): 64,978 cells (4.8e-4) above 1e-4 of the gradient rms, worst 1.21e-2; 80 degenerate cells.
- Cause: GPU `tetVerts = float(double(point) − double(corner))`, CPU `basePosition` (`Real`). Where a sub-domain's
  corner c lies in a finer binade than X = x − c, X rounds by up to 1.9e-6 Mpc (8.8e-6 of a cell) and slivers deform:
  99.7% of the cells above lie in x planes 284–312, the 3³ middle slab (c = 29.6256 Mpc) where X ≥ 32, growing with
  density (above 1e-4: 0.14% of the strip's cells below the mean density, 58% of those at ρ/ρ̄ ≥ 30). The face pass
  forms the frame in double: the exact side. Outside the strip the two agree to ~2e-7 rms (the 64³ demo's level),
  with a sparse tail (174 cells, worst 4.2e-3; attributed to float32, not tested).
- `std100_099_face_cpu256` (CPU exact average, 256³, 2³ × 10, an exact frame): 38 of 16.8 M cells above 1e-4, worst
  2.23e-4, rms ≤ 7e-7: the face pass is clean (no degenerate cell in this run, so the degenerate handling is not
  exercised), the CPU float build has the same tail; it does not exercise the strip.
- `std100_099_split6` (the same binary at 6³): its deposit's gradient differs from the 3³ face pass by 2.5e-5 rms in
  every x slab (3.7 M cells above 1e-4, 2.8%, max 7.6e-3). The binade rule predicts no 6³ strip in x, so this haze
  needed the 6³ run with the face check, `std100_099_split6_face` (2026-11-09 06:22–06:42, 1195 s, exit 0; its
  density, velocity and T-web grids byte-identical to `std100_099_split6`'s):
  - the check nearly passes at 6³: 903 of 134.2 M cells above 1e-4 (6.7e-6; 246 of the 1944 records carry 1–9 each),
    worst 1.9e-3, worst record rms 2.9e-6, 0 degenerate; no strip (against the global gradient rms: 194 cells, worst
    6.7e-4, in a few clusters of a handful of cells). So the strip is the 3³ plan's frame, as predicted, and what is
    left is the float32 tail cpu256 also shows. Under the restated bound below the 6³ run passes (0 records over 3e-6,
    pooled 6.7e-6 ≤ 1e-5) and the 3³ run fails (39 records over, pooled 4.8e-4).
  - the haze is not the deposit's: the 6³ face pass differs from the 3³ face pass exactly as the 6³ deposit does
    (3.72 M cells above 1e-4, 2.8%, rms 2.7e-5, max 7.6e-3; the per-x-plane profiles coincide), smoothly in x, y and z
    with no structure at the partition or sub-domain faces, and already in the first x slab, whose corner is 0 in
    both plans. The face pass forms its geometry in double, so the split enters through the one input the two passes
    share: the per-tetrahedron float gradient (`velocityGrad`, `field_computation.h`; the face pass evaluates the
    velocity on a face as the base value plus that gradient times the offset), `Real` from the `Real` inverse of the
    vertex-difference matrix. Its inputs (vertex differences, velocities) are exact and split-independent, but its
    vertex labelling (the base = CGAL's vertex 0) follows the triangulation's construction history, so the float
    rounding of the inverse and of the product differs between splits, by the matrix's condition number: ~1e-7 for a
    well-shaped tetrahedron, 1e-4 and more for a flat one. That is the leading candidate, untested; it fits the rms,
    the maximum, the smoothness, the thread-layout dependence (1.7e-5 between the 8- and 10-thread 6³ runs, which
    also route 6% of the tetrahedra differently between GPU and CPU) and PS-DTFE's immunity (fixed lattice labelling,
    fixed-point sums). The test is an hour: form the inverse and the product in double and round the gradient once
    (labelling-independent up to the last rounding), then compare the 3³ and 6³ face passes again; expected: the
    2.7e-5 collapses to the float sum order's ~1e-7. P10.8's second part.
- Restated bound, proposed (the demo's G1 keeps 1e-4): per record `rms_rel ≤ 3e-6`; pooled `above_1e-4 / compared ≤
  1e-5`; every cell above 1e-4 listed; degenerate cells reported apart; no per-record maximum. The clean records pass
  (worst record rms 2.6e-6, pooled fraction above 1e-4 3.4e-6 on the GPU; 7.0e-7 and 2.3e-6 on cpu256); this run fails
  (39 records over 3e-6, pooled 4.8e-4). The battery never runs FACE-CHECK on the GPU deposit or on such a frame.

**Status 2026-11-09, afternoon: P10.8 done** (the owner's go at 11:10; code, the demo gates and bytecmp; the 512³
reruns below; gates on the final tree: the battery 32 of 32 at 13:52, the Linux stages 7 of 7 at 13:47, after a
first battery had shown a stale `libDTFE.dylib` -- rebuilt, the header grew -- and a G6 bound too tight for the
double build's float32 GPU deposit). Uncommitted in `~/GPU` with the tuner batch.
- Step 0 (`std100_099_split6_face_t3`, 11:15–11:34): the 6³ face run with the face pass on 3 threads instead of 1
  (`DTFE_DEPOSIT_THREADS=3`; the same [1,3,3] thread sub-boxes, so the same tessellations) is byte-identical to
  `std100_099_split6_face` in `a_den`, `a_vel`, `a_velGrad`, `a_velDiv`, `a_velShear`, with the same 903 cells above
  1e-4: the face pass is thread-invariant, the box-wide source sat in the tessellation's inputs. (The premise above
  was inexact: the face pass walks one tessellation per THREAD sub-domain, 9 per partition at mc 10, hence 1944
  records; the thread count only deals out the cells.)
- A second candidate, refuted before coding: each sub-domain forms its grid corner and spacing in float
  (`subpartition.h` `region + dx·index`, `boxLength/nGrid`), so the grid planes of two plans sit at different absolute
  positions. Computed for the 3³ and 6³ x slabs (`frame_offsets.py`, session scratch): at most 1.6e-6 Mpc (7e-6 of a
  cell), zero in the first 71 planes and in planes 137–216, correlation −0.36 with the measured per-plane rms of the
  face-pass difference (the zero-offset planes carry the largest differences). The grid frame below removes it too.
- The fix, exact-average paths only (the Monte-Carlo paths keep their bytes, bytecmp below): every deposit — the CPU's
  r3d / ps_clip2d clipping, the GPU kernel, the face pass — forms a tetrahedron's geometry relative to the corner of
  the FULL grid's cell holding its base vertex, in double (`GridFrame`, `TetFrame`; `User_options::gridFrame` from
  `fullGridSize`, `fullGridOffset`, `fullGridBox`, recorded at the first split), names cells by global index minus the
  sub-grid's origin, and takes the hashed vertex order the Monte-Carlo average uses (`DTFE_MC_LEGACY=1` keeps CGAL's,
  the A/B hook); the CPU exact path and the face pass form the inverse and the gradients in double
  (`matrixInverseDouble`, `*GradDouble`) and narrow each share once; the face pass decides a plane's side on absolute
  double positions against `F.plane(g)`, one expression of the global plane index. The GPU gets a fourth per-tetrahedron
  buffer, the base cell (int32×3, Metal buffer 22); the sampled kernels add it to a sample's cell (0 for Monte Carlo),
  the exact kernel forms local cell offsets. The exact average's prefilter box is the region widened by 1e-6 of its
  length; its cell volume the frame's.
- Demo (z0_64, 64³): `dtfe_face_gradient_check.sh` PASS with the new G6 (GPU exact, `--partition 3 3 3` against one
  domain: density and divergence byte-identical, shear 6.6e-13 of its rms; 27 FACE-CHECK records, worst rms_rel
  2.1e-7, no cell above 1e-4); G4's CPU 2³-against-one-domain face difference is now 0 (was 1.8e-6).
  `dtfe_metal_check.sh` ALL PASS. Why the demo's shear is not byte-identical: its flow is v = 1000 x, the diagonal
  gradient ~1000 against off-diagonals ~1–10, and the GPU gradient's fixed-point unit is one per batch, set by the
  largest shares, so the small components are quantised at ~4e-6 of their own value (6e-13 of the field rms); the face
  pass's off-diagonals flip one float ulp in 17% of the cells for the same cancellation in double. The density, the
  velocity and the diagonal gradient are byte-identical.
- bytecmp against the tree before the change (rebuilt 11:31): 184 files, 21 differ, all of them the three CPU
  exact-average runs (single 8 files, double 8, 2D 5); every Monte-Carlo, GPU-sampled and PS-DTFE file identical. The
  CPU exact grids moved by 6e-8 of the rms (single; max 6.4e-6 in a cell), 5e-8 (2D), 1.8e-16 (double).
- 512³ (TNG100-3-Dark z = 0, GPU exact average, `--face-gradient`, `DTFE_FACE_CHECK=1`, 12:04–12:30; the scripts and
  logs in the session notes, `p108-reruns.sh`):
  - `std100_099_face_p108`, the auto 3³ × 10 plan (456 s against 485 s before; max RSS 42.3 GB): 243 records, no
    degenerate cell (the old run had 80), **294 cells above 1e-4 (pooled 2.2e-6; was 64,978 = 4.8e-4), worst 5.7e-4
    (was 1.21e-2), worst record rms 5.4e-7 (was 3.48e-5), no record over 3e-6 (was 39)**: the restated bound is met
    on the production plan, the strip is gone. `std100_099_split6_face_p108` (6³, 1098 s, 37.5 GB): 1944 records,
    648 above (4.8e-6; was 903), worst 1.06e-3 (was 1.93e-3), worst record rms 1.7e-6 (was 2.9e-6): met.
  - Across the two splits (`compare512.py`, session scratch): density differs in 1886 of 134 M cells (1.4e-5), the
    velocity in 12,308 of 403 M values (3.1e-5), the gradient in 493 of 1.2 G, the divergence in 5 cells, the shear in
    69 values, every one by a single float ulp (rms 6e-12 of the density rms, 1e-10 of the velocity's; maximum one
    ulp); the T-web and V-web classifications are byte-identical. Before: 98% of the cells, rms 2.5e-5, up to 7% in a
    cell. What is left is the GPU's fixed-point unit, one per batch (a different split rounds the shares to a
    different unit, so the last bit flips where a cell sum sits at a rounding boundary), and the order of the vertex
    density sums (double, so rare flips).
  - Against the old grids the new 3³ run differs in 98% of the cells by 2.0e-5 of the density rms (max 4.7% in one
    cell), 1.3e-6 of the velocity's, 2.2e-5 of the gradient's: the old split-to-split scatter, now removed, not a new
    bias: the box-mean density is 0.9999999652 on both new plans (the old 3³: 0.9999999781, the old 6³: 1.0000000297),
    within 3.5e-8 of 1 as every exact run's. Peak memory 42.3 GB max RSS on the 3³ plan against 41.8 GB before (the
    base-cell buffer and the frame), 37.5 GB on the 6³ plan as before.
- Open after this: the GPU gradient's fixed-point unit per batch (a split can still move a gradient grid by that unit;
  a run-wide unit would need a bound before the first batch), the CPU exact path's float sums in walk order (~1e-7
  across splits; fixed-point sums as the GPU's would end it), and byte identity across splits presumes identical
  Delaunay tetrahedra, which co-spherical lattice snapshots (000) may resolve differently per split.

**Goal.** One implementation for both estimators of the velocity gradient of the cell-averaged velocity field, exact
to the interpolant, with no spectral ringing:
- By Gauss, the cell average of ∇v is (1/V) times the surface integral of v n over the cell's faces. The binaries
  deposit **exact face integrals** (the same clipping idea as the exact cell average, on face planes) and take
  `g_ij = (F_i(+j face) − F_i(−j face)) / (A Δx_j)` after the merge.
- **Standard DTFE:** the interpolant is continuous, so this equals today's exact gradient (the check, below).
- **PS-DTFE:** with S = Σ_streams v_s and n = the stream count, the face sums S_F and A_F (covered area) give the
  gradient of the volume-weighted mean velocity in the quotient form `(ΔS_F − a_vel ΔA_F) / Σ_s V_s`, exact where n is
  constant through the cell and exact when integrated over a region elsewhere (the caustic sheets enter as flux
  differences, at grid resolution, with no ringing). The stored regular part ⟨∇v⟩ stays available, so the difference
  is the caustic contribution.
- **Mass flux (PS):** p = Σ_s ρ_s v_s gives `div p` exactly (ratio-free): the continuity equation's quantity, e.g. the
  rate at which a void loses mass. The mass-weighted (Favre) mean velocity = p / ρ comes with it.

**P10.1 The face sweep (shared geometry).** — new `src/CGAL_triangulation/face_sweep.h`
- The section of a tetrahedron by an axis-aligned plane (a triangle or a quadrilateral), clipped to the face squares
  of the grid on that plane (convex-polygon clip, generalising `ps_clip2d.h`'s `clipTriangleToCell`), with order-0 and
  order-1 moments of the linear field on each piece (order 2 for ρ v with a linear ρ, if P10.4 needs it).
- One tie rule for vertices and edges lying exactly on a grid plane, so each face piece is counted once.
- Verify: a unit test against brute-force sampling and the analytic section of a random tetrahedron; the pieces of a
  section sum to its area to rounding.

**P10.2 Standard DTFE: the face gradient.** — *the full-size check of 2026-11-09 04:00 was not met, on the deposit's side (the P10 status, P10.8); met after P10.8 the same afternoon (294 cells above 1e-4 of 134 M, worst record rms 5.4e-7); the default since 14:40 that day (D13).*
- Behind `--face-gradient` at first; the gradient and its derived fields (divergence, shear, vorticity, V-web) come
  from the face sums. Each sub-grid holds N+1 face layers per axis; a face on a partition or thread seam is owned once
  (the merge adds face layers like cells).
- Verify: on the battery's standard runs (CPU exact average), the face gradient equals the stored one within the split
  tolerance (~3e-6), except cells touching a zero-inverse (degenerate) tetrahedron, which are listed; the periodic box
  mean of the divergence is 0 to rounding; thread and partition splits agree.
- If the check holds everywhere it should, the face path can replace the per-tet gradient deposit later (D13).

**P10.3 PS-DTFE: face sums, mean-flow gradient, regular part.** — the exact deposit branch only (`--exact-average`)
- Accumulate S_F (3 components) and A_F per face orientation in the fixed-point grids (`fxAddL`, new `PSFixedBounds`
  entries); normalise and difference after the ordered merge.
- Outputs: the mean-flow gradient's divergence/shear/vorticity under the standard names (the quantity standard DTFE
  stores), the regular part under new names, both from one run. Refused with the sampled deposit and with
  `--ps-halo-release`.
- `--ps-window`: N+1 face layers for the window only.

**P10.4 PS-DTFE: mass flux.** P_F = Σ ∫ ρ_s v·n dA and M_F = Σ ∫ ρ_s dA per face; `div p` as a new grid, and the
cell momentum Σ ρ_s v_s (so p / ρ gives the mass-weighted mean velocity without a second run). ρ_s per tetrahedron:
its mass over its Eulerian volume (uniform), as the deposit already uses.

**P10.5 Memory model and tuner.** Face accumulators: standard 3 × 3 floats per cell (as the 9-component gradient); PS
3 × (3 + 1) [+ 3 × (3 + 1) for P10.4] int64 words per cell, about 96–192 B/cell: whole box at 512³ or a window. Book
them in `autoTuneBytesPerCell` and the GPU-free PS plan.

**P10.6 Python.** `FieldSet` reads the face-based grids; `load(name, mean_field=True)` returns them when present and
falls back to the FFT for older outputs; `plot_DTFE.py`'s quiver block-averages (8×8 cells) instead of taking every
8th cell; comparisons of the two estimators mask multi-stream cells.

**P10.7 Tests.**
- Standard: face gradient == stored gradient (P10.2).
- PS: in cells whose stream count is an integer throughout (streams within 1e-5 of an integer and all six face
  coverages n·Δx²) the face gradient equals the stored regular part; the periodic box mean of the mean-flow divergence
  is 0 to rounding.
- A 1D pancake (Zel'dovich, `generate_ps_test_data.py`) across a fold: the face ∂ū_x/∂x against the analytic mean flow
  of the lattice's piecewise-linear map, to rounding; a modulated variant where the stream count varies on a face.
- Mass flux: drift the particles by ±v δt, rerun, and check (M₊ − M₋)/(2δt) = −V div p to O(δt²).
- `--ps-window` face grids equal the full run's to float rounding.

Effort: L. GPU ports (Metal `sw_poly_rect` already intersects z-sections with rectangles) only after the CPU path is
settled; the CPU exact PS deposit is the faster one at production sizes.

**P10.8 The deposit's region frame.** — *new 2026-11-09 (phase R), decided (D16) and done the same day: the grid frame, the hashed vertex order and the double gradients (the P10 status of 2026-11-09 afternoon); changes the exact-average output bytes by rounding. P10.8 keeps the record below as written when it was planned.*
- Where: standard DTFE's per-tetrahedron deposit, GPU and CPU exact average: the host's vertex upload
  (`averaged_interpolation_1.cc`, `tetVerts`), the CPU's `basePosition` (`checkCellPosition`, `Real`) and the cell
  boxes the kernel forms in that frame (`dtfe_deposit.metal`). PS-DTFE's deposits: check for the same frame.
- What: a per-tetrahedron local origin (e.g. its base grid corner) formed in double on the host; the vertices
  relative to it narrowed to float for the GPU; the CPU's `basePosition` in that frame (or in double).
- What, second part: the per-tetrahedron gradients (`positionMatrix`, `densityGrad`, `velocityGrad` in
  `field_computation.h`, and the `vertexMatrix` inverse the exact-average paths take) formed in double and rounded
  once, or from a canonical labelling, so that the result does not follow CGAL's vertex order; effort S, tested by
  the 3³-against-6³ face-pass comparison (the P10 status). Step 0, no code: the face pass walks one tessellation per
  partition and splits only cell ownership among threads, so a 6³ face run with another thread count (20 min, by
  hand) must give an `a_velGrad` byte-identical to `std100_099_split6_face`'s if the labelling is the whole box-wide
  source; a ~1e-5 difference would mean something thread-level is in the face pass too. *Done 2026-11-09 11:15–11:34:
  byte-identical (the status above; the premise is "one tessellation per thread sub-domain").*
- Why: the face check's strip (the P10 status: the 6³ face-check run of 2026-11-09 ties the strip to the frame and
  clears the 6³ plan); the box-wide ~1e-5 split dependence of every standard exact-average grid (the P2 status) is
  not the frame but, as the leading candidate, the per-tetrahedron gradient's float labelling dependence: the second
  part.
- Risk: every standard exact-average grid may move: most in the rounding strips (the gradient there by up to
  ~1e-2 of its rms in single cells), elsewhere by an unmeasured amount; bytecmp's `ref0` standard files are re-taken
  after it.
- Verify: FACE-CHECK on a `std100_099_face` rerun (8 min) under the restated bound; the 3³-against-6³ face-pass
  comparison (its 2.7e-5 rms down to the float sum order's ~1e-7); bytecmp (only the standard exact-average files differ); G-gpu; the battery, plus a
  face-check case on such a split, on the GPU deposit too.
- Optional, in the tuner: a screen from the plan alone. A sub-domain with corner c > 0 rounds on an axis when its
  largest region-frame coordinate (width + padding) reaches 2^(⌊log₂ c⌋ + 1); `--auto-tune-report` could flag it.
  The night's 3³ plan hits it in x (c = 29.63, X up to 41.7 > 32); the 2³ cpu256 plan does not; at 6³ only low
  y/z thread corners might (6.1, X up to 9.8 > 8).
- Effort: M.

---

## P11. Which device for which phase, chosen by the program (added 2026-11-09, the owner's request)

**Owner's rule (13:30):** keep every configuration -- the sampled and the exact deposit, the face gradient and the
points, for both estimators -- and let the program use, phase by phase, whichever of CPU, GPU or both is fastest
on this machine. **Then (13:50):** turn CGAL's TBB insertion on for all binaries and widen the CPU-GPU overlap so
the CPU triangulates while the GPU deposits. Session notes: `p11-placement.md`, `p11-bench.sh`, `p11-tbb.sh`.

**The phases and where they run today.** Input, partition plan, padding, Delaunay triangulation (CGAL), vertex
densities: CPU, and the triangulation cannot move (no Metal Delaunay exists; the CUDA ones are approximate with a
CPU repair, and Metal has no double for the predicates). The sampled or exact deposit: CPU threads or the GPU, by
`--gpu`. The face gradient: a CPU pass for standard DTFE with any deposit; for PS-DTFE inside the CPU exact deposit,
or (D17, since 2026-11-09 16:00) as a CPU walk beside the GPU exact deposit. Points: CPU threads (exact predicates). The look-ahead triangulates partition k+1 while the GPU deposits
k, standard and PS, when the memory budget holds two partitions (the production 3³ standard run had it OFF: 24.4 +
18.0 GB against 30.7 GB left); within a partition the OpenMP team of thread sub-boxes already overlaps, each thread
triangulating its sub-box while another's batch is on the GPU. PS-DTFE's thin tetrahedra go from the GPU to the
CPU in the same pass. Merge, derived fields, the T-web FFT and the output: CPU.

**P11.1 The measurements** (TNG100-3-Dark z = 0, 512³, production fields, one run at a time, ~3–4 h): the nine
configurations of `p11-bench.sh` -- standard exact on CPU with and without the face pass, standard exact GPU with
face and points, standard sampled on both devices, PS exact GPU with points (never run at size), PS exact CPU with
face and points (the campaign configuration), PS sampled CPU, and the campaign configuration with the TBB insert --
plus `p11-tbb.sh`: the standard GPU exact run with the plain and the TBB (6 threads) binary, both with
`DTFE_GPU_TIMING=1` (GPU busy seconds per batch). Wall, user, peak memory; phase costs by differences. The bench runs
on the P11 code, after P11.2–P11.5.

**P11.1 results (2026-11-09 17:03–20:46).** The bench grew to 14 runs, on the binaries of the tree as committed that
evening, one run at a time on an idle machine; the logs' readings are in the session note p11-placement.md, section 6.

Standard DTFE (10 threads; the face gradient is the default):

| run | configuration | plan | wall | CPU (user) | peak memory |
|---|---|---|---|---|---|
| a | GPU exact (the production setting) | 3³ | 439 s | 2352 s | 34.1 GB |
| a2 | a with `--parallel-triangulation` | 3³ | 441 s | 2361 s | 35.2 GB |
| e | GPU sampled | 3³ | 433 s | 2368 s | 34.4 GB |
| f | CPU sampled | 2³ | 315 s | 1825 s | 38.6 GB |
| b | CPU exact, `--per-tet-gradient` | 2³ | 380 s | 2123 s | 38.1 GB |
| c | CPU exact | 2³ | 431 s | 2569 s | 38.5 GB |
| d | a with the 4.2 M points (`--pts-vel-grad`) | 3³ | 632 s | 3254 s (system 250 s) | 41.6 GB |

- At this size the run is CPU-bound whichever device deposits: a, e and c agree within 2%. The exact average costs
  nothing over the sampled one on the GPU, and the GPU exact run equals the CPU exact one at 4 GB less.
- The fastest standard run is the CPU sampled one (315 s). The tuner plans 3³ for the GPU (it books the device's
  records inside the budget) and 2³ for the CPU, and 27 partitions cost more padding and triangulation than the
  device saves on the deposit. Whether the GPU wins at the CPU's plan is not measured: two runs pinned to
  `--partition 2 2 2` (`p11-extra.sh`, ~44 GB expected) are left for an idle machine.
- The face pass costs about 50 s (12%) on the CPU exact run (c against b).
- a2 equals a in time and in bytes: at the auto plan the insert's thread budget is one core per tessellation, so the
  parallel insert does not engage even when forced.
- Full-size byte gate: a equals the P10.8 reference run in density, velocity and velocity gradient. The gradient
  skip, the face default and the insertion rule moved no byte of the production configuration.
- The points add 190 s, with 250 s of system time. Looked at on 2026-11-10 (P11.7 item 4): on a run a sixth of this
  size the system time is the parallel insert's lock waits, and both it and the added wall belong to the shape a
  points run takes (one tessellation per partition), not to the point files' I/O or the evaluation.

PS-DTFE (all with the 4.2 M points and `--pts-vel-grad`, `--ps-vertex-mass --ps-volume-weighted --ps-caustics`; the
tuner's 6³ plan, budget 41.2 GB):

| run | configuration | in flight | wall | CPU (user) | max RSS | peak footprint |
|---|---|---|---|---|---|---|
| k | sampled CPU (`--subsamples 3`) | 6 | 1644 s | 14829 s | 36.5 GB | 29.5 GB |
| k2 | sampled GPU | 4 | 1260 s | 4246 s | 36.8 GB | 28.6 GB |
| h | exact CPU | 6 | 1133 s | 9637 s | 37.5 GB | 29.6 GB |
| i | exact GPU | 4 | 1156 s | 2451 s | 36.1 GB | 29.9 GB |
| l | exact GPU + `--face-gradient` (D17) | 4 | 1454 s | 4947 s | 49.6 GB | 33.4 GB |
| j | exact CPU + `--face-gradient` | 3 | 1737 s | 14198 s | 46.4 GB | 30.4 GB |
| m | j with `--sequential-triangulation` | 3 | 1756 s | 14203 s | 46.3 GB | 29.4 GB |

(Both memory columns are `/usr/bin/time -l`'s, from the run logs; the footprint column was added on 2026-11-10. The
max RSS also counts the resident pages of the `--scratch-dir` files, which hold the full grids and which the plan
leaves out of RAM: the tuner's prediction and budget are to be read against the peak footprint.)

- Without the face gradient the exact deposit is the fastest PS option on both devices: h and i agree within 2%, the
  GPU at a quarter of the CPU seconds, and both beat the sampled deposits (GPU 1260 s, CPU 1644 s).
- The face gradient is expensive in work, not only in concurrency: +47% CPU seconds on the CPU path (h → j) and +102%
  on the GPU path (i → l), for +604 s and +298 s of wall. On the CPU path the face words also halve the concurrency.
- **No run exceeded the tuner's budget (corrected 2026-11-10).** This bullet set the face runs' 46.4 and 49.6 GB
  against the 41.2 GB budget, but those are max RSS; their peak footprints were 30.4 and 33.4 GB. Nor did the default
  CPU path book the face words at half their size: it booked them in full, and the half-size booking was
  `DTFE_GPU_FLOAT_MOMENTS=1`'s. What the plan did lack under `--face-gradient` is a GPU lane's state after its hold of
  the device, 0.2 and 0.5 GB at one lane on 64³-particle test runs (P11.7 item 1, done). j ran at the concurrency the
  corrected plan chooses (3); l ran with 4 in flight where the corrected plan chooses 3, so its wall is that of 4
  lanes.
- The parallel insert engaged in no PS run: a 6³ partition holds 771,057 padded points, below the 1e6 threshold.
  j equals m byte for byte, but that says nothing about the insert.
- Phase sums of l over its 216 passes: GPU deposit 1690 s, extraction 373 s, the CPU phase (the bound walk and the
  face walk) 1147 s. The per-pass deposit times sum to more than the wall, so passes wait for the device: unlike the
  standard run (GPU busy 9–13%) the PS exact GPU path looks device-bound (a reading of the phase sums, not timed
  with `DTFE_GPU_TIMING`).

**P11.2 Per-phase timing in both binaries.** Done 2026-11-09 14:00: `src/phase_totals.h` collects every
`printComputationTime` (the sub-domain threads' included) and `dtfeFinishFields` prints "phase totals (wall seconds
summed over all threads and sub-domains)" at the end; the tuner's input for P11.4. (Demo 2³ GPU exact: triangulation
31.4 s, GPU deposit 19.9 s, dummy points 9.5 s, face pass 2.6 s.)

**P11.3 The TBB insertion.** Done 2026-11-09 14:00 on the build side: `TBB_STD=1` compiles it into the standard
binaries (off by default; `o/` stamped `.tbb_`), `psParallelTriangulation` is a member in both builds, and the
standard sub-domain team declares its concurrency (`ConcurrentTriangulationPlan`), so the insert divides the cores
as in the PS loops; the gate in `triangulation.cpp` lets both binaries insert in parallel (it was hard-wired to the
sequential insert in the standard build). `--parallel-triangulation` and `--sequential-triangulation` exist in both
binaries; the DEFAULT is the parallel insert for PS-DTFE (its near-lattice points build 2× faster, measured; its
fixed-point grids do not depend on the insertion order; the vertex densities and the points can differ at ~1e-15
between runs) and the sequential one for standard DTFE, because the only measurement of the parallel insert on its
clustered Eulerian points is the demo's 4.5× slowdown. What `p11-tbb.sh` (14:05–14:13) measured: the TBB_STD
binary's Parallel_tag data structure without Fast_location, with the insert still SEQUENTIAL (the gate), on the auto
3³ plan: 435 s / 2343 user-s against 456 s / 2454 user-s for the plain run of the morning (which had the face
check on), and density and velocity grids **byte-identical** to that run's (0 of 134 M / 403 M values differ). The
parallel insert's gain at TNG scale is measured by runs n/o of the bench (`--max-concurrent 1`: one tessellation
per partition on 6 threads, plain against TBB, where the look-ahead also fits in memory) and run m for PS-DTFE;
the standard default follows them. **16:30, the default rule (both binaries):** the parallel insert for a
tessellation of at least `DTFE_TRIANGULATION_MIN_POINTS` points (1e6) where more than one core is free for it;
`--parallel-triangulation` forces it at any size, `--sequential-triangulation` turns it off (the two exclude each
other). Why a threshold: the mc1 experiment measured 3.1× on the triangulation at TNG scale (section 5 of the note),
the demo's 0.26M clustered points 4.5× slower, and the test suites' inputs stay sequential by construction. Found on
the way: (1) the "PS parallel default" of 14:00 never took effect -- `po::bool_switch(&member)` stores false when the
flag is absent, so the constructor's `true` was overwritten (bytecmp A at 16:03 was identical to the reference for
every PS run because of it); the flag now sets `parallelTriangulationForced`, the rule lives in
`psParallelTriangulation`. (2) bytecmp B (`DTFE_TRIANGULATION_THREADS=6`, 89 of 184 files differ) shows what the
insertion order moves: the standard CPU deposits' float sums (dtp: 68% of the values, ≤1.9e-4 of the rms), a sample's
cell in the PS sampled deposit (ps: 1 cell in 130k by 1e-3), PS-DTFE's velocity moments (psx a_vel: 5% of the values,
≤3.7e-7 of the rms; their shares are formed in each tetrahedron's vertex-0 frame), the GPU sampled grids at 2.6e-8;
unchanged: the PS fixed-point density (psx a_den byte-identical) and the GPU exact average (mc1 vs mc1 + parallel
insert: a_den and a_vel byte-identical at 512³, the face gradient's last bit in 2.2e-7 of the values). The option
help, the header and triangulation.cpp's comment no longer claim "PS-DTFE's grids do not depend on the order". (3)
The tuner's per-vertex constant for the insert's copies was 210 B (a 0.26M-point estimate); measured +4.4 GB on ~5.7M
padded vertices → 800 B, applied in `finishReport` only where the insert will run (≥ 2 threads per partition AND the
partition above the threshold, or forced); the per-vertex plan constant no longer carries it. **Run-to-run byte
reproducibility of the CPU deposits and the PS velocity moments is therefore no longer the default above the
threshold (D19, for the owner: keep the rule, or make the sequential insert the default and the parallel one the
driver's choice for low-concurrency runs).** Found on the way earlier: the plain "timing" run took 616 s because the Colima VM of
the Linux check was still resident, the tuner saw 37.9 GB free instead of 52 and planned 4³ instead of 3³ (the P8
status). Where the exact deposit's grids no longer depend on the construction order (P10.8:
the GPU, the PS fixed-point sums) the insert costs no reproducibility; the CPU float sums and the points still move
at rounding, so the tests that compare runs byte for byte pass the sequential insert explicitly. `--parallel-triangulation`
and a `--sequential-triangulation` opt-out in both binaries.

**P11.4 The device rule (D18).** A GPU build with a device present deposits on the GPU where P11.1 says it is faster,
the CPU otherwise. The mechanism exists in the drivers: `DTFE_GPU` empty = `auto` (`gpu_auto_resolve` in
scripts/config.sh: the GPU when the binary has one, and for PS-DTFE the CPU when DTFE_EXACT_AVERAGE is on), `DTFE_GPU=0/1`
override, the binaries' `--gpu` unchanged (the suites' semantics hold). What P11.1 adds is the rule's table (the
standard exact average too), which the bench of the final binaries fills in.

**P11.5 The PS face gradient beside the GPU exact deposit (D17).** Done 2026-11-09 16:00 (code; gate: the rebuilt
`tests/ps_face_gradient_check.sh` P11): `psFaceWalk`, one CPU walk over the kept tetrahedra after the GPU deposit
(the deposit loop's filter, mass and gradient; the deferred tetrahedra included, the loop's own face block off under a
GPU deposit), into the face words `psCpuFixedUnits` now allocates under either deposit; the bound walk runs once more
for their units, so a GPU and a CPU run of one pass form the same face words (byte-identical mean-flow divergence).
Sequential with the deposit for now (the GPU is idle during the walk; concurrency with the next sub-domain's deposit
is P11.6's pipeline). Since 2026-11-10 (P11.7 items 2 and 5) the single build takes those units from the bounds the
extraction summed, without the second bound walk, and the walk rejects on a copy of the loop's own span test.

**Status 2026-11-09 16:30.** P11.2, P11.3 (the default rule: the parallel insert for tessellations of ≥ 1e6 points
where cores are free, both binaries; D19 for the owner) and P11.5 (D17) done; P11.4's mechanism in the drivers, its
rule pending P11.1 (the bench of tonight, `p11-bench.sh`, 14 runs on the final binaries); P11.6 dropped on the mc1
experiment (section 5 of the note: one partition in flight loses to the thread sub-boxes by 33% even with the
parallel insert; the run is CPU-bound, the GPU busy 9–13%). After the battery on the D13 tree (29/33 at 15:17, the four red stages understood) three fixes:
the opt-in shared GPU grid (`DTFE_GPU_SHARED=1`) added the device's gradient on top of the face pass's (doubled; no
driver sets the knob, so no production or rerun grid carries it); the deposits now skip the per-tetrahedron gradient
under the face default (9 of 13 components; `DTFE_FACE_CHECK=1` keeps it); the test pins (`--per-tet-gradient`,
`DTFE_FACE_GRADIENT=0`, the help check) and a new shared-grid face case in `dtfe_metal_check.sh`. The four suites
are green on the rebuilt tree. **The full battery on the final tree (D17, the insert rule, the fixes): 33 of 33 at
17:03 on 2026-11-09, this time with the CI stages run to their end (p11b's had stopped at the help check). The tree
is ready to commit; `commit-batches-2026-11-09.md` in the session notes says how the three piled batches split.**

**Status 2026-11-09 21:00.** The tree of 17:03 was committed and pushed (branch GPU) the same evening. The bench
(P11.1) ran on its binaries and is recorded above; P11.7 lists what it leaves. D15 and D18 wait for the tuner fix
and the reruns of j and l; D19 is the owner's (the rule engaged in neither production configuration at 512³).

**Status 2026-11-10.** P11.7 item 1 is done: the plan books a GPU lane's state after its hold of the device, and the
face sums on their own. The bench's face runs had not exceeded the budget (their max RSS had been read as the peak;
the correction is under the PS table above), so D15 and D18 do not wait for a tuner fix. Run j stands as measured. Run
l ran with 4 in flight, at a peak footprint of 33.4 GB, where the corrected plan chooses 3: whether to run it again at
3, or first to trim the lanes' common over-booking so that the plan allows 4, is the owner's.

**Status 2026-11-10, evening.** P11.7 items 2, 4, 5 and 6 are done (below; the changes sit on `c8dd4fb`): the face
units from the extraction's bounds, the face walk on the loop's own test, the points' system time traced to the
parallel insert's lock waits (no code change), the speed texts. P11.7 leaves item 3: the two pinned runs for the
device rule's table (D18).

**P11.6 The sub-domain pipeline (the widened overlap).** **Dropped 2026-11-09 16:00 on the cheap experiment's
numbers** (`p11-mc1.sh`; p11-placement.md section 5). One tessellation in flight with the look-ahead (`--partition 3
3 3 --max-concurrent 1`, standard GPU exact + face, TNG100 z = 0, 512³): 811 s wall, 581 s with the parallel insert
(triangulation 675 → 218 s summed), against the auto plan's 435 s (3³ × 9 thread sub-boxes). The look-ahead does
overlap (the phase sums are 1193 / 755 s), but the run is CPU-bound -- the GPU is busy 9–13% of the wall;
triangulation, the face pass (227 s summed) and the extraction are 90% of the work -- and the OpenMP team of
sub-boxes already runs those phases 9-wide while batches go to the GPU. A builder/interpolator pipeline would
rebuild that team by hand. What stays: the parallel insert's thread budget (cores / planned concurrency) gives the
3× on the triangulation exactly where it matters, the memory-bound runs with few tessellations in flight (1024³,
small `--max-concurrent`), at +4.4 GB per in-flight partition here (the tuner's term), and nothing at the auto plan's
concurrency.

**P11.7 What the bench leaves (recorded 2026-11-09 21:00; items 1, 2, 4, 5 and 6 done 2026-11-10, item 3 open).**
1. **DONE 2026-11-10.** The PS tuner's face words (memory safety, first). As recorded on 2026-11-09: the term missing
   on the GPU path under `--face-gradient` (2·3·(3+2) int64 words = 240 B per sub-grid cell), and the same words
   booked as `Real` on the CPU path. As found: the default CPU path booked the words in full (in the single build 120
   B of `Real` slots and 120 B of word excess); the half-size booking was `DTFE_GPU_FLOAT_MOMENTS=1`'s. The GPU path's
   gap was real: with the face sums a lane's largest state comes after its hold of the device (the words the deposit
   handed back, the face words made then, the `Real` grids of their conversion, less the freed words the allocator
   returns), 514 against the hold's 388 B per sub-grid cell for the production selection, 902 against 388 in the
   double build. A lane now books the larger (`gpuAfterHoldCellBytes`, `afterHoldReuse`). With it: the face sums are
   booked on their own (`autoTuneFaceOn`, `autoTuneFaceBytes`; the double build and `DTFE_GPU_FLOAT_MOMENTS=1` lacked
   the conversion's overlap and counted the sums in the deposit threads' private copies), the cell mean velocity an
   `_a` face pass implies has its words (24 B per cell; a face pass of the cell-centre gradient alone deposits no
   velocity and loses the grid the plan gave it), the one-domain peak walks the face conversion in every build, and
   the mass flux's grid is out of the private copy. Measured on 64³ particles at grid 256 (2×2×2, two snapshots;
   old plan / new plan / peak footprint, GB): GPU with 1 in flight 5.89 / 6.26 / 6.06 and 6.87 / 7.45 / 7.37; with 4
   in flight 9.67 / 11.16 / 9.30 and 12.60 / 14.93 / 10.68; with 8 in flight 14.82 / 17.79 / 10.55 and 20.35 / 25.01
   / 13.05; one
   domain, CPU 14.10 / 13.50 / 8.95 and GPU 15.89 / 15.89 / 11.50. The new plan covers every measurement. Beyond one
   lane it is far above them, as the old one was: every lane is booked at the largest partition's sub-grid and at its
   own peak at once. So on the bench's command run l now plans 3 in flight (38.36 GB predicted; 4 would book 44.0 GB
   of the 41.23) where it ran 4 at a peak footprint of 33.4 GB; h, i and j keep their plans. The review of the change
   (the same day) found four things, corrected with it: the implied velocity is the `_a` pass's only (above; the
   1024³ tune line of a cell-centre gradient on a GPU lane goes 209.56 → 201.05 GB); a GPU lane left out the grids of
   a selected scalar and of its gradient (the production selection with `scalar_a scalarGradient_a`: 452 against 388
   B per sub-grid cell in the hold, 538 against 506 after it; without the face sums the 1024³ tune line goes 160.92 →
   162.62 GB); where the moments are float sums (the double build, `DTFE_GPU_FLOAT_MOMENTS=1`) the exact deposit's
   multiplicity is one more `Real` grid in each deposit thread's private copy (double build, density alone, grid 256,
   8 in flight: 7.87 GB planned where 5.20 were); and run_mode_check's M7b took the standard binary's team under
   gentle for G sub-domains where it is the largest 2^a·3^b ≤ G. Gates: ps_auto_tune_check (new section Z) and
   run_mode_check in both precisions, `tests/dev/tune_report_set.sh` (124 lines: the 110 from before, 11 for the face
   plans, 3 for the review's cases; against the binaries of `9bfa39a` 13 move -- 8 of the face lines, the 3 review
   lines and the double build's two "dens exact" lines that plan the private copies -- and none changes its partition
   or its count in flight), ps_face_gradient_check in both precisions, the 2D suite, the Mac battery (33 of 33
   stages) and the Linux check (every stage). Committed as `c8dd4fb`. Open, the owner's: runs j and l again;
   trimming the lanes' common over-booking, which would give l its fourth lane back.
2. **DONE 2026-11-10.** The face walk's second bound walk on the GPU path. The bounds the GPU extraction sums (the 10
   moment bounds and the three face bounds, in the bound walk's block order) stay in the pass's fixed-point state
   (`PSFxState::gpuBounds`, `gpuBoundsValid`) and stand in for `psFxBoundWalk` when `psCpuFixedUnits` forms the face
   units after a GPU deposit (`psFxBoundsFromGpu`). The walk remains where the two sums are not the same doubles --
   the double build, whose extraction works on float copies, and a run under the NaN test hook -- and where the
   extraction summed no bounds (`DTFE_GPU_FLOAT_MOMENTS=1` without a window). `DTFE_DEBUG_FX_BOUNDS=1` walks anyway
   and compares the 13 doubles byte for byte (a line on stderr; a difference stops the run); P12 of
   `tests/ps_face_gradient_check.sh` requires its "0 differ" in three single-build runs (a window, a non-periodic box,
   the lattice box). The time the walk took is not measured, so no saving is recorded.
3. The two standard runs pinned to 2³ (the GPU at the CPU's plan), for the device rule's table (P11.4, D18).
4. **DONE 2026-11-10 (measured; no code change).** The points' 250 s of system time. It is the parallel insert's lock
   waits, not the point files' I/O and not the evaluation. Measured on a sixth of the bench's particles (256³
   synthetic particles, grid 256, `--partition 2 2 2`, standard GPU exact with the face gradient, 2048² points with
   `--pts-vel-grad`; one run per line). A points run changes the run's shape: each partition is triangulated once,
   whole, so the partitions run one at a time on the parallel insert (6 threads here) with the look-ahead, where a run
   without points splits each partition into thread sub-boxes. Wall / system time: without points 58 / 8 s; with
   points 98 / 30 s (1.2 M involuntary context switches against 0.1 M); with points and `--sequential-triangulation`
   104 / 5 s; without points at `--max-concurrent 1`, the same shape, 92 / 30 s (sequential: 109 / 5 s). So the system
   time and the 40 s of wall belong to the shape: the evaluation itself is 3 s, summed over the 8 partitions. A stack
   sample of the points run (every 10 ms) puts the system calls in the insert's lock waits: 2342 samples in the yield
   call `swtch_pri`, 2273 of them from CGAL's parallel insert (its lock grid's `try_lock_cell_impl` and the insert's
   own retry), against 39 in `write`. By the insert's thread count (`DTFE_TRIANGULATION_THREADS`; wall / system /
   triangulation summed): 2: 96 / 6 / 61 s; 3: 102 / 8 / 40 s; 4: 93 / 8 / 31 s; 6, the default here: 95 / 11 / 28 s;
   8: 104 / 43 / 40 s. The look-ahead is not the source and earns its place: off (`DTFE_OVERLAP=0`) 124 and 118 s with
   23 and 17 s of system time, on 94 and 94 s with 23 and 26 s. Over these runs the default's system time varied
   between 11 and 30 s. No code change: the sequential insert removes the system time and costs wall, and no thread
   count beats the default's wall by more than the spread between runs. The full-size picture fits (P11.6's
   experiment: one tessellation in flight with the parallel insert 581 s against the auto plan's 435 s; run d 632 s),
   but the lock waits' share there is not measured. Open: the thread series on TNG100 at 512³. Candidates if it
   confirms: a lower thread count for this shape (4 has the default's wall here at 8 s of system time) and the
   resolution of CGAL's lock grid.
5. **DONE 2026-11-10.** The face walk's filter against the deposit loop's. The review (readers and a judge over the
   two code paths) found the walk and the loop apart in three cases, and a gap in a test hook. (a) A tetrahedron
   wholly outside a non-periodic grid: the loop skips it (`outsideGrid`); the walk, which had no such test without a
   window, deposited the section of a tetrahedron with a face in the grid's upper boundary plane (three particles in
   that plane, as on a lattice; in the single build also one within a rounding below it) into the last layer's upper
   face. (b) Under `--grid-window` in the single build the loop's span is a float quotient and the walk's
   (`psWindowTouched`) a double one; the float quotient can round a vertex just below a grid plane onto it, the loop
   then skips a tetrahedron the walk kept, and its section went into the window's upper face. (c) The loop's face call
   sat under its clipped-volume test, so a tetrahedron without clipped volume in any cell left no face in a CPU run
   and its sections under the walk. (d) `DTFE_TEST_NAN_SHARE` did not reach the walk: a GPU run dropped no face share
   where the CPU run dropped 66926. Fixed: the walk rejects on a copy of the loop's test (`psLoopSpanReaches`: the
   `Real` span, the outside-grid test, the window's span; a KEEP IN STEP note on both), while `psWindowTouched` stays
   for the extraction and the bound walk, where a superset is harmless (the bounds only set a unit); the loop deposits
   the faces ahead of its volume test; the hook is in the walk. The loop's move is the one change a CPU run can show:
   a tetrahedron whose clipped volume is lost in every cell now leaves its face sections, as it did beside a GPU
   deposit (none in the 180 files of the comparison below). Measured on the binaries of `c8dd4fb` (a 32³ lattice
   without jitter in a `--box` whose upper x face is a particle plane, grid 32): the GPU run's `.a_massFluxDiv`
   differed from the CPU run's in 1024 values, one layer, by up to 0.37 of the rms (`.a_velDiv` in 1024 values by
   3.1e-7, `.a_velShear` in 2048 by 6.3e-7). After: the mass flux divergence byte-identical, `.a_velDiv` and
   `.a_velShear` too below the last x layer, and within 3.3e-8 of the rms in it (1.7e-8 in the double build). That
   residue is not the face sums': neither device deposits the outside tetrahedra's section, so the layer's upper face
   has no stream and its x gradient stays the stream average, each device's own float sum. Depositing that section on
   both devices instead (the tetrahedron's face in the boundary plane is the grid's own face) would close it and is
   not done: it changes CPU bytes, so it is the owner's to ask for. Test: P12 of `tests/ps_face_gradient_check.sh` (a
   GPU build; a one-cell window, a non-periodic box and the NaN hook byte for byte, the lattice box as above). Case
   (b) has no constructed input: P12's window run covers the path, not the rounding.
6. **DONE 2026-11-10.** The stale speed claims. `gpu_auto_resolve`'s note had already gone with the run modes (P13:
   the GPU for PS-DTFE's exact deposit in every mode). Corrected now: PS-DTFE's `--gpu` help (it called the deposit
   "the dominant cost"; it gives the bench's 1156 s on the GPU against 1133 s on the CPU for the exact deposit and
   1260 against 1644 s for the sampled one), two comments in the launcher (`python/gui/app.py`) and the device table's
   comment in `scripts/config.sh` (the launcher's one difference, `GPU_PAYS_FROM`). What still reads "faster on the
   CPU" is standard DTFE's sampled average in performance (315 against 433 s), which is current.

**Items 2, 4, 5 and 6: checks (2026-11-10).** Against binaries built from a clean export of `c8dd4fb`:
`tests/dev/bytecmp.sh --gpu-exact` 261 files, 0 differ; `--face-gradient` runs, which bytecmp does not make (CPU:
periodic in one domain and in 2³ partitions, a one-cell window, a non-periodic box, the lattice box, crossed waves;
GPU: periodic in one domain and in 2³, crossed waves; both precisions), 180 files, 0 differ, so the moved face block
and the bounds hand-over move no byte there; `tests/dev/tune_report_set.sh` 124 lines identical. Green on the tree:
ps_face_gradient_check (23 s; 25 s double), ps_fixed_moments_check and ps_window_check in both precisions,
run_scripts_check, docs_drift_check, py_gui_runspec_test. The review of the batch (three readers, a judge) confirmed
no defect; it leaves one cosmetic difference: a NaN vertex mass (NaN particle weights) is counted as left out by the
loop and zeroed by the walk, the same words under a different count in the log. Then, on the tree as it stands: the
Mac battery, 33 of 33 stages (19:24 to 20:14, the tree unchanged over it), and CI's Linux stages in Colima (build,
core, double, 2d, variants, oldpy, ruff), all passed (20:14 to 20:52). Only this sentence was written after them. Item
3 stays open (two full-size runs, on the owner's go).

**Gates:** bytecmp (placement moves no byte), the GPU suites, the battery, Linux; P11.1 repeated with the rule on.

## P12. `--exact` / `--sampled`, exact by default (added 2026-11-09 evening, the owner's decision D20)

**Owner's decision (after the P11.1 bench):** "exact is the right strategy across the board, but keep sampled as an
option (both as flags `--exact` and `--sampled`)". The bench's numbers (TNG100-3-Dark z = 0, 512³, M1 Max, `2f428bc`):
standard DTFE exact 439 s against sampled 433 s on the GPU, 431 s against 315 s on the CPU; PS-DTFE exact 1133 s
(CPU) and 1156 s (GPU) against sampled 1260 s (GPU) and 1644 s (CPU). Session notes: `exact-sampled-contract.md`
(the rules every file follows; it wins over the inventory) and `inv-flags-plan.md` (the inventory and its open
decisions D-A to D-I, settled in the contract).

**P12.1 The rules (both binaries, every build).**
1. `--exact` (canonical) and `--sampled` (new). `--exact-average` (both binaries) and `--ps-exact-deposit` (PS-DTFE)
   are hidden deprecated aliases of `--exact` (`deprecatedOptions`; removal per D9). The canonical command tokens,
   the run log and the checkpoint key spell `--exact`.
2. Resolution, first match wins (`resolveAverageMode`, in `readOptions` once the field set is final): both flags →
   error; `--sampled`; `--exact`; the pin `DTFE_EXACT_AVERAGE` (`Env::tristate`: off = sampled, on = exact, unset or
   `auto` = none); an explicit sampling request → sampled (a non-defaulted `--subsamples`; standard DTFE also
   `--samples` and a non-defaulted `--average-method` of any value); otherwise exact where it applies (standard: the
   DTFE method, no `velocityStd_a`, no `--redshift-cone`, not `MY_SCALAR`; PS: the PS-DTFE method and at least one
   `_a` field), else sampled with the cause in the header (non-DTFE methods and PS runs without an `_a` field:
   silently, as before).
3. An explicit `--exact` keeps every error `--exact-average` raised, renamed; a count beside it is accepted and
   ignored; with `--redshift-cone` it warns and samples. The default never turns a command line that ran before into
   an error.
4. The results land in `exactAverage` / `psExactDeposit` (constructor defaults stay false for library callers);
   `updateEntries`' user sampling and redshift cone clear `exactAverage` too.
5. Header lines `volume averaging       : EXACT cell-tetrahedron integration ...` / `SAMPLED (<source>)` (standard,
   with an `_a` field and the DTFE method) and `grid deposit           : EXACT tetrahedron-cell intersection moments
   ...` / `SAMPLED on nSub^3 sub-points (<source>)` (PS). The checkpoint key gains `average=exact|sampled` and
   `gradient=face|per-tet` before `partition=` (old checkpoints no longer match, intended; it closes D13's pin gap).
6. Drivers: exact by default in both; with `DTFE_EXACT_AVERAGE` unset a set `SUBSAMPLES` selects sampled; they always
   pass `--exact` or `--sampled`, `--subsamples` only when sampled and set (run_ps_dtfe.sh's `SUBSAMPLES` default
   empty); `-e`, `--exact`, `--sampled` (the drivers' `--exact-average` warns for one release); `run_settings` maps the
   old spellings to `--exact` and drops `--sampled`, so under the exact default the sampled PS grid sets count as
   stale (intended); `gpu_auto_resolve` unchanged (P11.7 item 6 has its wording).
7. Tests: the suites outside `tests/dev` spell `--exact`; a run that means the sampled average and gives no count
   gets `--sampled`; a file-level `DTFE_EXACT_AVERAGE=0` only in `ps_auto_tune_check.sh`, `tests/dev/bytecmp.sh` and
   `tests/dev/tune_report_set.sh` (the dev tools keep the old spellings: they run the pre-change reference binaries);
   the help probes key on `--sampled`; new `flag_spelling_check.sh` cases (alias = `--exact`, the refused pair, flag
   against pin both ways, a count alone, the default's header line). Python and the launcher pass the mode
   explicitly, read a log with neither marker as sampled, and tick Exact by default (a saved choice is kept).

**What moves.** Unflagged runs with an `_a` field now average exactly; in PS-DTFE the exact deposit then drives every
grid pass of such a run (`.streams` included), adds `.tetTouch`, and the tuner books its terms (plans may change).
Runs that pass `--exact-average`, a sampling count, or no `_a` field keep their bytes by design; G-byte
and G-tuner below are what checks it.

**P12.2 Beside it (2026-11-09 evening): where a 512³ run's time goes, and what changed in the code for it.** The owner
asked why the CPU and the GPU times are so close. Session note `inv-findings-hybrid.md` has the measurements. In
short: each PS-DTFE run is bound by one device (with `--gpu` the one GPU queue is held for 91% of the bench run's wall
while about 2 cores are busy; on the CPU the exact deposit fills the cores beside an idle GPU), and standard DTFE is
bound by the triangulation on both devices. Three changes to PS-DTFE came out of it, all in this tree:
1. The periodic selection skips an image pass whose shifted Lagrangian extent misses the partition's padded box
   (`subpartition.h`; `lagExtent` of `PSPartitionPlan` in `DTFE.cpp`): 343 scans of every particle at `6 6 6` (512
   with the fallback seam at the box's upper face) of the 5832 before. Byte-identical.
2. The core lend (`CoreLend` in `ps_metal_host.cc`): a partition's thread that waits for the GPU or holds it lends its
   core to `CorePool` for that time, so the other partitions' CPU work can claim one helper more. A claim on the pool
   never changes a result.
3. The experiment `DTFE_PS_GPU_SHARE=<per mille>` (marked EXPERIMENT in `DTFE.cpp`; in the checkpoint key): that share
   of a `--gpu` run's partitions deposits on the GPU, the others on the CPU at the same time, chosen by partition
   index alone. Measured on the bench's input with `--partition 6 6 6 --max-concurrent 6` given by hand, with the
   prototype's binary, whose exact deposit put 4M items in a GPU buffer instead of this batch's 500k (those buffers
   alone, every partition on the GPU: 1025 s): 712 s at a share of 0.6 (753 s at 0.5, 747 s at 0.7) against the
   bench's 1133 s (CPU) and 1156 s (GPU); a mixed run at 500k is not measured. The grids differ from
   the pure runs' at float rounding, the mass stays exact. Unset = every partition on the GPU, as before. P13 made it
   `--gpu-share` and removed the name.
Checked against `2f428bc`'s binary: pure CPU and pure GPU runs of the exact deposit at `--partition 1`, `2 2 2`,
`3 3 3` and `4 4 4` on a small test snapshot, 15 files each, 0 differ; TNG100-3-Dark z = 0 at 512³ with `--gpu`, 23
of 23 files byte-identical to the bench's. The gates below ran on the tree with these changes.

**Status 2026-11-10: IMPLEMENTED, BUILT, GATES GREEN, COMMITTED as `39583f0`** (code, drivers, tests, Python and docs
edited in parallel, one file set per agent, by the contract's ownership table; then built, reviewed and the review's
findings fixed).

**Gates run 2026-11-09 23:44 to 2026-11-10 01:40, all passed:**
1. G-build: a clean rebuild of the six binaries, no warning; the Metal ones carry their kernels (`strings | grep -c`:
   `depositExactAverage` 5 in both DTFE builds, `depositFields` 15 in both PS-DTFE builds).
2. G-battery: 31 of 33 stages in the first run. The two failures (`ci_core`, `ci_double`) were test flakes older than
   this item, both fixed: the tessellation-cache test aid printed its fingerprint line in two writes, so another
   thread's progress figure could land inside it (now one `write()`), and the cone case of
   `tests/flag_spelling_check.sh` ran without `--seed`, whose default is the clock (now `--seed 7`). The two stages
   then passed in full (1397 and 1152 PASS lines, no FAIL; 300 stress runs per precision with no displaced fingerprint
   line). The other 31 stages were not repeated on this tree after the two fixes; the full battery passed 33 of 33 on
   2026-11-10 08:10 on the tree that adds the run modes to this one.
3. G-byte and G-gpu: `tests/dev/bytecmp.sh --gpu-exact` (the base set and the GPU's exact deposits) against the
   `2f428bc` reference binaries: 261 output files compared, 0 differ.
4. G-tuner: `tests/dev/tune_report_set.sh` against the `2f428bc` reference: 110 lines, 0 differ.
5. G-docs: `tests/docs_drift_check.py` 36 passed, 0 failed; `tests/run_scripts_check.sh` and
   `tests/flag_spelling_check.sh` passed in the battery's `ci_core` and `ci_double`.
6. G-gui: the battery's `ci_gui`, `gui_explore`, `gui_app2` and `runspec` stages passed.
7. G-linux: CI's Linux stages in Colima (build, core, double, 2d, variants, oldpy, ruff) all passed. The `variants`
   stage fails on the pushed `2f428bc` and passes with this tree's `demo/examples/Makefile` TBB block, so `master`
   should not be moved to `2f428bc` before that block is committed.

## P13. Run modes: gentle / balanced / performance (added 2026-11-10, the owner's request)

**Owner's request (2026-11-09, 23:25):** "maybe a good idea to make an option in the GUI as well as a flag to have a
gentle mode, balanced mode, performance mode". It came the evening exact became the default (P12), after the question
which method leaves the laptop usable for simple tasks during a run, and with the wish for an implementation that uses
the CPU and the GPU together. A design panel (three designs, two challenges, one judge) settled it on 2026-11-10.
Session notes: `run-mode-contract.md` (the final contract: the decisions, the interfaces and texts every file follows,
the test plan and the file ownership; it wins over the panel's answers) and `inv-findings-hybrid.md` (the
measurements). D1-D29 below are that contract's numbers, not the owner's table at the end of this file.

**P13.1 The decisions, in short.**
1. D1, the defaults: the binaries default to performance (every core, 0.25 s GPU buffers, no priority change,
   `--gpu-share 1`), the drivers and the launcher to balanced; the drivers' `DTFE_RUN_MODE=auto` is performance inside
   a SLURM or PBS job (`SLURM_JOB_ID`, `PBS_JOBID`); the drivers and the launcher always pass the resolved
   `--run-mode`, as they pass `--exact` or `--sampled`.
2. D2, CPU + GPU: `--gpu` puts every partition on the GPU in every mode; `--gpu-share auto` is the run mode's share
   (D11 in performance, 1 in gentle and balanced), and the drivers and the launcher pass it on every PS exact run on
   the GPU, so a bare `./PS-DTFE --gpu` stays byte-identical to `2f428bc`.
3. D3, the spelling: one valued option, `--run-mode gentle|balanced|performance` (exact lower case), in both binaries
   and every build; no `--gentle` / `--balanced` / `--performance` switches. `DTFE_RUN_MODE` is read by both binaries
   and the three drivers (any case; unset, empty or `auto` = not given; any other word an error); the drivers take
   `--run-mode MODE` (getopts letter R).
4. D4, the mechanism: the process-wide state in the new `src/run_mode.h` (`runModeState()`) with three helpers,
   `dtfeOmpMax`, `dtfeHardwareThreads` and `dtfeAdoptCoreBudget`, that return today's value unless the mode lowered
   the budget; they replace `omp_get_max_threads()` and `hardware_concurrency()` where a thread team is sized;
   `applyRunMode` calls `omp_set_num_threads` once before any thread exists, and the look-ahead thread adopts the
   budget first thing.
5. D5, the formula: N0 = the OpenMP default at start-up; base = min(N0, the performance cores) on Apple silicon, else
   N0; gentle max(1, base/2), balanced max(1, base - clamp(round(base/4), 1, 4)), performance N0 (4 / 6 / 10 on an
   M1 Max, 32 / 60 / 64 on 64 cores); no SMT detection.
6. D6, an explicit lever beats the mode: a non-empty `OMP_NUM_THREADS` keeps its count (logged 'kept'), the
   `DTFE_*_THREADS` pins keep their teams, a `--max-concurrent k` above the budget raises it to k (logged 'raised';
   not under a kept `OMP_NUM_THREADS`), `--max-concurrent 0` is the mode's budget; `--partition`, `DTFE_GPU` and
   `--gpu-share` stay as given.
7. D7, the tuner: no recalibration; `autoTuneCores()` returns the budget, so both tuners plan for it.
8. D8, the priority: gentle on Linux only, nice 10 before any thread exists, never lowered, the outcome in the header
   line; on macOS no mode sets a priority (no taskpolicy, background band, QoS class or I/O policy).
9. D9, the GPU pace: 0.05 s buffers in gentle and balanced, 0.25 s in performance, the 8 ms gap in every mode;
   `GpuPace` gains its base (`target0`, `gap0`), so an escalated pace recovers to the mode's base and not to 0.25 s;
   the eased line keeps `(the default)` at 0.25 s / 8 ms and says `(the run mode's)` at a lower base; the first-chunk
   references stay at 0.25 s.
10. D10, PS-DTFE's exact item cap: 4M items per GPU buffer in performance, 500k (the `2f428bc` value) in gentle and
    balanced; byte-neutral, with no tuner term; standard DTFE keeps its own cap.
11. D11, the share: partition pi deposits on the CPU when floor((pi+1) s / 1000) = floor(pi s / 1000), s the resolved
    per mille; `auto` gives 1000 outside performance, in the double build, under the float-sum variants or
    `DTFE_MASS_AUDIT`, below `DTFE_AUTO_MINN` particles and under 4 threads, else 50 round(20 / (1 + 0.09 (N0 - 1.8)))
    within [500, 1000] (600 at 10 threads; `DTFE_GPU_SHARE_RHO` replaces the 0.09).
12. D12, where it applies: a PS-DTFE 3D GPU build, `--gpu`, the exact deposit and a partitioned run; elsewhere the
    share is 1000, a number gets one note naming why, `auto` is silent, and a plan that ends in one domain is reset
    after the report.
13. D13, the mixed run's memory model: per lane the common terms plus the larger of the CPU deposit and the GPU's
    pre-lock arrays, and the GPU snapshot and the lock holder's grids once; the existing GPU and CPU branches
    untouched, so a share of 1 plans as before; to be validated on the bench input before it is trusted (predicted
    32.5-41.2 GB). **Validated 2026-11-10 (the plan only, no run): predicted 34.68 GB at `--partition 6 6 6
    --max-concurrent 3` on that night's 37.4 GB budget (snapshot term 5.07 GB, lane 3.33 GB); by the same terms the
    full 41.2 GB budget holds 4 at once (38.0 GB), not the 6 of the 712 s hand plan (44.7 GB by the model, 30.7 GB
    measured).** The dynamic concurrency (P8.10) is what can add the lanes the model does not book, and under a mixed
    plan it could not start: its start check and its ledger booked the once part with every partition (8.4 GB each).
    **Fixed 2026-11-10:** `AutoTuneReport::mixedLanes` marks a mixed split, `psOnceBytes` carries its snapshot term,
    and the start check, `PartitionMemoryGate::roomForOneMore` and the summary's prediction book it once and the lane
    per partition in flight; with no once part (every all-GPU and CPU plan) the arithmetic is the earlier one. The
    live check keeps the whole partition, the basic gate and the parking rule are untouched. `DTFE_AUTO_TUNE_DEBUG`
    prints the start check's terms; `tests/run_mode_check.sh` G8 holds the start check at both sides of the budget,
    the ledger's second partition and the summary's prediction. Left as it was: the all-GPU ledger books its snapshot
    per lane (D29's over-booking, the same separate commit). **Reviewed lightly 2026-11-10: correct and safe to
    keep**; what the review left for later commits: (a) the lane's max(CPU deposit, GPU pre-lock arrays) covers the
    host grids a GPU lane keeps after the lock only by argument, and adding (gpuCellBytes - grid - 4 B) x subCells to
    that max needs its own `tune_report_set` check; (b) the GPU snapshot's host copies live until the host function
    returns and could be freed before the unlock (`src/ps_metal_host.cc`); (c) `psParkRoom` does not shrink for the
    partitions the dynamic concurrency starts beyond the plan's (P8.10's behaviour from before this change), so
    parking could take such a run past the budget by up to a lane for each of them.
14. D14, the tuner's lines: AUTO-TUNE-REPORT unchanged (no share key, `over_budget` last); the human line's tag reads
    `, GPU deposit with the free cores (GPU share 0.60)` when mixed.
15. D15, the checkpoint key: `--run-mode` and `--gpu-share` leave the command lines; a mixed run writes
    `gpu-share=<per mille>` (none at 1000); `--resume` with `auto` adopts the checkpoint's share (missing = 1000), and
    an explicit share that differs is refused by name.
16. D16, look-ahead, CPU teams and merge: no new rule; a share partition takes today's CPU path (the core lend
    included) and never counts as a GPU fallback; the look-ahead line names the share's CPU partitions.
17. D17, the experiment's name: `DTFE_PS_GPU_SHARE` (P12.2's, in no release) is removed, with one note when it is set;
    the binary has no environment name for the share; the drivers' knob is `GPU_SHARE` (run_ps_dtfe.sh);
    `DTFE_GPU_SHARE_RHO` is the dev override.
18. D18, the run-log texts: a `run mode` header line before the `1Mpc =` line, a `GPU share` header line when the flag
    is given with `--gpu`, and the post-plan share line; every existing literal byte for byte, and no fallback phrase
    in the new texts.
19. D19, the errors: an unknown `--run-mode`, an unknown `DTFE_RUN_MODE`, a `--gpu-share` that is not 'auto' or a
    fraction from 0 to 1 (so the experiment's `600` is refused); a second `--run-mode` and standard DTFE's
    `--gpu-share` are boost's own refusals.
20. D20, the drivers: config.sh's `knob_run_mode`, `run_mode_option` and `run_mode_resolve` (flag, then the caller's
    knob, then auto); `gpu_auto_resolve <knob> <binary> <ps|dtfe> <exact> <mode>`, which the pipeline now calls too;
    the tokens after the existing `part_args`; a `Run mode:` banner line; the pipeline resolves once and forwards
    `DTFE_RUN_MODE`; its `run_settings` unchanged (the mode never makes a snapshot stale).
21. D21, the automatic device on a GPU build (drivers and launcher): the GPU for PS-DTFE's exact deposit too (before:
    the CPU), for its sampled deposit and for standard DTFE's exact average; standard DTFE's sampled average on the
    CPU in performance (315 s against 433 s on the GPU), on the GPU otherwise; the CPU on a CPU-only build.
22. D22, the launcher: `run_mode` (default balanced) in the three specs; `DTFE_RUN_MODE` in every job's environment,
    `GPU_SHARE=auto` in a PS job's; a 'Machine use' combo right after the GPU tick; the GPU advice mirrors D21 and a
    mode change re-applies the tick unless it was set by hand; a one-time `run_mode_start` migration ticks the GPU of
    saved PS exact jobs; the mode stays out of presets and 'Load settings'.
23. D23, the GUI's device advice rewritten per mode, with bench numbers only.
24. D24, a SLURM export runs performance, with one comment line saying so.
25. D25, Explore's zoom passes `--run-mode balanced`; dtfelib's `Estimator` takes `run_mode` (unset: nothing passed),
    and the launcher's Explore gives it balanced.
26. D26, the run records: `run_mode` from the logged command line (absent: performance) joins the time estimate's key
    and the job summary, not `settings()`.
27. D27, the tests: the new `tests/run_mode_check.sh` in `run_program_suites` (core and double, CI and the battery);
    `unset DTFE_RUN_MODE` in `tests/lib.sh`, `tests/dev/battery.sh` and `tests/dev/tune_report_set.sh`; no share pin;
    the driver cases in `run_scripts_check.sh`, the GUI cases in its two suites.
28. D28, the numbers the repository quotes: the P11.1 bench table, the two rows below with their conditions, the gap
    model and the display probes; the rest is stated as not measured. **Amended 2026-11-10:** plus the mode bench's
    rows below, each time with their conditions (the night's machine state and the control run's 789 s).
29. D29, left out on purpose (each a separate commit with a deliberate gate diff): the pure-GPU model's per-lane
    over-booking of its grids, the scratch-write side finding, a rho from the hardware, SMT detection, a mixed share
    for balanced.

**Measured** (D28; TNG100-3-Dark z = 0, 512³, M1 Max, PS-DTFE's exact deposit, 2026-11-09, `2f428bc` plus the
selection skip and the core lend): the GPU at 4M items per buffer (performance's buffers) with the tuner's plan
1025 s; a GPU share of 0.6 with `--partition 6 6 6 --max-concurrent 6` given by hand 712 s; against the `2f428bc`
bench's 1133 s on the CPU and 1156 s on the GPU. The gaps between buffers: 18.6% of the time the run holds the GPU at
50 ms buffers and 4.4% at 250 ms by the model, 17.4% measured at 53 ms. The display probes: late by more than a 60 Hz
frame in 0.0% of the samples at 53 ms buffers, in 5.6% (up to 214 ms) at 155 ms. **Not measured:** nice 10, a CPU
run at a lowered thread budget, and any mode on a machine at rest (the mode bench below ran under a screen saver).

**Mode bench 2026-11-10** (03:46-06:54, one run each with the binaries directly; the input and fields of the `2f428bc`
bench; the code of the copy's scratch checkpoint `11f269a`; session notes `run-mode-bench-notes.md`,
`inv/results-modes/summary.txt` and the contract's section 15). The machine was not at rest: a screen saver drew all
night, and with no run the GPU was 59% and the efficiency cores 67% busy (28% and 50% the evening before). Two
controls with the binaries from before the modes give the night's level: the binary and command of the 712 s run took
789 s, standard DTFE's `--gpu --exact` 478 s (439 s in the bench, 466 s the evening before). So the rows compare with
each other and with the controls, not with the bench table.

| run | wall | plan | predicted | peak | grids |
|---|---|---|---|---|---|
| PS performance at a 36.4 GB budget (40.4 GB free at the start) | 1028 s | 6 6 6, 3 at once, none added | 34.7 GB | 28.9 GB | the mixed run's |
| PS performance at the full 41.2 GB budget | 908 s | 6 6 6, 4 at once, up to 5 in flight | 38.0 GB (41.3 GB with the fifth) | 29.8 GB | identical to the row above |
| the same with `DTFE_FREE_GB=1000` (the free-memory reading overridden) | 893 s | the same, up to 5 in flight | 38.0 GB (41.3 GB with the fifth) | 29.3 GB | identical |
| PS performance, `--partition 6 6 6 --max-concurrent 6` by hand | 783 s | 6 at once | 44.7 GB | 31.4 GB | identical |
| control: the 712 s run's binary and command | 789 s | 6 at once | 48.9 GB (the earlier model) | 31.2 GB | identical |
| PS balanced (6 threads) | 1444 s | 6 6 6, 4 at once, every partition on the GPU | 39.4 GB | 30.0 GB | the bench's GPU run's, byte for byte |
| PS gentle (4 threads) | 1408 s | the same | 39.3 GB | 30.0 GB | the same bytes |
| standard performance (10 threads) | 486 s | 3 3 3, look-ahead off | 34.9 GB | 33.1 GB | the bench's, byte for byte |
| control: the standard binary from before the modes | 478 s | the same | 34.9 GB | 33.8 GB | the same bytes |
| standard balanced (6 threads) | 447 s | 3 3 3, look-ahead on | 41.2 GB | 41.3 GB | 8 of 10 differ at float rounding |
| standard gentle (4 threads) | 444 s | 3 3 3, look-ahead on | 37.6 GB | 40.7 GB | 8 of 10 differ at float rounding |

What it shows:

1. No regression from the prototype to this code: 783 s against the control's 789 s with the same plan and the same
   bytes; standard 486 s against 478 s. The night's level is about 11% above the evening's for PS-DTFE (789 against
   712 s).
2. Performance's own plan: since the once booking (item 13) the dynamic concurrency starts under a mixed plan and
   reaches 5 partitions in flight at the full budget. Its start check with the run's own numbers: own 15.63 GB + 5
   lanes and the once part 21.72 GB + reserve 3.58 GB = 40.94 GB against the 41.23 GB budget; six would need 44.27 GB.
   The run with the free-memory reading overridden reaches the same 5, so the budget term is the limit, not the free
   memory. At a 36.4 GB budget the plan is 3 at once and none is added.
3. The plan costs time: 908 s against 783 s with 6 at once by hand (125 s or 16% more), and the 6 peak at 31.4 GB
   where the model predicts 44.7 GB (the run with 5 in flight: 29.8 GB against 41.3 GB). The model books about 13 GB
   more than a mixed run takes at 6 lanes, so the mode's own plan runs fewer partitions at once than the memory holds.
4. PS balanced and gentle are within 3% of each other (1444 and 1408 s, 1.4 busy cores on average): with every
   partition on the GPU the thread budget hardly matters. They take 1.6 times performance's 908 s.
5. Standard DTFE: performance is the slowest of the three. The tuner's 10-thread plan leaves the look-ahead no memory
   on this input (the binary from before the modes plans the same), the 6- and 4-thread plans have it. Their peaks:
   41.3 GB at a 41.2 GB budget (balanced) and 40.7 GB against 37.6 GB predicted (gentle).
6. Standard DTFE's `--gpu` grids follow the thread count too: 8 of the 10 grids of balanced and of gentle differ from
   performance's and from each other's, the rms of the difference at most 1.4e-10 of the field's.
7. Responsiveness: the night's probes do not rank the modes. Two runs of one configuration differ more than the modes
   do (the GPU probe late by more than a 120 Hz frame in 18.5% of the samples of one performance run and in 44.3% of
   its repeat). The evening's numbers (`inv/results/`) stay the ones to quote.

Open after the bench, the owner's call (nothing of it is changed):

- PS-DTFE's memory model for a mixed plan (the lanes and the points output): a refit to the measured peaks would let
  performance plan 6 at once on this input. A planner change with its own `tests/dev/tune_report_set.sh` diff.
- Standard DTFE in performance: a plan that gives up thread sub-boxes for the look-ahead where the memory holds only
  one of the two (447 s at 6 threads against 486 s at 10 here). It changes default plans, and grids at float rounding.
- The standard look-ahead's prediction (gentle: 40.7 GB measured against 37.6 GB; balanced at the budget's edge).
- A repeat of the mode bench on a machine at rest, for numbers that compare with the bench table and for the
  responsiveness ranking.

**What moves.** The binaries' defaults keep the `2f428bc` bytes and tuner reports; the one intended difference,
performance's 4M item cap, changes no byte and has no tuner term. The drivers' and the launcher's runs default to
balanced: fewer threads (standard DTFE's grids, the GPU's too, then differ from earlier all-core runs at float
rounding, while the pipeline still counts those grids as current), and PS-DTFE's exact deposit on the GPU (in
performance with the free cores beside it). The drivers always pass `--run-mode`, so the scripts and the binaries ship
together.

**Status 2026-11-10: IMPLEMENTED, BUILT, BENCHED, GATES GREEN, committed as `9bfa39a` on top of `39583f0`** (code, drivers,
launcher, tests and docs edited in parallel, one file set per agent, by the contract's ownership table; then built,
reviewed and fixed in a copy of the tree, and applied to the tree as one patch). Passed in the copy before the bench:
G-build for the three pairs (`METAL=1`, `METAL=1 DOUBLE=1`, `DIM=2`), the Metal binaries checked with
`strings | grep -c`; the program suites in both precisions (`tests/run_mode_check.sh` included), the drivers' and the
launcher's suites; G-byte (`tests/dev/bytecmp.sh --gpu-exact` against the `2f428bc` reference set: every file
identical) and G-tuner (`tests/dev/tune_report_set.sh` against the `2f428bc` reference: every line identical). The k=9
cases of `gpu_retry_check.sh` still fault where they expect under the 4M cap: no re-pin.

**Gates run 2026-11-10 07:20 to 08:50, on the tree with the bench texts above, all passed:**
1. `make library METAL=1`: built, no warning.
2. G-battery: 33 of 33 stages, `deprecation_gate` included; `tests/run_mode_check.sh` 55 passed, 0 failed in `ci_core`
   and 47 passed, 0 failed in `ci_double` (two skips there by design: the double build's CPU sums follow the thread
   count, and it has no dynamic concurrency). The four Metal binaries carry their kernels (5 / 15 / 5 / 15).
3. G-linux: CI's Linux stages in Colima (build, core, double, 2d, variants, oldpy, ruff; the launcher's suites run in
   the Mac battery's `ci_gui`) all passed. `tests/run_mode_check.sh` there: 35 passed, 0 failed (core) and 34 passed,
   0 failed (double), on 8 threads without performance cores (gentle 4, balanced 6, performance 8); the priority block
   passed (gentle at nice 10, an inherited nice 15 kept, no nice named by the other modes); the GPU block is skipped
   on the CPU-only builds.

**Applied to the tree 2026-11-10 12:29** (on `39583f0`): every tracked file equals the gated copy's, byte for byte
(the copy differs from the gates' tree in the bench texts of `change_log.txt` and one comment line of
`src/user_options.cc`). The three pairs and the library rebuilt there without a warning, the Metal binaries carry
their kernels (5 / 15 / 5 / 15), and `tests/run_mode_check.sh` (55 passed, 0 failed; double 47 passed, 0 failed),
`tests/ci_suite.sh 2d` and `tests/docs_drift_check.py` (36 passed, 0 failed) passed again. The battery and the Linux
check were not repeated on the tree.

Then, for the owner and not a gate, on the bench input: one `--auto-tune-report` with `DTFE_AUTO_TUNE_DEBUG=1` of
`--gpu --exact --gpu-share auto --run-mode performance` (predicted_gb must land in 32.5-41.2 GB; the plan's
concurrency) — **done 2026-11-10, item 13 above: 34.68 GB, 3 at once**; optionally a performance run on the tuner's
plan against the 712 s hand plan, and a balanced run (the GPU at 50 ms) against 1156 s — **both done 2026-11-10, the
mode bench above** (with the binaries directly, not through the drivers).

## R. The rerun campaign (once, after every code item)

The owner's rule: no production reruns until every code item of P3–P10 is in and checked on small data.
Then, on a quiet machine, overnight, outputs into separate folders (production grids are never overwritten):
- **P2.3:** TNG50-3-Dark z = 0 (GPU sampled + 4096² points, with the logged run's `DTFE_MEM_BUDGET_GB=52`; needs
  ≥ 56 GB free at the start), the 1024³ CPU exact run, the CPU `--ps-window` runs at 2048² and 4096². The P2 status
  table and the P8.1 error table are filled from their logs.
- **P8.2:** the CPU exact term's constants refitted from the 1024³ run; **P8.1** re-baselined per class.
- **P10:** `ps100_099_face512` (decided 2026-11-08; in the rerun script after the 1024³ run): PS-DTFE on TNG100-3-Dark
  z = 0 at 512³, the exact CPU deposit, `--ps-volume-weighted --face-gradient`, the same nine fields as the sampled GPU
  `ps100_099` it is compared with (no sample points); then the exact deposit with the face outputs on the other
  snapshots the analysis uses; standard DTFE with `--face-gradient` on one snapshot as the full-size equality check.
- **Science:** standard vs PS velocity divergence and shear, per cell, masked by stream count (how far standard DTFE's
  derivatives are off in multi-stream cells); void boundaries from the single-stream region; void mass flux from div p.

**Status 2026-11-09.** The calibration night ran 2026-11-08/09 (binaries `4e2b35b`, Metal-only; outputs into the data
disk's `p23-reruns/`, the production folders only read): the P2.3 reruns (TNG50-3-Dark z = 0 as a hand 6³ mc 2 under
the owner's 50 GB ceiling, not the logged run's 52 GB budget; the 1024³ CPU exact run; TNG100-3-Dark z = 0 standard
and PS), `ps100_099_face512`, `std100_099_face` and two diagnostics; not the CPU `--ps-window` runs. Results in the
P2, P8 and P10 status (the morning's 6³ face-check diagnostic in the P10 status; every gate green by 07:35). The rest of R (the exact deposit with the face outputs on the other snapshots, the science
comparisons) was not launched; before it the owner decides D15 (the PS deposit) and D16 (the frame fix first).

---

## Decisions for the owner

Each decision has a recommended default, so the plan can proceed if no answer comes.

| ID | Decision | Recommended default | Alternative |
|---|---|---|---|
| D1 | `PS_GPU_CPU_SHARE` prototype | **delete** (P1.11): 4.5% at best, untested, about 45 lines in the GPU walk. **Decided 2026-11-06: delete.** | keep it opt-in and add a test plus a README line |
| D2 | canonical name for the exact deposit in both binaries | **`--exact-average`**: no change for standard DTFE's production command. **Reversed 2026-11-09 by D20: `--exact` (with `--sampled`) in both binaries, both old names deprecated aliases of it.** | `--exact` in both, with both old names as aliases |
| D3 | legacy switches | keep `DTFE_MC_LEGACY` and `DTFE_TESS_CACHE_LEGACY` while grids made the old way exist; **drop the `*_METAL_*` knob aliases** after P1.10 | keep all |
| D4 | always-on macros `VELOCITY`, `SCALAR`, `TEST_PADDING` and the `*_OPTIONS` help macros (**upstream**) | **leave them**; their off-states are not built anywhere, so do not advertise them | delete the off arms, or add a CI compile of each off-state |
| D5 | padding-test dummy points (**upstream**, semantic) | **keep**, unless P2.4 shows over 5% of a standard run | P9.4 |
| D6 | `--fixed-point-merge` as default | **leave it off** if the P2.3 reruns park at most a handful of partitions | make it the default for PS runs above some N |
| D7 | split `depositLoop` (1010 lines, hot) | **no**, unless P9.2 needs the restructuring; the proof would be bytes plus disassembly plus a ±1% timing check | split into a `DepositCtx` plus `psDepositExactTet` |
| D8 | launcher Advanced sections | **collapse** Resources and the expert PS options, with the state remembered | only shorten labels and tooltips |
| D9 | when to remove the flag aliases | **the release after** the one that introduces them | keep them indefinitely, hidden |
| D10 | **upstream** dedupe and deletions (P1.4's upstream half, P3.9) | **golden tests yes**; the dedupe and deletions **no**, since the repository is still a fork of upstream. **Changed 2026-11-07:** the owner no longer tracks upstream ("focus on this fork only"), so upstream-marked code is ordinary code now (D4 and P3.9 may proceed) | do both |
| D11 | the GPU backends | **Metal-only**: CUDA/HIP has never run on NVIDIA or AMD hardware, costs 4.5k lines; keep it on an archive branch (the analysis is in the 2026-11-07 session notes). **Half decided 2026-11-08:** its four CI jobs (`cudaemu` 32/64, `cuda-compile`, `hip-compile`) were removed on the owner's instruction. **Decided 2026-11-08 (evening): remove.** The port (six source files, 4.6k lines), its CPU emulation (`CUDAEMU=1`, `tests/cudaemu_check.sh`), the twin check (`tests/gpu_twin_check.sh`), `GPU_ARCH` and the `DTFE_GPU_EMU_WARP` knob are gone; `make CUDA=1`, `HIP=1` or `CUDAEMU=1` stops with one error line naming `archive/cuda-hip-2026-11-08` (= `f9f6e2f`, the last commit with the port; the branch must be on origin, since five pointers in the tree name it). Linux stays CPU-only. P4.3 step 2 followed (one `gpuRunPass`). | CPU-only after a same-tree A/B; or keep all three |
| D12 | PS velocity derivatives | **write both** the mean-flow gradient (with the caustic sheets) and the regular part ⟨∇v⟩ | only one of them |
| D13 | standard DTFE's gradient | **DECIDED 2026-11-09 (owner, 14:40): the face gradient is the default** (`--per-tet-gradient` / `DTFE_FACE_GRADIENT=0` keep the deposit's own; the drivers' `FACE_GRADIENT` empty = the binary's default, 0 = per-tet, 1 = face; bytecmp pins the per-tet route, the GPU suites pin it where they test the kernel's gradient; the production grids made before this date hold the per-tet gradient). The record of the decision's path: **keep the per-tet deposit as the default.** **Checked 2026-11-09 (phase R): the condition is not met.** The full-size check (`std100_099_face`: TNG100-3-Dark z = 0, 512³, GPU exact average, 3³) fails the test's bound in 133 of 243 records: 64,978 cells (4.8e-4) above 1e-4, worst 1.2e-2, 99.7% of them in one strip where the **deposit's** float32 region frame rounds (the P10 status); elsewhere, and on a CPU 256³ run with an exact frame, the two agree to an rms of 2–7e-7 with a sparse tail (attributed to float32, not tested). The face path replaces the deposit's gradient once P10.8 (the frame fix) is in and a `std100_099_face` rerun (8 min) passes the restated bound (the P10 status; the demo's G1 keeps 1e-4). Until then `--face-gradient` is the exact side where the frame rounds, except in the degenerate cells, where it keeps the flattened tetrahedron's flux that the deposit zeroes (neither route validated there). **At 6³ (`std100_099_split6_face`, 2026-11-09 06:42) the check nearly passes: 903 cells (6.7e-6) above 1e-4, worst 1.9e-3, worst record rms 2.9e-6, no strip; it meets the proposed restated bound, so the 3³ strip is the frame as predicted (the P10 status).** **P10.8 done 2026-11-09 (afternoon), and its 512³ rerun on the production 3³ plan meets the restated bound (294 cells above 1e-4 = 2.2e-6 pooled, worst record rms 5.4e-7, no degenerate cell, worst 5.7e-4; the P10 status), so the switch's condition holds. The switch itself — the face path as the default gradient of every standard run, which changes every standard gradient output (bytecmp re-taken for the Monte-Carlo runs' derivative files too) and the drivers' default — is the owner's call; `--face-gradient` stays opt-in until then.** | keep both paths |
| D14 | default field set | **density, velocity, divergence, shear in both drivers (done 2026-11-07)**; the rest on request | — |
| D15 | the PS deposit for the campaign (added 2026-11-09, phase R) | **measure first, then choose by the outputs the analysis needs; one deposit throughout a simulation.** The night does not support "CPU exact costs the same and the GPU is unreliable": like for like at TNG100-3-Dark z = 0 the two tie (1377 against 1379 s, different builds), at the eight other logged redshifts CPU exact took 1.21–1.43× the GPU wall (the 27 logged runs by ratio: 11.4 h against 9.8 h), the TNG50 run's length is attributed to its hand plan (mc 2 against the logged mc 6; no park room, so 58 spills), not to its GPU failures (about 4% of the wall), and both deposits conserve mass (mean ρ/ρ̄ = 1 to 3e-8, no empty cell). But the face grids (grad⟨v⟩, div p for R's void mass flux) exist only with CPU exact, whose stream counts and caustic bits are also fuller (32.9% multi-stream cells against 24.5%). The measurement that settles the cost: TNG50-3-Dark z = 0, CPU exact + `--face-gradient` + `--sample-points … --pts-vel-grad`, after an `--auto-tune-report` (0.75–2.4 h by scaling, up to 5.5 h; all 10 cores). A CPU exact command must carry the points (`ps100_099_face512`'s did not). **Bench 2026-11-09 (P11.1; the committed tree, TNG100-3-Dark z = 0, 512³, with the points):** exact CPU 1133 s, exact GPU 1156 s, sampled GPU 1260 s, sampled CPU 1644 s. With the face gradient, which since D17 also comes with the GPU exact deposit: exact CPU 1737 s (46.4 GB max RSS, 30.4 GB peak footprint), exact GPU 1454 s (49.6 GB, 33.4 GB) -- both inside the 41.2 GB budget by footprint (corrected 2026-11-10: the max RSS had been read as the peak). **Decided 2026-11-09 (evening, owner; D20): the exact deposit, the default of both binaries since; the device stays D18's.** The face runs waited on P11.7 item 1, done on 2026-11-10: j stands as measured, l ran with 4 in flight where the corrected plan chooses 3. | GPU sampled + points as logged (byte-reproducible at a given split; the TNG50 run, whose one pass fell back to the CPU, differs; faster at z > 0, about 7 cores left free), without the face grids |
| D16 | the frame fix before the campaign (added 2026-11-09, phase R) | **yes: P10.8 first, both parts.** The frame part is established by the 6³ face-check run (`std100_099_split6_face`, the P10 status): it removes the rounding strips (the 3³ plan's; a 6³ plan has none) and lets D13 switch under the restated bound. It does not, by itself, make the standard grids reproducible across splits: the ~1e-5 box-wide split dependence (the P2 status) is in the face pass too and is not the frame; its leading candidate, the per-tetrahedron gradient's float labelling dependence, is P10.8's second part, an hour to test and small to fix. Together they are what could bring the cross-split differences down to the float sum order (~1e-7); byte identity would need fixed-point sums as in PS-DTFE. Each moves the standard exact-average grids (bytecmp re-taken); effort M for the frame (kernel and host), S for the gradient; verify as P10.8 says. The PS campaign waits on it only if P10.8 finds the same frame in the PS deposits. **Done 2026-11-09 (the P10 status of that afternoon). PS-DTFE's deposits name their cells from the full box's corner and spacing in every partition (`ps_interpolation.cc`: `region` stays the box in a partition's options, the GPU's `boxLo` is the box's lower corner), so they have no sub-domain frame and the PS campaign never waited on this; the ~1e-5 PS-DTFE moved across splits at z = 1 (the P2 status) has another source, not established.** | run the standard campaign now with its ~1e-5 reproducibility across splits and builds recorded, D13 at "keep both paths" |
| D17 | the PS face gradient on the GPU exact deposit (added 2026-11-09, after the owner's rule "one method, sampled and exact, on whichever device is faster") | **DONE 2026-11-09 16:00 (code; P11.5): `--face-gradient --gpu` runs, the face sums from one CPU walk after the GPU deposit, byte-identical mean-flow grids to the CPU run's; its speed against the CPU exact run is the bench's run m.** Before: the PS face gradient (the mean-flow derivatives and the mass flux the void analysis needs) exists only on the CPU exact deposit, which refuses `--gpu`; the GPU exact PS deposit has no face sums and has never run at production size, so its speed against the CPU's is unmeasured. Without it the PS campaign is CPU exact (1.2–1.4× the GPU sampled wall at z > 0, equal at z = 0, the face pass up to +21%). Item sketch (P10.9): the face sums as a CPU pass beside the GPU deposit (the standard binary's way: the pass walks the tessellation, which the GPU run keeps), the volume-weighted mean from the GPU's grids; effort M | leave the PS face gradient on the CPU; the campaign as D15 says |
| D19 | run-to-run byte reproducibility vs the parallel insert's default (added 2026-11-09 16:30, P11.3) | **open, for the owner.** The default rule inserts tessellations of ≥ 1e6 points in parallel where cores are free (3.1× on the triangulation of a low-concurrency 512³ run, nothing at the auto plan's concurrency); two identical runs then differ at rounding in the CPU deposits' float sums, the face pass's last bit and PS-DTFE's velocity moments (the PS density and the GPU exact grids stay byte-identical). The owner's instruction of 13:50 ("turn the TBB insertion on for all binaries") stands; its premise -- the deposits no longer depend on the construction order -- holds for the GPU exact average and the PS density only. Alternative: `--sequential-triangulation` as the default and the parallel insert chosen by the drivers for runs with few tessellations in flight (1024³, `MAX_CONCURRENT` ≤ 2), or a canonical vertex order in PS-DTFE's deposit (the standard `mcVertexOrder`'s way) so its moments stop depending on the order (effort M). **Bench 2026-11-09 (P11.1):** the rule engaged in neither production configuration at 512³ (standard: the auto plan leaves one core per tessellation; PS-DTFE: 771,057 padded points per 6³ partition, below the threshold), so there the default costs no reproducibility and gains no time; the choice matters for the memory-bound runs with few tessellations in flight (1024³) | keep the rule (this tree) |
| D18 | the device chosen by the program (added 2026-11-09, the same rule) | **open (the mechanism exists, the rule awaits the bench; P11.4).** The drivers' `DTFE_GPU=auto` (`gpu_auto_resolve`) already picks the device per build and, for PS-DTFE, per deposit; the binaries' `--gpu` stays explicit; the auto-tuner plans partitions and threads around it but never picks the device. After P10.8 the GPU and the CPU exact deposits of standard DTFE differ only by float rounding, so the choice is speed and memory alone: a rule in the tuner (a GPU build with a device present → `--gpu` for the standard exact average, which is 6.6× the CPU on the demo; the PS exact deposit stays on the CPU while D17 is open) with `--gpu`/`--no-gpu` as overrides; effort S. The drivers and the launcher then stop passing `--gpu`. **Bench 2026-11-09 (P11.1):** the demo's 6.6× does not carry to 512³: the standard run is CPU-bound, GPU exact (439 s at 3³) ties CPU exact (431 s at 2³) and the CPU sampled run is the fastest (315 s); for PS-DTFE exact GPU ties exact CPU (1156 against 1133 s) at a quarter of the CPU seconds, and D17 is done. The rule's table still needs the GPU at 2³ (P11.7 item 3); the tuner's face booking is corrected (P11.7 item 1, 2026-11-10: the GPU face run now plans 3 in flight where it ran 4) | keep `--gpu` explicit (the drivers pass it) |
| D20 | the `_a` average's flags and default (added 2026-11-09 evening, after the P11.1 bench) | **DECIDED 2026-11-09 (owner): `--exact` / `--sampled`, exact is the default for both estimators** ("exact is the right strategy across the board, but keep sampled as an option"). Reverses D2's canonical name `--exact-average`, which stays a deprecated alias of `--exact` in both binaries (as `--ps-exact-deposit` does in PS-DTFE), removed per D9; settles D15. The bench behind it: the exact average costs about the sampled one's wall without its sampling noise (standard DTFE 439 against 433 s on the GPU, 431 against 315 s on the CPU; PS-DTFE exact CPU 1133 s, GPU 1156 s against sampled GPU 1260 s, CPU 1644 s). The rules, what moves and the gates: P12. Implemented 2026-11-09, gates green on 2026-11-10, committed as `39583f0` | keep the sampled default with `--exact-average` opt-in (D2) |

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
| Painting the 3D view's settled QPainter picture in threads, worker processes or tiles | Measured with `70e153e`'s speed-ups: threads 1.0–1.2× (PySide6 holds the GIL in `drawImage`), processes at most ~1.7× (memory bandwidth), and a tile clip starting at x > 0 changes Qt's pixels. The GPU renderer covers the on-screen case. |
| `logging` instead of `print` in `dtfelib` and the plot scripts | The launcher parses their stdout. |
| `-MMD` dependencies for the `.cu` rules | Cannot be tested without nvcc/hipcc. Keep the hand-listed headers. |
