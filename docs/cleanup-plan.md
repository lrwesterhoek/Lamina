# Lamina: cleanup and polish plan

Status (2026-11-07): P1 and P7 done; most of P2–P6 and P9 done (each phase's status line says what is open); P8 not
started. P0.1 and P0.2 landed in `60a072a`; P0.3's CI fixes and P0.4 landed in `70e153e` and `38eb4be`. Open in P0:
CI green on `38eb4be`, then `ref0` frozen (P0.3), and one change_log sentence (P0.4). The plan covers the C++
binaries (DTFE, PS-DTFE and their double and 2D builds), the Metal and CUDA/HIP GPU code, the driver scripts,
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
| **G-build** | all six build modes: plain, `METAL=1`, `METAL=1 DOUBLE=1`, `DIM=2`, `make library`, `CUDAEMU=1` | 0 warnings, 0 errors |
| **G-byte** | `tests/dev/bytecmp.sh <ref0> <out>` against the reference binaries taken in P0.3 | every file identical (184 output files after P0; 261 with `--gpu-exact`) |
| **G-gpu** | `tests/dev/bytecmp.sh --gpu-exact` (P1.12), plus `metal/validate_deposit` | identical bytes before and after a host-only change; 31/31 |
| **G-emu** | `CUDAEMU=1` build + `tests/cudaemu_check.sh` at warps 32 and 64 | PASS (warp 32 failed on `60a072a`; fixed in `70e153e`: P0.3) |
| **G-battery** | `tests/dev/battery.sh` | 31/31, tree unchanged during the run |
| **G-linux** | CI's Linux stages in a local Ubuntu 24.04 container: `tests/ci_suite.sh build`, `core`, `double`, `2d`, `variants`, and `CUDAEMU` + cudaemu_check | PASS; required before every push. glibc differs from macOS where the tests can see it: it writes a long unbuffered stderr line in 128-byte pieces (P0.3) |
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
| **P4** | GPU structure: shared host code, one params header, twin check | no | G-byte, G-gpu, G-emu, retry suites | P1 | L |
| **P5** | Python, plots, tools, bash drivers | no (same figures and data) | py suites, `run_scripts_check.sh`, G-docs, G-move | P0 | L |
| **P6** | One flag scheme with a deprecation path | no (names only) | resume suites, `run_scripts_check.sh`, runspec suite, G-docs, G-byte | P3 (P3.3) | M–L |
| **P7** | Launcher polish, including the 3D view | no | G-gui | P1, P6 | L |
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

**P0.3 Make CI green and freeze the baseline.**
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
- Step 2: the six binaries of `70e153e` (byte-identical source to `38eb4be`, which changes only the workflow) are
  kept as `ref0`, with checksums, in a folder outside the repository. Every byte-identical phase compares against
  it. It is frozen once CI on `38eb4be` is green.
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
- Gate: CI green on all seven jobs (the run on `38eb4be`), G-battery.
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
- Risk: none. Effort: S. Verify: G-build, which must include the CUDA/HIP compile jobs on CI.

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
- Open: P2.3's TNG50-3-Dark z = 0 rerun (GPU sampled + points, ~2.4 h logged), the 1024³ CPU exact run and the CPU
  `--ps-window` runs.

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

**P2.3 Rerun the runs the tuner table flags (see P8), with the current binary.** — *three of them done 2026-11-07 (see the status above).*
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

**Status 2026-11-07.** All done except P3.9's dedupe (D10).
- P3.5: `interpolateGrid_phaseSpace` 2533 → 1686 lines (`PSFxState`, `psMarkFlipped`, `psCentroidCell`,
  `psWindowTouched`, `psFxBoundWalk`, `psGpuExtractAndDeposit`). `depositLoop` stays inline (D7) and compiles to the
  same 7370 instructions (-O3, METAL object); only the three calls into the lifted functions differ.
- P3.7: 643 → 433 lines; not instruction-identical (the body is split), byte-identical outputs, MC average x1.004 and
  exact average x0.999 of the base's time. The helpers' `* const &` bindings matter: copies cost about 4%.
- P3.8: the whole model is in `src/auto_tune.cc` (anonymous namespace); the header says a pointer to
  `auto_tune.h <name>` now means the pair. Its follow-ups are done too: the header includes only what the
  interface uses (every includer's object byte-identical), and the PS branch's unused `countCell` is gone.
- P3.9, first step: `tests/dtfe_sampling_check.sh` (methods 2 and 3, the redshift cone, the grid's point values,
  user points through libDTFE, stored goldens for the cone and the user points). Upstream bugs it found, not fixed
  (D10): (a) `averaged_interpolation_2.cc` builds method 2's sample as `Point(uni()*dx+x, uni()*dy+y, uni()*dz+z)`,
  unsequenced, so GCC and Clang draw different realizations from one `--seed`; (b) `interpolateRedshiftCone`'s
  infinite-cell `continue` skips `psi += dx[2]`, shifting the rest of the row; (c)
  `interpolateUserSampling_averaged_2` skips a sample outside the region without pushing a value, so the output is
  shorter than the sample list.

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

**P3.9 (optional, upstream) The sampling variants.** — *partly done: the golden tests (D10) 2026-11-07.*
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

Gate for each item: G-byte, G-gpu, G-emu. The retry items add `tests/gpu_retry_check.sh`,
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

**P4.3 One chunk-and-retry loop.** — *partly done: step 1 and half of step 3 2026-11-07.*
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

**P4.4 A Metal↔CUDA twin check, before any kernel text changes.** — *done 2026-11-07.*
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

**Status 2026-11-07.** All done except part of P5.5.
- P5.5: `make_parser(snap_default=, snap_help=)` replaces the private argparse edits in six scripts; `add_sim_arg`,
  `add_snaps_arg`, `add_out_arg` (`--output` and `-o` too) and `add_rv_definition_arg`. Left, each for a reason:
  `add_void_args` (every script's void options have their own help), routing the three tools through `make_parser`
  (six new flags in their `--help`), one `tools/_bootstrap` (the launcher loads `void_outlines.py` in-process, where
  `tools/` is not on `sys.path`). `-o` is now a short option of `analyze.py` and nine plot scripts.
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

**P5.5 Argument helpers.** — *partly done 2026-11-07.*
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

**Status 2026-11-07.** P6.1–P6.5 done, with D2, D3 (the `*_METAL_*` knob aliases are no longer read) and D9;
P6.6 is for the next release. Choices made where the plan left room: the deprecation line goes to stderr in lower
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

**Status 2026-11-07.** Not started. P2.3's first three reruns (the P2 status) give P8.1 two classes: the standard
GPU exact average +43% and the PS GPU sampled with points +27% and +41%, all over-predicting now that the parking is
bounded; the TNG50, 1024³ exact and window reruns are still to do. The standard planner's margin costs time: it chose
6³ where 3³ fit, and the run took 1.8× the hand-set one. Since P9.2, the single-precision build's CPU deposit follows
the core pool (`CorePool` in `src/thread_count.h`: holders = min(planned concurrency, partitions left)), which P8's
`cores_eff` should use.

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

## Decisions for the owner

Each decision has a recommended default, so the plan can proceed if no answer comes.

| ID | Decision | Recommended default | Alternative |
|---|---|---|---|
| D1 | `PS_GPU_CPU_SHARE` prototype | **delete** (P1.11): 4.5% at best, untested, about 45 lines in the GPU walk. **Decided 2026-11-06: delete.** | keep it opt-in and add a test plus a README line |
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
| Painting the 3D view's settled QPainter picture in threads, worker processes or tiles | Measured with `70e153e`'s speed-ups: threads 1.0–1.2× (PySide6 holds the GIL in `drawImage`), processes at most ~1.7× (memory bandwidth), and a tile clip starting at x > 0 changes Qt's pixels. The GPU renderer covers the on-screen case. |
| `logging` instead of `print` in `dtfelib` and the plot scripts | The launcher parses their stdout. |
| `-MMD` dependencies for the `.cu` rules | Cannot be tested without nvcc/hipcc. Keep the hand-listed headers. |
