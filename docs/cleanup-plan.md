# Lamina: cleanup and polish plan

Status (2026-11-11): the code of phases P0 to P13 is done and committed; the last commit is `2d262e4` on the branch GPU
(P11.8, the planner's refit). A batch of leftovers on top of it -- planner booking gaps, the mixed run's GPU lane
memory, test skips under the debug switches, the launcher's face-gradient tick box (removed by the two-mode launcher,
`1f8511b`: P14), the 3D view from inside the box -- passed its gates on 2026-11-11 (the leftovers records under P11.8;
the 3D view's exception in the scope rules below) and waits for the owner's commit. Phase R ran its calibration night
on 2026-11-08/09 (the R status); the rest of R is not launched. What is left -- full-size runs, the owner's decisions
of 2026-11-11 turned into code, the campaign and a tail of small items -- is listed in "Open items (2026-11-11)" after
the scope rules; the owner's answers of that day are under the decision table. Each phase's status line says what that
phase left. Since 2026-11-13: P14, the owner's two-mode rule; its phase C (the launcher) is built on the scratch line
(the first of the open items). On the same line the face walk beside the GPU deposit became PS-DTFE's default
(`b62eb94`: P11's status of 2026-11-13 on the default) and the word merge holds the face sums once per face
(`3947bb9`, S7: P10's record after S6); both are checked there on small data, and neither has been through the full
gate, the Linux check or a 512³ run.

Earlier status (2026-11-09), kept for the record: P0, P1, P6 (P6.6, the alias removal, is for the next release), P7 and P10.1–P10.7 done; most of P2–P5 and P9
done (each phase's status line says what is open); phase R's calibration night ran 2026-11-08/09 (the R status; P10.8, decided and done on 2026-11-09 (D16), D15 settled the same evening by D20 (exact; P12, committed as `39583f0`); P13 (the run modes, added 2026-11-10: committed as `9bfa39a`); P11.7 items 1 (the PS tuner's face sums), 2, 4, 5 and 6 done 2026-11-10 (item 5's boundary-plane section built 2026-11-11; item 3, the pinned runs, open); the rest of R was not launched); P8
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
    - One asked-for exception, 2026-11-11 (the owner: the volume "looks like a sticker", and zooming in should show the
      field inside): in perspective a zoom in moves the eye into the box, and the volume is drawn for an eye inside it
      (each pixel from the slice stack its ray crosses most). The pictures from outside, measured against the commit
      before (QPainter, 640x480, a TNG100 256³ cube, fitted, seven turns): the parallel projection draws the pixels of
      before at every turn. In perspective a pixel whose ray runs more along another axis than the middle of the view's
      now takes that axis's slices, so an oblique view changes there: at the default turn a quarter of the volume's
      pixels (the middle difference 8 of 255, a tenth of them over 38), two thirds looking down the box's diagonal
      (there also more opaque, by 13 of 255 on average), a ninth at a third oblique turn, and none at the other four
      (face on, turned 20° and tilted 10°, tilted 60°, from nearly straight above). It is the same field on another
      stack of slices. Parts 10–11 gain the checks for the eye inside; the ones that pinned the old perspective (the
      eye stopping short of the window, one slicing axis for the whole view) gave way to the per-pixel and the dolly
      checks. `change_log.txt` has the entry.
  - The GPU must stay within its tolerance of QPainter: `volume_gl_parity` in the same suite, run in a child process
    on the default Qt platform. It SKIPs without a display or OpenGL, so CI never runs it: run it on the Mac for
    any item that touches the volume's drawing.
- **Hot-loop rule:** the PS deposit-loop lambda and the GPU extraction walk stay inline unless the split is proven
  output-identical and timing-neutral.
- (Changed 2026-11-07, D10: upstream-marked code is ordinary code now.) Upstream code from Cautun's DTFE (2011 headers)
  is touched only where this plan says so: `averaged_interpolation_2.cc`,
  `unaveraged_interpolation.cc`, the SPH/NGP interpolators, `src/kdtree`, and the `Particle_data`, `Box` and
  `miscellaneous.h` helpers. Changes there are marked **upstream**: they are the owner's call and low priority.

## Open items (2026-11-11)

Gathered from the later notes of each phase and the owner's answers of 2026-11-11 (under the decision table). A decided
item's code follows in a later batch; nothing here is built unless it says so.

- **The scratch line of 2026-11-13 and the owner's two-mode rule.** Six scratch commits on top of `39410e2`, built and
  smoke-tested on the scratch line (the time objective with its switch unset only); none of the six went through the
  battery, the Linux check or a 512³ run (what was measured on small data is in each record): S6, the exact
  partition merge in words (`--ps-merge fixed-point`, `f2aa3c1`, with its periodic seam for one domain, `d13425a`:
  P10's status, the record after S5); the planner's time objective (`DTFE_AUTO_TIME=1`, `148f253`: P8.8); the bound
  walk's block claim and the standard CPU words' booking (`71ab973`: P9's status at P9.2, P10's fix-3 record for
  standard DTFE); the face default's left-overs (`eba8096`: P10.11, P11's status of 2026-11-13 on the early walk's
  memory term); the GPU face kernel (`e29845d`: P11's status of 2026-11-13 on the kernel). Two more scratch commits
  followed the same day: the merge's review fixes (`41860cc`: P8.8's record; the planner suites' unsets, below) and
  the two-mode launcher (`1f8511b`: P14, phase C; through the four launcher suites, not through the battery). Then
  two more code commits: the face walk beside the GPU deposit as PS-DTFE's default (`b62eb94`: P11's status of
  2026-11-13 on the default) and the word merge's face sums once per face (`3947bb9`, S7: P10's record after S6),
  with a test-only commit after them (`b6fa0b8`: section AB of `tests/ps_auto_tune_check.sh` reads the line's
  normalization text); checked on the line with both: clean builds, the default bytes (261 files, 0 differ), the
  planner set (the 19 GPU face lines move with the walk default, none with S7) and the suites those records name.
  The owner's rule of 2026-11-13 (P14): each estimator has two quality modes for its users, `--exact` (the default)
  and `--sampled`, and nothing else is a user option. Every switch of the 2026-11 round -- `DTFE_AUTO_TIME` (with its
  hooks `DTFE_AUTO_TIME_GAIN`, `DTFE_AUTO_T_STD_PART` and `DTFE_AUTO_T_PS_PART`), `DTFE_FACE_F32`, `DTFE_GPU_FACES`,
  `DTFE_FACE_WALK_EARLY` (since `b62eb94` its `=0`, the walk after the device call, against the default),
  `DTFE_PS_RUN_UNITS` and the value `fixed-point` of `--ps-merge` -- is a developer switch for
  one measurement (README.md, "Developer switches"; the launcher shows none): once measured it is hard-wired or
  removed, and the measurement decides which: none is a user option to weigh. The two planner suites unset the
  switches that would move their plans themselves since `41860cc`: `tests/ps_auto_tune_check.sh` unsets
  `DTFE_GPU_FACES` and `DTFE_FACE_F32` (an exported `DTFE_GPU_FACES=1` booked face records in its GPU face plans), and
  `tests/dtfe_auto_tune_check.sh` unsets `DTFE_RUN_UNITS_LEGACY` (I2 and O set it on one of their twins: an exported
  one made both twins the same).
- **The owner's actions.** Commit the leftovers batch (gated on 2026-11-11, the records under P11.8). D11: the branch
  `archive/cuda-hip-2026-11-08` must be on origin, since five pointers in the tree name it. Merging GPU into master is
  the owner's too: CI runs on pushes to master and Phase-Space, on pull requests and by hand, so a push to GPU alone
  starts no CI job. P6.6 (the old spellings removed, `ref0` re-frozen) waits for the release after the one that carries
  the aliases.
- **Full-size runs** (each on the owner's go, on a quiet machine with Colima stopped):
  - P11.8's confirming series was run on 2026-11-11 (six runs: run l, performance and standard balanced on their own
    plans, standard 3³ one partition at a time, PS balanced at `--partition 5 5 5 --max-concurrent 4`, PS on the CPU;
    P11.8's "The confirming series"). Left from it: one 6³ balanced PS run on the same binaries for the split choice,
    and the thread step-down's measured run against the series' standard balanced run (the planner batch). Left of the
    three prediction errors P11.8's leftovers name: one TNG50-3 rerun on a night the owner picks, and the mixed lane's
    term (a) read from the campaign's own runs.
  - P11.7 item 3 (standard DTFE on the GPU at the CPU's 2³ plan; D18 was closed without it) and item 4's thread series
    for a points run at 512³.
  - P8.6's CPU `--grid-window` runs at 2048² and 4096² (P2.3's last reruns); P8.5's and P8.10's point sets on TNG50 z =
    0 and TNG100 z = 1, with P8.10's TNG50 check; P8.2's 1024³ run with a footprint trace (the start-up reserve or the
    first merge).
  - P13's mode bench again on a machine at rest (optional).
  - Phase R: the exact deposit with the face outputs on the other snapshots, and the science comparisons.
  - The scratch line of 2026-11-13 (each record names its run): S6's byte identity at 512³ and its wall time and
    footprint (P10's record after S5), and S7's at 512³ (the record after it); P8.8's sweep for the time objective;
    run l's command three ways for the face kernel (`DTFE_FACE_WALK_EARLY=0`, the default, `DTFE_GPU_FACES=1`), one
    512³ run of run l's plan under the walk default (its probe: 41.00 of 41.23 GB predicted) and the windowed 1024³
    GPU face run whose plan the default moves (P11's statuses of 2026-11-13).
- **Decided on 2026-11-11, code to come:** a GPU kernel for the PS face sweep, designed first. Built since (the next
  batch's records in P11.8): a standard plan that steps its threads down to keep the look-ahead (balanced only,
  `DTFE_LOOKAHEAD_TEAM=0` for the old plan), kept as the default after one measured A/B pair (429 s against 536 s);
  the Custom tab's `--ps-merge index-order`; the PS face-gradient tick in the visible "Phase-space options" group
  (removed by the two-mode launcher, P14); the two suites without `--samples 20` (the total itself goes with the
  aliases); the canonical vertex order in PS-DTFE's tessellations (D19: P11.3's records of 2026-11-11; PS-DTFE's
  output bytes move once); the regular part's full gradient, shear and vorticity in PS-DTFE (D12: P10.10); the face
  gradient as PS-DTFE's default where the run qualifies, as in standard DTFE (D13, after D15 and D17: P10's status).
  The face sweep's first step, without a kernel, was built as an opt-in (2026-11-12: the existing CPU walk beside the
  device call, `DTFE_FACE_WALK_EARLY=1`; 13.7–16.5% less wall time at 256³, 7.2% on one 512³ pair; P11's status of
  that day). Built since, on 2026-11-13 (the scratch line): the walk's memory term in the planner; then the walk as
  the default, which answers whether it becomes the default (`kPsFaceWalkEarlyDefault` true, `b62eb94`, built under
  the owner's two-mode rule and not yet through the full gate; `DTFE_FACE_WALK_EARLY=0`, the walk after the device
  call, stays as the developer's off switch for one more measurement of the pair); and the kernel behind the
  developer switches `DTFE_FACE_F32=1` and `DTFE_GPU_FACES=1` (P11's statuses of that day). Whether the kernel
  follows is the owner's; by the two-mode rule (above) a measurement settles each switch.
- **Decided on 2026-11-11 after the explanations:** the three small fixes that change output bytes at about 1e-8 (P4.4's
  linear weights from one compiled order, P11.7 item 5's boundary-plane section, a run-wide fixed-point unit after
  P10.8) are all to be built, before the production campaign, so that the campaign's grids come from the repaired code.
  The first two are built (2026-11-11, the owner's decision 2c: P4's status of that day and P11.7 item 5; gates green).
  Measured, they move more than 1e-8: the linear GPU grids by up to 7.5e-6 of the rms, and the last layer's
  `.a_massFluxDiv` of a lattice box by 0.37 of the rms, a value that was wrong before. The run-wide unit is built for
  standard DTFE (the same day: P10's status after P10.8) and, as an opt-in, for PS-DTFE's passes (S5, built
  2026-11-12, `DTFE_PS_RUN_UNITS=1`, off by default since the owner's decision of that evening: P10's status, its
  bullet on PS-DTFE); the exact merge in words is built as the developer switch `--ps-merge fixed-point` in the
  single build (S6 and S7, 2026-11-13, the scratch line: P10's status, its records after S5; built and checked on
  small data, not gated, not measured at full size). New grids then differ from the grids
  on disk in the last digits, and the byte comparisons' references move once, after these three and the canonical
  vertex order (D19). D6: the fixed-point merge stays opt-in (since 2026-11-13 a developer switch under the two-mode
  rule above: its measurement decides whether it is hard-wired or removed).
- **Small code items, no full-size run:** the split choice with the lane share (P11.8's "Still left of the leftovers":
  both runs are made, the pair on 2026-11-12: 5³ is 1.7% faster than 6³ for 6.34 GB more peak footprint; the owner
  decides); `psFxBoundWalk` claims its blocks from one counter (P9.2's follow-up, 2026-11-13 on the scratch line;
  not gated). Standard DTFE's `_a` average team stays outside the core pool
  by design (P9.2). Done on 2026-11-11 (the next batch's records in P11.8): a standard Grids job's `FACE_GRADIENT`,
  ps_runtime_check R3 under `DTFE_GPU_FLOAT_MOMENTS=1`, the deprecation gate reading the battery's stage logs (P6.7's
  item 6).
- **Larger:** P8.8, a time objective for the planner, and P8.9 step 2, the same for standard DTFE: built 2026-11-13
  behind the developer switch `DTFE_AUTO_TIME=1` (P8.8's record; the scratch line, not gated), the sweep of full-size
  runs that measures it still to run; why PS-DTFE's grids differ across splits (the dispersion moves by over 1% in
  0.02% of the cells, the P2 status of 2026-11-07): analysed on the existing z = 1 pair on 2026-11-12 (the P2 status,
  "Cross-split analysis": 85% of the flagged cells change by no more than rounding that the subtraction amplifies,
  just over half of the cells far beyond it sit on a moved centroid fallback of the sampled deposit, the rest are
  open; by a proxy the seams are not the main cause), a pair on today's binaries is left; the standard face trio's
  over-prediction (P11.8).

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
| **G-battery** | `tests/dev/battery.sh` | (33 stages since 2026-11-09: the P11 status of 16:30; P6.7 added the 32nd.) 31/31, tree unchanged during the run |
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
| **P11** | Which device for which phase, chosen by the program: per-phase timing, the parallel insert's rule, the PS face gradient beside the GPU exact deposit (D17), the 512³ bench (P11.1), the face walk's and the tuner's fixes (P11.7) and the lane-share refit (P11.8); P11.6 dropped; the face walk beside the device call, an opt-in on 2026-11-12 and PS-DTFE's default on the scratch line since 2026-11-13 (`b62eb94`, not gated; the statuses of those days). **Committed** in `2f428bc`, `c8dd4fb`, `ae81362` and `2d262e4`; P11.7 item 3 and P11.8's confirming runs open | no at a given plan (placement moves no byte: bytecmp); above 1e6 points the parallel insert moves the CPU float sums and PS-DTFE's velocity moments at rounding between runs (D19); plans change (P11.7, P11.8) | G-byte, G-tuner, the face suites, G-battery, G-linux; the bench | P10, P8 | 2026-11-09 to 2026-11-11, plus the 14-run bench (P11.1) |
| **P12** | `--exact` / `--sampled`, the exact average the default of both binaries (D20); PS-DTFE's selection skip and core lend. **Committed** as `39583f0` | yes for unflagged runs with an `_a` field (now exact; PS-DTFE adds `.tetTouch`); a run that names its average keeps its bytes | G-build, G-battery, G-byte, G-gpu, G-tuner, G-docs, G-gui, G-linux | P6, P11.1 | 2026-11-09 evening to 2026-11-10 (code and gates) |
| **P13** | Run modes `--run-mode gentle\|balanced\|performance` and `--gpu-share` (a mixed CPU + GPU PS exact run), the launcher's 'Machine use'; the mode bench of 2026-11-10. **Committed** as `9bfa39a` | the binaries' defaults: no; the drivers' and the launcher's balanced default runs fewer threads (standard DTFE's grids then differ from all-core runs at float rounding); a mixed run's grids differ at float rounding, the mass exact | G-build, G-battery (`tests/run_mode_check.sh`), G-byte, G-tuner, G-linux | P12 | 2026-11-10 (code, the mode bench, gates) |
| **P14** | Two modes per estimator, Exact (`--exact`, the default) and Fast (`--sampled`), and no other user option for accuracy or speed (the owner's rule of 2026-11-13). Phase C, the launcher's one Quality choice per run tab: **built** on the scratch line as `1f8511b`, not yet in the battery; phases A and B, the binaries': to do in the clean-up phase | phase C: no (the launcher's commands and environments are the ones of before for the matching settings: the golden comparison); phases A and B: some bare-binary bytes and plans, by the design's steps and the owner's answers to Q1-Q4 | G-gui (the four launcher suites), the golden comparison; then G-battery, G-linux; for A and B also G-build, G-byte, G-tuner, G-docs | P12, P13 | 2026-11-13 (phase C) |
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
- (P1.12 (b) added the script on 2026-11-06, `tests/dev/gui_shots.py`; the plan records no saved set.) Step 4: save the
  GUI renders (P1.12 adds the script that makes them; until then this step waits).
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
gone; of the upstream ones, `Box::isParticleLagrangianInBox`, `upperBoundCheck` and both `superfluous_options` were
deleted on 2026-11-11 (D10; no caller in src, metal, tests, python, scripts or demo); the rest stay.
- `particle_data_traits.h`: `compareParticles`, `sameParticle`.
- Not dead: `triangulation_miscellaneous.h:vertexPositionMatrix` has a caller in `demo/examples/main_particle_density.cpp`
  (the first grep left `demo/` out). It stays.
- **upstream**:
  - `Box::isParticleLagrangianInBox` (deleted 2026-11-11);
  - the `Particle_data` setters `setDelta`, `setLagrangianPosition`, `setScalar`, `setVelocity`, `setWeight`,
    `setPosition`;
  - `setDelta` and `setPosition` are not dead: `tests/dtfe_user_sampling.cpp:40-41` calls them (the `Sample_point`
    API);
  - in `miscellaneous.h`: `upperBoundCheck` and both `superfluous_options` (deleted 2026-11-11);
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

**Status 2026-11-07.** (Updated 2026-11-09: P2.3 done except the CPU `--ps-window` runs; the status of that date below.)
P2.1, P2.2, P2.4 and P2.5 done; P2.3 partly (three of its reruns).
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
    more than 1% in 0.02% of the cells (first read as the tetrahedra at the partition seams; analysed on 2026-11-12,
    the next item: by a proxy the seams are not the main cause), up to 30× in a few. Only the same-split run checks
    tonight's code against the logged one (the others also differ in build).
  - Cross-split analysis (designed 2026-11-11; run on the existing z = 1 pair on 2026-11-12, the result under the
    design). In a periodic run, which tetrahedra exist, who owns them and which samples they hold are the same in
    every split. The ownership is exact (`ps_cell_filter.h`), the inside test is exact and the seam copies are exact
    shifts. What moves with the split is rounding:
    - H1, each tetrahedron's velocity profile, formed in float in CGAL's vertex-0 frame. That frame follows each
      partition's insertion history (P11.3's bytecmp B: the PS velocity moments move, the density does not).
    - H2, the default merge, which adds each partition's once-rounded floats in index order.
    - Rarer: the per-pass fixed-point unit, a tetrahedron deferred to the CPU in one split only, and a centroid-fallback
      cell flip.

    σ² = ⟨v²⟩ − ⟨v⟩² turns one rounding of the moments into a relative change of about u⟨v²⟩/σ², so cold, fast cells
    (σ² ≲ 2e-5 ⟨v²⟩, an estimate) move by more than 1%. The recorded pair also differs in build: the logged 5³ run is
    labelled 49ccb9e, whose kernel still summed the moments with float atomics, and the fixed-point moments, the
    labelling-free degeneracy cut and the exact seam came with 2f232c9. Its run log's `Build:` line decides whether
    that pair measures the split at all.

    The analysis script (design note `cross-split.md`, script `cross_split.py`, both in the session notes) separates
    these on the existing grids: the change of σ² in units of ⟨v²⟩ (rounding: a few u), the density test (H1 leaves it
    identical; H2 moves it one ulp where partitions meet), the location against the Lagrangian faces, neighbour pairs,
    the points and the build lines.

    Result (2026-11-12; a read-only analysis of the existing z = 1 pair at 512³, the 6³ rerun of 2026-11-07 against
    the 5³ production set of 2026-10-14; the record `cross-split-z1-2026-11-12.txt` in the session notes). Both are
    sampled deposits, made before the canonical vertex order (D19). The builds differ (the 5³ set's `Build:` line
    reads 49ccb9e with uncommitted changes); on one split the same two builds gave the byte-identical z = 0 grids
    above, so the pair is read as a measure of the split, which is an inference. The dispersion differs by more than
    1% in 25176 cells (0.01876%), by up to 39.9×. By the change of σ² in units u = 2^-24 of ⟨v²⟩ they fall into three
    classes: 21520 cells (85.48%) up to 32 u, 977 (3.881%) from 32 to 1024 u, 2679 (10.64%) beyond.
    - Up to 32 u is rounding that the subtraction amplifies: 92.57% of that class are single-stream cells, 99.87% have
      σ²/⟨v²⟩ below 1e-4, and the density is identical in 88.17%. With the particle count matched, 35.6% of the class
      have a particle within one particle spacing of a Lagrangian face of either split, against 23.7% of random cells
      (within 2 spacings 44.2% against 34.4%, within 4 58.6% against 52.3%). By this proxy the seams account for about
      13 to 16% of the class. The proxy can miss a cell that two partitions feed, and the clustering below is not
      controlled for.
    - Beyond 1024 u, 1522 cells (56.81%) sit on a density cell that itself differs by more than 4 last places, 1521 of
      them on a pair of neighbours with opposite, equal changes: a tetrahedron that holds no sample, which the sampled
      deposit puts whole into the cell of its centroid, and whose centroid falls in one cell under one split and in
      the neighbour under the other. All 5718 density cells that differ by more than 4 last places carry that
      fallback's flag. The other 1157 cells keep their density (identical in 88.42%) and move by up to 29.8×: their
      cause is open. Candidates from the code, none shown: the vertex-0 frame at thin tetrahedra (H1), or a thin
      tetrahedron on the CPU in one split and on the GPU in the other. Neither class beyond 32 u shows an excess near
      the faces.
    - Not seen: other tetrahedra (over the whole grid the stream count differs in 1.387% of the cells, in each by at
      most 3 last places), a clamp of a negative diagonal, a partition on the other route. Not separated: H1, the
      per-pass unit and the deferred tetrahedra. Half of the flagged cells lie in 35 of the 512 blocks of 64³ cells,
      which is not explained.

    Whether today's default (exact deposits, the canonical vertex order) still shows either class is not measured.
    Proposed, not run: one 6³ / 5³ pair on today's binaries with the outputs kept, then the same analysis.
- (Done on 2026-11-08/09 except the CPU `--ps-window` runs: the status of 2026-11-09 below.) Open: P2.3's TNG50-3-Dark
  z = 0 rerun (GPU sampled + points, ~2.4 h logged), the 1024³ CPU exact run (53 min
  logged) and the CPU `--ps-window` runs. **Moved to phase R** (after all code items, owner 2026-11-07).
  - A first TNG50 attempt (2026-11-07, auto budget 38.7 GB = 42.7 GB free − 4 GB) planned 6³ **mc 1**: the 25.7 GB
    of point records left room for one partition (~2.7 GB, *est.* from the two plans). It was stopped at 69 of 216
    partitions (3 h 7 min; 26.5 GB peak footprint against 38.4 GB predicted) to run overnight. Input for P8.5 and
    P8.10.
  - (Not so in the end: the night of 2026-11-08/09 ran a hand 6³ mc 2 under the owner's 50 GB ceiling; the status
    below.) The rerun uses the logged run's own `DTFE_MEM_BUDGET_GB=52`, so its plan matches the logged one (6³ mc 6).
    The binary caps the knob at the free memory − 4 GB, so it needs ≥ 56 GB free at the start.

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

**Status 2026-11-07.** (Superseded 2026-11-08: all done or closed, the next status.) P4.1, P4.2 and P4.4 done; P4.3
partly; P4.5 not started.
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
  one compiled order. (Requested 2026-11-11: the owner asked for this fix; the owner's answers under the decision
  table. Built the same day: the status of 2026-11-11, below.)

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
  stays as the candidate item (built 2026-11-11: the next status).
- P4.5: moot, one kernel text is left.

**Status 2026-11-11.** P4.4's candidate item is built (the owner's decision 2c; code and tests, gates green). Under
`--ps-linear-deposit` both item passes of the GPU exact deposit run on ONE pipeline, `depositExactItems` specialized
with `kItemsPhase` 2, which reads the pass from `ExactItemParams.phase` and clips at order 2 in both
(`metal/ps_deposit.metal`; `ps_metal_host.cc` `runItems` takes it when `P.fLinear` is set). The cells' weights of the
two passes then come from the same compiled clip, and a tetrahedron deposits m to the rounding the uniform deposit has
(each item's float sum of its cells' weights, and the shares). That is argued from the code, not measured on the device:
no check compares the two passes' weights (one would sum the deposit pass's own weights into a second total under a
debug switch; not built), and a compiler that split the cell loop on the pass could still round them apart. What was
measured, the binaries of the commit before against this one (64³ particles, grid 128³): the `.den`/`.a_den` and `.vel`
grids of a GPU exact linear run change in 26.7% and 27.9% of the cells in the single build (92.3% and 92.7% in the
double), by at most 7.5e-6 of the rms, a density cell by at most 1.7e-6 of its value and a velocity component by at most
5.2e-6; the other five grids are byte for byte, and so is a repeated run. That is more than the 1e-8 of the decision's
text: the deposit pass is a new compilation of the float clip, whose rounding the cells show. Neither the totals nor the
CPU deposit show the repair. Under `DTFE_MASS_AUDIT` the grid's mass lies above the tetrahedra's by 1.854e-8 before and
1.860e-8 after (double: 1.851e-8 and 1.850e-8), where the uniform GPU deposit has 2.01e-8 on both binaries and the CPU's
linear deposit -1.4e-11 (double 4e-14): the excess belongs to the GPU exact deposit, is older than this change and is
not traced (a follow-up). Against the CPU's linear deposit of the same run the GPU density is as far off as it was (mean
relative distance 3.04e-6 for both binaries, at most 6.2e-3 of the rms; double 7.7e-8 and 1.2e-3), 279873 of the changed
cells moving toward the CPU's value and 279775 away from it. The CPU's own grids are byte for byte between the two
binaries. Values 0 and 1 of `kItemsPhase` fold to the constants they were, so the uniform deposit's two pipelines should
compile to the code they were (gate: `tests/dev/bytecmp.sh --gpu-exact`, 261 output files, 0 differ against the
reference the commit before matched; the uniform runs of the commit before and of this one byte for byte, 8 files in
each precision). Cost, only in GPU exact runs with `--ps-linear-deposit`: the summing pass computes the second moments
too, in a pipeline that also carries the deposit's code (its pass times under `DTFE_GPU_TIMING`, which names its passes
`exactItems linear`; 128³ particles on a 256³ grid, single build, six rounds with the four runs rotated, medians: the
summing passes 9.40 s before and 11.62 s after, the deposit passes 11.34 s and 12.50 s, the four together 20.74 s and
24.12 s, +16%, and the run 44.2 s and 47.8 s; the uniform deposit's four passes 20.68 s and 21.13 s on the same
binaries, their ratio 0.87 to 1.15 round by round with the machine's speed, the linear ratio 1.16 in four of the six
rounds, 1.00 and 1.35 in the first two). Every PS GPU process compiles the third specialization once, when it first
takes the device (no compile time is printed; the wall time of a 16³ run on a 32³ grid, medians of eight interleaved
pairs: 0.441 s before, 0.456 s after); if it does not compile it stays null with the compiler's reason, a linear run's
partitions take the CPU deposit ("GPU deposit unavailable") and every other run keeps the GPU. The CPU path already
summed the weights it deposits (`addPiece`, `exW`). Test: `tests/ps_linear_deposit_check.sh` E, runs 14-15 (each item
pass's pipeline name; the mass audit within 1e-7, which cannot resolve the 1e-8); the suite passes in both precisions
(31 checks, 8 s).

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
  --ps-point-conventions`) and `--ps-merge MODE` only when `PS_MERGE` is given (a given index-order too). (Reversed
  2026-11-08 by R7: a plain run passes `--ps-merge index-order` again; P6.7f.) So a plain
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
  `ps_vertex_mass_check`, `ps_convergence_test` and `py_dtfelib_test`, PASS after it). (Decided 2026-11-11: the Custom
  tab is to pass it as well; the owner's answers under the decision table.) Since 2026-11-11 (decision 1a of the
  leftovers list) the Custom tab passes `--ps-merge index-order` too, leaving it to its extra options when they name a
  merge (`CustomSpec._extra_names_merge`: `--ps-merge MODE` or an older switch, read through `runlog.canonical_flag`),
  so no launcher path runs the speed split without the gate; only a bare binary run does. No output bytes change (the
  merge is index order either way); py_gui_runspec_test and py_gui_app2_test pin the flag and the "extras name a merge"
  case.
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
  C15, `dtfe_resume_check` D4 (`--lambda_th`), the run-log reader cases of the python suites, and, until 2026-11-11,
  one deliberate deprecated total, `--samples 20`, in `dtfe_sampling_check` (S3/S7 golden values) and
  `dtfe_mc_average_check` (A1); by decision 11a the total goes with the aliases, and neither suite uses `--samples`
  any more (A1 runs `--subsamples 2`; the method-2 runs take method 2's default of 20 with no count; the owner's
  answers under the decision table). A test that reads an environment variable of the binaries itself reads the old
  name as a fallback. Measured in this step (Mac): the three builds and `make library METAL=1`, 0 warnings, the build stamps
  unchanged; `docs_drift_check` 36/36 (A-G and the synthetic half); `flag_spelling_check` 56/56; the 54 suites the step
  touched (the py and GUI suites, run_tests.py, run_scripts_check, the resume, merge, window, 2D, auto-tune, cache,
  point, GPU and retry suites) all pass, `ps_auto_tune_check` Y5 on the second run (the first saw 5 partitions in flight
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
  the files read. The five python suites (ps_3d_test, ps_standard_cross_check, ps_convergence_test, ps_nonperiodic_test,
  ps_regression_test) that let the binaries' stderr through to the battery's stage logs have logged every binary run
  into their temporary folder since `4e2b35b` (`ps_test_helpers.run(log=...)`), and since 2026-11-11 the gate reads the
  stage logs as well (`--stage-logs`, the next batch's record in P11). (7) Decided: the launcher exports
  `DTFE_AUTO_PTS_VEL_GRAD` only; the older spelling beside it
  would make the binary warn, and the alias goes with P6.6. (8) `ps_resume_check`'s kill cases print the binary's
  "checkpoint could not be written" warning and the free space when no checkpoint was left (`tests/lib.sh
  no_checkpoint_why`, C2 and C15). Review of the batch (one opus skeptic per group, one re-check of the gate): four
  real findings, three fixed (the gate's marker under `--only`, `grep -I` skipping NUL-bearing server logs, a
  symlinked temp folder read as empty) and one recorded as a known gap (the stage logs; closed on 2026-11-11, above);
  nits fixed (`void_outlines` exits in one line on an unknown simulation, a tautological launcher test, nested
  parentheses in `make_image_plane`'s usage error, `void_cache.py build` aligned, the Y5 comment, D2's assertion in
  D2's check). Battery after the batch
  (2026-11-08 14:16-14:44): 30 of 32 stages at first. `ci_core` failed `ps_tessellation_cache_check`'s standard-binary
  fingerprint check while a Colima image prune ran beside the battery; alone it passes, and that check keeps neither
  reader's output, so the cause is not known (an external-load flake, seen once). `ci_gui` failed `py_gui_app_test`'s
  Stop check, which sampled the process tree at a fixed 3 s after the exact 256^3 run had already ended (2.9 s now);
  it polls for the binary (up to 10 s) and the GUI stage passes (412/388/251/59). The `deprecation_gate` stage PASSED
  on the real battery folder (allow-listed: flag_spelling_check, ps_resume_check C8 and C15, ps_runtime_check R7).
  (The flakes' cause was found and fixed on 2026-11-10: P12's gates, item 2.) Left for a later batch: that fingerprint
  check discards both runs' output, so a repeat of the flake could not be
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
| `--ps-gpu`, `--ps-exact-deposit`, `--ps-no-overlap`, `--full_help`, `--lambda_th` | `--gpu`, `--exact-average`, `--no-overlap`, `--full-help`, `--lambda-th` | (Superseded 2026-11-09 by D20: `--ps-exact-deposit` and `--exact-average` are aliases of `--exact`, P12.1.) P6, unchanged |
| `--redshiftSpace` | stays | its `redshiftSpace=` line is in every cache key and globals record |

Driver knobs and options (`config.sh`, `run_dtfe.sh`, `run_ps_dtfe.sh`, `run_ps_pipeline.sh`; new wins with a warning
during the window, no conflict exit; the `VAR="${VAR:-...}"` line shape kept for the knob-discovery checks):

| Old | New | Note |
|---|---|---|
| `PS_GPU`, `PS_METAL`, `DTFE_METAL` | `DTFE_GPU` (auto\|0\|1) | both drivers; (Superseded 2026-11-10: auto puts PS-DTFE's exact deposit on the GPU, P13.1 item 21.) PS auto under exact resolves to the CPU |
| `DTFE_EXACT` (P6.3, never shipped), `PS_EXACT` | `DTFE_EXACT_AVERAGE` (the shipped 1.0.0 name) | (Superseded 2026-11-09 by D20: exact by default in both drivers, P12.1 rule 6.) each driver keeps its default (std 1, PS 0) |
| `PS_RESUME` | `DTFE_RESUME` | also the refusal texts (`START_OVER`) |
| `AVG_SUBSAMPLES`, `-n k` | `SUBSAMPLES`, `-n k` / `--subsamples k` | per axis in both; the standard driver passes it only when set |
| `ORDERED_MERGE` + `FIXED_POINT_MERGE` | `PS_MERGE` | converted |
| `PS_PARALLEL_TRI`, `PS_LAG_PADDING`, `PTS_CONVENTIONS`, `TESS_CACHE` | `PARALLEL_TRIANGULATION`, `PS_LAGRANGIAN_PADDING`, `PS_POINT_CONVENTIONS`, `TESSELLATION_CACHE` | R9 |
| `LAGRANGIAN_INPUT` | `PS_LAGRANGIAN_INPUT` | `${OLD+x}`: an explicit empty value still means InitialCoordinates |
| `DTFE_GRID_SIZE`, `DTFE_PADDING`, `DTFE_SPACE_CHECK` | `GRID_SIZE`, `PADDING`, `SPACE_CHECK` | the pipeline's early `GRID_SIZE` capture goes through the resolver too |
| `WINDOW`, `PY` | `PLANE_WINDOW` (`make_image_plane.py --plane-window`), `PYTHON` | also `runspec.py` `PIPELINE_ONLY_KEYS`, `install_gui_app.sh` |
| (none) | `THREADS` in `run_dtfe.sh` too | `MPC_UNIT` stays a constant |
| `= "1"` tests | `knob_on` (1/0/true/false/yes/no/on/off) | everywhere at once, incl. `run_ps_pipeline.sh`'s exact test and `gpu_auto_resolve` |
| long options | `--sim`, `--sim-dir`, `--grid`, `--gpu`, `--exact-average`, `--subsamples` | (Since 2026-11-09 `--exact` and `--sampled`; the drivers' `--exact-average` warns: P12.1 rule 6.) `-m` / `-e` stay; the pipeline keeps the letters in its `run_ps_dtfe.sh` call (`runspec.py` `_PLAN`) |

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
| exact cell averages | `--exact-average` | `--ps-exact-deposit` | (Superseded 2026-11-09 by D20: `--exact`, with `--sampled`; P12.) `--exact-average` (see D2) |
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
- (Superseded 2026-11-10: auto puts PS-DTFE's exact deposit on the GPU, P13.1 item 21.) The renamed knobs keep the
  exact-deposit rule: an unset or `auto` `PS_GPU` resolves to the CPU when `PS_EXACT=1`.
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

**P6.6 One release later: remove the aliases.** (Decided 2026-11-11 for the total that is no k^3: it goes with the
aliases, and the two suites that use it move to a cube; the owner's answers under the decision table.) The precedent is
the removal of `--metal` / `--ps-metal`. Effort: S.
The final alias tables (P6 and P6.7), all removed together:
- the binaries: `deprecatedOptions` (36 rows: the PS block, the standard `method` / `approxPSD` rows, the shared rest)
  with their hidden options, and `deprecatedSwitches` (PS-DTFE's 3 merge switches: `ordered-merge`,
  `completion-order-merge`, `fixed-point-merge`, their members `psOrderedMergeFlag` / `psCompletionOrderMergeFlag`
  then set by `--ps-merge` alone); the hidden `-m/--method` and `-s/--samples` of both binaries (standard: the total
  `noPoints` path in `readAveragingOptions` and `foldDeprecatedAliases`; a total that is no k^3 is then no longer
  expressible (decision 11a); no suite needs one since 2026-11-11, and method 2's default of 20 points per cell is no
  alias and stays); the `--ps-stream-density
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

**Status 2026-11-07 (evening).** (Updated 2026-11-09: the status of that date below.) P8.2 (as the partition span), P8.7, P8.9 step 1 and P8.10 done in code (below); the rest open. Gate changes first:
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
(The TNG50 and 1024³ reruns ran on 2026-11-08/09: the status below.) Earlier (2026-11-07): P2.3's first three reruns
(the P2 status) give P8.1 two classes: the standard
GPU exact average +43% and the PS GPU sampled with points +27% and +41%, all over-predicting now that the parking is
bounded; the TNG50, 1024³ exact and window reruns are still to do. The standard planner's margin costs time: it chose
6³ where 3³ fit, and the run took 1.8× the hand-set one. Since P9.2, the single-precision build's CPU deposit follows
the core pool (`CorePool` in `src/thread_count.h`: holders = min(planned concurrency, partitions left)), which P8's
`cores_eff` should use. P8.10 (dynamic concurrency) proposed 2026-11-07, from the stopped TNG50 rerun. Order (owner, 2026-11-07): the code items come first, with constants from the existing logs; P8.1's re-baseline and P8.2's refit run in phase R.

**Status 2026-11-09 (phase R).** (Later: P8.9 step 1 refitted on 2026-11-11, P8.9; P8.10 started in the mode bench of
2026-11-10, P13.) P8.1 re-baselined for the classes the night ran; P8.2 rescoped again (the start-up
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
  standard planner on the P2.3 run): +10.5%, in band. (Done 2026-11-11 another way, P11.8's post floor
  `kStdPostKeepGrids`: P8.9.) Its refit of `kStdPostKeep` / `kStdPostKeepFloor` from `before
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
  rerun yet. (Since 2026-11-11 P11.8's lane share books the other lanes in flight at a share, in runs with evaluation
  points.) The points were 53% / 67% of the sampled class's over-booking; the rest is the span terms (the snapshot
  + mc × the largest sub-grid, 1.7–1.9× what the runs held), which no item covers yet. P8.5's guard concerns CPU
  exact + points, not rerun: one `ps_exact` rerun settles it.
- P8.3 not needed: the 1024³ run parked one partition (2.03 GB) inside its 8.8 GB room. P8.4 no allowance: RSS is
  17.0 / 17.5 GB above the footprint (1024³ / face512), most likely the file-backed scratch pages (not verified;
  `std100_099_final`, without scratch, shows 9.8 GB): they are in RSS, not in the footprint the error is taken on.
- (It started on 2026-11-10: the mode bench's performance runs reached 5 in flight, P13's "What it shows" item 2; the
  TNG50 check is still open.) P8.10 not measured: the hand plan turns it off, and the three auto runs printed "dynamic
  concurrency not started: the free memory or the budget at the start holds no partition beyond it". The start check
  (`DTFE.cpp`) needs room for mc + 1 booked partitions (4.4 GB each for the TNG100 points run, 13.55 at 1024³) beside
  the reserve, in the live free memory and in the budget, from which the planner had already taken the largest mc it
  fits. It needs a start that passes: P8.10's TNG50 verify after P8.5, or a tighter span booking.
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
  `psRunPartitionLoop`, line 1761) and the spill budget fixed that. (Re-measured by P2.3's reruns, 2026-11-07 and
  2026-11-08/09: the P2 and P8 statuses.) The −45…−69% must be re-measured (P2.3) before the model
  changes.
- **The CPU exact deposit has no term of its own.** (P8.2, 2026-11-09: none is needed; the 1024³ miss fits the start-up
  zero fill.) In `AutoTunePSPlanner::partitionBytes`, the exact deposit's
  per-tet weights and item list are booked only in the `metalActive` branch. The CPU branch books the sampled
  deposit's copies or slab lists only. *Verified by reading.*
- **The point-stream budget is about 2× generous.** (Tightened to `1 + 1.4(est − 1)` on 2026-11-09: P8.5.) The estimator
  (`psEstimateMeanStreams`) is good: estimated vs real
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

**P8.2 A CPU exact term.** (The gate's I1 was adjusted the same day and the battery passed: the P8 status's Gates.) — *rescoped 2026-11-09 (phase R) and in code: no per-partition term; the 1024³ miss fits the start-up zero fill of the largest scratch grid; its battery gate fails (the status above).*
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
- **Built 2026-11-13 behind the developer switch `DTFE_AUTO_TIME=1`** (both binaries, from `DTFE_AUTO_MINN` particles;
  P8.9 step 2 with it; the scratch line's `148f253`: built, and smoke-tested with the switch unset only (no run with
  it on is recorded; `auto_tune.cc`: nothing was verified with the switch on), not gated, not measured). Without the
  switch every plan is to be as before: the switch is read once and every new branch is behind it (to be measured:
  `tests/dev/tune_report_set.sh` with `DTFE_AUTO_TIME` unset, against `39410e2`'s 127 lines).
  - The form, not the proposal's ⌈n³/c⌉ waves. PS-DTFE (`AutoTunePSPlanner::loopSeconds`): a closed queue -- c lanes,
    each a partition's CPU work Z, around the one GPU hold S -- solved by mean value analysis, in five classes
    (`kPSTime`: GPU, GPU with face sums, GPU under balanced or gentle, mixed, CPU), each with costs per padded vertex,
    sample point, particle in the lane, team core-second, GPU hold and partition, a fitted wall factor and the run
    outside the loop; the mixed class's team work scales with the CPU's share of the partitions. Standard DTFE
    (`AutoTuneStdPlanner::loopSeconds`): per partition the slowest thread sub-box's build, contended by the threads
    that build at once, plus the deposit (a serial part and a part the team divides, per cell), with the look-ahead
    the larger of the next build and this deposit, plus a constant per partition; a tessellation that inserts in
    parallel with `kAutoTimeParInsertThreads` threads or more builds in `kAutoTimeParInsert` of the time.
  - Constants: from the phase totals of the TNG100-3-Dark z = 0 runs at 512³ of 2026-11-09 to 12 (9.42e7 particles,
    the exact deposit, the development machine); `auto_tune.cc`'s block of the time objective names the runs of each
    and marks the ones fitted to measured loops. One size: beyond it every constant is an estimate, the grid limit
    `kAutoTimeMaxCells` (2e8 cells) among them. How far the model's loops lie from the measured ones (to be measured: the
    model's residuals on its training runs and on held-out runs, read from the logs).
  - The chooser: the memory objective's plan unless another plan the memory model accepts predicts below
    `kAutoTimeGain` = 0.90 of its loop. PS-DTFE: every split of the search at the count the search gives it
    (`countAt`), raised by the lane share (`shareRaise`), or at a user's `--max-concurrent` where the memory model
    holds that many; a one-domain plan falls back. Standard DTFE: every split with the threads and every 2^a 3^b team
    below them, each with the look-ahead where it fits. A plan the user fixes (standard DTFE: `--partition` and
    `--max-concurrent`; PS-DTFE: `--partition`) is predicted, never changed. Calibrated configurations only -- the 3-D
    single build and the exact deposit; standard DTFE on the GPU with the face pass, the volume average (method 1), no
    `--sample-points`, not `DTFE_GPU_FLOAT_GRIDS=1`; PS-DTFE with sample points, no window, the per-pass units (so not
    under `DTFE_PS_RUN_UNITS=1` nor `--ps-merge fixed-point`, whose reason line names the first), on a GPU face pass
    neither the GPU face kernel (`DTFE_GPU_FACES=1`) nor the walk beside the deposit (the default since `b62eb94`:
    only `DTFE_FACE_WALK_EARLY=0`, the walk after the device call, is in the class), the fixed-point moments without
    the float-sum or audit knobs, the face sums only in all-GPU performance runs, a CPU or a mixed run only in
    performance, a mixed one at a GPU share of 0.5 to 0.7; grids up to 2e8 cells -- else the
    memory plan and the reason. The report line gains `time_s` / `time_mem_s` (the predicted whole runs of the plan
    taken and of the memory objective's) where it predicted, and an `AUTO-TUNE: time objective (DTFE_AUTO_TIME)` line
    states what it found (both predictions where it leaves the memory objective's plan). `DTFE_AUTO_TIME_GAIN` (the
    band, 0 to 1) and `DTFE_AUTO_T_STD_PART` / `DTFE_AUTO_T_PS_PART` (seconds per partition) are the tests' hooks.
  - Tests: `tests/dtfe_auto_tune_check.sh` N (N1-N7), `tests/ps_auto_tune_check.sh` AA (AA1-AA7).
  - The merge's review fixes (`41860cc`, the same day): `timeClass` names two more configurations as not calibrated,
    so the time objective keeps their memory plan: a GPU face pass with the walk beside the deposit (tested by
    `psFaceWalkEarly()`, which the memory term `earlyFace` reads too, instead of `Env::flag`: `DTFE_FACE_WALK_EARLY=1`
    no longer takes a run without a GPU face pass out of its class), reason "the face walk beside the GPU deposit
    (DTFE_FACE_WALK_EARLY)", and a GPU face pass under the kernel (`DTFE_GPU_FACES=1`, `faceKernel`; its run has no
    CPU face walk to time, and before it was predicted with the CPU walk's constants), reason "the GPU face kernel
    (DTFE_GPU_FACES=1)". `tests/ps_auto_tune_check.sh` unsets `DTFE_GPU_FACES` and `DTFE_FACE_F32`,
    `tests/dtfe_auto_tune_check.sh` `DTFE_RUN_UNITS_LEGACY`. With the switch unset nothing of it runs.
    Since `b62eb94` (the same day) the walk beside the deposit is the default and `timeClass` tests the kernel before
    it, with the reason "the face walk beside the GPU deposit (the time constants were measured with the walk after
    it)": under the switch every single-precision GPU face run with the default walk keeps the memory plan, and the
    class "GPU with face sums" (`kPSTimeGpuFace`) is reachable only under `DTFE_FACE_WALK_EARLY=0`
    (`tests/ps_auto_tune_check.sh` AA6: a performance run under the default falls back, said, without time keys).
    Whether to measure the class again with that walk and drop the branch is the maintainer's (to be measured:
    `kPSTimeGpuFace` refitted from 512³ runs with the walk beside the deposit, the default).
  - Open: `timeClass` has no test for `DTFE_FACE_F32=1` alone (`auto_tune.cc` does not read it): a run whose CPU face
    walk forms the face sums in the kernel's float arithmetic keeps the default arithmetic's class (an all-GPU
    performance face run is classed "GPU with face sums"), whose constants come from runs of the default sweep.
    The SUMMARY's `time model` line
    (predicted against measured loop) is not built. The run outside the loop is not scaled with the size, so below
    512³ the difference `time_s - time_mem_s` is the figure to read. Dynamic concurrency (P8.10) is not modelled. The
    sweep that verifies the model and decides whether the switch is hard-wired or removed (to be measured: TNG100-3-Dark z =
    0 at 512³, a quiet machine, `DTFE_GPU_TIMING=1`: PS run l's command at 6³ x {3, 4}, 5³ x {3, 4} and 4³ x 4;
    standard performance 3³ x {10, 4 with the look-ahead, 6} and 4³ x 10, twice each, alternated; PS mixed and CPU
    exact with the face sums of today's default; PS balanced 5³ x 4 against 6³ x 4; a tight `DTFE_MEM_BUDGET_GB`,
    standard and PS, the switch off and on; one 1024³ run).

**P8.9 Standard DTFE: a prediction line and a time model.**
- All 27 standard production runs were hand-set and used 20–45% of the CPU. This is the largest untapped speed-up for
  the standard estimator.
- Steps:
  1. Check the standard planner's prediction on the P2.3 run.
  2. Give it the P8.8 objective, with the standard's serial partitions and threads per partition.
- Effort: L.
- **2026-11-11 (P11.8):** step 1's prediction is refitted where the mode bench found it too low: the look-ahead is
  booked at `kStdLookAheadFactor` = 1.275 times its count, and the post-processing side keeps at least
  `kStdPostKeepGrids` = 0.5 times the grids only the post-processing allocates (the 1 GB of `kStdPostKeepFloor` stays
  the minimum). Step 2 is open.
- **2026-11-13:** step 2 built with P8.8's objective, behind the same developer switch (`DTFE_AUTO_TIME=1`; P8.8's
  record): the scratch line, not gated, not measured.

**P8.10 Dynamic concurrency in the PS partition loop.** (It started in the mode bench of 2026-11-10: P13's "What it shows" item 2.) — *done in code 2026-11-07 (the owner: "continue with … the remaining plan items"); not yet measured: it did not start in phase R's runs (the P8 status).*
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
  review; since 2026-11-11 also the point evaluation's team and the look-ahead thread; standard DTFE's `_a` average
  team stays uncounted (that binary has no pool user, so nothing would read it).
  Follow-up, built 2026-11-13 on the scratch line (`71ab973`; not gated): `psFxBoundWalk` claims its blocks the same
  way (`PSBlockClaim`'s overload that also returns the block's index; one serial walk builds the block index; the team
  keeps `psDepositThreads()` threads under its `CorePool::Hold` and takes no free core). Its 13 results are double
  sums, each block's made by the one thread that claimed it, in the cells' order, and the blocks added in index order,
  so they are the static walk's at every thread count and claim order and still the GPU extraction's (`gpuBoundPart`):
  no switch, and no byte is to move (to be measured: `tests/dev/bytecmp.sh` against `39410e2`). `DTFE_DEBUG_FX_BOUNDS=1`
  prints each walk's sums in hex with its thread count; `tests/ps_fixed_moments_check.sh` F6 compares them at 1, 3 and
  6 threads, on the whole box and in a `--grid-window`. Time saved (to be measured: a CPU exact `--face-gradient` run at
  256³, one domain and 2³ with 4 in flight, old and new binaries). `psFaceWalk` keeps the static rule.
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

(Superseded: P9.5 was reverted and P9.7 decided against, the P9 status of 2026-11-07; since 2026-11-10 the drivers put
PS-DTFE's exact deposit on the GPU, P13.1 item 21.) Together, P9.5–P9.7 *(est.)* save 150–190 s of the 266 s. That
brings the 94M-particle GPU exact run from 17m39s to
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

**Status 2026-11-07.** P10.1–P10.7 done (CPU), with the drivers' `FACE_GRADIENT` knob. (Decided 2026-11-11: the tick
moves to the visible "Phase-space options" group; the owner's answers under the decision table.) The launcher has the
tick since 2026-11-11 (the visible 'Phase-space options' group of the Grids, Pipeline and Custom tabs since decision
15b of the same day, first built into the folded Advanced group; PS-DTFE jobs in 3D; on by default since the face
gradient became PS-DTFE's default (D13), the default one constant, runspec.FACE_GRADIENT_DEFAULT; a Grids or Pipeline
job sets `FACE_GRADIENT` to 1 where the job qualifies, 0 where the tick is off and empty elsewhere (the binary keeps
the stream average and says why), the Custom tab adds `--face-gradient` or `--per-tet-gradient` the same way; a
standard-DTFE Grids or Custom job pins `DTFE_FACE_GRADIENT` empty).
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
  them. (The `--gpu` refusal went on 2026-11-09: P11.5, D17.) Refused without `--exact-average` or
  `--ps-volume-weighted`, with `--gpu`, with `--ps-halo-release`, in 2D.
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
- Drivers (as built on 2026-11-07): `FACE_GRADIENT=1` passes `--face-gradient` (both). `run_ps_dtfe.sh` refused it,
  exit 2 before any snapshot, without `-e`, without `PS_VOLUME_WEIGHTED=1` or with the GPU deposit;
  `run_scripts_check.sh` (S) checked all five cases. README knob table, `demo/COMMANDS.md`. Since then: the GPU
  deposit is allowed with it (2026-11-09, P11.5, D17) and standard DTFE's default is the face gradient (D13). Since
  2026-11-11 PS-DTFE's driver default is empty too, the binary's route (D13: the face gradient where the run
  qualifies), with 0 = `--per-tet-gradient`, as in `run_dtfe.sh`; `run_ps_dtfe.sh` refuses `FACE_GRADIENT=1`, exit 2
  before any snapshot, with the sampled deposit (`--sampled`, `DTFE_EXACT_AVERAGE=0` or a lone `SUBSAMPLES` count) or
  without `PS_VOLUME_WEIGHTED=1`, and `run_scripts_check.sh` (S) checks these cases and the two defaults.
Background, measurements and the design review are in the session notes of
2026-11-07 (the velocity-method synthesis): standard DTFE's stored divergence/shear are the exact cell average of the
per-tetrahedron gradient; PS-DTFE's stored ones are the stream average ⟨∇v⟩ (the regular part of the gradient of the
volume-weighted mean velocity); dtfelib's FFT "mean field" adds the caustic sheets but smears them and rings
(TNG100-3-Dark z = 0 slice: correlation with standard 1.000 / 0.015 / 0.495 for standard stored / PS stored / PS FFT).

**Status 2026-11-09 (phase R).** (Superseded the same afternoon: P10.8 done and the full-size check met, the next
status; the face gradient is standard DTFE's default since 14:40 that day, D13.) P10.2's full-size check does not hold;
the cause is the deposit's float32 region frame (P10.8, new). D13 keeps the per-tetrahedron deposit.
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
    fixed-point sums) (superseded: PS-DTFE's labelling also follows each partition's insertion history --
    `tests/ps_partition_padding_check.sh`, P11.3's bytecmp B -- so its velocity moments move at rounding; only the
    density is immune). The test is an hour: form the inverse and the product in double and round the gradient once
    (labelling-independent up to the last rounding), then compare the 3³ and 6³ face passes again; expected: the 2.7e-5
    collapses to the float sum order's ~1e-7. P10.8's second part.
- Restated bound, proposed (the demo's G1 keeps 1e-4): per record `rms_rel ≤ 3e-6`; pooled `above_1e-4 / compared ≤
  1e-5`; every cell above 1e-4 listed; degenerate cells reported apart; no per-record maximum. The clean records pass
  (worst record rms 2.6e-6, pooled fraction above 1e-4 3.4e-6 on the GPU; 7.0e-7 and 2.3e-6 on cpu256); this run fails
  (39 records over 3e-6, pooled 4.8e-4). The battery never runs FACE-CHECK on the GPU deposit or on such a frame.

**Status 2026-11-09, afternoon: P10.8 done** (the owner's go at 11:10; code, the demo gates and bytecmp; the 512³
reruns below; gates on the final tree: the battery 32 of 32 at 13:52, the Linux stages 7 of 7 at 13:47, after a
first battery had shown a stale `libDTFE.dylib` -- rebuilt, the header grew -- and a G6 bound too tight for the
double build's float32 GPU deposit). (Committed 2026-11-09 as `2f428bc`.) Uncommitted in `~/GPU` with the tuner batch.
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
  - Correction (2026-11-11): the sentence "What is left is the GPU's fixed-point unit, one per batch ... and the order
    of the vertex density sums" leaves out the face pass's double sums, which move the gradient, the divergence and
    the shear (both runs had `--face-gradient` and `DTFE_FACE_CHECK=1`, so the face pass's words replaced the
    deposit's), and the single-cell volumes in CGAL's labelling.
  - Against the old grids the new 3³ run differs in 98% of the cells by 2.0e-5 of the density rms (max 4.7% in one
    cell), 1.3e-6 of the velocity's, 2.2e-5 of the gradient's: the old split-to-split scatter, now removed, not a new
    bias: the box-mean density is 0.9999999652 on both new plans (the old 3³: 0.9999999781, the old 6³: 1.0000000297),
    within 3.5e-8 of 1 as every exact run's. Peak memory 42.3 GB max RSS on the 3³ plan against 41.8 GB before (the
    base-cell buffer and the frame), 37.5 GB on the 6³ plan as before.
- Closed for standard DTFE on 2026-11-11 (the owner's decision 2c, the third fix; `src/fixed_units.h`). What moved a
  grid across splits before: the GPU exact deposit's fixed-point unit, one per batch (the gradient grids through it
  only under `--per-tet-gradient`; under the face default through the face pass's double sums in walk order,
  `avg1FacePass`), the CPU exact path's float sums in walk order (on the 64³ demo up to 25 last places of the density
  and 58 of the velocity between one domain and 3³ partitions), the vertex densities (double sums in cell order) and
  the volumes of single-cell tetrahedra (CGAL's labelling), the last two as rare flips. Now every grid has one unit
  2^-K for the whole run, set before the first batch from global quantities (the cell volume, the mean density, the
  particles' largest velocity component and scalar): the GPU's exact batches sum in 128-bit words (four 32-bit limbs
  per component) and need no bound pass, the CPU exact average and the face pass sum in 64-bit words with an exact
  carry, the vertex densities are integer star sums, and the single-cell volumes come from a canonical vertex order;
  each cell is rounded to Real once. A share that is not finite or reaches its sums' cap (2^94 units; 2^118 in the
  face pass) is left out and counted. Single build: the double build's CPU sums stay double (its GPU exact batches
  take the same words). Under `--approx-phase-space-density` the scalar grids keep the batch's unit and the CPU's
  float sums (f is formed after the units). `DTFE_RUN_UNITS_LEGACY=1` restores the earlier sums for one release,
  `DTFE_TEST_RUN_UNIT_SHIFT=n` moves the units for the overflow tests; the checkpoint key names both. Byte identity
  across splits still presumes identical Delaunay tetrahedra, which co-spherical lattice snapshots (000) may resolve
  differently per split. PS-DTFE's passes take the same units under `DTFE_PS_RUN_UNITS=1` (S5, built 2026-11-12, off
  by default: the next bullet); the exact merge in words is built (S6, 2026-11-13, the scratch line, not gated)
  behind the developer switch `--ps-merge fixed-point`, which switches these units on too (the bullet after the next).
  - Tests: `tests/split_bytes_check.sh` (28 runs on the 64³ demo. U1: one units line per run, the same whatever the
    split, threads or device; S1-S4: `cmp` of every output at `--partition 1 1 1`, `2 2 2` and `3 3 3` on the GPU and
    the CPU, at 1 and 6 deposit threads, with `DTFE_FACE_CHECK=1` and under the parallel insert, and the GPU pair
    under `DTFE_RUN_UNITS_LEGACY=1` must differ; O1: units 2^17 (CPU) and 2^40 (GPU) finer, no share left out; O2:
    2^60 finer, the cap's warning. The double build: SKIP), `tests/dtfe_resume_check.sh` D15 (a resume across the
    switch or the shift is refused), `tests/dtfe_face_gradient_check.sh` G6 (`cmp` of every output) and G7.
  - The legacy byte gate (`tests/dev/bytecmp.sh` under the switch) pins `DTFE_FACE_GRADIENT=0`, so it does not run the
    face pass: the face route under the switch is compared with the earlier binaries by hand.
  - Measured 2026-11-11 against the commit before (single build unless named). Under the switch:
    `tests/dev/bytecmp.sh` 261 files compared, 0 differ, and the 64³ demo's face default runs (CPU and GPU, one domain
    and 3³ partitions) byte-identical to the earlier binary's, 24 of 24 files. Without the switch 28 of the 261 files
    move, all in standard DTFE's exact rows (CPU 8 files, 2D 5, GPU 5, GPU partitioned 4, double GPU 6): no PS-DTFE
    row, no sampled row, not the double build's CPU row. How far, on the demo: a CPU run's density up to 25 last
    places and its velocity up to 43 (2e-4 and 3e-6 of the rms, at the largest cells); a GPU run's density and
    velocity not at all; the face gradient grids up to 4e-13 of their rms. In the byte comparison's rows
    (`--per-tet-gradient` gradients; 64³, the 2D row 128²): the gradient grids up to 8e-6 (CPU) and 5e-8 (GPU) of
    their rms, the T-web's eigenvalues up to 3e-5 (CPU), one velocity cell of the GPU row by one last place. The
    earlier binary across splits on the demo, one domain against 3³ partitions: on the CPU 60% of the density cells
    (up to 25 last places) and the velocity (up to 58), on the GPU only the face gradient grids (up to 4e-13 of their
    rms). The new binary: no file differs between the two splits on either device; a CPU and a GPU run differ in
    density and velocity only, by up to 5e-6 and 6e-7 of the rms (1.5e-4 and 3e-6 before). Under
    `--approx-phase-space-density`: density and velocity the same at both splits, the scalar grid not (CPU 60% of the
    cells, up to 38 last places; GPU 0.6%, up to 42). The tessellation cache: a run under the switch on a folder of 16
    new caches writes its own 16 beside them and gives the earlier binary's grids. The planner: 34 of the 127 lines of
    `tests/dev/tune_report_set.sh` move (all standard DTFE), 29 under the switch. The suites above and the battery
    passed (30 of 30 stages; its three build stages left out, the tree built clean without a warning). Not run: the
    Linux check and any 512³ run.
  - At full size (2026-11-12): TNG100-3-Dark z = 0 at 512³ on the committed binaries (30bae11), the planner's own 3³
    split and then `--partition 6 6 6` named, one run at a time: all 10 output files are byte-identical between the
    two. 3³ took 521.89 s at 33.93 GB of peak footprint, 6³ 1241.59 s at 26.34 GB (2.38 times as long for 7.59 GB
    less). The model predicted 35.65 and 33.73 GB; the measured peaks lie 1.71 GB (5%) and 7.39 GB (22%) under them,
    the second in a run whose post-processing set the peak (24.79 GB before it) on a split the planner does not choose
    here. No GPU retry or fallback line in either run.
  - What the units cost (2026-11-12, measured for the comparison with PS-DTFE's): the same snapshot at 512³ on the
    committed binaries (ff565f7), the production field list on the GPU, the planner's own 3³ split, four runs in
    turns, one at a time, the machine in use meanwhile: 496.93 s and 471.75 s in the run-wide units against 453.84 s
    and 459.61 s under `DTFE_RUN_UNITS_LEGACY=1`, +6.0% of wall time on the means (+9.5% and +2.6% by pair) and +8.6%
    of user time (+8.8% and +8.4%), at 33.56 to 33.78 GB of peak footprint in all four. Each arm's two runs are
    byte-identical in all 10 files; 2 of the 10 are the same in both arms. No GPU retry or fallback line in any run.
  - Open: the device's `maxBufferLength` is reached at half the cells of before (16 B per component: a sub-domain of
    over about 1e8 cells with `--per-tet-gradient` or `DTFE_FACE_CHECK=1`, which auto plans do not make). Settled
    2026-11-13 against the code (the scratch line's `71ab973`, not gated): under `--approx-phase-space-density` the
    planner booked CPU words for the scalar grids, which keep their Real sums there; the term now books a grid's words
    only where the grid has a run-wide unit (`RunUnits::has`, the loop's own test), and none in a MY_SCALAR build
    (`auto_tune.cc` `bCpuWordCell`; `tests/dtfe_auto_tune_check.sh` O). Only CPU exact plans with those fields, or
    with a grid whose unit lies outside 2^-120 .. 2^120, move: down by 8 B per component of such a grid (32 B per cell
    with both scalar fields, by arithmetic) (to be measured: which lines of `tests/dev/tune_report_set.sh` move, against
    `39410e2`). A GPU run's sub-domain that falls back to the CPU is no under-booking: its words come after its GPU
    attempt released the arrays, the device grids and their host copy, which the plan books at 2 x `bGpuCell` per
    cell, 8 B per component of grids that include every grid the words are for (the opt-in `DTFE_GPU_SHARED=1` grid
    stays allocated meanwhile). The star sums' 8 B per vertex is a member of every standard vertex
    (`vertexData::_volAccHi`), held under `DTFE_RUN_UNITS_LEGACY=1` too and booked in every mode, so a memory-bound
    plan under the switch differs from the earlier binary's because the vertex is larger, not by a booking error (pin
    `--partition` to compare bytes).
- Built for PS-DTFE on 2026-11-12 as an opt-in, `DTFE_PS_RUN_UNITS=1` (S5, the first half of PS-DTFE's part of the
  same fix). Under the switch every pass deposits its mass, its moment grids and its face sums in the units of the
  run's one "fixed-point units (run-wide):" line, on the CPU and the GPU alike: the GPU in 128-bit words (four 32-bit
  atomic limbs per component), the single build's CPU and the face sums in 64-bit words with an exact carry; each cell
  is rounded to Real once. A `--grid-window` run takes the full run's units (the globals record's version 2), so a
  sampled window equals the full run's cells byte for byte. Without the switch each pass takes units of its own in
  64-bit words, as before S5. The merge in words with one close-out followed as S6 (2026-11-13, behind the developer
  switch `--ps-merge fixed-point`: the next bullet); without it the partitions still merge in float, and PS-DTFE's
  grids still follow the split and the device in their last bits. Not built: the face sums once per face (S7, the
  owner's option). (Built on 2026-11-13 as S7, behind the same option: two bullets on.)
  - The default (2026-11-12 in the evening, the owner's decision after the full-size numbers below): the first build
    had the run-wide units as PS-DTFE's default and the per-pass units behind `DTFE_RUN_UNITS_LEGACY=1`; since the
    decision the per-pass units are the default and the run-wide ones come with `DTFE_PS_RUN_UNITS=1`
    (`src/fixed_units.h` `runUnitsLegacy`; `DTFE_RUN_UNITS_LEGACY=1` beats it), part of `--ps-merge
    fixed-point` from S6. A default run still measures the units, prints their line with a note that the passes sum in
    units of their own, and writes them to the globals record; its checkpoint key is the per-pass one, so a resume
    across the switch is refused; the planner books the 128-bit device words, and the run refuses a CPU pass of 2^32
    cells or more, only under the switch. In the records below "run-wide" is the first build's default and "per-pass"
    its `DTFE_RUN_UNITS_LEGACY=1` path, the two that the switch now selects. After the turn: to be measured: way 2 gate.
  - Speed: the first build with these words took 37.6% longer at 512³ (1853.15 s against 1346.82 s, the run named
    below). Four items took more than half of that back without moving a byte of the run-wide mode (on the per-pass
    path the third one gave back the bytes of before, which the first build had missed in 19 cells of one 512³ grid,
    the dispersion tensor): the host's add takes an inlined 64-bit path for a share that fits it; a word becomes Real
    by an exact power of two; the PS kernels are compiled once per word size (`PS_FX_WORDS`), so on the per-pass path
    the device runs the 64-bit kernels of before; and the kernel adds a share below 2^63 units in one piece
    (`addShare`, `metal/fixed_words.metal.inc`), which touches the two upper limbs only when the carry or the sign
    reaches them. Most of what is left of the cost is on the GPU, in the adding pass of the exact kernel (at 256³ on
    one domain 15.18 s busy against 9.92 s): 16-byte words instead of 8-byte ones, and the adds to the upper limbs.
  - Tests: `tests/split_bytes_check.sh` (PS-DTFE's part, which runs under `DTFE_PS_RUN_UNITS=1`. U1: one units line
    per run, the same at 1, 2³ and 3³ partitions on both devices; PO1: units 2^20 finer, no share left out; PO2: the
    cap's warning, which names the shares left out; P4: a window's cells byte-identical to the full run's; P1 to P3,
    P5 and M1, the byte identity across splits, the resume and the mixed devices: under the word merge since S6,
    the next bullet),
    `tests/ps_window_check.sh` (the globals record's version; by default a cached window run's "none" units line and
    the window's sums within 1e-8 of the peak; under the switch its units line and the sampled CPU window's bytes),
    `tests/ps_auto_tune_check.sh` (the booking of the device words, per pass and under the switch),
    `tests/ps_fixed_moments_check.sh`; `tests/ci_suite.sh` runs the window and the moment checks in both units.
  - Measured 2026-11-12 on the first build, against the commit before (single build unless named). Per-pass:
    `tests/dev/bytecmp.sh` 261 files compared, 0 differ. Run-wide: 64 of the 261 files move, all of them PS-DTFE's: no
    standard-DTFE file and not the double build's CPU rows. How far, in the byte comparison's rows (64³; the largest
    difference in units of the grid's rms): a GPU exact row's density in 18.8 to 20.5% of the cells by at most 4 last
    places (3.4e-6), its velocity in 48% (5.2e-7), its dispersion grids in up to 68.9% (3.2e-6), its tidal eigenvalues
    in up to 30.5% (3.0e-6), its gradient grids in up to 99.5% (1.8e-5); the double build's GPU exact row's density in
    86.4% of the cells (1.7e-7), its gradient grids in nearly every cell (6.0e-6); a CPU exact row's density in no
    cell, its velocity in at most 5 cells, its dispersion grids in at most 822 (7.0e-7), its gradient grids in up to
    57.3% (1.1e-6); a sampled row only in the grids made from the velocity gradient (2.1e-7). On 64³ crossed waves
    (exact; CPU and GPU, one domain and 3³ partitions): per-pass all four runs are byte-identical to the earlier
    binary's, 7 of 7 files; run-wide the devices and the splits are as far apart as before, as expected until S6
    (density, the cells that differ and the largest difference: CPU against GPU on one domain 97.6%, 6.1e-5 of the
    rms; one domain against 3³ partitions 19.3%, 6.8e-6 on the CPU and 19.4%, 3.4e-6 on the GPU; the same with either
    binary). The planner: 42 of the 127 lines of `tests/dev/tune_report_set.sh` move run-wide (every one a PS-DTFE GPU
    case), none per-pass. Passed on that build: 25 suites and 9 on the per-pass path, the battery (30 of 30 stages),
    the docs drift check, and the Linux check (build, core, double, 2d, variants, old Python, ruff: all passed).
  - At full size (2026-11-12, the first build): TNG100-3-Dark z = 0 at 512³, exact with the face gradient on the GPU
    and the production field list, `--partition 6 6 6 --max-concurrent 4` named, one run at a time. Run-wide: 1579.66
    s against 1346.82 s with the commit before (+17.3%), at 34.66 GB of peak footprint against 33.62 GB; 18 of the 28
    output files differ. Per-pass: 1364.38 s (+1.3%) at 33.03 GB, all 28 files byte-identical to the commit before's.
    Run-wide the GPU deposit phases sum to 44% more (2259.3 s against 1571.3 s, 215 passes each), the CPU deposit
    phases to 18% more (1363.6 s against 1159.6 s). The model predicted 41.51 GB run-wide (the measured peak lies 17%
    under it) and 38.26 GB per-pass (14% under). Left to itself the auto-tuner plans, run-wide, the same 6³ split with
    3 passes in flight (37.83 GB predicted; 4 in flight would be 41.51 GB against a budget of 40.85 GB) where the
    commit before plans 4 (38.26 GB): 1578.62 s at 33.16 GB of peak footprint (12% under the prediction), the pinned
    run's bytes in all 28 files, so the fourth pass in flight buys no time here. At 256³ (128³ crossed-wave particles,
    exact on the GPU, three timed runs per arm in turns): run-wide 46.25 s against 40.02 s on one domain (+15.6%) and
    32.43 s against 28.80 s on a 3³ split with 4 in flight (+12.6%); per-pass +0.5% and +0.6% with the earlier bytes
    in all 16 files; on one domain the pass's GPU checkpoint holds 6568 MB run-wide against 3347 MB.
- Built for PS-DTFE on 2026-11-13 behind the developer switch `--ps-merge fixed-point` (S6, the second half of
  PS-DTFE's part of the same fix; the scratch line's `f2aa3c1` and `d13425a`: built and smoke-tested, not gated, not
  measured). In the single 3-D build (`src/fixed_units.h` `runUnitsWordMergeBuild`) the switch turns the run-wide
  units on, as `DTFE_PS_RUN_UNITS=1` does (`runUnitsSetPsFixedMerge`, said once per `DTFE()` call in `runUnitsSetup`,
  before the units line, the planner, the checkpoint key and the passes read the mode; the units line ends with a note
  saying so), and every pass hands its exact sums over unconverted (`ps_interpolation.cc` `psHandOverWords`;
  `quantities.h` `PSWordGrids`): the mass, the moment grids and the face sums as 64-bit words with their carry and K,
  the stream counts as ints, the masks as before. The merge adds them as integers as each partition finishes
  (`ps_fixed_merge.h` `addWords`: owned adds, the partition's carries at their full-grid components, `fxCarryFinish`),
  in the run's units, with no table of its own, so nothing is ordered, parked or spilled. ONE close-out after the loop
  converts each component once and normalizes (`DTFE.cpp` `psCloseOutMoments`: `Quantities::psWordsClose`, then
  `normalizePhaseSpace`); a one-domain run given the switch (no split, or `--partition 1 1 1`) closes out its own
  pass's words the same way (`psOneDomainCloseOut`; the pass defers by the switch, not by `psDeferNormalization`,
  which the tessellation cache's key names). The MOMW / DW alias of the mass words holds on the merged words (DW is
  copied from the mass at the close-out where the density stands for it).
  - The periodic seam (`d13425a`): a periodic run's ownership seam is a function of the Lagrangian padding
    (`subpartition.h` `psPeriodicSeam`), and a one-domain run padded by the Eulerian fraction with its 10% floor, the
    partitions by `--ps-lagrangian-padding` mean spacings, so a tetrahedron between the two seams was deposited from
    another periodic image and its last bits followed the split. Under the word merge the one triangulation now pads
    as the partitions do and takes their padded period (`DTFE.cpp` `psLagrangianPeriodicSetup`, through the helpers
    `psLagMeanSpacing` and `psLagSpacingPadding` that `PSPartitionPlan` uses too; a `Lagrangian padding = ...` line
    says so); every other run keeps its padding and its bytes. With `--ps-lagrangian-padding 0` the partitions'
    padding follows the split, there is no one seam, and a one-domain run and a split are not promised the same bytes.
  - So, by construction, a run whose partitions all deposit on the CPU, or all on the GPU, is to write the same bytes
    at every split, one domain included (not under `--ps-lagrangian-padding 0`), at any concurrency and completion
    order (to be measured: the suites below, and one 512³ pair). A run that mixes the devices (`--gpu-share` below 1, the
    performance mode's automatic share, a GPU fallback) merges exactly too but stays split-dependent. The grids differ
    from the default merge's at float rounding.
  - Checkpoint and refusals: the checkpoint holds the words (`x.fxw.<side>.<accumulator>`), the carries (`x.fxk...`),
    the counts (`x.fxn.<side>`) and their K and sizes (`x.fxwords.state`); the key's `units=` line lists every
    accumulator's K (`partition_checkpoint.h` `wordMergeKeyUnits`), and `--resume` refuses a checkpoint of other units
    or of the other of the two fixed-point merges, naming both lines. `--resume` without `--ps-merge` is refused by
    name, as before S6: it does not take the mode from the checkpoint. Refused at start-up: a PS-DTFE grid without a
    run-wide unit (outside 2^-120 .. 2^120), `DTFE_GPU_FLOAT_MASS=1`, `DTFE_GPU_FLOAT_MOMENTS=1`; as before, the
    double build and `--grid-window`. Kept: the merge of the passes' rounded grids (9 B per component) under
    `DTFE_RUN_UNITS_LEGACY=1` and in the 2-D build.
  - Memory, booked by the planner (`auto_tune.cc` `AutoTunePS::wordMerge`, `bCellLane`, `oneDomainPeak`'s word walk,
    `autoTuneWordCloseOutBytes`): 8 B per summed component and full-grid cell during the loop (4 B per stream count)
    and each partition's words until its merge; not booked: the carries (16 B per carried component). The GPU face
    kernel (`DTFE_GPU_FACES=1`) does not take a pass under it (the run-wide units keep the CPU walk), and the
    planner's time objective (P8.8) keeps the memory plan. Without the switch no byte and no plan is to move
    (to be measured: `tests/dev/bytecmp.sh --gpu-exact` and `tests/dev/tune_report_set.sh` against `39410e2`).
  - Tests: `tests/split_bytes_check.sh` (PW: the switch's units line is `DTFE_PS_RUN_UNITS=1`'s with its note; P1-P3:
    the word merge at 1, 2³ and 3³ partitions, at `--max-concurrent` 1 and 4, in another completion order, in the copy
    and the slab mode and at the auto-tuner's split, exact on both devices and sampled with `--ps-vertex-mass`, `cmp`
    of every output per device; P5: a resume; M1: a mixed run, reported only), `tests/ps_fixed_merge_check.sh` (the
    word merge by default, the merge of before under `DTFE_RUN_UNITS_LEGACY=1`; X2's reference is the one domain; X4
    the carries at units 2^20 finer, the start-up units refusal, and the 9-B overflow stops under
    `DTFE_RUN_UNITS_LEGACY=1`; X5 the refusals, the float knobs among them; X6 the resume and its refusals: without
    `--ps-merge`, of a default run's checkpoint, of other units and across the two merges both ways),
    `tests/ps_mass_exact_check.sh` X1 (2³ writes the one domain's bytes), `tests/ps_partition_padding_check.sh` P1-P4
    under the switch (every split and padding variant byte-identical to its `--partition 1 1 1` run),
    `tests/ps_resume_check.sh` C17, `tests/ps_resume_points_check.sh` R11.
  - Measured: nothing yet (to be measured: the suites above, the battery and the Linux check; the byte identity at 512³ --
    TNG100-3-Dark z = 0, exact with the face gradient on the GPU, the planner's split against `--partition 6 6 6`; the
    wall time and the peak footprint against the default merge and against the merge of before at 512³, with the
    planner's predicted peak (`DTFE_AUTO_TUNE_DEBUG=1`), the report line's carried components and the `merge time:`
    line). (Later on 2026-11-13, with S7 on top: the next bullet's "Measured", on small data.)
  - Open: whether the switch is hard-wired or removed (the measurement above decides it: the two-mode rule under "Open
    items"); the refit of `kPSLaneShare` / `kPSLaneShareHold` under it; a booking for the carries if the 512³ run
    shows many; X2's bound against the default merge (2e-6 of each grid's peak, kept from the merge of before), not
    yet held against the word merge; the exact-lattice ties of `ps_mass_exact_check` X1 (if its one-domain identity
    fails while the padding and split suites pass, the lattice's co-spherical points are the first suspect).
- Built for PS-DTFE on 2026-11-13, behind the same developer switch `--ps-merge fixed-point`: the face sums once per
  face (S7; the scratch line's `3947bb9`, on top of the walk default `b62eb94`: built and checked on small data, not
  gated). The merged grid's words hold each face once (`quantities.h` `PSWordGrids` `faceOnce`, `cellComps`; on a
  grid that is not periodic, a run without `--periodic` such as a `--box` run, the boundary planes `plane[f][a]`); a
  pass keeps both faces per cell. The merge adds every upper face, skips a lower face whose twin the pass holds (the
  same share, written by `psDepositFaces` to the cell below in the same pass) and routes one without a twin to the
  cell below's upper face or to the plane (`ps_fixed_merge.h` `addFaceSlot`; the twin rule is a copy of
  `psDepositFaces`' `local()`, `psFaceLocal`, and a comment at the original asks to keep the two in step), so every
  share counts once whatever the split; where a pass held only one of a face's two cells, S6 missed that share in
  the other cell. A one-domain run's face sums take the same rule before its close-out (`oneDomainFacesOnce`, from
  `DTFE.cpp` `psOneDomainCloseOut`). The close-out keeps them as words and the gradient reads them so
  (`quantities.cc` `psFaceGradientCellWords`, one scalar conversion shared with `psWordsToGrid`), with no Real face
  grid; they are freed with the gradient (`psFaceGradientEnd`). The checkpoint holds the once layout and the planes
  (`x.fxw.<side>.<face sum>.plane<a>`); a checkpoint of S6's layout is refused by its state's size
  (`restoreWords`). `DTFE_DEBUG_FACE_TWINS=1` (a test hook) compares every twin pair word for word and counts the
  routed (lower) words, one line per side on stderr. The planner books the merged face words once per face, 120 B
  per cell and face pass with the planes beside them, and a pass's both faces, 240 B, in its lanes (`auto_tune.cc`
  `autoTuneWordFaceFullBytes`, `autoTuneWordFacePassBytes`, `bCellLane`, `oneDomainPeak`'s word walk: the
  destination map and the once words of `oneDomainFacesOnce`, no Real face grids, the words freed at the gradient).
  On AB's 64³ probe (2³ partitions, five face fields; `DTFE_AUTO_TUNE_DEBUG=1`) the deposit phase books 294 B per
  cell periodic (bCell 294) and 296 in a `--box` (the planes' 1.875 more, by arithmetic); beside it the
  normalization line names 60 B per cell on the line, the regular part's twins 56 and the stream average's
  divergence 4 (a term of `eba8096`; S7's own tree, without that commit, prints the twins' 56).
  - Tests: `tests/ps_fixed_merge_check.sh` X8 (the `_a` face set periodic and in a `--box` at 1, 2³ and 3³, both sides
    periodic at 1 and 3³, byte for byte; the twin check's counts; a kill and a resume with the boundary planes) and
    X7's GPU face pair; `tests/ps_auto_tune_check.sh` AB (AB0: both plans are the word merge's; AB1 and AB2: 294, 294
    and 60 periodic, 296, 296 and 60 in a `--box`, by hand; `b6fa0b8` made AB read the line's normalization text);
    `tests/split_bytes_check.sh` P1 and P5 run the face sums unchanged.
  - Measured (2026-11-13, small data only). On S7's own tree (base `d13425a`, before its merge onto the line): the
    builds (single, double, 2-D, library) exit 0 without a warning; the 16 reference runs of the word merge (one
    domain and split, with and without the face gradient, on the GPU and the CPU) write S6's bytes in every output
    file, and one domain against the splits differs in no file; `tests/ps_fixed_merge_check.sh` 41 passed, 0 failed
    in the single build (X8 included), 1 passed in the double build (X5 only), 20 passed under
    `DTFE_RUN_UNITS_LEGACY=1` (X8 skipped); `tests/ps_auto_tune_check.sh`, `tests/ps_face_gradient_check.sh` and
    `tests/split_bytes_check.sh` pass in the single build; `tests/dev/bytecmp.sh` 261 files, 0 differ. On the line
    (`3947bb9`'s code, the walk beside the deposit the default): the same 16 runs exit 0 with S6's bytes in every
    file and no file differing between one domain and the splits; `ps_fixed_merge_check.sh` with the same three
    counts; the other suites as P11's status of 2026-11-13 on the default lists them; S7 moves no line of
    `tests/dev/tune_report_set.sh` (none has `--ps-merge fixed-point`). Not run: the battery, the Linux check, any
    512³ run (to be measured: the peak footprint and wall time of a 512³ face run under the word merge against S6's and
    the default merge's, with the planner's prediction).
  - Open, each checked against this tree. (1) The twin line's count of non-zero routed words is 0 wherever S7 equals
    S6, but that is not sufficient: it counts the twinless LOWER words, and a pass that holds a face's lower cell but
    not its upper one leaves a twinless upper share, added in its cell as S6 did and not counted (`addFaceSlot`
    counts only in the lower faces' sweep); the 16 reference runs are identical, other data need not be. (2) The
    one-domain destination map: `oneDomainFacesOnce` builds `Merge::cellMap`'s identity map, one size_t per cell (an
    estimate by arithmetic: 1.07 GB at 512³), transient and booked; an identity flag on `addFaceSlot` would save it
    (the owner's choice). (3) The mass flux divergence grid is booked at 8 B in the loop under the word merge
    (`autoTuneBytesPerCell`'s face term adds one Real for it, which the word branch scales to 8 B): S6's booking, by
    reading 4 B per cell and face pass too much. (4) A resumed run's twin counts cover only the merges of its own
    process (by reading, not run). (5) Not checked: GCC and the Linux stages; `scripts/run_ps_dtfe.sh`'s comment
    "30 more numbers per cell in the merged grids", true for the default merge and in bytes for the word merge (15
    words of 8 B), left alone. Settled: PS-DTFE refuses `--partition-index` at parse time (`src/user_options.cc`), so
    the word merge never meets a single partition's run; X8's data-dependent counts and the `--box` 3³ identity,
    never run at S6, passed (above).

**Goal.** One implementation for both estimators of the velocity gradient of the cell-averaged velocity field, exact
to the interpolant, with no spectral ringing:
- By Gauss, the cell average of ∇v is (1/V) times the surface integral of v n over the cell's faces. The binaries
  deposit **exact face integrals** (the same clipping idea as the exact cell average, on face planes) and take
  `g_ij = (F_i(+j face) − F_i(−j face)) / (A Δx_j)` after the merge.
- **Standard DTFE:** the interpolant is continuous, so this equals today's exact gradient (the check, below).
- **PS-DTFE:** (Changed 2026-11-07: P10.3 keeps each cell's lower and upper face sums per axis, as the quotient form is
  not conservative in caustic cells; the P10 status.) with S = Σ_streams v_s and n = the stream count, the face sums S_F
  and A_F (covered area) give the gradient of the volume-weighted mean velocity in the quotient form `(ΔS_F − a_vel
  ΔA_F) / Σ_s V_s`, exact where n is constant through the cell and exact when integrated over a region elsewhere (the
  caustic sheets enter as flux differences, at grid resolution, with no ringing). The stored regular part ⟨∇v⟩ stays
  available, so the difference is the caustic contribution.
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
  stores), the regular part under new names, both from one run. An explicit `--face-gradient` is refused with the
  sampled deposit, without `--ps-volume-weighted` and with `--ps-halo-release`; since 2026-11-11 (D13) a run that
  names no route keeps the stream average in those cases and its header's 'velocity gradient' line says why. *Status
  2026-11-11: the regular part's divergence was written ('.a_velDivStreams'); its gradient, shear and vorticity follow
  in P10.10 (D12).*
- `--ps-window`: N+1 face layers for the window only.

**P10.4 PS-DTFE: mass flux.** P_F = Σ ∫ ρ_s v·n dA and M_F = Σ ∫ ρ_s dA per face; `div p` as a new grid, and the
cell momentum Σ ρ_s v_s (so p / ρ gives the mass-weighted mean velocity without a second run). ρ_s per tetrahedron:
its mass over its Eulerian volume (uniform), as the deposit already uses.

**P10.5 Memory model and tuner.** (As built: the face pass's 9 doubles a cell for standard DTFE, 30 values a cell and
their int64 words for PS-DTFE; the P10 status, P10.5.) Face accumulators: standard 3 × 3 floats per cell (as the
9-component gradient); PS 3 × (3 + 1) [+ 3 × (3 + 1) for P10.4] int64 words per cell, about 96–192 B/cell: whole box at
512³ or a window. Book them in `autoTuneBytesPerCell` and the GPU-free PS plan.

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

(Superseded: at 512³ the two exact PS deposits tie, P11.1 (2026-11-09); the face sums run beside the GPU deposit, P11.5;
a GPU kernel for the face sweep is the next large item, the owner's answers under the decision table.) Effort: L. GPU
ports (Metal `sw_poly_rect` already intersects z-sections with rectangles) only after the CPU path is
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

**P10.10 PS-DTFE: the regular part's own gradient, shear and vorticity (D12, decided 2026-11-11: 3b).** — *built
2026-11-11 (not committed); P10.3's "the regular part under new names", which had kept only the divergence.*
- What is there: every PS deposit (CPU sampled and exact, Metal sampled and exact, fixed-point and float moment sums,
  the double and the 2D builds, the window, the partition and the fixed-point merge, the checkpoint, the CPU deposit of
  a pass the GPU declines) sums the stream average ⟨∇v⟩ in full, 9 components (`velocity_gradient`, `PS_FX_GRAD`).
  Without the face gradient the run writes it as the velocity-gradient family; with it the normalization
  (`Quantities::psFaceGradientCell`) kept its trace as `.a_velDivStreams` and wrote the mean flow's gradient over it.
- The change: the same cell step keeps, before that overwrite, the regular part's gradient, shear and vorticity --
  each only where the run writes its mean-flow twin (`gradient_a`, `shear_a`, `vorticity_a`;
  `User_options::psStreamsTwins`, set by DTFE_setup) -- by web_fields.cc's own derivations
  (`Quantities::psStreamsTwinsCell`), written as `.a_velGradStreams` (9), `.a_velShearStreams` (5),
  `.a_velVortStreams` (3): the bytes a `--per-tet-gradient` run stores under the plain names. No deposit, kernel,
  buffer, fixed-point unit, merge or checkpoint change; the sampled deposit and the 2D build (no face sums) store the
  regular part under the plain names already. The u pass gets none, as `.a_velDivStreams`.
- Memory: 36 / 20 / 12 B per full-grid cell (512³: 4.83 / 2.68 / 1.61 GB; the double build twice), made in the
  normalization beside the face sums and kept to the output. The planner books them in the post phase and as the
  normalization's own peak (`normPeak`: the deposit phase's grids plus them, no partition in flight), the dynamic
  concurrency reserves them (`reserveEnd`) and the scratch volume's check counts them (`autoTuneStreamsTwinBytes`).
  The production face plan (6³, 4 in flight, 38.26 GB predicted) does not move: its post phase stays below its deposit
  phase, its normalization below its loop (the plan probe, 2026-11-11: the same line before and after; under
  `DTFE_AUTO_TUNE_DEBUG=1` the planner prints `PS normalization: the regular part's twins 68 B/cell beside the deposit
  phase's 240 (bCell 276), 19.622 GB in all`). 10 of the 16 tune-set lines naming `--face-gradient` predict more, none
  with another split or count in flight: the normalization is now their peak (the single build's four production-field
  lines at grid 1024: 259.97 to 273.61 GB before, 324.39 GB now, the fixed term's 251.37 GB and 68 B/cell).
- dtfelib: `FIELDS` `gradient_streams`, `shear_streams`, `vorticity_streams`; `FieldSet.load(name,
  stream_average=True)`, `has(name, stream_average=True)` (the `_a` grids).
- Tests: tests/ps_face_gradient_check.sh P15 (and P14's file list); P6 holds a window's twins to the window's own
  fixed-point units (single build: 7.7e-11 of the gradient twin's rms measured, in 2 of the slice's 9216 values, the
  same two with `--per-tet-gradient` and with the binaries of the commit before; double build: 0), the contract
  tests/ps_window_check.sh has for the stream average; tests/generate_ps_test_data.py `--linear-velocity`.
- Open: Q4 (`.a_velDivStreams` and `.a_massFluxDiv` unbooked in the post phase: booked since 2026-11-13, P10.11), Q5
  (no Explore map of the `*Streams` files: listed since 2026-11-13, P10.11), Q9 (the zero fill of scratch-backed
  twins); the exact zoom writes a `.a_velGradStreams` plane it does not show; a face run's unaveraged gradient grids
  carry no marker (the u pass takes the face sums, only the `_a` pass writes `.a_velDivStreams`): dtfelib reads their
  route from the run log since 2026-11-13 (P10.11).
- Gates (2026-11-11, one batch with D13's PS default): the single, double and 2D builds and the library exit 0 (a
  clean build); bytecmp `--gpu-exact` against the commit before: 261 files, 0 differ (with
  `DTFE_PS_VERTEX_ORDER_LEGACY=1` against the binaries before the canonical order: 261 files, 0 differ); a face run's
  files against the commit before, on the CPU, on the GPU and in 2³ partitions: 11 files each, 0 differ; a default
  run against the spelled-out face run: 14 files, 0 differ; the tuner's report set: 127 lines, 32 move (above and
  D13); tests/ps_face_gradient_check.sh in both precisions with P6 as above (its first version asked the twins for
  byte identity and failed in the single build), `ps_auto_tune_check`, `ps_window_check`, `gpu_scalar_check`,
  `dtfe_face_gradient_check`, `flag_spelling_check`, `run_scripts_check` and the python suites; the Mac battery
  without its three build stages: 28 of 29 stages passed, `ci_core` failed at P6's first version and passed in a
  re-run with the corrected test (1330 s). Not run on this batch: the Linux check.
- Effort: S–M.

**P10.11 The face default's left-overs (2026-11-13; the scratch line's `eba8096`: built and smoke-tested, not gated,
not measured).** -- *the "Left open" list of the face-default batch (D13 for PS-DTFE, D12).*
- Booking (Q4): `.a_velDivStreams` and `.a_massFluxDiv` (`Quantities::psFaceGradientBegin`, one Real per cell each,
  every face pass, the u pass too, which writes neither) in the post phase (`autoTuneFaceOutputBytes`), and the
  divergence at the normalization (`normR` = the twins + it: `normPeak`, `reserveEnd`, the scratch check; the mass
  flux is in `bCellDeposit` already) and in `oneDomainPeak`. By arithmetic a face run's prediction rises by 4 B per
  cell (8 in the double build) and face pass where the normalization is its peak, by 8 B (16) where the post phase is
  (T-web or V-web fields with the face gradient). No output byte moves (to be measured: which of the 127 lines of
  `tests/dev/tune_report_set.sh` move against `39410e2`: the face lines whose peak is the normalization are expected
  to, with no other split or count in flight).
- Over budget: the planner's warnings that a plan does not fit name the way out of the face sums where a pass makes
  them -- PS-DTFE's irreducible warning (`(f) the per-tetrahedron gradient: ...`, with the bytes per cell and the GB
  the face route adds to the full grids), its "even a single N^3-split partition" warning, standard DTFE's "no split
  fits" (where `--per-tet-gradient` books less than the face pass's 72 B per sub-domain cell: not in the single
  build's CPU exact average, where the per-tetrahedron route deposits 72 B of gradient words). Text only. Not covered:
  a plan over the budget only through its normalization peak prints no warning.
- `--interlace`: correct by reading (`main.cpp`: the second pass, into Quantities of its own, gives the density only;
  every face state is per `DTFE()` call), so it keeps the face default: `tests/ps_face_gradient_check.sh` P19 and
  `tests/dtfe_face_gradient_check.sh` G8 (every file but `.a_den` byte for byte a run without it).
- Python: the launcher's PS-DTFE jobs pin `DTFE_FACE_GRADIENT` empty (Grids, Pipeline, Custom and the memory checks;
  `runspec.py`, `runspec_pipeline.py`, `runspec_custom.py` `run_env`), so an exported value can neither change a job's
  route behind its command nor make the binary refuse a ticked job that does not qualify; the face tick comes back
  when the linear deposit is unticked again (`app_widgets._exclusive_restoring`; the loaders forget first,
  `_forget_exclusive`); the `--ps-gradient stored` texts say what `stored` reads by the run's route (`dtfelib/cli.py`,
  `environment.py`, `plot_DTFE.py`, `plot_velDiv_den.py`, `gui/grids.py`, Explore's menu); Explore lists the
  `*Streams` twins and `.a_massFluxDiv` after the velocity-gradient family (`grids.FIELD_LABELS`, `DIVERGING`,
  `CURL_FIELDS`); `grids_gb` counts a face job's face sums (30 numbers, 120 B per cell), its two face outputs (8 B),
  the twins of the fields asked (36, 20 and 12 B) and the mean velocity where `velocity_a` is not asked
  (`face_route_on`); the `_u` marker: `io.run_face_gradient` reads the run log's `velocity gradient` line for
  `FieldSet.face_gradient` with `averaged=False` (a folder of a binary before the PS face default, or one without a
  log, reads as before).
- Tests: `tests/py_dtfelib_test.py` (`t_face_gradient_unaveraged`), `tests/py_gui_runspec_test.py` (the PS pin,
  `grids_gb` with the face terms), `tests/py_gui_app_test.py` (the tick comes back), `tests/py_gui_app2_test.py`,
  `tests/py_gui_explore_test.py` (the twins in Explore); `tests/ps_face_gradient_check.sh` P19 and
  `tests/dtfe_face_gradient_check.sh` G8 (to be measured: the Python suites, G-gui, and the two face suites).
- Open: the same stale "`stored` = the stream average" wording in `plot_cosmic_web.py`, `plot_halo_tracking.py`,
  `plot_shear_triaxial.py` and `plot_web_streams.py`; the volume weighting's box stays off after the linear deposit is
  unticked (only the face tick comes back); `grids.OutputSet.face_gradient` has no `_u` rule; `grids_gb` adds the face
  sums to the post phase's web grids, so a T-web or V-web PS-DTFE job's estimate is high (the tab's Check memory gives
  the binary's own plan); a `--face-gradient` typed into the Custom tab's extra options is not seen by the estimate.
- Effort: S.

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
binaries; (Superseded at 16:30 by the default rule below.) the DEFAULT is the parallel insert for PS-DTFE (its
near-lattice points build 2× faster, measured; its
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
partition above the threshold, or forced); the per-vertex plan constant no longer carries it. (D19 answered 2026-11-11:
the decision table.) **Run-to-run byte reproducibility of the CPU deposits and the PS velocity moments is therefore no
longer the default above the threshold (D19, for the owner: keep the rule, or make the sequential insert the default and
the parallel one the
driver's choice for low-concurrency runs).** Found on the way earlier: the plain "timing" run took 616 s because the Colima VM of
the Linux check was still resident, the tuner saw 37.9 GB free instead of 52 and planned 4³ instead of 3³ (the P8
status). Where the exact deposit's grids no longer depend on the construction order (P10.8:
the GPU, the PS fixed-point sums) the insert costs no reproducibility; the CPU float sums and the points still move
at rounding, so the tests that compare runs byte for byte pass the sequential insert explicitly. `--parallel-triangulation`
and a `--sequential-triangulation` opt-out in both binaries.

**D19, 2026-11-11 (option c).** PS-DTFE renumbers every cell in a canonical vertex order right after the tessellation
exists (`buildTessellation`: after the insert, before the vertex densities and the cache save; after a cache load; the
composite server after each of its own cache loads), in `src/CGAL_triangulation/ps_vertex_order.h`: the vertices in
Lagrangian lexicographic order, the last two swapped back where that is an odd permutation of CGAL's numbering (CGAL's
positive orientation is the stream parity every deposit and the kernels read), each neighbour moved with its vertex;
one thread per tessellation (CGAL keeps the compact container's bookkeeping in a neighbour slot). Cost: 0.86 s for
23.1 million cells at 2.1e6 points, on one thread (0.037 s per million; "canonical vertex order" in the phase totals;
that run took 7.2 and 7.0 s against 6.2 and 6.0 s under the switch; at 512³ on 2026-11-12, the face-walk pair of that
day's status: 34.13 s and 34.32 s summed over the 216 tessellations of a run with 4 partitions in flight, against
1342.27 s and 1246.08 s of wall time; no run under the switch at that size). The cusp stencil of `--ps-caustic-cusps`
orders its ring strictly (distance, centroid, vertices) instead of by the cells' addresses.
`DTFE_PS_VERTEX_ORDER_LEGACY=1` keeps CGAL's numbering and the stencil's address order; the partition checkpoint's
settings key names it (`vertex-order=legacy`, partition_checkpoint.h settingsKey), so a `--resume` across the switch
is refused. Gates (2026-11-11): bytecmp `--gpu-exact` with `DTFE_PS_VERTEX_ORDER_LEGACY=1` against the commit before:
261 files, 0 differ; without it 104 of 261 differ (no dt* file; the ps* runs move once, their sizes in the next
paragraph); bytecmp B on the new binaries (`DTFE_TRIANGULATION_THREADS=6`, both sides): 38 of 261 differ (dtp 8, psd
8, psxd 11, psxdg 11), no single-precision ps* file among them; the same B under the switch (control): 102 differ, the
single-precision ps* runs among them (ps 6, psp 6, psg 8, psx 9, psxp 9, psxg 12, psxpg 12); the tuner's report set
unchanged (127 lines); `tests/ps_parallel_check.sh` stage D19 and `tests/ps_tessellation_cache_check.sh` case N (a
cache file in CGAL's numbering is renumbered on load). Still order-dependent: the double build, `DTFE_GPU_FLOAT_*`,
the vertex densities' and c_v's per-vertex sums (cell order), standard DTFE; the units' powers of two come from double
sums in walk order (a flip needs a bound within rounding of a binade edge; the stage compares the units lines).

**D19, the size of the one-time move (measured 2026-11-11).** Most cells move in the last bits; a short tail moves
far, where a near-flat tetrahedron at a caustic has its cell or its gradient decided by the last bits (single
precision). bytecmp's set (48³ crossed-wave points at 64³, the canonical order against the commit before): the CPU
sampled run `ps` a_vel ≤2.1e-4 of the rms, a_velGrad ≤9.6e-4 (159 of 262144 cells above 1e-4, none above 1e-3),
a_velVort ≤0.028 (363 cells above 1e-3, 20 above 1e-2, none above 0.1), no density cell; with partitions (`psp`) 2
density cells (1.1e-3 of the rms) and a_vel ≤3.5e-3. The GPU sampled run `psg`: a_den ≤5.2e-8, a_vel ≤2.2e-5,
a_velGrad above 1e-3 of the rms in 10 cells (4 above 1e-2, 2 above 1; at most 6.4), a_velVort in 479 (58 above 1e-2, 3
above 1; at most 16.5). The agreement of the GPU and the CPU on that sampled gradient has the same distribution in
both numberings (psg against ps, a_velGrad above 1e-3 of the rms: 2263 cells either way; above 1: 604 in CGAL's
numbering, 606 in the canonical one), so the large moves are those cells' conditioning and not an order mismatch
between the extraction and the kernel. The exact runs: `psx` a_vel 3.7e-7, a_velGrad 9.4e-7, a_velVort 1.3e-3, a_den
identical; `psxg` a_den 3.4e-6, a_vel 6.9e-7, a_velGrad 2.9e-4, a_velVort 9.4e-3; with partitions the vorticity 0.010;
`.tetTouch` in 7 to 65 cells, by at most 3. The double runs move at 1e-14 in the density and the velocity on the CPU
(psd, psxd; the gradient ≤3.1e-12, the vorticity ≤7.2e-11, the V-web eigenvalues ≤2.8e-8) and at 4.5e-7 in the density
under the GPU (psxdg). 2.1e6 crossed-wave points at 128³ (exact CPU deposit, parallel insert, two runs each way): the
canonical order against itself, 0 cells differ, and the GPU pair is identical too; CGAL's numbering against itself:
the density in 8 cells (pairs of neighbouring cells that exchange one tetrahedron's mass, ≤0.030 of the rms) and the
velocity above 1e-3 of the rms in 68 cells (7 above 1e-2, 3 above 0.1; at most 0.12); the canonical order against
CGAL's numbering: the density in 106 cells (86 of them by one tetrahedron's mass, ≤0.032 of the rms; the grids' sums
are equal) and the velocity above 1e-3 of the rms in 666 cells (48 above 1e-2, 5 above 0.1; at most 0.31); `.streams`
and `.hidden_streams` identical in every pair. So P11.3's figure for what the insertion order moves (psx a_vel: 5% of
the values, ≤3.7e-7 of the rms) is the 64³ set's: on caustic-rich data two runs in CGAL's numbering already differed
by a tetrahedron's mass in a few cells. The canonical order makes those cells repeatable; it does not make their
tetrahedra better conditioned. The input's particle order, measured once by hand (a shuffled copy of the 48³ set; ps,
psp, psg, psx, psxp, psxg): 0 of 89 files differ in the canonical order, 52 in CGAL's numbering; no test holds this
yet.

**P11.4 The device rule (D18).** (Closed 2026-11-11, D18: the drivers and the launcher pick the device, the binaries
keep `--gpu` explicit; since 2026-11-10 auto puts PS-DTFE's exact deposit on the GPU too, P13.1 item 21.) A GPU build
with a device present deposits on the GPU where P11.1 says it is faster,
the CPU otherwise. The mechanism exists in the drivers: `DTFE_GPU` empty = `auto` (`gpu_auto_resolve` in
scripts/config.sh: the GPU when the binary has one, and for PS-DTFE the CPU when DTFE_EXACT_AVERAGE is on), `DTFE_GPU=0/1`
override, the binaries' `--gpu` unchanged (the suites' semantics hold). What P11.1 adds is the rule's table (the
standard exact average too), which the bench of the final binaries fills in.

**P11.5 The PS face gradient beside the GPU exact deposit (D17).** Done 2026-11-09 16:00 (code; gate: the rebuilt
`tests/ps_face_gradient_check.sh` P11): `psFaceWalk`, one CPU walk over the kept tetrahedra after the GPU deposit
(the deposit loop's filter, mass and gradient; the deferred tetrahedra included, the loop's own face block off under a
GPU deposit), into the face words `psCpuFixedUnits` now allocates under either deposit; the bound walk runs once more
for their units, so a GPU and a CPU run of one pass form the same face words (byte-identical mean-flow divergence).
(P11.6 was dropped the same day; a GPU kernel for the face sweep is the next large item, the owner's answers under the
decision table.) Sequential with the deposit for now (the GPU is idle during the walk; concurrency with the next
sub-domain's deposit is P11.6's pipeline). Since 2026-11-10 (P11.7 items 2 and 5) the single build takes those units
from the bounds the extraction summed, without the second bound walk, and the walk rejects on a copy of the loop's own
span test. Since 2026-11-12 an opt-in switch starts the walk beside the pass's own device call (the status of that
day, below). Since 2026-11-13 (`b62eb94`, the scratch line) the single-precision build does so by default (the
status of that day on the default, below).

**Status 2026-11-09 16:30.** (Committed that evening as `2f428bc`, the next status; D19 answered 2026-11-11, the
decision table.) P11.2, P11.3 (the default rule: the parallel insert for tessellations of ≥ 1e6 points
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

**Status 2026-11-09 21:00.** (Superseded 2026-11-10: D15 and D18 do not wait for a tuner fix, the next status; D18
closed and D19 answered 2026-11-11, the decision table.) The tree of 17:03 was committed and pushed (branch GPU) the
same evening. The bench (P11.1) ran on its binaries and is recorded above; P11.7 lists what it leaves. D15 and D18 wait
for the tuner fix and the reruns of j and l; D19 is the owner's (the rule engaged in neither production configuration at
512³).

**Status 2026-11-10.** (Trimmed 2026-11-11: P11.8's plan gives l 4 in flight.) P11.7 item 1 is done: the plan books a
GPU lane's state after its hold of the device, and the
face sums on their own. The bench's face runs had not exceeded the budget (their max RSS had been read as the peak;
the correction is under the PS table above), so D15 and D18 do not wait for a tuner fix. Run j stands as measured. Run
l ran with 4 in flight, at a peak footprint of 33.4 GB, where the corrected plan chooses 3: whether to run it again at
3, or first to trim the lanes' common over-booking so that the plan allows 4, is the owner's.

**Status 2026-11-10, evening.** P11.7 items 2, 4, 5 and 6 are done (below; the changes sit on `c8dd4fb`): the face
units from the extraction's bounds, the face walk on the loop's own test, the points' system time traced to the
parallel insert's lock waits (no code change), the speed texts. P11.7 leaves item 3: the two pinned runs for the
device rule's table (D18).

**Status 2026-11-11.** (Committed 2026-11-11 as `2d262e4`. The confirming runs were made the same day: P11.8's "The
confirming series".) P11.8 (after P11.7's checks, below) trims the lanes' common over-booking, the open point of P11.7
item 1, and refits standard DTFE's two under-predictions of the mode bench (P13). In code, uncommitted. Run l's plan
allows 4 in flight again (38.3 GB predicted; 33.4 GB measured at 4), and performance's plan on the bench input is the 6
at once of the 783 s hand plan. The runs on the new plans were made on 2026-11-11 (P11.8's "The confirming series").

**Status 2026-11-12 (the face walk beside the device call: an opt-in, in code).** The first step of the face
sweep's item (the owner's answer of 2026-11-11, under the decision table): no kernel, P11.5's walk moved in
time. With `DTFE_FACE_WALK_EARLY=1` a single-precision `--face-gradient` pass on the GPU starts `psFaceWalk` on a
helper thread after the cost sort, as its device call starts, and joins it after the call (`PSEarlyFaceWalk` in
`ps_interpolation.cc`; the helper holds one core of the pool). P11.7 item 2 made that possible: the face units come
from the bounds the extraction summed, so the walk needs nothing of the deposit. The walk's left-out shares go to a
sink of its own and are counted only when its sums are kept, which is when the device call succeeded; after a failed
call the sums are thrown away and the CPU deposit makes them, as before. The switch does nothing in the double build
(its face pass has a bound walk of its own) and under `DTFE_TEST_NAN_SHARE`. The output bytes are the same with and
without it. The planner was unchanged while the switch was opt-in, although such a pass holds its face words during
the device call, which no term booked outside `--grid-window` (whose window term summed them beside the device copies,
for one partition in flight); booked since 2026-11-13 (the status of that day, below).

Gates, on the tree of P10's run-wide units plus this change: the byte comparison against that tree's binaries (261
files, 0 differ); `tests/ps_face_gradient_check.sh` P16 (seven lines in the single build: one domain, a one-cell
`--grid-window`, a 2³ split with 4 in flight, a failed device call on one domain and on the split, a NaN vertex mass,
the NaN hook; one line in the double build); the 12 suites around the PS GPU deposit without the switch (the face and
the retry suite in both builds) and 11 of them again with the switch in the environment, where only
`ps_parallel_check`'s runs take the early walk (2 one-domain logs), so P16 is the coverage; the battery, 30 of 30
without its build stages. Not run on it when this status was written: the Linux check, the battery's three build
stages.

Timed once at 256³ (2026-11-12; 128³ crossed-wave particles, `--exact --ps-volume-weighted --face-gradient --gpu`, the
`_a` density, velocity, divergence and shear; three runs without and three with the switch in turns per plan; the
screen saver was drawing on the machine throughout, for both arms alike). Wall seconds and peak footprint, the three
runs' range (mean):

| plan | wall without the switch, s | wall with, s | saved (means) | peak without, GB | peak with, GB |
|---|---|---|---|---|---|
| one domain | 37.55–38.22 (37.84) | 32.62–32.68 (32.65) | 5.19 s, 13.7% | 12.370–13.149 | 16.215–16.219 |
| 2³, 1 in flight | 37.63–38.52 (38.07) | 31.72–31.91 (31.80) | 6.27 s, 16.5% | 8.304–8.305 | 8.958–8.965 |
| 2³, 4 in flight | 25.32–26.61 (26.06) | 21.93–22.39 (22.20) | 3.86 s, 14.8% | 12.021–13.739 | 13.149–13.669 |
| 3³, 4 in flight | 25.96–26.65 (26.26) | 21.93–22.10 (22.01) | 4.26 s, 16.2% | 10.015–10.074 | 10.325–10.328 |

Every grid is byte-identical within each plan (9 files per run, all 24 runs). On one domain the walk takes 8.3–8.5 s
in either place; with one pass in flight it is over before the device call returns (at 2³ the walks take 9.0–9.1 s
over the 8 passes, under 1e-4 s of it after their calls). The device phase is longer beside the walk: 15.0–15.1 →
18.5–18.6 s on one domain, 15.5 → 18.2–18.3 s summed over the 8 passes, so not all of the walk's seconds are saved;
why the device call slows is not traced. With 4 in flight the walks take 31.6–36.0 s in all at 2³ and 30.4–31.5 s at
3³, 2.0–3.7 s and 2.3–2.5 s of it after their device calls had returned. Memory: the peak rises by 3.07–3.85 GB on one
domain (3.42 GB between the means), by 0.65–0.66 GB at 2³ with one in flight and by 0.25–0.31 GB at 3³ with 4 in
flight; at 2³ with 4 in flight the two ranges overlap. The plan's prediction covered every measured peak (25.78, 9.29,
22.44 and 18.97 GB predicted for the four plans).

**The 512³ pair (2026-11-12, 03:43–04:29).** TNG100-3-Dark z = 0, run l's command (P11.1) with `--partition 6 6 6
--max-concurrent 4` named, which is the plan the planner chooses for it (38.26 GB predicted), and `--face-gradient`
named; one binary for both arms, the switch off and then on; one run at a time, no other run, build or test beside
them, no samplers. Each output set was hashed and then removed.

| | without the switch | with |
|---|---|---|
| wall | 1342.27 s | 1246.08 s |
| the partition loop | 20 m 11 s | 18 m 23 s |
| field interpolation, summed over the 216 passes | 3630.84 s | 3044.2 s |
| peak footprint | 34.09 GB | 35.48 GB |
| maximum resident size | 44.09 GB | 47.18 GB |
| checkpoints written | 2 in 1 m 3 s | 3 in 1 m 46 s |

96.2 s less, 7.2% of the wall time, with all 28 output files byte-identical. The 216 walks took 1244.17 s in all, 5.47
s of it after their device calls had returned, and none was thrown away. The passes' field interpolation fell by 587 s
summed, under half of what the walks take: the device phase, which includes a pass's wait for the device, grew beside
the walks (2262.5 s over the 216 passes with the switch; the last total the log without it prints reads 1590.4 s over
213 passes, with 1206.8 s of CPU deposit, the walk's place there). That fits P11.1's reading of run l, that the passes
wait for the device (not timed with `DTFE_GPU_TIMING` here either), so the walk's seconds are only partly won. The run
without the switch repeats run l of P11.8's confirming series (1338.35 s, 33.09 GB) within 0.3% of the wall time
(1342.27 s). Memory: 1.39 GB more peak footprint (35.48 GB, under the plan's 38.26 GB) and 3.09 GB more resident size
(47.18 GB, against 44.09 GB).

Open, and the owner's: whether the switch becomes the default, which needs its memory term in the planner first
(outside `--grid-window`), and whether a GPU kernel for the face sweep follows. A reading of the pair for that
question, not a measurement: a kernel would move the walk's work onto the device, which the passes on this plan
already seem to wait for. (2026-11-13: both followed as developer switches, the two statuses below; by the two-mode
rule under "Open items" a measurement settles each.) Since 2026-11-13 the walk beside the deposit is the default
(`b62eb94`: the third status of that day, below).

**Status 2026-11-13 (the early walk's memory term; the scratch line's `eba8096`: built and smoke-tested, not gated,
not measured).** One reading of the developer switch for the deposit and the planner: `kPsFaceWalkEarlyDefault`
(false) and `psFaceWalkEarly()` in `src/auto_tune.h` (`DTFE_FACE_WALK_EARLY=1` or `=0` decides where set, else the
constant); `ps_interpolation.cc` reads it where it decides `earlyWalk`. Hard-wiring the walk on is that one constant.
The term (`AutoTunePSInputs::earlyFace`, under the pass's own conditions: a GPU face pass, single precision, no NaN
hook, the face units from the extraction's bounds): the face words `psFaceUnits` allocates before the device call, 30
int64 = 240 B per sub-grid cell by arithmetic, held through the call and the wait for the device. Booked beside every
lane's hold: `partitionBytes` max(hold + 240, after the hold) per sub-grid cell; `laneShareCut` each other lane at
max(`kPSLaneShareHold` x hold + 240, the share x after the hold); under `--grid-window` per partition in flight, out
of the deposit booked once (one in flight plans as before); a mixed lane holds them after its lock already.
`facewalk_gb` on the report line and a note on the AUTO-TUNE line, only where booked. A pass the GPU face kernel takes
(`DTFE_GPU_FACES=1`, the next status) runs no early walk, but the term does not test the kernel's conditions: with
both switches it is booked for such a pass too, which holds the kernel's face words on the device, of the same size,
in the deposit's hold after the deposit's buffers are released (the comment at `earlyFace`; whether the term then
over-books is not measured). Tests: `tests/ps_auto_tune_check.sh` U, `tests/ps_face_gradient_check.sh` P16 (the
failing split: 0 walks kept, 8 thrown away). Without the switch no plan, report line or output byte is to move
(to be measured: `tests/dev/tune_report_set.sh` against `39410e2`). Open: how close the term comes to the peaks measured
with the switch on 2026-11-12 (above), and whether run l's plan keeps 4 partitions in flight with it (to be measured: run
l's `--auto-tune-report` with the walk beside the deposit, the default since `b62eb94`, then one 512³ run of
that plan); the time objective keeps the
memory plan for a GPU face pass under the switch (P8.8, since `41860cc`); three low points of the review (an error of
the walk thrown although a failed device call discards its sums, a helper thread that cannot start for lack of memory,
the closing line's seconds counting discarded walks) and the stale comment above `earlyWalk` in `ps_interpolation.cc`
(the planner "books only under --grid-window"): proposed, not applied. (Later the same day, `b62eb94`, the third
status of the day: the constant is true, the three low points and the comment are fixed, and the probe of run l's
plan keeps 4 in flight at 41.00 GB predicted; the term over-books the passes the kernel takes, whose own face words
it does not cover.)

**Status 2026-11-13 (the GPU face kernel, behind two developer switches; the scratch line's `e29845d`: built and
smoke-tested, not gated, not measured).** Steps 2 and 3 of the kernel's design; no default changes. Step 2:
`src/CGAL_triangulation/face_sweep_f32.h`, the face sweep in one float arithmetic for both devices: correctly rounded
IEEE single operations with flush-to-zero, the Apple GPU's rule for subnormals, so the CPU runs the sweep, the
boundary-plane test (`psOwnsUpperBoundarySection`) and the records' plane counts (`psFaceRecordPush`) under AArch64's
flush-to-zero mode (`fsFlushed`, which sets it for the call and puts it back; on other CPUs, where no face kernel
runs, it is not set); an explicit fma for every product that feeds a sum, a combinatorial corner order, bounds formed
only as fma(i, h, -R0), division-free shares; one change to the design: a piece's moments and velocity about the
piece's own first vertex, not the face square's corner (a piece much smaller than its face square cancels there: the
header comment says why). `DTFE_FACE_F32=1` makes every CPU face sum of a single-build run with it. Step 3:
`metal/ps_face.metal` (`faceSweepItems`, `faceProbe`), `PSFaceParams` and the 28-float face record in
`ps_kernel_params.h`, `ps_face_msl.h` in the object folder (a library of its own, compiled with fast math off and in
Metal's safe math mode with precise functions where the macOS has those selectors), the host's face pass in the
deposit's hold after the deposit's buffers are released (`PSGpuFaceJob`, `psHostFacePass`; records from the GPU
extraction walk, the items' map at one item per block, `gpuRunPass`, the CPU walk as the fallback with the same
words). `DTFE_GPU_FACES=1` takes it and implies the canonical arithmetic for every CPU face sum of the run, so a CPU
run under the first switch and a GPU run under the second are to write the same face grids (P18); `auto` is 0. The
kernel does not take a pass in the run-wide units (`DTFE_PS_RUN_UNITS=1`, and so `--ps-merge fixed-point`), under the
NaN test hook or under `DTFE_GPU_FLOAT_MOMENTS=1` outside a window; with `DTFE_FACE_WALK_EARLY=1` it takes the pass
and the early walk does not run. The checkpoint key names the arithmetic (`face-sums=f32`, one line for both
switches). The kernels' shared pieces (`fixedShareOk`, `addFixedQ`, `psItemOf`) moved unchanged to
`metal/ps_common.metal.inc`. The planner books the face records under the switch only (`kFaceRecBytes` = 28 x 4 + 8 +
4 = 124 B times `kFaceRecFraction` = 1.0 per extracted tetrahedron, in the GPU lane's per-tetrahedron term; also for
the passes the kernel declines, an over-booking). Tests: `tests/face_sweep_check.sh` F4-F6,
`tests/ps_face_gradient_check.sh` P17 (the CPU arithmetic) and P18 (the kernel against the CPU at 0.0: one domain, the
boundary plane, a window, a 2³ split, a kill and resume across the devices, a NaN mass, the NaN hook, faults),
`metal/validate_deposit.cpp` T16a (the device probe) and T16 (the kernel's words against the CPU's). Open, each a
measurement that settles whether the switches are hard-wired or removed (the two-mode rule under "Open items"):
(to be measured: the harness's T16a and T16 on the device; F4-F6's and P17's bounds; `kFaceRecFraction` from
`DTFE_GPU_TIMING=1`'s "face pass: N records" lines at 512³; the face pass's device time and the `FS_REGISTERS` form
against the default, the harness's `BENCH=1`; the default bytes and plans against `39410e2`). Not booked: the device's
face words in the hold, 240 B per sub-grid cell by arithmetic, more while the S words come back (to be measured: the peak
footprint of one 512³ face run under the switch against its plan), except where `DTFE_FACE_WALK_EARLY=1` books its
term of the same size (the status above). The time objective has no class for the kernel: since `41860cc` it keeps
the memory plan for a GPU face pass under the switch and says so (P8.8; before, it predicted the CPU face walk's
class). `psGpuFaceSums` for a pass that deposits on the CPU has no caller yet. The runs that
measure the gain (to be measured: run l's command, P11.1: TNG100-3-Dark z = 0 at 512³, 6³, performance, three ways on one
binary with `DTFE_GPU_TIMING=1` -- the walk after the deposit, beside it (`DTFE_FACE_WALK_EARLY=1`) and the kernel
(`DTFE_GPU_FACES=1`) -- comparing the wall, the holds, which then span the face pass, and the peak footprint; under
the switch the face pass's seconds land in the pass's "GPU deposit" phase). (Since `b62eb94`, the next status: the
walk beside the deposit is the default, so the kernel takes the pass where that walk would run by default; the
walk's term is booked by default and over-books the passes the kernel takes, whose own face words it does not cover
(the comment at `earlyFace`); the three ways above are `DTFE_FACE_WALK_EARLY=0`, the default and
`DTFE_GPU_FACES=1`.)

**Status 2026-11-13 (the face walk beside the device call: the default; the scratch line's `b62eb94`: built and
checked on small data, not gated).** Built under the owner's two-mode rule (P14: a switch that only compares two ways
is hard-wired or removed after its one measurement); this walk's pair was measured on 2026-11-12 (the 512³ pair above,
on one binary of that day, before the scratch line). `kPsFaceWalkEarlyDefault = true` (`src/auto_tune.h`, read through
`psFaceWalkEarly()` by the deposit and the planner): a single-precision `--face-gradient` pass on the GPU walks its
face sums beside its device call wherever its units come from the bounds the extraction summed (not under
`DTFE_TEST_NAN_SHARE`, not where the GPU face kernel of `DTFE_GPU_FACES=1` takes the pass); the double build, whose
face pass walks after the deposit, and every CPU pass are as before. `DTFE_FACE_WALK_EARLY=0` is the developer's off
switch (the walk after the deposit, the plans without the term), kept for one more measurement of the pair; `=1` means
the default; neither is a user option. The output bytes are the walk after the deposit's. With it, the review's three
low points of the memory term's status (above) are fixed in `ps_interpolation.cc`: a walk thrown away after a failed
device call no longer rethrows its own error (`PSEarlyFaceWalk::join`, given the call's outcome); no memory for the
face words before the call (a `std::bad_alloc` caught around `earlyWalk`, which lets the words go) or for the helper
thread (`PSEarlyFaceWalk::start`) leaves the walk after the deposit; the closing line (`DTFE.cpp` `psReportLoopEnd`)
counts the kept walks' seconds only and names no variable, and a set `DTFE_FACE_WALK_EARLY` that is on where no pass
took the walk says so in one line. The comment above `earlyWalk` names the planner's term. The time objective
(`DTFE_AUTO_TIME=1`) tests the kernel before the walk and keeps its constants, measured with the walk after the
deposit (P8.8's record).
- The plans (measured on the line, `3947bb9`'s code; plan lines, not runs). The planner books the words each pass
  holds through its device call by default (`earlyFace`, 240 B per sub-grid cell; the report's `facewalk_gb`).
  Against the line's set of before the default and S7, 19 of the 127 lines of `tests/dev/tune_report_set.sh` move,
  the 19 GPU face lines, each gaining `facewalk_gb`: in 12 nothing else moves, in 6 `predicted_gb` rises with the
  partitions and the concurrency unchanged (`ps ram=64 grid=256 prod exact gpu` 25.01 → 29.23 GB,
  `ps face exact gpu one domain ram=64 grid=256 prod` 15.89 → 19.92 GB), and in one the plan moves:
  `ps window exact gpu grid=1024 prod` from 8 partitions in flight (22.44 GB predicted) to 5 (38.12 GB of a 38.40 GB
  budget; `facewalk_gb=4.027`) (to be measured: a windowed 1024³ GPU face run under the default: its wall time and peak
  footprint against that plan). No CPU, sampled, density-only, double-build or standard line moves.
- Run l's command (P11.1), a plan-only probe on a simulated quiet 64 GiB machine (budget 41.23 GB), plan lines too:
  the walk after the deposit plans 6³ with 4 in flight at 38.26 GB predicted; the walk beside it, the default, the
  same plan at 41.00 GB (`facewalk_gb=2.028`), 0.23 GB inside the budget; the face kernel on top 6³ with 3 in flight
  at 38.15 GB (to be measured: one 512³ run of run l's plan under the default: its wall time and peak footprint against
  the 41.00 GB predicted).
- Tests: P16 and P17 of `tests/ps_face_gradient_check.sh` pair the default (and `=1`) with `=0` -- among them a real
  NaN vertex mass with the walk beside a call that succeeds, beside one that fails and under `=0`, and the NaN hook's
  default against `=0` --; `tests/ps_auto_tune_check.sh` U pairs them, Z and AA6's balanced run take `=0`, and AA6's
  performance run under the default falls back, said, without time keys. Not covered by a test: the two
  `std::bad_alloc` fall-backs and a walk's own error after a failed device call (P16 fails device calls with
  `DTFE_GPU_FAULT_AT=1:persist`, whose walks are thrown away without an error of their own; no hook makes an
  allocation fail or a walk throw).
- Checked on the line with S7 (2026-11-13, `3947bb9`'s code): clean builds (single, double, 2-D, library) without a
  warning; `tests/dev/bytecmp.sh --gpu-exact` 261 files, 0 differ (it pins `DTFE_FACE_GRADIENT=0`, so it does not
  run the walk); single build: `ps_face_gradient_check.sh`, `split_bytes_check.sh`, `ps_parallel_check.sh` and
  `dtfe_auto_tune_check.sh` pass, `ps_auto_tune_check.sh` 111 OK, 0 FAIL, 1 SKIP (after `b6fa0b8`); double build:
  `ps_face_gradient_check.sh` and `ps_auto_tune_check.sh` pass; `tests/docs_drift_check.py` 36 passed, 0 failed.
  Not run with this code: the full gate (the battery), the Linux check and any 512³ or larger run. Open: the runs of
  run l's plan and of the windowed 1024³ plan (the markers above); `kPSTimeGpuFace` with the walk beside the deposit
  (P8.8's record); whether `DTFE_FACE_WALK_EARLY=0` goes after one more measurement of the pair (the two-mode rule).

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

**P11.7 What the bench leaves (recorded 2026-11-09 21:00; items 1, 2, 4, 5 and 6 done 2026-11-10, item 5's
boundary-plane section built 2026-11-11, item 3 open).**
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
   trimming the lanes' common over-booking, which would give l its fourth lane back. **Trimmed 2026-11-11 (P11.8): the
   plan gives l 4 in flight.**
2. **DONE 2026-11-10.** The face walk's second bound walk on the GPU path. The bounds the GPU extraction sums (the 10
   moment bounds and the three face bounds, in the bound walk's block order) stay in the pass's fixed-point state
   (`PSFxState::gpuBounds`, `gpuBoundsValid`) and stand in for `psFxBoundWalk` when `psCpuFixedUnits` forms the face
   units after a GPU deposit (`psFxBoundsFromGpu`). The walk remains where the two sums are not the same doubles --
   the double build, whose extraction works on float copies, and a run under the NaN test hook -- and where the
   extraction summed no bounds (`DTFE_GPU_FLOAT_MOMENTS=1` without a window). `DTFE_DEBUG_FX_BOUNDS=1` walks anyway
   and compares the 13 doubles byte for byte (a line on stderr; a difference stops the run); P12 of
   `tests/ps_face_gradient_check.sh` requires its "0 differ" in three single-build runs (a window, a non-periodic box,
   the lattice box). The time the walk took is not measured, so no saving is recorded.
3. (D18 was closed on 2026-11-11 without them: the drivers and the launcher pick the device.) The two standard runs
   pinned to 2³ (the GPU at the CPU's plan), for the device rule's table (P11.4, D18).
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
   residue was not the face sums': neither device deposited the outside tetrahedra's section, so the layer's upper face
   had no stream and its x gradient stayed the stream average, each device's own float sum. Depositing that section on
   both devices instead (the tetrahedron's face in the boundary plane is the grid's own face) would close it and was not
   done then: it changes CPU bytes, so it was the owner's to ask for (asked for on 2026-11-11, the owner's answers under
   the decision table; built the same day, below). Test: P12 of `tests/ps_face_gradient_check.sh` (a GPU build; a
   one-cell window, a non-periodic box and the NaN hook byte for byte, the lattice box as above; its form since
   2026-11-11 below). Case (b) has no constructed input: P12's window run covers the path, not the rounding. **Built
   2026-11-11 (the owner's decision 2c; code and tests, gates green).** Both devices deposit that section:
   `psOwnsUpperBoundarySection` (`ps_interpolation.cc`) takes a tetrahedron the span test puts outside a non-periodic
   grid when, on each axis where its span starts at the grid's end, its lowest vertex lies in the plane or below it
   (`psDepositFaces`' own double test, against the same rounded product nGrid dx as its first-plane test; exact in the
   single build, and in the double build the shared rounding is what makes the two agree) and its span reaches the grid
   on the other axes. The deposit loop's `outsideGrid` branch deposits that tetrahedron's face sums and nothing else of
   it, and `psFaceWalk` keeps it as well (KEEP IN STEP: the loop's span block, `psLoopSpanReaches` and this test). The
   lower boundary plane needs nothing: its owners reach the grid. Not in a `--grid-window` run: its units and (GPU,
   `--ps-vertex-mass`) its vertex degrees cover only the tetrahedra that reach the window, so a window keeps the set it
   had on both devices, as at its own planes inside the grid. Periodic runs, window runs and runs without
   `--face-gradient` cannot change; non-periodic face runs with such tetrahedra change in the last layer's gradient
   family and `.a_massFluxDiv` only. In a split run the section needs the sub-grid of the partition that holds such a
   tetrahedron (`psSubgridBox`) to hold the last layer; P13 checks that. Tests (`tests/ps_face_gradient_check.sh`): P12
   runs the lattice box on one domain (`--partition 1 1 1`) and, in a GPU build, requires the CPU and the GPU run byte
   for byte in every layer; on every build, a CPU-only one too, it requires the CPU run's last x layer divergence to be
   the face value (no longer `.a_velDivStreams` to the byte) in at least half as many cells as in the layer below, and
   within 1e-5 of the rms of the stream average (measured: 1024 of the layer's 1024 cells, as in the layer below, 2.5e-6
   of the rms from the stream average in the single build and 4.2e-10 in the double). P13 splits the same box 2³ on the
   CPU (`--partition 2 2 2 --max-concurrent 2`) and requires the face value in at least half as many cells of the last
   layer as on one domain (a split merges its partitions in float, so not the same count), within the same 1e-5 of the
   stream average, and `.a_massFluxDiv`, `.a_velDiv` and `.a_velShear` within 1e-5 of the one-domain rms in every x
   layer (measured: the face value in 1024 of 1024 cells, 4.0e-6 from the stream average, the three grids within 1.3e-6,
   8.0e-7 and 2.1e-6; in the double build 7.2e-11, and 1.1e-10, 9.8e-11 and 2.0e-10). The suite passes in both
   precisions (31 checks in the single build, 30 in the double; 26 s and 27 s). What the section changes, measured on
   the face set of 2026-11-10 against the commit before (180 files): 6 differ, all in the lattice box's last x layer
   (1024 cells), `.a_velDiv` by at most 3.1e-7 of the rms and `.a_velShear` by 6.3e-7 (5.2e-11 and 1.1e-10 in the double
   build), `.a_massFluxDiv` by 0.37 of the rms in both. These are the differences of this item's record of 2026-11-10:
   until that day the GPU run's walk had deposited the section, and that day's fix made it follow the loop, which did
   not. The 0.37 is a correction, not a rounding: without the section the layer's mass flux divergence was formed
   without the momentum through its upper face, mean 977 where the layers below have 24; with it the mean is 113, and
   the same cells on a grid extended four cells beyond the plane (36 × 32 × 32; the density steps from 0.4146 to 0.3759
   at the plane) hold 113 from the binary of the commit before and from this one, byte for byte the box's new layer, as
   the 31 layers below are. Standard DTFE's face pass already kept those tetrahedra (a closed bounding-box overlap with
   the region widened by 1e-6 of its size).
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
loop and zeroed by the walk, the same words under a different count in the log (aligned on 2026-11-11: the walk
clamps as the loop does and keeps a NaN, so both count those shares). Then, on the tree as it stands: the
Mac battery, 33 of 33 stages (19:24 to 20:14, the tree unchanged over it), and CI's Linux stages in Colima (build,
core, double, 2d, variants, oldpy, ruff), all passed (20:14 to 20:52). Only this sentence was written after them. Item
3 stays open (two full-size runs, on the owner's go).

**P11.8 The planner's refit: PS-DTFE's lane share, standard DTFE's look-ahead and post floor.** (Committed as `2d262e4`
the same day.) — *done in code 2026-11-11 (the owner: "Yes, start" to the planner batch); uncommitted; no run on a new
plan yet.* The mode bench (P13) and run l (P11.7 item 1) left two planner findings. PS-DTFE's model books every
partition in flight (a lane) whole and predicts 20 to 45% above the measured peak once several are in flight with
evaluation points: 44.7 GB for the 6 of the hand plan, 31.4 GB measured. Standard DTFE's predicts too little in two
places: gentle's look-ahead (37.6 GB against 40.7 GB measured) and the 6³ per-tetrahedron plan (31.7 GB against 35.1
GB). The accept band is a prediction from 5% under to 15% above the measured peak footprint; under is the unsafe side.
All sizes below are GB on TNG100-3-Dark z = 0 at 512³ unless a line says otherwise.
- **The lane share (PS-DTFE).** The first lane in flight is booked whole (`partitionBytes`) and each other one
  `laneShareCut` less; the report carries the difference as `laneOnce`, booked once. The share f is the larger of
  `kPSLaneShare` = 0.55 and the mean partition span over the largest. In a mixed run (`--gpu-share` under 1) a lane's
  cut is max(CPU deposit, GPU arrays before the lock) − max(f × the CPU's fraction of the partitions × CPU deposit,
  GPU arrays before the lock). In an all-GPU run it is (max(hold, after) − max(`kPSLaneShareHold` × hold, f × after))
  per sub-grid cell, with `kPSLaneShareHold` = 0.30, `hold` the grids of a deposit's hold of the device and `after` a
  face pass's state behind it (0 without `--face-gradient`): one deposit holds the device at a time, and a lane keeps
  its own sub-grids after its hold (116 of the 396 B per cell here). The tessellation, the particle copies, the point
  blocks and the GPU's arrays stay whole. Under `--scratch-dir` the model had no term for what the scratch-backed
  grids keep in RAM (the whole lanes hid it); the share books it as `laneShareCover`: at most `kPSScratchRamGrids` =
  3.8 times the scratch threshold, less the point outputs (0.216 GB here), and never more than the cut.
- **Scope (`laneShareOn`).** 3D, Metal, single precision, fixed-point moments (not under `DTFE_GPU_FLOAT_MASS` or
  `DTFE_MASS_AUDIT`), evaluation points, more than one partition, no `--ps-window`, sub-grids within the GPU's cell
  cap; a mixed run only from `--gpu-share` 0.5 and without a face pass. Counts in flight: 2 to `kPSLaneShareMaxGpu` =
  4 (all-GPU) and 2 to `kPSLaneShareMax` = 6 (mixed), the counts measured. Above the cap every lane is whole, and the
  dynamic concurrency (P8.10) stops at it (`psInFlightCap`; its line: "the memory model's lane share holds for no more
  partitions in flight than the plan's"). Only `memorySplit`'s plans use the share, a user's `--partition` and
  `--max-concurrent` within the range included. A CPU run, a run without points, the double build and 2D plan as
  before.
- **At run time** (`src/DTFE.cpp`). The memory gate's lane is the whole one (`psPartitionBytes` = `perPartition` +
  `laneOnce` + the snapshot); a mixed run's `psOnceBytes` carries `laneOnce`; the look-ahead's room subtracts it; and
  the staged merge's park room is the budget less the prediction with every lane whole (`predictedWholeLanes`), so
  parking is as it was (1.80, 1.94 and 2.38 GB for balanced, gentle and the GPU sampled run). `DTFE_AUTO_TUNE_DEBUG=1`
  prints the terms: "PS lane share: first lane … GB, each other … GB, spread …, share …, scratch grids … GB, used for
  2 to N in flight (M planned)".
- **The fit.** `kPSLaneShare` on the 8 mixed runs (3 to 6 in flight, GPU share 0.5 to 0.7) and on the all-GPU face run
  with points (4 in flight, 33.41 GB), which sets the ceiling: up to 0.562 it stays within +15%; a 2³ face run on the
  64³ pancake without points needs at least 0.489. `kPSLaneShareHold` on the 7 all-GPU runs with points: the ceiling
  is 0.36, and a flat 4-lane run needs at least 0.052. The table replays the formulas of before and of now on the
  runs' logs (6³ partitions; the names are the logs'); the binaries' `--auto-tune-report` lines give the same figures
  for the six probed (performance, run l, balanced, gentle, run i, GPU sampled).

  | run | in flight | measured | before | now |
  |---|---|---|---|---|
  | ps_perf (36.4 GB budget) | 3 | 28.93 | 34.68 (+19.9%) | 30.84 (+6.6%) |
  | ps_perf_dbg, the 5 it reached | 5 | 29.81 | 41.34 (+38.7%) | 33.44 (+12.2%) |
  | ps_perf_dbg, the plan's 4 | 4 | 29.81 | 38.01 (+27.5%) | 32.14 (+7.8%) |
  | ps_perf_free1000, the 5 it reached | 5 | 29.34 | 41.34 (+40.9%) | 33.44 (+14.0%) |
  | ps_perf_free1000, the plan's 4 | 4 | 29.34 | 38.01 (+29.6%) | 32.14 (+9.5%) |
  | ps_perf_mc6 | 6 | 31.41 | 44.67 (+42.2%) | 34.74 (+10.6%) |
  | ctl_ps_hyb600_mc6_old | 6 | 31.21 | 44.67 (+43.1%) | 34.74 (+11.3%) |
  | ps_hyb500_mc6 (share 0.5) | 6 | 32.49 | 44.67 (+37.5%) | 35.45 (+9.1%) |
  | ps_hyb600_mc6 | 6 | 30.72 | 44.67 (+45.4%) | 34.74 (+13.1%) |
  | ps_hyb700_mc6 | 6 | 31.01 | 44.67 (+44.0%) | 34.45 (+11.1%) |
  | ps_bal | 4 | 30.00 | 39.43 (+31.4%) | 32.62 (+8.7%) |
  | ps_gentle | 4 | 30.02 | 39.30 (+30.9%) | 32.49 (+8.2%) |
  | ps_exact_gpu_pts | 4 | 30.43 | 39.70 (+30.5%) | 32.89 (+8.1%) |
  | ps_gpu_cap | 4 | 30.12 | 39.70 (+31.8%) | 32.89 (+9.2%) |
  | bench i | 4 | 29.89 | 39.70 (+32.8%) | 32.89 (+10.1%) |
  | bench l (GPU, face gradient) | 4 | 33.41 | 44.03 (+31.8%) | 38.26 (+14.5%) |
  | bench k2 (GPU, sampled) | 4 | 28.61 | 38.85 (+35.8%) | 32.32 (+13.0%) |
  | phase R ps100_099_final (GPU, sampled) | 3 | 27.21 | 34.45 (+26.6%) | 30.17 (+10.9%) |

  These 18 rows of 16 runs: +6.6% to +14.5%, where they were +19.9% to +45.4%. Outside the share and as before: the
  CPU runs (+27.7% to +33.6%), phase R's face512 (+8.1%) and exact1024 (−0.9%), the pancake g1 (−6.8%). The TNG50-3
  run (2 in flight by hand, 34.93 GB) goes from 47.11 (+34.9%) to 43.44 (+24.4%). One count further the model gives
  48.00 (performance at 7, every lane whole), 43.95 (balanced at 5), 49.69 (run l at 5) and 43.25 (GPU sampled at 5),
  all above the 41.23 GB budget.
- **Plans that move** (the binaries' report lines, 41.23 GB budget; before → now). Performance: `partition=6 mc=4
  predicted_gb=38.01` → `mc=6` 34.74 (first lane 3.329, each other 1.299). Run l (GPU, exact, face gradient, points):
  `mc=3` 38.36 → `mc=4` 38.26 (5.669 and 3.677). Balanced: `mc=4` 39.43 → 32.62 (4.521 and 2.178). Gentle: `mc=4`
  39.30 → 32.49. Run i: `mc=4` 39.70 → 32.89. GPU sampled: `mc=4` 38.85 → 32.32. The same on both sides: the CPU runs
  h (`mc=6`, 39.60) and j (`mc=3`, 38.84), performance at 1 in flight (28.58) and at 7 (48.00, over budget). At the
  36.4 GB budget of the mode bench's night performance goes from `mc=3` (34.68) to `mc=6` (34.74).
- **Standard DTFE.** (1) The look-ahead is booked at `kStdLookAheadFactor` = 1.275 times its count (`aheadMax`), in
  one place (`reportSplit`): the gate that switches it off, the report, the prediction and the run's free-memory check
  (`stdLookAheadCounted`) read that figure. The count left out the TBB insert's sorted copies (mc1 with the parallel
  insert peaked 4.44 GB above mc1 without) and under-counted the busiest partition at 4 to 6 threads (gentle: 12.2 M
  padded vertices planned, 15.67 M in the run). Gentle needs a factor of at least 1.114 and mc1 with the parallel
  insert 1.258 to come within −5%; the run with points keeps its look-ahead under the budget up to 1.294. (2) Of the
  busiest partition the post side of `stdPeak` keeps at least `kStdPostKeepGrids` = 0.5 times the grids only the
  post-processing allocates (`fixed` − `fixedLoop`), with the 1 GB of P8.9 as the minimum and the partition as the
  maximum. It is fitted on one run: the 6³ per-tetrahedron plan (35.11 GB) needs at least 0.476 for −5%, and the
  timing plan (4³ at a 33.9 GB budget) stays chosen up to 0.511. Its three face-gradient twins peaked at 23.95 to
  24.43 GB, below the grids' own size: which of the two states a run is in after the loop depends on the machine, not
  on the gradient route, and the floor books the higher one. The rows of the record that move (the binaries' report
  lines give the same figures for the five probed: gentle, balanced, the 6³ per-tetrahedron plan, mc1 and the timing
  plan):

  | run | measured | before | now |
  |---|---|---|---|
  | std100_099 (interim build `745daac`; 6³, 8 threads, per-tet) | 26.46 | 30.08 (+13.7%) | 33.73 (+27.5%) |
  | std100_099_split6 (6³, 9 threads, per-tet) | 35.11 | 31.70 (−9.7%) | 33.73 (−3.9%) |
  | the three 6³ face-gradient runs | 24.00, 23.95, 24.43 | 31.71 (+32.1%, +32.4%, +29.8%) | 33.73 (+40.5%, +40.8%, +38.1%) |
  | bench mc1 (3³, one at a time, sequential insert) | 32.75 | 34.05 (+4.0%) | 35.45 (+8.2%) |
  | mc1 with the parallel insert | 37.19 | 34.05 (−8.4%) | 35.45 (−4.7%) |
  | GPU face run with 16.8 M points | 40.76 | 39.58 (−2.9%) | 41.08 (+0.8%) |
  | bench timing plan (4³, 9 threads, 33.9 GB budget) | 30.86 | 31.80 (+3.0%) | 33.73 (+9.3%) |
  | std_gentle (3³, 4 threads) | 40.66 | 37.57 (−7.6%) | 40.12 (−1.3%) |
  | std_bal (3³, 6 threads), with the look-ahead as it ran | 41.31 | 41.20 (−0.3%) | 44.36 (+7.4%) |

  The other 16 of the 27 rows keep their prediction. On 24 rows (27 less the interim build, the killed run and
  balanced, whose plan changes): mean +6.4% → +8.7%, lowest −9.7% → −4.7%, rows more than 5% under 3 → 0, rows more
  than 15% above 4 → 4 (the face trio and phase R's face run at +23.7%, all four above it before).
- **What it costs.** Balanced loses its look-ahead on this input: booked at 44.36 it no longer fits the 41.23 GB
  budget (it measured 41.31), and the 6-thread plan is now 33.73 without it ("look-ahead off for this run", which now
  names "the next one's look-ahead build", 14.66 GB). That plan has no measurement; the 9-thread runs of the same
  split without look-ahead measured 33.14 to 34.27 GB, and balanced took 447 s with the look-ahead. Performance's
  look-ahead stays off (it would need 57.8 GB; 52.9 GB before). The interim build's run and the face trio are booked
  further above their peaks (the table). At 1024³ without `--scratch-dir` the floor (62.3 GB with all fields, 12.9 GB
  with density, velocity, divergence and shear) exceeds any partition, so the post side is `fixed` plus the busiest
  partition, the sum of before 2026-11-07. On a 48 GiB machine (30.92 GB budget) a 512³ production plan fits a busiest
  partition of 4.98 GB where 14.2 GB fitted: a finer split or fewer threads. Not booked then, as before: an in-team
  TBB insert at `--max-concurrent` 2 to 5 by hand (booked since the leftovers batch, below).
- **The tune set** (`tests/dev/tune_report_set.sh`, 127 lines: the 124 and three for the lane share), against binaries
  built from a clean export of `ae81362`: 11 lines differ. Of the 124 no PS line moves, and 8 standard lines do, none
  in its partition or its count in flight: "dtfe ram=64 grid=1024 plain gpu" and "… exact gpu" by the factor
  (look-ahead 0.383 → 0.489 GB, predicted 28.48 → 28.58 GB at a 38.40 GB budget); and by the floor "dtfe prod ram=8
  grid=1024 plain" and "… exact" and "dtfe prod ram=64 grid=1024 plain" and "… exact" (82.69 → 84.68 GB), "dtfe all
  fields ram=64 grid=1024 exact" (181.51 → 183.50 GB) and "dtfe prod ram=64 grid=1024 exact gpu" (84.48 → 89.79 GB),
  all six `partition=6 mc=1` and over budget on both sides. The three new lines (the 128³ pancake at grid 256, 4 M
  points): every deposit on the GPU `mc=4` 4.50 → 4.22 GB (4.80 GB budget); `--gpu-share 0.5` `mc=5` 4.77 → `mc=6`
  4.71 GB; with the face gradient `mc=4` 6.52 → 6.28 GB (6.60 GB budget).
- **Reviews** (light: readers and a judge, twice). The first found one defect of weight, fixed: the share's search for
  the count in flight left out the TBB insert's copies (`kParInsertBytes`, 800 B per padded vertex of each partition
  in flight, where the insert runs in parallel), which `finishReport` adds, so a plan could print `over_budget=1`.
  `laneSharePeak` now holds them, and the search takes the largest count whose own prediction fits (the prediction
  does not grow with every count: the copies end where a partition has fewer than 2 threads, from 6 in flight on 10
  threads). Probed on the binaries before and after the fix (2³ partitions of the 64³ pancake at grid 256, 4 M points,
  14 budgets): without `--parallel-triangulation` all 28 plans are the same; with it 5 of 14 (every deposit on the
  GPU) and 4 of 14 (`--gpu-share 0.5`) were over budget and are within it, each with one partition fewer in flight.
  Also taken: no share above the GPU's 2³² cell cap, and a wrong comment in the test. The second review, of the fix,
  confirmed no defect in it and narrowed one comment. Left then, each small: the split search's own count did not book
  the copies (every deposit on the GPU, the insert forced, a 5.5 GB budget: `mc=5` at 5.51 GB, over budget, before and
  after); the planner did not read `DTFE_TRIANGULATION_THREADS` (the live memory gate covered an insert the plan did
  not see); near a run's end the cores are divided by the partitions left, so the last ones could insert in parallel
  where the plan priced none; section Q failed, where it should skip, under `DTFE_GPU_FLOAT_MOMENTS`,
  `DTFE_GPU_FLOAT_MASS` or `DTFE_MASS_AUDIT` and on a GPU build without TBB. These four are done (the leftovers batch,
  below). Still left: the split choice itself does not use the share (the 5³ run of the confirming series, below);
  an all-GPU run books a lane the dynamic concurrency adds whole (until the planner batch of 2026-11-11, below).
- **Gates (2026-11-11, 00:27 to 00:46, on the binaries with the fix, against binaries built from a clean export of
  `ae81362`).** `tests/dev/bytecmp.sh --gpu-exact`: 261 files, 0 differ. The tune set as above, no bad line on either
  side, the reference's first 124 lines the frozen set. The 17 plan probes (10 PS, 7 standard), whose lines are the
  ones quoted above. ps_auto_tune_check with the new section Q (Q1 to Q7): 82 OK single, 69 OK and 2 skipped double.
  dtfe_auto_tune_check: 22 OK single, 20 OK and 1 skipped double. run_mode_check 57 and 49 passed, ps_runtime_check 41
  and 41, dtfe_resume_check 44 and 44, docs_drift_check. The search's comment was narrowed after these (the code
  outside comments is the same text); the binaries rebuilt from it give the same 17 plan lines.
- **The owner's three full-size runs, as planned (made on 2026-11-11: the next bullet)** (a quiet machine, Colima
  stopped, `/usr/bin/time -l`, `DTFE_AUTO_TUNE_DEBUG=1`). (2026-11-11: the owner gave the go for the four runs, and the
  thread step-down is to be designed, with one measured run; the owner's answers under the decision table. The runs were
  made that day: the next bullet.) (1) Run l on its new plan (the bench's GPU face run with points, without
  `--partition` and `--max-concurrent`): expected `--partition 6 6 6 --max-concurrent 4` at 38.26 predicted; within the
  band at a peak footprint of 33.3 to 40.3 (it measured 33.4 at 4 in flight). (2) Performance on its own plan: 6³ with 6
  in flight at 34.74 predicted, within the band at 30.2 to 36.6, the wall time against 783 s. (3) Standard DTFE under
  `--run-mode balanced`: expected 3³, 6 threads, 33.73 predicted and the "look-ahead off for this run" line; within the
  band at 29.3 to 35.5, the wall time against 447 s. Since the planner batch this is the plan of DTFE_LOOKAHEAD_TEAM=0;
  the default balanced plan is 3^3 with 4 threads, 40.12 GB predicted with an 11.82 GB look-ahead (within the band at a
  peak footprint of 34.9 to 42.2 GB), the other half of the A/B pair (measured: P11.8's planner batch). A cheap
  fourth: `--partition 3 3 3
  --max-concurrent 1` (35.45 predicted, within the band at 30.8 to 37.3). Left for later: the CPU runs' +27.7% to
  +33.6%, the TNG50-3 run's +24.4%, and a standard plan that steps its threads down to keep the look-ahead (the mode
  bench's open point, now with balanced as a case). The pancake's −6.8% is closed (one partition in flight on the GPU,
  below).
- **The confirming series (2026-11-11, on the owner's go; six runs of the bench input, 9.42e7 particles on a 512³ grid,
  with the binaries of the leftovers batch).** `/usr/bin/time -l`, `DTFE_AUTO_TUNE_DEBUG=1`, Colima stopped, the
  41.23 GB budget (39.7 GB in the first run, where the other programs left 43.7 GB free). The machine was not at rest:
  other work read the source tree during the runs, so the wall times carry that load; the memory figures do not. The
  error is the prediction's, as in the tables above.

  | run | plan | measured | predicted | wall |
  |---|---|---|---|---|
  | PS performance, its own plan | 6³, 6 in flight, GPU share 0.60 | 31.39 | 34.74 (+10.7%) | 689 s |
  | PS run l (GPU, exact, face gradient, points), its own plan | 6³, 4 in flight | 33.09 | 38.26 (+15.6%) | 1338 s |
  | PS balanced at `--partition 5 5 5 --max-concurrent 4` | 5³, 4 in flight | 35.23 | 39.17 (+11.2%) | 1197 s |
  | PS on the CPU, its own plan | 6³, 6 in flight | 30.92 | 39.60 (+28.1%) | 1105 s |
  | standard balanced, its own plan | 3³, 6 threads, look-ahead off | 32.36 | 33.73 (+4.2%) | 510 s |
  | standard at `--partition 3 3 3 --max-concurrent 1` | 3³, 1 thread | 39.17 | 35.45 (−9.5%) | 578 s |

  PS-DTFE. Performance on its own plan is within the band (30.2 to 36.6) and runs the 6 in flight of the hand plan:
  689 s against the bench's 783 s. Run l measures 33.09 GB at 4 in flight, as on the bench (33.41), and so lies 0.2 GB
  under the band's lower edge: the prediction is 15.6% high, on the safe side. The balanced run at 5³ and the CPU run
  are predicted high as well; the CPU run's +28.1% repeats the +27.7% to +33.6% of the CPU runs above on the build with
  the lane share. No PS run of the series came near its budget.

  The split choice (the owner's, after this run). Balanced at 5³ with 4 in flight took 1197 s at 35.23 GB; the bench's
  balanced run at 6³ with 4 in flight took 1444 s at 30.00 GB on 2026-11-10. The two days do not compare directly: the
  same performance plan (6³, 6 in flight) took 783 s then and 689 s now, 12% less. Scaled by that, the 6³ balanced run
  would take about 1270 s now, which leaves 5³ a gain of about 6% for 5.2 GB more: inside the day's spread. One 6³
  balanced run on the series' binaries, on a machine at rest, settles it; the split stays 6³ until then.

  Standard DTFE under balanced gives the expected plan and the "look-ahead off for this run" line, within the band (29.3
  to 35.5): 32.36 GB, where the bench's run with the look-ahead measured 41.31 GB on a 41.2 GB budget. The price is
  time, 510 s against the bench's 447 s. The thread step-down of the planner batch (a team of 4 that keeps the
  look-ahead, 40.12 GB predicted) is to be measured against this run.

  The one-at-a-time plan is outside the band (30.8 to 37.3): 39.17 GB against 35.45 predicted, under its 41.23 GB
  budget; the same plan measured 37.19 GB on 2026-11-09 (the row above, an earlier build). Read from the run's footprint
  samples (one a second) and its log, no booked term runs short. The deposit's own rise is 2.7 to 6.5 GB a partition
  (3.70 GB at the peak), where the model books about 5.8 GB for it (a partition's own particles × 8.80 tetrahedra ×
  188 B; the run's 631 M tetrahedra are 6.7 per particle). The peak is partition 24's deposit on a level of 35.47 GB, of
  which the itemized data (the loop's grids, the partition's tessellation and particle copy, its sub-grid on the device
  and on the host, the next partition's look-ahead tessellation) are about 20.8 GB. The 14.7 GB between the two are in
  no item: within this one run the footprint before a deposit lies 2.9 to 14.7 GB above the itemized sum, the most
  through partitions 21 to 25, as memory does that the allocator keeps after a partition is freed. So a one-at-a-time
  plan with the parallel insert is predicted to about ±10%, and no constant is changed for it: `kStdLookAheadFactor`
  would need about 1.53 to hold −5% here (from the plan's look-ahead term, not probed with the binary), where the run
  with points keeps its look-ahead only up to 1.294 (the fit above). The owner's, if such plans come to matter: a wider
  band for them, or a look-ahead factor of their own.
- **The leftovers batch (2026-11-11, on the commit of the refit; no full-size run): the insert's copies.** What the
  reviews above had left, where it needs no run. The split search's own count books the parallel insert's copies
  (`kParInsertBytes`, 800 B per padded vertex of each partition in flight whose insert runs in parallel) and takes away
  a partition in flight that fits only without them. Probed on both binaries (2³ partitions of the 64³ pancake at grid
  256, 4 M points, every deposit on the GPU, `--parallel-triangulation`, 10 threads): at a 5.5 GB budget the commit
  before plans `mc=5` at 5.51 GB with `over_budget=1`, the tree `mc=4` at 3.91 GB; at 4.5 and 5.0 GB (`mc=4`, 3.91 GB)
  and at 6.0 GB (`mc=6`, 5.77 GB) both plan the same, as they do at 5.5 GB without the flag (`mc=5`, 5.07 GB). Both
  planners read `DTFE_TRIANGULATION_THREADS` as the run does (a count forces the insert, 1 keeps it sequential beside
  `--parallel-triangulation`). Standard DTFE books a team whose sub-domains each insert in parallel (`--max-concurrent`
  2 to 5 on 10 threads, sub-domains of 1e6 points or a forced insert) and one tessellation's insert where no look-ahead
  runs. In PS-DTFE an insert the plan priced as sequential (fewer than 2 threads a partition) stays sequential at the
  run's end, where the cores are divided by the partitions left (`triangulation.cpp`; which partition inserts last is a
  matter of timing, so no test at test size). Tests: ps_auto_tune_check O2 ("`DTFE_TRIANGULATION_THREADS=2` books the
  forced insert's copies (0.35 = 0.35 GB), `=1` beside `--parallel-triangulation` none (0.18 = 0.18 GB)") and O3 (a
  budget of 0.70 GB that holds 2 in flight without the copies and 1 with them plans 1, within it); dtfe_auto_tune_check
  J (a team of 2 with the forced insert 0.67 → 0.80 GB, one tessellation without a look-ahead 0.57 → 0.66 GB, each 800
  B per padded vertex).
- **One partition in flight on the GPU (the leftovers batch).** The pancake g1 row (64³ pancake, 256³, the production
  fields, exact, `--gpu --partition 2 2 2 --max-concurrent 1`) measured 5.38 GB against 5.02 predicted (−6.8%), its
  flat twin 4.16 against 4.04 (−2.8%). The GPU lane booked nothing for the partition's own sub-grids beside the
  device's hold, which the CPU branch (`copyOne`), the window path (`snapshotBytes`) and a face pass
  (`gpuAfterHoldCellBytes`) do book: per booked cell the runs held 702 to 713 B against 584, which is 118 to 129 B
  beyond the booking, of which the new term books 110 (with `fixed` at the loop's 110 B/cell). Now a GPU run with every
  partition on the GPU, split, one partition in flight, and neither a face pass nor a window books `bCellDeposit` x its
  sub-grid beside the hold (`loneLaneSubGrids`, in `finishReport`; the split search takes no split whose one partition
  fits only without it): 5.53 GB (+2.7%) and 4.37 GB (+5.1%), and the earlier build's pancake run (5.22 GB measured)
  goes from −3.8% to +5.9%. The face runs at one in flight (+1.0%, +3.4%) and every plan with two or more in flight
  keep their figure. The tune set: 9 of the 127 lines predict more, `predicted_gb` only, all PS at grid 1024 on the GPU
  with `partition=6 mc=1 over_budget=1` on both sides ("ps ram=8 grid=1024 dens sampled gpu" 15.67 → 15.94 GB, "… dens
  exact gpu" 20.48 → 20.83, "ps ram=8 grid=1024 prod sampled gpu" and its ram=64 twin 139.59 → 141.62, their exact
  twins 144.40 → 146.51, "ps N=128 ram=8 grid=1024 dens exact gpu" 20.54 → 20.86, "… prod exact gpu" 143.74 → 145.69,
  "… prod + scalar exact gpu" 162.62 → 164.86); no line changes its split or its count in flight, and the 17 plan
  probes are unchanged. `DTFE_AUTO_TUNE_DEBUG=1` prints 'PS one partition in flight: its own sub-grids beside the
  device's hold, B B/cell x N cells = X GB'. Tests: ps_auto_tune_check Z (the term without the face sums and none with
  them; a real run of that plan at test size, predicted at 0.97 of its measured peak or above) and Q8. That run's peak
  is not one number: with either binary it measured 4.79 GB in most runs and up to 5.61 GB (ten repeats of the test's
  command, the binary of the commit before and the tree's in turn: 4.79, 4.95, 4.79, 5.43, 4.79 GB and 4.79, 5.61,
  4.79, 4.79, 4.79 GB; 5.22, 5.38 and 5.41 GB in earlier runs; why it varies was not looked into). The 5.02 GB of
  before lay under five of those peaks, by up to 10.5%; the 5.53 GB of now lies under one (5.61 GB, by 1.4%) and 15.4%
  over the common 4.79 GB, so the pancake row's +2.7% is one draw from −1.4% to +15.4%. The flat twin was not repeated.
- **The mixed run's three review items (P13.1 item 13, a to c; the leftovers batch).** (a) A mixed run's GPU lane books
  the host grids the copy-back leaves it after the lock, inside the lane's larger-of (`gpuAfterLockCellBytes`: the lane
  is the larger of a CPU deposit and the pre-lock arrays with those grids). (b) The GPU snapshot's host copies are
  freed before the unlock, so the one the planner books for the run is the only one (`ps_metal_host.cc`). (c) The
  parking room shrinks by a lane for each partition the dynamic concurrency runs beyond the plan's (`DTFE.cpp`,
  `parkFloor`), the lane the ledger admitted it with. Bytes: none in the gate set (bytecmp pins its plans and has no
  mixed run). An auto-tuned mixed run where the term of (a) binds may get a finer split, and so different grids, as
  with any planner change; it moves none of the 17 probes and no tune-set line. Tests: ps_auto_tune_check V for (a)
  (plan-only, 128³ particles on a 128³ grid in 2³: the all-GPU lane less the mixed one 0.096 GB against the planner's
  own terms 0.098 GB, where the old booking would be 0.06 GB or more off); ps_runtime_check R7 for (c) (the dynamic run
  kept 1 partition in memory and spilled 25 where the plan's own 4 kept 26, byte-identical grids). No test at test size
  for (b) (gpu_retry_check R1 to R3 cover the passes and their restores). Not covered: (c) reads the count in flight
  without the gate's lock, so it can be off by one lane for one park, and the once part's share of the lane is not
  tested.
- **The batch's smaller items.** The upstream functions without a caller are deleted (P1.4). The core pool counts the
  point evaluation's team and the look-ahead thread (P9.2). `psFaceWalk` clamps a negative tetrahedron mass as the loop
  does and keeps a NaN (P11.7; ps_face_gradient_check P12, single build: "the NaN hook: 66926 face shares left out on
  both devices", "a NaN mass: 112 face shares left out on both devices"). The launcher has the face-gradient tick (P10)
  and lost a dead helper (`_gpu_unticked_by_exact`).
- **The suites under the debug switches (the leftovers batch).** Under `DTFE_MASS_AUDIT`, `DTFE_GPU_FLOAT_MASS` or
  `DTFE_GPU_FLOAT_MOMENTS` a single-precision run keeps the plan's partitions in flight (the dynamic concurrency starts
  only without them: `DTFE.cpp`, unchanged) and the plan books no share. ps_auto_tune_check assumed the dynamic count
  and failed there at the commit before, measured with its binaries: 8 checks under `DTFE_MASS_AUDIT=1` and under
  `DTFE_GPU_FLOAT_MASS=1` (Y1, Y5, Q1, Q2, Q4, Q5, Q6, Q7), 9 under `DTFE_GPU_FLOAT_MOMENTS=1` (S1 with them). Now Y1
  checks the plan's count there, Y5 and section Q skip, and S1 and V skip under the float moments: 79 OK and 2 skipped
  under `DTFE_MASS_AUDIT=1` and under `DTFE_GPU_FLOAT_MASS=1`, 77 OK and 4 skipped under `DTFE_GPU_FLOAT_MOMENTS=1`, 88
  OK with `DTFE_MASS_AUDIT=0`. ps_runtime_check's R7 skips there: 41 passed and 1 skipped under `DTFE_MASS_AUDIT=1`;
  under `DTFE_GPU_FLOAT_MOMENTS=1` 40 passed, 1 skipped and R3 failed, as at the commit before (40 passed and R3 failed
  under the switch, 41 passed without it).
- **Gates of the leftovers batch (2026-11-11, 11:16 to 12:30, against binaries built from a clean export of
  `2d262e4`).** `tests/dev/bytecmp.sh --gpu-exact`: 261 files, 0 differ. The tune set: 127 lines, no bad line on either
  side, the nine lines above and no other. The 17 plan probes: the report lines of the commit before. docs_drift_check:
  36 passed. ps_auto_tune_check 88 OK single, 72 OK and 3 skipped double (Y, V and Q, by design); dtfe_auto_tune_check
  24 OK single, 22 OK and 1 skipped double; ps_runtime_check 42 passed single, 41 and 1 skipped double;
  ps_face_gradient_check in both precisions; run_mode_check 57 passed, dtfe_resume_check 44 passed, gpu_retry_check.
  The launcher's suites with both pythons (numpy 2 and numpy 1.26): py_gui_explore_test 262 passed, py_gui_app2_test 59
  passed and 1 skipped, py_gui_runspec_test 435 passed, py_gui_app_test 401 passed. A review of the code (three
  readers, a judge each) found nothing of high or medium weight and confirmed two low points, both in the list below. A
  fact check of these records (three readers, three judges) found wrong sentences; each was measured again by hand
  before the records were written (the Z run's spread, the sampled plans' 106 B/cell, the 3D view's pictures from
  outside, the checks that failed under the switches at the commit before). Then, on the batch's tree as it stood
  after these records: the Mac battery (12:40 to 13:28), 33 of 33 stages passed with the tree unchanged during it,
  and the Linux check on a copy of it (13:29 to 14:05), all seven stages passed (build, core, double, 2d, variants,
  oldpy, ruff).
- **Still left of the leftovers (each needs a full-size run or is small).** (2026-11-11: the balanced 5³ run and a CPU
  run were made with the confirming series, above; the TNG50-3 error gets one rerun on a night the owner picks and the
  mixed lane's term (a) is read from the campaign's runs; the owner's answers under the decision table.) The split
  choice with the share: by a replica of the model it would move balanced and gentle from 6³ to 5³ at 4 in flight (about
  39.2 and 39.0 GB predicted, on a split the share was not measured on), so one measured balanced run at `--partition 5
  5 5 --max-concurrent 4` first. (Made: 1197 s at 35.23 GB, a gain of about 6% over 6³ that lies inside the day's
  spread. The pair on one night and one binary, 2026-11-12 (the confirming series' binary, built before the canonical
  vertex order; one run at a time): 6³ 1437.28 s at 29.96 GB of peak footprint, 5³ 1412.96 s at 36.30 GB, so 5³ is
  1.7% faster for 6.34 GB more. That 5³ run repeats the 2026-11-11 command on the same binary with byte-identical
  output and took 18.0% longer (its GPU deposit, summed over the passes, 27.7% longer); the cause is not established,
  so only the same-night difference stands. The owner decides; 6³ stays the plan until then.) The four small planner
  items are done (the planner batch, below). Three prediction errors that only runs can settle (the CPU runs' +27.7%
  to +33.6%, the TNG50-3 run's +24.4%, the mixed lane's term (a) against a measured mixed peak). The CPU error was
  measured again with the series: +28.1%. Under `DTFE_GPU_FLOAT_MOMENTS=1` two identical GPU runs give different grids
  (measured at test size with the binary of `2d262e4` and with this batch's alike; with the switch at 0 both are
  byte-identical); ps_runtime_check's R3 skips its byte comparison there since the next batch (below).
- **The launcher (next batch, 2026-11-11).** Decision 1a: the Custom tab passes `--ps-merge index-order` unless its
  extra options name a merge (R7 above). Decision 15b: the face-gradient tick is the last row of the visible
  'Phase-space options' group of the three run tabs (was: the folded Advanced group); its default for a new job is
  `runspec.FACE_GRADIENT_DEFAULT` (False), the one line a later "face gradient by default" changes (with it, the
  launcher must send the flag only where the run qualifies, or its refusals block every sampled job). The leftovers
  review's low point: a standard-DTFE Grids job pins `FACE_GRADIENT` empty (run_dtfe.sh's default, the binary's face
  gradient), as `SUBSAMPLES` and `THREADS`. No output bytes, plans or tune lines move. Suites, under both pythons (3.11
  with numpy 1.26, 3.14 with numpy 2.4): py_gui_runspec_test 438 cases, py_gui_app_test 402, py_gui_app2_test 59 and 1
  skipped (no grids in its data root).
- **The test items of the next batch (2026-11-11; no binary changes).** (1) `ps_runtime_check` R3 keeps its look-ahead
  check under every switch; its byte comparison SKIPs under `DTFE_GPU_FLOAT_MOMENTS` (the device gets no fixed-point
  unit, `ps_interpolation.cc` fxGpuWanted: the moment grids and the exact deposit's per-tetrahedron weight totals, which
  divide every share, the mass's too, are float atomic sums) and under `DTFE_GPU_FLOAT_MASS` (the device's mass grid a
  float sum, `ps_metal_host.cc` floatMass; by the code, not run); under `DTFE_MASS_AUDIT` every grid stays exact and R3
  compares. The switch loop reads an old `PS_` name only while its `DTFE_` name is unset, as `env_knobs.h` does (R7 runs
  under `DTFE_X=0 PS_X=1` now, as the binary does). Measured after the change (single build): under
  `DTFE_GPU_FLOAT_MOMENTS=1` and under `DTFE_GPU_FLOAT_MASS=1` R3's look-ahead check passes and its byte comparison is
  skipped (40 cases pass each time); under `DTFE_MASS_AUDIT=1` R3 compares and passes (41); without a switch 42 pass
  (R7's memory part runs only there). (2) The deprecation gate reads the battery's stage logs (`--stage-logs
  <scratch>/battery`, passed by the battery): every `<stage>.log` but the builds' and its own, a hit tied to the step of
  the last `==== ` line above it (ci_suite's step()), else to the stage; a second allow-list by step title admits only
  the suites whose every case is an old spelling (flag_spelling_check, py_cli_alias_test, run_scripts_check). The five
  python suites' binary runs were already in their temporary folders (`4e2b35b`); the stage logs add what no suite
  redirects (the data generators' stderr, a python suite's own). (3) Decision 11a, prepared: `dtfe_mc_average_check` A1
  runs `--subsamples 2` (8 per cell, at least 2 per tetrahedron; the bound holds for any count; measured +2.42e-5,
  double build -1.26e-5, against +1.99e-5 and +1.38e-5 at the default count and -3.35e-3 for the old realization), and
  `dtfe_sampling_check`'s four method-2 runs name no count, so method 2's default 20 gives the draws S7's golden file
  holds (by the code: the same `noPoints`, the same seed). The gate's allow-list arm for the two suites is gone.
- **The planner batch (2026-11-11, the owner's answer 4b and the small items; no full-size run).** (1) Standard DTFE
  under `--run-mode balanced`: a team whose busiest partition leaves no room for the look-ahead gives way to the largest
  smaller one of DTFE_parallel's teams (8, 6, 4, 3, 2 threads) that has room (`lookAheadTeam`), said on an 'AUTO-TUNE:
  look-ahead kept for this run with N threads per partition instead of M' line; not one tessellation (P11.6: 581 s with
  the look-ahead against 435 s for the 9-thread team without it); `DTFE_LOOKAHEAD_TEAM=0` keeps the plan of before. What
  a team costs: each sub-box is padded on its own faces, so 6 -> 4 threads takes the busiest partition from 15.1 to
  12.2 M padded vertices and the look-ahead from 14.66 to 11.82 GB (the snapshot grows from 0.257 to 0.274 GB); at 3 or
  2 threads a sub-domain gets 2 threads of balanced's 6 and inserts in parallel (800 B a padded vertex). The 512^3 bench
  plan becomes gentle's: 3^3, 4 threads, 40.12 GB predicted with the look-ahead (sbal, the one probe that moves). Which
  is faster, measured (2026-11-11, one balanced run each way with the batch's binary, the bench input and fields, the
  default first, Colima stopped): the default plan 429.07 s at a peak footprint of 41.61 GB (max RSS 42.41; predicted
  40.12, -3.6%), `DTFE_LOOKAHEAD_TEAM=0` 535.66 s at 30.17 GB (max RSS 31.05; predicted 33.73, +11.8%). The step takes
  19.9% less time for 11.44 GB more, so it stays the default. Its peak lies 0.38 GB above the 41.23 GB budget (inside
  the band). The partition loop: 6 min 16 s against 8 min 2 s; triangulation summed over the threads 942 s in 108
  calls against 1162 s in 162. This session's agents read files during both runs (no build, no other run); the
  series' run of the old plan took 509.54 s. The two runs cut each partition into different sub-boxes: their density
  grids differ in 1340 of the 512^3 cells, by 4.8e-7 at most, the divergence in 2 cells; the T-web classes are byte
  for byte the same. (2) PS-DTFE's dynamic concurrency of an all-GPU run books the snapshot and
  laneOnce once (`AutoTuneReport::onceBytes`, DTFE.cpp `psOnceBytes`), each partition at the plan's lane: D29's per-lane
  snapshot is gone. (3) PS-DTFE's split search checks one partition in flight with its insert copies and lone sub-grids
  after the copies' count, so a coarse split that fits only without them gives way to a finer one. (4) The uniform count
  forms the run's team (`parallelGrid`); reachable only through `DTFE_TEST_TUNE_UNIFORM=1`. (5) A look-ahead the budget
  switches off leaves one tessellation's insert copies in the partition (`bytesNoAhead`; the review's F2). Tests:
  dtfe_auto_tune_check K1, K2, L1 to L3, M1, M2; ps_auto_tune_check O4, Y7; run_mode_check G8 (the all-GPU once part).
  The step is taken only where the tuner chose the threads (the split may be the user's). Gates (a clone of the batch,
  2026-11-11): the tune set's 127 lines do not move; of the 17 plan probes only sbal does; bytecmp 261 files, 0 differ;
  dtfe_auto_tune_check 31 cases (double build 29, 1 skipped), ps_auto_tune_check 90 (double 73, 3 skipped),
  run_mode_check 57, ps_runtime_check 42 (double 41, 1 skipped: R7's memory part, the double build runs no dynamic
  concurrency); the retry, face-gradient, window and resume suites pass (the dynamic concurrency now starts in all-GPU
  runs where it did not). Measured, the review's open point (no lane share caps an all-GPU `--grid-window` run):
  TNG100-3-Dark at 512³, the window the box's lower octant, balanced. The plan's own 5³ with 6 in flight: 37.17 GB
  predicted, 35.45 GB measured (+4.8%), 367 s. The plan held at one in flight (`DTFE_TEST_PLAN_MC=1`): the dynamic
  concurrency reached 6 in flight and peaked at 35.05 GB under its 41.23 GB budget, 355 s (the SUMMARY line of a run
  under that hook adds the lanes to the plan's own six: no model check). No per-lane window term is needed. Left by the
  review, not built: `lookAheadPossible` does not read `DTFE_OVERLAP=0`, so under balanced a team can step down for a
  look-ahead the run bans (it costs threads, not memory); the standard search does not price a lone tessellation's
  copies where the look-ahead will be switched off, so such a plan can say `over_budget=1` where a finer split was never
  tried (the standard twin of (3)); under `--scratch-dir` the grids the first merge grows stay in RAM where they are
  below the threshold (`reserveFirstMerge` drops them; older than this batch); a partition beyond the plan's takes a
  shared lane, not a whole one, off the room for parked partitions (at most (cap - floor) × laneOnce); (5) also moves
  plans of an automatic concurrency that end at one thread (no line of the tune set); L reads the report, not a run, and
  M does not reach the uniform branch's copies.
- **The first two fixes of decision 2c (2026-11-11; code and tests, no full-size run).** The linear-deposit weights from
  one compiled clip (P4's status of 2026-11-11) and the boundary-plane section (P11.7 item 5); the records are there.
  Output bytes can move only in GPU exact runs with `--ps-linear-deposit` (the density and velocity grids, by up to
  7.5e-6 of the rms on the 64³ case) and in non-periodic `--face-gradient` runs without a window whose input has
  tetrahedra resting on an upper boundary plane from outside (the face-derived grids, in the last layer: the mean flow's
  gradients by up to 6.3e-7 of the rms on the lattice box, its `.a_massFluxDiv` by 0.37 of the rms, a value that was
  wrong before). Gates: `tests/dev/bytecmp.sh --gpu-exact` against the commit before, ps_linear_deposit_check and
  ps_face_gradient_check in both precisions, the P11.7 face set (the 180 files of 2026-11-10) against the commit before;
  the Mac battery and the Linux check run once before the push, on the stacked batches. Results: the byte comparison 261
  output files, 0 differ; the tune set's 127 lines unchanged; both suites pass in both precisions; the face set 6 of 180
  files, all in the lattice box's last layer; the linear deposit's item passes +16% (24.1 s against 20.7 s at 128³
  particles on a 256³ grid), the uniform deposit's unchanged. The grid total's distance to the tetrahedra's mass does
  not move (1.85e-8 before and after; the uniform GPU deposit has 2.0e-8, the CPU's 1e-11 at most): that excess of the
  GPU exact deposit is older and open.

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
   stale (intended); (Changed 2026-11-10: P13.1 items 20 and 21.) `gpu_auto_resolve` unchanged (P11.7 item 6 has its
   wording).
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
7. G-linux: CI's Linux stages in Colima (build, core, double, 2d, variants, oldpy, ruff) all passed. (This tree, with
   the block, was committed as `39583f0`: the status above.) The `variants`
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
    parking could take such a run past the budget by up to a lane for each of them. **2026-11-11 (P11.8): the lane
    share books a mixed run's other lanes at a share, and the full budget holds the 6 of the hand plan (34.7 GB
    predicted, 31.4 GB measured); the all-GPU ledger's snapshot per lane is as it was.** **2026-11-11 (the leftovers
    batch, P11.8): (a) to (c) done: the lane's larger-of holds those host grids (`gpuAfterLockCellBytes`;
    ps_auto_tune_check V), the snapshot's host copies are freed before the unlock, and the parking room shrinks by a
    lane for each partition beyond the plan's (ps_runtime_check R7).**
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
    for balanced. **2026-11-11: the per-lane over-booking is refitted for the runs with evaluation points (P11.8); the
    rest stays out.**

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
   **2026-11-11 (P11.8): refitted; the mode's plan on this input is the 6 at once (34.7 GB predicted). Not timed on
   its own plan.**
4. PS balanced and gentle are within 3% of each other (1444 and 1408 s, 1.4 busy cores on average): with every
   partition on the GPU the thread budget hardly matters. They take 1.6 times performance's 908 s.
5. Standard DTFE: performance is the slowest of the three. The tuner's 10-thread plan leaves the look-ahead no memory
   on this input (the binary from before the modes plans the same), the 6- and 4-thread plans have it. Their peaks:
   41.3 GB at a 41.2 GB budget (balanced) and 40.7 GB against 37.6 GB predicted (gentle). **2026-11-11 (P11.8): the
   look-ahead is booked at 1.275 times its count. Gentle predicts 40.1 GB and keeps it; balanced's 6-thread plan no
   longer has it (44.4 GB with it, 33.7 GB without; not timed).**
6. Standard DTFE's `--gpu` grids follow the thread count too: 8 of the 10 grids of balanced and of gentle differ from
   performance's and from each other's, the rms of the difference at most 1.4e-10 of the field's.
7. Responsiveness: the night's probes do not rank the modes. Two runs of one configuration differ more than the modes
   do (the GPU probe late by more than a 120 Hz frame in 18.5% of the samples of one performance run and in 44.3% of
   its repeat). The evening's numbers (`inv/results/`) stay the ones to quote.

Open after the bench, the owner's call (nothing of it is changed):

- PS-DTFE's memory model for a mixed plan (the lanes and the points output): a refit to the measured peaks would let
  performance plan 6 at once on this input. A planner change with its own `tests/dev/tune_report_set.sh` diff. **Done
  2026-11-11 (P11.8).**
- (2026-11-11: to be designed now; one measured run decides whether it stays. The owner's answers under the decision
  table.) Standard DTFE in performance: a plan that gives up thread sub-boxes for the look-ahead where the memory holds
  only one of the two (447 s at 6 threads against 486 s at 10 here). It changes default plans, and grids at float
  rounding.
- The standard look-ahead's prediction (gentle: 40.7 GB measured against 37.6 GB; balanced at the budget's edge).
  **Done 2026-11-11 (P11.8). Balanced's 6-thread plan now runs without the look-ahead on this input, which makes it a
  second case for the point above.**
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
`tests/ci_suite.sh 2d` and `tests/docs_drift_check.py` (36 passed, 0 failed) passed again. (Both passed on the next
commit's tree, `c8dd4fb`: P11.7 item 1.) The battery and the Linux
check were not repeated on the tree.

Then, for the owner and not a gate, on the bench input: one `--auto-tune-report` with `DTFE_AUTO_TUNE_DEBUG=1` of
`--gpu --exact --gpu-share auto --run-mode performance` (predicted_gb must land in 32.5-41.2 GB; the plan's
concurrency) — **done 2026-11-10, item 13 above: 34.68 GB, 3 at once**; optionally a performance run on the tuner's
plan against the 712 s hand plan, and a balanced run (the GPU at 50 ms) against 1156 s — **both done 2026-11-10, the
mode bench above** (with the binaries directly, not through the drivers).

## P14. Two modes, Exact and Fast (added 2026-11-13, the owner's request)

**The owner's rule (2026-11-13).** Each estimator has exactly two quality modes for its users, **Exact** (`--exact`,
the default) and **Fast** (`--sampled`), and no further user option for accuracy or for a faster way to the same
quality: the code chooses the rest, and a switch that only compares two ways (an A/B switch) is scaffolding,
hard-wired or removed after its one measurement. Such switches are named only as developer switches (README.md,
"Developer switches"; "Open items" lists the round's); the launcher shows none. The design is the session notes'
`two-modes-design.md`: its phases A and B are the binaries' part, phase C the launcher's.

What each mode is, as the launcher (`python/gui/runspec.py` `quality_mode`) and the drivers run it today; the bare
binaries take the same `--exact` / `--sampled`, and the other quality and resource options they still accept become
the code's choice or developer pins in phases A and B:

| | Standard DTFE | PS-DTFE |
|---|---|---|
| **Exact** (`--exact`, the default) | the exact cell average (an r3d integral over each cell); the velocity derivatives from exact face integrals of the interpolant (the face gradient, the default wherever it applies) | the exact deposit, every tetrahedron's exact share of every cell; the mean flow's derivatives ∇⟨v⟩ from exact face sums (with `--ps-volume-weighted`, which the launcher and the drivers pass), the stream average ⟨∇v⟩ beside them (`*Streams`) and the mass flux divergence `.a_massFluxDiv` |
| **Fast** (`--sampled`) | the Monte-Carlo mean of the interpolant over the cell, at the binary's own count; the derivatives still from exact face integrals (the face route applies whatever the average) | the sampled deposit, one point per cell axis since 2026-11-14 (D21; `-n 1` from the launcher, the binary's own count, 1, from the drivers; 3 before); the derivatives the stream average ⟨∇v⟩ (the face sums need the exact deposit) |
| where a part does not apply | the 2D build keeps the deposit's own gradient in both modes | Exact in 2D or with `--ps-halo-release`: the stream average, no `.a_massFluxDiv` (the launcher's Checks list says so) |

**Phase C, the launcher (2026-11-13).** Built in three steps on a scratch copy, then reviewed, and committed on the
scratch line as `1f8511b`. C1, the job specifications: one `quality` field ("exact", first and the default, or "fast")
replaces fifteen saved settings (`runspec.REMOVED_SETTINGS`; the Custom job keeps `precision` and `vertex_mass`, the
Pipeline job its pixel sub-samples); what a quality implies is one function, `runspec.quality_mode`, which the three
job kinds read (`mode()`), so the later step "the binary decides this itself" is a deletion there. C2, the run tabs:
one Quality pair (Exact, Fast) with its grey line and a grey "Runs on" line under the machine use; the controls of the
design's section 3.4 (38 by its count) and the three cell-average menus gone with their rules, settings keys and
timing messages; a folded *More* group per tab. C3: three built-in presets (Quick look, Standard, Publication; "Exact
deposit" gone); Explore's zoom pair ("Point values (instant)" / "Exact cell averages (recompute)") and no residency
control; the Runs tab's Exact / Fast; a sweep of the texts. The review (two readers and a judge) found no high point
and three medium ones, all fixed: the time estimates come only from runs of the same quality (`results.time_estimate`,
its `exact` argument), an older queue brings the one-time status note too and the note names the single precision of
the Grids and Pipeline tabs, and the queue's lines say Exact / Fast. Old saved jobs, queue entries, presets and run
logs load into the two modes (sampled is Fast, anything else Exact). Still passed by the launcher until the binaries
choose them (phases A and B): the device and the GPU share, the volume weighting, the vertex masses (an input
description, below), the merge order and the plane conventions, PS-DTFE Fast's count, and the exact zoom's `--gpu`;
the parallel insert is not forced (the binary's own rule), and nothing is passed for the split, the concurrency and
the memory budget.

Status: built on the scratch line; verified there by the four Python launcher suites, measured on 2026-11-13 on these
Python files: the job specifications (`tests/py_gui_runspec_test.py`) 444 passed, 0 failed; the end-to-end suite
(`tests/py_gui_app_test.py`) 396 passed, 0 failed; Explore (`tests/py_gui_explore_test.py`) 267 passed, 0 failed; the
second end-to-end suite (`tests/py_gui_app2_test.py`) 58 passed, 0 failed, 1 skipped; and a golden command comparison
identical to the launcher before the change (its cases, by the builders' report: every run tab's command and
environment for both estimators, both qualities and the three machine uses, on a faked GPU build and a faked CPU-only
build). Not checked: the native macOS look in a real window (the screenshots are the offscreen Fusion style), the time
estimate on real run logs (only on the suite's made-up records), the battery and the Linux check.

Left open from the review (low points, each with the judge's proposal):
- A queue entry saved by an older launcher keeps its old title (", exact" in lower case, or nothing for a sampled job)
  until it is queued again: `QueuedJob.from_dict` migrates its settings, not its summary. Proposed: leave it (the line
  goes with the job).
- `--ps-linear-deposit` typed into the Custom tab's *Extra options* is accepted, but the tab always adds
  `--ps-volume-weighted`, so the binary refuses the run when it starts. Proposed: one `problems()` error in
  `runspec_custom.py` with its test in the next launcher round, or none if the design's Q3 a) deletes the linear
  deposit.
- The golden script's gaps: non-default fields, caustics off, a resume, the scratch and tessellation-cache folders,
  the halo release, double precision, 2D with a window, a separate Lagrangian file, a TNG Pipeline with a real
  particle count, an old-data load. A reader's wider comparison of commands found no difference in the cases it
  covered. Proposed: fold it into the golden script before the next round that touches a command builder.
- `results.RunRecord.settings()` still returns the removed keys (the loaders migrate and drop them). Proposed: trim
  them when the log-spelling checks read the record's own properties.
- The name `REMOVED_SETTINGS` although the Custom job keeps `precision` and `vertex_mass` (its comment says so).
  Proposed: leave it.
- The PS-DTFE Exact grey line wraps with its last word alone at the left pane's minimum width (`app.py`, 690 px; the
  text is fixed, nothing is clipped). Proposed: accept it.

**Phase C, the opt-in Advanced settings (2026-11-13, the owner: "an advanced mode in the GUI that lets the user ticker
with the settings more? but make it opt in").** View > Advanced Settings (off by default, one settings key) brings
back nine removed controls as overrides of a job's mode, in an 'Advanced' group on the three run tabs, each on
Automatic first: Device, Samples per cell axis, Velocity derivatives, Partitions per axis, Concurrent triangulations,
Memory budget, Parallel triangulation, Merge order, Precision (Grids and Pipeline). The job kinds carry them in one
field, `advanced` (saved only when not empty); `runspec._refusal` is the one rule of which is passed (`apply_advanced`
puts the five a mode holds into `mode()`; `job_partition`, `job_concurrency`, `job_memory_budget` and `job_precision`
give the other four), and every builder reads only those, so when phases A and B let a binary decide a setting itself,
its override goes with one rule there. The flags and knobs an override passes, which the clean-up must keep alive
while the override exists, are in the table below (each is still read by the line's run scripts or binaries: checked
by a search on 2026-11-13). Built in two steps on a scratch copy (the job specifications, then the windows), reviewed
by two readers, and one medium point fixed (a row hidden for the loaded job kept its value, so the next read put a
cleared setting back). Status: built and committed on the scratch line; checked by the four launcher suites (457, 396,
267 and 78 passed, 1 skipped, 0 failed), the two-mode golden comparison (identical, 118 cases) and a golden comparison
of every override against the launcher before the two modes (identical, 498 cases; left out: the four cases it now
refuses with a warning, where the old launcher passed the setting: the GPU on a CPU-only build, double without the
double programs, the parallel insert without TBB, the completion-order merge beside extra options that name a merge);
the screenshots with the switch off pixel for pixel the two-mode ones; the battery's other stages, the Linux check,
the native window and a real run with a setting not run. Kept differences from the first design: standard DTFE gets no
Samples and no Velocity derivatives row (the launcher before the two modes had none for it); presets hold no override;
a row that holds a value the job cannot use stays shown, noted "Not passed", so it can be set back. Left open (low
points): a saved integer beyond the float range raises in `_advanced_value` (only a hand-edited store can hold one;
check the range first); Device CPU beside an extra `--gpu` typed on the Custom tab runs on the GPU while 'Runs on'
says the CPU (the old launcher did the same; one more `_refusal` rule would settle it); the Grids estimate line does
not say "(float64)" under Double; the queue's detail pane prints the Advanced info line with a warning sign.

| key (row) | Grids, PS-DTFE (`run_ps_dtfe.sh`) | Grids, standard (`run_dtfe.sh`) | Pipeline (`run_ps_pipeline.sh`) | Custom (the binary) |
|---|---|---|---|---|
| `gpu` (Device) | `-m` and `DTFE_GPU`/`PS_GPU`/`PS_METAL` 1 or 0 | `-m` and `DTFE_GPU`/`DTFE_METAL` | `DTFE_GPU`/`PS_GPU`/`PS_METAL`; `-m` in the memory check | `--gpu` (with `--gpu-share auto` under PS-DTFE Exact) |
| `nsub` (Samples per cell axis) | `-n N` | none | `SUBSAMPLES=N`, `AVG_SUBSAMPLES=N` | `--sampled --subsamples N` |
| `face_gradient` (Velocity derivatives) | `FACE_GRADIENT=0` (stream average); exact face integrals pass what Automatic passes | none | `FACE_GRADIENT=0` | `--per-tet-gradient` / `--face-gradient` |
| `partition` | `PARTITION="p p p"` | same | same | `--partition p` per axis |
| `max_concurrent` | `MAX_CONCURRENT=n` | same | same | `--max-concurrent n` |
| `mem_budget_gb` (Memory budget) | `DTFE_MEM_BUDGET_GB=v` | same | same | same, in the run's environment |
| `parallel_triangulation` | `PARALLEL_TRIANGULATION=1`, `PS_PARALLEL_TRI=1` | none | same as PS-DTFE | `--parallel-triangulation` |
| `ordered_merge` (Merge order) | `PS_MERGE=completion-order`, `ORDERED_MERGE=0` | none | same as PS-DTFE | `--ps-merge completion-order` |
| `precision` | `DTFE_PRECISION=double` | same | same | (its own Precision row: the -double program) |

**Phase C, the finishing round (2026-11-13, the owner: "shouldnt there be a CPU+GPU option as well? ... first fix all
open things regarding the launcher").** The Device row has three choices, CPU, GPU and CPU + GPU; the third needed one
new spelling in the program, `--gpu-share mixed` (auto's share model in every run mode; a number from the launcher
would have skipped the model's guards), and an explicit GPU now passes share 1. The table above holds with two changes
in its `gpu` row: Device GPU adds `GPU_SHARE=1` / `--gpu-share 1`, CPU + GPU `GPU_SHARE=mixed` / `--gpu-share mixed`,
both only for PS-DTFE's exact deposit in 3D. The low points left open by the Advanced settings and by the two-mode
review are fixed (the change log's entry lists them), one closed: `results.RunRecord.settings()` keeps the removed
keys, which a dozen suite checks read and no user sees. The Explore tab got contour lines on the map and in the 3D
view, with a level slider (the change log's entry; `python/gui/contours.py`). Status: built and committed on the
scratch line; checked by a build, `tests/run_mode_check.sh` (single 68, double 58 passed), the four launcher suites on
the merged tree (468, 396, 332 and 82 passed, 1 skipped, 0 failed), the golden comparisons and five real small runs;
the battery's other stages, the Linux check and a full-size run not run.

**Phase C, the void round (2026-11-14, the owner: "the void detection and visualization/projection onto a 2D plane
and in the 3D view is really off", and the slider "more levels or even continuous").** The detection was right and
is untouched (`python/dtfelib`: minima of the density smoothed in 3D, the Hessian's eigenvalues); the drawing was
wrong. Explore now draws each void as the ellipsoid where the quadratic fit around its minimum returns to the mean
density, as exact sections on the map and the same ellipsoids in the 3D view, can show the voids' own field (the
*3D* tick beside *Smoothing*), and has a continuous contour slider (the change log's entry). Status: built and
committed on the scratch line; checked by the four launcher suites (468, 396, 388 and 82 passed, 1 skipped, 0
failed) and by real-data pictures; the battery's other stages and the Linux check not run. Open, the owner's call:
Explore draws the catalogue's distinct sample only, so the deepest minima, whose fit the catalogue's size limit
drops, carry no ellipse.

**The measurement night (2026-11-14, the owner: "go for the measurement night after the current code
implementation run").** Phase A's measurements in one batch: nine 512^3 runs of TNG100-3-Dark, z = 0, one at a time,
in `--run-mode balanced` on the planner's own plan, on the binaries built from the scratch line's batch, with the
production field set and the 4096^2 sample plane, and one repeat. Wall time and peak memory footprint
(`/usr/bin/time -l`):

| run | what differs | wall | peak | plan |
| --- | --- | --- | --- | --- |
| PS-DTFE Exact as shipped | `--exact --face-gradient` | 1603.65 s | 32.25 GB | 2 partitions at once (budget 36.2 GB) |
| the same command, repeated | | 1327.94 s | 34.69 GB | 4 at once (budget 41.2 GB, as in every run below) |
| Fast with 1 point per cell axis | `--sampled --subsamples 1` | 552.77 s | 26.64 GB | 4 at once |
| Fast as shipped | `--sampled --subsamples 3` | 2371.19 s | 27.23 GB | 4 at once |
| exact deposit, stream-average derivatives | `--exact --per-tet-gradient` | 1222.77 s | 29.84 GB | 4 at once |
| Exact with the GPU face kernel | `DTFE_GPU_FACES=1` | 1646.84 s | 30.98 GB | 3 at once |
| the same with `DTFE_FACE_F32=1` | | 1698.95 s | 31.98 GB | 3 at once |
| standard Exact on the GPU | `--exact --gpu` | 450.64 s | 41.72 GB | 3x3x3, 4 threads, look-ahead |
| standard Fast on the GPU | `--sampled --gpu` | 428.14 s | 40.31 GB | 3x3x3, 4 threads, look-ahead |
| standard Fast on the CPU | `--sampled` | 407.73 s | 34.51 GB | 2x2x2, 6 threads |

Every number here was re-read from the runs' logs; each is ONE run. The two GPU face kernel runs are one
configuration (below) and differ by 3.2%, which is the only noise figure the night has. What the runs say:

- *The reference is the repeat.* The first Exact run started with 40.2 GB free, so its budget was 36.2 GB (free less
  the 4 GB headroom) and its plan 2 partitions at once; the repeat planned 4 and wrote the same 28 files byte for
  byte. The repeat's memory model predicted 40.73 GB against 34.69 GB measured (-15%).
- *One point per axis is 2.4 times faster than Exact* (552.77 against 1327.94 s; the 25% bar of the PS Fast question
  is passed). Its density per cell is far from Exact's: |relative difference| median 0.271, cells below 0.05 of the
  mean 31.470% against 19.884% (the mean is 1.000000 in both). After the Gaussian smoothing void work uses the
  difference is small: median 8.12e-03 (99% of the cells below 3.77e-02) at 2 cells, 1.65e-03 (1.02e-02) at 4 cells,
  2.67e-04 (1.88e-03) at 10 cells. The pipeline's void catalogue at the production smoothing (10 cells), built on
  both density grids: 495 minima on Exact and 497 on one point, 103 and 102 resolved, 83 and 82 in the distinct
  sample; every one of Exact's 495 minima lies within one cell of one of the other's; the matched voids' R_v
  differs by a median of 8.0e-05 (relative; one void's by 10.3 Mpc), their fitted semi-axes by a median of 7.7e-04
  (largest 1.9e-02). The distinct sample itself moves with its membership (three minima change their resolved or
  distinct flag): its median R_v is 7.418 Mpc on Exact and 7.498 Mpc on one point (+1.1%), its mean R_v 11.89 and
  12.17 Mpc (+2.4%). The velocity divergence, against the exact deposit with the same (stream-average) derivative:
  rms difference over the reference's rms 1.46 per cell, 0.104 at 2 cells, 0.049 at 10 cells, where it stays.
- *Fast as shipped (27 points per cell) took 1.79 times Exact's wall time.* Its partitions take the device in turns:
  the four lanes spent 6445.3 s in the GPU deposit (the sum over 215 passes; 143.4 s over 213 passes with one
  point) in a partition loop of 37m 46s, and 8242710 tetrahedra went to the CPU (363579 with one point). The same
  configuration on the bench of 2026-11-09 (an older build, before the run modes) took 1259.91 s with 1507.8 s
  in the GPU deposit, beside 1156.21 s for the exact deposit with the same derivatives there (1222.77 s now). Why
  the sampled deposit's GPU time is 4.3 times that of 2026-11-09 was NOT measured (the night's runs have no
  `DTFE_GPU_TIMING` lines): balanced mode's GPU work units (0.05 s against 0.25 s in performance mode) are a
  candidate, unmeasured; a reading of the uncommitted batch's diff found no change on this path. One timing pair
  (balanced against performance, `DTFE_GPU_TIMING=1`) settles it; it matters only while 27 points stay selectable.
- *The stream-average route is 7.9% faster than Exact* (one run each). Its density is Exact's byte for byte, and its
  divergence, gradient, shear and vorticity files are Exact's `...Streams` files byte for byte.
- *The GPU face kernel gives no gain.* Its 216 passes took 309.535 s in the device's hold, where the CPU walk beside
  the deposit takes 2042.72 thread-seconds of which 71.1 s after the device call has returned; its face records
  cost 0.476 GB per lane, with which the planner fitted 3 partitions at once instead of 4. `DTFE_FACE_F32=1` beside it
  changes nothing (the kernel implies that arithmetic: the two runs' 25 files are byte-identical). Of the 24
  complete files 17 are Exact's byte for byte and the 7 made from the mean flow's face gradient differ (divergence,
  gradient, shear, vorticity, the V-web and its eigenvalues, the mass flux divergence); by how much was not measured,
  because both runs
  ended with exit 1 while writing the last point files: the output disk ran full (the driver allowed 28 GB for a
  set, an Exact set is 33 GB; read from the folder sizes, the error line names no cause). Their computation was
  complete, so the times stand.
- *Standard DTFE.* Fast on the GPU and on the CPU are 5.0% apart, one run each: not separable on time. The solid
  difference is memory: the CPU run needs 5.8 GB less. Their 10 files all differ (27 against 8 partitions; density rms difference 3.58e-05 of the rms).
  Exact costs 5.3% over Fast on the GPU and 10.5% over Fast on the CPU: Fast keeps the face gradient pass (630.5 and
  617.6 thread-seconds), so the two modes differ in the deposit alone (223.4 against 64.52 s of GPU deposit, summed
  over the 108 calls). On 2026-11-09, on an older build with 10 threads, the same three configurations took 439.46,
  433.31 and 314.71 s.

What follows is the owner's call and is NOT decided here: PS Fast as one point per axis (the numbers above), or else
the stream-average route, whose gain is 7.9% (**decided 2026-11-14: one point per axis**, the next paragraph and
D21); the GPU face kernel and `DTFE_FACE_F32` out (**kept for now, 2026-11-14: a clean-up item**, the paragraph
"The GPU face kernel stays for now" below and D22); standard Fast on the CPU,
and whether a standard Fast that differs from Exact by 5 to 10% is worth a mode. Not done: the kernel's difference
from the CPU walk in numbers, the timing pair above, the shear after smoothing (per cell its rms
difference is 1.78 times the reference's rms), any run in performance mode, any other grid size or box.

**PS-DTFE Fast is one point per axis (2026-11-14, the owner: "Yes, one point").** Asked "Should PS-DTFE Fast become
one point per cell axis, or the stream-average route?", the owner chose one point. Changed on the scratch line: the
program's default count of the sampled deposit is 1 (`src/user_options.cc`): `--sampled` without a count writes the
grids of `--sampled --subsamples 1`, and `--subsamples N` keeps its meaning (`--subsamples 3` the earlier grids). The
exact deposit keeps its bytes (an exact run without a count holds 3 where the exact pass still reads the count,
`kPsExactSliverSubsamples`; a run with no `_a` field keeps it too) and standard DTFE is untouched. The drivers pass a count only when `SUBSAMPLES` is set,
so the program's default is their Fast, and `run_ps_pipeline.sh`'s `run_settings` reads a sampled command without
a count as 1 (a run log that names none by its header's count line, so an older set made at 3 is recomputed). The launcher's Fast passes 1 (`runspec.quality_mode`); its Advanced *Samples per cell axis* stays 1 to
6, Automatic 1. The tests that checked the 27-point deposit without naming a count pin `--subsamples 3` (the change
log's entry). Still open: the other calls of the paragraph above; the timing pair for 3 points per axis (balanced
against performance, `DTFE_GPU_TIMING=1`), which still matters because the count stays selectable; 2D builds inherit
the default, measured in 3D only; `tests/dev/bytecmp.sh`'s five sampled PS-DTFE runs name no count,
so they differ from a reference set built before this change (regenerate it, or give them the count). What was
checked: compiled, a byte proof against the earlier programs on a small box, and the touched suites once (the change
log's entry); the battery's other stages, the Linux check and a 512^3 run of the new default are not done.

**The GPU face kernel stays for now, as a clean-up item (2026-11-14, the owner: "keep it for now, but put it in the
plan for clean up").** `DTFE_GPU_FACES=1` and `DTFE_FACE_F32=1` (which the kernel implies) stay as they are:
developer switches, off by default, changing no result. The measurement night found no gain at 512^3 in balanced
mode: 1646.84 s and 1698.95 s against Exact's 1327.94 s. The kernel's 216 passes took 309.535 s in the device's hold,
the CPU walk it replaces runs beside the deposit, and its records cost 0.476 GB per lane, with which the planner
fitted 3 partitions at once instead of 4. For the clean-up phase (the face kernel's step of phase A): remove the
kernel (`metal/ps_face.metal`), its host code, the planner's term, the checkpoint key's line and its tests
(`tests/face_sweep_check.sh`, P17 and P18 of `tests/ps_face_gradient_check.sh`, T16 of `metal/validate_deposit.cpp`),
and `DTFE_FACE_F32` with it -- unless a measurement made first speaks for keeping it. Such a measurement would be:
its grids against the CPU walk's in numbers (not measured: the night's two runs ended on a full disk), a run in
performance mode, a run whose budget holds the fourth lane, a larger grid or another machine. The owner decides at
the clean-up.

**Phases A and B, the binaries (to do in the clean-up phase; not built).** The design's steps: A1 one mode with its
header lines (a quality line, PS-DTFE's conventions line), velocityStd_a taking `--average-method 2` itself; A2 the
gradient route's switches as developer pins; A3 the float-sum switches out; A4 to A7 the measurements that settle the
PS Fast count (settled 2026-11-14: one point per axis, D21) and the round's developer switches (the walk beside the
deposit -- built as the default on the scratch line on 2026-11-13, `b62eb94`, with `DTFE_FACE_WALK_EARLY=0` kept for
one more measurement of the pair --, the word
merge, the face kernel and its arithmetic (measured 2026-11-14 and kept for now: the paragraph above), the planner's
time objective), each switch hard-wired or removed after its
one measurement; B0 the readers
of the new header lines first; B1 PS-DTFE's volume weighting in the binary; B2 the device and the GPU share chosen by
the binary (one table instead of the drivers' and the launcher's copies); B3 the drivers passing only the mode; B4 the
help and the docs; B5 deleting what the owner does not keep (Q3); B6 the alias release (P6.6). So the binaries choose
the device, the share, the volume weighting, the merge order and the plane conventions themselves, and the launcher's
matching pins in `quality_mode` and its command builders (and the exact zoom's `--gpu`) are then deleted. The
tetrahedron masses are not among them: the design keeps `--ps-vertex-mass` as a description of the input (perturbed
initial positions, such as TNG's; a lattice needs none), which TNG jobs always pass and the Custom tab's *Initial
positions* row sets. The owner's questions that come first: Q1, PS-DTFE Exact with bytes free of the split; Q2, PS
Fast if no sampled count is clearly faster than Exact (answered 2026-11-14 by D21: one point per axis is); Q3, the
old PS conventions (mass-weighted moments, linear deposit, geometric stream density, legacy plane conventions); Q4,
whether the binary picks its own device and share (a yes reverses D18, P11.4).

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
P2, P8 and P10 status (the morning's 6³ face-check diagnostic in the P10 status; every gate green by 07:35). (D15 and D16 were settled on 2026-11-09: the decision table.) The rest of R (the exact deposit with the face outputs on the other snapshots, the science
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
| D6 | `--fixed-point-merge` as default | **leave it off** if the P2.3 reruns park at most a handful of partitions. **Decided 2026-11-11 (the owner, after the explanation): the merge stays opt-in (the P2 status of 2026-11-09 has the parked counts the condition asks for).** | make it the default for PS runs above some N |
| D7 | split `depositLoop` (1010 lines, hot) | **no**, unless P9.2 needs the restructuring; the proof would be bytes plus disassembly plus a ±1% timing check | split into a `DepositCtx` plus `psDepositExactTet` |
| D8 | launcher Advanced sections | **collapse** Resources and the expert PS options, with the state remembered | only shorten labels and tooltips |
| D9 | when to remove the flag aliases | **the release after** the one that introduces them | keep them indefinitely, hidden |
| D10 | **upstream** dedupe and deletions (P1.4's upstream half, P3.9) | **golden tests yes**; the dedupe and deletions **no**, since the repository is still a fork of upstream. **Changed 2026-11-07:** the owner no longer tracks upstream ("focus on this fork only"), so upstream-marked code is ordinary code now (D4 and P3.9 may proceed) | do both |
| D11 | the GPU backends | **Metal-only**: CUDA/HIP has never run on NVIDIA or AMD hardware, costs 4.5k lines; keep it on an archive branch (the analysis is in the 2026-11-07 session notes). **Half decided 2026-11-08:** its four CI jobs (`cudaemu` 32/64, `cuda-compile`, `hip-compile`) were removed on the owner's instruction. **Decided 2026-11-08 (evening): remove.** The port (six source files, 4.6k lines), its CPU emulation (`CUDAEMU=1`, `tests/cudaemu_check.sh`), the twin check (`tests/gpu_twin_check.sh`), `GPU_ARCH` and the `DTFE_GPU_EMU_WARP` knob are gone; `make CUDA=1`, `HIP=1` or `CUDAEMU=1` stops with one error line naming `archive/cuda-hip-2026-11-08` (= `f9f6e2f`, the last commit with the port; the branch must be on origin, since five pointers in the tree name it). Linux stays CPU-only. P4.3 step 2 followed (one `gpuRunPass`). | CPU-only after a same-tree A/B; or keep all three |
| D12 | PS velocity derivatives | **write both** the mean-flow gradient (with the caustic sheets) and the regular part ⟨∇v⟩. **Decided 2026-11-11 (owner, 3b): the regular part in full** -- under the face gradient (PS-DTFE's default where the run qualifies, D13) its gradient, shear and vorticity beside the mean flow's (`.a_velGradStreams`, `.a_velShearStreams`, `.a_velVortStreams`, each where its mean-flow twin is written), as its divergence already was (`.a_velDivStreams`); no deposit, kernel or checkpoint change (P10.10). Built 2026-11-11 (not committed) | only one of them |
| D13 | the velocity gradient's route (standard DTFE; PS-DTFE since 2026-11-11) | **DECIDED 2026-11-09 (owner, 14:40): the face gradient is the default** (`--per-tet-gradient` / `DTFE_FACE_GRADIENT=0` keep the deposit's own; the drivers' `FACE_GRADIENT` empty = the binary's default, 0 = per-tet, 1 = face; bytecmp pins the per-tet route, the GPU suites pin it where they test the kernel's gradient; the production grids made before this date hold the per-tet gradient). The record of the decision's path: **keep the per-tet deposit as the default.** **Checked 2026-11-09 (phase R): the condition is not met.** The full-size check (`std100_099_face`: TNG100-3-Dark z = 0, 512³, GPU exact average, 3³) fails the test's bound in 133 of 243 records: 64,978 cells (4.8e-4) above 1e-4, worst 1.2e-2, 99.7% of them in one strip where the **deposit's** float32 region frame rounds (the P10 status); elsewhere, and on a CPU 256³ run with an exact frame, the two agree to an rms of 2–7e-7 with a sparse tail (attributed to float32, not tested). The face path replaces the deposit's gradient once P10.8 (the frame fix) is in and a `std100_099_face` rerun (8 min) passes the restated bound (the P10 status; the demo's G1 keeps 1e-4). Until then `--face-gradient` is the exact side where the frame rounds, except in the degenerate cells, where it keeps the flattened tetrahedron's flux that the deposit zeroes (neither route validated there). **At 6³ (`std100_099_split6_face`, 2026-11-09 06:42) the check nearly passes: 903 cells (6.7e-6) above 1e-4, worst 1.9e-3, worst record rms 2.9e-6, no strip; it meets the proposed restated bound, so the 3³ strip is the frame as predicted (the P10 status).** **P10.8 done 2026-11-09 (afternoon), and its 512³ rerun on the production 3³ plan meets the restated bound (294 cells above 1e-4 = 2.2e-6 pooled, worst record rms 5.4e-7, no degenerate cell, worst 5.7e-4; the P10 status), so the switch's condition holds. The switch itself — the face path as the default gradient of every standard run, which changes every standard gradient output (bytecmp re-taken for the Monte-Carlo runs' derivative files too) and the drivers' default — is the owner's call; `--face-gradient` stays opt-in until then.** **The owner, 2026-11-11: the face gradient for both estimators, read as PS-DTFE's default too (to be confirmed by the owner). Built 2026-11-11 (not committed): PS-DTFE takes the face gradient by default where the run qualifies -- the exact deposit, `--ps-volume-weighted`, no `--ps-halo-release`, 3D (`user_options.cc` `faceGradientBlocker`) -- and elsewhere keeps the stream average and says why in the header's `velocity gradient` line, without stopping; `--per-tet-gradient` and `DTFE_FACE_GRADIENT` in both binaries; `run_ps_dtfe.sh`'s `FACE_GRADIENT` as `run_dtfe.sh`'s (empty = the binary's); the launcher's tick on by default (`face_gradient_applies` / `face_gradient_env` in `python/gui/runspec.py`), Explore reads `.a_velDivStreams` (`OutputSet.face_gradient`); tests `ps_face_gradient_check` P14, `ps_auto_tune_check` T; bytecmp unchanged (its `psx` run is pinned by `DTFE_FACE_GRADIENT=0`), of the tune set's 127 lines the 22 PS lines with the production fields, the exact deposit and no route named move (five of them, an 8 GB machine's at grid 256, to `over_budget=1`: 5.08 to 6.02 GB predicted on a 4.80 GB budget) and none of them under `DTFE_FACE_GRADIENT=0`; 10 of the 16 lines that name `--face-gradient` move with or without it, by D12's twins; the 512³ plan probes: the default GPU plan is the face plan (4 in flight, 32.89 → 38.26 GB predicted), the default CPU plan 3 in flight where it had 6 (39.60 → 38.84 GB). Back to "off by default": `kPsFaceGradientDefault = false` in `src/user_options.cc` and `FACE_GRADIENT_DEFAULT = False` in `python/gui/runspec.py`.** | keep both paths |
| D14 | default field set | **density, velocity, divergence, shear in both drivers (done 2026-11-07)**; the rest on request | — |
| D15 | the PS deposit for the campaign (added 2026-11-09, phase R) | **measure first, then choose by the outputs the analysis needs; one deposit throughout a simulation.** The night does not support "CPU exact costs the same and the GPU is unreliable": like for like at TNG100-3-Dark z = 0 the two tie (1377 against 1379 s, different builds), at the eight other logged redshifts CPU exact took 1.21–1.43× the GPU wall (the 27 logged runs by ratio: 11.4 h against 9.8 h), the TNG50 run's length is attributed to its hand plan (mc 2 against the logged mc 6; no park room, so 58 spills), not to its GPU failures (about 4% of the wall), and both deposits conserve mass (mean ρ/ρ̄ = 1 to 3e-8, no empty cell). But the face grids (grad⟨v⟩, div p for R's void mass flux) exist only with CPU exact, whose stream counts and caustic bits are also fuller (32.9% multi-stream cells against 24.5%). The measurement that settles the cost: TNG50-3-Dark z = 0, CPU exact + `--face-gradient` + `--sample-points … --pts-vel-grad`, after an `--auto-tune-report` (0.75–2.4 h by scaling, up to 5.5 h; all 10 cores). A CPU exact command must carry the points (`ps100_099_face512`'s did not). **Bench 2026-11-09 (P11.1; the committed tree, TNG100-3-Dark z = 0, 512³, with the points):** exact CPU 1133 s, exact GPU 1156 s, sampled GPU 1260 s, sampled CPU 1644 s. With the face gradient, which since D17 also comes with the GPU exact deposit: exact CPU 1737 s (46.4 GB max RSS, 30.4 GB peak footprint), exact GPU 1454 s (49.6 GB, 33.4 GB) -- both inside the 41.2 GB budget by footprint (corrected 2026-11-10: the max RSS had been read as the peak). **Decided 2026-11-09 (evening, owner; D20): the exact deposit, the default of both binaries since; the device stays D18's.** The face runs waited on P11.7 item 1, done on 2026-11-10: j stands as measured, l ran with 4 in flight where the corrected plan chooses 3. **The owner, 2026-11-11: the face gradient for both estimators; PS-DTFE's default was built the same day (D13).** | GPU sampled + points as logged (byte-reproducible at a given split; the TNG50 run, whose one pass fell back to the CPU, differs; faster at z > 0, about 7 cores left free), without the face grids |
| D16 | the frame fix before the campaign (added 2026-11-09, phase R) | **yes: P10.8 first, both parts.** The frame part is established by the 6³ face-check run (`std100_099_split6_face`, the P10 status): it removes the rounding strips (the 3³ plan's; a 6³ plan has none) and lets D13 switch under the restated bound. It does not, by itself, make the standard grids reproducible across splits: the ~1e-5 box-wide split dependence (the P2 status) is in the face pass too and is not the frame; its leading candidate, the per-tetrahedron gradient's float labelling dependence, is P10.8's second part, an hour to test and small to fix. Together they are what could bring the cross-split differences down to the float sum order (~1e-7); byte identity would need fixed-point sums as in PS-DTFE. Each moves the standard exact-average grids (bytecmp re-taken); effort M for the frame (kernel and host), S for the gradient; verify as P10.8 says. The PS campaign waits on it only if P10.8 finds the same frame in the PS deposits. **Done 2026-11-09 (the P10 status of that afternoon). PS-DTFE's deposits name their cells from the full box's corner and spacing in every partition (`ps_interpolation.cc`: `region` stays the box in a partition's options, the GPU's `boxLo` is the box's lower corner), so they have no sub-domain frame and the PS campaign never waited on this; the ~1e-5 PS-DTFE moved across splits at z = 1 (the P2 status) has another source, not established (the candidates and the analysis that separates them: the P2 status, "Cross-split analysis").** | run the standard campaign now with its ~1e-5 reproducibility across splits and builds recorded, D13 at "keep both paths" |
| D17 | the PS face gradient on the GPU exact deposit (added 2026-11-09, after the owner's rule "one method, sampled and exact, on whichever device is faster") | **DONE 2026-11-09 16:00 (code; P11.5): `--face-gradient --gpu` runs, the face sums from one CPU walk after the GPU deposit, byte-identical mean-flow grids to the CPU run's; its speed against the CPU exact run is the bench's run m.** Before: the PS face gradient (the mean-flow derivatives and the mass flux the void analysis needs) exists only on the CPU exact deposit, which refuses `--gpu`; the GPU exact PS deposit has no face sums and has never run at production size, so its speed against the CPU's is unmeasured. Without it the PS campaign is CPU exact (1.2–1.4× the GPU sampled wall at z > 0, equal at z = 0, the face pass up to +21%). Item sketch (P10.9): the face sums as a CPU pass beside the GPU deposit (the standard binary's way: the pass walks the tessellation, which the GPU run keeps), the volume-weighted mean from the GPU's grids; effort M. **The owner, 2026-11-11: the face gradient for both estimators; PS-DTFE's default was built the same day (D13).** | leave the PS face gradient on the CPU; the campaign as D15 says |
| D19 | run-to-run byte reproducibility vs the parallel insert's default (added 2026-11-09 16:30, P11.3) | (Decided 2026-11-11, below.) **open, for the owner.** The default rule inserts tessellations of ≥ 1e6 points in parallel where cores are free (3.1× on the triangulation of a low-concurrency 512³ run, nothing at the auto plan's concurrency); two identical runs then differ at rounding in the CPU deposits' float sums, the face pass's last bit and PS-DTFE's velocity moments (the PS density and the GPU exact grids stay byte-identical). The owner's instruction of 13:50 ("turn the TBB insertion on for all binaries") stands; its premise -- the deposits no longer depend on the construction order -- holds for the GPU exact average and the PS density only. Alternative: `--sequential-triangulation` as the default and the parallel insert chosen by the drivers for runs with few tessellations in flight (1024³, `MAX_CONCURRENT` ≤ 2), or a canonical vertex order in PS-DTFE's deposit (the standard `mcVertexOrder`'s way) so its moments stop depending on the order (effort M). **Bench 2026-11-09 (P11.1):** the rule engaged in neither production configuration at 512³ (standard: the auto plan leaves one core per tessellation; PS-DTFE: 771,057 padded points per 6³ partition, below the threshold), so there the default costs no reproducibility and gains no time; the choice matters for the memory-bound runs with few tessellations in flight (1024³). **2026-11-11 (the owner): a canonical vertex order in PS-DTFE's deposit is being designed, to be built only if the design shows that byte-identical repeats are reachable; the rule stays until then.** **Decided 2026-11-11 (the owner: "a or c depending on how c can be done"): c, built.** The design showed byte-identical repeats reachable for every grid of the single-precision PS-DTFE (CPU and GPU, sampled and exact, partitions under the index-order merge): its grids are order-free fixed-point sums, and what moved was each tetrahedron's share formed in its vertex-0 frame. The cells are renumbered once after the tessellation exists and after every cache load (`ps_vertex_order.h`: Lagrangian lexicographic order, an even permutation of CGAL's so the parity sign holds), so every reader -- deposit, GPU record and kernel, vertex densities, points, `--serve` -- sees one numbering; `--ps-caustic-cusps` orders its stencil strictly; `DTFE_PS_VERTEX_ORDER_LEGACY=1` keeps CGAL's numbering and the stencil's address order (A/B), and the checkpoint key names it (`vertex-order=legacy`: no `--resume` across the switch). Not reached: the double build, `DTFE_GPU_FLOAT_*`, the vertex densities' cell-order sums (`--ps-linear-deposit`, the points' 'linear' density, the vertex-mass c_v), standard DTFE (its CPU float sums). The rule (the parallel insert by default) stays. **Measured 2026-11-11:** two runs in the canonical order under the parallel insert are byte-identical (2.1e6 points at 128³, the exact deposit on the CPU and on the GPU), where two runs in CGAL's numbering differed in the velocity and in 8 density cells -- the 2026-11-09 text's "the PS density ... stay byte-identical" was the 64³ set's; the one-time move against the binaries before is last bits in most cells and far more in a few cells at caustics (P11.3, "the size of the one-time move"). Record: P11.3, status 2026-11-11. | keep the rule without the order (option a) |
| D18 | the device chosen by the program (added 2026-11-09, the same rule) | (Decided 2026-11-11, below.) **open (the mechanism exists, the rule awaits the bench; P11.4).** The drivers' `DTFE_GPU=auto` (`gpu_auto_resolve`) already picks the device per build and, for PS-DTFE, per deposit; the binaries' `--gpu` stays explicit; the auto-tuner plans partitions and threads around it but never picks the device. After P10.8 the GPU and the CPU exact deposits of standard DTFE differ only by float rounding, so the choice is speed and memory alone: a rule in the tuner (a GPU build with a device present → `--gpu` for the standard exact average, which is 6.6× the CPU on the demo; the PS exact deposit stays on the CPU while D17 is open) with `--gpu`/`--no-gpu` as overrides; effort S. The drivers and the launcher then stop passing `--gpu`. **Bench 2026-11-09 (P11.1):** the demo's 6.6× does not carry to 512³: the standard run is CPU-bound, GPU exact (439 s at 3³) ties CPU exact (431 s at 2³) and the CPU sampled run is the fastest (315 s); for PS-DTFE exact GPU ties exact CPU (1156 against 1133 s) at a quarter of the CPU seconds, and D17 is done. The rule's table still needs the GPU at 2³ (P11.7 item 3); the tuner's face booking is corrected (P11.7 item 1, 2026-11-10: the GPU face run now plans 3 in flight where it ran 4). **Closed 2026-11-11 (the owner): the drivers and the launcher pick the device (per run mode, P13.1 item 21); the binaries keep `--gpu` explicit.** | keep `--gpu` explicit (the drivers pass it) |
| D20 | the `_a` average's flags and default (added 2026-11-09 evening, after the P11.1 bench) | **DECIDED 2026-11-09 (owner): `--exact` / `--sampled`, exact is the default for both estimators** ("exact is the right strategy across the board, but keep sampled as an option"). Reverses D2's canonical name `--exact-average`, which stays a deprecated alias of `--exact` in both binaries (as `--ps-exact-deposit` does in PS-DTFE), removed per D9; settles D15. The bench behind it: the exact average costs about the sampled one's wall without its sampling noise (standard DTFE 439 against 433 s on the GPU, 431 against 315 s on the CPU; PS-DTFE exact CPU 1133 s, GPU 1156 s against sampled GPU 1260 s, CPU 1644 s). The rules, what moves and the gates: P12. Implemented 2026-11-09, gates green on 2026-11-10, committed as `39583f0` | keep the sampled default with `--exact-average` opt-in (D2) |
| D21 | PS-DTFE Fast's count (P14; added 2026-11-14, after the measurement night; this table's D21, not P13's design item D21) | **DECIDED 2026-11-14 (owner: "Yes, one point"): one point per cell axis**, the program's default for the sampled deposit (`--sampled` = `--sampled --subsamples 1`); `--subsamples N` stays selectable, in the launcher as the Advanced *Samples per cell axis* (1 to 6, Automatic 1). The night behind it (TNG100-3-Dark z = 0 at 512^3, balanced mode, one run each): 552.77 s against PS-DTFE Exact's 1327.94 s (2.4 times faster) and 2371.19 s at 3 points per axis; the void catalogue at the production smoothing (10 cells) 497 against 495 minima, 82 against 83 in the distinct sample (P14, "The measurement night"). The exact deposit and standard DTFE unchanged. Built on the scratch line 2026-11-14 (P14, the paragraph after the night's record) | the stream-average route instead (`--exact --per-tet-gradient`; its gain in P14), or 3 points per axis as before |
| D22 | the GPU face kernel (`DTFE_GPU_FACES`) and `DTFE_FACE_F32` (P14; added 2026-11-14, after the measurement night; this table's D22, not P13's design item D22) | **KEPT FOR NOW 2026-11-14 (owner: "keep it for now, but put it in the plan for clean up")**: developer switches, off by default; no gain measured at 512^3 in balanced mode (1646.84 s and 1698.95 s against Exact's 1327.94 s). A clean-up item (P14, "The GPU face kernel stays for now"): removed there unless a measurement made first speaks for keeping it. | remove them now (the recommendation after the night), or keep them for good as documented developer switches |

**The owner's other answers of 2026-11-11** (to the questions the leftovers batch left; decisions, not code: the code
follows in later batches, recorded under P11.8 as each is built). D6, D12, D13, D15, D17, D18 and D19 carry theirs in
the table.

- The launcher's Custom tab is to pass `--ps-merge index-order` for PS-DTFE, as the drivers do, unless its extra options
  already name a merge (P6.7f's R7 note).
- The three small fixes that change output bytes at about 1e-8 (one compiled order for Metal's linear weights, P4.4's
  finding; the boundary-plane section of tetrahedra outside a non-periodic grid, P11.7 item 5; a run-wide fixed-point
  unit, the P10 status after P10.8): the owner first asked for an explanation and then decided, the same day, that all
  three are to be built, before the production campaign. New grids then differ from the grids on disk in the last
  digits, which the owner accepts; the byte comparisons' references move once, after these three and the canonical
  vertex order (D19). The fixed-point merge as a default (D6) had the same question and stays opt-in. The first two are
  built (2026-11-11, decision 2c: P4's status of that day, P11.7 item 5, P11.8's record; gates green). Measured, the two
  move more than 1e-8: the linear GPU grids by up to 7.5e-6 of the rms, with no gain that the totals or the CPU
  comparison show, at +16% of the item passes of a linear GPU run; the boundary-plane section corrects the last layer's
  `.a_massFluxDiv` of a lattice box by 0.37 of the rms. Whether the first stays at that cost is the owner's to say.
  The run-wide unit is built for standard DTFE (P10's status after P10.8) and, as an opt-in, for PS-DTFE's passes
  (`DTFE_PS_RUN_UNITS=1`; the owner's decision of 2026-11-12, after the full-size numbers: +17.3% of wall time at 512³
  for a byte identity that only the merge in words completes; with that merge, built on 2026-11-13 as S6 behind the
  developer switch `--ps-merge fixed-point` (not gated), the units are part of it: P10's status, its bullets on
  PS-DTFE). Standard DTFE keeps its run-wide units as the default; what they cost there, measured the same evening
  (+6.0% of wall time at 512³: the same status), is the owner's to weigh.
- Standard DTFE's planner under `--run-mode balanced`: a plan that steps the threads down to keep the look-ahead is to
  be designed now, and one measured run decides whether it stays (P11.8, P13's open points).
- A GPU kernel for the PS face sweep is the next large item, designed first (the CPU phase with the face walk, P11.1's
  run l).
  First (the owner's answer of the same day) the cheap step: the existing CPU walk beside the device call, built and
  measured, and the numbers back to the owner before any kernel work. Built as an opt-in on 2026-11-12 and measured
  (13.7–16.5% less wall time at 256³, 7.2% on one 512³ pair; P11's status of that day). The kernel followed on
  2026-11-13 behind the developer switches `DTFE_FACE_F32=1` and `DTFE_GPU_FACES=1` (P11's status of that day; not
  gated, not measured). Since 2026-11-13 the walk beside the device call is PS-DTFE's default on the scratch line
  (`b62eb94`, built under the two-mode rule, not gated: P11's status of that day on the default).
- The owner gave the go for P11.8's four confirming runs; they were made on 2026-11-11 (P11.8, "The confirming
  series").
- The sample total that is not a cube (`--samples 20`) goes when the old aliases go (P6.6), and the two suites that use
  it (`dtfe_sampling_check`, `dtfe_mc_average_check`) move to a cube.
- The four run-mode extras left out on purpose (P13.1 item 29: the scratch-write finding, a rho from the hardware, SMT
  detection, a mixed GPU share for balanced) stay out.
- The plans stay as they are; one measured balanced run at `--partition 5 5 5 --max-concurrent 4` is in the running
  series, and then the split choice with the lane share is decided (P11.8's leftovers).
- The three prediction errors that only runs can settle (P11.8's leftovers) get runs in a reduced form: a CPU run in
  the running series, one TNG50-3 rerun on a night the owner picks, and the mixed lane's term (a) read from the
  campaign's own runs.
- The PS face-gradient tick box moves from the folded Advanced group to the visible "Phase-space options" group (P10's
  status).
- The real-run check in `ps_auto_tune_check` (section Z) stays in that suite, so the battery runs it.

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
| `-MMD` dependencies for the `.cu` rules | (Moot since 2026-11-08: D11 removed the `.cu` sources.) Cannot be tested without nvcc/hipcc. Keep the hand-listed headers. |
