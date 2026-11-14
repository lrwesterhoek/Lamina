#!/usr/bin/env python3
"""Headless checks for the GUI's job specifications (python/gui/runspec.py): the commands
RunSpec, PipelineSpec, PlotSpec and CustomSpec stand for, the combinations they refuse, script
export, progress parsing and data discovery; and the GUI's other Qt-free modules: run logs, stale
outputs, time estimates and failure advice (results.py), presets and the binary's own help text
(presets.py), and the Explore tab's grid reader (grids.py). No Qt needed.

Usage: python3 tests/py_gui_runspec_test.py
"""

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _harness import REPO as ROOT, check_ok as check, setup, summary

FIGS_TMP = setup("runspec_figs_", paths=("python/gui",), mpl_agg=False)    # BEFORE runspec is imported

import runspec as rs  # noqa: E402
from dtfelib.estimator import STANDARD_PADDING  # noqa: E402


def levels(spec):
    return {lvl for lvl, _ in spec.problems()}, " | ".join(m for _, m in spec.problems())


@contextlib.contextmanager
def fake_build(kind):
    """The Makefile's stamps of a GPU build ('gpu': Metal and TBB) or a CPU-only one ('cpu'), in a temporary folder
    that runspec._objdir points at inside the block: the quality's device (recommended_gpu) then does not depend on
    what this tree has built (CI builds CPU-only, a Mac with METAL=1)."""
    base = Path(tempfile.mkdtemp(prefix="gui_build_"))
    orig = rs._objdir
    for est in ("ps", "dtfe"):
        for prec in ("single", "double"):
            for dim in (3, 2):
                d = base / orig(est, prec, dim).name
                d.mkdir(parents=True, exist_ok=True)
                (d / (".gpu_mode_metal" if kind == "gpu" and dim == 3 else ".gpu_mode_off")).touch()
                if est == "ps" and dim == 3:
                    (d / ".tbb_1").touch()
    rs._objdir = lambda estimator, precision="single", dim=3: base / orig(estimator, precision, dim).name
    try:
        yield
    finally:
        rs._objdir = orig
        shutil.rmtree(base, ignore_errors=True)


@contextlib.contextmanager
def fake_stamps(stamps):
    """As fake_build, for the builds it does not cover (a pair without TBB, a double pair without the GPU): each
    object folder holds the Makefile stamps stamps(estimator, precision, dim) names."""
    base = Path(tempfile.mkdtemp(prefix="gui_stamps_"))
    orig = rs._objdir
    for est in ("ps", "dtfe"):
        for prec in ("single", "double"):
            for dim in (3, 2):
                d = base / orig(est, prec, dim).name
                d.mkdir(parents=True, exist_ok=True)
                for name in stamps(est, prec, dim):
                    (d / name).touch()
    rs._objdir = lambda estimator, precision="single", dim=3: base / orig(estimator, precision, dim).name
    try:
        yield
    finally:
        rs._objdir = orig
        shutil.rmtree(base, ignore_errors=True)


@contextlib.contextmanager
def double_pair(built: bool):
    """runspec.binary_built answering 'built' for the double-precision programs inside the block, whatever this tree
    has built (the single ones as they are)."""
    orig = rs.binary_built
    rs.binary_built = lambda estimator, precision="single", dim=3: (built if precision == "double"
                                                                    else orig(estimator, precision, dim))
    try:
        yield
    finally:
        rs.binary_built = orig


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gui_runspec_")) / "Illustris TNG"
    try:
        # a fake per-family data root with one simulation, two snapshots, one plane file
        sim = tmp / "TNG100" / "TNG100-3-Dark"
        for n, z in ((4, 10.0), (99, 0.0)):
            d = sim / f"snapdir_{n:03d}"
            d.mkdir(parents=True)
            import h5py
            with h5py.File(d / f"combined_{n:03d}.hdf5", "w") as f:
                f.create_group("Header").attrs["Redshift"] = z
        (sim / "snapdir_004" / "ps_output.a_den").touch()
        (sim / "hires_plane_z64.bin").write_bytes(b"\0" * 24 * 64 * 64)

        _discovery(tmp, sim)
        spec = _run_command(tmp, sim)
        _run_checks(tmp)
        _progress()
        _pipeline(tmp, sim)
        pipeline_options(tmp, sim)
        _plots(tmp, sim, spec)
        _download_merge(tmp)
        _memory_check(tmp)
        _queue(tmp)
        new_features(tmp, sim)
        two_d(tmp)
        cube_cache_and_paths(tmp, sim)
        cube_cache_threads()
        type_labels()
        launcher_fixes()
        figures_root_rule()
        void_outline_geometry()
        run_log_contract()
        ordered_merge_settings()
        cpp_fix_flags()
        pts_conventions_flag()
        pts_conventions_choice()
        estimator_per_snapshot(tmp)
        run_modes(tmp)
        two_quality_modes(tmp)
        runs_on_line(tmp)
        advanced_settings(tmp)
        device_cpu_gpu(tmp)
        finishing_round(tmp)
    finally:
        shutil.rmtree(tmp.parent)
    print("-" * 60)
    return summary(show_zero_skips=False)


def _discovery(tmp, sim):
    """Discovery: the fake data root's simulation, snapshots, outputs and plane file."""
    print("discovery:")
    check("simulations() finds the per-family sim", rs.simulations(tmp) == ["TNG100-3-Dark"])
    snaps = rs.snapshots(sim)
    check("snapshots() lists numbers and redshifts",
          [(s["n"], s["z"]) for s in snaps] == [(4, 10.0), (99, 0.0)], str(snaps))
    check("snapshots() marks existing outputs", [s["done"] for s in snaps] == [True, False])
    check("planes() + plane_points()",
          [p.name for p in rs.planes(sim)] == ["hires_plane_z64.bin"]
          and rs.plane_points(sim / "hires_plane_z64.bin") == 64 * 64)


def _run_command(tmp, sim):
    """RunSpec's command for both estimators: argv, environment, shell line, the quality and the drivers' defaults."""
    print("command:")
    with fake_build("gpu"):
        spec = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4], grid=256, caustics=True,
                          slice_plane=str(sim / "hires_plane_z64.bin"), output_prefix="ps_test")
        argv, env = spec.command()
        line = spec.shell_line()
    check("PS argv: script, sim, grid, GPU (Exact on a GPU build), exact, sorted snapshots (no nSub: the exact "
          "deposit has none, and the count is pinned empty under both names)",
          argv == [str(rs.PS_SCRIPT), "-s", "TNG100-3-Dark", "-g", "256", "-m", "-e", "4", "99"]
          and env["SUBSAMPLES"] == "" and env["AVG_SUBSAMPLES"] == "", f"{argv} {env}")
    with fake_build("cpu"):
        samp = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=256, quality="fast")
        samp_cmd = samp.command()
    check("... Fast passes the sampled deposit's count (-n 1, the quality's: one point per cell axis) and no -e, "
          "pinned by DTFE_EXACT_AVERAGE=0",
          samp_cmd[0] == [str(rs.PS_SCRIPT), "-s", "TNG100-3-Dark", "-g", "256", "-n", "1", "4"]
          and samp_cmd[1]["DTFE_EXACT_AVERAGE"] == "0" and "SUBSAMPLES" not in samp_cmd[1], str(samp_cmd))
    check("PS env carries the physics, slice and resources (no partition, concurrency or budget: the auto-tuner's)",
          env["DTFE_DATA_ROOT"] == str(tmp) and env["PS_CAUSTICS"] == "1"
          and env["PS_VOLUME_WEIGHTED"] == "1" and env["PS_VERTEX_MASS"] == "1"
          and "PARTITION" not in env and "MAX_CONCURRENT" not in env and "DTFE_MEM_BUDGET_GB" not in env
          and "DTFE_PRECISION" not in env
          and env["SAMPLE_POINTS"].endswith("hires_plane_z64.bin") and env["PTS_VEL_GRAD"] == "1"
          and env["OUTPUT_PREFIX"] == "ps_test" and "SCRATCH_DIR" not in env, str(env))
    check("shell_line is a runnable, quoted terminal command",
          line.startswith("DTFE_DATA_ROOT='") and "scripts/run_ps_dtfe.sh -s TNG100-3-Dark" in line
          and line.endswith(" 4 99"), line)
    auto = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4])
    check("auto partition/concurrency pass nothing (the binary auto-tunes)",
          "PARTITION" not in auto.command()[1] and "MAX_CONCURRENT" not in auto.command()[1])
    with fake_build("cpu"):
        dt = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe", quality="fast")
        dte = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe")
        dt_cmd, dte_cmd = dt.command(), dte.command()
    check("DTFE argv uses run_dtfe.sh with no PS options",
          dt_cmd[0] == [str(rs.DTFE_SCRIPT), "-s", "TNG100-3-Dark", "-g", "512", "4"]
          and "FIELDS" not in dt_cmd[1])
    check("standard DTFE at Exact: run_dtfe.sh -e (the exact cell average), named in the queue",
          dte_cmd[0] == [str(rs.DTFE_SCRIPT), "-s", "TNG100-3-Dark", "-g", "512", "-e", "4"]
          and rs.job_summary("grids", dte).endswith(", Exact")
          and rs.job_summary("grids", dt).endswith(", Fast"), str(dte_cmd[0]))
    # run_dtfe.sh's default is the exact average, and -e only switches it on: the job pins the knob
    check("... the cell average is pinned both ways (DTFE_EXACT_AVERAGE=0 for Fast, 1 for Exact; the older "
          "name DTFE_EXACT the same, until P6.6)",
          dt_cmd[1].get("DTFE_EXACT") == "0" and dte_cmd[1].get("DTFE_EXACT") == "1"
          and dt_cmd[1].get("DTFE_EXACT_AVERAGE") == "0" and dte_cmd[1].get("DTFE_EXACT_AVERAGE") == "1",
          f"{dt_cmd[1]} {dte_cmd[1]}")
    # a job STARTS at Exact, both estimators (both drivers' and the binaries' default since 2026-11-09, the
    # production setting); a choice saved in a settings dict is kept, an older one's deposit is read as the quality
    dflt = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], estimator="dtfe")
    check("a job starts at Exact (standard DTFE: -e, DTFE_EXACT_AVERAGE=1), PS-DTFE and standard DTFE, Grids, "
          "Pipeline and Custom alike; a saved choice is kept, a settings dict without one gets Exact, an older one's "
          "deposit becomes the quality ('sampled' Fast, anything else Exact)",
          dflt.quality == "exact" and "-e" in dflt.command()[0] and dflt.command()[1]["DTFE_EXACT_AVERAGE"] == "1"
          and rs.QUALITIES == ("exact", "fast") and rs.QUALITY_DEFAULT == "exact"
          and rs.RunSpec().quality == "exact" and rs.CustomSpec().quality == "exact"
          and rs.PipelineSpec().quality == "exact" and rs.CustomSpec(estimator="dtfe").quality == "exact"
          and rs.RunSpec.from_dict({"estimator": "dtfe", "deposit": "sampled"}).quality == "fast"
          and rs.RunSpec.from_dict({"deposit": "sampled"}).quality == "fast"
          and rs.RunSpec.from_dict({"deposit": ""}).quality == "exact"
          and rs.RunSpec.from_dict({"deposit": "exact"}).quality == "exact"
          and rs.RunSpec.from_dict({"estimator": "dtfe"}).quality == "exact"
          and rs.RunSpec.from_dict({"quality": "fast", "deposit": "exact"}).quality == "fast"
          and rs.PipelineSpec.from_dict({"deposit": "sampled"}).quality == "fast"
          and rs.CustomSpec.from_dict(rs.CustomSpec(estimator="dtfe", quality="fast").to_dict()).quality
          == "fast", f"{dflt.quality} {dflt.command()}")
    def _presets():                      # (imported here: the module's own tests import it later)
        import presets
        return presets

    def _script_default(script, var):    # the driver's own default, read from its 'VAR="${VAR:-...}"' line
        out = subprocess.run(["bash", "-c", f'eval "$(grep -E "^{var}=" "$1" | head -1)"; printf %s "${{{var}}}"', "_",
                       str(script)], capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
        return out.stdout
    check("the launcher's defaults are the drivers': the field lists, the caustics, the volume weighting",
          _script_default(rs.PS_SCRIPT, "FIELDS").split() == rs.PS_DEFAULT_FIELDS
          and _script_default(rs.DTFE_SCRIPT, "FIELDS").split() == rs.DTFE_DEFAULT_FIELDS
          and rs.PS_DEFAULT_FIELDS == rs.DTFE_DEFAULT_FIELDS == rs.dtfe_fields(rs.PS_DEFAULT_FIELDS)
          and _presets().BUILTIN_PRESETS["Standard"][1]["fields"] == rs.DEFAULT_FIELDS
          and _script_default(rs.PS_SCRIPT, "PS_CAUSTICS") == "1" and rs.RunSpec().caustics and rs.PipelineSpec().caustics
          and _script_default(rs.PS_SCRIPT, "PS_VOLUME_WEIGHTED") == "1" and rs.RunSpec().mode().volume_weighted
          and rs.PipelineSpec().mode().volume_weighted and rs.RunSpec().grid == rs.PipelineSpec().grid == 512,
          f"{_script_default(rs.PS_SCRIPT, 'FIELDS')!r} {_script_default(rs.DTFE_SCRIPT, 'FIELDS')!r}")
    check("to_dict/from_dict round trip", rs.RunSpec.from_dict(spec.to_dict()) == spec)
    return spec


def _run_checks(tmp):
    """RunSpec.problems(): the errors and warnings of a Grids job."""
    print("checks:")
    lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=128))
    check("a sane small run has no errors", "error" not in lv, msg)
    lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[]))
    check("no snapshots is an error", "error" in lv and "no snapshots" in msg, msg)
    lv, msg = levels(rs.RunSpec(data_root=str(tmp / "missing"), sim="TNG100-3-Dark", snapshots=[4]))
    check("a missing data root is an error", "error" in lv and "data root not found" in msg, msg)
    lv, msg = levels(rs.RunSpec(data_root="", sim="TNG100-3-Dark", snapshots=[4]))
    check("no data root is an error (never the working directory)", "error" in lv and "no data root" in msg, msg)
    lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4],
                                caustic_cusps=True, caustics=False))
    check("cusps without caustics is an error", "error" in lv and "cusps" in msg, msg)
    icloud = tmp.parent / "Mobile Documents" / "scratch"      # iCloud Drive's path component
    icloud.mkdir(parents=True)                               # (not the repo's own location: CI)
    lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4],
                                scratch_dir=str(icloud)))
    check("an iCloud scratch directory is an error", "error" in lv and "iCloud" in msg, msg)
    lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=1024))
    check("1024^3 without scratch warns about memory", "warning" in lv and "scratch" in msg, msg)
    ok = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=1024,
                    scratch_dir=str(tmp.parent))
    check("... and a scratch directory clears the warning",
          not any("scratch" in m for _, m in ok.problems()), " | ".join(m for _, m in ok.problems()))
    lv, msg = levels(rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], output_prefix="a b"))
    check("an output prefix with a space is an error", "error" in lv, msg)


def _progress():
    """parse_progress() and strip_ansi() on the drivers' progress lines."""
    print("progress:")
    p = rs.parse_progress("Processing snapshot 033...\n\t Done:  1% 2%  [partitions 45/125 | 36%]  "
                          "elapsed 8m 40s, ETA ~15m 25s\r")
    check("partition progress and snapshot parsed",
          p.get("snapshot") == 33 and p.get("done") == 45 and p.get("total") == 125
          and p.get("percent") == 36 and p.get("elapsed") == "8m 40s", str(p))
    check("ANSI colour codes stripped", rs.strip_ansi("\x1b[1mPS-DTFE:\x1b[0m ok") == "PS-DTFE: ok")
    check("snapshot completion detected",
          rs.parse_progress("  Snapshot 033 processed successfully\n").get("snapshot_finished") is True)


def _pipeline(tmp, sim):
    """PipelineSpec: its command, the render step, plan only, the stale() rule and its problems."""
    print("pipeline:")
    pipe = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, scratch_dir=str(tmp.parent))
    argv, env = pipe.command()
    check("pipeline: the script, SIMS/NU/AXIS/GRID_SIZE always, defaults otherwise omitted",
          argv == [str(rs.PIPELINE_SCRIPT)] and env["SIMS"] == "TNG100-3-Dark" and env["NU"] == "64"
          and env["AXIS"] == "z" and env["GRID_SIZE"] == "512" and env["PTS_VEL_GRAD"] == "1"
          and "FIELDS" not in env and "OUTPUT_PREFIX" not in env and "FORCE" not in env
          and "DRY_RUN" not in env and env["SCRATCH_DIR"] == str(tmp.parent), str(env))
    st = pipe.steps()
    check("... then plot_pointeval.py pinned to the pipeline's own plane",
          len(st) == 2 and st[1].argv[1].endswith("plot_pointeval.py")
          and st[1].argv[st[1].argv.index("--plane") + 1] == str(sim / "hires_plane_z64.json")
          and st[1].env["MPLBACKEND"] == "Agg", str(st[1].argv if len(st) > 1 else st))
    pipe.plan_only = True
    check("plan only: DRY_RUN=1 and no render step",
          len(pipe.steps()) == 1 and pipe.command()[1]["DRY_RUN"] == "1")
    pipe.plan_only = False
    check("stale(): no slices yet -> every snapshot", pipe.stale("TNG100-3-Dark") == ([4, 99], 2))
    (sim / "hires_plane_z64.bin").touch()
    # a sidecar as make_image_plane.py writes it (an unreadable one counts as 'differs')
    (sim / "hires_plane_z64.json").write_text('{"box": 100.0, "center": 50.0, "planes": 1, "thickness": 2.0, '
                                               '"nu": 64, "nv": 64, "u0": 0.0, "u1": 100.0, "v0": 0.0, "v1": 100.0}')
    for n in (4, 99):
        for ext in ("pts_den", "pts_velGrad"):
            (sim / f"snapdir_{n:03d}" / f"ps_output.{ext}").touch()
        # a slice OF this plane: 64 x 64 float64 (the count is compared)
        (sim / f"snapdir_{n:03d}" / "ps_output.pts_den").write_bytes(b"\0" * 8 * 64 * 64)
    # explicit times: "newer" is STRICT (the script's -nt), and Linux stamps files from a
    # coarse clock, so files made milliseconds apart can share one mtime (failed on Ubuntu)
    now = time.time()
    for f in [sim / "hires_plane_z64.bin", sim / "hires_plane_z64.json"] \
            + [sim / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5" for n in (4, 99)]:
        os.utime(f, (now - 100, now - 100))
    for n in (4, 99):
        for ext in ("pts_den", "pts_velGrad"):
            os.utime(sim / f"snapdir_{n:03d}" / f"ps_output.{ext}", (now, now))
    (sim / "snapdir_099" / "ps_output.pts_velGrad").unlink()
    check("stale(): fresh slices skipped, a missing .pts_velGrad is stale (the script's rule)",
          pipe.stale("TNG100-3-Dark") == ([99], 2), str(pipe.stale("TNG100-3-Dark")))
    pipe.force = True
    check("stale(): FORCE recomputes everything", pipe.stale("TNG100-3-Dark") == ([4, 99], 2))
    pipe.force = False
    # a slice newer than the plane but of ANOTHER plane size (TNG100's 4096^2 slices beside an older
    # hires_plane_z8192) is stale: read as current, the script would skip the compute and every render fail
    den4 = sim / "snapdir_004" / "ps_output.pts_den"
    den4.write_bytes(b"\0" * 8 * 32 * 32)
    os.utime(den4, (now, now))
    check("stale(): a slice of another plane size is stale, though newer than the plane (its point count)",
          pipe.stale("TNG100-3-Dark") == ([4, 99], 2), str(pipe.stale("TNG100-3-Dark")))
    den4.write_bytes(b"\0" * 8 * 64 * 64)
    os.utime(den4, (now, now))
    lv, msg = levels(rs.PipelineSpec(data_root=str(tmp), sims=[]))
    check("pipeline: no simulations is an error", "error" in lv and "no simulations" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], center="half"))
    check("pipeline: a non-numeric centre is an error", "error" in lv and "centre" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=8192))
    check("pipeline: an 8192^2 plane without scratch warns (it did not fit on 64 GB)",
          "warning" in lv and "scratch" in msg, msg)


def _plots(tmp, sim, spec):
    """PlotSpec: the calls per figure set and their problems, with the plot-option suites, and figures(since=)."""
    print("plots:")
    (sim / "snapdir_099" / "ps_output.a_den").touch()
    (sim / "snapdir_004" / "ps_output.a_den").unlink()       # 004: no grids now
    (sim / "pointeval_plane_64_x.json").touch()
    (sim / "pointeval_plane_64_x.bin").write_bytes(b"\0" * 24 * 64 * 64)
    every = rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4],
                        sets=[f.key for f in rs.FIGURE_SETS], method="ps", smooth=1.5,
                        options={"pointeval": {"fields": "density, velDiv", "force": True},
                                 "panels": {"plane": str(sim / "hires_plane_z64.json")},
                                 "halotrack": {"subhalos": "0, 17"},
                                 "core": {"only": "plot_DTFE.py"}})
    by = {}
    for stp in every.steps():
        by.setdefault(Path(stp.argv[1]).name, []).append(stp.argv[2:])
    check("per-snapshot sets: one call per snapshot, sorted, with the common flags",
          by["plot_PS_DTFE.py"] == [["--sim", "TNG100-3-Dark", "--snap", n, "--estimator", "ps", "--smooth", "1.5"]
                                    for n in ("4", "99")] and len(by["plot_cosmic_web.py"]) == 2,
          str(by.get("plot_PS_DTFE.py")))
    check("the series is one call with every snapshot",
          by["plot_DTFE.py"] == [["--sim", "TNG100-3-Dark", "--snaps", "4", "99", "--estimator", "ps",
                                  "--smooth", "1.5"]], str(by["plot_DTFE.py"]))
    check("point-eval maps: space-separated lists (the maps typed with commas too), --force, NO grid smoothing, "
          "the estimator",
          by["plot_pointeval.py"] == [["--sims", "TNG100-3-Dark", "--snaps", "4", "99", "--estimator", "ps", "--maps",
                                       "density", "velDiv", "--force"]], str(by["plot_pointeval.py"]))
    pan = by["plot_pointeval_panels.py"]
    check("panels: per snapshot, its own output root, the plane, an -o under python/figures",
          len(pan) == 2 and pan[0][:2] == ["--root", str(sim / "snapdir_004" / "ps_output")]
          and pan[0][pan[0].index("-o") + 1].startswith(str(rs.FIGURES_ROOT / "pointeval_panels")), str(pan))
    check("halo tracking: from the reference snapshot (--start-snap), explicit IDs replace --n-halos",
          by["plot_halo_tracking.py"][0][-3:] == ["--subhalos", "0", "17"]
          and by["plot_halo_tracking.py"][0][2] == "--start-snap", str(by["plot_halo_tracking.py"]))
    check("core set: analyze.py all, zero-padded --snaps, --only",
          by["analyze.py"] == [["all", "--snaps", "004", "099", "--only", "plot_DTFE.py"]], str(by["analyze.py"]))
    spec_e = rs.PlotSpec.from_dict(dict(spec.to_dict(), sets=["core"], options={"core": {"mode": "export", "only": "plot_DTFE.py"}}))
    argv_e = [st.argv for st in spec_e.steps()]
    check("core set, mode 'export': analyze.py export with the snapshots and no --only",
          [a[a.index("export"):] for a in argv_e if "export" in a] == [["export", "--snaps", "004", "099"]], str(argv_e))
    core_grids_warning()
    tick_helpers()
    run_outputs_not_inputs()
    run_options(tmp)
    audit_claims()
    cube_cap()
    cube_release()
    point_window_linear(tmp)
    plane_reads()
    plot_options(tmp)
    launcher_python_flags(tmp)
    saved_flag_migration(tmp)
    plot_void_sets(tmp)
    plot_void_script_options(tmp)
    fixed_r_size_function(tmp)
    crossbox_smoothing_constant()
    gpu_retry_causes(tmp)
    explore_extras(tmp)
    check("every plot step sets DTFE_SIM and MPLBACKEND=Agg (several scripts call plt.show())",
          all(stp.env["DTFE_SIM"] == "TNG100-3-Dark" and stp.env["MPLBACKEND"] == "Agg"
              for stp in every.steps()))
    lv, msg = levels(every)
    check("plots: snapshot 004 has no grids -> warned", "004: those figures will fail" in msg, msg)
    check("plots: two 64^2 planes -> 'auto' is ambiguous", "same pixel count" in msg, msg)
    lv, msg = levels(rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], sets=["panels"]))
    check("panels without a plane is an error", "error" in lv and "choose the image plane" in msg, msg)
    lv, msg = levels(rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[], sets=["fields"]))
    check("per-snapshot figures without snapshots is an error", "error" in lv and "no snapshots" in msg, msg)
    lv, msg = levels(rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[], sets=["voidpop"]))
    check("... but a per-simulation set needs none", "error" not in lv, msg)
    check("from_dict drops figure sets that no longer exist",
          rs.PlotSpec.from_dict({"sets": ["fields", "gone"]}).sets == ["fields"])
    t0 = time.time()
    time.sleep(0.05)
    (tmp / "figs").mkdir()
    (tmp / "figs" / "new.png").touch()
    check("figures(since=...) lists only what a run wrote",
          [p.name for p in rs.figures([tmp / "figs", ROOT / "python" / "gui"], since=t0)] == ["new.png"])
    check("pipeline plan line parsed (snapshots to compute, simulation)",
          rs.parse_progress("+ env A=1 /x/scripts/run_ps_dtfe.sh -s TNG100-3-Dark -g 512 -m 50 67 99\n"
                            "== TNG100-3-Dark: snapshots 0 4") == {"planned": 3, "sim": "TNG100-3-Dark"})


def _download_merge(tmp):
    """DataSpec: the download and merge steps, their problems, what is on disk, and the download progress lines."""
    print("download + merge:")
    check("the redshift ladder comes from config.py (16 snapshots, z(99) = 0)",
          len(rs.SNAPSHOT_LADDER) == 16 and rs.SNAPSHOT_LADDER.get(99) == 0.0 and rs.SNAPSHOT_LADDER.get(0) == 20.05)
    dsim = tmp / "TNG50" / "TNG50-4-Dark"                  # family folder exists, the run does not
    (tmp / "TNG50").mkdir()
    key_file, saved_key = rs.API_KEY_FILE, os.environ.pop("TNG_API_KEY", None)
    rs.API_KEY_FILE = tmp.parent / "no_key_here"
    try:
        d = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[99, 50], download_groupcats=True,
                        download_trees=True, download_ics=True, merge_groupcats=True, merge_trees=True,
                        convert_ics=True, delete_chunks=True)
        got = [(st.label, st.argv[1:] if st.argv[0] != rs.PYTHON else st.argv[2:]) for st in d.steps()]
        check("downloads (snapshots, groups, trees, ICs) first, then merges, into the family folder",
              [lbl for lbl, _ in got] == ["download snapshots 50 99", "download group catalogues 50 99",
                                          "download merger trees", "download initial conditions",
                                          "merge snapshots 50 99", "merge group catalogues 50 99",
                                          "merge merger trees", "convert initial conditions"]
              and got[0][1] == ["-s", "TNG50-4-Dark", "50", "99"] and got[1][1][:1] == ["-c"]
              and got[4][1] == ["-d", str(dsim), "--snaps", "50", "99", "--delete-chunks"]
              and all(a[-1] == "--delete-chunks" for _, a in got[4:]), str(got))
        check("no -k on any command line (the key stays in its file / env)",
              all("-k" not in st.argv for st in d.steps()))
        lv, msg = levels(d)
        check("no API key is an error naming the key file", "error" in lv and "no_key_here" in msg, msg)
        check("--delete-chunks is warned about", "only copy" in msg, msg)
        os.environ["TNG_API_KEY"] = "x"
        lv, msg = levels(d)
        check("... TNG_API_KEY clears it", "API key" not in msg, msg)
        # 64 x 615 GB: more than any disk here
        lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="TNG50-1-Dark", snapshots=list(range(64))))
        check("a download larger than the free space is an error", "error" in lv and "GB free" in msg, msg)
        lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="TNG50-4", snapshots=[99]))
        check("a full-physics run warns about gas and stars", "gas and stars" in msg, msg)
        lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="Millennium", snapshots=[99]))
        check("an unknown run warns", "not a TNG run" in msg, msg)
        lv, msg = levels(rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[],
                                     download_snapshots=False, merge_snapshots=False))
        check("nothing ticked is an error", "error" in lv and "nothing selected" in msg, msg)
        # on disk: 050 merged (+ chunks), 099 two chunks, group catalogue 099 merged, trees merged
        for n, files in ((50, ["snap_050.0.hdf5", "combined_050.hdf5"]), (99, ["snap_099.0.hdf5", "snap_099.1.hdf5"])):
            (dsim / f"snapdir_{n:03d}").mkdir(parents=True)
            for f in files:
                (dsim / f"snapdir_{n:03d}" / f).touch()
        (dsim / "groups_099").mkdir()
        (dsim / "groups_099" / "combined_fof_subhalo_tab_099.hdf5").touch()
        (dsim / "Merger Trees").mkdir()
        (dsim / "Merger Trees" / "combined_tree_extended.hdf5").touch()
        check("snapshot_state / sim_state read the disk",
              rs.snapshot_state(dsim, 50) == {"chunks": 1, "merged": True, "gc_chunks": 0, "gc_merged": False}
              and rs.snapshot_state(dsim, 99) == {"chunks": 2, "merged": False, "gc_chunks": 0, "gc_merged": True}
              and rs.sim_state(dsim) == {"tree_chunks": 0, "tree_merged": True, "ics_raw": False,
                                         "ics_converted": False})
        labels = [st.label for st in d.steps()]
        check("merged items are left out of every step (re-merging would stale their grids)",
              labels == ["download snapshots 99", "download group catalogues 50", "download initial conditions",
                         "merge snapshots 99", "merge group catalogues 50", "convert initial conditions"],
              str(labels))
        lv, msg = levels(d)
        check("... and named in a warning", "left out: snapshot 050, groups 099, merger trees" in msg, msg)
        n4 = 270 ** 3
        check("size estimate skips merged snapshots and chunks already on disk",
              rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[50, 99, 67]).download_bytes()
              == (n4 * 61, 2 * n4 * 32))
        check("the ladder plus what is on disk", 99 in rs.data_snapshots(dsim) and 8 in rs.data_snapshots(dsim))
        done = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[50], download_snapshots=True)
        lv, msg = levels(done)
        check("everything merged: nothing left to do (an error, so Run stays off)",
              done.steps() == [] and "nothing left to do" in msg, msg)
    finally:
        rs.API_KEY_FILE = key_file
        os.environ.pop("TNG_API_KEY", None)
        if saved_key is not None:
            os.environ["TNG_API_KEY"] = saved_key
    p = rs.parse_progress('2026-09-29 12:00:00 URL:https://x/files/snapshot-99/snap_099.0.hdf5 [1234/1234] '
                          '-> "/a b/snap_099.0.hdf5" [1]\n  [2/4] Reading snap_099.1.hdf5...\n'
                          '  Error downloading snapshot-5\n  Particles merged: [8]  (MISMATCH vs header total [9])')
    check("wget files, merge chunks and problem lines parsed",
          p.get("downloaded") == [("/a b/snap_099.0.hdf5", 1234)] and p.get("chunk") == (2, 4)
          and p.get("problem") == 2, str(p))
    os.environ["TNG_API_KEY"] = "k"
    try:
        dl_steps = [st for st in rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[67]).steps()
                    if "download" in st.label]
    finally:
        os.environ.pop("TNG_API_KEY", None)
    check("the download step carries PY (the script checks the chunks on disk with the merge's python)",
          dl_steps and all(st.env.get("PY") == rs.PYTHON for st in dl_steps), str([st.env for st in dl_steps]))
    os.environ["TNG_API_KEY"] = "k"
    try:
        both = rs.DataSpec(data_root=str(tmp), sim="TNG50-2-Dark", snapshots=[67], download_groupcats=True,
                           merge_groupcats=True, download_trees=True, merge_trees=True).steps()     # nothing on disk
        only = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[67], download_snapshots=False).steps()
    finally:
        os.environ.pop("TNG_API_KEY", None)
    after = {st.label: st.after for st in both}
    check("every merge step is chained to its own download step (skipped by the launcher when that failed)",
          after.get("merge snapshots 67") == "download snapshots 67"
          and after.get("merge group catalogues 67") == "download group catalogues 67"
          and after.get("merge merger trees") == "download merger trees", str(after))
    check("a merge without a download step is chained to nothing", [st.after for st in only] == [""], str(only))
    # a stopped merge's leftover: named, and its bytes counted as free for the merge that removes it
    sp = rs.sim_dir("TNG50-4-Dark", str(tmp))
    stale = sp / "snapdir_067" / "combined_067.hdf5.partial"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x" * 4096)
    try:
        d = rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[67], download_snapshots=False)
        lv, msg = levels(d)
        check("a stale .partial of a snapshot about to be merged is listed and named in a warning",
              [p.name for p in d.stale_partials()] == ["combined_067.hdf5.partial"] and "a stopped merge left" in msg
              and "combined_067.hdf5.partial" in msg, msg)
    finally:
        stale.unlink()


def _memory_check(tmp):
    """The auto-tune report lines and the memory-check steps of Grids and Pipeline jobs, the adaptive plan."""
    print("memory check:")
    rep = rs.parse_reports("  AUTO-TUNE-REPORT snap=099 partition=5 mc=2 predicted_gb=33.75 budget_gb=34.88 "
                           "fixed_gb=20.75 pts_gb=15.47 streams_est=3.633 streams_budget=6.266 over_budget=0\n"
                           "Processing snapshot 004...\n  AUTO-TUNE-REPORT snap=004 partition=6 mc=1 "
                           "predicted_gb=95.0 budget_gb=41.2 over_budget=1")
    check("report lines parsed per snapshot (ints for split/concurrency/verdict, floats for GB)",
          rep == {99: {"partition": 5, "mc": 2, "predicted_gb": 33.75, "budget_gb": 34.88, "fixed_gb": 20.75,
                       "pts_gb": 15.47, "streams_est": 3.633, "streams_budget": 6.266, "over_budget": 0},
                  4: {"partition": 6, "mc": 1, "predicted_gb": 95.0, "budget_gb": 41.2, "over_budget": 1}},
          str(rep))
    grids = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4], scratch_dir=str(tmp.parent))
    st = grids.check_step()
    check("a grids check is the SAME run_ps_dtfe.sh call plus AUTO_TUNE_REPORT=1",
          st.argv == grids.command()[0] and st.env == dict(grids.command()[1], AUTO_TUNE_REPORT="1"))
    # the standard binary's tuner reports too, through run_dtfe.sh's own AUTO_TUNE_REPORT
    sgrids = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4], estimator="dtfe")
    sst = sgrids.check_step()
    check("... and a standard-DTFE grids check is the same run_dtfe.sh call plus AUTO_TUNE_REPORT=1",
          sst.argv == sgrids.command()[0] and sst.argv[0].endswith("run_dtfe.sh")
          and sst.env == dict(sgrids.command()[1], AUTO_TUNE_REPORT="1"), str(sst.argv))
    check("the plan as the launcher says it: PS-DTFE runs mc partitions at a time, standard DTFE its partitions "
          "one after another with mc threads each",
          rs.report_split_text({"partition": 4, "mc": 5}, "ps") == "4³ partitions, 5 at a time"
          and rs.report_split_text({"partition": 1, "mc": 10}, "ps") == "one tessellation"
          and rs.report_split_text({"partition": 4, "mc": 4}, "dtfe") == "4³ partitions one after another, 4 threads each"
          and rs.report_split_text({"partition": 1, "mc": 1}, "dtfe", 2) == "one domain, 1 thread"
          and rs.report_split_text({"partition": 3, "mc": 9}, "dtfe", 2) == "3² partitions one after another, 9 threads each")
    check("the memory key ignores the snapshot choice but not the settings",
          rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4]).memory_key()
          == rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99]).memory_key()
          != rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[4], grid=256).memory_key())
    pm = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, grid=128, scratch_dir=str(tmp.parent))
    with fake_build("cpu"):
        have = pm.check_step("TNG100-3-Dark", [99, 4], 64)             # hires_plane_z64.bin exists
        need = pm.check_step("TNG100-3-Dark", [99], 32)                # hires_plane_z32 does not
        cpu_mode = pm.mode()
    with fake_build("gpu"):
        gpu_have = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64,
                                   grid=128).check_step("TNG100-3-Dark", [99, 4], 64)
    check("a pipeline check passes the pipeline's settings (Exact, the default; on a CPU build no -m, on a GPU build "
          "-m: the quality's device), the plane file when it exists ...",
          have.argv[1:] == ["-s", "TNG100-3-Dark", "-g", "128", "4", "99"] and not cpu_mode.gpu
          and gpu_have.argv[1:] == ["-s", "TNG100-3-Dark", "-g", "128", "-m", "4", "99"]
          and have.env["SAMPLE_POINTS"].endswith("hires_plane_z64.bin") and have.env["AUTO_TUNE_REPORT"] == "1"
          and have.env["PS_VOLUME_WEIGHTED"] == "1" and have.env["SCRATCH_DIR"] == str(tmp.parent), str(have.env))
    check("... and only its point count when it does not exist yet",
          need.env.get("DTFE_AUTO_PTS_N") == str(32 * 32) and "SAMPLE_POINTS" not in need.env
          and need.env.get("DTFE_AUTO_PTS_VEL_GRAD") == "1", str(need.env))
    check("adaptive sizes halve down to the minimum",
          rs.PipelineSpec(nu=8192).adaptive_sizes() == [8192, 4096, 2048, 1024])
    pm.adaptive = True
    lv, msg = levels(pm)
    check("adaptive without a check is an error", "error" in lv and "Check memory" in msg, msg)
    pm.plan, pm.plan_key = {"TNG100-3-Dark": {"004": 64, "099": 32}}, pm.memory_key()
    got = [(st.label, st.env.get("NU"), st.env.get("SNAPS")) for st in pm.steps()]
    check("a checked plan runs the pipeline once per plane size, on its own snapshots, rendering its plane",
          [(g[1], g[2]) for g in got if g[1]] == [("64", "4"), ("32", "99")]
          and [st.argv[st.argv.index("--plane") + 1].endswith(f"hires_plane_z{nu}.json")
               for st, nu in zip(pm.steps()[1::2], (64, 32))] == [True, True], str(got))
    # run_ps_pipeline.sh exits 1 when it computed nothing, and the render after it must not run then
    check("... each render waits for its own pipeline call (after = that call's label; the calls wait for nothing)",
          [st.after for st in pm.steps()[1::2]] == [st.label for st in pm.steps()[0::2]]
          and all(st.after == "" for st in pm.steps()[0::2]), str([(st.label, st.after) for st in pm.steps()]))
    pm.grid = 256
    lv, msg = levels(pm)
    check("a plan checked for other settings is refused", "error" in lv and "changed since" in msg, msg)


def _queue(tmp):
    """QueuedJob: the summaries, a saved queue's round trip and the states it restores."""
    print("queue:")
    jobs = [rs.QueuedJob.of("grids", rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99, 4])),
            rs.QueuedJob.of("pipeline", rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64)),
            rs.QueuedJob.of("plots", rs.PlotSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99],
                                                 sets=["fields", "voidpop"])),
            rs.QueuedJob.of("data", rs.DataSpec(data_root=str(tmp), sim="TNG50-4-Dark", snapshots=[99]))]
    check("each job kind gets a one-line summary (the exact deposit, the default, named)",
          [j.title for j in jobs] == ["Grids · TNG100-3-Dark · 004 099 · PS-DTFE 512³, Exact",
                                      "Pipeline · TNG100-3-Dark · plane 64² · grid 512³, Exact · + figures",
                                      "Plots · TNG100-3-Dark · 099 · Field slice maps, Void population",
                                      "Data · TNG50-4-Dark · 099 · download + merge"],
          str([j.title for j in jobs]))
    check("a queued job keeps its settings, not its commands (rebuilt exactly)",
          all(rs.QueuedJob.from_dict(j.to_dict()).build().to_dict() == j.spec for j in jobs)
          and all("steps" not in j.to_dict() for j in jobs))
    jobs[0].state, jobs[1].state = "done", "running"
    back = [rs.QueuedJob.from_dict(j.to_dict()) for j in jobs]
    check("restored: a job left running is 'stopped', the first waiting one is next",
          [j.state for j in back] == ["done", "stopped", "waiting", "waiting"] and rs.next_waiting(back) == 2)
    try:
        rs.QueuedJob.from_dict({"kind": "nonsense", "spec": {}})
        check("an unknown job kind is refused", False)
    except ValueError:
        check("an unknown job kind is refused", True)
    # a queue saved before the two quality modes: its jobs' deposit becomes the quality, the settings the modes
    # replaced are dropped (in the saved spec too, so the queue is saved back clean), without an error
    old_grids = dict(jobs[0].spec, deposit="sampled", nsub=1, gpu=True, face_gradient=False, volume_weighted=False,
                     linear_deposit=True, ordered_merge=False, parallel_triangulation=True, partition=2,
                     max_concurrent=3, mem_budget_gb=8.0, pts_conventions="legacy", precision="double",
                     vertex_mass=False)
    old_grids.pop("quality")
    old_pipe = dict(jobs[1].spec, deposit="exact", nsub=2, gpu=False, precision="double", supersample=2)
    old_pipe.pop("quality")
    old_custom = dict(rs.CustomSpec(input_file=str(tmp / "x.hdf5")).to_dict(), deposit="sampled", gpu=True,
                      stream_density="geometric", precision="double", vertex_mass=True, partition=4)
    old_custom.pop("quality")
    qd = [rs.QueuedJob.from_dict({"kind": k, "spec": d, "title": "older", "state": "waiting"})
          for k, d in (("grids", old_grids), ("pipeline", old_pipe), ("custom", old_custom))]
    built = [q.build() for q in qd]
    gone = set(rs.REMOVED_SETTINGS)
    check("a queue saved before the two modes loads: the deposit becomes the quality ('sampled' Fast, 'exact' "
          "Exact), every removed setting is dropped from the saved spec (a Custom job keeps its precision and initial "
          "positions, a Pipeline job its pixel sub-samples), the jobs build, and each title is made again from the "
          "migrated settings (the old one named the removed ones)",
          [b.quality for b in built] == ["fast", "exact", "fast"]
          and not (gone & set(qd[0].spec)) and not (gone & set(qd[1].spec))
          and not (gone - {"precision", "vertex_mass"}) & set(qd[2].spec)
          and built[2].precision == "double" and built[2].vertex_mass is True and built[1].supersample == 2
          and built[0] == rs.RunSpec(**dict(jobs[0].spec, quality="fast"))
          and [q.title for q in qd] == [rs.job_summary(k, b) for k, b in zip(("grids", "pipeline", "custom"), built)],
          str([q.spec for q in qd]))
    check("grids, pipeline and custom-snapshot jobs wait for other DTFE runs",
          set(rs.HEAVY_KINDS) == {"grids", "pipeline", "custom"})


def estimator_per_snapshot(tmp: Path):
    """The void catalogue build resolves estimator 'auto' per snapshot (never 'ps' for a snapdir with standard-DTFE
    grids only), and the point-evaluated sets read the estimator's own slice (not ps_output, so a standard-DTFE slice
    beside PS-DTFE grids can be rendered)."""
    import json
    print("estimator per snapshot:")
    root = tmp.parent / "est_root"
    sp = root / "BOX-1-Dark"
    for n in (4, 99):
        (sp / f"snapdir_{n:03d}").mkdir(parents=True, exist_ok=True)
    (sp / "snapdir_004" / "output.a_den").touch()                       # standard DTFE only
    (sp / "snapdir_099" / "ps_output.a_den").touch()                    # both
    (sp / "snapdir_099" / "output.a_den").touch()

    def methods(**kw):
        spec = rs.PlotSpec(data_root=str(root), sim="BOX-1-Dark", snapshots=[99, 4], sets=["voidcat"], **kw)
        return [st.argv[st.argv.index("--estimator") + 1] for st in spec.steps()]
    auto = rs.PlotSpec(data_root=str(root), sim="BOX-1-Dark", snapshots=[4, 99], sets=["voidcat"])
    lv, msg = levels(auto)
    check("void catalogue build, estimator auto: standard DTFE where only output.* grids are, PS-DTFE where "
          "ps_output.* are; the check agrees (no missing grids)",
          methods() == ["dtfe", "ps"] and "no " not in msg and "error" not in lv, f"{methods()} {msg}")
    check("... a chosen estimator, or a grid prefix (PS-DTFE), wins",
          methods(method="dtfe") == ["dtfe", "dtfe"] and methods(prefix="ps_caustic") == ["ps", "ps"]
          and methods(method="ps") == ["ps", "ps"], f"{methods(method='dtfe')} {methods(prefix='ps_caustic')}")

    side = sp / "hires_plane_z16.json"
    side.write_text(json.dumps({"box": 100.0, "nu": 16, "nv": 16, "planes": 1}))
    sd = sp / "snapdir_099"
    (sd / "output.pts_den").write_bytes(b"\0" * 8 * 16 * 16)            # a standard-DTFE slice beside PS-DTFE grids

    def pts(**kw):
        spec = rs.PlotSpec(data_root=str(root), sim="BOX-1-Dark", snapshots=[99], sets=["pointeval", "panels"],
                           options={"panels": {"plane": str(side)}}, **kw)
        st = {Path(x.argv[1]).name: x.argv for x in spec.steps()}
        pe, pa = st["plot_pointeval.py"], st["plot_pointeval_panels.py"]
        return (pe[pe.index("--estimator") + 1] if "--estimator" in pe else "auto", pa[pa.index("--root") + 1],
                any("no point-evaluated slice" in m for _, m in spec.problems()))
    check("point-evaluated sets, estimator DTFE: plot_pointeval gets --estimator dtfe, the panels read output.pts_*, "
          "and the check finds the slice",
          pts(method="dtfe") == ("dtfe", str(sd / "output"), False), str(pts(method="dtfe")))
    check("... auto: the slice that is there (output.pts_* here, though PS-DTFE grids are beside it); PS-DTFE: "
          "ps_output.pts_*, and its absence is warned about",
          pts() == ("auto", str(sd / "output"), False) and pts(method="ps") == ("ps", str(sd / "ps_output"), True),
          f"{pts()} {pts(method='ps')}")
    (sd / "ps_output.pts_den").write_bytes(b"\0" * 8 * 16 * 16)
    check("... auto with both slices: PS-DTFE first, as before",
          pts() == ("auto", str(sd / "ps_output"), False), str(pts()))
    # standard DTFE's grids are 'output.*' exactly; 'output*.a_den' would match the plane run's one-field
    # grid 'output_plane.a_den' (a plane run's one-field grid) and any other 'output_<variant>' beside them
    pl = sp / "snapdir_050"
    pl.mkdir()
    (pl / "output_plane.a_den").touch()
    (pl / "output_mc.a_den").touch()
    before = (rs.has_outputs(pl, "", "dtfe"), rs.has_grids(pl, "dtfe", ""), rs.has_grids(pl, "auto", ""))
    (pl / "output.a_den").touch()
    check("standard DTFE's grids are 'output.*' only: a snapdir with 'output_plane.a_den' and 'output_mc.a_den' has "
          "none (Grids marks, figure checks), one with 'output.a_den' has them",
          before == (False, False, False) and rs.has_outputs(pl, "", "dtfe") and rs.has_grids(pl, "dtfe", ""),
          str(before))


def gpu_branches(tmp: Path, snap: Path):
    """The GPU advice and recommendation for a GPU build AND a CPU-only one, whatever this host has: rs._objdir
    (where gpu_built, gpu_backend and tbb_built read the Makefile's stamps) points at a fake object directory with
    or without '.gpu_mode_off'."""
    real = rs._objdir
    fake = Path(tempfile.mkdtemp(prefix="gui_objdir_"))
    try:
        rs._objdir = lambda estimator, precision="single", dim=3: fake / (estimator + precision + str(dim))
        for built in (True, False):
            for d in (rs._objdir(e, p) for e in ("ps", "dtfe") for p in ("single", "double")):
                d.mkdir(exist_ok=True)
                (d / ".gpu_mode_off").unlink(missing_ok=True)
                (d / ".gpu_mode_metal").unlink(missing_ok=True)
                (d / (".gpu_mode_metal" if built else ".gpu_mode_off")).touch()
            print(f"  (faked build: {'GPU (Metal)' if built else 'CPU only'})")
            check(f"the faked stamp is what the build checks read ({built})",
                  rs.gpu_built("ps") is built and rs.gpu_built("dtfe") is built
                  and rs.gpu_backend("ps") == ("Metal" if built else ""))
            # a job's device is its quality's: the drivers' automatic device on the build (the run modes' device
            # table, D21), the GPU but for standard DTFE's Fast in performance and PS-DTFE's Fast on a small file (the
            # header's particle count: a value read from the data); a saved GPU tick is dropped
            small = tmp / "small32.hdf5"
            if not small.is_file():
                import h5py
                with h5py.File(small, "w") as h:
                    h.create_group("Header").attrs["NumPart_Total"] = [0, 32 ** 3, 0, 0, 0, 0]
            dev = lambda spec: spec.mode().gpu                              # noqa: E731
            check(f"a job's device is its quality's, the drivers' automatic one ({built}): PS-DTFE's Exact and Fast "
                  "on the GPU in every machine use (Grids, Pipeline, Custom; a double Custom job as well), standard "
                  "DTFE's Exact too, its Fast in gentle and balanced but not in performance; a Custom PS-DTFE Fast "
                  "run of 32³ particles on the CPU (its Exact on the GPU), a 2D one never; a saved GPU tick is "
                  "dropped either way",
                  all(dev(rs.RunSpec(run_mode=m)) is built and dev(rs.PipelineSpec(run_mode=m)) is built
                      and dev(rs.RunSpec(quality="fast", run_mode=m)) is built
                      and dev(rs.PipelineSpec(quality="fast", run_mode=m)) is built
                      and dev(rs.CustomSpec(run_mode=m)) is built
                      and dev(rs.CustomSpec(quality="fast", run_mode=m)) is built
                      and dev(rs.RunSpec(estimator="dtfe", run_mode=m)) is built
                      and dev(rs.RunSpec(estimator="dtfe", quality="fast", run_mode=m)) is (built and m != "performance")
                      and dev(rs.CustomSpec(estimator="dtfe", quality="fast", run_mode=m)) is (built
                                                                                             and m != "performance")
                      for m in rs.RUN_MODES)
                  and dev(rs.CustomSpec(precision="double", quality="fast")) is built
                  and dev(rs.CustomSpec(input_file=str(small), quality="fast")) is False
                  and dev(rs.CustomSpec(input_file=str(small))) is built
                  and dev(rs.CustomSpec(dim=2)) is False and dev(rs.CustomSpec(dim=2, estimator="dtfe")) is False
                  and dev(rs.RunSpec.from_dict({"gpu": not built})) is built
                  and dev(rs.PipelineSpec.from_dict({"gpu": not built})) is built
                  and dev(rs.CustomSpec.from_dict({"gpu": not built})) is built,
                  f"{dev(rs.RunSpec())} {dev(rs.PipelineSpec())} "
                  f"{dev(rs.RunSpec(estimator='dtfe', quality='fast', run_mode='performance'))}")
            _device_checks(tmp, snap)
    finally:
        rs._objdir = real
        shutil.rmtree(fake, ignore_errors=True)


def _device_checks(tmp: Path, snap: Path):
    """The quality's device (recommended_gpu, the drivers' device table) and the Checks list without GPU advice (the
    device is the table's, so no line compares a choice with it)."""
    ps_gpu, dt_gpu = rs.gpu_built("ps"), rs.gpu_built("dtfe")
    modes = rs.RUN_MODES
    check("the GPU advice is gone with the GPU tick (no helper left to warn about a device nobody sets)",
          not hasattr(rs, "gpu_advice"))
    rec = rs.recommended_gpu
    check("recommended GPU (the drivers' device table): PS-DTFE's exact deposit the GPU in every machine use, its "
          "sampled one from ~1M particles, standard DTFE the GPU but for its sampled average in performance; "
          "balanced when not given; all only when built",
          all(rec("exact", ["density_a"], n, run_mode=m) == ps_gpu for n in (1000, 10 ** 8, None) for m in modes)
          and rec("sampled", ["density_a"], 32 ** 3) is False
          and rec("sampled", ["density_a"], 19_000_000) == ps_gpu and rec("sampled", ["density_a"], None) == ps_gpu
          and all(rec(d, ["density_a"], 1e9, "dtfe", run_mode=m) == (dt_gpu and not (d == "sampled"
                                                                                        and m == "performance"))
                  for d in ("exact", "sampled") for m in modes)
          and rec("sampled", ["density_a"], 1e9, "dtfe") == dt_gpu)
    gpu_said = lambda spec: [m for level, m in spec.problems() if level == "warning" and "GPU" in m  # noqa: E731
                             and "METAL" not in m and "cusps" not in m]
    jobs = ([rs.RunSpec(sim="TNG50-4", snapshots=[99], estimator=e, quality=q, grid=512, run_mode=m,
                        data_root=str(tmp)) for e in ("ps", "dtfe") for q in rs.QUALITIES for m in modes]
            + [rs.CustomSpec(input_file=str(snap), estimator=e, quality=q, grid=512, run_mode=m)
               for e in ("ps", "dtfe") for q in rs.QUALITIES for m in modes])
    check("a Grids or Custom job of either estimator and quality gets no GPU warning in any machine use (the "
          "'untick / tick the GPU' warnings had a tick to talk about; the device is the quality's now)",
          not any(gpu_said(j) or any("faster" in m for _, m in j.problems()) for j in jobs),
          str([gpu_said(j) for j in jobs if gpu_said(j)][:3]))
    dts = rs.RunSpec(sim="TNG50-4", snapshots=[99], estimator="dtfe", quality="fast", grid=512,
                     run_mode="performance", data_root=str(tmp))
    check("... standard DTFE's Fast in performance runs on the CPU (faster there), in balanced on the GPU when built",
          dts.mode().gpu is False and "-m" not in dts.command()[0]
          and rs.RunSpec(estimator="dtfe", quality="fast").mode().gpu is dt_gpu, str(dts.command()[0]))
    dex = rs.RunSpec(sim="TNG50-4", snapshots=[99], estimator="dtfe", grid=256, data_root=str(tmp))
    cex = rs.CustomSpec(input_file=str(snap), estimator="dtfe", grid=256)
    csa = rs.CustomSpec(input_file=str(snap), estimator="dtfe", quality="fast").command()
    check("standard DTFE at Exact: the custom command passes --exact (no older name, no --sampled), at Fast "
          "--sampled (no count: the binary's own), and neither spec warns",
          "--exact" in cex.command() and "--sampled" not in cex.command()
          and not {"--exact-average", "--ps-exact-deposit"} & set(cex.command())
          and "--sampled" in csa and "--exact" not in csa and "--subsamples" not in csa
          and not any("faster" in m for _, m in dex.problems() + cex.problems()),
          str(cex.problems()))
    # the Custom tab's standard run starts at the exact cell average and pads
    # with 15 mean spacings (dtfelib.estimator.STANDARD_PADDING, run_dtfe.sh's and the standard server's value) instead
    # of the binary's 5, which leaves wrong values within a few spacings of a box face; PS-DTFE keeps its own padding
    cd0, cd2 = (rs.CustomSpec(input_file=str(snap), estimator="dtfe").command(),
                rs.CustomSpec(input_file=str(snap), estimator="dtfe", dim=2, quality="fast").command())
    cps = rs.CustomSpec(input_file=str(snap), estimator="ps").command()
    cpss = rs.CustomSpec(input_file=str(snap), estimator="ps", quality="fast").command()
    check("the Custom tab's standard run: --exact by default and '--padding 15' (3D and 2D, either quality); "
          "a PS-DTFE run passes no --padding, --exact by default (no count), '--sampled --subsamples 1' at Fast",
          "--exact" in cd0 and STANDARD_PADDING == 15.0
          and all(c.count("--padding") == 1 and c[c.index("--padding") + 1] == "15" for c in (cd0, cd2))
          and "--padding" not in cps and "--exact" not in cd2 and "--sampled" in cd2
          and "--exact" in cps and "--sampled" not in cps and "--subsamples" not in cps
          and "--exact" not in cpss and "--sampled" in cpss
          and cpss[cpss.index("--sampled") + 1:cpss.index("--sampled") + 3] == ["--subsamples", "1"],
          f"{cd0} {cd2} {cps} {cpss}")


def run_modes(tmp: Path):
    """The machine use (the run modes, D22-D26): every run job's field, its environment and command, the checks, the
    SLURM export, the queue line, the memory key, the run records and the time estimates, the presets; then the
    drivers' device table."""
    import textwrap
    import presets as P
    import results as R
    print("launcher: the machine use (run modes):")
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99])
    snap = tmp / "TNG100" / "TNG100-3-Dark" / "snapdir_099" / "combined_099.hdf5"
    pipe = dict(data_root=str(tmp), sims=["TNG100-3-Dark"])
    jobs = (rs.RunSpec(**base), rs.PipelineSpec(**pipe), rs.CustomSpec(input_file=str(snap)))
    check("every run job holds the machine use, balanced by default (the drivers' default); it survives a round trip "
          "and a queued job, and a job saved without it loads as balanced",
          rs.RUN_MODES == ("gentle", "balanced", "performance") and all(j.run_mode == "balanced" for j in jobs)
          and all(type(j).from_dict(dict(j.to_dict(), run_mode="gentle")).run_mode == "gentle" for j in jobs)
          and all(type(j).from_dict({k: v for k, v in j.to_dict().items() if k != "run_mode"}).run_mode == "balanced"
                  for j in jobs)
          and rs.QueuedJob.of("grids", rs.RunSpec(run_mode="performance", **base)).build().run_mode == "performance")
    lines = lambda t: sum(max(1, len(textwrap.wrap(ln, 90))) for ln in t.split("\n"))     # noqa: E731
    check("each machine use has a menu text and an item tip of at most two lines, neither with '--'",
          set(rs.RUN_MODE_TEXT) == set(rs.RUN_MODES)
          and all(label and lines(tip) <= 2 and "--" not in label + tip for label, tip in rs.RUN_MODE_TEXT.values()),
          str({k: lines(t) for k, (_, t) in rs.RUN_MODE_TEXT.items()}))
    gentle = rs.RunSpec(run_mode="gentle", **base)
    ps_env, dt_env = gentle.command()[1], rs.RunSpec(estimator="dtfe", run_mode="performance", **base).command()[1]
    pj = rs.PipelineSpec(run_mode="gentle", **pipe)
    pe = pj.command()[1]
    probes = (gentle.check_step().env, rs.RunSpec(estimator="dtfe", run_mode="gentle", **base).check_step().env,
              pj.check_step("TNG100-3-Dark", [99], 64).env)
    check("DTFE_RUN_MODE in every job's environment (Grids with either estimator, Pipeline) and in their memory checks; "
          "GPU_SHARE pinned to auto for run_ps_dtfe.sh (Grids and Pipeline: the machine use's share), not for "
          "run_dtfe.sh; the command shown names it",
          ps_env["DTFE_RUN_MODE"] == "gentle" and dt_env["DTFE_RUN_MODE"] == "performance"
          and pe["DTFE_RUN_MODE"] == "gentle" and all(e.get("DTFE_RUN_MODE") == "gentle" for e in probes)
          and ps_env["GPU_SHARE"] == pe["GPU_SHARE"] == probes[2]["GPU_SHARE"] == "auto" and "GPU_SHARE" not in dt_env
          and " DTFE_RUN_MODE=gentle " in f" {gentle.shell_line()} ", f"{ps_env} {dt_env}")

    def after(a, flag):
        return a[a.index(flag) + 1] if flag in a else None
    cps = rs.CustomSpec(input_file=str(snap), run_mode="gentle")
    with fake_build("gpu"):
        cps_cmd, cps_check = cps.command(), cps.check_step().argv
        no_share = [rs.CustomSpec(input_file=str(snap), quality="fast").command(),
                    rs.CustomSpec(input_file=str(snap), estimator="dtfe").command(),
                    rs.CustomSpec(input_file=str(snap), dim=2).command()]
        share_only = rs.CustomSpec(input_file=str(snap), extra_args="--gpu-share 0.5")
        rep2 = [m for lv, m in share_only.problems() if lv == "error" and "repeat" in m]
        share_warn = [m for lv, m in share_only.problems() if lv == "warning" and "--gpu-share without --gpu" in m]
        share_cmd = share_only.command()
    with fake_build("cpu"):
        no_share.append(rs.CustomSpec(input_file=str(snap)).command())
    check("the Custom command always names the machine use ('--run-mode', once), and '--gpu-share auto' only for "
          "PS-DTFE's Exact on the GPU (not Fast, not on a CPU build, not standard DTFE, not 2D); its memory check too",
          after(cps_cmd, "--run-mode") == "gentle" and cps_cmd.count("--run-mode") == 1
          and after(cps_cmd, "--gpu-share") == "auto" and after(cps_check, "--run-mode") == "gentle"
          and after(rs.CustomSpec(input_file=str(snap)).command(), "--run-mode") == "balanced"
          and after(rs.CustomSpec(input_file=str(snap), estimator="dtfe", run_mode="performance").command(),
                    "--run-mode") == "performance"
          and not any("--gpu-share" in c for c in no_share), str(cps_cmd))
    rep = [m for lv, m in rs.CustomSpec(input_file=str(snap), extra_args="--run-mode=performance").problems()
           if lv == "error" and "repeat" in m]
    # (the GPU share among the extra options names the device: the tab then adds neither '--gpu' nor its own share,
    # so the share is theirs, and without '--gpu' the run is on the CPU, which the checks say; the launcher's own
    # rule for the device, design 2.2 item 2, which replaced the 'repeat' error a share there gave)
    check("... the extra options cannot give the machine use again (one setting, the tab's); a GPU share they give "
          "is theirs (no 'repeat' error, the tab adds no '--gpu' nor share of its own), and given without '--gpu' "
          "the checks warn that the run is on the CPU",
          len(rep) == 1 and "--run-mode" in rep[0] and rep2 == [] and len(share_warn) == 1
          and "--gpu" not in share_cmd and share_cmd.count("--gpu-share") == 1 and after(share_cmd, "--gpu-share") == "0.5",
          f"{rep} {rep2} {share_warn} {share_cmd[-6:]}")
    was = os.environ.pop("OMP_NUM_THREADS", None)
    try:
        quiet = [m for j in jobs for _, m in j.problems() if "OMP_NUM_THREADS" in m]
        os.environ["OMP_NUM_THREADS"] = "3"
        said = [(lv, m) for j in jobs for lv, m in j.problems() if "OMP_NUM_THREADS" in m]
    finally:
        os.environ.pop("OMP_NUM_THREADS", None)
        if was is not None:
            os.environ["OMP_NUM_THREADS"] = was
    bad = (rs.RunSpec(run_mode="fast", **base), rs.PipelineSpec(run_mode="fast", **pipe),
           rs.CustomSpec(input_file=str(snap), run_mode="fast"))
    check("an unknown machine use is an error on every run tab; an OMP_NUM_THREADS the launcher inherited is said on "
          "each (it keeps its thread count in every mode), nothing without one",
          all(("error", "Machine use: unknown value fast") in j.problems() for j in bad)
          and not any(lv == "error" and "Machine use" in m for j in jobs for lv, m in j.problems())
          and not quiet and said == [("info", "OMP_NUM_THREADS=3 in the launcher's environment keeps the thread count "
                                              "whatever the machine use")] * 3, str(said))
    steps = gentle.steps() + rs.CustomSpec(input_file=str(snap), run_mode="gentle").steps()
    cluster = rs.export_script(steps, "t", slurm={"job_name": "x"})
    here = rs.export_script(steps, "t")
    check("a SLURM export runs in performance: each step's DTFE_RUN_MODE and a Custom step's '--run-mode' value, with "
          "one comment line saying so; a plain export and the job's own steps keep the launcher's choice",
          "DTFE_RUN_MODE=performance" in cluster and "--run-mode performance" in cluster
          and "DTFE_RUN_MODE=gentle" not in cluster and "--run-mode gentle" not in cluster
          and cluster.splitlines().count("# machine use: performance on the cluster (the launcher's choice is for this "
                                         "machine)") == 1
          and "DTFE_RUN_MODE=gentle" in here and "--run-mode gentle" in here and "machine use" not in here
          and steps[0].env["DTFE_RUN_MODE"] == "gentle" and after(steps[-1].argv, "--run-mode") == "gentle", cluster)
    check("the queue line names the machine use when it is not balanced (the default)",
          rs.job_summary("grids", rs.RunSpec(**base)).endswith("³, Exact")
          and rs.job_summary("grids", gentle).endswith("³, Exact, gentle")
          and ", performance · " in rs.job_summary("pipeline", rs.PipelineSpec(run_mode="performance", **pipe))
          and ", balanced" not in rs.job_summary("pipeline", rs.PipelineSpec(**pipe))
          and ", performance -> " in rs.job_summary("custom", rs.CustomSpec(input_file=str(snap), run_mode="performance")),
          rs.job_summary("grids", gentle))
    check("the machine use is part of every run job's memory-check key (its thread budget changes the tuner's plan)",
          all(j.memory_key() != type(j).from_dict(dict(j.to_dict(), run_mode="gentle")).memory_key() for j in jobs))

    def rec(cmd, sec):
        r = R.RunRecord(path=Path("/x/r.runlog"), sim="S", prefix="ps_output", binary="PS-DTFE")
        r.ok, r.wall_seconds, r.command = True, sec, f"/x/PS-DTFE in.hdf5 out --grid 64{cmd}"
        return r
    recs = [rec("", 100), rec(" --run-mode gentle", 300), rec(" --run-mode=balanced", 200), rec(" --run-mode performance", 120)]
    kw = dict(sim="S", grid=64, estimator="ps", gpu=False, sliced=False)
    check("run records: the machine use from the logged command ('--run-mode MODE' or '=MODE'; none: performance, the "
          "binaries' default); a time estimate is per machine use (performance when not given); 'Load settings' "
          "leaves it out",
          [r.run_mode for r in recs] == ["performance", "gentle", "balanced", "performance"]
          and R.time_estimate(recs, **kw) == (110, 2) and R.time_estimate(recs, run_mode="gentle", **kw) == (300, 1)
          and R.time_estimate(recs, run_mode="balanced", **kw) == (200, 1) and "run_mode" not in recs[1].settings(),
          f"{[r.run_mode for r in recs]} {R.time_estimate(recs, **kw)}")
    # ... and per quality: a Fast run must not predict an Exact one (the run tabs' and the queue's estimates pass the
    # job's, mode().deposit; the log's own word, RunRecord.exact: the header line, else '--exact' in the command)
    qrecs = [rec(" --exact", 400), rec(" --exact", 500), rec(" --sampled", 90), rec("", 110)]
    check("a time estimate is per quality when one is given: Exact runs predict Exact, Fast runs Fast; without "
          "one, every run counts (as before)",
          [r.exact for r in qrecs] == [True, True, False, False]
          and R.time_estimate(qrecs, exact=True, **kw) == (450, 2)
          and R.time_estimate(qrecs, exact=False, **kw) == (100, 2)
          and R.time_estimate(qrecs, **kw) == (255, 4)
          and R.time_estimate(qrecs[2:], exact=True, **kw) is None,
          f"{[r.exact for r in qrecs]} {R.time_estimate(qrecs, exact=True, **kw)} "
          f"{R.time_estimate(qrecs, exact=False, **kw)}")
    g = rs.RunSpec(run_mode="gentle", **base)
    c = rs.CustomSpec(input_file=str(snap), run_mode="performance")
    for name in P.BUILTIN_PRESETS:
        P.apply_preset(g, P.preset(name, {})[1])
        P.apply_preset(c, P.preset(name, {})[1])
    check("presets leave the machine use alone: not a preset key, untouched by every built-in preset (Grids and "
          "Custom); its label is the selector's",
          g.run_mode == "gentle" and c.run_mode == "performance" and "run_mode" not in P.PRESET_KEYS
          and "run_mode" not in P.capture(g) and P.OPTION_TEXT["run_mode"] == ("Machine use", "--run-mode"))
    driver_device_table()


def runs_on_line(tmp: Path):
    """The run tabs' grey line under the Machine use (runspec.runs_on_text): where a job runs, in words, from the
    device its quality implies (mode().gpu) on this build, for every estimator, quality and machine use; the 2D
    programs and a CPU-only build said as such; never a number or a '--'."""
    print("launcher: the 'Runs on' line:")
    small = tmp / "runs_on_small.hdf5"
    import h5py
    with h5py.File(small, "w") as h:
        h.create_group("Header").attrs["NumPart_Total"] = [0, 32 ** 3, 0, 0, 0, 0]
    for build in ("gpu", "cpu"):
        with fake_build(build):
            jobs = [rs.RunSpec(estimator=e, quality=q, run_mode=m) for e in ("ps", "dtfe") for q in rs.QUALITIES
                    for m in rs.RUN_MODES]
            jobs += [rs.PipelineSpec(quality=q, run_mode=m) for q in rs.QUALITIES for m in rs.RUN_MODES]
            jobs += [rs.CustomSpec(estimator=e, quality=q) for e in ("ps", "dtfe") for q in rs.QUALITIES]
            jobs += [rs.CustomSpec(input_file=str(small), quality="fast"), rs.CustomSpec(input_file=str(small))]
            texts = [(rs.runs_on_text(j), j) for j in jobs]
            flat = rs.runs_on_text(rs.CustomSpec(dim=2))
            words = lambda j: (f"chosen for {rs.QUALITY_NAME[j.quality]}, "             # noqa: E731
                               f"{j.run_mode.capitalize()}")
            if build == "gpu":
                both = lambda j: (getattr(j, "estimator", "ps") == "ps" and j.quality == "exact"     # noqa: E731
                                  and j.run_mode == "performance")
                ok = all((t.startswith("Runs on: the GPU") or t.startswith("Runs on: the CPU and the GPU together"))
                         == j.mode().gpu and (t.startswith("Runs on: the CPU (")) == (not j.mode().gpu)
                         and words(j) in t and t.startswith("Runs on: the CPU and the GPU together") == both(j)
                         and bool(rs.runs_on_tip(j)) == both(j) for t, j in texts)
                cpu = [t for t, _ in texts if t.startswith("Runs on: the CPU (")]
                # (the Grids job of standard DTFE's Fast in performance; the Custom jobs run balanced)
                ok = ok and sorted(cpu) == ["Runs on: the CPU (chosen for Fast, Balanced, a small snapshot)",
                                            "Runs on: the CPU (chosen for Fast, Performance)"]
            else:
                ok = all(t == "Runs on: the CPU (this build has no GPU)" and not rs.runs_on_tip(j) for t, j in texts)
            check(f"the 'Runs on' line ({build} build): the GPU exactly where the quality's device is the GPU (PS-DTFE's "
                  "Exact in performance: 'the CPU and the GPU together', the Device row's words, with the precise "
                  "statement as its tooltip), the CPU for standard DTFE's Fast in performance and PS-DTFE's Fast on a "
                  "small snapshot, with the quality and machine use it was chosen for; a CPU-only build says so; the "
                  "2D programs too; no digit or '--'",
                  ok and flat == "Runs on: the CPU (the 2D programs have no GPU)"
                  and not any(ch.isdigit() or "--" in t for t, _ in texts for ch in t),
                  str(sorted({t for t, _ in texts})))
    small.unlink()


def advanced_settings(tmp: Path):
    """The opt-in Advanced settings (the owner: 'an advanced mode in the GUI that lets the user ticker with the
    settings more? but make it opt in'): one field, advanced, on the Grids, Pipeline and Custom jobs, keyed by
    ADVANCED_KEYS (the removed fields' names), an absent key Automatic. G1: a job without an override saves and builds
    as the two-mode job. G2: an override builds the launcher's explicit setting before the two modes (its flags and
    knobs, in their places), at least once per row and tab, spelled out from the code of that launcher. Saved
    overrides are cleaned, an older saved job never becomes one with overrides, an override the job cannot honour is
    not passed and the checks list says why, the 'Runs on' line follows a Device override, and one place
    (runspec._refusal, read through mode() and the four accessors) decides what is passed."""
    import inspect
    import json
    import re
    import presets as P
    print("launcher: the Advanced settings (opt-in overrides):")
    specs = (rs.RunSpec, rs.PipelineSpec, rs.CustomSpec)
    kind_of = {rs.RunSpec: "grids", rs.PipelineSpec: "pipeline", rs.CustomSpec: "custom"}
    check("ADVANCED_KEYS are the removed fields' names in the rows' order; every run job has one new field, advanced, "
          "the last one, empty by default; a Custom job takes every key but 'precision' (its own row)",
          rs.ADVANCED_KEYS == ("gpu", "nsub", "face_gradient", "partition", "max_concurrent", "mem_budget_gb",
                               "parallel_triangulation", "ordered_merge", "precision")
          and set(rs.ADVANCED_KEYS) <= set(rs.REMOVED_SETTINGS)
          and all(list(cls.__dataclass_fields__)[-1] == "advanced" and cls().advanced == {} for cls in specs)
          and rs.advanced_keys(rs.RunSpec) == rs.advanced_keys(rs.PipelineSpec) == rs.ADVANCED_KEYS
          and rs.advanced_keys(rs.CustomSpec) == rs.ADVANCED_KEYS[:-1])

    root = str(tmp / "adv")
    inp = str(tmp / "adv_in" / "toy.hdf5")
    out = tmp / "adv_out"
    run = lambda adv, **k: rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[3, 1], advanced=adv, **k)  # noqa
    pipe = lambda adv, **k: rs.PipelineSpec(data_root=root, sims=["Toy-64"], advanced=adv, **k)           # noqa
    cust = lambda adv, **k: rs.CustomSpec(input_file=inp, output_dir=str(out), advanced=adv, **k)         # noqa
    full = lambda s: (s.command(), s.run_env()) if isinstance(s, rs.CustomSpec) else s.command()          # noqa

    # ---- G1: no override, the two-mode job (two_quality_modes holds the two-mode commands themselves)
    plain = [run({}), run({}, estimator="dtfe", quality="fast"), pipe({}), cust({}), cust({}, dim=2, estimator="dtfe")]
    saved = [x for s in plain for x in (json.dumps(s.to_dict()), s.memory_key(),
                                        json.dumps(rs.QueuedJob.of(kind_of[type(s)], s).to_dict()))]
    saved += [json.dumps(s.settings_sidecar()) for s in plain if isinstance(s, rs.CustomSpec)]
    with fake_build("gpu"):
        same_job = all(full(s) == full(type(s).from_dict(dict(s.to_dict(), advanced={}))) for s in plain)
        lines = [m for s in plain for _, m in s.problems()] + [rs.runs_on_text(s) for s in plain]
    check("G1: a job without an override saves exactly as the two-mode job (no 'advanced' key in to_dict, its queue "
          "entry, its memory key or a Custom settings sidecar; the other keys in field order), an explicit empty dict "
          "is the same job, and no check or 'Runs on' line speaks of Advanced",
          all(list(s.to_dict()) == [f for f in type(s).__dataclass_fields__ if f != "advanced"] for s in plain)
          and not any("advanced" in x for x in saved) and same_job
          and not any("Advanced" in t for t in lines), str([s.to_dict() for s in plain][:1]))

    # ---- G2: an override is the old launcher's explicit setting (its code path before the two modes)
    ps, dt = str(rs.PS_SCRIPT), str(rs.DTFE_SCRIPT)
    std_f = "density_a velocity_a"
    head = lambda b, mode="balanced", f="density_a velocity_a dispersion_a": (                           # noqa
        [str(rs.REPO_ROOT / b), inp, str(out / "toy"), "--grid", "128", "--input-format", "105", "--mpc-unit", "1",
         "--run-mode", mode, "--periodic", "--field", *f.split()])
    bad = []

    def want(name, got, ok):
        if not ok:
            bad.append(f"{name}: {got}")

    with fake_build("gpu"), double_pair(True):
        a, e = run({"gpu": False, "partition": 4, "max_concurrent": 2, "mem_budget_gb": 40}).command()
        want("Grids PS Exact, CPU, partitions 4, concurrency 2, budget 40", (a, e),
             a == [ps, "-s", "Toy-64", "-g", "512", "-e", "1", "3"]
             and list(e)[:4] == ["DTFE_DATA_ROOT", "PARTITION", "MAX_CONCURRENT", "DTFE_MEM_BUDGET_GB"]
             and (e["PARTITION"], e["MAX_CONCURRENT"], e["DTFE_MEM_BUDGET_GB"]) == ("4 4 4", "2", "40")
             and e["DTFE_GPU"] == e["PS_GPU"] == e["PS_METAL"] == "0" and e["GPU_SHARE"] == "auto"
             and "PARTITION='4 4 4' MAX_CONCURRENT=2 DTFE_MEM_BUDGET_GB=40" in rs.shell(a, e))
        a, e = run({"nsub": 5, "parallel_triangulation": True, "ordered_merge": False}, quality="fast").command()
        want("Grids PS Fast, samples 5, parallel, completion order", (a, e),
             a == [ps, "-s", "Toy-64", "-g", "512", "-n", "5", "-m", "1", "3"]
             and e["PARALLEL_TRIANGULATION"] == e["PS_PARALLEL_TRI"] == "1"
             and e["PS_MERGE"] == "completion-order" and e["ORDERED_MERGE"] == "0")
        stream = run({"face_gradient": False})
        want("Grids PS Exact, the stream average", stream.command()[1],
             stream.command()[1]["FACE_GRADIENT"] == "0" and stream.grid_gb() < run({}).grid_gb()
             and run({"face_gradient": True}).command() == run({}).command())
        dbl = run({"precision": "double"})
        a, e = dbl.command()
        want("Grids PS double", (a, e), list(e)[:2] == ["DTFE_DATA_ROOT", "DTFE_PRECISION"]
             and e["DTFE_PRECISION"] == "double" and "-m" in a and dbl.grid_gb() == 2 * run({}).grid_gb())
        want("Grids DTFE Exact, CPU", run({"gpu": False}, estimator="dtfe").command(),
             run({"gpu": False}, estimator="dtfe").command()[0] == [dt, "-s", "Toy-64", "-g", "512", "-e", "1", "3"]
             and run({"gpu": False}, estimator="dtfe").command()[1]["DTFE_GPU"] == "0"
             and run({"gpu": False}, estimator="dtfe").command()[1]["DTFE_METAL"] == "0")
        a, e = run({"gpu": True}, estimator="dtfe", quality="fast", run_mode="performance").command()
        want("Grids DTFE Fast, performance, GPU", (a, e),
             a == [dt, "-s", "Toy-64", "-g", "512", "-m", "1", "3"] and e["DTFE_GPU"] == e["DTFE_METAL"] == "1")
        p = pipe({"nsub": 2, "partition": 2, "gpu": False}, quality="fast")
        e, chk = p.command()[1], p.check_step("Toy-64", [3, 1], 1024)
        want("Pipeline Fast, samples 2, partitions 2, CPU", (e, chk.argv, chk.env),
             e["SUBSAMPLES"] == e["AVG_SUBSAMPLES"] == "2" and e["PARTITION"] == "2 2 2" and e["DTFE_GPU"] == "0"
             and "-m" not in chk.argv and chk.env["PARTITION"] == "2 2 2" and chk.env["AUTO_TUNE_REPORT"] == "1")
        e = pipe({"precision": "double", "max_concurrent": 3}).command()[1]
        want("Pipeline double, concurrency 3", e, e["DTFE_PRECISION"] == "double" and e["MAX_CONCURRENT"] == "3")
        c = cust({"gpu": False, "partition": 4, "max_concurrent": 2, "mem_budget_gb": 40})
        want("Custom PS Exact, CPU, partitions 4, concurrency 2, budget 40", (c.command(), c.run_env()),
             c.command() == head("PS-DTFE") + ["--exact", "--ps-volume-weighted", "--face-gradient", "--ps-merge",
                                               "index-order", "--partition", "4", "4", "4", "--max-concurrent", "2"]
             and c.run_env() == {"DTFE_MEM_BUDGET_GB": "40", "DTFE_FACE_GRADIENT": ""}
             and c.check_step().env == c.run_env())
        c = cust({"partition": 3}, dim=2)
        want("Custom PS 2D, partitions 3", c.command(),
             c.command() == head("PS-DTFE-2d") + ["--exact", "--ps-volume-weighted", "--ps-merge", "index-order",
                                                  "--partition", "3", "3"])
        c = cust({"nsub": 4, "parallel_triangulation": True, "ordered_merge": False, "gpu": False}, quality="fast")
        want("Custom PS Fast, samples 4, parallel, completion order, CPU", c.command(),
             c.command() == head("PS-DTFE") + ["--sampled", "--subsamples", "4", "--ps-volume-weighted",
                                               "--parallel-triangulation", "--ps-merge", "completion-order"])
        c = cust({"gpu": True}, estimator="dtfe", quality="fast", run_mode="performance")
        want("Custom DTFE Fast, performance, GPU", c.command(),
             c.command() == head("DTFE", "performance", std_f) + ["--gpu", "--sampled", "--padding", "15"])
        c = cust({"max_concurrent": 5}, estimator="dtfe")
        want("Custom DTFE Exact, concurrency 5", c.command(),
             c.command() == head("DTFE", f=std_f) + ["--gpu", "--exact", "--padding", "15", "--max-concurrent", "5"])
        c = cust({"partition": 2, "mem_budget_gb": 16}, estimator="dtfe")
        want("Custom DTFE Exact, partitions 2, budget 16", (c.command(), c.run_env()),
             c.command() == head("DTFE", f=std_f) + ["--gpu", "--exact", "--padding", "15", "--partition", "2", "2",
                                                     "2"]
             and c.run_env() == {"DTFE_MEM_BUDGET_GB": "16", "DTFE_FACE_GRADIENT": ""}
             and cust({"partition": 2}, estimator="dtfe", dim=2).command()[-3:] == ["--partition", "2", "2"])
        c = cust({"face_gradient": False})
        want("Custom PS Exact, the stream average", c.command(),
             c.command() == head("PS-DTFE") + ["--exact", "--gpu", "--ps-volume-weighted", "--per-tet-gradient",
                                               "--gpu-share", "auto", "--ps-merge", "index-order"])
        p = pipe({"face_gradient": False, "mem_budget_gb": 24, "parallel_triangulation": True, "ordered_merge": False})
        e, chk = p.command()[1], p.check_step("Toy-64", [1], 1024)
        want("Pipeline, the stream average, budget 24, parallel, completion order", (e, chk.env),
             e["FACE_GRADIENT"] == "0" and e["DTFE_MEM_BUDGET_GB"] == "24"
             and e["PARALLEL_TRIANGULATION"] == e["PS_PARALLEL_TRI"] == "1"
             and e["PS_MERGE"] == "completion-order" and e["ORDERED_MERGE"] == "0"
             and all(chk.env[k] == e[k] for k in ("FACE_GRADIENT", "DTFE_MEM_BUDGET_GB", "PARALLEL_TRIANGULATION",
                                                  "PS_MERGE")))
        s = run({"partition": 3, "max_concurrent": 4, "mem_budget_gb": 32, "precision": "double"}, estimator="dtfe")
        (a, e), chk = s.command(), s.check_step()
        want("Grids DTFE Exact, double, partitions 3, concurrency 4, budget 32", (a, e),
             a == [dt, "-s", "Toy-64", "-g", "512", "-m", "-e", "1", "3"]
             and list(e)[:5] == ["DTFE_DATA_ROOT", "DTFE_PRECISION", "PARTITION", "MAX_CONCURRENT", "DTFE_MEM_BUDGET_GB"]
             and (e["DTFE_PRECISION"], e["PARTITION"], e["MAX_CONCURRENT"], e["DTFE_MEM_BUDGET_GB"])
             == ("double", "3 3 3", "4", "32") and chk.env["PARTITION"] == "3 3 3"
             and chk.env["AUTO_TUNE_REPORT"] == "1" and s.grid_gb() == 2 * run({}, estimator="dtfe").grid_gb())
    check("G2: an override builds the launcher's explicit setting before the two modes: the device (-m / --gpu, the "
          "GPU knobs both ways), the samples (-n, SUBSAMPLES, --subsamples), the stream average (FACE_GRADIENT=0), the "
          "partitions (PARTITION, --partition per axis, 2D too), the concurrency (MAX_CONCURRENT, --max-concurrent), "
          "the budget (DTFE_MEM_BUDGET_GB, the Custom binary's environment too), the parallel insertion, the "
          "completion-order merge and the double pair (DTFE_PRECISION, twice the grid memory), in their old places",
          not bad, " || ".join(bad[:2]))

    # ---- not honoured: not passed (the Automatic command), and the checks list says why, as a warning
    def refused(spec, key):
        return [m for lvl, m in spec.problems() if lvl == "warning" and m.startswith(
            f"Advanced: {rs.ADVANCED_NAME[key]} ") and "is not passed" in m]
    bad = []
    with fake_build("cpu"):
        s = run({"gpu": True})
        want("CPU build, Device GPU", (s.command(), refused(s, "gpu")), s.command() == run({}).command()
             and refused(s, "gpu") == ["Advanced: Device set to GPU is not passed: this build has no GPU (METAL=1 on a Mac); "
                                       "Automatic applies"]
             and rs.runs_on_text(s) == "Runs on: the CPU (this build has no GPU)")
    with fake_build("gpu"):
        s = run({"nsub": 4})
        want("Exact, samples 4", refused(s, "nsub"), s.command() == run({}).command()
             and refused(s, "nsub") == ["Advanced: Samples set to 4 is not passed: Exact uses no samples; Automatic applies"])
        s = run({"nsub": 4, "face_gradient": False, "parallel_triangulation": True, "ordered_merge": False},
                estimator="dtfe")
        want("standard DTFE, the PS-DTFE overrides", s.problems(),
             s.command() == run({}, estimator="dtfe").command()
             and all(len(refused(s, k)) == 1 for k in ("nsub", "face_gradient", "parallel_triangulation",
                                                        "ordered_merge"))
             and refused(s, "face_gradient") == ["Advanced: Velocity derivatives set to stream average is not "
                                                 "passed: a PS-DTFE setting; Automatic applies"])
        s = run({"face_gradient": True}, quality="fast")
        want("Fast, exact face integrals", refused(s, "face_gradient"),
             s.command() == run({}, quality="fast").command() and s.command()[1]["FACE_GRADIENT"] == ""
             and "the face integrals need Exact" in " ".join(refused(s, "face_gradient")))
        s = cust({"gpu": True, "face_gradient": False, "parallel_triangulation": True}, dim=2)
        want("Custom 2D, device, route and parallel insertion", s.problems(),
             full(s) == full(cust({}, dim=2)) and all(refused(s, k) for k in ("gpu", "face_gradient",
                                                                                "parallel_triangulation")))
        s = cust({"ordered_merge": False, "face_gradient": False}, extra_args="--ps-merge fixed-point --face-gradient")
        want("Custom, extra options naming the merge and the route", s.problems(),
             full(s) == full(cust({}, extra_args="--ps-merge fixed-point --face-gradient"))
             and "completion-order" not in s.command()
             and "the extra options name the merge" in " ".join(refused(s, "ordered_merge"))
             and "the extra options name the route" in " ".join(refused(s, "face_gradient")))
        s = cust({"face_gradient": True}, halo_release=1000.0)
        want("Custom, halo release, exact face integrals", s.problems(),
             full(s) == full(cust({}, halo_release=1000.0)) and refused(s, "face_gradient"))
        with double_pair(False):
            s = run({"precision": "double"})
            want("Double without the pair", (s.command(), refused(s, "precision")),
                 s.command() == run({}).command() and s.grid_gb() == run({}).grid_gb()
                 and "PS-DTFE-double is not built" in " ".join(refused(s, "precision"))
                 and "DTFE-double is not built" in " ".join(refused(run({"precision": "double"}, estimator="dtfe"),
                                                                    "precision")))
        levels_seen = {lvl for s in (run({"gpu": True, "nsub": 4}), cust({"gpu": True}, dim=2))
                       for lvl, m in s.problems() if m.startswith("Advanced:")}
        want("no error from an override", levels_seen, levels_seen <= {"info", "warning"})
    with fake_stamps(lambda est, prec, dim: [".gpu_mode_metal" if dim == 3 else ".gpu_mode_off"]):   # no TBB
        s = run({"parallel_triangulation": True})
        want("no TBB, parallel insertion", s.problems(), s.command()[1]["PARALLEL_TRIANGULATION"] == "0"
             and "this build has no TBB" in " ".join(refused(s, "parallel_triangulation")))
    # a single pair with the GPU beside a CPU-only double pair: the precision is resolved first, so the device (its
    # Automatic, a GPU override, the 'Runs on' line) is the double program's
    with fake_stamps(lambda est, prec, dim: [".gpu_mode_metal" if dim == 3 and prec == "single" else ".gpu_mode_off"]
                     + ([".tbb_1"] if est == "ps" and dim == 3 else [])), double_pair(True):
        s, g = run({"precision": "double"}), run({"precision": "double", "gpu": True})
        want("double pair without the GPU", (s.command(), g.problems()),
             not s.mode().gpu and "-m" not in s.command()[0] and s.command()[1]["DTFE_PRECISION"] == "double"
             and rs.runs_on_text(s) == "Runs on: the CPU (this build has no GPU)"
             and g.command() == s.command() and "this build has no GPU" in " ".join(refused(g, "gpu"))
             and "-m" in run({"gpu": True}).command()[0] and "-m" in run({}).command()[0])
    check("an override the job cannot honour is not passed (the job builds the Automatic command) and the checks list "
          "says why as a warning, never an error: the GPU on a CPU-only build or on a CPU-only double pair, the "
          "samples under Exact, the PS-DTFE rows on a standard-DTFE job, the face integrals under Fast or with the halo "
          "release, the 2D programs' device, route and parallel insertion, a merge or route the extra options name, "
          "the parallel insertion without TBB, Double without the double programs",
          not bad, " || ".join(bad[:2]))

    # ---- the info line, the 'Runs on' line
    real_ram = rs.machine_ram_gb
    with fake_build("gpu"), double_pair(True):
        info = [m for lvl, m in run({"gpu": False, "partition": 4}).problems() if lvl == "info"]
        # the machine's memory as a known figure (a machine whose memory reads 0 would leave the budget warning
        # untested), and as 0 (the OS will not say: no warning, nothing to compare)
        try:
            rs.machine_ram_gb = lambda: 64.0
            over = run({"mem_budget_gb": 100}).problems()
            under = run({"mem_budget_gb": 64}).problems()
            rs.machine_ram_gb = lambda: 0.0
            unknown = run({"mem_budget_gb": 4096}).problems()
        finally:
            rs.machine_ram_gb = real_ram
        old_warnings = (run({"parallel_triangulation": True}).problems(),
                        run({"parallel_triangulation": True}, estimator="dtfe").problems())
        texts = (rs.advanced_text(run({"face_gradient": False, "mem_budget_gb": 40, "ordered_merge": False})),
                 rs.advanced_text(run({"face_gradient": False, "parallel_triangulation": True, "precision": "double",
                                       "nsub": 2, "max_concurrent": 3}, estimator="dtfe")),
                 rs.advanced_text(run({"gpu": "mixed"})), rs.advanced_text(run({"gpu": True})))
        on = [rs.runs_on_text(s) for s in (run({"gpu": False}), run({"gpu": True}, run_mode="performance"),
                                           run({"gpu": True}, estimator="dtfe", quality="fast",
                                               run_mode="performance"),
                                           pipe({"gpu": False}), cust({"gpu": True}, dim=2), run({"partition": 2}),
                                           cust({"gpu": False}, quality="fast"), run({"gpu": "mixed"}),
                                           cust({"gpu": "mixed"}, run_mode="gentle"), run({}, run_mode="performance"))]
    rounding = "parallel triangulation: faster on small sets, but two identical runs then differ at float rounding"
    swap = "memory budget 100 GB is more than this machine's 64 GB: the run may swap"
    check("the checks list names a job's overrides in one info line ('Advanced: Device: CPU; Partitions: 4'), in one "
          "form for every row ('Name: value'), the budget in GB, a held stream average in the row's own words on a "
          "standard-DTFE job too; a passed parallel insertion and a budget above this machine's memory keep the "
          "launcher's warnings from before the two modes (the memory read as 64 GB: 100 warns, 64 does not; read "
          "as 0, unknown: no warning)",
          "Advanced: Device: CPU; Partitions: 4" in info
          and ("warning", swap) in over and not any("may swap" in m for _, m in under + unknown)
          and any(lvl == "warning" and m.startswith(rounding) for lvl, m in old_warnings[0])
          and not any(m.startswith(rounding) for _, m in old_warnings[1])
          and texts == ("Velocity derivatives: stream average; Memory budget: 40 GB; Merge order: completion order",
                        "Samples: 2; Velocity derivatives: stream average; Concurrent triangulations: 3; Parallel "
                        "triangulation: on; Precision: double", "Device: CPU + GPU", "Device: GPU"),
          str((info, texts, over)))
    check("the 'Runs on' line follows a Device override and ends '(set in Advanced)'; GPU keeps every partition on the "
          "GPU in performance too, CPU + GPU reads 'the CPU and the GPU together' in any machine use (as Automatic "
          "does in performance); the 2D programs and an override of something else leave it as it was; no digit "
          "(but the 2D programs' name)",
          on == ["Runs on: the CPU (set in Advanced)", "Runs on: the GPU (set in Advanced)",
                 "Runs on: the GPU (set in Advanced)", "Runs on: the CPU (set in Advanced)",
                 "Runs on: the CPU (the 2D programs have no GPU)", "Runs on: the GPU (chosen for Exact, Balanced)",
                 "Runs on: the CPU (set in Advanced)", "Runs on: the CPU and the GPU together (set in Advanced)",
                 "Runs on: the CPU and the GPU together (set in Advanced)",
                 "Runs on: the CPU and the GPU together (chosen for Exact, Performance)"]
          and not any(re.search(r"\d", t.replace("2D", "")) for t in on), str(on))

    # ---- serialising and reading
    with fake_build("gpu"):
        jobs = [run({"partition": 4, "gpu": False, "mem_budget_gb": 40}), pipe({"ordered_merge": False, "nsub": 2}),
                cust({"max_concurrent": 2, "face_gradient": False}, dim=3)]
        dicts = [s.to_dict() for s in jobs]
        back = [type(s).from_dict(json.loads(json.dumps(d))) for s, d in zip(jobs, dicts)]
        queued = [rs.QueuedJob.from_dict(json.loads(json.dumps(rs.QueuedJob.of(kind_of[type(s)], s).to_dict())))
                  for s in jobs]
        cmds_kept = all(full(q.build()) == full(s) for q, s in zip(queued, jobs))
    check("serialising: a job with overrides saves them under the nested key 'advanced' (its last key, in the rows' "
          "order, the budget a float) and loads back the same job; its queue entry keeps them and rebuilds the same "
          "command; a Custom settings sidecar carries them; they change the memory key",
          [list(d)[-1] for d in dicts] == ["advanced"] * 3
          and dicts[0]["advanced"] == {"gpu": False, "partition": 4, "mem_budget_gb": 40.0}
          and list(dicts[0]["advanced"]) == ["gpu", "partition", "mem_budget_gb"]
          and isinstance(dicts[0]["advanced"]["mem_budget_gb"], float)
          and back == jobs and [q.spec["advanced"] for q in queued] == [d["advanced"] for d in dicts] and cmds_kept
          and jobs[2].settings_sidecar()["advanced"] == {"face_gradient": False, "max_concurrent": 2}
          and jobs[0].memory_key() != run({}).memory_key(), str(dicts))
    junk = {"bogus": 1, "gpu": "yes", "nsub": 7, "face_gradient": 1, "partition": 0, "max_concurrent": True,
            "mem_budget_gb": -1, "parallel_triangulation": False, "ordered_merge": True, "precision": "single"}
    edges = {"nsub": 6.0, "partition": 8, "max_concurrent": 16, "mem_budget_gb": 4096}
    beyond = {"nsub": 2.5, "partition": 9, "max_concurrent": float("nan"), "mem_budget_gb": 4097}
    # an integer beyond the float range (a hand-edited store): dropped, no OverflowError; the infinities too
    huge = {"nsub": 10 ** 400, "partition": -10 ** 400, "max_concurrent": float("inf"), "mem_budget_gb": float("-inf")}
    base = rs.RunSpec().to_dict()
    loaded = [rs.RunSpec.from_dict(dict(base, advanced=v)).advanced
              for v in (junk, edges, beyond, "x", None, [1], 3, huge)]
    devices = [rs.RunSpec.from_dict(dict(base, advanced={"gpu": v})).advanced
               for v in (True, False, "mixed", "MIXED", "both", 1, 0)]
    q_junk = rs.QueuedJob.from_dict({"kind": "grids", "spec": dict(base, advanced=dict(junk, partition=2))})
    q_none = rs.QueuedJob.from_dict({"kind": "grids", "spec": dict(base, advanced=junk)})
    check("reading: unknown keys and values outside a row's range are dropped without an error (a bool for a "
          "number, a value Automatic already has, a non-dict, an integer beyond the float range, an infinity); the "
          "range ends are kept, normalised; the Device keeps True, False and 'mixed' (CPU + GPU) and nothing else; a "
          "Custom job drops 'precision' (its own row); a queue entry is cleaned (no key left when nothing is)",
          loaded[0] == {} and loaded[1] == {"nsub": 6, "partition": 8, "max_concurrent": 16, "mem_budget_gb": 4096.0}
          and type(loaded[1]["nsub"]) is int and loaded[2:] == [{}] * 6
          and devices == [{"gpu": True}, {"gpu": False}, {"gpu": "mixed"}, {}, {}, {}, {}]
          and rs.CustomSpec.from_dict(dict(rs.CustomSpec().to_dict(), advanced={"precision": "double",
                                                                                  "partition": 2})).advanced
          == {"partition": 2} and rs.PipelineSpec(advanced={"precision": "double"}).advanced == {"precision": "double"}
          and q_junk.spec["advanced"] == {"partition": 2} and "advanced" not in q_none.spec
          and "advanced" not in rs.RunSpec.from_dict(dict(base, advanced=junk)).to_dict(), str(loaded))
    old = dict(base, deposit="sampled", gpu=True, nsub=2, partition=4, max_concurrent=3, mem_budget_gb=8.0,
               precision="double", parallel_triangulation=True, ordered_merge=False, face_gradient=False)
    old.pop("quality")
    old_pipe = dict(rs.PipelineSpec().to_dict(), deposit="exact", gpu=False, partition=2, precision="double")
    old_pipe.pop("quality")
    old_custom = dict(rs.CustomSpec().to_dict(), deposit="sampled", gpu=True, partition=4, precision="double")
    old_custom.pop("quality")
    olds = [rs.RunSpec.from_dict(old), rs.PipelineSpec.from_dict(old_pipe), rs.CustomSpec.from_dict(old_custom)]
    q_old = rs.QueuedJob.from_dict({"kind": "grids", "spec": old})
    preset_job = run({})
    P.apply_preset(preset_job, {"grid": 256, "deposit": "sampled", "gpu": False, "partition": 4, "nsub": 2})
    check("an older saved job (its removed settings at the top level: the GPU tick, nSub, the partition, the "
          "precision, ...) migrates as before and never becomes a job with overrides; nor does an older user preset",
          [j.advanced for j in olds] == [{}, {}, {}] and [j.quality for j in olds] == ["fast", "exact", "fast"]
          and olds[2].precision == "double" and not any("advanced" in j.to_dict() for j in olds)
          and "advanced" not in q_old.spec and rs.predates_modes(old) and preset_job.advanced == {}
          and preset_job.quality == "fast" and preset_job.grid == 256)
    check("the built-in presets stay three and carry no override; a preset never saves or sets one",
          len(P.BUILTIN_PRESETS) == 3 and "advanced" not in P.PRESET_KEYS
          and not any("advanced" in s for _, s in P.BUILTIN_PRESETS.values())
          and "advanced" not in P.capture(run({"gpu": False}))
          and not P.apply_preset(run({"gpu": False}), {"advanced": {"partition": 2}}))

    # ---- one place: _refusal decides, mode() and the four accessors carry it to every builder
    builders = [rs.RunSpec.command, rs.RunSpec.problems, rs.RunSpec.grid_gb, rs.RunSpec.mode, rs.PipelineSpec.command,
                rs.PipelineSpec.check_step, rs.PipelineSpec.problems, rs.PipelineSpec.mode, rs.CustomSpec._own_command,
                rs.CustomSpec.run_env, rs.CustomSpec.problems, rs.CustomSpec.mode, rs.ps_run_env, rs._run_option_env,
                rs._run_option_problems, rs.face_gradient_env, rs.runs_on_text, rs.advanced_problems,
                rs.apply_advanced, rs._refusal, rs._passed, rs.job_precision, rs.job_partition, rs.job_concurrency,
                rs.job_memory_budget, rs.automatic_text, rs.advanced_text, rs._line_text, rs.advanced_rows,
                rs.advanced_choices]
    removed = sorted(set(rs.REMOVED_SETTINGS) - {"precision", "vertex_mass"})
    reads = re.compile(r"\b(?:self|spec|sp|c)\.(advanced|" + "|".join(removed) + r")\b")
    compares = re.compile(r"quality\s*[!=]=|[\"']fast[\"']|[\"']sampled[\"']\s*(?:if|else)")
    leaks = {f.__qualname__: hits for f in builders
             for hits in [reads.findall(inspect.getsource(f)) + compares.findall(inspect.getsource(f))] if hits}
    many = {"gpu": False, "nsub": 2, "face_gradient": False, "partition": 4, "max_concurrent": 2, "mem_budget_gb": 40,
            "parallel_triangulation": True, "ordered_merge": False, "precision": "double"}
    real = rs._refusal
    with fake_build("gpu"), double_pair(True):
        jobs = [run(many), run(many, quality="fast"), run(many, estimator="dtfe"), pipe(many, quality="fast"),
                cust(many), cust(many, quality="fast", dim=2)]
        passed_some = all(full(s) != full(type(s)(**dict(s.to_dict(), advanced={}))) for s in jobs)
        rs._refusal = lambda *a: "refused here"
        try:
            none_passed = all(full(s) == full(type(s)(**dict(s.to_dict(), advanced={}))) for s in jobs)
            warned = all(sum("is not passed: refused here" in m for _, m in s.problems()) == len(s.advanced)
                         for s in jobs)
        finally:
            rs._refusal = real
    check("one place: no builder, accessor or text reads the overrides or a removed setting from the job itself (they "
          "read mode() and the accessors) or compares the quality; refusing every override in runspec._refusal alone "
          "turns every job's command and environment into the Automatic one, each override then warned about",
          not leaks and passed_some and none_passed and warned, str(leaks))

    # ---- the rows each tab shows, their choices (Automatic first), what Automatic gives
    with fake_build("gpu"), double_pair(True):
        rows = [rs.advanced_rows(s) for s in (run({}), run({}, estimator="dtfe"), pipe({}), cust({}), cust({}, dim=2),
                                              cust({}, estimator="dtfe"), cust({}, estimator="dtfe", dim=2))]
        gpu_choices = rs.advanced_choices(run({}), "gpu")
        route = (rs.advanced_choices(run({}), "face_gradient"), rs.advanced_choices(run({}, estimator="dtfe"),
                                                                                    "face_gradient"))
        firsts = {rs.advanced_choices(s, k)[0] for s in (run({}), cust({}, dim=2)) for k in rs.ADVANCED_KEYS}
        auto = {k: rs.automatic_text(run({k: v for k, v in many.items() if k != "precision"}), k)
                for k in rs.ADVANCED_KEYS}
        auto_fast = (rs.automatic_text(run({}, quality="fast"), "nsub"),
                     rs.automatic_text(run({}, quality="fast"), "face_gradient"),
                     rs.automatic_text(run({}, estimator="dtfe", quality="fast", run_mode="performance"), "gpu"))
    with fake_build("cpu"), double_pair(False):
        cpu_rows, cpu_gpu = rs.advanced_rows(run({})), rs.advanced_choices(run({}), "gpu")
    k = rs.ADVANCED_KEYS
    check("the rows each tab shows where the control was before the two modes: the device in 3D, the samples and the "
          "merge order for PS-DTFE (2D too), the route and the parallel insertion for PS-DTFE in 3D, the partitions, "
          "concurrency and budget always, the precision on Grids and Pipeline when the double programs are built",
          rows == [k, ("gpu", "partition", "max_concurrent", "mem_budget_gb", "precision"), k, k[:-1],
                   ("nsub", "partition", "max_concurrent", "mem_budget_gb", "ordered_merge"),
                   ("gpu", "partition", "max_concurrent", "mem_budget_gb"),
                   ("partition", "max_concurrent", "mem_budget_gb")]
          and cpu_rows == k[:-1], str(rows))
    check("each row's first choice is Automatic; the device offers the GPU and CPU + GPU only on a GPU build; the "
          "other route is the stream average (a PS-DTFE value, in its own words wherever it is held); Automatic's "
          "grey text from the job's mode without that row's override (Exact: no samples), no measured number",
          firsts == {(None, "Automatic")}
          and gpu_choices == [(None, "Automatic"), (False, "CPU"), (True, "GPU"), ("mixed", "CPU + GPU")]
          and cpu_gpu == [(None, "Automatic"), (False, "CPU")]
          and route == ([(None, "Automatic"), (True, "Exact face integrals"), (False, "Stream average")],
                        [(None, "Automatic"), (True, "Exact face integrals"), (False, "Stream average")])
          and auto == {"gpu": "Automatic: the GPU", "nsub": "Exact uses no samples",
                       "face_gradient": "Automatic: exact face integrals",
                       "partition": "Automatic: the program plans them",
                       "max_concurrent": "Automatic: the program plans them", "mem_budget_gb": "Automatic: the free memory",
                       "parallel_triangulation": "Automatic: the program's own rule",
                       "ordered_merge": "Automatic: index order", "precision": "Automatic: single"}
          and auto_fast == ("Automatic: 1", "Automatic: the stream average", "Automatic: the CPU")
          and not any(re.search(r"\d", t) for t in list(auto.values()) + list(rs.ADVANCED_ROW.values())),
          str((auto, auto_fast)))


def device_cpu_gpu(tmp: Path):
    """The Advanced Device row's three choices (the owner: 'shouldnt there be a CPU+GPU option as well?'): CPU, GPU
    and CPU + GPU, saved as False, True and 'mixed'. GPU keeps every partition of PS-DTFE's exact deposit on the GPU
    in every machine use ('--gpu-share 1'); CPU + GPU shares it at the program's own share in every machine use
    ('--gpu-share mixed', GPU_SHARE=mixed); a share is passed only where it applies (PS-DTFE's exact deposit on the
    GPU in 3D), never for standard DTFE, Fast or 2D. CPU + GPU is offered on a GPU build of a PS-DTFE job in 3D and
    single precision, and refused with its reason elsewhere. The Automatic note and the 'Runs on' line say the same
    thing in the same words; the memory check and the cluster script carry the run's device and share. A Custom
    job's extra options that name the device set it alone."""
    print("launcher: the Device row (CPU, GPU, CPU + GPU):")
    root = str(tmp / "dev")
    inp = str(tmp / "dev_in" / "toy.hdf5")
    out = tmp / "dev_out"
    run = lambda adv, **k: rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[3, 1], advanced=adv, **k)  # noqa
    pipe = lambda adv, **k: rs.PipelineSpec(data_root=root, sims=["Toy-64"], advanced=adv, **k)           # noqa
    cust = lambda adv, **k: rs.CustomSpec(input_file=inp, output_dir=str(out), advanced=adv, **k)         # noqa

    def after(a, flag):
        return a[a.index(flag) + 1] if flag in a else None

    def share(s):
        """The share a job passes: a driver job's GPU_SHARE (None for run_dtfe.sh), a Custom job's '--gpu-share'."""
        return after(s.command(), "--gpu-share") if isinstance(s, rs.CustomSpec) else s.command()[1].get("GPU_SHARE")

    def diff(a, b):
        return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}

    jobs = (run, pipe, cust)
    with fake_build("gpu"), double_pair(True):
        auto = [share(f({}, run_mode=m)) for f in jobs for m in rs.RUN_MODES]
        gpu = [share(f({"gpu": True}, run_mode=m)) for f in jobs for m in rs.RUN_MODES]
        mixed = [share(f({"gpu": "mixed"}, run_mode=m)) for f in jobs for m in rs.RUN_MODES]
        cpu = [share(f({"gpu": False})) for f in jobs]
        fast = [share(f({"gpu": v}, quality="fast")) for f in jobs for v in (True, "mixed")]
        std = [share(run({"gpu": True}, estimator="dtfe")), share(cust({"gpu": True}, estimator="dtfe")),
               share(cust({"gpu": True}, dim=2)), share(cust({"gpu": "mixed"}, dim=2))]
        # besides the share, the GPU and CPU + GPU jobs are the Automatic (GPU) job: the device flags, all else
        only_share = (diff(run({"gpu": True}).command()[1], run({}).command()[1]),
                      diff(pipe({"gpu": "mixed"}, run_mode="gentle").command()[1],
                           pipe({}, run_mode="gentle").command()[1]),
                      run({"gpu": "mixed"}).command()[0] == run({"gpu": True}).command()[0] == run({}).command()[0],
                      [a if a != "mixed" else "auto" for a in cust({"gpu": "mixed"}).command()] == cust({}).command())
        # the memory check and the cluster script: the same device and share as the run
        probes = [run({"gpu": "mixed"}).check_step(), pipe({"gpu": "mixed"}).check_step("Toy-64", [3, 1], 1024),
                  cust({"gpu": "mixed"}).check_step(), run({"gpu": True}).check_step(),
                  pipe({"gpu": True}).check_step("Toy-64", [3, 1], 1024), cust({"gpu": True}).check_step()]
        carried = [st.env.get("GPU_SHARE", after(st.argv, "--gpu-share")) for st in probes]
        on_gpu = ["-m" in probes[0].argv, "-m" in probes[1].argv, "--gpu" in probes[2].argv]
        steps = (run({"gpu": "mixed"}).steps() + pipe({"gpu": "mixed"}, render=False).steps()
                 + cust({"gpu": "mixed"}).steps())
        cluster, here = rs.export_script(steps, "t", slurm={"job_name": "x"}), rs.export_script(steps, "t")
        texts = {"choices": [rs.advanced_choices(s, "gpu") for s in (run({}), run({}, estimator="dtfe"),
                                                                      cust({}, dim=2), run({"precision": "double"}),
                                                                      cust({}, precision="double"))],
                 "automatic": [rs.automatic_text(s, "gpu") for s in (run({}), run({}, run_mode="performance"),
                                                                       run({"gpu": False}, run_mode="performance"),
                                                                       run({}, quality="fast", run_mode="performance"),
                                                                       run({"precision": "double"},
                                                                           run_mode="performance"),
                                                                       run({}, estimator="dtfe",
                                                                           run_mode="performance"))],
                 "runs_on": [rs.runs_on_text(s) for s in (run({"gpu": "mixed"}, run_mode="gentle"),
                                                           run({"gpu": True}, run_mode="performance"),
                                                           run({"precision": "double"}, run_mode="performance"))],
                 "tips": [rs.runs_on_tip(s) for s in (run({"gpu": "mixed"}), run({}, run_mode="performance"),
                                                       run({"gpu": True}, run_mode="performance"), run({}))]}

        def refusal(s):
            return [m for lvl, m in s.problems() if lvl == "warning" and m.startswith("Advanced: Device set to ")]
        refused = {name: (refusal(s), full_eq) for name, s, full_eq in (
            ("Fast", run({"gpu": "mixed"}, quality="fast"),
             run({"gpu": "mixed"}, quality="fast").command() == run({}, quality="fast").command()),
            ("standard DTFE", run({"gpu": "mixed"}, estimator="dtfe"),
             run({"gpu": "mixed"}, estimator="dtfe").command() == run({}, estimator="dtfe").command()),
            ("double", run({"gpu": "mixed", "precision": "double"}),
             run({"gpu": "mixed", "precision": "double"}).command() == run({"precision": "double"}).command()),
            ("2D", cust({"gpu": "mixed"}, dim=2), cust({"gpu": "mixed"}, dim=2).command() == cust({}, dim=2).command()))}
    with fake_build("cpu"):
        cpu_build = (refusal(run({"gpu": "mixed"})), run({"gpu": "mixed"}).command() == run({}).command(),
                     rs.advanced_choices(run({}), "gpu"))
    check("Device GPU: every partition on the GPU in every machine use (GPU_SHARE=1, '--gpu-share 1': the one "
          "intended change of an existing setting's command); CPU + GPU: GPU_SHARE=mixed, '--gpu-share mixed' in "
          "every machine use; Automatic keeps 'auto'; CPU, Fast, standard DTFE and 2D pass no share of their own (the "
          "drivers' pinned 'auto', which they do not pass); besides the share the command is the Automatic GPU job's",
          auto == ["auto"] * 9 and gpu == ["1"] * 9 and mixed == ["mixed"] * 9 and cpu == ["auto", "auto", None]
          and fast == ["auto"] * 4 + [None] * 2 and std == [None] * 4
          and only_share == ({"GPU_SHARE": ("1", "auto")}, {"GPU_SHARE": ("mixed", "auto")}, True, True),
          f"{auto} {gpu} {mixed} {cpu} {fast} {std} {only_share}")
    check("... the memory check of each tab and the exported scripts (a SLURM job too) carry the run's device and "
          "share",
          carried == ["mixed", "mixed", "mixed", "1", "1", "1"] and all(on_gpu)
          and all(s.count("GPU_SHARE=mixed") == 2 and s.count("--gpu-share mixed") == 1 for s in (cluster, here)),
          f"{carried} {on_gpu}")
    check("CPU + GPU is offered on a GPU build of a PS-DTFE job in 3D and single precision only (not standard DTFE, "
          "2D, double precision, a CPU-only build); held where it cannot run it is not passed (the Automatic "
          "command) and the checks say why in words: only the Exact deposit can be shared, standard DTFE runs on one "
          "device, the double-precision deposit is not shared, the 2D programs and a CPU-only build",
          texts["choices"] == [[(None, "Automatic"), (False, "CPU"), (True, "GPU"), ("mixed", "CPU + GPU")],
                               [(None, "Automatic"), (False, "CPU"), (True, "GPU")], [(None, "Automatic"), (False, "CPU")],
                               [(None, "Automatic"), (False, "CPU"), (True, "GPU")],
                               [(None, "Automatic"), (False, "CPU"), (True, "GPU")]]
          and all(eq and len(w) == 1 for w, eq in refused.values())
          and "only the Exact deposit can be shared" in refused["Fast"][0][0]
          and "standard DTFE runs on one device" in refused["standard DTFE"][0][0]
          and "the double-precision deposit is not shared" in refused["double"][0][0]
          and "the 2D programs run on the CPU" in refused["2D"][0][0]
          and cpu_build[0] and "this build has no GPU" in cpu_build[0][0] and cpu_build[1]
          and cpu_build[2] == [(None, "Automatic"), (False, "CPU")], str(refused))
    check("the Device row's Automatic note and the 'Runs on' line say the same thing in the same words: 'Automatic: "
          "CPU + GPU' and 'the CPU and the GPU together' where the job mixes by itself (PS-DTFE Exact in performance, "
          "single precision) or is set to; the precise statement (a split run only; the log says which) as the "
          "line's tooltip there, nowhere else",
          texts["automatic"] == ["Automatic: the GPU", "Automatic: CPU + GPU", "Automatic: CPU + GPU",
                                 "Automatic: the GPU", "Automatic: the GPU", "Automatic: the GPU"]
          and texts["runs_on"] == ["Runs on: the CPU and the GPU together (set in Advanced)",
                                   "Runs on: the GPU (set in Advanced)", "Runs on: the GPU (chosen for Exact, Performance)"]
          and texts["tips"][:2] == [rs.RUNS_ON_MIXED_TIP] * 2 and texts["tips"][2:] == ["", ""]
          and "split into partitions" in rs.RUNS_ON_MIXED_TIP and "log" in rs.RUNS_ON_MIXED_TIP
          and rs.RUNS_ON_MIXED_TIP.count("\n") == 1 and max(len(x) for x in rs.RUNS_ON_MIXED_TIP.split("\n")) <= 90,
          str(texts))

    # a Custom job's extra options that name the device ('--gpu', '--ps-gpu', '--gpu-share'): the tab adds no device
    # flag of its own, a Device setting is not passed, the 'Runs on' line says where the run goes and who set it
    with fake_build("gpu"):
        named = {extra: cust({"gpu": "mixed"}, extra_args=extra) for extra in
                 ("--gpu", "--ps-gpu --gpu-share=0.5", "--gpu-share mixed")}
        cmds = {e: s.command() for e, s in named.items()}
        lines = {e: rs.runs_on_text(s) for e, s in named.items()}
        warned = {e: [m for lvl, m in s.problems() if lvl == "warning"] for e, s in named.items()}
        errors = {e: [m for lvl, m in s.problems() if lvl == "error" and "extra options" in m]
                  for e, s in named.items()}
        auto_note = rs.automatic_text(cust({}, extra_args="--gpu"), "gpu")
        plain = cust({}, extra_args="--ps-merge fixed-point")
    check("Custom: extra options that name the device set it alone: the tab adds neither '--gpu' nor a share of its "
          "own (no 'repeat' error), a Device setting is not passed ('the extra options name the device'), and the "
          "'Runs on' line says where the run goes and that the Extra options set it ('--gpu-share' without '--gpu': "
          "the CPU, warned); extra options that do not name it change nothing",
          [c.count("--gpu") + c.count("--ps-gpu") for c in cmds.values()] == [1, 1, 0]
          and [c.count("--gpu-share") for c in cmds.values()] == [0, 0, 1]
          and cmds["--ps-gpu --gpu-share=0.5"][-2:] == ["--ps-gpu", "--gpu-share=0.5"]
          and list(lines.values()) == ["Runs on: the GPU (set in the Extra options)",
                                       "Runs on: the GPU (set in the Extra options, with their GPU share)",
                                       "Runs on: the CPU (set in the Extra options: a GPU share without the GPU)"]
          and all(any("Device set to CPU + GPU is not passed: the extra options name the device" in m for m in w)
                  for w in warned.values())
          and [any("--gpu-share without --gpu" in m for m in w) for w in warned.values()] == [False, False, True]
          and not any(errors.values()) and auto_note == "Automatic: the extra options name the device"
          and "--gpu" in plain.command() and after(plain.command(), "--gpu-share") == "auto",
          f"{cmds} {lines} {warned} {errors}")
    # ... but standard DTFE has no '--gpu-share': there it names no device, it is an error (the program refuses it)
    with fake_build("gpu"):
        std_share = cust({}, estimator="dtfe", extra_args="--gpu-share 0.5")
        std_errors = [m for lvl, m in std_share.problems() if lvl == "error" and "--gpu-share" in m]
        std_line, std_cmd = rs.runs_on_text(std_share), std_share.command()
        std_warned = [m for lvl, m in std_share.problems() if lvl == "warning" and "--gpu-share" in m]
        ps_errors = [m for lvl, m in cust({}, extra_args="--gpu --gpu-share 0.5").problems()
                     if lvl == "error" and "--gpu-share" in m]
    check("Custom, standard DTFE: '--gpu-share' among the extra options is an error (a PS-DTFE option, which the "
          "standard program refuses at the start) and names no device there (the tab keeps its own '--gpu' and "
          "'Runs on' line, no 'without --gpu' warning); a PS-DTFE job's is no error",
          len(std_errors) == 1 and "PS-DTFE option" in std_errors[0] and std_cmd.count("--gpu") == 1
          and std_line.startswith("Runs on: the GPU (chosen for ") and not std_warned and not ps_errors,
          f"{std_errors} {std_line} {std_cmd[-6:]} {std_warned} {ps_errors}")


def finishing_round(tmp: Path):
    """The launcher's open points closed in the finishing round (2026-11-13): a refused Advanced setting leaves the
    memory key alone; a queue entry saved before the two modes gets a title in their words; the Custom tab's extra
    '--ps-linear-deposit' is an error (the tab always passes the volume-weighted velocities, which the program refuses
    beside it); the parallel insertion is not offered without TBB, its Automatic note says so; the full-resolution
    grids are the most a run holds at once (a face run's face sums are gone before the derived fields and the web
    classes are made), so a 512^3 PS-DTFE job with the default fields and the web classes needs no scratch folder."""
    print("launcher: the finishing round's open points:")
    root = str(tmp / "fin")
    run = lambda adv, **k: rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[1], advanced=adv, **k)  # noqa
    pipe = lambda adv, **k: rs.PipelineSpec(data_root=root, sims=["Toy-64"], advanced=adv, **k)        # noqa
    with fake_build("gpu"), double_pair(False):
        keys = (run({"nsub": 4}).memory_key() == run({}).memory_key(),
                run({"nsub": 4}, quality="fast").memory_key() != run({}, quality="fast").memory_key(),
                pipe({"precision": "double"}).memory_key() == pipe({}).memory_key(),
                run({"gpu": "mixed"}, estimator="dtfe").memory_key() == run({}, estimator="dtfe").memory_key(),
                run({"gpu": "mixed"}).memory_key() != run({}).memory_key())
    check("a refused Advanced setting leaves the job's memory key alone (its command is the Automatic one: Samples "
          "under Exact, Double without the double programs, CPU + GPU on standard DTFE); a passed one changes it",
          all(keys), str(keys))
    old = {k: v for k, v in rs.RunSpec(sim="TNG50-4-Dark", snapshots=[99]).to_dict().items() if k != "quality"}
    old.update(deposit="sampled", gpu=True, nsub=2)
    q_old = rs.QueuedJob.from_dict({"kind": "grids", "spec": old, "title": "Grids · TNG50-4-Dark · 099 · PS-DTFE "
                                                                          "512³ · sampled, nSub 2, GPU"})
    new = rs.QueuedJob.of("grids", rs.RunSpec(sim="TNG50-4-Dark", snapshots=[99]))
    q_new = rs.QueuedJob.from_dict(dict(new.to_dict(), title="kept as saved"))
    check("a queue entry saved before the two modes gets its title made again from the migrated settings (the "
          "quality's word, no removed setting); an entry saved in the two modes keeps its title as saved",
          q_old.title == rs.job_summary("grids", q_old.build()) and q_old.title.endswith(", Fast")
          and "nSub" not in q_old.title and q_new.title == "kept as saved", q_old.title)
    inp = str(tmp / "fin_in" / "toy.hdf5")
    lin = [m for lvl, m in rs.CustomSpec(input_file=inp, extra_args="--ps-linear-deposit").problems()
           if lvl == "error" and "--ps-linear-deposit" in m]
    lin_std = [m for lvl, m in rs.CustomSpec(input_file=inp, estimator="dtfe",
                                             extra_args="--ps-linear-deposit").problems() if "volume-weighted" in m]
    check("Custom: '--ps-linear-deposit' among the extra options is an error in the checks (the tab always passes "
          "'--ps-volume-weighted' for PS-DTFE, and the program refuses the pair at the start); not for standard DTFE",
          len(lin) == 1 and "--ps-volume-weighted" in lin[0] and not lin_std
          and "--ps-volume-weighted" in rs.CustomSpec(input_file=inp).command(), str(lin))
    with fake_stamps(lambda est, prec, dim: [".gpu_mode_metal" if dim == 3 else ".gpu_mode_off"]):   # no TBB
        no_tbb = (rs.advanced_choices(run({}), "parallel_triangulation"),
                  rs.automatic_text(run({}), "parallel_triangulation"),
                  rs.advanced_rows(run({})), rs.advanced_choices(run({"parallel_triangulation": True}),
                                                                "parallel_triangulation"))
    with fake_build("gpu"):
        tbb = (rs.advanced_choices(run({}), "parallel_triangulation"),
               rs.automatic_text(run({}), "parallel_triangulation"))
    check("the parallel insertion: 'On' is offered only on a build with TBB, and without TBB its Automatic note says "
          "the build has none (the row stays: a held value can be set back)",
          no_tbb[0] == [(None, "Automatic")] and no_tbb[1] == "Automatic: the sequential insert (this build has no TBB)"
          and "parallel_triangulation" in no_tbb[2] and no_tbb[3] == [(None, "Automatic")]
          and tbb == ([(None, "Automatic"), (True, "On")], "Automatic: the program's own rule"), str((no_tbb, tbb)))
    prod = ["density_a", "velocity_a", "divergence_a", "shear_a"]
    web = prod + ["tweb_a", "vweb_a"]
    with fake_build("gpu"):
        big = rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[1], fields=web)
        scratch = [m for lvl, m in big.problems() if "set a scratch directory" in m]
        gb = big.grid_gb()
        bigger = [m for lvl, m in rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[1], fields=web,
                                             grid=640).problems() if "set a scratch directory" in m]
    check("the full-resolution grids are the most a run holds at once: a 512³ PS-DTFE Exact job with the default "
          "fields and both web classes books the face sums' phase, not the face sums and the web classes together "
          "(about 28 GB, no scratch folder asked for); a larger grid still is",
          not scratch and abs(gb - rs.grids_gb(512, web, face=True)) < 1e-12
          and abs(gb - rs.grids_gb(512, prod, face=True)) < 1e-12 and 25 < gb < 31 and len(bigger) == 1,
          f"{gb:.2f} {scratch} {bigger}")


def two_quality_modes(tmp: Path):
    """The two quality modes (the owner's 'one faster but less correct, one slower but exact for each method'): one
    field, quality, on the Grids, Pipeline and Custom jobs, Exact first and the default; what each implies decided in
    one place (runspec.quality_mode, read through each job's mode()); the golden rule -- an Exact job builds today's
    default command and environment, a Fast one today's sampled job at the removed controls' defaults but its count
    (one point per cell axis) -- for both
    estimators on every run tab, on a GPU build and a CPU-only one; jobs saved before the modes load into them."""
    import inspect
    import re
    print("launcher: the two quality modes (Exact, Fast):")
    specs = (rs.RunSpec, rs.PipelineSpec, rs.CustomSpec)
    fields = {cls.__name__: set(cls.__dataclass_fields__) for cls in specs}
    removed = set(rs.REMOVED_SETTINGS)
    check("one quality field on every run job: 'exact' (Exact, first and the default) or 'fast' (Fast); none of the "
          "settings it replaced is left (a Custom job keeps its precision and initial positions, a Pipeline job its "
          "pixel sub-samples)",
          rs.QUALITIES == ("exact", "fast") and rs.QUALITY_DEFAULT == rs.QUALITIES[0]
          and rs.QUALITY_NAME == {"exact": "Exact", "fast": "Fast"}
          and all("quality" in f for f in fields.values()) and all(cls().quality == "exact" for cls in specs)
          and not fields["RunSpec"] & removed and not fields["PipelineSpec"] & removed
          and fields["CustomSpec"] & removed == {"precision", "vertex_mass"} and "supersample" in fields["PipelineSpec"]
          and rs.QueuedJob.of("grids", rs.RunSpec(quality="fast")).build().quality == "fast", str(fields))
    texts = {"ps": {"exact": "Each tetrahedron's exact share of each cell; exact derivatives of the mean flow.",
                    "fast": "One sample per cell: quick, with noisy per-cell values (smooth before use)."},
             "dtfe": {"exact": "The exact cell average.",
                      "fast": "A Monte-Carlo mean of the cell; derivatives stay exact."}}
    check("each quality's words per estimator are the fixed texts (no measured number, no '--'); the status line's "
          "note for an older saved state says Exact and Fast, no number",
          rs.QUALITY_TEXT == texts
          and not any(re.search(r"\d|--", t) for d in rs.QUALITY_TEXT.values() for t in d.values())
          and "Exact" in rs.MODES_NOTE and "Fast" in rs.MODES_NOTE and not re.search(r"\d|--", rs.MODES_NOTE))
    snap = tmp / "TNG100" / "TNG100-3-Dark" / "snapdir_099" / "combined_099.hdf5"
    bad = (rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], quality="bogus"),
           rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], quality="bogus"),
           rs.CustomSpec(input_file=str(snap), quality="bogus"))
    check("an unknown quality is an error on every run tab (and builds as Exact meanwhile)",
          all(("error", "Quality: unknown value bogus (exact or fast)") in j.problems() for j in bad)
          and all(j.mode().deposit == "exact" for j in bad))

    # ---- the golden rule, self-contained: the commands and environments of today's default job (Exact) and of
    # today's job with the sampled cell average at the removed controls' defaults (Fast; its count one point per cell
    # axis, the program's default since 2026-11-14), copied from the launcher before the two modes, for a made-up
    # simulation on a GPU build (Metal) and a CPU-only one
    root = str(tmp / "golden")
    ps_common = {"DTFE_DATA_ROOT": root, "DTFE_FACE_GRADIENT": "", "DTFE_RESUME": "1", "DTFE_RUN_MODE": "balanced",
                 "FIELDS": "density_a velocity_a divergence_a shear_a", "GPU_SHARE": "auto", "ORDERED_MERGE": "1",
                 "OUTPUT_PREFIX": "ps_output", "PARALLEL_TRIANGULATION": "0", "PS_CAUSTICS": "1",
                 "PS_CAUSTIC_CUSPS": "0", "PS_LINEAR_DEPOSIT": "0", "PS_MERGE": "index-order", "PS_PARALLEL_TRI": "0",
                 "PS_RESUME": "1", "PS_VERTEX_MASS": "1", "PS_VOLUME_WEIGHTED": "1"}
    gpu_env = lambda on: {"DTFE_GPU": on, "PS_GPU": on, "PS_METAL": on}                   # noqa: E731
    ps_exact = dict(ps_common, SUBSAMPLES="", AVG_SUBSAMPLES="", DTFE_EXACT_AVERAGE="1", PS_EXACT="1",
                    FACE_GRADIENT="1")
    ps_fast = dict(ps_common, DTFE_EXACT_AVERAGE="0", PS_EXACT="0", FACE_GRADIENT="")
    dt_common = {"DTFE_DATA_ROOT": root, "DTFE_FACE_GRADIENT": "", "DTFE_RESUME": "1", "FACE_GRADIENT": "",
                 "SUBSAMPLES": "", "THREADS": ""}
    dt_env = lambda q, on, mode: dict(dt_common, DTFE_EXACT=q, DTFE_EXACT_AVERAGE=q,             # noqa: E731
                                      DTFE_GPU=on, DTFE_METAL=on, DTFE_RUN_MODE=mode)
    pipe_common = {"AXIS": "z", "GRID_SIZE": "512", "NU": "8192", "PLANES": "1", "SUPERSAMPLE": "1", "SIMS": "Toy-64",
                   "PTS_VEL_GRAD": "1", "PS_POINT_CONVENTIONS": "grid", "PTS_CONVENTIONS": "grid",
                   "PYTHON": rs.PYTHON, "PY": rs.PYTHON}
    pipe_exact = dict(ps_exact, **pipe_common)
    pipe_exact.pop("FIELDS")
    pipe_exact.pop("OUTPUT_PREFIX")
    pipe_fast = dict(ps_fast, **pipe_common, SUBSAMPLES="1", AVG_SUBSAMPLES="1")
    pipe_fast.pop("FIELDS")
    pipe_fast.pop("OUTPUT_PREFIX")
    ps, dt = str(rs.PS_SCRIPT), str(rs.DTFE_SCRIPT)
    out = tmp / "golden_out"
    inp = str(tmp / "golden_in" / "toy.hdf5")
    custom_head = lambda b, mode="balanced", f="density_a velocity_a dispersion_a": (      # noqa: E731
        [str(rs.REPO_ROOT / b), inp, str(out / "toy"), "--grid", "128", "--input-format", "105", "--mpc-unit", "1",
         "--run-mode", mode, "--periodic", "--field", *f.split()])
    std_f = "density_a velocity_a"
    bad_cases = []
    for kind, on in (("gpu", "1"), ("cpu", "0")):
        m = ["-m"] if kind == "gpu" else []
        g = ["--gpu"] if kind == "gpu" else []
        with fake_build(kind):
            run = lambda **k: rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[3, 1], **k).command()  # noqa: E731
            pipe = lambda **k: rs.PipelineSpec(data_root=root, sims=["Toy-64"], **k).command()           # noqa: E731
            cust = lambda **k: rs.CustomSpec(input_file=inp, output_dir=str(out), **k)                   # noqa: E731
            cases = {
                "Grids PS Exact": (run(), ([ps, "-s", "Toy-64", "-g", "512", *m, "-e", "1", "3"],
                                           dict(ps_exact, **gpu_env(on)))),
                "Grids PS Fast": (run(quality="fast"), ([ps, "-s", "Toy-64", "-g", "512", "-n", "1", *m, "1", "3"],
                                                        dict(ps_fast, **gpu_env(on)))),
                "Grids DTFE Exact": (run(estimator="dtfe"), ([dt, "-s", "Toy-64", "-g", "512", *m, "-e", "1", "3"],
                                                             dt_env("1", on, "balanced"))),
                "Grids DTFE Fast": (run(estimator="dtfe", quality="fast"),
                                    ([dt, "-s", "Toy-64", "-g", "512", *m, "1", "3"], dt_env("0", on, "balanced"))),
                "Grids DTFE Fast, performance": (run(estimator="dtfe", quality="fast", run_mode="performance"),
                                                 ([dt, "-s", "Toy-64", "-g", "512", "1", "3"],
                                                  dt_env("0", "0", "performance"))),
                "Pipeline Exact": (pipe(), ([str(rs.PIPELINE_SCRIPT)], dict(pipe_exact, **gpu_env(on)))),
                "Pipeline Fast": (pipe(quality="fast"), ([str(rs.PIPELINE_SCRIPT)], dict(pipe_fast, **gpu_env(on)))),
                "Custom PS Exact": ((cust().command(), cust().run_env()),
                                    (custom_head("PS-DTFE") + ["--exact", *g, "--ps-volume-weighted", "--face-gradient"]
                                     + (["--gpu-share", "auto"] if g else []) + ["--ps-merge", "index-order"],
                                     {"DTFE_FACE_GRADIENT": ""})),
                "Custom PS Fast": ((cust(quality="fast").command(), cust(quality="fast").run_env()),
                                   (custom_head("PS-DTFE") + ["--sampled", "--subsamples", "1", *g,
                                                              "--ps-volume-weighted", "--ps-merge", "index-order"],
                                    {"DTFE_FACE_GRADIENT": ""})),
                "Custom DTFE Exact, performance": (
                    (cust(estimator="dtfe", run_mode="performance").command(), {"DTFE_FACE_GRADIENT": ""}),
                    (custom_head("DTFE", "performance", std_f) + [*g, "--exact", "--padding", "15"],
                     {"DTFE_FACE_GRADIENT": ""})),
                "Custom DTFE Fast, performance": (
                    (cust(estimator="dtfe", quality="fast", run_mode="performance").command(),
                     cust(estimator="dtfe", quality="fast", run_mode="performance").run_env()),
                    (custom_head("DTFE", "performance", std_f) + ["--sampled", "--padding", "15"],
                     {"DTFE_FACE_GRADIENT": ""})),
                "Custom PS 2D Exact": ((cust(dim=2).command(), cust(dim=2).run_env()),
                                       (custom_head("PS-DTFE-2d") + ["--exact", "--ps-volume-weighted", "--ps-merge",
                                                                     "index-order"], {"DTFE_FACE_GRADIENT": ""})),
            }
        bad_cases += [f"{kind} build, {name}: {got} != {want}" for name, (got, want) in cases.items()
                      if (list(got[0]), dict(got[1])) != want]
    check("the golden rule: Exact builds today's default command and environment, Fast today's sampled job at the "
          "removed controls' defaults but its count (one point per cell axis: -n 1 / --subsamples 1; the volume "
          "weighting, the index-order merge, the "
          "sequential insertion, no linear deposit, the face route where Exact qualifies, the drivers' device), for "
          "both estimators on the Grids, Pipeline and Custom tabs, on a GPU build and a CPU-only one",
          not bad_cases, " || ".join(bad_cases[:2]))

    # ---- one place: quality_mode is what every builder reads (through mode()); none names a removed setting
    builders = [rs.RunSpec.command, rs.RunSpec.problems, rs.RunSpec.grid_gb, rs.PipelineSpec.command,
                rs.PipelineSpec.check_step, rs.PipelineSpec.problems, rs.CustomSpec._own_command, rs.CustomSpec.run_env,
                rs.CustomSpec.problems, rs.ps_run_env, rs.ps_exact_env, rs.face_gradient_applies, rs.face_gradient_env,
                rs.face_route_on, rs._quality_problems, rs._ps_option_problems, rs._run_option_problems,
                rs._run_option_env, rs.job_summary, rs.QueuedJob.from_dict]
    named = re.compile(r"\b(?:self|spec|sp|c)\.(" + "|".join(sorted(removed - {"precision", "vertex_mass"})) + r")\b")
    tng_own = re.compile(r"\b(?:self|spec)\.(precision|vertex_mass)\b")
    compares = re.compile(r"quality\s*[!=]=|[\"']fast[\"']|[\"']sampled[\"']\s*(?:if|else)")
    leaks = {f.__qualname__: hits for f in builders
             for hits in [named.findall(inspect.getsource(f))
                          + (tng_own.findall(inspect.getsource(f)) if "CustomSpec" not in f.__qualname__ else [])
                          + compares.findall(inspect.getsource(f))] if hits}
    mode_src = inspect.getsource(rs.quality_mode)
    check("one place says what a quality means: no command, environment, memory or check builder names a removed "
          "setting or compares the quality (they read mode(), runspec.quality_mode's answer); quality_mode itself "
          "maps Fast to the sampled cell average",
          not leaks and '"sampled" if quality == "fast" else "exact"' in mode_src
          and all("mode()" in inspect.getsource(f) for f in (rs.RunSpec.command, rs.PipelineSpec.command,
                                                             rs.CustomSpec._own_command, rs.ps_run_env)), str(leaks))
    stub = rs.QualityMode(deposit="sampled", nsub=5, gpu=True, face_gradient=True, volume_weighted=False,
                          vertex_mass=False, linear_deposit=True, parallel_triangulation=True, ordered_merge=False,
                          pts_conventions="legacy")
    real = rs.quality_mode
    rs.quality_mode = lambda *a, **k: stub
    try:
        argv, env = rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[1], slice_plane="/p.bin").command()
        penv = rs.PipelineSpec(data_root=root, sims=["Toy-64"]).command()[1]
        carg = rs.CustomSpec(input_file=inp, sample_points="/p.bin").command()
        cvm = rs.CustomSpec(input_file=inp, vertex_mass=True).command()
        dargv = rs.RunSpec(data_root=root, sim="Toy-64", snapshots=[1], estimator="dtfe").command()[0]
    finally:
        rs.quality_mode = real
    check("... and changing quality_mode alone changes every job: a stand-in mode (sampled at 5, the GPU, no volume "
          "weighting, the linear deposit, the parallel insertion, the completion-order merge, legacy planes) reaches "
          "the Grids, Pipeline and Custom commands and environments of an 'exact' job; a Custom job's initial "
          "positions stay its own",
          argv[-4:] == ["-n", "5", "-m", "1"] and "-e" not in argv and env["PS_VOLUME_WEIGHTED"] == "0"
          and env["PS_VERTEX_MASS"] == "0" and env["PS_LINEAR_DEPOSIT"] == "1" and env["PARALLEL_TRIANGULATION"] == "1"
          and env["PS_MERGE"] == "completion-order" and env["FACE_GRADIENT"] == "" and env["DTFE_GPU"] == "1"
          and env["PS_POINT_CONVENTIONS"] == "legacy" and env["DTFE_EXACT_AVERAGE"] == "0"
          and penv["SUBSAMPLES"] == "5" and penv["PS_GPU"] == "1" and penv["PS_MERGE"] == "completion-order"
          and carg[carg.index("--sampled"):carg.index("--sampled") + 4] == ["--sampled", "--subsamples", "5", "--gpu"]
          and "--ps-volume-weighted" not in carg and "--parallel-triangulation" in carg
          and "--ps-linear-deposit" in carg and carg[carg.index("--ps-merge") + 1] == "completion-order"
          and carg[carg.index("--ps-point-conventions") + 1] == "legacy" and "--gpu-share" not in carg
          and "--ps-vertex-mass" not in carg and "--ps-vertex-mass" in cvm
          and dargv == [str(rs.DTFE_SCRIPT), "-s", "Toy-64", "-g", "512", "-m", "1"],
          f"{argv} {env} {carg}")

    # ---- a launcher state saved before the two modes: the window's restore reads each job through from_dict
    # (app._restore_specs) and says so once on the status line (rs.predates_modes); the demo is an Exact run
    old_state = {"spec": dict(rs.RunSpec().to_dict(), deposit="sampled", gpu=True, nsub=2, precision="double"),
                 "pipeline": dict(rs.PipelineSpec().to_dict(), deposit="exact", gpu=False, ordered_merge=False),
                 "custom": dict(rs.CustomSpec().to_dict(), deposit="sampled", stream_density="geometric")}
    for d in old_state.values():
        d.pop("quality")
    restored = [rs.RunSpec.from_dict(old_state["spec"]), rs.PipelineSpec.from_dict(old_state["pipeline"]),
                rs.CustomSpec.from_dict(old_state["custom"])]
    check("an older launcher state: each job loads (its deposit as the quality, the removed settings dropped) and is "
          "noticed as older (one status-line note); a state saved now is not",
          [j.quality for j in restored] == ["fast", "exact", "fast"]
          and all(rs.predates_modes(d) for d in old_state.values())
          and not any(rs.predates_modes(j.to_dict()) for j in restored)
          and not rs.predates_modes({}) and not rs.predates_modes(None) and not rs.predates_modes("x")
          and rs.predates_modes({"deposit": "exact"}) and not rs.predates_modes({"deposit": "exact", "quality": "fast"}),
          str([j.quality for j in restored]))
    # ... a queue entry too (its job is its 'spec'): a queue saved before the modes gets the note as well
    q_old = rs.QueuedJob.of("grids", rs.RunSpec()).to_dict()
    q_old["spec"] = dict(old_state["spec"])
    q_new = rs.QueuedJob.from_dict(q_old).to_dict()
    check("a queue entry saved before the two modes is noticed as older (the status-line note), the same entry "
          "loaded and saved now is not; the note names the one removed setting that changes the result, the TNG "
          "tabs' precision",
          rs.predates_modes(q_old) and not rs.predates_modes(q_new)
          and not rs.predates_modes(rs.QueuedJob.of("grids", rs.RunSpec()).to_dict())
          and not rs.predates_modes({"kind": "grids", "spec": "x"}) and "single precision" in rs.MODES_NOTE,
          f"{q_old} {q_new}")
    demo = rs.demo_spec(tmp / "demo_modes")
    check("the first-run demo is an Exact PS-DTFE run (its device the quality's); demo_spec takes no GPU argument",
          demo.quality == "exact" and "--exact" in demo.command() and list(inspect.signature(rs.demo_spec).parameters)
          == ["output_dir"] and ("--gpu" in demo.command()) == demo.mode().gpu, str(demo.command()))


def driver_device_table():
    """The device of a job's quality, rs.recommended_gpu, against the drivers' own automatic device (scripts/config.sh
    gpu_auto_resolve, the run modes' device table D21): for every estimator, deposit, machine use and GPU stamp of the
    pair's object folder (metal, off, none) they equal what the driver's DRY_RUN plans (its command's --gpu). The
    drivers run copied into stand-in repositories with stand-in binaries, as tests/run_scripts_check.sh does, one per
    stamp; rs._objdir points at the same object folders."""
    import re as _re
    print("launcher: the drivers' automatic device:")
    real = rs._objdir
    base = Path(tempfile.mkdtemp(prefix="gui_devices_"))
    procs = {}
    try:
        data = base / "data"
        (data / "SIM" / "snapdir_099").mkdir(parents=True)
        (data / "SIM" / "snapdir_099" / "combined_099.hdf5").touch()
        (data / "SIM" / "combined_ics.hdf5").touch()          # run_ps_dtfe.sh: the Lagrangian positions are there
        # the knobs of the login shell out: every DTFE_* and PS_* (the GPU, the deposit, the mode, the precision ...),
        # the counts, the share and a batch job's id (a SLURM or PBS job means performance)
        keep = {k: v for k, v in os.environ.items() if not k.startswith(("DTFE_", "PS_")) and k not in (
            "SUBSAMPLES", "AVG_SUBSAMPLES", "GPU_SHARE", "SLURM_JOB_ID", "PBS_JOBID", "FIELDS", "OUTPUT_PREFIX",
            "PARTITION", "MAX_CONCURRENT", "THREADS", "SAMPLE_POINTS")}
        for stamp in ("metal", "off", "none"):
            repo = base / stamp
            (repo / "scripts").mkdir(parents=True)
            for name in ("run_dtfe.sh", "run_ps_dtfe.sh", "config.sh"):
                shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
            for b in ("DTFE", "PS-DTFE"):
                (repo / b).write_text("#!/usr/bin/env bash\nexit 0\n")
                (repo / b).chmod(0o755)
            if stamp != "none":
                for obj in ("o", "o_ps"):
                    (repo / obj).mkdir()
                    (repo / obj / f".gpu_mode_{stamp}").touch()
            for est, script in (("ps", "run_ps_dtfe.sh"), ("dtfe", "run_dtfe.sh")):
                for dep in ("exact", "sampled"):
                    for mode in rs.RUN_MODES:
                        env = dict(keep, DTFE_DATA_ROOT=str(data), DRY_RUN="1", DTFE_RUN_MODE=mode,
                                   DTFE_EXACT_AVERAGE="1" if dep == "exact" else "0")
                        procs[(stamp, est, dep, mode)] = subprocess.Popen(
                            [str(repo / "scripts" / script), "-s", "SIM", "-g", "16", "99"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wrong = []
        for (stamp, est, dep, mode), p in procs.items():
            out, _ = p.communicate(timeout=120)
            line = next((ln for ln in out.splitlines() if _re.match(r"^ +\./(PS-)?DTFE\s", ln)), None)
            planned = None if line is None else bool(_re.search(r"\s--(ps-)?gpu(\s|$)", line))
            rs._objdir = (lambda estimator, precision="single", dim=3, repo=base / stamp:
                          repo / real(estimator, precision, dim).name)
            quality = "exact" if dep == "exact" else "fast"
            gui = (rs.recommended_gpu(dep, ["density_a"], None, est, run_mode=mode),
                   rs.RunSpec(estimator=est, quality=quality, run_mode=mode).mode().gpu)
            if planned is None or gui != (planned, planned):
                wrong.append(f"{stamp} {est} {dep} {mode}: driver {planned} (exit {p.returncode}"
                             + ("" if line else f": {out.strip().splitlines()[-1:] or ''}") + f"), launcher {gui}")
        check("a job's device (its quality's) and the recommended GPU are the drivers' DRY_RUN device for every "
              "estimator, deposit, machine use and stamp (metal, off, none)", not wrong, "; ".join(wrong[:4]))
    finally:
        rs._objdir = real
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        shutil.rmtree(base, ignore_errors=True)


def tick_helpers():
    """The Explore map's axes and colour bar: round ticks, their labels, the scale bar."""
    import numpy as np
    import grids as G
    t = G.nice_ticks(0, 51.67)
    check("axis ticks: round values inside the range, about five (0 .. 51.7 Mpc: every 10)", t == [0, 10, 20, 30, 40, 50], str(t))
    t = G.nice_ticks(-6.46, 6.46)
    check("... a 2.5 step keeps its decimal ('2' for 2.5 would be wrong)",
          [G.tick_text(v, t) for v in t] == ["-5.0", "-2.5", "0.0", "2.5", "5.0"], str(t))
    check("... a degenerate range has none", G.nice_ticks(1, 1) == [] and G.nice_ticks(0, float("nan")) == [])
    check("... an axis in cells keeps to whole steps (a 16-cell grid: 0, 5, 10, 15, not 2.5 apart; 4 cells: every one)",
          G.nice_ticks(0, 16, integer=True) == [0, 5, 10, 15] and G.nice_ticks(0, 4, integer=True) == [0, 1, 2, 3, 4]
          and G.nice_ticks(0, 16) == [0, 2.5, 5, 7.5, 10, 12.5, 15], str(G.nice_ticks(0, 16)))
    t = G.nice_ticks(120, 120.004)
    check("... a deep zoom far from the origin keeps the labels apart (not five '120')",
          [G.tick_text(v, t) for v in t] == ["120.000", "120.001", "120.002", "120.003", "120.004"], str(t))
    z = np.zeros((64, 64))
    z[3, 4], z[10, 10] = 0.1, -0.05                 # two runs that agree in all but two cells
    check("a 0-centred range of a difference that is 0 almost everywhere: its largest value, not (0, 1) "
          "(which painted the whole map the scale's end)",
          G.value_range(z, symmetric=True) == (-0.1, 0.1) and G.value_range(np.zeros((4, 4)), symmetric=True) == (-1.0, 1.0),
          str(G.value_range(z, symmetric=True)))
    check("the scale bar: a round length near a fifth of the width",
          [G.nice_length(s) for s in (110.7, 51.6, 6.5)] == [20.0, 10.0, 1.0])
    dec = G.colorbar_ticks(-1.2, 3.7, log=True)
    check("a log colour bar over several decades: 10^k at the whole decades",
          [tx for _, tx in dec] == ["10⁻¹", "1", "10", "10²", "10³"] and [v for v, _ in dec] == [-1, 0, 1, 2, 3], str(dec))
    few = G.colorbar_ticks(np.log10(0.0249), np.log10(7.58), log=True)
    check("... under three decades: 1, 2, 5 x 10^k as numbers",
          [tx for _, tx in few] == ["0.05", "0.1", "0.2", "0.5", "1", "2", "5"], str(few))


def core_grids_warning():
    """The Core analysis set warns when a snapshot has no standard-DTFE grids in the modes
    that READ them -- all, compute and export (analyze.export builds its FieldSet from the grids, like compute) --
    and not in plot / check (they read python/cache). Phase-space grids do not count: compute reads the standard
    DTFE. Its own data root: an empty snapdir_099, then PS grids only, then a standard-DTFE run. The estimator and
    grid prefix the user set for the OTHER sets do not count either: the core step passes analyze.py neither."""
    root = Path(tempfile.mkdtemp(prefix="gui_core_"))
    try:
        sd = root / "TNG100" / "TNG100-3-Dark" / "snapdir_099"
        sd.mkdir(parents=True)
        (sd.parent / "snapdir_050").mkdir()              # stays empty: the two-snapshot spec names it every time

        def warned(snaps=(99,), **kw):
            got, errs, msgs = set(), [], set()
            for mode in ("all", "compute", "export", "plot", "check"):
                spec = rs.PlotSpec(data_root=str(root), sim="TNG100-3-Dark", snapshots=list(snaps), sets=["core"],
                                   options={"core": {"mode": mode}}, **kw)
                probs = spec.problems()
                errs += [m for lvl, m in probs if lvl == "error"]      # an error returns early: the warning never runs
                hit = [m for lvl, m in probs if lvl == "warning" and "no standard-DTFE grids for" in m]
                if [m for m in hit if "no standard-DTFE grids for 099" in m or len(snaps) > 1]:
                    got.add(mode)
                msgs.update(hit)
            return got, errs, msgs

        other = dict(method="ps", prefix="ps_output")    # the PS-DTFE choice for the other sets, on a PS run that exists
        empty, e0, _ = warned()
        two_empty, e3, m3 = warned((50, 99))
        (sd / "ps_output.a_den").touch()
        ps_only, e1, _ = warned()
        ps_other, e4, _ = warned(**other)
        (sd / "output.a_den").touch()
        std, e2, _ = warned()
        std_other, e5, _ = warned(**other)
        two_std, e6, m6 = warned((50, 99))
        check("core set: no standard-DTFE grids for 099 is warned in modes all, compute AND export, not in plot / check "
              "(they read the cache)", empty == {"all", "compute", "export"} and not e0, f"{sorted(empty)} {e0}")
        check("... phase-space grids alone do not count (compute and export read the standard DTFE)",
              ps_only == {"all", "compute", "export"} and not e1, f"{sorted(ps_only)} {e1}")
        check("... and a standard-DTFE run silences it in every mode", std == set() and not e2, f"{sorted(std)} {e2}")
        # analyze.py gets no --method / --prefix: a PS estimator and prefix set for the other sets (that run exists
        # here) must not silence it, nor make the standard-DTFE run count as missing
        check("... the estimator and grid prefix set for the other sets change nothing (analyze.py takes neither): "
              "PS-DTFE / ps_output still warned, the standard-DTFE run still silences it",
              ps_other == {"all", "compute", "export"} and std_other == set() and not (e4 + e5),
              f"{sorted(ps_other)} {sorted(std_other)} {e4 + e5}")
        # every snapshot without them is named, and only those: 050 stays empty, 099 gets its run in between
        check("... two snapshots: the warning names every one without them ('050 099'), then only 050 once 099 has its run",
              two_empty == two_std == {"all", "compute", "export"} and not (e3 + e6)
              and len(m3) == 1 and next(iter(m3)).endswith("grids for 050 099")
              and len(m6) == 1 and next(iter(m6)).endswith("grids for 050"),
              f"{sorted(two_empty)} {m3} {sorted(two_std)} {m6} {e3 + e6}")
    finally:
        shutil.rmtree(root)


def _write_log(path: Path, text: str, mtime: float | None = None):
    before = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    elif before is not None and path.stat().st_mtime_ns == before:
        # a coarse file clock (Linux: a tick of milliseconds) gives a quick same-size rewrite the old mtime, and
        # results.parse_runlog's (size, mtime_ns) cache would hand back the previous log's record
        os.utime(path, ns=(before + 1000, before + 1000))


def new_features(tmp: Path, sim: Path):
    import datetime as dt
    import json

    import numpy as np
    import grids as G
    import presets as P
    import results as R

    work, snap, h5py, c, steps = _custom_snapshot(tmp, np, P)
    _export(work, steps)
    snapdir = _run_logs(tmp, sim, dt, R)
    _presets_help(sim, P, R)
    _explore_data(tmp, json, np, G, work, snap, h5py, c, snapdir)


def _custom_snapshot(tmp, np, P):
    """CustomSpec: command, refusals, sidecar, memory check; the demo, the binaries' flags, GPU advice, precision."""
    print("custom snapshot (CustomSpec):")
    work = tmp.parent / "own"
    work.mkdir()
    snap = work / "cloud.hdf5"
    import h5py
    with h5py.File(snap, "w") as f:
        f.create_group("Header").attrs["BoxSize"] = 50000.0
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=np.zeros((8, 3)))
        g.create_dataset("Potential", data=np.zeros(8))
    c = rs.CustomSpec(input_file=str(snap), mpc_unit=1000.0, periodic=True, estimator="ps", grid=64,
                      fields=["density_a"], output_dir=str(work / "out"), output_name="run1", scalar_dataset="Potential")
    steps = c.steps()
    a = steps[-1].argv
    check("custom: the binary itself, units, periodic, fields, scalar, output root",
          Path(a[0]).name == "PS-DTFE" and a[1] == str(snap) and a[2] == str(work / "out" / "run1")
          and a[a.index("--mpc-unit") + 1] == "1000" and "--periodic" in a
          and a[a.index("--field") + 1:a.index("--field") + 3] == ["density_a", "scalar_a"]
          and a[a.index("--scalar-dataset") + 1] == "Potential", str(a))
    check("custom: a missing output folder gets its mkdir step first",
          steps[0].argv[:2] == ["mkdir", "-p"] and len(steps) == 2)
    check("custom: the run step tees its run log beside the outputs",
          steps[-1].tee == str(work / "out" / "run1.runlog")
          and steps[-1].line().endswith(f"2>&1 | tee {rs._q(steps[-1].tee)}"))
    check("custom: the repo binary is shown as ./PS-DTFE (never a PATH lookup)",
          " ./PS-DTFE " in " " + steps[-1].line() or steps[-1].line().startswith("./PS-DTFE"), steps[-1].line())
    lv, msg = levels(c)
    check("custom: a snapshot without InitialCoordinates is refused for PS-DTFE",
          "error" in lv and "InitialCoordinates" in msg, msg)
    c2 = rs.CustomSpec.from_dict({**c.to_dict(), "estimator": "dtfe", "scalar_dataset": ""})
    lv, msg = levels(c2)
    check("custom: ... but fine for standard DTFE (with the fixed field set)",
          "error" not in lv and rs.DTFE_DEFAULT_FIELDS[0] in c2.command(), msg)
    for bad, word in (({"input_type": 111}, "Gadget HDF5"), ({"box": [-1, 1, 0, 1, 0, 1]}, ">= 0"),
                      ({"box": [0, 1, 0]}, "6 numbers"), ({"input_file": str(work / 'nope.hdf5')}, "not found"),
                      ({"lagrangian": "separate", "lagrangian_file": ""}, "initial-conditions")):
        lv, msg = levels(rs.CustomSpec.from_dict({**c.to_dict(), **bad}))
        check(f"custom: refused -- {word}", "error" in lv and word in msg, msg)
    side = rs.CustomSpec.from_dict({**c.to_dict(), "box": []}).settings_sidecar()
    check("custom: the settings sidecar gives the box in Mpc (header BoxSize / MpcUnit)",
          side["box_mpc"] == [0.0, 50.0] * 3 and side["box"] == [], str(side["box_mpc"]))
    side = rs.CustomSpec.from_dict({**c.to_dict(), "box": [0, 20000, 0, 20000, 0, 40000]}).settings_sidecar()
    check("custom: ... and from --box (file units) when given", side["box_mpc"] == [0, 20, 0, 20, 0, 40])
    d = rs.demo_spec(work / "demo")
    ds = d.steps()
    check("demo: generate the snapshot, then run PS-DTFE on it (no download)",
          len(ds) == 3 and ds[1].argv[1].endswith("generate_ps_test_data.py") and "--crossed-waves" in ds[1].argv
          and Path(ds[2].argv[0]).name == "PS-DTFE" and "error" not in levels(d)[0], levels(d)[1])
    check("custom: to_dict/from_dict round trip", rs.CustomSpec.from_dict(c.to_dict()) == c)

    # ---- one flag scheme: the readers' permanent old-to-new table (dtfelib/runlog.py) holds every row the binaries
    # accept as a deprecated alias (user_options.cc deprecatedOptions, a subset during P6.7), the same table is
    # run_ps_pipeline.sh's old[] / sw[] maps and the docs check's; the GUI emits none of the binaries' deprecated names
    import re as _re
    import results as R
    from dtfelib import runlog as RL
    src = (ROOT / "src" / "user_options.cc").read_text()
    table = src[src.index("deprecatedOptions[] = {"):]
    cpp = dict(_re.findall(r'\{ "([\w-]+)", +"([\w-]+)" \}', table[:table.index("};")]))
    cpp = {"--" + k: "--" + v for k, v in cpp.items()}
    pipe = (ROOT / "scripts" / "run_ps_pipeline.sh").read_text()
    awk = dict(_re.findall(r'old\["(--[\w-]+)"\] = "(--[\w-]+)"', pipe))
    awk_sw = {k: tuple(v.split()) for k, v in _re.findall(r'sw\["(--[\w-]+)"\] = "(--[\w-]+ [\w-]+)"', pipe)}
    sys.path.insert(0, str(ROOT / "tests"))
    import docs_drift_check as DD
    check("the deprecated spellings: the binaries' rows are the readers' (runlog), whose table is the launcher's, "
          "the pipeline's and the docs check's",
          len(cpp) >= 5 and all(RL.RENAMES.get(k) == v for k, v in cpp.items()) and cpp == DD.DEPRECATED
          and R.DEPRECATED_FLAGS is RL.RENAMES and RL.RENAMES == awk and RL.SWITCHES == awk_sw,
          f"{cpp} {set(RL.RENAMES.items()) ^ set(awk.items())} {awk_sw}")
    with fake_build("gpu"):         # (Exact's device on a GPU build: '--gpu')
        cxa = rs.CustomSpec(input_file=str(snap), estimator="ps", fields=["density_a", "tweb_a"]).command()
    check("the Custom tab emits the canonical names (--exact, --gpu, --lambda-th), none the binaries deprecate",
          "--exact" in cxa and "--gpu" in cxa and cxa[cxa.index("--lambda-th") + 1] == "0.3"
          and not set(cxa) & set(cpp), str(cxa))
    old = R.RunRecord(path=Path("o.runlog"), binary="PS-DTFE", command="./PS-DTFE in o --grid 64 --avg-subsamples 2 "
                      "--lambda_th 0.3 --ps-gpu --ps-exact-deposit --ps-caustics")
    mid = R.RunRecord(path=Path("m.runlog"), binary="PS-DTFE", command="./PS-DTFE in m --grid 64 --avg-subsamples 2 "
                      "--lambda-th 0.3 --gpu --exact-average --ps-caustics")
    new = R.RunRecord(path=Path("n.runlog"), binary="PS-DTFE", command="./PS-DTFE in n --grid 64 --subsamples 2 "
                      "--lambda-th 0.3 --gpu --exact --ps-caustics")
    check("a run log with the deprecated spellings reads as one with the canonical names (settings, options, GPU; "
          "both older names of the exact deposit, '--ps-exact-deposit' and '--exact-average', are '--exact')",
          old.settings() == mid.settings() == new.settings()
          and old.options_text() == mid.options_text() == new.options_text() == "Exact, caustics, GPU"
          and old.exact and mid.exact and new.exact and old.has("--exact") and mid.has("--ps-exact-deposit")
          and new.has("--exact-average") and old.gpu and old.nsub == 2
          and R.canonical_command("--lambda_th=0.3 /a/--ps-gpu") == "--lambda-th=0.3 /a/--ps-gpu",
          f"{old.settings()} {old.options_text()}")
    # P6.7: every rename of the second pass, with its value transforms
    o7 = R.RunRecord(path=Path("o7.runlog"), binary="PS-DTFE", command=(
        "./PS-DTFE /a b/in.hdf5 o7 --input 105 --MpcUnit 1000 --grid 64 --avg-subsamples 3 --periodic "
        "--lagrangianInput /ics x.hdf5 --ps-lag-padding 8 --ps-window 0 10 0 10 0 10 --sample-points /p.txt "
        "--pts-conventions legacy --per-stream --pts-lagrangian --ps-stream-density dtfe --completion-order-merge "
        "--partition 2 --partNo 3"))
    n7 = R.RunRecord(path=Path("n7.runlog"), binary="PS-DTFE", command=(
        "./PS-DTFE /a b/in.hdf5 n7 --input-format 105 --mpc-unit 1000 --grid 64 --subsamples 3 --periodic "
        "--ps-lagrangian-input /ics x.hdf5 --ps-lagrangian-padding 8 --grid-window 0 10 0 10 0 10 --sample-points "
        "/p.txt --ps-point-conventions legacy --ps-per-stream --ps-per-stream-lagrangian --ps-stream-density linear "
        "--ps-merge completion-order --partition 2 --partition-index 3"))
    check("P6.7: an old-spelling log reads as its new-spelling twin (settings, options, merge, legacy plane, nSub)",
          o7.settings() == n7.settings() and o7.options_text() == n7.options_text() and o7.merge == "completion-order"
          and o7.nsub == 3 and o7.legacy_points and not o7.settings()["ordered_merge"]
          and o7.has("--completion-order-merge") and n7.has("--completion-order-merge") and o7.has("--grid-window")
          and o7.canonical.replace(" o7 ", " n7 ") == n7.canonical,
          f"{o7.canonical} | {n7.canonical}")
    cc = RL.canonical_command
    check("P6.7 value transforms: standard --samples N = k^D -> --subsamples k (else kept), -m -> --average-method, "
          "the merge switches -> one --ps-merge (fixed-point wins), --ps-stream-density dtfe -> linear",
          cc("./DTFE in o -m 2 -s 27") == "./DTFE in o --average-method 2 --subsamples 3"
          and cc("./DTFE in o --method 3 --samples 20") == "./DTFE in o --average-method 3 --samples 20"
          and cc("./DTFE-2d in o -m2 -s9") == "./DTFE-2d in o --average-method 2 --subsamples 3"
          and cc("./DTFE in o --samples=8", dim=3) == "./DTFE in o --subsamples 2"
          and cc("x --ordered-merge --fixed-point-merge --grid 8") == "x --ps-merge fixed-point --grid 8"
          and cc("x --resume --completion-order-merge --ordered-merge") == "x --resume --ps-merge index-order"
          and cc("x --ps-stream-density=dtfe") == "x --ps-stream-density=linear"
          and cc("x --ps-stream-density geometric") == "x --ps-stream-density geometric"
          and RL.canonical_flag("--fixed-point-merge") == "--ps-merge fixed-point"
          and R.RunRecord(path=Path("m.runlog"), binary="PS-DTFE", command="./PS-DTFE i o --grid 8").merge == ""
          and R.RunRecord(path=Path("m.runlog"), binary="DTFE-2d", command="./DTFE-2d i o -s 16").canonical
          == "./DTFE-2d i o --subsamples 4",
          cc("./DTFE in o -m 2 -s 27"))
    rec = R.RunRecord(path=work / "ri.runlog", prefix="ri",
                      command=f"./PS-DTFE {work}/x.hdf5 {work}/ri --ps-lagrangian-input {work}/my ics.hdf5 --grid 8")
    (work / "my ics.hdf5").write_text("x")
    check("run_inputs: the Lagrangian input by its new name too (a path with a space)",
          (work / "my ics.hdf5").resolve() in R.run_inputs(rec), str(R.run_inputs(rec)))

    # ---- every binary flag the GUI can emit exists in the built binary's own --full-help, and
    # every script knob it sets is one the script reads: a renamed option can never ship silently
    import re as _re
    ps_help, dt_help = P.flag_help("PS-DTFE"), P.flag_help("DTFE")
    if ps_help and dt_help:
        emitted = set()
        for spec in (rs.CustomSpec(input_file=str(snap), estimator="ps", vertex_mass=True,
                                   caustics=True, caustic_cusps=True, scalar_dataset="Masses",
                                   scratch_dir=str(work), lagrangian="separate",
                                   lagrangian_file=str(snap), box=[0, 1, 0, 1, 0, 1],
                                   alpha_shape=2.5, periodic=False),
                     rs.CustomSpec(input_file=str(snap), estimator="ps", quality="fast", caustics=True),
                     rs.CustomSpec(input_file=str(snap), estimator="ps", sample_points=str(snap), per_stream=True,
                                   per_stream_ids=True, halo_release=500, window=[0, 1, 0, 1, 0, 1]),
                     rs.CustomSpec(input_file=str(snap), estimator="dtfe"),
                     rs.CustomSpec(input_file=str(snap), estimator="dtfe", quality="fast")):
            for kind in ("gpu", "cpu"):     # (the device flags of both builds)
                with fake_build(kind):
                    emitted |= {a for a in spec.command() if a.startswith("--")}
        known = set(ps_help) | set(dt_help)
        unknown = sorted(emitted - known)
        check("every binary flag a custom-snapshot command can emit exists in the binaries' help",
              not unknown, str(unknown))
        script = (ROOT / "scripts" / "run_ps_dtfe.sh").read_text()
        knobs = set(_re.findall(r'^([A-Z_]+)="\$\{\1:-', script, _re.M))
        # a knob's older names, read for one release through config.sh's resolver (P6.7):
        # 'knob DTFE_GPU PS_GPU PS_METAL', 'knob -e ...', 'knob_merge PS_MERGE ORDERED_MERGE FIXED_POINT_MERGE'
        for names in _re.findall(r'^knob\w*\s+(?:-\S+\s+)*([A-Z_]+(?:[ \t]+[A-Z_]+)*)', script, _re.M):
            knobs |= set(names.split())
        used = set().union(*(rs.RunSpec(sim="TNG50-4", snapshots=[99], slice_plane=str(snap), scratch_dir=str(work),
                                        quality=q).command()[1] for q in rs.QUALITIES)) - {"DTFE_DATA_ROOT"}
        knobs |= {"DTFE_FACE_GRADIENT"}          # (the gradient route's pin: the binary's own switch)
        check("every environment knob the Grids job sets is one run_ps_dtfe.sh reads (or the gradient route's pin, "
              "the binary's)", used <= knobs, str(sorted(used - knobs)))
        def script_knobs(*names):
            out = set()
            for name in names:
                text = (ROOT / "scripts" / name).read_text()
                out |= set(_re.findall(r'^\s*([A-Z_]+)="\$\{\1:?-', text, _re.M))
                for found in _re.findall(r'^\s*knob\w*\s+(?:-\S+\s+)*([A-Z_]+(?:[ \t]+[A-Z_]+)*)', text, _re.M):
                    out |= set(found.split())
            return out
        std_used = set().union(*(rs.RunSpec(sim="TNG50-4", snapshots=[99], estimator="dtfe", slice_plane=str(snap),
                                            scratch_dir=str(work), tess_cache=str(work), quality=q,
                                            fields=["density_a", "tweb_a"]).command()[1]
                                 for q in rs.QUALITIES)) - {"DTFE_DATA_ROOT"}
        std_knobs = script_knobs("run_dtfe.sh", "config.sh") | {"DTFE_FACE_GRADIENT"}
        check("... and every one a standard Grids job sets is one run_dtfe.sh reads (or the gradient route's pin: the "
              "binary's)", std_used <= std_knobs, str(sorted(std_used - std_knobs)))
        pipe_used = set().union(*(rs.PipelineSpec(sims=["TNG50-4"], window="0 10 0 10", planes=4, center="5",
                                                  tess_cache=str(work), scratch_dir=str(work), quality=q,
                                                  force=True, plan_only=True).command()[1]
                                  for q in rs.QUALITIES)) - {"DTFE_DATA_ROOT"}
        pipe_knobs = script_knobs("run_ps_pipeline.sh", "run_ps_dtfe.sh", "config.sh") | {
            "SIMS",                                                     # (SIMS: required, read with no default)
            "DTFE_FACE_GRADIENT"}                                       # (the gradient route's pin: the binary's)
        check("... and every one a Pipeline job sets is one run_ps_pipeline.sh or its run_ps_dtfe.sh reads (or the "
              "gradient route's pin)", pipe_used <= pipe_knobs, str(sorted(pipe_used - pipe_knobs)))
    else:
        check("flags-exist check skipped: a binary is not built", True)

    # ---- GPU advice: where the deposit pays, one warning per condition; asserted for BOTH builds by faking
    # the Makefile's GPU stamp (on CPU-only CI the GPU branches would otherwise never be asserted)
    gpu_branches(tmp, snap)
    # ---- precision: the double pair is a setting, picked per job; the knob and binary names
    both_double = rs.binary_built("ps", "double") and rs.binary_built("dtfe", "double")
    check("precision: a Grids or Pipeline job never sets DTFE_PRECISION (TNG runs single precision: the double pair "
          "is the Custom tab's), in either quality",
          not any("DTFE_PRECISION" in job.command()[1] for q in rs.QUALITIES
                  for job in (rs.RunSpec(sim="TNG50-4", snapshots=[99], quality=q),
                              rs.RunSpec(sim="TNG50-4", snapshots=[99], estimator="dtfe", quality=q),
                              rs.PipelineSpec(sims=["TNG50-4"], quality=q))))
    check("precision: binary names and the memory estimate (twice the bytes)",
          rs.binary_name("ps", "double") == "PS-DTFE-double" and rs.binary_name("dtfe") == "DTFE"
          and abs(rs.grids_gb(256, ["density_a"], "double") / rs.grids_gb(256, ["density_a"]) - 2) < 1e-9)
    one = lambda f: rs.grids_gb(100, f) * 1e9 / 100 ** 3     # bytes per cell
    check("the memory estimate books the velocity gradient once for every field derived from it",
          abs(one(["divergence_a"]) - (8 + 4 + 36)) < 1e-9 and abs(one(["divergence_a", "shear_a"]) - (8 + 24 + 36)) < 1e-9
          and abs(one(["divergence_a", "shear_a", "vorticity_a"]) - one(["divergence_a", "shear_a"]) - 12) < 1e-9
          and abs(one(["density_a", "velocity_a"]) - (8 + 16)) < 1e-9,
          f"{one(['divergence_a'])} {one(['divergence_a', 'shear_a'])} {one(['density_a', 'velocity_a'])}")
    # a PS-DTFE run that takes the face gradient keeps more full grids per cell: the face sums (30 floats), the stream
    # average's divergence and the mass flux divergence (2), the twin of each gradient (9), shear (5) and vorticity
    # (3) asked, and the mean velocity (3) where 'velocity_a' is not asked; nothing without a field built on the
    # gradient, nothing in 2D. A job passes it where its route is the face gradient (face_route_on). Not all at once
    # (src/auto_tune.cc, the planner's max(bCell, bCellDeposit + normR)): the face sums and that mean velocity go
    # before the derived fields and the web classes are made, beside which only the two outputs and the twins stay;
    # the deposited grids (density, velocity, the gradient, the counts) are held throughout
    face = lambda f, **k: rs.grids_gb(100, f, face=True, **k) * 1e9 / 100 ** 3     # noqa: E731
    prod = ["density_a", "velocity_a", "divergence_a", "shear_a"]
    every = ["velocity_a", "gradient_a", "divergence_a", "shear_a", "vorticity_a", "vweb_a"]
    web = ["tweb_a", "vweb_a"]
    fgrid = lambda s: s.grid_gb() == rs.grids_gb(s.grid, s.fields, face=True)   # noqa: E731
    pgrid = lambda s: s.grid_gb() == rs.grids_gb(s.grid, s.fields)              # noqa: E731
    std_face = rs.RunSpec(grid=100, estimator="dtfe")
    check("... and, for a PS-DTFE run that takes the face gradient, its face sums, its two face outputs and the "
          "twins of the fields asked ('.a_velDivStreams', '.a_massFluxDiv', '.a_velGradStreams', "
          "'.a_velShearStreams', '.a_velVortStreams'), as the larger of its two moments: the face sums with the "
          "deposit's grids, or the derived fields and web classes after them (both: the outputs and twins); the "
          "Grids job books them where its route is the face gradient (Exact), not for standard DTFE, Fast, the halo "
          "release or 2D; without the face gradient every grid counts at once (as before)",
          abs(face(prod) - (8 + 16 + 36 + max(120 + 8 + 20, 24 + 8 + 20))) < 1e-9
          and abs(face(["divergence_a"]) - (8 + 36 + max(120 + 8 + 12, 4 + 8))) < 1e-9
          and abs(face(every) - (8 + 12 + 36 + max(120 + 8 + 68, 52 + 8 + 68))) < 1e-9
          and abs(face(prod + web) - face(prod)) < 1e-9 and abs(one(prod + web) - one(prod) - 88) < 1e-9
          and abs(face(["divergence_a", "shear_a", "vorticity_a"] + web)
                  - (8 + 36 + max(120 + 8 + 12 + 32, 36 + 88 + 8 + 32))) < 1e-9
          and face(["density_a", "velocity_a"]) == one(["density_a", "velocity_a"])
          and face(prod, dim=2) == rs.grids_gb(100, prod, dim=2) * 1e9 / 100 ** 3
          and abs(face(prod, precision="double") - 2 * face(prod)) < 1e-9
          and fgrid(rs.RunSpec(grid=100)) and pgrid(rs.RunSpec(grid=100, quality="fast"))
          and std_face.grid_gb() == rs.grids_gb(100, rs.dtfe_fields(std_face.fields))
          and rs.face_route_on(rs.PipelineSpec())
          and not rs.face_route_on(rs.PipelineSpec(quality="fast"))
          and rs.face_route_on(rs.CustomSpec())
          and not rs.face_route_on(rs.CustomSpec(quality="fast"))
          and not rs.face_route_on(rs.CustomSpec(halo_release=2.0))
          and not rs.face_route_on(rs.CustomSpec(dim=2))
          and not rs.face_route_on(rs.CustomSpec(estimator="dtfe")),
          f"{face(prod)} {one(prod)} {face(every)} {one(every)}")
    cd = rs.CustomSpec(input_file=str(snap), estimator="dtfe", precision="double")
    check("precision: a double custom job runs the -double binary; missing pair = a clear error naming Setup",
          cd.binary().name == "DTFE-double"
          and (any("double-precision" in m and level == "error" for level, m in cd.problems()) != both_double))
    check("precision: an unknown value is an error", any("precision" in m for level, m in
          rs.CustomSpec(input_file=str(snap), precision="half").problems() if level == "error"))
    check("precision: the GPU and TBB stamps are read from the pair's own object directory",
          rs._objdir("ps", "double").name == "o_ps_d" and rs._objdir("dtfe").name == "o"
          and isinstance(rs.gpu_built("ps", "double"), bool) and isinstance(rs.tbb_built("double"), bool))

    # ---- the Custom tab's memory check: a report-mode call keyed on what changes memory
    c1 = rs.CustomSpec(input_file=str(snap), grid=256, output_dir=str(work), output_name="a")
    c2 = rs.CustomSpec(input_file=str(snap), grid=256, output_dir=str(work / "elsewhere"), output_name="b")
    c3 = rs.CustomSpec(input_file=str(snap), grid=512, output_dir=str(work), output_name="a")
    check("custom memory key: the output folder and name do not change it, the grid does",
          c1.memory_key() == c2.memory_key() and c1.memory_key() != c3.memory_key())
    st = c1.check_step()
    check("custom check step: the run's own command in report mode, named after the file",
          st.argv[:-1] == c1.command() and st.argv[-1] == "--auto-tune-report" and "cloud.hdf5" in st.label, st.label)
    # the standard binary answers the same report-mode call, here on a real (synthetic) snapshot: the fixture
    # above has no velocities
    import subprocess as sp
    std_in = work / "std_in.hdf5"
    sp.run([sys.executable, str(ROOT / "tests" / "generate_ps_test_data.py"), "--out", str(std_in), "--n", "16"],
           capture_output=True, check=False)
    cs = rs.CustomSpec(input_file=str(std_in), estimator="dtfe", grid=32, output_dir=str(work), output_name="std")
    sst = cs.check_step()
    check("custom check step, standard DTFE: the standard binary's own command in report mode",
          sst.argv[:-1] == cs.command() and sst.argv[-1] == "--auto-tune-report" and Path(sst.argv[0]).name == "DTFE")
    if Path(sst.argv[0]).is_file() and std_in.is_file():
        r = sp.run(sst.argv, env=dict(os.environ, **sst.env, DTFE_RAM_GB="64"), capture_output=True, text=True,
                   cwd=str(ROOT))
        rec = rs.parse_reports(r.stdout).get(0, {})
        check("... which reports a plan the launcher parses (partition, threads, prediction, budget)",
              r.returncode == 0 and rec.get("partition", 0) >= 1 and rec.get("mc", 0) >= 1
              and rec.get("predicted_gb", 0) > 0 and rec.get("budget_gb") == 38.4
              and not list(work.glob("std.*")), r.stdout[-400:])
    # ---- the extra-options line: split as a shell splits it, appended after the tab's own options
    base = rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), output_name="x")
    ce = rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), output_name="x",
                       extra_args="--interlace --sample-points 'my points.txt'")
    errors = lambda spec: [m for lvl, m in spec.problems() if lvl == "error"]  # noqa: E731
    check("custom extra options: after the tab's own options, quoted words kept whole, saved with the settings",
          ce.command() == base.command() + ["--interlace", "--sample-points", "my points.txt"]
          and not [m for m in errors(ce) if "extra options" in m]
          and rs.CustomSpec.from_dict(ce.to_dict()).extra_args == ce.extra_args
          and ce.memory_key() != base.memory_key(), str(ce.command()[-4:]))
    bad = rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), extra_args="--seed '7")
    twice = rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), extra_args="--grid 128")
    check("... an unclosed quote is an error and adds nothing; an option the tab sets already is an error",
          bad.command() == bad._own_command() and any("do not parse" in m for m in errors(bad))
          and any("repeat --grid" in m for m in errors(twice)), str(errors(bad) + errors(twice)))
    # a deprecated spelling or an '=value' is still the option the tab sets (the binary takes it once: a saved
    # '--lambda_th 0.4' next to the tab's '--lambda-th 0.3' would abort the run)
    web = dict(input_file=str(snap), grid=64, output_dir=str(work), fields=["density_a", "tweb_a"])
    old_sp, old_eq, grid_eq, other = (rs.CustomSpec(**web, extra_args=e) for e in
                                      ("--lambda_th 0.4", "--lambda_th=0.4", "--grid=128", "--seed 7"))
    check("... a repeat in a deprecated spelling or as '--name=value' is the same error, under the canonical name",
          "--lambda-th" in old_sp._own_command()
          and all(any("repeat --lambda-th," in m for m in errors(s)) for s in (old_sp, old_eq))
          and any("repeat --grid," in m for m in errors(grid_eq))
          and not [m for m in errors(other) if "extra options" in m],
          str(errors(old_sp) + errors(old_eq) + errors(grid_eq) + errors(other)))
    # a merge switch in the extra options: the tab leaves its own '--ps-merge index-order' out, so the switch stands
    # (live spec and migrated copy: the copy reads '--ordered-merge' as '--ps-merge index-order')
    m_on = rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), extra_args="--ordered-merge")
    m_on_copy = rs.CustomSpec.from_dict(m_on.to_dict())
    check("... no error when the extra options name the merge (the tab then gives none), live spec and migrated copy",
          "--ps-merge" not in m_on._own_command()
          and not [m for s in (m_on, m_on_copy) for m in errors(s) if "extra options" in m]
          and m_on_copy.command()[-2:] == ["--ps-merge", "index-order"],
          str(errors(m_on) + errors(m_on_copy) + m_on_copy.command()[-3:]))
    # both qualities' merge is the index order: the tab names it itself (R7: a GIVEN index order keeps the memory gate
    # on under the auto-tuner's speed split, as the drivers' PS_MERGE does), unless the extra options name a merge in
    # any spelling the binary takes; then the extras' merge is the command's only one
    pair = lambda cmd, mode: any(cmd[i:i + 2] == ["--ps-merge", mode] for i in range(len(cmd)))  # noqa: E731
    plain = rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work))
    named = {e: rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), extra_args=e)
             for e in ("--ps-merge fixed-point", "--ps-merge=completion-order", "--fixed-point-merge",
                       "--completion-order-merge --seed 7")}
    named_copies = {e: rs.CustomSpec.from_dict(s.to_dict()) for e, s in named.items()}
    unclosed = rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), extra_args="--ps-merge 'fixed-point")
    check("... '--ps-merge index-order' on the tab's own command when the extra options name no merge "
          "(an unparsed extra line names none); a merge in the extra options ('--ps-merge MODE', '--ps-merge=MODE' or "
          "an older switch) is no repeat, and the command carries only the extras' merge (live spec and saved copy)",
          plain.mode().ordered_merge and pair(plain._own_command(), "index-order")
          and pair(plain.command(), "index-order")
          and pair(rs.CustomSpec(input_file=str(snap), quality="fast")._own_command(), "index-order")
          and plain.command().count("--ps-merge") == 1 and pair(unclosed._own_command(), "index-order")
          and all(s._extra_names_merge() and "--ps-merge" not in s._own_command()
                  and not [m for m in errors(s) if "extra options" in m]
                  and not pair(s.command(), "index-order")
                  for s in list(named.values()) + list(named_copies.values()))
          and pair(named["--ps-merge fixed-point"].command(), "fixed-point")
          and named["--ps-merge fixed-point"].command().count("--ps-merge") == 1
          and pair(named_copies["--fixed-point-merge"].command(), "fixed-point")
          and not plain._extra_names_merge(),
          f"{plain.command()[-4:]} {[(e, s.command()[-3:], errors(s)) for e, s in named.items()]}")
    # the cell average: the tab names it either way ('--exact' or '--sampled'), so either flag among the extra
    # options, or an older name of '--exact', repeats it (the binary refuses '--exact' beside '--sampled')
    avg = [rs.CustomSpec(input_file=str(snap), grid=64, output_dir=str(work), quality=q, extra_args=e)
           for q, e in (("exact", "--sampled"), ("fast", "--exact-average"), ("exact", "--ps-exact-deposit"))]
    check("... '--exact' or '--sampled' (or an older name of '--exact') among the extra options repeats the tab's cell "
          "average, whichever it is",
          all(any("repeat --exact / --sampled," in m for m in errors(s)) for s in avg), str([errors(s) for s in avg]))
    q = rs.QueuedJob.of("custom", c)
    check("custom: a queued custom-snapshot job rebuilds and has a summary",
          q.build() == c and q.title.startswith("Custom snapshot · cloud.hdf5 · PS-DTFE 64³"), q.title)
    return work, snap, h5py, c, steps


def _export(work, steps):
    """export_script(): the bash and SLURM scripts, and an exported script run for real under the run-log contract."""
    print("export:")
    txt = rs.export_script(steps, "a test job")
    check("bash export: shebang, cd to the repo, every step, failure count as exit status",
          txt.startswith("#!/usr/bin/env bash") and 'cd "$REPO"' in txt
          and all(rs.shell(st.argv, st.env) in txt for st in steps)
          and 'exit "$failed"' in txt and "#SBATCH" not in txt)
    tee = steps[-1].tee
    check("bash export: a step with a run log tees into '<log>.partial', never straight into '<log>' (the run-log "
          "contract)", bool(tee) and f"| tee {rs._q(tee + '.partial')}" in txt
          and f"| tee {rs._q(tee)}\n" not in txt and f"| tee {rs._q(tee)} " not in txt, txt[-600:])
    sl = rs.export_script(steps, "a test job", {"job_name": "j1", "time": "02:00:00", "cpus": 8, "mem": "32G",
                                                "partition": "regular", "mail": ""})
    check("SLURM export: the #SBATCH header, OMP threads from SLURM, no empty options",
          "#SBATCH --job-name=j1" in sl and "#SBATCH --time=02:00:00" in sl and "#SBATCH --cpus-per-task=8" in sl
          and "#SBATCH --mem=32G" in sl and "#SBATCH --partition=regular" in sl and "mail-user" not in sl
          and "OMP_NUM_THREADS" in sl)
    ex = work / "job.sh"
    ex.write_text(rs.export_script([rs.Step("echo", ["echo", "hello"], {}), rs.Step("fail", ["false"], {})], "t"))
    import subprocess
    r = subprocess.run(["bash", str(ex)], capture_output=True, text=True)
    check("an exported script runs, goes on after a failed step and exits with the failure count",
          r.returncode == 1 and "hello" in r.stdout, f"{r.returncode} {r.stdout} {r.stderr}")
    # the run-log contract in an exported script, run for real: a good step's log lands on '<log>' (and a stale
    # '<log>.failed' goes), a failing step's on '<log>.failed' with the old '<log>' untouched; no '.partial' left
    lg = work / "export logs"                      # a space: the quoting is exercised too
    lg.mkdir()
    (lg / "ok.runlog.failed").write_text("an older failed attempt\n")
    (lg / "bad.runlog").write_text("the log of the grids on disk\n")
    ex.write_text(rs.export_script([rs.Step("ok", ["echo", "hello"], {}, tee=str(lg / "ok.runlog")),
                                    rs.Step("bad", ["sh", "-c", "echo oops; exit 3"], {}, tee=str(lg / "bad.runlog"))],
                                   "t"))
    r = subprocess.run(["bash", str(ex)], capture_output=True, text=True)
    check("exported run logs: exit 0 -> '<log>' (stale .failed removed), a failure -> '<log>.failed' (the old log "
          "kept), no '.partial' left, the failure counted",
          r.returncode == 1 and (lg / "ok.runlog").read_text() == "hello\n" and not (lg / "ok.runlog.failed").exists()
          and (lg / "bad.runlog.failed").read_text() == "oops\n"
          and (lg / "bad.runlog").read_text() == "the log of the grids on disk\n"
          and not list(lg.glob("*.partial")), f"{r.returncode} {sorted(p.name for p in lg.iterdir())} {r.stderr}")


def _run_logs(tmp, sim, dt, R):
    """results.py: run logs found and parsed, the stale-output rules, time estimates and failure advice."""
    print("run logs (results.py):")
    snapdir = sim / "snapdir_099"
    for suffix in ("a_den", "a_streams", "a_vel"):
        (snapdir / f"ps_output.{suffix}").write_bytes(b"\0" * 4 * 8)
    log = snapdir / "ps_output.runlog"
    head = ("Build: PS-DTFE 308f08a-dirty (built 2026-09-01T10:00:00Z)\n"
            "RUNNING: ./PS-DTFE /x/y z/combined_099.hdf5 /x/ps_output --grid 512 --periodic --partition 5 5 5 "
            "--max-concurrent 3 --avg-subsamples 3 --input 105 --MpcUnit 1000 --field density_a velocity_a "
            "--ps-gpu --ps-volume-weighted\n"
            "   peak memory (RSS) : 47.24 GB\n   total wall time   : 1h 2m 3s\n"
            "         97186996544  peak memory footprint\n")
    _write_log(log, head)
    recs = R.scan_runs(tmp)
    r0 = recs[0] if recs else None
    check("scan_runs finds the per-family run log", len(recs) == 1 and r0.sim == "TNG100-3-Dark" and r0.snap == 99)
    check("parsed: build stamp, grid, partitions, GPU, wall time, RSS, footprint, fields, outputs",
          r0.binary == "PS-DTFE" and r0.build_rev == "308f08a-dirty" and r0.build_time is not None
          and r0.grid == 512 and r0.partitions == 125 and r0.partition == 5 and r0.max_concurrent == 3
          and r0.gpu and r0.wall_seconds == 3723 and r0.peak_rss_gb == 47.24 and abs(r0.footprint_gb - 90.51) < 0.01
          and r0.fields == ["density_a", "velocity_a"] and r0.ok
          and {p.name for p in r0.outputs} >= {"ps_output.a_den", "ps_output.a_streams", "ps_output.a_vel"}
          and not any(p.suffix == ".runlog" for p in r0.outputs)
          and r0.redshift == 0.0, f"{r0}")
    reasons = R.stale_reasons(r0)
    check("stale: a GPU + partitions build before the 2026-09-29 fix is WRONG",
          any(lv == "error" and "partition" in m for lv, m in reasons), str(reasons))
    check("stale: '.streams' without '.hidden_streams' predates the stream-count fix",
          any(lv == "warning" and "centroid-fallback" in m for lv, m in reasons))
    check("stale: a TNG PS run without --ps-vertex-mass is flagged", any("tetrahedron masses" in m for _, m in reasons))
    (snapdir / "ps_output.a_unresolved").write_bytes(b"\0" * 32)       # the old name of '.hidden_streams'
    _write_log(log, head.replace("2026-09-01", "2026-09-30").replace("--ps-gpu", "--ps-gpu --ps-vertex-mass"))
    reasons = R.stale_reasons(R.scan_runs(tmp)[0])            # a build from before the exact flag
    check("stale: an old '.unresolved' (before the exact hidden-streams flag) is flagged, and nothing else",
          len(reasons) == 1 and reasons[0][0] == "warning" and "hidden-streams flag" in reasons[0][1], str(reasons))
    (snapdir / "ps_output.a_unresolved").unlink()                    # a current build writes the new name
    (snapdir / "ps_output.a_hidden_streams").write_bytes(b"\0" * 32)
    _write_log(log, head.replace("2026-09-01T10:00", "2026-09-30T18:00").replace("--ps-gpu", "--ps-gpu --ps-vertex-mass"))
    r1 = R.scan_runs(tmp)[0]
    check("stale: the same run built after every fix is clean", R.stale_reasons(r1) == [], str(R.stale_reasons(r1)))
    np_head = head.replace("2026-09-01T10:00", "2026-09-30T18:00").replace("--periodic ", "--box 20 40 20 40 20 40 ") \
                  .replace("--ps-gpu", "--ps-gpu --ps-vertex-mass")
    _write_log(log, np_head)
    reasons = R.stale_reasons(R.scan_runs(tmp)[0])
    check("stale: a non-periodic --box run built before the region-face and alpha-shape fixes is flagged by both",
          any("crossing the region" in m for _, m in reasons)
          and any("alpha shape" in m for _, m in reasons) and len(reasons) == 2, str(reasons))
    _write_log(log, np_head.replace("2026-09-30T18:00", "2026-09-30T23:00"))
    r3 = R.scan_runs(tmp)[0]
    check("stale: the same non-periodic run built after both fixes is clean", R.stale_reasons(r3) == [], str(R.stale_reasons(r3)))
    dd_head = ("Build: PS-DTFE-double 3e0c1a6-dirty (built 2026-10-01T18:00:00Z)\n"
               "RUNNING: ./PS-DTFE-double /x/y z/combined_099.hdf5 /x/ps_output --grid 256 --periodic --input 105 "
               "--MpcUnit 1000 --field density_a --ps-gpu --ps-vertex-mass\n   peak memory (RSS) : 9.0 GB\n   total wall time   : 1m 2s\n")
    _write_log(log, dd_head)
    r5 = R.scan_runs(tmp)[0]
    check("run logs: the double pair is recognised (estimator, precision, settings, options text)",
          r5.estimator == "ps" and r5.precision == "double" and r5.settings()["precision"] == "double"
          and "double" in r5.options_text() and R.stale_reasons(r5) == [], str(R.stale_reasons(r5)))
    dt_head = ("Build: DTFE 3e0c1a6-dirty (built 2026-10-01T09:00:00Z)\n"
               "RUNNING: ./DTFE /x/y z/combined_099.hdf5 /x/ps_output --grid 512 --periodic --input 105 --MpcUnit 1000 "
               "--field density_a velocity_a --gpu\n"
               "DTFE Metal: Metal command buffer failed: Impacting Interactivity -- retrying the deposit from scratch "
               "(attempt 2/3, target 62 ms buffers, 24 ms gaps)\n"
               "   peak memory (RSS) : 7.24 GB\n   total wall time   : 6m 37s\n")
    _write_log(log, dt_head)
    r4 = R.scan_runs(tmp)[0]
    reasons = R.stale_reasons(r4)
    check("stale: a standard-DTFE GPU run whose deposit was restarted by the GPU watchdog is flagged (built before the fix)",
          r4.estimator == "dtfe" and r4.gpu and r4.gpu_retried and len(reasons) == 1
          and "watchdog" in reasons[0][1] and reasons[0][0] == "warning", str(reasons))
    _write_log(log, dt_head.replace("retrying the deposit from scratch", "nothing of the kind"))
    check("stale: ... the same run without a restart is clean", R.stale_reasons(R.scan_runs(tmp)[0]) == [])
    _write_log(log, dt_head.replace("2026-10-01T09:00", "2026-10-01T23:00"))
    check("stale: ... and a restart with the fixed build is clean (its retry drains the queue first)",
          R.stale_reasons(R.scan_runs(tmp)[0]) == [], str(R.stale_reasons(R.scan_runs(tmp)[0])))
    _write_log(log, head.split("\n", 1)[1], mtime=dt.datetime(2026, 8, 1).timestamp())
    r2 = R.scan_runs(tmp)[0]
    reasons = R.stale_reasons(r2)
    check("stale: without a build stamp the run date decides, and says so",
          r2.build_time is None and any("run date" in m for _, m in reasons), str(reasons))
    check("settings(): the Grids-tab settings of the run",
          r1.settings()["grid"] == 512 and r1.settings()["vertex_mass"] and r1.settings()["fields"] == ["density_a", "velocity_a"])
    est = R.time_estimate([r0, r1, r2], sim="TNG100-3-Dark", grid=512, estimator="ps", gpu=True, sliced=False)
    check("time estimate: the median of comparable earlier runs", est == (3723, 3), str(est))
    check("time estimate: none for another grid",
          R.time_estimate([r0], sim="TNG100-3-Dark", grid=256, estimator="ps", gpu=True, sliced=False) is None)
    _write_log(log, "RUNNING: ./PS-DTFE a b --grid 64\n~~~ ERROR ~~~ something\n")
    check("a log with an ERROR block is not ok", not R.scan_runs(tmp)[0].ok)
    for text, codes, key in (("... std::bad_alloc\n", [1], "Out of memory"), ("x", [137], "Out of memory"),
                             ("unrecognised option '--foo'", [1], "older"), ("points into iCloud Drive", [1], "iCloud"),
                             ("no snapshots under /data/sims", [1], "Data not found"),
                             ("no data root: set DTFE_DATA_ROOT to the folder that holds the simulations", [1],
                              "Data not found"), ("boom", [2], "failed")):
        found = [d.title for d in R.diagnose(text, codes)]
        check(f"diagnose: {key}", any(key in t for t in found), str(found))
    check("diagnose: a clean log with exit 0 has no advice", R.diagnose("all fine", [0]) == [])
    n = R.notify_command('a "quoted" title', "text")
    check("notify_command: a runnable command (or None where there is none)", n is None or isinstance(n, list))
    return snapdir


def _presets_help(sim, P, R):
    """presets.py: the binary's help text parsed, the built-in presets applied, no '--' in the user-facing texts."""
    import re
    print("presets and help (presets.py):")
    helptext = ("Main options:\n"
                "  -g [ --grid ] arg      specify grid size\n"
                "                         along each direction.\n"
                "  --ps-vertex-mass       PS-DTFE only: assign each\n"
                "                         tetrahedron its mass.\n"
                "  --a-really-long-option-name-that-pushes arg\n"
                "                         the text goes on the next line\n"
                "\nField choices:\n")
    h = P.parse_help(helptext)
    check("parse_help: short+long, wrapped continuation, long names with the text below",
          h.get("--grid") == "specify grid size along each direction."
          and h.get("--ps-vertex-mass") == "PS-DTFE only: assign each tetrahedron its mass."
          and h.get("--a-really-long-option-name-that-pushes") == "the text goes on the next line", str(h))
    spec = rs.RunSpec(grid=512, fields=list(rs.PS_DEFAULT_FIELDS))
    planes = [sim / "hires_plane_z64.bin"]
    # three built-ins, in the two-mode model: each names its quality and nothing the code decides; their words name
    # no measured number (no seconds, minutes, GB or percent: they go stale)
    measured = re.compile(r"\d\s*(s|sec|seconds?|min|minutes?|h|hours?|GB|MB|%)(?![A-Za-z])|about a minute")
    check("three built-in presets, Quick look (Fast), Standard and Publication (Exact): each names its quality, none a "
          "setting the two modes replaced or one outside the preset keys; no measured number in their words",
          list(P.BUILTIN_PRESETS) == ["Quick look", "Standard", "Publication"]
          and [s_["quality"] for _, s_ in P.BUILTIN_PRESETS.values()] == ["fast", "exact", "exact"]
          and not any(set(s_) & set(rs.REMOVED_SETTINGS) or set(s_) - set(P.PRESET_KEYS)
                      for _, s_ in P.BUILTIN_PRESETS.values())
          and not [t for t, _ in P.BUILTIN_PRESETS.values() if measured.search(t)]
          and P.BUILTIN_PRESETS["Quick look"][0].startswith("Fast") and P.BUILTIN_PRESETS["Standard"][0].startswith("Exact"),
          str([t for t, _ in P.BUILTIN_PRESETS.values()]))
    pub, std = P.BUILTIN_PRESETS["Publication"][1], P.BUILTIN_PRESETS["Standard"][1]
    check("... Publication is Standard plus the caustics and the hi-res slice on the largest image plane",
          {k: v for k, v in pub.items() if k not in ("caustics", "slice_plane", "slice_vel_grad")}
          == {k: v for k, v in std.items() if k not in ("caustics", "slice_plane", "slice_vel_grad")}
          and pub["caustics"] and not std["caustics"] and pub["slice_plane"] == "largest" and not std["slice_plane"],
          f"{pub} {std}")
    changed = P.apply_preset(spec, P.BUILTIN_PRESETS["Quick look"][1], planes)
    check("preset 'Quick look': 128^3, density+velocity, no slice, Fast",
          spec.grid == 128 and spec.fields == ["density_a", "velocity_a"] and spec.slice_plane == ""
          and spec.quality == "fast" and "grid" in changed and "quality" in changed)
    P.apply_preset(spec, P.BUILTIN_PRESETS["Publication"][1], planes)
    check("preset 'Publication': the largest image plane and caustics",
          spec.slice_plane == str(planes[0]) and spec.caustics and spec.grid == 512)
    cap = P.capture(spec)
    check("capture(): only preset-able settings, the quality among them (none the two modes removed)",
          "grid" in cap and "sim" not in cap and "snapshots" not in cap and cap.get("quality") == spec.quality
          and not set(cap) & set(rs.REMOVED_SETTINGS) and "quality" in P.PRESET_KEYS
          and not set(P.PRESET_KEYS) & set(rs.REMOVED_SETTINGS)
          and "vertex_mass" not in P.capture(rs.CustomSpec(vertex_mass=True)), str(cap))
    cc = rs.CustomSpec()
    P.apply_preset(cc, P.BUILTIN_PRESETS["Publication"][1], planes)
    check("a preset applies to a custom-snapshot job too (fields it lacks are skipped)", cc.grid == 512 and cc.caustics)
    # the presets give both estimators Exact, the production setting; 'Quick look' is Fast on both; a preset from
    # before the two modes names the cell average ('deposit', and 'deposit_dtfe' for a standard-DTFE job), which
    # becomes the quality, and its other removed settings are dropped without an error
    sd, sp, ql, cd = (rs.RunSpec(estimator="dtfe", quality="fast"), rs.RunSpec(quality="fast"),
                      rs.RunSpec(estimator="dtfe"), rs.CustomSpec(estimator="dtfe", quality="fast"))
    ch = P.apply_preset(sd, P.BUILTIN_PRESETS["Standard"][1], planes)
    chp = P.apply_preset(sp, P.BUILTIN_PRESETS["Standard"][1], planes)
    P.apply_preset(ql, P.BUILTIN_PRESETS["Quick look"][1], planes)
    P.apply_preset(cd, P.BUILTIN_PRESETS["Publication"][1], planes)
    older = {"grid": 256, "nsub": 2, "deposit": "sampled", "deposit_dtfe": "exact", "gpu": False,
             "vertex_mass": False, "volume_weighted": False, "caustics": True}
    old_ps, old_dt, old_cu = rs.RunSpec(), rs.RunSpec(estimator="dtfe"), rs.CustomSpec(vertex_mass=True)
    ch_ps, ch_dt, ch_cu = (P.apply_preset(j, older) for j in (old_ps, old_dt, old_cu))
    check("preset 'Standard'/'Publication': Exact on both estimators; 'Quick look' Fast on both; a preset naming the "
          "estimator follows it",
          sd.quality == "exact" and "quality" in ch and sp.quality == "exact" and "quality" in chp
          and ql.quality == "fast" and cd.quality == "exact" and "deposit_dtfe" not in sd.to_dict()
          and not any("deposit_dtfe" in s for _, s in P.BUILTIN_PRESETS.values())
          and P.apply_preset(rs.RunSpec(estimator="dtfe"), {"estimator": "ps", "deposit": "sampled",
                                                            "deposit_dtfe": "exact"}) == ["estimator", "quality"],
          f"{sd.quality} {sp.quality} {ql.quality} {cd.quality} {ch}")
    check("an older preset loads: 'deposit' sampled is Fast for a PS-DTFE job, its 'deposit_dtfe' exact is Exact for "
          "a standard-DTFE one; its sub-samples, GPU, weighting and vertex masses are dropped (a Custom job keeps its "
          "initial positions: they describe the input file)",
          old_ps.quality == "fast" and old_dt.quality == "exact" and old_cu.quality == "fast"
          and ch_ps == ["grid", "quality"] and ch_dt == ["grid"] and old_cu.vertex_mass is True
          and "vertex_mass" not in ch_cu and old_ps.caustics and old_ps.grid == old_dt.grid == old_cu.grid == 256,
          f"{ch_ps} {ch_dt} {ch_cu}")
    # a preset never sets what is not a preset key, whatever an older or hand-edited one holds (the machine use, the
    # output prefix, a removed setting): only the run's choices
    odd = rs.RunSpec(run_mode="gentle")
    ch_odd = P.apply_preset(odd, {"run_mode": "performance", "output_prefix": "x_", "scratch_dir": "/tmp/s",
                                  "grid": 64, "partition": 2, "gpu": True})
    check("... a preset sets only the preset keys (a stray machine use, prefix or scratch folder in it is left out)",
          ch_odd == ["grid"] and odd.run_mode == "gentle" and odd.output_prefix == rs.RunSpec().output_prefix
          and odd.scratch_dir == rs.RunSpec().scratch_dir, str(ch_odd))
    # the Quality pair's tips (the cell-average labels went with the cell-average menu): Exact names itself the default,
    # Fast does not; no timing in either (PS-DTFE's Fast is not measured to be quicker by any amount)
    check("the Quality pair's tips name Exact the default (Fast not), without a digit or a '--'; no label of a removed "
          "control is left in the option texts",
          "default" in rs.QUALITY_TIP["exact"] and "default" not in rs.QUALITY_TIP["fast"]
          and set(rs.QUALITY_TIP) == set(rs.QUALITIES)
          and not any(ch.isdigit() or "--" in t for t in rs.QUALITY_TIP.values() for ch in t)
          and not {"exact", "sampled", "exact_dtfe", "sampled_dtfe", "gpu", "gpu_dtfe", "volume_weighted",
                   "linear_deposit", "face_gradient", "stream_density", "pts_conventions", "parallel_triangulation",
                   "ordered_merge"} & set(P.OPTION_TEXT), str(rs.QUALITY_TIP))
    real = P.flag_help("PS-DTFE")
    if real:
        check("the binary's own --full_help: every production flag has a description",
              all(real.get(f) for f in ("--ps-vertex-mass", "--ps-volume-weighted", "--ps-caustics", "--grid",
                                        "--serve", "--ps-per-stream-lagrangian", "--pts-scalar")), str({k: bool(v) for k, v in real.items()}))
    else:
        print("   SKIP  flag_help (PS-DTFE not built)")

    texts = ([fx.message for fx in R.KNOWN_FIXES]
             + [d.title + " " + d.advice for d in R.diagnose(
                 "std::bad_alloc unrecognised option points into iCloud Operation not permitted "
                 "no snapshots under /data/x No 'InitialCoordinates' OVER BUDGET", [1, 137])]
             + [t for t, _ in P.BUILTIN_PRESETS.values()] + [lab for lab, _ in P.OPTION_TEXT.values()]
             + [P.tooltip(k) for k in P.OPTION_TEXT]
             + [fs.title + " " + fs.about + " " + " ".join(o.label + " " + o.help for o in fs.opts)
                for fs in rs.FIGURE_SETS])
    check("no '--' in the stale-output rules, failure advice, presets, option tooltips or figure sets",
          not [t for t in texts if "--" in t], str([t for t in texts if "--" in t][:3]))
    check("plain_help: option names lose their dashes, ' -- ' and 'a--b' become real dashes",
          P.plain_help("use '--partition' -- or --serve; the div--density slope")
          == "use 'partition' — or serve; the div–density slope")


def _explore_data(tmp, json, np, G, work, snap, h5py, c, snapdir):
    """grids.py: outputs found, slices, the exact fold flag, arrows, colours, the query server's settings and cost."""
    print("Explore data (grids.py):")
    n = 8
    rng = np.random.default_rng(0)
    den = rng.random((n, n, n)).astype(np.float32) + 0.5
    vel = rng.random((n, n, n, 3)).astype(np.float32)
    (snapdir / "ps_output.a_den").write_bytes(den.tobytes())
    (snapdir / "ps_output.a_vel").write_bytes(vel.tobytes())
    for suffix in ("a_streams", "a_hidden_streams"):
        (snapdir / f"ps_output.{suffix}").write_bytes(b"\0" * 4 * n ** 3)
    with h5py.File(snapdir / "combined_099.hdf5", "a") as f:
        f["Header"].attrs["BoxSize"] = 80000.0
    outs = G.find_outputs(tmp)
    o = outs[0][1] if outs else None
    check("find_outputs: the TNG-layout run, its box (h-free kpc -> Mpc) and redshift",
          o is not None and o.n == n and o.length[0] == 80.0 and o.redshift == 0.0
          and o.fields()[:2] == ["density", "streams"], str(outs))
    check("find_outputs: a TNG-layout output knows its simulation and snapshot number; natural simulation order",
          o is not None and o.sim == "TNG100-3-Dark" and o.snap == 99
          and sorted(["TNG300-3-Dark", "TNG50-3-Dark", "TNG100-3-Dark"], key=G.sim_sort_key)
          == ["TNG50-3-Dark", "TNG100-3-Dark", "TNG300-3-Dark"], f"{o and (o.sim, o.snap)}")
    check("slice: plane = cube[..., k] for the z normal", np.allclose(o.slice("density", 2, 3), den[:, :, 3]))
    check("slice: x normal, y normal", np.allclose(o.slice("density", 0, 1), den[1])
          and np.allclose(o.slice("density", 1, 5), den[:, 5, :]))
    check("slice: |v| and one component",
          np.allclose(o.slice("velocity", 2, 0, "norm"), np.linalg.norm(vel[:, :, 0], axis=-1), rtol=1e-6)
          and np.allclose(o.slice("velocity", 2, 0, "y"), vel[:, :, 0, 1]))
    check("value(): one cell", abs(o.value("density", (2, 3, 4)) - float(den[2, 3, 4])) < 1e-6)
    # the 'caustic' map of a periodic PS run is the EXACT fold flag,
    # dtfelib.io.caustic_is_fold(class mask, streams=, hidden=), not the stored '.caustic' -- which older grids
    # set only where a folded tetrahedron deposited a sample (21.0% of TNG100-3-Dark z=1 against 33.6%). A
    # non-periodic run (its sidecar) and an output without a streams grid show the stored grid; 'caustic_class' stays raw
    from dtfelib.io import caustic_is_fold
    fd = work / "fold"
    fd.mkdir(exist_ok=True)
    st = np.ones((n, n, n), np.float32)
    st[1, 2, 3] = 3.0                                   # multi-stream cells: a fold
    st[4, :, 5] = 3.0
    st[0, 0, 0] = 0.0                                   # an empty cell
    hid = np.zeros((n, n, n), np.float32)
    hid[2, 2, 3] = 1.0                                  # bit 1: a fold thinner than the sampling
    hid[3, 3, 3] = 2.0                                  # bit 2 alone: mass of an unsampled tetrahedron, no fold
    ccm = np.full((n, n, n), 1 + 4, np.float32)          # a positive tetrahedron with k = 0 everywhere
    ccm[6, 6, 3] = 1 + 2 + 8                             # both parity bits: a fold by the class mask
    raw = np.zeros((n, n, n), np.float32)
    raw[1, 2, 3] = 1.0                                  # the old per-sample flag: one of the fold cells only
    for sfx, arr in (("a_den", den), ("a_streams", st), ("a_hidden_streams", hid), ("caustic", raw),
                     ("causticClass", ccm)):
        (fd / f"fold.{sfx}").write_bytes(arr.tobytes())
    want = caustic_is_fold(ccm, streams=st, hidden=hid)
    of = G.OutputSet(fd / "fold")
    planes_ok = all(np.array_equal(of.slice("caustic", ax, k), np.take(want, k, axis=ax).astype(np.float64))
                    for ax in range(3) for k in (0, 2, 3, 5))
    check("Explore's caustic map of a periodic PS run: the exact fold flag on every axis (streams > 1, hidden bit 1, "
          "the class mask's fold bits; not hidden bit 2 alone or an empty cell), also in value(); the class map stays raw",
          of.fold_exact() and planes_ok and want[1, 2, 3] and want[4, 0, 5] and want[2, 2, 3] and want[6, 6, 3]
          and not want[3, 3, 3] and not want[0, 0, 0] and int(want.sum()) == 1 + n + 1 + 1
          and of.value("caustic", (2, 2, 3)) == 1.0 and of.value("caustic", (3, 3, 3)) == 0.0
          and np.array_equal(of.slice("caustic_class", 2, 3), ccm[:, :, 3].astype(np.float64)),
          f"{of.fold_exact()} {planes_ok} {int(want.sum())}")
    (fd / "fold.gui.json").write_text(json.dumps({"estimator": "ps", "periodic": False, "dim": 3}))
    onp = G.OutputSet(fd / "fold")
    nd = work / "fold_nostreams"
    nd.mkdir(exist_ok=True)
    for sfx, arr in (("a_den", den), ("caustic", raw)):
        (nd / f"fold.{sfx}").write_bytes(arr.tobytes())
    ons = G.OutputSet(nd / "fold")
    check("... a non-periodic run (the union over-flags there) and an output without a streams grid show the stored "
          "'.caustic'", not onp.fold_exact() and np.array_equal(onp.slice("caustic", 2, 3), raw[:, :, 3])
          and not ons.fold_exact() and np.array_equal(ons.slice("caustic", 2, 3), raw[:, :, 3]))
    (fd / "fold.gui.json").unlink()
    check("cell_center / plane_point", np.allclose(o.cell_center((0, 0, 7)), [5, 5, 75])
          and list(o.plane_point(2, 7, 1, 2)) == [1, 2, 7])
    ar = o.arrow_field("velocity", 2, 0, per_side=4)
    bu = vel[:, :, 0, 0].reshape(4, 2, 4, 2).mean(axis=(1, 3))        # 8 cells, 4 arrows a side
    check("arrow_field: in-plane components block-averaged, longest arrow = 1",
          ar is not None and ar["spacing"] == 2.0 and len(ar["x"]) == 16
          and np.allclose(ar["u"], bu, rtol=1e-6) and abs(np.hypot(ar["dx"], ar["dy"]).max() - 1) < 1e-9
          and o.arrow_field("density", 2, 0) is None, str(ar and ar["spacing"]))
    vd = work / "vec"                                   # a pure +y flow, to pin the orientation
    vd.mkdir(exist_ok=True)
    (vd / "flow.a_den").write_bytes(den.tobytes())
    flow = np.zeros_like(vel)
    flow[..., 1] = 2.0
    (vd / "flow.a_vel").write_bytes(flow.tobytes())
    ov = G.OutputSet(vd / "flow")
    up, right = ov.arrow_field("velocity", 2, 3, per_side=8), ov.arrow_field("velocity", 0, 3, per_side=8)
    check("arrow_field: +y points UP in a z slice (the image is plane.T[::-1]) and RIGHT in an x slice",
          np.allclose(up["dx"], 0) and np.allclose(up["dy"], -1) and np.allclose(right["dx"], 1)
          and np.allclose(right["dy"], 0) and right["axes"] == (1, 2))
    check("arrow_field: arrows sit on the block centres, image y measured from the top",
          np.allclose(sorted(set(up["x"])), np.arange(8) + 0.5) and np.allclose(sorted(set(up["y"])), np.arange(8) + 0.5))
    rgb, vr = G.colorize(np.array([[1.0, 10.0], [100.0, 1000.0]]), "viridis", log=True, clip=(0, 100))
    check("colorize: log10 range, RGB bytes", rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8 and vr == (0.0, 3.0))
    _, vr = G.colorize(np.array([[-1.0, 3.0]]), "RdBu_r", clip=(0, 100), symmetric=True)
    check("colorize: a symmetric range for diverging maps", vr == (-3.0, 3.0))
    kw = G.server_settings(o)
    check("server_settings (TNG layout): the snapdir's combined file, periodic, PS",
          kw and kw["snapshot"].endswith("combined_099.hdf5") and kw["phase_space"] and kw["periodic"], str(kw))
    od = work / "out"
    od.mkdir(exist_ok=True)
    (od / "run1.a_den").write_bytes(den.tobytes())
    (od / "run1.gui.json").write_text(json.dumps(c.settings_sidecar()))
    outs = G.find_outputs(tmp, [od])
    mine = [x for t, x in outs if t.startswith("custom ·")]
    check("find_outputs: a custom-snapshot run from its folder, box from its sidecar; no simulation or snapshot",
          len(mine) == 1 and mine[0].length[0] == 50.0 and mine[0].sim == "" and mine[0].snap is None, str(outs))
    kw = G.server_settings(mine[0])
    check("server_settings (custom snapshot): the job's own input, units and estimator",
          kw and kw["snapshot"] == str(snap) and kw["mpc_unit"] == 1000.0 and kw["phase_space"], str(kw))
    for name, npart in (("big.hdf5", 100_000_000), ("small.hdf5", 32_768)):
        with h5py.File(work / name, "w") as f:
            f.create_group("Header").attrs["NumPart_Total"] = np.array([0, npart, 0, 0, 0, 0], dtype=np.uint32)
    big = G.server_cost(work / "big.hdf5", ram_bytes=64 * 2 ** 30)
    small = G.server_cost(work / "small.hdf5", ram_bytes=64 * 2 ** 30)
    check("server_cost: a TNG100-sized box on 64 GB is split, with its cache size; a small one is not",
          big["split"] and 40 < big["cache_gb"] < 80 and big["single_gb"] > big["budget_gb"]
          and not small["split"], f"{big} {small}")
    check("server_cost: None for a file it cannot read", G.server_cost(work / "nope.hdf5") is None)
    (od / "broken.a_den").write_bytes(b"\0" * 7)
    check("an unreadable grid is skipped, not an error", not G.OutputSet(od / "broken"))



def two_d(tmp: Path):
    """The 2D programs (make DIM=2) in the launcher: names, the custom-snapshot job, the grid reader."""
    import json
    import subprocess

    import numpy as np
    import grids as G
    import results as R

    print("2D (the -2d programs):")
    check("binary and object-directory names: -2d before -double, _2d before _d",
          rs.binary_name("ps", "double", 2) == "PS-DTFE-2d-double" and rs.binary_name("dtfe", dim=2) == "DTFE-2d"
          and rs._objdir("ps", "double", 2).name == "o_ps_2d_d" and rs._objdir("dtfe", dim=2).name == "o_2d"
          and rs.build_hint("single", 2) == "make DTFE PS-DTFE DIM=2")
    check("a 2D run has no GPU and no parallel insertion (the kernels and CGAL's parallel build are 3D)",
          not rs.gpu_built("ps", dim=2) and not rs.gpu_built("dtfe", dim=2) and not rs.tbb_built(dim=2)
          and rs.gpu_backend("ps", dim=2) == "" and not rs.recommended_gpu("exact", ["density_a"], 10**8, dim=2))

    work = tmp.parent / "two_d"
    work.mkdir()
    import h5py
    for name, cols in (("plane.hdf5", 2), ("cube.hdf5", 3)):
        with h5py.File(work / name, "w") as f:
            f.create_group("Header").attrs["BoxSize"] = 100.0
            f.create_group("PartType1").create_dataset("Coordinates", data=np.zeros((16, cols)))
    (work / "plane.txt").write_text("2\n0 100 0 100\n1 2 1\n3 4 1\n")
    (work / "cube.txt").write_text("1\n0 1 0 1 0 1\n0.5 0.5 0.5 1\n")
    check("snapshot_dim: HDF5 coordinate columns, the text box line, unknown for Gadget binary",
          rs.snapshot_dim(work / "plane.hdf5") == 2 and rs.snapshot_dim(work / "cube.hdf5") == 3
          and rs.snapshot_dim(work / "plane.txt", 111) == 2 and rs.snapshot_dim(work / "cube.txt", 112) == 3
          and rs.snapshot_dim(work / "plane.hdf5", 101) is None and rs.snapshot_dim(work / "nope.hdf5") is None)

    c = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=2, estimator="ps", grid=64,
                      fields=["density_a", "velocity_a", "tweb_a"],
                      box=[0, 100, 0, 100], output_dir=str(work / "out"), output_name="p")
    d = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=2, estimator="dtfe", output_dir=str(work / "out"))
    with fake_build("gpu"):         # (a GPU build: still no GPU in 2D)
        a, a_fast = c.command(), rs.CustomSpec.from_dict(dict(c.to_dict(), quality="fast")).command()
        d_cmd = d.command()
        lv = {m: lvl for lvl, m in c.problems()}
    check("custom 2D: the -2d binary, a 4-number box, no split named (the auto-tuner's); no GPU, no parallel "
          "insertion in either quality, on a GPU build too; the T-web kept",
          Path(a[0]).name == "PS-DTFE-2d" and a[a.index("--box") + 1:a.index("--box") + 6] == ["0", "100", "0", "100", "--field"]
          and "--partition" not in a and not {"--gpu", "--ps-gpu", "--gpu-share"} & set(a + a_fast)
          and "--parallel-triangulation" not in a + a_fast and "tweb_a" in a and "velocity_a" in a, str(a))
    check("custom 2D: no GPU or parallel-insertion line (nothing to choose: the code's), nothing about the T-web (2D "
          "has it)",
          not any("GPU" in m or "parallel" in m or "T-web" in m for m in lv), str(lv))
    check("custom 2D standard DTFE: DTFE-2d, no GPU flag", Path(d_cmd[0]).name == "DTFE-2d" and "--gpu" not in d_cmd)
    bad = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=2, box=[0, 1, 0, 1, 0, 1], output_dir=str(work / "out"))
    wrong = rs.CustomSpec(input_file=str(work / "plane.hdf5"), dim=3, output_dir=str(work / "out"))
    wrong3 = rs.CustomSpec(input_file=str(work / "cube.hdf5"), dim=2, output_dir=str(work / "out"))
    check("custom 2D: a 6-number box is refused (it needs 4)",
          any(lvl == "error" and "4 numbers" in m for lvl, m in bad.problems()), levels(bad)[1])
    check("custom: the dimension must match the file (a 2D file run as 3D, and the reverse)",
          any(lvl == "error" and "is 2D" in m for lvl, m in wrong.problems())
          and any(lvl == "error" and "is 3D" in m for lvl, m in wrong3.problems()))
    side = c.settings_sidecar()
    check("custom 2D: the sidecar records the dimension and a 4-number box in Mpc",
          side["dim"] == 2 and side["box_mpc"] == [0.0, 100.0, 0.0, 100.0], str(side))
    demo = rs.CustomSpec(input_file=str(work / "demo" / "d.hdf5"), dim=2, demo_n=32, output_dir=str(work / "out"),
                         demo_file=str(work / "demo" / "d.hdf5"))      # generated only at the demo's own path
    gen = [st.argv for st in demo.steps() if "generate" in st.label]
    check("custom 2D demo: the generator writes a 2D snapshot", gen and gen[0][-2:] == ["--dim", "2"], str(gen))
    texts = [rs.DIM_TEXT[2]] + [m for _, m in c.problems() + bad.problems() + wrong.problems()]
    check("no '--' in the 2D texts", not [t for t in texts if "--" in t], str([t for t in texts if "--" in t]))

    # the grid reader on a synthetic 2D output (12^2 cells; no sidecar: the files tell)
    n = 12
    rng = np.random.default_rng(3)
    od = work / "grids"
    od.mkdir()
    den = (rng.random((n, n)) + 0.5).astype(np.float32)
    vel = rng.random((n, n, 2)).astype(np.float32)
    grad = rng.random((n, n, 4)).astype(np.float32)
    vort = rng.random((n, n)).astype(np.float32)
    disp = rng.random((n, n, 3)).astype(np.float32)
    for sfx, arr in (("a_den", den), ("a_vel", vel), ("a_velGrad", grad), ("a_velVort", vort),
                     ("a_velDispTensor", disp), ("a_streams", np.ones((n, n), np.float32))):
        (od / f"r2.{sfx}").write_bytes(arr.tobytes())
    o = G.OutputSet(od / "r2")
    check("OutputSet: a 2D output is told from its files (a 2-component velocity), every field kept",
          o.dim == 2 and o.n == n and {k: f.ncomp for k, f in o.files.items()}
          == {"density": 1, "velocity": 2, "gradient": 4, "vorticity": 1, "dispersion_tensor": 3, "streams": 1},
          f"{o.dim} {o.n} {[(k, f.ncomp) for k, f in o.files.items()]}")
    check("OutputSet 2D: the slice is the whole grid whatever the axis and index",
          np.allclose(o.slice("density", 0, 5), den) and np.allclose(o.slice("density"), den))
    check("OutputSet 2D: component menus and values (velocity x/y, gradient trace = xx + yy, tensor xy)",
          G.components("velocity", 2, 2) == ["norm", "x", "y"]
          and G.components("gradient", 4, 2) == ["trace (divergence)", "xx", "xy", "yx", "yy"]
          and np.allclose(o.slice("velocity", component="y"), vel[..., 1])
          and np.allclose(o.slice("gradient", component="trace (divergence)"), grad[..., 0] + grad[..., 3])
          and np.allclose(o.slice("gradient", component="yx"), grad[..., 2])
          and np.allclose(o.slice("dispersion_tensor", component="xy"), disp[..., 1]))
    check("OutputSet 2D: the vorticity map is the curl's z (the grid stores -w_z / 2)",
          G.components("vorticity", 1, 2) == ["value"] and np.allclose(o.slice("vorticity"), -2.0 * vort))
    ar = o.arrow_field("velocity", per_side=6)
    check("OutputSet 2D: velocity arrows from the two components; value and cell of (i, j)",
          ar is not None and ar["u"].shape == (6, 6) and ar["axes"] == (0, 1)
          and abs(o.value("density", (3, 4)) - float(den[3, 4])) < 1e-6 and list(o.plane_point(2, 0, 3, 4)) == [3, 4])
    o.box = ((0.0, 0.0), (60.0, 30.0))
    check("OutputSet 2D: a two-corner box; lo / length keep a third, zero-thickness entry",
          list(o.lo) == [0.0, 0.0, 0.0] and list(o.length) == [60.0, 30.0, 0.0])
    # scalar-only outputs: 32^2 float32 is also 8^3 float64, 256^2 float32 also 32^3 float64; 64^2 float32
    # is also 16^3 float32. A phase-space run's stream counts tell the precision; else 3D
    (od / "s32.a_den").write_bytes(np.ones((32, 32), np.float32).tobytes())
    (od / "s32.a_streams").write_bytes(np.ones((32, 32), np.float32).tobytes())
    (od / "c32.a_den").write_bytes(np.ones((32, 32, 32), np.float64).tobytes())
    (od / "c32.a_streams").write_bytes(np.ones((32, 32, 32), np.float64).tobytes())
    (od / "d32.a_den").write_bytes(np.ones((32, 32, 32), np.float64).tobytes())
    (od / "s64.a_den").write_bytes(np.ones((64, 64), np.float32).tobytes())
    (od / "s64.gui.json").write_text(json.dumps({"dim": 2}))
    s32, c32, d32 = G.OutputSet(od / "s32"), G.OutputSet(od / "c32"), G.OutputSet(od / "d32")
    check("OutputSet: same bytes in two precisions -- the stream counts pick the reading (a 32^2 float32 plane, "
          "a 32^3 float64 cube); without them 3D, as before; a sidecar settles any tie",
          (s32.dim, s32.n) == (2, 32) and (c32.dim, c32.n, c32.dtype) == (3, 32, np.float64)
          and (d32.dim, d32.n) == (3, 32) and (G.OutputSet(od / "s64").dim, G.OutputSet(od / "s64").n) == (2, 64),
          f"{(s32.dim, s32.n)} {(c32.dim, c32.n, c32.dtype)} {(d32.dim, d32.n)}")
    # one precision, same bytes (512^2 == 64^3): neighbouring cells decide -- the reading whose neighbour
    # stride holds clearly the smaller differences; a field without structure stays 3D
    def smooth(shape, ell):
        k2 = sum(a * a for a in np.meshgrid(*[np.fft.fftfreq(m) for m in shape], indexing="ij"))
        f = np.fft.ifftn(np.fft.fftn(rng.standard_normal(shape)) * np.exp(-0.5 * k2 * (2 * np.pi * ell) ** 2)).real
        return (1 + 0.6 * f / f.std()).clip(0.01).astype(np.float32)
    smooth((512, 512), 4).tofile(od / "p512.a_den")
    smooth((64, 64, 64), 2).tofile(od / "c64.a_den")
    np.ones((512, 512), np.float32).tofile(od / "f512.a_den")
    p512, c64, f512 = G.OutputSet(od / "p512"), G.OutputSet(od / "c64"), G.OutputSet(od / "f512")
    check("OutputSet: a smooth 512^2 plane and a smooth 64^3 cube of the same bytes are told apart by their "
          "neighbours; a featureless one stays 3D", (p512.dim, p512.n) == (2, 512) and (c64.dim, c64.n) == (3, 64)
          and f512.dim == 3, f"{(p512.dim, p512.n)} {(c64.dim, c64.n)} {f512.dim}")
    check("custom_box / server_settings: a 2D sidecar gives two-entry corners, the 2D server and a 4-number box",
          G.custom_box({"dim": 2, "box_mpc": [0, 50, 0, 40]}) == ((0, 0), (50, 40)))
    (od / "c2.a_den").write_bytes(den.tobytes())
    (od / "c2.a_vel").write_bytes(vel.tobytes())
    (od / "c2.gui.json").write_text(json.dumps(dict(side, input_file=str(work / "plane.hdf5"))))
    kw = G.server_settings(G.OutputSet(od / "c2"))
    check("server_settings 2D: dim 2 and the run's 4-number box", kw and kw.get("dim") == 2
          and kw["options"][:1] == ["--box"] and len(kw["options"]) == 5, str(kw))
    # a 3D vorticity grid holds [W_xy, W_xz, W_yz] = [-w_z, w_y, -w_x] / 2
    od3 = work / "v3"
    od3.mkdir()
    w3 = rng.random((4, 4, 4, 3)).astype(np.float32)
    (od3 / "v.a_den").write_bytes(np.ones((4, 4, 4), np.float32).tobytes())
    (od3 / "v.a_velVort").write_bytes(w3.tobytes())
    o3 = G.OutputSet(od3 / "v")
    check("OutputSet 3D: the vorticity map is the curl (x, y, z) = 2 (-W_yz, W_xz, -W_xy)",
          np.allclose(o3.slice("vorticity", 2, 1, "x"), -2 * w3[:, :, 1, 2])
          and np.allclose(o3.slice("vorticity", 2, 1, "y"), 2 * w3[:, :, 1, 1])
          and np.allclose(o3.slice("vorticity", 2, 1, "z"), -2 * w3[:, :, 1, 0]))

    # the binaries themselves: a rigid rotation (curl = 2 Omega along z) through DTFE-2d and DTFE
    rot_bins = [(dim, rs.REPO_ROOT / ("DTFE-2d" if dim == 2 else "DTFE")) for dim in (2, 3)]
    if all(b.is_file() for _, b in rot_bins):
        sys.path.insert(0, str(ROOT / "tests"))
        from generate_ps_test_data import write_snapshot
        got = {}
        for dim, b in rot_bins:
            m = 24 if dim == 2 else 12
            g = (np.arange(m) + 0.5) * 100.0 / m
            pos = np.stack([x.ravel() for x in np.meshgrid(*([g] * dim), indexing="ij")], 1)
            pos = pos + np.random.default_rng(1).uniform(-0.3, 0.3, pos.shape)
            v = np.zeros_like(pos)
            v[:, 0], v[:, 1] = -2.0 * (pos[:, 1] - 50), 2.0 * (pos[:, 0] - 50)
            if dim == 3:
                v[:, 2] = 0.3 * (pos[:, 0] - 50)                   # dv_z/dx: curl_y = -0.3
            write_snapshot(str(work / f"rot{dim}.hdf5"), pos, pos, v, 100.0, 1.0)
            subprocess.run([str(b), str(work / f"rot{dim}.hdf5"), str(work / f"rot{dim}"), "--grid", "8",
                            "--input-format", "105", "--mpc-unit", "1", "--box", *(["0", "100"] * dim),
                            "--field", "vorticity", "gradient", "--verbose", "1"], check=True, capture_output=True)
            ro = G.OutputSet(work / f"rot{dim}", dim=dim)
            got[dim] = ([ro.slice("vorticity")[4, 4]] if dim == 2 else
                        [ro.slice("vorticity", 2, 4, k)[4, 4] for k in "xyz"]) + \
                       [ro.slice("gradient", 2, 4, "xy")[4, 4]]
        check("the binaries' grids on a rigid rotation: the map shows the curl (2D: 4; 3D: 0, -0.3, 4) and "
              "gradient xy = dv_x/dy = -2", np.allclose(got[2], [4.0, -2.0], atol=1e-3)
              and np.allclose(got[3], [0.0, -0.3, 4.0, -2.0], atol=1e-3), str(got))
    else:
        print("   SKIP  the rigid-rotation check (DTFE-2d or DTFE not built)")

    # real outputs at the 64^2 == 16^3 tie, no sidecar: density, stream counts, all scalar
    tie_runs = ((2, "PS-DTFE-2d", 48, 64), (3, "PS-DTFE", 16, 16))
    if all((rs.REPO_ROOT / b).is_file() for _, b, _, _ in tie_runs):
        dims = {}
        for dim, b, n_, grid in tie_runs:
            snap_ = work / f"tie{dim}.hdf5"
            subprocess.run([sys.executable, str(ROOT / "tests" / "generate_ps_test_data.py"), "--out", str(snap_),
                            "--n", str(n_), "--box", "100", "--amplitude-factor", "1.2", "--crossed-waves",
                            "--dim", str(dim)], check=True, capture_output=True)
            subprocess.run([str(rs.REPO_ROOT / b), str(snap_), str(work / f"tie{dim}"), "--grid", str(grid),
                            "--periodic", "--input-format", "105", "--mpc-unit", "1", "--field", "density", "--verbose", "1"],
                           check=True, capture_output=True)
            t = G.OutputSet(work / f"tie{dim}")
            dims[dim] = (t.dim, t.n)
        check("real scalar-only outputs at the 64^2 == 16^3 tie (no sidecar): PS-DTFE-2d reads as 2D, PS-DTFE as 3D",
              dims == {2: (2, 64), 3: (3, 16)}, str(dims))
    else:
        print("   SKIP  the 64^2 == 16^3 tie check (PS-DTFE-2d or PS-DTFE not built)")

    log = work / "r2.runlog"
    _write_log(log, "Build: PS-DTFE-2d 3e0c1a65fd-dirty (built 2026-10-02T21:34:55Z)\n"
               "RUNNING: /x/PS-DTFE-2d in.hdf5 r2 --grid 64 --partition 2 2 --field density_a\n")
    rec = R.parse_runlog(log)
    check("run records: the 2D binary, its 2-number split, '²' in the run list's words",
          rec.dim == 2 and rec.partitions == 4 and rec.partition == 2 and "2² partitions" in rec.options_text()
          and "2D" in rec.options_text(), f"{rec.binary} {rec.partitions} {rec.options_text()}")
    _write_log(log, "Build: DTFE 3e0c1a65fd-dirty (built 2026-10-01T10:00:00Z)\n"
               "RUNNING: /x/DTFE in.hdf5 r2 --grid 64 --periodic --CIC --interlace\n")
    old = R.stale_reasons(R.parse_runlog(log))
    # built after the interlace-pass2 / cic-faces fixes too (cpp_fix_flags has the window)
    _write_log(log, "Build: DTFE 3e0c1a65fd-dirty (built 2026-10-12T08:00:00Z)\n"
               "RUNNING: /x/DTFE in.hdf5 r2 --grid 64 --periodic --CIC --interlace\n")
    new = R.stale_reasons(R.parse_runlog(log))
    check("stale outputs: an interlaced grid made before the phase fix is flagged, a newer one is not",
          any(lvl == "error" and "interlaced" in m for lvl, m in old) and not any("interlaced" in m for _, m in new),
          f"{old} | {new}")


def run_outputs_not_inputs():
    """A custom run named after its input (the default) in the input's own folder: 'X.hdf5' beside 'X.a_den'.
    The input is never one of the run's outputs -- the Runs browser's Move to Trash takes the outputs."""
    import json
    import results as R
    d = Path(tempfile.mkdtemp(prefix="gui_runs_"))
    try:
        for name in ("X.hdf5", "X.a_den", "X.den", "X.txt", "Y.txt", "r[1].a_den", "r1.a_den"):
            (d / name).write_bytes(b"\0" * 8)
        (d / "X.gui.json").write_text(json.dumps({"input_file": str(d / "X.hdf5")}))
        (d / "X.runlog").write_text(f"RUNNING: /x/PS-DTFE {d / 'X.hdf5'} X --grid 16\ntotal wall time : 1s\n")
        (d / "Y.runlog").write_text(f"RUNNING: /x/DTFE {d / 'Y.txt'} Y --grid 16\ntotal wall time : 1s\n")
        (d / "r[1].runlog").write_text("RUNNING: /x/DTFE in.hdf5 r[1] --grid 16\ntotal wall time : 1s\n")
        names = lambda log: [p_.name for p_ in R.parse_runlog(d / log).outputs]          # noqa: E731
        check("a run's outputs never include its input snapshot, nor a file its command line reads (Move to Trash)",
              names("X.runlog") == ["X.a_den", "X.den", "X.txt"] and names("Y.runlog") == [],
              f"{names('X.runlog')} {names('Y.runlog')}")
        check("... and a prefix with '[' matches only its own files", names("r[1].runlog") == ["r[1].a_den"],
              str(names("r[1].runlog")))
        sp = d / "My Runs"                    # the log prints paths unquoted: a space must not hide an input
        sp.mkdir()
        for name in ("halo.hdf5", "halo.pts", "halo.a_den"):
            (sp / name).write_bytes(b"\0" * 8)
        (sp / "halo.runlog").write_text(f"RUNNING: /x/PS-DTFE {sp / 'halo.hdf5'} {sp / 'halo'} --sample-points "
                                        f"{sp / 'halo.pts'} --grid 16\ntotal wall time : 1s\n")
        got = [p_.name for p_ in R.parse_runlog(sp / "halo.runlog").outputs]
        check("... nor an input in a folder whose path has a space (the sample-points file of 'My Runs/halo')",
              got == ["halo.a_den"], str(got))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def point_window_linear(tmp: Path):
    """Sample points for Custom and standard-DTFE runs, the PS window, the deposits' pins, and the velocity
    derivatives' route per quality (the binary options)."""
    pts = tmp / "pts.txt"
    pts.write_text("1 2 3\n4 5 6\n")
    snap = tmp / "TNG100" / "TNG100-3-Dark" / "snapdir_099" / "combined_099.hdf5"
    c = rs.CustomSpec(input_file=str(snap), sample_points=str(pts), window=[0, 10, 0, 10, 0, 1])
    a = c.command()
    check("a Custom run: --sample-points (+ the velocity gradient), the PS window; never the linear deposit (neither "
          "quality has it)",
          a[a.index("--sample-points") + 1] == str(pts) and "--pts-vel-grad" in a and "--ps-linear-deposit" not in a
          and "--ps-linear-deposit" not in rs.CustomSpec(input_file=str(snap), quality="fast").command()
          and a[a.index("--grid-window") + 1:a.index("--grid-window") + 7] == ["0", "10", "0", "10", "0", "1"], str(a))
    errs = lambda spec: [m for lv, m in spec.problems() if lv == "error"]            # noqa: E731
    dt = rs.CustomSpec(input_file=str(snap), estimator="dtfe", sample_points=str(pts), window=[0, 1, 0, 1, 0, 1])
    webw = rs.CustomSpec(input_file=str(snap), fields=["density_a", "tweb_a"], window=[0, 1, 0, 1, 0, 1])
    bad = rs.CustomSpec(input_file=str(snap), window=[float("nan")])
    notes = [m for lv, m in dt.problems() if lv == "info"]
    check("... refused where the binary would refuse: a window with a web class, an unreadable window; standard DTFE "
          "points are fine; a window left from PS-DTFE is ignored by a standard-DTFE run (a note, not a block: its "
          "field is greyed there)",
          not any("triangulation" in m for m in errs(dt)) and not any("window" in m for m in errs(dt))
          and any("ignores it" in m for m in notes)
          and any("whole periodic grid" in m for m in errs(webw))
          and any("window needs 6 numbers" in m for m in errs(bad)), f"{errs(dt)} {errs(webw)} {errs(bad)}")
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99])
    env = rs.RunSpec(estimator="dtfe", slice_plane=str(pts), **base).command()[1]
    check("the Grids job: a hi-res slice for standard DTFE too (SAMPLE_POINTS); PS_LINEAR_DEPOSIT spelled out, 0 in "
          "both qualities",
          env.get("SAMPLE_POINTS") == str(pts)
          and rs.RunSpec(**base).command()[1].get("PS_LINEAR_DEPOSIT") == "0"
          and rs.RunSpec(quality="fast", **base).command()[1].get("PS_LINEAR_DEPOSIT") == "0", str(env))
    # the velocity derivatives' route (PS-DTFE, docs/cleanup-plan.md D13), the quality's: FACE_GRADIENT spelled out
    # on the Grids and Pipeline jobs -- 1 (--face-gradient) at Exact where the job qualifies, EMPTY (the binary's
    # default: it keeps the stream average and says why) where it does not (Fast: the sampled deposit; the halo
    # release; 2D) -- and the matching flag on a 3D PS-DTFE Custom command; a standard-DTFE job pins it EMPTY (its
    # face gradient is the binary's default whatever the cell average)
    pbase = dict(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, grid=128)
    fg_pipe = rs.PipelineSpec(**pbase)
    fg_own = lambda **k: rs.CustomSpec(input_file=str(snap), **k).command()          # noqa: E731
    routes = lambda argv: [a for a in argv if a in ("--face-gradient", "--per-tet-gradient")]   # noqa: E731
    fg_env = lambda spec: spec.command()[1].get("FACE_GRADIENT")                    # noqa: E731
    check("face gradient: FACE_GRADIENT spelled out on the Grids and Pipeline jobs (and the Pipeline's memory check): "
          "1 at Exact, empty at Fast (the sampled deposit does not qualify); empty for standard DTFE in both. The "
          "Custom command the same way for PS-DTFE in 3D ('--face-gradient' at Exact), no route flag at Fast, with the "
          "halo release, in 2D, for standard DTFE, nor beside extra options that name a route (theirs is the run's)",
          fg_env(rs.RunSpec(**base)) == "1" and fg_env(rs.RunSpec(quality="fast", **base)) == ""
          and fg_env(fg_pipe) == "1" and fg_env(rs.PipelineSpec(quality="fast", **pbase)) == ""
          and fg_pipe.check_step("TNG100-3-Dark", [99], 64).env.get("FACE_GRADIENT") == "1"
          and fg_env(rs.RunSpec(estimator="dtfe", **base)) == ""
          and fg_env(rs.RunSpec(estimator="dtfe", quality="fast", **base)) == ""
          and routes(fg_own()) == ["--face-gradient"]
          and not routes(fg_own(quality="fast")) and not routes(fg_own(halo_release=2.0))
          and not routes(fg_own(estimator="dtfe")) and not routes(fg_own(estimator="dtfe", quality="fast"))
          and not routes(fg_own(dim=2)) and not routes(fg_own(dim=2, quality="fast"))
          and routes(fg_own(extra_args="--per-tet-gradient")) == ["--per-tet-gradient"]
          and routes(fg_own(quality="fast", extra_args="--face-gradient")) == ["--face-gradient"],
          f"{fg_pipe.command()[1]} {fg_own()}")
    # a standard-DTFE Grids job pins run_dtfe.sh's FACE_GRADIENT EMPTY, its memory check too, as it pins SUBSAMPLES
    # and THREADS: empty is the driver's default (the binary's own route, the face gradient), so the command is the one
    # a clean shell runs, and a FACE_GRADIENT exported in the login shell (0: '--per-tet-gradient', 1: an explicit
    # '--face-gradient') does not reach the job, whose environment is laid over the launcher's
    std_jobs = [rs.RunSpec(estimator="dtfe", **base), rs.RunSpec(estimator="dtfe", quality="fast", **base)]
    std_envs = [s.command()[1] for s in std_jobs] + [s.check_step().env for s in std_jobs]
    leaked = [dict({"FACE_GRADIENT": "0"}, **e).get("FACE_GRADIENT") for e in std_envs]
    std_custom = [rs.CustomSpec(input_file=str(snap), estimator="dtfe", quality=q) for q in rs.QUALITIES]
    std_custom_envs = [c.run_env() for c in std_custom] + [c.check_step().env for c in std_custom]
    check("a standard-DTFE Grids job pins FACE_GRADIENT empty (run_dtfe.sh's default: the binary's face gradient), in "
          "the memory check too and in both qualities, so an exported FACE_GRADIENT cannot leak in; the binary's own "
          "DTFE_FACE_GRADIENT the same way (a standard Custom run's too: install_gui_app.sh imports every DTFE_* from "
          "the login shell); the command names no gradient route",
          all(e.get("FACE_GRADIENT") == "" for e in std_envs) and leaked == [""] * len(std_envs)
          and all(e.get("DTFE_FACE_GRADIENT") == "" for e in std_envs + std_custom_envs)
          and not any(a in ("--face-gradient", "--per-tet-gradient") for s in std_jobs + std_custom
                      for a in s.command()[0 if isinstance(s, rs.RunSpec) else slice(None)]),
          str([{k: e.get(k) for k in ("FACE_GRADIENT", "DTFE_FACE_GRADIENT", "SUBSAMPLES", "THREADS")}
               for e in std_envs + std_custom_envs]))
    # a PS-DTFE job pins the binary's DTFE_FACE_GRADIENT EMPTY too (read as unset), on the Grids, Pipeline and Custom
    # jobs and their memory checks, in both qualities: where a job does not qualify (Fast, the halo release, 2D) the
    # driver passes no route flag, and a DTFE_FACE_GRADIENT=1 exported in the login shell would make the binary refuse
    # the job the note said keeps the stream average
    ps_jobs = [rs.RunSpec(quality=q, **base) for q in rs.QUALITIES]
    ps_pipes = [rs.PipelineSpec(quality=q, **pbase) for q in rs.QUALITIES]
    ps_custom = [rs.CustomSpec(input_file=str(snap), **k) for k in (
        {}, {"quality": "fast"}, {"halo_release": 2.0}, {"dim": 2}, {"dim": 2, "quality": "fast"})]
    ps_envs = ([s.command()[1] for s in ps_jobs] + [s.check_step().env for s in ps_jobs]
               + [p.command()[1] for p in ps_pipes] + [p.check_step("TNG100-3-Dark", [99], 64).env for p in ps_pipes]
               + [c.run_env() for c in ps_custom] + [c.check_step().env for c in ps_custom])
    ps_leaked = [dict({"DTFE_FACE_GRADIENT": "1"}, **e).get("DTFE_FACE_GRADIENT") for e in ps_envs]
    check("a PS-DTFE job (Grids, Pipeline, Custom, and their memory checks) pins the binary's DTFE_FACE_GRADIENT "
          "empty in both qualities, so an exported DTFE_FACE_GRADIENT=1 cannot make the binary refuse a job that "
          "does not qualify; the route itself stays FACE_GRADIENT's (Grids, Pipeline) or the command's (Custom)",
          all(e.get("DTFE_FACE_GRADIENT") == "" for e in ps_envs) and ps_leaked == [""] * len(ps_envs),
          str([e.get("DTFE_FACE_GRADIENT") for e in ps_envs]))
    # ... and what the quality leaves out is said before the run, as the binary says it (it goes on and says why):
    # one info line each, no error or warning (the Checks list's lines of the two modes). Only these messages: the
    # fake snapshot has other problems of its own
    said = lambda spec, lv, word: [m for lvl, m in spec.problems() if lvl == lv and word in m]   # noqa: E731
    deriv = lambda spec, lv="info": said(spec, lv, "stream average")                             # noqa: E731
    caust = lambda spec, lv="info": said(spec, lv, "fold cells")                                 # noqa: E731
    grad_fields = ["density_a", "velocity_a", "divergence_a"]
    fast_runs = [rs.RunSpec(quality="fast", **base), rs.PipelineSpec(quality="fast", **pbase),
                 rs.CustomSpec(input_file=str(snap), quality="fast", fields=grad_fields)]
    exact_out = [rs.CustomSpec(input_file=str(snap), halo_release=2.0, fields=grad_fields),
                 rs.CustomSpec(input_file=str(snap), dim=2, fields=grad_fields)]
    check("Checks list, Fast (PS-DTFE): 'Fast: PS-DTFE's velocity derivatives are the stream average' where a field "
          "built on the velocity gradient is ticked (Grids, Pipeline, Custom); Exact with the halo release or in 2D: "
          "'Exact with halo release, or in 2D: ... stream average; no mass-flux divergence'; each one info line",
          all(deriv(spec) == ["Fast: PS-DTFE's velocity derivatives are the stream average"] for spec in fast_runs)
          and all(deriv(spec) == ["Exact with halo release, or in 2D: the velocity derivatives are the stream "
                                  "average; no mass-flux divergence"] for spec in exact_out)
          and not any(deriv(spec, lv) for spec in fast_runs + exact_out for lv in ("error", "warning")),
          str([deriv(spec) for spec in fast_runs + exact_out]))
    check("Checks list, Fast with caustics (PS-DTFE): 'the classes are recorded per sample point; some fold cells "
          "show none', one info line (Grids and Pipeline: caustics on by default; Custom with them ticked); nothing "
          "at Exact or without caustics",
          all(caust(spec) == ["Fast with caustics: the classes are recorded per sample point; some fold cells show "
                              "none"] for spec in fast_runs[:2]
              + [rs.CustomSpec(input_file=str(snap), quality="fast", caustics=True)])
          and not caust(rs.CustomSpec(input_file=str(snap), quality="fast"))
          and not caust(rs.RunSpec(quality="fast", caustics=False, **base))
          and not caust(rs.RunSpec(**base)) and not caust(rs.PipelineSpec(**pbase))
          and not caust(rs.RunSpec(estimator="dtfe", quality="fast", **base)),
          str([caust(spec) for spec in fast_runs]))
    no_grad = rs.RunSpec(quality="fast", fields=["density_a"], **base)
    prod = [rs.RunSpec(**base), fg_pipe, rs.CustomSpec(input_file=str(snap), fields=grad_fields),
            rs.RunSpec(estimator="dtfe", quality="fast", **base)]
    named = rs.CustomSpec(input_file=str(snap), quality="fast", fields=grad_fields,
                          extra_args="--face-gradient")     # the extra options name the route: the binary decides
    check("... nothing said of the derivatives where no field built on the velocity gradient is ticked (the route "
          "changes nothing), at Exact in 3D (the production settings), for standard DTFE (its derivatives stay "
          "exact), or on a Custom run whose extra options name the route",
          not any(deriv(spec, lv) for spec in [no_grad, named] + prod for lv in ("error", "warning", "info")),
          f"{no_grad.problems()} {[spec.problems() for spec in prod]} {named.problems()}")
    check("... the memory check is kept per quality (the binary's tuner books the face sums of Exact)",
          rs.RunSpec().memory_key() != rs.RunSpec(quality="fast").memory_key()
          and fg_pipe.memory_key() != rs.PipelineSpec(quality="fast", **pbase).memory_key()
          and rs.CustomSpec().memory_key() != rs.CustomSpec(quality="fast").memory_key())
    # the route Exact asks for is the binary's own PS-DTFE default (src/user_options.cc kPsFaceGradientDefault): the
    # launcher names it, as the binary would take it, wherever a run qualifies; a saved job's face-gradient tick
    # (older than the two modes) is dropped, whatever it held: the route is the quality's
    import re
    kdef = re.search(r"^static constexpr bool kPsFaceGradientDefault\s*=\s*(true|false)\s*;",
                     (ROOT / "src" / "user_options.cc").read_text(), re.M)
    saved = {"Grids": (rs.RunSpec, rs.RunSpec(**base).to_dict()),
             "Pipeline": (rs.PipelineSpec, rs.PipelineSpec(**pbase).to_dict()),
             "Custom": (rs.CustomSpec, rs.CustomSpec(input_file=str(snap)).to_dict())}
    loaded = {k: [cls.from_dict(dict(d, face_gradient=t)) == cls.from_dict(d) for t in (True, False)]
              for k, (cls, d) in saved.items()}
    check("... Exact's route is the binary's PS-DTFE default (src/user_options.cc kPsFaceGradientDefault) on the "
          "Grids, Pipeline and Custom jobs; a saved job's face-gradient tick, on or off, loads as the same job",
          kdef is not None and rs.quality_mode("exact").face_gradient == (kdef.group(1) == "true")
          and all(cls().mode().face_gradient == (kdef.group(1) == "true")
                  for cls in (rs.RunSpec, rs.PipelineSpec, rs.CustomSpec))
          and all(v == [True, True] for v in loaded.values()), str(loaded))
    nodt = [m for lv, m in rs.CustomSpec(input_file=str(snap), estimator="dtfe", fields=["dispersion_a"]).problems()
            if lv == "error"]
    check("a Custom standard-DTFE run with only the dispersion ticked is refused (it would pass a bare --field)",
          any("no field standard DTFE computes" in m for m in nodt), str(nodt))


_HELP_OPTIONS: dict = {}


def help_options(script: Path) -> set:
    """The option names a python script's --help lists, as whole words: '--smooth' in the text of '--smooth-px' or
    '--size-rv' in '--size-rv-definition' is no match (a substring test would pass a renamed flag's old spelling,
    which --help hides). Cached per script."""
    import re
    import subprocess as _sp
    if script not in _HELP_OPTIONS:
        out = _sp.run([rs.PYTHON, str(script), "--help"], capture_output=True, text=True,
                      cwd=Path(__file__).resolve().parents[1] / "python",
                      env=dict(os.environ, MPLBACKEND="Agg", DTFE_SIM=os.environ.get("DTFE_SIM", "TNG100-3-Dark")))
        _HELP_OPTIONS[script] = set(re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", out.stdout))
    return _HELP_OPTIONS[script]


def saved_flag_migration(tmp: Path):
    """P6.7: the launcher's saved state written with the older flag names and values loads with the new ones, on every
    load path (CustomSpec / PlotSpec.from_dict: the settings, a queued job, a run's '.gui.json' sidecar); idempotent."""
    import runspec_custom as RC
    old = {"input_file": str(tmp / "x.hdf5"), "estimator": "ps", "stream_density": "dtfe",
           "extra_args": "--MpcUnit 1 --ps-window 0 5 0 5 0 5 --ordered-merge -s 27 --lambda_th=0.4"}
    c = rs.CustomSpec.from_dict(old)
    check("a saved Custom job's extra options load with the new names ('--MpcUnit 1 --ps-window ...' as "
          "'--mpc-unit 1 --grid-window ...', the merge switch as '--ps-merge'); PS-DTFE keeps '-s' (no effect there); "
          "its stream density (a setting the two modes removed) is dropped",
          c.extra_args == "--mpc-unit 1 --grid-window 0 5 0 5 0 5 --ps-merge index-order -s 27 --lambda-th=0.4"
          and not hasattr(c, "stream_density") and "--ps-stream-density" not in c.command(), c.extra_args)
    std = rs.CustomSpec.from_dict(dict(old, estimator="dtfe", extra_args="--NGP --regionMpc 0 1 0 1 0 1 -s 27 -m 3"))
    flat = rs.CustomSpec.from_dict(dict(old, estimator="dtfe", dim=2, extra_args="--samples 16 --paddingMpc 2"))
    check("... standard DTFE's too ('-s 27' -> '--subsamples 3', k^dim; '-m' -> '--average-method')",
          std.extra_args == "--ngp --region-mpc 0 1 0 1 0 1 --subsamples 3 --average-method 3"
          and flat.extra_args == "--subsamples 4 --padding-mpc 2", f"{std.extra_args!r} {flat.extra_args!r}")
    again = rs.CustomSpec.from_dict(c.to_dict())
    q = rs.QueuedJob.from_dict({"kind": "custom", "spec": old, "title": "x"})
    check("... idempotent, and a queued job's saved spec migrates the same way",
          again.to_dict() == c.to_dict() and RC.migrate_extra_args("") == ""
          and q.build().extra_args == c.extra_args, "")
    pl = rs.PlotSpec.from_dict({"options": {"voidenv": {"criterion": "density_raw"}, "panels": {"project": "mean"}}})
    check("saved plot options load with the scripts' values (criterion density_raw -> density-raw, panels' 'mean' -> "
          "'slab'), which the option choices offer",
          pl.opt("voidenv", "criterion") == "density-raw" and pl.opt("panels", "project") == "slab"
          and "density-raw" in rs.ENV_CRITERIA and "density_raw" not in rs.ENV_CRITERIA
          and next(o.choices for o in rs.FIGURE_SET["panels"].opts if o.key == "project") == ("plane", "slab"),
          str(pl.options))


def launcher_python_flags(tmp: Path):
    """Every python command the launcher builds (each Plots set with non-default options, the pipeline's render step,
    the Data tab's merge steps) passes only options its script's --help lists, matched as whole words: no old
    spelling (P6.7d's renames are hidden from --help) and no flag the script does not have."""
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[50, 99])
    opts = {"pointeval": {"fields": "density, velDiv", "project": "slab", "smooth_px": 2.0, "smooth_deriv": 4.0,
                          "fixed_range": True, "force": True},
            "panels": {"plane": str(tmp / "p.json"), "panels": "density sheet", "project": "slab", "vmin": 0.1},
            "voidpop": {"size_rv": "ridge", "size_smooth": 3.0, "smooth_mpc": 2.0, "keep_overlaps": True,
                        "size_fixed_r": True, "fixed_r_smooth_mpc": "2.1625"},
            "voidtrack": {"link": "particles", "exclusive": True, "radius": "r_eff"},
            "voidfate": {"ref_snap": 50, "exclusive": True}, "voidconv": {"rv_range": "2 6", "factors": "2 4 8"},
            "voidenv": {"criterion": "density-raw", "factor": 3.0, "r_v_definition": "delta08"},
            "voidevol": {"rv_definition": "ridge"}, "pk": {"nbins": 0, "ref": False},
            "skeleton": {"voids": True, "null_draws": 0, "all_caustic": True},
            "crossbox": {"smooth_mpc": 3.0, "z": "0 2", "n_boot": 10, "replot": True},
            "halotrack": {"subhalos": "0, 17", "environment": False, "min_ratio": 0.2},
            "core": {"mode": "export"}}
    spec = rs.PlotSpec(sets=[fs.key for fs in rs.FIGURE_SETS], method="ps", prefix="ps_x", options=opts,
                       figures_root=str(tmp / "figs"), **base)
    steps = spec.steps() + [rs.PlotSpec(sets=["halotrack"], **base).steps()[0]]
    steps += [st for st in rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], render_fields="density velDiv",
                                           render_smooth=1.0, render_smooth_derivatives=0.0, planes=4,
                                           render_project="slab", output_prefix="ps_y").steps()[1:]]
    steps += rs.DataSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], merge_snapshots=True,
                         merge_groupcats=True).steps()
    bad, seen = [], set()
    for st in steps:
        if len(st.argv) < 2 or not st.argv[1].endswith(".py"):
            continue
        script = Path(st.argv[1])
        seen.add(script.name)
        known = help_options(script)
        bad += [f"{script.name} {f}" for f in st.argv[2:] if f.startswith("-") and not f.lstrip("-")[:1].isdigit()
                and f not in known]
    check("every python flag the launcher emits is one its script's --help lists (whole words; no old spelling)",
          not bad and {"plot_pointeval.py", "plot_pointeval_panels.py", "plot_void_crossbox.py", "void_cache.py",
                       "analyze.py", "plot_halo_tracking.py", "merge_HDF5.py"} <= seen, f"{bad} {sorted(seen)}")


def plot_options(tmp: Path):
    """The Plots tab: the core scripts as a checklist (= analyze.py's PLOT_SCRIPTS),
    the point-eval gradient smoothing, the publication panels' range, dpi and pixel-exact images."""
    import ast
    import re
    repo = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "python" / "analyze.py").read_text())
    core = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "PLOT_SCRIPTS" for t in n.targets))
    deriv = float(re.search(r"^DERIVATIVE_SMOOTH = ([0-9.]+)", (repo / "python" / "dtfelib" / "pointeval.py").read_text(),
                            re.M).group(1))
    check("the core checklist is analyze.py's PLOT_SCRIPTS; the point-eval default is "
          "dtfelib's", list(rs.CORE_SCRIPTS) == core
          and rs.POINTEVAL_DERIV_SMOOTH == deriv, f"{rs.CORE_SCRIPTS} vs {core}; {deriv}")
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99])
    plane = str(tmp / "TNG100" / "TNG100-3-Dark" / "hires_plane_z64.json")
    spec = rs.PlotSpec(sets=["core", "pointeval", "panels"], **base,
                       options={"core": {"mode": "plot", "only": "plot_DTFE.py,plot_phi_delta.py"},
                                "pointeval": {"smooth_deriv": 4.0},
                                "panels": {"plane": plane, "vmin": 0.05, "vmax": 300.0, "dpi": 150, "image": True}})
    a_th = next(st.argv for st in spec.steps() if "analyze.py" in st.label)
    a_pe = next(st.argv for st in spec.steps() if "plot_pointeval.py" in " ".join(st.argv))
    a_pa = next(st.argv for st in spec.steps() if "plot_pointeval_panels.py" in " ".join(st.argv))
    check("... the ticked core scripts reach --only; the gradient smoothing reaches "
          "--smooth-derivatives-px; the panels get --vmin/--vmax/--dpi/--image",
          a_th[a_th.index("--only") + 1:] == ["plot_DTFE.py", "plot_phi_delta.py"]
          and a_pe[a_pe.index("--smooth-derivatives-px") + 1] == "4" and a_pa[a_pa.index("--vmin") + 1] == "0.05"
          and a_pa[a_pa.index("--vmax") + 1] == "300" and a_pa[a_pa.index("--dpi") + 1] == "150" and "--image" in a_pa,
          f"{a_th} | {a_pe} | {a_pa}")
    plain = rs.PlotSpec(sets=["pointeval", "panels"], **base, options={"panels": {"plane": plane}})
    flat = " ".join(" ".join(st.argv) for st in plain.steps())
    check("... and the defaults add none of those flags", not any(f in flat for f in ("--smooth-derivatives-px", "--vmin",
                                                                                     "--dpi", "--image")), flat)


def plot_void_sets(tmp: Path):
    """The Plots tab's void and phase-space sets: void profiles, streams by web environment, P(k) and the
    caustic skeleton, with the void radius, overlapping voids and the smoothing in Mpc; every flag one the script's own
    parser knows; the core set stays last (the app test picks it as the list's last item)."""
    repo = Path(__file__).resolve().parents[1]
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[50, 99])
    sets = ["voidpop", "voidprof", "webstreams", "pk", "skeleton"]
    plain = {st.argv[1].split("/")[-1]: st.argv for st in rs.PlotSpec(sets=sets, **base).steps()}
    flat = " ".join(" ".join(a) for a in plain.values())
    check("the void sets: one call each over the selected snapshots, the defaults adding no option flag (the skeleton "
          "its per-void statistics); the core set still last",
          len(plain) == 5 and all(a[a.index("--snaps") + 1:a.index("--snaps") + 3] == ["50", "99"]
                                  for k, a in plain.items() if k != "plot_void_population.py")
          and not any(f in flat for f in ("--radius", "--keep-overlaps", "--smooth-mpc", "--web", "--n-bins", "--ref-snap",
                                          "--sample"))
          and "--per-void" in plain["plot_caustic_skeleton.py"] and rs.FIGURE_SETS[-1].key == "core", flat)
    opts = {"voidpop": {"keep_overlaps": True, "smooth_mpc": 2.1625},
            "voidprof": {"radius": "r_eff", "sample": "deep", "keep_overlaps": True, "smooth_mpc": 2.1625},
            "webstreams": {"web": "tweb"}, "pk": {"nbins": 0, "ref": False},
            "skeleton": {"voids": False, "radius": "r_eff"}}
    got = {st.argv[1].split("/")[-1]: st.argv for st in rs.PlotSpec(sets=sets, options=opts, **base).steps()}
    def has(name, *flags):
        a = got[name]
        return all(f in a and (i + 1 == len(flags) or True) for i, f in enumerate(flags))
    pv, pp = got["plot_void_profiles.py"], got["plot_void_population.py"]
    check("... the options reach their flags (radius, deep sample, overlaps, smoothing in Mpc, web, k bins, no same-box "
          "panel, no per-void block)",
          pv[pv.index("--radius") + 1] == "r_eff" and pv[pv.index("--sample") + 1] == "deep" and "--keep-overlaps" in pv
          and pv[pv.index("--smooth-mpc") + 1] == "2.1625" and "--keep-overlaps" in pp and "--smooth-mpc" in pp
          and has("plot_web_streams.py", "--web", "tweb") and got["plot_pk_compare.py"][-2:] in (["--ref-snap", "-1"],)
          and "--n-bins" in got["plot_pk_compare.py"] and "--per-void" not in got["plot_caustic_skeleton.py"]
          and "--radius" in got["plot_caustic_skeleton.py"], str(got))
    unknown = [f"{name} {f}" for name, a in got.items() for f in a
               if f.startswith("--") and f not in help_options(repo / "python" / "plot" / name)]
    check("... and every flag is one the script's own parser knows (--help)", not unknown, str(unknown))
    both = rs.PlotSpec(sets=["voidprof"], smooth=5.0, options={"voidprof": {"smooth_mpc": 2.0}}, **base)
    sk = rs.PlotSpec(sets=["skeleton"], **base)
    check("... the smoothing in cells AND in Mpc is an error; the skeleton without caustic grids a warning",
          any(lvl == "error" and "cells and in Mpc" in m for lvl, m in both.problems())
          and any(lvl == "warning" and "causticClass" in m for lvl, m in sk.problems()),
          f"{both.problems()} | {sk.problems()}")


# (set key, option key) -> (script, the script's constant that is its default, or a literal)
_VOID_SCRIPT_DEFAULTS = {
    ("voidpop", "size_smooth"): ("plot_void_population", "SIZE_SMOOTH_CELLS"),
    ("voidpop", "size_rv"): ("plot_void_population", "SIZE_FUNCTION_RV"),
    ("voidpop", "fixed_r_smooth"): ("plot_void_population", "FIXED_R_SMOOTH_CELLS"),
    ("voidfate", "ref_snap"): ("plot_void_fate", "REF_SNAP"),
    ("voidconv", "rv_range"): ("plot_void_convergence", "DEFAULT_RV_RANGE"),
    ("skeleton", "null_draws"): ("plot_caustic_skeleton", "NULL_DRAWS"),
}


def plot_void_script_options(tmp: Path):
    """The void scripts' own options on the Plots tab: population size function, tracking link,
    fate links, convergence factors, environment factor and R_v definition, profile-evolution R_v definition,
    skeleton null and wall cuts. Defaults equal the scripts' own and add no flag; every set value reaches its
    flag, each one the script's parser knows; a malformed list is an error, not a crash in the script."""
    import subprocess as _sp
    repo = Path(__file__).resolve().parents[1]
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[50, 99])
    sets = ["voidpop", "voidtrack", "voidfate", "voidconv", "voidenv", "voidevol", "skeleton"]
    new = {"voidpop": ("size_smooth", "size_rv"), "voidtrack": ("link", "exclusive"),
           "voidfate": ("exclusive", "ref_snap"), "voidconv": ("factors", "match_sigma", "rv_range"),
           "voidenv": ("factor", "r_v_definition"), "voidevol": ("rv_definition",),
           "skeleton": ("null_draws", "min_component", "min_component_mass", "sheet_min_density", "all_caustic")}
    have = {k: {o.key for o in rs.FIGURE_SET[k].opts} for k in sets}
    missing = [f"{k}.{o}" for k, os_ in new.items() for o in os_ if o not in have[k]]
    check("the void scripts' options are Plots-tab options, each with a tooltip and no '--' in its text",
          not missing and all(o.help and "--" not in o.help + o.label for k in sets for o in rs.FIGURE_SET[k].opts
                              if o.key in new[k]), str(missing))
    if missing:
        return
    flags = ("--size-smooth", "--size-rv-definition", "--link", "--exclusive", "--ref-snap", "--factors", "--match-sigma",
             "--rv-range", "--factor", "--rv-definition", "--null-draws", "--min-component",
             "--min-component-mass", "--sheet-min-density", "--all-caustic")
    plain = " ".join(" ".join(st.argv) for st in rs.PlotSpec(sets=sets, **base).steps())
    check("... the defaults add none of their flags", not any(f" {f} " in plain + " " for f in flags), plain)
    # the defaults against the scripts' own: dtfelib's constants imported, the plot scripts' read in a subprocess
    # (they import matplotlib)
    sys.path.insert(0, str(repo / "python"))
    from dtfelib import convergence as cv, classify
    code = ("import sys, json; sys.path[:0] = ['plot', '.']; import importlib; out = {}\n"
            + "".join(f"out['{s}.{c}'] = getattr(importlib.import_module('{s}'), '{c}')\n"
                      for s, c in _VOID_SCRIPT_DEFAULTS.values())
            + "print(json.dumps(out))")
    import json
    r = _sp.run([rs.PYTHON, "-c", code], capture_output=True, text=True, cwd=repo / "python",
                env=dict(os.environ, MPLBACKEND="Agg"))
    try:
        consts = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        consts = {}
    d = {(k, o.key): o.default for k in sets for o in rs.FIGURE_SET[k].opts}
    want = {key: consts.get(f"{s}.{c}") for key, (s, c) in _VOID_SCRIPT_DEFAULTS.items()}
    want[("voidconv", "rv_range")] = " ".join(f"{v:g}" for v in want[("voidconv", "rv_range")] or ())
    want[("voidpop", "fixed_r_smooth")] = " ".join(f"{v:g}" for v in want[("voidpop", "fixed_r_smooth")] or ())
    want.update({("voidconv", "factors"): " ".join(map(str, cv.DEFAULT_FACTORS)),
                 ("voidconv", "match_sigma"): cv.DEFAULT_MATCH_SIGMA, ("voidenv", "factor"): classify.ENV_FACTOR,
                 ("voidtrack", "link"): "trees", ("voidenv", "r_v_definition"): "crossing",
                 ("voidevol", "rv_definition"): "crossing"})
    off = {key: (d.get(key), v) for key, v in want.items() if d.get(key) != v}
    check("... and they equal the scripts' own defaults", consts and not off, f"{off} {r.stderr[-300:]}")
    opts = {"voidpop": {"size_smooth": 0.0, "size_rv": "ridge"},
            "voidtrack": {"link": "particles", "exclusive": True},
            "voidfate": {"exclusive": True, "ref_snap": 50},
            "voidconv": {"factors": "2, 4 8", "match_sigma": 1.5, "rv_range": "2 6"},
            "voidenv": {"factor": 3.0, "r_v_definition": "delta08"}, "voidevol": {"rv_definition": "ridge"},
            "skeleton": {"null_draws": 0, "min_component": 100, "min_component_mass": 0.001,
                         "sheet_min_density": 2.5, "all_caustic": True}}
    got = {st.argv[1].split("/")[-1]: st.argv for st in rs.PlotSpec(sets=sets, options=opts, **base).steps()}

    def after(name, flag, n=1):
        a = got.get(name, [])
        return a[a.index(flag) + 1:a.index(flag) + 1 + n] if flag in a else None
    sk = got.get("plot_caustic_skeleton.py", [])
    check("... the set values reach their flags (size smoothing 0 is a value, not the default)",
          after("plot_void_population.py", "--size-smooth") == ["0"]
          and after("plot_void_population.py", "--size-rv-definition") == ["ridge"]
          and after("plot_void_tracking.py", "--link") == ["particles"] and "--exclusive" in got["plot_void_tracking.py"]
          and after("plot_void_fate.py", "--ref-snap") == ["50"] and "--exclusive" in got["plot_void_fate.py"]
          and after("plot_void_convergence.py", "--factors", 3) == ["2", "4", "8"]
          and after("plot_void_convergence.py", "--match-sigma") == ["1.5"]
          and after("plot_void_convergence.py", "--rv-range", 2) == ["2", "6"]
          and after("plot_void_environment.py", "--factor") == ["3"]
          and after("plot_void_environment.py", "--rv-definition") == ["delta08"]
          and after("plot_void_profile_evolution.py", "--rv-definition") == ["ridge"]
          and after("plot_caustic_skeleton.py", "--null-draws") == ["0"]
          and after("plot_caustic_skeleton.py", "--min-component") == ["100"]
          and after("plot_caustic_skeleton.py", "--min-component-mass") == ["0.001"]
          and after("plot_caustic_skeleton.py", "--sheet-min-density") == ["2.5"]
          and "--all-caustic" in sk and "--snaps" not in sk, str(got))
    unknown = [f"{name} {f}" for name, a in got.items() for f in a
               if f.startswith("--") and f not in help_options(repo / "python" / "plot" / name)]
    check("... every flag one the script's own parser knows (--help)", not unknown, str(unknown))
    bad = rs.PlotSpec(sets=["voidconv"], options={"voidconv": {"factors": "2 x", "rv_range": "3"}}, **base)
    allc = rs.PlotSpec(sets=["skeleton"], options={"skeleton": {"all_caustic": True}},
                       **dict(base, snapshots=[]))
    trees = rs.PlotSpec(sets=["voidtrack"], options={"voidtrack": {"exclusive": True}}, **base)
    msgs = " | ".join(m for lv, m in bad.problems() + trees.problems() if lv == "error")
    check("... a malformed factor list or R_v range, one void per particle with the tree link are errors; every "
          "caustic snapshot needs none selected",
          "factors" in msgs and "R_v range" in msgs and "particle link" in msgs
          and not any(lv == "error" and "no snapshots" in m for lv, m in allc.problems()),
          f"{msgs} || {allc.problems()}")


def fixed_r_size_function(tmp: Path):
    """The Void population set's 'size function at fixed R' tick: a step of its own with the script's
    fixed-R flag and its smoothings, run with DTFE_VOID_GATE=none in ITS environment (the script refuses it gated)
    and shown in the exported command; the population step and the job's other steps keep the caller's gate."""
    repo = Path(__file__).resolve().parents[1]
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[50, 99])
    opts_ = {o.key: o for o in rs.FIGURE_SET["voidpop"].opts}
    new = ("size_fixed_r", "fixed_r_smooth", "fixed_r_smooth_mpc")
    check("fixed R: the tick and its smoothings are Void population options on one row, each with a tooltip, no '--'",
          all(k in opts_ for k in new) and len({opts_[k].row for k in new if k in opts_}) == 1
          and all(opts_[k].row and opts_[k].help and "--" not in opts_[k].help + opts_[k].label for k in new
                  if k in opts_), str(sorted(opts_)))
    if not all(k in opts_ for k in new):
        return
    plain = rs.PlotSpec(sets=["voidpop"], method="ps", **base).steps()
    check("... off by default: one population step, no fixed-R flag, no gate in its environment",
          len(plain) == 1 and not any("fixed-r" in a for a in plain[0].argv) and "DTFE_VOID_GATE" not in plain[0].env,
          str([st.argv for st in plain]))
    on = rs.PlotSpec(sets=["voidpop", "voidprof"], method="ps", prefix="ps_x", smooth=0.0,
                     options={"voidpop": {"size_fixed_r": True, "smooth_mpc": 2.1625}}, **base)
    st = on.steps()
    fr = [s for s in st if "--size-fixed-r" in s.argv]
    others = [s for s in st if "--size-fixed-r" not in s.argv]
    a = fr[0].argv if fr else []
    check("... ticked: a step of its own after the population one, the estimator and prefix, the script's default "
          "smoothings (no flag), no population smoothing or snapshots",
          len(fr) == 1 and len(st) == 3 and st[1] is fr[0] and "plot_void_population.py" in a[1]
          and a[2:] == ["--sim", "TNG100-3-Dark", "--estimator", "ps", "--prefix", "ps_x", "--size-fixed-r"]
          and "--smooth-mpc" in st[0].argv, str(a))
    script = rs.export_script(st, "fixed R")
    check("... run ungated: DTFE_VOID_GATE=none in that step's environment and in its exported command; the population "
          "and the other sets' steps keep no gate of the launcher's",
          bool(fr) and fr[0].env.get("DTFE_VOID_GATE") == "none" and "DTFE_VOID_GATE=none" in fr[0].line()
          and script.count("DTFE_VOID_GATE=none") == 1 and fr[0].env.get("DTFE_SIM") == "TNG100-3-Dark"
          and all("DTFE_VOID_GATE" not in s.env for s in others), script)

    def fr_args(cells, mpc):
        s = rs.PlotSpec(sets=["voidpop"], options={"voidpop": {"size_fixed_r": True, "fixed_r_smooth": cells,
                                                               "fixed_r_smooth_mpc": mpc}}, **base)
        argv = s.steps()[-1].argv
        return argv[argv.index("--size-fixed-r") + 1:], [m for lv, m in s.problems() if lv == "error"]
    tng300, mpc_only, cells_only, dflt = (fr_args("3, 2", "2.1625"), fr_args("", "2.1625"), fr_args("5 3", ""),
                                          fr_args("10 5 3", ""))
    check("... the smoothings reach their flags: cells beside Mpc spelled out (the script takes exactly those given), "
          "Mpc alone, other cells alone; the default adds none",
          tng300 == (["--fixed-r-smooth", "3", "2", "--fixed-r-smooth-mpc", "2.1625"], [])
          and mpc_only == (["--fixed-r-smooth-mpc", "2.1625"], []) and cells_only == (["--fixed-r-smooth", "5", "3"], [])
          and dflt == ([], []), f"{tng300} {mpc_only} {cells_only} {dflt}")
    bad, empty, neg = fr_args("10 x", ""), fr_args("", ""), fr_args("0 5", "")
    check("... a word that is no number, no smoothing at all, or one of 0 is an error before the job runs",
          any("positive numbers" in m for m in bad[1]) and any("needs a smoothing" in m for m in empty[1])
          and any("positive numbers" in m for m in neg[1]), f"{bad[1]} {empty[1]} {neg[1]}")
    known = help_options(repo / "python" / "plot" / "plot_void_population.py")
    flags = {f for f in a + tng300[0] if f.startswith("--")}
    check("... every flag one the script's own parser knows (--help)", flags and all(f in known for f in flags),
          str(sorted(f for f in flags if f not in known)))


def crossbox_smoothing_constant():
    """The cross-box smoothing is python/config.py's CROSSBOX_SMOOTHING_MPC (2.1625 when a config lacks it): a
    changed config value is the launcher's default and its unchanged-default test."""
    code = ("import sys; sys.path[:0] = ['python', 'python/gui']; import config; "
            "config.CROSSBOX_SMOOTHING_MPC = 3.25; import runspec as rs; "
            "print(rs.CROSSBOX_SMOOTH_MPC, rs.FIGURE_SET['crossbox'].opts[1].default, "
            "'--smooth-mpc' in rs.PlotSpec(sim='X', sets=['crossbox']).steps()[0].argv)")
    r = subprocess.run([rs.PYTHON, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, DTFE_FIGURES_ROOT=FIGS_TMP))
    check("the cross-box smoothing comes from config.CROSSBOX_SMOOTHING_MPC (default and its no-flag test)",
          r.stdout.split() == ["3.25", "3.25", "False"], r.stdout + r.stderr[-400:])
    code0 = ("import sys; sys.path[:0] = ['python', 'python/gui']; import config\n"
             "if hasattr(config, 'CROSSBOX_SMOOTHING_MPC'): del config.CROSSBOX_SMOOTHING_MPC\n"
             "import runspec as rs; print(rs.CROSSBOX_SMOOTH_MPC)")
    r = subprocess.run([rs.PYTHON, "-c", code0], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, DTFE_FIGURES_ROOT=FIGS_TMP))
    check("... and 2.1625 when the config has no such constant", r.stdout.strip() == "2.1625", r.stdout + r.stderr[-400:])


def gpu_retry_causes(tmp: Path):
    """The GPU deposit retries by cause, from the run log's closing line ('GPU deposit retries: N
    (innocent a, timeout b, oom c, other d)') and from the progress sidecar's 'gpu_retries_by_cause'."""
    import results as R
    d = Path(tempfile.mkdtemp(prefix="gui_retries_"))
    try:
        log = d / "ps_output.runlog"
        log.write_text("RUNNING: /x/PS-DTFE in.hdf5 out --grid 64 --ps-gpu\n  total wall time : 1m 2s\n"
                       "  peak memory (RSS) : 3.5 GB\nGPU deposit retries: 3 (innocent 0, timeout 2, oom 1, other 0)\n")
        rec = R.parse_runlog(log)
        rec2 = R.parse_runlog(log)     # the second read comes from the log cache
        txt = R.retries_text(rec.gpu_retries, rec.gpu_retries_by_cause) if hasattr(R, "retries_text") else ""
        check("a run log's retry line: the total and the causes (the cached read too), the nonzero causes named",
              getattr(rec, "gpu_retries", None) == 3 and not rec.gpu_retried    # no from-scratch restart message
              and getattr(rec, "gpu_retries_by_cause", None) == {"innocent": 0, "timeout": 2, "oom": 1, "other": 0}
              and getattr(rec2, "gpu_retries_by_cause", None) == rec.gpu_retries_by_cause
              and txt == "3 GPU retries (timeout 2, oom 1)", f"{getattr(rec, 'gpu_retries', None)} "
                                                              f"{getattr(rec, 'gpu_retries_by_cause', None)} {txt!r}")
        plain = d / "plain.runlog"
        plain.write_text("RUNNING: /x/PS-DTFE in.hdf5 plain\n  total wall time : 1s\n")
        p = R.parse_runlog(plain)
        side = {"phase": "deposit", "gpu_retries": 2, "gpu_retries_by_cause": {"innocent": 1, "timeout": 0,
                                                                                 "oom": 0, "other": 1}}
        check("... none without the line; a sidecar's causes reach progress_text (its total alone as before)",
              getattr(p, "gpu_retries", None) == 0 and not getattr(p, "gpu_retries_by_cause", None)
              and R.progress_text(side) == "deposit, 2 GPU retries (innocent 1, other 1)"
              and R.progress_text({"phase": "deposit", "gpu_retries": 1}) == "deposit, 1 GPU retries",
              R.progress_text(side))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def explore_extras(tmp: Path):
    """The hi-res slices an output offers, a snapshot chosen by hand for the server, the scalar field, the Custom
    tab's research options."""
    import json
    import numpy as np
    import grids as G
    sim = Path(tempfile.mkdtemp(prefix="gui_extras_")) / "SIM"
    sd = sim / "snapdir_050"
    sd.mkdir(parents=True)
    try:
        np.zeros(8 ** 3, dtype=np.float32).tofile(sd / "ps_output.a_den")
        np.ones(8 ** 3, dtype=np.float32).tofile(sd / "ps_output.a_scalar")
        np.zeros(32 * 32, dtype=np.float64).tofile(sd / "ps_output.pts_den")       # one 32x32 plane
        geo = {"nu": 32, "nv": 32, "planes": 1, "supersample": 1, "axis": "z", "u_axis": "x", "v_axis": "y",
               "u0": 0, "u1": 10, "v0": 0, "v1": 10, "center": 5.0, "box": 10.0}
        (sim / "pointeval_plane_32_z.json").write_text(json.dumps(geo))
        (sim / "pointeval_plane_64_z.json").write_text(json.dumps(dict(geo, nu=64, nv=64)))   # another size: not this one
        o = G.OutputSet(sd / "ps_output", box=((0, 0, 0), (10, 10, 10)))
        planes = G.find_pts_planes(o)
        check("an output's hi-res slices: the plane sidecars whose geometry matches its '.pts_den' (not another size)",
              [Path(pth).name for _, pth in planes] == ["pointeval_plane_32_z.json"] and "z-plane" in planes[0][0],
              str(planes))
        check("the per-particle scalar is an Explore field ('.a_scalar')", "scalar" in o.fields(), str(o.fields()))
        snap = sim / "elsewhere.hdf5"
        snap.write_bytes(b"\0")
        kw0, kw1 = G.server_settings(o), G.server_settings(o, str(snap))
        check("a snapshot chosen by hand serves an output whose own is not found (and only an existing file)",
              kw0 is None and kw1 is not None and kw1["snapshot"] == str(snap)
              and G.server_settings(o, str(sim / "nope.hdf5")) is None, f"{kw0} {kw1}")
    finally:
        shutil.rmtree(sim.parent, ignore_errors=True)
    pts = tmp / "pts.txt"
    pts.write_text("1 2 3\n")
    inp = tmp / "TNG100" / "TNG100-3-Dark" / "snapdir_099" / "combined_099.hdf5"
    c = rs.CustomSpec(input_file=str(inp), sample_points=str(pts), per_stream=True, per_stream_ids=True,
                      halo_release=500)
    a = c.command()
    check("the Custom tab's research options: --ps-halo-release, and for the sample points --ps-per-stream, "
          "--ps-per-stream-ids (the stream density is the binary's default: no --ps-stream-density)",
          a[a.index("--ps-halo-release") + 1] == "500" and "--ps-stream-density" not in a
          and "--ps-per-stream" in a and "--ps-per-stream-ids" in a, str(a))
    nopts = [m for lv, m in rs.CustomSpec(input_file=str(inp), per_stream=True).problems() if lv == "warning"]
    dt = rs.CustomSpec(input_file=str(inp), estimator="dtfe", sample_points=str(pts), per_stream=True, halo_release=500)
    check("... per-stream without sample points warns; a standard-DTFE run leaves the PS-only ones out",
          any("apply to sample points" in m for m in nopts) and "--ps-per-stream" not in dt.command()
          and "--ps-halo-release" not in dt.command(), f"{nopts} {dt.command()}")


def plane_reads():
    """Explore reads a plane of an uncached grid with plain reads, never a memmap (a mapped page of an unplugged
    drive kills the launcher with SIGBUS; a read raises OSError, which the slice worker reports)."""
    import numpy as np
    import grids as G
    d = Path(tempfile.mkdtemp(prefix="gui_planes_"))
    try:
        n, rng, ok = 20, np.random.default_rng(2), True
        for nc, sfx in ((1, "a_den"), (3, "a_vel")):
            cube = rng.standard_normal((n, n, n, nc) if nc > 1 else (n, n, n)).astype(np.float32)
            cube.tofile(d / f"o.{sfx}")
            for axis in range(3):
                for idx in (0, 9, n - 1):
                    sl = [slice(None)] * 3
                    sl[axis] = idx
                    ok &= np.array_equal(G._read_plane(d / f"o.{sfx}", np.float32, n, nc, axis, idx), cube[tuple(sl)])
        gone = False
        try:
            G._read_plane(d / "missing.a_den", np.float32, n, 1, 0, 0)
        except OSError:
            gone = True
        check("a plane read from the file equals the cube's plane on every axis (1 and 3 components); a vanished "
              "file is an OSError, not a crash", ok and gone)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def cube_cap():
    """While a job runs, Explore's cube cache keeps at most RUN_CUBE_BYTES (the binary budgets from free memory)."""
    import numpy as np
    import grids as G
    d = Path(tempfile.mkdtemp(prefix="gui_cubecap_"))
    old = G.RUN_CUBE_BYTES
    try:
        small, big = d / "s.a_den", d / "b.a_den"
        np.zeros(16 ** 3, dtype=np.float32).tofile(small)        # 16 kB
        np.zeros(32 ** 3, dtype=np.float32).tofile(big)          # 128 kB
        G.drop_cubes()
        G.RUN_CUBE_BYTES = 100_000
        G.cap_cubes(True)
        capped = (G._cached_cube(small, np.float32, (16, 16, 16)) is not None,
                  G._cached_cube(big, np.float32, (32, 32, 32)) is None)
        G.cap_cubes(False)
        free = G._cached_cube(big, np.float32, (32, 32, 32)) is not None
        G._cached_cube(small, np.float32, (16, 16, 16))
        G.cap_cubes(True)                                        # a job starts: the cache trimmed to the cap
        held = G.drop_cubes()
        check("Explore's cube cache: capped while a job runs (a small cube cached, a big one memory-mapped), "
              "the cap trims what is held, lifted after", capped == (True, True) and free and held <= 100_000,
              f"{capped} {free} {held}")
    finally:
        G.cap_cubes(False)
        G.RUN_CUBE_BYTES = old
        G.drop_cubes()
        shutil.rmtree(d, ignore_errors=True)


def cube_release():
    """One release for the launcher's cached cubes: the 3D view's averaged and prepared cubes are registered with
    grids.drop_cubes (volume3d at import), which a job start and a memory check call alone; while the job runs the
    3D view keeps none."""
    import numpy as np
    import grids as G
    import volume3d as VOL
    d = Path(tempfile.mkdtemp(prefix="gui_release_"))
    try:
        n = 16
        np.random.default_rng(3).random((n, n, n)).astype(np.float32).tofile(d / "r.a_den")
        o = G.OutputSet(d / "r", box=((0.0,) * 3, (float(n),) * 3), redshift=0.0)
        G.drop_cubes()
        o.slice("density", 2, 3)                                    # the map's cube, cached
        avg = VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 8)
        prep = VOL.prepare(avg, True)                               # the 3D caches filled
        held = n ** 3 * 4 + avg["values"].nbytes + VOL.prepared_bytes(prep)
        freed = G.drop_cubes()                                      # the job-start path: drop_cubes() alone ...
        G.cap_cubes(True)                                           # ... then the cap (app_run._begin)
        left = VOL.drop_volumes()
        VOL.prepare(VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 8), True)
        kept = VOL.drop_volumes()
        check("a job start's drop_cubes() also frees the 3D view's averaged and prepared cubes (registered at import): "
              "the bytes of all three counted, drop_volumes() then 0, and while the job runs the 3D view keeps none",
              VOL.drop_volumes in G._DROP_CALLBACKS and freed == held and left == 0 and kept == 0,
              f"{freed} {held} {left} {kept}")
        src = {f: (ROOT / "python" / "gui" / f).read_text() for f in ("app_run.py", "app_check.py")}
        check("... the job start (app_run) and the memory check (app_check) call grids.drop_cubes() alone",
              all(t.count("G.drop_cubes()") == 1 and "drop_volumes" not in t for t in src.values()))
    finally:
        G.cap_cubes(False)
        G.drop_cubes()
        shutil.rmtree(d, ignore_errors=True)


def audit_claims():
    """An output name with a dot is listed; time estimates compare runs of one precision."""
    import numpy as np
    import results as R
    import grids as G
    d = Path(tempfile.mkdtemp(prefix="gui_claims_"))
    try:
        for name in ("snap_z0.5", "plain", "run.v1"):
            np.zeros(8 ** 3, dtype=np.float32).tofile(d / f"{name}.a_den")
        roots = sorted(o.prefix for _, o in G.find_outputs(d / "no-root", [str(d)]))
        check("an output name with a dot ('snap_z0.5', 'run.v1') is an output of its own, not cut at the first dot",
              roots == ["plain", "run.v1", "snap_z0.5"], str(roots))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    def rec(sec, binary):
        r = R.RunRecord(path=Path(f"/x/{binary}.runlog"), sim="S", prefix="ps_output")
        r.ok, r.wall_seconds, r.binary = True, sec, binary
        r.command = f"/x/{binary} in.hdf5 out --grid 64"
        return r
    recs = [rec(100, "PS-DTFE"), rec(1000, "PS-DTFE-double"), rec(1, "PS-DTFE-2d")]
    kw = dict(sim="S", grid=64, estimator="ps", gpu=False, sliced=False)
    got = (R.time_estimate(recs, **kw), R.time_estimate(recs, precision="double", **kw))
    check("time estimates: a double-precision run (2x slower) never predicts a single one, nor a 2D run a 3D one",
          got == ((100, 1), (1000, 1)), str(got))
    # a run under a second ('total wall time : 0s') still has a time (not one dropped as if it had none)
    fast = R.time_estimate([rec(0.0, "PS-DTFE"), rec(0.0, "PS-DTFE")], **kw)
    check("... and runs under a second ('0s') still give an estimate", fast == (0.0, 2), str(fast))


def run_options(tmp: Path):
    """The tessellation cache and the standard-DTFE field ticks (the memory budget is the binary's own)."""
    base = dict(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99])
    env0 = rs.RunSpec(estimator="dtfe", **base).command()[1]
    web = rs.RunSpec(estimator="dtfe", fields=["density_a", "tweb_a", "dispersion_a"],
                     tess_cache="/private/tmp/dtfe-tess", **base)
    env1 = web.command()[1]
    check("standard DTFE: the script's own field list by default; the ticks (the T-web) when they differ, without the "
          "phase-space dispersion", "FIELDS" not in env0 and env1.get("FIELDS") == "density_a tweb_a", str(env1))
    check("... the tessellation cache reaches the run scripts (TESS_CACHE) only when set; no memory budget (the "
          "binary's own: DTFE_MEM_BUDGET_GB never set)", env1.get("TESS_CACHE") == "/private/tmp/dtfe-tess"
          and "DTFE_MEM_BUDGET_GB" not in env1 and "DTFE_MEM_BUDGET_GB" not in env0 and "TESS_CACHE" not in env0,
          str(env1))
    pe = rs.PipelineSpec(data_root=str(tmp), sims=["TNG100-3-Dark"], tess_cache="/private/tmp/t").command()[1]
    check("... the pipeline too", "DTFE_MEM_BUDGET_GB" not in pe and pe.get("TESS_CACHE") == "/private/tmp/t", str(pe))
    icloud = Path.home() / "Library" / "Mobile Documents" / "dtfe-tess"
    probs = rs._run_option_problems(rs.RunSpec(tess_cache=str(icloud), **base))
    check("an iCloud tessellation cache is an error (the binary refuses it)",
          any(lv == "error" and "iCloud" in m for lv, m in probs), str(probs))
    nodtfe = rs.RunSpec(estimator="dtfe", fields=["dispersion_a"], **base).problems()
    check("... standard DTFE with only the dispersion ticked has nothing to compute: an error",
          any(lv == "error" and "no field standard DTFE computes" in m for lv, m in nodtfe), str(nodtfe))
    a = rs.RunSpec(estimator="ps", **base)
    check("the memory check is kept per quality, not per cache folder",
          a.memory_key() != rs.RunSpec(estimator="ps", quality="fast", **base).memory_key()
          and a.memory_key() == rs.RunSpec(estimator="ps", tess_cache="/x", **base).memory_key())


def pipeline_options(tmp: Path, sim: Path):
    """The Pipeline tab's run options (the quality and what it implies: deposit, sub-samples, GPU, vertex mass, volume
    weighting, sequential insertion; caustics, cusps), the plane's geometry (planes, thickness, sub-samples, window)
    and the render step's options reach the scripts, and the memory check follows every one of them."""
    import json
    import subprocess
    print("pipeline options:")
    base = dict(data_root=str(tmp), sims=["TNG100-3-Dark"], nu=64, grid=128, scratch_dir=str(tmp.parent))
    run_keys = ("PS_EXACT", "AVG_SUBSAMPLES", "PS_GPU", "PS_VERTEX_MASS", "PS_VOLUME_WEIGHTED", "PS_CAUSTICS",
                "PS_CAUSTIC_CUSPS", "PS_PARALLEL_TRI", "PLANES", "SUPERSAMPLE")
    env = rs.PipelineSpec(**base).command()[1]
    # the caustic classes are on by default (run_ps_dtfe.sh's default and the production runs'); the exact deposit
    # too (the default since 2026-11-09: no sub-sample count, pinned empty); either deposit's GPU when the PS pair
    # was built with one (rs.gpu_build_default, scripts/config.sh's rule; the exact deposit's in every machine use)
    gpu = "1" if rs.gpu_build_default("ps") else "0"
    env_s = rs.PipelineSpec(**base, quality="fast").command()[1]
    check("defaults: every run option is spelled out for run_ps_dtfe.sh (a stray PS_* in the login shell cannot leak "
          "in); Exact with the count pinned empty, Fast with the quality's nSub",
          [env[k] for k in run_keys] == ["1", "", gpu, "1", "1", "1", "0", "0", "1", "1"]
          and env["SUBSAMPLES"] == "" and [env_s[k] for k in run_keys[:3]] == ["0", "1", gpu]
          and env_s["SUBSAMPLES"] == "1" and "THICKNESS" not in env and "WINDOW" not in env, f"{env} {env_s}")
    q = rs.PipelineSpec(**base, quality="fast", caustics=True, caustic_cusps=True, planes=16, thickness=3.5,
                        supersample=2, window="10 60 0 25")
    with fake_build("cpu"):
        env = q.command()[1]
        st = q.check_step("TNG100-3-Dark", [99], 32)
    with fake_build("gpu"):
        st_gpu = q.check_step("TNG100-3-Dark", [99], 32)
    check("every option switched: the environment follows (Fast on a CPU build: the quality's count, no GPU)",
          [env[k] for k in run_keys + ("THICKNESS", "WINDOW")]
          == ["0", "1", "0", "1", "1", "1", "1", "0", "16", "2", "3.5", "10 60 0 25"]
          and "DTFE_PRECISION" not in env, str(env))
    # P6.7: each run-script knob under its new name AND its older one (the same value) until P6.6, so a stray older
    # export in the login shell is overwritten; the new names are the ones the drivers read first
    pairs = (("DTFE_EXACT_AVERAGE", "PS_EXACT"), ("SUBSAMPLES", "AVG_SUBSAMPLES"), ("DTFE_GPU", "PS_GPU"),
             ("DTFE_GPU", "PS_METAL"), ("PARALLEL_TRIANGULATION", "PS_PARALLEL_TRI"), ("DTFE_RESUME", "PS_RESUME"),
             ("PS_POINT_CONVENTIONS", "PTS_CONVENTIONS"), ("PLANE_WINDOW", "WINDOW"), ("PYTHON", "PY"),
             ("TESSELLATION_CACHE", "TESS_CACHE"))
    std_pairs = (("DTFE_EXACT_AVERAGE", "DTFE_EXACT"), ("DTFE_GPU", "DTFE_METAL"), ("TESSELLATION_CACHE", "TESS_CACHE"))
    both = lambda e, pp=pairs: [(n, o) for n, o in pp if (n in e or o in e) and e.get(n) != e.get(o)]   # noqa: E731
    tq = rs.PipelineSpec(**base, tess_cache="/private/tmp/t", window="10 60 0 25").command()[1]
    with fake_build("gpu"):         # (the GPU in both: Exact for PS-DTFE, Fast for standard DTFE in balanced)
        g_ps = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99],
                          slice_plane="/x.bin", tess_cache="/private/tmp/t", start_over=True).command()[1]
        g_dt = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], estimator="dtfe", quality="fast",
                          tess_cache="/private/tmp/t").command()[1]
    check("the knobs under their new names and their older ones alike (Pipeline, PS and standard Grids jobs; P6.7)",
          not both(env) and not both(tq) and not both(g_ps) and not both(g_dt, std_pairs)
          and env["SUBSAMPLES"] == "1" and env["DTFE_EXACT_AVERAGE"] == "0" and env["PLANE_WINDOW"] == "10 60 0 25"
          and env["PYTHON"] == rs.PYTHON and tq["TESSELLATION_CACHE"] == "/private/tmp/t"
          and g_ps["DTFE_EXACT_AVERAGE"] == "1" and g_ps["DTFE_GPU"] == "1" and g_ps["DTFE_RESUME"] == "0"
          and g_ps["PS_POINT_CONVENTIONS"] == "grid" and g_ps["PS_MERGE"] == "index-order"
          and (g_dt["DTFE_EXACT_AVERAGE"], g_dt["DTFE_EXACT"], g_dt["DTFE_GPU"], g_dt["DTFE_METAL"]) == ("0", "0", "1", "1")
          and "PS_MERGE" not in g_dt and "PS_EXACT" not in g_dt,
          f"{both(env)} {both(tq)} {both(g_ps)} {both(g_dt, std_pairs)}")
    lines = [rs.shell(*job.command()) for job in (rs.PipelineSpec(**base, window="10 60 0 25", tess_cache="/t"),
                                                    rs.PipelineSpec(**base, quality="fast"))] \
        + [rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], estimator=e, slice_plane="/x.bin",
                      tess_cache="/t").shell_line() for e in ("ps", "dtfe")]
    older = [k for line in lines for k in list(rs.OLDER_KNOB_NAMES) + ["ORDERED_MERGE"] if f" {k}=" in f" {line}"]
    check("... the command shown (and exported) names each knob once, by its new name",
          not older and all(" PS_MERGE=" in ln for ln in lines[:3]) and " PLANE_WINDOW=" in lines[0]
          and " PYTHON=" in lines[0], f"{older} {lines}")
    # run_dtfe.sh's Monte-Carlo count and thread cap: the Grids tab holds neither for a standard run, so the job pins
    # both EMPTY (the script's own defaults). A login export is overwritten only where the key is spelled out: in the
    # step's env (the job runner inserts each of its keys over the system environment) and in its line (the command
    # shown and the exported script put the assignments in front of run_dtfe.sh)
    pinned = ("SUBSAMPLES", "THREADS")
    st_dt = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], estimator="dtfe").steps()
    exported = [ln for ln in rs.export_script(st_dt, "t").splitlines() if "run_dtfe.sh" in ln and not ln.startswith("#")]
    ps_argv = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], quality="fast").command()[0]
    check("a standard Grids job pins SUBSAMPLES and THREADS empty (run_dtfe.sh: the binary's count, all cores): both "
          "keys in the env its step runs with and in the line shown and exported; a PS one leaves the thread cap to "
          "the shell and its count to -n (Fast: the quality's 1), or pins the count empty (Exact has none)",
          len(st_dt) == 1 and len(exported) == 1
          and all(k in e and e[k] == "" for e in (st_dt[0].env, g_dt) for k in pinned) and "AVG_SUBSAMPLES" not in g_dt
          and all(f" {k}='' " in f" {ln} " for ln in (lines[3], st_dt[0].line(), exported[0]) for k in pinned)
          and g_ps.get("SUBSAMPLES") == "" and "THREADS" not in g_ps and "-n" in ps_argv
          and ps_argv[ps_argv.index("-n") + 1] == "1",
          f"{st_dt[0].env if st_dt else None} | {g_dt} | {lines[3]} | {exported} | {ps_argv}")
    check("... the merge is the explicit index order in both qualities (PS_MERGE=index-order: the memory gate stays "
          "on under the speed split); ORDERED_MERGE the same until P6.6",
          (env["PS_MERGE"], env["ORDERED_MERGE"]) == ("index-order", "1")
          and [rs.PipelineSpec(**base).command()[1][k] for k in ("PS_MERGE", "ORDERED_MERGE")]
          == ["index-order", "1"], str(env))
    check("the memory check carries the run options, no -m without the GPU (a CPU build) and -m with it",
          st.argv[1:] == ["-s", "TNG100-3-Dark", "-g", "128", "99"] and "DTFE_PRECISION" not in st.env
          and st.env["PS_EXACT"] == "0" and st.env["AVG_SUBSAMPLES"] == "1" and st.env["PS_CAUSTICS"] == "1"
          and st.env["AUTO_TUNE_REPORT"] == "1" and st.env["PS_GPU"] == "0"
          and st_gpu.argv[1:] == ["-s", "TNG100-3-Dark", "-g", "128", "-m", "99"] and st_gpu.env["PS_GPU"] == "1"
          and st.env["PTS_VEL_GRAD"] == "1" and not any(k in st.env for k in rs.PIPELINE_ONLY_KEYS),
          f"{st.argv} {st.env} {st_gpu.argv}")
    # 16 planes x 32 x nv x 2^2, nv = round(32 * 25 / 50) = 16 (square pixels over the window's aspect)
    check("a plane not made yet is sized by planes x nu x nv x K^2, nv following the window's aspect",
          st.env.get("DTFE_AUTO_PTS_N") == str(16 * 32 * 16 * 4) and q.plane_points(32) == 16 * 32 * 16 * 4
          and rs.PipelineSpec(**base).plane_points() == 64 * 64, str(st.env.get("DTFE_AUTO_PTS_N")))
    keys = {name: rs.PipelineSpec(**base, **kw).memory_key() for name, kw in
            (("plain", {}), ("caustics", dict(caustics=False)), ("super", dict(supersample=2)),
             ("render", dict(render_smooth=3.0, render_fields="density", render_fixed_range=True, thickness=9.0)))}
    check("the memory key follows the compute options, not the render ones (nor the thickness)",
          len({keys["plain"], keys["caustics"], keys["super"]}) == 3 and keys["render"] == keys["plain"])

    side = sim / "hires_plane_z64.json"         # an old plane's sidecar, as make_image_plane.py wrote them
    side.write_text(json.dumps({"box": 110.7, "center": 55.35, "planes": 1, "thickness": 2.0, "nu": 64, "nv": 64,
                                "u0": 0.0, "u1": 110.7, "v0": 0.0, "v1": 110.7}))
    p = rs.PipelineSpec(**base)
    check("plane_matches: the default geometry matches an old single-plane sidecar (no 'supersample' key = 1)",
          p.plane_matches("TNG100-3-Dark") is True)
    differ = {k: rs.PipelineSpec(**base, **kw).plane_matches("TNG100-3-Dark")
              for k, kw in (("planes", dict(planes=4)), ("supersample", dict(supersample=2)),
                            ("window", dict(window="0 50 0 50")), ("center", dict(center="30")))}
    check("plane_matches: another plane count, sub-sampling, window or centre differs",
          all(v is False for v in differ.values()), str(differ))
    check("plane_matches: the sidecar's own centre and a window spelling the whole box both match",
          rs.PipelineSpec(**base, center="55.35", window="0 110.7 0 110.7").plane_matches("TNG100-3-Dark") is True)
    check("stale(): a plane of another geometry makes every snapshot stale, a matching one keeps the fresh ones",
          p.stale("TNG100-3-Dark") == ([99], 2)
          and rs.PipelineSpec(**base, planes=4).stale("TNG100-3-Dark") == ([4, 99], 2),
          f"{p.stale('TNG100-3-Dark')} {rs.PipelineSpec(**base, planes=4).stale('TNG100-3-Dark')}")
    check("check_step: a plane the script will regenerate is sized by its point count, not read from disk",
          "SAMPLE_POINTS" in p.check_step("TNG100-3-Dark", [99], 64).env
          and rs.PipelineSpec(**base, planes=4).check_step("TNG100-3-Dark", [99], 64).env.get("DTFE_AUTO_PTS_N")
          == str(4 * 64 * 64))
    tool = str(rs.REPO_ROOT / "python" / "tools" / "make_image_plane.py")
    before = {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in sim.iterdir()}
    same = subprocess.run([sys.executable, tool, "--same-as", str(side), "--nu", "64"], capture_output=True, text=True)
    diff = subprocess.run([sys.executable, tool, "--same-as", str(side), "--nu", "64", "--planes", "4"],
                          capture_output=True, text=True)
    bad = subprocess.run([sys.executable, tool, "--same-as", str(side), "--nu", "64", "--plane-window", "50", "10", "0",
                          "10"],
                         capture_output=True, text=True)
    gone = subprocess.run([sys.executable, tool, "--same-as", str(sim / "no_such_plane.json"), "--nu", "64"],
                          capture_output=True, text=True)
    check("make_image_plane.py --same-as (the script's check): exit 0 and 'same' for this geometry, 3 and 'differs' "
          "for another or an unreadable sidecar, 1 for invalid input (not read as 'differs'); nothing written",
          same.returncode == 0 and same.stdout.strip() == "same" and diff.returncode == 3
          and diff.stdout.strip() == "differs" and gone.returncode == 3 and bad.returncode == 1
          and {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in sim.iterdir()} == before,
          f"{same.returncode} {same.stdout!r} {same.stderr[-200:]} | {diff.returncode} {diff.stdout!r} | "
          f"{bad.returncode} {bad.stderr[-120:]!r} | {gone.returncode}")
    # the simulation it reads (require_sim's rule): --sim or --combined, else DTFE_SIM; an explicit --combined wins
    # over an exported DTFE_SIM (the pipeline exports it); with none of the three a usage error naming both ways
    bare = {k: v for k, v in os.environ.items() if k != "DTFE_SIM"}
    bare["DTFE_DATA_ROOT"] = str(tmp)

    def plane(*args, **env):
        return subprocess.run([sys.executable, tool, *args, "-o", str(tmp / "plane_none")], capture_output=True,
                              text=True, env=dict(bare, **env))
    none_, by_env = plane(), plane("--snap", "99", DTFE_SIM="NoSuchSim")
    explicit = plane("--combined", str(tmp / "no_such.hdf5"), DTFE_SIM="NoSuchSim")
    usage = [ln for ln in none_.stderr.splitlines() if ": error: " in ln]
    check("make_image_plane.py: neither --sim nor --combined nor DTFE_SIM is exit 2 naming both ways (no nested "
          "parentheses); DTFE_SIM alone names the simulation; an explicit --combined is read, not DTFE_SIM's snapshot",
          none_.returncode == 2 and len(usage) == 1 and "one of --sim / --combined is required; no simulation given: "
          "pass --sim NAME or set DTFE_SIM=NAME" in usage[0] and "required (" not in usage[0] and "))" not in usage[0]
          and by_env.returncode == 1 and str(Path("NoSuchSim") / "snapdir_099" / "combined_099.hdf5") in by_env.stderr
          and explicit.returncode == 1 and f"snapshot not found: {tmp / 'no_such.hdf5'}" in explicit.stderr
          and "NoSuchSim" not in explicit.stderr and not (tmp / "plane_none.bin").exists(),
          f"{none_.returncode} {none_.stderr[-200:]!r} | {by_env.returncode} {by_env.stderr[-200:]!r} | "
          f"{explicit.returncode} {explicit.stderr[-200:]!r}")
    # the launcher's reading of a broken or missing sidecar agrees with the script's (regenerate = all stale)
    good = side.read_text()
    side.write_text("{ not json")
    broken = (rs.PipelineSpec(**base).plane_matches("TNG100-3-Dark"), rs.PipelineSpec(**base).stale("TNG100-3-Dark"))
    side.unlink()
    missing = (rs.PipelineSpec(**base).plane_matches("TNG100-3-Dark"),
               "DTFE_AUTO_PTS_N" in rs.PipelineSpec(**base).check_step("TNG100-3-Dark", [99], 64).env)
    side.write_text(good)
    check("an unreadable or missing sidecar reads as 'differs': every snapshot stale, the check sized by point count",
          broken == (False, ([4, 99], 2)) and missing == (False, True), f"{broken} {missing}")
    check("a malformed window of the job itself is not the plane's fault (None; problems() reports it)",
          rs.PipelineSpec(**base, window="1 2").plane_matches("TNG100-3-Dark") is None)
    # the numbers the script receives read back exactly (':g' kept six digits)
    fine = rs.PipelineSpec(**base, window="12.3456789 50 0 50", planes=2, thickness=0.123456789)
    env = fine.command()[1]
    u0, u1, v0, v1 = (float(v) for v in env["WINDOW"].split())      # as make_image_plane.py stores what it got
    side2 = dict(json.loads(good), u0=u0, u1=u1, v0=v0, v1=v1, planes=2, thickness=float(env["THICKNESS"]))
    side.write_text(json.dumps(side2))
    check("WINDOW/THICKNESS reach the script at full precision, so the sidecar they produce matches the job",
          env["WINDOW"] == "12.3456789 50 0 50" and env["THICKNESS"] == "0.123456789"
          and fine.plane_matches("TNG100-3-Dark") is True, f"{env['WINDOW']} {env['THICKNESS']}")
    side.write_text(good)
    by_build = {}                       # the quality's device: the GPU on a GPU build, the CPU on a CPU-only one
    for kind in ("gpu", "cpu"):
        with fake_build(kind):
            by_build[kind] = (rs.PipelineSpec(**base).command()[1],
                              rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99]).command()[1],
                              rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99],
                                         estimator="dtfe").command()[1])
    envs = [by_build["gpu"][0], by_build["cpu"][0], by_build["gpu"][1], by_build["cpu"][1], by_build["cpu"][2]]
    check("the GPU knob is pinned in the environment too (PS_GPU / DTFE_GPU, and their older names PS_METAL / "
          "DTFE_METAL for one release), not only by -m",
          [(e.get("PS_GPU"), e.get("PS_METAL")) for e in envs[:4]] == [("1", "1"), ("0", "0"), ("1", "1"), ("0", "0")]
          and (envs[4].get("DTFE_GPU"), envs[4].get("DTFE_METAL")) == ("0", "0"), str(envs))

    r = rs.PipelineSpec(**base, render_fields="density, velDiv", render_smooth=2.0, render_smooth_derivatives=0.0,
                        render_fixed_range=True, render_force=True, planes=16, render_project="slab")
    a = r.steps()[1].argv
    check("the render step passes the figure options to plot_pointeval.py (the maps space-separated)",
          a[a.index("--maps") + 1:a.index("--maps") + 3] == ["density", "velDiv"]
          and a[a.index("--project") + 1] == "slab"
          and a[a.index("--smooth-px") + 1] == "2" and a[a.index("--smooth-derivatives-px") + 1] == "0"
          and "--fixed-range" in a and "--force" in a, str(a))
    a0 = rs.PipelineSpec(**base, render_project="slab").steps()[1].argv
    check("... the defaults add nothing, and 'slab' of a single plane is the plane itself",
          not any(f in a0 for f in ("--maps", "--project", "--smooth-px", "--smooth-derivatives-px", "--fixed-range",
                                    "--force")), str(a0))
    lv, msg = levels(rs.PipelineSpec(**base, caustic_cusps=True, caustics=False))
    check("cusps without caustics is an error (the Grids tab's rule, shared)", "error" in lv and "cusps" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(**base, planes=4))
    check("a handful of planes warns about ghosting", "warning" in lv and "ghosts" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(**base, window="0 50 0"))
    check("a malformed window is an error", "error" in lv and "window" in msg, msg)
    lv, msg = levels(rs.PipelineSpec(**base, render_fields="density,bogus"))
    check("an unknown point-evaluated field is an error", "error" in lv and "bogus" in msg, msg)
    check("the queue summary names the quality in the launcher's words: Exact (the default) or Fast; an unknown "
          "value reads as Exact (as quality_mode reads it), never verbatim",
          ", Exact" in rs.job_summary("pipeline", rs.PipelineSpec(**base))
          and ", Exact" not in rs.job_summary("pipeline", rs.PipelineSpec(**base, quality="fast"))
          and ", Fast" in rs.job_summary("pipeline", rs.PipelineSpec(**base, quality="fast"))
          and ", Exact" in rs.job_summary("pipeline", rs.PipelineSpec(**base, quality="bogus"))
          and "bogus" not in rs.job_summary("pipeline", rs.PipelineSpec(**base, quality="bogus"))
          and not any(w in rs.job_summary("pipeline", rs.PipelineSpec(**base, quality=q))
                      for w in (", exact", ", fast") for q in ("exact", "fast")),
          rs.job_summary("pipeline", rs.PipelineSpec(**base, quality="fast")))
    check("the pipeline's plan line is parsed with and without -m (PS_GPU)",
          rs.parse_progress("+ env SAMPLE_POINTS=/p.bin PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD=1 /r/scripts/run_ps_dtfe.sh "
                            "-s S -g 512 -m 50 99\n").get("planned") == 2
          and rs.parse_progress("+ env SAMPLE_POINTS=/p.bin PS_VOLUME_WEIGHTED=0 PTS_VEL_GRAD=1 /r/scripts/run_ps_dtfe.sh "
                                "-s S -g 512 50 99\n").get("planned") == 2)
    from dtfelib import pointeval
    check("runspec's point-eval constants match dtfelib.pointeval (not imported there: matplotlib)",
          rs.POINTEVAL_FIELDS == tuple(pointeval.STYLES) and rs.POINTEVAL_DERIVATIVE_SMOOTH == pointeval.DERIVATIVE_SMOOTH)
    old = rs.PipelineSpec.from_dict({"sims": ["TNG50-4"], "nu": 4096, "render": False})
    check("settings saved before these options load with the scripts' defaults (Exact, on the GPU when the PS pair "
          "has one: the quality's device)",
          old.quality == "exact" and old.mode().gpu is rs.gpu_build_default("ps") and old.planes == 1
          and old.render_project == "plane" and not old.render)


def cube_cache_and_paths(tmp: Path, sim: Path):
    """The Explore cube cache follows the file, a standard-DTFE run gets its
    scratch folder, the demo is generated only at its own path, one lambda_th default, every products() caller
    passes --data-root."""
    import tempfile as tf
    import numpy as np
    import grids as G
    print("cube cache per file, scratch folder, demo path, lambda_th, --data-root:")
    # a grid written again is read again (a cache keyed on the path alone would not)
    od = Path(tf.mkdtemp(prefix="gui_cube_"))
    try:
        n = 8
        np.arange(n ** 3, dtype=np.float32).tofile(od / "run.a_den")
        o1 = G.OutputSet(od / "run")
        before = float(o1.slice("density", 2, 3)[1, 2])
        np.full(n ** 3, 7.0, dtype=np.float32).tofile(od / "run.a_den")       # the re-run: same size
        st = (od / "run.a_den").stat()
        os.utime(od / "run.a_den", ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))   # a coarse clock cannot hide it
        after = float(G.OutputSet(od / "run").slice("density", 2, 3)[1, 2])
        keys = [k for k in G._CUBES if k[0] == (od / "run.a_den").resolve()]
        check("Explore's cube cache follows the file: a re-run's values replace the old ones (one entry per file)",
              before != 7.0 and after == 7.0 and len(keys) == 1, f"{before} {after} {len(keys)}")
    finally:
        shutil.rmtree(od)
    # the standard-DTFE Grids run carries its scratch folder
    sd = tmp.parent
    env = rs.RunSpec(data_root=str(tmp), sim="TNG100-3-Dark", snapshots=[99], estimator="dtfe", scratch_dir=str(sd)).command()[1]
    check("a standard-DTFE Grids run passes SCRATCH_DIR (run_dtfe.sh's SCRATCH_DIR hook)", env.get("SCRATCH_DIR") == str(sd), str(env))
    # the demo is generated only at the demo's own path
    demo = rs.demo_spec(output_dir=str(tmp / "demo"))
    other = rs.CustomSpec.from_dict(dict(demo.to_dict(), input_file=str(tmp / "elsewhere" / "missing.hdf5")))
    check("the demo generates its snapshot at its own path only; another missing path is 'file not found'",
          demo.demo_pending() and any("generate the demo" in st.label for st in demo.steps())
          and not other.demo_pending() and not any("generate" in st.label for st in other.steps())
          and any("not found" in m for lvl, m in other.problems() if lvl == "error"),
          str([st.label for st in other.steps()]))
    legacy = rs.CustomSpec.from_dict({"input_file": str(tmp / "gone.hdf5"), "demo_n": 32})   # saved before demo_file
    check("a saved demo_n without demo_file (older settings) generates nothing", not legacy.demo_pending())
    # one lambda_th default across the tabs
    cs = rs.CustomSpec(input_file=str(sim / "snapdir_099" / "combined_099.hdf5"), fields=["density_a", "tweb_a"])
    cs0 = rs.CustomSpec(input_file=str(sim / "snapdir_099" / "combined_099.hdf5"), fields=["density_a"])
    a, a0 = cs.command(), cs0.command()
    check("a Custom run with a web field passes the scripts' lambda_th (0.3), one without passes none",
          a[a.index("--lambda-th") + 1] == "0.3" and "--lambda-th" not in a0, str(a))
    check("the Grids/Pipeline specs pass LAMBDA_TH only when it differs from the scripts' default",
          "LAMBDA_TH" not in rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp)).command()[1]
          and rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp), lambda_th=0.2).command()[1]["LAMBDA_TH"] == "0.2"
          and rs.PipelineSpec(sims=["TNG100-3-Dark"], data_root=str(tmp), lambda_th=0.4).command()[1]["LAMBDA_TH"] == "0.4"
          and rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp), lambda_th=0.2).memory_key()
          == rs.RunSpec(sim="TNG100-3-Dark", snapshots=[99], data_root=str(tmp)).memory_key()
          and rs.CustomSpec(input_file="x.hdf5", lambda_th=0.2).memory_key() == rs.CustomSpec(input_file="x.hdf5").memory_key())
    # no plot script asks pipeline.products() without the root it was given
    import re
    bad = sorted(f.name for f in (ROOT / "python" / "plot").glob("*.py")
                 if re.search(r"pipeline\.products\((?![^)]*data_root)", f.read_text().replace("\n", " ")))
    check("every pipeline.products() call in python/plot passes data_root", bad == [], ", ".join(bad))
    from dtfelib import cli as dcli
    keep = dcli._data_root_override
    try:
        dcli.use_data_root(tmp)
        check("use_data_root(): the helpers without a root argument (trees, group catalogues, the ladder) follow --data-root",
              dcli.sim_dir("TNG100-3-Dark") == tmp / "TNG100" / "TNG100-3-Dark" and dcli.require_data_root() == tmp)
    finally:
        dcli.use_data_root(keep)


def cube_cache_threads():
    """The cube cache is thread-safe and droppable, a big cube is read
    whole only along its strided axis, a short file does not leave the lock held."""
    import tempfile as tf
    import threading
    import numpy as np
    import grids as G
    print("cube cache:")
    od = Path(tf.mkdtemp(prefix="gui_cubes_"))
    eager = G.CUBE_EAGER_BYTES
    try:
        n = 8
        ref = np.arange(n ** 3, dtype=np.float32).reshape(n, n, n)
        ref.tofile(od / "big.a_den")
        key = (od / "big.a_den").resolve()
        G.drop_cubes()
        G.CUBE_EAGER_BYTES = 1000                                   # the 2 KB cube counts as 'big'
        o = G.OutputSet(od / "big")
        p0 = o.slice("density", 0, 3)
        lazy = not any(k[0] == key for k in G._CUBES)
        p2 = o.slice("density", 2, 5)
        cached = any(k[0] == key for k in G._CUBES)
        p0b = o.slice("density", 0, 3)
        still = any(k[0] == key for k in G._CUBES)
        check("a cube above CUBE_EAGER_BYTES: an x plane comes from the memmap (not read whole), the strided z plane "
              "caches it, and the x plane then comes from the cache", lazy and cached and still
              and np.array_equal(p0, ref[3]) and np.array_equal(p2, ref[:, :, 5]) and np.array_equal(p0b, ref[3]),
              f"{lazy} {cached} {still}")
        np.arange(32 * 32, dtype=np.float32).tofile(od / "flat.a_den")
        G.OutputSet(od / "flat", dim=2).slice("density")
        check("a 2D grid is always read whole and cached, whatever its size against the threshold",
              any(k[0] == (od / "flat.a_den").resolve() for k in G._CUBES))
        G.CUBE_EAGER_BYTES = eager
        dropped = G.drop_cubes()
        check("drop_cubes() returns the bytes held and empties the cache", dropped == 32 * 32 * 4 + n ** 3 * 4
              and not G._CUBES, f"{dropped} {len(G._CUBES)}")
        # two threads reading the same cold cube: one entry, both the right array, no exception
        got, errs = [], []
        go = threading.Barrier(2)

        def reader():
            try:
                go.wait(5)
                got.append(G.OutputSet(od / "big").slice("density", 2, 1))
            except Exception as e:            # noqa: BLE001
                errs.append(repr(e))
        ts = [threading.Thread(target=reader) for _ in range(2)]
        [t.start() for t in ts]
        [t.join(10) for t in ts]
        entries = [k for k in G._CUBES if k[0] == key]
        check("two threads slicing the same cold cube at once: one cache entry, both get the plane",
              not errs and len(got) == 2 and len(entries) == 1 and all(np.array_equal(g, ref[:, :, 1]) for g in got),
              f"{errs} {len(got)} {len(entries)}")
        # a file shorter than its grid (a run rewriting it): the error comes out, the lock does not stay held
        o = G.OutputSet(od / "big")
        G.drop_cubes()
        with open(od / "big.a_den", "r+b") as fh:
            fh.truncate(n ** 3 * 4 - 16)
        raised = False
        try:
            o.slice("density", 2, 1)
        except ValueError:
            raised = True
        ok = G._CUBES_LOCK.acquire(timeout=2)
        if ok:
            G._CUBES_LOCK.release()
        again = G.OutputSet(od / "flat", dim=2).slice("density")
        check("a short file raises ValueError from slice() and leaves the lock free (the next slice works)",
              raised and ok and again.shape == (32, 32), f"{raised} {ok}")
    finally:
        G.CUBE_EAGER_BYTES = eager
        G.drop_cubes()
        shutil.rmtree(od)


def type_labels():
    """The figure grid's Type column: which estimator made an output -- the sidecar, else a streams grid
    (only the phase-space binary writes one), else the prefix; the prefix shown only when not the default."""
    import json
    import tempfile as tf
    import numpy as np
    import grids as G
    print("type labels:")
    od = Path(tf.mkdtemp(prefix="gui_types_"))
    try:
        n = 4
        def out(prefix, streams=False, estimator=None):
            np.ones(n ** 3, dtype=np.float32).tofile(od / f"{prefix}.a_den")
            if streams:
                np.ones(n ** 3, dtype=np.float32).tofile(od / f"{prefix}.a_streams")
            if estimator:
                (od / f"{prefix}.gui.json").write_text(json.dumps({"estimator": estimator}))
            return G.OutputSet(od / prefix)
        got = {
            "ps_output": G.type_label(out("ps_output")),
            "output": G.type_label(out("output")),
            "ps_x density only": G.type_label(out("ps_x")),
            "foo + streams": G.type_label(out("foo", streams=True)),
            "sidecar dtfe wins": G.type_label(out("ps_y", streams=True, estimator="dtfe")),
            "sidecar ps wins": G.type_label(out("bar", estimator="ps")),
        }
        want = {"ps_output": "PS-DTFE", "output": "DTFE", "ps_x density only": "PS-DTFE · ps_x",
                "foo + streams": "PS-DTFE · foo", "sidecar dtfe wins": "DTFE · ps_y", "sidecar ps wins": "PS-DTFE · bar"}
        check("type labels: the default prefixes bare, others with the prefix; a ps prefix without streams is still "
              "PS-DTFE; a streams grid makes any prefix PS-DTFE; the sidecar beats both", got == want, str(got))
    finally:
        shutil.rmtree(od)


def launcher_fixes():
    """running_jobs() classifies by command line and parent, and no plot script builds
    the flat <root>/<sim> path by hand (a data root may keep simulations per family)."""
    import subprocess
    print("running jobs:")
    me = os.getpid()
    # pgrep names five DTFE binaries; ps shows what they are: the launcher's query server (--serve, a direct
    # child), its exact zoom (a direct child writing into a 'dtfe-zoom-' folder), a run from another terminal
    # (bash -> binary: a grandchild), a run whose OUTPUT PATH contains '--serve' (the token rule), a memory check
    # from a terminal (--auto-tune-report, never counted), someone else's query server (listed apart, with its
    # memory) and a Custom-tab run (a direct child too, with a --ps-window of its own, but a run)
    pgrep_out = "100 PS-DTFE\n200 PS-DTFE\n300 DTFE-double\n400 PS-DTFE\n500 PS-DTFE\n600 PS-DTFE\n700 PS-DTFE\n"
    ps_out = (f"  100 {me} 24000000 /Users/x/My Code/DTFE/PS-DTFE /data/sims/snap.hdf5 /tmp/s/serve --serve "
              f"--input 105 --verbose 0 --serve-resident 4 --serve-progress\n"
              f"  200 {me} 1500000 /Users/x/My Code/DTFE/PS-DTFE /data/sims/snap.hdf5 "
              f"/var/folders/ab/T/{rs.ZOOM_DIR_PREFIX}k3_x9q/zoom --ps-window "
              f"1 2 3 4 5 6 --field density_a\n"
              f"  300 4242 18200000 /Users/x/My Code/DTFE/DTFE-double /data/sims/snap.hdf5 out --grid 256 256 256\n"
              f"  400 4343 900000 ./PS-DTFE snap.hdf5 /tmp/--serve-test/out --periodic\n"
              f"  500 4444 300000 ./PS-DTFE snap.hdf5 /tmp/r/out --grid 512 --auto-tune-report\n"
              f"  600 4545 9300000 ./PS-DTFE other.hdf5 /tmp/s2/serve --serve --input 105\n"
              f"  700 {me} 5100000 /Users/x/My Code/DTFE/PS-DTFE /Users/x/cloud.hdf5 /Users/x/out/zoom "
              f"--grid 64 --ps-window 0 1 0 1 0 1\n")
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[0] == "pgrep":
            return subprocess.CompletedProcess(argv, 0, stdout=pgrep_out, stderr="")
        if argv[0] == "ps":
            assert argv[-1] == "100,200,300,400,500,600,700" and "-ww" in argv and "rss=" in argv[3], argv
            return subprocess.CompletedProcess(argv, 0, stdout=ps_out, stderr="")
        raise AssertionError(argv)

    orig_run, orig_cache = rs.subprocess.run, rs._JOBS_CACHE
    try:
        rs.subprocess.run = fake_run
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.dtfe_processes(max_age=0)             # one classification; the two views below read the cache
        jobs = rs.running_jobs()
        check("runs: a run from another terminal, one whose path merely contains '--serve' and the launcher's own "
              "Custom-tab run (a direct child, even with --ps-window and an output named 'zoom'); not the launcher's "
              "own server or exact zoom, not a memory check (--auto-tune-report), not another server",
              jobs == [(300, "DTFE-double"), (400, "PS-DTFE"), (700, "PS-DTFE")], str(jobs))
        servers = rs.running_servers()
        check("servers: every query server, with its resident memory in GB (ps rss)",
              [(p, n, round(g, 1)) for p, n, g in servers] == [(100, "PS-DTFE", 24.0), (600, "PS-DTFE", 9.3)], str(servers))
        check("one pgrep, then one ps over exactly the named pids", [c[0] for c in calls] == ["pgrep", "ps"], str(calls))
        msgs = rs._busy_problems()
        check("the busy warning names the runs with their memory; the servers are an info line with theirs",
              [lvl for lvl, _ in msgs] == ["warning", "info"] and "DTFE-double (pid 300, 18.2 GB)" in msgs[0][1]
              and "PS-DTFE (pid 400, 0.9 GB)" in msgs[0][1] and "PS-DTFE (pid 100, 24.0 GB)" in msgs[1][1]
              and "(pid 600, 9.3 GB)" in msgs[1][1], str(msgs))
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.subprocess.run = lambda argv, **kw: (fake_run(argv, **kw) if argv[0] == "pgrep"
                                               else (_ for _ in ()).throw(OSError("no ps")))
        check("without ps every named binary counts as a run (no memory known), as before",
              rs.running_jobs(max_age=0) == [(100, "PS-DTFE"), (200, "PS-DTFE"), (300, "DTFE-double"), (400, "PS-DTFE"),
                                             (500, "PS-DTFE"), (600, "PS-DTFE"), (700, "PS-DTFE")]
              and rs.running_servers(max_age=0) == [] and "(pid 100)" in rs._busy_problems()[0][1])
        rs._JOBS_CACHE = (-1e9, {"runs": [], "servers": []})
        rs.subprocess.run = lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        check("nothing running: no ps call, empty lists, no message",
              rs.running_jobs(max_age=0) == [] and rs.running_servers(max_age=0) == [] and not rs._busy_problems())
    finally:
        rs.subprocess.run, rs._JOBS_CACHE = orig_run, orig_cache
    zoom = f"x /tmp/{rs.ZOOM_DIR_PREFIX}ab12/zoom --grid 8"
    check("the kinds: server by the --serve token, report by --auto-tune-report, own = a child writing into an "
          "exact-zoom folder, else a run (a Custom-tab child included)",
          rs._kind(1, "x --serve y", 99) == "server" and rs._kind(1, "x --auto-tune-report", 99) == "report"
          and rs._kind(99, zoom, 99) == "own" and rs._kind(99, "x --grid 8", 99) == "run"
          and rs._kind(1, zoom, 99) == "run" and rs._kind(1, "x --grid 8", 99) == "run"
          and rs._kind(99, "x --serve", 99) == "server")
    check("the token rule: '--serve' alone, not --serve-resident/-progress or a path",
          bool(rs._SERVE.search("a --serve b")) and bool(rs._SERVE.search("x --serve"))
          and not rs._SERVE.search("a --serve-resident 4 b") and not rs._SERVE.search("/tmp/--serve/out"))

    print("plot scripts:")
    flat = sorted(f.name for f in (ROOT / "python" / "plot").glob("*.py")
                  if "args.data_root / args.sim" in f.read_text())
    check("no plot script builds <root>/<sim> by hand (sim_dir resolves the per-family layout)",
          flat == [], ", ".join(flat))
    from dtfelib.cli import sim_dir
    root = Path(tempfile.mkdtemp(prefix="gui_simdir_"))
    try:
        (root / "TNG100" / "TNG100-3-Dark").mkdir(parents=True)
        check("sim_dir: a per-family simulation resolves under its family folder",
              sim_dir("TNG100-3-Dark", root) == root / "TNG100" / "TNG100-3-Dark")
    finally:
        shutil.rmtree(root)


def figures_root_rule():
    """Every figure under one root: DTFE_FIGURES_ROOT, else python/figures -- one rule for the launcher and its
    scripts; saved specs holding python/figures read as the current root."""
    check("this suite's figures go to its temp folder (DTFE_FIGURES_ROOT, set before runspec is imported)",
          rs.FIGURES_ROOT == Path(FIGS_TMP), str(rs.FIGURES_ROOT))
    env = {k: v for k, v in os.environ.items() if k != "DTFE_FIGURES_ROOT"}
    out = subprocess.run([sys.executable, "-c", "import config, dtfelib.cli as c; print(c.FIGURES_ROOT); "
                          "print(config.LOCAL_FIGURES_ROOT)"], cwd=ROOT / "python", env=env,
                         capture_output=True, text=True).stdout.split("\n")
    want = str(ROOT / "python" / "figures")
    check("without DTFE_FIGURES_ROOT: python/figures -- the same for dtfelib and config", out[:2] == [want, want],
          str(out[:2]))
    legacy = str(rs.LEGACY_FIGURES_ROOT)
    pipe, plot = rs.PipelineSpec.from_dict({"figures_root": legacy}), rs.PlotSpec.from_dict({"figures_root": legacy})
    own = rs.PlotSpec.from_dict({"figures_root": "/somewhere/else"})
    check("settings and queued jobs whose figure root is the legacy python/figures load as FIGURES_ROOT; a folder of "
          "the user's own stays", pipe.figures_root == plot.figures_root == str(rs.FIGURES_ROOT)
          and own.figures_root == "/somewhere/else", f"{pipe.figures_root} {plot.figures_root} {own.figures_root}")


def void_outline_geometry():
    """The Explore map's void outlines (grids.void_circles): where a sphere cuts the slice, in image pixels (x along
    the first remaining axis, y DOWN the second), its periodic images across the box edges, a zoom window with
    non-square pixels, the wrap along the slice normal, and what is left out."""
    import numpy as np
    import grids as G
    print("void outlines:")
    L = 100.0
    c = np.array([[10.0, 20.0, 53.0],             # 3 Mpc off the slice: a circle of radius 4
                  [98.0, 50.0, 50.0],             # in the slice, straddling x = L: two circles
                  [30.0, 30.0, 97.0],             # 95 Mpc away along z, i.e. 5 across the wrap from z = 2
                  [60.0, 60.0, 50.0]])            # no radius (NaN): left out
    r = np.array([5.0, 6.0, 6.0, np.nan])
    got = G.void_circles(c, r, L, 2, 50.0, [0, 1], (0.0, 0.0), (L, L), 200, 200)
    want = [(20.0, 160.0, 8.0, 8.0, 0), (-4.0, 100.0, 12.0, 12.0, 1), (196.0, 100.0, 12.0, 12.0, 1)]
    check("a sphere 3 Mpc off the slice leaves a circle of sqrt(R^2 - d^2), at (x, L - y) in image pixels; one "
          "straddling the box edge is drawn on both sides; a void without R_v is not",
          np.allclose(sorted(got), sorted(want)), str(got))
    near = G.void_circles(c, r, L, 2, 2.0, [0, 1], (0.0, 0.0), (L, L), 200, 200)
    check("the distance to the slice wraps too (z = 97 is 5 Mpc from z = 2)",
          [k for *_, k in near] == [2] and np.isclose(near[0][2], 2 * np.sqrt(11.0)), str(near))
    check("... unless the box is not periodic",
          G.void_circles(c, r, L, 2, 2.0, [0, 1], (0.0, 0.0), (L, L), 200, 200, periodic=False) == [])
    zoom = G.void_circles(c, r, L, 2, 50.0, [0, 1], (90.0, 40.0), (100.0, 60.0), 100, 100)
    check("a zoom window: only what reaches it, in its own pixels (10 per Mpc across, 5 up: an ellipse)",
          np.allclose(zoom, [(80.0, 50.0, 60.0, 30.0, 1)]), str(zoom))
    xs = G.void_circles(c, r, L, 0, 10.0, [1, 2], (0.0, 0.0), (L, L), 100, 100)
    check("an x slice: the remaining axes y, z", np.allclose(xs, [(20.0, 47.0, 5.0, 5.0, 0)]), str(xs))


def run_log_contract():
    """The run-log contract: a run writes '<prefix>.runlog.partial'; exit 0 renames it onto
    '<prefix>.runlog' (the log of the grids on disk), a failure or stop onto '<prefix>.runlog.failed' (the old
    '.runlog' untouched). The Runs browser reads the three names as ONE output set: the log of record, and a
    newer failed, running or killed attempt noted on it (results.py). Plus the deposit settings and the GPU-retry
    wording a log is read for, and the Pipeline job's render chained to its pipeline call."""
    import results as R
    print("run-log contract:")
    base = Path(tempfile.mkdtemp(prefix="gui_runlogs_"))
    child = None
    try:
        check("log_prefix: the three log names give the output prefix; any other name stays as it is",
              [R.log_prefix(n) for n in ("a.runlog", "a.runlog.failed", "a.runlog.partial", "a.a_den", "a.runlog.tmp")]
              == ["a", "a", "a", "a.a_den", "a.runlog.tmp"])
        d = base / "own"
        d.mkdir()
        t0 = time.time() - 3600.0        # every mtime set by hand: 'newer than the log' and the 5 s rule must not ride on the clock
        old = ("Build: PS-DTFE 1111111-dirty (built 2026-10-05T10:00:00Z)\n"
               "RUNNING: /x/PS-DTFE in.hdf5 run --grid 512 --periodic --field density_a --ps-vertex-mass\n"
               "   peak memory (RSS) : 2.0 GB\n   total wall time   : 1m 2s\n")
        new = ("Build: PS-DTFE 2222222-dirty (built 2026-10-07T10:00:00Z)\n"         # the attempt: other settings
               "RUNNING: /x/PS-DTFE in.hdf5 run --grid 256 --periodic --field density_a --ps-vertex-mass\n"
               "terminate called after throwing an instance of 'std::bad_alloc'\n")
        for sfx in ("a_den", "a_vel"):
            (d / f"run.{sfx}").write_bytes(b"\0" * 32)
            os.utime(d / f"run.{sfx}", (t0 - 10, t0 - 10))
        _write_log(d / "run.runlog", old, t0)
        for sfx in (".runlog.failed", ".runlog.partial", ".runlog.tmp"):
            _write_log(d / ("run" + sfx), old, t0 + 30)      # a summary and a wall time in each
        names = [p.name for p in R.parse_runlog(d / "run.runlog").outputs]
        check("a run's outputs leave out every log name: '.runlog.failed', '.runlog.partial' and a script's '.runlog.tmp' "
              "(their suffixes are '.failed', '.partial', '.tmp')", names == ["run.a_den", "run.a_vel"], str(names))
        att = [R.parse_runlog(d / ("run" + sfx)) for sfx in (".runlog.failed", ".runlog.partial")]
        check("... a '.failed' or '.partial' log is never ok, even with a summary and a wall time; its prefix is the output's",
              R.parse_runlog(d / "run.runlog").ok and all(not a.ok and a.wall_seconds == 62 and a.prefix == "run"
                                                          and [p.name for p in a.outputs] == names for a in att),
              str([(a.ok, a.wall_seconds, a.prefix) for a in att]))
        (d / "run.runlog.partial").unlink()
        _write_log(d / "run.runlog.failed", new, t0 + 60)      # a NEWER failed re-run under the same name
        none = base / "no data root"
        recs = R.scan_runs(none, [d])
        r = recs[0] if len(recs) == 1 else None
        reasons = R.stale_reasons(r) if r else []
        check("a '.runlog' and a newer '.runlog.failed' (and a '.runlog.tmp'): ONE record, the log of record with its "
              "own build and settings, ok; the attempt noted on it ('failed', its likely cause, nothing rewritten)",
              r is not None and r.path == d / "run.runlog" and r.ok and r.build_rev == "1111111-dirty" and r.grid == 512
              and r.attempt == "failed" and r.attempt_log == d / "run.runlog.failed" and r.attempt_cause == "Out of memory"
              and r.attempt_rewrote == 0, str(recs))
        check("... the Status column's note: one 'info' line naming the .failed file and the cause (the outputs are this "
              "run's still)", len(reasons) == 1 and reasons[0][0] == "info" and "run.runlog.failed" in reasons[0][1]
              and "Out of memory" in reasons[0][1] and "still this run's" in reasons[0][1], str(reasons))
        os.utime(d / "run.runlog.failed", (t0 - 60, t0 - 60))
        r = R.scan_runs(none, [d])[0]
        check("... a .failed OLDER than the .runlog is no attempt after it: nothing noted", r.attempt == ""
              and r.attempt_log is None and R.stale_reasons(r) == [], f"{r.attempt!r} {R.stale_reasons(r)}")
        os.utime(d / "run.runlog.failed", (t0 + 60, t0 + 60))
        os.utime(d / "run.a_den", (t0 + 10, t0 + 10))         # written by the attempt, after the log of record ended
        r = R.scan_runs(none, [d])[0]
        note = R.attempt_note(r)
        check("an output more than 5 s newer than the .runlog: the attempt rewrote it, a 'warning' to regenerate",
              r.attempt_rewrote == 1 and note is not None and note[0] == "warning" and "regenerate" in note[1]
              and "rewrote 1 of the outputs" in note[1], f"{r.attempt_rewrote} {note}")
        os.utime(d / "run.a_den", (t0 - 10, t0 - 10))
        (d / "run.runlog").unlink()
        recs = R.scan_runs(none, [d])
        r = recs[0] if len(recs) == 1 else None
        note = R.attempt_note(r) if r else None
        check("only a '.runlog.failed' (a first run that failed): it is the record, never ok, its 'warning' says none "
              "has finished", r is not None and r.path == d / "run.runlog.failed" and r.prefix == "run" and not r.ok
              and r.attempt == "failed" and note is not None and note[0] == "warning" and "none has finished" in note[1]
              and note in R.stale_reasons(r), f"{recs} {note}")
        # only '.runlog.partial': killed outright, or still going while a process names its root (one ps call)
        (d / "run.runlog.failed").unlink()
        (d / "run.runlog.tmp").unlink()
        _write_log(d / "run.runlog.partial", new.split("terminate")[0], t0 + 30)
        _write_log(d / "run2.runlog.partial", new.split("terminate")[0].replace(" run ", " run2 "), t0 + 30)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(d / "run2")])
        got = {rec.prefix: rec for rec in R.scan_runs(none, [d])}
        if shutil.which("ps"):
            check("only a '.runlog.partial': no live process names '<dir>/run' (a sibling '<dir>/run2' does not count) -> "
                  "'killed'; the one a live process names -> 'running'; neither ok, each the record",
                  set(got) == {"run", "run2"} and got["run"].attempt == "killed" and got["run2"].attempt == "running"
                  and not got["run"].ok and not got["run2"].ok and got["run"].path == d / "run.runlog.partial"
                  and R.attempt_note(got["run2"])[0] == "info",
                  str({k: (v.attempt, v.ok, v.path.name) for k, v in got.items()}))
        else:
            check("only a '.runlog.partial', without ps: 'partial' (killed or still going), never ok",
                  set(got) == {"run", "run2"} and all(v.attempt == "partial" and not v.ok for v in got.values()),
                  str({k: v.attempt for k, v in got.items()}))
        # a data-root snapdir that is also a custom output folder: one record, with its simulation
        sd = base / "data" / "FAM" / "FAM-1-Dark" / "snapdir_099"
        sd.mkdir(parents=True)
        (sd / "ps_output.a_den").write_bytes(b"\0" * 32)
        _write_log(sd / "ps_output.runlog", old.replace(" run ", f" {sd / 'ps_output'} "), t0)
        _write_log(sd / "ps_output.runlog.partial", new, t0 + 30)
        _write_log(sd / "ps_output.runlog.failed", new, t0 + 60)
        recs = R.scan_runs(base / "data", [sd])
        r = recs[0] if len(recs) == 1 else None
        check("a data-root snapdir also given as an extra folder: one record, the data root's simulation kept, the newer "
              "attempt (the .failed) noted; log_files() lists every log of the set (what Move to Trash takes)",
              r is not None and r.sim == "FAM-1-Dark" and r.snap == 99 and r.path == sd / "ps_output.runlog"
              and r.attempt == "failed" and [p.name for p in r.log_files()]
              == ["ps_output.runlog", "ps_output.runlog.failed", "ps_output.runlog.partial"],
              str([(x.sim, x.path, x.attempt, [p.name for p in x.log_files()]) for x in recs]))
        # the linear deposit in 'Load settings' and the Options column; the PS-DTFE wording of the GPU retry
        _write_log(d / "lin.runlog", "RUNNING: /x/PS-DTFE in.hdf5 lin --grid 64 --ps-linear-deposit\n")
        _write_log(d / "std.runlog", "RUNNING: /x/PS-DTFE in.hdf5 std --grid 64\n")
        lin, std = R.parse_runlog(d / "lin.runlog"), R.parse_runlog(d / "std.runlog")
        check("settings(): a --ps-linear-deposit run turns the box on, a standard-deposit run turns it OFF (the key is "
              "there); the Options column tells the two apart",
              lin.settings().get("linear_deposit") is True and std.settings().get("linear_deposit") is False
              and "linear deposit" in lin.options_text() and "linear deposit" not in std.options_text(),
              f"{lin.settings()} {std.settings()} {lin.options_text()!r}")
        # the face gradient the same way (a RunSpec field, so 'Load settings' sets it): the box is off only for a run
        # that named the stream average; the Options column reads the route from the header's 'velocity gradient' line
        # (the binary resolves a command without a route flag to its default, D13), else from the command (an older
        # binary's run without '--face-gradient' holds the stream average)
        _write_log(d / "fg.runlog", "RUNNING: /x/PS-DTFE in.hdf5 fg --grid 64 --face-gradient\n")
        _write_log(d / "pt.runlog", "RUNNING: /x/PS-DTFE in.hdf5 pt --grid 64 --per-tet-gradient\n")
        _write_log(d / "fh.runlog", "RUNNING: /x/PS-DTFE in.hdf5 fh --grid 64 --exact --ps-volume-weighted\n"
                   "\t velocity gradient      : from EXACT FACE sums, the gradient of the volume-weighted MEAN "
                   "velocity (the default where it applies; '.a_velDivStreams' holds the stream average's "
                   "divergence)\n")
        _write_log(d / "sh.runlog", "RUNNING: /x/PS-DTFE in.hdf5 sh --grid 64 --sampled --ps-volume-weighted\n"
                   "\t velocity gradient      : the stream average of each stream's own gradient (the face gradient "
                   "needs the exact deposit; this run deposits sampled (--sampled))\n")
        fg, pt, fh, sh = (R.parse_runlog(d / f"{n}.runlog") for n in ("fg", "pt", "fh", "sh"))
        again = R.parse_runlog(d / "fh.runlog")             # the same log again: from the cache of read logs
        check("settings(): a --face-gradient run turns the face-gradient box on, a --per-tet-gradient run OFF, a run "
              "that named no route on (the binary's default; the key is there); the Options column says 'face "
              "gradient' for a --face-gradient run and for one whose header's line names the face sums (also from the "
              "cache of read logs), not for a --per-tet-gradient run, a header naming the stream average or an older "
              "log with neither",
              fg.settings().get("face_gradient") is True and pt.settings().get("face_gradient") is False
              and std.settings().get("face_gradient") is True and fh.settings().get("face_gradient") is True
              and "face gradient" in fg.options_text() and "face gradient" in fh.options_text()
              and "face gradient" not in pt.options_text() and "face gradient" not in sh.options_text()
              and "face gradient" not in std.options_text()
              and (fh.header_gradient, sh.header_gradient, std.header_gradient) == ("face", "per-tet", "")
              and again.header_gradient == "face" and again.face_gradient and not sh.face_gradient
              and all(k in rs.RunSpec.__dataclass_fields__ or k in rs.REMOVED_SETTINGS
                      for k in rs.migrate_quality(fg.settings())),
              f"{fg.settings()} {fg.options_text()!r} {fh.header_gradient!r} {again.header_gradient!r}")
        # a standard-DTFE run's cell average reaches 'Load settings' both ways
        _write_log(d / "dx.runlog", "RUNNING: /x/DTFE in.hdf5 dx --grid 64 --periodic --exact-average --padding 15\n")
        _write_log(d / "dm.runlog", "RUNNING: /x/DTFE in.hdf5 dm --grid 64 --periodic --padding 15\n")
        dx, dm = R.parse_runlog(d / "dx.runlog"), R.parse_runlog(d / "dm.runlog")
        check("settings(): a standard run with --exact-average loads the exact choice, one without the Monte-Carlo one; "
              "the Options column names the quality as the launcher does, Exact or Fast (no 'nSub', 'sampled' or "
              "'exact average'), for both estimators",
              dx.settings().get("deposit") == "exact" and dm.settings().get("deposit") == "sampled"
              and dx.settings().get("estimator") == "dtfe" and dx.options_text().split(", ")[0] == "Exact"
              and dm.options_text().split(", ")[0] == "Fast" and lin.options_text().split(", ")[0] == "Fast"
              and fh.options_text().split(", ")[0] == "Exact" and sh.options_text().split(", ")[0] == "Fast"
              and not [t for t in (dx, dm, lin, std, fg, pt, fh, sh)
                       if any(w in t.options_text() for w in ("nSub", "sampled", "exact average"))],
              f"{dx.settings()} {dm.settings()} {dx.options_text()!r} {dm.options_text()!r}")
        # 'Load settings' (app._load_run_settings) reads a log's cell average as the quality through migrate_quality,
        # and sets only the RunSpec fields: the settings the two modes removed are left out (a sampled run's count
        # where Fast takes another comes back as the job's Advanced Samples: rs.keep_run_count, checked below)
        loaded = {n: {k: v for k, v in rs.migrate_quality(r.settings()).items() if k in rs.RunSpec.__dataclass_fields__}
                  for n, r in (("dx", dx), ("dm", dm), ("fg", fg), ("lin", lin))}
        check("'Load settings': a log's cell average becomes the quality (an exact run Exact, a sampled one Fast), "
              "the removed settings are not set",
              loaded["dx"]["quality"] == "exact" and loaded["dm"]["quality"] == "fast"
              and loaded["fg"]["quality"] == "fast" and not any(set(v) & set(rs.REMOVED_SETTINGS)
                                                                for v in loaded.values())
              and rs.RunSpec(**loaded["dm"]).quality == "fast", str(loaded))
        # the exact average is the default since 2026-11-09: a log is read by its header's cell-average line first
        # (a command without a flag ran exact), then by the command ('--exact' and its older names against
        # '--sampled'); an older log with neither was sampled (dm above, std)
        std_x = ("\t volume averaging       : EXACT cell-tetrahedron integration of the linear interpolant (--exact, "
                 "the default; r3d, or a z-sweep on the GPU with --gpu; --subsamples ignored)\n")
        heads = {
            "hx": "RUNNING: /x/DTFE in.hdf5 hx --grid 64 --field density_a\n" + std_x,
            "hs": "RUNNING: /x/DTFE in.hdf5 hs --grid 64 --subsamples 2 --field density_a\n\t volume averaging       : "
                  "SAMPLED (a sampling count or method was given; --exact integrates the cells exactly)\n",
            "hc": "RUNNING: /x/DTFE in.hdf5 hc --grid 64 --exact --redshift-cone 1 2 0 1 0 1 --field density_a\n"
                  "\t volume averaging       : SAMPLED (a redshift cone grid has no exact form)\n",
            "px": "RUNNING: /x/PS-DTFE in.hdf5 px --grid 64 --field density_a\n\t grid deposit           : EXACT "
                  "tetrahedron-cell intersection moments (--exact, the default; r3d clipping; nSub ignored)\n",
            "ps": "RUNNING: /x/PS-DTFE in.hdf5 ps --grid 64 --sampled --field density_a\n\t grid deposit           : "
                  "SAMPLED on nSub^3 sub-points (--sampled)\n",
            "po": "RUNNING: /x/PS-DTFE in.hdf5 po --grid 64 --ps-exact-deposit --field density_a\n\t grid deposit"
                  "           : EXACT tetrahedron-cell intersection moments (--exact-average; r3d clipping; nSub ignored)\n",
            "pf": "RUNNING: /x/PS-DTFE in.hdf5 pf --grid 64 --exact --field density\n",
            "two": "RUNNING: /x/DTFE in.hdf5 two --grid 64 --sampled --field density_a\n\t volume averaging       : "
                   "SAMPLED (--sampled)\nRUNNING: /x/DTFE in.hdf5 two --grid 64 --field density_a\n" + std_x,
        }
        for name, text in heads.items():
            _write_log(d / f"{name}.runlog", text)
        hr = {name: R.parse_runlog(d / f"{name}.runlog") for name in heads}
        got = {name: (r.exact, r.settings().get("deposit")) for name, r in hr.items()}
        check("the cell average read from a log: the header line first (a flagless new run exact, a count or a "
              "redshift cone sampled even beside '--exact'), then the command ('--exact', an old "
              "'--ps-exact-deposit'), an older flagless log sampled; a log of two runs reads its first; the Options "
              "column says Exact or Fast by it",
              got == {"hx": (True, "exact"), "hs": (False, "sampled"), "hc": (False, "sampled"),
                      "px": (True, "exact"), "ps": (False, "sampled"), "po": (True, "exact"), "pf": (True, "exact"),
                      "two": (False, "sampled")}
              and hr["hx"].header_average == "exact" and hr["pf"].header_average == ""
              and hr["hx"].options_text().split(", ")[0] == "Exact" and hr["px"].options_text().split(", ")[0] == "Exact"
              and hr["hs"].options_text().split(", ")[0] == "Fast" and hr["ps"].options_text().split(", ")[0] == "Fast"
              and not dm.exact and not std.exact, str(got))
        # the sampled deposit's count read from a log: the command's '--subsamples k', else the header's count line
        # (every sampled PS-DTFE run with an '_a' field prints it, src/user_options.cc). An older run that names none ran at the
        # binary's default then, 3 per cell axis, and must not read as Fast's one now: not in the Options column, not
        # in a Fast job's time estimate, not in the job 'Load settings' makes (rs.keep_run_count: its Advanced
        # Samples). A new log reads its own count
        def nsub_line(k):               # (an older binary's words at 3, today's at 1: the count is read alike)
            tail = ("one point per '_a' cell, at its centre: noisy per-cell values, smooth before use" if k == 1 else
                    f"each '_a' cell volume-averaged over nSub^3 = {k ** 3} points; this sets the dominant '_a' "
                    "interpolation cost")
            return f"\t avg sub-samples (nSub) : {k}   ({tail})\n"
        wall, samp = "   total wall time   : 12s\n", "\t grid deposit           : SAMPLED on nSub^3 sub-points (--sampled)\n"
        counts = {"cold": "RUNNING: /x/PS-DTFE in.hdf5 cold --grid 64 --field density_a\n" + nsub_line(3) + wall,
                  "cnew": "RUNNING: /x/PS-DTFE in.hdf5 cnew --grid 64 --sampled --subsamples 1 --field density_a\n"
                          + samp + nsub_line(1) + wall,
                  "cbare": "RUNNING: /x/PS-DTFE in.hdf5 cbare --grid 64 --sampled --field density_a\n" + samp
                           + nsub_line(1) + wall}
        for name, text in counts.items():
            _write_log(d / f"{name}.runlog", text)
        cr = {name: R.parse_runlog(d / f"{name}.runlog") for name in counts}
        cached = R.parse_runlog(d / "cold.runlog")          # the same log again: from the cache of read logs
        jobs = {name: rs.RunSpec(data_root=str(base), sim="S", snapshots=[99], quality="fast") for name in counts}
        for name, job in jobs.items():
            rs.keep_run_count(job, cr[name])
        argvs = {name: job.command()[0] for name, job in jobs.items()}
        est = {n: R.time_estimate(list(cr.values()), sim="", grid=64, estimator="ps", gpu=False, sliced=False,
                                  exact=False, nsub=n) for n in (1, 3, None)}
        check("the sampled deposit's count from a log: the command's, else the header's count line (an older log "
              "with no count reads 3 from it, also from the cache of read logs; a new log its own); the Options column "
              "names it, a Fast time estimate takes runs at the job's count only, and 'Load settings' of the older run "
              "keeps its 3 as the job's Advanced Samples (-n 3), of a new one nothing (-n 1)",
              {n: r.nsub for n, r in cr.items()} == {"cold": 3, "cnew": 1, "cbare": 1}
              and (cr["cold"].header_nsub, cr["cnew"].header_nsub, cached.header_nsub, cached.nsub) == (3, 1, 3, 3)
              and not any(r.exact for r in cr.values()) and cr["cold"].settings().get("nsub") == 3
              and "3 samples per cell axis" in cr["cold"].options_text()
              and "1 sample per cell axis" in cr["cbare"].options_text()
              and est == {1: (12, 2), 3: (12, 1), None: (12, 3)}
              and {n: j.advanced for n, j in jobs.items()} == {"cold": {"nsub": 3}, "cnew": {}, "cbare": {}}
              and {n: a[a.index("-n") + 1] for n, a in argvs.items()} == {"cold": "3", "cnew": "1", "cbare": "1"},
              f"{ {n: (r.nsub, r.header_nsub, r.options_text()) for n, r in cr.items()} } {est} "
              f"{ {n: j.advanced for n, j in jobs.items()} } {argvs}")
        _write_log(d / "gpu.runlog", "Build: PS-DTFE 3333333 (built 2026-10-01T09:00:00Z)\n"
                   "RUNNING: /x/PS-DTFE in.hdf5 gpu --grid 64 --periodic --partition 2 2 2 --ps-gpu --ps-vertex-mass\n"
                   "PS-DTFE Metal: Metal command buffer failed: Impacting Interactivity; retrying the partition deposit "
                   "from scratch (attempt 2/3)\n   peak memory (RSS) : 1.0 GB\n   total wall time   : 12s\n")
        g = R.parse_runlog(d / "gpu.runlog")
        check("PS-DTFE's 'retrying the partition deposit from scratch' counts as a GPU retry; built before the DTFE "
              "retry fix, a PS run is still not flagged (that fix is the standard DTFE's)",
              g.gpu_retried and g.estimator == "ps" and g.gpu and R.stale_reasons(g) == [], str(R.stale_reasons(g)))
        steps = rs.PipelineSpec(data_root=str(base / "data"), sims=["FAM-1-Dark", "FAM-2-Dark"], nu=64).steps()
        check("a Pipeline job's render steps wait for its pipeline call: a pipeline that computed nothing exits 1 "
              "(run_ps_pipeline.sh) and the job then skips them",
              [(st.label, st.after) for st in steps] == [("run_ps_pipeline.sh FAM-1-Dark FAM-2-Dark", ""),
                                                         ("plot_pointeval.py FAM-1-Dark", steps[0].label),
                                                         ("plot_pointeval.py FAM-2-Dark", steps[0].label)],
              str([(st.label, st.after) for st in steps]))
    finally:
        if child is not None:
            child.kill()
            child.wait()
        shutil.rmtree(base, ignore_errors=True)


def ordered_merge_settings():
    """The ordered merge from every tab into its run and back through 'Load settings', and a killed run's
    checkpoint directory in the Runs browser."""
    import results as R
    print("launcher: the ordered merge, checkpoints:")
    base = Path(tempfile.mkdtemp(prefix="gui_wave2_"))
    try:
        snap = base / "cloud.hdf5"
        on = rs.RunSpec(sim="TNG50-4", snapshots=[99]).command()[1]
        off = rs.RunSpec(sim="TNG50-4", snapshots=[99], quality="fast").command()[1]
        pon = rs.PipelineSpec(sims=["TNG50-4"])
        # the index order in both qualities (the staged merge: run_ps_dtfe.sh's and the binary's default), given
        pair = lambda cmd, mode: any(cmd[i:i + 2] == ["--ps-merge", mode] for i in range(len(cmd)))  # noqa: E731
        check("ordered merge: Grids and Pipeline jobs set ORDERED_MERGE=1 for run_ps_dtfe.sh in both qualities "
              "(spelled out, so a stray export cannot leak in), the Pipeline's memory check too; the Custom tab always "
              "names the merge (PS only): --ps-merge index-order",
              on.get("ORDERED_MERGE") == "1" and off.get("ORDERED_MERGE") == "1"
              and pon.command()[1].get("ORDERED_MERGE") == "1"
              and pon.check_step("TNG50-4", [99], 1024).env.get("ORDERED_MERGE") == "1"
              and rs.PipelineSpec(sims=["TNG50-4"], quality="fast").command()[1].get("ORDERED_MERGE") == "1"
              and pair(rs.CustomSpec(input_file=str(snap)).command(), "index-order")
              and pair(rs.CustomSpec(input_file=str(snap), quality="fast").command(), "index-order")
              and rs.CustomSpec(input_file=str(snap)).command().count("--ps-merge") == 1
              and not any({"--ordered-merge", "--completion-order-merge", "--ps-merge"} & set(
                  rs.CustomSpec(input_file=str(snap), estimator="dtfe", quality=q).command()) for q in rs.QUALITIES),
              str(on))
        d = base / "runs"
        d.mkdir()
        _write_log(d / "om.runlog", "RUNNING: /x/PS-DTFE in.hdf5 om --grid 64 --partition 2 2 2 --ordered-merge\n"
                   "   total wall time   : 12s\n")
        _write_log(d / "plain.runlog", "RUNNING: /x/PS-DTFE in.hdf5 plain --grid 64 --partition 2 2 2\n"
                   "   total wall time   : 12s\n")
        _write_log(d / "co.runlog", "RUNNING: /x/PS-DTFE in.hdf5 co --grid 64 --partition 2 2 2 --completion-order-merge\n"
                   "   total wall time   : 12s\n")
        om, plain = R.parse_runlog(d / "om.runlog"), R.parse_runlog(d / "plain.runlog")
        co = R.parse_runlog(d / "co.runlog")
        # a run without either flag loads the default (on); only '--completion-order-merge' turns it off
        check("ordered merge: settings() carries it (ON for a run without a flag, the default; OFF for a "
              "--completion-order-merge run), the Options column says which, and the settings load into a RunSpec "
              "(the merge, a setting the two modes removed, left out)",
              om.settings().get("ordered_merge") is True and plain.settings().get("ordered_merge") is True
              and co.settings().get("ordered_merge") is False
              and "ordered merge" in om.options_text() and "ordered merge" not in plain.options_text()
              and "completion-order merge" in co.options_text()
              and all(k in rs.RunSpec.__dataclass_fields__ or k in rs.REMOVED_SETTINGS
                      for k in rs.migrate_quality(om.settings())), f"{om.settings()} {om.options_text()!r}")

        # a killed partitioned run leaves '<prefix>.runlog.checkpoint/' (a directory); it is a run file, never an
        # output or a log, and with no live process the run is 'resumable' -- when run_ps_dtfe.sh ran it (a Grids or
        # Pipeline job: under the data root's snapdir_*), and its attempt did not fail on a refused resume
        t0 = time.time() - 600.0
        sd = base / "root" / "TNGK-1-Dark" / "snapdir_099"           # where run_ps_dtfe.sh writes
        sd.mkdir(parents=True)
        for sfx in ("a_den", "a_vel"):
            (sd / f"ck.{sfx}").write_bytes(b"\0" * 32)
        _write_log(sd / "ck.runlog.partial", "RUNNING: /x/PS-DTFE in.hdf5 ck --grid 64 --partition 2 2 2\n"
                   "PS-DTFE: checkpoints of the merged grids go to 'ck.runlog.checkpoint' (at most ~5% of the run; "
                   "removed when the outputs are written; '--resume' continues a killed run from it).\n"
                   "partition 3/8 done\n", t0)
        ckd = sd / "ck.runlog.checkpoint"
        ckd.mkdir()
        (ckd / "manifest.json").write_text("{}")
        (sd / "ck.runlog.checkpoint.tmp").write_text("")      # a checkpoint being written: a run file too
        _write_log(sd / "ck2.runlog", "RUNNING: /x/PS-DTFE in.hdf5 ck2 --grid 64\n   total wall time   : 1s\n", t0 - 60)
        _write_log(sd / "ck2.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 ck2 --grid 64 --partition 2 2 2\n"
                   "~~~ ERROR ~~~\n", t0)
        (sd / "ck2.runlog.checkpoint").mkdir()
        # the resume refused (the settings changed since the kill): the same job fails the same way every time
        _write_log(sd / "rf.runlog", "RUNNING: /x/PS-DTFE in.hdf5 rf --grid 64\n   total wall time   : 1s\n", t0 - 60)
        _write_log(sd / "rf.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 rf --grid 128 --partition 2 2 2 --resume\n"
                   "~~~ ERROR ~~~ '--resume' refused: the checkpoint 'rf.runlog.checkpoint' was written by a run with "
                   "other settings.\n", t0)
        (sd / "rf.runlog.checkpoint").mkdir()
        _write_log(sd / "rf0.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 rf0 --grid 64 --partition 2 2 2 --resume\n"
                   "~~~ ERROR ~~~ '--resume': nothing to resume -- no manifest. Run without '--resume' to start the "
                   "run.\n", t0)
        (sd / "rf0.runlog.checkpoint").mkdir()
        # a Custom run (its own output folder): it never resumes, a checkpoint or not
        _write_log(d / "cu.runlog.failed", "RUNNING: /x/PS-DTFE in.hdf5 cu --grid 64 --partition 2 2 2\n"
                   "~~~ ERROR ~~~\n", t0)
        (d / "cu.runlog.checkpoint").mkdir()
        _write_log(d / "gone.runlog.partial", "RUNNING: /x/PS-DTFE in.hdf5 gone --grid 64\n", t0)
        # a killed partitioned STANDARD run (run_dtfe.sh resumes it too, DTFE_RESUME) is 'resumable' alike
        _write_log(sd / "output.runlog.partial", "RUNNING: /x/DTFE in.hdf5 output --grid 64 --partition 2 2 2\n"
                   "DTFE: checkpoints of the merged grids go to 'output.runlog.checkpoint' (at most ~5% of the run; "
                   "removed when the outputs are written; '--resume' continues a killed run from it).\n", t0)
        (sd / "output.runlog.checkpoint").mkdir()
        check("a checkpoint directory and its files are run files (never an output, never a log)",
              R.is_run_file("ck.runlog.checkpoint") and R.is_run_file("ck.runlog.checkpoint.tmp")
              and not R.is_run_file("ck.runlog") and not R.is_run_file("ck.a_den"))
        got = {r.prefix: r for r in R.scan_runs(base / "root", [d])}
        ck, ck2, gone, sk = got.get("ck"), got.get("ck2"), got.get("gone"), got.get("output")
        rf, rf0, cu = got.get("rf"), got.get("rf0"), got.get("cu")
        if shutil.which("ps"):
            check("a killed run with its checkpoint is 'resumable' (a failed attempt after a finished log too); one "
                  "killed without a checkpoint stays 'killed'; the checkpoint is not among the outputs",
                  ck is not None and ck.attempt == "resumable" and not ck.ok
                  and [p.name for p in ck.outputs] == ["ck.a_den", "ck.a_vel"]
                  and ck2 is not None and ck2.ok and ck2.attempt == "resumable"
                  and gone is not None and gone.attempt == "killed",
                  str({k: (v.attempt, [p.name for p in v.outputs]) for k, v in got.items()}))
            check("... a killed standard-DTFE run with its checkpoint too (run_dtfe.sh resumes it)",
                  sk is not None and sk.estimator == "dtfe" and sk.attempt == "resumable",
                  str(None if sk is None else (sk.estimator, sk.attempt)))
            n1, n2 = R.attempt_note(ck), R.attempt_note(ck2)
            check("... and its note says the same job resumes it, and names the start-over tick",
                  n1 is not None and "resum" in n1[1] and n2 is not None and "resum" in n2[1]
                  and all("Start over" in m and "DTFE_RESUME=0" in m and "PS_RESUME" not in m for _, m in (n1, n2))
                  and not [m for _, m in (n1, n2) if "--" in m], f"{n1} {n2}")
        else:
            check("without ps a '.runlog.partial' stays 'partial' (its process cannot be ruled out), checkpoint or not",
                  ck is not None and ck.attempt == "partial")
        # a refused resume is never 'resumable' (rerunning fails again), nor is a Custom run's checkpoint
        nr, nr0 = (R.attempt_note(r) if r is not None else None for r in (rf, rf0))
        check("a resume the binary refused (other settings; nothing to resume) stays 'failed', its note and cause name "
              "'Start over' (DTFE_RESUME=0, no '--'); a Custom run's checkpoint is never 'resumable'",
              rf is not None and rf.attempt == "failed" and getattr(rf, "attempt_resume_refused", False) and rf.ok
              and rf0 is not None and rf0.attempt == "failed" and getattr(rf0, "attempt_resume_refused", False)
              and "Resume refused" in rf.attempt_cause
              and all(n is not None and n[0] == "warning" and "Start over" in n[1] and "DTFE_RESUME=0" in n[1]
                      and "--" not in n[1] for n in (nr, nr0))
              and cu is not None and cu.attempt == "failed" and not getattr(cu, "attempt_resume_refused", False),
              str({k: (v.attempt, getattr(v, "attempt_resume_refused", None), v.attempt_cause) for k, v in got.items()}) + f" {nr} {nr0}")
        dg = R.diagnose("~~~ ERROR ~~~ '--resume' refused: the checkpoint 'x' was written by a run with other settings.")
        check("... diagnose() offers the start-over tick for it; the binary's info lines about '--resume' are no refusal",
              [d_.action for d_ in dg] == ["start_over"] and "--" not in dg[0].advice
              and hasattr(R, "RESUME_REFUSED") and not R.RESUME_REFUSED.search("this run starts over and replaces it ('--resume' continues it).")
              and not R.RESUME_REFUSED.search("removed when the outputs are written; '--resume' continues a killed "
                                              "run from it)"), str(dg))
        # the start-over tick: DTFE_RESUME=0 for both drivers (PS_RESUME, run_ps_dtfe.sh's older name, the same until
        # P6.6), spelled out as 1 otherwise (a stray export cannot leak in)
        ps_resume = lambda e: (e.get("DTFE_RESUME"), e.get("PS_RESUME"))     # noqa: E731
        check("start over: Grids and Pipeline jobs set DTFE_RESUME=0 (and PS_RESUME=0 for PS-DTFE) when ticked, 1 "
              "otherwise; the memory check ignores it",
              ps_resume(rs.RunSpec(sim="TNG50-4", snapshots=[99], start_over=True).command()[1]) == ("0", "0")
              and ps_resume(off) == ("1", "1")
              and ps_resume(rs.PipelineSpec(sims=["TNG50-4"], start_over=True).command()[1]) == ("0", "0")
              and ps_resume(rs.PipelineSpec(sims=["TNG50-4"]).command()[1]) == ("1", "1")
              and rs.RunSpec(estimator="dtfe", sim="TNG50-4", snapshots=[99], start_over=True).command()[1].get("DTFE_RESUME") == "0"
              and rs.RunSpec(estimator="dtfe", sim="TNG50-4", snapshots=[99]).command()[1].get("DTFE_RESUME") == "1"
              and rs.RunSpec(start_over=True).memory_key() == rs.RunSpec().memory_key()
              and rs.PipelineSpec(start_over=True).memory_key() == rs.PipelineSpec().memory_key()
              and "start_over" not in om.settings(), str(off))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def cpp_fix_flags():
    """results.KNOWN_FIXES for the C++ fixes of an interlaced grid without a region (pass 2 gridded
    the same cells), PCS (no padding, half-cell shift), CIC (extrapolated face and seam cells), and a fractional
    region or padding on a box whose axes differ (the [i%2] axis indexing). Grids built inside the fixes' window
    (after the interlace-phase fix, before these) are flagged at their level, the same commands built after are not."""
    import results as R
    print("stale outputs: the interlace, PCS, CIC and axis-indexing C++ fixes:")
    work = Path(tempfile.mkdtemp(prefix="gui_cppfix_"))
    try:
        log = work / "r.runlog"

        def reasons(built, args):
            _write_log(log, f"Build: DTFE 3e0c1a65fd-dirty (built {built})\nRUNNING: /x/DTFE in.hdf5 r --grid 64 {args}\n")
            return R.stale_reasons(R.parse_runlog(log))

        before, after = "2026-10-05T08:00:00Z", "2026-10-12T08:00:00Z"
        has = lambda got, lvl, word: any(lv == lvl and word in m for lv, m in got)   # noqa: E731
        cases = (   # (arguments, the (level, word) a build inside the window must show, words no message may hold)
            ("--periodic --TSC --interlace", [("error", "second pass")], ["PCS", "CIC"]),
            ("--periodic --TSC --interlace --regionMpc 0 50 0 50 0 50", [], ["second pass"]),
            ("--periodic --TSC --interlace --region 0 0.5 0 0.5 0 0.5", [], ["second pass"]),
            ("--periodic --PCS", [("error", "PCS")], ["CIC", "second pass"]),
            ("--periodic --CIC", [("warning", "CIC")], ["PCS"]),
            ("--box 0 100 0 100 0 200 --region 0 1 0 1 0.5 1", [("error", "fractional region")], ["padding"]),
            ("--box 10 110 10 110 20 120 --region 0 1 0 1 0.5 1", [("error", "fractional region")], []),
            ("--box 0 100 0 100 0 100 --region 0 1 0 1 0.5 1 --padding 0.05", [], ["fractional"]),
            ("--box 0 100 0 100 0 200 --padding 0.05", [("warning", "fractional padding")], ["fractional region"]),
            ("--box 10 110 10 110 20 120 --padding 0.05", [], ["fractional"]),
            ("--box 0 100 0 100 0 200 --regionMpc 0 100 0 100 100 200 --paddingMpc 5 5 5 5 10 10", [], ["fractional"]),
        )
        bad = []
        for args, want, never in cases:
            old, new = reasons(before, args), reasons(after, args)
            gone = [w for _, w in want] + never          # after the fix: none of the messages
            if not all(has(old, lvl, w) for lvl, w in want) or any(w in m for _, m in old for w in never) \
                    or any(w in m for _, m in new for w in gone):
                bad.append(f"{args}: old {old} | new {new}")
        check("KNOWN_FIXES: grids built before 2026-10-11T23:00 are flagged for interlace without a region (error), "
              "PCS (error), CIC (warning), a fractional region (error) or padding (warning) on a box whose axes "
              "differ; the same commands built after it, and the unaffected ones, are not", not bad, " || ".join(bad))
        check("the new KNOWN_FIXES messages are launcher text (no '--')",
              not [f.key for f in R.KNOWN_FIXES if "--" in f.message], "")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def pts_conventions_flag():
    """results.KNOWN_FIXES 'pts-grid-conventions': a PS image plane built before the
    points followed the grids' --ps-vertex-mass / --ps-volume-weighted is flagged (warning); the same command built
    after it is not, nor a plane without the two grid options, one asked for '--pts-conventions legacy', a PS run
    without a plane, or a standard plane."""
    import results as R
    print("stale outputs: PS image planes in the grids' conventions:")
    work = Path(tempfile.mkdtemp(prefix="gui_ptsconv_"))
    try:
        log = work / "r.runlog"

        def flagged(exe, built, args):
            _write_log(log, f"Build: {exe} 3e0c1a65fd-dirty (built {built})\nRUNNING: /x/{exe} in.hdf5 r --grid 64 {args}\n")
            return any(lv == "warning" and "image plane made before" in m for lv, m in R.stale_reasons(R.parse_runlog(log)))

        before, after = "2026-10-13T12:56:00Z", "2026-10-14T08:00:00Z"
        plane = "--periodic --sample-points p.bin --pts-vel-grad"
        cases = (   # (binary, arguments, flagged when built before, after)
            ("PS-DTFE", f"{plane} --ps-vertex-mass --ps-volume-weighted", True, False),
            ("PS-DTFE", f"{plane} --ps-volume-weighted", True, False),
            ("PS-DTFE", f"{plane} --ps-vertex-mass --ps-volume-weighted --pts-conventions legacy", False, False),
            ("PS-DTFE", plane, False, False),
            ("PS-DTFE", "--periodic --ps-vertex-mass --ps-volume-weighted", False, False),
            ("DTFE", plane, False, False),
        )
        bad = [f"{exe} {args}" for exe, args, b, a in cases
               if flagged(exe, before, args) != b or flagged(exe, after, args) != a]
        check("KNOWN_FIXES pts-grid-conventions: a PS plane with the grids' vertex masses or volume weighting built "
              "before 2026-10-13T20:00 is flagged, the same built after it, a legacy or option-less plane, a PS run "
              "without a plane and a standard plane are not", not bad, " || ".join(bad))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def pts_conventions_choice():
    """The plane conventions where the launcher makes a phase-space image plane: the grids' in both qualities (a
    setting the two modes removed; the binary's 'legacy' stays for a developer). A Grids job with a slice and a
    Pipeline job set PTS_CONVENTIONS=grid for run_ps_dtfe.sh / run_ps_pipeline.sh, spelled out both ways; a Custom run
    passes nothing (grid is the binary's default); standard DTFE and a run without points get nothing; a saved job's
    choice is dropped; 'Load settings' reads a run's conventions from its log (the Runs tab names a legacy plane) but
    sets none."""
    import results as R
    print("the plane conventions:")
    work = Path(tempfile.mkdtemp(prefix="gui_ptschoice_"))
    try:
        plane = work / "hires_plane_z64.bin"
        plane.write_bytes(b"\0" * 24 * 64)
        snap = work / "snap.hdf5"
        snap.write_bytes(b"")
        g = lambda **kw: rs.RunSpec(sim="TNG50-4", snapshots=[99], **kw).command()[1]        # noqa: E731
        c = lambda **kw: rs.CustomSpec(input_file=str(snap), **kw).command()                # noqa: E731
        check("Grids: PTS_CONVENTIONS=grid with a PS slice in both qualities (PS_POINT_CONVENTIONS the same); none "
              "without a slice or for standard DTFE; Pipeline: always grid (its plane), its memory check too",
              all(g(slice_plane=str(plane), quality=q).get("PTS_CONVENTIONS") == "grid"
                  and g(slice_plane=str(plane), quality=q).get("PS_POINT_CONVENTIONS") == "grid"
                  and rs.PipelineSpec(sims=["TNG50-4"], quality=q).command()[1].get("PTS_CONVENTIONS") == "grid"
                  for q in rs.QUALITIES)
              and "PTS_CONVENTIONS" not in g() and "PTS_CONVENTIONS" not in g(estimator="dtfe", slice_plane=str(plane))
              and rs.PipelineSpec(sims=["TNG50-4"]).check_step("TNG50-4", [99], 1024).env.get(
                  "PTS_CONVENTIONS") == "grid")
        check("Custom: no '--ps-point-conventions' (the grids' conventions are the binary's default), with or without "
              "points, either quality or estimator",
              not any("--ps-point-conventions" in c(sample_points=str(plane), quality=q, estimator=e)
                      for q in rs.QUALITIES for e in ("ps", "dtfe")))
        msgs = lambda spec: [(lv, m) for lv, m in spec.problems() if "conventions" in m]     # noqa: E731
        said = (msgs(rs.RunSpec(sim="TNG50-4", snapshots=[99], slice_plane=str(plane)))
                + msgs(rs.PipelineSpec(sims=["TNG50-4"]))
                + msgs(rs.CustomSpec(input_file=str(snap), sample_points=str(plane))))
        old = [cls.from_dict(dict(cls().to_dict(), pts_conventions="legacy"))
               for cls in (rs.RunSpec, rs.PipelineSpec, rs.CustomSpec)]
        check("nothing said about conventions; a job saved with 'legacy' loads at the grids' conventions (the choice "
              "dropped)",
              not said and not any(hasattr(j, "pts_conventions") for j in old)
              and all(j.mode().pts_conventions == "grid" for j in old)
              and rs.RunSpec.from_dict({"pts_conventions": "legacy", "slice_plane": str(plane),
                                        "sim": "TNG50-4"}).command()[1].get("PTS_CONVENTIONS") == "grid", str(said))
        log = work / "ps_output.runlog"
        recs = {}
        for name, extra in (("legacy", " --pts-conventions legacy"), ("grid", ""), ("noplane", None)):
            args = "--grid 64 --ps-vertex-mass --ps-volume-weighted" + (
                "" if extra is None else f" --sample-points {plane} --pts-vel-grad{extra}")
            _write_log(log, f"RUNNING: /x/PS-DTFE in.hdf5 {work}/ps_output {args}\n")
            recs[name] = R.parse_runlog(log)
        st = {k: r.settings().get("pts_conventions") for k, r in recs.items()}
        check("a run's plane conventions are read from its run log (legacy only when it says so; none without a "
              "plane), the Runs tab names a legacy plane, and 'Load settings' sets none (a removed setting)",
              st == {"legacy": "legacy", "grid": "grid", "noplane": None}
              and "pts_conventions" in rs.REMOVED_SETTINGS
              and "pts_conventions" not in rs.RunSpec.__dataclass_fields__
              and "legacy conventions" in recs["legacy"].options_text() and "legacy" not in recs["grid"].options_text()
              and "--" not in recs["legacy"].options_text(), f"{st} {recs['legacy'].options_text()}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
