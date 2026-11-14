#!/usr/bin/env python3
"""Launcher features (python/gui): run-directory files of a run going (contract C1) in the Runs
browser, the memory status strip, the 'Running now' panel, the void-catalogue status and its build job, keep-awake,
the void Plots sets, the memory check's log; the Explore menu's Save Options…; the plane
conventions' menu on the Grids, Custom and Pipeline tabs.

Three tiers:
  * HEADLESS (always run; Ubuntu CI, numpy 1.26): results.py / runspec.py / void_cache.py on synthetic run folders
    and synthetic processes (a 'sleep' renamed to look like PS-DTFE through bash's exec -a, or run through a
    symlink named PS-DTFE where ps's comm must say so).
  * QT (when PySide6 imports, else SKIP): the RunningNow table, the Runs browser's Voids column, the window's
    status strip, 'Build void catalogue' -> a queue job, and a failed memory-check pass landing in the log.
  * DATA-BACKED (DTFE_DATA_ROOT, else SKIP): the void-catalogue status of the real TNG100-3-Dark grids.

Usage:  python3 tests/py_gui_app2_test.py            (exit 0 = pass)
        ~/.venvs/dtfe-gui/bin/python tests/py_gui_app2_test.py    (also the Qt tier)
"""

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _harness import FAIL, PASS, REPO as ROOT, SKIP, setup, summary

setup("gui_app2_figs_", paths=("python/gui",), mpl_agg=False)    # before runspec is imported
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["DTFE_GUI_NO_NOTIFY"] = "1"

import results as R  # noqa: E402
import runspec as rs  # noqa: E402


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond or not detail else f"  [{str(detail)[:400]}]"))


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP  {name}  ({why})")


def dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def snapdir(tmp: Path, sim: str = "TNGX-1-Dark", snap: int = 99) -> Path:
    d = tmp / "TNGX" / sim / f"snapdir_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


LOG_OK = ("Build: PS-DTFE abc123 (built 2026-10-10T12:00:00Z)\nRUNNING: ./PS-DTFE in.hdf5 {root} --grid 8 --periodic\n"
          "SUMMARY\n  total wall time : 1m 2s\n  peak memory (RSS) : 1.5 GB\n")


# ---------------------------------------------------------------------- C1 in the Runs browser
def t_run_files(tmp: Path):
    print("C1 run-directory files (results.py):")
    d = snapdir(tmp)
    root = d / "ps_output"
    (d / "ps_output.runlog").write_text(LOG_OK.format(root=root))
    old = time.time() - 100
    os.utime(d / "ps_output.runlog", (old, old))
    (d / "ps_output.a_den").write_bytes(b"\0" * 64)
    os.utime(d / "ps_output.a_den", (old - 10, old - 10))
    (d / "ps_output.a_vel.part.4242").write_bytes(b"\0" * 32)
    (d / "ps_output.runlog.partial").write_text(LOG_OK.splitlines()[0] + "\n[partitions 3/10 | 30%] elapsed 1m\n")
    lock = d / "ps_output.runlog.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()))
    # a live process ps names PS-DTFE: 'sleep' run through a symlink of that name (comm = the symlink's path on
    # macOS, its basename on Linux); killed as soon as the scan is done (the concurrent launcher tests poll pgrep)
    fake_bin = tmp / "bin" / "PS-DTFE"
    fake_bin.parent.mkdir(exist_ok=True)
    fake_bin.symlink_to(shutil.which("sleep") or "/bin/sleep")
    fake = subprocess.Popen([str(fake_bin), "30"])
    prog = {"pid": fake.pid, "phase": "deposit", "done": 3, "total": 10, "elapsed_s": 60, "eta_s": 7380,
            "rss_gb": 12.5, "gpu_retries": 1, "updated": time.time()}
    (d / "ps_output.runlog.progress").write_text(json.dumps(prog))
    check("is_run_file: the sidecar (and its .tmp), the lock and '<grid>.part.<pid>' are run files; grids and logs "
          "are not", all(R.is_run_file(n) for n in ("ps_output.runlog.progress", "ps_output.runlog.progress.tmp",
                                                     "ps_output.runlog.lock", "ps_output.a_den.part.123"))
          and not any(R.is_run_file(n) for n in ("ps_output.a_den", "ps_output.runlog", "ps_output.runlog.partial",
                                                  "ps_output.partition")))
    try:
        time.sleep(0.2)
        recs = R.scan_runs(tmp)
    finally:
        fake.kill()
        fake.wait()
    check("one record for the output set: the lock and the sidecar are not run logs", len(recs) == 1,
          [(r.path.name, r.attempt) for r in recs])
    rec = recs[0]
    names = sorted(p.name for p in rec.outputs)
    check("outputs: the grid only, never the '.part.<pid>', the sidecar or the lock", names == ["ps_output.a_den"], names)
    check("a '.partial' whose sidecar pid is alive is 'running' with the sidecar read", rec.attempt == "running"
          and rec.progress is not None and rec.progress.get("state") == "running", (rec.attempt, rec.progress))
    note = R.attempt_note(rec)
    text = R.progress_text(rec.progress)
    check("progress_text: phase, done/total (percent), ETA, memory, GPU retries; the attempt note carries it",
          text == "deposit 3/10 (30%), ETA 2h 03m, 12.5 GB, 1 GPU retries" and note is not None and text in note[1],
          (text, note))
    # the sidecar's pid alive but another program's (pids wrap within a day of the launcher's polling): this
    # process is python, not a DTFE binary, so the run was killed (a bare pid test would say 'running')
    prog["pid"] = os.getpid()
    (d / "ps_output.runlog.progress").write_text(json.dumps(prog))
    rec = R.scan_runs(tmp)[0]
    check("a sidecar pid now held by a program that is not a DTFE binary reads 'died', the attempt 'killed'",
          rec.attempt == "killed" and (rec.progress or {}).get("state") == "died", (rec.attempt, rec.progress))
    # the same sidecar with a dead pid: the run was killed outright
    prog["pid"] = dead_pid()
    (d / "ps_output.runlog.progress").write_text(json.dumps(prog))
    rec = R.scan_runs(tmp)[0]
    check("a dead pid in the sidecar: the attempt reads 'killed' and the progress 'died'",
          rec.attempt == "killed" and "died" in R.progress_text(rec.progress), (rec.attempt, rec.progress))
    check("read_progress: None for a missing or broken sidecar", R.read_progress(d / "nope.progress") is None
          and (d.joinpath("bad.runlog.progress").write_text("{not json") and R.read_progress(d / "bad.runlog.progress")
               is None))
    (d / "bad.runlog.progress").unlink()
    shutil.rmtree(lock)
    for f in ("ps_output.runlog.partial", "ps_output.runlog.progress", "ps_output.a_vel.part.4242"):
        (d / f).unlink()


def t_parse_cache(tmp: Path):
    print("run-log cache (results.parse_runlog):")
    d = snapdir(tmp, snap=50)
    log = d / "ps_output.runlog"
    log.write_text(LOG_OK.format(root=d / "ps_output"))
    a = R.parse_runlog(log, "TNGX-1-Dark", 50)
    key = str(log)
    cached = key in R._LOG_CACHE
    real_read = Path.read_text
    calls = []

    def counting(self, *aa, **kk):
        if str(self) == key:
            calls.append(1)
        return real_read(self, *aa, **kk)
    Path.read_text = counting
    try:
        b = R.parse_runlog(log, "TNGX-1-Dark", 50)
        n_unchanged = len(calls)
        log.write_text(LOG_OK.format(root=d / "ps_output").replace("1m 2s", "2m 4s") + "more\n")
        c = R.parse_runlog(log, "TNGX-1-Dark", 50)
    finally:
        Path.read_text = real_read
    check("an unchanged log is not read again (cache by size and mtime), with the same fields",
          cached and n_unchanged == 0 and (a.wall_seconds, a.binary, a.ok, a.grid) == (b.wall_seconds, b.binary, b.ok, b.grid),
          (cached, n_unchanged, a.wall_seconds, b.wall_seconds))
    check("... a rewritten log is read again", len(calls) == 1 and c.wall_seconds == 124.0, (len(calls), c.wall_seconds))
    log.unlink()


# ---------------------------------------------------------------------- memory strip, running now
def t_memory():
    print("memory status (runspec.memory_status / memory_text):")
    m = rs.memory_status()
    check("memory_status: the three keys; a number for free memory on macOS and Linux",
          set(m) == {"free_gb", "swap_used_gb", "swap_total_gb"}
          and (m["free_gb"] is None if sys.platform not in ("darwin", "linux") else m["free_gb"] > 0), m)
    old = os.environ.get("DTFE_FREE_GB")
    os.environ["DTFE_FREE_GB"] = "12.5"
    try:
        m2 = rs.memory_status()
    finally:
        if old is None:
            os.environ.pop("DTFE_FREE_GB")
        else:
            os.environ["DTFE_FREE_GB"] = old
    check("... DTFE_FREE_GB overrides the free reading (as for the C++ programs)", m2["free_gb"] == 12.5, m2)
    txt = rs.memory_text({"free_gb": 31.24, "swap_used_gb": 1.42, "swap_total_gb": 4.0},
                         {"runs": [(812, "PS-DTFE", 37.1)], "servers": [(90, "DTFE", None)]})
    check("memory_text: free, swap, each run and server with its memory; no '--'",
          txt == "free 31.2 GB · swap 1.4 of 4.0 GB · PS-DTFE pid 812, 37.1 GB · DTFE server pid 90" and "--" not in txt,
          txt)


def t_running_now(tmp: Path):
    print("running now (runspec.parse_run_args / running_now / progress_files):")
    a = rs.parse_run_args("./PS-DTFE /Users/x/My Data/Illustris TNG/TNG50/TNG50-3-Dark/snapdir_099/combined_099.hdf5 "
                          "/Users/x/My Data/Illustris TNG/TNG50/TNG50-3-Dark/snapdir_099/ps_output --grid 512 --periodic")
    check("parse_run_args: paths with spaces, the simulation and snapshot from the snapdir",
          a == {"exe": "PS-DTFE", "input": "/Users/x/My Data/Illustris TNG/TNG50/TNG50-3-Dark/snapdir_099/combined_099.hdf5",
                "output": "/Users/x/My Data/Illustris TNG/TNG50/TNG50-3-Dark/snapdir_099/ps_output",
                "sim": "TNG50-3-Dark", "snap": 99}, a)
    check("... a custom run (no snapdir) and a line that is not a run", rs.parse_run_args(
        "/x/DTFE-double /data/my run.hdf5 /out/my run --grid 64")["output"] == "/out/my run"
        and rs.parse_run_args("/x/DTFE-double /data/my run.hdf5 /out/my run --grid 64")["sim"] == ""
        and rs.parse_run_args("sleep 30")["output"] == "")
    d = snapdir(tmp, snap=33)
    out = d / "ps_output"
    live = {"pid": 0, "phase": "triangulation", "done": 2, "total": 8, "eta_s": 600, "rss_gb": 3.0, "elapsed_s": 50,
            "gpu_retries": 0, "updated": time.time()}
    # a 'sleep' that looks like PS-DTFE to ps (bash's exec -a sets its argv[0] to the whole command line)
    fake = f"{ROOT}/PS-DTFE {d}/combined_033.hdf5 {out} --grid 8 --periodic"
    proc = subprocess.Popen(["bash", "-c", 'exec -a "$0" sleep 30', fake])
    try:
        time.sleep(0.3)
        live["pid"] = proc.pid
        Path(str(out) + ".runlog.progress").write_text(json.dumps(live))
        left = snapdir(tmp, snap=17) / "ps_output.runlog.progress"
        left.write_text(json.dumps(dict(live, pid=dead_pid(), done=5)))
        files = rs.progress_files(tmp)
        check("progress_files: every sidecar under the data root's snapdirs",
              sorted(p.name for p in files) == ["ps_output.runlog.progress"] * 2 and len(files) == 2, files)
        got = rs.running_now(files, {"runs": [(proc.pid, "PS-DTFE", 2.5)], "servers": []})
        by = {e["state"]: e for e in got}
        e = by.get("running", {})
        check("running_now: the live run with its simulation, snapshot, output, ps elapsed time and sidecar",
              len(got) == 2 and e.get("sim") == "TNGX-1-Dark" and e.get("snap") == 33 and e.get("output") == str(out)
              and (e.get("progress") or {}).get("done") == 2 and e.get("gb") == 2.5 and e.get("elapsed_s") is not None,
              got)
        dd = by.get("died", {})
        check("... and the leftover sidecar of a dead pid reads 'died' with its snapshot",
              dd.get("snap") == 17 and dd.get("sim") == "TNGX-1-Dark" and (dd.get("progress") or {}).get("done") == 5, dd)
        # the sidecar's pid must be a live DTFE binary to ps (results.dtfe_pid_alive): the exec -a
        # 'sleep' above is 'sleep' to Linux's comm, so a symlink named PS-DTFE runs this one
        fake_bin = tmp / "bin" / "PS-DTFE"
        fake_bin.parent.mkdir(exist_ok=True)
        if not fake_bin.is_symlink():
            fake_bin.symlink_to(shutil.which("sleep") or "/bin/sleep")
        other = subprocess.Popen([str(fake_bin), "30"])
        time.sleep(0.2)
        Path(str(out) + ".runlog.progress").write_text(json.dumps(dict(live, pid=other.pid)))
        alone = rs.running_now(files, {"runs": [], "servers": []})
        other.kill()
        other.wait()
        check("... a live sidecar whose process is not among the runs is left to its own process row",
              [(e["state"], e["snap"]) for e in alone] == [("died", 17)], alone)
    finally:
        proc.kill()
        proc.wait()
        Path(str(out) + ".runlog.progress").unlink(missing_ok=True)
        left.unlink(missing_ok=True)


def t_keep_awake():
    print("keep-awake (runspec.keep_awake_argv):")
    script = rs.keep_awake_argv(["bash", str(rs.PS_SCRIPT), "-s", "TNG50-3-Dark"], 1234)
    binary = rs.keep_awake_argv([str(ROOT / "PS-DTFE"), "in.hdf5", "out"], 1234)
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        check("a run script keeps itself awake; a binary run directly (the Custom tab) gets caffeinate -i -m -w pid",
              script is None and binary is not None and binary[1:] == ["-i", "-m", "-w", "1234"], (script, binary))
    else:
        check("no caffeinate here: no keep-awake command", script is None and binary is None, (script, binary))


# ---------------------------------------------------------------------- Plots sets, void catalogue
def t_plot_sets(tmp: Path):
    print("Plots sets (runspec.FIGURE_SETS):")
    base = dict(data_root=str(tmp), sim="TNGX-1-Dark", snapshots=[33, 99])
    keys = [fs.key for fs in rs.FIGURE_SETS]
    check("the new sets are registered and the core set stays last",
          all(k in keys for k in ("crossbox", "voidevol", "voidenv", "voidfate", "voidconv", "voidcat")) and keys[-1] == "core", keys)
    plain = {st.argv[1].split("/")[-1]: st for st in rs.PlotSpec(
        sets=["voidtrack", "crossbox", "voidevol", "voidenv", "voidfate", "voidconv", "voidcat"], **base).steps()}
    cb = plain["plot_void_crossbox.py"].argv
    check("defaults: the cross-box comparison gets only its boxes and the data root; "
          "the C5 scripts and the convergence check (K4) '--sim SIM'; one catalogue build per snapshot",
          cb[2:] == ["--sims", "TNG50-3-Dark", "TNG100-3-Dark", "TNG300-3-Dark", "--data-root", str(tmp)]
          and all(plain[n].argv[2:] == ["--sim", "TNGX-1-Dark"] for n in
                  ("plot_void_profile_evolution.py", "plot_void_environment.py", "plot_void_fate.py",
                   "plot_void_convergence.py"))
          and plain["plot_void_convergence.py"].env.get("DTFE_SIM") == "TNGX-1-Dark"
          and [st.argv[2:] for st in rs.PlotSpec(sets=["voidcat"], **base).steps()]
          == [["build", "--sim", "TNGX-1-Dark", "--snap", str(n), "--estimator", "ps"] for n in (33, 99)],
          {k: v.argv[2:] for k, v in plain.items()})
    opts = {"crossbox": {"sims": "TNG50-3-Dark TNG100-3-Dark", "smooth_mpc": 3.0, "z": "0 2", "n_boot": 50, "replot": True},
            "voidtrack": {"radius": "r_eff", "smooth_mpc": 2.1625}}
    got = {st.argv[1].split("/")[-1]: st.argv for st in rs.PlotSpec(sets=["crossbox", "voidtrack"], options=opts,
                                                                       method="dtfe", prefix="ps_mw", **base).steps()}
    c, t = got["plot_void_crossbox.py"], got["plot_void_tracking.py"]
    check("... the options reach their flags (boxes, estimator, prefix, smoothing, redshifts, bootstrap, replot; the "
          "tracking's radius and smoothing in Mpc)",
          c[c.index("--sims") + 1:c.index("--data-root")] == ["TNG50-3-Dark", "TNG100-3-Dark"]
          and c[c.index("--estimator") + 1] == "dtfe" and c[c.index("--prefix") + 1] == "ps_mw"
          and c[c.index("--smooth-mpc") + 1] == "3"
          and c[c.index("--redshifts") + 1:c.index("--redshifts") + 3] == ["0", "2"]
          and c[c.index("--n-boot") + 1] == "50" and "--replot" in c
          and t[t.index("--radius") + 1] == "r_eff" and t[t.index("--smooth-mpc") + 1] == "2.1625", (c, t))
    # every flag the launcher passes is one the script's own parser knows (a script not on disk is judged only
    # once it exists)
    flat = {n: a for n, a in list(got.items()) + [(k, v.argv) for k, v in plain.items()]}
    for name, argv in sorted(flat.items()):
        script = ROOT / "python" / ("gui" if name == "void_cache.py" else "plot") / name
        if not script.is_file():
            skip(f"{name}: its flags in its help", "not in this checkout yet")
            continue
        h = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, MPLBACKEND="Agg"))
        flags = [x for x in argv[2:] if x.startswith("--")]
        check(f"{name}: every flag in its own help ({' '.join(flags)})", h.returncode == 0
              and all(f in h.stdout for f in flags), h.stderr[-300:] or [f for f in flags if f not in h.stdout])
    texts = [fs.title + " " + fs.about + " " + " ".join(o.label + " " + o.help for o in fs.opts)
             for fs in rs.FIGURE_SETS if fs.key in ("crossbox", "voidevol", "voidenv", "voidfate", "voidconv", "voidcat", "voidtrack")]
    check("no '--' in the new sets' texts", not [x for x in texts if "--" in x])
    sp = rs.PlotSpec(sets=["voidcat", "voidfate"], **base)
    (tmp / "TNGX" / "TNGX-1-Dark").mkdir(parents=True, exist_ok=True)
    probs = sp.problems()
    absent = not (ROOT / "python" / "plot" / "plot_void_fate.py").is_file()
    check("problems: no grids for the catalogue build is a warning; a script not in the checkout an error",
          any(lv == "warning" and "no grids" in m for lv, m in probs)
          and (any(lv == "error" and "plot_void_fate.py" in m for lv, m in probs) == absent), probs)


def t_void_status_headless(tmp: Path):
    print("void catalogue status (results.void_status / void_cache.py):")
    import void_cache as VC
    check("void_cache.products_prefix: the method's own prefix is the untagged namespace",
          VC.products_prefix("ps", "ps_output") is None and VC.products_prefix("dtfe", "output") is None
          and VC.products_prefix("ps", "ps_mw") == "ps_mw" and VC.products_prefix("ps", "") is None)
    d = snapdir(tmp, snap=4)
    (d / "ps_output.runlog").write_text(LOG_OK.format(root=d / "ps_output"))
    (d / "ps_output.a_den").write_bytes(b"\0" * 64)
    rec = next(r for r in R.scan_runs(tmp) if r.snap == 4)
    check("void_candidate: a finished TNG 3D run with a density grid (not a custom run, not a failed one)",
          R.void_candidate(rec) and not R.void_candidate(R.RunRecord(path=d / "x.runlog", ok=True))
          and not R.void_candidate(R.RunRecord(path=rec.path, sim=rec.sim, snap=4, prefix="ps_output",
                                               binary="PS-DTFE", ok=False, outputs=rec.outputs)))
    st = R.void_status([rec], tmp, python=sys.executable)
    check("a grid that is no real FieldSet: one 'error' row with a reason, nothing built", st.get(str(rec.path), ("",))[0]
          == "error" and st[str(rec.path)][1], st)
    st = R.void_status([rec], tmp, python=str(tmp / "no-such-python"))
    check("... and a helper that cannot start reads 'error' too (never an exception)",
          st.get(str(rec.path), ("",))[0] == "error", st)
    for f in ("ps_output.runlog", "ps_output.a_den"):
        (d / f).unlink()


def t_void_status_data():
    print("data-backed void status (TNG100-3-Dark under DTFE_DATA_ROOT):")
    if not os.environ.get("DTFE_DATA_ROOT"):
        skip("void status of TNG100-3-Dark 099", "no data root: DTFE_DATA_ROOT is unset")
        return
    root = Path(os.environ["DTFE_DATA_ROOT"]).expanduser()
    sd = rs.sim_dir("TNG100-3-Dark", str(root)) / "snapdir_099"
    if not (sd / "ps_output.runlog").is_file() or not (sd / "ps_output.a_den").is_file():
        skip("void status of TNG100-3-Dark 099", f"no grids under {sd}")
        return
    rec = R.parse_runlog(sd / "ps_output.runlog", "TNG100-3-Dark", 99)
    t0 = time.time()
    st = R.void_status([rec], root, python=sys.executable)
    state, why = st.get(str(rec.path), ("", ""))
    check(f"the production catalogue reads cached, stale or absent, never error ({state}: {why}; "
          f"{time.time() - t0:.1f} s)", state in ("cached", "stale", "absent"), st)


# ---------------------------------------------------------------------- Qt
def t_qt(tmp: Path):
    print("Qt (runs_view.RunningNow / RunsBrowser, app.MainWindow):")
    try:
        from PySide6.QtCore import QEventLoop, QSettings, QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        skip("the Qt tier", f"PySide6 not importable: {e}")
        return
    app = QApplication.instance() or QApplication([])  # noqa: F841  (kept referenced while the tier runs)
    import runs_view as RV

    def spin(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()
    rn = RV.RunningNow()
    rn.show_entries([{"pid": 812, "name": "PS-DTFE", "gb": 37.1, "elapsed_s": 3600, "sim": "TNG50-3-Dark", "snap": 99,
                      "output": "/x/ps_output", "state": "running",
                      "progress": {"phase": "deposit", "done": 9, "total": 27, "eta_s": 7200, "gpu_retries": 2}},
                     {"pid": 77, "name": "", "gb": None, "elapsed_s": None, "sim": "TNG50-3-Dark", "snap": 33,
                      "output": "/x/ps_output", "state": "died", "progress": {"done": 5, "total": 27, "rss_gb": 9.0}}])
    row = lambda k: [rn.table.item(k, c).text() for c in range(rn.table.columnCount())]   # noqa: E731
    check("RunningNow: simulation, snapshot, program, pid, phase, done/total, ETA, memory, elapsed, state",
          row(0) == ["TNG50-3-Dark", "099", "PS-DTFE", "812", "deposit", "9/27 (33%)", "2h 00m", "37.1 GB", "1h 00m",
                     "running, 2 GPU retries"] and row(1)[1] == "033" and row(1)[-1].startswith("died")
          and row(1)[7] == "9.0 GB" and "1 died" in rn.note.text(), (row(0), row(1), rn.note.text()))
    rn.table.selectRow(1)
    stop_dead = rn.b_stop.isEnabled()
    rn.table.selectRow(0)
    check("... Stop only for a run that is going", not stop_dead and rn.b_stop.isEnabled())
    check("... no '--' in what it shows", not any("--" in x for k in (0, 1) for x in row(k)))

    # the Runs browser: the Voids column filled from a background status (a fake helper that answers 'stale')
    d = snapdir(tmp, sim="TNGQ-1-Dark", snap=99)
    (d / "ps_output.runlog").write_text(LOG_OK.format(root=d / "ps_output"))
    (d / "ps_output.a_den").write_bytes(b"\0" * 64)
    fake = tmp / "fake_python.sh"
    fake.write_text('#!/bin/sh\ncat >/dev/null\necho \'[{"state": "stale", "why": "built from another version"}]\'\n')
    fake.chmod(0o755)
    rb = RV.RunsBrowser(lambda: str(tmp), lambda: [], str(fake))
    rb.reload()
    for _ in range(50):
        if rb.void_states:
            break
        spin(100)
    col = RV.COLUMNS.index("Voids")
    k = next((i for i, r in enumerate(rb._rows) if r.sim == "TNGQ-1-Dark"), None)
    check("RunsBrowser: the Voids column reads the background status ('stale'), the build button follows the selection",
          k is not None and rb.table.item(k, col).text() == "stale", (k, rb.void_states))
    got = []
    rb.build_voids.connect(got.append)
    rb.table.selectRow(k)
    rb.b_voids.click()
    check("... 'Build void catalogue' emits the selected run", len(got) == 1 and got[0].sim == "TNGQ-1-Dark")
    rb.filter.setText("voids stale")
    check("... and the filter knows the state", len(rb._rows) >= 1 and all(
        rb.void_states.get(str(r.path), ("",))[0] == "stale" for r in rb._rows))
    # a killed run with its checkpoint directory shows 'resumable' in the Status column
    dk = snapdir(tmp, sim="TNGK-1-Dark", snap=99)
    (dk / "ps_output.runlog.partial").write_text(f"RUNNING: ./PS-DTFE in.hdf5 {dk / 'ps_output'} --grid 8 "
                                                 "--partition 2 2 2\n")
    (dk / "ps_output.runlog.checkpoint").mkdir()
    rb.filter.setText("")
    rb.reload()
    kk = next((i for i, r in enumerate(rb._rows) if r.sim == "TNGK-1-Dark"), None)
    status = rb.table.item(kk, RV.COLUMNS.index("Status")).text() if kk is not None else None
    if shutil.which("ps"):
        check("RunsBrowser: a killed run with its checkpoint shows 'resumable', its tooltip says the job resumes it",
              status is not None and "resumable" in status
              and "resum" in rb.table.item(kk, RV.COLUMNS.index("Status")).toolTip(), status)
    else:
        skip("RunsBrowser: 'resumable'", "no ps here: a '.partial' run cannot be called killed")

    import app as A
    ini = tmp / "launcher.ini"
    w = A.MainWindow(remember=False, settings=QSettings(str(ini), QSettings.IniFormat))
    try:
        # the data root: DTFE_DATA_ROOT when the launcher starts with one, else none (its setup asks for the folder)
        # and every job refused for it; the folder chosen reaches each job's environment as DTFE_DATA_ROOT
        env_root = os.environ.get("DTFE_DATA_ROOT", "")
        w._read_widgets()
        probs = w.spec.problems()
        check("a new window starts at DTFE_DATA_ROOT, else with no data root and a job refused for it",
              w.root_edit.text() == env_root
              and (bool(env_root) or any(lv == "error" and "no data root" in m for lv, m in probs)),
              (w.root_edit.text(), probs[:2]))
        w.root_edit.setText(str(tmp))
        w._read_widgets()
        check("... the folder chosen is every job's data root and reaches its environment as DTFE_DATA_ROOT",
              w.spec.data_root == w.pipe.data_root == w.plots.data_root == w.data.data_root == str(tmp)
              and w.spec.command()[1]["DTFE_DATA_ROOT"] == str(tmp)
              and w.pipe.command()[1]["DTFE_DATA_ROOT"] == str(tmp), w.spec.data_root)
        for _ in range(60):
            if w.mem_strip.text():
                break
            spin(100)
        check("the window's status strip shows free memory (off the GUI thread)",
              w.mem_strip.text().startswith("free ") or w.mem_strip.text() == "memory: unknown", w.mem_strip.text())
        check("a 'Running now' tab beside Runs", w.results.indexOf(w.running_now) == w.results.indexOf(w.runs) + 1)
        t_qt_save_options(w)
        rec = R.parse_runlog(d / "ps_output.runlog", "TNGQ-1-Dark", 99)
        n0 = len(w.queue)
        w.queue_void_build(rec)
        job = w.queue[-1] if len(w.queue) > n0 else None
        steps = job.build().steps() if job else []
        check("'Build void catalogue' queues a Plots job: void_cache.py build for the run's snapshot and method",
              job is not None and job.kind == "plots" and len(steps) == 1
              and steps[0].argv[1].endswith("void_cache.py") and steps[0].argv[2:] ==
              ["build", "--sim", "TNGQ-1-Dark", "--snap", "99", "--estimator", "ps"]
              and steps[0].env.get("DTFE_SIM") == "TNGQ-1-Dark", steps and steps[0].argv)
        del w.queue[n0:]
        # a failed memory-check pass: its output goes into the log the status line points to
        class _P:
            def readAllStandardOutput(self):
                return b""
        p = _P()
        w.mem_results["k"] = {}
        w._mcheck = {"tab": A.GRIDS, "key": "k", "passes": [], "done": 0, "spec": None, "first": {},
                     "proc": p, "current": ("TNGQ-1-Dark", None, [99]),
                     "buf": "Error: input file not found: combined_099.hdf5\n"}
        w._check_pass_done(p, 1)
        check("a failed memory-check pass: 'see the log', and the log has its output",
              "see the log" in w.status.text() and "input file not found" in w.log.toPlainText(),
              (w.status.text(), w.log.toPlainText()[-300:]))
        check("the queue box names every kind that waits (custom included: HEAVY_KINDS)",
              "custom" in w.q_wait.text() and all(k in rs.HEAVY_KINDS for k in ("grids", "pipeline", "custom")))
        # the partitions merge in index order on the Grids, Pipeline and Custom jobs: the quality's (no control,
        # runspec.quality_mode), given as the drivers give it
        w._read_widgets()
        own = w.custom.command()
        check("the partitions merge in index order on the Grids, Pipeline and Custom jobs, the quality's "
              "(PS_MERGE=index-order, the older ORDERED_MERGE=1 beside it until P6.6; the Custom tab: '--ps-merge "
              "index-order', given as the drivers give it)",
              all(e.get("PS_MERGE") == "index-order" and e.get("ORDERED_MERGE") == "1"
                  for e in (w.spec.command()[1], w.pipe.command()[1]))
              and any(own[i:i + 2] == ["--ps-merge", "index-order"] for i in range(len(own)))
              and own.count("--ps-merge") == 1, own)
        # a standard Grids job pins run_dtfe.sh's Monte-Carlo count, thread cap and gradient route empty (the tab holds
        # none of them for it): a SUBSAMPLES / THREADS / FACE_GRADIENT exported in the login shell cannot leak in, nor
        # the binary's own DTFE_FACE_GRADIENT
        w.dtfe_radio.setChecked(True)
        w._read_widgets()
        std_env = w.spec.command()[1]
        w.ps_radio.setChecked(True)
        w._read_widgets()
        # PS-DTFE: Fast (the sampled deposit) takes its count from -n; Exact (the default) has none: no -n, and the
        # count pinned empty. The choice found is put back afterwards
        first = w.spec.quality
        w.quality_pair.button("fast").setChecked(True)
        w._read_widgets()
        sampled_cmd = w.spec.command()
        w.quality_pair.button("exact").setChecked(True)
        w._read_widgets()
        exact_cmd = w.spec.command()
        w.quality_pair.button(first).setChecked(True)
        w._read_widgets()
        check("a standard-DTFE Grids job pins SUBSAMPLES, THREADS and FACE_GRADIENT empty (run_dtfe.sh's own "
              "defaults), DTFE_FACE_GRADIENT too (the binary's default route); the PS job takes its count from -n "
              "(Fast) or pins it empty (Exact, which has none), and pins DTFE_FACE_GRADIENT empty with either (a "
              "Fast job names no route: an exported 1 must not reach the binary)",
              std_env.get("SUBSAMPLES") == "" and std_env.get("THREADS") == "" and std_env.get("FACE_GRADIENT") == ""
              and std_env.get("DTFE_FACE_GRADIENT") == ""
              and w.spec.estimator == "ps"
              and "SUBSAMPLES" not in sampled_cmd[1] and "-n" in sampled_cmd[0]
              and exact_cmd[1].get("SUBSAMPLES") == "" and "-n" not in exact_cmd[0] and w.spec.quality == first
              and sampled_cmd[1].get("DTFE_FACE_GRADIENT") == "" == exact_cmd[1].get("DTFE_FACE_GRADIENT"),
              ({k: std_env.get(k) for k in ("SUBSAMPLES", "THREADS", "FACE_GRADIENT", "DTFE_FACE_GRADIENT")},
               w.spec.estimator, sampled_cmd, exact_cmd))
        # 'Load settings' of a run made with the completion-order merge or the linear deposit (settings the two modes
        # replaced): the job takes the quality's (index order, no linear deposit) and its cell average is the
        # Quality; nothing of the removed controls is kept behind it
        (d / "co.runlog").write_text(LOG_OK.format(root=d / "co").replace("--periodic", "--periodic --completion-order-merge"))
        w._load_run_settings(R.parse_runlog(d / "co.runlog", "TNGQ-1-Dark", 99))
        co_env = w.spec.command()[1]
        ld_log = LOG_OK.format(root=d / "ld").replace("--periodic", "--periodic --ps-linear-deposit --exact")
        (d / "ld.runlog").write_text(ld_log)
        w._load_run_settings(R.parse_runlog(d / "ld.runlog", "TNGQ-1-Dark", 99))
        ld_env, ld_q = w.spec.command()[1], (w.spec.quality, w.quality_pair.value())
        w._load_run_settings(R.parse_runlog(d / "ps_output.runlog", "TNGQ-1-Dark", 99))
        check("'Load settings' of a run made with the completion-order merge or the linear deposit: the job takes the "
              "quality's (PS_MERGE=index-order, PS_LINEAR_DEPOSIT=0), an exact run's cell average is Exact, and no "
              "setting of the removed controls is kept",
              co_env.get("PS_MERGE") == "index-order" and ld_env.get("PS_LINEAR_DEPOSIT") == "0"
              and ld_q == ("exact", "exact") and not set(w.spec.to_dict()) & set(rs.REMOVED_SETTINGS),
              (co_env.get("PS_MERGE"), ld_env.get("PS_LINEAR_DEPOSIT"), ld_q))
        # the start-over tick (Grids, Pipeline) reaches run_ps_dtfe.sh as DTFE_RESUME=0 (and PS_RESUME); the banner of a
        # refused resume ticks it; its text has no '--'; 'Load settings' leaves it alone (a one-off choice)
        so, pso = getattr(w, "so_check", None), getattr(w, "p_so", None)
        ok = so is not None and pso is not None
        if ok:
            so.setChecked(True)
            pso.setChecked(True)
            w._read_widgets()
            ticked = (w.spec.command()[1].get("DTFE_RESUME"), w.pipe.command()[1].get("DTFE_RESUME"),
                      w.spec.command()[1].get("PS_RESUME"), w.pipe.command()[1].get("PS_RESUME"))
            so.setChecked(False)
            w._read_widgets()
            off = w.spec.command()[1].get("DTFE_RESUME")
            w._job_tab = A.GRIDS
            w._banner_action("start_over")
            banner = so.isChecked()
            w._load_run_settings(R.parse_runlog(d / "ps_output.runlog", "TNGQ-1-Dark", 99))
            ok = (ticked == ("0",) * 4 and off == "1" and banner and so.isChecked()
                  and "--" not in so.text() + so.toolTip() and "DTFE_RESUME=0" in so.toolTip())
        check("the start-over tick reaches the Grids and Pipeline jobs (DTFE_RESUME=0, else 1), the refused-resume banner "
              "ticks it, no '--' in its text, 'Load settings' leaves it", ok,
              (getattr(w.spec, "start_over", None), getattr(w.pipe, "start_over", None)))
        t_qt_void_options(w)
        t_qt_gpu_retries(tmp)
        t_qt_advanced(tmp, spin)
    finally:
        w._machine_timer.stop()
        w.close()
        spin(200)


def t_qt_save_options(w):
    """The Explore menu's Save Options…: right after Save Image…, enabled without a map, it brings up
    Explore and the Save options dialog; a window that does not remember its settings keeps them for the session."""
    import app as A
    menu = next(a.menu() for a in w.menuBar().actions() if a.text() == "Explore")
    acts = [a for a in menu.actions() if not a.isSeparator()]
    k = acts.index(w.save_options_action) if w.save_options_action in acts else -1
    opened = []
    real = w.explore.open_save_options
    w.explore.open_save_options = lambda: opened.append(w.tabs.currentIndex()) or True
    try:
        w.tabs.setCurrentIndex(A.GRIDS)
        w.save_options_action.trigger()
    finally:
        w.explore.open_save_options = real
    check("Explore > Save Options…: after Save Image…, enabled with no map shown, opens the dialog on the Explore tab; "
          "remember off: the options are not written to the settings",
          k > 0 and acts[k - 1] is w.save_image_action and w.save_options_action.text() == "Save Options…"
          and w.save_options_action.isEnabled() and opened == [A.EXPLORE] and w.explore._settings is None
          and "--" not in w.save_options_action.text(), (k, opened, w.save_options_action.isEnabled()))


def t_qt_void_options(w):
    """The void scripts' own options on the Plots tab: grouped options share one form row (compact),
    and what the widgets say reaches the job's commands (= the exported script)."""
    import app as A
    keys = [fs.key for fs in rs.FIGURE_SETS]
    page = w.fig_stack.widget(keys.index("skeleton"))
    opts = rs.FIGURE_SET["skeleton"].opts
    want_rows = 1 + len({o.row for o in opts if o.row}) + sum(1 for o in opts if not o.row)
    null_w = w.opt_widgets[("skeleton", "null_draws")][1]
    sheet_w = w.opt_widgets[("skeleton", "sheet_min_density")][1]
    page.resize(700, 600)
    page.show()
    page.layout().activate()
    check("Plots tab: the skeleton's grouped options share form rows (two widgets on one line), each with a tooltip",
          page.layout().rowCount() == want_rows and null_w.geometry().y() == sheet_w.geometry().y()
          and null_w.toolTip() and sheet_w.toolTip(),
          (page.layout().rowCount(), want_rows, null_w.geometry(), sheet_w.geometry()))
    page.hide()
    w.tabs.setCurrentIndex(A.PLOTS)
    for i in range(w.fig_list.count()):
        it = w.fig_list.item(i)
        it.setCheckState(A.Qt.Checked if it.data(A.Qt.UserRole) in ("voidtrack", "skeleton") else A.Qt.Unchecked)
    w.opt_widgets[("voidtrack", "link")][1].setCurrentText("particles")
    w.opt_widgets[("voidtrack", "exclusive")][1].setChecked(True)
    w.opt_widgets[("skeleton", "min_component_mass")][1].setValue(0.001)
    w._read_widgets()
    argv = {Path(st.argv[1]).name: st.argv for st in w.plots.steps()}
    tr, sk = argv.get("plot_void_tracking.py", []), argv.get("plot_caustic_skeleton.py", [])
    check("... and the widgets' values reach the job's commands (what an export writes)",
          tr[tr.index("--link") + 1:tr.index("--link") + 2] == ["particles"] and "--exclusive" in tr
          and sk[sk.index("--min-component-mass") + 1:][:1] == ["0.001"] and "--null-draws" not in sk,
          (w.plots.sets, tr, sk))


def t_qt_gpu_retries(tmp: Path):
    """The Runs tab shows the GPU deposit retries per cause: the 'Running now' state from the
    sidecar, a finished run's status and details from its log's closing line."""
    import runs_view as RV
    rn = RV.RunningNow()
    rn.show_entries([{"pid": 9, "name": "PS-DTFE", "sim": "S", "snap": 99, "state": "running",
                      "progress": {"phase": "deposit", "gpu_retries": 3,
                                   "gpu_retries_by_cause": {"innocent": 2, "timeout": 0, "oom": 1, "other": 0}}}])
    state = rn.table.item(0, rn.table.columnCount() - 1).text()
    root = tmp / "retries_root"
    d = snapdir(root, sim="TNGR-1-Dark", snap=99)
    (d / "ps_output.runlog").write_text(LOG_OK.format(root=d / "ps_output")
                                        + "GPU deposit retries: 2 (innocent 0, timeout 2, oom 0, other 0)\n")
    (d / "ps_output.a_den").write_bytes(b"\0" * 64)
    rb = RV.RunsBrowser(lambda: str(root), lambda: [], sys.executable)
    rb.reload()
    k = next((i for i, r in enumerate(rb._rows) if r.sim == "TNGR-1-Dark"), None)
    status = rb.table.item(k, RV.COLUMNS.index("Status")).text() if k is not None else ""
    if k is not None:
        rb.table.selectRow(k)
    detail = rb.detail.toPlainText()
    check("Runs tab: GPU retries per cause in 'Running now' (sidecar) and a finished run's status and details (log)",
          state == "running, 3 GPU retries (innocent 2, oom 1)" and status.endswith("2 GPU retries (timeout 2)")
          and "2 GPU retries (timeout 2)" in detail, (state, status, detail[-400:]))


def t_qt_advanced(tmp: Path, spin):
    """The opt-in Advanced settings (python/gui/app_advanced.py). View > Advanced Settings is off in a new window, and
    off the run tabs are the two-mode ones: no group, row or grey line of it anywhere, the same widgets and sizes. On:
    one open group 'Advanced' after the Grids, Custom snapshot and Pipeline tabs' own groups, the rows runspec's
    advanced_rows names for the job (estimator, dimension, build), each on Automatic with its grey note. An override
    reaches the command, the folded summary, the grey 'Advanced:' line under 'Runs on', the checks list and (a Device
    override) the 'Runs on' line. A queued job keeps its overrides, marked. Off puts the three jobs back to Automatic
    and says how many; a job loaded with overrides turns the switch on and says so; only a window that remembers
    keeps the switch (one settings key)."""
    print("Advanced settings (app_advanced.py):")
    import re
    import app as A
    import app_advanced as AD
    import app_tabs as T
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel, QScrollArea, QWidget
    from shiboken6 import getCppPointer
    from app_widgets import AutoRow, Collapsible, tip_lines
    ini = tmp / "advanced.ini"
    w = A.MainWindow(remember=False, settings=QSettings(str(ini), QSettings.IniFormat))
    try:
        w.root_edit.setText(str(tmp))
        w._root_changed()
        w.resize(1100, 750)
        w.show()
        spin(100)
        action = w.advanced_action
        view = next(a.menu() for a in w.menuBar().actions() if a.text() == "View")
        acts = view.actions()
        pages = {t: w._tab_page(t) for t in T.RUN_TABS}

        def traces():
            """Anything of the Advanced group in the run tabs' forms: its title, rows, grey lines, the row titles (not
            the Custom tab's own Precision row, in its Snapshot group)."""
            out = []
            for t, page in pages.items():
                for wd in page.findChildren(QWidget):
                    if isinstance(wd, AutoRow) or (isinstance(wd, QGroupBox) and wd.title() == AD.ADVANCED_TITLE):
                        out.append((t, type(wd).__name__))
                    elif (isinstance(wd, QLabel) and w.o_snap_form.indexOf(wd) < 0
                          and (wd.text() in rs.ADVANCED_ROW.values() or wd.text() == AD.ADVANCED_INTRO
                               or wd.text().startswith("Advanced:"))):
                        up = wd.parentWidget()
                        out.append((t, wd.text(), up.title() if isinstance(up, QGroupBox) else type(up).__name__))
            return out

        def shape():
            """Each run tab's widgets and size, measured while it is the tab on screen (a hidden tab's grey notes
            keep the width of their last layout)."""
            out, was = {}, w.tabs.currentIndex()
            for t, i in (("grids", A.GRIDS), ("custom", A.OWN), ("pipeline", A.PIPELINE)):
                w.tabs.setCurrentIndex(i)
                spin(30)
                p = pages[t]
                out[t] = (len(p.findChildren(QWidget)), p.minimumSizeHint().width(), p.minimumSizeHint().height(),
                          p.sizeHint().height())
            w.tabs.setCurrentIndex(was)
            spin(20)
            return out
        two_mode = shape()

        def focus_chain():
            """The window's chain of widgets Tab stops at, by their C++ objects (a wrapper's identity is not kept)."""
            out, me, x = [], getCppPointer(w)[0], w.nextInFocusChain()
            while x is not None and getCppPointer(x)[0] != me and len(out) < 20000:
                out.append(x)
                x = x.nextInFocusChain()
            return [x for x in out if x.focusPolicy().value & Qt.FocusPolicy.TabFocus.value]
        two_mode_chain = [getCppPointer(x)[0] for x in focus_chain()]
        check("View > Advanced Settings: a checkable item right after Show Commands, off in a new window; off, no "
              "Advanced group, row or grey line on any run tab (nothing of it reachable) and no job has an override",
              action in acts and acts.index(action) == acts.index(w.show_cmds_action) + 1 and action.isCheckable()
              and not action.isChecked() and action.text() == "Advanced Settings" and not w.advanced_on
              and not w.adv_boxes and not traces() and not w.findChildren(AutoRow)
              and all(rs.job_advanced(w._tab_spec(t)) == {} for t in T.RUN_TABS), (acts, traces()))

        action.trigger()
        spin(50)
        g = w.adv_boxes
        cols = {t: pages[t].layout() for t in T.RUN_TABS}
        others = [wd for i in (A.DATA, A.PLOTS, A.EXPLORE) for wd in w.tabs.widget(i).findChildren(Collapsible)
                  if wd.title() == AD.ADVANCED_TITLE]
        check("switched on: one open 'Advanced' group on each run tab, after the tab's own groups (its More group "
              "the one before), its first line the grey sentence; the Data, Plots and Explore tabs get none",
              w.advanced_on and action.isChecked() and set(g) == set(T.RUN_TABS) and not others
              and all(isinstance(g[t], Collapsible) and g[t].title() == AD.ADVANCED_TITLE and g[t].is_open()
                      and cols[t].indexOf(g[t]) == cols[t].indexOf(w._shared(t, "more")) + 1
                      and cols[t].itemAt(cols[t].count() - 1).spacerItem() is not None
                      and g[t].form.itemAt(0, QFormLayout.SpanningRole).widget().text() == AD.ADVANCED_INTRO
                      and g[t].key == f"{t}.advanced_settings" for t in T.RUN_TABS),
              {t: (cols[t].indexOf(g[t]), cols[t].count()) for t in g})

        def shown(t):
            return [k for k, r in w.adv_rows[t].items() if g[t].form.isRowVisible(r)]

        def on_automatic(t):
            """Every row shown on Automatic (its first choice), titled as runspec names it, its grey note what
            Automatic gives the job now; a drop-down offers runspec's choices."""
            spec, bad = w._tab_spec(t), []
            for k in shown(t):
                r = w.adv_rows[t][k]
                c = r.control
                first = c.specialValueText() if r.is_number else c.itemText(0)
                items = [] if r.is_number else [c.itemText(i) for i in range(c.count())]
                ok = (first == rs.AUTOMATIC and r.value() is None
                      and (c.value() == c.minimum() if r.is_number else c.currentIndex() == 0)
                      and g[t].form.labelForField(r).text() == rs.ADVANCED_ROW[k]
                      and r.note.text() == rs.automatic_text(spec, k)
                      and (r.is_number or items == [x for _, x in rs.advanced_choices(spec, k)]))
                if not ok:
                    bad.append((t, k, first, items, r.note.text()))
            return bad
        rows = {t: shown(t) for t in T.RUN_TABS}
        ps3 = {"gpu", "nsub", "face_gradient", "partition", "max_concurrent", "mem_budget_gb",
               "parallel_triangulation", "ordered_merge"}
        bad = [b for t in T.RUN_TABS for b in on_automatic(t)]
        check("PS-DTFE in 3D, every tab: the rows runspec.advanced_rows names (the device, samples, velocity "
              "derivatives, partitions, concurrency, memory budget, parallel triangulation, merge order; the "
              "precision on Grids and Pipeline with the double pair), each on Automatic with the grey note of what "
              "Automatic gives",
              all(rows[t] == list(rs.advanced_rows(w._tab_spec(t))) and ps3 <= set(rows[t]) for t in T.RUN_TABS)
              and "precision" not in w.adv_rows["custom"] and not bad, (rows, bad[:2]))
        nsub = w.adv_rows["grids"]["nsub"]
        exact = (nsub.isEnabled(), nsub.note.text())
        w.quality_pair.button("fast").setChecked(True)
        spin(20)
        fast = (nsub.isEnabled(), nsub.note.text(), rs.automatic_text(w.spec, "nsub"), not on_automatic("grids"))
        w.quality_pair.button("exact").setChecked(True)
        spin(20)
        check("the samples row: greyed under Exact with 'Exact uses no samples', open under Fast with Automatic's "
              "count", exact == (False, "Exact uses no samples") and fast[0] and fast[1] == fast[2]
              and fast[1].startswith("Automatic: ") and fast[3], (exact, fast))
        w.dtfe_radio.setChecked(True)
        w.o_dim.setCurrentIndex(w.o_dim.findData(2))
        spin(20)
        std, flat = shown("grids"), shown("custom")
        bad = on_automatic("grids") + on_automatic("custom")
        w.ps_radio.setChecked(True)
        w.o_dim.setCurrentIndex(w.o_dim.findData(3))
        spin(20)
        check("... standard DTFE: the device, partitions, concurrency and budget (and the precision); a 2D Custom "
              "job: no device, velocity derivatives or parallel triangulation (the samples and merge order stay)",
              std == list(rs.advanced_rows(rs.RunSpec(estimator="dtfe")))
              and set(std) - {"precision"} == {"gpu", "partition", "max_concurrent", "mem_budget_gb"}
              and flat == list(rs.advanced_rows(rs.CustomSpec(dim=2)))
              and set(flat) == {"nsub", "partition", "max_concurrent", "mem_budget_gb", "ordered_merge"}
              and not bad and shown("grids") == rows["grids"], (std, flat, bad[:2]))

        # an override on each tab reaches its command, and only it changes there
        base = {t: w._tab_spec(t).to_dict() for t in T.RUN_TABS}
        w.adv_rows["grids"]["partition"].control.setValue(4)
        w.adv_rows["custom"]["max_concurrent"].control.setValue(2)
        w.adv_rows["pipeline"]["mem_budget_gb"].control.setValue(40)
        spin(20)
        bare = {t: type(w._tab_spec(t)).from_dict(base[t]) for t in T.RUN_TABS}

        def diff(a, b):
            return {k: a.get(k) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        genv, penv = w.spec.command()[1], w.pipe.command()[1]
        own = w.custom.command()
        k = own.index("--max-concurrent") if "--max-concurrent" in own else -1
        check("an override reaches the job and its command, nothing else changes: Grids PARTITION='4 4 4', Custom "
              "'--max-concurrent 2', Pipeline DTFE_MEM_BUDGET_GB=40",
              (w.spec.advanced, w.custom.advanced, w.pipe.advanced)
              == ({"partition": 4}, {"max_concurrent": 2}, {"mem_budget_gb": 40.0})
              and diff(genv, bare["grids"].command()[1]) == {"PARTITION": "4 4 4"}
              and w.spec.command()[0] == bare["grids"].command()[0]
              and diff(penv, bare["pipeline"].command()[1]) == {"DTFE_MEM_BUDGET_GB": "40"}
              and w.pipe.command()[0] == bare["pipeline"].command()[0]
              and k > 0 and own[k + 1] == "2" and own[:k] + own[k + 2:] == bare["custom"].command(),
              (diff(genv, bare["grids"].command()[1]), diff(penv, bare["pipeline"].command()[1]), own))
        lines = {t: w.adv_lines[t] for t in T.RUN_TABS}

        def under_runs_on(t):
            form = lines[t].parentWidget().layout()
            return form.getWidgetPosition(lines[t])[0] == form.getWidgetPosition(w._shared(t, "runs_on"))[0] + 1
        w.tabs.setCurrentIndex(A.GRIDS)
        spin(20)
        checks = [w.checks.item(i).text() for i in range(w.checks.count())]
        check("... and the tab shows it without opening the group: the grey 'Advanced: Partitions: 4' line right "
              "under 'Runs on', the same as an info line of the checks, and the folded group's summary",
              [lines[t].text() for t in T.RUN_TABS] == ["Advanced: Partitions: 4", "Advanced: Concurrent "
                                                        "triangulations: 2", "Advanced: Memory budget: 40 GB"]
              and all(not lines[t].isHidden() and under_runs_on(t) for t in T.RUN_TABS)
              and "ℹ  Advanced: Partitions: 4" in checks and g["grids"].summary_text() == "Partitions: 4"
              and g["custom"].summary_text() == "Concurrent triangulations: 2", (checks, g["grids"].summary_text()))

        # the Device row: CPU on a GPU build moves the run and the 'Runs on' line says why (a GPU build faked: the
        # device's stamp too, which Automatic's device reads, so a CPU-only build checks the same)
        built, backend = rs.gpu_built, rs.gpu_build_default
        rs.gpu_built = lambda *a, **k: True
        rs.gpu_build_default = lambda *a, **k: True
        try:
            w.refresh()
            dev = w.adv_rows["grids"]["gpu"]
            items = [dev.control.itemText(i) for i in range(dev.control.count())]
            dev.control.setCurrentIndex(items.index("CPU"))
            spin(20)
            cpu = (w.spec.advanced.get("gpu"), w.runs_on_note.text(), w.spec.mode().gpu, "-m" in w.spec.command()[0])
            # CPU + GPU (the command and the memory check's, never run here: the prebuilt programs do not know
            # 'mixed' yet), then GPU under Performance (every partition on the GPU, where Automatic shares)
            dev.control.setCurrentIndex(items.index("CPU + GPU"))
            spin(20)
            both = (w.spec.advanced.get("gpu"), w.runs_on_note.text(), w.runs_on_note.toolTip(), dev.note.text(),
                    w.spec.command()[1]["GPU_SHARE"], w.spec.check_step().env["GPU_SHARE"], w.adv_lines["grids"].text())
            w.run_mode_combo.setCurrentIndex(w.run_mode_combo.findData("performance"))
            dev.control.setCurrentIndex(items.index("GPU"))
            spin(20)
            whole = (w.spec.advanced.get("gpu"), w.runs_on_note.text(), w.runs_on_note.toolTip(), dev.note.text(),
                     w.spec.command()[1]["GPU_SHARE"])
            dev.control.setCurrentIndex(0)
            spin(20)
            auto_perf = (w.runs_on_note.text(), dev.note.text(), bool(w.runs_on_note.toolTip()),
                         w.spec.command()[1]["GPU_SHARE"])
            w.quality_pair.button("fast").setChecked(True)
            dev.control.setCurrentIndex(items.index("CPU + GPU"))
            spin(20)
            fast = (dev.note.text(), w.spec.command()[1]["GPU_SHARE"],
                    [w.checks.item(i).text() for i in range(w.checks.count()) if "Device set to" in w.checks.item(i).text()])
            dev.control.setCurrentIndex(0)
            w.quality_pair.button("exact").setChecked(True)
            w.run_mode_combo.setCurrentIndex(w.run_mode_combo.findData("balanced"))
            spin(20)
        finally:
            rs.gpu_built, rs.gpu_build_default = built, backend
            w.refresh()
        check("the Device row (a GPU build): Automatic, CPU, GPU, CPU + GPU; CPU takes the run off the GPU and the "
              "'Runs on' line ends '(set in Advanced)'; back on Automatic it is gone",
              items == ["Automatic", "CPU", "GPU", "CPU + GPU"]
              and cpu == (False, "Runs on: the CPU (set in Advanced)", False, False)
              and "gpu" not in w.spec.advanced, (items, cpu))
        check("... CPU + GPU: 'the CPU and the GPU together (set in Advanced)' with the precise statement as the "
              "line's tooltip, GPU_SHARE=mixed in the run and its memory check, 'Device: CPU + GPU' under 'Runs on'; "
              "GPU under Performance: every partition on the GPU (GPU_SHARE=1), the line says the GPU; on Automatic "
              "in Performance the row's note and the line both say the job shares (GPU_SHARE=auto); under Fast CPU + "
              "GPU is not passed, the note and the Checks say why",
              both == ("mixed", "Runs on: the CPU and the GPU together (set in Advanced)", rs.RUNS_ON_MIXED_TIP, "",
                       "mixed", "mixed", "Advanced: Device: CPU + GPU; Partitions: 4")
              and whole == (True, "Runs on: the GPU (set in Advanced)", "", "", "1")
              and auto_perf == ("Runs on: the CPU and the GPU together (chosen for Exact, Performance)",
                                "Automatic: CPU + GPU", True, "auto")
              and fast[0] == "Not passed: only the Exact deposit can be shared" and fast[1] == "auto"
              and len(fast[2]) == 1 and fast[2][0].startswith("⚠  Advanced: Device set to CPU + GPU is not passed"),
              (both, whole, auto_perf, fast))

        # the Grids estimate line names a Double setting's precision, as before the two modes (the double pair faked)
        pair = rs.binary_built
        rs.binary_built = lambda est, precision="single", dim=3: precision == "double" or pair(est, precision, dim)
        try:
            w.tabs.setCurrentIndex(A.GRIDS)
            w.refresh()
            spin(20)
            prow = w.adv_rows["grids"]["precision"]
            single_est = (g["grids"].form.isRowVisible(prow), w.estimate.text())
            prow.control.setCurrentIndex([prow.control.itemText(i) for i in range(prow.control.count())]
                                         .index("Double"))
            spin(20)
            double_est = (w.spec.advanced.get("precision"), w.estimate.text())
            prow.control.setCurrentIndex(0)
            spin(20)
        finally:
            rs.binary_built = pair
            w.refresh()
        check("the Grids estimate line says '(float64)' beside the grids under an Advanced Precision Double, as "
              "before the two modes; not under single precision",
              single_est[0] and "full-resolution grids" in single_est[1] and "(float64)" not in single_est[1]
              and double_est[0] == "double" and "  (float64)" in double_est[1] and "precision" not in w.spec.advanced,
              (single_est, double_est))

        # a row holding a value the job cannot use stays shown and open, noted, so it can be set back; the Checks
        # list warns and the command is the Automatic one
        samples, merge = w.adv_rows["grids"]["nsub"], w.adv_rows["grids"]["ordered_merge"]
        w.quality_pair.button("fast").setChecked(True)
        spin(20)
        samples.control.setValue(2)
        spin(20)
        fast_line = (w.spec.mode().nsub, w.quality_note.text())   # Fast at an Advanced count: the grey line names none
        merge.control.setCurrentIndex([merge.control.itemText(i) for i in range(merge.control.count())]
                                      .index("Completion order"))
        w.quality_pair.button("exact").setChecked(True)
        spin(20)
        under_exact = (samples.isEnabled(), g["grids"].form.isRowVisible(samples), samples.note.text(),
                       w.spec.advanced.get("nsub"), "-n" in w.spec.command()[0])
        w.dtfe_radio.setChecked(True)
        spin(20)
        on_std = (g["grids"].form.isRowVisible(merge), merge.isEnabled(), merge.note.text(),
                  w.spec.advanced.get("ordered_merge"), w.spec.command() == rs.RunSpec.from_dict(
                      dict(w.spec.to_dict(), advanced={"partition": 4})).command())
        warned = [x for x in (w.checks.item(i).text() for i in range(w.checks.count())) if "is not passed" in x]
        w.ps_radio.setChecked(True)
        samples.control.setValue(samples.control.minimum())
        merge.control.setCurrentIndex(0)
        spin(20)
        check("a row holding a value the job cannot use stays shown and open, noted 'Not passed: <why>', the Checks "
              "list warns and the command is the Automatic one (Samples 2 under Exact, Completion order on standard "
              "DTFE); set back to Automatic, the value is gone; Fast at Samples 2 does not say 'One sample per cell'",
              under_exact == (True, True, "Not passed: Exact uses no samples", 2, False)
              and fast_line == (2, rs.QUALITY_TEXT_PS_FAST_SAMPLES)
              and on_std == (True, True, "Not passed: a PS-DTFE setting", False, True)
              and len(warned) == 2 and all(x.startswith("⚠  Advanced: ") for x in warned)
              and any("Merge order set to completion order is not passed: a PS-DTFE setting" in x for x in warned)
              and w.spec.advanced == {"partition": 4}, (under_exact, on_std, warned, w.spec.advanced, fast_line))

        # a queued job keeps its overrides, marked in its row and named in its tooltip
        n0 = len(w.queue)
        w.queue.append(rs.QueuedJob.of("grids", w.spec))
        plain = rs.RunSpec.from_dict(dict(w.spec.to_dict(), advanced={}))
        w.queue.append(rs.QueuedJob.of("grids", plain))
        w._refresh_queue()
        marked, unmarked = w.q_list.item(n0), w.q_list.item(n0 + 1)
        w._show_queued(n0)
        detail = w.q_detail.toPlainText().splitlines()
        check("a queued job with overrides: its row says 'advanced', its tooltip names them; one without: neither",
              marked.text().endswith("· advanced") and "Advanced: Partitions: 4" in marked.toolTip()
              and "advanced" not in unmarked.text() and "Advanced" not in unmarked.toolTip(),
              (marked.text(), marked.toolTip()))
        check("... its detail pane marks each check as the Checks list does: the Advanced information line 'ℹ', no "
              "warning sign on it",
              "ℹ Advanced: Partitions: 4" in detail and not any(x.startswith("⚠ Advanced: Partitions") for x in detail)
              and all(x[:2] in ("✖ ", "⚠ ", "ℹ ", "→ ") for x in detail[detail.index("Checks now:") + 1:] if x),
              detail[-6:])

        # the texts: plain, short tooltips (What's This for more), no '--', no measured number
        texts = []
        for t in T.RUN_TABS:
            for wd in [g[t]] + g[t].findChildren(QWidget) + [lines[t]]:
                texts += [wd.toolTip(), wd.whatsThis()]
                if isinstance(wd, QLabel):
                    texts.append(wd.text())
                if isinstance(wd, QComboBox):
                    texts += [wd.itemText(i) for i in range(wd.count())]
        measured = re.compile(r"\d\s*(?:s|min|GB|MB|%)\b")
        check("the group's texts: tooltips of at most two lines, no '--', no measured number (seconds, GB, percent)",
              all(tip_lines(r.toolTip()) <= 2 for t in T.RUN_TABS for r in w.adv_rows[t].values())
              and not [x for x in texts if "--" in x]
              and not [x for x in texts if measured.search(x) and not x.startswith("Advanced: Memory budget")],
              [x for x in texts if "--" in x or measured.search(x)][:3])

        # every run tab's form still fits its pane with the group open, and Tab walks into the group last
        fits = []
        for t, i in (("grids", A.GRIDS), ("custom", A.OWN), ("pipeline", A.PIPELINE)):
            w.tabs.setCurrentIndex(i)
            spin(30)
            s = w.tabs.widget(i)
            fits.append(isinstance(s, QScrollArea) and s.widget().minimumSizeHint().width() <= s.viewport().width())
        tabbing = focus_chain()
        order = all(
            (lambda own, mine: own and mine and max(own) + 1 == min(mine))(
                [i for i, x in enumerate(tabbing) if pages[t].isAncestorOf(x) and not g[t].isAncestorOf(x)],
                [i for i, x in enumerate(tabbing) if g[t].isAncestorOf(x)]) for t in T.RUN_TABS)
        check("with the groups open every run tab's form fits its pane at 1100 px, and Tab goes from the tab's own "
              "last field into its Advanced group", all(fits) and order, (fits, order))

        # off: the three jobs back to Automatic (the status line says how many), the forms the two-mode ones again;
        # the queued job keeps its overrides
        action.trigger()
        spin(50)
        status = w.status.text()
        check("switched off: the three tabs' jobs back to Automatic, 'Advanced settings off: 3 settings back to "
              "Automatic', no group, row or line left (the same widgets, sizes and Tab order as before it went on), "
              "the commands the two-mode ones; the queued job still runs with its overrides, still marked",
              not w.advanced_on and not action.isChecked() and not w.adv_boxes and not w.adv_rows
              and all(rs.job_advanced(w._tab_spec(t)) == {} for t in T.RUN_TABS)
              and status == "Advanced settings off: 3 settings back to Automatic"
              and not traces() and not w.findChildren(AutoRow) and shape() == two_mode
              and [getCppPointer(x)[0] for x in focus_chain()] == two_mode_chain
              and w.spec.command() == bare["grids"].command() and w.pipe.command() == bare["pipeline"].command()
              and w.custom.command() == bare["custom"].command()
              and w.queue[n0].build().command()[1]["PARTITION"] == "4 4 4"
              and w.q_list.item(n0).text().endswith("· advanced"), (status, traces(), shape(), two_mode))
        w.status_message("unchanged")
        action.trigger()
        action.trigger()
        spin(20)
        check("... on and off again with no override: nothing is said", w.status.text() == "unchanged"
              and not w.advanced_on, w.status.text())
        del w.queue[n0:]
        w._refresh_queue()

        # a job loaded with overrides while the switch is off turns it on, and the status line says so
        w.custom = rs.CustomSpec(output_dir=str(tmp), advanced={"partition": 2, "bogus": 1})
        w._load_into_widgets()
        w.refresh()

        def row_value(t, key):
            return w.adv_rows[t][key].value() if t in w.adv_rows else "no row"
        loaded = (w.advanced_on, action.isChecked(), row_value("custom", "partition"), w.status.text(),
                  w.custom.advanced)
        check("a job loaded with overrides (the switch off) turns it on: the row shows the value, a bad key is "
              "dropped, the status line 'Advanced settings on: the loaded job has 1 setting in the Advanced group'",
              loaded == (True, True, 2, "Advanced settings on: the loaded job has 1 setting in the Advanced group",
                         {"partition": 2}),
              loaded)
        action.trigger()                    # off again (the Custom job back to Automatic)
        # ... a custom run's settings sidecar too (Runs > Load Settings), its status line saying so
        side_dir = tmp / "advcustom"
        side_dir.mkdir(exist_ok=True)
        ran = rs.CustomSpec(input_file=str(side_dir / "in.hdf5"), output_dir=str(side_dir), output_name="out",
                            advanced={"max_concurrent": 3})
        (side_dir / "out.runlog").write_text(LOG_OK.format(root=side_dir / "out"))
        (side_dir / "out.gui.json").write_text(json.dumps(ran.to_dict()))
        w._load_run_settings(R.parse_runlog(side_dir / "out.runlog"))
        side = (w.advanced_on, w.custom.advanced, row_value("custom", "max_concurrent"), w.status.text())
        # ... and a TNG run's settings (its log names no Advanced settings): the Grids job's go back to Automatic
        w.adv_rows["grids"]["partition"].control.setValue(2)
        spin(20)
        d = snapdir(tmp, sim="TNGA-1-Dark", snap=99)
        (d / "ps_output.runlog").write_text(LOG_OK.format(root=d / "ps_output"))
        w._load_run_settings(R.parse_runlog(d / "ps_output.runlog", "TNGA-1-Dark", 99))
        tng = (w.spec.advanced, w.adv_rows["grids"]["partition"].value(), w.status.text())
        check("Runs > Load Settings: a custom run's sidecar with overrides turns the switch on (the status line says "
              "so); a TNG run's settings put the Grids job's overrides back to Automatic (said too)",
              side == (True, {"max_concurrent": 3}, 3,
                       "loaded the settings of the custom run out (Advanced settings on: 1 setting in the Advanced "
                       "group)")
              and tng[:2] == ({}, None) and tng[2].endswith("; the Advanced settings are back to Automatic"),
              (side, tng))
        # ... where the loaded job hides a row the tab held a value in (another estimator or dimension), the row goes
        # back to Automatic too: the next read of the rows must not bring the value back into the loaded job
        mrow = w.adv_rows["grids"]["ordered_merge"]
        mrow.control.setCurrentIndex([mrow.control.itemText(i) for i in range(mrow.control.count())]
                                     .index("Completion order"))
        spin(20)
        held = dict(w.spec.advanced)
        dd = snapdir(tmp, sim="TNGB-1-Dark", snap=99)
        (dd / "dtfe_output.runlog").write_text(LOG_OK.replace("PS-DTFE", "DTFE").format(root=dd / "dtfe_output"))
        std_rec = R.parse_runlog(dd / "dtfe_output.runlog", "TNGB-1-Dark", 99)
        w._load_run_settings(std_rec)
        spin(20)
        mrow = w.adv_rows["grids"]["ordered_merge"]
        std_load = (std_rec.estimator, w.spec.estimator, w.spec.advanced,
                    w.adv_boxes["grids"].form.isRowVisible(mrow), mrow.value(),
                    w.status.text().endswith("; the Advanced settings are back to Automatic"))
        w.ps_radio.setChecked(True)
        spin(20)
        back_ps = (w.spec.advanced, w.spec.command()[1].get("PS_MERGE"))
        check("... a standard-DTFE run's settings loaded over Completion order: the Grids job holds no setting (the "
              "hidden merge row back on Automatic, the status line saying so), nor once PS-DTFE is chosen again",
              held == {"ordered_merge": False} and std_load == ("dtfe", "dtfe", {}, False, None, True)
              and back_ps == ({}, "index-order"), (held, std_load, back_ps))
        prow = w.adv_rows["custom"]["parallel_triangulation"]
        prow.control.setCurrentIndex([prow.control.itemText(i) for i in range(prow.control.count())].index("On"))
        spin(20)
        held3d = dict(w.custom.advanced)
        flat = rs.CustomSpec(input_file=str(side_dir / "in.hdf5"), output_dir=str(side_dir), output_name="flat",
                             dim=2)
        (side_dir / "flat.runlog").write_text(LOG_OK.format(root=side_dir / "flat"))
        (side_dir / "flat.gui.json").write_text(json.dumps(flat.to_dict()))
        w._load_run_settings(R.parse_runlog(side_dir / "flat.runlog"))
        spin(20)
        flat_load = (w.custom.dim, w.custom.advanced)
        flat_status = w.status.text()
        w.o_dim.setCurrentIndex(w.o_dim.findData(3))
        spin(20)
        back_3d = (w.custom.dim, w.custom.advanced)
        check("... a 2D custom run's settings (none of them Advanced) loaded over Parallel triangulation On: the "
              "Custom job holds no setting, nor once it is 3D again, and the status line says the Advanced settings "
              "are back to Automatic (as a TNG run's load does)",
              held3d == {"max_concurrent": 3, "parallel_triangulation": True} and flat_load == (2, {})
              and back_3d == (3, {})
              and flat_status == "loaded the settings of the custom run flat; the Advanced settings are back to "
                                 "Automatic", (held3d, flat_load, back_3d, flat_status))
        action.trigger()
    finally:
        w._machine_timer.stop()
        w.close()
        spin(50)
    check("a window that does not remember never writes the switch's settings key",
          not QSettings(str(ini), QSettings.IniFormat).contains(AD.ADVANCED_SETTING))

    # remembered: the switch (one key) comes back on at the next start; a remembered job with overrides turns it on
    ini2 = str(tmp / "advanced-remember.ini")
    fresh = A.MainWindow(remember=True, settings=QSettings(ini2, QSettings.IniFormat))
    first = (fresh.advanced_on, fresh.advanced_action.isChecked(), dict(fresh.adv_boxes))
    fresh.advanced_action.trigger()
    fresh.adv_boxes["grids"].set_open(False)            # folded by the user: so it starts next time
    fresh._machine_timer.stop()
    fresh.close()
    kept_on = str(QSettings(ini2, QSettings.IniFormat).value(AD.ADVANCED_SETTING))
    w2 = A.MainWindow(remember=True, settings=QSettings(ini2, QSettings.IniFormat))
    again = (w2.advanced_on, w2.advanced_action.isChecked(), sorted(w2.adv_boxes),
             {t: g.is_open() for t, g in w2.adv_boxes.items()})
    w2.advanced_action.trigger()
    w2.advanced_action.trigger()                        # off and on again: switched on, every group opens
    reopened = {t: g.is_open() for t, g in w2.adv_boxes.items()}
    w2.advanced_action.trigger()
    w2._machine_timer.stop()
    w2.close()
    st = QSettings(ini2, QSettings.IniFormat)
    kept_off = str(st.value(AD.ADVANCED_SETTING))
    st.setValue("pipeline", json.dumps(rs.PipelineSpec(advanced={"max_concurrent": 2}).to_dict()))
    st.sync()
    w3 = A.MainWindow(remember=True, settings=QSettings(ini2, QSettings.IniFormat))
    started = (w3.advanced_on, w3.pipe.advanced, w3.status.text(),
               w3.adv_lines["pipeline"].text() if "pipeline" in w3.adv_lines else "no line")
    w3._machine_timer.stop()
    w3.close()
    check("a window that remembers keeps the switch under one settings key: off in a new store, on again at the next "
          "start once switched on (the groups built, each open or folded as left), every group open when it is "
          "switched on again, off once switched off; a remembered job with overrides starts it on, the status line "
          "saying so",
          first == (False, False, {}) and kept_on == "true"
          and again == (True, True, sorted(T.RUN_TABS), {"grids": False, "custom": True, "pipeline": True})
          and reopened == {t: True for t in T.RUN_TABS} and kept_off == "false"
          and started == (True, {"max_concurrent": 2},
                          "Advanced settings on: the loaded job has 1 setting in the Advanced group",
                          "Advanced: Concurrent triangulations: 2"),
          (first, kept_on, again, reopened, kept_off, started))

    # at the start, the two-mode note (a queue saved before the two modes) must not hide that a remembered job with
    # an Advanced setting switched them on
    ini3 = str(tmp / "advanced-modes-note.ini")
    st3 = QSettings(ini3, QSettings.IniFormat)
    st3.setValue("pipeline", json.dumps(rs.PipelineSpec(advanced={"max_concurrent": 2}).to_dict()))
    old_spec = {k: v for k, v in rs.RunSpec().to_dict().items() if k != "quality"}
    st3.setValue("queue", json.dumps([{"kind": "grids", "spec": dict(old_spec, deposit="sampled"), "title": "older"}]))
    st3.sync()
    w4 = A.MainWindow(remember=True, settings=QSettings(ini3, QSettings.IniFormat))
    both = (w4.advanced_on, w4.status.text(), [j.title for j in w4.queue])
    w4._machine_timer.stop()
    w4.close()
    check("at the start, the two-mode note does not hide 'Advanced settings on' (a remembered job with a setting "
          "beside a queue saved before the two modes): the status line says both; the old entry's title is made "
          "again in the two modes' words",
          both[:2] == (True, rs.MODES_NOTE + "; Advanced settings on") and both[2] != ["older"]
          and both[2][0].endswith(", Fast"), both)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gui_app2_"))
    atexit.register(shutil.rmtree, tmp, True)
    t_run_files(tmp)
    t_parse_cache(tmp)
    t_memory()
    t_running_now(tmp)
    t_keep_awake()
    t_plot_sets(tmp)
    t_void_status_headless(tmp)
    t_qt(tmp)
    t_void_status_data()
    print("-" * 60)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
