#!/usr/bin/env python3
"""The python CLIs' option names: each script's command line in the current names parses to the values given
(one argv per script, CURRENT; make_image_plane.py's --plane-window also as its --same-as reads it); no option is
guessed from an abbreviation (allow_abbrev=False); the smoothing variable DTFE_SMOOTH_MPC fixes the void smoothing
in Mpc.

Each script runs in a fresh interpreter, its parser stopped right after parse_args (argparse patched to raise with
the Namespace): nothing is read or written, no data is needed (make_image_plane.py --same-as reads a sidecar this
test writes, nothing else).

Usage:  python3 tests/py_cli_alias_test.py            (exit 0 = pass)
"""

import json
import os
import subprocess
import sys
import tempfile

from _harness import REPO, check, summary

PY = REPO / "python"
SIM = "TNG100-3-Dark"

# the driver run in each child: argparse.ArgumentParser.parse_args raises with the Namespace, so main() stops there;
# each case runs the script afresh (runpy, as __main__)
DRIVER = r'''
import argparse, contextlib, io, json, os, runpy, sys
script, cases = sys.argv[1], json.loads(sys.argv[2])
sys.path[:0] = [os.path.dirname(script), os.path.join(os.path.dirname(script), ".."), os.path.dirname(script) + "/.."]
class Parsed(Exception):
    pass
orig = argparse.ArgumentParser.parse_args
def grab(self, args=None, namespace=None):
    raise Parsed(orig(self, args, namespace))
argparse.ArgumentParser.parse_args = grab
out = []
for argv in cases:
    err, std, ns, code = io.StringIO(), io.StringIO(), None, None
    sys.argv = [script] + argv
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(std):
            runpy.run_path(script, run_name="__main__")
    except Parsed as p:
        ns = {k: (str(v) if not isinstance(v, (int, float, str, bool, list, type(None))) else v)
              for k, v in vars(p.args[0]).items()}
    except SystemExit as e:
        code = e.code
    out.append({"ns": ns, "stderr": err.getvalue(), "stdout": std.getvalue(), "code": code})
print("@@" + json.dumps(out))
'''


def parse(script: str, *cases: list) -> list[dict]:
    """[{'ns', 'stderr', 'stdout', 'code'}] for each argv of 'cases', parsed by python/<script> in one child."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, DTFE_SIM=SIM, DTFE_DATA_ROOT=tmp, MPLBACKEND="Agg", DTFE_FIGURES_ROOT=tmp)
        env.pop("DTFE_SMOOTH_MPC", None)
        r = subprocess.run([sys.executable, "-c", DRIVER, str(PY / script), json.dumps([list(c) for c in cases])],
                           capture_output=True, text=True, env=env, cwd=str(PY), timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("@@")), None)
    assert line, f"{script}: no result ({r.returncode}): {r.stderr[-800:]}"
    return json.loads(line[2:])


# one command line per script in the current names, and the values its parser must give (the Namespace's names: an
# option's dest where it differs, '--estimator' -> method, '--root' -> prefix, '--maps' -> fields, ...; a Path as
# its text). plot_void_crossbox.py has two parsers: the figure's and its per-box worker's ('--worker')
CURRENT = {
    "plot/plot_DTFE.py": [(["--sim", SIM, "--snap", "50", "--estimator", "ps"],
                           {"sim": SIM, "snap": 50, "method": "ps"})],
    "plot/plot_halo_tracking.py": [(["--sim", SIM, "--start-snap", "50", "--subhalos", "1", "2", "--n-halos", "3",
                                     "--estimator", "dtfe"],
                                    {"snap": 50, "subhalo": [1, 2], "top": 3, "method": "dtfe"})],
    "plot/plot_void_tracking.py": [(["--sim", SIM, "--start-snap", "50"], {"snap": 50})],
    "plot/plot_pk_compare.py": [(["--sim", SIM, "--n-bins", "12", "--out", "/x", "--snaps", "1", "2"],
                                 {"nbins": 12, "out": "/x", "snaps": [1, 2]})],
    "plot/plot_void_population.py": [(["--sim", SIM, "--size-rv-definition", "ridge"], {"size_rv": "ridge"})],
    "plot/plot_caustic_skeleton.py": [(["--sim", SIM, "--per-void"], {"voids": True})],
    "plot/plot_void_environment.py": [(["--sim", SIM, "--criterion", "density-raw", "--rv-definition", "delta08"],
                                       {"criterion": "density_raw", "r_v_definition": "delta08"})],
    "plot/plot_void_profile_evolution.py": [(["--sim", SIM, "--rv-range", "2", "4", "--edges", "0", "10", "20"],
                                             {"rv_range": [2.0, 4.0], "edges": [0.0, 10.0, 20.0]})],
    "plot/plot_void_profiles.py": [(["--sim", SIM, "--edges", "0", "4", "30"], {"edges": [0.0, 4.0, 30.0]})],
    "plot/plot_void_convergence.py": [(["--sim", SIM, "--redshifts", "0", "1"], {"z": [0.0, 1.0]})],
    "plot/plot_void_crossbox.py": [(["--sims", "A", "B", "--estimator", "dtfe", "--redshifts", "0", "1"],
                                    {"sims": ["A", "B"], "method": "dtfe", "z": [0.0, 1.0]}),
                                   (["--worker", "/x.npz", "--sim", "A", "--estimator", "dtfe", "--smooth-mpc", "2",
                                     "--redshifts", "0"],
                                    {"worker": "/x.npz", "sim": "A", "method": "dtfe", "smooth_mpc": 2.0,
                                     "z": [0.0]})],
    "plot/plot_pointeval.py": [(["--sims", "A", "B", "--snaps", "4", "99", "--maps", "density", "velDiv",
                                 "--estimator", "dtfe", "--smooth-px", "2", "--smooth-derivatives-px", "0"],
                                {"sims": ["A", "B"], "snaps": [4, 99], "fields": ["density", "velDiv"],
                                 "method": "dtfe", "smooth": 2.0, "smooth_derivatives": 0.0})],
    "plot/plot_pointeval_panels.py": [(["--root", "/r", "--plane", "/p.json", "--out", "/o.png", "--panels",
                                        "density", "sheet", "--project", "slab"],
                                       {"prefix": "/r", "plane": "/p.json", "out": "/o.png",
                                        "panels": ["density", "sheet"], "project": "slab"})],
    "tools/void_outlines.py": [(["--sim", SIM, "--snap", "99", "--estimator", "ps", "--smooth-mpc", "2"],
                                {"snap": 99, "method": "ps", "smooth_mpc": 2.0})],
    "tools/compare_deposits.py": [(["--sim", SIM, "--snap", "99", "--prefix-a", "x", "--prefix-b", "y"],
                                   {"snap": 99, "prefix_a": "x", "prefix_b": "y"})],
    "gui/void_cache.py": [(["build", "--sim", SIM, "--snap", "99", "--estimator", "dtfe"],
                           {"mode": "build", "sim": SIM, "snap": 99, "method": "dtfe"})],
    "tools/merge_HDF5.py": [(["--sim-dir", "/d", "--snaps", "8", "17"], {"data_dir": "/d", "snaps": [8, 17]})],
    "tools/make_image_plane.py": [(["--same-as", "x.json", "--plane-window", "0", "50", "0", "25", "--out", "s"],
                                   {"same_as": "x.json", "plane_window": [0.0, 50.0, 0.0, 25.0], "out": "s"})],
    "analyze.py": [(["export", "--snaps", "099", "050", "--out", "/o", "--sim", SIM],
                    {"mode": "export", "snaps": ["099", "050"], "out": "/o", "sim": SIM})],
}


def t_current(script):
    """One script's command lines of CURRENT parse to the values given."""
    cases = CURRENT[script]
    for (argv, want), r in zip(cases, parse(script, *[argv for argv, _ in cases])):
        assert r["ns"] is not None, f"{script} {argv}: not parsed ({r['code']}): {r['stderr'][-400:]}"
        got = {k: r["ns"].get(k) for k in want}
        assert got == want, f"{script} {argv}: {got} != {want}"


def t_plane_window_spread():
    """make_image_plane.py --plane-window U0 U1 V0 V1 is the plane's u0, u1, v0, v1: --same-as finds a sidecar with
    that window the same (exit 0) and one with the two ranges swapped different (exit 3)."""
    tool = PY / "tools" / "make_image_plane.py"
    with tempfile.TemporaryDirectory() as tmp:
        codes = []
        for u1, v1 in ((50.0, 25.0), (25.0, 50.0)):
            side = os.path.join(tmp, f"side_{u1:g}.json")
            with open(side, "w") as f:
                json.dump({"box": 100.0, "planes": 1, "u0": 0.0, "u1": u1, "v0": 0.0, "v1": v1}, f)
            r = subprocess.run([sys.executable, str(tool), "--same-as", side, "--plane-window", "0", "50", "0", "25"],
                               capture_output=True, text=True, cwd=str(PY), timeout=120)
            codes.append((r.returncode, r.stdout.strip()))
    assert codes == [(0, "same"), (3, "differs")], codes


def t_no_abbreviation():
    """allow_abbrev=False everywhere: a prefix of an option is an error, not a guess."""
    cases = {"plot/plot_pk_compare.py": ["--sim", SIM, "--n-bin", "3"],
             "plot/plot_pointeval.py": ["--sims", "A", "--smooth-p", "1"],
             "plot/plot_halo_tracking.py": ["--sim", SIM, "--start", "50"],
             "tools/merge_HDF5.py": ["-d", "/d", "--sim"],          # a prefix of --sim-dir
             "tools/make_image_plane.py": ["--plane-win", "0", "1", "0", "1"],
             "analyze.py": ["check", "--sna", "099"]}
    for script, argv in cases.items():
        r = parse(script, argv)[0]
        assert r["code"] == 2 and "unrecognized arguments" in r["stderr"], (script, r)


def t_smooth_mpc_env():
    """DTFE_SMOOTH_MPC fixes the void smoothing in Mpc: config's smoothing differs from the production one under
    it."""
    probe = "import config; print(config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE)"

    def run(**env):
        e = {k: v for k, v in os.environ.items() if k != "DTFE_SMOOTH_MPC"}
        e.update(env, DTFE_SIM=SIM)
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(PY), env=e,
                           timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
        return r.stdout.strip()
    plain, new = run(), run(DTFE_SMOOTH_MPC="5.0")
    assert new != plain, (plain, new)


def main():
    print("python CLI option names")
    for script in CURRENT:
        names = " ".join(dict.fromkeys(a for argv, _ in CURRENT[script] for a in argv if a.startswith("--")))
        check(f"{script}: {names} parse to the values given", lambda s=script: t_current(s))
    check("make_image_plane.py: --plane-window U0 U1 V0 V1 is the plane's u0, u1, v0, v1 (--same-as)",
          t_plane_window_spread)
    check("no abbreviation is guessed (allow_abbrev=False)", t_no_abbreviation)
    check("DTFE_SMOOTH_MPC fixes the void smoothing in Mpc", t_smooth_mpc_env)
    return summary(blank_line=True, show_zero_skips=False)


if __name__ == "__main__":
    sys.exit(main())
