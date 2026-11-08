#!/usr/bin/env python3
"""The python CLIs' renamed options (P6.7d): every old spelling gives the same parsed arguments as the new one, with
its warnings on stderr ('warning: <old> is deprecated, use <new>', never on stdout) and none for the new spelling;
no option is guessed from an abbreviation (allow_abbrev=False); the smoothing variable DTFE_SMOOTH_MPC and its older
name DTFE_SIGMA_MPC.

Each script runs in a fresh interpreter, its parser stopped right after parse_args (argparse patched to raise with
the Namespace): nothing is read or written, no data is needed.

Usage:  python3 tests/py_cli_alias_test.py            (exit 0 = pass)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _harness import REPO, check, summary

PY = REPO / "python"
SIM = "TNG100-3-Dark"

# the driver run in each child: argparse.ArgumentParser.parse_args raises with the Namespace, so main() stops there;
# each case runs the script afresh (runpy, as __main__) with dtfelib.cli's warned-once set cleared
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
    import dtfelib.cli as cli
    cli._WARNED.clear()
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
        for name in ("DTFE_SMOOTH_MPC", "DTFE_SIGMA_MPC"):
            env.pop(name, None)
        r = subprocess.run([sys.executable, "-c", DRIVER, str(PY / script), json.dumps([list(c) for c in cases])],
                           capture_output=True, text=True, env=env, cwd=str(PY), timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("@@")), None)
    assert line, f"{script}: no result ({r.returncode}): {r.stderr[-800:]}"
    return json.loads(line[2:])


def warnings(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.startswith("warning: ") and " is deprecated, use " in l]


def same(script: str, old: list, new: list, olds: tuple, normalize=None):
    """old and new parse to the same Namespace (after 'normalize', for a script that merges an option after
    parse_args); old warns once for each name in 'olds' (and on stderr only), new not at all."""
    o, n = parse(script, old, new)
    assert o["ns"] is not None and n["ns"] is not None, f"{script}: not parsed: {o} {n}"
    on, nn = (normalize(dict(o["ns"])), normalize(dict(n["ns"]))) if normalize else (o["ns"], n["ns"])
    assert on == nn, f"{script}: {on} != {nn}"
    w = warnings(o["stderr"])
    assert not warnings(n["stderr"]), f"{script}: the new spelling warned: {n['stderr']}"
    assert "deprecated" not in o["stdout"], f"{script}: a warning on stdout: {o['stdout']}"
    missing = [x for x in olds if not any(f"warning: {x} is deprecated" in l for l in w)]
    assert not missing and len(w) == len(set(w)), f"{script}: warnings {w}, missing {missing}"


def t_make_parser():
    # every make_parser script: --method -> --estimator (dest 'method', FieldSet's keyword)
    same("plot/plot_DTFE.py", ["--sim", SIM, "--method", "ps"], ["--sim", SIM, "--estimator", "ps"], ("--method",))


def t_tracking():
    same("plot/plot_halo_tracking.py",
         ["--sim", SIM, "--snap", "50", "--subhalo", "1", "2", "--top", "3", "--method", "dtfe"],
         ["--sim", SIM, "--start-snap", "50", "--subhalos", "1", "2", "--n-halos", "3", "--estimator", "dtfe"],
         ("--snap", "--subhalo", "--top", "--method"))
    same("plot/plot_halo_tracking.py", ["--sim", SIM, "--top", "7"], ["--sim", SIM, "--n-halos", "7"], ("--top",))
    same("plot/plot_void_tracking.py", ["--sim", SIM, "--snap", "50"], ["--sim", SIM, "--start-snap", "50"],
         ("--snap",))


def t_void_scripts():
    same("plot/plot_pk_compare.py", ["--sim", SIM, "--nbins", "12", "--output", "/x", "--snaps", "1,2"],
         ["--sim", SIM, "--n-bins", "12", "--out", "/x", "--snaps", "1", "2"],
         ("--nbins", "--output", "a comma-separated --snaps"))
    same("plot/plot_void_population.py", ["--sim", SIM, "--size-rv", "ridge"],
         ["--sim", SIM, "--size-rv-definition", "ridge"], ("--size-rv",))
    same("plot/plot_caustic_skeleton.py", ["--sim", SIM, "--voids"], ["--sim", SIM, "--per-void"], ("--voids",))
    o, n = parse("plot/plot_void_environment.py",
                 ["--sim", SIM, "--criterion", "density_raw", "--r-v-definition", "delta08"],
                 ["--sim", SIM, "--criterion", "density-raw", "--rv-definition", "delta08"])
    assert o["ns"] == n["ns"] and n["ns"]["criterion"] == "density_raw" and n["ns"]["r_v_definition"] == "delta08", \
        (o, n)      # the internal name stays (npz keys, figure tags)
    w = warnings(o["stderr"])
    assert any("--criterion density_raw is deprecated, use --criterion density-raw" in l for l in w) \
        and any("--r-v-definition is deprecated" in l for l in w) and not warnings(n["stderr"]), (w, n["stderr"])
    same("plot/plot_void_profile_evolution.py", ["--sim", SIM, "--rv-range", "2,4", "--edges", "0,10,20"],
         ["--sim", SIM, "--rv-range", "2", "4", "--edges", "0", "10", "20"],
         ("a comma-separated --rv-range", "a comma-separated --edges"))
    same("plot/plot_void_profiles.py", ["--sim", SIM, "--edges", "0,4,30"], ["--sim", SIM, "--edges", "0", "4", "30"],
         ("a comma-separated --edges",))
    same("plot/plot_void_convergence.py", ["--sim", SIM, "--z", "0", "1"], ["--sim", SIM, "--redshifts", "0", "1"],
         ("--z",))
    same("plot/plot_void_crossbox.py", ["--sims", "A", "B", "--method", "dtfe", "--z", "0", "1"],
         ["--sims", "A", "B", "--estimator", "dtfe", "--redshifts", "0", "1"], ("--method", "--z"))
    same("plot/plot_void_crossbox.py",
         ["--worker", "/x.npz", "--sim", "A", "--method", "dtfe", "--smooth-mpc", "2", "--z", "0"],
         ["--worker", "/x.npz", "--sim", "A", "--estimator", "dtfe", "--smooth-mpc", "2", "--redshifts", "0"],
         ("--method", "--z"))


def t_pointeval():
    same("plot/plot_pointeval.py",
         ["--sims", "A,B", "--snaps", "4,99", "--fields", "density,velDiv", "--method", "dtfe", "--smooth", "2",
          "--smooth-derivatives", "0"],
         ["--sims", "A", "B", "--snaps", "4", "99", "--maps", "density", "velDiv", "--estimator", "dtfe",
          "--smooth-px", "2", "--smooth-derivatives-px", "0"],
         ("a comma-separated --sims", "a comma-separated --snaps", "--fields", "--method", "--smooth",
          "--smooth-derivatives"))
    # the old --prefix alone satisfies the required --root
    same("plot/plot_pointeval_panels.py",
         ["--prefix", "/r", "--plane", "/p.json", "--output", "/o.png", "--panels", "density,sheet", "--project", "mean"],
         ["--root", "/r", "--plane", "/p.json", "--out", "/o.png", "--panels", "density", "sheet", "--project", "slab"],
         ("--prefix", "--output", "a comma-separated --panels", "--project mean"))
    r = parse("plot/plot_pointeval_panels.py", ["--plane", "/p.json"], ["--root", "/r", "--prefix", "/r", "--plane", "/p"])
    assert r[0]["code"] == 2 and "--root" in r[0]["stderr"] and "--prefix" not in r[0]["stderr"].split("required")[-1], r[0]
    assert r[1]["code"] == 2 and "not allowed with" in r[1]["stderr"], r[1]     # the two spellings at once


def t_tools():
    same("tools/void_outlines.py", ["--sim", SIM, "--snap", "99", "--method", "ps", "--sigma-mpc", "2"],
         ["--sim", SIM, "--snap", "99", "--estimator", "ps", "--smooth-mpc", "2"], ("--method", "--sigma-mpc"))
    same("tools/compare_deposits.py", ["--sim", SIM, "--snap", "99", "--a", "x", "--b", "y"],
         ["--sim", SIM, "--snap", "99", "--prefix-a", "x", "--prefix-b", "y"], ("--a", "--b"))
    same("gui/void_cache.py", ["build", "--sim", SIM, "--snap", "99", "--method", "dtfe"],
         ["build", "--sim", SIM, "--snap", "99", "--estimator", "dtfe"], ("--method",))

    def merged(ns):         # main() merges the positional snapshot numbers of before into --snaps
        ns["snaps"] = (ns.get("snaps") or []) + ns.pop("old_snapshots")
        return ns
    same("tools/merge_HDF5.py", ["--data-dir", "/d", "8", "17"], ["--sim-dir", "/d", "--snaps", "8", "17"],
         ("--data-dir",), normalize=merged)

    def window(ns):         # main() spreads --plane-window over the four of before
        if ns.get("plane_window"):
            ns.update(zip(("u0", "u1", "v0", "v1"), ns["plane_window"]))
        ns.pop("plane_window", None)
        return ns
    same("tools/make_image_plane.py",
         ["--same-as", "x.json", "--u0", "0", "--u1", "50", "--v0", "0", "--v1", "25", "--output", "s"],
         ["--same-as", "x.json", "--plane-window", "0", "50", "0", "25", "--out", "s"],
         ("--u0", "--u1", "--v0", "--v1", "--output"), normalize=window)

    def snaps(ns):
        ns["snaps"] = list(ns.get("snaps") or []) + ns.pop("old_snaps")
        return ns
    same("analyze.py", ["export", "099", "050", "--output", "/o", "--sim", SIM],
         ["export", "--snaps", "099", "050", "--out", "/o", "--sim", SIM], ("--output",), normalize=snaps)


def t_positional_warnings():
    """The positional snapshots of before (merge_HDF5.py, analyze.py) warn once, after parse_args."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, str(PY / "tools" / "merge_HDF5.py"), "-d", str(Path(tmp) / "none"),
                            "--trees", "5"], capture_output=True, text=True, timeout=120)
    assert "warning: a positional snapshot number is deprecated, use --snaps 5" in r.stderr, r.stderr[-500:]
    assert "deprecated" not in r.stdout, r.stdout[-300:]


def t_no_abbreviation():
    """allow_abbrev=False everywhere: a prefix of an option is an error, not a guess (an old name that is a prefix
    of a new one -- --size-rv, --smooth, --subhalo -- is an exact match of its alias, never a guess)."""
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
    """DTFE_SMOOTH_MPC fixes the void smoothing in Mpc; DTFE_SIGMA_MPC, its older name, the same with one warning;
    both set: the new one wins and the old one is said not to be read."""
    probe = "import config; print(config.SMOOTHING_SIGMA_CELLS, config.FOOTPRINT_SIZE)"

    def run(**env):
        e = {k: v for k, v in os.environ.items() if k not in ("DTFE_SMOOTH_MPC", "DTFE_SIGMA_MPC")}
        e.update(env, DTFE_SIM=SIM)
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(PY), env=e,
                           timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
        return r.stdout.strip(), warnings(r.stderr)
    plain, new, old, both = run(), run(DTFE_SMOOTH_MPC="5.0"), run(DTFE_SIGMA_MPC="5.0"), \
        run(DTFE_SMOOTH_MPC="5.0", DTFE_SIGMA_MPC="3.0")
    assert new[0] == old[0] == both[0] != plain[0], (plain, new, old, both)
    assert not new[1] and not plain[1], (plain, new)
    assert old[1] == ["warning: DTFE_SIGMA_MPC is deprecated, use DTFE_SMOOTH_MPC"], old
    assert len(both[1]) == 1 and "DTFE_SIGMA_MPC=3.0 is not read: DTFE_SMOOTH_MPC=5.0 wins" in both[1][0], both
    sys.path.insert(0, str(PY))
    from dtfelib.cli import SMOOTH_MPC_ENV_NAMES
    assert SMOOTH_MPC_ENV_NAMES == ("DTFE_SMOOTH_MPC", "DTFE_SIGMA_MPC"), SMOOTH_MPC_ENV_NAMES


def main():
    print("python CLI spellings")
    check("make_parser: --method is --estimator (args.method), one warning", t_make_parser)
    check("tracking: --snap -> --start-snap, --subhalo -> --subhalos, --top -> --n-halos", t_tracking)
    check("void scripts: --nbins, --output, comma lists, --size-rv, --voids, --criterion's values, --r-v-definition, "
          "--z", t_void_scripts)
    check("plot_pointeval(_panels): --fields -> --maps, --smooth(-derivatives) -> -px, --prefix -> --root (a required "
          "pair), --project mean -> slab, comma lists", t_pointeval)
    check("tools and the launcher's helpers: --sigma-mpc, --a/--b, --data-dir, --u0..--v1 -> --plane-window, "
          "analyze.py's --snaps", t_tools)
    check("the positional snapshots of before warn once", t_positional_warnings)
    check("no abbreviation is guessed (allow_abbrev=False)", t_no_abbreviation)
    check("DTFE_SMOOTH_MPC, and DTFE_SIGMA_MPC with one warning (the new name wins)", t_smooth_mpc_env)
    return summary(blank_line=True, show_zero_skips=False)


if __name__ == "__main__":
    sys.exit(main())
