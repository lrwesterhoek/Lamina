"""The voids of one snapshot's grids as JSON, for the launcher's Explore map ('voids': the outlines over the slice).

Runs the analysis pipeline's own catalogue (pipeline.SnapshotProducts.voids(): the smoothing, minima, Hessian,
ellipsoid cuts and R_v of the void scripts; cached, so only the first call per snapshot costs a minute) in config's
frame ADOPTED from the grid itself (pipeline.adopt_snapshot_grid: its header's box over its n, the smoothing kept
in Mpc), so any full-box grid set of a simulation works, at any resolution. Run it with DTFE_DATA_ROOT set to the
data root; --sim names the simulation (config reads the box for it; DTFE_SIM is its default):

    python3 tools/void_outlines.py --sim TNG100-3-Dark --snap 99 --estimator ps

With the production grids (no --prefix) the catalogue is the analysis scripts' own cache file. A BUILD costs
about 80 bytes per cell (10 GB at 512^3, 80 GB at 1024^3): when the catalogue is not cached and that would not fit
in what the other programs leave free (less 4 GB), the catalogue is found on the grid averaged in 2^3 or 4^3 blocks
at the same smoothing in Mpc (coarsening: 'coarsened' in the JSON, its own cache name), and only when that does not fit
either does the tool refuse instead of starting it (DTFE_FREE_GB overrides the free-memory reading, as for the C++
programs).

Prints ONE line of JSON on stdout (the pipeline's own messages go to stderr): the centres (cell centres in Mpc,
the map's frame), R_v and R_eff in Mpc (null where there is none), 'resolved' (the ellipsoid cuts), 'distinct'
(catalog.distinct_mask: the resolved voids with an R_v whose centre lies inside the sphere of min(R_v, R_eff) of no
deeper resolved void, one without R_v claiming its R_eff; the sample of the profiles and the abundance), the
central delta, 'deep' (the central delta below config.DEEP_VOID_THRESHOLD: the sample the Explore map draws as
ellipses, as plot/plot_ellipses_contour_3D.py does), each resolved void's fitted ellipsoid ('semi_axes_mpc'
a >= b >= c; null where not resolved), the Hessian at every minimum ('eigenvalues_cells', signed, in cell units, and
'ellipsoid_axes', its unit eigenvectors as rows, both in the semi-axes' order), and the frame and smoothing they were
found at. Exit 1 with the
reason on stderr when there is no catalogue (no grids, no snapshot header).

--smooth-mpc S finds the voids at a smoothing of S Mpc instead of the production one (the same physical scale in
every box: the cross-box comparison), cached under that smoothing's own namespace; 'r_v_defs' carries every R_v
definition the catalogue has (catalog.R_V_DEFINITIONS once it exists: 'crossing', 'delta08', 'ridge'). --csv PATH
also writes the table of the voids (write_csv: centre, the R_v's, R_eff, delta, the flags), as the launcher's
Explore map does for the voids it shows.
"""

import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/ on sys.path


BUILD_BYTES_PER_CELL = 80          # a catalogue build: smoothing FFTs, the Hessian cubes, delta (10 GB at 512^3)
HEADROOM_BYTES = 4e9


def available_bytes():
    """What the other programs leave free now (bytes), or None when it cannot be read: DTFE_FREE_GB, else psutil,
    else macOS vm_stat (free + inactive + speculative + purgeable pages), else Linux MemAvailable."""
    if os.environ.get("DTFE_FREE_GB"):
        return float(os.environ["DTFE_FREE_GB"]) * 1e9
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        import re
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            page = int(re.search(r"page size of (\d+) bytes", out.stdout).group(1))
            pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages ([\w ]+):\s+(\d+)\.", out.stdout, re.M)}
            return float(page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")))
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        pass
    return None


_R_V_KEYS = ("r_v_settings", "r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells")   # what catalog_r_v_ok reads


def r_v_stale(path) -> str | None:
    """Why voids() would re-measure the radii of the cached catalogue at 'path' (the pipeline's own test,
    pipeline.catalog_r_v_ok: the current R_v settings and all three radii), None when it reads them as they are. Only
    the four keys are loaded (an npz loads per key), so the Runs tab's status stays cheap; testing for
    'r_v_cells' alone would call a cache with older R_v settings, or none recorded, 'cached'."""
    import numpy as np
    from dtfelib import pipeline
    with np.load(path) as z:
        cat = {k: z[k] for k in _R_V_KEYS if k in z.files}
    if pipeline.catalog_r_v_ok(cat):
        return None
    if "r_v_cells" not in cat:
        return "built before the measured radius R_v"
    if "r_v_settings" not in cat:
        return "R_v measured before its settings were recorded (re-measured on next read)"
    return "R_v measured with older settings (re-measured on next read)"


# a catalogue that would not fit in memory at the grid's own resolution is found on the grid averaged in blocks of
# 2 or 4 cells a side (coarsening), the smoothing kept in Mpc, while the smoothing spans at least this many of the
# coarser cells: a 1024^3 set at the production 20 cells (86 GB) on its 512^3 average at 10 cells (11 GB), the same
# field to the averaging's sigma^2 of a quarter of a fine cell against the smoothing's 400
COARSE_MIN_SIGMA_CELLS = 4.0


def coarsening(n: int, sigma_cells: float, free) -> int | None:
    """The block size a catalogue build of an n^3 grid smoothed over sigma_cells cells takes: 1 when it fits in
    'free' bytes (less HEADROOM_BYTES; None = not known: fits), else the smallest of 2 and 4 that fits and keeps
    the smoothing over COARSE_MIN_SIGMA_CELLS coarser cells and n divisible; None when none does."""
    for k in (1, 2, 4):
        if k > 1 and (n % k or sigma_cells / k < COARSE_MIN_SIGMA_CELLS):
            break
        if free is None or BUILD_BYTES_PER_CELL * (float(n) / k) ** 3 <= free - HEADROOM_BYTES:
            return k
    return None


def coarse_cached(snap, sim, method, prefix, data_root, k: int) -> bool:
    """Whether the catalogue of the grid averaged k x k x k (pipeline.products(coarsen=k)) is cached and usable as it
    is: present, of this grid, its R_v and settings today's (needs_build says no). Leaves config's frame as it was."""
    from dtfelib import pipeline
    saved = (pipeline.config.FIELD_RESOLUTION, pipeline.config.BOX_SIZE, pipeline.config.CELL_SIZE,
             pipeline.config.SMOOTHING_SIGMA_CELLS, pipeline.config.FOOTPRINT_SIZE, pipeline.config.SMOOTHING_SIGMA_MPC)
    prod = pipeline.products(int(snap), sim=sim, method=method, prefix=prefix, data_root=data_root, coarsen=k)
    try:
        pipeline.adopt_snapshot_grid(prod)
        return not needs_build(prod)
    finally:
        prod.release()
        (pipeline.config.FIELD_RESOLUTION, pipeline.config.BOX_SIZE, pipeline.config.CELL_SIZE,
         pipeline.config.SMOOTHING_SIGMA_CELLS, pipeline.config.FOOTPRINT_SIZE, pipeline.config.SMOOTHING_SIGMA_MPC) = saved


def needs_build(prod) -> bool:
    """voids() would BUILD the catalogue (none cached for this grid) or re-measure its radii from the smoothed field
    (r_v_stale), not just read it: either reads the density grid and smooths it, so the memory gate applies."""
    p = prod._cached("voids")
    return p is None or r_v_stale(p) is not None


def _finite_or_none(x):
    return [float(v) if v == v else None for v in x]      # NaN -> null: JSON has no NaN


def r_v_definitions(cat) -> dict:
    """{definition: R_v in Mpc} for every R_v definition the catalogue HAS ('crossing' always, the first up-crossing of
    the mean; 'delta08' and 'ridge' only once catalog.R_V_DEFINITIONS exists and the cache holds them):
    a definition that is NaN everywhere (the key absent from an older cache) is left out, not shown as 'none'."""
    import numpy as np
    from dtfelib import catalog
    defs = getattr(catalog, "R_V_DEFINITIONS", ("crossing",))
    out = {}
    for d in defs:
        try:
            r = catalog.r_v_mpc(cat) if d == "crossing" else catalog.r_v_mpc(cat, definition=d)
        except (TypeError, KeyError, ValueError):
            continue
        r = np.asarray(r, dtype=np.float64)
        if d == "crossing" or np.isfinite(r).any():
            out[d] = r
    return out


CSV_HEAD = ("void", "x_mpc", "y_mpc", "z_mpc")


def write_csv(res: dict, path, rows=None, extra=None, extra_note: str = "") -> str:
    """The voids of an outlines() result as a CSV table (one row per void; 'rows' = the catalogue rows to write, all
    by default): the centre (cell centre, Mpc), R_v (the 'crossing' definition) and every other R_v definition the
    result has, R_eff, the central delta, resolved and distinct (0/1), then 'extra' columns ({name: values per
    catalogue row}: the launcher's drawn ellipsoid, said by 'extra_note' in the '#' line). Empty fields where a value
    is undefined. A '#' line first says the snapshot, the frame and the smoothing, so a quoted number can be traced
    back."""
    import csv
    path = Path(path)
    n = len(res["centres_mpc"])
    rows = range(n) if rows is None else [int(k) for k in rows]
    defs = {d: v for d, v in (res.get("r_v_defs") or {}).items() if d != "crossing"}
    extra = dict(extra or {})
    head = list(CSV_HEAD) + ["r_v_mpc"] + [f"r_v_{d}_mpc" for d in defs] + ["r_eff_mpc", "delta", "resolved",
                                                                            "distinct"] + list(extra)

    def num(v):
        return "" if v is None or v != v else f"{float(v):.6g}"
    with open(path, "w", newline="") as f:
        f.write(f"# {res.get('sim', '')} snapshot {res.get('snap', '')} {res.get('method', '')}"
                f"{(' prefix ' + res['prefix']) if res.get('prefix') else ''}: box {num(res.get('box_mpc'))} Mpc, "
                f"{res.get('n_grid', '')}^3 cells, sigma {num(res.get('sigma_mpc'))} Mpc; centres = cell centres; "
                f"R_v = first up-crossing of the mean, R_eff = (a b c)^(1/3) of the catalogue's resolved ellipsoid"
                f"{('; ' + extra_note) if extra_note else ''}\n")
        w = csv.writer(f)
        w.writerow(head)
        for k in rows:
            c = res["centres_mpc"][k]
            w.writerow([k] + [num(v) for v in c] + [num(res["r_v_mpc"][k])] + [num(v[k]) for v in defs.values()]
                       + [num(res["r_eff_mpc"][k]), num(res["delta"][k]), int(bool(res["resolved"][k])),
                          int(bool(res["distinct"][k]))] + [num(v[k]) for v in extra.values()])
    return str(path)


def name_simulation(sim: str) -> None:
    """Make 'sim' the simulation config reads the box for. config reads DTFE_SIM once, when it is first imported
    (config.SIMULATION), so DTFE_SIM is set to 'sim' for this process before that; a process that imported config
    already must have it for this simulation, else SystemExit naming DTFE_SIM (void_cache.py build: the same rule)."""
    cfg = sys.modules.get("config")
    if cfg is None:
        os.environ["DTFE_SIM"] = sim
    elif cfg.SIMULATION != sim:
        how = f"imported for {cfg.SIMULATION}" if cfg.SIMULATION else "imported with no DTFE_SIM"
        raise SystemExit(f"run with DTFE_SIM={sim}: config reads the box for the simulation it names (it was {how})")


def outlines(sim, snap, prefix=None, data_root=None, method="auto", sigma_mpc=None) -> dict:
    """The voids of one snapshot's grids (the module docstring); sigma_mpc > 0 = that smoothing in Mpc (set BEFORE
    the frame is adopted, which keeps it in Mpc), None or 0 = the production one."""
    import numpy as np
    from dtfelib import pipeline, catalog
    if sigma_mpc:
        pipeline.set_smoothing(mpc=float(sigma_mpc))
    prod = pipeline.products(int(snap), sim=sim, method=method, prefix=prefix, data_root=data_root)
    k = 1
    try:
        cell, box, n = pipeline.adopt_snapshot_grid(prod)
        if needs_build(prod):
            free = available_bytes()
            sigma_cells = float(pipeline.config.SMOOTHING_SIGMA_CELLS)
            # an averaged catalogue already cached is taken first, whatever the free memory says today (a tighter
            # day would otherwise build another at 4x beside the cached 2x)
            k = next((c for c in (2, 4) if n % c == 0 and sigma_cells / c >= COARSE_MIN_SIGMA_CELLS
                      and coarse_cached(snap, sim, method, prefix, data_root, c)), None) \
                or coarsening(n, sigma_cells, free)
            if k is None:
                need = BUILD_BYTES_PER_CELL * float(n) ** 3
                raise MemoryError(f"its catalogue is not cached (or its R_v is re-measured), and building it ({n}^3) "
                                  f"needs about {need / 1e9:.0f} GB "
                                  f"with {free / 1e9:.0f} GB free: build it with the analysis scripts when the machine is "
                                  f"free (analyze.py compute), then tick again")
            if k > 1:                           # the grid averaged k x k x k: the same smoothing in Mpc, own cache
                prod.release()
                prod = pipeline.products(int(snap), sim=sim, method=method, prefix=prefix, data_root=data_root,
                                         coarsen=k)
                cell, box, n = pipeline.adopt_snapshot_grid(prod)
        cat = prod.voids()
        method = str(prod.fs.method)
    finally:
        prod.release()
    resolved = np.asarray(cat["well_resolved"]).astype(bool)
    r_v, r_eff = catalog.r_v_mpc(cat), catalog.r_eff_mpc(cat)
    defs = r_v_definitions(cat)
    distinct = catalog.distinct_mask(cat, resolved & np.isfinite(r_v), n)
    centres = (np.asarray(cat["coords"], dtype=np.float64) + 0.5) * cell
    # the fitted ellipsoid of each resolved void (R_eff = (a b c)^(1/3) of it): its semi-axes a >= b >= c in Mpc, null
    # where not resolved
    semi = np.asarray(cat["semi_axes"], dtype=np.float64).reshape(-1, 3)
    semi_out = [s.round(6).tolist() if ok else None for s, ok in zip(semi, resolved)]
    # the Hessian at EVERY minimum (the catalogue's, in cell units: d^2 delta / d cell^2): its eigenvalues signed, and
    # its unit eigenvectors as rows, both in the order of the semi-axes above (|lambda| ascending:
    # pipeline._ellipsoid_fits' argsort; for a resolved void the rows are its orientation columns). The launcher draws
    # each minimum's ellipsoid where the quadratic fit reaches half its depth from them, resolved or not
    ev = np.asarray(cat["eigenvalues"], dtype=np.float64).reshape(-1, 3)
    order = np.argsort(np.abs(ev) + 1e-12, axis=1)
    ev = np.take_along_axis(ev, order, axis=1)
    vec = np.take_along_axis(np.asarray(cat["eigenvectors"], dtype=np.float64).reshape(-1, 3, 3), order[:, None, :],
                             axis=2)
    ev_out = [[float(v) for v in e] if np.isfinite(e).all() else None for e in ev]
    dirs_out = [d.T.round(6).tolist() if np.isfinite(d).all() else None for d in vec]
    return {"coarsened": int(k),
            "sim": sim, "snap": int(snap), "prefix": prefix or "", "method": method, "box_mpc": box, "cell_mpc": cell,
            "n_grid": n, "sigma_mpc": pipeline.smoothing_length_mpc(), "footprint": int(pipeline.config.FOOTPRINT_SIZE),
            "namespace": pipeline._param_hash(), "centres_mpc": centres.round(6).tolist(),
            "r_v_mpc": _finite_or_none(r_v), "r_eff_mpc": _finite_or_none(r_eff),
            "resolved": resolved.tolist(), "distinct": distinct.tolist(),
            "deep": np.asarray(cat["deep"]).astype(bool).tolist(),
            "delta": _finite_or_none(np.asarray(cat["delta_values"], dtype=np.float64)),
            "sigma_asked_mpc": float(sigma_mpc) if sigma_mpc else 0.0,
            "r_v_defs": {d: _finite_or_none(v) for d, v in defs.items()},
            "semi_axes_mpc": semi_out, "ellipsoid_axes": dirs_out, "eigenvalues_cells": ev_out}


def main(argv=None):
    from dtfelib.cli import add_sim_arg, new_parser, add_estimator_arg
    p = new_parser(description=__doc__.split("\n\n")[0])
    add_sim_arg(p)                                  # required unless DTFE_SIM names it
    p.add_argument("--snap", type=int, required=True)
    add_estimator_arg(p, help="the estimator of the grids")
    p.add_argument("--prefix", default=None, help="the grid set's output prefix (default the method's)")
    p.add_argument("--data-root", type=Path, default=None, help="the data root (default DTFE_DATA_ROOT's)")
    p.add_argument("--cache-dir", type=Path, default=None, help="the catalogue cache (default config.CACHE_DIR)")
    p.add_argument("--smooth-mpc", type=float, default=0.0, metavar="MPC",
                   help="the smoothing in Mpc (default 0: the production smoothing and its catalogue)")
    p.add_argument("--csv", type=Path, default=None, help="also write the table of all voids (write_csv) here")
    a = p.parse_args(argv)
    name_simulation(a.sim)
    import zipfile
    try:
        # importing config reads the box of --sim's simulation: an unknown one (KeyError) or no data root to look
        # it up under (NoDataRoot) is the same one-line exit as the missing grids
        import config
        if a.cache_dir is not None:
            config.CACHE_DIR = a.cache_dir
        with contextlib.redirect_stdout(sys.stderr):          # the pipeline's progress lines: not on the JSON line
            res = outlines(a.sim, a.snap, a.prefix, a.data_root, a.method, a.smooth_mpc)
    # a damaged cache member (pipeline._cached reads only the stamp) is a reason, not a traceback
    except (OSError, ValueError, KeyError, MemoryError, EOFError, zipfile.BadZipFile) as e:
        raise SystemExit(f"no void catalogue for {a.sim} snapshot {a.snap}: {e}")
    if a.csv is not None:
        write_csv(res, a.csv)
    sys.stdout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
