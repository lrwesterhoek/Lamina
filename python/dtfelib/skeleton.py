"""The caustic skeleton: walls, filaments and nodes from the 'caustic_class' bitmask of a '--ps-caustics' run, as
periodic connected components with their volume and mass fractions, and per void how much caustic sits on its edge
and how far the nearest wall is. Components are built from the cumulative masks (sheet k >= 1, line k >= 2, node k = 3)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ._threads import THREAD_MIN, THREADS, map_chunks
from .io import (CAUSTIC_DEGENERATE, CAUSTIC_PARITY_POS, caustic_collapse_max, caustic_is_cusp,
                 caustic_is_fold, caustic_is_swallowtail)

CLASSES = ("none", "k=0", "k=1", "k=2", "k=3")                 # exclusive, = webstreams.COLLAPSE_BINS
INDICATORS = ("fold", "cusp", "swallowtail", "umbilic")        # fractions only
CUMULATIVE = ("sheet", "line", "node")                         # k >= 1, k >= 2, k = 3: the components


def class_masks(caustic_class, fold=None) -> dict:
    """{'exclusive': {name: bool mask} over CLASSES, 'indicator': over INDICATORS, 'cumulative': over CUMULATIVE}
    of a caustic_class grid. 'fold' (bool grid) replaces the mask's own fold bits: FieldSet.fold_mask() is the
    geometric flag for any grid (the default is exact only for grids of the current binary)."""
    cc = np.rint(np.asarray(caustic_class)).astype(np.int64)
    kmax = caustic_collapse_max(cc)
    # every deposited SAMPLE sets a parity bit TOGETHER with one collapse bit; a cell with the det>0 parity bit and
    # no collapse bit comes from a parity-only grid, which would read as 99.96% 'none'. Bit 0 only: the det<0 bit
    # is the geometric overlap, legitimately alone in a cell no sample reached; a parity-only grid carries bit 0
    # in every deposited cell, so it is still refused.
    if np.any(((cc & CAUSTIC_PARITY_POS) != 0) & (kmax == -1)):
        raise ValueError("parity-only caustic_class grid (fold bits without collapse bits: an older binary's "
                         "--ps-caustics --ps-linear-deposit --gpu run); the stratification needs the CPU or the GPU's "
                         "full deposit")
    if fold is None:
        fold = caustic_is_fold(cc)
    else:
        fold = np.asarray(fold, dtype=bool)
        if fold.shape != cc.shape:
            raise ValueError(f"fold mask of shape {fold.shape} for a caustic_class grid of shape {cc.shape}")
    return {
        "exclusive": {"none": kmax == -1, "k=0": kmax == 0, "k=1": kmax == 1, "k=2": kmax == 2, "k=3": kmax == 3},
        "indicator": {"fold": fold, "cusp": caustic_is_cusp(cc), "swallowtail": caustic_is_swallowtail(cc),
                      "umbilic": (cc & CAUSTIC_DEGENERATE) != 0},
        "cumulative": {"sheet": kmax >= 1, "line": kmax >= 2, "node": kmax == 3},
    }


def _fraction(mask, density=None) -> dict:
    n = int(mask.size)
    vol = float(mask.sum()) / n if n else float("nan")
    if density is None:
        return {"volume": vol, "mass": None}
    d = np.asarray(density, dtype=np.float64)
    tot = float(d.sum())
    return {"volume": vol, "mass": float(d[mask].sum()) / tot if tot > 0 else None}


def fractions(caustic_class, density=None, fold=None) -> dict:
    """Volume (and mass, with a density grid) fractions: {'exclusive': {class: {'volume', 'mass'}},
    'indicator': {...}, 'cumulative': {...}}; the exclusive volumes sum to 1. 'fold': see class_masks."""
    m = class_masks(caustic_class, fold)
    return {group: {name: _fraction(mask, density) for name, mask in masks.items()} for group, masks in m.items()}


def label_periodic(mask, connectivity: int = 26) -> tuple[np.ndarray, int]:
    """(labels, n): the connected components of the boolean mask in a PERIODIC box, labelled 1..n in
    order of size (largest first), 0 outside the mask; 'connectivity' 6, 18 or 26."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("label_periodic wants a 3-D mask")
    rank = {6: 1, 18: 2, 26: 3}.get(int(connectivity))
    if rank is None:
        raise ValueError("connectivity must be 6, 18 or 26")
    structure = ndimage.generate_binary_structure(3, rank)
    padded = np.pad(mask, 1, mode="wrap")
    lab, n = ndimage.label(padded, structure=structure)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.int32), 0
    a_list, b_list = [], []
    for axis in range(3):
        size = mask.shape[axis]
        for halo, src in ((0, size), (size + 1, 1)):          # the padded halo cell and the interior cell it copies
            la = np.take(lab, halo, axis=axis).ravel()
            lb = np.take(lab, src, axis=axis).ravel()
            both = (la > 0) & (lb > 0) & (la != lb)
            a_list.append(la[both])
            b_list.append(lb[both])
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n + 1, n + 1))
    _ncomp, comp = connected_components(graph, directed=False)
    inner = lab[1:-1, 1:-1, 1:-1]
    merged = np.empty(mask.shape, dtype=comp.dtype)          # component id + 1 per cell, 0 outside the mask
    slabs = 0 if mask.size >= THREAD_MIN else mask.shape[0] + 1  # map_chunks' min_size, in slabs along axis 0

    def ids(lo, hi):
        m = inner[lo:hi]
        merged[lo:hi] = np.where(m > 0, comp[m] + 1, 0)
        return np.bincount(merged[lo:hi].ravel(), minlength=len(comp) + 1)

    full = sum(map_chunks(mask.shape[0], ids, min_size=slabs))   # np.unique by counting: the ids are 0..n
    present = np.flatnonzero(full)
    lut = np.zeros(len(full), dtype=np.intp)
    lut[present] = np.arange(len(present))
    counts = full[present]
    order = [i for i in np.argsort(-counts) if present[i] != 0]
    newid = np.zeros(len(present), dtype=np.int32)
    for rank_, i in enumerate(order, start=1):
        newid[i] = rank_
    out = np.empty(mask.shape, dtype=np.int32)

    def relabel(lo, hi):
        out[lo:hi] = newid[lut[merged[lo:hi]]]

    map_chunks(mask.shape[0], relabel, min_size=slabs)
    return out, len(order)


def components(mask, density=None, cell_mpc: float = 1.0, connectivity: int = 26, top: int = 10) -> dict:
    """The periodic components of a mask: {'n', 'labels', 'cells' (per component, largest first),
    'volume_mpc3', 'mass_fraction' (of the whole box's mass; None without a density), 'largest_fraction'
    (the largest component's share of the mask's cells), 'top': the first 'top' sizes}."""
    labels, n = label_periodic(mask, connectivity)
    cells = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    out = {"n": int(n), "labels": labels, "cells": cells, "volume_mpc3": cells * float(cell_mpc) ** 3,
           "largest_fraction": float(cells[0] / cells.sum()) if n else None, "top": cells[:top].tolist(),
           "mass_fraction": None}
    if density is not None and n:
        d = np.asarray(density, dtype=np.float64)
        tot = float(d.sum())
        mass = np.bincount(labels.ravel(), weights=d.ravel(), minlength=n + 1)[1:]
        out["mass_fraction"] = mass / tot if tot > 0 else None
    return out


def skeleton_summary(caustic_class, density=None, cell_mpc: float = 1.0, connectivity: int = 26, top: int = 10,
                     fold=None) -> dict:
    """fractions() plus, per CUMULATIVE class, its components' count, largest share and top sizes
    (cells and Mpc^3), as a JSON-able dict (no label grids). 'fold': see class_masks."""
    m = class_masks(caustic_class, fold)
    out = {"fractions": fractions(caustic_class, density, fold=fold), "cell_mpc": float(cell_mpc),
           "connectivity": int(connectivity), "components": {},
           # bits 7 and 8 exist only under --ps-caustic-cusps on the CPU deposit: 0% there means 'not computed'
           "cusp_bits_present": bool(m["indicator"]["cusp"].any() or m["indicator"]["swallowtail"].any())}
    for name in CUMULATIVE:
        c = components(m["cumulative"][name], density, cell_mpc, connectivity, top)
        out["components"][name] = {"n": c["n"], "largest_fraction": c["largest_fraction"], "top_cells": c["top"],
                                   "top_volume_mpc3": (np.asarray(c["top"]) * float(cell_mpc) ** 3).tolist(),
                                   "top_mass_fraction": None if c["mass_fraction"] is None else c["mass_fraction"][:top].tolist()}
    return out


def void_shell_fractions(caustic_class, centers_frac, r_eff_cells, shells=(1.0, 1.1, 1.2, 1.3), m: int = 200,
                         masks=None) -> dict:
    """Per void: the fold and sheet fractions of its shells x R_eff (nearest cell, profiles.shell_means), averaged
    over the shells: {'fold', 'sheet', 'shells', 'per_shell'}. 'masks' {'fold', 'sheet'} (bool grids) replaces the
    class masks of 'caustic_class'; without it the fold is the mask's own parity bits (pass FieldSet.fold_mask())."""
    from .profiles import fibonacci_sphere, shell_means
    if masks is None:
        cm = class_masks(caustic_class)
        masks = {"fold": cm["indicator"]["fold"], "sheet": cm["cumulative"]["sheet"]}
    dirs = fibonacci_sphere(m)
    centers = np.asarray(centers_frac, dtype=np.float64).reshape(-1, 3)
    r_eff = np.asarray(r_eff_cells, dtype=np.float64).reshape(-1)
    out = {"shells": np.asarray(shells, dtype=np.float64), "per_shell": {}}
    for name, mask in (("fold", masks["fold"]), ("sheet", masks["sheet"])):
        grid = np.asarray(mask, dtype=np.float32)           # once per mask (a 512^3 bool mask is a 512 MB copy)
        per = np.empty((len(r_eff), len(shells)))
        for j, x in enumerate(shells):
            per[:, j] = shell_means(grid, centers, x * r_eff, dirs, mode="ngp")
        out["per_shell"][name] = per
        out[name] = per.mean(axis=1)
    return out


class WallIndex:
    """A sheet mask (a cubic bool grid, the box periodic) prepared for nearest_wall_distance: the mask and a
    periodic k-d tree of its boundary cells (the nearest mask cell to an outside cell is always one). Build it
    once for many searches (random_sphere_null)."""

    def __init__(self, sheet_mask):
        from scipy.spatial import cKDTree
        self.mask = np.asarray(sheet_mask, dtype=bool)
        if self.mask.ndim != 3 or len(set(self.mask.shape)) != 1:
            raise ValueError(f"the wall search needs a cubic 3D mask, not one of shape {self.mask.shape}")
        self.n = int(self.mask.shape[0])
        inner = self.mask.copy()
        for ax in range(3):
            for step in (1, -1):
                inner &= np.roll(self.mask, step, axis=ax)
        self.cells = np.argwhere(self.mask & ~inner)        # (m, 3) int64
        self.tree = cKDTree(self.cells.astype(np.float64), boxsize=float(self.n)) if len(self.cells) else None


def nearest_wall_distance(sheet_mask, centers_cells, cutout_cells: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(distance, bounded): per centre (cells, (n, 3), rounded to the nearest cell), the periodic Euclidean
    distance in cells to the nearest True cell of 'sheet_mask' (a mask, or its WallIndex); 0 on the mask. With no
    sheet cell within max(1, cutout_cells) the distance is NaN and 'bounded' True (a lower bound)."""
    index = sheet_mask if isinstance(sheet_mask, WallIndex) else WallIndex(sheet_mask)
    centers = np.asarray(centers_cells, dtype=np.float64).reshape(-1, 3)
    bound = max(1, int(cutout_cells))
    dist = np.full(len(centers), np.nan)
    bounded = np.ones(len(centers), dtype=bool)
    if not len(centers):
        return dist, bounded
    n = index.n
    ci = np.mod(np.rint(centers).astype(np.int64), n)
    on = index.mask[ci[:, 0], ci[:, 1], ci[:, 2]]
    dist[on], bounded[on] = 0.0, False
    off = np.flatnonzero(~on)
    if index.tree is None or not off.size:
        return dist, bounded
    # the tree's nearest cell within the bound (its distance only selects: the value is recomputed exactly below)
    d, k = index.tree.query(ci[off].astype(np.float64), k=1, distance_upper_bound=bound + 1.0)
    found = np.isfinite(d)
    off, k = off[found], k[found]
    delta = np.abs(index.cells[k] - ci[off])
    delta = np.minimum(delta, n - delta)                   # the nearest periodic image, per axis
    exact = np.sqrt((delta * delta).sum(axis=1).astype(np.float64))
    near = exact <= bound
    dist[off[near]], bounded[off[near]] = exact[near], False
    return dist, bounded


def filter_components(mask, min_cells: int = 0, density=None, min_mass_fraction: float = 0.0,
                      connectivity: int = 26) -> tuple[np.ndarray, dict]:
    """The mask with only its periodic components of at least 'min_cells' cells AND at least 'min_mass_fraction'
    of the box's mass: (filtered mask, {'n', 'kept', 'cells_removed', 'min_cells', 'min_mass_fraction'}). The
    skeleton's control: a wall distance to a 3-cell flake inside a void says nothing about its walls."""
    mask = np.asarray(mask, dtype=bool)
    info = {"n": None, "kept": None, "cells_removed": 0, "min_cells": int(min_cells),
            "min_mass_fraction": float(min_mass_fraction)}
    if int(min_cells) <= 1 and not float(min_mass_fraction) > 0:
        return mask, info
    if float(min_mass_fraction) > 0 and density is None:
        raise ValueError("a mass threshold needs the density grid")
    c = components(mask, density if float(min_mass_fraction) > 0 else None, connectivity=connectivity, top=0)
    ok = c["cells"] >= int(min_cells)
    if float(min_mass_fraction) > 0 and c["n"]:
        ok &= np.asarray(c["mass_fraction"]) >= float(min_mass_fraction)
    keep = np.concatenate([[False], ok])                     # label 0 is outside the mask
    out = keep[c["labels"]]
    info.update(n=int(c["n"]), kept=int(ok.sum()), cells_removed=int(mask.sum() - out.sum()))
    return out, info


NULL_METRICS = ("fold", "sheet", "dist_r")                    # shell fold / sheet fraction, wall distance / R


def _median(a):
    """The median of the finite entries, NaN without one (no RuntimeWarning: none of numpy's empty-slice messages)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def sphere_metrics(masks, centers_cells, r_cells, n_grid: int, shells, cutout_cells: int, wall=None) -> dict:
    """The per-sphere wall metrics of plot_caustic_skeleton's --per-void for spheres at integer cells
    'centers_cells' (n, 3) of radii 'r_cells' (n,): {'fold', 'sheet' (shell fractions), 'dist_r' (nearest-sheet
    distance / r, NaN beyond the cut-out), 'bounded'}. 'wall': the sheet's WallIndex, else built from masks['sheet']."""
    c = np.asarray(centers_cells, dtype=np.int64).reshape(-1, 3)
    r = np.asarray(r_cells, dtype=np.float64).reshape(-1)
    sh = void_shell_fractions(None, (c + 0.5) / n_grid, r, shells=tuple(shells), masks=masks)
    dist, bounded = nearest_wall_distance(masks["sheet"] if wall is None else wall, c, cutout_cells=cutout_cells)
    return {"fold": sh["fold"], "sheet": sh["sheet"], "dist_r": dist / r, "bounded": bounded}


def random_sphere_null(masks, r_cells, n_grid: int, shells, cutout_cells: int, n_draws: int = 50,
                       seed: int = 0) -> dict:
    """The random-sphere null of the per-void wall metrics: 'n_draws' realisations, each one sphere per void at a
    uniformly random cell carrying THAT void's radius, the same metrics (sphere_metrics) and per realisation their
    median: {metric: (n_draws,), 'bounded_fraction': (n_draws,)}. Seeded: the same call gives the same null."""
    rng = np.random.default_rng(int(seed))
    r = np.asarray(r_cells, dtype=np.float64).reshape(-1)
    out = {k: np.full(int(n_draws), np.nan) for k in NULL_METRICS}
    out["bounded_fraction"] = np.full(int(n_draws), np.nan)
    if not r.size:
        return out
    grids = {name: np.asarray(masks[name], dtype=np.float32) for name in ("fold", "sheet")}
    wall = WallIndex(masks["sheet"])
    centres = [rng.integers(0, int(n_grid), size=(r.size, 3)) for _ in range(int(n_draws))]   # the serial draw order

    def one(i):
        m = sphere_metrics(grids, centres[i], r, n_grid, shells, cutout_cells, wall=wall)
        for k in NULL_METRICS:
            out[k][i] = _median(m[k])
        out["bounded_fraction"][i] = float(np.mean(m["bounded"]))

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(one, range(int(n_draws))))
    return out


def null_zscores(void_metrics: dict, null: dict) -> dict:
    """Per metric: the voids' median, the null's mean and standard deviation over its realisations' medians, and
    z = (void - null mean) / null std ({'void', 'null_mean', 'null_std', 'z', 'draws'}; NaN where undefined: no
    finite value, or a null without spread)."""
    out = {}
    for k in NULL_METRICS:
        v = _median(void_metrics[k])
        nv = np.asarray(null[k], dtype=float)
        nv = nv[np.isfinite(nv)]
        mean = float(nv.mean()) if nv.size else float("nan")
        std = float(nv.std(ddof=1)) if nv.size > 1 else float("nan")
        z = (v - mean) / std if np.isfinite(v) and np.isfinite(mean) and np.isfinite(std) and std > 0 else float("nan")
        out[k] = {"void": v, "null_mean": mean, "null_std": std, "z": z, "draws": int(nv.size)}
    return out


def json_safe(obj):
    """The summary with every NaN or infinity turned into None (json.dumps writes bare NaN, which no JSON
    reader accepts); numpy scalars and arrays become Python numbers and lists."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
