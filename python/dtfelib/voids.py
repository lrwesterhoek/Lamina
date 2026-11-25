"""Void TIME TRACKING via merger-tree tracers -- the tree side of plot_void_tracking.py. Properties come from the
same per-snapshot void catalogues as the correlation analysis; identity across time comes from walking the
subhalos in and around a void back along their main branches. Positions are FRACTIONAL box coordinates in [0,1)."""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ periodic helpers
def wrap_frac(x):
    return np.mod(x, 1.0)


def min_image(d):
    """Minimum-image displacement(s) in fractional units."""
    return d - np.rint(d)


def periodic_median_shift(prev, cur):
    """Median minimum-image displacement from tracer positions prev -> cur, both (M,3)."""
    return np.median(min_image(cur - prev), axis=0)


# ------------------------------------------------------------------ catalog matching
def match_catalog_void(cat, center_frac, n_grid, max_dist_frac):
    """Index of the catalog void nearest to center_frac (periodic), or None if the nearest
    lies beyond max_dist_frac. `cat` is a pipeline void catalog (build_void_catalog dict)."""
    frac = (np.asarray(cat["coords"], dtype=np.float64) + 0.5) / n_grid
    d = np.linalg.norm(min_image(frac - np.asarray(center_frac)), axis=1)
    i = int(np.argmin(d))
    return (i, float(d[i])) if d[i] <= max_dist_frac else (None, float(d[i]))


# ------------------------------------------------------------------ tree-side tracking
def tracer_branches(treeset, snap0, subfind_ids, box_ckpc_h):
    """Main-branch fractional positions for each tracer: {sid: {snap: frac_pos(3,)}}."""
    out = {}
    for sid in subfind_ids:
        try:
            br = treeset.main_branch(snap0, int(sid), ["SubhaloPos"])
        except KeyError:
            continue
        pos = wrap_frac(np.asarray(br["SubhaloPos"], dtype=np.float64) / box_ckpc_h)
        out[int(sid)] = dict(zip(br["SnapNum"].astype(int).tolist(), pos))
    return out


def track_center(branches, snap0, center0_frac, min_tracers=3):
    """Void centre per snapshot, walking BACK from snap0: between consecutive snapshots the centre moves by the
    median minimum-image displacement of the tracers present at both. Returns {snap: frac_center}, covering every
    snapshot at least `min_tracers` reach."""
    snaps = sorted({s for b in branches.values() for s in b}, reverse=True)
    snaps = [s for s in snaps if s <= snap0]
    centers = {snap0: np.asarray(center0_frac, dtype=np.float64)}
    prev = snap0
    for s in snaps:
        if s == snap0:
            continue
        pairs = [(b[prev], b[s]) for b in branches.values() if prev in b and s in b]
        if len(pairs) < min_tracers:
            break
        prev_pos = np.array([p for p, _ in pairs])
        cur_pos = np.array([c for _, c in pairs])
        centers[s] = wrap_frac(centers[prev] + periodic_median_shift(prev_pos, cur_pos))
        prev = s
    return centers


def select_tracers(pool_frac, center_frac, r_sel_frac, max_tracers=200):
    """Indices of pool positions within r_sel of the centre (periodic), nearest first."""
    d = np.linalg.norm(min_image(pool_frac - np.asarray(center_frac)), axis=1)
    idx = np.nonzero(d <= r_sel_frac)[0]
    return idx[np.argsort(d[idx])][:max_tracers]
