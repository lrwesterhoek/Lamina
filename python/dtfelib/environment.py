"""Sample DTFE / PS-DTFE grid fields at (halo) positions -- the merger-tree <-> field bridge.
Positions are fractional box coordinates from the raw-snapshot box size (ckpc/h). Continuous fields are sampled
trilinearly (periodic), web labels by nearest grid point; snapshots without fields give NaN (env["missing"])."""

from __future__ import annotations


import h5py
import numpy as np

from ._threads import map_chunks
from .cli import sim_dir, snapdir_path
from .io import FieldSet, MEAN_FIELD_NAMES

# fields whose values are categorical labels: sample NGP, never interpolate
LABEL_FIELDS = {"tweb", "vweb"}


def box_ckpc_h(sim: str, snap: int) -> float:
    """Box size in ckpc/h (the SubLink/raw-snapshot frame) for `sim` at `snap`: from a raw chunk or groupcat
    header, else the merged combined_*.hdf5 (BoxSize * h)."""
    root = sim_dir(sim)
    snapdir = snapdir_path(sim, snap)
    for pat in (f"snap_{snap:03d}.*.hdf5",):
        files = sorted(snapdir.glob(pat))
        if files:
            with h5py.File(files[0], "r") as f:
                return float(f["Header"].attrs["BoxSize"])
    gdir = root / f"groups_{snap:03d}"
    files = sorted(gdir.glob("fof_subhalo_tab_*.hdf5"))
    if files:
        with h5py.File(files[0], "r") as f:
            return float(f["Header"].attrs["BoxSize"])
    files = sorted(snapdir.glob(f"combined_{snap:03d}.hdf5"))
    if files:
        with h5py.File(files[0], "r") as f:
            h = f["Header"].attrs
            return float(h["BoxSize"]) * float(h["HubbleParam"])   # h-free ckpc -> ckpc/h
    raise FileNotFoundError(f"no header source for the box size of {sim} snap {snap}")


def sample_grid(grid: np.ndarray, frac: np.ndarray, mode: str = "trilinear") -> np.ndarray:
    """Sample a periodic (N,N,N) or (N,N,N,C) grid at fractional positions frac (M,3) in [0,1), mode='trilinear'
    (continuous) or 'ngp' (labels). Cell i covers [i/N, (i+1)/N); centres sit at (i+0.5)/N. Threaded over the
    points when there are many (each point the same computation: bit-identical)."""
    frac = np.asarray(frac, dtype=np.float64)
    parts = map_chunks(len(frac), lambda lo, hi: _sample_grid(grid, frac[lo:hi], mode), min_size=1 << 18)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _sample_grid(grid: np.ndarray, frac: np.ndarray, mode: str = "trilinear") -> np.ndarray:
    """sample_grid on one chunk of points."""
    n = grid.shape[0]
    frac = np.mod(np.asarray(frac, dtype=np.float64), 1.0)
    if mode == "ngp":
        idx = np.floor(frac * n).astype(np.int64) % n
        return grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    if mode != "trilinear":
        raise ValueError(f"unknown sampling mode {mode!r}")

    # cell-centre aligned trilinear interpolation with periodic wrap
    x = frac * n - 0.5
    i0 = np.floor(x).astype(np.int64)
    w = x - i0                                  # (M,3) weights toward the upper neighbour
    i0 %= n
    i1 = (i0 + 1) % n
    out = None
    for dx, wx in ((i0[:, 0], 1 - w[:, 0]), (i1[:, 0], w[:, 0])):
        for dy, wy in ((i0[:, 1], 1 - w[:, 1]), (i1[:, 1], w[:, 1])):
            for dz, wz in ((i0[:, 2], 1 - w[:, 2]), (i1[:, 2], w[:, 2])):
                wgt = (wx * wy * wz)
                vals = grid[dx, dy, dz]
                if vals.ndim > 1:
                    wgt = wgt[:, None]
                out = vals * wgt if out is None else out + vals * wgt
    return out


def sample_fields_at(sim: str, snap: int, pos_ckpc_h: np.ndarray,
                     fields=("density", "tweb", "vweb"),
                     method: str = "auto", averaged: bool = True,
                     prefix: str | None = None, mean_field: bool = True) -> dict:
    """Sample `fields` at positions (M,3) in ckpc/h for one snapshot. Density is rho/rho_bar for both estimators;
    missing fields come back as NaN. prefix reads an alternate grid set (e.g. 'ps_mw'); mean_field selects the
    mean-field velocity-gradient family (see FieldSet.load)."""
    fs = FieldSet(snapdir_path(sim, snap), method=method, averaged=averaged,
                  prefix=prefix)
    frac = np.asarray(pos_ckpc_h, dtype=np.float64) / box_ckpc_h(sim, snap)
    out = {"_method": fs.method, "_grid_n": fs.grid_n}
    for name in fields:
        mf = mean_field and name in MEAN_FIELD_NAMES
        if not fs.has(name, mean_field=mf):
            out[name] = np.full(frac.shape[0], np.nan)
            continue
        grid = fs.load(name, mean_field=mf)
        if name == "density":
            # mean units via the grid's OWN mean: mass conservation makes the volume mean equal rho_bar
            # in whatever units the file was written (unlike FieldSet.density).
            m = float(grid.mean())
            if m > 0:
                grid = grid / m
        out[name] = sample_grid(grid, frac, "ngp" if name in LABEL_FIELDS else "trilinear")
    return out


def branch_environment(sim: str, branch: dict, fields=("density", "tweb", "vweb"),
                       method: str = "auto", averaged: bool = True, prefix: str | None = None,
                       mean_field: bool = True) -> dict:
    """Field environment along a SubLink branch (output of TreeSet.main_branch, which must include 'SubhaloPos').
    One value per branch snapshot, NaN where no grids exist. Returns {field: (n,), "missing": [...], "method": {...}}.
    prefix, mean_field: as sample_fields_at; it samples one position per call."""
    snaps = branch["SnapNum"].astype(int)
    pos = np.asarray(branch["SubhaloPos"], dtype=np.float64)
    out = {name: np.full(snaps.size, np.nan) for name in fields}
    out["missing"] = []
    out["method"] = {}
    for i, s in enumerate(snaps):
        try:
            smp = sample_fields_at(sim, int(s), pos[i:i + 1], fields, method, averaged, prefix, mean_field)
        except (FileNotFoundError, ValueError):
            out["missing"].append(int(s))
            continue
        out["method"][int(s)] = smp["_method"]
        for name in fields:
            vec = smp[name]
            out[name][i] = float(vec[0]) if np.ndim(vec[0]) == 0 else np.nan
    return out
