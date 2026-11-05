"""The 3D view's volume, its Qt-free part (explore_3d.py draws it): the map's field over the window averaged down to
at most 'detail' cells a side (128^3 to 1024^3), quantised to 256 levels, a transfer function giving each level a
colour and an opacity, and the geometry of the slice stack that QPainter composites into the picture.

How it is drawn -- direct volume rendering by 2D texture slicing, the software form of the classic GPU method: the
averaged cube is cut into slices along the axis that faces the viewer most (orthographic: |cos| >= 1/sqrt 3, never
edge on; perspective: slice_axis), each slice an image of premultiplied RGBA (colour x opacity per cell, from the
transfer function), and the slices are composited front to back with the 'under' operator (QPainter's
DestinationOver), each one drawn onto the screen by the one map the camera gives its plane (view3d.plane_transform:
affine for an orthographic camera, a homography for a perspective one). The ray through a pixel crosses the slices
1/|cos| of their spacing apart, so the opacity of a slice is corrected for that path length: alpha' = 1 - (1 -
alpha)^(path / reference), the reference 1/REF_CELLS of the window's longest side. The look is then the same at any
detail, any window and any turn (orthographic; in perspective each ray's path differs from the reference ray's:
slice_plan). Compositing runs in 16 bits a channel: 8 bits lose most of a faint wall (256 slices of alpha 2/255
composite to 0.75 instead of 0.87, its blue to 0).

The grid is read by plain reads in slabs of whole x-layers, never memmapped (grids.OutputSet._memmap: a mapped page
of an external disk unplugged meanwhile kills the launcher with SIGBUS), a cube the map has cached already taken from
the cache, never added to it. The work runs on a small thread pool (numpy and file reads let go of the GIL: 8
threads gather a 1024^3 cube's colours 7.8x faster than one); memory stays near the cube's own size: its float32
average, its levels (1 byte a cell) and their transposed copy (the slices along z), and one coarse copy of at most
COARSE_MAX cells a side drawn while the view turns."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import grids as G
import view3d as V

DETAILS = (128, 256, 512, 1024)  # the averaged cube's cells a side, at most (a smaller window or grid: its own)
DEFAULT_DETAIL = 256
COARSE_MAX = 160                # the copy drawn while turning: at most this many cells a side, at least halved
                                # (a 512^3 cube turns at 24 ms a frame: 40 a second; at 256 a side, 47 ms)
LEVELS = 256                    # level 0: no value (NaN), transparent; 1..255 the scaled value's range
REF_CELLS = 128                 # the opacity is that of a cell 1/REF_CELLS of the window's longest side thick
OPACITY_RANGE = (0.003, 1.0)    # the opacity control's ends: a reference cell's opacity at the top of the range
DEFAULT_OPACITY = 0.3           # TNG100-3-Dark z = 0 at 256^3: filaments solid, haloes bright, the voids clear
DEFAULT_GAMMA = 1.6             # opacity ~ t^gamma along the shown range: walls faint, filaments and haloes solid
VOLUME_CACHE_BYTES = int(6e8)   # averaged cubes kept (float32: 67 MB at 256^3, 0.5 GB at 512^3; 1024^3 never)
PREPARED_BYTES = int(2.5e9)     # prepared cubes kept (2 bytes a cell: 2.1 GB at 1024^3); always the last one
BYTES_PER_CELL = 10.0           # a volume's peak memory a cell: float32 average, levels, transposed levels and the
                                # chunks' temporaries (measured: 9.5 GB at 1024^3, 1.7 GB at 512^3)


class Cancelled(Exception):
    """A newer request made the volume being read moot."""


_POOLS: dict = {}
_POOL_LOCK = threading.Lock()


def _executor(name: str) -> ThreadPoolExecutor:
    with _POOL_LOCK:
        if name not in _POOLS:
            _POOLS[name] = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) - 2)),
                                              thread_name_prefix=name)
        return _POOLS[name]


def pool() -> ThreadPoolExecutor:
    """The slice worker's threads: reading and averaging the grid, quantising it (tasks never wait on each other)."""
    return _executor("volume3d")


def gui_pool() -> ThreadPoolExecutor:
    """The GUI thread's own threads, for the slices' colours of the picture on screen: never queued behind a cube
    being averaged or coloured on pool() (the window would wait on them)."""
    return _executor("volume3d-gui")


def workers() -> int:
    return pool()._max_workers


def _run(fn, items, executor: ThreadPoolExecutor | None = None, progress: dict | None = None) -> None:
    """fn over items on 'executor' (pool()), at most its thread count queued at a time (another caller's task waits
    behind a few chunks, never a whole pass); the first failure raised once the running ones are done. 'progress'
    with 'cancel' set stops it (Cancelled)."""
    ex = executor or pool()
    flying: deque = deque()
    try:
        for item in items:
            if progress is not None and progress.get("cancel"):
                raise Cancelled()
            flying.append(ex.submit(fn, item))
            while len(flying) >= ex._max_workers:
                flying.popleft().result()
        while flying:
            flying.popleft().result()
    except BaseException:
        for f in flying:
            f.cancel()
        for f in flying:
            try:
                f.result()
            except BaseException:
                pass
        raise


# ---------------------------------------------------------------------- averaging the grid down
def block_edges(count: int, m: int) -> np.ndarray:
    """m + 1 edges cutting 'count' cells into m blocks as equal as whole cells allow (sizes differ by at most one):
    every cell in exactly one block, so a block mean conserves the field's integral (a density's mass)."""
    m = max(1, min(int(m), int(count)))
    return np.round(np.linspace(0, int(count), m + 1)).astype(np.int64)


def _runs(rows: np.ndarray):
    """The rows (indices along the first axis, periodic: maybe wrapped) as contiguous runs [(start, stop)]."""
    rows = np.asarray(rows, dtype=np.int64)
    breaks = np.nonzero(np.diff(rows) != 1)[0] + 1
    for seg in np.split(rows, breaks):
        if len(seg):
            yield int(seg[0]), int(seg[-1]) + 1


def read_rows(path, dtype, shape, rows, cube=None) -> np.ndarray:
    """Rows 'rows' (first-axis indices) of the C-order grid file 'path' of 'shape', by plain reads; from 'cube'
    instead when the cube cache holds the array. A short file raises ValueError, a drive gone OSError."""
    rows = np.asarray(rows, dtype=np.int64)
    if cube is not None:
        return cube[rows]
    dt = np.dtype(dtype)
    per = int(np.prod(shape[1:]))
    runs = list(_runs(rows))
    out = None if len(runs) == 1 else np.empty((len(rows),) + tuple(shape[1:]), dtype=dt)
    k = 0
    with open(path, "rb") as f:
        for s, e in runs:
            f.seek(s * per * dt.itemsize)
            a = np.fromfile(f, dtype=dt, count=(e - s) * per)
            if a.size != (e - s) * per:
                raise ValueError(f"{Path(path).name} is shorter than its grid (an interrupted write?)")
            a = a.reshape((e - s,) + tuple(shape[1:]))
            if out is None:
                return a
            out[k:k + e - s] = a
            k += e - s
    return out


def field_values(o, name: str, arr: np.ndarray, component: str) -> np.ndarray:
    """The values the map shows of 'name' (OutputSet.slice's rule, on a block of rows instead of a plane): its
    component (norm, trace, or one by name; the vorticity as the curl) in float64, a velocity in km/s."""
    ff = o.files[name]
    f64 = lambda a: np.asarray(a, dtype=np.float64)        # noqa: E731
    if name == "vorticity":         # [W_xy, W_xz, W_yz] = [-w_z, w_y, -w_x] / 2 -> the curl w
        arr = -2.0 * f64(arr) if ff.ncomp == 1 else \
            2.0 * np.stack([-f64(arr[..., 2]), f64(arr[..., 1]), -f64(arr[..., 0])], -1)
    if ff.ncomp > 1:
        comps = G.components(name, ff.ncomp, o.dim)
        if component not in comps:
            component = comps[0]
        if component == "norm":
            acc = np.zeros(arr.shape[:-1])
            for i in range(arr.shape[-1]):
                c = f64(arr[..., i])
                acc += c * c
            arr = np.sqrt(acc)
        elif component.startswith("trace"):
            d = int(round(ff.ncomp ** 0.5))
            arr = sum(f64(arr[..., i * d + i]) for i in range(d))
        else:
            idx = comps.index(component) - (1 if comps[0] == "norm" or comps[0].startswith("trace") else 0)
            arr = f64(arr[..., idx])
    factor = o._velocity_factor(name)
    arr = f64(arr)
    return arr * factor if factor != 1.0 else arr


def _sources(o, name: str) -> list[str]:
    """The grid files a volume of 'name' reads: the exact fold flag of a periodic PS run (OutputSet.fold_exact) is
    computed from up to three, any other field is its own file."""
    if name == "caustic" and o.fold_exact():
        return [n for n in ("caustic_class", "streams", "hidden_streams") if n in o.files]
    return [name]


def _file_shape(o, name: str) -> tuple:
    ff = o.files[name]
    return (int(o.n),) * 3 + ((ff.ncomp,) if ff.ncomp > 1 else ())


_VOLUMES: "OrderedDict[tuple, dict]" = OrderedDict()     # least recently used first; under _LOCK
_PREPARED: "OrderedDict[tuple, dict]" = OrderedDict()    # prepare()'s results, the same way
_LOCK = threading.Lock()


def volume_key(o, name: str, component: str, idx, detail: int) -> tuple | None:
    """What an averaged cube depends on: its files' identity (path, mtime, size: a grid written again is read
    again), the field and component, the window's cells and the detail."""
    keys = []
    for src in _sources(o, name):
        k = G._cube_key(o.files[src].path, o.dtype, _file_shape(o, src))
        if k is None:
            return None
        keys.append(k)
    window = tuple((int(i[0]), len(i)) for i in idx)
    return (tuple(keys), name, component, window, int(detail))


def drop_volumes() -> int:
    """Forget the averaged and prepared cubes (the bytes they held)."""
    with _LOCK:
        n = sum(v["values"].nbytes for v in _VOLUMES.values()) + sum(_prepared_bytes(p) for p in _PREPARED.values())
        _VOLUMES.clear()
        _PREPARED.clear()
    return n


def window_shape(o, lo, hi, detail: int) -> tuple[int, int, int]:
    """The averaged cube's cells a side for the window lo..hi at 'detail' (the window's own cells when fewer)."""
    idx, _wlo, _whi = V.cell_window(lo, hi, o.lo, o.length / int(o.n), int(o.n))
    return tuple(len(block_edges(len(i), detail)) - 1 for i in idx)


def need_bytes(shape) -> float:
    """The memory a volume of 'shape' cells needs at its peak (BYTES_PER_CELL a cell)."""
    return BYTES_PER_CELL * float(np.prod(shape))


def average_window(o, name: str, component: str, lo, hi, detail: int = DEFAULT_DETAIL, progress: dict | None = None
                   ) -> dict:
    """The field over the window lo..hi (Mpc, periodic: cell_window's cells), block-averaged to at most 'detail'
    cells a side: {'values': float32 (m0, m1, m2), 'lo', 'hi' (the window snapped to whole cells, Mpc), 'cells'
    (its fine cells a side), 'key'}. Read in blocks of whole x-layers on the pool, one block of averaged cells a task;
    the whole grid at its own cells, already in the cube cache, is that array itself (no copy). 'progress' (a dict
    shared with the GUI thread): 'done' / 'total' blocks are written as it goes, and 'cancel' set true stops it
    (Cancelled)."""
    n = int(o.n)
    cell = o.length / n
    idx, wlo, whi = V.cell_window(lo, hi, o.lo, cell, n)
    key = volume_key(o, name, component, idx, detail)
    with _LOCK:
        hit = _VOLUMES.get(key) if key is not None else None
        if hit is not None:
            _VOLUMES.move_to_end(key)
            return hit
    edges = [block_edges(len(idx[a]), detail) for a in range(3)]
    shape = tuple(len(e) - 1 for e in edges)
    full = [len(idx[a]) == n and int(idx[a][0]) == 0 for a in range(3)]   # the whole axis in order: no take
    srcs = _sources(o, name)
    cubes = {s: G._cached_cube(o.files[s].path, o.dtype, _file_shape(o, s), read=False) for s in srcs}
    base = {"lo": np.asarray(wlo, dtype=np.float64), "hi": np.asarray(whi, dtype=np.float64),
            "cells": tuple(len(i) for i in idx), "key": key}
    if all(full) and shape == (n, n, n) and srcs == [name] and o.files[name].ncomp == 1 and name != "vorticity" \
            and o._velocity_factor(name) == 1.0 and cubes[name] is not None:
        return dict(base, values=cubes[name])           # the cached grid itself (the cube cache keeps it): no averaging
    sizes1, sizes2 = np.diff(edges[1]).astype(np.float64), np.diff(edges[2]).astype(np.float64)
    out = np.empty(shape, dtype=np.float32)
    if progress is not None:
        progress.update(done=0, total=shape[0])

    def one(j):
        rows = idx[0][edges[0][j]:edges[0][j + 1]]
        blocks = {}
        for s in srcs:
            a = read_rows(o.files[s].path, o.dtype, _file_shape(o, s), rows, cubes[s])
            if not full[1]:
                a = np.take(a, idx[1], axis=1)
            if not full[2]:
                a = np.take(a, idx[2], axis=2)
            blocks[s] = a
        if name == "caustic" and o.fold_exact():
            vals = np.asarray(G.caustic_is_fold(blocks.get("caustic_class"), streams=blocks.get("streams"),
                                                hidden=blocks.get("hidden_streams")), dtype=np.float64)
        else:
            vals = field_values(o, name, blocks[name], component)
        del blocks
        vals = np.add.reduceat(vals, edges[2][:-1], axis=2)
        vals = np.add.reduceat(vals, edges[1][:-1], axis=1).sum(axis=0)
        out[j] = vals / (len(rows) * sizes1[:, None] * sizes2[None, :])

    flying: deque = deque()
    try:
        for j in range(shape[0]):
            if progress is not None and progress.get("cancel"):
                raise Cancelled()
            flying.append(pool().submit(one, j))
            while len(flying) >= workers():           # a few blocks in memory at a time
                flying.popleft().result()
                if progress is not None:
                    progress["done"] = progress.get("done", 0) + 1
        while flying:
            flying.popleft().result()
            if progress is not None:
                progress["done"] = progress.get("done", 0) + 1
    except BaseException:
        for f in flying:
            f.cancel()
        for f in flying:                                # the running ones end before the error goes up
            try:
                f.result()
            except BaseException:
                pass
        raise
    res = dict(base, values=out)
    if key is not None:
        # while the launcher runs a job (grids.cap_cubes) none is kept: the binary's auto-tuner budgets from the
        # memory free at its start; a cube larger than the cache is never kept
        budget = VOLUME_CACHE_BYTES if not G._CAPPED else 0
        with _LOCK:
            if out.nbytes <= budget:
                _VOLUMES[key] = res
            while _VOLUMES and sum(v["values"].nbytes for v in _VOLUMES.values()) > budget:
                _VOLUMES.popitem(last=False)
    return res


def shrink(values: np.ndarray, most: int = COARSE_MAX, progress: dict | None = None) -> np.ndarray | None:
    """The cube averaged over blocks (as equal as whole cells allow) to at most 'most' cells a side and at least
    halved: the coarse copy drawn while the view turns. None for a cube under 32 cells a side (cheap to draw whole)."""
    shape = values.shape
    if min(shape) < 32:
        return None
    edges = [block_edges(s, min(int(most), (s + 1) // 2)) for s in shape]
    sizes1, sizes2 = np.diff(edges[1]).astype(np.float64), np.diff(edges[2]).astype(np.float64)
    out = np.empty(tuple(len(e) - 1 for e in edges), dtype=np.float32)

    def one(j):
        blk = np.asarray(values[edges[0][j]:edges[0][j + 1]], dtype=np.float64)
        blk = np.add.reduceat(blk, edges[2][:-1], axis=2)
        blk = np.add.reduceat(blk, edges[1][:-1], axis=1).sum(axis=0)
        out[j] = blk / ((edges[0][j + 1] - edges[0][j]) * sizes1[:, None] * sizes2[None, :])
    _run(one, range(out.shape[0]), progress=progress)
    return out


# ---------------------------------------------------------------------- levels
def _chunks(m: int, step: int = 8):
    return [(i, min(i + step, m)) for i in range(0, m, step)]


def scaled(values: np.ndarray, log: bool, floor: float | None = None) -> np.ndarray:
    """The values on the map's scale, float32: log10 of the positive ones (the rest at 'floor', default the smallest
    positive one: grids._scaled's rule), NaN kept."""
    v = np.asarray(values, dtype=np.float32)
    if not log:
        return v.copy()
    if floor is None:
        pos = v > 0
        floor = float(v[pos].min()) if pos.any() else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(np.maximum(v, np.float32(floor)))     # maximum keeps NaN


def value_range(values: np.ndarray, log: bool, progress: dict | None = None) -> tuple[tuple[float, float], float]:
    """The scaled values' range and the log floor (the smallest positive value; 1 when none), in chunks on the pool:
    no copy of the cube."""
    m = values.shape[0]
    parts = [None] * len(_chunks(m))

    def one(k):
        i0, i1 = _chunks(m)[k]
        v = np.asarray(values[i0:i1], dtype=np.float32)
        ok = np.isfinite(v)
        pos = ok & (v > 0)
        parts[k] = (float(v[ok].min()) if ok.any() else np.inf, float(v[ok].max()) if ok.any() else -np.inf,
                    float(v[pos].min()) if pos.any() else np.inf)
    _run(one, range(len(parts)), progress=progress)
    vmin, vmax, pmin = (min(p[0] for p in parts), max(p[1] for p in parts), min(p[2] for p in parts))
    floor = pmin if np.isfinite(pmin) else 1.0
    if log:
        lo_, hi_ = np.log10(floor), np.log10(max(vmax, floor)) if np.isfinite(vmax) else np.log10(floor)
    else:
        lo_, hi_ = (vmin, vmax) if np.isfinite(vmin) else (0.0, 1.0)
    return (float(lo_), float(hi_)), float(floor)


def quantize(s: np.ndarray, qrange: tuple | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """Scaled values -> uint8 levels: 0 for no value (NaN), 1..255 over qrange (default: the finite values' range;
    beyond it clipped). Returns the levels and qrange."""
    finite = np.isfinite(s)
    if qrange is None:
        qlo, qhi = (float(s[finite].min()), float(s[finite].max())) if finite.any() else (0.0, 1.0)
    else:
        qlo, qhi = (float(v) for v in qrange)
    if not qhi > qlo:
        qhi = qlo + 1.0
    q = np.zeros(s.shape, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        q[finite] = (1.0 + np.clip(np.round((s[finite] - qlo) / (qhi - qlo) * (LEVELS - 2)), 0, LEVELS - 2)
                     ).astype(np.uint8)
    return q, (qlo, qhi)


def levels_of(values: np.ndarray, log: bool, qrange: tuple, floor: float, progress: dict | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """The cube's levels (quantize of its scaled values over qrange) and their counts, in chunks on the pool."""
    q = np.empty(values.shape, dtype=np.uint8)
    spans = _chunks(values.shape[0])
    counts = np.zeros((len(spans), LEVELS), dtype=np.int64)

    def one(k):
        i0, i1 = spans[k]
        q[i0:i1] = quantize(scaled(values[i0:i1], log, floor), qrange)[0]
        counts[k] = np.bincount(q[i0:i1].ravel(), minlength=LEVELS)
    _run(one, range(len(spans)), progress=progress)
    return q, counts.sum(axis=0)


def transposed(q: np.ndarray, progress: dict | None = None) -> np.ndarray:
    """q's slices along its last axis made contiguous: out[k] = q[:, :, k] (in chunks on the pool)."""
    out = np.empty((q.shape[2], q.shape[0], q.shape[1]), dtype=q.dtype)

    def one(span):
        i0, i1 = span
        out[:, i0:i1, :] = np.moveaxis(q[i0:i1], 2, 0)
    _run(one, _chunks(q.shape[0], 32), progress=progress)
    return out


def level_values(qrange: tuple) -> np.ndarray:
    """The scaled value of each level (level 0, no value: -inf)."""
    qlo, qhi = qrange
    v = qlo + (np.arange(LEVELS, dtype=np.float64) - 1.0) / (LEVELS - 2) * (qhi - qlo)
    v[0] = -np.inf
    return v


def oriented(q: np.ndarray) -> list[np.ndarray]:
    """The levels as three stacks of slice images, one per axis a: (m_a, rows, cols) contiguous, rows along the first
    remaining axis and columns along the second (np.moveaxis(q, a, 0)), as slice_plan places them."""
    return [np.ascontiguousarray(np.moveaxis(q, a, 0)) for a in range(3)]


def slice_images(src: dict, a: int) -> np.ndarray:
    """The slice images along axis a of a prepared cube ('fine' or 'coarse'): (m_a, rows, cols), slice i's rows along
    the first remaining axis, its columns along the second (the fine one's along y: a strided view of q, rows still
    contiguous)."""
    if "stacks" in src:
        return src["stacks"][a]
    q = src["q"]
    return q if a == 0 else q.transpose(1, 0, 2) if a == 1 else src["qT"]


def gather(lut: np.ndarray, images: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """lut[images] (a stack of slice images' levels -> their RGBA64 pixels), in slices on the GUI's threads."""
    if out is None:
        out = np.empty(images.shape, dtype=lut.dtype)

    def one(span):
        i0, i1 = span
        np.take(lut, images[i0:i1], out=out[i0:i1])
    _run(one, _chunks(images.shape[0], 8), gui_pool())
    return out


def percentile_level(counts: np.ndarray, levels: np.ndarray, pct: float) -> float:
    """The scaled value below which 'pct' percent of the cells with a value lie (from the level counts)."""
    c = np.cumsum(np.asarray(counts[1:], dtype=np.float64))
    if not c[-1] > 0:
        return float(levels[1])
    k = int(np.searchsorted(c, pct / 100.0 * c[-1]))
    return float(levels[1 + min(k, LEVELS - 2)])


def _prepared_bytes(p: dict) -> int:
    n = p["fine"]["q"].nbytes + p["fine"]["qT"].nbytes
    if p.get("coarse") is not None:
        n += sum(s.nbytes for s in p["coarse"]["stacks"])
    return n


def prepare(avg: dict, log: bool, progress: dict | None = None) -> dict:
    """What the GUI thread needs to draw an averaged cube: {'fine': its levels q (C order) and qT (q's slices along z,
    contiguous), 'coarse': the copy drawn while turning (its three stacks of slice images: oriented) or None, 'shape',
    'qrange', 'levels' (each level's scaled value), 'counts', 'lo', 'hi', 'cells', 'log'}. Runs on the slice worker,
    in chunks on the pool; the last ones kept up to PREPARED_BYTES (back from a close-up: no second quantisation),
    none while the launcher runs a job (grids.cap_cubes: the view holds the one it shows). 'progress' gets stage
    'colouring' while it runs, and its 'cancel' stops it between chunks (Cancelled)."""
    key = None if avg.get("key") is None else (avg["key"], bool(log))
    with _LOCK:
        hit = _PREPARED.get(key) if key is not None else None
        if hit is not None:
            _PREPARED.move_to_end(key)
            return hit
    if progress is not None:
        progress["stage"] = "colouring"
    prep = _prepare(avg, log, progress)
    if key is not None and not G._CAPPED and not (progress or {}).get("cancel"):
        with _LOCK:
            _PREPARED[key] = prep
            while len(_PREPARED) > 1 and sum(_prepared_bytes(p) for p in _PREPARED.values()) > PREPARED_BYTES:
                _PREPARED.popitem(last=False)
    return prep


def prepared_hit(o, name: str, component: str, lo, hi, detail: int, log: bool) -> dict | None:
    """The prepared cube of this window, field and detail when it is still kept: no reading, averaging or
    colouring at all (a 1024^3 average is never kept whole, its prepared cube is)."""
    idx, _wlo, _whi = V.cell_window(lo, hi, o.lo, o.length / int(o.n), int(o.n))
    key = volume_key(o, name, component, idx, detail)
    if key is None:
        return None
    with _LOCK:
        hit = _PREPARED.get((key, bool(log)))
        if hit is not None:
            _PREPARED.move_to_end((key, bool(log)))
    return hit


def _prepare(avg: dict, log: bool, progress: dict | None = None) -> dict:
    vals = avg["values"]
    qrange, floor = value_range(vals, log, progress)
    if not qrange[1] > qrange[0]:
        qrange = (qrange[0], qrange[0] + 1.0)
    q, counts = levels_of(vals, log, qrange, floor, progress)
    small = shrink(vals, COARSE_MAX, progress)
    coarse = None
    if small is not None:
        qc = quantize(scaled(small, log, floor), qrange)[0]
        coarse = {"stacks": oriented(qc), "shape": qc.shape}
    qT = transposed(q, progress)
    return {"fine": {"q": q, "qT": qT, "shape": q.shape}, "coarse": coarse, "shape": q.shape,
            "qrange": qrange, "levels": level_values(qrange), "counts": counts, "lo": avg["lo"], "hi": avg["hi"],
            "cells": avg["cells"], "log": bool(log)}


# ---------------------------------------------------------------------- the transfer function
def opacity_curve(t: np.ndarray, gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    """The opacity along the shown range, 0 at its low end to 1 at its high end (t in 0..1, the scaled value's place
    in the range: log10 of the density from the low threshold to the high one)."""
    # TODO(human) (optional: this default works): the shape of the opacity ramp. A power t^gamma with gamma ~ 1.6
    # keeps the walls just above the threshold faint and makes filaments and haloes solid; gamma 1 is a plain linear
    # ramp (walls and filaments alike, a foggier box), a larger gamma shows only the densest knots. A smoothstep
    # (3t^2 - 2t^3) is another choice: soft at both ends. The function must be 0 at t = 0, rise to 1 at t = 1 and
    # never fall (tests: transparent below the threshold, monotone).
    return np.power(np.clip(t, 0.0, 1.0), gamma)


def transfer(levels: np.ndarray, lo: float, hi: float, opacity: float, exponent: float, colours: np.ndarray,
             symmetric: bool = False, gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    """The 16-bit premultiplied RGBA of each level as QImage.Format_RGBA64_Premultiplied stores it (one uint64 a
    level: R, G, B, A halfwords in memory order): levels below 'lo' (and level 0) transparent, the opacity rising to
    'opacity' at 'hi' (opacity_curve) and corrected for the path through a slice ('exponent': the path over the
    reference thickness), the colour 'colours' ((256, 3) uint8, the colour map) over lo..hi. 'symmetric' (a field
    centred on 0, a diverging map): lo..hi bound |value|, the colour spans -hi..hi."""
    v = np.asarray(levels, dtype=np.float64)
    span = hi - lo if hi > lo else 1.0
    a = np.abs(v) if symmetric else v
    with np.errstate(invalid="ignore"):
        t = (a - lo) / span
        alpha = np.where(np.isfinite(t) & (t >= 0.0), opacity_curve(t, gamma) * float(opacity), 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = 1.0 - np.power(1.0 - alpha, max(float(exponent), 1e-6))
    alpha[0] = 0.0
    if symmetric:
        top = max(abs(hi), 1e-30)
        c = np.clip((v + top) / (2.0 * top), 0.0, 1.0)
    else:
        c = np.clip(np.nan_to_num((v - lo) / span, nan=0.0, neginf=0.0), 0.0, 1.0)
    rgb = np.asarray(colours, dtype=np.float64)[np.round(c * 255).astype(np.int64)] / 255.0
    out = np.empty((len(v), 4), dtype=np.uint16)
    out[:, :3] = np.round(rgb * alpha[:, None] * 65535.0)
    out[:, 3] = np.round(alpha * 65535.0)
    return out.view(np.uint64).ravel()


def default_range(prep: dict, name: str, symmetric: bool = False) -> tuple[float, float]:
    """The shown range (scaled) a volume opens with: a density from the mean (rho / rho_bar = 1: the voids and the
    underdense walls see-through) to its 99.9th percentile; any other field from its 70th percentile to the 99.9th
    (|value| when symmetric)."""
    levels, counts = prep["levels"], prep["counts"]
    hi = percentile_level(counts, levels, 99.9)
    if name == "density" and prep["log"]:
        lo = 0.0 if hi > 0.0 else percentile_level(counts, levels, 70.0)
    elif symmetric:
        top = max(abs(float(levels[1])), abs(float(levels[-1])))
        return 0.3 * top, 0.9 * top
    else:
        lo = percentile_level(counts, levels, 70.0)
    if not hi > lo:
        hi = float(levels[-1]) if levels[-1] > lo else lo + 1.0
    return float(lo), float(hi)


def visible_fraction(prep: dict, lo: float, symmetric: bool = False) -> float:
    """The fraction of the cells at or above the low threshold (|value| when symmetric): not transparent."""
    levels, counts = prep["levels"], np.asarray(prep["counts"], dtype=np.float64)
    a = np.abs(levels) if symmetric else levels
    total = counts[1:].sum()
    return float(counts[1:][a[1:] >= lo].sum() / total) if total > 0 else 0.0


# ---------------------------------------------------------------------- the slices on the screen
def slice_axis(cam: V.Camera, lo, hi) -> tuple[int, np.ndarray]:
    """The slicing axis and the reference ray's direction (unit, toward the window): orthographic, the axis facing
    the viewer most (|cos| >= 1/sqrt 3: never edge on) and the view direction. Perspective: every pixel's ray has
    its own direction, so the axis is the one whose worst ray -- over the rays from the eye to the window's
    corners -- is the least oblique (never an axis whose slab holds the eye: some of its slices would be seen
    edge on), and the reference ray is the one to the window's centre."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if not cam.perspective:
        t = cam.toward()
        return int(np.argmax(np.abs(t))), -t
    e = cam.eye()
    rays = V.box_corners(lo, hi) - e
    # the smallest |cos| of a ray to the window with each axis: at a corner (the nearest face's |gap| over the
    # farthest reach), or 0 when the eye is level with the axis's slab (a ray into the window then runs along it)
    worst = (np.abs(rays) / np.linalg.norm(rays, axis=1)[:, None]).min(axis=0)
    worst[(e >= lo) & (e <= hi)] = 0.0
    c = 0.5 * (lo + hi) - e
    a = int(np.argmax(worst)) if worst.max() > 0 else int(np.argmax(np.abs(cam.toward())))
    return a, c / np.linalg.norm(c)


def slice_plan(cam: V.Camera, lo, hi, shape) -> dict:
    """How the slice stack of a cube of 'shape' cells filling lo..hi (Mpc) lands on the screen of 'cam':
    {'axis' (slice_axis), 'order' (slice indices, front first: perspective, nearest the eye first), 'transforms'
    ((m, 9): slice i's image pixels -> screen, view3d.plane_transform's numbers; image rows along the first remaining
    axis, columns along the second, as slice_images gives them), 'size' (the image's width, height), 'exponent' (the
    path through a slice over the reference thickness, along the reference ray)}. Perspective: each ray crosses the
    slices at its own angle, so its true path is the spacing over its own |cos|, and a uniform field's optical
    depth varies across the picture by that ratio to the reference ray's (some tens of percent at the first fit,
    more zoomed in near a corner) -- 2D texture slicing's inherent error; orthographic it is exact."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    a, ray = slice_axis(cam, lo, hi)
    r0, r1 = V.rest_axes(a)
    m = int(shape[a])
    d = (hi - lo) / np.asarray(shape, dtype=np.float64)
    pos = lo[a] + (np.arange(m) + 0.5) * d[a]
    P = cam.matrix()
    ex, ey, step, origin = np.zeros(3), np.zeros(3), np.zeros(3), lo.copy()
    ex[r1], ey[r0], step[a] = d[r1], d[r0], d[a]
    origin[a] = pos[0]
    hx, hy = P[:, :3] @ ex, P[:, :3] @ ey
    h0 = (P @ np.append(origin, 1.0))[None, :] + np.arange(m)[:, None] * (P[:, :3] @ step)[None, :]
    # divided by |w| at each slice's origin: w keeps its sign (plane_transform's reason)
    T = np.concatenate([np.broadcast_to(hx, (m, 3)), np.broadcast_to(hy, (m, 3)), h0], axis=1) / np.abs(h0[:, 2:3])
    if cam.perspective:
        order = np.argsort(np.abs(pos - cam.eye()[a]), kind="stable")
    else:
        order = np.arange(m) if ray[a] > 0 else np.arange(m)[::-1]  # the viewer on the high side: high slices first
    ref = float(np.max(hi - lo)) / REF_CELLS
    return {"axis": a, "order": order, "transforms": T, "size": (int(shape[r1]), int(shape[r0])),
            "exponent": float(d[a] / max(abs(ray[a]), 1e-9) / ref) if ref > 0 else 1.0}


def composite(rgba: np.ndarray) -> np.ndarray:
    """The reference compositing of premultiplied RGBA samples along a ray, front first ((k, 4) floats in 0..1, or
    (..., k, 4)): C = sum_i c_i prod_{j < i} (1 - a_j), the 'under' operator QPainter's DestinationOver applies
    slice by slice. The tests hold the picture to it."""
    x = np.asarray(rgba, dtype=np.float64)
    trans = np.cumprod(1.0 - x[..., 3], axis=-1)
    before = np.concatenate([np.ones(trans.shape[:-1] + (1,)), trans[..., :-1]], axis=-1)
    return (x * before[..., None]).sum(axis=-2)
