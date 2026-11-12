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
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))     # the threads of each pool (pool(), gui_pool()), and the
                                # tasks _run keeps queued on one


class Cancelled(Exception):
    """A newer request made the volume being read moot."""


_POOLS: dict = {}
_POOL_LOCK = threading.Lock()


def _executor(name: str) -> ThreadPoolExecutor:
    with _POOL_LOCK:
        if name not in _POOLS:
            _POOLS[name] = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix=name)
        return _POOLS[name]


def pool() -> ThreadPoolExecutor:
    """The slice worker's threads: reading and averaging the grid, quantising it (tasks never wait on each other)."""
    return _executor("volume3d")


def gui_pool() -> ThreadPoolExecutor:
    """The GUI thread's own threads, for the slices' colours of the picture on screen: never queued behind a cube
    being averaged or coloured on pool() (the window would wait on them)."""
    return _executor("volume3d-gui")


def _run(fn, items, executor: ThreadPoolExecutor | None = None, progress: dict | None = None,
         count: bool = False) -> None:
    """fn over items on 'executor' (pool()), at most its thread count (WORKERS) queued at a time (another caller's
    task waits behind a few chunks, never a whole pass); the first failure raised once the running ones are done.
    'progress' with 'cancel' set stops it (Cancelled), checked before each item is queued; with 'count',
    progress['done'] goes up by one as each item finishes, in order (average_window's blocks: the GUI shows it)."""
    ex = executor or pool()
    flying: deque = deque()

    def finish_one():
        flying.popleft().result()
        if count and progress is not None:
            progress["done"] = progress.get("done", 0) + 1
    try:
        for item in items:
            if progress is not None and progress.get("cancel"):
                raise Cancelled()
            flying.append(ex.submit(fn, item))
            while len(flying) >= WORKERS:
                finish_one()
        while flying:
            finish_one()
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


def sources(o, name: str) -> list[str]:
    """The grid files a volume of 'name' reads: the exact fold flag of a periodic PS run (OutputSet.fold_exact) is
    computed from up to three, any other field is its own file."""
    if name == "caustic" and o.fold_exact():
        return [n for n in ("caustic_class", "streams", "hidden_streams") if n in o.files]
    return [name]


def file_shape(o, name: str) -> tuple:
    """The shape of a field's grid file: (n, n, n) and its components."""
    ff = o.files[name]
    return (int(o.n),) * 3 + ((ff.ncomp,) if ff.ncomp > 1 else ())


_VOLUMES: "OrderedDict[tuple, dict]" = OrderedDict()     # least recently used first; under _LOCK
_PREPARED: "OrderedDict[tuple, dict]" = OrderedDict()    # prepare()'s results, the same way
_LOCK = threading.Lock()


def volume_key(o, name: str, component: str, idx, detail: int) -> tuple | None:
    """What an averaged cube depends on: its files' identity (path, mtime, size: a grid written again is read
    again), the field and component, the window's cells and the detail."""
    keys = []
    for src in sources(o, name):
        k = G.cube_key(o.files[src].path, o.dtype, file_shape(o, src))
        if k is None:
            return None
        keys.append(k)
    window = tuple((int(i[0]), len(i)) for i in idx)
    return (tuple(keys), name, component, window, int(detail))


def drop_volumes() -> int:
    """Forget the averaged and prepared cubes (the bytes they held). grids.drop_cubes() calls it (registered below):
    a job start and a memory check release these with the map's cubes. The volume on screen and its GPU copy stay:
    View3D holds them (its prepared cube, two bytes a cell, and the GPU's byte a cell) until another volume replaces
    them or the 3D view closes -- the user is looking at it, and reading it again would cost more memory than it
    frees (the averaging's peak, BYTES_PER_CELL); the 3D view's own memory check (its detail choice) counts both."""
    with _LOCK:
        n = sum(v["values"].nbytes for v in _VOLUMES.values()) + sum(prepared_bytes(p) for p in _PREPARED.values())
        _VOLUMES.clear()
        _PREPARED.clear()
    return n


G.on_drop_cubes(drop_volumes)


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
    srcs = sources(o, name)
    cubes = {s: G.cached_cube(o.files[s].path, o.dtype, file_shape(o, s), read=False) for s in srcs}
    base = {"lo": np.asarray(wlo, dtype=np.float64), "hi": np.asarray(whi, dtype=np.float64),
            "cells": tuple(len(i) for i in idx), "key": key}
    if all(full) and shape == (n, n, n) and srcs == [name] and o.files[name].ncomp == 1 and name != "vorticity" \
            and o.velocity_factor(name) == 1.0 and cubes[name] is not None:
        return dict(base, values=cubes[name])           # the cached grid itself (the cube cache keeps it): no averaging
    sizes1, sizes2 = np.diff(edges[1]).astype(np.float64), np.diff(edges[2]).astype(np.float64)
    out = np.empty(shape, dtype=np.float32)
    if progress is not None:
        progress.update(done=0, total=shape[0])
    # a scalar field shown as stored (no component, no velocity factor, no fold flag) is the grid's own values: no
    # float64 copy of each block (summed in float64 straight from the file's floats: the same sums), and a window at
    # its own cells (the detail at least its cells) not summed at all, each averaged cell being one grid cell
    plain = srcs == [name] and o.files[name].ncomp == 1 and name != "vorticity" and o.velocity_factor(name) == 1.0
    unit = shape == tuple(len(i) for i in idx)
    # blocks of one size f <= 4 along y and z (1024 -> 512 or 256): each block summed by f strided adds in reduceat's
    # own order -- its first term plus the sum of the rest, that sum taken from 0 one term after another (numpy's
    # pairwise sum below 8 terms): a0 + (((0 + a1) + a2) + a3), the same sums bit for bit -- without reduceat's cost
    # a block (1024 -> 512 in 0.45 s against 1.1; from blocks of 8 on, reduceat is the faster)
    f1, f2 = len(idx[1]) // shape[1], len(idx[2]) // shape[2]
    even = all(len(idx[a]) % shape[a] == 0 for a in (1, 2)) and max(f1, f2) <= 4

    def strided_sum(a: np.ndarray, f: int, axis: int) -> np.ndarray:
        take = [slice(None)] * a.ndim
        take[axis] = slice(0, None, f)
        first = a[tuple(take)].astype(np.float64)
        if f == 1:
            return first
        rest = np.zeros(first.shape, dtype=np.float64)
        for k in range(1, f):
            take[axis] = slice(k, None, f)
            rest += a[tuple(take)]
        first += rest
        return first

    def one(j):
        rows = idx[0][edges[0][j]:edges[0][j + 1]]
        blocks = {}
        for s in srcs:
            a = read_rows(o.files[s].path, o.dtype, file_shape(o, s), rows, cubes[s])
            if not full[1]:
                a = np.take(a, idx[1], axis=1)
            if not full[2]:
                a = np.take(a, idx[2], axis=2)
            blocks[s] = a
        if plain and unit:
            out[j] = blocks[name][0]
            out[j] += np.float32(0.0)       # as the summing path's final sum from 0 did: a -0.0 becomes +0.0
            return
        if plain and even:
            vals = strided_sum(strided_sum(blocks.pop(name), f2, 2), f1, 1).sum(axis=0)
            out[j] = vals / (len(rows) * sizes1[:, None] * sizes2[None, :])
            return
        if plain:
            vals = np.add.reduceat(blocks.pop(name), edges[2][:-1], axis=2, dtype=np.float64)
        else:
            if name == "caustic" and o.fold_exact():
                vals = np.asarray(G.caustic_is_fold(blocks.get("caustic_class"), streams=blocks.get("streams"),
                                                    hidden=blocks.get("hidden_streams")), dtype=np.float64)
            else:
                vals = o.component_values(name, blocks[name], component, fresh=False)
            vals = np.add.reduceat(vals, edges[2][:-1], axis=2)
        del blocks
        vals = np.add.reduceat(vals, edges[1][:-1], axis=1).sum(axis=0)
        out[j] = vals / (len(rows) * sizes1[:, None] * sizes2[None, :])

    _run(one, range(shape[0]), progress=progress, count=True)        # a few blocks in memory at a time
    res = dict(base, values=out)
    if key is not None:
        # while the launcher runs a job (grids.cap_cubes) none is kept: the binary's auto-tuner budgets from the
        # memory free at its start; a cube larger than the cache is never kept
        budget = VOLUME_CACHE_BYTES if not G.cubes_capped() else 0
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
        blk = np.add.reduceat(values[edges[0][j]:edges[0][j + 1]], edges[2][:-1], axis=2, dtype=np.float64)
        blk = np.add.reduceat(blk, edges[1][:-1], axis=1).sum(axis=0)
        out[j] = blk / ((edges[0][j + 1] - edges[0][j]) * sizes1[:, None] * sizes2[None, :])
    _run(one, range(out.shape[0]), progress=progress)
    return out


# ---------------------------------------------------------------------- levels
def _chunks(m: int, step: int = 8):
    return [(i, min(i + step, m)) for i in range(0, m, step)]


CHUNK_CELLS = 1 << 20           # the cells a pass over the cube takes at a time: 4 MB of float32, in the core's cache
                                # (whole 8-row chunks of a 1024^3 cube, 32 MB, went to memory and back at every op)


def _cell_chunks(shape) -> list[tuple[int, int]]:
    """Spans of whole first-axis rows of about CHUNK_CELLS cells each."""
    per = max(1, int(np.prod(shape[1:])))
    return _chunks(int(shape[0]), max(1, CHUNK_CELLS // per))


def scaled(values: np.ndarray, log: bool, floor: float | None = None) -> np.ndarray:
    """The values on the map's scale, float32: log10 of the positive ones (the rest at 'floor', default the smallest
    positive one: grids._scaled's rule), NaN kept. Always a fresh array (levels_of works in it in place)."""
    v = np.asarray(values, dtype=np.float32)
    if not log:
        return v.copy()
    if floor is None:
        pos = v > 0
        floor = float(v[pos].min()) if pos.any() else 1.0
    out = np.maximum(v, np.float32(floor))                     # maximum keeps NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(out, out=out)


def value_range(values: np.ndarray, log: bool, progress: dict | None = None) -> tuple[tuple[float, float], float]:
    """The scaled values' range and the log floor (the smallest positive value; 1 when none), in chunks on the pool:
    no copy of the cube."""
    spans = _chunks(values.shape[0])        # 8 rows a chunk, as before: which zero (+0 or -0) is the minimum
    parts = [None] * len(spans)             # depends on how the cells are grouped

    def one(k):
        i0, i1 = spans[k]
        v = np.asarray(values[i0:i1], dtype=np.float32)
        lo_, hi_ = v.min(), v.max()                 # NaN or an infinity in the chunk: one of these is not finite
        if np.isfinite(lo_) and np.isfinite(hi_):
            parts[k] = (float(lo_), float(hi_), float(lo_) if lo_ > 0 else
                        float(v[v > 0].min()) if hi_ > 0 else np.inf)
            return
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


def _quantize_into(s: np.ndarray, qrange: tuple, out: np.ndarray) -> None:
    """quantize(s, qrange)'s levels written into 'out' (uint8, s's shape), computed in place in s (float32, consumed)
    with quantize's own float32 operations in its order -- the same levels, without its compacted copies of the
    finite values."""
    qlo, qhi = (float(v) for v in qrange)
    if not qhi > qlo:
        qhi = qlo + 1.0
    bad = ~np.isfinite(s)
    with np.errstate(invalid="ignore", over="ignore"):
        np.subtract(s, qlo, out=s)
        np.divide(s, qhi - qlo, out=s)
        np.multiply(s, LEVELS - 2, out=s)
        np.rint(s, out=s)
        np.maximum(s, np.float32(0), out=s)         # np.clip's values (NaN kept), at a quarter of its cost
        np.minimum(s, np.float32(LEVELS - 2), out=s)
        np.add(s, 1.0, out=s)
    if bad.any():
        s[bad] = 0.0
    out[...] = s


def level_counts(q: np.ndarray) -> np.ndarray:
    """How many cells of q (uint8 levels, contiguous) hold each level: np.bincount's counts, taken over pairs of
    cells (one uint16 each: bincount widens what it counts to 8 bytes a value, so half the values is half the cost),
    a pair's two levels its low and high byte."""
    flat = q.reshape(-1)
    even = flat[:flat.size - flat.size % 2]
    pairs = np.bincount(even.view(np.uint16), minlength=65536).reshape(256, 256)    # [high byte, low byte]
    counts = pairs.sum(axis=0) + pairs.sum(axis=1)
    if flat.size % 2:
        counts[flat[-1]] += 1
    return counts


def levels_of(values: np.ndarray, log: bool, qrange: tuple, floor: float, progress: dict | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """The cube's levels (quantize of its scaled values over qrange) and their counts, in chunks on the pool."""
    q = np.empty(values.shape, dtype=np.uint8)
    spans = _cell_chunks(values.shape)
    counts = np.zeros((len(spans), LEVELS), dtype=np.int64)

    # a range float32 cannot hold: numpy 1.26 (value-based casting) did quantize's arithmetic in float64 then
    big = np.finfo(np.float32).max
    plain = max(abs(float(qrange[0])), abs(float(qrange[1]) - float(qrange[0]))) <= big

    def one(k):
        i0, i1 = spans[k]
        s = scaled(values[i0:i1], log, floor)                                  # scaled: a fresh array
        if plain:
            _quantize_into(s, qrange, q[i0:i1])
        else:
            q[i0:i1] = quantize(s, qrange)[0]
        counts[k] = level_counts(q[i0:i1])
    _run(one, range(len(spans)), progress=progress)
    return q, counts.sum(axis=0)


def transposed(q: np.ndarray, progress: dict | None = None) -> np.ndarray:
    """q's slices along its last axis made contiguous: out[k] = q[:, :, k] (in chunks on the pool), one x-layer's
    2D transpose at a time: 5x faster than the 3D one at 1024^3 (0.6 s against 3.0)."""
    out = np.empty((q.shape[2], q.shape[0], q.shape[1]), dtype=q.dtype)

    def one(span):
        for i in range(*span):
            out[:, i, :] = q[i].T
    _run(one, _chunks(q.shape[0], 8), progress=progress)
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
        np.take(lut, images[i0:i1], out=out[i0:i1], mode="clip")   # levels < 256 = len(lut): clip never acts
    _run(one, _chunks(images.shape[0], 8), gui_pool())
    return out


def percentile_level(counts: np.ndarray, levels: np.ndarray, pct: float) -> float:
    """The scaled value below which 'pct' percent of the cells with a value lie (from the level counts)."""
    c = np.cumsum(np.asarray(counts[1:], dtype=np.float64))
    if not c[-1] > 0:
        return float(levels[1])
    k = int(np.searchsorted(c, pct / 100.0 * c[-1]))
    return float(levels[1 + min(k, LEVELS - 2)])


def prepared_bytes(p: dict) -> int:
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
    if key is not None and not G.cubes_capped() and not (progress or {}).get("cancel"):
        with _LOCK:
            _PREPARED[key] = prep
            while len(_PREPARED) > 1 and sum(prepared_bytes(p) for p in _PREPARED.values()) > PREPARED_BYTES:
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
    # Design: the ramp is a power t^gamma. gamma ~ 1.6 keeps the walls just above the threshold faint and makes
    # filaments and haloes solid; gamma 1 (a linear ramp) draws walls and filaments alike, a foggier box, and a larger
    # gamma shows only the densest knots. A smoothstep (3t^2 - 2t^3), soft at both ends, was the other candidate.
    # Whatever the shape, it must be 0 at t = 0, rise to 1 at t = 1 and never fall (the tests check that it is
    # transparent below the threshold and monotone).
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
def slice_axis(cam: V.Camera, lo, hi, shape=None) -> tuple[int, np.ndarray]:
    """The slicing axis and the reference ray's direction (unit, toward the window): orthographic, the axis facing
    the viewer most (|cos| >= 1/sqrt 3: never edge on) and the view direction. Perspective: every pixel's ray has
    its own direction and its own axis (slice_plan's parts); these are the ray through the middle of the view and
    the axis whose slices it crosses most of ('shape': the cells along each axis; cubic cells without it)."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    t = cam.toward()
    if not cam.perspective:
        return int(np.argmax(np.abs(t))), -t
    cell = np.ones(3) if shape is None else (hi - lo) / np.asarray(shape, dtype=np.float64)
    return int(np.argmax(np.abs(t) / cell)), -t


def _slice_transforms(cam: V.Camera, lo, cell, shape, a: int) -> tuple[np.ndarray, tuple, np.ndarray]:
    """The (m, 9) image -> screen numbers of the m slices across axis 'a' (slice_plan's 'transforms'), the image's
    (width, height), and per slice whether any of it is in front of the eye (all, orthographic)."""
    r0, r1 = V.rest_axes(a)
    m = int(shape[a])
    pos = lo[a] + (np.arange(m) + 0.5) * cell[a]
    P = cam.matrix()
    ex, ey, step, origin = np.zeros(3), np.zeros(3), np.zeros(3), lo.copy()
    ex[r1], ey[r0], step[a] = cell[r1], cell[r0], cell[a]
    origin[a] = pos[0]
    hx, hy = P[:, :3] @ ex, P[:, :3] @ ey
    h0 = (P @ np.append(origin, 1.0))[None, :] + np.arange(m)[:, None] * (P[:, :3] @ step)[None, :]
    T = np.concatenate([np.broadcast_to(hx, (m, 3)), np.broadcast_to(hy, (m, 3)), h0], axis=1)
    size = (int(shape[r1]), int(shape[r0]))
    if not cam.perspective:
        # divided by |w| at each slice's origin: w keeps its sign (plane_transform's reason)
        return T / np.abs(h0[:, 2:3]), size, np.ones(m, dtype=bool)
    # Perspective: w is the depth in front of the eye, and with the eye inside the window a slice can run from in
    # front of it to behind it (Qt cuts the image where w reaches 0). Divided by the largest w over the slice's
    # four corners: positive whenever any of the slice is in front, and Qt's cut is then at a fixed fraction of it.
    w = h0[:, 2:3] + np.array([0.0, size[0] * hx[2], size[1] * hy[2], size[0] * hx[2] + size[1] * hy[2]])[None, :]
    far = w.max(axis=1)
    ahead = far > 1e-9 * cam.distance()
    return T / np.where(ahead, far, 1.0)[:, None], size, ahead


def slice_plan(cam: V.Camera, lo, hi, shape) -> dict:
    """How the slice stacks of a cube of 'shape' cells filling lo..hi (Mpc) land on the screen of 'cam':
    {'perspective', 'exponent' (the path through a slice over the reference thickness, along the reference ray),
    'parts': the stacks to draw, each {'axis', 'sign' (+1: its rays run up the axis), 'order' (slice indices,
    front first), 'transforms' ((m, 9): slice i's image pixels -> screen, view3d.plane_transform's numbers; image
    rows along the first remaining axis, columns along the second, as slice_images gives them), 'size' (the
    image's width, height), 'first'}, and 'axis', 'order', 'transforms', 'size' of the part that draws the middle
    of the view}.

    Orthographic: one part draws the whole picture (slice_axis's), exactly.

    Perspective: each pixel's ray has its own direction, and the eye may be anywhere, also inside the window. A
    pixel is drawn from the stack across the axis its ray crosses the most slices of (in cells: the largest
    component of its direction), so the picture has up to six parts (an axis and a sense each: the faces of a
    cube around the eye; part_ids gives each pixel's), no slice is seen at more than 55 degrees from face on, and
    only the slices ahead of the eye along a part's sense are drawn, nearest first. Where two parts meet, a ray
    crosses as many slices of the one as of the other, so one exponent serves all: the reference ray's (through
    the middle of the view). A ray's true path through a slice is the spacing over its own |cos|, so a uniform
    field's optical depth varies across the picture by that ratio to the reference ray's (at most sqrt 3) -- 2D
    texture slicing's inherent error. 'first' = (slice, weight): the nearest slice ahead of the eye stands for the
    stretch between the eye and itself, so it counts by its distance in slices (weight < 1 within one slice of the
    eye): flying through the field, a slice fades as the eye reaches it and never pops. 'view' = (q0, qx, qy, eye):
    screen point (x, y) lies at q0 + x qx + y qy in cells from 'lo' on the target's plane, the eye at 'eye';
    'screen' = the view's size (cam.width, cam.height)."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    shape = tuple(int(n) for n in shape)
    cell = (hi - lo) / np.asarray(shape, dtype=np.float64)
    a, ray = slice_axis(cam, lo, hi, shape)
    ref = float(np.max(hi - lo)) / REF_CELLS
    plan = {"axis": a, "perspective": bool(cam.perspective),
            "exponent": float(cell[a] / max(abs(ray[a]), 1e-9) / ref) if ref > 0 else 1.0}
    if not cam.perspective:
        T, size, _ = _slice_transforms(cam, lo, cell, shape, a)
        m = shape[a]
        order = np.arange(m) if ray[a] > 0 else np.arange(m)[::-1]  # the viewer on the high side: high slices first
        plan["parts"] = [{"axis": a, "sign": 1 if ray[a] > 0 else -1, "order": order, "transforms": T, "size": size,
                          "first": None}]
    else:
        eye = (cam.eye() - lo) / cell
        corner = cam.target + (-0.5 * cam.width * cam.R[0] + 0.5 * cam.height * cam.R[1]) / cam.scale
        plan["view"] = ((corner - lo) / cell, cam.R[0] / (cam.scale * cell), -cam.R[1] / (cam.scale * cell), eye)
        plan["screen"] = (int(cam.width), int(cam.height))
        plan["parts"] = []
        for b in range(3):
            T, size, ahead = _slice_transforms(cam, lo, cell, shape, b)
            gap = np.arange(shape[b]) + 0.5 - eye[b]                # each slice's distance up the axis, in slices
            for sign in (1, -1):
                front = np.nonzero(sign * gap > 0)[0][::sign]       # nearest the eye first
                if front.size:
                    first = (int(front[0]), float(min(1.0, sign * gap[front[0]])))
                    plan["parts"].append({"axis": b, "sign": sign, "order": front[ahead[front]], "transforms": T,
                                          "size": size, "first": first})
    mid = [q for q in plan["parts"] if q["axis"] == a and q["sign"] == (1 if ray[a] > 0 else -1)]
    if mid:
        plan.update(order=mid[0]["order"], transforms=mid[0]["transforms"], size=mid[0]["size"])
    else:                                                           # the middle of the view looks away from the window
        r0, r1 = V.rest_axes(a)
        plan.update(order=np.zeros(0, dtype=np.intp), transforms=np.zeros((shape[a], 9)), size=(shape[r1], shape[r0]))
    return plan


def part_ids(plan: dict, width: int, height: int) -> np.ndarray | None:
    """Which of slice_plan's parts draws each pixel of a width x height picture of the plan's whole view: (height,
    width) int8 of 2 axis + (0: the ray runs up the axis, 1: down it), at the pixels' centres; None: orthographic
    (the one part draws all). Ties go to the lower axis (volume_gl's shader takes the same)."""
    if not plan["perspective"]:
        return None
    q0, qx, qy, eye = plan["view"]
    x = (np.arange(width) + 0.5) * (plan["screen"][0] / width)
    y = (np.arange(height) + 0.5) * (plan["screen"][1] / height)
    D = [(q0[k] - eye[k]) + x[None, :] * qx[k] + y[:, None] * qy[k] for k in range(3)]
    A = [np.abs(v) for v in D]
    out = np.where(A[1] >= A[2], np.int8(2), np.int8(4)) + (np.where(A[1] >= A[2], D[1], D[2]) < 0)
    first = (A[0] >= A[1]) & (A[0] >= A[2])
    out[first] = (D[0] < 0)[first]
    return out.astype(np.int8, copy=False)


def part_id(part: dict) -> int:
    """A part's number in part_ids."""
    return 2 * int(part["axis"]) + (1 if part["sign"] < 0 else 0)


def composite(rgba: np.ndarray) -> np.ndarray:
    """The reference compositing of premultiplied RGBA samples along a ray, front first ((k, 4) floats in 0..1, or
    (..., k, 4)): C = sum_i c_i prod_{j < i} (1 - a_j), the 'under' operator QPainter's DestinationOver applies
    slice by slice. The tests hold the picture to it."""
    x = np.asarray(rgba, dtype=np.float64)
    trans = np.cumprod(1.0 - x[..., 3], axis=-1)
    before = np.concatenate([np.ones(trans.shape[:-1] + (1,)), trans[..., :-1]], axis=-1)
    return (x * before[..., None]).sum(axis=-2)
