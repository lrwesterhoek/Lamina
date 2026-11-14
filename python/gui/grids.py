"""The data side of the GUI's Explore tab. Pure numpy (no Qt): tests/py_gui_runspec_test.py.

  OutputSet(root)          one run's grid files '<dir>/<prefix>.[a_]<suffix>': which fields exist,
                           the grid size, one slice of any field (read through a memmap, so a
                           1024^3 cube costs one plane of memory), a value at a cell. A 2D run's
                           grids (n^2 cells) are one plane: its slice is the whole grid
  find_outputs(...)        every output set under the data root and the custom-snapshot output directories
  colorize(plane, ...)     a 2D array -> RGB bytes with a matplotlib colormap (percentile range,
                           optional log10), plus the value range used
  server_settings(out)     how to start a dtfelib.Estimator on the snapshot behind an output (the
                           Explore tab's click-to-query), from the TNG layout or the custom-snapshot
                           job's settings sidecar ('<prefix>.gui.json')
  field_label(o, name, ..) a map's name for a field, the velocity derivatives' family said (a phase-space
                           output's divergence, shear, vorticity, gradient and V-web as the mean field grad<v> --
                           a face-gradient output's stored grids, else computed slice by slice from its velocity
                           grid -- or as its stored stream average <grad v>)
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from dtfelib.cli import PS_GRADIENT_CHOICES  # noqa: E402,F401  (the figure scripts' '--ps-gradient' choices)
from dtfelib.io import (FIELDS as _IO_FIELDS, LEGACY_SUFFIX, MEAN_FIELD_NAMES, _VELOCITY_SCALE_EXP,  # noqa: E402
                        caustic_is_fold, run_lambda_th)

# the launcher's field table: dtfelib's + the per-particle scalar a run with a scalar dataset writes ('.a_scalar',
# '--field scalar_a'; dtfelib's analysis never reads it, the Explore map can)
FIELDS = {**_IO_FIELDS, "scalar": ("scalar", 1, False)}

# the two families of a phase-space output's velocity derivatives, as a map's name says them (field_label)
GRADIENT_QUALIFIER = {"mean": "mean field ∇⟨v⟩", "stream": "stream average ⟨∇v⟩"}
# the fields worth a map, in menu order, with their labels. A face-gradient run writes the stream average of each
# stream's own gradient beside its mean-flow family (src/input_output.cc: '.a_velDivStreams' always, and
# '.a_velGradStreams', '.a_velShearStreams', '.a_velVortStreams' where 'gradient_a', 'shear_a', 'vorticity_a' are
# written; docs/cleanup-plan.md D12), and the mass flux divergence ('.a_massFluxDiv'): listed after the family, the
# twins named as the stream average a map of the family's own grid says (field_label)
FIELD_LABELS = {
    "density": "density ρ/ρ̄", "streams": "stream count", "hidden_streams": "hidden streams",
    "velocity": "velocity", "divergence": "velocity divergence", "vorticity": "vorticity",
    "shear": "shear", "gradient": "velocity gradient",
    "divergence_streams": f"velocity divergence ({GRADIENT_QUALIFIER['stream']})",
    "vorticity_streams": f"vorticity ({GRADIENT_QUALIFIER['stream']})",
    "shear_streams": f"shear ({GRADIENT_QUALIFIER['stream']})",
    "gradient_streams": f"velocity gradient ({GRADIENT_QUALIFIER['stream']})",
    "mass_flux_divergence": "mass flux divergence", "dispersion": "velocity dispersion",
    "dispersion_tensor": "dispersion tensor", "caustic": "caustic (fold) cells",
    "caustic_class": "caustic class", "tet_touch": "tetrahedra touching",
    "tweb": "T-web class", "tweb_eigenvalues": "T-web eigenvalues", "vweb": "V-web class",
    "vweb_eigenvalues": "V-web eigenvalues", "scalar": "per-particle scalar",
}
# what a map's values mean, where the label cannot say it (the field list's tooltips)
FIELD_HELP = {
    "streams": "How many streams overlap the cell: 1 = single-stream, 3, 5, … in folds.",
    "hidden_streams": "1 = the cell holds multi-stream volume its stream count misses (a fold "
                      "thinner than the sampling).\n2 = a tetrahedron too small to be sampled "
                      "left its mass here (usually of the same stream).\n3 = both.",
    "caustic": "1 = a fold caustic crosses the cell: a folded tetrahedron overlaps it (stream count above 1, or "
               "hidden-streams bit 1). Exact for every periodic phase-space run, also those made before "
               "2026-10-13, whose stored caustic grid missed the folds thinner than the sampling.",
    "scalar": "The particles' own value (the run's scalar dataset, e.g. a potential), interpolated like the density.",
    **{f"{n}_streams": f"The stream average of each stream's own {what} (from ⟨∇v⟩), written beside an Exact "
                       f"run's {what} of the mean flow (from ∇⟨v⟩, exact face sums): the grid a Fast run stores as "
                       f"its {what}. The two agree in single-stream cells."
       for n, what in (("divergence", "divergence"), ("vorticity", "vorticity"), ("shear", "shear"),
                       ("gradient", "velocity gradient"))},
    "mass_flux_divergence": "∇·(Σ ρ_s v_s), the streams' momentum (ρ/ρ̄ times velocity) out of the cell per volume, "
                            "from exact face sums: by continuity the rate the cell loses mass (an Exact run in 3D).",
}
# The velocity derivatives (dtfelib.io.MEAN_FIELD_NAMES). Standard DTFE stores them as the derivatives of
# its one velocity field, the mean field grad<v>; so does a PS-DTFE run with the face gradient (its default with the
# exact deposit and the volume weighting, docs/cleanup-plan.md D13; OutputSet.face_gradient). A PS-DTFE run without it
# stores, in a multi-stream cell, the stream average of each stream's own gradient <grad v> (dtfelib/io.py, the note
# above FIELDS: a divergence whose box mean is not 0, a V-web with almost no filaments) -- so a map of such a PS
# divergence beside a standard one compares two quantities. Explore shows that phase-space output's as either
# ('PS velocity derivatives'): 'mean-field', computed from its velocity grid
# one slice at a time (dtfelib.fields.mean_field_gradient_plane, the figure scripts' spectral derivative and their
# default, '--ps-gradient' of dtfelib.cli), or 'stored'. Every map, title, colour bar and figure-grid panel says
# which (field_label).
GRADIENT_FIELDS = frozenset(MEAN_FIELD_NAMES)
PS_GRADIENT_DEFAULT = "mean-field"
# the vorticity grids: the binary stores the antisymmetric part of a gradient, shown as its curl (component_values)
CURL_FIELDS = frozenset({"vorticity", "vorticity_streams"})
# vector fields that can be drawn as arrows (their in-plane components) over the colour map
VECTOR_FIELDS = {"velocity"} | CURL_FIELDS
# which component menu a field offers; 'norm' = the vector's length
COMPONENTS = {3: ["norm", "x", "y", "z"], 5: ["1", "2", "3", "4", "5"], 6: ["xx", "xy", "xz", "yy", "yz", "zz"],
              9: ["trace (divergence)", "xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]}
# A 2D build's grids (NO_DIM = 2, define.h): the fields whose component count differs from 3D --
# velocity 2, the gradient 2x2, the symmetric dispersion tensor (xx, xy, yy), the traceless shear
# (xx, xy), the eigenvalue pairs; the vorticity is one number (the z component of the curl)
NCOMP_2D = {"velocity": 2, "gradient": 4, "dispersion_tensor": 3, "shear": 2, "vorticity": 1,
            "tweb_eigenvalues": 2, "vweb_eigenvalues": 2}
COMPONENTS_2D = {"velocity": ["norm", "x", "y"], "gradient": ["trace (divergence)", "xx", "xy", "yx", "yy"],
                 "dispersion_tensor": ["xx", "xy", "yy"]}


def sym_diag(dim: int) -> list[int]:
    """Where the diagonal sits in a packed symmetric tensor (xx, xy, xz, yy, yz, zz / xx, xy, yy)."""
    return [i * dim - i * (i - 1) // 2 for i in range(dim)]


def ncomp_of(name: str, dim: int = 3) -> int:
    """The components per cell of a field's grid in a 'dim'-dimensional run."""
    return NCOMP_2D.get(name, FIELDS[name][1]) if dim == 2 else FIELDS[name][1]


def components(name: str, ncomp: int, dim: int = 3) -> list[str]:
    """The component menu of a field: ['value'] for a scalar; 'norm' / 'trace ...' first when the
    field has one (they are computed, not stored)."""
    if ncomp <= 1:
        return ["value"]
    if dim == 2:
        return COMPONENTS_2D.get(name, [str(i + 1) for i in range(ncomp)])
    return COMPONENTS.get(ncomp, [str(i + 1) for i in range(ncomp)])


def grid_shape(path: Path, ncomp: int, dim: int = 3) -> tuple[int, np.dtype]:
    """(N, dtype) of a raw N^dim x ncomp grid file, from its size: float32 or float64 (a DOUBLE=1
    build). dtfelib's _grid_shape_of for any dimension; ValueError when the size fits neither."""
    size = Path(path).stat().st_size
    for dt in (np.float32, np.float64):
        item = np.dtype(dt).itemsize
        n = round((size / item / ncomp) ** (1 / dim))
        if n > 0 and n ** dim * ncomp * item == size:
            return n, np.dtype(dt)
    raise ValueError(f"{path}: {size} bytes is not an N^{dim} x {ncomp} float32 or float64 grid")


def _fits(files: dict, dim: int) -> tuple[int, int | None, np.dtype | None]:
    """(how many of an output's files are N^dim grids of one N, that N, its dtype), N from the
    first file whose size fits."""
    n = dt = None
    for name, path in files.items():
        try:
            n, dt = grid_shape(path, ncomp_of(name, dim), dim)
            break
        except ValueError:
            continue
    if n is None:
        return 0, None, None
    hits = sum(1 for name, path in files.items()
               if path.stat().st_size == n ** dim * ncomp_of(name, dim) * dt.itemsize)
    return hits, n, dt


def _look_like_stream_counts(path: Path, dtype) -> bool:
    """Do a stream-count grid's first values, read as 'dtype', look like stream counts -- whole numbers
    >= 1 in most cells (dtfelib.STREAM_TOL)? Read in the other precision they do not: float32 1.0
    pairs read as float64 are ~0.0078, a float64 1.0 read as float32 is 0 and 1.875."""
    try:
        raw = np.fromfile(path, dtype=dtype, count=65536 // np.dtype(dtype).itemsize)
    except (OSError, ValueError):
        return False
    if raw.size == 0:
        return False
    whole = np.rint(raw)
    return float(np.mean(np.isfinite(raw) & (whole >= 1) & (np.abs(raw - whole) < 1e-3))) > 0.5


def _ambiguous_dim(root: Path, plane: tuple[int, np.dtype], cube: tuple[int, np.dtype]) -> int:
    """The dimension of an output whose files read equally well as an n x n plane and an m^3 cube,
    plane = (n, dtype), cube = (m, dtype): only scalar grids (density, stream counts) and no
    settings sidecar saying which. Same bytes, two readings: n^2 = m^3 in one precision (512^2 =
    64^3, 4096^2 = 256^3), or a float32 plane against a float64 cube, n^2 = 2 m^3 (256^2 = 32^3 in
    double), and the reverse (128^2 in double = 32^3 in float)."""
    if plane[1] != cube[1]:             # the readings differ in precision: a phase-space run's stream
        for sfx in ("a_streams", "streams"):            # counts are whole numbers in the right one only
            p = root.parent / f"{root.name}.{sfx}"
            if p.is_file():
                as_plane, as_cube = _look_like_stream_counts(p, plane[1]), _look_like_stream_counts(p, cube[1])
                if as_plane != as_cube:
                    return 2 if as_plane else 3
                break
        return 3                        # no evidence: the 3D reading
    # One precision (a 512^2 plane or a 64^3 cube): neighbouring cells of a real grid are alike. Read flat,
    # the cube's second axis steps m values and the plane's rows step n; in the wrong reading that stride
    # lands far away (m cells along a row of the plane, or sqrt(m) cells along the cube's second axis),
    # so the true neighbour stride shows clearly the smaller differences. A field without structure
    # across them (constant, noise, one varying along the slowest axis only) cannot tell: 3D.
    # Reads the first 64 rows of the plane at most (1 MB for 4096^2 in float32).
    path = next((p for p in (root.parent / f"{root.name}.{sfx}" for sfx in ("a_den", "den")) if p.is_file()), None)
    n, m = int(plane[0]), int(cube[0])
    if path is None or n <= m:
        return 3
    try:
        x = np.fromfile(path, dtype=plane[1], count=min(n * n, 64 * n)).astype(np.float64)
    except (OSError, ValueError):
        return 3
    if x.size <= n:
        return 3
    with np.errstate(invalid="ignore"):
        across_cube = np.abs(x[m:] - x[:-m])
        across_plane = np.abs(x[n:] - x[:-n])
    d_cube = float(np.nanmedian(across_cube)) if np.isfinite(across_cube).any() else 0.0
    d_plane = float(np.nanmedian(across_plane)) if np.isfinite(across_plane).any() else 0.0
    return 2 if d_plane < 0.8 * d_cube else 3
# maps that are best on a log scale by default / diverging maps / category maps
LOG_DEFAULT = {"density", "dispersion", "tet_touch"}
DIVERGING = {"divergence", "vorticity", "gradient", "shear", "tweb_eigenvalues", "vweb_eigenvalues",
             "divergence_streams", "vorticity_streams", "gradient_streams", "shear_streams", "mass_flux_divergence"}
CATEGORICAL = {"streams", "hidden_streams", "caustic", "caustic_class", "tweb", "vweb"}


@dataclass
class FieldFile:
    name: str                   # dtfelib field name
    path: Path
    ncomp: int
    averaged: bool


# whole fields up to this size are read into memory once (a strided z-plane read touches every
# page of the file: 1.9 s for a 512^3 grid on an external disk, every time); bigger ones go
# through a memmap per slice. The cache holds a quarter of the machine's memory, at most 16 GB
# (DTFE_GUI_CUBE_GB overrides): the figure grid shows four or nine outputs at once, and at
# 1.5 GB four 512^3 cubes evict each other, so every slice step re-reads them all (75 ms each
# instead of 1 ms).


def _cube_cache_bytes() -> int:
    env = os.environ.get("DTFE_GUI_CUBE_GB", "")
    if env:
        try:
            return int(float(env) * 1e9)
        except ValueError:
            pass
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = 8e9
    return int(min(max(0.25 * ram, 1.5e9), 16e9))


CUBE_CACHE_BYTES = _cube_cache_bytes()
# A cube above this is read whole only when a slice runs along its last axis (a memmap touches every page
# of the file for that one anyway) or once it is cached already: a single x or y plane of a 1024^3 field
# must not cost the whole 4 GB read (the 16 GB cap alone would make it so).
CUBE_EAGER_BYTES = min(int(2e9), CUBE_CACHE_BYTES)
# module-wide, least recently used first; the key carries the file's mtime and size, so a grid written
# again by a re-run is read again (keyed on the path alone, Explore would keep showing the old run's values
# until the launcher restarted). The slice worker and the GUI thread (Save figure...)
# both come here: every look at the dict is under the lock, the disk read is not (drop_cubes() from the
# Run button must never wait on a slow external disk).
_CUBES: dict[tuple, np.ndarray] = {}
_CUBES_LOCK = threading.Lock()
# While the launcher runs a job the cache keeps at most RUN_CUBE_BYTES: the binary's auto-tuner budgets from the
# memory free at ITS start, and Explore stepping through slices meanwhile must not refill 16 GB under it (a
# 512^3 float32 cube, 0.5 GB, still fits: stepping stays smooth). cap_cubes() is the launcher's switch.
RUN_CUBE_BYTES = int(1e9)
_CAPPED = False


def cap_cubes(on: bool) -> None:
    """The run-time cap on (a job starts: the cache is trimmed to it) or off (the job ended)."""
    global _CAPPED
    with _CUBES_LOCK:
        _CAPPED = bool(on)
        _make_room(0, _CUBES, _SMOOTH3D)
        while _MF_PLANES and sum(a.nbytes for a in _MF_PLANES.values()) > _mf_budget():
            _MF_PLANES.pop(next(iter(_MF_PLANES)))


def cubes_capped() -> bool:
    """True while the launcher runs a job (cap_cubes): the caches built beside the cube cache (the 3D view's) keep
    nothing then either."""
    return _CAPPED


def _cube_budget() -> int:
    return min(CUBE_CACHE_BYTES, RUN_CUBE_BYTES) if _CAPPED else CUBE_CACHE_BYTES


def _make_room(nbytes: int, first: dict, second: dict) -> None:
    """Under _CUBES_LOCK: the least recently used cubes dropped, those of 'first' before those of 'second', until
    'nbytes' more fit in the one budget the grids' cubes (_CUBES) and the cubes smoothed in 3D (_SMOOTH3D) share."""
    budget = _cube_budget()
    for d in (first, second):
        while d and sum(a.nbytes for a in _CUBES.values()) + sum(a.nbytes for a in _SMOOTH3D.values()) \
                + nbytes > budget:
            d.pop(next(iter(d)))


def cube_key(path: Path, dtype, shape) -> tuple | None:
    """A grid file's identity in the caches: its resolved path, mtime, size, dtype and shape (None: unreadable)."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (path.resolve(), st.st_mtime_ns, st.st_size, np.dtype(dtype).str, tuple(int(v) for v in shape))


def cached_cube(path: Path, dtype, shape, axis: int | None = None, read: bool = True) -> np.ndarray | None:
    """The whole array of a grid file, cached, or None when the caller should memmap it instead: it is
    larger than the cache, or larger than CUBE_EAGER_BYTES and not cached yet while 'axis' (the plane
    axis the caller slices next, 0 or 1 of a 3D grid) is one a memmap reads cheaply. axis None = the
    caller reads the whole array (a 2D grid) or slices along the last axis. read False = only a cube cached
    already, never a new read (the fold map's planes)."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    budget = _cube_budget()
    if nbytes > budget:
        return None
    key = cube_key(path, dtype, shape)
    if key is None:
        return None
    with _CUBES_LOCK:
        cube = _CUBES.get(key)
        if cube is not None:
            _CUBES[key] = _CUBES.pop(key)     # most recently used last
            return cube
    if not read:
        return None
    if nbytes > min(CUBE_EAGER_BYTES, budget) and axis is not None and axis < 2:
        return None
    cube = np.fromfile(path, dtype=dtype).reshape(shape)         # outside the lock: nobody waits on the disk
    with _CUBES_LOCK:
        other = _CUBES.get(key)
        if other is not None:                 # another thread read it meanwhile: one copy, theirs
            _CUBES[key] = _CUBES.pop(key)
            return other
        for stale in [k for k in _CUBES if k[0] == key[0]]:      # the same file, another version: drop it
            _CUBES.pop(stale)
        if nbytes > _cube_budget():           # the cap came on while this was read (a job started): use it, keep none
            return cube
        _make_room(nbytes, _CUBES, _SMOOTH3D)  # the grids' cubes first, then the ones smoothed in 3D
        _CUBES[key] = cube
    return cube


_cached_cube = cached_cube              # the name the tests reach


def _read_plane(path: Path, dtype, n: int, ncomp: int, axis: int, index: int) -> np.ndarray:
    """One plane of an (n, n, n[, ncomp]) C-order grid file by plain reads (no memmap: see OutputSet._memmap):
    axis 0 is one contiguous block, axis 1 one block per row, axis 2 the file in slabs of whole x-layers (the
    same bytes a memmap would touch), keeping one plane. Raises ValueError for a short file, OSError when
    the file cannot be read (a drive gone)."""
    dt = np.dtype(dtype)
    if not 0 <= index < n:
        raise ValueError(f"plane {index} outside 0..{n - 1}")
    row = n * ncomp                         # values per (i, j) row along the last axis
    layer = n * row                         # values per x-layer i
    out = np.empty((n, n, ncomp) if ncomp > 1 else (n, n), dtype=dt)
    with open(path, "rb") as f:
        def block(offset_values: int, count: int) -> np.ndarray:
            f.seek(offset_values * dt.itemsize)
            a = np.fromfile(f, dtype=dt, count=count)
            if a.size != count:
                raise ValueError(f"{Path(path).name} is shorter than its grid (an interrupted write?)")
            return a
        if axis == 0:
            return block(index * layer, layer).reshape(out.shape)
        if axis == 1:
            for i in range(n):
                out[i] = block(i * layer + index * row, row).reshape(out.shape[1:])
            return out
        step = max(1, int(64e6 // (layer * dt.itemsize)))      # ~64 MB of x-layers per read
        for i0 in range(0, n, step):
            k = min(step, n - i0)
            slab = block(i0 * layer, k * layer).reshape((k, n, n, ncomp) if ncomp > 1 else (k, n, n))
            out[i0:i0 + k] = slab[:, :, index]
        return out


# The other caches of grid data in this process (the 3D view's averaged and prepared cubes: volume3d registers its
# drop_volumes at import): drop_cubes() empties them too, so a job start or a memory check cannot miss one.
_DROP_CALLBACKS: list = []


def on_drop_cubes(fn) -> None:
    """Register fn() -> bytes freed, called by every drop_cubes() (once per function, however often registered)."""
    if fn not in _DROP_CALLBACKS:
        _DROP_CALLBACKS.append(fn)


def drop_cubes() -> int:
    """Forget every cached cube and mean-field plane, and every cache registered with on_drop_cubes (the bytes they
    held). The launcher calls it when a job or a memory check starts: the caches are anonymous memory of the GUI
    process, and the binary's auto-tuner budgets from the memory that is free at that moment. What is on screen stays:
    the map's plane, and the 3D view's volume and its GPU copy (View3D holds them, not the caches: volume3d's
    drop_volumes says why; the 3D view's own memory check, its detail choice, counts them)."""
    with _CUBES_LOCK:
        n = sum(a.nbytes for a in _CUBES.values()) + sum(a.nbytes for a in _MF_PLANES.values())
        _CUBES.clear()
        _MF_PLANES.clear()
    for fn in list(_DROP_CALLBACKS):    # outside the lock: each cache takes its own
        n += int(fn() or 0)
    return n


# The mean-field velocity gradient of the planes shown lately: (velocity file's cube_key, axis, index,
# box) -> float64 (N, N, 3, 3), least recently used first, under _CUBES_LOCK. A plane costs one pass over the velocity
# grid (0.3-0.6 s at 512^3 from the cube cache, a read of its 1.6 GB file otherwise); every derivative of it (the
# divergence, shear, vorticity, gradient, V-web) and every colour or range change then comes from here. At most a
# quarter of the cube cache's budget, 256 MB (13 planes of 512^2, 3 of 1024^2): the grid's panels fit.
_MF_PLANES: dict[tuple, np.ndarray] = {}
MF_PLANE_BYTES = int(256e6)


def _mf_budget() -> int:
    return min(MF_PLANE_BYTES, _cube_budget() // 4)


class _SlabFile:
    """An (n, ...) grid file read in slabs along its first axis by plain reads (no memmap, OutputSet._memmap's rule):
    what dtfelib.fields.mean_field_gradient_plane takes, .shape, .dtype and [s:e]. A short file raises ValueError, a
    drive gone OSError (the slice worker reports either)."""

    def __init__(self, path: Path, dtype, shape):
        self.path, self.dtype, self.shape = Path(path), np.dtype(dtype), tuple(int(v) for v in shape)

    def __getitem__(self, sl: slice) -> np.ndarray:
        s, e, _step = sl.indices(self.shape[0])
        per = int(np.prod(self.shape[1:]))
        with open(self.path, "rb") as f:
            f.seek(s * per * self.dtype.itemsize)
            a = np.fromfile(f, dtype=self.dtype, count=(e - s) * per)
        if a.size != (e - s) * per:
            raise ValueError(f"{self.path.name} is shorter than its grid (an interrupted write?)")
        return a.reshape((e - s,) + self.shape[1:])


class OutputSet:
    """The grid files of one run: '<dir>/<prefix>.*'. 'dim' 2 or 3, else the settings sidecar's,
    else read off the file sizes (a 2D grid is n^2 cells, its vectors and tensors have fewer
    components: a velocity file alone tells 2D from 3D)."""

    def __init__(self, root, box: tuple | None = None, redshift: float | None = None, title: str = "",
                 dim: int | None = None, sim: str = "", snap: int | None = None):
        self.root = Path(root)
        self.dir, self.prefix = self.root.parent, self.root.name
        self.title = title or f"{self.dir.name}/{self.prefix}"
        self.sim, self.snap = sim, snap          # the TNG layout's simulation and snapshot number ("" / None otherwise)
        found: dict[str, tuple[Path, bool]] = {}
        for name, (suffix, _ncomp, _ps) in FIELDS.items():
            for sfx in (suffix, LEGACY_SUFFIX.get(name)):   # the current name, then a renamed one
                for averaged in (True, False):              # prefer the volume-averaged grid
                    p = self.dir / f"{self.prefix}.{'a_' if averaged else ''}{sfx}"
                    if sfx and p.is_file() and name not in found:
                        found[name] = (p, averaged)
        paths = {name: p for name, (p, _a) in found.items()}
        if dim not in (2, 3):
            side = sidecar(self.root) or {}
            dim = side.get("dim") if side.get("dim") in (2, 3) else None
        fit3, n3, dt3 = _fits(paths, 3)
        fit2, n2, dt2 = _fits(paths, 2)
        if dim not in (2, 3):
            dim = 3 if fit3 > fit2 else 2 if fit2 > fit3 else \
                (_ambiguous_dim(self.root, (n2, dt2), (n3, dt3)) if fit3 else 3)
        self.dim = int(dim)
        self.n, self.dtype = (n3, dt3) if self.dim == 3 else (n2, dt2)
        self.files: dict[str, FieldFile] = {name: FieldFile(name, p, ncomp_of(name, self.dim), averaged)
                                            for name, (p, averaged) in found.items()}
        # a file whose size does not match the grid (an interrupted write) is left out
        if self.n:
            self.files = {name: ff for name, ff in self.files.items()
                          if ff.path.stat().st_size == self.n ** self.dim * ff.ncomp * self.dtype.itemsize}
        else:
            self.files = {}
        self.box = box                  # ((xlo, ylo[, zlo]), (xhi, yhi[, zhi])) in Mpc, or None
        self.redshift = redshift
        self._fold_exact: bool | None = None
        self._mf_reason: str | None = None      # mean_field_reason(), once known

    def __bool__(self):
        return bool(self.files) and bool(self.n)

    def fields(self) -> list[str]:
        return [n for n in FIELD_LABELS if n in self.files]

    # lo and length always have three entries: a 2D grid is the plane z = 0 of zero thickness, so the
    # slice machinery (an axis, an index along it) treats it as the one z slice it has
    @property
    def lo(self) -> np.ndarray:
        if self.box is None:
            return np.zeros(3)
        lo = np.asarray(self.box[0], float)
        return np.append(lo, 0.0) if lo.size == 2 else lo

    @property
    def length(self) -> np.ndarray:
        if self.box is None:
            out = np.full(3, float(self.n or 1))        # unknown box: cell units
        else:
            out = np.asarray(self.box[1], float) - np.asarray(self.box[0], float)
            if out.size == 2:
                out = np.append(out, 0.0)
        if self.dim == 2:
            out[2] = 0.0
        return out

    def _memmap(self, name: str, axis: int | None = None, index: int = 0, cache: bool = True) -> np.ndarray:
        """The field's array -- the cached cube when cached_cube() keeps one -- or, for a 3D grid, just the
        plane 'index' along 'axis', read from the file. NEVER a memmap: a mapped page of a drive unplugged
        meanwhile (an external disk) kills the whole launcher with SIGBUS, which Python cannot catch; a read fails with
        an OSError the slice worker reports. A 2D grid is read whole. cache False: a cube cached
        already, else the plane; no new cube read."""
        ff = self.files[name]
        shape = (self.n,) * self.dim + ((ff.ncomp,) if ff.ncomp > 1 else ())
        cube = cached_cube(ff.path, self.dtype, shape, axis if self.dim == 3 else None, read=cache)
        if cube is not None:
            return cube
        if self.dim == 2 or axis is None:
            return np.fromfile(ff.path, dtype=self.dtype).reshape(shape)
        return _read_plane(ff.path, self.dtype, self.n, ff.ncomp, int(axis), int(index))

    def _nbytes(self, name: str) -> int:
        """The bytes of a field's whole grid file."""
        return self.n ** self.dim * self.files[name].ncomp * self.dtype.itemsize

    def velocity_factor(self, name: str) -> float:
        """The factor from a field's stored units to the map's: u-units -> peculiar km/s (FieldSet) for the
        velocity and its derivatives at a known redshift, else 1."""
        exp = _VELOCITY_SCALE_EXP.get(name)
        if exp is None or self.redshift is None:
            return 1.0
        return (1.0 / (1.0 + self.redshift)) ** exp       # u-units -> peculiar km/s (FieldSet)

    _velocity_factor = velocity_factor          # the name the tests reach

    def _raw_plane(self, name: str, axis: int, index: int, cache: bool = True) -> np.ndarray:
        """One plane of a field as stored (the file's precision, components last); a 2D grid: the whole grid.
        cache False: no new cube read into the cache (_memmap)."""
        mm = self._memmap(name, int(axis), index, cache)
        if self.dim == 2:
            return np.array(mm)             # the file's precision: only the component shown is up-cast (slice)
        if mm.ndim == self.dim + (1 if self.files[name].ncomp > 1 else 0):     # the cached cube: cut the plane
            sl = [slice(None)] * 3
            sl[axis] = index
            return np.array(mm[tuple(sl)])
        return mm                           # already the plane (read from the file)

    def fold_exact(self) -> bool:
        """The 'caustic' map is the exact fold flag: a phase-space output with a streams grid, in a
        periodic box (every TNG run; a custom run's settings sidecar says). dtfelib.io.caustic_is_fold: streams > 1
        or '.hidden_streams' bit 1 (OR the class mask's fold bits) -- what '.caustic' stores; computing it here also
        covers the grids of older binaries, whose '.caustic' is set only where a folded tetrahedron deposited a
        SAMPLE (21.0% of the cells of TNG100-3-Dark z=1 against 33.6%). In a
        non-periodic run the union over-flags (a sheet's edge overlaps without folding): the stored grid is shown."""
        if self._fold_exact is None:
            self._fold_exact = "streams" in self.files and bool((sidecar(self.root) or {}).get("periodic", True))
        return self._fold_exact

    def _fold_plane(self, axis: int, index: int) -> np.ndarray:
        """The exact fold flag on one plane, 0/1 float64 (fold_exact), from up to three fields (the class mask, the
        streams, the hidden streams). When the three cubes together do not fit in the cube cache -- while a job runs
        it keeps 1 GB, and three 512³ cubes are 1.5 GB -- only their planes are read (a cube cached already is still
        used): read whole, 0.5 GB each, they would evict each other, so every step of the slice would read 1.5 GB
        into memory beside the run the cap is there for."""
        names = [n for n in ("caustic_class", "streams", "hidden_streams") if n in self.files]
        cache = sum(self._nbytes(n) for n in names) <= _cube_budget()

        def get(n):
            return self._raw_plane(n, axis, index, cache) if n in self.files else None
        fold = caustic_is_fold(get("caustic_class"), streams=get("streams"), hidden=get("hidden_streams"))
        return np.asarray(fold, dtype=np.float64)

    def mean_field_reason(self) -> str:
        """'' when this output's velocity derivatives can be shown as the mean field grad<v>, else why
        not, in plain words: a standard-DTFE output stores the mean field itself; a phase-space one needs its velocity
        grid, a 3D grid of a known cubic box, and a periodic one -- the spectral derivative wraps around the box, so
        not a non-periodic run nor a window of the grid (a custom run's settings sidecar says). Cached on the object
        (a refresh of the outputs makes new ones)."""
        if self._mf_reason is None:
            side = sidecar(self.root) or {}
            self._mf_reason = (
                "standard DTFE stores the mean field" if estimator_label(self) != "PS-DTFE" else
                "a 2D grid: the mean field is computed for 3D grids only" if self.dim != 3 else
                "no velocity grid to compute it from" if "velocity" not in self.files else
                "not a periodic box" if not side.get("periodic", True) else
                "a window of the grid, not the whole periodic box" if side.get("window") else
                "the box is not known" if self.box is None else
                "the box is not a cube" if float(np.ptp(self.length)) > 1e-9 * float(np.max(self.length)) else "")
        return self._mf_reason

    @property
    def face_gradient(self) -> bool:
        """A phase-space output whose run took the face gradient (PS-DTFE's default with the exact deposit and the
        volume weighting, docs/cleanup-plan.md D13, or '--face-gradient'): its stored velocity derivatives ARE the
        mean field grad<v>, from exact face sums. The marker is the stream average's divergence the run writes beside
        them ('.a_velDivStreams'), as dtfelib's FieldSet.face_gradient reads it."""
        return estimator_label(self) == "PS-DTFE" and "divergence_streams" in self.files

    def gradient_family(self, name: str, gradient: str = "stored") -> str:
        """Which velocity derivative a map of 'name' shows: '' none (another field); 'mean' the mean field
        grad<v> (a standard-DTFE output's or a face-gradient output's stored grid, or with gradient 'mean-field' a
        phase-space output's computed from its velocity where mean_field_reason allows); 'stream' a phase-space
        output's stored stream average."""
        if name not in GRADIENT_FIELDS:
            return ""
        if estimator_label(self) != "PS-DTFE" or self.face_gradient:   # (a face run stores the mean field)
            return "mean"
        return "mean" if gradient == "mean-field" and not self.mean_field_reason() else "stream"

    def _mean_field_plane(self, name: str, axis: int, index: int) -> np.ndarray:
        """The mean-field 'name' on one plane in the stored grid's layout and units (raw u-units per Mpc, the
        binary's component order; slice() scales and picks components as for the stored grid): dtfelib.fields'
        mean_field_from_gradient of the plane's spectral gradient (_mf_gradient_plane), the V-web at the snapshot's
        a and h and the run log's lambda_th, as FieldSet.mean_field classifies it."""
        from dtfelib import fields as F
        web = name in ("vweb", "vweb_eigenvalues")
        a = None if self.redshift is None else 1.0 / (1.0 + float(self.redshift))
        out = F.mean_field_from_gradient(_mf_gradient_plane(self, axis, index), name, real=self.dtype,
                                         scale_factor=a, hubble_h=header_hubble(self) if web else None,
                                         lambda_th=run_lambda_th(str(self.root) + ".runlog") if web else 0.3)
        return out[name]

    def slice(self, name: str, axis: int = 2, index: int | None = None, component: str = "norm",
              gradient: str = "stored") -> np.ndarray:
        """One plane of a field, (N, N) float64, rows = the first remaining axis (a 2D grid: the whole
        grid, rows = x; axis and index are ignored). Components of a vector/tensor field by name
        (components()); 'norm' = |v|, 'trace ...' = the trace. The vorticity is the curl of the
        velocity: the binary stores the antisymmetric part of the gradient, W_ab = (dv_a/dx_b -
        dv_b/dx_a)/2 for a < b (xy, xz, yz), which this turns into the curl's x, y, z (a 2D grid: its
        z, a single number); measured on a rigid rotation (curl = 2 Omega). The 'caustic'
        map of a periodic phase-space run is the exact fold flag, not the stored grid (fold_exact).
        gradient 'mean-field': a phase-space output's velocity derivatives as the mean field, computed
        from its velocity grid (gradient_family 'mean'; a face-gradient output's stored grid is the mean field
        already); 'stored' (the default) or any other field: the file's grid."""
        index = self.n // 2 if index is None else int(index)
        if name == "caustic" and self.fold_exact():
            return self._fold_plane(int(axis), index)
        if gradient == "mean-field" and self.gradient_family(name, gradient) == "mean" \
                and estimator_label(self) == "PS-DTFE" and not self.face_gradient:
            plane = self._mean_field_plane(name, int(axis), index)
        else:
            plane = self._raw_plane(name, int(axis), index)
        return self.component_values(name, plane, component)

    def component_values(self, name: str, arr: np.ndarray, component: str, fresh: bool = True) -> np.ndarray:
        """The values a map of 'name' shows, from its grid's values 'arr' as stored (a plane, or the 3D view's block
        of rows: components last): its component in float64 -- 'norm' = |v|, 'trace ...' = the trace, else one by
        name (components()); an unknown one the first -- the vorticity turned into the curl (slice()), a velocity
        and its derivatives in km/s (velocity_factor). One rule for the map, its saved figures and the 3D volume.
        'fresh' True: always a new array (a factor of 1 multiplied too, as slice() always did); False: a factor of 1
        not applied, so a float64 scalar block comes back as it is (the volume's temporaries: no copy)."""
        ff = self.files[name]
        f64 = lambda a: np.asarray(a, dtype=np.float64)        # noqa: E731
        if name in CURL_FIELDS:         # [W_xy, W_xz, W_yz] = [-w_z, w_y, -w_x] / 2 -> the curl w
            arr = -2.0 * f64(arr) if ff.ncomp == 1 else \
                2.0 * np.stack([-f64(arr[..., 2]), f64(arr[..., 1]), -f64(arr[..., 0])], -1)
        if ff.ncomp > 1:
            comps = components(name, ff.ncomp, self.dim)
            if component not in comps:
                component = comps[0]
            if component == "norm":     # component by component in float64: never the whole array at once
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
        factor = self.velocity_factor(name)
        if fresh:
            return f64(arr) * factor
        arr = f64(arr)
        return arr * factor if factor != 1.0 else arr

    def arrow_field(self, name: str, axis: int = 2, index: int | None = None, per_side: int = 32,
                    smooth: float = 0.0, gradient: str = "stored") -> dict | None:
        """Arrows for a vector field on one slice, in IMAGE coordinates (the Explore image is
        plane.T[::-1]: x = the first in-plane axis rightward, y = the second, drawn upward, in
        cell units from the image's top-left corner). The two in-plane components are
        block-averaged to about 'per_side' arrows along each side; dx, dy are scaled so the
        longest arrow is 1 (times 'spacing' cells when drawn). None for a non-vector field. 'gradient': slice()'s
        (the vorticity's arrows of the family its colours show)."""
        ff = self.files.get(name)
        if ff is None or ff.ncomp != self.dim:          # a vector: one component per axis
            return None
        rest = [0, 1] if self.dim == 2 else [a for a in range(3) if a != axis]
        u = self.slice(name, axis, index, "xyz"[rest[0]], gradient)      # rows = rest[0], columns = rest[1]
        v = self.slice(name, axis, index, "xyz"[rest[1]], gradient)
        if smooth > 0:
            try:
                from scipy.ndimage import gaussian_filter
                u, v = gaussian_filter(u, smooth, mode="wrap"), gaussian_filter(v, smooth, mode="wrap")
            except ImportError:
                pass
        n0, n1 = u.shape
        step = max(1, int(round(max(n0, n1) / per_side)))
        m0, m1 = n0 // step * step, n1 // step * step
        if m0 == 0 or m1 == 0:
            return None
        ub = u[:m0, :m1].reshape(m0 // step, step, m1 // step, step).mean(axis=(1, 3))
        vb = v[:m0, :m1].reshape(m0 // step, step, m1 // step, step).mean(axis=(1, 3))
        c0 = (np.arange(m0 // step) + 0.5) * step                 # block centres, plane cell units
        c1 = (np.arange(m1 // step) + 0.5) * step
        a0, a1 = np.meshgrid(c0, c1, indexing="ij")
        vmax = float(np.sqrt(ub ** 2 + vb ** 2).max())
        scale = 1.0 / vmax if vmax > 0 else 0.0
        return {"x": a0.ravel(), "y": (n1 - a1).ravel(),       # image: y grows downward
                "dx": (ub * scale).ravel(), "dy": (-vb * scale).ravel(),
                "u": ub, "v": vb, "c0": c0, "c1": c1,           # plane frame, for a matplotlib quiver
                "spacing": float(step), "vmax": vmax, "axes": (rest[0], rest[1])}

    def value(self, name: str, ijk, component: str = "norm", gradient: str = "stored") -> float:
        """The value of one cell (same component and 'gradient' convention as slice); a 2D grid's cell is (i, j)."""
        if self.dim == 2:
            i, j = (int(v) for v in list(ijk)[:2])
            return float(self.slice(name, 2, 0, component, gradient)[i, j])
        i, j, k = (int(v) for v in ijk)
        return float(self.slice(name, 0, i, component, gradient)[j, k])

    # -- coordinates
    def cell_center(self, ijk) -> np.ndarray:
        return self.lo + (np.asarray(ijk, float) + 0.5) * self.length / self.n

    def plane_point(self, axis: int, index: int, row: int, col: int) -> np.ndarray:
        """The 3D cell (i, j, k) of pixel (row, col) of slice 'index' along 'axis' (a 2D grid: (i, j))."""
        if self.dim == 2:
            return np.asarray([row, col])
        rest = [a for a in range(3) if a != axis]
        ijk = [0, 0, 0]
        ijk[axis], ijk[rest[0]], ijk[rest[1]] = index, row, col
        return np.asarray(ijk)


def _mf_gradient_plane(o: OutputSet, axis: int, index: int) -> np.ndarray:
    """The mean-field velocity gradient g[..., a, b] = dv_a/dx_b on one plane of a phase-space output (raw u-units per
    Mpc), from _MF_PLANES or computed (dtfelib.fields.mean_field_gradient_plane: one pass over the velocity grid).
    The velocity grid as the cube cache's rules give it -- the cached cube, or read whole when it fits the cache and
    CUBE_EAGER_BYTES (1.6 GB at 512^3: stepping then costs no read), else in 16 MB slabs from the file, never more:
    while a job runs (the 1 GB cap) a 512^3 plane takes about 100 MB, not the 3-7.5 GB of the whole-cube FFTs."""
    from dtfelib import fields as F
    ff = o.files["velocity"]
    shape = (o.n, o.n, o.n, 3)
    key = cube_key(ff.path, o.dtype, shape)
    if key is None:
        raise OSError(f"{ff.path} cannot be read")
    k = ("mean-field",) + key + (int(axis), int(index), float(o.length[0]))
    with _CUBES_LOCK:
        g = _MF_PLANES.get(k)
        if g is not None:
            _MF_PLANES[k] = _MF_PLANES.pop(k)               # most recently used last
            return g
    vel = cached_cube(ff.path, o.dtype, shape, axis=0)      # axis 0: a cube above CUBE_EAGER_BYTES is not read whole
    g = F.mean_field_gradient_plane(vel if vel is not None else _SlabFile(ff.path, o.dtype, shape),
                                    float(o.length[0]), int(axis), int(index))
    del vel
    with _CUBES_LOCK:
        for stale in [s for s in _MF_PLANES if s[1] == k[1] and s[:6] != k[:6]]:     # the file rewritten: drop
            _MF_PLANES.pop(stale)
        if g.nbytes <= _mf_budget():
            while _MF_PLANES and sum(a.nbytes for a in _MF_PLANES.values()) + g.nbytes > _mf_budget():
                _MF_PLANES.pop(next(iter(_MF_PLANES)))
            _MF_PLANES[k] = g
    return g


def header_hubble(o) -> float | None:
    """HubbleParam of the snapshot an output was computed from, whatever unit its box is in (hubble() asks an h-free
    box): the h the binary's V-web takes from the Gadget header (H0 = 100 h), for the mean-field V-web.
    The TNG layout's combined file, or a custom run's HDF5 input; None = not known (the binary's fallback, 1). Read once
    per output (a header only) and cached on it."""
    if hasattr(o, "_header_h"):
        return o._header_h
    h = None
    try:
        import h5py
        if getattr(o, "sim", ""):
            files = sorted(Path(o.dir).glob("combined_[0-9][0-9][0-9].hdf5"))
            src = files[0] if files else None
        else:
            src = (sidecar(o.root) or {}).get("input_file")
        if src:
            with h5py.File(src, "r") as f:
                a = f["Header"].attrs
                if "HubbleParam" in a and float(a["HubbleParam"]) > 0:
                    h = float(a["HubbleParam"])
    except Exception:
        h = None
    try:
        o._header_h = h
    except AttributeError:
        pass
    return h


def field_label(o, name: str, family: str = "", qualify: bool | None = None) -> str:
    """A map's name for a field (FIELD_LABELS'), with the velocity derivatives' family said: 'velocity
    divergence (mean field ∇⟨v⟩)' or '(stream average ⟨∇v⟩)' (family: OutputSet.gradient_family) for a phase-space
    output -- or, with 'qualify', for every output: a comparison or a figure grid with a phase-space panel names the
    standard-DTFE panel's family too (its stored grid is the mean field). A standard-DTFE map alone keeps its name."""
    base = FIELD_LABELS.get(name, name)
    if not family:
        return base
    if qualify is None:
        qualify = estimator_label(o) == "PS-DTFE"
    return f"{base} ({GRADIENT_QUALIFIER[family]})" if qualify else base


def component_suffix(component: str) -> str:
    """A map's component as its title and colour bar append it: ' [x]'; nothing for a scalar's 'value'."""
    return "" if component in ("value", "") else f" [{component}]"


def colourbar_label(label: str, req: dict) -> str:
    """The colour bar's text of a map request (the slice, a zoom, the 3D view): 'log₁₀ ' on a log map, the field's
    label (field_label), its component (component_suffix)."""
    return ("log₁₀ " if req["log"] else "") + label + component_suffix(req["component"])


def smoothed(plane, sigma: float):
    """Gaussian smoothing in cells (periodic: wrapped at the edges); unchanged for sigma 0 or without scipy."""
    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(plane, sigma, mode="wrap")
        except ImportError:
            pass
    return plane


# ---------------------------------------------------------------------- the map smoothed in 3D
# The map's Smoothing filters the SLICE; with its '3D' tick the whole cube is smoothed instead (the field the voids
# are found in: the catalogue smooths the cube) and the map shows a slice of that. One cube per (grid file, field,
# component, sigma), smoothed once on the slice worker and kept: scrolling the layers then costs a copy of a plane.
# Measured on a synthetic 512^3 float32 cube (this machine, numpy 2.4): 3.5 s, 0.8 GB above the cube itself while it
# runs, 0.54 GB kept. An exact periodic Gaussian by numpy's FFT (the catalogue's is scipy's, cut at 4 sigma: on
# TNG300-3-Dark 512^3 the two contrasts differ by 4e-3 at most): separable, so one axis at a time over slabs of
# about SMOOTH3D_CHUNK bytes -- the whole-cube rfftn peaked at 5.4 GB for the same cube. The smoothed cubes share the
# cube cache's budget with the grids' own cubes (_make_room), and those of another output go when the map's output
# changes (drop_smoothed3d): a few smoothings of a 1024^3 grid are 4.3 GB each.
SMOOTH3D_CHUNK = 32e6
SMOOTH3D_HEADROOM_GB = 4.0      # the tick is greyed out when the smoothing would leave less than this free
_SMOOTH3D: dict = {}            # (cube_key, field, component, sigma) -> float32 cube, least recently used first


def gauss3d(cube, sigma: float, chunk: float = SMOOTH3D_CHUNK) -> np.ndarray:
    """A periodic Gaussian of 'sigma' cells over a 3D array, in float32: exp(-sigma^2 k^2 / 2) applied one axis at a
    time (the Gaussian is separable: the three passes equal the 3D FFT filter to float32 rounding), each in slabs of
    about 'chunk' bytes."""
    out = np.array(cube, dtype=np.float32, copy=True)
    if not sigma > 0:
        return out
    for axis in range(3):
        m = out.shape[axis]
        g = np.exp(-0.5 * (float(sigma) * 2.0 * np.pi * np.fft.rfftfreq(m)) ** 2).astype(np.float32)
        shape = [1, 1, 1]
        shape[axis] = g.size
        g = g.reshape(shape)
        other = 1 if axis == 0 else 0
        step = max(1, int(chunk // (out.nbytes / out.shape[other])))
        for s in range(0, out.shape[other], step):
            sl = [slice(None)] * 3
            sl[other] = slice(s, s + step)
            f = np.fft.rfft(out[tuple(sl)], axis=axis)
            f *= g
            out[tuple(sl)] = np.fft.irfft(f, n=m, axis=axis)
    return out


def smooth3d_why(o, name: str, gradient: str = "stored", hires: bool = False) -> str:
    """'' when the map of field 'name' can be smoothed in 3D (the whole cube), else why not in plain words: a 2D grid,
    a hi-res slice, a map computed slice by slice (the fold flag, a phase-space output's mean-field derivatives), a
    class map; a smoothed cube larger than the cube cache keeps (while a job runs its cap: every layer would smooth
    the whole cube again); a cube whose smoothing would not fit in the memory free now (less SMOOTH3D_HEADROOM_GB)
    says what it needs. The tick asks it; map_plane asks it again before smoothing a cube (a comparison's B, a grid
    panel, or a job started since the tick was set)."""
    why = _smooth3d_kind(o, name, gradient, hires)
    if why:
        return why
    if 4.0 * float(o.n) ** 3 > _cube_budget():
        return ("while a job runs Explore keeps no cube this large, so every layer would be smoothed again" if _CAPPED
                else "the smoothed cube is larger than the memory Explore keeps for grids, so every layer would be "
                     "smoothed again")
    need = smooth3d_need(o, name)
    if need > 0:
        free = available_gb()
        if free is not None and need / 1e9 > free - SMOOTH3D_HEADROOM_GB:
            return (f"smoothing this {o.n}³ cube needs about {need / 1e9:.1f} GB and {free:.0f} GB is free (it keeps "
                    f"{SMOOTH3D_HEADROOM_GB:.0f} GB for the other programs)")
    return ""


def _smooth3d_kind(o, name: str, gradient: str = "stored", hires: bool = False) -> str:
    """smooth3d_why without the memory: '' when the map is a slice of a stored 3D grid, else why not."""
    if o is None or name not in getattr(o, "files", {}):
        return "no map shown"
    if o.dim != 3:
        return "a 2D grid is its one plane"
    if hires:
        return "a hi-res slice is a plane of points, not a slice of the grid"
    if name == "caustic" and o.fold_exact():
        return "the fold map is computed slice by slice"
    if name in ("tweb", "vweb"):
        return "a class map is not smoothed"
    if gradient == "mean-field" and o.gradient_family(name, gradient) == "mean" and estimator_label(o) == "PS-DTFE" \
            and not o.face_gradient:
        return "this map is computed slice by slice from the velocity (the mean field)"
    return ""


def smooth3d_need(o, name: str) -> float:
    """The bytes smoothing a field's cube in 3D takes at its peak (0 when it is kept already): the grid read whole
    (unless the cube cache holds it), the values of a vector's component in float64, the smoothed float32 cube."""
    ff = o.files[name]
    cells = float(o.n) ** 3
    shape = (o.n,) * 3 + ((ff.ncomp,) if ff.ncomp > 1 else ())
    raw = 0.0 if _cube_cached(ff.path, o.dtype, shape) else cells * ff.ncomp * o.dtype.itemsize
    values = 8.0 * cells if (ff.ncomp > 1 or name in CURL_FIELDS or o.dtype.itemsize > 4) else 0.0
    return raw + values + 4.0 * cells + 2 * SMOOTH3D_CHUNK


def _cube_cached(path, dtype, shape) -> bool:
    key = cube_key(Path(path), dtype, shape)
    with _CUBES_LOCK:
        return key is not None and key in _CUBES


def _smooth3d_key(o, name: str, component: str, sigma: float):
    ff = o.files[name]
    k = cube_key(ff.path, o.dtype, (o.n,) * 3 + ((ff.ncomp,) if ff.ncomp > 1 else ()))
    return None if k is None else (k, name, component, round(float(sigma), 6))


def smoothed3d_kept(o, name: str, component: str, sigma: float) -> bool:
    """The cube of this field, component and sigma is smoothed already (its slices are instant)."""
    key = _smooth3d_key(o, name, component, sigma)
    with _CUBES_LOCK:
        return key is not None and key in _SMOOTH3D


def smoothed3d_cube(o, name: str, component: str, sigma: float) -> np.ndarray:
    """The whole cube of a field's component (the map's values: slice()'s components and units) smoothed in 3D by a
    periodic Gaussian of 'sigma' cells (gauss3d), float32; kept (least recently used dropped first) within the cube
    cache's budget, shared with the grids' own cubes (_make_room) -- while a job runs its 1 GB cap -- and dropped
    with it (drop_cubes)."""
    key = _smooth3d_key(o, name, component, sigma)
    if key is None:
        raise OSError(f"{o.files[name].path} cannot be read")
    with _CUBES_LOCK:
        cube = _SMOOTH3D.get(key)
        if cube is not None:
            _SMOOTH3D[key] = _SMOOTH3D.pop(key)
            return cube
    ff = o.files[name]
    arr = o._memmap(name, None)                          # the whole grid (the cached cube when it is kept)
    if ff.ncomp == 1 and name not in CURL_FIELDS:
        cube = gauss3d(arr, sigma)
        factor = o.velocity_factor(name)
        if factor != 1.0:
            cube *= np.float32(factor)
    else:                                                # a vector's or tensor's component: slice()'s values first
        vals = o.component_values(name, arr, component, fresh=False)
        del arr
        cube = gauss3d(vals, sigma)
        del vals
    with _CUBES_LOCK:
        for stale in [k for k in _SMOOTH3D if k[0][0] == key[0][0] and k[0] != key[0]]:     # the file rewritten
            _SMOOTH3D.pop(stale)
        if cube.nbytes <= _cube_budget():
            # the grids' own cubes go first (a re-read is cheap, a smoothing is not): Compare's A and B, or a grid's
            # panels, keep their smoothed cubes side by side whenever those alone fit
            _make_room(cube.nbytes, _CUBES, _SMOOTH3D)
            _SMOOTH3D[key] = cube
    return cube


def drop_smoothed3d(keep=None) -> int:
    """Forget the cubes smoothed in 3D -- with 'keep' (an OutputSet) only those of other outputs' files, as when the
    map's output changes; the bytes freed."""
    files = set() if keep is None else {Path(ff.path).resolve() for ff in getattr(keep, "files", {}).values()}
    with _CUBES_LOCK:
        gone = [k for k in _SMOOTH3D if k[0][0] not in files]
        n = sum(_SMOOTH3D[k].nbytes for k in gone)
        for k in gone:
            _SMOOTH3D.pop(k)
    return n


def _drop_smoothed3d() -> int:
    return drop_smoothed3d()


on_drop_cubes(_drop_smoothed3d)


class Smooth3DRefused(ValueError):
    """map_plane will not smooth this cube in 3D: its text, a whole sentence, is what the view shows."""


def map_plane(o, name: str, axis: int, index: int, component: str, gradient: str = "stored", smooth: float = 0.0,
              smooth3d: bool = False) -> np.ndarray:
    """A map's plane as every view reads it (the map and its comparison, the figure grid's panels, the 3D view's
    planes): the slice smoothed in 2D (smoothed), or with 'smooth3d' the slice of the cube smoothed in 3D at 'smooth'
    cells (smoothed3d_cube); float64. A cube not smoothed yet is smoothed only where smooth3d_why allows it (the
    memory free now, the cache's budget), else Smooth3DRefused saying why (the view shows it)."""
    if smooth3d and smooth > 0 and not _smooth3d_kind(o, name, gradient):
        if not smoothed3d_kept(o, name, component, smooth):
            why = smooth3d_why(o, name, gradient)
            if why:
                raise Smooth3DRefused(f"Not smoothed in 3D: {why}. Untick 3D beside Smoothing to see the map.")
        cube = smoothed3d_cube(o, name, component, smooth)
        sl = [slice(None)] * 3
        sl[int(axis)] = int(index)
        return np.asarray(cube[tuple(sl)], dtype=np.float64)
    return smoothed(o.slice(name, axis, index, component, gradient), smooth)


# ---------------------------------------------------------------------- colour
_LUTS: dict[str, np.ndarray] = {}


def lut(cmap: str) -> np.ndarray:
    """(256, 3) uint8 colour table of a matplotlib colormap (a grey ramp without matplotlib)."""
    if cmap not in _LUTS:
        try:
            import matplotlib
            table = matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
        except Exception:
            table = np.repeat(np.linspace(0, 1, 256)[:, None], 3, axis=1)
        _LUTS[cmap] = (table * 255 + 0.5).astype(np.uint8)
    return _LUTS[cmap]


def _scaled(plane: np.ndarray, log: bool) -> np.ndarray:
    """The values a map shows: log10 of the positive ones (the rest at the bottom of the scale)."""
    data = np.asarray(plane, dtype=np.float64)
    if log:
        pos = data[data > 0]
        floor = pos.min() if pos.size else 1.0
        data = np.log10(np.where(data > 0, data, floor))
    return data


def value_range(plane: np.ndarray, log: bool = False, clip: tuple = (1.0, 99.0),
                symmetric: bool = False) -> tuple[float, float]:
    """The (vmin, vmax) colorize() would show: the percentiles 'clip' of the finite scaled values,
    'symmetric' centred on 0 (the figure grid shares such ranges between panels)."""
    data = _scaled(plane, log)
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(finite, clip)
    else:
        vmin, vmax = 0.0, 1.0
    if symmetric:                   # the percentiles of a map that is 0 almost everywhere are 0: then the largest
        m = max(abs(vmin), abs(vmax)) or (float(np.abs(finite).max()) if finite.size else 0.0) or 1.0   # value
        vmin, vmax = -m, m
    if not vmax > vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def colorize(plane: np.ndarray, cmap: str = "viridis", log: bool = False,
             clip: tuple = (1.0, 99.0), symmetric: bool = False,
             vrange: tuple | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """plane -> (H, W, 3) uint8 RGB and the (vmin, vmax) shown. log10 of positive values (the rest
    at the bottom of the scale); range from the percentiles 'clip' unless 'vrange' is given;
    'symmetric' centres a diverging map on 0."""
    data = _scaled(plane, log)
    if vrange is not None:
        vmin, vmax = vrange
        if symmetric:
            m = max(abs(vmin), abs(vmax)) or 1.0
            vmin, vmax = -m, m
        if not vmax > vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = value_range(plane, log, clip, symmetric)
    idx = np.clip((data - vmin) / (vmax - vmin) * 255.0, 0, 255)
    idx = np.where(np.isfinite(idx), idx, 0).astype(np.uint8)
    return lut(cmap)[idx], (float(vmin), float(vmax))


# ---------------------------------------------------------------------- ticks (the Explore axes and colour bar)
TICK_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)        # the round steps, per decade


def nice_ticks(lo: float, hi: float, n: int = 5, integer: bool = False) -> list[float]:
    """About n round values inside [lo, hi], a step of TICK_STEPS x 10^k apart (0, 20, 40 ...; 0.25, 0.5 ...).
    'integer': whole steps of 1, 2, 5 x 10^k only -- an axis in cells (0, 5, 10 on a 16-cell grid, not 0, 2.5,
    5: a cell boundary means nothing). Empty for a degenerate range."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or n < 1:
        return []
    raw = (hi - lo) / n
    mag = 10.0 ** np.floor(np.log10(raw))
    steps = [m * mag for m in TICK_STEPS if not (integer and m == 2.5)]
    if integer:
        steps = [max(st, 1.0) for st in steps]
    step = min(steps, key=lambda st: abs(np.log(st / raw)))    # the nearest round step
    k0, k1 = int(np.ceil(lo / step - 1e-9)), int(np.floor(hi / step + 1e-9))
    digits = max(0, 2 - int(np.floor(np.log10(step))))
    return [float(round(k * step, digits)) + 0.0 for k in range(k0, k1 + 1)]   # + 0.0: no '-0'


def tick_text(v: float, ticks: list[float]) -> str:
    """A tick label with the decimals the step needs (and no more): 0.25 / 0.5 -> '0.25', '0.50'."""
    if len(ticks) > 1:
        step = float(f"{min(abs(b - a) for a, b in zip(ticks, ticks[1:])):.3g}")   # 0.000999.. is 0.001
        if step >= 1e5 or step <= 0:
            return f"{v:.3g}"
        d = next((k for k in range(13) if abs(step * 10 ** k - round(step * 10 ** k)) < 1e-6 * max(1.0, step * 10 ** k)), 12)
        return f"{v:.{d}f}"                         # the fewest decimals that show every tick exactly (2.5 -> 1)
    return f"{v:.3g}"


def nice_length(span: float, frac: float = 0.2) -> float:
    """The scale bar's round length (1, 2 or 5 x 10^k) nearest frac * span (log distance)."""
    if not (np.isfinite(span) and span > 0):
        return 0.0
    target = frac * span
    mag = 10.0 ** np.floor(np.log10(target))
    cands = [m * mag for m in (1.0, 2.0, 5.0, 10.0)]
    return float(min(cands, key=lambda c: abs(np.log(c / target))))


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def decade_text(k: int) -> str:
    """10^k as the colour bar writes it: '1', '10', '10²', '10⁻¹'."""
    return {0: "1", 1: "10"}.get(k, "10" + str(k).translate(_SUPERSCRIPT))


def colorbar_ticks(lo: float, hi: float, log: bool = False, n: int = 5) -> list[tuple[float, str]]:
    """(position in the bar's units, label) for a colour bar spanning lo..hi. On a log map (lo, hi are
    log10 values) the whole decades inside, labelled 10^k (thinned to about n); with fewer than two
    decades, round log values labelled with their linear value."""
    if log:
        ks = list(range(int(np.ceil(lo - 1e-9)), int(np.floor(hi + 1e-9)) + 1))
        if len(ks) >= 3:
            every = max(1, int(np.ceil(len(ks) / n)))
            return [(float(k), decade_text(k)) for k in ks if k % every == 0]
        # under three decades: 1, 2, 5 x 10^k as plain numbers (0.05, 0.1, 0.2), else round log values
        sub = [(np.log10(m) + k, m, k) for k in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for m in (1, 2, 5)]
        sub = [(float(v), f"{m * 10.0 ** k:g}") for v, m, k in sub if lo - 1e-9 <= v <= hi + 1e-9]
        if len(sub) >= 2:
            return sub
        return [(v, f"{10.0 ** v:.3g}") for v in nice_ticks(lo, hi, n)]
    ticks = nice_ticks(lo, hi, n)
    return [(v, tick_text(v, ticks)) for v in ticks]


def void_circles(centres, radii, box: float, axis: int, position: float, rest, lo, hi, n0: int, n1: int,
                 periodic: bool = True) -> list[tuple[float, float, float, float, int]]:
    """Where spheres (the voids: centres (k, 3) and radii in Mpc) cut the shown slice, in IMAGE pixels of the
    Explore map: [(x, y, rx, ry, k)], k the sphere's row. The slice is the plane at 'position' along 'axis'; a
    sphere at (periodic) distance d < R from it leaves a circle of radius sqrt(R^2 - d^2), kept (with each periodic
    image) when it reaches the shown region lo..hi of the 'rest' axes. The map's convention: image = plane.T[::-1],
    so x runs along rest[0] (n0 pixels over lo[0]..hi[0]) and y DOWN along rest[1] (n1 pixels, hi[1] at the top);
    rx, ry differ when the pixels are not square in Mpc."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(radii, dtype=np.float64).reshape(-1)
    L = float(box)
    d = c[:, axis] - float(position)
    if periodic and L > 0:
        d -= L * np.round(d / L)
    with np.errstate(invalid="ignore"):
        cut = np.isfinite(r) & (np.abs(d) < r)
    rho = np.sqrt(np.where(cut, r * r - d * d, 0.0))
    (u0, v0), (u1, v1) = (float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))
    sx, sy = n0 / (u1 - u0), n1 / (v1 - v0)
    shifts = (-L, 0.0, L) if periodic and L > 0 else (0.0,)
    out = []
    for k in np.nonzero(cut)[0]:
        for du in shifts:
            u = c[k, rest[0]] + du
            if u + rho[k] < u0 or u - rho[k] > u1:
                continue
            for dv in shifts:
                v = c[k, rest[1]] + dv
                if v + rho[k] < v0 or v - rho[k] > v1:
                    continue
                out.append(((u - u0) * sx, (v1 - v) * sy, rho[k] * sx, rho[k] * sy, int(k)))
    return out


def void_circles_mpc(centres, radii, box: float, axis: int, position: float, rest, lo, hi,
                     periodic: bool = True) -> list[tuple[float, float, float, int]]:
    """void_circles in the DATA frame of a matplotlib imshow(extent=[lo0, hi0, lo1, hi1], origin='lower'):
    [(u, v, radius, k)] in Mpc, for a saved figure's matplotlib Circle patches. One pixel per Mpc
    and y measured down from hi1 make void_circles' image pixels the data offsets."""
    (u0, v0), (u1, v1) = (float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))
    return [(u0 + x, v1 - y, rx, k) for x, y, rx, _ry, k in
            void_circles(centres, radii, box, axis, position, rest, lo, hi, u1 - u0, v1 - v0, periodic)]


def draw_void_circles_mpl(ax, circles_mpc, dashed=(), color="white"):
    """The outlines on a matplotlib axes as the screen draws them: white over a dark edge, clipped to the axes;
    'dashed' = a second list (R_eff) drawn dashed. Returns the number of patches drawn."""
    from matplotlib.patches import Circle
    n = 0
    for circles, style in ((circles_mpc, "-"), (dashed, (0, (4, 3)))):
        for u, v, r, _k in circles:
            for lw, c in ((1.6, (0, 0, 0, 0.65)), (0.7, color)):
                ax.add_patch(Circle((u, v), r, fill=False, lw=lw, ec=c, ls=style, clip_on=True))
            n += 1
    return n


def draw_contours_mpl(ax, per_level, colours, to_xy, lw: float = 0.6, clip=None) -> int:
    """Contour lines (contours.map_lines' 'segments': one (k, 2, 2) array per level, in the plane's array
    coordinates) on a matplotlib axes as the screen draws them: each level in its colour (contours.colours, RGBA
    0..1), one path per colour stroked once -- a translucent line keeps one shade where its pieces meet. 'to_xy'
    maps an (m, 2) array of (i, j) to the axes' data coordinates; 'clip' a matplotlib patch to clip to (a figure
    grid's tile). Returns the number of pieces drawn."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    groups: dict = {}
    for segs, rgba in zip(per_level, colours):
        if np.size(segs):
            pts = np.asarray(segs, dtype=np.float64).reshape(-1, 2)
            groups.setdefault(tuple(float(c) for c in rgba), []).append(pts)
    n = 0
    for rgba, parts in groups.items():
        pts = to_xy(np.concatenate(parts))
        codes = np.tile(np.array([MPath.MOVETO, MPath.LINETO], dtype=MPath.code_type), len(pts) // 2)
        patch = PathPatch(MPath(pts, codes), fill=False, ec=rgba, lw=lw, capstyle="round", joinstyle="round",
                          clip_on=True)
        ax.add_patch(patch)
        if clip is not None:
            patch.set_clip_path(clip)
        n += len(pts) // 2
    return n


def mark_levels_mpl(cb, lines: dict | None, lo: float, hi: float, lw: float = 0.6) -> int:
    """A saved figure's colour bar (matplotlib Colorbar 'cb') marked at a map's contour levels inside lo..hi, each in
    its line's colour, as the screen's bar marks them; 'lines' is contours.map_lines' dict (None: nothing). The marks
    drawn."""
    if not lines:
        return 0
    levels, colours = np.asarray(lines["levels"]), np.asarray(lines["colours"])
    inside = (levels > lo) & (levels < hi)
    if inside.any():
        cb.add_lines(levels[inside], [tuple(c) for c in colours[inside]], [lw] * int(inside.sum()))
    return int(inside.sum())


def void_at(circles, x: float, y: float):
    """The catalogue row k of the outline under image position (x, y) (pixels), or None: of the circles that
    contain it the smallest one (a small void drawn inside a big one's outline stays reachable)."""
    best = None
    for cx, cy, rx, ry, k in circles or ():
        if rx <= 0 or ry <= 0:
            continue
        if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0 and (best is None or rx < best[0]):
            best = (rx, int(k))
    return None if best is None else best[1]


def _semi_axes_text(res: dict, k: int) -> str:
    """'; a, b, c = 9.10, 7.32, 5.04 Mpc' for a void whose fitted ellipsoid the result carries, else ''."""
    semi = (res.get("semi_axes_mpc") or [None] * (k + 1))[k]
    return "" if semi is None else "; a, b, c = " + ", ".join(f"{float(v):.2f}" for v in semi) + " Mpc"


# ---------------------------------------------------------------------- the voids drawn as their ellipsoids
def _rows_array(values, n: int, width: int) -> np.ndarray:
    """A JSON list of per-void lists (None where undefined) as an (n, width) float array, NaN where undefined."""
    vals = list(values) if values is not None else [None] * n
    out = [[np.nan] * width if v is None else np.ravel(v) for v in vals]
    return np.array(out, dtype=np.float64).reshape(n, width)


def drawn_ellipsoids(res: dict) -> dict:
    """The voids' ellipsoids as Explore draws them -- the ONE place their size is computed: the map's sections, the
    3D view, the inspector, the Sizes range and the saved CSV all read this. The quadratic fit around each minimum,
    delta(x) = delta_min + 1/2 sum lambda_i x_i^2 (the catalogue's Hessian eigenvalues lambda_i and eigenvectors), at
    the MEAN density delta = 0: centre the minimum, axes the eigenvectors, semi-axes sqrt(2 |delta_min| / lambda_i)
    (= the catalogue's 2 cell / sqrt(lambda_i) times sqrt(|delta_min| / 2)). Only where it exists: delta_min < 0 and
    all three eigenvalues > 0 (a true minimum below the mean) of a resolved void; elsewhere NaN and 'ok' False (drawn
    as its centre mark alone). {'semi' (k, 3) Mpc, a >= b >= c; 'dirs' (k, 3, 3) unit directions in rows; 'M'
    (k, 3, 3) the semi-axis vectors in columns (view3d's shape matrix); 'ok' (k,); 'r' (k,) (a b c)^(1/3) Mpc, 'R_ell'}. Kept
    in the result ('_drawn'). A result without 'eigenvalues_cells' (an older helper) takes the catalogue's |lambda|
    from its semi-axes, their signs unknown."""
    kept = res.get("_drawn")
    if kept is not None:
        return kept
    n = len(res["centres_mpc"])
    cell = float(res.get("cell_mpc") or 0.0) or float(res["box_mpc"]) / float(res["n_grid"])
    delta = np.array([np.nan if v is None else v for v in (res.get("delta") or [None] * n)], dtype=np.float64)
    semi_cat = _rows_array(res.get("semi_axes_mpc"), n, 3)
    dirs = _rows_array(res.get("ellipsoid_axes"), n, 9).reshape(n, 3, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        ev = _rows_array(res["eigenvalues_cells"], n, 3) if res.get("eigenvalues_cells") is not None \
            else (2.0 * cell / semi_cat) ** 2
        ok = (np.isfinite(delta) & (delta < 0) & np.all(np.isfinite(ev) & (ev > 0), axis=1)
              & np.all(np.isfinite(dirs).reshape(n, 9), axis=1))
        lam = np.where(ev > 0, ev, np.nan)
        semi = np.where(ok[:, None], cell * np.sqrt(2.0 * np.abs(delta)[:, None] / lam), np.nan)
    out = {"semi": semi, "dirs": dirs, "M": np.swapaxes(dirs, 1, 2) * semi[:, None, :], "ok": ok,
           "r": np.cbrt(np.prod(semi, axis=1))}
    res["_drawn"] = out
    return out


def void_ellipses(centres, M, box: float, axis: int, position: float, rest, lo, hi, n0, n1,
                  periodic: bool = True) -> list[tuple[float, float, float, float, float, int]]:
    """Where ellipsoids (centres (k, 3) Mpc, shape matrices M (k, 3, 3): view3d's centre + M u, |u| <= 1) cut the
    shown slice, in IMAGE pixels of the map (void_circles' frame: x along rest[0], y DOWN along rest[1]):
    [(x, y, rx, ry, angle, k)], the ellipse centred at (x, y) with semi-axes rx >= ry, the rx axis turned by 'angle'
    degrees from the image's x axis toward its y axis (QPainter's rotate; clockwise on the screen). The exact plane
    section, view3d.plane_cuts (the 3D view's planes draw the very same): with Q = M M^T it exists while the plane's
    distance d from the centre has d^2 < Q_aa; its centre moves in the plane by Q_ra d / Q_aa (a tilted ellipsoid's
    section is off its centre), its shape is (Q_rr - Q_ra Q_ar / Q_aa)(1 - d^2 / Q_aa). Each periodic image that
    reaches the region lo..hi kept; NaN shapes skipped."""
    import view3d as V
    if list(rest) != list(V.rest_axes(axis)):
        raise ValueError(f"the rest axes {list(rest)} are not the map's for axis {axis}")
    out = []
    for x, y, L, k in V.plane_cuts(centres, M, box, axis, position, lo, hi, n0, n1, periodic):
        w, U = np.linalg.eigh(np.asarray(L) @ np.asarray(L).T)       # the image of the unit circle under L
        rx, ry = float(np.sqrt(max(w[1], 0.0))), float(np.sqrt(max(w[0], 0.0)))
        out.append((float(x), float(y), rx, ry, float(np.degrees(np.arctan2(U[1, 1], U[0, 1]))), int(k)))
    return out


def void_marks(centres, reach, box: float, axis: int, position: float, rest, lo, hi, n0, n1, half: float,
               periodic: bool = True) -> list[tuple[float, float, int, bool]]:
    """Where the voids' centres (the minima) project into the shown slice, in image pixels: [(x, y, k, own)] for the
    voids within 'reach' (k,) of the plane along its normal (an ellipsoid's half extent along it: its section is
    drawn; NaN: only on the centre's own layer), own = the slice is the centre's own layer (|d| < 'half', half a cell), each
    periodic image inside the region lo..hi."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    reach = np.nan_to_num(np.asarray(reach, dtype=np.float64).reshape(-1), nan=0.0)
    L = float(box)
    d = float(position) - c[:, axis]
    if periodic and L > 0:
        d -= L * np.round(d / L)
    with np.errstate(invalid="ignore"):
        show = np.isfinite(d) & (np.abs(d) < np.maximum(reach, float(half)))
    (u0, v0), (u1, v1) = (float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))
    sx, sy = n0 / (u1 - u0), n1 / (v1 - v0)
    shifts = (-L, 0.0, L) if periodic and L > 0 else (0.0,)
    out = []
    for k in np.nonzero(show)[0]:
        own = bool(abs(d[k]) < float(half))
        for du in shifts:
            u = c[k, rest[0]] + du
            if not u0 <= u <= u1:
                continue
            for dv in shifts:
                v = c[k, rest[1]] + dv
                if v0 <= v <= v1:
                    out.append(((u - u0) * sx, (v1 - v) * sy, int(k), own))
    return out


def void_overlay(v: dict | None, axis: int, position: float, rest, lo, hi, n0, n1, half: float,
                 periodic: bool = True) -> dict:
    """The void outlines of one slice in image pixels, for the map, a zoom, a figure grid's panel and the saves alike:
    {'circles': [(x, y, rx, ry, row)] (spheres of R_v), 'dashed': the same for the other radius asked for,
    'ellipses': [(x, y, rx, ry, angle, row)] (void_ellipses), 'marks': [(x, y, row, own)] (void_marks)}, rows the
    catalogue's. 'v' (explore_voids: the rows shown): {'rows', 'centres', 'box', 'shape' ('ellipsoid' or 'sphere'),
    'r_v', 'M' and 'reach' (the drawn ellipsoids, NaN where none; reach (k, 3) their half extents along x, y, z, or
    (k,) along 'axis'), 'other' (the dashed circles' radii, or None)}.
    Spheres: R_v's circles, as before the ellipsoids existed. Ellipsoids: each one's section and the mark of its
    centre while the slice cuts it, a void without one its mark on its own layer."""
    out = {"circles": [], "dashed": [], "ellipses": [], "marks": []}
    if not v or not len(v["rows"]):
        return out
    rows = np.asarray(v["rows"], dtype=np.int64)
    box = float(v["box"])
    if v.get("shape") == "ellipsoid":
        out["ellipses"] = [(x, y, rx, ry, a, int(rows[k])) for x, y, rx, ry, a, k in
                           void_ellipses(v["centres"], v["M"], box, axis, position, rest, lo, hi, n0, n1, periodic)]
        reach = np.asarray(v["reach"], dtype=np.float64)
        reach = reach[:, axis] if reach.ndim == 2 else reach            # (k, 3): the half extent along each axis
        out["marks"] = [(x, y, int(rows[k]), own) for x, y, k, own in
                        void_marks(v["centres"], reach, box, axis, position, rest, lo, hi, n0, n1, half, periodic)]
    else:
        out["circles"] = [(x, y, rx, ry, int(rows[k])) for x, y, rx, ry, k in
                          void_circles(v["centres"], v["r_v"], box, axis, position, rest, lo, hi, n0, n1, periodic)]
    if v.get("other") is not None:
        out["dashed"] = [(x, y, rx, ry, int(rows[k])) for x, y, rx, ry, k in
                         void_circles(v["centres"], v["other"], box, axis, position, rest, lo, hi, n0, n1, periodic)]
    return out


def void_overlay_mpc(v: dict | None, axis: int, position: float, rest, lo, hi, half: float,
                     periodic: bool = True) -> dict:
    """void_overlay in the DATA frame of a matplotlib imshow(extent=[lo0, hi0, lo1, hi1], origin='lower') (one pixel
    per Mpc, y measured down from hi1: void_circles_mpc's trick): circles and dashed [(u, v, r, row)], ellipses
    [(u, v, rx, ry, angle, row)] with matplotlib's angle (counter-clockwise, y up), marks [(u, v, row, own)]."""
    (u0, v0), (u1, v1) = (float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))
    ov = void_overlay(v, axis, position, rest, lo, hi, u1 - u0, v1 - v0, half, periodic)
    return {"circles": [(u0 + x, v1 - y, rx, k) for x, y, rx, _ry, k in ov["circles"]],
            "dashed": [(u0 + x, v1 - y, rx, k) for x, y, rx, _ry, k in ov["dashed"]],
            "ellipses": [(u0 + x, v1 - y, rx, ry, -a, k) for x, y, rx, ry, a, k in ov["ellipses"]],
            "marks": [(u0 + x, v1 - y, k, own) for x, y, k, own in ov["marks"]]}


def scale_ellipse(rx: float, ry: float, angle: float, k0: float, k1: float) -> tuple[float, float, float]:
    """An ellipse (semi-axes rx >= ry, its rx axis at 'angle' degrees, matplotlib's sense) in axes scaled by k0 along
    x and k1 along y (a saved figure's unit: Mpc/h, cells): (rx', ry', angle') of the same curve."""
    if abs(k0 - k1) <= 1e-12 * max(abs(k0), abs(k1), 1e-300):
        return rx * k0, ry * k0, angle
    t = np.radians(angle)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    D = np.diag([k0, k1])
    w, U = np.linalg.eigh(D @ R @ np.diag([rx * rx, ry * ry]) @ R.T @ D)
    angle = float(np.degrees(np.arctan2(U[1, 1], U[0, 1])))
    return float(np.sqrt(max(w[1], 0.0))), float(np.sqrt(max(w[0], 0.0))), angle


MARK_PT = 3.2                   # a void centre's mark in a saved figure (points across; the screen's is 5 pixels)


def draw_void_overlay_mpl(ax, ov: dict, color="white") -> int:
    """void_overlay_mpc's outlines on a matplotlib axes as the screen draws them: the circles (draw_void_circles_mpl:
    exactly as before the ellipsoids), the ellipses white over a dark edge, the centres' marks (filled on their own
    layer, a ring elsewhere); clipped to the axes. Returns the outlines and marks drawn."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    n = draw_void_circles_mpl(ax, ov.get("circles") or (), ov.get("dashed") or (), color) \
        if ov.get("circles") or ov.get("dashed") else 0
    for u, v, rx, ry, a, _k in ov.get("ellipses") or ():
        for lw, c in ((1.6, (0, 0, 0, 0.65)), (0.7, color)):
            ax.add_patch(Ellipse((u, v), 2 * rx, 2 * ry, angle=a, fill=False, lw=lw, ec=c, clip_on=True))
        n += 1
    for u, v, _k, own in ov.get("marks") or ():
        for mew, ec in ((1.6, (0, 0, 0, 0.65)), (0.7, color)):
            ax.add_line(Line2D([u], [v], ls="none", marker="o", ms=MARK_PT, mew=mew, mec=ec,
                               mfc=color if own else "none", clip_on=True))
        n += 1
    return n


def draw_pixel_overlay(ax, ellipses=(), ellipse_clips=(), marks=(), mark_clips=()) -> int:
    """The figure grid's ellipses and centre marks over its full-resolution picture (image pixels, y down: the axes
    set as figsave.draw_pixel_circles sets them), each clipped to its tile as on screen. The patches drawn."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Rectangle
    n = 0
    for j, (x, y, rx, ry, a, _k) in enumerate(ellipses or ()):
        clip = ellipse_clips[j] if j < len(ellipse_clips or ()) else None
        for lw, col in ((1.6, (0, 0, 0, 0.65)), (0.7, "white")):
            e = Ellipse((x, y), 2 * rx, 2 * ry, angle=a, fill=False, lw=lw, ec=col)
            ax.add_patch(e)
            if clip is not None:
                e.set_clip_path(Rectangle((clip[0], clip[1]), clip[2], clip[3], transform=ax.transData))
        n += 1
    for j, (x, y, _k, own) in enumerate(marks or ()):
        clip = mark_clips[j] if j < len(mark_clips or ()) else None
        for mew, ec in ((1.6, (0, 0, 0, 0.65)), (0.7, "white")):
            ln = Line2D([x], [y], ls="none", marker="o", ms=MARK_PT, mew=mew, mec=ec, mfc="white" if own else "none")
            ax.add_line(ln)
            if clip is not None:
                ln.set_clip_path(Rectangle((clip[0], clip[1]), clip[2], clip[3], transform=ax.transData))
        n += 1
    return n


def outline_pick(x: float, y: float, circles=(), ellipses=(), marks=(), mark_radius: float = 3.0):
    """The catalogue row under image position (x, y): of the outlines that contain it -- circles [(x, y, rx, ry, row)]
    and turned ellipses [(x, y, rx, ry, angle, row)] -- the smallest (void_at's rule: a small void drawn inside a
    big one stays reachable); else the nearest centre mark within 'mark_radius' pixels (a void drawn as its mark
    alone); None."""
    best = None
    for o in list(circles or ()) + list(ellipses or ()):
        cx, cy, rx, ry = o[:4]
        a = np.radians(o[4]) if len(o) > 5 else 0.0
        if rx <= 0 or ry <= 0:
            continue
        dx, dy = x - cx, y - cy
        p, q = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
        if (p / rx) ** 2 + (q / ry) ** 2 <= 1.0 and (best is None or rx * ry < best[0]):
            best = (rx * ry, int(o[-1]))
    if best is not None:
        return best[1]
    near = None
    for mx, my, row, _own in marks or ():
        r2 = (x - mx) ** 2 + (y - my) ** 2
        if r2 <= mark_radius ** 2 and (near is None or r2 < near[0]):
            near = (r2, int(row))
    return None if near is None else near[1]


def ellipsoid_text(res: dict, k: int) -> str:
    """'26.41, 17.80, 13.60 Mpc, R_ell = (a b c)^⅓ 18.31 Mpc': the semi-axes a, b, c of void k's drawn ellipsoid
    (drawn_ellipsoids) and its R_ell; '' where it has none."""
    d = drawn_ellipsoids(res)
    if not d["ok"][k]:
        return ""
    abc = ", ".join(f"{float(s):.2f}" for s in d["semi"][k])
    return f"{abc} Mpc, R_ell = (a b c)^⅓ {float(d['r'][k]):.2f} Mpc"


def void_info(res: dict, k: int, short: bool = False) -> str:
    """What the inspector says about catalogue row k of a void_outlines.py result: R_v (every definition the
    catalogue has), the drawn ellipsoid (drawn_ellipsoids: its semi-axes and R_ell), the catalogue's own fit (R_eff,
    its a, b, c), the central delta, the resolved / distinct flags and the centre. 'short' = one line for the hover
    readout."""
    def mpc(v):
        return "none" if v is None or v != v else f"{float(v):.2f} Mpc"
    r_v = res["r_v_mpc"][k]
    others = {d: v[k] for d, v in (res.get("r_v_defs") or {}).items() if d != "crossing"}
    delta = res["delta"][k]
    d_text = "none" if delta is None else f"{float(delta):.3f}"
    flags = ", ".join([("resolved" if res["resolved"][k] else "not resolved"),
                       ("distinct" if res["distinct"][k] else "not distinct")])
    drawn = drawn_ellipsoids(res)
    r_ell = f"R_ell {mpc(drawn['r'][k])}, " if drawn["ok"][k] else ""
    if short:
        return f"void {k}: R_v {mpc(r_v)}, {r_ell}R_eff {mpc(res['r_eff_mpc'][k])}, δ {d_text}, {flags}"
    c = res["centres_mpc"][k]
    labels = {"delta08": "enclosed Δ reaches −0.8", "ridge": "compensation ridge"}
    radii = [f"R_v {mpc(r_v)} (back to the mean)"] + [f"{mpc(v)} ({labels.get(d, d)})" for d, v in others.items()]
    ell = ellipsoid_text(res, k)
    lines = [f"<b>void {k}</b> · {flags} · centre ({', '.join(f'{float(v):.2f}' for v in c)}) Mpc",
             " · ".join(radii),
             (f"drawn ellipsoid (where the fit returns to the mean density): a, b, c = {ell}" if ell else
              "no ellipsoid at the mean density (δ not below the mean, or not a minimum along every axis): its "
              "centre is marked"),
             f"catalogue's fit: R_eff {mpc(res['r_eff_mpc'][k])} ((a b c)^⅓{_semi_axes_text(res, k)}) · "
             f"central δ {d_text}"]
    return "<br>".join(lines)


def available_gb() -> float | None:
    """What the other programs leave free now (GB; None when unreadable): tools/void_outlines.available_bytes,
    the reading the catalogue build gate itself uses (DTFE_FREE_GB, psutil, vm_stat, MemAvailable)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_void_outlines_mem",
                                                  REPO_ROOT / "python" / "tools" / "void_outlines.py")
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        b = mod.available_bytes()
    except (OSError, ImportError, AttributeError, ValueError):
        return None
    return None if b is None else b / 1e9


# ---------------------------------------------------------------------- discovery
def _tng_box(snapdir: Path) -> tuple[tuple, float | None] | tuple[None, None]:
    """((lo, hi), redshift) of a TNG-layout snapshot from its combined file (h-free ckpc -> Mpc)."""
    files = sorted(Path(snapdir).glob("combined_[0-9][0-9][0-9].hdf5"))
    if not files:
        return None, None
    try:
        import h5py
        with h5py.File(files[0], "r") as h:
            box = float(h["Header"].attrs["BoxSize"]) / 1000.0
            z = float(h["Header"].attrs.get("Redshift", 0.0))
        return ((0.0, 0.0, 0.0), (box, box, box)), z
    except Exception:
        return None, None


def hubble(o) -> float | None:
    """h of an output's box, for the saved figures' Mpc/h axes, or None when it is not known: the
    HubbleParam of the snapshot the box was read from, when that box is in h-free Mpc. A TNG-layout output: its
    snapdir's combined file (merge_HDF5.py writes it h-free, the box Explore uses); a custom or opened run: its
    sidecar's HDF5 input when that file carries the 'HFreeUnits' mark (else its box may already be in Mpc/h).
    Read once per output (a header only) and cached on it."""
    if hasattr(o, "_hubble"):
        return o._hubble
    h = None
    if getattr(o, "box", None) is not None:
        try:
            import h5py
            if getattr(o, "sim", ""):
                files = sorted(Path(o.dir).glob("combined_[0-9][0-9][0-9].hdf5"))
                src, need_mark = (files[0] if files else None), False
            else:
                src, need_mark = (sidecar(o.root) or {}).get("input_file"), True
            if src:
                with h5py.File(src, "r") as f:
                    a = f["Header"].attrs
                    if not need_mark or "HFreeUnits" in a:
                        v = float(a["HubbleParam"]) if "HubbleParam" in a else float(a["HFreeUnits"])
                        h = v if 0.1 < v < 2.0 else None
        except Exception:
            h = None
    try:
        o._hubble = h
    except AttributeError:
        pass
    return h


def sidecar(root) -> dict | None:
    """The settings a custom-snapshot job saved beside its outputs ('<prefix>.gui.json')."""
    p = Path(str(root) + ".gui.json")
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


PS_PREFIX, DTFE_PREFIX = "ps_output", "output"       # the run scripts' default output prefixes


def estimator_label(o) -> str:
    """'PS-DTFE' or 'DTFE' for an output. The run's sidecar says (a custom-snapshot job records its
    estimator); else a streams grid decides -- only the phase-space binary writes one; else the prefix
    (the scripts' ps_* are phase-space). A phase-space run of the density alone has no streams grid, so
    the absence of one never means DTFE on its own. Cached on the object."""
    cached = getattr(o, "_estimator", None)
    if cached:
        return cached
    side = sidecar(o.root) or {}
    if side.get("estimator") in ("ps", "dtfe"):
        lab = "PS-DTFE" if side["estimator"] == "ps" else "DTFE"
    elif "streams" in o.files:
        lab = "PS-DTFE"
    else:
        lab = "PS-DTFE" if o.prefix.startswith("ps") else "DTFE"
    o._estimator = lab
    return lab


def type_label(o) -> str:
    """The figure grid's Type entry: the estimator, with the prefix when it is not that estimator's
    default one, so two phase-space runs of one snapshot stay apart ('PS-DTFE', 'PS-DTFE · ps_mw', 'DTFE')."""
    lab = estimator_label(o)
    default = PS_PREFIX if lab == "PS-DTFE" else DTFE_PREFIX
    return lab if o.prefix == default else f"{lab} · {o.prefix}"


def type_order(o, pref=None) -> tuple:
    """Sort key of a snapshot's outputs for the Type menu and the composer's fills: the estimator (DTFE
    first), then that estimator's default prefix, then the user's preferred one ('pref'), then the rest
    by prefix -- plain text order would put ps_mw (an A/B run kept beside ps_output) before the production
    run, so 'this snapshot across outputs' would compare the standard DTFE with the wrong phase-space run."""
    lab = estimator_label(o)
    default = PS_PREFIX if lab == "PS-DTFE" else DTFE_PREFIX
    return (lab, o.prefix != default, o.prefix != pref, o.prefix)


def custom_box(settings: dict) -> tuple | None:
    """A custom-snapshot run's box in Mpc: 'box_mpc', else its --box (file units) / MpcUnit, else the
    input snapshot's header BoxSize / MpcUnit. Two corners of 'dim' (the sidecar's, 3 if absent) entries."""
    unit = float(settings.get("mpc_unit") or 1.0)
    dim = settings.get("dim") if settings.get("dim") in (2, 3) else 3
    box = settings.get("box_mpc") or []
    if len(box) != 2 * dim and len(settings.get("box") or []) == 2 * dim:
        box = [v / unit for v in settings["box"]]
    if len(box) != 2 * dim and settings.get("input_file"):
        try:
            import h5py
            with h5py.File(settings["input_file"], "r") as h:
                b = float(h["Header"].attrs["BoxSize"]) / unit
            box = [0.0, b] * dim
        except Exception:
            box = []
    if len(box) == 2 * dim:
        return tuple(box[0::2]), tuple(box[1::2])
    return None


def custom_redshift(settings: dict) -> float | None:
    """The redshift of a custom-snapshot run's input, for the u-unit velocity scaling (sqrt(a), as FieldSet and the
    TNG layout apply it): the sidecar's 'redshift' (the Custom tab records the Gadget header's), else
    the header of its HDF5 input (input type 105) when it carries one; None = no scaling (no header, another input
    type, no Redshift attribute: a non-cosmological snapshot has no defined a). Without it a z > 0 Gadget snapshot
    would show its velocities in u-units, 1/sqrt(a) too large (1.41x at z = 1)."""
    if "redshift" in settings:          # recorded (None = the header has none): no need to open the snapshot
        try:
            return None if settings["redshift"] is None else float(settings["redshift"])
        except (TypeError, ValueError):
            return None
    f = settings.get("input_file")
    if not f or int(settings.get("input_type") or 105) != 105:
        return None
    try:
        import h5py
        with h5py.File(Path(f).expanduser(), "r") as h:
            hd = h["Header"].attrs
            return float(hd["Redshift"]) if "Redshift" in hd else None
    except Exception:
        return None


def sim_sort_key(name: str) -> tuple:
    """Natural order for simulation names: TNG50 before TNG100 before TNG300 (text sorting puts
    TNG300 first, since '3' < '5')."""
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name))


def find_outputs(data_root, extra_dirs=()) -> list[tuple[str, OutputSet]]:
    """(menu title, OutputSet) for every run with grids: the TNG layout under the data root, then
    the custom-snapshot output directories."""
    out = []
    root = Path(data_root).expanduser()
    seen = set()
    if str(data_root).strip() and root.is_dir():         # an empty root is none, not the working directory
        for pattern in ("*/snapdir_*/*.a_den", "*/*/snapdir_*/*.a_den", "*/snapdir_*/*.den", "*/*/snapdir_*/*.den"):
            for f in sorted(root.glob(pattern)):
                prefix = f.stem                 # 'snap_z0.5.a_den' -> 'snap_z0.5' (no grid suffix has a dot)
                key = f.parent / prefix
                if key in seen:
                    continue
                seen.add(key)
                box, z = _tng_box(f.parent)
                snap = f.parent.name.split("_")[-1]
                title = f"{f.parent.parent.name} · {snap}" + (f" (z={z:.2f})" if z is not None else "") + f" · {prefix}"
                o = OutputSet(key, box=box, redshift=z, title=title, sim=f.parent.parent.name,
                              snap=int(snap) if snap.isdigit() else None)
                if o:
                    out.append((title, o))
    for d in extra_dirs:
        for f in sorted(Path(d).glob("*.a_den")) + sorted(Path(d).glob("*.den")):
            prefix = f.stem
            key = f.parent / prefix
            if key in seen:
                continue
            seen.add(key)
            s = sidecar(key) or {}
            o = OutputSet(key, box=custom_box(s), redshift=custom_redshift(s), title=f"custom · {prefix} ({key.parent.name})",
                          dim=s.get("dim"))
            if o:
                out.append((o.title, o))
    return out


def machine_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 16 * 2 ** 30


def server_cost(snapshot, phase_space: bool = True, ram_bytes: int | None = None) -> dict | None:
    """What starting a query server on this snapshot will take, before it is started -- the
    binary's own serve model (auto_tune.h autoTuneServe) in round numbers: particles, memory of
    one tessellation, whether it will be split (budget 60% of RAM) and then the cache on disk
    (~150 B per cached vertex, the padded partitions hold ~4x the particles). None if unreadable."""
    try:
        import h5py
        with h5py.File(snapshot, "r") as h:
            hd = h["Header"].attrs
            n = int(np.sum(hd["NumPart_Total"])) if "NumPart_Total" in hd else int(np.sum(hd["NumPart_ThisFile"]))
    except Exception:
        return None
    ram = ram_bytes or machine_ram_bytes()
    budget = 0.6 * ram
    per_vertex = (658 if phase_space else 640) + 160
    # the standard server's periodic copies: its '--padding' (dtfelib.estimator.STANDARD_PADDING mean spacings) on
    # each face -- 1.18x the particles of TNG50-3, 1.37x of TNG50-4
    from dtfelib.estimator import STANDARD_PADDING
    copies = 1.73 if phase_space else (1.0 + 2.0 * STANDARD_PADDING / max(n, 1) ** (1.0 / 3.0)) ** 3
    single = n * copies * per_vertex + n * 64
    split = single > budget
    cache = n * 4.1 * 150 if split else n * 1.73 * 150
    return {"particles": n, "single_gb": single / 1e9, "budget_gb": budget / 1e9, "split": split,
            "cache_gb": cache / 1e9}


# ---------------------------------------------------------------- the tessellation cache on disk
# What the query server (and any run with --tessellation-cache) leaves in its cache folder, read
# without the binary: each partition's tessellation 'tess_<hash>_p<k>.tess' starts with its full
# descriptor (tessellation_cache.h makeDescriptor), and beside it '<that>.occ' holds the partition's
# 64^3 occupancy map -- the buckets of the box its streams reach (ps_point_eval.cc).
_TESS_MAGIC = b"DTFETESS"
_OCC_MAGIC = b"DTFEOCC1"
OCC_N = 64


def _read_descriptor(f, magic: bytes) -> str | None:
    import struct
    if f.read(8) != magic:
        return None
    raw = f.read(4)
    if len(raw) != 4:
        return None
    (n,) = struct.unpack("<I", raw)
    if not 0 < n < (1 << 20):
        return None
    return f.read(n).decode("utf-8", errors="replace")


def _descriptor_fields(desc: str) -> dict:
    out = {}
    for line in desc.splitlines():
        key, _, val = line.partition("=")
        if key and key not in out:
            out[key] = val
    return out


def _canonical(path) -> str:
    """The binary's canonical_path.h: realpath, so a symlinked or relative name matches the cache key
    the binary wrote (it canonicalizes); an unresolvable path is kept as given."""
    try:
        return os.path.realpath(os.path.expanduser(str(path)))
    except OSError:
        return str(path)


def _file_stamp(path) -> str:
    """The C++ fileStamp: '<size>:<mtime seconds>'."""
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return f"{st.st_size}:{int(st.st_mtime)}"


def cached_partition_sets(cache_dir, snapshot, lagrangian=None, periodic: bool | None = None) -> dict:
    """This snapshot's partition tessellations in a tessellation cache, by partition triple (nx, ny, nz)
    (a pair for a 2D run):
    a list of {'tess', 'occ' (Path or None), 'bytes', 'region' (6 floats), 'periodic'} per cached
    partition. Matched on the descriptor's input line (path and size:mtime stamp), the Lagrangian file
    and the periodicity -- only to choose a split and estimate a run: the binary itself compares the
    whole descriptor before it uses a file."""
    sets: dict = {}
    d = Path(cache_dir).expanduser() if cache_dir else None
    if d is None or not d.is_dir():
        return sets
    want_input = f"{_canonical(snapshot)} {_file_stamp(snapshot)}"
    for tess in sorted(d.glob("tess_*.tess")):
        try:
            with open(tess, "rb") as f:
                desc = _read_descriptor(f, _TESS_MAGIC)
        except OSError:
            continue
        if not desc:
            continue
        fl = _descriptor_fields(desc)
        if fl.get("input") != want_input:
            continue
        if lagrangian and not fl.get("lagrange", "").startswith(f"{_canonical(lagrangian)} "):
            continue
        per = fl.get("periodic", "1").strip() == "1"
        if periodic is not None and per != bool(periodic):
            continue
        part = fl.get("partition", "")
        try:
            on, triple = part.split(" ", 1)
            nums = tuple(int(v) for v in triple.split(" ")[0].strip(",").split(",") if v)
        except ValueError:
            continue
        if on != "1" or len(nums) not in (2, 3):         # a 2D run's split has two numbers
            continue
        try:
            region = [float(v) for v in fl.get("region", "").split(" ", 1)[1].strip(",").split(",")]
        except (IndexError, ValueError):
            region = []
        occ = Path(str(tess) + ".occ")
        sets.setdefault(nums, []).append({"tess": tess, "occ": occ if occ.is_file() else None,
                                          "bytes": tess.stat().st_size, "region": region, "periodic": per})
    return sets


def tess_chunked(path) -> bool:
    """A cached tessellation written with the chunked body (inflated on several threads)."""
    try:
        with open(path, "rb") as f:
            if _read_descriptor(f, _TESS_MAGIC) is None:
                return False
            return f.read(8) == b"DTFECHK1"
    except OSError:
        return False


def globals_recorded(cache_dir, snapshot, lagrangian=None) -> bool:
    """Is this snapshot's particle-derived globals record (ps_globals_record.h) in the cache? A window
    run whose needed partitions are all cached then never reads the snapshot. Matched on the record's
    own input and Lagrangian lines (paths and size:mtime stamps); the binary compares its whole key."""
    d = Path(cache_dir).expanduser() if cache_dir else None
    if d is None or not d.is_dir():
        return False
    want_input = f"input={_canonical(snapshot)} {_file_stamp(snapshot)}"
    want_lag = f"lagrange={_canonical(lagrangian)} {_file_stamp(lagrangian)}" if lagrangian else None
    for rec in d.glob("globals_*.txt"):
        try:
            lines = rec.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if want_input in lines and (want_lag is None or want_lag in lines) and "end" in lines:
            return True
    return False


def read_occupancy(path):
    """(n vertices, the 64^3 occupancy bits as a bool array [i, j, k]) of a '.occ' map, or None."""
    import struct
    try:
        with open(path, "rb") as f:
            if _read_descriptor(f, _OCC_MAGIC) is None:
                return None
            n, nvert, words = struct.unpack("<IQQ", f.read(20))
            if n != OCC_N or words not in (0, OCC_N ** 3 // 64):
                return None
            raw = np.frombuffer(f.read(8 * words), dtype="<u8")
    except (OSError, struct.error):
        return None
    if words == 0:
        return nvert, np.zeros((OCC_N,) * 3, bool)
    bits = np.unpackbits(raw.view(np.uint8), bitorder="little").astype(bool)
    return nvert, bits[: OCC_N ** 3].reshape((OCC_N,) * 3)


def occupancy_touches(occ_bits, region, periodic: bool, window) -> bool:
    """Does an occupancy map reach a window ((3, 2) in Mpc)? The binary's own test (ps_point_eval.cc
    psOccupancyTouchesWindow): the window's bucket range widened by one bucket on each side."""
    idx = []
    for d in range(3):
        lo, length = region[2 * d], region[2 * d + 1] - region[2 * d]
        w = OCC_N / length
        b0 = int(np.floor((window[d][0] - lo) * w)) - 1
        b1 = int(np.floor((window[d][1] - lo) * w)) + 1
        if periodic:
            if b1 - b0 + 1 >= OCC_N:
                b0, b1 = 0, OCC_N - 1
            idx.append(np.arange(b0, b1 + 1) % OCC_N)
        else:
            idx.append(np.arange(max(0, min(b0, OCC_N - 1)), max(0, min(b1, OCC_N - 1)) + 1))
    return bool(occ_bits[np.ix_(*idx)].any())


def find_pts_planes(o) -> list[tuple[str, Path]]:
    """The point-evaluated hi-res slices of an output: '<root>.pts_den' and every plane sidecar (tools/
    make_image_plane.py, in the simulation's folder or beside the output) whose geometry matches the file's
    size -- (label, sidecar). The '.pts_*' files carry no axis, so planes of one size (the x, y, z views)
    are all listed, each with its axis: the user picks."""
    den = Path(str(o.root) + ".pts_den")
    if not den.is_file():
        return []
    size = den.stat().st_size
    out, seen = [], set()
    for side in (sorted(o.dir.parent.glob("pointeval_plane_*.json")) + sorted(o.dir.parent.glob("hires_plane_*.json"))
                 + sorted(o.dir.glob("*plane*.json"))):
        if side in seen:
            continue
        seen.add(side)
        try:
            d = json.loads(side.read_text())
            k = int(d.get("supersample", 1))
            n = int(d["planes"]) * int(d["nu"]) * k * int(d["nv"]) * k
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if n * 8 == size:
            out.append((f"{d['nu']}×{d['nv']} {d.get('axis', '?')}-plane at {float(d.get('center', 0.0)):.1f} Mpc "
                        f"({side.stem})", side))
    return out


def server_settings(out: OutputSet, snapshot: str | None = None) -> dict | None:
    """dtfelib.Estimator arguments for the snapshot behind an output set, or None when unknown.
    TNG layout: the snapdir's combined file (+ the simulation's converted ICs when present);
    a custom-snapshot run: the job's own settings sidecar. 'snapshot': a file chosen by hand (Explore's
    'Snapshot…', when the input moved or the output has no sidecar) -- it replaces the one found."""
    s = sidecar(out.root)
    if snapshot and not Path(snapshot).is_file():
        return None
    if s:
        if snapshot:
            s = dict(s, input_file=str(snapshot))
        if not s.get("input_file") or not Path(s["input_file"]).is_file():
            return None
        dim = s.get("dim") if s.get("dim") in (2, 3) else 3
        kw = {"snapshot": s["input_file"], "phase_space": s.get("estimator", "ps") == "ps",
              "periodic": bool(s.get("periodic", True)), "mpc_unit": float(s.get("mpc_unit", 1.0)),
              "input_type": int(s.get("input_type", 105))}
        if dim == 2:
            kw["dim"] = 2               # the 2D server (Estimator: the -2d binary, 2-column points)
        if s.get("lagrangian_file"):
            kw["lagrangian_input"] = s["lagrangian_file"]
        if len(s.get("box") or []) == 2 * dim:
            kw["options"] = ["--box"] + [repr(float(v)) for v in s["box"]]
        # a non-periodic cloud's alpha shape, as the run (and its exact zoom) used it: the server's
        # answers and its cached occupancy maps then keep the same tetrahedra
        if not kw["periodic"] and kw["phase_space"] and float(s.get("alpha_shape", 3.0)) != 3.0:
            kw["options"] = list(kw.get("options", [])) + ["--ps-alpha-shape", f"{float(s['alpha_shape']):g}"]
        return kw
    combined = [Path(snapshot)] if snapshot else sorted(out.dir.glob("combined_[0-9][0-9][0-9].hdf5"))
    if not combined:
        return None
    kw = {"snapshot": str(combined[0]), "phase_space": not out.prefix == "output", "periodic": True}
    ics = combined[0].parent.parent / "combined_ics.hdf5"
    if kw["phase_space"] and ics.is_file():
        kw["lagrangian_input"] = str(ics)
    return kw
