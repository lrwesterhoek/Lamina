"""Unified loader for DTFE / PS-DTFE raw field grids: FieldSet hides what differs between the estimators (file
naming, density units, metadata from the combined_*.hdf5 header, PS-only fields, the velocity-gradient family)
and load() auto-tunes eager vs memmap reads. Usage: FieldSet(snapdir).density(units="mean"), .load("velocity")."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# canonical field name -> (file suffix, components, ps_only); the 'a_' averaged prefix is inserted automatically.
# Velocity-gradient family under PS-DTFE: a face-gradient run (the default where it qualifies) stores the MEAN flow's
# grad<v> and writes '.a_velDivStreams' (FieldSet.face_gradient); any other run stores the STREAM AVERAGE <grad v>,
# which differs from standard DTFE's in folds. load(name, mean_field=True) gives the mean-field derivatives for BOTH
# methods (fields.mean_field_derivatives); '--ps-gradient stored' restores the stored grids.
FIELDS = {
    "density":           ("den",           1, False),
    "velocity":          ("vel",           3, False),
    "streams":           ("streams",       1, True),
    "hidden_streams":    ("hidden_streams", 1, True),  # bitmask: bit 1 = multi-stream volume the stream count does not show (a fold thinner than the sampling), bit 2 = mass from a tetrahedron none of whose sub-samples lies in the cell. Single-stream mask = velocity_single_stream; '.unresolved' outputs hold a 0/1 mass flag instead (read as bit 1)
    "caustic":         ("caustic",       1, True),   # 0/1 fold-caustic cell flag (--ps-caustics): tetrahedra of BOTH orientations overlap the cell. Older grids miss folds thinner than the sampling: read the fold with FieldSet.fold_mask()
    "caustic_class":     ("causticClass",  1, True),   # --ps-caustics: caustic stratification bitmask (bits 0/1 parity = fold, 2-5 collapsed axes, 6 umbilic indicator, 7 A3 cusp, 8 A4 swallowtail; per sample); see src/CGAL_triangulation/ps_caustic_class.h and io.caustic_collapse_max / caustic_is_fold
    "tet_touch":         ("tetTouch",      1, True),   # --exact: raw count of tetrahedra touching the cell (NOT a stream count; use 'streams')
    "dispersion":        ("velDisp",       1, True),
    "dispersion_tensor": ("velDispTensor", 6, True),   # xx,xy,xz,yy,yz,zz
    "divergence":        ("velDiv",        1, False),   # PS: grad<v> (face gradient) or <grad v> (note above FIELDS)
    "shear":             ("velShear",      5, False),   # PS: grad<v> (face gradient) or <grad v> (note above FIELDS)
    "vorticity":         ("velVort",       3, False),   # PS: grad<v> (face gradient) or <grad v> (note above FIELDS)
    "gradient":          ("velGrad",       9, False),   # PS: grad<v> (face gradient) or <grad v> (note above FIELDS)
    "tweb":              ("tweb",          1, False),
    "tweb_eigenvalues":  ("twebEig",       3, False),
    "vweb":              ("velVweb",       1, False),   # PS: from grad<v> (face gradient) or <grad v> (see FIELDS)
    "vweb_eigenvalues":  ("velVwebEig",    3, False),   # PS: from grad<v> (face gradient) or <grad v> (see FIELDS)
    # the face gradient: the velocity-gradient family above is then the MEAN flow's grad<v> (from exact face sums),
    # and this one holds the divergence of the stream average <grad v>; these three its gradient, shear and
    # vorticity, each where the run wrote the mean flow's. load(name, stream_average=True) reads them
    "divergence_streams": ("velDivStreams", 1, True),
    "gradient_streams":   ("velGradStreams", 9, True),
    "shear_streams":      ("velShearStreams", 5, True),
    "vorticity_streams":  ("velVortStreams", 3, True),
    # ... and the mass flux divergence div(sum_s rho_s v_s) (rho/rho_bar times velocity per Mpc) from exact face momentum
    # sums: by continuity -a d(1+delta)/dt, the rate a region gains or loses mass, caustic sheets included
    "mass_flux_divergence": ("massFluxDiv", 1, True),
}

_PREFIX = {"ps": "ps_output.", "dtfe": "output."}

# The velocity-gradient family load(..., mean_field=True) serves (fields.MEAN_FIELD_NAMES, kept here too so that
# importing io never needs scipy), and the V-web threshold of a run whose log does not say it (the run scripts'
# LAMBDA_TH default: 0.3).
MEAN_FIELD_NAMES = ("gradient", "divergence", "shear", "vorticity", "vweb", "vweb_eigenvalues")
DEFAULT_LAMBDA_TH = 0.3
# The family load(..., stream_average=True) serves: the stream average <grad v> under either gradient route
STREAM_AVERAGE_NAMES = ("gradient", "divergence", "shear", "vorticity")

# Fields the binary writes ONCE, with no 'a_' (averaged) counterpart: the deposit produces a
# single grid for them regardless of the u/a pass (see writeOutputData in src/input_output.cc).
# Without this, averaged=True (the default) would look for a nonexistent 'ps_output.a_caustic'.
_NO_AVG_VARIANT = {"caustic", "caustic_class", "tet_touch"}
# a field's file suffix before a rename: read when the current name is absent (old outputs)
LEGACY_SUFFIX = {"hidden_streams": "unresolved"}


# Bit layout of the 'caustic_class' mask, mirroring src/CGAL_triangulation/ps_caustic_class.h.
CAUSTIC_PARITY_POS = 1 << 0
CAUSTIC_PARITY_NEG = 1 << 1
CAUSTIC_COLLAPSE   = (1 << 2, 1 << 3, 1 << 4, 1 << 5)   # indexed by the collapse multiplicity k
CAUSTIC_DEGENERATE  = 1 << 6
CAUSTIC_CUSP        = 1 << 7
CAUSTIC_SWALLOWTAIL = 1 << 8


def caustic_is_cusp(mask):
    """Boolean mask of the A3 CUSP indicator (bit 7) of a 'caustic_class' / '.pts_caustic' grid. Only set when the
    run passed '--ps-caustic-cusps'. A CANDIDATE MAP, not a classification: the gradient it needs is a least-squares
    estimate over neighbouring tetrahedra, at the tessellation's resolution."""
    import numpy as np
    return (np.asarray(mask).astype(np.int64) & CAUSTIC_CUSP) != 0


def caustic_is_swallowtail(mask):
    """Boolean mask of the A4 SWALLOWTAIL indicator (bit 8) of a 'caustic_class' grid (same '--ps-caustic-cusps'
    run). PROXIMITY, not membership: A4 points are isolated, so the bit marks a band along the cusp curve.
    Not a subset of caustic_is_cusp(); OR the two for the whole A3-and-above skeleton."""
    import numpy as np
    return (np.asarray(mask).astype(np.int64) & CAUSTIC_SWALLOWTAIL) != 0


_FOLD_SLAB_CELLS = 16 << 20     # caustic_is_fold's slab size in cells (a module constant: the tests shrink it)


def caustic_is_fold(mask=None, streams=None, hidden=None, tol: float | None = None):
    """Boolean mask of the fold (A2) condition: tetrahedra of BOTH orientations overlap the cell/point. fold =
    (streams > 1 + tol) | (hidden_streams & 1) [| both parity bits of mask]: in a periodic box a point is multi-stream
    exactly where a folded tetrahedron covers it. Periodic runs only; inputs may be memmaps (axis-0 slabs)."""
    import numpy as np
    pair = CAUSTIC_PARITY_POS | CAUSTIC_PARITY_NEG
    if streams is None:
        if mask is None:
            raise ValueError("caustic_is_fold needs a caustic_class mask, or the streams grid (with hidden_streams)")
        if hidden is not None:
            raise ValueError("caustic_is_fold: 'hidden' needs 'streams' too (the union is streams > 1 OR hidden bit 1)")
        m = np.asarray(mask).astype(np.int64)
        return (m & pair) == pair
    tol = STREAM_TOL if tol is None else float(tol)
    shape = np.shape(streams)
    for name, other in (("mask", mask), ("hidden", hidden)):
        if other is not None and np.shape(other) != shape:
            raise ValueError(f"caustic_is_fold: {name} has shape {np.shape(other)}, streams {shape}")

    def part(sl):
        s = np.asarray(streams[sl], dtype=np.float32)
        f = (s - 1.0) > tol                                   # = ~single_stream_mask(s) & not empty
        if hidden is not None:
            f |= (np.rint(np.asarray(hidden[sl], dtype=np.float32)).astype(np.int32) & 1) != 0
        if mask is not None:
            f |= (np.asarray(mask[sl]).astype(np.int64) & pair) == pair
        return f

    if len(shape) == 0:
        return part(())
    out = np.empty(shape, dtype=bool)
    per = max(1, int(np.prod(shape[1:], dtype=np.int64)))
    step = max(1, _FOLD_SLAB_CELLS // per)                    # ~16M cells per slab: ~200 MB of temporaries
    for a in range(0, shape[0], step):
        sl = slice(a, min(a + step, shape[0]))
        out[sl] = part(sl)
    return out


def caustic_collapse_max(mask):
    """Highest collapse multiplicity k present in each cell of a 'caustic_class' grid: 0 = single-stream,
    1 = wall, 2 = filament, 3 = node; -1 where nothing was deposited."""
    import numpy as np
    m = np.asarray(mask).astype(np.int64)
    out = np.full(m.shape, -1, dtype=np.int8)
    for k, bit in enumerate(CAUSTIC_COLLAPSE):
        out = np.where((m & bit) != 0, np.int8(k), out)
    return out

# Velocity-derived fields are stored in Gadget u-units (u = v_pec/sqrt(a)). FieldSet.load() converts them to
# peculiar km/s by scaling with a**exp: 0.5 for first moments, 1.0 for second moments (the dispersion is a
# velocity VARIANCE). At z=0 the factor is exactly 1.0 (bit for bit). NOT listed: vweb/vweb_eigenvalues -- the
# BINARY applies sqrt(a) itself (scaling here would double-correct); tweb, caustic, streams, tetTouch are
# dimensionless or density-based.
_VELOCITY_SCALE_EXP = {
    "velocity": 0.5, "divergence": 0.5, "shear": 0.5, "vorticity": 0.5, "gradient": 0.5,
    "divergence_streams": 0.5, "gradient_streams": 0.5, "shear_streams": 0.5, "vorticity_streams": 0.5,
    "mass_flux_divergence": 0.5,
    "dispersion": 1.0, "dispersion_tensor": 1.0,
}

# Tolerance for comparing 'streams' against an integer multiplicity: '.streams' is a float32 under every deposit,
# so an exact '== 1' test is never correct.
STREAM_TOL = 1e-3


# ---------------------------------------------------------------- machine awareness
# The Python loader's equivalent of the C++ auto-tune (src/auto_tune.h): FieldSet.load(mode="auto") returns an
# eager ndarray whenever the field fits the budget, else a memmap. Env overrides (also what makes tests
# deterministic): DTFE_PY_RAM_GB (available-RAM override in GB), DTFE_PY_LOAD_FRAC (fraction of available RAM
# one eager load may use, default 0.5).

def _available_ram_bytes() -> int:
    """Available (not total) RAM in bytes, best effort across platforms."""
    import os
    env = os.environ.get("DTFE_PY_RAM_GB")
    if env:
        try:
            v = float(env)
            if v > 0:
                return int(v * 1e9)
        except ValueError:
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:  # macOS: total physical RAM (no portable 'available'; the 0.5 budget absorbs the gap)
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if out.returncode == 0:
            return int(out.stdout.strip())
    except (OSError, ValueError):
        pass
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(8e9)  # conservative fallback


def _load_budget_bytes() -> int:
    """Bytes one eager load() may allocate before mode='auto' switches to memmap."""
    import os
    frac = 0.5
    env = os.environ.get("DTFE_PY_LOAD_FRAC")
    if env:
        try:
            v = float(env)
            if 0 < v <= 1:
                frac = v
        except ValueError:
            pass
    return int(frac * _available_ram_bytes())


# the PS-DTFE run-log header's line on the velocity moments (src/user_options.cc, printed by every PS-DTFE run beside
# the command line): 'velocity moments : volume-weighted (the dispersion mass-weighted)'. A log of an older binary
# has none (python/gui/results.py _MOMENTS reads it the same way)
_MOMENTS_LINE = r"^\s*velocity moments\s+:\s+volume-weighted"


def run_velocity_weighting(method: str, runlog) -> tuple[str, str]:
    """How the velocity grid of a 'method' run whose log is 'runlog' was deposited: ('volume' | 'mass', why). A coarser
    view must be the mean a coarser run of the SAME deposit gives. A PS log with neither the header line 'velocity
    moments : volume-weighted' nor the flag is an older mass-weighted run; without the log, volume is assumed, and said."""
    import re
    if method != "ps":
        return "volume", "standard DTFE: the volume average"
    log = Path(runlog)
    try:
        raw = log.read_text(errors="replace")
    except OSError:
        return "volume", f"no {log.name}: the volume weighting (PS-DTFE's) assumed"
    if re.search(_MOMENTS_LINE, raw, re.M):
        return "volume", f"{log.name}: its header line 'velocity moments : volume-weighted'"
    if re.search(r"(^|\s)--ps-volume-weighted(\s|$)", raw, re.M):
        return "volume", f"{log.name}: --ps-volume-weighted"
    return "mass", (f"{log.name}: no --ps-volume-weighted and no 'velocity moments' header line "
                    "(an older PS-DTFE run, mass-weighted)")


def run_lambda_th(runlog) -> float:
    """The web threshold of the run whose log is 'runlog': the last '--lambda-th X' in it, else the binary's own
    T-web report '(lambda_th = X)', else DEFAULT_LAMBDA_TH (0.3; also for a log that cannot be read)."""
    import re
    try:
        text = Path(runlog).read_text(errors="replace")
    except OSError:
        return DEFAULT_LAMBDA_TH
    # the option first; the report's 'lambda_th =' is not an option
    found = (re.findall(r"--lambda-th[ =]+([-+0-9.eE]+)", text)
             or re.findall(r"lambda_th = ([-+0-9.eE]+)", text))
    try:
        return float(found[-1]) if found else DEFAULT_LAMBDA_TH
    except ValueError:
        return DEFAULT_LAMBDA_TH


# the run-log header's line on the velocity gradient's route (src/user_options.cc, printed where a field built on the
# gradient is computed): 'velocity gradient : from EXACT FACE sums, ...' for the face gradient, any other text for the
# stream average of each stream's own gradient (the launcher reads it the same way: python/gui/results.py _GRADIENT)
_GRADIENT_LINE = r"^\s*velocity gradient\s+:\s+(from EXACT FACE)?"
# the warning of a pass whose face sums found no fixed-point unit: that pass keeps the stream average
# (src/CGAL_triangulation/ps_interpolation.cc), and the warning does not say which pass
_FACE_OFF_FOR_A_PASS = "'--face-gradient' is off for this pass"


def run_face_gradient(runlog) -> bool:
    """Did the PS-DTFE run whose log is 'runlog' take the face gradient? The unaveraged ('_u') grids carry no
    marker, so the route is read from the header's last 'velocity gradient' line ('from EXACT FACE sums'). False
    without the log, without that line, or where a pass's face sums found no fixed-point unit."""
    import re
    try:
        text = Path(runlog).read_text(errors="replace")
    except OSError:
        return False
    if _FACE_OFF_FOR_A_PASS in text:
        return False
    found = re.findall(_GRADIENT_LINE, text, re.M)
    return bool(found) and found[-1] != ""


def _grid_shape_of(path: Path, ncomp: int) -> tuple[int, np.dtype]:
    """(N, dtype) of a raw N^3 x ncomp grid file, from its size alone: float32 (the default
    build) or float64 (a DOUBLE=1 build). The two never collide -- N^3 = 2 M^3 has no integer
    solution -- so the size is unambiguous."""
    size = Path(path).stat().st_size
    for dt in (np.float32, np.float64):
        item = np.dtype(dt).itemsize
        n = round((size / item / ncomp) ** (1 / 3))
        if n > 0 and n ** 3 * ncomp * item == size:
            return n, np.dtype(dt)
    raise ValueError(f"{path}: {size} bytes is not an N^3 x {ncomp} float32 or float64 grid")


@dataclass
class SnapshotMeta:
    box_mpc: float        # comoving box size, h-free Mpc
    redshift: float
    hubble_h: float
    n_particles: int
    particle_mass: float  # 1e10 Msun, h-free
    rho_bar: float        # 1e10 Msun / Mpc^3, h-free (n * m / V)


class FieldSet:
    def __init__(self, snapdir, method: str = "auto", averaged: bool = True,
                 prefix: str | None = None):
        self.snapdir = Path(snapdir)
        if not self.snapdir.is_dir():
            raise FileNotFoundError(f"snapshot directory not found: {self.snapdir}")
        self.averaged = averaged
        if prefix is not None:
            # alternate on-disk prefix (run_ps_dtfe.sh's OUTPUT_PREFIX override), e.g. 'ps_mw'. Only the PS run
            # script writes alternate prefixes, so method 'auto' resolves to 'ps'; pass method='dtfe' for a
            # hand-produced standard-DTFE set.
            prefix = prefix.rstrip(".") + "."
            self.method = "ps" if method == "auto" else method
        else:
            self.method = self._detect_method() if method == "auto" else method
        if self.method not in _PREFIX:
            raise ValueError(f"method must be 'ps', 'dtfe' or 'auto', got {method!r}")
        self.prefix = _PREFIX[self.method] if prefix is None else prefix
        if prefix is not None and not self._has_fields(prefix):
            raise FileNotFoundError(
                f"no field grids with prefix {prefix!r} in {self.snapdir} "
                f"(written with run_ps_dtfe.sh OUTPUT_PREFIX={prefix.rstrip('.')}?)")
        self.meta = self._read_meta()
        self._grid_n: int | None = None

    # ------------------------------------------------------------------ discovery
    def _has_fields(self, prefix: str) -> bool:
        # only real field grids count -- run logs and other same-prefix artifacts (e.g. the
        # ps_output.runlog an aborted run leaves behind) must not trigger detection, or a
        # snapshot with no usable PS grids would be classified 'ps' and then fail to load
        suffixes = {s for s, _, _ in FIELDS.values()} | set(LEGACY_SUFFIX.values())
        return any(p.name[len(prefix):].removeprefix("a_") in suffixes
                   for p in self.snapdir.glob(prefix + "*"))

    def _detect_method(self) -> str:
        ps = self._has_fields("ps_output.")
        dt = self._has_fields("output.")
        if ps and dt:
            # both estimators on disk: default deterministically to PS-DTFE (the primary
            # estimator of this project) rather than erroring -- but say so, loudly.
            print(f"[dtfelib] {self.snapdir.name}: both ps_output.* and output.* present; "
                  "using method='ps' (pass --estimator dtfe to switch)")
            return "ps"
        if ps: return "ps"
        if dt: return "dtfe"
        raise FileNotFoundError(f"no DTFE output fields (ps_output.*/output.*) in {self.snapdir}")

    def _field_path(self, name: str) -> Path:
        if name not in FIELDS:
            raise KeyError(f"unknown field {name!r}; known: {', '.join(FIELDS)}")
        suffix, _, ps_only = FIELDS[name]
        if ps_only and self.method != "ps":
            raise ValueError(f"field {name!r} is only produced by PS-DTFE (method='ps')")
        avg = "a_" if (self.averaged and name not in _NO_AVG_VARIANT) else ""
        path = self.snapdir / f"{self.prefix}{avg}{suffix}"
        if name in LEGACY_SUFFIX and not path.exists():
            old = self.snapdir / f"{self.prefix}{avg}{LEGACY_SUFFIX[name]}"
            if old.exists():
                return old
        return path

    @property
    def face_gradient(self) -> bool:
        """Did this PS run store the MEAN flow's velocity derivatives itself (the face gradient)? Marker: the
        stream-average divergence '.a_velDivStreams'; the unaveraged grids have no marker, so the run log's route
        decides (run_face_gradient). Standard DTFE: False."""
        if self.method != "ps":
            return False
        try:
            if self._field_path("divergence_streams").exists():
                return True
        except (KeyError, ValueError):
            return False
        return not self.averaged and run_face_gradient(self.snapdir / f"{self.prefix}runlog")

    def has(self, name: str, mean_field: bool = False, stream_average: bool = False) -> bool:
        """Is the grid on disk? mean_field=True asks whether load(name, mean_field=True) can serve it (PS-DTFE needs
        the 'velocity' grid); stream_average=True the same for load(name, stream_average=True)."""
        if mean_field and stream_average:
            raise ValueError("mean_field=True and stream_average=True ask for two different quantities: give one")
        if stream_average and self.method == "ps" and self.face_gradient and name in STREAM_AVERAGE_NAMES:
            name = f"{name}_streams"         # the face run stored the mean flow's under the plain name
        if mean_field and self.method == "ps" and name in MEAN_FIELD_NAMES:
            if self.face_gradient:           # the face gradient: the stored grids ARE the mean flow's
                return self._field_path(name).exists()
            return self.has("velocity")
        try:
            return self._field_path(name).exists()
        except (KeyError, ValueError):
            return False

    @property
    def grid_n(self) -> int:
        if self._grid_n is None:
            for probe in ("density", "velocity"):
                p = self._field_path(probe)
                if p.exists():
                    self._grid_n = _grid_shape_of(p, FIELDS[probe][1])[0]
                    break
            else:
                raise FileNotFoundError(f"no density/velocity file to infer grid size in {self.snapdir}")
        return self._grid_n

    def _dtype_of(self, name: str) -> np.dtype:
        """On-disk element type of a field's grid: float32 from the default build, float64
        from a DOUBLE=1 build ('make PS-DTFE DOUBLE=1'). Read from the file size."""
        path = self._field_path(name)
        ncomp = FIELDS[name][1]
        n = self.grid_n
        size = path.stat().st_size
        for dt in (np.float32, np.float64):
            if size == n ** 3 * ncomp * np.dtype(dt).itemsize:
                return np.dtype(dt)
        raise ValueError(f"{path}: {size} bytes is neither {n}^3 x {ncomp} float32 nor float64")

    # ------------------------------------------------------------------ metadata
    def _read_meta(self) -> SnapshotMeta:
        import h5py
        snaps = sorted(self.snapdir.glob("combined_*.hdf5"))
        if not snaps:
            raise FileNotFoundError(f"no combined_*.hdf5 snapshot in {self.snapdir} (needed for units)")
        with h5py.File(snaps[0], "r") as f:
            h = f["Header"].attrs
            box_mpc = float(h["BoxSize"]) / 1000.0          # h-free ckpc -> Mpc (see tools/merge_HDF5.py)
            npart = int(h["NumPart_ThisFile"][1])
            mass = float(h["MassTable"][1])                  # 1e10 Msun, h-free
            if mass <= 0:                                    # per-particle masses instead of table
                mass = float(f["PartType1/Masses"][0])
            if "HFreeUnits" not in h:
                # combined files from before the units unification copy the raw TNG
                # MassTable (1e10 Msun/h) verbatim; normalize so meta is h-free either way
                mass /= float(h["HubbleParam"])
            return SnapshotMeta(
                box_mpc=box_mpc,
                redshift=float(h["Redshift"]),
                hubble_h=float(h["HubbleParam"]),
                n_particles=npart,
                particle_mass=mass,
                rho_bar=npart * mass / box_mpc ** 3,
            )

    # ------------------------------------------------------------------ loading
    @property
    def velocity_scale(self) -> float:
        """sqrt(a): the u-units -> peculiar km/s factor for FIRST velocity moments (second moments scale with a).
        load() applies these per _VELOCITY_SCALE_EXP; exactly 1.0 at z=0."""
        return float(np.sqrt(1.0 / (1.0 + self.meta.redshift)))

    def _scale_factor32(self, name: str) -> np.float32 | None:
        """The in-place u-units -> peculiar-km/s factor for 'name', or None when it is 1
        (z=0, or a field that carries no velocity units)."""
        exp = _VELOCITY_SCALE_EXP.get(name)
        if exp is None:
            return None
        factor = np.float32((1.0 / (1.0 + self.meta.redshift)) ** exp)
        return None if factor == np.float32(1.0) else factor

    def field_nbytes(self, name: str) -> int:
        """On-disk size of a field's grid in bytes, without reading it."""
        return self._field_path(name).stat().st_size

    def _open_memmap(self, name: str) -> np.memmap:
        """Read-only memmap of the raw on-disk grid, reshaped and size-checked."""
        path = self._field_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} (field {name!r}, method {self.method!r})")
        ncomp = FIELDS[name][1]
        n = self.grid_n
        dtype = self._dtype_of(name)
        shape = (n, n, n) if ncomp == 1 else (n, n, n, ncomp)
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def load(self, name: str, scaled: bool = True, mode: str = "auto", mean_field: bool = False,
             stream_average: bool = False) -> np.ndarray:
        """Field as stored on disk, shape (N,N,N) or (N,N,N,ncomp), velocity-derived fields converted from Gadget u-units
        to peculiar km/s (scaled=False: raw). mode='auto' reads eagerly within the load budget, else a READ-ONLY memmap
        ('ram' / 'memmap' force either); an over-budget field needing scaling raises MemoryError. mean_field=True /
        stream_average=True: the mean-flow / stream-average velocity derivatives (MEAN_FIELD_NAMES / STREAM_AVERAGE_NAMES)."""
        if mode not in ("auto", "ram", "memmap"):
            raise ValueError(f"mode must be 'auto', 'ram' or 'memmap', got {mode!r}")
        if stream_average:
            if mean_field:
                raise ValueError("mean_field=True and stream_average=True ask for two different quantities: give one")
            if name not in STREAM_AVERAGE_NAMES:
                raise ValueError(f"stream_average=True serves {', '.join(STREAM_AVERAGE_NAMES)}; not {name!r}")
            if self.method == "ps" and self.face_gradient:
                plain, name = name, f"{name}_streams"   # the face run stored the mean flow's under the plain name
                if not self._field_path(name).exists():
                    if not self.averaged:           # (the '_u' pass's face run: face_gradient read its run log)
                        raise FileNotFoundError(
                            f"{self._field_path(name)}: the unaveraged pass of a face-gradient run writes no stream "
                            f"average of {plain!r} (the binary writes it beside the '_a' grids only); a "
                            "'--per-tet-gradient' run stores it under the plain name")
                    raise FileNotFoundError(
                        f"{self._field_path(name)}: this face-gradient run wrote no stream average of {plain!r}; it is "
                        f"written beside the mean flow's when {plain + '_a'!r} is asked (an older binary "
                        "wrote the divergence alone, 'divergence_streams')")
        if mean_field:
            if name not in MEAN_FIELD_NAMES:
                raise ValueError(f"mean_field=True serves the velocity-gradient family {', '.join(MEAN_FIELD_NAMES)}; "
                                 f"not {name!r}")
            if self.method == "ps" and not self.face_gradient:
                return self.mean_field(name, scaled=scaled)
            # a PS run with the face gradient stores the mean flow's derivatives itself (exact face sums): read them
            # standard DTFE: the stored grid IS the gradient of its one single-valued velocity field, measured by the
            # estimator itself (cell averages of the interpolant's gradient) -- read it, never re-derive it
        factor = self._scale_factor32(name) if scaled else None
        if mode == "auto":
            mode = "ram" if self.field_nbytes(name) <= _load_budget_bytes() else "memmap"
        if mode == "memmap":
            if factor is not None:
                raise MemoryError(
                    f"field {name!r} ({self.field_nbytes(name)/1e9:.1f} GB) exceeds the load "
                    f"budget ({_load_budget_bytes()/1e9:.1f} GB) but needs the u-units factor "
                    f"{float(factor):.4g} (z={self.meta.redshift:.2f}), which a read-only "
                    f"memmap cannot carry. Use load_slice() for plot planes, iter_slabs() for "
                    f"streaming access, scaled=False for raw u-units, or mode='ram' to force "
                    f"an eager load.")
            return self._open_memmap(name)
        path = self._field_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} (field {name!r}, method {self.method!r})")
        ncomp = FIELDS[name][1]
        data = np.fromfile(path, dtype=self._dtype_of(name))
        n = self.grid_n
        if data.size != n ** 3 * ncomp:
            raise ValueError(f"{path}: {data.size} values != {n}^3 x {ncomp}")
        if factor is not None:
            data *= factor                    # in place: fromfile owns the buffer
        return data.reshape((n, n, n) if ncomp == 1 else (n, n, n, ncomp))

    @property
    def velocity_weighting(self) -> tuple[str, str]:
        """How this output's velocity grid was deposited, and how that is known: ('volume' | 'mass', why), from its
        run log (run_velocity_weighting). A coarser view of it (a block mean, a quiver arrow) must be the mean a
        coarser run of the SAME deposit gives."""
        return run_velocity_weighting(self.method, self.snapdir / f"{self.prefix}runlog")

    @property
    def lambda_th(self) -> float:
        """The web threshold this output was classified with (run_lambda_th of its run log '<prefix>runlog', else
        DEFAULT_LAMBDA_TH). mean_field() classifies its V-web with it."""
        return run_lambda_th(self.snapdir / f"{self.prefix}runlog")

    def mean_field(self, names, scaled: bool = True, lambda_th: float | None = None):
        """The MEAN-FIELD velocity derivatives computed from this output's 'velocity' grid, for either method
        (fields.mean_field_derivatives). names: one name of MEAN_FIELD_NAMES (returns its grid) or a sequence (returns
        {name: grid}). Units follow load(); classified with lambda_th. Raises MemoryError rather than swap."""
        from . import fields as _fields
        single = isinstance(names, str)
        wanted = (names,) if single else tuple(names)
        for nm in wanted:
            if nm not in MEAN_FIELD_NAMES:
                raise ValueError(f"no mean-field derivative {nm!r}; known: {', '.join(MEAN_FIELD_NAMES)}")
        vel = self.load("velocity", scaled=False, mode="memmap")          # raw u-units, never materialized whole
        peak = _fields.mean_field_peak_bytes(self.grid_n, wanted, vel.dtype.itemsize)
        if peak > _available_ram_bytes():
            raise MemoryError(f"the mean-field {', '.join(wanted)} of a {self.grid_n}^3 grid needs ~{peak / 1e9:.1f} GB "
                              f"(available {_available_ram_bytes() / 1e9:.1f} GB; DTFE_PY_RAM_GB overrides the detection)")
        a = 1.0 / (1.0 + self.meta.redshift)
        out = _fields.mean_field_derivatives(vel, self.meta.box_mpc, wanted, scale_factor=a,
                                             hubble_h=self.meta.hubble_h,
                                             lambda_th=self.lambda_th if lambda_th is None else float(lambda_th))
        del vel
        if scaled:
            for nm in out:
                factor = self._scale_factor32(nm)                         # vweb/vweb_eigenvalues: none (as stored)
                if factor is not None:
                    out[nm] *= factor
        return out[wanted[0]] if single else {nm: out[nm] for nm in wanted}

    def load_slice(self, name: str, axis: int = 2, index: int | None = None,
                   scaled: bool = True) -> np.ndarray:
        """One plane of a field, shape (N,N) or (N,N,ncomp), WITHOUT loading the cube: a read-only memmap, copying
        out only the plane. index=None takes the mid-plane. Same scaling as load(). axis=0 planes are contiguous
        on disk; axis 1/2 planes are strided (more read traffic, same bounded memory)."""
        if not 0 <= axis <= 2:
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        n = self.grid_n
        index = n // 2 if index is None else index
        if not 0 <= index < n:
            raise IndexError(f"slice index {index} outside [0, {n})")
        mm = self._open_memmap(name)
        sl = [slice(None)] * 3
        sl[axis] = index
        plane = np.array(mm[tuple(sl)])       # copy out the plane; the memmap is then dropped
        del mm
        if scaled:
            factor = self._scale_factor32(name)
            if factor is not None:
                plane *= factor
        return plane

    def iter_slabs(self, name: str, scaled: bool = True, max_bytes: int | None = None):
        """Stream a field as (start, stop, slab) tuples of axis-0 slabs, each a SCALED, writable ndarray: bounded
        memory at any grid size. A slab stays within max_bytes (default 1/8 of the load budget, at least one
        plane); concatenating every slab reproduces load(name, mode='ram')."""
        mm = self._open_memmap(name)
        n = self.grid_n
        row_bytes = int(np.prod(mm.shape[1:], dtype=np.int64)) * mm.dtype.itemsize
        if max_bytes is None:
            max_bytes = max(_load_budget_bytes() // 8, row_bytes)
        thick = max(1, min(n, int(max_bytes // row_bytes)))
        factor = self._scale_factor32(name) if scaled else None
        for start in range(0, n, thick):
            stop = min(start + thick, n)
            slab = np.array(mm[start:stop])   # copy: writable, detached from the mapping
            if factor is not None:
                slab *= factor
            yield start, stop, slab

    def field_stats(self, name: str, scaled: bool = True) -> dict:
        """Streaming min/max/mean/std of the full (scaled) field at slab-bounded memory:
        the run-summary numbers for any grid size. float64 accumulation."""
        cnt = 0
        s = s2 = 0.0
        mn, mx = np.inf, -np.inf
        for _, _, slab in self.iter_slabs(name, scaled=scaled):
            d = slab.astype(np.float64, copy=False)
            cnt += d.size
            s += float(d.sum())
            s2 += float(np.square(d).sum())
            mn = min(mn, float(d.min()))
            mx = max(mx, float(d.max()))
        mean = s / cnt
        var = max(s2 / cnt - mean * mean, 0.0)
        return {"n": cnt, "min": mn, "max": mx, "mean": mean, "std": var ** 0.5}

    def density(self, units: str = "mean") -> np.ndarray:
        """Density with METHOD-INDEPENDENT units: units='mean' -> rho/rho_bar, 'physical' -> 1e10 Msun/Mpc^3
        (comoving, h-free). The on-disk convention is auto-detected from the grid's own mean; a grid matching
        neither (an outdated standard-DTFE output) triggers a loud warning."""
        raw = self.load("density")
        conv = self._density_convention(raw)
        if units == "mean":
            return raw if conv == "mean" else raw / self.meta.rho_bar
        if units == "physical":
            return raw * self.meta.rho_bar if conv == "mean" else raw
        raise ValueError("units must be 'mean' or 'physical'")

    def _density_convention(self, raw: np.ndarray) -> str:
        """'mean' (grid mean ~1) or 'physical' (grid mean ~rho_bar), detected by sampling."""
        m = float(np.mean(raw.flat[::997], dtype=np.float64))
        if 0.2 < m < 5.0:
            return "mean"
        ratio = m / self.meta.rho_bar
        if 0.2 < ratio < 5.0:
            return "physical"
        # neither convention: stale pre-overhaul grid; pick the closer one in log space but say so
        guess = "mean" if abs(np.log10(max(m, 1e-30))) < abs(np.log10(max(ratio, 1e-30))) else "physical"
        print(f"[dtfelib] WARNING: {self.snapdir.name}: density grid mean {m:.3e} matches neither "
              f"rho/rho_bar (~1) nor the physical mean ({self.meta.rho_bar:.3e}); this looks like a "
              f"stale output of an older binary -- absolute values are unreliable, regenerate with the "
              f"current binary. Proceeding as units='{guess}'.")
        return guess

    def velocity_single_stream(self) -> np.ndarray:
        """Velocity (N,N,N,3) km/s with NaN wherever the stream multiplicity differs from 1. PS-DTFE only (ValueError
        under method='dtfe'). The mask is a TOLERANCE (|streams - 1| > 1e-3), never an exact float equality; cells with
        bit 1 of 'hidden_streams' set are masked as well. averaged=True uses 'a_streams', False the raw '.streams'."""
        if self.method != "ps":
            raise ValueError("velocity_single_stream() needs PS-DTFE outputs (method='ps'); "
                             "the stream-count field is only produced by PS-DTFE")
        # mode='ram': the NaN masking below WRITES into the array, so it must never be the
        # read-only memmap that mode='auto' returns for over-budget grids. The mask grids are
        # only compared against, so auto (possibly memmap) is fine there.
        vel = self.load("velocity", mode="ram")
        multi = ~single_stream_mask(self.load("streams"),            # THE rule, shared with webstreams and profiles
                                    self.load("hidden_streams") if self.has("hidden_streams") else None)
        vel[multi] = np.nan
        return vel

    def fold_mask(self) -> np.ndarray:
        """bool (N,N,N): the fold-caustic cells of this PS output (see caustic_is_fold: streams > 1 + tol, or
        '.hidden_streams' bit 1, OR the class mask's fold bits). Use this, never the raw 'caustic' grid, wherever a
        fold fraction is measured. Needs no '--ps-caustics' grid; grids are reduced in slabs from memmaps."""
        if self.method != "ps":
            raise ValueError("fold_mask() needs PS-DTFE outputs (method='ps'): standard DTFE has no streams")
        mask = self.load("caustic_class", mode="memmap") if self.has("caustic_class") else None
        if not self.has("streams"):
            if mask is not None:
                return caustic_is_fold(mask)
            if self.has("caustic"):
                return np.asarray(self.load("caustic", mode="memmap")) > 0.5
            raise FileNotFoundError(f"no streams, caustic_class or caustic grid in {self.snapdir} (prefix {self.prefix})")
        hidden = self.load("hidden_streams", mode="memmap") if self.has("hidden_streams") else None
        return caustic_is_fold(mask, streams=self.load("streams", mode="memmap"), hidden=hidden)

    # ------------------------------------------------------------------ helpers
    def slice(self, field: np.ndarray, axis: int = 2, index: int | None = None) -> np.ndarray:
        """Mid-plane (or given-index) slice; works for scalar and multi-component fields."""
        index = field.shape[axis] // 2 if index is None else index
        sl = [slice(None)] * 3
        sl[axis] = index
        return field[tuple(sl)]

    @property
    def cell_mpc(self) -> float:
        return self.meta.box_mpc / self.grid_n

    def __repr__(self):
        m = self.meta
        pfx = "" if self.prefix == _PREFIX[self.method] else f", prefix={self.prefix!r}"
        return (f"FieldSet({self.snapdir.name}, method={self.method!r}{pfx}, "
                f"grid={self.grid_n}^3, box={m.box_mpc:.2f} Mpc, z={m.redshift:.2f}, "
                f"rho_bar={m.rho_bar:.3f}e10 Msun/Mpc^3)")


def plane_geometry_matches(side: dict, center=None, planes: int = 1, thickness: float = 2.0,
                           supersample: int = 1, window=None) -> bool:
    """Does an image-plane sidecar (tools/make_image_plane.py's .json) describe this geometry? run_ps_pipeline.sh
    regenerates a plane whose sidecar differs, and the launcher predicts that. The centre counts only when one is
    given; a sidecar without 'supersample' means K = 1; 'window' is (u0, u1, v0, v1) in Mpc."""
    try:
        box = float(side["box"])
        if center is not None and abs(float(side["center"]) - float(center)) > 1e-9:
            return False
        if int(side.get("planes", 1)) != int(planes) or int(side.get("supersample", 1)) != int(supersample):
            return False
        if int(planes) > 1 and abs(float(side.get("thickness", 0.0)) - float(thickness)) > 1e-9:
            return False
        defaults = (0.0, box, 0.0, box)
        want = [defaults[i] if window is None or window[i] is None else float(window[i]) for i in range(4)]
        have = [float(side.get(k, d)) for k, d in zip(("u0", "u1", "v0", "v1"), defaults)]
        return all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(have, want))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


class PointPlane:
    """Point-evaluated image plane: the '<prefix>.pts_*' outputs of a '--sample-points' run on a plane from
    tools/make_image_plane.py, plus its JSON sidecar. field(name) returns the CENTRAL sampled plane (nv, nu[, c]).
    Units mirror FieldSet.load; derived names (velDiv/velShear/velVort) need '.pts_velGrad'; see available()."""

    #: field -> (file extension, components, u-unit scale exponent as in _VELOCITY_SCALE_EXP)
    _PTS_FIELDS = {
        "density":    (".pts_den",     1, None),
        "streams":    (".pts_streams", 1, None),
        "velocity":   (".pts_vel",     3, 0.5),
        "dispersion": (".pts_velDisp", 6, 1.0),
        "velGrad":    (".pts_velGrad", 9, 0.5),
        "denGrad":    (".pts_denGrad", 3, None),
        # --ps-caustics: per-point caustic mask, same bit layout as the grid 'caustic_class'
        # (both parity bits set = the point sits on a fold). Decode with caustic_collapse_max().
        "caustic":    (".pts_caustic", 1, None),
    }
    _DERIVED = ("speed", "dispTrace", "dispMag", "velDiv", "velShear", "velVort", "denGradMag")

    def __init__(self, prefix, plane_json, redshift: float | None = None,
                 project: str = "plane"):
        import json
        self.prefix = Path(prefix)
        self.side = json.loads(Path(plane_json).read_text())
        self.nu, self.nv = int(self.side["nu"]), int(self.side["nv"])
        self.planes = int(self.side["planes"])
        # KxK sub-samples per output pixel (make_image_plane --supersample): averaging the
        # block turns the POINT sample into an estimate of the pixel-AREA average, the
        # quantity the deposit grid reports. 1 = plain point sampling (the default).
        self.supersample = int(self.side.get("supersample", 1))
        # 'plane' = the central sampling plane (a zero-thickness cross-section);
        # 'slab' = mean over all planes, i.e. the slab-VOLUME average when combined with
        # enough planes (see make_image_plane --planes; few planes ghost).
        if project not in ("plane", "slab"):
            raise ValueError("project must be 'plane' or 'slab'")
        self.project = project
        # The plane file is generated once and reused for every snapshot, so its sidecar redshift belongs to
        # whichever snapshot defined it. Order: explicit argument > this snapshot's header > sidecar.
        if redshift is not None:
            self.redshift = float(redshift)
        else:
            self.redshift = self._redshift_from_snapdir()
            if self.redshift is None:
                self.redshift = float(self.side.get("redshift", 0.0))
                print(f"[dtfelib] PointPlane: no combined_*.hdf5 next to {self.prefix.name}; "
                      f"falling back to the plane sidecar's z={self.redshift:.3f} for the "
                      f"velocity scaling -- pass redshift= if that is not this snapshot's z")
        n_disk = Path(f"{self.prefix}.pts_den").stat().st_size // 8
        k = self.supersample
        if n_disk != self.planes * self.nv * k * self.nu * k:
            raise ValueError(
                f"{self.prefix}.pts_den holds {n_disk} points but the sidecar describes "
                f"{self.planes}x{self.nv * k}x{self.nu * k}; wrong plane file for this run?")

    def _redshift_from_snapdir(self) -> float | None:
        """This snapshot's redshift from its own combined_*.hdf5 (the prefix's directory)."""
        try:
            import h5py
        except ImportError:
            return None
        snaps = sorted(self.prefix.parent.glob("combined_*.hdf5"))
        if not snaps:
            return None
        try:
            with h5py.File(snaps[0], "r") as f:
                return float(f["Header"].attrs["Redshift"])
        except (OSError, KeyError):
            return None

    def _scale(self, exp: float | None) -> float:
        return 1.0 if exp is None else float((1.0 / (1.0 + self.redshift)) ** exp)

    def _load(self, name: str) -> np.ndarray:
        ext, comps, exp = self._PTS_FIELDS[name]
        dtype = np.int32 if name in ("streams", "caustic") else np.float64
        k = self.supersample
        raw = np.fromfile(f"{self.prefix}{ext}", dtype=dtype)
        shape = (self.planes, self.nv * k, self.nu * k) + ((comps,) if comps > 1 else ())
        arr = raw.reshape(shape)
        if name == "caustic":
            # A BITMASK must never be averaged: both reductions are a bitwise OR, as the grid merges the same bits
            # across partitions, so a pixel reports "covered by a tetrahedron with these properties".
            arr = np.bitwise_or.reduce(arr, axis=0) if self.project == "slab" \
                  else arr[self.planes // 2]
            if k > 1:
                arr = np.bitwise_or.reduce(arr.reshape(self.nv, k, self.nu, k), axis=(1, 3))
            return arr
        arr = arr.mean(axis=0) if self.project == "slab" else arr[self.planes // 2]
        if k > 1:                                   # average each KxK sub-sample block
            tail = (comps,) if comps > 1 else ()
            arr = arr.reshape((self.nv, k, self.nu, k) + tail).mean(axis=(1, 3))
        s = self._scale(exp)
        return arr if s == 1.0 else arr * s

    def available(self):
        base = [n for n, (ext, _, _) in self._PTS_FIELDS.items()
                if Path(f"{self.prefix}{ext}").is_file()]
        out = list(base)
        if "velocity" in base:
            out.append("speed")
        if "dispersion" in base:
            out += ["dispTrace", "dispMag"]
        if "velGrad" in base:
            out += ["velDiv", "velShear", "velVort"]
        if "denGrad" in base:
            out.append("denGradMag")
        return out

    def field(self, name: str) -> np.ndarray:
        if name in self._PTS_FIELDS:
            return self._load(name)
        if name == "speed":
            v = self._load("velocity")
            return np.sqrt((v ** 2).sum(axis=-1))
        if name == "dispTrace":
            d = self._load("dispersion")                     # xx xy xz yy yz zz
            tr = d[..., 0] + d[..., 3] + d[..., 5]
            return np.sqrt(np.maximum(tr, 0.0))              # 3D velocity dispersion sigma
        if name == "dispMag":
            d = self._load("dispersion")
            xx, xy, xz, yy, yz, zz = (d[..., i] for i in range(6))
            return np.sqrt(xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2))
        if name in ("velDiv", "velShear", "velVort"):
            # [d*3+j] = dv_j/dx_d (the binary's layout) -> g[d, j]
            g = self._load("velGrad").reshape(self.nv, self.nu, 3, 3)
            if name == "velDiv":
                return g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]
            if name == "velVort":
                wx = g[..., 1, 2] - g[..., 2, 1]             # eps_kij dv_j/dx_i
                wy = g[..., 2, 0] - g[..., 0, 2]
                wz = g[..., 0, 1] - g[..., 1, 0]
                return np.sqrt(wx**2 + wy**2 + wz**2)
            sym = 0.5 * (g + np.swapaxes(g, -1, -2))
            div3 = (g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]) / 3.0
            for i in range(3):
                sym[..., i, i] -= div3
            return np.sqrt((sym ** 2).sum(axis=(-1, -2)))
        if name == "denGradMag":
            return np.sqrt((self._load("denGrad") ** 2).sum(axis=-1))
        raise KeyError(f"unknown point field {name!r}; known: "
                       f"{', '.join(list(self._PTS_FIELDS) + list(self._DERIVED))}")

    @property
    def extent(self):
        """[u0, u1, v0, v1] in box units (h-free Mpc) for imshow."""
        return [self.side["u0"], self.side["u1"], self.side["v0"], self.side["v1"]]

    def __repr__(self):
        extra = f", {self.supersample}x{self.supersample} sub-samples" if self.supersample > 1 else ""
        return (f"PointPlane({self.prefix.name}, {self.nu}x{self.nv}x{self.planes}{extra}, "
                f"axis={self.side.get('axis')}, project={self.project}, "
                f"z={self.redshift:.2f}, fields: {', '.join(self.available())})")


def single_stream_mask(streams, hidden=None, tol: float = STREAM_TOL):
    """Boolean (N,N,N): the single-stream cells, |streams - 1| <= tol, and, when 'hidden_streams' is given,
    bit 1 clear. THE one rule: velocity_single_stream masks its complement, webstreams.stream_bins and
    profiles.sample it."""
    import numpy as np
    s = np.asarray(streams, dtype=np.float32)
    out = np.abs(s - 1.0) <= tol
    if hidden is not None:
        out &= (np.rint(np.asarray(hidden, dtype=np.float32)).astype(np.int32) & 1) == 0
    return out
