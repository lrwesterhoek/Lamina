"""Job specifications behind the DTFE / PS-DTFE GUI (app.py).

Three jobs, one per GUI tab, each holding what the terminal workflow sets for it plus the checks
for combinations the scripts reject, and the EXACT commands it stands for (``steps()``):

  RunSpec       one scripts/run_ps_dtfe.sh or run_dtfe.sh call: grids (+ a hi-res slice)
  PipelineSpec  scripts/run_ps_pipeline.sh (grids + image plane for every stale snapshot of
                the chosen simulations), optionally followed by plot_pointeval.py per simulation
  PlotSpec      the figure scripts in python/plot (and analyze.py), one step per script call,
                described declaratively by FIGURE_SETS
  DataSpec      scripts/download_snapshots.sh (snapshots, group catalogues, trees, ICs) and
                python/tools/merge_HDF5.py (the h-free combined files the other jobs read)
  CustomSpec    the binary itself on ANY snapshot file (Gadget HDF5/binary, text): the "your
                file" tab, for data that is not IllustrisTNG -- units, box, periodicity and the
                initial positions spelled out instead of assumed

QueuedJob freezes any of them for the GUI's job queue (kind + settings; commands at start).

The GUI never runs anything the terminal could not: it shows those commands and executes them.

Pure Python (no Qt), so it is testable headless: tests/py_gui_runspec_test.py.
"""
# PipelineSpec, PlotSpec, DataSpec and CustomSpec, with the helpers only they use, live in runspec_pipeline.py,
# runspec_plot.py, runspec_data.py and runspec_custom.py; imported back below (before JOB_KINDS), so every
# runspec.<name> still works.

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from dtfelib.cli import FIGURES_ROOT, find_sims  # noqa: E402  (after the path fix-up)
from dtfelib.cli import sim_dir as _cli_sim_dir  # noqa: E402
import config  # noqa: E402  (python/config.py: only dtfelib.cli at import; its h5py is lazy)

PS_SCRIPT = REPO_ROOT / "scripts" / "run_ps_dtfe.sh"
DTFE_SCRIPT = REPO_ROOT / "scripts" / "run_dtfe.sh"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_ps_pipeline.sh"
PLOT_DIR = REPO_ROOT / "python" / "plot"
ANALYZE = REPO_ROOT / "python" / "analyze.py"
# FIGURES_ROOT (dtfelib.cli, = config.LOCAL_FIGURES_ROOT): DTFE_FIGURES_ROOT, else python/figures
LEGACY_FIGURES_ROOT = REPO_ROOT / "python" / "figures"     # an earlier default, saved verbatim in specs


def _current_figures_root(d: dict) -> dict:
    """A saved spec's figures_root with the OLD default (python/figures, stored as a plain path because the
    field saved the default it showed) read as the current default: settings and queued jobs saved with it
    would otherwise keep writing into the repo."""
    root = d.get("figures_root")
    if root and Path(root) == LEGACY_FIGURES_ROOT:
        d = dict(d, figures_root=str(FIGURES_ROOT))
    return d
# The data root is the launcher's own field (its setup asks for it; every job gets it as DTFE_DATA_ROOT). A new job
# starts at DTFE_DATA_ROOT when the launcher was started with one, else at none: an empty root is no folder at all
# (never the working directory), so nothing is found under it and every job refuses to run until one is chosen.
_NO_ROOT = Path("/nonexistent-data-root")       # what sim_dir() puts an empty root's simulations under


def default_data_root() -> str:
    """A new job's data root: $DTFE_DATA_ROOT, else '' (none chosen yet). Read at call time."""
    return os.environ.get("DTFE_DATA_ROOT", "")


def root_ok(data_root) -> bool:
    """Whether 'data_root' names an existing folder ('' names none: Path('') would be the working directory)."""
    return bool(str(data_root).strip()) and Path(data_root).expanduser().is_dir()


def sim_dir(sim: str, data_root) -> Path:
    """dtfelib.cli.sim_dir under the launcher's data root 'data_root' only (never $DTFE_DATA_ROOT behind the
    field's back): an empty one puts the simulation under a folder that does not exist."""
    return _cli_sim_dir(sim, Path(data_root).expanduser() if str(data_root).strip() else _NO_ROOT)


# the interpreter for the figure scripts (numpy/scipy/matplotlib/h5py): $PYTHON as in the
# pipeline script ($PY, its older name, for one release), else the python3 on PATH -- NOT the GUI's own venv interpreter
PYTHON = os.environ.get("PYTHON") or os.environ.get("PY") or shutil.which("python3") or sys.executable

# (name passed to --field, label). run_ps_dtfe.sh takes any subset via FIELDS; run_dtfe.sh the subset it computes
# (DTFE_ALLOWED_FIELDS: no dispersion). Both drivers default to the same analysis set, DEFAULT_FIELDS.
PS_FIELDS = [
    ("density_a", "density"),
    ("velocity_a", "velocity"),
    ("gradient_a", "velocity gradient"),
    ("divergence_a", "divergence"),
    ("shear_a", "shear"),
    ("vorticity_a", "vorticity"),
    ("dispersion_a", "velocity dispersion"),
    ("tweb_a", "T-web"),
    ("vweb_a", "V-web"),
]
# both drivers' FIELDS default: density, velocity, its divergence and shear (the binaries derive these two from the
# per-tet velocity gradient without writing it). A Grids or Pipeline job passes FIELDS only when its ticks differ,
# so this must stay the scripts' default (tests/py_gui_runspec_test.py checks it against both).
DEFAULT_FIELDS = ["density_a", "velocity_a", "divergence_a", "shear_a"]
PS_DEFAULT_FIELDS = list(DEFAULT_FIELDS)
WEB_FIELDS = ("tweb_a", "vweb_a")        # the cosmic-web classes: the only consumers of lambda_th
LAMBDA_TH_DEFAULT = 0.3                  # run_ps_dtfe.sh / run_dtfe.sh LAMBDA_TH (the binary alone: 0.0, so a Custom
                                         # run without it would classify differently from a scripted one)
# run_dtfe.sh's FIELDS default: the same set (it has no PS-only field, so dtfe_fields() keeps it whole)
DTFE_DEFAULT_FIELDS = list(DEFAULT_FIELDS)
# what the standard binary computes (no dispersion: PS only)
DTFE_ALLOWED_FIELDS = ["density_a", "velocity_a", "gradient_a", "divergence_a", "shear_a", "vorticity_a"] + list(WEB_FIELDS)


def dtfe_fields(fields) -> list[str]:
    """The ticked fields a standard-DTFE run computes, in the list's order (the velocity dispersion is a
    phase-space field: left out)."""
    return [f for f in DTFE_ALLOWED_FIELDS if f in fields]

# full-resolution grid bytes per cell, roughly (src/auto_tune.cc has the exact model): each field's own grid, and
# the 9-component velocity gradient ONCE for any field derived from it (the binaries deposit it for all of them).
# Not all at once: the deposited grids stay from the deposit to the output, but the derived fields and the web
# classes (_POST_FIELDS; the T-web with its tidal stage) are made only after the deposit (auto_tune.cc
# autoTuneBytesPerCell's two phases, autoTuneDerivedBytes)
_FIELD_BYTES = {"density_a": 4, "velocity_a": 12, "gradient_a": 0, "divergence_a": 4,
                "shear_a": 20, "vorticity_a": 12, "dispersion_a": 24, "tweb_a": 72, "vweb_a": 16}
_POST_FIELDS = ("divergence_a", "shear_a", "vorticity_a", "tweb_a", "vweb_a")
_GRADIENT_BYTES = 36
_FROM_GRADIENT = ("gradient_a", "divergence_a", "shear_a", "vorticity_a", "vweb_a")
# ... and a PS-DTFE run that takes the face gradient (face_route_on) keeps more full-resolution grids, per cell
# (src/quantities.h, one Real each): the face sums from the deposit to the normalization, 30 numbers
# (face_velocity: 2 sides x 3 axes x 3 components, face_coverage 6, face_momentum 6); the two grids every face pass
# makes (Quantities::psFaceGradientBegin), the stream average's divergence and the mass flux divergence, written as
# '.a_velDivStreams' and '.a_massFluxDiv' (src/input_output.cc); the stream average's twin of each of 'gradient_a',
# 'shear_a', 'vorticity_a' asked (src/DTFE.cpp psStreamsTwins; '.a_velGradStreams', '.a_velShearStreams',
# '.a_velVortStreams'); and the mean velocity, deposited for the face gradient whether or not 'velocity_a' is written
# (src/DTFE.cpp). The face sums (and that mean velocity) are freed once the gradient is made from them, before the
# derived fields: the planner books the larger of the normalization (the deposit's grids, the twins, the two outputs)
# and the derived phase (auto_tune.cc: max(bCell, bCellDeposit + normR)). Standard DTFE's face sums are per
# sub-domain and it writes no twins: nothing is added for it
_FACE_SUM_BYTES = 30 * 4
_FACE_OUTPUT_BYTES = 2 * 4
_FACE_TWIN_BYTES = {"gradient_a": 36, "shear_a": 20, "vorticity_a": 12}


# ---------------------------------------------------------------------- the two quality modes
# One choice per estimator and job: Exact (first, the default) or Fast. quality_mode() is the ONE place that says what
# a choice implies; each job's mode() asks it and puts the job's Advanced overrides on top (apply_advanced), and every
# command and environment builder, the memory estimate and the checks read only that (and the four accessors of the
# overrides the mode does not hold). The launcher pins what it passed for these settings before the two modes, so a
# setting the binaries come to decide themselves is a deletion here.
QUALITIES = ("exact", "fast")
QUALITY_DEFAULT = "exact"
QUALITY_NAME = {"exact": "Exact", "fast": "Fast"}
# what each choice is, per estimator, in the launcher's words (the Quality pair's grey line: one line on the left
# pane at its minimum width, so no word wraps alone)
QUALITY_TEXT = {
    "ps": {"exact": "Each tetrahedron's exact share of each cell; exact derivatives of the mean flow.",
           "fast": "One sample per cell: quick, with noisy per-cell values (smooth before use)."},
    "dtfe": {"exact": "The exact cell average.",
             "fast": "A Monte-Carlo mean of the cell; derivatives stay exact."},
}
# PS-DTFE Fast at an Advanced Samples count above one (set by hand, or kept by 'Load settings' of a run made at 3):
# the grey line names no count then (app_tabs._sync_quality)
QUALITY_TEXT_PS_FAST_SAMPLES = "Sampled at the Advanced count per cell axis: a smoother cell average, more time."
# each choice's tooltip on the run tabs (no timing number: the measurements are the plan's, docs/cleanup-plan.md)
QUALITY_TIP = {"exact": "Slower, exact: the default",
               "fast": "Quicker, less correct: for a first look"}


@dataclass(frozen=True)
class QualityMode:
    """What a job's quality implies (quality_mode): the values its former controls had by default for that cell
    average (but the sampled deposit's count: one point per cell axis, the program's default now), in the controls'
    own terms, which the builders turn into flags and driver knobs as before; with the job's Advanced overrides on
    top (apply_advanced), where it passes them."""
    deposit: str                    # the cell average: "exact" (-e, --exact) | "sampled" (-n nsub, --sampled)
    nsub: int                       # the sampled PS-DTFE deposit's points per cell axis (standard DTFE: the binary's)
    gpu: bool                       # the device: the drivers' automatic one (recommended_gpu) on a GPU build
    face_gradient: bool             # PS-DTFE: the face gradient wherever the job qualifies (face_gradient_env)
    volume_weighted: bool           # PS-DTFE's volume-weighted velocities (the production convention)
    vertex_mass: bool               # tetrahedron masses from the particles: a TNG job's (a Custom job has its own)
    linear_deposit: bool            # PS-DTFE's density-weighted deposit inside each tetrahedron
    parallel_triangulation: bool    # True: the TBB parallel insertion at any size, forced (two identical runs then
                                    # differ at float rounding); False: nothing forced, the binary's own rule
    ordered_merge: bool             # PS-DTFE's partitions merged in index order, given (PS_MERGE=index-order)
    pts_conventions: str            # the image plane's estimators: the grids' ("grid", the binary's default)
    gpu_share: str = "auto"         # PS-DTFE's '--gpu-share' (GPU_SHARE) of an exact deposit on the GPU in 3D: "auto"
                                    # (the machine use's share: only performance mixes), "1" (every partition on
                                    # the GPU: a Device GPU override) or "mixed" (both devices in every machine use:
                                    # Device CPU + GPU); set by apply_advanced only where it applies, else "auto"


def quality_mode(quality: str, estimator: str = "ps", fields=DEFAULT_FIELDS, n_particles=None,
                 precision: str = "single", dim: int = 3, run_mode: str = "balanced") -> QualityMode:
    """What 'quality' (QUALITIES) implies for a job of 'estimator' with these fields, particle count, precision,
    dimension and machine use. Exact: the exact cell average (PS-DTFE's exact deposit and its face gradient wherever
    the job qualifies; standard DTFE's exact cell average). Fast: the sampled one, at one point per cell axis for
    PS-DTFE (the program's default: quick, with noisy per-cell values; standard DTFE: the binary's own count). Both:
    the device the drivers would pick (the GPU on a GPU build, by recommended_gpu: standard DTFE's sampled average on
    the CPU in performance, PS-DTFE's sampled deposit on the CPU below GPU_PAYS_FROM particles) with the machine use's
    GPU share ('auto'), the volume weighting, the vertex masses of a TNG job, no forced parallel insertion (the
    binary's own rule), the index order merge and the grids' plane conventions. Nothing at all is passed for the
    partition, the concurrency and the memory budget (the binary's auto-tuner) nor for a TNG job's precision
    (single). A job's Advanced overrides come on top, in each job's mode() and the accessors (apply_advanced). An
    unknown quality reads as Exact (the checks call it an error). n_particles may be a function: it is called only
    where the device depends on it (the sampled PS-DTFE deposit), so a Custom job reads its file's header only
    then."""
    deposit = "sampled" if quality == "fast" else "exact"
    if callable(n_particles):
        n_particles = n_particles() if estimator == "ps" and deposit == "sampled" else None
    gpu = bool(gpu_build_default(estimator, precision, dim)
               and recommended_gpu(deposit, fields, n_particles, estimator, precision, dim, run_mode=run_mode))
    return QualityMode(deposit=deposit, nsub=1, gpu=gpu, face_gradient=True, volume_weighted=True, vertex_mass=True,
                       linear_deposit=False, parallel_triangulation=False, ordered_merge=True, pts_conventions="grid")


# ---------------------------------------------------------------------- the opt-in Advanced settings
# The launcher's Advanced switch (off by default) brings back some controls the two modes removed, as overrides of a
# job's mode: each run job's 'advanced' dict, keyed by the removed field's name, with the meaning and range that field
# had before the two modes; a key that is absent is Automatic (the mode's own value, nothing new passed). The rules
# below are the ONE place that says which override a job passes and why another is not passed (_refusal); each job's
# mode() applies them (apply_advanced), and every command and environment builder, the memory estimate, the checks
# and the 'Runs on' line read that and the four accessors (job_partition, job_concurrency, job_memory_budget,
# job_precision), so "the binary decides this itself" stays a deletion here. Not brought back on purpose: the plane
# conventions, the volume weighting, the linear deposit, the stream density, the vertex masses of a TNG job.
ADVANCED_KEYS = ("gpu", "nsub", "face_gradient", "partition", "max_concurrent", "mem_budget_gb",
                 "parallel_triangulation", "ordered_merge", "precision")
# the keys QualityMode holds (apply_advanced puts a passed one there); the other four have the accessors
_MODE_KEYS = ("gpu", "nsub", "face_gradient", "parallel_triangulation", "ordered_merge")
# the number rows' ranges, the old controls' (the spin boxes before the two modes; their 0 was 'auto', Automatic now)
ADVANCED_RANGE = {"nsub": (1, 6), "partition": (1, 8), "max_concurrent": (1, 16), "mem_budget_gb": (1, 4096)}
AUTOMATIC = "Automatic"
# each row's title in the Advanced group
ADVANCED_ROW = {"gpu": "Device", "nsub": "Samples per cell axis", "face_gradient": "Velocity derivatives",
                "partition": "Partitions per axis", "max_concurrent": "Concurrent triangulations",
                "mem_budget_gb": "Memory budget (GB)", "parallel_triangulation": "Parallel triangulation",
                "ordered_merge": "Merge order", "precision": "Precision"}
# ... and its name in the one-line list of a job's overrides ('Advanced: Device: CPU; Partitions: 4')
ADVANCED_NAME = {"gpu": "Device", "nsub": "Samples", "face_gradient": "Velocity derivatives",
                 "partition": "Partitions", "max_concurrent": "Concurrent triangulations",
                 "mem_budget_gb": "Memory budget", "parallel_triangulation": "Parallel triangulation",
                 "ordered_merge": "Merge order", "precision": "Precision"}
# the Device row's three values: False the CPU, True the GPU (every partition: '--gpu-share 1' where a share applies),
# DEVICE_MIXED both together (PS-DTFE's exact deposit of a run split into partitions, shared at the program's own
# share in every machine use: '--gpu-share mixed'); a job saved with True or False keeps its meaning
DEVICE_MIXED = "mixed"


def _advanced_value(key, v):
    """A saved override's value in its row's range, normalised (ints, a float budget), or None when it is not one.
    A value that equals Automatic's (the merge in index order, no parallel insertion, single precision) is not an
    override: Automatic has one spelling, the absent key."""
    if key == "gpu":
        return v if isinstance(v, bool) or v == DEVICE_MIXED else None
    if key == "face_gradient":
        return v if isinstance(v, bool) else None
    if key == "parallel_triangulation":
        return True if v is True else None
    if key == "ordered_merge":
        return False if v is False else None
    if key == "precision":
        return "double" if v == "double" else None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    lo, hi = ADVANCED_RANGE[key]
    # the range first: it also drops NaN and the infinities, and an integer beyond the float range (int(v) of a huge
    # float, or a float test of a huge int, would raise)
    if not lo <= v <= hi or v != int(v):
        return None
    return float(v) if key == "mem_budget_gb" else int(v)


def advanced_keys(cls) -> tuple[str, ...]:
    """The override keys a job class takes: all of them, but a Custom job keeps its own Precision row (its
    'precision' field picks the program, as the dimension does)."""
    own = getattr(cls, "__dataclass_fields__", {})
    return tuple(k for k in ADVANCED_KEYS if k not in own)


def clean_advanced(raw, cls=None) -> dict:
    """A saved 'advanced' dict as a job of class 'cls' keeps it: known keys only, each value in its row's range
    (_advanced_value), in ADVANCED_KEYS order; anything else is dropped without an error."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k in (advanced_keys(cls) if cls is not None else ADVANCED_KEYS):
        v = _advanced_value(k, raw[k]) if k in raw else None
        if v is not None:
            out[k] = v
    return out


def job_advanced(spec) -> dict:
    """A job's overrides (Grids, Pipeline and Custom jobs, duck-typed), cleaned: what every reader starts from."""
    return clean_advanced(getattr(spec, "advanced", None), type(spec))


def _with_advanced(spec, d: dict) -> dict:
    """A job's to_dict: asdict's with the nested 'advanced' only when the job has an override, so a job without
    one is saved byte for byte as before the Advanced settings (its last key: the field is the last one)."""
    d.pop("advanced", None)
    adv = job_advanced(spec)
    if adv:
        d["advanced"] = adv
    return d


def _refusal(spec, key: str, value, deposit: str) -> str:
    """Why this job does not pass the override key=value ('' when it does): the controls' rules before the two
    modes (where each was enabled, what the run could use), so an override never makes a command the launcher
    could not make then -- but for the Device's CPU + GPU, which asks the program for '--gpu-share mixed' (only
    where its share applies: PS-DTFE's exact deposit on a GPU build, in 3D, single precision). deposit: the cell
    average of the job's quality (its mode's), which only the samples, the face integrals and the shared device
    depend on."""
    est, dim = getattr(spec, "estimator", "ps"), getattr(spec, "dim", 3)
    ps = est == "ps"
    if key == "gpu":
        if extra_device_flags(spec):
            return "the extra options name the device"
        if value is False:
            return ""
        if dim == 2:
            return "the 2D programs run on the CPU"
        if not gpu_built(est, job_precision(spec), dim):
            return f"this build has no GPU ({GPU_BUILD_HINT})"
        if value == DEVICE_MIXED:           # (offered only where all of these hold: advanced_choices)
            if not ps:
                return "standard DTFE runs on one device"
            if job_precision(spec) == "double":
                return "the double-precision deposit is not shared"
            if deposit != "exact":
                return "only the Exact deposit can be shared"
        return ""
    if key == "nsub":
        if not ps:
            return "standard DTFE samples at the program's own count"
        return "Exact uses no samples" if deposit == "exact" else ""
    if key == "face_gradient":
        if not ps:
            return "a PS-DTFE setting"
        if dim == 2:
            return "the 2D programs have no face integrals"
        if getattr(spec, "_extra_names_route", None) and spec._extra_names_route():
            return "the extra options name the route"
        if value and deposit != "exact":
            return "the face integrals need Exact"
        if value and getattr(spec, "halo_release", 0.0) > 0:
            return "the halo release keeps the stream average"
        return ""
    if key == "parallel_triangulation":
        if not ps:
            return "a PS-DTFE setting"
        if dim == 2:
            return "3D only"
        return "" if tbb_built(job_precision(spec), dim) else "this build has no TBB"
    if key == "ordered_merge":
        if not ps:
            return "a PS-DTFE setting"
        if getattr(spec, "_extra_names_merge", None) and spec._extra_names_merge():
            return "the extra options name the merge"
        return ""
    if key == "precision":                                      # (a Custom job has no such key: clean_advanced)
        return "" if binary_built(est, "double", dim) else \
            f"{binary_name(est, 'double', dim)} is not built (Setup builds the pair)"
    return ""                                                   # partitions, concurrency, budget: always passed


def extra_device_flags(spec) -> frozenset:
    """The device options a job's extra options name, canonical ('--gpu', '--gpu-share'; a Custom job's,
    CustomSpec._extra_device_flags; empty for none, and for the other jobs). Where they name any, the tab adds no
    device flag of its own and passes no Device override."""
    flags = getattr(spec, "_extra_device_flags", None)
    return flags() if flags else frozenset()


def apply_advanced(mode: QualityMode, spec) -> QualityMode:
    """The job's mode (quality_mode's answer for its quality) with the overrides it passes (_refusal) of the five
    settings QualityMode holds; the same object when there are none. The Device's value becomes the mode's device and,
    where a GPU share applies (PS-DTFE's exact deposit on the GPU in 3D), its share: '1' for the GPU (every
    partition on it, in every machine use), 'mixed' for CPU + GPU; elsewhere the share stays 'auto', which the
    drivers do not pass."""
    passed = {k: v for k, v in job_advanced(spec).items()
              if k in _MODE_KEYS and not _refusal(spec, k, v, mode.deposit)}
    if "gpu" in passed:
        device = passed["gpu"]
        passed["gpu"] = device is not False
        if (device is not False and mode.deposit == "exact" and getattr(spec, "estimator", "ps") == "ps"
                and getattr(spec, "dim", 3) != 2):
            passed["gpu_share"] = DEVICE_MIXED if device == DEVICE_MIXED else "1"
    return replace(mode, **passed) if passed else mode


def job_mixes(spec, m: QualityMode | None = None) -> bool:
    """Does the job deposit on the CPU and the GPU together? PS-DTFE's exact deposit on the GPU in 3D, single
    precision (the double build keeps one device), whose share mixes: 'mixed' (Device CPU + GPU) in every machine use,
    'auto' in performance (the machine use's own share). The program mixes only a run split into partitions, and
    keeps a small one on the GPU (runs_on_tip). m: the job's mode, when the caller has it."""
    m = m or spec.mode()
    if not (getattr(spec, "estimator", "ps") == "ps" and getattr(spec, "dim", 3) != 2 and m.gpu
            and m.deposit == "exact" and job_precision(spec) == "single" and not extra_device_flags(spec)):
        return False
    return m.gpu_share == DEVICE_MIXED or (m.gpu_share == "auto" and spec.run_mode == "performance")


def _passed(spec, key: str, automatic):
    """The value of one of the four overrides QualityMode does not hold, where the job passes it (_refusal), else
    'automatic' (none of the four depends on the cell average)."""
    v = job_advanced(spec).get(key)
    return automatic if v is None or _refusal(spec, key, v, "") else v


def job_precision(spec) -> str:
    """The precision a job runs at: a Custom job's own field; a Grids or Pipeline job's Advanced override 'double'
    where the double pair is built, else single (the two-mode TNG jobs' precision)."""
    own = getattr(spec, "precision", None)
    return own if own is not None else _passed(spec, "precision", "single")


def job_partition(spec) -> int:
    """The partitions per axis a job asks for (PARTITION, --partition); 0 = the binary's auto-tuner plans them."""
    return _passed(spec, "partition", 0)


def job_concurrency(spec) -> int:
    """The triangulations built at once (MAX_CONCURRENT, --max-concurrent); 0 = the auto-tuner's."""
    return _passed(spec, "max_concurrent", 0)


def job_memory_budget(spec) -> float:
    """The auto-tuner's memory budget in GB (DTFE_MEM_BUDGET_GB); 0 = the binary's own, the free memory."""
    return _passed(spec, "mem_budget_gb", 0.0)


def advanced_refused(spec) -> dict[str, str]:
    """{key: why} for each override of the job that is not passed (_refusal), in ADVANCED_KEYS order."""
    adv = job_advanced(spec)
    deposit = spec.mode().deposit if adv else ""
    return {k: why for k, v in adv.items() for why in [_refusal(spec, k, v, deposit)] if why}


def advanced_value_text(key: str, value) -> str:
    """An override's choice in its row's words ('CPU', 'CPU + GPU', 'Exact face integrals', '4'). The velocity
    derivatives' other route is PS-DTFE's stream average (the row is PS-DTFE's: a standard-DTFE job holds it only
    from a PS-DTFE job, and it is not passed there)."""
    if key == "gpu":
        return "CPU + GPU" if value == DEVICE_MIXED else "GPU" if value is True else "CPU"
    if key == "face_gradient":
        return "Exact face integrals" if value else "Stream average"
    if key == "parallel_triangulation":
        return "On"
    if key == "ordered_merge":
        return "Completion order"
    if key == "precision":
        return "Double"
    return f"{value:g}"


def advanced_choices(spec, key: str) -> list[tuple]:
    """A row's choices for this job, (value, text), Automatic (None) first: Device CPU, GPU only on a GPU build of
    the program the job runs, and CPU + GPU where a GPU share applies to it (PS-DTFE in 3D, single precision; under
    Fast it is offered and refused: only the exact deposit is shared); the velocity derivatives' two routes; the
    parallel insertion On on a build with TBB; the merge in completion order; Double. A number row has none here: its
    range is ADVANCED_RANGE."""
    est, dim, prec = getattr(spec, "estimator", "ps"), getattr(spec, "dim", 3), job_precision(spec)
    gpu = gpu_built(est, prec, dim)
    values = {"gpu": (False,) + ((True,) if gpu else ())
              + ((DEVICE_MIXED,) if gpu and est == "ps" and dim != 2 and prec == "single" else ()),
              "face_gradient": (True, False), "parallel_triangulation": (True,) if tbb_built(prec, dim) else (),
              "ordered_merge": (False,), "precision": ("double",)}.get(key, ())
    return [(None, AUTOMATIC)] + [(v, advanced_value_text(key, v)) for v in values]


def advanced_rows(spec) -> tuple[str, ...]:
    """The rows the Advanced group shows for this job (its tab, estimator, dimension and build), where each control
    was before the two modes: the device in 3D; the samples, the merge order (2D too) for PS-DTFE; the velocity
    derivatives and the parallel insertion for PS-DTFE in 3D; the partitions, concurrency and memory budget always;
    the precision on the Grids and Pipeline tabs when the double program is built (Custom has its own row)."""
    est, dim = getattr(spec, "estimator", "ps"), getattr(spec, "dim", 3)
    ps, three = est == "ps", dim != 2
    shown = {"gpu": three, "nsub": ps, "face_gradient": ps and three, "partition": True, "max_concurrent": True,
             "mem_budget_gb": True, "parallel_triangulation": ps and three, "ordered_merge": ps,
             "precision": getattr(spec, "precision", None) is None and binary_built(est, "double", dim)}
    return tuple(k for k in ADVANCED_KEYS if shown[k])


def _value_words(key: str, value) -> str:
    """An override's value inside a line of text: its choice's words ('CPU + GPU', '4', '40 GB', 'completion
    order')."""
    text = advanced_value_text(key, value)
    if key == "mem_budget_gb":
        return f"{text} GB"
    return text if key == "gpu" or key in ADVANCED_RANGE else text.lower()


def _line_text(key: str, value) -> str:
    """An override in a list of them, in one form for every row: 'Device: CPU', 'Memory budget: 40 GB', 'Merge
    order: completion order'."""
    return f"{ADVANCED_NAME[key]}: {_value_words(key, value)}"


def advanced_text(spec) -> str:
    """A job's overrides on one line, 'Device: CPU; Partitions: 4' ('' for none): the grey line on its tab, the
    folded group's summary, the checks list's info line and a queued job's tooltip."""
    return "; ".join(_line_text(k, v) for k, v in job_advanced(spec).items())


def advanced_count_text(n: int) -> str:
    """How many Advanced settings, in the status line's words: '1 setting in the Advanced group'."""
    return f"{n} setting{'s' if n != 1 else ''} in the Advanced group"


def automatic_text(spec, key: str) -> str:
    """What a row left on Automatic gives this job now (the grey text beside it): the job's mode without that
    override. The device: the CPU, the GPU, or CPU + GPU where the job mixes the two by itself (job_mixes: PS-DTFE's
    Exact in performance), in the Device row's own words. No measured number."""
    est, dim = getattr(spec, "estimator", "ps"), getattr(spec, "dim", 3)
    other = replace(spec, advanced={k: v for k, v in job_advanced(spec).items() if k != key})
    m = other.mode()
    if key == "gpu":
        if extra_device_flags(spec):
            return f"{AUTOMATIC}: the extra options name the device"
        return f"{AUTOMATIC}: " + ("CPU + GPU" if job_mixes(other, m) else f"the {'GPU' if m.gpu else 'CPU'}")
    if key == "nsub":
        if m.deposit == "exact":
            return "Exact uses no samples"
        return f"{AUTOMATIC}: {m.nsub}" if est == "ps" else f"{AUTOMATIC}: the program's own count"
    if key == "face_gradient":
        face = face_gradient_env(other) == "1"
        return f"{AUTOMATIC}: " + ("exact face integrals" if face else "the stream average")
    if key == "parallel_triangulation" and est == "ps" and dim != 2 and not tbb_built(job_precision(other), dim):
        return f"{AUTOMATIC}: the sequential insert (this build has no TBB)"
    return f"{AUTOMATIC}: " + {"partition": "the program plans them", "max_concurrent": "the program plans them",
                               "mem_budget_gb": "the free memory", "parallel_triangulation": "the program's own rule",
                               "ordered_merge": "index order", "precision": "single"}[key]


def keep_run_count(spec, rec):
    """'Load settings' of a sampled PS-DTFE run (rec: a results.RunRecord, duck-typed): its count (RunRecord.nsub,
    the log's) becomes the loaded job's Advanced Samples where the job's Fast takes another -- a run made before Fast
    became one point per cell axis, or at a count of its own -- so the loaded job is that run (the override turns the
    switch on). Nothing for a job that holds a count of its own, a count outside the row's range or any other run."""
    n, (lo, hi) = getattr(rec, "nsub", None), ADVANCED_RANGE["nsub"]
    if (getattr(rec, "estimator", "") != "ps" or rec.exact or not n or not lo <= n <= hi
            or getattr(spec, "estimator", "ps") != "ps" or "nsub" in job_advanced(spec)):
        return
    m = spec.mode()
    if m.deposit == "sampled" and m.nsub != n:
        spec.advanced = dict(spec.advanced, nsub=n)


def advanced_problems(spec) -> list[tuple[str, str]]:
    """The checks list's lines of a job's overrides: one info line naming them; a warning for each one the job
    cannot honour (not passed, so the job runs as on Automatic there, _refusal); and the launcher's warnings before
    the two modes for a passed parallel insertion and a memory budget above this machine's memory."""
    adv = job_advanced(spec)
    if not adv:
        return []
    out = [("info", f"Advanced: {advanced_text(spec)}")]
    refused = advanced_refused(spec)
    for k, why in refused.items():
        out.append(("warning", f"Advanced: {ADVANCED_NAME[k]} set to {_value_words(k, adv[k])} is not passed: {why}; "
                               "Automatic applies"))
    if "parallel_triangulation" in adv and "parallel_triangulation" not in refused:
        out.append(("warning", "parallel triangulation: faster on small sets, but two identical runs then differ at "
                               "float rounding (the stream counts do not)"))
    budget, ram = job_memory_budget(spec), machine_ram_gb()
    if budget and ram and budget > ram:
        out.append(("warning", f"memory budget {budget:g} GB is more than this machine's {ram:.0f} GB: the run may "
                               "swap"))
    return out


def runs_on_text(spec) -> str:
    """The run tabs' grey line under the Machine use: where a Grids, Pipeline or Custom job (duck-typed) runs, in
    words, from the device its quality implies (spec.mode()), or its Advanced Device override ('set in Advanced'), or
    a Custom job's extra options that name the device ('set in the Extra options'): the CPU, the GPU, or the CPU and
    the GPU together (job_mixes, in the Device row's words; runs_on_tip says when the program shares). Information
    only, no control behind it."""
    est, dim = getattr(spec, "estimator", "ps"), getattr(spec, "dim", 3)
    if dim == 2:
        return "Runs on: the CPU (the 2D programs have no GPU)"
    if not gpu_built(est, job_precision(spec), dim):
        return "Runs on: the CPU (this build has no GPU)"
    named = extra_device_flags(spec)
    if named:                           # the tab adds no device flag of its own (CustomSpec._own_command)
        if "--gpu" not in named:
            return "Runs on: the CPU (set in the Extra options: a GPU share without the GPU)"
        return ("Runs on: the GPU (set in the Extra options" + (", with their GPU share)" if "--gpu-share" in named
                                                                 else ")"))
    m = spec.mode()
    device = job_advanced(spec).get("gpu")
    if device is not None and not _refusal(spec, "gpu", device, m.deposit):
        why, small = "set in Advanced", ""
    else:
        why = (f"chosen for {QUALITY_NAME.get(spec.quality, QUALITY_NAME[QUALITY_DEFAULT])}, "
               f"{spec.run_mode.capitalize()}")
        small = ", a small snapshot" if est == "ps" else ""
    if not m.gpu:
        # (the sampled PS-DTFE deposit leaves the GPU only for a small snapshot, recommended_gpu)
        return f"Runs on: the CPU ({why}{small})"
    if job_mixes(spec, m):
        return f"Runs on: the CPU and the GPU together ({why})"
    return f"Runs on: the GPU ({why})"


# the precise statement behind 'the CPU and the GPU together' (the 'Runs on' line's tooltip; two lines at most)
RUNS_ON_MIXED_TIP = ("A run split into partitions deposits some of them on the CPU and the others on the GPU;\n"
                     "a small or unsplit run deposits on the GPU alone. The program's log says which.")


def runs_on_tip(spec) -> str:
    """The 'Runs on' line's tooltip: where the job mixes the two devices (job_mixes), when the program does; ''
    elsewhere (the line says it all)."""
    return RUNS_ON_MIXED_TIP if getattr(spec, "dim", 3) != 2 and job_mixes(spec) else ""


# the settings the two modes replaced: a saved job, queued job, settings sidecar or preset from before them may hold
# them; 'deposit' becomes the quality (migrate_quality), the rest is dropped on loading (the dataclasses no longer
# know them). CustomSpec keeps 'precision' and 'vertex_mass', PipelineSpec 'supersample' (the pixel's sub-samples)
REMOVED_SETTINGS = ("deposit", "nsub", "gpu", "face_gradient", "volume_weighted", "linear_deposit", "ordered_merge",
                    "parallel_triangulation", "partition", "max_concurrent", "mem_budget_gb", "pts_conventions",
                    "precision", "vertex_mass", "stream_density")


# the status line's note when the remembered jobs or queued ones predate the two modes (predates_modes), once, at
# the start: the one removed setting that changes what such a job computes, not only how fast, is named (a TNG
# job in double precision now runs the single binary); no measured number, and about as long as before (the status
# label sits in the run panel's button row)
MODES_NOTE = ("Older settings: the cell average is now the Quality, Exact or Fast; the code sets the rest (Grids "
              "and Pipeline: single precision)")


def predates_modes(d) -> bool:
    """Was this saved job (a dict: a tab's job, or a queue entry, whose job is its 'spec') written before the two
    quality modes? It names a cell average and no quality."""
    if isinstance(d, dict) and "kind" in d and isinstance(d.get("spec"), dict):
        d = d["spec"]
    return isinstance(d, dict) and "deposit" in d and "quality" not in d


def migrate_quality(d: dict) -> dict:
    """A saved job's settings (a launcher state, a queued job, a run's settings sidecar or log, a preset) in the
    two-mode model, before the dataclass drops what it does not know: an older one's cell average, 'deposit', becomes
    its quality -- 'sampled' Fast, anything else ('exact', '' = the production one) Exact; a 'quality' it holds
    wins."""
    d = dict(d)
    deposit = d.pop("deposit", None)
    if "quality" not in d and deposit is not None:
        d["quality"] = "fast" if deposit == "sampled" else "exact"
    return d


@dataclass
class RunSpec:
    data_root: str = field(default_factory=default_data_root)
    sim: str = ""
    snapshots: list[int] = field(default_factory=list)
    estimator: str = "ps"            # "ps" (run_ps_dtfe.sh) | "dtfe" (run_dtfe.sh)
    quality: str = QUALITY_DEFAULT   # "exact" | "fast" (QUALITIES): what it implies is quality_mode's (mode())
    grid: int = 512
    fields: list[str] = field(default_factory=lambda: list(PS_DEFAULT_FIELDS))
    run_mode: str = "balanced"       # DTFE_RUN_MODE, the machine use (RUN_MODES): how hard the run may lean on the
                                     # machine; balanced, the drivers' default
    caustics: bool = True            # --ps-caustics: on, as run_ps_dtfe.sh's default and every
                                     # production PS grid (+33% time, +1.2 GB on TNG)
    caustic_cusps: bool = False
    slice_plane: str = ""            # a --sample-points file (hi-res slice); "" = none
    slice_vel_grad: bool = True
    scratch_dir: str = ""
    output_prefix: str = "ps_output"
    lambda_th: float = LAMBDA_TH_DEFAULT   # T-web / V-web eigenvalue threshold (LAMBDA_TH; the scripts' default)
    tess_cache: str = ""             # TESSELLATION_CACHE: reuse the tessellation across runs (a local folder); "" = none
    start_over: bool = False         # DTFE_RESUME=0 (both drivers): a killed run's checkpoint is NOT resumed, the run
                                     # starts from the beginning and replaces it (the way out of a refused resume --
                                     # other settings, an incomplete checkpoint -- which fails every time)
    advanced: dict = field(default_factory=dict)   # the Advanced overrides (ADVANCED_KEYS -> value; absent =
                                     # Automatic); saved only when not empty (_with_advanced). The last field

    def __post_init__(self):
        if not self.quality:
            self.quality = QUALITY_DEFAULT
        self.advanced = clean_advanced(self.advanced, type(self))

    def mode(self) -> QualityMode:
        """What this job's quality implies (quality_mode), its device by the simulation's particle count, with the
        Advanced overrides it passes (apply_advanced; the precision first: it picks the program)."""
        return apply_advanced(quality_mode(self.quality, self.estimator, self.fields, lambda: dm_particles(self.sim),
                                           job_precision(self), run_mode=self.run_mode), self)

    # ------------------------------------------------------------------ data
    def sim_path(self) -> Path:
        return sim_dir(self.sim, self.data_root)

    # ------------------------------------------------------------------ checks
    def problems(self) -> list[tuple[str, str]]:
        """(level, message) pairs, level 'error' (blocks Run) or 'warning'."""
        out: list[tuple[str, str]] = _root_problems(self.data_root)
        if not out and not self.sim:
            out.append(("error", "no simulation selected"))
        elif not out and not self.sim_path().is_dir():
            out.append(("error", f"{self.sim} not found under {Path(self.data_root).expanduser()}"))
        if not self.snapshots:
            out.append(("error", "no snapshots selected"))
        if self.grid < 8 or self.grid > 4096:
            out.append(("error", f"grid {self.grid}^3 is outside 8..4096"))
        out += _scratch_problems(self.scratch_dir)
        out += _run_option_problems(self)
        out += advanced_problems(self)
        if self.estimator == "dtfe":
            if not dtfe_fields(self.fields):
                out.append(("error", "no field standard DTFE computes is ticked (the velocity dispersion is PS-DTFE's)"))
            elif "dispersion_a" in self.fields:
                out.append(("info", "the velocity dispersion is a phase-space field: standard DTFE leaves it out"))

        if self.estimator == "ps":
            out += _ps_option_problems(self)
            if self.slice_plane and not Path(self.slice_plane).is_file():
                out.append(("error", f"slice file not found: {self.slice_plane}"))
            if self.output_prefix.strip() == "" or re.search(r"[\s/]", self.output_prefix):
                out.append(("error", "output prefix must be a single word (no spaces or '/')"))
        else:
            if self.slice_plane and not Path(self.slice_plane).is_file():
                out.append(("error", f"slice file not found: {self.slice_plane}"))

        out += _busy_problems()

        # memory: the full-resolution grids are irreducible (partitions share them)
        gb = self.grid_gb()
        slice_pts = plane_points(self.slice_plane) if self.slice_plane else 0
        if not self.scratch_dir and gb + slice_pts * 545e-9 > 40:
            out.append(("warning", f"~{gb:.0f} GB of full-resolution grids"
                                   + (f" + ~{slice_pts * 545e-9:.0f} GB for {slice_pts/1e6:.0f}M slice points"
                                      if slice_pts else "")
                                   + " will not fit next to the triangulation on 64 GB: set a scratch "
                                     "directory (bit-identical results, grids on disk)"))
        return out

    def grid_gb(self) -> float:
        return grids_gb(self.grid, self.fields if self.estimator == "ps" else dtfe_fields(self.fields),
                        job_precision(self), face=face_route_on(self))

    def runnable(self) -> bool:
        return not any(level == "error" for level, _ in self.problems())

    # ------------------------------------------------------------------ command
    def command(self) -> tuple[list[str], dict[str, str]]:
        """argv of the run script and the environment variables it adds; the quality's settings from mode(), the
        Advanced precision, partitions and concurrency from their accessors (both drivers read these knobs)."""
        m = self.mode()
        env = {"DTFE_DATA_ROOT": str(self.data_root)}
        part, conc = job_partition(self), job_concurrency(self)
        if job_precision(self) == "double":
            env["DTFE_PRECISION"] = "double"
        if part > 0:
            env["PARTITION"] = f"{part} {part} {part}"
        if conc > 0:
            env["MAX_CONCURRENT"] = str(conc)
        if self.scratch_dir:                # both scripts take it (run_dtfe.sh too, so a
            env["SCRATCH_DIR"] = str(Path(self.scratch_dir).expanduser())   # standard run does not keep the grids in RAM)
        if self.lambda_th != LAMBDA_TH_DEFAULT:
            env["LAMBDA_TH"] = f"{self.lambda_th:g}"
        env.update(_run_option_env(self))
        snaps = [str(n) for n in sorted(self.snapshots)]
        if self.estimator == "dtfe":
            argv = [str(DTFE_SCRIPT), "-s", self.sim, "-g", str(self.grid)]
            if dtfe_fields(self.fields) != DTFE_DEFAULT_FIELDS:   # the ticks (T-web / V-web, a lighter set); else the script's
                env["FIELDS"] = " ".join(dtfe_fields(self.fields))
            if self.slice_plane:                            # the hi-res slice: the standard interpolant at the points
                env["SAMPLE_POINTS"] = str(self.slice_plane)
                env["PTS_VEL_GRAD"] = "1" if self.slice_vel_grad else "0"
            env["DTFE_GPU"] = "1" if m.gpu else "0"          # -m only switches the GPU ON; pin the env knob too
            env["DTFE_RUN_MODE"] = self.run_mode             # the machine use, spelled out (ps_run_env for PS-DTFE)
            # and the cell average, both ways: run_dtfe.sh's default is the exact one, so a
            # 'sampled' job without this would run exact (-e, like -m, only switches it on)
            env["DTFE_EXACT_AVERAGE"] = "1" if m.deposit == "exact" else "0"
            # the knobs' older names too, until P6.6 (run_dtfe.sh reads them with a warning; the new names win): a
            # stray older export in the login shell is overwritten, not left beside the new one
            env["DTFE_METAL"], env["DTFE_EXACT"] = env["DTFE_GPU"], env["DTFE_EXACT_AVERAGE"]
            # the start-over tick reaches run_dtfe.sh too (it resumes a killed partitioned run by itself,
            # DTFE_RESUME, the knob both drivers read), spelled out both ways: a stray export cannot leak in
            env["DTFE_RESUME"] = "0" if self.start_over else "1"
            # the Monte-Carlo count and the thread cap, pinned EMPTY (run_dtfe.sh's own defaults: the binary's count,
            # all cores): the Grids tab sets neither for a standard run, so a SUBSAMPLES or THREADS exported in the
            # login shell cannot leak in. An OMP_NUM_THREADS export still reaches the binary (run_dtfe.sh sets it
            # only for a non-empty THREADS)
            env["SUBSAMPLES"] = env["THREADS"] = ""
            # the gradient route, pinned EMPTY the same way: run_dtfe.sh's own default, the binary's (the face
            # gradient, standard DTFE's default since 2026-11-09, whatever the cell average). A standard job names no
            # route, so a FACE_GRADIENT=0 exported in the login shell cannot turn it into '--per-tet-gradient', nor a
            # 1 add '--face-gradient' to the command (and the run log) the job shows
            env["FACE_GRADIENT"] = ""
            # ... and the binary's own switch the same way (it reads an empty value as unset): install_gui_app.sh
            # imports every DTFE_* from the login shell, where a DTFE_FACE_GRADIENT=0 would turn the job to the
            # deposit's own gradient behind the command it shows
            env["DTFE_FACE_GRADIENT"] = ""
            if m.gpu:
                argv.append("-m")
            if m.deposit == "exact":
                argv.append("-e")
            return argv + snaps, env
        argv = [str(PS_SCRIPT), "-s", self.sim, "-g", str(self.grid)]
        if m.deposit == "exact":
            # the exact deposit has no sub-samples: no -n, and the count pinned EMPTY (the driver's own default), so a
            # SUBSAMPLES exported in the login shell does not reach the driver, which would note on every run that the
            # exact deposit ignores it
            env["SUBSAMPLES"] = env["AVG_SUBSAMPLES"] = ""
        else:
            argv += ["-n", str(m.nsub)]
        if m.gpu:
            argv.append("-m")
        if m.deposit == "exact":
            argv.append("-e")
        env.update({
            "FIELDS": " ".join(self.fields),
            "PS_VERTEX_MASS": "1" if m.vertex_mass else "0",
            "PS_VOLUME_WEIGHTED": "1" if m.volume_weighted else "0",
            "PS_CAUSTICS": "1" if self.caustics else "0",
            "PS_CAUSTIC_CUSPS": "1" if self.caustic_cusps else "0",
            "PS_LINEAR_DEPOSIT": "1" if m.linear_deposit else "0",
            "FACE_GRADIENT": face_gradient_env(self),
            # ... and the binary's own switch pinned EMPTY (read as unset), as for a standard job: where
            # FACE_GRADIENT is empty (a job that does not qualify) the driver passes no route flag, and a
            # DTFE_FACE_GRADIENT=1 the launcher imported from the login shell would make the binary refuse the
            # sampled job that keeps the stream average
            "DTFE_FACE_GRADIENT": "",
            "OUTPUT_PREFIX": self.output_prefix,
        })
        env.update(ps_run_env(self))
        env.update(ps_exact_env(m.deposit == "exact"))     # -e only switches it on: pinned both ways
        if self.slice_plane:
            env["SAMPLE_POINTS"] = str(self.slice_plane)
            env["PTS_VEL_GRAD"] = "1" if self.slice_vel_grad else "0"
            env.update(ps_conventions_env(m.pts_conventions))   # spelled out: a stray export cannot leak in
        return argv + snaps, env

    def shell_line(self) -> str:
        """The equivalent terminal command (repo-relative script, quoted as needed)."""
        return shell(*self.command())

    def memory_key(self) -> str:
        return _memory_key(self, ("snapshots", "lambda_th", "tess_cache", "start_over"))

    def check_step(self) -> "Step":
        """The same run_ps_dtfe.sh / run_dtfe.sh call in report mode: the binary's own memory prediction per
        snapshot, nothing computed or written. Both estimators (the standard binary's tuner reports too)."""
        argv, env = self.command()
        env["AUTO_TUNE_REPORT"] = "1"
        return Step(f"memory check {self.sim}", argv, env)

    def steps(self) -> list["Step"]:
        argv, env = self.command()
        return [Step(f"{Path(argv[0]).name} {' '.join(argv[len(argv) - len(self.snapshots):])}", argv, env)]

    def to_dict(self) -> dict:
        return _with_advanced(self, asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        """A saved job: an older one's deposit read as its quality (migrate_quality), unknown keys dropped (the
        settings the modes replaced among them: never an override), bad overrides too (__post_init__)."""
        known = {k: v for k, v in migrate_quality(d).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------- steps and quoting
@dataclass
class Step:
    """One command of a job: what the GUI runs, and shows, for it. tee: also copy the output to
    this file (the run log), exactly as '2>&1 | tee <file>' would."""
    label: str
    argv: list[str]
    env: dict[str, str]
    tee: str = ""
    after: str = ""         # the label of a step that must have exited 0 first (a merge after its download)

    def line(self) -> str:
        return shell(self.argv, self.env) + (f" 2>&1 | tee {_q(self.tee)}" if self.tee else "")


def _q(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./:=+,-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def _display_path(a: str) -> str:
    """Repo files repo-relative, the PATH python3 as 'python3'; everything else verbatim."""
    if a == PYTHON and shutil.which("python3") == a:
        return "python3"
    try:
        if not Path(a).is_absolute():
            return a
        rel = str(Path(a).relative_to(REPO_ROOT))
        return rel if "/" in rel else "./" + rel      # a repo-root binary must not be looked up on PATH
    except ValueError:
        return a


def shell(argv: list[str], env: dict[str, str]) -> str:
    """The terminal command for argv + added environment, runnable from the repository root. A knob's older name
    (OLDER_KNOB_NAMES) beside its new one with the same value is left out: the jobs export both until P6.6 (to
    overwrite a stray older export the launcher inherits), the command shown and exported names each knob once."""
    env = {k: v for k, v in env.items() if not (k in OLDER_KNOB_NAMES and env.get(OLDER_KNOB_NAMES[k]) == v)
           and not (k == "ORDERED_MERGE" and env.get("PS_MERGE") == {"1": "index-order", "0": "completion-order"}.get(v))}
    return " ".join([f"{k}={_q(v)}" for k, v in env.items()]
                    + [_q(_display_path(argv[0]))] + [_q(_display_path(a)) for a in argv[1:]])


def script_text(steps: list[Step]) -> str:
    return "\n".join(st.line() for st in steps)


def export_script(steps: list[Step], title: str, slurm: dict | None = None) -> str:
    """The job as a runnable bash script (run from the repository root, as the GUI does), or as a
    SLURM batch job for a cluster (e.g. Hábrók): slurm = {job_name, time, cpus, mem, partition,
    mail} (empty values are left out). Like the GUI, the script goes on after a failed step. A SLURM job runs in
    performance: each step's DTFE_RUN_MODE, and a Custom step's '--run-mode' value, become 'performance' (the
    launcher's machine use is for this machine)."""
    if slurm is not None:
        steps = [_cluster_step(st) for st in steps]
    lines = ["#!/usr/bin/env bash"]
    if slurm is not None:
        opts = [("job-name", slurm.get("job_name") or "dtfe"), ("time", slurm.get("time")),
                ("cpus-per-task", slurm.get("cpus")), ("mem", slurm.get("mem")),
                ("partition", slurm.get("partition")), ("output", "%x-%j.log")]
        if slurm.get("mail"):
            opts += [("mail-type", "END,FAIL"), ("mail-user", slurm["mail"])]
        lines += [f"#SBATCH --{k}={v}" for k, v in opts if v not in (None, "", 0)]
    lines += [f"# {title}",
              f"# exported by Lamina on {time.strftime('%Y-%m-%d %H:%M')}: {len(steps)} "
              f"command{'s' if len(steps) != 1 else ''}, run in order from the repository root",
              "# (on another machine: fix REPO and the data paths first)",
              f"REPO={_q(str(REPO_ROOT))}",
              'cd "$REPO" || exit 1']
    if slurm is not None:
        lines.append('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"   # the binaries use OpenMP')
        lines.append("# machine use: performance on the cluster (the launcher's choice is for this machine)")
    lines += ["set -o pipefail", "failed=0"]
    for st in steps:
        lines += [f"# {st.label}", _export_line(st)]
    lines += ['[ "$failed" -eq 0 ] || echo "$failed step(s) failed" >&2', 'exit "$failed"', ""]
    return "\n".join(lines)


def _cluster_step(st: Step) -> Step:
    """A step as a SLURM job runs it: its DTFE_RUN_MODE (a driver's knob) and its '--run-mode' value (a Custom step
    runs the binary itself) set to performance, in a copy (the job's own steps stay as they are)."""
    env, argv = dict(st.env), list(st.argv)
    if "DTFE_RUN_MODE" in env:
        env["DTFE_RUN_MODE"] = "performance"
    for i, a in enumerate(argv[:-1]):
        if a == "--run-mode":
            argv[i + 1] = "performance"
    return Step(st.label, argv, env, st.tee, st.after)


def _export_line(st: Step) -> str:
    """One step of an exported script. A step with a run log keeps the run-log contract as the launcher's own
    tee does (_close_tee): the output goes into '<log>.partial', which becomes '<log>' on exit 0 (a stale
    '<log>.failed' goes) and '<log>.failed' otherwise, so a failed re-run never truncates the log of the grids on
    disk. pipefail (set by the header) makes the pipe's status the program's, not tee's."""
    if not st.tee:
        return st.line() + " || failed=$((failed + 1))"
    log, part = _q(st.tee), _q(st.tee + ".partial")
    failed = _q(st.tee + ".failed")
    return (f"if {shell(st.argv, st.env)} 2>&1 | tee {part}; then\n"
            f"    mv -f {part} {log} && rm -f {failed}\n"
            f"else\n"
            f"    mv -f {part} {failed}; failed=$((failed + 1))\n"
            f"fi")


def _scratch_problems(scratch_dir: str) -> list[tuple[str, str]]:
    if not scratch_dir:
        return []
    sd = Path(scratch_dir).expanduser()
    if not sd.is_dir():
        return [("error", f"scratch directory does not exist: {sd}")]
    if "Mobile Documents" in str(sd.resolve()):
        return [("error", "the scratch directory is on iCloud Drive; the binary refuses it "
                          "(use a local path, e.g. /private/tmp/dtfe-scratch)")]
    return []


def machine_ram_gb() -> float:
    """This machine's physical memory in GB (0 when the OS will not say)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def _run_option_problems(spec) -> list[tuple[str, str]]:
    """The shared run options: the quality and the machine use (an unknown one is an error; an OMP_NUM_THREADS the
    launcher inherited keeps its thread count in every mode, the binaries' rule) and the tessellation cache folder
    (local, existing: the binary refuses iCloud and does not create it -- the scripts do, the Custom run does not)."""
    out = []
    quality = getattr(spec, "quality", QUALITY_DEFAULT)
    if quality not in QUALITIES:
        out.append(("error", f"Quality: unknown value {quality} (exact or fast)"))
    mode = getattr(spec, "run_mode", "balanced")
    if mode not in RUN_MODES:
        out.append(("error", f"Machine use: unknown value {mode}"))
    if os.environ.get("OMP_NUM_THREADS"):
        out.append(("info", f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} in the launcher's environment keeps "
                            "the thread count whatever the machine use"))
    tc = getattr(spec, "tess_cache", "")
    if tc:
        d = Path(tc).expanduser()
        if "Mobile Documents" in str(d.resolve()):
            out.append(("error", "the tessellation cache is on iCloud Drive; the binary refuses it (use a local "
                                 "folder, e.g. ~/Library/Caches/DTFE/tessellations)"))
        elif not d.is_dir() and isinstance(spec, CustomSpec):
            out.append(("error", f"tessellation cache folder does not exist: {d}"))
    return out


def _run_option_env(spec) -> dict[str, str]:
    """DTFE_MEM_BUDGET_GB (the auto-tuner's budget, an Advanced override: read by the binary itself, through any
    script; unset, the binary's own, the free memory) and TESSELLATION_CACHE (the run scripts' --tessellation-cache
    hook; TESS_CACHE, its older name, too until P6.6), only when set."""
    env = {}
    budget = job_memory_budget(spec)
    if budget > 0:
        env["DTFE_MEM_BUDGET_GB"] = f"{budget:g}"
    if getattr(spec, "tess_cache", ""):
        env["TESSELLATION_CACHE"] = env["TESS_CACHE"] = str(Path(spec.tess_cache).expanduser())
    return env


# the run scripts' knobs under their older names (P6.7, config.sh 'knob'): the launcher exports BOTH spellings until
# P6.6, the same value, so a stray older export in the login shell (install_gui_app.sh imports every DTFE_*) is
# overwritten instead of standing beside the new one; the drivers read the new name and warn once per older one
TESS_CACHE_KEYS = ("TESSELLATION_CACHE", "TESS_CACHE")
# older name -> new name of every knob the launcher exports both ways (the drivers' config.sh 'knob' / 'knob_merge'
# lines); ORDERED_MERGE's new name is PS_MERGE, with a value of its own (1 = index-order, 0 = completion-order)
OLDER_KNOB_NAMES = {"PS_GPU": "DTFE_GPU", "PS_METAL": "DTFE_GPU", "DTFE_METAL": "DTFE_GPU",
                    "PS_EXACT": "DTFE_EXACT_AVERAGE", "DTFE_EXACT": "DTFE_EXACT_AVERAGE", "PS_RESUME": "DTFE_RESUME",
                    "AVG_SUBSAMPLES": "SUBSAMPLES", "PS_PARALLEL_TRI": "PARALLEL_TRIANGULATION",
                    "PTS_CONVENTIONS": "PS_POINT_CONVENTIONS", "TESS_CACHE": "TESSELLATION_CACHE",
                    "WINDOW": "PLANE_WINDOW", "PY": "PYTHON"}


def ps_run_env(spec) -> dict[str, str]:
    """run_ps_dtfe.sh's run switches as a Grids or Pipeline job pins them (spelled out both ways: a stray export
    cannot leak in), each under its new name and its older one: the device (-m only switches the GPU on), the
    parallel triangulation and the merge order of its quality (spec.mode()), and the start-over tick. The index order
    is given explicitly (PS_MERGE=index-order, which keeps the memory gate on under the auto-tuner's speed split, as
    the launcher's ORDERED_MERGE=1 always did). The machine use (DTFE_RUN_MODE) and the GPU share of the mode
    (GPU_SHARE: 'auto', the machine use's share of an exact deposit on the GPU, which the driver passes only for that
    deposit; a Device override's '1' or 'mixed' only where it applies, apply_advanced: the driver passes any other
    value whatever the run), pinned the same way."""
    m = spec.mode()
    on = lambda b: "1" if b else "0"
    return {"DTFE_GPU": on(m.gpu), "PS_GPU": on(m.gpu), "PS_METAL": on(m.gpu),
            "PARALLEL_TRIANGULATION": on(m.parallel_triangulation), "PS_PARALLEL_TRI": on(m.parallel_triangulation),
            "PS_MERGE": "index-order" if m.ordered_merge else "completion-order",
            "ORDERED_MERGE": on(m.ordered_merge),       # K1
            "DTFE_RESUME": on(not spec.start_over), "PS_RESUME": on(not spec.start_over),
            "DTFE_RUN_MODE": spec.run_mode, "GPU_SHARE": m.gpu_share}


def ps_exact_env(exact: bool) -> dict[str, str]:
    """run_ps_dtfe.sh's deposit knob, spelled out both ways: DTFE_EXACT_AVERAGE (the knob both drivers read; a login
    export of it must not turn the job's choice around, and without the pin a SUBSAMPLES export would make the
    driver sample) and PS_EXACT, its older name."""
    return {"DTFE_EXACT_AVERAGE": "1" if exact else "0", "PS_EXACT": "1" if exact else "0"}


def ps_conventions_env(conv: str) -> dict[str, str]:
    """The plane conventions for run_ps_dtfe.sh / run_ps_pipeline.sh: PS_POINT_CONVENTIONS and PTS_CONVENTIONS."""
    return {"PS_POINT_CONVENTIONS": conv, "PTS_CONVENTIONS": conv}


def _root_problems(data_root: str) -> list[tuple[str, str]]:
    if not str(data_root).strip():
        return [("error", "no data root: choose the folder that holds the simulations (the Data root field, or Setup)")]
    root = Path(data_root).expanduser()
    return [] if root.is_dir() else [("error", f"data root not found: {root} (an external disk that is unplugged?)")]


def _busy_problems() -> list[tuple[str, str]]:
    """Another run on the machine is a warning (two production runs do not fit in memory together); a resident
    query server is told with its memory (a 24 GB server beside a large run can make the Mac swap), nothing
    more -- the auto-tuner's budget already starts from the free memory."""
    procs = dtfe_processes()
    out = []
    if procs["runs"]:
        out.append(("warning", "already running: " + ", ".join(f"{name} (pid {pid}{_gb_text(gb)})"
                                                                for pid, name, gb in procs["runs"])
                               + "; two production runs do not fit in memory together"))
    if procs["servers"]:
        out.append(("info", "query server in memory: " + ", ".join(f"{name} (pid {pid}{_gb_text(gb)})"
                                                                     for pid, name, gb in procs["servers"])
                            + "; a large run has to fit beside it (stop the server first if it does not)"))
    return out


# ---------------------------------------------------------------------- memory check
# run_ps_dtfe.sh (and run_dtfe.sh) AUTO_TUNE_REPORT=1 prints, per snapshot, what the binary's
# own auto-tuner would decide and how much memory it predicts -- the same model and stream estimate a real run uses.
# The binary itself (a custom-snapshot run) prints the line without 'snap=': it is keyed 0.
_REPORT = re.compile(r"AUTO-TUNE-REPORT (?:snap=(\d+) )?(\w+=.*)$")


def parse_reports(text: str) -> dict[int, dict]:
    """{snapshot: {partition, mc, predicted_gb, budget_gb, ..., over_budget}} from report output
    (the script's lines carry 'snap=NNN'; the bare binary's line is snapshot 0)."""
    out = {}
    for line in text.splitlines():
        m = _REPORT.search(line)
        if not m:
            continue
        rec = {}
        for kv in m.group(2).split():
            k, _, v = kv.partition("=")
            try:
                rec[k] = int(v) if k in ("partition", "mc", "over_budget") else float(v)
            except ValueError:
                pass
        out[int(m.group(1) or 0)] = rec
    return out


def report_split_text(rec: dict, estimator: str, dim: int = 3) -> str:
    """How a report's plan reads: PS-DTFE runs 'mc' of its partitions at a time; standard DTFE runs its
    partitions one after another, each triangulated by 'mc' threads (sub-boxes) at once -- the same report line, two
    meanings of mc (auto_tune.h AutoTuneReport)."""
    p, mc = int(rec.get("partition", 1)), int(rec.get("mc", 1))
    power = "²" if dim == 2 else "³"
    if estimator == "ps":
        return f"{p}{power} partitions, {mc} at a time" if p > 1 else "one tessellation"
    threads = f"{mc} thread{'s' if mc != 1 else ''}"
    return f"{p}{power} partitions one after another, {threads} each" if p > 1 else f"one domain, {threads}"


def _memory_key(spec, ignore: tuple) -> str:
    """The settings that change a run's memory, as a string (check results are kept per key). An Advanced override
    the job does not pass (advanced_refused) is left out: it changes no command, so neither the check's answer."""
    import json
    d = {k: v for k, v in spec.to_dict().items() if k not in ignore}
    if "advanced" in d:
        refused = advanced_refused(spec)
        d["advanced"] = {k: v for k, v in d["advanced"].items() if k not in refused}
        if not d["advanced"]:
            del d["advanced"]
    return json.dumps(d, sort_keys=True)


ADAPTIVE_MIN_NU = 1024          # the adaptive plane size never goes below this


# ---------------------------------------------------------------------- pipeline
POINTEVAL_DERIVATIVE_SMOOTH = 10.0   # == dtfelib.pointeval.DERIVATIVE_SMOOTH (not imported: it pulls in matplotlib)
POINTEVAL_FIELDS = ("density", "streams", "speed", "dispTrace", "dispMag", "velDiv", "velShear", "velVort",
                    "denGradMag")    # == the keys of dtfelib.pointeval.STYLES (plot_pointeval.py --fields)
PLANE_GHOSTING_MIN = 16              # make_image_plane.py: fewer planes across a slab ghost every inclined structure
# the options that only change the figures: they do not enter the memory key
PIPELINE_RENDER_KEYS = ("render", "render_fields", "render_project", "render_smooth", "render_smooth_derivatives",
                        "render_fixed_range", "render_force")
# what run_ps_pipeline.sh reads itself; everything else in command()'s environment is for run_ps_dtfe.sh
# (DTFE_GPU is run_ps_dtfe.sh's knob too: the pipeline's memory check hands it on). PLANE_WINDOW and PYTHON with
# their older names WINDOW and PY (both exported until P6.6)
PIPELINE_ONLY_KEYS = ("SIMS", "SNAPS", "NU", "AXIS", "CENTER", "PLANES", "THICKNESS", "SUPERSAMPLE", "PLANE_WINDOW",
                      "WINDOW", "GRID_SIZE", "FORCE", "DRY_RUN", "PYTHON", "PY")


def face_gradient_applies(spec) -> bool:
    """Does this PS-DTFE job qualify for the face gradient (duck-typed: Grids, Pipeline and Custom jobs)? The binary's
    rule (src/user_options.cc faceGradientBlocker) on its quality's settings (spec.mode()): the exact deposit, the
    volume-weighted velocities, no halo release, 3D."""
    m = spec.mode()
    return (m.deposit == "exact" and m.volume_weighted
            and not getattr(spec, "halo_release", 0.0) > 0 and getattr(spec, "dim", 3) == 3)


def face_gradient_env(spec) -> str:
    """The velocity gradient's route of a PS-DTFE job, as run_ps_dtfe.sh's FACE_GRADIENT: '1' (--face-gradient) where
    its quality asks for the face gradient (spec.mode()) and the job qualifies, '' (the binary's default, which keeps
    the stream average there and says why in the run log) where it does not qualify, '0' (--per-tet-gradient) where
    the quality asks for the stream average. Spelled out every time, so a FACE_GRADIENT exported in the login shell
    cannot leak in."""
    if not spec.mode().face_gradient:
        return "0"
    return "1" if face_gradient_applies(spec) else ""


def face_route_on(spec) -> bool:
    """Does this job's run take the face gradient and keep its full grids (grids_gb)? A PS-DTFE job (duck-typed:
    Grids, Pipeline and Custom; a job without an estimator is the Pipeline's) whose route is the face gradient
    (face_gradient_env '1'). The binary takes it only where a field built on the gradient is asked, which grids_gb
    checks itself."""
    return getattr(spec, "estimator", "ps") == "ps" and face_gradient_env(spec) == "1"


def _quality_problems(spec, derivatives: bool = True) -> list[tuple[str, str]]:
    """What a PS-DTFE job's quality leaves out (Grids, Pipeline and Custom jobs, duck-typed), said before the run. The
    velocity derivatives, where a field built on the velocity gradient is ticked (_FROM_GRADIENT; not when
    'derivatives' is False: a Custom job's extra options name the route): Exact takes the mean flow's from exact face
    sums wherever the job qualifies (face_gradient_applies); in 2D and with the halo release it keeps the stream
    average (and writes no mass-flux divergence), and the sampled deposit always does (the face sums need the exact
    one). The caustic classes of the sampled deposit are set per sample point, so a fold cell can show none."""
    m, out = spec.mode(), []
    if derivatives and any(f in _FROM_GRADIENT for f in spec.fields) and not face_gradient_applies(spec):
        if m.deposit != "exact":
            out.append(("info", "Fast: PS-DTFE's velocity derivatives are the stream average"))
        elif m.face_gradient:
            out.append(("info", "Exact with halo release, or in 2D: the velocity derivatives are the stream average; "
                                "no mass-flux divergence"))
    if m.deposit != "exact" and spec.caustics:
        out.append(("info", "Fast with caustics: the classes are recorded per sample point; some fold cells show "
                            "none"))
    return out


def _ps_option_problems(spec) -> list[tuple[str, str]]:
    """The phase-space run options' checks, shared by the Grids and Pipeline jobs (duck-typed on the option names
    both carry: fields, caustics, caustic_cusps, run_mode; the quality's settings from spec.mode())."""
    m = spec.mode()
    out: list[tuple[str, str]] = []
    if not spec.fields:
        out.append(("error", "no fields selected"))
    if spec.caustic_cusps and not spec.caustics:
        out.append(("error", "caustic cusps need 'caustics' switched on"))
    if spec.caustic_cusps and m.gpu:
        out.append(("warning", "caustic cusps are a CPU-deposit pass; the GPU run leaves them to the CPU"))
    out += _quality_problems(spec)
    return out


def _num(v: float) -> str:
    """A float for a script argument that reads back as the same float ('%g' kept six digits, so the
    launcher compared other numbers than the sidecar held): whole numbers plain, others as repr."""
    v = float(v)
    return str(int(v)) if v == int(v) else repr(v)


def words(text) -> list[str]:
    """A free-text list option as typed ('2, 4 8', 'density,velDiv'): its words, commas and spaces alike. The
    scripts take space-separated lists (R7), so each word goes into the command as its own argument."""
    return str(text).replace(",", " ").split()


def grids_gb(grid: int, fields: list[str], precision: str = "single", dim: int = 3, face: bool = False) -> float:
    """The full-resolution grids a run holds at once at its peak, in GB (roughly: src/auto_tune.cc has the exact
    model): the deposited grids throughout, and the larger of what the deposit and the normalization add beside them
    and what the derived fields and web classes add after them (_POST_FIELDS). face: a PS-DTFE run that takes the
    face gradient (face_route_on): its face sums (and the mean velocity it deposits for them) to the normalization,
    its two face outputs and the twins from there to the output (_FACE_*)."""
    b = 8 + sum(_FIELD_BYTES.get(f, 12) for f in fields if f not in _POST_FIELDS)   # + stream counts/weights
    before, after = 0, sum(_FIELD_BYTES[f] for f in fields if f in _POST_FIELDS)
    if any(f in _FROM_GRADIENT for f in fields):
        b += _GRADIENT_BYTES
        if face and dim == 3:                               # (the face gradient is 3D only)
            kept = _FACE_OUTPUT_BYTES + sum(_FACE_TWIN_BYTES.get(f, 0) for f in fields)
            before = (_FACE_SUM_BYTES + kept
                      + (0 if "velocity_a" in fields else _FIELD_BYTES["velocity_a"]))
            after += kept
    b += max(before, after)
    if precision == "double":
        b *= 2                                              # Real = double: every grid twice the bytes
    return grid ** dim * b / 1e9


PRECISIONS = ("single", "double")
PRECISION_TEXT = {"single": "single precision (the default)",
                  "double": "double precision: positions and fields in float64 from the input read onward, "
                            "about twice the memory"}


DIMS = (3, 2)
DIM_TEXT = {3: "3D", 2: "2D: a snapshot in a plane (two-column coordinates; the 2D programs, CPU only)"}

# The machine use: how hard a run may lean on the machine (both binaries' '--run-mode', the drivers' DTFE_RUN_MODE).
# Balanced is the drivers' and the launcher's default; the binaries' own is performance (a SLURM export runs it).
# Each mode's menu text and its item tip (at most two lines).
RUN_MODES = ("gentle", "balanced", "performance")
RUN_MODE_TEXT = {
    "gentle": ("Gentle: the machine stays fully usable",
               "Half the performance cores and short GPU buffers, and on Linux a lower priority:\n"
               "the machine stays usable while the run takes what is left."),
    "balanced": ("Balanced: most of the machine (the default)",
                 "The performance cores less a quarter and short GPU buffers:\n"
                 "the display never waits on the GPU."),
    "performance": ("Performance: the fastest run",
                    "Every core and long GPU buffers; an Exact phase-space run split into partitions\n"
                    "also deposits on the free cores. The binaries' default, and a cluster job's."),
}


def binary_name(estimator: str, precision: str = "single", dim: int = 3) -> str:
    """The binary a job runs: PS-DTFE / DTFE, then '-2d' for the 2D set ('make ... DIM=2') and
    '-double' for the double-precision pair -- the Makefile's names (PS-DTFE-2d-double)."""
    return (("PS-DTFE" if estimator == "ps" else "DTFE") + ("-2d" if dim == 2 else "")
            + ("-double" if precision == "double" else ""))


def binary_built(estimator: str, precision: str = "single", dim: int = 3) -> bool:
    return (REPO_ROOT / binary_name(estimator, precision, dim)).is_file()


def build_hint(precision: str = "single", dim: int = 3) -> str:
    """The make call that builds a pair (the targets are DTFE and PS-DTFE whatever the set)."""
    args = ["make", "DTFE", "PS-DTFE"] + (["DIM=2"] if dim == 2 else []) + (["DOUBLE=1"] if precision == "double" else [])
    return " ".join(args)


def _precision_problems(estimator: str, precision: str, dim: int = 3) -> list[tuple[str, str]]:
    if precision not in PRECISIONS:
        return [("error", f"unknown precision '{precision}' (single or double)")]
    if precision == "double" and not binary_built(estimator, "double", dim):
        gpu = "" if dim == 2 else ", with METAL=1 on Apple Silicon"
        return [("error", f"the double-precision program {binary_name(estimator, 'double', dim)} is not built: "
                          + ("Setup builds the pair (or " if dim == 3 else "build it with ")
                          + f"'{build_hint('double', dim)}'{gpu}" + (")" if dim == 3 else ""))]
    return []


def _plot_env(data_root: str, sim: str) -> dict[str, str]:
    # DTFE_SIM keeps python/config.py (box size, figure names) on the plotted simulation;
    # MPLBACKEND=Agg because several scripts end in plt.show(), which would block a GUI run
    return {"DTFE_DATA_ROOT": str(data_root), "DTFE_SIM": sim, "MPLBACKEND": "Agg"}


# ---------------------------------------------------------------------- figures
@dataclass(frozen=True)
class Opt:
    """One option of a figure set. kind: bool | int | float | choice | text | plane | multi (a checklist of
    'choices', the value a comma-separated string of the ticked ones). limits: a float's (min, max, decimals)
    when the default box (0-1000, 2 decimals) cannot hold its values. row: options sharing one row title sit
    side by side on one line under it (the void scripts' own options: compact), label then short."""
    key: str
    label: str
    kind: str
    default: object
    choices: tuple = ()
    help: str = ""
    limits: tuple = ()
    row: str = ""


# analyze.py's core plot scripts (its PLOT_SCRIPTS: kept equal by the runspec test)
CORE_SCRIPTS = ("plot_DTFE.py", "plot_velDiv_den.py", "plot_eigenvalues.py", "plot_shear_triaxial.py",
                "plot_PDF_CDF.py", "plot_contour_filter.py", "plot_shape_filter.py", "plot_ellipses_contour_3D.py",
                "plot_marked_correlation_BBKS.py", "plot_smoothing_comparison.py", "plot_phi_delta.py",
                "plot_tidal_correlation.py")
POINTEVAL_DERIV_SMOOTH = 10.0   # dtfelib.pointeval.DERIVATIVE_SMOOTH (plot_pointeval's own default): the runspec test
DENSITY_LIMITS = (0.0, 1e7, 4)  # rho/rho_bar: TNG100-3-Dark z=0 reaches 2.3e5 (99.99th pct 913); voids go below 0.01


@dataclass(frozen=True)
class FigureSet:
    key: str
    title: str
    script: str             # path under python/
    per: str                # "snapshot" (one call each) | "series" (one call, all) | "sim"
    about: str
    output: str             # where under FIGURES_ROOT the figures land
    common: bool = True     # takes dtfelib's --estimator/--prefix/--smooth
    opts: tuple = ()


_REF = Opt("ref", "Reference snapshot", "int", 99, help="The snapshot the tracking starts from")
_RAW = Opt("raw", "Unaveraged fields", "bool", False,
           help="The raw fields instead of the volume-averaged ones")
_PLANE = Opt("plane", "Image plane", "plane", "",
             help="the plane sidecar (.json) the .pts_* files were evaluated on")
# the common cross-box smoothing (contract G2): python/config.py's, one number for every Python user;
# 2.1625 (10.0002 TNG100-3-Dark cells) while a config predates the constant
CROSSBOX_SMOOTH_MPC = float(getattr(config, "CROSSBOX_SMOOTHING_MPC", 2.1625))
# the void figure sets' common options share one 'Void options' row on each set's page (radius, smoothing, overlaps)
VOID_ROW = "Void options"
# the void scripts' smoothing in Mpc: the same physical scale in every box, the minima window following
_SMOOTH_MPC = Opt("smooth_mpc", "smoothing (Mpc)", "float", 0.0,
                  help=f"0 = the default, 10 cells. E.g. {CROSSBOX_SMOOTH_MPC:g}, TNG100's 10 cells: TNG50, TNG100 and TNG300 then find the same voids; "
                       "the outputs get their own names", limits=(0.0, 100.0, 4), row=VOID_ROW)
_RADIUS = Opt("radius", "radius", "choice", "r_v", ("r_v", "r_eff"),
              help="r_v: the measured radius, where the smoothed density around the void rises back to the mean; "
                   "r_eff: the ellipsoid fit's curvature length, about 1.75 r_v", row=VOID_ROW)
_OVERLAPS = Opt("keep_overlaps", "keep overlapping voids", "bool", False,
                help="also the voids whose centre lies inside the region of a deeper void", row=VOID_ROW)

CROSSBOX_SIMS = "TNG50-3-Dark TNG100-3-Dark TNG300-3-Dark"     # the boxes the launcher's cross-box figure compares

# the void scripts' own options, their defaults the scripts' (the runspec test compares them): only a
# changed value adds its flag, so the exported command shows exactly what runs
RV_DEFINITIONS = ("crossing", "delta08", "ridge")      # dtfelib.catalog.R_V_DEFINITIONS (not imported: launch time)
SIZE_SMOOTH_CELLS = 5.0         # plot_void_population.SIZE_SMOOTH_CELLS
SIZE_FUNCTION_RV = "delta08"    # plot_void_population.SIZE_FUNCTION_RV
FIXED_R_SMOOTH_CELLS = "10 5 3"     # plot_void_population.FIXED_R_SMOOTH_CELLS (the size function at fixed R)
FATE_REF_SNAP = 33             # plot_void_fate.REF_SNAP (z = 2)
CONV_FACTORS = "2 4"            # dtfelib.convergence.DEFAULT_FACTORS
CONV_MATCH_SIGMA = 1.0          # dtfelib.convergence.DEFAULT_MATCH_SIGMA
CONV_RV_RANGE = "2.5 5"         # plot_void_convergence.DEFAULT_RV_RANGE
ENV_FACTOR = 2.0                # dtfelib.classify.ENV_FACTOR
# plot_void_environment.py --criterion: dtfelib.classify.CRITERIA as the script spells them (density_raw inside, in
# the npz keys and figure tags; density-raw on the command line, P6.7), the first the default. A saved value with '_'
# reads as the '-' one (PlotSpec.from_dict)
ENV_CRITERIA = ("velocity", "density-smoothed", "density-raw")
SKELETON_NULL_DRAWS = 50        # plot_caustic_skeleton.NULL_DRAWS
_RV_HELP = ("crossing: where the smoothed density first rises back to the mean; delta08: where the enclosed contrast "
            "reaches -0.8; ridge: the compensation ridge")
_POP_OPTS = (
    Opt("size_rv", "R_v", "choice", SIZE_FUNCTION_RV, (SIZE_FUNCTION_RV,) + tuple(d for d in RV_DEFINITIONS
                                                                                   if d != SIZE_FUNCTION_RV)
        + ("delta08raw",),
        help="the void radius of the size function figure: " + _RV_HELP + " (the default, the theory's radius; "
             "crossing while a catalogue lacks it); delta08raw: the same -0.8 radius on the unsmoothed density "
             "(2026-10-11)", row="Size function"),
    Opt("size_smooth", "smoothing (cells)", "float", SIZE_SMOOTH_CELLS,
        help=f"the size function's own catalogue smoothing in grid cells (default {SIZE_SMOOTH_CELLS:g}: at the "
             "production 10 cells the delta08 radius is mostly below the smoothing and undefined); 0 = the "
             "catalogue of the other figures", limits=(0.0, 100.0, 2), row="Size function"),
    # the size function at fixed R: a job step of its own (the script's flag makes ONLY that figure),
    # run with DTFE_VOID_GATE=none in its environment: it is defined on the ungated voids and refuses the gate
    Opt("size_fixed_r", "size function at fixed R", "bool", False,
        help="also the size function at fixed radius: the raw -0.8 radius of the distinct voids from catalogues at "
             "the smoothings beside, against the Vdn model over the radii where every smoothing is complete, at "
             "z = 0, 0.5, 1 and 2 (void_population/<sim>/gatenone). Its own step, run UNGATED (DTFE_VOID_GATE=none, "
             "shown in the command): the ellipsoid gate cuts the large voids differently at each smoothing",
        row="Fixed R"),
    Opt("fixed_r_smooth", "cells", "text", FIXED_R_SMOOTH_CELLS,
        help=f"the fixed-R catalogues' smoothings in grid cells, space-separated (default {FIXED_R_SMOOTH_CELLS}); "
             "together with the Mpc ones the run uses exactly the smoothings in both boxes (clear this one for the "
             "Mpc ones alone)", row="Fixed R"),
    Opt("fixed_r_smooth_mpc", "Mpc", "text", "",
        help=f"fixed-R smoothings in Mpc, the same scale in every box (e.g. {CROSSBOX_SMOOTH_MPC:g}, the cross-box "
             "smoothing); TNG300-3-Dark has no common range at 10 5 3 cells: use "
             f"{CROSSBOX_SMOOTH_MPC:g} Mpc with 3 2 cells there", row="Fixed R"))
_TRACK_OPTS = (
    Opt("link", "link", "choice", "trees", ("trees", "particles"),
        help="how a void is followed back in time: trees, along the merger-tree tracers; particles, by its member "
             "particles (the earlier void that gave it most of them; reads the combined snapshots once, the links "
             "are then cached)", row="Followed by"),
    Opt("exclusive", "each particle in one void", "bool", False,
        help="with the particle link only: a particle inside several voids' spheres belongs to the one whose centre "
             "is nearest in units of its R_v (the fate figure's partitioned links); at z of 1 and more the spheres "
             "overlap", row="Followed by"))
_FATE_OPTS = (
    Opt("ref_snap", "direct link from snap", "int", FATE_REF_SNAP,
        help=f"also link this snapshot directly to the last one (default {FATE_REF_SNAP}, z = 2); one not among the "
             "linked snapshots adds no direct link", row="Links"),
    Opt("exclusive", "each particle in one void", "bool", False,
        help="a particle inside several voids' spheres goes to the one whose centre is nearest in units of its R_v "
             "(a partition of the matter: overlapping high-z voids cannot both pass half their matter to one "
             "descendant); its files are tagged '_exclusive'", row="Links"))
_CONV_OPTS = (
    Opt("factors", "grid factors", "text", CONV_FACTORS,
        help="block-averaging factors of the grid, space-separated whole numbers (default 2 4: 512 to 256 and 128 "
             "cells)", row="Convergence"),
    Opt("match_sigma", "match (σ)", "float", CONV_MATCH_SIGMA,
        help=f"the radius two voids match within, in smoothing lengths (default {CONV_MATCH_SIGMA:g})",
        limits=(0.0, 100.0, 2), row="Convergence"),
    Opt("rv_range", "stack R_v (σ)", "text", CONV_RV_RANGE,
        help="the stacks' R_v range in smoothing lengths, two numbers lo hi (default 2.5 5)", row="Convergence"))
_ENV_OPTS = (
    Opt("criterion", "criterion", "choice", ENV_CRITERIA[0], ENV_CRITERIA,
        help="what makes a void void-in-cloud: velocity (default): net infall of the shell between 2 and 3 R_v; "
             "density-smoothed: the smoothed density enclosed within 3 R_v above the mean; density-raw: the raw "
             "density enclosed within the factor below (2026-10-11)", row="Void type"),
    Opt("factor", "within (R_v)", "float", ENV_FACTOR,
        help=f"density-raw only: its raw density contrast is taken within this many R_v (default {ENV_FACTOR:g}, "
             f"Ceccarelli et al. 2013); the default criterion is the infall of the shell between 2 and 3 R_v, "
             f"which this does not change", limits=(0.0, 100.0, 2), row="Void type"),
    Opt("r_v_definition", "R_v", "choice", "crossing", RV_DEFINITIONS,
        help="which void radius: " + _RV_HELP + "; a catalogue without it is rebuilt by the pipeline", row="Void type"))
_EVOL_OPTS = (
    Opt("rv_definition", "Void radius", "choice", "crossing", RV_DEFINITIONS,
        help="the void radius the profiles are scaled by: " + _RV_HELP + "; one the catalogue lacks falls back to "
             "crossing"),)
_SKEL_OPTS = (
    Opt("null_draws", "null draws", "int", SKELETON_NULL_DRAWS,
        help="per-void statistics: realisations of the random-sphere null (one sphere per void at a random cell, "
             f"with its radius), the voids' medians as z-scores against it (default {SKELETON_NULL_DRAWS}; 0 = no "
             "null; another count tags the set)", row="Per void"),
    Opt("sheet_min_density", "walls denser than", "float", 0.0,
        help="per-void statistics: count only the sheet cells denser than this many times the mean (default 0: "
             "every one); the single-fold sheet percolates through the voids, its dense part is the walls",
        limits=DENSITY_LIMITS, row="Per void"),
    Opt("min_component", "min component (cells)", "int", 0,
        help="per-void statistics: the shells and the wall distance on sheet components of at least this many "
             "cells only (default 0: every sheet cell)", row="Sheet components"),
    Opt("min_component_mass", "and mass fraction", "float", 0.0,
        help="per-void statistics: and of at least this fraction of the box's mass (default 0)",
        limits=(0.0, 1.0, 5), row="Sheet components"),
    Opt("all_caustic", "Every snapshot with caustic grids (instead of the selected ones)", "bool", False,
        help="every snapshot of the simulation that has a caustic class grid under this prefix; the snapshot "
             "selection is then not used"))

FIGURE_SETS: tuple[FigureSet, ...] = (
    FigureSet("fields", "Field slice maps", "plot/plot_PS_DTFE.py", "snapshot",
              "Slices through every field grid of a snapshot: density, streams, velocity, "
              "dispersion and web eigenvalues, plus the point-evaluated maps when a hi-res "
              "slice exists.", "fields/<sim>/snapNNN",
              opts=(Opt("pointeval_only", "Only the point-evaluated maps", "bool",
                        False, help="Skips every grid load"), _RAW)),
    FigureSet("web", "Cosmic-web classification", "plot/plot_cosmic_web.py", "snapshot",
              "T-web and V-web classes and eigenvalue maps. Needs the T-web / V-web fields.",
              "cosmic_web", opts=(_RAW,)),
    FigureSet("series", "Standard-DTFE field series", "plot/plot_DTFE.py", "series",
              "Density, velocity, divergence and shear slices across the selected snapshots "
              "(standard DTFE unless the estimator says otherwise).", "dtfe"),
    FigureSet("pointeval", "Point-evaluated hi-res maps", "plot/plot_pointeval.py", "series",
              "Every point-evaluated field of the hi-res slices (density, streams, speed, "
              "dispersion and, with the velocity gradient, divergence, shear and vorticity) as a "
              "zero-thickness cross-section. Skips figures newer than their data.",
              "fields/<sim>/snapNNN/<plane>", common=False,
              opts=(_PLANE,
                    Opt("fields", "Fields", "text", "", help="the maps to draw, separated by spaces or commas; blank = all available"),
                    Opt("project", "Projection", "choice", "plane", ("plane", "slab"),
                        help="'slab' averages every plane in the file"),
                    Opt("smooth_px", "Smoothing (output pixels)", "float", 0.0),
                    Opt("smooth_deriv", "Gradient maps also smoothed at (output pixels)", "float", POINTEVAL_DERIV_SMOOTH,
                        help="the smoothed variant of the divergence, shear and vorticity maps (0 = none)"),
                    Opt("fixed_range", "Fixed density range of the grid maps", "bool", False),
                    Opt("force", "Re-render up-to-date figures", "bool", False))),
    FigureSet("panels", "Publication panels", "plot/plot_pointeval_panels.py", "snapshot",
              "Projected density, the inverted 'sheet' view and the pointwise stream count of a "
              "hi-res slice, as one publication-resolution figure per snapshot.",
              "pointeval_panels/<sim>", common=False,
              opts=(_PLANE,
                    Opt("panels", "Panels", "text", "density,sheet,streams"),
                    Opt("project", "Projection", "choice", "plane", ("plane", "slab"),
                        help="'slab' averages every plane in the file (a saved 'mean', its older name, reads as "
                             "'slab')"),
                    Opt("vmin", "Density scale minimum (0 = auto)", "float", 0.0,
                        help="rho/rho_bar; auto = the 0.1th percentile, which adapts to each snapshot's contrast",
                        limits=DENSITY_LIMITS),
                    Opt("vmax", "Density scale maximum (0 = auto)", "float", 0.0, help="rho/rho_bar; auto = the 99.99th percentile",
                        limits=DENSITY_LIMITS),
                    Opt("dpi", "Output dpi (0 = native)", "int", 0, help="0 = native: one image pixel per sample"),
                    Opt("image", "Pixel-exact PNGs, one per panel (no axes; for zooming and posters)", "bool", False))),
    FigureSet("voidpop", "Void population", "plot/plot_void_population.py", "sim",
              "Counts, sizes, shapes and depths of all voids at every snapshot with grids.",
              "void_population/<sim>",
              opts=(Opt("panel_selected", "Distribution panels at the selected snapshots", "bool", False,
                        help="default: config.PANEL_SNAPSHOTS"), _OVERLAPS, _SMOOTH_MPC) + _POP_OPTS),
    FigureSet("voidtrack", "Void tracking", "plot/plot_void_tracking.py", "sim",
              "Size, BBKS shape and orientation of the deepest voids followed back in time "
              "through the merger trees.", "void_tracking/<sim>",
              opts=(_REF, Opt("n_voids", "Voids", "int", 5), _RADIUS, _SMOOTH_MPC) + _TRACK_OPTS),
    # the cross-box comparison: several boxes at ONE smoothing in Mpc; it takes the boxes itself
    # (--sims), not the selected simulation
    FigureSet("crossbox", "Void statistics across boxes", "plot/plot_void_crossbox.py", "sim",
              "Resolved fraction, distinct-void density, R_v and stacked profiles of several boxes side by side at "
              "one smoothing length in Mpc. Builds each box's void catalogues first (a minute and about 10 GB "
              "per 512 cubed snapshot) unless only redrawn.", "void_crossbox", common=False,
              opts=(Opt("sims", "Boxes", "text", CROSSBOX_SIMS, help="space-separated simulation names"),
                    Opt("smooth_mpc", "Common smoothing in Mpc", "float", CROSSBOX_SMOOTH_MPC,
                        help=f"{CROSSBOX_SMOOTH_MPC:g} = config.CROSSBOX_SMOOTHING_MPC (2.1625 = 10.0002 TNG100-3-Dark "
                             "cells)",
                        limits=(0.0, 100.0, 4)),
                    Opt("z", "Redshifts of the bottom row", "text", "0 0.5 1", help="space-separated, one figure each"),
                    Opt("n_boot", "Bootstrap resamples (0 = the script's default)", "int", 0),
                    Opt("replot", "Only redraw from the saved results", "bool", False))),
    # the void science figures (contract C5): registered by name with their default
    # arguments (--sim and the common grid options)
    FigureSet("voidevol", "Void profile evolution", "plot/plot_void_profile_evolution.py", "sim",
              "The stacked void density profiles across redshift, with the Hamaus, Sutter and Wandelt (2014) fit.",
              "void_profile_evolution/<sim>", opts=_EVOL_OPTS),
    FigureSet("voidenv", "Void environment", "plot/plot_void_environment.py", "sim",
              "Voids in clouds against voids in voids: by default the infall of the shell between 2 and 3 void "
              "radii (the density within two radii, Ceccarelli et al. 2013, as a choice), and how the two kinds "
              "evolve.", "void_environment/<sim>", opts=_ENV_OPTS),
    FigureSet("voidfate", "Void fate", "plot/plot_void_fate.py", "sim",
              "What becomes of each void between snapshots, linked by its member particles: survives, merges, "
              "splits or vanishes.", "void_fate/<sim>", opts=_FATE_OPTS),
    # the convergence check (contract K4): registered by name, default arguments '--sim SIM' (like
    # every void script it needs DTFE_SIM, which _plot_env sets) plus the common grid options when set
    FigureSet("voidconv", "Void convergence", "plot/plot_void_convergence.py", "sim",
              "Resolution and estimator convergence of the void statistics: abundance, R_v and stacked profiles "
              "at different grid resolutions, and PS-DTFE against standard DTFE on the same voids.",
              "void_convergence/<sim>", opts=_CONV_OPTS),
    # the catalogue itself, for the Runs tab's 'Build void catalogue': the pipeline's cached
    # catalogue of the selected snapshots' grids, as the Explore outlines and the void figures read it
    FigureSet("voidcat", "Void catalogue build", "gui/void_cache.py", "snapshot",
              "Builds the void catalogue of each selected snapshot's grids into the analysis cache (about a minute "
              "and 10 GB at 512 cubed; refused when the memory is not free), so the void figures and the Explore "
              "outlines need not. The estimator and grid prefix are this tab's.", "(no figures)", common=False),
    FigureSet("halotrack", "Halo tracking", "plot/plot_halo_tracking.py", "sim",
              "Mass, shape, orientation, mergers and field environment of subhalos along their "
              "main progenitor branch. Needs the merger trees (shapes also the group catalogues).",
              "halo_tracking/<sim>",
              opts=(_REF, Opt("top", "Most massive centrals", "int", 5),
                    Opt("subhalos", "Subhalo IDs", "text", "",
                        help="SubfindIDs at the reference snapshot, space-separated (replaces the top N)"),
                    Opt("environment", "Field environment", "bool", True),
                    Opt("min_ratio", "Merger ratio to mark", "float", 0.1))),
    FigureSet("voidprof", "Void profiles", "plot/plot_void_profiles.py", "series",
              "Stacked density, radial velocity and single-stream fraction around the voids, in units of each "
              "void's radius, per size bin, at the selected snapshots.", "void_profiles/<sim>",
              opts=(_RADIUS, Opt("sample", "Voids", "choice", "resolved", ("resolved", "deep"),
                                 help="deep: only the voids whose centre is below the deep-void threshold"),
                    _OVERLAPS, _SMOOTH_MPC)),
    FigureSet("webstreams", "Streams by web environment", "plot/plot_web_streams.py", "series",
              "Volume and mass fractions of every T-web and V-web class by stream multiplicity, per snapshot "
              "and across redshift.", "web_streams/<sim>",
              opts=(Opt("web", "Web", "choice", "both", ("both", "tweb", "vweb")),)),
    FigureSet("pk", "Power spectrum", "plot/plot_pk_compare.py", "series",
              "P(k) of the density grids against linear theory, with the cosmic-variance band, and divided by "
              "the same box's early spectrum grown linearly (its random fluctuations cancel there).",
              "power_spectrum/<sim>",
              opts=(Opt("nbins", "k bins (0 = one per fundamental mode)", "int", 40),
                    Opt("ref", "Same-box panel (from the first snapshot)", "bool", True))),
    FigureSet("skeleton", "Caustic skeleton", "plot/plot_caustic_skeleton.py", "series",
              "Walls, filaments and nodes from the collapse classes of a run with caustics on, and per void the "
              "fold fraction around its wall and the distance to the nearest wall.", "caustic_skeleton/<sim>",
              opts=(Opt("voids", "Per-void statistics", "bool", True), _RADIUS, _OVERLAPS, _SMOOTH_MPC) + _SKEL_OPTS),
    FigureSet("core", "Core analysis set", "analyze.py", "series",
              "The core analysis figures (eigenvalues, void shapes, correlations, ...) through "
              "analyze.py: 'compute' fills python/cache from the standard-DTFE grids with the "
              "smoothing of python/config.py, 'plot' draws from that cache, 'export' writes the void "
              "catalogues (CSV + HDF5, under void_catalog/<sim>).", "<analysis>/",
              common=False,
              opts=(Opt("mode", "Mode", "choice", "all", ("all", "compute", "plot", "check", "export")),
                    Opt("only", "Only these scripts", "multi", "", CORE_SCRIPTS,
                        help="tick some to run only those; none ticked = the full core set"))),
)
FIGURE_SET = {fs.key: fs for fs in FIGURE_SETS}


# ---------------------------------------------------------------------- download and merge
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_snapshots.sh"
MERGE_TOOL = REPO_ROOT / "python" / "tools" / "merge_HDF5.py"
API_KEY_FILE = Path.home() / ".tng_api_key"     # download_snapshots.sh reads it (or TNG_API_KEY)

# DM particles per side of every TNG run (the -Dark twin has the same DM count). A raw -Dark
# snapshot is ~61 bytes per particle (measured: TNG50-3/100-3/300-3, 4 chunks each),
# the merged combined_NNN.hdf5 32 (float32 x3 + uint64 + float32 x3).
TNG_DM_PER_SIDE = {"TNG50-1": 2160, "TNG50-2": 1080, "TNG50-3": 540, "TNG50-4": 270,
                   "TNG100-1": 1820, "TNG100-2": 910, "TNG100-3": 455,
                   "TNG300-1": 2500, "TNG300-2": 1250, "TNG300-3": 625}
TNG_SIMULATIONS = [f"{run}{suffix}" for run in TNG_DM_PER_SIDE for suffix in ("-Dark", "")]
RAW_BYTES_PER_PARTICLE, MERGED_BYTES_PER_PARTICLE = 61, 32


def _snapshot_ladder() -> dict[int, float]:
    """python/config.py's SNAPSHOT_TO_REDSHIFT, read as data (importing config has side effects)."""
    import ast
    try:
        tree = ast.parse((REPO_ROOT / "python" / "config.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SNAPSHOT_TO_REDSHIFT"
                                                   for t in node.targets):
                return {int(k): float(v) for k, v in ast.literal_eval(node.value).items()}
    except (OSError, SyntaxError, ValueError):
        pass
    return {}


SNAPSHOT_LADDER = _snapshot_ladder()


def dm_particles(sim: str) -> int | None:
    side = TNG_DM_PER_SIDE.get(sim.removesuffix("-Dark"))
    return side ** 3 if side else None


# ---------------------------------------------------------------------- your own snapshot file
CUSTOM_OUT_DEFAULT = Path.home() / "DTFE-output"
DEMO_GENERATOR = REPO_ROOT / "tests" / "generate_ps_test_data.py"
# --input types the binary reads (input_output.cc); only Gadget HDF5 carries the particle IDs and
# initial positions a phase-space run needs
INPUT_TYPES = {105: "Gadget HDF5 (.hdf5)", 101: "Gadget binary (type 1/2)",
               111: "Text: x y z mass per line", 112: "Text: x y z per line"}


# the Pipeline, Plots, Data and Custom jobs (each module imports this one: they come in after every name their
# class bodies read, and before JOB_KINDS names them)
from runspec_pipeline import PipelineSpec, _window_values  # noqa: E402,F401  (re-exported)
from runspec_plot import PlotSpec  # noqa: E402,F401
from runspec_data import DataSpec, api_key_available, data_snapshots, sim_state, snapshot_state  # noqa: E402,F401
from runspec_custom import (CustomSpec, demo_spec, hdf5_boxsize, hdf5_has,  # noqa: E402,F401
                            hdf5_particle_count, hdf5_redshift, snapshot_dim)


# ---------------------------------------------------------------------- job queue
JOB_KINDS = {"data": DataSpec, "grids": RunSpec, "pipeline": PipelineSpec, "plots": PlotSpec,
             "custom": CustomSpec}
QUEUE_STATES = ("waiting", "running", "done", "failed", "stopped", "skipped")
# jobs that hold a triangulation and full grids: never started next to another DTFE run
HEAVY_KINDS = ("grids", "pipeline", "custom")


def _snaps_text(snaps) -> str:
    snaps = sorted(set(snaps))
    return " ".join(f"{n:03d}" for n in snaps[:8]) + (f" (+{len(snaps) - 8})" if len(snaps) > 8 else "")


def job_summary(kind: str, spec) -> str:
    """One line naming what a job does, for the queue list: a run's quality in the launcher's words (Exact or Fast;
    an unknown value reads as Exact, as quality_mode reads it) and its machine use when it is not balanced, the
    default."""
    mode = getattr(spec, "run_mode", "balanced")
    mode = "" if mode == "balanced" else f", {mode}"
    quality = (f", {QUALITY_NAME.get(getattr(spec, 'quality', ''), QUALITY_NAME[QUALITY_DEFAULT])}"
               if kind in ("custom", "grids", "pipeline") else "")
    if kind == "custom":
        est = "PS-DTFE" if spec.estimator == "ps" else "DTFE"
        cells = "²" if getattr(spec, "dim", 3) == 2 else "³"
        return (f"Custom snapshot · {Path(spec.input_file).name} · {est} {spec.grid}{cells}{quality}{mode} -> "
                f"{spec.output_root()}")
    if kind == "grids":
        est = "PS-DTFE" if spec.estimator == "ps" else "DTFE"
        return f"Grids · {spec.sim} · {_snaps_text(spec.snapshots)} · {est} {spec.grid}³{quality}{mode}"
    if kind == "pipeline":
        return (f"Pipeline · {', '.join(spec.sims)} · plane {spec.nu}² · grid {spec.grid}³" + quality + mode
                + (" · plan only" if spec.plan_only else " · + figures" if spec.render else ""))
    if kind == "plots":
        names = [FIGURE_SET[k].title for k in spec.sets if k in FIGURE_SET]
        return f"Plots · {spec.sim} · {_snaps_text(spec.snapshots)} · {', '.join(names)}"
    parts = [w for w, on in (("download", spec._downloads()),
                             ("merge", spec.merge_snapshots or spec.merge_groupcats or spec.merge_trees
                              or spec.convert_ics)) if on]
    return f"Data · {spec.sim} · {_snaps_text(spec.snapshots)} · {' + '.join(parts) or 'nothing'}"


@dataclass
class QueuedJob:
    """A job frozen when it was queued: its settings (not its commands -- those are worked out
    when it starts, so a Data job fetches only what is still missing then, and a job queued after
    a download sees the new files). state: waiting | running | done | failed | stopped | skipped."""
    kind: str
    spec: dict
    title: str = ""
    state: str = "waiting"
    note: str = ""

    @classmethod
    def of(cls, kind: str, spec) -> "QueuedJob":
        return cls(kind, spec.to_dict(), job_summary(kind, spec))

    def build(self):
        return JOB_KINDS[self.kind].from_dict(self.spec)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QueuedJob":
        """A saved queue entry; one queued before the two modes has its settings brought into them (migrate_quality:
        the deposit becomes the quality, the removed settings are dropped) and its title made again from them (the
        old one named the old settings: 'exact deposit', 'GPU')."""
        job = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if job.kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {job.kind!r}")
        if isinstance(job.spec, dict):
            cls = JOB_KINDS[job.kind]
            old = predates_modes(job.spec)
            job.spec = {k: v for k, v in migrate_quality(job.spec).items() if k in cls.__dataclass_fields__}
            if "advanced" in job.spec:      # the overrides it was queued with, the bad ones dropped (none: no key)
                adv = clean_advanced(job.spec.pop("advanced"), cls)
                if adv:
                    job.spec["advanced"] = adv
            if old:
                try:
                    job.title = job_summary(job.kind, job.build())
                except (TypeError, ValueError):
                    pass                    # (a job that does not build keeps its title; starting it says why)
        if job.state == "running":          # the window closed during it: it was stopped
            job.state, job.note = "stopped", "interrupted when the window closed"
        return job


def next_waiting(queue: list[QueuedJob]) -> int | None:
    return next((i for i, j in enumerate(queue) if j.state == "waiting"), None)


# ---------------------------------------------------------------------- discovery helpers
def simulations(root) -> list[str]:
    """The simulations with merged snapshots under the data root 'root'; none for an empty root."""
    return find_sims(Path(root).expanduser(), "snapdir_*/combined_*.hdf5") if root_ok(root) else []


def snapshots(sim_path: Path, prefix: str = "ps_output", estimator: str = "ps", done=None) -> list[dict]:
    """Snapshots with a merged file: number, redshift (from the header) and whether outputs exist
    (done(snapdir) -> bool overrides the grid test, e.g. for point-evaluated slices)."""
    out = []
    for f in sorted(Path(sim_path).glob("snapdir_*/combined_[0-9][0-9][0-9].hdf5")):
        n = int(f.stem.split("_")[1])
        out.append({"n": n, "z": _redshift(f),
                    "done": done(f.parent) if done else has_outputs(f.parent, prefix, estimator)})
    return sorted(out, key=lambda s: s["n"])


def has_outputs(snapdir: Path, prefix: str, estimator: str) -> bool:
    """A density grid of the run: PS-DTFE '<prefix>.a_den' (or the unaveraged '.den'); standard DTFE 'output.*',
    what run_dtfe.sh writes for a Grids job (it passes no prefix) and dtfelib's FieldSet reads (not
    'output*.a_den', which matches every 'output_<variant>' beside it too -- 'output_plane.a_den', a plane run's
    one-field grid, would mark a snapshot without the grids done)."""
    if estimator == "ps":
        return (snapdir / f"{prefix}.a_den").exists() or (snapdir / f"{prefix}.den").exists()
    return ((snapdir / "output.a_den").exists() or (snapdir / "output.den").exists()
            or any((snapdir / "output").glob("*.a_den")))


def has_grids(snapdir: Path, method: str, prefix: str) -> bool:
    """Would a figure script's FieldSet(method=..., prefix=...) find grids here?"""
    ps = has_outputs(snapdir, prefix or "ps_output", "ps")
    if method == "ps" or prefix:
        return ps
    dtfe = has_outputs(snapdir, "", "dtfe")
    return dtfe if method == "dtfe" else (ps or dtfe)


def plane_sidecars(sim_path) -> list[Path]:
    """Image-plane sidecars (.json) in dtfelib.pointeval's discovery order."""
    sp = Path(sim_path)
    return sorted(sp.glob("pointeval_plane_*.json")) + sorted(sp.glob("hires_plane_*.json"))


IMAGE_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg", ".svg")


def figures(roots, since: float | None = None, limit: int = 2000) -> list[Path]:
    """Figure files under roots, newest first; since = only those modified at/after this time."""
    found: dict[Path, float] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(IMAGE_SUFFIXES):
                    path = Path(dirpath) / f
                    try:
                        mt = path.stat().st_mtime
                    except OSError:
                        continue
                    if since is None or mt >= since:
                        found[path.resolve()] = mt
    return sorted(found, key=lambda p: found[p], reverse=True)[:limit]


_Z_CACHE: dict[str, float | None] = {}


def _redshift(path: Path) -> float | None:
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _Z_CACHE:
        try:
            import h5py
            with h5py.File(path, "r") as h:
                _Z_CACHE[key] = float(h["Header"].attrs["Redshift"])
        except Exception:
            _Z_CACHE[key] = None
    return _Z_CACHE[key]


def planes(sim_path: Path) -> list[Path]:
    """--sample-points files next to the snapshots (make_image_plane.py output)."""
    return sorted(p for p in Path(sim_path).glob("*plane*.bin") if p.is_file())


def plane_points(path) -> int:
    """Points in a raw float64 x3 --sample-points file (0 if unreadable)."""
    try:
        return Path(path).stat().st_size // 24
    except OSError:
        return 0


def _objdir(estimator: str, precision: str = "single", dim: int = 3) -> Path:
    """The Makefile's object directory of a pair: o / o_ps, o_d / o_ps_d for the double pair, and
    '_2d' before the '_d' for the 2D set (o_2d, o_ps_2d_d)."""
    return REPO_ROOT / (("o_ps" if estimator == "ps" else "o") + ("_2d" if dim == 2 else "")
                        + ("_d" if precision == "double" else ""))


def gpu_built(estimator: str, precision: str = "single", dim: int = 3) -> bool:
    if dim == 2:                    # the GPU kernels deposit tetrahedra into 3D cells: a 2D build is CPU-only
        return False
    objdir = _objdir(estimator, precision)
    return objdir.is_dir() and not (objdir / ".gpu_mode_off").exists()


# the make argument that gives this machine a GPU build, for messages
GPU_BUILD_HINT = "METAL=1 on a Mac"
GPU_BACKENDS = {"metal": "Metal"}


def gpu_backend(estimator: str, precision: str = "single", dim: int = 3) -> str:
    """The GPU backend a pair was built with ('Metal'), from the Makefile's
    .gpu_mode_<mode> stamp; '' for a CPU-only build (every 2D one) or an unbuilt pair."""
    if dim == 2:
        return ""
    objdir = _objdir(estimator, precision)
    for stamp, name in GPU_BACKENDS.items():
        if (objdir / f".gpu_mode_{stamp}").exists():
            return name
    return ""


def gpu_build_default(estimator: str = "ps", precision: str = "single", dim: int = 3) -> bool:
    """May a job's quality put it on the GPU (quality_mode)? Only when the pair was built with a GPU backend --
    scripts/config.sh gpu_build_default's rule, the drivers' default for DTFE_GPU: the Makefile's '.gpu_mode_metal'
    stamp in the pair's object folder (gpu_backend). The binary itself cannot say ('--help' lists --gpu in every
    build, '--version' names no backend)."""
    return bool(gpu_backend(estimator, precision, dim))


def gpu_compiler() -> str:
    """The GPU build this machine offers: 'METAL=1' on a Mac (the Metal host needs only the system SDK);
    '' elsewhere -- the GPU code is Metal-only, so the build is then CPU-only."""
    return "METAL=1" if sys.platform == "darwin" else ""


def tbb_built(precision: str = "single", dim: int = 3) -> bool:
    """PS-DTFE was built with the TBB parallel insertion (o_ps/.tbb_1, the Makefile's stamp). The
    parallel insertion is 3D only (CGAL's parallel Delaunay): a 2D build inserts sequentially."""
    return dim == 3 and (_objdir("ps", precision) / ".tbb_1").exists()


# The device of each quality, as the drivers' automatic device picks it (scripts/config.sh gpu_auto_resolve; the
# binaries do not pick one yet). The sampled PS-DTFE deposit (Fast) pays on the GPU only from about a million
# particles (GPU_PAYS_FROM: below it the CPU is faster); a run that writes no grid (a slice alone) deposits nothing.
# PS-DTFE's exact deposit (Exact) runs on the GPU in every machine use, which leaves the cores to the machine in
# gentle and balanced; in performance '--gpu-share auto' (the jobs' GPU_SHARE) deposits part of a partitioned run on
# the free cores (the Advanced Device's GPU keeps it on the GPU, '1', and its CPU + GPU shares it in every machine
# use, 'mixed': apply_advanced). Standard DTFE's exact cell average (Exact) is a tie between the two (the
# triangulation and the I/O dominate), so the GPU; its sampled average (Fast) is faster on every CPU core, so the CPU
# in performance and the GPU in gentle and balanced (their lowered thread budget). The measurements: the drivers'
# bench notes, not repeated here.
GPU_PAYS_FROM = 1_000_000


def recommended_gpu(deposit: str, fields, n_particles, estimator: str = "ps", precision: str = "single",
                    dim: int = 3, run_mode: str = "balanced") -> bool:
    """The device a job of this cell average runs on in the machine use 'run_mode' (the GPU built; quality_mode's):
    the drivers' automatic device (scripts/config.sh gpu_auto_resolve; the note above GPU_PAYS_FROM says why).
    PS-DTFE's exact deposit: the GPU in every machine use (in performance the drivers' '--gpu-share auto' adds the
    free cores). Standard DTFE: the GPU, except its sampled average in performance. PS-DTFE's sampled deposit: the GPU
    from GPU_PAYS_FROM particles. A double build hands the GPU float copies of its tetrahedra: the deposit's sums are
    single precision, everything else double. A 2D job: never (the 2D programs are CPU-only)."""
    if not gpu_built(estimator, precision, dim):
        return False
    if estimator != "ps":
        return not (deposit == "sampled" and run_mode == "performance")
    if deposit == "exact":
        return True
    return bool(fields) and (n_particles is None or n_particles >= GPU_PAYS_FROM)


_JOBS_CACHE: tuple[float, dict] = (-1e9, {"runs": [], "servers": []})
_SERVE = re.compile(r"(^|\s)--serve(\s|$)")     # the '--serve' token itself, not --serve-resident or a path
_REPORT_RUN = re.compile(r"(^|\s)--auto-tune-report(\s|$)")   # a memory check: the tuner's prediction, no run
# the Explore tab's exact zoom writes into tempfile.mkdtemp(prefix=ZOOM_DIR_PREFIX) / "zoom": its output root
# names it, whatever else its command shares with a Custom-tab run (both may pass --ps-window)
ZOOM_DIR_PREFIX = "dtfe-zoom-"
_ZOOM_ROOT = re.compile(r"/" + re.escape(ZOOM_DIR_PREFIX) + r"[^/\s]*/zoom(\s|$)")


def dtfe_processes(max_age: float = 2.0) -> dict:
    """The DTFE / PS-DTFE binaries alive on this machine (from any terminal), classified once per max_age
    seconds (pgrep costs ~30 ms and the checks run on every click):
      'runs'     [(pid, name, gb)]  jobs that compute and occupy the machine; gb = their resident memory
                                    (None when ps could not say)
      'servers'  [(pid, name, gb)]  query servers ('--serve': resident, idle between requests, but their
                                    tessellations are real memory a large run has to live beside)
    Not listed at all: report-mode runs ('--auto-tune-report', a memory check from any tab or terminal) and
    this process's own exact zoom (a direct child writing into a ZOOM_DIR_PREFIX folder; its Custom-tab runs DO
    count). Counted as runs, the servers would make every
    tab warn while Explore's server is up and a queue with 'wait' ticked wait forever; the jobs the
    launcher runs through a script are grandchildren (bash -> binary) and do count."""
    global _JOBS_CACHE
    now = time.monotonic()
    if now - _JOBS_CACHE[0] > max_age:
        try:
            # every set: PS-DTFE, DTFE-double, PS-DTFE-2d-double (Linux truncates the name to 15 characters)
            lines = subprocess.run(["pgrep", "-l", "^(PS-)?DTFE(-.*)?$"], capture_output=True, text=True,
                                   timeout=5).stdout.splitlines()
        except (OSError, subprocess.TimeoutExpired):
            lines = []
        names = {int(pid): name for pid, name in (ln.split(None, 1) for ln in lines if " " in ln)}
        _JOBS_CACHE = (now, _classify(names))
    return _JOBS_CACHE[1]


def forget_processes():
    """Drop the cached classification (a check or job of our own just ended: its process must not linger)."""
    global _JOBS_CACHE
    _JOBS_CACHE = (-1e9, {"runs": [], "servers": []})


def running_jobs(max_age: float = 2.0) -> list[tuple[int, str]]:
    """(pid, name) of the DTFE / PS-DTFE binaries COMPUTING on this machine (dtfe_processes()['runs'])."""
    return [(pid, name) for pid, name, _gb in dtfe_processes(max_age)["runs"]]


def running_servers(max_age: float = 2.0) -> list[tuple[int, str, float | None]]:
    """(pid, name, resident GB) of the query servers alive on this machine (dtfe_processes()['servers'])."""
    return list(dtfe_processes(max_age)["servers"])


def _classify(names: dict[int, str]) -> dict:
    """One ps call lists pid, parent, resident set and the full command line (-ww: never truncated; the
    repository path has spaces, so the line is matched, not split); macOS and procps spell it the same. A
    pid gone meanwhile is simply absent; without ps every named binary counts as a run."""
    out = {"runs": [], "servers": []}
    if not names:
        return out
    try:
        text = subprocess.run(["ps", "-ww", "-o", "pid=,ppid=,rss=,args=", "-p", ",".join(map(str, names))],
                              capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        out["runs"] = [(pid, name, None) for pid, name in names.items()]
        return out
    me = os.getpid()
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3 or not all(v.isdigit() for v in parts[:3]):
            continue
        pid, ppid, rss_kb = int(parts[0]), int(parts[1]), int(parts[2])
        args = parts[3] if len(parts) > 3 else ""
        if pid not in names:
            continue
        kind = _kind(ppid, args, me)
        if kind in ("run", "server"):
            out["runs" if kind == "run" else "servers"].append((pid, names[pid], rss_kb / 1e6))
    return out


def _kind(ppid: int, args: str, me: int) -> str:
    """What a DTFE binary with this parent and command line is: 'run' (occupies the machine), 'server'
    (a query server: resident memory, idle CPU), 'report' (a memory check: the tuner's prediction only),
    'own' (a direct child of this launcher writing into an exact-zoom folder: its exact zoom).
    Report-mode runs are never counted (the memory check from any tab or terminal is not a run), servers are
    listed apart with their memory instead of being counted or hidden. A Custom-tab run is a direct child too
    (the binary itself, no script between), so only the zoom's output folder makes a child 'own'."""
    if _SERVE.search(args):              # a server is a server, the launcher's own included (its memory counts)
        return "server"
    if _REPORT_RUN.search(args):
        return "report"
    if ppid == me and _ZOOM_ROOT.search(args):
        return "own"
    return "run"


def _gb_text(gb) -> str:
    return "" if gb is None else f", {gb:.1f} GB"


# ---------------------------------------------------------------------- keep-awake
# the run scripts hold their own power assertion ('caffeinate -i -m -w $$'); run_ps_pipeline.sh runs
# run_ps_dtfe.sh, so it is covered too
SELF_CAFFEINATED = ("run_ps_dtfe.sh", "run_dtfe.sh", "run_ps_pipeline.sh")


def keep_awake_argv(step_argv, pid: int) -> list[str] | None:
    """The command that keeps the Mac awake while process 'pid' (a step the launcher started) lives, or None
    (not macOS, no caffeinate, or a run script that holds its own assertion). A Custom-tab run executes the
    binary directly, a long analysis or a multi-hour download from the Data tab has no script that
    caffeinates, and an idle-sleeping Mac suspends them (wget loses its connection)."""
    if sys.platform != "darwin" or not pid or any(Path(str(a)).name in SELF_CAFFEINATED for a in step_argv):
        return None
    exe = shutil.which("caffeinate")
    return [exe, "-i", "-m", "-w", str(int(pid))] if exe else None


# ---------------------------------------------------------------------- memory and runs started elsewhere
_SWAP = re.compile(r"total = ([\d.]+)([MG])\s+used = ([\d.]+)([MG])")


def memory_status() -> dict:
    """{'free_gb', 'swap_used_gb', 'swap_total_gb'} of this machine now (a value None when the OS will not say;
    the launcher's status strip). Free = what the other programs leave: DTFE_FREE_GB when set (as for
    the C++ programs and the void helper), else macOS vm_stat's free + inactive + speculative + purgeable pages
    (void_outlines.available_bytes' rule), else Linux MemAvailable. Swap: macOS sysctl vm.swapusage, Linux
    SwapTotal - SwapFree. Two short subprocess calls on a Mac: call it off the GUI thread."""
    out = {"free_gb": None, "swap_used_gb": None, "swap_total_gb": None}
    if os.environ.get("DTFE_FREE_GB"):
        try:
            out["free_gb"] = float(os.environ["DTFE_FREE_GB"])
        except ValueError:
            pass
    if sys.platform == "darwin":
        try:
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+) bytes", vm).group(1))
            pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages ([\w ]+):\s+(\d+)\.", vm, re.M)}
            if out["free_gb"] is None:
                out["free_gb"] = page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative",
                                                                        "purgeable")) / 1e9
        except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
            pass
        try:
            sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
            m = _SWAP.search(sw)
            if m:
                unit = {"M": 2**20 / 1e9, "G": 2**30 / 1e9}     # sysctl prints MiB / GiB as M / G
                out["swap_total_gb"] = float(m.group(1)) * unit[m.group(2)]
                out["swap_used_gb"] = float(m.group(3)) * unit[m.group(4)]
        except (OSError, subprocess.SubprocessError):
            pass
        return out
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = float(v.split()[0]) * 1024 / 1e9
        if out["free_gb"] is None and "MemAvailable" in info:
            out["free_gb"] = info["MemAvailable"]
        if "SwapTotal" in info:
            out["swap_total_gb"] = info["SwapTotal"]
            out["swap_used_gb"] = info["SwapTotal"] - info.get("SwapFree", 0.0)
    except (OSError, ValueError, IndexError):
        pass
    return out


def memory_text(mem: dict, procs: dict | None = None) -> str:
    """The status strip: 'free 31.2 GB · swap 1.4 of 4.0 GB · PS-DTFE pid 812, 37.1 GB'."""
    bits = []
    if mem.get("free_gb") is not None:
        bits.append(f"free {mem['free_gb']:.1f} GB")
    if mem.get("swap_used_gb") is not None:
        tot = mem.get("swap_total_gb")
        bits.append(f"swap {mem['swap_used_gb']:.1f}" + (f" of {tot:.1f} GB" if tot else " GB"))
    for key, what in (("runs", ""), ("servers", " server")):
        for pid, name, gb in (procs or {}).get(key, []):
            bits.append(f"{name}{what} pid {pid}{_gb_text(gb)}")
    return " · ".join(bits) or "memory: unknown"


# the binary's command line: '<exe> <input> <output root> [--options]'; the paths may hold spaces (an external
# disk's name, 'Mobile Documents'), so the input is matched up to its snapshot suffix and the output root up to the
# next option (the launcher's own Custom runs name the binary by its absolute path, which may hold spaces too)
_ARGV = re.compile(r"^(?P<exe>(?:.*?/)?(?:PS-)?DTFE(?:-[\w-]+)?)\s+(?P<input>.+?\.(?:hdf5|h5|hdf|bin|dat|txt|\d+))"
                   r"\s+(?P<output>.+?)(?=\s+--|\s*$)")
_SNAPDIR = re.compile(r"(?:^|/)snapdir_(\d{3})(?:/|$)")
_ETIME = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$")


def parse_run_args(args: str) -> dict:
    """What a DTFE / PS-DTFE command line says about its run: {'exe', 'input', 'output', 'sim', 'snap'}
    ('' / None where it does not say). sim = the folder above 'snapdir_NNN' (the TNG layout)."""
    m = _ARGV.match(args.strip())
    d = {"exe": "", "input": "", "output": "", "sim": "", "snap": None}
    if not m:
        return d
    d.update(exe=Path(m.group("exe")).name, input=m.group("input"), output=m.group("output").strip())
    for path in (d["output"], d["input"]):
        sm = _SNAPDIR.search(path)
        if sm:
            d["snap"] = int(sm.group(1))
            d["sim"] = Path(path[: sm.start() + (1 if path[sm.start()] == "/" else 0)]).name
            break
    return d


def _etime_seconds(text: str) -> float | None:
    """ps's elapsed time '[[dd-]hh:]mm:ss' -> seconds."""
    m = _ETIME.match(text.strip())
    if not m:
        return None
    dd, hh, mm, ss = (int(g) if g else 0 for g in m.groups())
    return float(((dd * 24 + hh) * 60 + mm) * 60 + ss)


def running_now(progress_files=(), procs: dict | None = None) -> list[dict]:
    """The runs going on this machine, wherever they were started (nohup, a terminal script, this launcher), for
    the launcher's 'Running now' panel: one dict per live DTFE / PS-DTFE run (dtfe_processes()
    'runs') with its pid, program, resident GB, elapsed seconds (ps), what its command line says (parse_run_args:
    simulation, snapshot, output root) and its progress sidecar '<output root>.runlog.progress' (C1; read with
    results.read_progress), plus one 'died' entry per leftover sidecar in progress_files whose pid is gone.
    Keys: pid, name, gb, elapsed_s, sim, snap, output, progress (dict or None), state ('running' | 'died')."""
    import results as R
    procs = procs if procs is not None else dtfe_processes(0.0)
    pids = [pid for pid, _n, _g in procs.get("runs", [])]
    info: dict[int, tuple[float | None, str]] = {}
    if pids:
        try:
            text = subprocess.run(["ps", "-ww", "-o", "pid=,etime=,args=", "-p", ",".join(map(str, pids))],
                                  capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            text = ""
        for line in text.splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit():
                info[int(parts[0])] = (_etime_seconds(parts[1]), parts[2] if len(parts) > 2 else "")
    out, seen = [], set()
    for pid, name, gb in procs.get("runs", []):
        elapsed, args = info.get(pid, (None, ""))
        d = parse_run_args(args)
        prog = None
        if d["output"]:
            side = Path(d["output"] + R.PROGRESS_SUFFIX)
            seen.add(str(side))
            prog = R.read_progress(side)
        out.append({"pid": pid, "name": name, "gb": gb, "elapsed_s": elapsed, "sim": d["sim"], "snap": d["snap"],
                    "output": d["output"], "progress": prog, "state": "running"})
    for f in progress_files:
        if str(f) in seen:
            continue
        prog = R.read_progress(f)
        if prog is None or prog.get("state") == "running":
            continue            # a live run that is not a DTFE binary's child we know: its own process row shows it
        output = str(f)[: -len(R.PROGRESS_SUFFIX)]
        d = parse_run_args(f"DTFE x.hdf5 {output}")
        out.append({"pid": prog.get("pid"), "name": "", "gb": None, "elapsed_s": prog.get("elapsed_s"),
                    "sim": d["sim"], "snap": d["snap"], "output": output, "progress": prog, "state": "died"})
    return out


def progress_files(data_root, extra_dirs=()) -> list[Path]:
    """Every progress sidecar '<prefix>.runlog.progress' (C1) under the data root's snapshot folders (per-family
    and flat layout) and in the custom-snapshot output folders."""
    found: list[Path] = []
    if root_ok(data_root):                      # an empty root is none (not the working directory)
        for pattern in ("*/snapdir_*/*.runlog.progress", "*/*/snapdir_*/*.runlog.progress"):
            found += Path(data_root).expanduser().glob(pattern)
    for d in extra_dirs:
        if Path(d).is_dir():
            found += Path(d).glob("*.runlog.progress")
    return sorted(set(found))


# ---------------------------------------------------------------------- progress parsing
_PARTS = re.compile(r"\[partitions (\d+)/(\d+) \| (\d+)%\]\s+elapsed ([^,]+),\s*"
                    r"(?:ETA ~?(\d+h(?: \d+m)?(?: \d+s)?|\d+m(?: \d+s)?|\d+s)|done)")
# wget -nv: one line per file fetched ('... URL:<url> [<bytes>/<bytes>] -> "<path>" [1]')
_WGET = re.compile(r'URL:\S+ \[(\d+)(?:/\d+)?\] -> "([^"]+\.hdf5)"')
_CHUNK = re.compile(r"^\s*\[(\d+)/(\d+)\] ")                  # merge_HDF5.py: '  [k/N] ...'
# lines that mean something went wrong although the tool exits 0 (download/merge report, not fail)
PROBLEM = re.compile(r"MISMATCH|Problem: Can't find|Error downloading|Error: |FAILED|NOT deleting")
PROGRESS = re.compile(r"\s*\[partitions \d+/\d+ \| \d+%\][^\n\[]*")   # the fragment to hide in logs
_SNAP = re.compile(r"Processing snapshot (\d+)")
# run_ps_pipeline.sh echoes the run_ps_dtfe.sh call with the snapshots it will compute
_PLAN = re.compile(r"^\+ .*run_ps_dtfe\.sh -s (\S+) -g \d+(?: -m)?((?: \d+)+)\s*$")   # -m only with DTFE_GPU=1
_SIM = re.compile(r"^== (\S+): snapshots")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


def parse_progress(text: str) -> dict:
    """The latest partition progress and snapshot seen in a chunk of run-script output."""
    info: dict = {}
    for m in _SNAP.finditer(text):
        info["snapshot"] = int(m.group(1))
    for m in _PARTS.finditer(text):
        info.update(done=int(m.group(1)), total=int(m.group(2)), percent=int(m.group(3)),
                    elapsed=m.group(4).strip(), eta=(m.group(5) or "done").strip())
    if "processed successfully" in text:
        info["snapshot_finished"] = True
    for m in _WGET.finditer(text):
        info["downloaded"] = info.get("downloaded", []) + [(m.group(2), int(m.group(1)))]
    for line in text.splitlines():
        m = _CHUNK.match(line)
        if m:
            info["chunk"] = (int(m.group(1)), int(m.group(2)))
        if PROBLEM.search(line):
            info["problem"] = info.get("problem", 0) + 1
        m = _PLAN.match(line)
        if m:
            info["planned"] = info.get("planned", 0) + len(m.group(2).split())
        m = _SIM.match(line)
        if m:
            info["sim"] = m.group(1)
    return info
