"""What earlier runs left on disk, read back from their run logs -- for the GUI's Runs browser,
its time estimates and its failure advice. Pure Python (no Qt): tests/py_gui_runspec_test.py.

  scan_runs(root, extra_dirs)   every '<prefix>.runlog' under the data root (per-family layout)
                                and under the custom-snapshot output directories -> RunRecord, with
                                a later failed, killed or running attempt ('.runlog.failed/.partial')
  stale_reasons(record)         what to know before using a run's outputs: an older version of the
                                programs made them (RunRecord.older), they show a known problem
                                (stream counts, no tetrahedron masses), the attempt after the run
                                (attempt_note)
  older_log(path)               run_ps_pipeline.sh's test of a run log, on its head (the Pipeline
                                tab's prediction of what the script recomputes)
  read_device(text, ...)        the device a run's header names (a run log's, a memory check's)
  time_estimate(records, ...)   the median wall time of comparable earlier runs
  diagnose(log, exit_codes)     plain-language causes and fixes for a failed job
  void_status(records, root)    whether each run's void catalogue is cached, stale or absent (void_cache.py)
  read_progress / is_run_file   the run-directory files of a run going (contract C1): the binary's
                                '<prefix>.runlog.progress', the scripts' '.runlog.lock' and '<grid>.part.<pid>'
                                are never outputs nor logs; a '.partial' with a live sidecar shows its progress
  notify_command(title, text)   the OS command that shows a desktop notification

A run log is the binary's own output, tee'd next to its outputs by run_ps_dtfe.sh, run_dtfe.sh
and the GUI's custom-snapshot jobs, followed (macOS) by '/usr/bin/time -l'. The run-log contract:
a run writes '<prefix>.runlog.partial'; exit 0 renames it onto '<prefix>.runlog' (the log of the
grids on disk), a failure or stop onto '<prefix>.runlog.failed' (the last failed attempt, the old
'.runlog' untouched); a '.partial' left behind is a run still going, or one killed outright.
"""

from __future__ import annotations

import datetime as _dt
import glob as _glob
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------- run records
_BUILD = re.compile(r"^Build: (\S+) (\S+) \(built ([0-9T:\-]+Z?)\)", re.M)
_RUNNING = re.compile(r"^RUNNING: (.*)$", re.M)
_WALL = re.compile(r"total wall time\s*:\s*((?:\d+h\s*)?(?:\d+m\s*)?(?:[\d.]+s)?)")
_RSS = re.compile(r"peak memory \(RSS\)\s*:\s*([\d.]+)\s*GB")
_FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M)
_GRID = re.compile(r"--grid (\d+)")
_PARTITION = re.compile(r"--partition (\d+) (\d+)(?: (\d+))?")    # a 2D run's split has two numbers
_CONC = re.compile(r"--max-concurrent (\d+)")
_NSUB = re.compile(r"--subsamples (\d+)")
# the PS-DTFE run-log header's count line, 'avg sub-samples (nSub) : N', printed by every sampled run with an '_a' field
# (src/user_options.cc): the count of a run whose command names none (the drivers pass '--subsamples' only when
# SUBSAMPLES is set), which took the binary's default -- 3 points per cell axis before Fast became one
_NSUB_HEAD = re.compile(r"^\s*avg sub-samples \(nSub\)\s*:\s*(\d+)", re.M)
_MERGE = re.compile(r"(?:^|\s)--ps-merge[ =](\S+)")
# the run-log header's run-mode line, every run's since the run modes: 'run mode : balanced (--run-mode balanced, the
# default; ...)', the mode the run took and why (the option, DTFE_RUN_MODE or the binary's default, which is balanced,
# or performance inside a batch job): RunRecord.run_mode. A log without it is an older version's (RunRecord.older,
# for a log with a command; older_log, the pipeline's test)
_RUN_MODE_HEAD = re.compile(r"^\s*run mode\s+:\s+(gentle|balanced|performance)\b", re.M)
# the run-log header's device line, where the device matters (standard DTFE's "'_a' interpolation : ..." with an '_a'
# field, PS-DTFE's 'PS deposit : ...' with any): a value starting with 'CPU' is the CPU ('CPU (DTFE_GPU=0)', 'CPU (the
# default for the sampled average; ...)'), else one holding ' GPU' the GPU ('Metal GPU (--gpu)', 'Metal GPU (the
# default; ...)'). A binary that picks its device prints it on the GPU and the CPU alike; an older one printed it only
# for a command with '--gpu', so a log without it is read by that flag (RunRecord.gpu)
_DEVICE = re.compile(r"^\s*(?:'_a' interpolation|PS deposit)\s+:\s+(.*)$", re.M)
# the run-log header's line on the cell average, the run's own word on it: standard DTFE 'volume averaging : EXACT
# cell-tetrahedron integration ...' or 'SAMPLED (...)', PS-DTFE 'grid deposit : EXACT tetrahedron-cell intersection
# moments ...' or 'SAMPLED on nSub^3 sub-points (...)'. Binaries since 2026-11-09 print it either way (the exact
# average is their default, so a command without a flag says nothing); older ones printed only the exact line
_AVERAGE = re.compile(r"^\s*(?:volume averaging|grid deposit)\s+:\s+(EXACT (?:cell-tetrahedron integration"
                      r"|tetrahedron-cell intersection moments)|SAMPLED\b)", re.M)
# the run-log header's line on the velocity gradient's route (both binaries, where a field built on the gradient is
# computed): 'velocity gradient : from EXACT FACE ...' for the face gradient, any other text for the deposit's own
# (PS-DTFE: the stream average of each stream's own gradient). A PS-DTFE log of a binary older than its face-gradient
# default has none: its route is the stream average unless the command names
# '--face-gradient'
_GRADIENT = re.compile(r"^\s*velocity gradient\s+:\s+(from EXACT FACE)?", re.M)
# the PS-DTFE run-log header's line on the velocity moments, printed by every PS-DTFE run beside its command line:
# 'velocity moments : volume-weighted (the dispersion mass-weighted)'. A log of an older binary has none: its moments
# were volume-weighted only where the command names '--ps-volume-weighted' (dtfelib.io.run_velocity_weighting reads
# both the same way)
_MOMENTS = re.compile(r"^\s*velocity moments\s+:\s+(volume-weighted)?", re.M)
_FIELDS = re.compile(r"--field ((?:(?!--)\S+\s*)+)")
# the run log's closing tally (contract G1), printed when N > 0
_GPU_RETRIES = re.compile(r"^GPU deposit retries: (\d+) \(innocent (\d+), timeout (\d+), oom (\d+), other (\d+)\)", re.M)
GPU_RETRY_CAUSES = ("innocent", "timeout", "oom", "other")
# the run-log contract's three names: the log of record, the last failed attempt, a run going or killed
LOG_SUFFIXES = (".runlog", ".runlog.failed", ".runlog.partial")
# the run-directory files of a run going (contract C1): the binary's JSON progress sidecar, the run
# scripts' lock DIRECTORY (holding the driver's 'pid'), and a grid written as '<file>.part.<pid>' until renamed
# at the end. None of them is an output or a run log.
PROGRESS_SUFFIX = ".runlog.progress"
LOCK_SUFFIX = ".runlog.lock"
# the checkpoint DIRECTORY of a killed partitioned PS-DTFE run (contract K2): its merged accumulators
# and the manifest of finished partitions; the next run of the same job resumes from it ('--resume', which
# run_ps_dtfe.sh passes by itself when it is there) and removes it on success
CHECKPOINT_SUFFIX = ".runlog.checkpoint"
# the binary's errors when '--resume' cannot continue the checkpoint (DTFE.cpp): other settings, nothing
# to resume (no manifest), not partitioned, a '--sample-points' run. Only the error phrasings: its info lines
# ("'--resume' continues it", "... continues a killed run from it") are in the logs of runs that CAN be resumed.
RESUME_REFUSED = re.compile(r"'--resume'(?::|\s+refused|\s+cannot|\s+continues a partitioned run)")
_PART = re.compile(r"\.part\.\d+$")


def is_run_file(name: str) -> bool:
    """A file of a run going (C1): its progress sidecar (and the sidecar's '.tmp'), its lock, a grid half written
    as '<file>.part.<pid>', a killed run's checkpoint directory and what is in it (K2). Never an output, never a
    run log."""
    return (PROGRESS_SUFFIX in name or LOCK_SUFFIX in name or CHECKPOINT_SUFFIX in name or _PART.search(name) is not None
            or name.endswith(".part.tmp"))


def pid_alive(pid) -> bool | None:
    """Whether process 'pid' exists (None: cannot say). EPERM means it exists under another user."""
    import os
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


_DTFE_NAME = re.compile(r"^(PS-)?DTFE(-.*)?$")    # every binary set (runspec.dtfe_processes' pgrep pattern)


def dtfe_pid_alive(pid) -> bool | None:
    """Whether process 'pid' is alive AND a DTFE / PS-DTFE binary (None: cannot say). A bare pid test is not
    enough for a leftover sidecar: a run killed outright with its driver (a reboot, a group SIGKILL)
    leaves its sidecar, and once the launcher's own polling has wrapped the pids (~a day) its pid belongs to some
    other process, which would read as the run still going. 'ps -o comm=' is argv[0] on macOS (a path, maybe with
    spaces) and the executable's 15-character name on Linux: its basename is matched. Without ps the bare pid
    test stands."""
    alive = pid_alive(pid)
    if not alive:
        return alive
    try:
        pr = subprocess.run(["ps", "-p", str(int(pid)), "-o", "comm="], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return alive
    comm = pr.stdout.strip()
    if pr.returncode != 0 or not comm:
        return False                                # gone between the two tests
    return _DTFE_NAME.match(comm.rstrip("/").rsplit("/", 1)[-1]) is not None


def read_progress(path) -> dict | None:
    """The binary's progress sidecar '<prefix>.runlog.progress' (C1) as a dict, with 'state' added: 'running'
    (its pid is a live DTFE binary, dtfe_pid_alive), 'died' (the pid is gone or now another program's: a run
    killed outright leaves the sidecar behind; a run that finishes removes it) or 'unknown'. None without a
    readable sidecar (the binary writes it atomically, so a half-written one is never seen; a missing key reads
    as absent)."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    alive = dtfe_pid_alive(d.get("pid"))
    d["state"] = "running" if alive else "died" if alive is False else "unknown"
    return d


def progress_text(d: dict | None) -> str:
    """'phase 3/27 (11%), ETA 2h 05m, 37.2 GB' -- one line for a progress sidecar; 'died' for a dead pid."""
    if not d:
        return ""
    bits = []
    phase = str(d.get("phase") or "").strip()
    done, total = d.get("done"), d.get("total")
    try:
        frac = f"{int(done)}/{int(total)}" + (f" ({100.0 * float(done) / float(total):.0f}%)" if float(total) > 0 else "")
    except (TypeError, ValueError):
        frac = ""
    head = " ".join(b for b in (phase, frac) if b)
    if head:
        bits.append(head)
    if d.get("state") == "died":
        bits.append("died (its process is gone)")
        return ", ".join(bits)
    for key, label in (("eta_s", "ETA"), ("elapsed_s", "elapsed")):
        try:
            v = float(d.get(key))
        except (TypeError, ValueError):
            continue
        if v >= 0 and math.isfinite(v):
            bits.append(f"{label} {format_duration(v)}")
            break
    try:
        bits.append(f"{float(d.get('rss_gb')):.1f} GB")
    except (TypeError, ValueError):
        pass
    try:
        if int(d.get("gpu_retries") or 0):
            bits.append(retries_text(int(d["gpu_retries"]), d.get("gpu_retries_by_cause")))
    except (TypeError, ValueError):
        pass
    return ", ".join(bits)


def retries_text(total: int, by_cause: dict | None = None) -> str:
    """'3 GPU retries (timeout 2, oom 1)': the nonzero causes of the sidecar's or the log's tally (G1); '' for none.
    innocent = Metal's 'Innocent Victim' (another process hung the GPU, ours was collateral), timeout = the GPU
    watchdog killed our own buffer, oom = out of GPU memory, other = any other GPU error."""
    if not total:
        return ""
    causes = []
    if isinstance(by_cause, dict):
        for c in GPU_RETRY_CAUSES:
            try:
                if int(by_cause.get(c) or 0):
                    causes.append(f"{c} {int(by_cause[c])}")
            except (TypeError, ValueError):
                pass
    return f"{int(total)} GPU retries" + (f" ({', '.join(causes)})" if causes else "")


def log_prefix(name: str) -> str:
    """'ps_output.runlog.failed' -> 'ps_output' (any of the three log names; a name without one stays)."""
    for sfx in LOG_SUFFIXES:
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return name


def parse_duration(text: str) -> float | None:
    """'1h 2m 3s' / '58m 11s' / '12.3s' -> seconds."""
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)\s*([hms])", text or ""):
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
        found = True
    return total if found else None


def format_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s" if m else f"{s}s"


@dataclass
class RunRecord:
    path: Path                      # the .runlog
    sim: str = ""                   # "" for a custom-snapshot run
    snap: int | None = None
    prefix: str = ""
    binary: str = ""                # PS-DTFE | DTFE (-2d, -double; "" when unknown)
    command: str = ""               # the RUNNING line (paths unquoted, as the binary printed it)
    header_average: str = ""        # 'exact' | 'sampled': the header's cell-average line (_AVERAGE); '' without one
    header_gradient: str = ""       # 'face' | 'per-tet': the header's velocity-gradient line (_GRADIENT); '' if none
    header_nsub: int | None = None  # the header's count line (_NSUB_HEAD): a sampled PS-DTFE run's; None without one
    header_moments: str = ""        # 'volume' | 'other': the header's velocity-moments line (_MOMENTS); '' if none
    header_device: str = ""         # 'cpu' | 'gpu': the header's device line (_DEVICE); '' without one
    header_run_mode: str = ""       # 'gentle' | 'balanced' | 'performance': the header's run-mode line (_RUN_MODE_HEAD)
    build_rev: str = ""
    build_time: _dt.datetime | None = None
    finished: _dt.datetime | None = None   # the log's modification time = when the run ended
    gpu_retries: int = 0            # the log's 'GPU deposit retries: N (...)' line (G1): N, and per cause
    gpu_retries_by_cause: dict = field(default_factory=dict)
    wall_seconds: float | None = None
    peak_rss_gb: float | None = None
    footprint_gb: float | None = None
    ok: bool = False                # reached the binary's summary without an ERROR block
    outputs: list[Path] = field(default_factory=list)
    # a later run under this name that left no log of record (scan_runs, the run-log contract):
    # 'failed' ('<prefix>.runlog.failed'), 'running' or 'killed' ('<prefix>.runlog.partial' with or without a
    # live process naming it; 'partial' when ps cannot say), 'resumable' (failed or killed, with its checkpoint
    # directory '<prefix>.runlog.checkpoint' left: the same job continues from it, K2), '' when none is newer
    # than this log. Without a '<prefix>.runlog' the record IS that attempt (path == attempt_log, never ok).
    attempt: str = ""
    attempt_log: Path | None = None
    attempt_time: _dt.datetime | None = None
    attempt_cause: str = ""         # diagnose()'s titles for the attempt's log ("Out of memory", ...)
    attempt_resume_refused: bool = False   # the attempt failed because the binary refused to resume its checkpoint
                                           # (RESUME_REFUSED): never 'resumable', the same job fails again
    attempt_rewrote: int = 0        # outputs modified after this log was: the attempt rewrote them
    progress: dict | None = None    # the binary's progress sidecar of a run going (C1, read_progress)

    # -- the settings the RUNNING line shows, read as written
    @property
    def older(self) -> bool:
        """A log of an older version of the programs: it has a RUNNING line and no header 'run mode' line (every
        run's since the run modes prints one, and every log the drivers and the launcher wrote since then spells its
        options in the current names). Its settings are not read: it is listed with its name, time, outcome and
        files, its settings read 'older version' (options_text), 'Load settings' is not offered for it (settings()
        gives nothing), time_estimate leaves it out and stale_reasons says to recompute it. A log without any
        command (a run that died before the program printed its header) is not one: it has no settings to read."""
        return bool(self.command) and not self.header_run_mode

    def has(self, flag: str) -> bool:
        """The command carries the option 'flag', as written."""
        return re.search(rf"(^|\s){re.escape(flag)}(\s|$)", self.command) is not None

    @property
    def exact(self) -> bool:
        """The run's cell averages were exact (standard DTFE's '_a' fields; PS-DTFE's grid deposit): the header's line
        says so when the log has one (a binary since 2026-11-09 prints it either way, and resolves a command without a
        flag to its exact default); else the command, '--exact' against '--sampled' or nothing -- an older log without
        either was sampled, the default until then."""
        if self.header_average:
            return self.header_average == "exact"
        return self.has("--exact")

    @property
    def face_gradient(self) -> bool:
        """PS-DTFE: the run's velocity-gradient family is the mean flow's, from exact face sums -- the header's line
        says so when the log has one (the binary prints it where a field built on the gradient is computed, and
        resolves a command without a route flag to its default); else the command's '--face-gradient' (an older
        binary's run without it holds the stream average, that binary's only default)."""
        if self.header_gradient:
            return self.header_gradient == "face"
        return self.has("--face-gradient")

    @property
    def volume_weighted(self) -> bool:
        """PS-DTFE: the run's velocity moments were volume-weighted -- the header's 'velocity moments' line says so
        when the log has one (every PS-DTFE run prints it: the program always volume-weights them); else the
        command's '--ps-volume-weighted' (an older binary's run without it holds mass-weighted moments)."""
        if self.header_moments:
            return self.header_moments == "volume"
        return self.has("--ps-volume-weighted")

    @property
    def estimator(self) -> str:
        return "ps" if self.binary.startswith("PS-DTFE") else "dtfe" if self.binary.startswith("DTFE") else ""

    @property
    def precision(self) -> str:
        """'double' for the DTFE-double / PS-DTFE-double pair (their Build line names it), else 'single'."""
        return "double" if self.binary.endswith("-double") else "single"

    @property
    def dim(self) -> int:
        """2 for the 2D programs (DTFE-2d, PS-DTFE-2d[-double]), else 3."""
        return 2 if "-2d" in self.binary else 3

    @property
    def grid(self) -> int | None:
        m = _GRID.search(self.command)
        return int(m.group(1)) if m else None

    @property
    def partitions(self) -> int:
        m = _PARTITION.search(self.command)
        return math.prod(int(g) for g in m.groups() if g) if m else 1

    @property
    def partition(self) -> int:
        m = _PARTITION.search(self.command)
        return int(m.group(1)) if m else 0

    @property
    def max_concurrent(self) -> int:
        m = _CONC.search(self.command)
        return int(m.group(1)) if m else 0

    @property
    def nsub(self) -> int | None:
        """The sampled deposit's points per cell axis: the command's '--subsamples k', else the header's count line
        (a PS-DTFE run whose command names none: the binary's default then), else None."""
        m = _NSUB.search(self.command)
        return int(m.group(1)) if m else self.header_nsub

    @property
    def gpu(self) -> bool:
        """The run's grid work ran on the GPU: the header's device line says so when the log has one (a binary that
        picks its device prints it wherever the device matters, the GPU or the CPU); else the command's '--gpu' (an
        older binary ran on the GPU only with it)."""
        if self.header_device:
            return self.header_device == "gpu"
        return self.has("--gpu")

    @property
    def sliced(self) -> bool:
        return self.has("--sample-points")

    @property
    def fields(self) -> list[str]:
        m = _FIELDS.search(self.command)
        return m.group(1).split() if m else []

    def checkpoint_dir(self) -> Path:
        """Where a killed partitioned run of this output keeps its checkpoint (K2; under '--scratch-dir' it lives
        there instead, which a scan of the output folder cannot see)."""
        return Path(str(self.path.parent / self.prefix) + CHECKPOINT_SUFFIX)

    def log_files(self) -> list[Path]:
        """Every log of this output set on disk (the run-log contract's three names): what Move to
        Trash must take with the outputs, or the attempt's log alone shows up as a failed run afterwards."""
        base = str(self.path.parent / self.prefix)
        return [p for p in (Path(base + sfx) for sfx in LOG_SUFFIXES) if p.is_file()]

    @property
    def size_bytes(self) -> int:
        total = 0
        for p in self.outputs:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def options_text(self) -> str:
        """The run's settings in a few words (the Runs browser's Options column, its details and its filter); an
        older version's log (older): 'older version', its settings not read; a log without any command: ''."""
        if not self.command:
            return ""
        if self.older:
            return "older version"
        bits = []
        if self.estimator in ("ps", "dtfe"):
            # the quality, in the launcher's words: Exact (the exact cell averages: the header's line, else
            # '--exact') or Fast (the sampled ones)
            bits.append("Exact" if self.exact else "Fast")
        if self.estimator == "ps":
            if not self.exact and self.nsub:          # Fast runs at two counts must differ here
                bits.append(f"{self.nsub} sample{'s' if self.nsub != 1 else ''} per cell axis")
            if self.face_gradient:                    # the mean flow's gradient: the same snapshot, another divergence
                bits.append("face gradient")
            if self.has("--ps-vertex-mass"):
                bits.append("vertex mass")
            if self.volume_weighted:                  # the header's line, else the command's flag
                bits.append("volume-weighted")
            if self.has("--ps-caustics"):
                bits.append("caustics")
            if self.merge == "index-order":
                bits.append("ordered merge")
            elif self.merge == "completion-order":    # opt-in (the default is the ordered merge)
                bits.append("completion-order merge")
        if self.gpu:
            bits.append("GPU")
        if self.precision == "double":
            bits.append("double")
        if self.partitions > 1:
            bits.append(f"{self.partition}{'²' if self.dim == 2 else '³'} partitions")
        if self.dim == 2:
            bits.append("2D")
        if self.sliced:
            bits.append("slice")
        return ", ".join(bits)

    @property
    def merge(self) -> str:
        """The PS-DTFE merge mode the command GAVE ('--ps-merge MODE'), '' when it gave none (the default, index
        order)."""
        m = _MERGE.search(self.command)
        return m.group(1) if m else ""

    @property
    def run_mode(self) -> str:
        """The machine use the run took: the header's 'run mode' line (it names the mode however it was chosen:
        '--run-mode', DTFE_RUN_MODE or the binary's default); else 'performance' (a log without the line is an older
        version's or holds no command: neither has settings that are read)."""
        return self.header_run_mode or "performance"

    def settings(self) -> dict:
        """The Grids-tab (RunSpec) settings this run used, as far as its command shows them; the quality is the cell
        average's (Exact or Fast). Nothing for an older version's log (older), nor for a log without any command."""
        if not self.command or self.older:
            return {}
        d = {"estimator": self.estimator or "ps", "gpu": self.gpu, "precision": self.precision}
        if self.grid:
            d["grid"] = self.grid
        if self.estimator == "ps":
            d.update(quality="exact" if self.exact else "fast",
                     # the tick: off only where the run named the stream average ('--per-tet-gradient'); a run that
                     # named no route took the binary's default, which a ticked job asks for where it qualifies
                     face_gradient=not self.has("--per-tet-gradient"),
                     vertex_mass=self.has("--ps-vertex-mass"),
                     volume_weighted=self.volume_weighted,
                     caustics=self.has("--ps-caustics"), caustic_cusps=self.has("--ps-caustic-cusps"),
                     parallel_triangulation=self.has("--parallel-triangulation"),
                     # the default is ON (the staged merge in index order), so only a run made with '--ps-merge
                     # completion-order' turns it off -- an older binary's run without the flag merged in completion
                     # order, but loading its settings starts the current default
                     ordered_merge=self.merge != "completion-order",
                     output_prefix=self.prefix or "ps_output")
            m = re.search(r"(?:^|\s)--ps-alpha-shape\s+([0-9.]+)", self.command)
            if m:
                d["alpha_shape"] = float(m.group(1))
            if self.nsub:
                d["nsub"] = self.nsub
            if self.fields:
                d["fields"] = self.fields
        elif self.estimator == "dtfe":      # its cell average, both ways (else the Grids tab's choice would stay)
            d["quality"] = "exact" if self.exact else "fast"
        return d


def custom_settings(rec: RunRecord) -> dict | None:
    """The settings a custom-snapshot run was made with: its '<prefix>.gui.json' sidecar (CustomSpec keys),
    None for a TNG run or a custom run without one."""
    if rec.sim:
        return None
    try:
        d = json.loads(Path(str(rec.path.parent / rec.prefix) + ".gui.json").read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("input_file") else None


INPUT_SUFFIXES = (".hdf5", ".h5", ".hdf")      # snapshots: the binaries read them, never write them


def run_inputs(rec: RunRecord) -> set[Path]:
    """The files a run READ that sit beside its outputs: every existing file named on its command line and
    the settings sidecar's input_file. A custom run named after its input (the default) in the input's own
    folder has 'X.hdf5' beside 'X.a_den': the input must never be one of the run's outputs (the Runs
    browser's Move to Trash takes the outputs)."""
    found = set()
    try:
        words = shlex.split(rec.command)
    except ValueError:
        words = rec.command.split()
    # the log prints the command UNQUOTED: a path with a space (anything under iCloud's 'Mobile Documents') splits
    # above, so the file options' values are also read up to the next ' --', and the sidecar names the rest
    for m in re.finditer(r"--(?:sample-points|ps-lagrangian-input|scalar-file)\s+(.+?)(?=\s--|$)", rec.command):
        words.append(m.group(1).strip())
    side = Path(str(rec.path.parent / rec.prefix) + ".gui.json")
    try:
        sd = json.loads(side.read_text())
        for key in ("input_file", "sample_points", "lagrangian_file"):
            if sd.get(key):
                words.append(str(sd[key]))
    except (OSError, ValueError, AttributeError):
        pass
    for w in words:
        f = Path(w).expanduser()
        f = f if f.is_absolute() else rec.path.parent / f
        try:
            if f.is_file():
                found.add(f.resolve())
        except OSError:
            pass
    return found


# the text-derived fields of every run log read, by path, with the (size, mtime_ns) they were read at:
# the launcher rescans every 30 s on its GUI thread, and a TNG50 z=0 log is large and on the
# external disk; an unchanged log is not read again
_LOG_FIELDS = ("binary", "build_rev", "build_time", "command", "header_average", "header_gradient", "header_nsub",
               "header_moments", "header_device", "header_run_mode", "gpu_retries", "gpu_retries_by_cause",
               "wall_seconds", "peak_rss_gb", "footprint_gb", "ok")
_LOG_CACHE: dict[str, tuple[int, int, dict]] = {}


def read_device(text: str, pos: int = 0, endpos: int | None = None) -> str:
    """The device a run's header names (_DEVICE) in text[pos:endpos] (one run's output: from its RUNNING line to the
    next): 'cpu', 'gpu', or '' without a device line. The LAST such line counts (a picked GPU that met method 2 prints
    a replacing line). Read for a run log (RunRecord.header_device) and for a memory check's reports
    (runspec.parse_reports)."""
    ds = list(_DEVICE.finditer(text, pos, len(text) if endpos is None else endpos))
    if not ds:
        return ""
    val = ds[-1].group(1).strip()
    return "cpu" if val.startswith("CPU") else "gpu" if " GPU" in val else ""


def _parse_text(rec: RunRecord, text: str):
    m = _BUILD.search(text)
    if m:
        rec.binary, rec.build_rev = m.group(1), m.group(2)
        try:
            rec.build_time = _dt.datetime.fromisoformat(m.group(3).replace("Z", "+00:00")) \
                .astimezone().replace(tzinfo=None)
        except ValueError:
            pass
    m = _RUNNING.search(text)
    if m:
        rec.command = m.group(1).strip()
        exe = rec.command.split()[0] if rec.command else ""
        if not rec.binary:
            for name in ("PS-DTFE-2d-double", "DTFE-2d-double", "PS-DTFE-2d", "DTFE-2d",
                         "PS-DTFE-double", "DTFE-double", "PS-DTFE", "DTFE"):
                if exe.endswith(name):
                    rec.binary = name
                    break
        # the cell-average line of THIS command's header (it follows the RUNNING line; a log may hold more runs)
        nxt = _RUNNING.search(text, m.end())
        a = _AVERAGE.search(text, m.end(), nxt.start() if nxt else len(text))
        if a:
            rec.header_average = "exact" if a.group(1).startswith("EXACT") else "sampled"
        g = _GRADIENT.search(text, m.end(), nxt.start() if nxt else len(text))   # the route's line, the same way
        if g:
            rec.header_gradient = "face" if g.group(1) else "per-tet"
        k = _NSUB_HEAD.search(text, m.end(), nxt.start() if nxt else len(text))   # the count's line, the same way
        if k:
            rec.header_nsub = int(k.group(1))
        v = _MOMENTS.search(text, m.end(), nxt.start() if nxt else len(text))     # the moments' line, the same way
        if v:
            rec.header_moments = "volume" if v.group(1) else "other"
        rec.header_device = read_device(text, m.end(), nxt.start() if nxt else len(text))   # the device's line too
        r = _RUN_MODE_HEAD.search(text, m.end(), nxt.start() if nxt else len(text))   # the run mode's line, too
        if r:
            rec.header_run_mode = r.group(1)
    tallies = _GPU_RETRIES.findall(text)
    if tallies:                     # the last: a resumed run's log holds each attempt's
        n = [int(x) for x in tallies[-1]]
        rec.gpu_retries, rec.gpu_retries_by_cause = n[0], dict(zip(GPU_RETRY_CAUSES, n[1:]))
    m = _WALL.search(text)
    rec.wall_seconds = parse_duration(m.group(1)) if m else None
    m = _RSS.search(text)
    rec.peak_rss_gb = float(m.group(1)) if m else None
    feet = _FOOTPRINT.findall(text)
    rec.footprint_gb = int(feet[-1]) / 2**30 if feet else None     # GiB: the unit of the RSS line and of the run scripts
    # a '.failed' or '.partial' log is an attempt that did not exit 0, whatever it printed
    rec.ok = bool(m or rec.wall_seconds is not None) and "~~~ ERROR ~~~" not in text and rec.path.name.endswith(".runlog")


def parse_runlog(path: Path, sim: str = "", snap: int | None = None) -> RunRecord:
    path = Path(path)
    rec = RunRecord(path=path, sim=sim, snap=snap, prefix=log_prefix(path.name))
    try:
        st = path.stat()
        rec.finished = _dt.datetime.fromtimestamp(st.st_mtime)
        hit = _LOG_CACHE.get(str(path))
        if hit is not None and hit[:2] == (st.st_size, st.st_mtime_ns):
            for k, v in hit[2].items():
                setattr(rec, k, v)
        else:
            _parse_text(rec, path.read_text(errors="replace"))
            _LOG_CACHE[str(path)] = (st.st_size, st.st_mtime_ns, {k: getattr(rec, k) for k in _LOG_FIELDS})
    except OSError:
        return rec
    inputs = run_inputs(rec)
    log = rec.prefix + ".runlog"    # by name: '.runlog.failed', '.runlog.partial' (and a script's '.runlog.tmp') too
    rec.outputs = sorted(p for p in path.parent.glob(_glob.escape(rec.prefix) + ".*")       # escaped: '[' in a name
                         if p.is_file() and p.suffix not in (".runlog", ".json") and not p.name.startswith(log)
                         and not is_run_file(p.name)         # C1: the sidecar, the lock, '<grid>.part.<pid>'
                         and p.suffix.lower() not in INPUT_SUFFIXES and p.resolve() not in inputs)
    return rec


_OLDER_CACHE: dict[str, tuple[int, int, bool]] = {}


def older_log(path, nbytes: int = 65536) -> bool:
    """run_ps_pipeline.sh's test of the run log at 'path' (the script's older_log), on the log's head, where the
    header is: no header 'run mode' line, so an older version of the programs wrote it and the script recomputes its
    snapshot. The Pipeline tab predicts that with it (runspec_pipeline.PipelineSpec.stale); the verdict is kept per
    log by its (size, mtime_ns), as parse_runlog keeps its reads. False for a missing log (the script judges a
    snapshot without a run log by the file times) and for one that cannot be read (the script recomputes that one)."""
    try:
        st = Path(path).stat()
        hit = _OLDER_CACHE.get(str(path))
        if hit is not None and hit[:2] == (st.st_size, st.st_mtime_ns):
            return hit[2]
        with open(path, "rb") as fh:
            older = _RUN_MODE_HEAD.search(fh.read(nbytes).decode(errors="replace")) is None
    except OSError:
        return False
    _OLDER_CACHE[str(path)] = (st.st_size, st.st_mtime_ns, older)
    return older


def _snap_of(snapdir: Path) -> int | None:
    try:
        return int(snapdir.name.split("_")[1])
    except (IndexError, ValueError):
        return None


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _live_roots(roots) -> set[str] | None:
    """Of these output roots ('<dir>/<prefix>'), the ones a live process names on its command line: the binary
    (its output argument) or a run script's 'tee <prefix>.runlog.partial'. One ps call; None when ps cannot
    say. (A run started with a relative path names its root differently: it reads as killed.)"""
    try:
        text = subprocess.run(["ps", "-A", "-ww", "-o", "args="], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {r for r in roots if re.search(re.escape(r) + r"(?=[\s.]|$)", text, re.M)}


def _tail(p: Path, nbytes: int = 65536) -> str:
    try:
        with open(p, "rb") as fh:
            fh.seek(max(0, fh.seek(0, 2) - nbytes))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def _read_run(logs: dict[str, Path], sim: str = "", snap: int | None = None) -> RunRecord | None:
    """The record of one output set from its logs ({suffix: path}, the run-log contract): '<prefix>.runlog'
    when there is one, the attempt after it ('.runlog.failed' or '.runlog.partial', the newer one) noted on
    it; without one the attempt is the record (a stopped or failed re-run does not truncate the log of the
    grids on disk, so the Runs tab keeps showing them as they are). None: the logs went meanwhile."""
    record = logs.get(".runlog")
    t0 = _mtime(record) if record is not None else None
    attempts = [(t, sfx, p) for sfx, p in logs.items() if sfx != ".runlog"
                for t in [_mtime(p)] if t is not None and (t0 is None or t > t0)]
    if record is None and not attempts:
        return None
    rec = parse_runlog(record if record is not None else max(attempts)[2], sim, snap)
    if attempts:
        t, sfx, p = max(attempts)
        rec.attempt = "failed" if sfx == ".runlog.failed" else "partial"
        rec.attempt_log, rec.attempt_time = p, _dt.datetime.fromtimestamp(t)
        tail = _tail(p)
        rec.attempt_cause = ", ".join(d.title for d in diagnose(tail))
        rec.attempt_resume_refused = RESUME_REFUSED.search(tail) is not None
        if t0 is not None:          # a run's outputs are all written before its log ends: newer ones are the attempt's
            rec.attempt_rewrote = sum(1 for q in rec.outputs if (_mtime(q) or 0.0) > t0 + 5.0)   # 5 s: FAT's 2 s mtimes
        if sfx == ".runlog.partial":
            rec.progress = read_progress(Path(str(p)[: -len(".partial")] + ".progress"))
    return rec


def scan_runs(root, extra_dirs=()) -> list[RunRecord]:
    """Every run under the data root (<family>/<sim>/snapdir_NNN or flat <sim>/snapdir_NNN) and in the
    extra (custom-snapshot output) directories, one per output prefix with any of the three log names
    (_read_run), newest first. A '.runlog.partial' is 'running' when a live process names its root, else
    'killed' (a run killed outright: no exit code reached its end)."""
    out: list[RunRecord] = []
    root = Path(root).expanduser() if str(root).strip() else None     # an empty root is none, not the working dir
    groups: dict[Path, tuple[dict, str, int | None]] = {}       # '<dir>/<prefix>' -> ({suffix: log}, sim, snap)

    def add(log: Path, sim: str = "", snap: int | None = None):
        sfx = next((s for s in LOG_SUFFIXES if log.name.endswith(s)), None)
        if sfx is None or not log.is_file():                    # '*.runlog*' also finds a script's '.runlog.tmp'
            return
        logs, _, _ = groups.setdefault(log.parent / log_prefix(log.name), ({}, sim, snap))   # found under the data
        logs[sfx] = log                                                                   # root first: its sim stays

    if root is not None and root.is_dir():
        for pattern in ("*/snapdir_*/*.runlog*", "*/*/snapdir_*/*.runlog*"):
            for log in root.glob(pattern):
                add(log, log.parent.parent.name, _snap_of(log.parent))
    for d in extra_dirs:
        d = Path(d)
        if d.is_dir():
            for log in d.glob("*.runlog*"):
                add(log)
    for logs, sim, snap in groups.values():
        rec = _read_run(logs, sim, snap)
        if rec is not None:
            out.append(rec)
    partial = [r for r in out if r.attempt == "partial"]
    live = _live_roots([str(r.path.parent / r.prefix) for r in partial]) if partial else set()
    for r in partial:
        state = (r.progress or {}).get("state")
        if state in ("running", "died"):            # the sidecar's pid decides, whatever path the run was given
            r.attempt = "running" if state == "running" else "killed"
        elif live is not None:
            r.attempt = "running" if str(r.path.parent / r.prefix) in live else "killed"
    # K2: no process continues it, but the next run can -- when it is a Grids or Pipeline job (run_ps_dtfe.sh passes
    # '--resume' by itself, and so does run_dtfe.sh; they write under the data root's snapdir_*, so the record
    # has a sim; a Custom run never resumes and starts over), and not when the attempt failed BECAUSE the resume was
    # refused (other settings, no manifest): the same job then fails the same way every time, and only
    # 'Start over' runs it
    for r in out:
        if (r.attempt in ("failed", "killed") and r.checkpoint_dir().is_dir() and r.sim and r.estimator in ("ps", "dtfe")
                and not r.attempt_resume_refused):
            r.attempt = "resumable"
    return sorted(out, key=lambda r: r.finished or _dt.datetime.min, reverse=True)


# ---------------------------------------------------------------------- void catalogues
VOID_CACHE_TOOL = Path(__file__).resolve().parent / "void_cache.py"


def void_candidate(rec: RunRecord) -> bool:
    """A run whose grids the void pipeline reads: a finished TNG 3D full-box run with a density grid."""
    return (bool(rec.sim) and rec.snap is not None and rec.ok and rec.dim == 3 and rec.estimator in ("ps", "dtfe")
            and any(p.name.endswith((".a_den", ".den")) for p in rec.outputs))


def void_status(records, data_root, python: str | None = None, timeout: float = 120.0) -> dict[str, tuple[str, str]]:
    """{str(record.path): (state, why)} of the void catalogue of every void_candidate run (void_cache.py status:
    'cached', 'stale' = built from an older grid or its R_v not today's, 'absent', 'error'), one subprocess per simulation
    (config reads the box for DTFE_SIM). Reads only; seconds for a whole data root, so call it off the GUI thread."""
    import os
    by_sim: dict[str, list[RunRecord]] = {}
    for r in records:
        if void_candidate(r):
            by_sim.setdefault(r.sim, []).append(r)
    out: dict[str, tuple[str, str]] = {}
    for sim, recs in by_sim.items():
        items = [{"sim": sim, "snap": r.snap, "method": r.estimator, "prefix": r.prefix} for r in recs]
        env = dict(os.environ, DTFE_SIM=sim, DTFE_DATA_ROOT=str(data_root), MPLBACKEND="Agg")
        try:
            pr = subprocess.run([python or sys.executable, str(VOID_CACHE_TOOL), "status"], input=json.dumps(items),
                                capture_output=True, text=True, timeout=timeout, env=env)
            got = json.loads(pr.stdout.strip().splitlines()[-1]) if pr.returncode == 0 and pr.stdout.strip() else None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            got = None
        for k, r in enumerate(recs):
            d = got[k] if isinstance(got, list) and k < len(got) and isinstance(got[k], dict) else \
                {"state": "error", "why": "the status helper failed"}
            out[str(r.path)] = (str(d.get("state", "error")), str(d.get("why", "")))
    return out


# ---------------------------------------------------------------------- what to know before using a run's outputs
def _any_output(rec: RunRecord, *suffixes) -> bool:
    names = {p.name[len(rec.prefix) + 1:] for p in rec.outputs}
    return any(s in names for s in suffixes)


# the launcher's start-over control, as the notes name it (no '--': launcher text)
START_OVER_HINT = "'Start over' on the Grids or Pipeline tab (DTFE_RESUME=0, both run scripts)"


def attempt_note(rec: RunRecord) -> tuple[str, str] | None:
    """(level, message) about the attempt after this run's log (RunRecord.attempt), or None: a 'warning' when
    it rewrote outputs (or the record is a killed or failed attempt itself), else 'info'."""
    if not rec.attempt or rec.attempt_log is None:
        return None
    name = rec.attempt_log.name
    cause = f"; likely cause: {rec.attempt_cause}" if rec.attempt_cause else ""
    when = f"{rec.attempt_time:%Y-%m-%d %H:%M}" if rec.attempt_time else "later"
    prog = progress_text(rec.progress) if rec.attempt == "running" else ""
    if prog:
        prog = f" ({prog})"
    if rec.attempt == "failed" and rec.attempt_resume_refused:     # rerunning as is fails again
        return ("warning", f"the last attempt ({name}, {when}) failed because its checkpoint could not be resumed (written "
                           f"with other settings, or incomplete): {START_OVER_HINT} runs the job from the beginning"
                + ("" if rec.path == rec.attempt_log else
                   f"; it rewrote {rec.attempt_rewrote} of the outputs" if rec.attempt_rewrote else
                   "; the outputs are still this log's"))
    if rec.path == rec.attempt_log:                     # no run under this name ever finished
        return {"failed": ("warning", f"the last attempt under this name failed ({name}{cause}); none has finished"),
                "running": ("info", f"still running{prog} (its log so far: {name})"),
                "killed": ("warning", f"killed outright: its log {name} stops mid-run, so the outputs may be "
                                      f"incomplete{cause}"),
                "resumable": ("warning", f"resumable: stopped before the end ({name}{cause}) with its checkpoint "
                                         f"kept; running the same job again continues from the finished partitions "
                                         f"({START_OVER_HINT} runs it from the beginning instead)"),
                }.get(rec.attempt, ("warning", f"killed outright, or still running: its log {name} has no end yet"))
    n = rec.attempt_rewrote
    if rec.attempt == "running":
        return ("info", f"a new run under this name is going{prog} ({name}): "
                + (f"it has rewritten {n} of the outputs so far" if n else "the outputs are this run's until it ends"))
    what = {"failed": f"failed on {when} ({name}{cause})",
            "killed": f"was killed outright on {when} ({name} stops mid-run{cause})",
            "resumable": f"stopped on {when} ({name}{cause}) with its checkpoint kept: running the same job "
                         f"again resumes it ({START_OVER_HINT} starts over)"}.get(
        rec.attempt, f"was killed outright or is still going ({name}, last written {when})")
    return (("warning", f"a later run under this name {what}; it rewrote {n} of the outputs, which no longer "
                        "match this log: regenerate") if n else
            ("info", f"a later run under this name {what}; the outputs are still this run's"))


def stale_reasons(rec: RunRecord) -> list[tuple[str, str]]:
    """(level, message) for what to know before using this run's outputs: the attempt after its log (attempt_note:
    a failed, killed or running re-run under its name), an older version of the programs made them
    (RunRecord.older), and problems its outputs show."""
    out: list[tuple[str, str]] = []
    note = attempt_note(rec)
    if note is not None:
        out.append(note)
    if rec.older:
        out.append(("warning", "made by an older version of the programs: its settings are not read here, and fixes "
                               "made since may apply to it; recompute it before use"))
    # file-based: '.streams' without '.hidden_streams' (or its old name '.unresolved') predates the
    # stream-count fix
    if rec.estimator == "ps" and _any_output(rec, "streams", "a_streams") \
            and not _any_output(rec, "hidden_streams", "a_hidden_streams", "unresolved", "a_unresolved"):
        out.append(("warning", "the stream counts still include centroid-fallback tetrahedra (+1 each; "
                               "fixed 2026-09-28): stream fractions are inflated, so regenerate "
                               "before quoting them"))
    # '--ps-vertex-mass' was never renamed, so this reads an older version's log too
    if rec.estimator == "ps" and rec.sim.startswith("TNG") and rec.command and not rec.has("--ps-vertex-mass"):
        out.append(("warning", "made without tetrahedron masses from the particles: with TNG's z=127 "
                               "initial positions the density contrast is suppressed by 1 - D(127)/D(z)"))
    return out


# ---------------------------------------------------------------------- time estimates
def time_estimate(records, *, sim: str, grid: int, estimator: str, gpu: bool | None,
                  sliced: bool, precision: str = "single", dim: int = 3,
                  run_mode: str = "performance", exact: bool | None = None,
                  nsub: int | None = None) -> tuple[float, int] | None:
    """(median seconds per snapshot, number of runs) of successful earlier runs with the same
    simulation, grid, estimator, GPU use (RunRecord.gpu; 'gpu' None, a job on Automatic: runs on either device, as
    the program picks the same device for the same settings on the same build), slice/no slice, precision,
    dimension, machine use and, when 'exact' is given, quality (RunRecord.exact: Exact True, Fast False), and when
    'nsub' is given (a sampled PS-DTFE job's count) that count (RunRecord.nsub); None without any (a
    double-precision run takes about twice as long: it must not predict a single one; a gentle run on half the
    performance cores must not predict a performance one, nor the other way round; a Fast run must not predict an
    Exact one, nor a run at another count a Fast one). 'run_mode' defaults to performance. An older version's log
    (RunRecord.older) is left out: its settings are not read.
    A run under a second is one: the binary's SUMMARY prints its wall time in whole seconds ('0s', 0.0 here), and
    a truthiness test would drop such runs as if they had no time (the launcher test's 32^3 run takes 0.5 s)."""
    times = [r.wall_seconds for r in records
             if r.ok and not r.older and r.wall_seconds is not None and r.sim == sim and r.grid == grid
             and r.estimator == estimator and (gpu is None or r.gpu == gpu) and r.sliced == sliced
             and r.precision == precision and r.dim == dim and r.run_mode == run_mode
             and (exact is None or r.exact == exact) and (nsub is None or r.nsub == nsub)]
    return (statistics.median(times), len(times)) if times else None


# ---------------------------------------------------------------------- failure advice
@dataclass
class Diagnosis:
    title: str
    advice: str
    action: str = ""                # a GUI action key ("check_memory", "scratch", "choose_root",
                                    # "open_data", "build", "") -- the button the banner offers


_OOM = re.compile(r"std::bad_alloc|Cannot allocate memory|out of memory|Killed: 9|MemoryError", re.I)


def diagnose(log: str, exit_codes=()) -> list[Diagnosis]:
    """Likely causes of a failed job from the tail of its log and its exit codes."""
    tail = "\n".join(log.splitlines()[-400:])
    out: list[Diagnosis] = []
    codes = list(exit_codes)
    if _OOM.search(tail) or any(c in (137, -9, 9) for c in codes):
        out.append(Diagnosis("Out of memory",
                             "The run needed more memory than the machine had. 'Check memory' shows "
                             "what each snapshot needs; a scratch directory (More) moves the grids to disk "
                             "(bit-identical results), and a smaller grid, fewer fields or a smaller slice "
                             "plane lower the rest.", "check_memory"))
    if re.search(r"no snapshots under|data root not found|no data root|No such file or directory: '?/Volumes/", tail):
        out.append(Diagnosis("Data not found",
                             "The data folder is missing. Is the external disk plugged in? Choose "
                             "the data root again.", "choose_root"))
    if re.search(r"No 'InitialCoordinates'|neither .*InitialCoordinates|lagrangian.*(not found|cannot open)",
                 tail, re.I) and (codes and any(codes)):
        out.append(Diagnosis("Initial positions missing",
                             "PS-DTFE needs each particle's initial (Lagrangian) position: download "
                             "and convert the initial conditions (Data tab), or use a snapshot that "
                             "stores 'InitialCoordinates'.", "open_data"))
    # boost's errors for a name the program does not know ("unrecognised option '--foo'") and for an abbreviation
    # of several ("option '--input' is ambiguous and matches '--input-file', and '--input-format'")
    if re.search(r"unrecogni[sz]ed option|option '[^'\n]*' is ambiguous", tail):
        out.append(Diagnosis("An option was not recognised",
                             "The program has no option of that name: a removed or misspelled name (its --full-help "
                             "lists every option), or binaries older than the scripts (rebuild them: make DTFE "
                             "PS-DTFE).", "build"))
    if "points into iCloud" in tail or "iCloud Drive" in tail:
        out.append(Diagnosis("iCloud path refused",
                             "Scratch and cache directories must be on a local disk, not iCloud Drive "
                             "(e.g. /private/tmp/dtfe-scratch).", "scratch"))
    if re.search(r"Operation not permitted|Permission denied", tail):
        out.append(Diagnosis("No permission",
                             "macOS refused access to a file. Apps started from Finder need Full Disk "
                             "Access for iCloud Drive and external disks (System Settings -> Privacy & "
                             "Security -> Full Disk Access).", ""))
    if RESUME_REFUSED.search(tail):                 # the same job would fail the same way
        out.append(Diagnosis("Resume refused",
                             "A killed run's checkpoint is there, but it cannot be continued (written with other "
                             "settings, or incomplete). Tick 'Start over' on the Grids or Pipeline tab "
                             "(DTFE_RESUME=0, both run scripts) to run the job from the beginning; it "
                             "replaces the checkpoint.",
                             "start_over"))
    if re.search(r"OVER BUDGET|over the memory budget|exceed the memory budget", tail) and not out:
        out.append(Diagnosis("Over the memory budget",
                             "The auto-tuner warned that this run does not fit in memory.", "check_memory"))
    if not out and any(codes):
        out.append(Diagnosis("The run failed",
                             "See the last lines of the log above for the program's own message.", ""))
    return out


# ---------------------------------------------------------------------- notifications
def notify_command(title: str, text: str) -> list[str] | None:
    """The command that shows a desktop notification, or None where there is none."""
    if sys.platform == "darwin" and shutil.which("osascript"):
        esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')   # noqa: E731
        return ["osascript", "-e", f'display notification "{esc(text)}" with title "{esc(title)}"']
    if shutil.which("notify-send"):
        return ["notify-send", title, text]
    return None
