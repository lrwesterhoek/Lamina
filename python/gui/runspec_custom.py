"""The Custom tab's job, moved out of runspec.py: CustomSpec (the binary itself on any snapshot file: Gadget
HDF5/binary or text, units, box, periodicity and initial positions spelled out), the snapshot-file probes it uses
(dimension, and the HDF5 header's box size, redshift, particle count and datasets) and the first-run demo job.
runspec re-exports all of it and is the module to import; this one reads runspec's own names through it at call
time (_rs.NAME)."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

import runspec as _rs
from dtfelib import runlog as _runlog
from dtfelib.estimator import STANDARD_PADDING
from runspec import CUSTOM_OUT_DEFAULT, LAMBDA_TH_DEFAULT, QUALITY_DEFAULT   # the dataclass defaults


@dataclass
class CustomSpec:
    """The binary on the user's own snapshot file: one call, every assumption the TNG scripts make
    (layout, kpc units, the IC file) spelled out as options. Writes '<output_dir>/<name>.*' and a
    run log beside it; the GUI keeps these settings next to the outputs ('<name>.gui.json') so
    the Explore tab can query the same snapshot later. dim = 2 runs the 2D set (DTFE-2d, PS-DTFE-2d:
    a snapshot whose coordinates have two columns), CPU only."""
    input_file: str = ""
    input_type: int = 105
    dim: int = 3                    # 2: the 2D programs (make DIM=2) on a 2D snapshot (snapshot_dim)
    mpc_unit: float = 1.0           # 1 Mpc in the file's length unit (1 = Mpc, 1000 = kpc)
    periodic: bool = True
    box: list[float] = field(default_factory=list)   # [] = the header's box; else xlo xhi ylo yhi [zlo zhi]
    estimator: str = "ps"
    lagrangian: str = "file"        # "file": InitialCoordinates in the snapshot; "separate": lagrangian_file
    lagrangian_file: str = ""
    grid: int = 128
    quality: str = QUALITY_DEFAULT  # "exact" | "fast", see RunSpec: what it implies is quality_mode's (mode())
    fields: list[str] = field(default_factory=lambda: ["density_a", "velocity_a", "dispersion_a"])
    run_mode: str = "balanced"      # --run-mode, the machine use, see RunSpec (the binary's own default: performance)
    vertex_mass: bool = False       # the initial positions: PERTURBED (e.g. TNG's z=127; masses from the particles,
                                    # --ps-vertex-mass) or an unperturbed lattice (False)
    caustics: bool = False
    caustic_cusps: bool = False
    precision: str = "single"       # "double" = the -double binary (float64 end to end, ~2x memory): it picks the
                                    # program, as the dimension does
    alpha_shape: float = 3.0        # --ps-alpha-shape, non-periodic clouds only: emitted when not the default 3
    scalar_dataset: str = ""
    output_dir: str = str(CUSTOM_OUT_DEFAULT)
    output_name: str = ""           # "" = the input file's name
    scratch_dir: str = ""
    lambda_th: float = LAMBDA_TH_DEFAULT   # --lambda-th with a web field (the binary's own default is 0.0)
    tess_cache: str = ""            # --tessellation-cache <folder>; "" = none
    sample_points: str = ""         # --sample-points <file>: the fields at these points too ('.pts_*'); "" = none
    pts_vel_grad: bool = True       # ... with the velocity gradient at each point (--pts-vel-grad)
    window: list[float] = field(default_factory=list)   # --grid-window x0 x1 y0 y1 [z0 z1] in Mpc: only these cells
    halo_release: float = 0.0       # --ps-halo-release D: tetrahedra denser than D x rho_bar deposited at their centroid
    per_stream: bool = False        # --ps-per-stream: the sample points' values stream by stream (ragged '.pts_*')
    per_stream_ids: bool = False    # --ps-per-stream-ids: ... with each stream's tetrahedron particle IDs
    extra_args: str = ""            # more program options as on a command line, appended after the tab's own
    demo_n: int = 0                 # > 0: first generate the synthetic demo snapshot (crossed waves, n^3) ...
    demo_file: str = ""             # ... but only at THIS path (demo_spec sets it): the setting is persisted, and
                                    # without the path any later missing input would quietly become a demo
    advanced: dict = field(default_factory=dict)   # the Advanced overrides, see RunSpec (not 'precision': this
                                    # tab's own row above); the last field

    def __post_init__(self):
        if not self.quality:
            self.quality = QUALITY_DEFAULT
        self.advanced = _rs.clean_advanced(self.advanced, type(self))

    def mode(self) -> _rs.QualityMode:
        """What this job's quality implies (runspec.quality_mode) for its estimator, precision and dimension, its
        device by the snapshot's particle count (read from the HDF5 header only where the device depends on it),
        with the Advanced overrides it passes (runspec.apply_advanced)."""
        return _rs.apply_advanced(_rs.quality_mode(self.quality, self.estimator, self.fields, self._particles,
                                                   self.precision, self.dim, run_mode=self.run_mode), self)

    def _particles(self) -> int | None:
        """The input's particle count, where its HDF5 header says (else None)."""
        if not self.input_file or self.input_type != 105:
            return None
        return hdf5_particle_count(Path(self.input_file).expanduser())

    def name(self) -> str:
        return self.output_name.strip() or (Path(self.input_file).stem if self.input_file else "output")

    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser() / self.name()

    @property
    def sim(self) -> str:           # the run panel's "choose a simulation" test
        return self.input_file

    @property
    def data_root(self) -> str:
        return str(Path(self.output_dir).expanduser())

    @property
    def snapshots(self) -> list[int]:
        return [0]

    def binary(self) -> Path:
        return _rs.REPO_ROOT / _rs.binary_name(self.estimator, self.precision, self.dim)

    def command(self) -> list[str]:
        return self._own_command() + _extra_options(self.extra_args)

    def _own_command(self) -> list[str]:
        """The command the tab's own controls make (command() appends the extra options); the quality's settings
        from mode()."""
        m = self.mode()
        ps = self.estimator == "ps"
        three = self.dim != 2           # a 2D run: no GPU, no parallel insertion (both 3D only)
        # extra options that name the device ('--gpu', '--gpu-share') set it alone: the tab's own '--gpu' would repeat
        # theirs (the binary takes each option once), and without theirs it would move the run behind their back
        gpu = m.gpu and three and not self._extra_device_flags()
        out = self.output_root()
        a = [str(self.binary()), str(Path(self.input_file).expanduser()), str(out),
             "--grid", str(self.grid), "--input-format", str(self.input_type), "--mpc-unit", f"{self.mpc_unit:g}"]
        # the machine use, always named (the binary's own default is performance: every core, long GPU buffers)
        a += ["--run-mode", self.run_mode]
        if self.periodic:
            a.append("--periodic")
        if len(self.box) == 2 * self.dim:
            a += ["--box"] + [f"{v:g}" for v in self.box]
        fields = list(self.fields) if ps else _rs.dtfe_fields(self.fields)
        if self.scalar_dataset and "scalar_a" not in fields:
            fields.append("scalar_a")
        a += ["--field", *fields]
        if any(f in _rs.WEB_FIELDS for f in fields):      # the scripts' threshold, not the binary's 0.0
            a += ["--lambda-th", f"{self.lambda_th:g}"]
        if self.scalar_dataset:
            a += ["--scalar-dataset", self.scalar_dataset]
        if self.tess_cache:
            a += ["--tessellation-cache", str(Path(self.tess_cache).expanduser())]
        if ps:
            # the cell average named either way (the binaries share both names), never left to the binary's default:
            # the run log then says which it was to any reader, and the count only beside the sampled deposit
            if m.deposit == "exact":
                a.append("--exact")
            else:
                a += ["--sampled", "--subsamples", str(m.nsub)]
            if self.lagrangian == "separate" and self.lagrangian_file:
                a += ["--ps-lagrangian-input", str(Path(self.lagrangian_file).expanduser())]
            # the velocity gradient's route (D13): the face gradient where the run qualifies; a run that does not
            # qualify names no route (the binary keeps the stream average and says why), nor does a 2D run or one
            # whose extra options name a route themselves
            route = _rs.face_gradient_env(self) if three and not self._extra_names_route() else ""
            for on, flag in ((gpu, "--gpu"), (self.vertex_mass, "--ps-vertex-mass"),
                             (m.volume_weighted, "--ps-volume-weighted"), (self.caustics, "--ps-caustics"),
                             (self.caustic_cusps, "--ps-caustic-cusps"),
                             (route == "1", "--face-gradient"), (route == "0", "--per-tet-gradient"),
                             (m.parallel_triangulation and three, "--parallel-triangulation")):
                if on:
                    a.append(flag)
            # the share of a partitioned exact GPU deposit: the mode's ('auto', the machine use's, as the drivers'
            # GPU_SHARE default; a Device override's '1' or 'mixed', runspec.apply_advanced)
            if gpu and m.deposit == "exact":
                a += ["--gpu-share", m.gpu_share]
            if not self.periodic and self.alpha_shape != 3.0:
                a += ["--ps-alpha-shape", f"{self.alpha_shape:g}"]
            if m.linear_deposit:
                a.append("--ps-linear-deposit")
            # the merge order always named, as the drivers' PS_MERGE does (R7, docs/cleanup-plan.md): a GIVEN index
            # order keeps the memory gate on under the auto-tuner's speed split (a run of fewer than 2e6 particles),
            # which the binary's default order alone leaves off. Left to the extra options when they name a merge
            # themselves ('--ps-merge fixed-point'): the tab's own would repeat the option, which the binary refuses
            if not m.ordered_merge:
                a += ["--ps-merge", "completion-order"]
            elif not self._extra_names_merge():
                a += ["--ps-merge", "index-order"]
            if self.halo_release > 0:
                a += ["--ps-halo-release", _rs._num(self.halo_release)]
            if len(self.window) == 2 * self.dim:
                a += ["--grid-window"] + [_rs._num(v) for v in self.window]
        else:
            if gpu:
                a.append("--gpu")
            # the cell average named either way, as above (the sampled one at the binary's own count)
            a.append("--exact" if m.deposit == "exact" else "--sampled")
            # the padding: the binary's own 5 mean spacings leave wrong values within a few spacings of
            # a box face (or of the --box region's faces); 15, run_dtfe.sh's and the standard server's value
            # (dtfelib.estimator.STANDARD_PADDING). PS-DTFE pads in Lagrangian space by its own rules: nothing added
            a += ["--padding", f"{STANDARD_PADDING:g}"]
        part, conc = _rs.job_partition(self), _rs.job_concurrency(self)     # the Advanced overrides (both binaries)
        if part > 0:
            a += ["--partition"] + [str(part)] * self.dim
        if conc > 0:
            a += ["--max-concurrent", str(conc)]
        if self.scratch_dir:
            a += ["--scratch-dir", str(Path(self.scratch_dir).expanduser())]
        if self.sample_points:
            a += ["--sample-points", str(Path(self.sample_points).expanduser())]
            if self.pts_vel_grad:
                a.append("--pts-vel-grad")
            if ps and self.per_stream:
                a.append("--ps-per-stream")
            if ps and self.per_stream_ids:
                a.append("--ps-per-stream-ids")
            if ps and m.pts_conventions == "legacy":     # 'grid' is the binary's default: no flag for it
                a += ["--ps-point-conventions", "legacy"]
        return a

    def _extra_names_merge(self) -> bool:
        """The extra options give the merge mode themselves: '--ps-merge MODE' (or '--ps-merge=MODE'), or one of the
        older switches the binary still takes ('--ordered-merge', '--completion-order-merge', '--fixed-point-merge'),
        each read through the run-log readers' table (canonical_flag), as problems() reads them."""
        return any(_runlog.canonical_flag(t.split("=", 1)[0]).split()[0] == "--ps-merge"
                   for t in _extra_options(self.extra_args) if t.startswith("--"))

    def _extra_names_route(self) -> bool:
        """The extra options name the velocity gradient's route themselves ('--face-gradient' or '--per-tet-gradient'):
        the tab's own would repeat or contradict it, which the binary refuses."""
        return any(t.split("=", 1)[0] in ("--face-gradient", "--per-tet-gradient")
                   for t in _extra_options(self.extra_args) if t.startswith("--"))

    def _extra_device_flags(self) -> frozenset:
        """The device options the extra options give themselves, canonical: '--gpu' (or its older '--ps-gpu') and,
        for PS-DTFE, '--gpu-share' (any value; standard DTFE has no such option, which problems() calls an error).
        Where they give one, the tab adds neither of its own, a Device setting is not passed (runspec._refusal) and
        the 'Runs on' line says the extra options set it (runspec.runs_on_text)."""
        device = ("--gpu", "--gpu-share") if self.estimator == "ps" else ("--gpu",)
        return frozenset(n for t in _extra_options(self.extra_args) if t.startswith("--")
                         for n in [_runlog.canonical_flag(t.split("=", 1)[0]).split()[0]]
                         if n in device)

    def _extra_linear_deposit(self) -> bool:
        """The extra options ask for the linear deposit ('--ps-linear-deposit'), which the program refuses beside the
        volume-weighted velocities the tab always passes for PS-DTFE."""
        return any(_runlog.canonical_flag(t.split("=", 1)[0]).split()[0] == "--ps-linear-deposit"
                   for t in _extra_options(self.extra_args) if t.startswith("--"))

    def memory_key(self) -> str:
        return _rs._memory_key(self, ("output_dir", "output_name", "demo_n", "demo_file", "lambda_th", "tess_cache"))

    def demo_pending(self) -> bool:
        """The demo snapshot is to be generated first: the demo's own input path, not on disk yet."""
        f = Path(self.input_file).expanduser() if self.input_file else None
        return (self.demo_n > 0 and bool(self.demo_file) and f is not None
                and f == Path(self.demo_file).expanduser() and not f.is_file())

    def check_step(self) -> _rs.Step:
        """The same binary call in report mode: its auto-tuner's split and memory prediction for this
        file, nothing computed or written (the report line parses as snapshot 0)."""
        return _rs.Step(f"memory check {Path(self.input_file).name or '?'}", self.command() + ["--auto-tune-report"],
                        self.run_env())

    def run_env(self) -> dict[str, str]:
        """The binary's environment (the cache is an argument here, not TESSELLATION_CACHE): the Advanced memory
        budget, DTFE_MEM_BUDGET_GB, when set (runspec._run_option_env), and DTFE_FACE_GRADIENT pinned
        empty (the binary reads that as unset), for both estimators: a standard run names no gradient route, nor does
        a phase-space run that does not qualify for the face gradient (or a 2D one), so a 0 exported in the login
        shell cannot turn it to the deposit's own gradient behind the command shown, nor a 1 make the binary refuse a
        sampled run (a route the command names beats the variable anyway)."""
        env = {k: v for k, v in _rs._run_option_env(self).items() if k not in _rs.TESS_CACHE_KEYS}
        env["DTFE_FACE_GRADIENT"] = ""
        return env

    def steps(self) -> list[_rs.Step]:
        out = []
        if self.demo_pending():
            demo = Path(self.input_file).expanduser()
            out.append(_rs.Step("create the demo folder", ["mkdir", "-p", str(demo.parent)], {}))
            out.append(_rs.Step(f"generate the demo snapshot ({self.demo_n}{'²' if self.dim == 2 else '³'} particles, "
                                "crossed waves)",
                                [_rs.PYTHON, str(_rs.DEMO_GENERATOR), "--out", str(demo), "--n", str(self.demo_n),
                                 "--box", "100", "--amplitude-factor", "1.5", "--crossed-waves"]
                                + (["--dim", "2"] if self.dim == 2 else []), {}))
        root = self.output_root()
        if not root.parent.is_dir() and str(root.parent) != str(Path(self.input_file).expanduser().parent):
            out.append(_rs.Step("create the output folder", ["mkdir", "-p", str(root.parent)], {}))
        out.append(_rs.Step(f"{self.binary().name} {Path(self.input_file).name or '?'}", self.command(), self.run_env(),
                            tee=str(root) + ".runlog"))
        return out

    def settings_sidecar(self) -> dict:
        """What the Explore tab needs to query this snapshot again (written as '<name>.gui.json').
        'box' stays in the file's units (what --box takes); 'box_mpc' is the box in Mpc, the unit of
        the grids and of the server's coordinates -- from --box, else the Gadget header's BoxSize."""
        d = self.to_dict()
        f = Path(self.input_file).expanduser()
        d["input_file"] = str(f)
        if self.lagrangian_file:
            d["lagrangian_file"] = str(Path(self.lagrangian_file).expanduser())
        if self.lagrangian != "separate":
            d["lagrangian_file"] = ""
        d["box_mpc"] = []
        if len(self.box) == 2 * self.dim and self.mpc_unit > 0:
            d["box_mpc"] = [v / self.mpc_unit for v in self.box]
        elif f.is_file() and self.input_type == 105 and self.mpc_unit > 0:
            b = hdf5_boxsize(f)
            if b:
                d["box_mpc"] = [0.0, b / self.mpc_unit] * self.dim
        # the header's redshift: Explore scales the u-unit velocities by sqrt(a) with it (without it a z > 0
        # snapshot's velocities show 1/sqrt(a) too large); None = no scaling
        d["redshift"] = hdf5_redshift(f) if f.is_file() and self.input_type == 105 else None
        return d

    def problems(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        ps = self.estimator == "ps"
        f = Path(self.input_file).expanduser() if self.input_file else None
        if f is None:
            return [("error", "no snapshot file chosen")]
        demo_pending = self.demo_pending()
        if not f.is_file() and not demo_pending:
            out.append(("error", f"snapshot file not found: {f}"))
        if self.input_type not in _rs.INPUT_TYPES:
            out.append(("error", f"unknown input type {self.input_type}"))
        d = self.dim
        if d not in _rs.DIMS:
            return out + [("error", f"unknown dimension {d} (2 or 3)")]
        found = snapshot_dim(f, self.input_type) if f.is_file() else None
        if found is not None and found != d:
            out.append(("error", f"this snapshot is {found}D (its coordinates have {found} columns): "
                                 f"set the dimensions to {found}D"))
        if ps and self.input_type != 105:
            out.append(("error", "the phase-space estimator needs a Gadget HDF5 snapshot (particle IDs and "
                                 "initial positions); use standard DTFE for this format"))
        if self.mpc_unit <= 0:
            out.append(("error", "the length unit must be positive (1 Mpc in the file's units)"))
        if self.alpha_shape < 0:
            out.append(("error", "the alpha shape takes a number of mean particle spacings >= 0 (0 keeps the whole hull)"))
        out += _rs._precision_problems(self.estimator, self.precision, d)
        if self.box:
            if len(self.box) != 2 * d or any(self.box[2 * i + 1] <= self.box[2 * i] for i in range(d)):
                names = "xlo xhi ylo yhi" + (" zlo zhi" if d == 3 else "")
                out.append(("error", f"the box needs {2 * d} numbers, {names}, each hi > lo"))
            elif min(self.box) < 0:
                out.append(("error", "box corners must be >= 0 (the binary's option parser reads a negative "
                                     "number as an option): shift the coordinates, or leave the box to the header"))
        if ps and f.is_file() and self.input_type == 105:
            has_ic = hdf5_has(f, "InitialCoordinates")
            if self.lagrangian == "file" and has_ic is False:
                out.append(("error", "this snapshot stores no 'InitialCoordinates': choose the initial-"
                                     "conditions file, or use standard DTFE"))
            if self.lagrangian == "separate":
                lf = Path(self.lagrangian_file).expanduser() if self.lagrangian_file else None
                if lf is None or not lf.is_file():
                    out.append(("error", "choose the initial-conditions file (the particles' Lagrangian positions)"))
            if self.scalar_dataset and hdf5_has(f, self.scalar_dataset) is False:
                out.append(("error", f"no dataset '{self.scalar_dataset}' in the snapshot's particle groups"))
        if self.scalar_dataset and self.input_type != 105:
            out.append(("error", "a per-particle scalar dataset needs Gadget HDF5 input"))
        if not 8 <= self.grid <= (65536 if d == 2 else 4096):
            out.append(("error", f"grid {self.grid}{'²' if d == 2 else '³'} is outside 8..{65536 if d == 2 else 4096}"))
        if ps and not self.fields:
            out.append(("error", "no fields selected"))
        if not ps and not _rs.dtfe_fields(self.fields):
            out.append(("error", "no field standard DTFE computes is ticked (the velocity dispersion is PS-DTFE's)"))
        name = self.name()
        if re.search(r"[\s/]", name):
            out.append(("error", "output name must be a single word (no spaces or '/')"))
        od = Path(self.output_dir).expanduser()
        if not self.output_dir.strip():
            out.append(("error", "choose an output folder"))
        elif "Mobile Documents" in str(od):
            out.append(("warning", "the output folder is in iCloud Drive: grids are large, and syncing them "
                                   "is slow, so a local folder is better"))
        if not self.binary().is_file() and self.precision == "single":
            how = f"'make {self.binary().name}'" if d == 3 else f"'{_rs.build_hint('single', 2)}'"
            out.append(("error", f"{self.binary().name} is not built (Setup, or {how})"))
        out += _rs._scratch_problems(self.scratch_dir)
        out += _rs._run_option_problems(self)
        out += _rs.advanced_problems(self)
        if self.sample_points:
            if not Path(self.sample_points).expanduser().is_file():
                out.append(("error", f"sample-points file not found: {self.sample_points}"))
            # (also for a split standard-DTFE run: the binary answers the points per partition, each from the
            # tetrahedra it owns)
        if self.window:
            if not ps:
                out.append(("info", "the window is a PS-DTFE option: this standard-DTFE run ignores it"))
            elif (len(self.window) != 2 * d or not all(math.isfinite(v) for v in self.window)
                  or any(self.window[2 * i + 1] <= self.window[2 * i] for i in range(d))):
                out.append(("error", f"the window needs {2 * d} numbers in Mpc, each upper > lower"))
            elif any(f in _rs.WEB_FIELDS for f in self.fields):
                out.append(("error", "the T-web / V-web classes need the whole periodic grid: not with a window"))
        if (self.per_stream or self.per_stream_ids) and not self.sample_points:
            out.append(("warning", "the per-stream outputs apply to sample points: choose a points file (or they do "
                                   "nothing)"))
        if not ps and (self.per_stream or self.per_stream_ids or self.halo_release > 0):
            out.append(("info", "per-stream outputs and the halo release are PS-DTFE options: this standard-DTFE run "
                                "leaves them out"))
        # what the quality leaves out (the derivatives' note not where the extra options name a route: that one is
        # the run's own, and the binary refuses a named face gradient that cannot run)
        if ps:
            out += _rs._quality_problems(self, derivatives=not self._extra_names_route())
        if (self.output_root().parent / (self.name() + ".a_den")).exists():
            out.append(("info", f"'{name}' already has outputs in this folder: they will be overwritten"))
        if self.extra_args.strip():
            try:
                extra = shlex.split(self.extra_args)
            except ValueError as e:
                out.append(("error", f"the extra options do not parse ({e}): close the quotes"))
            else:
                import results as R     # canonical_flag: the binaries' older option names (as runspec.py imports it)

                def names(tokens):      # the option names, canonical, no '=value': '--lambda_th=0.4' -> '--lambda-th';
                    # a merge switch is '--ps-merge' (canonical_flag gives its 'option value' pair; the run copy migrates
                    # the switch to that pair, so beside the tab's own '--ps-merge' the binary would get it twice).
                    # '--exact' and '--sampled' are one setting, the cell average, which the tab always names: either
                    # one among the extra options repeats it (the other one the binary refuses beside it). The
                    # machine use ('--run-mode', always named) is one setting, whatever value the extra options give
                    # it. (The device, '--gpu' and '--gpu-share', the tab leaves to extra options that name it.)
                    found = {R.canonical_flag(t.split("=", 1)[0]).split()[0] for t in tokens if t.startswith("--")}
                    return {"--exact / --sampled" if n in ("--exact", "--sampled") else n for n in found}
                twice = sorted(names(extra) & names(self._own_command()))
                if twice:
                    out.append(("error", f"the extra options repeat {', '.join(twice)}, which the tab sets already "
                                         "(the program takes each option once): change it on the tab"))
                if ps and self._extra_linear_deposit():
                    out.append(("error", "the extra options' --ps-linear-deposit cannot run beside the "
                                         "volume-weighted velocities the tab always passes (--ps-volume-weighted): "
                                         "the program refuses the pair, so take it out"))
                if self._extra_device_flags() == {"--gpu-share"}:
                    out.append(("warning", "the extra options give --gpu-share without --gpu: the run is on the CPU, "
                                           "where no share applies (add --gpu for the GPU)"))
                if not ps and "--gpu-share" in names(extra):
                    out.append(("error", "the extra options' --gpu-share is a PS-DTFE option, which standard DTFE "
                                         "refuses at the start: take it out"))
        out += _rs._busy_problems()
        return out

    def to_dict(self) -> dict:
        return _rs._with_advanced(self, asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "CustomSpec":
        """A saved job (the launcher's settings, a queued job, a run's '.gui.json' sidecar for 'Load settings'), its
        flag spellings brought up to date and an older one's deposit read as its quality (migrate_saved): every load
        path goes through here. Bad overrides are dropped (__post_init__)."""
        return cls(**{k: v for k, v in migrate_saved(d).items() if k in cls.__dataclass_fields__})


def migrate_extra_args(text: str, estimator: str = "ps", dim: int = 3) -> str:
    """The extra-options line with the binaries' older option names in their canonical spelling (P6.7, the run-log
    readers' table: '--MpcUnit 1 --ps-window 0 5 0 5 0 5' -> '--mpc-unit 1 --grid-window 0 5 0 5 0 5', the merge
    switches -> '--ps-merge MODE', '--ps-stream-density dtfe' -> 'linear'); the rest, quoting included, untouched.
    Standard DTFE: '-s N' / '--samples N' -> '--subsamples k' when N = k^dim, '-m'/'--method' -> '--average-method';
    PS-DTFE keeps those spellings (they have no effect there, and '--subsamples' would change the run).
    Idempotent: a canonical line comes back as it is."""
    if not text or not text.strip():
        return text
    return _runlog.canonical_command(text, dim, averaging=estimator == "dtfe")


def migrate_saved(d: dict) -> dict:
    """A saved CustomSpec dict brought up to date (idempotent, so no one-time flag is needed): the extra options' flag
    names (P6.7, migrate_extra_args) and the two modes (runspec.migrate_quality: an older deposit becomes the quality;
    from_dict then drops the settings the modes replaced)."""
    d = _rs.migrate_quality(d)
    if isinstance(d.get("extra_args"), str):
        try:
            dim = int(d.get("dim", 3))
        except (TypeError, ValueError):
            dim = 3
        d["extra_args"] = migrate_extra_args(d["extra_args"], str(d.get("estimator", "ps")), dim)
    return d


def _extra_options(text: str) -> list[str]:
    """The extra-options line split as a shell splits it; [] when its quotes do not close (problems() says so)."""
    try:
        return shlex.split(text)
    except ValueError:
        return []


def snapshot_dim(path, input_type: int = 105) -> int | None:
    """2 or 3: the dimension of a snapshot file, None when the file cannot tell. Gadget HDF5: the
    columns of its particles' Coordinates (a 2D snapshot holds (N, 2) datasets, as
    tests/generate_ps_test_data.py --dim 2 writes); the text formats: the box on the second line
    holds 2 x dim numbers (io/text_io.cc); Gadget binary carries no dimension (None)."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    if input_type == 105:
        try:
            import h5py
            with h5py.File(p, "r") as h:
                for name in sorted(h.keys()):
                    c = h[name].get("Coordinates") if name.startswith("PartType") else None
                    if c is not None and len(c.shape) == 2 and c.shape[1] in (2, 3):
                        return int(c.shape[1])
        except Exception:
            return None
        return None
    if input_type in (111, 112):
        try:
            with open(p, errors="replace") as fh:
                fh.readline()
                n = len(fh.readline().replace(",", " ").split())
        except OSError:
            return None
        return n // 2 if n in (4, 6) else None
    return None


def hdf5_boxsize(path) -> float | None:
    """The Gadget header's BoxSize of an HDF5 snapshot (file units), or None."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            b = float(h["Header"].attrs["BoxSize"])
        return b if b > 0 else None
    except Exception:
        return None


def hdf5_redshift(path) -> float | None:
    """The Gadget header's Redshift of an HDF5 snapshot, or None (no file, no header, no such attribute)."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            hd = h["Header"].attrs
            return float(hd["Redshift"]) if "Redshift" in hd else None
    except Exception:
        return None


_COUNT_CACHE: dict[str, int | None] = {}


def hdf5_particle_count(path) -> int | None:
    """The particles of an HDF5 snapshot (the header's NumPart_Total, all types), or None. Kept per file version
    (path, modification time, size): a Custom job's device asks for it on every check of a Fast run
    (CustomSpec.mode())."""
    try:
        st = Path(path).stat()
    except (OSError, TypeError, ValueError):
        return None
    key = f"{path}:{st.st_mtime_ns}:{st.st_size}"
    if key not in _COUNT_CACHE:
        try:
            import h5py
            with h5py.File(path, "r") as h:
                n = int(sum(int(x) for x in h["Header"].attrs["NumPart_Total"]))
            _COUNT_CACHE[key] = n if n > 0 else None
        except Exception:
            _COUNT_CACHE[key] = None
    return _COUNT_CACHE[key]


def hdf5_has(path, dataset: str) -> bool | None:
    """Whether any /PartTypeN group of an HDF5 snapshot holds 'dataset' (None: cannot tell)."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            groups = [k for k in h.keys() if k.startswith("PartType")]
            return any(dataset in h[g] for g in groups) if groups else False
    except Exception:
        return None


def demo_spec(output_dir=None) -> CustomSpec:
    """The first-run demo: a synthetic 32^3 crossed-wave snapshot (streams 1/3/9/27) and a quick
    PS-DTFE run on it at the default quality, Exact, with no download (its device is the quality's,
    CustomSpec.mode())."""
    od = Path(output_dir).expanduser() if output_dir else CUSTOM_OUT_DEFAULT / "demo"
    fields = ["density_a", "velocity_a", "dispersion_a", "divergence_a"]
    return CustomSpec(input_file=str(od / "crossed_waves_32.hdf5"), input_type=105, mpc_unit=1.0,
                      periodic=True, estimator="ps", lagrangian="file", grid=64,
                      fields=fields, vertex_mass=False, caustics=True,
                      output_dir=str(od), output_name="demo", demo_n=32, demo_file=str(od / "crossed_waves_32.hdf5"))
