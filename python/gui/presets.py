"""Presets and help text for the GUI's run settings. Pure Python (no Qt).

  PRESETS / preset_names / apply_preset   named starting points for a Grids / custom-snapshot run
                                          (built in, plus the user's own, kept in the GUI's settings)
  OPTION_TEXT                             plain-language label + the binary flag behind each option
  LABELS                                  the window's words that several tabs share (group titles, buttons)
  flag_help(binary)                       {flag: description} parsed from the binary's own
                                          '--full-help', so a tooltip can never drift from the code:
                                          short_tooltip (its first sentence, at most two lines) is the
                                          tooltip, tooltip (all of it) the control's What's This
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from results import canonical_flag as _canonical_flag   # a binary's older option names (pure Python, no Qt)

REPO_ROOT = Path(__file__).resolve().parents[2]

# the production fields: the drivers' default (runspec.DEFAULT_FIELDS; tests/py_gui_runspec_test.py checks it)
_PRODUCTION = ["density_a", "velocity_a", "divergence_a", "shear_a"]

# name -> (one-line description, settings). The settings are RunSpec / CustomSpec fields; a
# preset leaves every field it does not name as it was. 'slice_plane': "largest" = the largest
# image plane found for the simulation (none: no slice). 'deposit' is the cell average of both estimators (the
# exact one is the production setting of both since 2026-11-09); 'deposit_dtfe', when given, is the one a
# standard-DTFE job gets instead (no built-in preset needs it any more; apply_preset still reads it).
BUILTIN_PRESETS: dict[str, tuple[str, dict]] = {
    "Quick look": ("128³ grid, density and velocity only: about a minute on a laptop-sized "
                   "snapshot, for checking that everything works",
                   {"grid": 128, "nsub": 1, "deposit": "sampled", "fields": ["density_a", "velocity_a"],
                    "vertex_mass": True, "volume_weighted": True, "caustics": False,
                    "caustic_cusps": False, "slice_plane": ""}),
    "Standard": ("512³ grid, the production fields (density, velocity, divergence, shear) with the "
                 "literature-standard weighting",
                 {"grid": 512, "nsub": 3, "deposit": "exact", "fields": list(_PRODUCTION),
                  "vertex_mass": True, "volume_weighted": True, "caustics": False,
                  "caustic_cusps": False, "slice_plane": ""}),
    "Publication": ("Standard + caustic stratification + a hi-res point-evaluated slice (the "
                    "largest image plane of the simulation) for figures",
                    {"grid": 512, "nsub": 3, "deposit": "exact", "fields": list(_PRODUCTION),
                     "vertex_mass": True, "volume_weighted": True, "caustics": True,
                     "caustic_cusps": False, "slice_plane": "largest", "slice_vel_grad": True}),
    "Exact deposit": ("256³ grid, density and velocity with the exact tetrahedron-cell clipping (the default "
                      "cell average): noise-free cell values. PS-DTFE runs it on the CPU (9 s for a 0.26M-particle "
                      "snapshot; at 512³ TNG100-3-Dark 1133 s, faster than the sampled GPU deposit's 1260 s), "
                      "standard DTFE's exact average on the GPU",
                      {"grid": 256, "deposit": "exact", "fields": ["density_a", "velocity_a"],
                       "vertex_mass": True, "volume_weighted": True, "slice_plane": ""}),
}


def preset_names(user: dict) -> list[str]:
    return list(BUILTIN_PRESETS) + [n for n in user if n not in BUILTIN_PRESETS]


def preset(name: str, user: dict) -> tuple[str, dict] | None:
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    if name in user:
        return "your own preset", dict(user[name])
    return None


PRESET_KEYS = ("grid", "nsub", "deposit", "fields", "gpu", "vertex_mass", "volume_weighted",
               "caustics", "caustic_cusps", "slice_plane", "slice_vel_grad", "estimator")


def capture(spec) -> dict:
    """The preset-able settings of a RunSpec / CustomSpec (for 'Save as preset')."""
    d = spec.to_dict()
    return {k: d[k] for k in PRESET_KEYS if k in d}


def apply_preset(spec, settings: dict, planes=()) -> list[str]:
    """Sets the named fields on spec (in place); returns what was changed, for the status line.
    planes: the simulation's image-plane files, for 'slice_plane': 'largest'."""
    changed = []
    known = spec.to_dict()
    if settings.get("estimator", known.get("estimator")) == "dtfe" and "deposit_dtfe" in settings:
        settings = dict(settings, deposit=settings["deposit_dtfe"])     # standard DTFE: its cell average
    for k, v in settings.items():
        if k not in known:
            continue
        if k == "slice_plane" and v == "largest":
            v = str(max(planes, key=lambda p: Path(p).stat().st_size)) if planes else ""
        if known[k] != v:
            setattr(spec, k, list(v) if isinstance(v, list) else v)
            changed.append(k)
    return changed


# ---------------------------------------------------------------------- labels
# option -> (plain-language label, flag). The flag's full description comes from the binary.
OPTION_TEXT = {
    "gpu":             ("Use the GPU for the grid deposit", "--gpu"),
    "gpu_dtfe":        ("Use the GPU for the grid interpolation", "--gpu"),      # (its help in the standard binary)
    "vertex_mass":     ("Tetrahedron masses from the particles (needed for TNG initial conditions)",
                        "--ps-vertex-mass"),
    "volume_weighted": ("Volume-weighted velocities (the literature convention)", "--ps-volume-weighted"),
    "caustics":        ("Mark caustics: folds and how many axes collapsed", "--ps-caustics"),
    "caustic_cusps":   ("Also mark cusps and swallowtails (slower, CPU)", "--ps-caustic-cusps"),
    "linear_deposit":  ("Linear density inside each tetrahedron (excludes volume weighting)",
                        "--ps-linear-deposit"),
    "sample_points":   ("Also evaluate the fields at these points", "--sample-points"),
    "window":          ("Deposit only this window of the grid", "--grid-window"),
    "halo_release":    ("Release halo interiors above this density (rho/rho_bar; off = never)", "--ps-halo-release"),
    "stream_density":  ("Each stream's density at a point", "--ps-stream-density"),
    "per_stream":      ("The point values stream by stream (ragged files)", "--ps-per-stream"),
    "per_stream_ids":  ("... with each stream's tetrahedron particle IDs", "--ps-per-stream-ids"),
    "slice_vel_grad":  ("Velocity gradient at each slice point (divergence, shear, vorticity maps)",
                        "--pts-vel-grad"),
    "pts_conventions": ("Plane conventions", "--ps-point-conventions"),     # the image plane's estimators
    "exact":           ("Exact: analytic tetrahedron-cell clipping (noise-free; the default, best on the CPU)",
                        "--exact"),
    "parallel_triangulation": ("Parallel triangulation (faster; repeats differ at float rounding)",
                               "--parallel-triangulation"),
    "ordered_merge":   ("Merge the partitions in index order (bit-identical repeats; the default)",
                        "--ps-merge"),
    "alpha_shape":     ("Alpha shape: drop Lagrangian tetrahedra wider than this many mean particle spacings "
                        "(0 keeps the whole convex hull)", "--ps-alpha-shape"),
    # the two cell averages: their help is the binaries' '--exact' / '--sampled' text (a binary from before
    # 2026-11-09 knows '--exact' by its old name, '--exact-average', which flag_help files under both)
    "sampled":         ("Sampled: nSub³ points per cell", "--sampled"),
    "sampled_dtfe":    ("Sampled: the Monte-Carlo mean over points in each cell (faster on the CPU)", "--sampled"),
    "exact_dtfe":      ("Exact: each cell the exact average of the interpolated field (noise-free; the default)",
                        "--exact"),
    "periodic":        ("The box is periodic (a cosmological simulation)", "--periodic"),
    "scalar":          ("Also interpolate a per-particle value", "--scalar-dataset"),
}


def label(key: str) -> str:
    return OPTION_TEXT[key][0]


def noted_label(key: str, note: str = "") -> str:
    """An option's label, or, when this build or run cannot use the option ('note' says why: 'this build has no TBB:
    ignored'), its name alone with the note: 'Parallel triangulation (this build has no TBB: ignored)'. The label's
    own explanation in brackets goes, so the tick never gets wider than its form (the forms scroll only vertically)."""
    if not note:
        return label(key)
    return f"{label(key).split(' (')[0]} ({note})"


# ---------------------------------------------------------------------- the window's own words
# One table for the texts several tabs share: sentence case, nouns for groups, verbs for buttons, '…' only where a
# dialog opens. The option labels above are OPTION_TEXT's; the field names are the Explore tab's (grids.FIELD_LABELS).
LABELS = {
    # group titles
    "preset": "Start from a preset",
    "estimator": "Estimator",
    "estimator_grid": "Estimator and grid",     # Grids: the estimator choice, then the grid and its deposit
    "grid_deposit": "Grid and cell average",    # Custom snapshot and Pipeline (their estimator is chosen elsewhere)
    "fields": "Fields",
    "ps_options": "Phase-space options",
    "slice": "Hi-res slice (point evaluation)",
    "resources": "Resources and output",
    "advanced": "Advanced",
    # the snapshot lists' quick-selection buttons
    "select_all": "All",
    "select_none": "None",
    "select_not_run": "Not yet run",            # Grids: the snapshots without grids of this run's kind
    "select_sliced": "Sliced",                  # Pipeline: the snapshots with an image plane
    "select_with_grids": "With grids",          # Plots: the snapshots that have grids to plot
    "select_not_merged": "Not merged",          # Data: the snapshots not merged into one file yet
    # buttons
    "save_preset": "Save as preset…",
    "delete_preset": "Delete",
    "choose_preset": "choose a preset",
    "setup": "Setup…",
    # the collapsed groups' summaries
    "auto_plan": "auto plan",
    "auto_concurrency": "auto concurrency",
    "all_off": "all off",
}

# the expert options of the Advanced groups, as their one-line summary names them
ADVANCED_SHORT = {
    "caustic_cusps": "cusps and swallowtails",
    "linear_deposit": "linear density",
    "per_stream": "stream by stream",
    "per_stream_ids": "stream IDs",
    "render_fixed_range": "fixed density range",
    "render_force": "re-render",
}


# ---------------------------------------------------------------------- help from the binary
_OPT = re.compile(r"^  (?:-\w \[ )?--([A-Za-z0-9_\-]+)(?: \])?(?: arg(?: \(=[^)]*\))?)?\s{2,}(\S.*)$")
_OPT_NODESC = re.compile(r"^  (?:-\w \[ )?--([A-Za-z0-9_\-]+)(?: \])?(?: arg(?: \(=[^)]*\))?)?\s*$")
_HELP_CACHE: dict[str, dict[str, str]] = {}


def parse_help(text: str) -> dict[str, str]:
    """{flag (with '--'): description} from boost::program_options help output: an option line
    '  -g [ --grid ] arg      text' followed by continuation lines indented to the text column."""
    out: dict[str, str] = {}
    cur, col = None, 0
    for line in text.splitlines():
        m = _OPT.match(line)
        if m:
            cur, col = "--" + m.group(1), line.index(m.group(2))
            out[cur] = m.group(2).strip()
            continue
        m = _OPT_NODESC.match(line)
        if m:                           # a long option name pushes its text onto the next line
            cur, col = "--" + m.group(1), -1
            out[cur] = ""
            continue
        if cur and line.strip() and (col < 0 or line[:col].strip() == ""):
            if col < 0:
                col = len(line) - len(line.lstrip())
            out[cur] = (out[cur] + " " + line.strip()).strip()
        elif not line.strip() or not line.startswith(" "):
            cur = None
    return out


def flag_help(binary: str = "PS-DTFE") -> dict[str, str]:
    """The binary's '--full-help', parsed (cached per binary and modification time); {} when the
    binary is not built. A binary built before the option names were unified knows only '--full_help' (and lists
    '--ps-gpu', '--ps-exact-deposit' and '--ps-no-overlap' as its own): asked that way when '--full-help' prints
    nothing, its old names read as the canonical ones too (results.canonical_flag)."""
    exe = REPO_ROOT / binary
    try:
        key = f"{exe}:{exe.stat().st_mtime_ns}"
    except OSError:
        return {}
    if key not in _HELP_CACHE:
        parsed: dict[str, str] = {}
        try:
            for opt in ("--full-help", "--full_help"):     # (an older binary refuses the first on stdout)
                parsed = parse_help(subprocess.run([str(exe), opt], capture_output=True, text=True,
                                                   timeout=20, cwd=str(REPO_ROOT)).stdout)
                if parsed:
                    break
        except (OSError, subprocess.TimeoutExpired):
            parsed = {}
        # each entry under the binary's own name AND its canonical one (results.canonical_flag, the readers' permanent
        # table): a launcher that emits an older name still finds it, and so does a reader of the new one
        _HELP_CACHE[key] = dict(parsed)
        for k, v in parsed.items():
            _HELP_CACHE[key].setdefault(_canonical_flag(k), v)
    return _HELP_CACHE[key]


def plain_help(text: str) -> str:
    """The binary's help text as prose for the window: option names without their leading dashes
    ('--partition' reads 'partition') and ' -- ' as a real dash."""
    text = re.sub(r"(?<![\w-])--(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=\w)--(?=\w)", "–", text)          # 'velocity-divergence--density': an en dash
    return text.replace(" -- ", " — ").replace("--", "—")


def tooltip(key: str, binary: str = "PS-DTFE", width: int = 90) -> str:
    """The binary's own description of the option behind 'key', wrapped (the control's What's This text)."""
    if key not in OPTION_TEXT:
        return ""
    text = plain_help(flag_help(binary).get(OPTION_TEXT[key][1], ""))
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


# a sentence ends at '.', '!' or '?' before a space and a capital or an opening bracket -- not after these
_NOT_AN_END = ("e.g.", "i.e.", "cf.", "vs.", "etc.", "approx.")


def first_sentence(text: str) -> str:
    for m in re.finditer(r"[.!?](?=\s+[A-Z(])", text):
        head = text[:m.end()]
        if not head.lower().endswith(_NOT_AN_END):
            return head
    return text


def shorten(text: str, width: int = 90, lines: int = 2) -> str:
    """A text's first sentence in at most 'lines' lines of 'width' (cut with ' …' when longer): a tooltip."""
    import textwrap
    first = first_sentence(text)
    wrapped = textwrap.wrap(first, width, break_on_hyphens=False)
    if len(wrapped) > lines:                    # (room for the ' …' on the last line)
        wrapped = textwrap.wrap(first, width - 2, break_on_hyphens=False)[:lines]
        wrapped[-1] = wrapped[-1].rstrip(" ,;:(") + " …"
    return "\n".join(wrapped)


def short_tooltip(key: str, binary: str = "PS-DTFE", width: int = 90, lines: int = 2) -> str:
    """The tooltip of the option behind 'key': the first sentence of the binary's own description, at most 'lines'
    lines of 'width' (shorten). The whole description is the control's What's This (tooltip())."""
    if key not in OPTION_TEXT:
        return ""
    return shorten(plain_help(flag_help(binary).get(OPTION_TEXT[key][1], "")), width, lines)
