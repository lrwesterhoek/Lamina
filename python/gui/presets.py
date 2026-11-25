"""Presets and help text for the GUI's run settings. Pure Python (no Qt)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# the production fields: the drivers' default (runspec.DEFAULT_FIELDS; tests/py_gui_runspec_test.py checks it)
_PRODUCTION = ["density_a", "velocity_a", "divergence_a", "shear_a"]

# name -> (one-line description, settings). The settings are RunSpec / CustomSpec fields among PRESET_KEYS; a preset
# leaves every field it does not name as it was. 'slice_plane': "largest" = the largest image plane found for the
# simulation (none: no slice). Three built-ins: a first look, the production grids, the production grids for figures.
BUILTIN_PRESETS: dict[str, tuple[str, dict]] = {
    "Quick look": ("Fast, 128³ grid, density and velocity only: a first look, to check that everything works",
                   {"grid": 128, "quality": "fast", "fields": ["density_a", "velocity_a"], "caustics": False,
                    "caustic_cusps": False, "slice_plane": ""}),
    "Standard": ("Exact, 512³ grid, the production fields (density, velocity, divergence, shear)",
                 {"grid": 512, "quality": "exact", "fields": list(_PRODUCTION), "caustics": False,
                  "caustic_cusps": False, "slice_plane": ""}),
    "Publication": ("Standard plus the caustic stratification and a hi-res point-evaluated slice on the largest "
                    "image plane of the simulation, for figures",
                    {"grid": 512, "quality": "exact", "fields": list(_PRODUCTION), "caustics": True,
                     "caustic_cusps": False, "slice_plane": "largest", "slice_vel_grad": True}),
}


def preset_names(user: dict) -> list[str]:
    return list(BUILTIN_PRESETS) + [n for n in user if n not in BUILTIN_PRESETS]


def preset(name: str, user: dict) -> tuple[str, dict] | None:
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    if name in user:
        return "your own preset", dict(user[name])
    return None


# what a preset sets: the run's choices. Not the machine use (run_mode: about this machine, not about the grids) and
# not the Custom tab's initial positions (vertex_mass: a description of the input file, as its format and box are)
PRESET_KEYS = ("grid", "quality", "fields", "caustics", "caustic_cusps", "slice_plane", "slice_vel_grad", "estimator")


def capture(spec) -> dict:
    """The preset-able settings of a RunSpec / CustomSpec (for 'Save as preset')."""
    d = spec.to_dict()
    return {k: d[k] for k in PRESET_KEYS if k in d}


def apply_preset(spec, settings: dict, planes=()) -> list[str]:
    """A setting outside PRESET_KEYS (or one the job does not have) is left out."""
    changed = []
    known = spec.to_dict()
    for k, v in settings.items():
        if k not in PRESET_KEYS or k not in known:
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
    # (the Custom tab's initial positions: perturbed ones take the tetrahedron masses from the particles)
    "vertex_mass":     ("Tetrahedron masses from the particles (needed for TNG initial conditions)",
                        "--ps-vertex-mass"),
    "caustics":        ("Mark caustics: folds and how many axes collapsed", "--ps-caustics"),
    "caustic_cusps":   ("Also mark cusps and swallowtails (slower, CPU)", "--ps-caustic-cusps"),
    "sample_points":   ("Also evaluate the fields at these points", "--sample-points"),
    "window":          ("Deposit only this window of the grid", "--grid-window"),
    "halo_release":    ("Release halo interiors above this density (rho/rho_bar; off = never)", "--ps-halo-release"),
    "per_stream":      ("The point values stream by stream (ragged files)", "--ps-per-stream"),
    "per_stream_ids":  ("... with each stream's tetrahedron particle IDs", "--ps-per-stream-ids"),
    "slice_vel_grad":  ("Velocity gradient at each slice point (divergence, shear, vorticity maps)",
                        "--pts-vel-grad"),
    "run_mode":        ("Machine use", "--run-mode"),     # how hard the run leans on the machine (not in presets)
    "alpha_shape":     ("Alpha shape: drop Lagrangian tetrahedra wider than this many mean particle spacings "
                        "(0 keeps the whole convex hull)", "--ps-alpha-shape"),
    "periodic":        ("The box is periodic (a cosmological simulation)", "--periodic"),
    "scalar":          ("Also interpolate a per-particle value", "--scalar-dataset"),
}


def label(key: str) -> str:
    return OPTION_TEXT[key][0]


# ---------------------------------------------------------------------- the window's own words
# One table for the texts several tabs share: sentence case, nouns for groups, verbs for buttons, '…' only where a
# dialog opens. The option labels above are OPTION_TEXT's; the field names are the Explore tab's (grids.FIELD_LABELS).
LABELS = {
    # group titles
    "preset": "Start from a preset",
    "estimator": "Estimator",
    "estimator_quality": "Estimator and quality",   # Grids: the estimator, the Quality pair, grid, machine use
    "quality_grid": "Quality and grid",             # Custom snapshot and Pipeline (the estimator chosen elsewhere)
    "quality": "Quality",                           # the row of the Quality pair, Exact or Fast
    "fields": "Fields",
    "ps_options": "Phase space",
    "slice": "Hi-res slice (point evaluation)",
    "more": "More",                                 # the folded group: output, start over, folders
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
    binary is not built."""
    exe = REPO_ROOT / binary
    try:
        key = f"{exe}:{exe.stat().st_mtime_ns}"
    except OSError:
        return {}
    if key not in _HELP_CACHE:
        try:
            _HELP_CACHE[key] = parse_help(subprocess.run([str(exe), "--full-help"], capture_output=True, text=True,
                                                         timeout=20, cwd=str(REPO_ROOT)).stdout)
        except (OSError, subprocess.TimeoutExpired):
            _HELP_CACHE[key] = {}
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
