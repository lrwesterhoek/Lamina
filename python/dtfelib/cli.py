"""Shared command-line interface for analysis/plot scripts.

Every script gains the same flags so the whole pipeline is method- and snapshot-agnostic:

    python3 plot/plot_PS_DTFE.py --sim TNG50-3-Dark                # snap 99, method auto
    python3 plot/plot_PS_DTFE.py --sim TNG50-3-Dark --estimator dtfe
    DTFE_SIM=TNG100-3-Dark python3 plot/plot_PS_DTFE.py --snap 50 --smooth 1.5

There is no default simulation: every run names it, with --sim or DTFE_SIM (require_sim).

Scripts call make_fieldset() and get a ready FieldSet plus the parsed args for their own
extra options (pass extra=... to register script-specific flags on the same parser).
"""

from __future__ import annotations

import argparse
import numbers
import os
import sys
from pathlib import Path

from .io import FieldSet

# Same environment variables as the shell side (config.sh): DTFE_DATA_ROOT / DTFE_SIM. Neither has a default: the data
# root is read where data is needed (require_data_root), never at import, so every module imports without it.
# sim_dir() resolves both layouts under it: flat <root>/<sim>, or per family <root>/TNG100/TNG100-3-Dark.
NO_DATA_ROOT = "no data root: set DTFE_DATA_ROOT to the folder that holds the simulations"


class NoSimulation(ValueError):
    """No simulation was named: there is no default, a run gives --sim NAME or DTFE_SIM=NAME."""


class NoDataRoot(ValueError):
    """No data root was named: there is no default, a run sets DTFE_DATA_ROOT (or gives a script's --data-root)."""


_data_root_override = None          # use_data_root(): a script's --data-root for every helper of this process


def require_data_root(root=None) -> Path:
    """The folder that holds the simulations: 'root' when given, else the one use_data_root() set for this process,
    else $DTFE_DATA_ROOT (~ expanded). There is no default; without any of them this raises NoDataRoot."""
    if root:
        return Path(root).expanduser()
    if _data_root_override is not None:
        return _data_root_override
    env = os.environ.get("DTFE_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    raise NoDataRoot(NO_DATA_ROOT)


def require_sim(sim: str | None = None) -> str:
    """The simulation to work on: 'sim' when given, else $DTFE_SIM. There is no default; without either this raises
    NoSimulation, naming both ways to give one and the simulations found under the data root (when one is set)."""
    sim = sim or os.environ.get("DTFE_SIM")
    if sim:
        return sim
    try:
        root = require_data_root()
        found = find_sims(root) if root.is_dir() else []
    except (NoDataRoot, OSError):
        root, found = None, []
    raise NoSimulation("no simulation given: pass --sim NAME or set DTFE_SIM=NAME"
                       + (f" (under {root}: {', '.join(found)})" if found else ""))


# Where every figure goes (config.LOCAL_FIGURES_ROOT, the launcher's figure browser): DTFE_FIGURES_ROOT, else
# python/figures (git-ignored). Not derived from the data root: the launcher hands its data root only to the scripts
# it runs, so the launcher and its scripts would look in different places. The tests set DTFE_FIGURES_ROOT to a
# temp dir.
FIGURES_ROOT = Path(os.environ.get("DTFE_FIGURES_ROOT") or Path(__file__).resolve().parents[1] / "figures")
DEFAULT_SNAP = 99


def sim_dir(sim: str, root=None) -> Path:
    """Directory of simulation `sim` under `root` (default require_data_root()), in either layout: flat
    <root>/<sim>, or per family <root>/<family>/<sim> with <family> the name up to its first '-'
    (TNG100-3-Dark -> TNG100). An existing directory wins; a simulation not on disk yet goes into
    its family folder when that exists, else flat. Mirrors sim_dir in scripts/config.sh."""
    root = require_data_root(root)
    flat = root / sim
    family = root / sim.split("-", 1)[0]
    if flat.is_dir():
        return flat
    if family != flat and family.is_dir():
        return family / sim
    return flat


def snapdir_path(sim: str, snap, root=None) -> Path:
    """The folder of snapshot 'snap' of simulation 'sim': sim_dir(sim, root) / 'snapdir_NNN'. An integer is
    zero-padded to three digits (99 -> snapdir_099); a string is used as given (config's ladder keys are already
    '099'). 'root' as sim_dir: None = require_data_root()."""
    name = f"snapdir_{snap:03d}" if isinstance(snap, numbers.Integral) else f"snapdir_{snap}"
    return sim_dir(sim, root) / name


def snapshot_ladder(sim: str, root=None, require=None) -> list[int]:
    """The snapshot numbers of 'sim' on disk, ascending: every snapdir_NNN under sim_dir(sim, root) (an entry
    whose part after 'snapdir_' is not a number is no snapshot and is left out). 'require' narrows it: a glob
    pattern the snapdir must hold a match of (e.g. 'combined_*.hdf5'), or a callable(snapdir) -> bool."""
    out = set()
    for d in sim_dir(sim, root).glob("snapdir_*"):
        part = d.name.split("_")[1]
        if not part.isdigit():
            continue
        if require is not None and not (require(d) if callable(require) else any(d.glob(require))):
            continue
        out.add(int(part))
    return sorted(out)


def use_data_root(root) -> Path | None:
    """Make `root` this process's data root: what a script's --data-root means for every dtfelib helper that is
    not handed a root explicitly (groupcat, trees, environment, pipeline.products without data_root=...).
    require_data_root() reads it at call time, so the switch reaches them all. None (a script run without
    --data-root) clears it: DTFE_DATA_ROOT applies again. Returns the root now in force, None when none is."""
    global _data_root_override
    _data_root_override = Path(root).expanduser() if root else None
    try:
        return require_data_root()
    except NoDataRoot:
        return None


def find_sims(root=None, pattern: str = "snapdir_*") -> list[str]:
    """Names of the simulations under `root` (default require_data_root()), in either layout, whose
    directory contains something matching `pattern` (e.g. 'snapdir_*/combined_*.hdf5')."""
    root = require_data_root(root)
    if not root.is_dir():
        return []
    names = set()
    for d in list(root.iterdir()) + [c for f in root.iterdir() if f.is_dir() for c in f.iterdir()]:
        if d.is_dir() and any(d.glob(pattern)):
            names.add(d.name)
    return sorted(names)


# ---------------------------------------------------------------- old spellings (P6.7)
# The python CLIs' renamed options keep their old spelling for one release, as the binaries' and the drivers' do: a
# hidden option that warns once per process on STDERR, in the binaries' words ('warning: --old is deprecated, use
# --new'), and then does what the new one does. Never on stdout: void_outlines.py and void_cache.py print one JSON
# line there. P6.6 removes the deprecated_alias / deprecated_option calls (the readers' tables stay).
_WARNED: set = set()


def warn_deprecated(old: str, use: str, note: str = "") -> None:
    """'warning: <old> is deprecated, use <use>[ (<note>)]' on stderr, once per process for each 'old'."""
    if old in _WARNED:
        return
    _WARNED.add(old)
    print(f"warning: {old} is deprecated, use {use}" + (f" ({note})" if note else ""), file=sys.stderr, flush=True)


def new_parser(**kw) -> argparse.ArgumentParser:
    """The ArgumentParser every python CLI starts from (make_parser too): allow_abbrev=False, so no option name is
    ever guessed from a prefix. After a rename the old name is often a prefix of a new one (--size-rv /
    --size-rv-definition, --smooth / --smooth-px, --subhalo / --subhalos): a guess would make it a silent spelling
    for good, and an option added later could change what an abbreviation meant."""
    kw.setdefault("allow_abbrev", False)
    return argparse.ArgumentParser(**kw)


class _DeprecatedOption(argparse.Action):
    """An old spelling: warns once, then hands the value to the canonical option's own action (target) -- so a
    store_true, a list or a typed value behave exactly as the new spelling does -- or, without a target, stores it
    under dest. convert, when given, maps the parsed value first."""

    def __init__(self, option_strings, dest, use: str = "", target=None, convert=None, note: str = "", **kw):
        super().__init__(option_strings, dest, **kw)
        self.use, self.target, self.convert, self.note = use, target, convert, note

    def __call__(self, parser, namespace, values, option_string=None):
        warn_deprecated(option_string or self.option_strings[0], self.use, self.note)
        if self.convert is not None:
            values = self.convert(values)
            if self.target is not None and self.target.choices is not None and values not in self.target.choices:
                raise argparse.ArgumentError(self, f"invalid choice: {values!r}")
        if self.target is not None:
            self.target(parser, namespace, values, option_string)
        else:
            setattr(namespace, self.dest, values)


def _option_action(parser: argparse.ArgumentParser, option: str) -> argparse.Action:
    for a in parser._actions:
        if option in a.option_strings:
            return a
    raise ValueError(f"no option {option} on this parser")


def deprecated_alias(parser: argparse.ArgumentParser, old, new: str, convert=None, use: str | None = None,
                     note: str = "") -> argparse.Action:
    """Register 'old' (one option string, or a tuple of them) as a hidden, deprecated spelling of the option 'new'
    already on the parser: the same dest, nargs, type and choices, its value handed to the new option's own action;
    using it warns once on stderr ('use <new>', or 'use <use>'). 'convert' maps the parsed value first. A required
    new option becomes a required pair (argparse checks 'required' per spelling, so the old one alone would fail
    it): one of the two must be given, not both. Returns the alias's action."""
    target = _option_action(parser, new)
    names = (old,) if isinstance(old, str) else tuple(old)
    kw = dict(action=_DeprecatedOption, use=use or new, target=target, convert=convert, note=note,
              dest=target.dest, nargs=target.nargs, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    if target.nargs != 0:                   # an action with arguments: parse them as the new spelling does
        kw.update(type=target.type, choices=None if convert else target.choices, metavar=target.metavar)
    if target.required:
        target.required = False
        group = parser.add_mutually_exclusive_group(required=True)
        group._group_actions.append(target)
        return group.add_argument(*names, **kw)
    return parser.add_argument(*names, **kw)


def deprecated_option(parser, *names: str, use: str, note: str = "", **kw) -> argparse.Action:
    """Register a hidden, deprecated option that has no single new spelling to forward to (e.g. make_image_plane's
    --u0, now one of --plane-window's four values): it stores under its own dest ('kw' as for add_argument) and
    warns once on stderr, 'use <use>'."""
    kw.setdefault("default", None)
    return parser.add_argument(*names, action=_DeprecatedOption, use=use, note=note, help=argparse.SUPPRESS, **kw)


class _ListAction(argparse.Action):
    """A space-separated list option (R7): every value converted to 'kind'. A value with a comma in it, the
    comma-joined spelling of before, is split there, with one warning (for one release)."""

    def __init__(self, option_strings, dest, kind=str, count=None, **kw):
        super().__init__(option_strings, dest, **kw)
        self.kind, self.count = kind, count

    def __call__(self, parser, namespace, values, option_string=None):
        out, comma = [], False
        for v in (values if isinstance(values, list) else [values]):
            comma = comma or "," in v
            for w in v.split(","):
                w = w.strip()
                if not w:
                    continue
                try:
                    out.append(self.kind(w))
                except (TypeError, ValueError):
                    raise argparse.ArgumentError(self, f"invalid {getattr(self.kind, '__name__', 'list')} value: {w!r}")
        if self.count is not None and len(out) != self.count:
            raise argparse.ArgumentError(self, f"expected {self.count} values, got {len(out)}")
        if comma:
            warn_deprecated(f"a comma-separated {option_string}",
                            f"spaces ({option_string} {' '.join(f'{x:g}' if isinstance(x, float) else str(x) for x in out)})")
        setattr(namespace, self.dest, out)


def add_list_arg(parser, *names: str, kind=str, nargs: str | int = "+", count: int | None = None,
                 **kw) -> argparse.Action:
    """Register a space-separated list option ('--snaps 0 4 17'; nargs '+' by default) whose values are converted to
    'kind'; a comma-joined value ('0,4,17', the old spelling) is still split, with one warning. 'count': exactly that
    many values once split (taken as nargs '+', so the one comma-joined token of before still parses)."""
    return parser.add_argument(*names, action=_ListAction, kind=kind, nargs="+" if count else nargs, count=count, **kw)


def choice_type(flag: str, choices, old: dict | None = None, internal: dict | None = None):
    """An argparse 'type' for an option whose values were renamed: 'choices' are the canonical spellings; an 'old'
    one ({old: canonical}) warns once and counts as its canonical value; 'internal' maps a canonical value to the
    name the script uses inside (npz keys, figure tags: --criterion density-raw is 'density_raw' there), by default
    the value itself. Give the parser metavar='{a,b,c}' (argparse's own choices list would hold the internal names)."""
    old, internal = dict(old or {}), dict(internal or {})

    def convert(text: str):
        if text in old:
            warn_deprecated(f"{flag} {text}", f"{flag} {old[text]}")
            text = old[text]
        if text not in choices:
            raise argparse.ArgumentTypeError(f"invalid choice: {text!r} (choose from {', '.join(choices)})")
        return internal.get(text, text)
    convert.__name__ = flag.lstrip("-")
    return convert


# The void scripts' smoothing length fixed in Mpc, from the environment (R11: the flag is --smooth-mpc): its older
# name DTFE_SIGMA_MPC is read for one release with one warning. One reader -- config.py at import, the cache
# inventory, analyze.py's refusal --, and one list of names for whoever strips them from a child's environment (the
# Explore void outlines, plot_void_crossbox's workers: their smoothing is the command line's, never the shell's).
SMOOTH_MPC_ENV = "DTFE_SMOOTH_MPC"
SMOOTH_MPC_ENV_NAMES = (SMOOTH_MPC_ENV, "DTFE_SIGMA_MPC")


def smooth_mpc_env(environ=None) -> str:
    """The smoothing in Mpc the environment fixes, as text: DTFE_SMOOTH_MPC, else DTFE_SIGMA_MPC (one warning:
    deprecated); '' when neither is set (an empty value is unset). Beside a different DTFE_SMOOTH_MPC the old name
    is said not to be read."""
    env = os.environ if environ is None else environ
    new, old_name = SMOOTH_MPC_ENV, SMOOTH_MPC_ENV_NAMES[1]
    v_new, v_old = env.get(new, ""), env.get(old_name, "")
    if v_new:
        if v_old and v_old != v_new:
            warn_deprecated(old_name, new, f"{old_name}={v_old} is not read: {new}={v_new} wins")
        return v_new
    if v_old:
        warn_deprecated(old_name, new)
    return v_old


def add_sim_arg(parser: argparse.ArgumentParser, help: str | None = None) -> argparse.Action:
    """Register '--sim NAME' the way every script takes it: required unless DTFE_SIM is set, which is then its
    default (require_sim's rule: there is no default simulation). Returns the action."""
    env_sim = os.environ.get("DTFE_SIM")
    return parser.add_argument("--sim", default=env_sim, required=not env_sim, help=help)


ESTIMATORS = ("auto", "ps", "dtfe")


def add_estimator_arg(parser: argparse.ArgumentParser, choices=ESTIMATORS, default: str = "auto",
                      help: str = "field estimator to load: PS-DTFE, standard DTFE, or auto-detect") -> argparse.Action:
    """Register '--estimator {auto,ps,dtfe}' (args.method: FieldSet's keyword, the name every script reads) and its
    old spelling '--method' (deprecated; the binaries' --method was the cell-average scheme, now --average-method).
    The values match the drivers (run_ps_dtfe.sh / run_dtfe.sh) and the binaries (PS-DTFE / DTFE)."""
    act = parser.add_argument("--estimator", dest="method", choices=choices, default=default, help=help)
    deprecated_alias(parser, "--method", "--estimator")
    return act


def make_parser(description: str = "", snap_default: int | None = DEFAULT_SNAP,
                snap_help: str | None = None, snap_option: str = "--snap") -> argparse.ArgumentParser:
    """The parser every figure script starts from: --data-root, --sim (add_sim_arg), --snap, --estimator, --prefix,
    --raw and --smooth. snap_default / snap_help: a series script's --snap (default None = the whole series) and
    its help; by default snapshot DEFAULT_SNAP. snap_option: the option's name where the snapshot means something
    more particular -- the tracking scripts' '--start-snap', the snapshot they follow back from -- always args.snap;
    there '--snap' is its old spelling (deprecated). No abbreviation is guessed (new_parser)."""
    p = new_parser(description=description)
    p.add_argument("--data-root", type=Path, default=None,
                   help="the folder that holds the simulations (default: DTFE_DATA_ROOT; one of the two is needed)")
    add_sim_arg(p, help="simulation name (required unless DTFE_SIM is set; there is no default)")
    p.add_argument(snap_option, dest="snap", type=int, default=snap_default,
                   help=snap_help if snap_help is not None else f"snapshot number (default {snap_default})")
    if snap_option != "--snap":
        deprecated_alias(p, "--snap", snap_option, note="in this script it is the snapshot the tracking starts from")
    add_estimator_arg(p)
    p.add_argument("--prefix", default=None,
                   help="alternate on-disk grid prefix (run_ps_dtfe.sh OUTPUT_PREFIX, e.g. "
                        "'ps_mw'); implies --estimator ps unless one is given explicitly")
    p.add_argument("--raw", action="store_true",
                   help="use the unaveraged fields instead of the volume-averaged '_a' ones")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Gaussian smoothing applied at plot/analysis time, in grid cells (default 0)")
    return p


# The PS-DTFE velocity-gradient family a figure script uses: the scripts that set PS beside standard DTFE
# or linear theory -- plot_velDiv_den, plot_cosmic_web, plot_web_streams, plot_shear_triaxial, plot_DTFE,
# plot_halo_tracking -- read PS's divergence / shear / vorticity / gradient / V-web as the MEAN FIELD by default,
# the quantity standard DTFE stores (FieldSet.load(name, mean_field=True); dtfelib/fields.py), and '--ps-gradient
# stored' gives back the binary's PS grids, the stream average <grad v>. Standard-DTFE grids are read as stored
# either way (they are the mean field).
PS_GRADIENT_CHOICES = ("mean-field", "stored")


def add_ps_gradient_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register '--ps-gradient {mean-field,stored}' (default mean-field) on a script's parser; returns it."""
    parser.add_argument("--ps-gradient", choices=PS_GRADIENT_CHOICES, default="mean-field",
                        help="PS-DTFE grids only: which velocity-gradient family to use for PS -- 'mean-field' "
                             "(default): the derivatives of the PS velocity grid, grad<v>, the quantity standard "
                             "DTFE stores, computed by FFT; 'stored': the binary's PS grids, the stream average "
                             "<grad v> of each stream's own gradient. Standard-DTFE grids are always read as stored.")
    return parser


def add_snaps_arg(parser: argparse.ArgumentParser, help: str = "snapshots (default: --snap)",
                  nargs: str = "*") -> argparse.Action:
    """Register '--snaps N [N ...]' (integers, space-separated; default None = the script's own choice, by default
    --snap). A comma-joined list is split with one warning (add_list_arg)."""
    return add_list_arg(parser, "--snaps", kind=int, nargs=nargs, default=None, metavar="N", help=help)


def add_out_arg(parser: argparse.ArgumentParser, default_text: str | None = None,
                help: str | None = None, type=Path, required: bool = False, metavar: str | None = None
                ) -> argparse.Action:
    """Register '-o/--out PATH' (by default a Path: the folder a script writes into instead of its default; 'type',
    'metavar' and 'help' for a script whose -o is a file or a stem). Its older spelling '--output' is deprecated.
    help: the text, by default 'output folder (default <default_text>)'."""
    act = parser.add_argument("-o", "--out", type=type, default=None, required=required, metavar=metavar,
                              help=help if help is not None else f"output folder (default {default_text})")
    deprecated_alias(parser, "--output", "--out")
    return act


def add_rv_definition_arg(parser: argparse.ArgumentParser, choices, help: str,
                          dest: str = "rv_definition") -> argparse.Action:
    """Register '--rv-definition' (default 'crossing'; dest names args' attribute, so a script keeps the one it
    reads) and its old second name '--r-v-definition' (deprecated)."""
    act = parser.add_argument("--rv-definition", dest=dest, choices=choices, default="crossing", help=help)
    deprecated_alias(parser, "--r-v-definition", "--rv-definition")
    return act


def mean_field_wanted(args) -> bool:
    """True unless the script was run with '--ps-gradient stored' (a parser without the flag: True)."""
    return getattr(args, "ps_gradient", "mean-field") != "stored"


# ---------------------------------------------------------------- series scripts
# The core figure scripts (plot_PDF_CDF, plot_eigenvalues, plot_contour_filter, plot_ellipses_contour_3D,
# plot_shape_filter, plot_velDiv_den, plot_shear_triaxial, plot_marked_correlation_BBKS) run one snapshot after the
# other: snapshot_series names them, run_series runs them, report_series prints the closing lines. The lines are the
# scripts' own (tests read 'Completed: n/m' and 'Successfully processed: n/m'); the launcher goes by the exit code.

def snapshot_series(args) -> list:
    """The (snapshot id, redshift) pairs a series script runs over: config's whole ladder in its order
    (config.SNAPSHOT_TO_REDSHIFT, read at call time), or only --snap when one is given (zero-padded, its redshift
    from the ladder; None for an id the ladder lacks, which the script handles)."""
    import config
    if getattr(args, "snap", None) is None:
        return list(config.SNAPSHOT_TO_REDSHIFT.items())
    snap_id = f"{int(args.snap):03d}"
    return [(snap_id, config.get_redshift(snap_id))]


def run_series(items, fn, catch: bool = False) -> tuple:
    """Run fn(snapshot, redshift) for every (snapshot, redshift) pair of 'items', in order. Returns (outcomes,
    failed): outcomes[i] what the call for items[i] returned, failed the pairs whose call failed. Without 'catch' a
    call fails by returning None or False; with it a call fails only by raising an Exception, reported as
    'Failed to process snapshot S (z=Z): why' (its outcome None), and the series goes on."""
    outcomes, failed = [], []
    for snapshot, redshift in items:
        if catch:
            try:
                out = fn(snapshot, redshift)
            except Exception as e:
                print(f"Failed to process snapshot {snapshot} (z={redshift:.2f}): {str(e)}")
                outcomes.append(None)
                failed.append((snapshot, redshift))
                continue
        else:
            out = fn(snapshot, redshift)
            if out is None or out is False:
                failed.append((snapshot, redshift))
        outcomes.append(out)
    return outcomes, failed


def report_series(n_done: int, n_total: int, failed, label: str = "snapshots", listed: bool = False) -> None:
    """A series' closing lines: 'Completed: n/m <label> processed' and the failed ones on one line, or with 'listed'
    'Analysis complete', 'Successfully processed: n/m <label>', the failed count and one line per failure."""
    if not listed:
        print(f"\nCompleted: {n_done}/{n_total} {label} processed")
        if failed:
            print(f"Failed {label}: {', '.join(f'{s} (z={z:.2f})' for s, z in failed)}")
        return
    print("\nAnalysis complete")
    print(f"Successfully processed: {n_done}/{n_total} {label}")
    if failed:
        print(f"Failed {label}: {len(failed)}")
        for snapshot, redshift in failed:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")


def snapdir(args) -> Path:
    """The snapdir a parsed make_parser() run names: --sim, --snap and --data-root (snapdir_path)."""
    return snapdir_path(args.sim, args.snap, args.data_root)


def make_fieldset(description: str = "", extra=None, argv=None):
    """Parse standard flags (+ optional script-specific ones) and open the FieldSet.

    extra: callable(parser) that registers additional arguments.
    Returns (fieldset, args).
    """
    parser = make_parser(description)
    if extra is not None:
        extra(parser)
    args = parser.parse_args(argv)
    try:
        args.data_root = require_data_root(args.data_root)
    except NoDataRoot as e:
        parser.error(str(e))
    fs = FieldSet(snapdir(args), method=args.method, averaged=not args.raw,
                  prefix=args.prefix)
    return fs, args
