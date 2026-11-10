"""The binaries' option names as their run logs carry them: every older spelling mapped onto the canonical one.

A run log's 'RUNNING:' line is the binary's own command line, and grids on disk carry command lines years old.
Every reader of one (the launcher's Runs browser and 'Load settings', gui/results.py; dtfelib.io's threshold and
weighting reads; run_ps_pipeline.sh's run_settings, an awk copy of this table) maps the old names first, so a log
written with either spelling reads the same. This table is PERMANENT: the binaries drop their deprecated aliases
one release after the rename (docs/cleanup-plan.md P6.6), the readers never do.

  RENAMES               old option name -> canonical name, the value unchanged ('--MpcUnit' -> '--mpc-unit')
  SWITCHES              an old switch -> (canonical option, value) ('--ordered-merge' -> '--ps-merge index-order')
  VALUE_RENAMES         an option's old value -> its canonical value ('--ps-stream-density dtfe' -> 'linear')
  canonical_flag(name)  one option name in its canonical spelling (a SWITCHES row: 'option value')
  canonical_command(c)  a whole command line in canonical spellings, with the value transforms (see there)
  command_lines(text)   the 'RUNNING:' command lines of a run log's text

Standard library only (docs_drift_check.py loads it by path under any python3).
"""

from __future__ import annotations

import re

# ---- the permanent table (docs/cleanup-plan.md P6 and P6.7; the binaries' src/user_options.cc deprecatedOptions
# holds the rows they still accept, a subset; scripts/run_ps_pipeline.sh run_settings carries the same rows as its
# old[] / sw[] maps -- tests/docs_drift_check.py keeps the copies equal)
RENAMES = {
    # P6 (2026-11-07): one name for a concept both binaries have, and no snake_case
    "--ps-gpu": "--gpu",
    "--ps-exact-deposit": "--exact",        # straight to the 2026-11-09 name below (the table does not chain)
    "--ps-no-overlap": "--no-overlap",
    "--full_help": "--full-help",
    "--lambda_th": "--lambda-th",
    # P6.7: kebab-case whole words
    "--MpcUnit": "--mpc-unit",
    "--paddingMpc": "--padding-mpc",
    "--regionMpc": "--region-mpc",
    "--partNo": "--partition-index",
    "--redshiftCone": "--redshift-cone",
    "--origin": "--cone-origin",
    "--NGP": "--ngp",
    "--CIC": "--cic",
    "--TSC": "--tsc",
    "--PCS": "--pcs",
    "--SPH": "--sph",
    "--Voronoi": "--voronoi",
    "--inputFile": "--input-file",
    "--outputFile": "--output-file",
    # P6.7: names that said the wrong thing
    "--input": "--input-format",            # a type code (the file is the positional argument)
    "--output": "--output-format",          # a type code (the output root is the positional argument)
    "--method": "--average-method",         # standard DTFE's cell-average method 1|2|3 (the short '-m' goes)
    "--noTest": "--no-padding-test",
    "--density0": "--mean-density",
    "--randomSample": "--particle-fraction",
    "--approxPSD": "--approx-phase-space-density",
    "--extensive": "--scalars-extensive",   # not '--extensive-scalars': the old name would abbreviate it for good
    "--options": "--reader-options",
    # P6.7: 'ps-' exactly for stream / Lagrangian / caustic concepts
    "--avg-subsamples": "--subsamples",     # k per axis, k^D points per cell, in both binaries (the same unit)
    "--lagrangianInput": "--ps-lagrangian-input",
    "--ps-lag-padding": "--ps-lagrangian-padding",
    "--per-stream": "--ps-per-stream",
    "--per-stream-ids": "--ps-per-stream-ids",
    "--pts-lagrangian": "--ps-per-stream-lagrangian",
    "--pts-conventions": "--ps-point-conventions",
    "--ps-window": "--grid-window",         # it crops the output grid: no stream concept
    # 2026-11-09: the exact cell average is the default of both binaries, '--sampled' the other choice, and the
    # switch took the short name. A log with neither flag is read by its header line (gui/results.py RunRecord.exact)
    "--exact-average": "--exact",
}
# the three merge switches of PS-DTFE are one mode. Precedence when a log has more than one, as the binary
# reads them: '--fixed-point-merge' wins (beside '--ordered-merge' it is accepted), then '--ordered-merge' (beside
# '--completion-order-merge' only under '--resume', which merges in index order). No switch = no '--ps-merge'
# token: the default, NOT given (an explicit index-order merge keeps the memory gate on under the speed split).
SWITCHES = {
    "--ordered-merge": ("--ps-merge", "index-order"),
    "--completion-order-merge": ("--ps-merge", "completion-order"),
    "--fixed-point-merge": ("--ps-merge", "fixed-point"),
}
_MERGE_RANK = {"fixed-point": 0, "index-order": 1, "completion-order": 2}
VALUE_RENAMES = {"--ps-stream-density": {"dtfe": "linear"}}
# the binaries' short letters that go (the drivers' '-m' / '-s' mean the GPU and the simulation)
SHORT_LETTERS = {"-m": "--average-method", "-s": "--samples"}
# standard DTFE's sample count: '--samples N' (a total per cell) reads as '--subsamples k' when N = k^D exactly,
# else it stays '--samples N' (the canonical legacy token: the binary keeps accepting a total)
SAMPLES = "--samples"
SUBSAMPLES = "--subsamples"

_FLAG_TOKEN = re.compile(r"(?<!\S)--[A-Za-z][\w-]*")     # an option name at a token's start ('--x' or '--x=v')
_SHORT = re.compile(r"(?<!\S)(-[ms])(?:\s+|=)?(\d+)(?!\S)")
_SAMPLES = re.compile(r"(?<!\S)--samples(?:\s+|=)(\d+)(?!\S)")
_SWITCH = re.compile(r"(?<!\S)(?:--(ordered-merge|completion-order-merge|fixed-point-merge)"
                     r"|--ps-merge(?:\s+|=)(index-order|completion-order|fixed-point))(?!\S)")
_BINARY = re.compile(r"^(?:RUNNING:\s*)?(\S+)")
_RUNNING = re.compile(r"^RUNNING: (.*)$", re.M)


def canonical_flag(flag: str) -> str:
    """The canonical spelling of one option name: '--MpcUnit' -> '--mpc-unit'; a merge switch -> its mode
    ('--completion-order-merge' -> '--ps-merge completion-order', as canonical_command writes it); any other name
    unchanged."""
    if flag in SWITCHES:
        return " ".join(SWITCHES[flag])
    return RENAMES.get(flag, flag)


def command_dim(command: str) -> int:
    """2 for a command of a 2D program (DTFE-2d, PS-DTFE-2d[-double]), else 3: from its first word, the binary."""
    m = _BINARY.match(command.strip())
    return 2 if m and "-2d" in m.group(1).rsplit("/", 1)[-1] else 3


def _root(n: int, dim: int) -> int | None:
    """k when n = k^dim exactly (k >= 1), else None."""
    if n < 1:
        return None
    k = round(n ** (1.0 / dim))
    for c in (k - 1, k, k + 1):
        if c >= 1 and c ** dim == n:
            return c
    return None


def _samples(n: str, dim: int) -> str:
    k = _root(int(n), dim)
    return f"{SUBSAMPLES} {k}" if k is not None else f"{SAMPLES} {int(n)}"


_SUBS = re.compile(r"(\s*)(?<!\S)--subsamples(?:\s+|=)(\d+)(?!\S)")


def _dedupe_subsamples(command: str) -> str:
    """Every later '--subsamples k' of a value already given goes (with the blanks before it)."""
    seen = set()

    def one(m):
        if m.group(2) in seen:
            return ""
        seen.add(m.group(2))
        return m.group(0)
    return _SUBS.sub(one, command)


def _fold_merge(command: str) -> str:
    """Every merge switch (and a '--ps-merge MODE') as ONE '--ps-merge MODE' at the first one's place."""
    found = list(_SWITCH.finditer(command))
    if not found:
        return command
    modes = [SWITCHES["--" + m.group(1)][1] if m.group(1) else m.group(2) for m in found]
    mode = min(modes, key=_MERGE_RANK.__getitem__)
    out, last = [], 0
    for i, m in enumerate(found):
        start = m.start()
        if i:                                   # a later one goes with the blanks before it
            while start > last and command[start - 1].isspace():
                start -= 1
        out.append(command[last:start])
        if not i:
            out.append(f"--ps-merge {mode}")
        last = m.end()
    out.append(command[last:])
    return "".join(out)


def command_is_ps(command: str) -> bool:
    """True for a command of a PS-DTFE binary (PS-DTFE, PS-DTFE-2d, PS-DTFE-double, ...): the binary of its first
    'RUNNING:' line when the text holds one (a whole run log), else its first word."""
    run = _RUNNING.search(command)
    m = _BINARY.match((run.group(1) if run else command).strip())
    return bool(m) and m.group(1).rsplit("/", 1)[-1].startswith("PS-DTFE")


def canonical_command(command: str, dim: int | None = None, averaging: bool | None = None) -> str:
    """A command line with every older option name in its canonical spelling, the rest untouched:
    '--lambda_th=0.3' -> '--lambda-th=0.3', '--MpcUnit 1' -> '--mpc-unit 1'; a name inside a path or a value is not
    an option's ('/a/--ps-gpu' stays). The value transforms:
      '-m 2' / '-m2'                       -> '--average-method 2'
      '-s N' / '--samples N' (a total)     -> '--subsamples k' when N = k^dim exactly, else '--samples N'
      '--ordered-merge' & co.              -> one '--ps-merge index-order|completion-order|fixed-point'
      '--ps-stream-density dtfe'           -> '--ps-stream-density linear'
    'dim': the program's dimension (2 for a -2d binary); None reads it from the command's first word.
    'averaging' False leaves standard DTFE's averaging spellings ('-m', '-s', '--method', '--samples') as they are:
    for PS-DTFE commands, where they have no effect and no new name ('-s 27' would become a '--subsamples' that
    does change the run; PS-DTFE drops them from its checkpoint key too). None (the default) reads it from the
    binary, as 'dim': False for a PS-DTFE command (command_is_ps), True otherwise; the launcher's saved extra
    options (no binary) pass it. A '--subsamples k' given twice with one value (standard '--samples 27 --subsamples
    3', which the binary accepts) is kept once, as the binary's checkpoint key keeps it."""
    if dim is None:
        dim = command_dim(command)
    if averaging is None:
        averaging = not command_is_ps(command)
    keep = set() if averaging else {"--method"}
    text = _FLAG_TOKEN.sub(lambda m: m.group(0) if m.group(0) in keep else RENAMES.get(m.group(0), m.group(0)),
                           command)
    if averaging:
        text = _SHORT.sub(lambda m: (f"{SHORT_LETTERS['-m']} {m.group(2)}" if m.group(1) == "-m"
                                     else _samples(m.group(2), dim)), text)
        text = _SAMPLES.sub(lambda m: _samples(m.group(1), dim), text)
    if averaging:
        text = _dedupe_subsamples(text)
    text = _fold_merge(text)
    for opt, values in VALUE_RENAMES.items():
        text = re.sub(rf"(?<!\S)({re.escape(opt)})(\s+|=)(\S+)(?!\S)",
                      lambda m: m.group(1) + m.group(2) + values.get(m.group(3), m.group(3)), text)
    return text


def command_lines(text: str) -> list[str]:
    """The command lines of a run log's 'RUNNING:' lines (the binary prints one per run), as written."""
    return _RUNNING.findall(text)
