#!/usr/bin/env python3
"""Docs-drift check: what README.md and demo/COMMANDS.md tell a user to type must exist in the code,
and the knobs the drivers read must be documented somewhere a user looks.

  A) every '--flag' in README.md / demo/COMMANDS.md is an option of the binaries (src/user_options.cc's
     program_options names), of a script (scripts/*.sh) or of a python tool or test (argparse); and none is a
     DEPRECATED spelling (user_options.cc deprecatedOptions: hidden, accepted for one release), except on a line
     that says 'deprecated' (a note naming the old spelling);
  B) every DTFE_* name in them is read somewhere in the code;
  C) the README knob table: each knob it names is a knob of scripts/config.sh / run_ps_dtfe.sh / run_dtfe.sh (or
     an environment variable the code reads), and where the table gives a plain value as the default it equals
     the scripts' 'VAR="${VAR:-default}"';
  D) the other way round: every knob of the drivers and the pipeline, and every DTFE_* variable that
     scripts/config.sh or python/config.py reads, is named in README.md, demo/COMMANDS.md or a script's --help
     header (the comment block a script's --help prints);
  E) the run-log readers' permanent old-to-new table (python/dtfelib/runlog.py RENAMES / SWITCHES) is the one
     python/gui/results.py re-exports and scripts/run_ps_pipeline.sh's run_settings carries (its old[] / sw[]
     maps), and it holds every row the binaries accept as a deprecated alias (user_options.cc deprecatedOptions;
     PS-DTFE's merge switches, deprecatedSwitches, are runlog.py's SWITCHES);
  F) the renamed knobs (the binaries' env_knobs.h Env::renamed, the drivers' 'knob NEW OLD...' / 'knob_merge' lines,
     python's DTFE_SIGMA_MPC) appear in the docs (README.md, demo/COMMANDS.md, metal/README.md) under an old name only
     on a line that says 'deprecated' or 'old(er) name', and the README knob table names none of them;
  G) the binaries' own help (when ./DTFE and ./PS-DTFE are built; a SKIP line otherwise): no visible option name has
     a capital letter or a '_'; no name standard DTFE lists starts with 'ps-' (so every 'ps-' name is PS-only and
     every name both list lacks it); every '-h' name is a '--full-help' name of the same binary (not the other way
     round: '-h' is the short list by design).
Part A's deprecated spellings are the binaries' (deprecatedOptions, the merge switches and standard DTFE's total
'--samples') and the python CLIs' (dtfelib.cli deprecated_alias calls) whose old name no parser registers any more
(a per-script old name that another script still uses, '--prefix', '--snap', '--smooth', is not checked).
The synthetic half (S) feeds the same functions a made-up README with known drift, so a checker that stopped
finding anything fails too. Standard library only (runs under any python3 >= 3.10, no numpy).

    python3 tests/docs_drift_check.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "demo/COMMANDS.md", "metal/README.md")
DRIVERS = ("scripts/config.sh", "scripts/run_ps_dtfe.sh", "scripts/run_dtfe.sh")   # the README table's defaults
KNOB_SCRIPTS = DRIVERS + ("scripts/run_ps_pipeline.sh",)
SKIP_PARTS = {"tmp", "cache", "__pycache__", "figures", "reference"}
RUNLOG = "python/dtfelib/runlog.py"     # the readers' table: its new names are no evidence that an option exists
PIPELINE = "scripts/run_ps_pipeline.sh"
TABLE_ROW = re.compile(r'(old|sw)\["(--[\w-]+)"\] = "(--[\w-]+(?: [\w-]+)?)"')   # run_settings' copy of it

# a flag: '--' right after a non-word character and right before a letter. The README uses ' -- ' as a dash
# everywhere, so the letter after the dashes is what tells a flag from punctuation.
FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*[A-Za-z0-9]")
DTFE_RE = re.compile(r"\bDTFE_[A-Z0-9_]*[A-Z0-9]\b")
# the binaries' deprecated spellings and their canonical names (src/user_options.cc deprecatedOptions), the
# names the docs may give only on a 'deprecated' line; the readers' permanent table (python/dtfelib/runlog.py,
# re-exported by python/gui/results.py, copied into scripts/run_ps_pipeline.sh run_settings) is a superset (E)
DEPRECATED = {"--ps-gpu": "--gpu", "--ps-exact-deposit": "--exact", "--ps-no-overlap": "--no-overlap",
              # 2026-11-09: '--exact' / '--sampled', the exact average the default (both binaries)
              "--exact-average": "--exact",
              "--full_help": "--full-help", "--lambda_th": "--lambda-th",
              # P6.7 (the binaries' second pass; PS-DTFE's merge switches are deprecatedSwitches, the readers' SWITCHES)
              "--avg-subsamples": "--subsamples", "--lagrangianInput": "--ps-lagrangian-input",
              "--ps-lag-padding": "--ps-lagrangian-padding", "--per-stream": "--ps-per-stream",
              "--per-stream-ids": "--ps-per-stream-ids", "--pts-lagrangian": "--ps-per-stream-lagrangian",
              "--pts-conventions": "--ps-point-conventions", "--ps-window": "--grid-window",
              "--method": "--average-method", "--approxPSD": "--approx-phase-space-density",
              "--input": "--input-format", "--output": "--output-format", "--inputFile": "--input-file",
              "--outputFile": "--output-file", "--MpcUnit": "--mpc-unit", "--regionMpc": "--region-mpc",
              "--partNo": "--partition-index", "--paddingMpc": "--padding-mpc", "--noTest": "--no-padding-test",
              "--density0": "--mean-density", "--redshiftCone": "--redshift-cone", "--origin": "--cone-origin",
              "--NGP": "--ngp", "--CIC": "--cic", "--TSC": "--tsc", "--PCS": "--pcs", "--SPH": "--sph",
              "--Voronoi": "--voronoi", "--extensive": "--scalars-extensive", "--randomSample": "--particle-fraction",
              "--options": "--reader-options"}

# flags of other programs the docs quote (none at present), and the one binary option kept in camelCase on purpose: its
# 'redshiftSpace=' line is in every tessellation cache and globals record (docs/cleanup-plan.md P6.7)
FOREIGN: set[str] = set()
KEPT_CAMEL = {"redshiftSpace"}
# a doc line allowed to name an old spelling (a deprecation note): it says so
OLD_LINE = re.compile(r"deprecated|\bold(?:er)? names?\b", re.I)

passed = failed = skipped = 0


def check(ok, msg):
    global passed, failed
    if ok:
        passed += 1
        print(f"   PASS  {msg}")
    else:
        failed += 1
        print(f"   FAIL  {msg}")


def skip(msg):
    global skipped
    skipped += 1
    print(f"   SKIP  {msg}")


def read(rel):
    p = ROOT / rel
    return p.read_text(errors="replace") if p.exists() else ""


def code_files():
    pats = ["src/**/*.cc", "src/**/*.cpp", "src/**/*.h", "src/**/*.hpp", "src/**/*.metal",
            "scripts/*.sh", "python/**/*.py", "tests/**/*.py", "tests/**/*.sh", "Makefile", "metal/**/*.cpp"]
    seen = set()
    for pat in pats:
        for f in sorted(ROOT.glob(pat)):
            if (f in seen or SKIP_PARTS & set(f.relative_to(ROOT).parts) or f.name == Path(__file__).name
                    or str(f.relative_to(ROOT)) == RUNLOG):
                continue
            seen.add(f)
            yield f


def code_flags(texts):
    """Every option name the code defines: literal '--x' strings (scripts, argparse) plus the program_options
    names of the binaries, which are declared without dashes: ("grid", ...) or ("config,c", ...)."""
    flags = set()
    for name, text in texts:
        if name == PIPELINE:            # run_settings' copy of the readers' table: as RUNLOG, no evidence
            text = "\n".join(ln for ln in text.splitlines() if not TABLE_ROW.search(ln))
        flags.update(FLAG_RE.findall(text))
        if name.endswith(("user_options.cc", "user_options.h")):
            flags.update("--" + m for m in re.findall(r'\(\s*"([A-Za-z][\w-]*)(?:,\w)?"\s*,', text))
    return flags


def deprecated_uses(text, table=None):
    """[(old, canonical, line number)] of every deprecated spelling (table: {old: canonical}, DEPRECATED by default)
    in a document, outside a line that says 'deprecated' (where naming the old spelling is the point)."""
    table = DEPRECATED if table is None else table
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        if "deprecated" in ln.lower():
            continue
        out += [(f, table[f], i) for f in FLAG_RE.findall(ln) if f in table]
    return out


def cpp_switches(src):
    """{'--old': ('--canonical', 'value')} of user_options.cc's deprecatedSwitches table (PS-DTFE's merge switches)."""
    t = src[src.find("deprecatedSwitches[] = {"):]
    t = t[:t.find("};")]
    return {"--" + a: ("--" + b, v) for a, b, v in re.findall(r'\{ "([\w-]+)", +"([\w-]+)", +"([\w-]+)" \}', t)}


def python_deprecated(texts):
    """{'--old': '--new'} of the python CLIs' deprecated_alias(parser, "--old", "--new") calls whose old name no parser
    registers as an option of its own (add_argument("--old") / add_arg("--old") anywhere): an old spelling everywhere."""
    calls, registered = {}, set()
    for name, text in texts:
        if not name.startswith("python/"):
            continue
        calls.update(re.findall(r'deprecated_alias\(\s*\w+\s*,\s*"(--[\w-]+)"\s*,\s*"(--[\w-]+)"', text))
        for m in re.finditer(r'add_argument\(([^)]*)', text):
            registered.update(re.findall(r'"(--[\w-]+)"', m.group(1).split("=", 1)[0]))
    return {o: n for o, n in calls.items() if o not in registered}


def knob_renames(texts):
    """{old knob: new knob}: env_knobs.h's Env::renamed rows (the binaries), every 'knob [-e] NEW OLD...' and
    'knob_merge NEW OLD...' line of scripts/*.sh (the drivers), python's smoothing-length alias (DTFE_SIGMA_MPC)."""
    out = {}
    for name, text in texts:
        if name == "src/env_knobs.h":
            t = text[text.find("renamed[] = {"):]
            out.update({o: n for n, o in re.findall(r'\{ "([A-Z0-9_]+)", +"([A-Z0-9_]+)", +(?:true|false) \}', t[:t.find("};")])})
        elif name.startswith("scripts/") and name.endswith(".sh"):
            for new, olds in re.findall(r'^\s*knob(?:_merge)?[ \t]+(?:-e[ \t]+)?([A-Z][A-Z0-9_]*)((?:[ \t]+[A-Z][A-Z0-9_]*)+)', text, re.M):
                out.update({o: new for o in olds.split()})
        elif name == "python/dtfelib/cli.py":
            m = re.search(r'^SMOOTH_MPC_ENV_NAMES = \(SMOOTH_MPC_ENV, "([A-Z0-9_]+)"\)', text, re.M)
            if m:
                out[m.group(1)] = "DTFE_SMOOTH_MPC"
    return out


def old_knob_uses(text, olds):
    """[(old, new, line number)] of every old knob name in a document outside a line that says 'deprecated' or
    'old(er) name' (an upper-case word, not inside a longer name: 'PS_GPU' in 'DTFE_PS_GPU_X' is not one)."""
    if not olds:
        return []
    pat = re.compile(r"(?<![\w$])(" + "|".join(sorted(map(re.escape, olds), key=len, reverse=True)) + r")(?![\w])")
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        if OLD_LINE.search(ln):
            continue
        out += [(o, olds[o], i) for o in pat.findall(ln)]
    return out


def help_names(text):
    """The option names a binary's help lists ('  -g [ --grid ] arg', '  --box arg': grid, box): the lines that
    start an entry (two blanks), not the wrapped descriptions."""
    return set(re.findall(r"^  (?:-[A-Za-z] \[ )?--([A-Za-z0-9][\w-]*)", text, re.M))


def help_style(full, short, standard):
    """[problem] of one binary's help: a visible name with a capital or a '_'; a 'ps-' name in standard DTFE's help
    (standard=True); a '-h' name the '--full-help' lacks."""
    out = [f"--{n}: a capital or '_'" for n in sorted(full | short) if re.search(r"[A-Z_]", n) and n not in KEPT_CAMEL]
    if standard:
        out += [f"--{n}: 'ps-' in standard DTFE" for n in sorted(full | short) if n.startswith("ps-")]
    out += [f"--{n}: in -h, not in --full-help" for n in sorted(short - full)]
    return out


def load_runlog(path=None):
    """python/dtfelib/runlog.py as a module, by its path (stdlib only; not through the dtfelib package, whose
    __init__ needs numpy)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_runlog_table", path or ROOT / RUNLOG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reader_table_drift(renames, switches, pipeline_text, results_text, binary_table):
    """[problem] between the readers' table (RENAMES {old: new}, SWITCHES {old: (option, value)}), run_settings'
    old[] / sw[] maps, results.py (it must re-export the table, not hold one of its own) and the binaries'
    deprecated aliases (each a row of RENAMES with the same canonical name)."""
    out = []
    awk = {k: v for kind, k, v in TABLE_ROW.findall(pipeline_text) if kind == "old"}
    sw = {k: tuple(v.split()) for kind, k, v in TABLE_ROW.findall(pipeline_text) if kind == "sw"}
    for k in sorted(set(renames) | set(awk)):
        if renames.get(k) != awk.get(k):
            out.append(f"run_settings old[{k}] = {awk.get(k)}, runlog.py {renames.get(k)}")
    for k in sorted(set(switches) | set(sw)):
        if tuple(switches.get(k, ())) != sw.get(k, ()):
            out.append(f"run_settings sw[{k}] = {sw.get(k)}, runlog.py {switches.get(k)}")
    if not re.search(r"^from dtfelib import runlog as (\w+)", results_text, re.M) \
            or not re.search(r"^DEPRECATED_FLAGS = _runlog\.RENAMES$", results_text, re.M):
        out.append("results.py does not re-export runlog.py's table (DEPRECATED_FLAGS = _runlog.RENAMES)")
    if re.search(r"^DEPRECATED_FLAGS = \{", results_text, re.M):
        out.append("results.py holds a table of its own")
    for k, v in sorted(binary_table.items()):
        if renames.get(k) != v:
            out.append(f"the binaries alias {k} -> {v}, runlog.py says {renames.get(k)}")
    return out


def script_defaults(text):
    """{VAR: default} of every 'VAR="${VAR:-default}"' line; a nested '${A:-${B:-v}}' resolves to v."""
    out = {}
    for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)="\$\{\1:?-', text, re.M):
        i, depth = m.end(), 1
        while i < len(text) and depth:            # the matching brace of '${VAR:-'
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        val = text[m.end():i - 1]
        while True:
            n = re.fullmatch(r"\$\{[A-Z][A-Z0-9_]*:?-(.*)\}", val, re.S)
            if not n:
                break
            val = n.group(1)
        out.setdefault(m.group(1), val)
    return out


def help_header(text):
    """The comment block a script's --help prints: line 2 up to the first blank line."""
    lines = text.splitlines()[1:]
    out = []
    for ln in lines:
        if not ln.strip():
            break
        out.append(ln)
    return "\n".join(out)


def knob_table(readme):
    """[(knob names, default cell)] of the README table whose header starts '| knob (environment)'."""
    rows, on = [], False
    for ln in readme.splitlines():
        if ln.startswith("| knob (environment)"):
            on = True
            continue
        if on:
            if not ln.startswith("|"):
                break
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 3 or set(cells[0]) <= set("-: "):
                continue
            names = [n for n in re.findall(r"`([A-Z][A-Z0-9_]*)`", cells[0])]
            names += [n for n in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", re.sub(r"`[^`]*`", "", cells[0]))]
            rows.append((names, cells[-1]))
    return rows


def default_drift(rows, defaults):
    """[(knob, documented, script default)] where the table's plain value differs from the scripts'."""
    bad = []
    for names, cell in rows:
        items = [s.strip() for s in cell.split(",")] if len(names) > 1 else [cell]
        if len(items) != len(names):
            items = [cell] * len(names)
        for name, item in zip(names, items):
            if name not in defaults:
                continue
            want = defaults[name]
            if not want or re.search(r"\s|\$", want):       # empty or a list/expression: nothing to compare
                continue
            first = item.split()[0] if item.split() else ""
            plain = first.strip("`")
            if not (re.fullmatch(r"-?[0-9.]+", plain) or (first.startswith("`") and first.endswith("`"))):
                continue                                    # prose ('auto-tuned', 'off'): not a value
            if plain != want:
                bad.append((name, plain, want))
    return bad


def main():
    texts = [(str(f.relative_to(ROOT)), f.read_text(errors="replace")) for f in code_files()]
    code = "\n".join(t for _, t in texts)
    flags = code_flags(texts)
    docs = {d: read(d) for d in DOCS}
    headers = "\n".join(help_header(read(str(f.relative_to(ROOT)))) for f in sorted((ROOT / "scripts").glob("*.sh")))
    documented = "\n".join(docs.values()) + "\n" + headers

    src = read("src/user_options.cc")
    old_table = dict(DEPRECATED)
    old_table.update({k: " ".join(v) for k, v in cpp_switches(src).items()})
    old_table["--samples"] = "--subsamples"         # standard DTFE's total ('-s' too: no '--' to match)
    py_old = python_deprecated(texts)
    old_table.update({k: v for k, v in py_old.items() if k not in old_table})
    print("(A) every --flag in the docs is an option the code defines")
    for d, text in docs.items():
        miss = sorted(f for f in set(FLAG_RE.findall(text)) if f not in flags and f not in FOREIGN)
        check(not miss and text, f"{d}: {len(set(FLAG_RE.findall(text)))} flags, none unknown"
              + (f" -- unknown: {' '.join(miss)}" if miss else ""))
        old = deprecated_uses(text, old_table)
        check(not old, f"{d}: no deprecated spelling (the binaries' and the python CLIs')"
              + "".join(f" -- line {i}: {o} (use {c})" for o, c, i in old))

    print("(B) every DTFE_* name in the docs is read by the code")
    for d, text in docs.items():
        names = set(DTFE_RE.findall(text))
        miss = sorted(n for n in names if not re.search(r"\b" + n + r"\b", code))
        check(not miss, f"{d}: {len(names)} DTFE_* names, all in the code" + (f" -- missing: {' '.join(miss)}" if miss else ""))

    print("(C) the README knob table against the drivers")
    defaults = {}
    for s in DRIVERS:
        for k, v in script_defaults(read(s)).items():
            defaults.setdefault(k, v)
    rows = knob_table(docs["README.md"])
    check(len(rows) >= 8, f"the knob table is found ({len(rows)} rows)")
    knobs_in_code = set(defaults)
    for s in KNOB_SCRIPTS:
        knobs_in_code |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", read(s)))
    unknown = sorted({n for names, _ in rows for n in names
                      if n not in knobs_in_code and not re.search(r"\b" + n + r"\b", code)})
    check(not unknown, "every knob the table names is read by the code" + (f" -- unknown: {' '.join(unknown)}" if unknown else ""))
    drift = default_drift(rows, defaults)
    check(not drift, "every plain default in the table equals the scripts' default"
          + "".join(f" -- {k}: README {d}, scripts {w}" for k, d, w in drift))

    print("(D) every driver knob and configuration variable is documented")
    knobs = set()
    for s in KNOB_SCRIPTS:
        knobs |= set(script_defaults(read(s)))
    miss = sorted(k for k in knobs if not re.search(r"\b" + k + r"\b", documented))
    check(not miss, f"{len(knobs)} driver knobs, each in the README, COMMANDS.md or a script's --help"
          + (f" -- undocumented: {' '.join(miss)}" if miss else ""))
    env = set(re.findall(r"\$\{(DTFE_[A-Z0-9_]+)", read("scripts/config.sh")))
    env |= set(re.findall(r"""(?:environ(?:\.get)?\(|environ\[|getenv\()\s*["'](DTFE_[A-Z0-9_]+)""", read("python/config.py")))
    miss = sorted(e for e in env if not re.search(r"\b" + e + r"\b", documented))
    check(not miss, f"{len(env)} DTFE_* variables of scripts/config.sh and python/config.py, each documented"
          + (f" -- undocumented: {' '.join(miss)}" if miss else ""))

    print("(E) the run-log readers' table: runlog.py = results.py = run_settings, and the binaries' aliases in it")
    rl = load_runlog()
    src = read("src/user_options.cc")
    table = src[src.find("deprecatedOptions[] = {"):]
    cpp = {"--" + a: "--" + b for a, b in re.findall(r'\{ "([\w-]+)", +"([\w-]+)" \}', table[:table.find("};")])}
    drift = reader_table_drift(rl.RENAMES, rl.SWITCHES, read(PIPELINE), read("python/gui/results.py"), cpp)
    check(not drift and len(rl.RENAMES) >= 36, f"{len(rl.RENAMES)} renames and {len(rl.SWITCHES)} switches, one table"
          + "".join(f" -- {d}" for d in drift))
    check(all(rl.RENAMES.get(k) == v for k, v in DEPRECATED.items()), "the docs' deprecated names are rows of it")
    sw = cpp_switches(src)
    check(len(sw) == 3 and sw == {k: tuple(v) for k, v in rl.SWITCHES.items()},
          f"the binaries' merge switches (deprecatedSwitches) are runlog.py's SWITCHES ({sw})")

    print("(F) the renamed knobs: in the docs only on a 'deprecated' / 'old name' line, never in the README knob table")
    olds = knob_renames(texts)
    check(len(olds) >= 30, f"{len(olds)} old knob names found (env_knobs.h, the drivers' knob lines, python)")
    for d, text in docs.items():
        bad = old_knob_uses(text, olds)
        check(not bad, f"{d}: no old knob name outside a deprecation note"
              + "".join(f" -- line {i}: {o} (use {n})" for o, n, i in bad))
    in_table = sorted({n for names, _ in rows for n in names if n in olds})
    check(not in_table, "the README knob table names only the new knobs" + (f" -- old: {' '.join(in_table)}" if in_table else ""))

    print("(G) the binaries' help: kebab-case names, 'ps-' only in PS-DTFE, every -h name in --full-help")
    import subprocess
    helps = {}
    for b in ("DTFE", "PS-DTFE"):
        exe = ROOT / b
        if not exe.is_file():
            continue
        helps[b] = tuple(help_names(subprocess.run([str(exe), flag], capture_output=True, text=True).stdout)
                         for flag in ("--full-help", "-h"))
    if len(helps) < 2:
        skip("./DTFE and ./PS-DTFE are not both built (make DTFE PS-DTFE): the help rules are not checked")
    else:
        for b, (full, short) in helps.items():
            bad = help_style(full, short, b == "DTFE")
            check(len(full) >= 40 and not bad, f"{b}: {len(full)} --full-help names, {len(short)} -h names, the rules hold"
                  + "".join(f" -- {x}" for x in bad))
    print("(S) the checker itself finds planted drift (synthetic)")
    fake = ("x\n| knob (environment) | meaning | default |\n|---|---|---|\n"
            "| `GRID_SIZE` (or `-g`) | cells | 512 (`config.sh`) |\n"
            "| `FIELDS`, `OUTPUT_PREFIX`, `LAMBDA_TH` | f | the production set, `ps_out`, 0.3 |\n"
            "| `PARTITION` | split | auto-tuned |\n\nafter\n")
    sd = script_defaults('GRID_SIZE="${GRID_SIZE:-${DTFE_GRID_SIZE:-1024}}"\nOUTPUT_PREFIX="${OUTPUT_PREFIX:-ps_output}"   # c\n'
                         'LAMBDA_TH="${LAMBDA_TH:-0.3}"\nPARTITION="${PARTITION:-}"\nFIELDS="${FIELDS:-a b}"\n')
    check(sd == {"GRID_SIZE": "1024", "OUTPUT_PREFIX": "ps_output", "LAMBDA_TH": "0.3", "PARTITION": "", "FIELDS": "a b"},
          f"script_defaults resolves nested and commented defaults ({sd})")
    got = sorted(default_drift(knob_table(fake), sd))
    check(got == [("GRID_SIZE", "512", "1024"), ("OUTPUT_PREFIX", "ps_out", "ps_output")],
          f"a wrong number and a wrong backticked value are caught, prose and lists are not compared ({got})")
    check(FLAG_RE.findall("a -- b --grid 8 (--ps-gpu) x--y ---z") == ["--grid", "--ps-gpu"],
          "the flag pattern takes '--grid' and '(--ps-gpu)', not the ' -- ' dash or 'x--y'")
    check(code_flags([("src/user_options.cc", '("config,c", po::value)\n("NGP", "nearest")')]) >= {"--config", "--NGP"},
          "program_options names count as the binaries' flags")
    got = deprecated_uses("run it with --ps-gpu --grid 64\n--lambda_th 0.3 (deprecated: --lambda-th)\n`--full_help`\n")
    check(got == [("--ps-gpu", "--gpu", 1), ("--full_help", "--full-help", 3)],
          f"a deprecated spelling is caught, except on a line that says so ({got})")
    src = read("src/user_options.cc")
    table = src[src.find("deprecatedOptions[] = {"):]
    cpp = {"--" + a: "--" + b for a, b in re.findall(r'\{ "([\w-]+)", +"([\w-]+)" \}', table[:table.find("};")])}
    check(cpp == DEPRECATED, f"the deprecated table is the binaries' own (user_options.cc deprecatedOptions: {cpp})")
    good_pipe = 'old["--MpcUnit"] = "--mpc-unit"; old["--ps-gpu"] = "--gpu"\nsw["--ordered-merge"] = "--ps-merge index-order"\n'
    good_res = "from dtfelib import runlog as _runlog\nDEPRECATED_FLAGS = _runlog.RENAMES\n"
    ren, swi = {"--MpcUnit": "--mpc-unit", "--ps-gpu": "--gpu"}, {"--ordered-merge": ("--ps-merge", "index-order")}
    ok = reader_table_drift(ren, swi, good_pipe, good_res, {"--ps-gpu": "--gpu"})
    bad = reader_table_drift(dict(ren, **{"--NGP": "--ngp"}), swi, good_pipe.replace("index-order", "fixed-point"),
                             good_res + 'DEPRECATED_FLAGS = {"--a": "--b"}\n', {"--ps-window": "--window"})
    check(not ok and len(bad) == 4, f"the readers' table check passes equal copies and catches a missing pipeline row, a "
          f"changed switch, a table of results.py's own and a binary alias it lacks ({ok} {bad})")

    got = deprecated_uses("--ordered-merge here\n--samples 20 (deprecated)\n", {"--ordered-merge": "--ps-merge index-order",
                                                                              "--samples": "--subsamples"})
    check(got == [("--ordered-merge", "--ps-merge index-order", 1)], f"a second table (the switches) is checked too ({got})")
    got = python_deprecated([("python/a.py", 'deprecated_alias(p, "--z", "--redshifts")\ndeprecated_alias(p, "--prefix", "--root")'),
                             ("python/b.py", 'p.add_argument("--prefix", default="x")')])
    check(got == {"--z": "--redshifts"}, f"a python old name another parser still registers is not an old spelling ({got})")
    kr = knob_renames([("src/env_knobs.h", 'renamed[] = {\n    { "DTFE_OVERLAP", "DTFE_PS_OVERLAP", false },\n};'),
                       ("scripts/x.sh", "knob DTFE_GPU PS_GPU PS_METAL   # c\nknob -e PS_LAGRANGIAN_INPUT LAGRANGIAN_INPUT\n"
                                        "knob_merge PS_MERGE ORDERED_MERGE FIXED_POINT_MERGE\nknob_bool A B\n")])
    check(kr == {"DTFE_PS_OVERLAP": "DTFE_OVERLAP", "PS_GPU": "DTFE_GPU", "PS_METAL": "DTFE_GPU",
                 "LAGRANGIAN_INPUT": "PS_LAGRANGIAN_INPUT", "ORDERED_MERGE": "PS_MERGE", "FIXED_POINT_MERGE": "PS_MERGE"},
          f"the old knob names come from env_knobs.h and the drivers' knob lines, not knob_bool ({kr})")
    got = old_knob_uses("set PS_GPU=1\nPS_GPU (the older name) works\nDTFE_PS_GPU_X and $PS_GPU\n`PS_METAL`, deprecated\n",
                        {"PS_GPU": "DTFE_GPU", "PS_METAL": "DTFE_GPU"})
    check(got == [("PS_GPU", "DTFE_GPU", 1)], f"an old knob is caught outside a deprecation note only ({got})")
    hs = help_style({"grid", "ps-x", "MpcUnit"}, {"grid", "lambda_th"}, True)
    check(len(hs) == 4 and help_names("  -g [ --grid ] arg   x\n  --box arg   y\n        --not-an-entry\n") == {"grid", "box"},
          f"the help rules catch a capital, a '_', a 'ps-' name in standard DTFE and a '-h' name missing from --full-help ({hs})")

    print()
    print(f"RESULT: {passed} passed, {failed} failed" + (f", {skipped} skipped" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
