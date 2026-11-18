#!/usr/bin/env python3
"""Docs-drift check: what the documents (DOCS: README.md, the guides and the timings page in docs/,
demo/COMMANDS.md, metal/README.md) tell a user to type must exist in the code, and the knobs the drivers read must
be documented somewhere a user looks.

  A) every '--flag' in the documents is DECLARED: an entry of the binaries'
     option table (src/user_options.cc), a script's case arm (scripts/*.sh), a python parser's option (an
     add_argument / add_list_arg / add_*_arg call, make_parser's snap_option, a test of sys.argv), or one a test
     declares the same way for its own command line. A comment, a help text, a log reader or a command that
     passes the flag to a program is no evidence; FOREIGN names the flags of other programs the documents quote;
  B) every DTFE_* name in them is READ by the code (src/, scripts/, python/, the Makefile) or by a test itself: a
     script's '${NAME' or '$NAME', the Makefile's '$(NAME)', a python os.environ / getenv read, a C++ Env:: or
     getenv read or macro test (env_reads); a comment, an 'unset', a pop, an assignment or a string that only
     names the variable is no read;
  C) the knob table of docs/user-guide.md (KNOB_DOC): each knob it names is read by the code (B's reads), and
     where the table gives a plain value as the default it equals the scripts' 'VAR="${VAR:-default}"'
     (scripts/config.sh / run_ps_dtfe.sh / run_dtfe.sh);
  D) the other way round: every knob of the drivers and the pipeline, and every DTFE_* variable that
     scripts/config.sh or python/config.py reads, is named in one of the documents or in a script's --help
     header (the comment block a script's --help prints);
  E) every variable the documents set as 'NAME=value' (a knob, an environment or a make variable) is read by
     the code or a test (B's reads), except a shell variable a document sets and expands itself ('$NAME');
  F) every document of DOCS that lies in a folder .dockerignore leaves out of the Docker image has its own
     '!<document>' line there: the image's test stage runs this check, and a document it cannot read fails it;
  G) the binaries' own help (when ./DTFE and ./PS-DTFE are built; a SKIP line otherwise): no visible option name has
     a capital letter or a '_'; no name standard DTFE lists starts with 'ps-' (so every 'ps-' name is PS-only and
     every name both list lacks it); every '-h' name is a '--full-help' name of the same binary (not the other way
     round: '-h' is the short list by design); '-h' stays short, 20 to 35 option names, and carries no date
     (20dd-dd-dd) and no wall time (a number and then 's' or 'ms' as a word).
The synthetic half (S) feeds the same functions a made-up document with known drift, so a checker that stopped
finding anything fails too. Standard library only (runs under any python3 >= 3.10, no numpy).

    python3 tests/docs_drift_check.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# RELEASE_NOTES.md is left out on purpose: it names the removed options, variables and knobs (its table of old and
# new names, its Removed list, the examples of old settings), which rules A, B and E would fail by design
DOCS = ("README.md", "docs/user-guide.md", "docs/launcher-and-analysis.md", "docs/developer-guide.md",
        "docs/timings.md", "demo/COMMANDS.md", "metal/README.md")
KNOB_DOC = "docs/user-guide.md"                     # the document that holds the drivers' knob table (rule C)
DRIVERS = ("scripts/config.sh", "scripts/run_ps_dtfe.sh", "scripts/run_dtfe.sh")   # the knob table's defaults
KNOB_SCRIPTS = DRIVERS + ("scripts/run_ps_pipeline.sh",)
SKIP_PARTS = {"tmp", "cache", "__pycache__", "figures", "reference"}
CODE = ("src/**/*.cc", "src/**/*.cpp", "src/**/*.h", "src/**/*.hpp", "src/**/*.metal", "scripts/*.sh",
        "python/**/*.py", "Makefile", "metal/**/*.cpp")
TESTS = ("tests/**/*.py", "tests/**/*.sh")

# a flag: '--' right after a non-word character and right before a letter. A document may use ' -- ' as a dash,
# so the letter after the dashes is what tells a flag from punctuation.
FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*[A-Za-z0-9]")
DTFE_RE = re.compile(r"\bDTFE_[A-Z0-9_]*[A-Z0-9]\b")

# flags of other programs the docs quote (xcode-select's, python -m venv's), and the one binary option kept in camelCase
# on purpose: its 'redshiftSpace=' line is in every tessellation cache and globals record, whose text a later run
# compares verbatim
FOREIGN = {"--install", "--system-site-packages"}
KEPT_CAMEL = {"redshiftSpace"}
CPP = (".cc", ".cpp", ".h", ".hpp", ".metal")

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


def source_texts(patterns):
    """[(path relative to the repository, text)] of the files the glob patterns name (CODE or TESTS), this file and
    the folders of SKIP_PARTS left out."""
    seen, out = set(), []
    for pat in patterns:
        for f in sorted(ROOT.glob(pat)):
            if f in seen or SKIP_PARTS & set(f.relative_to(ROOT).parts) or f.name == Path(__file__).name:
                continue
            seen.add(f)
            out.append((str(f.relative_to(ROOT)), f.read_text(errors="replace")))
    return out


def code_lines(name, text):
    """'text' without its comments, which are never evidence: whole comment lines ('#' in a script, python or the
    Makefile; '//', '/*' or '*' in C++ and Metal) and a script's or python's comment after code (' # ...')."""
    if name.endswith(CPP):
        return re.sub(r"^\s*(?://|/\*|\*).*$", "", text, flags=re.M)
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)
    return re.sub(r"[ \t]#(?![!{])[^\n]*", "", text) if name.endswith((".sh", ".py")) else text


# a script's case arm: words separated by '|' and closed by ')' -- '--grid)', '-h|--help)'
ARM_RE = re.compile(r"(?:^|(?<=\s))([^\s|()]+(?:\|[^\s|()]+)*)\)", re.M)


def shell_arms(code):
    """The '--' names of a script's case arms: a pattern closed by ')' that follows ';;' or 'in' (the arm before it,
    or 'case ... in') on its line or the lines before. The closing line of an array ('exact_args=(--exact)', a list
    ended on a line of its own) or of a command ('$(cmd --x)') is no arm."""
    out = set()
    for m in ARM_RE.finditer(code):
        names = [w for w in m.group(1).split("|") if re.fullmatch(r"--[A-Za-z][\w-]*", w)]
        before = code[:m.start()].rstrip()
        if names and (before.endswith((";;", ";&")) or re.search(r"(?:^|\s)in$", before)):
            out.update(names)
    return out


def declared_flags(texts):
    """Every option name the code DECLARES: an entry of the binaries' option table (src/user_options.cc, a line
    '("grid,g", ...)': the names have no dashes there), a script's case arm (shell_arms), a python parser's names
    (the literal names before the first keyword of an add_argument / add_list_arg / add_*_arg call, make_parser's
    snap_option) and a python test of its own command line ('"--smoke" in sys.argv'). A comment, a help text, a log
    reader, a test of a command built for a program ('"--gpu" in argv') or a command that passes the flag to a
    program declares nothing."""
    flags = set()
    for name, text in texts:
        code = code_lines(name, text)
        if name.endswith("user_options.cc"):
            flags.update("--" + m for m in re.findall(r'^\s*\(\s*"([A-Za-z][\w-]*)(?:,\w)?"\s*,', code, re.M))
        elif name.endswith(".sh"):
            flags |= shell_arms(code)
        elif name.endswith(".py"):
            q = r"""["'](--[A-Za-z][\w-]*)["']"""
            for m in re.finditer(r"\badd_(?:argument|\w+_arg)\(([^=)]*)", code):
                flags.update(re.findall(q, m.group(1)))
            flags.update(re.findall(r"\bsnap_option\s*(?::\s*str\s*)?=\s*" + q, code))
            flags.update(re.findall(q + r"\s+(?:not\s+)?in\s+sys\.argv\b", code))
    return flags


def env_reads(texts):
    """The variables the code READS: a script's expansion ('${NAME', '$NAME'), the Makefile's ('$(NAME)', '${NAME}',
    'ifdef NAME'), a python os.environ / getenv read, a C++ or Metal Env:: or getenv read and a macro test ('#ifdef
    NAME', 'defined(NAME)'). A read may go through a constant that holds the name (python 'NAME_ENV = "DTFE_X"',
    C++ 'char const *const OFF = "DTFE_X"') or, in C++, a function that hands its 'char const *' parameter to an
    Env:: read. A comment, an 'unset', a pop, an assignment or a string that only names the variable is no read."""
    names, consts, via_const, wrappers, calls = set(), {}, set(), set(), []
    q = r"""["']([A-Za-z_]\w*)["']"""
    for name, text in texts:
        code = code_lines(name, text)
        if name.endswith(".sh"):
            names.update(re.findall(r"\$\{?([A-Za-z_]\w*)", code))
        elif name.endswith(".py"):
            names.update(re.findall(r"\benviron\.get\(\s*" + q, code))
            names.update(re.findall(r"\benviron\[\s*" + q + r"\s*\](?!\s*=(?!=))", code))
            names.update(re.findall(r"\bgetenv\(\s*" + q, code))
            names.update(re.findall(q + r"\s+(?:not\s+)?in\s+(?:os\.)?environ\b", code))
            consts.update(re.findall(r"^([A-Z][A-Z0-9_]*)\s*=\s*" + q + r"\s*$", code, re.M))
            via_const.update(re.findall(r"\b(?:environ|env)(?:\.get\(|\[)\s*([A-Z][A-Z0-9_]*)\b(?!\s*\]\s*=(?!=))",
                                        code))
        elif name.endswith(CPP):
            names.update(re.findall(r"""\b(?:Env::\w+|getenv)\(\s*\{?\s*"([A-Za-z_]\w*)\"""", code))
            names.update(re.findall(r"^\s*#\s*ifn?def\s+(\w+)", code, re.M))
            names.update(re.findall(r"\bdefined\s*\(\s*(\w+)\s*\)", code))
            consts.update(re.findall(r"""\b(\w+)\s*=\s*"([A-Za-z_]\w*)"\s*;""", code))
            via_const.update(re.findall(r"\bEnv::\w+\(\s*([A-Za-z_]\w*)\b", code))
            for m in re.finditer(r"\b(\w+)\s*\(\s*char\s+const\s*\*\s*(\w+)\s*\)\s*\{", code):
                if re.search(r"\bEnv::\w+\(\s*" + m.group(2) + r"\b", code[m.end():m.end() + 800]):
                    wrappers.add(m.group(1))
            calls += re.findall(r"""\b(\w+)\(\s*"([A-Za-z_]\w*)\"""", code)
        else:                                       # the Makefile
            names.update(re.findall(r"\$[({]([A-Za-z_]\w*)[)}:]", code))
            names.update(re.findall(r"^\s*ifn?def\s+(\w+)", code, re.M))
    return names | {consts[c] for c in via_const if c in consts} | {v for f, v in calls if f in wrappers}


def assigned_knobs(text):
    """The variables a document sets as 'NAME=value' (a knob, an environment or a make variable), but those it
    expands itself ('$NAME', '${NAME}': its own shell variables, set for the commands after them)."""
    own = set(re.findall(r"\$\{?([A-Za-z_]\w*)", text))
    return {n for n in re.findall(r"(?<![\w$-])([A-Z][A-Z0-9_]+)=(?!=)", text) if n not in own}


def unknown_flags(text, flags):
    """The '--flags' of a document that no declaration names (FOREIGN aside), sorted."""
    return sorted(f for f in set(FLAG_RE.findall(text)) if f not in flags and f not in FOREIGN)


def unread(names, reads):
    """Of these variable names, the ones no code reads, sorted."""
    return sorted(n for n in set(names) if n not in reads)


def docker_missing(docs, ignore):
    """Of these documents, the ones a .dockerignore text leaves out of the image: the document's top folder has
    a line of its own ('docs/') and the document has no '!<document>' line."""
    lines = {line.strip() for line in ignore.splitlines()}
    return sorted(d for d in docs if "/" in d and d.split("/")[0] + "/" in lines and "!" + d not in lines)


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


# the short help's size (option names) and the stale words it must not carry: a date, a wall time
SHORT_HELP_NAMES = (20, 35)
DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
WALL_RE = re.compile(r"(?<![\w.^])\d+(?:\.\d+)? (?:s|ms)(?![\w-])")


def short_help_shape(text, prog=""):
    """[problem] of one binary's '-h' text: fewer than 20 or more than 35 option names (help_names), a date
    (20dd-dd-dd), a wall time ('431 s', '50 ms'). The text is scanned with its whitespace folded to single blanks, so
    a number and its unit that the 80-column wrap put on two lines are still one wall time. prog: the program's path
    as the usage line prints it (argv[0]), left out of the scan (a folder name with a date is not the help's)."""
    lo, hi = SHORT_HELP_NAMES
    n = len(help_names(text))
    out = [] if lo <= n <= hi else [f"-h lists {n} option names, not {lo} to {hi}"]
    flat = " ".join((text.replace(prog, " ") if prog else text).split())
    out += [f"a date in -h: {d}" for d in DATE_RE.findall(flat)]
    out += [f"a wall time in -h: {t}" for t in WALL_RE.findall(flat)]
    return out


def script_defaults(text):
    """{VAR: default} of every 'VAR="${VAR:-default}"' line."""
    out = {}
    for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)="\$\{\1:?-', text, re.M):
        i, depth = m.end(), 1
        while i < len(text) and depth:            # the matching brace of '${VAR:-'
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out.setdefault(m.group(1), text[m.end():i - 1])
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


def knob_table(doc):
    """[(knob names, default cell)] of the document's table whose header starts '| knob (environment)'."""
    rows, on = [], False
    for ln in doc.splitlines():
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
    texts, tests = source_texts(CODE), source_texts(TESTS)
    flags = declared_flags(texts + tests)
    reads, test_reads = env_reads(texts), env_reads(tests)
    docs = {d: read(d) for d in DOCS}
    headers = "\n".join(help_header(read(str(f.relative_to(ROOT)))) for f in sorted((ROOT / "scripts").glob("*.sh")))
    documented = "\n".join(docs.values()) + "\n" + headers

    print("(A) every --flag in the docs is an option the code declares")
    for d, text in docs.items():
        miss = unknown_flags(text, flags)
        check(not miss and text, f"{d}: {len(set(FLAG_RE.findall(text)))} flags, none unknown"
              + (f" -- unknown: {' '.join(miss)}" if miss else ""))

    print("(B) every DTFE_* name in the docs is read by the code")
    for d, text in docs.items():
        names = set(DTFE_RE.findall(text))
        miss = unread(names, reads | test_reads)
        check(not miss, f"{d}: {len(names)} DTFE_* names, all read"
              + (f" -- not read: {' '.join(miss)}" if miss else ""))

    print(f"(C) the knob table of {KNOB_DOC} against the drivers")
    defaults = {}
    for s in DRIVERS:
        for k, v in script_defaults(read(s)).items():
            defaults.setdefault(k, v)
    rows = knob_table(docs[KNOB_DOC])
    check(len(rows) >= 8, f"the knob table is found ({len(rows)} rows)")
    unknown = unread([n for names, _ in rows for n in names], reads)
    check(not unknown, "every knob the table names is read by the code"
          + (f" -- not read: {' '.join(unknown)}" if unknown else ""))
    drift = default_drift(rows, defaults)
    check(not drift, "every plain default in the table equals the scripts' default"
          + "".join(f" -- {k}: {KNOB_DOC} {d}, scripts {w}" for k, d, w in drift))

    print("(D) every driver knob and configuration variable is documented")
    knobs = set()
    for s in KNOB_SCRIPTS:
        knobs |= set(script_defaults(read(s)))
    miss = sorted(k for k in knobs if not re.search(r"\b" + k + r"\b", documented))
    check(not miss, f"{len(knobs)} driver knobs, each in a document or a script's --help"
          + (f" -- undocumented: {' '.join(miss)}" if miss else ""))
    env = set(re.findall(r"\$\{(DTFE_[A-Z0-9_]+)", read("scripts/config.sh")))
    env |= set(re.findall(r"""(?:environ(?:\.get)?\(|environ\[|getenv\()\s*["'](DTFE_[A-Z0-9_]+)""", read("python/config.py")))
    miss = sorted(e for e in env if not re.search(r"\b" + e + r"\b", documented))
    check(not miss, f"{len(env)} DTFE_* variables of scripts/config.sh and python/config.py, each documented"
          + (f" -- undocumented: {' '.join(miss)}" if miss else ""))

    print("(E) every NAME=value the docs set is read by the code")
    for d, text in docs.items():
        names = assigned_knobs(text)
        miss = unread(names, reads | test_reads)
        check(not miss, f"{d}: {len(names)} variables set as NAME=value, all read"
              + (f" -- not read: {' '.join(miss)}" if miss else ""))

    print("(F) the Docker image holds every document this check reads")
    miss = docker_missing(DOCS, read(".dockerignore"))
    check(not miss, ".dockerignore: every document of a folder it leaves out has its '!' line"
          + (f" -- no line for: {' '.join(miss)}" if miss else ""))

    print("(G) the binaries' help: kebab-case names, 'ps-' only in PS-DTFE, every -h name in --full-help, -h short "
          "(20 to 35 names) with no date and no wall time")
    import subprocess
    helps = {}
    for b in ("DTFE", "PS-DTFE"):
        exe = ROOT / b
        if not exe.is_file():
            continue
        full, short = [subprocess.run([str(exe), flag], capture_output=True, text=True).stdout
                       for flag in ("--full-help", "-h")]
        helps[b] = (help_names(full), help_names(short), short_help_shape(short, str(exe)))
    if len(helps) < 2:
        skip("./DTFE and ./PS-DTFE are not both built (make DTFE PS-DTFE): the help rules are not checked")
    else:
        for b, (full, short, shape) in helps.items():
            bad = help_style(full, short, b == "DTFE")
            check(len(full) >= 40 and not bad, f"{b}: {len(full)} --full-help names, {len(short)} -h names, the rules hold"
                  + "".join(f" -- {x}" for x in bad))
            check(not shape, f"{b}: -h is the short list ({len(short)} option names, {SHORT_HELP_NAMES[0]} to "
                  f"{SHORT_HELP_NAMES[1]}), with no date and no wall time" + "".join(f" -- {x}" for x in shape))
    print("(S) the checker itself finds planted drift (synthetic)")
    fake = ("x\n| knob (environment) | meaning | default |\n|---|---|---|\n"
            "| `GRID_SIZE` (or `-g`) | cells | 512 (`config.sh`) |\n"
            "| `FIELDS`, `OUTPUT_PREFIX`, `LAMBDA_TH` | f | the production set, `ps_out`, 0.3 |\n"
            "| `PARTITION` | split | auto-tuned |\n\nafter\n")
    sd = script_defaults('GRID_SIZE="${GRID_SIZE:-1024}"\nOUTPUT_PREFIX="${OUTPUT_PREFIX:-ps_output}"   # c\n'
                         'LAMBDA_TH="${LAMBDA_TH:-0.3}"\nPARTITION="${PARTITION:-}"\nFIELDS="${FIELDS:-a b}"\n')
    check(sd == {"GRID_SIZE": "1024", "OUTPUT_PREFIX": "ps_output", "LAMBDA_TH": "0.3", "PARTITION": "", "FIELDS": "a b"},
          f"script_defaults reads plain, commented and empty defaults ({sd})")
    got = sorted(default_drift(knob_table(fake), sd))
    check(got == [("GRID_SIZE", "512", "1024"), ("OUTPUT_PREFIX", "ps_out", "ps_output")],
          f"a wrong number and a wrong backticked value are caught, prose and lists are not compared ({got})")
    got = docker_missing(("README.md", "docs/a.md", "docs/b.md", "demo/c.md", "metal/d.md"),
                         "# c\ndocs/\n!docs/a.md\ndemo/\n!demo/x.gadget\n")
    check(got == ["demo/c.md", "docs/b.md"],
          f"a document of a folder .dockerignore leaves out, without its '!' line, is caught ({got})")
    check(FLAG_RE.findall("a -- b --grid 8 (--gpu) x--y ---z") == ["--grid", "--gpu"],
          "the flag pattern takes '--grid' and '(--gpu)', not the ' -- ' dash or 'x--y'")
    # (A) a removed option that only a comment, a message, a log reader, an array or a command still names is unknown
    decl = declared_flags([
        ("src/user_options.cc", '    ("config,c", po::value)\n    ("ngp", "nearest")\n'
                                '    // --gone-option was another name\n    warning << "--gone-option is not read";\n'),
        ("scripts/a.sh", 'case "$1" in\n    -h|--help) usage ;;\n    --sim) s=1 ;; --sim-dir) d=1 ;;\n'
                         '    # --gone-option) a comment\nesac\nargs=(--array-word)\nlist=(\n    a\n    --list-end)\n'
                         'x="$(tool --command-word)"\n./PS-DTFE in out --passed-word\n'),
        ("python/b.py", 'p.add_argument("-o", "--out", help="not --help-word")\nadd_list_arg(p, "--maps", dest="m")\n'
                        'make_parser(snap_option="--start-snap")\nif "--smoke" in sys.argv:\n    pass\n'
                        'old = rec.has("--gone-option")   # a log reader\nrun([b, "--grid", "8"])\n'
                        'built = "--built-word" in argv\n')])
    doc = "Type `--gone-option`, --ngp, --sim-dir, --maps or --start-snap.\n"
    check(decl == {"--config", "--ngp", "--help", "--sim", "--sim-dir", "--out", "--maps", "--start-snap", "--smoke"}
          and unknown_flags(doc, decl) == ["--gone-option"],
          f"(A) declarations only: option table entries, case arms (two on a line too), parser calls, snap_option, a "
          f"sys.argv test; a comment, a message, a log reader, an array, a command, a built command's test or a help "
          f"text is none, so a document naming a removed option fails ({sorted(decl)} {unknown_flags(doc, decl)})")
    # (B, C, E) a variable that only a comment, an 'unset', a pop, an assignment or a help text names is not read
    reads = env_reads([
        ("scripts/a.sh", 'g="${GRID_KNOB:-512}"\n[ "$DTFE_READ_A" = 1 ]\n# ${DTFE_GONE_A} ${GONE_KNOB}\n'
                         'unset DTFE_GONE_B\n'),
        ("python/b.py", 'os.environ.get("DTFE_READ_B")\nNAME_ENV = "DTFE_READ_C"\nv = env.get(NAME_ENV, "")\n'
                        'env.pop("DTFE_GONE_C", None)\nos.environ["DTFE_GONE_D"] = "1"\nh = "set DTFE_GONE_E"\n'),
        ("src/c.h", 'bool a = Env::flag( "DTFE_READ_D" );\nstatic char const *const OFF = "DTFE_READ_E";\n'
                    'bool off = not Env::flag( OFF, true );\n'
                    'inline long team(char const *env) { return Env::integer( env, 0 ); }\n'
                    'long n = team( "DTFE_READ_F" );\n// Env::flag( "DTFE_GONE_F" )\n#ifndef MACRO_READ\n#endif\n'),
        ("Makefile", 'ifeq ($(MAKE_READ),1)\nendif\n# $(MAKE_GONE)\n')])
    want = {"GRID_KNOB", "DTFE_READ_A", "DTFE_READ_B", "DTFE_READ_C", "DTFE_READ_D", "DTFE_READ_E", "DTFE_READ_F",
            "MACRO_READ", "MAKE_READ"}
    doc = ("Set DTFE_GONE_A=1 or DTFE_READ_A=1: `GONE_KNOB=1 GRID_KNOB=8 ./scripts/run_ps_dtfe.sh`, make MAKE_READ=1.\n"
           "SIMDIR=/data/sim\nls $SIMDIR\n")
    gone_knobs = unread([n for names, _ in knob_table(fake.replace("`PARTITION`", "`GONE_KNOB`")) for n in names],
                        reads | {"FIELDS", "OUTPUT_PREFIX", "LAMBDA_TH", "GRID_SIZE"})
    check(want <= reads and not [n for n in reads if "GONE" in n]
          and unread(DTFE_RE.findall(doc), reads) == ["DTFE_GONE_A"]
          and unread(assigned_knobs(doc), reads) == ["DTFE_GONE_A", "GONE_KNOB"] and gone_knobs == ["GONE_KNOB"],
          f"(B, C, E) reads only: expansions, os.environ and Env:: reads (through a constant or a wrapper too), macro "
          f"and make tests; a comment, an 'unset', a pop, an assignment or a string is none, so a document or the "
          f"knob table naming a variable nothing reads fails, and a document's own shell variable is left out "
          f"({sorted(want - reads)} {unread(assigned_knobs(doc), reads)} {gone_knobs})")
    hs = help_style({"grid", "ps-x", "BoxSize"}, {"grid", "max_grid"}, True)
    check(len(hs) == 4 and help_names("  -g [ --grid ] arg   x\n  --box arg   y\n        --not-an-entry\n") == {"grid", "box"},
          f"the help rules catch a capital, a '_', a 'ps-' name in standard DTFE and a '-h' name missing from --full-help ({hs})")
    prog = "/a 2026-01-02 folder/DTFE"            # argv[0] in the usage line: a dated folder is not the help's date
    entries = "".join(f"  --opt-{i} arg               a line {i}.\n" for i in range(24))
    clean = (f"Usage: {prog} <input file> <output root> [options]\nTwo modes: --exact and --sampled.\n\nInput:\n"
             "  -i [ --input-format ] arg   105 Gadget HDF5 (the default) or 101 binary Gadget.\n"
             "  --padding arg               5 here; the drivers pass 15: 5 is too thin near a face.\n" + entries)
    dev = "".join(f"  --dev-{i}               a developer option.\n" for i in range(10))
    shapes = {"clean": short_help_shape(clean, prog), "path": short_help_shape(clean),
              "long": short_help_shape(clean + dev, prog), "few": short_help_shape("".join(entries.splitlines(True)[:5])),
              "at19": short_help_shape("".join(entries.splitlines(True)[:19])),
              "at20": short_help_shape("".join(entries.splitlines(True)[:20])),
              "at35": short_help_shape(clean + "".join(dev.splitlines(True)[:9]), prog),
              "date": short_help_shape(clean.replace("a line 3.", "the default since 2026-11-09."), prog),
              "time": short_help_shape(clean.replace("a line 5.", "it ran in 431\n                              s on the "
                                                                  "CPU and 75 ms on the GPU."), prog)}
    check(shapes["clean"] == [] and shapes["path"] == ["a date in -h: 2026-01-02"]
          and shapes["long"] == ["-h lists 36 option names, not 20 to 35"]
          and shapes["few"] == ["-h lists 5 option names, not 20 to 35"] and shapes["date"] == ["a date in -h: 2026-11-09"]
          and shapes["at19"] == ["-h lists 19 option names, not 20 to 35"] and shapes["at20"] == [] and shapes["at35"] == []
          and shapes["time"] == ["a wall time in -h: 431 s", "a wall time in -h: 75 ms"],
          f"the short-help guard passes a 26-name -h (its path's date left out, '5 is', '15:' and '101 binary' no times) and "
          f"catches a 36-name list, a 5-name list, a date and a wall time, one split by the wrap ({shapes})")

    print()
    print(f"RESULT: {passed} passed, {failed} failed" + (f", {skipped} skipped" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
