"""The pipeline cache's housekeeping, behind 'analyze.py cache': every namespace of config.CACHE_DIR with its size and
whether the current code can still reach it (cache_inventory), the listing and the prune (cache_command), and the
move to another folder with the setting recorded (move_cache), refused while a figure script, an analysis, the
launcher or a DTFE binary may be using the cache (cache_users). Moved out of analyze.py, which re-exports every name
(the 'cache' subcommand stays there)."""

import os
import re
import subprocess
import time
from pathlib import Path

import config


# ---------------------------------------------------------------- cache (list / prune the namespaces)

# The physical smoothings (Mpc) the current code asks for by default: the cross-box comparison's sigma,
# config.CROSSBOX_SMOOTHING_MPC (the one number plot_void_crossbox, the launcher and this read). A
# namespace made with another Mpc value (2.16: r512_s21.4043_f23 next to 2.1625's
# r512_s21.4291_f23) is reached only by typing that value again; 'cache --keep-mpc' and DTFE_SIGMA_MPC add values.
CACHE_SMOOTHING_MPC = (config.CROSSBOX_SMOOTHING_MPC,)
NAMESPACE_RE = r"^r(?P<res>\d+)_s(?P<sig>[^_]+)_f(?P<fp>\d+)_(?P<crit>.+)_g(?P<g>[^_]+)_v(?P<ver>\d+)$"


def _dir_size(path):
    """(bytes, files) under 'path' (files only, links not followed)."""
    total, n = 0, 0
    for p in Path(path).rglob('*'):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
            n += 1
    return total, n


def _human(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if nbytes < 1024 or unit == 'GB':
            return f"{nbytes:.0f} {unit}" if unit == 'B' else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, OverflowError):
        return True                                 # another user's process, or not a pid: not ours to call dead
    return True


FIXED_R_CELLS = (10.0, 5.0, 3.0, 2.0)     # see cache_inventory


def cache_inventory(root=None, keep_mpc=(), keep_cells=()):
    """Every entry of the pipeline cache (config.CACHE_DIR) with a verdict: a list of
    {'path', 'kind' ('namespace' / 'file'), 'bytes', 'files', 'newest', 'prune' (bool), 'reason'}.

    A NAMESPACE (r<N>_s<sigma>_f<footprint>_<criterion>_g<threshold>_v<version>, pipeline._param_hash) is UNREACHABLE,
    and pruned, when
      * its version is not the current one ('_v1': the format before the h-free unit overhaul; nothing reads it), or
      * no smoothing the current code asks for produces its sigma/footprint: the production one
        (config.PRODUCTION_SMOOTHING),
        a --smooth in cells with the production footprint for config.SMOOTHING_COMPARISON_SIGMAS (+ keep_cells), or a
        smoothing in Mpc -- CACHE_SMOOTHING_MPC, DTFE_SIGMA_MPC and keep_mpc -- for a simulation whose files it holds
        (round(mpc / cell, 9) cells, config.footprint_for), the way config / pipeline.set_smoothing spell it.
    Another resolution, criterion or gradient threshold is a config change away: kept. Inside a kept namespace a FILE
    is pruned when it is an interrupted save's temp file ('<name>.npz.<pid>.tmp.npz', pipeline._save_npz) whose
    process is gone, or a fate link (lagrangian) of an older LINK_VERSION, which cached_link never serves."""
    import re
    root = Path(config.CACHE_DIR if root is None else root)
    out = []
    if not root.is_dir():
        return out
    from dtfelib import pipeline
    m_now = re.match(NAMESPACE_RE, pipeline._param_hash())
    sig0, fp0 = config.PRODUCTION_SMOOTHING
    # the fixed-R size function's catalogues (plot_void_population.FIXED_R_SMOOTH_CELLS 10 5 3, and the 3 2 a coarse
    # box like TNG300 is run at: FIXED_R_CELLS) and its --size-smooth 5 are live too
    cells_ok = {f"{float(c):g}" for c in (sig0, *config.SMOOTHING_COMPARISON_SIGMAS, *FIXED_R_CELLS, *keep_cells)}
    mpcs = list(CACHE_SMOOTHING_MPC) + [float(m) for m in keep_mpc]
    if os.environ.get('DTFE_SIGMA_MPC'):
        mpcs.append(float(os.environ['DTFE_SIGMA_MPC']))
    try:
        from dtfelib.lagrangian import LINK_VERSION
    except ImportError:                             # h5py / scipy missing: the link check is left out, not guessed
        LINK_VERSION = None
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = re.match(NAMESPACE_RE, d.name)
        nbytes, nfiles = _dir_size(d)
        files = [p for p in d.iterdir() if p.is_file()]
        newest = max((p.stat().st_mtime for p in files), default=d.stat().st_mtime)
        entry = {'path': d, 'kind': 'namespace', 'bytes': nbytes, 'files': nfiles, 'newest': newest}
        if m is None:
            out.append(dict(entry, prune=False, reason="not a cache namespace name: left alone"))
            continue
        sims = sorted({p.name.split('_')[0] for p in files if p.suffix == '.npz' and not p.name[:1].isdigit()})
        if m['ver'] != m_now['ver']:
            out.append(dict(entry, prune=True, reason=f"cache format v{m['ver']}, today's is v{m_now['ver']}: nothing "
                                                       "reads it"))
            continue
        if (m['res'], m['crit'], m['g']) != (m_now['res'], m_now['crit'], m_now['g']):
            out.append(dict(entry, prune=False, reason="another resolution / criterion / gradient threshold (config): "
                                                        "kept"))
            continue
        fp = int(m['fp'])
        why = None
        if m['sig'] in cells_ok and fp == int(fp0):
            why = "production smoothing" if m['sig'] == f"{float(sig0):g}" else f"--smooth {m['sig']} (cells)"
        else:
            for sim in sims:
                box = config.SIMULATION_BOX_MPC.get(sim)
                if box is None:
                    continue
                for mpc in mpcs:
                    cells = round(mpc / (box / config.FIELD_RESOLUTION), 9)
                    if f"{cells:g}" == m['sig'] and config.footprint_for(cells) == fp:
                        why = f"--smooth-mpc {mpc:g} on {sim}"
        if why is None:
            held = ', '.join(sims) if sims else 'no simulation files'
            out.append(dict(entry, prune=True, reason=f"sigma {m['sig']} cells / footprint {fp}: no current smoothing "
                                                       f"(cells {sorted(cells_ok, key=float)}, Mpc {mpcs}) gives it for "
                                                       f"{held}"))
            continue
        out.append(dict(entry, prune=False, reason=why))
        for p in sorted(files):
            st = p.stat()
            fe = {'path': p, 'kind': 'file', 'bytes': st.st_size, 'files': 1, 'newest': st.st_mtime}
            t = re.match(r"^.+\.npz\.(\d+)\.tmp\.npz$", p.name)
            if t and not _pid_alive(t.group(1)):
                out.append(dict(fe, prune=True, reason=f"an interrupted save's temp file (pid {t.group(1)} is gone)"))
            elif LINK_VERSION is not None and re.match(r"^.+_fate(_excl)?_\d{3}_\d{3}\.npz$", p.name):
                try:
                    import numpy as np
                    with np.load(p) as z:
                        v = int(z['link_version']) if 'link_version' in z.files else 1
                except Exception as e:              # a cut file: cached_link relinks it too
                    out.append(dict(fe, prune=True, reason=f"an unreadable fate link ({type(e).__name__})"))
                    continue
                if v != LINK_VERSION:
                    out.append(dict(fe, prune=True, reason=f"fate link version {v}, today's is {LINK_VERSION}: never "
                                                           "served (relinked on the next census)"))
    return out


def cache_command(prune=False, yes=False, keep_mpc=(), keep_cells=(), root=None):
    """'analyze.py cache': every namespace with its size and verdict; --prune lists only what would go (a DRY RUN);
    --prune --yes deletes it. Returns the bytes listed for (or freed by) the prune."""
    import shutil
    root = Path(config.CACHE_DIR if root is None else root)
    inv = cache_inventory(root, keep_mpc, keep_cells)
    print(f"Pipeline cache {root}" + ("" if root.is_dir() else " (absent)"))
    gone = [e for e in inv if e['prune']]
    # the files of one namespace with one reason are one line (27 old fate links are not 27 lines)
    shown, groups = [], {}
    for e in (gone if prune else inv):
        if e['kind'] == 'namespace':
            shown.append(e)
        elif e['prune']:
            key = (e['path'].parent, e['reason'])
            if key not in groups:
                groups[key] = dict(e, bytes=0, files=0, newest=0.0, names=[])
                shown.append(groups[key])
            g = groups[key]
            g['bytes'] += e['bytes']
            g['files'] += 1
            g['newest'] = max(g['newest'], e['newest'])
            g['names'].append(e['path'].name)
    for e in shown:
        if e['kind'] == 'namespace':
            name = e['path'].name
        else:
            ex = e['names'][0] + (f" (+{len(e['names']) - 1} more)" if len(e['names']) > 1 else "")
            name = f"  {e['path'].parent.name}/{ex}"
        mark = ("DELETE " if yes else "would delete ") if (prune and e['prune']) else ("prune  " if e['prune'] else "keep   ")
        print(f"  {mark}{name:44s} {_human(e['bytes']):>9s} {e['files']:5d} file(s), newest "
              f"{time.strftime('%Y-%m-%d', time.localtime(e['newest']))}: {e['reason']}")
    total = sum(e['bytes'] for e in gone)
    if not gone:
        print("  nothing to prune")
        return 0
    if not prune:
        print(f"\n{len(gone)} unreachable entr{'y' if len(gone) == 1 else 'ies'}, {_human(total)}: "
              "'analyze.py cache --prune' lists them, '--prune --yes' deletes them.")
        return total
    if not yes:
        print(f"\nDRY RUN: {len(gone)} entr{'y' if len(gone) == 1 else 'ies'}, {_human(total)} would be deleted; "
              "add --yes to delete them.")
        return total
    freed = 0
    for e in gone:
        p = e['path']
        if root.resolve() not in p.resolve().parents:   # never outside the cache, whatever a link points at
            print(f"  !! {p} is not inside {root}: left alone")
            continue
        try:
            shutil.rmtree(p) if e['kind'] == 'namespace' else p.unlink()
            freed += e['bytes']
        except OSError as err:
            print(f"  !! {p}: {err}")
    print(f"\nDeleted {_human(freed)} of {_human(total)}.")
    return freed


# ---------------------------------------------------------------- cache --move

# what reads or writes the cache: the figure / analysis scripts, the launcher (its Explore tab and Runs status read
# catalogues in-process, its jobs run the scripts) and the binaries a Pipeline job runs before its analysis. A python
# whose command line names one of these is a user; the binaries are found by name, as the launcher's own check does
# (runspec.dtfe_processes' pgrep pattern). The interpreter may also be the macOS host bundle scripts/gui.sh starts the
# launcher in ('<venv>/Lamina.app/Contents/MacOS/Lamina <repo>/python/gui/app.py', or 'DTFE Launcher' for a host made
# by an older install_gui_app.sh; its sys.executable,
# runspec.PYTHON's last fallback for the jobs), and the launcher is matched by its script path whatever runs it: a
# pattern that wants a 'python...' word before it misses the usual way the launcher is open.
_CACHE_USER_SCRIPTS = re.compile(r"(?i)(?:python[^\s/]*|Lamina|DTFE Launcher)\s.*?(?:\banalyze\.py|\bselftest\.py|"
                                 r"\bplot_\w+\.py|\bvoid_outlines\.py|\bvoid_cache\.py)(?:\s|$)"
                                 r"|(?:^|[\s/])gui/app\.py(?:\s|$)")
_DTFE_BINARY = r"^(PS-)?DTFE(-.*)?$"


def _ps_table():
    """{pid: (ppid, args)} of every process (one ps call; -A and -o spelled the same on macOS and Linux). {} without ps."""
    try:
        text = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,args="], capture_output=True, text=True,
                              timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) > 2 else "")
    return out


def cache_users(table=None, dtfe_names=None, me=None):
    """[(pid, what)] of the processes that may be using the pipeline cache now: a python running an analysis or
    figure script or the launcher, and the DTFE / PS-DTFE binaries (a Pipeline job runs one, then its analysis).
    This process and its ANCESTORS are left out: 'analyze.py cache --move' run by the launcher's setup is the
    launcher's own child, and the launcher then points itself at the new folder. 'table' ({pid: (ppid, args)})
    and 'dtfe_names' ({pid: name}) are for tests; by default ps and pgrep are asked."""
    table = _ps_table() if table is None else table
    if dtfe_names is None:
        try:
            lines = subprocess.run(["pgrep", "-l", _DTFE_BINARY], capture_output=True, text=True,
                                   timeout=5).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            lines = []
        dtfe_names = {int(a): b for a, b in (ln.split(None, 1) for ln in lines if " " in ln) if a.isdigit()}
    me = os.getpid() if me is None else me
    mine, pid = set(), me
    while pid and pid not in mine:                  # me, my parent, ... up to launchd / init
        mine.add(pid)
        pid = table.get(pid, (0, ""))[0]
    users = []
    for pid, (_ppid, args) in sorted(table.items()):
        if pid not in mine and _CACHE_USER_SCRIPTS.search(args):
            users.append((pid, args if len(args) < 160 else args[:157] + "..."))
    seen = {pid for pid, _ in users}
    users += [(pid, name) for pid, name in sorted(dtfe_names.items()) if pid not in mine and pid not in seen]
    return users


_SF_DATALESS = 0x40000000                           # macOS <sys/stat.h>: the file's bytes are in the cloud only


def _st_flags(q):
    """st_flags of 'q' itself (0 where the platform has none or q is gone); a hook for the tests, which cannot set
    SF_DATALESS (only the file provider can)."""
    try:
        return getattr(q.lstat(), 'st_flags', 0)
    except OSError:
        return 0


def _evicted(q):
    """Whether 'q' is evicted to iCloud: a legacy '.<name>.icloud' stub, or (current macOS) dataless in place."""
    return q.name.endswith('.icloud') or bool(_st_flags(q) & _SF_DATALESS)


def _lstat_size(q):
    try:
        return q.lstat().st_size if not q.is_dir() else 0
    except OSError:
        return 0


def _same_device(a, b):
    try:
        return Path(a).stat().st_dev == Path(b).stat().st_dev
    except OSError:
        return True                                 # unknown: count the download, the safe side


def _same_tree(a, b):
    """Whether folder (or file) 'b' holds exactly what 'a' does: the same relative file names, sizes and bytes."""
    import filecmp
    a, b = Path(a), Path(b)
    if a.is_file():
        return b.is_file() and a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    fa = {p.relative_to(a): p for p in a.rglob('*') if p.is_file()}
    fb = {p.relative_to(b): p for p in b.rglob('*') if p.is_file()}
    if set(fa) != set(fb):
        return False
    return all(fa[k].stat().st_size == fb[k].stat().st_size and filecmp.cmp(fa[k], fb[k], shallow=False) for k in fa)


def move_cache(dest, yes=False, src=None, users=None, setting_file=None):
    """'analyze.py cache --move DIR' (and the launcher's setup): move the pipeline cache (config.CACHE_DIR) to DIR
    and record DIR as the cache folder (config.write_cache_setting), so every later script and the launcher use it
    without DTFE_CACHE_DIR. A DRY RUN unless yes: it lists what would move, how much, the room there, and whatever
    would stop it. With yes, refused (exit 1, nothing touched) while a run, a figure script or the launcher may be
    using the cache (cache_users; 'users' replaces the check in tests), when DIR is inside the cache or holds a
    namespace of the same name, or without room. Then EVERY entry is copied into a hidden '.<name>.<pid>.partial'
    beside its final name and verified byte for byte before anything is renamed: a failure leaves the old cache as
    it was and removes the copies. Only then each copy is renamed to its name (atomic: same folder), the setting
    written, and each source removed -- copy, verify, then remove. Returns the exit code."""
    import shutil
    src = Path(config.CACHE_DIR if src is None else src).expanduser()
    dest = Path(os.path.expanduser(str(dest))).resolve()
    setting_file = Path(config.cache_setting_file() if setting_file is None else setting_file)
    print(f"Pipeline cache {src}" + ("" if src.is_dir() else " (absent)") + f" -> {dest}")
    if os.environ.get('DTFE_CACHE_DIR'):
        print("  note: DTFE_CACHE_DIR is set in this shell; it overrides the recorded setting wherever it is set")
    src_r = src.resolve()
    if dest == src_r:
        print("  the cache is already there" + ("; recorded as the setting" if yes else ""))
        if yes:
            config.write_cache_setting(dest, setting_file)
        return 0
    if src_r in dest.parents or dest in src_r.parents:
        print(f"  !! refused: {dest} and {src_r} contain one another")
        return 1
    items = sorted(src.iterdir()) if src.is_dir() else []
    clash = [p.name for p in items if (dest / p.name).exists()]
    sizes = {p: (_dir_size(p) if p.is_dir() else (p.stat().st_size, 1)) for p in items}
    total = sum(b for b, _ in sizes.values())
    for p in items:
        b, n = sizes[p]
        print(f"  {'move ' if yes else 'would move '}{p.name:44s} {_human(b):>9s} {n:5d} file(s)")
    if not items:
        print("  nothing cached yet: only the setting is recorded")
    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = None
    why = []
    # an EVICTED iCloud file is a '.<name>.icloud' stub beside where it was: copying moves the stub, and removing it
    # from iCloud Drive deletes the file in the cloud -- the catalogue would be gone. Current macOS writes
    # no stubs: it evicts in place under the real name with SF_DATALESS in st_flags (_evicted), and copying
    # one downloads it first, onto the source's disk -- so its bytes count twice in the room needed when the
    # destination shares that disk (the copy takes up to 2x).
    evicted = [q for p in items for q in ([p] if p.is_file() else [p, *p.rglob('*')]) if _evicted(q)]
    evicted_bytes = sum(_lstat_size(q) for q in evicted if not q.name.endswith('.icloud'))
    if evicted:
        why.append(f"{len(evicted)} file(s) are evicted to iCloud ({evicted[0].name}, ...): download them first "
                   "(Finder: Download Now on the cache folder), then move")
    need = total * 1.05 + 100e6
    if evicted_bytes and free is not None and _same_device(probe, src):
        need += evicted_bytes
    partial = sorted(p.name for p in dest.glob('.*.partial')) if dest.is_dir() else []
    if partial:
        print(f"  note: {', '.join(partial)} in {dest} is an interrupted move's copy: remove it")
    if clash:
        why.append(f"{dest} already holds {', '.join(clash)} (merge by hand, or pick another folder)")
    if free is not None and free < need:
        why.append(f"{_human(free)} free at {probe}, {_human(total)} to copy"
                   + (f" (+{_human(evicted_bytes)} evicted to iCloud, downloaded onto the same disk first)"
                      if need > total * 1.05 + 100e6 else ""))
    users = cache_users() if users is None else users
    if users:
        why.append("the cache may be in use by: " + "; ".join(f"pid {pid} {what}" for pid, what in users)
                   + " (let them finish, close the launcher, then move)")
    print(f"  {len(items)} entr{'y' if len(items) == 1 else 'ies'}, {_human(total)}; the setting goes to {setting_file}")
    if why:
        print("  !! " + ("refused: " if yes else "it would be refused now: ") + "; ".join(why))
    if not yes:
        print("\nDRY RUN: nothing moved; add --yes to move the cache and record the setting.")
        return 0
    if why:
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    partial = {p: dest / f".{p.name}.{os.getpid()}.partial" for p in items}
    try:
        for p in items:                             # 1: copy and verify everything; the old cache untouched
            (shutil.copytree if p.is_dir() else shutil.copy2)(p, partial[p])
            if not _same_tree(p, partial[p]):
                raise OSError(f"the copy of {p.name} differs from it")
    except OSError as e:
        print(f"  !! the copy failed ({e}): removed it, the cache stays in {src}")
        for q in partial.values():
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
        return 1
    for p in items:                                 # 2: into place (one folder: atomic renames)
        os.replace(partial[p], dest / p.name)
    config.write_cache_setting(dest, setting_file)
    config.CACHE_DIR, config.CACHE_DIR_SOURCE = dest, f"the setting in {setting_file}"
    for p in items:                                 # 3: the sources, each verified copy now in place
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        except OSError as e:
            print(f"  !! {p}: {e} (its copy is in {dest}; remove it by hand)")
    try:
        src.rmdir()                                 # the folder itself when nothing else is left in it
    except OSError:
        pass
    print(f"\nMoved {_human(total)} to {dest}; recorded in {setting_file}.")
    return 0
