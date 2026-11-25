#!/usr/bin/env bash
# The PS-DTFE mass grid is exact on the CPU too: CPU shares go into the same fixed-point words as the GPU's.
#   X4  DTFE_MASS_AUDIT=1: every pass gained all its shares (skipped without the audit)   X6  the audit only observes
# Usage: tests/ps_mass_exact_check.sh [--no-build]   (GPU halves SKIP on a CPU build)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    make PS-DTFE $(cat o_ps/.build_mode 2>/dev/null || true) >/dev/null || { echo "FAIL: build"; exit 1; }
fi
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
OBJ="$PS_OBJ"
GPU=0                                                     # any one GPU backend stamp will do (gpu_mass_check.sh)
if [ -d "$OBJ" ] && [ ! -f "$OBJ/.gpu_mode_off" ]; then
    [ -n "$(gpu_backend_of "$OBJ")" ] && GPU=1
fi
TMP="${DTFE_TEST_TMP}/ps_mass_exact_check"
rm -rf "$TMP"; mkdir -p "$TMP"   # (X4 reads every log in the folder: none of an earlier run)
LAT="$TMP/lattice32.hdf5"; PAN="$TMP/pancake64.hdf5"; CW="$TMP/crossed32.hdf5"
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$LAT" --n 32 --amplitude-factor 0.5 --jitter-frac 0 >/dev/null \
    || { echo "FAIL data gen"; exit 1; }
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$CW" --n 32 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$PAN" --n 64 >/dev/null || { echo "FAIL data gen"; exit 1; }

echo "============================================================"
echo " PS-DTFE exact mass grid (${PS_BIN}; GPU halves: $([ $GPU = 1 ] && echo yes || echo 'no GPU build: SKIP'))"
echo "============================================================"
t0=$SECONDS
# run <tag> <snapshot> [env assignments...] -- <PS-DTFE args...>: a non-zero exit fails the suite; a --gpu run
# must have run the GPU deposit (no CPU fallback)
run() {
    local tag="$1" snap="$2"; shift 2
    local envs=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    rm -f "$TMP/$tag".*
    env DTFE_MASS_AUDIT=1 ${envs[@]+"${envs[@]}"} "${PS_BIN}" "$snap" "$TMP/$tag" --periodic --input-format 105 --mpc-unit 1 --verbose 2 \
        --field density_a velocity_a "$@" > "$TMP/$tag.log" 2>&1 || { echo "FAIL: $tag exited non-zero (see $TMP/$tag.log)"; exit 1; }
    case " $* " in
        *" --gpu "*)
            grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$TMP/$tag.log" \
                && { echo "FAIL: $tag fell back to the CPU (see $TMP/$tag.log)"; exit 1; }
            grep -q " GPU deposit (\|run total over .*GPU deposit" "$TMP/$tag.log" \
                || { echo "FAIL: $tag did not run the GPU deposit (see $TMP/$tag.log)"; exit 1; } ;;
    esac
    return 0
}
L=(--grid 2 --subsamples 16)
P=(--grid 4 --subsamples 4 --ps-vertex-mass --partition 1 1 1)
echo ">> X1 lattice (deferral-heavy), CPU"
run lat_cpu_p1 "$LAT" -- "${L[@]}" --partition 1 1 1
run lat_cpu_p2 "$LAT" -- "${L[@]}" --partition 2 2 2
if [ "$GPU" = 1 ]; then
    echo ">> X1 lattice, GPU (and its deferred tets on 1 and 4 CPU threads: X3)"
    run lat_gpu_p1 "$LAT" DTFE_DEPOSIT_THREADS=4 -- "${L[@]}" --partition 1 1 1 --gpu
    run lat_gpu_p1_t1 "$LAT" DTFE_DEPOSIT_THREADS=1 -- "${L[@]}" --partition 1 1 1 --gpu
    run lat_gpu_p2 "$LAT" -- "${L[@]}" --partition 2 2 2 --gpu
    run lat_gpu_rel "$LAT" -- "${L[@]}" --partition 1 1 1 --gpu --ps-halo-release 1.2
    run lat_gpu_p1_na "$LAT" DTFE_DEPOSIT_THREADS=4 DTFE_MASS_AUDIT=0 -- "${L[@]}" --partition 1 1 1 --gpu
fi
echo ">> X2/X3 pancake (heavy accumulation), CPU on 1 thread, 4 (copy mode) and 4 (slab mode)"
run pan_t1 "$PAN" DTFE_DEPOSIT_THREADS=1 -- "${P[@]}"
run pan_t4c "$PAN" DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=0 -- "${P[@]}"
run pan_t4s "$PAN" DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=1 -- "${P[@]}"
run pan_t4c_na "$PAN" DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=0 DTFE_MASS_AUDIT=0 -- "${P[@]}"
echo ">> X7 crossed waves with caustics, copy mode on 1, 4 and 10 threads (shared counts and flags)"
for t in 1 4 10; do
    run "cw_t$t" "$CW" DTFE_DEPOSIT_THREADS=$t DTFE_DEPOSIT_SLABS=0 DTFE_MASS_AUDIT=0 -- \
        --grid 64 --subsamples 2 --ps-caustics --partition 1 1 1
done

"$PY" - "$TMP" "$GPU" "${DTFE_TEST_PRECISION}" <<'PYEOF'
import re, sys
import numpy as np
from pathlib import Path
tmp, gpu, double = Path(sys.argv[1]), sys.argv[2] == "1", sys.argv[3] == "double"
REAL = np.float64 if double else np.float32
fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def den(tag):
    return np.fromfile(tmp / f"{tag}.a_den", dtype=REAL).astype(np.float64)
def mean_ok(tag, what, tol=1e-7):
    d = den(tag)
    dev = float(d.mean() - 1.0)
    check("X1" if tag.startswith("lat") else "X2", d.size > 0 and abs(dev) < tol,
          f"{what}: box mean - 1 = {dev:+.2e} (|.| < {tol:.1e}; {d.size} cells)")
# A partitioned run adds 8 float partition grids per cell (DTFE.cpp's merge), rounding once per add: 1e-7 cannot hold
# (one ulp at 1 is 1.19e-7), so two ulps; the deposit itself stays exact (the one-domain checks, X4's audit).
TOL_SPLIT = 1e-7 if double else 2.5e-7
def same(a, b, what, tag="X3"):
    x, y = (tmp / f"{a}.a_den").read_bytes(), (tmp / f"{b}.a_den").read_bytes()
    if double:   # double sums in thread order: equal to rounding, not bit for bit
        u, v = den(a), den(b)
        rel = float(np.max(np.abs(u - v)) / np.max(np.abs(u))) if u.size == v.size and u.size else 1.0
        check(tag, rel < 1e-12, f"{what}: '.a_den' agrees to {rel:.1e} of the peak (< 1e-12, double build)")
    else:
        check(tag, len(x) > 0 and x == y, f"{what}: '.a_den' byte-identical ({len(x)} bytes)")

mean_ok("lat_cpu_p1", "lattice, CPU, one domain")
mean_ok("lat_cpu_p2", "lattice, CPU, 2^3 partitions", TOL_SPLIT)
if gpu:
    log = (tmp / "lat_gpu_p1.log").read_text()
    m = re.search(r"GPU deposit \(.*?\), (\d+) tetrahedra \((\d+) with a sample on a face", log)
    frac = int(m.group(2)) / max(1, int(m.group(1))) if m else 0.
    check("X1", frac > 0.3, f"the lattice is deferral-heavy on the GPU: {m.group(2) if m else '?'} of "
          f"{m.group(1) if m else '?'} tetrahedra deferred to the CPU ({frac:.0%}, > 30%)")
    mean_ok("lat_gpu_p1", "lattice, GPU, one domain")
    mean_ok("lat_gpu_p2", "lattice, GPU, 2^3 partitions", TOL_SPLIT)
    rel = re.search(r"released (\d+) halo-interior", (tmp / "lat_gpu_rel.log").read_text())
    check("X1", rel is not None and int(rel.group(1)) > 0, f"--ps-halo-release 1.2 released {rel.group(1) if rel else 'no'} tetrahedra")
    mean_ok("lat_gpu_rel", "lattice, GPU, --ps-halo-release 1.2")
    same("lat_gpu_p1_t1", "lat_gpu_p1", "lattice, GPU, the deferred tets on 1 vs 4 CPU threads")
    def deferred_threads(tag):
        m = re.search(r"GPU-deferred tetrahedra on the CPU in [0-9.eE+-]+ s \((\d+) threads?\)", (tmp / f"{tag}.log").read_text())
        return int(m.group(1)) if m else None
    t1, t4 = deferred_threads("lat_gpu_p1_t1"), deferred_threads("lat_gpu_p1")
    s4 = re.search(r"GPU-deferred tetrahedra on the CPU in SLAB mode: (\d+) threads", (tmp / "lat_gpu_p1.log").read_text())
    slab4 = s4 is not None and t4 is not None and int(s4.group(1)) == t4
    check("X3", t1 == 1 and t4 is not None and t4 > 1 and slab4, f"the deferred tets ran on {t1} thread(s) and on {t4} "
          f"({'slab mode' if slab4 else 'NOT slab mode'}); want 1, and more than 1 in slab mode")
    same("lat_gpu_p1_na", "lat_gpu_p1", "lattice, GPU, DTFE_MASS_AUDIT=0 vs 1", "X6")
else:
    print("  SKIP X1/X3 GPU halves: no GPU build")
mean_ok("pan_t1", "pancake, CPU, 1 thread")
slab = re.search(r"CPU deposit in SLAB mode: (\d+) threads", (tmp / "pan_t4s.log").read_text())
check("X3", slab is not None and int(slab.group(1)) > 1, f"the slab run used slab mode ({slab.group(1) if slab else 'NOT'} threads)")
same("pan_t1", "pan_t4c", "pancake, CPU, 1 thread vs 4 (copy mode)")
same("pan_t1", "pan_t4s", "pancake, CPU, 1 thread vs 4 (slab mode)")
same("pan_t4c_na", "pan_t4c", "pancake, CPU, 4 threads (copy mode), DTFE_MASS_AUDIT=0 vs 1", "X6")

# X7: the copy mode on 1/4/10 threads, its counts and flags shared -- the thread count from the log, every file cmp'ed
SHARED = {".a_streams", ".a_hidden_streams", ".caustic", ".causticClass"}   # integer counts, flags, masks
ref = sorted(p for p in tmp.glob("cw_t1.*") if p.suffix != ".log")
exts = {p.name[len("cw_t1"):] for p in ref}
check("X7", SHARED <= exts, f"the 1-thread run wrote {' '.join(sorted(exts))} (needs {' '.join(sorted(SHARED))})")
for t in (4, 10):
    log = (tmp / f"cw_t{t}.log").read_text()
    m = re.search(r"CPU deposit on (\d+) threads? \(copy mode: ([0-9.eE+-]+) MB of private accumulators", log)
    nt, mb = (int(m.group(1)), float(m.group(2))) if m else (None, None)
    check("X7", nt == t and (double or mb == 0.), f"{t}-thread run: the copy mode on {nt} threads with {mb} MB of "
          f"private accumulators per extra thread (want {t}{'' if double else ', 0 MB: nothing copied'})")
    same_b, close, bad = [], [], []
    for p in ref:
        ext = p.name[len("cw_t1"):]
        q = tmp / f"cw_t{t}{ext}"
        x, y = p.read_bytes(), (q.read_bytes() if q.exists() else b"")
        if x == y and x:
            same_b.append(ext)
        elif double and ext not in SHARED and len(x) == len(y) and x:
            u, v = np.frombuffer(x, REAL).astype(np.float64), np.frombuffer(y, REAL).astype(np.float64)
            rel = float(np.max(np.abs(u - v)) / max(np.max(np.abs(u)), 1e-300))
            (close if rel < 1e-12 else bad).append(f"{ext} ({rel:.1e})")
        else:
            bad.append(ext)
    check("X7", not bad and same_b, f"{t} vs 1 thread: byte-identical {' '.join(same_b) or 'nothing'}"
          + (f"; to 1e-12 {' '.join(close)}" if close else "") + (f"; DIFFER {' '.join(bad)}" if bad else ""))

# X4: the audit lines, when the binary has them; a line that does not parse fails X4 on its own
# (glibc spells a NaN or an infinity '-nan' / '-inf').
num = r"([-+]?(?:nan|inf|[0-9.]+(?:[eE][-+]?[0-9]+)?))"
rows, cut = [], []
for f in sorted(tmp.glob("*.log")):
    for l in f.read_text().splitlines():
        if not l.startswith("PS mass audit: partition"): continue
        g, v = re.search(r"gained/shares-1 " + num, l), re.search(r"grid/tets-1 " + num, l)
        if g and v: rows.append((f.stem, float(g.group(1)), float(v.group(1))))
        else: cut.append(f.stem)
if cut:
    check("X4", False, f"every DTFE_MASS_AUDIT line parses: {len(cut)} of {len(rows) + len(cut)} do not "
          f"({' '.join(sorted(set(cut)))}: cut by another thread's output?)")
if rows:
    gs = [(t, g) for t, g, _ in rows]; gt = [(t, v) for t, _, v in rows]
    wg = max(gs, key=lambda p: abs(p[1])); wt = max(gt, key=lambda p: abs(p[1]))
    check("X4", (abs(wg[1]) < 1e-9) if double else all(v == 0. for _, v in gs),
          f"DTFE_MASS_AUDIT: gained/shares-1 over {len(rows)} passes: worst {wg[1]:.1e} ({wg[0]}; "
          f"{'< 1e-9' if double else 'must be exactly 0'})")
    check("X4", abs(wt[1]) < 1e-7, f"DTFE_MASS_AUDIT: grid/tets-1 worst {wt[1]:+.1e} ({wt[0]}; |.| < 1e-7)")
elif not cut:
    print("  SKIP X4: this binary prints no DTFE_MASS_AUDIT lines")
print("RESULT:", "PASS  (the mass grid is exact and thread-independent on the CPU, and after a GPU deposit)"
      if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
rc=$?
echo ">> ps_mass_exact_check: $((SECONDS - t0)) s"
exit $rc
