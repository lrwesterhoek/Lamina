#!/usr/bin/env bash
# PS-DTFE parallel-vs-serial correctness check: runs serial and all-cores, asserts
# outputs agree (a race shows up as a mismatch). .streams must match exactly (integer
# counts are order-independent), float fields only within tolerance.
# Then the CPU deposit's dynamic team (ps_interpolation.cc PSBlockClaim): on all cores, the default run and
# DTFE_DEPOSIT_THREADS=1 (one thread per partition, the serial loop) and =4 (a pinned team of 4) must write
# BYTE-IDENTICAL grids, and the verbose-3 log must say which rule ran: the dynamic team in the single build
# (every accumulator a shared integer word), the static copy mode in the double build (its sums depend on the
# block assignment, so it must keep the static rule). So must a slab team of 4 (DTFE_DEPOSIT_SLABS=1
# DTFE_DEPOSIT_THREADS=4): a static team, counted in the core pool (thread_count.h CorePool::Hold), beside the
# other partitions' pooled loops.
#
# Usage:
#   tests/ps_parallel_check.sh                 # build, run x2, compare
#   tests/ps_parallel_check.sh --no-build
#   N=64 GRID=128 tests/ps_parallel_check.sh   # bigger problem
# Requires the build toolchain plus python3 with numpy and h5py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-32}"; GRID="${GRID:-64}"; BOX="${BOX:-100.0}"; AMP="${AMP:-1.8}"
PARTITION="${PARTITION:-2 2 2}"   # Lagrangian partition grid; parallelism is over these
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
# Input name must NOT share a prefix with the output roots (ps_serial/ps_par),
# or the per-run 'rm -f <root>.*' would delete it.
SNAP="${TMP}/ps_pcheck_input.hdf5"
NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "============================================================"
echo " PS-DTFE parallel-vs-serial check   N=${N}^3  grid=${GRID}^3  cores=${NPROC}  partition=[${PARTITION}]"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" \
    --out "${SNAP}" --n "${N}" --box "${BOX}" --amplitude-factor "${AMP}" >/dev/null

run() {  # $1 = thread count, $2 = output root
    rm -f "$2".*
    local log="$2.log"
    local t0=$SECONDS
    set +e
    OMP_NUM_THREADS="$1" "${BIN}" "${SNAP}" "$2" \
        --grid "${GRID}" --periodic --partition ${PARTITION} --field density velocity density_a \
        --input 105 --MpcUnit 1 --verbose 1 > "$log" 2>&1
    local rc=$?
    set -e
    echo "   [threads=$1] wall time: $(( SECONDS - t0 )) s"
    die_on_failed_run "$rc" "$log" "PS-DTFE (threads=$1)"
}

echo ">> run 1: serial  (OMP_NUM_THREADS=1, --partition ${PARTITION})"
run 1 "${TMP}/ps_serial"
echo ">> run 2: parallel (OMP_NUM_THREADS=${NPROC}, --partition ${PARTITION})"
run "${NPROC}" "${TMP}/ps_par"

echo ">> files produced:"
ls -la "${TMP}"/ps_serial.* "${TMP}"/ps_par.* 2>&1 | sed 's/^/   /' || true

echo ">> comparing serial vs parallel outputs ..."
"${PY}" - "${TMP}/ps_serial" "${TMP}/ps_par" "${GRID}" <<'PY'
import os, sys
import numpy as np

a, b, grid = sys.argv[1], sys.argv[2], int(sys.argv[3])
ncell = grid ** 3
fails = []

def load(path, ncomp):
    raw = np.fromfile(path, dtype=np.uint8); n = ncell * ncomp
    if raw.size == n * 4: return raw.view(np.float32).astype(np.float64)
    if raw.size == n * 8: return raw.view(np.float64)
    sys.exit(f"FAIL: {path} unexpected size {raw.size}")

def compare(ext, ncomp, exact, tol=1e-3):
    fa, fb = a + ext, b + ext
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(f"   {ext:10s} MISSING (serial={os.path.exists(fa)}, parallel={os.path.exists(fb)})")
        fails.append(f"{ext} not produced by both runs"); return
    x, y = load(fa, ncomp), load(fb, ncomp)
    dmax = float(np.abs(x - y).max())
    rel = dmax / (float(np.abs(x).max()) + 1e-30)
    print(f"   {ext:10s} max|Δ|={dmax:.3e}  max relΔ={rel:.3e}"
          + ("  [must be exact]" if exact else f"  [tol {tol:.0e}]"))
    if exact and dmax != 0.0:
        fails.append(f"{ext} not identical ({dmax:.3e}) -- possible race")
    if (not exact) and rel > tol:
        fails.append(f"{ext} relΔ {rel:.2e} exceeds {tol:.0e} -- possible race")

compare(".streams",  1, True)       # integer counts -> order-independent -> exact
# .a_streams is volume-averaged (fractional): partitions sum in a run-dependent order,
# so it agrees only to ~1 float32 ULP. Benign FP non-associativity, not a race (a race
# would drop/double whole streams, rel ~1e-2 >> tol).
compare(".a_streams", 1, False)
compare(".den",      1, False)
compare(".a_den",    1, False)
compare(".vel",      3, False)

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (parallel == serial: integer stream counts identical, "
      "averaged/float fields agree within rounding)")
PY

echo ">> the CPU deposit's team: default vs DTFE_DEPOSIT_THREADS=1 and =4 and a slab team of 4 (--verbose 3, all cores), byte for byte"
run_team() {  # $1 = DTFE_DEPOSIT_THREADS ('' = unset), $2 = output root, $3 = one more VAR=value (optional)
    rm -f "$2".*
    set +e
    env ${1:+DTFE_DEPOSIT_THREADS=$1} ${3:-} OMP_NUM_THREADS="${NPROC}" "${BIN}" "${SNAP}" "$2" \
        --grid "${GRID}" --periodic --partition ${PARTITION} --field density velocity density_a velocity_a \
        --input 105 --MpcUnit 1 --verbose 3 > "$2.log" 2>&1
    local rc=$?
    set -e
    die_on_failed_run "$rc" "$2.log" "PS-DTFE (DTFE_DEPOSIT_THREADS=${1:-unset})"
}
run_team "" "${TMP}/ps_team_def"
run_team 1 "${TMP}/ps_team_t1"
run_team 4 "${TMP}/ps_team_t4"
run_team 4 "${TMP}/ps_team_slab" DTFE_DEPOSIT_SLABS=1
team_fail=0
# (the double build's copy mode sums its private copies in thread order: only its integer stream counts must agree;
# slab mode adds into every cell in the one-thread loop's order, so its grids must agree in both builds)
for f in "${TMP}"/ps_team_t1.*; do
    case "$f" in *.log|*.runlog*) continue ;; esac
    ext="${f#"${TMP}/ps_team_t1"}"
    for other in def t4 slab; do
        if precision_double && [ "${other}" != slab ] && [ "${ext}" != ".streams" ]; then continue; fi
        g="${TMP}/ps_team_${other}${ext}"
        if cmp -s "$f" "$g"; then :; else echo "   FAIL: ${ext} differs (DTFE_DEPOSIT_THREADS=1 vs ${other})"; team_fail=1; fi
    done
done
n_dyn_def=$(grep -c "a dynamic team" "${TMP}/ps_team_def.log" || true)
n_dyn_t4=$(grep -c "a dynamic team: blocks of 256 cells claimed from one counter by the 4 pinned threads" "${TMP}/ps_team_t4.log" || true)
n_static=$(grep -cE "CPU deposit on [0-9]+ threads? \(copy mode: [0-9.e+-]+ MB of private accumulators per extra thread, budget [0-9.e+-]+ GB; the fixed-point grids, counts and flags are shared\)\.$" "${TMP}/ps_team_def.log" || true)
n_slab=$(grep -c "CPU deposit in SLAB mode: 4 threads" "${TMP}/ps_team_slab.log" || true)
echo "   logs: ${n_dyn_def} partition deposit(s) in a dynamic team (default), ${n_dyn_t4} in a pinned team of 4, ${n_static} on the static rule, ${n_slab} in a slab team of 4"
[ "${n_slab}" -gt 0 ] || { echo "   FAIL: DTFE_DEPOSIT_SLABS=1 DTFE_DEPOSIT_THREADS=4 should run a slab team of 4"; team_fail=1; }
if precision_double; then
    # the double build sums private copies in thread order: the static rule, never the dynamic team
    [ "${n_dyn_def}" = 0 ] && [ "${n_dyn_t4}" = 0 ] && [ "${n_static}" -gt 0 ] \
        || { echo "   FAIL: the double build must keep the static rule"; team_fail=1; }
else
    [ "${n_dyn_def}" -gt 0 ] && [ "${n_dyn_t4}" -gt 0 ] \
        || { echo "   FAIL: the single build's CPU deposit should run as a dynamic team"; team_fail=1; }
fi
[ "${team_fail}" = 0 ] || { echo "RESULT: FAIL (the deposit's team)"; exit 1; }
echo "   PASS: the grids agree across the four team rules ($(precision_double && echo "double build: .streams, the slab team every grid" || echo "every grid byte for byte"))"

echo "============================================================"
echo " parallel check PASSED"
echo "============================================================"
