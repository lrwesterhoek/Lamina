#!/usr/bin/env bash
# PS-DTFE: a split run (--partition 3 3 3) against one domain, every output grid within a bound of the grid's peak.
#   M1 CPU, both field sets   M2 GPU builds only: the same with '--gpu', no pass falls back to the CPU
# Usage: tests/ps_split_merge_check.sh [--no-build]   (DTFE_TEST_PRECISION=double for the double build)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"
build_target "${1:-}" PS-DTFE o_ps

BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}/ps_split_merge_check"
rm -rf "${TMP}"; mkdir -p "${TMP}"
# <bound of every grid> <bound of the exact deposit's touch count '.tetTouch'>
if precision_double; then LIM=1e-9; LIM_T=1e-9; else LIM=1e-5; LIM_T=3e-3; fi
unset DTFE_TEST_EXIT_AFTER_PARTITIONS DTFE_TEST_DELAY_PARTITION DTFE_CHECKPOINT

echo "============================================================"
echo " PS-DTFE: a split run against the one domain   (${DTFE_TEST_PRECISION})"
echo "============================================================"
SNAP="${TMP}/waves.hdf5"
"${PY}" tests/generate_ps_test_data.py --out "${SNAP}" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
# a scalar per particle (read with --scalar-dataset Potential), so the scalar moments are merged too
"${PY}" - "${SNAP}" <<'PYEOF' || { echo "FAIL scalar dataset"; exit 1; }
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "a") as f:
    x = f["PartType1/Coordinates"][:]
    f["PartType1/Potential"] = (1e3 * np.sin(x[:, 0] / 10.) + x[:, 1]).astype(np.float32)
PYEOF
# check / check_result: tests/lib.sh
# table <a> <b>: the largest |a - b| of every output grid relative to its peak; a grid fails above its bound (LIM_T for '.tetTouch', else LIM)
table() {
    "${PY}" - "$1" "$2" "${LIM}" "${LIM_T}" <<'PYEOF'
import glob, numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
a, b, lim_all, lim_t = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
bad, n = 0, 0
print(f"    {'grid':<16} {'max|d|/peak':>12} {'cells equal':>12}")
for f in sorted(glob.glob(a + ".*")):
    e = f[len(a):]
    if e.endswith(".log") or ".runlog" in e or ".part." in e: continue
    if not os.path.exists(b + e): print(f"    {e}: missing in the other run"); sys.exit(1)
    x = np.fromfile(f, dtype=REAL).astype(np.float64); y = np.fromfile(b + e, dtype=REAL).astype(np.float64)
    if x.size != y.size or not x.size: print(f"    {e}: sizes differ"); sys.exit(1)
    peak = max(np.abs(x).max(), 1e-300)
    r = float(np.abs(x - y).max() / peak)
    lim = lim_t if e == ".tetTouch" else lim_all
    n += 1; bad += not r < lim
    print(f"    {e:<16} {r:12.2e} {np.mean(x == y):12.4f}   (bound {lim:g}){'' if r < lim else '   <-- ABOVE'}")
print(f"    {n} grids, {bad} above their bound")
sys.exit(0 if n and not bad else 1)
PYEOF
}
C0=(--grid 48 --periodic --input-format 105 --mpc-unit 1 --verbose 2)
F1=(--field density velocity gradient dispersion scalar scalarGradient density_a velocity_a --scalar-dataset Potential
    --sampled --subsamples 3)
F2=(--field density velocity dispersion --exact --ps-caustics)
# pair <tag> <set> [extra options]: the set in one domain and split 3 x 3 x 3
pair() {
    local tag="$1" s="$2"; shift 2
    eval "F=(\"\${F$s[@]}\")"
    OMP_NUM_THREADS=4 "${BIN}" "${SNAP}" "${TMP}/${tag}${s}one" "${C0[@]}" --partition 1 1 1 "${F[@]}" "$@" \
        > "${TMP}/${tag}${s}one.log" 2>&1 || { echo "FAIL: ${tag}${s}one (see ${TMP}/${tag}${s}one.log)"; exit 1; }
    OMP_NUM_THREADS=4 "${BIN}" "${SNAP}" "${TMP}/${tag}${s}split" "${C0[@]}" --partition 3 3 3 --max-concurrent 4 "${F[@]}" "$@" \
        > "${TMP}/${tag}${s}split.log" 2>&1 || { echo "FAIL: ${tag}${s}split (see ${TMP}/${tag}${s}split.log)"; exit 1; }
}
echo ">> M1: the CPU deposit"
for s in 1 2; do
    pair c "$s"
    table "${TMP}/c${s}split" "${TMP}/c${s}one"
    check "M1 set $s: split 3 x 3 x 3 against the one domain, every output grid within its bound (${LIM}; .tetTouch ${LIM_T})" $?
done
echo ">> M2: the GPU deposit"
OBJ="$PS_OBJ"
if [ "$(gpu_backend_of "${OBJ}")" != metal ] || [ -f "${OBJ}/.gpu_mode_off" ]; then
    echo "  SKIP M2: ${OBJ} is not a GPU build"
else
    for s in 1 2; do
        pair g "$s" --gpu
        ! grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead\|gave up" "${TMP}/g${s}one.log" "${TMP}/g${s}split.log"
        check "M2 set $s: no GPU pass fell back to the CPU" $?
        table "${TMP}/g${s}split" "${TMP}/g${s}one"
        check "M2 set $s: split 3 x 3 x 3 against the one domain on the GPU, every output grid within its bound (${LIM}; .tetTouch ${LIM_T})" $?
    done
fi
check_result
