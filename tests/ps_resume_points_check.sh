#!/usr/bin/env bash
# Resume of a killed partitioned PS-DTFE run with '--sample-points' (point records in the checkpoint).
# Checks R1-R10 are labelled by their own output lines.
# Usage: tests/ps_resume_points_check.sh [--no-build]   (--no-build is accepted and ignored)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
TMP="${DTFE_TEST_TMP}/ps_resume_points_check"
rm -rf "$TMP"; mkdir -p "$TMP"
TMP="$(cd "$TMP" && pwd -P)"
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
"$PY" - "$TMP" <<'PYEOF' || { echo "FAIL points"; exit 1; }
import sys, numpy as np
tmp = sys.argv[1]
c = (np.arange(128) + 0.5) * 100. / 128
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 50.3)], 1).astype(np.float64).tofile(f"{tmp}/plane.bin")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 31.7)], 1).astype(np.float64).tofile(f"{tmp}/plane_other.bin")
PYEOF
# check / check_result, same <a> <b> [pattern], merged_count, points_list, merged_list: tests/lib.sh
C=(--grid 48 --periodic --input-format 105 --mpc-unit 1 --field density velocity density_a velocity_a dispersion_a
   --partition 4 4 4 --sample-points "$TMP/plane.bin" --ps-per-stream --pts-vel-grad --ps-caustics
   --ps-vertex-mass --verbose 2)

echo ">> R1: the uninterrupted --ps-merge index-order reference (64 partitions, --max-concurrent 4)"
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/ref" "${C[@]}" --max-concurrent 4 --ps-merge index-order > "$TMP/ref.log" 2>&1 \
    || { echo "FAIL: ref (see $TMP/ref.log)"; exit 1; }
grep -q "checkpoints of the merged grids go to '$TMP/ref.runlog.checkpoint'" "$TMP/ref.log" \
    && grep -q "with the '--sample-points' stream records of every merged partition" "$TMP/ref.log" \
    && grep -q "checkpoints: [0-9]* written" "$TMP/ref.log" && ! grep -q "no checkpoints" "$TMP/ref.log" \
    && [ ! -e "$TMP/ref.runlog.checkpoint" ] && [ -s "$TMP/ref.pts_den" ]
check "R1 a --sample-points run checkpoints (grids + point records) and removes the checkpoint after its outputs" $?

echo ">> R2: killed after 20 partitions"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=20 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" --max-concurrent 4 --ps-merge index-order \
    > "$TMP/k.log" 2>&1; rc=$?
CK="$TMP/k.runlog.checkpoint"
n="$(merged_count "$CK" 2>/dev/null || echo 0)"
[ "$rc" = 137 ] && [ ! -e "$TMP/k.pts_den" ] && [ "${n:-0}" -ge 20 ] && [ "${n:-0}" -lt 64 ] \
    && [ "$(points_list "$CK")" = "$(merged_list "$CK")" ]
check "R2 exit 137, no outputs, a manifest of ${n:-?} of 64 partitions with one points.<i> file for each" $?

echo ">> R3: --resume with another sample-point file under the same name refuses"
cp -p "$TMP/plane.bin" "$TMP/plane.keep"
cp "$TMP/plane_other.bin" "$TMP/plane.bin"
"$BIN" "$SNAP" "$TMP/k" "${C[@]}" --max-concurrent 4 --resume > "$TMP/k3.log" 2>&1; rc=$?
cp -p "$TMP/plane.keep" "$TMP/plane.bin"
[ "$rc" != 0 ] && grep -q "'--resume' refused" "$TMP/k3.log" && grep -q "sample-points" "$TMP/k3.log" \
    && [ -s "$CK/checkpoint" ] && [ ! -e "$TMP/k.pts_den" ]
check "R3 refused, naming the sample-point file's key line; the checkpoint is kept" $?

echo ">> R4: --resume"
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" --max-concurrent 4 --resume > "$TMP/k4.log" 2>&1; rc=$?
[ "$rc" = 0 ] && grep -q -- "--resume: $n of 64 partitions are already in the checkpoint" "$TMP/k4.log" \
    && grep -q "stream records were read back" "$TMP/k4.log" && grep -q "partitions $((64 - n))/$((64 - n))" "$TMP/k4.log"
check "R4 the resume read the $n partitions' point records back and computed the other $((64 - n))" $?
same "$TMP/ref" "$TMP/k" 'pts_*'; check "R4 every .pts_* output byte-identical to the uninterrupted run" $?
same "$TMP/ref" "$TMP/k"; check "R4 ... and every grid" $?
[ ! -e "$CK" ]; check "R4 the checkpoint removed on success" $?

echo ">> R5: a damaged points file is refused"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=8 "$BIN" "$SNAP" "$TMP/d" "${C[@]}" --max-concurrent 4 --ps-merge index-order \
    > "$TMP/d.log" 2>&1
DK="$TMP/d.runlog.checkpoint"
first="$(ls -S "$DK" | sed -n 's/^points\.\([0-9]*\)$/\1/p' | head -n 1)"   # the largest: one with records
"$PY" -c "import os,sys; p=sys.argv[1]; os.truncate(p, os.path.getsize(p) - 10)" "$DK/points.${first:-0}"
"$BIN" "$SNAP" "$TMP/d" "${C[@]}" --max-concurrent 4 --resume > "$TMP/d2.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "points.${first:-0}' is not the size its header lists" "$TMP/d2.log" && [ ! -e "$TMP/d.pts_den" ]
check "R5 a cut points.${first:-?} is refused by name (exit $rc)" $?

echo ">> R6: killed while merging in completion order, then resumed"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=25 "$BIN" "$SNAP" "$TMP/u" "${C[@]}" --max-concurrent 4 \
    --ps-merge completion-order > "$TMP/u.log" 2>&1; rc1=$?
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/u" "${C[@]}" --max-concurrent 4 --resume > "$TMP/u2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$rc2" = 0 ] && [ ! -e "$TMP/u.runlog.checkpoint" ] && same "$TMP/ref" "$TMP/u" 'pts_*'
check "R6 killed (137), resumed (0): every .pts_* byte-identical to the reference" $?
"$PY" - "$TMP/ref" "$TMP/u" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
worst = 0.
for e in ("den", "a_den", "a_vel"):
    a, b = (np.fromfile(f"{p}.{e}", dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
    if a.size != b.size or not a.size: sys.exit(1)
    worst = max(worst, float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)))
print(f"    grids: peak-relative difference {worst:.1e}")
sys.exit(0 if worst < 1e-5 else 1)
PYEOF
check "R6 the grids equal the reference to float rounding" $?

echo ">> R7: a fresh run over an old checkpoint replaces its point records"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=20 "$BIN" "$SNAP" "$TMP/f" "${C[@]}" --max-concurrent 4 --ps-merge index-order \
    > "$TMP/f1.log" 2>&1
FK="$TMP/f.runlog.checkpoint"
before="$(points_list "$FK" | wc -w | tr -d ' ')"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=3 "$BIN" "$SNAP" "$TMP/f" "${C[@]}" --max-concurrent 1 --ps-merge index-order \
    > "$TMP/f2.log" 2>&1; rc=$?
[ "$rc" = 137 ] && [ "${before:-0}" -ge 20 ] && grep -q "this run starts over and replaces it" "$TMP/f2.log" \
    && [ "$(merged_count "$FK")" = 3 ] && [ "$(points_list "$FK")" = "$(merged_list "$FK")" ]
check "R7 ${before:-?} old points files, then $(points_list "$FK" | wc -w | tr -d ' ') after the fresh run's first writes (its 3 partitions)" $?

echo ">> R8: the look-ahead path (--max-concurrent 1, DTFE_OVERLAP=1)"
L=(env OMP_NUM_THREADS=4 DTFE_OVERLAP=1)
"${L[@]}" "$BIN" "$SNAP" "$TMP/lref" "${C[@]}" --max-concurrent 1 > "$TMP/lref.log" 2>&1 || { echo "FAIL: lref"; exit 1; }
"${L[@]}" DTFE_TEST_EXIT_AFTER_PARTITIONS=30 "$BIN" "$SNAP" "$TMP/l" "${C[@]}" --max-concurrent 1 > "$TMP/l1.log" 2>&1; rc1=$?
"${L[@]}" "$BIN" "$SNAP" "$TMP/l" "${C[@]}" --max-concurrent 1 --resume > "$TMP/l2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$rc2" = 0 ] && grep -q "look-ahead on" "$TMP/l1.log" && grep -q "look-ahead on" "$TMP/l2.log" \
    && same "$TMP/lref" "$TMP/l" && same "$TMP/ref" "$TMP/l" 'pts_*'
check "R8 killed and resumed under the look-ahead: every output byte-identical (and the points equal R1's)" $?

echo ">> R10: --scratch-dir holds the point records with the checkpoint"
SCR="$TMP/scratch"; mkdir -p "$SCR"
SCK="$(source scripts/config.sh && ps_checkpoint_dir "$TMP/s" "$SCR")"
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "$BIN" "$SNAP" "$TMP/s" "${C[@]}" --max-concurrent 4 --ps-merge index-order \
    --scratch-dir "$SCR" > "$TMP/s1.log" 2>&1; rc1=$?
ok1=1
[ "$rc1" = 137 ] && [ -s "$SCK/checkpoint" ] && [ ! -e "$TMP/s.runlog.checkpoint" ] \
    && [ "$(points_list "$SCK")" = "$(merged_list "$SCK")" ] && ok1=0
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s" "${C[@]}" --max-concurrent 4 --scratch-dir "$SCR" --resume > "$TMP/s2.log" 2>&1; rc2=$?
[ "$ok1" = 0 ] && [ "$rc2" = 0 ] && [ ! -e "$SCK" ] && same "$TMP/ref" "$TMP/s"
check "R10 the points.<i> files beside the checkpoint in the scratch folder ($(basename "$SCK")); resumed byte-identical" $?

echo ">> R9: a killed --gpu run"
OBJ="$PS_OBJ"
if [ ! -d "$OBJ" ] || [ -f "$OBJ/.gpu_mode_off" ] || [ ! -f "$OBJ/.gpu_mode_metal" ]; then
    echo "  SKIP R9: $OBJ is not a Metal GPU build"
else
    G=("${C[@]}" --max-concurrent 1 --ps-merge index-order --gpu)
    "$BIN" "$SNAP" "$TMP/gref" "${G[@]}" > "$TMP/gref.log" 2>&1 || { echo "FAIL: gref (see $TMP/gref.log)"; exit 1; }
    DTFE_TEST_EXIT_AFTER_PARTITIONS=12 "$BIN" "$SNAP" "$TMP/g" "${G[@]}" > "$TMP/g1.log" 2>&1; rc1=$?
    "$BIN" "$SNAP" "$TMP/g" "${G[@]}" --resume > "$TMP/g2.log" 2>&1; rc2=$?
    [ "$rc1" = 137 ] && [ "$rc2" = 0 ] && grep -q -- "--resume: 12 of 64" "$TMP/g2.log" \
        && same "$TMP/gref" "$TMP/g" 'pts_*' && same "$TMP/ref" "$TMP/g" 'pts_*'
    check "R9 killed after 12 GPU partitions and resumed: every .pts_* byte-identical (to the CPU run's too)" $?
fi

check_result
