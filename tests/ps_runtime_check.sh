#!/usr/bin/env bash
# The partition loop's run-time features, on small synthetic snapshots:
#   R1  '--ps-merge index-order': two runs at --max-concurrent 4 merge in index order and write byte-identical grids; the
#       partitions start in index order (at most 13 of the 27 parked at once, 3 measured; 21 under a static-steal
#       schedule -- the margin is for a loaded CI runner); the 'merge time:' line counts the 27 merged partitions and
#       gives the seconds of their adds and checkpoints (>= 0)
#   R2  the partition look-ahead (forced on the CPU deposit with DTFE_OVERLAP=1): every output byte-identical
#       to '--no-overlap', and the log says it ran
#   R3  GPU builds only: the look-ahead of a '--gpu' run at --max-concurrent 1 (on by default) against
#       '--no-overlap': every grid byte-identical (.den, .a_den, .streams, ... and the velocity, an exact
#       fixed-point sum like the rest); SKIP on a CPU build
#   R4  crash-safe outputs: a finished run leaves no '<file>.part.<pid>' and no progress sidecar; a stale
#       '.part.<pid>' of a dead pid is removed at the start, one of a live pid is kept, and one whose final file
#       is missing is kept with its mv (it may be the only copy); a run whose output
#       folder is missing writes the whole set to $TMPDIR/dtfe-rescue/<folder>_<name>, byte-identical to a
#       normal run, says so and exits 3
#   R5  the progress sidecar '<output>.runlog.progress' while a run goes: the contract's keys, phase
#       'partitions', done counting up to total; also predicted_gb (> 0: a split and concurrency given in full
#       are predicted too) and peak_gb (> 0)
#   R6  the SUMMARY's 'memory model' line: a run whose concurrency the auto-tuner picks prints its
#       predicted peak against the measured one (macOS: the peak footprint), both positive; so does R5's run,
#       whose plan is given in full
#   R7  the STAGED ordered merge, the default. Partition 0 is held 2 s before its merge
#       (DTFE_TEST_DELAY_PARTITION=0:2), so the others finish first: with NO merge flag the run says it merges in
#       index order, its 'ordered merge:' line counts staged partitions (> 0, none waiting), and every output is
#       byte-identical to R1's '--ps-merge index-order' run; DTFE_MERGE_STAGE_GB=0 (no staging: the plain ordered merge) and
#       a stage cap no partition fits under (no room to stage) wait instead, and a machine with no free memory
#       (DTFE_FREE_GB=0.001) runs one partition at a time behind the memory gate; all byte-identical too;
#       '--ps-merge completion-order' says so, stages nothing and equals R1 to float rounding; the deprecated
#       '--completion-order-merge' and '--ordered-merge' together are refused, and so is '--ps-merge ordered'; a default run killed after 10 merges (DTFE_TEST_EXIT_AFTER_PARTITIONS=10) while
#       partitions are staged has checkpointed exactly partitions 0-9 (the merge order is the index order) and
#       resumes to R1's grids byte for byte; with the concurrency left to the auto-tuner (the memory gate guards
#       every partition start) a partition is always staged, and the grids equal DTFE_MERGE_STAGE_GB=0's byte for byte.
#       THE SPILL: with no RAM room to park (DTFE_TEST_PARK_ROOM_GB=0) a parked partition is spilled to disk (--scratch-dir, which is
#       empty afterwards: the files are unlinked) instead of waiting -- the 'ordered merge:' line counts the spills,
#       none waited -- and DTFE_MERGE_SPILL=1 spills every parked one (the system temp folder, with DTFE_SPILL_MIN_GB=2:
#       a CI runner has less than its 16 GB margin free; and under the memory gate): every output byte-identical to a
#       '--max-concurrent 1' run and to R1; a spilling run killed after 10 merges checkpointed partitions 0-9 and
#       resumes byte-identical; DTFE_MERGE_SPILL=0 brings back the wait. The 'merge time:' line: partitions parked
#       behind the held-back partition 0 (> 0 s parked), none parked but threads waiting with DTFE_MERGE_STAGE_GB=0
#       (> 0 s holding a thread), the resumed run's count (the 17 it merged), the completion-order run's lock wait
#   R8  the progress sidecar of a run that does not finish: one that stops on an error (a missing input, found by the
#       reader after the sidecar is armed; also the standard binary when it is built) exits 1 and leaves none, as the
#       launcher then reads a failed attempt from its run log instead of a death; one killed after 2 partitions
#       (DTFE_TEST_EXIT_AFTER_PARTITIONS=2, exit 137) keeps it; one whose outputs could not be put in place (a folder
#       where the '.den' goes, exit 3) keeps it with phase 'rescued'
# Usage: tests/ps_runtime_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
TMP="${DTFE_TEST_TMP}/ps_runtime_check"
rm -rf "$TMP"; mkdir -p "$TMP"
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# same <a> <b>: tests/lib.sh
C=(--grid 48 --periodic --input-format 105 --mpc-unit 1 --verbose 2 --field density velocity density_a)

echo ">> R1: --ps-merge index-order, --partition 3 3 3 --max-concurrent 4, twice"
H="$("${BIN}" --help 2>&1)"
grep -q -- "--ps-merge" <<< "$H" && grep -q "index-order" <<< "$H"; check "R1 '--ps-merge index-order' is in the help" $?
for r in o1 o2; do OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/$r" "${C[@]}" --partition 3 3 3 --max-concurrent 4 --ps-merge index-order \
    > "$TMP/$r.log" 2>&1 || { echo "FAIL: $r (see $TMP/$r.log)"; exit 1; }; done
grep -q "partition merge order  : partition index" "$TMP/o1.log"; check "R1 the run says it merges in index order" $?
# the partitions START in index order (the loop's monotonic schedule), so a finished one waits for the few in flight
# before it: under libomp's plain schedule(dynamic, 1), a static steal, 21 of these 27 were parked at once
amax="$(sed -n 's/.*ordered merge: .*(at most \([0-9]*\) at once.*/\1/p' "$TMP/o1.log")"
[ -n "$amax" ] && [ "$amax" -le 13 ]; check "R1 the partitions start in index order: at most ${amax:-?} parked at once (<= 13)" $?
same "$TMP/o1" "$TMP/o2"; check "R1 two --ps-merge index-order runs at --max-concurrent 4: byte-identical grids" $?
# the 'merge time:' line: the partitions this run merged, the seconds of their adds and of their checkpoints
mtime() { sed -n 's/.*merge time: \([0-9]*\) partition(s) added to the full grids in \([0-9.]*\) s, their checkpoints \([0-9.]*\) s.*/\1 \2 \3/p' "$1"; }
read -r mtn mta mtc <<< "$(mtime "$TMP/o1.log")"
[ "${mtn:-0}" = 27 ] && awk -v a="${mta:--1}" -v c="${mtc:--1}" 'BEGIN { exit !(a >= 0 && c >= 0) }'
check "R1 the 'merge time:' line: ${mtn:-?} of 27 partitions added in ${mta:-?} s (checkpoints ${mtc:-?} s)" $?

echo ">> R7: the staged ordered merge (the default), partition 0 held back 2 s"
grep -q "completion-order" <<< "$H"; check "R7 '--ps-merge completion-order' is in the help" $?
R7=(env OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP")
P3=(--partition 3 3 3 --max-concurrent 4)
"${R7[@]}" "$TMP/st" "${C[@]}" "${P3[@]}" > "$TMP/st.log" 2>&1 || { echo "FAIL: st (see $TMP/st.log)"; exit 1; }
nst="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/st.log")"
nwt="$(sed -n 's/.* \([0-9]*\) waited for their turn.*/\1/p' "$TMP/st.log")"
grep -q "partition merge order  : partition index" "$TMP/st.log" && grep -q "PS-DTFE: partitions merge in index order" "$TMP/st.log" \
    && [ "${nst:-0}" -gt 0 ] && [ "${nwt:-1}" = 0 ]
check "R7 no merge flag: index order, ${nst:-?} partition(s) staged, ${nwt:-?} waited" $?
same "$TMP/o1" "$TMP/st"; check "R7 ... every output byte-identical to the --ps-merge index-order run (R1)" $?
DTFE_MERGE_STAGE_GB=0 "${R7[@]}" "$TMP/st0" "${C[@]}" "${P3[@]}" > "$TMP/st0.log" 2>&1 || { echo "FAIL: st0 (see $TMP/st0.log)"; exit 1; }
# no room under the stage cap and no spill (with the spill on, such a partition goes to disk; below)
DTFE_MERGE_SPILL=0 DTFE_MERGE_STAGE_GB=0.000001 "${R7[@]}" "$TMP/stf" "${C[@]}" "${P3[@]}" > "$TMP/stf.log" 2>&1 || { echo "FAIL: stf (see $TMP/stf.log)"; exit 1; }
# no free memory: the plan given in full is predicted, so the memory gate runs one partition at a time
DTFE_MERGE_SPILL=0 DTFE_FREE_GB=0.001 "${R7[@]}" "$TMP/stg" "${C[@]}" "${P3[@]}" > "$TMP/stg.log" 2>&1 || { echo "FAIL: stg (see $TMP/stg.log)"; exit 1; }
grep -q "ordered merge: 0 partition(s) finished before their turn" "$TMP/st0.log" && ! grep -q " 0 waited for their turn" "$TMP/st0.log" \
    && grep -q "ordered merge: 0 partition(s) finished before their turn" "$TMP/stf.log" && grep -q "(no room to stage)" "$TMP/stf.log" \
    && grep -q "running one partition at a time" "$TMP/stg.log" \
    && same "$TMP/o1" "$TMP/st0" && same "$TMP/o1" "$TMP/stf" && same "$TMP/o1" "$TMP/stg"
check "R7 DTFE_MERGE_STAGE_GB=0, a stage cap nothing fits under, no free memory (DTFE_MERGE_SPILL=0): wait their turn / one at a time, byte-identical" $?
# ... the 'merge time:' line's waits behind the held-back partition 0: parked (st), or holding a thread (st0: nothing parks)
mwait() { sed -n 's/.*merge time: .*waiting for their turn: \([0-9.]*\) s parked, \([0-9.]*\) s holding a thread.*/\1 \2/p' "$1"; }
read -r stp _ <<< "$(mwait "$TMP/st.log")"; read -r s0p s0h <<< "$(mwait "$TMP/st0.log")"
awk -v a="${stp:-0}" -v b="${s0p:--1}" -v h="${s0h:-0}" 'BEGIN { exit !(a > 0 && b == 0 && h > 0) }'
check "R7 the 'merge time:' line: ${stp:-?} s parked behind partition 0; DTFE_MERGE_STAGE_GB=0: ${s0p:-?} s parked, ${s0h:-?} s holding a thread" $?
# THE SPILL: the reference is the same split at --max-concurrent 1 (no parking at all), on ONE thread:
# four partitions on four threads deposit each on one, and the double build's CPU sums are equal between thread
# counts only to rounding (README: run-to-run identical grids)
OMP_NUM_THREADS=1 "$BIN" "$SNAP" "$TMP/m1" "${C[@]}" --partition 3 3 3 --max-concurrent 1 --no-overlap > "$TMP/m1.log" 2>&1 \
    || { echo "FAIL: m1 (see $TMP/m1.log)"; exit 1; }
nspill() { sed -n 's/.*ordered merge: .* \([0-9]*\) spilled to disk.*/\1/p' "$1"; }
SPD="$TMP/spill"; mkdir -p "$SPD"
# no RAM room to park (DTFE_TEST_PARK_ROOM_GB=0) while the free memory admits the partitions: a partition that finishes
# before its turn goes to disk. (Not 'no free memory': the memory gate then runs one partition at a time in index
# order, so none finishes before its turn -- the gate let later partitions past it before 2026-11-07, out of order.)
DTFE_FREE_GB=1000 DTFE_TEST_PARK_ROOM_GB=0 "${R7[@]}" "$TMP/sp" "${C[@]}" "${P3[@]}" --scratch-dir "$SPD" > "$TMP/sp.log" 2>&1 || { echo "FAIL: sp (see $TMP/sp.log)"; exit 1; }
nsp="$(nspill "$TMP/sp.log")"
grep -q "when the memory cannot hold it, spilled to disk in '$SPD'" "$TMP/sp.log" && [ "${nsp:-0}" -gt 0 ] \
    && grep -q " 0 waited for their turn" "$TMP/sp.log" && [ -z "$(ls -A "$SPD")" ] && same "$TMP/m1" "$TMP/sp" && same "$TMP/o1" "$TMP/sp"
check "R7 no RAM room to park: ${nsp:-?} partition(s) spilled to --scratch-dir (left empty), none waited; byte-identical to --max-concurrent 1 and R1" $?
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 "${R7[@]}" "$TMP/sf" "${C[@]}" "${P3[@]}" > "$TMP/sf.log" 2>&1 || { echo "FAIL: sf (see $TMP/sf.log)"; exit 1; }
nsf="$(nspill "$TMP/sf.log")"
nst2="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/sf.log")"
grep -q "every one with sub-grids is spilled to disk" "$TMP/sf.log" && [ "${nsf:-0}" -gt 0 ] && [ "$nsf" = "$nst2" ] \
    && same "$TMP/m1" "$TMP/sf"
check "R7 DTFE_MERGE_SPILL=1 (the system temp folder): all ${nsf:-?} parked partition(s) spilled, byte-identical to --max-concurrent 1" $?
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "${R7[@]}" "$TMP/sk3" "${C[@]}" "${P3[@]}" > "$TMP/sk3.log" 2>&1; rc1=$?
bits="$(grep -a -m1 '^merged ' "$TMP/sk3.runlog.checkpoint/checkpoint" 2>/dev/null | awk '{print $2}')"
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/sk3" "${C[@]}" "${P3[@]}" --resume > "$TMP/sk4.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$bits" = "111111111100000000000000000" ] && [ "$rc2" = 0 ] && same "$TMP/m1" "$TMP/sk3"
check "R7 spilling run killed after 10 merges (exit $rc1): partitions 0-9 checkpointed (${bits:-none}), resumed byte-identical (exit $rc2)" $?
"${R7[@]}" "$TMP/co" "${C[@]}" "${P3[@]}" --ps-merge completion-order > "$TMP/co.log" 2>&1 || { echo "FAIL: co (see $TMP/co.log)"; exit 1; }
grep -q "partition merge order  : completion order" "$TMP/co.log" && ! grep -q "ordered merge:" "$TMP/co.log" \
    && "$PY" - "$TMP/o1" "$TMP/co" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
for e in ("den", "a_den", "vel"):
    a, b = (np.fromfile(f"{p}.{e}", dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
    if a.size != b.size or not a.size or np.abs(a - b).max() > 1e-5 * max(np.abs(a).max(), 1e-300): sys.exit(1)
PYEOF
check "R7 --ps-merge completion-order: completion order, nothing staged, the grids equal R1's to float rounding" $?
"$BIN" "$SNAP" "$TMP/bad" "${C[@]}" "${P3[@]}" --ordered-merge --completion-order-merge > "$TMP/bad.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "contradict each other" "$TMP/bad.log" && [ ! -e "$TMP/bad.den" ]
check "R7 '--ordered-merge --completion-order-merge' (the deprecated pair) refused (exit $rc)" $?
"$BIN" "$SNAP" "$TMP/badv" "${C[@]}" "${P3[@]}" --ps-merge ordered > "$TMP/badv.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "Unknown value 'ordered' for '--ps-merge'" "$TMP/badv.log" && [ ! -e "$TMP/badv.den" ]
check "R7 '--ps-merge ordered' (no such mode) refused, naming the value (exit $rc)" $?
DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "${R7[@]}" "$TMP/sk" "${C[@]}" "${P3[@]}" > "$TMP/sk.log" 2>&1; rc1=$?
bits="$(grep -a -m1 '^merged ' "$TMP/sk.runlog.checkpoint/checkpoint" 2>/dev/null | awk '{print $2}')"
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/sk" "${C[@]}" "${P3[@]}" --resume > "$TMP/sk2.log" 2>&1; rc2=$?
[ "$rc1" = 137 ] && [ "$bits" = "111111111100000000000000000" ] && [ "$rc2" = 0 ] && same "$TMP/o1" "$TMP/sk"
check "R7 killed after 10 merges with partitions staged (exit $rc1): partitions 0-9 checkpointed (${bits:-none}), resumed byte-identical (exit $rc2)" $?
mtr="$(mtime "$TMP/sk2.log" | awk '{print $1}')"
[ "${mtr:-0}" = 17 ] && grep -q "merge time: 27 partition(s) .*; waiting for the merge lock: [0-9.]* s" "$TMP/co.log"
check "R7 'merge time:' counts the ${mtr:-?} partitions the resumed run merged (17), and gives the completion-order run's lock wait" $?
# the gate-guarded case: '--partition 3 3 3' with the concurrency left to the tuner (psPartitionBytes > 0)
GA=(env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP")
"${GA[@]}" "$TMP/sg" "${C[@]}" --partition 3 3 3 > "$TMP/sg.log" 2>&1 || { echo "FAIL: sg (see $TMP/sg.log)"; exit 1; }
DTFE_MERGE_STAGE_GB=0 "${GA[@]}" "$TMP/sg0" "${C[@]}" --partition 3 3 3 > "$TMP/sg0.log" 2>&1 || { echo "FAIL: sg0 (see $TMP/sg0.log)"; exit 1; }
nsg="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/sg.log")"
grep -q "the memory gate admits the next partition" "$TMP/sg.log" && [ "${nsg:-0}" -gt 0 ] && same "$TMP/sg0" "$TMP/sg"
check "R7 auto-tuned concurrency (the memory gate guards the starts): ${nsg:-?} staged, byte-identical to DTFE_MERGE_STAGE_GB=0" $?
DTFE_MERGE_SPILL=1 DTFE_SPILL_MIN_GB=2 "${GA[@]}" "$TMP/sgs" "${C[@]}" --partition 3 3 3 > "$TMP/sgs.log" 2>&1 || { echo "FAIL: sgs (see $TMP/sgs.log)"; exit 1; }
nsgs="$(nspill "$TMP/sgs.log")"
grep -q "the memory gate admits the next partition" "$TMP/sgs.log" && [ "${nsgs:-0}" -gt 0 ] && same "$TMP/sg0" "$TMP/sgs"
check "R7 ... and spilling under the memory gate (DTFE_MERGE_SPILL=1): ${nsgs:-?} spilled, byte-identical" $?
# the budget room: the free memory plentiful (DTFE_FREE_GB=1000), but no room in the budget beside the
# plan's predicted peak -- every parked partition must spill (on the free-memory rule alone TNG100 z=0 parks 74 GB)
DTFE_TEST_PARK_ROOM_GB=0 DTFE_SPILL_MIN_GB=2 "${GA[@]}" "$TMP/sgr" "${C[@]}" --partition 3 3 3 > "$TMP/sgr.log" 2>&1 || { echo "FAIL: sgr (see $TMP/sgr.log)"; exit 1; }
nsgr="$(nspill "$TMP/sgr.log")"
nstr="$(sed -n 's/.*ordered merge: \([0-9]*\) partition(s) finished before their turn.*/\1/p' "$TMP/sgr.log")"
grep -q "the memory budget leaves beside the plan" "$TMP/sgr.log" && [ "${nsgr:-0}" -gt 0 ] && [ "$nsgr" = "$nstr" ] \
    && same "$TMP/sg0" "$TMP/sgr"
check "R7 ... and no room in the memory budget (DTFE_TEST_PARK_ROOM_GB=0): all ${nsgr:-?} parked partition(s) spilled, byte-identical" $?

echo ">> R2: the look-ahead on the CPU deposit (DTFE_OVERLAP=1) against --no-overlap"
"$BIN" "$SNAP" "$TMP/n1" "${C[@]}" --partition 3 3 3 --max-concurrent 1 --no-overlap > "$TMP/n1.log" 2>&1 \
    || { echo "FAIL: n1 (see $TMP/n1.log)"; exit 1; }
DTFE_OVERLAP=1 "$BIN" "$SNAP" "$TMP/l1" "${C[@]}" --partition 3 3 3 --max-concurrent 1 > "$TMP/l1.log" 2>&1 \
    || { echo "FAIL: l1 (see $TMP/l1.log)"; exit 1; }
grep -q "look-ahead on" "$TMP/l1.log" && ! grep -q "look-ahead on" "$TMP/n1.log"; check "R2 the look-ahead ran (and not with --no-overlap)" $?
same "$TMP/n1" "$TMP/l1"; check "R2 look-ahead vs --no-overlap: every output byte-identical" $?
# the production shape: point evaluation and the tessellation cache (built and saved by the look-ahead thread,
# then a second run whose look-ahead only loads)
"$PY" -c "import numpy as np; np.random.default_rng(3).uniform(0, 100, (2000, 3)).astype(np.float64).tofile('$TMP/pts.bin')"
mkdir -p "$TMP/cache_n" "$TMP/cache_l"
S2=(--sample-points "$TMP/pts.bin" --pts-vel-grad --partition 3 3 3 --max-concurrent 1)
"$BIN" "$SNAP" "$TMP/n2" "${C[@]}" "${S2[@]}" --tessellation-cache "$TMP/cache_n" --no-overlap > "$TMP/n2.log" 2>&1 \
    || { echo "FAIL: n2 (see $TMP/n2.log)"; exit 1; }
for r in l2 l3; do DTFE_OVERLAP=1 "$BIN" "$SNAP" "$TMP/$r" "${C[@]}" "${S2[@]}" --tessellation-cache "$TMP/cache_l" \
    > "$TMP/$r.log" 2>&1 || { echo "FAIL: $r (see $TMP/$r.log)"; exit 1; }; done
[ -s "$TMP/l2.pts_den" ] && grep -q "look-ahead on" "$TMP/l3.log" && grep -q "cached" "$TMP/l3.log"
check "R2 with --sample-points and --tessellation-cache: the look-ahead ran, the second run from the cache" $?
same "$TMP/n2" "$TMP/l2" && same "$TMP/n2" "$TMP/l3"
check "R2 ... grids and '.pts_*' byte-identical (look-ahead building, look-ahead loading, no look-ahead)" $?

echo ">> R3: the look-ahead of a --gpu run"
OBJ=o_ps; precision_double && OBJ=o_ps_d
if [ ! -d "$OBJ" ] || [ -f "$OBJ/.gpu_mode_off" ] || [ ! -f "$OBJ/.gpu_mode_metal" ]; then
    echo "  SKIP R3: $OBJ is not a Metal GPU build"
else
    G=("${C[@]}" --partition 3 3 3 --max-concurrent 1 --gpu)
    "$BIN" "$SNAP" "$TMP/gn" "${G[@]}" --no-overlap > "$TMP/gn.log" 2>&1 || { echo "FAIL: gn (see $TMP/gn.log)"; exit 1; }
    "$BIN" "$SNAP" "$TMP/gl" "${G[@]}" > "$TMP/gl.log" 2>&1 || { echo "FAIL: gl (see $TMP/gl.log)"; exit 1; }
    grep -q "look-ahead on: .* on the GPU" "$TMP/gl.log" \
        && ! grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "$TMP/gl.log" "$TMP/gn.log"
    check "R3 a --gpu run at --max-concurrent 1 looks ahead by default (and neither run fell back to the CPU)" $?
    ok=0
    # the velocity too: every PS grid of a GPU deposit is an exact fixed-point sum
    for e in den a_den streams a_streams hidden_streams a_hidden_streams vel; do
        cmp -s "$TMP/gn.$e" "$TMP/gl.$e" || { echo "    differs: .$e"; ok=1; }
    done
    check "R3 GPU look-ahead vs --no-overlap: every grid (the velocity included) byte-identical" $ok
fi

echo ">> R4: crash-safe outputs"
OUT="$TMP/safe"; mkdir -p "$OUT"
sleep 0 & DEAD=$!; wait "$DEAD"   # a pid that no longer runs
echo stale > "$OUT/r.den.part.$DEAD"
echo old > "$OUT/r.den"            # the stale part's final file exists: the part is a leftover, removed
echo only > "$OUT/r.only.part.$DEAD"   # no final file: maybe the only copy of an output, kept
echo alive > "$OUT/r.den.part.$$"
"$BIN" "$SNAP" "$OUT/r" "${C[@]}" --partition 2 2 2 --max-concurrent 1 > "$OUT/r.log" 2>&1; rc=$?
[ "$rc" = 0 ] && [ -s "$OUT/r.den" ] && [ -s "$OUT/r.a_den" ]; check "R4 a normal run exits 0 with its outputs in place" $?
[ ! -e "$OUT/r.den.part.$DEAD" ] && grep -q "stale partial output" "$OUT/r.log"; check "R4 the dead pid's '.part' file was removed, and said so" $?
[ -e "$OUT/r.den.part.$$" ]; check "R4 a live pid's '.part' file was kept" $?
[ -e "$OUT/r.only.part.$DEAD" ] && grep -q "Kept the partial output '$OUT/r.only.part.$DEAD'" "$OUT/r.log" \
    && grep -q "mv '$OUT/r.only.part.$DEAD' '$OUT/r.only'" "$OUT/r.log"
check "R4 a dead pid's '.part' file whose final file is missing was kept, with the mv that puts it in place" $?
rm -f "$OUT/r.den.part.$$" "$OUT/r.only.part.$DEAD"
n=$(find "$OUT" -name 'r.*.part.*' | wc -l | tr -d ' ')
[ "$n" = 0 ] && [ ! -e "$OUT/r.runlog.progress" ]; check "R4 no '.part' file and no progress sidecar left after the run" $?
# a MISSING output folder (an unmounted disk is a missing path; a read-only folder would not stop root in CI)
RO="$TMP/ro/gone"; mkdir -p "$TMP/ro"; RTMP="$TMP/rescuetmp"; mkdir -p "$RTMP"
TMPDIR="$RTMP" DTFE_OUTPUT_WAIT=0 "$BIN" "$SNAP" "$RO/r" "${C[@]}" --partition 2 2 2 --max-concurrent 1 > "$TMP/ro.log" 2>&1; rc=$?
RESCUE="$RTMP/dtfe-rescue/gone_r"
[ "$rc" = 3 ] && grep -q "OUTPUT RESCUE" "$TMP/ro.log" && grep -q "mv '$RESCUE'" "$TMP/ro.log"
check "R4 a missing output folder: exit 3 and the rescue folder named with the mv command" $?
ok=0; nf=0
for f in "$OUT"/r.*; do
    case "$f" in *.log|*.runlog*|*.part.*) continue ;; esac
    nf=$((nf+1)); cmp -s "$f" "$RESCUE/$(basename "$f")" || { echo "    rescue differs/missing: $(basename "$f")"; ok=1; }
done
[ "$nf" -gt 0 ] || ok=1
check "R4 the rescued set is complete and byte-identical to the normal run ($nf files)" $ok

echo ">> R5: the progress sidecar while a run goes"
P5="$TMP/prog"
"$BIN" "$SNAP" "$P5" --grid 64 --periodic --input-format 105 --mpc-unit 1 --verbose 1 --field density gradient \
    --partition 4 4 4 --max-concurrent 1 > "$P5.log" 2>&1 &
BPID=$!
"$PY" - "$P5.runlog.progress" "$BPID" > "$P5.snaps" <<'PYEOF'
import json, os, sys, time
path, pid = sys.argv[1], int(sys.argv[2])
seen = []
while True:
    try:
        with open(path) as f: seen.append(f.read())
    except OSError:
        pass
    try: os.kill(pid, 0)
    except OSError: break
    time.sleep(0.02)
for s in seen:
    if s.strip(): print(s.strip())
PYEOF
wait "$BPID"; rc=$?
[ "$rc" = 0 ]; check "R5 the run finished" $?
"$PY" - "$P5.snaps" "$BPID" "$P5.runlog.progress" <<'PYEOF'
import json, os, sys
snaps = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
keys = {"pid", "phase", "done", "total", "elapsed_s", "eta_s", "rss_gb", "gpu_retries", "predicted_gb", "peak_gb",
        "updated"}
parts = [s for s in snaps if s.get("phase") == "partitions"]
ok = (bool(snaps) and all(keys <= set(s) for s in snaps) and all(s["pid"] == int(sys.argv[2]) for s in snaps)
      and len(parts) >= 2 and all(p["total"] == 64 for p in parts)
      and [p["done"] for p in parts] == sorted(p["done"] for p in parts) and max(p["done"] for p in parts) >= 32
      and all(p["rss_gb"] > 0 and p["peak_gb"] > 0 and p["predicted_gb"] > 0 for p in parts)
      and not os.path.exists(sys.argv[3]))
print(f"    {len(snaps)} snapshots read, phases {sorted({s.get('phase') for s in snaps})}, "
      f"done up to {max([p['done'] for p in parts] or [0])}/64")
sys.exit(0 if ok else 1)
PYEOF
check "R5 the sidecar carries the contract's keys, counts the partitions up and is gone at the end" $?

echo ">> R6: the SUMMARY's predicted-vs-measured memory line"
"$BIN" "$SNAP" "$TMP/mm" --grid 32 --periodic --input-format 105 --mpc-unit 1 --verbose 1 --field density \
    --partition 2 2 2 > "$TMP/mm.log" 2>&1 || { echo "FAIL: mm (see $TMP/mm.log)"; exit 1; }
"$PY" - "$TMP/mm.log" "$P5.log" <<'PYEOF'
import re, sys
ok = True
for path in sys.argv[1:]:
    m = re.search(r"memory model\s*:\s*([\d.]+) GB predicted \(auto-tune\) vs ([\d.]+) GB measured", open(path).read())
    print(f"    {m.group(0) if m else 'no memory model line'}")
    ok = ok and bool(m) and float(m.group(1)) > 0 and float(m.group(2)) > 0
sys.exit(0 if ok else 1)
PYEOF
check "R6 an auto-tuned run and one whose plan is given in full print their predicted and measured peak" $?

echo ">> R8: the progress sidecar of a run that stops on an error, is killed, or cannot put its outputs in place"
# an error ends in exit(), whose handler takes the sidecar; the missing input is found by the reader, AFTER the
# sidecar is armed (an option error would stop the run before it and pass this check without testing anything)
E8=("$BIN")
[ -x "${DTFE_BIN}" ] && E8+=("$(cd "$(dirname "${DTFE_BIN}")" && pwd)/$(basename "${DTFE_BIN}")")
for b in "${E8[@]}"; do
    n="$(basename "$b")"
    "$b" "$TMP/absent.hdf5" "$TMP/e8_$n" "${C[@]}" > "$TMP/e8_$n.log" 2>&1; rc=$?
    [ "$rc" = 1 ] && grep -q "READING INPUT" "$TMP/e8_$n.log" && grep -q "could not open the input" "$TMP/e8_$n.log" \
        && grep -q "ended unsucessfully" "$TMP/e8_$n.log" && [ ! -e "$TMP/e8_$n.runlog.progress" ] \
        && ! ls "$TMP/e8_$n.runlog.progress.tmp."* >/dev/null 2>&1
    check "R8 $n: a missing input stops the run in the reader (exit $rc) and leaves no progress sidecar" $?
done
# a kill runs no exit handler: the sidecar stays, its pid gone (the launcher's 'died')
DTFE_TEST_EXIT_AFTER_PARTITIONS=2 "$BIN" "$SNAP" "$TMP/k8" "${C[@]}" --partition 2 2 2 --max-concurrent 1 \
    > "$TMP/k8.log" 2>&1; rc=$?
[ "$rc" = 137 ] && [ -s "$TMP/k8.runlog.progress" ]
check "R8 a run killed after 2 partitions (exit $rc, 137 = SIGKILL) keeps its progress sidecar" $?
# exit 3 in a writable folder (a missing one, as in R4, never gets a sidecar): the 'rescued' sidecar is kept
mkdir -p "$TMP/c8.den"; : > "$TMP/c8.den/blocker"   # a folder where the '.den' output goes: its rename fails
"$BIN" "$SNAP" "$TMP/c8" "${C[@]}" --partition 2 2 2 --max-concurrent 1 > "$TMP/c8.log" 2>&1; rc=$?
phase="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["phase"])' "$TMP/c8.runlog.progress" 2>/dev/null)"
[ "$rc" = 3 ] && grep -q "could not be put in place" "$TMP/c8.log" && [ "$phase" = rescued ]
check "R8 outputs that could not be put in place (exit $rc) keep the progress sidecar, phase '${phase:-none}'" $?
rm -rf "$TMP/c8.den" "$TMP"/c8.*.part.* "$TMP/c8.runlog.checkpoint" "$TMP/k8.runlog.checkpoint"

echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ] || exit 1
exit 0
