#!/usr/bin/env bash
# '--ps-merge fixed-point' (opt-in; src/ps_fixed_merge.h): every partition is added AS IT FINISHES into exact integer
# full grids, on small synthetic snapshots. Two merges answer to the option (fix 3 S6):
#   THE WORD MERGE, the single 3-D build's (User_options::psWordMerge): the option switches the run-wide fixed-point
#     units on, every pass hands its exact words over unconverted (64-bit words with an exact carry, int stream
#     counts), the merge adds them as integers and ONE close-out converts and normalizes, a one-domain run's too;
#   THE MERGE OF BEFORE (the 9-B merge): each partition's grids, rounded to Real by its pass, become int64 words +
#     int8 carries in units of the merge's own -- the 2-D build's, and the single build's under DTFE_RUN_UNITS_LEGACY=1.
# The suite runs the word merge's cases. With DTFE_RUN_UNITS_LEGACY=1 exported by the caller it runs every case on the
# merge of before instead, with the expectations it had before the word merge (the 9-B lines below). The double build
# refuses the option (X5 only).
#   X1  the CPU deposit, two field sets (the moments with a scalar, sampled; and volume-weighted + exact deposit +
#       caustics, which fill mass_weight, disp_weight, disp_velocity, tet_touch and the OR-merged caustic mask):
#       --max-concurrent 1, 2 and 4, and 4 with partition 0 held back 2 s (DTFE_TEST_DELAY_PARTITION=0:2: another
#       completion order) -- every output byte-identical among them; the run log says the merge mode, prints its
#       'fixed-point merge:' line and no 'ordered merge:' line. Word merge: the parse-time line, the setup line, the
#       units line's note (the option switched the units on), the merge's line and its report ('the word merge: ...',
#       the carried components counted), no per-accumulator unit table. 9-B: the report with each accumulator's unit
#       and headroom
#   X2  the word merge's reference is the one domain: both sets at '--partition 1 1 1' (the pass's own words through
#       the same close-out) and at '2 2 2' -- every output byte-identical to the 3^3 runs. Both merges: against the
#       default (ordered) merge of the same settings, a table of the largest difference per output grid relative to
#       the grid's peak, every one below 2e-6 (the default merge rounds each partition's grids and adds floats)
#   X3  '--scratch-dir' with a tiny threshold (DTFE_SCRATCH_MIN_GB): the integer grids are scratch-backed, the outputs
#       byte-identical to X1's; on macOS they take the zero-fill route (the count after the loop is > 0)
#   X4  no wrap, whatever the order. Word merge: DTFE_TEST_RUN_UNIT_SHIFT=20 (every run-wide unit 2^20 finer: a cell
#       of the mean density holds 2^67..2^68 units, beyond the 64-bit words, so the passes and the merge take carries)
#       at --max-concurrent 1 and at 4 with partition 0 held back: byte-identical, the report counts carried
#       components, no share left out, '.den' '.vel' '.a_den' '.a_vel' within 1e-6 of each grid's rms of the shift-0
#       run; DTFE_TEST_RUN_UNIT_SHIFT=200 (no unit in range) is refused at start-up, naming the grids, with no output.
#       (The words cannot overflow: a cell's carry would have to leave the int64 range, a cell of 2^127 units; that
#       stop, fxCarryFinish's, has no test hook.) The 9-B merge (DTFE_RUN_UNITS_LEGACY=1 on each run, in both modes):
#       DTFE_TEST_FXMERGE_SHIFT=24 (units 2^24 finer: densities above 8 do not fit 2^63 units) stops the run at
#       --max-concurrent 1 and at 4 with partition 0 held back, naming the accumulator, with no output written;
#       DTFE_TEST_FXMERGE_SUM_LIMIT=40 (the sum bound's limit lowered) stops it with the sum message
#   X5  refusals: the deprecated '--completion-order-merge' beside it, '--grid-window', the double build (when built).
#       Word merge: DTFE_GPU_FLOAT_MASS=1 and DTFE_GPU_FLOAT_MOMENTS=1 (float sums, no words) refused at start-up,
#       naming the knob; beside DTFE_RUN_UNITS_LEGACY=1 accepted (the 9-B merge, its report with the units)
#   X6  resume: a run killed after 10 merges (DTFE_TEST_EXIT_AFTER_PARTITIONS=10, 4 in flight, partition 0 held back)
#       holds the integer words in its checkpoint (the word merge: 'x.fxw.<side>.<accumulator>', the counts, the
#       state, and the key's 'units=run-wide ... K=' line); '--resume' at --max-concurrent 2 writes X1's outputs byte
#       for byte; refused by name: a '--resume' without '--ps-merge' of that checkpoint (the merge mode is not taken
#       from it), a '--ps-merge fixed-point --resume' of a default run's checkpoint, a resume with other units (the
#       word merge: DTFE_TEST_RUN_UNIT_SHIFT=1, both 'units=' lines named; the 9-B merge: DTFE_TEST_FXMERGE_SHIFT=1),
#       and, word merge, a resume across the two merges in either direction (DTFE_RUN_UNITS_LEGACY=1)
#   X7  GPU builds only (SKIP otherwise): the --gpu deposit at --max-concurrent 1, 2 and 4 with partition 0 held
#       back -- byte-identical (none falling back to the CPU); the word merge: and at '--partition 1 1 1'; against the
#       GPU default merge, the X2 table. The word merge: X8's '_a' face set on the GPU (its face sums from the CPU
#       walk beside the device), '--partition 1 1 1' and '3 3 3' byte-identical
#   X8  the word merge's face sums ONCE PER FACE (fix 3 S7; the word merge only): the merged grid holds a cell's
#       upper faces (and, on a grid that is not periodic, one boundary plane per axis), a pass both faces per cell;
#       the merge skips a lower face whose twin the pass holds and routes one without. The '_a' face set FG
#       ('gradient_a divergence_a shear_a vorticity_a', '--exact --ps-volume-weighted --face-gradient'), periodic:
#       '--partition 1 1 1', '2 2 2' and '3 3 3' byte-identical; the face set on both sides FU ('divergence' and
#       'divergence_a'): '1 1 1' and '3 3 3'; FG in a '--box' (not periodic): '1 1 1', '2 2 2' and '3 3 3'.
#       DTFE_DEBUG_FACE_TWINS=1 on some of them (the bytes the same as without it): one 'face twins (side u|a)' line
#       per side, 0 mismatches; a periodic one domain finds every lower face's twin (48^3 cells x 3 axes x 5
#       components = 1658880 pairs) and routes none; one in a '--box' routes exactly layer 0's lower faces (48^2 x 3
#       x 5 = 34560 words), all to the boundary planes (some of them non-zero), and finds the other 47 layers' twins
#       (1624320 pairs); a 3^3 split routes words and, periodic, none to a plane. Resume: FG in a '--box' at 3^3
#       killed after 10 merges holds the face sums in the once layout ('x.fxw.a.faceS 8 995328': 48^3 x 9; faceN,
#       faceP 48^3 x 3) with the boundary planes ('x.fxw.a.faceS.plane<a> 8 6912': 48^2 x 3; faceN, faceP 2304) and
#       the state (2 x 24 entries); its '--resume' writes the one domain's outputs byte for byte. (The sampled
#       deposit has no face sums: X2's set 1 holds its one-domain identity, set 2 a u-side exact one.)
# Usage: tests/ps_fixed_merge_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
#        DTFE_RUN_UNITS_LEGACY=1 tests/ps_fixed_merge_check.sh   (every case on the merge of before)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair (which refuses the flag: X5 only)
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
BIN="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
TMP="${DTFE_TEST_TMP}/ps_fixed_merge_check"
rm -rf "$TMP"; mkdir -p "$TMP"
# the merge under test: WM=1 the word merge (the default), 0 the merge of before (DTFE_RUN_UNITS_LEGACY=1 exported by
# the caller; the binaries read any value but 0, false, no, off and empty as on). The knobs that would steer the runs
# otherwise are the suite's own: each case sets what it needs
case "${DTFE_RUN_UNITS_LEGACY:-}" in
    ""|0|false|FALSE|False|no|NO|No|off|OFF|Off) WM=1 ;;
    *) WM=0 ;;
esac
unset DTFE_PS_RUN_UNITS DTFE_TEST_RUN_UNIT_SHIFT DTFE_TEST_FXMERGE_SHIFT DTFE_TEST_FXMERGE_SUM_LIMIT DTFE_GPU_FLOAT_MASS \
      DTFE_GPU_FLOAT_MOMENTS DTFE_TEST_EXIT_AFTER_PARTITIONS DTFE_TEST_DELAY_PARTITION DTFE_CHECKPOINT \
      DTFE_DEBUG_FACE_TWINS
if [ "$WM" = 1 ]; then
    unset DTFE_RUN_UNITS_LEGACY
    echo ">> the merge under test: the word merge (the run-wide units' words, one close-out)"
else
    export DTFE_RUN_UNITS_LEGACY=1
    echo ">> the merge under test: the merge of before (DTFE_RUN_UNITS_LEGACY=1: each partition's rounded grids)"
fi
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 40 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
# a scalar per particle (read with --scalar-dataset Potential), so the scalar moments are merged too
"$PY" - "$SNAP" <<'PYEOF' || { echo "FAIL scalar dataset"; exit 1; }
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "a") as f:
    x = f["PartType1/Coordinates"][:]
    f["PartType1/Potential"] = (1e3 * np.sin(x[:, 0] / 10.) + x[:, 1]).astype(np.float32)
PYEOF
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# same <a> <b>: tests/lib.sh
# table <a> <b> <limit>: the largest |a - b| of every output grid relative to its peak; fails above the limit
table() {
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import glob, numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
a, b, lim = sys.argv[1], sys.argv[2], float(sys.argv[3])
worst, n = 0., 0
print(f"    {'grid':<16} {'max|d|/peak':>12} {'cells equal':>12}")
for f in sorted(glob.glob(a + ".*")):
    e = f[len(a):]
    if e.endswith(".log") or ".runlog" in e or ".part." in e: continue
    x = np.fromfile(f, dtype=REAL).astype(np.float64); y = np.fromfile(b + e, dtype=REAL).astype(np.float64)
    if x.size != y.size or not x.size: print(f"    {e}: sizes differ"); sys.exit(1)
    peak = max(np.abs(x).max(), 1e-300)
    r = float(np.abs(x - y).max() / peak)
    worst = max(worst, r); n += 1
    print(f"    {e:<16} {r:12.2e} {np.mean(x == y):12.4f}")
print(f"    worst {worst:.2e} over {n} grids (limit {lim:g})")
sys.exit(0 if n and worst < lim else 1)
PYEOF
}
C0=(--grid 48 --periodic --input-format 105 --mpc-unit 1 --verbose 2)
C=("${C0[@]}" --partition 3 3 3)
# set 1 runs the sampled deposit at nSub 3 (its '_a' fields would select the exact one) and set 2 the exact one, so
# X1 and X2 cover both; X4's overflow and sum bounds of the 9-B merge were set on set 1's sampled grids
F1=(--field density velocity gradient dispersion scalar scalarGradient density_a velocity_a --scalar-dataset Potential
    --sampled --subsamples 3)
F2=(--field density velocity dispersion --ps-volume-weighted --exact --ps-caustics)
FX=(--ps-merge fixed-point)
# the word merge's lines: the option's note on the units line, and its 'partition merge order' text
NOTE_WM="on for '--ps-merge fixed-point' (its merge adds the passes' words)"
ORDER_WM="partition merge order  : as they finish, into exact integer grids (--ps-merge fixed-point: the passes' words"

if precision_double; then
    echo ">> X5 (double build): --ps-merge fixed-point is refused"
    "$BIN" "$SNAP" "$TMP/dbl" "${C[@]}" "${F1[@]}" "${FX[@]}" > "$TMP/dbl.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -q "not available in the double build" "$TMP/dbl.log" && [ ! -e "$TMP/dbl.den" ]
    check "X5 the double build refuses --ps-merge fixed-point (exit $rc)" $?
    echo "RESULT: $PASS passed, $FAILS failed"
    [ "$FAILS" = 0 ] || exit 1
    exit 0
fi

echo ">> X1: the CPU deposit at --max-concurrent 1, 2, 4 and 4 with partition 0 held back"
"$BIN" --help > "$TMP/help.txt" 2>&1   # (to a file: 'grep -q' on the pipe can SIGPIPE the writer under pipefail)
grep -q -- "--ps-merge" "$TMP/help.txt" && grep -q "fixed-point" "$TMP/help.txt"; check "X1 '--ps-merge fixed-point' is in the help" $?
for s in 1 2; do
    eval "F=(\"\${F$s[@]}\")"
    OMP_NUM_THREADS=1 "$BIN" "$SNAP" "$TMP/s${s}m1" "${C[@]}" "${F[@]}" "${FX[@]}" --max-concurrent 1 --no-overlap > "$TMP/s${s}m1.log" 2>&1 \
        || { echo "FAIL: s${s}m1 (see $TMP/s${s}m1.log)"; exit 1; }
    OMP_NUM_THREADS=2 "$BIN" "$SNAP" "$TMP/s${s}m2" "${C[@]}" "${F[@]}" "${FX[@]}" --max-concurrent 2 > "$TMP/s${s}m2.log" 2>&1 \
        || { echo "FAIL: s${s}m2 (see $TMP/s${s}m2.log)"; exit 1; }
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s${s}m4" "${C[@]}" "${F[@]}" "${FX[@]}" --max-concurrent 4 > "$TMP/s${s}m4.log" 2>&1 \
        || { echo "FAIL: s${s}m4 (see $TMP/s${s}m4.log)"; exit 1; }
    OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP" "$TMP/s${s}d4" "${C[@]}" "${F[@]}" "${FX[@]}" --max-concurrent 4 \
        > "$TMP/s${s}d4.log" 2>&1 || { echo "FAIL: s${s}d4 (see $TMP/s${s}d4.log)"; exit 1; }
    same "$TMP/s${s}m1" "$TMP/s${s}m2" && same "$TMP/s${s}m1" "$TMP/s${s}m4" && same "$TMP/s${s}m1" "$TMP/s${s}d4"
    check "X1 set $s: --max-concurrent 1, 2, 4 and 4 in another completion order -- every output byte-identical" $?
done
if [ "$WM" = 1 ]; then
    grep -qF "$ORDER_WM" "$TMP/s1d4.log" \
        && grep -qF -- "--ps-merge fixed-point: the passes hand over their exact words" "$TMP/s1d4.log" \
        && grep -qF "$NOTE_WM" "$TMP/s1d4.log" \
        && grep -qF "each partition's exact words are added as it finishes" "$TMP/s1d4.log" \
        && grep -q "fixed-point merge: .* GB of integer grids at most (the word merge" "$TMP/s1d4.log" \
        && grep -q "fixed-point merge: .* GB of integer grids at most (the word merge" "$TMP/s2d4.log" \
        && grep -q "component(s) carried beyond 64 bits" "$TMP/s2d4.log" \
        && ! grep -q "u.scalar: unit 2^-" "$TMP/s1d4.log" \
        && ! grep -q "ordered merge:" "$TMP/s1d4.log" && ! grep -q "partitions merge in index order" "$TMP/s1d4.log"
    check "X1 the log names the word merge (parse-time line, setup, units note, merge line, report), no unit table and no ordered-merge line" $?
else
    grep -q "partition merge order  : as they finish, into exact integer grids" "$TMP/s1d4.log" \
        && grep -q "fixed-point merge: .* GB of integer grids" "$TMP/s1d4.log" && grep -q "u.scalar: unit 2^-" "$TMP/s1d4.log" \
        && grep -q "u.tet_touch: unit 2^-" "$TMP/s2d4.log" && grep -q "u.disp_velocity: unit 2^-" "$TMP/s2d4.log" \
        && ! grep -q "ordered merge:" "$TMP/s1d4.log" && ! grep -q "partitions merge in index order" "$TMP/s1d4.log" \
        && ! grep -qF -- "--ps-merge fixed-point: the passes hand over their exact words" "$TMP/s1d4.log"
    check "X1 the log names the mode and gives the 'fixed-point merge:' line (units, headroom), no ordered-merge line" $?
fi
grep -a "fixed-point merge:" "$TMP/s1d4.log" | tr ';' '\n' | sed 's/^ */    /' | head -20

echo ">> X2: the reference"
if [ "$WM" = 1 ]; then
    # the word merge adds integers: the one domain (its pass's words, the same close-out) and every split are one sum
    for s in 1 2; do
        eval "F=(\"\${F$s[@]}\")"
        OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s${s}p1" "${C0[@]}" --partition 1 1 1 "${F[@]}" "${FX[@]}" \
            > "$TMP/s${s}p1.log" 2>&1 || { echo "FAIL: s${s}p1 (see $TMP/s${s}p1.log)"; exit 1; }
        OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s${s}p2" "${C0[@]}" --partition 2 2 2 "${F[@]}" "${FX[@]}" --max-concurrent 4 \
            > "$TMP/s${s}p2.log" 2>&1 || { echo "FAIL: s${s}p2 (see $TMP/s${s}p2.log)"; exit 1; }
        same "$TMP/s${s}m1" "$TMP/s${s}p1" && same "$TMP/s${s}m1" "$TMP/s${s}p2" \
            && grep -qF "$NOTE_WM" "$TMP/s${s}p1.log" && grep -qF "the one domain's grid after the close-out" "$TMP/s${s}p1.log"
        check "X2 set $s: one domain ('--partition 1 1 1', its close-out) and 2^3 -- every output byte-identical to 3^3" $?
    done
fi
for s in 1 2; do
    eval "F=(\"\${F$s[@]}\")"
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/s${s}def" "${C[@]}" "${F[@]}" --max-concurrent 4 > "$TMP/s${s}def.log" 2>&1 \
        || { echo "FAIL: s${s}def (see $TMP/s${s}def.log)"; exit 1; }
    table "$TMP/s${s}def" "$TMP/s${s}m4" 2e-6
    check "X2 set $s: --ps-merge fixed-point differs from the ordered merge by < 2e-6 of each grid's peak" $?
done

echo ">> X3: --scratch-dir backs the integer grids"
SC="$TMP/scratch"; mkdir -p "$SC"
OMP_NUM_THREADS=4 DTFE_SCRATCH_MIN_GB=0.0001 "$BIN" "$SNAP" "$TMP/sc" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 4 \
    --scratch-dir "$SC" > "$TMP/sc.log" 2>&1 || { echo "FAIL: sc (see $TMP/sc.log)"; exit 1; }
grep -q "SCRATCH:" "$TMP/sc.log" && same "$TMP/s1m1" "$TMP/sc" && [ -z "$(ls -A "$SC")" ]
check "X3 scratch-backed integer grids (a tiny threshold): byte-identical, the scratch folder left empty" $?
# the zero-fill route for the integer grids too (ScratchAlloc::zeroAssign, ps_fixed_merge.h): the line after the loop
# counts the grids moved onto their scratch file unwritten since the merge's setup -- on macOS every integer grid of
# the setup (all scratch mappings here), so N > 0; elsewhere the route is off by design (scratch_alloc.h), N = 0
NFX="$(sed -n "s/^SCRATCH: .*, \([0-9][0-9]*\) of them since the fixed-point merge's setup.*/\1/p" "$TMP/sc.log" | head -1)"
if [ "$(uname -s)" = Darwin ]; then
    [ "${NFX:-0}" -gt 0 ]
    check "X3 ${NFX:-no} integer grid(s) zero-filled in memory and moved onto their scratch file (the zero-fill route)" $?
else
    [ "${NFX:-x}" = 0 ]
    check "X3 the zero-fill route is off here (macOS only): ${NFX:-no} integer grid(s) moved, the count printed" $?
fi

echo ">> X4: no wrap, whatever the order"
if [ "$WM" = 1 ]; then
    # the carries: PO1's fields of tests/split_bytes_check.sh (no gradient, whose caustic shares would reach the cap
    # at this shift), the exact deposit
    FC=(--field density velocity density_a velocity_a dispersion_a --exact)
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/c0" "${C[@]}" "${FC[@]}" "${FX[@]}" --max-concurrent 4 > "$TMP/c0.log" 2>&1 \
        || { echo "FAIL: c0 (see $TMP/c0.log)"; exit 1; }
    OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=20 "$BIN" "$SNAP" "$TMP/c1" "${C[@]}" "${FC[@]}" "${FX[@]}" \
        --max-concurrent 1 --no-overlap > "$TMP/c1.log" 2>&1 || { echo "FAIL: c1 (see $TMP/c1.log)"; exit 1; }
    OMP_NUM_THREADS=4 DTFE_TEST_RUN_UNIT_SHIFT=20 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP" "$TMP/c4" "${C[@]}" \
        "${FC[@]}" "${FX[@]}" --max-concurrent 4 > "$TMP/c4.log" 2>&1 || { echo "FAIL: c4 (see $TMP/c4.log)"; exit 1; }
    NC="$(sed -n 's/.*; \([0-9][0-9]*\) component(s) carried beyond 64 bits.*/\1/p' "$TMP/c4.log" | head -1)"
    same "$TMP/c1" "$TMP/c4" && [ "${NC:-0}" -gt 0 ] \
        && ! grep -q "share(s) of the exact fixed-point sums" "$TMP/c1.log" "$TMP/c4.log"
    check "X4 DTFE_TEST_RUN_UNIT_SHIFT=20 at --max-concurrent 1 and 4 (partition 0 held back): byte-identical, ${NC:-no} component(s) carried, no share left out" $?
    "$PY" - "$TMP/c4" "$TMP/c0" <<'PYEOF'
import numpy as np, os, sys
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))
ok = True
for e in ("den", "vel", "a_den", "a_vel"):
    x, y = (np.fromfile(p + "." + e, dtype=REAL).astype(np.float64) for p in sys.argv[1:3])
    r = float(np.sqrt((y ** 2).mean())) if y.size else 0.
    q = float(np.abs(x - y).max()) / r if x.size == y.size and x.size and r > 0. else float("inf")
    print(f"    .{e}: shift 20 against shift 0, max difference {q:.2g} of the rms (bound 1e-6)")
    ok = ok and q <= 1e-6
sys.exit(0 if ok else 1)
PYEOF
    check "X4 the shifted words (carries) give the shift-0 grids to 1e-6 of their rms" $?
    OMP_NUM_THREADS=4 DTFE_TEST_RUN_UNIT_SHIFT=200 "$BIN" "$SNAP" "$TMP/cx" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 4 \
        > "$TMP/cx.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -q "which needs one fixed-point unit per grid for the whole run" "$TMP/cx.log" \
        && grep -q "lies outside 2^-120 .. 2^120" "$TMP/cx.log" && [ ! -e "$TMP/cx.den" ]
    check "X4 DTFE_TEST_RUN_UNIT_SHIFT=200 (no unit in range): refused at start-up (exit $rc), naming the grids, no output" $?
fi
# the merge of before (DTFE_RUN_UNITS_LEGACY=1, exported already when it is the merge under test)
for r in "o1:1:" "o4:4:0:2"; do
    IFS=: read -r name mc dp ds <<< "$r"
    env DTFE_RUN_UNITS_LEGACY=1 OMP_NUM_THREADS=4 DTFE_TEST_FXMERGE_SHIFT=24 ${dp:+DTFE_TEST_DELAY_PARTITION=$dp:$ds} "$BIN" \
        "$SNAP" "$TMP/$name" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent "$mc" > "$TMP/$name.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -q -- "--ps-merge fixed-point: partition [0-9]* holds a value of [ua]\.[a-z_]*" "$TMP/$name.log" \
        && grep -q "2^63 units" "$TMP/$name.log" && [ ! -e "$TMP/$name.den" ]
    check "X4 (the merge of before) DTFE_TEST_FXMERGE_SHIFT=24 at --max-concurrent $mc${dp:+ (partition 0 held back)}: stopped (exit $rc), naming the accumulator, no output" $?
done
env DTFE_RUN_UNITS_LEGACY=1 OMP_NUM_THREADS=4 DTFE_TEST_FXMERGE_SUM_LIMIT=40 "$BIN" "$SNAP" "$TMP/os" "${C[@]}" "${F1[@]}" \
    "${FX[@]}" --max-concurrent 4 > "$TMP/os.log" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "the merged partitions' largest values of .* sum to 2^" "$TMP/os.log" && [ ! -e "$TMP/os.den" ]
check "X4 (the merge of before) DTFE_TEST_FXMERGE_SUM_LIMIT=40: the sum bound stops the run (exit $rc)" $?

echo ">> X5: refusals"
# (two modes cannot be one --ps-merge: the deprecated switch pair is the refusal kept)
"$BIN" "$SNAP" "$TMP/bad1" "${C[@]}" "${F1[@]}" --fixed-point-merge --completion-order-merge > "$TMP/bad1.log" 2>&1; rc1=$?
"$BIN" "$SNAP" "$TMP/bad2" "${C[@]}" "${F1[@]}" "${FX[@]}" --grid-window 0 50 0 50 0 50 > "$TMP/bad2.log" 2>&1; rc2=$?
[ "$rc1" != 0 ] && grep -q "contradict each other" "$TMP/bad1.log" && [ "$rc2" != 0 ] \
    && grep -q "cannot be combined with '--grid-window'" "$TMP/bad2.log" && [ ! -e "$TMP/bad1.den" ] && [ ! -e "$TMP/bad2.den" ]
check "X5 the deprecated '--completion-order-merge' (exit $rc1) and '--grid-window' (exit $rc2) beside it are refused" $?
if [ -x ./PS-DTFE-double ]; then
    ./PS-DTFE-double "$SNAP" "$TMP/dbl" "${C[@]}" --field density "${FX[@]}" > "$TMP/dbl.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -q "not available in the double build" "$TMP/dbl.log" && [ ! -e "$TMP/dbl.den" ]
    check "X5 the double build refuses it (exit $rc)" $?
else
    echo "  SKIP X5 double: ./PS-DTFE-double is not built"
fi
if [ "$WM" = 1 ]; then
    # the float A/B knobs have no words to merge: refused before any work, the knob named
    DTFE_GPU_FLOAT_MASS=1 "$BIN" "$SNAP" "$TMP/fm" "${C[@]}" "${F1[@]}" "${FX[@]}" > "$TMP/fm.log" 2>&1; rc1=$?
    DTFE_GPU_FLOAT_MOMENTS=1 "$BIN" "$SNAP" "$TMP/fv" "${C[@]}" "${F1[@]}" "${FX[@]}" > "$TMP/fv.log" 2>&1; rc2=$?
    [ "$rc1" != 0 ] && grep -q "DTFE_GPU_FLOAT_MASS=1 sums in floats, which have none" "$TMP/fm.log" \
        && [ "$rc2" != 0 ] && grep -q "DTFE_GPU_FLOAT_MOMENTS=1 sums in floats, which have none" "$TMP/fv.log" \
        && [ ! -e "$TMP/fm.den" ] && [ ! -e "$TMP/fv.den" ] && ! grep -q "Lagrangian-space partitioning" "$TMP/fm.log"
    check "X5 DTFE_GPU_FLOAT_MASS=1 (exit $rc1) and DTFE_GPU_FLOAT_MOMENTS=1 (exit $rc2) refused at start-up, the knob named" $?
    # ... and beside DTFE_RUN_UNITS_LEGACY=1 the merge of before takes them (its report names each accumulator's unit)
    OMP_NUM_THREADS=4 DTFE_RUN_UNITS_LEGACY=1 DTFE_GPU_FLOAT_MASS=1 "$BIN" "$SNAP" "$TMP/fl" "${C[@]}" "${F1[@]}" "${FX[@]}" \
        --max-concurrent 4 > "$TMP/fl.log" 2>&1; rc=$?
    [ "$rc" = 0 ] && grep -q "u.scalar: unit 2^-" "$TMP/fl.log" && ! grep -qF "$NOTE_WM" "$TMP/fl.log" \
        && ! grep -qF -- "--ps-merge fixed-point: the passes hand over their exact words" "$TMP/fl.log"
    check "X5 DTFE_GPU_FLOAT_MASS=1 beside DTFE_RUN_UNITS_LEGACY=1 runs the merge of before (exit $rc)" $?
fi

echo ">> X6: resume"
OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" \
    "${FX[@]}" --max-concurrent 4 > "$TMP/k.log" 2>&1; rc1=$?
CK="$TMP/k.runlog.checkpoint/checkpoint"
nm="$(grep -a -m1 '^merged_count' "$CK" 2>/dev/null | awk '{print $2}')"
if [ "$WM" = 1 ]; then
    [ "$rc1" = 137 ] && [ "${nm:-0}" = 10 ] && grep -aq "^field x.fxw.u.mass 8 " "$CK" && grep -aq "^field x.fxn.u 4 " "$CK" \
        && grep -aq "^field x.fxwords.state " "$CK" && grep -aq "^merge=fixed-point" "$CK" \
        && grep -aq "^units=run-wide N=48/61/72 shift=0 K=mass:" "$CK" && ! grep -aq "^field u.density " "$CK"
    check "X6 killed after 10 merges (exit $rc1): the checkpoint holds ${nm:-?} partitions as words, counts and their units" $?
else
    [ "$rc1" = 137 ] && [ "${nm:-0}" = 10 ] && grep -aq "^field x.fxw.u.density 8 " "$CK" && grep -aq "^field x.fxc.u.density 1 " "$CK" \
        && grep -aq "^field x.fxmerge.state " "$CK" && grep -aq "^merge=fixed-point" "$CK" && ! grep -aq "^field u.density " "$CK"
    check "X6 killed after 10 merges (exit $rc1): the checkpoint holds ${nm:-?} partitions as integer words and carries" $?
fi
# '--resume' without '--ps-merge': the merge mode is not taken from the checkpoint, the resume is refused by name
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" --max-concurrent 4 --resume > "$TMP/k_other.log" 2>&1; rc2=$?
if [ "$WM" = 1 ]; then
    OMP_NUM_THREADS=4 DTFE_TEST_RUN_UNIT_SHIFT=1 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 4 \
        --resume > "$TMP/k_shift.log" 2>&1; rc3=$?
    [ "$rc2" != 0 ] && grep -q "was written by a run with '--ps-merge fixed-point'" "$TMP/k_other.log" \
        && [ "$rc3" != 0 ] && grep -qF "holds the '--ps-merge fixed-point' sums of other fixed-point units" "$TMP/k_shift.log" \
        && grep -qF "checkpoint: units=run-wide N=48/61/72 shift=0 K=mass:" "$TMP/k_shift.log" \
        && grep -qF "this run  : units=run-wide N=48/61/72 shift=1 K=mass:" "$TMP/k_shift.log" && [ -s "$CK" ]
    check "X6 a '--resume' without '--ps-merge' (exit $rc2) and one with other units (exit $rc3, both 'units=' lines named) are refused; the checkpoint is kept" $?
    # the other of the two merges cannot continue it
    OMP_NUM_THREADS=4 DTFE_RUN_UNITS_LEGACY=1 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 4 \
        --resume > "$TMP/k_leg.log" 2>&1; rc=$?
    [ "$rc" != 0 ] && grep -qF "holds the '--ps-merge fixed-point' sums of other fixed-point units" "$TMP/k_leg.log" \
        && grep -qF "checkpoint: units=run-wide N=48/61/72 shift=0 K=mass:" "$TMP/k_leg.log" \
        && grep -qF "this run  : units=legacy N=48/61/72 shift=0" "$TMP/k_leg.log" && [ -s "$CK" ]
    check "X6 the word merge's checkpoint under DTFE_RUN_UNITS_LEGACY=1 (the merge of before) is refused (exit $rc)" $?
else
    OMP_NUM_THREADS=4 DTFE_TEST_FXMERGE_SHIFT=1 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 4 --resume \
        > "$TMP/k_shift.log" 2>&1; rc3=$?
    [ "$rc2" != 0 ] && grep -q "was written by a run with '--ps-merge fixed-point'" "$TMP/k_other.log" \
        && [ "$rc3" != 0 ] && grep -q "the checkpoint's fixed-point unit of u.density" "$TMP/k_shift.log" && [ -s "$CK" ]
    check "X6 a default '--resume' of it (exit $rc2) and one with other units (exit $rc3) are refused; the checkpoint is kept" $?
fi
OMP_NUM_THREADS=2 "$BIN" "$SNAP" "$TMP/k" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 2 --resume > "$TMP/k2.log" 2>&1; rc4=$?
[ "$rc4" = 0 ] && same "$TMP/s1m1" "$TMP/k" && [ ! -e "$TMP/k.runlog.checkpoint" ] \
    && { [ "$WM" = 0 ] || grep -qF "words added to the checkpoint's as it finishes" "$TMP/k2.log"; }
check "X6 '--ps-merge fixed-point --resume' at --max-concurrent 2 (exit $rc4): byte-identical to the uninterrupted runs" $?
OMP_NUM_THREADS=4 DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "$BIN" "$SNAP" "$TMP/kd" "${C[@]}" "${F1[@]}" --max-concurrent 4 \
    > "$TMP/kd.log" 2>&1; rc5=$?
OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/kd" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 4 --resume > "$TMP/kd2.log" 2>&1; rc6=$?
[ "$rc5" = 137 ] && [ "$rc6" != 0 ] && grep -q "was written by a run with the default merge" "$TMP/kd2.log"
check "X6 a default run's checkpoint resumed with '--ps-merge fixed-point' is refused (exit $rc6)" $?
if [ "$WM" = 1 ]; then
    # a checkpoint of the merge of before (killed under DTFE_RUN_UNITS_LEGACY=1) cannot be continued by the word merge
    OMP_NUM_THREADS=4 DTFE_RUN_UNITS_LEGACY=1 DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "$BIN" "$SNAP" "$TMP/kl" "${C[@]}" "${F1[@]}" \
        "${FX[@]}" --max-concurrent 4 > "$TMP/kl.log" 2>&1; rc5=$?
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/kl" "${C[@]}" "${F1[@]}" "${FX[@]}" --max-concurrent 4 --resume \
        > "$TMP/kl2.log" 2>&1; rc6=$?
    [ "$rc5" = 137 ] && [ "$rc6" != 0 ] && grep -qF "holds the '--ps-merge fixed-point' sums of other fixed-point units" "$TMP/kl2.log" \
        && grep -qF "checkpoint: units=legacy N=48/61/72 shift=0" "$TMP/kl2.log" \
        && grep -qF "this run  : units=run-wide N=48/61/72 shift=0 K=mass:" "$TMP/kl2.log" \
        && [ -s "$TMP/kl.runlog.checkpoint/checkpoint" ]
    check "X6 a checkpoint of the merge of before (exit $rc5) resumed by the word merge is refused (exit $rc6), both 'units=' lines named" $?
fi

echo ">> X7: the GPU deposit"
OBJ=o_ps
if [ ! -d "$OBJ" ] || [ -f "$OBJ/.gpu_mode_off" ] || [ ! -f "$OBJ/.gpu_mode_metal" ]; then
    echo "  SKIP X7: $OBJ is not a Metal GPU build"
else
    # '--sampled --subsamples 3': the GPU's sampled deposit at nSub 3, which this case has always run (with
    # 'density_a' and no flag the deposit is the exact one since 2026-11-09)
    GF=(--field density velocity gradient dispersion density_a --ps-volume-weighted --gpu --sampled --subsamples 3)
    G=("${C[@]}" "${GF[@]}")
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/g1" "${G[@]}" "${FX[@]}" --max-concurrent 1 > "$TMP/g1.log" 2>&1 || { echo "FAIL: g1 (see $TMP/g1.log)"; exit 1; }
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/g2" "${G[@]}" "${FX[@]}" --max-concurrent 2 > "$TMP/g2.log" 2>&1 || { echo "FAIL: g2 (see $TMP/g2.log)"; exit 1; }
    OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 "$BIN" "$SNAP" "$TMP/g4" "${G[@]}" "${FX[@]}" --max-concurrent 4 > "$TMP/g4.log" 2>&1 \
        || { echo "FAIL: g4 (see $TMP/g4.log)"; exit 1; }
    gl=("$TMP/g1.log" "$TMP/g2.log" "$TMP/g4.log")
    if [ "$WM" = 1 ]; then
        OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/g0" "${C0[@]}" --partition 1 1 1 "${GF[@]}" "${FX[@]}" > "$TMP/g0.log" 2>&1 \
            || { echo "FAIL: g0 (see $TMP/g0.log)"; exit 1; }
        gl+=("$TMP/g0.log")
    fi
    ! grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead\|gave up" "${gl[@]}"
    check "X7 no GPU pass fell back to the CPU" $?
    same "$TMP/g1" "$TMP/g2" && same "$TMP/g1" "$TMP/g4"
    check "X7 --gpu at --max-concurrent 1, 2 and 4 (partition 0 held back): every output byte-identical" $?
    if [ "$WM" = 1 ]; then
        same "$TMP/g1" "$TMP/g0"
        check "X7 --gpu at '--partition 1 1 1' (the one domain's close-out): every output byte-identical to 3^3" $?
    fi
    OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/gdef" "${G[@]}" --max-concurrent 4 > "$TMP/gdef.log" 2>&1 || { echo "FAIL: gdef (see $TMP/gdef.log)"; exit 1; }
    table "$TMP/gdef" "$TMP/g4" 2e-6
    check "X7 against the GPU run's default merge: < 2e-6 of each grid's peak" $?
    if [ "$WM" = 1 ]; then
        # X8's '_a' face set on the GPU: the face sums from the CPU walk beside the device, once per face in the merge
        GFG=(--field density_a velocity_a gradient_a divergence_a shear_a vorticity_a --exact --ps-volume-weighted
             --face-gradient --gpu)
        for p in 1 3; do
            OMP_NUM_THREADS=4 "$BIN" "$SNAP" "$TMP/gf$p" "${C0[@]}" --partition $p $p $p "${GFG[@]}" "${FX[@]}" \
                --max-concurrent 4 > "$TMP/gf$p.log" 2>&1 || { echo "FAIL: gf$p (see $TMP/gf$p.log)"; exit 1; }
        done
        ! grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead\|gave up" "$TMP/gf1.log" \
            "$TMP/gf3.log" && same "$TMP/gf1" "$TMP/gf3" && [ -e "$TMP/gf1.a_velDivStreams" ]
        check "X7 the face set on the GPU: '1 1 1' and '3 3 3' -- every output byte-identical, no CPU fallback" $?
    fi
fi

echo ">> X8: the face sums once per face"
if [ "$WM" != 1 ]; then
    echo "  SKIP X8: the merge of before merges the partitions' Real face grids (no face words to hold once)"
else
    FG=(--field density_a velocity_a gradient_a divergence_a shear_a vorticity_a --exact --ps-volume-weighted
        --face-gradient)
    FU=(--field density velocity divergence density_a velocity_a divergence_a --exact --ps-volume-weighted
        --face-gradient)
    B0=(--grid 48 --box 20 80 20 80 20 80 --input-format 105 --mpc-unit 1 --verbose 2)
    # twins_of <log> <side>: the numbers of the side's DTFE_DEBUG_FACE_TWINS line (pairs, mismatches, words routed,
    # of them non-zero, to a boundary plane, of them non-zero, passes), or "none"
    twins_of() {
        local re v
        re="DTFE_DEBUG_FACE_TWINS: face twins (side $2): \([0-9]*\) pairs, \([0-9]*\) mismatches; \([0-9]*\) words"
        re="$re routed, \([0-9]*\) non-zero; \([0-9]*\) to a boundary plane, \([0-9]*\) non-zero (\([0-9]*\) pass"
        v="$(LC_ALL=C sed -n "s/.*${re}.*/\1 \2 \3 \4 \5 \6 \7/p" "$1" | tail -n 1)"
        echo "${v:-none}"
    }
    # x8run <name> <knob> <the run's grid options...>: one run of the set in F, DTFE_DEBUG_FACE_TWINS=<knob>; the
    # suite stops when it fails
    x8run() {
        local name="$1" knob="$2"
        shift 2
        OMP_NUM_THREADS=4 DTFE_DEBUG_FACE_TWINS="$knob" "$BIN" "$SNAP" "$TMP/$name" "$@" "${F[@]}" "${FX[@]}" \
            --max-concurrent 4 > "$TMP/$name.log" 2>&1 || { echo "FAIL: $name (see $TMP/$name.log)"; exit 1; }
    }
    F=("${FG[@]}")
    x8run fg1 1 "${C0[@]}" --partition 1 1 1
    x8run fg2 0 "${C0[@]}" --partition 2 2 2
    x8run fg3 1 "${C0[@]}" --partition 3 3 3
    same "$TMP/fg1" "$TMP/fg2" && same "$TMP/fg1" "$TMP/fg3" && [ -e "$TMP/fg1.a_velDivStreams" ] \
        && grep -qF "velocity gradient      : from EXACT FACE sums" "$TMP/fg1.log"
    check "X8 the '_a' face set, periodic: one domain, 2^3 and 3^3 -- every output byte-identical (2^3 no knob)" $?
    F=("${FU[@]}")
    x8run fu1 1 "${C0[@]}" --partition 1 1 1
    x8run fu3 1 "${C0[@]}" --partition 3 3 3
    same "$TMP/fu1" "$TMP/fu3" && [ -e "$TMP/fu1.a_velDivStreams" ]
    check "X8 the face sums of both sides (u and '_a'), periodic: one domain and 3^3 -- byte-identical" $?
    F=("${FG[@]}")
    x8run fb1 1 "${B0[@]}" --partition 1 1 1
    x8run fb2 0 "${B0[@]}" --partition 2 2 2
    x8run fb3 1 "${B0[@]}" --partition 3 3 3
    same "$TMP/fb1" "$TMP/fb2" && same "$TMP/fb1" "$TMP/fb3" && [ -e "$TMP/fb1.a_velDivStreams" ]
    check "X8 the '_a' face set in a '--box' (boundary planes): one domain, 2^3 and 3^3 -- byte-identical" $?
    # DTFE_DEBUG_FACE_TWINS=1: one line per side; the counts that follow from the geometry alone (48^3 cells)
    for r in fg1 fg3 fu1 fu3 fb1 fb3; do
        for s in u a; do echo "    $r side $s: $(twins_of "$TMP/$r.log" "$s")"; done
    done
    read -r p1 m1 r1 z1 b1 bz1 n1 <<< "$(twins_of "$TMP/fg1.log" a)"
    [ "${p1:-}" = 1658880 ] && [ "${m1:-}" = 0 ] && [ "${r1:-}" = 0 ] && [ "${b1:-}" = 0 ] && [ "${n1:-}" = 1 ] \
        && [ "$(twins_of "$TMP/fg1.log" u)" = none ]
    st=$?
    check "X8 the twins, periodic one domain: ${p1:-no} pairs (1658880), ${m1:-?} mismatches, ${r1:-?} routed" $st
    read -r p3 m3 r3 z3 b3 bz3 n3 <<< "$(twins_of "$TMP/fg3.log" a)"
    read -r pu mu ru zu bu bzu nu <<< "$(twins_of "$TMP/fg3.log" u)"
    [ "${m3:-}" = 0 ] && [ "${p3:-0}" -gt 0 ] && [ "${r3:-0}" -gt 0 ] && [ "${b3:-}" = 0 ] && [ "${n3:-}" = 27 ] \
        && [ "${pu:-}" = 0 ] && [ "${mu:-}" = 0 ] && [ "${nu:-}" = 0 ]
    st=$?
    msg="X8 the twins, periodic 3^3: ${p3:-no} pairs, ${m3:-?} mismatches, ${r3:-?} words routed (${z3:-?}"
    check "$msg non-zero), none to a plane; side u: no face pass" $st
    ok=0
    for r in fu1 fu3; do
        for s in u a; do
            read -r pf mf rf zf bf bzf nf <<< "$(twins_of "$TMP/$r.log" "$s")"
            [ "${mf:-}" = 0 ] && [ "${pf:-0}" -gt 0 ] && [ "${bf:-}" = 0 ] || ok=1
        done
    done
    check "X8 the twins, the face sums of both sides: a line per side, 0 mismatches, none to a plane" $ok
    read -r pb mb rb zb bb bzb nb <<< "$(twins_of "$TMP/fb1.log" a)"
    [ "${pb:-}" = 1624320 ] && [ "${mb:-}" = 0 ] && [ "${rb:-}" = 34560 ] && [ "${bb:-}" = 34560 ] \
        && [ "${zb:-}" = "${bzb:-x}" ] && [ "${bzb:-0}" -gt 0 ]
    st=$?
    msg="X8 the twins, one domain in a '--box': ${pb:-no} pairs (1624320), ${mb:-?} mismatches, ${rb:-?} words"
    check "$msg routed (34560), ${bb:-?} to the planes, ${bzb:-?} of them non-zero" $st
    read -r pc mc rc zc bc bzc nc <<< "$(twins_of "$TMP/fb3.log" a)"
    [ "${mc:-}" = 0 ] && [ "${pc:-0}" -gt 0 ] && [ "${rc:-0}" -gt "${bc:-0}" ] && [ "${bc:-0}" -gt 0 ] \
        && [ "${nc:-}" = 27 ]
    st=$?
    msg="X8 the twins, 3^3 in a '--box': ${pc:-no} pairs, ${mc:-?} mismatches, ${rc:-?} words routed, ${bc:-?} of"
    check "$msg them to the planes" $st
    # resume across the merge with the boundary planes: FG in a '--box' at 3^3, killed after 10 merges, resumed
    rm -rf "$TMP/fk.runlog.checkpoint"
    OMP_NUM_THREADS=4 DTFE_TEST_DELAY_PARTITION=0:2 DTFE_TEST_EXIT_AFTER_PARTITIONS=10 "$BIN" "$SNAP" "$TMP/fk" \
        "${B0[@]}" --partition 3 3 3 "${FG[@]}" "${FX[@]}" --max-concurrent 4 > "$TMP/fk.log" 2>&1; rc1=$?
    CK="$TMP/fk.runlog.checkpoint/checkpoint"
    nm="$(grep -a -m1 '^merged_count' "$CK" 2>/dev/null | awk '{print $2}')"
    planes=0
    for f in faceS:6912 faceN:2304 faceP:2304; do
        for a in 0 1 2; do grep -aq "^field x.fxw.a.${f%%:*}.plane$a 8 ${f##*:}\$" "$CK" || planes=1; done
    done
    [ "$rc1" = 137 ] && [ "${nm:-0}" = 10 ] && grep -aq '^field x.fxw.a.faceS 8 995328$' "$CK" \
        && grep -aq '^field x.fxw.a.faceN 8 331776$' "$CK" && grep -aq '^field x.fxw.a.faceP 8 331776$' "$CK" \
        && [ "$planes" = 0 ] && grep -aq '^field x.fxwords.state 16 48$' "$CK"
    st=$?
    msg="X8 killed after 10 merges in a '--box' (exit $rc1): the checkpoint holds ${nm:-?} partitions' face sums"
    check "$msg once per face, with their boundary planes" $st
    OMP_NUM_THREADS=2 "$BIN" "$SNAP" "$TMP/fk" "${B0[@]}" --partition 3 3 3 "${FG[@]}" "${FX[@]}" --max-concurrent 2 \
        --resume > "$TMP/fk2.log" 2>&1; rc2=$?
    [ "$rc2" = 0 ] && grep -qF "words added to the checkpoint's as it finishes" "$TMP/fk2.log" \
        && same "$TMP/fb1" "$TMP/fk" && [ ! -e "$TMP/fk.runlog.checkpoint" ]
    check "X8 '--resume' of it at --max-concurrent 2 (exit $rc2): the one domain's outputs byte for byte" $?
fi

echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ] || exit 1
exit 0
