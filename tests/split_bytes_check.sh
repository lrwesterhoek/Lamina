#!/usr/bin/env bash
# Output bytes that do not depend on the split: the run-wide fixed-point units (src/fixed_units.h RunUnits); cmp of every output file.
# Checks U1, S1-S4, O1, O2 print their own labels; the double build SKIPs (its CPU sums are doubles).
# Usage: tests/split_bytes_check.sh [--no-build]   (--no-build is accepted and ignored: nothing is built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair (SKIP below)
precision_require "${DTFE_BIN}"
if precision_double; then
    echo "SKIP: the double build keeps its double CPU sums, its double vertex star sums and CGAL's single-cell"
    echo "      volumes (src/fixed_units.h: the run-wide words serve the single build); their last bits follow the"
    echo "      split"
    exit 0
fi
[ -x "${DTFE_BIN}" ] || { echo "FAIL: ${DTFE_BIN} is not built"; exit 1; }
SNAP="demo/z0_64.gadget"
[ -f "$SNAP" ] || { echo "FAIL: $SNAP is missing (it is tracked in the repository)"; exit 1; }
TMP="${DTFE_TEST_TMP}/split_bytes_check"
rm -rf "$TMP"; mkdir -p "$TMP/u1"
GPU=0
[ -n "$(gpu_backend_of o)" ] && GPU=1
# what steers a run's sums, from the caller's shell: every case sets its own
unset DTFE_TEST_RUN_UNIT_SHIFT DTFE_FACE_CHECK DTFE_FACE_GRADIENT DTFE_TRIANGULATION_THREADS DTFE_DEPOSIT_THREADS \
      DTFE_GPU_ITEMS DTFE_GPU_FAULT_AT DTFE_GPU_TIMING OMP_NUM_THREADS
# pass / fail (FAIL=1 on a failure): tests/lib.sh
note() { echo "   NOTE  $*"; }

# the options of every run
COMMON=(--grid 64 --periodic --padding 12 --exact --run-mode performance --verbose 3)
F=(--field density_a velocity_a divergence_a shear_a)
FDV=(--field density_a velocity_a)

# run <name> [VAR=value ...] -- [args ...]: the standard binary on SNAP, its log in <name>.log. A failed run, or a
# '--gpu' run that fell back to the CPU, stops the suite; its units line is filed for U1 under its test shift.
run() {
    local name="$1"; shift
    local envs=()
    local e a ushift=0 gpu=0 n line
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    for e in ${envs[@]+"${envs[@]}"}; do
        case "$e" in
            DTFE_TEST_RUN_UNIT_SHIFT=*) ushift="${e#*=}" ;;
        esac
    done
    for a in "$@"; do [ "$a" = "--gpu" ] && gpu=1; done
    env ${envs[@]+"${envs[@]}"} "${DTFE_BIN}" "$SNAP" "$TMP/$name" "${COMMON[@]}" "$@" > "$TMP/$name.log" 2>&1 \
        || { echo "FAIL: run $name (see $TMP/$name.log)"; exit 1; }
    # (the reason's text, and the closing count: only the master thread's partitions print a reason)
    if [ "$gpu" = 1 ] && grep -aq "using the CPU interpolation\|ran on the CPU instead" "$TMP/$name.log"; then
        echo "FAIL: the --gpu run $name fell back to the CPU (see $TMP/$name.log)"; exit 1
    fi
    n="$(grep -a -c 'fixed-point units (run-wide): cell volume' "$TMP/$name.log")"
    line="$(grep -a -o 'fixed-point units (run-wide): cell volume.*' "$TMP/$name.log" | head -1)"
    echo "$name|$n|$line" >> "$TMP/u1/std.shift$ushift"
}
# outputs <run>: the suffixes of its output files, one per line; the logs, the run's bookkeeping and the padding
# test's '.badGridPoints<n>' are not outputs
outputs() {
    local f s
    for f in "$TMP/$1".*; do
        [ -e "$f" ] || continue
        s="${f#"$TMP/$1"}"
        case "$s" in .log|.runlog*|*.part.*|.badGridPoints*) continue ;; esac
        echo "$s"
    done
}
# differing <a> <b>: the outputs of run a that are not byte-identical to run b's, and those only b wrote (nothing:
# every output identical)
differing() {
    local d="" s n=0
    for s in $(outputs "$1"); do
        n=$((n+1))
        cmp -s "$TMP/$1$s" "$TMP/$2$s" || d="$d $s"
    done
    for s in $(outputs "$2"); do [ -e "$TMP/$1$s" ] || d="$d $s(only in $2)"; done
    [ "$n" -gt 0 ] || d=" (run $1 wrote no output)"
    echo "$d"
}
# expect_same <label> <a> <b>: every output of runs a and b byte-identical
expect_same() {
    local d
    d="$(differing "$2" "$3")"
    if [ -z "$d" ]; then pass "$1: every output byte-identical"
    else fail "$1:$d differ"; fi
}
# expect_moved <label> <a> <b>: at least one output of runs a and b differs
expect_moved() {
    local d
    d="$(differing "$2" "$3")"
    if [ -n "$d" ]; then pass "$1:$d differ"
    else fail "$1: every output byte-identical"; fi
}
# no_drops <label> <run>: the run left no share out of its words (no guard warning)
no_drops() {
    if grep -aq "share(s) of the exact fixed-point sums" "$TMP/$2.log"; then
        fail "$1: shares were left out ($(grep -a -o 'left out ([^)]*)' "$TMP/$2.log" | head -1))"
    else pass "$1: no share left out"; fi
}
# timing_words <run> <text>: whether every exact batch's timing line of the run (DTFE_GPU_TIMING=1) names the sums
# 'text' (and there is at least one)
timing_words() {
    local nx nw
    nx="$(grep -a -c 'DTFE Metal timing: exact kernel' "$TMP/$1.log")"
    nw="$(grep -a -c "$2" "$TMP/$1.log")"
    echo "$nw of $nx"
    [ "$nx" -gt 0 ] && [ "$nw" = "$nx" ]
}

echo ">> S2: the CPU exact average, --partition 1 1 1 / 2 2 2 / 3 3 3 (OMP_NUM_THREADS 1 / 4 / 6)"
run c_p1 OMP_NUM_THREADS=1 -- "${F[@]}" --partition 1 1 1
run c_p2 OMP_NUM_THREADS=4 DTFE_DEPOSIT_THREADS=1 -- "${F[@]}" --partition 2 2 2
run c_p3 OMP_NUM_THREADS=6 -- "${F[@]}" --partition 3 3 3
expect_same "S2 CPU, 2 2 2 (teams of 4) == 1 1 1" c_p1 c_p2
expect_same "S2 CPU, 3 3 3 (teams of 6) == 1 1 1" c_p1 c_p3
run c_d6 OMP_NUM_THREADS=4 DTFE_DEPOSIT_THREADS=6 -- "${F[@]}" --partition 2 2 2
expect_same "S2 CPU at 2 2 2, 6 deposit threads per sub-domain == 1" c_p2 c_d6

echo ">> S2: DTFE_TEST_RUN_UNIT_SHIFT=-20"
run c_m1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --partition 1 1 1
run c_m3 OMP_NUM_THREADS=6 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --partition 3 3 3
expect_same "S2 CPU at shift -20, 3 3 3 == 1 1 1" c_m1 c_m3
expect_moved "S2 the shift reaches the CPU words: shift -20 against shift 0 (1 1 1)" c_p1 c_m1

echo ">> S3: DTFE_FACE_CHECK=1 on the CPU (the deposit's gradient computed too, the face words replacing it)"
run c_k1 OMP_NUM_THREADS=1 DTFE_FACE_CHECK=1 -- "${F[@]}" --partition 1 1 1
run c_k3 OMP_NUM_THREADS=6 DTFE_FACE_CHECK=1 -- "${F[@]}" --partition 3 3 3
if grep -aq "FACE-CHECK" "$TMP/c_k1.log" && grep -aq "FACE-CHECK" "$TMP/c_k3.log"; then
    expect_same "S3 CPU with DTFE_FACE_CHECK=1, 3 3 3 == 1 1 1" c_k1 c_k3
    expect_same "S3 CPU, DTFE_FACE_CHECK=1 == the face default (1 1 1)" c_p1 c_k1
else fail "S3 CPU: DTFE_FACE_CHECK=1 printed no FACE-CHECK line (see $TMP/c_k1.log, $TMP/c_k3.log)"; fi

echo ">> S4: the parallel insert on the CPU (DTFE_TRIANGULATION_THREADS=6) against the sequential one"
run c_ti OMP_NUM_THREADS=6 DTFE_TRIANGULATION_THREADS=6 -- "${F[@]}" --partition 1 1 1 --max-concurrent 1
ins="$(grep -a -o '(parallel, TBB, [0-9]* threads)' "$TMP/c_ti.log" | head -1)"
if [ -n "$ins" ]; then
    expect_same "S4 CPU, the insert $ins == the sequential insert (1 1 1)" c_p1 c_ti
else
    note "S4 no '(parallel, TBB, N threads)' line in $TMP/c_ti.log (a build without TBB, or one free core)"
    expect_same "S4 CPU, 6 threads == 1 thread (both inserts sequential; 1 1 1)" c_p1 c_ti
fi

echo ">> O1, O2: shift +17 (the host carries) and +60 (the share cap) on the CPU, density_a velocity_a"
run c_o1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=17 -- "${FDV[@]}" --partition 1 1 1
run c_o3 OMP_NUM_THREADS=6 DTFE_TEST_RUN_UNIT_SHIFT=17 -- "${FDV[@]}" --partition 3 3 3
no_drops "O1 CPU at shift +17, 1 1 1" c_o1
no_drops "O1 CPU at shift +17, 3 3 3" c_o3
expect_same "O1 CPU at shift +17, 3 3 3 == 1 1 1" c_o1 c_o3
run c_x1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=60 -- "${FDV[@]}" --partition 1 1 1
run c_x3 OMP_NUM_THREADS=6 DTFE_TEST_RUN_UNIT_SHIFT=60 -- "${FDV[@]}" --partition 3 3 3
for r in c_x1 c_x3; do
    w="$(grep -a -o 'left out (density grid [0-9]*, velocity grid [0-9]*' "$TMP/$r.log" | head -1)"
    if [ -n "$w" ] && [ "$(grep -a -c 'share(s) of the exact fixed-point sums' "$TMP/$r.log")" = 1 ]; then
        pass "O2 $r warns once about its left-out shares: ${w#left out (}"
    else fail "O2 $r: no single warning naming left-out density and velocity shares (see $TMP/$r.log)"; fi
done
expect_same "O2 CPU at shift +60, 3 3 3 == 1 1 1" c_x1 c_x3

if [ "$GPU" = "1" ]; then
    echo ">> S1: the GPU exact average, --partition 1 1 1 / 2 2 2 / 3 3 3 (OMP_NUM_THREADS 1 / 4 / 6)"
    run g_p1 OMP_NUM_THREADS=1 DTFE_GPU_TIMING=1 -- "${F[@]}" --gpu --partition 1 1 1
    run g_p2 OMP_NUM_THREADS=4 -- "${F[@]}" --gpu --partition 2 2 2
    run g_p3 OMP_NUM_THREADS=6 DTFE_GPU_TIMING=1 -- "${F[@]}" --gpu --partition 3 3 3
    for r in g_p1 g_p3; do
        if t="$(timing_words "$r" 'run-wide 128-bit fixed-point sums')"; then
            pass "S1 $r: every exact batch sums in the run-wide 128-bit words ($t timing lines)"
        else fail "S1 $r: $t exact batch timing lines name the run-wide 128-bit words"; fi
    done
    expect_same "S1 GPU, 2 2 2 (teams of 4) == 1 1 1" g_p1 g_p2
    expect_same "S1 GPU, 3 3 3 (teams of 6) == 1 1 1" g_p1 g_p3

    echo ">> S1: DTFE_TEST_RUN_UNIT_SHIFT=-20"
    run g_m1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --gpu --partition 1 1 1
    run g_m3 OMP_NUM_THREADS=6 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --gpu --partition 3 3 3
    expect_same "S1 GPU at shift -20, 3 3 3 == 1 1 1" g_m1 g_m3
    expect_moved "S1 the shift reaches the GPU words: shift -20 against shift 0 (1 1 1)" g_p1 g_m1

    echo ">> S3: DTFE_FACE_CHECK=1 on the GPU; the face family of the GPU's 3 3 3 against the CPU's"
    run g_k1 OMP_NUM_THREADS=1 DTFE_FACE_CHECK=1 -- "${F[@]}" --gpu --partition 1 1 1
    run g_k3 OMP_NUM_THREADS=6 DTFE_FACE_CHECK=1 -- "${F[@]}" --gpu --partition 3 3 3
    if grep -aq "FACE-CHECK" "$TMP/g_k1.log" && grep -aq "FACE-CHECK" "$TMP/g_k3.log"; then
        expect_same "S3 GPU with DTFE_FACE_CHECK=1, 3 3 3 == 1 1 1" g_k1 g_k3
        expect_same "S3 GPU, DTFE_FACE_CHECK=1 == the face default (1 1 1)" g_p1 g_k1
    else fail "S3 GPU: DTFE_FACE_CHECK=1 printed no FACE-CHECK line (see $TMP/g_k1.log, $TMP/g_k3.log)"; fi
    d=""
    for e in .a_velDiv .a_velShear; do cmp -s "$TMP/c_p3$e" "$TMP/g_p3$e" || d="$d $e"; done
    if [ -z "$d" ]; then pass "S3 the GPU's 3 3 3 face family == the CPU's (byte-identical)"
    else fail "S3 the GPU's 3 3 3 face family differs from the CPU's:$d"; fi

    echo ">> S4: the parallel insert with --gpu (DTFE_TRIANGULATION_THREADS=6) against the sequential one"
    run g_ti OMP_NUM_THREADS=6 DTFE_TRIANGULATION_THREADS=6 -- "${F[@]}" --gpu --partition 1 1 1 --max-concurrent 1
    ins="$(grep -a -o '(parallel, TBB, [0-9]* threads)' "$TMP/g_ti.log" | head -1)"
    if [ -n "$ins" ]; then
        expect_same "S4 GPU, the insert $ins == the sequential insert (1 1 1)" g_p1 g_ti
    else
        note "S4 no '(parallel, TBB, N threads)' line in $TMP/g_ti.log (a build without TBB, or one free core)"
        expect_same "S4 GPU, 6 threads == 1 thread (both inserts sequential; 1 1 1)" g_p1 g_ti
    fi

    echo ">> O1: shift +40 on the GPU (device limbs 2 and 3), density_a velocity_a"
    run g_o1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=40 -- "${FDV[@]}" --gpu --partition 1 1 1
    run g_o3 OMP_NUM_THREADS=6 DTFE_TEST_RUN_UNIT_SHIFT=40 -- "${FDV[@]}" --gpu --partition 3 3 3
    no_drops "O1 GPU at shift +40, 1 1 1" g_o1
    no_drops "O1 GPU at shift +40, 3 3 3" g_o3
    expect_same "O1 GPU at shift +40, 3 3 3 == 1 1 1" g_o1 g_o3

    echo ">> O2 on the GPU: shift +60, one domain (the device's share cap, fixed_words.metal.inc shareLimbs)"
    run g_x1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=60 -- "${FDV[@]}" --gpu --partition 1 1 1
    w="$(grep -a -o 'left out (density grid [0-9]*, velocity grid [0-9]*' "$TMP/g_x1.log" | head -1)"
    if [ -n "$w" ] && [ "$(grep -a -c 'share(s) of the exact fixed-point sums' "$TMP/g_x1.log")" = 1 ]; then
        pass "O2 g_x1 warns once about its left-out shares: ${w#left out (}"
    else fail "O2 g_x1: no single warning naming left-out density and velocity shares (see $TMP/g_x1.log)"; fi
else
    echo "   (S1, the GPU halves of S3 and S4, and O1 and O2 on the GPU: not a GPU build)"
fi

echo ">> O1: the shifted runs against shift 0, each grid within 1e-6 of its rms"
pairs=(c_o1 c_p1)
[ "$GPU" = "1" ] && pairs+=(g_o1 g_p1)
"$PY" - "$TMP" "${DTFE_TEST_REAL}" "${pairs[@]}" <<'PY' || FAIL=1
import sys
import numpy as np
tmp, dt = sys.argv[1], sys.argv[2]
ok = True
for a, b in zip(sys.argv[3::2], sys.argv[4::2]):
    for e in ("a_den", "a_vel"):
        x = np.fromfile(tmp + "/" + a + "." + e, dtype=dt).astype(np.float64)
        y = np.fromfile(tmp + "/" + b + "." + e, dtype=dt).astype(np.float64)
        r = float(np.sqrt((y ** 2).mean())) if y.size else 0.
        q = float(np.abs(x - y).max()) / r if x.size == y.size and x.size and r > 0. else float("inf")
        good = q <= 1e-6
        print(("   PASS  " if good else "   FAIL  ")
              + "O1 %s against %s (shift 0), .%s: max difference %.2g of the rms (bound 1e-6)" % (a, b, e, q))
        ok = ok and good
sys.exit(0 if ok else 1)
PY

echo ">> U1: the units line of every run, per test shift"
for f in "$TMP"/u1/*; do
    [ -f "$f" ] || continue
    g="${f##*/}"
    runs="$(wc -l < "$f" | tr -d ' ')"
    bad="$(awk -F'|' '$2 != 1 { printf " %s (%s lines)", $1, $2 }' "$f")"
    kinds="$(cut -d'|' -f3- "$f" | sort -u | wc -l | tr -d ' ')"
    filled="$(cut -d'|' -f3- "$f" | grep -c -F '2^-K:')"
    if [ -z "$bad" ] && [ "$kinds" = 1 ] && [ "$filled" = "$runs" ]; then
        pass "U1 $g: one units line in each of its $runs runs, the same in all"
    else
        fail "U1 $g: ${bad:+runs without exactly one line:$bad; }$kinds different lines, $filled of $runs with units"
        cut -d'|' -f1,3- "$f" | sed 's/^/      /'
    fi
done
out="$(grep -a -l -F 'lies outside 2^-120' "$TMP"/*.log)"
if [ -z "$out" ]; then pass "U1 no run has a grid without its run-wide unit (K outside -120 .. 120)"
else fail "U1 a grid fell back to its per-batch unit (K outside -120 .. 120) in:$(echo "$out" | sed 's|.*/| |')"; fi
grep -h -o 'fixed-point units (run-wide): cell volume.*' "$TMP/c_p1.log" | head -1 | sed 's/^/   (shift 0: /; s/$/)/'

if [ "$FAIL" = "0" ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; exit 1; fi
