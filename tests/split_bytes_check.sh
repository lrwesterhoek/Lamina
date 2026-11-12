#!/usr/bin/env bash
# Output bytes that do not depend on the split: the run-wide fixed-point units (src/fixed_units.h RunUnits). Every
# exact '_a' grid of an all-GPU or all-CPU standard-DTFE run sums its shares in ONE unit per grid for the whole run,
# set before any partition or thread sub-domain exists (DTFE.cpp runUnitsSetup), in words no order of the adds can
# wrap -- the GPU's four 32-bit limbs (metal/fixed_words.metal.inc), the CPU's and the face pass's 64-bit words with
# an exact carry (FxWords) -- from inputs that follow neither CGAL's cell order nor its labelling (the exact vertex
# star sums, triangulation.cpp vertexDensityStarSums; the single-cell volumes, canonicalTetVolume). So the partition
# split, the thread sub-domain split, the deposit's thread count and the parallel insert's order reach no output
# byte. Every comparison is cmp of every output file. demo/z0_64.gadget (tracked in the repo) at 64^3, --periodic
# --padding 12 (enough for every split to hold the same tetrahedra), '--exact', density_a velocity_a divergence_a
# shear_a (the gradient family from the face pass, the default), single precision:
#   U1  every run prints ONE 'fixed-point units (run-wide):' line, the same in every run of one test shift and one
#       state of DTFE_RUN_UNITS_LEGACY, whatever its split, threads or device
#   S1  (a GPU build) '--gpu' at '--partition 1 1 1' (OMP_NUM_THREADS=1: one sub-domain), '2 2 2' (4: a team of 4
#       sub-domains in each partition) and '3 3 3' (6): every output byte-identical, and every batch's timing line
#       says 'run-wide 128-bit fixed-point sums'; with DTFE_TEST_RUN_UNIT_SHIFT=-20 (every unit 2^20 coarser, so
#       nearly every share rounds) '1 1 1' and '3 3 3' byte-identical too, and different from the shift-0 run (the
#       hook reaches the words). The discriminator: the same -20 pair under DTFE_RUN_UNITS_LEGACY=1 (a unit per GPU
#       batch, shifted alike; its timing lines say 'fixed-point sums' without 'run-wide') must DIFFER -- the proof
#       that this pair can fail
#   S2  the same without '--gpu' (the CPU exact average in the run-wide words), and DTFE_DEPOSIT_THREADS 1 against
#       6 at '2 2 2': every output byte-identical. The CPU pair under the switch (float sums in the tessellation's
#       walk order, double face sums) is a NOTE: whether 64^3 shows the walk order there is no gate
#   S3  DTFE_FACE_CHECK=1 (the deposits compute the per-tetrahedron gradient as well; the face words replace it, and
#       the FACE-CHECK lines compare the two) at '1 1 1' and '3 3 3' on each device: byte-identical to each other and
#       to the run without it; and the GPU's '3 3 3' face family ('.a_velDiv', '.a_velShear') == the CPU's (the face
#       pass reads the tessellation only, whatever the deposit)
#   S4  the parallel insert: DTFE_TRIANGULATION_THREADS=6 (OMP_NUM_THREADS=6, '--partition 1 1 1 --max-concurrent 1':
#       one tessellation on up to 6 TBB threads, its cell order and labelling other than the sequential insert's)
#       against S1's / S2's sequential '1 1 1' run: every output byte-identical. Without a '(parallel, TBB, N
#       threads)' line (a build without TBB, or one free core) the pair compares 6 threads against 1 only
#   O1  limbs and carries, density_a velocity_a (no face pass: at these shifts every face flux would take a carry
#       entry): shift +40 on the GPU (a cell of the mean density holds 2^87..2^88 units: device limb 2; limb 3 in
#       cells of a few hundred times the mean), +17 on the CPU (a cell of the mean density holds 2^64..2^65 units:
#       the words wrap, the carry lists take the crossings and the high parts of shares of 2^63 units or more; at
#       +40 every CPU share would take an entry). No share left out (each density share lies below 4.01 cell
#       volumes, 2^90 units at +40, the cap 2^94); '1 1 1' and '3 3 3' byte-identical; each grid within 1e-6 of
#       its rms of the shift-0 run
#   O2  the share cap: shift +60, density_a velocity_a, puts nearly every share at or above the cap of 2^94 units:
#       each run warns once, naming its left-out density and velocity shares; on the CPU '1 1 1' and '3 3 3'
#       byte-identical; on the GPU (a GPU build) one '1 1 1' run, for the device's cap compare
# Not here: PS-DTFE (the design's P1-P5 and M1): its passes take the run-wide units in the design's later steps, and
# its split identity needs the canonical vertex order; the section before the result is where they go.
# The double build keeps its double CPU sums, its double vertex star sums and CGAL's single-cell volumes, whose last
# bits follow the split: SKIP there. 28 runs (15 on the CPU, 13 more on the GPU in a GPU build).
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
unset DTFE_RUN_UNITS_LEGACY DTFE_TEST_RUN_UNIT_SHIFT DTFE_FACE_CHECK DTFE_FACE_GRADIENT DTFE_TRIANGULATION_THREADS \
      DTFE_TRIANGULATION_MIN_POINTS DTFE_DEPOSIT_THREADS DTFE_MC_LEGACY DTFE_GPU_FLOAT_GRIDS DTFE_GPU_SHARED \
      DTFE_GPU_ITEMS DTFE_GPU_FAULT_AT DTFE_GPU_TIMING OMP_NUM_THREADS
FAIL=0
pass() { echo "   PASS  $*"; }
fail() { echo "   FAIL  $*"; FAIL=1; }
note() { echo "   NOTE  $*"; }

# The estimator under test: its tag (U1's groups), binary, input and options. The PS-DTFE cases set their own.
EST=std; BIN="${DTFE_BIN}"; DATA="$SNAP"
COMMON=(--grid 64 --periodic --padding 12 --exact --run-mode performance --verbose 3)
F=(--field density_a velocity_a divergence_a shear_a)
FDV=(--field density_a velocity_a)

# run <name> [VAR=value ...] -- [args ...]: BIN on DATA with COMMON and the args, its log in <name>.log. A failed run,
# or a '--gpu' run that fell back to the CPU, stops the suite. Its units line is filed for U1 under its estimator,
# its test shift and its switch (read from the assignments).
run() {
    local name="$1"; shift
    local envs=()
    local e a ushift=0 ulegacy=0 gpu=0 n line
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    for e in ${envs[@]+"${envs[@]}"}; do
        case "$e" in
            DTFE_TEST_RUN_UNIT_SHIFT=*) ushift="${e#*=}" ;;
            DTFE_RUN_UNITS_LEGACY=*) ulegacy="${e#*=}" ;;
        esac
    done
    for a in "$@"; do [ "$a" = "--gpu" ] && gpu=1; done
    env ${envs[@]+"${envs[@]}"} "$BIN" "$DATA" "$TMP/$name" "${COMMON[@]}" "$@" > "$TMP/$name.log" 2>&1 \
        || { echo "FAIL: run $name (see $TMP/$name.log)"; exit 1; }
    if [ "$gpu" = 1 ] && grep -aq "using the CPU interpolation" "$TMP/$name.log"; then
        echo "FAIL: the --gpu run $name fell back to the CPU (see $TMP/$name.log)"; exit 1
    fi
    n="$(grep -a -c 'fixed-point units (run-wide): cell volume' "$TMP/$name.log")"
    line="$(grep -a -o 'fixed-point units (run-wide): cell volume.*' "$TMP/$name.log" | head -1)"
    echo "$name|$n|$line" >> "$TMP/u1/$EST.shift$ushift.legacy$ulegacy"
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

echo ">> S2: DTFE_TEST_RUN_UNIT_SHIFT=-20, and the same pair under DTFE_RUN_UNITS_LEGACY=1"
run c_m1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --partition 1 1 1
run c_m3 OMP_NUM_THREADS=6 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --partition 3 3 3
expect_same "S2 CPU at shift -20, 3 3 3 == 1 1 1" c_m1 c_m3
expect_moved "S2 the shift reaches the CPU words: shift -20 against shift 0 (1 1 1)" c_p1 c_m1
run c_l1 OMP_NUM_THREADS=1 DTFE_RUN_UNITS_LEGACY=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --partition 1 1 1
run c_l3 OMP_NUM_THREADS=6 DTFE_RUN_UNITS_LEGACY=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --partition 3 3 3
d="$(differing c_l1 c_l3)"
if [ -n "$d" ]; then note "S2 under DTFE_RUN_UNITS_LEGACY=1 the CPU's 3 3 3 differs from its 1 1 1:$d"
else note "S2 under DTFE_RUN_UNITS_LEGACY=1 the CPU's 3 3 3 equals its 1 1 1 at 64^3 (walk-order sums; no gate)"; fi

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

    echo ">> S1: DTFE_TEST_RUN_UNIT_SHIFT=-20, and the same pair under DTFE_RUN_UNITS_LEGACY=1 (must differ)"
    run g_m1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --gpu --partition 1 1 1
    run g_m3 OMP_NUM_THREADS=6 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --gpu --partition 3 3 3
    expect_same "S1 GPU at shift -20, 3 3 3 == 1 1 1" g_m1 g_m3
    expect_moved "S1 the shift reaches the GPU words: shift -20 against shift 0 (1 1 1)" g_p1 g_m1
    run g_l1 OMP_NUM_THREADS=1 DTFE_RUN_UNITS_LEGACY=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 DTFE_GPU_TIMING=1 -- \
        "${F[@]}" --gpu --partition 1 1 1
    run g_l3 OMP_NUM_THREADS=6 DTFE_RUN_UNITS_LEGACY=1 DTFE_TEST_RUN_UNIT_SHIFT=-20 -- "${F[@]}" --gpu --partition 3 3 3
    if t="$(timing_words g_l1 'fixed-point sums')" && ! grep -aq "run-wide 128-bit" "$TMP/g_l1.log"; then
        pass "S1 DTFE_RUN_UNITS_LEGACY=1: the batches sum in their own units' 64-bit words ($t timing lines)"
    else
        fail "S1 DTFE_RUN_UNITS_LEGACY=1: the timing lines of $TMP/g_l1.log name other sums ($t say 'fixed-point sums')"
    fi
    expect_moved "S1 discriminator: under DTFE_RUN_UNITS_LEGACY=1 at shift -20 the GPU's 3 3 3 differs from 1 1 1" \
        g_l1 g_l3

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

# PS-DTFE: the design's P1-P5 (CPU and GPU exact, sampled, the window, resume) and M1 (a mixed run, reported only) go
# here, with EST=ps, BIN="${PS_BIN}", the crossed waves as DATA (tests/generate_ps_test_data.py --n 64 --box 100
# --amplitude-factor 1.8 --crossed-waves) and their own COMMON; run, differing, expect_same and U1 take them as they
# are. They wait for the PS-DTFE steps of the run-wide units and for the canonical vertex order.
echo ">> PS-DTFE (P1-P5, M1): SKIP -- its passes do not take the run-wide units (the design's steps S5, S6)"

echo ">> U1: the units line of every run, per test shift and switch"
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
