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
#       state of DTFE_RUN_UNITS_LEGACY, whatever its split, threads or device (per estimator and, in PS-DTFE, per
#       flag family: the exact and the sampled runs below differ in the moment weight's unit and the share cap)
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
# PS-DTFE (its own section before the result; the crossed waves of tests/generate_ps_test_data.py, --n 64 --box 100
# --amplitude-factor 1.8, at 64^3), under DTFE_PS_RUN_UNITS=1 (by default its passes take units of their own; the
# section sets the switch for its runs up to P4, the word merge's below need none): every pass sums in the same
# run-wide units, on the CPU and the GPU alike
#   PU1 '--exact --ps-volume-weighted --ps-caustics', density_a velocity_a dispersion_a divergence_a shear_a
#       '--face-gradient', '--partition 1 1 1', '2 2 2' and '3 3 3', on the CPU and (a GPU build) on the GPU: each
#       run prints the same one units line (U1)
#   PO1 shift +20 (not +40: the PS share cap of 2^(94 - ceil(log2 nSub^3)) units lies only ~2^41.5 S_g above the
#       unit at nSub 3, so at +40 halo and caustic shares would be left out; derived, not measured), density_a
#       velocity_a dispersion_a, one domain, CPU and GPU: no share left out; '.a_den' and '.a_vel' within 1e-6 of
#       their rms of the shift-0 run
#   PO2 shift +60, the same fields: each run warns once, naming its left-out velocity moment and mass (density)
#       shares
#   P4  the sampled deposit ('--sampled --subsamples 2'), one domain on the CPU: every output of a '--grid-window'
#       run byte-identical to the full run's window cells (the window run sums in the full run's units)
# PS-DTFE's byte identity across splits, under THE WORD MERGE (the end of its section; '--ps-merge fixed-point' in the
# single build, fix 3 S6: every pass hands its exact words over, the merge adds them as integers and ONE close-out
# converts and normalizes, a one-domain run's too). Its runs set no DTFE_PS_RUN_UNITS: the option switches the
# run-wide units on by itself. The CPU and the GPU runs are never compared with each other (their arithmetic differs):
#   PW  the units line of a word-merge run is PU1's with the note "; on for '--ps-merge fixed-point' (its merge adds
#       the passes' words)" (the option takes the switch's units), and the logs name the word merge (its setup line,
#       the merge's line and report, the one-domain close-out's statistics line) and no merge of 9 B per component
#   P1  the exact CPU deposit with PU1's fields: '--partition 1 1 1' (OMP_NUM_THREADS=1), '2 2 2' (4), '3 3 3' (6),
#       '3 3 3' at '--max-concurrent 1' and at 4 with partition 0 held back 2 s (another completion order), one domain
#       on 4 deposit threads in the copy mode and in the slab mode (DTFE_DEPOSIT_SLABS=0 / 1), and no '--partition'
#       (the auto-tuner's own split, which follows the machine): every output byte-identical to '1 1 1'
#   P2  (a GPU build) the same on the GPU, '1 1 1', '2 2 2', '3 3 3' and '3 3 3' at '--max-concurrent 1': every output
#       byte-identical to the GPU's '1 1 1'
#   P3  the sampled deposit ('--sampled --subsamples 2 --ps-vertex-mass', no face sums: they need the exact one):
#       '1 1 1', '2 2 2' and '3 3 3' on the CPU, '1 1 1' and '3 3 3' on the GPU: byte-identical per device
#   P5  P1's '3 3 3' killed after 5 merges (DTFE_TEST_EXIT_AFTER_PARTITIONS=5, 4 in flight), then '--resume' at
#       '--max-concurrent 2': the checkpoint's words continued, every output byte-identical to '1 1 1'
#   M1  (a GPU build; REPORT only) '--gpu-share 0.5' at '2 2 2' and '3 3 3': the merge adds exactly, but the devices
#       form their shares in different arithmetic and which one deposits a partition follows the split, so a mixed
#       run's bytes may differ between the splits and from the all-GPU run (the option's help says so); NOTE lines
# The double build keeps its double CPU sums, its double vertex star sums and CGAL's single-cell volumes, whose last
# bits follow the split (and it refuses '--ps-merge fixed-point'): SKIP there. 63 runs (36 on the CPU, 27 more on the
# GPU in a GPU build).
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
      DTFE_GPU_ITEMS DTFE_GPU_FAULT_AT DTFE_GPU_TIMING OMP_NUM_THREADS DTFE_PS_RUN_UNITS
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
    local e a ushift=0 ulegacy=0 gpu=0 n line fallback
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
    # (the reason's text of the standard estimator, then PS-DTFE's; the closing count of either: only the master
    # thread's partitions print a reason)
    fallback="using the CPU interpolation\|ran on the CPU instead"
    [ "$EST" != std ] && fallback="using the CPU deposit\|deposit unavailable\|ran on the CPU instead\|gave up"
    if [ "$gpu" = 1 ] && grep -aq "$fallback" "$TMP/$name.log"; then
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

# PS-DTFE, under DTFE_PS_RUN_UNITS=1 (exported below; by default its passes take units of their own): its passes sum
# in the same run-wide units (EST=ps*, BIN="${PS_BIN}", the crossed waves as DATA, their own COMMON; run, differing,
# expect_same and U1 take them as they are). Each case makes its own runs. The groups of U1 are per flag family:
# '--ps-volume-weighted' sets the moment weight's unit and '--subsamples' the share cap, so the exact and the sampled
# runs print two different units lines.
PS_DATA="$TMP/ps_waves.hdf5"
if [ ! -x "${PS_BIN}" ]; then
    echo ">> PS-DTFE: SKIP -- ${PS_BIN} is not built"
elif ! "$PY" tests/generate_ps_test_data.py --out "$PS_DATA" --n 64 --box 100 --amplitude-factor 1.8 --crossed-waves \
        > "$TMP/ps_waves.log" 2>&1; then
    fail "PS-DTFE: the crossed waves were not generated (see $TMP/ps_waves.log)"
else
    GPU_PS=0
    [ -n "$(gpu_backend_of o_ps)" ] && GPU_PS=1
    BIN="${PS_BIN}"; DATA="$PS_DATA"
    export DTFE_PS_RUN_UNITS=1   # PS-DTFE's switch for the run-wide units (no standard run follows; it reads none)
    COMMON=(--grid 64 --periodic --mpc-unit 1 --exact --ps-volume-weighted --ps-caustics --verbose 3)
    PF=(--field density_a velocity_a dispersion_a divergence_a shear_a --face-gradient)
    # the limbs, carries and cap without the face sums: at these shifts every face flux would take a carry entry
    PFDV=(--field density_a velocity_a dispersion_a)

    EST=ps
    echo ">> PU1: the PS-DTFE exact deposit on the CPU, --partition 1 1 1 / 2 2 2 / 3 3 3 (OMP_NUM_THREADS 1 / 4 / 6)"
    run pc_p1 OMP_NUM_THREADS=1 -- "${PF[@]}" --partition 1 1 1
    run pc_p2 OMP_NUM_THREADS=4 -- "${PF[@]}" --partition 2 2 2
    run pc_p3 OMP_NUM_THREADS=6 -- "${PF[@]}" --partition 3 3 3
    if [ "$GPU_PS" = "1" ]; then
        echo ">> PU1: the same on the GPU"
        run pg_p1 OMP_NUM_THREADS=1 -- "${PF[@]}" --gpu --partition 1 1 1
        run pg_p2 OMP_NUM_THREADS=4 -- "${PF[@]}" --gpu --partition 2 2 2
        run pg_p3 OMP_NUM_THREADS=6 -- "${PF[@]}" --gpu --partition 3 3 3
    fi
    echo "   (PS-DTFE P1-P3, the byte identity across splits, P5 and M1: under the word merge, after P4)"

    # PO1: shift +20, not +40: the PS share cap 2^(94 - ceil(log2 nSub^3)) units lies only ~2^41.5 S_g above the
    # unit at nSub 3 (derived from the cap and the units' table, not measured), so at +40 the halo and caustic
    # shares would be left out; at +20 a cell of the mean density holds 2^67..2^68 units, beyond the 64-bit words
    # (the device's limb 2, the host's carry lists)
    echo ">> PO1: shift +20 (device limb 2, the host carries) against shift 0, one domain, ${PFDV[*]}"
    run pc_d0 OMP_NUM_THREADS=1 -- "${PFDV[@]}" --partition 1 1 1
    run pc_o1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=20 -- "${PFDV[@]}" --partition 1 1 1
    no_drops "PO1 CPU at shift +20" pc_o1
    ps_pairs=(pc_o1 pc_d0)
    if [ "$GPU_PS" = "1" ]; then
        run pg_d0 OMP_NUM_THREADS=1 -- "${PFDV[@]}" --gpu --partition 1 1 1
        run pg_o1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=20 -- "${PFDV[@]}" --gpu --partition 1 1 1
        no_drops "PO1 GPU at shift +20" pg_o1
        ps_pairs+=(pg_o1 pg_d0)
    fi
    "$PY" - "$TMP" "${DTFE_TEST_REAL}" "${ps_pairs[@]}" <<'PY' || FAIL=1
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
              + "PO1 %s against %s (shift 0), .%s: max difference %.2g of the rms (bound 1e-6)" % (a, b, e, q))
        ok = ok and good
sys.exit(0 if ok else 1)
PY

    echo ">> PO2: shift +60 (the share cap), one domain, ${PFDV[*]}"
    po2=(pc_x1)
    run pc_x1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=60 -- "${PFDV[@]}" --partition 1 1 1
    if [ "$GPU_PS" = "1" ]; then
        run pg_x1 OMP_NUM_THREADS=1 DTFE_TEST_RUN_UNIT_SHIFT=60 -- "${PFDV[@]}" --gpu --partition 1 1 1
        po2+=(pg_x1)
    fi
    for r in "${po2[@]}"; do
        # (the slot names hold parentheses: 'mass (density) grid N'; the moment slots come first)
        if [ "$(grep -a -c 'share(s) of the exact fixed-point sums' "$TMP/$r.log")" = 1 ] \
           && grep -aq 'velocity moment grid [0-9]' "$TMP/$r.log" \
           && grep -aq 'mass (density) grid [0-9]' "$TMP/$r.log"; then
            pass "PO2 $r warns once, naming its left-out velocity moment and mass (density) shares"
        else
            fail "PO2 $r: no single warning naming left-out velocity moment and mass (density) shares" \
                 "(see $TMP/$r.log)"
        fi
    done

    # P4: the sampled window == the full run's window cells, every output byte for byte (one device, the CPU): the
    # window run sums in the full run's units (ps_window_check.sh holds the other window cases). The window: cells
    # [8,24) x [40,52) x [30,31) of the 64^3 grid of the 100 Mpc box (ps_window_check.sh window A).
    EST=ps_sampled
    COMMON=(--grid 64 --periodic --mpc-unit 1 --sampled --subsamples 2 --ps-caustics --verbose 3)
    echo ">> P4: the sampled deposit, a --grid-window run against the full run (CPU, one domain)"
    run ps_full OMP_NUM_THREADS=4 -- "${PFDV[@]}" --partition 1 1 1
    run ps_win OMP_NUM_THREADS=4 -- "${PFDV[@]}" --partition 1 1 1 --grid-window 12.5 37.5 62.5 81.25 46.875 48.4375
    "$PY" - "$TMP" ps_full ps_win $(outputs ps_win) <<'PY' || FAIL=1
import os, sys
import numpy as np
tmp, full, win, sufs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
G, LO, DIMS = 64, (8, 40, 30), (16, 12, 1)
nw = DIMS[0] * DIMS[1] * DIMS[2]
ok, n = True, 0
for s in sufs:
    pf, pw = tmp + "/" + full + s, tmp + "/" + win + s
    if not os.path.isfile(pf):
        print("   FAIL  P4 %s: the full run wrote no such output" % s); ok = False; continue
    sf, sw = os.path.getsize(pf), os.path.getsize(pw)
    if sw % (4 * nw) or sf % (4 * G ** 3) or sw // (4 * nw) != sf // (4 * G ** 3):
        print("   NOTE  P4 %s: not a grid of the window (%d B) and the full run (%d B); not compared" % (s, sw, sf))
        continue
    nc = sw // (4 * nw)
    a = np.fromfile(pf, dtype="<u4").reshape((G, G, G, nc))
    b = np.fromfile(pw, dtype="<u4").reshape(DIMS + (nc,))
    idx = [np.arange(LO[d], LO[d] + DIMS[d]) % G for d in range(3)]
    same = np.array_equal(a[np.ix_(*idx)], b)
    n += 1
    print(("   PASS  " if same else "   FAIL  ") + "P4 %s: the window cells %s" % (s, "byte-identical" if same else
          "differ in %d of %d words" % (int((a[np.ix_(*idx)] != b).sum()), b.size)))
    ok = ok and same
if n == 0:
    print("   FAIL  P4: no grid compared"); ok = False
sys.exit(0 if ok else 1)
PY

    # THE WORD MERGE ('--ps-merge fixed-point', fix 3 S6): PW, P1-P3, P5 and M1 (the header). No DTFE_PS_RUN_UNITS:
    # the option switches the run-wide units on by itself. What else could steer these runs is pinned here: the float
    # knobs (refused beside the option), a pinned slab mode, a held-back or killed partition, no checkpoints, CGAL's
    # vertex numbering (the identity needs the canonical order)
    unset DTFE_PS_RUN_UNITS DTFE_GPU_FLOAT_MASS DTFE_GPU_FLOAT_MOMENTS DTFE_DEPOSIT_SLABS DTFE_TEST_DELAY_PARTITION \
          DTFE_TEST_EXIT_AFTER_PARTITIONS DTFE_CHECKPOINT DTFE_PS_VERTEX_ORDER_LEGACY
    EST=ps_wm
    COMMON=(--grid 64 --periodic --mpc-unit 1 --exact --ps-volume-weighted --ps-caustics --verbose 3
            --ps-merge fixed-point)
    echo ">> P1: the word merge on the CPU (exact), --partition 1 1 1 / 2 2 2 / 3 3 3, --max-concurrent 1 / 4, the" \
         "copy and the slab mode, the auto-tuner's split"
    run wc_p1 OMP_NUM_THREADS=1 -- "${PF[@]}" --partition 1 1 1
    run wc_p2 OMP_NUM_THREADS=4 -- "${PF[@]}" --partition 2 2 2
    run wc_p3 OMP_NUM_THREADS=6 -- "${PF[@]}" --partition 3 3 3
    run wc_m1 OMP_NUM_THREADS=6 -- "${PF[@]}" --partition 3 3 3 --max-concurrent 1
    run wc_m4 OMP_NUM_THREADS=6 DTFE_TEST_DELAY_PARTITION=0:2 -- "${PF[@]}" --partition 3 3 3 --max-concurrent 4
    run wc_cp OMP_NUM_THREADS=4 DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=0 -- "${PF[@]}" --partition 1 1 1
    run wc_sl OMP_NUM_THREADS=4 DTFE_DEPOSIT_THREADS=4 DTFE_DEPOSIT_SLABS=1 -- "${PF[@]}" --partition 1 1 1
    run wc_at OMP_NUM_THREADS=4 -- "${PF[@]}"
    # PW: the option's units are the switch's (PU1's pc_p1: the same flags under DTFE_PS_RUN_UNITS=1), with its note
    note_wm="; on for '--ps-merge fixed-point' (its merge adds the passes' words)"
    lp="$(grep -a -o 'fixed-point units (run-wide): cell volume.*' "$TMP/pc_p1.log" | head -1)"
    lw="$(grep -a -o 'fixed-point units (run-wide): cell volume.*' "$TMP/wc_p1.log" | head -1)"
    if [ -n "$lp" ] && [ "$lw" = "$lp$note_wm" ]; then
        pass "PW the word merge's units line is PU1's (DTFE_PS_RUN_UNITS=1) with the option's note"
    else
        fail "PW the units lines differ: PU1 [${lp:-none}], the word merge [${lw:-none}]"
    fi
    if grep -aqF -- "--ps-merge fixed-point: the passes hand over their exact words" "$TMP/wc_p3.log" \
       && grep -aqF "each partition's exact words are added as it finishes" "$TMP/wc_p3.log" \
       && grep -aqF "(the word merge: 8 B per summed component" "$TMP/wc_p3.log" \
       && grep -aqF "the one domain's grid after the close-out" "$TMP/wc_p1.log" \
       && ! grep -aqF "9 B per summed component" "$TMP/wc_p3.log"; then
        pass "PW the logs name the word merge (its setup, its merge and report, the one-domain close-out)"
    else
        fail "PW the logs do not name the word merge (see $TMP/wc_p3.log and $TMP/wc_p1.log)"
    fi
    expect_same "P1 CPU, 2 2 2 == 1 1 1" wc_p1 wc_p2
    expect_same "P1 CPU, 3 3 3 == 1 1 1" wc_p1 wc_p3
    expect_same "P1 CPU, 3 3 3 at --max-concurrent 1 == 1 1 1" wc_p1 wc_m1
    expect_same "P1 CPU, 3 3 3 at --max-concurrent 4, partition 0 held back == 1 1 1" wc_p1 wc_m4
    expect_same "P1 CPU, one domain on 4 deposit threads in the copy mode == on 1" wc_p1 wc_cp
    expect_same "P1 CPU, one domain on 4 deposit threads in the slab mode == on 1" wc_p1 wc_sl
    sp="$(grep -a -m1 'Lagrangian-space partitioning with' "$TMP/wc_at.log" | sed 's/.*partitions \[//; s/\].*//')"
    expect_same "P1 CPU, the auto-tuner's split (${sp:-one domain}) == 1 1 1" wc_p1 wc_at
    for r in wc_cp wc_sl; do
        m="$(grep -a -o -e 'CPU deposit in SLAB mode: [0-9]* threads' -e 'CPU deposit on [0-9]* threads* (copy mode' \
             "$TMP/$r.log" | head -1)"
        note "P1 $r: ${m:-no deposit-mode line in its log}"
    done

    echo ">> P5: P1's 3 3 3 killed after 5 merges (4 in flight), then '--resume' at --max-concurrent 2"
    rm -rf "$TMP/wc_k.runlog.checkpoint"
    env OMP_NUM_THREADS=6 DTFE_TEST_EXIT_AFTER_PARTITIONS=5 "$BIN" "$DATA" "$TMP/wc_k" "${COMMON[@]}" "${PF[@]}" \
        --partition 3 3 3 --max-concurrent 4 > "$TMP/wc_k_killed.log" 2>&1
    rc=$?
    nk="$(merged_count "$TMP/wc_k.runlog.checkpoint" 2>/dev/null || echo 0)"
    if [ "$rc" = 137 ] && [ "${nk:-0}" -ge 5 ] && [ "${nk:-0}" -lt 27 ]; then
        pass "P5 killed (exit $rc) after $nk of 27 merges, its checkpoint left"
        run wc_k OMP_NUM_THREADS=6 -- "${PF[@]}" --partition 3 3 3 --max-concurrent 2 --resume
        if grep -aqF -- "--resume: $nk of 27 partitions are already in the checkpoint" "$TMP/wc_k.log" \
           && grep -aqF "words added to the checkpoint's as it finishes" "$TMP/wc_k.log"; then
            pass "P5 the resume continued the word merge's checkpoint ($nk partitions' words)"
        else
            fail "P5 the resume's log does not say it continued the word merge's checkpoint (see $TMP/wc_k.log)"
        fi
        expect_same "P5 the resumed 3 3 3 == 1 1 1" wc_p1 wc_k
    else
        fail "P5 the 3 3 3 run was not killed after a checkpoint (exit $rc, ${nk:-no} merged;" \
             "see $TMP/wc_k_killed.log)"
        no_checkpoint_why "$TMP/wc_k_killed.log" "$TMP"
    fi

    if [ "$GPU_PS" = "1" ]; then
        echo ">> P2: the word merge on the GPU (exact), --partition 1 1 1 / 2 2 2 / 3 3 3, --max-concurrent 1"
        run gc_p1 OMP_NUM_THREADS=1 -- "${PF[@]}" --gpu --partition 1 1 1
        run gc_p2 OMP_NUM_THREADS=4 -- "${PF[@]}" --gpu --partition 2 2 2
        run gc_p3 OMP_NUM_THREADS=6 -- "${PF[@]}" --gpu --partition 3 3 3
        run gc_m1 OMP_NUM_THREADS=6 -- "${PF[@]}" --gpu --partition 3 3 3 --max-concurrent 1
        expect_same "P2 GPU, 2 2 2 == 1 1 1" gc_p1 gc_p2
        expect_same "P2 GPU, 3 3 3 == 1 1 1" gc_p1 gc_p3
        expect_same "P2 GPU, 3 3 3 at --max-concurrent 1 == 1 1 1" gc_p1 gc_m1

        # M1: no gate. The devices form their shares in different arithmetic, and which one deposits a partition
        # follows the split (the share deals partitions by index), so these bytes may move with the split
        echo ">> M1 (REPORT only): a mixed run, '--gpu-share 0.5', at 2 2 2 and 3 3 3"
        run mx_p2 OMP_NUM_THREADS=6 -- "${PF[@]}" --gpu --gpu-share 0.5 --partition 2 2 2
        run mx_p3 OMP_NUM_THREADS=6 -- "${PF[@]}" --gpu --gpu-share 0.5 --partition 3 3 3
        for r in mx_p2 mx_p3; do
            s="$(grep -a -o 'GPU share 0.50 (--gpu-share 0.5): [0-9]* of the [0-9]* partitions' "$TMP/$r.log" \
                 | head -1)"
            note "M1 $r: ${s:-no GPU share line in its log}"
        done
        d="$(differing mx_p2 mx_p3)"
        note "M1 the mixed 3 3 3 against the mixed 2 2 2:${d:- every output byte-identical}"
        d="$(differing gc_p1 mx_p3)"
        note "M1 the mixed 3 3 3 against the all-GPU 1 1 1:${d:- every output byte-identical}"
    fi

    EST=ps_wm_sampled
    COMMON=(--grid 64 --periodic --mpc-unit 1 --sampled --subsamples 2 --ps-vertex-mass --ps-volume-weighted
            --ps-caustics --verbose 3 --ps-merge fixed-point)
    PS3=(--field density_a velocity_a dispersion_a divergence_a shear_a)
    echo ">> P3: the word merge, the sampled deposit (--ps-vertex-mass), --partition 1 1 1 / 2 2 2 / 3 3 3 on the CPU"
    run sc_p1 OMP_NUM_THREADS=1 -- "${PS3[@]}" --partition 1 1 1
    run sc_p2 OMP_NUM_THREADS=4 -- "${PS3[@]}" --partition 2 2 2
    run sc_p3 OMP_NUM_THREADS=6 -- "${PS3[@]}" --partition 3 3 3
    expect_same "P3 CPU sampled, 2 2 2 == 1 1 1" sc_p1 sc_p2
    expect_same "P3 CPU sampled, 3 3 3 == 1 1 1" sc_p1 sc_p3
    if [ "$GPU_PS" = "1" ]; then
        echo ">> P3: the same on the GPU, --partition 1 1 1 / 3 3 3"
        run sg_p1 OMP_NUM_THREADS=1 -- "${PS3[@]}" --gpu --partition 1 1 1
        run sg_p3 OMP_NUM_THREADS=6 -- "${PS3[@]}" --gpu --partition 3 3 3
        expect_same "P3 GPU sampled, 3 3 3 == 1 1 1" sg_p1 sg_p3
    else
        echo "   (P2, M1 and the GPU half of P3: not a GPU build)"
    fi
fi

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
