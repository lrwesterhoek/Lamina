#!/usr/bin/env bash
# The run modes (docs/cleanup-plan.md P13): '--run-mode gentle|balanced|performance' in both binaries (or the
# environment's DTFE_RUN_MODE), PS-DTFE's '--gpu-share auto|<fraction>', and the run log's lines that say what was set
# and why (src/run_mode.h; src/user_options.cc printOptions). On a synthetic 16^3 snapshot of crossed waves, grid 16.
# The thread budgets expected here are src/run_mode.h runModeCoreBudget recomputed from this machine: N0 = its
# threads (nproc on Linux, hw.logicalcpu on macOS), perf = its performance cores (hw.perflevel0.logicalcpu on macOS,
# else none), base = min(N0, perf) where perf is known, else N0; gentle max(1, base/2), balanced
# max(1, base - clamp(round(base/4), 1, 4)), performance N0. The header line checked is
#   '\t run mode               : <mode> (<source>; <threads>[; GPU buffers up to <250|50> ms][; <priority>])'.
# CPU block (every build, both precisions):
#   M1  no flag, with DTFE_RUN_MODE and OMP_NUM_THREADS unset, both binaries: 'performance (--run-mode performance,
#       the default; N0 of N0 threads)'
#   M2  '--run-mode balanced' and '--run-mode gentle', both binaries: 'B of N0 threads' and 'G of N0 threads'; PS-DTFE's
#       plan of '--partition 3 3 3' (DTFE_RAM_GB=64, DTFE_AUTO_MINN=1000, '--auto-tune-report') runs min(27, the
#       budget) partitions at once, and its report line still ends with 'over_budget'
#   M3  DTFE_RUN_MODE=GENTLE reads '--run-mode gentle from DTFE_RUN_MODE=GENTLE'; DTFE_RUN_MODE=auto and an empty
#       DTFE_RUN_MODE give M1's line
#   M4  the flag beats the variable: DTFE_RUN_MODE=gentle with '--run-mode performance' is '(--run-mode performance;'
#   M5  refused (a non-zero exit, the binary's text): '--run-mode fast', DTFE_RUN_MODE=fast and '--run-mode' twice in
#       both binaries, PS-DTFE's '--gpu-share 600' and '--gpu-share x', standard DTFE's '--gpu-share 1' (no such option)
#   M6  OMP_NUM_THREADS=2 under gentle keeps its count ('2 threads (kept: OMP_NUM_THREADS=2 was given)'), and the PS
#       plan runs at most 2 partitions at once
#   M7  gentle with '--max-concurrent G+1' (where G+1 <= N0) raises the budget to G+1 ('(raised: --max-concurrent G+1
#       was given)'), both binaries
#   M7b gentle with DTFE_TRIANGULATION_THREADS=G+1 (where G+1 <= N0; a build with the parallel insert, else SKIP):
#       PS-DTFE's one domain inserts on G+1 threads, as it does under performance ('(parallel, TBB, G+1 threads)');
#       the standard team (the largest 2^a*3^b <= G sub-domains) on min(G+1, N0/team) threads each -- a pinned count
#       is cut by the machine's threads shared among the builds, not by the mode's lowered budget
#   M8  Linux: gentle runs at nice 10 ('; nice 10)'), under 'nice -n 15' it keeps 15 ('; nice 15 kept)'); balanced
#       and performance name no nice; macOS: no mode's line names one
#   M9a OMP_NUM_THREADS=2: PS-DTFE '--exact --partition 2 2 2 --max-concurrent 2' and standard '--exact --partition 1'
#       write byte-identical outputs in all three modes
#   M9b single precision, no thread pin: the same PS-DTFE run under gentle and performance, byte-identical (the double
#       build's CPU sums follow the thread count)
#   M10 a run killed after 3 partitions under gentle (DTFE_TEST_EXIT_AFTER_PARTITIONS=3) resumes under '--resume
#       --run-mode performance' to the uninterrupted outputs; its checkpoint's settings key has no 'run-mode' or
#       'gpu-share' line
#   M11 '--gpu-share 0.5' without '--gpu': one note ("does not apply to a run without '--gpu'"; a build without the
#       GPU deposit names 'this CPU-only build' instead) and no key line; '--gpu-share auto' without '--gpu' is silent
# GPU block (a Metal build of the precision's pair; SKIP otherwise):
#   G1  '--gpu --exact --partition 2 2 2 --max-concurrent 2', gentle against performance: 'GPU buffers up to 50 ms'
#       against '250 ms', byte-identical grids, no CPU fallback
#   G2  '--gpu-share 1' is the run without the flag (the outputs and the checkpoint key); '--gpu-share 0 --gpu' is the
#       run without '--gpu'; '--gpu-share 0.5' twice: byte-identical, 'PS-DTFE: GPU share 0.50 (--gpu-share 0.5): 4 of
#       the 8 partitions ...', 'gpu-share=500' in the key, no CPU fallback
#   G3  a 0.5 run killed after 3 partitions: '--resume --gpu-share 1' is refused, naming 'gpu-share=500'; '--resume
#       --gpu-share auto --run-mode performance' takes the checkpoint's share ("the GPU share 0.50 is the
#       checkpoint's") and writes the uninterrupted 0.5 run's outputs
#   G4  OMP_NUM_THREADS=10, DTFE_AUTO_MINN=1000: '--run-mode performance --gpu-share auto --partition 3 3 3' at
#       '--max-concurrent' 1, 2 and 4 is 'GPU share 0.60' each time, byte-identical; '--run-mode balanced' is 'GPU share
#       1 (--gpu-share auto: balanced' with no mixed line (the double build's auto share is 1 in every mode)
#   G5  '--gpu-share 0.5' with '--sampled', and with '--partition 1': the 'sampled deposit' and 'one domain' notes
#   G6  balanced, a watchdog fault at buffer 3 (DTFE_GPU_FAULT_AT=3:timeout) and DTFE_GPU_PACE_RECOVER_SEC=0: the
#       pace eases back to '50 ms buffers, 8 ms gaps (the run mode's)', and no line of the run names 250 ms
#   G7  '--auto-tune-report' (OMP_NUM_THREADS=10, DTFE_AUTO_MINN=1000, DTFE_RAM_GB=64) of '--gpu --exact --gpu-share
#       auto --run-mode performance': the AUTO-TUNE line's 'GPU deposit with the free cores (GPU share 0.60)', the
#       report line as before ('over_budget' last); balanced: no 'GPU share' on the AUTO-TUNE line (the double build:
#       none in either mode)
#   G8  the dynamic concurrency of a mixed run (single precision; DTFE_TEST_PLAN_MC=1, DTFE_FREE_GB=1000, grid 128,
#       '--gpu-share 0.5 --partition 2 2 2', DTFE_AUTO_TUNE_DEBUG=1): the start check's line names a 'once' part
#       between 0 and the partition's bytes, more than the plan's 1 partition is in flight, and the grids are the
#       '--max-concurrent 1' run's; an all-GPU run's line says 'once 0 B'; with DTFE_MEM_BUDGET_GB at own + reserve
#       + 2 lanes + 1.5 times the once part the dynamic concurrency starts (the once part is booked once, not per
#       partition) and a second partition gets in flight (the ledger books it once too), at own + reserve + 2 lanes
#       + half the once part it does not start (it is booked); each of the two runs' own start lines puts its budget
#       on that side of the start check's total, and the summary's predictions of the two differ by a lane for each
#       partition the first had in flight beyond the plan's (the once part is in both once)
# Usage: tests/run_mode_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}" "${DTFE_BIN}"
for b in "${PS_BIN}" "${DTFE_BIN}"; do [ -x "$b" ] || { echo "FAIL: $b is not built"; exit 1; }; done
PS="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
STD="$(cd "$(dirname "${DTFE_BIN}")" && pwd)/$(basename "${DTFE_BIN}")"
TMP="${DTFE_TEST_TMP}/run_mode_check"
rm -rf "$TMP"; mkdir -p "$TMP"
TMP="$(cd "$TMP" && pwd -P)"
# what the cases set themselves: the mode and the thread count (an export from a login shell would steer every run),
# the GPU retry hooks, what moves the auto share off its formula (the float-sum variants, the mass audit, the
# model's constant: src/auto_tune.cc autoTuneGpuShare), and what steers the plan and the dynamic concurrency (G8)
unset DTFE_RUN_MODE OMP_NUM_THREADS OMP_THREAD_LIMIT DTFE_GPU_FAULT_AT DTFE_GPU_PACE_RECOVER_SEC PS_GPU_CHUNK \
      DTFE_GPU_CHECKPOINT_CHUNKS DTFE_GPU_FLOAT_MOMENTS DTFE_GPU_FLOAT_MASS DTFE_MASS_AUDIT DTFE_GPU_SHARE_RHO \
      DTFE_DYNAMIC_CONCURRENCY DTFE_MEM_BUDGET_GB DTFE_MEM_FRACTION DTFE_TEST_PLAN_MC DTFE_AUTO_TUNE_DEBUG
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 16 --box 100 --crossed-waves --seed 7 >/dev/null \
    || { echo "FAIL data gen"; exit 1; }
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }

# ---- the machine, and the budgets the formula gives it
case "$(uname -s)" in
    Darwin) N0="$(sysctl -n hw.logicalcpu 2>/dev/null)"; PERF="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null)" ;;
    *)      N0="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null)"; PERF=0 ;;
esac
[ "${N0:-0}" -ge 1 ] 2>/dev/null || N0=1
[ "${PERF:-0}" -ge 0 ] 2>/dev/null || PERF=0
PERF="${PERF:-0}"
# budget <mode>: src/run_mode.h runModeCoreBudget for N0 and PERF (lround(base/4) of a base >= 0 is (base+2)/4)
budget() {
    local base="$N0" r
    [ "$PERF" -gt 0 ] && [ "$PERF" -lt "$N0" ] && base="$PERF"
    case "$1" in
        gentle)   r=$((base / 2)); [ "$r" -ge 1 ] || r=1 ;;
        balanced) r=$(((base + 2) / 4)); [ "$r" -ge 1 ] || r=1; [ "$r" -le 4 ] || r=4
                  r=$((base - r)); [ "$r" -ge 1 ] || r=1 ;;
        *)        r="$N0" ;;
    esac
    echo "$r"
}
G="$(budget gentle)"; B="$(budget balanced)"
LINUX=0; [ "$(uname -s)" = Linux ] && LINUX=1
# gentle's priority note on Linux (src/user_options.cc applyRunMode: nice 10, never lowered); none on macOS
CUR_NICE=0; NICE_NOTE=""
if [ "$LINUX" = 1 ]; then
    CUR_NICE="$(nice 2>/dev/null)"; [ "${CUR_NICE:-x}" -eq "${CUR_NICE:-x}" ] 2>/dev/null || CUR_NICE=0
    if [ "$CUR_NICE" -lt 10 ]; then NICE_NOTE="nice 10"; else NICE_NOTE="nice $CUR_NICE kept"; fi
fi
echo ">> this machine: N0 = $N0 threads, $PERF performance cores: gentle $G, balanced $B, performance $N0 threads${NICE_NOTE:+ (gentle: $NICE_NOTE)}"

# go <out> <binary> <args...>: one run of '<binary> <snapshot> <out> <args...>', stdout and stderr in <out>.log, with
# the NAME=value list RUN_ENV in its environment (OMP_NUM_THREADS too comes only from it) and the command RUN_WRAP
# ('nice -n 15') in front; the exit status is the binary's (the shell's own 'Killed: 9' line for a
# DTFE_TEST_EXIT_AFTER_PARTITIONS run is not output)
RUN_ENV=(); RUN_WRAP=()
go() {
    local out="$1" bin="$2"; shift 2
    { env ${RUN_ENV[@]+"${RUN_ENV[@]}"} ${RUN_WRAP[@]+"${RUN_WRAP[@]}"} "$bin" "$SNAP" "$out" "$@" > "$out.log" 2>&1; } 2>/dev/null
}
# mode_of <log>: the value of the header's 'run mode' line; share_of <log>: of its 'GPU share' line (D18 a, b)
mode_of()  { LC_ALL=C sed -n 's/^[[:space:]]*run mode *: //p' "$1" | head -n 1; }
share_of() { LC_ALL=C sed -n 's/^[[:space:]]*GPU share *: //p' "$1" | head -n 1; }
# no_nice <log>: the log has a 'run mode' line and it names no nice (a missing line fails: M8 must see one)
no_nice()  { local m; m="$(mode_of "$1" 2>/dev/null)"; [ -n "$m" ] && case "$m" in *nice*) false ;; *) true ;; esac; }
# want <mode> <source> <threads> [<GPU part>]: the value the 'run mode' line should have (gentle adds the priority
# note on Linux)
want() {
    local w="$1 ($2; $3${4:+; $4}"
    [ "$1" = gentle ] && [ -n "$NICE_NOTE" ] && w="$w; $NICE_NOTE"
    echo "$w)"
}
of_n() { echo "$1 of $N0 threads"; }
# is_mode <log> <value>: the 'run mode' line reads exactly that (else both are printed)
is_mode() {
    local got; got="$(mode_of "$1")"
    [ "$got" = "$2" ] && return 0
    echo "    ${1##*/}: run mode : '$got'"; echo "    expected : '$2'"; return 1
}
# mc_of <log>: the concurrency of its AUTO-TUNE-REPORT line
mc_of() { LC_ALL=C sed -n 's/^AUTO-TUNE-REPORT partition=[0-9]* mc=\([0-9]*\) .*/\1/p' "$1" | head -n 1; }
# key_of <output>: the settings key block of its checkpoint (the header is text; the grids follow it)
key_of() { LC_ALL=C sed -n '/^key-begin$/,/^key-end$/p;/^header-end$/q' "$1.runlog.checkpoint/checkpoint" 2>/dev/null; }
# has <log> <text>: the literal text is in the log
has() { grep -aqF -- "$2" "$1"; }
FALLBACK='using the CPU deposit|deposit unavailable|ran on the CPU instead|gave up'
no_fallback() { ! grep -aqE "$FALLBACK" "$@"; }
# the D19 texts and the not-applicable note (contract section 7)
E_MODE="Unknown value 'fast' for '--run-mode'; use 'gentle', 'balanced' or 'performance' (the default)."
E_ENV="Unknown value 'fast' for DTFE_RUN_MODE; use gentle, balanced or performance (in any case), or leave it unset (or 'auto') for the default."
e_share() { echo "Unknown value '$1' for '--gpu-share'; use a fraction from 0 to 1 (0.6: 60% of the partitions deposit on the GPU) or 'auto' (the run mode's share)."; }
note_of() { echo "note: '--gpu-share $1' does not apply to $2: every partition deposits on $3."; }
# mixed_line <share> <source> <on the GPU> <to deposit>: the post-plan line of a mixed run (D18 c)
mixed_line() { echo "PS-DTFE: GPU share $1 ($2): $3 of the $4 partitions to deposit do so on the GPU, the other $(($4 - $3)) on the CPU with r3d clipping (a mixed run's grids differ from an all-GPU run's at float rounding)."; }
MIXED="partitions to deposit do so on the GPU, the other"

A=(--input-format 105 --mpc-unit 1 --periodic --grid 16 --verbose 2 --field density_a velocity_a)
DEF="--run-mode performance, the default"

echo ">> M1: no flag, DTFE_RUN_MODE and OMP_NUM_THREADS unset"
for b in PS STD; do
    bin="$PS"; [ "$b" = STD ] && bin="$STD"
    go "$TMP/M1_$b" "$bin" "${A[@]}" --auto-tune-report; rc=$?
    [ "$rc" = 0 ] && is_mode "$TMP/M1_$b.log" "$(want performance "$DEF" "$(of_n "$N0")")"
    check "M1 $b: 'run mode : performance ($DEF; $N0 of $N0 threads)' (exit $rc)" $?
done

echo ">> M2: --run-mode balanced and gentle"
for m in balanced gentle; do
    k="$B"; [ "$m" = gentle ] && k="$G"
    RUN_ENV=(DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000)
    go "$TMP/M2_PS_$m" "$PS" "${A[@]}" --partition 3 3 3 --auto-tune-report --run-mode "$m"; rc=$?
    RUN_ENV=()
    mc=$(( k < 27 ? k : 27 )); got="$(mc_of "$TMP/M2_PS_$m.log")"
    [ "$rc" = 0 ] && is_mode "$TMP/M2_PS_$m.log" "$(want "$m" "--run-mode $m" "$(of_n "$k")")" \
        && grep -aEq "^AUTO-TUNE-REPORT partition=3 mc=$mc .*over_budget=[01]$" "$TMP/M2_PS_$m.log"
    check "M2 PS $m: '$k of $N0 threads', and the plan of --partition 3 3 3 runs mc=$mc (mc=${got:-none}; exit $rc)" $?
    go "$TMP/M2_STD_$m" "$STD" "${A[@]}" --auto-tune-report --run-mode "$m"; rc=$?
    [ "$rc" = 0 ] && is_mode "$TMP/M2_STD_$m.log" "$(want "$m" "--run-mode $m" "$(of_n "$k")")"
    check "M2 STD $m: '$k of $N0 threads' (exit $rc)" $?
done

echo ">> M3: DTFE_RUN_MODE (any case; 'auto' and empty = not given)"
for b in PS STD; do
    bin="$PS"; [ "$b" = STD ] && bin="$STD"
    RUN_ENV=(DTFE_RUN_MODE=GENTLE); go "$TMP/M3_${b}_up" "$bin" "${A[@]}" --auto-tune-report; r1=$?
    RUN_ENV=(DTFE_RUN_MODE=auto);   go "$TMP/M3_${b}_auto" "$bin" "${A[@]}" --auto-tune-report; r2=$?
    RUN_ENV=(DTFE_RUN_MODE=);       go "$TMP/M3_${b}_empty" "$bin" "${A[@]}" --auto-tune-report; r3=$?
    RUN_ENV=()
    [ "$r1" = 0 ] && is_mode "$TMP/M3_${b}_up.log" "$(want gentle "--run-mode gentle from DTFE_RUN_MODE=GENTLE" "$(of_n "$G")")"
    check "M3 $b: DTFE_RUN_MODE=GENTLE is '--run-mode gentle from DTFE_RUN_MODE=GENTLE' (exit $r1)" $?
    [ "$r2$r3" = 00 ] && is_mode "$TMP/M3_${b}_auto.log" "$(want performance "$DEF" "$(of_n "$N0")")" \
        && is_mode "$TMP/M3_${b}_empty.log" "$(want performance "$DEF" "$(of_n "$N0")")"
    check "M3 $b: DTFE_RUN_MODE=auto and DTFE_RUN_MODE= give M1's line (exit $r2, $r3)" $?
done

echo ">> M4: the flag beats DTFE_RUN_MODE"
for b in PS STD; do
    bin="$PS"; [ "$b" = STD ] && bin="$STD"
    RUN_ENV=(DTFE_RUN_MODE=gentle); go "$TMP/M4_$b" "$bin" "${A[@]}" --auto-tune-report --run-mode performance; rc=$?
    RUN_ENV=()
    [ "$rc" = 0 ] && is_mode "$TMP/M4_$b.log" "$(want performance "--run-mode performance" "$(of_n "$N0")")"
    check "M4 $b: DTFE_RUN_MODE=gentle with --run-mode performance is '(--run-mode performance; ...)' (exit $rc)" $?
done

echo ">> M5: refused values"
for b in PS STD; do
    bin="$PS"; [ "$b" = STD ] && bin="$STD"
    go "$TMP/M5_${b}_flag" "$bin" "${A[@]}" --auto-tune-report --run-mode fast; rc=$?
    [ "$rc" != 0 ] && has "$TMP/M5_${b}_flag.log" "$E_MODE"
    check "M5 $b: '--run-mode fast' is refused with the --run-mode text (exit $rc)" $?
    RUN_ENV=(DTFE_RUN_MODE=fast); go "$TMP/M5_${b}_env" "$bin" "${A[@]}" --auto-tune-report; rc=$?; RUN_ENV=()
    [ "$rc" != 0 ] && has "$TMP/M5_${b}_env.log" "$E_ENV"
    check "M5 $b: DTFE_RUN_MODE=fast is refused with the DTFE_RUN_MODE text (exit $rc)" $?
    go "$TMP/M5_${b}_twice" "$bin" "${A[@]}" --auto-tune-report --run-mode gentle --run-mode balanced; rc=$?
    [ "$rc" != 0 ] && has "$TMP/M5_${b}_twice.log" "'--run-mode' cannot be specified more than once"
    check "M5 $b: '--run-mode' twice is refused (boost: 'cannot be specified more than once'; exit $rc)" $?
done
for v in 600 x; do
    go "$TMP/M5_PS_share$v" "$PS" "${A[@]}" --auto-tune-report --gpu-share "$v"; rc=$?
    [ "$rc" != 0 ] && has "$TMP/M5_PS_share$v.log" "$(e_share "$v")"
    check "M5 PS: '--gpu-share $v' is refused with the --gpu-share text (exit $rc)" $?
done
go "$TMP/M5_STD_share" "$STD" "${A[@]}" --auto-tune-report --gpu-share 1; rc=$?
[ "$rc" != 0 ] && has "$TMP/M5_STD_share.log" "unrecognised option '--gpu-share'"
check "M5 STD: '--gpu-share 1' is refused, an unknown option of standard DTFE (exit $rc)" $?

echo ">> M6: OMP_NUM_THREADS=2 keeps its count under gentle"
RUN_ENV=(OMP_NUM_THREADS=2 DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000)
go "$TMP/M6_PS" "$PS" "${A[@]}" --partition 3 3 3 --auto-tune-report --run-mode gentle; r1=$?
go "$TMP/M6_STD" "$STD" "${A[@]}" --auto-tune-report --run-mode gentle; r2=$?
RUN_ENV=()
KEPT="2 threads (kept: OMP_NUM_THREADS=2 was given)"
mc="$(mc_of "$TMP/M6_PS.log")"
[ "$r1" = 0 ] && is_mode "$TMP/M6_PS.log" "$(want gentle "--run-mode gentle" "$KEPT")" && [ -n "$mc" ] && [ "$mc" -le 2 ]
check "M6 PS: '$KEPT', and the plan runs at most 2 partitions at once (mc=${mc:-none}; exit $r1)" $?
[ "$r2" = 0 ] && is_mode "$TMP/M6_STD.log" "$(want gentle "--run-mode gentle" "$KEPT")"
check "M6 STD: '$KEPT' (exit $r2)" $?

echo ">> M7: a --max-concurrent above gentle's threads raises them"
K=$((G + 1))
if [ "$K" -le "$N0" ]; then
    RUN_ENV=(DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000)
    go "$TMP/M7_PS" "$PS" "${A[@]}" --partition 3 3 3 --max-concurrent "$K" --auto-tune-report --run-mode gentle; r1=$?
    go "$TMP/M7_STD" "$STD" "${A[@]}" --partition 2 2 2 --max-concurrent "$K" --auto-tune-report --run-mode gentle; r2=$?
    RUN_ENV=()
    RAISED="$(of_n "$K") (raised: --max-concurrent $K was given)"
    [ "$r1" = 0 ] && is_mode "$TMP/M7_PS.log" "$(want gentle "--run-mode gentle" "$RAISED")"
    check "M7 PS: gentle with --max-concurrent $K reads '$RAISED' (exit $r1)" $?
    [ "$r2" = 0 ] && is_mode "$TMP/M7_STD.log" "$(want gentle "--run-mode gentle" "$RAISED")"
    check "M7 STD: the same (exit $r2)" $?
else
    echo "  SKIP M7: gentle's $G threads + 1 is more than this machine's $N0"
fi

echo ">> M7b: a pinned DTFE_TRIANGULATION_THREADS keeps its count under gentle"
# src/CGAL_triangulation/triangulation.cpp delaunayTriangulation: the pin is an explicit lever, like OMP_NUM_THREADS
# and '--max-concurrent' -- capped by the machine's threads shared among the builds in flight (and by
# OMP_NUM_THREADS), not by the mode's lowered budget
if [ "$K" -le "$N0" ]; then
    PINNED="(parallel, TBB, $K threads)"
    RUN_ENV=(DTFE_TRIANGULATION_THREADS="$K")
    go "$TMP/M7b_PS_perf" "$PS" "${A[@]}" --partition 1 1 1 --run-mode performance; r0=$?
    go "$TMP/M7b_PS" "$PS" "${A[@]}" --partition 1 1 1 --run-mode gentle; r1=$?
    RUN_ENV=()
    if [ "$r0" = 0 ] && has "$TMP/M7b_PS_perf.log" "$PINNED"; then
        [ "$r1" = 0 ] && has "$TMP/M7b_PS.log" "$PINNED"
        check "M7b PS: gentle with DTFE_TRIANGULATION_THREADS=$K inserts on $K threads, as performance does (exit $r1)" $?
    else
        echo "  SKIP M7b PS: no '$PINNED' line under performance (a build without the parallel insert; exit $r0)"
    fi
    # the standard team under gentle: NP sub-domains, the largest 2^a*3^b <= G (src/subpartition.h parallelGrid),
    # each on min(K, N0/NP) threads. Whether the build has the parallel insert shows under performance with a
    # team of 2 (N0/2 threads each, capped at the 2 pinned).
    NP=1; a=1
    while [ "$a" -le "$G" ]; do
        b=$a; while [ "$b" -le "$G" ]; do [ "$b" -gt "$NP" ] && NP=$b; b=$((b * 3)); done
        a=$((a * 2))
    done
    T=$((N0 / NP)); [ "$T" -le "$K" ] || T="$K"
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_TRIANGULATION_THREADS=2)
    go "$TMP/M7b_STD_probe" "$STD" "${A[@]}" --run-mode performance; r0=$?
    RUN_ENV=(DTFE_TRIANGULATION_THREADS="$K")
    go "$TMP/M7b_STD" "$STD" "${A[@]}" --run-mode gentle; r2=$?
    RUN_ENV=()
    if [ "$N0" -ge 4 ] && [ "$T" -ge 2 ] && [ "$r0" = 0 ] && has "$TMP/M7b_STD_probe.log" "(parallel, TBB, 2 threads)"; then
        [ "$r2" = 0 ] && has "$TMP/M7b_STD.log" "(parallel, TBB, $T threads)"
        check "M7b STD: gentle's team of $NP with DTFE_TRIANGULATION_THREADS=$K inserts on $T threads each (exit $r2)" $?
    else
        echo "  SKIP M7b STD: no '(parallel, TBB, 2 threads)' line from a team of 2 under performance (a build without the parallel insert, or fewer than 4 threads; exit $r0)"
    fi
else
    echo "  SKIP M7b: gentle's $G threads + 1 is more than this machine's $N0"
fi

echo ">> M8: the priority (Linux: gentle at nice 10; macOS: none)"
if [ "$LINUX" = 1 ]; then
    got="$(mode_of "$TMP/M2_PS_gentle.log")"
    case "$got" in *"; $NICE_NOTE)") rc=0 ;; *) rc=1; echo "    run mode : '$got'" ;; esac
    check "M8 gentle ends with '; $NICE_NOTE)' (this shell's niceness: $CUR_NICE)" $rc
    N15=$((CUR_NICE + 15)); [ "$N15" -le 19 ] || N15=19
    RUN_WRAP=(nice -n 15); go "$TMP/M8_nice15" "$PS" "${A[@]}" --auto-tune-report --run-mode gentle; rc=$?; RUN_WRAP=()
    got="$(mode_of "$TMP/M8_nice15.log")"
    [ "$rc" = 0 ] && case "$got" in *"; nice $N15 kept)") true ;; *) echo "    run mode : '$got'"; false ;; esac
    check "M8 under 'nice -n 15' gentle ends with '; nice $N15 kept)' (exit $rc)" $?
    no_nice "$TMP/M2_PS_balanced.log" && no_nice "$TMP/M1_PS.log"
    check "M8 balanced and performance name no nice" $?
else
    rc=0
    for l in "$TMP"/M1_*.log "$TMP"/M2_*.log "$TMP"/M6_*.log; do
        no_nice "$l" || { echo "    ${l##*/}: run mode : '$(mode_of "$l" 2>/dev/null)'"; rc=1; }
    done
    check "M8 not Linux: no mode's line names a nice" $rc
fi

echo ">> M9a: OMP_NUM_THREADS=2, all three modes: byte-identical outputs"
PSX=(--exact --partition 2 2 2 --max-concurrent 2)
RUN_ENV=(OMP_NUM_THREADS=2)
for m in gentle balanced performance; do
    go "$TMP/M9a_PS_$m" "$PS" "${A[@]}" "${PSX[@]}" --run-mode "$m"; eval "rp_$m=\$?"
    go "$TMP/M9a_STD_$m" "$STD" "${A[@]}" --exact --partition 1 --run-mode "$m"; eval "rs_$m=\$?"
done
RUN_ENV=()
[ "$rp_gentle$rp_balanced$rp_performance" = 000 ] && same "$TMP/M9a_PS_performance" "$TMP/M9a_PS_gentle" \
    && same "$TMP/M9a_PS_performance" "$TMP/M9a_PS_balanced"
check "M9a PS --exact --partition 2 2 2 --max-concurrent 2: every output byte-identical across the modes (exit $rp_gentle, $rp_balanced, $rp_performance)" $?
[ "$rs_gentle$rs_balanced$rs_performance" = 000 ] && same "$TMP/M9a_STD_performance" "$TMP/M9a_STD_gentle" \
    && same "$TMP/M9a_STD_performance" "$TMP/M9a_STD_balanced"
check "M9a STD --exact --partition 1: every output byte-identical across the modes (exit $rs_gentle, $rs_balanced, $rs_performance)" $?

echo ">> M9b: no thread pin, gentle against performance (single precision)"
if precision_double; then
    echo "  SKIP M9b: the double build's CPU sums follow the thread count"
else
    go "$TMP/M9b_gentle" "$PS" "${A[@]}" "${PSX[@]}" --run-mode gentle; r1=$?
    go "$TMP/M9b_performance" "$PS" "${A[@]}" "${PSX[@]}" --run-mode performance; r2=$?
    [ "$r1$r2" = 00 ] && same "$TMP/M9b_performance" "$TMP/M9b_gentle"
    check "M9b PS: $G against $N0 threads, every output byte-identical (exit $r1, $r2)" $?
fi

echo ">> M10: killed under gentle, resumed under performance"
RUN_ENV=(OMP_NUM_THREADS=2 DTFE_TEST_EXIT_AFTER_PARTITIONS=3)
go "$TMP/M10" "$PS" "${A[@]}" "${PSX[@]}" --run-mode gentle; r1=$?
RUN_ENV=(OMP_NUM_THREADS=2)
KEY="$(key_of "$TMP/M10")"
mv "$TMP/M10.log" "$TMP/M10.killed.log"
[ -n "$KEY" ] || no_checkpoint_why "$TMP/M10.killed.log" "$TMP"
go "$TMP/M10" "$PS" "${A[@]}" "${PSX[@]}" --resume --run-mode performance; r2=$?
RUN_ENV=()
[ "$r1" = 137 ] && [ "$r2" = 0 ] && has "$TMP/M10.log" "--resume: " && same "$TMP/M9a_PS_performance" "$TMP/M10"
check "M10 killed after 3 partitions (exit $r1), resumed under --run-mode performance (exit $r2): the uninterrupted outputs" $?
grep -q '^opt ' <<< "$KEY" && ! grep -qE 'run-mode|gpu-share' <<< "$KEY"
check "M10 the checkpoint's settings key has no 'run-mode' or 'gpu-share' line" $?

echo ">> M11: --gpu-share without --gpu"
OBJ=o_ps; precision_double && OBJ=o_ps_d
GPU=0
if [ -d "$OBJ" ] && [ ! -f "$OBJ/.gpu_mode_off" ] && [ -n "$(gpu_backend_of "$OBJ")" ]; then GPU=1; fi
WHY="a run without '--gpu'"; [ "$GPU" = 1 ] || WHY="this CPU-only build"   # the first cause that holds
RUN_ENV=(OMP_NUM_THREADS=2 DTFE_TEST_EXIT_AFTER_PARTITIONS=3)
go "$TMP/M11" "$PS" "${A[@]}" "${PSX[@]}" --gpu-share 0.5; r1=$?
RUN_ENV=(OMP_NUM_THREADS=2)
go "$TMP/M11_auto" "$PS" "${A[@]}" "${PSX[@]}" --gpu-share auto; r2=$?
RUN_ENV=()
NOTE="$(note_of 0.5 "$WHY" "the CPU")"
KEY="$(key_of "$TMP/M11")"
[ -n "$KEY" ] || no_checkpoint_why "$TMP/M11.log" "$TMP"
[ "$r1" = 137 ] && [ "$(grep -acF -- "$NOTE" "$TMP/M11.log")" = 1 ] && grep -q '^opt ' <<< "$KEY" \
    && ! grep -q 'gpu-share' <<< "$KEY"
check "M11 '--gpu-share 0.5' without --gpu: the note \"$NOTE\" once, and no key line (exit $r1)" $?
[ "$r2" = 0 ] && ! grep -aqE "does not apply|GPU share" "$TMP/M11_auto.log"
check "M11 '--gpu-share auto' without --gpu says nothing (exit $r2)" $?

echo ">> G: the GPU block"
if [ "$GPU" != 1 ]; then
    echo "  SKIP G1-G8: $OBJ is not a Metal GPU build"
else
    GX=(--gpu --exact --partition 2 2 2 --max-concurrent 2)

    echo ">> G1: gentle against performance on the GPU"
    go "$TMP/G1_gentle" "$PS" "${A[@]}" "${GX[@]}" --run-mode gentle; r1=$?
    go "$TMP/G1_performance" "$PS" "${A[@]}" "${GX[@]}" --run-mode performance; r2=$?
    [ "$r1$r2" = 00 ] && is_mode "$TMP/G1_gentle.log" "$(want gentle "--run-mode gentle" "$(of_n "$G")" "GPU buffers up to 50 ms")" \
        && is_mode "$TMP/G1_performance.log" "$(want performance "--run-mode performance" "$(of_n "$N0")" "GPU buffers up to 250 ms")"
    check "G1 the header's 'GPU buffers up to 50 ms' (gentle) against '250 ms' (performance) (exit $r1, $r2)" $?
    [ "$r1$r2" = 00 ] && same "$TMP/G1_performance" "$TMP/G1_gentle" && no_fallback "$TMP/G1_gentle.log" "$TMP/G1_performance.log"
    check "G1 every grid byte-identical, and no CPU fallback" $?

    echo ">> G2: --gpu-share 1, 0 and 0.5 (OMP_NUM_THREADS=2)"
    RUN_ENV=(OMP_NUM_THREADS=2)
    go "$TMP/G2_none" "$PS" "${A[@]}" "${GX[@]}"; r1=$?
    go "$TMP/G2_one" "$PS" "${A[@]}" "${GX[@]}" --gpu-share 1; r2=$?
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_TEST_EXIT_AFTER_PARTITIONS=3)
    go "$TMP/G2_k" "$PS" "${A[@]}" "${GX[@]}"; r3=$?; k1="$(key_of "$TMP/G2_k")"
    [ -n "$k1" ] || no_checkpoint_why "$TMP/G2_k.log" "$TMP"; rm -rf "$TMP"/G2_k.*
    go "$TMP/G2_k" "$PS" "${A[@]}" "${GX[@]}" --gpu-share 1; r4=$?; k2="$(key_of "$TMP/G2_k")"
    [ -n "$k2" ] || no_checkpoint_why "$TMP/G2_k.log" "$TMP"; rm -rf "$TMP"/G2_k.*
    RUN_ENV=()
    [ "$r1$r2" = 00 ] && [ "$r3$r4" = 137137 ] && same "$TMP/G2_none" "$TMP/G2_one" && grep -q '^opt ' <<< "$k1" \
        && [ "$k1" = "$k2" ] && [ "$(share_of "$TMP/G2_one.log")" = "1 (--gpu-share)" ]
    rc=$?; [ "$rc" = 0 ] || diff <(echo "$k1") <(echo "$k2") | sed 's/^/      | /'
    check "G2 '--gpu-share 1' is the run without it: the outputs and the checkpoint key (exit $r1, $r2; killed $r3, $r4)" $rc
    RUN_ENV=(OMP_NUM_THREADS=2)
    go "$TMP/G2_zero" "$PS" "${A[@]}" "${GX[@]}" --gpu-share 0; r1=$?
    go "$TMP/G2_cpu" "$PS" "${A[@]}" --exact --partition 2 2 2 --max-concurrent 2; r2=$?
    [ "$r1$r2" = 00 ] && same "$TMP/G2_cpu" "$TMP/G2_zero"
    check "G2 '--gpu-share 0 --gpu' writes the outputs of the run without --gpu (exit $r1, $r2)" $?
    go "$TMP/G2_half" "$PS" "${A[@]}" "${GX[@]}" --gpu-share 0.5; r1=$?
    go "$TMP/G2_half2" "$PS" "${A[@]}" "${GX[@]}" --gpu-share 0.5; r2=$?
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_TEST_EXIT_AFTER_PARTITIONS=3)
    go "$TMP/G3" "$PS" "${A[@]}" "${GX[@]}" --gpu-share 0.5; r3=$?
    RUN_ENV=()
    KEY="$(key_of "$TMP/G3")"
    [ -n "$KEY" ] || no_checkpoint_why "$TMP/G3.log" "$TMP"
    HALF="$(mixed_line 0.50 "--gpu-share 0.5" 4 8)"
    [ "$r1$r2" = 00 ] && same "$TMP/G2_half" "$TMP/G2_half2" && has "$TMP/G2_half.log" "$HALF" \
        && [ "$(share_of "$TMP/G2_half.log")" = "0.5 (--gpu-share)" ] && no_fallback "$TMP/G2_half.log" "$TMP/G2_half2.log"
    check "G2 '--gpu-share 0.5' twice: byte-identical, '${HALF%% to deposit*}', no CPU fallback (exit $r1, $r2)" $?
    [ "$r3" = 137 ] && grep -qx 'gpu-share=500' <<< "$KEY"
    check "G2 the key of a killed 0.5 run holds 'gpu-share=500' (exit $r3)" $?

    echo ">> G3: the killed 0.5 run, resumed"
    mv "$TMP/G3.log" "$TMP/G3.killed.log"
    RUN_ENV=(OMP_NUM_THREADS=2)
    go "$TMP/G3" "$PS" "${A[@]}" "${GX[@]}" --resume --gpu-share 1; r1=$?
    mv "$TMP/G3.log" "$TMP/G3.refused.log"
    go "$TMP/G3" "$PS" "${A[@]}" "${GX[@]}" --resume --gpu-share auto --run-mode performance; r2=$?
    RUN_ENV=()
    [ "$r1" != 0 ] && has "$TMP/G3.refused.log" "'--resume' refused" && has "$TMP/G3.refused.log" "gpu-share=500"
    check "G3 '--resume --gpu-share 1' is refused, naming 'gpu-share=500' (exit $r1)" $?
    [ "$r2" = 0 ] && has "$TMP/G3.log" "PS-DTFE: --resume: the GPU share 0.50 is the checkpoint's." \
        && same "$TMP/G2_half" "$TMP/G3" && no_fallback "$TMP/G3.log"
    check "G3 '--resume --gpu-share auto --run-mode performance' takes the checkpoint's 0.50: the uninterrupted outputs (exit $r2)" $?

    echo ">> G4: --gpu-share auto at OMP_NUM_THREADS=10 (DTFE_AUTO_MINN=1000)"
    RUN_ENV=(OMP_NUM_THREADS=10 DTFE_AUTO_MINN=1000)
    G4=(--gpu --exact --partition 3 3 3 --gpu-share auto)
    for c in 1 2 4; do
        go "$TMP/G4_mc$c" "$PS" "${A[@]}" "${G4[@]}" --max-concurrent "$c" --run-mode performance; eval "r_$c=\$?"
    done
    go "$TMP/G4_balanced" "$PS" "${A[@]}" "${G4[@]}" --max-concurrent 2 --run-mode balanced; rb=$?
    RUN_ENV=()
    if precision_double; then   # D11 rule 2: the double build's auto share is 1 in every mode
        PRE="PS-DTFE: GPU share 1 (--gpu-share auto: "; SUF="): every partition deposits on the GPU."
    else
        PRE="PS-DTFE: GPU share 0.60 (--gpu-share auto: "
        SUF="): 16 of the 27 partitions to deposit do so on the GPU, the other 11 on the CPU with r3d clipping (a mixed run's grids differ from an all-GPU run's at float rounding)."
    fi
    ok=0
    for c in 1 2 4; do
        eval "r=\$r_$c"
        [ "$r" = 0 ] && grep -aF -- "$PRE" "$TMP/G4_mc$c.log" | grep -qF -- "$SUF" || { echo "    G4_mc$c: exit $r, no '$PRE...$SUF'"; ok=1; }
    done
    [ "$ok" = 0 ] && [ "$(share_of "$TMP/G4_mc1.log")" = "auto (the run mode's share, resolved with the plan)" ] \
        && same "$TMP/G4_mc1" "$TMP/G4_mc2" && same "$TMP/G4_mc1" "$TMP/G4_mc4" \
        && no_fallback "$TMP/G4_mc1.log" "$TMP/G4_mc2.log" "$TMP/G4_mc4.log"
    check "G4 performance: '${PRE}...', byte-identical at --max-concurrent 1, 2 and 4, no CPU fallback" $?
    [ "$rb" = 0 ] && has "$TMP/G4_balanced.log" "PS-DTFE: GPU share 1 (--gpu-share auto: balanced" \
        && ! has "$TMP/G4_balanced.log" "$MIXED"
    check "G4 balanced: 'GPU share 1 (--gpu-share auto: balanced', no mixed line (exit $rb)" $?

    echo ">> G5: the notes of a --gpu-share that does not apply"
    RUN_ENV=(OMP_NUM_THREADS=2)
    go "$TMP/G5_sampled" "$PS" "${A[@]}" --gpu --sampled --partition 2 2 2 --max-concurrent 2 --gpu-share 0.5; r1=$?
    go "$TMP/G5_one" "$PS" "${A[@]}" --gpu --exact --partition 1 --gpu-share 0.5; r2=$?
    RUN_ENV=()
    [ "$r1" = 0 ] && has "$TMP/G5_sampled.log" "$(note_of 0.5 "the sampled deposit" "the GPU")" \
        && ! has "$TMP/G5_sampled.log" "PS-DTFE: GPU share"
    check "G5 '--gpu-share 0.5 --sampled': the 'sampled deposit' note, no share line (exit $r1)" $?
    [ "$r2" = 0 ] && has "$TMP/G5_one.log" "$(note_of 0.5 "one domain (no partition loop)" "the GPU")" \
        && ! has "$TMP/G5_one.log" "PS-DTFE: GPU share"
    check "G5 '--gpu-share 0.5 --partition 1': the 'one domain' note, no share line (exit $r2)" $?

    echo ">> G6: balanced, a watchdog fault: the pace eases back to the run mode's base"
    # pinned chunking (as tests/gpu_retry_check.sh): the opening chunk and a checkpoint every 2 buffers
    RUN_ENV=(PS_GPU_CHUNK=2000 DTFE_GPU_CHECKPOINT_CHUNKS=2 DTFE_GPU_FAULT_AT=3:timeout DTFE_GPU_PACE_RECOVER_SEC=0)
    go "$TMP/G6" "$PS" "${A[@]}" --gpu --exact --partition 2 2 2 --max-concurrent 1 --run-mode balanced; rc=$?
    RUN_ENV=()
    [ "$rc" = 0 ] && has "$TMP/G6.log" "[watchdog]" && has "$TMP/G6.log" "pace eased to 50 ms buffers, 8 ms gaps (the run mode's)" \
        && ! grep -aqE '(^|[^0-9.])250 ms' "$TMP/G6.log" && no_fallback "$TMP/G6.log"
    check "G6 the watchdog fault eases back to '50 ms buffers, 8 ms gaps (the run mode's)', never 250 ms (exit $rc)" $?

    echo ">> G7: the auto-tuner's lines (OMP_NUM_THREADS=10, DTFE_AUTO_MINN=1000, DTFE_RAM_GB=64)"
    RUN_ENV=(OMP_NUM_THREADS=10 DTFE_AUTO_MINN=1000 DTFE_RAM_GB=64)
    go "$TMP/G7_performance" "$PS" "${A[@]}" --gpu --exact --gpu-share auto --run-mode performance --auto-tune-report; r1=$?
    go "$TMP/G7_balanced" "$PS" "${A[@]}" --gpu --exact --gpu-share auto --run-mode balanced --auto-tune-report; r2=$?
    RUN_ENV=()
    tune_of() { grep -a '^AUTO-TUNE: .* -> --partition' "$1" | head -n 1; }
    TAG="GPU deposit with the free cores (GPU share 0.60)"
    if precision_double; then   # (D11 rule 2: not mixed)
        TAG=", GPU deposit"
        tune_of "$TMP/G7_performance.log" | grep -qF -- "$TAG" && ! tune_of "$TMP/G7_performance.log" | grep -q "GPU share"
    else
        tune_of "$TMP/G7_performance.log" | grep -qF -- "$TAG"
    fi
    rc=$?
    [ "$r1" = 0 ] && [ "$rc" = 0 ] && grep -aEq '^AUTO-TUNE-REPORT .*over_budget=[01]$' "$TMP/G7_performance.log"
    check "G7 performance: the AUTO-TUNE line's '$TAG', the report's over_budget last (exit $r1)" $?
    [ "$r2" = 0 ] && tune_of "$TMP/G7_balanced.log" | grep -qF ", GPU deposit" && ! tune_of "$TMP/G7_balanced.log" | grep -q "GPU share" \
        && grep -aEq '^AUTO-TUNE-REPORT .*over_budget=[01]$' "$TMP/G7_balanced.log"
    check "G7 balanced: no 'GPU share' on the AUTO-TUNE line (exit $r2)" $?

    echo ">> G8: the dynamic concurrency of a mixed run books the GPU lock holder's grids once"
    if precision_double; then
        echo "  SKIP G8: the double build keeps the plan's concurrency (no dynamic concurrency)"
    else
        A8=(--input-format 105 --mpc-unit 1 --periodic --grid 128 --verbose 2 --field density_a velocity_a)
        G8=(--gpu --exact --partition 2 2 2)
        E8=(OMP_NUM_THREADS=4 DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_TEST_PLAN_MC=1 DTFE_AUTO_TUNE_DEBUG=1)
        # start_of <log>: 'partition once reserve own' of the start check's debug line (bytes, bytes, bytes, GB)
        start_of() {
            LC_ALL=C sed -n 's/^AUTO-TUNE-DEBUG PS dynamic start: partition \([0-9]*\) B, once \([0-9]*\) B, reserve \([0-9]*\) B, free [0-9.]* GB, own \([0-9.]*\) GB, budget .*/\1 \2 \3 \4/p' "$1" | head -n 1
        }
        # most_of <log>: the most partitions the dynamic concurrency had in flight
        most_of() { LC_ALL=C sed -n 's/^PS-DTFE: dynamic concurrency: at most \([0-9]*\) partitions in flight at once.*/\1/p' "$1" | head -n 1; }
        # pred_of <log>: the summary's predicted memory (GB), which holds a lane for each partition beyond the plan's
        pred_of() { LC_ALL=C sed -n 's/.*memory model *: \([0-9.]*\) GB predicted (auto-tune).*/\1/p' "$1" | head -n 1; }
        STARTED="(dynamic: at least the plan's 1,"
        NOT_STARTED="dynamic concurrency not started: the free memory or the budget at the start holds no partition beyond it"
        RUN_ENV=("${E8[@]}")
        go "$TMP/G8_dyn" "$PS" "${A8[@]}" "${G8[@]}" --gpu-share 0.5; r1=$?
        go "$TMP/G8_all" "$PS" "${A8[@]}" "${G8[@]}" --gpu-share 1; r3=$?
        RUN_ENV=(OMP_NUM_THREADS=4 DTFE_RAM_GB=64 DTFE_FREE_GB=1000)
        go "$TMP/G8_ref" "$PS" "${A8[@]}" "${G8[@]}" --gpu-share 0.5 --max-concurrent 1; r2=$?
        read -r P8 O8 R8 OWN8 <<< "$(start_of "$TMP/G8_dyn.log")"
        [ "$r1" = 0 ] && [ "${O8:-0}" -gt 0 ] && [ "$O8" -lt "${P8:-0}" ] && has "$TMP/G8_dyn.log" "$STARTED" \
            && [ "$(most_of "$TMP/G8_dyn.log")" -ge 2 ] 2>/dev/null
        check "G8 a mixed run: the start check's once part ${O8:-?} B of the partition's ${P8:-?} B, '${STARTED}...', $(most_of "$TMP/G8_dyn.log") partitions in flight (exit $r1)" $?
        [ "$r1$r2" = 00 ] && same "$TMP/G8_ref" "$TMP/G8_dyn" && no_fallback "$TMP/G8_dyn.log" "$TMP/G8_ref.log"
        check "G8 every grid byte-identical to the --max-concurrent 1 run's, no CPU fallback (exit $r2)" $?
        read -r Pa Oa Ra OWNa <<< "$(start_of "$TMP/G8_all.log")"
        [ "$r3" = 0 ] && [ "${Oa:-x}" = 0 ] && [ "${Pa:-0}" -gt 0 ] && has "$TMP/G8_all.log" "$STARTED"
        check "G8 an all-GPU run: 'once 0 B' (every partition books its ${Pa:-?} B in full) (exit $r3)" $?
        if [ "${O8:-0}" -gt 0 ] 2>/dev/null; then
            # the budget the start check compares: own + reserve + (floor + 1) lanes + the once part once
            B_IN="$(awk -v p="$P8" -v o="$O8" -v r="$R8" -v own="$OWN8" 'BEGIN { printf "%.6f", ( own*1e9 + r + 2*(p-o) + 1.5*o ) / 1e9 }')"
            B_OUT="$(awk -v p="$P8" -v o="$O8" -v r="$R8" -v own="$OWN8" 'BEGIN { printf "%.6f", ( own*1e9 + r + 2*(p-o) + 0.5*o ) / 1e9 }')"
            RUN_ENV=("${E8[@]}" DTFE_MEM_BUDGET_GB="$B_IN")
            go "$TMP/G8_in" "$PS" "${A8[@]}" "${G8[@]}" --gpu-share 0.5; r4=$?
            RUN_ENV=("${E8[@]}" DTFE_MEM_BUDGET_GB="$B_OUT")
            go "$TMP/G8_out" "$PS" "${A8[@]}" "${G8[@]}" --gpu-share 0.5; r5=$?
            RUN_ENV=()
            # each run's own start line: its total (own + reserve + 2 lanes + the once part once) against its budget
            read -r Pi Oi Ri OWNi <<< "$(start_of "$TMP/G8_in.log")"
            read -r Po Oo Ro OWNo <<< "$(start_of "$TMP/G8_out.log")"
            awk -v p="${Pi:-x}" -v o="${Oi:-x}" -v r="${Ri:-x}" -v own="${OWNi:-x}" -v b="$B_IN" \
                -v p2="${Po:-x}" -v o2="${Oo:-x}" -v r2="${Ro:-x}" -v own2="${OWNo:-x}" -v b2="$B_OUT" 'BEGIN {
                    if ( ( p o r own p2 o2 r2 own2 ) !~ /^[0-9.]+$/ ) exit 1
                    exit ( own*1e9 + r + 2*(p-o) + o <= b*1e9 && own2*1e9 + r2 + 2*(p2-o2) + o2 > b2*1e9 ) ? 0 : 1 }'
            check "G8 each budget on its side of that run's own start total (own ${OWNi:-?} and ${OWNo:-?} GB, the first run's ${OWN8:-?}; the two checks below rest on this)" $?
            MOi="$(most_of "$TMP/G8_in.log")"
            [ "$r4" = 0 ] && has "$TMP/G8_in.log" "$STARTED" && ! has "$TMP/G8_in.log" "$NOT_STARTED" && same "$TMP/G8_ref" "$TMP/G8_in" \
                && [ "${MOi:-0}" -ge 2 ] 2>/dev/null
            check "G8 a budget of $B_IN GB (own + reserve + 2 lanes + 1.5 times the once part): the dynamic concurrency starts, ${MOi:-?} partitions in flight, the same grids (exit $r4)" $?
            [ "$r5" = 0 ] && has "$TMP/G8_out.log" "$NOT_STARTED" && ! has "$TMP/G8_out.log" "$STARTED" && same "$TMP/G8_ref" "$TMP/G8_out"
            check "G8 a budget of $B_OUT GB (own + reserve + 2 lanes + half the once part): not started, the same grids (exit $r5)" $?
            # the summary's prediction: the plan's plus a lane (not a whole partition) for each partition beyond it
            PRi="$(pred_of "$TMP/G8_in.log")"; PRo="$(pred_of "$TMP/G8_out.log")"
            awk -v a="${PRi:-x}" -v b="${PRo:-x}" -v m="${MOi:-x}" -v p="$P8" -v o="$O8" 'BEGIN {
                    if ( ( a b m ) !~ /^[0-9.]+$/ || m < 2 ) exit 1
                    d = ( a - b ) - ( m - 1 ) * ( p - o ) / 1e9
                    exit ( d < 0.015 && d > -0.015 ) ? 0 : 1 }'
            check "G8 the summary predicts ${PRi:-?} GB with ${MOi:-?} in flight against ${PRo:-?} GB at the plan's 1: a lane for each partition beyond the plan's" $?
        else
            RUN_ENV=()
            check "G8 the budget pair (no start line to read the bytes from)" 1
        fi
    fi
fi

echo ""
echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ]
