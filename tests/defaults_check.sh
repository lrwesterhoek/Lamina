#!/usr/bin/env bash
# The programs' own choice of the device: '--gpu', the environment's DTFE_GPU (1 = as '--gpu', 0 = the CPU; unset,
# empty or 'auto' = the program picks: PS-DTFE the GPU for an '_a' deposit, exact or sampled, standard DTFE the GPU for
# the exact average and the CPU for the sampled one, each only where the build and the run have a GPU path), the run
# log's device line that says which and why, the checkpoint key's device line, and the header's padding (standard
# DTFE 15, PS-DTFE 5; src/user_options.cc checkGpuOptions, printOptions; src/partition_checkpoint.h settingsKey).
# Elsewhere: the run mode's default in tests/run_mode_check.sh (M1, M1b, M3), the T-web threshold's in
# tests/tweb_check.sh (L), standard DTFE's padding on a void-rich snapshot in tests/point_exact_serve_check.sh (P3),
# the 2D programs' device line in tests/ps_2d_check.sh (DV).
# tests/lib.sh pins DTFE_GPU=0 and DTFE_RUN_MODE=performance for every suite: each run here unsets DTFE_GPU and sets it
# where a case gives it (the mode stays pinned, so a GPU run's 'run mode' line names 250 ms buffers). The device line
# is the header's "\t '_a' interpolation     : <text>" (standard DTFE) or "\t PS deposit             : <text>"
# (PS-DTFE), compared whole. Both binaries of the suite's precision, on a synthetic 16^3 snapshot of crossed waves,
# grid 16; whether a binary has a GPU path is read from the Metal stamp of its object folder (o / o_ps, o_d / o_ps_d),
# as the other suites do, and 'Metal GPU' is the only backend's name. A deposit ran on the GPU when its dispatch line
# is in the log ('Metal GPU (<device>, N tetrahedra)', 'PS-DTFE: Metal GPU deposit (') and no fall-back line is.
# D: one triangulation ('--partition 1') on one thread, '--field density_a velocity_a':
#   D1  '--gpu': 'Metal GPU (--gpu)', on the GPU; a build without a GPU path: 'CPU (--gpu given but this binary was
#       built without GPU support; falling back)' and the deposit's '--gpu requested but this binary was built without
#       GPU support' warning
#   D2  DTFE_GPU=1 is '--gpu': 'Metal GPU (DTFE_GPU=1)', on the GPU (elsewhere the CPU line and the warning, naming
#       'DTFE_GPU=1'), and every output of D1 byte for byte
#   D3  '--gpu' beats DTFE_GPU=0: D1's line and D1's outputs
#   D4  DTFE_GPU=0 and DTFE_GPU=OFF (the value as given): 'CPU (DTFE_GPU=0)' and 'CPU (DTFE_GPU=OFF)', no dispatch
#       line, no GPU warning, no GPU buffers in the 'run mode' line, the same outputs
#   D5  nothing given (DTFE_GPU unset; 'auto' and empty alike): a Metal build picks the GPU for the exact average --
#       standard DTFE 'Metal GPU (the default for the exact average; DTFE_GPU=0 keeps the run on the CPU)', PS-DTFE
#       'Metal GPU (the default; DTFE_GPU=0 keeps the run on the CPU)' -- deposits there ('; GPU buffers up to 250 ms'
#       in the 'run mode' line), warns nothing and writes D1's outputs byte for byte; a build without a GPU path:
#       'CPU (this binary was built without GPU support)', no warning, D4's outputs
#   D6  '--sampled', nothing given: standard DTFE keeps the CPU, on a Metal build 'CPU (the default for the sampled
#       average; --gpu runs it on the GPU)', no dispatch line, the outputs of the DTFE_GPU=0 run; PS-DTFE picks the
#       GPU for it too (D5's line, on the GPU); a build without a GPU path: the 'built without' line; no warning
#   D7  a run without a GPU path, nothing given: standard '--average-method 2' and PS-DTFE '--ps-caustics
#       --ps-caustic-cusps' read 'CPU (the default: this average has no GPU path)' on a Metal build (the 'built
#       without' line elsewhere), no dispatch line, no warning; asked, they warn as '--gpu' does, naming the source:
#       '--gpu accelerates only the method-1 ...' and 'DTFE_GPU=1 accelerates only the method-1 ...' (every build),
#       PS-DTFE '... DTFE_GPU=1 is not used for the deposit of this run.' (a Metal build; elsewhere 'DTFE_GPU=1
#       requested but this binary was built without GPU support')
#   D8  no '_a' field ('--field density'), nothing given: no device line, no dispatch line, no GPU warning
#   D9  the header's padding (D4's runs): 'number padding particles: 15' in standard DTFE, 5 in PS-DTFE
# K: '--exact --partition 2 2 2 --max-concurrent 2' on two threads, '--field density_a velocity_a':
#   K1  a run killed after 3 partitions under DTFE_GPU=0: the line after its key's 'gradient=' line is 'device=cpu'
#   K2  its '--resume' under DTFE_GPU=1 (an asked GPU: 'device=gpu' in every build) is refused, naming
#       'checkpoint: device=cpu' and 'this run  : device=gpu'; the checkpoint is kept
#   K3  a Metal build: its '--resume' with nothing given (the program picks the GPU) is refused likewise
#   K4  the resume that keeps the CPU (DTFE_GPU=0 on a Metal build, nothing given elsewhere) finishes: every output of
#       the uninterrupted DTFE_GPU=0 run byte for byte, the checkpoint removed
# Usage: tests/defaults_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}" "${DTFE_BIN}"
for b in "${PS_BIN}" "${DTFE_BIN}"; do [ -x "$b" ] || { echo "FAIL: $b is not built"; exit 1; }; done
PS="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
STD="$(cd "$(dirname "${DTFE_BIN}")" && pwd)/$(basename "${DTFE_BIN}")"
TMP="${DTFE_TEST_TMP}/defaults_check"
rm -rf "$TMP"; mkdir -p "$TMP"
TMP="$(cd "$TMP" && pwd -P)"
# what the cases set themselves: the thread count and the checkpoint switches (an export from a login shell would
# steer every run)
unset OMP_NUM_THREADS OMP_THREAD_LIMIT DTFE_CHECKPOINT DTFE_TEST_EXIT_AFTER_PARTITIONS
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 16 --box 100 --crossed-waves --seed 7 >/dev/null \
    || { echo "FAIL data gen"; exit 1; }
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }

# ---- the builds: a GPU path where the object folder carries the Metal stamp (the 3D builds the suite runs)
OBJ=o; OBJ_PS=o_ps; precision_double && { OBJ=o_d; OBJ_PS=o_ps_d; }
GPU_STD=0; [ -d "$OBJ" ] && [ ! -f "$OBJ/.gpu_mode_off" ] && [ -n "$(gpu_backend_of "$OBJ")" ] && GPU_STD=1
GPU_PS=0; [ -d "$OBJ_PS" ] && [ ! -f "$OBJ_PS/.gpu_mode_off" ] && [ -n "$(gpu_backend_of "$OBJ_PS")" ] && GPU_PS=1
echo ">> GPU path (the Metal stamp of $OBJ / $OBJ_PS): standard DTFE $GPU_STD, PS-DTFE $GPU_PS"

# go <out> <binary> <args...>: one run of '<binary> <snapshot> <out> <args...>', stdout and stderr in <out>.log, with
# DTFE_GPU unset and then the NAME=value list RUN_ENV (DTFE_GPU=<value> where a case gives it); the exit status is the
# binary's (the shell's own 'Killed: 9' line for a DTFE_TEST_EXIT_AFTER_PARTITIONS run is not output)
RUN_ENV=()
go() {
    local out="$1" bin="$2"; shift 2
    { env -u DTFE_GPU ${RUN_ENV[@]+"${RUN_ENV[@]}"} "$bin" "$SNAP" "$out" "$@" > "$out.log" 2>&1; } 2>/dev/null
}
# dev_of <log>: the device line's text after its label (the last one: a later line replaces the header's); ndev: how many
dev_of() { LC_ALL=C sed -n -e "s/^[[:space:]]*'_a' interpolation *: //p" -e 's/^[[:space:]]*PS deposit *: //p' "$1" | tail -n 1; }
ndev()   { LC_ALL=C grep -acE "^[[:space:]]*('_a' interpolation|PS deposit) *: " "$1"; }
# is_dev <log> <text>: the device line reads exactly that (else both are printed)
is_dev() {
    local got; got="$(dev_of "$1")"
    [ "$got" = "$2" ] && return 0
    echo "    ${1##*/}: device : '$got'"; echo "    expected : '$2'"; return 1
}
# mode_of <log>: the value of the header's 'run mode' line; pad_of <log>: of its 'number padding particles' line
mode_of() { LC_ALL=C sed -n 's/^[[:space:]]*run mode *: //p' "$1" | head -n 1; }
pad_of()  { LC_ALL=C sed -n 's/^[[:space:]]*number padding particles: *//p' "$1" | head -n 1; }
# has <log> <text>: the literal text is in the log
has() { grep -aqF -- "$2" "$1"; }
# the deposits' dispatch lines (standard DTFE, PS-DTFE one domain); the fall-back lines; the warnings a GPU that was
# asked for can give
DISPATCH=' GPU deposit \(| GPU \(.*, [0-9]+ tetrahedra\)'
FALLBACK='using the CPU (deposit|interpolation)|deposit unavailable|ran on the CPU instead|gave up'
GPUWARN='requested but this binary was built without GPU support|accelerates only the method-1|is not used for the deposit|falling back'
dispatched() { grep -aqE "$DISPATCH" "$1"; }
quiet()      { ! grep -aqE "$FALLBACK|$GPUWARN" "$1"; }
on_gpu()     { dispatched "$1" && quiet "$1"; }
# asked <log> <source> <GPU path>: an asked GPU's deposit -- on the GPU where the build has a path, else on the CPU
# with the deposit's warning naming the source
asked() {
    if [ "$3" = 1 ]; then on_gpu "$1"
    else has "$1" "$2 requested but this binary was built without GPU support" && ! dispatched "$1"; fi
}
# buffers <log> <yes|no>: the log has a 'run mode' line, and it names the GPU buffers (250 ms: the pinned
# performance's) or none
buffers() {
    local m; m="$(mode_of "$1")"
    if [ -n "$m" ]; then
        case "$2" in
            yes) case "$m" in *"; GPU buffers up to 250 ms"*) return 0 ;; esac ;;
            *)   case "$m" in *"GPU buffers"*) ;; *) return 0 ;; esac ;;
        esac
    fi
    echo "    ${1##*/}: run mode : '$m'"; return 1
}
# key_of <output>: the settings key block of its checkpoint (the header is text; the grids follow it)
key_of() { LC_ALL=C sed -n '/^key-begin$/,/^key-end$/p;/^header-end$/q' "$1.runlog.checkpoint/checkpoint" 2>/dev/null; }
# same <a> <b>: every output of run a has a byte-identical twin in run b, and there is at least one; the logs, the
# run's bookkeeping and the padding test's '.badGridPoints*' (written in place, the last sub-domain's) are not outputs
same() {
    local n=0 f
    for f in "$1".*; do
        case "$f" in *.log|*.runlog*|*.part.*|*.badGridPoints*) continue ;; esac
        cmp -s "$f" "$2${f#"$1"}" || { echo "    differs: ${f#"$1"}"; return 1; }
        n=$((n+1))
    done
    [ "$n" -gt 0 ]
}
BN="Metal"   # GPU_BACKEND_NAME of the only GPU backend
NOGPU="CPU (this binary was built without GPU support)"
NOPATH="CPU (the default: this average has no GPU path)"
SAMPLED_STD="CPU (the default for the sampled average; --gpu runs it on the GPU)"
asked_cpu() { echo "CPU ($1 given but this binary was built without GPU support; falling back)"; }
M2="accelerates only the method-1 ('--average-method 1') volume-averaged interpolation; method 2 will run on the CPU."
CUSPS="--ps-caustic-cusps needs the CPU deposit (its stencil gathers the triangulation's neighbours); "

B0=(--input-format 105 --mpc-unit 1 --periodic --grid 16 --verbose 2)
D=("${B0[@]}" --partition 1 --field density_a velocity_a)
for b in STD PS; do
    if [ "$b" = STD ]; then bin="$STD"; g="$GPU_STD"; pick="$BN GPU (the default for the exact average; DTFE_GPU=0 keeps the run on the CPU)"
    else bin="$PS"; g="$GPU_PS"; pick="$BN GPU (the default; DTFE_GPU=0 keeps the run on the CPU)"; fi
    if [ "$g" = 1 ]; then L1="$BN GPU (--gpu)"; L2="$BN GPU (DTFE_GPU=1)"; LP="$pick"
    else L1="$(asked_cpu --gpu)"; L2="$(asked_cpu DTFE_GPU=1)"; LP="$NOGPU"; fi
    echo ">> D: $b, the device (GPU path: $g)"
    RUN_ENV=(OMP_NUM_THREADS=1);               go "$TMP/D1_$b"  "$bin" "${D[@]}" --gpu; r1=$?
    RUN_ENV=(OMP_NUM_THREADS=1 DTFE_GPU=1);    go "$TMP/D2_$b"  "$bin" "${D[@]}"; r2=$?
    RUN_ENV=(OMP_NUM_THREADS=1 DTFE_GPU=0);    go "$TMP/D3_$b"  "$bin" "${D[@]}" --gpu; r3=$?
                                               go "$TMP/D4_$b"  "$bin" "${D[@]}"; r4=$?
                                               go "$TMP/D6c_$b" "$bin" "${D[@]}" --sampled; r6c=$?
    RUN_ENV=(OMP_NUM_THREADS=1 DTFE_GPU=OFF);  go "$TMP/D4o_$b" "$bin" "${D[@]}"; r4o=$?
    RUN_ENV=(OMP_NUM_THREADS=1);               go "$TMP/D5_$b"  "$bin" "${D[@]}"; r5=$?
                                               go "$TMP/D6_$b"  "$bin" "${D[@]}" --sampled; r6=$?
                                               go "$TMP/D8_$b"  "$bin" "${B0[@]}" --partition 1 --field density; r8=$?
    RUN_ENV=(OMP_NUM_THREADS=1 DTFE_GPU=auto); go "$TMP/D5a_$b" "$bin" "${D[@]}"; r5a=$?
    RUN_ENV=(OMP_NUM_THREADS=1 DTFE_GPU=);     go "$TMP/D5e_$b" "$bin" "${D[@]}"; r5e=$?
    RUN_ENV=()

    [ "$r1" = 0 ] && is_dev "$TMP/D1_$b.log" "$L1" && asked "$TMP/D1_$b.log" --gpu "$g"
    check "D1 $b: '--gpu' reads '$L1' (exit $r1)" $?
    [ "$r2" = 0 ] && is_dev "$TMP/D2_$b.log" "$L2" && asked "$TMP/D2_$b.log" DTFE_GPU=1 "$g" && same "$TMP/D1_$b" "$TMP/D2_$b"
    check "D2 $b: DTFE_GPU=1 is '--gpu': '$L2', the outputs of '--gpu' byte for byte (exit $r2)" $?
    [ "$r3" = 0 ] && is_dev "$TMP/D3_$b.log" "$L1" && asked "$TMP/D3_$b.log" --gpu "$g" && same "$TMP/D1_$b" "$TMP/D3_$b"
    check "D3 $b: '--gpu' beats DTFE_GPU=0: '$L1', the outputs of '--gpu' (exit $r3)" $?
    [ "$r4$r4o" = 00 ] && is_dev "$TMP/D4_$b.log" "CPU (DTFE_GPU=0)" && is_dev "$TMP/D4o_$b.log" "CPU (DTFE_GPU=OFF)" \
        && ! dispatched "$TMP/D4_$b.log" && ! dispatched "$TMP/D4o_$b.log" && quiet "$TMP/D4_$b.log" && quiet "$TMP/D4o_$b.log" \
        && buffers "$TMP/D4_$b.log" no && same "$TMP/D4_$b" "$TMP/D4o_$b"
    check "D4 $b: DTFE_GPU=0 and DTFE_GPU=OFF keep the CPU ('CPU (DTFE_GPU=0)', 'CPU (DTFE_GPU=OFF)'), silently, the same outputs (exit $r4, $r4o)" $?
    if [ "$g" = 1 ]; then
        [ "$r5$r5a$r5e" = 000 ] && is_dev "$TMP/D5_$b.log" "$LP" && is_dev "$TMP/D5a_$b.log" "$LP" && is_dev "$TMP/D5e_$b.log" "$LP" \
            && on_gpu "$TMP/D5_$b.log" && buffers "$TMP/D5_$b.log" yes && same "$TMP/D1_$b" "$TMP/D5_$b"
        check "D5 $b: nothing given (DTFE_GPU unset, auto, empty) picks the GPU: '$LP', on the GPU without a word, the outputs of '--gpu' byte for byte (exit $r5, $r5a, $r5e)" $?
    else
        [ "$r5$r5a$r5e" = 000 ] && is_dev "$TMP/D5_$b.log" "$LP" && is_dev "$TMP/D5a_$b.log" "$LP" && is_dev "$TMP/D5e_$b.log" "$LP" \
            && quiet "$TMP/D5_$b.log" && ! dispatched "$TMP/D5_$b.log" && same "$TMP/D4_$b" "$TMP/D5_$b"
        check "D5 $b: nothing given (DTFE_GPU unset, auto, empty) in a build without a GPU path: '$LP', no warning, the outputs of DTFE_GPU=0 (exit $r5, $r5a, $r5e)" $?
    fi
    if [ "$b" = STD ]; then
        L6="$NOGPU"; [ "$g" = 1 ] && L6="$SAMPLED_STD"
        [ "$r6$r6c" = 00 ] && is_dev "$TMP/D6_$b.log" "$L6" && ! dispatched "$TMP/D6_$b.log" && quiet "$TMP/D6_$b.log" \
            && same "$TMP/D6c_$b" "$TMP/D6_$b"
        check "D6 $b: '--sampled' with nothing given keeps the CPU: '$L6', no warning, the outputs of DTFE_GPU=0 (exit $r6, $r6c)" $?
    elif [ "$g" = 1 ]; then
        [ "$r6" = 0 ] && is_dev "$TMP/D6_$b.log" "$LP" && on_gpu "$TMP/D6_$b.log"
        check "D6 $b: '--sampled' with nothing given picks the GPU too: '$LP', on the GPU without a word (exit $r6)" $?
    else
        [ "$r6$r6c" = 00 ] && is_dev "$TMP/D6_$b.log" "$LP" && quiet "$TMP/D6_$b.log" && same "$TMP/D6c_$b" "$TMP/D6_$b"
        check "D6 $b: '--sampled' with nothing given: '$LP', no warning, the outputs of DTFE_GPU=0 (exit $r6, $r6c)" $?
    fi
    L7="$NOGPU"; [ "$g" = 1 ] && L7="$NOPATH"
    if [ "$b" = STD ]; then
        RUN_ENV=(OMP_NUM_THREADS=1);            go "$TMP/D7_$b"  "$bin" "${D[@]}" --average-method 2; r7=$?
                                                go "$TMP/D7g_$b" "$bin" "${D[@]}" --average-method 2 --gpu; r7g=$?
        RUN_ENV=(OMP_NUM_THREADS=1 DTFE_GPU=1); go "$TMP/D7e_$b" "$bin" "${D[@]}" --average-method 2; r7e=$?
        RUN_ENV=()
        [ "$r7$r7g$r7e" = 000 ] && is_dev "$TMP/D7_$b.log" "$L7" && ! dispatched "$TMP/D7_$b.log" && quiet "$TMP/D7_$b.log" \
            && has "$TMP/D7g_$b.log" "--gpu $M2" && has "$TMP/D7e_$b.log" "DTFE_GPU=1 $M2" && ! has "$TMP/D7e_$b.log" "--gpu $M2"
        check "D7 $b: '--average-method 2' with nothing given: '$L7', no warning; asked, '--gpu ...' and 'DTFE_GPU=1 accelerates only the method-1 ...' (exit $r7, $r7g, $r7e)" $?
    else
        W7="DTFE_GPU=1 requested but this binary was built without GPU support"; [ "$g" = 1 ] && W7="${CUSPS}DTFE_GPU=1 is not used for the deposit of this run."
        RUN_ENV=(OMP_NUM_THREADS=1);            go "$TMP/D7_$b"  "$bin" "${D[@]}" --ps-caustics --ps-caustic-cusps; r7=$?
        RUN_ENV=(OMP_NUM_THREADS=1 DTFE_GPU=1); go "$TMP/D7e_$b" "$bin" "${D[@]}" --ps-caustics --ps-caustic-cusps; r7e=$?
        RUN_ENV=()
        [ "$r7$r7e" = 00 ] && is_dev "$TMP/D7_$b.log" "$L7" && ! dispatched "$TMP/D7_$b.log" && quiet "$TMP/D7_$b.log" \
            && has "$TMP/D7e_$b.log" "$W7"
        check "D7 $b: '--ps-caustics --ps-caustic-cusps' with nothing given: '$L7', no warning; under DTFE_GPU=1 the warning naming it (exit $r7, $r7e)" $?
    fi
    n8="$(ndev "$TMP/D8_$b.log")"
    [ "$r8" = 0 ] && [ "$n8" = 0 ] && ! dispatched "$TMP/D8_$b.log" && quiet "$TMP/D8_$b.log"
    check "D8 $b: no '_a' field, nothing given: no device line (${n8:-?}), no GPU deposit, no GPU warning (exit $r8)" $?
    want=15; [ "$b" = PS ] && want=5
    got="$(pad_of "$TMP/D4_$b.log")"
    [ "$got" = "$want" ]
    check "D9 $b: the header's 'number padding particles: ${got:-none}' ($want, the binary's default)" $?
done

# ---- K: the checkpoint key's device line
# refused_dev <log>: a '--resume' refused for the device: the checkpoint's 'device=cpu' against this run's 'device=gpu'
refused_dev() { has "$1" "'--resume' refused" && has "$1" "checkpoint: device=cpu" && has "$1" "this run  : device=gpu"; }
KA=("${B0[@]}" --field density_a velocity_a --exact --partition 2 2 2 --max-concurrent 2)
for b in STD PS; do
    bin="$STD"; g="$GPU_STD"; [ "$b" = PS ] && { bin="$PS"; g="$GPU_PS"; }
    echo ">> K: $b, the checkpoint key's device line (GPU path: $g)"
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_GPU=0);                                  go "$TMP/K_${b}_ref" "$bin" "${KA[@]}"; r0=$?
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_GPU=0 DTFE_TEST_EXIT_AFTER_PARTITIONS=3); go "$TMP/K_$b" "$bin" "${KA[@]}"; r1=$?
    mv "$TMP/K_$b.log" "$TMP/K_$b.killed.log"
    KEY="$(key_of "$TMP/K_$b")"
    [ -n "$KEY" ] || no_checkpoint_why "$TMP/K_$b.killed.log" "$TMP"
    after="$(awk 'p { print; exit } /^gradient=/ { p = 1 }' <<< "$KEY")"
    [ "$r0" = 0 ] && [ "$r1" = 137 ] && [ "$after" = device=cpu ]
    check "K1 $b: killed after 3 partitions under DTFE_GPU=0 (exit $r1; the uninterrupted run's $r0): the key's line after 'gradient=' is '${after:-none}' (device=cpu)" $?
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_GPU=1); go "$TMP/K_$b" "$bin" "${KA[@]}" --resume; r2=$?
    mv "$TMP/K_$b.log" "$TMP/K_$b.asked.log"
    [ "$r2" != 0 ] && refused_dev "$TMP/K_$b.asked.log" && [ -s "$TMP/K_$b.runlog.checkpoint/checkpoint" ]
    check "K2 $b: '--resume' under DTFE_GPU=1 is refused, naming 'checkpoint: device=cpu' and 'this run  : device=gpu'; the checkpoint is kept (exit $r2)" $?
    if [ "$g" = 1 ]; then
        RUN_ENV=(OMP_NUM_THREADS=2); go "$TMP/K_$b" "$bin" "${KA[@]}" --resume; r3=$?
        mv "$TMP/K_$b.log" "$TMP/K_$b.picked.log"
        [ "$r3" != 0 ] && refused_dev "$TMP/K_$b.picked.log" && [ -s "$TMP/K_$b.runlog.checkpoint/checkpoint" ]
        check "K3 $b: '--resume' with nothing given (the program picks the GPU) is refused likewise (exit $r3)" $?
        RUN_ENV=(OMP_NUM_THREADS=2 DTFE_GPU=0); keep="DTFE_GPU=0"
    else
        echo "  SKIP K3 $b: no GPU path in this build, so the program picks the CPU (K4 resumes with nothing given)"
        RUN_ENV=(OMP_NUM_THREADS=2); keep="nothing given"
    fi
    go "$TMP/K_$b" "$bin" "${KA[@]}" --resume; r4=$?
    RUN_ENV=()
    [ "$r4" = 0 ] && has "$TMP/K_$b.log" "--resume: " && same "$TMP/K_${b}_ref" "$TMP/K_$b" && [ ! -e "$TMP/K_$b.runlog.checkpoint" ]
    check "K4 $b: the resume with $keep finishes (exit $r4): every output of the uninterrupted run, the checkpoint removed" $?
done

echo ""
echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ]
