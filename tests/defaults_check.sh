#!/usr/bin/env bash
# The programs' own choice of the device: '--gpu', DTFE_GPU (1 = GPU, 0 = CPU, unset/empty/'auto' = the program picks),
# the run log's device line, the checkpoint key's device line, the header's padding, PS-DTFE's default GPU share and the
# 2^32-cell grid limit of the GPU kernels. Other defaults: tests/run_mode_check.sh, tweb_check.sh, ps_2d_check.sh.
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
# what the cases set themselves: the thread count, the checkpoint switches and the GPU cell cap's test hook (an
# export from a login shell would steer every run)
unset OMP_NUM_THREADS OMP_THREAD_LIMIT DTFE_CHECKPOINT DTFE_TEST_EXIT_AFTER_PARTITIONS DTFE_TEST_GPU_CELL_CAP
# ... and what would move the GPU share's model or the plan the share checks read (S)
unset DTFE_MASS_AUDIT DTFE_MEM_BUDGET_GB DTFE_MEM_FRACTION
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 16 --box 100 --crossed-waves --seed 7 >/dev/null \
    || { echo "FAIL data gen"; exit 1; }
# check / check_result, has, same, key_of, mode_of, share_of: tests/lib.sh

# ---- the builds: a GPU path where the object folder carries the Metal stamp (the 3D builds the suite runs)
OBJ="$DTFE_OBJ"; OBJ_PS="$PS_OBJ"
GPU_STD=0; [ -d "$OBJ" ] && [ ! -f "$OBJ/.gpu_mode_off" ] && [ -n "$(gpu_backend_of "$OBJ")" ] && GPU_STD=1
GPU_PS=0; [ -d "$OBJ_PS" ] && [ ! -f "$OBJ_PS/.gpu_mode_off" ] && [ -n "$(gpu_backend_of "$OBJ_PS")" ] && GPU_PS=1
echo ">> GPU path (the Metal stamp of $OBJ / $OBJ_PS): standard DTFE $GPU_STD, PS-DTFE $GPU_PS"

# go <out> <binary> <args...>: one run of '<binary> <snapshot> <out> <args...>', stdout and stderr in <out>.log, with
# DTFE_GPU unset and then the NAME=value list RUN_ENV (DTFE_GPU=<value> where a case gives it).
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
# pad_of <log>: the value of the header's 'number padding particles' line (mode_of, share_of: tests/lib.sh)
pad_of()  { LC_ALL=C sed -n 's/^[[:space:]]*number padding particles: *//p' "$1" | head -n 1; }
# note_of <log>: the partition loop's 'PS-DTFE: GPU share' note; tune_of: the planner's AUTO-TUNE line of its split;
# rep_of: the AUTO-TUNE-REPORT line
note_of()  { LC_ALL=C grep -a '^PS-DTFE: GPU share ' "$1" | head -n 1; }
tune_of()  { LC_ALL=C grep -a '^AUTO-TUNE: .* -> --partition' "$1" | head -n 1; }
rep_of()   { LC_ALL=C grep -a '^AUTO-TUNE-REPORT ' "$1" | head -n 1; }
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
BN="Metal"   # GPU_BACKEND_NAME of the only GPU backend
NOGPU="CPU (this binary was built without GPU support)"
NOPATH="CPU (the default: this average has no GPU path)"
NOPATH_PS="CPU (the default: this deposit has no GPU path)"
SAMPLED_STD="CPU (the default for the sampled average; --gpu runs it on the GPU)"
SHARE_AUTO="auto (the run mode's share, resolved with the plan)"
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
    L7="$NOGPU"; [ "$g" = 1 ] && L7="$NOPATH"; [ "$g$b" = 1PS ] && L7="$NOPATH_PS"
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
    if [ "$b" = STD ]; then
        [ "$r8" = 0 ] && [ "$n8" = 0 ] && ! dispatched "$TMP/D8_$b.log" && quiet "$TMP/D8_$b.log"
        check "D8 $b: no '_a' field, nothing given: no device line (${n8:-?}), no GPU deposit, no GPU warning (exit $r8)" $?
    elif [ "$g" = 1 ]; then
        [ "$r8" = 0 ] && is_dev "$TMP/D8_$b.log" "$LP" && on_gpu "$TMP/D8_$b.log"
        check "D8 $b: no '_a' field, nothing given: the deposit picks the GPU as for an '_a' field: '$LP', on the GPU without a word (exit $r8)" $?
    else
        [ "$r8" = 0 ] && is_dev "$TMP/D8_$b.log" "$LP" && ! dispatched "$TMP/D8_$b.log" && quiet "$TMP/D8_$b.log"
        check "D8 $b: no '_a' field, nothing given, a build without a GPU path: '$LP', no GPU deposit, no warning (exit $r8)" $?
    fi
    want=15; [ "$b" = PS ] && want=5
    got="$(pad_of "$TMP/D4_$b.log")"
    [ "$got" = "$want" ]
    check "D9 $b: the header's 'number padding particles: ${got:-none}' ($want, the binary's default)" $?
    # the header's GPU share line without '--gpu-share': where a given 'auto' is resolved (PS-DTFE's exact deposit on
    # the GPU of a Metal build), nowhere else
    wshare=""; [ "$b" = PS ] && [ "$g" = 1 ] && wshare="$SHARE_AUTO"
    sbad=""
    for t in D1 D2 D3 D5 D5a D5e; do [ "$(share_of "$TMP/${t}_$b.log")" = "$wshare" ] || sbad="$sbad $t"; done
    for t in D4 D4o D6 D6c D7 D8; do [ -z "$(share_of "$TMP/${t}_$b.log")" ] || sbad="$sbad $t"; done
    [ -z "$sbad" ]
    check "D10 $b: the header's GPU share line: '${wshare:-none}' for the exact deposit on the GPU (D1-D3, D5), none in D4, D6-D8${sbad:+; differ:$sbad}" $?
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
    [ "$r0" = 0 ] && [ "$r1" = 137 ] && [ "$after" = device=cpu ] && grep -qx 'average=exact' <<< "$KEY"
    check "K1 $b: killed after 3 partitions under DTFE_GPU=0 (exit $r1; the run's $r0): the key's line after 'gradient=' is '${after:-none}' (device=cpu); average=exact" $?
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
    # K5: the sampled average's key, selected by the count alone (no '--sampled')
    K5O=("${B0[@]}" --partition 2 2 2 --max-concurrent 2 --subsamples 2 --lambda-th 0.3 --no-overlap)
    if [ "$b" = PS ]; then K5O+=(--ps-lagrangian-padding 8 --ps-merge index-order --field density_a velocity_a dispersion_a)
    else K5O+=(--field density_a velocity_a); fi
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_GPU=0 DTFE_TEST_EXIT_AFTER_PARTITIONS=3); go "$TMP/K5_$b" "$bin" "${K5O[@]}"; r5s=$?
    RUN_ENV=()
    KEY5="$(key_of "$TMP/K5_$b")"
    [ -n "$KEY5" ] || no_checkpoint_why "$TMP/K5_$b.log" "$TMP"
    [ "$r5s" = 137 ] && grep -q '^opt ' <<< "$KEY5" && grep -qx 'average=sampled' <<< "$KEY5"
    check "K5 $b: a sampled run ('--subsamples 2', no flag) killed after 3 partitions (exit $r5s): its key's 'average=' line is 'average=sampled'" $?
done

# ---- S: PS-DTFE's GPU share without '--gpu-share' is '--gpu-share auto'
SX=("${B0[@]}" --field density_a velocity_a --exact --run-mode performance)
echo ">> S: PS, the GPU share without '--gpu-share' (GPU path: $GPU_PS)"
if [ "$GPU_PS" != 1 ]; then
    echo "  SKIP S1-S2 PS: no GPU path in this build (the share is the GPU deposit's; D10: no share line)"
else
    RUN_ENV=(OMP_NUM_THREADS=10 DTFE_AUTO_MINN=1000 DTFE_RAM_GB=64)
    go "$TMP/S1_none" "$PS" "${SX[@]}" --auto-tune-report; r0=$?
    go "$TMP/S1_auto" "$PS" "${SX[@]}" --gpu-share auto --auto-tune-report; ra=$?
    go "$TMP/S1_one" "$PS" "${SX[@]}" --gpu-share 1 --auto-tune-report; r1=$?
    RUN_ENV=()
    t0="$(tune_of "$TMP/S1_none.log")"; t1="$(tune_of "$TMP/S1_one.log")"
    [ "$r0$ra$r1" = 000 ] && [ -n "$t0" ] && [ "$t0" = "$(tune_of "$TMP/S1_auto.log")" ] \
        && [ -n "$(rep_of "$TMP/S1_none.log")" ] && [ "$(rep_of "$TMP/S1_none.log")" = "$(rep_of "$TMP/S1_auto.log")" ] \
        && [ "$(share_of "$TMP/S1_none.log")" = "$SHARE_AUTO" ] && [ "$(share_of "$TMP/S1_auto.log")" = "$SHARE_AUTO" ] \
        && [ "$(share_of "$TMP/S1_one.log")" = "1 (--gpu-share)" ]
    rc=$?
    if precision_double; then   # the double build's auto share is 1 in every run mode: the plan of '--gpu-share 1'
        [ "$rc" = 0 ] && [ "$t0" = "$t1" ] && case "$t0" in *"GPU share"*) false ;; *", GPU deposit"*) true ;; *) false ;; esac
    else                        # performance at 10 threads: the free cores deposit beside the GPU
        [ "$rc" = 0 ] && [ "$t0" != "$t1" ] \
            && case "$t0" in *"GPU deposit with the free cores (GPU share "*) true ;; *) false ;; esac
    fi
    rc=$?
    [ "$rc" = 0 ] || { echo "    no option         : '$t0'"; echo "    '--gpu-share 1'   : '$t1'"; }
    check "S1 PS: no '--gpu-share' plans as '--gpu-share auto' (the header's share line, the AUTO-TUNE and report lines) (exit $r0, $ra, $r1)" $rc
    RUN_ENV=(OMP_NUM_THREADS=10 DTFE_AUTO_MINN=1000)
    go "$TMP/S2_none" "$PS" "${SX[@]}" --partition 3 3 3 --max-concurrent 2; r0=$?
    go "$TMP/S2_auto" "$PS" "${SX[@]}" --partition 3 3 3 --max-concurrent 2 --gpu-share auto; ra=$?
    RUN_ENV=()
    n0="$(note_of "$TMP/S2_none.log")"
    [ "$r0$ra" = 00 ] && [ -n "$n0" ] && [ "$n0" = "$(note_of "$TMP/S2_auto.log")" ] \
        && case "$n0" in "PS-DTFE: GPU share "*" (--gpu-share auto: performance"*) true ;; *) false ;; esac \
        && ! grep -aqE "$FALLBACK" "$TMP/S2_none.log" "$TMP/S2_auto.log" && same "$TMP/S2_auto" "$TMP/S2_none"
    check "S2 PS: a run without '--gpu-share' is '--gpu-share auto': the loop note '${n0%% (*}', every output byte for byte, no fall-back (exit $r0, $ra)" $?
fi

# ---- C: a full grid above 2^32 cells is picked for the GPU; the passes test their own grids against the cap
CG=(--input-format 105 --mpc-unit 1 --periodic --grid 2048 --verbose 2 --field density_a velocity_a --auto-tune-report)
CAP=4294967296   # 2^32: the GPU kernels number a sub-grid's cells with 32 bits
WARN32="above the 2^32 the GPU kernels number"
for b in STD PS; do
    if [ "$b" = STD ]; then bin="$STD"; g="$GPU_STD"; pick="$BN GPU (the default for the exact average; DTFE_GPU=0 keeps the run on the CPU)"
    else bin="$PS"; g="$GPU_PS"; pick="$BN GPU (the default; DTFE_GPU=0 keeps the run on the CPU)"; fi
    echo ">> C: $b, a 2048^3 grid planned (GPU path: $g)"
    if [ "$g" != 1 ]; then
        echo "  SKIP C1-C2 $b: no GPU path in this build (the pick is the CPU: D5)"
        continue
    fi
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1)
    go "$TMP/C1_$b" "$bin" "${CG[@]}"; r0=$?
    go "$TMP/C1g_$b" "$bin" "${CG[@]}" --gpu; rg=$?
    RUN_ENV=()
    rep="$(rep_of "$TMP/C1_$b.log")"
    [ "$r0$rg" = 00 ] && is_dev "$TMP/C1_$b.log" "$pick" && is_dev "$TMP/C1g_$b.log" "$BN GPU (--gpu)" \
        && buffers "$TMP/C1_$b.log" yes && [ -n "$rep" ] && [ "$rep" = "$(rep_of "$TMP/C1g_$b.log")" ]
    rc=$?
    if [ "$b" = STD ]; then
        check "C1 $b: '--grid 2048', nothing given: the GPU ('$pick'), planned as '--gpu' is (exit $r0, $rg)" $rc
        continue
    fi
    np="$(LC_ALL=C sed -n 's/^AUTO-TUNE-REPORT partition=\([0-9]*\) .*/\1/p' "$TMP/C1_$b.log" | head -n 1)"
    sg="$(LC_ALL=C sed -n 's/^AUTO-TUNE-DEBUG PS booked sub-grid: \([0-9]*\) cells.*/\1/p' "$TMP/C1_$b.log" | tail -n 1)"
    [ "$rc" = 0 ] && [ "${np:-0}" -ge 2 ] && [ -n "$sg" ] && [ "$sg" -lt "$CAP" ]
    check "C1 $b: '--grid 2048', nothing given: the GPU, planned as '--gpu' is: --partition ${np:-?}, ${sg:-?} cells booked per sub-grid, under 2^32 (exit $r0, $rg)" $?
    RUN_ENV=(OMP_NUM_THREADS=2 DTFE_RAM_GB=64)
    go "$TMP/C2_$b" "$bin" "${CG[@]}" --partition 1 1 1; r0=$?
    go "$TMP/C2g_$b" "$bin" "${CG[@]}" --partition 1 1 1 --gpu; rg=$?
    RUN_ENV=()
    [ "$r0$rg" = 00 ] && is_dev "$TMP/C2_$b.log" "$pick" && has "$TMP/C2_$b.log" "$WARN32" && has "$TMP/C2g_$b.log" "$WARN32"
    check "C2 $b: '--grid 2048 --partition 1 1 1' (above 2^32 cells): the planner's warning for the picked GPU as for '--gpu' (exit $r0, $rg)" $?
done

# ---- G: the passes' own guard of the GPU cell cap, the cap lowered to one cell (DTFE_TEST_GPU_CELL_CAP=1)
WCAP_STD="the GPU interpolation does not support 2^32 or more grid cells in one pass; using the CPU interpolation."
WCAP_PS="the GPU deposit does not support 2^32 or more cells per partition sub-grid (this one has "
MEANT="were meant for the GPU and ran on the CPU instead"
# capped <log> <warning> <count line>: the pass's warning and the closing count are in the log, no dispatch line
capped() {
    has "$1" "$2" || { echo "    ${1##*/}: no '$2' warning"; return 1; }
    has "$1" "$3" || { echo "    ${1##*/}: no '$3' line"; return 1; }
    ! dispatched "$1" || { echo "    ${1##*/}: a deposit ran on the GPU"; return 1; }
}
for b in STD PS; do
    if [ "$b" = STD ]; then
        bin="$STD"; g="$GPU_STD"; wcap="$WCAP_STD"; cnt="1 sub-domain deposit(s) $MEANT"
        GA=("${B0[@]}" --exact --partition 1 --field density_a velocity_a)
    else
        bin="$PS"; g="$GPU_PS"; wcap="$WCAP_PS"; cnt="8 partition deposit pass(es) $MEANT"
        GA=("${B0[@]}" --exact --partition 2 2 2 --max-concurrent 1 --no-overlap --gpu-share 1 --field density_a velocity_a)
    fi
    echo ">> G: $b, every pass above a GPU cell cap of one cell (GPU path: $g)"
    if [ "$g" != 1 ]; then
        echo "  SKIP G1-G2 $b: no GPU path in this build (no pass is meant for the GPU)"
        continue
    fi
    RUN_ENV=(OMP_NUM_THREADS=1 DTFE_TEST_GPU_CELL_CAP=1 DTFE_GPU=0); go "$TMP/G0_$b" "$bin" "${GA[@]}"; r0=$?
    RUN_ENV=(OMP_NUM_THREADS=1 DTFE_TEST_GPU_CELL_CAP=1);            go "$TMP/G1_$b" "$bin" "${GA[@]}" --gpu; r1=$?
                                                                    go "$TMP/G2_$b" "$bin" "${GA[@]}"; r2=$?
    RUN_ENV=()
    [ "$r0$r1" = 00 ] && quiet "$TMP/G0_$b.log" && capped "$TMP/G1_$b.log" "$wcap" "$cnt" && same "$TMP/G0_$b" "$TMP/G1_$b"
    check "G1 $b: '--gpu', every pass above the cap: its warning, '$cnt', no GPU deposit, the outputs of DTFE_GPU=0 byte for byte (exit $r1, $r0)" $?
    [ "$r0$r2" = 00 ] && quiet "$TMP/G0_$b.log" && capped "$TMP/G2_$b.log" "$wcap" "$cnt" && same "$TMP/G0_$b" "$TMP/G2_$b"
    check "G2 $b: nothing given (the picked GPU), every pass above the cap: the same (exit $r2, $r0)" $?
done

# ---- A: the '_a' average without a flag, under DTFE_EXACT_AVERAGE and with a count, and the header line naming it
SNAP_A="$TMP/waves24.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP_A" --n 24 --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 \
    --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen (A)"; exit 1; }
# ga <out> <binary> [NAME=value ...] -- <args...>: one run on SNAP_A on the CPU (DTFE_GPU=0), two threads, the NAME=value
# list in its environment; stdout and stderr in <out>.log; the exit status is the binary's
ga() {
    local out="$1" bin="$2" e=()
    shift 2
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do e+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    env OMP_NUM_THREADS=2 DTFE_GPU=0 ${e[@]+"${e[@]}"} "$bin" "$SNAP_A" "$out" "$@" > "$out.log" 2>&1
}
for b in STD PS; do
    bin="$STD"; XL="volume averaging       : EXACT cell-tetrahedron integration of the linear interpolant ("
    SL="volume averaging       : SAMPLED ("; CL="a sampling count or method was given; --exact integrates the cells exactly"
    if [ "$b" = PS ]; then
        bin="$PS"; XL="grid deposit           : EXACT tetrahedron-cell intersection moments ("
        SL="grid deposit           : SAMPLED on nSub^3 sub-points ("; CL="a --subsamples count was given; --exact deposits every tetrahedron exactly"
    fi
    echo ">> A: $b, the '_a' average"
    M=(--input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 1 --field density_a velocity_a)
    o="$TMP/A_$b"
    ga "$o.x" "$bin" -- "${M[@]}" --exact; r1=$?
    ga "$o.s" "$bin" -- "${M[@]}" --sampled; r2=$?
    ga "$o.d" "$bin" -- "${M[@]}"; r3=$?
    [ "$r1$r2$r3" = 000 ] && has "$o.x.log" "${XL}--exact; " && has "$o.s.log" "${SL}--sampled)" \
        && has "$o.d.log" "${XL}--exact, the default; " && same "$o.x" "$o.d" && ! cmp -s "$o.x.a_den" "$o.s.a_den"
    check "A1 $b: no flag runs the exact average ('--exact, the default'), the outputs of '--exact'; '--sampled' samples (exit $r1, $r2, $r3)" $?
    if [ "$b" = PS ]; then   # PS-DTFE's sampled deposit without a count: one point per cell axis
        ga "$o.s1" "$bin" -- "${M[@]}" --sampled --subsamples 1; r1=$?
        [ "$r1" = 0 ] && has "$o.s.log" "avg sub-samples (nSub) : 1 " && same "$o.s" "$o.s1"
        check "A2 $b: '--sampled' without a count deposits on 1 point per cell axis (its header's count), the outputs of '--sampled --subsamples 1' (exit $r1)" $?
        # an exact run without a count keeps the count 3 where an exact pass still reads one (src/user_options.cc
        # kPsExactSliverSubsamples), told by its header's 'grid deposit' line
        ga "$o.x3" "$bin" -- "${M[@]}" --exact --subsamples 3; r1=$?
        XC="; count 3 only for a tetrahedron too thin to clip"
        [ "$r1" = 0 ] && same "$o.d" "$o.x3" && has "$o.d.log" "$XC" && has "$o.x3.log" "$XC"
        check "A3 $b: the exact default without a count keeps the count 3 (its header's count, the outputs of '--exact --subsamples 3'; exit $r1)" $?
    fi
    ga "$o.p0" "$bin" DTFE_EXACT_AVERAGE=0 -- "${M[@]}"; r1=$?
    ga "$o.fx" "$bin" DTFE_EXACT_AVERAGE=0 -- "${M[@]}" --exact; r2=$?
    ga "$o.p1" "$bin" DTFE_EXACT_AVERAGE=1 -- "${M[@]}"; r3=$?
    ga "$o.fs" "$bin" DTFE_EXACT_AVERAGE=1 -- "${M[@]}" --sampled; r4=$?
    [ "$r1$r3" = 00 ] && has "$o.p0.log" "${SL}--sampled from DTFE_EXACT_AVERAGE=0)" && same "$o.s" "$o.p0" \
        && has "$o.p1.log" "${XL}--exact from DTFE_EXACT_AVERAGE=1; " && same "$o.x" "$o.p1"
    check "A4 $b: DTFE_EXACT_AVERAGE=0 alone samples, =1 alone integrates exactly, each said in the header (exit $r1, $r3)" $?
    [ "$r2$r4" = 00 ] && has "$o.fx.log" "${XL}--exact; " && same "$o.x" "$o.fx" \
        && has "$o.fs.log" "${SL}--sampled)" && same "$o.s" "$o.fs"
    check "A5 $b: a flag beats the pin both ways (DTFE_EXACT_AVERAGE=0 with --exact, =1 with --sampled; exit $r2, $r4)" $?
    ga "$o.c" "$bin" -- "${M[@]}" --subsamples 2; r1=$?
    ga "$o.cs" "$bin" -- "${M[@]}" --sampled --subsamples 2; r2=$?
    [ "$r1$r2" = 00 ] && has "$o.c.log" "${SL}${CL})" && same "$o.cs" "$o.c"
    check "A6 $b: a '--subsamples' count alone selects the sampled average, said in the header (exit $r1, $r2)" $?
    if [ "$b" = STD ]; then   # an '--average-method' of any value is a sampling request too, 1 (the default method) included
        ga "$o.m1" "$bin" -- "${M[@]}" --average-method 1; r1=$?
        [ "$r1" = 0 ] && has "$o.m1.log" "${SL}${CL})" && same "$o.s" "$o.m1"
        check "A7 $b: '--average-method 1' alone selects the sampled average, the outputs of '--sampled' (exit $r1)" $?
        # a redshift cone grid has no exact form (its '_a' fields are sampled, method 2): '--exact' warns and
        # samples, the default samples without a word; both give the outputs of '--sampled'. The seed is pinned:
        K=(--input-format 105 --mpc-unit 1 --periodic --grid 4 4 6 --redshift-cone 5 40 30 150 0 300
           --cone-origin 50 50 50 --partition 1 --field density_a velocity_a --seed 7)
        KW="'--exact' does not apply to a redshift cone grid"; KS="${SL}a redshift cone grid has no exact form)"
        ga "$o.kx" "$bin" -- "${K[@]}" --exact; r1=$?
        ga "$o.ks" "$bin" -- "${K[@]}" --sampled; r2=$?
        ga "$o.kd" "$bin" -- "${K[@]}"; r3=$?
        [ "$r1$r2$r3" = 000 ] && has "$o.kx.log" "$KW" && ! has "$o.ks.log" "$KW" && ! has "$o.kd.log" "$KW" \
            && has "$o.kx.log" "$KS" && has "$o.kd.log" "$KS" && has "$o.ks.log" "${SL}--sampled)" \
            && same "$o.kx" "$o.ks" && same "$o.kd" "$o.ks"
        check "A8 $b: on a redshift cone grid '--exact' warns and samples, no flag samples silently, both the outputs of '--sampled' (exit $r1, $r2, $r3)" $?
    fi
    ga "$o.xs" "$bin" -- "${M[@]}" --exact --sampled; r1=$?
    [ "$r1" = 1 ] && has "$o.xs.log" "'--exact' and '--sampled' exclude each other" && [ ! -e "$o.xs.a_den" ]
    check "A9 $b: '--exact --sampled' is refused at option parsing, naming both (exit $r1)" $?
done

echo ""
check_result
