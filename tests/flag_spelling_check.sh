#!/usr/bin/env bash
# The one flag scheme (docs/cleanup-plan.md P6.7): a run spelled the OLD way (the deprecated options, the merge
# switches, standard DTFE's '-m'/'-s', the renamed environment knobs) is the same run as the canonical spelling, in
# both binaries. For each pair of command lines (old, new) on small synthetic snapshots:
#   - the old run warns once per deprecated spelling ("warning: --X is deprecated, use --Y"), the new one never;
#   - every output of the old run is byte-identical to the new run's (the logs and run logs aside);
#   - for a partitioned run, a run killed after a few partitions (DTFE_TEST_EXIT_AFTER_PARTITIONS) leaves the SAME
#     checkpoint settings key under both spellings (the 'key-begin' .. 'key-end' block of '<output>.runlog.checkpoint/
#     checkpoint', compared line for line), and the old run's checkpoint resumes under the new spelling to the
#     uninterrupted run's outputs, byte for byte.
# The pairs:
#   F1  PS-DTFE sampled, '--partition 2 2 2': -i/-p/-g/-f, --MpcUnit, --avg-subsamples, --ps-lag-padding, --lambda_th,
#       --ordered-merge, --ps-stream-density dtfe, --ps-no-overlap, and PS-DTFE's '-m 2 -s 27' (no effect there: a
#       warning each, left out of the key) against --input-format, --mpc-unit, --subsamples, --ps-lagrangian-padding,
#       --lambda-th, --ps-merge index-order, --ps-stream-density linear, --no-overlap; the key names the sampled
#       average (the count selects it under either spelling)
#   F2  PS-DTFE exact deposit, fixed-point merge: --ps-exact-deposit --fixed-point-merge (with the old --ordered-merge
#       beside it, accepted: fixed-point wins) against --exact --ps-merge fixed-point; the key names the merge
#       (the double pair, which refuses the fixed-point merge: --ordered-merge against --ps-merge index-order)
#   F3  PS-DTFE window: --ps-window against --grid-window, '--partition 2 2 2'
#   F4  PS-DTFE point evaluation, '--partition 2 2 2': --per-stream --per-stream-ids --pts-lagrangian --pts-conventions
#       against --ps-per-stream --ps-per-stream-ids --ps-per-stream-lagrangian --ps-point-conventions
#   F5  standard DTFE, '--partition 2 2 2' (a fixed '--seed': method 2 draws random points, seeded by the clock
#       otherwise): --inputFile/--outputFile, --input, --MpcUnit, '-m 2 -s 27', --lambda_th
#       against --input-file/--output-file, --input-format, --mpc-unit, --average-method 2 --subsamples 3, --lambda-th
#   F6  standard DTFE, '-m3 -s8' (letters with the value attached) against --average-method 3 --subsamples 2; standard
#       DTFE's '-m' and '-s' warn as '-m/--method' and '-s/--samples' (either spelling may have been typed; the samples
#       line names the k: '--subsamples 2')
#   F7  standard DTFE, one domain: --CIC --randomSample --density0 against --cic --particle-fraction --mean-density
#   F8  standard '--samples N' beside '--subsamples k': accepted when N = k^3 (one checkpoint key with '--subsamples k'
#       alone; the '=' forms too), refused when not
#   F9  the 2D build (when built): '-m 3 -s 9' against --average-method 3 --subsamples 3, '--partition 2 2'
#   F10 a '--config' file: old names alone give the canonical file's outputs; under the command line's canonical
#       spelling of the same option the command line wins
#   F12 the remaining renames, one pair per binary: standard '--output 101 --extensive --SPH 32' against
#       '--output-format 101 --scalars-extensive --sph 32'; PS-DTFE '--lagrangianInput X' against '--ps-lagrangian-input X'
#   F13 a '--config' file's 'samples = 8' under the command line's '--subsamples 3' (standard DTFE): the command line
#       wins, warned once, the run '--subsamples 3' alone gives; and 'subsamples = 2' under '-s 27'
#   F11 the renamed environment knobs, in both binaries: DTFE_PS_CHECKPOINT=0 writes no checkpoint, as
#       DTFE_CHECKPOINT=0 does; DTFE_NO_PROGRESS=1 leaves no progress sidecar, as DTFE_PROGRESS=0 does; each old name
#       warns once at start-up ("warning: OLD is deprecated, use NEW"), the new one never
#   F14 the '_a' average (2026-11-09: '--exact', the default, and '--sampled'), in both binaries: '--exact-average'
#       against '--exact' (standard DTFE partitioned, its key naming 'average=exact'; PS-DTFE one domain); then, one
#       domain, the run log's header line that names the average and why (the literals the drivers and python/ read)
#       with the outputs of the run it stands for: no flag is '--exact, the default'; DTFE_EXACT_AVERAGE=0 / =1 alone
#       sample / integrate exactly; a flag beats the pin both ways; a '--subsamples' count alone (standard DTFE: an
#       '--average-method' too, 1 included) selects the sampled average; '--exact --sampled' is refused; standard
#       DTFE on a redshift cone grid: '--exact' warns and samples, no flag samples silently, both the outputs of
#       '--sampled' and the header 'SAMPLED (a redshift cone grid has no exact form)'
# The reference runs of tests/dev/bytecmp.sh and tests/dev/tune_report_set.sh keep the old spellings on purpose (they
# also run binaries from before the scheme); this suite is where the two spellings are compared.
# About half a minute on the Mac.
# Usage: tests/flag_spelling_check.sh [--no-build]   (--no-build is accepted and ignored: the binaries are used as built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}" "${DTFE_BIN}"
for b in "${PS_BIN}" "${DTFE_BIN}"; do [ -x "$b" ] || { echo "FAIL: $b is not built"; exit 1; }; done
PS="$(cd "$(dirname "${PS_BIN}")" && pwd)/$(basename "${PS_BIN}")"
STD="$(cd "$(dirname "${DTFE_BIN}")" && pwd)/$(basename "${DTFE_BIN}")"
TMP="${DTFE_TEST_TMP}/flag_spelling_check"
rm -rf "$TMP"; mkdir -p "$TMP"
TMP="$(cd "$TMP" && pwd -P)"
# the '_a' average's environment pin: F14 sets it where it tests it, and its no-flag case needs it unset (an export
# from a login shell would steer every run without a flag)
unset DTFE_EXACT_AVERAGE
SNAP="$TMP/waves.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 24 --box 100 --amplitude-factor 1.8 \
    --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null || { echo "FAIL data gen"; exit 1; }
# a multi-stream plane of 32^2 points through the crossed waves (float64 x y z)
"$PY" - "$TMP/plane.bin" <<'PYEOF' || { echo "FAIL point file"; exit 1; }
import sys
import numpy as np
c = (np.arange(32) + 0.5) * 100.0 / 32
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 50.3)], 1).astype(np.float64).tofile(sys.argv[1])
PYEOF
PASS=0; FAILS=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
# key_of <output>: the settings key block of its checkpoint (the header is text; the grids follow it)
key_of() { LC_ALL=C sed -n '/^key-begin$/,/^key-end$/p;/^header-end$/q' "$1.runlog.checkpoint/checkpoint" 2>/dev/null; }
# warned_once <log> <line>...: each line exactly once in the log, and no other deprecation warning there
warned_once() {
    local log="$1" w n want ok=0; shift
    for w in "$@"; do
        n="$(grep -cxF -- "$w" "$log")"
        [ "$n" = 1 ] || { echo "    '$w': $n times in ${log##*/}"; ok=1; }
    done
    n="$(grep -c "deprecated\|has no effect" "$log")"; want="$(printf '%s\n' "$@" | sort -u | grep -c .)"
    [ "$n" = "$want" ] || { echo "    ${log##*/}: $n warning lines, $want expected"; grep "deprecated\|has no effect" "$log" | sed 's/^/      | /'; ok=1; }
    return $ok
}
# go <binary> <output> <args...>: the run, the positional '@SNAP@' and '@OUT@' put in; OMP_NUM_THREADS=2 and any
# variables of RUN_ENV (a list of NAME=value) in its environment; the exit status is the binary's
RUN_ENV=(); LASTKEY=""
go() {
    local bin="$1" out="$2"; shift 2
    local a=("${@//@SNAP@/$SNAP}"); a=("${a[@]//@OUT@/$out}")
    # (the shell's own 'Killed: 9' line for a DTFE_TEST_EXIT_AFTER_PARTITIONS run is not output)
    { env OMP_NUM_THREADS=2 ${RUN_ENV[@]+"${RUN_ENV[@]}"} "$bin" "${a[@]}" > "$out.log" 2>&1; } 2>/dev/null
}

# pair <tag> <binary> <kill after (0: a one-domain run, no checkpoint)> <the old run's warning lines>...
# OLD and NEW (arrays, set by the caller): the two spellings of one run; LASTKEY is left holding the old one's key
pair() {
    local tag="$1" bin="$2" kill="$3"; shift 3
    local o="$TMP/$tag" rc rc1 rc2 k1 k2
    go "$bin" "$o.ref" "${NEW[@]}"; rc=$?
    [ "$rc" = 0 ] && ! grep -q "deprecated\|has no effect" "$o.ref.log"
    check "$tag the canonical spelling runs without a warning (exit $rc)" $?
    [ "$rc" = 0 ] || { tail -n 15 "$o.ref.log" | sed 's/^/      | /'; return; }
    go "$bin" "$o.old" "${OLD[@]}"; rc=$?
    [ "$rc" = 0 ] && warned_once "$o.old.log" "$@"
    check "$tag the old spelling runs, each deprecated spelling warned once (exit $rc)" $?
    same "$o.ref" "$o.old"; check "$tag every output of the old spelling byte-identical to the canonical run's" $?
    [ "$kill" -gt 0 ] || return
    RUN_ENV=(DTFE_TEST_EXIT_AFTER_PARTITIONS="$kill")
    go "$bin" "$o.x" "${NEW[@]}"; rc1=$?
    k1="$(key_of "$o.x")"
    rm -rf "$o.x".*
    go "$bin" "$o.x" "${OLD[@]}"; rc2=$?
    k2="$(key_of "$o.x")"; LASTKEY="$k2"
    RUN_ENV=()
    [ "$rc1" = 137 ] && [ "$rc2" = 137 ] && grep -q '^opt ' <<< "$k1" && [ "$k1" = "$k2" ]
    rc=$?
    [ "$rc" = 0 ] || diff <(echo "$k1") <(echo "$k2") | sed 's/^/      | /'
    check "$tag killed after $kill partitions under each spelling (exit $rc1, $rc2): one settings key, line for line" $rc
    mv "$o.x.log" "$o.x1.log"
    go "$bin" "$o.x" "${NEW[@]}" --resume; rc=$?
    [ "$rc" = 0 ] && grep -q -- "--resume: $kill of " "$o.x.log" && ! grep -q "deprecated" "$o.x.log" && same "$o.ref" "$o.x"
    check "$tag the old spelling's checkpoint resumes under the canonical one to the uninterrupted outputs (exit $rc)" $?
}
dep() { echo "warning: --$1 is deprecated, use --$2"; }
# the two whose letters go too: the warning names both spellings ('vm' cannot tell which one was typed)
MDEP="warning: -m/--method is deprecated, use --average-method"
sdep() { echo "warning: -s/--samples is deprecated, use --subsamples${1:+ $1}"; }

echo ">> F1: PS-DTFE sampled deposit, --partition 2 2 2"
OLD=(@SNAP@ @OUT@ -i 105 --MpcUnit 1 -p -g 24 --partition 2 2 2 --max-concurrent 2 --avg-subsamples 2 --ps-lag-padding 8
     --lambda_th 0.3 --ordered-merge --ps-stream-density dtfe --ps-no-overlap -m 2 -s 27 -f density_a velocity_a dispersion_a)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --subsamples 2
     --ps-lagrangian-padding 8 --lambda-th 0.3 --ps-merge index-order --ps-stream-density linear --no-overlap
     --field density_a velocity_a dispersion_a)
NOEFF="(its cell averages take '--subsamples k', k^3 points per cell); drop it"
pair F1 "$PS" 3 "$(dep MpcUnit mpc-unit)" "$(dep avg-subsamples subsamples)" "$(dep ps-lag-padding ps-lagrangian-padding)" \
    "$(dep lambda_th lambda-th)" "$(dep ordered-merge 'ps-merge index-order')" \
    "$(dep 'ps-stream-density dtfe' 'ps-stream-density linear')" "$(dep ps-no-overlap no-overlap)" \
    "warning: -m/--method has no effect in PS-DTFE $NOEFF" "warning: -s/--samples has no effect in PS-DTFE $NOEFF"
grep -qx "average=sampled" <<< "$LASTKEY"; check "F1 the key names the sampled average ('average=sampled')" $?

echo ">> F2: PS-DTFE exact deposit, fixed-point merge"
MO=(--fixed-point-merge --ordered-merge); MN=(--ps-merge fixed-point)
MW="$(dep fixed-point-merge 'ps-merge fixed-point')"
if precision_double; then   # the double build refuses the fixed-point merge (64-bit words): the index order there
    echo "  (the double pair: '--ordered-merge' against '--ps-merge index-order' instead)"
    MO=(--ordered-merge); MN=(--ps-merge index-order); MW="$(dep ordered-merge 'ps-merge index-order')"
fi
OLD=(@SNAP@ @OUT@ --input 105 --MpcUnit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --ps-exact-deposit
     "${MO[@]}" --field density_a velocity_a dispersion_a)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --exact
     "${MN[@]}" --field density_a velocity_a dispersion_a)
pair F2 "$PS" 3 "$(dep input input-format)" "$(dep MpcUnit mpc-unit)" "$(dep ps-exact-deposit exact)" "$MW" \
    "$(dep ordered-merge 'ps-merge index-order')"
if ! precision_double; then
    grep -qx "merge=fixed-point" <<< "$LASTKEY"; check "F2 the key names the fixed-point merge ('merge=fixed-point')" $?
fi

echo ">> F3: PS-DTFE window"
W=(12.5 37.5 62.5 81.25 46.875 48.4375)
OLD=(@SNAP@ @OUT@ --input 105 --MpcUnit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --ps-window "${W[@]}"
     --field density_a velocity_a)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2
     --grid-window "${W[@]}" --field density_a velocity_a)
pair F3 "$PS" 2 "$(dep input input-format)" "$(dep MpcUnit mpc-unit)" "$(dep ps-window grid-window)"

echo ">> F4: PS-DTFE point evaluation, --partition 2 2 2"
OLD=(@SNAP@ @OUT@ --input 105 --MpcUnit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --field density_a
     --sample-points "$TMP/plane.bin" --per-stream --per-stream-ids --pts-lagrangian --pts-conventions grid)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --field density_a
     --sample-points "$TMP/plane.bin" --ps-per-stream --ps-per-stream-ids --ps-per-stream-lagrangian
     --ps-point-conventions grid)
pair F4 "$PS" 3 "$(dep input input-format)" "$(dep MpcUnit mpc-unit)" "$(dep per-stream ps-per-stream)" \
    "$(dep per-stream-ids ps-per-stream-ids)" "$(dep pts-lagrangian ps-per-stream-lagrangian)" \
    "$(dep pts-conventions ps-point-conventions)"
ls "$TMP"/F4.ref.pts_stream_lagpos >/dev/null 2>&1; check "F4 the per-stream Lagrangian points are written ('.pts_stream_lagpos')" $?

echo ">> F5: standard DTFE Monte-Carlo average, --partition 2 2 2"
OLD=(--inputFile @SNAP@ --outputFile @OUT@ --input 105 --MpcUnit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2
     -m 2 -s 27 --seed 11 --lambda_th 0.3 --field density density_a velocity_a)
NEW=(--input-file @SNAP@ --output-file @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2
     --max-concurrent 2 --average-method 2 --subsamples 3 --seed 11 --lambda-th 0.3 --field density density_a velocity_a)
pair F5 "$STD" 3 "$(dep inputFile input-file)" "$(dep outputFile output-file)" "$(dep input input-format)" \
    "$(dep MpcUnit mpc-unit)" "$(dep lambda_th lambda-th)" "$MDEP" "$(sdep 3)"

echo ">> F6: standard DTFE regular sub-samples, letters with their values attached"
OLD=(@SNAP@ @OUT@ -i105 --mpc-unit 1 -p -g24 --partition 2 2 2 --max-concurrent 2 -m3 -s8 -f density_a velocity_a)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2
     --average-method 3 --subsamples 2 --field density_a velocity_a)
pair F6 "$STD" 3 "$MDEP" "$(sdep 2)"

echo ">> F7: standard DTFE, one domain: the mass-assignment and sampling names"
OLD=(@SNAP@ @OUT@ --input 105 --MpcUnit 1 --periodic --grid 24 --CIC --randomSample 0.5 --seed 11 --density0 2 --field density)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --cic --particle-fraction 0.5 --seed 11
     --mean-density 2 --field density)
pair F7 "$STD" 0 "$(dep input input-format)" "$(dep MpcUnit mpc-unit)" "$(dep CIC cic)" "$(dep randomSample particle-fraction)" \
    "$(dep density0 mean-density)"

echo ">> F8: standard '--samples N' beside '--subsamples k'"
B=(--input-format 105 --mpc-unit 1 --periodic --grid 24 --average-method 2 --seed 11 --field density_a)
go "$STD" "$TMP/F8a" "$SNAP" "$TMP/F8a" "${B[@]}" --samples 27 --subsamples 3; rc=$?
go "$STD" "$TMP/F8r" "$SNAP" "$TMP/F8r" "${B[@]}" --subsamples 3; rc2=$?
[ "$rc" = 0 ] && [ "$rc2" = 0 ] && same "$TMP/F8r" "$TMP/F8a"
check "F8 '--samples 27 --subsamples 3' is accepted, the same run as '--subsamples 3' (exit $rc, $rc2)" $?
go "$STD" "$TMP/F8b" "$SNAP" "$TMP/F8b" "${B[@]}" --samples 20 --subsamples 3; rc=$?
[ "$rc" != 0 ] && grep -q -- "--samples" "$TMP/F8b.log" && grep -q -- "--subsamples" "$TMP/F8b.log" && [ ! -e "$TMP/F8b.a_den" ]
check "F8 '--samples 20 --subsamples 3' is refused, naming both (exit $rc)" $?
# the checkpoint key: '--samples 27 --subsamples 3' is the key of '--subsamples 3' alone (one line), and the '=' forms
# '--samples=27' and '--subsamples=3' give one key too
F8K=(--partition 2 2 2 --max-concurrent 2)
f8key() { rm -rf "$TMP/F8k".*; RUN_ENV=(DTFE_TEST_EXIT_AFTER_PARTITIONS=2); go "$STD" "$TMP/F8k" "$SNAP" "$TMP/F8k" "${B[@]}" "${F8K[@]}" "$@"
          RUN_ENV=(); key_of "$TMP/F8k"; }
k1="$(f8key --samples 27 --subsamples 3)"; k2="$(f8key --subsamples 3)"; k3="$(f8key --samples=27)"; k4="$(f8key --subsamples=3)"
[ -n "$k1" ] && [ "$k1" = "$k2" ] && [ "$(grep -c -- '--subsamples' <<< "$k1")" = 1 ] && [ -n "$k3" ] && [ "$k3" = "$k4" ]
check "F8 one checkpoint key for '--samples 27 --subsamples 3' and '--subsamples 3', and for '--samples=27' and '--subsamples=3'" $?

echo ">> F9: the 2D build"
if precision_double || [ ! -x ./DTFE-2d ]; then
    echo "  SKIP F9: ./DTFE-2d is not built (or the double pair runs)"
else
    SNAP2="$TMP/waves2d.hdf5"
    "$PY" tests/generate_ps_test_data.py --out "$SNAP2" --n 48 --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 \
        --seed 7 --crossed-waves --dim 2 >/dev/null || { echo "FAIL 2D data gen"; exit 1; }
    S3="$SNAP"; SNAP="$SNAP2"
    OLD=(@SNAP@ @OUT@ --input 105 --mpc-unit 1 --periodic --grid 48 --partition 2 2 --max-concurrent 2 -m 3 -s 9
         --field density_a velocity_a)
    NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 48 --partition 2 2 --max-concurrent 2
         --average-method 3 --subsamples 3 --field density_a velocity_a)
    pair F9 "$(pwd)/DTFE-2d" 2 "$(dep input input-format)" "$MDEP" "$(sdep 3)"
    SNAP="$S3"
fi

echo ">> F10: a '--config' file in the old spelling"
# the file alone: the same run as the canonical file; the file under the command line's other spelling: the command
# line wins (boost keeps the value stored first), warned once
printf 'MpcUnit = 1\nlambda_th = 0.3\navg-subsamples = 2\n' > "$TMP/old.cfg"
printf 'mpc-unit = 1\nlambda-th = 0.3\nsubsamples = 2\n' > "$TMP/new.cfg"
printf 'lambda_th = 0.05\n' > "$TMP/over.cfg"
CB=(--input-format 105 --periodic --grid 24 --field density_a tweb_a)
go "$PS" "$TMP/F10n" "$SNAP" "$TMP/F10n" "${CB[@]}" --config "$TMP/new.cfg"; rc1=$?
go "$PS" "$TMP/F10o" "$SNAP" "$TMP/F10o" "${CB[@]}" --config "$TMP/old.cfg"; rc2=$?
[ "$rc1" = 0 ] && [ "$rc2" = 0 ] && ! grep -q "deprecated" "$TMP/F10n.log" \
    && warned_once "$TMP/F10o.log" "$(dep MpcUnit mpc-unit)" "$(dep lambda_th lambda-th)" "$(dep avg-subsamples subsamples)" \
    && same "$TMP/F10n" "$TMP/F10o"
check "F10 the old-spelling file gives the canonical file's outputs, each old name warned once (exit $rc1, $rc2)" $?
go "$PS" "$TMP/F10c" "$SNAP" "$TMP/F10c" "${CB[@]}" --config "$TMP/over.cfg" --mpc-unit 1 --lambda-th 0.3 --subsamples 2; rc1=$?
go "$PS" "$TMP/F10f" "$SNAP" "$TMP/F10f" "${CB[@]}" --config "$TMP/over.cfg" --mpc-unit 1 --subsamples 2; rc2=$?
[ "$rc1" = 0 ] && [ "$rc2" = 0 ] && warned_once "$TMP/F10c.log" "$(dep lambda_th lambda-th)" && same "$TMP/F10n" "$TMP/F10c" \
    && ! cmp -s "$TMP/F10n.a_tweb" "$TMP/F10f.a_tweb"
check "F10 '--lambda-th 0.3' on the command line beats the file's 'lambda_th = 0.05' (which alone moves the T-web) (exit $rc1, $rc2)" $?

echo ">> F12: the remaining renames (standard output format, scalars, SPH; PS-DTFE's Lagrangian input)"
OLD=(@SNAP@ @OUT@ --input 105 --mpc-unit 1 --periodic --grid 24 --output 101 --extensive --SPH 32 --field density)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --output-format 101 --scalars-extensive --sph 32
     --field density)
pair F12s "$STD" 0 "$(dep input input-format)" "$(dep output output-format)" "$(dep extensive scalars-extensive)" "$(dep SPH sph)"
OLD=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --lagrangianInput "$SNAP" --field density_a)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --ps-lagrangian-input "$SNAP" --field density_a)
pair F12p "$PS" 0 "$(dep lagrangianInput ps-lagrangian-input)"

echo ">> F13: a '--config' file's 'samples' (the deprecated total) and the command line's '--subsamples' (standard DTFE)"
printf 'samples = 8\n' > "$TMP/s8.cfg"; printf 'subsamples = 2\n' > "$TMP/k2.cfg"
SB=(--input-format 105 --mpc-unit 1 --periodic --grid 24 --average-method 3 --field density_a)
go "$STD" "$TMP/F13r" "$SNAP" "$TMP/F13r" "${SB[@]}" --subsamples 3; rc1=$?
go "$STD" "$TMP/F13a" "$SNAP" "$TMP/F13a" "${SB[@]}" --config "$TMP/s8.cfg" --subsamples 3; rc2=$?
go "$STD" "$TMP/F13b" "$SNAP" "$TMP/F13b" "${SB[@]}" --config "$TMP/k2.cfg" -s 27; rc3=$?
[ "$rc1" = 0 ] && [ "$rc2" = 0 ] && [ "$rc3" = 0 ] && ! grep -q "deprecated" "$TMP/F13r.log" \
    && warned_once "$TMP/F13a.log" "$(sdep)" && warned_once "$TMP/F13b.log" "$(sdep 3)" \
    && same "$TMP/F13r" "$TMP/F13a" && same "$TMP/F13r" "$TMP/F13b"
check "F13 the command line's spelling beats the file's other one ('samples = 8' / '--subsamples 3', 'subsamples = 2' / '-s 27'), warned once (exit $rc1, $rc2, $rc3)" $?

echo ">> F11: the renamed environment knobs"
for b in PS STD; do
    bin="$PS"; [ "$b" = STD ] && bin="$STD"
    A=("$SNAP" @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2
       --field density_a)
    RUN_ENV=(DTFE_TEST_EXIT_AFTER_PARTITIONS=2 DTFE_PS_CHECKPOINT=0 DTFE_NO_PROGRESS=1)
    go "$bin" "$TMP/F11${b}o" "${A[@]}"; rc1=$?
    RUN_ENV=(DTFE_TEST_EXIT_AFTER_PARTITIONS=2 DTFE_CHECKPOINT=0 DTFE_PROGRESS=0)
    go "$bin" "$TMP/F11${b}n" "${A[@]}"; rc2=$?
    RUN_ENV=(DTFE_TEST_EXIT_AFTER_PARTITIONS=2)
    go "$bin" "$TMP/F11${b}d" "${A[@]}"; rc3=$?
    RUN_ENV=()
    [ "$rc1" = 137 ] && [ "$rc2" = 137 ] && [ "$rc3" = 137 ] \
        && [ ! -e "$TMP/F11${b}o.runlog.checkpoint" ] && [ ! -e "$TMP/F11${b}o.runlog.progress" ] \
        && [ ! -e "$TMP/F11${b}n.runlog.checkpoint" ] && [ ! -e "$TMP/F11${b}n.runlog.progress" ] \
        && [ -e "$TMP/F11${b}d.runlog.checkpoint/checkpoint" ] && [ -e "$TMP/F11${b}d.runlog.progress" ]
    check "F11 $b: the old and the new names both turn the checkpoint and the progress sidecar off (exit $rc1, $rc2, $rc3)" $?
    warned_once "$TMP/F11${b}o.log" "warning: DTFE_PS_CHECKPOINT is deprecated, use DTFE_CHECKPOINT" \
            "warning: DTFE_NO_PROGRESS is deprecated, use DTFE_PROGRESS=0" \
        && ! grep -q "deprecated" "$TMP/F11${b}n.log" "$TMP/F11${b}d.log"
    check "F11 $b: each old name warned once, the new names never" $?
done

echo ">> F14: the '_a' average, both binaries: --exact (the default since 2026-11-09), --sampled, DTFE_EXACT_AVERAGE"
# the old spelling '--exact-average' is '--exact' (PS-DTFE's '--ps-exact-deposit' is F2's): standard DTFE partitioned,
# so the key is compared too and must name the exact average; PS-DTFE on one domain
OLD=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --exact-average
     --field density_a velocity_a)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 2 2 2 --max-concurrent 2 --exact
     --field density_a velocity_a)
LASTKEY=""
pair F14std "$STD" 3 "$(dep exact-average exact)"
grep -qx "average=exact" <<< "$LASTKEY"; check "F14std the key names the exact average ('average=exact')" $?
OLD=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 1 --exact-average --field density_a velocity_a)
NEW=(@SNAP@ @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 1 --exact --field density_a velocity_a)
pair F14ps "$PS" 0 "$(dep exact-average exact)"
# hdr <log> <text>: the run log's header carries the line (src/user_options.cc printOptions: which average runs and
# why, in the literals the drivers and python/ read)
hdr() { grep -qF -- "$2" "$1"; }
for b in STD PS; do
    bin="$STD"; XL="volume averaging       : EXACT cell-tetrahedron integration of the linear interpolant ("
    SL="volume averaging       : SAMPLED ("; CL="a sampling count or method was given; --exact integrates the cells exactly"
    if [ "$b" = PS ]; then
        bin="$PS"; XL="grid deposit           : EXACT tetrahedron-cell intersection moments ("
        SL="grid deposit           : SAMPLED on nSub^3 sub-points ("; CL="a --subsamples count was given; --exact deposits every tetrahedron exactly"
    fi
    M=("$SNAP" @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 24 --partition 1 --field density_a velocity_a)
    o="$TMP/F14mode_$b"
    go "$bin" "$o.x" "${M[@]}" --exact; r1=$?
    go "$bin" "$o.s" "${M[@]}" --sampled; r2=$?
    go "$bin" "$o.d" "${M[@]}"; r3=$?
    [ "$r1$r2$r3" = 000 ] && hdr "$o.x.log" "${XL}--exact; " && hdr "$o.s.log" "${SL}--sampled)" \
        && hdr "$o.d.log" "${XL}--exact, the default; " && same "$o.x" "$o.d" && ! cmp -s "$o.x.a_den" "$o.s.a_den"
    check "F14 $b: no flag runs the exact average ('--exact, the default'), the outputs of '--exact'; '--sampled' samples (exit $r1, $r2, $r3)" $?
    RUN_ENV=(DTFE_EXACT_AVERAGE=0); go "$bin" "$o.p0" "${M[@]}"; r1=$?; go "$bin" "$o.fx" "${M[@]}" --exact; r2=$?
    RUN_ENV=(DTFE_EXACT_AVERAGE=1); go "$bin" "$o.p1" "${M[@]}"; r3=$?; go "$bin" "$o.fs" "${M[@]}" --sampled; r4=$?
    RUN_ENV=()
    [ "$r1$r3" = 00 ] && hdr "$o.p0.log" "${SL}--sampled from DTFE_EXACT_AVERAGE=0)" && same "$o.s" "$o.p0" \
        && hdr "$o.p1.log" "${XL}--exact from DTFE_EXACT_AVERAGE=1; " && same "$o.x" "$o.p1"
    check "F14 $b: DTFE_EXACT_AVERAGE=0 alone samples, =1 alone integrates exactly, each said in the header (exit $r1, $r3)" $?
    [ "$r2$r4" = 00 ] && hdr "$o.fx.log" "${XL}--exact; " && same "$o.x" "$o.fx" \
        && hdr "$o.fs.log" "${SL}--sampled)" && same "$o.s" "$o.fs"
    check "F14 $b: a flag beats the pin both ways (DTFE_EXACT_AVERAGE=0 with --exact, =1 with --sampled; exit $r2, $r4)" $?
    go "$bin" "$o.c" "${M[@]}" --subsamples 2; r1=$?
    go "$bin" "$o.cs" "${M[@]}" --sampled --subsamples 2; r2=$?
    [ "$r1$r2" = 00 ] && hdr "$o.c.log" "${SL}${CL})" && same "$o.cs" "$o.c"
    check "F14 $b: a '--subsamples' count alone selects the sampled average, said in the header (exit $r1, $r2)" $?
    if [ "$b" = STD ]; then   # an '--average-method' of any value is a sampling request too, 1 (the default method) included
        go "$bin" "$o.m1" "${M[@]}" --average-method 1; r1=$?
        [ "$r1" = 0 ] && hdr "$o.m1.log" "${SL}${CL})" && same "$o.s" "$o.m1"
        check "F14 $b: '--average-method 1' alone selects the sampled average, the outputs of '--sampled' (exit $r1)" $?
        # a redshift cone grid has no exact form (its '_a' fields are sampled, method 2): '--exact' warns and
        # samples, the default samples without a word; both give the outputs of '--sampled'. The seed is pinned:
        # method 2 draws random points, and without '--seed' the generator starts from the clock (two of the
        # three runs then agree only when they start within the same second).
        K=("$SNAP" @OUT@ --input-format 105 --mpc-unit 1 --periodic --grid 4 4 6 --redshift-cone 5 40 30 150 0 300
           --cone-origin 50 50 50 --partition 1 --field density_a velocity_a --seed 7)
        KW="'--exact' does not apply to a redshift cone grid"; KS="${SL}a redshift cone grid has no exact form)"
        go "$bin" "$o.kx" "${K[@]}" --exact; r1=$?
        go "$bin" "$o.ks" "${K[@]}" --sampled; r2=$?
        go "$bin" "$o.kd" "${K[@]}"; r3=$?
        [ "$r1$r2$r3" = 000 ] && grep -qF -- "$KW" "$o.kx.log" && ! grep -qF -- "$KW" "$o.ks.log" "$o.kd.log" \
            && hdr "$o.kx.log" "$KS" && hdr "$o.kd.log" "$KS" && hdr "$o.ks.log" "${SL}--sampled)" \
            && same "$o.kx" "$o.ks" && same "$o.kd" "$o.ks"
        check "F14 $b: on a redshift cone grid '--exact' warns and samples, no flag samples silently, both the outputs of '--sampled' (exit $r1, $r2, $r3)" $?
    fi
    go "$bin" "$o.xs" "${M[@]}" --exact --sampled; r1=$?
    [ "$r1" = 1 ] && grep -qF -- "'--exact' and '--sampled' exclude each other" "$o.xs.log" && [ ! -e "$o.xs.a_den" ]
    check "F14 $b: '--exact --sampled' is refused at option parsing, naming both (exit $r1)" $?
done

echo ""
echo "RESULT: $PASS passed, $FAILS failed"
[ "$FAILS" = 0 ]
