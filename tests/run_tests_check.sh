#!/usr/bin/env bash
# The suites' own skip / fail contract (SKIP exits 0 unless DTFE_TEST_REQUIRED=1), run on copies beside stand-in binaries.
# Cases R1-R5 and D1 print their own labels.
# Usage: tests/run_tests_check.sh [--no-build]   (--no-build is accepted and ignored: nothing is built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/lib.sh"   # PY: the python with numpy and h5py
# (no precision.sh: the cases set the precision themselves) a missing ./DTFE is a skip, or a failure when required
# (skip_or_fail, tests/lib.sh)
[ -x ./DTFE ] || skip_or_fail "./DTFE is not built (make DTFE)"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/run_tests_check"
rm -rf "$TMP"; mkdir -p "$TMP/rt/tests" "$TMP/dc/tests" "$TMP/dc/scripts"

fails=0
# check_rc <0 = pass> <text> (it counts a failure in fails), has01 <file> <pattern> (echoes 0 or 1): tests/lib.sh
count() { grep -acE -e "$2" "$1" || true; }

# the scratch tree of run_tests.py: the current script (copied now, never a stale one) beside the stand-in ./DTFE
cp tests/run_tests.py "$TMP/rt/tests/"
cat > "$TMP/rt/DTFE" <<'EOF'
#!/usr/bin/env bash
# stand-in ./DTFE: STUB_HELP=fail -> --full-help exits 3; STUB_HELP=noflag -> the real --full-help without the
# lines that name --sampled (the gate of the exact-average tests); everything else is the real binary (REAL_DTFE)
if [ "${1:-}" = "--full-help" ]; then
    case "${STUB_HELP:-}" in
        fail)   echo "stand-in: --full-help is broken" >&2; exit 3 ;;
        noflag) help="$("${REAL_DTFE}" --full-help)" || exit $?
                printf '%s\n' "$help" | grep -v -e '--sampled'; exit 0 ;;
    esac
fi
exec "${REAL_DTFE}" "$@"
EOF
chmod +x "$TMP/rt/DTFE"
# rt <log> <reference dir> [VAR=value ...]: the copied run_tests.py; echoes its exit status. The precision and the
# requirement are cleared first, so each case sets exactly what it tests wherever ci_suite.sh runs this suite.
rt() {
    local log="$1" ref="$2"; shift 2
    local rc=0
    env -u DTFE_TEST_PRECISION -u DTFE_TEST_REQUIRED -u STUB_HELP REAL_DTFE="$ROOT/DTFE" \
        DTFE_TEST_REFERENCE_DIR="$ref" "$@" "$PY" "$TMP/rt/tests/run_tests.py" > "$log" 2>&1 || rc=$?
    echo "$rc"
}
fresh() { mktemp -d "$TMP/ref.XXXXXX"; }

echo ">> R1: --full-help exits 3"
for req in 0 1; do
    rc=$(rt "$TMP/r1_$req.log" "$(fresh)" STUB_HELP=fail DTFE_TEST_REQUIRED=$req)
    check_rc "$([ "$rc" = "1" ] && [ "$(count "$TMP/r1_$req.log" '^  FAIL  exact_average_(lattice|convergence) -- DTFE --full-help exited with 3$')" = "2" ] \
             && [ "$(count "$TMP/r1_$req.log" '^  FAIL ')" = "2" ] && echo 0 || echo 1)" \
          "DTFE_TEST_REQUIRED=$req: both exact-average tests FAIL (--full-help exited with 3), nothing else does, exit 1 ($rc)"
done
echo ">> R2: --full-help without --sampled"
rc=$(rt "$TMP/r2_0.log" "$(fresh)" STUB_HELP=noflag)
check_rc "$([ "$rc" = "0" ] && [ "$(count "$TMP/r2_0.log" '^  SKIP  exact_average_(lattice|convergence) -- DTFE has no --sampled$')" = "2" ] \
         && [ "$(count "$TMP/r2_0.log" '^  FAIL ')" = "0" ] && echo 0 || echo 1)" \
      "both exact-average tests SKIP (DTFE has no --sampled), no failure, exit 0 ($rc)"
check_rc "$(has01 "$TMP/r2_0.log" '^Results: [0-9]+/[0-9]+ passed, 3 skipped$')" "... the Results line counts them (and the regression's) as skipped"
rc=$(rt "$TMP/r2_1.log" "$(fresh)" STUB_HELP=noflag DTFE_TEST_REQUIRED=1)
check_rc "$([ "$rc" = "1" ] && [ "$(count "$TMP/r2_1.log" '^  FAIL  exact_average_(lattice|convergence) -- DTFE_TEST_REQUIRED=1, but DTFE has no --sampled$')" = "2" ] \
         && echo 0 || echo 1)" "DTFE_TEST_REQUIRED=1: both FAIL instead, exit 1 ($rc)"
echo ">> R3: no regression reference (a fresh DTFE_TEST_REFERENCE_DIR)"
ref="$(fresh)"
rc=$(rt "$TMP/r3.log" "$ref" DTFE_TEST_REQUIRED=1)
check_rc "$([ "$rc" = "0" ] && [ "$(has01 "$TMP/r3.log" "^  SKIP  regression -- no reference in ")" = "0" ] \
         && [ "$(count "$TMP/r3.log" '^  FAIL ')" = "0" ] && echo 0 || echo 1)" \
      "the regression is a SKIP and the run exits 0 even under DTFE_TEST_REQUIRED=1 ($rc)"
check_rc "$(has01 "$TMP/r3.log" '^Results: ([0-9]+)/[0-9]+ passed, 1 skipped$')" "... the Results line says '1 skipped'"
rc=$(rt "$TMP/r3b.log" "$ref" DTFE_TEST_REQUIRED=1)
check_rc "$([ "$rc" = "0" ] && [ -s "$ref/regression_periodic.txt" ] && [ "$(has01 "$TMP/r3b.log" '^  PASS  regression$')" = "0" ] \
         && [ "$(has01 "$TMP/r3b.log" 'skipped')" = "1" ] && echo 0 || echo 1)" \
      "... it wrote a reference, and the next run compares against it: PASS, nothing skipped ($rc)"
echo ">> R5: the stored reference, as stored and perturbed"
STORED="$ROOT/tests/reference/regression_periodic.txt"
if [ -s "$STORED" ]; then
    # r5 <log> <scale every value> <scale value 100 of the density column>: a copy of the stored reference, scaled
    r5() {
        local d; d="$(fresh)"
        awk -v all="$2" -v one="$3" '/^#/ { print; next } { n++; a = $1 * all; if (n == 100) a *= one; printf "%.9e %.9e\n", a, $2 * all }' \
            "$STORED" > "$d/regression_periodic.txt"
        rt "$1" "$d"
    }
    rc=$(r5 "$TMP/r5a.log" 1 1)
    check_rc "$([ "$rc" = "0" ] && [ "$(has01 "$TMP/r5a.log" '^  PASS  regression$')" = "0" ] && echo 0 || echo 1)" \
          "this build matches the stored reference: PASS ($rc)"
    rc=$(r5 "$TMP/r5b.log" 1.00005 1)
    check_rc "$([ "$rc" = "0" ] && [ "$(has01 "$TMP/r5b.log" '^  PASS  regression$')" = "0" ] && echo 0 || echo 1)" \
          "... every value moved by 5e-5 relative (rounding-level): still PASS ($rc)"
    rc=$(r5 "$TMP/r5c.log" 1 1.001)
    check_rc "$([ "$rc" = "1" ] && [ "$(has01 "$TMP/r5c.log" '^  FAIL  regression -- .*density: 1/1728 values differ')" = "0" ] && echo 0 || echo 1)" \
          "... ONE value moved by 1e-3 relative: FAIL, exit 1 ($rc)"
else
    check_rc 1 "the stored reference $STORED exists"
fi
echo ">> R4: DTFE_TEST_PRECISION=double without ./DTFE-double"
rc=$(rt "$TMP/r4_0.log" "$(fresh)" DTFE_TEST_PRECISION=double)
check_rc "$([ "$rc" = "0" ] && [ "$(has01 "$TMP/r4_0.log" '^SKIP: .*DTFE-double is not built')" = "0" ] && echo 0 || echo 1)" "SKIP: and exit 0 ($rc)"
rc=$(rt "$TMP/r4_1.log" "$(fresh)" DTFE_TEST_PRECISION=double DTFE_TEST_REQUIRED=1)
check_rc "$([ "$rc" = "1" ] && [ "$(has01 "$TMP/r4_1.log" '^FAIL \(required\): .*DTFE-double is not built')" = "0" ] && echo 0 || echo 1)" \
      "DTFE_TEST_REQUIRED=1: FAIL (required): and exit 1 ($rc)"

echo ">> D1: ps_double_check.sh beside ./DTFE and ./PS-DTFE only"
# the suite and what it sources (tests/lib.sh: skip_or_fail; lib.sh sources scripts/h5py_python.sh)
cp tests/ps_double_check.sh tests/lib.sh "$TMP/dc/tests/"; cp scripts/h5py_python.sh "$TMP/dc/scripts/"
printf '#!/bin/sh\nexit 0\n' > "$TMP/dc/DTFE"; cp "$TMP/dc/DTFE" "$TMP/dc/PS-DTFE"; chmod +x "$TMP/dc/DTFE" "$TMP/dc/PS-DTFE"
for req in 0 1; do
    rc=0; env -u DTFE_TEST_PRECISION DTFE_TEST_REQUIRED=$req DTFE_TEST_TMP="$TMP/dc_tmp" bash "$TMP/dc/tests/ps_double_check.sh" --no-build \
        > "$TMP/d1_$req.out" 2> "$TMP/d1_$req.err" || rc=$?
    if [ "$req" = "0" ]; then
        check_rc "$([ "$rc" = "0" ] && [ "$(has01 "$TMP/d1_0.out" '^SKIP: \./DTFE-double is not built')" = "0" ] && echo 0 || echo 1)" "SKIP: and exit 0 ($rc)"
    else
        check_rc "$([ "$rc" = "1" ] && [ "$(has01 "$TMP/d1_1.err" '^FAIL \(required\): \./DTFE-double is not built')" = "0" ] \
                 && [ "$(has01 "$TMP/d1_1.out" 'SKIP')" = "1" ] && echo 0 || echo 1)" \
              "DTFE_TEST_REQUIRED=1: FAIL (required): on stderr, no SKIP, exit 1 ($rc)"
    fi
done
check_rc "$([ ! -e "$TMP/dc_tmp/double_check/waves.hdf5" ] && echo 0 || echo 1)" "... it stops before generating any data"

echo ""
if [ "$fails" -eq 0 ]; then
    rm -rf "$TMP"
    echo "RESULT: PASS  (run_tests.py and ps_double_check.sh skip what they cannot run, fail it when it is required,"
    echo "               fail a broken binary, and say so when the regression had nothing to compare)"
    exit 0
fi
echo "RESULT: FAIL  (${fails} check(s); logs kept under ${TMP})"
exit 1
