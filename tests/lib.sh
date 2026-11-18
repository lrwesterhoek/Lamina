# Sourced by tests/precision.sh (so by every shell suite that sources it) and by the suites that skip it: the python
# the suites run and the helpers several suites share. Prints nothing when sourced and runs only the python probe;
# the suite's own set -e / -u apply.

# The '_a' average's pin (both binaries and the drivers read it): an export from the caller's shell would steer
# every run that names no '--exact' / '--sampled'. A suite that wants the pin exports it after sourcing this file
# (tests/ps_auto_tune_check.sh).
unset DTFE_EXACT_AVERAGE
# The run mode and the device: the suites' runs were written for the programs' earlier defaults (performance, and the
# CPU unless '--gpu'), so both are pinned here, which also overrides a login export; the programs read DTFE_RUN_MODE
# and DTFE_GPU (0 = the CPU), and the drivers hand both on as set. A suite that tests the defaults unsets them
# itself, per run (env -u DTFE_RUN_MODE -u DTFE_GPU; with -u SLURM_JOB_ID -u PBS_JOBID where the plain run mode is
# tested, since inside a batch job its default is performance).
export DTFE_RUN_MODE=performance
export DTFE_GPU=0

# A python with numpy and a WORKING h5py: $PYTHON, else the probe in scripts/h5py_python.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/h5py_python.sh"
PY="$(h5py_python)"

# build_target <the suite's first argument> <make target> <object folder>: unless the argument is --no-build or the
# suite runs the double pair (built by 'make ... DOUBLE=1', never here), 'make <target>' in the build mode that
# '<object folder>/.build_mode' records (one make argument, e.g. METAL=1: a plain make would strip the GPU backend
# from the binary). BUILD_MODE is left set to it.
build_target() {
    if [ "${1:-}" != "--no-build" ] && ! precision_double; then
        echo ">> building $2 ..."
        BUILD_MODE="$(cat "$3/.build_mode" 2>/dev/null || true)"
        make "$2" ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
    fi
}

# gpu_backend_of <object folder>: the GPU backend stamp the Makefile's mode check left there ('metal', the only
# backend), empty when there is none.
gpu_backend_of() {
    if [ -f "$1/.gpu_mode_metal" ]; then echo metal; else echo ""; fi
}

# require_gpu_backend <object folder>: exits 0 with a SKIP line unless the folder exists without '.gpu_mode_off' and
# carries the Metal stamp.
require_gpu_backend() {
    if [ ! -d "$1" ] || [ -f "$1/.gpu_mode_off" ]; then echo "SKIP: $1 carries no GPU stamp (build with METAL=1)"; exit 0; fi
    if [ -z "$(gpu_backend_of "$1")" ]; then echo "SKIP: $1 is not a Metal build"; exit 0; fi
}

# A suite that cannot run (a binary or a feature missing) exits 0 with a SKIP line -- unless the stage says the
# thing MUST be there (DTFE_TEST_REQUIRED=1, as ci_suite.sh's double and 2d stages do after building it), in
# which case the skip is a FAILURE, on stderr: a green CI that ran no assertion is worth nothing. (run_tests_check.sh
# D1 holds ps_double_check.sh to both lines.)
skip_or_fail() {
    if [ "${DTFE_TEST_REQUIRED:-0}" = "1" ]; then echo "FAIL (required): $*" >&2; exit 1; fi
    echo "SKIP: $*"; exit 0
}

# die_on_failed_run <exit status> <log> [<program>]: for a run that exited non-zero, the program (default PS-DTFE),
# the status and the log's last 25 lines, then exit 1.
die_on_failed_run() {
    if [ "$1" -ne 0 ]; then
        echo "   ERROR: ${3:-PS-DTFE} exited with code $1 -- last 25 lines of $2:"
        tail -n 25 "$2" | sed 's/^/      | /'
        exit 1
    fi
}

# The PASS / OK / FAIL lines of the suites that count their checks:
#   check <label> <status>    '  PASS <label>' when the status is 0, else '  FAIL <label>'; counted in PASS / FAILS
#   check_result              'RESULT: <PASS> passed, <FAILS> failed', then exit 1 when a check failed, else exit 0
#   check_yes <flag> <text>   '   PASS  <text>' when the flag is 1, else '   FAIL  <text>'
#   check_rc <status> <text>  '   PASS  <text>' when the status is 0, else '   FAIL  <text>'
#                             (both count a failure in fails, which the suite sets to 0 and reports at its end)
#   check_ok <label> <flag>   '   OK   <label>' when the flag is 1, else '   FAIL <label>'; counts a failure in
#                             FAILS, which the suite reports at its end
#   pass <text>, fail <text>  '   PASS  <text>', '   FAIL  <text>'; fail sets FAIL=1 (so does a suite's own
#                             '|| FAIL=1'), and the suite's last line turns FAIL into its RESULT line
PASS=0; FAILS=0; FAIL=0
check() { if [ "$2" = 0 ]; then PASS=$((PASS+1)); echo "  PASS $1"; else FAILS=$((FAILS+1)); echo "  FAIL $1"; fi; }
check_result() { echo "RESULT: $PASS passed, $FAILS failed"; [ "$FAILS" = 0 ] || exit 1; exit 0; }
check_yes() { if [ "$1" = "1" ]; then echo "   PASS  $2"; else echo "   FAIL  $2"; fails=$((fails+1)); fi; }
check_rc() { if [ "$1" = "0" ]; then printf '   PASS  %s\n' "$2"; else printf '   FAIL  %s\n' "$2"; fails=$((fails+1)); fi; }
check_ok() { if [ "$2" -eq 1 ]; then echo "   OK   $1"; else echo "   FAIL $1"; FAILS=$((FAILS+1)); fi; }
pass() { echo "   PASS  $*"; }
fail() { echo "   FAIL  $*"; FAIL=1; }
# has <file> <text>: true when the file holds the literal text; has01 <file> <pattern>: echoes 0 when the extended
# regex matches the file, else 1, for check_rc ('-e': most patterns start with '--', which grep would take for an
# option of its own)
has() { grep -aqF -- "$2" "$1"; }
has01() { grep -aqE -e "$2" "$1" && echo 0 || echo 1; }
# agrep <label> <pattern> <file>: 'check_ok <label> 1' when the extended regex matches the file, else
# 'check_ok <label> 0' (a grep that finds nothing fails the check, not the suite under set -e)
agrep() { if grep -Eq -- "$2" "$3"; then check_ok "$1" 1; else check_ok "$1" 0; fi; }

# same <a> <b> [pattern]: every output of run a ('<a>.<pattern>', default every '<a>.*') has a byte-identical twin in
# run b, and there is at least one; the logs, the run logs and the staging files ('.part.<pid>') are not outputs.
# Says the first that differs.
same() {
    local n=0 f
    for f in "$1".${3:-*}; do
        case "$f" in *.log|*.runlog*|*.part.*) continue ;; esac
        cmp -s "$f" "$2${f#"$1"}" || { echo "    differs: ${f#"$1"}"; return 1; }
        n=$((n+1))
    done
    [ "$n" -gt 0 ]
}
# merged_count <checkpoint folder>: how many partitions its 'checkpoint' file says are merged; merged_bits: its
# 'merged' bit string (one character per partition); merged_list: the merged partitions' indices; points_list: the
# indices i of its 'points.<i>' files, sorted
merged_count() { grep -a -m1 '^merged_count' "$1/checkpoint" | awk '{print $2}'; }
merged_bits() { grep -a -m1 '^merged ' "$1/checkpoint" | awk '{print $2}'; }
merged_list() { merged_bits "$1" | fold -w1 | grep -n 1 | cut -d: -f1 | awk '{print $1-1}' | tr '\n' ' '; }
points_list() { ls "$1" 2>/dev/null | sed -n 's/^points\.\([0-9]*\)$/\1/p' | sort -n | tr '\n' ' '; }
# key_of <output>: the settings key block of its checkpoint (the header is text; the grids follow it)
key_of() { LC_ALL=C sed -n '/^key-begin$/,/^key-end$/p;/^header-end$/q' "$1.runlog.checkpoint/checkpoint" 2>/dev/null; }
# mode_of <log>: the value of the header's 'run mode' line; share_of <log>: of its 'GPU share' line
mode_of()  { LC_ALL=C sed -n 's/^[[:space:]]*run mode *: //p' "$1" | head -n 1; }
share_of() { LC_ALL=C sed -n 's/^[[:space:]]*GPU share *: //p' "$1" | head -n 1; }
# no_checkpoint_why <child log> [folder]: for a kill case that found no checkpoint, the binary's own reason (its
# 'could not be written' warning carries the room rule's 'only N GB free' text) and the volume's free space, so a
# FAIL on a nearly full disk says why. Prints only: the caller's check keeps its own pass/fail.
no_checkpoint_why() {
    local log="$1" dir="${2:-$(dirname "$1")}" w
    w="$(grep -a -m1 'partition checkpoint could not be written' "$log" 2>/dev/null | cut -c1-1000)"
    if [ -n "$w" ]; then echo "    NOTE  no checkpoint was left; the binary said: $w"
    else echo "    NOTE  no checkpoint was left and ${log##*/} has no 'could not be written' warning (killed before the first write?)"; fi
    echo "    NOTE  free space in $dir: $(df -k "$dir" 2>/dev/null | tail -1)"
    return 0
}
