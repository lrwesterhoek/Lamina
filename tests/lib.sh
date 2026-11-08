# Sourced by tests/precision.sh (so by every shell suite that sources it) and by the suites that skip it: the python
# the suites run and the helpers several suites share. Prints nothing when sourced and runs only the python probe;
# the suite's own set -e / -u apply.

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

# die_on_failed_run <exit status> <log> [<program>]: for a run that exited non-zero, the program (default PS-DTFE),
# the status and the log's last 25 lines, then exit 1.
die_on_failed_run() {
    if [ "$1" -ne 0 ]; then
        echo "   ERROR: ${3:-PS-DTFE} exited with code $1 -- last 25 lines of $2:"
        tail -n 25 "$2" | sed 's/^/      | /'
        exit 1
    fi
}

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
# merged_count <checkpoint folder>: how many partitions its 'checkpoint' file says are merged
merged_count() { grep -a -m1 '^merged_count' "$1/checkpoint" | awk '{print $2}'; }
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
