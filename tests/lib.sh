# Sourced by tests/precision.sh (so by every shell suite that sources it) and by the suites that skip it: the python
# the suites run and the helpers several suites share. Prints nothing when sourced and runs only the python probe;
# the suite's own set -e / -u apply.

# The '_a' average's pin (both binaries and the drivers read it): an export from the caller's shell would steer
# every run that names no '--exact' / '--sampled'. A suite that wants the pin exports it after sourcing this file
# (tests/ps_auto_tune_check.sh, tests/dev/bytecmp.sh).
unset DTFE_EXACT_AVERAGE
# The run mode and the device: the suites' runs were written for the programs' earlier defaults (performance, and the
# CPU unless '--gpu'), so both are pinned here, which also overrides a login export; the programs read DTFE_RUN_MODE
# and DTFE_GPU (0 = the CPU) as the drivers do. A suite that tests the defaults unsets them itself, per run
# (env -u DTFE_RUN_MODE -u DTFE_GPU; with -u SLURM_JOB_ID -u PBS_JOBID where the plain run mode is tested, since
# inside a batch job its default is performance).
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

# die_on_failed_run <exit status> <log> [<program>]: for a run that exited non-zero, the program (default PS-DTFE),
# the status and the log's last 25 lines, then exit 1.
die_on_failed_run() {
    if [ "$1" -ne 0 ]; then
        echo "   ERROR: ${3:-PS-DTFE} exited with code $1 -- last 25 lines of $2:"
        tail -n 25 "$2" | sed 's/^/      | /'
        exit 1
    fi
}

# The three PS-DTFE choices that were removed: the program stops with these texts (src/user_options.cc, on stdout
# with the program's other errors). The suites that check a refusal look for them as fixed strings.
PS_NO_LINEAR_DEPOSIT="'--ps-linear-deposit' was removed: PS-DTFE spreads each tetrahedron's mass uniformly, with '--exact' (the default) or '--sampled'."
PS_NO_GEOMETRIC="'--ps-stream-density geometric' was removed: the per-stream density at points is the linear interpolation of the vertex densities ('linear')."
PS_NO_LEGACY_POINTS="'--ps-point-conventions legacy' was removed: point values follow the grid's conventions ('grid')."

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
