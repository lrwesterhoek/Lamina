# Shared configuration for run_dtfe.sh, run_ps_dtfe.sh, run_ps_pipeline.sh and download_snapshots.sh.
# Sourced, not executed. Every value can be overridden per run:
#   - environment: DTFE_DATA_ROOT=/path/to/IllustrisTNG DTFE_SIM=TNG50-3-Dark ./run_ps_dtfe.sh
#                  GRID_SIZE=256 PADDING=40 ./run_ps_dtfe.sh -s TNG50-3-Dark
#   - script flags: ./run_ps_dtfe.sh -s TNG50-3-Dark -d /path/to/sim -g 256 99
#                   (long forms: --sim, --sim-dir, --grid, --gpu, --exact-average, --subsamples)
# The data root and the simulation have NO default: every run sets DTFE_DATA_ROOT (require_data_root; a driver's -d,
# the simulation's own folder, stands in for it) and names the simulation with -s or DTFE_SIM (require_simulation).
# The Python side (python/dtfelib/cli.py) honours the same DTFE_DATA_ROOT / DTFE_SIM variables.

source "$(dirname "${BASH_SOURCE[0]}")/h5py_python.sh"   # h5py_python: the python with numpy and h5py

# ---- the knobs' names and values (docs/cleanup-plan.md P6.7) -------------------------------------------------
# One name per knob in both drivers; a renamed knob's older names are still read for one release (P6.6 drops them).
# An EMPTY value is the same as an unset one, and a switch is off for 0, false, no or off (any case) and on for
# anything else -- the binaries' own rule (src/env_knobs.h), so PS_CAUSTICS=true is on, as DTFE_PROGRESS=true is.
#
# knob [-e] NEW OLD...: NEW keeps its value when it is set; otherwise the first OLD that is set is copied onto NEW
# (exported when the OLD was). Every OLD that is set earns one stderr line 'warning: OLD is deprecated, use NEW'
# (saying that NEW wins when both are set to different values: no refusal during the window, a stray export must
# not stop a run), and is unset afterwards, so a driver started from here (run_ps_pipeline.sh -> run_ps_dtfe.sh)
# sees the new name only and does not warn again. -e: an EMPTY value counts as set (PS_LAGRANGIAN_INPUT, whose
# empty value means "the snapshot's own InitialCoordinates"). Call it BEFORE the knob's 'VAR="${VAR:-default}"'
# line, which stays the one place a default is given (tests/docs_drift_check.py and py_gui_runspec_test.py read
# those lines, and the knob lines for the older names).
knob() {
    local keep_empty=0 new old v
    if [ "$1" = "-e" ]; then keep_empty=1; shift; fi
    new="$1"; shift
    for old in "$@"; do
        if [ "${keep_empty}" = "1" ]; then
            [ -n "${!old+x}" ] || continue
        elif [ -z "${!old:-}" ]; then
            unset "${old}"; continue
        fi
        v="${!old}"
        if { [ "${keep_empty}" = "1" ] && [ -n "${!new+x}" ]; } || { [ "${keep_empty}" = "0" ] && [ -n "${!new:-}" ]; }; then
            if [ "${!new}" = "${v}" ]; then
                echo "warning: ${old} is deprecated, use ${new}" >&2
            else
                echo "warning: ${old} is deprecated, use ${new} (${old}=${v} is not read: ${new}=${!new} wins)" >&2
            fi
        else
            printf -v "${new}" '%s' "${v}"
            case "$(declare -p "${old}" 2>/dev/null)" in "declare -x"*) export "${new}" ;; esac
            echo "warning: ${old} is deprecated, use ${new}" >&2
        fi
        unset "${old}"
    done
}

# knob_merge PS_MERGE ORDERED_MERGE FIXED_POINT_MERGE: run_ps_dtfe.sh's merge mode (PS_MERGE = index-order |
# completion-order | fixed-point; empty = not given, the driver's default index-order) from the older pair of switches,
# converted as the binary converts its old flags: FIXED_POINT_MERGE on is fixed-point (whatever ORDERED_MERGE says, as
# before), else ORDERED_MERGE on is index-order and off completion-order; FIXED_POINT_MERGE=0 alone gives nothing. A
# warning per old name set, PS_MERGE winning when it is set too, and the old names unset, as knob does.
knob_merge() {
    local new="$1" ordered="$2" fixed="$3" v="" m old exp=0
    if [ -n "${!fixed:-}" ] && knob_on "${fixed}"; then v=fixed-point
    elif [ -n "${!ordered:-}" ]; then
        if knob_on "${ordered}"; then v=index-order; else v=completion-order; fi
    fi
    for old in "${ordered}" "${fixed}"; do
        [ -n "${!old:-}" ] || { unset "${old}"; continue; }
        case "$(declare -p "${old}" 2>/dev/null)" in "declare -x"*) exp=1 ;; esac
        m=""                                         # what this name alone says
        if [ "${old}" = "${fixed}" ]; then knob_on "${old}" && m=fixed-point
        elif knob_on "${old}"; then m=index-order; else m=completion-order; fi
        if [ -n "${!new:-}" ] && [ -n "${m}" ] && [ "${!new}" != "${m}" ]; then
            echo "warning: ${old} is deprecated, use ${new}${m:+=${m}} (${old}=${!old} is not read: ${new}=${!new} wins)" >&2
        elif [ -z "${!new:-}" ] && [ -n "${m}" ] && [ "${m}" != "${v}" ]; then
            echo "warning: ${old} is deprecated, use ${new}=${m} (${old}=${!old} is not read: ${fixed}=${!fixed} wins)" >&2
        else
            echo "warning: ${old} is deprecated, use ${new}${m:+=${m}}" >&2
        fi
        unset "${old}"
    done
    if [ -z "${!new:-}" ] && [ -n "${v}" ]; then
        printf -v "${new}" '%s' "${v}"
        [ "${exp}" = "1" ] && export "${new}"
    fi
    return 0
}

# knob_on NAME: status 0 when the switch NAME is on (set, not empty, and not 0/false/no/off).
knob_on() {
    case "${!1:-}" in
        ""|0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff]) return 1 ;;
    esac
    return 0
}

# knob_bool NAME...: each switch normalised in place to 1 or 0 (knob_on's rule; an empty one is left empty), so the
# driver's own tests, banners and the knobs it hands on read one spelling. knob_gpu NAME: the same for a switch that
# also takes 'auto' (any case; empty is auto too).
knob_bool() {
    local k
    for k in "$@"; do
        [ -n "${!k:-}" ] || continue
        if knob_on "${k}"; then printf -v "${k}" '%s' 1; else printf -v "${k}" '%s' 0; fi
    done
}
knob_gpu() {
    case "${!1:-}" in
        ""|[Aa][Uu][Tt][Oo]) printf -v "$1" '%s' auto ;;
        *) knob_bool "$1" ;;
    esac
}

# Root of the simulation data, the folder that holds the simulations. Two layouts are recognised (see sim_dir):
#   per family  $DATA_ROOT/<family>/<sim>/snapdir_NNN/   e.g. TNG100/TNG100-3-Dark, TNG300/TNG300-3-Dark, ...
#   flat        $DATA_ROOT/<sim>/snapdir_NNN/            e.g. TNG100-3-Dark
# where <family> is the simulation name up to its first '-' (TNG100-3-Dark -> TNG100). A root that does not
# exist (a data disk not mounted) stops the run scripts rather than letting them write elsewhere.
DATA_ROOT="${DTFE_DATA_ROOT:-}"

# The directory of simulation $1 under DATA_ROOT, in whichever layout holds it: an existing
# <sim> or <family>/<sim> directory wins; a simulation not on disk yet goes into its family
# folder when that exists (so downloads land beside their siblings), else flat.
# Python: dtfelib.cli.sim_dir -- keep the two in step.
sim_dir() {
    local sim="$1" fam="${1%%-*}"
    if [ -d "${DATA_ROOT}/${sim}" ]; then
        echo "${DATA_ROOT}/${sim}"
    elif [ "${fam}" != "${sim}" ] && [ -d "${DATA_ROOT}/${fam}" ]; then
        echo "${DATA_ROOT}/${fam}/${sim}"
    else
        echo "${DATA_ROOT}/${sim}"
    fi
}

# Simulation to process; snapshots live in $(sim_dir $SIMULATION)/snapdir_NNN/. No default: -s or DTFE_SIM.
SIMULATION="${DTFE_SIM:-}"

# Stops a driver that was not told where the simulations are (after its option parsing, so --help still works).
# Every driver that reads or writes data calls it unless its -d names the simulation's folder itself.
require_data_root() {
    [ -n "${DATA_ROOT}" ] && return 0
    echo "Error: no data root: set DTFE_DATA_ROOT to the folder that holds the simulations (e.g. DTFE_DATA_ROOT=~/IllustrisTNG)." >&2
    exit 2
}

# Stops a driver that was not told which simulation to process (after its option parsing, so --help still works).
require_simulation() {
    [ -n "${SIMULATION}" ] && return 0
    echo "Error: no simulation given: pass -s SIMULATION (or set DTFE_SIM), e.g. -s TNG100-3-Dark." >&2
    exit 2
}

# Snapshot numbers (each corresponds to a redshift) the run scripts process by default;
# positional arguments to the scripts override this list.
SNAPSHOTS=(99)

# Interpolation grid cells per axis and padding (mean interparticle spacings).
# Environment-overridable like DATA_ROOT above. The run scripts source this file BEFORE parsing
# their flags, so precedence is:  -g flag  >  environment  >  the default here. PADDING's 50 is the PS-DTFE
# scripts' default; run_dtfe.sh has its own, 15 (complete at z = 0 on TNG50/100/300; see there).
# GRID_SIZE 512: the production grids of both estimators (run_ps_pipeline.sh's and the launcher's default too)
# are 512^3; 1024 would make a bare driver run a grid 8x the production one (~14 GB per field set, a scratch dir
# needed) that nothing downstream reads. DTFE_GRID_SIZE and DTFE_PADDING, their older second names, are read for one
# release. CALLER_GRID_SIZE / CALLER_PADDING: what the caller set (empty: nothing), for a script with a default of its
# own (run_dtfe.sh's padding 15).
knob GRID_SIZE DTFE_GRID_SIZE
knob PADDING DTFE_PADDING
CALLER_GRID_SIZE="${GRID_SIZE:-}"
CALLER_PADDING="${PADDING:-}"
GRID_SIZE="${GRID_SIZE:-512}"
PADDING="${PADDING:-50}"
# SPACE_CHECK=0 skips the free-space half of the preflight (output_space_problem); DTFE_SPACE_CHECK, its older name
knob SPACE_CHECK DTFE_SPACE_CHECK
SPACE_CHECK="${SPACE_CHECK:-1}"
knob_bool SPACE_CHECK

# Subdirectory prefix of the snapshot chunks inside the simulation directory.
INPUT_SUBDIR="snapdir"

# ---- run guard + preflight, shared by run_dtfe.sh and run_ps_dtfe.sh -------------------------------
# Functions only: nothing here runs or exits when the file is sourced (run_ps_pipeline.sh and
# download_snapshots.sh source it too and print their own messages).

# data_root_problem <data dir>: prints why the data cannot be there and returns 1 when the DATA ROOT
# itself is missing (a data disk not mounted, or mounted under another name); returns 0 otherwise. A
# missing SIMULATION under an existing root is not this function's business: the drivers keep skipping
# its snapshots and exit 1 with 'nothing processed'.
data_root_problem() {
    local dir="$1" root="${DATA_ROOT}"
    [ -d "${dir}" ] && return 0
    case "${dir}" in "${root}"|"${root}"/*) ;; *) root="$(dirname "${dir}")" ;; esac   # -d given: its parent
    [ -d "${root}" ] && return 0
    echo "Error: the data folder '${root}' does not exist: is the data disk mounted?" >&2
    echo "       Set DTFE_DATA_ROOT (or -d DATA_DIR) to where the snapshots are." >&2
    return 1
}

# expected_output_bytes <grid> <bytes per value> <fields...>: an APPROXIMATE size of the grids a run
# writes -- components per field from a TNG100 512^3 run's files (density 1, velocity 3, gradient 9,
# divergence 1, shear 5, vorticity 3, dispersion 1+6, tweb/vweb 1+3 eigenvalues), plus the stream count
# and the hidden-streams flag every PS run writes, plus 10% for the point files and the caustic grids.
expected_output_bytes() {
    local g="$1" b="$2" f n=2; shift 2
    for f in "$@"; do
        case "${f%_a}" in
            density|divergence) n=$((n + 1)) ;;
            velocity|vorticity) n=$((n + 3)) ;;
            gradient) n=$((n + 9)) ;;
            shear) n=$((n + 5)) ;;
            dispersion) n=$((n + 7)) ;;
            tweb|vweb) n=$((n + 4)) ;;
            *) n=$((n + 1)) ;;
        esac
    done
    echo $(( g * g * g * b * n * 11 / 10 ))
}

# output_space_problem <output dir> <bytes needed>: 1 (with the reason) when the folder the grids go to
# is not writable or has less free space than the grids need. The grids are written beside the old ones
# and renamed at the end (<file>.part.<pid>), so the old set does not count as free. SPACE_CHECK=0
# skips the space half (the estimate is approximate).
output_space_problem() {
    local dir="$1" need="$2" avail_kb
    if [ ! -w "${dir}" ]; then
        echo "  Error: the output folder '${dir}' is not writable" >&2
        return 1
    fi
    [ "${SPACE_CHECK:-1}" = "0" ] && return 0
    avail_kb="$(df -Pk "${dir}" 2>/dev/null | awk 'NR==2 {print $4}')"
    [[ "${avail_kb}" =~ ^[0-9]+$ ]] || return 0           # no df answer: do not guess
    if [ $((avail_kb * 1024)) -lt "${need}" ]; then
        echo "  Error: '${dir}' has $(awk -v k="${avail_kb}" 'BEGIN{printf "%.1f", k/1048576}') GB free; this run's grids need about $(awk -v b="${need}" 'BEGIN{printf "%.1f", b/1073741824}') GB (SPACE_CHECK=0 skips this check)" >&2
        return 1
    fi
    return 0
}

# The single-instance lock (shared contract C1): '<prefix>.runlog.lock' is a DIRECTORY made with mkdir
# (atomic on every filesystem; macOS has no flock(1)) holding 'pid', the driver's pid, for as long as a
# driver runs that snapshot into that prefix (two drivers on the same snapshot side by side both get
# SIGKILLed for memory). A lock whose pid is gone, or now
# belongs to a process that is not one of the drivers (a reused pid), is stale: removed with a message --
# unless a live process still works on that snapshot: a driver SIGKILLed on its own leaves its
# binary (and tee) running, and removing that lock would start a second 40-50 GB binary beside the orphan.
# run_lock_take <lock dir>: 0 = taken; 1 = held by a live driver or process (the refusal printed).
# run_lock_live_process <output root>: prints 'pid args' of the first live process naming the root on its
# command line (the binary's output argument, or tee's '<root>.runlog.partial'), as the launcher's
# _live_roots does (python/gui/results.py); status 1 when there is none or ps cannot say
run_lock_live_process() {
    local root="$1" line pid args
    [ -n "${root}" ] || return 1
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"      # ps right-aligns the pid
        pid="${line%% *}"; args="${line#* }"
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        [ "${pid}" = "$$" ] && continue
        case " ${args} " in
            *" ${root} "*|*" ${root}."*)
                echo "${pid} ${args}"; return 0 ;;
        esac
    done < <(ps -A -ww -o pid=,args= 2>/dev/null)
    return 1
}
run_lock_take() {
    local lock="$1" pid cmd tries live root
    root="${lock%.runlog.lock}"
    for tries in 1 2; do
        if mkdir "${lock}" 2>/dev/null; then
            echo "$$" > "${lock}/pid"
            return 0
        fi
        [ -d "${lock}" ] || { echo "  Error: cannot create the lock '${lock}' (is the output folder writable?)" >&2; return 1; }
        pid="$(cat "${lock}/pid" 2>/dev/null || true)"
        if [ -z "${pid}" ]; then                     # mkdir done, pid not yet written: give it a moment
            sleep 1; pid="$(cat "${lock}/pid" 2>/dev/null || true)"
        fi
        cmd=""
        [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null && cmd="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
        case "${cmd}" in
            *run_dtfe.sh*|*run_ps_dtfe.sh*)
                echo "  Error: another driver (pid ${pid}) is running this snapshot into the same output:" >&2
                echo "         ${lock}" >&2
                echo "         Two runs of one snapshot do not fit in memory together; wait for it, or stop it." >&2
                return 1 ;;
        esac
        # the driver is gone; its binary may not be (the lock's root is '<root>.runlog.lock')
        if [ "${root}" != "${lock}" ] && live="$(run_lock_live_process "${root}")"; then
            echo "  Error: the driver that locked this snapshot (pid ${pid:-unknown}) is gone, but a process still" >&2
            echo "         works on it (pid ${live%% *}): ${live#* }" >&2
            echo "         Two runs of one snapshot do not fit in memory together; wait for it, or stop it" >&2
            echo "         (kill ${live%% *}). The lock stays: ${lock}" >&2
            return 1
        fi
        echo "  Note: removing a stale lock (pid ${pid:-unknown} is no longer running): ${lock}"
        rm -rf "${lock}"
    done
    echo "  Error: could not take the lock '${lock}'" >&2
    return 1
}
# run_lock_release <lock dir>: removes it when it is this driver's own (never another driver's)
run_lock_release() {
    [ -n "$1" ] && [ -d "$1" ] || return 0
    [ "$(cat "$1/pid" 2>/dev/null)" = "$$" ] && rm -rf "$1"
    return 0
}

# gpu_build_default <repo root> <binary>: 1 when the Makefile built <binary> (./DTFE, ./DTFE-double, ./PS-DTFE,
# ./PS-DTFE-double) with a GPU backend, else 0 -- the drivers' default for DTFE_GPU ('auto': every
# production grid runs on the GPU, so a driver run without -m does too). The answer is the stamp the
# Makefile's mode check leaves in the pair's object folder (o, o_d, o_ps, o_ps_d): '.gpu_mode_metal' (a CPU-only
# build leaves '.gpu_mode_off'). The binary itself cannot be asked: '--help' lists --gpu in every build and
# '--version' names no backend. No object folder (binaries copied without their build) = 0. A stale stamp (a binary put
# back by install_lib.sh after a failed rebuild in another mode) costs nothing: '--gpu' on a CPU build falls back to
# the CPU with a warning, and the CPU on a GPU build is slower, not different.
gpu_build_default() {
    local root="$1" obj
    case "${2##*/}" in
        DTFE) obj=o ;;
        DTFE-double) obj=o_d ;;
        PS-DTFE) obj=o_ps ;;
        PS-DTFE-double) obj=o_ps_d ;;
        *) echo 0; return 0 ;;
    esac
    [ -e "${root}/${obj}/.gpu_mode_metal" ] && { echo 1; return 0; }
    echo 0
}

# ---- a rerun's left-over files -------------------------------------------------------------------------------
# The binary replaces its outputs as a set (output_staging.h: '<file>.part.<pid>' renamed into place only once all
# are written), but it never removes what an EARLIER run with the same prefix wrote and this one does not: a rerun
# without the image plane, the caustics or the T-web would leave the old '.pts_*', '.caustic*' or '.a_tweb*' beside
# the new grids, a mixed set that reads as one run (the README promises a folder never holds a mix). So the drivers,
# after a run that EXITED 0, move every '<prefix>.*' file the run did not write into '<prefix>.superseded/<stamp>/'
# beside them -- moved, never deleted -- with a copy of the old run's log, so the moved files keep their provenance.
# 'Did not write': the file existed before the run with the same inode (the binary renames every output into place,
# so a written file is a new inode) AND is not newer than the run's start (the one output written in place, a
# thread's '.badGridPoints<n>', keeps its inode but not its time). A failed, stopped or killed run moves nothing:
# the old set stays complete, as the run-log contract keeps its log. Neither a dry run nor an auto-tune report
# writes anything, so neither moves anything. Not moved: the run's own bookkeeping ('<prefix>.runlog' and every
# '<prefix>.runlog.*': the failed / partial log, the lock, the progress sidecar, a PS checkpoint), the launcher's
# settings sidecar '<prefix>.gui.json' and the binary's own '<file>.part.<pid>' (output_staging.h handles those).
# The launcher's Runs browser does not see the folder (it reads '<snapdir>/<prefix>.runlog*' and only the FILES
# '<prefix>.*' as a run's outputs).
#
# supersede_snapshot <output root>: what a run may later supersede, recorded BEFORE it starts -- one line
# '<inode> <name>' per candidate file (stdout); the caller keeps the text and passes it back afterwards.
supersede_snapshot() {
    local root="$1" f n ino
    for f in "${root}".*; do
        [ -f "${f}" ] && [ ! -L "${f}" ] || continue        # files only: '.superseded/', '.runlog.lock/' are folders
        n="${f##*/}"
        case "${n#"${root##*/}"}" in
            .runlog|.runlog.*|.gui.json|*.part.[0-9]*) continue ;;
        esac
        ino="$(ls -id -- "${f}" 2>/dev/null | awk '{ print $1 }')"
        [[ "${ino}" =~ ^[0-9]+$ ]] || continue
        printf '%s %s\n' "${ino}" "${n}"
    done
    return 0
}
# supersede_extras <output root> <recorded list> <run start marker file> <saved old runlog or ''>: after a run that
# exited 0, moves each recorded file the run did not write (same inode, not newer than the marker) into
# '<root>.superseded/<YYYY-MM-DD_HHMMSS>/' (the marker's time; '-<pid>' added when that folder exists), with the
# saved old log as '<prefix>.runlog' there. Prints one line per move and a summary; a failed move is said and left
# (the run itself succeeded). Never deletes, never overwrites ('mv -n', checked). Returns 0.
supersede_extras() {
    local root="$1" list="$2" marker="$3" oldlog="$4" dir pfx line ino n f now dest moved=0 failed=0
    [ -n "${list}" ] || return 0
    dir="${root%/*}"; pfx="${root##*/}"
    while IFS= read -r line; do
        [ -n "${line}" ] || continue
        ino="${line%% *}"; n="${line#* }"; f="${dir}/${n}"
        [ -f "${f}" ] || continue                                  # gone (the binary removed it): nothing to do
        now="$(ls -id -- "${f}" 2>/dev/null | awk '{ print $1 }')"
        [ "${now}" = "${ino}" ] || continue                        # a new inode: this run wrote it
        [ -n "${marker}" ] && [ "${f}" -nt "${marker}" ] && continue   # rewritten in place by this run
        if [ -z "${dest:-}" ]; then
            dest="${root}.superseded/$(date -r "${marker}" '+%Y-%m-%d_%H%M%S' 2>/dev/null || date '+%Y-%m-%d_%H%M%S')"
            [ -e "${dest}" ] && dest="${dest}-$$"
            if ! mkdir -p "${dest}"; then
                echo "  Warning: cannot create '${dest}': the earlier run's files this run did not write stay beside the new grids" >&2
                return 0
            fi
        fi
        if [ -e "${dest}/${n}" ] || ! mv -n "${f}" "${dest}/${n}" || [ -e "${f}" ]; then
            echo "  Warning: could not move '${n}' into ${dest} (left in place)" >&2
            failed=$((failed + 1)); continue
        fi
        echo "  Superseded: ${n} (an earlier run's; this run did not write it)"
        moved=$((moved + 1))
    done <<< "${list}"
    if [ "${moved}" -gt 0 ] && [ -n "${oldlog}" ] && [ -f "${oldlog}" ] && [ ! -e "${dest}/${pfx}.runlog" ]; then
        cp -p "${oldlog}" "${dest}/${pfx}.runlog" 2>/dev/null || true
    fi
    if [ "${moved}" -gt 0 ]; then
        echo "  ${moved} file(s) of an earlier run with this prefix moved to ${dest} (nothing deleted; its run log copied there)"
    fi
    [ -n "${dest:-}" ] && [ "${moved}" -eq 0 ] && rmdir "${dest}" "${root}.superseded" 2>/dev/null
    return 0
}

# ps_checkpoint_dir <output root> [<scratch dir>]: where PS-DTFE -- and standard DTFE, by the same rule -- keeps
# the checkpoint of a partitioned run (src/partition_checkpoint.h directoryFor -- keep the two in step):
# '<output>.runlog.checkpoint' beside the outputs, or with a scratch dir '<scratch>/<basename>.<FNV-1a 64 of the
# absolute output path, 16 hex>.runlog.checkpoint'. run_ps_dtfe.sh and run_dtfe.sh pass '--resume' when its 'checkpoint'
# file exists. The absolute path is the one the binary sees: a relative output is taken from the physical current
# directory (getcwd, so 'pwd -P').
ps_checkpoint_dir() {
    local out="$1" scratch="${2:-}"
    if [ -z "${scratch}" ]; then
        printf '%s.runlog.checkpoint\n' "${out}"
        return 0
    fi
    case "${out}" in /*) ;; *) out="$(pwd -P)/${out}" ;; esac
    local LC_ALL=C h=-3750763034362895579 i c   # the FNV-1a offset basis 14695981039346656037, as a signed 64-bit
    for (( i = 0; i < ${#out}; i++ )); do
        printf -v c '%d' "'${out:i:1}"
        h=$(( (h ^ (c & 255)) * 1099511628211 ))   # bash wraps at 64 bits, as the C++ uint64_t does
    done
    while [ "${scratch}" != "/" ] && [ "${scratch%/}" != "${scratch}" ]; do scratch="${scratch%/}"; done
    printf '%s/%s.%016x.runlog.checkpoint\n' "${scratch}" "${out##*/}" "${h}"
}

# ---- the drivers' shared steps (run_dtfe.sh, run_ps_dtfe.sh; snapshot_args also download_snapshots.sh) -------
# They read and set the drivers' own variables where a comment says so: n_str (the snapshot being run, '099'),
# DATA_DIR, OUTPUT_PREFIX, GRID_SIZE, FIELDS, SCRATCH_DIR, AUTO_TUNE_REPORT, REPO_ROOT and the run-log state of
# runlog_contract_start.
#
# Strictness: the batch drivers (run_dtfe.sh, run_ps_dtfe.sh, run_ps_pipeline.sh, download_snapshots.sh) run under
# 'set -uo pipefail' -- an unset variable is an error, and a pipeline fails when any part of it fails -- but NOT -e,
# on purpose: a batch goes on after a failed snapshot (a binary that exits non-zero, a missing input, a refused lock,
# a failed fetch), counts it and moves to the next, and its exit status says so at the end. Under -e the first failing
# command would end the whole batch, and so would a '[ ... ] && ...' line whose test is false when it ends a
# function. So each command whose failure matters is checked where it runs. Everything here must hold under
# 'set -u' (a value that may be unset is read as "${VAR:-}", an array that may be empty as ${ARR[@]+"${ARR[@]}"}:
# bash 3.2, macOS's /bin/bash, calls even an empty array unset) and run on bash 3.2: no associative arrays, no ';&'
# or ';;&', no ${var,,} / ${var^^}, no mapfile / readarray.

# driver_options <own option letters> <handler> "$@": the option loop of run_dtfe.sh, run_ps_dtfe.sh and
# download_snapshots.sh. The options they share are handled here: -d DATA_DIR (the simulation's own folder),
# -s SIMULATION and -h (the caller's 'usage', exit 0); an unknown option, or one missing its value, prints getopts'
# message and the usage and exits 1. Each of the caller's own letters (getopts' spelling: 'g:' takes a value) goes to
# '<handler> <letter>', with its value in OPTARG. Then the two checks every driver makes after its options (so that -h
# works without them): the data root (require_data_root) unless -d named the simulation's folder, and the
# simulation (require_simulation). The caller drops the options afterwards: shift $((OPTIND - 1)).
# The long forms (driver_long_options) are read as their letters, where the caller has the letter.
driver_options() {
    local own="$1" handler="$2" opt
    shift 2
    driver_long_options "d:s:${own}h" "$@"
    set -- ${_drv_args[@]+"${_drv_args[@]}"}
    OPTIND=1
    while getopts "d:s:${own}h" opt; do
        case "${opt}" in
            d) DATA_DIR="${OPTARG}" ;;
            s) SIMULATION="${OPTARG}" ;;
            h) usage; exit 0 ;;
            "?") usage; exit 1 ;;
            *) "${handler}" "${opt}" ;;
        esac
    done
    [ -n "${DATA_DIR:-}" ] || require_data_root   # -d names the simulation's folder itself
    require_simulation
}

# driver_long_options <getopts spec> "$@": _drv_args = the arguments with each long option the drivers share spelled
# as its letter, token for token (so the caller's 'shift $((OPTIND - 1))' on its own arguments stays right):
#   --sim S -> -s S, --sim-dir D -> -d D, --grid N -> -g N, --gpu -> -m, --exact-average -> -e, --subsamples K -> -n K,
#   --help -> -h; download_snapshots.sh's --groupcats -> -c, --trees -> -t, --ics -> -i, --api-key K -> -k K
#   (merge_HDF5.py's names for the same products); '--grid=N' -> '-gN'.
# Only up to the first argument that is not an option (or '--'), as getopts reads them, and a letter's value is never
# translated (-s --gpu names a simulation '--gpu'). A long option whose letter the script lacks, or an unknown one,
# prints the usage and exits 1, as an unknown letter does.
driver_long_options() {
    local spec="$1" t name val letter i c want=0 stop=0
    shift
    _drv_args=()
    for t in "$@"; do
        if [ "${stop}" = "1" ] || [ "${want}" = "1" ]; then _drv_args+=("${t}"); want=0; continue; fi
        case "${t}" in
            --)  _drv_args+=("${t}"); stop=1 ;;
            --*)
                name="${t%%=*}"; val=""
                case "${t}" in *=*) val="${t#*=}" ;; esac
                case "${name}" in
                    --sim) letter=s ;; --sim-dir) letter=d ;; --grid) letter=g ;; --gpu) letter=m ;;
                    --exact-average) letter=e ;; --subsamples) letter=n ;; --help) letter=h ;;
                    --groupcats) letter=c ;; --trees) letter=t ;; --ics) letter=i ;; --api-key) letter=k ;;
                    *) letter="" ;;
                esac
                case "${spec}" in *"${letter}"*) ;; *) letter="" ;; esac
                if [ -z "${letter}" ]; then
                    echo "Error: unknown option '${name}'." >&2; usage; exit 1
                fi
                case "${spec}" in
                    *"${letter}:"*)
                        if [ "${t}" = "${name}" ]; then _drv_args+=("-${letter}"); want=1
                        elif [ -n "${val}" ]; then _drv_args+=("-${letter}${val}")
                        else echo "Error: option '${name}' needs a value." >&2; usage; exit 1
                        fi ;;
                    *)
                        if [ "${t}" != "${name}" ]; then
                            echo "Error: option '${name}' takes no value." >&2; usage; exit 1
                        fi
                        _drv_args+=("-${letter}") ;;
                esac ;;
            -?*)
                _drv_args+=("${t}")
                for (( i = 1; i < ${#t}; i++ )); do       # a letter that takes a value ends the cluster
                    c="${t:i:1}"
                    case "${spec}" in *"${c}:"*) [ "${i}" = "$(( ${#t} - 1 ))" ] && want=1; break ;; esac
                done ;;
            *)   _drv_args+=("${t}"); stop=1 ;;
        esac
    done
}

# driver_help_version <script> [its first argument]: '-h' / '--help' prints the script's header (line 2 up to the
# first blank line, '# ' stripped) and the caller's usage line (its 'usage' function); '--version' prints the
# repository revision. Both exit 0; any other argument returns.
driver_help_version() {
    case "${2:-}" in
        -h|--help) sed -n '2,/^$/p' "$1" | sed -e 's/^# \{0,1\}//'; usage; exit 0 ;;
        --version) git -C "${REPO_ROOT}" describe --always --dirty 2>/dev/null || echo unknown; exit 0 ;;
    esac
}

# snapshot_args [snapshot ...]: positional arguments, when there are any, replace SNAPSHOTS; every entry is then
# read as a decimal number (bash's printf reads a leading zero as OCTAL, so '010' would become 008 and '099' an
# error, then 000 -- the folders are named snapdir_0NN, so typing them that way is natural). A non-number exits 2.
snapshot_args() {
    local _k _s
    if [ "$#" -gt 0 ]; then
        SNAPSHOTS=("$@")
    fi
    for _k in "${!SNAPSHOTS[@]}"; do
        _s="${SNAPSHOTS[$_k]}"
        if ! [[ "${_s}" =~ ^[0-9]+$ ]]; then
            echo "Error: snapshot '${_s}' is not a number." >&2
            exit 2
        fi
        SNAPSHOTS[$_k]=$((10#${_s}))
    done
}

# driver_binary <variable> <binary> <precision>: sets <variable> to './<binary>' for precision 'single' or to
# './<binary>-double' for 'double' (the DTFE_PRECISION knob); exits 1 for another precision, or when that binary is
# not executable (naming the make command that builds it). Nothing falls back silently.
driver_binary() {
    local bin
    case "$3" in
        single) bin="./$2" ;;
        double) bin="./$2-double" ;;
        *) echo "Error: DTFE_PRECISION must be 'single' or 'double' (got '$3')." >&2; exit 1 ;;
    esac
    printf -v "$1" '%s' "${bin}"
    if [ ! -x "${bin}" ]; then
        if [ "$3" = "double" ]; then
            echo "Error: ${bin} not found. Build the double-precision pair with 'make DTFE PS-DTFE DOUBLE=1' (add METAL=1 on Apple Silicon), or let the launcher's Setup do it." >&2
        else
            echo "Error: ${bin} not found or not executable. Build it with 'make $2'." >&2
        fi
        exit 1
    fi
}

# gpu_auto_resolve <knob> <binary> [cpu_first knob]: the GPU knob (DTFE_GPU) left at 'auto' (or empty) becomes
# gpu_build_default's answer for <binary> (1 on a GPU build), and gpu_auto the note the start banner adds; a knob set
# to an on or off value is kept as 1 or 0 (knob_gpu) and gpu_auto is empty. When the switch named by cpu_first is on
# (knob_on), 'auto' is the CPU on every build: run_ps_dtfe.sh names DTFE_EXACT_AVERAGE, as PS-DTFE's exact deposit is
# faster on the CPU on production runs (TNG100-3-Dark z=0 at 512^3: 14.5 min against 17.7 min on the GPU).
gpu_auto_resolve() {
    local knob="$1" bin="$2" cpu_first="${3:-}" v
    gpu_auto=""
    knob_gpu "${knob}"
    [ "${!knob}" = "auto" ] || return 0
    v="$(gpu_build_default "${REPO_ROOT}" "${bin}")"
    if [ -n "${cpu_first}" ] && knob_on "${cpu_first}" && [ "${v}" = "1" ]; then
        printf -v "${knob}" '%s' 0
        gpu_auto=" (${knob} unset: the exact deposit is faster on the CPU; ${knob}=1 runs it on the GPU)"
        return 0
    fi
    printf -v "${knob}" '%s' "${v}"
    gpu_auto=" (${knob} unset: ${bin#./} is $([ "${v}" = "1" ] && echo "a GPU" || echo "a CPU-only") build; ${knob}=0 or 1 chooses)"
}

# time_wrap_resolve: TIME_WRAP, the peak-memory wrapper put in front of the binary, resolved once and OPTIONAL:
# macOS spells the flag -l, GNU -v, a bare container has no /usr/bin/time. Without one TIME_WRAP is empty and the
# run identical, just with no peak-memory line.
time_wrap_resolve() {
    TIME_WRAP=()
    if [ -x /usr/bin/time ]; then
        if /usr/bin/time -l true >/dev/null 2>&1; then
            TIME_WRAP=(/usr/bin/time -l)          # BSD/macOS: "<bytes>  maximum resident set size"
        elif /usr/bin/time -v true >/dev/null 2>&1; then
            TIME_WRAP=(/usr/bin/time -v)          # GNU: "Maximum resident set size (kbytes): <n>"
        fi
    fi
}

# keep_awake <DRY_RUN>: keeps the Mac awake for the whole batch (caffeinate follows this shell's pid and lifts on
# exit); not for a dry run, nor without caffeinate.
keep_awake() {
    if [ "$1" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
        caffeinate -i -m -w $$ &
    fi
}

# The run-log contract, the same in run_dtfe.sh, run_ps_dtfe.sh and the launcher's own runs: while the
# binary runs, its log is '<prefix>.runlog.partial'. Exit 0 renames it onto '<prefix>.runlog' (the log
# of the grids now on disk) and removes a stale '<prefix>.runlog.failed'; a failure or a stop renames it
# to '<prefix>.runlog.failed' and leaves '<prefix>.runlog' alone. A log tee'd straight into
# '<prefix>.runlog' would let a re-run that failed early, or was stopped or killed, replace the log of the
# grids still on disk: the Runs browser would show those grids as a failed run with the new settings,
# and their own settings, build stamp, wall time and peak memory would be gone. A '.runlog.partial' left
# behind is a run killed outright (SIGKILL), or one still going.
#
# runlog_contract_start: the driver's run-log state, empty, and its traps. A stop (the launcher's Stop sends
# SIGTERM to the binary, then to the driver; ^C sends SIGINT) runs on_stop once the binary's pipeline has ended;
# any other way out (an error, set -u) still releases the lock and the temp files.
runlog_contract_start() {
    live_log=""; live_rc=""            # the .partial being written, and its binary's exit status once known
    live_lock=""                       # this driver's '<prefix>.runlog.lock' (run_lock_take), while held
    sup_marker=""; sup_oldlog=""       # the run's start marker and a copy of the old run log (supersede_extras)
    trap 'on_stop INT' INT; trap 'on_stop TERM' TERM; trap 'on_stop HUP' HUP
    trap 'run_lock_release "${live_lock}"; sup_clean' EXIT
}
sup_clean() { rm -f ${sup_marker:+"${sup_marker}"} ${sup_oldlog:+"${sup_oldlog}"}; sup_marker=""; sup_oldlog=""; }
finish_runlog() {                  # <the .partial> <the binary's exit status>
    local part="$1" final="${1%.partial}"
    [ -f "${part}" ] || return 0
    # Strip ANSI colour codes so the kept log stays grep-able. Written through a temp file rather
    # than 'sed -i': BSD sed needs a suffix ARGUMENT after -i and GNU sed does not, so the portable
    # spelling of in-place editing is not to use it.
    if sed -E $'s/\033\\[[0-9;]*m//g' "${part}" > "${part}.tmp" 2>/dev/null; then
        mv "${part}.tmp" "${part}"
    else
        rm -f "${part}.tmp"
    fi
    if [ "$2" = "0" ]; then
        mv -f "${part}" "${final}" && rm -f "${final}.failed"
    else
        mv -f "${part}" "${final}.failed"
    fi
}
# on_stop <signal name>: the attempt's log becomes '.runlog.failed', the lock and the temp files are released, and
# the signal is re-raised, so a caller (run_ps_pipeline.sh) sees a stop, not a plain failure it would continue past.
on_stop() {
    trap - "$1"
    if [ -n "${live_log}" ] && [ -f "${live_log}" ]; then
        finish_runlog "${live_log}" "${live_rc:-1}"
        [ "${live_rc:-1}" = "0" ] || echo "  Stopped: snapshot ${n_str} did not finish; its log is ${live_log%.partial}.failed"
    fi
    run_lock_release "${live_lock}"
    sup_clean
    kill -s "$1" $$
}

# snapshot_start <snapshot number>: n_str ('099'), input_dir (<DATA_DIR>/snapdir_099), input_file (its
# combined_099.hdf5) and output_root (<input_dir>/<OUTPUT_PREFIX>: the outputs are <output_root>.*) of the snapshot,
# and its 'Processing' line. Returns 1, after saying so, when the input file is missing (the snapshot is skipped).
snapshot_start() {
    n_str=$(printf "%03d" "$1")
    input_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
    input_file="${input_dir}/combined_${n_str}.hdf5"
    output_root="${input_dir}/${OUTPUT_PREFIX}"
    echo "Processing snapshot ${n_str}..."
    if [ ! -f "${input_file}" ]; then
        echo "  Warning: Input file not found: ${input_file}"
        echo "  Skipping snapshot ${n_str}"
        return 1
    fi
}

# resume_args_for <output root> <resume knob> <its value>: resume_args=(--resume) when a killed partitioned run left
# its checkpoint (the binary removes it once the outputs are in place; ps_checkpoint_dir, under SCRATCH_DIR when one
# is set) and the knob (DTFE_RESUME) is not 0; either way it says so. Empty for a report
# (AUTO_TUNE_REPORT=1 computes nothing, so there is nothing to resume). The FILE, not the folder: a folder with no
# 'checkpoint' in it (a kill during the first write) holds nothing to resume, the binary refuses '--resume' there,
# and every rerun of the snapshot would fail the same way.
resume_args_for() {
    local ckpt_dir
    resume_args=()
    ckpt_dir="$(ps_checkpoint_dir "$1" "${SCRATCH_DIR}")"
    if [ "${AUTO_TUNE_REPORT}" != "1" ] && [ -f "${ckpt_dir}/checkpoint" ]; then
        if [ "$3" = "0" ]; then
            echo "  Note: a killed run's checkpoint is in ${ckpt_dir}; $2=0, so this run starts over and replaces it."
        else
            echo "  Resuming: a killed run's checkpoint is in ${ckpt_dir}; passing --resume, only the missing partitions are computed ($2=0 starts over)."
            resume_args=(--resume)
        fi
    fi
}

# dry_run_print <input file> <output root> <run log> <command...>: what DRY_RUN=1 prints for a snapshot (the
# expanded command and the paths) instead of running it.
dry_run_print() {
    local input="$1" root="$2" log="$3"; shift 3
    echo "  DRY RUN -- would run:"
    printf '    %q' "$@"; echo
    echo "  input:   ${input} (exists)"
    echo "  outputs: ${root}.* and the run log ${log}"
}

# auto_tune_report_line <temp log> <the binary's exit status>: prints the snapshot's AUTO-TUNE-REPORT line from the
# report run's temp log and removes the log; returns 1, with an error line, when the run failed or wrote no report.
auto_tune_report_line() {
    local report
    report=$(grep -a '^AUTO-TUNE-REPORT' "$1" | head -1)
    rm -f "$1"
    if [ "$2" -eq 0 ] && [ -n "${report}" ]; then
        echo "  AUTO-TUNE-REPORT snap=${n_str} ${report#AUTO-TUNE-REPORT }"
        return 0
    fi
    echo "  Error: no auto-tune report for snapshot ${n_str} (exit $2)"
    return 1
}

# snapshot_preflight <input dir> <run log> <precision>: room in <input dir> for the grids of GRID_SIZE and FIELDS
# (output_space_problem; 8 bytes a value for precision 'double', else 4), then the snapshot's single-instance lock
# '<run log>.lock' (run_lock_take), held in live_lock. Returns 1, with the reason, when either refuses: nothing ran.
snapshot_preflight() {
    local bpv=4
    [ "$3" = "double" ] && bpv=8
    if ! output_space_problem "$1" "$(expected_output_bytes "${GRID_SIZE%% *}" "${bpv}" ${FIELDS})"; then
        echo "  Error processing snapshot ${n_str}: no room for its grids (nothing was run)"
        return 1
    fi
    if ! run_lock_take "$2.lock"; then
        echo "  Error processing snapshot ${n_str}: refused, another run holds it (nothing was run)"
        return 1
    fi
    live_lock="$2.lock"
}

# sup_record <output root> <run log>: before a run, what an earlier run with this prefix left (sup_list,
# supersede_snapshot), the run's start marker (sup_marker) and, when there is something to supersede, a copy of the
# old run log (sup_oldlog). What the run does not rewrite is moved aside once it has exited 0 (run_finish).
sup_record() {
    sup_clean
    sup_marker="$(mktemp "${TMPDIR:-/tmp}/dtfe-run-start.XXXXXX" 2>/dev/null)" || sup_marker=""
    sup_list="$(supersede_snapshot "$1")"
    if [ -n "${sup_list}" ] && [ -f "$2" ]; then
        sup_oldlog="$(mktemp "${TMPDIR:-/tmp}/dtfe-old-runlog.XXXXXX" 2>/dev/null)" && cp -p "$2" "${sup_oldlog}" \
            || sup_oldlog=""
    fi
}

# run_finish <the binary's exit status> <output root>: after the run, the de-ANSI'd log becomes '<prefix>.runlog'
# on success and '<prefix>.runlog.failed' otherwise (finish_runlog); on exit 0 the earlier run's files this one did
# not write are moved aside (supersede_extras); the temp files and the lock are released.
run_finish() {
    finish_runlog "${live_log}" "$1"
    live_log=""
    [ "$1" -eq 0 ] && supersede_extras "$2" "${sup_list}" "${sup_marker}" "${sup_oldlog}"
    sup_clean
    run_lock_release "${live_lock}"; live_lock=""
}

# peak_memory_text <run log>: the run's peak memory from the TIME_WRAP lines in its log, as 'F GB footprint, R GB
# resident' or 'R GB resident'; nothing without such a line. macOS also reports the peak FOOTPRINT: what the run
# really needed, including the memory the system compressed or swapped out -- RSS alone hid 90 GB behind "47 GB" at
# TNG100-3 z=0. The dialects:
#   BSD/macOS : "  1234567  maximum resident set size"            -- BYTES in $1
#   GNU       : "  Maximum resident set size (kbytes): 1234"      -- KBYTES in $NF
peak_memory_text() {
    local peak_bytes peak_gb fp_bytes
    peak_bytes=$(awk 'tolower($0) ~ /maximum resident set size/ {
                          if (tolower($0) ~ /kbytes/) { gsub(/[^0-9]/, "", $NF); print $NF * 1024 }
                          else                        { print $1 }
                          exit }' "$1")
    [ -n "${peak_bytes}" ] || return 0
    peak_gb=$(awk -v b="${peak_bytes}" 'BEGIN{printf "%.1f", b/1073741824}')
    fp_bytes=$(awk 'tolower($0) ~ /peak memory footprint/ { print $1; exit }' "$1")
    if [ -n "${fp_bytes}" ]; then
        echo "$(awk -v b="${fp_bytes}" 'BEGIN{printf "%.1f", b/1073741824}') GB footprint, ${peak_gb} GB resident"
    else
        echo "${peak_gb} GB resident"
    fi
}

# snapshot_outcome <the binary's exit status> <its log>: the wall time (SECONDS since the run started) and the
# snapshot's result line. Exit 3 means the run finished but its output folder could not be written, so the binary
# put the grids in its rescue folder: the log's OUTPUT RESCUE line names it and the mv that puts them in place.
# Returns 0 for a processed snapshot, 1 for a failed one.
snapshot_outcome() {
    echo "  Wall time: $((SECONDS/60)) min $((SECONDS%60)) s"
    if [ "$1" -eq 0 ]; then
        echo "  Snapshot ${n_str} processed successfully"
        return 0
    fi
    echo "  Error processing snapshot ${n_str} (its log: $2)"
    [ "$1" -eq 3 ] && echo "  The grids were computed and saved to the rescue folder: see the OUTPUT RESCUE line in $2"
    return 1
}

# dry_run_exit <planned> <skipped> <data dir>: a dry run's summary line; exit 0 when a snapshot was planned, else 1.
dry_run_exit() {
    echo "DRY RUN: $1 snapshot(s) planned, $2 skipped for missing inputs; nothing was run."
    [ "$1" -gt 0 ] && exit 0
    echo "ERROR: nothing to run (looked under '$3')." >&2
    exit 1
}
