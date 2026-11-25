# Shared configuration for run_dtfe.sh, run_ps_dtfe.sh, run_ps_pipeline.sh and download_snapshots.sh (sourced).
# Every value can be overridden per run by environment (DTFE_DATA_ROOT=/path DTFE_SIM=TNG50-3-Dark GRID_SIZE=256 ...)
# or by script flag (-s, -d, -g; long forms --sim, --sim-dir, --grid, --gpu, --exact, --sampled, --subsamples, --run-mode).
# The data root and the simulation have NO default: every run sets DTFE_DATA_ROOT (require_data_root; a driver's -d
# stands in for it) and names the simulation with -s or DTFE_SIM (require_simulation). dtfelib.cli honours the same variables.

source "$(dirname "${BASH_SOURCE[0]}")/h5py_python.sh"   # h5py_python: the python with numpy and h5py

# ---- the knobs' values ---------------------------------------------------------------------------------------
# An EMPTY value is the same as an unset one, and a switch is off for 0, false, no or off (any case) and on for
# anything else -- the binaries' own rule (src/env_knobs.h). A knob with a default gives it on one
# 'VAR="${VAR:-default}"' line (tests/docs_drift_check.py and py_gui_runspec_test.py read those lines).

# knob_on NAME: status 0 when the switch NAME is on (set, not empty, and not 0/false/no/off).
knob_on() {
    case "${!1:-}" in
        ""|0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff]) return 1 ;;
    esac
    return 0
}

# knob_bool NAME...: each switch normalised in place to 1 or 0 (knob_on's rule; an empty one is left empty).
knob_bool() {
    local k
    for k in "$@"; do
        [ -n "${!k:-}" ] || continue
        if knob_on "${k}"; then printf -v "${k}" '%s' 1; else printf -v "${k}" '%s' 0; fi
    done
}

# Root of the simulation data, the folder that holds the simulations. Two layouts are recognised (see sim_dir):
#   per family  $DATA_ROOT/<family>/<sim>/snapdir_NNN/   e.g. TNG100/TNG100-3-Dark
#   flat        $DATA_ROOT/<sim>/snapdir_NNN/            e.g. TNG100-3-Dark
# A root that does not exist (a data disk not mounted) stops the run scripts rather than letting them write elsewhere.
DATA_ROOT="${DTFE_DATA_ROOT:-}"

# The directory of simulation $1 under DATA_ROOT: an existing <sim> or <family>/<sim> directory wins; else the
# family folder when that exists, else flat. Python: dtfelib.cli.sim_dir -- keep the two in step.
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

# Snapshot numbers the run scripts process by default; positional arguments override this list.
SNAPSHOTS=(99)

# Interpolation grid cells per axis and padding (mean interparticle spacings), environment-overridable. The run
# scripts source this file BEFORE parsing their flags, so precedence is: -g flag > environment > the default here.
# PADDING's 50 is the PS-DTFE scripts' default; run_dtfe.sh has its own, 15. GRID_SIZE 512 is the production grid.
# CALLER_GRID_SIZE / CALLER_PADDING: what the caller set (empty: nothing), for a script with a default of its own.
CALLER_GRID_SIZE="${GRID_SIZE:-}"
CALLER_PADDING="${PADDING:-}"
GRID_SIZE="${GRID_SIZE:-512}"
PADDING="${PADDING:-50}"
# SPACE_CHECK=0 skips the free-space half of the preflight (output_space_problem)
SPACE_CHECK="${SPACE_CHECK:-1}"
knob_bool SPACE_CHECK

# Subdirectory prefix of the snapshot chunks inside the simulation directory.
INPUT_SUBDIR="snapdir"

# ---- run guard + preflight, shared by run_dtfe.sh and run_ps_dtfe.sh -------------------------------
# Functions only: nothing here runs or exits when the file is sourced.

# data_root_problem <data dir>: prints why the data cannot be there and returns 1 when the DATA ROOT itself is
# missing (a data disk not mounted); returns 0 otherwise. A missing SIMULATION under an existing root is not its business.
data_root_problem() {
    local dir="$1" root="${DATA_ROOT}"
    [ -d "${dir}" ] && return 0
    case "${dir}" in "${root}"|"${root}"/*) ;; *) root="$(dirname "${dir}")" ;; esac   # -d given: its parent
    [ -d "${root}" ] && return 0
    echo "Error: the data folder '${root}' does not exist: is the data disk mounted?" >&2
    echo "       Set DTFE_DATA_ROOT (or -d DATA_DIR) to where the snapshots are." >&2
    return 1
}

# expected_output_bytes <grid> <bytes per value> <fields...>: an APPROXIMATE size of the grids a run writes
# (components per field from a TNG100 512^3 run's files), plus the stream count, the hidden-streams flag and 10%
# for the point files and the caustic grids.
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

# scalar_dataset_resolve: SCALAR_DATASET=NAME grids the particles' own value NAME (a per-particle dataset of the input,
# e.g. Potential) as the 'scalar' field ('.a_scalar'): scalar_a joins FIELDS, and density_a when the density is not;
# scalar_args carries '--scalar-dataset NAME'. Refused, exit 2: a scalar field in FIELDS without SCALAR_DATASET, an
# option in FIELDS, a NAME with '/', a space or a leading '-'. Run it into a prefix of its own (OUTPUT_PREFIX).
scalar_dataset_resolve() {
    local f has_den=0 has_scalar=0
    scalar_args=()
    for f in ${FIELDS}; do
        case "${f}" in
            -*) echo "Error: FIELDS names fields, not options (got '${f}');" \
                     "a particle value to grid is SCALAR_DATASET=NAME." >&2
                exit 2 ;;
            density|density_a) has_den=1 ;;
            scalar|scalar_a|scalarGradient|scalarGradient_a) has_scalar=1 ;;
        esac
    done
    if [ -z "${SCALAR_DATASET}" ]; then
        if [ "${has_scalar}" = "1" ]; then
            echo "Error: FIELDS names a scalar field but SCALAR_DATASET is not set: the program would read no" >&2
            echo "       particle value and write zeros. Name the dataset, e.g. SCALAR_DATASET=Potential." >&2
            exit 2
        fi
        return 0
    fi
    case "${SCALAR_DATASET}" in
        */*|*" "*|-*)
            echo "Error: SCALAR_DATASET must be the NAME of a per-particle dataset, e.g. Potential" \
                 "(got '${SCALAR_DATASET}')." >&2
            exit 2 ;;
    esac
    case " ${FIELDS} " in *" scalar "*|*" scalar_a "*) ;; *) FIELDS="${FIELDS} scalar_a" ;; esac
    [ "${has_den}" = "1" ] || FIELDS="density_a ${FIELDS}"
    scalar_args=(--scalar-dataset "${SCALAR_DATASET}")
}

# output_space_problem <output dir> <bytes needed>: 1 (with the reason) when the folder is not writable or has less
# free space than the grids need. The grids are written beside the old ones (<file>.part.<pid>), so the old set does
# not count as free. SPACE_CHECK=0 skips the space half.
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

# The single-instance lock: '<prefix>.runlog.lock' is a DIRECTORY made with mkdir (atomic; macOS has no flock(1))
# holding 'pid', the driver's pid. A lock whose pid is gone or belongs to a process that is not a driver (a reused
# pid) is stale and removed, unless a live process still works on that snapshot (a SIGKILLed driver leaves its binary).

# run_lock_take <lock dir>: 0 = taken; 1 = held by a live driver or process (the refusal printed).
# run_lock_live_process <output root>: prints 'pid args' of the first live process naming the root on its command
# line, as the launcher's _live_roots does (python/gui/results.py); status 1 when there is none or ps cannot say
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

# The binary never removes what an EARLIER run with the same prefix wrote and this one does not, so after a run that
# EXITED 0 the drivers move every '<prefix>.*' file it did not write (same inode as before, not newer than the run's
# start) into '<prefix>.superseded/<stamp>/' (never deleted), with the old run's log. A failed, stopped or dry run and
# an auto-tune report move nothing; not moved: '<prefix>.runlog*', '<prefix>.gui.json', '<file>.part.<pid>'.

# supersede_snapshot <output root>: what a run may later supersede, recorded BEFORE it starts: one line
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
# exited 0, moves each recorded file the run did not write into '<root>.superseded/<YYYY-MM-DD_HHMMSS>/' ('-<pid>'
# added when that folder exists), with the saved old log as '<prefix>.runlog' there. Never deletes, never
# overwrites ('mv -n', checked); a failed move is said and left. Returns 0.
supersede_extras() {
    local root="$1" list="$2" marker="$3" oldlog="$4" dir pfx line ino n f now dest moved=0
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
            continue
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

# ps_checkpoint_dir <output root> [<scratch dir>]: where a partitioned run keeps its checkpoint, for PS-DTFE and
# standard DTFE alike (src/partition_checkpoint.h directoryFor -- keep the two in step): '<output>.runlog.checkpoint'
# beside the outputs, or '<scratch>/<basename>.<FNV-1a 64 of the absolute output path, 16 hex>.runlog.checkpoint'.
# The absolute path is the one the binary sees (a relative output is taken from 'pwd -P').
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
# They read and set the drivers' own variables where a comment says so: n_str, DATA_DIR, OUTPUT_PREFIX, GRID_SIZE,
# FIELDS, SCRATCH_DIR, AUTO_TUNE_REPORT, REPO_ROOT and the run-log state of runlog_contract_start.

# Strictness: the batch drivers run under 'set -uo pipefail' but NOT -e, on purpose: a batch goes on after a failed
# snapshot and exits non-zero at the end. Everything here must hold under 'set -u' (a possibly unset value is
# "${VAR:-}", a possibly empty array ${ARR[@]+"${ARR[@]}"}) and run on bash 3.2: no associative arrays, no ';&',
# no ${var,,} / ${var^^}, no mapfile / readarray.

# driver_options <own option letters> <handler> "$@": the option loop of run_dtfe.sh, run_ps_dtfe.sh and
# download_snapshots.sh. The shared options -d, -s and -h are handled here; an unknown option, or one missing its
# value, prints the usage and exits 1. Each own letter goes to '<handler> <letter>' with its value in OPTARG. Then the
# data root (unless -d) and the simulation are required. The caller drops the options: shift $((OPTIND - 1)).
# The long forms (driver_long_options) are read as their letters.
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

# driver_long_options <getopts spec> "$@": _drv_args = the arguments with each long option spelled as its letter,
# token for token (so the caller's 'shift $((OPTIND - 1))' stays right):
#   --sim S -> -s S, --sim-dir D -> -d D, --grid N -> -g N, --gpu -> -m, --exact -> -e, --sampled -> -S,
#   --subsamples K -> -n K, --run-mode M -> -R M, --help -> -h; download_snapshots.sh's --groupcats -> -c, --trees -> -t,
# Only up to the first non-option argument; a letter's value is never translated. An unknown long option exits 1.
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
                    --exact) letter=e ;; --sampled) letter=S ;; --subsamples) letter=n ;; --help) letter=h ;;
                    --run-mode) letter=R ;;
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

# average_option e|S: a driver's -e (--exact) or --sampled (-S): DTFE_EXACT_AVERAGE becomes 1 or 0, and _avg_flag
# names the option for average_resolve ('both' once the two were given: refused there).
average_option() {
    local f=--sampled v=0
    if [ "$1" = "e" ]; then f=-e; v=1; fi
    if [ -n "${_avg_flag:-}" ] && [ "${_avg_flag}" != "${f}" ]; then
        _avg_flag=both
    else
        _avg_flag="${f}"; DTFE_EXACT_AVERAGE="${v}"
    fi
}

# average_resolve [what]: DTFE_EXACT_AVERAGE as 1 or 0 by the binaries' own rule (exact is the default). Order: -e or
# --sampled (not both), then DTFE_EXACT_AVERAGE when the caller set it (_caller_exact; empty and 'auto' are unset),
# then a SUBSAMPLES count (environment or -n: the sampled one), then the default. average_src names which decided.
# With <what> an exact one beside a count earns one note that the count is ignored (run_ps_pipeline.sh passes none).
average_resolve() {
    case "${_caller_exact:-}" in [Aa][Uu][Tt][Oo]) _caller_exact="" ;; esac   # no pin (src/env_knobs.h tristate)
    if [ "${_avg_flag:-}" = "both" ]; then
        echo "Error: '-e' ('--exact') and '--sampled' exclude each other: the '_a' cell averages are integrated exactly or sampled." >&2
        usage; exit 1
    elif [ -n "${_avg_flag:-}" ]; then
        average_src="${_avg_flag}"
    elif [ -n "${_caller_exact:-}" ]; then
        knob_bool DTFE_EXACT_AVERAGE
        average_src="DTFE_EXACT_AVERAGE=${DTFE_EXACT_AVERAGE}"
    elif [ -n "${SUBSAMPLES:-}" ]; then
        DTFE_EXACT_AVERAGE=0; average_src="SUBSAMPLES=${SUBSAMPLES}"
    else
        DTFE_EXACT_AVERAGE=1; average_src="the default"
    fi
    if [ -n "${1:-}" ] && [ "${DTFE_EXACT_AVERAGE}" = "1" ] && [ -n "${SUBSAMPLES:-}" ]; then
        echo "Note: SUBSAMPLES=${SUBSAMPLES} is ignored: ${average_src} asks for the exact $1, which the driver runs without a count (--sampled or DTFE_EXACT_AVERAGE=0 runs the sampled one with it)"
    fi
}

# run_mode_option <mode>: a driver's --run-mode MODE (-R), passed on to the binary as given: run_mode_args=(--run-mode MODE),
# the last one given. 'auto' chooses nothing (empty): the binary then reads DTFE_RUN_MODE, else takes its default.
run_mode_option() {
    if [ "$1" = "auto" ]; then run_mode_args=(); else run_mode_args=(--run-mode "$1"); fi
}

# device_given, run_mode_given: a driver's banner words for the device and the run mode, as the driver was GIVEN them.
# device_given: 'the GPU (-m)' after -m (_gpu_flag), else 'DTFE_GPU=<value>' when set, else "the program's pick".
# run_mode_given: '--run-mode MODE' when the option named a mode, else 'DTFE_RUN_MODE=<value>' when set, else
# "the program's default".
device_given() {
    if [ -n "${_gpu_flag:-}" ]; then
        printf '%s\n' "the GPU (-m)"
    elif [ -n "${DTFE_GPU:-}" ]; then
        printf '%s\n' "DTFE_GPU=${DTFE_GPU}"
    else
        printf '%s\n' "the program's pick"
    fi
}
run_mode_given() {
    if [ -n "${run_mode_args[*]+x}" ]; then
        printf '%s\n' "${run_mode_args[*]}"
    elif [ -n "${DTFE_RUN_MODE:-}" ]; then
        printf '%s\n' "DTFE_RUN_MODE=${DTFE_RUN_MODE}"
    else
        printf '%s\n' "the program's default"
    fi
}

# driver_help_version <script> [its first argument]: '-h' / '--help' prints the script's header (line 2 up to the
# first blank line, '# ' stripped) and the caller's usage line; '--version' prints the revision. Both exit 0.
driver_help_version() {
    case "${2:-}" in
        -h|--help) sed -n '2,/^$/p' "$1" | sed -e 's/^# \{0,1\}//'; usage; exit 0 ;;
        --version) git -C "${REPO_ROOT}" describe --always --dirty 2>/dev/null || echo unknown; exit 0 ;;
    esac
}

# snapshot_args [snapshot ...]: positional arguments, when there are any, replace SNAPSHOTS; every entry is read as a
# decimal number (bash's printf reads a leading zero as OCTAL, so '010' would become 008). A non-number exits 2.
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

# driver_binary <variable> <binary> <precision>: sets <variable> to './<binary>' ('single') or './<binary>-double'
# ('double'); exits 1 for another precision or a binary that is not executable. Nothing falls back silently.
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

# time_wrap_resolve: TIME_WRAP, the optional peak-memory wrapper put in front of the binary (macOS spells the flag
# -l, GNU -v, a bare container has no /usr/bin/time); without one it is empty and the run identical.
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

# The run-log contract, the same in run_dtfe.sh, run_ps_dtfe.sh and the launcher's own runs: while the binary runs,
# its log is '<prefix>.runlog.partial'. Exit 0 renames it onto '<prefix>.runlog' (the log of the grids now on disk)
# and removes a stale '<prefix>.runlog.failed'; a failure or a stop renames it to '<prefix>.runlog.failed' and
# leaves '<prefix>.runlog' alone, so a failed rerun never replaces the log of the grids still on disk.

# runlog_contract_start: the driver's run-log state, empty, and its traps. A stop (SIGTERM, SIGINT) runs on_stop once
# the binary's pipeline has ended; any other way out still releases the lock and the temp files.
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
    # Strip ANSI colour codes so the kept log stays grep-able. Written through a temp file, not 'sed -i' (BSD and GNU
    # sed differ on its suffix argument).
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
# on_stop <signal name>: the attempt's log becomes '.runlog.failed', the lock and temp files are released, and the
# signal is re-raised, so a caller (run_ps_pipeline.sh) sees a stop.
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

# snapshot_start <snapshot number>: n_str ('099'), input_dir, input_file and output_root (<input_dir>/<OUTPUT_PREFIX>)
# of the snapshot, and its 'Processing' line. Returns 1, after saying so, when the input file is missing.
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
# its checkpoint (ps_checkpoint_dir) and the knob (DTFE_RESUME) is not 0. Empty for a report. The FILE 'checkpoint'
# is tested, not the folder: the binary refuses '--resume' on an empty folder, and every rerun would fail the same way.
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

# dry_run_print <input file> <output root> <run log> <command...>: what DRY_RUN=1 prints instead of running.
dry_run_print() {
    local input="$1" root="$2" log="$3"; shift 3
    echo "  DRY RUN -- would run:"
    printf '    %q' "$@"; echo
    echo "  input:   ${input} (exists)"
    echo "  outputs: ${root}.* and the run log ${log}"
}

# auto_tune_report_line <temp log> <exit status>: prints the AUTO-TUNE-REPORT line from the report run's temp log and
# removes it; returns 1, with an error line, when the run failed or wrote no report.
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

# snapshot_preflight <input dir> <run log> <precision>: room for the grids of GRID_SIZE and FIELDS (output_space_problem;
# 8 bytes a value for 'double', else 4), then the single-instance lock '<run log>.lock' (held in live_lock). Returns 1
# with the reason when either refuses.
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

# sup_record <output root> <run log>: before a run, what an earlier run with this prefix left (supersede_snapshot),
# the run's start marker and a copy of the old run log; run_finish moves aside what the run did not rewrite.
sup_record() {
    sup_clean
    sup_marker="$(mktemp "${TMPDIR:-/tmp}/dtfe-run-start.XXXXXX" 2>/dev/null)" || sup_marker=""
    sup_list="$(supersede_snapshot "$1")"
    if [ -n "${sup_list}" ] && [ -f "$2" ]; then
        sup_oldlog="$(mktemp "${TMPDIR:-/tmp}/dtfe-old-runlog.XXXXXX" 2>/dev/null)" && cp -p "$2" "${sup_oldlog}" \
            || sup_oldlog=""
    fi
}

# run_finish <exit status> <output root>: the de-ANSI'd log becomes '<prefix>.runlog' on success and
# '<prefix>.runlog.failed' otherwise; on exit 0 the earlier run's unwritten files are moved aside (supersede_extras);
# the temp files and the lock are released.
run_finish() {
    finish_runlog "${live_log}" "$1"
    live_log=""
    [ "$1" -eq 0 ] && supersede_extras "$2" "${sup_list}" "${sup_marker}" "${sup_oldlog}"
    sup_clean
    run_lock_release "${live_lock}"; live_lock=""
}

# peak_memory_text <run log>: the run's peak memory from the TIME_WRAP lines, as 'F GB footprint, R GB resident' or
# 'R GB resident'; nothing without such a line. macOS also reports the peak FOOTPRINT, which includes compressed or
# swapped memory that RSS alone hides. The dialects:
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

# snapshot_outcome <exit status> <its log>: the wall time and the snapshot's result line. Exit 3 means the grids went to
# the binary's rescue folder: the log's OUTPUT RESCUE line names it and the mv that puts them in place. Returns 0 for
# a processed snapshot, 1 for a failed one.
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
