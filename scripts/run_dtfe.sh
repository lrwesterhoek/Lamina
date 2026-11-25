#!/usr/bin/env bash
#
# Batch-run standard DTFE over TNG snapshots. Needs DTFE_DATA_ROOT (or -d, one simulation's folder) and the
# simulation (-s or DTFE_SIM); the shared defaults live in config.sh.
#
# Options: -d DIR (--sim-dir), -s SIM (--sim), -g N (--grid), -n K (--subsamples: k^3 Monte-Carlo points per cell),
#   -m (--gpu: starts the program with DTFE_GPU=1), -e (--exact, the default) or --sampled, --run-mode MODE (gentle,
#   balanced or performance; auto passes nothing), -h (--help), --version; then snapshot numbers (default 99).
# Defaults: 512^3 cells, padding 15, the exact cell average, the density, the velocity and its divergence and shear,
#   LAMBDA_TH 0.3. Every knob (environment) is in the README's knob table; DRY_RUN=1 prints the commands and runs
#   nothing, AUTO_TUNE_REPORT=1 prints the auto-tuner's report per snapshot and computes nothing.
# Outputs: <snapdir>/<OUTPUT_PREFIX>.* (default 'output'; a 'ps_*' prefix is refused) beside <prefix>.runlog. A rerun
#   moves the earlier run's files it did not write to <prefix>.superseded/; a killed partitioned run resumes from its
#   checkpoint (DTFE_RESUME=0 starts over); <prefix>.runlog.lock refuses a second driver on the snapshot.

set -uo pipefail   # no -e: a batch goes on after a failed snapshot (config.sh, 'Strictness')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
# Padding of each partition and of each thread's sub-box: standard DTFE's default is 15, not config.sh's 50
# (the PS-DTFE scripts' value). 15 is complete (no '*.badGridPoints*' file) on the z = 0 snapshots of
# TNG50/100/300-3-Dark. A PADDING from the caller wins: config.sh keeps it in CALLER_PADDING.
source "${SCRIPT_DIR}/config.sh"
PADDING="${CALLER_PADDING:-15}"

# ---- Script-specific configuration ----------------------------------------
PARTITION="${PARTITION:-}"       # EMPTY (default) = the binary AUTO-TUNES the split; set to override, e.g. PARTITION="2 2 2"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap on concurrent triangulations to bound peak RAM (0 = all cores); EMPTY = auto-tuned
# DTFE_GPU and DTFE_RUN_MODE reach the binary in its environment as the caller set them (-m sets DTFE_GPU=1,
# --run-mode wins over DTFE_RUN_MODE); same knobs as run_ps_dtfe.sh's.
_caller_exact="${DTFE_EXACT_AVERAGE:-}"   # (the caller's own value: unset, a SUBSAMPLES count selects the sampled mean)
DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE:-1}"  # 1 (default) = exact cell averages (--exact); 0 = the Monte-Carlo
                                 # sample mean (--sampled). Unset, a SUBSAMPLES count makes it 0 (config.sh
                                 # average_resolve, the binary's rule); the driver passes --exact or --sampled every run
SUBSAMPLES="${SUBSAMPLES:-}"     # k = --subsamples k: k points per axis, k^3 per cell, for the Monte-Carlo mean;
                                 # EMPTY (default) = the binary's own count. Set while DTFE_EXACT_AVERAGE is unset, it
                                 # selects the Monte-Carlo mean; beside an exact one it is ignored; -n k sets it
THREADS="${THREADS:-}"           # cap OpenMP threads globally (OMP_NUM_THREADS); empty = all cores (as run_ps_dtfe.sh)
FACE_GRADIENT="${FACE_GRADIENT:-}"   # the velocity gradient's route. EMPTY (default) = the binary's own: the exact face
                                 # integrals of the interpolant (Gauss), in double; 1 = --face-gradient (the same,
                                 # explicit); 0 = --per-tet-gradient, the deposit's per-tetrahedron gradient
                                 # (float32 on the GPU). 3D only. Same knob as run_ps_dtfe.sh's.
DRY_RUN="${DRY_RUN:-0}"          # 1 = print the commands and paths, run nothing
AUTO_TUNE_REPORT="${AUTO_TUNE_REPORT:-0}"  # 1 = only ASK the binary's auto-tuner what this run would need
                                 # (--auto-tune-report): one AUTO-TUNE-REPORT line per snapshot after reading the
                                 # input; nothing is computed or written. The launcher's memory check uses it.
DTFE_PRECISION="${DTFE_PRECISION:-single}"  # double = the double-precision binary ./DTFE-double ('make DTFE DOUBLE=1')
SCRATCH_DIR="${SCRATCH_DIR:-}"   # out-of-core mode (--scratch-dir): the full-resolution grids as mmap'ed files in this
                                 # LOCAL directory instead of RAM (bit-identical results; iCloud paths are refused)
SAMPLE_POINTS="${SAMPLE_POINTS:-}"  # a --sample-points file: the standard DTFE interpolant at those points too ('.pts_*').
                                 # With a partition split each partition answers the points inside the tetrahedra it
                                 # owns (one triangulation per partition, no thread sub-boxes); a killed run resumes
                                 # (DTFE_RESUME) with its finished partitions' points kept
PTS_VEL_GRAD="${PTS_VEL_GRAD:-0}"   # 1 = with SAMPLE_POINTS, also the velocity gradient at each point
TESSELLATION_CACHE="${TESSELLATION_CACHE:-}"  # a LOCAL folder: reuse the Delaunay tessellation of an identical earlier run
                                 # (--tessellation-cache; created if missing), the same knob as run_ps_dtfe.sh
LAMBDA_TH="${LAMBDA_TH:-0.3}"    # eigenvalue threshold of the T-web / V-web classes (--lambda-th); passed on every run
DTFE_RESUME="${DTFE_RESUME:-1}"  # 1 (default) = resume a killed partitioned run when the snapshot's checkpoint exists
                                 # (config.sh ps_checkpoint_dir): --resume is passed, only missing partitions are
                                 # computed. The binary refuses when the settings changed; DTFE_RESUME=0 starts over

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
OUTPUT_PREFIX="${OUTPUT_PREFIX:-output}"   # -> <snapdir>/<prefix>.* (default 'output'); another prefix keeps a variant
                                 # beside the default set, e.g. OUTPUT_PREFIX=output_mc with DTFE_EXACT_AVERAGE=0.
                                 # 'ps_*' is refused below (PS-DTFE's naming)

# the grids to write; env-overridable as in run_ps_dtfe.sh, e.g. FIELDS="density_a velocity_a" for a lighter run
# (the launcher passes its ticks here; no dispersion_a: that is a phase-space field). Divergence and shear come from
# the per-tet velocity gradient (derived after the loop, not written), so the gradient, vorticity, T-web and V-web
# grids are only written when FIELDS names them
FIELDS="${FIELDS:-density_a velocity_a divergence_a shear_a}"
case " ${FIELDS} " in               # a FIELDS exported for run_ps_dtfe.sh may carry the phase-space dispersion
    *" dispersion"*) echo "Note: the velocity dispersion is a PS-DTFE field; standard DTFE leaves it out"
                     FIELDS=$(printf '%s\n' ${FIELDS} | grep -v '^dispersion' | tr '\n' ' ') ;;
esac
if [ -z "${FIELDS// }" ]; then
    echo "Error: no field standard DTFE computes in FIELDS (the velocity dispersion is a PS-DTFE field)" >&2
    exit 2
fi
SCALAR_DATASET="${SCALAR_DATASET:-}"   # a per-particle dataset of the input to grid as the 'scalar' field, e.g. Potential:
                                 # --scalar-dataset NAME, scalar_a (and density_a) added to FIELDS (config.sh
                                 # scalar_dataset_resolve, as run_ps_dtfe.sh). The cell value is the cell average of the
                                 # linear interpolant of the particles' values, in the dataset's own units.

knob_bool DTFE_EXACT_AVERAGE FACE_GRADIENT DRY_RUN AUTO_TUNE_REPORT PTS_VEL_GRAD DTFE_RESUME   # 1 or 0 (config.sh)
usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n SUBSAMPLES] [-m] [-e | --sampled] [--run-mode MODE] [snapshot ...]"; }

driver_help_version "${BASH_SOURCE[0]}" "${1:-}"   # -h / --help print the header and run nothing; --version the revision
scalar_dataset_resolve           # (SCALAR_DATASET, above; after --help, which a stray export must not block)
# OUTPUT_PREFIX (above): one word, and not PS-DTFE's 'ps_*' naming (after --help, which a stray export must not block)
case "${OUTPUT_PREFIX}" in
    ""|*/*|*" "*|.*)
        echo "Error: OUTPUT_PREFIX must be one word without '/' or spaces (got '${OUTPUT_PREFIX}')." >&2; exit 2 ;;
    ps_*)
        echo "Error: OUTPUT_PREFIX='${OUTPUT_PREFIX}' is PS-DTFE's naming (an OUTPUT_PREFIX exported for run_ps_dtfe.sh?):" >&2
        echo "       standard DTFE would write its grids over the phase-space ones. Use another prefix (default 'output')." >&2
        exit 2 ;;
esac

# ---- Parse arguments ------------------------------------------------------
own_option() {   # <letter>: this driver's own options; -d, -s, -h and the data checks are config.sh driver_options'
    case "$1" in
        g) GRID_SIZE="${OPTARG}" ;;
        n) SUBSAMPLES="${OPTARG}" ;;
        m) _gpu_flag=1; export DTFE_GPU=1 ;;   # the binary reads it (config.sh device_given names it)
        e|S) average_option "$1" ;;      # -e / --exact, --sampled (config.sh: S only carries the long form)
        R) run_mode_option "${OPTARG}" ;;   # --run-mode MODE (config.sh: R only carries the long form)
    esac
}
run_mode_args=()                 # --run-mode MODE as given (config.sh run_mode_option); empty: the binary decides
driver_options "g:n:meSR:" own_option "$@"   # then the data root (unless -d) and the simulation are required
shift $((OPTIND - 1))
average_resolve "cell average"   # DTFE_EXACT_AVERAGE 1 or 0: the options, the knob, a SUBSAMPLES count, the default

snapshot_args "$@"              # positional arguments override the default snapshot list (config.sh)

[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"
data_root_problem "${DATA_DIR}" || exit 2     # preflight: the data disk unmounted, said up front

cd "$REPO_ROOT" || exit 1

driver_binary DTFE_BIN DTFE "${DTFE_PRECISION}"      # ./DTFE or ./DTFE-double (config.sh)
time_wrap_resolve                                     # TIME_WRAP: the optional peak-memory wrapper
[ -n "${THREADS}" ] && export OMP_NUM_THREADS="${THREADS}"
keep_awake "${DRY_RUN}"
# the run-log contract (config.sh): 'output.runlog.partial' while the binary runs, then 'output.runlog' on exit 0
# or 'output.runlog.failed' on a failure or a stop; a stop also releases the lock and re-raises the signal
runlog_contract_start

echo "Starting DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}   Fields: ${FIELDS}   Output prefix: ${OUTPUT_PREFIX}"
[ -n "${SCALAR_DATASET}" ] && echo "Particle value gridded as the scalar: ${SCALAR_DATASET} (--scalar-dataset)"
if [ "${DTFE_EXACT_AVERAGE}" = "1" ]; then
    echo "Cell averages: exact (--exact; ${average_src})"
else
    echo "Cell averages: sampled, Monte-Carlo sample means (--sampled${SUBSAMPLES:+ --subsamples ${SUBSAMPLES}}; ${average_src})"
fi
echo "'_a' interpolation: $(device_given)"      # what the driver was given (config.sh); the run log says what ran
echo "Run mode: $(run_mode_given)${OMP_NUM_THREADS:+ (OMP_NUM_THREADS=${OMP_NUM_THREADS} keeps its thread count)}"
[ -n "${SCRATCH_DIR}" ] && echo "Scratch directory: ${SCRATCH_DIR} (grids out of core)"
echo ""

n_ok=0; n_skipped=0; n_failed=0; n_planned=0
for i in "${SNAPSHOTS[@]}"; do
    # n_str, input_dir, input_file and output_root (config.sh): files land as <snapdir>/output.a_den etc. (the
    # 'dtfe' convention in dtfelib; OUTPUT_PREFIX)
    if ! snapshot_start "$i"; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # the cell average, named both ways (the run log then says which ran); the count only for the sampled one
    exact_args=(--exact)
    if [ "${DTFE_EXACT_AVERAGE}" != "1" ]; then
        exact_args=(--sampled)
        [ -n "${SUBSAMPLES}" ] && exact_args+=(--subsamples "${SUBSAMPLES}")   # (SUBSAMPLES, see the top: only when set)
    fi
    [ "${FACE_GRADIENT}" = "1" ] && exact_args+=(--face-gradient)     # (FACE_GRADIENT, see the top)
    [ "${FACE_GRADIENT}" = "0" ] && exact_args+=(--per-tet-gradient)  # the deposit's own gradient (the older default)

    # pass --partition/--max-concurrent only when set; otherwise the binary auto-tunes them
    part_args=()
    [ -n "${PARTITION}" ] && part_args+=(--partition ${PARTITION})
    [ -n "${MAX_CONCURRENT}" ] && part_args+=(--max-concurrent "${MAX_CONCURRENT}")
    # the run mode only when --run-mode named one (DTFE_RUN_MODE, see the top, reaches the binary otherwise)
    part_args+=(${run_mode_args[@]+"${run_mode_args[@]}"})
    scratch_args=()
    [ -n "${SCRATCH_DIR}" ] && scratch_args=(--scratch-dir "${SCRATCH_DIR}")
    sp_args=()
    [ -n "${SAMPLE_POINTS}" ] && sp_args=(--sample-points "${SAMPLE_POINTS}")
    [ -n "${SAMPLE_POINTS}" ] && [ "${PTS_VEL_GRAD}" = "1" ] && sp_args+=(--pts-vel-grad)
    tess_args=()
    if [ -n "${TESSELLATION_CACHE}" ]; then
        [ "${DRY_RUN:-0}" = "1" ] || mkdir -p "${TESSELLATION_CACHE}"    # (a dry run creates nothing)
        tess_args=(--tessellation-cache "${TESSELLATION_CACHE}")
    fi
    # a killed partitioned run left its checkpoint: resume it (config.sh), as run_ps_dtfe.sh does
    resume_args_for "${output_root}" DTFE_RESUME "${DTFE_RESUME}"

    # the tee pipe hides the terminal from the binary's isatty() check: force colours through it when
    # this script itself runs on a terminal (the runlog is de-ANSI'd below), as run_ps_dtfe.sh does
    [ -t 1 ] && export CLICOLOR_FORCE=1
    echo "  Running DTFE on ${input_file}..."
    # the run log beside the outputs (<snapdir>/<prefix>.runlog), as run_ps_dtfe.sh keeps one: the
    # GUI's Runs browser reads the settings, build stamp, wall time and peak memory back from it
    run_log="${output_root}.runlog"
    # a report writes nothing next to the data: not even over the snapshot's real run log
    [ "${AUTO_TUNE_REPORT}" = "1" ] && [ "${DRY_RUN}" != "1" ] && run_log="$(mktemp -t dtfe-auto-tune-report.XXXXXX)"
    report_args=()
    [ "${AUTO_TUNE_REPORT}" = "1" ] && report_args=(--auto-tune-report)
    SECONDS=0
    cmd=("${DTFE_BIN}" "${input_file}" "${output_root}"
        --grid ${GRID_SIZE}
        --padding ${PADDING}
        --periodic
        ${part_args[@]+"${part_args[@]}"}
        --field ${FIELDS}
        --lambda-th ${LAMBDA_TH}
        ${scalar_args[@]+"${scalar_args[@]}"}
        ${scratch_args[@]+"${scratch_args[@]}"}
        ${tess_args[@]+"${tess_args[@]}"}
        ${sp_args[@]+"${sp_args[@]}"}
        ${exact_args[@]+"${exact_args[@]}"}
        ${report_args[@]+"${report_args[@]}"}
        ${resume_args[@]+"${resume_args[@]}"})
    if [ "${DRY_RUN}" = "1" ]; then                 # DRY_RUN=1: the expanded command and the paths, nothing run
        dry_run_print "${input_file}" "${output_root}" "${run_log}" "${cmd[@]}"
        n_planned=$((n_planned + 1))
        echo ""
        continue
    fi
    # A report (AUTO_TUNE_REPORT=1) computes and writes nothing: no space check, no lock (the launcher's memory check
    # runs one beside a live run), no run-log contract -- the binary's one line, from a temp log (as run_ps_dtfe.sh)
    if [ "${AUTO_TUNE_REPORT}" = "1" ]; then
        "${cmd[@]}" 2>&1 | tee "${run_log}"
        rc=${PIPESTATUS[0]}
        if auto_tune_report_line "${run_log}" "${rc}"; then n_ok=$((n_ok + 1)); else n_failed=$((n_failed + 1)); fi
        echo ""
        continue
    fi
    # room for the grids, then the snapshot's single-instance lock (config.sh; as run_ps_dtfe.sh)
    if ! snapshot_preflight "${input_dir}" "${run_log}" "${DTFE_PRECISION}"; then
        n_failed=$((n_failed + 1)); echo ""; continue
    fi
    # what an earlier run with this prefix left, recorded before this one starts: what it does not rewrite is
    # moved aside once it has exited 0 (config.sh sup_record, run_finish)
    sup_record "${output_root}" "${run_log}"
    # the binary's output goes to '<prefix>.runlog.partial' until it exits (the run-log contract above)
    live_log="${run_log}.partial"; live_rc=""
    ${TIME_WRAP[@]+"${TIME_WRAP[@]}"} "${cmd[@]}" 2>&1 | tee "${live_log}"
    rc=${PIPESTATUS[0]}                 # the binary's status, taken BEFORE anything else runs a pipe
    live_rc="${rc}"

    # the log becomes 'output.runlog' or 'output.runlog.failed', the earlier run's leftovers move aside on exit 0
    # (config.sh run_finish); the peak memory below is read from whichever log it became
    run_finish "${rc}" "${output_root}"
    [ "${rc}" -eq 0 ] || run_log="${run_log}.failed"
    peak_txt="$(peak_memory_text "${run_log}")"
    [ -n "${peak_txt}" ] && echo "  Peak memory: ${peak_txt}"
    if snapshot_outcome "${rc}" "${run_log}"; then n_ok=$((n_ok + 1)); else n_failed=$((n_failed + 1)); fi
    echo ""
done

# Honest exit status, as run_ps_dtfe.sh: a failed snapshot, or nothing processed, exits 1 (so the launcher does
# not read a failed standard-DTFE job as done). A partial skip stays 0.
[ "${DRY_RUN}" = "1" ] && dry_run_exit "${n_planned}" "${n_skipped}" "${DATA_DIR}"
echo "DTFE processing complete: ${n_ok} processed, ${n_skipped} skipped, ${n_failed} failed."
if [ "${n_failed}" -gt 0 ]; then
    echo "ERROR: ${n_failed} snapshot(s) failed; see the messages above and their .runlog.failed files." >&2
    exit 1
fi
if [ "${n_ok}" -eq 0 ]; then
    echo "ERROR: no snapshot was processed (${n_skipped} skipped for missing inputs) -- check DATA_ROOT/SIMULATION (looked under '${DATA_DIR}')." >&2
    exit 1
fi
