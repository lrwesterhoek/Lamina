#!/usr/bin/env bash
#
# Batch-run standard DTFE over a set of TNG snapshots (redshifts).
#
# Companion to download_snapshots.sh, which fetches the snapshot files first.
# DTFE_DATA_ROOT (the folder that holds the simulations; -d names a simulation's folder instead) and the simulation
# (-s or DTFE_SIM) are required; the
# shared defaults (SNAPSHOTS, GRID_SIZE) live in config.sh. PADDING, the
# padding in mean particle spacings, defaults to 15 here, not config.sh's 50 (the PS-DTFE scripts' value).
#
# Usage:
#   ./run_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n SUBSAMPLES] [-m] [-e | --sampled]
#                 [--run-mode MODE] [snapshot ...]
#     long forms: --sim-dir, --sim, --grid, --subsamples, --gpu, --exact (the same as the letters); --sampled,
#          --run-mode
#     -m   run the '_a' interpolation on the GPU (--gpu; same as DTFE_GPU=1). The default (DTFE_GPU unset, 'auto')
#          is the GPU when the binary was built with one (the Makefile's o/.gpu_mode_metal stamp;
#          o_d for DTFE_PRECISION=double), else the CPU -- and the CPU for the sampled average in performance, faster
#          there; DTFE_GPU=0 runs the CPU on a GPU build.
#     -e   every '_a' cell the EXACT average of the linear interpolant over the cell (--exact). The DEFAULT
#          (DTFE_EXACT_AVERAGE=1, the production standard: no sampling noise, the mass exact to ~3e-8); -e overrides
#          an exported DTFE_EXACT_AVERAGE=0 and keeps the exact average beside a SUBSAMPLES count (ignored, said).
#     --sampled  the Monte-Carlo sample mean instead (the same as DTFE_EXACT_AVERAGE=0). At 512^3 (TNG100-3-Dark
#          z=0, 2026-11-09) it took 315 s on the CPU and 433 s on the GPU, the exact average with the face
#          gradient 431 s and 439 s (both measured with every core, as in performance).
#     -n   SUBSAMPLES: k sampling points per axis (k^3 per cell) of the Monte-Carlo mean (--subsamples k, the same
#          unit as run_ps_dtfe.sh's); passed only when set (the binary's own default otherwise). While
#          DTFE_EXACT_AVERAGE is unset (and neither -e nor --sampled is given) a count selects the Monte-Carlo mean,
#          as the binary does; beside -e or DTFE_EXACT_AVERAGE=1 it is ignored.
#     --run-mode MODE  how hard the run may lean on the machine (DTFE_RUN_MODE the same; the option wins): gentle (half
#          the performance cores, short GPU buffers, on Linux nice 10), balanced (the performance cores less a quarter,
#          short GPU buffers) or performance (every core, long GPU buffers); auto = balanced, performance in a batch job.
#     --exact-average, the older name of --exact, is still read for one release (with a warning).
#     The drivers' letters are their own: -m is the GPU and -s the simulation here (the binaries' -m and -s, the
#          averaging method and a total sample count, are deprecated: --average-method, --subsamples).
#   Production defaults: GRID_SIZE 512 (config.sh), padding 15, the GPU when built, the exact cell
#   average, FIELDS = density, velocity, its divergence and shear (the analysis set, the same as run_ps_dtfe.sh's;
#   the gradient, vorticity, T-web and V-web stay available through FIELDS), LAMBDA_TH 0.3.
#   Knobs (environment; each is explained where the script sets it): PARTITION MAX_CONCURRENT DTFE_GPU DTFE_RUN_MODE
#   DTFE_EXACT_AVERAGE SUBSAMPLES THREADS FACE_GRADIENT SCRATCH_DIR SAMPLE_POINTS PTS_VEL_GRAD TESSELLATION_CACHE
#   LAMBDA_TH DTFE_RESUME OUTPUT_PREFIX FIELDS DTFE_PRECISION DRY_RUN AUTO_TUNE_REPORT, and config.sh's GRID_SIZE,
#   PADDING, SPACE_CHECK. A switch is off for 0, false, no or off and on for any other value (the binaries' rule).
#   Older names, read for one release with a warning (the new name wins when both are set): DTFE_METAL (DTFE_GPU),
#   DTFE_EXACT (DTFE_EXACT_AVERAGE), TESS_CACHE (TESSELLATION_CACHE), DTFE_PADDING, DTFE_GRID_SIZE, DTFE_SPACE_CHECK.
#   OUTPUT_PREFIX (default 'output'): the files are <snapdir>/<prefix>.*; another prefix keeps a variant beside
#   the default set (dtfelib reads 'output.' for standard DTFE). A 'ps_*' prefix is refused: that is PS-DTFE's
#   naming (an OUTPUT_PREFIX exported for run_ps_dtfe.sh would otherwise overwrite the phase-space grids).
#   A rerun into a prefix that holds an earlier run's files: once it exits 0, the files it did not write (an image
#   plane, caustics or a field the new run lacks) are MOVED into <prefix>.superseded/<date_time>/ with the old
#   run log (config.sh supersede_extras; never deleted), so the folder holds one run's set.
#   -h / --help prints this header and runs nothing; --version the repository revision.
#   DRY_RUN=1 ./run_dtfe.sh ...   prints every snapshot's expanded command and paths, runs nothing (exit 0
#          when at least one snapshot is planned).
#   AUTO_TUNE_REPORT=1 ./run_dtfe.sh ...   only asks the binary's auto-tuner what each snapshot would need (one
#          'AUTO-TUNE-REPORT snap=NNN partition=.. mc=.. predicted_gb=..' line each; mc = the threads of a partition,
#          the partitions run one after the other); nothing is computed or written (as run_ps_dtfe.sh).
#   Run logs, beside the outputs: output.runlog belongs to the grids on disk, output.runlog.failed to the
#   last failed or stopped attempt, output.runlog.partial to a run still going (or killed outright).
#   While a snapshot runs, <prefix>.runlog.lock (a folder holding the driver's pid) refuses a second driver on
#   it; a lock left by a killed driver is removed with a note. Before anything runs: the data disk must be
#   there (exit 2 otherwise) and the output folder writable with room for the grids (SPACE_CHECK=0 skips
#   the space half).
#   DTFE_RESUME (default 1): a partitioned run killed part-way resumes, as run_ps_dtfe.sh's does. When the
#   snapshot's checkpoint (<prefix>.runlog.checkpoint, or in SCRATCH_DIR) exists, the next run passes --resume and
#   computes only the partitions it lacks; DTFE_RESUME=0 starts over (the new run replaces the checkpoint). A run with
#   SAMPLE_POINTS too (the binary saves each finished partition's point answers beside the grids).

set -uo pipefail   # no -e: a batch goes on after a failed snapshot (config.sh, 'Strictness')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
# Padding (mean interparticle spacings) of each partition and of each thread's sub-box: standard DTFE's own default
# is 15, not config.sh's 50 (which stays the PS-DTFE scripts' value). 15 is complete -- no '*.badGridPoints*' file --
# on the z = 0 snapshots of TNG50/100/300-3-Dark, the hardest case (the emptiest voids give the largest Delaunay
# circumspheres at the partition edges), and 22-29% faster than 20 there; 10 is not complete at z = 0, and 50 costs
# 33x the vertices. A PADDING from the caller (or DTFE_PADDING, its older name) still wins: config.sh keeps what the
# caller set in CALLER_PADDING before it fills PADDING with its own default.
source "${SCRIPT_DIR}/config.sh"
PADDING="${CALLER_PADDING:-15}"

# ---- Script-specific configuration ----------------------------------------
PARTITION="${PARTITION:-}"       # EMPTY (default) = the binary AUTO-TUNES the split from the particle
                                 # count, grid, fields and available RAM/cores (see the AUTO-TUNE line
                                 # it prints). Set to override, e.g. PARTITION="2 2 2".
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap on concurrent triangulations to bound peak RAM (0 = all cores).
                                 # EMPTY (default) = auto-tuned together with the partition split.
knob DTFE_GPU DTFE_METAL          # (config.sh knob: the older name read for one release, with a warning)
DTFE_GPU="${DTFE_GPU:-auto}"     # 1 = the '_a' interpolation on the GPU (--gpu), 0 = the CPU; 'auto'
                                 # (default, the drivers' rule) = 1 when the binary is a GPU build (config.sh
                                 # gpu_build_default), resolved below once DTFE_PRECISION has chosen the binary --
                                 # but 0 for the sampled average in performance, faster on the CPU (config.sh
                                 # gpu_auto_resolve). The same knob as run_ps_dtfe.sh's
_caller_run_mode="${DTFE_RUN_MODE:-}"   # (the caller's own value: config.sh run_mode_resolve reads it after the options)
DTFE_RUN_MODE="${DTFE_RUN_MODE:-auto}"  # how hard the run may lean on the machine, passed on every run as '--run-mode
                                 # MODE': gentle, balanced or performance (any case); 'auto' (default) = balanced, or
                                 # performance inside a SLURM or PBS job (config.sh run_mode_resolve; --run-mode MODE
                                 # wins). It also steers DTFE_GPU's 'auto', above. The same knob as run_ps_dtfe.sh's
knob DTFE_EXACT_AVERAGE DTFE_EXACT
_caller_exact="${DTFE_EXACT_AVERAGE:-}"   # (the caller's own value: unset, a SUBSAMPLES count selects the sampled mean)
DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE:-1}"  # 1 (default, the production standard) = exact cell
                                 # averages (--exact: no sampling noise, mass exact to ~3e-8); 0 = the Monte-Carlo
                                 # sample mean (--sampled). Unset, a SUBSAMPLES count makes it 0 (config.sh
                                 # average_resolve, the binary's rule). The same knob, default and rule as
                                 # run_ps_dtfe.sh's; the driver passes --exact or --sampled on every run
SUBSAMPLES="${SUBSAMPLES:-}"     # k = --subsamples k: k points per axis, k^3 per cell, for the Monte-Carlo mean;
                                 # EMPTY (default) = the binary's own count. Set while DTFE_EXACT_AVERAGE is unset, it
                                 # selects the Monte-Carlo mean; beside an exact one it is ignored. The same unit
                                 # as run_ps_dtfe.sh's SUBSAMPLES; -n k sets it
THREADS="${THREADS:-}"           # cap OpenMP threads globally (OMP_NUM_THREADS); empty = all cores (as run_ps_dtfe.sh)
FACE_GRADIENT="${FACE_GRADIENT:-}"   # the velocity gradient's route (and the divergence and shear built on it). EMPTY
                                 # (default) = the binary's own: since 2026-11-09 the exact face integrals of the
                                 # interpolant (Gauss; docs/cleanup-plan.md D13), in double, the same for every
                                 # deposit and device; 1 = --face-gradient (the same, asked explicitly); 0 =
                                 # --per-tet-gradient, the deposit's own per-tetrahedron gradient (float32 on the
                                 # GPU; the route of the production grids made before 2026-11-09). 3D only.
                                 # run_ps_dtfe.sh's FACE_GRADIENT is the same knob with the same default (PS-DTFE:
                                 # the mean flow's gradient where the run qualifies)
DRY_RUN="${DRY_RUN:-0}"          # 1 = print the commands and paths, run nothing
AUTO_TUNE_REPORT="${AUTO_TUNE_REPORT:-0}"  # 1 = only ASK the binary's auto-tuner what this run would need
                                 # (--auto-tune-report): per snapshot one line
                                 #   AUTO-TUNE-REPORT snap=NNN partition=.. mc=.. predicted_gb=.. budget_gb=.. over_budget=0|1 ...
                                 # after reading the input; nothing is computed or written (the snapshot's .runlog,
                                 # lock and space check are left alone). The launcher's memory check uses it.
DTFE_PRECISION="${DTFE_PRECISION:-single}"  # double = the double-precision binary ./DTFE-double ('make DTFE DOUBLE=1'):
                                 # everything in double from the input read onward, float64 outputs, ~2x the memory
SCRATCH_DIR="${SCRATCH_DIR:-}"   # out-of-core mode (--scratch-dir): the full-resolution grids as mmap'ed files in this
                                 # LOCAL directory instead of RAM (bit-identical results; iCloud paths are refused) --
                                 # the same knob as run_ps_dtfe.sh; the launcher's scratch folder arrives here
SAMPLE_POINTS="${SAMPLE_POINTS:-}"  # a --sample-points file: the standard DTFE interpolant at those points too
                                 # ('.pts_*'). With PARTITION (or an auto-tuned split) as well: each
                                 # partition answers the points inside the tetrahedra it owns, equal to one triangulation
                                 # of the whole box to float rounding, so a full-box image plane fits at production
                                 # scale; such a partition is ONE triangulation (no thread sub-boxes), its grids equal a
                                 # run without points to float rounding (the dummy-point padding test is off). A killed
                                 # one resumes (DTFE_RESUME), its finished partitions' points kept
PTS_VEL_GRAD="${PTS_VEL_GRAD:-0}"   # 1 = with SAMPLE_POINTS, also the velocity gradient at each point
knob TESSELLATION_CACHE TESS_CACHE
TESSELLATION_CACHE="${TESSELLATION_CACHE:-}"  # a LOCAL folder: reuse the Delaunay tessellation of an identical earlier run
                                 # (--tessellation-cache; created if missing; a partitioned run caches one file per
                                 # thread sub-box of each partition, one per partition with SAMPLE_POINTS) -- the
                                 # same knob as run_ps_dtfe.sh
LAMBDA_TH="${LAMBDA_TH:-0.3}"    # eigenvalue threshold of the T-web / V-web classes (--lambda-th), the SAME default as
                                 # run_ps_dtfe.sh (the binary's own default is 0.0: a run without this flag classifies
                                 # differently from a scripted one)
DTFE_RESUME="${DTFE_RESUME:-1}"  # 1 (default) = resume a killed partitioned run, the same knob as run_ps_dtfe.sh's:
                                 # when the snapshot's checkpoint file exists (<prefix>.runlog.checkpoint/checkpoint
                                 # beside the outputs, or under SCRATCH_DIR; config.sh ps_checkpoint_dir, the same rule
                                 # for both binaries), --resume is passed and only the partitions it lacks are computed.
                                 # The binary refuses, naming the differing settings, when the settings changed; then
                                 # DTFE_RESUME=0 starts over (the new run replaces the checkpoint)

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
OUTPUT_PREFIX="${OUTPUT_PREFIX:-output}"   # -> <snapdir>/<prefix>.* (default 'output'). Another
                                 # prefix keeps a variant beside the default set, e.g. OUTPUT_PREFIX=output_mc with
                                 # DTFE_EXACT_AVERAGE=0; dtfelib reads 'output.' for standard DTFE. 'ps_*' is refused
                                 # below: PS-DTFE's naming, so an export meant for run_ps_dtfe.sh cannot overwrite
                                 # the phase-space grids with standard ones

# the grids to write; env-overridable as in run_ps_dtfe.sh, e.g. FIELDS="density_a velocity_a" for a lighter run
# (the launcher passes its ticks here; no dispersion_a: that is a phase-space field). The default is the analysis
# set, the same in both drivers: density, velocity, divergence, shear. Divergence and shear come from the per-tet
# velocity gradient (the binary deposits it and derives them after the loop, without writing it), so the gradient,
# vorticity, T-web and V-web grids are only written when FIELDS names them (+104 B per cell at their peak)
FIELDS="${FIELDS:-density_a velocity_a divergence_a shear_a}"
case " ${FIELDS} " in               # a FIELDS exported for run_ps_dtfe.sh may carry the phase-space dispersion
    *" dispersion"*) echo "Note: the velocity dispersion is a PS-DTFE field; standard DTFE leaves it out"
                     FIELDS=$(printf '%s\n' ${FIELDS} | grep -v '^dispersion' | tr '\n' ' ') ;;
esac
if [ -z "${FIELDS// }" ]; then
    echo "Error: no field standard DTFE computes in FIELDS (the velocity dispersion is a PS-DTFE field)" >&2
    exit 2
fi

knob_bool DTFE_EXACT_AVERAGE FACE_GRADIENT DRY_RUN AUTO_TUNE_REPORT PTS_VEL_GRAD DTFE_RESUME   # 1 or 0 (config.sh)
usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n SUBSAMPLES] [-m] [-e | --sampled] [--run-mode MODE] [snapshot ...]"; }

driver_help_version "${BASH_SOURCE[0]}" "${1:-}"   # -h / --help print the header and run nothing; --version the revision
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
        m) DTFE_GPU=1 ;;
        e|S) average_option "$1" ;;      # -e / --exact, --sampled (config.sh: S only carries the long form)
        R) run_mode_option "${OPTARG}" ;;   # --run-mode MODE (config.sh: R only carries the long form)
    esac
}
driver_options "g:n:meSR:" own_option "$@"   # then the data root (unless -d) and the simulation are required
shift $((OPTIND - 1))
average_resolve "cell average"   # DTFE_EXACT_AVERAGE 1 or 0: the options, the knob, a SUBSAMPLES count, the default
run_mode_resolve                 # DTFE_RUN_MODE: --run-mode, the knob, the default (balanced; performance in a batch job)

snapshot_args "$@"              # positional arguments override the default snapshot list (config.sh)

[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"
data_root_problem "${DATA_DIR}" || exit 2     # preflight: the data disk unmounted, said up front

cd "$REPO_ROOT" || exit 1

driver_binary DTFE_BIN DTFE "${DTFE_PRECISION}"      # ./DTFE or ./DTFE-double (config.sh)
gpu_auto_resolve DTFE_GPU "${DTFE_BIN}" dtfe "${DTFE_EXACT_AVERAGE}" "${DTFE_RUN_MODE}"   # DTFE_GPU=auto: the GPU when
                                                      # this binary was built with one, but the CPU for the sampled
                                                      # average in performance (config.sh)
time_wrap_resolve                                     # TIME_WRAP: the optional peak-memory wrapper
[ -n "${THREADS}" ] && export OMP_NUM_THREADS="${THREADS}"
keep_awake "${DRY_RUN}"
# the run-log contract (config.sh): 'output.runlog.partial' while the binary runs, then 'output.runlog' on exit 0
# or 'output.runlog.failed' on a failure or a stop; a stop also releases the lock and re-raises the signal
runlog_contract_start

echo "Starting DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}   Fields: ${FIELDS}   Output prefix: ${OUTPUT_PREFIX}"
if [ "${DTFE_EXACT_AVERAGE}" = "1" ]; then
    echo "Cell averages: exact (--exact; ${average_src})"
else
    echo "Cell averages: sampled, Monte-Carlo sample means (--sampled${SUBSAMPLES:+ --subsamples ${SUBSAMPLES}}; ${average_src})"
fi
echo "'_a' interpolation: $([ "${DTFE_GPU}" = "1" ] && echo "GPU (--gpu)" || echo "CPU")${gpu_auto}"
echo "Run mode: ${DTFE_RUN_MODE} (${run_mode_src}${OMP_NUM_THREADS:+; OMP_NUM_THREADS=${OMP_NUM_THREADS} keeps its thread count})"
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

    # GPU interpolation toggle (-m / DTFE_GPU=1); ignored with a warning on non-GPU builds.
    metal_args=()
    [ "${DTFE_GPU}" = "1" ] && metal_args=(--gpu)
    # the cell average, named both ways (the run log then says which ran); the count only for the sampled one
    exact_args=(--exact)
    if [ "${DTFE_EXACT_AVERAGE}" != "1" ]; then
        exact_args=(--sampled)
        [ -n "${SUBSAMPLES}" ] && exact_args+=(--subsamples "${SUBSAMPLES}")   # (SUBSAMPLES, see the top: only when set)
    fi
    [ "${FACE_GRADIENT}" = "1" ] && exact_args+=(--face-gradient)     # (FACE_GRADIENT, see the top)
    [ "${FACE_GRADIENT}" = "0" ] && exact_args+=(--per-tet-gradient)  # the deposit's own gradient (the route before 2026-11-09)

    # pass --partition/--max-concurrent only when set; otherwise the binary auto-tunes them
    part_args=()
    [ -n "${PARTITION}" ] && part_args+=(--partition ${PARTITION})
    [ -n "${MAX_CONCURRENT}" ] && part_args+=(--max-concurrent "${MAX_CONCURRENT}")
    # the run mode, on every run (DTFE_RUN_MODE, see the top: the run log then says which ran and why)
    part_args+=(--run-mode "${DTFE_RUN_MODE}")
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
        ${scratch_args[@]+"${scratch_args[@]}"}
        ${tess_args[@]+"${tess_args[@]}"}
        ${sp_args[@]+"${sp_args[@]}"}
        ${metal_args[@]+"${metal_args[@]}"}
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
