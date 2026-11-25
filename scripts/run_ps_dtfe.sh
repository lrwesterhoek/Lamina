#!/usr/bin/env bash
# Batch-run PS-DTFE over TNG snapshots: each particle needs its position and its initial (Lagrangian) position
# (Gadget HDF5: the snapshot's own or combined_ics.hdf5). Needs DTFE_DATA_ROOT (or -d, one simulation's folder) and
# the simulation (-s or DTFE_SIM); the shared defaults live in config.sh.
#
# Options: -d DIR (--sim-dir), -s SIM (--sim), -g N (--grid), -n K (--subsamples: k^3 points per cell of the sampled
#   deposit), -m (--gpu: starts the program with DTFE_GPU=1), -e (--exact, the default) or --sampled, --run-mode MODE
#   (gentle, balanced or performance; auto passes nothing), -h (--help), --version; then snapshot numbers (default 99).
# Defaults: 512^3 cells, the exact deposit, vertex masses, the caustics, the density, the velocity and its divergence
#   and shear, LAMBDA_TH 0.3, the program's gradient route and GPU share. Every knob (environment) is in the README's
#   knob table; DRY_RUN=1 prints the commands and runs nothing, AUTO_TUNE_REPORT=1 prints the auto-tuner's report.
# Outputs: <snapdir>/<OUTPUT_PREFIX>.* (default 'ps_output') beside <prefix>.runlog. A rerun moves the earlier run's
#   files it did not write to <prefix>.superseded/; a killed partitioned run resumes from its checkpoint
#   (DTFE_RESUME=0 starts over); <prefix>.runlog.lock refuses a second driver on the snapshot.

set -uo pipefail   # no -e: a batch goes on after a failed snapshot (config.sh, 'Strictness')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
source "${SCRIPT_DIR}/config.sh"

PARTITION="${PARTITION:-}"        # Lagrangian-partition grid. EMPTY (default) = the binary AUTO-TUNES the split;
                           # set to override, e.g. PARTITION="5 5 5".
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap on concurrent partitions. EMPTY (default) = auto-tuned with the split;
                           # set to override (0 = all cores).
SUBSAMPLES="${SUBSAMPLES:-}"    # nSub per axis, nSub^3 sub-points for the sampled deposit's '_a' fields (--subsamples,
                           # passed only with --sampled); EMPTY (default) = the binary's own 1. Set while
                           # DTFE_EXACT_AVERAGE is unset (-n k or SUBSAMPLES=k), it selects the sampled deposit;
                           # beside the exact one the driver drops it
MPC_UNIT=1000              # length of 1 Mpc in the input's units (1000 for ckpc/h): a constant, not a knob (TNG only)
THREADS="${THREADS:-}"     # cap OpenMP threads globally; empty = all cores
SCRATCH_DIR="${SCRATCH_DIR:-}"  # out-of-core mode (--scratch-dir): back the full-resolution grid accumulators
                           # (>= 1 GB allocations) with mmap'ed files in this LOCAL directory instead of RAM
                           # (bit-identical results; scratch files self-delete on any exit). Required for
                           # GRID_SIZE=1024. A non-synced path, e.g. /private/tmp/dtfe-scratch (mkdir it first):
                           # the binary rejects iCloud paths.
SAMPLE_POINTS="${SAMPLE_POINTS:-}"  # path to a --sample-points file (e.g. from python/tools/make_image_plane.py):
                           # evaluate the continuous field at those points in addition to the grid deposit, writing
                           # <snapdir>/<prefix>.pts_* (CPU, double precision), sharing the run's triangulation
                           # (see run_ps_pipeline.sh).
PTS_VEL_GRAD="${PTS_VEL_GRAD:-0}"   # 1 = with SAMPLE_POINTS, also write '.pts_velGrad' (--pts-vel-grad): the velocity
                           # gradient at each sample point (float64 x9, the plain mean over its streams), from which
                           # dtfelib.PointPlane derives the divergence / shear / vorticity maps
# DTFE_GPU and DTFE_RUN_MODE reach the binary in its environment as the caller set them (-m sets DTFE_GPU=1,
# --run-mode wins over DTFE_RUN_MODE); same knobs as run_dtfe.sh's.
GPU_SHARE="${GPU_SHARE:-auto}"   # the exact GPU deposit's --gpu-share, passed only when not auto (the binary's
                                 # default, the run mode's share): mixed (both devices in every run mode) or a
                                 # fraction 0..1
PS_MERGE="${PS_MERGE:-index-order}"   # how the partitions' grids are merged (--ps-merge, always passed):
                           #   index-order (default): partitions' grids are added in partition index order, so two identical
                           #   runs write bit-identical grids (and a killed run resumes to exactly the grids of an
                           #   uninterrupted one); STAGED: a finished partition parks its sub-grids in memory so no
                           #   thread idles. completion-order: added as they finish; grids differ in the last bits.
                           # Any other value stops the driver (exit 2, below)
DTFE_RESUME="${DTFE_RESUME:-1}"  # 1 (default) = resume a killed partitioned run when the snapshot's checkpoint file
                           # exists (config.sh ps_checkpoint_dir): --resume is passed, only missing partitions are
                           # computed. The binary refuses when the settings changed; DTFE_RESUME=0 starts over.
AUTO_TUNE_REPORT="${AUTO_TUNE_REPORT:-0}"  # 1 = only ASK the binary's auto-tuner what this run would need
                           # (--auto-tune-report): one AUTO-TUNE-REPORT line per snapshot after reading the input;
                           # nothing is computed or written. The GUI's memory check uses it;
                           # DTFE_AUTO_PTS_N / DTFE_AUTO_PTS_VEL_GRAD=1 size a plane not generated yet.
PS_VERTEX_MASS="${PS_VERTEX_MASS:-1}"  # 1 (default) = chart-independent tet masses (--ps-vertex-mass): each particle's
                           # mass splits equally among its incident tetrahedra. REQUIRED for TNG runs: the IC
                           # positions (z=127) already carry the growth factor, so the default rho_bar*V_lag masses
                           # would filter every density mode. Set 0 only for true-lattice Lagrangian inputs or A/B.
PS_CAUSTICS="${PS_CAUSTICS:-1}"      # 1 (default) = flag fold-caustic cells (--ps-caustics): writes '.caustic' (0/1
                           # fold flag) and '.causticClass', the caustic stratification bitmask, on the CPU and the
                           # GPU deposit (byte-identical: tests/ps_caustic_class_check.sh).
PS_CAUSTIC_CUSPS="${PS_CAUSTIC_CUSPS:-0}"  # 1 = also estimate the A3 CUSP (bit 7) and A4 SWALLOWTAIL (bit 8)
                           # indicators of '.causticClass'; requires PS_CAUSTICS=1, CPU deposit. Opt-in: both RANK
                           # candidates rather than certify them.
TESSELLATION_CACHE="${TESSELLATION_CACHE:-}"  # directory for the reusable on-disk tessellation (--tessellation-cache):
                           # re-running a snapshot with another grid, field set or point set skips the triangulation
                           # and the vertex-density pass. A LOCAL, non-synced dir.
_caller_exact="${DTFE_EXACT_AVERAGE:-}"   # (the caller's own value: unset, a SUBSAMPLES count selects the sampled deposit)
DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE:-1}"  # 1 (default) = exact conservative deposit (--exact): analytic tet-cell
                           # clipping instead of the nSub^3 sub-sampled deposit. Runs on the GPU when built (float32
                           # z-sweep; GPU_SHARE) and on the CPU with DTFE_GPU=0 (r3d in double, the reference).
                           # SUBSAMPLES is ignored (no count is passed beside --exact). 0 = the sampled deposit
                           # (--sampled); unset, a SUBSAMPLES count makes it 0 (config.sh average_resolve).
                           # Same knob and rule as run_dtfe.sh's; the driver passes --exact or --sampled every run
FACE_GRADIENT="${FACE_GRADIENT:-}"   # the velocity gradient's route (divergence, shear, vorticity and V-web are built
                           # on it), as run_dtfe.sh's. EMPTY (default) = the binary's own: the face gradient where
                           # the run qualifies (exact deposit, 3D), else the stream average (the run log's 'velocity
                           # gradient' line says which); 1 = --face-gradient (refused with the sampled deposit);
                           # 0 = --per-tet-gradient. The face run also writes '.a_velDivStreams' (and the other
                           # '.a_vel*Streams' for those FIELDS) and '.a_massFluxDiv' (div of sum_s rho_s v_s).

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
PARALLEL_TRIANGULATION="${PARALLEL_TRIANGULATION:-0}"  # 1 = force the parallel Delaunay insert at any size
                           # (--parallel-triangulation; needs the TBB build). 0 (default) passes nothing: the
                           # binary's own rule. Where the parallel insert runs, the single-precision grids repeat
                           # to the byte; point values (SAMPLE_POINTS) and the double build move at rounding.
PS_LAGRANGIAN_PADDING="${PS_LAGRANGIAN_PADDING:-}"   # k = --ps-lagrangian-padding k: pad each Lagrangian partition by k mean particle
                           # spacings. Empty (default) = the binary's default, k = 8.
                           # PS_LAGRANGIAN_PADDING=0 = the Eulerian rule (the PADDING fraction floored at 0.3 of a
                           # partition), which also reuses partition tessellation caches written under that
                           # rule; any k > 0 rebuilds them once.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ps_output}"   # -> <snapdir>/<prefix>.*  Override to keep incompatible runs side by
                           # side, e.g. OUTPUT_PREFIX=ps_sampled with --sampled. dtfelib.FieldSet reads the
                           # 'ps_output.' prefix, so keep the primary set there.

usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n SUBSAMPLES] [-m] [-e | --sampled] [--run-mode MODE] [snapshot ...]"; }

driver_help_version "${BASH_SOURCE[0]}" "${1:-}"   # -h / --help print the header and run nothing; --version the revision

DRY_RUN="${DRY_RUN:-0}"          # 1 = print the commands and paths, run nothing
# every switch 1 or 0 (config.sh knob_bool: 0, false, no, off are off, any other value on)
knob_bool DRY_RUN AUTO_TUNE_REPORT PTS_VEL_GRAD DTFE_RESUME PS_VERTEX_MASS PS_CAUSTICS PS_CAUSTIC_CUSPS \
          DTFE_EXACT_AVERAGE FACE_GRADIENT PARALLEL_TRIANGULATION
own_option() {   # <letter>: this driver's own options; -d, -s, -h and the data checks are config.sh driver_options'
    case "$1" in
        g) GRID_SIZE="${OPTARG}" ;;
        n) SUBSAMPLES="${OPTARG}" ;;
        m) _gpu_flag=1; export DTFE_GPU=1 ;;   # the binary reads it (config.sh device_given names it)
        e|S) average_option "$1" ;;      # -e / --exact, --sampled (config.sh: S only carries the long form)
        R) run_mode_option "${OPTARG}" ;;   # --run-mode MODE (config.sh: R only carries the long form)
    esac
}
run_mode_args=()            # --run-mode MODE as given (config.sh run_mode_option); empty: the binary decides
driver_options "g:n:meSR:" own_option "$@"   # then the data root (unless -d) and the simulation are required
shift $((OPTIND - 1))
average_resolve "deposit"   # DTFE_EXACT_AVERAGE 1 or 0: the options, the knob, a SUBSAMPLES count, the default

snapshot_args "$@"              # positional arguments override the default snapshot list (config.sh)
# PS_MERGE's value, refused before anything runs (a dry run too)
case "${PS_MERGE}" in
    index-order|completion-order) ;;
    *) echo "Error: PS_MERGE must be 'index-order' or 'completion-order' (got '${PS_MERGE}')." >&2; exit 2 ;;
esac

[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"
# preflight: an unmounted data disk is said once, up front, rather than as an 'Input file not found' for every
# snapshot under a path that was never there
data_root_problem "${DATA_DIR}" || exit 2

# Lagrangian positions, matched to present-day Coordinates by ParticleID, in the SAME units as the snapshots
# (h-removed ckpc: the IC was rescaled by 1/h via python/tools/convert_ic_units.py). Do NOT reconstruct the grid
# from ParticleID: TNG IDs are Peano-Hilbert ordered.

# Passed when present but NOT required: PS-DTFE reads Lagrangian positions from the snapshot's own
# 'InitialCoordinates' when it has them, and the file only when it has not. ${VAR-default} not ${VAR:-default}, so
# an explicitly empty PS_LAGRANGIAN_INPUT= forces the snapshot path even if an IC file sits beside it.
PS_LAGRANGIAN_INPUT="${PS_LAGRANGIAN_INPUT-${DATA_DIR}/combined_ics.hdf5}"

# python with a WORKING h5py, used only to look for the 'InitialCoordinates' dataset below (config.sh h5py_python)
PY_H5="$(h5py_python)"

# Echoes yes/no/unknown for whether $1 carries its own Lagrangian positions. On 'unknown' (no usable h5py) the
# run is attempted and the binary decides: guessing "no" would silently drop usable snapshots.
snapshot_has_initial_coords() {
    "${PY_H5}" - "$1" 2>/dev/null <<'PYEOF' || echo unknown
import sys
try:
    import h5py
except ImportError:
    sys.exit(1)
with h5py.File(sys.argv[1], "r") as f:
    for key in f:
        if key.startswith("PartType") and "InitialCoordinates" in f[key]:
            print("yes")
            sys.exit(0)
    print("no")
PYEOF
}

# Phase-space fields; '.streams' and '.hidden_streams' are always written, each field also gets a '_a' (averaged)
# form. Single-stream cells are streams==1 AND bit 1 of hidden_streams clear. The exact deposit makes the two forms
# coincide; without the face gradient (FACE_GRADIENT=0) the divergence/shear/vorticity are the stream average
# (rigorous only in single-stream cells; vweb inherits that). tweb = T-web from the tidal tensor (FFT Poisson solve of
# the RAW density grid); vweb = V-web from the velocity shear; LAMBDA_TH thresholds both. No smoothing in the binary.
LAMBDA_TH="${LAMBDA_TH:-0.3}"
# The analysis set, the same as run_dtfe.sh's: density, velocity, divergence, shear, derived from the velocity
# gradient (the mean velocity's face sums by default, else the stream average; see FACE_GRADIENT). The gradient,
# vorticity, the PS-only dispersion and the T-web / V-web classes are written only when FIELDS names them (all nine:
# the grids of the production runs).
# Env-overridable, e.g. FIELDS="density_a velocity_a" for a lighter batch.
FIELDS="${FIELDS:-density_a velocity_a divergence_a shear_a}"
SCALAR_DATASET="${SCALAR_DATASET:-}"   # a per-particle dataset of the input to grid as the 'scalar' field, e.g.
                           # Potential: --scalar-dataset NAME, scalar_a (and density_a) added to FIELDS (config.sh
                           # scalar_dataset_resolve, as run_dtfe.sh). A cell holds the mean of each stream's linear
                           # interpolant weighted by the stream's volume in it, in the dataset's own units. Give it
                           # a prefix of its own
scalar_dataset_resolve

cd "$REPO_ROOT" || exit 1

# DTFE_PRECISION=double runs the double-precision pair (./PS-DTFE-double, 'make PS-DTFE DOUBLE=1'); default
# single (./PS-DTFE). Nothing falls back silently: a missing binary stops.
DTFE_PRECISION="${DTFE_PRECISION:-single}"
driver_binary PS_BIN PS-DTFE "${DTFE_PRECISION}"      # ./PS-DTFE or ./PS-DTFE-double (config.sh)
if [ "${FACE_GRADIENT}" = "1" ]; then                 # the binary refuses these too, but per snapshot: say it once here
    # (the exact deposit is the default: what conflicts is --sampled, DTFE_EXACT_AVERAGE=0 or a count alone)
    [ "${DTFE_EXACT_AVERAGE}" = "1" ] || { echo "Error: FACE_GRADIENT=1 needs the exact deposit (the default): ${average_src} asks for the sampled one." >&2; exit 2; }
    # (the face sums also come beside the GPU deposit, from a CPU walk: -m / DTFE_GPU=1 is allowed)
fi

n_ok=0; n_skipped=0; n_failed=0; n_planned=0   # summarised at the end; see the exit-status note there

time_wrap_resolve                                     # TIME_WRAP: the optional peak-RSS wrapper (config.sh)
[ -n "$THREADS" ] && export OMP_NUM_THREADS="$THREADS"
keep_awake "${DRY_RUN}"
# the run-log contract (config.sh), the same in run_dtfe.sh and the launcher's own runs: '<prefix>.runlog.partial'
# while the binary runs, then '<prefix>.runlog' on exit 0 or '<prefix>.runlog.failed' on a failure or a stop; a
# stop also releases the lock and re-raises the signal
runlog_contract_start

echo "Starting PS-DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}   Partition: [${PARTITION:-auto}]   Fields: ${FIELDS}"
[ -n "${SCALAR_DATASET}" ] && echo "Particle value gridded as the scalar: ${SCALAR_DATASET} (--scalar-dataset)"
if [ "${DTFE_EXACT_AVERAGE}" = "1" ]; then
    echo "Cell averages: exact deposit (--exact; ${average_src})"
else
    echo "Cell averages: sampled deposit, nSub^3 sub-points per cell (--sampled${SUBSAMPLES:+ --subsamples ${SUBSAMPLES}}; ${average_src})"
fi
# the device and the run mode as the driver was given them (config.sh device_given); the run log says what ran
echo "Deposit: $(device_given)   caustics: ${PS_CAUSTICS}   face gradient: ${FACE_GRADIENT:-default}   prefix: ${OUTPUT_PREFIX}"
echo "Run mode: $(run_mode_given)${OMP_NUM_THREADS:+ (OMP_NUM_THREADS=${OMP_NUM_THREADS} keeps its thread count)}"
echo ""

for i in "${SNAPSHOTS[@]}"; do
    if ! snapshot_start "$i"; then          # n_str, input_dir, input_file, output_root (config.sh)
        n_skipped=$((n_skipped+1))
        continue
    fi

    # Caustic flagging (PS_CAUSTICS=1) and the opt-in A3 cusp indicator (PS_CAUSTIC_CUSPS=1).
    caustic_args=()
    [ "${PS_CAUSTICS}" = "1" ] && caustic_args=(--ps-caustics)
    [ "${PS_CAUSTICS}" = "1" ] && [ "${PS_CAUSTIC_CUSPS}" = "1" ] && caustic_args+=(--ps-caustic-cusps)

    # Reusable on-disk tessellation (TESSELLATION_CACHE=<dir>); created if missing, since the binary
    # requires an EXISTING directory and refuses iCloud paths.
    tess_args=()
    if [ -n "${TESSELLATION_CACHE}" ]; then
        [ "${DRY_RUN:-0}" = "1" ] || mkdir -p "${TESSELLATION_CACHE}"    # (a dry run creates nothing)
        tess_args=(--tessellation-cache "${TESSELLATION_CACHE}")
    fi

    # The deposit, named both ways (the run log then says which ran): the exact one (-e / DTFE_EXACT_AVERAGE=1, the
    # default; CPU and GPU deposits) or --sampled, with --subsamples only when SUBSAMPLES is set (the binary's 1 else).
    exact_args=(--exact)
    if [ "${DTFE_EXACT_AVERAGE}" != "1" ]; then
        exact_args=(--sampled)
        [ -n "${SUBSAMPLES}" ] && exact_args+=(--subsamples "${SUBSAMPLES}")
    fi

    # Chart-independent tet masses (PS_VERTEX_MASS=1, default -- see the comment at the top).
    vmass_args=()
    [ "${PS_VERTEX_MASS}" = "1" ] && vmass_args=(--ps-vertex-mass)

    # The velocity gradient's route (FACE_GRADIENT, see the top; empty passes neither: the binary's own).
    grad_args=()
    [ "${FACE_GRADIENT}" = "1" ] && grad_args+=(--face-gradient)
    [ "${FACE_GRADIENT}" = "0" ] && grad_args+=(--per-tet-gradient)  # the stream average of each stream's own gradient

    # The parallel Delaunay insert forced at any size (PARALLEL_TRIANGULATION=1; 0 leaves it to the binary's own
    # rule -- see the comment at the top).
    ptri_args=()
    [ "${PARALLEL_TRIANGULATION}" = "1" ] && ptri_args=(--parallel-triangulation)
    # (PS_LAGRANGIAN_PADDING, see the top; 0 passed too)
    [ -n "${PS_LAGRANGIAN_PADDING}" ] && ptri_args+=(--ps-lagrangian-padding "${PS_LAGRANGIAN_PADDING}")

    # Point evaluation on top of the grid run (SAMPLE_POINTS=<file>, see the comment at the top).
    sp_args=()
    [ -n "${SAMPLE_POINTS}" ] && sp_args=(--sample-points "${SAMPLE_POINTS}")
    [ -n "${SAMPLE_POINTS}" ] && [ "${PTS_VEL_GRAD}" = "1" ] && sp_args+=(--pts-vel-grad)

    # Lagrangian positions: pass the separate IC file when it exists (the binary still uses the snapshot's own
    # 'InitialCoordinates' when the snapshot has them, and reads the file only when it has not), else rely on
    # the snapshot's own 'InitialCoordinates'. Skip only when NEITHER is available.
    lag_args=()
    if [ -n "${PS_LAGRANGIAN_INPUT}" ] && [ -f "${PS_LAGRANGIAN_INPUT}" ]; then
        lag_args=(--ps-lagrangian-input "${PS_LAGRANGIAN_INPUT}")
    else
        case "$(snapshot_has_initial_coords "${input_file}")" in
            yes)
                echo "  Lagrangian positions: the snapshot's own 'InitialCoordinates'"
                [ -n "${PS_LAGRANGIAN_INPUT}" ] && echo "    (no ${PS_LAGRANGIAN_INPUT##*/} needed)"
                ;;
            no)
                echo "  Warning: snapshot ${n_str} has NO Lagrangian positions:"
                [ -n "${PS_LAGRANGIAN_INPUT}" ] && \
                    echo "           '${PS_LAGRANGIAN_INPUT}' does not exist, and"
                echo "           the snapshot carries no 'InitialCoordinates' dataset either."
                echo "           Convert an IC snapshot with python/tools/convert_ic_units.py (it must"
                echo "           be in the snapshot's units: h-removed ckpc for combined_*.hdf5)."
                echo "  Skipping snapshot ${n_str}"
                n_skipped=$((n_skipped+1))
                continue
                ;;
            *)
                echo "  Note: cannot check for 'InitialCoordinates' (no usable h5py); running without"
                echo "        --ps-lagrangian-input. PS-DTFE reports precisely if the dataset is missing."
                ;;
        esac
    fi

    # TIME_WRAP (resolved above) captures peak memory. The tee pipe hides the terminal from the binary's isatty()
    # check, so force colours through it (CLICOLOR_FORCE) only when this script itself runs on a terminal; the
    # runlog is de-ANSI'd after the run.
    [ -t 1 ] && export CLICOLOR_FORCE=1
    echo "  Running PS-DTFE on ${input_file}..."
    run_log="${input_dir}/${OUTPUT_PREFIX}.runlog"
    # a report writes nothing next to the data: not even over the snapshot's real run log
    [ "${AUTO_TUNE_REPORT}" = "1" ] && [ "${DRY_RUN}" != "1" ] && run_log="$(mktemp -t dtfe-auto-tune-report.XXXXXX)"
    SECONDS=0
    # pass --partition/--max-concurrent only when set; otherwise the binary auto-tunes them
    part_args=()
    report_args=()
    [ "${AUTO_TUNE_REPORT}" = "1" ] && report_args=(--auto-tune-report)
    [ -n "${PARTITION}" ] && part_args+=(--partition ${PARTITION})
    [ -n "${MAX_CONCURRENT}" ] && part_args+=(--max-concurrent "${MAX_CONCURRENT}")
    [ -n "${SCRATCH_DIR}" ] && part_args+=(--scratch-dir "${SCRATCH_DIR}")
    # the merge mode (PS_MERGE, see the top: index-order by default, passed explicitly so the memory gate stays on
    # under the auto-tuner's speed split; the run log names it, results.py reads it back)
    part_args+=(--ps-merge "${PS_MERGE}")
    # the run mode only when --run-mode named one (DTFE_RUN_MODE reaches the binary otherwise), and the GPU share only
    # when GPU_SHARE is not auto; a share given for a run it does not apply to is passed too, so the binary's note
    # says why ('mixed' is silent there)
    part_args+=(${run_mode_args[@]+"${run_mode_args[@]}"})
    [ "${GPU_SHARE}" != "auto" ] && part_args+=(--gpu-share "${GPU_SHARE}")
    # a killed partitioned run left its checkpoint: resume it (config.sh resume_args_for)
    resume_args_for "${output_root}" DTFE_RESUME "${DTFE_RESUME}"

    cmd=("${PS_BIN}" "${input_file}" "${output_root}"
        --grid ${GRID_SIZE}
        --padding ${PADDING}
        --periodic
        ${part_args[@]+"${part_args[@]}"}
        --input-format 105
        --mpc-unit ${MPC_UNIT}
        --field ${FIELDS}
        --lambda-th ${LAMBDA_TH}
        ${scalar_args[@]+"${scalar_args[@]}"}
        ${exact_args[@]+"${exact_args[@]}"}
        ${vmass_args[@]+"${vmass_args[@]}"}
        ${ptri_args[@]+"${ptri_args[@]}"}
        ${grad_args[@]+"${grad_args[@]}"}
        ${sp_args[@]+"${sp_args[@]}"}
        ${caustic_args[@]+"${caustic_args[@]}"}
        ${tess_args[@]+"${tess_args[@]}"}
        ${report_args[@]+"${report_args[@]}"}
        ${resume_args[@]+"${resume_args[@]}"}
        ${lag_args[@]+"${lag_args[@]}"})
    if [ "${DRY_RUN}" = "1" ]; then                 # DRY_RUN=1: the expanded command and the paths, nothing run
        dry_run_print "${input_file}" "${output_root}" "${run_log}" "${cmd[@]}"
        n_planned=$((n_planned + 1))
        echo ""
        continue
    fi
    # Real runs only: room for the grids, then the snapshot's single-instance lock (config.sh).
    # A report writes nothing, so it takes neither: the launcher's memory check runs one beside a live run.
    if [ "${AUTO_TUNE_REPORT}" != "1" ] && ! snapshot_preflight "${input_dir}" "${run_log}" "${DTFE_PRECISION}"; then
        n_failed=$((n_failed+1)); echo ""; continue
    fi
    # the binary's output goes to '<run_log>.partial' until it exits (the run-log contract above);
    # a report's temp file takes it directly
    sup_list=""
    if [ "${AUTO_TUNE_REPORT}" = "1" ]; then
        tee_log="${run_log}"
    else
        tee_log="${run_log}.partial"; live_log="${tee_log}"; live_rc=""
        # what an earlier run with this prefix left, recorded before this one starts: what it does not rewrite is
        # moved aside once it has exited 0 (config.sh sup_record, run_finish)
        sup_record "${output_root}" "${run_log}"
    fi
    ${TIME_WRAP[@]+"${TIME_WRAP[@]}"} "${cmd[@]}" 2>&1 | tee "${tee_log}"
    rc=${PIPESTATUS[0]}
    live_rc="${rc}"

    if [ "${AUTO_TUNE_REPORT}" = "1" ]; then
        if auto_tune_report_line "${run_log}" "${rc}"; then n_ok=$((n_ok+1)); else n_failed=$((n_failed+1)); fi
        continue
    fi

    # de-ANSI'd, the log becomes '<prefix>.runlog' on success and '<prefix>.runlog.failed' otherwise
    # (config.sh run_finish); the peak memory below is read from whichever it became
    run_finish "${rc}" "${output_root}"
    [ "${rc}" -eq 0 ] || run_log="${run_log}.failed"

    # peak memory (config.sh peak_memory_text; absent without TIME_WRAP) and the auto-tuner's budget
    peak_gb="$(peak_memory_text "${run_log}")"
    if [ -n "${peak_gb}" ]; then
        peak_gb="${peak_gb};"
        # the budget the binary's auto-tuner planned against (see src/auto_tune.h) -- absent when --partition and
        # --max-concurrent were both given
        budget_gb=$(sed -n 's/^.*memory budget: \([0-9.]*\) GB.*$/\1/p' "${run_log}" | head -1)
        if [ -n "${budget_gb}" ]; then
            echo "  Peak memory: ${peak_gb} auto-tune budget ${budget_gb} GB (DTFE_MEM_BUDGET_GB or MAX_CONCURRENT=1 caps it further)"
        else
            echo "  Peak memory: ${peak_gb} partition and concurrency set by hand, so no auto-tune budget applied"
        fi
    fi
    if snapshot_outcome "${rc}" "${run_log}"; then n_ok=$((n_ok+1)); else n_failed=$((n_failed+1)); fi
    echo ""
done

# Honest exit status: processing NOTHING, or failing a snapshot that was attempted, exits 1. A
# partial skip stays exit 0 -- a partly-downloaded snapshot ladder is normal.
[ "${DRY_RUN}" = "1" ] && dry_run_exit "${n_planned}" "${n_skipped}" "${DATA_DIR}"
echo "PS-DTFE processing complete: ${n_ok} processed, ${n_skipped} skipped, ${n_failed} failed."
if [ "${n_failed}" -gt 0 ]; then
    echo "ERROR: ${n_failed} snapshot(s) failed; see the messages above and their .runlog.failed files." >&2
    exit 1
fi
if [ "${n_ok}" -eq 0 ]; then
    echo "ERROR: no snapshot was processed. ${n_skipped} were skipped for missing inputs -- check" >&2
    echo "       DATA_ROOT/SIMULATION (looked under '${DATA_DIR}'), and that each snapshot has" >&2
    echo "       Lagrangian positions: either a combined_ics.hdf5 beside it or its own" >&2
    echo "       'InitialCoordinates' dataset (the per-snapshot warnings above say which)." >&2
    exit 1
fi
