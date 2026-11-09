#!/usr/bin/env bash
# Phase-Space DTFE counterpart of run_dtfe.sh: tessellates in Lagrangian space, so each
# particle needs Eulerian Coordinates plus a Lagrangian position. HDF5-only (input 105).
# DTFE_DATA_ROOT (the folder that holds the simulations; -d names a simulation's folder instead) and the simulation
# (-s or DTFE_SIM) are required; the
# shared defaults (SNAPSHOTS, GRID_SIZE, PADDING) live in config.sh.
#
# Usage:
#   ./run_ps_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n SUBSAMPLES] [-m] [-e] [snapshot ...]
#     long forms: --sim-dir, --sim, --grid, --subsamples, --gpu, --exact-average (the same as the letters)
#     -n   SUBSAMPLES: k sub-points per axis (k^3 per cell) of the sampled deposit's '_a' fields (--subsamples k;
#          default 3)
#     -m   run the deposit on the GPU (--gpu; same as DTFE_GPU=1). The default (DTFE_GPU unset, 'auto') is the GPU
#          when the binary was built with one (the Makefile's o_ps/.gpu_mode_metal stamp; o_ps_d for
#          DTFE_PRECISION=double), else the CPU -- but the CPU for the exact deposit (-e), faster there on
#          production runs; DTFE_GPU=0 runs the CPU deposit on a GPU build.
#     -e   exact conservative deposit (same as DTFE_EXACT_AVERAGE=1; --exact-average, on the CPU unless -m)
#     The drivers' letters are their own: -m is the GPU and -s the simulation here (PS-DTFE's own -m and -s are
#          deprecated and have no effect there).
#   Production defaults: GRID_SIZE 512 (config.sh), the GPU when built, vertex masses, volume-weighted
#   velocity moments, the caustic classes, FIELDS = density, velocity, its divergence and shear (the same set as
#   run_dtfe.sh's; the gradient, vorticity, dispersion, T-web and V-web through FIELDS), LAMBDA_TH 0.3.
#   A rerun into a prefix that holds an earlier run's files: once it exits 0, the files it did not write (an image
#   plane, caustics or a field the new run lacks) are MOVED into <prefix>.superseded/<date_time>/ with the old
#   run log (config.sh supersede_extras; never deleted), so the folder holds one run's set.
#   -h / --help prints this header and runs nothing; --version the repository revision.
#   DRY_RUN=1 ./run_ps_dtfe.sh ...   prints every snapshot's expanded command and paths, runs nothing
#          (exit 0 when at least one snapshot is planned).
#   Run logs, beside the outputs: <prefix>.runlog belongs to the grids on disk, <prefix>.runlog.failed
#   to the last failed or stopped attempt, <prefix>.runlog.partial to a run still going (or killed outright).
#   Knobs (environment; each is explained where the script sets it): PARTITION MAX_CONCURRENT SUBSAMPLES THREADS
#   SCRATCH_DIR SAMPLE_POINTS PTS_VEL_GRAD PS_POINT_CONVENTIONS DTFE_GPU DTFE_EXACT_AVERAGE FACE_GRADIENT
#   PS_VERTEX_MASS PS_VOLUME_WEIGHTED PS_LINEAR_DEPOSIT PS_CAUSTICS PS_CAUSTIC_CUSPS PARALLEL_TRIANGULATION
#   PS_LAGRANGIAN_PADDING TESSELLATION_CACHE OUTPUT_PREFIX FIELDS LAMBDA_TH PS_LAGRANGIAN_INPUT DTFE_PRECISION DRY_RUN
#   AUTO_TUNE_REPORT (1 = only the auto-tuner's memory report per snapshot, nothing computed) PS_MERGE DTFE_RESUME,
#   and config.sh's GRID_SIZE, PADDING, SPACE_CHECK. A switch is off for 0, false, no or off and on for any other
#   value (the binaries' rule). DTFE_GPU, DTFE_EXACT_AVERAGE and DTFE_RESUME are run_dtfe.sh's knobs too (each driver
#   keeps its default: the exact average is run_dtfe.sh's, the sampled deposit this one's).
#   Older names, read for one release with a warning (the new name wins when both are set): PS_GPU and PS_METAL
#   (DTFE_GPU), PS_EXACT (DTFE_EXACT_AVERAGE), PS_RESUME (DTFE_RESUME), AVG_SUBSAMPLES (SUBSAMPLES), ORDERED_MERGE and
#   FIXED_POINT_MERGE (PS_MERGE), PS_PARALLEL_TRI (PARALLEL_TRIANGULATION), PS_LAG_PADDING (PS_LAGRANGIAN_PADDING),
#   LAGRANGIAN_INPUT (PS_LAGRANGIAN_INPUT), PTS_CONVENTIONS (PS_POINT_CONVENTIONS), TESS_CACHE (TESSELLATION_CACHE),
#   DTFE_PADDING, DTFE_GRID_SIZE, DTFE_SPACE_CHECK.
#   A partitioned run killed part-way resumes: when the snapshot's checkpoint (<prefix>.runlog.checkpoint, or in
#   SCRATCH_DIR) exists, the next run passes --resume (DTFE_RESUME=0 starts over).
#   While a snapshot runs, <prefix>.runlog.lock (a folder holding the driver's pid) refuses a second driver on
#   it; a lock left by a killed driver is removed with a note. Before anything runs: the data disk must be
#   there (exit 2 otherwise) and the output folder writable with room for the grids (SPACE_CHECK=0 skips
#   the space half).

set -uo pipefail   # no -e: a batch goes on after a failed snapshot (config.sh, 'Strictness')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
source "${SCRIPT_DIR}/config.sh"

PARTITION="${PARTITION:-}"        # Lagrangian-partition grid. EMPTY (default) = the binary AUTO-TUNES the split
                           # from the particle count, grid, fields and available RAM/cores (see the AUTO-TUNE
                           # line it prints). Set to override, e.g. PARTITION="5 5 5"; manual reference points:
                           # 5 5 5 fits TNG300-3-Dark (2.4e8 particles) at MAX_CONCURRENT=2 in ~40 GB,
                           # 2 2 2 suits small sims like TNG50-4 (fewer partition overheads = faster).
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap on concurrent partitions. EMPTY (default) = auto-tuned together with the
                           # partition split; set to override (0 = all cores). Peak RAM ~ fixed + cap x
                           # per-triangulation, verify via the "Peak memory (RSS)" line after each run.
knob SUBSAMPLES AVG_SUBSAMPLES     # (config.sh knob: the older name read for one release, with a warning)
SUBSAMPLES="${SUBSAMPLES:-3}"   # nSub per axis, nSub^3 sub-points for the '_a' fields (--subsamples); dominant runtime cost (~nSub^3), 1 = no averaging. Override: -n 1 or SUBSAMPLES=1
MPC_UNIT=1000              # length of 1 Mpc in the input's units (1000 for ckpc/h): a constant, not a knob (TNG only)
THREADS="${THREADS:-}"     # cap OpenMP threads globally; empty = all cores
SCRATCH_DIR="${SCRATCH_DIR:-}"  # out-of-core mode (--scratch-dir): back the full-resolution grid
                           # accumulators (>= 1 GB allocations) with mmap'ed files in this LOCAL
                           # directory instead of RAM. Required for GRID_SIZE=1024 (~90 GB of
                           # accumulators with the default FIELDS, ~146 GB with all nine); bit-identical
                           # results, RSS stays bounded, scratch files self-delete on any exit.
                           # MUST be a local non-synced path, e.g. /private/tmp/dtfe-scratch
                           # (mkdir it first) -- iCloud paths are rejected by the binary.
SAMPLE_POINTS="${SAMPLE_POINTS:-}"  # path to a --sample-points file (e.g. from
                           # python/tools/make_image_plane.py): evaluate the continuous field at
                           # those points IN ADDITION to the grid deposit, writing
                           # <snapdir>/<prefix>.pts_* (CPU, double precision). Shares the run's
                           # triangulation, so the marginal cost is just the evaluation --
                           # this is how the high-resolution figure slices piggyback on a
                           # production grid run (see run_ps_pipeline.sh).
PTS_VEL_GRAD="${PTS_VEL_GRAD:-0}"   # 1 = with SAMPLE_POINTS, also write '.pts_velGrad'
                           # (--pts-vel-grad): the density-weighted velocity gradient at each
                           # sample point (float64 x9). dtfelib.PointPlane derives the
                           # divergence / shear / vorticity maps from it, so this is what makes
                           # the velocity-derivative fields available to the hi-res figures.
knob PS_POINT_CONVENTIONS PTS_CONVENTIONS
PS_POINT_CONVENTIONS="${PS_POINT_CONVENTIONS:-grid}"  # with SAMPLE_POINTS: the points' conventions, the binary's
                           # --ps-point-conventions. 'grid' (default; the binary's own) = the grids' vertex
                           # masses (PS_VERTEX_MASS) and volume weighting (PS_VOLUME_WEIGHTED) at the points too;
                           # 'legacy' = the reference implementation's estimators (rho_bar*V_lag masses, density-
                           # weighted stream means), which reproduce a plane made with them -- about 0.82 of the grid's
                           # density contrast at z = 20 (the launcher's Runs tab warns about such planes). Only
                           # 'legacy' is passed, so a 'grid' run's command line carries no --ps-point-conventions.
knob DTFE_GPU PS_GPU PS_METAL
DTFE_GPU="${DTFE_GPU:-auto}"  # 1 = the deposit on the GPU (--gpu), 0 = the CPU; 'auto' (default: it
                           # follows the build) = 1 when the binary is a GPU build (config.sh gpu_build_default),
                           # resolved below once DTFE_PRECISION has chosen the binary; 0 with DTFE_EXACT_AVERAGE=1.
                           # The same knob as run_dtfe.sh's
knob_merge PS_MERGE ORDERED_MERGE FIXED_POINT_MERGE
PS_MERGE="${PS_MERGE:-index-order}"   # how the partitions' grids are merged (--ps-merge, always passed):
                           #   index-order (default) = the binary's own order, GIVEN: the partitions' grids are added in
                           #   partition index order, so two identical runs write bit-identical grids (and a killed run
                           #   resumes to exactly the grids of an uninterrupted one); STAGED: a partition finished before
                           #   its turn parks its sub-grids in memory (while the machine has it free) and its thread goes
                           #   on, so no thread idles behind a slow partition. Passed explicitly, as the older
                           #   '--ordered-merge' default was, it also keeps the memory gate on when the auto-tuner splits
                           #   a small run for speed (the binary's --ps-merge help; a bare binary run drops it there).
                           #   completion-order = added as they finish; the grids then differ run to run in the last bits.
                           #   fixed-point (OPT-IN, to compare against the default) = each partition added AS IT FINISHES
                           #   into exact integer full grids (9 B per summed component and cell during the loop instead
                           #   of 4), so the grids are the same bytes at any concurrency and completion order and nothing
                           #   parks or spills; they differ from the index order at ~1e-7. Not with the double build
                           #   (DTFE_PRECISION=double): the binary refuses it. A checkpoint of one mode is not resumed
                           #   by the other (the binary refuses; DTFE_RESUME=0 starts over).
                           # ORDERED_MERGE=1 / 0 and FIXED_POINT_MERGE=1, the older pair, read as index-order /
                           # completion-order and fixed-point (config.sh knob_merge)
knob DTFE_RESUME PS_RESUME
DTFE_RESUME="${DTFE_RESUME:-1}"  # 1 (default) = resume a killed partitioned run: when the snapshot's checkpoint file
                           # exists (<prefix>.runlog.checkpoint/checkpoint beside the outputs, or under SCRATCH_DIR; config.sh
                           # ps_checkpoint_dir), --resume is passed and only the partitions it lacks are computed.
                           # The binary refuses, naming the differing settings, when the settings changed; then
                           # DTFE_RESUME=0 starts over (the new run replaces the checkpoint). The same knob as run_dtfe.sh's
AUTO_TUNE_REPORT="${AUTO_TUNE_REPORT:-0}"  # 1 = only ASK the binary's auto-tuner what this run would need
                           # (--auto-tune-report): per snapshot one line
                           #   AUTO-TUNE-REPORT snap=NNN partition=.. mc=.. predicted_gb=.. budget_gb=.. over_budget=0|1 ...
                           # after reading the input, and nothing is computed or written (the
                           # snapshot's .runlog is left alone). The GUI's memory check uses it;
                           # DTFE_AUTO_PTS_N / DTFE_AUTO_PTS_VEL_GRAD=1 size a plane not generated yet.
PS_VERTEX_MASS="${PS_VERTEX_MASS:-1}"  # 1 (default) = chart-independent tet masses (--ps-vertex-mass):
                           # each particle's mass splits equally among its incident tetrahedra. REQUIRED for
                           # TNG runs: combined_ics.hdf5 holds the z=127 IC positions, whose configuration
                           # already carries delta_ic = D(127)/D(z)*delta -- the default rho_bar*V_lag masses
                           # then filter every density mode by 1-D(127)/D(z) (-16.5% at z=20, -3% at z=2;
                           # verified against regular DTFE on TNG50-3). Velocities are unaffected either way.
                           # Set PS_VERTEX_MASS=0 only for true-lattice Lagrangian inputs or A/B comparisons.
_caller_vw="${PS_VOLUME_WEIGHTED:-}"   # (the caller's own value: an unset one yields to PS_LINEAR_DEPOSIT=1, below)
PS_VOLUME_WEIGHTED="${PS_VOLUME_WEIGHTED:-1}"  # 1 (default, the production convention, as in
                           # run_ps_pipeline.sh and the launcher) = volume-weighted velocity moments
                           # (--ps-volume-weighted): velocity/gradient/div/shear/vort become VOLUME averages per
                           # cell (the standard-DTFE '_a' convention that -aHf*delta refers to) instead of the
                           # mass-weighted (momentum-like) means of PS_VOLUME_WEIGHTED=0. The DISPERSION is
                           # excluded and stays mass-weighted (sigma_ij is an f-weighted CBE moment by definition)
                           # -- it comes out bit-identical to a mass-weighted run, so this ONE config is the
                           # literature-standard estimator for every field at once. Works with the CPU and GPU (-m)
                           # deposits. Not with PS_LINEAR_DEPOSIT=1 (the binary refuses both): left unset, it is 0
                           # then (said below).
PS_CAUSTICS="${PS_CAUSTICS:-1}"      # 1 (default, as the production run of all 27
                           # snapshots: +33% time, +1.2 GB on TNG, every other output unchanged) =
                           # flag fold-caustic cells (--ps-caustics): writes '.caustic'
                           # (0/1 fold flag) and '.causticClass' -- the caustic stratification bitmask
                           # (fold parity, how many principal axes have collapsed, and the umbilic-
                           # degeneracy indicator) -- on the CPU AND the GPU deposit (byte-identical:
                           # tests/ps_caustic_class_check.sh); the linear deposit on the GPU writes the
                           # fold parity only (and says so): run that combination on the CPU.
PS_CAUSTIC_CUSPS="${PS_CAUSTIC_CUSPS:-0}"  # 1 = also estimate the A3 CUSP (bit 7) and A4
                           # SWALLOWTAIL (bit 8) indicators of '.causticClass'; requires
                           # PS_CAUSTICS=1, CPU deposit. Opt-in: both are finite differences over
                           # the deduplicated 64-nearest ring of tetrahedra, ~+22% on the caustic
                           # pass. Both RANK candidates rather than certify them.
knob TESSELLATION_CACHE TESS_CACHE
TESSELLATION_CACHE="${TESSELLATION_CACHE:-}"  # directory for the reusable on-disk tessellation
                           # (--tessellation-cache). Re-running the same snapshot with a different
                           # grid, field set or point set then skips the triangulation AND the
                           # vertex-density pass (measured 10.9 s -> 4.6 s at 884736 particles).
                           # Must be a LOCAL, non-synced dir; costs ~0.15 KB per tessellation
                           # vertex (the file is deflated: 225 MB per 1.5e6 vertices).
knob DTFE_EXACT_AVERAGE PS_EXACT
DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE:-0}"  # 1 = exact conservative deposit (--exact-average): analytic tet-cell
                           # clipping instead of the nSub^3 sub-sampled deposit -- no sampling noise, mass
                           # conservation to the arithmetic's precision. Runs on the CPU by default (r3d in
                           # double, the reference; TNG100-3-Dark z=0 at 512^3 in 14.5 min, the sampled GPU
                           # deposit's time) and on the GPU with -m (a float32 z-sweep, 17.7 min there; faster
                           # only around a few million particles). SUBSAMPLES is ignored (the exact
                           # deposit is the nSub->infinity limit, so '.den' == '.a_den'). Override: -e.
                           # The same knob as run_dtfe.sh's, whose default is 1 (its exact cell average)
FACE_GRADIENT="${FACE_GRADIENT:-0}"  # 1 = --face-gradient: the velocity gradient family (divergence, shear, ...) of the
                           # volume-weighted MEAN velocity from exact face sums, the quantity standard DTFE stores
                           # (conservative, the caustics' jumps as exact face differences), where the default stores
                           # the stream average of each stream's own gradient; it also writes '.a_velDivStreams' (that
                           # stream average's divergence) and '.a_massFluxDiv' (div of sum_s rho_s v_s: by continuity
                           # the rate a cell gains or loses mass). Needs the exact CPU deposit (-e, not -m) and
                           # PS_VOLUME_WEIGHTED=1 (the default); 3D. 30 more numbers per cell in the merged grids.

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
PS_LINEAR_DEPOSIT="${PS_LINEAR_DEPOSIT:-0}"  # 1 = --ps-linear-deposit: each tetrahedron's mass spread by the linear
                                 # density profile inside it (renormalized: mass-conserving); NOT with PS_VOLUME_WEIGHTED=1
knob_bool PS_LINEAR_DEPOSIT PS_VOLUME_WEIGHTED   # 1 or 0 (config.sh; the caller's own PS_VOLUME_WEIGHTED is _caller_vw)
# the volume weighting's default (1) yields to an asked-for linear deposit, which the binary refuses beside it
if [ -z "${_caller_vw}" ] && [ "${PS_LINEAR_DEPOSIT}" = "1" ]; then
    echo "Note: PS_LINEAR_DEPOSIT=1 weights by density inside each tetrahedron, so the velocity moments stay mass-weighted (PS_VOLUME_WEIGHTED=0)"
    PS_VOLUME_WEIGHTED=0
fi
knob PARALLEL_TRIANGULATION PS_PARALLEL_TRI
PARALLEL_TRIANGULATION="${PARALLEL_TRIANGULATION:-0}"  # 1 = build each tessellation's Delaunay triangulation in parallel
                           # (--parallel-triangulation; needs the TBB build, 'make' detects the library).
                           # Faster on small sets; OFF by default because two identical runs then
                           # differ at float rounding (stream counts do not). See the binary's help.
knob PS_LAGRANGIAN_PADDING PS_LAG_PADDING
PS_LAGRANGIAN_PADDING="${PS_LAGRANGIAN_PADDING:-}"   # k = --ps-lagrangian-padding k: pad each Lagrangian partition by k mean particle
                           # spacings. Empty (default) = the binary's default, k = 8: on a 4.1M-particle set at 2^3
                           # it runs ~6x faster than the Eulerian rule in 3.5x less memory with identical stream counts
                           # (density/velocity grids move in ~0.5% of cells, as two runs at PADDING 40 and 50 do).
                           # PS_LAGRANGIAN_PADDING=0 = the Eulerian rule (the PADDING fraction floored at 0.3 of a
                           # partition: 7-9x the owned particles per TNG partition), which also reuses partition
                           # tessellation caches written under that rule; any k > 0 rebuilds them once.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ps_output}"   # -> <snapdir>/<prefix>.*  Override to keep incompatible runs side by
                           # side instead of clobbering, e.g. OUTPUT_PREFIX=ps_mw with PS_VOLUME_WEIGHTED=0 for the
                           # mass-weighted (physical) dispersion next to the default volume-weighted shear/divergence
                           # set. dtfelib.FieldSet reads the 'ps_output.' prefix, so keep the primary set there.

usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n SUBSAMPLES] [-m] [-e] [snapshot ...]"; }

driver_help_version "${BASH_SOURCE[0]}" "${1:-}"   # -h / --help print the header and run nothing; --version the revision

DRY_RUN="${DRY_RUN:-0}"          # 1 = print the commands and paths, run nothing
# every switch 1 or 0 (config.sh knob_bool: 0, false, no, off are off, any other value on); DTFE_GPU in gpu_auto_resolve
knob_bool DRY_RUN AUTO_TUNE_REPORT PTS_VEL_GRAD DTFE_RESUME PS_VERTEX_MASS PS_CAUSTICS PS_CAUSTIC_CUSPS \
          DTFE_EXACT_AVERAGE FACE_GRADIENT PARALLEL_TRIANGULATION
own_option() {   # <letter>: this driver's own options; -d, -s, -h and the data checks are config.sh driver_options'
    case "$1" in
        g) GRID_SIZE="${OPTARG}" ;;
        n) SUBSAMPLES="${OPTARG}" ;;
        m) DTFE_GPU=1 ;;
        e) DTFE_EXACT_AVERAGE=1 ;;
    esac
}
driver_options "g:n:me" own_option "$@"   # then the data root (unless -d) and the simulation are required
shift $((OPTIND - 1))

snapshot_args "$@"              # positional arguments override the default snapshot list (config.sh)
case "${PS_POINT_CONVENTIONS}" in
    grid|legacy) ;;
    *) echo "Error: PS_POINT_CONVENTIONS must be 'grid' or 'legacy' (got '${PS_POINT_CONVENTIONS}')." >&2; exit 2 ;;
esac
case "${PS_MERGE}" in
    index-order|completion-order|fixed-point) ;;
    *) echo "Error: PS_MERGE must be 'index-order', 'completion-order' or 'fixed-point' (got '${PS_MERGE}')." >&2; exit 2 ;;
esac

[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"
# preflight: an unmounted data disk is said once, up front, rather than as an 'Input file not found' for every
# snapshot under a path that was never there
data_root_problem "${DATA_DIR}" || exit 2

# Lagrangian positions, matched to present-day Coordinates by ParticleID. Must be in the
# SAME units as the snapshots (combined_*.hdf5 are h-removed ckpc, so the IC was rescaled
# by 1/h via python/tools/convert_ic_units.py). Do NOT reconstruct the grid from ParticleID: TNG IDs
# are Peano-Hilbert ordered, so id->(ix,iy,iz) is wrong.
#
# Used when present but NOT required: PS-DTFE reads Lagrangian positions from the snapshot's own
# 'InitialCoordinates' when '--ps-lagrangian-input' is omitted. ${VAR-default} not ${VAR:-default}, so
# an explicitly empty PS_LAGRANGIAN_INPUT= forces the snapshot path even if an IC file sits beside it (and so does
# an empty LAGRANGIAN_INPUT=, its older name: knob -e counts an empty value as set).
knob -e PS_LAGRANGIAN_INPUT LAGRANGIAN_INPUT
PS_LAGRANGIAN_INPUT="${PS_LAGRANGIAN_INPUT-${DATA_DIR}/combined_ics.hdf5}"

# python with a WORKING h5py, used only to look for the 'InitialCoordinates' dataset below (config.sh h5py_python)
PY_H5="$(h5py_python)"

# Echoes yes/no/unknown for whether $1 carries its own Lagrangian positions. On 'unknown' (no usable
# h5py) we neither guess nor skip: the run is attempted and the binary decides, since it errors
# precisely when the dataset is absent. Guessing "no" would silently drop usable snapshots.
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

# Phase-space fields; '.streams' and '.hidden_streams' are always written, each field also gets a '_a'
# (averaged) form. Single-stream cells are streams==1 AND bit 1 of hidden_streams clear (see demo/COMMANDS.md).
# Both forms are mass-conserving (each tetrahedron deposits its full mass onto the grid). The '_a'
# fields resolve each grid cell with an nSub^3 sub-sample grid (SUBSAMPLES, default 3), so they
# are smoother / better resolved at caustics -- prefer them for science plots. The plain fields use a
# single sample per cell (coarser, but the same conserved quantity).
# divergence/shear/vorticity are rigorous only in single-stream cells; for multi-stream kinematics
# use 'dispersion_a' + the stream count. vweb classifies the cosmic web from the velocity
# gradient (density-weighted across streams), so it inherits the same single-stream caveat.
# tweb = T-web from the TIDAL tensor (FFT Poisson solve of the RAW density grid, in C++); vweb =
# V-web from the multi-stream velocity shear. LAMBDA_TH is the eigenvalue threshold for BOTH webs
# (dimensionless; literature ~0.2-0.4; the binary's own default is 0.0).
# The binary applies NO smoothing anywhere; smoothing is a plot-time choice in plot_PS_DTFE.py.
LAMBDA_TH="${LAMBDA_TH:-0.3}"
# The analysis set, the same as run_dtfe.sh's: density, velocity, divergence, shear. The divergence and shear grids
# are derived from the multi-stream velocity gradient, the stream average <grad v> (single-stream caveat above);
# dtfelib's figures differentiate the velocity grid instead (the mean field, as standard DTFE's), unless run with
# '--ps-gradient stored'. The gradient, vorticity, the PS-only dispersion and the T-web / V-web classes are written
# only when FIELDS names them (all nine: the grids of the 2026-10 production runs).
# Env-overridable, e.g. FIELDS="density_a velocity_a" for a lighter batch.
FIELDS="${FIELDS:-density_a velocity_a divergence_a shear_a}"

cd "$REPO_ROOT" || exit 1

# DTFE_PRECISION=double runs the double-precision pair (./PS-DTFE-double, 'make PS-DTFE DOUBLE=1'):
# every position, density and field in double from the input read onward, float64 outputs, about
# twice the memory. Default single (./PS-DTFE). Nothing falls back silently: a missing binary stops.
DTFE_PRECISION="${DTFE_PRECISION:-single}"
driver_binary PS_BIN PS-DTFE "${DTFE_PRECISION}"      # ./PS-DTFE or ./PS-DTFE-double (config.sh)
gpu_auto_resolve DTFE_GPU "${PS_BIN}" DTFE_EXACT_AVERAGE   # DTFE_GPU=auto: the GPU when this binary was built with
                                                      # one, the CPU for the exact deposit
if [ "${FACE_GRADIENT}" = "1" ]; then                 # the binary refuses these too, but per snapshot: say it once here
    [ "${DTFE_EXACT_AVERAGE}" = "1" ] || { echo "Error: FACE_GRADIENT=1 needs the exact deposit: add -e (DTFE_EXACT_AVERAGE=1)." >&2; exit 2; }
    [ "${PS_VOLUME_WEIGHTED}" = "1" ] || { echo "Error: FACE_GRADIENT=1 needs the volume-weighted moments (PS_VOLUME_WEIGHTED=1)." >&2; exit 2; }
    # (since 2026-11-09 the face sums also come beside the GPU deposit, from a CPU walk: -m / DTFE_GPU=1 is allowed; D17)
fi

n_ok=0; n_skipped=0; n_failed=0; n_planned=0   # summarised at the end; see the exit-status note there

time_wrap_resolve                                     # TIME_WRAP: the optional peak-RSS wrapper (config.sh)
[ -n "$THREADS" ] && export OMP_NUM_THREADS="$THREADS"
keep_awake "${DRY_RUN}"
# the run-log contract (config.sh), the same in run_dtfe.sh and the launcher's own runs: '<prefix>.runlog.partial'
# while the binary runs, then '<prefix>.runlog' on exit 0 or '<prefix>.runlog.failed' on a failure or a stop; a
# stop also releases the lock and re-raises the signal, so a caller (run_ps_pipeline.sh) sees a stop
runlog_contract_start

echo "Starting PS-DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}   Partition: [${PARTITION:-auto}]   Fields: ${FIELDS}"
echo "Deposit: $([ "${DTFE_GPU}" = "1" ] && echo "GPU (--gpu)" || echo "CPU")${gpu_auto}   volume-weighted: ${PS_VOLUME_WEIGHTED}   caustics: ${PS_CAUSTICS}   face gradient: ${FACE_GRADIENT}   prefix: ${OUTPUT_PREFIX}"
echo ""

for i in "${SNAPSHOTS[@]}"; do
    if ! snapshot_start "$i"; then          # n_str, input_dir, input_file, output_root (config.sh)
        n_skipped=$((n_skipped+1))
        continue
    fi

    # GPU deposit toggle (DTFE_GPU=1 ./run_ps_dtfe.sh); ignored with a warning on non-GPU builds.
    metal_args=()
    [ "${DTFE_GPU}" = "1" ] && metal_args=(--gpu)

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

    # Exact conservative deposit toggle (-e / DTFE_EXACT_AVERAGE=1); CPU and GPU deposits.
    exact_args=()
    [ "${DTFE_EXACT_AVERAGE}" = "1" ] && exact_args=(--exact-average)

    # Chart-independent tet masses (PS_VERTEX_MASS=1, default -- see the comment at the top).
    vmass_args=()
    [ "${PS_VERTEX_MASS}" = "1" ] && vmass_args=(--ps-vertex-mass)

    # Volume-weighted velocity moments (PS_VOLUME_WEIGHTED=1); CPU and GPU deposits.
    vw_args=()
    [ "${PS_VOLUME_WEIGHTED}" = "1" ] && vw_args=(--ps-volume-weighted)
    [ "${FACE_GRADIENT}" = "1" ] && vw_args+=(--face-gradient)     # (FACE_GRADIENT, see the top)

    # Parallel Delaunay insertion (PARALLEL_TRIANGULATION=1, opt-in -- see the comment at the top).
    ptri_args=()
    [ "${PARALLEL_TRIANGULATION}" = "1" ] && ptri_args=(--parallel-triangulation)
    [ "${PS_LINEAR_DEPOSIT}" = "1" ] && ptri_args+=(--ps-linear-deposit)     # (PS_LINEAR_DEPOSIT, see the top)
    # (PS_LAGRANGIAN_PADDING, see the top; 0 passed too)
    [ -n "${PS_LAGRANGIAN_PADDING}" ] && ptri_args+=(--ps-lagrangian-padding "${PS_LAGRANGIAN_PADDING}")

    # Point evaluation on top of the grid run (SAMPLE_POINTS=<file>, see the comment at the top).
    sp_args=()
    [ -n "${SAMPLE_POINTS}" ] && sp_args=(--sample-points "${SAMPLE_POINTS}")
    [ -n "${SAMPLE_POINTS}" ] && [ "${PTS_VEL_GRAD}" = "1" ] && sp_args+=(--pts-vel-grad)
    [ -n "${SAMPLE_POINTS}" ] && [ "${PS_POINT_CONVENTIONS}" = "legacy" ] && sp_args+=(--ps-point-conventions legacy)

    # Lagrangian positions: the separate IC file when it exists, else the snapshot's own
    # 'InitialCoordinates'. Skip only when NEITHER is available.
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

    # TIME_WRAP (resolved above) captures peak memory, verifying MAX_CONCURRENT fits RAM.
    # The tee pipe hides the terminal from the binary's isatty() check, so force colours through
    # it (CLICOLOR_FORCE, honoured by message.h) -- but only when this script itself runs on a
    # terminal, keeping cron/CI output clean. The runlog is de-ANSI'd after the run.
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
    # a killed partitioned run left its checkpoint: resume it (config.sh resume_args_for)
    resume_args_for "${output_root}" DTFE_RESUME "${DTFE_RESUME}"

    cmd=("${PS_BIN}" "${input_file}" "${output_root}"
        --grid ${GRID_SIZE}
        --padding ${PADDING}
        --periodic
        ${part_args[@]+"${part_args[@]}"}
        --subsamples ${SUBSAMPLES}
        --input-format 105
        --mpc-unit ${MPC_UNIT}
        --field ${FIELDS}
        --lambda-th ${LAMBDA_TH}
        ${metal_args[@]+"${metal_args[@]}"}
        ${exact_args[@]+"${exact_args[@]}"}
        ${vmass_args[@]+"${vmass_args[@]}"}
        ${ptri_args[@]+"${ptri_args[@]}"}
        ${vw_args[@]+"${vw_args[@]}"}
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
        # the budget the binary's auto-tuner planned against (60% of RAM, less what other programs
        # use; see src/auto_tune.h) -- absent when --partition and --max-concurrent were both given
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
