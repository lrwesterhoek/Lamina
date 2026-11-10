#!/usr/bin/env bash
# COMPUTE ONLY: for every snapshot of every simulation, regenerate the analysis grids
# (the production physics: --ps-vertex-mass + per-field --ps-volume-weighted, both defaults) AND point-evaluate a
# full-box high-resolution image plane of the continuous phase-space field. Both share the
# snapshot's single triangulation (run_ps_dtfe.sh SAMPLE_POINTS), so the image planes cost
# only the point evaluation on top of the grid regen.
#
# This script does no plotting. Render the figures afterwards with:
#     python3 python/plot/plot_pointeval.py --sims all # all sims/snapshots, one file per field
# (the .pts_* files stay on disk, so figures can be restyled any time without recomputing).
#
#   SIMS=all ./run_ps_pipeline.sh                   # everything: all sims, all snapshots on disk
#   SIMS="TNG100-3-Dark" ./run_ps_pipeline.sh       # one simulation
#   DRY_RUN=1 ./run_ps_pipeline.sh                  # print the plan, run nothing
#   ./run_ps_pipeline.sh --help                     # this header, nothing run (--version: the revision)
#
# RESUMABLE: a snapshot is skipped when its <prefix>.pts_den exists and is newer than both
# its combined_*.hdf5 and the plane file (and, with PTS_VEL_GRAD=1, its .pts_velGrad exists),
# AND the settings in its <prefix>.runlog match this run's (grid, sub-samples, padding, lambda-th,
# precision, the deposit options, and every requested field among its fields; not the GPU/CPU
# choice, the run mode, the GPU share, the split or the build).
# A snapshot without a run log is judged by the file times alone (said per snapshot).
# Interrupted batches continue where they stopped. FORCE=1 recomputes everything.
#
# Env knobs:
#   DTFE_DATA_ROOT  the folder that holds the simulations -- REQUIRED, no default
#   SIMS          simulations to process -- REQUIRED, no default: names, or 'all' for every simulation under
#                 $DTFE_DATA_ROOT with snapshots, in either layout -- flat <sim>/ or per-family <family>/<sim>/)
#   The rest are optional:
#   NU            image-plane pixels across the box (default 8192; ~13 kpc/px at TNG100-3)
#   AXIS, CENTER  plane orientation/position (default z, box centre). Changing CENTER with an
#                 existing plane file regenerates the plane -- which makes every snapshot's
#                 image plane stale, so the batch recomputes (as it must: geometry changed).
#   GRID_SIZE     analysis grid passed to run_ps_dtfe.sh (default 512)
#   FIELDS        forwarded to run_ps_dtfe.sh (default: its own, density velocity divergence shear)
#   SNAPS         only these snapshot numbers, e.g. "50 67 99" (default: every snapshot on disk);
#                 the GUI's adaptive plane size runs the pipeline once per plane size with it
#   FORCE         1 = ignore all freshness checks (default 0)
#   PTS_VEL_GRAD  1 (default) = also evaluate the velocity gradient at every sample point
#                 (.pts_velGrad), which is what makes the velDiv/velShear/velVort maps
#                 possible. A snapshot lacking it counts as STALE, so a rerun recomputes it.
#   PS_POINT_CONVENTIONS  grid (default) | legacy: the image plane's estimators (run_ps_dtfe.sh's knob, the binary's
#                 --ps-point-conventions): 'grid' = the grids' vertex masses and volume weighting at the points,
#                 'legacy' = rho_bar*V_lag masses and density-weighted stream means, which reproduce a plane made
#                 with them. A snapshot whose run log shows the other one is recomputed.
#   PLANES, THICKNESS, SUPERSAMPLE, PLANE_WINDOW
#                 the plane's geometry, forwarded to make_image_plane.py: PLANES sampling planes
#                 (default 1 = one crisp cross-section; a handful ghosts, use >= 16 with
#                 plot_pointeval.py --project slab) across a slab of THICKNESS Mpc (default 2), KxK sub-samples per pixel
#                 averaged into pixel-area means (default 1), and "u0 u1 v0 v1" in Mpc to image
#                 only that part of the plane (default: the whole box).
#   DTFE_GPU      1 = the deposit on the GPU (run_ps_dtfe.sh -m), 0 = the CPU; unset (auto) = the GPU when the
#                 PS binary was built with one, as run_ps_dtfe.sh decides
#                 (config.sh gpu_auto_resolve: PS-DTFE, or PS-DTFE-double with DTFE_PRECISION=double),
#                 for both deposits and in every run mode
#   DTFE_EXACT_AVERAGE  1 (default) = the exact deposit (--exact), 0 = the sampled one (--sampled); unset, a
#                 SUBSAMPLES count selects the sampled one (run_ps_dtfe.sh's rule, config.sh average_resolve).
#                 A run log of an older sampled run names no deposit flag: its grids are stale for the exact
#                 default and current for a sampled pipeline run (the same count, or none: the binary's 3).
#   DTFE_RUN_MODE gentle | balanced | performance | auto (default): how hard each run may lean on the machine
#                 (run_ps_dtfe.sh's --run-mode): auto = balanced, or performance inside a SLURM or PBS job
#                 (config.sh run_mode_resolve); resolved once here and handed on. It never makes grids stale.
#   PS_VOLUME_WEIGHTED  1 (default) = volume-weighted velocity moments, the production convention
#   PYTHON        the python that makes the plane (default: config.sh h5py_python, the first with numpy and h5py)
# A switch is off for 0, false, no or off and on for any other value (the binaries' rule). Older names, read for
# one release with a warning (the new name wins when both are set): PS_GPU and PS_METAL (DTFE_GPU), PS_EXACT
# (DTFE_EXACT_AVERAGE), PTS_CONVENTIONS (PS_POINT_CONVENTIONS), WINDOW (PLANE_WINDOW), PY (PYTHON), DTFE_GRID_SIZE.
# The other run options reach run_ps_dtfe.sh through the environment exactly as from a terminal:
# SUBSAMPLES, PS_VERTEX_MASS, PS_CAUSTICS, PS_CAUSTIC_CUSPS, PARALLEL_TRIANGULATION,
# PARTITION, MAX_CONCURRENT, SCRATCH_DIR, TESSELLATION_CACHE, DTFE_PRECISION, OUTPUT_PREFIX, LAMBDA_TH,
# THREADS, PS_MERGE, DTFE_RESUME, GPU_SHARE (see its header for each, and the older names it still reads). The GUI's
# Pipeline tab sets all of them.
# Snapshots are auto-discovered (snapdir_*/combined_NNN.hdf5). The plane file is generated
# once per simulation (box size is constant across its snapshots) and reused -- as long as its
# sidecar has the requested geometry (make_image_plane.py --same-as, exit 3 = differs); otherwise it
# is regenerated, which makes every snapshot of that simulation stale (the geometry changed, so it
# must). A comparison that FAILS (no python, a bad PLANE_WINDOW, a missing module) is not 'differs': the
# plane is kept, the simulation skipped with a '!!' line, and the script exits 1.
#
# DISK: the per-point files are KEPT (~9 GB per snapshot at NU=8192) -- they are the input
# to every figure. Delete <snapdir>/<prefix>.pts_vel* by hand once the figures are final.

set -uo pipefail   # no -e: a failed simulation is said and the batch goes on (config.sh, 'Strictness')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # python/ and the binaries live one level up
# -h / --help print this header and run nothing; --version the revision; anything else is an error (rather than
# starting the whole batch) -- the knobs are environment variables
case "${1:-}" in
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'; exit 0 ;;
    --version) git -C "${REPO_ROOT}" describe --always --dirty 2>/dev/null || echo unknown; exit 0 ;;
    "") ;;
    *) echo "Error: run_ps_pipeline.sh takes no arguments (its knobs are environment variables; --help lists them)" >&2; exit 2 ;;
esac
source "${SCRIPT_DIR}/config.sh"    # (GRID_SIZE, and DTFE_GRID_SIZE, its older name, through config.sh's knob)
require_data_root                    # before SIMS=all looks for simulations under it

knob PYTHON PY                       # (config.sh knob: the older name read for one release, with a warning)
PYTHON="${PYTHON:-$(h5py_python)}"   # the launcher passes its own (config.sh h5py_python otherwise)
NU="${NU:-8192}"
AXIS="${AXIS:-z}"
CENTER="${CENTER:-}"
GRID_SIZE="${CALLER_GRID_SIZE:-512}"   # 512^3 analysis grids by default (figures don't depend on it)
PTS_VEL_GRAD="${PTS_VEL_GRAD:-1}"
knob PS_POINT_CONVENTIONS PTS_CONVENTIONS
PS_POINT_CONVENTIONS="${PS_POINT_CONVENTIONS:-grid}"
PLANES="${PLANES:-1}"
THICKNESS="${THICKNESS:-2}"
SUPERSAMPLE="${SUPERSAMPLE:-1}"
knob PLANE_WINDOW WINDOW
PLANE_WINDOW="${PLANE_WINDOW:-}"
knob DTFE_GPU PS_GPU PS_METAL
DTFE_GPU="${DTFE_GPU:-auto}"         # auto: resolved below, as run_ps_dtfe.sh resolves its DTFE_GPU
_caller_run_mode="${DTFE_RUN_MODE:-}"   # (the caller's own value: run_mode_resolve, below)
DTFE_RUN_MODE="${DTFE_RUN_MODE:-auto}"  # auto: resolved below, as run_ps_dtfe.sh resolves its DTFE_RUN_MODE
knob DTFE_EXACT_AVERAGE PS_EXACT     # (run_ps_dtfe.sh's own knob: resolved below by its rule and handed on)
# run_ps_dtfe.sh's other knobs with older names, resolved here once (exported under the new name when the old one
# was), so the driver does not warn again for every simulation
knob SUBSAMPLES AVG_SUBSAMPLES
knob_merge PS_MERGE ORDERED_MERGE FIXED_POINT_MERGE
knob DTFE_RESUME PS_RESUME
knob TESSELLATION_CACHE TESS_CACHE
knob PARALLEL_TRIANGULATION PS_PARALLEL_TRI
knob PS_LAGRANGIAN_PADDING PS_LAG_PADDING
knob -e PS_LAGRANGIAN_INPUT LAGRANGIAN_INPUT
PS_VOLUME_WEIGHTED="${PS_VOLUME_WEIGHTED:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFIX="${OUTPUT_PREFIX:-ps_output}"
knob_bool PTS_VEL_GRAD PS_VOLUME_WEIGHTED FORCE DRY_RUN   # 1 or 0 (config.sh); DTFE_GPU: knob_gpu below
knob_gpu DTFE_GPU

# the plane's geometry, as make_image_plane.py takes it (shared by the generation and the --same-as check)
plane_args=(--axis "${AXIS}" --nu "${NU}" --planes "${PLANES}" --thickness "${THICKNESS}" --supersample "${SUPERSAMPLE}")
[ -n "${CENTER}" ] && plane_args+=(--center "${CENTER}")
if [ -n "${PLANE_WINDOW}" ]; then
    read -r w_u0 w_u1 w_v0 w_v1 w_more <<< "${PLANE_WINDOW}"
    if [ -z "${w_v1}" ] || [ -n "${w_more:-}" ]; then
        echo "!! PLANE_WINDOW must be four numbers 'u0 u1 v0 v1' (Mpc), got '${PLANE_WINDOW}'" >&2; exit 2
    fi
    plane_args+=(--plane-window "${w_u0}" "${w_u1}" "${w_v0}" "${w_v1}")
fi
# The deposit run_ps_dtfe.sh will run, by its own rule (config.sh average_resolve): DTFE_EXACT_AVERAGE when given
# (true is 1 there too), else the sampled one when SUBSAMPLES is set, else the exact one, the default. Handed on as
# DTFE_EXACT_AVERAGE (env_args below), so the GPU choice here and the deposit there cannot disagree.
_caller_exact="${DTFE_EXACT_AVERAGE:-}"
average_resolve
# The run mode, by run_ps_dtfe.sh's rule too (config.sh run_mode_resolve): DTFE_RUN_MODE when given, else balanced, or
# performance inside a SLURM or PBS job. Resolved once and handed on as DTFE_RUN_MODE (env_args below).
run_mode_resolve
# DTFE_GPU=auto (the default): the drivers' device for the deposit and the run mode (config.sh gpu_auto_resolve, the
# rule run_ps_dtfe.sh applies; DTFE_PRECISION picks the binary there too): the GPU when the binary run_ps_dtfe.sh will
# run was built with one, so a CPU-only build is not asked for the GPU (and a fallback warning) on every snapshot.
case "${DTFE_PRECISION:-single}" in double) ps_bin="PS-DTFE-double" ;; *) ps_bin="PS-DTFE" ;; esac
gpu_auto_resolve DTFE_GPU "${ps_bin}" ps "${DTFE_EXACT_AVERAGE}" "${DTFE_RUN_MODE}"
gpu_args=(); [ "${DTFE_GPU}" = "1" ] && gpu_args=(-m)

if [ -z "${SIMS:-}" ]; then
    echo "Error: no simulation given: set SIMS to the simulations to process (e.g. SIMS=\"TNG100-3-Dark\"), or SIMS=all" \
         "for every simulation under ${DATA_ROOT}." >&2
    exit 2
fi
if [ "${SIMS}" = "all" ]; then
    SIMS=""
    for d in "${DATA_ROOT}"/*/ "${DATA_ROOT}"/*/*/; do
        compgen -G "${d}snapdir_*/combined_[0-9][0-9][0-9].hdf5" > /dev/null || continue
        case " ${SIMS} " in *" $(basename "$d") "*) ;; *) SIMS="${SIMS} $(basename "$d")" ;; esac
    done
fi
echo "Simulations:${SIMS}"
echo "Slice: axis ${AXIS}, ${NU} px, centre ${CENTER:-box/2}, planes ${PLANES}$([ "${PLANES}" != "1" ] && echo " over ${THICKNESS} Mpc"), ${SUPERSAMPLE}x${SUPERSAMPLE} sub-samples${PLANE_WINDOW:+, window ${PLANE_WINDOW}}   Grid: ${GRID_SIZE}^3   GPU: ${DTFE_GPU}${gpu_auto}   deposit: $([ "${DTFE_EXACT_AVERAGE}" = "1" ] && echo exact || echo sampled) (${average_src})   vel-grad: ${PTS_VEL_GRAD}   force: ${FORCE}"
echo "Run mode: ${DTFE_RUN_MODE} (${run_mode_src})"

run() { echo "+ $*"; [ "${DRY_RUN}" = "1" ] || "$@"; }

# run_settings <command line>: the options of a run_ps_dtfe.sh command that change its grids, one per line,
# sorted -- from the RUNNING: line a run log starts with, or from a DRY_RUN=1 'would run' line (its %q
# quoting only touches the paths, which are not compared). The binary's name carries the precision. Each option
# is read in its canonical spelling first: old[] (renamed options) and sw[] (the merge switches, one --ps-merge
# mode) are python/dtfelib/runlog.py's permanent table (RENAMES, SWITCHES; tests/docs_drift_check.py keeps the two
# equal), so a log written with '--ps-exact-deposit --lambda_th 0.3 --avg-subsamples 3' has the settings of
# '--exact --lambda-th 0.3 --subsamples 3', and a driver that emits the new names does not find every older
# snapshot stale. val[] / flag[] name the canonical options compared (the merge mode is not: it moves the last float
# bits only, and older logs carry no merge flag; nor are '--run-mode' and '--gpu-share', for the same reasons). The
# deposit: '--exact' is compared (older logs: '--exact-average' or '--ps-exact-deposit'), '--sampled' is not (an
# older sampled run named no flag), and the '--subsamples' count only for the sampled deposit: the exact one ignores
# it (older drivers passed '--subsamples 3' beside it), and a sampled one without a count runs the binary's own 3
# (older drivers always named it). So an older exact log and an older sampled one each compare equal to the command
# the driver now emits for the same deposit.
run_settings() {
    printf '%s\n' "$1" | awk '
        function out(l) { if (l ~ /^--subsamples( |$)/) subs = l; else print l }
        BEGIN { split("--grid --padding --subsamples --field --lambda-th --ps-point-conventions " \
                      "--ps-lagrangian-padding --ps-alpha-shape --ps-halo-release", v); for (i in v) val[v[i]] = 1
                split("--periodic --ps-vertex-mass --ps-volume-weighted --exact --ps-linear-deposit " \
                      "--ps-caustics --ps-caustic-cusps --pts-vel-grad --face-gradient", b); for (i in b) flag[b[i]] = 1
                old["--ps-gpu"] = "--gpu"; old["--exact-average"] = "--exact"; old["--ps-exact-deposit"] = "--exact"
                old["--ps-no-overlap"] = "--no-overlap"
                old["--full_help"] = "--full-help"; old["--lambda_th"] = "--lambda-th"; old["--MpcUnit"] = "--mpc-unit"
                old["--paddingMpc"] = "--padding-mpc"; old["--regionMpc"] = "--region-mpc"; old["--partNo"] = "--partition-index"
                old["--redshiftCone"] = "--redshift-cone"; old["--origin"] = "--cone-origin"; old["--NGP"] = "--ngp"
                old["--CIC"] = "--cic"; old["--TSC"] = "--tsc"; old["--PCS"] = "--pcs"; old["--SPH"] = "--sph"
                old["--Voronoi"] = "--voronoi"; old["--inputFile"] = "--input-file"; old["--outputFile"] = "--output-file"
                old["--input"] = "--input-format"; old["--output"] = "--output-format"; old["--method"] = "--average-method"
                old["--noTest"] = "--no-padding-test"; old["--density0"] = "--mean-density"
                old["--randomSample"] = "--particle-fraction"; old["--approxPSD"] = "--approx-phase-space-density"
                old["--extensive"] = "--scalars-extensive"; old["--options"] = "--reader-options"
                old["--avg-subsamples"] = "--subsamples"; old["--lagrangianInput"] = "--ps-lagrangian-input"
                old["--ps-lag-padding"] = "--ps-lagrangian-padding"; old["--per-stream"] = "--ps-per-stream"
                old["--per-stream-ids"] = "--ps-per-stream-ids"; old["--pts-lagrangian"] = "--ps-per-stream-lagrangian"
                old["--pts-conventions"] = "--ps-point-conventions"; old["--ps-window"] = "--grid-window"
                sw["--ordered-merge"] = "--ps-merge index-order"; sw["--completion-order-merge"] = "--ps-merge completion-order"
                sw["--fixed-point-merge"] = "--ps-merge fixed-point" }
        { i = 1; if ($1 == "RUNNING:") i = 2
          bin = $i; sub(/.*\//, "", bin); print "binary " bin
          cur = ""; line = ""; subs = ""; exact = 0
          for (i = i + 1; i <= NF; i++) {
              t = $i; sv = ""
              if (t ~ /^--/) {
                  if (t in old) t = old[t]
                  else if (t in sw) { split(sw[t], p, " "); t = p[1]; sv = p[2] }
                  if (line != "") out(line)
                  line = ""; cur = ""
                  if (t in val) { if (sv != "") line = t " " sv; else { cur = t; line = t } }
                  else if (t in flag) { print t; if (t == "--exact") exact = 1 }
              } else if (cur != "") line = line " " t
          }
          if (line != "") out(line)
          if (!exact) { if (subs == "") subs = "--subsamples 3"; print subs } }' | sort
}

# same_settings <have> <want> (run_settings outputs): equal, except that the grids on disk may carry MORE
# fields than asked for (grids made with a longer FIELDS list hold every field a rerun with a shorter one asks
# for: that rerun must not recompute them)
same_settings() {
    [ "$(printf '%s\n' "$1" | grep -v '^--field ')" = "$(printf '%s\n' "$2" | grep -v '^--field ')" ] || return 1
    local have=" $(printf '%s\n' "$1" | sed -n 's/^--field //p') " f
    for f in $(printf '%s\n' "$2" | sed -n 's/^--field //p'); do
        case "${have}" in *" ${f} "*) ;; *) return 1 ;; esac
    done
    return 0
}

# fresh <target> <prereq...>: target exists and is strictly newer than every prerequisite
# (a missing prerequisite counts as stale -- we cannot verify against what is not there).
fresh() {
    local t="$1" p; shift
    [ -e "$t" ] || return 1
    for p in "$@"; do { [ -e "$p" ] && [ "$t" -nt "$p" ]; } || return 1; done
    return 0
}

overall_rc=0
n_found=0          # snapshots found (after SNAPS) over all simulations; 0 at the end is an error
for sim in ${SIMS}; do
    simdir="$(sim_dir "${sim}")"

    # ---- discover snapshots (numbers, sorted) --------------------------------------------
    snaps=()
    for f in "${simdir}"/snapdir_*/combined_[0-9][0-9][0-9].hdf5; do
        [ -e "$f" ] || continue
        n=$(basename "$f"); n=${n#combined_}; n=${n%.hdf5}
        n=$((10#$n))
        if [ -n "${SNAPS:-}" ]; then
            case " ${SNAPS} " in *" ${n} "*|*" $(printf "%03d" "${n}") "*) ;; *) continue ;; esac
        fi
        snaps+=("${n}")
    done
    if [ ${#snaps[@]} -eq 0 ]; then echo "-- ${sim}: no combined snapshots${SNAPS:+ numbered ${SNAPS}}, skipping"; continue; fi
    n_found=$((n_found + ${#snaps[@]}))
    echo ""
    echo "== ${sim}: snapshots ${snaps[*]}"

    # ---- one plane file per simulation ---------------------------------------------------
    plane="${simdir}/hires_plane_${AXIS}${NU}"
    plane_new=0        # set when the plane is (re)generated: invalidates every slice/figure
    if [ -f "${plane}.json" ]; then
        # a stored geometry differing from the requested one (CENTER when given, planes, thickness,
        # sub-samples, window) must regenerate the plane -- and thereby invalidate every slice
        # computed against the old one. Exit 3 = differs (or an unreadable sidecar); any other failure
        # of the tool (no python, a bad PLANE_WINDOW, a missing module) keeps the plane and skips the sim.
        same_rc=0
        same_err="$("${PYTHON}" "${REPO_ROOT}/python/tools/make_image_plane.py" --same-as "${plane}.json" \
                    "${plane_args[@]}" 2>&1 >/dev/null)" || same_rc=$?
        if [ "${same_rc}" = "3" ]; then
            echo "-- ${sim}: the existing plane's geometry (centre, planes, thickness, sub-samples or window) differs; regenerating"
            run rm -f "${plane}.bin" "${plane}.json"
            plane_new=1
        elif [ "${same_rc}" != "0" ]; then
            echo "!! ${sim}: could not compare the plane's geometry (make_image_plane.py --same-as exited ${same_rc}); the plane is kept, the simulation skipped"
            [ -n "${same_err}" ] && echo "${same_err}" | sed 's/^/   /'
            overall_rc=1; continue
        fi
    fi
    if [ "${plane_new}" = "1" ] || [ ! -f "${plane}.bin" ] || [ ! -f "${plane}.json" ]; then
        first=$(printf "%03d" "${snaps[0]}")
        run "${PYTHON}" "${REPO_ROOT}/python/tools/make_image_plane.py" \
            --combined "${simdir}/snapdir_${first}/combined_${first}.hdf5" \
            "${plane_args[@]}" \
            -o "${plane}" || { echo "!! ${sim}: plane generation failed"; overall_rc=1; continue; }
        plane_new=1
    else
        echo "-- reusing ${plane}.bin"
    fi
    # The points a slice of THIS plane holds: planes x nv x nu x K^2 from its sidecar (K = 1 when the sidecar
    # has no 'supersample' entry), else the plane file's own count (3 float64 per point). The file times alone would take
    # an older 4096^2 slice for current against an even older hires_plane_z8192 (the default NU 8192), so the
    # compute would be skipped and every render fail on the size mismatch.
    plane_pts=""
    if [ "${plane_new}" != "1" ]; then
        plane_pts="$("${PYTHON}" -c 'import json, sys
s = json.load(open(sys.argv[1]))
print(int(s.get("planes", 1)) * int(s["nv"]) * int(s["nu"]) * int(s.get("supersample", 1)) ** 2)' "${plane}.json" 2>/dev/null)" \
            || plane_pts=""
        [ -n "${plane_pts}" ] || plane_pts=$(( $(wc -c < "${plane}.bin") / 24 ))
    fi

    # ---- grids + point evaluation, one triangulation per snapshot ------------------------
    # run_ps_dtfe.sh's DTFE_GPU follows this one: -m only switches the GPU ON, an exported DTFE_GPU=1 would otherwise
    # leak in (an older PS_GPU / PS_METAL was read above and unset: the driver sees the new name only); its deposit
    # (DTFE_EXACT_AVERAGE) and run mode (DTFE_RUN_MODE), resolved above, too, so the GPU choice was made for the
    # deposit and the mode it runs
    env_args=(SAMPLE_POINTS="${plane}.bin" PS_VOLUME_WEIGHTED="${PS_VOLUME_WEIGHTED}" PTS_VEL_GRAD="${PTS_VEL_GRAD}"
              PS_POINT_CONVENTIONS="${PS_POINT_CONVENTIONS}" DTFE_GPU="${DTFE_GPU}" DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE}"
              DTFE_RUN_MODE="${DTFE_RUN_MODE}")
    [ -n "${FIELDS:-}" ] && env_args+=(FIELDS="${FIELDS}")
    # The settings this batch would run each snapshot with, asked from run_ps_dtfe.sh itself (DRY_RUN=1: reads,
    # runs and writes nothing), so the comparison below cannot drift from what it runs (file times alone would
    # print 'up to date' for every snapshot after a changed grid or field list).
    planned=""
    if [ "${FORCE}" != "1" ] && [ "${plane_new}" != "1" ]; then
        planned="$(env "${env_args[@]}" DRY_RUN=1 "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" \
                       ${gpu_args[@]+"${gpu_args[@]}"} "${snaps[@]}" 2>/dev/null \
                   | awk '/^Processing snapshot / { s = $3; sub(/\.+$/, "", s) }
                          /DRY RUN -- would run:/ { want = 1; next }
                          want { sub(/^ +/, ""); print s "\t" $0; want = 0 }')"
    fi
    # resume: only snapshots whose slice is missing, older than its inputs, or made with other settings are computed
    compute=()
    for n in "${snaps[@]}"; do
        sd="${simdir}/snapdir_$(printf "%03d" "$n")"
        cf="${sd}/combined_$(printf "%03d" "$n").hdf5"
        # with PTS_VEL_GRAD on, a snapshot lacking .pts_velGrad is stale even if its
        # .pts_den is current (a run without the flag made it)
        vg_ok=1
        if [ "${PTS_VEL_GRAD}" = "1" ] && [ ! -f "${sd}/${PREFIX}.pts_velGrad" ]; then vg_ok=0; fi
        if [ "${FORCE}" != "1" ] && [ "${plane_new}" != "1" ] && [ "${vg_ok}" = "1" ] \
            && fresh "${sd}/${PREFIX}.pts_den" "${cf}" "${plane}.bin"; then
            # newer than the plane is not enough: it must be a slice OF this plane (see plane_pts)
            have_pts=$(( $(wc -c < "${sd}/${PREFIX}.pts_den") / 8 ))
            if [ "${have_pts}" != "${plane_pts}" ]; then
                echo "-- snap ${n}: ${PREFIX}.pts_den holds ${have_pts} points, the plane ${plane_pts}: a slice of another plane, recomputing"
                compute+=("$n")
                continue
            fi
            rl="${sd}/${PREFIX}.runlog"
            want="$(printf '%s\n' "${planned}" | awk -F '\t' -v s="$(printf "%03d" "$n")" '$1 == s { print $2; exit }')"
            if [ ! -f "${rl}" ]; then
                echo "-- snap ${n}: ${PREFIX}.pts_den up to date by file times (no ${PREFIX}.runlog to compare the settings with), skipping compute"
            elif [ -z "${want}" ]; then
                echo "-- snap ${n}: ${PREFIX}.pts_den up to date by file times (this run's settings could not be worked out: run_ps_dtfe.sh's dry run failed), skipping compute"
            else
                have_s="$(run_settings "$(grep -a -m1 '^RUNNING:' "${rl}")")"
                want_s="$(run_settings "${want}")"
                if same_settings "${have_s}" "${want_s}"; then
                    echo "-- snap ${n}: ${PREFIX}.pts_den up to date (same settings), skipping compute"
                else
                    echo "-- snap ${n}: settings differ from its ${PREFIX}.runlog, recomputing"
                    comm -23 <(printf '%s\n' "${have_s}") <(printf '%s\n' "${want_s}") | sed 's/^/     was: /'
                    comm -13 <(printf '%s\n' "${have_s}") <(printf '%s\n' "${want_s}") | sed 's/^/     now: /'
                    compute+=("$n")
                fi
            fi
        else
            compute+=("$n")
        fi
    done
    if [ ${#compute[@]} -gt 0 ]; then
        run env "${env_args[@]}" \
            "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" ${gpu_args[@]+"${gpu_args[@]}"} "${compute[@]}" \
            || { echo "!! ${sim}: run_ps_dtfe.sh reported failure (continuing with the next simulation)"; overall_rc=1; }
    else
        echo "-- ${sim}: all snapshots up to date"
    fi

done

echo ""
# Finding NOTHING is a failure, as in run_ps_dtfe.sh: with the data drive unmounted, a mistyped SIMS or a
# SNAPS matching no snapshot, nothing runs, and ALL DONE with exit 0 would show as a success in the
# launcher's Pipeline tab. Every snapshot found is either computed, up to
# date, or in a simulation that failed with a '!!' line (overall_rc 1), so a partial skip stays exit 0.
if [ "${n_found}" -eq 0 ]; then
    echo "!! nothing to compute: no combined snapshot${SNAPS:+ numbered ${SNAPS}} of $([ -n "${SIMS// }" ] && echo ${SIMS} || echo "any simulation") under '${DATA_ROOT}'"
    if [ ! -d "${DATA_ROOT}" ]; then
        echo "   That folder does not exist: is the data drive mounted?"
    else
        echo "   Check SIMS and SNAPS, and the data root (DTFE_DATA_ROOT)."
    fi
    overall_rc=1
fi
[ ${overall_rc} -eq 0 ] && echo "ALL DONE (compute). Render figures with: python3 python/plot/plot_pointeval.py" \
    || echo "DONE WITH FAILURES (see !! lines above)"
exit ${overall_rc}
