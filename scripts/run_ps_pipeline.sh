#!/usr/bin/env bash
# COMPUTE ONLY: for every snapshot of every simulation in SIMS, run run_ps_dtfe.sh and point-evaluate a full-box
# image plane of the phase-space fields in the same run (one triangulation). No plotting: render the figures with
#     python3 python/plot/plot_pointeval.py --sims all
#
#   SIMS=all ./run_ps_pipeline.sh                   # every simulation and snapshot on disk
#   SIMS="TNG100-3-Dark" ./run_ps_pipeline.sh       # one simulation
#   DRY_RUN=1 ./run_ps_pipeline.sh                  # print the plan, run nothing
#   ./run_ps_pipeline.sh --help                     # this header, nothing run (--version: the revision)
#
# Resumable: a snapshot is skipped when its outputs are newer than its inputs and its run log shows the same settings
# (not the device, run mode, GPU share, split or build); FORCE=1 recomputes everything. The plane file is made once
# per simulation and remade, making every snapshot stale, when its geometry changes. The .pts_* files stay on disk:
# they are the input of every figure.
# Knobs (environment; the README's pipeline table): DTFE_DATA_ROOT and SIMS (both required), SNAPS, NU, AXIS, CENTER,
# PLANES, THICKNESS, SUPERSAMPLE, PLANE_WINDOW, PTS_VEL_GRAD, FORCE, PYTHON; the others (GRID_SIZE, FIELDS, DTFE_GPU,
# DTFE_EXACT_AVERAGE, DTFE_RUN_MODE, ...) go on to run_ps_dtfe.sh.

set -uo pipefail   # no -e: a failed simulation is said and the batch goes on (config.sh, 'Strictness')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # python/ and the binaries live one level up
# -h / --help print this header and run nothing; --version the revision; anything else is an error (the knobs are
# environment variables)
case "${1:-}" in
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'; exit 0 ;;
    --version) git -C "${REPO_ROOT}" describe --always --dirty 2>/dev/null || echo unknown; exit 0 ;;
    "") ;;
    *) echo "Error: run_ps_pipeline.sh takes no arguments (its knobs are environment variables; --help lists them)" >&2; exit 2 ;;
esac
source "${SCRIPT_DIR}/config.sh"    # (CALLER_GRID_SIZE, the caller's GRID_SIZE, is read below)
require_data_root                    # before SIMS=all looks for simulations under it

PYTHON="${PYTHON:-$(h5py_python)}"   # the launcher passes its own (config.sh h5py_python otherwise)
NU="${NU:-8192}"
AXIS="${AXIS:-z}"
CENTER="${CENTER:-}"
GRID_SIZE="${CALLER_GRID_SIZE:-512}"   # 512^3 analysis grids by default (figures don't depend on it)
PTS_VEL_GRAD="${PTS_VEL_GRAD:-1}"
PLANES="${PLANES:-1}"
THICKNESS="${THICKNESS:-2}"
SUPERSAMPLE="${SUPERSAMPLE:-1}"
PLANE_WINDOW="${PLANE_WINDOW:-}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFIX="${OUTPUT_PREFIX:-ps_output}"
knob_bool PTS_VEL_GRAD FORCE DRY_RUN   # 1 or 0 (config.sh)
# PS_MERGE's value, refused here before anything runs (a dry run too), in run_ps_dtfe.sh's words
case "${PS_MERGE:-}" in
    ''|index-order|completion-order) ;;
    *) echo "Error: PS_MERGE must be 'index-order' or 'completion-order' (got '${PS_MERGE}')." >&2; exit 2 ;;
esac

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
# The deposit run_ps_dtfe.sh will run, by its own rule (config.sh average_resolve): DTFE_EXACT_AVERAGE when given,
# else the sampled one when SUBSAMPLES is set, else the exact one. Handed on as DTFE_EXACT_AVERAGE (env_args below).
_caller_exact="${DTFE_EXACT_AVERAGE:-}"
average_resolve

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
echo "Slice: axis ${AXIS}, ${NU} px, centre ${CENTER:-box/2}, planes ${PLANES}$([ "${PLANES}" != "1" ] && echo " over ${THICKNESS} Mpc"), ${SUPERSAMPLE}x${SUPERSAMPLE} sub-samples${PLANE_WINDOW:+, window ${PLANE_WINDOW}}   Grid: ${GRID_SIZE}^3   GPU: $(device_given)   deposit: $([ "${DTFE_EXACT_AVERAGE}" = "1" ] && echo exact || echo sampled) (${average_src})   vel-grad: ${PTS_VEL_GRAD}   force: ${FORCE}"
echo "Run mode: $(run_mode_given)"   # (GPU: and Run mode: as given, config.sh device_given; each run log says what ran)

run() { echo "+ $*"; [ "${DRY_RUN}" = "1" ] || "$@"; }

# run_settings <command line>: the options of a run_ps_dtfe.sh command that change its grids, one per line, sorted,
# from the RUNNING: line of a run log or a DRY_RUN=1 'would run' line. Not compared: merge mode, '--run-mode',
# '--gpu-share', the velocity gradient's route. '--exact' is compared, '--sampled' is not (a log without '--exact'
# is read as a sampled run's); '--subsamples' only for the sampled deposit (no count reads as 1; a log's header
# count, the last one read, is the one compared).
run_settings() {
    printf '%s\n' "$1" | awk '
        function out(l) { if (l ~ /^--subsamples( |$)/) subs = l; else print l }
        BEGIN { split("--grid --padding --subsamples --field --lambda-th --scalar-dataset " \
                      "--ps-lagrangian-padding --ps-alpha-shape --ps-halo-release", v); for (i in v) val[v[i]] = 1
                split("--periodic --ps-vertex-mass --exact " \
                      "--ps-caustics --ps-caustic-cusps --pts-vel-grad --face-gradient --per-tet-gradient", b)
                for (i in b) flag[b[i]] = 1 }
        { i = 1; if ($1 == "RUNNING:") i = 2
          bin = $i; sub(/.*\//, "", bin); print "binary " bin
          cur = ""; line = ""; subs = ""; exact = 0
          for (i = i + 1; i <= NF; i++) {
              t = $i
              if (t ~ /^--/) {
                  if (line != "") out(line)
                  line = ""; cur = ""
                  if (t in val) { cur = t; line = t }
                  else if (t in flag) { print t; if (t == "--exact") exact = 1 }
              } else if (cur != "") line = line " " t
          }
          if (line != "") out(line)
          if (!exact) { if (subs == "") subs = "--subsamples 1"; print subs } }' | sort
}

# older_log <run log>: the log has no header 'run mode' line ('\t run mode               : balanced (...)',
# src/user_options.cc); its command may spell options in names run_settings does not read. The launcher's
# Pipeline tab predicts with the same test: python/gui/results.py, older_log.
older_log() {
    local pat='^[[:blank:]]*run mode[[:blank:]]+:[[:blank:]]+(gentle|balanced|performance)([^[:alnum:]_]|$)'
    ! grep -a -q -E -e "${pat}" "$1"
}

# same_settings <have> <want> (run_settings outputs): equal, except that grids on disk may carry MORE fields than
# asked for (a rerun with a shorter FIELDS list must not recompute them); a want that names a dataset
# (SCALAR_DATASET) needs grids made with the same one, a want that names none is content with either
same_settings() {
    local skip='^--field |^--scalar-dataset '
    [ "$(printf '%s\n' "$1" | grep -v -E "${skip}")" = "$(printf '%s\n' "$2" | grep -v -E "${skip}")" ] || return 1
    local want_ds have_ds
    want_ds="$(printf '%s\n' "$2" | grep '^--scalar-dataset ' || true)"
    have_ds="$(printf '%s\n' "$1" | grep '^--scalar-dataset ' || true)"
    [ -z "${want_ds}" ] || [ "${want_ds}" = "${have_ds}" ] || return 1
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
        # a stored geometry differing from the requested one must regenerate the plane (and so invalidate every slice).
        # Exit 3 = differs (or an unreadable sidecar); any other failure of the tool keeps the plane and skips the sim.
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
    # has no 'supersample' entry), else the plane file's own count (3 float64 per point). File times alone
    # would take a slice of another plane size for current.
    plane_pts=""
    if [ "${plane_new}" != "1" ]; then
        plane_pts="$("${PYTHON}" -c 'import json, sys
s = json.load(open(sys.argv[1]))
print(int(s.get("planes", 1)) * int(s["nv"]) * int(s["nu"]) * int(s.get("supersample", 1)) ** 2)' "${plane}.json" 2>/dev/null)" \
            || plane_pts=""
        [ -n "${plane_pts}" ] || plane_pts=$(( $(wc -c < "${plane}.bin") / 24 ))
    fi

    # ---- grids + point evaluation, one triangulation per snapshot ------------------------
    # run_ps_dtfe.sh's deposit (DTFE_EXACT_AVERAGE) is the one resolved above; the other options reach it through the environment
    env_args=(SAMPLE_POINTS="${plane}.bin" PTS_VEL_GRAD="${PTS_VEL_GRAD}" DTFE_EXACT_AVERAGE="${DTFE_EXACT_AVERAGE}")
    [ -n "${FIELDS:-}" ] && env_args+=(FIELDS="${FIELDS}")
    # The settings this batch would run each snapshot with, asked from run_ps_dtfe.sh itself (DRY_RUN=1: reads,
    # runs and writes nothing), so the comparison below cannot drift from what it runs.
    planned=""
    if [ "${FORCE}" != "1" ] && [ "${plane_new}" != "1" ]; then
        planned="$(env "${env_args[@]}" DRY_RUN=1 "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" \
                       "${snaps[@]}" 2>/dev/null \
                   | awk '/^Processing snapshot / { s = $3; sub(/\.+$/, "", s) }
                          /DRY RUN -- would run:/ { want = 1; next }
                          want { sub(/^ +/, ""); print s "\t" $0; want = 0 }')"
    fi
    # resume: only snapshots whose slice is missing, older than its inputs, made by an older version of the programs
    # (a run log without the header's 'run mode' line) or made with other settings are computed
    compute=()
    for n in "${snaps[@]}"; do
        sd="${simdir}/snapdir_$(printf "%03d" "$n")"
        cf="${sd}/combined_$(printf "%03d" "$n").hdf5"
        # with PTS_VEL_GRAD on, a snapshot lacking .pts_velGrad is stale even if its .pts_den is current
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
            elif older_log "${rl}"; then
                echo "-- snap ${n}: ${PREFIX}.runlog was written by an older version of the programs, recomputing"
                compute+=("$n")
            elif [ -z "${want}" ]; then
                echo "-- snap ${n}: ${PREFIX}.pts_den up to date by file times (this run's settings could not be worked out: run_ps_dtfe.sh's dry run failed), skipping compute"
            else
                # (the log's own count line after its command: run_settings keeps the last count it reads)
                have_s="$(run_settings "$(grep -a -m1 '^RUNNING:' "${rl}") $(grep -a -m1 -o 'avg sub-samples (nSub) : [0-9]*' "${rl}" | sed 's/.*: /--subsamples /')")"
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
            "${SCRIPT_DIR}/run_ps_dtfe.sh" -s "${sim}" -g "${GRID_SIZE}" "${compute[@]}" \
            || { echo "!! ${sim}: run_ps_dtfe.sh reported failure (continuing with the next simulation)"; overall_rc=1; }
    else
        echo "-- ${sim}: all snapshots up to date"
    fi

done

echo ""
# Finding NOTHING is a failure, as in run_ps_dtfe.sh (an unmounted drive, a mistyped SIMS, a SNAPS matching no
# snapshot): ALL DONE with exit 0 would look like success in the launcher's Pipeline tab. A skip stays exit 0.
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
