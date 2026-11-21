#!/usr/bin/env bash
# Contract check for the batch driver scripts/run_ps_dtfe.sh (and the shared scripts/config.sh, and
# run_dtfe.sh's exact / sampled cell average).
#
# Every other suite tests the BINARY; this one tests the shell around it, where failures stay silent
# (a discarded environment variable, or snapshots skipped for a file they do not need). Runs against a
# FAKE DATA_ROOT built here, so it never touches real data.
#
#  A) GRID_SIZE from the ENVIRONMENT reaches the binary, and '-g' still beats it:
#     precedence must be  -g flag > environment > config.sh default.
#  B) A snapshot carrying its own 'InitialCoordinates' is PROCESSED without a combined_ics.hdf5 --
#     PS-DTFE reads them when '--ps-lagrangian-input' is omitted, so skipping it drops usable data.
#  C) A combined_ics.hdf5, when present, is still passed as '--ps-lagrangian-input' (B must not have
#     moved everyone onto the snapshot path), and an empty PS_LAGRANGIAN_INPUT= forces it off. The binary's header
#     states the rule, and after the read it says that the file was not read (this snapshot has its own
#     'InitialCoordinates').
#  D) A snapshot with NEITHER source is skipped with a message naming both, and the run exits 1.
#  E) Exit status is honest: processing nothing exits 1, while a PARTIAL skip stays 0 -- a
#     partly-downloaded snapshot ladder is normal.
#  F) No macOS-only command on the critical path ('/usr/bin/time -l', 'sed -i ""'), which would
#     kill every snapshot on Linux. Asserted structurally, so it holds on either platform.
#  G) The PER-FAMILY data layout (<root>/<family>/<sim>, e.g. <root>/TNG100/TNG100-3-Dark), under a
#     root whose name has a space: the driver finds and processes the simulation there, config.sh's sim_dir puts a new simulation beside its
#     family and lets a flat <root>/<sim> win, and run_ps_pipeline.sh discovers nested sims -- and exits 1
#     when it finds nothing to compute (no data root, a mistyped SIMS, a SNAPS matching nothing). Its default call of
#     run_ps_dtfe.sh: the exact deposit, and no device or run mode (no -m, no DTFE_GPU or DTFE_RUN_MODE word).
#  H) scripts/run_dtfe.sh passes --exact to the standard binary by default (-e, --exact and DTFE_EXACT_AVERAGE=1 too);
#     SCALAR_DATASET grids a particle value as the scalar (run_dtfe.sh: a constant one comes back in every cell;
#     run_ps_dtfe.sh: its planned command carries it), and a scalar field without it, or an option inside FIELDS, is
#     refused before anything runs
#     (and an unknown run-mode word, '--run-mode fast', reaches the binary, which refuses it: the snapshot fails and
#     the driver exits 1)
#     and --sampled for DTFE_EXACT_AVERAGE=0 or --sampled, or a SUBSAMPLES count (-n) while the knob is unset (then
#     with --subsamples N); -e or DTFE_EXACT_AVERAGE=1 beside a count stays exact, with one note; -e and --sampled
#     together are refused. Its padding default is 15 (a caller's PADDING wins), run_ps_dtfe.sh keeps
#     config.sh's 50. run_dtfe.sh AUTO_TUNE_REPORT=1 (as
#     run_ps_dtfe.sh): the standard binary's own report line per snapshot ('snap=099'), exit 0, nothing written.
#  I) download_snapshots.sh + merge_HDF5.py never turn a half-finished download into a merged snapshot; a chunk the
#     data host refuses once is fetched on the retry ('attempt 2 of 3', exit 0), one refused every time is an error.
#     A merged snapshot carries the chunks' other per-particle fields, h-free, or the merge is refused; --add-fields
#     puts them into a file merged without them, which keeps its positions, velocities, IDs and time stamp; --verify
#     compares a combined file with its chunks and changes nothing (--delete-chunks, also beside --add-fields, then
#     only of what compares the same: a moved position or a changed value keeps the chunks, so does a particle type
#     the file lacks); the datasets are written in one piece.
#  J) --help runs nothing (run_ps_pipeline.sh --help must not start the whole batch), --version prints the
#     revision, an argument to the pipeline is an error, DRY_RUN=1 prints the drivers' commands and runs nothing.
#     There is no default data root or simulation: every driver without DTFE_DATA_ROOT (or without -s / DTFE_SIM /
#     SIMS) refuses with exit 2 and says so, -d or not, while --help still works and sourcing config.sh exits nothing.
#     The drivers' shared option loop (config.sh driver_options): a bad option exits 1 with the usage, -h anywhere 0;
#     the long forms are the letters, --run-mode MODE in both drivers (download_snapshots.sh refuses it).
#  K) the binaries' own help lists the options the scripts and the launcher use: the short help (--help, -h) the two
#     modes' options (--exact, --sampled, --gpu, --run-mode, --lambda-th, --mpc-unit, --auto-tune-report and --full-help in both;
#     --ps-merge and --ps-vertex-mass in PS-DTFE's), the full help (--full-help) the ones not in
#     -h (--subsamples, --no-overlap, --sample-points and --serve in both; --gpu-share, --grid-window,
#     --tessellation-cache, --serve-progress, --ps-per-stream and --ps-caustics in PS-DTFE's,
#     --average-method in standard DTFE's; --exact and --sampled in both), and the help texts carry none of the
#     known misspellings; standard DTFE's help names none of PS-DTFE's phase-space options. A --config file's value
#     gives way to the command line's for the same option, in both binaries: 'lambda-th = 5' in the file under
#     '--lambda-th 0.3' runs at 0.3, the file alone at 5.
#  L) scripts/install_lib.sh keeps a user's binaries through a failed rebuild: backup, restore (the 2D double pair
#     too), the failure hint; install.sh's detection of the extra sets it rebuilds (sourced alone, never run).
#  M) the run-log contract of run_dtfe.sh / run_ps_dtfe.sh, against stand-in binaries: a run's log is
#     <prefix>.runlog.partial while it goes, <prefix>.runlog after exit 0, <prefix>.runlog.failed after a
#     failure or a stop; a failed, stopped or killed re-run never touches the .runlog of the grids on disk.
#  N) the Makefile stamps BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS in the object dirs (a dry run, builds nothing).
#  O) the drivers' single-instance lock (<prefix>.runlog.lock: a second driver refused, stale locks removed, a stop
#     releases it) and their preflight (data disk mounted, output folder writable with room for the grids).
#  P) run_ps_pipeline.sh's freshness test compares the run settings with the snapshot's run log, not only file times
#     (a deposit option switched off, PS_CAUSTICS=0, counts too; the device, the run mode and the GPU share do not).
#     The deposit: an exact log ('--exact') matches the exact default, a sampled one (a count, no deposit flag)
#     matches '--sampled' with that count (none is the binary's 1, so a log's 3 is current for SUBSAMPLES=3 only) and
#     not the exact default; a sampled log that names no count is read by its header's count line. A log whose
#     command carries '--ps-volume-weighted', or '--gpu --run-mode balanced --gpu-share auto', none of them compared,
#     is current for the same settings; a log without the header's 'run mode' line (an older version of the programs)
#     is recomputed, its settings unread.
#  R) run_ps_dtfe.sh PS_MERGE: unset passes '--ps-merge index-order' (the driver's default, given explicitly so the
#     memory gate stays on under the auto-tuner's speed split), a given mode passes '--ps-merge MODE', another value
#     is refused ('fixed-point' too); SAMPLE_POINTS reaches the binary as '--sample-points'; a killed run's checkpoint
#     (<prefix>.runlog.checkpoint, or config.sh ps_checkpoint_dir's name under SCRATCH_DIR) makes the next run pass
#     --resume and say so; DTFE_RESUME=0, an auto-tune report and a checkpoint folder with no 'checkpoint' file in it
#     do not.
#  T) a driver switch reads 0/false/no/off (any case) as off and any other value as on, as the binaries do.
#  S) the production defaults and the rerun hygiene, against the stand-in binaries:
#     the device is the binary's: no driver passes '--gpu', DTFE_GPU reaches the binary it starts as the caller set it
#     (unset stays unset, a value is not normalised), -m (or --gpu) exports DTFE_GPU=1 and wins over the caller's
#     value; the banner names what the driver was given ('the GPU (-m)', 'DTFE_GPU=<value>' or 'the program's pick'),
#     on a run and an auto-tune report; the drivers' production flags by default (run_ps_dtfe.sh: caustics, the four
#     analysis fields, the exact deposit (--exact; a SUBSAMPLES count or DTFE_EXACT_AVERAGE=0: the sampled one);
#     run_dtfe.sh: no run mode, the exact average and the same fields; FACE_GRADIENT=1 passes --face-gradient,
#     run_ps_dtfe.sh refusing it with the sampled deposit; unset it passes neither route flag and 0 passes
#     --per-tet-gradient, in both drivers),
#     run_dtfe.sh's OUTPUT_PREFIX (a 'ps_*' one refused), and a rerun's left-over files: after exit 0 the files the run
#     did not write are MOVED to <prefix>.superseded/<stamp>/ with the old run log (the rewritten ones, the run log
#     family, a lock and a checkpoint stay); a failed run, a dry run and an auto-tune report move nothing.
#  U) the run mode and the GPU share, against the stand-in binaries: the drivers pass '--run-mode MODE' only when the
#     option named one (any word: the binary checks it; 'auto' passes nothing), DTFE_RUN_MODE reaches the binary as
#     the caller set it; the banner names what was given ('--run-mode MODE', 'DTFE_RUN_MODE=<value>' or 'the
#     program's default') and a kept OMP_NUM_THREADS. run_ps_dtfe.sh passes '--gpu-share GPU_SHARE' only when
#     GPU_SHARE is not auto, whatever the device and the deposit; run_dtfe.sh passes none. run_ps_pipeline.sh resolves
#     neither the device nor the run mode: the caller's DTFE_GPU, DTFE_RUN_MODE and GPU_SHARE
#     reach each run_ps_dtfe.sh run, and its binary, in the environment, and its header names what was given.
#
# Usage: tests/run_scripts_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the drivers run the double pair
precision_double && export DTFE_PRECISION=double     # the drivers' own switch (run_ps_dtfe.sh / run_dtfe.sh)
# tests/lib.sh pins the binaries' run mode and device for the suites (DTFE_RUN_MODE=performance, DTFE_GPU=0). This
# suite tests what the drivers hand the binaries (an unset DTFE_RUN_MODE and DTFE_GPU reach them unset: the binaries'
# default run mode and their own device pick), so it unsets both: a case that wants one sets it itself, and the
# binaries started directly in (K) get the pins back there
unset DTFE_RUN_MODE DTFE_GPU
unset SCALAR_DATASET                    # (a developer's export would add a particle field to every driver run here)

export PYTHON="${PY}"                   # (tests/lib.sh) run_ps_pipeline.sh runs make_image_plane.py with the same python

N="${N:-16}"
GRID="${GRID:-24}"
BOX="${BOX:-100.0}"
TMP="${TMP_DIR:-${DTFE_TEST_TMP}}/run_scripts"
export DTFE_FIGURES_ROOT="${TMP}/figures"   # the pipeline's plots: never the caller's figure root
DRIVER="scripts/run_ps_dtfe.sh"

echo "============================================================"
echo " run_ps_dtfe.sh contract check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} not built"; exit 1; }

fails=0
# check_rc <0 = pass> <text> (it counts a failure in fails), has01 <file> <pattern> (echoes 0 when the extended
# regex matches, else 1): tests/lib.sh. lacks <file> <pattern>: the opposite of has01
lacks() { grep -aqE -e "$2" "$1" && echo 1 || echo 0; }
# the drivers' banner words when they were given no device and no run mode (config.sh device_given, run_mode_given)
PICK="the program's pick"; MODE_DEFAULT="the program's default"

rm -rf "${TMP}"
mkdir -p "${TMP}/HASIC/snapdir_099" "${TMP}/NOIC/snapdir_099" "${TMP}/PARTIAL/snapdir_099"

echo ">> building the fake DATA_ROOT ..."
# HASIC: snapshot carries InitialCoordinates, no combined_ics.hdf5 beside it
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/HASIC/snapdir_099/combined_099.hdf5" \
        --n "${N}" --box "${BOX}" >/dev/null
# NOIC: same snapshot with InitialCoordinates removed, and still no combined_ics.hdf5
"${PY}" - "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/NOIC/snapdir_099/combined_099.hdf5" <<'PYEOF'
import shutil, sys
import h5py
shutil.copy(sys.argv[1], sys.argv[2])
with h5py.File(sys.argv[2], "a") as f:
    del f["PartType1"]["InitialCoordinates"]
PYEOF
# PARTIAL: one good snapshot; snapshot 098 is requested below but never created
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/PARTIAL/snapdir_099/combined_099.hdf5"

export DTFE_DATA_ROOT="${TMP}"
# keep every run tiny and quiet: one field, no averaging sub-samples worth mentioning (a count alone selects the
# sampled deposit: these runs pass '--sampled --subsamples 1')
COMMON_ENV=(FIELDS=density_a SUBSAMPLES=1)

# Runs the driver with the given env assignments, capturing stdout+stderr and the exit code.
# Echoes the exit code; the log path is "$2".
drive() {
    local log="$1"; shift
    local rc=0
    env "${COMMON_ENV[@]}" "$@" "${DRIVER}" 99 >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}

echo ""
echo "(A) GRID_SIZE from the environment reaches the binary, and -g still overrides it."
rc=$(drive "${TMP}/a_env.log" GRID_SIZE=24 DTFE_SIM=HASIC)
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "env-only run succeeds (exit ${rc})"
check_rc "$(has01 "${TMP}/a_env.log" '--grid 24( |$)')" "GRID_SIZE=24 reached the binary as '--grid 24'"
check_rc "$(has01 "${TMP}/a_env.log" 'Grid size: 24')" "GRID_SIZE=24 is what the script reports"
if precision_double; then
    check_rc "$(has01 "${TMP}/a_env.log" 'PS-DTFE-double ')" "DTFE_PRECISION=double: the driver ran ./PS-DTFE-double"
fi
# -g must win over the environment
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_SIM=HASIC "${DRIVER}" -g 32 99 >"${TMP}/a_flag.log" 2>&1 || true
check_rc "$(has01   "${TMP}/a_flag.log" '--grid 32( |$)')" "-g 32 beats GRID_SIZE=24 (flag > environment)"
check_rc "$(lacks "${TMP}/a_flag.log" '--grid 24( |$)')" \
      "the overridden GRID_SIZE=24 appears nowhere in the command"
# and with neither, config.sh's own default stands
default_grid=$(bash -c 'source scripts/config.sh; echo "${GRID_SIZE}"')
check_rc "$([ "${default_grid}" = "512" ] && echo 0 || echo 1)" \
      "config.sh's default is the production 512 when nothing overrides it (got ${default_grid})"

echo ""
echo "(B) a snapshot with its own 'InitialCoordinates' runs WITHOUT a combined_ics.hdf5."
check_rc "$(has01 "${TMP}/a_env.log" "Lagrangian positions: the snapshot's own 'InitialCoordinates'")" \
      "the script says it is using the snapshot's own Lagrangian positions"
check_rc "$(has01 "${TMP}/a_env.log" "read from 'InitialCoordinates' dataset in main input file")" \
      "the BINARY confirms it read them from the snapshot"
check_rc "$(lacks "${TMP}/a_env.log" '--(ps-lagrangian-input|lagrangianInput)')" "no --ps-lagrangian-input was passed"
check_rc "$([ -f "${TMP}/HASIC/snapdir_099/ps_output.a_den" ] && echo 0 || echo 1)" \
      "the density grid was actually written"

echo ""
echo "(C) a combined_ics.hdf5 beside the snapshot is still passed (the binary reads it only when the snapshot has no InitialCoordinates), and can be forced off."
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/HASIC/combined_ics.hdf5"
rc=$(drive "${TMP}/c_ics.log" GRID_SIZE=24 DTFE_SIM=HASIC)
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "run with an IC file succeeds (exit ${rc})"
check_rc "$(has01 "${TMP}/c_ics.log" '--ps-lagrangian-input .*combined_ics\.hdf5')" \
      "--ps-lagrangian-input points at the IC file"
check_rc "$(has01 "${TMP}/c_ics.log" "Lagrangian positions +: the snapshot's 'InitialCoordinates' when it has them, else '.*combined_ics\.hdf5' \(--ps-lagrangian-input, matched by ParticleIDs\)")" \
      "the BINARY's header states the rule: the snapshot's 'InitialCoordinates' when it has them, else the IC file"
check_rc "$(has01 "${TMP}/c_ics.log" "Lagrangian positions: the snapshot's 'InitialCoordinates'; '.*combined_ics\.hdf5' \(--ps-lagrangian-input\) was not read")" \
      "... and after the read the BINARY says it used the snapshot's 'InitialCoordinates' and did not read the IC file"
# explicitly empty PS_LAGRANGIAN_INPUT must fall back to the snapshot even though the file exists
rc=$(drive "${TMP}/c_forced.log" GRID_SIZE=24 DTFE_SIM=HASIC PS_LAGRANGIAN_INPUT=)
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "PS_LAGRANGIAN_INPUT= run succeeds (exit ${rc})"
check_rc "$(has01 "${TMP}/c_forced.log" "read from 'InitialCoordinates' dataset in main input file")" \
      "PS_LAGRANGIAN_INPUT= forces the snapshot path despite the IC file"
rm -f "${TMP}/HASIC/combined_ics.hdf5"

echo ""
echo "(D) a snapshot with NEITHER Lagrangian source is skipped, naming both, and exits 1."
rc=$(drive "${TMP}/d_none.log" GRID_SIZE=24 DTFE_SIM=NOIC)
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "exit status is 1 (got ${rc})"
check_rc "$(has01 "${TMP}/d_none.log" 'has NO Lagrangian positions')" "the warning names the real problem"
check_rc "$(has01 "${TMP}/d_none.log" "no 'InitialCoordinates' dataset either")" \
      "it says the snapshot itself was checked too"
check_rc "$(has01 "${TMP}/d_none.log" 'Skipping snapshot 099')" "the snapshot is skipped rather than failed"

echo ""
echo "(E) exit status is honest about doing nothing, and forgiving about partial ladders."
rc=$(drive "${TMP}/e_nothing.log" GRID_SIZE=24 DTFE_SIM=DOES_NOT_EXIST)
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a wrong DTFE_SIM exits 1, not 0 (got ${rc})"
check_rc "$(has01 "${TMP}/e_nothing.log" '0 processed')" "the summary reports 0 processed"
# 98 is absent, 99 is fine: a partly-downloaded ladder must still succeed
rc=0
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_SIM=PARTIAL "${DRIVER}" 98 99 >"${TMP}/e_partial.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a partial ladder still exits 0 (got ${rc})"
check_rc "$(has01 "${TMP}/e_partial.log" '1 processed, 1 skipped')" "the summary counts both outcomes"

echo ""
echo "(F) no macOS-only command sits on the critical path (so Linux/CI can run this driver)."
# The test is about the INVOCATION, not the mention: a guarded probe of '/usr/bin/time -l' is
# exactly how the portable wrapper decides which dialect exists, so grepping for the string alone
# would fail on the very code that fixes the problem. What must not exist is time(1) sitting
# directly in front of the binary.
check_rc "$(lacks "${DRIVER}" '/usr/bin/time[^|]*\./PS-DTFE')" \
      "time(1) does not prefix the binary unconditionally (GNU spells -l as -v; containers have neither)"
check_rc "$(has01   "${DRIVER}" 'TIME_WRAP\[@\].*"\${cmd\[@\]}"')" \
      "the binary is invoked through the resolved-once TIME_WRAP instead (PS_BIN: the pair chosen by DTFE_PRECISION)"
check_rc "$(lacks "${DRIVER}" "sed .*-i +''")" \
      "no BSD-only \"sed -i ''\" (GNU sed reads the '' as the script, not as the suffix)"
# and the run really did produce its log without a time(1) present
check_rc "$([ -f "${TMP}/HASIC/snapdir_099/ps_output.runlog" ] && echo 0 || echo 1)" \
      "the .runlog is written and de-ANSI'd whether or not time(1) exists"

echo ""
echo "(G) the per-family layout (<root>/<family>/<sim>) under a root with a space in its name."
FAMROOT="${TMP}/fam root"
mkdir -p "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099"
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/"
rc=0
# GRID_SIZE=24 like every other section: without it the config default 1024^3 (~14 GB of grids)
# would run here -- slow on a 64 GB Mac, and killed for memory in a 16 GB Linux container
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_DATA_ROOT="${FAMROOT}" DTFE_SIM=FAM-1-Dark "${DRIVER}" 99 \
    >"${TMP}/g_family.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a run on the per-family layout succeeds (exit ${rc})"
check_rc "$([ -f "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.a_den" ] && echo 0 || echo 1)" \
      "its grids land in <root>/FAM/FAM-1-Dark/snapdir_099"
# sim_dir: an unknown simulation of an existing family goes beside its siblings; a flat
# <root>/<sim> directory, when present, wins over the family folder
newsim="$(DTFE_DATA_ROOT="${FAMROOT}" bash -c 'source scripts/config.sh; sim_dir FAM-2-Dark')"
check_rc "$([ "${newsim}" = "${FAMROOT}/FAM/FAM-2-Dark" ] && echo 0 || echo 1)" \
      "sim_dir places a new FAM-2-Dark in its family folder (got '${newsim}')"
mkdir -p "${FAMROOT}/FAM-3-Dark"
flat="$(DTFE_DATA_ROOT="${FAMROOT}" bash -c 'source scripts/config.sh; sim_dir FAM-3-Dark')"
check_rc "$([ "${flat}" = "${FAMROOT}/FAM-3-Dark" ] && echo 0 || echo 1)" \
      "an existing flat <root>/FAM-3-Dark wins (got '${flat}')"
G_AVG=(-u DTFE_EXACT_AVERAGE -u SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID)   # the deposit's and the run mode's knobs: the defaults here
rc=0; env "${G_AVG[@]}" DRY_RUN=1 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline.log" 2>&1 || rc=$?
check_rc "$(has01 "${TMP}/g_pipeline.log" '^Simulations:.* FAM-1-Dark')" \
      "run_ps_pipeline.sh discovers the nested FAM-1-Dark"
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... and a dry run with a snapshot to plan exits 0 (${rc})"
# the pipeline's knobs: the plane's geometry reaches make_image_plane.py, and a plane whose sidecar has another
# geometry is regenerated. Its run_ps_dtfe.sh call hands the deposit on (the exact one by default, DTFE_EXACT_AVERAGE=1;
# a SUBSAMPLES count alone: the sampled one) and neither the device nor the run mode: no -m and no DTFE_GPU or
# DTFE_RUN_MODE word (both reach run_ps_dtfe.sh in the environment, as the caller set them: (U)); the header says so
env "${G_AVG[@]}" DRY_RUN=1 SUBSAMPLES=2 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline_s.log" 2>&1 || true
check_rc "$([ "$(has01 "${TMP}/g_pipeline.log" '^\+ env .* DTFE_EXACT_AVERAGE=1 .*run_ps_dtfe\.sh -s FAM-1-Dark -g [0-9]+ 99$')" = 0 ] \
         && [ "$(lacks "${TMP}/g_pipeline.log" '^\+ env .*( DTFE_GPU=| DTFE_RUN_MODE=| -m( |$))')" = 0 ] \
         && [ "$(has01 "${TMP}/g_pipeline.log" '^Slice: .* deposit: exact \(the default\) ')" = 0 ] \
         && grep -aqF "   GPU: ${PICK}   deposit: " "${TMP}/g_pipeline.log" \
         && grep -aqxF "Run mode: ${MODE_DEFAULT}" "${TMP}/g_pipeline.log" && echo 0 || echo 1)" \
      "the pipeline's default run_ps_dtfe.sh call: the exact deposit (DTFE_EXACT_AVERAGE=1), no -m, no DTFE_GPU or DTFE_RUN_MODE word; the header: the exact deposit, '${PICK}', '${MODE_DEFAULT}'"
check_rc "$([ "$(has01 "${TMP}/g_pipeline_s.log" '^\+ env .* DTFE_EXACT_AVERAGE=0 .*run_ps_dtfe\.sh -s FAM-1-Dark -g [0-9]+ 99$')" = 0 ] \
         && [ "$(lacks "${TMP}/g_pipeline_s.log" '^\+ env .*( DTFE_GPU=| DTFE_RUN_MODE=| -m( |$))')" = 0 ] && echo 0 || echo 1)" \
      "... SUBSAMPLES=2 alone: the sampled deposit (DTFE_EXACT_AVERAGE=0 handed on), no -m"
check_rc "$(has01 "${TMP}/g_pipeline.log" 'make_image_plane.py .*--planes 1 --thickness 2 --supersample 1 -o ')" \
      "... and makes a single plane without sub-sampling"
DRY_RUN=1 DTFE_GPU=0 PTS_VEL_GRAD=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 PLANE_WINDOW="0 50 0 25" CENTER=40 \
    SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline2.log" 2>&1 || true
check_rc "$([ "$(has01 "${TMP}/g_pipeline2.log" '^\+ env .*PTS_VEL_GRAD=0 .*run_ps_dtfe\.sh -s FAM-1-Dark -g [0-9]+ 99$')" = 0 ] \
         && [ "$(lacks "${TMP}/g_pipeline2.log" '^\+ env .* DTFE_GPU=')" = 0 ] \
         && grep -aqF '   GPU: DTFE_GPU=0   deposit: ' "${TMP}/g_pipeline2.log" && echo 0 || echo 1)" \
      "PTS_VEL_GRAD=0 is passed on; DTFE_GPU=0 is no word of the call (the environment carries it), the header names it"
check_rc "$(has01 "${TMP}/g_pipeline2.log" 'make_image_plane.py .*--axis z --nu [0-9]+ --planes 4 --thickness 3 --supersample 2 --center 40 --plane-window 0 50 0 25 -o ')" \
      "PLANES, THICKNESS, SUPERSAMPLE, CENTER and PLANE_WINDOW reach make_image_plane.py"
# run_ps_dtfe.sh's other knobs, which reach it in the environment: a dry run plans its call and makes no cache folder
DRY_RUN=1 DTFE_GPU=0 PTS_VEL_GRAD=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 PLANE_WINDOW="0 50 0 25" CENTER=40 \
    SUBSAMPLES=2 PS_MERGE=index-order DTFE_RESUME=0 TESSELLATION_CACHE="${TMP}/g_tc" PARALLEL_TRIANGULATION=1 \
    PS_LAGRANGIAN_PADDING=4 PS_LAGRANGIAN_INPUT= SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh \
    >"${TMP}/g_pipeline2n.log" 2>&1 || true
check_rc "$([ ! -e "${TMP}/g_tc" ] && [ "$(has01 "${TMP}/g_pipeline2n.log" '^\+ env .*run_ps_dtfe\.sh -s FAM-1-Dark -g [0-9]+ 99$')" = 0 ] \
         && echo 0 || echo 1)" \
      "... with run_ps_dtfe.sh's other knobs set (TESSELLATION_CACHE among them) a dry run plans its call and creates no cache folder"
# finding nothing to compute is a failure: with the data drive unmounted, a mistyped SIMS or a SNAPS
# matching no snapshot, an ALL DONE with exit 0 would show in the launcher as a success
mkdir -p "${TMP}/empty root"
rc=0; env -u SIMS DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_nosims.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "2" ] && [ "$(has01 "${TMP}/g_pipe_nosims.log" 'no simulation given: set SIMS')" = 0 ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh without SIMS refuses with exit 2: there is no default simulation (SIMS=all discovers) (${rc})"
rc=0; SIMS=all DTFE_DATA_ROOT="${TMP}/empty root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_none.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "run_ps_pipeline.sh with no simulation under the data root exits 1 (${rc})"
check_rc "$(has01 "${TMP}/g_pipe_none.log" '^!! nothing to compute')" "... says there is nothing to compute"
check_rc "$(lacks "${TMP}/g_pipe_none.log" 'ALL DONE')" "... and not ALL DONE"
rc=0; SIMS=all DTFE_DATA_ROOT="${TMP}/no such root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_unmounted.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a data root that does not exist (the drive unplugged) exits 1 (${rc})"
check_rc "$(has01 "${TMP}/g_pipe_unmounted.log" 'is the data drive mounted')" "... and asks whether the drive is mounted"
rc=0; DRY_RUN=1 SIMS=FAM-1-Drak DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_typo.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a mistyped SIMS exits 1 (${rc})"
check_rc "$(has01 "${TMP}/g_pipe_typo.log" '^!! nothing to compute: no combined snapshot of FAM-1-Drak ')" "... naming the simulation"
rc=0; DRY_RUN=1 SNAPS="50 67" SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_snaps.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a SNAPS matching no snapshot exits 1 (${rc})"
check_rc "$(has01 "${TMP}/g_pipe_snaps.log" 'FAM-1-Dark: no combined snapshots numbered 50 67, skipping')" "... naming the numbers it looked for"
rc=0; DRY_RUN=1 SIMS="FAM-1-Drak FAM-1-Dark" DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_half.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "one simulation of two without snapshots is a partial skip: exit 0 (${rc})"
# a real plane (tiny), then the same geometry is reused and another one regenerates it
"${PY}" python/tools/make_image_plane.py --combined "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/combined_099.hdf5" \
    --mpc-unit 1 --axis z --nu 8 --planes 1 -o "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8" >/dev/null 2>&1 || true
if [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ]; then
    DRY_RUN=1 NU=8 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline3.log" 2>&1 || true
    check_rc "$(has01 "${TMP}/g_pipeline3.log" '^-- reusing .*hires_plane_z8.bin')" \
          "an existing plane of the requested geometry is reused"
    DRY_RUN=1 NU=8 PLANES=16 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline4.log" 2>&1 || true
    check_rc "$(has01 "${TMP}/g_pipeline4.log" 'geometry .* differs; regenerating')" \
          "another plane count regenerates the plane (and so every snapshot)"
    check_rc "$([ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.bin" ] && echo 0 || echo 1)" \
          "... but a DRY_RUN removes nothing"
    # a comparison that FAILS (here: no python) is not 'differs': the plane stays, the sim is skipped with !!
    rc5=0; PYTHON=/nonexistent/python3 NU=8 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline5.log" 2>&1 || rc5=$?
    check_rc "$(has01 "${TMP}/g_pipeline5.log" '^!! FAM-1-Dark: could not compare the plane')" \
          "a failing geometry check is reported, not read as 'differs'"
    check_rc "$([ "${rc5}" != "0" ] && [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ] && echo 0 || echo 1)" \
          "... the plane is kept and the pipeline exits non-zero (${rc5})"
    # every snapshot up to date is a success, not 'nothing to compute'; a slice of the 8^2 plane holds
    # 64 points (the pipeline compares the count)
    head -c 512 /dev/zero > "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den"
    touch -t 203001010000 "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den" \
                          "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_velGrad"
    # the settings of the (A)-style run that made these grids: the pipeline compares them too (P)
    rc6=0; DRY_RUN=1 NU=8 GRID_SIZE=24 FIELDS=density_a SUBSAMPLES=1 PTS_VEL_GRAD=0 \
        SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline6.log" 2>&1 || rc6=$?
    check_rc "$([ "${rc6}" = "0" ] && echo 0 || echo 1)" "every snapshot up to date still exits 0 (${rc6})"
    check_rc "$(has01 "${TMP}/g_pipeline6.log" 'FAM-1-Dark: all snapshots up to date')" "... with nothing to recompute"
    rm -f "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den" "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_velGrad"
else
    echo "   SKIP  plane reuse/regeneration (make_image_plane.py could not write a plane with ${PY})"
fi

echo ""
echo "(H) run_dtfe.sh runs the standard binary with --exact by default (-e, DTFE_EXACT_AVERAGE=1), --sampled for DTFE_EXACT_AVERAGE=0 or a count alone."
if [ -x "${DTFE_BIN}" ]; then
    rc=0; scripts/run_dtfe.sh -s HASIC -g 16 -e 99 >"${TMP}/h_exact.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh -e succeeds (exit ${rc})"
    check_rc "$(has01 "${TMP}/h_exact.log" 'RUNNING: .*--exact( |$)')" "the binary was called with --exact"
    if precision_double; then
        check_rc "$(has01 "${TMP}/h_exact.log" 'RUNNING: .*DTFE-double ')" "DTFE_PRECISION=double: run_dtfe.sh ran ./DTFE-double"
    fi
    check_rc "$(has01 "${TMP}/h_exact.log" 'volume averaging +: EXACT cell-tetrahedron integration of the linear interpolant \(--exact;')" \
          "... and reports the exact cell average, asked for by --exact"
    # what the wrapper run_ps_dtfe.sh has -- wall time, peak memory, a de-ANSI'd runlog
    check_rc "$(has01 "${TMP}/h_exact.log" '^  Wall time: [0-9]+ min [0-9]+ s$')" "run_dtfe.sh reports the wall time per snapshot"
    if [ -x /usr/bin/time ]; then
        check_rc "$(has01 "${TMP}/h_exact.log" '^  Peak memory: .*GB resident')" "... and the peak memory (through /usr/bin/time)"
    fi
    CLICOLOR_FORCE=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_color.log" 2>&1 || true
    check_rc "$(lacks "${TMP}/HASIC/snapdir_099/output.runlog" $'\033\\[')" "the runlog is stripped of colour codes when colours are forced"
    check_rc "$(has01 "${TMP}/HASIC/snapdir_099/output.runlog" 'total wall time')" "... and still carries the binary's own wall-time line for the Runs browser"
    # a binary that fails (a truncated input) must still be a failed snapshot: the status is taken before the strip
    mkdir -p "${TMP}/BADIN/snapdir_099"; head -c 1000 "${TMP}/HASIC/snapdir_099/combined_099.hdf5" > "${TMP}/BADIN/snapdir_099/combined_099.hdf5"
    rc_bad=0; scripts/run_dtfe.sh -s BADIN -g 16 99 >"${TMP}/h_badin.log" 2>&1 || rc_bad=$?
    check_rc "$([ "${rc_bad}" = "1" ] && echo 0 || echo 1)" "a snapshot the binary cannot read is a FAILED snapshot, exit 1 (${rc_bad})"
    check_rc "$(has01 "${TMP}/h_badin.log" 'processing complete: 0 processed, 0 skipped, 1 failed')" "... counted as failed, not processed (the exit status is read before the log is post-processed)"
    check_rc "$([ -s "${TMP}/BADIN/snapdir_099/output.runlog.failed" ] && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog" ] \
             && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog.partial" ] && echo 0 || echo 1)" \
          "... its log is output.runlog.failed: no output.runlog claims grids it never wrote"
    check_rc "$([ -f "${TMP}/HASIC/snapdir_099/output.a_den" ] && echo 0 || echo 1)" "its density grid was written"
    env -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_env.log" 2>&1 || true
    # run_dtfe.sh reports failures in its exit status, passes the
    # launcher's scratch folder and the scripts' LAMBDA_TH default, and the binaries refuse --redshiftSpace
    rc_none=0; scripts/run_dtfe.sh -s HASIC -g 16 123 >"${TMP}/h_none.log" 2>&1 || rc_none=$?
    check_rc "$([ "${rc_none}" != "0" ] && echo 0 || echo 1)" "run_dtfe.sh exits non-zero when no snapshot was processed (${rc_none})"
    check_rc "$(has01 "${TMP}/h_none.log" 'processing complete: 0 processed, 1 skipped, 0 failed')" "... and says so in its summary line"
    SCR_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-scratch-check.XXXXXX")"   # local: the binary refuses iCloud paths (and Linux has no /private/tmp)
    rc_scr=0; SCRATCH_DIR="${SCR_DIR}" scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_scratch.log" 2>&1 || rc_scr=$?
    check_rc "$([ "${rc_scr}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with SCRATCH_DIR succeeds (exit ${rc_scr})"
    check_rc "$(has01 "${TMP}/h_scratch.log" "RUNNING: .*--scratch-dir ${SCR_DIR}")" "... and hands the folder to the binary (--scratch-dir)"
    check_rc "$(has01 "${TMP}/h_scratch.log" 'RUNNING: .*--lambda-th 0.3')" "run_dtfe.sh passes the scripts' lambda-th default (0.3, the binary's own default too)"
    # standard DTFE pads with 15 spacings by default (complete at z = 0 on TNG50/100/300), not config.sh's
    # 50, which stays the PS-DTFE scripts' value; a caller's PADDING still wins
    pad_of() { env -u PADDING -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID "$@" DRY_RUN=1 "${PAD_DRIVER}" -s HASIC -g 16 99 2>&1 \
                   | grep -oE -e '--padding +[0-9]+' | head -1 | grep -oE '[0-9]+$'; }
    PAD_DRIVER=scripts/run_dtfe.sh
    pad_def="$(pad_of)"; pad_p="$(pad_of PADDING=12)"
    check_rc "$([ "${pad_def}" = "15" ] && echo 0 || echo 1)" "run_dtfe.sh pads with 15 spacings by default (${pad_def:-none})"
    check_rc "$([ "${pad_p}" = "12" ] && echo 0 || echo 1)" "... a caller's PADDING wins (${pad_p:-none})"
    PAD_DRIVER=scripts/run_ps_dtfe.sh; pad_ps="$(pad_of)"
    check_rc "$([ "${pad_ps}" = "50" ] && echo 0 || echo 1)" "... while run_ps_dtfe.sh keeps config.sh's 50 (${pad_ps:-none})"
    rm -rf "${SCR_DIR}"
    # FIELDS reaches the standard binary (the launcher's ticks: the T-web), a phase-space dispersion is left out;
    # TESSELLATION_CACHE hands a (created) local folder to --tessellation-cache
    TC_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-tess-check.XXXXXX")/cache"
    rc_f=0; FIELDS="density_a tweb_a dispersion_a" TESSELLATION_CACHE="${TC_DIR}" scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_fields.log" 2>&1 || rc_f=$?
    check_rc "$([ "${rc_f}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with FIELDS and TESSELLATION_CACHE succeeds (exit ${rc_f})"
    check_rc "$(has01 "${TMP}/h_fields.log" 'RUNNING: .*--field density_a tweb_a +--')" "... FIELDS is the field list (the T-web for standard DTFE)"
    check_rc "$(has01 "${TMP}/h_fields.log" 'standard DTFE leaves it out')" "... the phase-space dispersion is left out, and it says so"
    check_rc "$(has01 "${TMP}/h_fields.log" "RUNNING: .*--tessellation-cache ${TC_DIR}")" "... TESSELLATION_CACHE becomes --tessellation-cache"
    check_rc "$([ -d "${TC_DIR}" ] && echo 0 || echo 1)" "... and the cache folder is created"
    check_rc "$([ -f "${TMP}/HASIC/snapdir_099/output.a_tweb" ] || ls "${TMP}/HASIC/snapdir_099/"output.*tweb* >/dev/null 2>&1; echo $?)" \
        "... and the T-web grid is written"
    rm -rf "$(dirname "${TC_DIR}")"
    # SCALAR_DATASET: a particle value gridded as the scalar -- '--scalar-dataset NAME', scalar_a and density_a join
    # FIELDS; a constant, negative value (the potential's sign) is that constant in every cell, the interpolant of a
    # constant being the constant
    mkdir -p "${TMP}/HASPOT/snapdir_099"
    "${PY}" - "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/HASPOT/snapdir_099/combined_099.hdf5" <<'PYEOF'
import shutil, sys
import h5py, numpy as np
shutil.copy(sys.argv[1], sys.argv[2])
with h5py.File(sys.argv[2], "r+") as f:
    g = f["PartType1"]
    g.create_dataset("Potential", data=np.full(g["ParticleIDs"].shape[0], -5.0, dtype=np.float32))
PYEOF
    rc_sc=0; SCALAR_DATASET=Potential FIELDS=velocity_a OUTPUT_PREFIX=pot scripts/run_dtfe.sh -s HASPOT -g 16 99 \
        >"${TMP}/h_scalar.log" 2>&1 || rc_sc=$?
    "${PY}" - "${TMP}/HASPOT/snapdir_099/pot.a_scalar" >"${TMP}/h_scalar.check" 2>&1 <<'PYEOF' || true
import sys
import numpy as np
import os
v = np.fromfile(sys.argv[1], dtype=np.dtype(os.environ.get("DTFE_TEST_REAL", "float32")))
ok = v.size == 16 ** 3 and bool(np.all(np.abs(v + 5.0) < 1e-4))
print("SCALAR OK" if ok else f"SCALAR BAD size {v.size} range {v.min() if v.size else 0} {v.max() if v.size else 0}")
PYEOF
    check_rc "$([ "${rc_sc}" = "0" ] && grep -q '^SCALAR OK' "${TMP}/h_scalar.check" && echo 0 || echo 1)" \
        "run_dtfe.sh SCALAR_DATASET=Potential: the particles' constant -5 is -5 in every cell of '.a_scalar' (${rc_sc})"
    check_rc "$(has01 "${TMP}/h_scalar.log" 'RUNNING: .*--field density_a velocity_a scalar_a +--lambda-th 0.3 +--scalar-dataset Potential')" \
        "... density_a and scalar_a joined FIELDS, and the dataset reached the binary (--scalar-dataset)"
    rc_scp=0; DRY_RUN=1 SCALAR_DATASET=Potential FIELDS=density_a OUTPUT_PREFIX=ps_pot scripts/run_ps_dtfe.sh -s HASPOT -g 16 99 \
        >"${TMP}/h_scalar_ps.log" 2>&1 || rc_scp=$?
    check_rc "$([ "${rc_scp}" = "0" ] && echo 0 || echo 1)" "run_ps_dtfe.sh takes the same SCALAR_DATASET (${rc_scp})"
    check_rc "$(has01 "${TMP}/h_scalar_ps.log" '--field +density_a +scalar_a +--lambda-th +0\.3 +--scalar-dataset +Potential( |$)')" \
        "... its command carries scalar_a and --scalar-dataset too"
    rc_z=0; FIELDS="density_a scalar_a" OUTPUT_PREFIX=pot0 scripts/run_dtfe.sh -s HASPOT -g 16 99 >"${TMP}/h_scalar_zero.log" 2>&1 || rc_z=$?
    rc_o=0; FIELDS="density_a --scalar-dataset Potential" OUTPUT_PREFIX=pot0 scripts/run_ps_dtfe.sh -s HASPOT -g 16 99 \
        >"${TMP}/h_scalar_opt.log" 2>&1 || rc_o=$?
    check_rc "$([ "${rc_z}" = "2" ] && [ "${rc_o}" = "2" ] && ! ls "${TMP}/HASPOT/snapdir_099/"pot0.* >/dev/null 2>&1 && echo 0 || echo 1)" \
        "a scalar field without SCALAR_DATASET, and an option inside FIELDS: refused, exit 2, nothing written (${rc_z} ${rc_o})"
    check_rc "$(has01 "${TMP}/h_scalar_zero.log" 'write zeros')" "... the first says the grid would be zeros"
    check_rc "$(has01 "${TMP}/h_scalar_opt.log" 'not options.*SCALAR_DATASET=NAME')" "... the second names SCALAR_DATASET"
    rm -rf "${TMP}/HASPOT"
    # the pipeline's settings comparison (run_ps_pipeline.sh same_settings) knows the dataset: grids made for Potential
    # satisfy a plain want and a Potential want, not a SubfindHsml one
    eval "$(sed -n '/^run_settings() {/,/^}/p; /^same_settings() {/,/^}/p' scripts/run_ps_pipeline.sh)"
    ss_have="$(run_settings 'RUNNING: ./PS-DTFE in out --grid 16 --field density_a scalar_a --lambda-th 0.3 --scalar-dataset Potential --exact')"
    ss_plain="$(run_settings './PS-DTFE in out --grid 16 --field density_a --lambda-th 0.3 --exact')"
    ss_same="$(run_settings './PS-DTFE in out --grid 16 --field density_a scalar_a --lambda-th 0.3 --scalar-dataset Potential --exact')"
    ss_other="$(run_settings './PS-DTFE in out --grid 16 --field density_a scalar_a --lambda-th 0.3 --scalar-dataset SubfindHsml --exact')"
    ss_r=""
    same_settings "${ss_have}" "${ss_plain}" && ss_r="${ss_r}p"
    same_settings "${ss_have}" "${ss_same}" && ss_r="${ss_r}s"
    same_settings "${ss_have}" "${ss_other}" || ss_r="${ss_r}o"
    same_settings "${ss_plain}" "${ss_same}" || ss_r="${ss_r}n"
    check_rc "$([ "${ss_r}" = "pson" ] && echo 0 || echo 1)" \
        "run_ps_pipeline.sh compares the dataset: a Potential set serves a plain or a Potential want, not a SubfindHsml one, and a plain set no Potential want (${ss_r})"
    # SAMPLE_POINTS: the standard DTFE interpolant at a points file too (the launcher's hi-res slice)
    printf '1 1 1\n2 3 4\n5 5 5\n' > "${TMP}/h_pts.txt"
    rc_sp=0; SAMPLE_POINTS="${TMP}/h_pts.txt" PTS_VEL_GRAD=1 scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_pts.log" 2>&1 || rc_sp=$?
    check_rc "$([ "${rc_sp}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with SAMPLE_POINTS succeeds (exit ${rc_sp})"
    check_rc "$(has01 "${TMP}/h_pts.log" "RUNNING: .*--sample-points ${TMP}/h_pts.txt +--pts-vel-grad")" \
        "... and hands the points to the standard binary (with the velocity gradient)"
    check_rc "$([ -s "${TMP}/HASIC/snapdir_099/output.pts_den" ] && echo 0 || echo 1)" "... which writes the point values (.pts_den)"
    # zero-padded snapshot numbers are decimal (bash's printf reads '099' as octal: an error, then snapshot 000)
    rc_oct=0; scripts/run_dtfe.sh -s HASIC -g 16 099 >"${TMP}/h_oct.log" 2>&1 || rc_oct=$?
    check_rc "$([ "${rc_oct}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh accepts a zero-padded snapshot number (099, exit ${rc_oct})"
    check_rc "$(has01 "${TMP}/h_oct.log" 'Snapshot 099 processed successfully')" "... and runs snapshot 99, not 0 (octal)"
    rc_bad=0; scripts/run_dtfe.sh -s HASIC -g 16 9x >"${TMP}/h_bad.log" 2>&1 || rc_bad=$?
    check_rc "$([ "${rc_bad}" = "2" ] && echo 0 || echo 1)" "a non-numeric snapshot argument is refused with exit 2 (${rc_bad})"
    rc_oct=0; scripts/run_ps_dtfe.sh -s HASIC -g 16 -n 1 099 >"${TMP}/h_oct_ps.log" 2>&1 || rc_oct=$?
    check_rc "$(has01 "${TMP}/h_oct_ps.log" 'Processing snapshot 099')" "run_ps_dtfe.sh reads 099 as snapshot 99 too (exit ${rc_oct})"
    rc_rs=0; "${DTFE_BIN}" "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/rs_out" --grid 16 --redshiftSpace 0 0 1 --input-format 105 >"${TMP}/h_rs.log" 2>&1 || rc_rs=$?
    check_rc "$([ "${rc_rs}" != "0" ] && echo 0 || echo 1)" "--redshiftSpace is refused (not a silent no-op), exit ${rc_rs}"
    check_rc "$(has01 "${TMP}/h_rs.log" 'not implemented in this version')" "... with a message saying the transform is not implemented"
    check_rc "$(has01 "${TMP}/h_env.log" 'RUNNING: .*--exact( |$)')" "DTFE_EXACT_AVERAGE=1 does the same"
    # the exact average is the default (the production standard); DTFE_EXACT_AVERAGE=0 is the sampled one
    rc_pl=0; env -u DTFE_EXACT_AVERAGE -u SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_plain.log" 2>&1 || rc_pl=$?
    check_rc "$([ "${rc_pl}" = "0" ] && [ "$(has01 "${TMP}/h_plain.log" 'RUNNING: .*--exact( |$)')" = 0 ] \
             && [ "$(has01 "${TMP}/h_plain.log" 'RUNNING: .*--field density_a velocity_a divergence_a shear_a ')" = 0 ] \
             && [ "$(has01 "${TMP}/h_plain.log" 'volume averaging +: EXACT cell-tetrahedron integration')" = 0 ] \
             && echo 0 || echo 1)" "a plain run passes --exact and the four analysis fields (the production defaults; exit ${rc_pl})"
    env -u SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=0 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_mc.log" 2>&1 || true
    check_rc "$([ "$(lacks "${TMP}/h_mc.log" 'RUNNING: .*--exact( |$)')" = 0 ] && [ "$(has01 "${TMP}/h_mc.log" 'RUNNING: .*--sampled( |$)')" = 0 ] \
             && [ "$(has01 "${TMP}/h_mc.log" 'volume averaging +: SAMPLED \(--sampled\)')" = 0 ] \
             && [ "$(has01 "${TMP}/h_mc.log" "'_a' interpolation +: CPU \((the default for the sampled average; --gpu runs it on the GPU|this binary was built without GPU support)\)")" = 0 ] \
             && echo 0 || echo 1)" \
          "DTFE_EXACT_AVERAGE=0 passes --sampled, not --exact: the Monte-Carlo cell mean, said in the run log; on the CPU, the binary's own pick (DTFE_GPU unset: the driver passes no device), which its device line names"
    # a DTFE_GPU the caller sets reaches the binary as given (the driver does not read or rewrite it)
    env -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=0 DTFE_GPU=off \
        scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_mc_off.log" 2>&1 || true
    check_rc "$(has01 "${TMP}/h_mc_off.log" "'_a' interpolation +: CPU \(DTFE_GPU=off\)")" \
          "... DTFE_GPU=off: the binary's device line names the caller's own word, 'CPU (DTFE_GPU=off)'"
    # a run-mode word the binary does not know: the driver passes it on, the binary refuses it and the snapshot fails
    rc_rm=0; env -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=0 \
        scripts/run_dtfe.sh -s HASIC -g 16 --run-mode fast 99 >"${TMP}/h_rm_bad.log" 2>&1 || rc_rm=$?
    check_rc "$([ "${rc_rm}" = "1" ] && [ "$(has01 "${TMP}/h_rm_bad.log" "Unknown value 'fast' for '--run-mode'")" = 0 ] \
             && echo 0 || echo 1)" \
          "... '--run-mode fast': the binary refuses the word, the snapshot fails and the driver exits 1 (exit ${rc_rm})"
    # SUBSAMPLES (-n) reaches the Monte-Carlo mean as --subsamples, only when set
    env -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=0 DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 -n 2 99 >"${TMP}/h_mc_n.log" 2>&1 || true
    check_rc "$([ "$(has01 "${TMP}/h_mc_n.log" '--lambda-th +0\.3 .*--sampled +--subsamples +2$')" = 0 ] && [ "$(lacks "${TMP}/h_mc.log" '--subsamples')" = 0 ] \
             && echo 0 || echo 1)" "... -n 2 (SUBSAMPLES) passes --subsamples 2; unset, nothing (the binary's own count)"
    # the rule (config.sh average_resolve, the binary's own): -e / --exact or --sampled first, then DTFE_EXACT_AVERAGE,
    # then a SUBSAMPLES count (the Monte-Carlo mean), then the default (exact); the banner names what decided. Dry
    # runs with the knobs unset first; the planned command always names the cell average (right after --lambda-th),
    # --subsamples only the sampled one
    hplan() {   # <log> [VAR=value ...] [-- driver options]: echoes run_dtfe.sh's planned command line
        local log="$1" e=()
        shift
        while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do e+=("$1"); shift; done
        [ "$#" -gt 0 ] && shift
        env -u DTFE_EXACT_AVERAGE -u SUBSAMPLES -u FACE_GRADIENT -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID \
            ${e[@]+"${e[@]}"} DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 "$@" 99 >"${log}" 2>&1 || true
        sed -n '/would run:/{n;p;}' "${log}"
    }
    h_def="$(hplan "${TMP}/h_def.log")"; h_sub="$(hplan "${TMP}/h_sub.log" SUBSAMPLES=2)"
    h_n="$(hplan "${TMP}/h_n.log" -- -n 2)"; h_off="$(hplan "${TMP}/h_off.log" DTFE_EXACT_AVERAGE=0)"
    check_rc "$(printf '%s\n' "${h_def}" | grep -qE -e '--lambda-th +0\.3 +--exact$' \
             && grep -aqxF 'Cell averages: exact (--exact; the default)' "${TMP}/h_def.log" \
             && printf '%s\n' "${h_sub}" | grep -qE -e '--lambda-th +0\.3 +--sampled +--subsamples +2$' && [ "${h_n}" = "${h_sub}" ] \
             && grep -aqxF 'Cell averages: sampled, Monte-Carlo sample means (--sampled --subsamples 2; SUBSAMPLES=2)' "${TMP}/h_sub.log" \
             && printf '%s\n' "${h_off}" | grep -qE -e '--lambda-th +0\.3 +--sampled$' \
             && grep -aqxF 'Cell averages: sampled, Monte-Carlo sample means (--sampled; DTFE_EXACT_AVERAGE=0)' "${TMP}/h_off.log" \
             && echo 0 || echo 1)" \
          "the default passes --exact; SUBSAMPLES=2 or -n 2 alone --sampled --subsamples 2; DTFE_EXACT_AVERAGE=0 --sampled (the banner says why)"
    h_esub="$(hplan "${TMP}/h_esub.log" SUBSAMPLES=2 -- -e)"; h_kesub="$(hplan "${TMP}/h_kesub.log" DTFE_EXACT_AVERAGE=1 -- -n 2)"
    check_rc "$([ "${h_esub}" = "${h_def}" ] && [ "${h_kesub}" = "${h_def}" ] \
             && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: -e asks for the exact cell average' "${TMP}/h_esub.log")" = 1 ] \
             && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: DTFE_EXACT_AVERAGE=1 asks for the exact cell average' "${TMP}/h_kesub.log")" = 1 ] \
             && echo 0 || echo 1)" "-e beside SUBSAMPLES=2, and DTFE_EXACT_AVERAGE=1 beside -n 2: --exact without --subsamples, one note each"
    h_lex="$(hplan "${TMP}/h_lex.log" -- --exact)"; h_lsa="$(hplan "${TMP}/h_lsa.log" DTFE_EXACT_AVERAGE=1 -- --sampled)"
    h_le0="$(hplan "${TMP}/h_le0.log" DTFE_EXACT_AVERAGE=0 -- -e)"
    check_rc "$([ -n "${h_def}" ] && [ "${h_lex}" = "${h_def}" ] && [ "${h_lsa}" = "${h_off}" ] && [ "${h_le0}" = "${h_def}" ] && echo 0 || echo 1)" \
          "--exact is -e; --sampled beats DTFE_EXACT_AVERAGE=1 and -e DTFE_EXACT_AVERAGE=0"
    rc=0; env -u DTFE_EXACT_AVERAGE -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 -e --sampled 99 >"${TMP}/h_both.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "1" ] && [ "$(has01 "${TMP}/h_both.log" "^Error: '-e' \('--exact'\) and '--sampled' exclude each other")" = 0 ] \
             && [ "$(has01 "${TMP}/h_both.log" '^Usage: ')" = 0 ] && [ "$(lacks "${TMP}/h_both.log" 'would run:')" = 0 ] && echo 0 || echo 1)" \
          "-e and --sampled together are refused with the usage, exit 1, nothing planned (${rc})"
    # the standard binary's auto-tuner reports (not 'AUTO-TUNE-REPORT unsupported'), and the driver
    # asks it the way run_ps_dtfe.sh does: a report per snapshot, nothing computed, nothing written beside the data
    rm -rf "${TMP}/HASIC/snapdir_099/output."*        # (the grids, run logs and the superseded folder of the runs above)
    rc_rep=0; AUTO_TUNE_REPORT=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_report.log" 2>&1 || rc_rep=$?
    check_rc "$([ "${rc_rep}" = "0" ] && [ "$(grep -acE '^  AUTO-TUNE-REPORT snap=099 partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=[0-9.]+ .*over_budget=[01]$' "${TMP}/h_report.log")" = 1 ] \
             && echo 0 || echo 1)" "run_dtfe.sh AUTO_TUNE_REPORT=1: one valid standard report line for snapshot 099 (exit ${rc_rep})"
    check_rc "$([ -z "$(ls "${TMP}/HASIC/snapdir_099/" | grep '^output\.')" ] && [ "$(lacks "${TMP}/h_report.log" 'unsupported')" = 0 ] \
             && echo 0 || echo 1)" "... and nothing is written beside the data: no grids, no run log, no lock"
else
    echo "   SKIP  (${DTFE_BIN} not built)"
fi

echo ""
echo "(I) download_snapshots.sh + merge_HDF5.py: a half-finished download never becomes a merged snapshot."
# TNG-like raw chunks: the header of each promises NumFilesPerSnapshot = 4
DL="${TMP}/DL/SIMX"
mkdir -p "${DL}/snapdir_099"
"${PY}" - "${DL}/snapdir_099" <<'PYEOF'
import sys
from pathlib import Path
import h5py, numpy as np
d, n, chunks, per = Path(sys.argv[1]), 99, 4, 512
rng = np.random.default_rng(99)
for c in range(chunks):
    with h5py.File(d / f"snap_{n:03d}.{c}.hdf5", "w") as f:
        hd = f.create_group("Header").attrs
        hd["BoxSize"], hd["HubbleParam"], hd["Redshift"], hd["Time"] = 75000.0, 0.6774, 0.0, 1.0
        hd["MassTable"] = np.array([0, 0.5, 0, 0, 0, 0], dtype=float)
        hd["NumPart_ThisFile"] = np.array([0, per, 0, 0, 0, 0], dtype=np.int64)
        hd["NumPart_Total"] = np.array([0, per * chunks, 0, 0, 0, 0], dtype=np.uint32)
        hd["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hd["NumFilesPerSnapshot"] = chunks
        p = f.create_group("PartType1")
        p["Coordinates"] = rng.uniform(0, 75000.0, (per, 3))
        p["ParticleIDs"] = np.arange(c * per, (c + 1) * per, dtype=np.uint64)
        p["Velocities"] = rng.normal(0, 100, (per, 3)).astype(np.float32)
# group catalogue of snapshot 099: 4 chunks (the header says NumFiles = 4), the last with no groups at all
g = d.parent / "groups_099"; g.mkdir(exist_ok=True)
ng = [5, 3, 4, 0]
for c in range(4):
    with h5py.File(g / f"fof_subhalo_tab_099.{c}.hdf5", "w") as f:
        hd = f.create_group("Header").attrs
        hd["BoxSize"], hd["HubbleParam"], hd["Redshift"] = 75000.0, 0.6774, 0.0
        hd["NumFiles"] = np.int32(4)
        hd["Ngroups_Total"], hd["Nsubgroups_Total"] = np.int32(sum(ng)), np.int32(sum(ng))
        hd["Ngroups_ThisFile"], hd["Nsubgroups_ThisFile"] = np.int32(ng[c]), np.int32(ng[c])
        f.create_group("IDs")
        if ng[c]:
            grp = f.create_group("Group"); grp["GroupMass"] = np.full(ng[c], 1.5, dtype=np.float32)
            grp["GroupPos"] = rng.uniform(0, 75000.0, (ng[c], 3)).astype(np.float32)
            sub = f.create_group("Subhalo"); sub["SubhaloMass"] = np.full(ng[c], 0.5, dtype=np.float32)
PYEOF
cp "${DL}/snapdir_099/snap_099.0.hdf5" "${TMP}/snap_099.0.keep"
merge_dl() { "${PY}" python/tools/merge_HDF5.py -d "${DL}" "$@"; }     # the path has spaces: no word splitting
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"            # a merge stopped half-way last time
rc=0; merge_dl --snaps 99 >"${TMP}/i_merge_ok.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "merging 4 complete chunks exits 0 (${rc})"
check_rc "$([ -f "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "... the combined file is written under its final name; a stale .partial does not block it and is gone"
check_rc "$(has01 "${TMP}/i_merge_ok.log" 'Particles merged: \[2048\]$')" "... and the totals verify against the header"
"${PY}" - "${DL}/snapdir_099/combined_099.hdf5" >"${TMP}/i_content.log" 2>&1 <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    a = f["Header"].attrs
    ok = (list(a["NumPart_Total"]) == [0, 2048, 0, 0, 0, 0] and not any(a["NumPart_Total_HighWord"])
          and np.array_equal(f["PartType1/ParticleIDs"][...], np.arange(2048))
          and abs(float(a["BoxSize"]) - 75000.0 / 0.6774) < 1e-6 and "HFreeUnits" in a)
print("CONTENT OK" if ok else f"CONTENT BAD {dict(a)}")
PYEOF
check_rc "$(has01 "${TMP}/i_content.log" 'CONTENT OK')" "... its header holds int64 totals, a zero high word, the h-free box; the IDs are 0..2047 in chunk order"
rm -f "${DL}/snapdir_099/combined_099.hdf5"
mv "${DL}/snapdir_099/snap_099.2.hdf5" "${TMP}/snap_099.2.keep"
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"            # left by a stopped merge: a refusal removes it too
rc=0; merge_dl --snaps 99 >"${TMP}/i_merge_missing.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a chunk missing from disk (3 of the header's 4): exit 1 (${rc})"
check_rc "$([ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "... no combined file at all (not the 3 merged and called done)"
check_rc "$(has01 "${TMP}/i_merge_missing.log" "Problem: Can't find 1 subfile\(s\) \(the header promises 4\)")" "... the header's count is the yardstick"
check_rc "$(has01 "${TMP}/i_merge_missing.log" 'Missing: .*snap_099\.2\.hdf5')" "... and the missing chunk is named"
check_rc "$(has01 "${TMP}/i_merge_missing.log" 'Done, with problems: no combined file for snapshot 099')" "... the summary line says what has none"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL}/snapdir_099/snap_099.2.hdf5"      # a download stopped half-way
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"
rc=0; merge_dl --snaps 99 >"${TMP}/i_merge_trunc.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a truncated chunk: exit 1 and no combined file (not merged past an 'Error ... continue') (${rc})"
check_rc "$(has01 "${TMP}/i_merge_trunc.log" "Error: couldn't read snap_099\.2\.hdf5 \(an incomplete download\)")" "... the chunk is named as incomplete"
check_rc "$(has01 "${TMP}/i_merge_trunc.log" 'Run the download again: it removes an incomplete chunk')" "... with the way out"
head -c 30 "${TMP}/snap_099.2.keep" > "${DL}/snapdir_099/snap_099.2.hdf5"         # stopped inside the signature
rc=0; merge_dl --snaps 99 >"${TMP}/i_merge_tiny.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a 30-byte chunk (HDF5 says 'bad byte number' or no signature): exit 1 (${rc})"
check_rc "$(has01 "${TMP}/i_merge_tiny.log" "Error: couldn't read snap_099\.2\.hdf5 \(an incomplete download\)")" "... classed as incomplete by its size"
rm -f "${DL}/snapdir_099/snap_099.2.hdf5"
cp "${TMP}/snap_099.2.keep" "${DL}/snapdir_099/snap_099.2.hdf5"
rc=0; merge_dl --snaps 99 >"${TMP}/i_merge_again.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ -f "${DL}/snapdir_099/combined_099.hdf5" ] && echo 0 || echo 1)" "the chunk restored: the merge succeeds again (${rc})"
# the totals gate: -n 3 merges three of the four chunks -> MISMATCH, no file, --delete-chunks deletes nothing
rm -f "${DL}/snapdir_099/combined_099.hdf5"
rc=0; merge_dl -n 3 --delete-chunks --snaps 99 >"${TMP}/i_merge_n3.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "-n 3 against a header of 4: MISMATCH -> exit 1, no combined file, no .partial (${rc})"
check_rc "$(has01 "${TMP}/i_merge_n3.log" 'MISMATCH vs header total \[2048\]')" "... the MISMATCH line names the header's total"
check_rc "$(has01 "${TMP}/i_merge_n3.log" 'WARNING: -n 3 given, but the header says 4')" "... and -n disagreeing with the header is said"
check_rc "$([ "$(ls "${DL}"/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" "... --delete-chunks deleted nothing after the refusal"
# the high word of the totals counts (the -1 runs exceed 2^32)
"${PY}" - "${DL}/snapdir_099/snap_099.0.hdf5" <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r+") as f:
    f["Header"].attrs["NumPart_Total_HighWord"] = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint32)
PYEOF
rc=0; merge_dl --snaps 99 >"${TMP}/i_merge_hw.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && echo 0 || echo 1)" "a high word in the header's total is part of the expected count: MISMATCH (${rc})"
check_rc "$(has01 "${TMP}/i_merge_hw.log" 'MISMATCH vs header total \[4294969344\]')" "... 2^32 + 2048 named as the total"
cp "${TMP}/snap_099.0.keep" "${DL}/snapdir_099/snap_099.0.hdf5"
# a merge with --delete-chunks, then the same command again: done, not a problem
DLC="${TMP}/DLC/SIMX"; mkdir -p "${DLC}"; cp -R "${DL}/snapdir_099" "${DLC}/snapdir_099"
rc=0; "${PY}" python/tools/merge_HDF5.py -d "${DLC}" --delete-chunks --snaps 99 >"${TMP}/i_merge_del.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ -f "${DLC}/snapdir_099/combined_099.hdf5" ] && [ -z "$(ls "${DLC}"/snapdir_099/snap_099.*.hdf5 2>/dev/null)" ] && echo 0 || echo 1)" \
      "--delete-chunks after a verified merge: exit 0, the combined file, no chunk left (${rc})"
rc=0; "${PY}" python/tools/merge_HDF5.py -d "${DLC}" --delete-chunks --snaps 99 >"${TMP}/i_merge_del2.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "the same merge again (no chunks, the combined file there): exit 0, nothing to do (${rc})"
check_rc "$(has01 "${TMP}/i_merge_del2.log" 'already exists and no chunks are left')" "... and says so"
check_rc "$(lacks "${TMP}/i_merge_del2.log" 'Done, with problems')" "... with no problem reported"
# the disk failing while the .partial is written: no file, the error named (ulimit -f caps the file size)
rm -f "${DL}/snapdir_099/combined_099.hdf5"
rc=0; ( ulimit -f 40; PYTHONUNBUFFERED=1 merge_dl --snaps 99 ) >"${TMP}/i_merge_disk.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" != "0" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a write failure (file size limit): non-zero exit, no combined file, no .partial (${rc})"
check_rc "$(has01 "${TMP}/i_merge_disk.log" 'could not write .*combined_099\.hdf5\.partial')" "... reported as the disk's, not a chunk's"
# the chunks' other per-particle fields (TNG: the potential and four Subfind quantities, each with its unit
# attributes): carried into the merged file h-free, or the merge refused; --add-fields for a file merged without
DF="${TMP}/DF"
"${PY}" - "${DF}" <<'PYEOF'
import sys
from pathlib import Path
import h5py, numpy as np
root, chunks, per, h = Path(sys.argv[1]), 4, 300, 0.6774
units = {"Potential": 0.0, "SubfindDMDensity": 2.0, "SubfindDensity": 2.0, "SubfindHsml": -1.0, "SubfindVelDisp": 0.0}
for sim, drop in (("NEW", None), ("SHORT", (2, "SubfindHsml")), ("OLD", None), ("OLD2", None), ("OTHER", None),
                  ("UNITS", None), ("GAS", None)):
    d = root / sim / "snapdir_099"; d.mkdir(parents=True)
    rng = np.random.default_rng(11)
    for c in range(chunks):
        with h5py.File(d / f"snap_099.{c}.hdf5", "w") as f:
            hd = f.create_group("Header").attrs
            hd["BoxSize"], hd["HubbleParam"], hd["Redshift"], hd["Time"] = 75000.0, h, 0.0, 1.0
            hd["MassTable"] = np.array([0, 0.5, 0, 0, 0, 0], dtype=float)
            hd["NumPart_ThisFile"] = np.array([0, per, 0, 0, 0, 0], dtype=np.int64)
            hd["NumPart_Total"] = np.array([0, per * chunks, 0, 0, 0, 0], dtype=np.uint32)
            hd["NumFilesPerSnapshot"] = chunks
            p = f.create_group("PartType1")
            p["Coordinates"] = rng.uniform(0, 75000.0, (per, 3))
            p["ParticleIDs"] = rng.permutation(np.arange(c * per, (c + 1) * per, dtype=np.uint64))
            p["Velocities"] = rng.normal(0, 100, (per, 3)).astype(np.float32)
            for name, power in units.items():
                if drop != (c, name):
                    p.create_dataset(name, data=rng.lognormal(0, 1, per).astype(np.float32)).attrs["h_scaling"] = power
            if sim == "UNITS":          # an attribute other than the table has; no attribute; a field nothing knows
                p["SubfindVelDisp"].attrs["h_scaling"] = 1.0
                del p["SubfindHsml"].attrs["h_scaling"]
                p["Potential"].attrs["to_cgs"] = 2.5
                p["Unknown"] = rng.normal(0, 1, per).astype(np.float32)
            if sim == "GAS":            # a full-physics run: the header counts gas too
                hd["NumPart_Total"] = np.array([5, per * chunks, 0, 0, 0, 0], dtype=np.uint32)
    if sim in ("OLD", "OLD2", "OTHER"): # a file as merges wrote it before: the three datasets, h-free, nothing else
        with h5py.File(d / "combined_099.hdf5", "w") as f, h5py.File(d / "snap_099.0.hdf5", "r") as f0:
            f.copy(f0["Header"], "Header")
            hd = f["Header"].attrs
            hd["BoxSize"], hd["MassTable"], hd["HFreeUnits"] = hd["BoxSize"] / h, np.asarray(hd["MassTable"]) / h, h
            hd["NumPart_ThisFile"] = hd["NumPart_Total"] = np.array([0, per * chunks, 0, 0, 0, 0], dtype=np.int64)
            p = f.create_group("PartType1")
            get = lambda name: np.concatenate([h5py.File(d / f"snap_099.{c}.hdf5", "r")["PartType1"][name][...] for c in range(chunks)])
            p.create_dataset("Coordinates", data=(get("Coordinates") / h).astype(np.float32), chunks=True)
            p.create_dataset("ParticleIDs", data=get("ParticleIDs"), chunks=True)
            p.create_dataset("Velocities", data=get("Velocities"), chunks=True)
    if sim == "OTHER":                  # ... and chunks that are not the ones it was merged from
        with h5py.File(d / "snap_099.1.hdf5", "r+") as f:
            f["PartType1"]["ParticleIDs"][...] = f["PartType1"]["ParticleIDs"][...][::-1].copy()
PYEOF
merge_df() { "${PY}" python/tools/merge_HDF5.py -d "${DF}/$1" --snaps 99 "${@:2}"; }
fields_df() {   # <sim>: the merged file's other fields against its chunks; 'FIELDS OK', or what is wrong
"${PY}" - "${DF}/$1/snapdir_099" <<'PYEOF'
import sys
from pathlib import Path
import h5py, numpy as np
d, h = Path(sys.argv[1]), 0.6774
units = {"Potential": 0.0, "SubfindDMDensity": 2.0, "SubfindDensity": 2.0, "SubfindHsml": -1.0, "SubfindVelDisp": 0.0}
get = lambda name: np.concatenate([h5py.File(d / f"snap_099.{c}.hdf5", "r")["PartType1"][name][...] for c in range(4)])
with h5py.File(d / "combined_099.hdf5", "r") as f:
    g = f["PartType1"]
    bad = [n for n, p in units.items() if n not in g
           or not np.array_equal(g[n][...], (get(n).astype(np.float64) * h ** p).astype(np.float32))
           or float(g[n].attrs["h_power_applied"]) != p or float(g[n].attrs["h_scaling"]) != 0.0]
    base = (np.array_equal(g["ParticleIDs"][...], get("ParticleIDs")) and np.array_equal(g["Velocities"][...], get("Velocities"))
            and np.array_equal(g["Coordinates"][...], (get("Coordinates") / h).astype(np.float32)) and len(g) == 8)
print("FIELDS OK" if base and not bad else f"FIELDS BAD {bad} {base}")
PYEOF
}
rc=0; merge_df NEW >"${TMP}/i_fields_new.log" 2>&1 || rc=$?
fields_df NEW >"${TMP}/i_fields_new.check" 2>&1
check_rc "$([ "${rc}" = "0" ] && grep -q '^FIELDS OK' "${TMP}/i_fields_new.check" && echo 0 || echo 1)" \
      "a merge carries the chunks' other particle fields: the raw values times h^h_scaling in the particles' order, the power applied recorded; positions, velocities and IDs as before (${rc})"
check_rc "$(has01 "${TMP}/i_fields_new.log" 'also carrying Potential, SubfindDMDensity, SubfindDensity, SubfindHsml, SubfindVelDisp')" "... and names them"
layout_df() { "${PY}" - "$1" <<'PYEOF'
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    print("ONE PIECE" if all(f["PartType1"][n].chunks is None for n in f["PartType1"]) else "IN PIECES")
PYEOF
}
check_rc "$([ "$(layout_df "${DF}/NEW/snapdir_099/combined_099.hdf5")" = "ONE PIECE" ] && echo 0 || echo 1)" "... every dataset written at its full size, in one piece"
sum0="$(cksum < "${DF}/NEW/snapdir_099/combined_099.hdf5")"
rc=0; merge_df NEW --verify >"${TMP}/i_fields_verify.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(cksum < "${DF}/NEW/snapdir_099/combined_099.hdf5")" = "${sum0}" ] && [ "$(ls "${DF}"/NEW/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "--verify of that file: exit 0, the file and its chunks untouched (${rc})"
check_rc "$(has01 "${TMP}/i_fields_verify.log" 'VERIFIED: combined_099\.hdf5 holds everything its 4 chunks do')" "... VERIFIED, particle for particle"
# the comparison gates --delete-chunks: one position a single float step off, one carried value doubled
"${PY}" - "${DF}/NEW/snapdir_099/combined_099.hdf5" <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r+") as f:
    c = f["PartType1"]["Coordinates"]
    c[100, 2] = np.nextafter(c[100, 2], np.float32(np.inf))
    f["PartType1"]["SubfindHsml"][700] *= 2
PYEOF
rc=0; merge_df NEW --verify --delete-chunks >"${TMP}/i_fields_verify_bad.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ "$(ls "${DF}"/NEW/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "a position one float step off and a carried value doubled: --verify --delete-chunks exit 1, NO chunk deleted (${rc})"
check_rc "$(has01 "${TMP}/i_fields_verify_bad.log" 'snap_099\.0\.hdf5: PartType1/Coordinates differs within rows 0 to 300')" "... the position's dataset and rows named"
check_rc "$(has01 "${TMP}/i_fields_verify_bad.log" 'snap_099\.2\.hdf5: PartType1/SubfindHsml differs within rows 600 to 900')" "... and the carried value's"
rc=0; merge_df NEW --add-fields --delete-chunks >"${TMP}/i_fields_add_bad.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ "$(ls "${DF}"/NEW/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "--add-fields --delete-chunks of that file (nothing to add): compared first, exit 1, NO chunk deleted (${rc})"
check_rc "$(has01 "${TMP}/i_fields_add_bad.log" 'Done, with problems: not verified against its chunks: snapshot 099')" "... and the summary says so"
rc=0; merge_df NEW --groupcats --verify --delete-chunks >"${TMP}/i_fields_modes.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "2" ] && [ "$(ls "${DF}"/NEW/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "--verify beside --groupcats: refused before anything is done, exit 2 (${rc})"
check_rc "$(has01 "${TMP}/i_fields_modes.log" 'work on snapshots alone')" "... and says why"
rc=0; merge_df SHORT >"${TMP}/i_fields_short.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ ! -e "${DF}/SHORT/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DF}/SHORT/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a chunk without one of the fields: exit 1, no combined file, no .partial (${rc})"
check_rc "$(has01 "${TMP}/i_fields_short.log" 'not every particle has a value of PartType1/SubfindHsml')" "... the short field named"
old_sum() { "${PY}" - "$1" <<'PYEOF'
import hashlib, sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    print(hashlib.sha256(b"".join(f["PartType1"][n][...].tobytes() for n in ("Coordinates", "ParticleIDs", "Velocities"))).hexdigest())
PYEOF
}
OLDF="${DF}/OLD/snapdir_099/combined_099.hdf5"
rc=0; merge_df OLD --verify --delete-chunks >"${TMP}/i_fields_verify_old.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ "$(ls "${DF}"/OLD/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "--verify --delete-chunks of a file merged without the other fields: exit 1, NO chunk deleted (${rc})"
check_rc "$([ "$(grep -c 'does not carry PartType1/Potential, PartType1/SubfindDMDensity' "${TMP}/i_fields_verify_old.log")" = "1" ] && echo 0 || echo 1)" \
      "... the fields it lacks named, once"
check_rc "$([ "$(grep -c 'the same (in what the file carries)' "${TMP}/i_fields_verify_old.log")" = "4" ] && echo 0 || echo 1)" \
      "... its positions, velocities and IDs the chunks' own, in each of the 4 chunks"
check_rc "$(has01 "${TMP}/i_fields_verify_old.log" 'Done, with problems: not verified against its chunks: snapshot 099')" "... and the summary says so"
touch -t 202601020304.05 "${OLDF}" "${TMP}/i_fields_stamp"
sum0="$(old_sum "${OLDF}")"; size0="$(wc -c < "${OLDF}" | tr -d ' ')"
rc=0; merge_df OLD --add-fields >"${TMP}/i_fields_add.log" 2>&1 || rc=$?
fields_df OLD >"${TMP}/i_fields_add.check" 2>&1
check_rc "$([ "${rc}" = "0" ] && grep -q '^FIELDS OK' "${TMP}/i_fields_add.check" && echo 0 || echo 1)" \
      "--add-fields puts them into a file merged without them (${rc})"
check_rc "$([ "$(old_sum "${OLDF}")" = "${sum0}" ] && [ "$(wc -c < "${OLDF}" | tr -d ' ')" -gt "${size0}" ] && [ ! -e "${OLDF}.partial" ] && echo 0 || echo 1)" \
      "... positions, velocities and IDs bit for bit as they were; the file grew; no .partial"
check_rc "$([ ! "${OLDF}" -nt "${TMP}/i_fields_stamp" ] && [ ! "${TMP}/i_fields_stamp" -nt "${OLDF}" ] && echo 0 || echo 1)" \
      "... and it keeps its time stamp: no grid built from it looks out of date"
check_rc "$(has01 "${TMP}/i_fields_add.log" 'keeps its time stamp')" "... which the tool says"
check_rc "$([ "$(layout_df "${OLDF}")" = "ONE PIECE" ] && echo 0 || echo 1)" "... and its datasets, grown in pieces before, are in one piece now"
rc=0; merge_df OLD --verify >"${TMP}/i_fields_verify_added.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... --verify of it now: exit 0 (${rc})"
rc=0; merge_df OLD --add-fields >"${TMP}/i_fields_add2.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "the same again: exit 0 (${rc})"
check_rc "$(has01 "${TMP}/i_fields_add2.log" 'already carries every per-particle dataset')" "... nothing to add"
OTHF="${DF}/OTHER/snapdir_099/combined_099.hdf5"
sum0="$(cksum < "${OTHF}")"
rc=0; merge_df OTHER --add-fields >"${TMP}/i_fields_other.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ "$(cksum < "${OTHF}")" = "${sum0}" ] && [ ! -e "${OTHF}.partial" ] && echo 0 || echo 1)" \
      "--add-fields with chunks whose particles are not the file's, ID for ID: exit 1, the file untouched, no .partial (${rc})"
check_rc "$(has01 "${TMP}/i_fields_other.log" 'are not the ones at row 300 of combined_099\.hdf5')" "... the first row that differs named"
check_rc "$(has01 "${TMP}/i_fields_other.log" 'Done, with problems: not added: the other particle fields of snapshot 099')" "... and the summary says what was not added"
rc=0; merge_df OTHER --add-fields --delete-chunks >"${TMP}/i_fields_other_del.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ "$(cksum < "${OTHF}")" = "${sum0}" ] && [ "$(ls "${DF}"/OTHER/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "... the same beside --delete-chunks: exit 1, the file untouched, NO chunk deleted (${rc})"
line_of() { grep -n -- "$2" "$1" | head -1 | cut -d: -f1; }     # <log> <text>: the first line that holds it
rc=0; merge_df OLD --verify --delete-chunks >"${TMP}/i_fields_verify_del.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(ls "${DF}/OLD/snapdir_099" | grep -c '^snap_099')" = "0" ] && [ -f "${OLDF}" ] && echo 0 || echo 1)" \
      "--verify --delete-chunks of a file that holds everything: exit 0, its 4 chunks deleted (${rc})"
check_rc "$([ "$(line_of "${TMP}/i_fields_verify_del.log" 'VERIFIED')" -lt "$(line_of "${TMP}/i_fields_verify_del.log" 'Deleted 4 snapshot chunk file')" ] && echo 0 || echo 1)" \
      "... after the comparison"
rc=0; merge_df OLD --verify >"${TMP}/i_fields_verify_nochunk.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "--verify with no chunks left: exit 0 (${rc})"
check_rc "$(has01 "${TMP}/i_fields_verify_nochunk.log" 'Not compared, no chunks left: snapshot 099')" "... the snapshot named as not compared"
rc=0; merge_df OLD2 --add-fields --delete-chunks >"${TMP}/i_fields_add_del.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(ls "${DF}/OLD2/snapdir_099" | grep -c '^snap_099')" = "0" ] && [ "$(layout_df "${DF}/OLD2/snapdir_099/combined_099.hdf5")" = "ONE PIECE" ] && echo 0 || echo 1)" \
      "--add-fields --delete-chunks of a file merged without the fields: exit 0, its chunks deleted (${rc})"
check_rc "$([ "$(line_of "${TMP}/i_fields_add_del.log" 'Added to:')" -lt "$(line_of "${TMP}/i_fields_add_del.log" 'VERIFIED')" ] && [ "$(line_of "${TMP}/i_fields_add_del.log" 'VERIFIED')" -lt "$(line_of "${TMP}/i_fields_add_del.log" 'Deleted 4 snapshot chunk file')" ] && echo 0 || echo 1)" \
      "... in this order: added, compared with the chunks, deleted"
rc=0; merge_df OLD2 --add-fields --delete-chunks >"${TMP}/i_fields_add_del2.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... the same again, the chunks gone: exit 0 (${rc})"
check_rc "$(has01 "${TMP}/i_fields_add_del2.log" 'Not compared, no chunks left: snapshot 099')" "... the snapshot named as not compared"
rm -f "${DF}"/OTHER/snapdir_099/snap_099.*.hdf5
rc=0; merge_df OLD --add-fields >"${TMP}/i_fields_nochunk.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "--add-fields with no chunks left and the fields in the file: exit 0, nothing to add (${rc})"
check_rc "$(has01 "${TMP}/i_fields_nochunk.log" 'no chunks are left: nothing to add')" "... and says so"
rc=0; merge_df OTHER --add-fields >"${TMP}/i_fields_nochunk2.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(cksum < "${OTHF}")" = "${sum0}" ] && echo 0 || echo 1)" "... and NO other field in the file: exit 0, the file untouched (${rc})"
check_rc "$(has01 "${TMP}/i_fields_nochunk2.log" 'Nothing to add from, no chunks left: snapshot 099')" "... the snapshot named: nothing to add from"
rc=0; merge_df UNITS >"${TMP}/i_fields_units.log" 2>&1 || rc=$?
"${PY}" - "${DF}/UNITS/snapdir_099" >"${TMP}/i_fields_units.check" 2>&1 <<'PYEOF' || true
import sys
from pathlib import Path
import h5py, numpy as np
d, h = Path(sys.argv[1]), 0.6774
get = lambda name: np.concatenate([h5py.File(d / f"snap_099.{c}.hdf5", "r")["PartType1"][name][...] for c in range(4)])
times = lambda name, p: (get(name).astype(np.float64) * h ** p).astype(np.float32)
with h5py.File(d / "combined_099.hdf5", "r") as f:
    g = f["PartType1"]
    ok = [np.array_equal(g["SubfindVelDisp"][...], times("SubfindVelDisp", 1.0)) and float(g["SubfindVelDisp"].attrs["h_power_applied"]) == 1.0,
          np.array_equal(g["SubfindHsml"][...], times("SubfindHsml", -1.0)) and float(g["SubfindHsml"].attrs["h_power_applied"]) == -1.0
          and "h_scaling" not in g["SubfindHsml"].attrs,
          float(g["Potential"].attrs["to_cgs"]) == 2.5 and float(g["Potential"].attrs["h_scaling"]) == 0.0,
          np.array_equal(g["Unknown"][...], get("Unknown")) and len(g["Unknown"].attrs) == 0]
print("UNITS OK" if all(ok) else f"UNITS BAD {ok}")
PYEOF
check_rc "$([ "${rc}" = "0" ] && grep -q '^UNITS OK' "${TMP}/i_fields_units.check" && echo 0 || echo 1)" \
      "the power of h: a chunk's own h_scaling before the table, the table when it has none, the other unit attributes kept; a field nothing knows copied raw (${rc})"
check_rc "$(has01 "${TMP}/i_fields_units.log" 'WARNING: /PartType1/Unknown cannot be made h-free')" "... with the loud warning"
rc=0; merge_df UNITS --verify >"${TMP}/i_fields_units_verify.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... and --verify agrees with what the merge wrote: exit 0 (${rc})"
# rows that shift between two chunks while the totals still add up: one row a particle is checked chunk by chunk
"${PY}" - "${DF}/UNITS/snapdir_099" <<'PYEOF'
import sys, h5py, numpy as np
for c, rows in ((1, 299), (2, 301)):
    with h5py.File(f"{sys.argv[1]}/snap_099.{c}.hdf5", "r+") as f:
        del f["PartType1"]["Potential"]
        f["PartType1"].create_dataset("Potential", data=np.ones(rows, dtype=np.float32)).attrs["h_scaling"] = 0.0
PYEOF
sum0="$(cksum < "${DF}/UNITS/snapdir_099/combined_099.hdf5")"
rc=0; merge_df UNITS --verify --delete-chunks >"${TMP}/i_fields_rows_verify.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ "$(ls "${DF}"/UNITS/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "a field with 299 rows in one chunk and 301 in the next: --verify --delete-chunks exit 1, NO chunk deleted (${rc})"
check_rc "$(has01 "${TMP}/i_fields_rows_verify.log" 'snap_099\.1\.hdf5: PartType1/Potential holds 299 rows, the chunk.s particles 300')" "... the chunk and its rows named"
rc=0; merge_df UNITS >"${TMP}/i_fields_rows_merge.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ "$(cksum < "${DF}/UNITS/snapdir_099/combined_099.hdf5")" = "${sum0}" ] && [ ! -e "${DF}/UNITS/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "... a merge of those chunks is refused: exit 1, the combined file that was there untouched, no .partial (${rc})"
check_rc "$(has01 "${TMP}/i_fields_rows_merge.log" 'snap_099\.1\.hdf5 holds 299 rows of PartType1/Potential, its particles 300')" "... the same rows named"
rc=0; merge_df GAS >"${TMP}/i_fields_gas.log" 2>&1 || rc=$?
rc2=0; merge_df GAS --verify --delete-chunks >"${TMP}/i_fields_gas_verify.log" 2>&1 || rc2=$?
check_rc "$([ "${rc}" = "0" ] && [ "${rc2}" = "1" ] && [ "$(ls "${DF}"/GAS/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" \
      "chunks whose header counts gas too: the merge works (${rc}); --verify --delete-chunks exit 1, NO chunk deleted (${rc2})"
check_rc "$(has01 "${TMP}/i_fields_gas_verify.log" 'the chunks hold 5 particles of PartType0, which a merged snapshot does not carry')" "... the type the file lacks named"
# group catalogues: the header's NumFiles, an empty chunk without Group/Subhalo groups, a missing chunk
rc=0; merge_dl --groupcats --snaps 99 >"${TMP}/i_gc_ok.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ -f "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5" ] && [ ! -e "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a group catalogue of 4 chunks, one with no groups at all, merges: exit 0, no .partial (${rc})"
check_rc "$(has01 "${TMP}/i_gc_ok.log" 'Groups: 12/12  Subhalos: 12/12  \(verified\)')" "... its totals verify"
rm -f "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5"
mv "${DL}/groups_099/fof_subhalo_tab_099.1.hdf5" "${TMP}/gc1.keep"
rc=0; merge_dl --groupcats --snaps 99 >"${TMP}/i_gc_missing.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && [ ! -e "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5" ] && echo 0 || echo 1)" "a group-catalogue chunk missing: exit 1, no combined file (${rc})"
check_rc "$(has01 "${TMP}/i_gc_missing.log" "Problem: Can't find 1 subfile\(s\) \(the header promises 4\)")" "... the header's NumFiles is the yardstick"
check_rc "$(has01 "${TMP}/i_gc_missing.log" 'Done, with problems: no combined file for group catalogue 099')" "... named in the summary"
mv "${TMP}/gc1.keep" "${DL}/groups_099/fof_subhalo_tab_099.1.hdf5"
# the download script: a fake wget (no network) that fails for snapshot-98; a truncated chunk aged past
# the one-minute guard is removed before wget runs, a fresh one is kept
DL2="${TMP}/DL2/SIMX"
mkdir -p "${DL2}/snapdir_099" "${TMP}/fakebin"
cp "${TMP}/snap_099.2.keep" "${DL2}/snapdir_099/snap_099.0.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL2}/snapdir_099/snap_099.1.hdf5"
touch -t 202001010000 "${DL2}/snapdir_099/snap_099.1.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL2}/snapdir_099/snap_099.2.hdf5"          # written just now
touch -t 209901010000 "${DL2}/snapdir_099/snap_099.2.hdf5"   # ... and stays "just now" however loaded the machine (the guard keeps a chunk younger than 60 s; a slow run would age a real 'now' past that)
cat > "${TMP}/fakebin/wget" <<'WEOF'
#!/usr/bin/env bash
# no network: records every argument and URL, fails for snapshot-98. '-O - <listing>?format=json' prints
# the product's listing with chunk 0 as an http:// link (as the real API does since 2026-11); '-i -' reads
# the chunk links from stdin, REFUSES a plain-http one, and leaves each chunk (a 'fake' byte file, named as
# the API's Content-Disposition names it) in -P <dir> unless that file exists (as -nc would); a direct URL
# (ics.hdf5) is fetched the same way. FAKE_WGET_EMPTY=1: the listing names nothing. FAKE_WGET_COUNT=<file>: a
# per-run counter of the 'wget -i -' calls; the first FAKE_WGET_CHUNK_FAIL (default 0) of them answer every link with
# a 403 and exit 8, as wget does (the data host refusing a chunk that a later try gets).
url=""; dest=""; out=""; list=""; all="$*"
while [ $# -gt 0 ]; do case "$1" in http*) url="$1" ;; -P) dest="$2"; shift ;; -O) out="$2"; shift ;; -i) list="$2"; shift ;; esac; shift; done
echo "ARGS: $all" >> "${FAKE_WGET_LOG}"; [ -n "$url" ] && echo "$url" >> "${FAKE_WGET_LOG}"
case "$url" in *snapshot-98*) echo "fake wget: 404 for $url" >&2; exit 8 ;; esac
if [ "$out" = "-" ]; then
    [ -n "${FAKE_WGET_EMPTY:-}" ] && { echo '{"files":[]}'; exit 0; }
    base="${url%%\?*}"; base="${base%/}"
    echo "{\"files\":[\"$(echo "$base" | sed 's|^https://|http://|').0.hdf5\"]}"; exit 0
fi
fetch_one() {
    local u="$1" f name=""
    f="${u##*/}"
    case "$f" in
        snapshot-*) name=$(echo "$f" | sed -n 's|^snapshot-\([0-9]*\)\.\([0-9]*\)\.hdf5$|\1 \2|p' | { read -r n k; printf "snap_%03d.%d.hdf5" "$n" "$k"; }) ;;
        groupcat-*) name=$(echo "$f" | sed -n 's|^groupcat-\([0-9]*\)\.\([0-9]*\)\.hdf5$|\1 \2|p' | { read -r n k; printf "fof_subhalo_tab_%03d.%d.hdf5" "$n" "$k"; }) ;;
        sublink.*)  name="tree_extended.${f#sublink.}" ;;
        ics.hdf5)   name="ics.hdf5" ;;
    esac
    [ -n "$name" ] && [ -n "$dest" ] && [ ! -e "$dest/$name" ] && echo fake > "$dest/$name"
    return 0
}
if [ "$list" = "-" ]; then
    call=0; refused=""
    if [ -n "${FAKE_WGET_COUNT:-}" ]; then
        call=$(( $(cat "${FAKE_WGET_COUNT}" 2>/dev/null || echo 0) + 1 )); echo "${call}" > "${FAKE_WGET_COUNT}"
    fi
    while read -r u; do
        echo "$u" >> "${FAKE_WGET_LOG}"
        case "$u" in https://*) ;; *) echo "fake wget: refusing non-https $u" >&2; exit 9 ;; esac
        if [ "${call}" -gt 0 ] && [ "${call}" -le "${FAKE_WGET_CHUNK_FAIL:-0}" ]; then
            echo "fake wget: ERROR 403: Forbidden for $u (call ${call})" >&2; refused=1; continue
        fi
        fetch_one "$u"
    done
    [ -n "${refused}" ] && exit 8
    exit 0
fi
fetch_one "$url"
exit 0
WEOF
chmod +x "${TMP}/fakebin/wget"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls.log" TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 98 99 >"${TMP}/i_download.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "download_snapshots.sh exits 1 when a fetch failed, not 0 (${rc})"
check_rc "$(has01 "${TMP}/i_download.log" 'Error downloading snapshot-98')" "... the failed endpoint is reported"
check_rc "$(has01 "${TMP}/i_download.log" 'Download finished with 2 error')" "... and the last line counts it and the kept unreadable chunk (no 'Download complete!')"
check_rc "$(has01 "${TMP}/i_download.log" 'Error downloading snapshot-99: an unreadable chunk is still in place')" "... a kept unreadable chunk is an error of this run (wget -nc would skip it for ever)"
check_rc "$(has01 "${TMP}/i_urls.log" '^ARGS: .*--https-only')" "the fetches are https-only (redirects with the key header too)"
check_rc "$(has01 "${TMP}/i_urls.log" '^https://www.tng-project.org/api/SIMX/files/snapshot-99\.0\.hdf5$')" "the listing's http:// chunk link is fetched, upgraded to https"
check_rc "$(lacks "${TMP}/i_urls.log" '^http://')" "... and no plain-http URL is ever requested"
check_rc "$(has01 "${TMP}/i_urls.log" '^ARGS: .*--header=API-Key: dummykey')" "... and the key travels as a header"
check_rc "$(lacks "${TMP}/i_download.log" 'Download complete!')" "... really no 'Download complete!'"
check_rc "$([ "$(grep -c '^https://www.tng-project.org/api/SIMX/files/snapshot-9[89]/' "${TMP}/i_urls.log")" = "2" ] && echo 0 || echo 1)" \
      "both fetches went to https (the key is a header, never sent over plain http)"
check_rc "$(lacks "${TMP}/i_download.log" 'dummykey')" "the key is not in the output"
check_rc "$([ ! -e "${DL2}/snapdir_099/snap_099.1.hdf5" ] && echo 0 || echo 1)" "an old truncated chunk is removed before wget runs (wget -nc would skip it for ever)"
check_rc "$(has01 "${TMP}/i_download.log" 'Removed incomplete chunk snap_099\.1\.hdf5')" "... and the log says which"
check_rc "$([ -f "${DL2}/snapdir_099/snap_099.0.hdf5" ] && echo 0 || echo 1)" "a complete chunk stays"
check_rc "$([ -f "${DL2}/snapdir_099/snap_099.2.hdf5" ] && echo 0 || echo 1)" "a truncated chunk written less than a minute ago stays (another download may be writing it)"
check_rc "$(has01 "${TMP}/i_download.log" 'snap_099\.2\.hdf5 is incomplete but was written less than a minute ago')" "... and is noted"
touch -t 202001010000 "${DL2}/snapdir_099/snap_099.2.hdf5"                                  # a minute later...
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls2.log" TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 99 >"${TMP}/i_download_ok.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a run with no failure exits 0 (${rc})"
check_rc "$(has01 "${TMP}/i_download_ok.log" 'Removed incomplete chunk snap_099\.2\.hdf5')" "... the chunk that aged past the guard is removed now"
check_rc "$(has01 "${TMP}/i_download_ok.log" 'Download complete!')" "... and ends with 'Download complete!'"
# the other three products, with an aged truncated file each: pruned, fetched over https with the key as a header
DL3="${TMP}/DL3/SIMX"; mkdir -p "${DL3}/Merger Trees" "${DL3}/groups_099"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/ics.hdf5";                      touch -t 202001010000 "${DL3}/ics.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/Merger Trees/tree_extended.0.hdf5"; touch -t 202001010000 "${DL3}/Merger Trees/tree_extended.0.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5"; touch -t 202001010000 "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5"
rc_i=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_i.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -i -d "${DL3}" -s SIMX-Dark >"${TMP}/i_dl_ics.log" 2>&1 || rc_i=$?
rc_t=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_t.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -t -d "${DL3}" -s SIMX >"${TMP}/i_dl_trees.log" 2>&1 || rc_t=$?
rc_c=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_c.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -c -d "${DL3}" -s SIMX 99 >"${TMP}/i_dl_gc.log" 2>&1 || rc_c=$?
check_rc "$([ "${rc_i}" = "0" ] && [ "${rc_t}" = "0" ] && [ "${rc_c}" = "0" ] && echo 0 || echo 1)" "-i, -t and -c runs with a working fetch exit 0 (${rc_i} ${rc_t} ${rc_c})"
check_rc "$(has01 "${TMP}/i_urls_i.log" '^https://www.tng-project.org/api/SIMX/files/ics.hdf5$')" "-i: https, and '-Dark' stripped for the API name"
check_rc "$(has01 "${TMP}/i_urls_t.log" '^https://www.tng-project.org/api/SIMX/files/sublink/')" "-t: https"
check_rc "$(has01 "${TMP}/i_urls_c.log" '^https://www.tng-project.org/api/SIMX/files/groupcat-99/')" "-c: https"
check_rc "$([ "$(cat "${TMP}"/i_urls_i.log "${TMP}"/i_urls_t.log "${TMP}"/i_urls_c.log | grep -c 'header=API-Key: dummykey')" = "5" ] && echo 0 || echo 1)" "the key is a header in all five wget calls (ics; listing + chunks for trees and catalogues)"
check_rc "$([ -f "${DL3}/Merger Trees/tree_extended.0.hdf5" ] && [ -f "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5" ] && echo 0 || echo 1)" "-t and -c fetch the chunks their listings name"
check_rc "$(lacks "${TMP}/i_urls_i.log" '^https.*dummykey')" "... never in a URL"
check_rc "$(has01 "${TMP}/i_dl_ics.log" 'Removed incomplete chunk ics\.hdf5')" "-i prunes an aged truncated ics.hdf5 first"
check_rc "$(has01 "${TMP}/i_dl_trees.log" 'Removed incomplete chunk tree_extended\.0\.hdf5')" "-t prunes an aged truncated tree chunk first"
check_rc "$(has01 "${TMP}/i_dl_gc.log" 'Removed incomplete chunk fof_subhalo_tab_099\.0\.hdf5')" "-c prunes an aged truncated catalogue chunk first"
# a fetch that exits 0 but fetched nothing is an error (the listing named no chunk)
DL4="${TMP}/DL4/SIMX"; mkdir -p "${DL4}"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls4.log" FAKE_WGET_EMPTY=1 TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL4}" -s SIMX 99 >"${TMP}/i_dl_empty.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "wget exiting 0 with no chunk fetched: exit 1 (${rc})"
check_rc "$(has01 "${TMP}/i_dl_empty.log" 'Error downloading snapshot-99: no chunk was fetched')" "... and named"
# the chunk list is tried up to three times (fetch_listing): a chunk the data host refuses once (a 403 a rerun does
# not get) costs a retry, not the snapshot; a chunk refused every time is still an error
DL5="${TMP}/DL5/SIMX"; mkdir -p "${DL5}"; rm -f "${TMP}/i_retry.count"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls5.log" FAKE_WGET_COUNT="${TMP}/i_retry.count" FAKE_WGET_CHUNK_FAIL=1 \
    TNG_API_KEY=dummykey scripts/download_snapshots.sh -d "${DL5}" -s SIMX 99 >"${TMP}/i_dl_retry.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a chunk refused once, then served: exit 0 (${rc})"
check_rc "$(has01 "${TMP}/i_dl_retry.log" 'a chunk failed; trying the listing again \(attempt 2 of 3\)$')" "... after one retry, said as 'attempt 2 of 3'"
check_rc "$([ "$(cat "${TMP}/i_retry.count" 2>/dev/null)" = "2" ] && [ "$(lacks "${TMP}/i_dl_retry.log" 'attempt 3 of 3')" = 0 ] && echo 0 || echo 1)" \
      "... the chunk list went to wget twice, not a third time ($(cat "${TMP}/i_retry.count" 2>/dev/null || echo none))"
check_rc "$([ -f "${DL5}/snapdir_099/snap_099.0.hdf5" ] && echo 0 || echo 1)" "... the chunk is in place"
check_rc "$([ "$(has01 "${TMP}/i_dl_retry.log" '^  snapshot-99 downloaded successfully$')" = 0 ] && [ "$(has01 "${TMP}/i_dl_retry.log" '^Download complete!$')" = 0 ] \
         && [ "$(lacks "${TMP}/i_dl_retry.log" 'Error downloading')" = 0 ] && echo 0 || echo 1)" "... and the snapshot counts as downloaded: 'Download complete!', no error"
DL6="${TMP}/DL6/SIMX"; mkdir -p "${DL6}"; rm -f "${TMP}/i_refused.count"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls6.log" FAKE_WGET_COUNT="${TMP}/i_refused.count" FAKE_WGET_CHUNK_FAIL=99 \
    TNG_API_KEY=dummykey scripts/download_snapshots.sh -d "${DL6}" -s SIMX 99 >"${TMP}/i_dl_refused.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a chunk refused every time: exit 1, not 0 (${rc})"
check_rc "$([ "$(cat "${TMP}/i_refused.count" 2>/dev/null)" = "3" ] && [ "$(has01 "${TMP}/i_dl_refused.log" '\(attempt 3 of 3\)$')" = 0 ] && echo 0 || echo 1)" \
      "... after three tries, no fourth ($(cat "${TMP}/i_refused.count" 2>/dev/null || echo none))"
check_rc "$([ "$(has01 "${TMP}/i_dl_refused.log" '^  Error downloading snapshot-99$')" = 0 ] && [ "$(has01 "${TMP}/i_dl_refused.log" 'Download finished with 1 error')" = 0 ] \
         && [ "$(lacks "${TMP}/i_dl_refused.log" 'Download complete!')" = 0 ] && echo 0 || echo 1)" "... named as a failed snapshot and counted in the last line"
check_rc "$([ ! -e "${DL6}/snapdir_099/snap_099.0.hdf5" ] && echo 0 || echo 1)" "... and no chunk was left in place"
check_rc "$(lacks "${TMP}/i_dl_refused.log" 'dummykey')" "... the key in none of the retries' output"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls3.log" TNG_API_KEY=dummykey PYTHON=/nonexistent/python3 \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 99 >"${TMP}/i_download_nopy.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "without a python with h5py the check is skipped, the download still runs (${rc})"
check_rc "$(has01 "${TMP}/i_download_nopy.log" 'has no h5py; existing chunks are not checked')" "... and says so"

echo ""
echo "(J) --help runs nothing, --version prints the revision, DRY_RUN=1 prints the commands and runs nothing."
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/run_ps_pipeline.sh; do
    rc=0; "${scr}" --help >"${TMP}/j_help_${scr##*/}.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "${scr##*/} --help exits 0 (${rc})"
    check_rc "$(lacks "${TMP}/j_help_${scr##*/}.log" 'Processing snapshot|RUNNING:|Slice: axis')" "... and starts nothing"
done
check_rc "$(has01 "${TMP}/j_help_run_ps_pipeline.sh.log" 'COMPUTE ONLY')" "run_ps_pipeline.sh --help prints its header (and does not start the whole batch)"
check_rc "$(has01 "${TMP}/j_help_run_dtfe.sh.log" '^Usage: ')" "run_dtfe.sh --help ends with the usage line"
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_SIM "${scr}" 99 >"${TMP}/j_nosim_${scr##*/}.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "2" ] && [ "$(has01 "${TMP}/j_nosim_${scr##*/}.log" 'no simulation given: pass -s SIMULATION')" = 0 ] && echo 0 || echo 1)" \
          "${scr##*/} without -s or DTFE_SIM refuses with exit 2: there is no default simulation (${rc})"
done
# no default data root either (config.sh require_data_root): every driver refuses before it reads or writes anything
NOROOT='no data root: set DTFE_DATA_ROOT to the folder that holds the simulations \(e\.g\. DTFE_DATA_ROOT=~/IllustrisTNG\)\.'
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_DATA_ROOT TNG_API_KEY=dummykey "${scr}" -s HASIC 99 >"${TMP}/j_noroot_${scr##*/}.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "2" ] && [ "$(has01 "${TMP}/j_noroot_${scr##*/}.log" "^Error: ${NOROOT}$")" = 0 ] \
             && [ "$(lacks "${TMP}/j_noroot_${scr##*/}.log" 'Processing snapshot|RUNNING:|Downloading|Data directory')" = 0 ] && echo 0 || echo 1)" \
          "${scr##*/} without DTFE_DATA_ROOT refuses with exit 2 and says so, before anything runs: there is no default data root (${rc})"
done
# -d names the simulation's folder, so it stands in for the data root: without a simulation the next refusal is the
# missing simulation, not the data root (nothing runs or downloads)
rc_d=""
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_DATA_ROOT -u DTFE_SIM TNG_API_KEY=dummykey "${scr}" -d "${TMP}/HASIC" 99 >"${TMP}/j_noroot_d.log" 2>&1 || rc=$?
    [ "$(has01 "${TMP}/j_noroot_d.log" 'no simulation given')" = 0 ] && [ "$(lacks "${TMP}/j_noroot_d.log" "${NOROOT}")" = 0 ] || rc="${rc}-msg"
    rc_d="${rc_d} ${rc}"
done
check_rc "$([ "${rc_d}" = " 2 2 2" ] && echo 0 || echo 1)" "... -d DATA_DIR stands in for it: the refusal is then the missing simulation (${rc_d# })"
# the option loop the three share (config.sh driver_options): an unknown option or one missing its value prints
# getopts' message and the usage and exits 1, -h after other options prints the usage and exits 0 (nothing runs)
rc_o=""
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; TNG_API_KEY=dummykey "${scr}" -s HASIC -Z 99 >"${TMP}/j_opt1.log" 2>&1 || rc=$?
    [ "$(has01 "${TMP}/j_opt1.log" 'option -- .?Z')" = 0 ] && [ "$(has01 "${TMP}/j_opt1.log" '^Usage: ')" = 0 ] || rc="${rc}-msg"
    rc2=0; TNG_API_KEY=dummykey "${scr}" -s >"${TMP}/j_opt2.log" 2>&1 || rc2=$?
    [ "$(has01 "${TMP}/j_opt2.log" 'requires an argument -- .?s')" = 0 ] && [ "$(has01 "${TMP}/j_opt2.log" '^Usage: ')" = 0 ] || rc2="${rc2}-msg"
    rc3=0; TNG_API_KEY=dummykey "${scr}" -s HASIC -h 99 >"${TMP}/j_opt3.log" 2>&1 || rc3=$?
    [ "$(has01 "${TMP}/j_opt3.log" '^Usage: ')" = 0 ] && [ "$(lacks "${TMP}/j_opt3.log" 'Processing snapshot|Downloading|Data directory')" = 0 ] || rc3="${rc3}-msg"
    rc_o="${rc_o} ${rc}/${rc2}/${rc3}"
done
check_rc "$([ "${rc_o}" = " 1/1/0 1/1/0 1/1/0" ] && echo 0 || echo 1)" \
      "an unknown option or a missing value: the usage, exit 1; -h after other options: the usage, exit 0 (${rc_o# })"
# a driver's own options mix with the shared ones in any order, clustered too (-me); -m asks for the GPU through the
# binary's environment (DTFE_GPU=1), so the planned command names no '--gpu' and the banner says '-m'
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -g 16 -n 2 -d "${TMP}/HASIC" -me -s HASIC 99 >"${TMP}/j_opt4.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(has01 "${TMP}/j_opt4.log" 'HASIC/snapdir_099/combined_099\.hdf5 .*--grid +16 .*--lambda-th +0\.3 +--exact( |$)')" = 0 ] \
         && [ "$(lacks "${TMP}/j_opt4.log" '^ +\./PS-DTFE.*( --subsamples| --gpu( |$))')" = 0 ] \
         && grep -aqF 'Deposit: the GPU (-m)   caustics: ' "${TMP}/j_opt4.log" \
         && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: -e asks for the exact deposit' "${TMP}/j_opt4.log")" = 1 ] \
         && echo 0 || echo 1)" "... -g, -n, -m and -e mix with -d and -s in any order, -me clustered: -e keeps the exact deposit, -n 2 said ignored, -m named in the banner and not on the command (exit ${rc})"
# the long forms are the letters (config.sh driver_long_options): the same planned command, token for token, and the
# same device in the banner (--gpu is -m)
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" --grid 16 --subsamples=2 --sim-dir "${TMP}/HASIC" --gpu --exact --sim HASIC 99 \
    >"${TMP}/j_opt5.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt5.log")" = "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt4.log")" ] \
         && [ -n "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt5.log")" ] \
         && [ "$(grep -a '^Deposit: ' "${TMP}/j_opt5.log")" = "$(grep -a '^Deposit: ' "${TMP}/j_opt4.log")" ] && echo 0 || echo 1)" \
      "... --grid, --subsamples=K, --sim-dir, --gpu, --exact and --sim plan the same command and banner as the letters (exit ${rc})"
# --sampled: the sampled deposit, named; with a count, the count too (in place of -e's --exact)
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" --grid 16 --subsamples=2 --sim-dir "${TMP}/HASIC" --gpu --sampled --sim HASIC 99 \
    >"${TMP}/j_opt5s.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(has01 "${TMP}/j_opt5s.log" 'combined_099\.hdf5 .*--grid +16 .*--lambda-th +0\.3 +--sampled +--subsamples +2 ')" = 0 ] \
         && [ "$(lacks "${TMP}/j_opt5s.log" '^ +\./PS-DTFE.*--exact( |$)')" = 0 ] && echo 0 || echo 1)" \
      "... --sampled with --subsamples=2: --sampled --subsamples 2, no --exact (exit ${rc})"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 scripts/run_dtfe.sh --sim HASIC --grid=16 --subsamples 2 99 >"${TMP}/j_opt6.log" 2>&1 || rc=$?
rc2=0; "${DRIVER}" -s HASIC --bogus 99 >"${TMP}/j_opt7.log" 2>&1 || rc2=$?
rc3=0; TNG_API_KEY=dummykey scripts/download_snapshots.sh -s HASIC --gpu 99 >"${TMP}/j_opt8.log" 2>&1 || rc3=$?
rc4=0; TNG_API_KEY=dummykey scripts/download_snapshots.sh -s HASIC --sampled 99 >"${TMP}/j_opt9.log" 2>&1 || rc4=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(has01 "${TMP}/j_opt6.log" '--grid +16 .*--sampled +--subsamples +2')" = 0 ] \
         && [ "${rc2}" = "1" ] && [ "$(has01 "${TMP}/j_opt7.log" "unknown option '--bogus'")" = 0 ] && [ "$(has01 "${TMP}/j_opt7.log" '^Usage: ')" = 0 ] \
         && [ "${rc3}" = "1" ] && [ "$(has01 "${TMP}/j_opt8.log" "unknown option '--gpu'")" = 0 ] \
         && [ "${rc4}" = "1" ] && [ "$(has01 "${TMP}/j_opt9.log" "unknown option '--sampled'")" = 0 ] && echo 0 || echo 1)" \
      "... run_dtfe.sh takes them too (a count alone: --sampled --subsamples 2); an unknown long option, or one the script lacks (download_snapshots.sh --gpu, --sampled): the usage, exit 1 (${rc} ${rc2} ${rc3} ${rc4})"
# --run-mode MODE (getopts' R): both drivers plan the mode it names; download_snapshots.sh runs no binary and has none
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s HASIC -g 16 --run-mode gentle 99 >"${TMP}/j_rm1.log" 2>&1 || rc=$?
rc2=0; env "${COMMON_ENV[@]}" DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 --run-mode=performance 99 >"${TMP}/j_rm2.log" 2>&1 || rc2=$?
rc3=0; TNG_API_KEY=dummykey scripts/download_snapshots.sh -s HASIC --run-mode gentle 99 >"${TMP}/j_rm3.log" 2>&1 || rc3=$?
check_rc "$([ "${rc}" = "0" ] && [ "$(has01 "${TMP}/j_rm1.log" 'combined_099\.hdf5 .*--run-mode +gentle ')" = 0 ] \
         && [ "${rc2}" = "0" ] && [ "$(has01 "${TMP}/j_rm2.log" 'combined_099\.hdf5 .*--run-mode +performance ')" = 0 ] \
         && [ "${rc3}" = "1" ] && [ "$(has01 "${TMP}/j_rm3.log" "unknown option '--run-mode'")" = 0 ] && [ "$(has01 "${TMP}/j_rm3.log" '^Usage: ')" = 0 ] \
         && echo 0 || echo 1)" \
      "... --run-mode MODE (and --run-mode=MODE) plans that mode in both drivers; download_snapshots.sh has no run mode: the usage, exit 1 (${rc} ${rc2} ${rc3})"
rc=0; DTFE_DATA_ROOT= scripts/run_ps_dtfe.sh -s HASIC 99 >"${TMP}/j_noroot_empty.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "2" ] && [ "$(has01 "${TMP}/j_noroot_empty.log" "${NOROOT}")" = 0 ] && echo 0 || echo 1)" \
      "... an empty DTFE_DATA_ROOT= is no data root either (${rc})"
rc=0; env -u DTFE_DATA_ROOT DRY_RUN=1 SIMS=all scripts/run_ps_pipeline.sh >"${TMP}/j_noroot_pipe.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "2" ] && [ "$(has01 "${TMP}/j_noroot_pipe.log" "^Error: ${NOROOT}$")" = 0 ] \
         && [ "$(lacks "${TMP}/j_noroot_pipe.log" '^Simulations:')" = 0 ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh without DTFE_DATA_ROOT refuses with exit 2 before SIMS=all looks for simulations (${rc})"
rc_h=""
for c in "scripts/run_dtfe.sh --help" "scripts/run_ps_dtfe.sh --help" "scripts/run_ps_pipeline.sh --help" "scripts/download_snapshots.sh -h"; do
    rc=0; env -u DTFE_DATA_ROOT ${c} >"${TMP}/j_noroot_help.log" 2>&1 || rc=$?
    rc_h="${rc_h} ${rc}"
done
check_rc "$([ "${rc_h}" = " 0 0 0 0" ] && echo 0 || echo 1)" "... while --help (-h) of all four still works without it (${rc_h# })"
src="$(env -u DTFE_DATA_ROOT bash -c 'source scripts/config.sh; echo "sourced root=[${DATA_ROOT}]"' 2>&1)"
check_rc "$([ "${src}" = "sourced root=[]" ] && echo 0 || echo 1)" "... and sourcing config.sh without it exits nothing and leaves DATA_ROOT empty (${src})"
rc=0; scripts/run_ps_pipeline.sh bogus >"${TMP}/j_bogus.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "2" ] && echo 0 || echo 1)" "run_ps_pipeline.sh with an argument refuses with exit 2 (${rc})"
check_rc "$(has01 "${TMP}/j_bogus.log" 'takes no arguments')" "... and says why"
rc=0; scripts/run_ps_dtfe.sh --version >"${TMP}/j_version.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && [ -s "${TMP}/j_version.log" ] && echo 0 || echo 1)" "run_ps_dtfe.sh --version exits 0 with a revision"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s HASIC -g 16 99 98 >"${TMP}/j_dry.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "DRY_RUN=1 run_ps_dtfe.sh exits 0 with a planned snapshot (${rc})"
check_rc "$(has01 "${TMP}/j_dry.log" 'DRY RUN -- would run:')" "... prints the expanded command"
check_rc "$(has01 "${TMP}/j_dry.log" 'PS-DTFE.*combined_099\.hdf5.*--grid +16')" "... with the binary, the input and the grid"
check_rc "$(has01 "${TMP}/j_dry.log" 'DRY RUN: 1 snapshot\(s\) planned, 1 skipped')" "... one planned, the missing 098 skipped, nothing run"
check_rc "$(lacks "${TMP}/j_dry.log" 'RUNNING:')" "... the binary never ran"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s NOSUCH -g 16 99 >"${TMP}/j_dry2.log" 2>&1 || rc=$?
check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "DRY_RUN=1 with nothing to run exits 1 (${rc})"
if [ -x "${DTFE_BIN}" ]; then
    touch "${TMP}/j_marker"; sleep 1
    rc=0; DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/j_dry3.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "DRY_RUN=1 run_dtfe.sh exits 0 (${rc})"
    check_rc "$(has01 "${TMP}/j_dry3.log" 'DTFE.*combined_099\.hdf5.*--grid +16')" "... prints the standard binary's command"
    check_rc "$([ ! "${TMP}/HASIC/snapdir_099/output.runlog" -nt "${TMP}/j_marker" ] && echo 0 || echo 1)" "... and leaves the run log untouched"
fi

echo ""
echo "(K) the binaries' --help (-h) lists the two modes' options, --full-help the phase-space, GPU-share,"
echo "    point-evaluation and server options that are not in -h; no known misspelling."
"${PS_BIN}" --help >"${TMP}/k_help_ps.log" 2>&1 || true
"${DTFE_BIN}" --help >"${TMP}/k_help_dtfe.log" 2>&1 || true
# an entry is boost's line '  --name ...' (two blanks); a wrapped description line that starts with the name (the
# 80-column wrap indents it further) does not count. The short help (-h): the options a run of the two modes needs
for o in gpu exact sampled full-help lambda-th ps-merge mpc-unit ps-vertex-mass auto-tune-report run-mode; do
    check_rc "$(has01 "${TMP}/k_help_ps.log" "^  --${o}( |$)")" "PS-DTFE --help lists --${o}"
done
for o in gpu exact sampled full-help lambda-th mpc-unit auto-tune-report run-mode; do
    check_rc "$(has01 "${TMP}/k_help_dtfe.log" "^  --${o}( |$)")" "DTFE --help lists --${o}"
done
"${PS_BIN}" --full-help >"${TMP}/k_full_ps.log" 2>"${TMP}/k_full_ps.err" || true
"${DTFE_BIN}" --full-help >"${TMP}/k_full_dtfe.log" 2>&1 || true
# the options the scripts and the launcher use that are not in -h: in --full-help (whichever of its groups holds them)
for o in no-overlap grid-window subsamples ps-per-stream ps-caustics sample-points serve serve-progress tessellation-cache gpu-share; do
    check_rc "$(has01 "${TMP}/k_full_ps.log" "^  --${o}( |$)")" "PS-DTFE --full-help lists --${o}"
done
for o in no-overlap average-method subsamples sample-points serve; do
    check_rc "$(has01 "${TMP}/k_full_dtfe.log" "^  --${o}( |$)")" "DTFE --full-help lists --${o}"
done
check_rc "$([ "$(lacks "${TMP}/k_help_dtfe.log" '^ *--(ps-per-stream|ps-merge|ps-vertex-mass|ps-lagrangian-input)')" = 0 ] \
         && [ "$(lacks "${TMP}/k_full_dtfe.log" '^ *--(ps-per-stream|ps-merge|grid-window)')" = 0 ] && echo 0 || echo 1)" \
      "... and not PS-DTFE's phase-space options (short and full help)"
check_rc "$([ "$(has01 "${TMP}/k_full_ps.log" '^ *--exact +PS-DTFE')" = 0 ] && [ "$(has01 "${TMP}/k_full_ps.log" '^ *--sampled( |$)')" = 0 ] \
         && [ "$(has01 "${TMP}/k_full_dtfe.log" '^ *--sampled( |$)')" = 0 ] && [ ! -s "${TMP}/k_full_ps.err" ] && echo 0 || echo 1)" \
      "the full help of both binaries lists --exact and --sampled; PS-DTFE's writes nothing on stderr"
check_rc "$(lacks "${TMP}/k_full_ps.log" 'encompasing|documenation|vortivity|chuncks|coordiante|tesselation|specifed|suplied|strech|mannually|DMPC_UNIT')" \
      "the full help carries none of the known misspellings"
check_rc "$(has01 "${TMP}/k_full_ps.log" "'MPC_UNIT'")" "... and names the Makefile's MPC_UNIT, not DMPC_UNIT"
# a --config file's value gives way to the command line's for the same option: the file's 'lambda-th = 5' under the
# command line's '--lambda-th 0.3' runs at 0.3, the file alone at its 5 (every cell a void: a T-web the comparisons
# tell from 0.3's, which 0.1-1 are not on this snapshot). These runs start the binaries directly and were written for
# their earlier defaults (performance, the CPU unless '--gpu'): tests/lib.sh's two pins again, until (K) ends
export DTFE_RUN_MODE=performance DTFE_GPU=0
KS="${TMP}/HASIC/snapdir_099/combined_099.hdf5"
KC=(--grid 16 --periodic --input-format 105 --mpc-unit 1 --verbose 1 --field density_a tweb_a)
k_same() {   # <tag a> <tag b>: every output file of the two runs byte-identical (the logs aside)
    local f e n=0
    for f in "${TMP}/k_$1".*; do
        e="${f#"${TMP}/k_$1."}"
        case "${e}" in log|err) continue ;; esac
        cmp -s "${f}" "${TMP}/k_$2.${e}" || return 1
        n=$((n + 1))
    done
    [ "${n}" -gt 0 ]
}
k_run() {   # <binary> <tag> <options...>: run on the K snapshot, stdout in k_<tag>.log, stderr in k_<tag>.err; echo the exit
    local bin="$1" tag="$2" r=0
    shift 2
    "${bin}" "${KS}" "${TMP}/k_${tag}" "${KC[@]}" "$@" >"${TMP}/k_${tag}.log" 2>"${TMP}/k_${tag}.err" || r=$?
    echo "${r}"
}
printf 'lambda-th = 5\n' > "${TMP}/k_lam.cfg"
for b in ps dtfe; do
    if [ "${b}" = ps ]; then bin="${PS_BIN}"; extra=(--exact --no-overlap)
    else bin="${DTFE_BIN}"; extra=(); fi
    r0="$(k_run "${bin}" "${b}_ref" ${extra[@]+"${extra[@]}"} --lambda-th 0.3)"
    r1="$(k_run "${bin}" "${b}_lfile" ${extra[@]+"${extra[@]}"} --config "${TMP}/k_lam.cfg")"
    r2="$(k_run "${bin}" "${b}_lcmd" ${extra[@]+"${extra[@]}"} --config "${TMP}/k_lam.cfg" --lambda-th 0.3)"
    check_rc "$([ "${r0}" = 0 ] && [ "${r1}" = 0 ] && [ "${r2}" = 0 ] && ! k_same "${b}_ref" "${b}_lfile" \
             && k_same "${b}_ref" "${b}_lcmd" && echo 0 || echo 1)" \
          "${b}: '--lambda-th 0.3' on the command line over a --config file's 'lambda-th = 5' runs at 0.3; the file alone at 5 (${r0} ${r1} ${r2})"
done
unset DTFE_RUN_MODE DTFE_GPU     # (unset again, as at the top)

echo ""
echo "(L) scripts/install_lib.sh: the binaries kept through a failed rebuild."
LREPO="${TMP}/fake_repo"; rm -rf "${LREPO}"; mkdir -p "${LREPO}"
# the 2D double pair too: install.sh rebuilds it, so a failed build must be able to bring it back
( cd "${LREPO}" && printf 'old DTFE\n' > DTFE && printf 'old PS-DTFE\n' > PS-DTFE && printf 'old 2d\n' > DTFE-2d \
      && printf 'old 2d double\n' > DTFE-2d-double && printf 'old PS 2d double\n' > PS-DTFE-2d-double \
      && chmod +x DTFE PS-DTFE DTFE-2d DTFE-2d-double PS-DTFE-2d-double
  source "${ROOT_DIR}/scripts/install_lib.sh"
  n="$(backup_binaries "${LREPO}/.backup")"
  echo "backed up ${n}" > "${TMP}/l_backup.log"
  rm -f DTFE PS-DTFE DTFE-2d DTFE-2d-double PS-DTFE-2d-double; printf 'half-linked\n' > PS-DTFE   # 'make clean' then a failing build
  m="$(restore_binaries "${LREPO}/.backup")"
  echo "restored ${m}" >> "${TMP}/l_backup.log"
  printf 'line1\nline2\nld: symbol not found\n' > build.log
  build_failed_hint build.log 2 > "${TMP}/l_hint.log" )
check_rc "$(has01 "${TMP}/l_backup.log" '^backed up 5$')" "backup_binaries counts the five binaries present (of the eight names)"
check_rc "$(has01 "${TMP}/l_backup.log" '^restored 5$')" "restore_binaries puts the five back"
check_rc "$([ "$(cat "${LREPO}/PS-DTFE")" = "old PS-DTFE" ] && [ "$(cat "${LREPO}/DTFE")" = "old DTFE" ] && [ -x "${LREPO}/DTFE-2d" ] && echo 0 || echo 1)" \
      "... byte for byte, the half-linked one replaced, the mode kept"
check_rc "$([ "$(cat "${LREPO}/DTFE-2d-double")" = "old 2d double" ] && [ "$(cat "${LREPO}/PS-DTFE-2d-double")" = "old PS 2d double" ] \
         && [ -x "${LREPO}/PS-DTFE-2d-double" ] && echo 0 || echo 1)" "... the 2D double pair among them"
check_rc "$(has01 "${TMP}/l_hint.log" 'ld: symbol not found')" "build_failed_hint prints the log's tail"
check_rc "$(lacks "${TMP}/l_hint.log" '^line1$')" "... only the last N lines"
check_rc "$(has01 "${TMP}/l_hint.log" 'make deps-check')" "... and the usual causes"
# install.sh rebuilds the double, 2D and 2D double sets it finds before 'make clean' (else a re-run would throw
# them away). install.sh itself is never run here: its detection block, cut out between 'EXTRA_SETS=()'
# and the 'also to be rebuilt' line, is sourced in fake folders with info() stubbed out
sed -n '/^EXTRA_SETS=()/,/also to be rebuilt/p' scripts/install.sh > "${TMP}/l_extra_sets.sh"
check_rc "$(grep -q '^EXTRA_SETS=()' "${TMP}/l_extra_sets.sh" && grep -q 'DIM=2 DOUBLE=1' "${TMP}/l_extra_sets.sh" \
         && grep -q 'also to be rebuilt' "${TMP}/l_extra_sets.sh" && echo 0 || echo 1)" \
      "install.sh's extra-set block is found (EXTRA_SETS=() ... also to be rebuilt): a moved marker fails here, not silently"
extra_sets() {   # <PREC_ARG> <GPU_ARG> <binaries present...>: the make arguments of each set, one [..] per line
    local d="${TMP}/l_sets"; rm -rf "${d}"; mkdir -p "${d}"
    ( cd "${d}" && PREC_ARG="$1" && GPU_ARG="$2" && shift 2 && for b in DTFE PS-DTFE "$@"; do : > "${b}"; done
      info() { :; }
      source "${TMP}/l_extra_sets.sh" || true                   # its last line is false when there is no extra set
      for s in ${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}; do printf '[%s]' "${s}"; done )
}
got="$(extra_sets "" "METAL=1")"
check_rc "$([ -z "${got}" ] && echo 0 || echo 1)" "... the single pair alone: no extra set (${got:-none})"
got="$(extra_sets "" "METAL=1" DTFE-2d DTFE-double DTFE-2d-double)"
check_rc "$([ "${got}" = "[METAL=1 DOUBLE=1][DIM=2][DIM=2 DOUBLE=1]" ] && echo 0 || echo 1)" \
      "... DTFE-double, DTFE-2d and DTFE-2d-double present: the double set with the GPU backend, then DIM=2 and DIM=2 DOUBLE=1 (${got})"
got="$(extra_sets "" "" PS-DTFE-2d-double)"
check_rc "$([ "${got}" = "[DIM=2 DOUBLE=1]" ] && echo 0 || echo 1)" "... either binary of a pair is enough: PS-DTFE-2d-double alone (${got})"
got="$(extra_sets "DOUBLE=1" "")"
check_rc "$([ "${got}" = "[ DOUBLE=1]" ] && echo 0 || echo 1)" "... --double with no double pair yet builds it, CPU-only here (${got})"

echo ""
echo "(M) the run-log contract: a failed, stopped or killed re-run keeps the .runlog of the grids on disk."
# A driver that tees straight into <prefix>.runlog lets a re-run that fails early, or is stopped or killed,
# replace the log of the grids still on disk (the Runs browser then shows them as a failed run with the
# new settings). Copies of the drivers run here beside STAND-IN binaries (the drivers find theirs one level
# above their own folder), so a success, a failure, a stop and a SIGKILL are all deterministic and need no build.
MREPO="${TMP}/runlog_repo"; MDATA="${TMP}/runlog_data"; MPID="${TMP}/m_binary.pid"
mkdir -p "${MREPO}/scripts" "${MDATA}/SIM/snapdir_099"
cp scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/config.sh "${MREPO}/scripts/"
cat > "${MREPO}/DTFE" <<'FEOF'
#!/usr/bin/env bash
# stand-in binary. FAKE_MODE: ok (exit 0), fail (exit 1), hang (wait, in place of the binary, until killed),
# report (an auto-tune report line), write (as the real binary: '<out>.a_den' and '<out>.a_vel' written as
# '.part.<pid>' and renamed into place, a second later '<out>.badGridPoints0' rewritten IN PLACE when it exists).
# FAKE_ENV=1 adds two lines, the DTFE_GPU and the DTFE_RUN_MODE it was started with ('unset' when there is none),
# which the binaries read. Every run prints the header's 'run mode' line after its command, as the binaries do
# (run_ps_pipeline.sh recomputes a snapshot whose run log lacks it)
echo "RUNNING: $0 $*"
printf '\t run mode               : performance (the stand-in)\n'
printf 'fake attempt %s \033[1mcoloured\033[0m\n' "${FAKE_TAG:-}"
if [ -n "${FAKE_ENV:-}" ]; then
    echo "fake environment: DTFE_GPU=${DTFE_GPU-unset}"
    echo "fake environment: DTFE_RUN_MODE=${DTFE_RUN_MODE-unset}"
fi
case "${FAKE_MODE:-ok}" in
    ok)     echo "total wall time : 1 s"; exit 0 ;;
    write)  sleep 1
            [ -f "$2.badGridPoints0" ] && echo new > "$2.badGridPoints0"
            for s in a_den a_vel; do echo new > "$2.$s.part.$$" && mv -f "$2.$s.part.$$" "$2.$s"; done
            echo "total wall time : 1 s"; exit 0 ;;
    fail)   echo "~~~ ERROR ~~~ a fake failure"; exit 1 ;;
    hang)   echo $$ > "${FAKE_PID}"; exec sleep 60 ;;
    report) echo "AUTO-TUNE-REPORT partition=1x1x1 mc=1 predicted_gb=0.1 budget_gb=1 over_budget=0"; exit 0 ;;
esac
FEOF
chmod +x "${MREPO}/DTFE"
for b in DTFE-double PS-DTFE PS-DTFE-double; do cp "${MREPO}/DTFE" "${MREPO}/${b}"; done
: > "${MDATA}/SIM/snapdir_099/combined_099.hdf5"
: > "${MDATA}/SIM/combined_ics.hdf5"          # run_ps_dtfe.sh: Lagrangian positions present, no h5py probe
mdrive() {   # <driver> <log> [VAR=value ...]: the copied driver on SIM 99; echoes its exit status
    local scr="$1" log="$2"; shift 2
    local rc=0
    env DTFE_DATA_ROOT="${MDATA}" "$@" "${MREPO}/scripts/${scr}" -s SIM -g 16 99 >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
mhang() {    # <driver> <log> <tag>: starts a run whose binary hangs, waits until it is in place; sets mpid
    rm -f "${MPID}"
    env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=hang FAKE_TAG="$3" FAKE_PID="${MPID}" \
        "${MREPO}/scripts/$1" -s SIM -g 16 99 >"$2" 2>&1 &
    mpid=$!
    for _ in $(seq 100); do [ -s "${MPID}" ] && break; sleep 0.1; done
}
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    L="${MDATA}/SIM/snapdir_099/${pfx}.runlog"
    rm -f "${L}" "${L}.failed" "${L}.partial"
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_ok.log" FAKE_MODE=ok FAKE_TAG=first)
    check_rc "$([ "${rc}" = "0" ] && [ -f "${L}" ] && [ ! -e "${L}.partial" ] && [ ! -e "${L}.failed" ] && echo 0 || echo 1)" \
          "${scr}: a good run's log is ${pfx}.runlog, and nothing else is left (exit ${rc})"
    check_rc "$(lacks "${L}" $'\033\\[')" "... de-ANSI'd"
    cp "${L}" "${TMP}/m_${pfx}_first.runlog"
    # a failure (an early error, a bad option, a missing input): the grids on disk keep their log
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_fail.log" FAKE_MODE=fail FAKE_TAG=second)
    check_rc "$([ "${rc}" = "1" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... a failed re-run exits 1 and leaves the .runlog of the grids on disk byte for byte (${rc})"
    check_rc "$([ ! -e "${L}.partial" ] && grep -aq 'fake attempt second' "${L}.failed" && ! grep -aq $'\033\\[' "${L}.failed" && echo 0 || echo 1)" \
          "... its own log is kept, de-ANSI'd, as ${pfx}.runlog.failed (for the failure diagnosis); no .partial left"
    check_rc "$(has01 "${TMP}/m_${pfx}_fail.log" "Error processing snapshot 099 \(its log: .*${pfx}\.runlog\.failed\)")" "... and the driver names it"
    # a stop as the launcher's Stop does it: SIGTERM to the driver (the trap waits for the binary), then the binary
    mhang "${scr}" "${TMP}/m_${pfx}_stop.log" third
    check_rc "$([ -f "${L}.partial" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... while a re-run goes its log is ${pfx}.runlog.partial; the .runlog stays as it was"
    kill -TERM "${mpid}" 2>/dev/null || true
    sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    rc=0; wait "${mpid}" 2>/dev/null || rc=$?
    check_rc "$([ "${rc}" = "143" ] && echo 0 || echo 1)" "... a stopped run ends by its signal, 143, so a calling script stops too (${rc})"
    check_rc "$([ ! -e "${L}.partial" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && grep -aq 'fake attempt third' "${L}.failed" && echo 0 || echo 1)" \
          "... the stopped attempt's log is the .runlog.failed now; the .runlog is untouched, no .partial left"
    check_rc "$(has01 "${TMP}/m_${pfx}_stop.log" 'Stopped: snapshot 099 did not finish')" "... and the driver says so"
    # a run killed outright (SIGKILL, a restart): its .partial is left, the .runlog still untouched
    mhang "${scr}" "${TMP}/m_${pfx}_kill.log" fourth
    kill -KILL "${mpid}" 2>/dev/null || true
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true       # the orphaned binary
    wait "${mpid}" 2>/dev/null || true
    sleep 1                                                             # its orphaned tee writes its last lines
    check_rc "$(grep -aq 'fake attempt fourth' "${L}.partial" && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... a run killed outright leaves its ${pfx}.runlog.partial, and the .runlog still untouched"
    # the next success: its log is the .runlog, the stale .failed and .partial are gone
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_ok2.log" FAKE_MODE=ok FAKE_TAG=fifth)
    check_rc "$([ "${rc}" = "0" ] && grep -aq 'fake attempt fifth' "${L}" && [ ! -e "${L}.failed" ] && [ ! -e "${L}.partial" ] && echo 0 || echo 1)" \
          "... the next good run: its log is the .runlog; the stale .runlog.failed and .runlog.partial are gone (${rc})"
done
# an auto-tune report writes nothing next to the data: not the .runlog, nor a .partial or .failed (run_dtfe.sh has
# the knob too)
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    cp "${MDATA}/SIM/snapdir_099/${pfx}.runlog" "${TMP}/m_before_report.runlog"
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_report.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
    check_rc "$([ "${rc}" = "0" ] && grep -aqE '^  AUTO-TUNE-REPORT snap=099 ' "${TMP}/m_${pfx}_report.log" \
             && [ "$(has01 "${TMP}/m_${pfx}_report.log" '^RUNNING: .*--auto-tune-report')" = 0 ] && echo 0 || echo 1)" \
          "${scr} AUTO_TUNE_REPORT=1 passes --auto-tune-report and reports (${rc})"
    check_rc "$(cmp -s "${MDATA}/SIM/snapdir_099/${pfx}.runlog" "${TMP}/m_before_report.runlog" \
             && [ ! -e "${MDATA}/SIM/snapdir_099/${pfx}.runlog.partial" ] \
             && [ ! -e "${MDATA}/SIM/snapdir_099/${pfx}.runlog.failed" ] && echo 0 || echo 1)" \
          "... and leaves the snapshot's run logs alone"
done

echo ""
echo "(O) the single-instance lock and the preflight of run_dtfe.sh / run_ps_dtfe.sh."
# '<prefix>.runlog.lock' is a directory holding the driver's pid while it runs that snapshot. A
# second driver on the same snapshot is refused (two TNG runs together get both SIGKILLed for memory);
# a lock whose driver is gone is stale and removed with a message; a stop releases it.
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    L="${MDATA}/SIM/snapdir_099/${pfx}.runlog"; K="${L}.lock"
    rm -rf "${K}"
    mhang "${scr}" "${TMP}/o_${pfx}_hold.log" sixth
    check_rc "$([ -d "${K}" ] && [ "$(cat "${K}/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "${scr}: while it runs, ${pfx}.runlog.lock holds the driver's pid (${mpid})"
    cp "${L}" "${TMP}/o_${pfx}_before.runlog"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_second.log" FAKE_MODE=ok FAKE_TAG=intruder)
    check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... a second driver on the same snapshot is refused: exit 1 (${rc})"
    check_rc "$(has01 "${TMP}/o_${pfx}_second.log" "another driver \(pid ${mpid}\) is running this snapshot")" "... naming the pid that holds it"
    check_rc "$(lacks "${TMP}/o_${pfx}_second.log" '^RUNNING:')" "... and its binary never started"
    check_rc "$([ -f "${L}.partial" ] && ! grep -aq 'intruder' "${L}.partial" && cmp -s "${L}" "${TMP}/o_${pfx}_before.runlog" \
             && [ "$(cat "${K}/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "... the running attempt's .partial, the grids' .runlog and the lock are untouched"
    kill -TERM "${mpid}" 2>/dev/null || true
    sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    rc=0; wait "${mpid}" 2>/dev/null || rc=$?
    check_rc "$([ "${rc}" = "143" ] && [ ! -e "${K}" ] && echo 0 || echo 1)" "... a stop still exits 143 and releases the lock (${rc})"
    # the driver alone SIGKILLed: its binary and tee keep running on the snapshot, so its lock is not
    # stale (removing it would start a second binary beside the orphan)
    mhang "${scr}" "${TMP}/o_${pfx}_orphan.log" orphan
    kill -KILL "${mpid}" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_beside.log" FAKE_MODE=ok FAKE_TAG=beside)
    check_rc "$([ "${rc}" = "1" ] && [ -d "${K}" ] && grep -aq 'is gone, but a process still' "${TMP}/o_${pfx}_beside.log" \
             && ! grep -aq '^RUNNING:' "${TMP}/o_${pfx}_beside.log" && echo 0 || echo 1)" \
          "... a driver SIGKILLed while its run goes on: the next one is refused, names the process, keeps the lock (${rc})"
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    unlocked_root_wait() {  # until no process names '<prefix>.runlog.partial' (tee ends once the binary has)
        local a
        for _ in $(seq 50); do a="$(ps -A -ww -o args= 2>/dev/null)"; case "${a}" in *"${L}.partial"*) sleep 0.1 ;; *) return 0 ;; esac; done
    }
    unlocked_root_wait; rm -rf "${K}"
    # killed outright: the lock stays behind with a dead pid; the next run removes it and goes ahead
    mhang "${scr}" "${TMP}/o_${pfx}_kill.log" seventh
    kill -KILL "${mpid}" 2>/dev/null || true
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
    unlocked_root_wait
    check_rc "$([ -d "${K}" ] && echo 0 || echo 1)" "... a SIGKILLed driver leaves its lock"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_after.log" FAKE_MODE=ok FAKE_TAG=eighth)
    check_rc "$([ "${rc}" = "0" ] && [ ! -e "${K}" ] && grep -aq 'fake attempt eighth' "${L}" && echo 0 || echo 1)" \
          "... the next run removes the stale lock, succeeds and leaves no lock (${rc})"
    check_rc "$(has01 "${TMP}/o_${pfx}_after.log" 'removing a stale lock \(pid [0-9]+ is no longer running\)')" "... and says so"
    # a reused pid: alive, but not a driver -- stale as well
    sleep 30 & other=$!
    mkdir -p "${K}"; echo "${other}" > "${K}/pid"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_reused.log" FAKE_MODE=ok)
    kill "${other}" 2>/dev/null || true; wait "${other}" 2>/dev/null || true
    check_rc "$([ "${rc}" = "0" ] && [ ! -e "${K}" ] && echo 0 || echo 1)" "... a lock whose pid now belongs to another program is stale too (${rc})"
    # preflight: the grids must fit on the output volume (a 20000^3 grid needs ~1e14 bytes); SPACE_CHECK=0 skips
    rc=0; env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=ok "${MREPO}/scripts/${scr}" -s SIM -g 20000 99 >"${TMP}/o_${pfx}_space.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... grids larger than the free space: refused, exit 1 (${rc})"
    check_rc "$(has01 "${TMP}/o_${pfx}_space.log" "GB free; this run's grids need about")" "... saying how much is free and needed"
    check_rc "$(lacks "${TMP}/o_${pfx}_space.log" '^RUNNING:')" "... before the binary started"
    rc=0; env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=ok SPACE_CHECK=0 "${MREPO}/scripts/${scr}" -s SIM -g 20000 99 >"${TMP}/o_${pfx}_nospace.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... SPACE_CHECK=0 skips the space check (${rc})"
    # the data disk unplugged: one clear message up front, exit 2 (not a skip of every snapshot under a wrong path)
    rc=0; env DTFE_DATA_ROOT="${TMP}/no such disk/Illustris TNG" "${MREPO}/scripts/${scr}" -s SIM 99 >"${TMP}/o_${pfx}_root.log" 2>&1 || rc=$?
    check_rc "$([ "${rc}" = "2" ] && echo 0 || echo 1)" "... a data root that does not exist: exit 2 (${rc})"
    check_rc "$(has01 "${TMP}/o_${pfx}_root.log" 'does not exist: is the data disk mounted')" "... asking whether the disk is mounted"
    check_rc "$(lacks "${TMP}/o_${pfx}_root.log" 'Processing snapshot')" "... before any snapshot"
    # an output folder that is not writable (root writes anywhere: skipped under root, as in the Docker image)
    if [ "$(id -u)" != "0" ]; then
        chmod a-w "${MDATA}/SIM/snapdir_099"
        rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_ro.log" FAKE_MODE=ok)
        chmod u+w "${MDATA}/SIM/snapdir_099"
        check_rc "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... an output folder that is not writable: refused, exit 1 (${rc})"
        check_rc "$(has01 "${TMP}/o_${pfx}_ro.log" 'is not writable')" "... saying so"
    fi
done
# a report and a dry run take no lock: the launcher's memory check runs a report beside a live run
for pair in "run_ps_dtfe.sh ps_output" "run_dtfe.sh output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    mhang "${scr}" "${TMP}/o_hold_report_${pfx}.log" ninth
    rc=$(mdrive "${scr}" "${TMP}/o_report_${pfx}.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
    rc2=$(mdrive "${scr}" "${TMP}/o_dry_${pfx}.log" DRY_RUN=1)
    check_rc "$([ "${rc}" = "0" ] && [ "${rc2}" = "0" ] && [ "$(cat "${MDATA}/SIM/snapdir_099/${pfx}.runlog.lock/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "${scr}: an auto-tune report and a dry run go ahead beside a live run, and leave its lock alone (${rc}, ${rc2})"
    kill -TERM "${mpid}" 2>/dev/null || true; sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
done

echo ""
echo "(R) run_ps_dtfe.sh: PS_MERGE (index-order by default, passed explicitly), and --resume when a killed run left its checkpoint."
# The binary checkpoints a partitioned run in '<output>.runlog.checkpoint' (under --scratch-dir:
# '<scratch>/<basename>.<hash of the output path>.runlog.checkpoint') and removes it on success, so a folder there
# means a killed run. Against the stand-in binaries of (M), which print their arguments.
RSD="${MDATA}/SIM/snapdir_099"; RCK="${RSD}/ps_output.runlog.checkpoint"
rm -rf "${RCK}" "${RSD}/ps_output.runlog.lock"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_plain.log" FAKE_MODE=ok)
check_rc "$([ "${rc}" = "0" ] && grep -aq '^RUNNING: ' "${TMP}/r_plain.log" && echo 0 || echo 1)" "a plain run goes (${rc})"
# the index-order (staged) merge is the binary's default; the driver passes it explicitly ('--ps-merge index-order':
# given, the memory gate stays on under the auto-tuner's speed split)
merge_of() {   # <log>: the merge option of the RUNNING line with its mode, or 'none'
    local m; m="$(grep -a '^RUNNING: ' "$1" | grep -oE -e '--ps-merge( [a-z][a-z-]*)?' | tr '\n' ' ')"
    echo "${m:-none}"
}
check_rc "$([ "$(merge_of "${TMP}/r_plain.log")" = "--ps-merge index-order " ] \
         && [ "$(lacks "${TMP}/r_plain.log" '^RUNNING: .*--resume')" = 0 ] && echo 0 || echo 1)" \
      "... with '--ps-merge index-order' (the driver's default, explicit) and no --resume"
m=""
for e in PS_MERGE=index-order PS_MERGE=completion-order; do
    rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_merge.log" FAKE_MODE=ok "${e}")
    m="${m}$(merge_of "${TMP}/r_merge.log")| "
done
check_rc "$([ "${m}" = "--ps-merge index-order | --ps-merge completion-order | " ] && echo 0 || echo 1)" \
      "PS_MERGE=MODE passes '--ps-merge MODE' (${m})"
# another value stops the driver before anything runs ('fixed-point' too)
for e in PS_MERGE=ordered PS_MERGE=fixed-point; do
    v="${e#*=}"
    rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_merge_bad.log" FAKE_MODE=ok "${e}")
    check_rc "$([ "${rc}" = "2" ] && [ "$(has01 "${TMP}/r_merge_bad.log" "PS_MERGE must be 'index-order' or 'completion-order' \(got '${v}'\)")" = 0 ] \
             && [ "$(lacks "${TMP}/r_merge_bad.log" '^RUNNING: ')" = 0 ] && echo 0 || echo 1)" "... ${e} is refused before anything runs (exit ${rc})"
done
mkdir -p "${RCK}"; : > "${RCK}/checkpoint"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_resume.log" FAKE_MODE=ok)
check_rc "$([ "${rc}" = "0" ] && grep -aqE -e '^RUNNING: .*--resume' "${TMP}/r_resume.log" && echo 0 || echo 1)" "a checkpoint folder beside the outputs: --resume is passed (${rc})"
check_rc "$(has01 "${TMP}/r_resume.log" "Resuming: a killed run's checkpoint is in .*ps_output\.runlog\.checkpoint")" "... and the driver says so, naming the folder"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_noresume.log" FAKE_MODE=ok DTFE_RESUME=0)
check_rc "$(lacks "${TMP}/r_noresume.log" '^RUNNING: .*--resume')" "DTFE_RESUME=0: no --resume (${rc})"
check_rc "$(has01 "${TMP}/r_noresume.log" 'DTFE_RESUME=0, so this run starts over')" "... and the driver says the run starts over"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_report.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
check_rc "$(lacks "${TMP}/r_report.log" '(--resume|Resuming)')" "an auto-tune report never resumes (${rc})"
# a folder with no 'checkpoint' file (a run killed during its first write, or one that never had room) holds
# nothing to resume: the binary refuses '--resume' there, so passing it would make every rerun fail
rm -f "${RCK}/checkpoint"; : > "${RCK}/checkpoint.tmp.99999"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_emptyck.log" FAKE_MODE=ok)
check_rc "$(lacks "${TMP}/r_emptyck.log" '(^RUNNING: .*--resume|Resuming)')" "a checkpoint folder without a 'checkpoint' file: no --resume (${rc})"
rm -rf "${RCK}"
RSCR="${TMP}/r scratch"; mkdir -p "${RSCR}"
SCK="$(source scripts/config.sh && ps_checkpoint_dir "${RSD}/ps_output" "${RSCR}")"
check_rc "$([[ "$(basename "${SCK}")" =~ ^ps_output\.[0-9a-f]{16}\.runlog\.checkpoint$ ]] && echo 0 || echo 1)" \
      "config.sh ps_checkpoint_dir under a scratch dir: <basename>.<16 hex>.runlog.checkpoint ($(basename "${SCK}"))"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_scr0.log" FAKE_MODE=ok SCRATCH_DIR="${RSCR}")
check_rc "$(lacks "${TMP}/r_scr0.log" '^RUNNING: .*--resume')" "SCRATCH_DIR set, no checkpoint there: no --resume (${rc})"
mkdir -p "${SCK}"; : > "${SCK}/checkpoint"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_scr1.log" FAKE_MODE=ok SCRATCH_DIR="${RSCR}")
check_rc "$(has01 "${TMP}/r_scr1.log" '^RUNNING: .*--resume')" "... the checkpoint in the scratch folder: --resume is passed (${rc})"
rm -rf "${SCK}"
# SAMPLE_POINTS: the points reach the binary as '--sample-points FILE'
: > "${TMP}/r_points.txt"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_pts.log" FAKE_MODE=ok SAMPLE_POINTS="${TMP}/r_points.txt")
check_rc "$([ "${rc}" = 0 ] && [ "$(has01 "${TMP}/r_pts.log" '^RUNNING: .*--sample-points .*r_points\.txt')" = 0 ] && echo 0 || echo 1)" \
      "SAMPLE_POINTS: the points reach the binary as '--sample-points' (${rc})"

echo ""
echo "(P) run_ps_pipeline.sh's 'up to date' compares the run settings, not only the file times."
# File times alone would call a changed grid or field list 'up to date, skipping compute'. The pipeline asks
# run_ps_dtfe.sh (DRY_RUN=1) for the command it would run and compares its options with the RUNNING: line of the
# snapshot's <prefix>.runlog. Against the stand-in binaries of (M), on a real snapshot so the plane can be made.
PDATA="${TMP}/pipe_data"; rm -rf "${PDATA}"; mkdir -p "${PDATA}/SIM/snapdir_099"
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${PDATA}/SIM/snapdir_099/"
: > "${PDATA}/SIM/combined_ics.hdf5"
cp scripts/run_ps_pipeline.sh "${MREPO}/scripts/"
ln -sfn "${ROOT_DIR}/python" "${MREPO}/python"
pdrive() {   # <log> [VAR=value ...]: the copied pipeline on SIM (the deposit's knobs unset first); echoes its exit status
    local log="$1"; shift
    local rc=0
    env -u DTFE_EXACT_AVERAGE -u SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID \
        DTFE_DATA_ROOT="${PDATA}" SIMS=SIM NU=8 GRID_SIZE=16 FIELDS="density_a velocity_a" DTFE_GPU=0 FAKE_MODE=ok \
        "$@" "${MREPO}/scripts/run_ps_pipeline.sh" >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
SD="${PDATA}/SIM/snapdir_099"
rc=$(pdrive "${TMP}/p_first.log")
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"           # a slice of the 8^2 plane: 64 float64, as a real run leaves
touch -t 203001010000 "${SD}/ps_output.pts_den" "${SD}/ps_output.pts_velGrad"   # what a real run leaves (newer than its inputs)
check_rc "$([ "${rc}" = "0" ] && grep -aq '^RUNNING: ./PS-DTFE[^ ]* .*--grid 16 .*--exact ' "${SD}/ps_output.runlog" && echo 0 || echo 1)" \
      "a first pipeline run computes the snapshot (the exact deposit, the default), its run log carries the command (${rc})"
rc=$(pdrive "${TMP}/p_same.log")
check_rc "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... a rerun with the same settings exits 0 (${rc})"
check_rc "$(has01 "${TMP}/p_same.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" "... and skips the snapshot: same settings"
check_rc "$(lacks "${TMP}/p_same.log" 'run_ps_dtfe.sh -s SIM')" "... running nothing"
# a slice newer than the plane but of ANOTHER plane size (a 16^2 one against the 8^2 plane: TNG100's
# 4096^2 slices beside an older hires_plane_z8192) is not 'up to date': skipping the compute would fail every render
head -c 2048 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den"
rc=$(pdrive "${TMP}/p_size.log" DRY_RUN=1)
check_rc "$(has01 "${TMP}/p_size.log" 'snap 99: ps_output.pts_den holds 256 points, the plane 64: a slice of another plane, recomputing')" \
      "a slice of another plane size: recomputed, though newer than the plane and with the same settings (${rc})"
check_rc "$(has01 "${TMP}/p_size.log" 'run_ps_dtfe.sh -s SIM -g 16 99')" "... run_ps_dtfe.sh is called for it"
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den"
rc=$(pdrive "${TMP}/p_grid.log" GRID_SIZE=24 DRY_RUN=1)
check_rc "$(has01 "${TMP}/p_grid.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" "a different grid: the snapshot is recomputed (${rc})"
check_rc "$(grep -aq 'was: --grid 16$' "${TMP}/p_grid.log" && grep -aq 'now: --grid 24$' "${TMP}/p_grid.log" && echo 0 || echo 1)" "... the difference is shown"
check_rc "$(has01 "${TMP}/p_grid.log" 'run_ps_dtfe.sh -s SIM -g 24 99')" "... and run_ps_dtfe.sh is called for it"
rc=$(pdrive "${TMP}/p_cau.log" PS_CAUSTICS=0 DRY_RUN=1)
check_rc "$([ "$(has01 "${TMP}/p_cau.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'was: --ps-caustics$' "${TMP}/p_cau.log" && echo 0 || echo 1)" \
      "a deposit option switched off (PS_CAUSTICS=0) counts too (${rc})"
rc=$(pdrive "${TMP}/p_sub.log" FIELDS=density_a DRY_RUN=1)
check_rc "$(has01 "${TMP}/p_sub.log" 'up to date \(same settings\)')" "fewer fields than the grids on disk carry: still up to date"
rc=$(pdrive "${TMP}/p_more.log" FIELDS="density_a velocity_a tweb_a" DRY_RUN=1)
check_rc "$(has01 "${TMP}/p_more.log" 'settings differ')" "... a field the grids lack: recomputed"
rc=$(pdrive "${TMP}/p_gpu.log" DTFE_GPU=1 DRY_RUN=1)
check_rc "$(has01 "${TMP}/p_gpu.log" 'up to date \(same settings\)')" "the GPU/CPU choice alone does not make grids stale"
# a run log whose command carries '--ps-volume-weighted' after the deposit flag, as the grid sets on disk have it
# (PS-DTFE volume-weights its velocity moments in every run; run_settings does not read the token): up to date for
# the same settings
cp "${SD}/ps_output.runlog" "${SD}/ps_output.runlog.keep"
sed -e '/^RUNNING:/s/ --exact / --exact --ps-volume-weighted /' "${SD}/ps_output.runlog.keep" > "${SD}/ps_output.runlog"
rc=$(pdrive "${TMP}/p_vw.log" DRY_RUN=1)
check_rc "$(grep -aq '^RUNNING: .* --exact --ps-volume-weighted ' "${SD}/ps_output.runlog" \
         && [ "$(has01 "${TMP}/p_vw.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" = 0 ] \
         && echo 0 || echo 1)" \
      "a run log whose command carries '--ps-volume-weighted' (after '--exact') is up to date for the same settings (${rc})"
# a run log as the drivers wrote it when they resolved the device, the run mode and the share themselves ('--gpu',
# '--run-mode balanced', '--gpu-share auto' on its RUNNING line): up to date for the command they plan now, which
# names none of the three
sed -e '/^RUNNING:/s/ --ps-merge index-order / --ps-merge index-order --run-mode balanced --gpu-share auto /' \
    -e '/^RUNNING:/s/ --exact / --gpu --exact /' "${SD}/ps_output.runlog.keep" > "${SD}/ps_output.runlog"
rc=$(pdrive "${TMP}/p_olddrv.log" DRY_RUN=1)
check_rc "$(grep -aq '^RUNNING: .* --ps-merge index-order --run-mode balanced --gpu-share auto .* --gpu --exact ' "${SD}/ps_output.runlog" \
         && [ "$(has01 "${TMP}/p_olddrv.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" = 0 ] \
         && echo 0 || echo 1)" \
      "a run log with the '--gpu --run-mode balanced --gpu-share auto' the drivers used to pass is up to date for their command now (${rc})"
mv "${SD}/ps_output.runlog.keep" "${SD}/ps_output.runlog"
# run_settings itself (the awk of the copied run_ps_pipeline.sh) on command lines: an exact log with a count and
# '--ps-volume-weighted' and the '--exact' command the driver plans are the same settings (the count is no setting of
# the exact deposit, '--ps-volume-weighted' and '--no-overlap' none at all); another threshold is not
eval "$(awk '$0 == "run_settings() {" { on = 1 } on { print } on && $0 == "}" { exit }' "${MREPO}/scripts/run_ps_pipeline.sh")"
rs_xlog="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --grid 64 --subsamples 3 --field density_a --lambda-th 0.3 --exact --ps-volume-weighted --ps-caustics')"
rs_new="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --exact --no-overlap --ps-caustics')"
rs_other="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.2 --exact --ps-caustics')"
check_rc "$([ -n "${rs_new}" ] && [ "${rs_xlog}" = "${rs_new}" ] && printf '%s\n' "${rs_new}" | grep -qx -e '--exact' \
         && ! printf '%s\n' "${rs_new}" | grep -q -e '^--subsamples' \
         && printf '%s\n' "${rs_new}" | grep -qx -e '--lambda-th 0.3' && [ "${rs_other}" != "${rs_new}" ] && echo 0 || echo 1)" \
      "run_settings: an exact log with a count and '--ps-volume-weighted' and the '--exact' command are the same settings (neither is a setting of the exact deposit's grids), another threshold is not"
# a sampled run's log names no deposit flag ('--sampled' is not compared) and may name a count: it is the command a
# sampled pipeline run emits with that count ('--sampled --subsamples 3' for a log naming 3), not '--sampled' alone
# (no count: the binary's own 1, '--sampled --subsamples 1'), and not the exact default's
rs_s_old="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --grid 64 --subsamples 3 --field density_a --lambda-th 0.3 --ps-caustics')"
rs_s_new="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --sampled --ps-caustics')"
rs_s_n1="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --sampled --subsamples 1 --ps-caustics')"
rs_s_n3="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --sampled --subsamples 3 --ps-caustics')"
rs_s_n2="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --sampled --subsamples 2 --ps-caustics')"
rs_x="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --exact --ps-caustics')"
check_rc "$([ -n "${rs_s_new}" ] && [ "${rs_s_old}" = "${rs_s_n3}" ] && [ "${rs_s_n1}" = "${rs_s_new}" ] && [ "${rs_s_old}" != "${rs_s_new}" ] \
         && [ "${rs_s_n2}" != "${rs_s_new}" ] && [ "${rs_s_n2}" != "${rs_s_old}" ] \
         && [ "${rs_x}" != "${rs_s_old}" ] && printf '%s\n' "${rs_s_new}" | grep -qx -e '--subsamples 1' \
         && ! printf '%s\n' "${rs_s_new}" | grep -qE -e '^--(exact|sampled)$' && echo 0 || echo 1)" \
      "run_settings: a sampled log (a count, no flag) is '--sampled' with that count (3), not '--sampled' alone (the binary's 1), not '--sampled --subsamples 2', not '--exact'"
# the options run_settings compares beside the deposit: the count, the Lagrangian padding, the face gradient, the
# alpha shape and the halo release, each read as written; each one alone makes a command differ
rs7="$(run_settings './PS-DTFE in.hdf5 out --input-format 105 --mpc-unit 1000 --grid 64 --subsamples 3 --field density_a --lambda-th 0.3 --ps-lagrangian-padding 8 --ps-lagrangian-input ics.hdf5 --sample-points p.txt --ps-merge index-order --face-gradient --ps-alpha-shape 2.5 --ps-halo-release 500 --grid-window 0 1 0 1 0 1')"
rs7_ok=0
for o in '--subsamples 3' '--ps-lagrangian-padding 8' '--face-gradient' '--ps-alpha-shape 2.5' '--ps-halo-release 500'; do
    printf '%s\n' "${rs7}" | grep -qx -e "${o}" || rs7_ok=1
done
for o in 'face-gradient' 'per-tet-gradient' 'ps-lagrangian-padding 9' 'ps-alpha-shape 3' 'ps-halo-release 400' \
         'subsamples 2'; do   # each one differs alone
    [ "$(run_settings "./PS-DTFE in out --grid 64 --${o}")" != "$(run_settings './PS-DTFE in out --grid 64')" ] || rs7_ok=1
done
check_rc "$([ -n "${rs7}" ] && [ "${rs7_ok}" = 0 ] && echo 0 || echo 1)" \
      "run_settings compares the count, the Lagrangian padding, the face gradient, the alpha shape and the halo release, each read as written; each one alone differs"
# the run log as a sampled run writes it (a count, no deposit flag), made from the exact run's log on disk: current for
# a sampled pipeline run at that count (SUBSAMPLES=3), stale for the exact default, for another count and for
# DTFE_EXACT_AVERAGE=0 without one (the binary's own count, 1)
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den" "${SD}/ps_output.pts_velGrad"
sed -e '/^RUNNING:/s/ --exact / --subsamples 3 /' "${SD}/ps_output.runlog" > "${SD}/ps_output.runlog.old" \
    && mv "${SD}/ps_output.runlog.old" "${SD}/ps_output.runlog"
rc=$(pdrive "${TMP}/p_os0.log" DTFE_EXACT_AVERAGE=0 DRY_RUN=1); rc2=$(pdrive "${TMP}/p_os3.log" SUBSAMPLES=3 DRY_RUN=1)
rc3=$(pdrive "${TMP}/p_osd.log" DRY_RUN=1); rc4=$(pdrive "${TMP}/p_os2.log" SUBSAMPLES=2 DRY_RUN=1)
check_rc "$([ "$(lacks "${SD}/ps_output.runlog" '^RUNNING: .*--(exact|sampled) ')" = 0 ] && grep -aq '^RUNNING: .* --subsamples 3 ' "${SD}/ps_output.runlog" \
         && [ "$(has01 "${TMP}/p_os0.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'was: --subsamples 3$' "${TMP}/p_os0.log" && grep -aq 'now: --subsamples 1$' "${TMP}/p_os0.log" \
         && [ "$(has01 "${TMP}/p_os3.log" 'snap 99: ps_output.pts_den up to date \(same settings\)')" = 0 ] \
         && [ "$(has01 "${TMP}/p_osd.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'was: --subsamples 3$' "${TMP}/p_osd.log" && grep -aq 'now: --exact$' "${TMP}/p_osd.log" \
         && [ "$(has01 "${TMP}/p_os2.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'now: --subsamples 2$' "${TMP}/p_os2.log" && echo 0 || echo 1)" \
      "a sampled run's log (the count 3, no flag): up to date for SUBSAMPLES=3, recomputed for DTFE_EXACT_AVERAGE=0 without a count (now: --subsamples 1), the exact default and SUBSAMPLES=2 (${rc} ${rc2} ${rc3} ${rc4})"
# a sampled run's log that names NO count (the drivers pass one only when SUBSAMPLES is set): its count is its
# header's 'avg sub-samples (nSub)' line, so a set made at 3 is not kept as current for a sampled run without a count
sed -e '/^RUNNING:/s/ --subsamples 3 / --sampled /' "${SD}/ps_output.runlog" > "${SD}/ps_output.runlog.old" \
    && mv "${SD}/ps_output.runlog.old" "${SD}/ps_output.runlog"
printf '\t avg sub-samples (nSub) : 3   (each _a cell volume-averaged over nSub^3 = 27 points)\n' >> "${SD}/ps_output.runlog"
rc=$(pdrive "${TMP}/p_oh0.log" DTFE_EXACT_AVERAGE=0 DRY_RUN=1); rc2=$(pdrive "${TMP}/p_oh3.log" SUBSAMPLES=3 DRY_RUN=1)
sed -e 's/(nSub) : 3 /(nSub) : 1 /' "${SD}/ps_output.runlog" > "${SD}/ps_output.runlog.old" \
    && mv "${SD}/ps_output.runlog.old" "${SD}/ps_output.runlog"
rc3=$(pdrive "${TMP}/p_oh1.log" DTFE_EXACT_AVERAGE=0 DRY_RUN=1)
check_rc "$(grep -aq '^RUNNING: .* --sampled ' "${SD}/ps_output.runlog" \
         && [ "$(lacks "${SD}/ps_output.runlog" '^RUNNING: .*--subsamples ')" = 0 ] \
         && [ "$(has01 "${TMP}/p_oh0.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'was: --subsamples 3$' "${TMP}/p_oh0.log" && grep -aq 'now: --subsamples 1$' "${TMP}/p_oh0.log" \
         && [ "$(has01 "${TMP}/p_oh3.log" 'snap 99: ps_output.pts_den up to date \(same settings\)')" = 0 ] \
         && [ "$(has01 "${TMP}/p_oh1.log" 'snap 99: ps_output.pts_den up to date \(same settings\)')" = 0 ] \
         && echo 0 || echo 1)" \
      "a sampled run's log that names no count is read by its header's count line: one made at 3 is recomputed for DTFE_EXACT_AVERAGE=0 without a count and up to date for SUBSAMPLES=3; one made at 1 is up to date (${rc} ${rc2} ${rc3})"
rc=$(pdrive "${TMP}/p_back.log")                 # a real run again (the exact default) for the checks below
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den" "${SD}/ps_output.pts_velGrad"
mv "${SD}/ps_output.runlog" "${SD}/ps_output.runlog.keep"
rc=$(pdrive "${TMP}/p_nolog.log" GRID_SIZE=24 DRY_RUN=1)
check_rc "$(has01 "${TMP}/p_nolog.log" 'up to date by file times \(no ps_output.runlog to compare the settings with\)')" \
      "no run log: judged by the file times, and said so"
mv "${SD}/ps_output.runlog.keep" "${SD}/ps_output.runlog"
# a run log without the header's 'run mode' line, which the programs print on every run: an older version of the
# programs wrote it, so its settings are not read and the snapshot is recomputed, though they are this run's (the
# same log with the line: up to date)
rc=$(pdrive "${TMP}/p_cur.log" DRY_RUN=1)
cp "${SD}/ps_output.runlog" "${SD}/ps_output.runlog.keep"
sed -e '/^[[:blank:]]*run mode[[:blank:]]*:/d' "${SD}/ps_output.runlog.keep" > "${SD}/ps_output.runlog"
rc2=$(pdrive "${TMP}/p_older.log" DRY_RUN=1)
check_rc "$(grep -aqE -e '^[[:blank:]]*run mode[[:blank:]]+: ' "${SD}/ps_output.runlog.keep" \
         && grep -aq '^RUNNING: ' "${SD}/ps_output.runlog" \
         && ! grep -aqE -e '^[[:blank:]]*run mode[[:blank:]]*:' "${SD}/ps_output.runlog" \
         && [ "$(has01 "${TMP}/p_cur.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" = 0 ] \
         && grep -aqxF -e '-- snap 99: ps_output.runlog was written by an older version of the programs, recomputing' \
                "${TMP}/p_older.log" \
         && [ "$(has01 "${TMP}/p_older.log" 'run_ps_dtfe\.sh -s SIM -g 16 99')" = 0 ] \
         && [ "$(lacks "${TMP}/p_older.log" '(snap 99: .*(up to date|settings differ)|^ +(was|now): )')" = 0 ] \
         && echo 0 || echo 1)" \
      "a run log without the header's 'run mode' line (an older version of the programs) is recomputed though its settings are this run's, with the pipeline's message; the same log with the line is up to date (${rc} ${rc2})"
mv "${SD}/ps_output.runlog.keep" "${SD}/ps_output.runlog"

echo ""
echo "(S) production defaults, OUTPUT_PREFIX and a rerun's left-over files."
# Against the stand-in binaries of (M) in a data folder of its own.
SDATA="${TMP}/sup_data"; SSD="${SDATA}/SIM/snapdir_099"; rm -rf "${SDATA}"; mkdir -p "${SSD}"
: > "${SSD}/combined_099.hdf5"; : > "${SDATA}/SIM/combined_ics.hdf5"
sdrive() {   # <driver> <log> [VAR=value ...] [-- driver options]: the copied driver on SDATA's SIM 99; echoes its exit status
    local scr="$1" log="$2" e=() rc=0
    shift 2
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do e+=("$1"); shift; done
    [ "$#" -gt 0 ] && shift
    env -u DTFE_GPU -u DTFE_EXACT_AVERAGE -u PS_CAUSTICS -u FIELDS -u OUTPUT_PREFIX \
        -u SUBSAMPLES -u PS_MERGE -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID \
        -u FACE_GRADIENT -u DTFE_PRECISION -u OMP_NUM_THREADS -u THREADS DTFE_DATA_ROOT="${SDATA}" ${e[@]+"${e[@]}"} \
        "${MREPO}/scripts/${scr}" -s SIM -g 16 "$@" 99 >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
sline() { grep -aE -e '^ +\./(PS-)?DTFE(-double)? ' "$1" | head -1; }      # the DRY RUN's planned command
# dev_line <log>: the device word of a driver's banner (run_dtfe.sh's "'_a' interpolation: ", run_ps_dtfe.sh's
# 'Deposit: ', the 'GPU: ' of run_ps_pipeline.sh's Slice line)
dev_line() {
    sed -n -e "s/^'_a' interpolation: //p" -e 's/^Deposit: \(.*\)   caustics: .*/\1/p' \
        -e 's/^Slice: .*   GPU: \(.*\)   deposit: .*/\1/p' "$1" | head -1
}
# pass_fields <log>: what a run beside the stand-ins handed the binary, and what the banner named, as
# '<DTFE_GPU>/<DTFE_RUN_MODE>/<options>|<device word>|<run-mode word>': the environment the stand-in printed
# (FAKE_ENV=1; 'unset' for a variable it did not get), the --gpu, --run-mode and --gpu-share options of its command,
# the banner's device word and its 'Run mode: ' line
pass_fields() {
    printf '%s/%s/%s|%s|%s' "$(sed -n 's/^fake environment: DTFE_GPU=//p' "$1" | head -1)" \
        "$(sed -n 's/^fake environment: DTFE_RUN_MODE=//p' "$1" | head -1)" \
        "$(awk '/^RUNNING: / { for (i = 3; i <= NF; i++)
                                   if ($i == "--gpu") o = o " " $i; else if ($i ~ /^--(run-mode|gpu-share)$/) o = o " " $i " " $(i + 1)
                               sub(/^ /, "", o); print o; exit }' "$1")" \
        "$(dev_line "$1")" "$(sed -n 's/^Run mode: //p' "$1" | head -1)"
}
pass_of() {   # <driver> [VAR=value ...] [-- driver options]: pass_fields of a run of the copied driver (sdrive)
    local scr="$1"
    shift
    sdrive "${scr}" "${TMP}/s_pass.log" FAKE_MODE=ok FAKE_ENV=1 "$@" >/dev/null
    pass_fields "${TMP}/s_pass.log"
}
# the device: no driver resolves it or passes '--gpu'. DTFE_GPU reaches the binary as the caller set it (unset stays
# unset, a value is not rewritten), -m exports DTFE_GPU=1 and wins over the caller's value, on a run and on an
# auto-tune report alike; the banner names what the driver was given. Real runs of the stand-ins, their run logs
# removed again afterwards
s_bad=""
for scr in run_dtfe.sh run_ps_dtfe.sh; do
    # '<what pass_fields gives>|<the knobs and options>'
    for c in "unset/unset/|${PICK}|${MODE_DEFAULT}|" "0/unset/|DTFE_GPU=0|${MODE_DEFAULT}|DTFE_GPU=0" \
             "1/unset/|DTFE_GPU=1|${MODE_DEFAULT}|DTFE_GPU=1" "off/unset/|DTFE_GPU=off|${MODE_DEFAULT}|DTFE_GPU=off" \
             "1/unset/|the GPU (-m)|${MODE_DEFAULT}|-- -m" "1/unset/|the GPU (-m)|${MODE_DEFAULT}|DTFE_GPU=0 -- -m" \
             "1/unset/|DTFE_GPU=1|${MODE_DEFAULT}|DTFE_GPU=1 FAKE_MODE=report AUTO_TUNE_REPORT=1"; do
        # shellcheck disable=SC2086
        got="$(pass_of "${scr}" ${c##*|})"
        [ "${got}" = "${c%|*}" ] || s_bad="${s_bad} [${scr} ${c##*|}: ${got}]"
    done
done
rm -rf "${SSD}/output."* "${SSD}/ps_output."*
check_rc "$([ -z "${s_bad}" ] && echo 0 || echo 1)" \
      "both drivers pass no '--gpu': DTFE_GPU unset, 0, 1 and off, -m (beside DTFE_GPU=0 too) and an auto-tune report reach the binary as given, the banner naming '${PICK}', 'DTFE_GPU=<value>' or 'the GPU (-m)'${s_bad:+ -- differ:${s_bad}}"
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_psdef.log" DRY_RUN=1)
check_rc "$([ "${rc}" = "0" ] && sline "${TMP}/s_psdef.log" | grep -qE -e '--field +density_a +velocity_a +divergence_a +shear_a +--lambda-th +0\.3 +--exact +.*--ps-caustics' \
         && ! sline "${TMP}/s_psdef.log" | grep -q -e '--subsamples' && grep -aqxF 'Cell averages: exact deposit (--exact; the default)' "${TMP}/s_psdef.log" \
         && echo 0 || echo 1)" "run_ps_dtfe.sh's defaults are production's: the four analysis fields, the exact deposit (--exact, no count), the caustic classes (${rc})"
# the sampled deposit: DTFE_EXACT_AVERAGE=0 names no count (the binary's 1), a SUBSAMPLES count alone selects it and is
# passed; -e (or DTFE_EXACT_AVERAGE=1) beside a count keeps the exact deposit, with one note
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_pssam.log" DRY_RUN=1 DTFE_EXACT_AVERAGE=0)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_pssub.log" DRY_RUN=1 SUBSAMPLES=2)
rc3=$(sdrive run_ps_dtfe.sh "${TMP}/s_psksub.log" DRY_RUN=1 SUBSAMPLES=2 DTFE_EXACT_AVERAGE=1)
check_rc "$([ "${rc}${rc2}${rc3}" = "000" ] && sline "${TMP}/s_pssam.log" | grep -qE -e '--lambda-th +0\.3 +--sampled +--ps-vertex-mass ' \
         && grep -aqxF 'Cell averages: sampled deposit, nSub^3 sub-points per cell (--sampled; DTFE_EXACT_AVERAGE=0)' "${TMP}/s_pssam.log" \
         && sline "${TMP}/s_pssub.log" | grep -qE -e '--lambda-th +0\.3 +--sampled +--subsamples +2 +--ps-vertex-mass ' \
         && grep -aqxF 'Cell averages: sampled deposit, nSub^3 sub-points per cell (--sampled --subsamples 2; SUBSAMPLES=2)' "${TMP}/s_pssub.log" \
         && sline "${TMP}/s_psksub.log" | grep -qE -e '--lambda-th +0\.3 +--exact +--ps-vertex-mass ' \
         && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: DTFE_EXACT_AVERAGE=1 asks for the exact deposit' "${TMP}/s_psksub.log")" = 1 ] \
         && echo 0 || echo 1)" \
      "... DTFE_EXACT_AVERAGE=0: --sampled (no count); SUBSAMPLES=2 alone: --sampled --subsamples 2; DTFE_EXACT_AVERAGE=1 beside it: --exact, the count said ignored (${rc} ${rc2} ${rc3})"
# FACE_GRADIENT=1: --face-gradient in both drivers; run_ps_dtfe.sh refuses it, exit 2 before any snapshot, with the
# sampled deposit (DTFE_EXACT_AVERAGE=0, or a count alone; the exact one is the default); under DTFE_GPU=1 it plans
# the same command (the device is the binary's)
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface.log" DRY_RUN=1 FACE_GRADIENT=1 DTFE_EXACT_AVERAGE=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface2.log" DRY_RUN=1 FACE_GRADIENT=1)
rc4=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface4.log" DRY_RUN=1 FACE_GRADIENT=1 DTFE_EXACT_AVERAGE=1 DTFE_GPU=1)
rc5=$(sdrive run_dtfe.sh "${TMP}/s_dtface.log" DRY_RUN=1 FACE_GRADIENT=1)
rc6=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface6.log" DRY_RUN=1 FACE_GRADIENT=1 DTFE_EXACT_AVERAGE=0)
rc7=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface7.log" DRY_RUN=1 FACE_GRADIENT=1 SUBSAMPLES=2)
check_rc "$([ "${rc}" = "0" ] && sline "${TMP}/s_psface.log" | grep -qE -e '--exact .*--face-gradient' \
         && [ "${rc2}" = "0" ] && sline "${TMP}/s_psface2.log" | grep -qE -e '--exact .*--face-gradient' \
         && [ "${rc4}" = "0" ] && [ "$(sline "${TMP}/s_psface4.log")" = "$(sline "${TMP}/s_psface.log")" ] \
         && [ "${rc5}" = "0" ] && sline "${TMP}/s_dtface.log" | grep -qE -e '--exact +--face-gradient$' \
         && [ "${rc6}" = "2" ] && grep -aqxF 'Error: FACE_GRADIENT=1 needs the exact deposit (the default): DTFE_EXACT_AVERAGE=0 asks for the sampled one.' "${TMP}/s_psface6.log" \
         && [ "${rc7}" = "2" ] && grep -aqxF 'Error: FACE_GRADIENT=1 needs the exact deposit (the default): SUBSAMPLES=2 asks for the sampled one.' "${TMP}/s_psface7.log" \
         && echo 0 || echo 1)" \
      "FACE_GRADIENT=1 passes --face-gradient in both drivers, alone (the exact deposit, the default) and under DTFE_GPU=1 (the same command); PS with the sampled deposit (DTFE_EXACT_AVERAGE=0, SUBSAMPLES=2): exit 2 (${rc} ${rc2} ${rc4} ${rc5} ${rc6} ${rc7})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_dtpertet.log" DRY_RUN=1 FACE_GRADIENT=0)
check_rc "$([ "${rc}" = "0" ] && sline "${TMP}/s_dtpertet.log" | grep -qE -e '--exact +--per-tet-gradient$' && echo 0 || echo 1)" \
      "FACE_GRADIENT=0 passes --per-tet-gradient to run_dtfe.sh (the face gradient is the standard binary's default) (${rc})"
# run_ps_dtfe.sh's FACE_GRADIENT as run_dtfe.sh's: unset passes neither route flag (the binary's default, the
# face gradient where the run qualifies), 0 passes --per-tet-gradient
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_psfgdef.log" DRY_RUN=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_pspertet.log" DRY_RUN=1 FACE_GRADIENT=0)
check_rc "$([ "${rc}" = "0" ] && [ -n "$(sline "${TMP}/s_psfgdef.log")" ] \
         && ! sline "${TMP}/s_psfgdef.log" | grep -qE -e '--(face|per-tet)-gradient' \
         && [ "${rc2}" = "0" ] \
         && sline "${TMP}/s_pspertet.log" | grep -qE -e '--exact .*--per-tet-gradient' \
         && echo 0 || echo 1)" \
      "run_ps_dtfe.sh FACE_GRADIENT unset: no route flag (the binary's default); 0: --per-tet-gradient (${rc} ${rc2})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_dtdef.log" DRY_RUN=1)
check_rc "$([ "${rc}" = "0" ] && sline "${TMP}/s_dtdef.log" | grep -qE -e 'snapdir_099/output +--grid +16 +--padding +15 +--periodic +--field +density_a +velocity_a +divergence_a +shear_a +--lambda-th +0\.3 +--exact$' \
         && echo 0 || echo 1)" "run_dtfe.sh's defaults: padding 15, no run mode (the binary's default), the same four fields as run_ps_dtfe.sh, the exact cell average (${rc})"
# OUTPUT_PREFIX for run_dtfe.sh: the files and the run log under it; a PS-DTFE-style prefix refused before anything runs
rc=$(sdrive run_dtfe.sh "${TMP}/s_pfx.log" OUTPUT_PREFIX=output_mc DTFE_EXACT_AVERAGE=0 FAKE_MODE=ok)
check_rc "$([ "${rc}" = "0" ] && [ -f "${SSD}/output_mc.runlog" ] && [ ! -e "${SSD}/output.runlog" ] \
         && grep -aq '^RUNNING: .*snapdir_099/output_mc --grid' "${SSD}/output_mc.runlog" && echo 0 || echo 1)" \
      "run_dtfe.sh OUTPUT_PREFIX=output_mc: the binary's output and the run log are output_mc.* (${rc})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_pfxps.log" OUTPUT_PREFIX=ps_output FAKE_MODE=ok)
check_rc "$([ "${rc}" = "2" ] && [ "$(has01 "${TMP}/s_pfxps.log" "is PS-DTFE's naming")" = 0 ] && [ "$(lacks "${TMP}/s_pfxps.log" '^RUNNING:')" = 0 ] \
         && echo 0 || echo 1)" "... a ps_* prefix (an export meant for run_ps_dtfe.sh) is refused, exit 2, nothing run (${rc})"
rm -f "${SSD}/output_mc."*
# a rerun's left-over files: seeded as an earlier, fuller run left them
s_seed() {
    rm -rf "${SSD}/output."* "${SSD}/output.superseded"
    for f in a_den a_vel pts_den a_tweb badGridPoints0; do echo old > "${SSD}/output.${f}"; done
    echo '{}' > "${SSD}/output.gui.json"; mkdir -p "${SSD}/output.runlog.checkpoint"
    printf 'RUNNING: ./DTFE old run\nold log\n' > "${SSD}/output.runlog"; echo failed > "${SSD}/output.runlog.failed"
    touch -t 202601010000 "${SSD}/output."*
}
s_seed
rc=$(sdrive run_dtfe.sh "${TMP}/s_sup.log" FAKE_MODE=write)
SUP="$(ls -d "${SSD}/output.superseded/"*/ 2>/dev/null | head -1)"; SUP="${SUP%/}"
check_rc "$([ "${rc}" = "0" ] && [ -n "${SUP}" ] && [ "$(ls "${SUP}" | tr '\n' ' ')" = "output.a_tweb output.pts_den output.runlog " ] \
         && [ "$(cat "${SUP}/output.pts_den")" = old ] && grep -aq '^old log$' "${SUP}/output.runlog" \
         && [[ "$(basename "${SUP}")" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{6}$ ]] && echo 0 || echo 1)" \
      "a rerun that exits 0: the files it did not write (.pts_den, .a_tweb) MOVED to output.superseded/<date_time>/ with the old run log (${rc}; $(ls "${SUP}" 2>/dev/null | tr '\n' ' '))"
check_rc "$([ "$(cat "${SSD}/output.a_den")" = new ] && [ "$(cat "${SSD}/output.a_vel")" = new ] && [ "$(cat "${SSD}/output.badGridPoints0")" = new ] \
         && grep -aq 'fake attempt' "${SSD}/output.runlog" && [ -f "${SSD}/output.gui.json" ] && [ -d "${SSD}/output.runlog.checkpoint" ] \
         && [ ! -e "${SSD}/output.pts_den" ] && [ ! -e "${SSD}/output.a_tweb" ] && [ ! -e "${SSD}/output.runlog.failed" ] && echo 0 || echo 1)" \
      "... the files it wrote stay (renamed into place, or rewritten in place like .badGridPoints0), the run log is the new one; the settings sidecar and a checkpoint are not touched"
check_rc "$(has01 "${TMP}/s_sup.log" '2 file\(s\) of an earlier run with this prefix moved to .*output\.superseded/.* \(nothing deleted')" "... and the driver says so"
s_seed
rc=$(sdrive run_dtfe.sh "${TMP}/s_supfail.log" FAKE_MODE=fail)
check_rc "$([ "${rc}" = "1" ] && [ ! -e "${SSD}/output.superseded" ] && [ "$(cat "${SSD}/output.pts_den")" = old ] && grep -aq '^old log$' "${SSD}/output.runlog" \
         && echo 0 || echo 1)" "a FAILED rerun moves nothing: the earlier run's set stays whole with its run log (${rc})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_supdry.log" DRY_RUN=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_suprep.log" FAKE_MODE=report AUTO_TUNE_REPORT=1 OUTPUT_PREFIX=output)
rc3=$(sdrive run_dtfe.sh "${TMP}/s_suprep_std.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
check_rc "$([ ! -e "${SSD}/output.superseded" ] && [ -f "${SSD}/output.pts_den" ] && [ -f "${SSD}/output.a_tweb" ] && echo 0 || echo 1)" \
      "... nor does a dry run or an auto-tune report (${rc}, ${rc2}, run_dtfe.sh ${rc3})"
rm -rf "${SSD}/output."* "${SSD}/output.superseded"
rc=$(sdrive run_dtfe.sh "${TMP}/s_supnone.log" FAKE_MODE=write)
check_rc "$([ "${rc}" = "0" ] && [ ! -e "${SSD}/output.superseded" ] && [ "$(lacks "${TMP}/s_supnone.log" 'Superseded')" = 0 ] && echo 0 || echo 1)" \
      "... and a first run into an empty prefix makes no superseded folder (${rc})"

echo ""
echo "(T) a driver switch reads 0/false/no/off (any case) as off and any other value as on, as the binaries do."
# a switch reads like the binaries' (src/env_knobs.h): 0/false/no/off (any case) off, any other value on
sdrive run_ps_dtfe.sh "${TMP}/t_on.log" DRY_RUN=1 PS_CAUSTICS=1 PS_VERTEX_MASS=1 PS_CAUSTIC_CUSPS=1 >/dev/null
sdrive run_ps_dtfe.sh "${TMP}/t_true.log" DRY_RUN=1 PS_CAUSTICS=true PS_VERTEX_MASS=YES PS_CAUSTIC_CUSPS=on >/dev/null
sdrive run_ps_dtfe.sh "${TMP}/t_off.log" DRY_RUN=1 PS_CAUSTICS=0 PS_VERTEX_MASS=0 >/dev/null
sdrive run_ps_dtfe.sh "${TMP}/t_false.log" DRY_RUN=1 PS_CAUSTICS=Off PS_VERTEX_MASS=false >/dev/null
t_on="$(sline "${TMP}/t_on.log")"; t_true="$(sline "${TMP}/t_true.log")"
t_off="$(sline "${TMP}/t_off.log")"; t_false="$(sline "${TMP}/t_false.log")"
check_rc "$([ -n "${t_on}" ] && [ "${t_on}" = "${t_true}" ] && [ -n "${t_off}" ] && [ "${t_off}" = "${t_false}" ] \
         && printf '%s\n' "${t_true}" | grep -q -e '--ps-caustic-cusps' && ! printf '%s\n' "${t_false}" | grep -qE -e '--ps-(caustics|vertex-mass)' \
         && echo 0 || echo 1)" "a switch set to true / YES / on is on, to Off / false off, as in the binaries (no longer a silent off for anything but 1)"

echo ""
echo "(U) the run mode and the GPU share: passed on as given; the pipeline hands the caller's own words on."
# Real runs of the copied drivers beside the stand-in binaries of (M), on SDATA's SIM 99 (pass_of, (S)): no driver
# resolves the run mode. '--run-mode MODE' is passed only when the option names one, whatever the word (the binary
# checks it; 'auto' passes nothing); DTFE_RUN_MODE reaches the binary as the caller set it (the binary reads it, and a
# given '--run-mode' beats it there); the banner names what was given, and a kept OMP_NUM_THREADS
u_bad=""
for scr in run_dtfe.sh run_ps_dtfe.sh; do
    # '<what pass_fields gives>|<the knobs and options>'
    for c in "unset/unset/--run-mode gentle|${PICK}|--run-mode gentle|-- --run-mode gentle" \
             "unset/unset/--run-mode performance|${PICK}|--run-mode performance|-- --run-mode=performance" \
             "unset/unset/|${PICK}|${MODE_DEFAULT}|-- --run-mode auto" \
             "unset/unset/--run-mode fast|${PICK}|--run-mode fast|-- --run-mode fast" \
             "unset/GENTLE/|${PICK}|DTFE_RUN_MODE=GENTLE|DTFE_RUN_MODE=GENTLE" \
             "unset/auto/|${PICK}|DTFE_RUN_MODE=auto|DTFE_RUN_MODE=auto" \
             "unset/fast/|${PICK}|DTFE_RUN_MODE=fast|DTFE_RUN_MODE=fast" \
             "unset/gentle/--run-mode performance|${PICK}|--run-mode performance|DTFE_RUN_MODE=gentle -- --run-mode performance" \
             "unset/gentle/|${PICK}|DTFE_RUN_MODE=gentle|DTFE_RUN_MODE=gentle -- --run-mode auto" \
             "unset/unset/|${PICK}|${MODE_DEFAULT} (OMP_NUM_THREADS=3 keeps its thread count)|THREADS=3"; do
        # shellcheck disable=SC2086
        got="$(pass_of "${scr}" ${c##*|})"
        [ "${got}" = "${c%|*}" ] || u_bad="${u_bad} [${scr} ${c##*|}: ${got}]"
    done
done
check_rc "$([ -z "${u_bad}" ] && echo 0 || echo 1)" \
      "both drivers pass '--run-mode MODE' as given (=MODE too; any word; auto: nothing) and DTFE_RUN_MODE in the environment as given (GENTLE, auto, fast; beside --run-mode too), the banner naming '--run-mode MODE', 'DTFE_RUN_MODE=<value>' or '${MODE_DEFAULT}', and a kept OMP_NUM_THREADS${u_bad:+ -- differ:${u_bad}}"
# the GPU share: run_ps_dtfe.sh passes '--gpu-share GPU_SHARE' exactly when GPU_SHARE is not auto (unset is auto: the
# binary's own default), whatever the device and the deposit (the binary says why a number does not apply);
# run_dtfe.sh passes none
u_bad=""
for c in "unset/unset/--gpu-share 0.5|${PICK}|${MODE_DEFAULT}|GPU_SHARE=0.5" \
         "unset/unset/--gpu-share mixed|${PICK}|${MODE_DEFAULT}|GPU_SHARE=mixed" "unset/unset/|${PICK}|${MODE_DEFAULT}|GPU_SHARE=auto" \
         "0/unset/--gpu-share 0.5|DTFE_GPU=0|${MODE_DEFAULT}|DTFE_GPU=0 GPU_SHARE=0.5" \
         "unset/unset/--gpu-share 0.5|${PICK}|${MODE_DEFAULT}|DTFE_EXACT_AVERAGE=0 GPU_SHARE=0.5"; do
    # shellcheck disable=SC2086
    got="$(pass_of run_ps_dtfe.sh ${c##*|})"
    [ "${got}" = "${c%|*}" ] || u_bad="${u_bad} [${c##*|}: ${got}]"
done
got="$(pass_of run_dtfe.sh GPU_SHARE=0.5)"
[ "${got}" = "unset/unset/|${PICK}|${MODE_DEFAULT}" ] || u_bad="${u_bad} [run_dtfe.sh GPU_SHARE=0.5: ${got}]"
rm -rf "${SSD}/output."* "${SSD}/ps_output."*
check_rc "$([ -z "${u_bad}" ] && echo 0 || echo 1)" \
      "run_ps_dtfe.sh passes '--gpu-share' for GPU_SHARE=0.5 and mixed (beside DTFE_GPU=0 and the sampled deposit too), none for auto or unset; run_dtfe.sh none${u_bad:+ -- differ:${u_bad}}"
# run_ps_pipeline.sh: real runs (FORCE=1) of the copy (P) made, on (P)'s SIM and plane, beside the stand-ins. It
# resolves neither the device nor the run mode: its run_ps_dtfe.sh call carries no -m and no DTFE_GPU or DTFE_RUN_MODE
# word, and the caller's DTFE_GPU, DTFE_RUN_MODE and GPU_SHARE reach run_ps_dtfe.sh, and its binary, in the
# environment; its header names what was given
ppass_of() {   # [VAR=value ...]: pass_fields of a pipeline run, its output in u_ppass.log
    env -u DTFE_GPU -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID -u DTFE_EXACT_AVERAGE \
        -u SUBSAMPLES -u DTFE_PRECISION -u OMP_NUM_THREADS -u THREADS DTFE_DATA_ROOT="${PDATA}" \
        SIMS=SIM NU=8 GRID_SIZE=16 FIELDS="density_a velocity_a" FORCE=1 FAKE_MODE=ok FAKE_ENV=1 "$@" \
        "${MREPO}/scripts/run_ps_pipeline.sh" >"${TMP}/u_ppass.log" 2>&1 || true
    pass_fields "${TMP}/u_ppass.log"
}
u_bad=""
for c in "unset/unset/|${PICK}|${MODE_DEFAULT}|" "0/unset/|DTFE_GPU=0|${MODE_DEFAULT}|DTFE_GPU=0" \
         "1/unset/|DTFE_GPU=1|${MODE_DEFAULT}|DTFE_GPU=1" \
         "unset/gentle/|${PICK}|DTFE_RUN_MODE=gentle|DTFE_RUN_MODE=gentle" \
         "unset/unset/--gpu-share 0.5|${PICK}|${MODE_DEFAULT}|GPU_SHARE=0.5" \
         "unset/unset/--gpu-share mixed|${PICK}|${MODE_DEFAULT}|GPU_SHARE=mixed" "unset/unset/|${PICK}|${MODE_DEFAULT}|GPU_SHARE=auto"; do
    # shellcheck disable=SC2086
    got="$(ppass_of ${c##*|})"
    grep -aqE -e '^\+ env .*( DTFE_GPU=| DTFE_RUN_MODE=| -m( |$))' "${TMP}/u_ppass.log" && got="${got} (a device or run-mode word in its call)"
    [ "${got}" = "${c%|*}" ] || u_bad="${u_bad} [${c##*|}: ${got}]"
done
check_rc "$([ -z "${u_bad}" ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh hands DTFE_GPU (unset, 0, 1), DTFE_RUN_MODE and GPU_SHARE (0.5, mixed, auto) to run_ps_dtfe.sh and its binary as given, no -m and no DTFE_GPU or DTFE_RUN_MODE word in its call; the header names '${PICK}', 'DTFE_GPU=<value>', '${MODE_DEFAULT}' or 'DTFE_RUN_MODE=<value>'${u_bad:+ -- differ:${u_bad}}"
rm -rf "${SDATA}"

echo ""
echo "(N) the Makefile stamps BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS in the object dirs."
# Without the stamp, 'make DTFE BUILD_MODE=debug' after a release build would find every object up to date and link
# the -O3 objects with the sanitizers (an ARCH_FLAGS / EXTRA_FLAGS change would keep the old objects the same way).
# The stamp's name carries the mode and a checksum of the three, so a switch wipes the objects. A dry run of the
# mode check alone (make -n -p) prints the stamp and builds nothing; the PS-DTFE target would start its recursive
# make, which make runs even under -n.
stamp() { make -n -p ps_gpu_mode_check "$@" 2>/dev/null | sed -n 's/^FLAGS_STAMP := //p' | head -1 || true; }
s_rel="$(stamp)"; s_dbg="$(stamp BUILD_MODE=debug)"
s_ext="$(stamp EXTRA_FLAGS=-DDTFE_STAMP_PROBE)"; s_arch="$(stamp ARCH_FLAGS=-mtune=stamp-probe)"
s_quote="$(stamp "EXTRA_FLAGS=-DQ='1'")"
check_rc "$([[ "${s_rel}" =~ ^\.flags_release_[0-9]+$ ]] && [[ "${s_dbg}" =~ ^\.flags_debug_[0-9]+$ ]] \
         && [ "${s_rel#.flags_release_}" != "${s_dbg#.flags_debug_}" ] && echo 0 || echo 1)" \
      "BUILD_MODE=debug and the default release print different stamps (${s_rel} / ${s_dbg})"
check_rc "$([[ "${s_ext}" =~ ^\.flags_release_[0-9]+$ ]] && [[ "${s_arch}" =~ ^\.flags_release_[0-9]+$ ]] \
         && [ "${s_ext}" != "${s_rel}" ] && [ "${s_arch}" != "${s_rel}" ] && echo 0 || echo 1)" \
      "... EXTRA_FLAGS and ARCH_FLAGS change the checksum too (${s_ext} / ${s_arch})"
check_rc "$([[ "${s_quote}" =~ ^\.flags_release_[0-9]+$ ]] && echo 0 || echo 1)" "... and a quote in EXTRA_FLAGS still gives a stamp (${s_quote:-none})"
OBJ_PS="$PS_OBJ"
if [ -d "${OBJ_PS}" ]; then
    stamps="$(ls -A "${OBJ_PS}" | grep -E '^\.flags_' || true)"
    check_rc "$([ "$(printf '%s\n' "${stamps}" | grep -cE '^\.flags_[a-z]+_[0-9]+$')" = "1" ] && [ "$(printf '%s\n' "${stamps}" | grep -c .)" = "1" ] \
             && echo 0 || echo 1)" "the build of ${PS_BIN} left exactly one such stamp in ${OBJ_PS} (${stamps:-none})"
fi

echo ""
echo "------------------------------------------------------------"
if [ "${fails}" -eq 0 ]; then
    rm -rf "${TMP}"
    echo "RESULT: PASS  (the batch driver honours its environment, needs no IC file when the"
    echo "               snapshot carries one, reports honest exit codes, is portable, and"
    echo "               handles the per-family data layout; run_dtfe.sh's exact cell average;"
    echo "               downloads merge only when complete; --help and DRY_RUN run nothing)"
    exit 0
fi
echo "RESULT: FAIL  (${fails} check(s); logs kept under ${TMP})"
exit 1
