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
#     moved everyone onto the snapshot path), and an empty PS_LAGRANGIAN_INPUT= (or LAGRANGIAN_INPUT=, its older
#     name) forces it off.
#  D) A snapshot with NEITHER source is skipped with a message naming both, and the run exits 1.
#  E) Exit status is honest: processing nothing exits 1, while a PARTIAL skip stays 0 -- a
#     partly-downloaded snapshot ladder is normal.
#  F) No macOS-only command on the critical path ('/usr/bin/time -l', 'sed -i ""'), which would
#     kill every snapshot on Linux. Asserted structurally, so it holds on either platform.
#  G) The PER-FAMILY data layout (<root>/<family>/<sim>, e.g. <root>/TNG100/TNG100-3-Dark), under a
#     root whose name has a space: the driver finds and processes the simulation there, config.sh's sim_dir puts a new simulation beside its
#     family and lets a flat <root>/<sim> win, and run_ps_pipeline.sh discovers nested sims -- and exits 1
#     when it finds nothing to compute (no data root, a mistyped SIMS, a SNAPS matching nothing). Its default call of
#     run_ps_dtfe.sh: the exact deposit and the balanced run mode, on the GPU exactly when this build has one.
#  H) scripts/run_dtfe.sh passes --exact to the standard binary by default (-e, --exact and DTFE_EXACT_AVERAGE=1 too)
#     and --sampled for DTFE_EXACT_AVERAGE=0 or --sampled, or a SUBSAMPLES count (-n) while the knob is unset (then
#     with --subsamples N); -e or DTFE_EXACT_AVERAGE=1 beside a count stays exact, with one note; -e and --sampled
#     together are refused; --exact-average, the older long form, is -e with one warning; the older knob name
#     DTFE_EXACT still read, with a warning, and the new name wins over it. Its padding default is 15 (a caller's
#     PADDING / DTFE_PADDING wins), run_ps_dtfe.sh keeps
#     config.sh's 50. run_dtfe.sh AUTO_TUNE_REPORT=1 (as
#     run_ps_dtfe.sh): the standard binary's own report line per snapshot ('snap=099'), exit 0, nothing written.
#  I) download_snapshots.sh + merge_HDF5.py never turn a half-finished download into a merged snapshot; a chunk the
#     data host refuses once is fetched on the retry ('attempt 2 of 3', exit 0), one refused every time is an error.
#  J) --help runs nothing (run_ps_pipeline.sh --help must not start the whole batch), --version prints the
#     revision, an argument to the pipeline is an error, DRY_RUN=1 prints the drivers' commands and runs nothing.
#     There is no default data root or simulation: every driver without DTFE_DATA_ROOT (or without -s / DTFE_SIM /
#     SIMS) refuses with exit 2 and says so, -d or not, while --help still works and sourcing config.sh exits nothing.
#     The drivers' shared option loop (config.sh driver_options): a bad option exits 1 with the usage, -h anywhere 0;
#     the long forms are the letters, --run-mode MODE in both drivers (download_snapshots.sh refuses it).
#  K) the binaries' own --help lists the phase-space / GPU / point-evaluation / server options the scripts and the
#     launcher use (--exact, --sampled and --run-mode in both, --gpu-share in PS-DTFE's), and the help texts carry
#     none of the known misspellings. The
#     deprecated spellings (--ps-gpu, --exact-average, --ps-exact-deposit, --ps-no-overlap, --lambda_th, --full_help)
#     are hidden from the help but still accepted, on
#     the command line and in a --config file: one 'deprecated' warning line each and the same outputs as the
#     canonical names; an old and a new spelling with different values from one source are refused, while across
#     the command line and the file the command line wins (as for any option), whichever spelling each uses.
#  L) scripts/install_lib.sh keeps a user's binaries through a failed rebuild: backup, restore (the 2D double pair
#     too), the failure hint; install.sh's detection of the extra sets it rebuilds (sourced alone, never run).
#  M) the run-log contract of run_dtfe.sh / run_ps_dtfe.sh, against stand-in binaries: a run's log is
#     <prefix>.runlog.partial while it goes, <prefix>.runlog after exit 0, <prefix>.runlog.failed after a
#     failure or a stop; a failed, stopped or killed re-run never touches the .runlog of the grids on disk.
#  N) the Makefile stamps BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS in the object dirs (a dry run, builds nothing).
#  O) the drivers' single-instance lock (<prefix>.runlog.lock: a second driver refused, stale locks removed, a stop
#     releases it) and their preflight (data disk mounted, output folder writable with room for the grids).
#  P) run_ps_pipeline.sh's freshness test compares the run settings with the snapshot's run log, not only file times
#     (the plane's conventions too, PS_POINT_CONVENTIONS, which it hands to run_ps_dtfe.sh); a run log written with the
#     binaries' deprecated spellings reads the same as the canonical command the driver now emits (unchanged). The
#     deposit: an older exact log ('--subsamples 3 --ps-exact-deposit' / '--exact-average') matches '--exact', an older
#     sampled one (a count, no flag) matches '--sampled' with that count or none, and not the exact default.
#  R) run_ps_dtfe.sh PS_MERGE: unset passes '--ps-merge index-order' (the driver's default, given explicitly so the
#     memory gate stays on under the auto-tuner's speed split), a given mode passes '--ps-merge MODE', the older
#     ORDERED_MERGE=1 / 0 and
#     FIXED_POINT_MERGE=1 convert (fixed-point winning, as before), another value is refused;
#     PS_POINT_CONVENTIONS=legacy with SAMPLE_POINTS passes '--ps-point-conventions legacy', grid (the default) and a
#     run without points pass nothing, another value is refused; a killed run's checkpoint (<prefix>.runlog.checkpoint,
#     or config.sh ps_checkpoint_dir's name under SCRATCH_DIR) makes the next run pass --resume and say so; DTFE_RESUME=0
#     (PS_RESUME=0, its older name), an auto-tune report and a checkpoint folder with no 'checkpoint' file in it do not.
#  T) the knob names (config.sh knob, P6.7): each older name alone gives the same command as its new name, with one
#     'warning: OLD is deprecated, use NEW' line on stderr; both set: the new one wins (a warning, no refusal); a
#     switch reads 0/false/no/off as off and any other value as on, as the binaries do; the drivers' long options
#     (--sim --sim-dir --grid --gpu --exact --subsamples --run-mode) are their letters, --sampled the sampled deposit,
#     and --exact-average (the older --exact) is -e with one warning (J runs them; download_snapshots.sh has no
#     --run-mode).
#  S) the production defaults and the rerun hygiene, against the stand-in binaries:
#     the GPU when the Makefile's stamp says the binary has one (metal; not off; DTFE_GPU=0 or 1
#     wins, and so do its older names DTFE_METAL / PS_GPU / PS_METAL when it is unset), by run mode: PS-DTFE's two
#     deposits and standard DTFE's exact average on the GPU in every mode, standard DTFE's sampled average in gentle
#     and balanced, the CPU for it in performance (config.sh gpu_auto_resolve, said in the banner); the drivers'
#     production flags by default (run_ps_dtfe.sh: volume weighting, caustics, the four analysis
#     fields, the exact deposit (--exact; a SUBSAMPLES count or DTFE_EXACT_AVERAGE=0: the sampled one); run_dtfe.sh:
#     the balanced run mode, the exact average and the same fields; PS_LINEAR_DEPOSIT=1 drops the
#     volume weighting; FACE_GRADIENT=1 passes --face-gradient, run_ps_dtfe.sh refusing it with the sampled deposit or
#     without the volume weighting),
#     run_dtfe.sh's OUTPUT_PREFIX (a 'ps_*' one refused), and a rerun's left-over files: after exit 0 the files the run
#     did not write are MOVED to <prefix>.superseded/<stamp>/ with the old run log (the rewritten ones, the run log
#     family, a lock and a checkpoint stay); a failed run, a dry run and an auto-tune report move nothing.
#     run_ps_pipeline.sh's DTFE_GPU unset follows the PS binary's stamp the same way, for both deposits.
#  U) the run mode (config.sh run_mode_resolve), against the stand-in binaries: both drivers pass '--run-mode MODE' on
#     every run, balanced by default; --run-mode MODE (or =MODE) first, then DTFE_RUN_MODE in any case, then the
#     default, auto (or empty) passing on to the next, performance inside a SLURM or PBS job; the banner names the
#     source and a kept OMP_NUM_THREADS; a bad option value exits 1 with the usage, a bad knob 2. run_ps_dtfe.sh passes
#     '--gpu-share GPU_SHARE' (auto by default) with the exact deposit on the GPU, and a share given for any other run;
#     run_dtfe.sh's sampled average in performance runs on the CPU; run_ps_pipeline.sh resolves the mode once and hands
#     DTFE_RUN_MODE on after DTFE_EXACT_AVERAGE.
#
# Usage: tests/run_scripts_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the drivers run the double pair
precision_double && export DTFE_PRECISION=double     # the drivers' own switch (run_ps_dtfe.sh / run_dtfe.sh)

export PYTHON="${PY}"                   # (tests/lib.sh) run_ps_pipeline.sh runs make_image_plane.py with the same python

N="${N:-16}"
GRID="${GRID:-24}"
BOX="${BOX:-100.0}"
TMP="${TMP_DIR:-${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}}/run_scripts"
export DTFE_FIGURES_ROOT="${TMP}/figures"   # the pipeline's plots: never the caller's figure root
DRIVER="scripts/run_ps_dtfe.sh"

echo "============================================================"
echo " run_ps_dtfe.sh contract check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} not built"; exit 1; }

fails=0
check() {
    if [ "$1" = "0" ]; then printf '   PASS  %s\n' "$2"; else printf '   FAIL  %s\n' "$2"; fails=$((fails+1)); fi
}
# Does <file> ($1) match extended-regex <pattern> ($2)? Echo 0 for yes, 1 for no, so the result
# feeds check() directly. '-e' is mandatory here: nearly every pattern below starts with '--'
# (the flags we are asserting about), which grep would otherwise read as its own option.
has()   { grep -aqE -e "$2" "$1" && echo 0 || echo 1; }
lacks() { grep -aqE -e "$2" "$1" && echo 1 || echo 0; }

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
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "env-only run succeeds (exit ${rc})"
check "$(has "${TMP}/a_env.log" '--grid 24( |$)')" "GRID_SIZE=24 reached the binary as '--grid 24'"
check "$(has "${TMP}/a_env.log" 'Grid size: 24')" "GRID_SIZE=24 is what the script reports"
if precision_double; then
    check "$(has "${TMP}/a_env.log" 'PS-DTFE-double ')" "DTFE_PRECISION=double: the driver ran ./PS-DTFE-double"
fi
# -g must win over the environment
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_SIM=HASIC "${DRIVER}" -g 32 99 >"${TMP}/a_flag.log" 2>&1 || true
check "$(has   "${TMP}/a_flag.log" '--grid 32( |$)')" "-g 32 beats GRID_SIZE=24 (flag > environment)"
check "$(lacks "${TMP}/a_flag.log" '--grid 24( |$)')" \
      "the overridden GRID_SIZE=24 appears nowhere in the command"
# and with neither, config.sh's own default stands
default_grid=$(bash -c 'source scripts/config.sh; echo "${GRID_SIZE}"')
check "$([ "${default_grid}" = "512" ] && echo 0 || echo 1)" \
      "config.sh's default is the production 512 when nothing overrides it (got ${default_grid})"

echo ""
echo "(B) a snapshot with its own 'InitialCoordinates' runs WITHOUT a combined_ics.hdf5."
check "$(has "${TMP}/a_env.log" "Lagrangian positions: the snapshot's own 'InitialCoordinates'")" \
      "the script says it is using the snapshot's own Lagrangian positions"
check "$(has "${TMP}/a_env.log" "read from 'InitialCoordinates' dataset in main input file")" \
      "the BINARY confirms it read them from the snapshot"
check "$(lacks "${TMP}/a_env.log" '--(ps-lagrangian-input|lagrangianInput)')" "no --ps-lagrangian-input was passed"
check "$([ -f "${TMP}/HASIC/snapdir_099/ps_output.a_den" ] && echo 0 || echo 1)" \
      "the density grid was actually written"

echo ""
echo "(C) a combined_ics.hdf5 beside the snapshot is still used, and can be forced off."
cp "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/HASIC/combined_ics.hdf5"
rc=$(drive "${TMP}/c_ics.log" GRID_SIZE=24 DTFE_SIM=HASIC)
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "run with an IC file succeeds (exit ${rc})"
check "$(has "${TMP}/c_ics.log" '--ps-lagrangian-input .*combined_ics\.hdf5')" \
      "--ps-lagrangian-input points at the IC file"
check "$(lacks "${TMP}/c_ics.log" 'is deprecated')" "... the driver emits no deprecated flag (the binary warns of none)"
# explicitly empty PS_LAGRANGIAN_INPUT must fall back to the snapshot even though the file exists
rc=$(drive "${TMP}/c_forced.log" GRID_SIZE=24 DTFE_SIM=HASIC PS_LAGRANGIAN_INPUT=)
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "PS_LAGRANGIAN_INPUT= run succeeds (exit ${rc})"
check "$(has "${TMP}/c_forced.log" "read from 'InitialCoordinates' dataset in main input file")" \
      "PS_LAGRANGIAN_INPUT= forces the snapshot path despite the IC file"
# the older name, empty, does the same (config.sh knob -e: an empty value is a value here), with its warning
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_SIM=HASIC LAGRANGIAN_INPUT= DRY_RUN=1 "${DRIVER}" 99 >"${TMP}/c_forced_old.log" 2>&1 || true
check "$([ "$(lacks "${TMP}/c_forced_old.log" '--ps-lagrangian-input')" = 0 ] \
         && [ "$(grep -c '^warning: LAGRANGIAN_INPUT is deprecated, use PS_LAGRANGIAN_INPUT$' "${TMP}/c_forced_old.log")" = 1 ] && echo 0 || echo 1)" \
      "... and so does LAGRANGIAN_INPUT=, the older name, with one deprecation line"
rm -f "${TMP}/HASIC/combined_ics.hdf5"

echo ""
echo "(D) a snapshot with NEITHER Lagrangian source is skipped, naming both, and exits 1."
rc=$(drive "${TMP}/d_none.log" GRID_SIZE=24 DTFE_SIM=NOIC)
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "exit status is 1 (got ${rc})"
check "$(has "${TMP}/d_none.log" 'has NO Lagrangian positions')" "the warning names the real problem"
check "$(has "${TMP}/d_none.log" "no 'InitialCoordinates' dataset either")" \
      "it says the snapshot itself was checked too"
check "$(has "${TMP}/d_none.log" 'Skipping snapshot 099')" "the snapshot is skipped rather than failed"

echo ""
echo "(E) exit status is honest about doing nothing, and forgiving about partial ladders."
rc=$(drive "${TMP}/e_nothing.log" GRID_SIZE=24 DTFE_SIM=DOES_NOT_EXIST)
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a wrong DTFE_SIM exits 1, not 0 (got ${rc})"
check "$(has "${TMP}/e_nothing.log" '0 processed')" "the summary reports 0 processed"
# 98 is absent, 99 is fine: a partly-downloaded ladder must still succeed
rc=0
env "${COMMON_ENV[@]}" GRID_SIZE=24 DTFE_SIM=PARTIAL "${DRIVER}" 98 99 >"${TMP}/e_partial.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a partial ladder still exits 0 (got ${rc})"
check "$(has "${TMP}/e_partial.log" '1 processed, 1 skipped')" "the summary counts both outcomes"

echo ""
echo "(F) no macOS-only command sits on the critical path (so Linux/CI can run this driver)."
# The test is about the INVOCATION, not the mention: a guarded probe of '/usr/bin/time -l' is
# exactly how the portable wrapper decides which dialect exists, so grepping for the string alone
# would fail on the very code that fixes the problem. What must not exist is time(1) sitting
# directly in front of the binary.
check "$(lacks "${DRIVER}" '/usr/bin/time[^|]*\./PS-DTFE')" \
      "time(1) does not prefix the binary unconditionally (GNU spells -l as -v; containers have neither)"
check "$(has   "${DRIVER}" 'TIME_WRAP\[@\].*"\${cmd\[@\]}"')" \
      "the binary is invoked through the resolved-once TIME_WRAP instead (PS_BIN: the pair chosen by DTFE_PRECISION)"
check "$(lacks "${DRIVER}" "sed .*-i +''")" \
      "no BSD-only \"sed -i ''\" (GNU sed reads the '' as the script, not as the suffix)"
# and the run really did produce its log without a time(1) present
check "$([ -f "${TMP}/HASIC/snapdir_099/ps_output.runlog" ] && echo 0 || echo 1)" \
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
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a run on the per-family layout succeeds (exit ${rc})"
check "$([ -f "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.a_den" ] && echo 0 || echo 1)" \
      "its grids land in <root>/FAM/FAM-1-Dark/snapdir_099"
# sim_dir: an unknown simulation of an existing family goes beside its siblings; a flat
# <root>/<sim> directory, when present, wins over the family folder
newsim="$(DTFE_DATA_ROOT="${FAMROOT}" bash -c 'source scripts/config.sh; sim_dir FAM-2-Dark')"
check "$([ "${newsim}" = "${FAMROOT}/FAM/FAM-2-Dark" ] && echo 0 || echo 1)" \
      "sim_dir places a new FAM-2-Dark in its family folder (got '${newsim}')"
mkdir -p "${FAMROOT}/FAM-3-Dark"
flat="$(DTFE_DATA_ROOT="${FAMROOT}" bash -c 'source scripts/config.sh; sim_dir FAM-3-Dark')"
check "$([ "${flat}" = "${FAMROOT}/FAM-3-Dark" ] && echo 0 || echo 1)" \
      "an existing flat <root>/FAM-3-Dark wins (got '${flat}')"
G_AVG=(-u DTFE_EXACT_AVERAGE -u PS_EXACT -u SUBSAMPLES -u AVG_SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID)   # the deposit's and the run mode's knobs: the defaults here
rc=0; env "${G_AVG[@]}" DRY_RUN=1 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline.log" 2>&1 || rc=$?
check "$(has "${TMP}/g_pipeline.log" '^Simulations:.* FAM-1-Dark')" \
      "run_ps_pipeline.sh discovers the nested FAM-1-Dark"
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... and a dry run with a snapshot to plan exits 0 (${rc})"
# the pipeline's knobs: the GPU flag follows DTFE_GPU, the plane's geometry reaches make_image_plane.py, and a plane
# whose sidecar has another geometry is regenerated. DTFE_GPU unset: the GPU exactly when this repository's PS binary
# is a GPU build (config.sh gpu_auto_resolve; (S) varies the stamp and the run mode), for the exact deposit (the
# default, handed on as DTFE_EXACT_AVERAGE=1) and the sampled one (a SUBSAMPLES count) alike; the run mode (balanced,
# the default) is handed on after the deposit
g_bin="$([ "${DTFE_PRECISION:-single}" = double ] && echo PS-DTFE-double || echo PS-DTFE)"
g_gpu="$(bash -c 'source scripts/config.sh >/dev/null 2>&1; gpu_build_default "$1" "$2"' _ "${ROOT_DIR}" "${g_bin}")"
env "${G_AVG[@]}" DRY_RUN=1 SUBSAMPLES=2 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline_s.log" 2>&1 || true
if [ "${g_gpu}" = "1" ]; then
    check "$([ "$(has "${TMP}/g_pipeline.log" 'DTFE_GPU=1 DTFE_EXACT_AVERAGE=1 DTFE_RUN_MODE=balanced .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ -m 99')" = 0 ] \
             && [ "$(has "${TMP}/g_pipeline.log" '^Slice: .* deposit: exact \(the default\) ')" = 0 ] \
             && [ "$(has "${TMP}/g_pipeline.log" '^Run mode: balanced \(the default\)$')" = 0 ] && echo 0 || echo 1)" \
          "the pipeline's default run_ps_dtfe.sh call: the exact deposit (DTFE_EXACT_AVERAGE=1) in the balanced run mode, both said in the header, with -m (${g_bin} is a GPU build here)"
    check "$(has "${TMP}/g_pipeline_s.log" 'DTFE_GPU=1 DTFE_EXACT_AVERAGE=0 DTFE_RUN_MODE=balanced .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ -m 99')" \
          "... SUBSAMPLES=2 alone: the sampled deposit (DTFE_EXACT_AVERAGE=0 handed on), with -m (${g_bin} is a GPU build here)"
else
    check "$([ "$(has "${TMP}/g_pipeline.log" 'DTFE_GPU=0 DTFE_EXACT_AVERAGE=1 DTFE_RUN_MODE=balanced .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ 99')" = 0 ] \
             && [ "$(has "${TMP}/g_pipeline.log" '^Slice: .* deposit: exact \(the default\) ')" = 0 ] \
             && [ "$(has "${TMP}/g_pipeline.log" '^Run mode: balanced \(the default\)$')" = 0 ] && echo 0 || echo 1)" \
          "the pipeline's default run_ps_dtfe.sh call: the exact deposit (DTFE_EXACT_AVERAGE=1) in the balanced run mode, both said in the header, no -m (${g_bin} is a CPU-only build here)"
    check "$(has "${TMP}/g_pipeline_s.log" 'DTFE_GPU=0 DTFE_EXACT_AVERAGE=0 DTFE_RUN_MODE=balanced .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ 99')" \
          "... SUBSAMPLES=2 alone: the sampled deposit (DTFE_EXACT_AVERAGE=0 handed on), no -m (${g_bin} is a CPU-only build here)"
fi
check "$(has "${TMP}/g_pipeline.log" 'make_image_plane.py .*--planes 1 --thickness 2 --supersample 1 -o ')" \
      "... and makes a single plane without sub-sampling"
DRY_RUN=1 DTFE_GPU=0 PS_VOLUME_WEIGHTED=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 PLANE_WINDOW="0 50 0 25" CENTER=40 \
    SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline2.log" 2>&1 || true
check "$(has "${TMP}/g_pipeline2.log" 'PS_VOLUME_WEIGHTED=0 .*run_ps_dtfe.sh -s FAM-1-Dark -g [0-9]+ 99')" \
      "DTFE_GPU=0 drops -m and PS_VOLUME_WEIGHTED=0 is passed on"
check "$(has "${TMP}/g_pipeline2.log" 'make_image_plane.py .*--axis z --nu [0-9]+ --planes 4 --thickness 3 --supersample 2 --center 40 --plane-window 0 50 0 25 -o ')" \
      "PLANES, THICKNESS, SUPERSAMPLE, CENTER and PLANE_WINDOW reach make_image_plane.py"
# the older names (WINDOW, PS_GPU): the same plan, one deprecation line each
DRY_RUN=1 PS_GPU=0 PS_VOLUME_WEIGHTED=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 WINDOW="0 50 0 25" CENTER=40 \
    SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline2o.log" 2>&1 || true
check "$([ "$(grep -v '^warning: ' "${TMP}/g_pipeline2o.log")" = "$(cat "${TMP}/g_pipeline2.log")" ] \
         && [ "$(grep -c '^warning: WINDOW is deprecated, use PLANE_WINDOW$' "${TMP}/g_pipeline2o.log")" = 1 ] \
         && [ "$(grep -c '^warning: PS_GPU is deprecated, use DTFE_GPU$' "${TMP}/g_pipeline2o.log")" = 1 ] && echo 0 || echo 1)" \
      "... WINDOW and PS_GPU, the older names, plan the same, with one deprecation line each"
# run_ps_dtfe.sh's own older knob names (the launcher still exports them): the pipeline resolves them once and hands
# them on under the new names, so each is warned exactly once however many simulations its driver plans
DRY_RUN=1 DTFE_GPU=0 PS_VOLUME_WEIGHTED=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 PLANE_WINDOW="0 50 0 25" CENTER=40 \
    AVG_SUBSAMPLES=2 ORDERED_MERGE=1 PS_RESUME=0 TESS_CACHE="${TMP}/g_tc" PS_PARALLEL_TRI=1 PS_LAG_PADDING=4 \
    LAGRANGIAN_INPUT= SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline2k.log" 2>&1 || true
g_k=""
for k in AVG_SUBSAMPLES ORDERED_MERGE PS_RESUME TESS_CACHE PS_PARALLEL_TRI PS_LAG_PADDING LAGRANGIAN_INPUT; do
    [ "$(grep -c "^warning: ${k} is deprecated, use " "${TMP}/g_pipeline2k.log")" = 1 ] || g_k="${g_k} ${k}"
done
DRY_RUN=1 DTFE_GPU=0 PS_VOLUME_WEIGHTED=0 PLANES=4 THICKNESS=3 SUPERSAMPLE=2 PLANE_WINDOW="0 50 0 25" CENTER=40 \
    SUBSAMPLES=2 PS_MERGE=index-order DTFE_RESUME=0 TESSELLATION_CACHE="${TMP}/g_tc" PARALLEL_TRIANGULATION=1 \
    PS_LAGRANGIAN_PADDING=4 PS_LAGRANGIAN_INPUT= SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh \
    >"${TMP}/g_pipeline2n.log" 2>&1 || true
check "$([ -z "${g_k}" ] && [ ! -e "${TMP}/g_tc" ] && [ "$(grep -v '^warning: ' "${TMP}/g_pipeline2k.log")" = "$(cat "${TMP}/g_pipeline2n.log")" ] \
         && ! grep -q 'deprecated' "${TMP}/g_pipeline2n.log" && echo 0 || echo 1)" \
      "... run_ps_dtfe.sh's older knob names, resolved once by the pipeline: one deprecation line each, none repeated per simulation (${g_k:- ok}), the plan of the new names (LAGRANGIAN_INPUT= empty too); a dry run creates no cache folder"
# finding nothing to compute is a failure: with the data drive unmounted, a mistyped SIMS or a SNAPS
# matching no snapshot, an ALL DONE with exit 0 would show in the launcher as a success
mkdir -p "${TMP}/empty root"
rc=0; env -u SIMS DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_nosims.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/g_pipe_nosims.log" 'no simulation given: set SIMS')" = 0 ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh without SIMS refuses with exit 2: there is no default simulation (SIMS=all discovers) (${rc})"
rc=0; SIMS=all DTFE_DATA_ROOT="${TMP}/empty root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_none.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "run_ps_pipeline.sh with no simulation under the data root exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_none.log" '^!! nothing to compute')" "... says there is nothing to compute"
check "$(lacks "${TMP}/g_pipe_none.log" 'ALL DONE')" "... and not ALL DONE"
rc=0; SIMS=all DTFE_DATA_ROOT="${TMP}/no such root" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_unmounted.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a data root that does not exist (the drive unplugged) exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_unmounted.log" 'is the data drive mounted')" "... and asks whether the drive is mounted"
rc=0; DRY_RUN=1 SIMS=FAM-1-Drak DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_typo.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a mistyped SIMS exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_typo.log" '^!! nothing to compute: no combined snapshot of FAM-1-Drak ')" "... naming the simulation"
rc=0; DRY_RUN=1 SNAPS="50 67" SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_snaps.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a SNAPS matching no snapshot exits 1 (${rc})"
check "$(has "${TMP}/g_pipe_snaps.log" 'FAM-1-Dark: no combined snapshots numbered 50 67, skipping')" "... naming the numbers it looked for"
rc=0; DRY_RUN=1 SIMS="FAM-1-Drak FAM-1-Dark" DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipe_half.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "one simulation of two without snapshots is a partial skip: exit 0 (${rc})"
# a real plane (tiny), then the same geometry is reused and another one regenerates it
"${PY}" python/tools/make_image_plane.py --combined "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/combined_099.hdf5" \
    --mpc-unit 1 --axis z --nu 8 --planes 1 -o "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8" >/dev/null 2>&1 || true
if [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ]; then
    DRY_RUN=1 NU=8 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline3.log" 2>&1 || true
    check "$(has "${TMP}/g_pipeline3.log" '^-- reusing .*hires_plane_z8.bin')" \
          "an existing plane of the requested geometry is reused"
    DRY_RUN=1 NU=8 PLANES=16 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline4.log" 2>&1 || true
    check "$(has "${TMP}/g_pipeline4.log" 'geometry .* differs; regenerating')" \
          "another plane count regenerates the plane (and so every snapshot)"
    check "$([ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.bin" ] && echo 0 || echo 1)" \
          "... but a DRY_RUN removes nothing"
    # a comparison that FAILS (here: no python) is not 'differs': the plane stays, the sim is skipped with !!
    rc5=0; PYTHON=/nonexistent/python3 NU=8 SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline5.log" 2>&1 || rc5=$?
    check "$(has "${TMP}/g_pipeline5.log" '^!! FAM-1-Dark: could not compare the plane')" \
          "a failing geometry check is reported, not read as 'differs'"
    check "$([ "${rc5}" != "0" ] && [ -f "${FAMROOT}/FAM/FAM-1-Dark/hires_plane_z8.json" ] && echo 0 || echo 1)" \
          "... the plane is kept and the pipeline exits non-zero (${rc5})"
    # every snapshot up to date is a success, not 'nothing to compute'; a slice of the 8^2 plane holds
    # 64 points (the pipeline compares the count)
    head -c 512 /dev/zero > "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den"
    touch -t 203001010000 "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den" \
                          "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_velGrad"
    # the settings of the (A)-style run that made these grids: the pipeline compares them too (P)
    rc6=0; DRY_RUN=1 NU=8 GRID_SIZE=24 FIELDS=density_a SUBSAMPLES=1 PS_VOLUME_WEIGHTED=1 PTS_VEL_GRAD=0 \
        SIMS=all DTFE_DATA_ROOT="${FAMROOT}" scripts/run_ps_pipeline.sh >"${TMP}/g_pipeline6.log" 2>&1 || rc6=$?
    check "$([ "${rc6}" = "0" ] && echo 0 || echo 1)" "every snapshot up to date still exits 0 (${rc6})"
    check "$(has "${TMP}/g_pipeline6.log" 'FAM-1-Dark: all snapshots up to date')" "... with nothing to recompute"
    rm -f "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_den" "${FAMROOT}/FAM/FAM-1-Dark/snapdir_099/ps_output.pts_velGrad"
else
    echo "   SKIP  plane reuse/regeneration (make_image_plane.py could not write a plane with ${PY})"
fi

echo ""
echo "(H) run_dtfe.sh runs the standard binary with --exact by default (-e, DTFE_EXACT_AVERAGE=1), --sampled for DTFE_EXACT_AVERAGE=0 or a count alone."
if [ -x "${DTFE_BIN}" ]; then
    rc=0; scripts/run_dtfe.sh -s HASIC -g 16 -e 99 >"${TMP}/h_exact.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh -e succeeds (exit ${rc})"
    check "$(has "${TMP}/h_exact.log" 'RUNNING: .*--exact( |$)')" "the binary was called with --exact"
    if precision_double; then
        check "$(has "${TMP}/h_exact.log" 'RUNNING: .*DTFE-double ')" "DTFE_PRECISION=double: run_dtfe.sh ran ./DTFE-double"
    fi
    check "$(has "${TMP}/h_exact.log" 'volume averaging +: EXACT cell-tetrahedron integration of the linear interpolant \(--exact;')" \
          "... and reports the exact cell average, asked for by --exact"
    # what the wrapper run_ps_dtfe.sh has -- wall time, peak memory, a de-ANSI'd runlog
    check "$(has "${TMP}/h_exact.log" '^  Wall time: [0-9]+ min [0-9]+ s$')" "run_dtfe.sh reports the wall time per snapshot"
    if [ -x /usr/bin/time ]; then
        check "$(has "${TMP}/h_exact.log" '^  Peak memory: .*GB resident')" "... and the peak memory (through /usr/bin/time)"
    fi
    CLICOLOR_FORCE=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_color.log" 2>&1 || true
    check "$(lacks "${TMP}/HASIC/snapdir_099/output.runlog" $'\033\\[')" "the runlog is stripped of colour codes when colours are forced"
    check "$(has "${TMP}/HASIC/snapdir_099/output.runlog" 'total wall time')" "... and still carries the binary's own wall-time line for the Runs browser"
    # a binary that fails (a truncated input) must still be a failed snapshot: the status is taken before the strip
    mkdir -p "${TMP}/BADIN/snapdir_099"; head -c 1000 "${TMP}/HASIC/snapdir_099/combined_099.hdf5" > "${TMP}/BADIN/snapdir_099/combined_099.hdf5"
    rc_bad=0; scripts/run_dtfe.sh -s BADIN -g 16 99 >"${TMP}/h_badin.log" 2>&1 || rc_bad=$?
    check "$([ "${rc_bad}" = "1" ] && echo 0 || echo 1)" "a snapshot the binary cannot read is a FAILED snapshot, exit 1 (${rc_bad})"
    check "$(has "${TMP}/h_badin.log" 'processing complete: 0 processed, 0 skipped, 1 failed')" "... counted as failed, not processed (the exit status is read before the log is post-processed)"
    check "$([ -s "${TMP}/BADIN/snapdir_099/output.runlog.failed" ] && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog" ] \
             && [ ! -e "${TMP}/BADIN/snapdir_099/output.runlog.partial" ] && echo 0 || echo 1)" \
          "... its log is output.runlog.failed: no output.runlog claims grids it never wrote"
    check "$([ -f "${TMP}/HASIC/snapdir_099/output.a_den" ] && echo 0 || echo 1)" "its density grid was written"
    env -u DTFE_EXACT -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_env.log" 2>&1 || true
    # run_dtfe.sh reports failures in its exit status, passes the
    # launcher's scratch folder and the scripts' lambda_th default, and the binaries refuse --redshiftSpace
    rc_none=0; scripts/run_dtfe.sh -s HASIC -g 16 123 >"${TMP}/h_none.log" 2>&1 || rc_none=$?
    check "$([ "${rc_none}" != "0" ] && echo 0 || echo 1)" "run_dtfe.sh exits non-zero when no snapshot was processed (${rc_none})"
    check "$(has "${TMP}/h_none.log" 'processing complete: 0 processed, 1 skipped, 0 failed')" "... and says so in its summary line"
    SCR_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-scratch-check.XXXXXX")"   # local: the binary refuses iCloud paths (and Linux has no /private/tmp)
    rc_scr=0; SCRATCH_DIR="${SCR_DIR}" scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_scratch.log" 2>&1 || rc_scr=$?
    check "$([ "${rc_scr}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with SCRATCH_DIR succeeds (exit ${rc_scr})"
    check "$(has "${TMP}/h_scratch.log" "RUNNING: .*--scratch-dir ${SCR_DIR}")" "... and hands the folder to the binary (--scratch-dir)"
    check "$(has "${TMP}/h_scratch.log" 'RUNNING: .*--lambda-th 0.3')" "run_dtfe.sh passes the scripts' lambda-th default (0.3, the binary's own is 0)"
    # standard DTFE pads with 15 spacings by default (complete at z = 0 on TNG50/100/300), not config.sh's
    # 50, which stays the PS-DTFE scripts' value; a caller's PADDING or DTFE_PADDING still wins, PADDING first
    pad_of() { env -u PADDING -u DTFE_PADDING -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID "$@" DRY_RUN=1 "${PAD_DRIVER}" -s HASIC -g 16 99 2>&1 \
                   | grep -oE -e '--padding +[0-9]+' | head -1 | grep -oE '[0-9]+$'; }
    PAD_DRIVER=scripts/run_dtfe.sh
    pad_def="$(pad_of)"; pad_p="$(pad_of PADDING=12)"; pad_d="$(pad_of DTFE_PADDING=13)"; pad_pd="$(pad_of PADDING=11 DTFE_PADDING=13)"
    check "$([ "${pad_def}" = "15" ] && echo 0 || echo 1)" "run_dtfe.sh pads with 15 spacings by default (${pad_def:-none})"
    check "$([ "${pad_p}" = "12" ] && [ "${pad_d}" = "13" ] && [ "${pad_pd}" = "11" ] && echo 0 || echo 1)" \
          "... a caller's PADDING or DTFE_PADDING wins, PADDING first (${pad_p:-none} ${pad_d:-none} ${pad_pd:-none})"
    PAD_DRIVER=scripts/run_ps_dtfe.sh; pad_ps="$(pad_of)"
    check "$([ "${pad_ps}" = "50" ] && echo 0 || echo 1)" "... while run_ps_dtfe.sh keeps config.sh's 50 (${pad_ps:-none})"
    rm -rf "${SCR_DIR}"
    # FIELDS reaches the standard binary (the launcher's ticks: the T-web), a phase-space dispersion is left out;
    # TESSELLATION_CACHE hands a (created) local folder to --tessellation-cache
    TC_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-tess-check.XXXXXX")/cache"
    rc_f=0; FIELDS="density_a tweb_a dispersion_a" TESSELLATION_CACHE="${TC_DIR}" scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_fields.log" 2>&1 || rc_f=$?
    check "$([ "${rc_f}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with FIELDS and TESSELLATION_CACHE succeeds (exit ${rc_f})"
    check "$(has "${TMP}/h_fields.log" 'RUNNING: .*--field density_a tweb_a +--')" "... FIELDS is the field list (the T-web for standard DTFE)"
    check "$(has "${TMP}/h_fields.log" 'standard DTFE leaves it out')" "... the phase-space dispersion is left out, and it says so"
    check "$(has "${TMP}/h_fields.log" "RUNNING: .*--tessellation-cache ${TC_DIR}")" "... TESSELLATION_CACHE becomes --tessellation-cache"
    check "$([ -d "${TC_DIR}" ] && echo 0 || echo 1)" "... and the cache folder is created"
    check "$([ -f "${TMP}/HASIC/snapdir_099/output.a_tweb" ] || ls "${TMP}/HASIC/snapdir_099/"output.*tweb* >/dev/null 2>&1; echo $?)" \
        "... and the T-web grid is written"
    rm -rf "$(dirname "${TC_DIR}")"
    # SAMPLE_POINTS: the standard DTFE interpolant at a points file too (the launcher's hi-res slice)
    printf '1 1 1\n2 3 4\n5 5 5\n' > "${TMP}/h_pts.txt"
    rc_sp=0; SAMPLE_POINTS="${TMP}/h_pts.txt" PTS_VEL_GRAD=1 scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_pts.log" 2>&1 || rc_sp=$?
    check "$([ "${rc_sp}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh with SAMPLE_POINTS succeeds (exit ${rc_sp})"
    check "$(has "${TMP}/h_pts.log" "RUNNING: .*--sample-points ${TMP}/h_pts.txt +--pts-vel-grad")" \
        "... and hands the points to the standard binary (with the velocity gradient)"
    check "$([ -s "${TMP}/HASIC/snapdir_099/output.pts_den" ] && echo 0 || echo 1)" "... which writes the point values (.pts_den)"
    # PS_LINEAR_DEPOSIT reaches the phase-space binary (a dry run: the command only)
    DRY_RUN=1 PS_LINEAR_DEPOSIT=1 PS_VOLUME_WEIGHTED=0 scripts/run_ps_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_lin.log" 2>&1 || true
    check "$(has "${TMP}/h_lin.log" '--ps-linear-deposit')" "run_ps_dtfe.sh: PS_LINEAR_DEPOSIT=1 passes --ps-linear-deposit"
    # zero-padded snapshot numbers are decimal (bash's printf reads '099' as octal: an error, then snapshot 000)
    rc_oct=0; scripts/run_dtfe.sh -s HASIC -g 16 099 >"${TMP}/h_oct.log" 2>&1 || rc_oct=$?
    check "$([ "${rc_oct}" = "0" ] && echo 0 || echo 1)" "run_dtfe.sh accepts a zero-padded snapshot number (099, exit ${rc_oct})"
    check "$(has "${TMP}/h_oct.log" 'Snapshot 099 processed successfully')" "... and runs snapshot 99, not 0 (octal)"
    rc_bad=0; scripts/run_dtfe.sh -s HASIC -g 16 9x >"${TMP}/h_bad.log" 2>&1 || rc_bad=$?
    check "$([ "${rc_bad}" = "2" ] && echo 0 || echo 1)" "a non-numeric snapshot argument is refused with exit 2 (${rc_bad})"
    rc_oct=0; scripts/run_ps_dtfe.sh -s HASIC -g 16 -n 1 099 >"${TMP}/h_oct_ps.log" 2>&1 || rc_oct=$?
    check "$(has "${TMP}/h_oct_ps.log" 'Processing snapshot 099')" "run_ps_dtfe.sh reads 099 as snapshot 99 too (exit ${rc_oct})"
    rc_rs=0; "${DTFE_BIN}" "${TMP}/HASIC/snapdir_099/combined_099.hdf5" "${TMP}/rs_out" --grid 16 --redshiftSpace 0 0 1 --input-format 105 >"${TMP}/h_rs.log" 2>&1 || rc_rs=$?
    check "$([ "${rc_rs}" != "0" ] && echo 0 || echo 1)" "--redshiftSpace is refused (not a silent no-op), exit ${rc_rs}"
    check "$(has "${TMP}/h_rs.log" 'not implemented in this version')" "... with a message saying the transform is not implemented"
    check "$(has "${TMP}/h_env.log" 'RUNNING: .*--exact( |$)')" "DTFE_EXACT_AVERAGE=1 does the same"
    # the exact average is the default (the production standard); DTFE_EXACT_AVERAGE=0 is the sampled one
    rc_pl=0; env -u DTFE_EXACT -u DTFE_EXACT_AVERAGE -u SUBSAMPLES -u AVG_SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID scripts/run_dtfe.sh -s HASIC -g 16 99 \
        >"${TMP}/h_plain.log" 2>&1 || rc_pl=$?
    check "$([ "${rc_pl}" = "0" ] && [ "$(has "${TMP}/h_plain.log" 'RUNNING: .*--exact( |$)')" = 0 ] \
             && [ "$(has "${TMP}/h_plain.log" 'RUNNING: .*--field density_a velocity_a divergence_a shear_a ')" = 0 ] \
             && [ "$(has "${TMP}/h_plain.log" 'volume averaging +: EXACT cell-tetrahedron integration')" = 0 ] \
             && echo 0 || echo 1)" "a plain run passes --exact and the four analysis fields (the production defaults; exit ${rc_pl})"
    env -u DTFE_EXACT -u SUBSAMPLES -u AVG_SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=0 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_mc.log" 2>&1 || true
    check "$([ "$(lacks "${TMP}/h_mc.log" 'RUNNING: .*--exact( |$)')" = 0 ] && [ "$(has "${TMP}/h_mc.log" 'RUNNING: .*--sampled( |$)')" = 0 ] \
             && [ "$(has "${TMP}/h_mc.log" 'volume averaging +: SAMPLED \(--sampled\)')" = 0 ] && echo 0 || echo 1)" \
          "DTFE_EXACT_AVERAGE=0 passes --sampled, not --exact: the Monte-Carlo cell mean, said in the run log"
    # SUBSAMPLES (-n) reaches the Monte-Carlo mean as --subsamples, only when set
    env -u DTFE_EXACT -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT_AVERAGE=0 DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 -n 2 99 >"${TMP}/h_mc_n.log" 2>&1 || true
    check "$([ "$(has "${TMP}/h_mc_n.log" '--lambda-th +0\.3 .*--sampled +--subsamples +2$')" = 0 ] && [ "$(lacks "${TMP}/h_mc.log" '--subsamples')" = 0 ] \
             && echo 0 || echo 1)" "... -n 2 (SUBSAMPLES) passes --subsamples 2; unset, nothing (the binary's own count)"
    # P6.7: DTFE_EXACT, the knob's older name, is read when the new one is unset (one warning); the new one wins over it
    env -u DTFE_EXACT_AVERAGE -u SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_EXACT=0 DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_mc_old.log" 2>&1 || true
    DTFE_EXACT=0 DTFE_EXACT_AVERAGE=1 DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_mc_both.log" 2>&1 || true
    check "$([ "$(lacks "${TMP}/h_mc_old.log" '--exact( |$)')" = 0 ] && [ "$(has "${TMP}/h_mc_old.log" '--lambda-th +0\.3 .*--sampled$')" = 0 ] \
             && [ "$(grep -c '^warning: DTFE_EXACT is deprecated, use DTFE_EXACT_AVERAGE$' "${TMP}/h_mc_old.log")" = 1 ] \
             && [ "$(has "${TMP}/h_mc_both.log" '--lambda-th +0\.3 .*--exact( |$)')" = 0 ] \
             && [ "$(has "${TMP}/h_mc_both.log" '^warning: DTFE_EXACT is deprecated, use DTFE_EXACT_AVERAGE \(DTFE_EXACT=0 is not read: DTFE_EXACT_AVERAGE=1 wins\)$')" = 0 ] \
             && echo 0 || echo 1)" \
          "... DTFE_EXACT=0, the older name, does the same while DTFE_EXACT_AVERAGE is unset, with one warning; DTFE_EXACT_AVERAGE=1 wins over it (said)"
    # the rule (config.sh average_resolve, the binary's own): -e / --exact or --sampled first, then DTFE_EXACT_AVERAGE,
    # then a SUBSAMPLES count (the Monte-Carlo mean), then the default (exact); the banner names what decided. Dry
    # runs with the knobs unset first (and the CPU, so nothing stands between --lambda-th and the cell average); the
    # planned command always names the cell average, --subsamples only the sampled one
    hplan() {   # <log> [VAR=value ...] [-- driver options]: echoes run_dtfe.sh's planned command line
        local log="$1" e=()
        shift
        while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do e+=("$1"); shift; done
        [ "$#" -gt 0 ] && shift
        env -u DTFE_EXACT -u DTFE_EXACT_AVERAGE -u SUBSAMPLES -u AVG_SUBSAMPLES -u FACE_GRADIENT -u DTFE_METAL -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_GPU=0 \
            ${e[@]+"${e[@]}"} DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 "$@" 99 >"${log}" 2>&1 || true
        sed -n '/would run:/{n;p;}' "${log}"
    }
    h_def="$(hplan "${TMP}/h_def.log")"; h_sub="$(hplan "${TMP}/h_sub.log" SUBSAMPLES=2)"
    h_n="$(hplan "${TMP}/h_n.log" -- -n 2)"; h_off="$(hplan "${TMP}/h_off.log" DTFE_EXACT_AVERAGE=0)"
    check "$(printf '%s\n' "${h_def}" | grep -qE -e '--lambda-th +0\.3 +--exact$' \
             && grep -aqxF 'Cell averages: exact (--exact; the default)' "${TMP}/h_def.log" \
             && printf '%s\n' "${h_sub}" | grep -qE -e '--lambda-th +0\.3 +--sampled +--subsamples +2$' && [ "${h_n}" = "${h_sub}" ] \
             && grep -aqxF 'Cell averages: sampled, Monte-Carlo sample means (--sampled --subsamples 2; SUBSAMPLES=2)' "${TMP}/h_sub.log" \
             && printf '%s\n' "${h_off}" | grep -qE -e '--lambda-th +0\.3 +--sampled$' \
             && grep -aqxF 'Cell averages: sampled, Monte-Carlo sample means (--sampled; DTFE_EXACT_AVERAGE=0)' "${TMP}/h_off.log" \
             && echo 0 || echo 1)" \
          "the default passes --exact; SUBSAMPLES=2 or -n 2 alone --sampled --subsamples 2; DTFE_EXACT_AVERAGE=0 --sampled (the banner says why)"
    h_esub="$(hplan "${TMP}/h_esub.log" SUBSAMPLES=2 -- -e)"; h_kesub="$(hplan "${TMP}/h_kesub.log" DTFE_EXACT_AVERAGE=1 -- -n 2)"
    check "$([ "${h_esub}" = "${h_def}" ] && [ "${h_kesub}" = "${h_def}" ] \
             && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: -e asks for the exact cell average' "${TMP}/h_esub.log")" = 1 ] \
             && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: DTFE_EXACT_AVERAGE=1 asks for the exact cell average' "${TMP}/h_kesub.log")" = 1 ] \
             && echo 0 || echo 1)" "-e beside SUBSAMPLES=2, and DTFE_EXACT_AVERAGE=1 beside -n 2: --exact without --subsamples, one note each"
    h_lex="$(hplan "${TMP}/h_lex.log" -- --exact)"; h_lsa="$(hplan "${TMP}/h_lsa.log" DTFE_EXACT_AVERAGE=1 -- --sampled)"
    h_lea="$(hplan "${TMP}/h_lea.log" DTFE_EXACT_AVERAGE=0 -- --exact-average)"; h_le0="$(hplan "${TMP}/h_le0.log" DTFE_EXACT_AVERAGE=0 -- -e)"
    check "$([ -n "${h_def}" ] && [ "${h_lex}" = "${h_def}" ] && [ "${h_lsa}" = "${h_off}" ] && [ "${h_lea}" = "${h_def}" ] && [ "${h_le0}" = "${h_def}" ] \
             && [ "$(grep -c '^warning: --exact-average is deprecated, use --exact$' "${TMP}/h_lea.log")" = 1 ] \
             && [ "$(lacks "${TMP}/h_le0.log" 'deprecated')" = 0 ] && [ "$(lacks "${TMP}/h_lex.log" 'deprecated')" = 0 ] && echo 0 || echo 1)" \
          "--exact is -e; --sampled beats DTFE_EXACT_AVERAGE=1 and -e DTFE_EXACT_AVERAGE=0; --exact-average, the older long form, is -e with one warning"
    rc=0; env -u DTFE_EXACT -u DTFE_EXACT_AVERAGE -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 -e --sampled 99 >"${TMP}/h_both.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "1" ] && [ "$(has "${TMP}/h_both.log" "^Error: '-e' \('--exact'\) and '--sampled' exclude each other")" = 0 ] \
             && [ "$(has "${TMP}/h_both.log" '^Usage: ')" = 0 ] && [ "$(lacks "${TMP}/h_both.log" 'would run:')" = 0 ] && echo 0 || echo 1)" \
          "-e and --sampled together are refused with the usage, exit 1, nothing planned (${rc})"
    # the standard binary's auto-tuner reports (not 'AUTO-TUNE-REPORT unsupported'), and the driver
    # asks it the way run_ps_dtfe.sh does: a report per snapshot, nothing computed, nothing written beside the data
    rm -rf "${TMP}/HASIC/snapdir_099/output."*        # (the grids, run logs and the superseded folder of the runs above)
    rc_rep=0; AUTO_TUNE_REPORT=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/h_report.log" 2>&1 || rc_rep=$?
    check "$([ "${rc_rep}" = "0" ] && [ "$(grep -acE '^  AUTO-TUNE-REPORT snap=099 partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=[0-9.]+ .*over_budget=[01]$' "${TMP}/h_report.log")" = 1 ] \
             && echo 0 || echo 1)" "run_dtfe.sh AUTO_TUNE_REPORT=1: one valid standard report line for snapshot 099 (exit ${rc_rep})"
    check "$([ -z "$(ls "${TMP}/HASIC/snapdir_099/" | grep '^output\.')" ] && [ "$(lacks "${TMP}/h_report.log" 'unsupported')" = 0 ] \
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
rc=0; merge_dl 99 >"${TMP}/i_merge_ok.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "merging 4 complete chunks exits 0 (${rc})"
check "$([ -f "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "... the combined file is written under its final name; a stale .partial does not block it and is gone"
check "$(has "${TMP}/i_merge_ok.log" 'Particles merged: \[2048\]$')" "... and the totals verify against the header"
"${PY}" - "${DL}/snapdir_099/combined_099.hdf5" >"${TMP}/i_content.log" 2>&1 <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    a = f["Header"].attrs
    ok = (list(a["NumPart_Total"]) == [0, 2048, 0, 0, 0, 0] and not any(a["NumPart_Total_HighWord"])
          and np.array_equal(f["PartType1/ParticleIDs"][...], np.arange(2048))
          and abs(float(a["BoxSize"]) - 75000.0 / 0.6774) < 1e-6 and "HFreeUnits" in a)
print("CONTENT OK" if ok else f"CONTENT BAD {dict(a)}")
PYEOF
check "$(has "${TMP}/i_content.log" 'CONTENT OK')" "... its header holds int64 totals, a zero high word, the h-free box; the IDs are 0..2047 in chunk order"
rm -f "${DL}/snapdir_099/combined_099.hdf5"
mv "${DL}/snapdir_099/snap_099.2.hdf5" "${TMP}/snap_099.2.keep"
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"            # left by a stopped merge: a refusal removes it too
rc=0; merge_dl 99 >"${TMP}/i_merge_missing.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a chunk missing from disk (3 of the header's 4): exit 1 (${rc})"
check "$([ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "... no combined file at all (not the 3 merged and called done)"
check "$(has "${TMP}/i_merge_missing.log" "Problem: Can't find 1 subfile\(s\) \(the header promises 4\)")" "... the header's count is the yardstick"
check "$(has "${TMP}/i_merge_missing.log" 'Missing: .*snap_099\.2\.hdf5')" "... and the missing chunk is named"
check "$(has "${TMP}/i_merge_missing.log" 'Done, with problems: no combined file for snapshot 099')" "... the summary line says what has none"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL}/snapdir_099/snap_099.2.hdf5"      # a download stopped half-way
echo "stale" > "${DL}/snapdir_099/combined_099.hdf5.partial"
rc=0; merge_dl 99 >"${TMP}/i_merge_trunc.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a truncated chunk: exit 1 and no combined file (not merged past an 'Error ... continue') (${rc})"
check "$(has "${TMP}/i_merge_trunc.log" "Error: couldn't read snap_099\.2\.hdf5 \(an incomplete download\)")" "... the chunk is named as incomplete"
check "$(has "${TMP}/i_merge_trunc.log" 'Run the download again: it removes an incomplete chunk')" "... with the way out"
head -c 30 "${TMP}/snap_099.2.keep" > "${DL}/snapdir_099/snap_099.2.hdf5"         # stopped inside the signature
rc=0; merge_dl 99 >"${TMP}/i_merge_tiny.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a 30-byte chunk (HDF5 says 'bad byte number' or no signature): exit 1 (${rc})"
check "$(has "${TMP}/i_merge_tiny.log" "Error: couldn't read snap_099\.2\.hdf5 \(an incomplete download\)")" "... classed as incomplete by its size"
rm -f "${DL}/snapdir_099/snap_099.2.hdf5"
cp "${TMP}/snap_099.2.keep" "${DL}/snapdir_099/snap_099.2.hdf5"
rc=0; merge_dl 99 >"${TMP}/i_merge_again.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -f "${DL}/snapdir_099/combined_099.hdf5" ] && echo 0 || echo 1)" "the chunk restored: the merge succeeds again (${rc})"
# the totals gate: -n 3 merges three of the four chunks -> MISMATCH, no file, --delete-chunks deletes nothing
rm -f "${DL}/snapdir_099/combined_099.hdf5"
rc=0; merge_dl -n 3 --delete-chunks 99 >"${TMP}/i_merge_n3.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "-n 3 against a header of 4: MISMATCH -> exit 1, no combined file, no .partial (${rc})"
check "$(has "${TMP}/i_merge_n3.log" 'MISMATCH vs header total \[2048\]')" "... the MISMATCH line names the header's total"
check "$(has "${TMP}/i_merge_n3.log" 'WARNING: -n 3 given, but the header says 4')" "... and -n disagreeing with the header is said"
check "$([ "$(ls "${DL}"/snapdir_099/snap_099.*.hdf5 | wc -l | tr -d ' ')" = "4" ] && echo 0 || echo 1)" "... --delete-chunks deleted nothing after the refusal"
# the high word of the totals counts (the -1 runs exceed 2^32)
"${PY}" - "${DL}/snapdir_099/snap_099.0.hdf5" <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r+") as f:
    f["Header"].attrs["NumPart_Total_HighWord"] = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint32)
PYEOF
rc=0; merge_dl 99 >"${TMP}/i_merge_hw.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && echo 0 || echo 1)" "a high word in the header's total is part of the expected count: MISMATCH (${rc})"
check "$(has "${TMP}/i_merge_hw.log" 'MISMATCH vs header total \[4294969344\]')" "... 2^32 + 2048 named as the total"
cp "${TMP}/snap_099.0.keep" "${DL}/snapdir_099/snap_099.0.hdf5"
# a merge with --delete-chunks, then the same command again: done, not a problem
DLC="${TMP}/DLC/SIMX"; mkdir -p "${DLC}"; cp -R "${DL}/snapdir_099" "${DLC}/snapdir_099"
rc=0; "${PY}" python/tools/merge_HDF5.py -d "${DLC}" --delete-chunks 99 >"${TMP}/i_merge_del.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -f "${DLC}/snapdir_099/combined_099.hdf5" ] && [ -z "$(ls "${DLC}"/snapdir_099/snap_099.*.hdf5 2>/dev/null)" ] && echo 0 || echo 1)" \
      "--delete-chunks after a verified merge: exit 0, the combined file, no chunk left (${rc})"
rc=0; "${PY}" python/tools/merge_HDF5.py -d "${DLC}" --delete-chunks 99 >"${TMP}/i_merge_del2.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "the same merge again (no chunks, the combined file there): exit 0, nothing to do (${rc})"
check "$(has "${TMP}/i_merge_del2.log" 'already exists and no chunks are left')" "... and says so"
check "$(lacks "${TMP}/i_merge_del2.log" 'Done, with problems')" "... with no problem reported"
# the disk failing while the .partial is written: no file, the error named (ulimit -f caps the file size)
rm -f "${DL}/snapdir_099/combined_099.hdf5"
rc=0; ( ulimit -f 40; PYTHONUNBUFFERED=1 merge_dl 99 ) >"${TMP}/i_merge_disk.log" 2>&1 || rc=$?
check "$([ "${rc}" != "0" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5" ] && [ ! -e "${DL}/snapdir_099/combined_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a write failure (file size limit): non-zero exit, no combined file, no .partial (${rc})"
check "$(has "${TMP}/i_merge_disk.log" 'could not write .*combined_099\.hdf5\.partial')" "... reported as the disk's, not a chunk's"
# group catalogues: the header's NumFiles, an empty chunk without Group/Subhalo groups, a missing chunk
rc=0; merge_dl --groupcats 99 >"${TMP}/i_gc_ok.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -f "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5" ] && [ ! -e "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5.partial" ] && echo 0 || echo 1)" \
      "a group catalogue of 4 chunks, one with no groups at all, merges: exit 0, no .partial (${rc})"
check "$(has "${TMP}/i_gc_ok.log" 'Groups: 12/12  Subhalos: 12/12  \(verified\)')" "... its totals verify"
rm -f "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5"
mv "${DL}/groups_099/fof_subhalo_tab_099.1.hdf5" "${TMP}/gc1.keep"
rc=0; merge_dl --groupcats 99 >"${TMP}/i_gc_missing.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && [ ! -e "${DL}/groups_099/combined_fof_subhalo_tab_099.hdf5" ] && echo 0 || echo 1)" "a group-catalogue chunk missing: exit 1, no combined file (${rc})"
check "$(has "${TMP}/i_gc_missing.log" "Problem: Can't find 1 subfile\(s\) \(the header promises 4\)")" "... the header's NumFiles is the yardstick"
check "$(has "${TMP}/i_gc_missing.log" 'Done, with problems: no combined file for group catalogue 099')" "... named in the summary"
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
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "download_snapshots.sh exits 1 when a fetch failed, not 0 (${rc})"
check "$(has "${TMP}/i_download.log" 'Error downloading snapshot-98')" "... the failed endpoint is reported"
check "$(has "${TMP}/i_download.log" 'Download finished with 2 error')" "... and the last line counts it and the kept unreadable chunk (no 'Download complete!')"
check "$(has "${TMP}/i_download.log" 'Error downloading snapshot-99: an unreadable chunk is still in place')" "... a kept unreadable chunk is an error of this run (wget -nc would skip it for ever)"
check "$(has "${TMP}/i_urls.log" '^ARGS: .*--https-only')" "the fetches are https-only (redirects with the key header too)"
check "$(has "${TMP}/i_urls.log" '^https://www.tng-project.org/api/SIMX/files/snapshot-99\.0\.hdf5$')" "the listing's http:// chunk link is fetched, upgraded to https"
check "$(lacks "${TMP}/i_urls.log" '^http://')" "... and no plain-http URL is ever requested"
check "$(has "${TMP}/i_urls.log" '^ARGS: .*--header=API-Key: dummykey')" "... and the key travels as a header"
check "$(lacks "${TMP}/i_download.log" 'Download complete!')" "... really no 'Download complete!'"
check "$([ "$(grep -c '^https://www.tng-project.org/api/SIMX/files/snapshot-9[89]/' "${TMP}/i_urls.log")" = "2" ] && echo 0 || echo 1)" \
      "both fetches went to https (the key is a header, never sent over plain http)"
check "$(lacks "${TMP}/i_download.log" 'dummykey')" "the key is not in the output"
check "$([ ! -e "${DL2}/snapdir_099/snap_099.1.hdf5" ] && echo 0 || echo 1)" "an old truncated chunk is removed before wget runs (wget -nc would skip it for ever)"
check "$(has "${TMP}/i_download.log" 'Removed incomplete chunk snap_099\.1\.hdf5')" "... and the log says which"
check "$([ -f "${DL2}/snapdir_099/snap_099.0.hdf5" ] && echo 0 || echo 1)" "a complete chunk stays"
check "$([ -f "${DL2}/snapdir_099/snap_099.2.hdf5" ] && echo 0 || echo 1)" "a truncated chunk written less than a minute ago stays (another download may be writing it)"
check "$(has "${TMP}/i_download.log" 'snap_099\.2\.hdf5 is incomplete but was written less than a minute ago')" "... and is noted"
touch -t 202001010000 "${DL2}/snapdir_099/snap_099.2.hdf5"                                  # a minute later...
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls2.log" TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 99 >"${TMP}/i_download_ok.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a run with no failure exits 0 (${rc})"
check "$(has "${TMP}/i_download_ok.log" 'Removed incomplete chunk snap_099\.2\.hdf5')" "... the chunk that aged past the guard is removed now"
check "$(has "${TMP}/i_download_ok.log" 'Download complete!')" "... and ends with 'Download complete!'"
# the other three products, with an aged truncated file each: pruned, fetched over https with the key as a header
DL3="${TMP}/DL3/SIMX"; mkdir -p "${DL3}/Merger Trees" "${DL3}/groups_099"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/ics.hdf5";                      touch -t 202001010000 "${DL3}/ics.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/Merger Trees/tree_extended.0.hdf5"; touch -t 202001010000 "${DL3}/Merger Trees/tree_extended.0.hdf5"
head -c 20000 "${TMP}/snap_099.2.keep" > "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5"; touch -t 202001010000 "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5"
rc_i=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_i.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -i -d "${DL3}" -s SIMX-Dark >"${TMP}/i_dl_ics.log" 2>&1 || rc_i=$?
rc_t=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_t.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -t -d "${DL3}" -s SIMX >"${TMP}/i_dl_trees.log" 2>&1 || rc_t=$?
rc_c=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls_c.log" TNG_API_KEY=dummykey scripts/download_snapshots.sh -c -d "${DL3}" -s SIMX 99 >"${TMP}/i_dl_gc.log" 2>&1 || rc_c=$?
check "$([ "${rc_i}" = "0" ] && [ "${rc_t}" = "0" ] && [ "${rc_c}" = "0" ] && echo 0 || echo 1)" "-i, -t and -c runs with a working fetch exit 0 (${rc_i} ${rc_t} ${rc_c})"
check "$(has "${TMP}/i_urls_i.log" '^https://www.tng-project.org/api/SIMX/files/ics.hdf5$')" "-i: https, and '-Dark' stripped for the API name"
check "$(has "${TMP}/i_urls_t.log" '^https://www.tng-project.org/api/SIMX/files/sublink/')" "-t: https"
check "$(has "${TMP}/i_urls_c.log" '^https://www.tng-project.org/api/SIMX/files/groupcat-99/')" "-c: https"
check "$([ "$(cat "${TMP}"/i_urls_i.log "${TMP}"/i_urls_t.log "${TMP}"/i_urls_c.log | grep -c 'header=API-Key: dummykey')" = "5" ] && echo 0 || echo 1)" "the key is a header in all five wget calls (ics; listing + chunks for trees and catalogues)"
check "$([ -f "${DL3}/Merger Trees/tree_extended.0.hdf5" ] && [ -f "${DL3}/groups_099/fof_subhalo_tab_099.0.hdf5" ] && echo 0 || echo 1)" "-t and -c fetch the chunks their listings name"
check "$(lacks "${TMP}/i_urls_i.log" '^https.*dummykey')" "... never in a URL"
check "$(has "${TMP}/i_dl_ics.log" 'Removed incomplete chunk ics\.hdf5')" "-i prunes an aged truncated ics.hdf5 first"
check "$(has "${TMP}/i_dl_trees.log" 'Removed incomplete chunk tree_extended\.0\.hdf5')" "-t prunes an aged truncated tree chunk first"
check "$(has "${TMP}/i_dl_gc.log" 'Removed incomplete chunk fof_subhalo_tab_099\.0\.hdf5')" "-c prunes an aged truncated catalogue chunk first"
# a fetch that exits 0 but fetched nothing is an error (the listing named no chunk)
DL4="${TMP}/DL4/SIMX"; mkdir -p "${DL4}"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls4.log" FAKE_WGET_EMPTY=1 TNG_API_KEY=dummykey \
    scripts/download_snapshots.sh -d "${DL4}" -s SIMX 99 >"${TMP}/i_dl_empty.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "wget exiting 0 with no chunk fetched: exit 1 (${rc})"
check "$(has "${TMP}/i_dl_empty.log" 'Error downloading snapshot-99: no chunk was fetched')" "... and named"
# the chunk list is tried up to three times (fetch_listing): a chunk the data host refuses once (a 403 a rerun does
# not get) costs a retry, not the snapshot; a chunk refused every time is still an error
DL5="${TMP}/DL5/SIMX"; mkdir -p "${DL5}"; rm -f "${TMP}/i_retry.count"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls5.log" FAKE_WGET_COUNT="${TMP}/i_retry.count" FAKE_WGET_CHUNK_FAIL=1 \
    TNG_API_KEY=dummykey scripts/download_snapshots.sh -d "${DL5}" -s SIMX 99 >"${TMP}/i_dl_retry.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "a chunk refused once, then served: exit 0 (${rc})"
check "$(has "${TMP}/i_dl_retry.log" 'a chunk failed; trying the listing again \(attempt 2 of 3\)$')" "... after one retry, said as 'attempt 2 of 3'"
check "$([ "$(cat "${TMP}/i_retry.count" 2>/dev/null)" = "2" ] && [ "$(lacks "${TMP}/i_dl_retry.log" 'attempt 3 of 3')" = 0 ] && echo 0 || echo 1)" \
      "... the chunk list went to wget twice, not a third time ($(cat "${TMP}/i_retry.count" 2>/dev/null || echo none))"
check "$([ -f "${DL5}/snapdir_099/snap_099.0.hdf5" ] && echo 0 || echo 1)" "... the chunk is in place"
check "$([ "$(has "${TMP}/i_dl_retry.log" '^  snapshot-99 downloaded successfully$')" = 0 ] && [ "$(has "${TMP}/i_dl_retry.log" '^Download complete!$')" = 0 ] \
         && [ "$(lacks "${TMP}/i_dl_retry.log" 'Error downloading')" = 0 ] && echo 0 || echo 1)" "... and the snapshot counts as downloaded: 'Download complete!', no error"
DL6="${TMP}/DL6/SIMX"; mkdir -p "${DL6}"; rm -f "${TMP}/i_refused.count"
rc=0; PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls6.log" FAKE_WGET_COUNT="${TMP}/i_refused.count" FAKE_WGET_CHUNK_FAIL=99 \
    TNG_API_KEY=dummykey scripts/download_snapshots.sh -d "${DL6}" -s SIMX 99 >"${TMP}/i_dl_refused.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "a chunk refused every time: exit 1, not 0 (${rc})"
check "$([ "$(cat "${TMP}/i_refused.count" 2>/dev/null)" = "3" ] && [ "$(has "${TMP}/i_dl_refused.log" '\(attempt 3 of 3\)$')" = 0 ] && echo 0 || echo 1)" \
      "... after three tries, no fourth ($(cat "${TMP}/i_refused.count" 2>/dev/null || echo none))"
check "$([ "$(has "${TMP}/i_dl_refused.log" '^  Error downloading snapshot-99$')" = 0 ] && [ "$(has "${TMP}/i_dl_refused.log" 'Download finished with 1 error')" = 0 ] \
         && [ "$(lacks "${TMP}/i_dl_refused.log" 'Download complete!')" = 0 ] && echo 0 || echo 1)" "... named as a failed snapshot and counted in the last line"
check "$([ ! -e "${DL6}/snapdir_099/snap_099.0.hdf5" ] && echo 0 || echo 1)" "... and no chunk was left in place"
check "$(lacks "${TMP}/i_dl_refused.log" 'dummykey')" "... the key in none of the retries' output"
# (PY, PYTHON's older name, with the suite's PYTHON unset: the alias is read, with its warning)
rc=0; env -u PYTHON PATH="${TMP}/fakebin:${PATH}" FAKE_WGET_LOG="${TMP}/i_urls3.log" TNG_API_KEY=dummykey PY=/nonexistent/python3 \
    scripts/download_snapshots.sh -d "${DL2}" -s SIMX 99 >"${TMP}/i_download_nopy.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "without a python with h5py the check is skipped, the download still runs (${rc})"
check "$(has "${TMP}/i_download_nopy.log" 'has no h5py; existing chunks are not checked')" "... and says so"
check "$(has "${TMP}/i_download_nopy.log" '^warning: PY is deprecated, use PYTHON$')" "... (PY=, the older name of PYTHON, read with one warning)"

echo ""
echo "(J) --help runs nothing, --version prints the revision, DRY_RUN=1 prints the commands and runs nothing."
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/run_ps_pipeline.sh; do
    rc=0; "${scr}" --help >"${TMP}/j_help_${scr##*/}.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "${scr##*/} --help exits 0 (${rc})"
    check "$(lacks "${TMP}/j_help_${scr##*/}.log" 'Processing snapshot|RUNNING:|Slice: axis')" "... and starts nothing"
done
check "$(has "${TMP}/j_help_run_ps_pipeline.sh.log" 'COMPUTE ONLY')" "run_ps_pipeline.sh --help prints its header (and does not start the whole batch)"
check "$(has "${TMP}/j_help_run_dtfe.sh.log" '^Usage: ')" "run_dtfe.sh --help ends with the usage line"
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_SIM "${scr}" 99 >"${TMP}/j_nosim_${scr##*/}.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_nosim_${scr##*/}.log" 'no simulation given: pass -s SIMULATION')" = 0 ] && echo 0 || echo 1)" \
          "${scr##*/} without -s or DTFE_SIM refuses with exit 2: there is no default simulation (${rc})"
done
# no default data root either (config.sh require_data_root): every driver refuses before it reads or writes anything
NOROOT='no data root: set DTFE_DATA_ROOT to the folder that holds the simulations \(e\.g\. DTFE_DATA_ROOT=~/IllustrisTNG\)\.'
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_DATA_ROOT TNG_API_KEY=dummykey "${scr}" -s HASIC 99 >"${TMP}/j_noroot_${scr##*/}.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_noroot_${scr##*/}.log" "^Error: ${NOROOT}$")" = 0 ] \
             && [ "$(lacks "${TMP}/j_noroot_${scr##*/}.log" 'Processing snapshot|RUNNING:|Downloading|Data directory')" = 0 ] && echo 0 || echo 1)" \
          "${scr##*/} without DTFE_DATA_ROOT refuses with exit 2 and says so, before anything runs: there is no default data root (${rc})"
done
# -d names the simulation's folder, so it stands in for the data root: without a simulation the next refusal is the
# missing simulation, not the data root (nothing runs or downloads)
rc_d=""
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; env -u DTFE_DATA_ROOT -u DTFE_SIM TNG_API_KEY=dummykey "${scr}" -d "${TMP}/HASIC" 99 >"${TMP}/j_noroot_d.log" 2>&1 || rc=$?
    [ "$(has "${TMP}/j_noroot_d.log" 'no simulation given')" = 0 ] && [ "$(lacks "${TMP}/j_noroot_d.log" "${NOROOT}")" = 0 ] || rc="${rc}-msg"
    rc_d="${rc_d} ${rc}"
done
check "$([ "${rc_d}" = " 2 2 2" ] && echo 0 || echo 1)" "... -d DATA_DIR stands in for it: the refusal is then the missing simulation (${rc_d# })"
# the option loop the three share (config.sh driver_options): an unknown option or one missing its value prints
# getopts' message and the usage and exits 1, -h after other options prints the usage and exits 0 (nothing runs)
rc_o=""
for scr in scripts/run_dtfe.sh scripts/run_ps_dtfe.sh scripts/download_snapshots.sh; do
    rc=0; TNG_API_KEY=dummykey "${scr}" -s HASIC -Z 99 >"${TMP}/j_opt1.log" 2>&1 || rc=$?
    [ "$(has "${TMP}/j_opt1.log" 'option -- .?Z')" = 0 ] && [ "$(has "${TMP}/j_opt1.log" '^Usage: ')" = 0 ] || rc="${rc}-msg"
    rc2=0; TNG_API_KEY=dummykey "${scr}" -s >"${TMP}/j_opt2.log" 2>&1 || rc2=$?
    [ "$(has "${TMP}/j_opt2.log" 'requires an argument -- .?s')" = 0 ] && [ "$(has "${TMP}/j_opt2.log" '^Usage: ')" = 0 ] || rc2="${rc2}-msg"
    rc3=0; TNG_API_KEY=dummykey "${scr}" -s HASIC -h 99 >"${TMP}/j_opt3.log" 2>&1 || rc3=$?
    [ "$(has "${TMP}/j_opt3.log" '^Usage: ')" = 0 ] && [ "$(lacks "${TMP}/j_opt3.log" 'Processing snapshot|Downloading|Data directory')" = 0 ] || rc3="${rc3}-msg"
    rc_o="${rc_o} ${rc}/${rc2}/${rc3}"
done
check "$([ "${rc_o}" = " 1/1/0 1/1/0 1/1/0" ] && echo 0 || echo 1)" \
      "an unknown option or a missing value: the usage, exit 1; -h after other options: the usage, exit 0 (${rc_o# })"
# a driver's own options mix with the shared ones in any order, clustered too (-me)
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -g 16 -n 2 -d "${TMP}/HASIC" -me -s HASIC 99 >"${TMP}/j_opt4.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(has "${TMP}/j_opt4.log" 'HASIC/snapdir_099/combined_099\.hdf5 .*--grid +16 .*--gpu +--exact( |$)')" = 0 ] \
         && [ "$(lacks "${TMP}/j_opt4.log" '^ +\./PS-DTFE.*--subsamples')" = 0 ] \
         && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: -e asks for the exact deposit' "${TMP}/j_opt4.log")" = 1 ] \
         && echo 0 || echo 1)" "... -g, -n, -m and -e mix with -d and -s in any order, -me clustered: -e keeps the exact deposit, -n 2 said ignored (exit ${rc})"
# the long forms are the letters (config.sh driver_long_options): the same planned command, token for token
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" --grid 16 --subsamples=2 --sim-dir "${TMP}/HASIC" --gpu --exact --sim HASIC 99 \
    >"${TMP}/j_opt5.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt5.log")" = "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt4.log")" ] \
         && [ -n "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt5.log")" ] && [ "$(lacks "${TMP}/j_opt5.log" 'deprecated')" = 0 ] && echo 0 || echo 1)" \
      "... --grid, --subsamples=K, --sim-dir, --gpu, --exact and --sim plan the same command as the letters (exit ${rc})"
# --exact-average, the older long form of -e, for one release: the same plan, one warning (the binaries' wording)
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" --grid 16 --subsamples=2 --sim-dir "${TMP}/HASIC" --gpu --exact-average --sim HASIC 99 \
    >"${TMP}/j_opt5o.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt5o.log")" = "$(sed -n '/would run:/{n;p;}' "${TMP}/j_opt4.log")" ] \
         && [ "$(grep -c '^warning: --exact-average is deprecated, use --exact$' "${TMP}/j_opt5o.log")" = 1 ] \
         && [ "$(grep -c 'deprecated' "${TMP}/j_opt5o.log")" = 1 ] && echo 0 || echo 1)" \
      "... --exact-average (the older --exact) plans the same, with one 'warning: --exact-average is deprecated, use --exact' (exit ${rc})"
# --sampled: the sampled deposit, named; with a count, the count too (in place of -e's --exact)
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" --grid 16 --subsamples=2 --sim-dir "${TMP}/HASIC" --gpu --sampled --sim HASIC 99 \
    >"${TMP}/j_opt5s.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ "$(has "${TMP}/j_opt5s.log" 'combined_099\.hdf5 .*--grid +16 .*--gpu +--sampled +--subsamples +2 ')" = 0 ] \
         && [ "$(lacks "${TMP}/j_opt5s.log" '^ +\./PS-DTFE.*--exact( |$)')" = 0 ] && echo 0 || echo 1)" \
      "... --sampled with --subsamples=2: --sampled --subsamples 2, no --exact (exit ${rc})"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 scripts/run_dtfe.sh --sim HASIC --grid=16 --subsamples 2 99 >"${TMP}/j_opt6.log" 2>&1 || rc=$?
rc2=0; "${DRIVER}" -s HASIC --bogus 99 >"${TMP}/j_opt7.log" 2>&1 || rc2=$?
rc3=0; TNG_API_KEY=dummykey scripts/download_snapshots.sh -s HASIC --gpu 99 >"${TMP}/j_opt8.log" 2>&1 || rc3=$?
rc4=0; TNG_API_KEY=dummykey scripts/download_snapshots.sh -s HASIC --sampled 99 >"${TMP}/j_opt9.log" 2>&1 || rc4=$?
check "$([ "${rc}" = "0" ] && [ "$(has "${TMP}/j_opt6.log" '--grid +16 .*--sampled +--subsamples +2')" = 0 ] \
         && [ "${rc2}" = "1" ] && [ "$(has "${TMP}/j_opt7.log" "unknown option '--bogus'")" = 0 ] && [ "$(has "${TMP}/j_opt7.log" '^Usage: ')" = 0 ] \
         && [ "${rc3}" = "1" ] && [ "$(has "${TMP}/j_opt8.log" "unknown option '--gpu'")" = 0 ] \
         && [ "${rc4}" = "1" ] && [ "$(has "${TMP}/j_opt9.log" "unknown option '--sampled'")" = 0 ] && echo 0 || echo 1)" \
      "... run_dtfe.sh takes them too (a count alone: --sampled --subsamples 2); an unknown long option, or one the script lacks (download_snapshots.sh --gpu, --sampled): the usage, exit 1 (${rc} ${rc2} ${rc3} ${rc4})"
# --run-mode MODE (getopts' R): both drivers plan the mode it names; download_snapshots.sh runs no binary and has none
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s HASIC -g 16 --run-mode gentle 99 >"${TMP}/j_rm1.log" 2>&1 || rc=$?
rc2=0; env "${COMMON_ENV[@]}" DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 --run-mode=performance 99 >"${TMP}/j_rm2.log" 2>&1 || rc2=$?
rc3=0; TNG_API_KEY=dummykey scripts/download_snapshots.sh -s HASIC --run-mode gentle 99 >"${TMP}/j_rm3.log" 2>&1 || rc3=$?
check "$([ "${rc}" = "0" ] && [ "$(has "${TMP}/j_rm1.log" 'combined_099\.hdf5 .*--run-mode +gentle ')" = 0 ] \
         && [ "${rc2}" = "0" ] && [ "$(has "${TMP}/j_rm2.log" 'combined_099\.hdf5 .*--run-mode +performance ')" = 0 ] \
         && [ "${rc3}" = "1" ] && [ "$(has "${TMP}/j_rm3.log" "unknown option '--run-mode'")" = 0 ] && [ "$(has "${TMP}/j_rm3.log" '^Usage: ')" = 0 ] \
         && echo 0 || echo 1)" \
      "... --run-mode MODE (and --run-mode=MODE) plans that mode in both drivers; download_snapshots.sh has no run mode: the usage, exit 1 (${rc} ${rc2} ${rc3})"
rc=0; DTFE_DATA_ROOT= scripts/run_ps_dtfe.sh -s HASIC 99 >"${TMP}/j_noroot_empty.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_noroot_empty.log" "${NOROOT}")" = 0 ] && echo 0 || echo 1)" \
      "... an empty DTFE_DATA_ROOT= is no data root either (${rc})"
rc=0; env -u DTFE_DATA_ROOT DRY_RUN=1 SIMS=all scripts/run_ps_pipeline.sh >"${TMP}/j_noroot_pipe.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/j_noroot_pipe.log" "^Error: ${NOROOT}$")" = 0 ] \
         && [ "$(lacks "${TMP}/j_noroot_pipe.log" '^Simulations:')" = 0 ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh without DTFE_DATA_ROOT refuses with exit 2 before SIMS=all looks for simulations (${rc})"
rc_h=""
for c in "scripts/run_dtfe.sh --help" "scripts/run_ps_dtfe.sh --help" "scripts/run_ps_pipeline.sh --help" "scripts/download_snapshots.sh -h"; do
    rc=0; env -u DTFE_DATA_ROOT ${c} >"${TMP}/j_noroot_help.log" 2>&1 || rc=$?
    rc_h="${rc_h} ${rc}"
done
check "$([ "${rc_h}" = " 0 0 0 0" ] && echo 0 || echo 1)" "... while --help (-h) of all four still works without it (${rc_h# })"
src="$(env -u DTFE_DATA_ROOT bash -c 'source scripts/config.sh; echo "sourced root=[${DATA_ROOT}]"' 2>&1)"
check "$([ "${src}" = "sourced root=[]" ] && echo 0 || echo 1)" "... and sourcing config.sh without it exits nothing and leaves DATA_ROOT empty (${src})"
rc=0; scripts/run_ps_pipeline.sh bogus >"${TMP}/j_bogus.log" 2>&1 || rc=$?
check "$([ "${rc}" = "2" ] && echo 0 || echo 1)" "run_ps_pipeline.sh with an argument refuses with exit 2 (${rc})"
check "$(has "${TMP}/j_bogus.log" 'takes no arguments')" "... and says why"
rc=0; scripts/run_ps_dtfe.sh --version >"${TMP}/j_version.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && [ -s "${TMP}/j_version.log" ] && echo 0 || echo 1)" "run_ps_dtfe.sh --version exits 0 with a revision"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s HASIC -g 16 99 98 >"${TMP}/j_dry.log" 2>&1 || rc=$?
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "DRY_RUN=1 run_ps_dtfe.sh exits 0 with a planned snapshot (${rc})"
check "$(has "${TMP}/j_dry.log" 'DRY RUN -- would run:')" "... prints the expanded command"
check "$(has "${TMP}/j_dry.log" 'PS-DTFE.*combined_099\.hdf5.*--grid +16')" "... with the binary, the input and the grid"
check "$(has "${TMP}/j_dry.log" 'DRY RUN: 1 snapshot\(s\) planned, 1 skipped')" "... one planned, the missing 098 skipped, nothing run"
check "$(lacks "${TMP}/j_dry.log" 'RUNNING:')" "... the binary never ran"
rc=0; env "${COMMON_ENV[@]}" DRY_RUN=1 "${DRIVER}" -s NOSUCH -g 16 99 >"${TMP}/j_dry2.log" 2>&1 || rc=$?
check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "DRY_RUN=1 with nothing to run exits 1 (${rc})"
if [ -x "${DTFE_BIN}" ]; then
    touch "${TMP}/j_marker"; sleep 1
    rc=0; DRY_RUN=1 scripts/run_dtfe.sh -s HASIC -g 16 99 >"${TMP}/j_dry3.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "DRY_RUN=1 run_dtfe.sh exits 0 (${rc})"
    check "$(has "${TMP}/j_dry3.log" 'DTFE.*combined_099\.hdf5.*--grid +16')" "... prints the standard binary's command"
    check "$([ ! "${TMP}/HASIC/snapdir_099/output.runlog" -nt "${TMP}/j_marker" ] && echo 0 || echo 1)" "... and leaves the run log untouched"
fi

echo ""
echo "(K) the binaries' --help lists the phase-space, GPU, point-evaluation and server options; no known misspelling."
"${PS_BIN}" --help >"${TMP}/k_help_ps.log" 2>&1 || true
"${DTFE_BIN}" --help >"${TMP}/k_help_dtfe.log" 2>&1 || true
for o in gpu exact sampled no-overlap full-help lambda-th grid-window ps-merge subsamples ps-per-stream ps-point-conventions mpc-unit ps-caustics ps-vertex-mass ps-volume-weighted sample-points serve serve-progress tessellation-cache auto-tune-report run-mode gpu-share; do
    check "$(has "${TMP}/k_help_ps.log" "^ *--${o}( |$)")" "PS-DTFE --help lists --${o}"
done
for o in gpu exact sampled no-overlap full-help lambda-th average-method subsamples mpc-unit sample-points serve auto-tune-report run-mode; do
    check "$(has "${TMP}/k_help_dtfe.log" "^ *--${o}( |$)")" "DTFE --help lists --${o}"
done
"${PS_BIN}" --full-help >"${TMP}/k_full_ps.log" 2>"${TMP}/k_full_ps.err" || true
"${DTFE_BIN}" --full-help >"${TMP}/k_full_dtfe.log" 2>&1 || true
check "$([ "$(lacks "${TMP}/k_help_dtfe.log" '^ *--(ps-gpu|ps-per-stream)')" = 0 ] \
         && [ "$(lacks "${TMP}/k_full_dtfe.log" '^ *--(ps-per-stream|ps-merge|grid-window)')" = 0 ] && echo 0 || echo 1)" \
      "... and not PS-DTFE's old --ps-gpu, nor its stream options (short and full help; the TBB insert is in both binaries since 2026-11-09)"
check "$([ "$(lacks "${TMP}/k_help_ps.log" '(^|[^-])--(ps-gpu|exact-average|ps-exact-deposit|ps-no-overlap|lambda_th|full_help|MpcUnit|paddingMpc|regionMpc|partNo|redshiftCone|origin|NGP|CIC|TSC|PCS|SPH|Voronoi|input|output|inputFile|outputFile|method|samples|noTest|density0|randomSample|approxPSD|extensive|options|avg-subsamples|lagrangianInput|ps-lag-padding|per-stream|per-stream-ids|pts-lagrangian|pts-conventions|ps-window|ordered-merge|completion-order-merge|fixed-point-merge)([^-a-zA-Z_]|$)')" = 0 ] \
         && [ "$(lacks "${TMP}/k_full_ps.log" '(^|[^-])--(ps-gpu|exact-average|ps-exact-deposit|ps-no-overlap|lambda_th|full_help|MpcUnit|paddingMpc|regionMpc|partNo|redshiftCone|origin|NGP|CIC|TSC|PCS|SPH|Voronoi|input|output|inputFile|outputFile|method|samples|noTest|density0|randomSample|approxPSD|extensive|options|avg-subsamples|lagrangianInput|ps-lag-padding|per-stream|per-stream-ids|pts-lagrangian|pts-conventions|ps-window|ordered-merge|completion-order-merge|fixed-point-merge)([^-a-zA-Z_]|$)')" = 0 ] \
         && [ "$(lacks "${TMP}/k_full_dtfe.log" '(^|[^-])--(ps-gpu|exact-average|ps-exact-deposit|ps-no-overlap|lambda_th|full_help|MpcUnit|paddingMpc|regionMpc|partNo|redshiftCone|origin|NGP|CIC|TSC|PCS|SPH|Voronoi|input|output|inputFile|outputFile|method|samples|noTest|density0|randomSample|approxPSD|extensive|options|avg-subsamples|lagrangianInput|ps-lag-padding|per-stream|per-stream-ids|pts-lagrangian|pts-conventions|ps-window|ordered-merge|completion-order-merge|fixed-point-merge)([^-a-zA-Z_]|$)')" = 0 ] \
         && [ "$(has "${TMP}/k_full_ps.log" '^ *--exact +PS-DTFE')" = 0 ] && [ "$(has "${TMP}/k_full_ps.log" '^ *--sampled( |$)')" = 0 ] \
         && [ "$(has "${TMP}/k_full_dtfe.log" '^ *--sampled( |$)')" = 0 ] && [ ! -s "${TMP}/k_full_ps.err" ] && echo 0 || echo 1)" \
      "the help (short and full, both binaries) names only the canonical spellings, no deprecated one (P6 and P6.7)"
"${PS_BIN}" --full_help >"${TMP}/k_full_old.log" 2>"${TMP}/k_full_old.err" || true
check "$(cmp -s "${TMP}/k_full_ps.log" "${TMP}/k_full_old.log" && grep -qx 'warning: --full_help is deprecated, use --full-help' "${TMP}/k_full_old.err" \
         && echo 0 || echo 1)" "--full_help (deprecated) still prints the full help, and one warning line on stderr"
check "$(lacks "${TMP}/k_full_ps.log" 'encompasing|documenation|vortivity|chuncks|coordiante|tesselation|specifed|suplied|strech|mannually|DMPC_UNIT')" \
      "the full help carries none of the known misspellings"
check "$(has "${TMP}/k_full_ps.log" "'MPC_UNIT'")" "... and names the Makefile's MPC_UNIT, not DMPC_UNIT"
# the deprecated spellings, for one release: hidden options folded onto the canonical ones (user_options.cc
# foldDeprecatedAliases), on the command line and in a --config file
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
rc1=0; "${PS_BIN}" "${KS}" "${TMP}/k_new" "${KC[@]}" --exact --lambda-th 0.3 --no-overlap >"${TMP}/k_new.log" 2>"${TMP}/k_new.err" || rc1=$?
rc2=0; "${PS_BIN}" "${KS}" "${TMP}/k_old" "${KC[@]}" --ps-exact-deposit --lambda_th 0.3 --ps-no-overlap >"${TMP}/k_old.log" 2>"${TMP}/k_old.err" || rc2=$?
rc3=0; "${PS_BIN}" "${KS}" "${TMP}/k_ea" "${KC[@]}" --exact-average --lambda-th 0.3 --no-overlap >"${TMP}/k_ea.log" 2>"${TMP}/k_ea.err" || rc3=$?
check "$([ "${rc1}" = 0 ] && [ "${rc2}" = 0 ] && [ "${rc3}" = 0 ] && k_same new old && k_same new ea && echo 0 || echo 1)" \
      "PS-DTFE: --ps-exact-deposit --lambda_th 0.3 --ps-no-overlap, and --exact-average, give the outputs of --exact --lambda-th 0.3 --no-overlap, byte for byte (${rc1}, ${rc2}, ${rc3})"
check "$([ "$(grep -c 'is deprecated, use' "${TMP}/k_old.err")" = 3 ] \
         && grep -qx 'warning: --ps-exact-deposit is deprecated, use --exact' "${TMP}/k_old.err" \
         && grep -qx 'warning: --lambda_th is deprecated, use --lambda-th' "${TMP}/k_old.err" \
         && [ "$(grep -c 'is deprecated, use' "${TMP}/k_ea.err")" = 1 ] && grep -qx 'warning: --exact-average is deprecated, use --exact' "${TMP}/k_ea.err" \
         && [ "$(lacks "${TMP}/k_new.err" 'deprecated')" = 0 ] && [ "$(lacks "${TMP}/k_old.log" 'deprecated')" = 0 ] && echo 0 || echo 1)" \
      "... one 'warning: --old is deprecated, use --new' line per old name, on stderr only; none for the canonical names"
printf '# a configuration file of an older release\nps-exact-deposit = true\nlambda_th = 0.3\nps-no-overlap = 1\n' > "${TMP}/k_old.cfg"
rc=0; "${PS_BIN}" "${KS}" "${TMP}/k_cfg" "${KC[@]}" --config "${TMP}/k_old.cfg" >"${TMP}/k_cfg.log" 2>"${TMP}/k_cfg.err" || rc=$?
check "$([ "${rc}" = 0 ] && k_same new cfg && [ "$(grep -c 'is deprecated, use' "${TMP}/k_cfg.err")" = 3 ] && echo 0 || echo 1)" \
      "... and in a --config file: the same outputs, the same warnings (${rc})"
rc=0; "${PS_BIN}" "${KS}" "${TMP}/k_bad" "${KC[@]}" --lambda_th 0.3 --lambda-th 0.2 >"${TMP}/k_bad.log" 2>&1 || rc=$?
rc_ok=0; "${DTFE_BIN}" "${KS}" "${TMP}/k_dok" "${KC[@]}" --lambda_th 0.3 --lambda-th 0.3 >"${TMP}/k_dok.log" 2>&1 || rc_ok=$?
check "$([ "${rc}" != 0 ] && [ "$(has "${TMP}/k_bad.log" "'--lambda_th 0.3' \(the deprecated spelling\) and '--lambda-th 0.2' disagree")" = 0 ] \
         && [ "$(lacks "${TMP}/k_bad.log" 'RUNNING:')" = 0 ] && [ "${rc_ok}" = 0 ] && echo 0 || echo 1)" \
      "... both spellings with different values are refused before anything runs; with the same value they agree (${rc}, DTFE ${rc_ok})"
rc=0; "${DTFE_BIN}" "${KS}" "${TMP}/k_dold" "${KC[@]}" --lambda_th 0.3 >"${TMP}/k_dold.log" 2>"${TMP}/k_dold.err" || rc=$?
rc_d=0; "${DTFE_BIN}" "${KS}" "${TMP}/k_dps" "${KC[@]}" --ps-gpu >"${TMP}/k_dps.log" 2>&1 || rc_d=$?
check "$([ "${rc}" = 0 ] && [ "${rc_ok}" = 0 ] && k_same dok dold && grep -qx 'warning: --lambda_th is deprecated, use --lambda-th' "${TMP}/k_dold.err" \
         && [ "${rc_d}" != 0 ] && [ "$(has "${TMP}/k_dps.log" "unrecognised option '--ps-gpu'")" = 0 ] && echo 0 || echo 1)" \
      "standard DTFE: --lambda_th too (the same outputs, a warning); PS-DTFE's old --ps-gpu was never its own option (${rc}, ${rc_d})"
# the command line overrides the --config file across the two spellings too (user_options.cc
# dropConfigAliasesOfCommandLine): the file's 5 under the other name gives way to the command line's 0.3, with the
# one warning the deprecated name earns; the file alone runs at its 5 (every cell a void: a T-web the comparisons
# tell from 0.3's, which 0.1-1 are not on this snapshot); two spellings disagreeing inside one file are still refused
printf 'lambda_th = 5\n' > "${TMP}/k_lam_old.cfg"
printf 'lambda-th = 5\n' > "${TMP}/k_lam_new.cfg"
printf 'lambda_th = 0.3\nlambda-th = 0.2\n' > "${TMP}/k_lam_both.cfg"
k_run() {   # <binary> <tag> <options...>: run on the K snapshot, stdout in k_<tag>.log, stderr in k_<tag>.err; echo the exit
    local bin="$1" tag="$2" r=0
    shift 2
    "${bin}" "${KS}" "${TMP}/k_${tag}" "${KC[@]}" "$@" >"${TMP}/k_${tag}.log" 2>"${TMP}/k_${tag}.err" || r=$?
    echo "${r}"
}
k_one_warning() {   # <tag>: exactly one deprecation line on stderr, the --lambda_th one
    [ "$(grep -c 'is deprecated, use' "${TMP}/k_$1.err")" = 1 ] \
        && grep -qx 'warning: --lambda_th is deprecated, use --lambda-th' "${TMP}/k_$1.err"
}
for b in ps dtfe; do
    if [ "${b}" = ps ]; then bin="${PS_BIN}"; ref=new; extra=(--exact --no-overlap)
    else bin="${DTFE_BIN}"; ref=dok; extra=(); fi
    r1="$(k_run "${bin}" "${b}_lfile" ${extra[@]+"${extra[@]}"} --config "${TMP}/k_lam_old.cfg")"
    r2="$(k_run "${bin}" "${b}_lold" ${extra[@]+"${extra[@]}"} --config "${TMP}/k_lam_old.cfg" --lambda-th 0.3)"
    r3="$(k_run "${bin}" "${b}_lnew" ${extra[@]+"${extra[@]}"} --config "${TMP}/k_lam_new.cfg" --lambda_th 0.3)"
    r4="$(k_run "${bin}" "${b}_lboth" ${extra[@]+"${extra[@]}"} --config "${TMP}/k_lam_both.cfg")"
    check "$([ "${r1}" = 0 ] && [ "${r2}" = 0 ] && [ "${r3}" = 0 ] && ! k_same "${ref}" "${b}_lfile" \
             && k_same "${ref}" "${b}_lold" && k_same "${ref}" "${b}_lnew" && k_one_warning "${b}_lold" && k_one_warning "${b}_lnew" \
             && echo 0 || echo 1)" \
          "${b}: '--lambda-th 0.3' over a file's 'lambda_th = 5', and '--lambda_th 0.3' over 'lambda-th = 5', run at 0.3 with one warning; the file alone at 5 (${r1} ${r2} ${r3})"
    check "$([ "${r4}" != 0 ] && [ "$(has "${TMP}/k_${b}_lboth.log" "'--lambda_th 0.3' \(the deprecated spelling\) and '--lambda-th 0.2' disagree")" = 0 ] \
             && [ "$(lacks "${TMP}/k_${b}_lboth.log" 'RUNNING:')" = 0 ] && echo 0 || echo 1)" \
          "${b}: ... and a file giving both spellings with different values is refused before anything runs (${r4})"
done

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
check "$(has "${TMP}/l_backup.log" '^backed up 5$')" "backup_binaries counts the five binaries present (of the eight names)"
check "$(has "${TMP}/l_backup.log" '^restored 5$')" "restore_binaries puts the five back"
check "$([ "$(cat "${LREPO}/PS-DTFE")" = "old PS-DTFE" ] && [ "$(cat "${LREPO}/DTFE")" = "old DTFE" ] && [ -x "${LREPO}/DTFE-2d" ] && echo 0 || echo 1)" \
      "... byte for byte, the half-linked one replaced, the mode kept"
check "$([ "$(cat "${LREPO}/DTFE-2d-double")" = "old 2d double" ] && [ "$(cat "${LREPO}/PS-DTFE-2d-double")" = "old PS 2d double" ] \
         && [ -x "${LREPO}/PS-DTFE-2d-double" ] && echo 0 || echo 1)" "... the 2D double pair among them"
check "$(has "${TMP}/l_hint.log" 'ld: symbol not found')" "build_failed_hint prints the log's tail"
check "$(lacks "${TMP}/l_hint.log" '^line1$')" "... only the last N lines"
check "$(has "${TMP}/l_hint.log" 'make deps-check')" "... and the usual causes"
# install.sh rebuilds the double, 2D and 2D double sets it finds before 'make clean' (else a re-run would throw
# them away). install.sh itself is never run here: its detection block, cut out between 'EXTRA_SETS=()'
# and the 'also to be rebuilt' line, is sourced in fake folders with info() stubbed out
sed -n '/^EXTRA_SETS=()/,/also to be rebuilt/p' scripts/install.sh > "${TMP}/l_extra_sets.sh"
check "$(grep -q '^EXTRA_SETS=()' "${TMP}/l_extra_sets.sh" && grep -q 'DIM=2 DOUBLE=1' "${TMP}/l_extra_sets.sh" \
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
check "$([ -z "${got}" ] && echo 0 || echo 1)" "... the single pair alone: no extra set (${got:-none})"
got="$(extra_sets "" "METAL=1" DTFE-2d DTFE-double DTFE-2d-double)"
check "$([ "${got}" = "[METAL=1 DOUBLE=1][DIM=2][DIM=2 DOUBLE=1]" ] && echo 0 || echo 1)" \
      "... DTFE-double, DTFE-2d and DTFE-2d-double present: the double set with the GPU backend, then DIM=2 and DIM=2 DOUBLE=1 (${got})"
got="$(extra_sets "" "" PS-DTFE-2d-double)"
check "$([ "${got}" = "[DIM=2 DOUBLE=1]" ] && echo 0 || echo 1)" "... either binary of a pair is enough: PS-DTFE-2d-double alone (${got})"
got="$(extra_sets "DOUBLE=1" "")"
check "$([ "${got}" = "[ DOUBLE=1]" ] && echo 0 || echo 1)" "... --double with no double pair yet builds it, CPU-only here (${got})"

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
# '.part.<pid>' and renamed into place, a second later '<out>.badGridPoints0' rewritten IN PLACE when it exists)
echo "RUNNING: $0 $*"
printf 'fake attempt %s \033[1mcoloured\033[0m\n' "${FAKE_TAG:-}"
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
    check "$([ "${rc}" = "0" ] && [ -f "${L}" ] && [ ! -e "${L}.partial" ] && [ ! -e "${L}.failed" ] && echo 0 || echo 1)" \
          "${scr}: a good run's log is ${pfx}.runlog, and nothing else is left (exit ${rc})"
    check "$(lacks "${L}" $'\033\\[')" "... de-ANSI'd"
    cp "${L}" "${TMP}/m_${pfx}_first.runlog"
    # a failure (an early error, a bad option, a missing input): the grids on disk keep their log
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_fail.log" FAKE_MODE=fail FAKE_TAG=second)
    check "$([ "${rc}" = "1" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... a failed re-run exits 1 and leaves the .runlog of the grids on disk byte for byte (${rc})"
    check "$([ ! -e "${L}.partial" ] && grep -aq 'fake attempt second' "${L}.failed" && ! grep -aq $'\033\\[' "${L}.failed" && echo 0 || echo 1)" \
          "... its own log is kept, de-ANSI'd, as ${pfx}.runlog.failed (for the failure diagnosis); no .partial left"
    check "$(has "${TMP}/m_${pfx}_fail.log" "Error processing snapshot 099 \(its log: .*${pfx}\.runlog\.failed\)")" "... and the driver names it"
    # a stop as the launcher's Stop does it: SIGTERM to the driver (the trap waits for the binary), then the binary
    mhang "${scr}" "${TMP}/m_${pfx}_stop.log" third
    check "$([ -f "${L}.partial" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... while a re-run goes its log is ${pfx}.runlog.partial; the .runlog stays as it was"
    kill -TERM "${mpid}" 2>/dev/null || true
    sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    rc=0; wait "${mpid}" 2>/dev/null || rc=$?
    check "$([ "${rc}" = "143" ] && echo 0 || echo 1)" "... a stopped run ends by its signal, 143, so a calling script stops too (${rc})"
    check "$([ ! -e "${L}.partial" ] && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && grep -aq 'fake attempt third' "${L}.failed" && echo 0 || echo 1)" \
          "... the stopped attempt's log is the .runlog.failed now; the .runlog is untouched, no .partial left"
    check "$(has "${TMP}/m_${pfx}_stop.log" 'Stopped: snapshot 099 did not finish')" "... and the driver says so"
    # a run killed outright (SIGKILL, a restart): its .partial is left, the .runlog still untouched
    mhang "${scr}" "${TMP}/m_${pfx}_kill.log" fourth
    kill -KILL "${mpid}" 2>/dev/null || true
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true       # the orphaned binary
    wait "${mpid}" 2>/dev/null || true
    sleep 1                                                             # its orphaned tee writes its last lines
    check "$(grep -aq 'fake attempt fourth' "${L}.partial" && cmp -s "${L}" "${TMP}/m_${pfx}_first.runlog" && echo 0 || echo 1)" \
          "... a run killed outright leaves its ${pfx}.runlog.partial, and the .runlog still untouched"
    # the next success: its log is the .runlog, the stale .failed and .partial are gone
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_ok2.log" FAKE_MODE=ok FAKE_TAG=fifth)
    check "$([ "${rc}" = "0" ] && grep -aq 'fake attempt fifth' "${L}" && [ ! -e "${L}.failed" ] && [ ! -e "${L}.partial" ] && echo 0 || echo 1)" \
          "... the next good run: its log is the .runlog; the stale .runlog.failed and .runlog.partial are gone (${rc})"
done
# an auto-tune report writes nothing next to the data: not the .runlog, nor a .partial or .failed (run_dtfe.sh has
# the knob too)
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    cp "${MDATA}/SIM/snapdir_099/${pfx}.runlog" "${TMP}/m_before_report.runlog"
    rc=$(mdrive "${scr}" "${TMP}/m_${pfx}_report.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
    check "$([ "${rc}" = "0" ] && grep -aqE '^  AUTO-TUNE-REPORT snap=099 ' "${TMP}/m_${pfx}_report.log" \
             && [ "$(has "${TMP}/m_${pfx}_report.log" '^RUNNING: .*--auto-tune-report')" = 0 ] && echo 0 || echo 1)" \
          "${scr} AUTO_TUNE_REPORT=1 passes --auto-tune-report and reports (${rc})"
    check "$(cmp -s "${MDATA}/SIM/snapdir_099/${pfx}.runlog" "${TMP}/m_before_report.runlog" \
             && [ ! -e "${MDATA}/SIM/snapdir_099/${pfx}.runlog.partial" ] \
             && [ ! -e "${MDATA}/SIM/snapdir_099/${pfx}.runlog.failed" ] && echo 0 || echo 1)" \
          "... and leaves the snapshot's run logs alone"
done

echo ""
echo "(O) the single-instance lock and the preflight of run_dtfe.sh / run_ps_dtfe.sh."
# Contract C1: '<prefix>.runlog.lock' is a directory holding the driver's pid while it runs that snapshot. A
# second driver on the same snapshot is refused (two TNG runs together get both SIGKILLed for memory);
# a lock whose driver is gone is stale and removed with a message; a stop releases it.
for pair in "run_dtfe.sh output" "run_ps_dtfe.sh ps_output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    L="${MDATA}/SIM/snapdir_099/${pfx}.runlog"; K="${L}.lock"
    rm -rf "${K}"
    mhang "${scr}" "${TMP}/o_${pfx}_hold.log" sixth
    check "$([ -d "${K}" ] && [ "$(cat "${K}/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "${scr}: while it runs, ${pfx}.runlog.lock holds the driver's pid (${mpid})"
    cp "${L}" "${TMP}/o_${pfx}_before.runlog"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_second.log" FAKE_MODE=ok FAKE_TAG=intruder)
    check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... a second driver on the same snapshot is refused: exit 1 (${rc})"
    check "$(has "${TMP}/o_${pfx}_second.log" "another driver \(pid ${mpid}\) is running this snapshot")" "... naming the pid that holds it"
    check "$(lacks "${TMP}/o_${pfx}_second.log" '^RUNNING:')" "... and its binary never started"
    check "$([ -f "${L}.partial" ] && ! grep -aq 'intruder' "${L}.partial" && cmp -s "${L}" "${TMP}/o_${pfx}_before.runlog" \
             && [ "$(cat "${K}/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "... the running attempt's .partial, the grids' .runlog and the lock are untouched"
    kill -TERM "${mpid}" 2>/dev/null || true
    sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    rc=0; wait "${mpid}" 2>/dev/null || rc=$?
    check "$([ "${rc}" = "143" ] && [ ! -e "${K}" ] && echo 0 || echo 1)" "... a stop still exits 143 and releases the lock (${rc})"
    # the driver alone SIGKILLed: its binary and tee keep running on the snapshot, so its lock is not
    # stale (removing it would start a second binary beside the orphan)
    mhang "${scr}" "${TMP}/o_${pfx}_orphan.log" orphan
    kill -KILL "${mpid}" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_beside.log" FAKE_MODE=ok FAKE_TAG=beside)
    check "$([ "${rc}" = "1" ] && [ -d "${K}" ] && grep -aq 'is gone, but a process still' "${TMP}/o_${pfx}_beside.log" \
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
    check "$([ -d "${K}" ] && echo 0 || echo 1)" "... a SIGKILLed driver leaves its lock"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_after.log" FAKE_MODE=ok FAKE_TAG=eighth)
    check "$([ "${rc}" = "0" ] && [ ! -e "${K}" ] && grep -aq 'fake attempt eighth' "${L}" && echo 0 || echo 1)" \
          "... the next run removes the stale lock, succeeds and leaves no lock (${rc})"
    check "$(has "${TMP}/o_${pfx}_after.log" 'removing a stale lock \(pid [0-9]+ is no longer running\)')" "... and says so"
    # a reused pid: alive, but not a driver -- stale as well
    sleep 30 & other=$!
    mkdir -p "${K}"; echo "${other}" > "${K}/pid"
    rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_reused.log" FAKE_MODE=ok)
    kill "${other}" 2>/dev/null || true; wait "${other}" 2>/dev/null || true
    check "$([ "${rc}" = "0" ] && [ ! -e "${K}" ] && echo 0 || echo 1)" "... a lock whose pid now belongs to another program is stale too (${rc})"
    # preflight: the grids must fit on the output volume (a 20000^3 grid needs ~1e14 bytes); SPACE_CHECK=0 skips
    rc=0; env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=ok "${MREPO}/scripts/${scr}" -s SIM -g 20000 99 >"${TMP}/o_${pfx}_space.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... grids larger than the free space: refused, exit 1 (${rc})"
    check "$(has "${TMP}/o_${pfx}_space.log" "GB free; this run's grids need about")" "... saying how much is free and needed"
    check "$(lacks "${TMP}/o_${pfx}_space.log" '^RUNNING:')" "... before the binary started"
    rc=0; env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=ok SPACE_CHECK=0 "${MREPO}/scripts/${scr}" -s SIM -g 20000 99 >"${TMP}/o_${pfx}_nospace.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... SPACE_CHECK=0 skips the space check (${rc})"
    rc=0; env DTFE_DATA_ROOT="${MDATA}" FAKE_MODE=ok DTFE_SPACE_CHECK=off "${MREPO}/scripts/${scr}" -s SIM -g 20000 99 >"${TMP}/o_${pfx}_nospace2.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "0" ] && [ "$(has "${TMP}/o_${pfx}_nospace2.log" '^warning: DTFE_SPACE_CHECK is deprecated, use SPACE_CHECK$')" = 0 ] && echo 0 || echo 1)" \
          "... and so does DTFE_SPACE_CHECK=off, the older name (one warning; 'off' is off, as in the binaries) (${rc})"
    # the data disk unplugged: one clear message up front, exit 2 (not a skip of every snapshot under a wrong path)
    rc=0; env DTFE_DATA_ROOT="${TMP}/no such disk/Illustris TNG" "${MREPO}/scripts/${scr}" -s SIM 99 >"${TMP}/o_${pfx}_root.log" 2>&1 || rc=$?
    check "$([ "${rc}" = "2" ] && echo 0 || echo 1)" "... a data root that does not exist: exit 2 (${rc})"
    check "$(has "${TMP}/o_${pfx}_root.log" 'does not exist: is the data disk mounted')" "... asking whether the disk is mounted"
    check "$(lacks "${TMP}/o_${pfx}_root.log" 'Processing snapshot')" "... before any snapshot"
    # an output folder that is not writable (root writes anywhere: skipped under root, as in the Docker image)
    if [ "$(id -u)" != "0" ]; then
        chmod a-w "${MDATA}/SIM/snapdir_099"
        rc=$(mdrive "${scr}" "${TMP}/o_${pfx}_ro.log" FAKE_MODE=ok)
        chmod u+w "${MDATA}/SIM/snapdir_099"
        check "$([ "${rc}" = "1" ] && echo 0 || echo 1)" "... an output folder that is not writable: refused, exit 1 (${rc})"
        check "$(has "${TMP}/o_${pfx}_ro.log" 'is not writable')" "... saying so"
    fi
done
# a report and a dry run take no lock: the launcher's memory check runs a report beside a live run
for pair in "run_ps_dtfe.sh ps_output" "run_dtfe.sh output"; do
    scr="${pair% *}"; pfx="${pair#* }"
    mhang "${scr}" "${TMP}/o_hold_report_${pfx}.log" ninth
    rc=$(mdrive "${scr}" "${TMP}/o_report_${pfx}.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
    rc2=$(mdrive "${scr}" "${TMP}/o_dry_${pfx}.log" DRY_RUN=1)
    check "$([ "${rc}" = "0" ] && [ "${rc2}" = "0" ] && [ "$(cat "${MDATA}/SIM/snapdir_099/${pfx}.runlog.lock/pid" 2>/dev/null)" = "${mpid}" ] && echo 0 || echo 1)" \
          "${scr}: an auto-tune report and a dry run go ahead beside a live run, and leave its lock alone (${rc}, ${rc2})"
    kill -TERM "${mpid}" 2>/dev/null || true; sleep 0.3
    kill -TERM "$(cat "${MPID}" 2>/dev/null)" 2>/dev/null || true
    wait "${mpid}" 2>/dev/null || true
done

echo ""
echo "(R) run_ps_dtfe.sh: PS_MERGE (index-order by default, passed explicitly), and --resume when a killed run left its checkpoint."
# Contracts K1/K2: the binary checkpoints a partitioned run in '<output>.runlog.checkpoint' (under --scratch-dir:
# '<scratch>/<basename>.<hash of the output path>.runlog.checkpoint') and removes it on success, so a folder there
# means a killed run. Against the stand-in binaries of (M), which print their arguments.
RSD="${MDATA}/SIM/snapdir_099"; RCK="${RSD}/ps_output.runlog.checkpoint"
rm -rf "${RCK}" "${RSD}/ps_output.runlog.lock"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_plain.log" FAKE_MODE=ok)
check "$([ "${rc}" = "0" ] && grep -aq '^RUNNING: ' "${TMP}/r_plain.log" && echo 0 || echo 1)" "a plain run goes (${rc})"
# the index-order (staged) merge is the binary's default; the driver passes it explicitly ('--ps-merge index-order':
# given, the memory gate stays on under the auto-tuner's speed split, as its older '--ordered-merge' default did)
MERGE_FLAGS='--(ps-merge|ordered-merge|completion-order-merge|fixed-point-merge)'
merge_of() {   # <log>: the merge flags of the RUNNING line, or 'none'
    local m; m="$(grep -a '^RUNNING: ' "$1" | grep -oE -e "${MERGE_FLAGS}( [a-z][a-z-]*)?" | tr '\n' ' ')"
    echo "${m:-none}"
}
check "$([ "$(merge_of "${TMP}/r_plain.log")" = "--ps-merge index-order " ] \
         && [ "$(lacks "${TMP}/r_plain.log" '^RUNNING: .*--resume')" = 0 ] && echo 0 || echo 1)" \
      "... with '--ps-merge index-order' (the driver's default, explicit) and no --resume"
m=""
for e in PS_MERGE=index-order PS_MERGE=completion-order PS_MERGE=fixed-point ORDERED_MERGE=1 ORDERED_MERGE=0 \
         FIXED_POINT_MERGE=1 "ORDERED_MERGE=1 FIXED_POINT_MERGE=1" FIXED_POINT_MERGE=0 "PS_MERGE=completion-order ORDERED_MERGE=1"; do
    # shellcheck disable=SC2086
    rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_merge.log" FAKE_MODE=ok ${e})
    m="${m}$(merge_of "${TMP}/r_merge.log")| "
done
check "$([ "${m}" = "--ps-merge index-order | --ps-merge completion-order | --ps-merge fixed-point | --ps-merge index-order | --ps-merge completion-order | --ps-merge fixed-point | --ps-merge fixed-point | --ps-merge index-order | --ps-merge completion-order | " ] \
         && echo 0 || echo 1)" \
      "PS_MERGE=MODE passes '--ps-merge MODE'; ORDERED_MERGE=1 / 0 and FIXED_POINT_MERGE=1, the older pair, convert (fixed-point winning; FIXED_POINT_MERGE=0 alone leaves the default); PS_MERGE wins over them (${m})"
check "$([ "$(grep -c '^warning: ORDERED_MERGE is deprecated, use PS_MERGE=index-order (ORDERED_MERGE=1 is not read: PS_MERGE=completion-order wins)$' "${TMP}/r_merge.log")" = 1 ] \
         && echo 0 || echo 1)" "... the older name is said deprecated, and that the new one wins"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_merge_bad.log" FAKE_MODE=ok PS_MERGE=ordered)
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/r_merge_bad.log" "PS_MERGE must be 'index-order', 'completion-order' or 'fixed-point'")" = 0 ] \
         && [ "$(lacks "${TMP}/r_merge_bad.log" '^RUNNING: ')" = 0 ] && echo 0 || echo 1)" "... another value is refused before anything runs (exit ${rc})"
mkdir -p "${RCK}"; : > "${RCK}/checkpoint"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_resume.log" FAKE_MODE=ok)
check "$([ "${rc}" = "0" ] && grep -aqE -e '^RUNNING: .*--resume' "${TMP}/r_resume.log" && echo 0 || echo 1)" "a checkpoint folder beside the outputs: --resume is passed (${rc})"
check "$(has "${TMP}/r_resume.log" "Resuming: a killed run's checkpoint is in .*ps_output\.runlog\.checkpoint")" "... and the driver says so, naming the folder"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_noresume.log" FAKE_MODE=ok DTFE_RESUME=0)
check "$(lacks "${TMP}/r_noresume.log" '^RUNNING: .*--resume')" "DTFE_RESUME=0: no --resume (${rc})"
check "$(has "${TMP}/r_noresume.log" 'DTFE_RESUME=0, so this run starts over')" "... and the driver says the run starts over"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_noresume_old.log" FAKE_MODE=ok PS_RESUME=0)
check "$([ "$(lacks "${TMP}/r_noresume_old.log" '^RUNNING: .*--resume')" = 0 ] \
         && [ "$(has "${TMP}/r_noresume_old.log" '^warning: PS_RESUME is deprecated, use DTFE_RESUME$')" = 0 ] && echo 0 || echo 1)" \
      "... PS_RESUME=0, the older name, does the same, with its warning (${rc})"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_report.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
check "$(lacks "${TMP}/r_report.log" '(--resume|Resuming)')" "an auto-tune report never resumes (${rc})"
# a folder with no 'checkpoint' file (a run killed during its first write, or one that never had room) holds
# nothing to resume: the binary refuses '--resume' there, so passing it would make every rerun fail
rm -f "${RCK}/checkpoint"; : > "${RCK}/checkpoint.tmp.99999"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_emptyck.log" FAKE_MODE=ok)
check "$(lacks "${TMP}/r_emptyck.log" '(^RUNNING: .*--resume|Resuming)')" "a checkpoint folder without a 'checkpoint' file: no --resume (${rc})"
rm -rf "${RCK}"
RSCR="${TMP}/r scratch"; mkdir -p "${RSCR}"
SCK="$(source scripts/config.sh && ps_checkpoint_dir "${RSD}/ps_output" "${RSCR}")"
check "$([[ "$(basename "${SCK}")" =~ ^ps_output\.[0-9a-f]{16}\.runlog\.checkpoint$ ]] && echo 0 || echo 1)" \
      "config.sh ps_checkpoint_dir under a scratch dir: <basename>.<16 hex>.runlog.checkpoint ($(basename "${SCK}"))"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_scr0.log" FAKE_MODE=ok SCRATCH_DIR="${RSCR}")
check "$(lacks "${TMP}/r_scr0.log" '^RUNNING: .*--resume')" "SCRATCH_DIR set, no checkpoint there: no --resume (${rc})"
mkdir -p "${SCK}"; : > "${SCK}/checkpoint"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_scr1.log" FAKE_MODE=ok SCRATCH_DIR="${RSCR}")
check "$(has "${TMP}/r_scr1.log" '^RUNNING: .*--resume')" "... the checkpoint in the scratch folder: --resume is passed (${rc})"
rm -rf "${SCK}"
# the plane's conventions: legacy only on request and only with points; the binary's default otherwise
: > "${TMP}/r_points.txt"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_def.log" FAKE_MODE=ok SAMPLE_POINTS="${TMP}/r_points.txt")
rc2=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_leg.log" FAKE_MODE=ok SAMPLE_POINTS="${TMP}/r_points.txt" PS_POINT_CONVENTIONS=legacy)
check "$([ "$(lacks "${TMP}/r_pc_def.log" '^RUNNING: .*--(ps-point|pts)-conventions')" = 0 ] \
         && [ "$(has "${TMP}/r_pc_leg.log" '^RUNNING: .*--sample-points .*r_points\.txt.* --ps-point-conventions legacy')" = 0 ] && echo 0 || echo 1)" \
      "PS_POINT_CONVENTIONS: the grids' (the default) passes nothing, legacy passes --ps-point-conventions legacy with the points (${rc} ${rc2})"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_nopts.log" FAKE_MODE=ok PS_POINT_CONVENTIONS=legacy)
check "$(lacks "${TMP}/r_pc_nopts.log" '^RUNNING: .*--(ps-point|pts)-conventions')" "... and nothing without points (${rc})"
rc=$(mdrive run_ps_dtfe.sh "${TMP}/r_pc_bad.log" FAKE_MODE=ok PS_POINT_CONVENTIONS=old)
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/r_pc_bad.log" "PS_POINT_CONVENTIONS must be 'grid' or 'legacy'")" = 0 ] \
         && [ "$(lacks "${TMP}/r_pc_bad.log" '^RUNNING: ')" = 0 ] && echo 0 || echo 1)" \
      "... another value is refused before anything runs (exit ${rc})"

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
    env -u DTFE_EXACT_AVERAGE -u PS_EXACT -u SUBSAMPLES -u AVG_SUBSAMPLES -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID \
        DTFE_DATA_ROOT="${PDATA}" SIMS=SIM NU=8 GRID_SIZE=16 FIELDS="density_a velocity_a" DTFE_GPU=0 FAKE_MODE=ok \
        "$@" "${MREPO}/scripts/run_ps_pipeline.sh" >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
SD="${PDATA}/SIM/snapdir_099"
rc=$(pdrive "${TMP}/p_first.log")
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"           # a slice of the 8^2 plane: 64 float64, as a real run leaves
touch -t 203001010000 "${SD}/ps_output.pts_den" "${SD}/ps_output.pts_velGrad"   # what a real run leaves (newer than its inputs)
check "$([ "${rc}" = "0" ] && grep -aq '^RUNNING: ./PS-DTFE[^ ]* .*--grid 16 .*--exact ' "${SD}/ps_output.runlog" && echo 0 || echo 1)" \
      "a first pipeline run computes the snapshot (the exact deposit, the default), its run log carries the command (${rc})"
rc=$(pdrive "${TMP}/p_same.log")
check "$([ "${rc}" = "0" ] && echo 0 || echo 1)" "... a rerun with the same settings exits 0 (${rc})"
check "$(has "${TMP}/p_same.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" "... and skips the snapshot: same settings"
check "$(lacks "${TMP}/p_same.log" 'run_ps_dtfe.sh -s SIM')" "... running nothing"
# a slice newer than the plane but of ANOTHER plane size (a 16^2 one against the 8^2 plane: TNG100's
# 4096^2 slices beside an older hires_plane_z8192) is not 'up to date': skipping the compute would fail every render
head -c 2048 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den"
rc=$(pdrive "${TMP}/p_size.log" DRY_RUN=1)
check "$(has "${TMP}/p_size.log" 'snap 99: ps_output.pts_den holds 256 points, the plane 64: a slice of another plane, recomputing')" \
      "a slice of another plane size: recomputed, though newer than the plane and with the same settings (${rc})"
check "$(has "${TMP}/p_size.log" 'run_ps_dtfe.sh -s SIM -g 16 99')" "... run_ps_dtfe.sh is called for it"
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den"
rc=$(pdrive "${TMP}/p_grid.log" GRID_SIZE=24 DRY_RUN=1)
check "$(has "${TMP}/p_grid.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" "a different grid: the snapshot is recomputed (${rc})"
check "$(grep -aq 'was: --grid 16$' "${TMP}/p_grid.log" && grep -aq 'now: --grid 24$' "${TMP}/p_grid.log" && echo 0 || echo 1)" "... the difference is shown"
check "$(has "${TMP}/p_grid.log" 'run_ps_dtfe.sh -s SIM -g 24 99')" "... and run_ps_dtfe.sh is called for it"
rc=$(pdrive "${TMP}/p_vw.log" PS_VOLUME_WEIGHTED=0 DRY_RUN=1)
check "$(has "${TMP}/p_vw.log" 'was: --ps-volume-weighted')" "a deposit option switched off (PS_VOLUME_WEIGHTED=0) counts too"
rc=$(pdrive "${TMP}/p_sub.log" FIELDS=density_a DRY_RUN=1)
check "$(has "${TMP}/p_sub.log" 'up to date \(same settings\)')" "fewer fields than the grids on disk carry: still up to date"
rc=$(pdrive "${TMP}/p_more.log" FIELDS="density_a velocity_a tweb_a" DRY_RUN=1)
check "$(has "${TMP}/p_more.log" 'settings differ')" "... a field the grids lack: recomputed"
rc=$(pdrive "${TMP}/p_gpu.log" DTFE_GPU=1 DRY_RUN=1)
check "$(has "${TMP}/p_gpu.log" 'up to date \(same settings\)')" "the GPU/CPU choice alone does not make grids stale"
# the plane's conventions are a setting of the slice; the pipeline hands them to run_ps_dtfe.sh
rc=$(pdrive "${TMP}/p_pc.log" PS_POINT_CONVENTIONS=legacy DRY_RUN=1)
check "$([ "$(has "${TMP}/p_pc.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'now: --ps-point-conventions legacy$' "${TMP}/p_pc.log" \
         && [ "$(has "${TMP}/p_pc.log" '^\+ env .*PS_POINT_CONVENTIONS=legacy .*run_ps_dtfe\.sh -s SIM')" = 0 ] && echo 0 || echo 1)" \
      "PS_POINT_CONVENTIONS=legacy: a slice in the grids' conventions is recomputed, the difference shown, the knob passed on (${rc})"
rc=$(pdrive "${TMP}/p_pcg.log" PS_POINT_CONVENTIONS=grid DRY_RUN=1)
check "$(has "${TMP}/p_pcg.log" 'up to date \(same settings\)')" "... PS_POINT_CONVENTIONS=grid (the default, said or not): up to date (${rc})"
rc=$(pdrive "${TMP}/p_pco.log" PTS_CONVENTIONS=legacy DRY_RUN=1)
check "$([ "$(has "${TMP}/p_pco.log" '^\+ env .*PS_POINT_CONVENTIONS=legacy .*run_ps_dtfe\.sh -s SIM')" = 0 ] \
         && [ "$(lacks "${TMP}/p_pco.log" '^\+ env .*PTS_CONVENTIONS=')" = 0 ] \
         && [ "$(grep -c '^warning: PTS_CONVENTIONS is deprecated, use PS_POINT_CONVENTIONS$' "${TMP}/p_pco.log")" = 1 ] && echo 0 || echo 1)" \
      "... PTS_CONVENTIONS, the older name, is read with one warning and handed on under the new name only (${rc})"
# the binaries' deprecated spellings: a run log written with '--subsamples 3 --ps-exact-deposit --lambda_th 0.3' (an
# older driver, which named the count beside the exact deposit too) against the '--exact --lambda-th 0.3' the driver
# emits now must compare UNCHANGED -- else every snapshot on disk would be recomputed. run_settings alone first, then
# the whole freshness test on a run log rewritten that way.
eval "$(awk '$0 == "run_settings() {" { on = 1 } on { print } on && $0 == "}" { exit }' "${MREPO}/scripts/run_ps_pipeline.sh")"
rs_old="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --grid 64 --subsamples 3 --field density_a --lambda_th 0.3 --ps-gpu --ps-exact-deposit --ps-no-overlap --ps-caustics')"
rs_ea="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --grid 64 --avg-subsamples 3 --field density_a --lambda-th 0.3 --exact-average --ps-caustics')"
rs_new="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --gpu --exact --no-overlap --ps-caustics')"
rs_other="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda_th 0.2 --ps-exact-deposit --ps-caustics')"
check "$([ -n "${rs_new}" ] && [ "${rs_old}" = "${rs_new}" ] && [ "${rs_ea}" = "${rs_new}" ] && printf '%s\n' "${rs_new}" | grep -qx -e '--exact' \
         && ! printf '%s\n' "${rs_new}" | grep -q -e '^--subsamples' \
         && printf '%s\n' "${rs_new}" | grep -qx -e '--lambda-th 0.3' && [ "${rs_other}" != "${rs_new}" ] && echo 0 || echo 1)" \
      "run_settings: an old-spelling exact log (--ps-exact-deposit or --exact-average, beside the old count) and the new --exact command are unchanged (the count is no setting of the exact deposit), another threshold is not"
# an older SAMPLED run named no deposit flag and always a count: it is the command a sampled pipeline run emits now
# ('--sampled', with that count or none: the binary's own 3), and not the exact default's
rs_s_old="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --grid 64 --subsamples 3 --field density_a --lambda-th 0.3 --ps-caustics')"
rs_s_new="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --sampled --ps-caustics')"
rs_s_n3="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --sampled --subsamples 3 --ps-caustics')"
rs_s_n2="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --sampled --subsamples 2 --ps-caustics')"
rs_x="$(run_settings './PS-DTFE in.hdf5 out --grid 64 --field density_a --lambda-th 0.3 --exact --ps-caustics')"
check "$([ -n "${rs_s_new}" ] && [ "${rs_s_old}" = "${rs_s_new}" ] && [ "${rs_s_n3}" = "${rs_s_new}" ] && [ "${rs_s_n2}" != "${rs_s_new}" ] \
         && [ "${rs_x}" != "${rs_s_old}" ] && printf '%s\n' "${rs_s_new}" | grep -qx -e '--subsamples 3' \
         && ! printf '%s\n' "${rs_s_new}" | grep -qE -e '^--(exact|sampled)$' && echo 0 || echo 1)" \
      "run_settings: an older sampled log (a count, no flag) is '--sampled' with that count or none, not '--sampled --subsamples 2', not '--exact'"
# P6.7: every renamed option run_settings compares, old spelling against new (python/dtfelib/runlog.py's table),
# and the options it newly compares (--face-gradient, the Lagrangian padding, the alpha shape, the halo release)
rs7_old="$(run_settings 'RUNNING: ./PS-DTFE in.hdf5 out --input 105 --MpcUnit 1000 --grid 64 --avg-subsamples 3 --field density_a --lambda_th 0.3 --ps-lag-padding 8 --lagrangianInput ics.hdf5 --sample-points p.txt --pts-conventions legacy --ordered-merge --face-gradient --ps-alpha-shape 2.5 --ps-halo-release 500 --ps-window 0 1 0 1 0 1')"
rs7_new="$(run_settings './PS-DTFE in.hdf5 out --input-format 105 --mpc-unit 1000 --grid 64 --subsamples 3 --field density_a --lambda-th 0.3 --ps-lagrangian-padding 8 --ps-lagrangian-input ics.hdf5 --sample-points p.txt --ps-point-conventions legacy --ps-merge index-order --face-gradient --ps-alpha-shape 2.5 --ps-halo-release 500 --grid-window 0 1 0 1 0 1')"
rs7_ok=0
for o in '--subsamples 3' '--ps-lagrangian-padding 8' '--ps-point-conventions legacy' '--face-gradient' '--ps-alpha-shape 2.5' '--ps-halo-release 500'; do
    printf '%s\n' "${rs7_new}" | grep -qx -e "${o}" || rs7_ok=1
done
for o in 'face-gradient' 'ps-lagrangian-padding 9' 'ps-alpha-shape 3' 'ps-halo-release 400' 'subsamples 2'; do   # each one differs alone
    [ "$(run_settings "./PS-DTFE in out --grid 64 --${o}")" != "$(run_settings './PS-DTFE in out --grid 64')" ] || rs7_ok=1
done
check "$([ -n "${rs7_new}" ] && [ "${rs7_old}" = "${rs7_new}" ] && [ "${rs7_ok}" = 0 ] && echo 0 || echo 1)" \
      "run_settings (P6.7): --avg-subsamples 3 matches --subsamples 3, every renamed option reads the same, and the face gradient, Lagrangian padding, alpha shape and halo release count"
rc=$(pdrive "${TMP}/p_ex.log" DTFE_EXACT_AVERAGE=1)
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den" "${SD}/ps_output.pts_velGrad"
sed -e '/^RUNNING:/s/ --exact / --avg-subsamples 3 --ps-exact-deposit /' -e '/^RUNNING:/s/ --lambda-th / --lambda_th /' \
    "${SD}/ps_output.runlog" > "${SD}/ps_output.runlog.old" && mv "${SD}/ps_output.runlog.old" "${SD}/ps_output.runlog"
rc2=$(pdrive "${TMP}/p_old.log" PS_EXACT=1 DRY_RUN=1)     # (the knob's older name: the same settings)
rc3=$(pdrive "${TMP}/p_old_def.log" DRY_RUN=1)            # (the exact default: the same settings too)
check "$([ "${rc}" = 0 ] && [ "${rc2}" = 0 ] && [ "${rc3}" = 0 ] \
         && grep -aq '^RUNNING: .* --lambda_th 0.3 .*--avg-subsamples 3 --ps-exact-deposit ' "${SD}/ps_output.runlog" \
         && [ "$(lacks "${SD}/ps_output.runlog" '^RUNNING: .*--(exact|exact-average|lambda-th|sampled) ')" = 0 ] \
         && [ "$(has "${TMP}/p_old.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" = 0 ] \
         && [ "$(has "${TMP}/p_old_def.log" 'snap 99: ps_output.pts_den up to date \(same settings\), skipping compute')" = 0 ] \
         && echo 0 || echo 1)" "a run log with the old spellings (--avg-subsamples 3 --ps-exact-deposit --lambda_th) is up to date for a driver emitting the new ones, PS_EXACT=1 or the default (${rc}, ${rc2}, ${rc3})"
# the same log as an older SAMPLED run wrote it (the count, no deposit flag): current for a sampled pipeline run
# (DTFE_EXACT_AVERAGE=0 without a count, or SUBSAMPLES=3), stale for the exact default and for another count
sed -e '/^RUNNING:/s/ --ps-exact-deposit / /' "${SD}/ps_output.runlog" > "${SD}/ps_output.runlog.old" \
    && mv "${SD}/ps_output.runlog.old" "${SD}/ps_output.runlog"
rc=$(pdrive "${TMP}/p_os0.log" DTFE_EXACT_AVERAGE=0 DRY_RUN=1); rc2=$(pdrive "${TMP}/p_os3.log" SUBSAMPLES=3 DRY_RUN=1)
rc3=$(pdrive "${TMP}/p_osd.log" DRY_RUN=1); rc4=$(pdrive "${TMP}/p_os2.log" SUBSAMPLES=2 DRY_RUN=1)
check "$([ "$(lacks "${SD}/ps_output.runlog" '^RUNNING: .*--(exact|ps-exact-deposit|exact-average|sampled) ')" = 0 ] \
         && [ "$(has "${TMP}/p_os0.log" 'snap 99: ps_output.pts_den up to date \(same settings\)')" = 0 ] \
         && [ "$(has "${TMP}/p_os3.log" 'snap 99: ps_output.pts_den up to date \(same settings\)')" = 0 ] \
         && [ "$(has "${TMP}/p_osd.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'was: --subsamples 3$' "${TMP}/p_osd.log" && grep -aq 'now: --exact$' "${TMP}/p_osd.log" \
         && [ "$(has "${TMP}/p_os2.log" 'snap 99: settings differ from its ps_output.runlog, recomputing')" = 0 ] \
         && grep -aq 'now: --subsamples 2$' "${TMP}/p_os2.log" && echo 0 || echo 1)" \
      "an older sampled run's log (a count, no flag): up to date for DTFE_EXACT_AVERAGE=0 and SUBSAMPLES=3, recomputed for the exact default and SUBSAMPLES=2 (${rc} ${rc2} ${rc3} ${rc4})"
rc=$(pdrive "${TMP}/p_back.log")                 # a real run again (the exact default) for the checks below
head -c 512 /dev/zero > "${SD}/ps_output.pts_den"; touch -t 203001010000 "${SD}/ps_output.pts_den" "${SD}/ps_output.pts_velGrad"
mv "${SD}/ps_output.runlog" "${SD}/ps_output.runlog.keep"
rc=$(pdrive "${TMP}/p_nolog.log" GRID_SIZE=24 DRY_RUN=1)
check "$(has "${TMP}/p_nolog.log" 'up to date by file times \(no ps_output.runlog to compare the settings with\)')" \
      "no run log: judged by the file times, and said so"
mv "${SD}/ps_output.runlog.keep" "${SD}/ps_output.runlog"

echo ""
echo "(S) production defaults, OUTPUT_PREFIX and a rerun's left-over files."
# Against the stand-in binaries of (M) in a data folder of its own. The drivers' GPU default follows the Makefile's
# stamp in the pair's object folder (MREPO/o, o_ps, o_d), which the stand-in repository lacks until made here.
SDATA="${TMP}/sup_data"; SSD="${SDATA}/SIM/snapdir_099"; rm -rf "${SDATA}"; mkdir -p "${SSD}"
: > "${SSD}/combined_099.hdf5"; : > "${SDATA}/SIM/combined_ics.hdf5"
sdrive() {   # <driver> <log> [VAR=value ...]: the copied driver on SDATA's SIM 99; echoes its exit status
    local scr="$1" log="$2"; shift 2
    local rc=0
    env -u DTFE_GPU -u PS_GPU -u DTFE_METAL -u PS_METAL -u PS_EXACT -u DTFE_EXACT -u DTFE_EXACT_AVERAGE -u PS_VOLUME_WEIGHTED -u PS_CAUSTICS -u FIELDS -u OUTPUT_PREFIX \
        -u SUBSAMPLES -u AVG_SUBSAMPLES -u PS_MERGE -u ORDERED_MERGE -u FIXED_POINT_MERGE -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID \
        -u PS_LINEAR_DEPOSIT -u FACE_GRADIENT -u DTFE_PRECISION DTFE_DATA_ROOT="${SDATA}" "$@" "${MREPO}/scripts/${scr}" -s SIM -g 16 99 >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
sline() { grep -aE -e '^ +\./(PS-)?DTFE(-double)? ' "$1" | head -1; }      # the DRY RUN's planned command
gpu_of() {   # <driver> <stamp or none> [VAR=value ...]: 'gpu' / 'cpu' as the planned command says
    local scr="$1" st="$2"; shift 2
    rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
    if [ "${st}" != "none" ]; then mkdir -p "${MREPO}/${st%%/*}"; : > "${MREPO}/${st}"; fi
    sdrive "${scr}" "${TMP}/s_gpu.log" DRY_RUN=1 "$@" >/dev/null
    sline "${TMP}/s_gpu.log" | grep -qE -e ' --(ps-)?gpu( |$)' && echo gpu || echo cpu
}
g=""
# (the drivers' defaults: the exact average and deposit, the balanced run mode; the table by mode is below)
for c in "run_dtfe.sh o/.gpu_mode_metal" "run_dtfe.sh o/.gpu_mode_off" "run_dtfe.sh none" \
         "run_ps_dtfe.sh o_ps/.gpu_mode_metal" "run_ps_dtfe.sh o_ps/.gpu_mode_off" "run_ps_dtfe.sh o/.gpu_mode_metal"; do
    # shellcheck disable=SC2086
    g="${g} $(gpu_of ${c})"
done
check "$([ "${g}" = " gpu cpu cpu gpu cpu cpu" ] && echo 0 || echo 1)" \
      "the GPU by default exactly when the pair's object folder says metal (not off, none, or the other pair's) (${g# })"
g="$(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_GPU=0) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal DTFE_GPU=0)"
g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_off DTFE_GPU=1) $(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_PRECISION=double)"
g="${g} $(gpu_of run_dtfe.sh o_d/.gpu_mode_metal DTFE_PRECISION=double)"
check "$([ "${g}" = "cpu cpu gpu cpu gpu" ] && echo 0 || echo 1)" \
      "... DTFE_GPU = 0 or 1 wins over the stamp in both drivers; the double pair reads o_d (${g})"
check "$(has "${TMP}/s_gpu.log" "^'_a' interpolation: GPU \(--gpu\) \(DTFE_GPU unset: DTFE-double is a GPU build")" "... and the driver says which, and why"
g="$(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_METAL=0) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal PS_METAL=0)"
g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_off DTFE_METAL=1) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_off PS_METAL=1)"
g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_off DTFE_GPU=0 DTFE_METAL=1) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal DTFE_GPU=1 PS_GPU=0 PS_METAL=0)"
g="${g} $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_off PS_GPU=1) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal PS_GPU=0 PS_METAL=1)"
check "$([ "${g}" = "cpu cpu gpu gpu cpu gpu gpu cpu" ] && echo 0 || echo 1)" \
      "... the older names DTFE_METAL / PS_GPU / PS_METAL are read while DTFE_GPU is unset (PS_GPU before PS_METAL); DTFE_GPU wins over them (${g})"
# the device by run mode (config.sh gpu_auto_resolve; the launcher's recommended_gpu mirrors it), on GPU builds: per
# mode PS-DTFE's exact and sampled deposits, then standard DTFE's exact and sampled averages
g=""
for m in gentle balanced performance; do
    g="${g}${m}: $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal DTFE_RUN_MODE=${m})"
    g="${g} $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal DTFE_RUN_MODE=${m} DTFE_EXACT_AVERAGE=0)"
    g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_RUN_MODE=${m})"
    g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_RUN_MODE=${m} DTFE_EXACT_AVERAGE=0) | "
done
check "$([ "${g}" = "gentle: gpu gpu gpu gpu | balanced: gpu gpu gpu gpu | performance: gpu gpu gpu cpu | " ] && echo 0 || echo 1)" \
      "... by run mode (PS exact, PS sampled, standard exact, standard sampled): the GPU on a GPU build, but the CPU for standard DTFE's sampled average in performance, faster there with every core (${g})"
g="$(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal DTFE_EXACT_AVERAGE=1 DTFE_GPU=auto) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal PS_EXACT=1)"
g="${g} $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal PS_EXACT=1 PS_GPU=AUTO) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal DTFE_EXACT_AVERAGE=true)"
g="${g} $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal SUBSAMPLES=3 DTFE_EXACT_AVERAGE=1) $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_metal DTFE_GPU=0)"
g="${g} $(gpu_of run_ps_dtfe.sh o_ps/.gpu_mode_off DTFE_RUN_MODE=performance) $(gpu_of run_dtfe.sh o/.gpu_mode_off DTFE_RUN_MODE=gentle)"
g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_RUN_MODE=performance DTFE_EXACT_AVERAGE=0 DTFE_GPU=1)"
g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_metal SUBSAMPLES=3 DTFE_RUN_MODE=performance)"
g="${g} $(gpu_of run_dtfe.sh o/.gpu_mode_metal DTFE_EXACT_AVERAGE=0 SLURM_JOB_ID=1)"
check "$([ "${g}" = "gpu gpu gpu gpu gpu cpu cpu cpu gpu cpu cpu" ] \
         && grep -aqF "'_a' interpolation: CPU (DTFE_GPU unset: performance: standard DTFE's sampled average is faster on the CPU (315 s against 433 s, TNG100-3-Dark z=0 at 512^3); DTFE_GPU=1 runs it on the GPU)" "${TMP}/s_gpu.log" \
         && echo 0 || echo 1)" \
      "... the exact deposit (DTFE_EXACT_AVERAGE=1, PS_EXACT=1, =true, auto, the older names) on the GPU as the sampled one, DTFE_GPU=0 or a CPU-only build the CPU; standard DTFE's sampled average in performance (a count alone, or the batch-job default) the CPU, said, unless DTFE_GPU=1 (${g})"
rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_psdef.log" DRY_RUN=1)
check "$([ "${rc}" = "0" ] && sline "${TMP}/s_psdef.log" | grep -qE -e '--field +density_a +velocity_a +divergence_a +shear_a +--lambda-th +0\.3 +--exact +.*--ps-volume-weighted .*--ps-caustics' \
         && ! sline "${TMP}/s_psdef.log" | grep -q -e '--subsamples' && grep -aqxF 'Cell averages: exact deposit (--exact; the default)' "${TMP}/s_psdef.log" \
         && echo 0 || echo 1)" "run_ps_dtfe.sh's defaults are production's: the four analysis fields, the exact deposit (--exact, no count), volume-weighted moments, the caustic classes (${rc})"
# the sampled deposit: DTFE_EXACT_AVERAGE=0 names no count (the binary's 3), a SUBSAMPLES count alone selects it and is
# passed; -e (or DTFE_EXACT_AVERAGE=1) beside a count keeps the exact deposit, with one note
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_pssam.log" DRY_RUN=1 DTFE_EXACT_AVERAGE=0)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_pssub.log" DRY_RUN=1 SUBSAMPLES=2)
rc3=$(sdrive run_ps_dtfe.sh "${TMP}/s_psksub.log" DRY_RUN=1 SUBSAMPLES=2 DTFE_EXACT_AVERAGE=1)
check "$([ "${rc}${rc2}${rc3}" = "000" ] && sline "${TMP}/s_pssam.log" | grep -qE -e '--lambda-th +0\.3 +--sampled +--ps-vertex-mass ' \
         && grep -aqxF 'Cell averages: sampled deposit, nSub^3 sub-points per cell (--sampled; DTFE_EXACT_AVERAGE=0)' "${TMP}/s_pssam.log" \
         && sline "${TMP}/s_pssub.log" | grep -qE -e '--lambda-th +0\.3 +--sampled +--subsamples +2 +--ps-vertex-mass ' \
         && grep -aqxF 'Cell averages: sampled deposit, nSub^3 sub-points per cell (--sampled --subsamples 2; SUBSAMPLES=2)' "${TMP}/s_pssub.log" \
         && sline "${TMP}/s_psksub.log" | grep -qE -e '--lambda-th +0\.3 +--exact +--ps-vertex-mass ' \
         && [ "$(grep -c '^Note: SUBSAMPLES=2 is ignored: DTFE_EXACT_AVERAGE=1 asks for the exact deposit' "${TMP}/s_psksub.log")" = 1 ] \
         && echo 0 || echo 1)" \
      "... DTFE_EXACT_AVERAGE=0: --sampled (no count); SUBSAMPLES=2 alone: --sampled --subsamples 2; DTFE_EXACT_AVERAGE=1 beside it: --exact, the count said ignored (${rc} ${rc2} ${rc3})"
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_pslin.log" DRY_RUN=1 PS_LINEAR_DEPOSIT=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_pslin2.log" DRY_RUN=1 PS_LINEAR_DEPOSIT=1 PS_VOLUME_WEIGHTED=1)
check "$(sline "${TMP}/s_pslin.log" | grep -q -e '--ps-linear-deposit' && ! sline "${TMP}/s_pslin.log" | grep -q -e '--ps-volume-weighted' \
         && grep -aq 'velocity moments stay mass-weighted' "${TMP}/s_pslin.log" && sline "${TMP}/s_pslin2.log" | grep -q -e '--ps-volume-weighted' \
         && echo 0 || echo 1)" "... PS_LINEAR_DEPOSIT=1 drops the default volume weighting (said); an explicit PS_VOLUME_WEIGHTED=1 stays (${rc}, ${rc2})"
# FACE_GRADIENT=1: --face-gradient in both drivers; run_ps_dtfe.sh refuses it, exit 2 before any snapshot, with the
# sampled deposit (DTFE_EXACT_AVERAGE=0, or a count alone; the exact one is the default) or without the volume
# weighting; with the GPU deposit it passes both flags (since 2026-11-09, D17)
rc=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface.log" DRY_RUN=1 FACE_GRADIENT=1 DTFE_EXACT_AVERAGE=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface2.log" DRY_RUN=1 FACE_GRADIENT=1)
rc3=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface3.log" DRY_RUN=1 FACE_GRADIENT=1 DTFE_EXACT_AVERAGE=1 PS_VOLUME_WEIGHTED=0)
rc4=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface4.log" DRY_RUN=1 FACE_GRADIENT=1 DTFE_EXACT_AVERAGE=1 DTFE_GPU=1)
rc5=$(sdrive run_dtfe.sh "${TMP}/s_dtface.log" DRY_RUN=1 FACE_GRADIENT=1)
rc6=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface6.log" DRY_RUN=1 FACE_GRADIENT=1 DTFE_EXACT_AVERAGE=0)
rc7=$(sdrive run_ps_dtfe.sh "${TMP}/s_psface7.log" DRY_RUN=1 FACE_GRADIENT=1 SUBSAMPLES=2)
check "$([ "${rc}" = "0" ] && sline "${TMP}/s_psface.log" | grep -qE -e '--exact .*--ps-volume-weighted +--face-gradient' \
         && [ "${rc2}" = "0" ] && sline "${TMP}/s_psface2.log" | grep -qE -e '--exact .*--ps-volume-weighted +--face-gradient' \
         && [ "${rc3}" = "2" ] && grep -aq 'needs the volume-weighted' "${TMP}/s_psface3.log" \
         && [ "${rc4}" = "0" ] && sline "${TMP}/s_psface4.log" | grep -qE -e '--gpu +--exact .*--ps-volume-weighted +--face-gradient' \
         && [ "${rc5}" = "0" ] && sline "${TMP}/s_dtface.log" | grep -qE -e '--exact +--face-gradient$' \
         && [ "${rc6}" = "2" ] && grep -aqxF 'Error: FACE_GRADIENT=1 needs the exact deposit (the default): DTFE_EXACT_AVERAGE=0 asks for the sampled one.' "${TMP}/s_psface6.log" \
         && [ "${rc7}" = "2" ] && grep -aqxF 'Error: FACE_GRADIENT=1 needs the exact deposit (the default): SUBSAMPLES=2 asks for the sampled one.' "${TMP}/s_psface7.log" \
         && echo 0 || echo 1)" \
      "FACE_GRADIENT=1 passes --face-gradient in both drivers, alone (the exact deposit, the default) and with --gpu; PS with the sampled deposit (DTFE_EXACT_AVERAGE=0, SUBSAMPLES=2) or without the volume weighting: exit 2 (${rc} ${rc2} ${rc3} ${rc4} ${rc5} ${rc6} ${rc7})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_dtpertet.log" DRY_RUN=1 FACE_GRADIENT=0)
check "$([ "${rc}" = "0" ] && sline "${TMP}/s_dtpertet.log" | grep -qE -e '--exact +--per-tet-gradient$' && echo 0 || echo 1)" \
      "FACE_GRADIENT=0 passes --per-tet-gradient to run_dtfe.sh (the face gradient is the standard binary's default since 2026-11-09) (${rc})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_dtdef.log" DRY_RUN=1)
check "$([ "${rc}" = "0" ] && sline "${TMP}/s_dtdef.log" | grep -qE -e 'snapdir_099/output +--grid +16 +--padding +15 +--periodic +--run-mode +balanced +--field +density_a +velocity_a +divergence_a +shear_a +--lambda-th +0\.3 +--exact$' \
         && echo 0 || echo 1)" "run_dtfe.sh's defaults: padding 15, the balanced run mode, the same four fields as run_ps_dtfe.sh, the exact cell average (${rc})"
# OUTPUT_PREFIX for run_dtfe.sh: the files and the run log under it; a PS-DTFE-style prefix refused before anything runs
rc=$(sdrive run_dtfe.sh "${TMP}/s_pfx.log" OUTPUT_PREFIX=output_mc DTFE_EXACT_AVERAGE=0 FAKE_MODE=ok)
check "$([ "${rc}" = "0" ] && [ -f "${SSD}/output_mc.runlog" ] && [ ! -e "${SSD}/output.runlog" ] \
         && grep -aq '^RUNNING: .*snapdir_099/output_mc --grid' "${SSD}/output_mc.runlog" && echo 0 || echo 1)" \
      "run_dtfe.sh OUTPUT_PREFIX=output_mc: the binary's output and the run log are output_mc.* (${rc})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_pfxps.log" OUTPUT_PREFIX=ps_output FAKE_MODE=ok)
check "$([ "${rc}" = "2" ] && [ "$(has "${TMP}/s_pfxps.log" "is PS-DTFE's naming")" = 0 ] && [ "$(lacks "${TMP}/s_pfxps.log" '^RUNNING:')" = 0 ] \
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
check "$([ "${rc}" = "0" ] && [ -n "${SUP}" ] && [ "$(ls "${SUP}" | tr '\n' ' ')" = "output.a_tweb output.pts_den output.runlog " ] \
         && [ "$(cat "${SUP}/output.pts_den")" = old ] && grep -aq '^old log$' "${SUP}/output.runlog" \
         && [[ "$(basename "${SUP}")" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{6}$ ]] && echo 0 || echo 1)" \
      "a rerun that exits 0: the files it did not write (.pts_den, .a_tweb) MOVED to output.superseded/<date_time>/ with the old run log (${rc}; $(ls "${SUP}" 2>/dev/null | tr '\n' ' '))"
check "$([ "$(cat "${SSD}/output.a_den")" = new ] && [ "$(cat "${SSD}/output.a_vel")" = new ] && [ "$(cat "${SSD}/output.badGridPoints0")" = new ] \
         && grep -aq 'fake attempt' "${SSD}/output.runlog" && [ -f "${SSD}/output.gui.json" ] && [ -d "${SSD}/output.runlog.checkpoint" ] \
         && [ ! -e "${SSD}/output.pts_den" ] && [ ! -e "${SSD}/output.a_tweb" ] && [ ! -e "${SSD}/output.runlog.failed" ] && echo 0 || echo 1)" \
      "... the files it wrote stay (renamed into place, or rewritten in place like .badGridPoints0), the run log is the new one; the settings sidecar and a checkpoint are not touched"
check "$(has "${TMP}/s_sup.log" '2 file\(s\) of an earlier run with this prefix moved to .*output\.superseded/.* \(nothing deleted')" "... and the driver says so"
s_seed
rc=$(sdrive run_dtfe.sh "${TMP}/s_supfail.log" FAKE_MODE=fail)
check "$([ "${rc}" = "1" ] && [ ! -e "${SSD}/output.superseded" ] && [ "$(cat "${SSD}/output.pts_den")" = old ] && grep -aq '^old log$' "${SSD}/output.runlog" \
         && echo 0 || echo 1)" "a FAILED rerun moves nothing: the earlier run's set stays whole with its run log (${rc})"
rc=$(sdrive run_dtfe.sh "${TMP}/s_supdry.log" DRY_RUN=1)
rc2=$(sdrive run_ps_dtfe.sh "${TMP}/s_suprep.log" FAKE_MODE=report AUTO_TUNE_REPORT=1 OUTPUT_PREFIX=output)
rc3=$(sdrive run_dtfe.sh "${TMP}/s_suprep_std.log" FAKE_MODE=report AUTO_TUNE_REPORT=1)
check "$([ ! -e "${SSD}/output.superseded" ] && [ -f "${SSD}/output.pts_den" ] && [ -f "${SSD}/output.a_tweb" ] && echo 0 || echo 1)" \
      "... nor does a dry run or an auto-tune report (${rc}, ${rc2}, run_dtfe.sh ${rc3})"
rm -rf "${SSD}/output."* "${SSD}/output.superseded"
rc=$(sdrive run_dtfe.sh "${TMP}/s_supnone.log" FAKE_MODE=write)
check "$([ "${rc}" = "0" ] && [ ! -e "${SSD}/output.superseded" ] && [ "$(lacks "${TMP}/s_supnone.log" 'Superseded')" = 0 ] && echo 0 || echo 1)" \
      "... and a first run into an empty prefix makes no superseded folder (${rc})"
# run_ps_pipeline.sh's GPU default: DTFE_GPU unset follows the PS binary's stamp as run_ps_dtfe.sh does (config.sh
# gpu_auto_resolve, for both deposits in every run mode), so a CPU-only build is never asked for the GPU (the copy (P)
# made, on SDATA's SIM: a dry run plans the plane and the snapshot)
pgpu_of() {   # <stamp or none> [VAR=value ...]: 'gpu' / 'cpu' as the pipeline's planned run_ps_dtfe.sh call says
    local st="$1"; shift
    rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
    if [ "${st}" != "none" ]; then mkdir -p "${MREPO}/${st%%/*}"; : > "${MREPO}/${st}"; fi
    env -u DTFE_GPU -u PS_GPU -u PS_METAL -u DTFE_EXACT_AVERAGE -u PS_EXACT -u SUBSAMPLES -u AVG_SUBSAMPLES -u DTFE_PRECISION -u SIMS -u SNAPS \
        -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID DTFE_DATA_ROOT="${SDATA}" SIMS=SIM NU=8 DRY_RUN=1 "$@" \
        "${MREPO}/scripts/run_ps_pipeline.sh" >"${TMP}/s_pgpu.log" 2>&1 || true
    local line
    line="$(grep -aE -e 'run_ps_dtfe\.sh -s SIM ' "${TMP}/s_pgpu.log" | tail -1)"
    if [ -z "${line}" ]; then echo none
    elif [[ "${line}" == *" -m 99"* && "${line}" == *"DTFE_GPU=1 "* ]]; then echo gpu
    elif [[ "${line}" != *" -m "* && "${line}" == *"DTFE_GPU=0 "* ]]; then echo cpu
    else echo mixed; fi
}
g=""
for c in "o_ps/.gpu_mode_metal" "o_ps/.gpu_mode_off" "none" "o/.gpu_mode_metal"; do
    # shellcheck disable=SC2086
    g="${g} $(pgpu_of ${c} DTFE_EXACT_AVERAGE=0)"
done
check "$([ "${g}" = " gpu cpu cpu cpu" ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh, DTFE_GPU unset, the sampled deposit: the GPU (-m, DTFE_GPU=1 for run_ps_dtfe.sh) exactly when the PS binary's object folder says metal (not off, none, or the standard pair's) (${g# })"
g="$(pgpu_of o_ps/.gpu_mode_metal DTFE_GPU=0 DTFE_EXACT_AVERAGE=0) $(pgpu_of o_ps/.gpu_mode_off DTFE_GPU=1)"
g="${g} $(pgpu_of o_ps/.gpu_mode_metal DTFE_PRECISION=double DTFE_EXACT_AVERAGE=0) $(pgpu_of o_ps_d/.gpu_mode_metal DTFE_PRECISION=double DTFE_EXACT_AVERAGE=0)"
check "$([ "${g}" = "cpu gpu cpu gpu" ] && echo 0 || echo 1)" "... DTFE_GPU = 0 or 1 wins over the stamp; the double pair reads o_ps_d (${g})"
check "$(has "${TMP}/s_pgpu.log" 'GPU: 1 \(DTFE_GPU unset: PS-DTFE-double is a GPU build; balanced: the GPU; DTFE_GPU=0 or 1 chooses\)')" \
      "... and the header says which, and why (the run mode too)"
# the exact deposit, the default: the same rule, the GPU on a GPU build in every run mode (in performance with the free
# cores beside it, through run_ps_dtfe.sh's '--gpu-share auto')
g="$(pgpu_of o_ps/.gpu_mode_metal DTFE_EXACT_AVERAGE=1) $(pgpu_of o_ps/.gpu_mode_metal PS_EXACT=1)"
g="${g} $(pgpu_of o_ps/.gpu_mode_metal DTFE_EXACT_AVERAGE=yes DTFE_RUN_MODE=gentle) $(pgpu_of o_ps/.gpu_mode_metal SUBSAMPLES=3)"
g="${g} $(pgpu_of o_ps/.gpu_mode_metal DTFE_EXACT_AVERAGE=1 DTFE_GPU=0) $(pgpu_of o_ps/.gpu_mode_off DTFE_RUN_MODE=performance)"
g="${g} $(pgpu_of o_ps/.gpu_mode_metal DTFE_RUN_MODE=performance)"
check "$([ "${g}" = "gpu gpu gpu gpu cpu cpu gpu" ] \
         && grep -aqF 'GPU: 1 (DTFE_GPU unset: PS-DTFE is a GPU build; performance: the exact deposit on the GPU and the free cores (--gpu-share auto); DTFE_GPU=0 or 1 chooses)' "${TMP}/s_pgpu.log" \
         && echo 0 || echo 1)" \
      "... the exact deposit (DTFE_EXACT_AVERAGE=1, the older PS_EXACT=1, =yes, or nothing: the default; the driver's own rule): the GPU on a GPU build in every run mode, as the sampled one (a SUBSAMPLES count); DTFE_GPU=0 or a CPU-only build: the CPU; performance says the free cores join in (${g})"
g="$(pgpu_of o_ps/.gpu_mode_metal PS_METAL=0) $(pgpu_of o_ps/.gpu_mode_off PS_METAL=1) $(pgpu_of o_ps/.gpu_mode_metal PS_EXACT=1 PS_METAL=1)"
g="${g} $(pgpu_of o_ps/.gpu_mode_off PS_GPU=1)"
check "$([ "${g}" = "cpu gpu gpu gpu" ] && echo 0 || echo 1)" "... PS_GPU and PS_METAL, the older names, are read while DTFE_GPU is unset (${g})"
rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"

echo ""
echo "(T) the knobs' older names (config.sh knob, P6.7): the same command as the new name, one warning; the new one wins."
TCD="${TMP}/t cache"
t_pair() {   # <driver> <old=value> <new=value> [VAR=value ...]: 0 when the old name plans the new name's command,
             # with exactly one 'warning: OLD is deprecated, use NEW' line, and the new name warns of nothing
    local scr="$1" old="$2" new="$3"; shift 3
    sdrive "${scr}" "${TMP}/t_old.log" DRY_RUN=1 "${old}" "$@" >/dev/null
    sdrive "${scr}" "${TMP}/t_new.log" DRY_RUN=1 "${new}" "$@" >/dev/null
    [ -n "$(sline "${TMP}/t_new.log")" ] && [ "$(sline "${TMP}/t_old.log")" = "$(sline "${TMP}/t_new.log")" ] \
        && [ "$(grep -c "^warning: ${old%%=*} is deprecated, use ${new%%=*}$" "${TMP}/t_old.log")" = 1 ] \
        && [ "$(grep -c 'is deprecated' "${TMP}/t_old.log")" = 1 ] && [ "$(lacks "${TMP}/t_new.log" 'is deprecated')" = 0 ]
}
t_bad=""
for c in "run_ps_dtfe.sh AVG_SUBSAMPLES=2 SUBSAMPLES=2" "run_ps_dtfe.sh PS_PARALLEL_TRI=1 PARALLEL_TRIANGULATION=1" \
         "run_ps_dtfe.sh PS_LAG_PADDING=4 PS_LAGRANGIAN_PADDING=4" "run_ps_dtfe.sh PS_EXACT=1 DTFE_EXACT_AVERAGE=1" \
         "run_ps_dtfe.sh PS_GPU=1 DTFE_GPU=1" "run_ps_dtfe.sh PS_METAL=1 DTFE_GPU=1" "run_ps_dtfe.sh DTFE_PADDING=7 PADDING=7" \
         "run_ps_dtfe.sh LAGRANGIAN_INPUT=${SDATA}/SIM/combined_ics.hdf5 PS_LAGRANGIAN_INPUT=${SDATA}/SIM/combined_ics.hdf5" \
         "run_dtfe.sh DTFE_METAL=1 DTFE_GPU=1" "run_dtfe.sh DTFE_EXACT=0 DTFE_EXACT_AVERAGE=0" \
         "run_dtfe.sh DTFE_PADDING=7 PADDING=7"; do
    # shellcheck disable=SC2086
    t_pair ${c} || t_bad="${t_bad} [${c}]"
done
t_pair run_ps_dtfe.sh "TESS_CACHE=${TCD}" "TESSELLATION_CACHE=${TCD}" || t_bad="${t_bad} [TESS_CACHE ps]"
t_pair run_dtfe.sh "TESS_CACHE=${TCD}" "TESSELLATION_CACHE=${TCD}" || t_bad="${t_bad} [TESS_CACHE std]"
: > "${TMP}/t_points.txt"
t_pair run_ps_dtfe.sh PTS_CONVENTIONS=legacy PS_POINT_CONVENTIONS=legacy SAMPLE_POINTS="${TMP}/t_points.txt" || t_bad="${t_bad} [PTS_CONVENTIONS]"
check "$([ -z "${t_bad}" ] && echo 0 || echo 1)" \
      "each older knob alone plans the same command as its new name, with one deprecation line (the new name: none)${t_bad:+ -- differ:${t_bad}}"
# both set to different values: the new one wins, with a warning, and nothing is refused (a stray export or the
# launcher's older names beside the new ones must not stop a run during the window)
rc=$(sdrive run_ps_dtfe.sh "${TMP}/t_both.log" DRY_RUN=1 SUBSAMPLES=2 AVG_SUBSAMPLES=4 DTFE_EXACT_AVERAGE=0 PS_EXACT=1)
check "$([ "${rc}" = 0 ] && sline "${TMP}/t_both.log" | grep -qE -e '--sampled +--subsamples +2 ' && ! sline "${TMP}/t_both.log" | grep -qE -e '--exact( |$)' \
         && grep -aqx 'warning: AVG_SUBSAMPLES is deprecated, use SUBSAMPLES (AVG_SUBSAMPLES=4 is not read: SUBSAMPLES=2 wins)' "${TMP}/t_both.log" \
         && grep -aqx 'warning: PS_EXACT is deprecated, use DTFE_EXACT_AVERAGE (PS_EXACT=1 is not read: DTFE_EXACT_AVERAGE=0 wins)' "${TMP}/t_both.log" \
         && echo 0 || echo 1)" "both names set apart: the new one wins and the warning says so, exit 0 (${rc})"
# config.sh's own knobs, sourced alone
t_cfg="$(env -u GRID_SIZE DTFE_GRID_SIZE=64 bash -c 'source scripts/config.sh; echo "${GRID_SIZE} ${CALLER_GRID_SIZE}"' 2>"${TMP}/t_cfg.err")"
check "$([ "${t_cfg}" = "64 64" ] && grep -qx 'warning: DTFE_GRID_SIZE is deprecated, use GRID_SIZE' "${TMP}/t_cfg.err" && echo 0 || echo 1)" \
      "config.sh: DTFE_GRID_SIZE, the older name, is GRID_SIZE (with a warning) (${t_cfg})"
# a switch reads like the binaries' (src/env_knobs.h): 0/false/no/off (any case) off, any other value on
sdrive run_ps_dtfe.sh "${TMP}/t_on.log" DRY_RUN=1 PS_CAUSTICS=1 PS_VERTEX_MASS=1 PS_CAUSTIC_CUSPS=1 >/dev/null
sdrive run_ps_dtfe.sh "${TMP}/t_true.log" DRY_RUN=1 PS_CAUSTICS=true PS_VERTEX_MASS=YES PS_CAUSTIC_CUSPS=on >/dev/null
sdrive run_ps_dtfe.sh "${TMP}/t_off.log" DRY_RUN=1 PS_CAUSTICS=0 PS_VERTEX_MASS=0 >/dev/null
sdrive run_ps_dtfe.sh "${TMP}/t_false.log" DRY_RUN=1 PS_CAUSTICS=Off PS_VERTEX_MASS=false >/dev/null
t_on="$(sline "${TMP}/t_on.log")"; t_true="$(sline "${TMP}/t_true.log")"
t_off="$(sline "${TMP}/t_off.log")"; t_false="$(sline "${TMP}/t_false.log")"
check "$([ -n "${t_on}" ] && [ "${t_on}" = "${t_true}" ] && [ -n "${t_off}" ] && [ "${t_off}" = "${t_false}" ] \
         && printf '%s\n' "${t_true}" | grep -q -e '--ps-caustic-cusps' && ! printf '%s\n' "${t_false}" | grep -qE -e '--ps-(caustics|vertex-mass)' \
         && echo 0 || echo 1)" "a switch set to true / YES / on is on, to Off / false off, as in the binaries (no longer a silent off for anything but 1)"
rm -rf "${TCD}" "${TMP}/t_points.txt"

echo ""
echo "(U) the run mode: --run-mode, DTFE_RUN_MODE and the batch-job default; GPU_SHARE beside the exact GPU deposit."
# Dry runs of the copied drivers beside the stand-in binaries of (M), on SDATA's SIM 99, the stamps made as (S) makes
# them: the drivers resolve one mode (config.sh run_mode_resolve) and pass it on every run as '--run-mode MODE'
rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
udrive() {   # <driver> <log> [VAR=value ...] [-- driver options]: a dry run of the copied driver; echoes its exit status
    local scr="$1" log="$2" e=() rc=0
    shift 2
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do e+=("$1"); shift; done
    [ "$#" -gt 0 ] && shift
    env -u DTFE_GPU -u PS_GPU -u DTFE_METAL -u PS_METAL -u DTFE_EXACT_AVERAGE -u PS_EXACT -u DTFE_EXACT -u SUBSAMPLES \
        -u AVG_SUBSAMPLES -u DTFE_PRECISION -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID -u OMP_NUM_THREADS \
        -u THREADS DTFE_DATA_ROOT="${SDATA}" DRY_RUN=1 ${e[@]+"${e[@]}"} "${MREPO}/scripts/${scr}" -s SIM -g 16 "$@" 99 \
        >"${log}" 2>&1 || rc=$?
    echo "${rc}"
}
umode() {    # <driver> [VAR=value ...] [-- driver options]: '<the planned --run-mode> [<the banner's run mode line>]'
    local scr="$1"; shift
    udrive "${scr}" "${TMP}/u_mode.log" "$@" >/dev/null
    printf '%s [%s]' "$(sline "${TMP}/u_mode.log" | grep -oE -e '--run-mode +[a-z]+' | sed -e 's/^--run-mode *//')" \
        "$(sed -n 's/^Run mode: //p' "${TMP}/u_mode.log")"
}
rc=$(udrive run_dtfe.sh "${TMP}/u_std.log"); rc2=$(udrive run_ps_dtfe.sh "${TMP}/u_ps.log")
check "$([ "${rc}${rc2}" = "00" ] && sline "${TMP}/u_std.log" | grep -qE -e ' --periodic +--run-mode +balanced +--field ' \
         && sline "${TMP}/u_ps.log" | grep -qE -e ' --ps-merge +index-order +--run-mode +balanced +--input-format ' \
         && grep -aqxF 'Run mode: balanced (the default)' "${TMP}/u_std.log" && grep -aqxF 'Run mode: balanced (the default)' "${TMP}/u_ps.log" \
         && echo 0 || echo 1)" "both drivers pass '--run-mode balanced' by default, and the banner says why (${rc} ${rc2})"
u_bad=""
for scr in run_dtfe.sh run_ps_dtfe.sh; do
    for c in "balanced [balanced (the default)]|" "gentle [gentle (--run-mode)]|-- --run-mode gentle" \
             "performance [performance (--run-mode)]|-- --run-mode=performance" \
             "gentle [gentle (DTFE_RUN_MODE=GENTLE)]|DTFE_RUN_MODE=GENTLE" "balanced [balanced (the default)]|DTFE_RUN_MODE=auto" \
             "balanced [balanced (the default)]|DTFE_RUN_MODE=" "performance [performance (--run-mode)]|DTFE_RUN_MODE=gentle -- --run-mode performance" \
             "gentle [gentle (DTFE_RUN_MODE=gentle)]|DTFE_RUN_MODE=gentle -- --run-mode auto" \
             "gentle [gentle (--run-mode)]|DTFE_RUN_MODE=fast -- --run-mode gentle" \
             "performance [performance (the default inside a batch job (SLURM_JOB_ID))]|SLURM_JOB_ID=1" \
             "performance [performance (the default inside a batch job (PBS_JOBID))]|PBS_JOBID=1" \
             "balanced [balanced (DTFE_RUN_MODE=balanced)]|SLURM_JOB_ID=1 DTFE_RUN_MODE=balanced" \
             "balanced [balanced (the default; OMP_NUM_THREADS=3 keeps its thread count)]|THREADS=3"; do
        # shellcheck disable=SC2086
        got="$(umode "${scr}" ${c#*|})"
        [ "${got}" = "${c%%|*}" ] || u_bad="${u_bad} [${scr} ${c#*|}: ${got}]"
    done
done
check "$([ -z "${u_bad}" ] && echo 0 || echo 1)" \
      "--run-mode MODE (or =MODE) first (DTFE_RUN_MODE beside it is not read, a bad one neither), then DTFE_RUN_MODE in any case, then the default: balanced, performance inside a SLURM or PBS job; auto (or empty) chooses nothing; the banner names the source and a kept OMP_NUM_THREADS${u_bad:+ -- differ:${u_bad}}"
rc=$(udrive run_ps_dtfe.sh "${TMP}/u_badopt.log" -- --run-mode fast); rc2=$(udrive run_dtfe.sh "${TMP}/u_badopt2.log" -- --run-mode GENTLE)
check "$([ "${rc}" = "1" ] && grep -aqxF "Error: --run-mode must be 'gentle', 'balanced', 'performance' or 'auto' (got 'fast')." "${TMP}/u_badopt.log" \
         && [ "${rc2}" = "1" ] && grep -aqxF "Error: --run-mode must be 'gentle', 'balanced', 'performance' or 'auto' (got 'GENTLE')." "${TMP}/u_badopt2.log" \
         && [ "$(has "${TMP}/u_badopt.log" '^Usage: ')" = 0 ] && [ "$(lacks "${TMP}/u_badopt.log" 'would run:')" = 0 ] && echo 0 || echo 1)" \
      "... another --run-mode word (or a capital one: the option is lower case, as the binaries take it) is refused with the usage, exit 1, nothing planned (${rc} ${rc2})"
rc=$(udrive run_ps_dtfe.sh "${TMP}/u_badknob.log" DTFE_RUN_MODE=fast); rc2=$(udrive run_dtfe.sh "${TMP}/u_badknob2.log" DTFE_RUN_MODE=fast)
check "$([ "${rc}" = "2" ] && grep -aqxF "Error: DTFE_RUN_MODE must be 'gentle', 'balanced', 'performance' or 'auto' (got 'fast')." "${TMP}/u_badknob.log" \
         && [ "${rc2}" = "2" ] && grep -aqxF "Error: DTFE_RUN_MODE must be 'gentle', 'balanced', 'performance' or 'auto' (got 'fast')." "${TMP}/u_badknob2.log" \
         && [ "$(lacks "${TMP}/u_badknob.log" 'would run:')" = 0 ] && echo 0 || echo 1)" \
      "... another DTFE_RUN_MODE value is refused before anything runs, exit 2 (${rc} ${rc2})"
# the GPU share: '--gpu-share GPU_SHARE' (auto by default) beside the exact deposit on the GPU; a share given for
# another run is passed too (the binary says why it does not apply), auto is not
share_of() { sline "$1" | grep -oE -e '--gpu-share +[^ ]+' | sed -e 's/^--gpu-share *//' || true; }   # (none: empty)
mkdir -p "${MREPO}/o_ps"; : > "${MREPO}/o_ps/.gpu_mode_metal"
udrive run_ps_dtfe.sh "${TMP}/u_sh_def.log" >/dev/null; udrive run_ps_dtfe.sh "${TMP}/u_sh_half.log" GPU_SHARE=0.5 >/dev/null
udrive run_ps_dtfe.sh "${TMP}/u_sh_perf.log" DTFE_RUN_MODE=performance >/dev/null
udrive run_ps_dtfe.sh "${TMP}/u_sh_cpu.log" DTFE_GPU=0 >/dev/null; udrive run_ps_dtfe.sh "${TMP}/u_sh_cpuh.log" DTFE_GPU=0 GPU_SHARE=0.5 >/dev/null
udrive run_ps_dtfe.sh "${TMP}/u_sh_sam.log" DTFE_EXACT_AVERAGE=0 >/dev/null
u_sh="$(share_of "${TMP}/u_sh_def.log")|$(share_of "${TMP}/u_sh_half.log")|$(share_of "${TMP}/u_sh_perf.log")"
u_sh="${u_sh}|$(share_of "${TMP}/u_sh_cpu.log")|$(share_of "${TMP}/u_sh_cpuh.log")|$(share_of "${TMP}/u_sh_sam.log")"
check "$([ "${u_sh}" = "auto|0.5|auto||0.5|" ] && sline "${TMP}/u_sh_def.log" | grep -qE -e '--gpu +--exact( |$)' \
         && ! sline "${TMP}/u_sh_cpu.log" | grep -qE -e ' --gpu( |$)' \
         && grep -aqF 'Deposit: GPU (--gpu) (DTFE_GPU unset: PS-DTFE is a GPU build; performance: the exact deposit on the GPU and the free cores (--gpu-share auto); DTFE_GPU=0 or 1 chooses)' "${TMP}/u_sh_perf.log" \
         && echo 0 || echo 1)" \
      "run_ps_dtfe.sh on a GPU build: the exact deposit on the GPU with '--gpu-share auto' (GPU_SHARE=0.5: 0.5; performance says the free cores join in); DTFE_GPU=0 and the sampled deposit pass none, a given share still (${u_sh})"
# standard DTFE's sampled average in performance on a GPU build: the CPU (faster there), said
mkdir -p "${MREPO}/o"; : > "${MREPO}/o/.gpu_mode_metal"
rc=$(udrive run_dtfe.sh "${TMP}/u_std_perf.log" DTFE_RUN_MODE=performance DTFE_EXACT_AVERAGE=0)
check "$([ "${rc}" = "0" ] && ! sline "${TMP}/u_std_perf.log" | grep -qE -e ' --gpu( |$)' && sline "${TMP}/u_std_perf.log" | grep -qE -e '--run-mode +performance ' \
         && grep -aqxF "'_a' interpolation: CPU (DTFE_GPU unset: performance: standard DTFE's sampled average is faster on the CPU (315 s against 433 s, TNG100-3-Dark z=0 at 512^3); DTFE_GPU=1 runs it on the GPU)" "${TMP}/u_std_perf.log" \
         && echo 0 || echo 1)" "run_dtfe.sh's sampled average in performance on a GPU build: no --gpu, and the banner says why (${rc})"
# the pipeline resolves the mode once and hands it on after the deposit; the exact deposit goes to the GPU (-m)
upipe() {    # <log> [VAR=value ...]: the pipeline's planned run_ps_dtfe.sh call (the copy (P) made, on SDATA's SIM)
    local log="$1"; shift
    env -u DTFE_GPU -u PS_GPU -u PS_METAL -u DTFE_EXACT_AVERAGE -u PS_EXACT -u SUBSAMPLES -u AVG_SUBSAMPLES -u DTFE_PRECISION \
        -u SIMS -u SNAPS -u DTFE_RUN_MODE -u GPU_SHARE -u SLURM_JOB_ID -u PBS_JOBID \
        DTFE_DATA_ROOT="${SDATA}" SIMS=SIM NU=8 DRY_RUN=1 "$@" "${MREPO}/scripts/run_ps_pipeline.sh" >"${log}" 2>&1 || true
    grep -aE -e 'run_ps_dtfe\.sh -s SIM ' "${log}" | tail -1 || true
}
u_p1="$(upipe "${TMP}/u_pipe1.log")"; u_p2="$(upipe "${TMP}/u_pipe2.log" DTFE_RUN_MODE=GENTLE)"
u_p3="$(upipe "${TMP}/u_pipe3.log" SLURM_JOB_ID=1)"
rc=0; env -u SIMS DTFE_DATA_ROOT="${SDATA}" SIMS=SIM NU=8 DRY_RUN=1 DTFE_RUN_MODE=fast "${MREPO}/scripts/run_ps_pipeline.sh" \
    >"${TMP}/u_pipe_bad.log" 2>&1 || rc=$?
check "$([[ "${u_p1}" == *" DTFE_GPU=1 DTFE_EXACT_AVERAGE=1 DTFE_RUN_MODE=balanced "*"run_ps_dtfe.sh -s SIM "*" -m 99" ]] \
         && grep -aqxF 'Run mode: balanced (the default)' "${TMP}/u_pipe1.log" \
         && [[ "${u_p2}" == *" DTFE_EXACT_AVERAGE=1 DTFE_RUN_MODE=gentle "* ]] && grep -aqxF 'Run mode: gentle (DTFE_RUN_MODE=GENTLE)' "${TMP}/u_pipe2.log" \
         && [[ "${u_p3}" == *" DTFE_EXACT_AVERAGE=1 DTFE_RUN_MODE=performance "* ]] \
         && grep -aqxF 'Run mode: performance (the default inside a batch job (SLURM_JOB_ID))' "${TMP}/u_pipe3.log" \
         && [ "${rc}" = "2" ] && [ "$(has "${TMP}/u_pipe_bad.log" "^Error: DTFE_RUN_MODE must be ")" = 0 ] && echo 0 || echo 1)" \
      "run_ps_pipeline.sh resolves the mode once (said in its header) and hands DTFE_RUN_MODE=<mode> on after DTFE_EXACT_AVERAGE, the exact deposit on the GPU (-m); a bad knob: exit 2 (${rc})"
rm -rf "${MREPO}/o" "${MREPO}/o_ps" "${MREPO}/o_d" "${MREPO}/o_ps_d"
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
check "$([[ "${s_rel}" =~ ^\.flags_release_[0-9]+$ ]] && [[ "${s_dbg}" =~ ^\.flags_debug_[0-9]+$ ]] \
         && [ "${s_rel#.flags_release_}" != "${s_dbg#.flags_debug_}" ] && echo 0 || echo 1)" \
      "BUILD_MODE=debug and the default release print different stamps (${s_rel} / ${s_dbg})"
check "$([[ "${s_ext}" =~ ^\.flags_release_[0-9]+$ ]] && [[ "${s_arch}" =~ ^\.flags_release_[0-9]+$ ]] \
         && [ "${s_ext}" != "${s_rel}" ] && [ "${s_arch}" != "${s_rel}" ] && echo 0 || echo 1)" \
      "... EXTRA_FLAGS and ARCH_FLAGS change the checksum too (${s_ext} / ${s_arch})"
check "$([[ "${s_quote}" =~ ^\.flags_release_[0-9]+$ ]] && echo 0 || echo 1)" "... and a quote in EXTRA_FLAGS still gives a stamp (${s_quote:-none})"
OBJ_PS=o_ps; precision_double && OBJ_PS=o_ps_d
if [ -d "${OBJ_PS}" ]; then
    stamps="$(ls -A "${OBJ_PS}" | grep -E '^\.flags_' || true)"
    check "$([ "$(printf '%s\n' "${stamps}" | grep -cE '^\.flags_[a-z]+_[0-9]+$')" = "1" ] && [ "$(printf '%s\n' "${stamps}" | grep -c .)" = "1" ] \
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
