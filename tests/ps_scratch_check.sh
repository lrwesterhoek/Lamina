#!/usr/bin/env bash
# --scratch-dir (out-of-core mmap accumulators, src/scratch_alloc.cc): outputs must be bit-identical to an in-RAM run.
# DTFE_SCRATCH_MIN_GB is set tiny so small grids use the route. Portable to Linux; B (GPU) SKIPs on a CPU build.
#   C no leftover files in the scratch folder   D an iCloud path and a missing folder are rejected   S staged-merge spill
# Usage: tests/ps_scratch_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"   # PY: the python with numpy and h5py
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-64}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
SNAP="${TMP}/scr_input_pancake.hdf5"
# a LOCAL temp folder on either system (not /private/tmp: macOS only). TMPDIR on a Mac is
# /var/folders/..., local and never under iCloud, which the binary would refuse (section D)
SC="$(mktemp -d "${TMPDIR:-/tmp}/dtfe-scratch-check.XXXXXX")"
trap 'rm -rf "${SC}"' EXIT

echo "============================================================"
echo " --scratch-dir check   N=${N}^3   scratch=${SC}"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
# GPU deposit compiled in: the build mode names METAL=1 (the mere absence of o_ps/.gpu_mode_off would also call a
# never-stamped tree "GPU")
GPU_BUILT=0
grep -q "METAL=1" o_ps/.build_mode 2>/dev/null && GPU_BUILT=1

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

FAILS=0
# check_ok <label> <1 = ok> (it counts a failure in FAILS): tests/lib.sh

BASEARGS=(--grid 128 --field density_a velocity_a dispersion_a --input-format 105 --mpc-unit 1
          --verbose 2 --periodic --subsamples 1)

compare_outputs() {  # $1 = ref root, $2 = scratch root, $3 = label
    local allsame=1
    for f in "$1".*; do
        local e="${f#$1}"
        [ "${e}" = ".log" ] && continue
        cmp -s "${f}" "$2${e}" || { allsame=0; echo "        differs: ${e}"; }
    done
    check_ok "$3 outputs bit-identical to the in-RAM run" "${allsame}"
}

# ---------- (A) partitioned (deterministic mc=1) and unpartitioned bit-identity ----------
echo ">> A: bit-identity, partitioned (mc=1) + unpartitioned"
"${BIN}" "${SNAP}" "${TMP}/scr_p_ref" "${BASEARGS[@]}" --partition 2 2 2 --max-concurrent 1 \
    > "${TMP}/scr_p_ref.log" 2>&1
DTFE_SCRATCH_MIN_GB=0.0001 "${BIN}" "${SNAP}" "${TMP}/scr_p_mm" "${BASEARGS[@]}" \
    --partition 2 2 2 --max-concurrent 1 --scratch-dir "${SC}" > "${TMP}/scr_p_mm.log" 2>&1 \
    && check_ok "A scratch partitioned run exits 0" 1 || check_ok "A scratch partitioned run exits 0" 0
compare_outputs "${TMP}/scr_p_ref" "${TMP}/scr_p_mm" "A partitioned"

"${BIN}" "${SNAP}" "${TMP}/scr_u_ref" "${BASEARGS[@]}" > "${TMP}/scr_u_ref.log" 2>&1
DTFE_SCRATCH_MIN_GB=0.0001 "${BIN}" "${SNAP}" "${TMP}/scr_u_mm" "${BASEARGS[@]}" \
    --scratch-dir "${SC}" > "${TMP}/scr_u_mm.log" 2>&1
compare_outputs "${TMP}/scr_u_ref" "${TMP}/scr_u_mm" "A unpartitioned"
if grep -q "SCRATCH:" "${TMP}/scr_p_mm.log"; then check_ok "A arming banner printed" 1; else check_ok "A arming banner printed" 0; fi
# the zero-fill route (ScratchAlloc::ZeroFillScope): on macOS every grid is zero-filled then moved onto its scratch file,
# so N > 0 here (the printed count is the only proof); elsewhere N = 0, and the unpartitioned run prints no count.
NZF="$(sed -n 's/^SCRATCH: .*; \([0-9][0-9]*\) zero-filled in memory.*/\1/p' "${TMP}/scr_p_mm.log" | head -1)"
if [ "$(uname -s)" = Darwin ]; then
    check_ok "A ${NZF:-no} grid(s) zero-filled in memory and moved onto their scratch file (the zero-fill route)" \
        $([ "${NZF:-0}" -gt 0 ] && echo 1 || echo 0)
else
    check_ok "A the zero-fill route is off here (macOS only): ${NZF:-no} grid(s) moved, the count printed" \
        $([ "${NZF:-x}" = 0 ] && echo 1 || echo 0)
fi

# ---------- (B) GPU parity path: exact fixed-point sums, merged in index order, so the scratch run ----------
# equals the plain --gpu run byte for byte
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> B: GPU deposit path (exact fixed-point sums: bit-identity)"
    "${BIN}" "${SNAP}" "${TMP}/scr_g_ref" "${BASEARGS[@]}" --gpu --partition 2 2 2 \
        --max-concurrent 1 > "${TMP}/scr_g_ref.log" 2>&1
    DTFE_SCRATCH_MIN_GB=0.0001 "${BIN}" "${SNAP}" "${TMP}/scr_g_mm" "${BASEARGS[@]}" --gpu \
        --partition 2 2 2 --max-concurrent 1 --scratch-dir "${SC}" > "${TMP}/scr_g_mm.log" 2>&1
    # a CPU fallback in one run of the pair (GPU contention) breaks the identity for another reason: say so
    if grep -q "using the CPU deposit\|deposit unavailable\|ran on the CPU instead" "${TMP}/scr_g_ref.log" "${TMP}/scr_g_mm.log"; then
        echo "        (a GPU run fell back to the CPU: see ${TMP}/scr_g_ref.log / scr_g_mm.log)"
        check_ok "B GPU runs stayed on the GPU" 0
    else
        compare_outputs "${TMP}/scr_g_ref" "${TMP}/scr_g_mm" "B GPU scratch run:"
    fi
else
    echo "   SKIP  B (this build has no GPU backend)"
fi

# ---------- (S) the staged ordered merge's spill into the scratch folder ----------
# four partitions in flight, partition 0 held back, parked ones SPILLED (DTFE_MERGE_SPILL=1); byte-identical to A's mc=1 run
echo ">> S: the ordered merge's spill to the scratch folder (4 in flight, mmap'ed sub-grids)"
DTFE_MERGE_SPILL=1 DTFE_TEST_DELAY_PARTITION=0:2 DTFE_SCRATCH_MIN_GB=0.0001 OMP_NUM_THREADS=4 "${BIN}" "${SNAP}" \
    "${TMP}/scr_s_mm" "${BASEARGS[@]}" --partition 2 2 2 --max-concurrent 4 --scratch-dir "${SC}" \
    > "${TMP}/scr_s_mm.log" 2>&1 && check_ok "S spilling scratch run exits 0" 1 || check_ok "S spilling scratch run exits 0" 0
NSP="$(sed -n 's/.*ordered merge: .* \([0-9]*\) spilled to disk.*/\1/p' "${TMP}/scr_s_mm.log")"
check_ok "S ${NSP:-no} partition(s) spilled to the scratch folder" $([ "${NSP:-0}" -gt 0 ] && echo 1 || echo 0)
compare_outputs "${TMP}/scr_p_ref" "${TMP}/scr_s_mm" "S spilled"

# ---------- (C) nothing left behind ----------
NLEFT=$(ls "${SC}" 2>/dev/null | wc -l | tr -d ' ')
check_ok "C scratch dir empty after runs (unlink-at-creation)" $([ "${NLEFT}" -eq 0 ] && echo 1 || echo 0)

# ---------- (D) guards ----------
echo ">> D: path guards"
ICLOUD_DIR="${SC}/Mobile Documents/fake"
mkdir -p "${ICLOUD_DIR}"
set +e
"${BIN}" "${SNAP}" "${TMP}/scr_x" "${BASEARGS[@]}" --scratch-dir "${ICLOUD_DIR}" > "${TMP}/scr_x.log" 2>&1
RC_ICLOUD=$?
"${BIN}" "${SNAP}" "${TMP}/scr_y" "${BASEARGS[@]}" --scratch-dir /nonexistent/dtfe/path > "${TMP}/scr_y.log" 2>&1
RC_NODIR=$?
set -e
check_ok "D iCloud path rejected (nonzero exit)" $([ "${RC_ICLOUD}" -ne 0 ] && echo 1 || echo 0)
if grep -q "iCloud" "${TMP}/scr_x.log"; then check_ok "D iCloud rejection names the reason" 1; else check_ok "D iCloud rejection names the reason" 0; fi
check_ok "D nonexistent dir rejected (nonzero exit)" $([ "${RC_NODIR}" -ne 0 ] && echo 1 || echo 0)

echo "------------------------------------------------------------"
if [ "${FAILS}" -gt 0 ]; then
    echo "RESULT: FAIL (${FAILS} check(s))"
    exit 1
fi
echo "RESULT: PASS  (--scratch-dir is bit-identical to in-RAM runs, leaves no files behind,"
echo "               and rejects unsafe directories)"
echo "============================================================"
