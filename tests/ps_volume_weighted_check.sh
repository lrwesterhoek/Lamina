#!/usr/bin/env bash
# PS-DTFE's velocity moments are volume-weighted in every run: the velocity, its gradient and the fields built on it
# (divergence, shear, vorticity) are Eulerian-VOLUME-weighted means over the streams -- the standard-DTFE '_a'
# convention that linear theory's -aHf*delta refers to -- and the dispersion stays mass-weighted (sigma_ij is an
# f-weighted moment by definition). '--ps-volume-weighted' is still accepted (the drivers and the launcher pass it)
# and changes nothing. Asserted on the multi-stream pancake:
#  A) THE FLAG: a run with '--ps-volume-weighted' writes the bytes of the same run without it, every output: the
#     grids (density, streams, velocity, dispersion, divergence, shear) and the point files of '--sample-points
#     --ps-per-stream --pts-vel-grad --pts-den-grad' on a plane with single- and three-stream points. (That the
#     points' velocity is the plain stream mean and their dispersion the density-weighted one is
#     tests/ps_point_conventions_check.sh D.)
#  H) THE HEADER LINE: every PS-DTFE run's log carries the line
#         '<tab> velocity moments       : volume-weighted (the dispersion mass-weighted)'
#     once, at exactly the verbosities that print its 'RUNNING:' line: at the lowest one that does, the line is
#     there; a '--verbose 0' log holds neither. A standard-DTFE run at that verbosity prints its 'RUNNING:' line
#     and no 'velocity moments' line.
#  D) '--ps-linear-deposit' (removed) beside the flag stops the run with its own message: the program checks the
#     removed options first.
#  E) CPU vs GPU parity under the volume weighting (GPU builds only).
# The flag must parse (the drivers pass it on every PS-DTFE command): a binary that refuses it fails here, it is
# never a skip.
#
# Usage: tests/ps_volume_weighted_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}" "${DTFE_BIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
SNAP_PAN="${TMP}/pvw_input_pancake.hdf5"
PTS_PLANE="${TMP}/pvw_input_plane.bin"
# the header line of every PS-DTFE run (src/user_options.cc printOptions), character for character
VM_LINE=$'\t velocity moments       : volume-weighted (the dispersion mass-weighted)'

echo "============================================================"
echo " PS-DTFE volume-weighted velocity moments check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
build_target "${1:-}" DTFE o
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
[ -x "${DTFE_BIN}" ] || { echo "FAIL: ${DTFE_BIN} not built (H compares a standard run's log)"; exit 1; }
GPU_BUILT=0
[ -f o_ps/.gpu_mode_off ] || GPU_BUILT=1
# the flag is a hidden option (no '--full-help' entry): a binary that knows it prints the same help with it given and
# nothing on stderr; one that refuses it FAILS (never a skip: the drivers pass the flag on every PS-DTFE command)
FULL_HELP="$("${BIN}" --full-help 2>/dev/null)" || { echo "FAIL: ${BIN} --full-help exited non-zero"; exit 1; }
FLAG_HELP="$("${BIN}" --ps-volume-weighted --full-help 2>"${TMP}/pvw_flag_help.err")" || FLAG_HELP=""
if [ "${FLAG_HELP}" != "${FULL_HELP}" ] || [ -s "${TMP}/pvw_flag_help.err" ]; then
    echo "FAIL: ${BIN} does not accept --ps-volume-weighted silently (see ${TMP}/pvw_flag_help.err)"; exit 1
fi

echo ">> generating test snapshot and a plane of sample points ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null
"${PY}" - "${PTS_PLANE}" "${BOX}" <<'PY'
import sys
import numpy as np
out, box = sys.argv[1], float(sys.argv[2])
# a plane across the pancake (single- and three-stream points), off the box's symmetry planes
c = (np.arange(64) + 0.5) * box / 64
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.503 * box)], 1).astype(np.float64).tofile(out)
PY

run() {  # $1 = output root, rest = extra flags
    local out="$1"; shift
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "${SNAP_PAN}" "${out}" --grid "${GRID}" \
        --field density velocity dispersion divergence shear --input-format 105 --mpc-unit 1 \
        --verbose 1 --subsamples 1 --periodic "$@" > "$log" 2>&1
    local rc=$?
    set -e
    die_on_failed_run "$rc" "$log"
}
PTS=(--sample-points "${PTS_PLANE}" --ps-per-stream --pts-vel-grad --pts-den-grad)

echo ">> run 1: without the flag (grids and point files)"
run "${TMP}/pvw_def" "${PTS[@]}"
echo ">> run 2: --ps-volume-weighted (grids and point files)"
run "${TMP}/pvw_vw" "${PTS[@]}" --ps-volume-weighted
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> run 3: --ps-volume-weighted --gpu (GPU parity)"
    run "${TMP}/pvw_vwg" "${PTS[@]}" --ps-volume-weighted --gpu
else
    echo ">> run 3 skipped (CPU-only build: no GPU parity to check)"
fi

echo ">> H: the header line at the verbosities 0 to 3, and a standard-DTFE run"
for v in 0 1 2 3; do
    rm -f "${TMP}/pvw_v${v}".*
    "${BIN}" "${SNAP_PAN}" "${TMP}/pvw_v${v}" --grid 16 --field density --input-format 105 --mpc-unit 1 \
        --subsamples 1 --periodic --verbose "${v}" > "${TMP}/pvw_v${v}.log" 2>&1 \
        || die_on_failed_run 1 "${TMP}/pvw_v${v}.log"
done
H_LOW=""                    # the lowest verbosity whose log holds the 'RUNNING:' line
H_RULE=1                    # every level: the header line once where 'RUNNING:' is, nowhere else
H_SEEN=""
for v in 0 1 2 3; do
    nr=$(grep -ac '^RUNNING: ' "${TMP}/pvw_v${v}.log" || true)
    nh=$(grep -acxF -e "${VM_LINE}" "${TMP}/pvw_v${v}.log" || true)
    H_SEEN="${H_SEEN} v${v}:${nr}/${nh}"
    if [ "${nr}" -gt 0 ]; then
        [ -n "${H_LOW}" ] || H_LOW="${v}"
        [ "${nh}" = 1 ] || H_RULE=0
    else
        [ "${nh}" = 0 ] || H_RULE=0
    fi
done
H_STD=0
if [ -n "${H_LOW}" ]; then
    rm -f "${TMP}/pvw_std".*
    "${DTFE_BIN}" "${SNAP_PAN}" "${TMP}/pvw_std" --grid 16 --field density velocity --input-format 105 --mpc-unit 1 \
        --periodic --verbose "${H_LOW}" > "${TMP}/pvw_std.log" 2>&1 || die_on_failed_run 1 "${TMP}/pvw_std.log" DTFE
    if grep -aq '^RUNNING: ' "${TMP}/pvw_std.log" && ! grep -aqF 'velocity moments' "${TMP}/pvw_std.log"; then H_STD=1; fi
fi

echo ">> D: the removed --ps-linear-deposit beside --ps-volume-weighted must stop the run with its own message"
set +e
"${BIN}" "${SNAP_PAN}" "${TMP}/pvw_rej" --grid "${GRID}" --field density --input-format 105 \
    --mpc-unit 1 --periodic --ps-volume-weighted --ps-linear-deposit > "${TMP}/pvw_rej.log" 2>&1
RC_REJ=$?
set -e

echo ">> checking ..."
"${PY}" - "${TMP}" "${GRID}" "${GPU_BUILT}" "${RC_REJ}" "${PS_NO_LINEAR_DEPOSIT}" \
    "${H_LOW:-none}" "${H_RULE}" "${H_STD}" "${H_SEEN}" <<'PY'
import os
import sys

import numpy as np
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

tmp, grid, gpu, rc_rej = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
removed_msg = sys.argv[5]
h_low, h_rule, h_std, h_seen = sys.argv[6], sys.argv[7] == "1", sys.argv[8] == "1", sys.argv[9].strip()
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load(root, ext, ncomp=1):
    d = np.fromfile(root + ext, dtype=REAL)
    assert d.size == ncell * ncomp, (root + ext, d.size)
    return d

def outputs(root):
    """The output files of run 'root' (its suffixes): the logs and the run-log family are not outputs."""
    name = os.path.basename(root)
    return sorted(f[len(name):] for f in os.listdir(tmp)
                  if f.startswith(name + ".") and not f.endswith(".log") and ".runlog" not in f)

streams = load(f"{tmp}/pvw_def", ".streams")
multi = streams > 1.001                      # float contract: never compare == 1 exactly
check("pancake actually multi-streams", 0.05 < float(multi.mean()) < 0.5,
      f"{100*multi.mean():.1f}% multi-stream cells")
pst = np.fromfile(f"{tmp}/pvw_def.pts_streams", dtype=np.int32)
check("the plane holds single- and multi-stream points", bool((pst == 1).any()) and bool((pst > 1).any()),
      f"{int((pst == 1).sum())} single-stream, {int((pst > 1).sum())} multi-stream points")

# ---------- (A) the flag changes no byte ----------
a, b = outputs(f"{tmp}/pvw_def"), outputs(f"{tmp}/pvw_vw")
differ = [e for e in a if e in b and open(f"{tmp}/pvw_def{e}", "rb").read() != open(f"{tmp}/pvw_vw{e}", "rb").read()]
npts, ngrid = sum(e.startswith(".pts_") for e in a), sum(not e.startswith(".pts_") for e in a)
check("A --ps-volume-weighted writes the bytes of a run without it", a == b and npts >= 6 and ngrid >= 6 and not differ,
      f"{ngrid} grids and {npts} point files compared"
      + ("" if a == b else f"; the file sets differ: {sorted(set(a) ^ set(b))}")
      + (f"; DIFFER: {differ}" if differ else ""))

# ---------- (H) the header line ----------
check("H the header line is printed exactly where the 'RUNNING:' line is (RUNNING/header lines per level)",
      h_low != "none" and h_rule, f"lowest verbosity with 'RUNNING:': {h_low}; {h_seen}")
check("H a '--verbose 0' log holds neither", h_seen.startswith("v0:0/0"), h_seen)
check("H a standard-DTFE run prints its 'RUNNING:' line and no 'velocity moments' line", h_std,
      f"{tmp}/pvw_std.log, at --verbose {h_low}")

# ---------- (D) the removed --ps-linear-deposit beside the flag ----------
with open(f"{tmp}/pvw_rej.log", errors="replace") as fh:
    rej_log = fh.read()
check("D --ps-linear-deposit beside the flag refused with the removal message", rc_rej != 0 and removed_msg in rej_log,
      f"exit code {rc_rej}, the message {'given' if removed_msg in rej_log else 'MISSING'} ({tmp}/pvw_rej.log)")

# ---------- (E) GPU parity under the volume weighting ----------
if gpu:
    for ext, ncomp, tol in ((".den", 1, 1e-4), (".vel", 3, 1e-4),
                            (".velDisp", 1, 1e-4), (".velShear", 5, 1e-3)):
        c = load(f"{tmp}/pvw_vw", ext, ncomp).astype(np.float64)
        g = load(f"{tmp}/pvw_vwg", ext, ncomp).astype(np.float64)
        mrel = float(np.abs(c - g).mean() / (np.abs(c).max() + 1e-30))
        check(f"E GPU{ext} matches CPU", mrel < tol, f"mean rel = {mrel:.3e}")
    sc = load(f"{tmp}/pvw_vw", ".streams")
    sg = load(f"{tmp}/pvw_vwg", ".streams")
    eq = float((sc == sg).mean())
    check("E GPU streams match CPU", eq == 1.0, f"{eq*100:.4f}% equal (exact inside test on both: must be 100%)")
else:
    print("   SKIP E GPU parity (CPU-only build)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (--ps-volume-weighted changes no byte; every PS-DTFE run log names its volume-weighted moments)")
PY

echo "============================================================"
echo " volume-weighted check PASSED"
echo "============================================================"
