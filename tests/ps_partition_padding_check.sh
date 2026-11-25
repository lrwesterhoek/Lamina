#!/usr/bin/env bash
# PS-DTFE partition/padding independence: a partitioned run deposits the same tetrahedra as one triangulation, whatever the padding.
#   P1 hidden-stream flags   P2 '.a_den'   P3 '.a_streams'   P4 vertex-mass conservation
# Usage: tests/ps_partition_padding_check.sh [--no-build]   (DTFE_TEST_PRECISION=double for the double build)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"

N="${N:-32}"; GRID="${GRID:-64}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}/ps_padcheck"
mkdir -p "${TMP}"
SNAP="${TMP}/input_cw.hdf5"

echo "============================================================"
echo " PS-DTFE partition/padding independence   N=${N}^3  grid=${GRID}^3  (${DTFE_TEST_PRECISION})"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box 100 \
    --amplitude-factor 1.8 --crossed-waves >/dev/null

# the sampled deposit at 3 points per cell axis ('density_a' would select the exact one): P2 and P3's tolerances
# are its sub-samples'
run() {  # $1 = output name, rest = extra options
    local name="$1"; shift
    local out="${TMP}/run_${name}"
    rm -f "${out}".a_den "${out}".a_streams "${out}".a_hidden_streams
    if ! "${BIN}" "${SNAP}" "${out}" --grid "${GRID}" --periodic --field density_a \
            --sampled --subsamples 3 --ps-vertex-mass --input-format 105 --mpc-unit 1 --verbose 1 "$@" > "${out}.log" 2>&1; then
        echo "FAIL: run ${name} exited non-zero -- last lines of ${out}.log:"
        tail -n 15 "${out}.log" | sed 's/^/   | /'
        exit 1
    fi
    echo "   ran ${name}"
}
run single
run p2       --partition 2 2 2
run p2pad50  --partition 2 2 2 --padding 50
run p2lag8   --partition 2 2 2 --ps-lagrangian-padding 8

"${PY}" - "${TMP}" "${GRID}" "${DTFE_TEST_REAL}" <<'PY'
import sys
import numpy as np
tmp, grid, real = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ncell = grid ** 3
double = real == "float64"
den_tol = 1e-10 if double else 1e-4
fails = []

def load(name, ext):
    raw = np.fromfile(f"{tmp}/run_{name}{ext}", dtype=np.uint8)
    per = raw.size // ncell
    if ext == ".a_hidden_streams":
        return raw.view({1: np.uint8, 4: np.int32, 8: np.float64}[per]).astype(np.int64)
    return raw.view({4: np.float32, 8: np.float64}[per]).astype(np.float64)

ref = {e: load("single", e) for e in (".a_den", ".a_streams", ".a_hidden_streams")}
for name in ("single", "p2", "p2pad50", "p2lag8"):
    den = load(name, ".a_den")
    drift = abs(den.sum() - ncell) / ncell
    print(f"   P4 {name:8s} sum(.a_den) - cells = {den.sum() - ncell:+.3e}  (relative {drift:.1e})")
    if drift > 1e-6:
        fails.append(f"P4 {name}: vertex-mass total off by {drift:.1e} of the mass")
    if name == "single":
        continue
    hs = load(name, ".a_hidden_streams")
    nflip = int((hs != ref[".a_hidden_streams"]).sum())
    a, b = ref[".a_den"], den
    big = np.abs(a) > 1e-6 * np.abs(a).max()
    rel = np.abs(a - b)[big] / np.abs(a)[big]
    nbad = int((rel > den_tol).sum())
    ds = float(np.abs(load(name, ".a_streams") - ref[".a_streams"]).max())
    print(f"   {name:8s} vs single: hidden-stream flips {nflip}, .a_den max rel {rel.max():.2e} "
          f"({nbad} cells > {den_tol:.0e}), .a_streams max |d| {ds:.2e}")
    if nflip:
        fails.append(f"P1 {name}: {nflip} hidden-stream flags differ from the single run")
    if nbad:
        fails.append(f"P2 {name}: {nbad} '.a_den' cells differ by more than {den_tol:.0e} (max {rel.max():.2e})")
    if ds > 1e-3:
        fails.append(f"P3 {name}: '.a_streams' differs by {ds:.2e}")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (every partition/padding variant deposits the single run's tetrahedra)")
PY

echo "============================================================"
echo " partition/padding check PASSED"
echo "============================================================"
