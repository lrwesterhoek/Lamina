#!/usr/bin/env bash
# PS-DTFE periodic seam: the period must end where every periodic copy is an exact -L shift (subpartition.h psPeriodicSeam).
#   S1 OFFSET single stream   S2 OFFSET crossed waves   S3 NEARLAT single stream   S4 the logs name the exact seam
# Usage: tests/ps_periodic_seam_check.sh [--no-build]   (DTFE_TEST_PRECISION=double for the double build)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"

BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}/ps_seam_check"
mkdir -p "${TMP}"
GRID=64

echo "============================================================"
echo " PS-DTFE periodic seam (exact periodic copies)   grid=${GRID}^3  (${DTFE_TEST_PRECISION})"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

X0=131022; X1=131122
OFFBOX=(--box "${X0}" "${X1}" "${X0}" "${X1}" "${X0}" "${X1}")
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/off_ss.hdf5" --n 32 --box 100 --offset "${X0}" \
    --amplitude-factor 0.5 --crossed-waves --jitter-frac 0.2 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/off_ms.hdf5" --n 32 --box 100 --offset "${X0}" \
    --amplitude-factor 1.8 --crossed-waves --jitter-frac 0.2 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/nearlat.hdf5" --n 32 --box 100 \
    --amplitude-factor 0.5 --crossed-waves --jitter-frac 0.00001 >/dev/null

# the sampled deposit ('density_a' would select the exact one): 3 points per cell axis for S1 and S3, 1 for S2
run() {  # $1 = output name, $2 = snapshot, rest = extra options
    local name="$1" snap="$2"; shift 2
    local out="${TMP}/run_${name}"
    rm -f "${out}".a_den "${out}".a_streams "${out}".a_hidden_streams
    if ! DTFE_MASS_AUDIT=2 "${BIN}" "${snap}" "${out}" --grid "${GRID}" --periodic --field density_a --sampled --ps-vertex-mass \
            --input-format 105 --mpc-unit 1 --verbose 2 "$@" > "${out}.log" 2>&1; then
        echo "FAIL: run ${name} exited non-zero -- last lines of ${out}.log:"
        tail -n 15 "${out}.log" | sed 's/^/   | /'
        exit 1
    fi
    echo "   ran ${name}"
}
run off_p1 "${TMP}/off_ss.hdf5" --partition 1 1 1 --subsamples 3 "${OFFBOX[@]}"
run off_p2 "${TMP}/off_ss.hdf5" --partition 2 2 2 --subsamples 3 "${OFFBOX[@]}"
run off_p3 "${TMP}/off_ss.hdf5" --partition 3 3 3 --subsamples 3 "${OFFBOX[@]}"
run ms_p1  "${TMP}/off_ms.hdf5" --partition 1 1 1 --subsamples 1 "${OFFBOX[@]}"
run ms_p2  "${TMP}/off_ms.hdf5" --partition 2 2 2 --subsamples 1 "${OFFBOX[@]}"
run near   "${TMP}/nearlat.hdf5" --partition 1 1 1 --subsamples 3

"${PY}" - "${TMP}" "${GRID}" "${DTFE_TEST_REAL}" <<'PY'
import re, sys
from pathlib import Path
import numpy as np
tmp, g, real = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
double = real == "float64"
dt = np.float64 if double else np.float32
mean_tol, st_tol = (1e-12, 1e-9) if double else (1e-7, 1e-5)
fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def grid(name, ext):
    return np.fromfile(tmp / f"run_{name}.{ext}", dtype=dt).astype(np.float64).reshape(g, g, g)
def log(name):
    return (tmp / f"run_{name}.log").read_text(errors="replace")
def closure(name):
    m = re.findall(r"per-particle closure: (\d+) of (\d+) particles off \(net ([^)]*)\)", log(name))
    return (int(m[-1][0]), m[-1][2]) if m else (None, None)
idx = np.indices((g, g, g))
facedist = np.minimum(idx, g - 1 - idx).min(axis=0)

# S1
for name in ("off_p1", "off_p2", "off_p3"):
    st, den = grid(name, "a_streams"), grid(name, "a_den")
    bad = np.abs(st - 1.0) > st_tol
    check("S1", not bad.any(), f"{name}: '.a_streams' = 1 in every cell ({int((st < 1 - st_tol).sum())} below, "
          f"{int((st > 1 + st_tol).sum())} above; min {st.min():.4f}, max {st.max():.4f})")
    dev = den.mean() - 1.0
    check("S1", abs(dev) < mean_tol, f"{name}: box mean of '.a_den' - 1 = {dev:+.2e} (|.| < {mean_tol:g})")
n, net = closure("off_p1")
check("S1", n == 0, f"off_p1: per-particle closure: {n} particles off (net {net})")

# S2
for name in ("ms_p1", "ms_p2"):
    st, den = grid(name, "a_streams"), grid(name, "a_den")
    r = np.rint(st)
    nonint = np.abs(st - r) > st_tol
    even = (~nonint) & (r % 2 == 0)
    check("S2", not nonint.any() and not even.any(),
          f"{name}: every sample's stream count odd ({int(even.sum())} even, {int(nonint.sum())} non-integer of {st.size}; "
          f"counts {sorted(set(r[~nonint].astype(int).tolist()))})")
    dev = den.mean() - 1.0
    check("S2", abs(dev) < mean_tol, f"{name}: box mean of '.a_den' - 1 = {dev:+.2e}")

# S3
n, net = closure("near")
check("S3", n == 0, f"near: per-particle closure: {n} particles off (net {net})")
den = grid("near", "a_den"); st = grid("near", "a_streams")
dev = den.mean() - 1.0
check("S3", abs(dev) < mean_tol, f"near: box mean of '.a_den' - 1 = {dev:+.2e}")
band = facedist < 4
bad = (np.abs(st - 1.0) > st_tol) & band
check("S3", not bad.any(), f"near: '.a_streams' = 1 within 4 cells of the faces ({int(bad.sum())} of {int(band.sum())} off)")

# S4
for name in ("off_p1", "off_p2", "off_p3", "ms_p1", "ms_p2", "near"):
    lines = [l for l in log(name).splitlines() if "periodic ownership period per axis" in l]
    ok = bool(lines) and all("top particle plane" not in l for l in lines)
    check("S4", ok, f"{name}: the exact seam on every axis ({lines[-1].split(':', 1)[1].strip() if lines else 'no seam line in the log'})")

print(f"  {'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
PY
