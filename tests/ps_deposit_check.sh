#!/usr/bin/env bash
# PS-DTFE grid deposit check: the exact conservative deposit ('--exact', r3d) against the sampled one ('--sampled').
#
# Verified here, on a multi-stream Zel'dovich pancake:
#  A) the exact deposit: mass conservation at (or below) the sampled deposit's float floor, '.den' == '.a_den'
#     bit-exact (the exact deposit is nSub-independent), the raw integer '.tetTouch' count bit-exact between
#     same-split partitioned runs at 1 and N threads, '.streams' (the volume-weighted multiplicity -- a float, hence
#     checked to a tolerance, not bit-exactly) hitting the pancake's ANALYTIC values of 1 and 3, the sampled '.a_den'
#     at nSub = 1/3/5 approaching the exact grid MONOTONICALLY (L1), and (GPU builds) the GPU exact deposit against
#     the CPU's (a float32 z-sweep against the double r3d clip: float parity).
#  B) (GPU builds) the GPU exact deposit's item passes run on its two pipelines ('depositExactItems'; DTFE_GPU_TIMING
#     names them), and they conserve mass to the float rounding of their shares: DTFE_MASS_AUDIT's
#     'handed-deferred-fixed' within 1e-7 of the mass handed to the GPU, in every pass.
#  C) '--ps-lagrangian-input' with an input type other than 105 (a binary Gadget snapshot, demo/z0_64.gadget, read as
#     type 101) stops the program with its message and writes no grid: only the HDF5 Gadget reader gives the particle
#     IDs the Lagrangian positions are matched by.
#
# Usage: tests/ps_deposit_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}"
SNAP_PAN="${TMP}/pdep_input_pancake.hdf5"

echo "============================================================"
echo " PS-DTFE deposit check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
GPU_BUILT=0
[ -f o_ps/.gpu_mode_off ] || GPU_BUILT=1
# capture FIRST -- piping the binary straight into grep -q would SIGPIPE it under pipefail
FULL_HELP="$("${BIN}" --full-help 2>/dev/null)" || { echo "FAIL: ${BIN} --full-help exited non-zero"; exit 1; }
# '--sampled' marks a binary with a flag for each deposit ('--exact', the default for an '_a' field, and '--sampled');
# an older one may lack the exact deposit, which every section here needs
grep -q -- '--sampled' <<<"${FULL_HELP}" || skip_or_fail "this PS-DTFE has no --exact / --sampled."
SAMPLED="--sampled --subsamples 3"      # the sampled deposit at nSub 3 (the binary's default count is 1)

echo ">> generating the test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_PAN}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

run() {  # $1 = output root, $2 = input snapshot, rest = extra flags
    # out right after the input: multitoken options (--partition X Y Z) must not be able
    # to swallow the trailing positional
    local out="$1" snap="$2"; shift 2
    rm -f "${out}".*
    local log="${out}.log"
    set +e
    "${BIN}" "${snap}" "${out}" --grid "${GRID}" --field density velocity density_a \
        --input-format 105 --mpc-unit 1 --verbose 1 "$@" > "$log" 2>&1
    local rc=$?
    set -e
    die_on_failed_run "$rc" "$log"
}

# run 1: the sampled deposit at nSub 3 ('density_a' alone would select the exact one): the sampled mass floor of
# A and the nSub = 3 rung of its convergence ladder
echo ">> run 1: pancake, sampled deposit (nSub 3)"
run "${TMP}/pdep_pan_u" "${SNAP_PAN}" --periodic ${SAMPLED}
echo ">> run 2: pancake, --exact"
run "${TMP}/pdep_pan_ex" "${SNAP_PAN}" --periodic --exact
echo ">> run 3: pancake, --exact, --partition 2 2 2, 1 thread"
OMP_NUM_THREADS=1 run "${TMP}/pdep_pan_ex_p1" "${SNAP_PAN}" --periodic --exact --partition 2 2 2
echo ">> run 4: pancake, --exact, --partition 2 2 2, all threads"
run "${TMP}/pdep_pan_ex_pN" "${SNAP_PAN}" --periodic --exact --partition 2 2 2
echo ">> runs 5-6: pancake, sampled '.a_den' at nSub 1/5 (3 is run 1) for the convergence ladder"
run "${TMP}/pdep_pan_ns1" "${SNAP_PAN}" --periodic --subsamples 1
run "${TMP}/pdep_pan_ns5" "${SNAP_PAN}" --periodic --subsamples 5
if [ "${GPU_BUILT}" -eq 1 ]; then
    echo ">> runs 7-8: exact deposit + dispersion, CPU vs --gpu (float32 exact-clip parity)"
    # dedicated pair (with the dispersion field run() does not request): the GPU exact
    # deposit is a hand-written float32 z-sweep (not r3d), and this pair is its parity coverage in the suite
    for tag in exd exg; do
        GPUFLAG=""; [ "${tag}" = "exg" ] && GPUFLAG="--gpu"
        rm -f "${TMP}/pdep_pan_${tag}".*
        set +e
        "${BIN}" "${SNAP_PAN}" "${TMP}/pdep_pan_${tag}" --grid "${GRID}" \
            --field density velocity dispersion --input-format 105 --mpc-unit 1 --verbose 1 \
            --periodic --exact ${GPUFLAG} > "${TMP}/pdep_pan_${tag}.log" 2>&1
        rc=$?
        set -e
        if [ "${rc}" -ne 0 ]; then
            echo "   ERROR: PS-DTFE exited with code ${rc} -- last 25 lines:"
            tail -n 25 "${TMP}/pdep_pan_${tag}.log" | sed 's/^/      | /'
            exit 1
        fi
    done
    echo ">> run 9: exact deposit --gpu (its pipelines and the mass audit: B)"
    DTFE_MASS_AUDIT=1 DTFE_GPU_TIMING=1 run "${TMP}/pdep_pan_exug" "${SNAP_PAN}" --periodic --exact --gpu --partition 1 1 1
else
    echo ">> runs 7-9 skipped (CPU-only build: no exact-deposit GPU parity to check)"
fi

echo ">> C: '--ps-lagrangian-input' on a binary Gadget snapshot, named ('--input-format 101') and found (refused: no particle IDs)"
PS_LAGRANGIAN_ONLY_105="'--ps-lagrangian-input' needs an HDF5 Gadget snapshot ('--input-format 105'): only that reader"
PS_LAGRANGIAN_ONLY_105="${PS_LAGRANGIAN_ONLY_105} gives the particle IDs the Lagrangian positions are matched by."
[ -f "${ROOT}/demo/z0_64.gadget" ] || { echo "FAIL: demo/z0_64.gadget is missing"; exit 1; }
LAG_RC=""
lag_refused() {  # $1 = run tag, rest = the input type's option (none: the reader finds the binary type in the file)
    local tag="$1" rc=0; shift
    rm -f "${TMP}/${tag}".*
    "${BIN}" "${ROOT}/demo/z0_64.gadget" "${TMP}/${tag}" --grid "${GRID}" --field density_a "$@" \
        --ps-lagrangian-input "${SNAP_PAN}" --verbose 1 > "${TMP}/${tag}.log" 2>&1 || rc=$?
    LAG_RC="${LAG_RC}${LAG_RC:+ and }${rc}"
    [ "${rc}" -ne 0 ] && grep -qF -e "${PS_LAGRANGIAN_ONLY_105}" "${TMP}/${tag}.log" && [ ! -e "${TMP}/${tag}.a_den" ]
}
LAG_REFUSED=1
lag_refused pdep_lag --input-format 101 || LAG_REFUSED=0
lag_refused pdep_lags || LAG_REFUSED=0

echo ">> checking the numbers ..."
"${PY}" - "${TMP}" "${GRID}" "${GPU_BUILT}" "${LAG_REFUSED}" "${LAG_RC}" <<'PY'
import sys
import numpy as np
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh

tmp, grid, gpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lag_refused, lag_rc = int(sys.argv[4]), sys.argv[5]
ncell = grid ** 3
fails = []

def check(name, ok, detail):
    print(f"   {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)

def load(root, ext, ncomp=1):
    d = np.fromfile(root + ext, dtype=REAL).astype(np.float64)
    assert d.size == ncell * ncomp, (root + ext, d.size)
    return d

# ---------- (A) exact conservative deposit (--exact, r3d) ----------
du = load(f"{tmp}/pdep_pan_u", ".den")                 # the sampled deposit (run 1)
su = np.fromfile(f"{tmp}/pdep_pan_u.streams", dtype=REAL)
ex = load(f"{tmp}/pdep_pan_ex", ".den")
exa = load(f"{tmp}/pdep_pan_ex", ".a_den")
du_mean_err = abs(du.mean() - 1.0)                     # sampled-deposit floor (run 1)
check("A mass conserved (exact <= sampled floor)",
      abs(ex.mean() - 1.0) <= du_mean_err + 1e-7,
      f"|mean-1| = {abs(ex.mean()-1.0):.3e} (sampled: {du_mean_err:.3e})")
check("A .den == .a_den (nSub-independent)",
      np.array_equal(ex, exa), "the exact deposit is the nSub->infinity limit")
# '.tetTouch': the RAW INTEGER count of tetrahedra with a nonzero cell intersection.
# This is the field that carries the integer stream-count contract -- quantities.cc
# sums it linearly across partitions, so it stays bit-exact under any thread order.
tx = np.fromfile(f"{tmp}/pdep_pan_ex.tetTouch", dtype=REAL).astype(np.float64)
# 'integer-valued' alone is vacuous -- any int < 2^24 casts exactly, so an all-zero or
# scrambled grid would pass it. Pin the count to >= 1 as well: the periodic tessellation
# is space-filling, so EVERY cell is touched by at least one tet (measured min = 2).
check("A exact tetTouch integer-valued and >= 1",
      bool(np.allclose(tx, np.round(tx), atol=1e-6)) and tx.min() >= 1,
      f"max frac dev = {np.abs(tx - np.round(tx)).max():.2e}, "
      f"touch range = [{tx.min():.0f}, {tx.max():.0f}]")
t1 = np.fromfile(f"{tmp}/pdep_pan_ex_p1.tetTouch", dtype=REAL)
tN = np.fromfile(f"{tmp}/pdep_pan_ex_pN.tetTouch", dtype=REAL)
check("A partitioned tetTouch thread-invariant (bit-exact)", np.array_equal(t1, tN),
      "same split, 1 vs all threads (integer counts merge order-independently)")

# '.streams' is the VOLUME-WEIGHTED multiplicity (1/V_cell) sum V_int -- a float, so
# neither integer-valued nor bit-exact (float shares sum order-dependently). What this
# definition buys is that it is the ANALYTIC nSub->infinity limit, so the two properties
# below are exact statements about the Zel'dovich pancake rather than sampling artefacts.
sx = np.fromfile(f"{tmp}/pdep_pan_ex.streams", dtype=REAL).astype(np.float64)
# The pancake folds into exactly 3 streams behind the caustic. The sampled deposit
# overshoots this badly (a grazing sub-sample counts a full +1; run 1's 'su' reports
# ~14), while the exact deposit weights each tet by its intersected volume and lands on 3.
check("A exact streams max == analytic 3", abs(sx.max() - 3.0) < 0.01,
      f"max = {sx.max():.6f} (analytic 3.00; the sampled deposit gives {su.max():.2f})")
# Outside the caustic the tessellation covers each cell exactly once, so those cells
# must read exactly 1.0 -- and every cell is covered at least once (space-filling).
near1 = np.abs(sx - 1.0) < 1e-3
check("A exact streams single-stream cells == 1.0",
      sx.min() > 1.0 - 1e-3 and near1.mean() > 0.5,
      f"min = {sx.min():.6f}, {near1.mean()*100:.2f}% of cells within 1e-3 of 1.0")
# ties the two fields together: each touching tet contributes exactly 1 to '.tetTouch' and
# its volume share V_int/V_cell <= 1 to '.streams', so streams <= tetTouch pointwise. This
# is what would catch a zeroed/mis-normalized accumulator in EITHER field. (The bound needs
# the bbox rasterization -- --ps-halo-release's centroid fallback deposits V_tet/V_cell,
# which may exceed 1 for a tet bigger than a cell. No run here uses it.)
viol = int(np.sum(sx > tx + 1e-6))
check("A exact streams <= tetTouch (pointwise)", viol == 0,
      f"{viol} cells violate; min slack = {(tx - sx).min():.6f}")
# partition split: same tolerance form as the density check just below
s1 = np.fromfile(f"{tmp}/pdep_pan_ex_p1.streams", dtype=REAL)
sN = np.fromfile(f"{tmp}/pdep_pan_ex_pN.streams", dtype=REAL)
smax = np.abs(s1.astype(np.float64) - sN).max() / (np.abs(s1).max() + 1e-30)
check("A partitioned streams thread-invariant", smax < 1e-5, f"max rel = {smax:.3e}")
d1 = load(f"{tmp}/pdep_pan_ex_p1", ".den")
dN = load(f"{tmp}/pdep_pan_ex_pN", ".den")
dmax = np.abs(d1 - dN).max() / (np.abs(d1).max() + 1e-30)
check("A partitioned density thread-invariant", dmax < 1e-5, f"max rel = {dmax:.3e}")
# sampled '.a_den' must approach the exact grid MONOTONICALLY with nSub
l1 = {}   # nSub=3 is the count run 1 gives (pdep_pan_u)
for ns, root in ((1, "pdep_pan_ns1"), (3, "pdep_pan_u"), (5, "pdep_pan_ns5")):
    l1[ns] = np.abs(load(f"{tmp}/{root}", ".a_den") - exa).mean()
check("A sampled a_den converges to exact (L1, nSub 1>3>5)",
      l1[1] > l1[3] > l1[5],
      f"L1 = {l1[1]:.4e} (nSub=1) > {l1[3]:.4e} (nSub=3) > {l1[5]:.4e} (nSub=5)")
# float32 exact-clip parity: the GPU exact deposit (depositExactItems in metal/ps_deposit.metal, clipping
# with sweep_moments from metal/exact_clip.metal.inc) vs the CPU double r3d
if gpu:
    for ext, nc, tol in ((".den", 1, 1e-4), (".vel", 3, 1e-4), (".velDisp", 1, 1e-4)):
        c = load(f"{tmp}/pdep_pan_exd", ext, nc)
        g = load(f"{tmp}/pdep_pan_exg", ext, nc)
        mrel = np.abs(c - g).mean() / (np.abs(c).max() + 1e-30)
        check(f"A exact GPU{ext} matches CPU", mrel < tol, f"mean rel = {mrel:.3e}")
    sc = np.fromfile(f"{tmp}/pdep_pan_exd.streams", dtype=REAL).astype(np.float64)
    sg = np.fromfile(f"{tmp}/pdep_pan_exg.streams", dtype=REAL).astype(np.float64)
    srel = np.abs(sc - sg).max() / (np.abs(sc).max() + 1e-30)
    check("A exact GPU streams match CPU", srel < 1e-3,
          f"max rel = {srel:.3e} (float volume-weighted multiplicity)")

    # ---------- (B) run 9: which pipeline ran each item pass (DTFE_GPU_TIMING), and the mass audit of every pass ----------
    import re
    num = r"([-+]?(?:nan|inf|[0-9.]+(?:[eE][-+]?[0-9]+)?))"
    with open(f"{tmp}/pdep_pan_exug.log") as fh:
        text = fh.read()
    # passes on the two-pipeline pair, and on the run-time-pass pipeline (which only the removed linear deposit used)
    nd = len(re.findall(r"PS-DTFE Metal: depositExactItems\s", text))
    nl = len(re.findall(r"PS-DTFE Metal: exactItems linear\s", text))
    check("B exact GPU: its item passes on the two pipelines", nd >= 2 and nl == 0,
          f"{nd} pass(es) on 'depositExactItems', {nl} on 'exactItems linear' (want none there)")
    rows = []                # [(handed, handed-deferred-fixed)] of every pass line with a GPU part
    for line in text.splitlines():
        if not line.startswith("PS mass audit: partition"):
            continue
        h, x = re.search(r"gpu handed " + num, line), re.search(r"handed-deferred-fixed " + num, line)
        if h and x:
            rows.append((float(h.group(1)), float(x.group(1))))
    worst = max((abs(x) / h if h > 0 else float("inf") for h, x in rows), default=float("inf"))
    check("B exact GPU: mass audit", len(rows) > 0 and worst <= 1e-7,
          f"|handed-deferred-fixed| / handed = {worst:.2e} over {len(rows)} pass(es) (<= 1e-7: the shares' rounding)")
else:
    print("   SKIP A exact GPU parity, B (CPU-only build)")

# ---------- (C) a Lagrangian input needs the HDF5 reader ----------
check("C '--ps-lagrangian-input' on a binary Gadget snapshot stops the run with its message, no grid written",
      lag_refused == 1, f"the type named and found in the file: exit {lag_rc} (logs: {tmp}/pdep_lag.log, pdep_lags.log)")

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (the exact deposit conserves mass, is nSub-independent and the sampled deposit's limit)")
PY

echo "============================================================"
echo " deposit check PASSED"
echo "============================================================"
