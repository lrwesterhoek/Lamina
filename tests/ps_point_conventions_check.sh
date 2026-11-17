#!/usr/bin/env bash
# PS-DTFE point evaluation in the grids' conventions (the only conventions; '--ps-point-conventions grid' names them).
#
# The points follow the run's grid conventions (ps_point_eval.cc, 'The grid deposit's conventions at the points'):
# its mass model (--ps-vertex-mass) and PS-DTFE's velocity moments, volume-weighted in every run (the dispersion
# mass-weighted; '--ps-volume-weighted' is accepted and changes nothing): with evolved (z = 127) initial positions as
# the Lagrangian input, the rho_bar*V_lag tetrahedron masses of a run without --ps-vertex-mass put a plane's density
# contrast (and that run's grid's) at ~0.8 of the true one at z = 20. Verified here, on synthetic boxes:
#  A) THE NAME: '--ps-point-conventions grid' (a hidden option) gives byte-identical outputs to a run without it,
#     with --ps-vertex-mass (and the accepted --ps-volume-weighted), on multi-stream crossed waves.
#  B) THE IC BIAS: single-stream crossed waves whose InitialCoordinates are displaced towards the final positions
#     by alpha = 0.1645 (TNG's D(z=127)/D(z=20)), --ps-vertex-mass --ps-volume-weighted --exact, a slab
#     of four sub-planes through one cell layer: the plane's cell blocks against the grid's cells (and against the
#     analytic cell averages) have slope ~1, and the perturbed-IC plane equals the true-IC plane to ~1%. The control:
#     the same plane from a run without --ps-vertex-mass has slope ~1-alpha (the box does carry the bias). The exact
#     deposit isolates the points: the sampled nSub=3 deposit of this one-particle-per-cell box is itself noisy at
#     the cell scale (printed).
#  C) MASS AND NOISE on a uniform near-lattice box (jitter 0.01, TNG-like initial conditions): random points
#     average to rho_bar, and the point noise stays below the vertex-share grid's own (the per-tetrahedron
#     m_tet/V_eul -- the grid's density -- puts a normal mass into near-flat slivers no point hits: a mean of 0.89).
#  D) THE WEIGHTS at the points, recomputed from the per-stream records of ONE run (a pancake, --ps-per-stream,
#     without '--ps-volume-weighted'): '.pts_vel' is the plain mean of the stream velocities (the volume weighting:
#     every stream fills the point) and '.pts_velDisp' the density-weighted second moment about the density-weighted
#     mean velocity (the dispersion stays mass-weighted, as in the grids), 0 at a single-stream point; on this plane
#     the density-weighted mean velocity is another value at the multi-stream points (the weighting is seen).
#     (That '--ps-volume-weighted' changes no point file is tests/ps_volume_weighted_check.sh A.)
#  E) PARTITIONS: --partition 2 2 2 equals the single run to float rounding with --ps-vertex-mass (the vertex
#     ratios need the partition's 2-ring, inside its padding).
#  F) --serve (dtfelib.Estimator with --ps-vertex-mass alone: it passes no '--ps-volume-weighted'): bit-identical to
#     the --sample-points run with --ps-vertex-mass --ps-volume-weighted.
#  R) '--ps-point-conventions legacy' was removed: the program stops with its message and writes no point file, under
#     the older spelling '--pts-conventions legacy' too; another value stops it with the one value left ('grid').
# Usage: tests/ps_point_conventions_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-32}"; GRID="${GRID:-32}"; BOX="${BOX:-100.0}"; ALPHA="${ALPHA:-0.1645}"; AMP="${AMP:-0.4}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}/ps_point_conventions"; rm -rf "${TMP}"; mkdir -p "${TMP}"

echo "============================================================"
echo " PS-DTFE point conventions check   N=${N}^3  grid=${GRID}^3  alpha=${ALPHA}"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
FULL_HELP="$("${BIN}" --full-help 2>/dev/null)" || { echo "FAIL: ${BIN} --full-help exited non-zero"; exit 1; }
# the option is hidden (parsed, not listed in --full-help): a binary that knows it prints the same help with it given,
# one that does not refuses the unknown option
[ "$("${BIN}" --ps-point-conventions grid --full-help 2>/dev/null || true)" = "${FULL_HELP}" ] \
    || skip_or_fail "this PS-DTFE has no --ps-point-conventions."

echo ">> test data ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_lin.hdf5" --n "${N}" --box "${BOX}" \
    --amplitude-factor "${AMP}" --crossed-waves --jitter-frac 0.05 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_uni.hdf5" --n "${N}" --box "${BOX}" \
    --amplitude-factor 0 --jitter-frac 0.01 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_crw.hdf5" --n 24 --box "${BOX}" \
    --amplitude-factor 1.8 --crossed-waves --jitter-frac 0.02 --seed 7 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${TMP}/in_pan.hdf5" --n 24 --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null
"${PY}" - "${TMP}" "${ALPHA}" "${BOX}" "${GRID}" <<'PY'
import shutil, sys
import h5py
import numpy as np
tmp, alpha, box, g = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
# the perturbed initial positions: q + alpha (x - q), the min-image displacement (tests/ps_vertex_mass_check.sh)
shutil.copyfile(f"{tmp}/in_lin.hdf5", f"{tmp}/in_pic.hdf5")
with h5py.File(f"{tmp}/in_pic.hdf5", "r+") as f:
    q = f["PartType1/InitialCoordinates"][...].astype(np.float64)
    x = f["PartType1/Coordinates"][...].astype(np.float64)
    d = x - q
    d -= box * np.round(d / box)
    f["PartType1/InitialCoordinates"][...] = ((q + alpha * d) % box).astype(np.float32)
# the slab: four sub-planes of (4g)^2 points filling cell layer g/2
n, k0 = 4 * g, g // 2
c = (np.arange(n) + 0.5) * box / n
zs = (k0 + (np.arange(4) + 0.5) / 4) * box / g
X, Y, Z = np.meshgrid(c, c, zs, indexing="ij")
np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float64).tofile(f"{tmp}/in_slab.bin")
# a regular 2g^3 set (noise) and 2e5 uniform random points (mean)
c2 = (np.arange(2 * g) + 0.5) * box / (2 * g)
X, Y, Z = np.meshgrid(c2, c2, c2, indexing="ij")
np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float64).tofile(f"{tmp}/in_full.bin")
np.random.default_rng(11).uniform(0., box, (200000, 3)).astype(np.float64).tofile(f"{tmp}/in_rand.bin")
# a multi-stream plane through the crossed waves (not on the wave's symmetry planes)
c3 = (np.arange(96) + 0.5) * box / 96
X, Y = np.meshgrid(c3, c3, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.503 * box)], 1).astype(np.float64).tofile(f"{tmp}/in_plane.bin")
PY

run() {  # $1 = output root, $2 = snapshot, rest = flags
    local out="$1" snap="$2"; shift 2
    rm -f "${out}".*
    if ! OMP_NUM_THREADS=2 "${BIN}" "${snap}" "${out}" --grid "${GRID}" --periodic --input-format 105 --mpc-unit 1 \
            --verbose 1 "$@" > "${out}.log" 2>&1; then
        echo "   ERROR: PS-DTFE failed -- last 20 lines of ${out}.log:"; tail -n 20 "${out}.log" | sed 's/^/      | /'
        exit 1
    fi
}
VMW=(--ps-vertex-mass --ps-volume-weighted)
PE=(--ps-per-stream --pts-vel-grad --pts-den-grad)

echo ">> A: '--ps-point-conventions grid' named (multi-stream crossed waves, --partition 2 2 2)"
AF=(--field density_a velocity_a dispersion_a --sample-points "${TMP}/in_plane.bin" "${PE[@]}"
    --partition 2 2 2 --max-concurrent 2 --ps-merge index-order)
run "${TMP}/a_on_grid"    "${TMP}/in_crw.hdf5" "${AF[@]}" "${VMW[@]}"
run "${TMP}/a_on_named"   "${TMP}/in_crw.hdf5" "${AF[@]}" "${VMW[@]}" --ps-point-conventions grid
echo ">> B: perturbed-IC slab (exact deposit); the sampled deposit for comparison"
run "${TMP}/b_linear_grid" "${TMP}/in_pic.hdf5" --field density_a "${VMW[@]}" --exact \
    --sample-points "${TMP}/in_slab.bin" --ps-stream-density linear
run "${TMP}/b_linear_true" "${TMP}/in_lin.hdf5" --field density_a --exact \
    --sample-points "${TMP}/in_slab.bin" --ps-stream-density linear
# the control: the same plane without --ps-vertex-mass (rho_bar*V_lag masses at the points) carries the IC bias
run "${TMP}/b_linear_nomass" "${TMP}/in_pic.hdf5" --field density_a --ps-volume-weighted --exact \
    --sample-points "${TMP}/in_slab.bin" --ps-stream-density linear
# the comparison is the nSub = 3 sampled deposit ('--sampled --subsamples 3': 'density_a' alone selects the exact
# one, and '--sampled' alone takes 1 point per cell axis)
run "${TMP}/b_sampled_grid"   "${TMP}/in_pic.hdf5" --field density_a "${VMW[@]}" --sampled --subsamples 3 \
    --sample-points "${TMP}/in_slab.bin"
echo ">> C: uniform near-lattice box"
run "${TMP}/c_linear_full" "${TMP}/in_uni.hdf5" --field density_a "${VMW[@]}" --exact \
    --sample-points "${TMP}/in_full.bin" --ps-stream-density linear
run "${TMP}/c_linear_rand" "${TMP}/in_uni.hdf5" --field density_a "${VMW[@]}" \
    --sample-points "${TMP}/in_rand.bin" --ps-stream-density linear
echo ">> D: the weights at the points (a pancake: single- and three-stream points, --ps-per-stream, no flag)"
DF=(--field density_a --sample-points "${TMP}/in_plane.bin" "${PE[@]}")
run "${TMP}/d_grid"      "${TMP}/in_pan.hdf5" "${DF[@]}"
# (F's reference: the crossed waves with the vertex masses)
run "${TMP}/d_both_grid" "${TMP}/in_crw.hdf5" "${DF[@]}" "${VMW[@]}"
echo ">> E: --partition 2 2 2 vs one triangulation (--ps-vertex-mass)"
run "${TMP}/e_single" "${TMP}/in_crw.hdf5" "${DF[@]}" "${VMW[@]}"
run "${TMP}/e_part"   "${TMP}/in_crw.hdf5" "${DF[@]}" "${VMW[@]}" --partition 2 2 2 --max-concurrent 2
echo ">> R: '--ps-point-conventions legacy' (removed), its older spelling, an unknown value: refused before anything runs"
refused() {  # $1 = output root, $2 = the text it must stop with, rest = flags: 0 when the run exits non-zero with that
             # text and writes no point file
    local out="$1" text="$2"; shift 2
    rm -f "${out}".*
    local rc=0
    OMP_NUM_THREADS=2 "${BIN}" "${TMP}/in_pan.hdf5" "${out}" --grid "${GRID}" --periodic --input-format 105 --mpc-unit 1 \
        --verbose 1 --field density_a --sample-points "${TMP}/in_plane.bin" "$@" > "${out}.log" 2>&1 || rc=$?
    [ "${rc}" -ne 0 ] && grep -qF -e "${text}" "${out}.log" && [ ! -e "${out}.pts_den" ]
}
R_LEG=0; R_OLD=0; R_BAD=0
if refused "${TMP}/r_legacy" "${PS_NO_LEGACY_POINTS}" --ps-point-conventions legacy; then R_LEG=1; fi
# (the older spelling: the program says once that it is the older name, then refuses the value;
# tests/dev/deprecation_gate.sh allows that line in this log)
if refused "${TMP}/r_pts_legacy" "${PS_NO_LEGACY_POINTS}" --pts-conventions legacy; then R_OLD=1; fi
if refused "${TMP}/r_bad" "Unknown value 'old' for '--ps-point-conventions'; the only value is 'grid'." \
        --ps-point-conventions old; then R_BAD=1; fi

echo ">> checks"
PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
"${PY}" - "${TMP}" "${GRID}" "${ALPHA}" "${AMP}" "${BOX}" "${BIN}" "${R_LEG}" "${R_OLD}" "${R_BAD}" <<'PY'
import os, sys
import numpy as np
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
tmp, g, alpha, amp, box, binary = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
r_leg, r_old, r_bad = (a == "1" for a in sys.argv[7:10])
fails = []

def check(tag, ok, msg):
    print(f"   {'OK  ' if ok else 'FAIL'} {tag} {msg}")
    if not ok:
        fails.append(tag)

def pts(root, ext, dtype=np.float64):
    return np.fromfile(f"{tmp}/{root}.pts_{ext}", dtype=dtype)

def outputs(root):
    return sorted(f[len(root) + 1:] for f in os.listdir(tmp)
                  if f.startswith(root + ".") and not f.endswith((".log", ".runlog")) and ".runlog" not in f)

def same_files(r1, r2, select):
    names = [e for e in outputs(r1) if select(e)]
    diff = [e for e in names if open(f"{tmp}/{r1}.{e}", "rb").read() != open(f"{tmp}/{r2}.{e}", "rb").read()]
    return names, diff

# ---- A
names, diff = same_files("a_on_grid", "a_on_named", lambda e: True)
check("A1", len(names) > 10 and not diff, f"--ps-vertex-mass --ps-volume-weighted: '--ps-point-conventions grid' named "
      f"and not named byte-identical in all {len(names)} outputs (differ: {diff})")

# ---- B
def slope(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / np.dot(b, b))
def slab_blocks(root):
    p = pts(root, "den").reshape(4 * g, 4 * g, 4)
    return p.reshape(g, 4, g, 4, 4).mean(axis=(1, 3, 4)) - 1.0
def grid_layer(root):
    return np.fromfile(f"{tmp}/{root}.a_den", dtype=REAL).astype(np.float64).reshape(g, g, g)[:, :, g // 2] - 1.0
# analytic cell averages of the separable crossed waves (x = q + A sin kq per axis, single stream): the fraction
# of Lagrangian length mapped into each cell, per axis (the jitter of the lattice aside)
k = 2. * np.pi / box
A = amp / k
def q_of(x):
    lo, hi = x - 2. * A - 1., x + 2. * A + 1.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid + A * np.sin(k * mid) < x else (lo, mid)
    return 0.5 * (lo + hi)
qe = np.array([q_of(x) for x in np.arange(g + 1) * box / g])
f1 = np.diff(qe) / (box / g)
ana = np.einsum("i,j->ij", f1, f1) * f1[g // 2] - 1.0
for est in ("linear",):
    grid = grid_layer(f"b_{est}_grid")
    sg, ag = slope(slab_blocks(f"b_{est}_grid"), grid), slope(slab_blocks(f"b_{est}_grid"), ana)
    sl, al = slope(slab_blocks(f"b_{est}_nomass"), grid), slope(slab_blocks(f"b_{est}_nomass"), ana)
    st = slope(slab_blocks(f"b_{est}_grid"), slab_blocks(f"b_{est}_true"))
    print(f"   {est}: plane-vs-grid slope {sg:.4f} (without --ps-vertex-mass {sl:.4f}); vs analytic {ag:.4f} "
          f"({al:.4f}); plane vs the true-IC plane {st:.4f}; exact grid vs analytic {slope(grid, ana):.4f}")
    check(f"B1 {est}", abs(sg - 1.) < 0.05 and abs(ag - 1.) < 0.05,
          f"the plane's contrast is the grid's (slope {sg:.3f}) and the true one ({ag:.3f})")
    # the control (the waves are not linear, contrast up to ~3, which deepens the suppression a little beyond 1-alpha)
    check(f"B2 {est}", abs(sl - (1. - alpha)) < 0.08 and abs(al - (1. - alpha)) < 0.08 and sg - sl > 0.12,
          f"without --ps-vertex-mass the plane carries the evolved-IC suppression, ~1-alpha = {1. - alpha:.3f} "
          f"(slopes {sl:.3f}, {al:.3f})")
    check(f"B3 {est}", abs(st - 1.) < 0.02, f"the perturbed-IC plane equals the true-IC plane (slope {st:.4f})")
gs = grid_layer("b_sampled_grid")
print(f"   (the nSub=3 sampled deposit of this box: plane-vs-grid slope "
      f"{slope(slab_blocks('b_sampled_grid'), gs):.3f}; "
      f"that grid against the analytic cells {slope(gs, ana):.3f}, residual rms {np.std(gs - ana):.3f} -- "
      f"the exact deposit's {np.std(grid_layer('b_linear_grid') - ana):.3f})")

# ---- C
for est in ("linear",):
    full = pts(f"c_{est}_full", "den")
    rnd = pts(f"c_{est}_rand", "den")
    cells = np.fromfile(f"{tmp}/c_{est}_full.a_den", dtype=REAL).astype(np.float64)
    se = rnd.std() / np.sqrt(rnd.size)
    check(f"C1 {est}", abs(rnd.mean() - 1.) < max(5. * se, 1e-3),
          f"2e5 random points average {rnd.mean():.5f} (rho_bar = 1; standard error {se:.1e})")
    check(f"C2 {est}", full.std() < 0.08 and full.std() < 1.2 * cells.std(),
          f"point noise std {full.std():.4f} on the near-lattice (the vertex-share grid's own: {cells.std():.4f} per cell)")

# ---- D: the weights at the points, from the per-stream records of the one run (density, vx, vy, vz per stream)
off = pts("d_grid", "stream_offsets", np.uint64).astype(np.int64)
rec = pts("d_grid", "stream_records").reshape(-1, 4)
vel = pts("d_grid", "vel").reshape(-1, 3)
disp = pts("d_grid", "velDisp").reshape(-1, 6)            # xx xy xz yy yz zz
n = np.diff(off)
idx = np.repeat(np.arange(n.size), n)
def stream_sum(x):          # per point, the sum over its streams of x (one row per stream record)
    out = np.zeros((n.size,) + x.shape[1:])
    np.add.at(out, idx, x)
    return out
covered, multi = n > 0, n > 1
v, w = rec[:, 1:], rec[:, 0]
plain = stream_sum(v)[covered] / n[covered][:, None]
W = stream_sum(w)
wmean = stream_sum(v * w[:, None]) / np.where(W > 0, W, 1.)[:, None]
vscale = float(np.abs(v).max())
err = float(np.abs(vel[covered] - plain).max() / vscale)
# (the difference the weighting makes, relative to the peak of the density-weighted mean)
seen =float(np.abs(wmean[multi] - vel[multi]).max() / np.abs(wmean[covered]).max()) if multi.any() else 0.
check("D1", err < 1e-12 and multi.any() and seen > 1e-3,
      f"'.pts_vel' is the plain mean of the stream velocities: max difference {err:.1e} of the peak stream velocity "
      f"over {int(covered.sum())} points ({int(multi.sum())} multi-stream, where the density-weighted mean is "
      f"another value: up to {seen:.3f} of its peak)")
pairs = [(a, b) for a in range(3) for b in range(a, 3)]
m2 = stream_sum(np.stack([w * v[:, a] * v[:, b] for a, b in pairs], 1)) / np.where(W > 0, W, 1.)[:, None]
expect = np.stack([m2[:, k] - wmean[:, a] * wmean[:, b] for k, (a, b) in enumerate(pairs)], 1)
for k, (a, b) in enumerate(pairs):          # a variance: FP-noise negatives clamped to 0, as the program does
    if a == b:
        expect[:, k] = np.maximum(expect[:, k], 0.)
expect[~multi] = 0.                          # one stream: no dispersion
derr = float(np.abs(disp - expect).max() / vscale ** 2)
check("D2", derr < 1e-12 and float(np.abs(disp[multi]).max()) > 0. and float(np.abs(disp[~multi]).max()) == 0.,
      f"'.pts_velDisp' is the density-weighted second moment about the density-weighted mean (mass-weighted, as in "
      f"the grids): max difference {derr:.1e} of the peak stream velocity squared; 0 at the single-stream points")

# ---- E
worst = 0.
for e, dt in (("den", np.float64), ("vel", np.float64), ("velDisp", np.float64), ("velGrad", np.float64)):
    a, b = pts("e_single", e, dt), pts("e_part", e, dt)
    worst = max(worst, float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)))
same_s = np.array_equal(pts("e_single", "streams", np.int32), pts("e_part", "streams", np.int32))
tol = 1e-12 if REAL == np.float64 else 2e-5
check("E1", same_s and worst < tol, f"--partition 2 2 2 vs one triangulation: streams identical, values within "
      f"{worst:.1e} of their peak (float rounding of the periodic copies; at most {tol:.0e})")

# ---- F
from dtfelib import Estimator
P = np.fromfile(f"{tmp}/in_plane.bin").reshape(-1, 3)
prec = "double" if REAL == np.float64 else "single"
# (the Estimator passes no '--ps-volume-weighted': its stream means are the volume-weighted ones of the run with it)
with Estimator(f"{tmp}/in_crw.hdf5", mpc_unit=1, per_stream=True, density_gradient=True, velocity_gradient=True,
               binary=os.path.abspath(binary), precision=prec, options=["--ps-vertex-mass"]) as est:
    f = est(P)
    ok = (np.array_equal(f.density, pts("d_both_grid", "den"))
          and np.array_equal(f.velocity.ravel(), pts("d_both_grid", "vel"))
          and np.array_equal(f.dispersion.ravel(), pts("d_both_grid", "velDisp"))
          and np.array_equal(f.velocity_gradient.ravel(), pts("d_both_grid", "velGrad"))
          and np.array_equal(f.density_gradient.ravel(), pts("d_both_grid", "denGrad"))
          and np.array_equal(np.column_stack([f.stream_density, f.stream_velocity]).ravel(),
                             pts("d_both_grid", "stream_records")))
    g2 = est(P[::397])          # a small request: the server's cell index, the same vertex ratios
    ok2 = np.array_equal(g2.density, f.density[::397]) and np.array_equal(g2.velocity, f.velocity[::397])
check("F1", ok and ok2, "--serve with --ps-vertex-mass alone: bit-identical to --sample-points with --ps-vertex-mass "
      "--ps-volume-weighted (full walk and a small request)")

# ---- R
check("R1", r_leg, f"'--ps-point-conventions legacy' stops the run with its message, no point file ({tmp}/r_legacy.log)")
check("R2", r_old, f"... so does the older spelling '--pts-conventions legacy' ({tmp}/r_pts_legacy.log)")
check("R3", r_bad, f"... and another value, with the one value left ('grid') ({tmp}/r_bad.log)")

print("-" * 60)
if fails:
    print("RESULT: FAIL (" + ", ".join(fails) + ")")
    sys.exit(1)
print("RESULT: PASS")
PY
echo "============================================================"
echo " point conventions check PASSED"
echo "============================================================"
