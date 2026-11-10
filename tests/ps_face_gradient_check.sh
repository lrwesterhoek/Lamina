#!/usr/bin/env bash
# PS-DTFE's '--face-gradient' (docs/cleanup-plan.md P10.3): the gradient of the volume-weighted MEAN velocity from exact
# face sums -- each cell's lower and upper face per axis carry the integrals of S (the sum of the streams' velocities)
# and of N (their count), and the gradient is the difference of the face-averaged mean velocities, (S_up/N_up -
# S_lo/N_lo) / dx -- instead of the stream average of each stream's own gradient, which the run writes beside it as
# '.a_velDivStreams'. On Zel'dovich pancakes (tests/generate_ps_test_data.py, 32^3 particles, box 100 Mpc, 20% jitter,
# 32^3 grid, --exact --ps-volume-weighted, density_a velocity_a divergence_a shear_a):
#   P1  single stream (amplitude factor 0.6): the face gradient equals the stream average to 1e-5 of the divergence's
#       rms (measured 9.5e-7), and '.a_velDivStreams' is byte-identical to the divergence of a run without the flag
#   P2  three streams across two folds (amplitude factor 2): the periodic box mean of the mean flow's divergence is 0 to
#       1e-6 of its rms (measured 5e-11; Gauss: the face form telescopes), while the stream average's is not (+0.98 of
#       its rms: the streams expand after shell crossing); in the cells whose stream count is an integer, and their six
#       neighbours' too (no caustic on a face), the two agree to 1e-4 of the rms (measured 1.5e-6)
#   P3  the folds carry the mean flow's convergence: the two fold planes' mean divergence is below -5 x the
#       single-stream region's mean and the two agree to 1e-3 of each other (the pancake is symmetric)
#   P4  determinism: 10 deposit threads and slab mode (DTFE_DEPOSIT_SLABS=1, 6 threads) byte-identical to one thread
#       (the double build's stream average to 1e-12: its float sums are per thread count, as every double-build moment)
#   P5  '--partition 2 2 2': within 1e-5 of the rms of one domain (measured 9e-7), box mean below 1e-6 of the rms
#   P6  '--grid-window' (one z cell across the box): the full run's z = 16 slice, byte-identical in the single build
#       (the double build shows the window's own fixed-point unit: within 1e-9 of the rms, measured 2e-11)
#   P7  refusals at option parsing (exit 1): with '--sampled' (the exact deposit is the default for these '_a' fields,
#       so only the sampled one asked for conflicts), without '--ps-volume-weighted'
#   P11 (a GPU build) '--gpu': the GPU exact deposit with the face sums from the CPU walk beside it (docs/cleanup-plan.md
#       D17): the mean flow's divergence and the mass flux divergence byte-identical to the CPU run's (the same face
#       words, the same units), the density within the GPU deposit's float tolerance, the stream average within 5e-2
#       of its rms (the device's float per-stream gradient), and the run neither fell back nor skipped the walk
#   P12 (a GPU build) the face walk keeps the tetrahedra the CPU deposit loop keeps (docs/cleanup-plan.md P11.7): a CPU
#       and a '--gpu' run write the mean flow's grids and the mass flux divergence byte for byte in a one-cell
#       '--grid-window', in a non-periodic '--box', and under the NaN test hook (DTFE_TEST_NAN_SHARE=50, single
#       build, with the same number of face shares left out). In a '--box' whose upper x face is a particle plane of
#       a lattice without jitter (tetrahedra outside the box with a face in that plane: neither device deposits
#       their section) the mass flux divergence is byte-identical, and the mean flow's gradient too below the last x
#       layer: that layer's upper face has no stream, so its x gradient stays the stream average, the device's own
#       float. In the single build the bounds a face pass takes over from the GPU extraction equal the bound walk's
#       (DTFE_DEBUG_FX_BOUNDS=1)
#   P8  dtfelib: FieldSet.face_gradient is True for the run (False without the flag) and load('divergence',
#       mean_field=True) returns the stored grid itself
#   P9  the face sums ride the checkpoint: a '--partition 2 2 2' run killed after 3 partitions
#       (DTFE_TEST_EXIT_AFTER_PARTITIONS=3, exit 137) and resumed ('--resume') writes the uninterrupted run's grids
#       byte for byte; '--ps-merge fixed-point' (the face sums as integer words across partitions) within 1e-5 of the
#       ordered merge (measured 9e-7), box mean below 1e-6 of the rms; a one-cell window over a 2^3 split equals the
#       full run's slice as P6 does
#   P10 the mass flux divergence '.a_massFluxDiv': periodic box mean 0 to 1e-6 of its rms; in the double build,
#       continuity over a drift of the 3-stream pancake and of crossed waves (24^3, amplitude factor 1.5: the y and z
#       momentum words) (x + v t, 9 times in [-1e-4, 1e-4]): each cell's mass change equals minus its outflow
#       integrated by Simpson's rule, caustic cells included, to 0.05 max / 2e-3 rms of the rms (measured 0.022 /
#       7.2e-4 and 0.028 / 7.1e-4), and 4 -> 8 intervals shrink the rms by at least 2.5 (measured 3.8 and 4.0); the
#       thread, slab-mode and resume checks (P4, P9) compare '.a_massFluxDiv' too
# About 25 s on the Mac in a GPU build, single or double (measured 23 s and 25 s).
# Usage: tests/ps_face_gradient_check.sh [--no-build]   (--no-build is accepted and ignored: nothing is built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
TMP="${DTFE_TEST_TMP}/ps_face_gradient_check"
rm -rf "$TMP"; mkdir -p "$TMP"
OBJ=o_ps; precision_double && OBJ=o_ps_d
GPU=0
[ -n "$(gpu_backend_of "$OBJ")" ] && GPU=1
FAIL=0
pass() { echo "   PASS  $*"; }
fail() { echo "   FAIL  $*"; FAIL=1; }

for amp in 0.6 2.0; do
    "$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$TMP/pan_$amp.hdf5" --n 32 --box 100 \
        --amplitude-factor "$amp" --jitter-frac 0.2 > "$TMP/gen_$amp.log" 2>&1 \
        || { echo "FAIL: generate_ps_test_data.py (see $TMP/gen_$amp.log)"; exit 1; }
done
C=(--grid 32 --periodic --input-format 105 --mpc-unit 1 --field density_a velocity_a divergence_a shear_a --exact)
# run <name> <input> [env assignments ...] -- [extra args ...]
run() {
    local name="$1" input="$2"; shift 2
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    env ${envs[@]+"${envs[@]}"} "${PS_BIN}" "$TMP/$input.hdf5" "$TMP/$name" "${C[@]}" "$@" > "$TMP/$name.log" 2>&1 \
        || { echo "FAIL: run $name (see $TMP/$name.log)"; exit 1; }
}
echo ">> pancakes at amplitude factors 0.6 (one stream) and 2 (three across two folds): with and without --face-gradient"
for amp in 0.6 2.0; do
    run "b$amp" "pan_$amp" OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --ps-volume-weighted
    run "f$amp" "pan_$amp" OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --ps-volume-weighted --face-gradient
done
echo ">> determinism (10 threads, slab mode), --partition 2 2 2, a one-cell --grid-window"
run t10  pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=10 -- --ps-volume-weighted --face-gradient
run slab pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=6 DTFE_DEPOSIT_SLABS=1 -- --ps-volume-weighted --face-gradient
run part pan_2.0 OMP_NUM_THREADS=1 -- --ps-volume-weighted --face-gradient --partition 2 2 2 --max-concurrent 2
run win  pan_2.0 OMP_NUM_THREADS=1 -- --ps-volume-weighted --face-gradient --grid-window 0.01 99.99 0.01 99.99 50.1 53.0
same() { cmp -s "$TMP/$1.a_velDiv" "$TMP/$2.a_velDiv" && cmp -s "$TMP/$1.a_velShear" "$TMP/$2.a_velShear" \
         && cmp -s "$TMP/$1.a_velDivStreams" "$TMP/$2.a_velDivStreams" && cmp -s "$TMP/$1.a_massFluxDiv" "$TMP/$2.a_massFluxDiv"; }
echo ">> a 2^3 split: killed after 3 partitions and resumed; --ps-merge fixed-point; a one-cell window over it"
P8=(--ps-volume-weighted --face-gradient --partition 2 2 2 --max-concurrent 2)
run full8 pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}"
DTFE_TEST_EXIT_AFTER_PARTITIONS=3 OMP_NUM_THREADS=1 "${PS_BIN}" "$TMP/pan_2.0.hdf5" "$TMP/res8" "${C[@]}" "${P8[@]}" \
    > "$TMP/res8a.log" 2>&1
rc_kill=$?
run res8 pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}" --resume
precision_double || run fxm pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}" --ps-merge fixed-point   # (not in the double build)
run winp pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}" --grid-window 0.01 99.99 0.01 99.99 50.1 53.0
if [ "$rc_kill" = "137" ] && grep -q -- "--resume: 3 of 8 partitions" "$TMP/res8.log" && same full8 res8 \
   && cmp -s "$TMP/full8.a_vel" "$TMP/res8.a_vel" && cmp -s "$TMP/full8.a_den" "$TMP/res8.a_den"; then
    pass "P9 killed after 3 of 8 partitions and resumed: the uninterrupted run's grids byte for byte"
else
    fail "P9 the resumed face run (kill exit $rc_kill; see $TMP/res8.log)"
fi
# the mean flow's grids come from fixed-point face sums alone: byte-identical in both builds. The stream average
# ('.a_velDivStreams') is a float sum per private copy in the double build (deterministic per thread count only, as
# every double-build moment): byte-identical in the single build, to 1e-12 of the rms in the double (measured 2.5e-15)
meanflow() { cmp -s "$TMP/$1.a_velDiv" "$TMP/$2.a_velDiv" && cmp -s "$TMP/$1.a_velShear" "$TMP/$2.a_velShear" \
             && cmp -s "$TMP/$1.a_massFluxDiv" "$TMP/$2.a_massFluxDiv"; }
if precision_double; then
    streams_ok=$("$PY" -c "import numpy as np; a=np.fromfile('$TMP/f2.0.a_velDivStreams'); b=np.fromfile('$TMP/t10.a_velDivStreams'); c=np.fromfile('$TMP/slab.a_velDivStreams'); r=np.sqrt((a**2).mean()); print(int(max(abs(a-b).max(), abs(a-c).max()) <= 1e-12*r))")
else
    streams_ok=0; same f2.0 t10 && same f2.0 slab && streams_ok=1
fi
if meanflow f2.0 t10 && meanflow f2.0 slab && [ "$streams_ok" = "1" ]; then
    pass "P4 10 threads and slab mode: the mean flow byte-identical to one thread (the stream average too, or to 1e-12 in double)"
else fail "P4 the thread count or slab mode changes the face gradient"; fi
if cmp -s "$TMP/f0.6.a_velDivStreams" "$TMP/b0.6.a_velDiv"; then
    pass "P1 '.a_velDivStreams' is the divergence a run without the flag stores (byte-identical)"
else fail "P1 '.a_velDivStreams' differs from the divergence of a run without the flag"; fi

"$PY" - "$TMP" "${DTFE_TEST_REAL}" <<'PY' || FAIL=1
import sys
import numpy as np
tmp, dt = sys.argv[1], sys.argv[2]
n = 32
rd = lambda name: np.fromfile(f"{tmp}/{name}", dtype=dt).astype(np.float64)
ok = True
def out(good, what):
    global ok
    print(("   PASS  " if good else "   FAIL  ") + what)
    ok = ok and good
# P1
b, f = rd("b0.6.a_velDiv"), rd("f0.6.a_velDiv")
rms = np.sqrt((b**2).mean())
e = np.abs(f - b).max() / rms
out(e <= 1e-5, f"P1 one stream: the face gradient == the stream average ({e:.2g} of the rms)")
# P2
b, f, s = rd("b2.0.a_velDiv"), rd("f2.0.a_velDiv"), rd("f2.0.a_streams")
rmsb, rmsf = np.sqrt((b**2).mean()), np.sqrt((f**2).mean())
mf, mb = abs(f.mean()) / rmsf, b.mean() / rmsb
out(mf <= 1e-6 and mb > 0.5, f"P2 periodic box mean: mean flow {mf:.2g} of its rms, stream average {mb:+.2f} of its rms")
s3 = s.reshape(n, n, n)
integ = np.abs(s3 - np.round(s3)) < 1e-5
calm = integ.copy()
for ax in range(3):
    for sh in (-1, 1):
        calm &= np.roll(integ, sh, axis=ax)
e = (np.abs(f - b).reshape(n, n, n)[calm].max() / rmsb) if calm.any() else 1.0
out(calm.mean() > 0.5 and e <= 1e-4, f"P2 cells without a caustic on a face ({calm.mean():.2f} of them): the two agree to {e:.2g} of the rms")
# P3: the fold planes (x layers with a non-integer stream count), against the single-stream layers
fx, sx = f.reshape(n, n, n).mean(axis=(1, 2)), s3.mean(axis=(1, 2))
folds = np.where(np.abs(sx - np.round(sx)) > 1e-3)[0]
single = np.abs(sx - 1.0) < 1e-6
good = len(folds) == 2 and single.any()
if good:
    a, c = fx[folds]
    ref = abs(fx[single].mean())
    good = a < -5 * ref and c < -5 * ref and abs(a - c) <= 1e-3 * abs(a)
    what = f"P3 the two folds (x = {folds[0]}, {folds[1]}): mean divergence {a:.1f}, {c:.1f} against the single-stream {fx[single].mean():+.1f}"
else:
    what = f"P3 expected two fold layers and a single-stream region (found {len(folds)} folds)"
out(good, what)
# P5
p = rd("part.a_velDiv")
e, m = np.abs(p - f).max() / rmsf, abs(p.mean()) / rmsf
out(e <= 1e-5 and m <= 1e-6, f"P5 --partition 2 2 2: {e:.2g} of the rms from one domain, box mean {m:.2g}")
# P6
w = rd("win.a_velDiv").reshape(n, n, -1)
ws = rd("win.a_velDivStreams").reshape(n, n, -1)
full, fulls = f.reshape(n, n, n), rd("f2.0.a_velDivStreams").reshape(n, n, n)
# the window's face words take the window's own fixed-point unit (as every fixed-point grid of a window run does): the
# single build's float output rounds that away (byte-identical); the double build shows it (measured 2e-11 of the rms)
e = np.abs(w[:, :, 0] - full[:, :, 16]).max() / rmsf if w.shape[2] == 1 else 1.0
good = w.shape[2] == 1 and np.array_equal(ws[:, :, 0], fulls[:, :, 16]) and \
       (np.array_equal(w[:, :, 0], full[:, :, 16]) if dt == "float32" else e <= 1e-9)
out(good, f"P6 a one-cell --grid-window vs the full run's z = 16 slice: {e:.2g} of the rms (float32: byte-identical)")
# P9: the fixed-point merge, and a window over a split
f8 = rd("full8.a_velDiv")
if dt == "float32":   # (the double build has no --ps-merge fixed-point)
    x = rd("fxm.a_velDiv")
    e, m = np.abs(x - f8).max() / rmsf, abs(x.mean()) / rmsf
    out(e <= 1e-5 and m <= 1e-6, f"P9 --ps-merge fixed-point vs the ordered merge: {e:.2g} of the rms, box mean {m:.2g}")
wp = rd("winp.a_velDiv").reshape(n, n, -1)
f8 = f8.reshape(n, n, n)
e = np.abs(wp[:, :, 0] - f8[:, :, 16]).max() / rmsf if wp.shape[2] == 1 else 1.0
good = wp.shape[2] == 1 and (np.array_equal(wp[:, :, 0], f8[:, :, 16]) if dt == "float32" else e <= 1e-9)
out(good, f"P9 a one-cell window over a 2^3 split vs that split's full run: {e:.2g} of the rms")
sys.exit(0 if ok else 1)
PY

echo ">> the mass flux: periodic box mean; continuity over a drift (the double build)"
# P10: '.a_massFluxDiv' = div(sum_s rho_s v_s) from the face momentum sums. Its periodic box mean telescopes to 0. The
# double build drifts the 3-stream pancake's particles (and, for the y and z momentum words, crossed waves: 24^3
# particles, amplitude factor 1.5, 20% jitter) along their velocities, x + v t, for t in [-T, T] at 9 times:
# a cell's mass change M(T) - M(-T) equals minus the time integral of its outflow, caustic cells too (the PS model is
# exact for each tetrahedron at each instant). Simpson's rule on 8 intervals converges (4 -> 8 intervals: rms x 3.8)
# but slowly where a caustic sheet crosses a face in the interval (the flux is not smooth in t there).
if precision_double; then
    "$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$TMP/cw.hdf5" --n 24 --box 100 --amplitude-factor 1.5 \
        --jitter-frac 0.2 --crossed-waves > "$TMP/gen_cw.log" 2>&1 \
        || { echo "FAIL: generate_ps_test_data.py (see $TMP/gen_cw.log)"; exit 1; }
    "$PY" - "$TMP" <<'PY' || { echo "FAIL: writing the drift snapshots"; exit 1; }
import shutil, sys
import h5py
import numpy as np
tmp = sys.argv[1]
T, K = 1e-4, 8
for src, tag in (("pan_2.0", "drift"), ("cw", "cwdrift")):
    with h5py.File(f"{tmp}/{src}.hdf5", "r") as f:
        x0 = f["PartType1/Coordinates"][...].astype(np.float64)
        v = f["PartType1/Velocities"][...].astype(np.float64)
    for k in range(K + 1):
        fn = f"{tmp}/{tag}_{k}.hdf5"
        shutil.copy(f"{tmp}/{src}.hdf5", fn)
        with h5py.File(fn, "r+") as f:
            del f["PartType1/Coordinates"]
            f.create_dataset("PartType1/Coordinates", data=x0 + v * (-T + k * 2 * T / K))
PY
    for tag in drift cwdrift; do for k in 0 1 2 3 4 5 6 7 8; do
        "${PS_BIN}" "$TMP/${tag}_$k.hdf5" "$TMP/${tag}_$k" --grid 32 --periodic --input-format 105 --mpc-unit 1 \
            --field density_a divergence_a --exact --ps-volume-weighted --face-gradient \
            > "$TMP/${tag}_$k.log" 2>&1 || { echo "FAIL: run ${tag}_$k (see $TMP/${tag}_$k.log)"; exit 1; }
    done; done
fi
"$PY" - "$TMP" "${DTFE_TEST_REAL}" <<'PY' || FAIL=1
import os, sys
import numpy as np
tmp, dt = sys.argv[1], sys.argv[2]
rd = lambda name: np.fromfile(f"{tmp}/{name}", dtype=dt).astype(np.float64)
ok = True
def out(good, what):
    global ok
    print(("   PASS  " if good else "   FAIL  ") + what)
    ok = ok and good
p = rd("f2.0.a_massFluxDiv")
rms = np.sqrt((p**2).mean())
m = abs(p.mean()) / rms
out(rms > 0 and m <= 1e-6, f"P10 periodic box mean of the mass flux divergence: {m:.2g} of its rms")
if os.path.exists(f"{tmp}/drift_8.a_den"):
    T, K = 1e-4, 8
    for tag, what in (("drift", "the pancake (x)"), ("cwdrift", "crossed waves (x, y, z)")):
        D = np.array([rd(f"{tag}_{k}.a_massFluxDiv") for k in range(K + 1)])
        rms = np.sqrt((D[K // 2]**2).mean())
        dm = rd(f"{tag}_{K}.a_den") - rd(f"{tag}_0.a_den")
        def simpson(m):   # Simpson's rule on m intervals of [-T, T] (every K/m-th sample)
            w = np.ones(m + 1); w[1:-1:2] = 4; w[2:-1:2] = 2
            return 2 * T / m / 3 * (w[:, None] * D[::K // m]).sum(0)
        e4, e8 = (np.abs(dm + simpson(m)) / (2 * T) / rms for m in (4, 8))
        r4, r8 = np.sqrt((e4**2).mean()), np.sqrt((e8**2).mean())
        out(e8.max() <= 0.05 and r8 <= 2e-3 and r8 <= r4 / 2.5,
            f"P10 continuity over the drift of {what}: mass change vs the integrated outflow, {e8.max():.2g} max / "
            f"{r8:.2g} rms of the rms (4 -> 8 intervals: rms x {r4 / r8:.1f})")
else:
    print("   (P10's drift: the double build only, its float64 positions resolve the drift)")
sys.exit(0 if ok else 1)
PY

echo ">> refusals"
refuse() {   # refuse <what> <expected message part> [args ...]
    local what="$1" msg="$2"; shift 2
    "${PS_BIN}" "$TMP/pan_0.6.hdf5" "$TMP/r" --grid 8 --periodic --input-format 105 --mpc-unit 1 --field density_a divergence_a "$@" \
        > "$TMP/r.log" 2>&1
    local rc=$?
    if [ "$rc" = "1" ] && grep -q -- "$msg" "$TMP/r.log"; then pass "P7 $what: exit 1 at option parsing"
    else fail "P7 $what: exit $rc (see $TMP/r.log)"; fi
}
refuse "with --sampled" "add '--exact' (or drop '--sampled')" --sampled --ps-volume-weighted --face-gradient
refuse "without --ps-volume-weighted" "add '--ps-volume-weighted'" --exact --face-gradient

if [ "$GPU" = "1" ]; then
    echo ">> the GPU exact deposit with the face sums from the CPU walk beside it (--gpu --face-gradient)"
    run gpu pan_2.0 OMP_NUM_THREADS=1 -- --ps-volume-weighted --face-gradient --gpu
    if ! grep -q " GPU (" "$TMP/gpu.log" || grep -q "using the CPU deposit" "$TMP/gpu.log"; then
        fail "P11 the --gpu run did not deposit on the GPU (see $TMP/gpu.log)"
    elif ! grep -q "face sums (--face-gradient) of the GPU-deposited" "$TMP/gpu.log"; then
        fail "P11 the --gpu run did not run the face walk (see $TMP/gpu.log)"
    else
        "$PY" - "$TMP" "$DTFE_TEST_REAL" <<'EOF' || FAIL=1
import numpy as np, sys
tmp, dt = sys.argv[1], np.dtype(sys.argv[2])
rd = lambda name: np.fromfile(f"{tmp}/{name}", dtype=dt).astype(np.float64)
ok = True
for suf, tol, what in (("a_velDiv", 0.0, "the mean flow's divergence"), ("a_massFluxDiv", 0.0, "the mass flux divergence")):
    c, g = rd(f"f2.0.{suf}"), rd(f"gpu.{suf}")
    r = np.sqrt((c ** 2).mean()) or 1.0
    d = float(np.abs(c - g).max()) / r
    if d <= tol: print(f"   PASS  P11 {what}: GPU exact + face walk == CPU exact + face ({d:.1e} of the rms; byte-identical)")
    else: print(f"   FAIL  P11 {what}: GPU exact + face walk differs from the CPU run by {d:.3g} of the rms"); ok = False
c, g = rd("f2.0.a_den"), rd("gpu.a_den")
scale = float(np.abs(c).max()) or 1.0
mean_rel, peak = abs(float(c.mean()) - float(g.mean())) / scale, float(np.abs(c - g).max()) / scale
if mean_rel <= 1e-5 and peak <= 5e-2: print(f"   PASS  P11 density: GPU exact vs CPU exact mean-rel {mean_rel:.1e}, peak-rel {peak:.1e}")
else: print(f"   FAIL  P11 density: GPU exact vs CPU exact mean-rel {mean_rel:.1e}, peak-rel {peak:.1e}"); ok = False
c, g = rd("f2.0.a_velDivStreams"), rd("gpu.a_velDivStreams")
r = np.sqrt((c ** 2).mean()) or 1.0
d = float(np.abs(c - g).max()) / r
if d <= 5e-2: print(f"   PASS  P11 the stream average's divergence (the device's float gradient): {d:.1e} of the rms")
else: print(f"   FAIL  P11 the stream average's divergence differs by {d:.3g} of the rms"); ok = False
sys.exit(0 if ok else 1)
EOF
    fi
else echo "   (P11's --gpu case: not a GPU build)"; fi

if [ "$GPU" = "1" ]; then
    echo ">> the face walk's filter against the deposit loop's (--gpu): a window, non-periodic boxes, the NaN hook"
    "$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$TMP/lat.hdf5" --n 32 --box 100 \
        --amplitude-factor 2.0 --jitter-frac 0 > "$TMP/gen_lat.log" 2>&1 \
        || { echo "FAIL: generate_ps_test_data.py (see $TMP/gen_lat.log)"; exit 1; }
    # the lattice's x range for '--box': the upper face is a particle plane (one float32 x per lattice plane) and, with
    # 28.125 = 32 x 0.87890625 exact in float and in double, grid plane 32 in both builds
    "$PY" - "$TMP/lat.hdf5" > "$TMP/edge.txt" 2> "$TMP/edge.log" <<'EOF' || { echo "FAIL: the lattice box's edge (see $TMP/edge.log)"; exit 1; }
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    x = np.unique(f["PartType1/Coordinates"][:, 0])
assert x.size == 32 and x.dtype == np.float32, (x.size, x.dtype)
hi = float(x[np.argmin(np.abs(x - 70.0))])
lo = hi - 28.125
assert float(np.float32(lo)) == lo and hi - lo == 28.125, (lo, hi)
print(repr(lo), repr(hi))
EOF
    read -r XLO XHI < "$TMP/edge.txt"
    B=(--grid 32 --input-format 105 --mpc-unit 1 --field density_a velocity_a divergence_a shear_a --exact
       --ps-volume-weighted --face-gradient)
    runb() { local C=("${B[@]}"); run "$@"; }   # 'run' without --periodic
    JB=(--box 35.9375 64.0625 35.9375 64.0625 35.9375 64.0625)
    LB=(--box "$XLO" "$XHI" 30 70 30 70)
    run  gwin pan_2.0 OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 -- --ps-volume-weighted --face-gradient \
        --grid-window 0.01 99.99 0.01 99.99 50.1 53.0 --gpu
    runb jc pan_2.0 OMP_NUM_THREADS=1 -- "${JB[@]}"
    runb jg pan_2.0 OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 -- "${JB[@]}" --gpu
    runb lc lat     OMP_NUM_THREADS=1 -- "${LB[@]}"
    runb lg lat     OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 -- "${LB[@]}" --gpu
    gpu_walked() {   # the run deposited on the GPU and ran the face walk beside it
        grep -q " GPU (" "$TMP/$1.log" && ! grep -q "using the CPU deposit" "$TMP/$1.log" \
            && grep -q "face sums (--face-gradient) of the GPU-deposited" "$TMP/$1.log"
    }
    bounds_ok() {    # single build: the extraction's bounds were taken over and equal the bound walk's
        precision_double || grep -q "13 bounds of the GPU extraction checked against the bound walk, 0 differ" "$TMP/$1.log"
    }
    gpu_run_ok() {   # gpu_run_ok <gpu run> <what>: says what is wrong with the run, if anything
        if ! gpu_walked "$1"; then
            fail "P12 $2: the --gpu run did not deposit on the GPU or skipped the face walk (see $TMP/$1.log)"; return 1
        elif ! bounds_ok "$1"; then
            fail "P12 $2: no bounds check line of DTFE_DEBUG_FX_BOUNDS (see $TMP/$1.log)"; return 1
        fi
    }
    pair() {         # pair <cpu run> <gpu run> <what>
        gpu_run_ok "$2" "$3" || return 0
        if meanflow "$1" "$2"; then pass "P12 $3: the mean flow's grids and the mass flux divergence, GPU == CPU byte for byte"
        else fail "P12 $3: the GPU run's mean flow grids differ from the CPU run's"; fi
    }
    pair win gwin "a one-cell --grid-window"
    pair jc  jg   "a non-periodic --box"
    if gpu_run_ok lg "a particle plane as the box face"; then
        "$PY" - "$TMP" "$DTFE_TEST_REAL" <<'EOF' || FAIL=1
import numpy as np, sys
tmp, dt = sys.argv[1], np.dtype(sys.argv[2])
n, ok = 32, True
flux = [np.fromfile(f"{tmp}/{k}.a_massFluxDiv", dtype=dt) for k in ("lc", "lg")]
rms = float(np.sqrt((flux[0].astype(np.float64) ** 2).mean()))
if rms > 0 and np.array_equal(flux[0], flux[1]):
    print("   PASS  P12 a particle plane as the box face: the mass flux divergence, GPU == CPU byte for byte")
else:
    d = float(np.abs(flux[0].astype(np.float64) - flux[1]).max()) / (rms or 1.0)
    print(f"   FAIL  P12 a particle plane as the box face: the mass flux divergence differs by {d:.3g} of the rms "
          f"(rms {rms:.3g})"); ok = False
for suf, nc in (("a_velDiv", 1), ("a_velShear", 5)):
    c, g = (np.fromfile(f"{tmp}/{k}.{suf}", dtype=dt).reshape(n, n, n, nc) for k in ("lc", "lg"))
    r = float(np.sqrt((c.astype(np.float64) ** 2).mean())) or 1.0
    last = float(np.abs(c[n - 1].astype(np.float64) - g[n - 1]).max()) / r
    if np.array_equal(c[: n - 1], g[: n - 1]) and last <= 1e-5:
        print(f"   PASS  P12 a particle plane as the box face: {suf} byte-identical below the last x layer "
              f"(that layer, the stream average along x: {last:.1e} of the rms)")
    else:
        below = float(np.abs(c[: n - 1].astype(np.float64) - g[: n - 1]).max()) / r
        print(f"   FAIL  P12 a particle plane as the box face: {suf} differs by {below:.3g} of the rms below the last x "
              f"layer, {last:.3g} in it"); ok = False
sys.exit(0 if ok else 1)
EOF
    fi
    if ! precision_double; then   # the hook's shares are fixed-point words: the single build (ps_fixed_moments_check.sh F5)
        run nc pan_2.0 OMP_NUM_THREADS=1 DTFE_TEST_NAN_SHARE=50 -- --ps-volume-weighted --face-gradient
        run ng pan_2.0 OMP_NUM_THREADS=1 DTFE_TEST_NAN_SHARE=50 -- --ps-volume-weighted --face-gradient --gpu
        drops() { sed -n 's/.*face mass flux (--face-gradient) grid \([0-9][0-9]*\).*/\1/p' "$TMP/$1.log" | head -1; }
        dc="$(drops nc)"; dg="$(drops ng)"
        if ! gpu_walked ng; then
            fail "P12 the NaN hook: the --gpu run did not deposit on the GPU or skipped the face walk (see $TMP/ng.log)"
        elif [ -n "$dc" ] && [ "$dc" -gt 0 ] && [ "$dc" = "$dg" ] && meanflow nc ng; then
            pass "P12 the NaN hook: $dc face shares left out on both devices, the mean flow's grids byte for byte"
        else
            fail "P12 the NaN hook: face shares left out CPU '${dc}' / GPU '${dg}', or the grids differ (see $TMP/ng.log)"
        fi
    fi
else echo "   (P12's --gpu cases: not a GPU build)"; fi

echo ">> dtfelib reads the stored mean-flow grids"
for k in f2.0 b2.0; do
    mkdir -p "$TMP/snap_$k"
    ln -sf "$TMP/pan_2.0.hdf5" "$TMP/snap_$k/combined_099.hdf5"
    for g in "$TMP"/$k.a_*; do ln -sf "$g" "$TMP/snap_$k/ps_output.${g##*/$k.}"; done
done
PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$TMP" "${DTFE_TEST_REAL}" <<'PY' || FAIL=1
import sys
import numpy as np
from dtfelib.io import FieldSet
tmp, dt = sys.argv[1], sys.argv[2]
fs, fb = FieldSet(f"{tmp}/snap_f2.0", method="ps"), FieldSet(f"{tmp}/snap_b2.0", method="ps")
d = fs.load("divergence", mean_field=True, scaled=False).ravel()
raw = np.fromfile(f"{tmp}/f2.0.a_velDiv", dtype=dt)
good = fs.face_gradient and not fb.face_gradient and np.array_equal(d, raw.astype(d.dtype))
print(("   PASS  " if good else "   FAIL  ") + f"P8 FieldSet.face_gradient {fs.face_gradient} / {fb.face_gradient}; "
      "load('divergence', mean_field=True) returns the stored grid")
sys.exit(0 if good else 1)
PY

if [ "$FAIL" = "0" ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; exit 1; fi
