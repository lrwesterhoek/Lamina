#!/usr/bin/env bash
# Golden and analytic checks of the standard binary's sampling variants that no other suite runs: the volume
# averages '--average-method 2' (random points in each grid cell, seeded) and '--average-method 3' (equidistant points), the
# redshift-cone grid ('--redshift-cone', unaveraged and averaged), the regular grid's point values, and user-given
# sampling points (unaveraged and averaged; only the library API reaches them: tests/dtfe_user_sampling.cpp, built
# against libDTFE when 'make library' has made one). Upstream code (averaged_interpolation_2.cc,
# unaveraged_interpolation.cc), checked as it is, so that a later change to it can be held to these numbers.
#
# A seeded synthetic snapshot: 4096 particles uniform in a 10 Mpc box (an LCG, so the positions do not depend on
# numpy), NOT periodic, with a LINEAR velocity field v = A x + b. The DTFE interpolant of a linear field is the field
# itself inside the particles' hull, so every velocity is known in closed form; the densities are checked against
# other code paths or against stored values. Everything is sampled inside the region [2.5, 7.5]^3 (well inside the
# hull); '--no-padding-test' everywhere, since '--sample-points' turns the padding test's dummy points off and they move the
# densities near the hull. OMP_NUM_THREADS=1 on every run: method 2 adds 1000 x the sub-domain number to its seed,
# and more threads split a regular grid into sub-domains.
#   S1  the regular grid's point values: velocity = A x + b and gradient = A at the cell centres; density = the
#       '--sample-points' evaluation at the same points (exact point location, another code path)
#   S2  method 3, 3^3 equidistant points: density_a = the mean of the 27 point values, i.e. of the 3x finer grid's
#       '.den' over each cell; velocity_a = A x_c + b (the points are symmetric about the centre); gradient_a = A
#   S3  method 2, 20 random points: gradient_a = A; the samples' mean position, A^-1 (velocity_a - b), lies inside
#       its own cell and scatters about the centre by dx / sqrt(12 x 20) per axis (within 40%); density_a against
#       '--exact-average' (Monte-Carlo noise only: rms 0.082 of the mean, 0.04-0.16 accepted); velocityStd_a's mean
#       square = the linear field's expectation |A|_F^2 dx^2 / 12 x 19/20 (within 10%). These hold for any
#       realization: the regular grid's samples are drawn as Point( uni()*dx + x, uni()*dy + y, uni()*dz + z ),
#       three calls in one argument list, whose order C++ leaves to the compiler (GCC and Clang differ), so its
#       realization is NOT golden
#   S4  the redshift cone, point values: density = '--sample-points' at the cone cell centres (computed here from
#       the cone's definition), velocity = A x + b, gradient = A
#   S5  the redshift cone, averaged (method 2, the only one it has): gradient_a = A, the samples' mean position
#       within the cone cell's extent around its centre
#   S6  user sampling points (libDTFE): point values as S4; averaged (method 2 in a cube of the given size about
#       each point): gradient_a = A, mean positions inside the cube, scatter as in S3
#   S7  GOLDEN: the seeded realizations of the cone's and the user points' density_a and velocity_a (their loops
#       draw the coordinates in separate statements, so the order is fixed) against
#       tests/reference/sampling_golden.txt (the single build's), 1e-4 relative per value (of the largest value for
#       the vector fields): the single build repeats it to the print rounding (5e-8), the double build sits 7e-7
#       from it, so another platform's rounding fits; the stream itself is fixed by '--seed' and Boost's mt19937,
#       whose sequence is specified
# The user-sampling part (S6, S7's user values) needs libDTFE (single precision): without it, or in the double pass,
# it is a SKIP line and the rest still runs; '--require-library' makes a missing library a failure.
# About 3 s on the Mac (six runs of 4096 particles), plus ~10 s to build the user-sampling driver the first time.
# Usage: tests/dtfe_sampling_check.sh [--no-build] [--require-library] [--update-reference]
#   --no-build           accepted and ignored: nothing of the binaries is built (the driver is, when it is stale)
#   --update-reference   rewrite tests/reference/sampling_golden.txt from this run (single precision, with libDTFE)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
REQUIRE_LIB=0; UPDATE=0
for a in "$@"; do
    case "$a" in
        --no-build) ;;
        --require-library) REQUIRE_LIB=1 ;;
        --update-reference) UPDATE=1 ;;
        *) echo "unknown option $a"; exit 2 ;;
    esac
done
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${DTFE_BIN}"
[ -x "${DTFE_BIN}" ] || { echo "FAIL: ${DTFE_BIN} is not built"; exit 1; }
TMP="${DTFE_TEST_TMP}/dtfe_sampling_check"
rm -rf "$TMP"; mkdir -p "$TMP"
REF="tests/reference/sampling_golden.txt"
SEED=4242

# the snapshot, the point lists and the user samples
"$PY" - "$TMP" <<'PYEOF' || { echo "FAIL: could not write the test snapshot"; exit 1; }
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
from generate_ps_test_data import write_snapshot
from dtfe_sampling_helpers import A, B, BOX, N, REGION, GRID, cone_centres, user_samples
tmp = sys.argv[1]
state, x = 20261107, []
for _ in range(3 * N):                       # the LCG of tests/run_tests.py
    state = (1664525 * state + 1013904223) % 2 ** 32
    x.append(state / 2 ** 32 * BOX)
x = np.array(x, dtype=np.float32).reshape(N, 3).astype(np.float64)
v = x @ A.T + B
write_snapshot(os.path.join(tmp, "snap.hdf5"), x, x, v, BOX, 1.0)
lo, hi = REGION
c1 = lo + (np.arange(GRID) + 0.5) * (hi - lo) / GRID
gx, gy, gz = np.meshgrid(c1, c1, c1, indexing="ij")
grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
us = user_samples()
np.savetxt(os.path.join(tmp, "pts.txt"), np.vstack([grid_pts, cone_centres(), us[:, :3]]))
np.savetxt(os.path.join(tmp, "user.txt"), us)
PYEOF

# method 2's 20 random points per cell: a total that no '--subsamples k' (k^3) gives, and S7's golden values are
# drawn with it, so these runs keep the deprecated total '--samples 20' (one warning line each; P6.6 decides)
C=(--input-format 105 --mpc-unit 1 --no-padding-test --region-mpc 2.5 7.5 2.5 7.5 2.5 7.5 --verbose 1)
CONE=(--grid 4 4 6 --redshift-cone 0.5 2.0 30 150 0 300 --cone-origin 5 5 5)
# run <name> <program> [args...]: one run on one thread, its log beside it
run() {
    local name="$1" prog="$2"; shift 2
    OMP_NUM_THREADS=1 "$prog" "$TMP/snap.hdf5" "$TMP/$name" "${C[@]}" "$@" > "$TMP/$name.log" 2>&1 \
        || { echo "FAIL: run $name (see $TMP/$name.log)"; tail -5 "$TMP/$name.log"; exit 1; }
}
echo ">> the regular grid (point values + '--sample-points'), method 2, method 3, the 3x finer grid, --exact-average"
run u "${DTFE_BIN}" --grid 8 --field density velocity gradient density_a velocity_a gradient_a velocityStd_a \
    --average-method 2 --samples 20 --seed "$SEED" --sample-points "$TMP/pts.txt"
run m3 "${DTFE_BIN}" --grid 8 --field density_a velocity_a gradient_a --average-method 3 --subsamples 3
run f "${DTFE_BIN}" --grid 24 --field density
run x "${DTFE_BIN}" --grid 8 --field density_a --exact-average
echo ">> the redshift cone (point values and method 2)"
run c "${DTFE_BIN}" "${CONE[@]}" --field density velocity gradient density_a velocity_a gradient_a --average-method 2 --samples 20 --seed "$SEED"

# user sampling: the driver against libDTFE (single precision only: the library is built single)
USER_OK=0; USER_WHY=""
LIB=""; for l in libDTFE.dylib libDTFE.so; do [ -f "$l" ] && LIB="$l"; done
if precision_double; then
    USER_WHY="the double pass (libDTFE is built in single precision)"
elif [ -z "$LIB" ]; then
    USER_WHY="no libDTFE (make library builds it)"
else
    DRV="$(cd "${DTFE_TEST_TMP}" && pwd)/dtfe_user_sampling"   # absolute: make runs in demo/examples
    if [ ! -x "$DRV" ] || [ "$DRV" -ot "$LIB" ] || [ "$DRV" -ot tests/dtfe_user_sampling.cpp ]; then
        echo ">> building the user-sampling driver against $LIB (demo/examples/Makefile's flags)"
        make -s -C demo/examples -f Makefile -f ../../tests/dtfe_user_sampling.mk USER_SAMPLING_BIN="$DRV" "$DRV" \
            > "$TMP/driver_build.log" 2>&1 || { echo "FAIL: building the user-sampling driver (see $TMP/driver_build.log)"; tail -5 "$TMP/driver_build.log"; exit 1; }
    fi
    echo ">> user sampling points (libDTFE: point values and method 2)"
    OMP_NUM_THREADS=1 "$DRV" "$TMP/user.txt" "$TMP/snap.hdf5" "$TMP/us" "${C[@]}" --grid 1 \
        --field density velocity gradient density_a velocity_a gradient_a --average-method 2 --samples 20 --seed "$SEED" > "$TMP/us.log" 2>&1 \
        || { echo "FAIL: the user-sampling run (see $TMP/us.log)"; tail -5 "$TMP/us.log"; exit 1; }
    USER_OK=1
fi
if [ "$USER_OK" = 0 ]; then
    if [ "$REQUIRE_LIB" = 1 ]; then echo "FAIL (required): the user-sampling part cannot run: $USER_WHY"; exit 1; fi
    echo ">> S6 SKIP (user sampling points): $USER_WHY"
fi
if [ "$UPDATE" = 1 ] && { precision_double || [ "$USER_OK" = 0 ]; }; then
    echo "FAIL: --update-reference needs the single-precision pass with libDTFE"; exit 1
fi

"$PY" - "$TMP" "$USER_OK" "$UPDATE" "$REF" <<'PYEOF'
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
from dtfe_sampling_helpers import A, B, REGION, GRID, NS, CONE, cone_centres, user_samples
tmp, user_ok, update, ref_path = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", sys.argv[4]
REAL = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
fails = 0
def check(tag, ok, msg):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {tag} {msg}")
def rd(name, ext, comps=1):
    a = np.fromfile(os.path.join(tmp, f"{name}.{ext}"), dtype=REAL).astype(np.float64)
    return a.reshape(-1, comps) if comps > 1 else a
pts_den = np.fromfile(os.path.join(tmp, "u.pts_den"), dtype=np.float64)
Ainv = np.linalg.inv(A)
vscale = float(np.abs(rd("u", "vel", 3)).max())
gscale = float(np.abs(A).max())
lo, hi = REGION
dx = (hi - lo) / GRID
c1 = lo + (np.arange(GRID) + 0.5) * dx
gx, gy, gz = np.meshgrid(c1, c1, c1, indexing="ij")
xc = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)     # the cell centres, in the grids' order
ng = xc.shape[0]
cone_x = cone_centres()
nc = cone_x.shape[0]
us = user_samples()
# measured on the Mac (single; the double build is closer): velocities 8e-8, gradients 8e-6 (a thin tetrahedron's
# rounding), densities against the point evaluation 2e-6 and method 3 against the finer grid 5e-6
TOL_V, TOL_G, TOL_D = 1e-6, 1e-4, 2e-5    # relative to the largest |v|, to max|A|, per value

def vel_check(tag, what, v, x):
    err = float(np.abs(v - (x @ A.T + B)).max()) / vscale
    check(tag, err < TOL_V, f"{what}: velocity = A x + b to {err:.1e} of max|v| (< {TOL_V:.0e})")
def grad_check(tag, what, g):
    err = float(np.abs(g - A.ravel()[None, :]).max()) / gscale
    check(tag, err < TOL_G, f"{what}: gradient = A to {err:.1e} of max|A| (< {TOL_G:.0e})")
def den_check(tag, what, d, ref):
    rel = np.abs(d - ref) / np.abs(ref)
    check(tag, d.size == ref.size and float(rel.max()) < TOL_D,
          f"{what}: {d.size} densities against the point evaluation, largest difference {rel.max():.1e} (< {TOL_D:.0e})")
def mean_check(tag, what, va, centre, half, spread=None):
    xbar = (va - B) @ Ainv.T                      # the samples' mean position, from the linear field's mean
    off = xbar - centre
    inside = bool((np.abs(off) <= half * (1 + 1e-4)).all())
    msg = f"{what}: the samples' mean positions lie within the sampling cells ({inside})"
    ok = inside
    if spread is not None:
        ratio = float(np.sqrt((off ** 2).mean()) / spread)
        ok = ok and 0.6 < ratio < 1.4
        msg += f", scatter {ratio:.2f} x the expected {spread:.3g} (0.6-1.4)"
    check(tag, ok, msg)

# S1 the regular grid's point values
vel_check("S1", "grid points", rd("u", "vel", 3), xc)
grad_check("S1", "grid points", rd("u", "velGrad", 9))
den_check("S1", "grid points", rd("u", "den"), pts_den[:ng])
# S2 method 3
fine = rd("f", "den").reshape(3 * GRID, 3 * GRID, 3 * GRID)
block = fine.reshape(GRID, 3, GRID, 3, GRID, 3).mean(axis=(1, 3, 5)).ravel()
m3 = rd("m3", "a_den")
rel = np.abs(m3 - block) / np.abs(block)
check("S2", float(rel.max()) < TOL_D, f"method 3: density_a = the 3x finer grid's mean over each cell to {rel.max():.1e} (< {TOL_D:.0e})")
vel_check("S2", "method 3", rd("m3", "a_vel", 3), xc)
grad_check("S2", "method 3", rd("m3", "a_velGrad", 9))
# S3 method 2
grad_check("S3", "method 2", rd("u", "a_velGrad", 9))
mean_check("S3", "method 2", rd("u", "a_vel", 3), xc, dx / 2, dx / np.sqrt(12 * NS))
m2, ex = rd("u", "a_den"), rd("x", "a_den")
rms = float(np.sqrt(((m2 - ex) ** 2).mean()) / ex.mean())
# velocityStd_a of a linear field: sqrt of the mean |A (x_i - xbar)|^2 over the 20 samples, whose expectation is
# |A|_F^2 dx^2 / 12 x 19/20 (uniform in the cell); the cells' mean square within 10% of it
vs = rd("u", "a_velStd")
vratio = float((vs ** 2).mean() / ((A ** 2).sum() * dx ** 2 / 12 * (NS - 1) / NS))
check("S3", 0.9 < vratio < 1.1 and bool(np.isfinite(vs).all()),
      f"method 2: velocityStd_a, the cells' mean square {vratio:.3f} x the linear field's expectation (0.9-1.1)")
check("S3", 0.04 < rms < 0.16, f"method 2: density_a against --exact-average, rms {rms:.3f} of the mean (Monte-Carlo noise: 0.04-0.16)")
# S4 / S5 the redshift cone
den_check("S4", "cone points", rd("c", "den"), pts_den[ng:ng + nc])
vel_check("S4", "cone points", rd("c", "vel", 3), cone_x)
grad_check("S4", "cone points", rd("c", "velGrad", 9))
grad_check("S5", "cone, method 2", rd("c", "a_velGrad", 9))
(r0, r1, nr), (t0, t1, nt), (p0, p1, npsi) = CONE
cell = np.array([(r1 - r0) / nr, (r1 * np.radians((t1 - t0) / nt)), (r1 * np.radians((p1 - p0) / npsi))])
mean_check("S5", "cone, method 2", rd("c", "a_vel", 3), cone_x, np.full(3, float(np.linalg.norm(cell))))
# S6 user sampling
if user_ok:
    den_check("S6", "user points", rd("us", "den"), pts_den[ng + nc:])
    vel_check("S6", "user points", rd("us", "vel", 3), us[:, :3])
    grad_check("S6", "user points", rd("us", "velGrad", 9))
    grad_check("S6", "user points, method 2", rd("us", "a_velGrad", 9))
    mean_check("S6", "user points, method 2", rd("us", "a_vel", 3), us[:, :3], us[:, 3:] / 2,
               float(np.sqrt((us[:, 3:] ** 2).mean()) / np.sqrt(12 * NS)))
# S7 the golden realizations
golden = {"cone_density_a": rd("c", "a_den"), "cone_velocity_a": rd("c", "a_vel")}   # (gradient_a = A: S5, S6)
if user_ok:
    golden.update({"user_density_a": rd("us", "a_den"), "user_velocity_a": rd("us", "a_vel")})
RTOL = 1e-4
if update:
    with open(ref_path, "w") as f:
        f.write("# tests/dtfe_sampling_check.sh S7: the seeded sampling realizations (--seed 4242, 20 samples), "
                "single precision; one section per output (vector components interleaved), six values per line\n")
        for k, a in golden.items():
            f.write(f"{k} {a.size}\n")
            flat = a.ravel()
            for i in range(0, flat.size, 6):
                f.write(" ".join(f"{x:.7e}" for x in flat[i:i + 6]) + "\n")
    print(f"  UPDATED {ref_path} ({', '.join(f'{k} {a.size}' for k, a in golden.items())})")
else:
    ref, key = {}, None
    if not os.path.exists(ref_path):
        print(f"  FAIL S7 no reference {ref_path} (tests/dtfe_sampling_check.sh --update-reference writes one)")
        sys.exit(1)
    with open(ref_path) as f:
        for line in f:
            w = line.split()
            if not w or w[0].startswith("#"):
                continue
            if not w[0][0].isdigit() and w[0][0] not in "+-.":
                key = w[0]; ref[key] = []
            else:
                ref[key].extend(float(x) for x in w)
    for k, a in golden.items():
        a = a.ravel()
        r = np.array(ref.get(k, []))
        if r.size != a.size:
            check("S7", False, f"{k}: {a.size} values, the reference has {r.size}")
            continue
        # per value for the density; for a vector field against its largest component (a component crossing 0)
        scale = np.abs(r) if k.endswith("density_a") else np.full(r.size, float(np.abs(r).max()))
        rel = np.abs(a - r) / np.maximum(scale, 1e-30)
        check("S7", float(rel.max()) < RTOL, f"{k}: {a.size} values against {ref_path}, largest difference {rel.max():.1e} (< {RTOL:.0e})")
print("RESULT:", "PASS  (the sampling variants match their closed forms, the other code paths and the golden values)"
      if not fails else f"FAIL ({fails} check(s))")
sys.exit(1 if fails else 0)
PYEOF
