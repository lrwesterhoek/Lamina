#!/usr/bin/env bash
# The standard binary's '--face-gradient' (docs/cleanup-plan.md P10.2): the velocity gradient and the fields built on
# it from EXACT FACE INTEGRALS of the interpolant (Gauss: (1/V) times the integral of v n over a cell's faces is the
# cell average of grad v), instead of adding up each tetrahedron's gradient. For standard DTFE's interpolant -- linear
# in every tetrahedron, continuous across them -- the two routes agree to rounding; this suite pins that, and what
# the face route adds. The geometry itself is tests/face_sweep_check.sh's. Asserted on demo/z0_64.gadget (262144
# particles, z=0, tracked in the repo) at 64^3, --periodic --padding 12, density_a divergence_a shear_a:
#   G1  DTFE_FACE_CHECK=1: the face integrals equal the exact average's per-tetrahedron gradient in every cell to
#       1e-4 of the gradient's rms (measured at most 1.1e-5, rms 1.2e-7; the demo has no degenerate tetrahedron)
#   G2  the deposit threads, one domain (DTFE_DEPOSIT_THREADS 1, 3, 10): '.a_velDiv' and '.a_velShear' byte-identical
#   G3  the face gradient does not depend on the deposit: the Monte-Carlo average's and, in a GPU build, the --gpu
#       deposit's '.a_velDiv' and '.a_velShear' byte-identical to the exact CPU run's
#   G4  the final grids: face vs per-tetrahedron divergence and shear within 1e-4 of their rms (measured 4.4e-6 and
#       9.0e-6); a '--partition 2 2 2' split within 1e-4 of the single domain (measured 1.8e-6) with every
#       sub-domain's FACE-CHECK passing; the periodic box mean of the divergence below 1e-6 of its rms (8e-10)
#   G5  '--face-gradient' with '--cic' stops at option parsing with exit 1 and the error naming the DTFE method
#   G6  (a GPU build) the GPU exact average on '--partition 3 3 3' against one domain: byte-identical density and
#       divergence -- every tetrahedron's shares are formed in the grid frame (plan P10.8) and summed in fixed point,
#       so the split cannot move them -- and the shear within 1e-9 of its rms (measured 7e-13: the demo's flow is
#       v = 1000 x, so the off-diagonal gradient is a thousandth of the diagonal and sits at the last bits of the
#       fixed-point unit, which is the batch's, and of the face sums' cancellation); and every sub-domain's
#       FACE-CHECK against the GPU deposit within the full-size bound (rms_rel <= 3e-6 per record, above_1e-4
#       <= 1e-5 of the cells pooled)
# About 50 s on the Mac (9 runs of 2-7 s).
# Usage: tests/dtfe_face_gradient_check.sh [--no-build]   (--no-build is accepted and ignored: nothing is built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
precision_require "${DTFE_BIN}"
[ -x "${DTFE_BIN}" ] || { echo "FAIL: ${DTFE_BIN} is not built"; exit 1; }
SNAP="demo/z0_64.gadget"
[ -f "$SNAP" ] || { echo "FAIL: $SNAP is missing (it is tracked in the repository)"; exit 1; }
TMP="${DTFE_TEST_TMP}/dtfe_face_gradient_check"
rm -rf "$TMP"; mkdir -p "$TMP"
OBJ=o; precision_double && OBJ=o_d
GPU=0
[ -n "$(gpu_backend_of "$OBJ")" ] && GPU=1
FAIL=0
pass() { echo "   PASS  $*"; }
fail() { echo "   FAIL  $*"; FAIL=1; }

C=(--grid 64 --periodic --padding 12 --field density_a divergence_a shear_a)
# run <name> [env assignments ...] -- [extra args ...]
run() {
    local name="$1"; shift
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    env ${envs[@]+"${envs[@]}"} "${DTFE_BIN}" "$SNAP" "$TMP/$name" "${C[@]}" "$@" > "$TMP/$name.log" 2>&1 \
        || { echo "FAIL: run $name (see $TMP/$name.log)"; exit 1; }
}
same() { cmp -s "$TMP/$1.a_velDiv" "$TMP/$2.a_velDiv" && cmp -s "$TMP/$1.a_velShear" "$TMP/$2.a_velShear"; }

echo ">> demo/z0_64.gadget at 64^3: --exact-average with --per-tet-gradient and with --face-gradient (one domain, one thread)"
run e  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --exact-average --per-tet-gradient   # the deposit's own gradient (the face route is the default since 2026-11-09)
run f  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 DTFE_FACE_CHECK=1 -- --exact-average --face-gradient
line="$(grep -a 'FACE-CHECK' "$TMP/f.log" | head -1 | sed 's/.*FACE-CHECK/FACE-CHECK/')"
read -r mx above <<< "$(echo "$line" | awk '{for (i=1;i<=NF;i++) { if ($i=="max_rel") m=$(i+1); if ($i=="above_1e-4") a=$(i+1) } print m, a}')"
if [ -n "$line" ] && "$PY" -c "import sys; sys.exit(0 if float('$mx') <= 1e-4 and int('$above') == 0 else 1)"; then
    pass "G1 face integrals == the per-tetrahedron gradient in every cell ($line)"
else
    fail "G1 face integrals vs the per-tetrahedron gradient (${line:-no FACE-CHECK line})"
fi

echo ">> deposit threads 3 and 10 (one domain); the Monte-Carlo average; the GPU"
run f3  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=3  -- --exact-average --face-gradient
run f10 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=10 -- --exact-average --face-gradient
if same f f3 && same f f10; then pass "G2 1, 3 and 10 deposit threads: byte-identical divergence and shear"
else fail "G2 the deposit threads change the face gradient"; fi
run m  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --face-gradient
if same f m; then pass "G3 the Monte-Carlo average's face gradient == the exact average's (byte-identical)"
else fail "G3 the Monte-Carlo average changes the face gradient"; fi
if [ "$GPU" = "1" ]; then
    run g OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --exact-average --face-gradient --gpu
    if grep -q "using the CPU interpolation" "$TMP/g.log"; then fail "G3 the --gpu run fell back to the CPU (see $TMP/g.log)"
    elif same f g; then pass "G3 the GPU deposit's face gradient == the CPU's (byte-identical)"
    else fail "G3 the GPU deposit changes the face gradient"; fi
else
    echo "   (G3's GPU half: not a GPU build)"
fi

echo ">> --partition 2 2 2 (each sub-domain checked)"
run p DTFE_FACE_CHECK=1 -- --exact-average --face-gradient --partition 2 2 2 --max-concurrent 1
nchk=$(grep -ac 'FACE-CHECK' "$TMP/p.log")
nbad=$(grep -a 'FACE-CHECK' "$TMP/p.log" | sed 's/.*FACE-CHECK/FACE-CHECK/' \
       | awk '{for (i=1;i<=NF;i++) { if ($i=="max_rel" && $(i+1)+0 > 1e-4) b++; if ($i=="above_1e-4" && $(i+1)+0 > 0) b++ } } END {print b+0}')
"$PY" - "$TMP" "${DTFE_TEST_REAL}" "$nchk" "$nbad" <<'PY' || FAIL=1
import sys
import numpy as np
tmp, dt, nchk, nbad = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rd = lambda n: np.fromfile(f"{tmp}/{n}", dtype=dt).astype(np.float64)
ok = True
def rel(a, b):
    r = np.sqrt((b**2).mean())
    return np.abs(a - b).max() / r, r
d_fe, rms = rel(rd("f.a_velDiv"), rd("e.a_velDiv"))
s_fe, _ = rel(rd("f.a_velShear"), rd("e.a_velShear"))
d_p, _ = rel(rd("p.a_velDiv"), rd("f.a_velDiv"))
mean = abs(rd("f.a_velDiv").mean()) / rms
for good, what in ((d_fe <= 1e-4 and s_fe <= 1e-4,
                    f"G4 face vs per-tetrahedron grids: divergence {d_fe:.2g}, shear {s_fe:.2g} of their rms"),
                   (d_p <= 1e-4 and nchk >= 2 and nbad == 0,
                    f"G4 --partition 2 2 2 vs one domain: {d_p:.2g} of the rms; {nchk} sub-domain checks, {nbad} failing"),
                   (mean <= 1e-6, f"G4 periodic box mean of the divergence {mean:.2g} of its rms")):
    print(("   PASS  " if good else "   FAIL  ") + what)
    ok = ok and good
sys.exit(0 if ok else 1)
PY

echo ">> the GPU exact average on a 3^3 split against one domain (the deposit's grid frame, plan P10.8)"
if [ "$GPU" = "1" ]; then
    run g3 DTFE_FACE_CHECK=1 -- --exact-average --face-gradient --gpu --partition 3 3 3 --max-concurrent 1
    if grep -q "using the CPU interpolation" "$TMP/g3.log"; then fail "G6 the --gpu 3^3 run fell back to the CPU (see $TMP/g3.log)"
    # the double build keeps what float rounding hides: the GPU deposit is float32 (its grids are converted to float
    # once), so its density can differ by a float ulp where the two splits' fixed-point units round apart, and the
    # face sums' order shows in the divergence; there every grid is compared within 1e-7 of its rms instead
    elif DBL=0 && { ! precision_double || DBL=1; } && { precision_double || { cmp -s "$TMP/g.a_den" "$TMP/g3.a_den" && cmp -s "$TMP/g.a_velDiv" "$TMP/g3.a_velDiv"; }; } \
         && "$PY" -c "
import sys, numpy as np
ok = True
for f in ('a_den', 'a_velDiv', 'a_velShear'):
    a = np.fromfile('$TMP/g.' + f, dtype='${DTFE_TEST_REAL}').astype(np.float64); b = np.fromfile('$TMP/g3.' + f, dtype='${DTFE_TEST_REAL}').astype(np.float64)
    r = np.abs(a - b).max() / np.sqrt((a**2).mean()); print('   (G6 %s: 3^3 split vs one domain %.2g of its rms)' % (f, r)); ok = ok and r <= ($DBL and 1e-7 or 1e-9)
sys.exit(0 if ok else 1)"; then
        if precision_double; then pass "G6 GPU exact average, --partition 3 3 3 == one domain within 1e-7 of each grid's rms (double build: a float32 deposit)"
        else pass "G6 GPU exact average, --partition 3 3 3 == one domain: byte-identical density and divergence, shear within 1e-9"; fi
    else fail "G6 the 3^3 split moves the GPU exact average (the density, the divergence or the shear differ from the one-domain run)"; fi
    read -r nchk6 worst6 pooled6 <<< "$(grep -a 'FACE-CHECK' "$TMP/g3.log" | sed 's/.*FACE-CHECK/FACE-CHECK/' \
        | awk '{for (i=1;i<=NF;i++) { if ($i=="rms_rel" && $(i+1)+0 > w) w=$(i+1)+0; if ($i=="above_1e-4") a+=$(i+1); if ($i=="compared") c+=$(i+1) }; n++} END {printf "%d %.3g %.3g", n, w, (c>0 ? a/c : 1)}')"
    if [ "${nchk6:-0}" -ge 27 ] && "$PY" -c "import sys; sys.exit(0 if float('$worst6') <= 3e-6 and float('$pooled6') <= 1e-5 else 1)"; then
        pass "G6 FACE-CHECK against the GPU deposit on the 3^3 split: $nchk6 records, worst rms_rel $worst6, $pooled6 of the cells above 1e-4"
    else
        fail "G6 FACE-CHECK against the GPU deposit on the 3^3 split: $nchk6 records, worst rms_rel $worst6, $pooled6 of the cells above 1e-4 (bound 3e-6, 1e-5)"
    fi
else
    echo "   (G6: not a GPU build)"
fi

echo ">> refusal: --face-gradient --cic"
"${DTFE_BIN}" "$SNAP" "$TMP/c" --grid 16 --periodic --field density_a divergence_a --face-gradient --cic > "$TMP/c.log" 2>&1
rc=$?
if [ "$rc" = "1" ] && grep -q "'--face-gradient' needs the DTFE interpolation method" "$TMP/c.log"; then
    pass "G5 --face-gradient --cic stops at option parsing (exit 1, names the DTFE method)"
else
    fail "G5 --face-gradient --cic: exit $rc (see $TMP/c.log)"
fi

if [ "$FAIL" = "0" ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; exit 1; fi
