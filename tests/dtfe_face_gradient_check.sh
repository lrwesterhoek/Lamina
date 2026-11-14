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
#   G3  the face gradient does not depend on the deposit: the Monte-Carlo average's ('--sampled') and, in a GPU build, the --gpu
#       deposit's '.a_velDiv' and '.a_velShear' byte-identical to the exact CPU run's
#   G4  the final grids: face vs per-tetrahedron divergence and shear within 1e-4 of their rms (measured 4.4e-6 and
#       9.0e-6); a '--partition 2 2 2' split within 1e-4 of the single domain (measured 1.8e-6) with every
#       sub-domain's FACE-CHECK passing; the periodic box mean of the divergence below 1e-6 of its rms (8e-10)
#   G5  '--face-gradient' with '--cic' stops at option parsing with exit 1 and the error naming the DTFE method
#   G6  (a GPU build) the GPU exact average on '--partition 3 3 3' against one domain: every output byte-identical
#       (cmp of every output file) -- every tetrahedron's shares are formed in the grid frame
#       (plan P10.8), from vertex densities summed exactly over each vertex's star and from single-cell volumes in
#       a vertex order of their own (canonicalTetVolume, not CGAL's labelling), and summed in the run's fixed-point
#       unit (src/fixed_units.h RunUnits), and the face pass, whose
#       sums replace the deposit's gradient (the 3^3 run g3 passes DTFE_FACE_CHECK=1, the one-domain g does not:
#       S3 of tests/split_bytes_check.sh pins that the check moves no byte), sums each face piece's flux as an
#       integer in its own run-wide unit, so the split moves no sum; the shear, the off-diagonal gradient of the
#       demo's v = 1000 x flow, sits at the last bits of the face sums' cancellation, which identical integers repeat.
#       The double build keeps its double face sums (walk order): there each grid is compared within 1e-7 of its rms.
#       And every sub-domain's FACE-CHECK against the GPU deposit within the full-size bound (rms_rel <= 3e-6 per
#       record, above_1e-4 <= 1e-5 of the cells pooled)
#   G7  (the single build) the order of the work does not reach the grids: the CPU exact average in a team of 2
#       sub-domains (OMP_NUM_THREADS=2) byte-identical to G2's one-domain run, and -- in a build with the parallel
#       insert (TBB), on a machine of 4 threads or more -- with each sub-domain inserted on 2 TBB threads
#       (DTFE_TRIANGULATION_THREADS=2: another cell order and labelling, from run to run) byte-identical to the
#       sequential insert, on the CPU and (a GPU build) on the GPU against G3's one-domain run: the vertex densities
#       are exact star sums and the single-cell volumes canonical (src/CGAL_triangulation/triangulation.cpp
#       vertexDensityStarSums, triangulation_miscellaneous.h canonicalTetVolume). The double build keeps its double
#       sums: SKIP there
#   G8  '--interlace' (two passes, the second on the particles moved by half a cell, of which only the density is
#       kept: main.cpp): G1's face run with it takes the face gradient (its header's 'velocity gradient' line) and
#       writes every output of the run without it byte for byte but '.a_den', which is interlaced
# About 65 s on the Mac (12 runs of 2-7 s), before G8's interlaced run (TO MEASURE: two passes of G1's run).
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
# differing <run a> <run b>: the outputs of a that are not byte-identical to b's, and those only b wrote (empty: every
# output identical); the logs and the run's bookkeeping are not outputs
differing() {
    local d="" n=0 f s
    for f in "$TMP/$1".*; do
        [ -e "$f" ] || continue
        s="${f#"$TMP/$1"}"
        case "$s" in .log|.runlog*|*.part.*) continue ;; esac
        n=$((n+1))
        cmp -s "$f" "$TMP/$2$s" || d="$d $s"
    done
    for f in "$TMP/$2".*; do
        [ -e "$f" ] || continue
        s="${f#"$TMP/$2"}"
        case "$s" in .log|.runlog*|*.part.*) continue ;; esac
        [ -e "$TMP/$1$s" ] || d="$d $s(only in $2)"
    done
    [ "$n" -gt 0 ] || d=" (run $1 wrote no output)"
    echo "$d"
}

echo ">> demo/z0_64.gadget at 64^3: --exact with --per-tet-gradient and with --face-gradient (one domain, one thread)"
run e  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --exact --per-tet-gradient   # the deposit's own gradient (the face route is the default since 2026-11-09)
run f  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 DTFE_FACE_CHECK=1 -- --exact --face-gradient
line="$(grep -a 'FACE-CHECK' "$TMP/f.log" | head -1 | sed 's/.*FACE-CHECK/FACE-CHECK/')"
read -r mx above <<< "$(echo "$line" | awk '{for (i=1;i<=NF;i++) { if ($i=="max_rel") m=$(i+1); if ($i=="above_1e-4") a=$(i+1) } print m, a}')"
if [ -n "$line" ] && "$PY" -c "import sys; sys.exit(0 if float('$mx') <= 1e-4 and int('$above') == 0 else 1)"; then
    pass "G1 face integrals == the per-tetrahedron gradient in every cell ($line)"
else
    fail "G1 face integrals vs the per-tetrahedron gradient (${line:-no FACE-CHECK line})"
fi

echo ">> deposit threads 3 and 10 (one domain); the Monte-Carlo average; the GPU"
run f3  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=3  -- --exact --face-gradient
run f10 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=10 -- --exact --face-gradient
if same f f3 && same f f10; then pass "G2 1, 3 and 10 deposit threads: byte-identical divergence and shear"
else fail "G2 the deposit threads change the face gradient"; fi
run m  OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --sampled --face-gradient   # (the exact average is the default since 2026-11-09)
if same f m; then pass "G3 the Monte-Carlo average's face gradient == the exact average's (byte-identical)"
else fail "G3 the Monte-Carlo average changes the face gradient"; fi
if [ "$GPU" = "1" ]; then
    run g OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --exact --face-gradient --gpu
    if grep -q "using the CPU interpolation" "$TMP/g.log"; then fail "G3 the --gpu run fell back to the CPU (see $TMP/g.log)"
    elif same f g; then pass "G3 the GPU deposit's face gradient == the CPU's (byte-identical)"
    else fail "G3 the GPU deposit changes the face gradient"; fi
else
    echo "   (G3's GPU half: not a GPU build)"
fi

echo ">> --partition 2 2 2 (each sub-domain checked)"
run p DTFE_FACE_CHECK=1 -- --exact --face-gradient --partition 2 2 2 --max-concurrent 1
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
    run g3 DTFE_FACE_CHECK=1 -- --exact --face-gradient --gpu --partition 3 3 3 --max-concurrent 1
    if grep -q "using the CPU interpolation" "$TMP/g3.log"; then fail "G6 the --gpu 3^3 run fell back to the CPU (see $TMP/g3.log)"
    elif ! precision_double; then
        # every output: the deposit's shares in the run's fixed-point unit, the face fluxes in the face pass's words
        differ="$(differing g g3)"
        if [ -z "$differ" ]; then
            pass "G6 GPU exact average, --partition 3 3 3 == one domain: every output byte-identical"
        else fail "G6 the 3^3 split moves the GPU exact average:$differ differ from the one-domain run"; fi
    # the double build keeps its double face sums (walk order: the run-wide words serve the single build), so their
    # order shows in the divergence and the shear; there every grid is compared within 1e-7 of its rms instead
    elif "$PY" -c "
import sys, numpy as np
ok = True
for f in ('a_den', 'a_velDiv', 'a_velShear'):
    a = np.fromfile('$TMP/g.' + f, dtype='${DTFE_TEST_REAL}').astype(np.float64)
    b = np.fromfile('$TMP/g3.' + f, dtype='${DTFE_TEST_REAL}').astype(np.float64)
    r = np.abs(a - b).max() / np.sqrt((a**2).mean())
    print('   (G6 %s: 3^3 split vs one domain %.2g of its rms)' % (f, r)); ok = ok and r <= 1e-7
sys.exit(0 if ok else 1)"; then
        pass "G6 GPU exact average, --partition 3 3 3 == one domain within 1e-7 of each grid's rms (double build)"
    else fail "G6 the 3^3 split moves the GPU exact average (a grid differs by over 1e-7 of its rms)"; fi
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

echo ">> --interlace: every grid but the density is the first pass's, the face gradient's among them"
# G8: G1's face run with '--interlace' -- the second pass, on the particles moved by half a cell, gives the density
# only (main.cpp)
run fi OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 DTFE_FACE_CHECK=1 -- --exact --face-gradient --interlace
d="$(differing f fi)"
if ! grep -qF "velocity gradient      : from EXACT FACE integrals" "$TMP/fi.log"; then
    fail "G8 the --interlace run does not take the face gradient (see $TMP/fi.log)"
elif [ "$d" = " .a_den" ]; then
    pass "G8 --interlace: every output but '.a_den' byte-identical to the face run without it (divergence, shear)"
else fail "G8 --interlace: the outputs that differ from the face run without it are '$d', want ' .a_den' alone"; fi

echo ">> refusal: --face-gradient --cic"
"${DTFE_BIN}" "$SNAP" "$TMP/c" --grid 16 --periodic --field density_a divergence_a --face-gradient --cic > "$TMP/c.log" 2>&1
rc=$?
if [ "$rc" = "1" ] && grep -q "'--face-gradient' needs the DTFE interpolation method" "$TMP/c.log"; then
    pass "G5 --face-gradient --cic stops at option parsing (exit 1, names the DTFE method)"
else
    fail "G5 --face-gradient --cic: exit $rc (see $TMP/c.log)"
fi

echo ">> the parallel insert against the sequential one, a team of 2 sub-domains against one domain"
# OMP_NUM_THREADS=2: DTFE_parallel's team of 2 sub-domains; DTFE_TRIANGULATION_THREADS=2 inserts each on 2 TBB
# threads, =1 sequentially (src/CGAL_triangulation/triangulation.cpp delaunayTriangulation); differing: above
T7=(--exact --face-gradient --run-mode performance)
if precision_double; then
    echo "   (G7: the double build keeps its double sums: SKIP)"
else
    run t1 OMP_NUM_THREADS=2 DTFE_TRIANGULATION_THREADS=1 -- "${T7[@]}"
    run t2 OMP_NUM_THREADS=2 DTFE_TRIANGULATION_THREADS=2 -- "${T7[@]}"
    d="$(differing f3 t1)"
    if [ -z "$d" ]; then pass "G7 CPU exact average, a team of 2 sub-domains == one domain: every output byte-identical"
    else fail "G7 the team of 2 sub-domains moves the CPU exact average:$d differ from the one-domain run"; fi
    if ! grep -qF "(parallel, TBB, 2 threads)" "$TMP/t2.log"; then
        echo "   (G7 parallel insert: no '(parallel, TBB, 2 threads)' line in $TMP/t2.log -- a build without the"
        echo "    parallel insert, or fewer than 4 threads: SKIP)"
    else
        d="$(differing t1 t2)"
        if [ -z "$d" ]; then pass "G7 CPU exact average, the parallel insert == the sequential one: outputs identical"
        else fail "G7 the parallel insert moves the CPU exact average:$d differ from the sequential insert"; fi
        if [ "$GPU" = "1" ]; then
            run tg2 OMP_NUM_THREADS=2 DTFE_TRIANGULATION_THREADS=2 -- "${T7[@]}" --gpu
            d="$(differing g tg2)"
            if grep -q "using the CPU interpolation" "$TMP/tg2.log"; then
                fail "G7 the --gpu team run fell back to the CPU (see $TMP/tg2.log)"
            elif [ -z "$d" ]; then
                pass "G7 GPU exact average, the parallel insert in a team of 2 == one domain: every output identical"
            else fail "G7 the parallel insert in a team of 2 moves the GPU exact average:$d differ from one domain"; fi
        else
            echo "   (G7's GPU half: not a GPU build)"
        fi
    fi
fi

if [ "$FAIL" = "0" ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; exit 1; fi
