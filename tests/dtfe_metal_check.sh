#!/usr/bin/env bash
# CPU vs GPU (--gpu) parity of the standard-DTFE method-1 '_a' interpolation on a synthetic Zel'dovich snapshot, single box and partitioned.
# Float-rounding tolerances (tests/ps_parallel_check.sh); the single box also runs the GPU twice and with the per-tetrahedron kernel.
# Requires a Metal build; fails loudly if the binary fell back to the CPU.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair (built by make ... DOUBLE=1)


TMP="${DTFE_TEST_TMP}/metal_check"
mkdir -p "$TMP"

# this test REQUIRES a GPU build: reuse the current mode if it is a GPU one, else default
GPU_BUILD="${GPU_BUILD:-$(cat o/.build_mode 2>/dev/null || true)}"
[ -z "${GPU_BUILD}" ] && GPU_BUILD="METAL=1"
echo ">> building DTFE ${GPU_BUILD} (test requires a GPU build)"
# shellcheck disable=SC2086  # the build mode may be two words (METAL=1 DOUBLE=1): word-split it
if precision_double; then
    precision_require "${DTFE_BIN}"
    [ -f o_d/.gpu_mode_off ] && { echo "SKIP: the double pair was built without a GPU backend"; exit 0; }
else
    make DTFE ${GPU_BUILD} -j4 >/dev/null || { echo "FAIL build"; exit 1; }
fi

SNAP="$TMP/xcheck.hdf5"
[ -f "$SNAP" ] || "$PY" tests/generate_ps_test_data.py --out "$SNAP" --n 48 --box 100 \
    --amplitude-factor 1.8 --jitter-frac 0.02 --seed 42 >/dev/null || { echo "FAIL data gen"; exit 1; }

FIELDS="density_a velocity_a gradient_a"
# --per-tet-gradient: this suite validates the KERNEL's gradient against the CPU's; the face gradient (the default since
# 2026-11-09) would make both sides the same CPU face pass
COMMON=(--grid 96 --periodic --input-format 105 --mpc-unit 1 --field $FIELDS --per-tet-gradient)
# The single-box and partitioned runs pass '--sampled'.

run() { # run <label> <outprefix> [extra args...]
    local label="$1" out="$2"; shift 2
    "${DTFE_BIN}" "$SNAP" "$out" "${COMMON[@]}" "$@" > "$out.log" 2>&1 \
        || { echo "FAIL run $label (see $out.log)"; exit 1; }
}

# The '<Backend> GPU (' dispatch line prints BEFORE the deposit call, so its presence alone
assert_gpu() { # assert_gpu <label> <log>
    grep -q " GPU (" "$2" || { echo "FAIL: $1 run did not attempt the GPU"; exit 1; }
    grep -q "using the CPU interpolation" "$2" && { echo "FAIL: $1 run fell back to the CPU at runtime"; exit 1; }
    return 0
}

echo ">> single box"
run cpu       "$TMP/s_cpu" --sampled
run gpu       "$TMP/s_gpu" --sampled --gpu
assert_gpu "--gpu" "$TMP/s_gpu.log"
run gpu-again "$TMP/s_gpu2" --sampled --gpu  # run-to-run: only atomic summation order may differ
assert_gpu "--gpu (second run)" "$TMP/s_gpu2.log"
echo ">> single box, per-tetrahedron kernel (DTFE_GPU_ITEMS=0: the A/B reference kernel)"
DTFE_GPU_ITEMS=0 run gpu-pertet "$TMP/s_gpu0" --sampled --gpu
assert_gpu "--gpu per-tet kernel" "$TMP/s_gpu0.log"
# A coarse grid makes most tetrahedra single-cell: the fast path.
echo ">> coarse 32^3 grid: the single-cell fast path"
COARSE=(--grid 32 --periodic --input-format 105 --mpc-unit 1 --field $FIELDS --per-tet-gradient --sampled)   # (the kernel's gradient, as COMMON)
"${DTFE_BIN}" "$SNAP" "$TMP/c_cpu" "${COARSE[@]}" > "$TMP/c_cpu.log" 2>&1 || { echo "FAIL run coarse cpu"; exit 1; }
"${DTFE_BIN}" "$SNAP" "$TMP/c_gpu1" "${COARSE[@]}" --gpu > "$TMP/c_gpu1.log" 2>&1 || { echo "FAIL run coarse gpu per-call"; exit 1; }
assert_gpu "coarse --gpu per-call" "$TMP/c_gpu1.log"

# The face gradient (the default): the deposits skip their per-tetrahedron gradient and the CPU face pass writes it
# in every run, so the GPU run's gradient must be the CPU run's to float order
echo ">> coarse 32^3 grid, the face gradient (the default)"
FACE=(--grid 32 --periodic --input-format 105 --mpc-unit 1 --field $FIELDS --sampled)
"${DTFE_BIN}" "$SNAP" "$TMP/cf_cpu" "${FACE[@]}" > "$TMP/cf_cpu.log" 2>&1 || { echo "FAIL run coarse face cpu"; exit 1; }
"${DTFE_BIN}" "$SNAP" "$TMP/cf_gpu1" "${FACE[@]}" --gpu > "$TMP/cf_gpu1.log" 2>&1 || { echo "FAIL run coarse face gpu per-call"; exit 1; }
assert_gpu "coarse face --gpu per-call" "$TMP/cf_gpu1.log"
grep -q "face gradient pass" "$TMP/cf_gpu1.log" || { echo "FAIL: the coarse face --gpu run did not run the face pass (is the face gradient still the default?)"; exit 1; }

echo ">> partitioned (2 2 2)"
run cpu-part  "$TMP/p_cpu" --partition 2 2 2 --max-concurrent 2 --sampled
run gpu-part  "$TMP/p_gpu" --partition 2 2 2 --max-concurrent 2 --sampled --gpu
assert_gpu "partitioned --gpu" "$TMP/p_gpu.log"
# --exact: the analytic volume average (r3d) on the GPU (depositExactAverage) against the CPU's double-precision r3d
echo ">> exact volume average (--exact)"
run cpu-exact "$TMP/e_cpu" --exact
run gpu-exact "$TMP/e_gpu" --exact --gpu
assert_gpu "--exact --gpu" "$TMP/e_gpu.log"
# a FINE region (the launcher's exact zoom, 512^2 cells of 1.95 kpc): the GPU clips each piece in a frame near it
FINE=(--grid 512 512 1 --periodic --input-format 105 --mpc-unit 1 --field density_a velocity_a --exact --region-mpc 40 41 40 41 50 50.002)
"${DTFE_BIN}" "$SNAP" "$TMP/f_cpu" "${FINE[@]}" > "$TMP/f_cpu.log" 2>&1 || { echo "FAIL run fine region cpu"; exit 1; }
"${DTFE_BIN}" "$SNAP" "$TMP/f_gpu" "${FINE[@]}" --gpu > "$TMP/f_gpu.log" 2>&1 || { echo "FAIL run fine region gpu"; exit 1; }
assert_gpu "--exact --gpu on a fine region" "$TMP/f_gpu.log"

"$PY" - "$TMP" <<'EOF'
import numpy as np, sys
import os as _os; REAL = np.dtype(_os.environ.get("DTFE_TEST_REAL", "float32"))   # tests/precision.sh
tmp = sys.argv[1]
REL_TOL   = 5e-2    # per-cell ceiling (borderline tets flip a sample across a cell wall)
FRAC_TOL  = 1e-3    # at most this fraction of cells may exceed 1e-3 relative
MEAN_TOL  = 1e-5    # field means must agree this closely (relative to max|field|)
REPEAT_TOL = 1e-4   # two GPU runs: atomic summation order only (a stray deposit would be ~1e-2)
# the velocity GRADIENT is constant per tetrahedron and jumps across its faces.
GRAD_FRAC_TOL = 1e-4
GRAD_PEAK_TOL = 0.5
GRAD_MEANABS_TOL = 1e-3

def judge(pre, suf, c, g):
    """(bad, line) of one CPU / GPU pair of grids."""
    c = np.asarray(c, dtype=float); g = np.asarray(g, dtype=float)
    scale = float(np.max(np.abs(c))) or 1.0
    d = np.abs(c - g)
    meandiff = abs(float(c.mean()) - float(g.mean())) / scale
    frac = float((d > 1e-3 * scale).sum()) / c.size
    peak = float(d.max()) / scale
    line = f"{pre}.{suf:10s} mean-rel={meandiff:.2e} peak-rel={peak:.2e} frac>1e-3={frac:.2e}"
    grad = suf == "a_velGrad"
    flip = grad and pre in ("s", "p") and frac <= GRAD_FRAC_TOL and peak <= GRAD_PEAK_TOL
    bad = meandiff > MEAN_TOL or (peak > REL_TOL and not flip) or frac > FRAC_TOL
    if grad:
        ma = float(np.abs(c).mean()) or 1.0
        meanabs = abs(ma - float(np.abs(g).mean())) / ma
        line += f" mean|abs|-rel={meanabs:.2e}"
        bad = bad or meanabs > GRAD_MEANABS_TOL
    return bad, line

# the rule itself on synthetic grids (a rule that let any velGrad peak pass below FRAC_TOL would fail it)
_rng = np.random.default_rng(1)
_c = _rng.normal(size=96 ** 3 * 9)
_flip = _c.copy(); _flip[_rng.choice(_c.size, 270, replace=False)] += 0.077 * np.abs(_c).max()   # CI's case
_moved = _c.copy(); _i = _rng.choice(_c.size, 3000, replace=False); _moved[_i] = _c[_rng.permutation(_i)]
for _pre, _g, _want in (("s", _flip, False), ("p", _flip, False), ("e", _flip, True), ("s", _moved, True)):
    _bad, _line = judge(_pre, "a_velGrad", _c, _g)
    if _bad != _want:
        print(f"FAIL rule self-test: {'passed' if _want else 'failed'} {_line}"); sys.exit(1)
print("PASS rule self-test (a CI face flip passes in s / p only; 3000 misplaced velGrad values fail)")

ok = True
for pre in ("s", "p", "e"):
    for suf in ("a_den", "a_vel", "a_velGrad"):
        c = np.fromfile(f"{tmp}/{pre}_cpu.{suf}", dtype=REAL)
        g = np.fromfile(f"{tmp}/{pre}_gpu.{suf}", dtype=REAL)
        if c.shape != g.shape:
            print(f"FAIL {pre}.{suf}: shape {c.shape} vs {g.shape}"); ok = False; continue
        scale = float(np.max(np.abs(c))) or 1.0
        bad, line = judge(pre, suf, c, g)
        if bad:
            print("FAIL " + line); ok = False
        else:
            print("PASS " + line)
        if pre == "s":   # the item kernel repeated, and the per-tet kernel, against the first GPU run
            # both kernels place the same samples with the same float arithmetic,
            # so like the repeat only the atomic summation order may differ
            for other, what in (("s_gpu2", "repeat"), ("s_gpu0", "per-tet kernel")):
                h = np.fromfile(f"{tmp}/{other}.{suf}", dtype=REAL)
                peak2 = float(np.abs(g - h).max()) / scale
                line = f"gpu vs {what:14s} {suf:10s} peak-rel={peak2:.2e}"
                if peak2 > REPEAT_TOL:
                    print("FAIL " + line); ok = False
                else:
                    print("PASS " + line)
for suf in ("a_den", "a_vel"):                  # the fine region (kpc cells): the clip frame must not cancel
    c = np.fromfile(f"{tmp}/f_cpu.{suf}", dtype=REAL).astype(float)
    g = np.fromfile(f"{tmp}/f_gpu.{suf}", dtype=REAL).astype(float)
    scale = float(np.max(np.abs(c))) or 1.0
    peak = float(np.abs(c - g).max()) / scale
    line = f"fine-region.{suf:10s} peak-rel={peak:.2e} (1.95 kpc cells; < 1e-4)"
    if peak > 1e-4:
        print("FAIL " + line); ok = False
    else:
        print("PASS " + line)
# the coarse grid: the GPU run against the CPU run, fast-path cells ('c': the kernel gradient). 'cf': the face
# gradient (the default), which the same CPU face pass writes in both runs, so that grid must agree to float order
FACE_GRAD_TOL = 1e-5
for pre in ("c", "cf"):
    for suf in ("a_den", "a_vel", "a_velGrad"):
        c = np.fromfile(f"{tmp}/{pre}_cpu.{suf}", dtype=REAL)
        g = np.fromfile(f"{tmp}/{pre}_gpu1.{suf}", dtype=REAL)
        scale = float(np.max(np.abs(c))) or 1.0
        d = np.abs(c - g)
        face = pre == "cf" and suf == "a_velGrad"
        peak_tol = FACE_GRAD_TOL if face else REL_TOL
        meandiff = abs(float(c.mean()) - float(g.mean())) / scale
        peak = float(d.max()) / scale
        frac = float((d > 1e-3 * scale).sum()) / c.size
        line = (f"{pre}.{suf:10s} mean-rel={meandiff:.2e} peak-rel={peak:.2e} frac>1e-3={frac:.2e}"
                + (" (the face pass in both: float order)" if face else ""))
        if meandiff > MEAN_TOL or peak > peak_tol or frac > FRAC_TOL:
            print("FAIL " + line); ok = False
        else:
            print("PASS " + line)
sys.exit(0 if ok else 1)
EOF
rc=$?
[ $rc -eq 0 ] && echo "dtfe_metal_check: ALL PASS" || echo "dtfe_metal_check: FAILURES"
exit $rc
