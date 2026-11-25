#!/usr/bin/env bash
# The STANDARD-DTFE auto-tuner (src/auto_tune.h): the report, the counted split, the thread plan and the memory model
# of the production '--exact' runs, on synthetic crossed waves and the repository's binaries. The tuner replays the run's own
# split on a count grid, since a box-average particle count per thread under-counts a dense partition's padded vertices.
# Usage: tests/dtfe_auto_tune_check.sh [--no-build]     (DTFE_TEST_TMP, DTFE_TEST_PRECISION=double as the others)

set -euo pipefail
unset DTFE_TRIANGULATION_THREADS   # the insert's own switch moves the plans: the suite sets none
unset DTFE_TEST_TUNE_UNIFORM    # (M sets it where it checks it)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

BIN="${DTFE_BIN}"
precision_require "${BIN}"
# The thread count is pinned to a 10-core machine's (as ps_auto_tune_check.sh H1 does): the tuner plans with every
export OMP_NUM_THREADS=10
TMP="${DTFE_TEST_TMP}/dtfe_auto_tune"; mkdir -p "${TMP}"
N="${N:-64}"; BOX="${BOX:-100.0}"
SNAP="${TMP}/dat_crossed${N}.hdf5"

echo "============================================================"
echo " standard-DTFE auto-tune check   N=${N}^3 crossed waves"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building DTFE ..."
    make DTFE $(cat o/.build_mode 2>/dev/null || true) >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }
GPU=0; grep -q -- "--gpu" <<< "$("${BIN}" --help 2>&1)" && grep -q "METAL=1" "o$(precision_double && echo _d)/.build_mode" 2>/dev/null && GPU=1

echo ">> generating the clustered test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" --crossed-waves \
    --amplitude-factor 3.0 >/dev/null

FAILS=0
# check_ok <label> <1 = ok> (it counts a failure in FAILS), agrep <label> <pattern> <file>: tests/lib.sh
COMMON=(--periodic --input-format 105 --mpc-unit 1 --padding 10)
REPORT_RE='^AUTO-TUNE-REPORT partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=[0-9.]+ fixed_gb=[0-9.]+ .*over_budget=[01]$'
field() { sed -n "s/^AUTO-TUNE-REPORT .* $1=\([0-9.]*\) .*/\1/p; s/^AUTO-TUNE-REPORT .* $1=\([0-9.]*\)$/\1/p" "$2" | head -1; }

# ---------- (A) the report ----------
echo ">> A: --auto-tune-report on the standard binary"
rm -f "${TMP}"/da_a.*
rc=0; env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_a" --grid 64 "${COMMON[@]}" --field density_a velocity_a \
    --auto-tune-report > "${TMP}/da_a_log.txt" 2>&1 || rc=$?
check_ok "A exits 0 (${rc})" "$([ "${rc}" = "0" ] && echo 1 || echo 0)"
check_ok "A prints exactly one valid report line (not 'unsupported')" \
      "$([ "$(grep -cE "${REPORT_RE}" "${TMP}/da_a_log.txt")" = "1" ] && ! grep -q 'unsupported' "${TMP}/da_a_log.txt" && echo 1 || echo 0)"
check_ok "A nothing is written next to the output root" "$([ -z "$(ls "${TMP}" | grep '^da_a\.')" ] && echo 1 || echo 0)"
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_a" --grid 64 "${COMMON[@]}" --field density_a \
    --partition 2 2 2 --max-concurrent 3 --auto-tune-report > "${TMP}/da_a2_log.txt" 2>&1 || true
agrep "A a hand-set split and thread count are reported as given" "^AUTO-TUNE-REPORT partition=2 mc=3 " "${TMP}/da_a2_log.txt"

# ---------- (B) + (C) the counted padded vertices against the run's own ----------
echo ">> B/C: the counted padded vertices against the run's 'Thread i had' lines"
compare_counts() {  # $1 = tag, $2 = split, $3 = threads; prints the checks
    local tag="$1" split="$2" mc="$3"
    env DTFE_AUTO_TUNE_DEBUG=1 DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_${tag}" --grid 48 "${COMMON[@]}" \
        --field density_a --partition ${split} --max-concurrent "${mc}" --auto-tune-report > "${TMP}/da_${tag}_rep.txt" 2>&1 || true
    "${BIN}" "${SNAP}" "${TMP}/da_${tag}" --grid 48 "${COMMON[@]}" --field density_a --partition ${split} \
        --max-concurrent "${mc}" --verbose 2 > "${TMP}/da_${tag}_run.txt" 2>&1 || true
    "${PY}" - "${TMP}/da_${tag}_rep.txt" "${TMP}/da_${tag}_run.txt" "${N}" "${split%% *}" "${mc}" "${BOX}" <<'PYEOF'
import re, sys
rep, run = open(sys.argv[1], errors="replace").read(), open(sys.argv[2], errors="replace").read()
n, P, T = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
m = re.search(r"AUTO-TUNE-DEBUG standard: counted from the particles .*busiest partition (\d+) of (\d+): (\d+) padded "
              r"vertices.*per thread:([ 0-9]+)", rep)
parts = run.split("<<< Interpolating the fields for partition")[1:]
real = [[int(x) for x in re.findall(r"Thread \d+ had (\d+) particles", p)] for p in parts]
ok_b = ok_c = False
if m and real and all(real):
    pm, nparts, vm = int(m.group(1)), int(m.group(2)), float(m.group(3))
    tm = [float(x) for x in m.group(4).split()]
    ir = max(range(len(real)), key=lambda k: sum(real[k]))
    vr, tr = sum(real[ir]), real[ir]
    worst = max(abs(a - b) / b for a, b in zip(tm, tr)) if len(tm) == len(tr) else 9.
    print(f"   .... busiest partition {pm} of {nparts} (the run's: {ir + 1} of {len(real)}): {vm:.0f} counted vs {vr} "
          f"padded vertices; per thread worst {100 * worst:.1f}%")
    ok_b = nparts == len(real) and abs(vm - vr) / vr < 0.10 and worst < 0.10 and sum(real[pm - 1]) > 0.9 * vr
    # the box-average count: every sub-box at the box average, (N / P^3) x prod(1 + 2 pad P T^(1/3)), pad in box units
    pad = 10. / n
    uniform = n**3 / P**3 * (1. + 2. * pad * P * T ** (1. / 3.)) ** 3
    print(f"   .... the uniform count of the old model: {uniform:.0f} ({vr / uniform:.2f}x less than the run's)")
    ok_c = vr > 1.5 * uniform
print(f"OKB {int(ok_b)} OKC {int(ok_c)}")
PYEOF
}
for cfg in "b3|3 3 3|8" "b4|4 4 4|4"; do
    tag="${cfg%%|*}"; rest="${cfg#*|}"; split="${rest%%|*}"; mc="${rest##*|}"
    out="$(compare_counts "${tag}" "${split}" "${mc}")"
    printf '%s\n' "${out}" | grep '^   \.\.\.\.' || true
    check_ok "B ${split// /x} x ${mc}: the busiest partition and its padded vertices (sum and per thread) within 10% of the run's" \
          "$(grep -q '^OKB 1' <<< "${out}" && echo 1 || echo 0)"
    check_ok "C ... and the run's busiest partition holds >= 1.5x the box-average count the old model used" \
          "$(grep -q 'OKC 1' <<< "${out}" && echo 1 || echo 0)"
done
rm -f "${TMP}"/da_b*.a_den "${TMP}"/da_b*badGridPoints*

# ---------- (D) a user's --max-concurrent ----------
echo ">> D: a user's --max-concurrent is planned with"
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_d" --grid 48 "${COMMON[@]}" --field density_a --max-concurrent 2 \
    --auto-tune-report > "${TMP}/da_d.txt" 2>&1 || true
agrep "D '--max-concurrent 2' with the split left to the tuner: reported mc=2" "^AUTO-TUNE-REPORT partition=[0-9]+ mc=2 " "${TMP}/da_d.txt"

# ---------- (E) the SUMMARY's memory model line, predicted vs measured ----------
echo ">> E: predicted vs measured peak on real runs (128^3, the production fields)"
FIELDS=(density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a)
run_e() {  # $1 = tag, rest = extra args
    local tag="$1"; shift
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/da_${tag}" --grid 128 "${COMMON[@]}" --field "${FIELDS[@]}" \
        --lambda-th 0.3 --partition 2 2 2 "$@" > "${TMP}/da_${tag}.txt" 2>&1 || true
    "${PY}" - "${TMP}/da_${tag}.txt" <<'PYEOF'
import re, sys
m = re.search(r"memory model\s*: ([0-9.]+) GB predicted \(auto-tune\) vs ([0-9.]+) GB measured", open(sys.argv[1], errors="replace").read())
if not m:
    print("NOLINE"); sys.exit(0)
p, q = float(m.group(1)), float(m.group(2))
print(f"   .... predicted {p:.2f} GB, measured {q:.2f} GB ({q / p:.2f}x)")
print("OKE 1" if q <= 1.35 * p else "OKE 0")
PYEOF
    rm -f "${TMP}/da_${tag}".a_* "${TMP}/da_${tag}"*badGridPoints*
}
out="$(run_e e_cpu --sampled)"; printf '%s\n' "${out}" | grep '^   \.\.\.\.' || true
check_ok "E the SUMMARY of an auto-tuned standard run carries the 'memory model' line" "$(grep -q NOLINE <<< "${out}" && echo 0 || echo 1)"
check_ok "E CPU, sampled average: measured <= 1.35x predicted" "$(grep -q 'OKE 1' <<< "${out}" && echo 1 || echo 0)"
if [ "${GPU}" = 1 ]; then
    out="$(run_e e_gpu --gpu --exact)"; printf '%s\n' "${out}" | grep '^   \.\.\.\.' || true
    check_ok "E GPU, exact average: measured <= 1.35x predicted" "$(grep -q 'OKE 1' <<< "${out}" && echo 1 || echo 0)"
else
    echo "   SKIP  E GPU (this build has no GPU backend)"
fi

# ---------- (F) the tuner only chooses ----------
echo ">> F: an auto-tuned run equals the same split given by hand"
env DTFE_RAM_GB=1.2 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/da_f_auto" --grid 48 "${COMMON[@]}" --field density_a velocity_a \
    > "${TMP}/da_f_auto.txt" 2>&1 || true
fsplit="$(sed -n 's/.*AUTO-TUNE: .* -> --partition \([0-9 ]*\), \([0-9]*\) thread(s).*/\1|\2/p' "${TMP}/da_f_auto.txt" | head -1)"
if [ -n "${fsplit}" ]; then
    "${BIN}" "${SNAP}" "${TMP}/da_f_hand" --grid 48 "${COMMON[@]}" --field density_a velocity_a --partition ${fsplit%%|*} \
        --max-concurrent "${fsplit##*|}" > "${TMP}/da_f_hand.txt" 2>&1 || true
    check_ok "F the tuner's split (${fsplit%%|*}, ${fsplit##*|} threads) given by hand: byte-identical .a_den and .a_vel" \
          "$([ -s "${TMP}/da_f_auto.a_den" ] && cmp -s "${TMP}/da_f_auto.a_den" "${TMP}/da_f_hand.a_den" \
             && cmp -s "${TMP}/da_f_auto.a_vel" "${TMP}/da_f_hand.a_vel" && echo 1 || echo 0)"
else
    check_ok "F the tuner chose a split on a simulated 1.2 GB machine (no AUTO-TUNE line)" 0
fi
rm -f "${TMP}"/da_f_*.a_* "${TMP}"/da_f_*badGridPoints*

# ---------- (G) a tight machine ----------
echo ">> G: a split that fits a simulated small machine"
env DTFE_RAM_GB=1.2 "${BIN}" "${SNAP}" "${TMP}/da_g" --grid 96 "${COMMON[@]}" --field density_a velocity_a gradient_a \
    --auto-tune-report > "${TMP}/da_g.txt" 2>&1 || true
agrep "G the plan fits its budget (over_budget=0) with a split" "^AUTO-TUNE-REPORT partition=[2-6] mc=[0-9]+ .*over_budget=0$" "${TMP}/da_g.txt"
agrep "G ... and the AUTO-TUNE line names the busiest partition's padded vertices" \
      "AUTO-TUNE: .*the busiest partition triangulates ~[0-9.]+ M padded vertices at once" "${TMP}/da_g.txt"
# A budget between two plans, on 128^3 particles of their own.
if ! precision_double; then
    SNAPG="${TMP}/dat_crossed128_g.hdf5"
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAPG}" --n 128 --box "${BOX}" --crossed-waves \
        --amplitude-factor 3.0 >/dev/null
    # gplan <tag> <env ...> -- <options ...>: the predicted_gb of one report (DTFE_RAM_GB=64 unless the env sets it)
    gplan() {
        local tag="$1"; shift
        local envs=(DTFE_RAM_GB=64)
        while [ "$1" != "--" ]; do envs+=("$1"); shift; done
        shift
        env "${envs[@]}" "${BIN}" "$@" > "${TMP}/da_g2_${tag}.txt" 2>&1 || true
        field predicted_gb "${TMP}/da_g2_${tag}.txt"
    }
    GR=("${SNAPG}" "${TMP}/da_g2" --grid 48 "${COMMON[@]}" --field density_a velocity_a gradient_a --auto-tune-report
        --exact --face-gradient --no-overlap)
    q22="$(gplan c22 DTFE_GPU=0 -- "${GR[@]}" --partition 2 2 2 --max-concurrent 2)"
    q210="$(gplan c210 DTFE_GPU=0 -- "${GR[@]}" --partition 2 2 2 --max-concurrent 10)"
    q310="$(gplan c310 DTFE_GPU=0 -- "${GR[@]}" --partition 3 3 3 --max-concurrent 10)"
    gr="$(awk -v a="${q22:-0}" -v b="${q210:-0}" -v c="${q310:-0}" 'BEGIN{m = (a > c ? a : c);
        if (a > 0 && c > 0 && b - m >= 0.1) printf "%.4f", (m + b) / 2 / 0.6}')"
    if [ -n "${gr}" ]; then
        gbud="$(awk -v r="${gr}" 'BEGIN{printf "%.2f", 0.6*r}')"
        env DTFE_GPU=0 DTFE_RAM_GB="${gr}" "${BIN}" "${GR[@]}" > "${TMP}/da_g2_cm.txt" 2>&1 || true
        check_ok "G1 the CPU deposit at a budget of ${gbud} GB (2^3 x 2: ${q22} GB, 3^3 x 10: ${q310} GB, 2^3 x 10: ${q210} GB): the plan is 3^3 x 10 threads within it, the most threads at the coarsest split that holds them" \
              "$(grep -Eq '^AUTO-TUNE-REPORT partition=3 mc=10 .*over_budget=0$' "${TMP}/da_g2_cm.txt" && echo 1 || echo 0)"
    else
        check_ok "G1 a budget between two CPU plans: 2^3 x 2 and 3^3 x 10 threads (${q22:-?} / ${q310:-?} GB) at least 0.1 GB below 2^3 x 10 (${q210:-?} GB)" 0
    fi
    if [ "${GPU}" = 1 ]; then
        q21="$(gplan g21 -- "${GR[@]}" --gpu --partition 2 2 2 --max-concurrent 1)"
        q11="$(gplan g11 -- "${GR[@]}" --gpu --partition 1 1 1 --max-concurrent 1)"
        q12="$(gplan g12 -- "${GR[@]}" --gpu --partition 1 1 1 --max-concurrent 2)"
        q14="$(gplan g14 -- "${GR[@]}" --gpu --partition 1 1 1 --max-concurrent 4)"
        q110="$(gplan g110 -- "${GR[@]}" --gpu --partition 1 1 1 --max-concurrent 10)"
        gg="$(awk -v a="${q21:-0}" -v b="${q11:-0}" -v c="${q12:-0}" -v d="${q14:-0}" -v e="${q110:-0}" 'BEGIN{
            m = b; if (c < m) m = c; if (d < m) m = d; if (e < m) m = e;
            if (a > 0 && m > 0 && m - a >= 0.1) printf "%.4f", (a + m) / 2 / 0.6}')"
        if [ -n "${gg}" ]; then
            gbud="$(awk -v r="${gg}" 'BEGIN{printf "%.2f", 0.6*r}')"
            env DTFE_RAM_GB="${gg}" "${BIN}" "${GR[@]}" --gpu > "${TMP}/da_g2_gm.txt" 2>&1 || true
            check_ok "G2 the GPU deposit at a budget of ${gbud} GB (2^3 x 1: ${q21} GB; one domain x 1/2/4/10: ${q11}/${q12}/${q14}/${q110} GB): the plan is 2^3 x 1, one tessellation per partition at the coarsest split that holds it" \
                  "$(grep -Eq '^AUTO-TUNE-REPORT partition=2 mc=1 .*over_budget=0$' "${TMP}/da_g2_gm.txt" && echo 1 || echo 0)"
        else
            check_ok "G2 a budget between two GPU plans: 2^3 x 1 (${q21:-?} GB) at least 0.1 GB below one domain's least (${q11:-?}/${q12:-?}/${q14:-?}/${q110:-?} GB)" 0
        fi
    else
        echo "   SKIP  G2 the GPU deposit's plan (a build without the GPU backend)"
    fi
    rm -f "${TMP}"/da_g2_* "${SNAPG}"
else
    echo "   SKIP  G1/G2 at a budget between two plans (the double build)"
fi

# ---------- (H) 2D ----------
echo ">> H: the 2D build"
BIN2D="./DTFE-2d"; precision_double && BIN2D="./DTFE-2d-double"
if [ -x "${BIN2D}" ]; then
    SNAP2D="${TMP}/dat_crossed2d.hdf5"
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP2D}" --n 512 --box "${BOX}" --dim 2 --crossed-waves \
        --amplitude-factor 3.0 >/dev/null
    env DTFE_AUTO_TUNE_DEBUG=1 DTFE_RAM_GB=64 "${BIN2D}" "${SNAP2D}" "${TMP}/da_h" --grid 128 "${COMMON[@]}" --field density_a \
        --partition 3 3 --max-concurrent 4 --auto-tune-report > "${TMP}/da_h_rep.txt" 2>&1 || true
    "${BIN2D}" "${SNAP2D}" "${TMP}/da_h" --grid 128 "${COMMON[@]}" --field density_a --partition 3 3 --max-concurrent 4 \
        --verbose 2 > "${TMP}/da_h_run.txt" 2>&1 || true
    agrep "H a valid 2D report" "${REPORT_RE}" "${TMP}/da_h_rep.txt"
    check_ok "H the counted busiest partition within 15% of the 2D run's" "$("${PY}" - "${TMP}/da_h_rep.txt" "${TMP}/da_h_run.txt" <<'PYEOF'
import re, sys
rep, run = open(sys.argv[1], errors="replace").read(), open(sys.argv[2], errors="replace").read()
m = re.search(r"busiest partition \d+ of \d+: (\d+) padded vertices", rep)
real = [sum(int(x) for x in re.findall(r"Thread \d+ had (\d+) particles", p))
        for p in run.split("<<< Interpolating the fields for partition")[1:]]
print(1 if m and real and abs(float(m.group(1)) - max(real)) < 0.15 * max(real) else 0)
PYEOF
)"
    rm -f "${TMP}"/da_h.a_* "${TMP}"/da_h*badGridPoints*
else
    echo "   SKIP  H (${BIN2D} is not built)"
fi

# ---------- (I) the plane records, the GPU snapshot and the look-ahead in the report ----------
echo ">> I: the plane records, the GPU snapshot and the look-ahead in the report line"
"${PY}" - "${TMP}/da_i_plane.bin" "${BOX}" <<'PYEOF'
import sys, numpy as np
c = (np.arange(16) + 0.5) * float(sys.argv[2]) / 16
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.5 * float(sys.argv[2]))], 1).astype(np.float64).tofile(sys.argv[1])
PYEOF
IR=("${SNAP}" "${TMP}/da_i" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2
    --auto-tune-report)
env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 1 > "${TMP}/da_i0.txt" 2>&1 || true
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=4194304 DTFE_AUTO_PTS_VEL_GRAD=1 "${BIN}" "${IR[@]}" --max-concurrent 1 \
    --sample-points "${TMP}/da_i_plane.bin" > "${TMP}/da_i1.txt" 2>&1 || true
p1="$(field pts_gb "${TMP}/da_i1.txt")"; f0="$(field fixed_gb "${TMP}/da_i0.txt")"; f1="$(field fixed_gb "${TMP}/da_i1.txt")"
ok="$(awk -v p="${p1:-0}" -v a="${f0:-0}" -v b="${f1:-0}" 'BEGIN{print (p > 1.362 && p < 1.372 && b - a > p - 0.02 && b - a < p + 0.02) ? 1 : 0}')"
grep -q 'streams_budget=1.000 ' "${TMP}/da_i1.txt" || ok=0
check_ok "I1 a 2048^2 plane with --pts-vel-grad: pts_gb ${p1:-?} (4194304 x 326 B = 1.367), in fixed_gb (${f0:-?} -> ${f1:-?}), one record per point" "${ok}"
s0="$(field snapshot_gb "${TMP}/da_i0.txt")"; l0="$(field lookahead_gb "${TMP}/da_i0.txt")"
check_ok "I2/I3 a CPU run counts no GPU snapshot and no look-ahead (${s0:-?}, ${l0:-?})" \
      "$([ "${s0}" = "0.000" ] && [ "${l0}" = "0.000" ] && echo 1 || echo 0)"
if [ "${GPU}" = 1 ]; then
    env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu > "${TMP}/da_i2.txt" 2>&1 || true
    env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu --no-overlap > "${TMP}/da_i4.txt" 2>&1 || true
    env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu --sampled > "${TMP}/da_i3.txt" 2>&1 || true
    s2="$(field snapshot_gb "${TMP}/da_i2.txt")"; s3="$(field snapshot_gb "${TMP}/da_i3.txt")"
    # the exact average's run-wide 128-bit words: 16 B of snapshot per 4 B grid component, the sampled average's
    # 64-bit words 8 B (the GB carry three decimals: a few per cent of play)
    check_ok "I2 --gpu: the snapshot of the largest sub-grid ${s2:-?} GB = 2x the sampled average's (${s3:-?} GB)" \
          "$(awk -v a="${s2:-0}" -v b="${s3:-0}" 'BEGIN{print (a > 0.01 && b > 0. && a / b > 1.9 && a / b < 2.1) ? 1 : 0}')"
    l2="$(field lookahead_gb "${TMP}/da_i2.txt")"; l4="$(field lookahead_gb "${TMP}/da_i4.txt")"
    q2="$(field predicted_gb "${TMP}/da_i2.txt")"; q4="$(field predicted_gb "${TMP}/da_i4.txt")"
    check_ok "I3 --gpu over 8 partitions: the look-ahead (${l2:-?} GB) is in predicted_gb (${q4:-?} -> ${q2:-?}); none with --no-overlap (${l4:-?})" \
          "$(awk -v l="${l2:-0}" -v m="${l4:-1}" -v a="${q4:-0}" -v b="${q2:-0}" 'BEGIN{d = b - a - l; print (l > 0.01 && m == 0 && d > -0.011 && d < 0.011) ? 1 : 0}')"
    # a budget between the plan without the look-ahead and the plan with it (the budget is 60% of DTFE_RAM_GB)
    ram="$(awk -v a="${q4:-0}" -v l="${l2:-0}" 'BEGIN{printf "%.4f", (a + l / 2.) / 0.6}')"
    env DTFE_RAM_GB="${ram}" "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu > "${TMP}/da_i5.txt" 2>&1 || true
    check_ok "I4 a budget that holds the plan (${q4:-?} GB) but not its look-ahead: switched off, said, over_budget=0" \
          "$(grep -q 'look-ahead off for this run' "${TMP}/da_i5.txt" && grep -Eq 'lookahead_gb=0.000 over_budget=0$' "${TMP}/da_i5.txt" \
             && echo 1 || echo 0)"
else
    echo "   SKIP  I2-I4 (this build has no GPU backend)"
fi
rm -f "${TMP}"/da_i.* "${TMP}"/da_i*badGridPoints*

# ---------- (J) the in-team parallel insert ----------
echo ">> J: a team's parallel insert is in the plan"
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
if grep -aq "parallel, TBB" "${BIN}" && [ "${NCPU}" -ge 4 ]; then
    JR=("${SNAP}" "${TMP}/da_j" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2
        --auto-tune-report --no-overlap)
    for t in 1 2; do
        env DTFE_GPU=0 DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${JR[@]}" --max-concurrent "${t}" > "${TMP}/da_j${t}.txt" 2>&1 || true
        env DTFE_GPU=0 DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${JR[@]}" --max-concurrent "${t}" --parallel-triangulation \
            > "${TMP}/da_j${t}f.txt" 2>&1 || true
    done
    # one tessellation on a 32^3 grid, whose deposit phase is smaller than the insert's copies
    JC=("${SNAP}" "${TMP}/da_j" --grid 32 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2
        --auto-tune-report --max-concurrent 1 --no-overlap)
    env DTFE_GPU=0 DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${JC[@]}" > "${TMP}/da_jc.txt" 2>&1 || true
    env DTFE_GPU=0 DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${JC[@]}" --parallel-triangulation > "${TMP}/da_jcf.txt" 2>&1 || true
    busiest() { sed -n 's/^AUTO-TUNE-DEBUG standard: .* busiest partition [0-9]* of [0-9]*: \([0-9]*\) padded vertices.*/\1/p' "$1" | head -1; }
    # per padded vertex the parallel insert's transient less the sequential insert's particle copy.
    if precision_double; then jbase=24; jslope=104; else jbase=36; jslope=80; fi
    rise() { awk -v a="${1:-0}" -v b="${2:-0}" -v u="${3:-0}" -v f="${4:-0}" -v c="${jbase}" -v s="${jslope}" \
                 'BEGIN{e = b - a; lo = (u < f ? u : f) * (c + s) / 1.e9; hi = (u > f ? u : f) * (c + 2. * s) / 1.e9;
                        print (u > 0 && f > 0 && e > lo - 0.011 && e < hi + 0.011) ? 1 : 0}'; }
    u1="$(busiest "${TMP}/da_j1.txt")"; f1="$(busiest "${TMP}/da_j1f.txt")"
    u2="$(busiest "${TMP}/da_j2.txt")"; f2="$(busiest "${TMP}/da_j2f.txt")"
    a1="$(field predicted_gb "${TMP}/da_j1.txt")"; b1="$(field predicted_gb "${TMP}/da_j1f.txt")"
    a2="$(field predicted_gb "${TMP}/da_j2.txt")"; b2="$(field predicted_gb "${TMP}/da_j2f.txt")"
    check_ok "J a team of 2 books the forced insert's transient (${a2:-?} -> ${b2:-?} GB: ${jbase} + ${jslope} g B x ${u2:-?} to ${f2:-?} padded vertices)" \
          "$(rise "${a2}" "${b2}" "${u2}" "${f2}")"
    uc="$(busiest "${TMP}/da_jc.txt")"; fc="$(busiest "${TMP}/da_jcf.txt")"
    ac="$(field predicted_gb "${TMP}/da_jc.txt")"; bc="$(field predicted_gb "${TMP}/da_jcf.txt")"
    check_ok "J one tessellation without a look-ahead, 32^3: its insert phase is the larger, the transient is booked (${ac:-?} -> ${bc:-?} GB: ${jbase} + ${jslope} g B x ${uc:-?} to ${fc:-?})" \
          "$(rise "${ac}" "${bc}" "${uc}" "${fc}")"
    check_ok "J one tessellation without a look-ahead, 192^3: its deposit phase is the larger, the copies add nothing (${a1:-?} -> ${b1:-?} GB)" \
          "$(awk -v a="${a1:-0}" -v b="${b1:-x}" 'BEGIN{d = b - a; print (a > 0 && d > -0.011 && d < 0.011) ? 1 : 0}')"
    rm -f "${TMP}"/da_j*
else
    echo "   SKIP  J (a build without the parallel insert, or fewer than 4 threads)"
fi

# ---------- (K) the look-ahead's switch-off with the insert's copies ----------
echo ">> K: a look-ahead the parallel insert's copies leave no room for"
if [ "${GPU}" = 1 ] && grep -aq "parallel, TBB" "${BIN}" && [ "${NCPU}" -ge 4 ]; then
    KR=("${SNAP}" "${TMP}/da_k" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2 --gpu
        --auto-tune-report)
    k_rep() {  # $1 = tag, $2 = DTFE_RAM_GB, then options
        local tag="$1" ram="$2"; shift 2
        env DTFE_RAM_GB="${ram}" "${BIN}" "${KR[@]}" "$@" > "${TMP}/da_k_${tag}.txt" 2>&1 || true
    }
    # K1, a team of 2 on 128^3 particles (on 64^3 the forced insert moves the look-ahead too little to show in the report)
    SNAPK="${TMP}/dat_crossed128_k.hdf5"
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAPK}" --n 128 --box "${BOX}" --crossed-waves \
        --amplitude-factor 3.0 >/dev/null
    KR1=("${SNAPK}" "${TMP}/da_k" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2
         --gpu --auto-tune-report --max-concurrent 2)
    k1_rep() {  # $1 = tag, $2 = DTFE_RAM_GB, then options
        local tag="$1" ram="$2"; shift 2
        env DTFE_RAM_GB="${ram}" "${BIN}" "${KR1[@]}" "$@" > "${TMP}/da_k_${tag}.txt" 2>&1 || true
    }
    k1_rep b2 64 --no-overlap --parallel-triangulation
    k1_rep p0 64
    k1_rep p1 64 --parallel-triangulation
    b2="$(field predicted_gb "${TMP}/da_k_b2.txt")"; p0="$(field predicted_gb "${TMP}/da_k_p0.txt")"
    p1="$(field predicted_gb "${TMP}/da_k_p1.txt")"
    l0="$(field lookahead_gb "${TMP}/da_k_p0.txt")"; l1="$(field lookahead_gb "${TMP}/da_k_p1.txt")"
    kr="$(awk -v b="${b2:-0}" -v p="${p0:-0}" -v q="${p1:-0}" 'BEGIN{if (b > 0 && p > b && q - p >= 0.04) printf "%.4f", (p + q)/2/0.6}')"
    if [ -n "${kr}" ]; then
        k1_rep on2 "${kr}"
        k1_rep off2 "${kr}" --parallel-triangulation
        o2="$(field lookahead_gb "${TMP}/da_k_on2.txt")"; pf="$(field predicted_gb "${TMP}/da_k_off2.txt")"
        check_ok "K1 a budget of $(awk -v r="${kr}" 'BEGIN{printf "%.2f", 0.6*r}') GB holds a team of 2 with its look-ahead (${p0:-?} GB, lookahead ${l0:-?}), not with the forced insert's transient in it (${p1:-?} GB, lookahead ${l1:-?}): switched off (said, predicted ${pf:-?} = ${b2:-?} GB, lookahead_gb=0, over_budget=0); kept without it (lookahead_gb ${o2:-?})" \
              "$(grep -q 'look-ahead off for this run' "${TMP}/da_k_off2.txt" \
                 && grep -Eq 'lookahead_gb=0.000 over_budget=0$' "${TMP}/da_k_off2.txt" \
                 && ! grep -q 'look-ahead off' "${TMP}/da_k_on2.txt" && grep -Eq 'over_budget=0$' "${TMP}/da_k_on2.txt" \
                 && awk -v o="${o2:-0}" -v l="${l0:-x}" -v p="${pf:-0}" -v b="${b2:-0}" \
                        'BEGIN{d = p - b; exit !(o == l && d > -0.011 && d < 0.011)}' && echo 1 || echo 0)"
    else
        check_ok "K1 the budget between the bookings (not formed from ${b2:-none}, ${p0:-none}, ${p1:-none} GB)" 0
    fi
    rm -f "${SNAPK}"
    # K2, one tessellation: the look-ahead's factor holds its copies, so the forced insert books none while the
    # look-ahead runs; a budget half a look-ahead above the plan switches it off.
    k_rep q1 64 --max-concurrent 1 --no-overlap
    k_rep b1 64 --max-concurrent 1 --no-overlap --parallel-triangulation
    k_rep l1 64 --max-concurrent 1 --parallel-triangulation
    q1="$(field predicted_gb "${TMP}/da_k_q1.txt")"; b1="$(field predicted_gb "${TMP}/da_k_b1.txt")"
    l1="$(field lookahead_gb "${TMP}/da_k_l1.txt")"
    kr="$(awk -v q="${q1:-0}" -v b="${b1:-0}" -v l="${l1:-0}" 'BEGIN{m = (b > q ? b : q); if (q > 0 && l > 0.04) printf "%.4f", (m + l/2)/0.6}')"
    if [ -n "${kr}" ]; then
        k_rep off1 "${kr}" --max-concurrent 1 --parallel-triangulation
        k_rep seq1 "${kr}" --max-concurrent 1
        p1="$(field predicted_gb "${TMP}/da_k_off1.txt")"; s1="$(field predicted_gb "${TMP}/da_k_seq1.txt")"
        check_ok "K2 one tessellation, a budget of $(awk -v r="${kr}" 'BEGIN{printf "%.2f", 0.6*r}') GB (half its look-ahead of ${l1:-?} GB above the plan): switched off, booked by its phases as '--no-overlap' books it, with the forced insert (${p1:-?} = ${b1:-?} GB) and without it (${s1:-?} = ${q1:-?} GB)" \
              "$(grep -q 'look-ahead off for this run' "${TMP}/da_k_off1.txt" && grep -q 'look-ahead off for this run' "${TMP}/da_k_seq1.txt" \
                 && grep -Eq 'lookahead_gb=0.000 over_budget=[01]$' "${TMP}/da_k_off1.txt" \
                 && awk -v p="${p1:-0}" -v b="${b1:-x}" -v s="${s1:-0}" -v q="${q1:-x}" \
                        'BEGIN{d = p - b; e = s - q; exit !(d > -0.011 && d < 0.011 && e > -0.011 && e < 0.011)}' \
                 && echo 1 || echo 0)"
    else
        check_ok "K2 the budget under the look-ahead (not formed from ${q1:-none}, ${b1:-none}, ${l1:-none} GB)" 0
    fi
    rm -f "${TMP}"/da_k*
else
    echo "   SKIP  K (a build without the GPU or the parallel insert, or fewer than 4 threads)"
fi

# ---------- (L) the GPU deposit: one tessellation per partition, with or without its look-ahead ----------
echo ">> L: the GPU deposit runs a given split with one tessellation per partition"
if [ "${GPU}" = 1 ]; then
    LR=("${SNAP}" "${TMP}/da_l" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2 --gpu
        --auto-tune-report)
    env DTFE_RAM_GB=64 "${BIN}" "${LR[@]}" --max-concurrent 1 > "${TMP}/da_l1.txt" 2>&1 || true
    env DTFE_RAM_GB=64 "${BIN}" "${LR[@]}" > "${TMP}/da_la.txt" 2>&1 || true
    a1="$(field predicted_gb "${TMP}/da_l1.txt")"; l1="$(field lookahead_gb "${TMP}/da_l1.txt")"
    r1="$(grep -E "${REPORT_RE}" "${TMP}/da_l1.txt" | head -1 | sed 's/ mc=[0-9]* / mc=X /' || true)"
    ra="$(grep -E "${REPORT_RE}" "${TMP}/da_la.txt" | head -1 | sed 's/ mc=[0-9]* / mc=X /' || true)"
    check_ok "L1 the tuner's threads: mc=1 with its look-ahead (${l1:-?} GB), the '--max-concurrent 1' plan" \
          "$(grep -Eq '^AUTO-TUNE-REPORT partition=2 mc=1 .*over_budget=0$' "${TMP}/da_la.txt" && [ -n "${r1}" ] \
             && [ "${ra}" = "${r1}" ] && awk -v l="${l1:-0}" 'BEGIN{exit !(l > 0)}' && echo 1 || echo 0)"
    # half way between the plan without its look-ahead and with it (60% of DTFE_RAM_GB)
    lr="$(awk -v a="${a1:-0}" -v l="${l1:-0}" 'BEGIN{if (a > 0 && l > 0.04) printf "%.4f", (a - l/2)/0.6}')"
    if [ -n "${lr}" ]; then
        lbud="$(awk -v r="${lr}" 'BEGIN{printf "%.2f", 0.6*r}')"
        env DTFE_RAM_GB="${lr}" "${BIN}" "${LR[@]}" > "${TMP}/da_lp.txt" 2>&1 || true
        env DTFE_RAM_GB="${lr}" "${BIN}" "${LR[@]}" --run-mode gentle > "${TMP}/da_lg.txt" 2>&1 || true
        env DTFE_RAM_GB="${lr}" "${BIN}" "${LR[@]}" --run-mode balanced > "${TMP}/da_lb.txt" 2>&1 || true
        rp="$(grep -E "${REPORT_RE}" "${TMP}/da_lp.txt" | head -1 || true)"
        check_ok "L2 performance at a budget of ${lbud} GB (with its look-ahead ${a1:-?} GB): mc=1, the look-ahead off (said)" \
              "$(grep -Eq '^AUTO-TUNE-REPORT partition=2 mc=1 .*lookahead_gb=0.000 over_budget=0$' "${TMP}/da_lp.txt" \
                 && grep -q 'look-ahead off for this run' "${TMP}/da_lp.txt" && echo 1 || echo 0)"
        check_ok "L3 balanced and gentle plan as performance does (the same report line, no smaller team)" \
              "$([ -n "${rp}" ] && [ "$(grep -E "${REPORT_RE}" "${TMP}/da_lb.txt" | head -1 || true)" = "${rp}" ] \
                 && [ "$(grep -E "${REPORT_RE}" "${TMP}/da_lg.txt" | head -1 || true)" = "${rp}" ] \
                 && ! grep -q 'look-ahead kept' "${TMP}/da_lb.txt" "${TMP}/da_lg.txt" && echo 1 || echo 0)"
    else
        check_ok "L a budget below one tessellation's look-ahead (not formed from ${a1:-none}, ${l1:-none} GB)" 0
    fi
    rm -f "${TMP}"/da_l*
else
    echo "   SKIP  L (this build has no GPU backend)"
fi

# ---------- (M) the uniform count plans the run's team ----------
echo ">> M: the uniform count plans the team DTFE_parallel forms"
MR=("${SNAP}" "${TMP}/da_m" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2
    --auto-tune-report)
if [ "${GPU}" = 1 ]; then MR+=(--gpu); fi   # (the GPU snapshot and the look-ahead follow the team too)
for t in 4 5; do
    env DTFE_RAM_GB=64 DTFE_TEST_TUNE_UNIFORM=1 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${MR[@]}" --max-concurrent "${t}" \
        > "${TMP}/da_m${t}.txt" 2>&1 || true
done
m4="$(grep -E "${REPORT_RE}" "${TMP}/da_m4.txt" | head -1 | sed 's/ mc=[0-9]* / /' || true)"
m5="$(grep -E "${REPORT_RE}" "${TMP}/da_m5.txt" | head -1 | sed 's/ mc=[0-9]* / /' || true)"
check_ok "M1 the uniform count (DTFE_TEST_TUNE_UNIFORM=1) plans '--max-concurrent 5' as the team of 4 it runs: the report of '--max-concurrent 4' but its mc" \
      "$(grep -q '^AUTO-TUNE-DEBUG standard: uniform' "${TMP}/da_m5.txt" && [ -n "${m4}" ] && [ "${m4}" = "${m5}" ] && echo 1 || echo 0)"
# 64^3 particles over 2^3 partitions, each of the 2 x 2 x 1 sub-boxes padded by 10 spacings of 64 per side:
# N^3/8 x (1 + 2 x 10/N x 2 x 2)^2 x (1 + 2 x 10/N x 2): 269568 at N=64 (the isotropic 4^(1/3) of before: 259067)
vm="$(sed -n 's/^AUTO-TUNE-DEBUG standard: uniform .* busiest partition [0-9]* of [0-9]*: \([0-9]*\) padded vertices.*/\1/p' "${TMP}/da_m4.txt" | head -1)"
ve="$(awk -v n="${N}" 'BEGIN{printf "%.0f", n*n*n/8 * (1 + 80/n)^2 * (1 + 40/n)}')"
check_ok "M2 ... and pads each of the team's 2 x 2 x 1 sub-boxes on its own faces: ${vm:-?} padded vertices (${ve})" \
      "$(awk -v v="${vm:-0}" -v e="${ve}" 'BEGIN{exit !(v > e*0.995 && v < e*1.005)}' && echo 1 || echo 0)"
rm -f "${TMP}"/da_m*

# ---------- (N) the --scratch-dir start-up peak ----------
echo ">> N: the zero fill of the largest scratch-backed grid at start-up (--scratch-dir)"
NSC="${TMP}/da_n_scratch"; mkdir -p "${NSC}"
NR=("${SNAP}" "${TMP}/da_n" --grid 384 "${COMMON[@]}" --field density_a velocity_a divergence_a --partition 4 4 4
    --max-concurrent 1 --auto-tune-report)
env DTFE_RAM_GB=64 "${BIN}" "${NR[@]}" --scratch-dir "${NSC}" > "${TMP}/da_n1.txt" 2>&1 || true
env DTFE_RAM_GB=64 "${BIN}" "${NR[@]}" > "${TMP}/da_n0.txt" 2>&1 || true
pn="$(field predicted_gb "${TMP}/da_n1.txt")"
# the particles (36 B a particle, 72 in the double build), the density, velocity and gradient grids (1, 3 and 9
# components) each on the heap under 1 GB and file-backed from it, the largest file-backed one zero-filled
if precision_double; then npb=72; nrb=8; else npb=36; nrb=4; fi
nexp="$(awk -v n="${N}" -v p="${npb}" -v r="${nrb}" 'BEGIN{heap = 0; big = 0; split("1 3 9", c, " ");
    for (i = 1; i <= 3; ++i) { b = 384^3 * c[i] * r; if (b < 1e9) heap += b; else if (b > big) big = b }
    printf "%.4f", (n*n*n*p + heap + big)/1e9}')"
if [ "$(uname -s)" = Darwin ]; then
    check_ok "N1 --scratch-dir (macOS): predicted ${pn:-?} GB, the particles, the heap grids and the zero fill of the 384^3 gradient (${nexp} GB), said" \
          "$(grep -q 'the predicted peak is the start-up one' "${TMP}/da_n1.txt" \
             && awk -v a="${pn:-0}" -v e="${nexp}" 'BEGIN{d = a - e; exit !(d > -0.011 && d < 0.011)}' && echo 1 || echo 0)"
else
    check_ok "N1 --scratch-dir (Linux): no start-up peak booked, the zero fill goes to the file" \
          "$(grep -Eq "${REPORT_RE}" "${TMP}/da_n1.txt" && ! grep -q 'the predicted peak is the start-up one' "${TMP}/da_n1.txt" \
             && echo 1 || echo 0)"
fi
check_ok "N2 without --scratch-dir: no start-up peak (the grids are in the plan's own bytes)" \
      "$(grep -Eq "${REPORT_RE}" "${TMP}/da_n0.txt" && ! grep -q 'the predicted peak is the start-up one' "${TMP}/da_n0.txt" \
         && echo 1 || echo 0)"
# the post phase: the derived divergence (1 component) beside the density and velocity grids, each on the heap under
# 1 GB, with the particles
nf="$(field fixed_gb "${TMP}/da_n1.txt")"
fexp="$(awk -v n="${N}" -v p="${npb}" -v r="${nrb}" 'BEGIN{heap = 0; split("1 3 9 1", c, " ");
    for (i = 1; i <= 4; ++i) { b = 384^3 * c[i] * r; if (b < 1e9) heap += b }
    printf "%.4f", (n*n*n*p + heap)/1e9}')"
check_ok "N3 --scratch-dir: after the loop the derived divergence on the heap beside the heap grids: fixed ${nf:-?} GB (${fexp})" \
      "$(awk -v a="${nf:-0}" -v e="${fexp}" 'BEGIN{d = a - e; exit !(d > -0.011 && d < 0.011)}' && echo 1 || echo 0)"
NA=("${SNAP}" "${TMP}/da_n" --grid 128 "${COMMON[@]}" --field density_a velocity_a gradient_a divergence_a shear_a
    vorticity_a tweb_a vweb_a --lambda-th 0.3 --partition 2 2 2 --max-concurrent 1 --auto-tune-report)
env DTFE_RAM_GB=64 DTFE_SCRATCH_MIN_GB=1000 "${BIN}" "${NA[@]}" --scratch-dir "${NSC}" > "${TMP}/da_n4s.txt" 2>&1 || true
env DTFE_RAM_GB=64 "${BIN}" "${NA[@]}" > "${TMP}/da_n4.txt" 2>&1 || true
r4s="$(grep -E "${REPORT_RE}" "${TMP}/da_n4s.txt" | head -1 || true)"
r4="$(grep -E "${REPORT_RE}" "${TMP}/da_n4.txt" | head -1 || true)"
check_ok "N4 --scratch-dir with no allocation at the threshold (DTFE_SCRATCH_MIN_GB=1000): the report without it" \
      "$([ -n "${r4}" ] && [ "${r4}" = "${r4s}" ] && echo 1 || echo 0)"
rm -rf "${TMP}"/da_n*

# ---------- (O) the CPU exact average's words: only for the grids with a run-wide unit ----------
echo ">> O: the CPU exact average's run-wide words in the plan (none for the scalar grids under --approx-phase-space-density)"
if precision_double; then
    echo "   SKIP  O (the double build's CPU exact sums are doubles: no words, no term)"
else
    SNAPN="${TMP}/dat_crossed${N}_phi.hdf5"
    cp "${SNAP}" "${SNAPN}"
    "${PY}" - "${SNAPN}" <<'PYEOF'
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], "a") as f:
    p = f["PartType1"]
    x = p["Coordinates"][:]
    p["Phi"] = (np.sin(2.0 * np.pi * x[:, 0] / 100.0) + 0.3 * x[:, 2] / 100.0).astype(np.float32)
PYEOF
    NR=("${SNAPN}" "${TMP}/da_n" --grid 256 "${COMMON[@]}" --field density_a scalar_a scalarGradient_a
        --scalar-dataset Phi --partition 1 1 1 --max-concurrent 1 --auto-tune-report)
    on_top() {  # the busiest partition's GB on top of 'fixed' (the AUTO-TUNE-DEBUG line): one domain's own bytes
        sed -n 's/^AUTO-TUNE-DEBUG standard: .* padded vertices, \([0-9.]*\) GB on top of .*/\1/p' "$1" | head -1
    }
    for a in plain approx; do
        na=()
        if [ "${a}" = approx ]; then na=(--approx-phase-space-density); fi
        env DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${NR[@]}" --exact ${na[@]+"${na[@]}"} \
            > "${TMP}/da_n_${a}.txt" 2>&1 || true
        env DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${NR[@]}" --sampled ${na[@]+"${na[@]}"} \
            > "${TMP}/da_n_${a}_sampled.txt" 2>&1 || true
    done
    n1="$(on_top "${TMP}/da_n_plain.txt")"; n1s="$(on_top "${TMP}/da_n_plain_sampled.txt")"
    n2="$(on_top "${TMP}/da_n_approx.txt")"; n2s="$(on_top "${TMP}/da_n_approx_sampled.txt")"
    # 256^3 cells x 8 B per component of words: density 1 + scalar 1 + its gradient 3 = 40 B (0.671 GB); the density
    # alone 8 B (0.134 GB). The sampled average sums no words on the CPU.
    nd() {  # nd <exact> <sampled twin> <lo> <hi>: 1 when both are numbers and their difference lies in (lo, hi)
        awk -v a="${1:-x}" -v b="${2:-x}" -v lo="$3" -v hi="$4" \
            'BEGIN{d = a - b; print (a != "x" && b != "x" && d > lo && d < hi) ? 1 : 0}'
    }
    check_ok "O1 the words of density_a, scalar_a and scalarGradient_a, 40 B per cell: ${n1:-?} - ${n1s:-?} GB = 0.671" \
          "$(nd "${n1}" "${n1s}" 0.665 0.677)"
    check_ok "O2 under --approx-phase-space-density the density's alone, 8 B per cell: ${n2:-?} - ${n2s:-?} GB = 0.134" \
          "$(nd "${n2}" "${n2s}" 0.128 0.140)"
    rm -f "${SNAPN}" "${TMP}"/da_n*
fi

echo "------------------------------------------------------------"
if [ "${FAILS}" -gt 0 ]; then
    echo "RESULT: FAIL (${FAILS} check(s))"
    exit 1
fi
echo "RESULT: PASS  (the standard auto-tuner reports its plan, counts the padded vertices of the run's own split,"
echo "               plans with the user's thread count, and its prediction holds the measured peak)"
echo "============================================================"
