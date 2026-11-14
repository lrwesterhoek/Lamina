#!/usr/bin/env bash
# The STANDARD-DTFE auto-tuner (src/auto_tune.h): the report, the counted split and the thread plan of PS-DTFE's
# tuner, and the memory model of the production '--exact' runs.
#
# The exact average allocates little extra (on the CPU its run-wide words: 8 B per deposited component of each
# sub-domain cell); what drives the peak is the padded vertex count. The load-balanced split
# gives a dense partition a small volume and a fixed-length padding shell, so a box-average particle count per thread
# sub-box under-counts it (TNG50-3-Dark z=0: 35.4M padded particles in one partition's 4 threads, 7.8M at the box
# average; the exact run, whose threads all wait on the GPU with their triangulations alive, peaks at 44.5 GB). The
# tuner replays the run's own split on a count grid of the particles. Verified here, on synthetic crossed waves
# (clustered knots) and the repository's binaries:
#
#  A) '--auto-tune-report' on the standard binary: ONE valid 'AUTO-TUNE-REPORT partition=.. mc=..' line (not
#     'AUTO-TUNE-REPORT unsupported'), exit 0, nothing written; a hand-set split and thread count are reported as
#     given (mc = the threads of a partition; the partitions run one after another).
#  B) the counted padded vertices (DTFE_AUTO_TUNE_DEBUG=1) against the run's own 'Thread i had N particles' lines:
#     the same busiest partition, its sum and every thread's count within 10%, at 3^3 x 8 and 4^3 x 4 threads.
#  C) the reason for B: on this clustered set the busiest partition holds well over the box-average count
#     (N / P^3 x the padding factor) -- at least 1.5x.
#  D) a user's --max-concurrent is the thread count the tuner plans with when it chooses the split (not every
#     core, whatever the user gave).
#  E) an auto-tuned run ends its SUMMARY with the 'memory model' line (predicted vs measured, as PS runs do), and the
#     measured peak stays within 1.35x of the prediction (the allocator keeps freed triangulation memory between
#     partitions: up to ~30% above the plan on sub-GB runs, while at TNG scale the prediction is 19-90% HIGH, see
#     auto_tune.h): CPU sampled average ('--sampled': the 1.35x was set on its measured peak; the exact average is
#     the default since 2026-11-09), and the GPU exact average when the build has a GPU.
#     The other runs name no mode: the standard tuner's terms for the exact average (auto_tune.cc: its CPU words and
#     its GPU words) are a few bytes per grid cell, and what A-D and F-I check -- the plan, the threads' particle
#     counts, a byte identity -- holds under either average.
#  F) the tuner only chooses: an auto-tuned run is byte-identical to the same split given by hand.
#  G) on a simulated small machine the tuner picks a split its counted model says fits (over_budget=0) and names the
#     busiest partition's vertex count in its AUTO-TUNE line.
#  H) the 2D build (DTFE-2d, when built): a valid report from the counted model, counts within 15% of the run's.
#  I) the plan's further terms, each a field of the report line: I1 a '--sample-points' plane's records
#     (pts_gb: 326 B per point with --pts-vel-grad, one record per point, in fixed_gb too; DTFE_AUTO_PTS_N sizes a plane
#     that is not on disk); on GPU builds I2 the GPU deposit's host snapshot of its grids and the growth of its device
#     words (snapshot_gb: the exact average's run-wide 128-bit words, 16 B per 4-B component, plus 8 B more per
#     component for each of two batches' device words = 8x the float grids' under DTFE_GPU_FLOAT_GRIDS=1; the 64-bit
#     words of DTFE_RUN_UNITS_LEGACY=1 2x; none on the CPU), I3 the partition loop's look-ahead (lookahead_gb, in
#     predicted_gb; none with --no-overlap or on one domain) and I4 a budget that holds the plan but not its
#     look-ahead: the look-ahead is switched off for the run (said, lookahead_gb=0, over_budget=0).
#  J) a team whose sub-domains each insert in parallel (TBB builds, 4 threads or more) books the insert's copies:
#     at '--max-concurrent 2' a forced '--parallel-triangulation' adds 800 B per padded vertex of the busiest
#     partition, and so does one tessellation ('--max-concurrent 1') where no look-ahead is booked (as on the CPU).
#  K) the look-ahead's switch-off with the insert's copies (GPU and TBB builds, 4 threads or more): K1 a budget that
#     holds a team of 2 with its look-ahead, and with the forced insert's copies, but not with both: the copies
#     switch the look-ahead off (said, lookahead_gb=0, over_budget=0, the plan with the copies predicted), without
#     them it stays; K2 one tessellation, whose copies the look-ahead's factor holds: a budget that switches the
#     look-ahead off books them in the partition again (predicted as with '--no-overlap').
#  L) '--run-mode balanced' (GPU builds): at a budget that holds 10 threads (a team of 9) but not their look-ahead,
#     and a team of 8 with its own, the plan keeps the look-ahead with 8 threads (said; the '--max-concurrent 8'
#     plan); performance, gentle and balanced with DTFE_LOOKAHEAD_TEAM=0 plan 10 threads without it.
#  M) the uniform count (DTFE_TEST_TUNE_UNIFORM=1: the plan of a call without the particles) plans the team the run
#     forms: '--max-concurrent 5' is DTFE_parallel's team of 4 (2 x 2 x 1 sub-boxes, each padded on its own faces),
#     the same report as '--max-concurrent 4' but its mc, and that team's padded vertices.
#  N) the time objective (DTFE_AUTO_TIME=1, docs/cleanup-plan.md P8.9 step 2; plan-only but N6): N1 without it (unset or
#     0) the report line is the one of before, without time keys; N2 on the CPU deposit (and in the double build)
#     it keeps the memory objective's plan and says why ('not calibrated for ...'), the report line unchanged; on GPU
#     builds (single precision) N3 with a band of 0 (DTFE_AUTO_TIME_GAIN=0: never leaves the memory plan) the report
#     line is N1's with 'time_s=X time_mem_s=X' before over_budget; N4 a budget that holds 3^3 x 10 threads and a
#     coarser split with fewer threads but not 2^3 x 10, and 1000 s a partition (DTFE_AUTO_T_STD_PART): the coarser
#     plan is taken (time_s < time_mem_s, over_budget=0, said); N5 '--sample-points' falls back (one tessellation per
#     partition); N6 a real run with the switch writes the grids of the same plan without it; N7 a plan given in full
#     is predicted and kept (time_s = time_mem_s, said).
#  O) the CPU exact average's run-wide words are booked for the grids the loop gives words, the ones with a run-wide
#     unit (single build): one domain at 256^3 with density_a, scalar_a and scalarGradient_a (a scalar dataset in a
#     copy of the snapshot), each plan against its DTFE_RUN_UNITS_LEGACY=1 twin (no words): 40 B per cell (0.671 GB)
#     where every grid has a unit, 8 B per cell (0.134 GB, the density alone) under '--approx-phase-space-density',
#     whose scalar grids keep their Real sums (DTFE.cpp runUnitsSetup). The forced unaveraged scalar is in both twins.
#
# Usage: tests/dtfe_auto_tune_check.sh [--no-build]     (DTFE_TEST_TMP, DTFE_TEST_PRECISION=double as the others)

set -euo pipefail
unset DTFE_TRIANGULATION_THREADS DTFE_TBB_THREADS   # the insert's own switch moves the plans: the suite sets none
unset DTFE_LOOKAHEAD_TEAM DTFE_TEST_TUNE_UNIFORM    # (L and M set them where they check them)
unset DTFE_AUTO_TIME DTFE_AUTO_TIME_GAIN DTFE_AUTO_T_STD_PART   # the time objective: only N sets them
unset DTFE_RUN_UNITS_LEGACY   # (I2 and O set it on their twins: an inherited one would make both twins the same)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

BIN="${DTFE_BIN}"
precision_require "${BIN}"
# The thread count is pinned to a 10-core machine's (as ps_auto_tune_check.sh H1 does): the tuner plans with every
# core, so C, D, F and G would plan other splits on a 4-core runner; the real runs (B, E, F) then use the same threads
# the plan assumed, oversubscribed on a smaller machine, which changes nothing they check.
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
check() {  # $1 = name, $2 = 0/1 ok
    if [ "$2" -eq 1 ]; then echo "   OK   $1"; else echo "   FAIL $1"; FAILS=$((FAILS+1)); fi
}
agrep() {  # set-e-safe grep assertion (ERE): $1 = check name, $2 = pattern, $3 = file
    if grep -Eq -- "$2" "$3"; then check "$1" 1; else check "$1" 0; fi
}
COMMON=(--periodic --input-format 105 --mpc-unit 1 --padding 10)
REPORT_RE='^AUTO-TUNE-REPORT partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=[0-9.]+ fixed_gb=[0-9.]+ .*over_budget=[01]$'

# ---------- (A) the report ----------
echo ">> A: --auto-tune-report on the standard binary"
rm -f "${TMP}"/da_a.*
rc=0; env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/da_a" --grid 64 "${COMMON[@]}" --field density_a velocity_a \
    --auto-tune-report > "${TMP}/da_a_log.txt" 2>&1 || rc=$?
check "A exits 0 (${rc})" "$([ "${rc}" = "0" ] && echo 1 || echo 0)"
check "A prints exactly one valid report line (not 'unsupported')" \
      "$([ "$(grep -cE "${REPORT_RE}" "${TMP}/da_a_log.txt")" = "1" ] && ! grep -q 'unsupported' "${TMP}/da_a_log.txt" && echo 1 || echo 0)"
check "A nothing is written next to the output root" "$([ -z "$(ls "${TMP}" | grep '^da_a\.')" ] && echo 1 || echo 0)"
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
    check "B ${split// /x} x ${mc}: the busiest partition and its padded vertices (sum and per thread) within 10% of the run's" \
          "$(grep -q '^OKB 1' <<< "${out}" && echo 1 || echo 0)"
    check "C ... and the run's busiest partition holds >= 1.5x the box-average count the old model used" \
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
check "E the SUMMARY of an auto-tuned standard run carries the 'memory model' line" "$(grep -q NOLINE <<< "${out}" && echo 0 || echo 1)"
check "E CPU, sampled average: measured <= 1.35x predicted" "$(grep -q 'OKE 1' <<< "${out}" && echo 1 || echo 0)"
if [ "${GPU}" = 1 ]; then
    out="$(run_e e_gpu --gpu --exact)"; printf '%s\n' "${out}" | grep '^   \.\.\.\.' || true
    check "E GPU, exact average: measured <= 1.35x predicted" "$(grep -q 'OKE 1' <<< "${out}" && echo 1 || echo 0)"
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
    check "F the tuner's split (${fsplit%%|*}, ${fsplit##*|} threads) given by hand: byte-identical .a_den and .a_vel" \
          "$([ -s "${TMP}/da_f_auto.a_den" ] && cmp -s "${TMP}/da_f_auto.a_den" "${TMP}/da_f_hand.a_den" \
             && cmp -s "${TMP}/da_f_auto.a_vel" "${TMP}/da_f_hand.a_vel" && echo 1 || echo 0)"
else
    check "F the tuner chose a split on a simulated 1.2 GB machine (no AUTO-TUNE line)" 0
fi
rm -f "${TMP}"/da_f_*.a_* "${TMP}"/da_f_*badGridPoints*

# ---------- (G) a tight machine ----------
echo ">> G: a split that fits a simulated small machine"
env DTFE_RAM_GB=1.2 "${BIN}" "${SNAP}" "${TMP}/da_g" --grid 96 "${COMMON[@]}" --field density_a velocity_a gradient_a \
    --auto-tune-report > "${TMP}/da_g.txt" 2>&1 || true
agrep "G the plan fits its budget (over_budget=0) with a split" "^AUTO-TUNE-REPORT partition=[2-6] mc=[0-9]+ .*over_budget=0$" "${TMP}/da_g.txt"
agrep "G ... and the AUTO-TUNE line names the busiest partition's padded vertices" \
      "AUTO-TUNE: .*the busiest partition triangulates ~[0-9.]+ M padded vertices at once" "${TMP}/da_g.txt"

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
    check "H the counted busiest partition within 15% of the 2D run's" "$("${PY}" - "${TMP}/da_h_rep.txt" "${TMP}/da_h_run.txt" <<'PYEOF'
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
field() { sed -n "s/^AUTO-TUNE-REPORT .* $1=\([0-9.]*\) .*/\1/p; s/^AUTO-TUNE-REPORT .* $1=\([0-9.]*\)$/\1/p" "$2" | head -1; }
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
check "I1 a 2048^2 plane with --pts-vel-grad: pts_gb ${p1:-?} (4194304 x 326 B = 1.367), in fixed_gb (${f0:-?} -> ${f1:-?}), one record per point" "${ok}"
s0="$(field snapshot_gb "${TMP}/da_i0.txt")"; l0="$(field lookahead_gb "${TMP}/da_i0.txt")"
check "I2/I3 a CPU run counts no GPU snapshot and no look-ahead (${s0:-?}, ${l0:-?})" \
      "$([ "${s0}" = "0.000" ] && [ "${l0}" = "0.000" ] && echo 1 || echo 0)"
if [ "${GPU}" = 1 ]; then
    env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu > "${TMP}/da_i2.txt" 2>&1 || true
    env DTFE_RAM_GB=64 DTFE_GPU_FLOAT_GRIDS=1 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu \
        > "${TMP}/da_i3.txt" 2>&1 || true
    env DTFE_RAM_GB=64 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu --no-overlap > "${TMP}/da_i4.txt" 2>&1 || true
    env DTFE_RAM_GB=64 DTFE_RUN_UNITS_LEGACY=1 "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu \
        > "${TMP}/da_i6.txt" 2>&1 || true
    s2="$(field snapshot_gb "${TMP}/da_i2.txt")"; s3="$(field snapshot_gb "${TMP}/da_i3.txt")"
    s6="$(field snapshot_gb "${TMP}/da_i6.txt")"
    # (the GB carry three decimals: the float grids' ~0.02 GB leaves the ratios a few per cent of play)
    check "I2 --gpu: snapshot + device words of the largest sub-grid ${s2:-?} GB = 8x the float grids (${s3:-?} GB), \
2x under DTFE_RUN_UNITS_LEGACY=1 (${s6:-?} GB)" \
          "$(awk -v a="${s2:-0}" -v b="${s3:-0}" -v c="${s6:-0}" \
             'BEGIN{print (a > 0.02 && b > 0. && a / b > 7.7 && a / b < 8.3 && c / b > 1.95 && c / b < 2.05) ? 1 : 0}')"
    l2="$(field lookahead_gb "${TMP}/da_i2.txt")"; l4="$(field lookahead_gb "${TMP}/da_i4.txt")"
    q2="$(field predicted_gb "${TMP}/da_i2.txt")"; q4="$(field predicted_gb "${TMP}/da_i4.txt")"
    check "I3 --gpu over 8 partitions: the look-ahead (${l2:-?} GB) is in predicted_gb (${q4:-?} -> ${q2:-?}); none with --no-overlap (${l4:-?})" \
          "$(awk -v l="${l2:-0}" -v m="${l4:-1}" -v a="${q4:-0}" -v b="${q2:-0}" 'BEGIN{d = b - a - l; print (l > 0.01 && m == 0 && d > -0.011 && d < 0.011) ? 1 : 0}')"
    # a budget between the plan without the look-ahead and the plan with it (the budget is 60% of DTFE_RAM_GB)
    ram="$(awk -v a="${q4:-0}" -v l="${l2:-0}" 'BEGIN{printf "%.4f", (a + l / 2.) / 0.6}')"
    env DTFE_RAM_GB="${ram}" "${BIN}" "${IR[@]}" --max-concurrent 2 --gpu > "${TMP}/da_i5.txt" 2>&1 || true
    check "I4 a budget that holds the plan (${q4:-?} GB) but not its look-ahead: switched off, said, over_budget=0" \
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
        --auto-tune-report)
    for t in 1 2; do
        env DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${JR[@]}" --max-concurrent "${t}" > "${TMP}/da_j${t}.txt" 2>&1 || true
        env DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${JR[@]}" --max-concurrent "${t}" --parallel-triangulation \
            > "${TMP}/da_j${t}f.txt" 2>&1 || true
    done
    busiest() { sed -n 's/^AUTO-TUNE-DEBUG standard: .* busiest partition [0-9]* of [0-9]*: \([0-9]*\) padded vertices.*/\1/p' "$1" | head -1; }
    # the forced plan's busiest partition can be another one: the rise lies between 800 B x either one's vertices
    rise() { awk -v a="${1:-0}" -v b="${2:-0}" -v u="${3:-0}" -v f="${4:-0}" \
                 'BEGIN{e = b - a; print (u > 0 && f > 0 && e > 800.*u/1.e9 - 0.011 && e < 800.*f/1.e9 + 0.011) ? 1 : 0}'; }
    u1="$(busiest "${TMP}/da_j1.txt")"; f1="$(busiest "${TMP}/da_j1f.txt")"
    u2="$(busiest "${TMP}/da_j2.txt")"; f2="$(busiest "${TMP}/da_j2f.txt")"
    a1="$(field predicted_gb "${TMP}/da_j1.txt")"; b1="$(field predicted_gb "${TMP}/da_j1f.txt")"
    a2="$(field predicted_gb "${TMP}/da_j2.txt")"; b2="$(field predicted_gb "${TMP}/da_j2f.txt")"
    check "J a team of 2 books the forced insert's copies (${a2:-?} -> ${b2:-?} GB: 800 B x ${u2:-?} to ${f2:-?} padded vertices)" \
          "$(rise "${a2}" "${b2}" "${u2}" "${f2}")"
    check "J one tessellation without a look-ahead books them too (${a1:-?} -> ${b1:-?} GB: 800 B x ${u1:-?} to ${f1:-?})" \
          "$(rise "${a1}" "${b1}" "${u1}" "${f1}")"
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
    # K1, a team of 2: q the plan without its look-ahead, b with the forced insert's copies (the team books them), l
    # its look-ahead; the budget holds q + l and b each by x and not b + l (x short), x half the smaller of the two
    k_rep q2 64 --max-concurrent 2 --no-overlap
    k_rep b2 64 --max-concurrent 2 --no-overlap --parallel-triangulation
    k_rep l2 64 --max-concurrent 2
    q2="$(field predicted_gb "${TMP}/da_k_q2.txt")"; b2="$(field predicted_gb "${TMP}/da_k_b2.txt")"
    l2="$(field lookahead_gb "${TMP}/da_k_l2.txt")"
    kr="$(awk -v q="${q2:-0}" -v b="${b2:-0}" -v l="${l2:-0}" 'BEGIN{t = b - q; x = (t < l ? t : l)/2; m = (q + l > b ? q + l : b);
        if (q > 0 && x > 0.02) printf "%.4f", (m + x)/0.6}')"
    if [ -n "${kr}" ]; then
        k_rep on2 "${kr}" --max-concurrent 2
        k_rep off2 "${kr}" --max-concurrent 2 --parallel-triangulation
        o2="$(field lookahead_gb "${TMP}/da_k_on2.txt")"; p2="$(field predicted_gb "${TMP}/da_k_off2.txt")"
        check "K1 a budget of $(awk -v r="${kr}" 'BEGIN{printf "%.2f", 0.6*r}') GB holds a team of 2 with its look-ahead (${q2:-?} + ${l2:-?} GB) and with the forced insert's copies (${b2:-?} GB), not with both: switched off (said, predicted ${p2:-?} GB, lookahead_gb=0, over_budget=0); kept without them (lookahead_gb ${o2:-?})" \
              "$(grep -q 'look-ahead off for this run' "${TMP}/da_k_off2.txt" \
                 && grep -Eq 'lookahead_gb=0.000 over_budget=0$' "${TMP}/da_k_off2.txt" \
                 && ! grep -q 'look-ahead off' "${TMP}/da_k_on2.txt" && grep -Eq 'over_budget=0$' "${TMP}/da_k_on2.txt" \
                 && awk -v o="${o2:-0}" -v l="${l2:-x}" -v p="${p2:-0}" -v b="${b2:-0}" \
                        'BEGIN{d = p - b; exit !(o == l && d > -0.011 && d < 0.011)}' && echo 1 || echo 0)"
    else
        check "K1 the budget between the bookings (not formed from ${q2:-none}, ${b2:-none}, ${l2:-none} GB)" 0
    fi
    # K2, one tessellation: the look-ahead's factor holds its copies, so the forced insert books none while the
    # look-ahead runs; a budget half a look-ahead above the plan switches it off, and the partition books them again:
    # predicted as '--no-overlap' predicts (b1), not as without the copies (q1)
    k_rep q1 64 --max-concurrent 1 --no-overlap
    k_rep b1 64 --max-concurrent 1 --no-overlap --parallel-triangulation
    k_rep l1 64 --max-concurrent 1 --parallel-triangulation
    q1="$(field predicted_gb "${TMP}/da_k_q1.txt")"; b1="$(field predicted_gb "${TMP}/da_k_b1.txt")"
    l1="$(field lookahead_gb "${TMP}/da_k_l1.txt")"
    kr="$(awk -v q="${q1:-0}" -v b="${b1:-0}" -v l="${l1:-0}" 'BEGIN{if (q > 0 && b - q > 0.03 && l > 0.04) printf "%.4f", (q + l/2)/0.6}')"
    if [ -n "${kr}" ]; then
        k_rep off1 "${kr}" --max-concurrent 1 --parallel-triangulation
        k_rep seq1 "${kr}" --max-concurrent 1
        p1="$(field predicted_gb "${TMP}/da_k_off1.txt")"; s1="$(field predicted_gb "${TMP}/da_k_seq1.txt")"
        check "K2 one tessellation, a budget of $(awk -v r="${kr}" 'BEGIN{printf "%.2f", 0.6*r}') GB (half its look-ahead of ${l1:-?} GB above the plan): switched off, the forced insert's copies booked again (${p1:-?} = ${b1:-?} GB), none without the flag (${s1:-?} = ${q1:-?} GB)" \
              "$(grep -q 'look-ahead off for this run' "${TMP}/da_k_off1.txt" && grep -q 'look-ahead off for this run' "${TMP}/da_k_seq1.txt" \
                 && grep -Eq 'lookahead_gb=0.000 over_budget=[01]$' "${TMP}/da_k_off1.txt" \
                 && awk -v p="${p1:-0}" -v b="${b1:-x}" -v s="${s1:-0}" -v q="${q1:-x}" \
                        'BEGIN{d = p - b; e = s - q; exit !(d > -0.011 && d < 0.011 && e > -0.011 && e < 0.011)}' \
                 && echo 1 || echo 0)"
    else
        check "K2 the budget under the look-ahead (not formed from ${q1:-none}, ${b1:-none}, ${l1:-none} GB)" 0
    fi
    rm -f "${TMP}"/da_k*
else
    echo "   SKIP  K (a build without the GPU or the parallel insert, or fewer than 4 threads)"
fi

# ---------- (L) --run-mode balanced: a smaller team that keeps the look-ahead ----------
echo ">> L: '--run-mode balanced' keeps the look-ahead with a smaller team"
if [ "${GPU}" = 1 ]; then
    LR=("${SNAP}" "${TMP}/da_l" --grid 192 "${COMMON[@]}" --field density_a velocity_a gradient_a --partition 2 2 2 --gpu
        --auto-tune-report)
    env DTFE_RAM_GB=64 "${BIN}" "${LR[@]}" --max-concurrent 10 > "${TMP}/da_l10.txt" 2>&1 || true
    env DTFE_RAM_GB=64 "${BIN}" "${LR[@]}" --max-concurrent 8 > "${TMP}/da_l8.txt" 2>&1 || true
    a10="$(field predicted_gb "${TMP}/da_l10.txt")"; a8="$(field predicted_gb "${TMP}/da_l8.txt")"
    l10="$(field lookahead_gb "${TMP}/da_l10.txt")"; l8="$(field lookahead_gb "${TMP}/da_l8.txt")"
    # half way between the team of 8 and the team of 9 (10 threads), each with its look-ahead (OMP_NUM_THREADS=10 is
    # every mode's thread count here); 10 threads fit without theirs
    lr="$(awk -v a="${a10:-0}" -v b="${a8:-0}" -v l="${l10:-0}" 'BEGIN{x = (a - b)/2;
        if (b > 0 && x > 0.02 && l - x > 0.02) printf "%.4f", (b + x)/0.6}')"
    if [ -n "${lr}" ]; then
        env DTFE_RAM_GB="${lr}" "${BIN}" "${LR[@]}" > "${TMP}/da_lp.txt" 2>&1 || true
        env DTFE_RAM_GB="${lr}" "${BIN}" "${LR[@]}" --run-mode gentle > "${TMP}/da_lg.txt" 2>&1 || true
        env DTFE_RAM_GB="${lr}" "${BIN}" "${LR[@]}" --run-mode balanced > "${TMP}/da_lb.txt" 2>&1 || true
        env DTFE_RAM_GB="${lr}" DTFE_LOOKAHEAD_TEAM=0 "${BIN}" "${LR[@]}" --run-mode balanced > "${TMP}/da_lb0.txt" 2>&1 || true
        lbud="$(awk -v r="${lr}" 'BEGIN{printf "%.2f", 0.6*r}')"
        rp="$(grep -E "${REPORT_RE}" "${TMP}/da_lp.txt" | head -1 || true)"
        mb="$(sed -n 's/^AUTO-TUNE-REPORT partition=[0-9]* mc=\([0-9]*\) .*/\1/p' "${TMP}/da_lb.txt" | head -1)"
        pb="$(field predicted_gb "${TMP}/da_lb.txt")"; lb="$(field lookahead_gb "${TMP}/da_lb.txt")"
        check "L1 performance at a budget of ${lbud} GB (10 threads with their look-ahead ${a10:-?} GB, 8 with theirs ${a8:-?} GB): 10 threads, the look-ahead off (said)" \
              "$(grep -Eq '^AUTO-TUNE-REPORT partition=2 mc=10 .*lookahead_gb=0.000 over_budget=0$' "${TMP}/da_lp.txt" \
                 && grep -q 'look-ahead off for this run' "${TMP}/da_lp.txt" && echo 1 || echo 0)"
        check "L2 balanced: 8 threads keep the look-ahead (mc=${mb:-?}, lookahead_gb ${lb:-?} = ${l8:-?}, predicted ${pb:-?} = ${a8:-?} GB, over_budget=0), said" \
              "$([ "${mb:-}" = 8 ] && [ "${lb:-x}" = "${l8:-y}" ] && [ "${pb:-x}" = "${a8:-y}" ] \
                 && grep -q 'look-ahead kept for this run with 8 threads per partition instead of 10' "${TMP}/da_lb.txt" \
                 && grep -Eq '^AUTO-TUNE-REPORT .*over_budget=0$' "${TMP}/da_lb.txt" && ! grep -q 'look-ahead off' "${TMP}/da_lb.txt" \
                 && echo 1 || echo 0)"
        check "L3 balanced with DTFE_LOOKAHEAD_TEAM=0, and gentle, plan as performance does (the same report line)" \
              "$([ -n "${rp}" ] && [ "$(grep -E "${REPORT_RE}" "${TMP}/da_lb0.txt" | head -1 || true)" = "${rp}" ] \
                 && [ "$(grep -E "${REPORT_RE}" "${TMP}/da_lg.txt" | head -1 || true)" = "${rp}" ] \
                 && ! grep -q 'look-ahead kept' "${TMP}/da_lb0.txt" "${TMP}/da_lg.txt" && echo 1 || echo 0)"
    else
        check "L the budget between the two teams (not formed from ${a10:-none}, ${a8:-none}, ${l10:-none} GB)" 0
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
check "M1 the uniform count (DTFE_TEST_TUNE_UNIFORM=1) plans '--max-concurrent 5' as the team of 4 it runs: the report of '--max-concurrent 4' but its mc" \
      "$(grep -q '^AUTO-TUNE-DEBUG standard: uniform' "${TMP}/da_m5.txt" && [ -n "${m4}" ] && [ "${m4}" = "${m5}" ] && echo 1 || echo 0)"
# 64^3 particles over 2^3 partitions, each of the 2 x 2 x 1 sub-boxes padded by 10 spacings of 64 per side:
# N^3/8 x (1 + 2 x 10/N x 2 x 2)^2 x (1 + 2 x 10/N x 2): 269568 at N=64 (the isotropic 4^(1/3) of before: 259067)
vm="$(sed -n 's/^AUTO-TUNE-DEBUG standard: uniform .* busiest partition [0-9]* of [0-9]*: \([0-9]*\) padded vertices.*/\1/p' "${TMP}/da_m4.txt" | head -1)"
ve="$(awk -v n="${N}" 'BEGIN{printf "%.0f", n*n*n/8 * (1 + 80/n)^2 * (1 + 40/n)}')"
check "M2 ... and pads each of the team's 2 x 2 x 1 sub-boxes on its own faces: ${vm:-?} padded vertices (${ve})" \
      "$(awk -v v="${vm:-0}" -v e="${ve}" 'BEGIN{exit !(v > e*0.995 && v < e*1.005)}' && echo 1 || echo 0)"
rm -f "${TMP}"/da_m*

# ---------- (N) the time objective (DTFE_AUTO_TIME=1) ----------
echo ">> N: the time objective (DTFE_AUTO_TIME)"
# DTFE_AUTO_MINN=1000: this 64^3 set counts as large, as the tuner's other checks of the large-set path do
NR=("${SNAP}" "${TMP}/da_n" --grid 48 "${COMMON[@]}" --field density_a velocity_a gradient_a --auto-tune-report)
n_rep() {  # $1 = log, then the environment's assignments, '--', the options: one report run
    local log="$1"; shift
    local envs=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$#" -gt 0 ] && shift
    env DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 ${envs[@]+"${envs[@]}"} "${BIN}" "${NR[@]}" "$@" > "${log}" 2>&1 || true
}
n_line() { grep -E "${REPORT_RE}" "$1" | head -1 || true; }   # the report line
n_strip() { sed -E 's/ time_s=[0-9]+ time_mem_s=[0-9]+ over_budget=/ over_budget=/'; }   # ... without the time keys
n_keys() {  # the report line's 'time_s time_mem_s', empty without them
    sed -n -E 's/^AUTO-TUNE-REPORT .* time_s=([0-9]+) time_mem_s=([0-9]+) over_budget=[01]$/\1 \2/p' "$1" | head -1
}
TIME_LINE="AUTO-TUNE: time objective (DTFE_AUTO_TIME):"
n_rep "${TMP}/da_n1a.txt" --
n_rep "${TMP}/da_n1b.txt" DTFE_AUTO_TIME=0 --
r1="$(n_line "${TMP}/da_n1a.txt")"
check "N1 without the switch (unset, or DTFE_AUTO_TIME=0): the same report line, no time keys, no time line" \
      "$([ -n "${r1}" ] && [ "$(n_line "${TMP}/da_n1b.txt")" = "${r1}" ] \
         && ! grep -q 'time_s=' "${TMP}/da_n1a.txt" "${TMP}/da_n1b.txt" \
         && ! grep -qF -e "${TIME_LINE}" "${TMP}/da_n1a.txt" "${TMP}/da_n1b.txt" && echo 1 || echo 0)"
NWHY="the CPU deposit"; if precision_double; then NWHY="the double build"; fi
n_rep "${TMP}/da_n2.txt" DTFE_AUTO_TIME=1 --
check "N2 DTFE_AUTO_TIME=1 with ${NWHY}: the memory objective's plan, said; the report line of N1" \
      "$(grep -qF -e "${TIME_LINE} not calibrated for ${NWHY}; the memory objective's plan." "${TMP}/da_n2.txt" \
         && [ "$(n_line "${TMP}/da_n2.txt")" = "${r1}" ] && echo 1 || echo 0)"
if [ "${GPU}" = 1 ] && ! precision_double; then
    NG=(--gpu --exact --face-gradient)
    # N3: a band of 0 never leaves the memory objective's plan: its report line gains the two keys, equal
    n_rep "${TMP}/da_n3a.txt" -- "${NG[@]}"
    n_rep "${TMP}/da_n3b.txt" DTFE_AUTO_TIME=1 DTFE_AUTO_TIME_GAIN=0 -- "${NG[@]}"
    r3="$(n_line "${TMP}/da_n3a.txt")"; k3="$(n_keys "${TMP}/da_n3b.txt")"
    check "N3 --gpu, a band of 0: the switch-off report line with time_s = time_mem_s (${k3:-none}) before over_budget, and the time line" \
          "$([ -n "${r3}" ] && [ -n "${k3}" ] && [ "${k3% *}" = "${k3#* }" ] \
             && [ "$(n_line "${TMP}/da_n3b.txt" | n_strip)" = "${r3}" ] \
             && grep -qF -e "${TIME_LINE} this plan predicts ~" "${TMP}/da_n3b.txt" && echo 1 || echo 0)"
    # N4: a forced departure, without the look-ahead (its switch-off would move the bookings): the plans' predictions
    # given by hand are chooseSplit's own test quantity
    NP=("${NG[@]}" --no-overlap)
    n_rep "${TMP}/da_n4q22.txt" -- "${NP[@]}" --partition 2 2 2 --max-concurrent 2
    n_rep "${TMP}/da_n4q210.txt" -- "${NP[@]}" --partition 2 2 2 --max-concurrent 10
    n_rep "${TMP}/da_n4q310.txt" -- "${NP[@]}" --partition 3 3 3 --max-concurrent 10
    q22="$(field predicted_gb "${TMP}/da_n4q22.txt")"; q210="$(field predicted_gb "${TMP}/da_n4q210.txt")"
    q310="$(field predicted_gb "${TMP}/da_n4q310.txt")"
    # a budget 0.02 GB above the larger of 2^3 x 2 and 3^3 x 10 threads and at least 0.02 GB below 2^3 x 10 (the
    # budget is 60% of DTFE_RAM_GB): the memory objective plans 3^3 x 10, and a coarser split fits with fewer threads
    nr="$(awk -v a="${q22:-0}" -v b="${q210:-0}" -v c="${q310:-0}" 'BEGIN{m = (a > c ? a : c) + 0.02;
        if (a > 0 && c > 0 && m + 0.02 < b) printf "%.4f", m / 0.6}')"
    if [ -n "${nr}" ]; then
        nbud="$(awk -v r="${nr}" 'BEGIN{printf "%.2f", 0.6*r}')"
        n_rep "${TMP}/da_n4m.txt" DTFE_RAM_GB="${nr}" -- "${NP[@]}"
        n_rep "${TMP}/da_n4t.txt" DTFE_RAM_GB="${nr}" DTFE_AUTO_TIME=1 DTFE_AUTO_TIME_GAIN=1 DTFE_AUTO_T_STD_PART=1000 \
            -- "${NP[@]}"
        k4="$(n_keys "${TMP}/da_n4t.txt")"
        check "N4 a budget of ${nbud} GB (2^3 x 2: ${q22} GB, 3^3 x 10: ${q310} GB, 2^3 x 10: ${q210} GB): the memory objective plans 3^3 x 10 threads, the time objective at 1000 s a partition a coarser split (time_s time_mem_s: ${k4:-none}), over_budget=0, said" \
              "$(grep -Eq '^AUTO-TUNE-REPORT partition=3 mc=10 ' "${TMP}/da_n4m.txt" \
                 && grep -Eq '^AUTO-TUNE-REPORT partition=[12] mc=[0-9]+ .*over_budget=0$' "${TMP}/da_n4t.txt" \
                 && [ -n "${k4}" ] && [ "${k4% *}" -lt "${k4#* }" ] \
                 && grep -qF -e "for the memory objective's --partition 3 3 3, 10 thread(s), look-ahead off." \
                        "${TMP}/da_n4t.txt" && echo 1 || echo 0)"
    else
        echo "   SKIP  N4 (no budget between 2^3 x 2 / 3^3 x 10 threads, ${q22:-?} / ${q310:-?} GB, and 2^3 x 10, ${q210:-?} GB)"
    fi
    # N5: '--sample-points' (one tessellation per partition) is not calibrated
    "${PY}" - "${TMP}/da_n_plane.bin" "${BOX}" <<'PYEOF'
import sys, numpy as np
c = (np.arange(16) + 0.5) * float(sys.argv[2]) / 16
X, Y = np.meshgrid(c, c, indexing="ij")
np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.5 * float(sys.argv[2]))], 1).astype(np.float64).tofile(sys.argv[1])
PYEOF
    n_rep "${TMP}/da_n5.txt" DTFE_AUTO_TIME=1 -- --gpu --exact --sample-points "${TMP}/da_n_plane.bin"
    check "N5 --sample-points: the memory objective's plan, said (one tessellation per partition), no time keys" \
          "$(grep -qF -e "${TIME_LINE} not calibrated for --sample-points (one tessellation per partition)" \
                "${TMP}/da_n5.txt" && ! grep -q 'time_s=' "${TMP}/da_n5.txt" && echo 1 || echo 0)"
    # N6: a real run under the switch (a band of 0: the plan stays the memory objective's) writes the grids of the
    # same run without it
    rc=0
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_AUTO_MINN=1000 DTFE_AUTO_TIME=1 DTFE_AUTO_TIME_GAIN=0 "${BIN}" "${SNAP}" \
        "${TMP}/da_n6on" --grid 48 "${COMMON[@]}" --field density_a velocity_a gradient_a "${NG[@]}" \
        > "${TMP}/da_n6on.txt" 2>&1 || rc=$?
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" \
        "${TMP}/da_n6off" --grid 48 "${COMMON[@]}" --field density_a velocity_a gradient_a "${NG[@]}" \
        > "${TMP}/da_n6off.txt" 2>&1 || true
    check "N6 a real run with the switch (exit ${rc}) states its prediction and writes the switch-off run's .a_den and .a_vel byte for byte" \
          "$([ "${rc}" = 0 ] && grep -qF -e "${TIME_LINE} this plan predicts ~" "${TMP}/da_n6on.txt" \
             && [ -s "${TMP}/da_n6on.a_den" ] && cmp -s "${TMP}/da_n6on.a_den" "${TMP}/da_n6off.a_den" \
             && cmp -s "${TMP}/da_n6on.a_vel" "${TMP}/da_n6off.a_vel" && echo 1 || echo 0)"
    # N7: a plan given in full is predicted, never changed
    n_rep "${TMP}/da_n7a.txt" -- "${NG[@]}" --partition 3 3 3 --max-concurrent 10
    n_rep "${TMP}/da_n7b.txt" DTFE_AUTO_TIME=1 -- "${NG[@]}" --partition 3 3 3 --max-concurrent 10
    r7="$(n_line "${TMP}/da_n7a.txt")"; k7="$(n_keys "${TMP}/da_n7b.txt")"
    check "N7 a plan given in full: kept (the switch-off report line), predicted (time_s = time_mem_s: ${k7:-none}), said" \
          "$([ -n "${r7}" ] && [ -n "${k7}" ] && [ "${k7% *}" = "${k7#* }" ] \
             && [ "$(n_line "${TMP}/da_n7b.txt" | n_strip)" = "${r7}" ] \
             && grep -qF -e "--partition and --max-concurrent are given, so it is only predicted." "${TMP}/da_n7b.txt" \
             && echo 1 || echo 0)"
else
    echo "   SKIP  N3-N7 (a build without the GPU backend, or the double build: the time objective falls back there, N2)"
fi
rm -f "${TMP}"/da_n*

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
        --scalar-dataset Phi --exact --partition 1 1 1 --max-concurrent 1 --auto-tune-report)
    on_top() {  # the busiest partition's GB on top of 'fixed' (the AUTO-TUNE-DEBUG line): one domain's own bytes
        sed -n 's/^AUTO-TUNE-DEBUG standard: .* padded vertices, \([0-9.]*\) GB on top of .*/\1/p' "$1" | head -1
    }
    for a in plain approx; do
        na=()
        if [ "${a}" = approx ]; then na=(--approx-phase-space-density); fi
        env DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${NR[@]}" ${na[@]+"${na[@]}"} \
            > "${TMP}/da_n_${a}.txt" 2>&1 || true
        env DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 DTFE_RUN_UNITS_LEGACY=1 "${BIN}" "${NR[@]}" ${na[@]+"${na[@]}"} \
            > "${TMP}/da_n_${a}_legacy.txt" 2>&1 || true
    done
    n1="$(on_top "${TMP}/da_n_plain.txt")"; n1l="$(on_top "${TMP}/da_n_plain_legacy.txt")"
    n2="$(on_top "${TMP}/da_n_approx.txt")"; n2l="$(on_top "${TMP}/da_n_approx_legacy.txt")"
    # 256^3 cells x 8 B per component: density 1 + scalar 1 + its gradient 3 = 40 B (0.671 GB); the density alone 8 B
    # (0.134 GB). The GB carry three decimals, so each difference is good to 0.001.
    nd() {  # nd <with words> <legacy twin> <lo> <hi>: 1 when both are numbers and their difference lies in (lo, hi)
        awk -v a="${1:-x}" -v b="${2:-x}" -v lo="$3" -v hi="$4" \
            'BEGIN{d = a - b; print (a != "x" && b != "x" && d > lo && d < hi) ? 1 : 0}'
    }
    check "O1 the words of density_a, scalar_a and scalarGradient_a, 40 B per cell: ${n1:-?} - ${n1l:-?} GB = 0.671" \
          "$(nd "${n1}" "${n1l}" 0.665 0.677)"
    check "O2 under --approx-phase-space-density the density's alone, 8 B per cell: ${n2:-?} - ${n2l:-?} GB = 0.134" \
          "$(nd "${n2}" "${n2l}" 0.128 0.140)"
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
