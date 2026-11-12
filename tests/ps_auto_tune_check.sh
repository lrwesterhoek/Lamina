#!/usr/bin/env bash
# Auto-tune (src/auto_tune.h) behaviour checks.
# Drives the REAL binary through the documented env overrides (DTFE_RAM_GB simulates the
# machine's RAM, DTFE_AUTO_MINN lowers the particle-count gate) so every scenario is fast,
# deterministic and allocation-safe. Verified here:
#
#  A) OVER-BUDGET REFUSAL. When the irreducible full-grid accumulators alone exceed the
#     budget, auto-tune must say so BEFORE the search -- naming the grid term, stating that
#     --partition cannot reduce it, suggesting the largest grid that WOULD fit, and forcing
#     --max-concurrent 1 -- not pick '--partition 2 2 2', blame the split, and advise "a smaller
#     grid or fewer fields" while predicting a bogus peak.
#     This fires even when N < the small-N gate: the grid term does not depend on N.
#  B) NORMAL TUNING. With RAM to spare it picks a partition and concurrency and prints a
#     predicted peak.
#  C) CONSERVATIVE CONTRACT. The header promises the model "deliberately errs HIGH so the
#     tuner never picks a configuration that measured tighter than predicted": on a real
#     partitioned 256^3 run (volume-weighted, dispersion -- fields whose extra grids the model
#     must count), predicted peak >= measured peak RSS.
#  D) EXPLICIT FLAGS WIN. --partition/--max-concurrent given by the user are never overridden --
#     including '--partition 1 1 1', which asks for a single triangulation. Both given, the plan is still
#     predicted (the memory gate and the parking room come from it): the AUTO-TUNE line states the user's plan.
#  E) THE MEMORY BUDGET. 60% of RAM by default (80% lets a 64 GB Mac swap 19 GB with a desktop
#     open next to the run); DTFE_MEM_FRACTION and DTFE_MEM_BUDGET_GB override it; a simulated
#     machine (DTFE_RAM_GB) skips the live "what do other programs leave free" term, and the live
#     run states its budget.
#  G) the stream estimate of a non-periodic cloud larger than its output region stays physical.
#  H) the TIME-AWARE split of a small set: a run whose time is the sequential
#     triangulation (a slice of 0.26M particles) is split 2^3 for speed; a deposit-bound run
#     (a 256^3 grid) and a GPU run (its deposit serialises on the GPU) keep one domain, and so
#     does a non-periodic cloud that makes no periodic copies (the model must not charge it 1.7x);
#     the single-domain report predicts a sane peak (the GUI's memory check believes it); and two
#     auto-split runs give bit-identical grids (the split merges its partitions in index order).
#  I) Very fine grids: GPU sub-grids kept under 2^32 cells by the split; a small set on a huge grid
#     splits for memory instead of staying in one domain over budget.
#  F) --auto-tune-report (the GUI's memory check): ONE machine-readable line, exit 0, nothing
#     written; a hand-set --partition/--max-concurrent is reported as given (the prediction uses
#     the concurrency the run will use); DTFE_AUTO_PTS_N sizes a plane that does not exist yet.
#  J) the machine's free memory over a plan: an explicit budget capped by it, the partition memory gate.
#  M) the GPU deposit's host snapshot (ps_metal_host.cc GpuSnapshot) is in the prediction, ONCE (the deposits serialise).
#  N) the '--sample-points' stream estimate is made on the points' own slab: on a 1-D pancake
#     (80^3, A k = 1.8; the 3-stream band covers ~16% of the box, box mean 1.33) a plane outside the band
#     estimates ~1 stream/point and one through it ~3 (both measured exactly in '.pts_streams'), each within
#     10%; the box-wide estimate (DTFE_AUTO_PTS_BOXMEAN=1) gives ~1.3 for both. On the TNG50-3-Dark
#     z=0 z2048 plane the box mean 17.98 budgets 34.95 streams/point (19.3 GB) while its points average 5.38.
#  O) '--parallel-triangulation' (TBB builds) is costed: its insert copies (800 B per vertex in flight,
#     measured) raise the predicted peak; O2 DTFE_TRIANGULATION_THREADS forces them (=1 none); O3 the split search's
#     own count books them: a budget that holds 2 in flight without them and 1 with them plans 1; O4 at one in flight
#     too: a budget that holds one 2^3 partition without them and not with them (the split left to the tuner, one
#     thread) plans a finer split within it. SKIP without TBB; O2-O4 also below 4 threads.
#  P) the tessellation model pads a partition by the binary's DEFAULT Lagrangian padding ('--ps-lagrangian-padding
#     8' spacings, not the Eulerian '--padding' fraction floored at 0.3 of a partition): the default predicts what
#     an explicit '--ps-lagrangian-padding 8' does, and '--ps-lagrangian-padding 0' (the Eulerian rule) predicts something else.
#  K) partitions in flight never outnumber the threads: neither a hand-set split with the
#     concurrency left to the tuner nor '--max-concurrent 0' ("all threads") starts every
#     partition at once (64 threads for '--partition 4 4 4').
#  R) STANDARD DTFE '--region' / '--region-mpc': the tuner plans the region's particles, not the whole
#     snapshot's (planning the snapshot splits the launcher's exact zoom of a 20^3-cell TNG50 region 3x3x3 with
#     10 threads: ~21 min instead of ~1). Uses the standard binary (built here too); SKIP when it is not built.
#  S) ONE DOMAIN: '--partition 1 1 1' deposits into the run's own grids, so the prediction counts them
#     once -- the 768^3 crossed-waves report sits between the measured peak and 15% above it --
#     and a real one-domain run's peak stays under its prediction.
#  W) --grid-window: every partition's sub-grid is the WHOLE window (not the crop of a full-grid partition), its GPU
#     deposit's grids are held once: a window+exact run's prediction covers its measured peak, and on macOS stays
#     within 1.5x of it. GPU builds only.
#  X) the partition sub-grid the plan books (plan P8.2: measured on the particles, psPartitionSpans) is never
#     below the largest one the run allocates (DTFE_AUTO_TUNE_DEBUG=1: the planner's 'booked sub-grid' line against
#     every pass's 'sub-grid' line), on the cases a review found wrong: one domain of a non-periodic clump (the whole
#     grid, never the particles' span), crossed waves with '--ps-lagrangian-padding 0' (the Eulerian padding moves the
#     periodic seam), splits that are not cubic (1 2 2, 2 3 4, 4 1 1), and the default 2^3 on crossed waves.
#  Y) DYNAMIC CONCURRENCY (plan P8.10): over a concurrency the tuner chose (DTFE_TEST_PLAN_MC=1 stands for a plan
#     of one partition at a time), a run whose free memory holds more starts more (up to the cores) and writes the
#     '--max-concurrent 1' run's grids byte for byte (the ordered merge); one whose free memory holds no second
#     partition keeps the plan's 1, as do DTFE_DYNAMIC_CONCURRENCY=0 and a user's '--max-concurrent'; room for about
#     five admits 2-6, not every waiting thread at once (the gate counts what it admitted); the double build keeps the
#     plan's (its sums follow the thread order); a dynamic run killed after 3 partitions and resumed writes the same grids.
#     Y7 (GPU builds): an all-GPU run's ledger books the GPU snapshot once, as the plan does (the start check's once
#     part is the report's snapshot_gb).
#  Z) a GPU lane books what it holds AFTER the device's deposit when '--face-gradient' forms its face sums on the
#     CPU (plan P11.7 item 1: the handed-back host words, the face sums' words and the Real grids their conversion
#     builds, less the freed words the allocator hands back). Plan-only: the lane with the face sums minus the lane
#     without, per booked sub-grid cell, is the planner's own after-hold minus hold (DTFE_AUTO_TUNE_DEBUG=1), and
#     no less than 100 B/cell. One partition in flight on the GPU books its own sub-grids beside the hold without
#     the face sums, none with them; a real run of that plan (single build: about 10 s, a peak of 4.8 to 5.6 GB from
#     run to run) is predicted at 0.97 of its measured peak or above. GPU builds only.
#  V) a MIXED run's GPU lane (plan P13.1 item 13 a) books the host grids it keeps after the lock, beside its arrays,
#     where those outweigh its CPU deposit (128^3 particles on a 128^3 grid, 2^3): plan-only, the all-GPU lane less
#     the mixed one is the planner's pre-lock + once - max(cpu, pre-lock + after the lock) (DTFE_AUTO_TUNE_DEBUG=1),
#     and the old booking would be 0.06 GB or more off. Single build, GPU builds only.
#  Q) THE LANE SHARE (plan P11.8): with several partitions in flight the plan books the first lane whole and each
#     other one at a share of it (they do not peak together, and one of them is the largest), for 2 to 4 in flight
#     on the GPU alone, 2 to 6 with the free cores depositing; above that, without evaluation points and in the
#     double build every lane is whole. Plan-only against the planner's own 'PS lane share' line, the cover under
#     '--scratch-dir' included; then a run under a budget of 4 whole lanes: 6 in flight, none added by the dynamic
#     concurrency, the grids of '--max-concurrent 1' byte for byte; and one partition in flight on the GPU alone
#     books its own sub-grids beside the device's hold (Q8). GPU builds only.
#
# The 2^32 GPU sub-grid guard (ps_interpolation.cc) is NOT exercised here: it needs a real
# >4.29e9-cell partition sub-grid, i.e. >32 GB of buffers. It mirrors the validated guard in
# averaged_interpolation_1.cc line-for-line.
#
# Portable: runs on Linux too (tests/ci_suite.sh core) -- the peak RSS from getrusage, no macOS-only
# command; the GPU-only parts (H3, I2/I3, L, M, Q, V, W, Y7, Z) SKIP on a CPU build.
#
# Usage: tests/ps_auto_tune_check.sh [--no-build]

set -euo pipefail
unset DTFE_TRIANGULATION_THREADS DTFE_TBB_THREADS   # the insert's own switch moves the plans: the suite sets it itself

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
# the plans, reports and peaks asserted here were measured on the sampled deposit, which an '_a' run without a count
# no longer gets by default: pin it for every run (both binaries read the pin; W's and Z's own '--exact' beats it)
export DTFE_EXACT_AVERAGE=0

N="${N:-64}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
SNAP="${TMP}/at_input_pancake.hdf5"

echo "============================================================"
echo " auto-tune check   N=${N}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
    echo ">> building DTFE (section R) ..."
    make DTFE $(cat o/.build_mode 2>/dev/null || true) >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
    --amplitude-factor 1.8 >/dev/null

FAILS=0
check() {  # $1 = name, $2 = 0/1 ok
    if [ "$2" -eq 1 ]; then echo "   OK   $1"; else echo "   FAIL $1"; FAILS=$((FAILS+1)); fi
}

# ---------- (A) over-budget refusal: grid term alone exceeds a simulated 0.05 GB machine ----------
echo ">> A: simulated over-budget run (DTFE_RAM_GB=0.05, production field set)"
LOG_A="${TMP}/at_a.log"
DTFE_RAM_GB=0.05 "${BIN}" "${SNAP}" "${TMP}/at_a" --grid 64 \
    --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a \
    --input-format 105 --mpc-unit 1 --verbose 2 --periodic --subsamples 1 --ps-volume-weighted \
    > "${LOG_A}" 2>&1 || true
agrep() {  # set-e-safe grep assertion (ERE): $1 = check name, $2 = pattern, $3 = file
    if grep -Eq -- "$2" "$3"; then check "$1" 1; else check "$1" 0; fi
}
agrep "A refusal names the real cause"          "does not fit in memory, and --partition cannot fix it" "${LOG_A}"
agrep "A quantifies the irreducible grid term"  "full-resolution output grids need"                     "${LOG_A}"
agrep "A suggests --scratch-dir as the way out" "--scratch-dir <local dir>"                             "${LOG_A}"
agrep "A suggests the max feasible in-RAM grid" "largest that fits at these fields is about"            "${LOG_A}"
agrep "A warns off the broken escape hatches"   "--partition-index'/'--region' are NOT a way out"                "${LOG_A}"
# the misleading blame-the-split message must not appear
if grep -q "even a single .*-split partition is predicted to exceed the memory budget" "${LOG_A}"; then
    check "A no blame-the-split warning" 0
else
    check "A no blame-the-split warning" 1
fi
# Over budget with a LARGE particle set (DTFE_AUTO_MINN=1000 makes this one count as large): the
# fallback must still SPLIT -- one partition at a time, the smallest per-partition triangulation --
# rather than build one unpartitioned triangulation on top of an already over-budget run (for
# 94M particles that is ~100 GB, e.g. TNG100-3 with an 8192^2 --sample-points slice).
LOG_A2="${TMP}/at_a2.log"
DTFE_RAM_GB=0.05 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_a2" --grid 64 \
    --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a \
    --input-format 105 --mpc-unit 1 --verbose 2 --periodic --subsamples 1 --ps-volume-weighted \
    > "${LOG_A2}" 2>&1 || true
agrep "A over budget + large N: a real split at --max-concurrent 1" \
      "OVER BUDGET: --partition [2-6] [2-6] [2-6] --max-concurrent 1" "${LOG_A2}"

# ---------- (B) normal tuning at a comfortable simulated budget ----------
echo ">> B: normal tuning (DTFE_RAM_GB=64, DTFE_AUTO_MINN=1000)"
LOG_B="${TMP}/at_b.log"
DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_b" --grid 128 \
    --field density_a velocity_a --input-format 105 --mpc-unit 1 --verbose 2 --periodic \
    --subsamples 1 > "${LOG_B}" 2>&1
agrep "B picks a partition and concurrency" "AUTO-TUNE: .* -> --partition [0-9]+ [0-9]+ [0-9]+ --max-concurrent [0-9]+" "${LOG_B}"
agrep "B prints a predicted peak" "predicted peak ~[0-9.]+ GB" "${LOG_B}"

# ---------- (C) conservative contract: predicted >= measured on a REAL partitioned run ----------
echo ">> C: predicted >= measured (256^3, volume-weighted dispersion, real run)"
LOG_C="${TMP}/at_c.log"
# The peak RSS comes from getrusage(RUSAGE_CHILDREN) in a python parent, not '/usr/bin/time -l' (macOS
# only; GNU time spells it -v and is not even installed on a bare Ubuntu): ru_maxrss is in BYTES on macOS
# and in KILOBYTES on Linux.
peak_rss_run() {  # $1 = log; the rest = the command. Appends 'PEAK_RSS_BYTES <n>' to the log; exits as the command
    "${PY}" - "$@" <<'PYEOF'
import resource, subprocess, sys
with open(sys.argv[1], "w") as log:
    rc = subprocess.call(sys.argv[2:], stdout=log, stderr=subprocess.STDOUT)
    kb_or_b = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    log.write(f"\nPEAK_RSS_BYTES {kb_or_b if sys.platform == 'darwin' else kb_or_b * 1024}\n")
sys.exit(rc)
PYEOF
}
peak_rss_run "${LOG_C}" env DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_c" --grid 256 \
    --field density_a velocity_a dispersion_a --input-format 105 --mpc-unit 1 --verbose 2 --periodic \
    --subsamples 1 --ps-volume-weighted
if "${PY}" - "${LOG_C}" <<'PYEOF'
import re, sys
log = open(sys.argv[1]).read()
mp = re.search(r"predicted peak ~([0-9.]+) GB", log)
mm = re.search(r"PEAK_RSS_BYTES (\d+)", log)
assert mp and mm, "missing predicted-peak or peak-RSS output"
pred, meas = float(mp.group(1)), int(mm.group(1)) / 1e9
print(f"   .... predicted {pred:.2f} GB vs measured {meas:.2f} GB")
sys.exit(0 if pred >= meas else 1)
PYEOF
then check "C model errs HIGH (predicted >= measured peak RSS)" 1
else check "C model errs HIGH (predicted >= measured peak RSS)" 0; fi

# ---------- (D) explicit flags always win ----------
echo ">> D: explicit --partition/--max-concurrent are respected"
LOG_D="${TMP}/at_d.log"
DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_d" --grid 128 \
    --field density_a --input-format 105 --mpc-unit 1 --verbose 2 --periodic --subsamples 1 \
    --partition 2 2 2 --max-concurrent 2 > "${LOG_D}" 2>&1
if grep -q "partition grid of {2, 2, 2}" "${LOG_D}" || grep -Eq "2,2,2|2 2 2" "${LOG_D}"; then
    check "D user partition survives" 1
else
    check "D user partition survives" 0
fi
if grep -Eq "AUTO-TUNE: .* -> --partition 2 2 2 --max-concurrent 2 \(predicted peak ~[0-9.]+ GB.*Both are given" "${LOG_D}" \
   && grep -q "building up to 2 partition triangulation(s) concurrently" "${LOG_D}"; then
    check "D both flags given: the plan is predicted as given (--partition 2 2 2 --max-concurrent 2) and run so" 1
else
    check "D both flags given: the plan is predicted as given (--partition 2 2 2 --max-concurrent 2) and run so" 0
fi
# '--partition 1 1 1' asks for ONE triangulation. It leaves partitionOn false (nothing is split),
# which the auto-tuner must not read as "not given" (and then split the data anyway). Same
# settings as B, where the tuner does pick a split.
LOG_D1="${TMP}/at_d1.log"
DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_d1" --grid 128 \
    --field density_a --input-format 105 --mpc-unit 1 --verbose 2 --periodic --subsamples 1 \
    --partition 1 1 1 > "${LOG_D1}" 2>&1
if grep -Eq "AUTO-TUNE: .* -> --partition ([2-9]|[1-9][0-9])" "${LOG_D1}"; then
    check "D '--partition 1 1 1' keeps one triangulation (no auto split)" 0
else
    check "D '--partition 1 1 1' keeps one triangulation (no auto split)" 1
fi

# ---------- (E) the memory budget ----------
echo ">> E: memory budget rules"
LOG_E="${TMP}/at_e.log"
run_e() {
    env "$@" DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_e" --grid 128 --field density_a velocity_a \
        --input-format 105 --mpc-unit 1 --verbose 2 --periodic --subsamples 1 > "${LOG_E}" 2>&1
}
run_e DTFE_RAM_GB=64
agrep "E default budget = 60% of RAM (simulated 64 GB: no live term)" \
      "memory budget: 38\.4 GB = 60% of 64 GB RAM$" "${LOG_E}"
run_e DTFE_RAM_GB=64 DTFE_MEM_FRACTION=0.5
agrep "E DTFE_MEM_FRACTION sets the share" "memory budget: 32 GB = 50% of 64 GB RAM" "${LOG_E}"
run_e DTFE_RAM_GB=64 DTFE_MEM_BUDGET_GB=20
agrep "E DTFE_MEM_BUDGET_GB sets the budget outright" "memory budget: 20 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
run_e PATH="${PATH}"
agrep "E a live run states its budget (60% of RAM, or less when other programs use memory)" \
      "memory budget: [0-9.]+ GB( = 60% of|: other programs leave)" "${LOG_E}"

# ---------- (F) --auto-tune-report ----------
echo ">> F: --auto-tune-report"
LOG_F="${TMP}/at_f.log"
run_f() {
    env "$@" DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_f" --grid 64 \
        --field density_a velocity_a --input-format 105 --mpc-unit 1 --verbose 2 --periodic --subsamples 1 \
        --auto-tune-report ${F_ARGS[@]+"${F_ARGS[@]}"} > "${LOG_F}" 2>&1
}
F_ARGS=()
rm -f "${TMP}"/at_f.*
run_f PATH="${PATH}" && rc=0 || rc=$?       # (a bare 'run_f; rc=$?' aborts under set -e before the check)
check "F exits 0" "$([ "${rc}" = "0" ] && echo 1 || echo 0)"
check "F prints exactly one report line" "$([ "$(grep -c '^AUTO-TUNE-REPORT ' "${LOG_F}")" = "1" ] && echo 1 || echo 0)"
agrep "F the line carries the decision, prediction and budget" \
      "^AUTO-TUNE-REPORT partition=[0-9]+ mc=[0-9]+ predicted_gb=[0-9.]+ budget_gb=38\.40 .*over_budget=[01]$" "${LOG_F}"
check "F nothing is written next to the output root" \
      "$([ -z "$(ls "${TMP}"/at_f.* 2>/dev/null | grep -v 'at_f.log$')" ] && echo 1 || echo 0)"
F_ARGS=(--partition 2 2 2 --max-concurrent 3)
run_f PATH="${PATH}"
agrep "F a hand-set split and concurrency are reported as given" "^AUTO-TUNE-REPORT partition=2 mc=3 " "${LOG_F}"
F_ARGS=()
run_f PATH="${PATH}"
p0=$(sed -n 's/.* pts_gb=\([0-9.]*\) .*/\1/p' "${LOG_F}")
run_f DTFE_AUTO_PTS_N=4000000 DTFE_AUTO_PTS_VEL_GRAD=1
p1=$(sed -n 's/.* pts_gb=\([0-9.]*\) .*/\1/p' "${LOG_F}")
check "F DTFE_AUTO_PTS_N sizes a plane that is not on disk (pts ${p0} -> ${p1} GB)" \
      "$(awk -v a="${p0:-x}" -v b="${p1:-0}" 'BEGIN{print (a == 0 && b > 0.5) ? 1 : 0}')"

# ---------- (G) the stream estimate of a NON-periodic cloud with a smaller output region ----------
# The estimate tiles the cloud's own Lagrangian bounding box. Tiling the OUTPUT region (--box) and
# wrapping initial positions into it folds far-apart particles of a cloud larger than the region into
# one patch, whose Delaunay tetrahedra then span the cloud in Eulerian space -- 8.1 million "streams"
# per point on a TNG100-3 Lagrangian sub-cube, so the tuner would run every point evaluation serially
# on the smallest split (222 s instead of seconds).
echo ">> G: non-periodic stream estimate with a region smaller than the cloud"
SNAP_G="${TMP}/at_input_clump.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_G}" --n 48 --box "${BOX}" \
    --amplitude-factor 1.0 --margin-frac 0.2 >/dev/null
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=1000000 "${BIN}" "${SNAP_G}" "${TMP}/at_g" --grid 16 --field density \
    --input-format 105 --mpc-unit 1 --verbose 2 --box 40 60 40 60 40 60 --auto-tune-report > "${TMP}/at_g.log" 2>&1
sg=$(sed -n 's/.* streams_est=\([0-9.]*\) .*/\1/p' "${TMP}/at_g.log")
# (below 1 is fine: for a non-periodic cloud the Eulerian/Lagrangian volume ratio is its mean
# compression -- this clump's is ~0.5 -- and the budget clamps the estimate at 1 stream per point)
check "G the estimate stays physical: 0 < ${sg:-none} < 20 streams/point" \
      "$(awk -v s="${sg:-0}" 'BEGIN{print (s > 0 && s < 20) ? 1 : 0}')"

# ---------- (H) the time-aware split of a small set ----------
echo ">> H: time-aware split for small sets (speed, not memory)"
GPU_H=0; grep -q "METAL=1" o_ps/.build_mode 2>/dev/null && GPU_H=1
COMMON_H=(--input-format 105 --mpc-unit 1 --verbose 2 --auto-tune-report)
# H1: a slice of the 64^3 box (4.2M points, 16^3 grid) is triangulation-bound -> 2^3 partitions. The split pays
# only with enough threads (on 4 it rightly stays one domain), so the thread
# count is pinned to a 10-core machine's: the report path only plans, it runs nothing
env OMP_NUM_THREADS=10 DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=4194304 "${BIN}" "${SNAP}" "${TMP}/at_h1" --grid 16 --field density --periodic \
    "${COMMON_H[@]}" > "${TMP}/at_h1.log" 2>&1
agrep "H1 a triangulation-bound small run is split for speed" "^AUTO-TUNE-REPORT partition=[23] mc=[2-9]|^AUTO-TUNE-REPORT partition=[23] mc=[1-9][0-9]" "${TMP}/at_h1.log"
# H2: a 256^3 sampled grid is deposit-bound -> one domain, and its predicted peak is sane
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h2" --grid 256 --field density_a velocity_a --periodic \
    "${COMMON_H[@]}" > "${TMP}/at_h2.log" 2>&1
agrep "H2 a deposit-bound small run keeps one domain" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h2.log"
ph=$(sed -n 's/.* predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_h2.log")
check "H2 the single-domain report predicts a sane peak: ${ph:-none} GB < 12" \
      "$(awk -v p="${ph:-99}" 'BEGIN{print (p > 0 && p < 12) ? 1 : 0}')"
# H3 (GPU builds): the GPU deposit serialises on one queue -> a split cannot speed it up
if [ "${GPU_H}" = 1 ]; then
    env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h3" --grid 256 --field density_a velocity_a --periodic --gpu \
        "${COMMON_H[@]}" > "${TMP}/at_h3.log" 2>&1
    agrep "H3 a GPU grid keeps one domain" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h3.log"
else
    echo "   SKIP  H3 (this build has no GPU backend)"
fi
# H4: the non-periodic clump (48^3) makes no periodic copies, so its single domain is cheap
# enough to keep (charged the periodic 1.7x it would be split)
env DTFE_RAM_GB=64 DTFE_AUTO_PTS_N=4194304 "${BIN}" "${SNAP_G}" "${TMP}/at_h4" --grid 16 --field density \
    "${COMMON_H[@]}" > "${TMP}/at_h4.log" 2>&1
agrep "H4 a non-periodic small cloud is costed without periodic copies (one domain)" "^AUTO-TUNE-REPORT partition=1 " "${TMP}/at_h4.log"
# H5: two auto-split runs (a 16^3 grid of the 64^3 box: triangulation-bound -> 2^3) give bit-identical grids
for r in a b; do
    "${BIN}" "${SNAP}" "${TMP}/at_h5$r" --grid 16 --field density --input-format 105 --mpc-unit 1 --periodic --verbose 0 > "${TMP}/at_h5$r.log" 2>&1
done
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_h5r" --grid 16 --field density --periodic "${COMMON_H[@]}" > "${TMP}/at_h5r.log" 2>&1
agrep "H5 the 16^3 grid run is a speed split" "^AUTO-TUNE-REPORT partition=[23] " "${TMP}/at_h5r.log"
if cmp -s "${TMP}/at_h5a.den" "${TMP}/at_h5b.den" && cmp -s "${TMP}/at_h5a.streams" "${TMP}/at_h5b.streams"; then
    check "H5 two auto-split runs are bit-identical (.den, .streams: ordered merge)" 1
else
    check "H5 two auto-split runs are bit-identical (.den, .streams: ordered merge)" 0
fi
rm -f "${TMP}"/at_h*.den "${TMP}"/at_h*.streams "${TMP}"/at_h*.hidden_streams

# ---------- (I) very fine grids: the GPU's 2^32-cell sub-grid cap, and a small set's memory ----------
# The GPU kernels number a partition's sub-grid cells with 32 bits: above 2^32 the deposit falls back to
# the CPU. The tuner splits to keep every GPU sub-grid under the cap (a 1700^3 grid: 2^3 at least; since the
# sub-grids are measured on the particles (P8.2), the pancake's 2^3 sub-grid is ~1.1e9 cells -- 760 x 428 x 428 at
# 850^3, measured -- not the 7.05e8 of the old G/n + 40 rule, and the memory budget takes the split further, to 5^3)
# and warns about a user split that does not. And a small particle set on such a grid does not stay in one
# domain when that domain's deposit buffers do not fit (83 GB unsplit): it splits for memory.
echo ">> I: very fine grids (1700^3, grids on scratch)"
ISCR="${TMPDIR:-/tmp}/dtfe-at-scratch"; mkdir -p "${ISCR}"
COMMON_I=(--grid 1700 --periodic --field density_a --input-format 105 --mpc-unit 1 --verbose 2 --scratch-dir "${ISCR}" --auto-tune-report)
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_i_cpu" "${COMMON_I[@]}" > "${TMP}/at_i_cpu.log" 2>&1
pi=$(sed -n 's/^AUTO-TUNE-REPORT partition=\([0-9]*\) .*/\1/p' "${TMP}/at_i_cpu.log" | head -1)
ob=$(sed -n 's/^AUTO-TUNE-REPORT .*over_budget=\([01]\).*/\1/p' "${TMP}/at_i_cpu.log" | head -1)
# The start-up peak (P8.2, phase R 2026-11-09): under --scratch-dir each full grid is zero-filled in anonymous memory
# before its file is mapped over it, so the prediction is at least particles + kept read blocks + the largest grid. The
# single build's 1700^3 density grid is 19.7 GB and stays under the 38.4 GB budget; the double build's is 39.3 GB, which
# no split can lower: there the planner must say so (the start-up line) and report over_budget=1 -- as the production
# 1536^3 run of 2026-11-05 should have been told (its 203 GB of zero fills on a 68.7 GB machine, predicted 39.8 GB).
su=$(grep -c 'the predicted peak is the start-up one' "${TMP}/at_i_cpu.log" || true)
check "I1 CPU: a small set on a 1700^3 grid splits for memory (partition ${pi:-none}, over_budget ${ob:-none}; over budget only with the start-up peak named, ${su} line(s))" \
      "$([ -n "${pi}" ] && [ "${pi}" -ge 2 ] && { [ "${ob}" = "0" ] || [ "${su:-0}" -ge 1 ]; } && echo 1 || echo 0)"
env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_i_gpu" "${COMMON_I[@]}" --gpu > "${TMP}/at_i_gpu.log" 2>&1
if grep -q "GPU (--gpu)" "${TMP}/at_i_gpu.log"; then
    pg=$(sed -n 's/^AUTO-TUNE-REPORT partition=\([0-9]*\) .*/\1/p' "${TMP}/at_i_gpu.log" | head -1)
    # the split keeps the GPU deposit (the 'GPU deposit' plan, or the cap's own line when the cap decides it) and its
    # booked sub-grid under the 2^32 cells (DTFE_AUTO_TUNE_DEBUG's 'booked sub-grid')
    env DTFE_RAM_GB=64 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${SNAP}" "${TMP}/at_i_gpu" "${COMMON_I[@]}" --gpu > "${TMP}/at_i_gpu_dbg.log" 2>&1
    sg=$(sed -n 's/^AUTO-TUNE-DEBUG PS booked sub-grid: \([0-9]*\) cells.*/\1/p' "${TMP}/at_i_gpu_dbg.log" | tail -1)
    check "I2 GPU: the tuner splits a 1700^3 grid so each sub-grid stays under 2^32 cells (partition ${pg:-none}, ${sg:-?} cells)" \
          "$([ -n "${pg}" ] && [ "${pg}" -ge 2 ] && [ -n "${sg}" ] && [ "${sg}" -lt 4294967296 ] \
              && grep -Eq "GPU deposit|under the 2\^32 cells the GPU kernels number" "${TMP}/at_i_gpu.log" && echo 1 || echo 0)"
    env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_i_user" "${COMMON_I[@]}" --gpu --partition 1 1 1 > "${TMP}/at_i_user.log" 2>&1
    agrep "I3 GPU: a user's unsplit 1700^3 grid is told which split keeps the GPU" "above the 2\^32 the GPU kernels number" "${TMP}/at_i_user.log"
else
    echo "   SKIP  I2/I3 (this build has no GPU backend)"
fi
rm -rf "${ISCR}"

# ---------- (J) the machine's free memory over a plan ----------
# An explicit budget must not skip the free-memory rule: DTFE_MEM_BUDGET_GB=52 on a 64 GB Mac gets a TNG50-3-Dark
# z=0 run SIGKILLed by macOS once other programs take memory. DTFE_FREE_GB simulates what the
# other programs leave free (the OS query's test hook).
echo ">> J: an explicit budget capped by the free memory; the partition memory gate"
run_e DTFE_FREE_GB=10 DTFE_MEM_BUDGET_GB=40
agrep "J1 DTFE_MEM_BUDGET_GB above the free memory is capped (10 GB free - 4 GB headroom = 6 GB), saying so" \
      "memory budget: 6 GB: DTFE_MEM_BUDGET_GB asks 40 GB, but the other programs leave 10 GB free, minus 4 GB headroom" "${LOG_E}"
run_e DTFE_FREE_GB=10 DTFE_MEM_BUDGET_GB=5
agrep "J2 a budget the free memory holds stays as set" "memory budget: 5 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
run_e DTFE_FREE_GB=44.2 DTFE_MEM_BUDGET_GB=40.5
agrep "J2b ... and so does one the free memory would trim by under 0.5 GB (44.2 - 4 vs 40.5): no 'but' line" \
      "memory budget: 40.5 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
agrep "J2c ... naming the free memory it was set against (P8.7)" \
      "memory budget: 40.5 GB set by DTFE_MEM_BUDGET_GB \(the other programs leave 44.2 GB free: 40.2 GB after the 4 GB headroom\)$" "${LOG_E}"
run_e DTFE_FREE_GB=60
agrep "J2d the RAM-share budget names the free memory too when it does not cut it" \
      "memory budget: [0-9.]+ GB = 60% of [0-9.]+ GB RAM \(the other programs leave 60 GB free: 56 GB after the 4 GB headroom\)$" "${LOG_E}"
run_e DTFE_RAM_GB=64 DTFE_FREE_GB=10 DTFE_MEM_BUDGET_GB=40
agrep "J3 a simulated machine (DTFE_RAM_GB) skips the live cap: tests stay deterministic" \
      "memory budget: 40 GB set by DTFE_MEM_BUDGET_GB" "${LOG_E}"
# the gate: a hand-set split with the concurrency left to the tuner (its report sizes one partition), the
# same run with ~no free memory: every partition beyond the first waits, the run completes, the grids agree
run_j() {
    env "$@" DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_j" --grid 64 --field density_a \
        --input-format 105 --mpc-unit 1 --verbose 2 --periodic --subsamples 1 --partition 2 2 2 > "${TMP}/at_j.log" 2>&1
}
run_j DTFE_FREE_GB=1000                      # ample by decree: the live reading would make J5b depend on the machine
for f in a_den a_streams; do cp "${TMP}/at_j.${f}" "${TMP}/at_j_free.${f}" 2>/dev/null || true; done
cp "${TMP}/at_j.log" "${TMP}/at_j_free.log"
rm -f "${TMP}"/at_j.a_*                      # a gated run that writes nothing must not compare the free one
run_j DTFE_FREE_GB=0.001 && rc=0 || rc=$?    # (a bare 'run_j; rc=$?' aborts under set -e: J4 could never fail)
mcj=$(sed -n 's/.*building up to \([0-9]*\) partition triangulation.*/\1/p' "${TMP}/at_j.log" | head -1)
if [ "${mcj:-1}" -ge 2 ]; then
    check "J4 low free memory: the run still completes (exit ${rc})" "$([ "${rc}" = "0" ] && echo 1 || echo 0)"
    agrep "J5 ... partitions wait for one in flight to finish, and it says so" "PS-DTFE memory gate: partition [0-9]+ waits" "${TMP}/at_j.log"
    if grep -q "PS-DTFE memory gate" "${TMP}/at_j_free.log"; then
        check "J5b with the memory free the gate holds nothing" 0
    else
        check "J5b with the memory free the gate holds nothing" 1
    fi
    if "${PY}" - "${TMP}" <<'PYEOF'
import os, sys, numpy as np
t = sys.argv[1]
real = np.dtype(os.environ.get("DTFE_TEST_REAL", "float32"))   # the grids are float64 in double mode
worst = 0.0
for f in ("a_den", "a_streams"):
    a = np.fromfile(f"{t}/at_j_free.{f}", real)
    b = np.fromfile(f"{t}/at_j.{f}", real)
    assert a.size == b.size and a.size > 0, f
    worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30))))
print(f"   .... gated vs free: max rel {worst:.1e} (the merge's float sums, in index order)")
sys.exit(0 if worst < 1e-5 else 1)
PYEOF
    then check "J6 ... and its grids equal the ungated run's to float rounding" 1
    else check "J6 ... and its grids equal the ungated run's to float rounding" 0; fi
else
    echo "   SKIP  J4-J6 (the tuner ran this split one partition at a time: nothing to gate)"
fi

# ---------- (K) partitions in flight never outnumber the threads ----------
# OMP_NUM_THREADS=2 makes the thread count the machine's own; '--partition 4 4 4' has 64 partitions. A small set
# with the split hand-set and the concurrency left to the tuner (K1), and '--max-concurrent 0' = all threads (K2),
# would each start all 64 at once.
echo ">> K: partitions in flight are capped at the threads"
run_k() {
    env "$@" OMP_NUM_THREADS=2 DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/at_k" --grid 16 --field density \
        --input-format 105 --mpc-unit 1 --verbose 2 --periodic --partition 4 4 4 ${K_ARGS[@]+"${K_ARGS[@]}"} > "${TMP}/at_k.log" 2>&1
}
for k in K1 K2; do
    if [ "${k}" = K1 ]; then K_ARGS=(); what="a hand-set split, the concurrency left to the tuner"
    else K_ARGS=(--max-concurrent 0); what="'--max-concurrent 0'"; fi
    run_k PATH="${PATH}" && rc=0 || rc=$?
    mck=$(sed -n 's/.*building up to \([0-9]*\) partition triangulation.*/\1/p' "${TMP}/at_k.log" | head -1)
    check "${k} ${what}: at most 2 of the 64 partitions at once on 2 threads (${mck:-none}, exit ${rc})" \
          "$([ "${rc}" = "0" ] && [ -n "${mck}" ] && [ "${mck}" -le 2 ] && echo 1 || echo 0)"
done
K_ARGS=()
rm -f "${TMP}"/at_k.*

# ---------- (L) the look-ahead and the budget agree ----------
# At --max-concurrent 1 with the GPU deposit the run prepares the next partition during the deposit. Its bytes
# belong in the prediction the split is chosen by: added after it, a plan that fits reports over_budget=1 and
# the launcher's memory check halves a plane that fits. A budget between 'one partition' and 'one partition + the
# look-ahead' must switch the look-ahead off and report a prediction within the budget.
echo ">> L: the look-ahead is counted against the budget"
run_l() {  # $1 = budget in GB, $2 = log
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_AUTO_MINN=1000 DTFE_MEM_BUDGET_GB="$1" "${BIN}" "${SNAP}" "${TMP}/at_l" \
        --grid 64 --field density_a --input-format 105 --mpc-unit 1 --verbose 2 --periodic --gpu \
        --partition 2 2 2 --max-concurrent 1 --auto-tune-report > "$2" 2>&1 || true
}
run_l 50 "${TMP}/at_l1.log"
if grep -q "GPU (--gpu)" "${TMP}/at_l1.log"; then
    p1=$(sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_l1.log" | head -1)
    # just under 'one partition + the look-ahead' (~0.1 GB at 2^3 of 64^3 particles), above one partition
    bl=$(awk -v p="${p1:-0}" 'BEGIN{printf "%.4f", p - 0.03}')
    run_l "${bl}" "${TMP}/at_l2.log"
    pl=$(sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_l2.log" | head -1)
    ol=$(sed -n 's/^AUTO-TUNE-REPORT .*over_budget=\([01]\).*/\1/p' "${TMP}/at_l2.log" | head -1)
    check "L a budget ${bl} GB under one partition + look-ahead (${p1:-none} GB): look-ahead off, predicted ${pl:-none} GB, over_budget ${ol:-none}" \
          "$([ "${ol}" = "0" ] && grep -q "look-ahead off for this run" "${TMP}/at_l2.log" \
             && ! grep -q "look-ahead off" "${TMP}/at_l1.log" \
             && awk -v p="${pl:-99}" -v b="${bl}" 'BEGIN{exit !(p <= b)}' && echo 1 || echo 0)"
else
    echo "   SKIP  L (this build has no GPU backend)"
fi
rm -f "${TMP}"/at_l*

# ---------- (M) the GPU deposit's host snapshot is in the prediction, once ----------
# The Metal deposit keeps a host copy of its accumulator buffers (ps_metal_host.cc GpuSnapshot: the grids, the
# 8 B fixed-point mass, the per-tet masses) to resume a failed command buffer: ~4.7 GB
# per partition at 1024^3 split 2^3. The deposits run one at a time, so the copy is counted ONCE whatever the
# concurrency: predicted(c) = fixed + c x per-partition + snapshot, i.e. snapshot = p(2) - fixed - 2 (p(3) - p(2)).
echo ">> M: the GPU deposit's snapshot is counted once"
run_m() {  # $1 = --max-concurrent, $2 = log
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/at_m" --grid 256 \
        --field density_a velocity_a --ps-volume-weighted --input-format 105 --mpc-unit 1 --verbose 2 --periodic --gpu \
        --partition 2 2 2 --max-concurrent "$1" --auto-tune-report > "$2" 2>&1 || true
}
run_m 2 "${TMP}/at_m2.log"
if grep -q "GPU (--gpu)" "${TMP}/at_m2.log"; then
    run_m 3 "${TMP}/at_m3.log"
    gb() { sed -n "s/^AUTO-TUNE-REPORT .*$1=\([0-9.]*\) .*/\1/p" "$2" | head -1; }
    p2=$(gb predicted_gb "${TMP}/at_m2.log"); p3=$(gb predicted_gb "${TMP}/at_m3.log"); f2=$(gb fixed_gb "${TMP}/at_m2.log")
    # expected: (24 B of selected grids + 4 B of the 64-bit mass) x the (256/2 + 40)^3 sub-grid cells = 0.133 GB
    # (+ the per-tet masses; measured 0.114 GB) -- with the velocity moment and its weight (16 B) as
    # 64-bit words, 16 B more: 0.209 GB (measured 0.190 GB). DTFE_GPU_FLOAT_MOMENTS=1 keeps the floats.
    sm=$(awk -v a="${p2:-0}" -v b="${p3:-0}" -v f="${f2:-0}" 'BEGIN{printf "%.3f", a - f - 2*(b - a)}')
    if [ "${DTFE_GPU_FLOAT_MOMENTS:-${PS_GPU_FLOAT_MOMENTS:-0}}" != "0" ]; then   # (the old name: deprecated, still read)
        lo=0.10; hi=0.17; ex=0.13
    else lo=0.16; hi=0.25; ex=0.21; fi
    check "M the GPU snapshot is counted once: ${sm} GB beyond 2 partitions (expected ~${ex} GB; predicted ${p2:-none} / ${p3:-none} GB at 2 / 3 in flight)" \
          "$(awk -v s="${sm}" -v lo="${lo}" -v hi="${hi}" 'BEGIN{exit !(s >= lo && s <= hi)}' && echo 1 || echo 0)"
else
    echo "   SKIP  M (this build has no GPU backend)"
fi
rm -f "${TMP}"/at_m*

# ---------- (N) the stream estimate on the sample points' slab ----------
echo ">> N: the sample-point stream estimate on the plane's own slab"
SNAP_N="${TMP}/at_input_pancake80.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_N}" --n 80 --box "${BOX}" \
    --amplitude-factor 1.8 --jitter-frac 0.02 --seed 7 >/dev/null
"${PY}" - "${TMP}" "${BOX}" <<'PYEOF'
import numpy as np, sys
tmp, box = sys.argv[1], float(sys.argv[2])
u = (np.arange(64) + 0.5) * box / 64
Y, Z = np.meshgrid(u, u, indexing="ij")
for name, x in (("far", 0.1 * box), ("fold", 0.5 * box)):    # outside / through the 3-stream band
    np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1).astype(np.float64).tofile(f"{tmp}/at_n_{name}.bin")
PYEOF
est_n() {  # $1 = plane, $2 = DTFE_AUTO_PTS_BOXMEAN
    env DTFE_RAM_GB=64 DTFE_AUTO_PTS_BOXMEAN="$2" "${BIN}" "${SNAP_N}" "${TMP}/at_n" --grid 32 --periodic --input-format 105 \
        --mpc-unit 1 --field density_a --sample-points "${TMP}/at_n_$1.bin" --auto-tune-report 2>/dev/null \
        | sed -n 's/^AUTO-TUNE-REPORT .* streams_est=\([0-9.]*\) .*/\1/p'
}
nf=$(est_n far 0); nd=$(est_n fold 0); nb=$(est_n far 1)
check "N a plane outside the fold estimates ${nf:-none} streams/point (true 1, within 10%)" \
      "$(awk -v s="${nf:-0}" 'BEGIN{print (s > 0.9 && s < 1.1) ? 1 : 0}')"
check "N a plane through the fold estimates ${nd:-none} streams/point (true 3, within 10%)" \
      "$(awk -v s="${nd:-0}" 'BEGIN{print (s > 2.7 && s < 3.3) ? 1 : 0}')"
check "N the box-wide estimate (DTFE_AUTO_PTS_BOXMEAN=1) is the box mean, ${nb:-none} (true 1.33)" \
      "$(awk -v s="${nb:-0}" 'BEGIN{print (s > 1.2 && s < 1.45) ? 1 : 0}')"
rm -f "${TMP}"/at_n*

# ---------- (O) the parallel insert's transient ----------
echo ">> O: --parallel-triangulation is in the predicted peak"
if grep -aq "parallel, TBB" "${BIN}"; then   # the insert's own message: only in a build with TBB
    pred_o() {
        env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_o" --grid 32 --periodic --input-format 105 --mpc-unit 1 --field density \
            --partition 2 2 2 --max-concurrent 2 --auto-tune-report "$@" 2>/dev/null \
            | sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p'
    }
    o0=$(pred_o); o1=$(pred_o --parallel-triangulation)
    check "O the parallel insert raises the predicted peak (${o0:-none} -> ${o1:-none} GB)" \
          "$(awk -v a="${o0:-0}" -v b="${o1:-0}" 'BEGIN{print (a > 0 && b > a) ? 1 : 0}')"
    if awk -v a="${o0:-0}" -v b="${o1:-0}" 'BEGIN{exit !(a > 0 && b > a)}'; then
        # O2: DTFE_TRIANGULATION_THREADS is the insert's own switch (triangulation.cpp): a count forces the parallel
        # insert at any size, and 1 keeps it sequential beside '--parallel-triangulation'; the plan books what runs
        o2=$(DTFE_TRIANGULATION_THREADS=2 pred_o); o3=$(DTFE_TRIANGULATION_THREADS=1 pred_o --parallel-triangulation)
        check "O2 DTFE_TRIANGULATION_THREADS=2 books the forced insert's copies (${o2:-none} = ${o1} GB), =1 beside --parallel-triangulation none (${o3:-none} = ${o0} GB)" \
              "$([ "${o2}" = "${o1}" ] && [ "${o3}" = "${o0}" ] && echo 1 || echo 0)"
        # O3: the split search's own count books the copies too. The concurrency left to the tuner (grid 256, planned
        # for memory: DTFE_AUTO_MINN=1000): a budget that holds 2 partitions in flight without the forced insert's
        # copies and 1 with them plans 1 in flight within it (the count was 2, reported over_budget=1)
        rep_o() {  # prints 'mc over_budget predicted_gb'
            env DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_o" --grid 256 --periodic --input-format 105 \
                --mpc-unit 1 --field density --partition 2 2 2 --auto-tune-report "$@" 2>/dev/null \
                | sed -n 's/^AUTO-TUNE-REPORT .* mc=\([0-9]*\) predicted_gb=\([0-9.]*\) .*over_budget=\([01]\)$/\1 \3 \2/p'
        }
        read -r _ _ s1 <<< "$(rep_o --max-concurrent 1)"
        read -r _ _ s2 <<< "$(rep_o --max-concurrent 2)"
        read -r _ _ f2 <<< "$(rep_o --max-concurrent 2 --parallel-triangulation)"
        # A = a lane, T = its copies; the budget is 2 lanes + x beside the grids and particles, x = min(A, 2 T)/2
        ob=$(awk -v a="${s1:-0}" -v b="${s2:-0}" -v f="${f2:-0}" 'BEGIN{A = b - a; T = (f - b)/2; x = (A < 2*T ? A : 2*T)/2;
            if (a > 0 && x > 0.03 && A + x - T > 0.03) printf "%.2f", b + x}')
        if [ -n "${ob}" ]; then
            read -r om oo _ <<< "$(DTFE_MEM_BUDGET_GB="${ob}" rep_o --parallel-triangulation)"
            check "O3 a budget of ${ob} GB (2 in flight fit without the insert's copies, 1 with them): ${om:-none} in flight, over_budget=${oo:-none}" \
                  "$([ "${om:-}" = 1 ] && [ "${oo:-}" = 0 ] && echo 1 || echo 0)"
        else
            check "O3 the budget between the two bookings (not formed from ${s1:-none}, ${s2:-none}, ${f2:-none} GB)" 0
        fi
        # O4: ... and at one partition in flight: the split left to the tuner on one thread (OMP_NUM_THREADS=1, so one
        # in flight at every split, and the search stops at the first split that has it). A budget that holds one 2^3
        # partition without the forced insert's copies and not with them, and one 3^3 partition with them: the plan is
        # a finer split, within the budget (it was 2^3 with over_budget=1)
        rep_o4() {  # prints 'partition mc over_budget predicted_gb'
            env OMP_NUM_THREADS=1 DTFE_RAM_GB=64 DTFE_AUTO_MINN=1000 "${BIN}" "${SNAP}" "${TMP}/at_o" --grid 256 --periodic \
                --input-format 105 --mpc-unit 1 --field density --auto-tune-report "$@" 2>/dev/null \
                | sed -n 's/^AUTO-TUNE-REPORT partition=\([0-9]*\) mc=\([0-9]*\) predicted_gb=\([0-9.]*\) .*over_budget=\([01]\)$/\1 \2 \4 \3/p'
        }
        read -r _ _ _ u2 <<< "$(rep_o4 --partition 2 2 2 --max-concurrent 1)"
        read -r _ _ _ c2 <<< "$(rep_o4 --partition 2 2 2 --max-concurrent 1 --parallel-triangulation)"
        read -r _ _ _ c3 <<< "$(rep_o4 --partition 3 3 3 --max-concurrent 1 --parallel-triangulation)"
        # the budget half way between one 2^3 partition without and with the copies, above one 3^3 one with them
        b4=$(awk -v a="${u2:-0}" -v b="${c2:-0}" -v c="${c3:-0}" 'BEGIN{x = (b - a)/2;
            if (a > 0 && c > 0 && x > 0.02 && a + x - c > 0.02) printf "%.3f", a + x}')
        if [ -n "${b4}" ]; then
            read -r op om oo _ <<< "$(DTFE_MEM_BUDGET_GB="${b4}" rep_o4 --parallel-triangulation)"
            check "O4 a budget of ${b4} GB (one 2^3 partition ${u2} GB, with the insert's copies ${c2} GB; one 3^3 with them ${c3} GB): --partition ${op:-none}, ${om:-none} in flight, over_budget=${oo:-none}" \
                  "$([ "${op:-0}" -ge 3 ] 2>/dev/null && [ "${om:-}" = 1 ] && [ "${oo:-}" = 0 ] && echo 1 || echo 0)"
        else
            check "O4 the budget between the two bookings (not formed from ${u2:-none}, ${c2:-none}, ${c3:-none} GB)" 0
        fi
    else
        echo "   SKIP  O2, O3, O4 (the forced insert books nothing here: fewer than 4 threads)"
    fi
else
    echo "   SKIP  O (no TBB in this build)"
fi

# ---------- (P) the default Lagrangian padding is what the tuner models ----------
# A hand-set 4^3 split of the 64^3 set (report path, runs nothing): k = 8 spacings pad each side by 0.5 of a
# partition, the Eulerian rule by max(0.3, padding fraction x 4) ~ 0.31, so the two predictions differ.
echo ">> P: the tuner models the default '--ps-lagrangian-padding 8'"
pred_p() {
    env DTFE_RAM_GB=64 "${BIN}" "${SNAP}" "${TMP}/at_p" --grid 32 --periodic --input-format 105 --mpc-unit 1 --field density \
        --partition 4 4 4 --max-concurrent 4 --auto-tune-report "$@" 2>/dev/null \
        | sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p'
}
pd=$(pred_p); p8=$(pred_p --ps-lagrangian-padding 8); p0=$(pred_p --ps-lagrangian-padding 0)
check "P the default predicts like '--ps-lagrangian-padding 8' (${pd:-none} = ${p8:-none} GB), not like 0 (${p0:-none} GB)" \
      "$(awk -v d="${pd:-x}" -v e="${p8:-y}" -v z="${p0:-x}" 'BEGIN{print (d != "x" && d == e && d != z) ? 1 : 0}')"

# ---------- (R) standard DTFE --region: the tuner plans the region, not the snapshot ----------
# A '--region'/'--region-mpc' run triangulates only the region plus its padding, so the standard binary's tuner
# plans the region's particle count and box; planning the whole snapshot's turns the launcher's exact zoom of a
# 20^3-cell TNG50 region into '--partition 3 3 3, 10 thread(s)' -- 27 x 10 sub-triangulations, each padded again
# around its tiny sub-box. On a simulated 0.2 GB machine (the standard tuner has no small-N gate): the whole 64^3 box
# needs a split (the control, so the machine is tight enough to show the difference); a 10 x 10 x 2 Mpc region of
# 264 particles keeps one domain with every thread (the tuner stays silent); a region that IS the box plans exactly
# like the box; and a real region run left to the tuner is byte-identical to one with '--partition 1 1 1' (a tuner
# planning the snapshot cuts that one to ONE thread, with a bogus 'no split fits' warning, and splits the other).
echo ">> R: standard DTFE --region plans the region's particles (${DTFE_BIN})"
if [ -x "${DTFE_BIN}" ]; then
    std_r() {  # $1 = log, rest = extra args (report mode: nothing is written)
        local log="$1"; shift
        env DTFE_RAM_GB=0.2 "${DTFE_BIN}" "${SNAP}" "${TMP}/at_r" --grid 16 16 1 --periodic --input-format 105 --mpc-unit 1 \
            --field density_a --auto-tune-report "$@" > "${log}" 2>&1 || true
    }
    std_r "${TMP}/at_r_box.log"
    std_r "${TMP}/at_r_reg.log" --region-mpc 40 50 40 50 49 51
    std_r "${TMP}/at_r_all.log" --region 0 1 0 1 0 1
    agrep "R control: the whole box needs a split on this simulated machine" \
          "AUTO-TUNE: .* -> --partition [2-6] [2-6] [2-6]" "${TMP}/at_r_box.log"
    r_reg="$(grep -o 'AUTO-TUNE: .*' "${TMP}/at_r_reg.log" || true)"
    check "R a 10x10x2 Mpc region keeps one domain and every thread (no AUTO-TUNE line; got: '${r_reg}')" \
          "$([ -z "${r_reg}" ] && grep -q 'AUTO-TUNE-REPORT' "${TMP}/at_r_reg.log" && echo 1 || echo 0)"
    r_box="$(grep -o 'AUTO-TUNE: .*' "${TMP}/at_r_box.log" || true)"
    r_all="$(grep -o 'AUTO-TUNE: .*' "${TMP}/at_r_all.log" || true)"
    check "R a region that is the whole box plans exactly like the box" \
          "$([ -n "${r_box}" ] && [ "${r_box}" = "${r_all}" ] && echo 1 || echo 0)"
    run_r() {  # $1 = output root, rest = extra args: a real (CPU) region run on the same simulated machine
        local out="$1"; shift
        env DTFE_RAM_GB=0.2 "${DTFE_BIN}" "${SNAP}" "${out}" --grid 32 32 4 --periodic --input-format 105 --mpc-unit 1 \
            --field density_a velocity_a --region-mpc 30 60 30 60 45 50 "$@" > "${out}.log" 2>&1 || true
    }
    run_r "${TMP}/at_r_auto"
    run_r "${TMP}/at_r_p1" --partition 1 1 1
    check "R a real region run left to the tuner == the same run with '--partition 1 1 1', byte for byte" \
          "$([ -s "${TMP}/at_r_auto.a_den" ] && cmp -s "${TMP}/at_r_auto.a_den" "${TMP}/at_r_p1.a_den" \
             && cmp -s "${TMP}/at_r_auto.a_vel" "${TMP}/at_r_p1.a_vel" && ! grep -q 'AUTO-TUNE:' "${TMP}/at_r_p1.log" \
             && echo 1 || echo 0)"
    rm -f "${TMP}"/at_r*
else
    echo "   SKIP  R (${DTFE_BIN} is not built)"
fi

# The debug switches DTFE_GPU_FLOAT_MOMENTS, DTFE_GPU_FLOAT_MASS and DTFE_MASS_AUDIT (the binary reads empty and
# 0/false/no/off as unset, and the old PS_ names) change what a plan books and what a run may do: with one set the run
# keeps the plan's partitions in flight (Y: those sums follow the thread order) and the plan books no lane share (Q);
# with the float moments the numbers measured on the 64-bit words do not apply (S1, V). SW_SET names the ones set.
SW_SET=""; SW_FMOM=0
for k in DTFE_GPU_FLOAT_MOMENTS DTFE_GPU_FLOAT_MASS DTFE_MASS_AUDIT \
         PS_GPU_FLOAT_MOMENTS PS_GPU_FLOAT_MASS PS_MASS_AUDIT; do
    case "$(printf '%s' "${!k:-0}" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off) ;;
        *) SW_SET="${SW_SET:+${SW_SET} }${k}"; case "${k}" in *FLOAT_MOMENTS) SW_FMOM=1 ;; esac ;;
    esac
done

# ---------- (S) one domain: its grids are counted once ----------
# '--partition 1 1 1' (and one domain picked by the tuner) deposits straight into the run's own grids, which 'fixed'
# books, so the partition term must not add them again (nor every fixed-point grid's conversion at once). The 64^3
# crossed waves on a 768^3 grid measure 22.9 GB peak footprint (24.87 GB in the double build, whose peak
# is the export). S1 pins that report (the run itself takes 23 GB): at or above the measurement, within 15% of it; the
# threads and the machine are pinned (the copy budget and the threads' sample lists follow both). S2 runs one domain
# for real and its peak must stay under the prediction -- the model's contract; no upper bound there: the model follows
# macOS's allocator, which keeps freed blocks under 4 GiB, and glibc returns them (the Linux peak is lower).
echo ">> S: one domain (the 768^3 report; a real 256^3 run)"
SNAP_W="${TMP}/at_input_waves.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_W}" --n 64 --crossed-waves >/dev/null
env OMP_NUM_THREADS=10 DTFE_RAM_GB=64 "${BIN}" "${SNAP_W}" "${TMP}/at_s1" --grid 768 --periodic --input-format 105 --mpc-unit 1 \
    --subsamples 1 --field density_a velocity_a --partition 1 1 1 --auto-tune-report > "${TMP}/at_s1.log" 2>&1
ps1=$(sed -n 's/^AUTO-TUNE-REPORT .*predicted_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_s1.log" | head -1)
if precision_double; then ms1=24.87; else ms1=22.9; fi
if [ "${SW_FMOM}" = 1 ]; then
    echo "   SKIP  S1 (DTFE_GPU_FLOAT_MOMENTS is set: the ${ms1} GB were measured with the 64-bit moments; ${ps1:-none} GB predicted)"
else
check "S1 one domain at 768^3 predicts ${ps1:-none} GB: at least the measured ${ms1} GB, at most 15% above it" \
      "$(awk -v p="${ps1:-0}" -v m="${ms1}" 'BEGIN{print (p >= m && p <= 1.15*m) ? 1 : 0}')"
fi
"${BIN}" "${SNAP_W}" "${TMP}/at_s2" --grid 256 --periodic --input-format 105 --mpc-unit 1 --verbose 1 \
    --subsamples 1 --field density_a velocity_a --partition 1 1 1 > "${TMP}/at_s2.log" 2>&1 || true
ms2=$("${PY}" - "${TMP}/at_s2.log" <<'PYEOF'
import re, sys
m = re.search(r"memory model\s*:\s*([\d.]+) GB predicted \(auto-tune\) vs ([\d.]+) GB measured", open(sys.argv[1]).read())
print(f"{m.group(1)} {m.group(2)}" if m else "none none")
PYEOF
)
check "S2 a real one-domain 256^3 run stays under its prediction (predicted / measured GB: ${ms2})" \
      "$(echo "${ms2}" | awk '$1 != "none" && $1 >= $2 {ok=1} END{print ok ? 1 : 0}')"
rm -f "${SNAP_W}" "${TMP}"/at_s1* "${TMP}"/at_s2.*[a-z]

# ---------- (W) --grid-window: the window's grids ----------
# Under --grid-window every partition deposits into the whole window, whatever the split (ps_interpolation.cc
# psSubgridBox), and the GPU deposit holding its grids runs one at a time. The model used to book the crop of a
# full-grid partition instead -- 1064^2 of these 2048^2 cells at 2^3 -- and predicted 0.37 GB for this run, which
# measures 0.60 GB (the TNG z=0 zooms on a 4096^2 slice: 11.1 GB for 19.75 measured, the rest being the input's
# freed read buffers, which a window's small grids never take over: auto_tune.h autoTuneReadKeptBytes). Window+exact
# over the 64^3 pancake on a one-cell 2048^2 slice, one partition in flight (none parks): the prediction covers the
# measured peak and, on macOS (whose allocator the model follows), stays within 1.5x of it (0.74 GB). The CPU
# window deposit is not modelled this closely: GPU builds only.
echo ">> W: --grid-window books the whole window, its deposit's grids once"
run_w() {  # $1 = log; the rest = extra flags
    local log="$1"; shift
    "${BIN}" "${SNAP}" "${TMP}/at_w" --grid 2048 --periodic --input-format 105 --mpc-unit 1 --verbose 2 --field density_a \
        --gpu --ps-vertex-mass --ps-volume-weighted --exact --partition 2 2 2 --max-concurrent 1 \
        --grid-window 0 "${BOX}" 0 "${BOX}" "$(awk -v b="${BOX}" 'BEGIN{print b/2}')" "$(awk -v b="${BOX}" 'BEGIN{print b/2 + 1e-4}')" \
        "$@" > "${log}" 2>&1 || true
}
run_w "${TMP}/at_w0.log" --auto-tune-report
if grep -q "GPU (--gpu)" "${TMP}/at_w0.log"; then
    run_w "${TMP}/at_w.log"
    mw=$("${PY}" - "${TMP}/at_w.log" <<'PYEOF'
import re, sys
m = re.search(r"memory model\s*:\s*([\d.]+) GB predicted \(auto-tune\) vs ([\d.]+) GB measured", open(sys.argv[1]).read())
print(f"{m.group(1)} {m.group(2)}" if m else "none none")
PYEOF
)
    hi=$([ "$(uname)" = Darwin ] && echo 1.5 || echo 1000)
    check "W a window+exact run's prediction covers its peak (predicted / measured GB: ${mw}; at most ${hi}x on macOS)" \
          "$(echo "${mw}" | awk -v h="${hi}" '$1 != "none" && $1 >= $2 && $1 <= h*$2 {ok=1} END{print ok ? 1 : 0}')"
else
    echo "   SKIP  W (this build has no GPU backend)"
fi
rm -f "${TMP}"/at_w*

# ---------- (Y) dynamic concurrency: more partitions in flight when the memory holds them, same grids ----------
echo ">> Y: dynamic concurrency (DTFE_TEST_PLAN_MC=1: a plan of one partition at a time)"
if precision_double; then
    echo "   SKIP  Y (the double build keeps the plan's concurrency: the single pass's Y6 checks it)"
else
SNAP_Y="${TMP}/at_y.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_Y}" --n 48 --box 100 --amplitude-factor 1.8 >/dev/null
Y_ARGS=(--input-format 105 --mpc-unit 1 --periodic --grid 96 --field density_a velocity_a divergence_a --partition 3 3 3 --verbose 2)
run_y() {  # $1 = tag, then 'VAR=value' assignments, then '--' and extra options
    local tag="$1"; shift
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    env DTFE_RAM_GB=64 ${envs[@]+"${envs[@]}"} "${BIN}" "${SNAP_Y}" "${TMP}/at_y_${tag}" "${Y_ARGS[@]}" "$@" \
        > "${TMP}/at_y_${tag}.log" 2>&1 || true
}
y_same() { cmp -s "${TMP}/at_y_ref.a_den" "${TMP}/at_y_$1.a_den" && cmp -s "${TMP}/at_y_ref.a_vel" "${TMP}/at_y_$1.a_vel" \
           && cmp -s "${TMP}/at_y_ref.a_velDiv" "${TMP}/at_y_$1.a_velDiv"; }
y_most() { sed -n 's/.*dynamic concurrency: at most \([0-9]*\) partitions in flight.*/\1/p' "${TMP}/at_y_$1.log" | tail -1; }
run_y ref -- --max-concurrent 1
run_y dyn DTFE_TEST_PLAN_MC=1 DTFE_FREE_GB=1000 --
m=$(y_most dyn)
if [ -n "${SW_SET}" ]; then   # those sums follow the thread order, so the run keeps the plan's count (DTFE.cpp orderFree)
    check "Y1 with ${SW_SET} set the run keeps the plan's one partition at a time, the grids of '--max-concurrent 1'" \
          "$(grep -q "building up to 1 partition.*dynamic concurrency not started: the double build" "${TMP}/at_y_dyn.log" \
             && [ -z "${m}" ] && y_same dyn && echo 1 || echo 0)"
else
check "Y1 a plan of one partition at a time with the memory for more: up to ${m:-?} in flight, the grids of '--max-concurrent 1'" \
      "$([ -n "${m}" ] && [ "${m}" -gt 1 ] && y_same dyn && echo 1 || echo 0)"
fi
run_y low DTFE_TEST_PLAN_MC=1 DTFE_FREE_GB=4.02 --
check "Y2 free memory for no second partition (4.02 GB less the 4 GB headroom): the plan's 1, same grids" \
      "$(grep -q "building up to 1 partition" "${TMP}/at_y_low.log" && [ -z "$(y_most low)" ] && y_same low && echo 1 || echo 0)"
run_y off DTFE_TEST_PLAN_MC=1 DTFE_FREE_GB=1000 DTFE_DYNAMIC_CONCURRENCY=0 --
run_y user DTFE_FREE_GB=1000 -- --max-concurrent 1
check "Y3 DTFE_DYNAMIC_CONCURRENCY=0 and a user's --max-concurrent keep their one partition at a time" \
      "$(grep -q "building up to 1 partition" "${TMP}/at_y_off.log" && grep -q "building up to 1 partition" "${TMP}/at_y_user.log" \
         && y_same off && echo 1 || echo 0)"
# a binding limit: 4.2 GB free less the 4 GB headroom holds about five of this plan's partitions; all threads reach
# the gate at once, and it admits only as far as that room goes (counting the ones admitted). Where it stops moves by
# one with the gate's timing (4, now and then 5, alone; 4 or 5 under load), so the band is 2..6: more than the plan's
# one and fewer than every thread at once (Y1 reaches one per core, 10 on a 10-core Mac), with no flake at its edge
if [ -n "${SW_SET}" ]; then
    echo "   SKIP  Y5 (${SW_SET} is set: the run keeps the plan's count, Y1)"
else
run_y room DTFE_TEST_PLAN_MC=1 DTFE_FREE_GB=4.2 --
m=$(y_most room)
check "Y5 room for about five partitions: ${m:-?} in flight at most (2..6, not every thread at once), the same grids" \
      "$([ -n "${m}" ] && [ "${m}" -ge 2 ] && [ "${m}" -le 6 ] && y_same room && echo 1 || echo 0)"
fi
# the double build sums per-thread copies in thread order: its grids would follow the count, so it keeps the plan's
if [ -x "${PS_BIN_DOUBLE:-./PS-DTFE-double}" ]; then
    env DTFE_RAM_GB=64 DTFE_TEST_PLAN_MC=1 DTFE_FREE_GB=1000 "${PS_BIN_DOUBLE:-./PS-DTFE-double}" "${SNAP_Y}" "${TMP}/at_y_dbl" \
        "${Y_ARGS[@]}" > "${TMP}/at_y_dbl.log" 2>&1 || true
    agrep "Y6 the double build keeps the plan's one partition at a time (its sums follow the thread order)" \
          "building up to 1 partition.*dynamic concurrency not started: the double build" "${TMP}/at_y_dbl.log"
else
    echo "   SKIP  Y6 (no double build)"
fi
run_y kill DTFE_TEST_PLAN_MC=1 DTFE_FREE_GB=1000 DTFE_TEST_EXIT_AFTER_PARTITIONS=3 --
run_y kill DTFE_TEST_PLAN_MC=1 DTFE_FREE_GB=1000 -- --resume
check "Y4 a dynamic run killed after 3 partitions and resumed: the same grids" \
      "$(grep -q -- "--resume: 3 of 27 partitions" "${TMP}/at_y_kill.log" && y_same kill && echo 1 || echo 0)"
# Y7 (GPU builds): the ledger of an all-GPU run books what the plan books once beside its lanes -- the GPU deposit's
# host snapshot (one deposit holds the device at a time; no evaluation points, so no lane share) -- once, and each
# partition in flight at its lane, as the plan does: the start check's 'once' part is the report's snapshot_gb
Y7_ARGS=(--input-format 105 --mpc-unit 1 --periodic --grid 192 --field density_a velocity_a --partition 2 2 2 --verbose 2
         --gpu)
env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP_Y}" "${TMP}/at_y_g" "${Y7_ARGS[@]}" --auto-tune-report \
    > "${TMP}/at_y_gr.log" 2>&1 || true
if grep -q "GPU (--gpu)" "${TMP}/at_y_gr.log"; then
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_TEST_PLAN_MC=1 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${SNAP_Y}" "${TMP}/at_y_g" \
        "${Y7_ARGS[@]}" > "${TMP}/at_y_g.log" 2>&1 || true
    sg=$(sed -n 's/^AUTO-TUNE-REPORT .* snapshot_gb=\([0-9.]*\) .*/\1/p' "${TMP}/at_y_gr.log" | head -1)
    read -r yp yo <<< "$(sed -n 's/^AUTO-TUNE-DEBUG PS dynamic start: partition \([0-9]*\) B, once \([0-9]*\) B,.*/\1 \2/p' \
                         "${TMP}/at_y_g.log" | head -1)"
    check "Y7 an all-GPU run books its GPU snapshot once: the start check's once part (${yo:-?} B of the partition's ${yp:-?} B) is the plan's snapshot_gb ${sg:-?}" \
          "$(awk -v p="${yp:-0}" -v o="${yo:-0}" -v s="${sg:-x}" \
                 'BEGIN{exit !(s ~ /^[0-9.]+$/ && s > 0.002 && o > 0 && o < p && o/1e9 - s < 0.0006 && s - o/1e9 < 0.0006)}' \
             && echo 1 || echo 0)"
else
    echo "   SKIP  Y7 (this build has no GPU backend)"
fi
rm -rf "${SNAP_Y}" "${TMP}"/at_y_*
fi

# ---------- (X) the partition sub-grid the plan books >= the largest the run allocates ----------
echo ">> X: the booked partition sub-grid against the run's own (DTFE_AUTO_TUNE_DEBUG)"
SNAP_XC="${TMP}/at_x_clump.hdf5"; SNAP_XW="${TMP}/at_x_waves.hdf5"; SNAP_XP="${TMP}/at_x_pancake.hdf5"
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_XC}" --n 32 --box 100 --margin-frac 0.3 >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_XW}" --n 32 --box 100 --amplitude-factor 2.5 --crossed-waves >/dev/null
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_XP}" --n 48 --box 100 --amplitude-factor 1.8 >/dev/null
run_x() {  # $1 = label, $2 = snapshot, rest = options: the booked sub-grid covers the largest real one
    local label="$1" snap="$2"; shift 2
    DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${snap}" "${TMP}/at_x" --input-format 105 --mpc-unit 1 --field density_a --grid 64 "$@" \
        > "${TMP}/at_x.log" 2>&1 || true
    local r
    r="$("${PY}" - "${TMP}/at_x.log" <<'PYEOF'
import re, sys
t = open(sys.argv[1], errors="replace").read()
b = re.findall(r"booked sub-grid: (\d+) cells", t)
a = [int(x) for x in re.findall(r"AUTO-TUNE-DEBUG sub-grid:.*= (\d+) cells", t)]
print(f"{int(b[-1])} {max(a)}" if b and a else "none none")
PYEOF
)"
    local booked="${r% *}" real="${r#* }"
    echo "   .... ${label}: booked ${booked}, largest real ${real}"
    check "X ${label}: the booked sub-grid covers the largest real one" \
          "$([ "${booked}" != none ] && [ "${booked}" -ge "${real}" ] && echo 1 || echo 0)"
    rm -f "${TMP}"/at_x.a_* "${TMP}"/at_x.den* "${TMP}"/at_x.*streams*
}
run_x "one domain, non-periodic clump" "${SNAP_XC}" --partition 1 1 1
run_x "2^3, non-periodic clump" "${SNAP_XC}" --partition 2 2 2
run_x "2^3, crossed waves, --ps-lagrangian-padding 0" "${SNAP_XW}" --periodic --ps-lagrangian-padding 0 --partition 2 2 2
run_x "3^3, crossed waves, --ps-lagrangian-padding 0" "${SNAP_XW}" --periodic --ps-lagrangian-padding 0 --partition 3 3 3
run_x "2^3, crossed waves" "${SNAP_XW}" --periodic --partition 2 2 2
run_x "1 2 2, pancake" "${SNAP_XP}" --periodic --partition 1 2 2
run_x "2 3 4, pancake" "${SNAP_XP}" --periodic --partition 2 3 4
run_x "4 1 1, pancake" "${SNAP_XP}" --periodic --partition 4 1 1
rm -f "${SNAP_XC}" "${SNAP_XW}" "${SNAP_XP}" "${TMP}"/at_x*

# ---------- (Z) a GPU lane books the face sums it holds after the device's deposit ----------
# With '--face-gradient' beside the GPU deposit a lane's largest state comes AFTER the hold: the host words the
# device hands back, the fixed-point face sums made for the face walk, and the Real grids the conversion builds
# beside them (ps_interpolation.cc psCpuFixedUnits, psFaceWalk, psFixedToReal), less the freed words the allocator
# hands back (auto_tune.cc gpuAfterHoldCellBytes, afterHoldReuse). The plan booked the hold alone until 2026-11-10
# (plan P11.7 item 1): the face sums added nothing to a GPU lane. Plan-only: a lane is p(2 in flight) - p(1), p(1)
# less the sub-grids one partition in flight books beside the hold without the face sums (its 'PS one partition in
# flight' line, auto_tune.cc loneLaneSubGrids); the lane with the face sums minus the lane without, per booked
# sub-grid cell, against the planner's own 'PS GPU lane grids' line -- in this selection 514 against 388 B/cell in
# the single build, 902 against 388 in the double build (no fixed-point words to reuse), 582 against 212 under
# DTFE_GPU_FLOAT_MOMENTS=1. The report's GB have two decimals: 4 B/cell of rounding on this sub-grid. Then the plan
# without the face sums at one in flight runs for real, with the production fields (single build): its prediction
# is within 3% of the measured peak or above it.
echo ">> Z: a GPU lane books the face sums it holds after the device's deposit"
run_z() {  # $1 = --max-concurrent, $2 = log, rest = options
    local mc="$1" log="$2"; shift 2
    env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${SNAP}" "${TMP}/at_z" --grid 256 \
        --field density_a velocity_a gradient_a dispersion_a --ps-volume-weighted --input-format 105 --mpc-unit 1 \
        --verbose 2 --periodic --exact --gpu --partition 2 2 2 --max-concurrent "${mc}" --auto-tune-report "$@" \
        > "${log}" 2>&1 || true
}
V_GPU=0   # (section V below runs on a GPU build only)
run_z 1 "${TMP}/at_z_f1.log" --face-gradient
if grep -q "GPU (--gpu)" "${TMP}/at_z_f1.log"; then
    V_GPU=1
    run_z 2 "${TMP}/at_z_f2.log" --face-gradient
    run_z 1 "${TMP}/at_z_n1.log"; run_z 2 "${TMP}/at_z_n2.log"
    r="$("${PY}" - "${TMP}/at_z_f1.log" "${TMP}/at_z_f2.log" "${TMP}/at_z_n1.log" "${TMP}/at_z_n2.log" <<'PYEOF'
import re, sys
t = [open(p, errors="replace").read() for p in sys.argv[1:5]]
p = [re.search(r"^AUTO-TUNE-REPORT .*predicted_gb=([0-9.]+) ", x, re.M) for x in t]
cells = re.findall(r"booked sub-grid: (\d+) cells", t[0])
lane = re.findall(r"PS GPU lane grids: hold (\d+) B/cell, after the hold (\d+) B/cell", t[0])
# what one partition in flight books beside the hold of the device (none with the face sums: their after-hold term
# has it); "device.s" in the pattern, since bash 3.2 pairs the apostrophes of a here-document in a substitution
k = [re.search(r"PS one partition in flight: its own sub-grids beside the device.s hold, (\d+) B/cell x \d+ cells "
               r"= ([0-9.]+) GB", x) for x in t]
if all(p) and cells and lane:
    g = [float(m.group(1)) for m in p]
    kg = [float(m.group(2)) if m else 0. for m in k]
    extra = ((g[1] - g[0] + kg[0]) - (g[3] - g[2] + kg[2]))*1e9/int(cells[-1])
    print(f"{extra:.0f} {int(lane[-1][1]) - int(lane[-1][0])} {lane[-1][0]} {lane[-1][1]} "
          f"{k[2].group(1) if k[2] else 'none'} {1 if k[0] else 0}")
else:
    print("none none none none none 1")
PYEOF
)"
    read -r zx zd zh za zk zf <<< "${r}"
    check "Z the plan books a GPU lane's state after the hold: +${zx} B/cell with the face sums (the planner's after-hold ${za} - hold ${zh} = ${zd} B/cell)" \
          "$([ "${zx}" != none ] && awk -v x="${zx}" -v d="${zd}" 'BEGIN{exit !(d >= 100 && x >= d - 10 && x <= d + 10)}' && echo 1 || echo 0)"
    check "Z one partition in flight books its own sub-grids beside the hold (${zk} B/cell), none with the face sums" \
          "$([ "${zk}" != none ] && [ "${zf}" = 0 ] && echo 1 || echo 0)"
    # one partition in flight on the GPU, for real (the pancake at 2^3 with the production fields: 4.79 to 5.61 GB
    # measured over repeats of this run, 5.02 predicted before loneLaneSubGrids, 5.53 with it): the prediction is
    # within 3% of the peak or above it
    if ! precision_double; then
        env OMP_NUM_THREADS=10 DTFE_RAM_GB=64 DTFE_FREE_GB=1000 "${BIN}" "${SNAP}" "${TMP}/at_z_g1" --grid 256 \
            --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a \
            --ps-volume-weighted --input-format 105 --mpc-unit 1 --verbose 2 --periodic --exact --gpu \
            --partition 2 2 2 --max-concurrent 1 > "${TMP}/at_z_g1.log" 2>&1 || true
        mz="$("${PY}" - "${TMP}/at_z_g1.log" <<'PYEOF'
import re, sys
m = re.search(r"memory model\s*:\s*([\d.]+) GB predicted \(auto-tune\) vs ([\d.]+) GB measured",
              open(sys.argv[1], errors="replace").read())
print(f"{m.group(1)} {m.group(2)}" if m else "none none")
PYEOF
)"
        check "Z one partition in flight on the GPU, a real run: predicted / measured GB ${mz} (at least 0.97 of it)" \
              "$(echo "${mz}" | awk '$1 != "none" && $1 >= 0.97*$2 {ok=1} END{print ok ? 1 : 0}')"
    fi
else
    echo "   SKIP  Z (this build has no GPU backend)"
fi
rm -f "${TMP}"/at_z*

# ---------- (V) a mixed run's GPU lane books the host grids it keeps after the lock ----------
# '--gpu-share' below 1 books the lock holder's device and host grids once (snapshotBytes) and each lane at the larger
# of a CPU deposit and a GPU lane: its arrays before the lock with the host grids the copy-back leaves it after the
# lock (auto_tune.cc gpuAfterLockCellBytes; before plan P13.1 item 13 a only a face pass's state). Where the arrays
# outweigh the CPU deposit -- 128^3 particles on a 128^3 grid in 2^3 partitions -- those host grids are in the lane.
# Plan-only, against the same plan with every deposit on the GPU, whose lane books the whole hold ('once' in the mixed
# lane's line): a lane is p(2 in flight) - p(1) + what p(1) books beside its lane (the look-ahead and, on the GPU
# alone, the sub-grids of its 'PS one partition in flight' line), and the all-GPU lane less the mixed one is
# pre-lock + once - max(cpu, pre-lock + after the lock). The run must exercise the term: max(cpu, pre-lock), the old
# booking, would be 0.06 GB or more off (twice the reports' rounding).
echo ">> V: a mixed run's GPU lane books the host grids it keeps after the lock"
if precision_double; then
    echo "   SKIP  V (the single build's mixed lane is the one checked)"
elif [ "${V_GPU}" != 1 ]; then
    echo "   SKIP  V (this build has no GPU backend)"
elif [ "${SW_FMOM}" = 1 ]; then
    echo "   SKIP  V (DTFE_GPU_FLOAT_MOMENTS is set: the CPU deposit outweighs this lane there, so the term does not bind)"
else
    SNAP_V="${TMP}/at_v_pancake128.hdf5"
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP_V}" --n 128 --box "${BOX}" --amplitude-factor 1.8 \
        >/dev/null
    run_v() {  # $1 = tag, $2 = --max-concurrent, rest = options
        local tag="$1" mc="$2"; shift 2
        env DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_AUTO_MINN=1000 DTFE_AUTO_TUNE_DEBUG=1 "${BIN}" "${SNAP_V}" \
            "${TMP}/at_v" --grid 128 --field density_a velocity_a gradient_a dispersion_a --ps-volume-weighted \
            --input-format 105 --mpc-unit 1 --verbose 2 --periodic --exact --gpu --partition 2 2 2 \
            --max-concurrent "${mc}" --auto-tune-report "$@" > "${TMP}/at_v_${tag}.log" 2>&1 || true
    }
    run_v m1 1 --gpu-share 0.5; run_v m2 2 --gpu-share 0.5; run_v g1 1; run_v g2 2
    r="$("${PY}" - "${TMP}" <<'PYEOF'
import re, sys
def lane(tag):  # p(2) - p(1) + what p(1) books beside its lane (the look-ahead, a lone partition its sub-grids), GB
    p = {}
    for c in (1, 2):
        t = open(f"{sys.argv[1]}/at_v_{tag}{c}.log", errors="replace").read()
        m = re.search(r"^AUTO-TUNE-REPORT .*predicted_gb=([0-9.]+) .*lookahead_gb=([0-9.]+) ", t, re.M)
        k = re.search(r"PS one partition in flight: .* = ([0-9.]+) GB", t)
        p[c] = (float(m.group(1)), float(m.group(2)) + (float(k.group(1)) if k else 0.))
    return p[2][0] - p[1][0] + p[1][1]
try:
    t = open(f"{sys.argv[1]}/at_v_m1.log", errors="replace").read()
    C, G, O, H = (float(x) for x in re.findall(r"PS mixed lane: cpu ([0-9.]+) GB, gpu pre-lock ([0-9.]+) GB, "
                                               r"once ([0-9.]+) GB, after the lock ([0-9.]+) GB", t)[-1])
    cells = re.findall(r"booked sub-grid: (\d+) cells", t)[-1]
    d = lane("g") - lane("m")
    want = G + O - max(C, G + H)
    ok = abs(d - want) <= 0.03 and G + H - max(C, G) >= 0.06
    print(f"{1 if ok else 0}|the all-GPU lane less the mixed one {d:.3f} GB = pre-lock {G:.3f} + once {O:.3f} - "
          f"max(cpu {C:.3f}, pre-lock + after the lock {H:.3f}) = {want:.3f} GB ({cells} cells booked)")
except Exception as e:  # a missing report or debug line
    print(f"0|not read from the plans ({type(e).__name__}: {e})")
PYEOF
)"
    check "V a mixed run's GPU lane books the host grids it keeps after the lock: ${r#*|}" "${r%%|*}"
    rm -f "${SNAP_V}" "${TMP}"/at_v*
fi

# ---------- (Q) the lane share of several partitions in flight ----------
# Partitions in flight do not reach their largest state at the same moment, and only one of them has the largest
# sub-grid: the plan books the first lane whole and each other lane at a share of it (plan P11.8; auto_tune.cc
# laneShareCut: the larger of 0.55 and the mean span over the largest, the device's hold grids at 0.30), for 2 to 4
# in flight when every deposit is on the GPU and 2 to 6 when the free cores deposit too. Above that every lane is
# booked whole, as it is without evaluation points (no measurement there) and in the double build.
# Plan-only, from the report's 'predicted_gb' at a given '--max-concurrent' and the planner's own 'PS lane share'
# line (DTFE_AUTO_TUNE_DEBUG=1), with base = p(1) less its look-ahead and, on the GPU alone, the sub-grids one
# partition in flight books beside the device's hold (its 'PS one partition in flight' line), A = a whole lane,
# B = a shared one:
#   Q1 all on the GPU: p(c) = base + (c-1) B up to 4 in flight, base + 4 A at 5;
#   Q2 '--gpu-share 0.5': the same up to 6 in flight, base + 6 A at 7;
#   Q3 no evaluation points: no share (every lane whole: p(3) - base is twice p(2) - base);
#   Q4 '--scratch-dir': what the share leaves out is covered up to what the scratch grids keep in memory (C):
#      p(4) = base + 3 B + min(C, 3 (A - B)).
# Q5 runs Q2's selection under a budget that holds 4 whole lanes: the plan is 6 in flight, the dynamic concurrency
# adds none (the share is measured for no more), and the grids are those of '--max-concurrent 1' byte for byte.
# Q6 and Q7: the share's own search counts the parallel insert's copies (it left them out and the report added
# them: a plan could exceed its own budget; found in review, 2026-11-11). Q6 reads what a forced
# '--parallel-triangulation' books a lane (T: 3 in flight with it less 3 without); Q7 plans under a budget that
# holds 3 in flight by the share without the copies and not with them: 2 in flight, within the budget.
# Q8: one partition in flight on the GPU alone books its own sub-grids beside the device's hold (auto_tune.cc
# loneLaneSubGrids: 30 B/cell in this selection, the density, the velocity, the counts and flags); none at two in
# flight, none in a mixed run (Z: none with the face sums).
# The report's GB have two decimals: sums of a few of them agree within 0.03 GB. GPU builds only; skipped with
# DTFE_GPU_FLOAT_MOMENTS, DTFE_GPU_FLOAT_MASS or DTFE_MASS_AUDIT set (no share is booked), Q6/Q7 also without TBB.
echo ">> Q: the lane share of several partitions in flight"
# The share is fitted on the fixed-point moments, the integer mass and no audit (auto_tune.cc laneShareOn): a set
# switch (the planner reads empty and 0/false/no/off as unset, and the old PS_ names) prints no share line at all
Q_OFF="${SW_SET}"   # (read before section S)
# Q6 and Q7 need the parallel insert compiled in (triangulation.cpp's message names it), else a lane has no copies
Q_TBB=0; grep -aq "parallel, TBB" "${BIN}" && Q_TBB=1
if precision_double; then
    echo "   SKIP  Q (the double build books every lane whole)"
elif [ -n "${Q_OFF}" ]; then
    echo "   SKIP  Q (${Q_OFF} is set: the plan books no lane share there)"
else
QSCR="${TMP}/at_q_scr"; mkdir -p "${QSCR}"
Q_ARGS=(--field density_a velocity_a --input-format 105 --mpc-unit 1 --verbose 2 --periodic --exact --gpu --partition 2 2 2)
run_q() {  # $1 = tag, then 'VAR=value' assignments, then '--' and extra options
    local tag="$1"; shift
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    env OMP_NUM_THREADS=10 DTFE_RAM_GB=64 DTFE_FREE_GB=1000 DTFE_AUTO_MINN=1000 DTFE_AUTO_TUNE_DEBUG=1 \
        ${envs[@]+"${envs[@]}"} "${BIN}" "${SNAP}" "${TMP}/at_q_${tag}" "${Q_ARGS[@]}" "$@" \
        > "${TMP}/at_q_${tag}.log" 2>&1 || true
}
for c in 1 2 3 4 5; do
    run_q "g${c}" DTFE_AUTO_PTS_N=4000000 -- --grid 256 --max-concurrent "${c}" --auto-tune-report
done
if grep -q "GPU (--gpu)" "${TMP}/at_q_g1.log"; then
    if [ "${Q_TBB}" = 1 ]; then
        run_q f3 DTFE_AUTO_PTS_N=4000000 -- --grid 256 --parallel-triangulation --max-concurrent 3 --auto-tune-report
    fi
    for c in 1 2 6 7; do
        run_q "m${c}" DTFE_AUTO_PTS_N=4000000 -- --grid 256 --gpu-share 0.5 --max-concurrent "${c}" --auto-tune-report
    done
    for c in 1 2 3; do run_q "n${c}" -- --grid 256 --max-concurrent "${c}" --auto-tune-report; done
    for c in 1 4; do
        run_q "c${c}" DTFE_AUTO_PTS_N=100000 -- --grid 512 --scratch-dir "${QSCR}" --max-concurrent "${c}" \
            --auto-tune-report
    done
    r="$("${PY}" - "${TMP}" "${Q_TBB}" <<'PYEOF'
import re, sys
TOL = 0.03
def read(tag):  # the predicted peak, what it books beside its lanes at one in flight (the look-ahead, and on the GPU
                # alone the own sub-grids of the partition: the "PS one partition in flight" line) and the last
                # lane-share line (A, B, C, most in flight) of a log
    t = open(f"{sys.argv[1]}/at_q_{tag}.log", errors="replace").read()
    m = re.search(r"^AUTO-TUNE-REPORT .*predicted_gb=([0-9.]+) .*lookahead_gb=([0-9.]+) ", t, re.M)
    k = re.search(r"PS one partition in flight: .* = ([0-9.]+) GB", t)
    s = re.findall(r"PS lane share: first lane ([0-9.]+) GB, each other ([0-9.]+) GB, .*scratch grids ([0-9.]+) GB, "
                   r"used for 2 to ([0-9]+) in flight", t)
    return (float(m.group(1)), float(m.group(2)) + (float(k.group(1)) if k else 0.),
            tuple(float(x) for x in s[-1]) if s else None)
def near(a, b):
    return abs(a - b) <= TOL
def q1():
    p = {c: read(f"g{c}") for c in (1, 2, 3, 4, 5)}
    A, B, C, most = p[2][2]
    base = p[1][0] - p[1][1]
    ok = (most == 4 and A - B > 0.1 and all(near(p[c][0], base + (c - 1)*B) for c in (2, 3, 4))
          and near(p[5][0], base + 4*A))
    return ok, (f"all on the GPU: a whole lane {A:.3f} GB, a shared one {B:.3f} GB; 1 to 5 in flight "
                + " ".join(f"{p[c][0]:.2f}" for c in (1, 2, 3, 4, 5)) + " GB (shared up to 4, whole at 5)"), None
def q2():
    p = {c: read(f"m{c}") for c in (1, 2, 6, 7)}
    A, B, C, most = p[2][2]
    base = p[1][0] - p[1][1]
    ok = (most == 6 and A - B > 0.1 and near(p[2][0], base + B) and near(p[6][0], base + 5*B)
          and near(p[7][0], base + 6*A) and base + 4*A - p[6][0] > 0.2)
    # the budget of Q5: half way between 6 in flight by the share and 5 whole lanes
    return ok, (f"'--gpu-share 0.5': a whole lane {A:.3f} GB, a shared one {B:.3f} GB; 1, 2, 6 and 7 in flight "
                + " ".join(f"{p[c][0]:.2f}" for c in (1, 2, 6, 7)) + " GB (shared up to 6, whole at 7)"), \
           (p[6][0] + base + 4*A)/2
def q3():
    p = {c: read(f"n{c}") for c in (1, 2, 3)}
    base = p[1][0] - p[1][1]
    lane = p[2][0] - base
    ok = all(p[c][2] is None for c in p) and lane > 0.1 and near(p[3][0], base + 2*lane)
    return ok, (f"no evaluation points: no share, every lane {lane:.2f} GB (1 to 3 in flight "
                + " ".join(f"{p[c][0]:.2f}" for c in (1, 2, 3)) + " GB)"), None
def q4():
    p = {c: read(f"c{c}") for c in (1, 4)}
    A, B, C, most = p[4][2]
    base = p[1][0] - p[1][1]
    want = base + 3*B + min(C, 3*(A - B))
    ok = C > 0.1 and near(p[4][0], want)
    return ok, (f"'--scratch-dir': 4 in flight {p[4][0]:.2f} GB = base {base:.2f} + 3 shared lanes of {B:.3f} + the "
                f"smaller of {C:.3f} GB of scratch grids and the {3*(A - B):.3f} GB the share leaves out"), None
def q6():
    p3, f3 = read("g3")[0], read("f3")[0]
    A, B, C, most = read("g2")[2]
    T = (f3 - p3)/3
    ok = 0.02 < T < B
    # the budget of Q7: 3 in flight by the share fit it without the copies, not with them
    return ok, (f"'--parallel-triangulation': {T:.3f} GB of insert copies a lane (3 in flight {f3:.2f} GB, "
                f"{p3:.2f} GB without it)"), p3 + 1.5*T
def q8():   # one partition in flight on the GPU alone: its own sub-grids beside the hold of the device
    def lone(tag):
        t = open(f"{sys.argv[1]}/at_q_{tag}.log", errors="replace").read()
        m = re.search(r"PS one partition in flight: its own sub-grids beside the device.s hold, (\d+) B/cell x "
                      r"(\d+) cells = ([0-9.]+) GB", t)
        return (int(m.group(1)), int(m.group(2)), float(m.group(3))) if m else None
    n1, n2, m1 = lone("n1"), lone("n2"), lone("m1")
    ok = (n1 is not None and n1[0] == 30 and n1[2] > 0.05 and abs(n1[0]*n1[1]/1e9 - n1[2]) < 0.001
          and n2 is None and m1 is None)
    return ok, ("one partition in flight on the GPU alone, its own sub-grids beside the hold: "
                + (f"{n1[0]} B/cell x {n1[1]} cells = {n1[2]:.3f} GB" if n1 else "no line")
                + f"; at two in flight {'none' if n2 is None else 'a line'}, in a mixed run "
                + f"{'none' if m1 is None else 'a line'}"), None
checks = ([("Q1", q1), ("Q2", q2), ("Q3", q3), ("Q4", q4)] + ([("Q6", q6)] if sys.argv[2] == "1" else [])
          + [("Q8", q8)])
for name, f in checks:
    try:
        ok, text, budget = f()
    except Exception as e:  # a missing report or lane-share line
        ok, text, budget = False, f"not read from the plans ({type(e).__name__}: {e})", None
    print(f"{name}|{1 if ok else 0}|{text}")
    if ok and budget is not None:
        print(f"BUDGET{name}|{budget:.2f}|")
PYEOF
)"
    qb=""; qb6=""
    while IFS='|' read -r qn qo qd; do
        case "${qn}" in
            BUDGETQ2) qb="${qo}" ;;
            BUDGETQ6) qb6="${qo}" ;;
            *) check "${qn} ${qd}" "${qo}" ;;
        esac
    done <<< "${r}"
    if [ -n "${qb}" ]; then
        run_q run DTFE_AUTO_PTS_N=4000000 DTFE_MEM_BUDGET_GB="${qb}" -- --grid 256 --gpu-share 0.5
        run_q ref DTFE_AUTO_PTS_N=4000000 DTFE_MEM_BUDGET_GB="${qb}" -- --grid 256 --gpu-share 0.5 --max-concurrent 1
        check "Q5 a budget of ${qb} GB (4 whole lanes): the plan is 6 in flight, none added by the dynamic concurrency, the grids of '--max-concurrent 1'" \
              "$(grep -q -- "--partition 2 2 2 --max-concurrent 6 " "${TMP}/at_q_run.log" \
                 && grep -q "dynamic concurrency not started: the memory model's lane share" "${TMP}/at_q_run.log" \
                 && [ -s "${TMP}/at_q_ref.a_den" ] && cmp -s "${TMP}/at_q_ref.a_den" "${TMP}/at_q_run.a_den" \
                 && cmp -s "${TMP}/at_q_ref.a_vel" "${TMP}/at_q_run.a_vel" && echo 1 || echo 0)"
    else
        check "Q5 the run under a budget between the two bookings (no budget: Q2 failed)" 0
    fi
    if [ "${Q_TBB}" != 1 ]; then
        echo "   SKIP  Q6/Q7 (no TBB in this build: a lane has no insert copies to book)"
    elif [ -n "${qb6}" ]; then
        run_q f DTFE_AUTO_PTS_N=4000000 DTFE_MEM_BUDGET_GB="${qb6}" -- --grid 256 --parallel-triangulation \
            --auto-tune-report
        check "Q7 a budget of ${qb6} GB (3 in flight by the share without the insert's copies): the plan is 2 in flight and within the budget" \
              "$(grep -q "^AUTO-TUNE-REPORT partition=2 mc=2 .*over_budget=0" "${TMP}/at_q_f.log" && echo 1 || echo 0)"
    else
        check "Q7 the plan under a budget between the two (no budget: Q6 failed)" 0
    fi
else
    echo "   SKIP  Q (this build has no GPU backend)"
fi
fi
rm -rf "${TMP}"/at_q_*

echo "------------------------------------------------------------"
if [ "${FAILS}" -gt 0 ]; then
    echo "RESULT: FAIL (${FAILS} check(s))"
    exit 1
fi
echo "RESULT: PASS  (auto-tune refuses honestly when the grid cannot fit, tunes when it can,"
echo "               its predictions bound the measured peak, splits a small set only when that"
echo "               is faster, and it keeps to a 60%-of-RAM budget that also leaves room for"
echo "               what other programs use)"
echo "============================================================"
