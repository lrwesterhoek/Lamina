#!/usr/bin/env bash
# PS-DTFE parallel-vs-serial correctness check: runs serial and all-cores, asserts
# outputs agree (a race shows up as a mismatch). .streams must match exactly (integer
# counts are order-independent), float fields only within tolerance.
# Then the CPU deposit's dynamic team (ps_interpolation.cc PSBlockClaim): on all cores, the default run and
# DTFE_DEPOSIT_THREADS=1 (one thread per partition, the serial loop) and =4 (a pinned team of 4) must write
# BYTE-IDENTICAL grids, and the verbose-3 log must say which rule ran: the dynamic team in the single build
# (every accumulator a shared integer word), the static copy mode in the double build (its sums depend on the
# block assignment, so it must keep the static rule). So must a slab team of 4 (DTFE_DEPOSIT_SLABS=1
# DTFE_DEPOSIT_THREADS=4): a static team, counted in the core pool (thread_count.h CorePool::Hold), beside the
# other partitions' pooled loops.
# Then D19 (docs/cleanup-plan.md): the parallel insert (DTFE_TRIANGULATION_THREADS=4, forced at this size) against
# the sequential one ('--sequential-triangulation') must write the same bytes -- the cells' canonical vertex order
# (src/CGAL_triangulation/ps_vertex_order.h) -- for the exact deposit with the face gradient and the caustics, the
# sampled deposit with the vertex mass on partitions, an exact lattice, the cusp stencil on that lattice and, on a
# Metal build, the GPU, with the same fixed-point units; single build only, and only where the build has the
# parallel insert (else SKIP). A run under DTFE_PS_VERTEX_ORDER_LEGACY=1 says so in its log and renumbers nothing.
#
# Usage:
#   tests/ps_parallel_check.sh                 # build, run x2, compare
#   tests/ps_parallel_check.sh --no-build
#   N=64 GRID=128 tests/ps_parallel_check.sh   # bigger problem
# Requires the build toolchain plus python3 with numpy and h5py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-32}"; GRID="${GRID:-64}"; BOX="${BOX:-100.0}"; AMP="${AMP:-1.8}"
PARTITION="${PARTITION:-2 2 2}"   # Lagrangian partition grid; parallelism is over these
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"; mkdir -p "${TMP}"
# Input name must NOT share a prefix with the output roots (ps_serial/ps_par),
# or the per-run 'rm -f <root>.*' would delete it.
SNAP="${TMP}/ps_pcheck_input.hdf5"
NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "============================================================"
echo " PS-DTFE parallel-vs-serial check   N=${N}^3  grid=${GRID}^3  cores=${NPROC}  partition=[${PARTITION}]"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

echo ">> generating test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" \
    --out "${SNAP}" --n "${N}" --box "${BOX}" --amplitude-factor "${AMP}" >/dev/null

# Every run here is the sampled deposit ('density_a' would select the exact one): the exact integer '.streams' below
# and the team rules the logs name are its own.
run() {  # $1 = thread count, $2 = output root
    rm -f "$2".*
    local log="$2.log"
    local t0=$SECONDS
    set +e
    OMP_NUM_THREADS="$1" "${BIN}" "${SNAP}" "$2" \
        --grid "${GRID}" --periodic --partition ${PARTITION} --field density velocity density_a --sampled \
        --input-format 105 --mpc-unit 1 --verbose 1 > "$log" 2>&1
    local rc=$?
    set -e
    echo "   [threads=$1] wall time: $(( SECONDS - t0 )) s"
    die_on_failed_run "$rc" "$log" "PS-DTFE (threads=$1)"
}

echo ">> run 1: serial  (OMP_NUM_THREADS=1, --partition ${PARTITION})"
run 1 "${TMP}/ps_serial"
echo ">> run 2: parallel (OMP_NUM_THREADS=${NPROC}, --partition ${PARTITION})"
run "${NPROC}" "${TMP}/ps_par"

echo ">> files produced:"
ls -la "${TMP}"/ps_serial.* "${TMP}"/ps_par.* 2>&1 | sed 's/^/   /' || true

echo ">> comparing serial vs parallel outputs ..."
"${PY}" - "${TMP}/ps_serial" "${TMP}/ps_par" "${GRID}" <<'PY'
import os, sys
import numpy as np

a, b, grid = sys.argv[1], sys.argv[2], int(sys.argv[3])
ncell = grid ** 3
fails = []

def load(path, ncomp):
    raw = np.fromfile(path, dtype=np.uint8); n = ncell * ncomp
    if raw.size == n * 4: return raw.view(np.float32).astype(np.float64)
    if raw.size == n * 8: return raw.view(np.float64)
    sys.exit(f"FAIL: {path} unexpected size {raw.size}")

def compare(ext, ncomp, exact, tol=1e-3):
    fa, fb = a + ext, b + ext
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(f"   {ext:10s} MISSING (serial={os.path.exists(fa)}, parallel={os.path.exists(fb)})")
        fails.append(f"{ext} not produced by both runs"); return
    x, y = load(fa, ncomp), load(fb, ncomp)
    dmax = float(np.abs(x - y).max())
    rel = dmax / (float(np.abs(x).max()) + 1e-30)
    print(f"   {ext:10s} max|Δ|={dmax:.3e}  max relΔ={rel:.3e}"
          + ("  [must be exact]" if exact else f"  [tol {tol:.0e}]"))
    if exact and dmax != 0.0:
        fails.append(f"{ext} not identical ({dmax:.3e}) -- possible race")
    if (not exact) and rel > tol:
        fails.append(f"{ext} relΔ {rel:.2e} exceeds {tol:.0e} -- possible race")

compare(".streams",  1, True)       # integer counts -> order-independent -> exact
# .a_streams is volume-averaged (fractional): partitions sum in a run-dependent order,
# so it agrees only to ~1 float32 ULP. Benign FP non-associativity, not a race (a race
# would drop/double whole streams, rel ~1e-2 >> tol).
compare(".a_streams", 1, False)
compare(".den",      1, False)
compare(".a_den",    1, False)
compare(".vel",      3, False)

print("-" * 60)
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("RESULT: PASS  (parallel == serial: integer stream counts identical, "
      "averaged/float fields agree within rounding)")
PY

echo ">> the CPU deposit's team: default vs DTFE_DEPOSIT_THREADS=1 and =4 and a slab team of 4 (--verbose 3, all cores), byte for byte"
run_team() {  # $1 = DTFE_DEPOSIT_THREADS ('' = unset), $2 = output root, $3 = one more VAR=value (optional)
    rm -f "$2".*
    set +e
    env ${1:+DTFE_DEPOSIT_THREADS=$1} ${3:-} OMP_NUM_THREADS="${NPROC}" "${BIN}" "${SNAP}" "$2" \
        --grid "${GRID}" --periodic --partition ${PARTITION} --field density velocity density_a velocity_a --sampled \
        --input-format 105 --mpc-unit 1 --verbose 3 > "$2.log" 2>&1
    local rc=$?
    set -e
    die_on_failed_run "$rc" "$2.log" "PS-DTFE (DTFE_DEPOSIT_THREADS=${1:-unset})"
}
run_team "" "${TMP}/ps_team_def"
run_team 1 "${TMP}/ps_team_t1"
run_team 4 "${TMP}/ps_team_t4"
run_team 4 "${TMP}/ps_team_slab" DTFE_DEPOSIT_SLABS=1
team_fail=0
# (the double build's copy mode sums its private copies in thread order: only its integer stream counts must agree;
# slab mode adds into every cell in the one-thread loop's order, so its grids must agree in both builds)
for f in "${TMP}"/ps_team_t1.*; do
    case "$f" in *.log|*.runlog*) continue ;; esac
    ext="${f#"${TMP}/ps_team_t1"}"
    for other in def t4 slab; do
        if precision_double && [ "${other}" != slab ] && [ "${ext}" != ".streams" ]; then continue; fi
        g="${TMP}/ps_team_${other}${ext}"
        if cmp -s "$f" "$g"; then :; else echo "   FAIL: ${ext} differs (DTFE_DEPOSIT_THREADS=1 vs ${other})"; team_fail=1; fi
    done
done
n_dyn_def=$(grep -c "a dynamic team" "${TMP}/ps_team_def.log" || true)
n_dyn_t4=$(grep -c "a dynamic team: blocks of 256 cells claimed from one counter by the 4 pinned threads" "${TMP}/ps_team_t4.log" || true)
n_static=$(grep -cE "CPU deposit on [0-9]+ threads? \(copy mode: [0-9.e+-]+ MB of private accumulators per extra thread, budget [0-9.e+-]+ GB; the fixed-point grids, counts and flags are shared\)\.$" "${TMP}/ps_team_def.log" || true)
n_slab=$(grep -c "CPU deposit in SLAB mode: 4 threads" "${TMP}/ps_team_slab.log" || true)
echo "   logs: ${n_dyn_def} partition deposit(s) in a dynamic team (default), ${n_dyn_t4} in a pinned team of 4, ${n_static} on the static rule, ${n_slab} in a slab team of 4"
[ "${n_slab}" -gt 0 ] || { echo "   FAIL: DTFE_DEPOSIT_SLABS=1 DTFE_DEPOSIT_THREADS=4 should run a slab team of 4"; team_fail=1; }
if precision_double; then
    # the double build sums private copies in thread order: the static rule, never the dynamic team
    [ "${n_dyn_def}" = 0 ] && [ "${n_dyn_t4}" = 0 ] && [ "${n_static}" -gt 0 ] \
        || { echo "   FAIL: the double build must keep the static rule"; team_fail=1; }
else
    [ "${n_dyn_def}" -gt 0 ] && [ "${n_dyn_t4}" -gt 0 ] \
        || { echo "   FAIL: the single build's CPU deposit should run as a dynamic team"; team_fail=1; }
fi
[ "${team_fail}" = 0 ] || { echo "RESULT: FAIL (the deposit's team)"; exit 1; }
echo "   PASS: the grids agree across the four team rules ($(precision_double && echo "double build: .streams, the slab team every grid" || echo "every grid byte for byte"))"

echo ">> D19: the parallel insert against the sequential one, byte for byte (the canonical vertex order)"
# src/CGAL_triangulation/ps_vertex_order.h renumbers every cell in a canonical vertex order, so each tetrahedron's
# arithmetic does not depend on CGAL's numbering, and every grid of the single build is an order-free fixed-point
# sum: a tessellation inserted in parallel (another cell order, another numbering) must give the sequential run's
# bytes. At this small size the insert is forced both ways: '--sequential-triangulation' against
# DTFE_TRIANGULATION_THREADS=4 (a count forces the parallel insert at any size). Cases: the exact deposit with the
# face gradient and the caustics on one tessellation; the sampled deposit with the vertex mass on the suite's
# partitions (index-order merge, one in flight, so every partition logs its units); an exact lattice
# (--jitter-frac 0: CGAL's symbolic perturbation must give the same tetrahedra in any insertion order); the cusp
# stencil of '--ps-caustic-cusps' on that lattice (its distance ties take the strict order); the GPU's exact deposit
# on a Metal build. The fixed-point units lines (--verbose 3) are compared too, so a unit that moved would be named.
# A build without the parallel insert has one order only (SKIP). The double build sums in walk order (SKIP). Under
# DTFE_PS_VERTEX_ORDER_LEGACY=1 (CGAL's numbering) the pair should differ: reported, not a gate.
order_fail=0
if precision_double; then
    echo "   SKIP: the double build sums in walk order (its private copies, its serial adds)"
else
    # (DTFE_EXACT_AVERAGE and DTFE_OVERLAP from a login shell would change the deposit or silence a look-ahead build)
    unset DTFE_TRIANGULATION_THREADS DTFE_TBB_THREADS DTFE_PS_VERTEX_ORDER_LEGACY DTFE_EXACT_AVERAGE DTFE_OVERLAP
    LAT="${TMP}/ps_order_lattice.hdf5"
    "${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${LAT}" --n "${N}" --box "${BOX}" \
        --amplitude-factor "${AMP}" --jitter-frac 0 >/dev/null
    # run_order <root> <snapshot> <sequential|parallel> [VAR=value ...] -- <options>
    run_order() {
        local root="$1" snap="$2" how="$3"; shift 3
        local envs=()
        local opts=()
        while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
        [ $# -gt 0 ] && shift
        opts=("$@")
        if [ "${how}" = parallel ]; then envs+=(DTFE_TRIANGULATION_THREADS=4)
        else opts+=(--sequential-triangulation); fi
        rm -f "${root}".*
        set +e
        env ${envs[@]+"${envs[@]}"} "${BIN}" "${snap}" "${root}" --periodic --input-format 105 --mpc-unit 1 \
            --verbose 3 ${opts[@]+"${opts[@]}"} > "${root}.log" 2>&1
        local rc=$?
        set -e
        die_on_failed_run "${rc}" "${root}.log" "PS-DTFE (${how} insert)"
    }
    # order_same <a> <b> <label>: every output of run a has a byte-identical twin in run b (logs, run logs and
    # staging files are no outputs), each one that differs named; then the units lines of the two logs
    order_same() {
        local n=0 bad=0 f
        for f in "$1".*; do
            case "$f" in *.log|*.runlog*|*.part.*|*.units) continue ;; esac
            n=$((n+1))
            cmp -s "$f" "$2${f#"$1"}" || { echo "      differs: ${f#"$1"}"; bad=$((bad+1)); }
        done
        grep -F "fixed-point units" "$1.log" | sort > "$1.units" || true
        grep -F "fixed-point units" "$2.log" | sort > "$2.units" || true
        if ! cmp -s "$1.units" "$2.units"; then
            echo "      the fixed-point units differ (a bound within rounding of a power of two):"
            diff "$1.units" "$2.units" | sed 's/^/        /' || true
            bad=$((bad+1))
        fi
        echo "   $3: ${n} outputs, ${bad} differ"
        [ "${n}" -gt 0 ] && [ "${bad}" = 0 ]
    }
    # a GPU run that deposited on the GPU: a GPU line (the run header's 'PS deposit ... GPU (--gpu)'; a partition's own
    # deposit line is suppressed) and no fallback or decline line
    gpu_ran() {
        grep -qE ' GPU \(| GPU deposit \(' "$1" \
            && ! grep -qE 'using the CPU deposit|deposit unavailable|ran on the CPU instead|gave up|falling back' "$1" \
            && ! grep -qF 'is not used for the deposit' "$1"
    }
    RENUM="(the canonical vertex order renumbered "
    EX=(--grid "${GRID}" --partition 1 --exact --field density_a velocity_a gradient_a divergence_a shear_a
        vorticity_a dispersion_a --ps-caustics --ps-volume-weighted --face-gradient)
    run_order "${TMP}/ps_ord_par" "${SNAP}" parallel -- "${EX[@]}"
    if ! grep -qF "(parallel, TBB, " "${TMP}/ps_ord_par.log"; then
        echo "   SKIP: no '(parallel, TBB, N threads)' line (a build without the parallel insert, or one thread:"
        echo "         one insertion order only)"
    else
        run_order "${TMP}/ps_ord_seq" "${SNAP}" sequential -- "${EX[@]}"
        grep -qF "${RENUM}" "${TMP}/ps_ord_seq.log" \
            || { echo "   FAIL: the log does not name the canonical vertex order's pass (see ${TMP}/ps_ord_seq.log)"
                 order_fail=1; }
        order_same "${TMP}/ps_ord_seq" "${TMP}/ps_ord_par" "exact deposit, face gradient, caustics (one tessellation)" \
            || order_fail=1
        SA=(--grid "${GRID}" --partition ${PARTITION} --max-concurrent 1 --sampled --ps-vertex-mass
            --field density_a velocity_a dispersion_a)
        run_order "${TMP}/ps_ord_sseq" "${SNAP}" sequential -- "${SA[@]}"
        run_order "${TMP}/ps_ord_spar" "${SNAP}" parallel -- "${SA[@]}"
        # (the forced count is capped by the machine's threads and OMP_NUM_THREADS: with one, this pair inserts
        # sequentially on both sides, and then passes without exercising the order)
        grep -qF "(parallel, TBB, " "${TMP}/ps_ord_spar.log" \
            || echo "   NOTE: the partitioned pair inserted sequentially (one thread per tessellation): a trivial pass"
        order_same "${TMP}/ps_ord_sseq" "${TMP}/ps_ord_spar" "sampled deposit, vertex mass, --partition ${PARTITION}" \
            || order_fail=1
        LA=(--grid "${GRID}" --partition 1 --exact --field density_a velocity_a)
        run_order "${TMP}/ps_ord_lseq" "${LAT}" sequential -- "${LA[@]}"
        run_order "${TMP}/ps_ord_lpar" "${LAT}" parallel -- "${LA[@]}"
        order_same "${TMP}/ps_ord_lseq" "${TMP}/ps_ord_lpar" \
            "an exact lattice (if only this differs: the tetrahedra did)" || order_fail=1
        CU=(--grid "${GRID}" --partition 1 --exact --field density_a --ps-caustics --ps-caustic-cusps)
        run_order "${TMP}/ps_ord_cseq" "${LAT}" sequential -- "${CU[@]}"
        run_order "${TMP}/ps_ord_cpar" "${LAT}" parallel -- "${CU[@]}"
        order_same "${TMP}/ps_ord_cseq" "${TMP}/ps_ord_cpar" "the cusp stencil on the lattice (--ps-caustic-cusps)" \
            || order_fail=1
        if [ "$(gpu_backend_of o_ps)" = metal ]; then
            run_order "${TMP}/ps_ord_gseq" "${SNAP}" sequential -- "${EX[@]}" --gpu
            run_order "${TMP}/ps_ord_gpar" "${SNAP}" parallel -- "${EX[@]}" --gpu
            if gpu_ran "${TMP}/ps_ord_gseq.log" && gpu_ran "${TMP}/ps_ord_gpar.log"; then
                order_same "${TMP}/ps_ord_gseq" "${TMP}/ps_ord_gpar" "the GPU exact deposit" || order_fail=1
            else
                echo "   SKIP the GPU pair: a run did not deposit on the GPU (see ${TMP}/ps_ord_g*.log)"
            fi
        else
            echo "   SKIP the GPU pair: o_ps is not a Metal build"
        fi
        run_order "${TMP}/ps_ord_xseq" "${SNAP}" sequential DTFE_PS_VERTEX_ORDER_LEGACY=1 -- "${EX[@]}"
        run_order "${TMP}/ps_ord_xpar" "${SNAP}" parallel DTFE_PS_VERTEX_ORDER_LEGACY=1 -- "${EX[@]}"
        grep -qF "DTFE_PS_VERTEX_ORDER_LEGACY=1" "${TMP}/ps_ord_xseq.log" \
            || { echo "   FAIL: a run under DTFE_PS_VERTEX_ORDER_LEGACY=1 should say so in its log"; order_fail=1; }
        if grep -qF "${RENUM}" "${TMP}/ps_ord_xseq.log"; then
            echo "   FAIL: a run under DTFE_PS_VERTEX_ORDER_LEGACY=1 renumbered its cells"; order_fail=1
        fi
        if order_same "${TMP}/ps_ord_xseq" "${TMP}/ps_ord_xpar" "CGAL's numbering" >/dev/null; then
            echo "   NOTE: under CGAL's numbering the two inserts agreed too: this case does not exercise the numbering"
        else
            echo "   (control) under CGAL's numbering (DTFE_PS_VERTEX_ORDER_LEGACY=1) the two inserts differ"
        fi
    fi
    [ "${order_fail}" = 0 ] \
        || { echo "RESULT: FAIL (the parallel insert moved bytes the canonical vertex order holds)"; exit 1; }
fi

echo "============================================================"
echo " parallel check PASSED"
echo "============================================================"
