#!/usr/bin/env bash
# PS-DTFE --tessellation-cache: a stale or mismatched cache would corrupt results silently, so most checks are about when it must NOT be used.
#   A miss then hit   A2 body deflated, header plaintext   B hit outputs bit-identical   C another grid still hits
#   D touched input or --mean-density misses   E partitions cached separately   E2 partition padding is part of the identity
#   F partitioned hit bit-identical   G standard binary caches thread sub-domains   G2 ... and per partition
#   H chunked format=3, format 2 misses   I block-wise reader/writer match CGAL   J symlinked or relative input name hits
#   K libdeflate and zlib read each other's files   L concurrent writers   M dead writers' temp files removed   N canonical vertex order
# Usage: tests/ps_tessellation_cache_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${DTFE_TEST_TMP}"
SNAP="${TMP}/ptc_input.hdf5"
# The cache must live on a LOCAL volume (the binary rejects iCloud Drive); ${TMPDIR:-/tmp} also works on Linux CI,
# where /private/tmp does not exist.
CACHE="${TESS_CACHE_DIR:-${TMPDIR:-/tmp}/dtfe-tesscache-test}"

echo "============================================================"
echo " PS-DTFE --tessellation-cache check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

build_target "${1:-}" PS-DTFE o_ps
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

rm -rf "${CACHE}"; mkdir -p "${CACHE}"
rm -f "${TMP}"/ptc_out_*

echo ">> generating the test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
        --crossed-waves >/dev/null

# a small point set for the --sample-points half of the contract
POINTS="${TMP}/ptc_points.bin"
"${PY}" - "$POINTS" "$BOX" <<'EOF'
import numpy as np, sys
rng = np.random.default_rng(7)
L = float(sys.argv[2])
rng.uniform(0.05*L, 0.95*L, size=(20000, 3)).astype(np.float64).tofile(sys.argv[1])
EOF

COMMON=(--periodic --field density velocity --mpc-unit 1 --max-concurrent 1 --sample-points "${POINTS}")

fails=0
note() { echo "   $1"; }
# check_yes <1 = pass> <text> (it counts a failure in fails): tests/lib.sh

# run <outroot> <grid> <extra args...>; captures stdout in RUN_OUT. A non-zero exit must not kill
# the script under 'set -e' -- the checks below report it as a failure instead.
run() {
    local out="$1" grid="$2"; shift 2
    RUN_OUT="$( "${BIN}" "${SNAP}" "${out}" --grid "${grid}" "${COMMON[@]}" "$@" --verbose 3 2>&1 || true )"
}
hits()    { echo "${RUN_OUT}" | grep -c "Reused the cached" || true; }

echo ""
echo "(A) first run misses and writes, second run hits"
run "${TMP}/ptc_out_nocache" "${GRID}"                                   # reference, no cache
run "${TMP}/ptc_out_write" "${GRID}" --tessellation-cache "${CACHE}"
check_yes "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "first cached run is a MISS"
nfiles=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
check_yes "$([ "${nfiles}" = "1" ] && echo 1 || echo 0)" "exactly one cache file written (got ${nfiles})"
run "${TMP}/ptc_out_hit" "${GRID}" --tessellation-cache "${CACHE}"
check_yes "$([ "$(hits)" -ge "1" ] && echo 1 || echo 0)" "second run HITS (different output root)"

echo ""
echo "(A2) the file is compressed, with a PLAINTEXT header"
# The body is deflated but magic + version + descriptor stay plaintext, so a staleness check needs no inflation;
# asserting the layout stops a silent revert to an uncompressed body.
tessfile=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | head -1)
if [ -n "${tessfile}" ]; then
    head -c 8 "${tessfile}" | grep -q "DTFETESS" \
        && check_yes 1 "header starts with the plaintext magic" \
        || check_yes 0 "header starts with the plaintext magic"
    # the gzip magic (1f 8b) must appear AFTER the header, not at byte 0
    zoff=$("${PY}" -c "
import sys
d = open(sys.argv[1], 'rb').read(65536)
sys.stdout.write(str(d.find(b'\x1f\x8b')))
" "${tessfile}")
    check_yes "$([ "${zoff}" -gt "8" ] && echo 1 || echo 0)" "deflated body begins after the header (gzip magic at byte ${zoff})"
    grep -aq "format=" "${tessfile}" \
        && check_yes 1 "descriptor is readable without inflating" \
        || check_yes 0 "descriptor is readable without inflating"
else
    check_yes 0 "a cache file exists to inspect"
fi

echo ""
echo "(B) a hit reproduces every output bit-for-bit"
for f in "${TMP}"/ptc_out_nocache.*; do
    ext="${f##*ptc_out_nocache}"
    g="${TMP}/ptc_out_hit${ext}"
    if [ -f "${g}" ] && cmp -s "${f}" "${g}"; then
        check_yes 1 "bit-identical ${ext}"
    else
        check_yes 0 "bit-identical ${ext}"
    fi
done

echo ""
echo "(C) the grid is not part of the tessellation: a different grid still hits"
run "${TMP}/ptc_out_grid" "$((GRID*2))" --tessellation-cache "${CACHE}"
check_yes "$([ "$(hits)" -ge "1" ] && echo 1 || echo 0)" "different --grid still HITS"

echo ""
echo "(D) staleness is detected"
run "${TMP}/ptc_out_dens" "${GRID}" --tessellation-cache "${CACHE}" --mean-density 2.5
check_yes "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "changed --mean-density MISSES"
touch "${SNAP}"
run "${TMP}/ptc_out_touched" "${GRID}" --tessellation-cache "${CACHE}"
check_yes "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "touched input file MISSES"

echo ""
echo "(E) a multi-partition run caches EACH partition separately"
# 'partNo' is -1 for every partition of a Lagrangian split, so the descriptor must separate them by their unpadded
# Lagrangian region, else partitions 2..N would load the first partition's tessellation.
PCACHE="${CACHE}-part"; rm -rf "${PCACHE}"; mkdir -p "${PCACHE}"
run "${TMP}/ptc_part_cold" "${GRID}" --tessellation-cache "${PCACHE}" --partition 2 2 2
nfiles=$(ls -1 "${PCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
check_yes "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "cold partitioned run is all misses"
check_yes "$([ "${nfiles}" = "8" ] && echo 1 || echo 0)" "one cache file per partition (got ${nfiles}, want 8)"
run "${TMP}/ptc_part_warm" "${GRID}" --tessellation-cache "${PCACHE}" --partition 2 2 2
check_yes "$([ "$(hits)" = "8" ] && echo 1 || echo 0)" "warm partitioned run hits all 8 partitions"
# the partitions must NOT all be the same tessellation
ndistinct=$(echo "${RUN_OUT}" | grep -oE "Reused the cached tessellation \([0-9]+" | grep -oE "[0-9]+$" | sort -u | wc -l | tr -d ' ')
check_yes "$([ "${ndistinct}" -ge "2" ] && echo 1 || echo 0)" "partitions restored DIFFERENT tessellations (${ndistinct} distinct vertex counts)"
# E2: the default padding is 8 spacings ('lagPadding=8'); an Eulerian-padded cache (no 'lagPadding' line) must MISS
# in a default run, and '--ps-lagrangian-padding 0' must find it
ndesc=$( { grep -l -a "lagPadding=8" "${PCACHE}"/*.tess 2>/dev/null || true; } | wc -l | tr -d ' ')
check_yes "$([ "${ndesc}" = "8" ] && echo 1 || echo 0)" "E2 every default partition records 'lagPadding=8' in its descriptor (got ${ndesc} of 8)"
LCACHE2="${CACHE}-lagpad"; rm -rf "${LCACHE2}"; mkdir -p "${LCACHE2}"
run "${TMP}/ptc_lagpad0_cold" "${GRID}" --tessellation-cache "${LCACHE2}" --partition 2 2 2 --ps-lagrangian-padding 0
nold=$( { grep -L -a "lagPadding=" "${LCACHE2}"/*.tess 2>/dev/null || true; } | wc -l | tr -d ' ')   # grep -L exits 1 (pipefail)
check_yes "$([ "$(hits)" = "0" ] && [ "${nold}" = "8" ] && echo 1 || echo 0)" "E2 '--ps-lagrangian-padding 0' writes 8 Eulerian-padded partitions with no 'lagPadding' line (got ${nold})"
run "${TMP}/ptc_lagpad_dflt" "${GRID}" --tessellation-cache "${LCACHE2}" --partition 2 2 2
check_yes "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "E2 a default run misses the caches of the Eulerian padding (they rebuild once)"
run "${TMP}/ptc_lagpad0_warm" "${GRID}" --tessellation-cache "${LCACHE2}" --partition 2 2 2 --ps-lagrangian-padding 0
check_yes "$([ "$(hits)" = "8" ] && echo 1 || echo 0)" "E2 '--ps-lagrangian-padding 0' (the way back) hits its 8 Eulerian partitions"
rm -rf "${LCACHE2}"

echo ""
echo "(F) a partitioned cache hit reproduces the uncached run bit-for-bit"
run "${TMP}/ptc_part_ref" "${GRID}" --partition 2 2 2          # no cache at all
pf=0
for f in "${TMP}"/ptc_part_ref.*; do
    ext="${f##*ptc_part_ref}"
    g="${TMP}/ptc_part_warm${ext}"
    if [ -f "${g}" ] && cmp -s "${f}" "${g}"; then check_yes 1 "bit-identical ${ext}"; else check_yes 0 "bit-identical ${ext}"; pf=1; fi
done

echo ""
echo "(G) the STANDARD binary caches its processor-split sub-domains too"
# DTFE's thread sub-domains are not --partition: each has its own region in the descriptor and its own cache file.
# Only one 'Reused' line prints (other threads run at verbosity 0), so the evidence is that the warm run adds no files.
SCACHE="${CACHE}-std"; rm -rf "${SCACHE}"; mkdir -p "${SCACHE}"
if [ -x "${DTFE_BIN}" ]; then
    "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_cold" --grid "${GRID}" --periodic --field density \
           --mpc-unit 1 --tessellation-cache "${SCACHE}" >/dev/null 2>&1 || true
    n_cold=$(ls -1 "${SCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
    check_yes "$([ "${n_cold}" -ge "2" ] && echo 1 || echo 0)" "standard binary wrote one file per sub-domain (${n_cold})"
    "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_warm" --grid "${GRID}" --periodic --field density \
           --mpc-unit 1 --tessellation-cache "${SCACHE}" >/dev/null 2>&1 || true
    n_warm=$(ls -1 "${SCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
    check_yes "$([ "${n_warm}" = "${n_cold}" ] && echo 1 || echo 0)" "warm run added no files (${n_warm} == ${n_cold}) => every sub-domain hit"
    # '.den' is the deterministic unaveraged deposit. ('.a_den' would NOT do here: its Monte-Carlo
    # seed is std::rand() unless --seed is given, so it differs between two UNCACHED runs as well.)
    if cmp -s "${TMP}/ptc_std_cold.den" "${TMP}/ptc_std_warm.den"; then
        check_yes 1 "standard '.den' is bit-identical across the cache"
    else
        check_yes 0 "standard '.den' is bit-identical across the cache"
    fi
    rm -rf "${SCACHE}"
else
    echo "   SKIP  (${DTFE_BIN} not built)"
fi

echo ""
echo "(G2) a PARTITIONED standard run caches every partition's sub-domains"
# A standard partition is split into thread sub-boxes, each its own tessellation, so a 2x2x2 split at 2 threads writes
# 16 files, each descriptor naming its partition's region. Under '--sample-points' a partition is ONE tessellation.
PSCACHE="${CACHE}-stdpart"; rm -rf "${PSCACHE}"; mkdir -p "${PSCACHE}"
if [ -x "${DTFE_BIN}" ]; then
    # the seeded Monte Carlo average, as written: '--seed' alone does not select it beside the '_a' fields
    SP=(--grid "${GRID}" --periodic --mpc-unit 1 --padding 5 --field density density_a velocity_a --sampled --seed 5
        --partition 2 2 2 --max-concurrent 2)
    srun() { env OMP_NUM_THREADS=2 "${DTFE_BIN}" "${SNAP}" "$@" >/dev/null 2>&1 || true; }
    sameout() {  # every output of $1 byte-identical in $2 (the padding test's per-thread files are not outputs)
        local n=0 f
        for f in "$1".*; do
            case "$f" in *badGridPoints*|*.runlog*) continue ;; esac
            cmp -s "$f" "$2${f#"$1"}" || return 1
            n=$((n+1))
        done
        [ "$n" -gt 0 ]
    }
    srun "${TMP}/ptc_sp_ref" "${SP[@]}"
    srun "${TMP}/ptc_sp_cold" "${SP[@]}" --tessellation-cache "${PSCACHE}"
    n_cold=$(ls -1 "${PSCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
    n_named=$(for f in "${PSCACHE}"/*.tess; do grep -a -m1 '^box=' "$f"; done 2>/dev/null | sort -u | wc -l | tr -d ' ')
    check_yes "$([ "${n_cold}" = "16" ] && [ "${n_named}" = "8" ] && echo 1 || echo 0)" \
          "G2 2x2x2 partitions x 2 threads: 16 sub-domain files (${n_cold}), naming the 8 partitions' regions (${n_named})"
    before="$(ls -li "${PSCACHE}")"
    srun "${TMP}/ptc_sp_warm" "${SP[@]}" --tessellation-cache "${PSCACHE}"
    check_yes "$([ "${before}" = "$(ls -li "${PSCACHE}")" ] && echo 1 || echo 0)" "G2 the warm run loaded every sub-domain (no file rewritten)"
    check_yes "$(sameout "${TMP}/ptc_sp_ref" "${TMP}/ptc_sp_cold" && sameout "${TMP}/ptc_sp_ref" "${TMP}/ptc_sp_warm" && echo 1 || echo 0)" \
          "G2 the cold and the warm cached run: every output byte-identical to the uncached run"
    rm -rf "${PSCACHE}"; mkdir -p "${PSCACHE}"
    SPP=("${SP[@]}" --sample-points "${POINTS}")
    srun "${TMP}/ptc_spp_ref" "${SPP[@]}"
    srun "${TMP}/ptc_spp_cold" "${SPP[@]}" --tessellation-cache "${PSCACHE}"
    n_cold=$(ls -1 "${PSCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
    n_named=$(for f in "${PSCACHE}"/*.tess; do grep -a -m1 '^region=' "$f"; done 2>/dev/null | sort -u | wc -l | tr -d ' ')
    before="$(ls -li "${PSCACHE}")"
    srun "${TMP}/ptc_spp_warm" "${SPP[@]}" --tessellation-cache "${PSCACHE}"
    check_yes "$([ "${n_cold}" = "8" ] && [ "${n_named}" = "8" ] && [ "${before}" = "$(ls -li "${PSCACHE}")" ] && echo 1 || echo 0)" \
          "G2 --sample-points: one file per partition (${n_cold} of 8, keyed by its own region), all loaded by the warm run"
    check_yes "$(sameout "${TMP}/ptc_spp_ref" "${TMP}/ptc_spp_warm" && [ -s "${TMP}/ptc_spp_warm.pts_den" ] && echo 1 || echo 0)" \
          "G2 --sample-points: the warm cached run's outputs (.pts_* and grids) byte-identical to the uncached run"
    rm -rf "${PSCACHE}" "${TMP}"/ptc_sp_* "${TMP}"/ptc_spp_*
else
    echo "   SKIP  (${DTFE_BIN} not built)"
fi

echo ""
echo "(H) the CHUNKED body (parallel inflate), format 3, and a file of format 2"
tessfile=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | head -1)
chunked=$("${PY}" -c "
import struct, sys
f = open(sys.argv[1], 'rb'); f.read(8); (n,) = struct.unpack('<I', f.read(4)); f.read(n)
sys.stdout.write('1' if f.read(8) == b'DTFECHK1' else '0')
" "${tessfile}")
check_yes "${chunked}" "a new cache file carries the chunked body (marker after the descriptor)"
# fmt_of <file>: the value of the 'format=' line of its plaintext descriptor (empty without one)
fmt_of() {
    "${PY}" - "$1" <<'PYEOF'
import struct, sys
f = open(sys.argv[1], "rb"); f.read(8); (n,) = struct.unpack("<I", f.read(4))
for line in f.read(n).split(b"\n"):
    if line.startswith(b"format="):
        sys.stdout.write(line[7:].decode("ascii", "replace")); break
PYEOF
}
# set_fmt <file> <from> <to>: rewrites the descriptor's 'format=<from>' line as 'format=<to>' in place (one digit for
# another: the descriptor's length and the file's name stay); fails when there is no such line
set_fmt() {
    "${PY}" - "$1" "$2" "$3" <<'PYEOF'
import struct, sys
path, old, new = sys.argv[1], ("format=" + sys.argv[2]).encode(), ("format=" + sys.argv[3]).encode()
with open(path, "r+b") as f:
    f.read(8); (n,) = struct.unpack("<I", f.read(4)); lines = f.read(n).split(b"\n")
    if len(old) != len(new) or old not in lines:
        sys.exit(1)
    lines[lines.index(old)] = new
    f.seek(12); f.write(b"\n".join(lines))
PYEOF
}
FCACHE="${CACHE}-format"; rm -rf "${FCACHE}"; mkdir -p "${FCACHE}"
run "${TMP}/ptc_fmt_cold" "${GRID}" --tessellation-cache "${FCACHE}"
ffile=$(ls -1 "${FCACHE}"/*.tess 2>/dev/null | head -1 || true)
fcur=""; [ -n "${ffile}" ] && fcur="$(fmt_of "${ffile}" || true)"
check_yes "$([ "${fcur}" = 3 ] && echo 1 || echo 0)" "a new cache file is of format 3 (its descriptor: 'format=${fcur:-?}')"
if [ -n "${ffile}" ] && set_fmt "${ffile}" 3 2; then
    run "${TMP}/ptc_fmt_miss" "${GRID}" --tessellation-cache "${FCACHE}"
    check_yes "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" \
          "a file of format 2 (the new file, its format line rewritten; the same name) is a MISS"
    nf=$(ls -1 "${FCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ' || true)
    check_yes "$([ "$(fmt_of "${ffile}")" = 3 ] && [ "${nf}" = "1" ] && echo 1 || echo 0)" \
          "... and is rebuilt: the run rewrites it as format 3 (${nf} file in the folder)"
    same=1
    for f in "${TMP}"/ptc_out_nocache.*; do
        ext="${f##*ptc_out_nocache}"; cmp -s "${f}" "${TMP}/ptc_fmt_miss${ext}" || same=0
    done
    check_yes "${same}" "... and that run reproduces the uncached run bit-for-bit"
    run "${TMP}/ptc_fmt_hit" "${GRID}" --tessellation-cache "${FCACHE}"
    check_yes "$([ "$(hits)" -ge "1" ] && echo 1 || echo 0)" "... and the next run hits the rebuilt file"
else
    check_yes 0 "a new cache file with a 'format=3' line to rewrite as format 2"
fi
rm -rf "${FCACHE}"

echo ""
echo "(I) the block-wise reader and writer: exactly CGAL's triangulation, exactly CGAL's bytes"
# The block-wise reader (readTriangulationBlocks) replaces CGAL's operator>>: DTFE_TESSELLATION_CACHE_CGAL_IO=1
# restores CGAL's, and DTFE_TESSELLATION_CACHE_FINGERPRINT=1 prints a hash of the loaded structure; same hashes expected.
fprints() { echo "${RUN_OUT}" | grep "^TESS_FINGERPRINT" | sort || true; }
export DTFE_TESSELLATION_CACHE_FINGERPRINT=1
for which in single part; do
    if [ "${which}" = "single" ]; then dir="${CACHE}"; extra=(); else dir="${PCACHE}"; extra=(--partition 2 2 2); fi
    export DTFE_TESSELLATION_CACHE_CGAL_IO=1
    run "${TMP}/ptc_rd_cgal_${which}" "${GRID}" --tessellation-cache "${dir}" ${extra[@]+"${extra[@]}"}
    unset DTFE_TESSELLATION_CACHE_CGAL_IO
    fp_cgal="$(fprints)"; h_cgal=$(hits)
    run "${TMP}/ptc_rd_block_${which}" "${GRID}" --tessellation-cache "${dir}" ${extra[@]+"${extra[@]}"}
    fp_block="$(fprints)"; h_block=$(hits)
    nfp=$(echo "${fp_block}" | grep -c . || true)
    check_yes "$([ "${h_cgal}" -ge 1 ] && [ "${h_cgal}" = "${h_block}" ] && [ "${nfp}" = "${h_block}" ] && echo 1 || echo 0)" \
          "${which}: both readers hit (${h_cgal} / ${h_block}) and fingerprint every load (${nfp})"
    check_yes "$([ -n "${fp_block}" ] && [ "${fp_cgal}" = "${fp_block}" ] && echo 1 || echo 0)" \
          "${which}: identical structure fingerprints"
    same=1
    for f in "${TMP}"/ptc_rd_cgal_${which}.*; do
        ext="${f##*ptc_rd_cgal_${which}}"
        cmp -s "${f}" "${TMP}/ptc_rd_block_${which}${ext}" || same=0
    done
    check_yes "${same}" "${which}: bit-identical outputs, sample points included"
done
if [ -x "${DTFE_BIN}" ]; then
    SCACHE="${CACHE}-std"; rm -rf "${SCACHE}"; mkdir -p "${SCACHE}"
    # The plan is pinned to one partition: the auto-tuner reads live free memory, so a loaded machine could give the
    # writer and a reader different plans.
    "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_rd0" --grid "${GRID}" --periodic --field density --mpc-unit 1 \
           --partition 1 1 1 --tessellation-cache "${SCACHE}" >/dev/null 2>&1 || true
    # Each reader's whole output stays beside its grids: a mismatch that a rerun does not repeat (a loaded machine)
    # can still be read afterwards, and the failure shows where the two lists part and how each run used the cache.
    DTFE_TESSELLATION_CACHE_CGAL_IO=1 "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_rd1" --grid "${GRID}" --periodic \
           --field density --mpc-unit 1 --partition 1 1 1 --tessellation-cache "${SCACHE}" > "${TMP}/ptc_std_rd1.log" 2>&1 || true
    "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_rd2" --grid "${GRID}" --periodic \
           --field density --mpc-unit 1 --partition 1 1 1 --tessellation-cache "${SCACHE}" > "${TMP}/ptc_std_rd2.log" 2>&1 || true
    fp_cgal="$(grep "^TESS_FINGERPRINT" "${TMP}/ptc_std_rd1.log" | sort || true)"
    fp_block="$(grep "^TESS_FINGERPRINT" "${TMP}/ptc_std_rd2.log" | sort || true)"
    same_fp=$([ -n "${fp_block}" ] && [ "${fp_cgal}" = "${fp_block}" ] && echo 1 || echo 0)
    check_yes "${same_fp}" "standard binary: identical structure fingerprints ($(echo "${fp_block}" | grep -c . || true) sub-domains)"
    if [ "${same_fp}" = 0 ]; then
        echo "      CGAL's reader (<, ${TMP}/ptc_std_rd1.log) vs the block reader (>, ${TMP}/ptc_std_rd2.log):"
        diff <(printf '%s\n' "${fp_cgal}") <(printf '%s\n' "${fp_block}") | grep '^[<>]' | head -n 20 | sed 's/^/      | /' || true
        for n in 1 2; do   # the RUNNING line names the cache folder too: the command, not how the run used it
            grep -iE 'cache|hit' "${TMP}/ptc_std_rd${n}.log" | grep -v '^RUNNING:' | sed "s/^/      rd${n} | /" || true
        done
    fi
    cmp -s "${TMP}/ptc_std_rd1.den" "${TMP}/ptc_std_rd2.den" && check_yes 1 "standard binary: bit-identical '.den'" \
                                                            || check_yes 0 "standard binary: bit-identical '.den'"
    SCACHE2="${CACHE}-std2"; rm -rf "${SCACHE2}"; mkdir -p "${SCACHE2}"
    DTFE_TESSELLATION_CACHE_CGAL_IO=1 "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_wr" --grid "${GRID}" --periodic --field density \
           --mpc-unit 1 --tessellation-cache "${SCACHE2}" >/dev/null 2>&1 || true
    same=1; nw=0
    for f in "${SCACHE2}"/*.tess; do nw=$((nw+1)); cmp -s "${f}" "${SCACHE}/$(basename "${f}")" || same=0; done
    check_yes "$([ "${nw}" -ge 2 ] && [ "${same}" = 1 ] && echo 1 || echo 0)" "standard binary: the block writer's ${nw} sub-domain files are byte-identical to CGAL's"
    rm -rf "${SCACHE}" "${SCACHE2}"
fi
unset DTFE_TESSELLATION_CACHE_FINGERPRINT
# ... and the block-wise WRITER writes CGAL's bytes: the same run cached by each writer gives
# byte-identical files (the chunks deflate independently, so equal content means equal files)
for which in single part; do
    if [ "${which}" = "single" ]; then extra=(); else extra=(--partition 2 2 2); fi
    WA="${CACHE}-wcgal"; WB="${CACHE}-wblock"; rm -rf "${WA}" "${WB}"; mkdir -p "${WA}" "${WB}"
    export DTFE_TESSELLATION_CACHE_CGAL_IO=1
    run "${TMP}/ptc_wr_cgal" "${GRID}" --tessellation-cache "${WA}" ${extra[@]+"${extra[@]}"}
    unset DTFE_TESSELLATION_CACHE_CGAL_IO
    run "${TMP}/ptc_wr_block" "${GRID}" --tessellation-cache "${WB}" ${extra[@]+"${extra[@]}"}
    same=1; nw=0
    for f in "${WA}"/*.tess; do
        nw=$((nw+1))
        cmp -s "${f}" "${WB}/$(basename "${f}")" || same=0
    done
    check_yes "$([ "${nw}" -ge 1 ] && [ "${same}" = 1 ] && echo 1 || echo 0)" "${which}: the block writer's ${nw} cache file(s) are byte-identical to CGAL's"
    rm -rf "${WA}" "${WB}"
done

echo ""
echo "(J) one key per FILE: a symlinked or relative name of the input hits the cache its absolute name wrote"
# The descriptor stores the input's realpath (canonical_path.h), so '/tmp/x' and '/private/tmp/x' or 'x' run
# from its folder are one key, not three (two needless rebuilds of the same tessellation).
LINKDIR="${TMP}_link"; rm -f "${LINKDIR}"; ln -s "${TMP}" "${LINKDIR}"
nbefore=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
RUN_OUT="$( "${BIN}" "${LINKDIR}/$(basename "${SNAP}")" "${TMP}/ptc_sym" --grid "${GRID}" "${COMMON[@]}" \
            --tessellation-cache "${CACHE}" --verbose 3 2>&1 || true )"
check_yes "$([ "$(hits)" -ge 1 ] && echo 1 || echo 0)" "a path through a symlinked folder HITS"
ABSBIN="$(pwd)/${PS_BIN#./}"
RUN_OUT="$( cd "${TMP}" && "${ABSBIN}" "$(basename "${SNAP}")" "${TMP}/ptc_rel" --grid "${GRID}" "${COMMON[@]}" \
            --tessellation-cache "${CACHE}" --verbose 3 2>&1 || true )"
check_yes "$([ "$(hits)" -ge 1 ] && echo 1 || echo 0)" "a relative path HITS"
same=1
for f in "${TMP}"/ptc_out_hit.*; do
    ext="${f##*ptc_out_hit}"
    for g in "${TMP}/ptc_sym${ext}" "${TMP}/ptc_rel${ext}"; do cmp -s "${f}" "${g}" || same=0; done
done
check_yes "${same}" "... both bit-identical to the absolute-path hit"
nafter=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
check_yes "$([ "${nafter}" = "${nbefore}" ] && echo 1 || echo 0)" "and no new cache file was written (${nbefore} -> ${nafter})"
rm -f "${LINKDIR}"

echo ""
echo "(K) the chunk codec: libdeflate and zlib read each other's files (both write plain gzip members)"
run "${TMP}/ptc_codec_probe" "${GRID}" --tessellation-cache "${CACHE}"
if grep -q "read with libdeflate" <<< "${RUN_OUT}"; then
    for w in zlib libdeflate; do
        KC="${CACHE}-codec-${w}"; rm -rf "${KC}"; mkdir -p "${KC}"
        if [ "${w}" = zlib ]; then export DTFE_TESSELLATION_CACHE_ZLIB=1; fi
        run "${TMP}/ptc_codec_w_${w}" "${GRID}" --tessellation-cache "${KC}"
        unset DTFE_TESSELLATION_CACHE_ZLIB
        if [ "${w}" = libdeflate ]; then export DTFE_TESSELLATION_CACHE_ZLIB=1; fi
        run "${TMP}/ptc_codec_r_${w}" "${GRID}" --tessellation-cache "${KC}"
        reader=$(echo "${RUN_OUT}" | grep -o "read with [a-z]*" | head -1 | cut -d' ' -f3)
        unset DTFE_TESSELLATION_CACHE_ZLIB
        same=1
        for f in "${TMP}"/ptc_out_nocache.*; do
            ext="${f##*ptc_out_nocache}"; cmp -s "${f}" "${TMP}/ptc_codec_r_${w}${ext}" || same=0
        done
        check_yes "$([ "$(hits)" -ge 1 ] && [ "${reader}" != "${w}" ] && [ "${same}" = 1 ] && echo 1 || echo 0)" \
              "written with ${w}, read with ${reader}: a hit, bit-identical to the uncached run"
        rm -rf "${KC}"
    done
else
    echo "   SKIP  (this build has no libdeflate: every chunk goes through zlib)"
fi

echo ""
echo "(L) two runs writing ONE cache folder at the same moment: each writer has its own temp file"
# Two writers of one partition must not share one '<file>.part': each writes '<file>.part.<pid>' (the map
# '.occ.tmp.<pid>') and renames it into place. Either way both succeed, no temp file is left, and the cache is whole.
CC="${CACHE}-conc"; rm -rf "${CC}"; mkdir -p "${CC}"
CARGS=(--grid "${GRID}" "${COMMON[@]}" --partition 2 2 2 --tessellation-cache "${CC}" --verbose 3)
"${BIN}" "${SNAP}" "${TMP}/ptc_conc1" "${CARGS[@]}" > "${TMP}/ptc_conc1.log" 2>&1 & pid1=$!
"${BIN}" "${SNAP}" "${TMP}/ptc_conc2" "${CARGS[@]}" > "${TMP}/ptc_conc2.log" 2>&1 & pid2=$!
rc1=0; wait "${pid1}" || rc1=$?
rc2=0; wait "${pid2}" || rc2=$?
check_yes "$([ "${rc1}" = "0" ] && [ "${rc2}" = "0" ] && echo 1 || echo 0)" "both concurrent runs exit 0 (${rc1}, ${rc2})"
ntess=$(ls -1 "${CC}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
nocc=$(ls -1 "${CC}"/*.occ 2>/dev/null | wc -l | tr -d ' ')
ntmp=$(find "${CC}" -name '*.part*' -o -name '*.tmp*' | wc -l | tr -d ' ')
check_yes "$([ "${ntess}" = "8" ] && [ "${nocc}" = "8" ] && [ "${ntmp}" = "0" ] && echo 1 || echo 0)" \
      "8 cache files and 8 occupancy maps, no '.part.<pid>' or '.occ.tmp.<pid>' left (${ntess}, ${nocc}, ${ntmp})"
same=1
for f in "${TMP}"/ptc_part_ref.*; do
    ext="${f##*ptc_part_ref}"
    for g in "${TMP}/ptc_conc1${ext}" "${TMP}/ptc_conc2${ext}"; do cmp -s "${f}" "${g}" || same=0; done
done
check_yes "${same}" "... both bit-identical to the uncached run"
run "${TMP}/ptc_conc3" "${GRID}" --tessellation-cache "${CC}" --partition 2 2 2
same=1
for f in "${TMP}"/ptc_part_ref.*; do ext="${f##*ptc_part_ref}"; cmp -s "${f}" "${TMP}/ptc_conc3${ext}" || same=0; done
check_yes "$([ "$(hits)" = "8" ] && [ "${same}" = 1 ] && echo 1 || echo 0)" "a third run hits all 8 partitions ($(hits)) and reproduces the uncached run bit-for-bit"

echo ""
echo "(M) a DEAD writer's temp file is removed by the next save of that file; a live writer's is kept"
# save() first removes the '<file>.part.<pid>' siblings whose writer is dead (kill(pid, 0) fails with ESRCH) and keeps
# a live writer's; the dead pid is one that has just exited (a fixed number could be alive on Linux).
victim=$(ls -1 "${CC}"/*.tess 2>/dev/null | head -1)
other=$(ls -1 "${CC}"/*.tess 2>/dev/null | sed -n 2p)
if [ -n "${victim}" ] && [ -n "${other}" ]; then
    sh -c 'exit 0' & dead=$!; wait "${dead}" || true
    sleep 60 & live=$!
    echo "dead writer" > "${victim}.part.${dead}"; echo "live writer" > "${victim}.part.${live}"
    echo "old shared name" > "${victim}.part"; echo "dead writer" > "${other}.part.${dead}"
    rm -f "${victim}"                                       # its partition misses, so it is saved again
    run "${TMP}/ptc_stale" "${GRID}" --tessellation-cache "${CC}" --partition 2 2 2
    kill "${live}" 2>/dev/null || true; wait "${live}" 2>/dev/null || true
    check_yes "$([ "$(hits)" = "7" ] && [ -f "${victim}" ] && echo 1 || echo 0)" "the deleted partition misses and is saved again (hits $(hits) of 8)"
    check_yes "$([ ! -e "${victim}.part.${dead}" ] && echo 1 || echo 0)" "... the dead writer's '.part.${dead}' beside it is removed"
    check_yes "$([ -f "${victim}.part.${live}" ] && echo 1 || echo 0)" "... a live writer's '.part.${live}' is kept"
    check_yes "$([ -f "${victim}.part" ] && [ -f "${other}.part.${dead}" ] && echo 1 || echo 0)" \
          "... the old shared '.part' and another cache file's temp file are left alone"
    same=1
    for f in "${TMP}"/ptc_part_ref.*; do ext="${f##*ptc_part_ref}"; cmp -s "${f}" "${TMP}/ptc_stale${ext}" || same=0; done
    check_yes "${same}" "... and the run reproduces the uncached run bit-for-bit"
else
    check_yes 0 "the concurrent runs left two cache files to plant temp files beside"
fi
rm -rf "${CC}"

echo ""
echo "(N) a cache file holds the canonical vertex order: its load renumbers no cell"
# PS-DTFE renumbers every cell canonically before the vertex densities and the save, so a cache file holds that order
# and the hit writes the uncached run's grids byte for byte.
OCACHE="${CACHE}-order"; rm -rf "${OCACHE}"; mkdir -p "${OCACHE}"
ORD=(--grid "${GRID}" --periodic --partition 1 --max-concurrent 1 --mpc-unit 1 --verbose 3
     --field density velocity density_a velocity_a)
run_ord() {   # run_ord <outroot> <extra args...>; the log beside the outputs
    local out="$1"; shift
    rm -f "${out}".*
    "${BIN}" "${SNAP}" "${out}" "${ORD[@]}" "$@" > "${out}.log" 2>&1 || true
}
# the count of '(the canonical vertex order renumbered N cells)' in a log (--verbose 3), empty when there is none
renumbered() {
    { sed -n 's/.*(the canonical vertex order renumbered \([0-9][0-9]*\) cells).*/\1/p' "$1" | head -1; } || true
}
grids_same() {   # every output of run a (not its log) byte-identical in run b, and at least one
    local n=0 f
    for f in "$1".*; do
        case "$f" in *.log|*.runlog*|*.part.*) continue ;; esac
        cmp -s "$f" "$2${f#"$1"}" || { note "differs: ${f#"$1"}"; return 1; }
        n=$((n+1))
    done
    [ "${n}" -gt 0 ]
}
run_ord "${TMP}/ptc_ord_ref"
run_ord "${TMP}/ptc_ord_w" --tessellation-cache "${OCACHE}"
run_ord "${TMP}/ptc_ord_hit" --tessellation-cache "${OCACHE}"
nl=$(grep -c "Reused the cached" "${TMP}/ptc_ord_hit.log" || true)
rh="$(renumbered "${TMP}/ptc_ord_hit.log")"
ok=0
if [ "${nl}" -ge 1 ] && { [ -z "${rh}" ] || [ "${rh}" = 0 ]; } && grids_same "${TMP}/ptc_ord_ref" "${TMP}/ptc_ord_hit"; then
    ok=1
fi
check_yes "${ok}" "a cache file loads (${nl} hit) with ${rh:-no} cells renumbered, the grids byte for byte"
rw="$(renumbered "${TMP}/ptc_ord_w.log")"
lo="$({ grep -n 'canonical vertex order)' "${TMP}/ptc_ord_w.log" | head -1 | cut -d: -f1; } || true)"
ls="$({ grep -n 'tessellation save (--tessellation-cache)' "${TMP}/ptc_ord_w.log" | head -1 | cut -d: -f1; } || true)"
ok=0
if [ -n "${rw}" ] && [ -n "${lo}" ] && [ -n "${ls}" ] && [ "${lo}" -lt "${ls}" ]; then ok=1; fi
check_yes "${ok}" "the run that writes the cache orders its cells (${rw:-no line} renumbered) before it saves (log lines ${lo:-?} < ${ls:-?})"
rm -rf "${OCACHE}"

echo ""
echo "------------------------------------------------------------"
if [ "${fails}" -eq 0 ]; then
    echo "ps_tessellation_cache_check: ALL PASS"
    rm -rf "${CACHE}" "${CACHE}-part"
    exit 0
fi
echo "ps_tessellation_cache_check: ${fails} FAILURE(S)"
exit 1
