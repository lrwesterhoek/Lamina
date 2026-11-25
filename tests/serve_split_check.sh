#!/usr/bin/env bash
# The composite server's partition split (auto_tune.h autoTuneServe) when '--serve' runs out of memory and no '--partition' is given.
#   A nothing cached: ~8 partitions in memory   B complete split of this snapshot reused   C another snapshot's split ignored
#   D a cached split too coarse is not reused   E a split cached under another partition padding is ignored
# Usage: tests/serve_split_check.sh [--no-build]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"
BIN="${PS_BIN}"
precision_require "${BIN}"
if [ "${1:-}" != "--no-build" ] && ! precision_double; then
    make PS-DTFE $(cat o_ps/.build_mode 2>/dev/null || true) >/dev/null
fi
TMP="${DTFE_TEST_TMP}/serve_split"; rm -rf "$TMP"; mkdir -p "$TMP"
echo "============================================================"
echo " composite server split check"
echo "============================================================"
"$PY" tests/generate_ps_test_data.py --out "$TMP/a.hdf5" --n 40 --box 100 >/dev/null
"$PY" tests/generate_ps_test_data.py --out "$TMP/b.hdf5" --n 40 --box 100 --seed 7 >/dev/null
COMMON=( --periodic --mpc-unit 1 )
seed() {   # seed <cache> <snapshot> <n> [args]: a grid run with --partition n n n writes that split's partitions
    "$BIN" "$2" "$TMP/g" --grid 16 "${COMMON[@]}" --field density --partition "$3" "$3" "$3" \
        --tessellation-cache "$1" "${@:4}" > "$TMP/seed.log" 2>&1
}
serve() {  # serve <cache>: the auto split of a.hdf5's server; prints "<n> <reason> | <partitions built>"
    DTFE_MEM_BUDGET_GB=0.075 "$BIN" "$TMP/a.hdf5" "$TMP/srv" "${COMMON[@]}" --serve --tessellation-cache "$1" \
        < /dev/null > "$TMP/srv.log" 2>&1
    local line; line="$(grep -a -F 'AUTO-TUNE (serve)' "$TMP/srv.log")"
    local n; n="$(sed -n 's/.*composite of \([0-9]*\)^3 partitions.*/\1/p' <<<"$line")"
    local why; why="$(sed -n 's/.*partitions (\([^;]*\);.*/\1/p' <<<"$line")"
    echo "$n|$why|$(grep -a -c 'built,' "$TMP/srv.log")"
}
fails=0
# check_yes <1 = pass> <text> (it counts a failure in fails): tests/lib.sh

mkdir -p "$TMP/c_empty"
IFS='|' read -r nA whyA builtA <<<"$(serve "$TMP/c_empty")"
check_yes "$([ "${nA:-0}" -ge 3 ] && [[ "$whyA" == nothing\ cached* ]] && echo 1)" \
      "A nothing cached: a split of ${nA}^3 (${whyA}), ${builtA} partitions built"

mkdir -p "$TMP/c_same"; seed "$TMP/c_same" "$TMP/a.hdf5" 3
IFS='|' read -r nB whyB builtB <<<"$(serve "$TMP/c_same")"
check_yes "$([ "$nB" = 3 ] && [ "$builtB" = 0 ] && [[ "$whyB" == *fully\ in\ the\ tessellation\ cache* ]] && echo 1)" \
      "B this snapshot's 3^3 set cached: reused (${nB}^3, ${builtB} partitions built)"

mkdir -p "$TMP/c_other"; seed "$TMP/c_other" "$TMP/b.hdf5" 3
IFS='|' read -r nC whyC builtC <<<"$(serve "$TMP/c_other")"
check_yes "$([ "$nC" = "$nA" ] && [[ "$whyC" == nothing\ cached* ]] && echo 1)" \
      "C another snapshot's 3^3 set: ignored (${nC}^3, ${whyC})"

mkdir -p "$TMP/c_coarse"; seed "$TMP/c_coarse" "$TMP/a.hdf5" 2
IFS='|' read -r nD whyD builtD <<<"$(serve "$TMP/c_coarse")"
check_yes "$([ "$nD" = "$nA" ] && [[ "$whyD" == *too\ coarse* ]] && echo 1)" \
      "D this snapshot's 2^3 set, too coarse for the budget: not reused (${nD}^3, ${whyD})"

mkdir -p "$TMP/c_pad0"; seed "$TMP/c_pad0" "$TMP/a.hdf5" 3 --ps-lagrangian-padding 0
IFS='|' read -r nE whyE builtE <<<"$(serve "$TMP/c_pad0")"
check_yes "$([ "$nE" = "$nA" ] && [[ "$whyE" == nothing\ cached* ]] && echo 1)" \
      "E this snapshot's 3^3 set under the Eulerian padding ('--ps-lagrangian-padding 0'): ignored (${nE}^3, ${whyE})"

echo "------------------------------------------------------------"
if [ "$fails" -gt 0 ]; then echo "RESULT: FAIL ($fails)"; exit 1; fi
echo "RESULT: PASS  (the composite server reuses a cached split of the same snapshot, else holds ~8 partitions)"
