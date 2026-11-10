#!/usr/bin/env bash
# Byte-for-byte comparison of the outputs of two sets of binaries: a reference set against this checkout's own
# (./DTFE, ./PS-DTFE, their -double and -2d builds). The gate for a change that must not move any output (a
# refactor, a cleanup): 17 runs on two synthetic crossed-wave snapshots -- standard DTFE and PS-DTFE, single, double
# and 2D, CPU and GPU, one domain and '--partition 2 2 2', every derived and web field, and the CPU exact deposits
# ('--exact-average', '--ps-exact-deposit': the GPU's exact clipping is float32 and compiled with fast math, so it
# is compared with a tolerance elsewhere) -- and every output file compared with cmp. Logs are kept but not
# compared (they carry times and memory).
#
# The reference set is any folder holding the six binaries, e.g. a copy taken before the change:
#     mkdir -p /tmp/ref && cp DTFE DTFE-double DTFE-2d PS-DTFE PS-DTFE-double PS-DTFE-2d /tmp/ref/
# (build all three pairs first: make DTFE PS-DTFE METAL=1; ... METAL=1 DOUBLE=1; ... DIM=2). The GPU runs need a
# METAL=1 build on both sides; a run that fell back to the CPU is listed at the end.
#
# --gpu-exact adds 6 runs of the GPU's exact deposits ('--ps-exact-deposit --ps-gpu', '--exact-average --gpu'; single
# and double, one domain and '--partition 2 2 2'). Those outputs are float32 and fast math, so they are comparable
# only between two builds of the SAME kernel source: use it for a host-only change (the GPU sums every grid in
# fixed point, so a run repeats to the byte). A kernel change is checked by metal/validate_deposit and the GPU
# suites' tolerances instead.
# Every run pins its plan ('--partition 1' or '--partition 2 2 2 --max-concurrent 2'): an auto-tuner change (plan
# P8) moves a plan by design, and a different split moves the grids at rounding, so an auto-planned run would show
# differences the change under test did not make. The tuner's plans have their own gate (tests/dev/tune_report_set.sh).
# The runs spell PS-DTFE's options the old way ('--ps-gpu', '--ps-exact-deposit'): both sides run the same command,
# and a reference set from before the one flag scheme knows only those; the current binaries take them as deprecated
# aliases (a 'warning: ... is deprecated' line in the log, the same outputs). The same holds for '--exact-average',
# which both binaries took until '--exact' replaced it on 2026-11-09.
# Usage: tests/dev/bytecmp.sh [--gpu-exact] <reference binaries dir> <output dir>   (exit 0 = every file identical)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXACT=0; [ "${1:-}" = "--gpu-exact" ] && { GPU_EXACT=1; shift; }
[ $# -eq 2 ] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 2; }
REF="$(cd "$1" && pwd)" || { echo "no reference folder $1"; exit 2; }
OUT="$2"; mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
cd "$ROOT" || exit 2
source tests/lib.sh   # PY: a python with a working h5py
for b in DTFE DTFE-double DTFE-2d PS-DTFE PS-DTFE-double PS-DTFE-2d; do
    [ -x "$REF/$b" ] || { echo "missing reference binary $REF/$b"; exit 2; }
    [ -x "$ROOT/$b" ] || { echo "missing binary $ROOT/$b (build all three pairs)"; exit 2; }
done
"$PY" tests/generate_ps_test_data.py --out "$OUT/w48.hdf5" --n 48 --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 --seed 7 --crossed-waves >/dev/null
"$PY" tests/generate_ps_test_data.py --out "$OUT/w2d.hdf5" --n 64 --box 100 --amplitude-factor 1.8 --jitter-frac 0.02 --seed 7 --crossed-waves --dim 2 >/dev/null
# the velocity gradient's route: the standard binary's default became the face gradient on 2026-11-09 (D13), which moves
# every standard derivative file against a reference set from before; DTFE_FACE_GRADIENT=0 keeps the deposit's own
# gradient on a binary that knows the switch and is ignored by an older one, so both sides compare the same route
export DTFE_FACE_GRADIENT=0
# the '_a' average: the exact one became both binaries' default on 2026-11-09, which moves every unflagged '_a' run
# against a reference set from before; DTFE_EXACT_AVERAGE=0 keeps those runs sampled on a binary that reads the pin
# (a binary from before reads no such variable, and samples them anyway), and the '--exact-average' /
# '--ps-exact-deposit' runs stay exact on both sides, since a flag beats the pin
export DTFE_EXACT_AVERAGE=0
C=(--periodic --input 105 --MpcUnit 1 --verbose 1)
F3=(--field density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a)
F2=(--field density_a velocity_a gradient_a divergence_a vorticity_a)
runs=(
  "dt:DTFE:w48:--grid 64 --partition 1 ${F3[*]}"
  "dtp:DTFE:w48:--grid 64 --partition 2 2 2 --max-concurrent 2 ${F3[*]}"
  "dtg:DTFE:w48:--grid 64 --partition 1 --gpu ${F3[*]}"
  "dtd:DTFE-double:w48:--grid 64 --partition 1 ${F3[*]}"
  "ps:PS-DTFE:w48:--grid 64 --partition 1 ${F3[*]} --ps-caustics"
  "psp:PS-DTFE:w48:--grid 64 --partition 2 2 2 --max-concurrent 2 ${F3[*]}"
  "psg:PS-DTFE:w48:--grid 64 --partition 1 --ps-gpu ${F3[*]} --ps-caustics"
  "psd:PS-DTFE-double:w48:--grid 64 --partition 1 ${F3[*]}"
  "dt2:DTFE-2d:w2d:--grid 128 --partition 1 ${F2[*]}"
  "ps2:PS-DTFE-2d:w2d:--grid 128 --partition 1 ${F2[*]}"
  "dtx:DTFE:w48:--grid 64 --partition 1 ${F3[*]} --exact-average"
  "dtxd:DTFE-double:w48:--grid 64 --partition 1 ${F3[*]} --exact-average"
  "psx:PS-DTFE:w48:--grid 64 --partition 1 ${F3[*]} dispersion_a --ps-exact-deposit --ps-caustics --ps-vertex-mass --ps-volume-weighted"
  "psxp:PS-DTFE:w48:--grid 64 --partition 2 2 2 --max-concurrent 2 ${F3[*]} dispersion_a --ps-exact-deposit"
  "psxd:PS-DTFE-double:w48:--grid 64 --partition 1 ${F3[*]} dispersion_a --ps-exact-deposit"
  "dtx2:DTFE-2d:w2d:--grid 128 --partition 1 ${F2[*]} --exact-average"
  "psx2:PS-DTFE-2d:w2d:--grid 128 --partition 1 ${F2[*]} --ps-exact-deposit"
)
# the GPU exact deposits (--gpu-exact); each tag ends in 'g', so the fallback check below sees its log
[ "$GPU_EXACT" = 1 ] && runs+=(
  "dtxg:DTFE:w48:--grid 64 --partition 1 ${F3[*]} --exact-average --gpu"
  "dtxpg:DTFE:w48:--grid 64 --partition 2 2 2 --max-concurrent 2 ${F3[*]} --exact-average --gpu"
  "dtxdg:DTFE-double:w48:--grid 64 --partition 1 ${F3[*]} --exact-average --gpu"
  "psxg:PS-DTFE:w48:--grid 64 --partition 1 ${F3[*]} dispersion_a --ps-exact-deposit --ps-caustics --ps-gpu"
  "psxpg:PS-DTFE:w48:--grid 64 --partition 2 2 2 --max-concurrent 2 ${F3[*]} dispersion_a --ps-exact-deposit --ps-gpu"
  "psxdg:PS-DTFE-double:w48:--grid 64 --partition 1 ${F3[*]} dispersion_a --ps-exact-deposit --ps-gpu"
)
n=0; d=0
for r in "${runs[@]}"; do
  IFS=: read -r tag bin snap args <<< "$r"
  for side in ref new; do
    B="$ROOT/$bin"; [ $side = ref ] && B="$REF/$bin"
    mkdir -p "$OUT/$side"
    # shellcheck disable=SC2086   # $args: the run's options, word-split on purpose
    "$B" "$OUT/$snap.hdf5" "$OUT/$side/$tag" "${C[@]}" $args > "$OUT/$side/$tag.log" 2>&1 || echo "  RUN FAILED: $side $tag (rc $?)"
  done
  for f in "$OUT/ref/$tag".*; do
    case "$f" in *.log) continue;; esac
    g="$OUT/new/${f#"$OUT/ref/"}"; n=$((n+1))
    cmp -s "$f" "$g" || { d=$((d+1)); echo "  DIFFERS: ${f#"$OUT/ref/"}"; }
  done
  [ "$(ls "$OUT/ref/$tag".* | wc -l)" = "$(ls "$OUT/new/$tag".* | wc -l)" ] || { d=$((d+1)); echo "  FILE SETS DIFFER: $tag"; }
done
grep -l -E 'using the CPU deposit|deposit unavailable|gave up|falling back' "$OUT"/new/*g.log "$OUT"/ref/*g.log 2>/dev/null | sed 's/^/  GPU FALLBACK in /'
echo "BYTECMP: $n output files compared, $d differ"
[ "$d" = 0 ]
