#!/usr/bin/env bash
# The G-tuner baseline of docs/cleanup-plan.md (P0.3 step 3): a FIXED list of 91 '--auto-tune-report'
# configurations, one line each, "<label><TAB><the AUTO-TUNE-REPORT line>". Run it with the reference binaries once
# and keep the set with them; rerun it with the binaries of a change and diff the two: only the configurations the
# change targets may differ.
#
# usage: tests/dev/tune_report_set.sh <binary dir> <out file> [python with numpy + h5py]
#   <binary dir> holds DTFE, PS-DTFE, DTFE-double, PS-DTFE-double (a missing one: its lines say MISSING)
#
# What is pinned, so that the lines depend on the binaries and the flags only:
#   - the inputs: synthetic snapshots made here with the generator's fixed seed (42);
#   - the machine: DTFE_RAM_GB simulates its RAM (no live free-memory term), OMP_NUM_THREADS the threads;
#   - DTFE_AUTO_MINN=1000: these small snapshots are tuned as the large ones are.
# hardware_concurrency() (the point-evaluation blocks) is the one machine term left: compare on the same machine.
set -u
BIN="$(cd "${1:?binary dir}" && pwd)" || exit 2
OUT="${2:?out file}"; mkdir -p "$(dirname "$OUT")" && OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
PY="${3:-${PY:-python3}}"
THREADS="${TUNE_SET_THREADS:-10}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2   # the repository root: tests/generate_ps_test_data.py
T="$(mktemp -d "${TMPDIR:-/tmp}/tune-set.XXXXXX")"
PAN64="$T/pancake64.hdf5"; PAN128="$T/pancake128.hdf5"; CW64="$T/crossed64.hdf5"
"$PY" tests/generate_ps_test_data.py --out "$PAN64" --n 64 --box 100.0 --amplitude-factor 1.8 >/dev/null || exit 3
"$PY" tests/generate_ps_test_data.py --out "$PAN128" --n 128 --box 100.0 --amplitude-factor 1.8 >/dev/null || exit 3
"$PY" tests/generate_ps_test_data.py --out "$CW64" --n 64 --box 100.0 --crossed-waves --amplitude-factor 3.0 \
    >/dev/null || exit 3

PS_COMMON=(--input 105 --MpcUnit 1 --verbose 2 --periodic --avg-subsamples 1)
DT_COMMON=(--periodic --input 105 --MpcUnit 1 --padding 10)
PROD=(--field density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a --ps-volume-weighted)
DENS=(--field density_a)

: > "$OUT"
n=0
one() {  # one <label> <ram GB> <binary name> <snapshot> <args...>
    local label="$1" ram="$2" b="$BIN/$3" snap="$4"; shift 4
    n=$((n + 1))
    if [ ! -x "$b" ]; then printf '%s\tMISSING %s\n' "$label" "$3" >> "$OUT"; return; fi
    local line
    line="$(env OMP_NUM_THREADS="$THREADS" DTFE_RAM_GB="$ram" DTFE_AUTO_MINN=1000 ${EXTRA_ENV:-} \
        "$b" "$snap" "$T/out" "$@" --auto-tune-report 2>&1 | grep -m1 '^AUTO-TUNE-REPORT ')"
    printf '%s\t%s\n' "$label" "${line:-NO REPORT LINE}" >> "$OUT"
}

# ---- PS-DTFE: RAM x grid x fields x deposit x GPU
for ram in 8 64; do for g in 64 256 1024; do for f in dens prod; do for dep in sampled exact; do for gpu in cpu gpu; do
    a=(--grid "$g")
    [ "$f" = prod ] && a+=("${PROD[@]}") || a+=("${DENS[@]}")
    [ "$dep" = exact ] && a+=(--ps-exact-deposit)
    [ "$gpu" = gpu ] && a+=(--ps-gpu)
    one "ps ram=$ram grid=$g $f $dep $gpu" "$ram" PS-DTFE "$PAN64" "${PS_COMMON[@]}" "${a[@]}"
done; done; done; done; done
# ---- PS-DTFE: hand-set split, window, sample points, a larger N, the double build
one "ps hand-set 2x2x2 mc=3" 64 PS-DTFE "$PAN64" "${PS_COMMON[@]}" --grid 256 "${DENS[@]}" --partition 2 2 2 --max-concurrent 3
one "ps window exact grid=1024" 64 PS-DTFE "$PAN64" "${PS_COMMON[@]}" --grid 1024 "${DENS[@]}" --ps-exact-deposit \
    --ps-window 25 75 25 75 50 56.25
one "ps window exact gpu grid=1024 prod" 64 PS-DTFE "$PAN64" "${PS_COMMON[@]}" --grid 1024 "${PROD[@]}" \
    --ps-exact-deposit --ps-gpu --ps-window 25 75 25 75 50 56.25
EXTRA_ENV="DTFE_AUTO_PTS_N=4000000 DTFE_AUTO_PTS_VELGRAD=1" \
    one "ps sample-points 4M velgrad" 64 PS-DTFE "$PAN64" "${PS_COMMON[@]}" --grid 256 --field density_a velocity_a
for ram in 8 64; do for g in 256 1024; do
    one "ps N=128 ram=$ram grid=$g prod" "$ram" PS-DTFE "$PAN128" "${PS_COMMON[@]}" --grid "$g" "${PROD[@]}"
    one "ps N=128 ram=$ram grid=$g dens exact gpu" "$ram" PS-DTFE "$PAN128" "${PS_COMMON[@]}" --grid "$g" "${DENS[@]}" \
        --ps-exact-deposit --ps-gpu
done; done
for g in 256 1024; do
    one "ps-double ram=64 grid=$g prod" 64 PS-DTFE-double "$PAN64" "${PS_COMMON[@]}" --grid "$g" "${PROD[@]}"
    one "ps-double ram=64 grid=$g dens exact" 64 PS-DTFE-double "$PAN64" "${PS_COMMON[@]}" --grid "$g" "${DENS[@]}" --ps-exact-deposit
done

# ---- standard DTFE: RAM x grid x exact x GPU
for ram in 8 64; do for g in 64 256 1024; do for dep in plain exact; do for gpu in cpu gpu; do
    a=(--grid "$g" --field density_a velocity_a)
    [ "$dep" = exact ] && a+=(--exact-average)
    [ "$gpu" = gpu ] && a+=(--gpu)
    one "dtfe ram=$ram grid=$g $dep $gpu" "$ram" DTFE "$CW64" "${DT_COMMON[@]}" "${a[@]}"
done; done; done; done
one "dtfe hand-set 2x2x2 mc=3" 64 DTFE "$CW64" "${DT_COMMON[@]}" --grid 256 --field density_a --partition 2 2 2 --max-concurrent 3
for g in 256 1024; do
    one "dtfe-double ram=64 grid=$g exact" 64 DTFE-double "$CW64" "${DT_COMMON[@]}" --grid "$g" --field density_a --exact-average
done

echo "$n configurations -> $OUT ($(grep -c $'\tAUTO-TUNE-REPORT ' "$OUT") with a report line)"
rm -rf "$T"
