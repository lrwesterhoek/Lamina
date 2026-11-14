#!/usr/bin/env bash
# PS-DTFE's '--face-gradient' (docs/cleanup-plan.md P10.3): the gradient of the volume-weighted MEAN velocity from exact
# face sums -- each cell's lower and upper face per axis carry the integrals of S (the sum of the streams' velocities)
# and of N (their count), and the gradient is the difference of the face-averaged mean velocities, (S_up/N_up -
# S_lo/N_lo) / dx -- instead of the stream average of each stream's own gradient, which the run writes beside it as
# '.a_velDivStreams'. On Zel'dovich pancakes (tests/generate_ps_test_data.py, 32^3 particles, box 100 Mpc, 20% jitter,
# 32^3 grid, --exact --ps-volume-weighted, density_a velocity_a gradient_a divergence_a shear_a vorticity_a):
#   P1  single stream (amplitude factor 0.6): the face gradient equals the stream average to 1e-5 of the divergence's
#       rms (measured 9.5e-7), and '.a_velDivStreams' is byte-identical to the divergence of a '--per-tet-gradient' run
#   P2  three streams across two folds (amplitude factor 2): the periodic box mean of the mean flow's divergence is 0 to
#       1e-6 of its rms (measured 5e-11; Gauss: the face form telescopes), while the stream average's is not (+0.98 of
#       its rms: the streams expand after shell crossing); in the cells whose stream count is an integer, and their six
#       neighbours' too (no caustic on a face), the two agree to 1e-4 of the rms (measured 1.5e-6)
#   P3  the folds carry the mean flow's convergence: the two fold planes' mean divergence is below -5 x the
#       single-stream region's mean and the two agree to 1e-3 of each other (the pancake is symmetric)
#   P4  determinism: 10 deposit threads and slab mode (DTFE_DEPOSIT_SLABS=1, 6 threads) byte-identical to one thread
#       (the double build's stream average to 1e-12: its float sums are per thread count, as every double-build moment)
#   P5  '--partition 2 2 2': within 1e-5 of the rms of one domain (measured 9e-7), box mean below 1e-6 of the rms
#   P6  '--grid-window' (one z cell across the box): the full run's z = 16 slice, byte-identical in the single build
#       (the double build shows the window's own fixed-point unit: within 1e-9 of the rms, measured 2e-11)
#   P7  refusals at option parsing (exit 1): with '--sampled' (the exact deposit is the default for these '_a' fields,
#       so only the sampled one asked for conflicts), without '--ps-volume-weighted'; both route flags at once
#       ('--face-gradient --per-tet-gradient'); DTFE_FACE_GRADIENT=1 beside '--sampled' (read as the flag)
#   P11 (a GPU build) '--gpu': the GPU exact deposit with the face sums from the CPU walk beside it (docs/cleanup-plan.md
#       D17): the mean flow's divergence and the mass flux divergence byte-identical to the CPU run's (the same face
#       words, the same units), the density within the GPU deposit's float tolerance, the stream average within 5e-2
#       of its rms (the device's float per-stream gradient), and the run neither fell back nor skipped the walk
#   P12 (a GPU build, but for the lattice's CPU check) the face walk keeps the tetrahedra the CPU deposit loop keeps
#       (docs/cleanup-plan.md P11.7): a CPU and a '--gpu' run write the mean flow's grids and the mass flux divergence
#       byte for byte in a one-cell '--grid-window', in a non-periodic '--box', under the NaN test hook
#       (DTFE_TEST_NAN_SHARE=50, single build, with the same number of face shares left out) and for a real NaN vertex
#       mass (one particle's mass NaN under '--ps-vertex-mass', '--mean-density' given; the same). In a '--box' whose
#       upper x face is a particle plane of a lattice without jitter (tetrahedra outside the box with a face in that
#       plane, which own the box's section there: both devices deposit it, psOwnsUpperBoundarySection), both runs on
#       one domain ('--partition 1 1 1': the two devices cannot plan different splits), the mean flow's grids and the
#       mass flux divergence are byte-identical in every layer. On every build (a CPU-only build too: the CPU run
#       alone) the last x layer's upper face carries that section: the flow is along x alone, so where that face had
#       no stream '.a_velDiv' was the stream average '.a_velDivStreams' to the byte; now it is the face value in at
#       least half as many of that layer's cells as in the layer below (whose faces are all covered), and as a
#       single-stream cell must, within 1e-5 of the rms of the stream average. In the single build the bounds a face
#       pass takes over from the GPU extraction equal the bound walk's (DTFE_DEBUG_FX_BOUNDS=1)
#   P13 (every build) that lattice box split 2^3 on the CPU ('--partition 2 2 2 --max-concurrent 2'): the section
#       survives the split (the sub-grid of the partition that holds the tetrahedra beyond the box must hold the last
#       x layer): the last x layer's divergence is the face value in at least half as many cells as on one domain
#       (and P12's way, within 1e-5 of the rms of the stream average), and the mass flux divergence and the mean
#       flow's divergence and shear are within 1e-5 of the one-domain rms in every x layer (P5's tolerance; a lost
#       section would leave the mass flux divergence off by the whole outflow through that face). A failure names
#       the count, or the field, its deviation and the x layers
#   P14 the default (docs/cleanup-plan.md D13; read from the run's header, so the check follows
#       src/user_options.cc kPsFaceGradientDefault): a qualifying run without a flag (--exact --ps-volume-weighted)
#       writes the face run's files byte for byte; DTFE_FACE_GRADIENT=0 writes the '--per-tet-gradient' run's; the
#       face run differs from that one only in the gradient family and its extra files ('.a_massFluxDiv',
#       '.a_velDivStreams' and P15's '.a_velGradStreams', '.a_velShearStreams', '.a_velVortStreams'); a run that does
#       not qualify (the sampled deposit, mass-weighted moments, the halo release) exits 0, keeps the stream average
#       and names the cause in its 'velocity gradient' line
#   P15 the regular part's twins (docs/cleanup-plan.md D12): '.a_velGradStreams', '.a_velShearStreams' and
#       '.a_velVortStreams' are byte-identical to the '.a_velGrad', '.a_velShear', '.a_velVort' of a
#       '--per-tet-gradient' run (one stream and three); a 'shear_a'-only run makes the shear twin alone (the bytes of
#       a '--per-tet-gradient' run's '.a_velShear' at the same fields); the gradient twin's trace is
#       '.a_velDivStreams' byte for byte; on a linear velocity field (a non-periodic clump, v = G x) the twins and
#       the mean flow's gradient are G's own to 4e-5 of max|G| (measured 3.5e-6 single, 1.2e-6 double); with '--gpu'
#       the twins are within 5e-2 of the CPU's (of the gradient twin's rms) and the trace holds; P4 and P9 compare
#       the twins too, as their other grids; P6 holds a window's twins to the window's own fixed-point units (the
#       contract of tests/ps_window_check.sh: a window pass rounds its sums in its own units): 1e-8 of the gradient
#       twin's rms in the single build (measured 7.7e-11, in 2 of the slice's 9216 gradient values, both near zero),
#       1e-9 in the double build (measured 0); dtfelib's load(name, stream_average=True) gives the same grid for a
#       face run and a '--per-tet-gradient' run
#   P16 (a GPU build) the face walk beside the device call (the default; DTFE_FACE_WALK_EARLY=1 means the same, and
#       =0, a developer's off switch, keeps the walk after the device's deposit): in the single build a '--gpu' run
#       says that its face sums were made beside the GPU deposit and writes the grids of the =0 run byte for byte
#       (the mean flow's, the mass flux divergence, the density and the velocity) -- on one domain (the default and
#       =1), in a one-cell '--grid-window' (the mean flow of the CPU run too) and on a 2^3 split with 4 partitions in
#       flight, whose closing line counts the walks: all 8 kept, none thrown away (=0: no such line; the default
#       run's log names no variable); a CPU split under =1 says that no pass took the walk. After a device call that
#       fails (DTFE_GPU_FAULT_AT=1:persist) the walk's sums are thrown away and the CPU deposit makes them: the grids
#       of the same run under =0 byte for byte, the mean flow the CPU run's; on the split no walk is kept and all 8
#       are thrown away (every partition's device call fails under the persistent fault, and each of them had
#       started its walk). A NaN vertex mass leaves out, and counts, the same face shares with the walk beside a
#       device call that succeeds and beside one that fails (once, not twice), and after one under =0; under the NaN
#       test hook the walk stays after the deposit by default, the grids of =0. So it does in the double build, by
#       default and under =1, whose face pass takes its bounds from a bound walk of its own
#   P17 the canonical float face sums (DTFE_FACE_F32=1, docs/cleanup-plan.md P11's GPU face kernel, step 2:
#       face_sweep_f32.h, the GPU face kernel's arithmetic, on the CPU; opt-in): in the single build a CPU run says so
#       in its log, its face grids ('.a_velDiv', '.a_velGrad', '.a_velShear', '.a_velVort', '.a_massFluxDiv') are
#       within 3e-5 of the default run's rms (about ten times the largest value the PASS lines print; the vorticity
#       against the gradient's rms: it is the gradient's antisymmetric part and this pancake flow has almost none,
#       so its own rms is no scale for a rounding difference), every other output file is the default run's byte
#       for byte, 10 deposit threads and slab mode write its grids byte for byte, and a CPU run under DTFE_GPU_FACES=1
#       (which implies the same arithmetic for every CPU face sum) writes all its files byte for byte; with '--gpu' (a
#       GPU build) the face walk beside the device's deposit (the default) and after it (DTFE_FACE_WALK_EARLY=0) write
#       the CPU run's face grids byte for byte. In the double build the switch is ignored with one note line: the
#       default run's files byte for byte
#   P18 (a GPU build) the GPU face kernel (DTFE_GPU_FACES=1, metal/ps_face.metal; opt-in): in the single build a
#       '--gpu' run's face family (P17's five grids) is the CPU DTFE_FACE_F32=1 run's byte for byte and every other
#       file the same '--gpu' run's without the switch (the design's contract (c)) -- on one domain, in P12's lattice
#       box with its upper boundary plane, in a one-cell '--grid-window', and for a real NaN vertex mass (the same
#       face shares left out, and counted, as on the CPU); on a 2^3 split with 4 in flight the CPU split's face grids,
#       the closing line counting 8 face passes made and none fallen back; that split killed after 3 partitions
#       under the kernel and resumed with the CPU walk (DTFE_FACE_F32=1: the checkpoint key's same 'face-sums=f32'
#       line) writes the uninterrupted run's face grids and density; under the NaN test hook the kernel does not take
#       the pass and the CPU walk after the deposit writes the CPU F32 run's face grids. A face pass the fault hook
#       hits (DTFE_GPU_FAULT_AT at the deposit's buffers + 1, read off DTFE_GPU_TIMING=1's kernel lines; up to five
#       tries around it, as the chunk sizes follow the buffers' times): persistent, the face pass gives up and the CPU walk
#       makes the same words; innocent once, the face pass redoes its words from zero on the device. In the double
#       build the switch is ignored with one note line, the GPU run's grids unchanged
#   P19 '--interlace' (two passes, the second on the particles moved by half a cell, of which only the density is
#       kept: main.cpp): P14's default run with it takes the same gradient route (its header's 'velocity gradient'
#       line) and writes every file of the run without it byte for byte -- the face gradient's family, the mass flux
#       divergence and the stream average's twins included -- but '.a_den', which is interlaced
#   P8  dtfelib: FieldSet.face_gradient is True for the run (False with '--per-tet-gradient') and load('divergence',
#       mean_field=True) returns the stored grid itself
#   P9  the face sums ride the checkpoint: a '--partition 2 2 2' run killed after 3 partitions
#       (DTFE_TEST_EXIT_AFTER_PARTITIONS=3, exit 137) and resumed ('--resume') writes the uninterrupted run's grids
#       byte for byte; '--ps-merge fixed-point' (the face sums as integer words across partitions) within 1e-5 of the
#       ordered merge (measured 9e-7), box mean below 1e-6 of the rms; a one-cell window over a 2^3 split equals the
#       full run's slice as P6 does
#   P10 the mass flux divergence '.a_massFluxDiv': periodic box mean 0 to 1e-6 of its rms; in the double build,
#       continuity over a drift of the 3-stream pancake and of crossed waves (24^3, amplitude factor 1.5: the y and z
#       momentum words) (x + v t, 9 times in [-1e-4, 1e-4]): each cell's mass change equals minus its outflow
#       integrated by Simpson's rule, caustic cells included, to 0.05 max / 2e-3 rms of the rms (measured 0.022 /
#       7.2e-4 and 0.028 / 7.1e-4), and 4 -> 8 intervals shrink the rms by at least 2.5 (measured 3.8 and 4.0); the
#       thread, slab-mode and resume checks (P4, P9) compare '.a_massFluxDiv' too
# About 47 s on the Mac in a GPU build (single; 38 s in the double build, where P16 is one run; each measured once;
# both before P17's runs; P18 adds 15 to 23 runs in the single build, one in the double, and P19 two passes of
# P14's 32^3 run: TO MEASURE; P16's three runs more since the walk beside the device call is the default: TO MEASURE).
# Usage: tests/ps_face_gradient_check.sh [--no-build]   (--no-build is accepted and ignored: nothing is built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair
# the gradient route's environment pin: the runs below name their route by flag, and P14 checks the binary's default
unset DTFE_FACE_GRADIENT
unset DTFE_FACE_WALK_EARLY   # (the face walk beside the device call: the default applies; P16 and P17 set it per run)
unset DTFE_FACE_F32 DTFE_GPU_FACES   # (the canonical float face sums: P17 sets them per run)
precision_require "${PS_BIN}"
[ -x "${PS_BIN}" ] || { echo "FAIL: ${PS_BIN} is not built"; exit 1; }
TMP="${DTFE_TEST_TMP}/ps_face_gradient_check"
rm -rf "$TMP"; mkdir -p "$TMP"
OBJ=o_ps; precision_double && OBJ=o_ps_d
GPU=0
[ -n "$(gpu_backend_of "$OBJ")" ] && GPU=1
FAIL=0
pass() { echo "   PASS  $*"; }
fail() { echo "   FAIL  $*"; FAIL=1; }

for amp in 0.6 2.0; do
    "$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$TMP/pan_$amp.hdf5" --n 32 --box 100 \
        --amplitude-factor "$amp" --jitter-frac 0.2 > "$TMP/gen_$amp.log" 2>&1 \
        || { echo "FAIL: generate_ps_test_data.py (see $TMP/gen_$amp.log)"; exit 1; }
done
C=(--grid 32 --periodic --input-format 105 --mpc-unit 1 --field density_a velocity_a gradient_a divergence_a shear_a
   vorticity_a --exact)
# run <name> <input> [env assignments ...] -- [extra args ...]
run() {
    local name="$1" input="$2"; shift 2
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    env ${envs[@]+"${envs[@]}"} "${PS_BIN}" "$TMP/$input.hdf5" "$TMP/$name" "${C[@]}" "$@" > "$TMP/$name.log" 2>&1 \
        || { echo "FAIL: run $name (see $TMP/$name.log)"; exit 1; }
}
echo ">> pancakes at amplitude factors 0.6 (one stream) and 2 (three across two folds), each with both routes"
for amp in 0.6 2.0; do
    run "b$amp" "pan_$amp" OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --ps-volume-weighted --per-tet-gradient
    run "f$amp" "pan_$amp" OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --ps-volume-weighted --face-gradient
done
echo ">> determinism (10 threads, slab mode), --partition 2 2 2, a one-cell --grid-window"
run t10  pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=10 -- --ps-volume-weighted --face-gradient
run slab pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=6 DTFE_DEPOSIT_SLABS=1 -- --ps-volume-weighted --face-gradient
run part pan_2.0 OMP_NUM_THREADS=1 -- --ps-volume-weighted --face-gradient --partition 2 2 2 --max-concurrent 2
run win  pan_2.0 OMP_NUM_THREADS=1 -- --ps-volume-weighted --face-gradient --grid-window 0.01 99.99 0.01 99.99 50.1 53.0
# the regular part's twins (P15): its gradient, shear and vorticity beside the mean flow's
twins_same() { cmp -s "$TMP/$1.a_velGradStreams" "$TMP/$2.a_velGradStreams" \
               && cmp -s "$TMP/$1.a_velShearStreams" "$TMP/$2.a_velShearStreams" \
               && cmp -s "$TMP/$1.a_velVortStreams" "$TMP/$2.a_velVortStreams"; }
same() { cmp -s "$TMP/$1.a_velDiv" "$TMP/$2.a_velDiv" && cmp -s "$TMP/$1.a_velShear" "$TMP/$2.a_velShear" \
         && cmp -s "$TMP/$1.a_velDivStreams" "$TMP/$2.a_velDivStreams" \
         && cmp -s "$TMP/$1.a_massFluxDiv" "$TMP/$2.a_massFluxDiv" && twins_same "$1" "$2"; }
echo ">> a 2^3 split: killed after 3 partitions and resumed; --ps-merge fixed-point; a one-cell window over it"
P8=(--ps-volume-weighted --face-gradient --partition 2 2 2 --max-concurrent 2)
run full8 pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}"
DTFE_TEST_EXIT_AFTER_PARTITIONS=3 OMP_NUM_THREADS=1 "${PS_BIN}" "$TMP/pan_2.0.hdf5" "$TMP/res8" "${C[@]}" "${P8[@]}" \
    > "$TMP/res8a.log" 2>&1
rc_kill=$?
run res8 pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}" --resume
precision_double || run fxm pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}" --ps-merge fixed-point   # (not in the double build)
run winp pan_2.0 OMP_NUM_THREADS=1 -- "${P8[@]}" --grid-window 0.01 99.99 0.01 99.99 50.1 53.0
if [ "$rc_kill" = "137" ] && grep -q -- "--resume: 3 of 8 partitions" "$TMP/res8.log" && same full8 res8 \
   && cmp -s "$TMP/full8.a_vel" "$TMP/res8.a_vel" && cmp -s "$TMP/full8.a_den" "$TMP/res8.a_den"; then
    pass "P9 killed after 3 of 8 partitions and resumed: the uninterrupted run's grids byte for byte"
else
    fail "P9 the resumed face run (kill exit $rc_kill; see $TMP/res8.log)"
fi
# the mean flow's grids come from fixed-point face sums alone: byte-identical in both builds. The stream average
# ('.a_velDivStreams') is a float sum per private copy in the double build (deterministic per thread count only, as
# every double-build moment): byte-identical in the single build, to 1e-12 of the rms in the double (measured 2.5e-15)
meanflow() { cmp -s "$TMP/$1.a_velDiv" "$TMP/$2.a_velDiv" && cmp -s "$TMP/$1.a_velShear" "$TMP/$2.a_velShear" \
             && cmp -s "$TMP/$1.a_massFluxDiv" "$TMP/$2.a_massFluxDiv"; }
if precision_double; then
    # (the regular part's twins too, P15: the same float sums)
    streams_ok=$("$PY" -c "import numpy as np
ok = 1
for s in ('a_velDivStreams', 'a_velGradStreams', 'a_velShearStreams', 'a_velVortStreams'):
    a = np.fromfile('$TMP/f2.0.' + s); b = np.fromfile('$TMP/t10.' + s); c = np.fromfile('$TMP/slab.' + s)
    r = np.sqrt((a**2).mean()) if a.size else 0.0
    ok = ok and a.size > 0 and a.size == b.size == c.size and max(abs(a-b).max(), abs(a-c).max()) <= 1e-12*(r or 1.0)
print(int(ok))")
else
    streams_ok=0; same f2.0 t10 && same f2.0 slab && streams_ok=1
fi
if meanflow f2.0 t10 && meanflow f2.0 slab && [ "$streams_ok" = "1" ]; then
    pass "P4 10 threads and slab mode: the mean flow byte-identical to one thread (the stream average too, or to 1e-12 in double)"
else fail "P4 the thread count or slab mode changes the face gradient"; fi
if cmp -s "$TMP/f0.6.a_velDivStreams" "$TMP/b0.6.a_velDiv"; then
    pass "P1 '.a_velDivStreams' is the divergence a '--per-tet-gradient' run stores (byte-identical)"
else fail "P1 '.a_velDivStreams' differs from the divergence of a '--per-tet-gradient' run"; fi
# P15 the twins are what a '--per-tet-gradient' run stores under the plain names, one stream and three
t_ok=1
for amp in 0.6 2.0; do
    { cmp -s "$TMP/f$amp.a_velGradStreams" "$TMP/b$amp.a_velGrad" \
      && cmp -s "$TMP/f$amp.a_velShearStreams" "$TMP/b$amp.a_velShear" \
      && cmp -s "$TMP/f$amp.a_velVortStreams" "$TMP/b$amp.a_velVort"; } || t_ok=0
done
if [ "$t_ok" = "1" ]; then
    pass "P15 the gradient, shear and vorticity twins: a '--per-tet-gradient' run's grids byte for byte (1, 3 streams)"
else fail "P15 a regular-part twin differs from the grid a '--per-tet-gradient' run stores"; fi
# ... and only the twins of the fields written: a 'shear_a'-only run, against a '--per-tet-gradient' run at the same
# fields (one '--field' per command line: both spelled out)
for k in fsh bsh; do
    route=--face-gradient; [ "$k" = bsh ] && route=--per-tet-gradient
    OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 "${PS_BIN}" "$TMP/pan_2.0.hdf5" "$TMP/$k" --grid 32 --periodic \
        --input-format 105 --mpc-unit 1 --field density_a shear_a --exact --ps-volume-weighted "$route" \
        > "$TMP/$k.log" 2>&1 || { echo "FAIL: run $k (see $TMP/$k.log)"; exit 1; }
done
if [ -f "$TMP/fsh.a_velDivStreams" ] && [ ! -e "$TMP/fsh.a_velGradStreams" ] && [ ! -e "$TMP/fsh.a_velVortStreams" ] \
   && cmp -s "$TMP/fsh.a_velShearStreams" "$TMP/bsh.a_velShear"; then
    pass "P15 'shear_a' alone: the shear twin (a '--per-tet-gradient' run's shear), the divergence one, no other"
else fail "P15 the twins of a 'shear_a'-only run (see $TMP/fsh.log)"; fi

"$PY" - "$TMP" "${DTFE_TEST_REAL}" <<'PY' || FAIL=1
import sys
import numpy as np
tmp, dt = sys.argv[1], sys.argv[2]
n = 32
rd = lambda name: np.fromfile(f"{tmp}/{name}", dtype=dt).astype(np.float64)
ok = True
def out(good, what):
    global ok
    print(("   PASS  " if good else "   FAIL  ") + what)
    ok = ok and good
# P1
b, f = rd("b0.6.a_velDiv"), rd("f0.6.a_velDiv")
rms = np.sqrt((b**2).mean())
e = np.abs(f - b).max() / rms
out(e <= 1e-5, f"P1 one stream: the face gradient == the stream average ({e:.2g} of the rms)")
# P2
b, f, s = rd("b2.0.a_velDiv"), rd("f2.0.a_velDiv"), rd("f2.0.a_streams")
rmsb, rmsf = np.sqrt((b**2).mean()), np.sqrt((f**2).mean())
mf, mb = abs(f.mean()) / rmsf, b.mean() / rmsb
out(mf <= 1e-6 and mb > 0.5, f"P2 periodic box mean: mean flow {mf:.2g} of its rms, stream average {mb:+.2f} of its rms")
s3 = s.reshape(n, n, n)
integ = np.abs(s3 - np.round(s3)) < 1e-5
calm = integ.copy()
for ax in range(3):
    for sh in (-1, 1):
        calm &= np.roll(integ, sh, axis=ax)
e = (np.abs(f - b).reshape(n, n, n)[calm].max() / rmsb) if calm.any() else 1.0
out(calm.mean() > 0.5 and e <= 1e-4, f"P2 cells without a caustic on a face ({calm.mean():.2f} of them): the two agree to {e:.2g} of the rms")
# P3: the fold planes (x layers with a non-integer stream count), against the single-stream layers
fx, sx = f.reshape(n, n, n).mean(axis=(1, 2)), s3.mean(axis=(1, 2))
folds = np.where(np.abs(sx - np.round(sx)) > 1e-3)[0]
single = np.abs(sx - 1.0) < 1e-6
good = len(folds) == 2 and single.any()
if good:
    a, c = fx[folds]
    ref = abs(fx[single].mean())
    good = a < -5 * ref and c < -5 * ref and abs(a - c) <= 1e-3 * abs(a)
    what = f"P3 the two folds (x = {folds[0]}, {folds[1]}): mean divergence {a:.1f}, {c:.1f} against the single-stream {fx[single].mean():+.1f}"
else:
    what = f"P3 expected two fold layers and a single-stream region (found {len(folds)} folds)"
out(good, what)
# P15 the gradient twin's trace is the divergence twin, byte for byte (the same three adds, in order, in the grids' own
# precision: Quantities::psFaceGradientCell)
g = np.fromfile(f"{tmp}/f2.0.a_velGradStreams", dtype=dt).reshape(-1, 9)
out(np.array_equal((g[:, 0] + g[:, 4]) + g[:, 8], np.fromfile(f"{tmp}/f2.0.a_velDivStreams", dtype=dt)),
    "P15 the trace of '.a_velGradStreams' is '.a_velDivStreams' (byte-identical)")
# P5
p = rd("part.a_velDiv")
e, m = np.abs(p - f).max() / rmsf, abs(p.mean()) / rmsf
out(e <= 1e-5 and m <= 1e-6, f"P5 --partition 2 2 2: {e:.2g} of the rms from one domain, box mean {m:.2g}")
# P6
w = rd("win.a_velDiv").reshape(n, n, -1)
ws = rd("win.a_velDivStreams").reshape(n, n, -1)
full, fulls = f.reshape(n, n, n), rd("f2.0.a_velDivStreams").reshape(n, n, n)
# the window's face words take the window's own fixed-point unit (as every fixed-point grid of a window run does): the
# single build's float output rounds that away (byte-identical); the double build shows it (measured 2e-11 of the rms)
e = np.abs(w[:, :, 0] - full[:, :, 16]).max() / rmsf if w.shape[2] == 1 else 1.0
# (the regular part's twins, P15: a window's stream-average sums round in the window's own units too, and the single
# build's float output shows that in a few values near zero: 1e-8 of the gradient twin's rms, measured 7.7e-11 in 2 of
# the slice's 9216 gradient values; the double build 1e-9, measured 0. The divergence twin: the full run's bytes)
rmsg = np.sqrt((rd("f2.0.a_velGradStreams") ** 2).mean())
et = max(np.abs(rd("win." + s).reshape(n, n, -1, c)[:, :, 0] - rd("f2.0." + s).reshape(n, n, n, c)[:, :, 16]).max()
         for s, c in (("a_velGradStreams", 9), ("a_velShearStreams", 5), ("a_velVortStreams", 3))) / rmsg \
     if w.shape[2] == 1 else 1.0
tw = et <= (1e-8 if dt == "float32" else 1e-9)
good = w.shape[2] == 1 and np.array_equal(ws[:, :, 0], fulls[:, :, 16]) and tw and \
       (np.array_equal(w[:, :, 0], full[:, :, 16]) if dt == "float32" else e <= 1e-9)
out(good, f"P6 a one-cell --grid-window vs the full run's z = 16 slice: {e:.2g} of the rms (float32: byte-identical)"
          f"; the regular part's twins {et:.2g} of the gradient twin's rms")
# P9: the fixed-point merge, and a window over a split
f8 = rd("full8.a_velDiv")
if dt == "float32":   # (the double build has no --ps-merge fixed-point)
    x = rd("fxm.a_velDiv")
    e, m = np.abs(x - f8).max() / rmsf, abs(x.mean()) / rmsf
    out(e <= 1e-5 and m <= 1e-6, f"P9 --ps-merge fixed-point vs the ordered merge: {e:.2g} of the rms, box mean {m:.2g}")
wp = rd("winp.a_velDiv").reshape(n, n, -1)
f8 = f8.reshape(n, n, n)
e = np.abs(wp[:, :, 0] - f8[:, :, 16]).max() / rmsf if wp.shape[2] == 1 else 1.0
good = wp.shape[2] == 1 and (np.array_equal(wp[:, :, 0], f8[:, :, 16]) if dt == "float32" else e <= 1e-9)
out(good, f"P9 a one-cell window over a 2^3 split vs that split's full run: {e:.2g} of the rms")
sys.exit(0 if ok else 1)
PY

echo ">> the mass flux: periodic box mean; continuity over a drift (the double build)"
# P10: '.a_massFluxDiv' = div(sum_s rho_s v_s) from the face momentum sums. Its periodic box mean telescopes to 0. The
# double build drifts the 3-stream pancake's particles (and, for the y and z momentum words, crossed waves: 24^3
# particles, amplitude factor 1.5, 20% jitter) along their velocities, x + v t, for t in [-T, T] at 9 times:
# a cell's mass change M(T) - M(-T) equals minus the time integral of its outflow, caustic cells too (the PS model is
# exact for each tetrahedron at each instant). Simpson's rule on 8 intervals converges (4 -> 8 intervals: rms x 3.8)
# but slowly where a caustic sheet crosses a face in the interval (the flux is not smooth in t there).
if precision_double; then
    "$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$TMP/cw.hdf5" --n 24 --box 100 --amplitude-factor 1.5 \
        --jitter-frac 0.2 --crossed-waves > "$TMP/gen_cw.log" 2>&1 \
        || { echo "FAIL: generate_ps_test_data.py (see $TMP/gen_cw.log)"; exit 1; }
    "$PY" - "$TMP" <<'PY' || { echo "FAIL: writing the drift snapshots"; exit 1; }
import shutil, sys
import h5py
import numpy as np
tmp = sys.argv[1]
T, K = 1e-4, 8
for src, tag in (("pan_2.0", "drift"), ("cw", "cwdrift")):
    with h5py.File(f"{tmp}/{src}.hdf5", "r") as f:
        x0 = f["PartType1/Coordinates"][...].astype(np.float64)
        v = f["PartType1/Velocities"][...].astype(np.float64)
    for k in range(K + 1):
        fn = f"{tmp}/{tag}_{k}.hdf5"
        shutil.copy(f"{tmp}/{src}.hdf5", fn)
        with h5py.File(fn, "r+") as f:
            del f["PartType1/Coordinates"]
            f.create_dataset("PartType1/Coordinates", data=x0 + v * (-T + k * 2 * T / K))
PY
    for tag in drift cwdrift; do for k in 0 1 2 3 4 5 6 7 8; do
        "${PS_BIN}" "$TMP/${tag}_$k.hdf5" "$TMP/${tag}_$k" --grid 32 --periodic --input-format 105 --mpc-unit 1 \
            --field density_a divergence_a --exact --ps-volume-weighted --face-gradient \
            > "$TMP/${tag}_$k.log" 2>&1 || { echo "FAIL: run ${tag}_$k (see $TMP/${tag}_$k.log)"; exit 1; }
    done; done
fi
"$PY" - "$TMP" "${DTFE_TEST_REAL}" <<'PY' || FAIL=1
import os, sys
import numpy as np
tmp, dt = sys.argv[1], sys.argv[2]
rd = lambda name: np.fromfile(f"{tmp}/{name}", dtype=dt).astype(np.float64)
ok = True
def out(good, what):
    global ok
    print(("   PASS  " if good else "   FAIL  ") + what)
    ok = ok and good
p = rd("f2.0.a_massFluxDiv")
rms = np.sqrt((p**2).mean())
m = abs(p.mean()) / rms
out(rms > 0 and m <= 1e-6, f"P10 periodic box mean of the mass flux divergence: {m:.2g} of its rms")
if os.path.exists(f"{tmp}/drift_8.a_den"):
    T, K = 1e-4, 8
    for tag, what in (("drift", "the pancake (x)"), ("cwdrift", "crossed waves (x, y, z)")):
        D = np.array([rd(f"{tag}_{k}.a_massFluxDiv") for k in range(K + 1)])
        rms = np.sqrt((D[K // 2]**2).mean())
        dm = rd(f"{tag}_{K}.a_den") - rd(f"{tag}_0.a_den")
        def simpson(m):   # Simpson's rule on m intervals of [-T, T] (every K/m-th sample)
            w = np.ones(m + 1); w[1:-1:2] = 4; w[2:-1:2] = 2
            return 2 * T / m / 3 * (w[:, None] * D[::K // m]).sum(0)
        e4, e8 = (np.abs(dm + simpson(m)) / (2 * T) / rms for m in (4, 8))
        r4, r8 = np.sqrt((e4**2).mean()), np.sqrt((e8**2).mean())
        out(e8.max() <= 0.05 and r8 <= 2e-3 and r8 <= r4 / 2.5,
            f"P10 continuity over the drift of {what}: mass change vs the integrated outflow, {e8.max():.2g} max / "
            f"{r8:.2g} rms of the rms (4 -> 8 intervals: rms x {r4 / r8:.1f})")
else:
    print("   (P10's drift: the double build only, its float64 positions resolve the drift)")
sys.exit(0 if ok else 1)
PY

echo ">> the regular part's twins of a linear velocity field (a non-periodic clump, v = G x)"
# P15: without displacements the clump is one stream, and a linear field's gradient is G in every tetrahedron and in
# every cell, the stream average's and the mean flow's alike. The box [30, 70] lies inside the clump (about [20.6,
# 79.4] on each axis); a cell counts where '.a_streams' is 1 (to 1e-3)
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$TMP/lin.hdf5" --n 32 --box 100 --amplitude-factor 0 \
    --jitter-frac 0.2 --margin-frac 0.2 --linear-velocity 3 7 -2 -5 1 4 6 -3 -2.5 > "$TMP/gen_lin.log" 2>&1 \
    || { echo "FAIL: generate_ps_test_data.py --linear-velocity (see $TMP/gen_lin.log)"; exit 1; }
OMP_NUM_THREADS=1 "${PS_BIN}" "$TMP/lin.hdf5" "$TMP/lin" --grid 32 --box 30 70 30 70 30 70 --input-format 105 \
    --mpc-unit 1 --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a --exact \
    --ps-volume-weighted --face-gradient > "$TMP/lin.log" 2>&1 || { echo "FAIL: run lin (see $TMP/lin.log)"; exit 1; }
"$PY" - "$TMP" "${DTFE_TEST_REAL}" <<'PY' || FAIL=1
import sys
import numpy as np
tmp, dt = sys.argv[1], sys.argv[2]
G = np.array([3, 7, -2, -5, 1, 4, 6, -3, -2.5], dtype=np.float64).reshape(3, 3)   # G[j][i] = dv_j/dx_i
gm = np.abs(G).max()
rd = lambda s, c: np.fromfile(f"{tmp}/lin.{s}", dtype=dt).astype(np.float64).reshape(-1, c)
cov = np.abs(rd("a_streams", 1)[:, 0] - 1.0) < 1e-3     # the cells the clump covers whole (one stream)
S = 0.5 * (G + G.T) - np.trace(G) / 3.0 * np.eye(3)    # web_fields.cc velocityShear: xx xy xz yy yz, the trace removed
want = {"a_velGradStreams": G.ravel(), "a_velGrad": G.ravel(),
        "a_velShearStreams": np.array([S[0, 0], S[0, 1], S[0, 2], S[1, 1], S[1, 2]]),
        "a_velVortStreams": 0.5 * np.array([G[0, 1] - G[1, 0], G[0, 2] - G[2, 0], G[1, 2] - G[2, 1]])}
ok = cov.mean() > 0.9
print(("   PASS  " if ok else "   FAIL  ") + f"P15 linear field: {cov.mean():.2f} of the box covered by one stream")
for s, w in want.items():
    try:
        e = np.abs(rd(s, w.size)[cov] - w).max() / gm if cov.any() else 1.0
    except (OSError, ValueError) as err:
        e = float("inf"); print(f"   (lin.{s}: {err})")
    good = e <= 4e-5      # (about ten times the single build's largest measured error, 3.5e-6: the mean flow's)
    print(("   PASS  " if good else "   FAIL  ") + f"P15 linear field v = G x: {s} is G's own to {e:.2g} of max|G|")
    ok = ok and good
sys.exit(0 if ok else 1)
PY

echo ">> refusals"
refuse() {   # refuse <what> <expected message part> [args ...]
    local what="$1" msg="$2"; shift 2
    "${PS_BIN}" "$TMP/pan_0.6.hdf5" "$TMP/r" --grid 8 --periodic --input-format 105 --mpc-unit 1 --field density_a divergence_a "$@" \
        > "$TMP/r.log" 2>&1
    local rc=$?
    if [ "$rc" = "1" ] && grep -q -- "$msg" "$TMP/r.log"; then pass "P7 $what: exit 1 at option parsing"
    else fail "P7 $what: exit $rc (see $TMP/r.log)"; fi
}
refuse "with --sampled" "add '--exact' (or drop '--sampled')" --sampled --ps-volume-weighted --face-gradient
refuse "without --ps-volume-weighted" "add '--ps-volume-weighted'" --exact --face-gradient
refuse "both route flags" "exclude each other" --exact --ps-volume-weighted --face-gradient --per-tet-gradient
export DTFE_FACE_GRADIENT=1
refuse "DTFE_FACE_GRADIENT=1 with --sampled" "add '--exact' (or drop '--sampled')" --sampled --ps-volume-weighted
unset DTFE_FACE_GRADIENT

echo ">> the default (D13): the face gradient where the run qualifies, else the stream average and why"
run d2.0 pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --ps-volume-weighted
run e2.0 pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 DTFE_FACE_GRADIENT=0 -- --ps-volume-weighted
# the binary's default (src/user_options.cc kPsFaceGradientDefault) as its header names it
dref=f2.0; nf_why=""
if grep -qF "is not PS-DTFE's default" "$TMP/d2.0.log"; then dref=b2.0; nf_why="is not PS-DTFE's default"; fi
sameset() {   # sameset <run a> <run b>: the same output files, byte for byte (logs and run-log sidecars aside)
    local f g ok=1 na=0 nb=0
    for f in "$TMP/$1".*; do
        case "$f" in *.log|*.runlog*) continue ;; esac
        na=$((na + 1)); g="$TMP/$2${f#"$TMP/$1"}"
        [ -f "$g" ] && cmp -s "$f" "$g" || ok=0
    done
    for f in "$TMP/$2".*; do case "$f" in *.log|*.runlog*) ;; *) nb=$((nb + 1)) ;; esac; done
    [ "$ok" = 1 ] && [ "$na" -gt 0 ] && [ "$na" = "$nb" ]
}
if sameset "$dref" d2.0; then pass "P14 a qualifying run without a flag writes the $dref run's files byte for byte"
else fail "P14 a qualifying run without a flag differs from the $dref run (see $TMP/d2.0.log)"; fi
if sameset b2.0 e2.0; then pass "P14 DTFE_FACE_GRADIENT=0 is '--per-tet-gradient' byte for byte"
else fail "P14 DTFE_FACE_GRADIENT=0 differs from '--per-tet-gradient' (see $TMP/e2.0.log)"; fi
p14c=1
for f in "$TMP"/b2.0.*; do
    case "$f" in *.log|*.runlog*|*.a_velGrad|*.a_velDiv|*.a_velShear|*.a_velVort) continue ;; esac
    cmp -s "$f" "$TMP/f2.0${f#"$TMP/b2.0"}" || { p14c=0; echo "     differs: ${f##*/}"; }
done
extra=""
for f in "$TMP"/f2.0.*; do
    case "$f" in *.log|*.runlog*) continue ;; esac
    [ -e "$TMP/b2.0${f#"$TMP/f2.0"}" ] || extra="${extra}${f#"$TMP/f2.0"} "
done
# (the stream average's divergence, gradient, shear and vorticity: P15's twins; the mass flux divergence)
if [ "$p14c" = 1 ] \
   && [ "$extra" = ".a_massFluxDiv .a_velDivStreams .a_velGradStreams .a_velShearStreams .a_velVortStreams " ]; then
    pass "P14 the face run changes only the gradient family and adds '.a_massFluxDiv' and the stream average's twins"
else fail "P14 the face run's other files (extra: '$extra')"; fi
nfn=0
noface() {   # noface <what> <expected reason part> [args ...]: exit 0, the stream average, and the header's reason
    local what="$1" msg="$2"; shift 2
    [ -n "$nf_why" ] && msg="$nf_why"
    nfn=$((nfn + 1))
    local o="$TMP/nf$nfn"
    "${PS_BIN}" "$TMP/pan_0.6.hdf5" "$o" --grid 8 --periodic --input-format 105 --mpc-unit 1 \
        --field density_a divergence_a "$@" > "$o.log" 2>&1
    local rc=$?
    if [ "$rc" = "0" ] && [ ! -e "$o.a_velDivStreams" ] \
       && grep -qF "velocity gradient      : the stream average of each stream" "$o.log" \
       && grep -qF -- "$msg" "$o.log"; then pass "P14 $what: runs, keeps the stream average and says why"
    else fail "P14 $what: exit $rc or no reason line (see $o.log)"; fi
}
noface "the sampled deposit" "needs the exact deposit" --sampled --subsamples 3 --ps-volume-weighted
noface "mass-weighted moments" "needs '--ps-volume-weighted'" --exact
noface "the halo release" "cannot run with '--ps-halo-release'" --exact --ps-volume-weighted --ps-halo-release 2

echo ">> --interlace: every grid but the density is the first pass's, the face gradient's family among them"
# P19: the default run above (d2.0) with '--interlace' -- the second pass, on the particles moved by half a cell,
# gives the density only (main.cpp); the face sums, their normalization and the face outputs are the first pass's
run i2.0 pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 -- --ps-volume-weighted --interlace
gline() { grep -F "velocity gradient      :" "$TMP/$1.log" | head -1; }
p19=1
for f in "$TMP"/d2.0.*; do
    case "$f" in *.log|*.runlog*) continue ;; esac
    s="${f#"$TMP/d2.0"}"
    if [ "$s" = ".a_den" ]; then
        cmp -s "$f" "$TMP/i2.0$s" && { p19=0; echo "     .a_den: the first pass's (not interlaced)"; }
    else
        cmp -s "$f" "$TMP/i2.0$s" || { p19=0; echo "     differs: $s"; }
    fi
done
for f in "$TMP"/i2.0.*; do
    case "$f" in *.log|*.runlog*) continue ;; esac
    [ -e "$TMP/d2.0${f#"$TMP/i2.0"}" ] || { p19=0; echo "     only with --interlace: ${f##*/}"; }
done
if [ "$p19" = 1 ] && [ -e "$TMP/i2.0.a_den" ] && [ -n "$(gline d2.0)" ] && [ "$(gline d2.0)" = "$(gline i2.0)" ]; then
    pass "P19 --interlace: the same gradient route and every file but '.a_den' byte for byte as without it"
else fail "P19 --interlace changes the gradient route or a grid other than the density (see $TMP/i2.0.log)"; fi

if [ "$GPU" = "1" ]; then
    echo ">> the GPU exact deposit with the face sums from the CPU walk beside it (--gpu --face-gradient)"
    run gpu pan_2.0 OMP_NUM_THREADS=1 -- --ps-volume-weighted --face-gradient --gpu
    if ! grep -q " GPU (" "$TMP/gpu.log" || grep -q "using the CPU deposit" "$TMP/gpu.log"; then
        fail "P11 the --gpu run did not deposit on the GPU (see $TMP/gpu.log)"
    elif ! grep -q "face sums (--face-gradient) of the GPU-deposited" "$TMP/gpu.log"; then
        fail "P11 the --gpu run did not run the face walk (see $TMP/gpu.log)"
    else
        "$PY" - "$TMP" "$DTFE_TEST_REAL" <<'EOF' || FAIL=1
import numpy as np, sys
tmp, dt = sys.argv[1], np.dtype(sys.argv[2])
rd = lambda name: np.fromfile(f"{tmp}/{name}", dtype=dt).astype(np.float64)
ok = True
for suf, tol, what in (("a_velDiv", 0.0, "the mean flow's divergence"), ("a_massFluxDiv", 0.0, "the mass flux divergence")):
    c, g = rd(f"f2.0.{suf}"), rd(f"gpu.{suf}")
    r = np.sqrt((c ** 2).mean()) or 1.0
    d = float(np.abs(c - g).max()) / r
    if d <= tol: print(f"   PASS  P11 {what}: GPU exact + face walk == CPU exact + face ({d:.1e} of the rms; byte-identical)")
    else: print(f"   FAIL  P11 {what}: GPU exact + face walk differs from the CPU run by {d:.3g} of the rms"); ok = False
c, g = rd("f2.0.a_den"), rd("gpu.a_den")
scale = float(np.abs(c).max()) or 1.0
mean_rel, peak = abs(float(c.mean()) - float(g.mean())) / scale, float(np.abs(c - g).max()) / scale
if mean_rel <= 1e-5 and peak <= 5e-2: print(f"   PASS  P11 density: GPU exact vs CPU exact mean-rel {mean_rel:.1e}, peak-rel {peak:.1e}")
else: print(f"   FAIL  P11 density: GPU exact vs CPU exact mean-rel {mean_rel:.1e}, peak-rel {peak:.1e}"); ok = False
c, g = rd("f2.0.a_velDivStreams"), rd("gpu.a_velDivStreams")
r = np.sqrt((c ** 2).mean()) or 1.0
d = float(np.abs(c - g).max()) / r
if d <= 5e-2: print(f"   PASS  P11 the stream average's divergence (the device's float gradient): {d:.1e} of the rms")
else: print(f"   FAIL  P11 the stream average's divergence differs by {d:.3g} of the rms"); ok = False
# P15 the regular part's twins: the device's float per-stream gradient, on one scale (the gradient twin's rms)
r = np.sqrt((rd("f2.0.a_velGradStreams") ** 2).mean()) or 1.0
for suf in ("a_velGradStreams", "a_velShearStreams", "a_velVortStreams"):
    d = float(np.abs(rd(f"f2.0.{suf}") - rd(f"gpu.{suf}")).max()) / r
    if d <= 5e-2: print(f"   PASS  P15 GPU {suf} (the device's float gradient): {d:.1e} of the gradient twin's rms")
    else: print(f"   FAIL  P15 GPU {suf} differs from the CPU's by {d:.3g} of the gradient twin's rms"); ok = False
gg = np.fromfile(f"{tmp}/gpu.a_velGradStreams", dtype=dt).reshape(-1, 9)
if np.array_equal((gg[:, 0] + gg[:, 4]) + gg[:, 8], np.fromfile(f"{tmp}/gpu.a_velDivStreams", dtype=dt)):
    print("   PASS  P15 GPU: the trace of the gradient twin is its divergence twin (byte-identical)")
else: print("   FAIL  P15 GPU: the trace of the gradient twin differs from '.a_velDivStreams'"); ok = False
sys.exit(0 if ok else 1)
EOF
    fi
else echo "   (P11's --gpu case: not a GPU build)"; fi

echo ">> a lattice particle plane as a non-periodic box's upper x face (CPU, every build): one domain, a 2^3 split"
"$PY" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "$TMP/lat.hdf5" --n 32 --box 100 \
    --amplitude-factor 2.0 --jitter-frac 0 > "$TMP/gen_lat.log" 2>&1 \
    || { echo "FAIL: generate_ps_test_data.py (see $TMP/gen_lat.log)"; exit 1; }
# the lattice's x range for '--box': the upper face is a particle plane (one float32 x per lattice plane) and, with
# 28.125 = 32 x 0.87890625 exact in float and in double, grid plane 32 in both builds
"$PY" - "$TMP/lat.hdf5" > "$TMP/edge.txt" 2> "$TMP/edge.log" <<'EOF' || { echo "FAIL: the lattice box's edge (see $TMP/edge.log)"; exit 1; }
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    x = np.unique(f["PartType1/Coordinates"][:, 0])
assert x.size == 32 and x.dtype == np.float32, (x.size, x.dtype)
hi = float(x[np.argmin(np.abs(x - 70.0))])
lo = hi - 28.125
assert float(np.float32(lo)) == lo and hi - lo == 28.125, (lo, hi)
print(repr(lo), repr(hi))
EOF
read -r XLO XHI < "$TMP/edge.txt"
B=(--grid 32 --input-format 105 --mpc-unit 1 --field density_a velocity_a divergence_a shear_a --exact
   --ps-volume-weighted --face-gradient)
runb() { local C=("${B[@]}"); run "$@"; }   # 'run' without --periodic
# one domain for lc and its GPU twin lg (P12): the two devices cannot plan different splits. P13 splits the same box
# 2^3 (its own option list: one '--partition' per command line)
LBOX=(--box "$XLO" "$XHI" 30 70 30 70)
LB=("${LBOX[@]}" --partition 1 1 1)
runb lc  lat OMP_NUM_THREADS=1 -- "${LB[@]}"
runb lc8 lat OMP_NUM_THREADS=1 -- "${LBOX[@]}" --partition 2 2 2 --max-concurrent 2
"$PY" - "$TMP" "$DTFE_TEST_REAL" <<'EOF' || FAIL=1
import numpy as np, sys
tmp, dt = sys.argv[1], np.dtype(sys.argv[2])
n, ok = 32, True
# the last layer upper face carries the section of the tetrahedra resting on the box face from outside
# (psOwnsUpperBoundarySection). The flow is along x alone (v_y = v_z = 0 at every particle),
# so the divergence is the x gradient alone: with no stream on that face it fell back to the stream average, and
# .a_velDiv equalled .a_velDivStreams there to the byte (0 cells apart). From the face sums it is the face value,
# whose last bits differ from the stream average in nearly every cell, as in the layer below (all faces covered).
# The box face lies in the single-stream region, where the two agree to rounding (P1: 1e-5 of the rms).
d, s = (np.fromfile(f"{tmp}/lc.{k}", dtype=dt).reshape(n, n, n).astype(np.float64) for k in ("a_velDiv", "a_velDivStreams"))
last, below = int((d[n - 1] != s[n - 1]).sum()), int((d[n - 2] != s[n - 2]).sum())
rs = float(np.sqrt((s ** 2).mean())) or 1.0
e = float(np.abs(d[n - 1] - s[n - 1]).max()) / rs
if below > 0 and 2 * last >= below and e <= 1e-5:
    print(f"   PASS  P12 a particle plane as the box face: the last x layer upper face carries the section, its divergence "
          f"the face value in {last} of {n * n} cells (the layer below: {below}), {e:.1e} of the rms from the stream average")
else:
    print(f"   FAIL  P12 a particle plane as the box face: the last x layer divergence is the face value in {last} of "
          f"{n * n} cells (the layer below: {below}; want at least half as many), {e:.2g} of the rms from the stream "
          f"average (want <= 1e-5)"); ok = False
# P13: the same box split 2^3. The section must survive the split (the sub-grid of the partition that holds the
# tetrahedra beyond the box must hold the last x layer). Where a split lost it, the x gradient falls
# back to the stream average (the count sees that: within rounding of the face value, no tolerance does) and the
# mass flux divergence misses the whole outflow through that face (far beyond the tolerance). Not byte for byte:
# a split merges its partitions in float (P5), so the count is held to half of the one-domain count, not to it.
rd = lambda run, suf: np.fromfile(f"{tmp}/{run}.{suf}", dtype=dt).astype(np.float64)
d8, s8 = (rd("lc8", k).reshape(n, n, n) for k in ("a_velDiv", "a_velDivStreams"))
last8, below8 = int((d8[n - 1] != s8[n - 1]).sum()), int((d8[n - 2] != s8[n - 2]).sum())
e8 = float(np.abs(d8[n - 1] - s8[n - 1]).max()) / (float(np.sqrt((s8 ** 2).mean())) or 1.0)
if last > 0 and below8 > 0 and 2 * last8 >= last and e8 <= 1e-5:
    print(f"   PASS  P13 a 2^3 split of the lattice box: the last x layer divergence is the face value in {last8} of "
          f"{n * n} cells (one domain: {last}; the layer below: {below8}), {e8:.1e} of the rms from the stream average")
else:
    print(f"   FAIL  P13 a 2^3 split of the lattice box: the last x layer divergence is the face value in {last8} of "
          f"{n * n} cells, one domain in {last} (want at least half as many, and one domain above 0; the layer "
          f"below: {below8}), {e8:.2g} of the rms from the stream average (want <= 1e-5)"); ok = False
for suf, nc in (("a_massFluxDiv", 1), ("a_velDiv", 1), ("a_velShear", 5)):
    c, p = (rd(k, suf).reshape(n, n, n, nc) for k in ("lc", "lc8"))
    r = float(np.sqrt((c ** 2).mean()))
    dev = np.abs(p - c).reshape(n, -1).max(axis=1) / (r or 1.0)   # per x layer, in units of the one-domain rms
    if r > 0 and dev.max() <= 1e-5:
        print(f"   PASS  P13 a 2^3 split of the lattice box: {suf} within {dev.max():.1e} of the one-domain rms in every x layer")
    else:
        bad = [int(i) for i in np.where(~(dev <= 1e-5))[0]]
        print(f"   FAIL  P13 a 2^3 split of the lattice box: {suf} differs from one domain by {dev.max():.3g} of the rms "
              f"(rms {r:.3g}), worst in x layer {int(np.argmax(dev))}, beyond 1e-5 in x layers {bad[:4]} "
              f"({len(bad)} of {n}; the last x layer: {dev[n - 1]:.3g})"); ok = False
sys.exit(0 if ok else 1)
EOF

if [ "$GPU" = "1" ]; then
    echo ">> the face walk's filter against the deposit loop's (--gpu): a window, non-periodic boxes, the NaN hook"
    JB=(--box 35.9375 64.0625 35.9375 64.0625 35.9375 64.0625)
    run  gwin pan_2.0 OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 -- --ps-volume-weighted --face-gradient \
        --grid-window 0.01 99.99 0.01 99.99 50.1 53.0 --gpu
    runb jc pan_2.0 OMP_NUM_THREADS=1 -- "${JB[@]}"
    runb jg pan_2.0 OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 -- "${JB[@]}" --gpu
    runb lg lat     OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 -- "${LB[@]}" --gpu
    gpu_walked() {   # the run deposited on the GPU and ran the face walk beside it
        grep -q " GPU (" "$TMP/$1.log" && ! grep -q "using the CPU deposit" "$TMP/$1.log" \
            && grep -q "face sums (--face-gradient) of the GPU-deposited" "$TMP/$1.log"
    }
    bounds_ok() {    # single build: the extraction's bounds were taken over and equal the bound walk's
        precision_double || grep -q "13 bounds of the GPU extraction checked against the bound walk, 0 differ" "$TMP/$1.log"
    }
    gpu_run_ok() {   # gpu_run_ok <gpu run> <what>: says what is wrong with the run, if anything
        if ! gpu_walked "$1"; then
            fail "P12 $2: the --gpu run did not deposit on the GPU or skipped the face walk (see $TMP/$1.log)"; return 1
        elif ! bounds_ok "$1"; then
            fail "P12 $2: no bounds check line of DTFE_DEBUG_FX_BOUNDS (see $TMP/$1.log)"; return 1
        fi
    }
    pair() {         # pair <cpu run> <gpu run> <what>
        gpu_run_ok "$2" "$3" || return 0
        if meanflow "$1" "$2"; then pass "P12 $3: the mean flow's grids and the mass flux divergence, GPU == CPU byte for byte"
        else fail "P12 $3: the GPU run's mean flow grids differ from the CPU run's"; fi
    }
    pair win gwin "a one-cell --grid-window"
    pair jc  jg   "a non-periodic --box"
    if gpu_run_ok lg "a particle plane as the box face"; then
        "$PY" - "$TMP" "$DTFE_TEST_REAL" <<'EOF' || FAIL=1
import numpy as np, sys
tmp, dt = sys.argv[1], np.dtype(sys.argv[2])
n, ok = 32, True
# both devices deposit the section of the tetrahedra resting on the box face from outside (psOwnsUpperBoundarySection):
# the face-derived grids byte for byte in EVERY layer, the last x layer included
for suf, nc in (("a_massFluxDiv", 1), ("a_velDiv", 1), ("a_velShear", 5)):
    c, g = (np.fromfile(f"{tmp}/{k}.{suf}", dtype=dt).reshape(n, n, n, nc) for k in ("lc", "lg"))
    r = float(np.sqrt((c.astype(np.float64) ** 2).mean()))
    if r > 0 and np.array_equal(c, g):
        print(f"   PASS  P12 a particle plane as the box face: {suf}, GPU == CPU byte for byte in every x layer")
    else:
        bad = [i for i in range(n) if not np.array_equal(c[i], g[i])]
        d = float(np.abs(c.astype(np.float64) - g).max()) / (r or 1.0)
        print(f"   FAIL  P12 a particle plane as the box face: {suf} differs by {d:.3g} of the rms "
              f"in x layers {bad[:4]} ({len(bad)} of {n})"); ok = False
# (the CPU run lc alone shows that the last x layer upper face carries the section: P12 above, on every build)
sys.exit(0 if ok else 1)
EOF
    fi
    if ! precision_double; then   # the hook's shares are fixed-point words: the single build (ps_fixed_moments_check.sh F5)
        run nc pan_2.0 OMP_NUM_THREADS=1 DTFE_TEST_NAN_SHARE=50 -- --ps-volume-weighted --face-gradient
        run ng pan_2.0 OMP_NUM_THREADS=1 DTFE_TEST_NAN_SHARE=50 -- --ps-volume-weighted --face-gradient --gpu
        drops() { sed -n 's/.*face mass flux (--face-gradient) grid \([0-9][0-9]*\).*/\1/p' "$TMP/$1.log" | head -1; }
        dc="$(drops nc)"; dg="$(drops ng)"
        if ! gpu_walked ng; then
            fail "P12 the NaN hook: the --gpu run did not deposit on the GPU or skipped the face walk (see $TMP/ng.log)"
        elif [ -n "$dc" ] && [ "$dc" -gt 0 ] && [ "$dc" = "$dg" ] && meanflow nc ng; then
            pass "P12 the NaN hook: $dc face shares left out on both devices, the mean flow's grids byte for byte"
        else
            fail "P12 the NaN hook: face shares left out CPU '${dc}' / GPU '${dg}', or the grids differ (see $TMP/ng.log)"
        fi
        # a real NaN vertex mass (one particle's mass NaN under --ps-vertex-mass; --mean-density keeps rho_bar finite,
        # which the particles' own mean would not be): the walk keeps the NaN as the loop does, so both devices leave
        # out, and count, the same face shares
        "$PY" - "$TMP/pan_2.0.hdf5" "$TMP/pan_nanm.hdf5" > "$TMP/gen_nanm.log" 2>&1 <<'EOF' \
            || { echo "FAIL: the NaN-mass snapshot (see $TMP/gen_nanm.log)"; exit 1; }
import shutil, sys
import h5py
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
shutil.copyfile(src, dst)
with h5py.File(dst, "r+") as f:
    mt = np.array(f["Header"].attrs["MassTable"], dtype=np.float64)
    n = int(f["Header"].attrs["NumPart_ThisFile"][1])
    m = np.full(n, mt[1], dtype=np.float32)
    m[n // 3] = np.nan
    mt[1] = 0.0                        # per-particle masses: the reader takes the 'Masses' dataset
    f["Header"].attrs["MassTable"] = mt
    f["PartType1"].create_dataset("Masses", data=m)
EOF
        NM=(--ps-volume-weighted --face-gradient --ps-vertex-mass --mean-density 0.032768)   # 32^3 x mass 1 / 100^3
        run nmc pan_nanm OMP_NUM_THREADS=1 -- "${NM[@]}"
        run nmg pan_nanm OMP_NUM_THREADS=1 -- "${NM[@]}" --gpu
        dc="$(drops nmc)"; dg="$(drops nmg)"
        if ! gpu_walked nmg; then
            fail "P12 a NaN mass: the --gpu run did not deposit on the GPU or skipped the face walk (see $TMP/nmg.log)"
        elif [ -n "$dc" ] && [ "$dc" -gt 0 ] && [ "$dc" = "$dg" ] && meanflow nmc nmg; then
            pass "P12 a NaN mass: $dc face shares left out on both devices, the mean flow's grids byte for byte"
        else
            fail "P12 a NaN mass: face shares left out CPU '${dc}' / GPU '${dg}', or grids differ (see $TMP/nmg.log)"
        fi
    fi
else echo "   (P12's --gpu cases: not a GPU build)"; fi

if [ "$GPU" = "1" ]; then
    echo ">> the face walk beside the device call (default) against DTFE_FACE_WALK_EARLY=0: a window, splits, faults"
    E=DTFE_FACE_WALK_EARLY=1              # (what the default means)
    O=DTFE_FACE_WALK_EARLY=0              # the developer's off switch: the walk after the device's deposit
    FAULT=DTFE_GPU_FAULT_AT=1:persist     # every command buffer fails: the CPU deposits (tests/gpu_retry_check.sh R4)
    beside() { grep -q "of the GPU-deposited tetrahedra on the CPU in .*(beside the GPU deposit)" "$TMP/$1.log"; }
    fell_back() {    # the run's device call failed and the CPU deposit made the face sums
        grep -q "using the CPU deposit\|ran on the CPU instead" "$TMP/$1.log" \
            && ! grep -q "face sums (--face-gradient) of the GPU-deposited" "$TMP/$1.log"
    }
    # the partition loop's closing line (src/DTFE.cpp): '<walks kept> <walks thrown away>' (empty without one)
    wre='s/.* of \([0-9]*\) pass(es) were walked beside the GPU deposit.*; \([0-9]*\) walk(s) thrown away.*/\1 \2/p'
    walks() { sed -n "$wre" "$TMP/$1.log" | head -1; }
    # what two '--gpu' runs share byte for byte: the face-derived grids and the exact fixed-point sums
    grids_same() {
        meanflow "$1" "$2" && cmp -s "$TMP/$1.a_den" "$TMP/$2.a_den" && cmp -s "$TMP/$1.a_vel" "$TMP/$2.a_vel"
    }
    PG=(--ps-volume-weighted --face-gradient --partition 2 2 2 --max-concurrent 4 --gpu)
    run gpue pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=4 "$E" -- --ps-volume-weighted --face-gradient --gpu
    if precision_double; then
        if gpu_walked gpue && ! beside gpue && ! beside gpu && grids_same gpu gpue; then
            pass "P16 the double build: the walk stays after the deposit, by default and under =1 (the same grids)"
        else fail "P16 the double build: a walk beside the device call, or a grid differs (see $TMP/gpue.log)"; fi
    else
        run gpuo pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=4 "$O" -- --ps-volume-weighted --face-gradient --gpu
        run gwino pan_2.0 OMP_NUM_THREADS=1 "$O" -- --ps-volume-weighted --face-gradient \
            --grid-window 0.01 99.99 0.01 99.99 50.1 53.0 --gpu
        run gp8o pan_2.0 OMP_NUM_THREADS=4 "$O" -- "${PG[@]}"
        run gp8  pan_2.0 OMP_NUM_THREADS=4 -- "${PG[@]}"
        run gflo pan_2.0 OMP_NUM_THREADS=1 "$FAULT" "$O" -- --ps-volume-weighted --face-gradient --gpu
        run gfe  pan_2.0 OMP_NUM_THREADS=1 "$FAULT" "$E" -- --ps-volume-weighted --face-gradient --gpu
        run gp8f pan_2.0 OMP_NUM_THREADS=4 "$FAULT" -- "${PG[@]}"
        # (a CPU split under =1: no pass can take the walk, which its closing line says)
        run c8e  pan_2.0 OMP_NUM_THREADS=4 "$E" -- --ps-volume-weighted --face-gradient --partition 2 2 2 \
            --max-concurrent 4
        if ! gpu_walked gpue || ! beside gpue || ! beside gpu || ! gpu_walked gpuo || beside gpuo; then
            fail "P16 one domain: no face walk beside the device call by default or under =1, or one under =0"
        elif grids_same gpuo gpu && grids_same gpuo gpue; then
            pass "P16 one domain: the face sums beside the device call (default, =1), the grids of =0 byte for byte"
        else fail "P16 one domain: the face walk beside the device call changes a grid (see $TMP/gpue.log)"; fi
        if ! gpu_walked gwin || ! beside gwin || ! gpu_walked gwino || beside gwino; then
            fail "P16 a one-cell --grid-window: no face walk beside the device call by default, or one under =0"
        elif grids_same gwino gwin && meanflow win gwino; then
            pass "P16 a one-cell --grid-window: the grids of the walk after the deposit (=0), the CPU's mean flow"
        else fail "P16 a one-cell --grid-window: the face walk beside the device call changes a grid"; fi
        w8="$(walks gp8)"
        if [ -n "$(walks gp8o)" ] || [ "$w8" != "8 0" ] || grep -qF "DTFE_FACE_WALK_EARLY" "$TMP/gp8.log"; then
            fail "P16 a 2^3 split, 4 in flight: '${w8}' walks kept / thrown away, want '8 0' (see $TMP/gp8.log)"
        elif grids_same gp8o gp8; then
            pass "P16 a 2^3 split, 4 in flight: 8 walks beside their device calls by default, the grids of =0"
        else fail "P16 a 2^3 split, 4 in flight: the face walks beside the device calls change a grid"; fi
        if [ -z "$(walks c8e)" ] && ! grep -qF "DTFE_FACE_WALK_EARLY" "$TMP/full8.log" \
           && grep -qF "DTFE_FACE_WALK_EARLY is on, but no pass walked its face sums" "$TMP/c8e.log"; then
            pass "P16 a CPU split: under =1 its closing line says no pass took the walk; by default nothing is said"
        else fail "P16 a CPU split: no line under =1, or a line by default (see $TMP/c8e.log, $TMP/full8.log)"; fi
        if ! fell_back gflo || ! fell_back gfe; then
            fail "P16 a failed device call: the run did not fall back to the CPU deposit (see $TMP/gfe.log)"
        elif grids_same gflo gfe && same gflo gfe && meanflow f2.0 gfe; then
            pass "P16 a failed device call: the walk's sums thrown away, the grids of the walk after the deposit (=0)"
        else fail "P16 a failed device call: the walk beside the call differs from the =0 run ($TMP/gfe.log)"; fi
        wf="$(walks gp8f)"
        if [ "${wf}" = "0 8" ] && meanflow gp8o gp8f; then
            pass "P16 a 2^3 split, every device call failing: 8 walks thrown away, none kept, the same mean flow"
        else
            fail "P16 a 2^3 split, every device call failing: '${wf}' kept / thrown away, not '0 8' ($TMP/gp8f.log)"
        fi
        # a NaN vertex mass (P12's snapshot): the shares the walk leaves out are counted once, with either outcome
        run nmge pan_nanm OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=4 "$E" -- "${NM[@]}" --gpu
        run nmgo pan_nanm OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=4 "$O" -- "${NM[@]}" --gpu
        run nmfe pan_nanm OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=4 "$FAULT" "$E" -- "${NM[@]}" --gpu
        dc="$(drops nmc)"; dg="$(drops nmge)"; dz="$(drops nmgo)"; df="$(drops nmfe)"
        if ! gpu_walked nmge || ! beside nmge || ! gpu_walked nmgo || beside nmgo || ! fell_back nmfe; then
            fail "P16 a NaN mass: no walk beside the call, one under =0, or no fallback (see $TMP/nmge.log)"
        elif [ -n "$dc" ] && [ "$dc" -gt 0 ] && [ "$dc" = "$dg" ] && [ "$dc" = "$dz" ] && [ "$dc" = "$df" ] \
             && meanflow nmc nmge && meanflow nmc nmgo && meanflow nmc nmfe; then
            pass "P16 a NaN mass: $dc face shares left out beside a device call that succeeds, one that fails, and =0"
        else
            fail "P16 a NaN mass: shares left out CPU '${dc}' / beside '${dg}' / =0 '${dz}' / a failed call '${df}'"
        fi
        # the NaN test hook: its bounds are not the extraction's, so the walk stays after the deposit (P12's ng: the
        # default; ngo: =0)
        run ngo pan_2.0 OMP_NUM_THREADS=1 DTFE_TEST_NAN_SHARE=50 "$O" -- --ps-volume-weighted --face-gradient --gpu
        dc="$(drops ng)"; dg="$(drops ngo)"
        if gpu_walked ngo && ! beside ngo && ! beside ng && [ -n "$dc" ] && [ "$dc" = "$dg" ] && meanflow ng ngo; then
            pass "P16 the NaN hook: the walk stays after the deposit by default, as under =0 ($dc shares left out)"
        else fail "P16 the NaN hook: a walk beside the device call, or a grid differs from =0's (see $TMP/ng.log)"; fi
    fi
else echo "   (P16's --gpu cases: not a GPU build)"; fi

echo ">> the canonical float face sums (DTFE_FACE_F32=1): against the default, threads, DTFE_GPU_FACES=1, --gpu"
F32=(--ps-volume-weighted --face-gradient)
run f32 pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 DTFE_FACE_F32=1 -- "${F32[@]}"
if precision_double; then
    if sameset f2.0 f32 && grep -q "DTFE_FACE_F32 and DTFE_GPU_FACES are switches of the single-precision 3-D build" \
       "$TMP/f32.log"; then
        pass "P17 the double build: DTFE_FACE_F32=1 is ignored (one note line), the default run's files byte for byte"
    else fail "P17 the double build: DTFE_FACE_F32=1 changes the run or has no note line (see $TMP/f32.log)"; fi
else
    run f32t pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=10 DTFE_FACE_F32=1 -- "${F32[@]}"
    run f32s pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=6 DTFE_DEPOSIT_SLABS=1 DTFE_FACE_F32=1 -- "${F32[@]}"
    run g32  pan_2.0 OMP_NUM_THREADS=1 DTFE_DEPOSIT_THREADS=1 DTFE_GPU_FACES=1 -- "${F32[@]}"
    if grep -q "canonical float arithmetic of the GPU face kernel (DTFE_FACE_F32=1" "$TMP/f32.log" \
       && grep -q "canonical float arithmetic of the GPU face kernel (DTFE_GPU_FACES=1" "$TMP/g32.log" \
       && ! grep -q "canonical float arithmetic" "$TMP/f2.0.log"; then
        pass "P17 the runs under DTFE_FACE_F32=1 and DTFE_GPU_FACES=1 name the canonical arithmetic, the default not"
    else fail "P17 no line of the canonical face arithmetic (see $TMP/f32.log, $TMP/g32.log)"; fi
    # every output but the face family byte for byte, and the same set of files
    p17=1; na=0; nb=0
    for f in "$TMP"/f2.0.*; do
        case "$f" in *.log|*.runlog*) continue ;; esac
        na=$((na + 1))
        case "$f" in *.a_velGrad|*.a_velDiv|*.a_velShear|*.a_velVort|*.a_massFluxDiv) continue ;; esac
        cmp -s "$f" "$TMP/f32${f#"$TMP/f2.0"}" || { p17=0; echo "     differs: ${f##*/}"; }
    done
    for f in "$TMP"/f32.*; do case "$f" in *.log|*.runlog*) ;; *) nb=$((nb + 1)) ;; esac; done
    if [ "$p17" = 1 ] && [ "$na" -gt 0 ] && [ "$na" = "$nb" ]; then
        pass "P17 DTFE_FACE_F32=1: every output but the face family is the default run's byte for byte ($na files)"
    else fail "P17 DTFE_FACE_F32=1 changes an output outside the face family, or the set of files ($na / $nb)"; fi
    if same f32 f32t && same f32 f32s; then
        pass "P17 DTFE_FACE_F32=1: 10 threads and slab mode write the face grids of one thread byte for byte"
    else fail "P17 DTFE_FACE_F32=1: the thread count or slab mode changes a face grid"; fi
    if sameset f32 g32; then
        pass "P17 a CPU run under DTFE_GPU_FACES=1 writes the DTFE_FACE_F32=1 run's files byte for byte"
    else fail "P17 a CPU run under DTFE_GPU_FACES=1 differs from the DTFE_FACE_F32=1 run (see $TMP/g32.log)"; fi
    "$PY" - "$TMP" "$DTFE_TEST_REAL" <<'EOF' || FAIL=1
import sys
import numpy as np
tmp, dt = sys.argv[1], sys.argv[2]
TOL = 3e-5   # about ten times the largest value these lines print (the gradient's and the shear's)
ok = True
for suf in ("a_velDiv", "a_velGrad", "a_velShear", "a_velVort", "a_massFluxDiv"):
    a = np.fromfile(tmp + "/f2.0." + suf, dtype=dt).astype(np.float64)
    b = np.fromfile(tmp + "/f32." + suf, dtype=dt).astype(np.float64)
    # (the vorticity against the gradient's rms: the gradient's antisymmetric part, almost none in this flow)
    vort = suf == "a_velVort"
    s = np.fromfile(tmp + "/f2.0.a_velGrad", dtype=dt).astype(np.float64) if vort else a
    r = float(np.sqrt((s ** 2).mean())) if s.size else 0.0
    d = float(np.abs(a - b).max()) / r if (a.size > 0 and a.size == b.size and r > 0) else float("inf")
    nd = int((a != b).sum()) if a.size == b.size else -1
    good = d <= TOL
    print(("   PASS  " if good else "   FAIL  ") + "P17 DTFE_FACE_F32=1 against the default: " + suf + " within "
          + f"{d:.2g} of " + ("the gradient's" if vort else "its") + f" rms ({nd} of {a.size} values differ; "
          + f"tolerance {TOL:g})")
    ok = ok and good
sys.exit(0 if ok else 1)
EOF
    if [ "$GPU" = "1" ]; then
        # (the default walks beside the device call; DTFE_FACE_WALK_EARLY=0 after it: P16's beside)
        run gf32  pan_2.0 OMP_NUM_THREADS=1 DTFE_FACE_F32=1 -- "${F32[@]}" --gpu
        run gf32o pan_2.0 OMP_NUM_THREADS=1 DTFE_FACE_F32=1 DTFE_FACE_WALK_EARLY=0 -- "${F32[@]}" --gpu
        if gpu_walked gf32 && gpu_walked gf32o && beside gf32 && ! beside gf32o && meanflow f32 gf32 \
           && meanflow f32 gf32o; then
            pass "P17 --gpu under DTFE_FACE_F32=1: the walk beside the deposit and after it, the CPU run's face grids"
        else fail "P17 --gpu under DTFE_FACE_F32=1: the walk's face grids differ from the CPU run's ($TMP/gf32.log)"; fi
    else echo "   (P17's --gpu case: not a GPU build)"; fi
fi

if [ "$GPU" = "1" ]; then
    echo ">> the GPU face kernel (DTFE_GPU_FACES=1): the CPU's DTFE_FACE_F32=1 face grids, the rest unchanged"
    K=DTFE_GPU_FACES=1
    FACEF=(a_velGrad a_velDiv a_velShear a_velVort a_massFluxDiv)   # (P17's face family)
    faces_same() {   # faces_same <a> <b>: the face family's files the runs write, byte for byte (at least one)
        local s n=0
        for s in "${FACEF[@]}"; do
            [ -e "$TMP/$1.$s" ] || [ -e "$TMP/$2.$s" ] || continue
            cmp -s "$TMP/$1.$s" "$TMP/$2.$s" || return 1
            n=$((n + 1))
        done
        [ "$n" -gt 0 ]
    }
    others_same() {  # others_same <a> <b>: the same set of files, each one outside the face family byte for byte
        local f g na=0 nb=0
        for f in "$TMP/$1".*; do
            case "$f" in *.log|*.runlog*) continue ;; esac
            na=$((na + 1))
            case "$f" in *.a_velGrad|*.a_velDiv|*.a_velShear|*.a_velVort|*.a_massFluxDiv) continue ;; esac
            g="$TMP/$2${f#"$TMP/$1"}"
            [ -f "$g" ] && cmp -s "$f" "$g" || return 1
        done
        for f in "$TMP/$2".*; do case "$f" in *.log|*.runlog*) ;; *) nb=$((nb + 1)) ;; esac; done
        [ "$na" -gt 0 ] && [ "$na" = "$nb" ]
    }
    dv_same() { cmp -s "$TMP/$1.a_den" "$TMP/$2.a_den" && cmp -s "$TMP/$1.a_vel" "$TMP/$2.a_vel"; }
    kernel_made() {  # the run deposited on the GPU and the face kernel made the face sums of every pass
        grep -q " GPU (" "$TMP/$1.log" && ! grep -q "using the CPU deposit" "$TMP/$1.log" \
            && grep -q "of the GPU-deposited tetrahedra by the GPU face kernel in" "$TMP/$1.log" \
            && ! grep -q "the GPU face kernel did not make" "$TMP/$1.log" \
            && ! grep -q "of the GPU-deposited tetrahedra on the CPU in" "$TMP/$1.log"
    }
    if precision_double; then
        run gk pan_2.0 OMP_NUM_THREADS=1 "$K" -- "${F32[@]}" --gpu
        nnote=$(grep -c "DTFE_FACE_F32 and DTFE_GPU_FACES are switches of the single-precision 3-D build" \
                "$TMP/gk.log")
        if [ "$nnote" = "1" ] && gpu_walked gk && ! grep -q "by the GPU face kernel" "$TMP/gk.log" \
           && meanflow gpu gk && dv_same gpu gk; then
            pass "P18 the double build: DTFE_GPU_FACES=1 is ignored (one note line), the GPU run's grids unchanged"
        else fail "P18 the double build: DTFE_GPU_FACES=1 changes the GPU run, or no note line (see $TMP/gk.log)"; fi
    else
        # one domain (with DTFE_GPU_TIMING=1: only stderr lines; the face fault below reads its buffer counts)
        run gk pan_2.0 OMP_NUM_THREADS=1 DTFE_GPU_TIMING=1 "$K" -- "${F32[@]}" --gpu
        if ! kernel_made gk; then
            fail "P18 one domain: the face kernel did not make the face sums (see $TMP/gk.log)"
        elif faces_same f32 gk && others_same gpu gk; then
            pass "P18 one domain: the CPU DTFE_FACE_F32=1 run's face grids, the other files of the run without it"
        else fail "P18 one domain: a face grid differs from the CPU F32 run's, or another file from the plain run"; fi
        # a non-periodic box whose upper x face is a lattice particle plane (P12's lattice: the boundary sections of
        # the tetrahedra outside the box, psOwnsUpperBoundarySection's canonical test), one domain
        runb lc32 lat OMP_NUM_THREADS=1 DTFE_FACE_F32=1 -- "${LB[@]}"
        runb lk   lat OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 "$K" -- "${LB[@]}" --gpu
        if kernel_made lk && faces_same lc32 lk && others_same lg lk; then
            pass "P18 the lattice box's upper boundary plane: the CPU F32 run's face grids, the rest unchanged"
        else fail "P18 the lattice box: the face grids differ from the CPU's, or another file (see $TMP/lk.log)"; fi
        # a one-cell --grid-window (the early reject lets the face walk's own tetrahedra through for their records)
        W=(--grid-window 0.01 99.99 0.01 99.99 50.1 53.0)
        run win32 pan_2.0 OMP_NUM_THREADS=1 DTFE_FACE_F32=1 -- "${F32[@]}" "${W[@]}"
        run gwk   pan_2.0 OMP_NUM_THREADS=1 DTFE_DEBUG_FX_BOUNDS=1 "$K" -- "${F32[@]}" "${W[@]}" --gpu
        if kernel_made gwk && faces_same win32 gwk && others_same gwin gwk; then
            pass "P18 a one-cell --grid-window: the CPU DTFE_FACE_F32=1 window's face grids, the rest unchanged"
        else fail "P18 a one-cell --grid-window: the kernel's face grids or another file (see $TMP/gwk.log)"; fi
        # a 2^3 split with 4 partitions in flight (P16's), and its closing line: 8 passes made, none fallen back
        PK=(--ps-volume-weighted --face-gradient --partition 2 2 2 --max-concurrent 4)
        run c8f pan_2.0 OMP_NUM_THREADS=4 DTFE_FACE_F32=1 -- "${PK[@]}"
        run gk8 pan_2.0 OMP_NUM_THREADS=4 "$K" -- "${PK[@]}" --gpu
        k8re='s/.*by the GPU face kernel (DTFE_GPU_FACES=1) in \([0-9]*\) pass(es).*; \([0-9]*\) pass(es) whose.*/\1 \2/p'
        k8="$(sed -n "$k8re" "$TMP/gk8.log" | head -1)"
        if [ "$k8" != "8 0" ]; then
            fail "P18 a 2^3 split, 4 in flight: '${k8}' passes made / fallen back, want '8 0' (see $TMP/gk8.log)"
        elif faces_same c8f gk8 && dv_same gp8 gk8; then
            pass "P18 a 2^3 split, 4 in flight: 8 face passes on the device, the CPU split's face grids"
        else fail "P18 a 2^3 split, 4 in flight: the face grids differ from the CPU split's (see $TMP/gk8.log)"; fi
        # P9 across the face devices: that split killed after 3 partitions under the kernel and resumed with the CPU
        # walk making the face sums (DTFE_FACE_F32=1: the same 'face-sums=f32' line of the checkpoint key)
        env DTFE_TEST_EXIT_AFTER_PARTITIONS=3 OMP_NUM_THREADS=4 "$K" "${PS_BIN}" "$TMP/pan_2.0.hdf5" "$TMP/gk8r" \
            "${C[@]}" "${PK[@]}" --gpu > "$TMP/gk8ra.log" 2>&1
        rc_k=$?
        run gk8r pan_2.0 OMP_NUM_THREADS=4 DTFE_FACE_F32=1 -- "${PK[@]}" --gpu --resume
        if [ "$rc_k" = "137" ] && grep -q -- "--resume: [3-7] of 8 partitions" "$TMP/gk8r.log" \
           && faces_same gk8 gk8r && dv_same gk8 gk8r; then
            pass "P18 killed under the kernel after 3 of 8 partitions, resumed with the CPU walk: the same grids"
        else fail "P18 the resume across the face devices (kill exit $rc_k; see $TMP/gk8r.log)"; fi
        # a NaN vertex mass (P12's snapshot): the momentum shares it leaves out, counted alike on both devices
        run nm32c pan_nanm OMP_NUM_THREADS=1 DTFE_FACE_F32=1 -- "${NM[@]}"
        run nm32g pan_nanm OMP_NUM_THREADS=1 "$K" -- "${NM[@]}" --gpu
        dc="$(drops nm32c)"; dg="$(drops nm32g)"
        if kernel_made nm32g && [ -n "$dc" ] && [ "$dc" -gt 0 ] && [ "$dc" = "$dg" ] && faces_same nm32c nm32g \
           && others_same nmg nm32g; then
            pass "P18 a NaN mass: $dc face shares left out by the kernel as by the CPU, the face grids byte for byte"
        else fail "P18 a NaN mass: face shares left out CPU '${dc}' / kernel '${dg}', or a grid ($TMP/nm32g.log)"; fi
        # the NaN test hook: the kernel does not take the pass (its bounds are not the extraction's); the CPU walk
        # after the deposit makes the face sums, in the canonical arithmetic the switch implies
        run nc32 pan_2.0 OMP_NUM_THREADS=1 DTFE_TEST_NAN_SHARE=50 DTFE_FACE_F32=1 -- "${F32[@]}"
        run nk   pan_2.0 OMP_NUM_THREADS=1 DTFE_TEST_NAN_SHARE=50 "$K" -- "${F32[@]}" --gpu
        dc="$(drops nc32)"; dg="$(drops nk)"
        if gpu_walked nk && ! grep -q "by the GPU face kernel" "$TMP/nk.log" && [ -n "$dc" ] && [ "$dc" = "$dg" ] \
           && faces_same nc32 nk && others_same ng nk; then
            pass "P18 the NaN hook: the CPU walk makes the face sums ($dc shares left out), the CPU F32 run's grids"
        else fail "P18 the NaN hook under DTFE_GPU_FACES=1 (see $TMP/nk.log)"; fi
        # a face pass that fails: DTFE_GPU_FAULT_AT counts the process's command buffers, the face pass's after the
        # deposit's of the same hold, so its first one is the deposit's buffers + 1 (DTFE_GPU_TIMING=1's kernel lines
        # of the run gk). The chunk sizes follow the buffers' times, so a busy device can move that count either way:
        # up to five tries around it (+1, +2, +0, +3, -1) until the fault meets the face pass. Persistent: the face
        # pass gives up and the CPU walk makes the same words; innocent once: the face pass redoes its words from zero
        dbuf=$(awk '/^PS-DTFE Metal: / && / threads / && !/faceSweepItems/ {
                        for (i = 1; i < NF; ++i) if ($(i + 1) == "buffers") s += $i }
                    END { print s + 0 }' "$TMP/gk.log")
        nface=$(awk '/^PS-DTFE Metal: faceSweepItems/ {
                         for (i = 1; i < NF; ++i) if ($(i + 1) == "threads") { print $i; exit } }' "$TMP/gk.log")
        kp=0; ki=0
        for t in 1 2 0 3 -1; do   # (persistent: a fault that fired, in the face pass, the deposit kept)
            [ $((dbuf + t)) -gt 0 ] || continue
            run gkf pan_2.0 OMP_NUM_THREADS=1 "DTFE_GPU_FAULT_AT=$((dbuf + t)):persist" "$K" -- "${F32[@]}" --gpu
            if grep -q "injected fault" "$TMP/gkf.log" && ! grep -q "using the CPU deposit" "$TMP/gkf.log"; then
                kp=$((dbuf + t)); break
            fi
        done
        for t in 1 2 0 3 -1; do   # (innocent: the restart names the face pass's items)
            [ $((dbuf + t)) -gt 0 ] || continue
            run gki pan_2.0 OMP_NUM_THREADS=1 "DTFE_GPU_FAULT_AT=$((dbuf + t))" "$K" -- "${F32[@]}" --gpu
            if grep -q "resuming from the pass start at thread 0 of ${nface} " "$TMP/gki.log"; then
                ki=$((dbuf + t)); break
            fi
        done
        if [ "$kp" -gt 0 ] && grep -q "the face pass gave up" "$TMP/gkf.log" \
           && grep -q "the GPU face kernel did not make this pass's face sums" "$TMP/gkf.log" \
           && grep -q "of the GPU-deposited tetrahedra on the CPU in" "$TMP/gkf.log" \
           && faces_same f32 gkf && others_same gpu gkf; then
            pass "P18 a face pass that gives up (every buffer failing from the $kp-th): the CPU walk, the same words"
        else
            nf="$(grep -c "injected fault" "$TMP/gkf.log")"
            fail "P18 a face pass that gives up: no CPU walk, or other words (deposit buffers $dbuf, faults $nf; $TMP/gkf.log)"
        fi
        if [ "$ki" -gt 0 ] && kernel_made gki && faces_same f32 gki && others_same gpu gki; then
            pass "P18 a face buffer that fails once (the $ki-th): redone from zeroed words, the same face grids"
        else
            nf="$(grep -c "injected fault" "$TMP/gki.log")"
            fail "P18 a face buffer that fails once (deposit buffers $dbuf, face items '$nface', faults $nf; $TMP/gki.log)"
        fi
    fi
else echo "   (P18's --gpu cases: not a GPU build)"; fi

echo ">> dtfelib reads the stored mean-flow grids"
for k in f2.0 b2.0; do
    mkdir -p "$TMP/snap_$k"
    ln -sf "$TMP/pan_2.0.hdf5" "$TMP/snap_$k/combined_099.hdf5"
    for g in "$TMP"/$k.a_*; do ln -sf "$g" "$TMP/snap_$k/ps_output.${g##*/$k.}"; done
done
PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$TMP" "${DTFE_TEST_REAL}" <<'PY' || FAIL=1
import sys
import numpy as np
from dtfelib.io import FieldSet
tmp, dt = sys.argv[1], sys.argv[2]
fs, fb = FieldSet(f"{tmp}/snap_f2.0", method="ps"), FieldSet(f"{tmp}/snap_b2.0", method="ps")
d = fs.load("divergence", mean_field=True, scaled=False).ravel()
raw = np.fromfile(f"{tmp}/f2.0.a_velDiv", dtype=dt)
good = fs.face_gradient and not fb.face_gradient and np.array_equal(d, raw.astype(d.dtype))
print(("   PASS  " if good else "   FAIL  ") + f"P8 FieldSet.face_gradient {fs.face_gradient} / {fb.face_gradient}; "
      "load('divergence', mean_field=True) returns the stored grid")
# P15 the stream average under either gradient route: the face run's twin is the '--per-tet-gradient' run's stored grid
for nm in ("gradient", "divergence", "shear", "vorticity"):
    try:
        a, b = fs.load(nm, stream_average=True, scaled=False), fb.load(nm, stream_average=True, scaled=False)
        ok_nm = fs.has(nm, stream_average=True) and a.shape == b.shape and np.array_equal(a, b)
    except (OSError, ValueError) as err:
        ok_nm = False; print(f"   ({nm}: {err})")
    print(("   PASS  " if ok_nm else "   FAIL  ") + f"P15 load({nm!r}, stream_average=True): the face run's twin == "
          "the stored grid of a '--per-tet-gradient' run")
    good = good and ok_nm
sys.exit(0 if good else 1)
PY

if [ "$FAIL" = "0" ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; exit 1; fi
