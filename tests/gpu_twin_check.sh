#!/usr/bin/env bash
# The Metal <-> CUDA twin check of the exact tetrahedron-cell clip (tests/gpu_twin_check.cpp has the details): the
# CUDA/HIP clip src/CGAL_triangulation/gpu_exact_clip.cuh, compiled UNCHANGED as plain C++ through gpu_cuda_compat.h's
# CPU emulation (-DGPU_EMU), run on metal/validate_deposit.cpp's T7, T13 and T14 tetrahedra and every cell of their
# windows, against the CPU's double-precision r3d -- and on macOS also against the Metal clip
# (metal/exact_clip.metal.inc, compiled at run time like the production kernels). Run it after any change to either
# twin; metal/README.md lists the other lockstep pairs.
#
# It builds its own small program (no make, no CGAL, none of the binaries) into $DTFE_TEST_TMP/gpu_twin_check/:
#   - a C++20 compiler ($CXX, else g++ first on Linux as in the Makefile, the system clang++ on macOS):
#     gpu_cuda_emu.h needs <barrier> and std::atomic_ref, as the Makefile's CUDAEMU=1 objects do (GCC 11 or newer
#     -- CI's GCC 13 builds them -- or a recent clang), on arm64 or x86-64 (the emulation's lane switch exists for
#     those two only; this program never switches lanes); a C compiler for r3d ($CC, else cc); pthreads;
#   - on macOS the Metal framework and the vendored metal-cpp; GPU_TWIN_METAL=0 builds without Metal (as on Linux).
# The flags are the CUDAEMU objects' (-std=c++20 -O2 -DGPU_EMU, -fcf-protection=none on x86-64) plus -Wall -Wextra;
# warnings are shown, not fatal.
# About 5 s on 10 cores, more on fewer (3 million tet-cell pairs, each clipped by the CUDA twin twice and by r3d once).
#
# Usage: tests/gpu_twin_check.sh [--no-build]   (the flag is accepted for the battery; this script always compiles
#        its own program, which takes a few seconds)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
# g++ first on Linux (the Makefile's choice there); on macOS the system clang++ (metal-cpp and the frameworks)
if [ "$(uname -s)" = Darwin ]; then CXX="${CXX:-$(command -v clang++ || command -v g++)}"
else CXX="${CXX:-$(command -v g++ || command -v clang++)}"; fi
CC="${CC:-$(command -v cc || command -v gcc || command -v clang)}"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/gpu_twin_check"
mkdir -p "$TMP" || exit 1
echo "============================================================"
echo " exact-clip twin check (gpu_exact_clip.cuh vs r3d / Metal)"
echo "============================================================"

# macOS: the SDK passed explicitly, as the Makefile's MACOS_ISYSROOT does (a compiler may guess a missing one)
SYSROOT=()
if [ "$(uname -s)" = Darwin ]; then
    SDK="$(xcrun --show-sdk-path 2>/dev/null || true)"
    [ -n "$SDK" ] && [ -d "$SDK" ] && SYSROOT=( -isysroot "$SDK" )
fi
# -Wno-unknown-pragmas: GCC does not know the CUDA source's '#pragma unroll' (nvcc and clang do) and would warn at each
FLAGS=( -std=c++20 -O2 -DGPU_EMU -Wall -Wextra -Wno-unknown-pragmas ${SYSROOT[@]+"${SYSROOT[@]}"} )
[ "$(uname -m)" = x86_64 ] && FLAGS+=( -fcf-protection=none )
LIBS=( -lpthread )
if [ "$(uname -s)" = Darwin ] && [ "${GPU_TWIN_METAL:-1}" != 0 ]; then
    FLAGS+=( -DTWIN_METAL -isystem third_party/metal-cpp )
    LIBS+=( -framework Metal -framework Foundation )
fi

# r3d as the Makefile builds it (R3D_FLAGS: plain C99)
if ! "$CC" -x c -std=c99 -O3 ${SYSROOT[@]+"${SYSROOT[@]}"} -c third_party/r3d/r3d.c -o "$TMP/r3d.o" \
        > "$TMP/build.log" 2>&1; then
    echo "   FAIL  r3d does not compile ($TMP/build.log):"; sed 's/^/      | /' "$TMP/build.log" | tail -20
    echo "RESULT: FAIL"; exit 1
fi
if ! "$CXX" -x c++ "${FLAGS[@]}" tests/gpu_twin_check.cpp -x none "$TMP/r3d.o" "${LIBS[@]}" -o "$TMP/gpu_twin_check" \
        >> "$TMP/build.log" 2>&1; then
    echo "   FAIL  the twin check does not compile ($TMP/build.log):"; sed 's/^/      | /' "$TMP/build.log" | tail -30
    echo "RESULT: FAIL"; exit 1
fi
if grep -q "warning:" "$TMP/build.log"; then
    echo "   (compiler warnings, see $TMP/build.log)"
    grep "warning:" "$TMP/build.log" | head -10 | sed 's/^/      | /'
fi

"$TMP/gpu_twin_check" metal/exact_clip.metal.inc
rc=$?
if [ "$rc" -eq 0 ]; then echo "RESULT: PASS (exact-clip twins)"
else echo "RESULT: FAIL (exact-clip twins, exit $rc)"; fi
exit "$rc"
