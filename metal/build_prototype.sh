#!/usr/bin/env bash
# Build the PS-DTFE Metal deposit validation harness.
# Offline-compiles the kernels to ps_deposit.metallib (runtime compile is the fallback), and the face kernel
# (metal/ps_face.metal, DTFE_GPU_FACES=1) to ps_face.metallib for a harness that tests it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The harness loads metal/ps_deposit.metallib whenever it exists, so a library left over from
# an older kernel is silently tested instead of the current source. Remove it first; without
# the offline compiler (Xcode's optional Metal toolchain) the harness compiles from source.
rm -f metal/ps_deposit.air metal/ps_deposit.metallib metal/ps_face.air metal/ps_face.metallib
if xcrun -sdk macosx metal -v >/dev/null 2>&1; then
    echo ">> compiling Metal kernels (offline)"
    # the exact clip helpers, the run-wide fixed-point words, the parameter blocks and the PS kernels' shared guard,
    # 64-bit add and items' map reader come first, as the Makefile prepends them to the binaries' kernel source;
    # PS_FX_WORDS 0 = the 64-bit words the harness runs (the binaries compile one library per word size)
    xcrun -sdk macosx metal    -c -D PS_FX_WORDS=0 \
        -include metal/exact_clip.metal.inc -include metal/fixed_words.metal.inc \
        -include src/CGAL_triangulation/ps_kernel_params.h -include metal/ps_common.metal.inc \
        metal/ps_deposit.metal -o metal/ps_deposit.air
    xcrun -sdk macosx metallib metal/ps_deposit.air      -o metal/ps_deposit.metallib
    # The face kernel, as the Makefile embeds it (no exact clip; the canonical face arithmetic face_sweep_f32.h last)
    # and in the safe math mode its host compiles it in (-fno-fast-math, the offline twin of
    # MTLCompileOptions.fastMathEnabled = false; the newer -fmetal-math-mode=safe is not used here, unverified on this
    # toolchain). A failure here leaves the deposit's library and the harness standing: the harness then compiles the
    # face kernel from source at run time and reports the compiler's message there.
    if ! xcrun -sdk macosx metal -c -D PS_FX_WORDS=0 -fno-fast-math \
            -include metal/fixed_words.metal.inc -include src/CGAL_triangulation/ps_kernel_params.h \
            -include metal/ps_common.metal.inc -include src/CGAL_triangulation/face_sweep_f32.h \
            metal/ps_face.metal -o metal/ps_face.air \
       || ! xcrun -sdk macosx metallib metal/ps_face.air -o metal/ps_face.metallib; then
        rm -f metal/ps_face.air metal/ps_face.metallib
        echo "!! the face kernel (metal/ps_face.metal) did not compile offline: see the compiler's message above"
    fi
else
    echo ">> no offline Metal compiler (xcodebuild -downloadComponent MetalToolchain); the harness compiles the kernel at run time"
fi

echo ">> building host binary"
clang++ -std=c++17 -O2 -I third_party/metal-cpp metal/validate_deposit.cpp \
    -framework Metal -framework Foundation -o metal/validate_deposit
echo "built metal/validate_deposit"
