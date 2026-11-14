#!/usr/bin/env bash
# The face sweep's geometry (src/CGAL_triangulation/face_sweep.h, the face method of docs/cleanup-plan.md P10)
# against exact references: tests/face_sweep_check.cpp (F1 sections integrate to the tetrahedron's volume and first
# moments; F2 the pieces of a section add up to it; F3 Gauss on a Kuhn tessellation with a linear field, grid planes
# on and off the vertex levels). And the canonical float sweep (src/CGAL_triangulation/face_sweep_f32.h, PS-DTFE's
# face sums under DTFE_FACE_F32=1 / DTFE_GPU_FACES=1, docs/cleanup-plan.md P11's GPU face kernel, step 2): F4 against
# the double sweep word by word and a kernel's loop bit for bit, F5 Gauss in float, F6 its exact properties (shared
# strip bounds, the half-open rule, the corner order, a vertex in a plane, the local index and the periodic seam; the
# .cpp lists them, with their provisional tolerances, TO MEASURE). Header-only: compiled here with $CXX (else clang++
# on macOS, g++ elsewhere), no binary or data needed. About 2 s before F4-F6 (with them: TO MEASURE).
# Usage: tests/face_sweep_check.sh [--no-build]   (the flag is accepted for the battery; this always compiles)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
if [ "$(uname -s)" = Darwin ]; then CXX="${CXX:-$(command -v clang++ || command -v g++)}"
else CXX="${CXX:-$(command -v g++ || command -v clang++)}"; fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"${CXX}" -std=c++17 -O2 -Wall -Wextra -I"${ROOT}/src/CGAL_triangulation" "${SCRIPT_DIR}/face_sweep_check.cpp" \
    -o "${TMP}/face_sweep_check" || { echo "FAIL: face_sweep_check.cpp does not compile"; exit 1; }
"${TMP}/face_sweep_check"
