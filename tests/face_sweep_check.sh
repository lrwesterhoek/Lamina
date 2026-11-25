#!/usr/bin/env bash
# The face sweep's geometry (src/CGAL_triangulation/face_sweep.h) against exact references (tests/face_sweep_check.cpp).
# Header-only: compiled here with $CXX (else clang++ on macOS, g++ elsewhere); no binary or data needed.
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
