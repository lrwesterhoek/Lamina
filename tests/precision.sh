# Sourced by the shell suites: the precision they run at. DTFE_TEST_PRECISION=single (default) uses ./PS-DTFE and ./DTFE
# (float32 grids), =double uses ./PS-DTFE-double and ./DTFE-double ('make DTFE PS-DTFE DOUBLE=1 [METAL=1]', float64 grids).
# Grids are read with DTFE_TEST_REAL; .pts_* outputs have fixed types. PS_BIN / DTFE_BIN name the pair, PS_OBJ / DTFE_OBJ its object folders.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
case "${DTFE_TEST_PRECISION:-single}" in
    double)
        PS_BIN=./PS-DTFE-double; DTFE_BIN=./DTFE-double; PS_OBJ=o_ps_d; DTFE_OBJ=o_d
        REAL_BYTES=8; export DTFE_TEST_REAL=float64 ;;
    single)
        PS_BIN=./PS-DTFE; DTFE_BIN=./DTFE; PS_OBJ=o_ps; DTFE_OBJ=o
        REAL_BYTES=4; export DTFE_TEST_REAL=float32 ;;
    *)
        echo "DTFE_TEST_PRECISION must be 'single' or 'double' (got '${DTFE_TEST_PRECISION}')" >&2; exit 2 ;;
esac
export DTFE_TEST_PRECISION="${DTFE_TEST_PRECISION:-single}"
# Where the suites put their grids: OUTSIDE the repository by default (a repository in a cloud-synced folder would
# upload GBs of throwaway grids). DTFE_TEST_TMP overrides.
export DTFE_TEST_TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"
mkdir -p "${DTFE_TEST_TMP}"
echo ">> precision: ${DTFE_TEST_PRECISION} (${PS_BIN}, ${DTFE_BIN}; grid outputs ${DTFE_TEST_REAL})"
# true in double mode: the suites skip their own build step
precision_double() { [ "${DTFE_TEST_PRECISION}" = "double" ]; }
# exit 0 with a SKIP line when a binary this suite needs is not built in double mode
precision_require() {
    local b
    for b in "$@"; do
        if precision_double && [ ! -x "${b}" ]; then
            skip_or_fail "${b} is not built (make DTFE PS-DTFE DOUBLE=1 builds the double pair)"
        fi
    done
}
