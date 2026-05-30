#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CANN_HOME="${1:-${ASCEND_HOME_PATH:-}}"

if [ -z "${CANN_HOME}" ]; then
    set +u
    source "${TILEXR_ROOT}/scripts/common_env.sh"
    set -u
else
    export ASCEND_HOME_PATH="${CANN_HOME}"
    export ASCEND_TOOLKIT_HOME="${CANN_HOME}"
    export ASCEND_OPP_PATH="${CANN_HOME}/opp"
    if [ -f "${CANN_HOME}/set_env.sh" ]; then
        set +u
        source "${CANN_HOME}/set_env.sh"
        set -u
    fi
fi

export ARCH="${ARCH:-${TILEXR_OS_ARCH:-$(uname -m)}}"
if [ "${ARCH}" = "arm64" ]; then
    export ARCH="aarch64"
fi
export ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"

ROOT_BUILD="${TILEXR_ROOT}/build-sdma-tests"
ROOT_INSTALL="${TILEXR_ROOT}/install"
TEST_BUILD="${SCRIPT_DIR}/build"
TEST_INSTALL="${SCRIPT_DIR}/install"

rm -rf "${ROOT_BUILD}" "${TEST_BUILD}" "${TEST_INSTALL}"
mkdir -p "${ROOT_BUILD}" "${TEST_BUILD}" "${TEST_INSTALL}"

cmake -S "${TILEXR_ROOT}" -B "${ROOT_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${ROOT_INSTALL}" \
    -DTILEXR_BUILD_TESTS=OFF
cmake --build "${ROOT_BUILD}" --target install -j"$(nproc)"

for header in comm_args.h tilexr_sdma_types.h tilexr_sdma_config.h; do
    if [ ! -f "${ROOT_INSTALL}/include/${header}" ]; then
        echo "ERROR: expected installed header missing: ${ROOT_INSTALL}/include/${header}" >&2
        exit 1
    fi
done

if command -v bisheng >/dev/null 2>&1; then
    DEMO_OPTION="-DBUILD_TILEXR_SDMA_DEMO=ON"
else
    echo "WARN: bisheng not found; TileXR SDMA demo target will be skipped."
    DEMO_OPTION="-DBUILD_TILEXR_SDMA_DEMO=OFF"
fi

cmake -S "${SCRIPT_DIR}" -B "${TEST_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${TEST_INSTALL}" \
    ${DEMO_OPTION}
cmake --build "${TEST_BUILD}" --target install -j"$(nproc)"

echo "SDMA tests installed to ${TEST_INSTALL}/bin"
