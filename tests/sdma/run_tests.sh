#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL_DIR="${SCRIPT_DIR}/install"
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

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: ASCEND_HOME_PATH is not set; pass CANN_HOME or source scripts/common_env.sh" >&2
    exit 1
fi

export ARCH="${ARCH:-${TILEXR_OS_ARCH:-$(uname -m)}}"
if [ "${ARCH}" = "arm64" ]; then
    export ARCH="aarch64"
fi
export ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:${ASCEND_DRIVER_PATH}/lib64/driver:${ASCEND_HOME_PATH}/${ARCH}-linux/lib64:${LD_LIBRARY_PATH:-}"

"${INSTALL_DIR}/bin/test_tilexr_sdma_metadata"
"${INSTALL_DIR}/bin/test_tilexr_sdma_api_invalid"
"${INSTALL_DIR}/bin/test_tilexr_sdma_transport_disabled"
"${INSTALL_DIR}/bin/test_tilexr_sdma_comm_wiring"

if command -v npu-smi >/dev/null 2>&1; then
    "${INSTALL_DIR}/bin/test_tilexr_sdma_disabled_comm"
else
    echo "Skip test_tilexr_sdma_disabled_comm: npu-smi not found"
fi

echo "TileXR SDMA unit tests passed"
