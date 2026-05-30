#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL_DIR="${SCRIPT_DIR}/install"
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

set +u
source "${TILEXR_ROOT}/scripts/common_env.sh"
set -u

export ARCH="${ARCH:-${TILEXR_OS_ARCH:-$(uname -m)}}"
if [ "${ARCH}" = "arm64" ]; then
    export ARCH="aarch64"
fi
export ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:${ASCEND_HOME_PATH}/${ARCH}-linux/lib64:${ASCEND_DRIVER_PATH}/lib64/driver:${LD_LIBRARY_PATH:-}"

"${INSTALL_DIR}/bin/test_tilexr_sdma_metadata"
"${INSTALL_DIR}/bin/test_tilexr_sdma_api_invalid"

echo "TileXR SDMA unit tests passed"
