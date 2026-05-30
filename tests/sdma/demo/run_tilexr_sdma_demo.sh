#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SDMA_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
TILEXR_ROOT=$(cd "${SDMA_DIR}/../.." && pwd)
INSTALL_DIR="${SDMA_DIR}/install"

CANN_HOME="${1:-${ASCEND_HOME_PATH:-}}"
DEVICE_ID="${2:-0}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true
SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(64 4096 1048576)
fi

if [ -z "${CANN_HOME}" ]; then
    echo "ERROR: CANN_HOME argument or ASCEND_HOME_PATH is required"
    exit 1
fi

if [ -f "${CANN_HOME}/set_env.sh" ]; then
    set +u
    source "${CANN_HOME}/set_env.sh"
    set -u
fi

export TILEXR_ENABLE_SDMA=1
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:${CANN_HOME}/lib64:${CANN_HOME}/$(uname -m)-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

bin="${INSTALL_DIR}/bin/tilexr_sdma_demo"
if [ ! -x "${bin}" ]; then
    echo "ERROR: ${bin} not found. Run: cd ${SDMA_DIR} && bash build.sh ${CANN_HOME}"
    exit 1
fi

echo "=========================================="
echo "  TileXR SDMA Demo"
echo "=========================================="
echo "CANN_HOME: ${CANN_HOME}"
echo "DEVICE_ID: ${DEVICE_ID}"
echo "Sizes: ${SIZES[*]}"
echo "Binary: ${bin}"
echo "=========================================="

for bytes in "${SIZES[@]}"; do
    echo "---- bytes=${bytes} ----"
    "${bin}" "${bytes}"
done
