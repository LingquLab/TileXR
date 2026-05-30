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

ARCH="${ARCH:-${TILEXR_OS_ARCH:-$(uname -m)}}"
if [ "${ARCH}" = "arm64" ]; then
    ARCH="aarch64"
fi
export ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"

SANITIZED_LD_LIBRARY_PATH=""
IFS=':' read -r -a LD_LIBRARY_PATH_PARTS <<< "${LD_LIBRARY_PATH:-}"
for path in "${LD_LIBRARY_PATH_PARTS[@]}"; do
    if [[ -z "${path}" || "${path}" == *devlib* ]]; then
        continue
    fi
    if [ -z "${SANITIZED_LD_LIBRARY_PATH}" ]; then
        SANITIZED_LD_LIBRARY_PATH="${path}"
    else
        SANITIZED_LD_LIBRARY_PATH="${SANITIZED_LD_LIBRARY_PATH}:${path}"
    fi
done

export TILEXR_ENABLE_SDMA=1
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export LD_LIBRARY_PATH="${ASCEND_DRIVER_PATH}/lib64/driver:${ASCEND_DRIVER_PATH}/lib64/common:${ASCEND_DRIVER_PATH}/lib64:${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:${CANN_HOME}/lib64:${CANN_HOME}/${ARCH}-linux/lib64"
if [ -n "${SANITIZED_LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SANITIZED_LD_LIBRARY_PATH}"
fi

bin="${INSTALL_DIR}/bin/tilexr_sdma_demo"
if [ ! -x "${bin}" ]; then
    echo "ERROR: ${bin} not found. Run: cd ${SDMA_DIR} && bash build.sh ${CANN_HOME}"
    exit 1
fi

deps=$(ldd "${bin}" || true)
if echo "${deps}" | grep -E 'libascend_hal.so => not found' >/dev/null; then
    echo "ERROR: libascend_hal.so not found; SDMA demo requires driver HAL from ${ASCEND_DRIVER_PATH}/lib64/driver"
    echo "${deps}"
    exit 1
fi

if echo "${deps}" | grep -E 'libascend_hal.so => .*devlib' >/dev/null; then
    echo "ERROR: libascend_hal.so resolved from CANN devlib; SDMA demo must use driver HAL"
    echo "${deps}"
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
