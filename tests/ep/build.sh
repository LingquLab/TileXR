#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILEXR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODE="${1:-source-only}"
COMBINE_DAF_WQE_MODE="${TILEXR_COMBINE_DAF_WQE_MODE:-simt_raw}"

: "${ASCEND_HOME_PATH:=}"
: "${LD_LIBRARY_PATH:=}"
source "${TILEXR_ROOT}/scripts/common_env.sh"
export ARCH="${TILEXR_OS_ARCH}"

case "${COMBINE_DAF_WQE_MODE}" in
    api|scalar_raw|simt_raw)
        ;;
    *)
        echo "Usage: TILEXR_COMBINE_DAF_WQE_MODE={api|scalar_raw|simt_raw} $0 [source-only|full]" >&2
        exit 2
        ;;
esac

case "${MODE}" in
    full)
        ROOT_BUILD_DIR="${TILEXR_ROOT}/build_ep"
        cmake -S "${TILEXR_ROOT}" -B "${ROOT_BUILD_DIR}" \
            -DCMAKE_INSTALL_PREFIX="${TILEXR_ROOT}/install" \
            -DTILEXR_BUILD_EP=ON \
            -DTILEXR_COMBINE_DAF_WQE_MODE="${COMBINE_DAF_WQE_MODE}"
        cmake --build "${ROOT_BUILD_DIR}" --target install -j"$(nproc)"
        DEMO_OPTION="-DBUILD_TILEXR_EP_DEMO=ON"
        ;;
    source-only)
        DEMO_OPTION="-DBUILD_TILEXR_EP_DEMO=OFF"
        ;;
    *)
        echo "Usage: $0 [source-only|full]" >&2
        exit 2
        ;;
esac

BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    "${DEMO_OPTION}"
cmake --build "${BUILD_DIR}" --target install -j"$(nproc)"

echo "TileXR EP test build complete (${MODE})."
echo "TileXR combine DataAsFlag WQE mode: ${COMBINE_DAF_WQE_MODE}"
echo "Installed test binaries: ${INSTALL_DIR}/bin"
