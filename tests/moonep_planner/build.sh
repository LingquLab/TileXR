#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILEXR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODE="${1:-source-only}"

case "${MODE}" in
    source-only)
        DEMO=OFF
        ;;
    full)
        : "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
        : "${ASCEND_DRIVER_PATH:=/usr/local/Ascend/driver}"
        export ASCEND_HOME_PATH ASCEND_DRIVER_PATH
        cmake -S "${TILEXR_ROOT}" -B "${TILEXR_ROOT}/build_moonep" \
            -DCMAKE_INSTALL_PREFIX="${TILEXR_ROOT}/install" \
            -DTILEXR_BUILD_MOONEP_PLANNER=ON
        cmake --build "${TILEXR_ROOT}/build_moonep" --target install -j"$(nproc)"
        DEMO=ON
        ;;
    *)
        echo "Usage: $0 [source-only|full]" >&2
        exit 2
        ;;
esac

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
    -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/install" \
    -DBUILD_TILEXR_MOONEP_PLANNER_DEMO="${DEMO}"
cmake --build "${SCRIPT_DIR}/build" --target install -j"$(nproc)"
ctest --test-dir "${SCRIPT_DIR}/build" --output-on-failure

