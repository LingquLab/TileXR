#!/usr/bin/env bash
set -euo pipefail

CANN_DIR="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
OUT="${RUNNER_BIN:-/tmp/tilexr_moe_epcard_compare_runner}"
SIMULATOR_ARCH="${SIMULATOR_ARCH:-dav_3510}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-${CANN_DIR}/aarch64-linux/simulator/${SIMULATOR_ARCH}}"
SIMULATOR_CAMODEL_DIR="${SIMULATOR_CAMODEL_DIR:-${SIMULATOR_ROOT}/camodel}"
SIMULATOR_LIB_DIR="${SIMULATOR_LIB_DIR:-${SIMULATOR_ROOT}/lib}"

export ASCEND_HOME_PATH="${CANN_DIR}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$CANN_DIR}"

g++ -std=c++17 -O2 \
  -I"${CANN_DIR}/aarch64-linux/pkg_inc" \
  -I"${CANN_DIR}/aarch64-linux/pkg_inc/runtime" \
  -I"${CANN_DIR}/aarch64-linux/pkg_inc/runtime/runtime" \
  -I"${CANN_DIR}/aarch64-linux/include" \
  "$(dirname "$0")/runner.cpp" \
  -L"${SIMULATOR_CAMODEL_DIR}" \
  -L"${SIMULATOR_LIB_DIR}" \
  -L"${CANN_DIR}/aarch64-linux/lib64" \
  -Wl,-rpath-link,"${SIMULATOR_CAMODEL_DIR}" \
  -Wl,-rpath-link,"${SIMULATOR_LIB_DIR}" \
  -Wl,-rpath-link,"${CANN_DIR}/aarch64-linux/lib64" \
  -lruntime_camodel -lnpu_drv_camodel -lstars -lascendcl \
  -o "${OUT}"

echo "${OUT}"
