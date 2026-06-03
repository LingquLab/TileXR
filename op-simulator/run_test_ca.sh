#!/usr/bin/env bash
set -euo pipefail

EXECUTABLE="test_template"
ulimit -n 65536
CANN_DIR="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SOC_VERSION="${SOC_VERSION:-Ascend950}"
SIMULATOR_ARCH="${SIMULATOR_ARCH:-dav_3510}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-${CANN_DIR}/aarch64-linux/simulator/${SIMULATOR_ARCH}}"
SIMULATOR_CAMODEL_DIR="${SIMULATOR_CAMODEL_DIR:-${SIMULATOR_ROOT}/camodel}"
SIMULATOR_LIB_DIR="${SIMULATOR_LIB_DIR:-${SIMULATOR_ROOT}/lib}"
KERNEL_OBJECT="${KERNEL_OBJECT:-./op/my_kernel.o}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/tilexr_op_simulator_demo_results}"

echo $EXECUTABLE
echo $CANN_DIR
echo $SOC_VERSION
echo $SIMULATOR_ROOT

export ASCEND_HOME_PATH="${CANN_DIR}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$CANN_DIR}"

g++ -std=c++17 -g -I.. -o "${EXECUTABLE}" "${EXECUTABLE}.cpp" \
   -I"${CANN_DIR}/aarch64-linux/pkg_inc" \
   -I"${CANN_DIR}/aarch64-linux/pkg_inc/runtime" \
   -I"${CANN_DIR}/aarch64-linux/pkg_inc/runtime/runtime" \
   -I"${CANN_DIR}/aarch64-linux/include" \
   -I"${CANN_DIR}/aarch64-linux/include/experiment/runtime" \
   -I"${CANN_DIR}/aarch64-linux/include/experiment/runtime/runtime" \
   -I"${CANN_DIR}/aarch64-linux/include/experiment/msprof" \
   -L"${SIMULATOR_CAMODEL_DIR}" \
   -L"${SIMULATOR_LIB_DIR}" \
   -L"${CANN_DIR}/aarch64-linux/lib64" \
   -L"${CANN_DIR}/aarch64-linux/lib64/plugin/opskernel" \
   -Wl,-rpath-link,"${SIMULATOR_CAMODEL_DIR}" \
   -Wl,-rpath-link,"${SIMULATOR_LIB_DIR}" \
   -Wl,-rpath-link,"${CANN_DIR}/aarch64-linux/lib64" \
   -lruntime_camodel -lnpu_drv_camodel -lstars -lascendcl

export LD_LIBRARY_PATH="${SIMULATOR_CAMODEL_DIR}:${SIMULATOR_LIB_DIR}:${CANN_DIR}/aarch64-linux/lib64:${CANN_DIR}/aarch64-linux/lib64/plugin/opskernel:${LD_LIBRARY_PATH:-}"

if command -v cannsim >/dev/null 2>&1; then
  mkdir -p "${RESULTS_DIR}"
  cannsim record -g -o "${RESULTS_DIR}" -s "${SOC_VERSION}" -n 0 -f "${KERNEL_OBJECT}" ./"${EXECUTABLE}"
else
  msprof op simulator --soc-version="${SOC_VERSION}" ./"${EXECUTABLE}"
fi
