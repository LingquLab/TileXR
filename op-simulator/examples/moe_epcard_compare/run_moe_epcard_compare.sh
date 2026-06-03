#!/usr/bin/env bash
set -euo pipefail

C="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SIMULATOR_ARCH="${SIMULATOR_ARCH:-dav_3510}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-$C/aarch64-linux/simulator/$SIMULATOR_ARCH}"
SIMULATOR_CAMODEL_DIR="${SIMULATOR_CAMODEL_DIR:-$SIMULATOR_ROOT/camodel}"
SIMULATOR_LIB_DIR="${SIMULATOR_LIB_DIR:-$SIMULATOR_ROOT/lib}"

export ASCEND_HOME_PATH="$C"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$C}"
export LD_LIBRARY_PATH="$SIMULATOR_CAMODEL_DIR:$SIMULATOR_LIB_DIR:$C/aarch64-linux/lib64:$C/aarch64-linux/lib64/plugin/opskernel:${LD_LIBRARY_PATH:-}"

if [ -f /usr/lib64/libgomp.so.1.0.0 ]; then
  export LD_PRELOAD="/usr/lib64/libgomp.so.1.0.0${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec "${RUNNER_BIN:-/tmp/tilexr_moe_epcard_compare_runner}" "$@"
