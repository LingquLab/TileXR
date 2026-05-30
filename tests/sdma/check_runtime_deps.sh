#!/usr/bin/env bash
set -euo pipefail

lib="${1:-install/lib/libtile-comm.so}"

if [ ! -f "${lib}" ]; then
    echo "ERROR: ${lib} not found"
    exit 1
fi

ld_path=""
IFS=':' read -r -a ld_parts <<< "${LD_LIBRARY_PATH:-}"
for part in "${ld_parts[@]}"; do
    if [[ -z "${part}" || "${part}" == *devlib* ]]; then
        continue
    fi
    if [ -z "${ld_path}" ]; then
        ld_path="${part}"
    else
        ld_path="${ld_path}:${part}"
    fi
done

deps=$(LD_LIBRARY_PATH="${ld_path}" ldd "${lib}")
echo "${deps}"

if echo "${deps}" | grep -E 'libascend_hal.so => .*devlib' >/dev/null; then
    echo "ERROR: libascend_hal.so resolved from CANN devlib; runtime must use driver HAL"
    exit 1
fi

if echo "${deps}" | grep -i 'shmem' >/dev/null; then
    echo "ERROR: tile-comm links shmem unexpectedly"
    exit 1
fi

echo "TileXR SDMA runtime dependency check passed"
