#!/usr/bin/env bash
set -euo pipefail

lib="${1:-install/lib/libtile-comm.so}"

if [ ! -f "${lib}" ]; then
    echo "ERROR: ${lib} not found"
    exit 1
fi

deps=$(ldd "${lib}")
echo "${deps}"

if echo "${deps}" | grep -E 'libascend_hal.so => .*devlib' >/dev/null; then
    echo "ERROR: libascend_hal.so resolved from CANN devlib; runtime must use driver HAL"
    exit 1
fi

if echo "${deps}" | grep -E 'lib(acl)?shmem|=> .*shmem.*\.so' >/dev/null; then
    echo "ERROR: tile-comm links shmem unexpectedly"
    exit 1
fi

echo "TileXR SDMA runtime dependency check passed"
