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

if echo "${deps}" | grep -E 'libascend_hal.so => not found' >/dev/null; then
    echo "WARNING: libascend_hal.so not found; HAL-dependent SDMA runtime tests may be skipped"
fi

if echo "${deps}" | grep -E 'libnnopbase.so => not found' >/dev/null; then
    echo "ERROR: libnnopbase.so not found; PTO SDMA runtime must use CANN runtime lib64, not devlib-only linkage"
    exit 1
fi

if echo "${deps}" | grep -E 'libnnopbase.so => .*devlib' >/dev/null; then
    echo "ERROR: libnnopbase.so resolved from CANN devlib; runtime must use CANN runtime lib64"
    exit 1
fi

if echo "${deps}" | grep -E 'lib(acl)?shmem|=> .*shmem.*\.so' >/dev/null; then
    echo "ERROR: tile-comm links shmem unexpectedly"
    exit 1
fi

echo "TileXR SDMA runtime dependency check passed"
