#!/bin/bash
#
# Run UDMA tests.
#

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT="${SCRIPT_DIR}/../.."
INSTALL_DIR="${SCRIPT_DIR}/install"

source "${TILEXR_ROOT}/scripts/common_env.sh"

if [ -x /usr/local/mpi/bin/mpirun ]; then
    export PATH="/usr/local/mpi/bin:${PATH}"
fi

export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${INSTALL_DIR}/lib64:${TILEXR_ROOT}/install/lib:${TILEXR_ROOT}/install/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================="
echo "  Running UDMA Tests"
echo "=========================================="
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

detect_ok_npus() {
    command -v npu-smi >/dev/null 2>&1 || return 0
    local ids=()
    for id in $(seq 0 15); do
        local health
        health=$(npu-smi info -t health -i "${id}" 2>/dev/null |
            awk -F: '/Health Status/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
        if [ "${health}" = "OK" ]; then
            ids+=("${id}")
        fi
    done
    local IFS=,
    echo "${ids[*]}"
}

if [ -z "${TILEXR_TEST_DEVICES:-}" ]; then
    TILEXR_TEST_DEVICES=$(detect_ok_npus)
    export TILEXR_TEST_DEVICES
fi
if [ -n "${TILEXR_TEST_DEVICES:-}" ]; then
    echo "TILEXR_TEST_DEVICES: ${TILEXR_TEST_DEVICES}"
fi

required_bins=(
    test_tilexr_udma_transport_layout
    test_tilexr_udma_registry
    test_tilexr_udma_alltoall_layout
    test_tilexr_udma_allreduce_layout
    test_tilexr_chip_map_sources
    test_tilexr_ipc_pid_mode_sources
    test_tilexr_udma
)

for bin in "${required_bins[@]}"; do
    if [ ! -f "${INSTALL_DIR}/bin/${bin}" ]; then
        echo "ERROR: ${INSTALL_DIR}/bin/${bin} not found. Please run build.sh first."
        exit 1
    fi
done

echo "=========================================="
echo "Test 1: TileXR UDMA Transport Layout Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_udma_transport_layout"
TEST1_RESULT=$?
echo ""

echo "=========================================="
echo "Test 2: TileXR UDMA Registry Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_udma_registry"
TEST2_RESULT=$?
echo ""

echo "=========================================="
echo "Test 3: TileXR UDMA All-To-All Layout Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_udma_alltoall_layout"
TEST3_RESULT=$?
echo ""

echo "=========================================="
echo "Test 4: TileXR UDMA All-Reduce Layout Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_udma_allreduce_layout"
TEST4_RESULT=$?
echo ""

echo "=========================================="
echo "Test 5: TileXR Chip Map Source Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_chip_map_sources"
TEST5_RESULT=$?
echo ""

echo "=========================================="
echo "Test 6: TileXR IPC PID Mode Source Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_ipc_pid_mode_sources"
TEST6_RESULT=$?
echo ""

echo "=========================================="
echo "Test 7: TileXR Integration Tests (Single Process)"
echo "=========================================="
export RANK=0
export RANK_SIZE=1
"${INSTALL_DIR}/bin/test_tilexr_udma"
TEST7_RESULT=$?
echo ""

echo "=========================================="
echo "Test 8: TileXR Multi-Process Tests (MPI)"
echo "=========================================="

if command -v mpirun >/dev/null 2>&1; then
    NPU_COUNT=${TILEXR_ASCEND_DEV_NUM:-0}
    echo "Detected ${NPU_COUNT} NPU(s)"

    DEVICE_COUNT=0
    if [ -n "${TILEXR_TEST_DEVICES:-}" ]; then
        DEVICE_COUNT=$(echo "${TILEXR_TEST_DEVICES}" | awk -F, '{print NF}')
    fi

    if [ "${NPU_COUNT}" -ge 2 ] && [ "${DEVICE_COUNT}" -ge 2 ]; then
        echo "Running 2-rank test..."
        unset RANK
        unset RANK_SIZE
        mpirun -n 2 "${INSTALL_DIR}/bin/test_tilexr_udma"
        TEST8_RESULT=$?
    else
        echo "SKIP: Need at least 2 usable NPUs for multi-rank test"
        TEST8_RESULT=0
    fi
else
    echo "SKIP: mpirun not found, skipping multi-process tests"
    TEST8_RESULT=0
fi
echo ""

echo "=========================================="
echo "  Test Results Summary"
echo "=========================================="
echo "Test 1 (UDMA Layout):     $([ ${TEST1_RESULT} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 2 (UDMA Registry):   $([ ${TEST2_RESULT} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 3 (AllToAll Layout): $([ ${TEST3_RESULT} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 4 (AllReduce Layout): $([ ${TEST4_RESULT} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 5 (Chip Map):        $([ ${TEST5_RESULT} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 6 (IPC PID Mode):    $([ ${TEST6_RESULT} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 7 (TileXR Single):   $([ ${TEST7_RESULT} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 8 (TileXR Multi):    $([ ${TEST8_RESULT} -eq 0 ] && echo 'PASS' || echo 'SKIP/FAIL')"
echo "=========================================="

if [ ${TEST1_RESULT} -ne 0 ] || [ ${TEST2_RESULT} -ne 0 ] || [ ${TEST3_RESULT} -ne 0 ] ||
   [ ${TEST4_RESULT} -ne 0 ] || [ ${TEST5_RESULT} -ne 0 ] || [ ${TEST6_RESULT} -ne 0 ] ||
   [ ${TEST7_RESULT} -ne 0 ] || [ ${TEST8_RESULT} -ne 0 ]; then
    exit 1
fi

exit 0
