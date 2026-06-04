#!/bin/bash
#
# Run TileXR comm tests
#

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT="${SCRIPT_DIR}/../.."
INSTALL_DIR="${SCRIPT_DIR}/install"

source "${TILEXR_ROOT}/scripts/common_env.sh"

if [ ! -f "${INSTALL_DIR}/bin/test_tilexr_log" ] ||
   [ ! -f "${INSTALL_DIR}/bin/test_tilexr_log_spdlog_compile" ]; then
    echo "ERROR: Test binaries not found. Please run build.sh first."
    exit 1
fi

echo "=========================================="
echo "Test 1: TileXR Log Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_log"
TEST1_RESULT=$?
echo ""

echo "=========================================="
echo "Test 2: TileXR Optional Spdlog Compile Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_log_spdlog_compile"
TEST2_RESULT=$?
echo ""

echo "=========================================="
echo "  Test Results Summary"
echo "=========================================="
echo "Test 1 (TileXR Log):      $([ $TEST1_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 2 (Spdlog compile):  $([ $TEST2_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "=========================================="

if [ $TEST1_RESULT -ne 0 ] || [ $TEST2_RESULT -ne 0 ]; then
    exit 1
fi

exit 0
