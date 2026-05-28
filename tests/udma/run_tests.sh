#!/bin/bash
#
# 运行 UDMA 测试
#

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT="${SCRIPT_DIR}/../.."
INSTALL_DIR="${SCRIPT_DIR}/install"

# 加载环境
source "${TILEXR_ROOT}/scripts/common_env.sh"

# 设置 LD_LIBRARY_PATH：优先使用当前仓库刚编译安装的库，避免被 /usr/local/lib 中的旧库覆盖
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:/usr/local/lib:${LD_LIBRARY_PATH}"

echo "=========================================="
echo "  Running UDMA Tests"
echo "=========================================="
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

# 检查测试二进制是否存在
if [ ! -f "${INSTALL_DIR}/bin/test_tilexr_udma_registry" ] || [ ! -f "${INSTALL_DIR}/bin/test_tilexr_udma" ]; then
    echo "ERROR: Test binaries not found. Please run build.sh first."
    exit 1
fi

# 测试 1: TileXR UDMA registry 单元测试（host-only）
echo "=========================================="
echo "Test 1: TileXR UDMA Registry Unit Test"
echo "=========================================="
"${INSTALL_DIR}/bin/test_tilexr_udma_registry"
TEST1_RESULT=$?
echo ""

# 测试 2: TileXR 集成测试（单进程，单卡）
echo "=========================================="
echo "Test 2: TileXR Integration Tests (Single Process)"
echo "=========================================="
export RANK=0
export RANK_SIZE=1
"${INSTALL_DIR}/bin/test_tilexr_udma"
TEST2_RESULT=$?
echo ""

# 测试 3: TileXR 多进程测试（需要 mpirun）
echo "=========================================="
echo "Test 3: TileXR Multi-Process Tests (MPI)"
echo "=========================================="

# 检查是否有 mpirun
if command -v mpirun &> /dev/null; then
    # 检测可用的 NPU 数量
    NPU_COUNT=${TILEXR_ASCEND_DEV_NUM:-0}
    echo "Detected ${NPU_COUNT} NPU(s)"

    if [ "${NPU_COUNT}" -ge 2 ]; then
        echo "Running 2-rank test..."
        unset RANK
        unset RANK_SIZE
        mpirun -n 2 "${INSTALL_DIR}/bin/test_tilexr_udma"
        TEST3_RESULT=$?
    else
        echo "SKIP: Need at least 2 NPUs for multi-rank test"
        TEST3_RESULT=0
    fi
else
    echo "SKIP: mpirun not found, skipping multi-process tests"
    TEST3_RESULT=0
fi
echo ""

# 汇总结果
echo "=========================================="
echo "  Test Results Summary"
echo "=========================================="
echo "Test 1 (UDMA Registry):    $([ $TEST1_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 2 (TileXR Single):    $([ $TEST2_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 3 (TileXR Multi):     $([ $TEST3_RESULT -eq 0 ] && echo 'PASS' || echo 'SKIP/FAIL')"
echo "=========================================="

# 返回失败状态
if [ $TEST1_RESULT -ne 0 ] || [ $TEST2_RESULT -ne 0 ] || [ $TEST3_RESULT -ne 0 ]; then
    exit 1
fi

exit 0
