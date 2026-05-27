#!/bin/bash
#
# 构建 UDMA 测试
#

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT="${SCRIPT_DIR}/../.."

# 加载环境
source "${TILEXR_ROOT}/scripts/common_env.sh"

export ARCH="${TILEXR_OS_ARCH}"

echo "=========================================="
echo "  Building UDMA Tests"
echo "=========================================="

# 创建构建目录
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}"

cd "${BUILD_DIR}"

# 配置
cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" ..

# 构建
make -j$(nproc)

# 安装
make install

echo ""
echo "=========================================="
echo "  Build Complete"
echo "=========================================="
echo "Test binaries installed to: ${INSTALL_DIR}/bin"
echo ""
echo "Available tests:"
echo "  - test_shmem_api       : shmem API unit tests"
echo "  - test_tilexr_udma     : TileXR integration tests"
echo ""
echo "Run tests with:"
echo "  bash run_tests.sh"
echo "=========================================="
