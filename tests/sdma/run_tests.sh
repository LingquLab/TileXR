#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL_DIR="${SCRIPT_DIR}/install"
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:${LD_LIBRARY_PATH:-}"

"${INSTALL_DIR}/bin/test_tilexr_sdma_metadata"

echo "TileXR SDMA unit tests passed"
