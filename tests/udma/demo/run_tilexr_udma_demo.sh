#!/bin/bash
#
# Run the TileXR UDMA communication demo.
#

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UDMA_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
TILEXR_ROOT=$(cd "${UDMA_DIR}/../.." && pwd)
INSTALL_DIR="${UDMA_DIR}/install"

test_type=${1:-0}
rank_size=${2:-2}
elements_per_rank=${3:-16}
npu_count=${4:-${rank_size}}
first_npu=${5:-0}

source "${TILEXR_ROOT}/scripts/common_env.sh"

export TILEXR_COMM_ID=${TILEXR_COMM_ID:-127.0.0.1:10067}
export TILEXR_DEMO_NPUS=${npu_count}
export TILEXR_DEMO_FIRST_NPU=${first_npu}
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${TILEXR_ROOT}/install/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

bin="${INSTALL_DIR}/bin/tilexr_udma_demo"
if [ ! -x "${bin}" ]; then
    echo "ERROR: ${bin} not found. Run: cd ${UDMA_DIR} && bash build.sh"
    exit 1
fi

log_dir="${UDMA_DIR}/logs/tilexr_udma_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"

echo "=========================================="
echo "  TileXR UDMA Communication Demo"
echo "=========================================="
echo "Binary:            ${bin}"
echo "Test type:         ${test_type} (0=all-gather put, 1=put-signal)"
echo "Rank size:         ${rank_size}"
echo "Elements/rank:     ${elements_per_rank}"
echo "NPU count:         ${npu_count}"
echo "First NPU:         ${first_npu}"
echo "TILEXR_COMM_ID:    ${TILEXR_COMM_ID}"
echo "Log dir:           ${log_dir}"
echo "=========================================="

pids=()
for rank in $(seq 0 $((rank_size - 1))); do
    log_file="${log_dir}/rank_${rank}.log"
    echo "Starting rank ${rank}, log=${log_file}"
    RANK=${rank} RANK_SIZE=${rank_size} "${bin}" \
        "${rank_size}" "${rank}" "${test_type}" "${elements_per_rank}" "${npu_count}" "${first_npu}" \
        >"${log_file}" 2>&1 &
    pids+=("$!")
done

ret=0
for idx in "${!pids[@]}"; do
    pid=${pids[$idx]}
    rank=${idx}
    if wait "${pid}"; then
        echo "rank ${rank} finished successfully"
    else
        r=$?
        echo "rank ${rank} failed with exit code ${r}"
        ret=${r}
    fi
done

echo "=========================================="
echo "  Rank Log Tails"
echo "=========================================="
for rank in $(seq 0 $((rank_size - 1))); do
    log_file="${log_dir}/rank_${rank}.log"
    echo "----- rank ${rank}: ${log_file} -----"
    tail -n 80 "${log_file}" || true
done

exit "${ret}"
