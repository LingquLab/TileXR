#!/bin/bash
#
# Run the TileXR 2-card directed UDMA P2P performance demo.
#

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UDMA_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
TILEXR_ROOT=$(cd "${UDMA_DIR}/../.." && pwd)
INSTALL_DIR="${UDMA_DIR}/install"

src_rank=${1:-0}
dst_rank=${2:-1}
min_bytes=${3:-4096}
max_bytes=${4:-16777216}
step_factor=${5:-2}
iters=${6:-20}
warmup_iters=${7:-5}
first_npu=${8:-0}
check=${9:-1}
transport=${10:-direct_urma}
block_dim=${11:-1}
traffic=${12:-unidir}

source "${TILEXR_ROOT}/scripts/common_env.sh"

export TILEXR_COMM_ID=${TILEXR_COMM_ID:-127.0.0.1:10067}
export TILEXR_DEMO_NPUS=2
export TILEXR_DEMO_FIRST_NPU=${first_npu}
if [ "${transport}" = "direct_urma_multi_jetty" ] ||
    [ "${transport}" = "direct_urma_multi_jetty_parallel" ] ||
    [ "${transport}" = "direct_urma_multi_jetty_parallel_fixed_wqe" ]; then
    export TILEXR_UDMA_QP_NUM="${block_dim}"
else
    export TILEXR_UDMA_QP_NUM="${TILEXR_UDMA_QP_NUM:-1}"
fi
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${INSTALL_DIR}/lib64:${TILEXR_ROOT}/install/lib:${TILEXR_ROOT}/install/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

bin="${INSTALL_DIR}/bin/tilexr_udma_demo"
if [ ! -x "${bin}" ]; then
    echo "ERROR: ${bin} not found. Run: cd ${UDMA_DIR} && bash build.sh"
    exit 1
fi

log_dir="${UDMA_DIR}/logs/tilexr_udma_p2p_perf_$(date +%Y%m%d_%H%M%S)_${transport}_${traffic}_bd${block_dim}_${src_rank}to${dst_rank}"
csv_path="${log_dir}/p2p_perf.csv"
mkdir -p "${log_dir}"

echo "=========================================="
echo "  TileXR UDMA P2P Performance Demo"
echo "=========================================="
echo "Binary:        ${bin}"
echo "Direction:     ${src_rank}->${dst_rank}"
echo "Rank size:     2"
echo "Min bytes:     ${min_bytes}"
echo "Max bytes:     ${max_bytes}"
echo "Step factor:   ${step_factor}"
echo "Iters:         ${iters}"
echo "Warmup iters:  ${warmup_iters}"
echo "First NPU:     ${first_npu}"
echo "Check:         ${check}"
echo "Transport:     ${transport}"
echo "Block dim:     ${block_dim}"
echo "Traffic:       ${traffic}"
echo "UDMA QP num:   ${TILEXR_UDMA_QP_NUM}"
echo "TILEXR_COMM_ID:${TILEXR_COMM_ID}"
echo "Log dir:       ${log_dir}"
echo "CSV:           ${csv_path}"
echo "=========================================="

pids=()
for rank in 0 1; do
    log_file="${log_dir}/rank_${rank}.log"
    echo "Starting rank ${rank}, log=${log_file}"
    RANK=${rank} RANK_SIZE=2 TILEXR_P2P_LOG_DIR="${log_dir}" TILEXR_P2P_CSV="${csv_path}" "${bin}" \
        2 "${rank}" 4 0 2 "${first_npu}" \
        "${src_rank}" "${dst_rank}" "${min_bytes}" "${max_bytes}" "${step_factor}" \
        "${iters}" "${warmup_iters}" "${check}" "${csv_path}" "${log_dir}" "${transport}" \
        "${block_dim}" "${traffic}" \
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
for rank in 0 1; do
    log_file="${log_dir}/rank_${rank}.log"
    echo "----- rank ${rank}: ${log_file} -----"
    tail -n 80 "${log_file}" || true
done

if [ -f "${csv_path}" ]; then
    echo "=========================================="
    echo "  CSV"
    echo "=========================================="
    cat "${csv_path}"
fi

exit "${ret}"
