#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TILEXR_UDMA_QP_ROUTE_SPEC=port_count:6,port_count:6,port_count:2
if [[ -x "${SCRIPT_DIR}/tilexr_moonep_flow_demo" ]]; then
    TILEXR_INSTALL_PREFIX="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    TILEXR_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    TILEXR_INSTALL_PREFIX="${TILEXR_ROOT}/install"
fi

rank_size="${1:-8}"
s="${2:-8}"
k="${3:-2}"
experts="${4:-$((rank_size * 2))}"
hidden="${5:-32}"
physical_device_count="${6:-$((rank_size < 8 ? rank_size : 8))}"

if ((rank_size < 4 || rank_size > 128 || physical_device_count <= 0 ||
     s <= 0 || k < 2 || experts <= 0 || hidden <= 0 || hidden % 32 != 0 ||
     experts % rank_size != 0)); then
    echo "invalid rank/dimension configuration" >&2
    exit 2
fi

ranks_per_device=$(((rank_size + physical_device_count - 1) / physical_device_count))
oversubscribed=false
if ((rank_size > physical_device_count)); then
    oversubscribed=true
fi
if [[ -z "${TILEXR_MOONEP_PLANNER_BLOCK_DIM:-}" ]]; then
    block_dim=$((64 / ranks_per_device))
    if ((block_dim < rank_size)); then
        echo "computed Planner blockDim is below logical rank count" >&2
        exit 2
    fi
    export TILEXR_MOONEP_PLANNER_BLOCK_DIM="${block_dim}"
fi

export TILEXR_PHYSICAL_DEVICE_COUNT="${physical_device_count}"
export TILEXR_COMM_ID="${TILEXR_COMM_ID:-127.0.0.1:10201}"
export TILEXR_ENABLE_UDMA=1
export LD_LIBRARY_PATH="${TILEXR_INSTALL_PREFIX}/lib64:${TILEXR_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

binary="${TILEXR_MOONEP_FLOW_BIN:-${TILEXR_INSTALL_PREFIX}/bin/tilexr_moonep_flow_demo}"
if [[ ! -x "${binary}" ]]; then
    echo "native flow demo not found or not executable: ${binary}" >&2
    exit 2
fi

log_dir="${TILEXR_MOONEP_FLOW_LOG_DIR:-${SCRIPT_DIR}/../logs/flow_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${log_dir}"
echo "logical_ranks=${rank_size} physical_devices=${physical_device_count} ranks_per_device=${ranks_per_device} oversubscribed=${oversubscribed} block_dim=${TILEXR_MOONEP_PLANNER_BLOCK_DIM} prefetch_transport=udma_registered torch_validated=false transport_performance_valid=false" |
    tee "${log_dir}/run_metadata.txt"

pids=()
cleanup_children()
{
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup_children EXIT INT TERM

for rank in $(seq 0 $((rank_size - 1))); do
    device=$((rank % physical_device_count))
    "${binary}" "${rank_size}" "${rank}" "${device}" "${s}" "${k}" "${experts}" "${hidden}" >"${log_dir}/rank_${rank}.log" 2>&1 &
    pids+=("$!")
done

status=0
for rank in $(seq 0 $((rank_size - 1))); do
    if ! wait "${pids[$rank]}"; then
        status=1
    fi
done
trap - EXIT INT TERM

for rank in $(seq 0 $((rank_size - 1))); do
    echo "----- rank ${rank} -----"
    tail -n 100 "${log_dir}/rank_${rank}.log"
done
echo "logs=${log_dir}"
exit "${status}"
