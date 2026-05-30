#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TILEXR_ROOT="$(cd "${EP_DIR}/../.." && pwd)"
INSTALL_DIR="${EP_DIR}/install"

rank_size="${1:-2}"
npu_count="${2:-${rank_size}}"
first_npu="${3:-0}"

source "${TILEXR_ROOT}/scripts/common_env.sh"

export TILEXR_COMM_ID="${TILEXR_COMM_ID:-127.0.0.1:10077}"
export TILEXR_DEMO_NPUS="${npu_count}"
export TILEXR_DEMO_FIRST_NPU="${first_npu}"
export LD_LIBRARY_PATH="${TILEXR_ROOT}/install/lib:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

bin="${INSTALL_DIR}/bin/tilexr_ep_dispatch_demo"
if [[ ! -x "${bin}" ]]; then
    echo "Missing demo binary: ${bin}" >&2
    echo "Build it with: cd ${EP_DIR} && bash build.sh full" >&2
    exit 1
fi

log_dir="${EP_DIR}/logs/tilexr_ep_dispatch_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"

pids=()
logs=()
for ((rank = 0; rank < rank_size; ++rank)); do
    log="${log_dir}/rank_${rank}.log"
    logs+=("${log}")
    (
        export RANK="${rank}"
        export RANK_SIZE="${rank_size}"
        exec "${bin}" "${rank_size}" "${rank}" "${npu_count}" "${first_npu}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
done

ret=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        ret=1
    fi
done

for log in "${logs[@]}"; do
    echo "===== ${log} (last 120 lines) ====="
    tail -n 120 "${log}" || true
done

exit "${ret}"
