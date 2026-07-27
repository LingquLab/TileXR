#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TILEXR_ROOT="$(cd "${EP_DIR}/../.." && pwd)"
INSTALL_DIR="${EP_DIR}/install"

rank_size="${1:-2}"
if [[ $# -gt 0 ]]; then
    shift
fi
npu_count="${1:-${rank_size}}"
if [[ $# -gt 0 ]]; then
    shift
fi
first_npu="${1:-0}"
if [[ $# -gt 0 ]]; then
    shift
fi

: "${ASCEND_HOME_PATH:=}"
: "${LD_LIBRARY_PATH:=}"
source "${TILEXR_ROOT}/scripts/common_env.sh"

export TILEXR_COMM_ID="${TILEXR_COMM_ID:-127.0.0.1:10077}"
export TILEXR_DEMO_NPUS="${npu_count}"
export TILEXR_DEMO_FIRST_NPU="${first_npu}"
export LD_LIBRARY_PATH="${TILEXR_ROOT}/install/lib64:${TILEXR_ROOT}/install/lib:${INSTALL_DIR}/lib64:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

demo_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            export TILEXR_EP_DEMO_MODE="${2:?missing value for --mode}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --bs)
            export TILEXR_EP_DEMO_BS="${2:?missing value for --bs}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --h)
            export TILEXR_EP_DEMO_H="${2:?missing value for --h}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --k)
            export TILEXR_EP_DEMO_TOPK="${2:?missing value for --k}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --moe-expert-num)
            export TILEXR_EP_DEMO_MOE_EXPERT_NUM="${2:?missing value for --moe-expert-num}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --expert-token-nums-type)
            export TILEXR_EP_DEMO_EXPERT_TOKEN_NUMS_TYPE="${2:?missing value for --expert-token-nums-type}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --active-mask)
            export TILEXR_EP_DEMO_ACTIVE_MASK=1
            demo_args+=("$1")
            shift
            ;;
        --tp-world-size)
            export TILEXR_EP_DEMO_TP_WORLD_SIZE="${2:?missing value for --tp-world-size}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --tp-rank-id)
            export TILEXR_EP_DEMO_TP_RANK_ID="${2:?missing value for --tp-rank-id}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --tp-recv-counts)
            export TILEXR_EP_DEMO_TP_RECV_COUNTS=1
            demo_args+=("$1")
            shift
            ;;
        --shared-expert-num)
            export TILEXR_EP_DEMO_SHARED_EXPERT_NUM="${2:?missing value for --shared-expert-num}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --shared-expert-rank-num)
            export TILEXR_EP_DEMO_SHARED_EXPERT_RANK_NUM="${2:?missing value for --shared-expert-rank-num}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --quant-mode)
            export TILEXR_EP_DEMO_QUANT_MODE="${2:?missing value for --quant-mode}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --static-quant-scale)
            export TILEXR_EP_DEMO_STATIC_QUANT_SCALE="${2:?missing value for --static-quant-scale}"
            demo_args+=("$1" "$2")
            shift 2
            ;;
        --dump-window)
            export TILEXR_EP_DEMO_DUMP_WINDOW=1
            demo_args+=("$1")
            shift
            ;;
        *)
            demo_args+=("$1")
            shift
            ;;
    esac
done

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
        exec "${bin}" "${rank_size}" "${rank}" "${npu_count}" "${first_npu}" "${demo_args[@]}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
done

ret=0
for pid in "${pids[@]}"; do
    set +e
    wait "${pid}"
    status=$?
    set -e
    if [[ "${status}" -ne 0 && "${ret}" -eq 0 ]]; then
        ret="${status}"
    fi
done

for log in "${logs[@]}"; do
    echo "===== ${log} (last 120 lines) ====="
    tail -n 120 "${log}" || true
done

exit "${ret}"
