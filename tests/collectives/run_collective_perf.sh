#!/usr/bin/env bash
set -u

usage() {
  echo "Usage: $0 rank_size first_npu bin_dir [extra tilexr_collective_perf args...]" >&2
  echo "Example: $0 2 0 ./install/bin --op allgather --min-bytes 4 --max-bytes 4096 --datatype int32 --check 1" >&2
  echo "Set TILEXR_SKIP_IF_INSUFFICIENT_NPUS=1 to skip cleanly when npu-smi reports fewer devices." >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

rank_size="${1:-2}"
first_npu="${2:-0}"
bin_dir="${3:-./install/bin}"
shift $(( $# >= 3 ? 3 : $# ))
binary="${bin_dir}/tilexr_collective_perf"

if [[ ! -x "${binary}" ]]; then
  echo "ERROR: ${binary} is not executable" >&2
  exit 1
fi

available_npus=""
if command -v npu-smi >/dev/null 2>&1; then
  available_npus="$(npu-smi info -l 2>/dev/null | sed -n 's/.*Total Count[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "${available_npus}" ]]; then
    available_npus="$(npu-smi info -l 2>/dev/null | grep -Ec '^[[:space:]]*[0-9]+[[:space:]]+')"
  fi
fi
if [[ -z "${available_npus}" && -n "${TILEXR_AVAILABLE_NPUS:-}" ]]; then
  available_npus="${TILEXR_AVAILABLE_NPUS}"
fi

required_npus=$((first_npu + rank_size))
if [[ -n "${available_npus}" && "${available_npus}" =~ ^[0-9]+$ && "${available_npus}" -lt "${required_npus}" ]]; then
  message="insufficient NPUs: required ${required_npus}, available ${available_npus}"
  if [[ "${TILEXR_SKIP_IF_INSUFFICIENT_NPUS:-0}" == "1" || "${TILEXR_SKIP_IF_INSUFFICIENT_NPUS:-}" == "true" ]]; then
    echo "SKIP: ${message}"
    exit 0
  fi
  echo "ERROR: ${message}" >&2
  exit 1
fi

pids=()
for ((rank = 0; rank < rank_size; rank++)); do
  log="collective_perf_rank${rank}.log"
  "${binary}" --rank-size "${rank_size}" --rank "${rank}" --first-npu "${first_npu}" "$@" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  for ((rank = 0; rank < rank_size; rank++)); do
    log="collective_perf_rank${rank}.log"
    if [[ -f "${log}" ]]; then
      echo "===== ${log} =====" >&2
      tail -n 80 "${log}" >&2
    fi
  done
fi

exit "${status}"
