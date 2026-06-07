#!/usr/bin/env bash
set -u

usage() {
  echo "Usage: $0 rank_size count first_npu bin_dir [allgather|alltoall|allreduce|reducescatter|broadcast|both] [extra correctness args...]" >&2
  echo "Set TILEXR_SKIP_IF_INSUFFICIENT_NPUS=1 to skip cleanly when npu-smi reports fewer devices." >&2
  echo "Set TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=N to override the per-launch timeout (default: 600)." >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

rank_size="${1:-2}"
count="${2:-16}"
first_npu="${3:-0}"
bin_dir="${4:-./install/bin}"
op="${5:-both}"
if (( $# > 5 )); then
  shift 5
else
  shift $#
fi
binary="${bin_dir}/test_tilexr_collectives_correctness"
timeout_sec="${TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC:-600}"

case "${op}" in
  allgather|alltoall|allreduce|reducescatter|broadcast|both)
    ;;
  *)
    echo "ERROR: op must be allgather, alltoall, allreduce, reducescatter, broadcast, or both" >&2
    usage
    exit 1
    ;;
esac

if [[ ! -x "${binary}" ]]; then
  echo "ERROR: ${binary} is not executable" >&2
  exit 1
fi
if [[ ! "${timeout_sec}" =~ ^[0-9]+$ || "${timeout_sec}" -le 0 ]]; then
  echo "ERROR: TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC must be a positive integer" >&2
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
tail_logs() {
  for ((rank = 0; rank < rank_size; rank++)); do
    log="collectives_correctness_rank${rank}.log"
    if [[ -f "${log}" ]]; then
      echo "===== ${log} =====" >&2
      tail -n 80 "${log}" >&2
    fi
  done
}

kill_remaining_children() {
  local pid
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 1
  for pid in "${pids[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

for ((rank = 0; rank < rank_size; rank++)); do
  log="collectives_correctness_rank${rank}.log"
  TILEXR_RANK_SIZE="${rank_size}" TILEXR_RANK="${rank}" TILEXR_COUNT="${count}" TILEXR_FIRST_NPU="${first_npu}" \
    "${binary}" --rank-size "${rank_size}" --rank "${rank}" --count "${count}" --first-npu "${first_npu}" --op "${op}" \
    "$@" > "${log}" 2>&1 &
  pids+=("$!")
done

sleep "${timeout_sec}" >/dev/null 2>&1 &
watchdog_pid="$!"

trap 'echo "ERROR: interrupted; killing remaining ranks" >&2; kill "${watchdog_pid}" 2>/dev/null || true; wait "${watchdog_pid}" 2>/dev/null || true; kill_remaining_children; tail_logs; exit 130' INT TERM

completed_count=0
status=0
while (( completed_count < rank_size )); do
  if wait -n; then
    if ! kill -0 "${watchdog_pid}" 2>/dev/null; then
      echo "ERROR: Timed out after ${timeout_sec}s; killing remaining ranks" >&2
      status=124
      kill_remaining_children
      tail_logs
      trap - INT TERM
      exit "${status}"
    fi
    completed_count=$((completed_count + 1))
    continue
  else
    rc="$?"
  fi

  if ! kill -0 "${watchdog_pid}" 2>/dev/null; then
    echo "ERROR: Timed out after ${timeout_sec}s; killing remaining ranks" >&2
    status=124
  else
    echo "ERROR: rank process exited with status ${rc}; killing remaining ranks" >&2
    status=1
  fi
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
  kill_remaining_children
  tail_logs
  trap - INT TERM
  exit "${status}"
done

kill "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true
trap - INT TERM

exit 0
