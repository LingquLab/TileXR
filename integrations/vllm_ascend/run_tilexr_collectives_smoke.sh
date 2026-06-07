#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_SIZE="${1:-2}"
COUNT="${2:-16}"
FIRST_NPU="${3:-0}"
INSTALL_PREFIX="${4:?usage: $0 rank_size count first_npu install_prefix op dtype}"
OP="${5:-allgather}"
DTYPE="${6:-int32}"
TIMEOUT_SEC="${TILEXR_VLLM_SMOKE_TIMEOUT_SEC:-600}"
PYTHON_BIN="${TILEXR_VLLM_SMOKE_PYTHON:-python3}"
EXTRA_PYTHONPATH="${TILEXR_VLLM_SMOKE_PYTHONPATH:-}"

if [[ ! "${RANK_SIZE}" =~ ^[0-9]+$ || "${RANK_SIZE}" -le 0 ]]; then
  echo "ERROR: rank_size must be a positive integer" >&2
  exit 2
fi
if [[ ! "${COUNT}" =~ ^[0-9]+$ || "${COUNT}" -le 0 ]]; then
  echo "ERROR: count must be a positive integer" >&2
  exit 2
fi
case "${OP}" in
  allgather|alltoall|allreduce|reducescatter|broadcast) ;;
  *)
    echo "ERROR: op must be allgather, alltoall, allreduce, reducescatter, or broadcast" >&2
    exit 2
    ;;
esac
if [[ "${DTYPE}" != "int32" && "${DTYPE}" != "fp16" ]]; then
  echo "ERROR: dtype must be int32 or fp16" >&2
  exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python command not found: ${PYTHON_BIN}" >&2
  exit 2
fi
python_probe="$("${PYTHON_BIN}" -c 'import sys; print("tilexr_python_preflight_ok")' 2>/dev/null || true)"
if [[ "${python_probe}" != "tilexr_python_preflight_ok" ]]; then
  echo "ERROR: Python command failed interpreter preflight: ${PYTHON_BIN}" >&2
  exit 2
fi

echo "TileXR vllm collectives smoke"
echo "  python: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)' 2>/dev/null || printf '%s' "${PYTHON_BIN}")"
echo "  rank_size: ${RANK_SIZE}"
echo "  op: ${OP}"
echo "  dtype: ${DTYPE}"

if [[ -n "${EXTRA_PYTHONPATH}" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}:${EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
fi
export TILEXR_INSTALL_PREFIX="${INSTALL_PREFIX}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"

pids=()
tail_logs() {
  local rank log
  for ((rank = 0; rank < RANK_SIZE; rank++)); do
    log="tilexr_vllm_collectives_${OP}_${DTYPE}_rank${rank}.log"
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

for ((rank = 0; rank < RANK_SIZE; rank++)); do
  log="tilexr_vllm_collectives_${OP}_${DTYPE}_rank${rank}.log"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_collectives.py" \
    --rank-size "${RANK_SIZE}" \
    --rank "${rank}" \
    --first-npu "${FIRST_NPU}" \
    --count "${COUNT}" \
    --dtype "${DTYPE}" \
    --op "${OP}" \
    --install-prefix "${INSTALL_PREFIX}" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

sleep "${TIMEOUT_SEC}" >/dev/null 2>&1 &
watchdog_pid="$!"

trap 'echo "ERROR: interrupted; killing remaining ranks" >&2; kill "${watchdog_pid}" 2>/dev/null || true; wait "${watchdog_pid}" 2>/dev/null || true; kill_remaining_children; tail_logs; exit 130' INT TERM

completed_count=0
while (( completed_count < RANK_SIZE )); do
  if wait -n; then
    if ! kill -0 "${watchdog_pid}" 2>/dev/null; then
      echo "ERROR: timed out after ${TIMEOUT_SEC}s" >&2
      kill_remaining_children
      tail_logs
      trap - INT TERM
      exit 124
    fi
    completed_count=$((completed_count + 1))
  else
    rc="$?"
    echo "ERROR: rank process exited with status ${rc}" >&2
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
    kill_remaining_children
    tail_logs
    trap - INT TERM
    exit 1
  fi
done

kill "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true
trap - INT TERM
echo "PASS TileXR vllm collectives smoke rank_size=${RANK_SIZE} op=${OP} dtype=${DTYPE}"
