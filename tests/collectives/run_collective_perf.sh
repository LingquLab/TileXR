#!/usr/bin/env bash
set -u

usage() {
  echo "Usage: $0 rank_size first_npu bin_dir [extra tilexr_collective_perf args...]" >&2
  echo "Example: $0 2 0 ./install/bin --op allgather --min-bytes 4 --max-bytes 4096 --datatype int32 --check 1" >&2
  echo "Set TILEXR_SKIP_IF_INSUFFICIENT_NPUS=1 to skip cleanly when npu-smi reports fewer devices." >&2
  echo "Set TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=N to override the per-launch timeout (default: 600)." >&2
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
timeout_sec="${TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC:-600}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile_enabled="0"
profile_dir=""
profile_ai_prompt="0"
profile_sample_every="1"
warmup_iters="5"
measured_iters="20"

parse_profile_args() {
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --profile)
        if (( i + 1 < ${#args[@]} )); then profile_enabled="${args[$((i + 1))]}"; fi
        ;;
      --profile-dir)
        if (( i + 1 < ${#args[@]} )); then profile_dir="${args[$((i + 1))]}"; fi
        ;;
      --profile-ai-prompt)
        if (( i + 1 < ${#args[@]} )); then profile_ai_prompt="${args[$((i + 1))]}"; fi
        ;;
      --profile-sample-every)
        if (( i + 1 < ${#args[@]} )); then profile_sample_every="${args[$((i + 1))]}"; fi
        ;;
      --warmup-iters)
        if (( i + 1 < ${#args[@]} )); then warmup_iters="${args[$((i + 1))]}"; fi
        ;;
      --iters)
        if (( i + 1 < ${#args[@]} )); then measured_iters="${args[$((i + 1))]}"; fi
        ;;
    esac
  done
}

write_profile_report_if_enabled() {
  if [[ "${profile_enabled}" != "1" && "${profile_enabled}" != "true" ]]; then
    return 0
  fi
  local root="${profile_dir:-run/prof/collectives}"
  local helper="${script_dir}/tilexr_collective_profile_report.py"
  local python_cmd=""
  if command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
  elif command -v python >/dev/null 2>&1; then
    python_cmd="python"
  fi
  if [[ -z "${python_cmd}" ]]; then
    echo "WARNING: python3 not found; skipping aggregate profile report" >&2
    return 0
  fi
  if [[ ! -f "${helper}" ]]; then
    echo "WARNING: ${helper} not found; skipping aggregate profile report" >&2
    return 0
  fi
  local prompt_args=()
  if [[ "${profile_ai_prompt}" == "1" || "${profile_ai_prompt}" == "true" ]]; then
    prompt_args+=(--emit-ai-prompt)
  fi
  "${python_cmd}" "${helper}" "${root}" \
    --warmup-iters "${warmup_iters}" \
    --iters "${measured_iters}" \
    --profile-sample-every "${profile_sample_every}" \
    "${prompt_args[@]}"
}

parse_profile_args "$@"

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
    log="collective_perf_rank${rank}.log"
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
  log="collective_perf_rank${rank}.log"
  "${binary}" --rank-size "${rank_size}" --rank "${rank}" --first-npu "${first_npu}" "$@" \
    > "${log}" 2>&1 &
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

if ! write_profile_report_if_enabled; then
  echo "ERROR: aggregate profile report generation failed" >&2
  exit 1
fi

exit 0
