#!/usr/bin/env bash
set -euo pipefail

devices=
log=/dev/null
while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=${2:?--devices requires a value}; shift 2 ;;
        --log) log=${2:?--log requires a value}; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ ! "${devices}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--devices must be a positive integer" >&2
    exit 2
fi

mkdir -p "$(dirname "${log}")"
accelerator_process_pattern='pretrain_gpt.py|torch.distributed.launch|hccl_test/bin/|all_reduce_test|alltoallv_test|tilexr_udma_dem'
accelerator_processes=$(pgrep -fc "${accelerator_process_pattern}" || true)
if timeout 15s npu-smi info >"${log}" 2>&1; then
    idle=$(grep -c 'No running processes found in NPU' "${log}" || true)
    if [[ "${idle}" -eq "${devices}" && "${accelerator_processes}" -eq 0 ]]; then
        echo "${idle}"
        exit 0
    fi
    printf 'probe=full reported_idle=%s expected=%s accelerator_processes=%s; checking live ownership\n' \
        "${idle}" "${devices}" "${accelerator_processes}" >>"${log}"
fi

list_log=${log}.list
if ! timeout 5s npu-smi info -l >"${list_log}" 2>&1; then
    printf 'probe=fallback list=failed\n' >>"${log}"
    echo 0
    exit 0
fi
listed=$(sed -n 's/.*Total Count[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
    "${list_log}" | head -1)
device_nodes=$(find /dev -maxdepth 1 -type c -name 'davinci[0-9]*' 2>/dev/null | wc -l)
device_users=0
if command -v fuser >/dev/null 2>&1; then
    device_users=$(fuser /dev/davinci[0-9]* 2>/dev/null | wc -w || true)
else
    device_users=-1
fi
printf 'probe=fallback listed=%s device_nodes=%s device_users=%s accelerator_processes=%s\n' \
    "${listed:-unknown}" "${device_nodes}" "${device_users}" "${accelerator_processes}" \
    >>"${log}"
if [[ "${listed:-0}" -eq "${devices}" && "${device_nodes}" -eq "${devices}" && \
      "${device_users}" -eq 0 && "${accelerator_processes}" -eq 0 ]]; then
    echo "${devices}"
else
    echo 0
fi
