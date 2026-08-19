#!/usr/bin/env bash
set -euo pipefail

readonly remote_root=/home/h00580772/tilexr_combine_v2_cab0_2_fast
readonly source_dir=${remote_root}
readonly install_dir=${remote_root}/install
readonly source_hostfile=${remote_root}/hostfile
readonly cann_path=/home/pkg/b131/cann-9.1.0
readonly world=${1:-128}
readonly iterations=${2:-3}
readonly host_offset=${3:-0}
readonly hostfile=${remote_root}/logs/combine_v2_hostfile_${world}p_o${host_offset}
readonly log_file=${remote_root}/logs/combine_v2_credit_${world}p_o${host_offset}_$(date +%Y%m%d_%H%M%S).log

case "${world}" in
    2|8|16|32|64|128) ;;
    *) echo "world size must be one of 2, 8, 16, 32, 64, or 128" >&2; exit 2 ;;
esac
[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || {
    echo "iterations must be a positive integer" >&2
    exit 2
}
[[ "${host_offset}" =~ ^[0-9]+$ ]] || {
    echo "host offset must be a non-negative integer" >&2
    exit 2
}

mkdir -p "${remote_root}/logs"
remaining=${world}
host_index=0
: >"${hostfile}"
while IFS=: read -r host slots; do
    if (( host_index < host_offset )); then
        host_index=$((host_index + 1))
        continue
    fi
    (( remaining > 0 )) || break
    used=${slots}
    (( used <= remaining )) || used=${remaining}
    printf '%s:%d\n' "${host}" "${used}" >>"${hostfile}"
    remaining=$((remaining - used))
done <"${source_hostfile}"
(( remaining == 0 )) || { echo "hostfile has insufficient slots" >&2; exit 1; }
primary_host=$(head -n 1 "${hostfile}")
primary_host=${primary_host%%:*}

bash "${source_dir}/tools/moonep/run_combine_v2_perf_multihost.sh" \
    --hostfile "${hostfile}" \
    --install-dir "${install_dir}" \
    --cann-path "${cann_path}" \
    --ssh-user root \
    --bs 8192 \
    --warmup 0 \
    --iterations "${iterations}" \
    --experts $((world * 2)) \
    --hidden-size 3584 \
    --comm-domain 141 \
    --comm-id "${primary_host}:$((15000 + world))" \
    --timeout 600 \
    --log-file "${log_file}"

printf 'COMBINE_V2_CREDIT_LOG=%s\n' "${log_file}"
