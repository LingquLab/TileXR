#!/usr/bin/env bash
set -euo pipefail

readonly remote_root=/home/h00580772/tilexr_combine_v2_cab0_2_fast
readonly source_dir=${remote_root}
readonly install_dir=${remote_root}/install
readonly log_dir=${remote_root}/logs
readonly hostfile=${remote_root}/hostfile

mkdir -p "${log_dir}"
printf '%s\n' \
    '141.61.53.150:8' \
    '141.61.53.83:8' \
    '141.61.53.142:8' \
    '141.61.53.138:8' \
    '141.61.53.110:8' \
    '141.61.53.106:8' \
    '141.61.53.146:8' \
    '141.61.53.98:8' \
    '141.61.52.35:8' \
    '141.61.52.39:8' \
    '141.61.52.43:8' \
    '141.61.52.47:8' \
    '141.61.52.75:8' \
    '141.61.52.79:8' \
    '141.61.52.83:8' \
    '141.61.52.87:8' >"${hostfile}"

bash "${source_dir}/tools/moonep/sync_combine_v2_perf_runtime.sh" \
    --hostfile "${hostfile}" \
    --install-dir "${install_dir}" \
    --ssh-user root |& tee "${log_dir}/sync_runtime_latest.log"
