#!/usr/bin/env bash
set -euo pipefail

readonly remote_root=/home/h00580772/tilexr_combine_v2_cab0_2_fast
readonly source_dir=${remote_root}
readonly install_dir=${remote_root}/install
readonly log_dir=${remote_root}/logs
readonly hostfile=${remote_root}/hostfile
readonly cann_path=/home/pkg/b131/cann-9.1.0
readonly log_file=${log_dir}/combine_v2_128p_noprofile_$(date +%Y%m%d_%H%M%S).log

test -f "${hostfile}"
mkdir -p "${log_dir}"

bash "${source_dir}/tools/moonep/run_combine_v2_perf_multihost.sh" \
    --hostfile "${hostfile}" \
    --install-dir "${install_dir}" \
    --cann-path "${cann_path}" \
    --ssh-user root \
    --bs 8192 \
    --warmup 20 \
    --iterations 80 \
    --experts 256 \
    --hidden-size 3584 \
    --comm-domain 141 \
    --comm-id '141.61.53.150:12067' \
    --timeout 1200 \
    --log-file "${log_file}"

printf 'LOG_FILE=%s\n' "${log_file}"
