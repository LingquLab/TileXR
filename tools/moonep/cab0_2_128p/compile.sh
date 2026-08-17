#!/usr/bin/env bash
set -eo pipefail

readonly remote_root=/home/h00580772/tilexr_combine_v2_cab0_2_fast
readonly source_dir=${remote_root}
readonly build_dir=${remote_root}/build
readonly install_dir=${remote_root}/install
readonly log_dir=${remote_root}/logs
readonly cann_path=/home/pkg/b131/cann-9.1.0
readonly cmake_bin=/home/h00580772/0803/tilexr_moonep_b87f_53_150/env/util/cmake/bin/cmake

mkdir -p "${build_dir}" "${install_dir}" "${log_dir}"
source "${cann_path}/set_env.sh" >/dev/null
set -u
test -x "${cmake_bin}"
export PATH="$(dirname "${cmake_bin}"):${PATH}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "${cmake_bin}")")/lib:${LD_LIBRARY_PATH:-}"

bash "${source_dir}/tools/moonep/build_combine_v2_perf.sh" \
    --source-dir "${source_dir}" \
    --build-dir "${build_dir}" \
    --install-dir "${install_dir}" \
    --cann-path "${cann_path}" \
    --jobs 16 |& tee "${log_dir}/build_latest.log"
