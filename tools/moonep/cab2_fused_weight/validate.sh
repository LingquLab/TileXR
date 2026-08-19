#!/usr/bin/env bash
set -euo pipefail

readonly source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly validation_root="${TILEXR_CAB2_VALIDATION_ROOT:-/home/h00580772/tilexr_validation/cab2_fused_weight}"
readonly build_dir="${validation_root}/build"
readonly install_dir="${validation_root}/install"
readonly log_dir="${validation_root}/logs"
readonly source_hostfile="${source_dir}/tools/moonep/cab2_fused_weight/hosts.txt"
readonly cann_path="${TILEXR_CANN_PATH:-/home/pkg/b131/cann-9.1.0}"
readonly cmake_bin="${TILEXR_CMAKE_BIN:-/usr/bin/cmake}"
readonly mpi_home="${TILEXR_MPI_HOME:-/home/ltl/mpich-4.1.3}"
readonly hccl_test_home="${cann_path}/tools/hccl_test"

mkdir -p "${build_dir}" "${install_dir}" "${log_dir}"
readonly status_file="${validation_root}/validate.status"
trap 'status=$?; printf "%d\n" "${status}" >"${status_file}"' EXIT
rm -f "${status_file}"
test -x "${cmake_bin}"
test -x "${hccl_test_home}/bin/all_reduce_test"
export PATH="$(dirname "${cmake_bin}"):${mpi_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${mpi_home}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

make_hostfile() {
    local world=$1
    local output=$2
    local remaining=${world}
    : >"${output}"
    while IFS=: read -r host slots; do
        (( remaining > 0 )) || break
        local used=${slots}
        (( used <= remaining )) || used=${remaining}
        printf '%s:%d\n' "${host}" "${used}" >>"${output}"
        remaining=$((remaining - used))
    done <"${source_hostfile}"
    (( remaining == 0 ))
}

wait_for_available_npus() {
    local world=$1
    local hostfile="${log_dir}/hosts_${world}p.txt"
    make_hostfile "${world}" "${hostfile}"
    local attempt host blockers
    for attempt in $(seq 0 8); do
        blockers=""
        while IFS=: read -r host _slots; do
            local found npu_info
            if ! npu_info=$(ssh -o BatchMode=yes -o ConnectTimeout=10 \
                    root@"${host}" npu-smi info); then
                blockers+=$'\n'"${host}: unable to query npu-smi"
                continue
            fi
            found=$(awk -F'|' '
                /Process id/ && /Process name/ { in_process_table = 1; next }
                in_process_table && /^\|/ {
                    pid = $3
                    name = $4
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
                    if (pid ~ /^[0-9]+$/ && name !~ /^tilexr_/) {
                        print pid, name
                    }
                }
            ' <<<"${npu_info}")
            if [[ -n "${found}" ]]; then
                blockers+=$'\n'"${host}: ${found}"
            fi
        done <"${hostfile}"
        if [[ -z "${blockers}" ]]; then
            return 0
        fi
        if (( attempt == 8 )); then
            printf 'NPU validation blocked for 120s:%s\n' "${blockers}" >&2
            return 75
        fi
        printf 'NPU validation busy; retry %d/8 in 15s:%s\n' \
            "$((attempt + 1))" "${blockers}" >&2
        sleep 15
    done
}

run_hccl_baseline() {
    local world=$1
    local hostfile="${log_dir}/hccl_hosts_${world}p.txt"
    local log_file="${log_dir}/hccl_aiv_${world}p_$(date +%Y%m%d_%H%M%S).log"
    make_hostfile "${world}" "${hostfile}"
    wait_for_available_npus "${world}"
    source "${cann_path}/set_env.sh" >/dev/null
    export PATH="${mpi_home}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    export LD_LIBRARY_PATH="${mpi_home}/lib:${cann_path}/aarch64-linux/lib64:${cann_path}/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
    export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300
    export HCCL_SOCKET_FAMILY=AF_INET
    export HCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth5,enp35s0f2,mng
    export HCCL_HOST_SOCKET_PORT_RANGE=auto ASCEND_GLOBAL_LOG_LEVEL=3
    local per_host=${world}
    (( per_host <= 8 )) || per_host=8
    timeout --signal=TERM --kill-after=30 600 \
        "${mpi_home}/bin/mpirun" -f "${hostfile}" -n "${world}" \
        "${hccl_test_home}/bin/all_reduce_test" \
        -a aiv_only -p "${per_host}" -b 8K -e 8K \
        -n 1 -w 0 -d fp32 -o sum -c 2 |& tee "${log_file}"
    if grep -Eq 'hccl interface return err|hccl_op_base execute failed|Result.*failed' \
            "${log_file}" ||
        ! grep -Eq '\|[[:space:]]*success[[:space:]]*$' "${log_file}"; then
        echo "HCCL ${world}P AIV baseline failed: ${log_file}" >&2
        return 1
    fi
    printf 'HCCL_AIV_%sP_LOG=%s\n' "${world}" "${log_file}"
}

run_fused_weight() {
    local world=$1
    local hostfile="${log_dir}/combine_hosts_${world}p.txt"
    local log_file="${log_dir}/combine_fused_weight_${world}p_$(date +%Y%m%d_%H%M%S).log"
    make_hostfile "${world}" "${hostfile}"
    wait_for_available_npus "${world}"
    bash "${source_dir}/tools/moonep/run_combine_v2_perf_multihost.sh" \
        --hostfile "${hostfile}" \
        --install-dir "${install_dir}" \
        --cann-path "${cann_path}" \
        --ssh-user root \
        --bs 128 \
        --warmup 1 \
        --iterations 3 \
        --experts "$((world * 2))" \
        --hidden-size 3584 \
        --comm-domain 241 \
        --comm-id "141.61.52.35:$((18000 + world))" \
        --timeout 900 \
        --reduce-hidden \
        --fused-weight \
        --log-file "${log_file}"
    local passed
    passed=$(grep -Ec 'correctness=passed.*weight_correctness=passed|weight_correctness=passed.*correctness=passed' \
        "${log_file}" || true)
    if (( passed < world )) || grep -Eq 'correctness=(failed|self_only_failed)|weight_correctness=failed' \
            "${log_file}"; then
        echo "Combine V2 ${world}P fused-weight validation failed: ${log_file}" >&2
        return 1
    fi
    printf 'COMBINE_FUSED_WEIGHT_%sP_LOG=%s\n' "${world}" "${log_file}"
}

source "${cann_path}/set_env.sh" >/dev/null
bash "${source_dir}/tools/moonep/build_combine_v2_perf.sh" \
    --source-dir "${source_dir}" \
    --build-dir "${build_dir}" \
    --install-dir "${install_dir}" \
    --cann-path "${cann_path}" \
    --jobs 16 |& tee "${log_dir}/build_latest.log"

"${cmake_bin}" --build "${build_dir}" --parallel 16
"${cmake_bin}" --build "${build_dir}" --target test
bash "${source_dir}/tools/moonep/sync_combine_v2_perf_runtime.sh" \
    --hostfile "${source_hostfile}" \
    --install-dir "${install_dir}" \
    --ssh-user root |& tee "${log_dir}/sync_runtime_latest.log"

for world in 8 32 64; do
    run_hccl_baseline "${world}"
    run_fused_weight "${world}"
done
