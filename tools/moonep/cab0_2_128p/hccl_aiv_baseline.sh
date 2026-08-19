#!/usr/bin/env bash
set -euo pipefail

readonly remote_root=/home/h00580772/tilexr_combine_v2_cab0_2_fast
readonly source_hostfile=${remote_root}/hostfile
readonly cann_path=/home/pkg/b131/cann-9.1.0
readonly hccl_test_home=${cann_path}/tools/hccl_test
readonly mpi_home=/home/ltl/mpich-4.1.3
readonly world=${1:-128}
readonly host_offset=${2:-0}
readonly hostfile=${remote_root}/logs/hccl_hostfile_${world}p_o${host_offset}
readonly log_file=${remote_root}/logs/hccl_aiv_allreduce_${world}p_o${host_offset}_$(date +%Y%m%d_%H%M%S).log

case "${world}" in
    2|8|16|32|64|128) ;;
    *) echo "world size must be one of 2, 8, 16, 32, 64, or 128" >&2; exit 2 ;;
esac
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

source "${cann_path}/set_env.sh" >/dev/null
export PATH="${mpi_home}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
export LD_LIBRARY_PATH="${mpi_home}/lib:${cann_path}/aarch64-linux/lib64:${cann_path}/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_SOCKET_FAMILY=AF_INET
export HCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth5,enp35s0f2,mng
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export ASCEND_GLOBAL_LOG_LEVEL=3

npus_per_host=${world}
(( npus_per_host <= 8 )) || npus_per_host=8
timeout --signal=TERM --kill-after=30 600 \
    "${mpi_home}/bin/mpirun" -f "${hostfile}" -n "${world}" \
    "${hccl_test_home}/bin/all_reduce_test" \
    -a aiv_only -p "${npus_per_host}" -b 8K -e 8K \
    -n 1 -w 0 -d fp32 -o sum -c 2 |& tee "${log_file}"

if grep -Eq 'hccl interface return err|hccl_op_base execute failed' "${log_file}" ||
    ! grep -Eq '\|[[:space:]]*success[[:space:]]*$' "${log_file}"; then
    echo "HCCL AIV baseline correctness failed; see ${log_file}" >&2
    exit 1
fi

printf 'HCCL_AIV_BASELINE_LOG=%s\n' "${log_file}"
