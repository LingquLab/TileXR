#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "usage: $0 <rank-size> <rank> <device> <comm-address> <control-address> <evidence-dir> <phase>" >&2
    exit 2
fi

rank_size="$1"
rank="$2"
device="$3"
comm_address="$4"
control_address="$5"
evidence_dir="$6"
phase="$7"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tilexr_root="$(cd "${script_dir}/../../.." && pwd)"
install_dir="${tilexr_root}/work/install-plan-multirank"
binary="${tilexr_root}/work/build-plan-multirank-test/test_tilexr_ep_plan_multirank"

source "${tilexr_root}/scripts/common_env.sh"
export TILEXR_COMM_ID="${comm_address}"
export TILEXR_PLAN_CONTROL_ADDR="${control_address}"
export RANK_SIZE="${rank_size}"
export RANK="${rank}"
export DEVICE_ID="${device}"
export TILEXR_PLAN_DEBUG="${TILEXR_PLAN_DEBUG:-1}"
export LD_LIBRARY_PATH="${install_dir}/lib64:${install_dir}/lib:${ASCEND_HOME_PATH}/${TILEXR_OS_ARCH}-linux/lib64:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

mkdir -p "${evidence_dir}"
log_file="${evidence_dir}/${phase}-rank-${rank}.log"
exit_file="${evidence_dir}/${phase}-rank-${rank}.exit"
pid_file="${evidence_dir}/${phase}-rank-${rank}.pid"
meta_file="${evidence_dir}/${phase}-rank-${rank}.meta"

if [[ ! -x "${binary}" ]]; then
    echo "missing binary: ${binary}" >"${log_file}"
    echo 127 >"${exit_file}"
    exit 127
fi

{
    echo "host=$(hostname)"
    echo "rank_size=${rank_size}"
    echo "rank=${rank}"
    echo "device=${device}"
    echo "comm_address=${comm_address}"
    echo "control_address=${control_address}"
    echo "binary=${binary}"
    echo "started_at=$(date -Iseconds)"
} >"${meta_file}"

echo "$$" >"${pid_file}"
child_pid=""
cleanup_child() {
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
}
trap cleanup_child INT TERM EXIT

set +e
"${binary}" "${rank_size}" "${rank}" "${device}" >"${log_file}" 2>&1 &
child_pid="$!"
echo "${child_pid}" >>"${pid_file}"
wait "${child_pid}"
rc="$?"
child_pid=""
set -e
trap - INT TERM EXIT

echo "${rc}" >"${exit_file}"
{
    echo "finished_at=$(date -Iseconds)"
    echo "exit_code=${rc}"
} >>"${meta_file}"
exit "${rc}"
