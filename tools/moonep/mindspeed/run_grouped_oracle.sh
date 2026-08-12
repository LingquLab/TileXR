#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tilexr_home=${TILEXR_HOME:-$(cd "${script_dir}/../../.." && pwd)}
install_prefix=${TILEXR_INSTALL_PREFIX:-${tilexr_home}/install}
cann_env=${TILEXR_CANN_ENV:-/home/pkg/b131/cann/set_env.sh}
conda_sh=${TILEXR_CONDA_SH:-/home/miniconda3/etc/profile.d/conda.sh}
conda_env=${TILEXR_CONDA_ENV:-ai_moe_test}
output_root=${TILEXR_MOONEP_OUTPUT_ROOT:-${tilexr_home}/run/moonep/mindspeed}
run_dir=${output_root}/grouped_oracle_$(date +%Y%m%d-%H%M%S)

export LD_LIBRARY_PATH=
export PYTHONPATH=
unset ASCEND_HOME ASCEND_HOME_PATH ASCEND_AICPU_PATH ASCEND_OPP_PATH TOOLCHAIN_HOME
source "${cann_env}"
source "${conda_sh}"
conda activate "${conda_env}"
if [[ -n "${TILEXR_MOONEP_NATIVE_ENV:-}" ]]; then
    source "${TILEXR_MOONEP_NATIVE_ENV}"
fi

export PYTHONPATH="${tilexr_home}/integrations/moonep_torch:${tilexr_home}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${install_prefix}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TILEXR_INSTALL_PREFIX=${install_prefix}
export TILEXR_UDMA_QP_ROUTE_SPEC=${TILEXR_UDMA_QP_ROUTE_SPEC:-port_count:6,port_count:2}
export TILEXR_UDMA_ATTACH_EXISTING_RA=${TILEXR_UDMA_ATTACH_EXISTING_RA:-1}
unset TILEXR_MOONEP_DISPATCH_TRANSPORT
export TILEXR_MOONEP_DISPATCH_PEER_MODE=group
export TILEXR_MOONEP_DISPATCH_GROUP_WIDTH=16
export TILEXR_MOONEP_DUMP_DFX_ON_ERROR=${TILEXR_MOONEP_DUMP_DFX_ON_ERROR:-1}
export TILEXR_ORACLE_SOURCE_ROOT=${tilexr_home}
export TILEXR_ORACLE_RUN_DIR=${run_dir}
export TILEXR_ORACLE_ITERATIONS=${TILEXR_ORACLE_ITERATIONS:-20}
export TILEXR_ORACLE_HIDDEN_SIZE=${TILEXR_ORACLE_HIDDEN_SIZE:-7168}
export TILEXR_ORACLE_ROUTE_MODE=${TILEXR_ORACLE_ROUTE_MODE:-model_skew}
export TILEXR_ORACLE_WITH_ROUTE_WEIGHTS=${TILEXR_ORACLE_WITH_ROUTE_WEIGHTS:-0}
export TILEXR_ORACLE_WITH_COMBINE=${TILEXR_ORACLE_WITH_COMBINE:-1}
export TILEXR_ORACLE_SWITCH_REGISTRATION=${TILEXR_ORACLE_SWITCH_REGISTRATION:-1}
export TILEXR_ORACLE_EXTRA_PLAN_COUNT=${TILEXR_ORACLE_EXTRA_PLAN_COUNT:-5}
export TILEXR_ORACLE_WITH_PREFETCH=${TILEXR_ORACLE_WITH_PREFETCH:-1}
export TILEXR_ORACLE_PROJECTION_SIZE=${TILEXR_ORACLE_PROJECTION_SIZE:-256}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-120}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-120}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}

mkdir -p "${run_dir}"
cd "${tilexr_home}"
exec timeout --signal=TERM --kill-after=20s 120s python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port "${TILEXR_ORACLE_MASTER_PORT:-45678}" \
    --use-env \
    "${script_dir}/grouped_urma_dispatch_oracle.py"
