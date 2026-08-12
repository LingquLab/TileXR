#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tilexr_home=${TILEXR_HOME:-$(cd "${script_dir}/../../.." && pwd)}
install_prefix=${TILEXR_INSTALL_PREFIX:-${tilexr_home}/install}
cann_env=${TILEXR_CANN_ENV:-/home/pkg/b131/cann/set_env.sh}
conda_sh=${TILEXR_CONDA_SH:-/home/miniconda3/etc/profile.d/conda.sh}
conda_env=${TILEXR_CONDA_ENV:-ai_moe_test}
output_root=${TILEXR_MOONEP_OUTPUT_ROOT:-${tilexr_home}/run/moonep/mindspeed}

export LD_LIBRARY_PATH=
export PYTHONPATH=
unset ASCEND_HOME ASCEND_HOME_PATH ASCEND_AICPU_PATH ASCEND_OPP_PATH TOOLCHAIN_HOME
source "${cann_env}"
source "${conda_sh}"
conda activate "${conda_env}"
if [[ -n "${TILEXR_MOONEP_NATIVE_ENV:-}" ]]; then
    source "${TILEXR_MOONEP_NATIVE_ENV}"
fi

export TILEXR_INSTALL_PREFIX=${install_prefix}
export TILEXR_MOONEP_CONDA_ENV=${conda_env}
export TILEXR_UDMA_QP_ROUTE_SPEC=${TILEXR_UDMA_QP_ROUTE_SPEC:-port_count:6,port_count:2}
export TILEXR_UDMA_ATTACH_EXISTING_RA=${TILEXR_UDMA_ATTACH_EXISTING_RA:-1}
unset TILEXR_MOONEP_DISPATCH_TRANSPORT
export TILEXR_MOONEP_DISPATCH_PEER_MODE=group
export TILEXR_MOONEP_DISPATCH_GROUP_WIDTH=16
export TILEXR_MOONEP_DUMP_DFX_ON_ERROR=${TILEXR_MOONEP_DUMP_DFX_ON_ERROR:-1}
export TILEXR_MOONEP_OUTPUT_DIR="${output_root}/case15_iter32_$(date +%Y%m%d-%H%M%S)"

cd "${tilexr_home}"
exec bash scripts/run_moonep.sh \
    --mode benchmark \
    --rank-size 8 \
    --case-id 15 \
    --visible-devices 0,1,2,3,4,5,6,7 \
    --warmup 0 \
    --iterations 32
