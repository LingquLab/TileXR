#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

cp ${TILEXR_HOME}/.gitignore ${TILEXR_OPS_HOME}/.gitignore

cd ${TILEXR_OPS_HOME}

ops=${1:-all_gather_matmul}

rm -f ${TILEXR_OPS_HOME}/build_out/cann-ops*.run

echo "${ASCEND_PROCESS_LOG_PATH}" > ${TILEXR_PLOG_FILE_PATH}

export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

CMD="bash build.sh --run_example ${ops} eager cust -p ${TILEXR_CANN_HOME}/cann --soc=${TILEXR_SOC_NAME}"
warn ${CMD}
colorful_time ${CMD} | tee ${TILEXR_RUN_HOME}/ops.log
ops_run_status=${PIPESTATUS[0]}

cd ${TILEXR_HOME}

if [ ${ops_run_status} -ne 0 ]; then
    error "run ops-transformer failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "run ops-transformer success in ${TILEXR_CANN_HOME}"
fi
