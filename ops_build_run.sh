#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

cp ${TILEXR_HOME}/.ops_gitignore ${TILEXR_OPS_HOME}/.gitignore

cd ${TILEXR_OPS_HOME}

ops=${1:-all_gather_matmul}

rm -f ${TILEXR_OPS_HOME}/build_out/cann-ops*.run

# CMD="bash build.sh --opapi -j`nproc` -p ${TILEXR_CANN_HOME}/cann --soc=${TILEXR_SOC_NAME}"
# warn ${CMD}
# colorful_time ${CMD}

# mkdir -p ${TILEXR_HOME}/lib
# \cp ${TILEXR_OPS_HOME}/build/libopapi_*.so ${ASCEND_HOME_PATH}/lib64/

CMD="bash build.sh --pkg -j`nproc` -p ${TILEXR_CANN_HOME}/cann --soc=${TILEXR_SOC_NAME} --ops=${ops}"
warn ${CMD}
colorful_time ${CMD}

cd ${TILEXR_HOME}

if [ $? -ne 0 ]; then
    error "build ops-transformer failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "build ops-transformer success in ${TILEXR_CANN_HOME}"
fi

bash ${TILEXR_OPS_HOME}/build_out/cann-ops*.run --install-path=${TILEXR_CANN_HOME}/cann

if [ $? -ne 0 ]; then
    error "install ops-transformer failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "install ops-transformer success in ${TILEXR_CANN_HOME}"
fi

cd ${OPS_HOME}

echo "${ASCEND_PROCESS_LOG_PATH}" > ${PLOG_FILE_PATH}

CMD="bash build.sh --run_example ${ops} eager -p ${TILEXR_CANN_HOME}/cann --soc=${TILEXR_SOC_NAME}"
warn ${CMD}
colorful_time ${CMD} | tee ${RUN_HOME}/ops.log

cd ${TILEXR_HOME}

if [ $? -ne 0 ]; then
    error "run ops-transformer failed in ${TILEXR_CANN_HOME}"
    error "logpath: ${ASCEND_PROCESS_LOG_PATH}"
    exit 1
else
    success "run ops-transformer success in ${TILEXR_CANN_HOME}"
    success "logpath: ${ASCEND_PROCESS_LOG_PATH}"
fi
