#!/bin/bash



script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`

source ${script_path}/common_env.sh



env_print



cp ${TILEXR_HOME}/.ops_gitignore ${TILEXR_OPS_HOME}/.gitignore



cd ${TILEXR_OPS_HOME}



ops=${1:-all_gather_matmul}



rm -f ${TILEXR_OPS_HOME}/build_out/cann-ops*.run



echo "${ASCEND_PROCESS_LOG_PATH}" > ${TILEXR_PLOG_FILE_PATH}



CMD="bash build.sh --run_example ${ops} eager cust -p ${TILEXR_CANN_HOME}/cann --soc=`soc_name`"

warn ${CMD}

colorful_time ${CMD} | tee ${TILEXR_RUN_HOME}/ops.log



cd ${TILEXR_HOME}



if [ $? -ne 0 ]; then

    error "run ops-transformer failed in ${TILEXR_CANN_HOME}"

    exit 1

else

    success "run ops-transformer success in ${TILEXR_CANN_HOME}"

fi
