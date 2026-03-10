#!/bin/bash



script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`

source ${script_path}/common_env.sh



colorful_time bash ${TILEXR_HCOMM_HOME}/build_out/cann-hcomm_*.run --full -q --pylocal --install-path=${CANN_HOME}/cann

# colorful_time bash ${TEMP_HOME}/cann-hcomm_${CANN_VER}*${HELPER_OS_ARCH}.run --full -q --pylocal --install-path=${CANN_HOME}/cann | tee ${ENV_HOME}/hcomm.log



if [ $? -ne 0 ]; then

    error "install hcomm failed in ${TILEXR_CANN_HOME}"

    exit 1

else

    success "install hcomm success in ${TILEXR_CANN_HOME}"

fi

