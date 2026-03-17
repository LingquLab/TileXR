#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

cp ${TILEXR_HOME}/.hcomm_gitignore ${TILEXR_HCOMM_HOME}/.gitignore

cd ${TILEXR_HCOMM_HOME}

rm -rf ${TILEXR_HCOMM_HOME}/build_out/cann-hcomm_*.run

CMD="bash ${TILEXR_HCOMM_HOME}/build.sh -j`nproc` --full -p ${TILEXR_CANN_HOME}/cann"
warn ${CMD}
colorful_time ${CMD}

cd ${TILEXR_HOME}

if [ $? -ne 0 ]; then
    error "build hcomm failed"
    exit 1
else
    success "build hcomm success"
fi

colorful_time bash ${TILEXR_HCOMM_HOME}/build_out/cann-hcomm_*.run --full -q --pylocal --install-path=${TILEXR_CANN_HOME}/cann

if [ $? -ne 0 ]; then
    error "install hcomm failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "install hcomm success in ${TILEXR_CANN_HOME}"
fi
