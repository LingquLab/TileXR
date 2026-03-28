#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

cd ${TILEXR_OPBASE_HOME}

rm -rf ${TILEXR_OPBASE_HOME}/build_out/cann-opbase_*.run

CMD="bash ${TILEXR_OPBASE_HOME}/build.sh -j${TILEXR_HALF_NPROC}"
warn ${CMD}
colorful_time ${CMD}

cd ${TILEXR_HOME}

if [ $? -ne 0 ]; then
    error "build opbase failed"
    exit 1
else
    success "build opbase success"
fi

colorful_time bash ${TILEXR_OPBASE_HOME}/build_out/cann-opbase_*.run --full -q --install-path=${TILEXR_CANN_HOME}/cann

if [ $? -ne 0 ]; then
    error "install opbase failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "install opbase success in ${TILEXR_CANN_HOME}"
fi
