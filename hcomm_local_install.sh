#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

colorful_time bash ${TILEXR_HCOMM_HOME}/build_out/cann-hcomm_*.run --full -q --pylocal --install-path=${TILEXR_CANN_HOME}/cann

if [ $? -ne 0 ]; then
    error "install hcomm failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "install hcomm success in ${TILEXR_CANN_HOME}"
fi
