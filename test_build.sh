#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

cd ${TILEXR_HCCL_TEST_HOME}

colorful_time make

cd ${HELPER_HOME}

if [ $? -ne 0 ]; then
    error "build hccl_test failed"
    exit 1
else
    success "build hccl_test success"
fi
