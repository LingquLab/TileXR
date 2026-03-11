#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

op=all_reduce_test

$(which mpirun) -n ${TILEXR_ASCEND_DEV_NUM} ${TILEXR_HCCL_TEST_HOME}/bin/${op} -p ${TILEXR_ASCEND_DEV_NUM} -b 1k -e 128m -f 2 -d int8 -n 20 -w 10 -c 1

if [ $? -ne 0 ]; then
    error "run ${op} failed"
    exit 1
else
    success "run ${op} success"
fi
