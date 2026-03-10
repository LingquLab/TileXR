#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

op=all_reduce_test

$(which mpirun) -n ${ASCEND_DEV_NUM} ${HCCL_TEST_HOME}/bin/${op} -p ${ASCEND_DEV_NUM} -b 1k -e 128m -f 2 -d int8 -n 20 -w 10 -c 1

if [ $? -ne 0 ]; then
    error "run ${op} failed"
    exit 1
else
    success "run ${op} success"
fi
