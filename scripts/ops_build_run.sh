#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

cp ${TILEXR_HOME}/.ops_gitignore ${TILEXR_OPS_HOME}/.gitignore

cd ${TILEXR_HOME}
cmake . && make && make install
comm_build_status=$?

if [ ${comm_build_status} -ne 0 ]; then
    error "install tilexr-comm failed in ${TILEXR_HOME}"
    exit 1
else
    success "install tilexr-comm success in ${TILEXR_HOME}"
fi

cd ${TILEXR_OPS_HOME}

ops=${1:-all_gather_matmul}

rm -f ${TILEXR_OPS_HOME}/build_out/cann-ops*.run

# CMD="bash build.sh --opapi -j`nproc` -p ${TILEXR_CANN_HOME}/cann --soc=${TILEXR_SOC_NAME}"
# warn ${CMD}
# colorful_time ${CMD}

# mkdir -p ${TILEXR_HOME}/lib
# \cp ${TILEXR_OPS_HOME}/build/libopapi_*.so ${ASCEND_HOME_PATH}/lib64/

opsdir=${TILEXR_OPS_HOME}/mc2/${ops}
if [ ! -d "$opsdir" ]; then
    cp -rf ${TILEXR_HOME}/src/mc2/${ops} ${TILEXR_OPS_HOME}/mc2
fi
cp -rf ${TILEXR_HOME}/src/mc2/* ${TILEXR_OPS_HOME}/mc2/

commargs=${TILEXR_OPS_HOME}/common/include/kernel/comm_args.h
if [ ! -f "$commargs" ]; then
    cp -rf ${TILEXR_HOME}/src/include/comm_args.h ${TILEXR_OPS_HOME}/common/include/kernel
fi
tilexrsync=${TILEXR_OPS_HOME}/common/include/kernel/tilexr_sync.h
if [ ! -f "$tilexrsync" ]; then
    cp -rf ${TILEXR_HOME}/src/include/tilexr_sync.h ${TILEXR_OPS_HOME}/common/include/kernel
fi
cp -f ${TILEXR_HOME}/src/mc2/build.sh ${TILEXR_OPS_HOME}/build.sh
CMD="bash build.sh --pkg -j`nproc` -p ${TILEXR_CANN_HOME}/cann --soc=${TILEXR_SOC_NAME} --ops=${ops}"
warn ${CMD}
colorful_time ${CMD}
ops_build_status=$?

cd ${TILEXR_HOME}

if [ ${ops_build_status} -ne 0 ]; then
    error "build ops-transformer failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "build ops-transformer success in ${TILEXR_CANN_HOME}"
fi

LD_LIBRARY_PATH=${TILEXR_OPS_HOME}/build:${LD_LIBRARY_PATH} bash ${TILEXR_OPS_HOME}/build_out/cann-ops*.run --install-path=${TILEXR_CANN_HOME}/cann

if [ $? -ne 0 ]; then
    error "install ops-transformer failed in ${TILEXR_CANN_HOME}"
    exit 1
else
    success "install ops-transformer success in ${TILEXR_CANN_HOME}"
fi

cd ${TILEXR_HOME}

bash ${script_path}/ops_only_run.sh ${ops}
