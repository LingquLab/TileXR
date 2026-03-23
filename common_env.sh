#!/bin/bash

if [ -z "${BASH_SOURCE[0]}" ]; then
    echo "请在 bash 中执行脚本"
    return
fi

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_util.sh

export TILEXR_OS_ARCH=`uname -m`
export TILEXR_CANN_VER="9.0.0-beta.1"

export TILEXR_SOC_NAME=`soc_name`
export TILEXR_OPS_NAME=`ops_name`
export TILEXR_HOME=${script_path}
export TILEXR_3RD_HOME=${script_path}/3rdparty
export TILEXR_3RD_OPEN_HOME=${TILEXR_3RD_HOME}/open_source
export TILEXR_ENV_HOME=${TILEXR_HOME}/env
export TILEXR_CANN_HOME=${TILEXR_ENV_HOME}/cann
export TILEXR_TEMP_HOME=${TILEXR_ENV_HOME}/temp
export TILEXR_UTIL_HOME=${TILEXR_ENV_HOME}/util

mkdir -p ${TILEXR_ENV_HOME}
mkdir -p ${TILEXR_TEMP_HOME}
mkdir -p ${TILEXR_UTIL_HOME}

export TILEXR_HCCL_TEST_HOME=${TILEXR_CANN_HOME}/cann/tools/hccl_test

export ASCEND_DIR=${TILEXR_CANN_HOME}/cann
export MPI_HOME=${TILEXR_UTIL_HOME}/mpich

export TILEXR_HCOMM_HOME=${TILEXR_3RD_HOME}/hcomm
export TILEXR_OPS_HOME=${TILEXR_3RD_HOME}/ops-transformer
export TILEXR_OPBASE_HOME=${TILEXR_3RD_HOME}/opbase

# 运行日志相关目录
export TILEXR_RUN_HOME=${TILEXR_HOME}/run
export TILEXR_PLOG_HOME=${TILEXR_RUN_HOME}/plog
export TILEXR_PROF_HOME=${TILEXR_RUN_HOME}/prof
export TILEXR_PLOG_FILE_PATH=${TILEXR_TEMP_HOME}/logfile

# 机器的卡数
export TILEXR_ASCEND_DEV_NUM=$((`lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | wc -l`))

date_str=`date '+%y%m%d%H%M'`
export ASCEND_PROCESS_LOG_PATH=${TILEXR_PLOG_HOME}/$date_str
export ASCEND_GLOBAL_LOG_LEVEL=3
mkdir -p ${TILEXR_PLOG_HOME}

if [ -f ${TILEXR_CANN_HOME}/cann/set_env.sh ]; then
    source ${TILEXR_CANN_HOME}/cann/set_env.sh
fi

if [ -f ${TILEXR_CANN_HOME}/cann/vendors/custom_transformer/bin/set_env.bash ]; then
    source ${TILEXR_CANN_HOME}/cann/vendors/custom_transformer/bin/set_env.bash
fi

export PATH=${MPI_HOME}/bin:${PATH}
export PATH=${TILEXR_UTIL_HOME}/cmake/bin:${PATH}
export PATH=${TILEXR_UTIL_HOME}/ccache:${TILEXR_UTIL_HOME}/ripgrep:${TILEXR_UTIL_HOME}/sshpass/bin:${PATH}
export PATH=${TILEXR_UTIL_HOME}/time/bin:${TILEXR_UTIL_HOME}/patch/bin:${TILEXR_UTIL_HOME}/pigz:${PATH}

export LD_LIBRARY_PATH=${MPI_HOME}/lib:${LD_LIBRARY_PATH}

env_print() {
    line
    success "TILEXR_OS_ARCH = ${TILEXR_OS_ARCH}"
    success "TILEXR_SOC_NAME = ${TILEXR_SOC_NAME}"
    success "TILEXR_OPS_NAME = ${TILEXR_OPS_NAME}"
    success "TILEXR_CANN_VER = ${TILEXR_CANN_VER}"
    success "TILEXR_HOME = ${TILEXR_HOME}"
    success "TILEXR_CANN_HOME = ${TILEXR_CANN_HOME}"
    success "TILEXR_HCOMM_HOME = ${TILEXR_HCOMM_HOME}"
    success "TILEXR_OPS_HOME = ${TILEXR_OPS_HOME}"
    line
    env | grep ASCEND
    line
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    env_print
fi
