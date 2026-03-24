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

# 构建 hcomm，可选传入 --noclean 跳过清理步骤
_hcomm_build() {
    local noclean_flag=${1:-""}
    rm -rf ${TILEXR_HCOMM_HOME}/build_out/cann-hcomm_*.run
    local CMD="bash ${TILEXR_HCOMM_HOME}/build.sh -j`nproc` --full ${noclean_flag} -p ${TILEXR_CANN_HOME}/cann"
    warn ${CMD}
    colorful_time ${CMD}
    if [ $? -ne 0 ]; then
        error "build hcomm failed"
        return 1
    else
        success "build hcomm success"
    fi
}

# 解压 tarball 并通过 autoconf configure/make/make install 安装工具包
# 用法: _install_autoconf_pkg <pkg_name> <tarball> [configure_extra_args...]
# 安装到 ${TILEXR_UTIL_HOME}/<pkg_name>/，日志追加到 ${TILEXR_TEMP_HOME}/3rd.log
_install_autoconf_pkg() {
    local pkg_name=$1
    local tarball=$2
    shift 2
    warn "install ${pkg_name} begin"
    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/
    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/${tarball} --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/
    cd ${TILEXR_TEMP_HOME}/${pkg_name}/
    colorful_time ./configure --prefix=${TILEXR_UTIL_HOME}/${pkg_name}/ "$@" >> ${TILEXR_TEMP_HOME}/3rd.log
    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd.log
    make install >> ${TILEXR_TEMP_HOME}/3rd.log
    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
        return 1
    fi
    cd ${TILEXR_HOME}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    env_print
fi
