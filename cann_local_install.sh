#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

mkdir -p ${TILEXR_CANN_HOME}
mkdir -p ${TILEXR_TEMP_HOME}

fix_path=${TILEXR_CANN_HOME}
while [ "${fix_path}" != "/" ] && [ "${fix_path}" != "/home" ]; do
    perm=`stat -c "%a" ${fix_path}`
    if [ "${perm}" != "755" ]; then
        warn "fix permission to 755 for ${fix_path}"
	chmod 755 ${fix_path}
    fi
    fix_path=$(dirname "${fix_path}")
done

chmod +x ${TILEXR_TEMP_HOME}/Ascend-cann_*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run

colorful_time bash ${TILEXR_TEMP_HOME}/Ascend-cann_*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run --full -q --force --whitelist=toolkit --install-path=${TILEXR_CANN_HOME}

chmod +x ${TILEXR_TEMP_HOME}/Ascend-cann-${TILEXR_OPS_NAME}-ops*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run

colorful_time bash ${TILEXR_TEMP_HOME}/Ascend-cann-${TILEXR_OPS_NAME}-ops*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run --install -q --install-path=${TILEXR_CANN_HOME}

line

