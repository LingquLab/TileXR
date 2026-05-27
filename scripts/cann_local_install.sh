#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

mkdir -p ${TILEXR_CANN_HOME}
mkdir -p ${TILEXR_TEMP_HOME}

fix_permissions ${TILEXR_CANN_HOME}

chmod +x ${TILEXR_TEMP_HOME}/Ascend-cann-toolkit_*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run

colorful_time bash ${TILEXR_TEMP_HOME}/Ascend-cann-toolkit_*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run --full -q --force --install-path=${TILEXR_CANN_HOME}

chmod +x ${TILEXR_TEMP_HOME}/Ascend-cann-${TILEXR_OPS_NAME}-ops*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run

colorful_time bash ${TILEXR_TEMP_HOME}/Ascend-cann-${TILEXR_OPS_NAME}-ops*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run --install -q --install-path=${TILEXR_CANN_HOME}

line
