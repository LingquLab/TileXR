#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

if [[ "${TILEXR_CI_SEALED_CANN_HOME:-0}" == 1 ]]; then
    cann_marker="${TILEXR_CANN_HOME}/.tilexr-ci-installing"
    if [[ ! -d "${TILEXR_CANN_HOME}" || -L "${TILEXR_CANN_HOME}" ||
          ! -f "${cann_marker}" || -L "${cann_marker}" ]]; then
        error "sealed CI CANN home or ownership marker is invalid: ${TILEXR_CANN_HOME}"
        exit 1
    fi
else
    mkdir -p "${TILEXR_CANN_HOME}"
    fix_permissions "${TILEXR_CANN_HOME}"
fi
mkdir -p "${TILEXR_TEMP_HOME}"

chmod +x ${TILEXR_TEMP_HOME}/Ascend-cann-toolkit_*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run

colorful_time bash ${TILEXR_TEMP_HOME}/Ascend-cann-toolkit_*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run --full -q --force --install-path=${TILEXR_CANN_HOME}
if [ $? -ne 0 ]; then
    error "install CANN toolkit failed"
    exit 1
fi

chmod +x ${TILEXR_TEMP_HOME}/Ascend-cann-${TILEXR_OPS_NAME}-ops*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run

colorful_time bash ${TILEXR_TEMP_HOME}/Ascend-cann-${TILEXR_OPS_NAME}-ops*${TILEXR_CANN_VER}*${TILEXR_OS_ARCH}.run --install -q --install-path=${TILEXR_CANN_HOME}
if [ $? -ne 0 ]; then
    error "install CANN ops failed"
    exit 1
fi

line
