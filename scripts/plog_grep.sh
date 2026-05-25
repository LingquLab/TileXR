#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

if [ -z "${TILEXR_PLOG_FILE_PATH}" ]; then
    error "TILEXR_PLOG_FILE_PATH undefined"
    exit 1
fi

if [ ! -f "${TILEXR_PLOG_FILE_PATH}" ]; then
    error "${TILEXR_PLOG_FILE_PATH} not exist"
    exit 1
fi

# find `cat ${TILEXR_PLOG_FILE_PATH}` -name '*.log' | xargs grep "$@"

rg --no-ignore-vcs "${@:-ERROR}" `cat ${TILEXR_PLOG_FILE_PATH}`
