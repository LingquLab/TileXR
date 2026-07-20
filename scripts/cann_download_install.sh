#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

if [[ ! -d "${TILEXR_CANN_HOME}" || -L "${TILEXR_CANN_HOME}" ]]; then
    error "TILEXR_CANN_HOME must be a pre-created real directory: ${TILEXR_CANN_HOME}"
    exit 1
fi
mkdir -p "${TILEXR_TEMP_HOME}"

line

toolkit_run=Ascend-cann-toolkit_${TILEXR_CANN_VER}_linux-${TILEXR_OS_ARCH}.run
ops_run=Ascend-cann-${TILEXR_OPS_NAME}-ops_${TILEXR_CANN_VER}_linux-${TILEXR_OS_ARCH}.run

obs_base="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260520000325981"
cann_url=${obs_base}/${toolkit_run}
ops_url=${obs_base}/${ops_run}

success "TILEXR_OS_ARCH = ${TILEXR_OS_ARCH}"
success "TILEXR_CANN_VER = ${TILEXR_CANN_VER}"

success "start download cann from ${cann_url}"
cd "${TILEXR_TEMP_HOME}" || exit 1
curl --fail --location --continue-at - --remote-name "${cann_url}" \
    > "${TILEXR_TEMP_HOME}/toolkit.log" 2>&1 &
pid_cann=$!

success "start download ops from ${ops_url}"
curl --fail --location --continue-at - --remote-name "${ops_url}" \
    > "${TILEXR_TEMP_HOME}/ops.log" 2>&1 &
pid_ops=$!
cd "${TILEXR_HOME}" || exit 1

_cancel_downloads() {
    trap - INT TERM HUP
    kill "${pid_cann}" "${pid_ops}" 2>/dev/null || true
    wait "${pid_cann}" 2>/dev/null || true
    wait "${pid_ops}" 2>/dev/null || true
    exit 130
}
trap _cancel_downloads INT TERM HUP

if wait "${pid_cann}"; then
    toolkit_status=0
else
    toolkit_status=$?
fi
if wait "${pid_ops}"; then
    ops_status=0
else
    ops_status=$?
fi
trap - INT TERM HUP

if [[ "${toolkit_status}" -ne 0 || "${ops_status}" -ne 0 ]]; then
    error "CANN downloads failed: toolkit=${toolkit_status}, ops=${ops_status}"
    exit 1
fi
success "cann downloaded."
success "ops downloaded."

if [[ ! -s "${TILEXR_TEMP_HOME}/${toolkit_run}" ]]; then
    error "downloaded toolkit is missing or empty: ${toolkit_run}"
    exit 1
fi
if [[ ! -s "${TILEXR_TEMP_HOME}/${ops_run}" ]]; then
    error "downloaded 910B Ops package is missing or empty: ${ops_run}"
    exit 1
fi

success "begin install."
bash ${script_path}/cann_local_install.sh
if [ $? -ne 0 ]; then
    error "install CANN failed"
    exit 1
fi

line
