#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

mkdir -p ${TILEXR_CANN_HOME}
mkdir -p ${TILEXR_TEMP_HOME}

line

toolkit_run=Ascend-cann-toolkit_${TILEXR_CANN_VER}_linux-${TILEXR_OS_ARCH}.run
ops_run=Ascend-cann-${TILEXR_OPS_NAME}-ops_${TILEXR_CANN_VER}_linux-${TILEXR_OS_ARCH}.run

obs_base="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260520000325981"
cann_url=${obs_base}/${toolkit_run}
ops_url=${obs_base}/${ops_run}

success "TILEXR_OS_ARCH = ${TILEXR_OS_ARCH}"
success "TILEXR_CANN_VER = ${TILEXR_CANN_VER}"

# 检查 PID 文件中记录的进程是否仍是 curl，决定是否接管或重新 fork
# 用法: _ensure_curl_running <pid_file> <url> <log_file>
# 返回: 设置全局变量 _curl_pid
_ensure_curl_running() {
    local pid_file=$1
    local url=$2
    local log_file=$3

    if [ -f "${pid_file}" ]; then
        local stored_pid
        stored_pid=$(cat "${pid_file}")
        if kill -0 "${stored_pid}" 2>/dev/null; then
            local comm
            comm=$(cat /proc/${stored_pid}/comm 2>/dev/null)
            if [ "${comm}" = "curl" ]; then
                success "curl already running (pid=${stored_pid}), resuming wait"
                _curl_pid=${stored_pid}
                return
            else
                warn "pid ${stored_pid} is not curl (comm=${comm}), restarting download"
            fi
        else
            warn "pid ${stored_pid} no longer alive, restarting download"
        fi
        rm -f "${pid_file}"
    fi

    cd ${TILEXR_TEMP_HOME}
    curl -k -C - -O ${url} > ${log_file} 2>&1 &
    _curl_pid=$!
    echo ${_curl_pid} > "${pid_file}"
    cd ${TILEXR_HOME}
}

toolkit_pid_file=${TILEXR_TEMP_HOME}/cann_toolkit.pid
ops_pid_file=${TILEXR_TEMP_HOME}/cann_ops.pid

success "start download cann from ${cann_url}"
_ensure_curl_running "${toolkit_pid_file}" "${cann_url}" "${TILEXR_TEMP_HOME}/toolkit.log"
pid_cann=${_curl_pid}

success "start download ops from ${ops_url}"
_ensure_curl_running "${ops_pid_file}" "${ops_url}" "${TILEXR_TEMP_HOME}/ops.log"
pid_ops=${_curl_pid}

while kill -0 ${pid_cann} 2>/dev/null; do
    tail -n1 ${TILEXR_TEMP_HOME}/toolkit.log | awk '{printf "\r%s", $0; fflush()}'
    sleep 1
done
echo ""
rm -f "${toolkit_pid_file}"
success "cann downloaded."

while kill -0 ${pid_ops} 2>/dev/null; do
    tail -n1 ${TILEXR_TEMP_HOME}/ops.log | awk '{printf "\r%s", $0; fflush()}'
    sleep 1
done
echo ""
rm -f "${ops_pid_file}"
success "ops downloaded."

success "begin install."
bash ${script_path}/cann_local_install.sh

line
