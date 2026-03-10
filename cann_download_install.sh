#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

env_print

mkdir -p ${TILEXR_CANN_HOME}
mkdir -p ${TILEXR_TEMP_HOME}

line

fix_path=${TILEXR_CANN_HOME}
while [ "${fix_path}" != "/" ] && [ "${fix_path}" != "/home" ]; do
    perm=`stat -c "%a" ${fix_path}`
    if [ "${perm}" != "755" ]; then
        warn "fix permission to 755 for ${fix_path}"
	chmod 755 ${fix_path}
    fi
    fix_path=$(dirname "${fix_path}")
done

toolkit_run=Ascend-cann_${TILEXR_CANN_VER}_linux-${TILEXR_OS_ARCH}.run
ops_run=Ascend-cann-${TILEXR_OPS_NAME}-ops_${TILEXR_CANN_VER}_linux-${TILEXR_OS_ARCH}.run

cann_url=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T2/${toolkit_run}
ops_url=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T2/${ops_run}

success "TILEXR_OS_ARCH = ${TILEXR_OS_ARCH}"
success "TILEXR_CANN_VER = ${TILEXR_CANN_VER}"
# success "cann_url = ${cann_url}"
# success "toolkit_run = ${toolkit_run}"
# success "ops_url = ${ops_url}"
# success "ops_run = ${ops_run}"

cd ${TILEXR_TEMP_HOME}

success "start download cann from ${cann_url}"
curl -k -C - -O ${cann_url} > ${TILEXR_TEMP_HOME}/toolkit.log 2>&1 &
pid_cann=$!

success "start download ops from ${ops_url}"
curl -k -C - -O ${ops_url} > ${TILEXR_TEMP_HOME}/ops.log 2>&1 &
pid_ops=$!

cd ${TILEXR_HOME}

while kill -0 ${pid_cann} 2>/dev/null; do
    cat ${TILEXR_TEMP_HOME}/toolkit.log | tail -n1 | awk '{printf "\r%s", $0; fflush()}'
    sleep 1
done
# colorful_time wait ${pid_cann}
echo ""

success "cann downloaded, begin install."

chmod +x ${TILEXR_TEMP_HOME}/${toolkit_run}

colorful_time bash ${TILEXR_TEMP_HOME}/${toolkit_run} --full --whitelist=toolkit -q --force --install-path=${TILEXR_CANN_HOME}

while kill -0 ${pid_ops} 2>/dev/null; do
    cat ${TILEXR_TEMP_HOME}/ops.log | tail -n1 | awk '{printf "\r%s", $0; fflush()}'
    sleep 1
done
# colorful_time wait ${pid_ops}
echo ""

success "ops downloaded, begin install."
chmod +x ${TILEXR_TEMP_HOME}/${ops_run}

colorful_time bash ${TILEXR_TEMP_HOME}/${ops_run} --install -q --install-path=${TILEXR_CANN_HOME}

line

