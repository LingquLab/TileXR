#!/usr/bin/env bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`

GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
RED=$'\033[0;31m'
PURPLE=$'\033[38;5;171m'
END=$'\033[0m'

colorful_time() {
    _desc="time cost"
    if command -v ${script_path}/env/util/time/bin/time >/dev/null 2>&1; then
        ${script_path}/env/util/time/bin/time -f "[OK] ${GREEN}✅${END} ${_desc}: ${PURPLE}%e${END}s" "$@"
    elif command -v /usr/bin/time >/dev/null 2>&1; then
        /usr/bin/time -f "[OK] ${GREEN}✅${END} ${_desc}: ${PURPLE}%e${END}s" "$@"
    else
        TIMEFORMAT="[OK] ${_desc}: %Rs" && time $@
    fi
}

colorful_time_with_desc() {
    _desc=$1
    shift 1
    if command -v ${script_path}/env/util/time/bin/time >/dev/null 2>&1; then
        ${script_path}/env/util/time/bin/time -f "[OK] ${GREEN}✅${END} ${_desc}: ${PURPLE}%e${END}s" "$@"
    elif command -v /usr/bin/time >/dev/null 2>&1; then
        /usr/bin/time -f "[OK] ${GREEN}✅${END} ${_desc}: ${PURPLE}%e${END}s" "$@"
    else
        TIMEFORMAT="[OK] ${_desc}: %Rs" && time $@
    fi
}

success() { echo -e "[OK] \033[0;32m✅\033[0m $@"; }
error()   { echo -e "[ERROR] \033[0;31m❌\033[0m $@"; }
warn()    { echo -e "[WARN] \033[0;33m🔶\033[0m $@"; }

line()    { echo "##########################################################"; }

davinci_detect() { echo `lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2`; }

soc_name() {
    declare -A SOC_MAP=(
        ["d802"]="ascend910b"
        ["d803"]="ascend910_93"
    )
    davinci_type=`davinci_detect`
    if [ "${davinci_type}" = "" ]; then
        echo "ascend910b"
        return
    fi
    echo "${SOC_MAP[$davinci_type]}"
}

ops_name() {
    declare -A OPS_MAP=(
        ["ascend910b"]="910b"
        ["ascend910_93"]="A3"
    )
    name=`soc_name`
    echo "${OPS_MAP[$name]}"
}

# 修复指定路径及其所有祖先目录的权限为 755（直至 / 或 /home）
fix_permissions() {
    local fix_path=$1
    while [ "${fix_path}" != "/" ] && [ "${fix_path}" != "/home" ]; do
        local perm=`stat -c "%a" ${fix_path}`
        if [ "${perm}" != "755" ]; then
            warn "fix permission to 755 for ${fix_path}"
            chmod 755 ${fix_path}
        fi
        fix_path=$(dirname "${fix_path}")
    done
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

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    success "test success" 1 2 3
    error "test error"
    warn "test warn"

    success `davinci_detect`
    success `soc_name`
fi
