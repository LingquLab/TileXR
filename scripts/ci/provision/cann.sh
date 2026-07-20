#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${script_dir}/common.sh"
parse_args "$@"
require_root

install_work="${CI_HOME}/install-work"
toolkit_run=Ascend-cann-toolkit_9.1.0_linux-aarch64.run
ops_run=Ascend-cann-910b-ops_9.1.0_linux-aarch64.run
toolkit_info="${CANN_HOME}/cann/aarch64-linux/ascend_toolkit_install.info"
ops_info="${CANN_HOME}/cann/aarch64-linux/ascend_ops_install.info"

verify_metadata() {
    local path="$1"
    local package="$2"
    [[ -f "${path}" && ! -L "${path}" ]] || return 1
    grep -Fx "package_name=${package}" "${path}" >/dev/null &&
        grep -Fx 'version=9.1.0' "${path}" >/dev/null
}

verify_cann_tree() {
    verify_metadata "${toolkit_info}" Ascend-cann-toolkit &&
        verify_metadata "${ops_info}" Ascend-cann-910b-ops &&
        [[ -x "${CANN_HOME}/cann/compiler/ccec_compiler/bin/bisheng" ]]
}

check_blue_host() {
    local driver_version total_count device product health available
    driver_version="$(sed -n 's/^Version=//p' /usr/local/Ascend/driver/version.info | head -n1)"
    if [[ "${driver_version}" != 25.5.0 ]]; then
        echo "ERROR: driver 25.5.0 is required, found ${driver_version:-unknown}" >&2
        return 1
    fi

    total_count="$(npu-smi info -l | awk -F: '/Total Count/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
    if [[ "${total_count}" != 8 ]]; then
        echo "ERROR: exactly eight NPU devices are required, found ${total_count:-unknown}" >&2
        return 1
    fi
    for device in 0 1 2 3 4 5 6 7; do
        product="$(npu-smi info -t product -i "${device}")"
        health="$(npu-smi info -t health -i "${device}")"
        if ! grep -Eq 'Name[[:space:]]*:[[:space:]]*(Ascend[[:space:]]*)?910B3([[:space:]]|$)' <<< "${product}"; then
            echo "ERROR: device ${device} is not an Ascend 910B3" >&2
            return 1
        fi
        if ! grep -Eq '^[[:space:]]*Health[[:space:]]*:[[:space:]]*OK[[:space:]]*$' <<< "${health}"; then
            echo "ERROR: device ${device} is not healthy" >&2
            return 1
        fi
    done

    available="$(df --output=avail -B1 /home | tail -n1 | tr -d '[:space:]')"
    if [[ ! "${available}" =~ ^[0-9]+$ ]] || (( available < 30 * 1024 * 1024 * 1024 )); then
        echo "ERROR: at least 30 GiB must be free on /home" >&2
        return 1
    fi
}

if [[ "${DRY_RUN}" == 1 ]]; then
    run grep -Fx Version=25.5.0 /usr/local/Ascend/driver/version.info
    run npu-smi info -l
    for device in 0 1 2 3 4 5 6 7; do
        run npu-smi info -t product -i "${device}"
        run npu-smi info -t health -i "${device}"
    done
    run df --output=avail -B1 /home
else
    check_blue_host
fi

if [[ "${DRY_RUN}" != 1 && ( -e "${CANN_HOME}" || -L "${CANN_HOME}" ) ]]; then
    if verify_cann_tree; then
        run chown -R "root:${CI_GROUP}" "${CANN_HOME}"
        run chmod -R u+rwX,g+rX,o-rwx,go-w "${CANN_HOME}"
        run install -d -o root -g "${CI_PRIMARY_GROUP}" -m 0750 "${CI_HOME}"
        run install -d -o root -g "${CI_GROUP}" -m 0750 \
            "${CI_HOME}/toolchains" "${CI_HOME}/toolchains/cann"
        echo "CANN 9.1.0 Toolkit and 910B Ops are already installed at ${CANN_HOME}"
        exit 0
    fi
    echo "ERROR: refusing to replace incomplete or unexpected CANN tree: ${CANN_HOME}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" == 1 ]]; then
    stage="${install_work}/cann.dry-run"
else
    stage="$(mktemp -d "${install_work}/cann.XXXXXX")"
    trap 'rm -rf -- "${stage}"' EXIT
fi

run install -d -o root -g "${CI_GROUP}" -m 0750 "${install_work}" "${stage}/scripts"
for source_file in cann_download_install.sh cann_local_install.sh common_env.sh common_util.sh; do
    run install -o root -g "${CI_GROUP}" -m 0750 \
        "${repo_root}/scripts/${source_file}" "${stage}/scripts/${source_file}"
done
run env TILEXR_CANN_HOME="${CANN_HOME}" bash "${stage}/scripts/cann_download_install.sh"
run test -s "${stage}/env/temp/${toolkit_run}"
run test -s "${stage}/env/temp/${ops_run}"

if [[ "${DRY_RUN}" != 1 ]] && ! verify_cann_tree; then
    echo "ERROR: installed CANN tree failed Toolkit, 910B Ops, or bisheng verification" >&2
    exit 1
fi

run chown -R "root:${CI_GROUP}" "${CANN_HOME}"
run chmod -R u+rwX,g+rX,o-rwx,go-w "${CANN_HOME}"
run install -d -o root -g "${CI_PRIMARY_GROUP}" -m 0750 "${CI_HOME}"
run install -d -o root -g "${CI_GROUP}" -m 0750 \
    "${CI_HOME}/toolchains" "${CI_HOME}/toolchains/cann"
