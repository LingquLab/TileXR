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
cann_marker="${CANN_HOME}/.tilexr-ci-installing"
cann_cleanup_state=none
cann_tree_identity=
cann_marker_identity=
cann_marker_token=
cann_parent_state=none
cann_ci_home_identity=
cann_toolchains_identity=
cann_toolchain_parent_identity=
stage=

cann_path_identity() {
    local path="$1"
    stat -c '%d:%i' -- "${path}" 2>/dev/null ||
        stat -f '%d:%i' -- "${path}"
}

cann_path_owner() {
    local path="$1"
    stat -c '%U' -- "${path}" 2>/dev/null ||
        stat -f '%Su' -- "${path}"
}

cann_path_size() {
    local path="$1"
    stat -c '%s' -- "${path}" 2>/dev/null ||
        stat -f '%z' -- "${path}"
}

cann_owned_directory_matches() {
    local identity owner
    [[ -d "${CANN_HOME}" && ! -L "${CANN_HOME}" ]] || return 1
    identity="$(cann_path_identity "${CANN_HOME}")" || return 1
    owner="$(cann_path_owner "${CANN_HOME}")" || return 1
    [[ "${identity}" == "${cann_tree_identity}" && "${owner}" == "${CANN_OWNER}" ]]
}

cann_owned_install_matches() {
    local identity owner size token
    cann_owned_directory_matches || return 1
    [[ -f "${cann_marker}" && ! -L "${cann_marker}" ]] || return 1
    identity="$(cann_path_identity "${cann_marker}")" || return 1
    owner="$(cann_path_owner "${cann_marker}")" || return 1
    size="$(cann_path_size "${cann_marker}")" || return 1
    token="$(< "${cann_marker}")" || return 1
    [[ "${identity}" == "${cann_marker_identity}" &&
       "${owner}" == "${CANN_OWNER}" &&
       "${size}" == "$((${#cann_marker_token} + 1))" &&
       "${token}" == "${cann_marker_token}" ]]
}

cann_parent_directories_match_invocation() {
    local ci_home_identity toolchains_identity toolchain_parent_identity
    cann_parent_directories_are_real || return 1
    ci_home_identity="$(cann_path_identity "${CI_HOME}")" || return 1
    toolchains_identity="$(cann_path_identity "${CI_HOME}/toolchains")" || return 1
    toolchain_parent_identity="$(cann_path_identity "${CI_HOME}/toolchains/cann")" || return 1
    [[ "${ci_home_identity}" == "${cann_ci_home_identity}" &&
       "${toolchains_identity}" == "${cann_toolchains_identity}" &&
       "${toolchain_parent_identity}" == "${cann_toolchain_parent_identity}" ]]
}

record_cann_parent_directories() {
    cann_parent_directories_are_real || return 1
    cann_ci_home_identity="$(cann_path_identity "${CI_HOME}")" || return 1
    cann_toolchains_identity="$(cann_path_identity "${CI_HOME}/toolchains")" || return 1
    cann_toolchain_parent_identity="$(cann_path_identity "${CI_HOME}/toolchains/cann")" || return 1
    cann_parent_state=recorded
    cann_parent_directories_match_invocation
}

converge_cann_parent_directories() {
    cann_parent_directories_match_invocation || return 1
    if cann_parent_directories_are_sealed; then
        return 0
    fi
    chown "${CANN_OWNER}:${CI_PRIMARY_GROUP}" "${CI_HOME}" || return 1
    chmod 0755 "${CI_HOME}" || return 1
    chown "${CANN_OWNER}:${CI_GROUP}" \
        "${CI_HOME}/toolchains" "${CI_HOME}/toolchains/cann" || return 1
    chmod 0755 "${CI_HOME}/toolchains" "${CI_HOME}/toolchains/cann" || return 1
    cann_parent_directories_match_invocation &&
        cann_parent_directories_are_sealed
}

restore_cann_parent_directories() {
    [[ "${cann_parent_state}" == recorded ]] || return 0
    if ! cann_parent_directories_match_invocation; then
        echo "ERROR: refusing to restore replaced CANN parent directories" >&2
        return 1
    fi
    if ! converge_cann_parent_directories; then
        echo "ERROR: failed to restore sealed CANN parent directories" >&2
        return 1
    fi
}

cleanup_cann_provision() {
    local status=$?
    trap - EXIT INT TERM HUP
    set +e

    if [[ "${status}" -ne 0 ]]; then
        if [[ "${cann_cleanup_state}" == marked ]]; then
            if cann_owned_install_matches; then
                if ! rm -rf -- "${CANN_HOME}"; then
                    echo "ERROR: failed to clean owned CANN install tree: ${CANN_HOME}" >&2
                fi
            else
                echo "ERROR: refusing to clean replaced CANN install tree: ${CANN_HOME}" >&2
            fi
        elif [[ "${cann_cleanup_state}" == directory ]]; then
            if cann_owned_directory_matches; then
                rmdir -- "${CANN_HOME}" 2>/dev/null || true
            fi
        fi
        restore_cann_parent_directories || true
    fi
    if [[ -n "${stage}" ]]; then
        rm -rf -- "${stage}"
    fi
    exit "${status}"
}

abort_cann_provision() {
    exit "$1"
}

check_driver_version() {
    local driver_version
    driver_version="$(awk -F= '$1 == "Version" {print $2; exit}' \
        /usr/local/Ascend/driver/version.info 2>/dev/null)" || driver_version=""
    if ! version_at_least "${driver_version}" 25.5.0; then
        echo "ERROR: driver >= 25.5.0 is required, found ${driver_version:-unknown}" >&2
        return 1
    fi
}

check_blue_host() {
    local total_count npu_info available
    check_driver_version || return 1

    total_count="$(npu-smi info -l | awk -F: '/Total Count/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
    if [[ "${total_count}" != 8 ]]; then
        echo "ERROR: exactly eight NPU devices are required, found ${total_count:-unknown}" >&2
        return 1
    fi
    if ! npu_info="$(npu-smi info)"; then
        echo "ERROR: could not read the NPU inventory" >&2
        return 1
    fi
    if ! npu_smi_info_has_expected_devices "${npu_info}"; then
        echo "ERROR: expected exactly devices 0..7 as healthy 910B3 NPUs" >&2
        return 1
    fi

    available="$(df --output=avail -B1 /home | tail -n1 | tr -d '[:space:]')"
    if [[ ! "${available}" =~ ^[0-9]+$ ]] || (( available < 30 * 1024 * 1024 * 1024 )); then
        echo "ERROR: at least 30 GiB must be free on /home" >&2
        return 1
    fi
}

if [[ "${DRY_RUN}" != 1 ]]; then
    trap cleanup_cann_provision EXIT
    trap 'abort_cann_provision 130' INT
    trap 'abort_cann_provision 143' TERM
    trap 'abort_cann_provision 129' HUP
    if ! record_cann_parent_directories; then
        echo "ERROR: CANN parent paths must be pre-created real directories" >&2
        exit 1
    fi
    if ! converge_cann_parent_directories; then
        echo "ERROR: CANN parent directories could not be sealed" >&2
        exit 1
    fi
else
    run install -d -o "${CANN_OWNER}" -g "${CI_PRIMARY_GROUP}" -m 0755 \
        "${CI_HOME}"
    run install -d -o "${CANN_OWNER}" -g "${CI_GROUP}" -m 0755 \
        "${CI_HOME}/toolchains" "${CI_HOME}/toolchains/cann"
fi

if [[ "${DRY_RUN}" == 1 ]]; then
    run check_driver_version
    run npu-smi info
    run npu-smi info -l
    run df --output=avail -B1 /home
else
    check_blue_host
fi

if [[ "${DRY_RUN}" != 1 && ( -e "${CANN_HOME}" || -L "${CANN_HOME}" ) ]]; then
    if [[ ! -e "${cann_marker}" && ! -L "${cann_marker}" ]] &&
        cann_tree_is_trusted; then
        echo "CANN 9.1.0 Toolkit and 910B Ops are already installed at ${CANN_HOME}"
        exit 0
    fi
    echo "ERROR: refusing untrusted pre-existing CANN tree: ${CANN_HOME}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" == 1 ]]; then
    stage="${install_work}/cann.dry-run"
else
    stage="$(mktemp -d "${install_work}/cann.XXXXXX")"
fi

run install -d -o root -g "${CI_GROUP}" -m 0750 \
    "${install_work}" "${stage}/scripts" "${stage}/tmp"
for source_file in cann_download_install.sh cann_local_install.sh common_env.sh common_util.sh; do
    run install -o root -g "${CI_GROUP}" -m 0750 \
        "${repo_root}/scripts/${source_file}" "${stage}/scripts/${source_file}"
done

if [[ "${DRY_RUN}" == 1 ]]; then
    run mkdir -m 0755 -- "${CANN_HOME}"
    run chown "${CANN_OWNER}:${CI_GROUP}" "${CANN_HOME}"
    run bash -c \
        'set -o noclobber; umask 077; printf "%s\n" "$1" > "$2"' \
        cann-marker dry-run-token "${cann_marker}"
    run chown "${CANN_OWNER}:${CI_GROUP}" "${cann_marker}"
else
    mkdir -m 0755 -- "${CANN_HOME}"
    cann_tree_identity="$(cann_path_identity "${CANN_HOME}")"
    cann_cleanup_state=directory
    chown "${CANN_OWNER}:${CI_GROUP}" "${CANN_HOME}"
    cann_marker_token="${BASHPID:-$$}:$$:${RANDOM}:${RANDOM}"
    bash -c \
        'set -o noclobber; umask 077; printf "%s\n" "$1" > "$2"' \
        cann-marker "${cann_marker_token}" "${cann_marker}"
    cann_marker_identity="$(cann_path_identity "${cann_marker}")"
    cann_cleanup_state=marked
    chown "${CANN_OWNER}:${CI_GROUP}" "${cann_marker}"
    if ! cann_owned_install_matches; then
        echo "ERROR: CANN install ownership marker could not be verified" >&2
        exit 1
    fi
fi

run env TMPDIR="${stage}/tmp" \
    TILEXR_CI_SEALED_CANN_HOME=1 TILEXR_CANN_HOME="${CANN_HOME}" \
    bash "${stage}/scripts/cann_download_install.sh"
run test -s "${stage}/env/temp/${toolkit_run}"
run test -s "${stage}/env/temp/${ops_run}"

if [[ "${DRY_RUN}" != 1 ]] && ! cann_tree_has_expected_payload; then
    echo "ERROR: installed CANN tree failed Toolkit, 910B Ops, or bisheng verification" >&2
    exit 1
fi

run chown -R "${CANN_OWNER}:${CI_GROUP}" "${CANN_HOME}"
run find -P "${CANN_HOME}" \( -type f -o -type d \) \
    -exec chmod u-s,g-s,o-t,u+rwX,g+rX,o+rX,go-w '{}' +
if [[ "${DRY_RUN}" != 1 ]] && ! cann_tree_is_trusted; then
    echo "ERROR: installed CANN tree could not be sealed" >&2
    exit 1
fi
if [[ "${DRY_RUN}" != 1 ]]; then
    if ! converge_cann_parent_directories; then
        echo "ERROR: CANN parent directories changed during installation" >&2
        exit 1
    fi
    if ! cann_owned_install_matches; then
        echo "ERROR: CANN install ownership changed before finalization" >&2
        exit 1
    fi
    rm -- "${cann_marker}"
    cann_cleanup_state=none
fi
