#!/usr/bin/env bash

set -euo pipefail

CI_USER=tilexr-ci
CI_PRIMARY_GROUP=tilexr-ci
CI_GROUP=HwHiAiUser
CI_HOME=/home/tilexr-ci
CANN_HOME=/home/tilexr-ci/toolchains/cann/9.1.0
CANN_OWNER=root
CONTROL_HOME=/home/tilexr-ci/control/v1
RUNNER_HOME=/home/tilexr-ci/actions-runner

DRY_RUN=0

parse_args() {
    if [[ "$#" -eq 0 ]]; then
        DRY_RUN=0
    elif [[ "$#" -eq 1 && "$1" == "--dry-run" ]]; then
        DRY_RUN=1
    else
        echo "Usage: $0 [--dry-run]" >&2
        return 2
    fi
}

run() {
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

require_root() {
    if [[ "${DRY_RUN}" != 1 && "$(id -u)" -ne 0 ]]; then
        echo "ERROR: provisioning must run as root" >&2
        return 1
    fi
}

ci_primary_group_has_non_root_gid() {
    local entry group_name password group_gid members
    entry="$(getent group "${CI_PRIMARY_GROUP}")" || return 1
    IFS=: read -r group_name password group_gid members <<< "${entry}"
    [[ "${group_name}" == "${CI_PRIMARY_GROUP}" &&
       "${group_gid}" =~ ^[0-9]+$ && "${group_gid}" -gt 0 ]]
}

ci_user_ids_are_non_root() {
    local user="$1"
    local uid gid
    uid="$(id -u "${user}")" || return 1
    gid="$(id -g "${user}")" || return 1
    [[ "${uid}" =~ ^[0-9]+$ && "${gid}" =~ ^[0-9]+$ &&
       "${uid}" -gt 0 && "${gid}" -gt 0 ]]
}

ci_identity_is_bounded() {
    local user="$1"
    local primary_group primary_gid group_entry group_name password group_gid members
    local groups actual_supplementary expected_supplementary

    ci_user_ids_are_non_root "${user}" || return 1
    ci_primary_group_has_non_root_gid || return 1
    primary_group="$(id -gn "${user}")" || return 1
    primary_gid="$(id -g "${user}")" || return 1
    group_entry="$(getent group "${CI_PRIMARY_GROUP}")" || return 1
    IFS=: read -r group_name password group_gid members <<< "${group_entry}"
    [[ "${primary_group}" == "${CI_PRIMARY_GROUP}" &&
       "${primary_gid}" == "${group_gid}" ]] || return 1

    groups="$(id -nG "${user}")" || return 1
    actual_supplementary="$(
        tr ' ' '\n' <<< "${groups}" |
            awk -v primary="${CI_PRIMARY_GROUP}" 'NF && $0 != primary' |
            LC_ALL=C sort -u
    )"
    expected_supplementary="$(printf '%s\n' "${CI_GROUP}" | LC_ALL=C sort -u)"
    [[ "${actual_supplementary}" == "${expected_supplementary}" ]]
}

cann_metadata_matches() {
    local path="$1"
    local package="$2"
    local package_count version_count

    [[ -f "${path}" && ! -L "${path}" ]] || return 1
    package_count="$(grep -c '^package_name=' "${path}" || true)"
    version_count="$(grep -c '^version=' "${path}" || true)"
    [[ "${package_count}" -eq 1 && "${version_count}" -eq 1 ]] || return 1
    grep -Fx "package_name=${package}" "${path}" >/dev/null &&
        grep -Fx 'version=9.1.0' "${path}" >/dev/null
}

cann_realpath_is_contained() {
    local path="$1"
    local root_real path_real
    root_real="$(realpath "${CANN_HOME}")" || return 1
    path_real="$(realpath "${path}")" || return 1
    [[ "${path_real}" == "${root_real}" || "${path_real}" == "${root_real}/"* ]]
}

cann_tree_links_are_contained() {
    local link
    while IFS= read -r -d '' link; do
        cann_realpath_is_contained "${link}" || return 1
    done < <(find -P "${CANN_HOME}" -type l -print0)
}

cann_tree_has_expected_payload() {
    local toolkit_info="${CANN_HOME}/cann/aarch64-linux/ascend_toolkit_install.info"
    local ops_info="${CANN_HOME}/cann/aarch64-linux/ascend_ops_install.info"
    local compiler="${CANN_HOME}/cann/compiler/ccec_compiler/bin/bisheng"

    [[ -d "${CANN_HOME}" && ! -L "${CANN_HOME}" ]] || return 1
    cann_metadata_matches "${toolkit_info}" Ascend-cann-toolkit || return 1
    cann_metadata_matches "${ops_info}" Ascend-cann-910b-ops || return 1
    [[ -x "${compiler}" ]] || return 1
    cann_realpath_is_contained "${toolkit_info}" || return 1
    cann_realpath_is_contained "${ops_info}" || return 1
    cann_realpath_is_contained "${compiler}" || return 1
    cann_tree_links_are_contained
}

cann_entry_stat() {
    local path="$1"
    stat -c '%U:%G:%a' "${path}" 2>/dev/null ||
        stat -f '%Su:%Sg:%Lp' "${path}"
}

cann_tree_regular_entries_are_sealed() {
    local path entry owner group mode
    while IFS= read -r -d '' path; do
        entry="$(cann_entry_stat "${path}")" || return 1
        IFS=: read -r owner group mode <<< "${entry}"
        [[ "${owner}" == "${CANN_OWNER}" && "${group}" == "${CI_GROUP}" ]] || return 1
        [[ "${mode}" =~ ^[0-7]*[0-7][0145][0145]$ ]] || return 1
    done < <(find -P "${CANN_HOME}" \( -type f -o -type d \) -print0)
}

cann_tree_symlinks_are_owned() {
    local link entry owner group mode
    while IFS= read -r -d '' link; do
        entry="$(cann_entry_stat "${link}")" || return 1
        IFS=: read -r owner group mode <<< "${entry}"
        [[ "${owner}" == "${CANN_OWNER}" && "${group}" == "${CI_GROUP}" ]] || return 1
    done < <(find -P "${CANN_HOME}" -type l -print0)
}

cann_tree_is_trusted() {
    cann_tree_has_expected_payload &&
        cann_tree_regular_entries_are_sealed &&
        cann_tree_symlinks_are_owned
}

sudo_user_has_allowed_rule() {
    local user="$1"
    local output status

    if output="$(env LC_ALL=C LANG=C sudo -n -l -U "${user}" 2>&1)"; then
        status=0
    else
        status=$?
    fi

    if [[ "${status}" -eq 0 ]] &&
        grep -Eq '^User .+ may run the following commands on .+:$' <<< "${output}"; then
        return 0
    fi
    if grep -Eq '^User .+ is not allowed to run sudo on .+[.]?$' <<< "${output}"; then
        return 1
    fi

    echo "ERROR: could not determine sudo permissions for ${user}" >&2
    printf '%s\n' "${output}" >&2
    return 2
}

runner_service_matches() {
    local service_name="$1"
    local output status user_count exec_count user_line exec_line
    local expected_exec="${RUNNER_HOME}/runsvc.sh"

    if [[ ! "${service_name}" =~ ^actions\.runner\.[A-Za-z0-9_.-]+\.service$ ]]; then
        return 1
    fi
    if output="$(env LC_ALL=C LANG=C systemctl show \
        --property=User --property=ExecStart -- "${service_name}" 2>&1)"; then
        status=0
    else
        status=$?
    fi
    if [[ "${status}" -ne 0 ]]; then
        echo "ERROR: could not inspect runner service ${service_name}" >&2
        printf '%s\n' "${output}" >&2
        return 1
    fi

    user_count="$(awk '/^User=/{count++} END{print count+0}' <<< "${output}")"
    exec_count="$(awk '/^ExecStart=/{count++} END{print count+0}' <<< "${output}")"
    [[ "${user_count}" -eq 1 && "${exec_count}" -eq 1 ]] || return 1
    user_line="$(awk '/^User=/{print}' <<< "${output}")"
    exec_line="$(awk '/^ExecStart=/{print}' <<< "${output}")"
    [[ "${user_line}" == "User=${CI_USER}" ]] || return 1
    [[ "${exec_line}" == \
        "ExecStart={ path=${expected_exec} ; argv[]=${expected_exec} ;"* ]]
}

remove_ci_ssh_entry() {
    run rm -rf -- "${CI_HOME}/.ssh"
}

seal_runner_modes() {
    run chmod 0750 "${RUNNER_HOME}"
    run find "${RUNNER_HOME}" -mindepth 1 -maxdepth 1 \
        ! -name _work ! -name _diag \
        -exec chmod -R u+rwX,g+rX,o-rwx,go-w '{}' +
    run chmod 0440 "${RUNNER_HOME}/.env"
    run chmod -R u+rwX,g+rX,o-rwx,go-w \
        "${RUNNER_HOME}/_work" "${RUNNER_HOME}/_diag"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    parse_args "$@"
fi
