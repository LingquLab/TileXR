#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
parse_args "$@"
require_root

if [[ "${DRY_RUN}" != 1 ]] && ! getent group "${CI_GROUP}" >/dev/null; then
    echo "ERROR: required device-access group does not exist: ${CI_GROUP}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" == 1 ]] || ! getent group "${CI_PRIMARY_GROUP}" >/dev/null; then
    run groupadd --system "${CI_PRIMARY_GROUP}"
fi
if [[ "${DRY_RUN}" != 1 ]] && ! ci_primary_group_has_non_root_gid; then
    echo "ERROR: ${CI_PRIMARY_GROUP} must have a non-root GID" >&2
    exit 1
fi

runner_work="${RUNNER_HOME}/_work"
toolchains_home="${CI_HOME}/toolchains"
toolchain_parent="${toolchains_home}/cann"
control_parent="${CI_HOME}/control"
install_work="${CI_HOME}/install-work"

if [[ "${DRY_RUN}" != 1 ]]; then
    for directory in \
        "${CI_HOME}" "${RUNNER_HOME}" "${runner_work}" \
        "${toolchains_home}" "${toolchain_parent}" \
        "${control_parent}" "${install_work}"; do
        if [[ -L "${directory}" || ( -e "${directory}" && ! -d "${directory}" ) ]]; then
            echo "ERROR: provisioning directory must be a real directory: ${directory}" >&2
            exit 1
        fi
    done
fi

if [[ "${DRY_RUN}" != 1 ]] && getent passwd "${CI_USER}" >/dev/null &&
    ! ci_user_ids_are_non_root "${CI_USER}"; then
    echo "ERROR: refusing to modify ${CI_USER} with a root UID or GID" >&2
    exit 1
fi
if [[ "${DRY_RUN}" == 1 ]] || ! getent passwd "${CI_USER}" >/dev/null; then
    run useradd --system --create-home --home-dir "${CI_HOME}" \
        --shell /usr/sbin/nologin --gid "${CI_PRIMARY_GROUP}" "${CI_USER}"
fi
if [[ "${DRY_RUN}" != 1 ]] && ! ci_user_ids_are_non_root "${CI_USER}"; then
    echo "ERROR: ${CI_USER} must have a non-root UID and GID" >&2
    exit 1
fi

run usermod --home "${CI_HOME}" --shell /usr/sbin/nologin "${CI_USER}"
run usermod --gid "${CI_PRIMARY_GROUP}" "${CI_USER}"
run usermod --groups "${CI_GROUP}" "${CI_USER}"
run passwd --lock "${CI_USER}"
if [[ "${DRY_RUN}" != 1 ]] && ! ci_identity_is_bounded "${CI_USER}"; then
    echo "ERROR: ${CI_USER} did not converge to the required group set" >&2
    exit 1
fi

run install -d -o root -g "${CI_PRIMARY_GROUP}" -m 0755 \
    "${CI_HOME}"
run install -d -o root -g "${CI_PRIMARY_GROUP}" -m 0750 \
    "${RUNNER_HOME}"
run install -d -o "${CI_USER}" -g "${CI_PRIMARY_GROUP}" -m 0750 \
    "${runner_work}"
run install -d -o root -g "${CI_GROUP}" -m 0755 \
    "${toolchains_home}" "${toolchain_parent}"
run install -d -o root -g "${CI_GROUP}" -m 0750 \
    "${control_parent}" "${install_work}"
if [[ "${DRY_RUN}" != 1 ]] && ! cann_parent_directories_are_sealed; then
    echo "ERROR: CANN parent directories could not be sealed" >&2
    exit 1
fi
remove_ci_ssh_entry
