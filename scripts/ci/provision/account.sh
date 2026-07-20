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

runner_work="${RUNNER_HOME}/_work"
lock_home="${CI_HOME}/locks"
artifact_home="${CI_HOME}/artifacts"
toolchain_parent="${CI_HOME}/toolchains/cann"
control_parent="${CI_HOME}/control"
install_work="${CI_HOME}/install-work"

if [[ "${DRY_RUN}" != 1 ]]; then
    for directory in \
        "${CI_HOME}" "${RUNNER_HOME}" "${runner_work}" "${lock_home}" \
        "${artifact_home}" "${toolchain_parent}" "${control_parent}" "${install_work}"; do
        if [[ -L "${directory}" || ( -e "${directory}" && ! -d "${directory}" ) ]]; then
            echo "ERROR: provisioning directory must be a real directory: ${directory}" >&2
            exit 1
        fi
    done
fi

if [[ "${DRY_RUN}" == 1 ]] || ! getent passwd "${CI_USER}" >/dev/null; then
    run useradd --system --create-home --home-dir "${CI_HOME}" \
        --shell /usr/sbin/nologin --gid "${CI_PRIMARY_GROUP}" "${CI_USER}"
fi

if [[ "${DRY_RUN}" == 1 ]] || ! id -nG "${CI_USER}" | tr ' ' '\n' | grep -Fx "${CI_GROUP}" >/dev/null; then
    run usermod -aG "${CI_GROUP}" "${CI_USER}"
fi

run usermod --home "${CI_HOME}" --shell /usr/sbin/nologin "${CI_USER}"
run usermod --gid "${CI_PRIMARY_GROUP}" "${CI_USER}"
run passwd --lock "${CI_USER}"

if [[ "${DRY_RUN}" == 1 ]] || {
    getent group docker >/dev/null &&
    id -nG "${CI_USER}" | tr ' ' '\n' | grep -Fx docker >/dev/null
}; then
    run gpasswd --delete "${CI_USER}" docker
fi

run install -d -o root -g "${CI_PRIMARY_GROUP}" -m 0750 \
    "${CI_HOME}"
run install -d -o "${CI_USER}" -g "${CI_PRIMARY_GROUP}" -m 0750 \
    "${RUNNER_HOME}" "${runner_work}" "${lock_home}" "${artifact_home}"
run install -d -o root -g "${CI_GROUP}" -m 0750 \
    "${toolchain_parent}" "${control_parent}" "${install_work}"
remove_ci_ssh_entry
