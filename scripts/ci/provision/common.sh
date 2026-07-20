#!/usr/bin/env bash

set -euo pipefail

CI_USER=tilexr-ci
CI_PRIMARY_GROUP=tilexr-ci
CI_GROUP=HwHiAiUser
CI_HOME=/home/tilexr-ci
CANN_HOME=/home/tilexr-ci/toolchains/cann/9.1.0
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
