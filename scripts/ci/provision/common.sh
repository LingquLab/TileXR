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

seal_runner_modes() {
    run chmod -R u+rwX,g+rX,o-rwx,go-w "${RUNNER_HOME}"
    run chmod 0440 "${RUNNER_HOME}/.env"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    parse_args "$@"
fi
