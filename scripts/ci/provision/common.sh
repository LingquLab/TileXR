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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    parse_args "$@"
fi
