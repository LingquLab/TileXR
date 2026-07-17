#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT

expected_cann_home="${temp_dir}/cann"
mkdir -p "${expected_cann_home}"

actual_cann_home="$(
    export TILEXR_CANN_HOME="${expected_cann_home}"
    source "${repo_root}/scripts/common_env.sh"
    printf '%s\n' "${TILEXR_CANN_HOME}"
)"

printf '%s\n' "${actual_cann_home}"
if [[ "${actual_cann_home}" != "${expected_cann_home}" ]]; then
    echo "TILEXR_CANN_HOME was overwritten: expected ${expected_cann_home}, got ${actual_cann_home}" >&2
    exit 1
fi
