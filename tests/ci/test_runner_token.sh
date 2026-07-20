#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/scripts/ci/provision/common.sh"

if ! declare -F run_with_runner_registration_token >/dev/null; then
    echo "runner registration environment helper is missing" >&2
    exit 1
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT
fake_runuser="${temp_dir}/runuser"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    '[[ "${ACTIONS_RUNNER_INPUT_TOKEN:-}" == runner-token-contract-secret ]] || exit 91' \
    'printf "%s\n" "$@" > "${TEST_ARGV_FILE}"' \
    'exit "${MOCK_RUNUSER_STATUS:-0}"' \
    > "${fake_runuser}"
chmod +x "${fake_runuser}"

export TEST_ARGV_FILE="${temp_dir}/argv.txt"
export MOCK_RUNUSER_STATUS=0
registration_token=runner-token-contract-secret
run_with_runner_registration_token "${registration_token}" \
    "${fake_runuser}" -u tilexr-ci -- /runner/config.sh \
    --url https://github.com/LingquLab --unattended

if [[ -n "${registration_token+x}" ||
      -n "${ACTIONS_RUNNER_INPUT_TOKEN+x}" ]]; then
    echo "runner token remained in the root shell after configuration" >&2
    exit 1
fi
if grep -F -- '--token' "${TEST_ARGV_FILE}" >/dev/null ||
    grep -F -- runner-token-contract-secret "${TEST_ARGV_FILE}" >/dev/null; then
    echo "runner registration token appeared in process arguments" >&2
    exit 1
fi

export MOCK_RUNUSER_STATUS=23
registration_token=runner-token-contract-secret
set +e
run_with_runner_registration_token "${registration_token}" \
    "${fake_runuser}" -u tilexr-ci -- /runner/config.sh --unattended
status=$?
set -e
if [[ "${status}" -ne 23 || -n "${registration_token+x}" ||
      -n "${ACTIONS_RUNNER_INPUT_TOKEN+x}" ]]; then
    echo "failed runner registration did not clear its token environment" >&2
    exit 1
fi
