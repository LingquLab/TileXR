#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
provision_root="${repo_root}/scripts/ci/provision"
temp_dir="$(mktemp -d)"
output_file="${temp_dir}/dry-run.out"
trap 'rm -rf "${temp_dir}"' EXIT

scripts=(common account cann control verify)
for script in "${scripts[@]}"; do
    bash "${provision_root}/${script}.sh" --dry-run >> "${output_file}" 2>&1
    if bash "${provision_root}/${script}.sh" --unexpected >/dev/null 2>&1; then
        echo "${script}.sh accepted an unsupported option" >&2
        exit 1
    fi
done

registration_token="tilexr-registration-token-must-not-leak"
printf '%s\n' "${registration_token}" |
    bash "${provision_root}/runner.sh" --dry-run >> "${output_file}" 2>&1
if bash "${provision_root}/runner.sh" --unexpected </dev/null >/dev/null 2>&1; then
    echo "runner.sh accepted an unsupported option" >&2
    exit 1
fi

required_output=(
    'groupadd --system tilexr-ci'
    'useradd --system --create-home --home-dir /home/tilexr-ci --shell /usr/sbin/nologin --gid tilexr-ci tilexr-ci'
    'usermod -aG HwHiAiUser tilexr-ci'
    'install -d -o root -g HwHiAiUser -m 0750 /home/tilexr-ci'
    '/home/tilexr-ci/toolchains/cann/9.1.0'
    'Ascend-cann-toolkit_9.1.0_linux-aarch64.run'
    'Ascend-cann-910b-ops_9.1.0_linux-aarch64.run'
    '/home/tilexr-ci/control/v1'
    '/home/tilexr-ci/control/current/job_completed.sh'
    'ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/home/tilexr-ci/control/current/job_completed.sh'
    'TileXR-NPU'
    'blue-tilexr-npu8'
    'tilexr,ascend910b,npu8'
)

for expected in "${required_output[@]}"; do
    if ! grep -F -- "${expected}" "${output_file}" >/dev/null; then
        echo "dry-run output is missing: ${expected}" >&2
        cat "${output_file}" >&2
        exit 1
    fi
done

groupadd_line="$(grep -nFx -- 'groupadd --system tilexr-ci ' "${output_file}" | cut -d: -f1)"
useradd_line="$(grep -nFx -- \
    'useradd --system --create-home --home-dir /home/tilexr-ci --shell /usr/sbin/nologin --gid tilexr-ci tilexr-ci ' \
    "${output_file}" | cut -d: -f1)"
if [[ -z "${groupadd_line}" || -z "${useradd_line}" || "${groupadd_line}" -ge "${useradd_line}" ]]; then
    echo "the explicit primary group must be created before the CI account" >&2
    exit 1
fi

if ! grep -Fx -- \
    'install -d -o root -g tilexr-ci -m 0750 /home/tilexr-ci ' \
    "${output_file}" >/dev/null; then
    echo "the CI home itself must be administrator-owned" >&2
    exit 1
fi
if ! grep -Fx -- \
    'chown -R root:tilexr-ci /home/tilexr-ci/actions-runner ' \
    "${output_file}" >/dev/null; then
    echo "the installed runner and its root-owned environment must be sealed" >&2
    exit 1
fi

for forbidden in \
    '/etc/sudoers' \
    'docker group' \
    '/usr/local/Ascend/ascend-toolkit' \
    'curl -k' \
    "${registration_token}"; do
    if grep -F -- "${forbidden}" "${output_file}" >/dev/null; then
        echo "dry-run output contains forbidden text: ${forbidden}" >&2
        exit 1
    fi
done

downloader="${repo_root}/scripts/cann_download_install.sh"
if ! grep -F -- \
    'curl --fail --location --continue-at - --remote-name' \
    "${downloader}" >/dev/null; then
    echo "CANN downloader does not use the required TLS-verifying curl options" >&2
    exit 1
fi
if grep -F -- 'curl -k' "${downloader}" >/dev/null; then
    echo "CANN downloader still disables TLS verification" >&2
    exit 1
fi
if ! grep -F -- '[[ ! -s "${TILEXR_TEMP_HOME}/${toolkit_run}" ]]' \
    "${downloader}" >/dev/null; then
    echo "CANN downloader does not validate the exact toolkit file" >&2
    exit 1
fi
if ! grep -F -- '[[ ! -s "${TILEXR_TEMP_HOME}/${ops_run}" ]]' \
    "${downloader}" >/dev/null; then
    echo "CANN downloader does not validate the exact 910B Ops file" >&2
    exit 1
fi

control_provisioner="${provision_root}/control.sh"
if ! grep -F -- 'git -C "${repo_root}" archive' "${control_provisioner}" >/dev/null; then
    echo "control provisioning does not restrict installation to tracked files" >&2
    exit 1
fi
if grep -F -- 'cp -a "${control_source}/."' "${control_provisioner}" >/dev/null; then
    echo "control provisioning can copy untracked files into the trusted package" >&2
    exit 1
fi

if ! grep -F -- 'offline runner has an unexpected registration name' \
    "${provision_root}/runner.sh" >/dev/null; then
    echo "offline replacement does not constrain the existing runner name" >&2
    exit 1
fi
for service_script in runner verify; do
    if ! grep -F -- 'systemctl is-active --quiet --' \
        "${provision_root}/${service_script}.sh" >/dev/null; then
        echo "${service_script}.sh does not terminate systemctl option parsing" >&2
        exit 1
    fi
done

mock_bin="${temp_dir}/mock-bin"
mkdir -p "${mock_bin}"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    '[[ "${LC_ALL:-}" == C && "${LANG:-}" == C ]] || exit 91' \
    '[[ "$*" == "-n -l -U tilexr-ci" ]] || exit 92' \
    'printf "%s" "${MOCK_SUDO_OUTPUT:-}"' \
    'exit "${MOCK_SUDO_STATUS:-0}"' > "${mock_bin}/sudo"
chmod +x "${mock_bin}/sudo"

source "${provision_root}/common.sh"

if ! grep -F -- 'sudo_user_has_allowed_rule "${CI_USER}"' \
    "${provision_root}/verify.sh" >/dev/null; then
    echo "verify.sh does not use the parsed sudo permission state" >&2
    exit 1
fi

assert_sudo_state() {
    local expected_state="$1"
    local command_status="$2"
    local command_output="$3"
    local actual_state
    if MOCK_SUDO_STATUS="${command_status}" \
        MOCK_SUDO_OUTPUT="${command_output}" \
        PATH="${mock_bin}:${PATH}" \
        sudo_user_has_allowed_rule tilexr-ci 2>/dev/null; then
        actual_state=0
    else
        actual_state=$?
    fi
    if [[ "${actual_state}" -ne "${expected_state}" ]]; then
        echo "unexpected sudo permission state: expected ${expected_state}, got ${actual_state}" >&2
        exit 1
    fi
}

assert_sudo_state 1 0 $'Matching Defaults entries for tilexr-ci on blue:\n    env_reset\n\nUser tilexr-ci is not allowed to run sudo on blue.\n'
assert_sudo_state 1 1 $'User tilexr-ci is not allowed to run sudo on blue.\n'
assert_sudo_state 0 0 $'Matching Defaults entries for tilexr-ci on blue:\n    env_reset\n\nUser tilexr-ci may run the following commands on blue:\n    (ALL : ALL) ALL\n'
assert_sudo_state 2 0 $'Matching Defaults entries for tilexr-ci on blue:\n    env_reset\n'
assert_sudo_state 2 1 $'sudo: unable to initialize policy plugin\n'

if ! grep -F -- 'seal_runner_modes' "${provision_root}/runner.sh" >/dev/null; then
    echo "runner.sh does not restore stable modes after recursive sealing" >&2
    exit 1
fi

runner_test_home="${temp_dir}/runner-home"
mkdir -p "${runner_test_home}/_work" "${runner_test_home}/_diag"
printf '%s\n' 'ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/test/hook' > "${runner_test_home}/.env"
chmod 0440 "${runner_test_home}/.env"

file_mode() {
    stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"
}

(
    RUNNER_HOME="${runner_test_home}"
    DRY_RUN=0
    seal_runner_modes
    [[ "$(file_mode "${RUNNER_HOME}/.env")" == 440 ]] || {
        echo "first runner seal changed .env away from mode 0440" >&2
        exit 1
    }
    seal_runner_modes
    [[ "$(file_mode "${RUNNER_HOME}/.env")" == 440 ]] || {
        echo "runner reseal is not idempotent for .env mode" >&2
        exit 1
    }
)

cat "${output_file}"
