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
    'usermod --groups HwHiAiUser tilexr-ci'
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

if grep -F -- 'usermod -aG ' "${output_file}" >/dev/null; then
    echo "account provisioning must replace, not append, supplementary groups" >&2
    exit 1
fi

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
    'install -d -o root -g HwHiAiUser -m 0750 /home/tilexr-ci/toolchains /home/tilexr-ci/toolchains/cann ' \
    "${output_file}" >/dev/null; then
    echo "the CANN parent directories are not explicitly sealed" >&2
    exit 1
fi
if ! grep -F -- 'TILEXR_CI_SEALED_CANN_HOME=1' "${output_file}" >/dev/null; then
    echo "the sealed CANN installer permission switch is missing" >&2
    exit 1
fi
if grep -F -- 'run chmod -R u+rwX,g+rX,o-rwx,go-w "${CANN_HOME}"' \
    "${provision_root}/cann.sh" >/dev/null; then
    echo "CANN sealing still uses recursive chmod across symlinks" >&2
    exit 1
fi
if ! grep -F -- 'run find -P "${CANN_HOME}"' \
    "${provision_root}/cann.sh" >/dev/null; then
    echo "CANN sealing does not use a no-follow filesystem walk" >&2
    exit 1
fi
for provision_script in account cann verify; do
    if ! grep -F -- 'cann_parent_directories_are_sealed' \
        "${provision_root}/${provision_script}.sh" >/dev/null; then
        echo "${provision_script}.sh does not verify sealed CANN parents" >&2
        exit 1
    fi
done
if ! grep -Fx -- \
    'chown root:tilexr-ci /home/tilexr-ci/actions-runner ' \
    "${output_file}" >/dev/null; then
    echo "the runner root itself must be administrator-owned" >&2
    exit 1
fi
if ! grep -F -- '! -name _diag' "${provision_root}/runner.sh" >/dev/null; then
    echo "runner reinstall does not preserve the diagnostic directory" >&2
    exit 1
fi
if grep -F -- 'run chown -R "root:${CI_PRIMARY_GROUP}" "${RUNNER_HOME}"' \
    "${provision_root}/runner.sh" >/dev/null; then
    echo "runner sealing must not make mutable work trees root-owned" >&2
    exit 1
fi
if ! grep -F -- \
    'chown -R tilexr-ci:tilexr-ci /home/tilexr-ci/actions-runner/_work /home/tilexr-ci/actions-runner/_diag ' \
    "${output_file}" >/dev/null; then
    echo "runner mutable trees are not recursively restored to the CI account" >&2
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
if grep -F -- '"${RUNNER_HOME}/config.sh" remove' \
    "${provision_root}/runner.sh" >/dev/null ||
    grep -Eq 'remove[^[:cntrl:]]*registration_token' \
        "${provision_root}/runner.sh"; then
    echo "runner registration tokens must not be used to remove runners" >&2
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

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '    -u) printf "%s\\n" "${MOCK_ID_UID}" ;;' \
    '    -g) printf "%s\\n" "${MOCK_ID_GID}" ;;' \
    '    -gn) printf "%s\\n" "${MOCK_ID_PRIMARY_GROUP}" ;;' \
    '    -nG) printf "%s\\n" "${MOCK_ID_GROUPS}" ;;' \
    '    *) exit 93 ;;' \
    'esac' \
    > "${mock_bin}/id"
chmod +x "${mock_bin}/id"

printf '%s\n' \
    '#!/usr/bin/env bash' \
    '[[ "$1" == group && "$2" == tilexr-ci ]] || exit 94' \
    'printf "tilexr-ci:x:%s:\\n" "${MOCK_PRIMARY_GROUP_GID}"' \
    > "${mock_bin}/getent"
chmod +x "${mock_bin}/getent"

assert_identity_state() {
    local expected_state="$1"
    local uid="$2"
    local gid="$3"
    local primary_group="$4"
    local groups="$5"
    local primary_group_gid="$6"
    local actual_state
    if MOCK_ID_UID="${uid}" \
        MOCK_ID_GID="${gid}" \
        MOCK_ID_PRIMARY_GROUP="${primary_group}" \
        MOCK_ID_GROUPS="${groups}" \
        MOCK_PRIMARY_GROUP_GID="${primary_group_gid}" \
        PATH="${mock_bin}:${PATH}" \
        ci_identity_is_bounded tilexr-ci 2>/dev/null; then
        actual_state=0
    else
        actual_state=$?
    fi
    if [[ "${actual_state}" -ne "${expected_state}" ]]; then
        echo "unexpected CI identity state: expected ${expected_state}, got ${actual_state}" >&2
        exit 1
    fi
}

assert_identity_state 0 990 991 tilexr-ci 'HwHiAiUser tilexr-ci' 991
assert_identity_state 1 0 991 tilexr-ci 'tilexr-ci HwHiAiUser' 991
assert_identity_state 1 990 0 tilexr-ci 'tilexr-ci HwHiAiUser' 991
assert_identity_state 1 990 991 tilexr-ci 'tilexr-ci HwHiAiUser' 0
assert_identity_state 1 990 991 tilexr-ci 'tilexr-ci HwHiAiUser disk lxd' 991
assert_identity_state 1 990 991 tilexr-ci 'tilexr-ci' 991
assert_identity_state 1 990 991 root 'root HwHiAiUser' 991

if [[ "$(grep -Fc 'ci_identity_is_bounded "${CI_USER}"' \
        "${provision_root}/account.sh")" -lt 1 ]]; then
    echo "account.sh does not verify the converged CI identity" >&2
    exit 1
fi
if ! grep -F -- 'ci_identity_is_bounded "${CI_USER}"' \
    "${provision_root}/verify.sh" >/dev/null; then
    echo "verify.sh does not enforce the exact CI identity" >&2
    exit 1
fi

printf '%s\n' \
    '#!/usr/bin/env bash' \
    '[[ "${LC_ALL:-}" == C && "${LANG:-}" == C ]] || exit 91' \
    '[[ "$*" == "show --property=User --property=ExecStart -- actions.runner.LingquLab-TileXR.blue-tilexr-npu8.service" ]] || exit 92' \
    'printf "%s" "${MOCK_SYSTEMCTL_OUTPUT:-}"' \
    'exit "${MOCK_SYSTEMCTL_STATUS:-0}"' > "${mock_bin}/systemctl"
chmod +x "${mock_bin}/systemctl"

assert_runner_service_state() {
    local expected_state="$1"
    local command_status="$2"
    local command_output="$3"
    local actual_state
    if MOCK_SYSTEMCTL_STATUS="${command_status}" \
        MOCK_SYSTEMCTL_OUTPUT="${command_output}" \
        PATH="${mock_bin}:${PATH}" \
        runner_service_matches \
            actions.runner.LingquLab-TileXR.blue-tilexr-npu8.service 2>/dev/null; then
        actual_state=0
    else
        actual_state=$?
    fi
    if [[ "${actual_state}" -ne "${expected_state}" ]]; then
        echo "unexpected runner service state: expected ${expected_state}, got ${actual_state}" >&2
        exit 1
    fi
}

good_service=$'User=tilexr-ci\nExecStart={ path=/home/tilexr-ci/actions-runner/runsvc.sh ; argv[]=/home/tilexr-ci/actions-runner/runsvc.sh ; ignore_errors=no ; }\n'
assert_runner_service_state 0 0 "${good_service}"
assert_runner_service_state 1 0 $'User=root\nExecStart={ path=/home/tilexr-ci/actions-runner/runsvc.sh ; argv[]=/home/tilexr-ci/actions-runner/runsvc.sh ; }\n'
assert_runner_service_state 1 0 $'User=tilexr-ci\nExecStart={ path=/tmp/runsvc.sh ; argv[]=/tmp/runsvc.sh ; }\n'
assert_runner_service_state 1 0 $'User=tilexr-ci\nExecStart={ path=/home/tilexr-ci/actions-runner-evil/runsvc.sh ; argv[]=/home/tilexr-ci/actions-runner-evil/runsvc.sh ; }\n'
assert_runner_service_state 1 1 'systemctl failed'

if [[ "$(grep -Fc 'runner_service_matches "${service_name}"' \
        "${provision_root}/runner.sh")" -lt 2 ]]; then
    echo "runner.sh must validate the service before accepting and starting it" >&2
    exit 1
fi
if ! grep -F -- 'runner_service_matches "${service_name}"' \
    "${provision_root}/verify.sh" >/dev/null; then
    echo "verify.sh does not validate the runner service identity" >&2
    exit 1
fi

account_last_command="$(awk '
    NF && $1 !~ /^#/ { line = $0 }
    END { sub(/^[[:space:]]*/, "", line); print line }
' "${provision_root}/account.sh")"
if [[ "${account_last_command}" != remove_ci_ssh_entry ]]; then
    echo "account.sh must remove the SSH entry only after sealing the CI home" >&2
    exit 1
fi

ssh_test_home="${temp_dir}/ssh-home"
ssh_external="${temp_dir}/external-ssh"
mkdir -p "${ssh_test_home}" "${ssh_external}"
printf '%s\n' preserve > "${ssh_external}/authorized_keys"
ln -s "${ssh_external}" "${ssh_test_home}/.ssh"
(
    CI_HOME="${ssh_test_home}"
    DRY_RUN=0
    remove_ci_ssh_entry
    [[ ! -e "${CI_HOME}/.ssh" && ! -L "${CI_HOME}/.ssh" ]] || {
        echo "account SSH entry was not removed" >&2
        exit 1
    }
    remove_ci_ssh_entry
)
if [[ "$(< "${ssh_external}/authorized_keys")" != preserve ]]; then
    echo "removing the account SSH entry followed its external symlink" >&2
    exit 1
fi

if ! grep -F -- 'seal_runner_modes' "${provision_root}/runner.sh" >/dev/null; then
    echo "runner.sh does not restore stable modes after recursive sealing" >&2
    exit 1
fi

runner_test_home="${temp_dir}/runner-home"
mkdir -p "${runner_test_home}/_work/job/nested" "${runner_test_home}/_diag/logs"
printf '%s\n' work > "${runner_test_home}/_work/job/nested/output.txt"
printf '%s\n' diag > "${runner_test_home}/_diag/logs/runner.log"
chmod -R 0777 "${runner_test_home}/_work" "${runner_test_home}/_diag"
chmod 0666 \
    "${runner_test_home}/_work/job/nested/output.txt" \
    "${runner_test_home}/_diag/logs/runner.log"
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
    [[ "$(file_mode "${RUNNER_HOME}/_work/job/nested")" == 750 &&
       "$(file_mode "${RUNNER_HOME}/_work/job/nested/output.txt")" == 640 &&
       "$(file_mode "${RUNNER_HOME}/_diag/logs")" == 750 &&
       "$(file_mode "${RUNNER_HOME}/_diag/logs/runner.log")" == 640 ]] || {
        echo "runner mutable trees were not recursively normalized" >&2
        exit 1
    }
    seal_runner_modes
    [[ "$(file_mode "${RUNNER_HOME}/.env")" == 440 ]] || {
        echo "runner reseal is not idempotent for .env mode" >&2
        exit 1
    }
)

cat "${output_file}"
