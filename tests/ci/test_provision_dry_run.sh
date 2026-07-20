#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
provision_root="${repo_root}/scripts/ci/provision"
output_file="$(mktemp)"
trap 'rm -f "${output_file}"' EXIT

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
    'useradd --system --create-home --home-dir /home/tilexr-ci --shell /usr/sbin/nologin'
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

cat "${output_file}"
