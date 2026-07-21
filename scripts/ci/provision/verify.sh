#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
parse_args "$@"
require_root

toolkit_info="${CANN_HOME}/cann/aarch64-linux/ascend_toolkit_install.info"
ops_info="${CANN_HOME}/cann/aarch64-linux/ascend_ops_install.info"
control_current="${CI_HOME}/control/current"
hook="${control_current}/job_completed.sh"
env_entry="ACTIONS_RUNNER_HOOK_JOB_COMPLETED=${hook}"
python_pidfd_probe='import os, signal, sys
if sys.version_info < (3, 9):
    raise SystemExit("Python 3.9 or newer is required")
if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
    raise SystemExit("Python pidfd support is required")'

if [[ "${DRY_RUN}" == 1 ]]; then
    run getent passwd "${CI_USER}"
    run id -nG "${CI_USER}"
    run passwd -S "${CI_USER}"
    run test ! -e "${CI_HOME}/.ssh/authorized_keys"
    run env LC_ALL=C LANG=C sudo -n -l -U "${CI_USER}"
    run grep -Fx version=9.1.0 "${toolkit_info}"
    run grep -Fx package_name=Ascend-cann-910b-ops "${ops_info}"
    run test -x "${CANN_HOME}/cann/tools/bisheng_compiler/bin/bisheng"
    run npu-smi info
    run npu-smi info -l
    run readlink -f "${control_current}"
    run test -x "${hook}"
    run grep -Fx "${env_entry}" "${RUNNER_HOME}/.env"
    run env LC_ALL=C LANG=C systemctl show \
        --property=User --property=ExecStart -- \
        actions.runner.LingquLab-TileXR.blue-tilexr-npu8.service
    run systemctl is-active actions.runner.LingquLab-TileXR.blue-tilexr-npu8.service
    run df -h / /home
    run python3 -c "${python_pidfd_probe}"
    run python3 "${CONTROL_HOME}/npu_state.py"
    exit 0
fi

python3 -c "${python_pidfd_probe}"

passwd_entry="$(getent passwd "${CI_USER}")" || {
    echo "ERROR: ${CI_USER} does not exist" >&2
    exit 1
}
IFS=: read -r _ _ _ _ _ account_home account_shell <<< "${passwd_entry}"
[[ "${account_home}" == "${CI_HOME}" && "${account_shell}" == /usr/sbin/nologin ]] || {
    echo "ERROR: ${CI_USER} has an unexpected home or shell" >&2
    exit 1
}
if ! ci_identity_is_bounded "${CI_USER}"; then
    echo "ERROR: ${CI_USER} has an unexpected UID, GID, or group set" >&2
    exit 1
fi
[[ "$(passwd -S "${CI_USER}" | awk '{print $2}')" == L ]] || {
    echo "ERROR: ${CI_USER} password is not locked" >&2
    exit 1
}
[[ ! -e "${CI_HOME}/.ssh/authorized_keys" && ! -e "${CI_HOME}/.ssh/authorized_keys2" ]] || {
    echo "ERROR: ${CI_USER} has an authorized_keys file" >&2
    exit 1
}
if sudo_user_has_allowed_rule "${CI_USER}"; then
    echo "ERROR: ${CI_USER} has a sudo rule" >&2
    exit 1
else
    sudo_state=$?
    if [[ "${sudo_state}" -ne 1 ]]; then
        echo "ERROR: sudo permission verification failed" >&2
        exit 1
    fi
fi

if ! cann_parent_directories_are_sealed; then
    echo "ERROR: CANN parent directories are not sealed" >&2
    exit 1
fi
if ! cann_tree_is_trusted; then
    echo "ERROR: CANN tree is not the expected sealed and contained 9.1.0 toolchain" >&2
    exit 1
fi

total_count="$(npu-smi info -l | awk -F: '/Total Count/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
[[ "${total_count}" == 8 ]] || {
    echo "ERROR: exactly eight NPU devices are required" >&2
    exit 1
}
npu_info="$(npu-smi info)" || {
    echo "ERROR: could not read the NPU inventory" >&2
    exit 1
}
if ! npu_smi_info_has_expected_devices "${npu_info}"; then
    echo "ERROR: expected exactly devices 0..7 as healthy 910B3 NPUs" >&2
    exit 1
fi

[[ "$(< "${CONTROL_HOME}/VERSION")" == v1 ]] || {
    echo "ERROR: sealed controller is not v1" >&2
    exit 1
}
[[ "$(readlink -f "${control_current}")" == "${CONTROL_HOME}" ]] || {
    echo "ERROR: current controller link does not resolve to v1" >&2
    exit 1
}
[[ "$(stat -c '%U:%G' "${CONTROL_HOME}")" == "root:${CI_GROUP}" ]] || {
    echo "ERROR: controller ownership is not root:${CI_GROUP}" >&2
    exit 1
}
[[ "$(stat -c '%U:%G' "${CI_HOME}")" == "root:${CI_PRIMARY_GROUP}" ]] || {
    echo "ERROR: CI home is not administrator-owned" >&2
    exit 1
}
if runuser -u "${CI_USER}" -- test -w "${CI_HOME}"; then
    echo "ERROR: CI home permits replacement of sealed paths" >&2
    exit 1
fi
if runuser -u "${CI_USER}" -- find "${CONTROL_HOME}" -writable -print -quit | grep -q .; then
    echo "ERROR: controller is writable by ${CI_USER}" >&2
    exit 1
fi
[[ -x "${hook}" ]] || {
    echo "ERROR: job-completed hook is not executable" >&2
    exit 1
}
if runuser -u "${CI_USER}" -- test -w "${hook}"; then
    echo "ERROR: job-completed hook is writable by ${CI_USER}" >&2
    exit 1
fi
grep -Fx "${env_entry}" "${RUNNER_HOME}/.env" >/dev/null
[[ "$(stat -c '%U:%G' "${RUNNER_HOME}")" == "root:${CI_PRIMARY_GROUP}" ]] || {
    echo "ERROR: runner installation is not administrator-owned" >&2
    exit 1
}
[[ "$(stat -c '%U:%G' "${RUNNER_HOME}/.env")" == "root:${CI_PRIMARY_GROUP}" ]] || {
    echo "ERROR: runner environment is not administrator-owned" >&2
    exit 1
}
if runuser -u "${CI_USER}" -- test -w "${RUNNER_HOME}/.env"; then
    echo "ERROR: runner environment is writable by ${CI_USER}" >&2
    exit 1
fi
service_name="$(< "${RUNNER_HOME}/.service")"
[[ "${service_name}" =~ ^actions\.runner\.[A-Za-z0-9_.-]+\.service$ ]] || {
    echo "ERROR: runner service name is invalid" >&2
    exit 1
}
runner_service_matches "${service_name}" || {
    echo "ERROR: runner service has an unexpected user or executable" >&2
    exit 1
}
systemctl is-active --quiet -- "${service_name}" || {
    echo "ERROR: runner service is not active" >&2
    exit 1
}

echo 'Filesystem usage before runner activation:'
df -h / /home
python3 - "${CONTROL_HOME}/npu_state.py" "${CI_USER}" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("tilexr_npu_state", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
snapshot = module.read_snapshot()
if not snapshot.healthy:
    raise SystemExit("all eight devices must be healthy")
leftovers = [process for process in snapshot.processes if process.owner == sys.argv[2]]
if leftovers:
    raise SystemExit("CI-owned NPU processes remain before activation")
PY
