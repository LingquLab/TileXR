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

if [[ "${DRY_RUN}" == 1 ]]; then
    run getent passwd "${CI_USER}"
    run id -nG "${CI_USER}"
    run passwd -S "${CI_USER}"
    run test ! -e "${CI_HOME}/.ssh/authorized_keys"
    run sudo -n -l -U "${CI_USER}"
    run grep -Fx version=9.1.0 "${toolkit_info}"
    run grep -Fx package_name=Ascend-cann-910b-ops "${ops_info}"
    run test -x "${CANN_HOME}/cann/compiler/ccec_compiler/bin/bisheng"
    for device in 0 1 2 3 4 5 6 7; do
        run npu-smi info -t health -i "${device}"
    done
    run readlink -f "${control_current}"
    run test -x "${hook}"
    run grep -Fx "${env_entry}" "${RUNNER_HOME}/.env"
    run systemctl is-active actions.runner.LingquLab-TileXR.blue-tilexr-npu8.service
    run df -h / /home
    run python3 "${CONTROL_HOME}/npu_state.py"
    exit 0
fi

passwd_entry="$(getent passwd "${CI_USER}")" || {
    echo "ERROR: ${CI_USER} does not exist" >&2
    exit 1
}
IFS=: read -r _ _ _ _ _ account_home account_shell <<< "${passwd_entry}"
[[ "${account_home}" == "${CI_HOME}" && "${account_shell}" == /usr/sbin/nologin ]] || {
    echo "ERROR: ${CI_USER} has an unexpected home or shell" >&2
    exit 1
}
id -nG "${CI_USER}" | tr ' ' '\n' | grep -Fx "${CI_GROUP}" >/dev/null || {
    echo "ERROR: ${CI_USER} is not a member of ${CI_GROUP}" >&2
    exit 1
}
[[ "$(id -gn "${CI_USER}")" == "${CI_PRIMARY_GROUP}" ]] || {
    echo "ERROR: ${CI_USER} has an unexpected primary group" >&2
    exit 1
}
if id -nG "${CI_USER}" | tr ' ' '\n' | grep -Fx docker >/dev/null; then
    echo "ERROR: ${CI_USER} must not be a Docker member" >&2
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
if sudo -n -l -U "${CI_USER}" >/dev/null 2>&1; then
    echo "ERROR: ${CI_USER} has a sudo rule" >&2
    exit 1
fi

grep -Fx package_name=Ascend-cann-toolkit "${toolkit_info}" >/dev/null
grep -Fx version=9.1.0 "${toolkit_info}" >/dev/null
grep -Fx package_name=Ascend-cann-910b-ops "${ops_info}" >/dev/null
grep -Fx version=9.1.0 "${ops_info}" >/dev/null
if find "${CANN_HOME}" \( ! -user root -o -perm /022 \) -print -quit | grep -q .; then
    echo "ERROR: CANN tree is not sealed root-owned and read-only to group/other" >&2
    exit 1
fi
bisheng="${CANN_HOME}/cann/compiler/ccec_compiler/bin/bisheng"
[[ -x "${bisheng}" && "$(realpath "${bisheng}")" == "${CANN_HOME}"/* ]] || {
    echo "ERROR: bisheng does not resolve from the sealed CANN tree" >&2
    exit 1
}

total_count="$(npu-smi info -l | awk -F: '/Total Count/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
[[ "${total_count}" == 8 ]] || {
    echo "ERROR: exactly eight NPU devices are required" >&2
    exit 1
}
for device in 0 1 2 3 4 5 6 7; do
    product="$(npu-smi info -t product -i "${device}")"
    health="$(npu-smi info -t health -i "${device}")"
    grep -Eq 'Name[[:space:]]*:[[:space:]]*(Ascend[[:space:]]*)?910B3([[:space:]]|$)' <<< "${product}" || {
        echo "ERROR: device ${device} is not an Ascend 910B3" >&2
        exit 1
    }
    grep -Eq '^[[:space:]]*Health[[:space:]]*:[[:space:]]*OK[[:space:]]*$' <<< "${health}" || {
        echo "ERROR: device ${device} is not healthy" >&2
        exit 1
    }
done

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
