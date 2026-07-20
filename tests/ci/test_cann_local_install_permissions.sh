#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
real_installer="${repo_root}/scripts/cann_local_install.sh"
temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT

fixture="${temp_dir}/fixture"
cann_home="${fixture}/cann"
temp_home="${fixture}/temp"
mkdir -p "${fixture}/scripts" "${cann_home}" "${temp_home}"
cp "${real_installer}" "${fixture}/scripts/cann_local_install.sh"

printf '%s\n' \
    ': "${TEST_FIXTURE:?}"' \
    'TILEXR_CANN_HOME="${TEST_FIXTURE}/cann"' \
    'TILEXR_TEMP_HOME="${TEST_FIXTURE}/temp"' \
    'TILEXR_CANN_VER=9.1.0' \
    'TILEXR_OS_ARCH=aarch64' \
    'TILEXR_OPS_NAME=910b' \
    'env_print() { :; }' \
    'line() { :; }' \
    'error() { printf "ERROR: %s\n" "$*" >&2; }' \
    'fix_permissions() { printf "%s\n" "$1" >> "${TEST_FIX_LOG}"; }' \
    'colorful_time() { return 0; }' \
    > "${fixture}/scripts/common_env.sh"
printf toolkit > "${temp_home}/Ascend-cann-toolkit_9.1.0_linux-aarch64.run"
printf ops > "${temp_home}/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run"

fix_log="${fixture}/fix.log"
TEST_FIXTURE="${fixture}" TEST_FIX_LOG="${fix_log}" \
    bash "${fixture}/scripts/cann_local_install.sh"
[[ "$(< "${fix_log}")" == "${cann_home}" ]] || {
    echo "default local install no longer fixes ancestor permissions" >&2
    exit 1
}

rm -f "${fix_log}"
printf owned > "${cann_home}/.tilexr-ci-installing"
TEST_FIXTURE="${fixture}" TEST_FIX_LOG="${fix_log}" \
    TILEXR_CI_SEALED_CANN_HOME=1 \
    bash "${fixture}/scripts/cann_local_install.sh"
[[ ! -e "${fix_log}" ]] || {
    echo "sealed CI install still called fix_permissions" >&2
    exit 1
}

rm -f "${cann_home}/.tilexr-ci-installing"
if TEST_FIXTURE="${fixture}" TEST_FIX_LOG="${fix_log}" \
    TILEXR_CI_SEALED_CANN_HOME=1 \
    bash "${fixture}/scripts/cann_local_install.sh" >/dev/null 2>&1; then
    echo "sealed CI permission bypass did not require its ownership marker" >&2
    exit 1
fi

printf owned > "${cann_home}/.tilexr-ci-installing"
mv "${cann_home}" "${cann_home}.real"
ln -s "${cann_home}.real" "${cann_home}"
if TEST_FIXTURE="${fixture}" TEST_FIX_LOG="${fix_log}" \
    TILEXR_CI_SEALED_CANN_HOME=1 \
    bash "${fixture}/scripts/cann_local_install.sh" >/dev/null 2>&1; then
    echo "sealed CI permission bypass accepted a symlinked CANN home" >&2
    exit 1
fi
