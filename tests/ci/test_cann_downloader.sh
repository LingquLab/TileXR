#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
downloader="${repo_root}/scripts/cann_download_install.sh"
temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT

run_scenario() {
    local name="$1"
    local toolkit_status="$2"
    local ops_status="$3"
    local expected_status="$4"
    local sealed_mode="${5:-0}"
    local fixture="${temp_dir}/${name}"
    local status

    mkdir -p "${fixture}/scripts" "${fixture}/mock-bin" "${fixture}/temp"
    if [[ "${sealed_mode}" == 1 ]]; then
        mkdir "${fixture}/cann"
        printf marker > "${fixture}/cann/.tilexr-ci-installing"
    fi
    cp "${downloader}" "${fixture}/scripts/cann_download_install.sh"

    printf '%s\n' \
        ': "${TEST_FIXTURE_ROOT:?}"' \
        'TILEXR_CANN_HOME="${TEST_FIXTURE_ROOT}/cann"' \
        'TILEXR_TEMP_HOME="${TEST_FIXTURE_ROOT}/temp"' \
        'TILEXR_HOME="${TEST_FIXTURE_ROOT}"' \
        'TILEXR_CANN_VER=9.1.0' \
        'TILEXR_OS_ARCH=aarch64' \
        'TILEXR_OPS_NAME=910b' \
        'env_print() { :; }' \
        'line() { :; }' \
        'success() { :; }' \
        'warn() { :; }' \
        'error() { printf "ERROR: %s\\n" "$*" >&2; }' \
        > "${fixture}/scripts/common_env.sh"

    printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        'printf installed > "${TEST_INSTALL_MARKER}"' \
        > "${fixture}/scripts/cann_local_install.sh"

    printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        'url=""' \
        'for argument in "$@"; do url="${argument}"; done' \
        'filename="${url##*/}"' \
        'case "${filename}" in' \
        '    Ascend-cann-toolkit_9.1.0_linux-aarch64.run)' \
        '        kind=toolkit; status="${MOCK_TOOLKIT_STATUS}" ;;' \
        '    Ascend-cann-910b-ops_9.1.0_linux-aarch64.run)' \
        '        kind=ops; status="${MOCK_OPS_STATUS}" ;;' \
        '    *) exit 93 ;;' \
        'esac' \
        'printf partial > "${filename}"' \
        'printf done > "${TEST_FIXTURE_ROOT}/curl-${kind}.done"' \
        'exit "${status}"' \
        > "${fixture}/mock-bin/curl"
    chmod +x "${fixture}/mock-bin/curl"

    set +e
    PATH="${fixture}/mock-bin:${PATH}" \
        TEST_FIXTURE_ROOT="${fixture}" \
        TEST_INSTALL_MARKER="${fixture}/installed" \
        MOCK_TOOLKIT_STATUS="${toolkit_status}" \
        MOCK_OPS_STATUS="${ops_status}" \
        TILEXR_CI_SEALED_CANN_HOME="${sealed_mode}" \
        bash "${fixture}/scripts/cann_download_install.sh" \
        > "${fixture}/downloader.log" 2>&1
    status=$?
    set -e

    [[ -f "${fixture}/curl-toolkit.done" && -f "${fixture}/curl-ops.done" ]] || {
        echo "${name}: downloader did not wait for both curl children" >&2
        exit 1
    }
    [[ -d "${fixture}/cann" && ! -L "${fixture}/cann" ]] || {
        echo "${name}: standalone downloader did not create a real CANN home" >&2
        exit 1
    }
    if [[ "${expected_status}" == success ]]; then
        [[ "${status}" -eq 0 && -f "${fixture}/installed" ]] || {
            echo "${name}: successful downloads did not install" >&2
            cat "${fixture}/downloader.log" >&2
            exit 1
        }
    else
        [[ "${status}" -ne 0 && ! -e "${fixture}/installed" ]] || {
            echo "${name}: failed partial download was accepted" >&2
            cat "${fixture}/downloader.log" >&2
            exit 1
        }
    fi
}

run_scenario toolkit-fails 22 0 failure
run_scenario ops-fails 0 23 failure
run_scenario both-succeed 0 0 success
run_scenario sealed-succeeds 0 0 success 1

missing_fixture="${temp_dir}/missing-home"
mkdir -p \
    "${missing_fixture}/scripts" \
    "${missing_fixture}/temp" \
    "${missing_fixture}/mock-bin"
cp "${downloader}" "${missing_fixture}/scripts/cann_download_install.sh"
printf '%s\n' \
    ': "${TEST_FIXTURE_ROOT:?}"' \
    'TILEXR_CANN_HOME="${TEST_FIXTURE_ROOT}/cann"' \
    'TILEXR_TEMP_HOME="${TEST_FIXTURE_ROOT}/temp"' \
    'TILEXR_HOME="${TEST_FIXTURE_ROOT}"' \
    'env_print() { :; }' \
    'error() { :; }' \
    > "${missing_fixture}/scripts/common_env.sh"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'printf called > "${TEST_CURL_MARKER}"' \
    'exit 99' \
    > "${missing_fixture}/mock-bin/curl"
chmod +x "${missing_fixture}/mock-bin/curl"

if PATH="${missing_fixture}/mock-bin:${PATH}" \
    TEST_FIXTURE_ROOT="${missing_fixture}" \
    TEST_CURL_MARKER="${missing_fixture}/curl-called" \
    TILEXR_CI_SEALED_CANN_HOME=1 \
    bash "${missing_fixture}/scripts/cann_download_install.sh" >/dev/null 2>&1; then
    echo "sealed downloader accepted a missing CANN home" >&2
    exit 1
fi
if [[ -e "${missing_fixture}/cann" || -L "${missing_fixture}/cann" ]]; then
    echo "sealed downloader created a missing CANN home" >&2
    exit 1
fi
if [[ -e "${missing_fixture}/curl-called" ]]; then
    echo "sealed downloader started downloads before validating its CANN home" >&2
    exit 1
fi

mkdir "${missing_fixture}/cann"
if PATH="${missing_fixture}/mock-bin:${PATH}" \
    TEST_FIXTURE_ROOT="${missing_fixture}" \
    TEST_CURL_MARKER="${missing_fixture}/curl-called" \
    TILEXR_CI_SEALED_CANN_HOME=1 \
    bash "${missing_fixture}/scripts/cann_download_install.sh" >/dev/null 2>&1; then
    echo "sealed downloader accepted an unmarked CANN home" >&2
    exit 1
fi
if [[ -e "${missing_fixture}/curl-called" ]]; then
    echo "sealed downloader started downloads before validating its marker" >&2
    exit 1
fi
rmdir "${missing_fixture}/cann"

replacement_target="${missing_fixture}/replacement-target"
mkdir "${replacement_target}"
printf external > "${replacement_target}/sentinel"
printf marker > "${replacement_target}/.tilexr-ci-installing"
ln -s "${replacement_target}" "${missing_fixture}/cann"
if PATH="${missing_fixture}/mock-bin:${PATH}" \
    TEST_FIXTURE_ROOT="${missing_fixture}" \
    TEST_CURL_MARKER="${missing_fixture}/curl-called" \
    TILEXR_CI_SEALED_CANN_HOME=1 \
    bash "${missing_fixture}/scripts/cann_download_install.sh" >/dev/null 2>&1; then
    echo "sealed downloader accepted a replaced CANN home" >&2
    exit 1
fi
if [[ ! -L "${missing_fixture}/cann" ||
      "$(< "${replacement_target}/sentinel")" != external ]]; then
    echo "sealed downloader mutated a replaced CANN home" >&2
    exit 1
fi
if [[ -e "${missing_fixture}/curl-called" ]]; then
    echo "sealed downloader started downloads for a replaced CANN home" >&2
    exit 1
fi

if grep -Eq '_ensure_curl_running|/proc/[^[:space:]]*/comm|cann_(toolkit|ops)\.pid' \
    "${downloader}"; then
    echo "downloader must not adopt unverifiable orphan curl PIDs" >&2
    exit 1
fi
