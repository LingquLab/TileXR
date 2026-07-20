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
    local fixture="${temp_dir}/${name}"
    local status

    mkdir -p "${fixture}/scripts" "${fixture}/mock-bin" \
        "${fixture}/temp" "${fixture}/cann"
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
        bash "${fixture}/scripts/cann_download_install.sh" \
        > "${fixture}/downloader.log" 2>&1
    status=$?
    set -e

    [[ -f "${fixture}/curl-toolkit.done" && -f "${fixture}/curl-ops.done" ]] || {
        echo "${name}: downloader did not wait for both curl children" >&2
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

missing_fixture="${temp_dir}/missing-home"
mkdir -p "${missing_fixture}/scripts" "${missing_fixture}/temp"
cp "${downloader}" "${missing_fixture}/scripts/cann_download_install.sh"
printf '%s\n' \
    ': "${TEST_FIXTURE_ROOT:?}"' \
    'TILEXR_CANN_HOME="${TEST_FIXTURE_ROOT}/cann"' \
    'TILEXR_TEMP_HOME="${TEST_FIXTURE_ROOT}/temp"' \
    'TILEXR_HOME="${TEST_FIXTURE_ROOT}"' \
    'env_print() { :; }' \
    'error() { :; }' \
    > "${missing_fixture}/scripts/common_env.sh"
if TEST_FIXTURE_ROOT="${missing_fixture}" \
    bash "${missing_fixture}/scripts/cann_download_install.sh" >/dev/null 2>&1; then
    echo "downloader accepted a CANN home it did not own" >&2
    exit 1
fi
if [[ -e "${missing_fixture}/cann" || -L "${missing_fixture}/cann" ]]; then
    echo "downloader created an unowned CANN home" >&2
    exit 1
fi

if grep -Eq '_ensure_curl_running|/proc/[^[:space:]]*/comm|cann_(toolkit|ops)\.pid' \
    "${downloader}"; then
    echo "downloader must not adopt unverifiable orphan curl PIDs" >&2
    exit 1
fi
