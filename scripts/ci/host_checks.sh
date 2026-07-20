#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_BUILD_PARENT="${ROOT_DIR}/.ci-build"

build_root_error() {
    echo "ERROR: refusing unsafe TILEXR_CI_BUILD_ROOT: $1" >&2
    exit 2
}

if [[ "${TILEXR_CI_BUILD_ROOT+x}" == "x" ]]; then
    if [[ -z "${TILEXR_CI_BUILD_ROOT}" || "${TILEXR_CI_BUILD_ROOT}" != /* ]]; then
        build_root_error "${TILEXR_CI_BUILD_ROOT:-<empty>}"
    fi
    BUILD_ROOT_INPUT="${TILEXR_CI_BUILD_ROOT}"
else
    BUILD_ROOT_INPUT="${REPO_BUILD_PARENT}/tilexr-host-default"
fi

BUILD_ROOT="$(
    python3 -c 'import os, sys; print(os.path.normpath(sys.argv[1]))' \
        "${BUILD_ROOT_INPUT}"
)"
BUILD_PARENT="$(dirname -- "${BUILD_ROOT}")"
BUILD_BASENAME="$(basename -- "${BUILD_ROOT}")"
if [[ ! "${BUILD_BASENAME}" =~ ^\.?tilexr-host-.+ ]]; then
    build_root_error "${BUILD_ROOT}"
fi

if [[ "${BUILD_PARENT}" == "${REPO_BUILD_PARENT}" ]]; then
    if [[ -L "${REPO_BUILD_PARENT}" || \
          ( -e "${REPO_BUILD_PARENT}" && ! -d "${REPO_BUILD_PARENT}" ) ]]; then
        build_root_error "${BUILD_ROOT} (invalid repository build parent)"
    fi
    mkdir -p -- "${REPO_BUILD_PARENT}"
    BUILD_PARENT_REAL="$(cd "${REPO_BUILD_PARENT}" && pwd -P)"
    if [[ "${BUILD_PARENT_REAL}" != "${REPO_BUILD_PARENT}" ]]; then
        build_root_error "${BUILD_ROOT} (repository build parent is redirected)"
    fi
else
    RUNNER_BUILD_PARENT="${RUNNER_TEMP:-/tmp}"
    if [[ -z "${RUNNER_BUILD_PARENT}" || "${RUNNER_BUILD_PARENT}" != /* ]]; then
        build_root_error "${BUILD_ROOT} (invalid RUNNER_TEMP)"
    fi
    RUNNER_BUILD_PARENT="$(
        python3 -c 'import os, sys; print(os.path.normpath(sys.argv[1]))' \
            "${RUNNER_BUILD_PARENT}"
    )"
    if [[ "${BUILD_PARENT}" != "${RUNNER_BUILD_PARENT}" || \
          -L "${RUNNER_BUILD_PARENT}" || \
          ! -d "${RUNNER_BUILD_PARENT}" ]]; then
        build_root_error "${BUILD_ROOT} (outside an allowed build parent)"
    fi
    BUILD_PARENT_REAL="$(cd "${RUNNER_BUILD_PARENT}" && pwd -P)"
    if [[ "${BUILD_PARENT_REAL}" != "${RUNNER_BUILD_PARENT}" ]]; then
        build_root_error "${BUILD_ROOT} (runner build parent is redirected)"
    fi
fi

BUILD_ROOT="${BUILD_PARENT_REAL}/${BUILD_BASENAME}"
if [[ -L "${BUILD_ROOT}" ]]; then
    build_root_error "${BUILD_ROOT} (build root is a symbolic link)"
fi

ARTIFACT_DIR="${ROOT_DIR}/.ci-artifacts/host"
CASES_FILE="${ARTIFACT_DIR}/cases.tsv"
SUMMARY_FILE="${ARTIFACT_DIR}/summary.md"
HOST_STARTED="$(date +%s)"

if [[ -L "${ARTIFACT_DIR}" ]]; then
    echo "ERROR: host artifact directory must not be a symbolic link: ${ARTIFACT_DIR}" >&2
    exit 2
fi

rm -rf -- "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}" "${ARTIFACT_DIR}"
rm -f -- \
    "${CASES_FILE}" \
    "${SUMMARY_FILE}" \
    "${ARTIFACT_DIR}/ctest-ci.xml" \
    "${ARTIFACT_DIR}/ctest-ep.xml" \
    "${ARTIFACT_DIR}/ctest-data-as-flag.xml"
: > "${CASES_FILE}"

if ! command -v nproc >/dev/null 2>&1; then
    nproc() {
        if command -v getconf >/dev/null 2>&1; then
            getconf _NPROCESSORS_ONLN
        else
            sysctl -n hw.ncpu
        fi
    }
fi

run_case() {
    local name="$1"
    shift
    local started finished duration status result
    started="$(date +%s)"
    set +e
    (set -e; "$@")
    status="$?"
    set -e
    finished="$(date +%s)"
    duration=$((finished - started))
    result="PASS"
    if [[ "${status}" -ne 0 ]]; then
        result="FAIL"
    fi
    printf '%s\t%s\t%s\t%s\n' \
        "${name}" "${result}" "${status}" "${duration}" >> "${CASES_FILE}"
    return "${status}"
}

copy_ctest_xml() {
    local testing_dir="$1"
    local destination="$2"
    local test_xml
    test_xml="$(find "${testing_dir}" -name Test.xml -print -quit)"
    if [[ -z "${test_xml}" || ! -f "${test_xml}" ]]; then
        echo "ERROR: CTest did not produce Test.xml under ${testing_dir}" >&2
        return 1
    fi
    cp "${test_xml}" "${destination}"
}

check_tracked_shell_syntax() {
    local relative
    local -a tracked_shell_files=()
    while IFS= read -r -d '' relative; do
        if [[ "${relative}" == *.sh ]]; then
            tracked_shell_files+=("${ROOT_DIR}/${relative}")
        fi
    done < <(
        git -C "${ROOT_DIR}" ls-files -z -- \
            scripts/ci \
            tests/comm \
            tests/ep \
            tests/memory \
            tests/sdma \
            tests/udma \
            tests/collectives
    )
    if [[ "${#tracked_shell_files[@]}" -eq 0 ]]; then
        echo "ERROR: tracked shell syntax scan found no scripts" >&2
        return 1
    fi
    bash -n "${tracked_shell_files[@]}"
}

run_ci_ctest() {
    cmake -S "${ROOT_DIR}/tests/ci" -B "${BUILD_ROOT}/ci"
    (cd "${BUILD_ROOT}/ci" && ctest --output-on-failure -T Test --no-compress-output)
    copy_ctest_xml "${BUILD_ROOT}/ci/Testing" "${ARTIFACT_DIR}/ctest-ci.xml"
}

run_comm_host_tests() {
    cmake -S "${ROOT_DIR}/tests/comm" -B "${BUILD_ROOT}/comm" \
        -DCMAKE_INSTALL_PREFIX="${BUILD_ROOT}/comm-install"
    cmake --build "${BUILD_ROOT}/comm" --target install -j"$(nproc)"
    "${BUILD_ROOT}/comm-install/bin/test_tilexr_log"
    "${BUILD_ROOT}/comm-install/bin/test_tilexr_log_spdlog_compile"
    "${BUILD_ROOT}/comm-install/bin/test_tilexr_source_guards"
}

run_ep_source_tests() {
    cmake -S "${ROOT_DIR}/tests/ep" -B "${BUILD_ROOT}/ep" \
        -DBUILD_TILEXR_EP_DEMO=OFF
    cmake --build "${BUILD_ROOT}/ep" -j"$(nproc)"
    (cd "${BUILD_ROOT}/ep" && ctest --output-on-failure -T Test --no-compress-output)
    copy_ctest_xml "${BUILD_ROOT}/ep/Testing" "${ARTIFACT_DIR}/ctest-ep.xml"
}

run_data_as_flag_tests() {
    cmake -S "${ROOT_DIR}/tests/data_as_flag" -B "${BUILD_ROOT}/data-as-flag"
    cmake --build "${BUILD_ROOT}/data-as-flag" -j"$(nproc)"
    (cd "${BUILD_ROOT}/data-as-flag" && ctest --output-on-failure -T Test --no-compress-output)
    copy_ctest_xml \
        "${BUILD_ROOT}/data-as-flag/Testing" \
        "${ARTIFACT_DIR}/ctest-data-as-flag.xml"
}

finalize_host_checks() {
    local status="$?"
    local finished duration total passed failed failed_names
    local name result code case_duration
    trap - EXIT
    set +e

    finished="$(date +%s)"
    duration=$((finished - HOST_STARTED))
    total=0
    passed=0
    failed=0
    failed_names=""
    while IFS=$'\t' read -r name result code case_duration; do
        [[ -n "${name}" ]] || continue
        total=$((total + 1))
        if [[ "${result}" == "PASS" ]]; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
            if [[ -n "${failed_names}" ]]; then
                failed_names="${failed_names}, ${name}"
            else
                failed_names="${name}"
            fi
        fi
    done < "${CASES_FILE}"
    failed_names="${failed_names:-none}"

    {
        printf '# TileXR Host Checks\n\n'
        printf -- '- PR: `%s`\n' "${TILEXR_CI_PR_NUMBER:-local}"
        printf -- '- Head SHA: `%s`\n' "${TILEXR_CI_HEAD_SHA:-local}"
        printf -- '- Base SHA: `%s`\n' "${TILEXR_CI_BASE_SHA:-local}"
        printf -- '- Merge SHA: `%s`\n' "${TILEXR_CI_EXPECTED_MERGE_SHA:-local}"
        printf -- '- Host cases: %s total, %s passed, %s failed\n' \
            "${total}" "${passed}" "${failed}"
        printf -- '- Failed cases: %s\n' "${failed_names}"
        printf -- '- Duration: %s seconds\n' "${duration}"
        printf -- '- Exit code: %s\n' "${status}"
        printf -- '- NPU-only coverage is not run by Host Checks; it runs in the queued NPU Gate on blue.\n'
    } > "${SUMMARY_FILE}"
    local summary_status="$?"

    if [[ -n "${GITHUB_STEP_SUMMARY:-}" && "${summary_status}" -eq 0 ]]; then
        cat "${SUMMARY_FILE}" >> "${GITHUB_STEP_SUMMARY}"
        if [[ "$?" -ne 0 && "${status}" -eq 0 ]]; then
            status=1
        fi
    fi
    if [[ "${summary_status}" -ne 0 && "${status}" -eq 0 ]]; then
        status="${summary_status}"
    fi
    exit "${status}"
}

trap finalize_host_checks EXIT

run_case shell-syntax check_tracked_shell_syntax
run_case ci-ctest run_ci_ctest
run_case comm-host run_comm_host_tests
run_case ep-source-only run_ep_source_tests
run_case data-as-flag run_data_as_flag_tests
run_case collectives-vllm-patch \
    python3 -m pytest -q "${ROOT_DIR}/tests/collectives/unit/test_vllm_collectives_patch.py"
run_case collectives-vllm-integration-sources \
    python3 -m pytest -q "${ROOT_DIR}/tests/collectives/unit/test_vllm_collectives_integration_sources.py"
run_case collectives-profile-report \
    python3 "${ROOT_DIR}/tests/collectives/unit/test_collective_profile_report.py"
