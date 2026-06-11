#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${ROOT_DIR}/reference/download_cann_repos.sh"
README="${ROOT_DIR}/reference/README.md"
legacy_scripts=(
    "${ROOT_DIR}/reference/download_shmem.sh"
    "${ROOT_DIR}/reference/download_ascend_transformer_boost.sh"
)

if [[ ! -x "${SCRIPT}" ]]; then
    echo "script is not executable: ${SCRIPT}" >&2
    exit 1
fi

for legacy_script in "${legacy_scripts[@]}"; do
    if [[ -e "${legacy_script}" ]]; then
        echo "legacy per-repo script should be merged into download_cann_repos.sh: ${legacy_script}" >&2
        exit 1
    fi
done

assert_contains() {
    local haystack=$1
    local needle=$2
    if [[ "${haystack}" != *"${needle}"* ]]; then
        echo "expected output to contain: ${needle}" >&2
        echo "actual output:" >&2
        echo "${haystack}" >&2
        exit 1
    fi
}

list_output="$(bash "${SCRIPT}" --list)"
assert_contains "${list_output}" "asc-devkit"
assert_contains "${list_output}" "runtime"
assert_contains "${list_output}" "driver"
assert_contains "${list_output}" "asc-tools"
assert_contains "${list_output}" "oam-tools"
assert_contains "${list_output}" "ascend-transformer-boost"
assert_contains "${list_output}" "shmem"

dry_run_output="$(bash "${SCRIPT}" --dry-run)"
assert_contains "${dry_run_output}" "repo: https://gitcode.com/cann/asc-devkit.git"
assert_contains "${dry_run_output}" "branch: master"
assert_contains "${dry_run_output}" "target: ${ROOT_DIR}/reference/asc-devkit"
assert_contains "${dry_run_output}" "repo: https://gitcode.com/cann/oam-tools.git"
assert_contains "${dry_run_output}" "repo: https://gitcode.com/cann/ascend-transformer-boost.git"
assert_contains "${dry_run_output}" "repo: https://gitcode.com/cann/shmem.git"
assert_contains "${dry_run_output}" "target: ${ROOT_DIR}/reference/shmem"

subset_output="$(CANN_RUNTIME_BRANCH=release-test bash "${SCRIPT}" --dry-run runtime)"
assert_contains "${subset_output}" "repo: https://gitcode.com/cann/runtime.git"
assert_contains "${subset_output}" "branch: release-test"
if [[ "${subset_output}" == *"asc-devkit"* ]]; then
    echo "runtime subset unexpectedly included asc-devkit" >&2
    echo "${subset_output}" >&2
    exit 1
fi

shmem_output="$(SHMEM_BRANCH=shmem-test bash "${SCRIPT}" --dry-run shmem)"
assert_contains "${shmem_output}" "repo: https://gitcode.com/cann/shmem.git"
assert_contains "${shmem_output}" "branch: shmem-test"
assert_contains "${shmem_output}" "target: ${ROOT_DIR}/reference/shmem"

atb_output="$(ATB_BRANCH=atb-test bash "${SCRIPT}" --dry-run ascend-transformer-boost)"
assert_contains "${atb_output}" "repo: https://gitcode.com/cann/ascend-transformer-boost.git"
assert_contains "${atb_output}" "branch: atb-test"
assert_contains "${atb_output}" "target: ${ROOT_DIR}/reference/ascend-transformer-boost"

if bash "${SCRIPT}" --dry-run not-a-repo >/tmp/tilexr-cann-repos-invalid.out 2>&1; then
    echo "invalid repository name unexpectedly succeeded" >&2
    cat /tmp/tilexr-cann-repos-invalid.out >&2
    exit 1
fi
assert_contains "$(cat /tmp/tilexr-cann-repos-invalid.out)" "unknown CANN reference repository: not-a-repo"

if grep -q "current TileXR build targets must not include or link it" "${README}"; then
    echo "reference README still contains removed build-target wording" >&2
    exit 1
fi

readme_text="$(cat "${README}")"
assert_contains "${readme_text}" "## Repository Guide"
assert_contains "${readme_text}" '| `asc-devkit` |'
assert_contains "${readme_text}" '| `runtime` |'
assert_contains "${readme_text}" '| `driver` |'
assert_contains "${readme_text}" '| `asc-tools` |'
assert_contains "${readme_text}" '| `oam-tools` |'
assert_contains "${readme_text}" '| `ascend-transformer-boost` |'
assert_contains "${readme_text}" '| `shmem` |'
