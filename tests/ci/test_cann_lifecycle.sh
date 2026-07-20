#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
real_common="${repo_root}/scripts/ci/provision/common.sh"
real_cann="${repo_root}/scripts/ci/provision/cann.sh"
temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT

fixture="${temp_dir}/fixture"
ci_home="${fixture}/ci-home"
cann_home="${ci_home}/toolchains/cann/9.1.0"
external_tree="${fixture}/external-tree"
current_user="$(id -un)"
current_group="$(id -gn)"
mkdir -p \
    "${fixture}/scripts/ci/provision" \
    "${fixture}/mock-bin" \
    "${ci_home}/toolchains/cann" \
    "${ci_home}/install-work" \
    "${external_tree}"
chmod 0750 \
    "${ci_home}" \
    "${ci_home}/toolchains" \
    "${ci_home}/toolchains/cann" \
    "${ci_home}/install-work"
printf external > "${external_tree}/sentinel"
cp "${real_cann}" "${fixture}/scripts/ci/provision/cann.sh"

{
    printf 'source %q\n' "${real_common}"
    printf '%s\n' \
        'CI_HOME="${TEST_CI_HOME:?}"' \
        'CANN_HOME="${TEST_CANN_HOME:?}"' \
        'CI_GROUP="${TEST_GROUP:?}"' \
        'CI_PRIMARY_GROUP="${TEST_GROUP:?}"' \
        'CANN_OWNER="${TEST_OWNER:?}"' \
        'require_root() { :; }'
} > "${fixture}/scripts/ci/provision/common.sh"

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    '[[ -f "${TILEXR_CANN_HOME}/.tilexr-ci-installing" && ! -L "${TILEXR_CANN_HOME}/.tilexr-ci-installing" ]] || exit 96' \
    'stage_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' \
    'download_dir="${stage_root}/env/temp"' \
    'mkdir -p "${download_dir}"' \
    'write_downloads() {' \
    '    printf toolkit > "${download_dir}/Ascend-cann-toolkit_9.1.0_linux-aarch64.run"' \
    '    printf ops > "${download_dir}/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run"' \
    '}' \
    'write_valid_tree() {' \
    '    local toolkit_info="${TILEXR_CANN_HOME}/cann/aarch64-linux/ascend_toolkit_install.info"' \
    '    local ops_info="${TILEXR_CANN_HOME}/cann/aarch64-linux/ascend_ops_install.info"' \
    '    local compiler_dir="${TILEXR_CANN_HOME}/cann/compiler/ccec_compiler/bin"' \
    '    mkdir -p "$(dirname "${toolkit_info}")" "${compiler_dir}"' \
    '    printf "%s\n" package_name=Ascend-cann-toolkit version=9.1.0 > "${toolkit_info}"' \
    '    printf "%s\n" package_name=Ascend-cann-910b-ops version=9.1.0 > "${ops_info}"' \
    '    printf "%s\n" "#!/usr/bin/env bash" "exit 0" > "${compiler_dir}/bisheng-real"' \
    '    chmod 0750 "${compiler_dir}/bisheng-real"' \
    '    ln -s bisheng-real "${compiler_dir}/bisheng"' \
    '}' \
    'simulate_ancestor_permission_fix() {' \
    '    if [[ "${TILEXR_CI_SEALED_CANN_HOME:-0}" != 1 ]]; then' \
    '        chmod 0755 "${TEST_CI_HOME}" "${TEST_CI_HOME}/toolchains" "${TEST_CI_HOME}/toolchains/cann"' \
    '    fi' \
    '}' \
    'case "${TEST_PHASE}" in' \
    '    fail-download)' \
    '        printf partial > "${TILEXR_CANN_HOME}/download.partial"; exit 21 ;;' \
    '    fail-install)' \
    '        write_downloads; simulate_ancestor_permission_fix' \
    '        printf partial > "${TILEXR_CANN_HOME}/install.partial"; exit 22 ;;' \
    '    fail-validation)' \
    '        write_downloads; simulate_ancestor_permission_fix' \
    '        mkdir -p "${TILEXR_CANN_HOME}/cann"; printf invalid > "${TILEXR_CANN_HOME}/cann/invalid" ;;' \
    '    success)' \
    '        write_downloads; simulate_ancestor_permission_fix; write_valid_tree ;;' \
    '    mutate-parents-fail)' \
    '        chmod 0755 "${TEST_CI_HOME}" "${TEST_CI_HOME}/toolchains" "${TEST_CI_HOME}/toolchains/cann"' \
    '        exit 20 ;;' \
    '    replacement)' \
    '        mv "${TILEXR_CANN_HOME}" "${TILEXR_CANN_HOME}.owned"' \
    '        mkdir "${TILEXR_CANN_HOME}"' \
    '        printf replacement > "${TILEXR_CANN_HOME}/sentinel"' \
    '        exit 23 ;;' \
    '    symlink-replacement)' \
    '        mv "${TILEXR_CANN_HOME}" "${TILEXR_CANN_HOME}.owned"' \
    '        ln -s "${TEST_EXTERNAL_TREE}" "${TILEXR_CANN_HOME}"' \
    '        exit 24 ;;' \
    '    marker-replacement)' \
    '        marker="${TILEXR_CANN_HOME}/.tilexr-ci-installing"' \
    '        token="$(< "${marker}")"' \
    '        mv "${marker}" "${marker}.owned"' \
    '        printf "%s\n" "${token}" > "${marker}"' \
    '        printf marker-replacement > "${TILEXR_CANN_HOME}/sentinel"' \
    '        exit 25 ;;' \
    '    parent-replacement)' \
    '        parent="${TEST_CI_HOME}/toolchains"' \
    '        mv "${parent}" "${parent}.owned"' \
    '        mkdir -p "${parent}/cann"' \
    '        chmod 0711 "${parent}"' \
    '        printf parent-replacement > "${parent}/sentinel"' \
    '        exit 26 ;;' \
    '    parent-symlink)' \
    '        parent="${TEST_CI_HOME}/toolchains"' \
    '        mv "${parent}" "${parent}.owned"' \
    '        ln -s "${TEST_EXTERNAL_TREE}" "${parent}"' \
    '        exit 27 ;;' \
    '    *) exit 97 ;;' \
    'esac' \
    > "${fixture}/scripts/cann_download_install.sh"

for file in cann_local_install.sh common_env.sh common_util.sh; do
    printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "${fixture}/scripts/${file}"
done

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'directory_mode=0' \
    'mode=' \
    'operands=()' \
    'while [[ "$#" -gt 0 ]]; do' \
    '    case "$1" in' \
    '        -d) directory_mode=1; shift ;;' \
    '        -o|-g|-m) [[ "$1" == -m ]] && mode="$2"; shift 2 ;;' \
    '        *) operands+=("$1"); shift ;;' \
    '    esac' \
    'done' \
    'if [[ "${directory_mode}" -eq 1 ]]; then' \
    '    mkdir -p "${operands[@]}"' \
    '    [[ -z "${mode}" ]] || chmod "${mode}" "${operands[@]}"' \
    'else' \
    '    cp "${operands[0]}" "${operands[1]}"' \
    '    [[ -z "${mode}" ]] || chmod "${mode}" "${operands[1]}"' \
    'fi' \
    > "${fixture}/mock-bin/install"

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'printf "%s\n" "$*" >> "${TEST_CHOWN_LOG}"' \
    'exit 0' \
    > "${fixture}/mock-bin/chown"

printf '%s\n' '#!/usr/bin/env bash' 'printf "25.5.0\n"' \
    > "${fixture}/mock-bin/sed"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$*" in' \
    '    "info -l") printf "Total Count: 8\n" ;;' \
    '    *"-t product"*) printf "Name : Ascend 910B3\n" ;;' \
    '    *"-t health"*) printf "Health : OK\n" ;;' \
    '    *) exit 98 ;;' \
    'esac' \
    > "${fixture}/mock-bin/npu-smi"
printf '%s\n' '#!/usr/bin/env bash' 'printf "Avail\n68719476736\n"' \
    > "${fixture}/mock-bin/df"
chmod +x "${fixture}/mock-bin/"*

run_phase() {
    local phase="$1"
    local status
    set +e
    PATH="${fixture}/mock-bin:${PATH}" \
        TEST_PHASE="${phase}" \
        TEST_CI_HOME="${ci_home}" \
        TEST_CANN_HOME="${cann_home}" \
        TEST_GROUP="${current_group}" \
        TEST_OWNER="${current_user}" \
        TEST_EXTERNAL_TREE="${external_tree}" \
        TEST_CHOWN_LOG="${fixture}/chown.log" \
        bash "${fixture}/scripts/ci/provision/cann.sh" \
        > "${fixture}/${phase}.log" 2>&1
    status=$?
    set -e
    return "${status}"
}

entry_metadata() {
    local path="$1"
    stat -c '%U:%G:%a' "${path}" 2>/dev/null ||
        stat -f '%Su:%Sg:%Lp' "${path}"
}

assert_sealed_parents() {
    local expected_home="${current_user}:${current_group}:750"
    local expected_toolchain="${current_user}:${current_group}:750"
    local path expected
    while IFS=: read -r path expected; do
        [[ -d "${path}" && ! -L "${path}" ]] || {
            echo "CANN parent is not a real directory: ${path}" >&2
            exit 1
        }
        [[ "$(entry_metadata "${path}")" == "${expected}" ]] || {
            echo "CANN parent is not sealed: ${path} ($(entry_metadata "${path}"))" >&2
            exit 1
        }
    done <<EOF
${ci_home}:${expected_home}
${ci_home}/toolchains:${expected_toolchain}
${ci_home}/toolchains/cann:${expected_toolchain}
EOF
}

assert_failure_then_success() {
    local phase="$1"
    if run_phase "${phase}"; then
        echo "${phase}: expected provisioning failure" >&2
        exit 1
    fi
    [[ ! -e "${cann_home}" && ! -L "${cann_home}" ]] || {
        echo "${phase}: current-invocation partial tree was not cleaned" >&2
        exit 1
    }
    assert_sealed_parents
    if ! run_phase success; then
        echo "${phase}: clean rerun did not succeed" >&2
        cat "${fixture}/success.log" >&2
        exit 1
    fi
    [[ -d "${cann_home}" && ! -e "${cann_home}/.tilexr-ci-installing" ]] || {
        echo "${phase}: successful rerun did not finalize the CANN tree" >&2
        exit 1
    }
    assert_sealed_parents
    if ! run_phase success; then
        echo "${phase}: finalized CANN tree was not idempotently accepted" >&2
        cat "${fixture}/success.log" >&2
        exit 1
    fi
    rm -rf "${cann_home}"
}

assert_failure_then_success fail-download
assert_failure_then_success fail-install
assert_failure_then_success fail-validation
assert_failure_then_success mutate-parents-fail

if run_phase replacement; then
    echo "replacement: expected provisioning failure" >&2
    exit 1
fi
[[ "$(< "${cann_home}/sentinel")" == replacement ]] || {
    echo "replacement tree was deleted by stale cleanup ownership" >&2
    exit 1
}
rm -rf "${cann_home}" "${cann_home}.owned"

if run_phase symlink-replacement; then
    echo "symlink replacement: expected provisioning failure" >&2
    exit 1
fi
[[ -L "${cann_home}" && "$(< "${external_tree}/sentinel")" == external ]] || {
    echo "symlink replacement or its target was deleted" >&2
    exit 1
}
rm -f "${cann_home}"
rm -rf "${cann_home}.owned"

if run_phase marker-replacement; then
    echo "marker replacement: expected provisioning failure" >&2
    exit 1
fi
[[ "$(< "${cann_home}/sentinel")" == marker-replacement ]] || {
    echo "tree with a replaced ownership marker was deleted" >&2
    exit 1
}
rm -rf "${cann_home}"

if run_phase parent-replacement; then
    echo "parent replacement: expected provisioning failure" >&2
    exit 1
fi
[[ "$(< "${ci_home}/toolchains/sentinel")" == parent-replacement &&
   "$(entry_metadata "${ci_home}/toolchains" | awk -F: '{print $3}')" == 711 ]] || {
    echo "replacement toolchains parent was mutated" >&2
    exit 1
}
if ! grep -F 'refusing to restore replaced CANN parent directories' \
    "${fixture}/parent-replacement.log" >/dev/null; then
    echo "replacement toolchains parent was not explicitly refused" >&2
    exit 1
fi
rm -rf "${ci_home}/toolchains" "${ci_home}/toolchains.owned"
mkdir -p "${ci_home}/toolchains/cann"
chmod 0750 "${ci_home}/toolchains" "${ci_home}/toolchains/cann"

mkdir -p "${external_tree}/cann"
chmod 0711 "${external_tree}"
if run_phase parent-symlink; then
    echo "parent symlink: expected provisioning failure" >&2
    exit 1
fi
[[ -L "${ci_home}/toolchains" && "$(< "${external_tree}/sentinel")" == external &&
   "$(entry_metadata "${external_tree}" | awk -F: '{print $3}')" == 711 ]] || {
    echo "symlinked toolchains parent or its target was mutated" >&2
    exit 1
}
if ! grep -F 'refusing to restore replaced CANN parent directories' \
    "${fixture}/parent-symlink.log" >/dev/null; then
    echo "symlinked toolchains parent was not explicitly refused" >&2
    exit 1
fi
rm -f "${ci_home}/toolchains"
rm -rf "${ci_home}/toolchains.owned"
mkdir -p "${ci_home}/toolchains/cann"
chmod 0750 "${ci_home}/toolchains" "${ci_home}/toolchains/cann"

mv "${ci_home}/toolchains" "${ci_home}/toolchains.real"
ln -s "${external_tree}" "${ci_home}/toolchains"
if run_phase success; then
    echo "initial parent symlink was accepted" >&2
    exit 1
fi
[[ -L "${ci_home}/toolchains" && ! -e "${external_tree}/cann/9.1.0" &&
   "$(< "${external_tree}/sentinel")" == external &&
   "$(entry_metadata "${external_tree}" | awk -F: '{print $3}')" == 711 ]] || {
    echo "initial symlinked parent or its target was mutated" >&2
    exit 1
}
if ! grep -F 'CANN parent paths must be pre-created real directories' \
    "${fixture}/success.log" >/dev/null; then
    echo "initial symlinked parent was not explicitly refused" >&2
    exit 1
fi
rm -f "${ci_home}/toolchains"
mv "${ci_home}/toolchains.real" "${ci_home}/toolchains"

mkdir "${cann_home}"
printf preexisting > "${cann_home}/sentinel"
if run_phase success; then
    echo "untrusted pre-existing tree was accepted" >&2
    exit 1
fi
[[ "$(< "${cann_home}/sentinel")" == preexisting ]] || {
    echo "pre-existing tree was deleted" >&2
    exit 1
}
