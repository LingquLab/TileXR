#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
real_common="${repo_root}/scripts/ci/provision/common.sh"
real_cann="${repo_root}/scripts/ci/provision/cann.sh"
temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT

current_user="$(id -un)"
current_group="$(id -gn)"
cann_home="${temp_dir}/tree"
toolkit_info="${cann_home}/cann/aarch64-linux/ascend_toolkit_install.info"
ops_info="${cann_home}/cann/aarch64-linux/ascend_ops_install.info"
compiler_dir="${cann_home}/cann/compiler/ccec_compiler/bin"

mkdir -p "$(dirname "${toolkit_info}")" "${compiler_dir}"
printf '%s\n' \
    package_name=Ascend-cann-toolkit \
    version=9.1.0 > "${toolkit_info}"
printf '%s\n' \
    package_name=Ascend-cann-910b-ops \
    version=9.1.0 > "${ops_info}"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "${compiler_dir}/bisheng-real"
chmod 0750 "${compiler_dir}/bisheng-real"
ln -s bisheng-real "${compiler_dir}/bisheng"
find "${cann_home}" -type d -exec chmod 0750 '{}' +
find "${cann_home}" -type f ! -name bisheng-real -exec chmod 0640 '{}' +

source "${real_common}"
CANN_HOME="${cann_home}"
CI_GROUP="${current_group}"
CANN_OWNER="${current_user}"

if ! cann_tree_is_trusted; then
    echo "sealed CANN tree with an internal 0777 symlink was rejected" >&2
    exit 1
fi

chmod 0700 "${compiler_dir}"
if cann_tree_is_trusted; then
    echo "CANN tree accepted a directory that the CI group cannot traverse" >&2
    exit 1
fi
chmod 0750 "${compiler_dir}"

chmod 0600 "${toolkit_info}"
if cann_tree_is_trusted; then
    echo "CANN tree accepted metadata that the CI group cannot read" >&2
    exit 1
fi
chmod 0640 "${toolkit_info}"

chmod 0700 "${compiler_dir}/bisheng-real"
if cann_tree_is_trusted; then
    echo "CANN tree accepted a compiler that the CI group cannot execute" >&2
    exit 1
fi
chmod 0750 "${compiler_dir}/bisheng-real"

printf outside > "${temp_dir}/outside"
ln -s "${temp_dir}/outside" "${cann_home}/cann/external-link"
if cann_tree_is_trusted; then
    echo "CANN tree accepted an external symlink target" >&2
    exit 1
fi
rm -f "${cann_home}/cann/external-link"

chmod 0660 "${toolkit_info}"
if cann_tree_is_trusted; then
    echo "CANN tree accepted a group-writable regular file" >&2
    exit 1
fi
chmod 0640 "${toolkit_info}"

rm -f "${compiler_dir}/bisheng"
ln -s "${temp_dir}/outside" "${compiler_dir}/bisheng"
chmod +x "${temp_dir}/outside"
if cann_tree_is_trusted; then
    echo "CANN tree accepted a compiler resolving outside the sealed root" >&2
    exit 1
fi
rm -f "${compiler_dir}/bisheng"
ln -s bisheng-real "${compiler_dir}/bisheng"

fixture="${temp_dir}/fixture"
mkdir -p "${fixture}/scripts/ci/provision" "${fixture}/mock-bin"
cp "${real_cann}" "${fixture}/scripts/ci/provision/cann.sh"
{
    printf 'source %q\n' "${real_common}"
    printf '%s\n' \
        'CANN_HOME="${TEST_CANN_HOME:?}"' \
        'CI_HOME="${TEST_CI_HOME:?}"' \
        'CI_GROUP="${TEST_CANN_GROUP:?}"' \
        'CANN_OWNER="${TEST_CANN_OWNER:?}"' \
        'require_root() { :; }'
} > "${fixture}/scripts/ci/provision/common.sh"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'printf called > "${TEST_CHOWN_MARKER}"' \
    'exit 95' > "${fixture}/mock-bin/chown"
chmod +x "${fixture}/mock-bin/chown"

run_existing_tree() {
    local expected_status="$1"
    local name="$2"
    local status
    rm -f "${temp_dir}/chown-called"
    set +e
    PATH="${fixture}/mock-bin:${PATH}" \
        TEST_CANN_HOME="${cann_home}" \
        TEST_CI_HOME="${temp_dir}/ci-home" \
        TEST_CANN_GROUP="${current_group}" \
        TEST_CANN_OWNER="${current_user}" \
        TEST_CHOWN_MARKER="${temp_dir}/chown-called" \
        bash "${fixture}/scripts/ci/provision/cann.sh" \
        > "${temp_dir}/${name}.log" 2>&1
    status=$?
    set -e
    if [[ "${expected_status}" == success ]]; then
        [[ "${status}" -eq 0 ]] || {
            echo "${name}: trusted pre-existing CANN tree was rejected" >&2
            cat "${temp_dir}/${name}.log" >&2
            exit 1
        }
    else
        [[ "${status}" -ne 0 ]] || {
            echo "${name}: untrusted pre-existing CANN tree was accepted" >&2
            exit 1
        }
    fi
    [[ ! -e "${temp_dir}/chown-called" ]] || {
        echo "${name}: pre-existing CANN tree was mutated before trust" >&2
        exit 1
    }
}

run_existing_tree success trusted

chmod 0660 "${toolkit_info}"
run_existing_tree failure unsealed
[[ "$(stat -c '%a' "${toolkit_info}" 2>/dev/null || stat -f '%Lp' "${toolkit_info}")" == 660 ]] || {
    echo "unsealed CANN fixture was mutated" >&2
    exit 1
}
chmod 0640 "${toolkit_info}"

printf '%s\n' package_name=Ascend-cann-toolkit version=9.0.0 > "${toolkit_info}"
run_existing_tree failure wrong-version
printf '%s\n' package_name=Ascend-cann-toolkit version=9.1.0 > "${toolkit_info}"
chmod 0640 "${toolkit_info}"

rm -f "${compiler_dir}/bisheng"
ln -s "${temp_dir}/outside" "${compiler_dir}/bisheng"
run_existing_tree failure spoofed-compiler
