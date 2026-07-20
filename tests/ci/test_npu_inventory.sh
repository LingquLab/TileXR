#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture_dir="${repo_root}/tests/ci/fixtures"
source "${repo_root}/scripts/ci/provision/common.sh"

live_info="$(< "${fixture_dir}/npu_smi_info_blue_25_5.txt")"
if ! npu_smi_info_has_expected_devices "${live_info}"; then
    echo "the captured blue npu-smi table was rejected" >&2
    exit 1
fi

if npu_smi_info_has_expected_devices \
    "$(< "${fixture_dir}/npu_smi_product_unsupported.txt")"; then
    echo "the unsupported product-query response was accepted as inventory" >&2
    exit 1
fi
if npu_smi_info_has_expected_devices "${live_info/0     910B3/0     910B4}"; then
    echo "an unexpected product name was accepted" >&2
    exit 1
fi
if npu_smi_info_has_expected_devices "${live_info/OK            | 92.5/Warning       | 92.5}"; then
    echo "an unhealthy device was accepted" >&2
    exit 1
fi
if npu_smi_info_has_expected_devices "${live_info/| 7     910B3/| 0     910B3}"; then
    echo "duplicate and missing device IDs were accepted" >&2
    exit 1
fi

for provision_script in cann verify; do
    script="${repo_root}/scripts/ci/provision/${provision_script}.sh"
    if grep -E -- 'npu-smi info -t (product|health)' "${script}" >/dev/null; then
        echo "${provision_script}.sh still uses per-device inventory queries" >&2
        exit 1
    fi
    if ! grep -F -- 'npu_smi_info_has_expected_devices' "${script}" >/dev/null; then
        echo "${provision_script}.sh does not use the shared inventory parser" >&2
        exit 1
    fi
done
