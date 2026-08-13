#!/usr/bin/env bash
set -euo pipefail

runtime_link=/home/h00580772/tilexr_selfcopy_b150/cann_runtime
cabinet9=(
    141.61.55.118 141.61.55.114 141.61.55.110 141.61.55.106
    141.61.55.78 141.61.55.74 141.61.55.70 141.61.55.66
)
cabinet4=(
    141.61.52.116 141.61.52.120 141.61.52.128 141.61.52.124
    141.61.52.156 141.61.52.160 141.61.52.164 141.61.52.167
)

prepare_group() {
    local target=$1
    shift
    local host
    for host in "$@"; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "root@${host}" \
            "test -d '${target}/aarch64-linux' && mkdir -p '$(dirname "${runtime_link}")' && ln -sfn '${target}' '${runtime_link}' && test \"\$(readlink '${runtime_link}')\" = '${target}'"
        printf '%s: %s -> %s\n' "${host}" "${runtime_link}" "${target}"
    done
}

prepare_group /home/pkg/910_B150/cann-9.1.0 "${cabinet9[@]}"
prepare_group /home/pkg/b131/cann-9.1.0 "${cabinet4[@]}"
