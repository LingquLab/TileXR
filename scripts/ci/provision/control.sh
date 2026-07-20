#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${script_dir}/common.sh"
parse_args "$@"
require_root

control_source="${repo_root}/scripts/ci/control"
control_parent="${CI_HOME}/control"
control_current="${control_parent}/current"
stage="${control_parent}/.v1.new"
current_stage="${control_parent}/.current.new"

if [[ "$(< "${control_source}/VERSION")" != v1 ]]; then
    echo "ERROR: repository control package must have version v1" >&2
    exit 1
fi

run install -d -o root -g "${CI_GROUP}" -m 0750 "${control_parent}"
run rm -rf "${stage}"
run install -d -o root -g "${CI_GROUP}" -m 0750 "${stage}"

if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'git -C %q archive --format=tar HEAD scripts/ci/control | tar -xf - -C %q --strip-components=3\n' \
        "${repo_root}" "${stage}"
else
    git -C "${repo_root}" archive --format=tar HEAD scripts/ci/control |
        tar -xf - -C "${stage}" --strip-components=3
fi
run chown -R "root:${CI_GROUP}" "${stage}"
run chmod -R u+rwX,g+rX,o-rwx,go-w "${stage}"

if [[ "${DRY_RUN}" == 1 ]]; then
    run mv -T "${stage}" "${CONTROL_HOME}"
elif [[ -d "${CONTROL_HOME}" && ! -L "${CONTROL_HOME}" ]]; then
    if ! diff -qr "${stage}" "${CONTROL_HOME}" >/dev/null; then
        echo "ERROR: installed v1 control differs; publish a new control version" >&2
        exit 1
    fi
    run rm -rf "${stage}"
    run chown -R "root:${CI_GROUP}" "${CONTROL_HOME}"
    run chmod -R u+rwX,g+rX,o-rwx,go-w "${CONTROL_HOME}"
else
    [[ ! -e "${CONTROL_HOME}" && ! -L "${CONTROL_HOME}" ]] || {
        echo "ERROR: unsafe control destination: ${CONTROL_HOME}" >&2
        exit 1
    }
    run mv -T "${stage}" "${CONTROL_HOME}"
fi

run rm -f "${current_stage}"
run ln -s "${CONTROL_HOME}" "${current_stage}"
run mv -Tf "${current_stage}" "${control_current}"
run test -x "${control_current}/job_completed.sh"
