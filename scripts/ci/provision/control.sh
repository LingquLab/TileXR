#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
git_dir="${repo_root}/.git"
source "${script_dir}/common.sh"

STAGE_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            [[ "${DRY_RUN}" == 0 ]] || {
                echo "Usage: $0 [--dry-run] [--stage-only]" >&2
                exit 2
            }
            DRY_RUN=1
            ;;
        --stage-only)
            [[ "${STAGE_ONLY}" == 0 ]] || {
                echo "Usage: $0 [--dry-run] [--stage-only]" >&2
                exit 2
            }
            STAGE_ONLY=1
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--stage-only]" >&2
            exit 2
            ;;
    esac
done
require_root

control_source="${repo_root}/scripts/ci/control"
control_parent="${CI_HOME}/control"
control_current="${control_parent}/current"
stage="${control_parent}/.${CONTROL_VERSION}.new"
current_stage="${control_parent}/.current.new"

validate_control_package() {
    local package="$1"
    local script

    run test -d "${package}"
    run grep -Fx "${CONTROL_VERSION}" "${package}/VERSION"
    for script in build_blue.sh job_completed.sh run_hardware.sh; do
        run test -x "${package}/${script}"
        run bash -n "${package}/${script}"
    done
    for script in gate.py npu_state.py; do
        run test -r "${package}/${script}"
    done
    run python3 -c \
        'import pathlib, sys; [compile(pathlib.Path(p).read_text(encoding="utf-8"), p, "exec") for p in sys.argv[1:]]' \
        "${package}/gate.py" "${package}/npu_state.py"
}

if ! [[ -d "${git_dir}" && ! -L "${git_dir}" ]]; then
    echo "ERROR: provisioning checkout must have a real .git directory" >&2
    exit 1
fi

if [[ "$(< "${control_source}/VERSION")" != "${CONTROL_VERSION}" ]]; then
    echo "ERROR: repository control package must have version ${CONTROL_VERSION}" >&2
    exit 1
fi

run install -d -o root -g "${CI_GROUP}" -m 0750 "${control_parent}"
run rm -rf "${stage}"
run install -d -o root -g "${CI_GROUP}" -m 0750 "${stage}"

if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'git --git-dir=%q archive --format=tar HEAD scripts/ci/control | tar -xf - -C %q --strip-components=3\n' \
        "${git_dir}" "${stage}"
else
    git --git-dir="${git_dir}" archive \
        --format=tar HEAD scripts/ci/control |
        tar -xf - -C "${stage}" --strip-components=3
fi
run chown -R "root:${CI_GROUP}" "${stage}"
run chmod -R u+rwX,g+rX,o-rwx,go-w "${stage}"

if [[ "${DRY_RUN}" == 1 ]]; then
    run mv -T "${stage}" "${CONTROL_HOME}"
elif [[ -d "${CONTROL_HOME}" && ! -L "${CONTROL_HOME}" ]]; then
    if ! diff -qr "${stage}" "${CONTROL_HOME}" >/dev/null; then
        echo "ERROR: installed ${CONTROL_VERSION} control differs; publish a new control version" >&2
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

validate_control_package "${CONTROL_HOME}"
if [[ "${STAGE_ONLY}" == 1 ]]; then
    echo "Staged sealed controller ${CONTROL_VERSION}; current was not changed"
    exit 0
fi

run rm -f "${current_stage}"
run ln -s "${CONTROL_HOME}" "${current_stage}"
run mv -Tf "${current_stage}" "${control_current}"
run test -x "${control_current}/job_completed.sh"
