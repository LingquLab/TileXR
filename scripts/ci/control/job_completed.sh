#!/bin/bash

set -euo pipefail

RUNNER_WORK_ROOT=/home/tilexr-ci/actions-runner/_work
WORKSPACE_PARENT=/home/tilexr-ci/actions-runner/_work/TileXR
WORKSPACE=/home/tilexr-ci/actions-runner/_work/TileXR/TileXR

case "${WORKSPACE}" in
    "${RUNNER_WORK_ROOT}"/*) ;;
    *)
        echo "ERROR: refusing unsafe runner workspace: ${WORKSPACE}" >&2
        exit 1
        ;;
esac
if [[ "${WORKSPACE}" == "${RUNNER_WORK_ROOT}" || "${WORKSPACE}" == "/" ]]; then
    echo "ERROR: refusing unsafe runner workspace: ${WORKSPACE}" >&2
    exit 1
fi
if [[ -L "${RUNNER_WORK_ROOT}" || -L "${WORKSPACE_PARENT}" || -L "${WORKSPACE}" ]]; then
    echo "ERROR: runner work root and workspace must be real directories" >&2
    exit 1
fi
if [[ ! -d "${RUNNER_WORK_ROOT}" ]]; then
    echo "ERROR: runner work root and workspace must be real directories" >&2
    exit 1
fi

RUNNER_WORK_ROOT_REAL="$(cd "${RUNNER_WORK_ROOT}" && pwd -P)"
if [[ "${RUNNER_WORK_ROOT_REAL}" != "${RUNNER_WORK_ROOT}" ]]; then
    echo "ERROR: fixed runner paths contain a symbolic-link redirection" >&2
    exit 1
fi
if [[ ! -e "${WORKSPACE_PARENT}" ]]; then
    exit 0
fi
if [[ ! -d "${WORKSPACE_PARENT}" ]]; then
    echo "ERROR: runner workspace parent must be a real directory" >&2
    exit 1
fi
WORKSPACE_PARENT_REAL="$(cd "${WORKSPACE_PARENT}" && pwd -P)"
if [[ "${WORKSPACE_PARENT_REAL}" != "${WORKSPACE_PARENT}" ]]; then
    echo "ERROR: fixed runner paths contain a symbolic-link redirection" >&2
    exit 1
fi
if [[ ! -e "${WORKSPACE}" ]]; then
    exit 0
fi
if [[ ! -d "${WORKSPACE}" ]]; then
    echo "ERROR: runner workspace must be a real directory" >&2
    exit 1
fi
WORKSPACE_REAL="$(cd "${WORKSPACE}" && pwd -P)"
if [[ "${WORKSPACE_REAL}" != "${WORKSPACE}" ]]; then
    echo "ERROR: fixed runner paths contain a symbolic-link redirection" >&2
    exit 1
fi
case "${WORKSPACE_REAL}" in
    "${RUNNER_WORK_ROOT_REAL}"/*) ;;
    *)
        echo "ERROR: resolved workspace escaped runner work root: ${WORKSPACE_REAL}" >&2
        exit 1
        ;;
esac
if [[ "${WORKSPACE_REAL}" == "${RUNNER_WORK_ROOT_REAL}" || "${WORKSPACE_REAL}" == "/" ]]; then
    echo "ERROR: refusing resolved unsafe workspace: ${WORKSPACE_REAL}" >&2
    exit 1
fi

# The hook runs after Actions post-job steps and removes only checkout children.
/usr/bin/find -P "${WORKSPACE_REAL}" -mindepth 1 -maxdepth 1 \
    -exec /usr/bin/rm -rf -- {} +
