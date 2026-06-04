#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/ascend-transformer-boost"
REPO_URL="${ATB_REPO_URL:-https://gitcode.com/cann/ascend-transformer-boost.git}"
BRANCH="${ATB_BRANCH:-master}"

if [ "${1:-}" = "--dry-run" ]; then
    echo "repo: ${REPO_URL}"
    echo "branch: ${BRANCH}"
    echo "target: ${TARGET_DIR}"
    exit 0
fi

if [ -d "${TARGET_DIR}/.git" ]; then
    git -C "${TARGET_DIR}" fetch origin "${BRANCH}"
    git -C "${TARGET_DIR}" checkout "${BRANCH}"
    git -C "${TARGET_DIR}" pull --ff-only origin "${BRANCH}"
else
    rm -rf "${TARGET_DIR}"
    git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${TARGET_DIR}"
fi

echo "ascend-transformer-boost reference source is ready at ${TARGET_DIR}"
