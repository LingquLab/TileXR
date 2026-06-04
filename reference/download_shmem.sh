#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/shmem"
REPO_URL="${SHMEM_REPO_URL:-https://github.com/LingquLab/shmem.git}"
BRANCH="${SHMEM_BRANCH:-tilexr-udma-integration}"

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

echo "shmem reference source is ready at ${TARGET_DIR}"
