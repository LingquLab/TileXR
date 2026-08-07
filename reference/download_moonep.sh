#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${MOONEP_REPO_URL:-https://github.com/MoonshotAI/MoonEP.git}"
BRANCH="${MOONEP_BRANCH:-master}"
TARGET_DIR="${SCRIPT_DIR}/MoonEP"

usage() {
    cat <<'EOF'
Usage: bash reference/download_moonep.sh [--dry-run] [--force]

Download or update MoonEP reference source in reference/MoonEP.

Options:
  --dry-run   Print the repository, branch, and target without cloning.
  --force     Replace an existing non-git target directory before cloning.
  -h, --help  Show this help text.

Environment:
  MOONEP_REPO_URL  Repository URL, default: https://github.com/MoonshotAI/MoonEP.git
  MOONEP_BRANCH    Branch, default: master
EOF
}

dry_run=0
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --force) force=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ${dry_run} -eq 1 ]]; then
    echo "repo: ${REPO_URL}"
    echo "branch: ${BRANCH}"
    echo "target: ${TARGET_DIR}"
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    echo "git is required to download MoonEP reference source" >&2
    exit 1
fi

if [[ -d "${TARGET_DIR}/.git" ]]; then
    git -C "${TARGET_DIR}" fetch origin "${BRANCH}"
    git -C "${TARGET_DIR}" checkout "${BRANCH}"
    git -C "${TARGET_DIR}" pull --ff-only origin "${BRANCH}"
elif [[ -e "${TARGET_DIR}" ]]; then
    if [[ ${force} -ne 1 ]]; then
        echo "target exists but is not a git checkout: ${TARGET_DIR}" >&2
        echo "rerun with --force to replace it" >&2
        exit 1
    fi
    rm -rf "${TARGET_DIR}"
    git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${TARGET_DIR}"
else
    git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${TARGET_DIR}"
fi

echo "MoonEP reference source is ready at ${TARGET_DIR}"
