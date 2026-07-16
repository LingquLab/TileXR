#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${SCRIPT_DIR}"
CANN_BASE_URL="${CANN_GITCODE_BASE_URL:-https://gitcode.com/cann}"
DEFAULT_BRANCH="${CANN_REPO_BRANCH:-master}"

repos=(
    "asc-devkit"
    "runtime"
    "driver"
    "asc-tools"
    "oam-tools"
    "ascend-transformer-boost"
    "ops-transformer"
    "shmem"
)

dry_run=0
force=0
selected_repos=()

usage() {
    cat <<'EOF'
Usage: bash reference/download_cann_repos.sh [options] [repo ...]

Download CANN open-source reference repositories into reference/.

Repos:
  asc-devkit
  runtime
  driver
  asc-tools
  oam-tools
  ascend-transformer-boost
  ops-transformer
  shmem

Options:
  --dry-run   Print the repositories, branches, and targets without cloning.
  --force     Replace an existing non-git target directory before cloning.
  --list      List supported repository names.
  -h, --help  Show this help text.

Environment:
  CANN_GITCODE_BASE_URL       Base URL, default: https://gitcode.com/cann
  CANN_REPO_BRANCH            Default branch for all repos, default: master
  CANN_ASC_DEVKIT_BRANCH      Branch override for asc-devkit
  CANN_RUNTIME_BRANCH         Branch override for runtime
  CANN_DRIVER_BRANCH          Branch override for driver
  CANN_ASC_TOOLS_BRANCH       Branch override for asc-tools
  CANN_OAM_TOOLS_BRANCH       Branch override for oam-tools
  CANN_ASCEND_TRANSFORMER_BOOST_BRANCH
                              Branch override for ascend-transformer-boost
  CANN_OPS_TRANSFORMER_BRANCH Branch override for ops-transformer
  CANN_SHMEM_BRANCH           Branch override for shmem
  ATB_BRANCH                  Backward-compatible branch override for ascend-transformer-boost
  SHMEM_BRANCH                Branch override for shmem
EOF
}

contains_repo() {
    local candidate=$1
    local repo
    for repo in "${repos[@]}"; do
        if [[ "${repo}" == "${candidate}" ]]; then
            return 0
        fi
    done
    return 1
}

branch_for_repo() {
    local repo=$1
    local env_name

    case "${repo}" in
        ascend-transformer-boost)
            printf '%s\n' "${ATB_BRANCH:-${CANN_ASCEND_TRANSFORMER_BOOST_BRANCH:-${DEFAULT_BRANCH}}}"
            return
            ;;
        shmem)
            printf '%s\n' "${SHMEM_BRANCH:-${CANN_SHMEM_BRANCH:-${DEFAULT_BRANCH}}}"
            return
            ;;
    esac

    env_name="$(printf 'CANN_%s_BRANCH' "${repo}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
    printf '%s\n' "${!env_name:-${DEFAULT_BRANCH}}"
}

list_repos() {
    local repo
    for repo in "${repos[@]}"; do
        printf '%s\n' "${repo}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        --force)
            force=1
            ;;
        --list)
            list_repos
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            selected_repos+=("$1")
            ;;
    esac
    shift
done

while [[ $# -gt 0 ]]; do
    selected_repos+=("$1")
    shift
done

if [[ ${#selected_repos[@]} -eq 0 ]]; then
    selected_repos=("${repos[@]}")
fi

for repo in "${selected_repos[@]}"; do
    if ! contains_repo "${repo}"; then
        echo "unknown CANN reference repository: ${repo}" >&2
        echo "supported repositories:" >&2
        list_repos >&2
        exit 2
    fi
done

if [[ ${dry_run} -eq 0 ]]; then
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to download CANN reference repositories" >&2
        exit 1
    fi
    mkdir -p "${TARGET_ROOT}"
fi

for repo in "${selected_repos[@]}"; do
    branch="$(branch_for_repo "${repo}")"
    repo_url="${CANN_BASE_URL%/}/${repo}.git"
    target_dir="${TARGET_ROOT}/${repo}"

    if [[ ${dry_run} -eq 1 ]]; then
        echo "repo: ${repo_url}"
        echo "branch: ${branch}"
        echo "target: ${target_dir}"
        continue
    fi

    if [[ -d "${target_dir}/.git" ]]; then
        git -C "${target_dir}" fetch origin "${branch}"
        git -C "${target_dir}" checkout "${branch}"
        git -C "${target_dir}" pull --ff-only origin "${branch}"
    else
        if [[ -e "${target_dir}" ]]; then
            if [[ ${force} -ne 1 ]]; then
                echo "target exists but is not a git checkout: ${target_dir}" >&2
                echo "rerun with --force to replace it" >&2
                exit 1
            fi
            rm -rf "${target_dir}"
        fi
        git clone --branch "${branch}" --single-branch "${repo_url}" "${target_dir}"
    fi

    echo "${repo} reference source is ready at ${target_dir}"
done
