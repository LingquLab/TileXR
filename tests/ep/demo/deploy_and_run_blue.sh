#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TILEXR_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

REMOTE=${TILEXR_EP_REMOTE:-blue}
REMOTE_BASE=${TILEXR_EP_REMOTE_BASE:-/home/d00520898/tilexr_ep_dispatch_verify}
REMOTE_REPO=${REMOTE_BASE}/TileXR
REMOTE_LOG=${REMOTE_BASE}/deploy_$(date +%Y%m%d_%H%M%S).log

branch=$(git -C "${TILEXR_ROOT}" rev-parse --abbrev-ref HEAD)
commit=$(git -C "${TILEXR_ROOT}" rev-parse HEAD)
STAGING_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tilexr_ep_deploy.XXXXXX")
trap 'rm -rf "${STAGING_DIR}"' EXIT
STAGING_REPO="${STAGING_DIR}/TileXR"

echo "Deploying TileXR EP dispatch verification"
echo "  remote: ${REMOTE}"
echo "  remote repo: ${REMOTE_REPO}"
echo "  branch: ${branch}"
echo "  commit: ${commit}"

git clone --no-hardlinks --no-checkout "${TILEXR_ROOT}" "${STAGING_REPO}"
git -C "${STAGING_REPO}" checkout --detach "${commit}"

remote_prepare=$(cat <<EOF
set -euo pipefail
remote_base=$(printf '%q' "${REMOTE_BASE}")
remote_repo=$(printf '%q' "${REMOTE_REPO}")
case "\${remote_repo}" in
  "\${remote_base}"/TileXR)
    rm -rf -- "\${remote_repo}"
    mkdir -p -- "\${remote_repo}"
    ;;
  *)
    echo "Refusing to clean unexpected remote repo: \${remote_repo}" >&2
    exit 2
    ;;
esac
EOF
)

ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_prepare}")"

rsync -a --delete \
  --exclude='.worktrees' \
  --exclude='build' \
  --exclude='build_*' \
  --exclude='install' \
  --exclude='tests/ep/build' \
  --exclude='tests/ep/install' \
  --exclude='tests/ep/logs' \
  "${STAGING_REPO}/" "${REMOTE}:${REMOTE_REPO}/"

remote_script=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${REMOTE_REPO}")
{
  echo "Remote branch source: ${branch}"
  echo "Remote commit source: ${commit}"
  git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init 3rdparty/hcomm 3rdparty/ops-transformer 3rdparty/spdlog
  : "\${ASCEND_HOME_PATH:=}"
  : "\${LD_LIBRARY_PATH:=}"
  source scripts/common_env.sh
  bash tests/ep/build.sh full
  bash tests/ep/demo/run_tilexr_ep_dispatch_demo.sh 2
} 2>&1 | tee $(printf '%q' "${REMOTE_LOG}")
EOF
)

ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_script}")"

echo "Remote verification log: ${REMOTE}:${REMOTE_LOG}"
