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

echo "Deploying TileXR EP dispatch verification"
echo "  remote: ${REMOTE}"
echo "  remote repo: ${REMOTE_REPO}"
echo "  branch: ${branch}"
echo "  commit: ${commit}"

ssh "${REMOTE}" "mkdir -p $(printf '%q' "${REMOTE_BASE}") $(printf '%q' "${REMOTE_REPO}")"

rsync -a --delete \
  --exclude='.worktrees' \
  --exclude='build' \
  --exclude='build_*' \
  --exclude='install' \
  --exclude='tests/ep/build' \
  --exclude='tests/ep/install' \
  --exclude='tests/ep/logs' \
  "${TILEXR_ROOT}/" "${REMOTE}:${REMOTE_REPO}/"

remote_script=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${REMOTE_REPO}")
{
  echo "Remote branch source: ${branch}"
  echo "Remote commit source: ${commit}"
  git submodule update --init --recursive
  source scripts/common_env.sh
  bash tests/ep/build.sh full
  bash tests/ep/demo/run_tilexr_ep_dispatch_demo.sh 2
} 2>&1 | tee $(printf '%q' "${REMOTE_LOG}")
EOF
)

ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_script}")"

echo "Remote verification log: ${REMOTE}:${REMOTE_LOG}"
