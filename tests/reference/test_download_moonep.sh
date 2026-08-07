#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${ROOT_DIR}/reference/download_moonep.sh"
CANN_SCRIPT="${ROOT_DIR}/reference/download_cann_repos.sh"
INVALID_OUTPUT="$(mktemp)"
trap 'rm -f "${INVALID_OUTPUT}"' EXIT

if [[ ! -x "${SCRIPT}" ]]; then
    echo "script is not executable: ${SCRIPT}" >&2
    exit 1
fi

dry_run_output="$(bash "${SCRIPT}" --dry-run)"
if [[ "${dry_run_output}" != *"repo: https://github.com/MoonshotAI/MoonEP.git"* ||
      "${dry_run_output}" != *"branch: master"* ||
      "${dry_run_output}" != *"target: ${ROOT_DIR}/reference/MoonEP"* ]]; then
    echo "unexpected MoonEP dry-run output:" >&2
    echo "${dry_run_output}" >&2
    exit 1
fi

override_output="$(MOONEP_REPO_URL=https://example.invalid/custom.git \
    MOONEP_BRANCH=custom bash "${SCRIPT}" --dry-run)"
if [[ "${override_output}" != *"repo: https://example.invalid/custom.git"* ||
      "${override_output}" != *"branch: custom"* ]]; then
    echo "MoonEP environment overrides were not applied:" >&2
    echo "${override_output}" >&2
    exit 1
fi

if bash "${SCRIPT}" --unknown >"${INVALID_OUTPUT}" 2>&1; then
    echo "unknown MoonEP option unexpectedly succeeded" >&2
    exit 1
fi

if bash "${CANN_SCRIPT}" --list | grep -qx "MoonEP"; then
    echo "MoonEP must not be part of the CANN reference downloader" >&2
    exit 1
fi
