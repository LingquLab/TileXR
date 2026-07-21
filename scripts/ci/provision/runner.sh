#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
parse_args "$@"
require_root

runner_url=https://github.com/LingquLab
runner_group=TileXR-NPU
runner_name=blue-tilexr-npu8
runner_labels=tilexr,ascend910b,npu8
runner_work=_work
install_work="${CI_HOME}/install-work"
hook="${CI_HOME}/control/current/job_completed.sh"
env_entry="ACTIONS_RUNNER_HOOK_JOB_COMPLETED=${hook}"
preloaded_asset="${TILEXR_CI_RUNNER_ASSET:-}"

IFS= read -r registration_token || true
if [[ -z "${registration_token:-}" ]]; then
    echo "ERROR: a short-lived registration token is required on standard input" >&2
    exit 2
fi
unset ACTIONS_RUNNER_INPUT_TOKEN
trap 'unset registration_token ACTIONS_RUNNER_INPUT_TOKEN' EXIT

if [[ "${DRY_RUN}" != 1 && ( -L "${RUNNER_HOME}" || ! -d "${RUNNER_HOME}" ) ]]; then
    echo "ERROR: runner home must be a real directory: ${RUNNER_HOME}" >&2
    exit 1
fi
if [[ -n "${preloaded_asset}" &&
      ( "${preloaded_asset}" != /* || ! -f "${preloaded_asset}" ||
        -L "${preloaded_asset}" ) ]]; then
    echo "ERROR: TILEXR_CI_RUNNER_ASSET must be an absolute regular file" >&2
    exit 1
fi

runner_name_matches() {
    python3 - "$1" "${runner_name}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    runner = json.load(stream)
raise SystemExit(0 if runner.get("agentName") == sys.argv[2] else 1)
PY
}

service_name=""
if [[ "${DRY_RUN}" != 1 && -f "${RUNNER_HOME}/.service" ]]; then
    service_name="$(< "${RUNNER_HOME}/.service")"
    if [[ ! "${service_name}" =~ ^actions\.runner\.[A-Za-z0-9_.-]+\.service$ ]]; then
        echo "ERROR: runner service name is invalid" >&2
        exit 1
    fi
    if systemctl is-active --quiet -- "${service_name}"; then
        if ! runner_service_matches "${service_name}"; then
            echo "ERROR: active runner service has an unexpected user or executable" >&2
            exit 1
        fi
        if [[ "$(stat -c '%U:%G' "${RUNNER_HOME}")" != "root:${CI_PRIMARY_GROUP}" ]] ||
            [[ ! -f "${RUNNER_HOME}/.env" ]] ||
            ! grep -Fx "${env_entry}" "${RUNNER_HOME}/.env" >/dev/null ||
            [[ "$(stat -c '%U:%G' "${RUNNER_HOME}/.env")" != "root:${CI_PRIMARY_GROUP}" ]] ||
            [[ "$(stat -c '%a' "${RUNNER_HOME}/.env")" != 440 ]]; then
            echo "ERROR: active runner has an unexpected job-completed hook" >&2
            exit 1
        fi
        if [[ ! -f "${RUNNER_HOME}/.runner" ]] ||
            ! runner_name_matches "${RUNNER_HOME}/.runner"
        then
            echo "ERROR: active runner has an unexpected registration name" >&2
            exit 1
        fi
        echo "Runner ${runner_name} is active; leaving the online registration unchanged."
        exit 0
    fi
fi

if [[ "${DRY_RUN}" == 1 ]]; then
    stage="${install_work}/runner.dry-run"
else
    install -d -o root -g "${CI_GROUP}" -m 0750 "${install_work}"
    stage="$(mktemp -d "${install_work}/runner.XXXXXX")"
    trap 'unset registration_token ACTIONS_RUNNER_INPUT_TOKEN; rm -rf -- "${stage}"' EXIT
fi
release_json="${stage}/release.json"

run install -d -o root -g "${CI_GROUP}" -m 0750 "${stage}"
run curl --fail --location --silent --show-error \
    https://api.github.com/repos/actions/runner/releases/latest -o "${release_json}"

if [[ "${DRY_RUN}" == 1 ]]; then
    asset_name=actions-runner-linux-arm64-LATEST.tar.gz
    asset_url=https://github.com/actions/runner/releases/download/LATEST/actions-runner-linux-arm64-LATEST.tar.gz
    asset_digest=sha256:RELEASE-ASSET-DIGEST
else
    mapfile -t release_asset < <(python3 - "${release_json}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    release = json.load(stream)
if release.get("prerelease") or release.get("draft"):
    raise SystemExit("latest Actions runner release is not stable")
assets = [
    asset for asset in release.get("assets", [])
    if re.fullmatch(r"actions-runner-linux-arm64-[0-9.]+[0-9]\.tar\.gz", asset.get("name", ""))
]
if len(assets) != 1:
    raise SystemExit("expected exactly one Linux ARM64 runner asset")
asset = assets[0]
digest = asset.get("digest", "")
if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
    raise SystemExit("runner release asset has no valid SHA-256 digest")
print(asset["name"])
print(asset["browser_download_url"])
print(digest)
PY
    )
    [[ "${#release_asset[@]}" -eq 3 ]] || {
        echo "ERROR: could not resolve the official Actions runner asset" >&2
        exit 1
    }
    asset_name="${release_asset[0]}"
    asset_url="${release_asset[1]}"
    asset_digest="${release_asset[2]}"
fi

asset_path="${stage}/${asset_name}"
if [[ -n "${preloaded_asset}" ]]; then
    run install -o root -g "${CI_GROUP}" -m 0640 \
        "${preloaded_asset}" "${asset_path}"
else
    run curl --fail --location --silent --show-error "${asset_url}" -o "${asset_path}"
fi
if [[ "${DRY_RUN}" == 1 ]]; then
    run sha256sum --check "${asset_path}.sha256"
else
    printf '%s  %s\n' "${asset_digest#sha256:}" "${asset_path}" |
        sha256sum --check -
fi

if [[ "${DRY_RUN}" != 1 && -x "${RUNNER_HOME}/svc.sh" ]]; then
    "${RUNNER_HOME}/svc.sh" stop || true
    "${RUNNER_HOME}/svc.sh" uninstall || true
fi
if [[ "${DRY_RUN}" != 1 && -f "${RUNNER_HOME}/.runner" ]]; then
    if ! runner_name_matches "${RUNNER_HOME}/.runner"; then
        echo "ERROR: offline runner has an unexpected registration name" >&2
        exit 1
    fi
fi

if [[ "${DRY_RUN}" != 1 ]]; then
    if [[ -n "${service_name}" ]] && systemctl is-active --quiet -- "${service_name}"; then
        echo "ERROR: runner service is still active before registration" >&2
        exit 1
    fi
    if pgrep -u "${CI_USER}" >/dev/null; then
        echo "ERROR: another ${CI_USER} process is running before registration" >&2
        exit 1
    fi
fi

run find "${RUNNER_HOME}" -mindepth 1 -maxdepth 1 \
    ! -name _work ! -name _diag -exec rm -rf -- '{}' +
run tar -xzf "${asset_path}" -C "${RUNNER_HOME}"
run chown -R "${CI_USER}:${CI_PRIMARY_GROUP}" "${RUNNER_HOME}"

if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'export ACTIONS_RUNNER_INPUT_TOKEN=%q\n' '<registration-token>'
    printf 'runuser -u %s -- %s --url %s --runnergroup %s --name %s --labels %s --work %s --unattended --replace --disableupdate\n' \
        "${CI_USER}" "${RUNNER_HOME}/config.sh" "${runner_url}" \
        "${runner_group}" "${runner_name}" "${runner_labels}" "${runner_work}"
    printf 'unset ACTIONS_RUNNER_INPUT_TOKEN\n'
else
    if run_with_runner_registration_token "${registration_token}" \
        runuser -u "${CI_USER}" -- "${RUNNER_HOME}/config.sh" \
        --url "${runner_url}" \
        --runnergroup "${runner_group}" \
        --name "${runner_name}" \
        --labels "${runner_labels}" \
        --work "${runner_work}" \
        --unattended --replace --disableupdate; then
        registration_status=0
    else
        registration_status=$?
    fi
    if [[ "${registration_status}" -ne 0 ]]; then
        echo "ERROR: runner registration failed" >&2
        exit "${registration_status}"
    fi
fi

if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'printf %%s\\n %q > %q\n' "${env_entry}" "${RUNNER_HOME}/.env"
else
    env_stage="$(mktemp "${install_work}/runner-env.XXXXXX")"
    printf '%s\n' "${env_entry}" > "${env_stage}"
    install -o root -g "${CI_PRIMARY_GROUP}" -m 0440 "${env_stage}" "${RUNNER_HOME}/.env"
    rm -f "${env_stage}"
fi
run chown "root:${CI_PRIMARY_GROUP}" "${RUNNER_HOME}"
run find "${RUNNER_HOME}" -mindepth 1 -maxdepth 1 \
    ! -name _work ! -name _diag \
    -exec chown -R "root:${CI_PRIMARY_GROUP}" '{}' +
run chown -R "${CI_USER}:${CI_PRIMARY_GROUP}" \
    "${RUNNER_HOME}/_work" "${RUNNER_HOME}/_diag"
seal_runner_modes
run install -d -o "${CI_USER}" -g "${CI_PRIMARY_GROUP}" -m 0750 \
    "${RUNNER_HOME}/_work" "${RUNNER_HOME}/_diag"
run "${RUNNER_HOME}/svc.sh" install "${CI_USER}"
if [[ "${DRY_RUN}" == 1 ]]; then
    service_name=actions.runner.LingquLab-TileXR.blue-tilexr-npu8.service
    run env LC_ALL=C LANG=C systemctl show \
        --property=User --property=ExecStart -- "${service_name}"
else
    [[ -f "${RUNNER_HOME}/.service" && ! -L "${RUNNER_HOME}/.service" ]] || {
        echo "ERROR: runner service identity file is missing or unsafe" >&2
        exit 1
    }
    service_name="$(< "${RUNNER_HOME}/.service")"
    if ! runner_service_matches "${service_name}"; then
        echo "ERROR: installed runner service has an unexpected user or executable" >&2
        exit 1
    fi
fi
run "${RUNNER_HOME}/svc.sh" start
