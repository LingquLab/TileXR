#!/usr/bin/env bash
set -euo pipefail

HOSTFILE=""
INSTALL_DIR=""
SSH_USER="$(id -un)"

usage() {
    cat <<'EOF'
Usage: bash tools/moonep/sync_combine_v2_perf_runtime.sh --hostfile PATH --install-dir PATH [options]

Options:
  --hostfile PATH      Rank hostfile with host:slots entries
  --install-dir PATH   Runtime directory to mirror at the same path on each host
  --ssh-user USER      SSH user (default: current user)
  --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostfile) HOSTFILE="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        --ssh-user) SSH_USER="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${HOSTFILE}" || -z "${INSTALL_DIR}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -f "${HOSTFILE}" || ! -x "${INSTALL_DIR}/bin/tilexr_moonep_combine_v2_perf" ]]; then
    echo "hostfile or staged benchmark is missing" >&2
    exit 1
fi
if [[ "${INSTALL_DIR}" != /* || "${INSTALL_DIR}" == *"'"* ]]; then
    echo "--install-dir must be an absolute path without single quotes" >&2
    exit 2
fi

mapfile -t host_entries < <(awk '
    /^[[:space:]]*($|#)/ { next }
    { gsub(/[[:space:]]/, "", $0); print $0 }
' "${HOSTFILE}")
if [[ ${#host_entries[@]} -eq 0 ]]; then
    echo "hostfile has no hosts: ${HOSTFILE}" >&2
    exit 1
fi

manifest="$(mktemp)"
trap 'rm -f "${manifest}"' EXIT
(cd "${INSTALL_DIR}" && find bin lib64 -type f -print0 | sort -z | \
    xargs -0 sha256sum) >"${manifest}"
local_ips=" $(hostname -I 2>/dev/null || true) 127.0.0.1 localhost "

for entry in "${host_entries[@]}"; do
    host="${entry%%:*}"
    if [[ "${local_ips}" == *" ${host} "* || "${host}" == "$(hostname)" ]]; then
        echo "${host}: local runtime retained"
    else
        target="${SSH_USER}@${host}"
        ssh -o BatchMode=yes "${target}" "mkdir -p '${INSTALL_DIR}'"
        rsync -a --delete -e "ssh -o BatchMode=yes" \
            "${INSTALL_DIR}/" "${target}:${INSTALL_DIR}/"
        echo "${host}: runtime synchronized"
    fi
done

for entry in "${host_entries[@]}"; do
    host="${entry%%:*}"
    if [[ "${local_ips}" == *" ${host} "* || "${host}" == "$(hostname)" ]]; then
        (cd "${INSTALL_DIR}" && sha256sum -c "${manifest}") >/dev/null
    else
        ssh -o BatchMode=yes "${SSH_USER}@${host}" \
            "cd '${INSTALL_DIR}' && sha256sum -c -" <"${manifest}" >/dev/null
    fi
    echo "${host}: SHA256 verified"
done
