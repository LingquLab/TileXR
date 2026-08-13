#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=""
HOSTFILE=""
SOURCE_DIR=""
BUILD_DIR=""
INSTALL_DIR=""
CANN_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SSH_USER="$(id -un)"
BS=""
BS_LIST=""
WARMUP=20
ITERATIONS=80
EXPERTS=64
COMM_DOMAIN=141
COMM_PORT=10067
WAIT_SECONDS=120
RETRY_SECONDS=15
RANK_TIMEOUT=600
BUILD_JOBS="$(nproc)"
LOG_FILE=""
SKIP_BUILD=0
SKIP_RUNTIME_SYNC=0
SKIP_ITERATION_BARRIERS=0
PROFILE=0
SKIP_NPU_PREFLIGHT=0
ALLOW_SELF_ONLY_FAILURE=0

usage() {
    cat <<'EOF'
Usage: bash tools/moonep/run_combine_v2_perf_cluster.sh --remote-root PATH [options]

Options:
  --remote-root PATH        Remote task root containing source/build/install/logs
  --source-dir PATH         Source directory (default: REMOTE_ROOT/source)
  --build-dir PATH          Build directory (default: REMOTE_ROOT/build)
  --install-dir PATH        Runtime directory (default: REMOTE_ROOT/install)
  --hostfile PATH           Rank hostfile (default: REMOTE_ROOT/hostfile)
  --cann-path PATH          CANN root
  --ssh-user USER           SSH user for worker launch/sync (default: current user)
  --bs N                    Run one batch size (default: 128)
  --bs-list N[,N...]        Run multiple batch sizes
  --warmup N                Warmup launches per BS (default: 20)
  --iterations N            Timed launches per BS (default: 80)
  --experts N               Total expert count (default: 64)
  --comm-domain N           Shared-QP domain (default: 141)
  --comm-port N             Bootstrap TCP port on first host (default: 10067)
  --wait-seconds N          Maximum NPU wait (default: 120)
  --retry-seconds N         NPU retry interval (default: 15)
  --rank-timeout N          Per-rank timeout (default: 600)
  --build-jobs N            Parallel build jobs (default: nproc)
  --log-file PATH           Controller log path
  --skip-build              Reuse an existing install directory
  --skip-runtime-sync       Do not rsync install to worker hosts
  --skip-iteration-barriers Forward to benchmark launcher
  --profile                 Capture per-AIV kernel cycle timestamps
  --skip-npu-preflight      Forward after manual NPU validation
  --allow-self-only-failure Forward to benchmark launcher
  --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
        --source-dir) SOURCE_DIR="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        --hostfile) HOSTFILE="$2"; shift 2 ;;
        --cann-path) CANN_PATH="$2"; shift 2 ;;
        --ssh-user) SSH_USER="$2"; shift 2 ;;
        --bs) BS="$2"; shift 2 ;;
        --bs-list) BS_LIST="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --experts) EXPERTS="$2"; shift 2 ;;
        --comm-domain) COMM_DOMAIN="$2"; shift 2 ;;
        --comm-port) COMM_PORT="$2"; shift 2 ;;
        --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
        --retry-seconds) RETRY_SECONDS="$2"; shift 2 ;;
        --rank-timeout) RANK_TIMEOUT="$2"; shift 2 ;;
        --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-runtime-sync) SKIP_RUNTIME_SYNC=1; shift ;;
        --skip-iteration-barriers) SKIP_ITERATION_BARRIERS=1; shift ;;
        --profile) PROFILE=1; shift ;;
        --skip-npu-preflight) SKIP_NPU_PREFLIGHT=1; shift ;;
        --allow-self-only-failure) ALLOW_SELF_ONLY_FAILURE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${REMOTE_ROOT}" ]]; then
    usage >&2
    exit 2
fi
SOURCE_DIR="${SOURCE_DIR:-${REMOTE_ROOT}/source}"
BUILD_DIR="${BUILD_DIR:-${REMOTE_ROOT}/build}"
INSTALL_DIR="${INSTALL_DIR:-${REMOTE_ROOT}/install}"
HOSTFILE="${HOSTFILE:-${REMOTE_ROOT}/hostfile}"
if [[ -z "${LOG_FILE}" ]]; then
    LOG_FILE="${REMOTE_ROOT}/logs/combine_v2_cluster_$(date +%Y%m%d_%H%M%S).log"
fi

for path in "${REMOTE_ROOT}" "${SOURCE_DIR}" "${BUILD_DIR}" "${INSTALL_DIR}" \
    "${HOSTFILE}" "${CANN_PATH}" "${LOG_FILE}"; do
    if [[ "${path}" != /* || "${path}" == *"'"* ]]; then
        echo "paths must be absolute and cannot contain single quotes: ${path}" >&2
        exit 2
    fi
done
if [[ -n "${BS}" && -n "${BS_LIST}" ]]; then
    echo "--bs and --bs-list are mutually exclusive" >&2
    exit 2
fi
if [[ -z "${BS}" && -z "${BS_LIST}" ]]; then
    BS=128
fi
for value in "${WARMUP}" "${ITERATIONS}" "${EXPERTS}" "${COMM_DOMAIN}" \
    "${COMM_PORT}" "${WAIT_SECONDS}" "${RETRY_SECONDS}" "${RANK_TIMEOUT}" \
    "${BUILD_JOBS}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "numeric arguments must be non-negative integers" >&2
        exit 2
    fi
done
if (( ITERATIONS == 0 || EXPERTS == 0 || COMM_DOMAIN == 0 ||
      COMM_PORT == 0 || COMM_PORT > 65535 || RETRY_SECONDS == 0 ||
      RANK_TIMEOUT == 0 || BUILD_JOBS == 0 )); then
    echo "iterations, experts, domain, port, retry, timeout, and jobs must be positive" >&2
    exit 2
fi
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "hostfile is missing: ${HOSTFILE}" >&2
    exit 1
fi

first_host="$(awk '
    /^[[:space:]]*($|#)/ { next }
    { gsub(/[[:space:]]/, "", $0); split($0, item, ":"); print item[1]; exit }
' "${HOSTFILE}")"
if [[ -z "${first_host}" ]]; then
    echo "hostfile has no hosts: ${HOSTFILE}" >&2
    exit 1
fi

build_script="${SOURCE_DIR}/tools/moonep/build_combine_v2_perf.sh"
sync_script="${SOURCE_DIR}/tools/moonep/sync_combine_v2_perf_runtime.sh"
run_script="${SOURCE_DIR}/tools/moonep/run_combine_v2_perf_multihost.sh"
for script in "${build_script}" "${sync_script}" "${run_script}"; do
    if [[ ! -f "${script}" ]]; then
        echo "required script is missing: ${script}" >&2
        exit 1
    fi
done

if (( ! SKIP_BUILD )); then
    bash "${build_script}" \
        --source-dir "${SOURCE_DIR}" \
        --build-dir "${BUILD_DIR}" \
        --install-dir "${INSTALL_DIR}" \
        --cann-path "${CANN_PATH}" \
        --jobs "${BUILD_JOBS}"
fi

if (( ! SKIP_RUNTIME_SYNC )); then
    bash "${sync_script}" \
        --hostfile "${HOSTFILE}" \
        --install-dir "${INSTALL_DIR}" \
        --ssh-user "${SSH_USER}"
fi

run_args=(
    --hostfile "${HOSTFILE}"
    --install-dir "${INSTALL_DIR}"
    --cann-path "${CANN_PATH}"
    --ssh-user "${SSH_USER}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --experts "${EXPERTS}"
    --comm-domain "${COMM_DOMAIN}"
    --comm-id "${first_host}:${COMM_PORT}"
    --wait-seconds "${WAIT_SECONDS}"
    --retry-seconds "${RETRY_SECONDS}"
    --timeout "${RANK_TIMEOUT}"
    --log-file "${LOG_FILE}"
)
if [[ -n "${BS_LIST}" ]]; then
    run_args+=(--bs-list "${BS_LIST}")
else
    run_args+=(--bs "${BS}")
fi
if (( SKIP_ITERATION_BARRIERS )); then
    run_args+=(--skip-iteration-barriers)
fi
if (( PROFILE )); then
    run_args+=(--profile)
fi
if (( SKIP_NPU_PREFLIGHT )); then
    run_args+=(--skip-npu-preflight)
fi
if (( ALLOW_SELF_ONLY_FAILURE )); then
    run_args+=(--allow-self-only-failure)
fi

bash "${run_script}" "${run_args[@]}"
echo "Completed. Primary log: ${LOG_FILE}"
