#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UDMA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TILEXR_ROOT="$(cd "${UDMA_DIR}/../.." && pwd)"
INSTALL_DIR="${UDMA_DIR}/install"

HOSTS=""
RANK_SIZE=2
COMM_ID=""
MPI_HOME="${MPI_HOME:-/usr/local/mpich}"
LAUNCHER="auto"
TIMEOUT_SECONDS=600
WARMUP="${TILEXR_PROFILE_PROBE_WARMUP:-2}"
ITERATIONS="${TILEXR_PROFILE_PROBE_ITERATIONS:-5}"
OUTPUT_DIR="${UDMA_DIR}/logs/tilexr_udma_profile_probe_$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<EOF
Usage: bash tests/udma/demo/run_tilexr_udma_profile_probe_mpi.sh \\
  --hosts host1:1,host2:1 --comm-id rank0_data_ip:port [options]

Options:
  --hosts <hosts>       Two hosts, one rank per host
  --comm-id <ip:port>   TileXR rendezvous address on rank 0
  --launcher <mode>     auto, mpi, or ssh (default: auto)
  --mpi-home <dir>      MPI installation (default: /usr/local/mpich)
  --warmup <N>          Warmup launches per size/QP/mode (default: 2)
  --iterations <N>      Timed launches per size/QP/mode (default: 5)
  --output-dir <dir>    Log and JSONL destination
  --timeout <seconds>   Complete-run timeout (default: 600)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts) HOSTS="$2"; shift 2 ;;
        --comm-id) COMM_ID="$2"; shift 2 ;;
        --mpi-home) MPI_HOME="$2"; shift 2 ;;
        --launcher) LAUNCHER="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${HOSTS}" || -z "${COMM_ID}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "${WARMUP}" =~ ^[0-9]+$ || ! "${ITERATIONS}" =~ ^[1-9][0-9]*$ ||
      ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "warmup must be non-negative; iterations and timeout must be positive" >&2
    exit 2
fi
if [[ ! "${LAUNCHER}" =~ ^(auto|mpi|ssh)$ ]]; then
    echo "launcher must be auto, mpi, or ssh" >&2
    exit 2
fi
if [[ ! "${COMM_ID}" =~ ^[0-9]+(\.[0-9]+){3}:[1-9][0-9]*$ ]]; then
    echo "comm-id must be an IPv4 address and port" >&2
    exit 2
fi

: "${ASCEND_HOME_PATH:=}"
: "${LD_LIBRARY_PATH:=}"
source "${TILEXR_ROOT}/scripts/common_env.sh"

bin="${TILEXR_PROFILE_PROBE_BIN:-${INSTALL_DIR}/bin/tilexr_udma_profile_probe}"
if [[ ! -x "${bin}" ]]; then
    echo "Missing profile probe binary: ${bin}" >&2
    echo "Build it with: cd ${UDMA_DIR} && bash build.sh" >&2
    exit 1
fi

mpi_bin="${MPI_HOME}/bin/mpirun"
if [[ ! -x "${mpi_bin}" ]]; then
    mpi_bin="$(command -v mpirun || true)"
fi
if [[ "${LAUNCHER}" == "auto" ]]; then
    if [[ -n "${mpi_bin}" && -x "${mpi_bin}" ]]; then
        LAUNCHER="mpi"
    else
        LAUNCHER="ssh"
    fi
fi
if [[ "${LAUNCHER}" == "mpi" && ( -z "${mpi_bin}" || ! -x "${mpi_bin}" ) ]]; then
    echo "mpirun not found; set --mpi-home" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
combined_log="${OUTPUT_DIR}/combined.log"
jsonl="${OUTPUT_DIR}/timings.jsonl"

export TILEXR_COMM_ID="${COMM_ID}"
export TILEXR_UDMA_QP_ROUTE_SPEC="port_count:6,port_count:6,port_count:2"
export TILEXR_PROFILE_PROBE_WARMUP="${WARMUP}"
export TILEXR_PROFILE_PROBE_ITERATIONS="${ITERATIONS}"
export TILEXR_PROFILE_PROBE_DEVICE_BASE=0
export TILEXR_ENABLE_IPC=0
export TILEXR_ENABLE_SDMA=0
export LD_LIBRARY_PATH="${TILEXR_ROOT}/install/lib64:${TILEXR_ROOT}/install/lib:${INSTALL_DIR}/lib64:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

if [[ "${LAUNCHER}" == "mpi" ]]; then
    set +e
    timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECONDS}" \
        "${mpi_bin}" -hosts "${HOSTS}" -n "${RANK_SIZE}" \
        -genv TILEXR_COMM_ID "${TILEXR_COMM_ID}" \
        -genv TILEXR_UDMA_QP_ROUTE_SPEC "${TILEXR_UDMA_QP_ROUTE_SPEC}" \
        -genv TILEXR_PROFILE_PROBE_WARMUP "${TILEXR_PROFILE_PROBE_WARMUP}" \
        -genv TILEXR_PROFILE_PROBE_ITERATIONS "${TILEXR_PROFILE_PROBE_ITERATIONS}" \
        -genv TILEXR_PROFILE_PROBE_DEVICE_BASE "${TILEXR_PROFILE_PROBE_DEVICE_BASE}" \
        -genv TILEXR_ENABLE_IPC "${TILEXR_ENABLE_IPC}" \
        -genv TILEXR_ENABLE_SDMA "${TILEXR_ENABLE_SDMA}" \
        -genv LD_LIBRARY_PATH "${LD_LIBRARY_PATH}" \
        "${bin}" 2>&1 | tee "${combined_log}"
    run_status=${PIPESTATUS[0]}
    set -e
else
    IFS=',' read -r host0_spec host1_spec extra_host <<< "${HOSTS}"
    if [[ -n "${extra_host:-}" || ! "${host0_spec:-}" =~ ^[A-Za-z0-9._-]+:1$ ||
          ! "${host1_spec:-}" =~ ^[A-Za-z0-9._-]+:1$ ]]; then
        echo "ssh launcher requires --hosts host1:1,host2:1" >&2
        exit 2
    fi
    host0="${host0_spec%:1}"
    host1="${host1_spec%:1}"
    for value in "${TILEXR_ROOT}" "${ASCEND_HOME_PATH}" "${LD_LIBRARY_PATH}" "${bin}"; do
        if [[ ! "${value}" =~ ^[-A-Za-z0-9_./:]+$ ]]; then
            echo "ssh launcher paths must not contain shell metacharacters: ${value}" >&2
            exit 2
        fi
    done

    run_ssh_rank() {
        local host="$1"
        local rank="$2"
        ssh -o BatchMode=yes "root@${host}" \
            "cd ${TILEXR_ROOT}; source ${ASCEND_HOME_PATH}/set_env.sh; \
             export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}; \
             export TILEXR_COMM_ID=${TILEXR_COMM_ID}; \
             export TILEXR_UDMA_QP_ROUTE_SPEC=${TILEXR_UDMA_QP_ROUTE_SPEC}; \
             export TILEXR_PROFILE_PROBE_WARMUP=${TILEXR_PROFILE_PROBE_WARMUP}; \
             export TILEXR_PROFILE_PROBE_ITERATIONS=${TILEXR_PROFILE_PROBE_ITERATIONS}; \
             export TILEXR_PROFILE_PROBE_DEVICE_BASE=${TILEXR_PROFILE_PROBE_DEVICE_BASE}; \
             export TILEXR_ENABLE_IPC=${TILEXR_ENABLE_IPC}; \
             export TILEXR_ENABLE_SDMA=${TILEXR_ENABLE_SDMA}; \
             export RANK=${rank}; export RANK_SIZE=${RANK_SIZE}; \
             exec timeout --signal=TERM --kill-after=30 ${TIMEOUT_SECONDS} ${bin}"
    }

    set +e
    run_ssh_rank "${host0}" 0 > "${OUTPUT_DIR}/rank0.log" 2>&1 &
    pid0=$!
    run_ssh_rank "${host1}" 1 > "${OUTPUT_DIR}/rank1.log" 2>&1 &
    pid1=$!
    wait "${pid0}"
    status0=$?
    wait "${pid1}"
    status1=$?
    cat "${OUTPUT_DIR}/rank0.log" "${OUTPUT_DIR}/rank1.log" | tee "${combined_log}"
    run_status=$((status0 != 0 || status1 != 0))
    set -e
fi

sed -n 's/^TILEXR_UDMA_PROFILE_PROBE_JSON //p' "${combined_log}" > "${jsonl}"
success_count=$(grep -c 'TileXR UDMA profile probe success' "${combined_log}" || true)
if [[ "${run_status}" -ne 0 || "${success_count}" -ne "${RANK_SIZE}" ]]; then
    echo "Profile probe failed: launcher=${LAUNCHER}, status=${run_status}, success_ranks=${success_count}/${RANK_SIZE}" >&2
    echo "Log: ${combined_log}" >&2
    exit 1
fi

echo "Profile probe passed on ${RANK_SIZE} ranks"
echo "Combined log: ${combined_log}"
echo "Machine-readable results: ${jsonl}"
