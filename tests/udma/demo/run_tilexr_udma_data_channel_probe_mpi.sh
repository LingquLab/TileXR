#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UDMA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TILEXR_ROOT="$(cd "${UDMA_DIR}/../.." && pwd)"
INSTALL_DIR="${UDMA_DIR}/install"

HOSTS=""
RANK_SIZE=""
COMM_ID=""
TEST_TYPE="0"
ELEMENTS="16"
NPU_COUNT="4"
FIRST_NPU="0"
DEVICES="0,1,2,3"
REQUIRE_SDMA=0
MPI_HOME="${MPI_HOME:-/usr/local/mpich}"

usage() {
    cat <<EOF
Usage: bash tests/udma/demo/run_tilexr_udma_data_channel_probe_mpi.sh --hosts host1:4,host2:4 --rank-size 8 --comm-id host1_ip:port [options]

Options:
  --hosts <hosts>        MPI hosts with slots, e.g. 141.62.19.156:4,141.62.19.108:4
  --rank-size <N>        Global rank size
  --comm-id <ip:port>    TileXR rendezvous address; rank 0 host must listen here
  --test-type <0|1>      0: all-gather UDMA put, 1: put-signal (default: 0)
  --elements <count>     Elements per rank (default: 16)
  --npu-count <count>    Per-host visible NPU count (default: 4)
  --first-npu <id>       First logical NPU id when --devices is not used (default: 0)
  --devices <ids>        Per-host logical device ids, comma separated (default: 0,1,2,3)
  --mpi-home <dir>       MPI home (default: /usr/local/mpich or MPI_HOME)
  --require-sdma         Fail if TileXR SDMA is unavailable
  --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)
            HOSTS="$2"
            shift 2
            ;;
        --rank-size)
            RANK_SIZE="$2"
            shift 2
            ;;
        --comm-id)
            COMM_ID="$2"
            shift 2
            ;;
        --test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --elements)
            ELEMENTS="$2"
            shift 2
            ;;
        --npu-count)
            NPU_COUNT="$2"
            shift 2
            ;;
        --first-npu)
            FIRST_NPU="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --mpi-home)
            MPI_HOME="$2"
            shift 2
            ;;
        --require-sdma)
            REQUIRE_SDMA=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${HOSTS}" || -z "${RANK_SIZE}" || -z "${COMM_ID}" ]]; then
    usage >&2
    exit 2
fi

SELECTED_MPI_HOME="${MPI_HOME}"
: "${ASCEND_HOME_PATH:=}"
: "${LD_LIBRARY_PATH:=}"
source "${TILEXR_ROOT}/scripts/common_env.sh"
MPI_HOME="${SELECTED_MPI_HOME}"

bin="${INSTALL_DIR}/bin/tilexr_udma_demo"
if [[ ! -x "${bin}" ]]; then
    echo "Missing demo binary: ${bin}" >&2
    echo "Build it with: cd ${UDMA_DIR} && bash build.sh" >&2
    exit 1
fi

export TILEXR_COMM_ID="${COMM_ID}"
export TILEXR_DEMO_BARRIER_ADDR="${COMM_ID}"
export TILEXR_DEMO_TEST_TYPE="${TEST_TYPE}"
export TILEXR_DEMO_ELEMENTS_PER_RANK="${ELEMENTS}"
export TILEXR_DEMO_NPUS="${NPU_COUNT}"
export TILEXR_DEMO_FIRST_NPU="${FIRST_NPU}"
export TILEXR_DEMO_DEVICES="${DEVICES}"
export TILEXR_DEMO_REQUIRE_SDMA="${REQUIRE_SDMA}"
export LD_LIBRARY_PATH="${TILEXR_ROOT}/install/lib64:${TILEXR_ROOT}/install/lib:${INSTALL_DIR}/lib64:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

MPI_BIN="${MPI_HOME}/bin/mpirun"
if [[ ! -x "${MPI_BIN}" ]]; then
    MPI_BIN="$(command -v mpirun || true)"
fi
if [[ -z "${MPI_BIN}" || ! -x "${MPI_BIN}" ]]; then
    echo "mpirun not found; set --mpi-home or MPI_HOME" >&2
    exit 1
fi

exec "${MPI_BIN}" -hosts "${HOSTS}" -n "${RANK_SIZE}" \
    -genv TILEXR_COMM_ID "${TILEXR_COMM_ID}" \
    -genv TILEXR_DEMO_BARRIER_ADDR "${TILEXR_DEMO_BARRIER_ADDR}" \
    -genv TILEXR_DEMO_TEST_TYPE "${TILEXR_DEMO_TEST_TYPE}" \
    -genv TILEXR_DEMO_ELEMENTS_PER_RANK "${TILEXR_DEMO_ELEMENTS_PER_RANK}" \
    -genv TILEXR_DEMO_NPUS "${TILEXR_DEMO_NPUS}" \
    -genv TILEXR_DEMO_FIRST_NPU "${TILEXR_DEMO_FIRST_NPU}" \
    -genv TILEXR_DEMO_DEVICES "${TILEXR_DEMO_DEVICES}" \
    -genv TILEXR_DEMO_REQUIRE_SDMA "${TILEXR_DEMO_REQUIRE_SDMA}" \
    -genv LD_LIBRARY_PATH "${LD_LIBRARY_PATH}" \
    "${bin}"
