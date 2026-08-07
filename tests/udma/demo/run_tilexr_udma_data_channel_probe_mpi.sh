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
QP_ROUTE_SPEC="${TILEXR_UDMA_QP_ROUTE_SPEC:-}"
MISMATCH_QP_ROUTE_SPEC="${TILEXR_DEMO_MISMATCH_QP_ROUTE_SPEC:-}"
EXPECT_UDMA="${TILEXR_DEMO_EXPECT_UDMA:-1}"
EXPECT_QP_COUNT="${TILEXR_DEMO_EXPECT_QP_COUNT:-}"
REGISTERED_BYTES="${TILEXR_DEMO_REGISTERED_BYTES:-2097152}"
REREGISTER="${TILEXR_DEMO_REREGISTER:-1}"
WARMUP_ITERS="${TILEXR_DEMO_WARMUP_ITERS:-0}"
TIMED_ITERS="${TILEXR_DEMO_TIMED_ITERS:-1}"
PERF_BARRIER_ADDR="${TILEXR_DEMO_PERF_BARRIER_ADDR:-}"
PERF_BARRIER_RANK_BASE="${TILEXR_DEMO_PERF_BARRIER_RANK_BASE:-0}"
PERF_BARRIER_SIZE="${TILEXR_DEMO_PERF_BARRIER_SIZE:-0}"
TIMEOUT_SECONDS="${TILEXR_DEMO_TIMEOUT_SECONDS:-300}"
CASE_NAME=""
UNSET_QP_ROUTE_SPEC=0
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
  --qp-route-spec <spec> Generic UDMA route rules, e.g. port_count:6,port_count:2
  --mismatch-qp-route-spec <spec> Override the route spec on rank 1
  --case <name>          legacy, exact-2m, two-qp, three-qp, missing-route, mismatch, or reuse
  --expect-udma <0|1>    Require UDMA unavailable or available (default: 1)
  --expect-qp-count <N>  Require the exact initialized QP count (default: 1, or 0 with --expect-udma 0)
  --registered-bytes <N> Register exactly N bytes (default: 2097152)
  --reregister <0|1>     Enable or disable the register/unregister reuse probe (default: 1)
  --no-reregister        Disable the register/unregister reuse probe
  --warmup-iters <N>     Warmup kernel launches before measured runs (default: 0)
  --iterations <N>       Timed kernel launches (default: 1)
  --perf-barrier-addr <ip:port> Shared timed-window barrier endpoint
  --perf-barrier-rank-base <N>  First global participant id for this MPI job (default: 0)
  --perf-barrier-size <N>       Total participants across synchronized MPI jobs (default: 0, disabled)
  --timeout <seconds>    Bound the complete MPI run (default: 300)
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
        --qp-route-spec)
            QP_ROUTE_SPEC="$2"
            shift 2
            ;;
        --mismatch-qp-route-spec)
            MISMATCH_QP_ROUTE_SPEC="$2"
            shift 2
            ;;
        --case)
            CASE_NAME="$2"
            shift 2
            ;;
        --expect-udma)
            EXPECT_UDMA="$2"
            shift 2
            ;;
        --expect-qp-count)
            EXPECT_QP_COUNT="$2"
            shift 2
            ;;
        --registered-bytes)
            REGISTERED_BYTES="$2"
            shift 2
            ;;
        --reregister)
            REREGISTER="$2"
            shift 2
            ;;
        --no-reregister)
            REREGISTER=0
            shift
            ;;
        --warmup-iters)
            WARMUP_ITERS="$2"
            shift 2
            ;;
        --iterations)
            TIMED_ITERS="$2"
            shift 2
            ;;
        --perf-barrier-addr)
            PERF_BARRIER_ADDR="$2"
            shift 2
            ;;
        --perf-barrier-rank-base)
            PERF_BARRIER_RANK_BASE="$2"
            shift 2
            ;;
        --perf-barrier-size)
            PERF_BARRIER_SIZE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
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

case "${CASE_NAME}" in
    "") ;;
    legacy)
        QP_ROUTE_SPEC=""; MISMATCH_QP_ROUTE_SPEC=""; EXPECT_UDMA=1; EXPECT_QP_COUNT=1; REREGISTER=0
        UNSET_QP_ROUTE_SPEC=1 ;;
    exact-2m)
        QP_ROUTE_SPEC=""; MISMATCH_QP_ROUTE_SPEC=""; EXPECT_UDMA=1; EXPECT_QP_COUNT=1
        REGISTERED_BYTES=2097152; REREGISTER=0; UNSET_QP_ROUTE_SPEC=1 ;;
    two-qp)
        QP_ROUTE_SPEC="port_count:6,port_count:2"; MISMATCH_QP_ROUTE_SPEC=""
        EXPECT_UDMA=1; EXPECT_QP_COUNT=2; REREGISTER=0 ;;
    three-qp)
        QP_ROUTE_SPEC="port_count:6,port_count:6,port_count:2"; MISMATCH_QP_ROUTE_SPEC=""
        EXPECT_UDMA=1; EXPECT_QP_COUNT=3; REREGISTER=0 ;;
    missing-route)
        QP_ROUTE_SPEC="port_count:4294967295"; MISMATCH_QP_ROUTE_SPEC=""
        EXPECT_UDMA=0; EXPECT_QP_COUNT=0; REREGISTER=0 ;;
    mismatch)
        QP_ROUTE_SPEC="port_count:6"; MISMATCH_QP_ROUTE_SPEC="port_count:2"
        EXPECT_UDMA=0; EXPECT_QP_COUNT=0; REREGISTER=0 ;;
    reuse)
        QP_ROUTE_SPEC=""; MISMATCH_QP_ROUTE_SPEC=""; EXPECT_UDMA=1; EXPECT_QP_COUNT=1; REREGISTER=1
        UNSET_QP_ROUTE_SPEC=1 ;;
    *)
        echo "unknown --case '${CASE_NAME}'" >&2
        exit 2
        ;;
esac

if [[ -z "${HOSTS}" || -z "${RANK_SIZE}" || -z "${COMM_ID}" ]]; then
    usage >&2
    exit 2
fi

if [[ -z "${EXPECT_QP_COUNT}" ]]; then
    EXPECT_QP_COUNT=$([[ "${EXPECT_UDMA}" == "0" ]] && echo 0 || echo 1)
fi
if [[ ! "${EXPECT_UDMA}" =~ ^[01]$ ]]; then
    echo "--expect-udma must be 0 or 1" >&2
    exit 2
fi
if [[ ! "${EXPECT_QP_COUNT}" =~ ^[0-8]$ ]]; then
    echo "--expect-qp-count must be an integer from 0 through 8" >&2
    exit 2
fi
if [[ ( "${EXPECT_UDMA}" == "1" && "${EXPECT_QP_COUNT}" == "0" ) ||
      ( "${EXPECT_UDMA}" == "0" && "${EXPECT_QP_COUNT}" != "0" ) ]]; then
    echo "--expect-qp-count must be positive when UDMA is expected and zero otherwise" >&2
    exit 2
fi
if [[ ! "${REGISTERED_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--registered-bytes must be a positive decimal integer" >&2
    exit 2
fi
if [[ ! "${REREGISTER}" =~ ^[01]$ ]]; then
    echo "--reregister must be 0 or 1" >&2
    exit 2
fi
if [[ ! "${WARMUP_ITERS}" =~ ^[0-9]+$ ]]; then
    echo "--warmup-iters must be a non-negative decimal integer" >&2
    exit 2
fi
if [[ ! "${TIMED_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--iterations must be a positive decimal integer" >&2
    exit 2
fi
if [[ ! "${PERF_BARRIER_RANK_BASE}" =~ ^[0-9]+$ ]]; then
    echo "--perf-barrier-rank-base must be a non-negative decimal integer" >&2
    exit 2
fi
if [[ ! "${PERF_BARRIER_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "--perf-barrier-size must be a non-negative decimal integer" >&2
    exit 2
fi
if [[ ( -z "${PERF_BARRIER_ADDR}" &&
        ( "${PERF_BARRIER_RANK_BASE}" != "0" || "${PERF_BARRIER_SIZE}" != "0" ) ) ||
      ( -n "${PERF_BARRIER_ADDR}" && "${PERF_BARRIER_SIZE}" == "0" ) ]]; then
    echo "performance barrier requires --perf-barrier-addr and a positive --perf-barrier-size" >&2
    exit 2
fi
if [[ ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--timeout must be a positive decimal integer" >&2
    exit 2
fi
if [[ -n "${MISMATCH_QP_ROUTE_SPEC}" && "${RANK_SIZE}" -lt 2 ]]; then
    echo "--mismatch-qp-route-spec requires at least two ranks" >&2
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
export TILEXR_DEMO_EXPECT_UDMA="${EXPECT_UDMA}"
export TILEXR_DEMO_EXPECT_QP_COUNT="${EXPECT_QP_COUNT}"
export TILEXR_DEMO_REGISTERED_BYTES="${REGISTERED_BYTES}"
export TILEXR_DEMO_REREGISTER="${REREGISTER}"
export TILEXR_DEMO_WARMUP_ITERS="${WARMUP_ITERS}"
export TILEXR_DEMO_TIMED_ITERS="${TIMED_ITERS}"
export TILEXR_DEMO_PERF_BARRIER_ADDR="${PERF_BARRIER_ADDR}"
export TILEXR_DEMO_PERF_BARRIER_RANK_BASE="${PERF_BARRIER_RANK_BASE}"
export TILEXR_DEMO_PERF_BARRIER_SIZE="${PERF_BARRIER_SIZE}"
export TILEXR_ENABLE_IPC="${TILEXR_ENABLE_IPC:-0}"
export TILEXR_ENABLE_SDMA="${TILEXR_ENABLE_SDMA:-0}"
export LD_LIBRARY_PATH="${TILEXR_ROOT}/install/lib64:${TILEXR_ROOT}/install/lib:${INSTALL_DIR}/lib64:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

MPI_BIN="${MPI_HOME}/bin/mpirun"
if [[ ! -x "${MPI_BIN}" ]]; then
    MPI_BIN="$(command -v mpirun || true)"
fi
if [[ -z "${MPI_BIN}" || ! -x "${MPI_BIN}" ]]; then
    echo "mpirun not found; set --mpi-home or MPI_HOME" >&2
    exit 1
fi

MPI_ARGS=(
    -hosts "${HOSTS}" -n "${RANK_SIZE}"
    -genv TILEXR_COMM_ID "${TILEXR_COMM_ID}"
    -genv TILEXR_DEMO_BARRIER_ADDR "${TILEXR_DEMO_BARRIER_ADDR}"
    -genv TILEXR_DEMO_TEST_TYPE "${TILEXR_DEMO_TEST_TYPE}"
    -genv TILEXR_DEMO_ELEMENTS_PER_RANK "${TILEXR_DEMO_ELEMENTS_PER_RANK}"
    -genv TILEXR_DEMO_NPUS "${TILEXR_DEMO_NPUS}"
    -genv TILEXR_DEMO_FIRST_NPU "${TILEXR_DEMO_FIRST_NPU}"
    -genv TILEXR_DEMO_DEVICES "${TILEXR_DEMO_DEVICES}"
    -genv TILEXR_DEMO_REQUIRE_SDMA "${TILEXR_DEMO_REQUIRE_SDMA}"
    -genv TILEXR_DEMO_EXPECT_UDMA "${TILEXR_DEMO_EXPECT_UDMA}"
    -genv TILEXR_DEMO_EXPECT_QP_COUNT "${TILEXR_DEMO_EXPECT_QP_COUNT}"
    -genv TILEXR_DEMO_REGISTERED_BYTES "${TILEXR_DEMO_REGISTERED_BYTES}"
    -genv TILEXR_DEMO_REREGISTER "${TILEXR_DEMO_REREGISTER}"
    -genv TILEXR_DEMO_WARMUP_ITERS "${TILEXR_DEMO_WARMUP_ITERS}"
    -genv TILEXR_DEMO_TIMED_ITERS "${TILEXR_DEMO_TIMED_ITERS}"
    -genv TILEXR_ENABLE_IPC "${TILEXR_ENABLE_IPC}"
    -genv TILEXR_ENABLE_SDMA "${TILEXR_ENABLE_SDMA}"
    -genv LD_LIBRARY_PATH "${LD_LIBRARY_PATH}"
)
if [[ -n "${TILEXR_DEMO_PERF_BARRIER_ADDR}" ]]; then
    MPI_ARGS+=(
        -genv TILEXR_DEMO_PERF_BARRIER_ADDR "${TILEXR_DEMO_PERF_BARRIER_ADDR}"
        -genv TILEXR_DEMO_PERF_BARRIER_RANK_BASE "${TILEXR_DEMO_PERF_BARRIER_RANK_BASE}"
        -genv TILEXR_DEMO_PERF_BARRIER_SIZE "${TILEXR_DEMO_PERF_BARRIER_SIZE}"
    )
fi
RUN_COMMAND=( "${bin}" )
if [[ "${UNSET_QP_ROUTE_SPEC}" == "0" ]]; then
    MPI_ARGS+=( -genv TILEXR_UDMA_QP_ROUTE_SPEC "${QP_ROUTE_SPEC}" )
fi
if [[ -n "${MISMATCH_QP_ROUTE_SPEC}" ]]; then
    MPI_ARGS+=( -genv TILEXR_DEMO_MISMATCH_QP_ROUTE_SPEC "${MISMATCH_QP_ROUTE_SPEC}" )
fi
if [[ "${UNSET_QP_ROUTE_SPEC}" == "1" || -n "${MISMATCH_QP_ROUTE_SPEC}" ]]; then
    MPI_ARGS+=( -genv TILEXR_DEMO_UNSET_QP_ROUTE_SPEC "${UNSET_QP_ROUTE_SPEC}" )
    RUN_COMMAND=( bash -lc '
        if [[ "${TILEXR_DEMO_UNSET_QP_ROUTE_SPEC:-0}" == "1" ]]; then
            unset TILEXR_UDMA_QP_ROUTE_SPEC
        fi
        rank=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${MV2_COMM_WORLD_RANK:-${RANK:-0}}}}
        if [[ "${rank}" == "1" && -n "${TILEXR_DEMO_MISMATCH_QP_ROUTE_SPEC:-}" ]]; then
            export TILEXR_UDMA_QP_ROUTE_SPEC="${TILEXR_DEMO_MISMATCH_QP_ROUTE_SPEC}"
        fi
        exec "$0"
    ' "${bin}" )
fi

exec timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECONDS}" \
    "${MPI_BIN}" "${MPI_ARGS[@]}" "${RUN_COMMAND[@]}"
