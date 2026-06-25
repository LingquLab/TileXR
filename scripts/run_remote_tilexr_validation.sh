#!/usr/bin/env bash

set -u

DEFAULT_PROJECT_DIR="/home/chy/TileXR"
DEFAULT_CANN_HOME="/usr/local/Ascend/cann-9.1.T500"
DEFAULT_MPI_HOME="/usr/local/mpich"

RUN_MODE="remote"
HOSTS=""
PASSWORD=""
PROJECT_DIR="${DEFAULT_PROJECT_DIR}"
DEVICES="0,1,2,3"
CANN_HOME="${DEFAULT_CANN_HOME}"
MPI_HOME="${DEFAULT_MPI_HOME}"
COLLECTIVES="unit"
SKIP_BUILD=0
SKIP_HCCL=0
SKIP_SUBTESTS=0
SCRIPT_PATH="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

usage() {
    cat <<EOF
Usage:
  bash scripts/run_remote_tilexr_validation.sh --hosts root@host1,root@host2 [options]
  bash scripts/run_remote_tilexr_validation.sh --local [options]

Options:
  --hosts <hosts>             Remote hosts, comma separated
  --password <password>       SSH password; omit to use SSH keys or promptless SSH
  --local                     Run only on this machine
  --project-dir <dir>         Project directory on target machines (default: ${DEFAULT_PROJECT_DIR})
  --devices <ids>             Device ids, comma separated (default: 0,1,2,3)
  --cann-home <dir>           CANN home path (default: ${DEFAULT_CANN_HOME})
  --mpi-home <dir>            MPI home path (default: ${DEFAULT_MPI_HOME})
  --collectives <mode>        skip | unit | correctness (default: unit)
  --skip-build                Skip core library build/install
  --skip-hccl                 Skip HCCL allreduce build/run
  --skip-subtests             Skip comm/sdma/udma/memory/ep tests
  --help                      Show this help
EOF
}

log_info() { echo "[INFO] $*"; }
log_ok() { echo "[OK] $*"; }
log_warn() { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*"; }

count_devices() {
    local value="$1"
    if [ -z "${value}" ]; then
        echo 0
        return
    fi
    awk -F',' '{ print NF }' <<< "${value}"
}

run_step() {
    local name="$1"
    shift
    log_info "BEGIN ${name}"
    "$@"
    local rc=$?
    if [ ${rc} -eq 0 ]; then
        log_ok "PASS ${name}"
    else
        log_error "FAIL ${name} rc=${rc}"
    fi
    return ${rc}
}

append_result() {
    local name="$1"
    local rc="$2"
    VALIDATION_RESULTS+=("${name}:${rc}")
    if [ "${rc}" -ne 0 ]; then
        VALIDATION_FAILED=1
    fi
}

setup_target_env() {
    export ASCEND_HOME_PATH="${CANN_HOME}"
    export ASCEND_DIR="${CANN_HOME}"
    export MPI_HOME="${MPI_HOME}"
    export PATH="${MPI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${MPI_HOME}/lib:${CANN_HOME}/${TILEXR_OS_ARCH:-aarch64}-linux/lib64:${LD_LIBRARY_PATH:-}"

    if [ -f "${CANN_HOME}/set_env.sh" ]; then
        # shellcheck disable=SC1090
        source "${CANN_HOME}/set_env.sh"
    fi

    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/scripts/common_env.sh"

    export ASCEND_HOME_PATH="${CANN_HOME}"
    export ASCEND_DIR="${CANN_HOME}"
    export MPI_HOME="${MPI_HOME}"
    export PATH="${MPI_HOME}/bin:${PATH}"
    export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
    export TILEXR_TEST_DEVICES="${DEVICES}"
    export TILEXR_ASCEND_DEV_NUM="$(count_devices "${DEVICES}")"
}

build_core() {
    local cmake_args=(-DCMAKE_INSTALL_PREFIX="${PROJECT_DIR}/install")
    if [ "${COLLECTIVES}" != "skip" ]; then
        cmake_args+=(-DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON)
    fi
    cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" "${cmake_args[@]}" && \
    cmake --build "${PROJECT_DIR}/build" --target install -j"$(nproc)"
}

build_hccl_test() {
    if [ ! -d "${CANN_HOME}/tools/hccl_test" ]; then
        log_warn "hccl_test directory not found: ${CANN_HOME}/tools/hccl_test"
        return 1
    fi
    cmake --build "${PROJECT_DIR}/build" --target install -j"$(nproc)" >/dev/null 2>&1 || true
    make -C "${CANN_HOME}/tools/hccl_test"
}

run_hccl_allreduce() {
    local rank_count
    rank_count="$(count_devices "${DEVICES}")"
    "${MPI_HOME}/bin/mpirun" -n "${rank_count}" \
        "${CANN_HOME}/tools/hccl_test/bin/all_reduce_test" \
        -p "${rank_count}" -b 1k -e 128m -f 2 -d int8 -n 20 -w 10 -c 1
}

run_cmake_test_dir() {
    local name="$1"
    local source_dir="${PROJECT_DIR}/tests/${name}"
    local build_dir="${PROJECT_DIR}/build/tests/${name}"
    shift

    if [ ! -f "${source_dir}/CMakeLists.txt" ]; then
        log_warn "skip ${name}: missing ${source_dir}/CMakeLists.txt"
        return 0
    fi

    cmake -S "${source_dir}" -B "${build_dir}" "$@" && \
    cmake --build "${build_dir}" -j"$(nproc)" && \
    run_configured_tests "${build_dir}"
}

run_configured_tests() {
    local build_dir="$1"
    if [ -f "${build_dir}/CTestTestfile.cmake" ] && grep -q "add_test" "${build_dir}/CTestTestfile.cmake"; then
        (cd "${build_dir}" && ctest --output-on-failure)
    else
        run_test_binaries "${build_dir}"
    fi
}

run_test_binaries() {
    local build_dir="$1"
    local rc=0
    local found=0
    while IFS= read -r exe; do
        found=1
        log_info "RUN ${exe}"
        "${exe}" || rc=1
    done < <(find "${build_dir}" -maxdepth 1 -type f -perm -111 \
        ! -name "cmake_install.cmake" \
        ! -name "CMakeCache.txt" \
        ! -name "Makefile" | sort)

    if [ ${found} -eq 0 ]; then
        log_warn "no test binaries found in ${build_dir}"
    fi
    return ${rc}
}

run_subtests() {
    local rc=0
    run_step "tests/comm" run_cmake_test_dir comm || rc=1
    run_step "tests/sdma" run_cmake_test_dir sdma -DBUILD_TILEXR_SDMA_DEMO=OFF || rc=1
    run_step "tests/udma" run_cmake_test_dir udma -DBUILD_TILEXR_UDMA_DEMO=OFF || rc=1
    run_step "tests/memory" run_cmake_test_dir memory -DBUILD_TILEXR_MEMORY_DEMO=OFF || rc=1
    run_step "tests/ep" run_cmake_test_dir ep -DBUILD_TILEXR_EP_DEMO=OFF || rc=1
    return ${rc}
}

ensure_collectives_installed() {
    if [ -f "${PROJECT_DIR}/install/lib/libtilexr-collectives.so" ] || \
       [ -f "${PROJECT_DIR}/install/lib64/libtilexr-collectives.so" ]; then
        return 0
    fi
    log_info "collectives library not found under install; building with TILEXR_BUILD_COLLECTIVES=ON"
    build_core
}

run_collectives_tests() {
    case "${COLLECTIVES}" in
        skip)
            log_info "skip collectives tests"
            return 0
            ;;
        unit)
            ensure_collectives_installed && \
            run_cmake_test_dir collectives -DBUILD_TILEXR_COLLECTIVES_CORRECTNESS=OFF
            ;;
        correctness)
            ensure_collectives_installed && \
            run_cmake_test_dir collectives -DBUILD_TILEXR_COLLECTIVES_CORRECTNESS=ON
            ;;
        *)
            log_error "invalid --collectives value: ${COLLECTIVES}"
            return 2
            ;;
    esac
}

run_local_validation() {
    cd "${PROJECT_DIR}" || return 1
    setup_target_env

    VALIDATION_RESULTS=()
    VALIDATION_FAILED=0

    log_info "host=$(hostname) project=${PROJECT_DIR} devices=${DEVICES} rank_count=${TILEXR_ASCEND_DEV_NUM}"
    log_info "ASCEND_HOME_PATH=${ASCEND_HOME_PATH} MPI_HOME=${MPI_HOME} collectives=${COLLECTIVES}"

    if [ ${SKIP_BUILD} -eq 0 ]; then
        run_step "core build/install" build_core
        append_result "core-build" "$?"
    else
        log_info "skip core build/install"
    fi

    if [ ${SKIP_HCCL} -eq 0 ]; then
        run_step "hccl_test build" build_hccl_test
        append_result "hccl-build" "$?"
        run_step "hccl allreduce" run_hccl_allreduce
        append_result "hccl-allreduce" "$?"
    else
        log_info "skip HCCL allreduce"
    fi

    if [ ${SKIP_SUBTESTS} -eq 0 ]; then
        run_step "project subtests" run_subtests
        append_result "subtests" "$?"
    else
        log_info "skip project subtests"
    fi

    run_step "collectives ${COLLECTIVES}" run_collectives_tests
    append_result "collectives-${COLLECTIVES}" "$?"

    echo "========== TileXR validation summary =========="
    local item name rc
    for item in "${VALIDATION_RESULTS[@]}"; do
        name="${item%%:*}"
        rc="${item##*:}"
        if [ "${rc}" -eq 0 ]; then
            echo "PASS ${name}"
        else
            echo "FAIL ${name} rc=${rc}"
        fi
    done
    echo "==============================================="

    return ${VALIDATION_FAILED}
}

shell_quote() {
    printf "%q" "$1"
}

remote_command() {
    cat <<EOF
cd $(shell_quote "${PROJECT_DIR}") && bash scripts/run_remote_tilexr_validation.sh --__remote-child --local --project-dir $(shell_quote "${PROJECT_DIR}") --devices $(shell_quote "${DEVICES}") --cann-home $(shell_quote "${CANN_HOME}") --mpi-home $(shell_quote "${MPI_HOME}") --collectives $(shell_quote "${COLLECTIVES}") $( [ ${SKIP_BUILD} -eq 1 ] && printf '%s' '--skip-build ' )$( [ ${SKIP_HCCL} -eq 1 ] && printf '%s' '--skip-hccl ' )$( [ ${SKIP_SUBTESTS} -eq 1 ] && printf '%s' '--skip-subtests ' )
EOF
}

run_remote_validation() {
    local rc=0
    local host
    local ssh_cmd
    local command_text

    IFS=',' read -ra HOST_ARRAY <<< "${HOSTS}"
    for host in "${HOST_ARRAY[@]}"; do
        log_info "BEGIN remote validation on ${host}"
        command_text="$(remote_command)"
        if [ -n "${PASSWORD}" ]; then
            if ! command -v sshpass >/dev/null 2>&1; then
                log_error "sshpass is required when --password is used"
                return 1
            fi
            sshpass -p "${PASSWORD}" ssh -o StrictHostKeyChecking=no "${host}" "${command_text}"
        else
            ssh "${host}" "${command_text}"
        fi
        local host_rc=$?
        if [ ${host_rc} -eq 0 ]; then
            log_ok "PASS remote validation on ${host}"
        else
            log_error "FAIL remote validation on ${host} rc=${host_rc}"
            rc=1
        fi
    done
    return ${rc}
}

while [ $# -gt 0 ]; do
    case "$1" in
        --hosts)
            HOSTS="$2"
            shift 2
            ;;
        --password)
            PASSWORD="$2"
            shift 2
            ;;
        --local)
            RUN_MODE="local"
            shift
            ;;
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --cann-home)
            CANN_HOME="$2"
            shift 2
            ;;
        --mpi-home)
            MPI_HOME="$2"
            shift 2
            ;;
        --collectives)
            COLLECTIVES="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-hccl)
            SKIP_HCCL=1
            shift
            ;;
        --skip-subtests)
            SKIP_SUBTESTS=1
            shift
            ;;
        --__remote-child)
            REMOTE_CHILD=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

case "${COLLECTIVES}" in
    skip|unit|correctness) ;;
    *)
        log_error "invalid --collectives value: ${COLLECTIVES}"
        usage
        exit 2
        ;;
esac

if [ "${RUN_MODE}" = "remote" ] && [ -z "${HOSTS}" ]; then
    log_error "--hosts is required unless --local is used"
    usage
    exit 2
fi

if [ "${RUN_MODE}" = "local" ]; then
    run_local_validation
else
    run_remote_validation
fi
