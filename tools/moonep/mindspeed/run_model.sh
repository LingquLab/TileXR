#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_tilexr_home=$(cd "${script_dir}/../../.." && pwd)
default_config=${default_tilexr_home}/run/moonep/mindspeed/model_runner.env

usage() {
    cat <<'EOF'
Usage: run_model.sh [options]

Options:
  --mode single|multi       Launch locally or orchestrate all nodes over SSH.
  --backend tilexr|native   Select the MoonEP backend (default: tilexr).
  --seq-length COUNT        Model sequence length (default: 4096).
  --hidden-size COUNT       Model hidden size (default: 7168).
  --moe-router-topk COUNT   MoE router top-k (default: 8).
  --route-capture-dir PATH  Capture exact model routes into this host-local path.
  --route-capture-id ID     Stable capture ID paired with --route-capture-dir.
  --route-capture-skip-calls COUNT
                            Forward calls skipped before capture (default: 60).
  --route-capture-calls COUNT
                            Forward calls captured per rank (default: 10).
  --performance-capture-dir PATH
                            Write lightweight per-rank NPU Event timings here.
  --performance-capture-id ID
                            Stable ID paired with --performance-capture-dir.
  --performance-capture-skip-operators COUNT
                            Operators skipped before timing (default: 330).
  --performance-capture-operators COUNT
                            Operators timed per rank (default: 55).
  --collect-artifacts       Collect every node's profile/capture files on node 0.
  --profile                 Enable the one-iteration NPU profiler window.
  --stage-barrier           Add diagnostic world barriers before Dispatch/Combine.
  --hccl-inter-hccs-disable true|false
                            Select HCCL inter-HCCS behavior explicitly.
  --rank-table-file PATH    Use the same explicit HCCL rank table on every node.
  --udma-rootinfo-path PATH Use each node's TileXR UDMA RootInfo file.
  --configure               Replace the cached configuration interactively.
  --config PATH             Use an alternate cached configuration file.
  --master-port PORT        Distributed launcher port (default: 29501).
  --run-tag TAG             Result directory name (default: timestamped).
  --timeout SECONDS         Per-node model timeout (default: 900).
  --idle-wait SECONDS       Wait for a stable all-node idle window (default: 600).
  --dry-run                 Print commands without connecting or launching.
  -h, --help                Show this help.

The first run prompts for deployment paths and nodes, then caches the answers.
Only --configure updates an existing cache. SSH authentication remains external
to this script; passwords are never stored. Deploy code separately with Mutagen.
EOF
}

mode=single
backend=tilexr
sequence_length=4096
hidden_size=7168
router_topk=8
route_capture_dir=""
route_capture_id=""
route_capture_skip_calls=60
route_capture_calls=10
performance_capture_dir=""
performance_capture_id=""
performance_capture_skip_operators=330
performance_capture_operators=55
collect_artifacts=0
profile=0
stage_barrier=0
hccl_inter_hccs_disable=""
rank_table_file=""
udma_rootinfo_path=""
configure=0
dry_run=0
config=${TILEXR_MODEL_RUNNER_CONFIG:-${default_config}}
master_port=29501
timeout_sec=900
idle_wait_sec=600
run_tag=

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode=${2:?--mode requires a value}; shift 2 ;;
        --backend) backend=${2:?--backend requires a value}; shift 2 ;;
        --seq-length) sequence_length=${2:?--seq-length requires a value}; shift 2 ;;
        --hidden-size) hidden_size=${2:?--hidden-size requires a value}; shift 2 ;;
        --moe-router-topk) router_topk=${2:?--moe-router-topk requires a value}; shift 2 ;;
        --route-capture-dir)
            route_capture_dir=${2:?--route-capture-dir requires a value}
            shift 2
            ;;
        --route-capture-id)
            route_capture_id=${2:?--route-capture-id requires a value}
            shift 2
            ;;
        --route-capture-skip-calls)
            route_capture_skip_calls=${2:?--route-capture-skip-calls requires a value}
            shift 2
            ;;
        --route-capture-calls)
            route_capture_calls=${2:?--route-capture-calls requires a value}
            shift 2
            ;;
        --performance-capture-dir)
            performance_capture_dir=${2:?--performance-capture-dir requires a value}
            shift 2
            ;;
        --performance-capture-id)
            performance_capture_id=${2:?--performance-capture-id requires a value}
            shift 2
            ;;
        --performance-capture-skip-operators)
            performance_capture_skip_operators=${2:?--performance-capture-skip-operators requires a value}
            shift 2
            ;;
        --performance-capture-operators)
            performance_capture_operators=${2:?--performance-capture-operators requires a value}
            shift 2
            ;;
        --collect-artifacts) collect_artifacts=1; shift ;;
        --profile) profile=1; shift ;;
        --stage-barrier) stage_barrier=1; shift ;;
        --hccl-inter-hccs-disable)
            hccl_inter_hccs_disable=${2:?--hccl-inter-hccs-disable requires a value}
            shift 2
            ;;
        --rank-table-file) rank_table_file=${2:?--rank-table-file requires a value}; shift 2 ;;
        --udma-rootinfo-path)
            udma_rootinfo_path=${2:?--udma-rootinfo-path requires a value}
            shift 2
            ;;
        --configure) configure=1; shift ;;
        --config) config=${2:?--config requires a value}; shift 2 ;;
        --master-port) master_port=${2:?--master-port requires a value}; shift 2 ;;
        --run-tag) run_tag=${2:?--run-tag requires a value}; shift 2 ;;
        --timeout) timeout_sec=${2:?--timeout requires a value}; shift 2 ;;
        --idle-wait) idle_wait_sec=${2:?--idle-wait requires a value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${mode}" in single|multi) ;; *) printf 'Invalid mode: %s\n' "${mode}" >&2; exit 2 ;; esac
case "${backend}" in tilexr|native) ;; *) printf 'Invalid backend: %s\n' "${backend}" >&2; exit 2 ;; esac
for value in sequence_length hidden_size router_topk route_capture_calls \
    performance_capture_operators; do
    if [[ ! "${!value}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer\n' "${value}" >&2
        exit 2
    fi
done
if [[ ! "${route_capture_skip_calls}" =~ ^[0-9]+$ ||
      ! "${performance_capture_skip_operators}" =~ ^[0-9]+$ ]]; then
    echo "capture skip counts must be non-negative integers" >&2
    exit 2
fi
if [[ -n "${performance_capture_dir}" || -n "${performance_capture_id}" ]]; then
    if [[ "${backend}" != "tilexr" || -z "${performance_capture_dir}" ||
          -z "${performance_capture_id}" ||
          ! "${performance_capture_id}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "performance capture requires TileXR, a directory, and a safe capture ID" >&2
        exit 2
    fi
fi
if (( hidden_size % 128 != 0 || router_topk > 32 )); then
    echo "hidden-size must be divisible by 128 and moe-router-topk must be <= 32" >&2
    exit 2
fi
if [[ -n "${route_capture_dir}" || -n "${route_capture_id}" ]]; then
    if [[ -z "${route_capture_dir}" || -z "${route_capture_id}" ]]; then
        echo "--route-capture-dir and --route-capture-id must be provided together" >&2
        exit 2
    fi
    if [[ "${backend}" != "tilexr" ]]; then
        echo "route capture requires --backend tilexr" >&2
        exit 2
    fi
    if [[ ! "${route_capture_id}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "unsafe route capture ID: ${route_capture_id}" >&2
        exit 2
    fi
fi
case "${hccl_inter_hccs_disable}" in ""|true|false) ;;
    *) printf 'Invalid HCCL inter-HCCS value: %s\n' "${hccl_inter_hccs_disable}" >&2; exit 2 ;;
esac
if [[ ! "${master_port}" =~ ^[0-9]+$ ]] || (( master_port < 1 || master_port > 65535 )); then
    printf 'Invalid master port: %s\n' "${master_port}" >&2
    exit 2
fi
if [[ ! "${timeout_sec}" =~ ^[0-9]+$ ]] || (( timeout_sec < 1 )); then
    printf 'Invalid timeout: %s\n' "${timeout_sec}" >&2
    exit 2
fi
if [[ ! "${idle_wait_sec}" =~ ^[0-9]+$ ]]; then
    printf 'Invalid idle wait: %s\n' "${idle_wait_sec}" >&2
    exit 2
fi
if [[ -n "${run_tag}" && ! "${run_tag}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf 'Unsafe run tag: %s\n' "${run_tag}" >&2
    exit 2
fi

if [[ "${config}" =~ ^[A-Za-z]:[\\/] ]] && command -v cygpath >/dev/null 2>&1; then
    config=$(cygpath -u "${config}")
fi

prompt_value() {
    local variable=$1
    local label=$2
    local default_value=$3
    local answer=
    printf '%s [%s]: ' "${label}" "${default_value}" >&2
    IFS= read -r answer || true
    answer=${answer%$'\r'}
    printf -v "${variable}" '%s' "${answer:-${default_value}}"
}

write_config() {
    local config_dir
    config_dir=$(dirname "${config}")
    mkdir -p "${config_dir}"
    umask 077
    {
        printf '# Generated by tools/moonep/mindspeed/run_model.sh --configure\n'
        local variable
        for variable in \
            MODEL_RUNNER_NODES MODEL_RUNNER_SSH_USER MODEL_RUNNER_DEVICES_PER_NODE \
            MODEL_RUNNER_TILEXR_HOME MODEL_RUNNER_MODEL_ROOT MODEL_RUNNER_INSTALL_PREFIX \
            MODEL_RUNNER_CANN_ENV MODEL_RUNNER_CONDA_SH MODEL_RUNNER_CONDA_ENV \
            MODEL_RUNNER_NATIVE_ENV MODEL_RUNNER_TOKENIZER_PATH MODEL_RUNNER_DATA_PATH; do
            printf '%s=%q\n' "${variable}" "${!variable}"
        done
    } >"${config}"
    chmod 600 "${config}"
}

configure_runner() {
    local host_default
    host_default=$( (hostname -I 2>/dev/null || true) | awk '{print $1}')
    host_default=${host_default:-127.0.0.1}
    prompt_value MODEL_RUNNER_NODES "Node IPs or hostnames (space separated)" "${host_default}"
    prompt_value MODEL_RUNNER_SSH_USER "SSH user" root
    prompt_value MODEL_RUNNER_DEVICES_PER_NODE "Devices per node" 8
    prompt_value MODEL_RUNNER_TILEXR_HOME "TileXR path on every node" /home/c30061605/ai/TileXR
    prompt_value MODEL_RUNNER_MODEL_ROOT "Model stack root" "${MODEL_RUNNER_TILEXR_HOME}/run/multinode-validation/model"
    prompt_value MODEL_RUNNER_INSTALL_PREFIX "TileXR install prefix" "${MODEL_RUNNER_TILEXR_HOME}/install"
    prompt_value MODEL_RUNNER_CANN_ENV "CANN environment script" /home/pkg/b131/cann/set_env.sh
    prompt_value MODEL_RUNNER_CONDA_SH "Conda shell script" /home/miniconda3/etc/profile.d/conda.sh
    prompt_value MODEL_RUNNER_CONDA_ENV "Conda environment" ai_moe_test
    prompt_value MODEL_RUNNER_NATIVE_ENV "Native MoonEP environment script" "${MODEL_RUNNER_MODEL_ROOT}/moonep-native-build-97350ce0/moonep-native.env"
    prompt_value MODEL_RUNNER_TOKENIZER_PATH "Tokenizer path" /home/dataset/deepseek3
    prompt_value MODEL_RUNNER_DATA_PATH "Training data prefix" /home/dataset/deepseek3/enwiki_text_document
    write_config
}

if [[ "${configure}" -eq 1 || ! -f "${config}" ]]; then
    configure_runner
fi

# shellcheck disable=SC1090
source "${config}"
required_variables=(
    MODEL_RUNNER_NODES MODEL_RUNNER_SSH_USER MODEL_RUNNER_DEVICES_PER_NODE
    MODEL_RUNNER_TILEXR_HOME MODEL_RUNNER_MODEL_ROOT MODEL_RUNNER_INSTALL_PREFIX
    MODEL_RUNNER_CANN_ENV MODEL_RUNNER_CONDA_SH MODEL_RUNNER_CONDA_ENV
    MODEL_RUNNER_NATIVE_ENV MODEL_RUNNER_TOKENIZER_PATH MODEL_RUNNER_DATA_PATH
)
for variable in "${required_variables[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        printf 'Missing %s in %s; run with --configure\n' "${variable}" "${config}" >&2
        exit 2
    fi
done
if [[ ! "${MODEL_RUNNER_DEVICES_PER_NODE}" =~ ^[0-9]+$ ]] || \
    (( MODEL_RUNNER_DEVICES_PER_NODE < 1 )); then
    printf 'Invalid devices per node in %s\n' "${config}" >&2
    exit 2
fi

node_spec=${MODEL_RUNNER_NODES//,/ }
read -r -a nodes <<<"${node_spec}"
if [[ ${#nodes[@]} -eq 0 ]]; then
    printf 'No nodes configured in %s\n' "${config}" >&2
    exit 2
fi
if [[ "${mode}" == multi && ${#nodes[@]} -lt 2 ]]; then
    printf 'Multi-node mode needs at least two configured nodes\n' >&2
    exit 2
fi

if [[ -z "${run_tag}" ]]; then
    if [[ "${dry_run}" -eq 1 ]]; then
        run_tag=dry-run
    else
        run_tag="model_${backend}_${mode}_$(date +%Y%m%d-%H%M%S)_$$"
    fi
fi

quote_command() {
    local quoted=()
    local value
    for value in "$@"; do
        printf -v value '%q' "${value}"
        quoted+=("${value}")
    done
    local IFS=' '
    printf '%s' "${quoted[*]}"
}

node_env_prefix() {
    local env_args=()
    local name
    local pass_through=(
        TILEXR_MOONEP_DUMP_DFX_ON_ERROR
        TILEXR_MOONEP_DISPATCH_GROUP_WIDTH
        TILEXR_MOONEP_DISPATCH_PEER_MODE
        TILEXR_MOONEP_FLAG_DUMP_DIR
        TILEXR_MOONEP_FLAG_DUMP_MODE
        TILEXR_MOONEP_PLANNER_WAIT_ITERATIONS
        TILEXR_MINDSPEED_PREWARM_FRAMEWORK_OPS
    )
    for name in "${pass_through[@]}"; do
        if [[ -n "${!name:-}" ]]; then
            env_args+=("${name}=${!name}")
        fi
    done
    if [[ "${#env_args[@]}" -gt 0 ]]; then
        quote_command env "${env_args[@]}"
    fi
}

node_arguments() {
    local node_count=$1
    local node_rank=$2
    local master_addr=$3
    local args=(
        --backend "${backend}"
        --seq-length "${sequence_length}"
        --hidden-size "${hidden_size}"
        --moe-router-topk "${router_topk}"
        --node-count "${node_count}"
        --node-rank "${node_rank}"
        --master-addr "${master_addr}"
        --master-port "${master_port}"
        --devices-per-node "${MODEL_RUNNER_DEVICES_PER_NODE}"
        --tilexr-home "${MODEL_RUNNER_TILEXR_HOME}"
        --model-root "${MODEL_RUNNER_MODEL_ROOT}"
        --install-prefix "${MODEL_RUNNER_INSTALL_PREFIX}"
        --cann-env "${MODEL_RUNNER_CANN_ENV}"
        --conda-sh "${MODEL_RUNNER_CONDA_SH}"
        --conda-env "${MODEL_RUNNER_CONDA_ENV}"
        --native-env "${MODEL_RUNNER_NATIVE_ENV}"
        --tokenizer-path "${MODEL_RUNNER_TOKENIZER_PATH}"
        --data-path "${MODEL_RUNNER_DATA_PATH}"
        --run-tag "${run_tag}"
        --timeout "${timeout_sec}"
    )
    [[ "${profile}" -eq 1 ]] && args+=(--profile)
    [[ "${stage_barrier}" -eq 1 ]] && args+=(--stage-barrier)
    if [[ -n "${route_capture_dir}" ]]; then
        args+=(
            --route-capture-dir "${route_capture_dir}"
            --route-capture-id "${route_capture_id}"
            --route-capture-skip-calls "${route_capture_skip_calls}"
            --route-capture-calls "${route_capture_calls}"
        )
    fi
    if [[ -n "${performance_capture_dir}" ]]; then
        args+=(
            --performance-capture-dir "${performance_capture_dir}"
            --performance-capture-id "${performance_capture_id}"
            --performance-capture-skip-operators \
                "${performance_capture_skip_operators}"
            --performance-capture-operators "${performance_capture_operators}"
        )
    fi
    [[ -n "${hccl_inter_hccs_disable}" ]] && \
        args+=(--hccl-inter-hccs-disable "${hccl_inter_hccs_disable}")
    [[ -n "${rank_table_file}" ]] && args+=(--rank-table-file "${rank_table_file}")
    [[ -n "${udma_rootinfo_path}" ]] && \
        args+=(--udma-rootinfo-path "${udma_rootinfo_path}")
    quote_command "${args[@]}"
}

remote_runner=${MODEL_RUNNER_TILEXR_HOME}/tools/moonep/mindspeed/run_model_node.sh
remote_idle_probe=${MODEL_RUNNER_TILEXR_HOME}/tools/moonep/mindspeed/probe_idle.sh
local_runner=${script_dir}/run_model_node.sh
ssh_options=(-o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)

if [[ "${mode}" == single ]]; then
    args=$(node_arguments 1 0 127.0.0.1)
    command=$(quote_command bash "${local_runner}")
    env_prefix=$(node_env_prefix)
    [[ -n "${env_prefix}" ]] && command="${env_prefix} ${command}"
    command+=" ${args}"
    if [[ "${dry_run}" -eq 1 ]]; then
        printf 'mode=single local=1 node_rank=0\n%s\n' "${command}"
        exit 0
    fi
    eval "${command}"
    exit $?
fi

node_count=${#nodes[@]}
master_addr=${nodes[0]}
commands=()
targets=()
for node_rank in "${!nodes[@]}"; do
    target=${MODEL_RUNNER_SSH_USER}@${nodes[${node_rank}]}
    args=$(node_arguments "${node_count}" "${node_rank}" "${master_addr}")
    command=$(quote_command bash "${remote_runner}")
    env_prefix=$(node_env_prefix)
    [[ -n "${env_prefix}" ]] && command="${env_prefix} ${command}"
    command+=" ${args}"
    targets+=("${target}")
    commands+=("${command}")
    if [[ "${dry_run}" -eq 1 ]]; then
        printf 'node_rank=%s host=%s\n' "${node_rank}" "${nodes[${node_rank}]}"
        printf 'ssh %q %s\n' "${target}" "${command}"
    fi
done
[[ "${dry_run}" -eq 1 ]] && exit 0

controller_dir=${default_tilexr_home}/run/moonep/mindspeed/${run_tag}/controller
mkdir -p "${controller_dir}"

for index in "${!targets[@]}"; do
    probe=$(quote_command test -f "${remote_runner}")
    if ! ssh "${ssh_options[@]}" "${targets[${index}]}" "${probe}"; then
        printf 'SSH preflight failed for %s\n' "${targets[${index}]}" >&2
        exit 1
    fi
done

wait_for_stable_idle() {
    local deadline=$((SECONDS + idle_wait_sec))
    local consecutive=0
    local index idle all_idle probe_status
    local probe_dir=${controller_dir}/idle_probe
    local -a probe_pids=()
    mkdir -p "${probe_dir}"
    while true; do
        all_idle=1
        probe_pids=()
        for index in "${!targets[@]}"; do
            ssh "${ssh_options[@]}" "${targets[${index}]}" \
                "timeout 25s bash ${remote_idle_probe} --devices ${MODEL_RUNNER_DEVICES_PER_NODE}" \
                >"${probe_dir}/${index}.out" 2>"${probe_dir}/${index}.err" &
            probe_pids+=("$!")
        done
        for index in "${!targets[@]}"; do
            probe_status=0
            wait "${probe_pids[${index}]}" || probe_status=$?
            idle=$(tail -n 1 "${probe_dir}/${index}.out" 2>/dev/null || true)
            if [[ "${probe_status}" -ne 0 || ! "${idle}" =~ ^[0-9]+$ ]]; then
                idle=0
            fi
            if [[ "${idle}" -ne "${MODEL_RUNNER_DEVICES_PER_NODE}" ]]; then
                all_idle=0
            fi
            printf 'idle_gate host=%s idle=%s/%s stable=%s/3\n' \
                "${nodes[${index}]}" "${idle}" "${MODEL_RUNNER_DEVICES_PER_NODE}" \
                "${consecutive}" | tee -a "${controller_dir}/idle_gate.log"
        done
        if [[ "${all_idle}" -eq 1 ]]; then
            consecutive=$((consecutive + 1))
            [[ "${consecutive}" -ge 3 ]] && return 0
        else
            consecutive=0
        fi
        if (( SECONDS >= deadline )); then
            printf 'No stable all-node idle window within %s seconds\n' \
                "${idle_wait_sec}" >&2
            return 1
        fi
        sleep 5
    done
}

wait_for_stable_idle

declare -a pids=()
cleanup_started=0
cleanup_remote_runs() {
    [[ "${cleanup_started}" -eq 1 ]] && return
    cleanup_started=1
    local index stop_args stop_command cleanup_pid
    local cleanup_pids=()
    for index in "${!targets[@]}"; do
        stop_args=$(quote_command \
            --stop --backend "${backend}" --node-rank "${index}" \
            --tilexr-home "${MODEL_RUNNER_TILEXR_HOME}" --run-tag "${run_tag}")
        stop_command=$(quote_command bash "${remote_runner}")
        stop_command+=" ${stop_args}"
        ssh "${ssh_options[@]}" "${targets[${index}]}" "${stop_command}" \
            >"${controller_dir}/cleanup_${index}.log" 2>&1 &
        cleanup_pids+=("$!")
    done
    for cleanup_pid in "${cleanup_pids[@]}"; do
        wait "${cleanup_pid}" || true
    done
}

handle_signal() {
    local signal=$1
    printf 'Received %s; stopping all remote model jobs\n' "${signal}" >&2
    cleanup_remote_runs
    exit 130
}
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap 'handle_signal HUP' HUP

for index in "${!targets[@]}"; do
    status_file=${controller_dir}/status_${index}
    (
        set +e
        ssh "${ssh_options[@]}" "${targets[${index}]}" "${commands[${index}]}" \
            >"${controller_dir}/node_${index}.log" 2>&1
        status=$?
        printf '%s\n' "${status}" >"${status_file}"
        exit "${status}"
    ) &
    pids+=("$!")
done

remaining=${#pids[@]}
failed=0
declare -a completed=()
while (( remaining > 0 )); do
    for index in "${!pids[@]}"; do
        [[ "${completed[${index}]:-0}" -eq 1 ]] && continue
        status_file=${controller_dir}/status_${index}
        [[ -f "${status_file}" ]] || continue
        status=$(<"${status_file}")
        wait "${pids[${index}]}" || true
        completed[${index}]=1
        remaining=$((remaining - 1))
        printf 'node_rank=%s host=%s exit_code=%s log=%s\n' \
            "${index}" "${nodes[${index}]}" "${status}" \
            "${controller_dir}/node_${index}.log"
        if [[ "${status}" -ne 0 && "${failed}" -eq 0 ]]; then
            failed=1
            cleanup_remote_runs
        fi
    done
    (( remaining > 0 )) && sleep 1
done

if [[ "${failed}" -eq 0 ]] && grep -Eiq \
    'grad norm:[[:space:]]*(-?inf|nan)' "${controller_dir}"/node_*.log; then
    echo "A node reported a non-finite gradient norm" >&2
    failed=1
fi
if [[ "${failed}" -eq 0 ]] && ! grep -Eq \
    'iteration[[:space:]]+8/[[:space:]]*8.*lm loss:[[:space:]]*[0-9]' \
    "${controller_dir}"/node_*.log; then
    echo "No node reported iteration 8/8 with a finite language-model loss" >&2
    failed=1
fi
if [[ "${failed}" -ne 0 ]]; then
    printf 'Multi-node model failed; see %s\n' "${controller_dir}" >&2
    exit 1
fi
collect_model_artifacts() {
    local collected_root model_root index source target rsync_ssh
    collected_root=${default_tilexr_home}/run/moonep/mindspeed/${run_tag}/collected
    model_root=${MODEL_RUNNER_TILEXR_HOME}/run/moonep/mindspeed/${run_tag}/${backend}
    mkdir -p "${collected_root}"
    source=${model_root}/node_0/
    rsync -a --protect-args "${source}" "${collected_root}/node_0/"
    rsync_ssh="ssh ${ssh_options[*]}"
    for ((index = 1; index < node_count; index++)); do
        target=${targets[${index}]}:${model_root}/node_${index}/
        rsync -a --protect-args -e "${rsync_ssh}" \
            "${target}" "${collected_root}/node_${index}/"
        if [[ -n "${route_capture_dir}" ]]; then
            target=${targets[${index}]}:${route_capture_dir}/
            rsync -a --protect-args -e "${rsync_ssh}" \
                "${target}" "${route_capture_dir}/"
        fi
        if [[ -n "${performance_capture_dir}" ]]; then
            target=${targets[${index}]}:${performance_capture_dir}/
            rsync -a --protect-args -e "${rsync_ssh}" \
                "${target}" "${performance_capture_dir}/"
        fi
    done
    printf 'Collected model artifacts: %s\n' "${collected_root}"
}
if [[ "${collect_artifacts}" -eq 1 ]]; then
    collect_model_artifacts
fi
printf 'Multi-node model completed: %s\n' "${controller_dir}"
