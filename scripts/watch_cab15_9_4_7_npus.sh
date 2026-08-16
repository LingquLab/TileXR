#!/usr/bin/env bash
set -uo pipefail

readonly REFRESH_MS=2000
readonly PROBE_TIMEOUT=1.8
readonly HOSTS_PER_CABINET=8
readonly NPUS_PER_CABINET=64
readonly EXPECTED_HOSTS=32
readonly SSH_USER="${TILEXR_NPU_WATCH_SSH_USER:-root}"

readonly -a CABINETS=(15 9 4 7)
readonly -a HOSTS=(
    # Cabinet 15, CPU1 through CPU8.
    141.62.17.70
    141.62.17.66
    141.62.17.62
    141.62.17.58
    141.62.17.30
    141.62.17.26
    141.62.17.22
    141.62.17.18
    # Cabinet 9, CPU1 through CPU8.
    141.61.55.118
    141.61.55.114
    141.61.55.110
    141.61.55.106
    141.61.55.78
    141.61.55.74
    141.61.55.70
    141.61.55.66
    # Cabinet 4, CPU1 through CPU8.
    141.61.52.116
    141.61.52.120
    141.61.52.128
    141.61.52.124
    141.61.52.156
    141.61.52.160
    141.61.52.164
    141.61.52.167
    # Cabinet 7, CPU1 through CPU8.
    141.61.54.220
    141.61.54.216
    141.61.54.212
    141.61.54.208
    141.61.54.180
    141.61.54.176
    141.61.54.172
    141.61.54.168
)

usage() {
    cat <<'EOF'
Usage: bash scripts/watch_cab15_9_4_7_npus.sh [--once]

Watch all 256 NPUs in cabinets 15, 9, 4, and 7 from cabinet 8 CPU1.
The display refreshes every two seconds. Set TILEXR_NPU_WATCH_SSH_USER to
override the default SSH user (root).
EOF
}

once=0
case "${1:-}" in
    "") ;;
    --once) once=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

for command_name in ssh timeout npu-smi; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'ERROR: required command not found: %s\n' "${command_name}" >&2
        exit 1
    fi
done
if (( ${#HOSTS[@]} != EXPECTED_HOSTS ||
        ${#CABINETS[@]} * HOSTS_PER_CABINET != EXPECTED_HOSTS )); then
    printf 'ERROR: cabinet/host configuration is incomplete\n' >&2
    exit 1
fi

runtime_dir=$(mktemp -d "${TMPDIR:-/tmp}/tilexr-npu-watch.XXXXXX") || exit 1
local_ips=" $(hostname -I 2>/dev/null || true) "
terminal_output=0
[[ -t 1 ]] && terminal_output=1

ssh_options=(
    -o BatchMode=yes
    -o ConnectTimeout=1
    -o ConnectionAttempts=1
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
    -o ControlMaster=auto
    -o ControlPersist=10
    -o "ControlPath=${runtime_dir}/ssh-%C"
)

cleanup() {
    local host
    if (( terminal_output )); then
        printf '\033[?25h'
    fi
    for host in "${HOSTS[@]}"; do
        ssh "${ssh_options[@]}" -O exit "${SSH_USER}@${host}" \
            >/dev/null 2>&1 || true
    done
    rm -f -- "${runtime_dir}"/* 2>/dev/null || true
    rmdir -- "${runtime_dir}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

read -r -d '' remote_probe <<'REMOTE_PROBE' || true
set -u
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/tilexr-npu-probe.XXXXXX") || exit 1
cleanup_probe() {
    rm -f -- "${probe_dir}"/* 2>/dev/null || true
    rmdir -- "${probe_dir}" 2>/dev/null || true
}
trap cleanup_probe EXIT

pids=()
for device in {0..7}; do
    LC_ALL=C npu-smi info -t proc-mem -i "${device}" \
        >"${probe_dir}/${device}.out" 2>/dev/null &
    pids[device]=$!
done
for device in {0..7}; do
    if ! wait "${pids[device]}"; then
        printf '?'
    elif grep -q 'Process id[[:space:]]*:' "${probe_dir}/${device}.out"; then
        printf '1'
    elif grep -q "NPU ID[[:space:]]*:[[:space:]]*${device}" \
            "${probe_dir}/${device}.out"; then
        printf '0'
    else
        printf '?'
    fi
done
printf '\n'
REMOTE_PROBE

probe_host() {
    local host=$1
    if [[ "${local_ips}" == *" ${host} "* ]]; then
        timeout "${PROBE_TIMEOUT}" bash -s <<<"${remote_probe}"
    else
        timeout "${PROBE_TIMEOUT}" ssh "${ssh_options[@]}" \
            "${SSH_USER}@${host}" bash -s <<<"${remote_probe}"
    fi
}

warm_connections() {
    local host index
    local -a warm_pids=()
    for index in "${!HOSTS[@]}"; do
        host=${HOSTS[index]}
        [[ "${local_ips}" == *" ${host} "* ]] && continue
        timeout 3 ssh "${ssh_options[@]}" "${SSH_USER}@${host}" true \
            >/dev/null 2>&1 &
        warm_pids+=("$!")
    done
    for index in "${!warm_pids[@]}"; do
        wait "${warm_pids[index]}" 2>/dev/null || true
    done
}

render_snapshot() {
    local -a states=("$@")
    local cabinet cabinet_index first_host host_index
    local row state state_without_free
    local free_count

    if (( terminal_output )); then
        printf '\033[H'
    fi
    for cabinet_index in "${!CABINETS[@]}"; do
        cabinet=${CABINETS[cabinet_index]}
        first_host=$((cabinet_index * HOSTS_PER_CABINET))
        row=""
        free_count=0
        for ((host_index = first_host;
                host_index < first_host + HOSTS_PER_CABINET; host_index++)); do
            state=${states[host_index]}
            row+="${state} "
            state_without_free=${state//0/}
            ((free_count += ${#state} - ${#state_without_free}))
        done
        printf 'Cabinet %-2s: %s  freeNum=%d/%d\n' \
            "${cabinet}" "${row% }" "${free_count}" "${NPUS_PER_CABINET}"
    done
    printf '0=idle  1=busy  ?=probe_failed\n'
}

if (( terminal_output )); then
    printf '\033[2J\033[H\033[?25l'
fi
warm_connections

while true; do
    cycle_start_ms=$(date +%s%3N)
    declare -a pids=()
    declare -a states=()

    for index in "${!HOSTS[@]}"; do
        probe_host "${HOSTS[index]}" >"${runtime_dir}/${index}.state" 2>/dev/null &
        pids[index]=$!
    done
    for index in "${!HOSTS[@]}"; do
        wait "${pids[index]}" 2>/dev/null || true
        IFS= read -r state <"${runtime_dir}/${index}.state" || state=""
        state=${state//$'\r'/}
        [[ ${state} =~ ^[01?]{8}$ ]] || state="????????"
        states[index]=${state}
    done

    render_snapshot "${states[@]}"
    (( once )) && break

    now_ms=$(date +%s%3N)
    remaining_ms=$((REFRESH_MS - (now_ms - cycle_start_ms)))
    if (( remaining_ms > 0 )); then
        printf -v sleep_seconds '%d.%03d' \
            $((remaining_ms / 1000)) $((remaining_ms % 1000))
        sleep "${sleep_seconds}"
    fi
done
