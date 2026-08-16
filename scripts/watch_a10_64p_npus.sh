#!/usr/bin/env bash
set -uo pipefail

readonly REFRESH_MS=5000
readonly PROBE_TIMEOUT=4.5
readonly NPUS_PER_HOST=8
readonly SSH_USER="${TILEXR_NPU_WATCH_SSH_USER:-root}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly HOST_FILE="${TILEXR_A10_64P_HOST_FILE:-${script_dir}/hosts_a10_64p.txt}"

declare -a HOSTS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/watch_a10_64p_npus.sh [--once]

Watch every server listed in scripts/hosts_a10_64p.txt from 141.61.49.226.
The display refreshes every five seconds and reports freeNum per host and in total.
Set TILEXR_A10_64P_HOST_FILE to use an updated host file, or set
TILEXR_NPU_WATCH_SSH_USER to override the default SSH user (root).
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

load_hosts() {
    local host extra known_host

    [[ -f ${HOST_FILE} ]] || die "host file not found: ${HOST_FILE}"
    while read -r host extra || [[ -n ${host:-} ]]; do
        [[ -z ${host} || ${host} == \#* ]] && continue
        [[ -z ${extra} && ${host} =~ ^[0-9]+(\.[0-9]+){3}$ ]] ||
            die "invalid host row: ${host} ${extra}"
        for known_host in "${HOSTS[@]}"; do
            [[ ${host} != "${known_host}" ]] || die "duplicate host: ${host}"
        done
        HOSTS+=("${host}")
    done <"${HOST_FILE}"
    (( ${#HOSTS[@]} > 0 )) || die "host file is empty: ${HOST_FILE}"
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
    command -v "${command_name}" >/dev/null 2>&1 ||
        die "required command not found: ${command_name}"
done
load_hosts

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
    -o ControlPersist=15
    -o "ControlPath=${runtime_dir}/ssh-%C"
)

cleanup() {
    local host
    if (( terminal_output )); then
        printf '\033[?25h'
    fi
    for host in "${HOSTS[@]}"; do
        [[ "${local_ips}" == *" ${host} "* ]] && continue
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
    local host state state_without_free index
    local host_free total_free=0
    local total_npus=$((${#HOSTS[@]} * NPUS_PER_HOST))

    if (( terminal_output )); then
        printf '\033[H'
    fi
    for index in "${!HOSTS[@]}"; do
        host=${HOSTS[index]}
        state=${states[index]}
        state_without_free=${state//0/}
        host_free=$((${#state} - ${#state_without_free}))
        ((total_free += host_free))
        printf '%-15s: %s  freeNum=%d/%d\n' \
            "${host}" "${state}" "${host_free}" "${NPUS_PER_HOST}"
    done
    printf 'Total          : freeNum=%d/%d\n' "${total_free}" "${total_npus}"
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
