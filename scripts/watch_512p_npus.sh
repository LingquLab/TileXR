#!/usr/bin/env bash
set -uo pipefail

readonly REFRESH_MS="${TILEXR_NPU_WATCH_REFRESH_MS:-5000}"
readonly PROBE_TIMEOUT="${TILEXR_NPU_WATCH_PROBE_TIMEOUT:-4.5}"
readonly HOSTS_PER_CABINET=8
readonly NPUS_PER_CABINET=64
readonly SSH_USER="${TILEXR_NPU_WATCH_SSH_USER:-root}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly HOST_FILE="${TILEXR_512P_HOST_FILE:-${script_dir}/hosts_512p.txt}"

declare -a CABINET_FRAMES=()
declare -a CABINETS=()
declare -a HOSTS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/watch_512p_npus.sh [--once]

Watch every server listed in scripts/hosts_512p.txt from cabinet 8 CPU1.
The display refreshes every five seconds and reports freeNum per cabinet.
Set TILEXR_512P_HOST_FILE to use an updated mapping file, or set
TILEXR_NPU_WATCH_SSH_USER to override the default SSH user (root).
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

load_hosts() {
    local frame cabinet cpu host extra
    local current_cabinet=""
    local current_frame=""
    local expected_cpu=1

    [[ -f ${HOST_FILE} ]] || die "host mapping not found: ${HOST_FILE}"
    while read -r frame cabinet cpu host extra; do
        [[ -z ${frame} || ${frame} == \#* ]] && continue
        [[ -z ${extra} && ${frame} =~ ^C[0-9]+-[AB][0-9]+$ &&
            ${cabinet} =~ ^[0-9]+$ && ${cpu} =~ ^CPU[1-8]$ &&
            ${host} =~ ^[0-9]+(\.[0-9]+){3}$ ]] ||
            die "invalid host mapping row: ${frame} ${cabinet} ${cpu} ${host} ${extra}"

        if [[ ${cabinet} != "${current_cabinet}" ]]; then
            if [[ -n ${current_cabinet} && ${expected_cpu} -ne 9 ]]; then
                die "cabinet ${current_cabinet} does not contain CPU1 through CPU8"
            fi
            CABINET_FRAMES+=("${frame}")
            CABINETS+=("${cabinet}")
            current_cabinet=${cabinet}
            current_frame=${frame}
            expected_cpu=1
        fi
        [[ ${frame} == "${current_frame}" && ${cpu} == "CPU${expected_cpu}" ]] ||
            die "cabinet ${cabinet} host order must be CPU1 through CPU8"
        HOSTS+=("${host}")
        ((expected_cpu++))
    done <"${HOST_FILE}"

    [[ -n ${current_cabinet} && ${expected_cpu} -eq 9 ]] ||
        die "last cabinet does not contain CPU1 through CPU8"
    (( ${#HOSTS[@]} == ${#CABINETS[@]} * HOSTS_PER_CABINET )) ||
        die "cabinet/host mapping is incomplete"
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
[[ ${REFRESH_MS} =~ ^[1-9][0-9]*$ ]] || die "refresh interval must be positive milliseconds"
[[ ${PROBE_TIMEOUT} =~ ^[0-9]+([.][0-9]+)?$ ]] || die "probe timeout must be positive seconds"
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
    local cabinet frame cabinet_index first_host host_index
    local row state state_without_free
    local free_count

    if (( terminal_output )); then
        printf '\033[H'
    fi
    for cabinet_index in "${!CABINETS[@]}"; do
        cabinet=${CABINETS[cabinet_index]}
        frame=${CABINET_FRAMES[cabinet_index]}
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
        printf '%-7s Cabinet %-2s: %s  freeNum=%d/%d\n' \
            "${frame}" "${cabinet}" "${row% }" "${free_count}" "${NPUS_PER_CABINET}"
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
