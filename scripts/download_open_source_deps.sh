#!/usr/bin/env bash
set -eo pipefail

script_path=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source "${script_path}/common_env.sh"
set -u

check_only=0
force=0

usage() {
    cat <<'EOF'
Usage: bash scripts/download_open_source_deps.sh [--check] [--force]

Download third-party source archives into 3rdparty/open_source/.

Options:
  --check   Verify that all archives are present and match their SHA256.
  --force   Redownload archives even when a matching file already exists.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            check_only=1
            ;;
        --force)
            force=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum is required" >&2
    exit 1
fi

downloader=()
if command -v curl >/dev/null 2>&1; then
    downloader=(curl -fL --retry 3 --connect-timeout 20 -o)
elif command -v wget >/dev/null 2>&1; then
    downloader=(wget --tries=3 --timeout=20 -O)
elif [[ ${check_only} -eq 0 ]]; then
    echo "curl or wget is required to download archives" >&2
    exit 1
fi

mkdir -p "${TILEXR_3RD_OPEN_HOME}"

deps=(
    "ccache-4.12.2-linux-aarch64.tar.xz|b01c270c245e41998ab777164aba085dbeb23ce515f4e2134a1fdddabf0bf6ad|https://github.com/ccache/ccache/releases/download/v4.12.2/ccache-4.12.2-linux-aarch64.tar.xz"
    "ccache-4.12.2-linux-x86_64.tar.xz|630c34ec94d451b200f5b14a6a25580d6a45bc80c394b7e0b93e33556eee5d32|https://github.com/ccache/ccache/releases/download/v4.12.2/ccache-4.12.2-linux-x86_64.tar.xz"
    "cmake-3.22.6.tar.gz|73933163670ea4ea95c231549007b0c7243282293506a2cf4443714826ad5ec3|https://github.com/Kitware/CMake/releases/download/v3.22.6/cmake-3.22.6.tar.gz"
    "mpich-4.3.1.tar.gz|acc11cb2bdc69678dc8bba747c24a28233c58596f81f03785bf2b7bb7a0ef7dc|https://www.mpich.org/static/downloads/4.3.1/mpich-4.3.1.tar.gz"
    "patch-2.8.tar.gz|308a4983ff324521b9b21310bfc2398ca861798f02307c79eb99bb0e0d2bf980|https://ftp.gnu.org/gnu/patch/patch-2.8.tar.gz"
    "pigz-2.8.tar.gz|eb872b4f0e1f0ebe59c9f7bd8c506c4204893ba6a8492de31df416f0d5170fd0|https://zlib.net/pigz/pigz-2.8.tar.gz"
    "ripgrep-15.1.0-aarch64-unknown-linux-gnu.tar.gz|2b661c6ef508e902f388e9098d9c4c5aca72c87b55922d94abdba830b4dc885e|https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-aarch64-unknown-linux-gnu.tar.gz"
    "ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz|1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599|https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz"
    "sshpass-1.06.tar.gz|c6324fcee608b99a58f9870157dfa754837f8c48be3df0f5e2f3accf145dee60|https://sourceforge.net/projects/sshpass/files/sshpass/1.06/sshpass-1.06.tar.gz/download"
    "time-1.9.tar.gz|fbacf0c81e62429df3e33bda4cee38756604f18e01d977338e23306a3e3b521e|https://ftp.gnu.org/gnu/time/time-1.9.tar.gz"
)

verify_archive() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]]
}

failures=0

for dep in "${deps[@]}"; do
    IFS='|' read -r file_name sha256 url <<< "${dep}"
    target="${TILEXR_3RD_OPEN_HOME}/${file_name}"

    if [[ -f "${target}" && ${force} -eq 0 ]]; then
        if verify_archive "${target}" "${sha256}"; then
            echo "OK ${file_name}"
            continue
        fi
        echo "checksum mismatch: ${file_name}" >&2
        if [[ ${check_only} -eq 1 ]]; then
            failures=$((failures + 1))
            continue
        fi
    elif [[ ${check_only} -eq 1 ]]; then
        echo "missing: ${file_name}" >&2
        failures=$((failures + 1))
        continue
    fi

    tmp="${target}.tmp.$$"
    rm -f "${tmp}"
    echo "Downloading ${file_name}"
    "${downloader[@]}" "${tmp}" "${url}"

    if ! verify_archive "${tmp}" "${sha256}"; then
        rm -f "${tmp}"
        echo "downloaded archive failed checksum: ${file_name}" >&2
        failures=$((failures + 1))
        continue
    fi

    mv "${tmp}" "${target}"
    echo "OK ${file_name}"
done

if [[ ${failures} -ne 0 ]]; then
    echo "${failures} open-source dependency archive checks failed" >&2
    exit 1
fi
