#!/usr/bin/env bash

set -euo pipefail

MAX_ARTIFACT_BYTES=$((100 * 1024 * 1024))

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 SOURCE_DIR ARTIFACT_DIR" >&2
    exit 2
fi

SOURCE_DIR="$1"
ARTIFACT_DIR="$2"
if [[ ! -d "${SOURCE_DIR}" || -L "${SOURCE_DIR}" ]]; then
    echo "ERROR: SOURCE_DIR must be an existing real directory: ${SOURCE_DIR}" >&2
    exit 2
fi
if [[ -L "${ARTIFACT_DIR}" ]]; then
    echo "ERROR: ARTIFACT_DIR must not be a symbolic link: ${ARTIFACT_DIR}" >&2
    exit 2
fi

SOURCE_DIR="$(cd "${SOURCE_DIR}" && pwd -P)"
mkdir -p "${ARTIFACT_DIR}"
ARTIFACT_DIR="$(cd "${ARTIFACT_DIR}" && pwd -P)"
if [[ "${SOURCE_DIR}" == "/" || "${ARTIFACT_DIR}" == "/" || \
      "${SOURCE_DIR}" == "${ARTIFACT_DIR}" ]]; then
    echo "ERROR: unsafe source or artifact directory" >&2
    exit 2
fi
if [[ "${SOURCE_DIR}" == "${ARTIFACT_DIR}/"* ]]; then
    echo "ERROR: ARTIFACT_DIR must not contain SOURCE_DIR" >&2
    exit 2
fi

EVIDENCE_ERROR=""

COLLECTED_ROOT="${ARTIFACT_DIR}/source"
case "${COLLECTED_ROOT}" in
    "${ARTIFACT_DIR}"/*) ;;
    *)
        echo "ERROR: unsafe collected artifact path" >&2
        exit 2
        ;;
esac
/bin/rm -rf "${COLLECTED_ROOT}"
mkdir -p "${COLLECTED_ROOT}"

REJECTION_TMP="$(mktemp "${TMPDIR:-/tmp}/tilexr-artifact-rejections.XXXXXX")"
MANIFEST_TMP="$(mktemp "${TMPDIR:-/tmp}/tilexr-artifact-manifest.XXXXXX")"
CANDIDATES_TMP="$(mktemp "${TMPDIR:-/tmp}/tilexr-artifact-candidates.XXXXXX")"
LINKS_TMP="$(mktemp "${TMPDIR:-/tmp}/tilexr-artifact-links.XXXXXX")"
FILES_TMP="$(mktemp "${TMPDIR:-/tmp}/tilexr-artifact-files.XXXXXX")"
ENVIRONMENT_TMP="$(mktemp "${TMPDIR:-/tmp}/tilexr-artifact-environment.XXXXXX")"
cleanup_temporary_files() {
    /bin/rm -f "${REJECTION_TMP}" "${MANIFEST_TMP}" "${CANDIDATES_TMP}" \
        "${LINKS_TMP}" "${FILES_TMP}" "${ENVIRONMENT_TMP}"
}
trap cleanup_temporary_files EXIT

reject_artifact() {
    printf '%s\t%s\n' "$1" "$2" >> "${REJECTION_TMP}"
}

record_evidence_error() {
    local message="$1"
    if [[ -z "${EVIDENCE_ERROR}" ]]; then
        EVIDENCE_ERROR="${message}"
    else
        EVIDENCE_ERROR="${EVIDENCE_ERROR}; ${message}"
    fi
}

check_root_evidence() {
    local name="$1"
    local path="${ARTIFACT_DIR}/${name}"
    local size
    if [[ -L "${path}" ]]; then
        record_evidence_error "required root evidence is a symbolic link: ${name}"
        reject_artifact "${name}" "unsafe-required-evidence"
        /bin/rm -f "${path}"
        return 0
    fi
    if [[ ! -e "${path}" ]]; then
        record_evidence_error "required root evidence is missing: ${name}"
        reject_artifact "${name}" "missing-required-evidence"
        return 0
    fi
    if [[ ! -f "${path}" ]]; then
        record_evidence_error "required root evidence is not a regular file: ${name}"
        reject_artifact "${name}" "unsafe-required-evidence"
        /bin/rm -rf "${path}"
        return 0
    fi
    if ! size="$(wc -c < "${path}" | tr -d '[:space:]')"; then
        record_evidence_error "required root evidence is unreadable: ${name}"
        reject_artifact "${name}" "unreadable-required-evidence"
        /bin/rm -f "${path}"
        return 0
    fi
    if [[ ! "${size}" =~ ^[0-9]+$ || "${size}" -gt "${MAX_ARTIFACT_BYTES}" ]]; then
        record_evidence_error "required root evidence exceeds 100 MiB: ${name}"
        reject_artifact "${name}" "oversized-required-evidence"
        /bin/rm -f "${path}"
    fi
}

check_root_evidence "cases.tsv"
check_root_evidence "summary.md"

is_allowed_artifact() {
    local relative="$1"
    local basename="${relative##*/}"
    case "${relative}" in
        *$'\n'*|*$'\t'*) return 1 ;;
    esac
    case "${basename}" in
        *.log|*.xml|summary.csv|trace.json|cases.tsv|summary.md|environment.txt|\
        ldd-*.txt|readelf-*.txt|dependencies-*.txt|npu-state-*.txt|npu-topology.txt|version-*.txt)
            return 0
            ;;
    esac
    return 1
}

copy_candidate() {
    local path="$1"
    local relative destination size
    if [[ ! -f "${path}" || -L "${path}" ]]; then
        return 0
    fi
    relative="${path#"${SOURCE_DIR}/"}"
    if [[ "${relative}" == "${path}" || "${relative}" == /* || "${relative}" == ../* ]]; then
        reject_artifact "${relative}" "path-outside-source"
        return 0
    fi
    if ! is_allowed_artifact "${relative}"; then
        return 0
    fi
    size="$(wc -c < "${path}" | tr -d '[:space:]')"
    if [[ ! "${size}" =~ ^[0-9]+$ || "${size}" -gt "${MAX_ARTIFACT_BYTES}" ]]; then
        reject_artifact "source/${relative}" "larger-than-100-MiB"
        return 0
    fi
    destination="${COLLECTED_ROOT}/${relative}"
    mkdir -p "$(dirname "${destination}")"
    /bin/cp -pP "${path}" "${destination}"
    if [[ -L "${destination}" || ! -f "${destination}" ]]; then
        echo "ERROR: unsafe artifact copy result: ${destination}" >&2
        return 1
    fi
}

if [[ "${ARTIFACT_DIR}" == "${SOURCE_DIR}/"* ]]; then
    /usr/bin/find -P "${SOURCE_DIR}" -path "${ARTIFACT_DIR}" -prune -o \
        -type f -print0 > "${CANDIDATES_TMP}"
else
    /usr/bin/find -P "${SOURCE_DIR}" -type f -print0 > "${CANDIDATES_TMP}"
fi
while IFS= read -r -d '' candidate; do
    copy_candidate "${candidate}"
done < "${CANDIDATES_TMP}"

# The untrusted phase can write directly into ARTIFACT_DIR. Sanitize everything
# that Actions will upload, including pre-existing files, before manifesting it.
/usr/bin/find -P "${ARTIFACT_DIR}" -type l -print0 > "${LINKS_TMP}"
while IFS= read -r -d '' link; do
    relative="${link#"${ARTIFACT_DIR}/"}"
    reject_artifact "${relative}" "symbolic-link"
    /bin/rm -f "${link}"
done < "${LINKS_TMP}"

/usr/bin/find -P "${ARTIFACT_DIR}" -type f -print0 > "${FILES_TMP}"
while IFS= read -r -d '' file; do
    relative="${file#"${ARTIFACT_DIR}/"}"
    if [[ "${relative}" == "manifest.txt" || "${relative}" == "artifact-rejections.txt" ]]; then
        continue
    fi
    if ! is_allowed_artifact "${relative}"; then
        reject_artifact "${relative}" "unsupported-artifact-type"
        /bin/rm -f "${file}"
        continue
    fi
    size="$(wc -c < "${file}" | tr -d '[:space:]')"
    if [[ ! "${size}" =~ ^[0-9]+$ || "${size}" -gt "${MAX_ARTIFACT_BYTES}" ]]; then
        reject_artifact "${relative}" "larger-than-100-MiB"
        /bin/rm -f "${file}"
    fi
done < "${FILES_TMP}"

{
    echo "TileXR CI environment"
    echo "source=${SOURCE_DIR}"
    echo "artifacts=${ARTIFACT_DIR}"
    echo "TILEXR_CANN_HOME=${TILEXR_CANN_HOME:-/home/tilexr-ci/toolchains/cann/9.1.0}"
    echo "uname=$(uname -a 2>&1 || true)"
    if command -v cmake >/dev/null 2>&1; then
        cmake --version 2>&1 | head -n 1 || true
    fi
    if command -v c++ >/dev/null 2>&1; then
        c++ --version 2>&1 | head -n 1 || true
    fi
    SEALED_BISHENG="${TILEXR_CANN_HOME:-/home/tilexr-ci/toolchains/cann/9.1.0}/cann/compiler/ccec_compiler/bin/bisheng"
    if [[ -x "${SEALED_BISHENG}" ]]; then
        "${SEALED_BISHENG}" -v 2>&1 | head -n 1 || true
    elif command -v bisheng >/dev/null 2>&1; then
        bisheng -v 2>&1 | head -n 1 || true
    fi
    if command -v mpirun >/dev/null 2>&1; then
        mpirun --version 2>&1 | head -n 1 || true
    fi
    if command -v npu-smi >/dev/null 2>&1; then
        if [[ -x /usr/bin/timeout ]]; then
            /usr/bin/timeout --signal=TERM --kill-after=2 10 npu-smi info 2>&1 || true
            /usr/bin/timeout --signal=TERM --kill-after=2 10 npu-smi info -l 2>&1 || true
        else
            npu-smi info 2>&1 || true
            npu-smi info -l 2>&1 || true
        fi
    fi
    if command -v git >/dev/null 2>&1; then
        echo "commit=$(git -C "${SOURCE_DIR}" rev-parse HEAD 2>/dev/null || true)"
    fi
} > "${ENVIRONMENT_TMP}"
if [[ -L "${ARTIFACT_DIR}/environment.txt" || -f "${ARTIFACT_DIR}/environment.txt" ]]; then
    /bin/rm -f "${ARTIFACT_DIR}/environment.txt"
elif [[ -e "${ARTIFACT_DIR}/environment.txt" ]]; then
    reject_artifact "environment.txt" "unsafe-environment-destination"
    /bin/rm -rf "${ARTIFACT_DIR}/environment.txt"
fi
/bin/mv -f "${ENVIRONMENT_TMP}" "${ARTIFACT_DIR}/environment.txt"

if [[ -s "${REJECTION_TMP}" ]]; then
    LC_ALL=C sort -u "${REJECTION_TMP}" > "${ARTIFACT_DIR}/artifact-rejections.txt"
else
    /bin/rm -f "${ARTIFACT_DIR}/artifact-rejections.txt"
fi

/usr/bin/find -P "${ARTIFACT_DIR}" -type f -print0 > "${FILES_TMP}"
while IFS= read -r -d '' file; do
    relative="${file#"${ARTIFACT_DIR}/"}"
    if [[ "${relative}" == "manifest.txt" || "${relative}" == *$'\n'* || "${relative}" == *$'\t'* ]]; then
        continue
    fi
    size="$(wc -c < "${file}" | tr -d '[:space:]')"
    printf '%s\t%s\n' "${relative}" "${size}" >> "${MANIFEST_TMP}"
done < "${FILES_TMP}"
LC_ALL=C sort "${MANIFEST_TMP}" > "${ARTIFACT_DIR}/manifest.txt"

if [[ -n "${EVIDENCE_ERROR}" ]]; then
    echo "ERROR: ${EVIDENCE_ERROR}" >&2
    exit 1
fi
