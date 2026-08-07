#!/usr/bin/env bash

set -eo pipefail

usage() {
    cat <<EOF
Usage: $0 -m <benchmark|reference|correctness> -r <positive-integer> [options]
       $0 --mode <benchmark|reference|correctness> --rank-size <positive-integer> [options]

Run a TileXR MoonEP case with the requested mode and logical rank count.

Options:
  -m, --mode MODE       benchmark, reference, or correctness
  -r, --rank-size SIZE  Logical rank count; at most two ranks per physical NPU
  -c, --case-id ID      Case to run (default: skewed-padding)
  -v, --visible-devices LIST
                        Comma-separated physical NPU IDs (default: start at 0)
  -d, --dump-stage-tensors
                        Save complete stage inputs/outputs (default for reference/correctness)
      --no-dump-stage-tensors
                        Disable stage tensor snapshots
  -p, --tensor-preview-elements COUNT
                        Values printed per tensor (default: 8)
      --generate-flowcharts
                        Generate six numbered MMD/SVG/PNG flowcharts (default: off)
      --hccl-npu-socket-port-range START-END
                        HCCL device socket ports (default: 47000-47100)
  -h, --help            Show this usage

Available case IDs:
  manual-small              Small tensors for manual inspection (S=2, K=2, E=4, P=1)
  manual-2rank-imbalanced   2-rank Planner load migration from [8,0] to [4,4]
  manual-2rank-dedup-3      2-rank [12,4] to [8,8] migration plus a 3-duplicate group
  planning-small            Balanced Planner case with padded layout (P=4)
  skewed-padding            Default skewed-routing case with padded layout (P=4)

Environment:
  ASCEND_RT_VISIBLE_DEVICES    Legacy fallback when --visible-devices is omitted
  HCCL_NPU_SOCKET_PORT_RANGE   Legacy fallback when the port option is omitted
  TILEXR_INSTALL_PREFIX        TileXR installation prefix
  TILEXR_MOONEP_CONDA_ENV      Conda environment (default: ai_moe_test)
  TILEXR_MOONEP_OUTPUT_DIR     Result directory (default: timestamped /tmp directory)
  TILEXR_MOONEP_TIMEOUT_SEC    Launcher timeout in seconds (default: 600)
  TILEXR_MOONEP_TENSOR_PREVIEW_ELEMENTS
                               Default number of values printed per tensor
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

mode=""
rank_size=""
case_id="skewed-padding"
dump_stage_tensors=""
generate_flowcharts="false"
tensor_preview_elements="${TILEXR_MOONEP_TENSOR_PREVIEW_ELEMENTS:-8}"
visible_device_spec=""
hccl_npu_socket_port_range="47000-47100"
if [[ -n "${HCCL_NPU_SOCKET_PORT_RANGE:-}" ]]; then
    hccl_npu_socket_port_range="${HCCL_NPU_SOCKET_PORT_RANGE}"
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --mode" >&2
                usage >&2
                exit 2
            fi
            mode="$2"
            shift 2
            ;;
        -r|--rank-size)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --rank-size" >&2
                usage >&2
                exit 2
            fi
            rank_size="$2"
            shift 2
            ;;
        -c|--case-id)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --case-id" >&2
                usage >&2
                exit 2
            fi
            case_id="$2"
            shift 2
            ;;
        -v|--visible-devices)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --visible-devices" >&2
                usage >&2
                exit 2
            fi
            visible_device_spec="$2"
            shift 2
            ;;
        -d|--dump-stage-tensors)
            dump_stage_tensors="true"
            shift
            ;;
        --no-dump-stage-tensors)
            dump_stage_tensors="false"
            shift
            ;;
        -p|--tensor-preview-elements)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --tensor-preview-elements" >&2
                usage >&2
                exit 2
            fi
            tensor_preview_elements="$2"
            shift 2
            ;;
        --generate-flowcharts)
            generate_flowcharts="true"
            shift
            ;;
        --hccl-npu-socket-port-range)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --hccl-npu-socket-port-range" >&2
                usage >&2
                exit 2
            fi
            hccl_npu_socket_port_range="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${mode}" || -z "${rank_size}" ]]; then
    echo "Both --mode and --rank-size are required" >&2
    usage >&2
    exit 2
fi

case "${mode}" in
    benchmark|reference|correctness) ;;
    *)
        echo "--mode must be benchmark, reference, or correctness: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac
if [[ ! "${case_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "--case-id contains invalid characters: ${case_id}" >&2
    exit 2
fi
if [[ -z "${dump_stage_tensors}" ]]; then
    if [[ "${mode}" == "benchmark" ]]; then
        dump_stage_tensors="false"
    else
        dump_stage_tensors="true"
    fi
fi
if [[ "${mode}" == "benchmark" && "${dump_stage_tensors}" == "true" ]]; then
    echo "--dump-stage-tensors cannot be used in benchmark mode" >&2
    exit 2
fi
if [[ "${mode}" == "benchmark" && "${generate_flowcharts}" == "true" ]]; then
    echo "--generate-flowcharts cannot be used in benchmark mode" >&2
    exit 2
fi
if [[ "${generate_flowcharts}" == "true" && "${dump_stage_tensors}" != "true" ]]; then
    echo "--generate-flowcharts requires stage tensor snapshots; remove --no-dump-stage-tensors" >&2
    exit 2
fi
if [[ ! "${rank_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--rank-size must be a positive integer: ${rank_size}" >&2
    usage >&2
    exit 2
fi
if [[ ! "${tensor_preview_elements}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--tensor-preview-elements must be a positive integer: ${tensor_preview_elements}" >&2
    exit 2
fi
if [[ ! "${hccl_npu_socket_port_range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    echo "--hccl-npu-socket-port-range must use START-END: ${hccl_npu_socket_port_range}" >&2
    exit 2
fi
hccl_port_start="${BASH_REMATCH[1]}"
hccl_port_end="${BASH_REMATCH[2]}"
if (( 10#${hccl_port_start} < 1024 || 10#${hccl_port_end} > 65535 || 10#${hccl_port_start} > 10#${hccl_port_end} )); then
    echo "HCCL NPU socket ports must satisfy 1024 <= START <= END <= 65535: ${hccl_npu_socket_port_range}" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${script_dir}/common_env.sh"

# MoonEP stages currently use IPC; UDMA initialization interferes with the HCCL reference path.
export TILEXR_ENABLE_UDMA=0

for conda_setup in \
    /home/miniconda3/etc/profile.d/conda.sh \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${conda_setup}" ]]; then
        # Conda's shell function is required for `conda activate`.
        source "${conda_setup}"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found; expected /home/miniconda3 or a conda command on PATH" >&2
    exit 1
fi
conda activate "${TILEXR_MOONEP_CONDA_ENV:-ai_moe_test}"
set -u

if [[ "${generate_flowcharts}" == "true" ]]; then
    if ! command -v mmdc >/dev/null 2>&1; then
        echo "Mermaid CLI (mmdc) was not found; install a Mermaid CLI and browser" >&2
        exit 1
    fi
    mmdc_bin="$(command -v mmdc)"
    mmdc_help="$("${mmdc_bin}" --help 2>&1)"
    if grep -q -- '--puppeteerConfigFile' <<< "${mmdc_help}"; then
        mmdc_config_flag="--puppeteerConfigFile"
        mmdc_background_flag="--backgroundColor"
    elif grep -q -- '--playwright-config-file' <<< "${mmdc_help}"; then
        mmdc_config_flag="--playwright-config-file"
        mmdc_background_flag="--background-color"
    else
        echo "Unsupported Mermaid CLI interface: ${mmdc_bin}" >&2
        exit 1
    fi
    browser_config="${TILEXR_HOME}/tools/moonep/mermaid-browser-config.json"
    if [[ ! -f "${browser_config}" ]]; then
        echo "Mermaid browser config not found: ${browser_config}" >&2
        exit 1
    fi

    mermaid_smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/tilexr-mermaid-smoke.XXXXXX")"
    printf '%s\n' 'flowchart LR' '    A --> B' > "${mermaid_smoke_dir}/smoke.mmd"
    if ! "${mmdc_bin}" \
        "${mmdc_config_flag}" "${browser_config}" \
        --input "${mermaid_smoke_dir}/smoke.mmd" \
        --output "${mermaid_smoke_dir}/mermaid-smoke.svg" \
        "${mmdc_background_flag}" transparent \
        --quiet; then
        rm -rf -- "${mermaid_smoke_dir}"
        echo "Mermaid renderer preflight failed before NPU launch." >&2
        echo "For Playwright CLI run: python -m playwright install chromium" >&2
        echo "For npm CLI install a browser compatible with this host architecture." >&2
        exit 1
    fi
    rm -rf -- "${mermaid_smoke_dir}"
fi

if [[ -n "${visible_device_spec}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="${visible_device_spec}"
fi
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
    for device in "${visible_devices[@]}"; do
        if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
            echo "ASCEND_RT_VISIBLE_DEVICES contains an invalid device: ${device}" >&2
            exit 2
        fi
    done
else
    if [[ ! "${TILEXR_ASCEND_DEV_NUM}" =~ ^[1-9][0-9]*$ ]]; then
        echo "No Ascend devices were detected by common_env.sh" >&2
        exit 1
    fi
    physical_device_count="${rank_size}"
    if (( physical_device_count > TILEXR_ASCEND_DEV_NUM )); then
        physical_device_count="${TILEXR_ASCEND_DEV_NUM}"
    fi
    visible_devices=()
    for ((device = 0; device < physical_device_count; device++)); do
        visible_devices+=("${device}")
    done
    export ASCEND_RT_VISIBLE_DEVICES="$(IFS=,; echo "${visible_devices[*]}")"
fi

physical_device_count="${#visible_devices[@]}"
if (( rank_size <= physical_device_count )); then
    physical_device_count="${rank_size}"
    ranks_per_device=1
elif (( rank_size <= physical_device_count * 2 )); then
    ranks_per_device=2
else
    echo "rank_size ${rank_size} exceeds two ranks per visible NPU (${#visible_devices[@]} NPUs)" >&2
    exit 2
fi

install_prefix="${TILEXR_INSTALL_PREFIX:-${TILEXR_HOME}/install-moonep-b131-20260805}"
timeout_sec="${TILEXR_MOONEP_TIMEOUT_SEC:-600}"
output_dir="${TILEXR_MOONEP_OUTPUT_DIR:-/tmp/tilexr-moonep-${mode}-${rank_size}r-$(date +%Y%m%d-%H%M%S)-$$}"
case_file="${TILEXR_HOME}/tools/moonep/cases/correctness.json"
summary_file="${output_dir}/${case_id}/summary.json"

if [[ ! "${timeout_sec}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TILEXR_MOONEP_TIMEOUT_SEC must be a positive integer: ${timeout_sec}" >&2
    exit 2
fi
for required_file in "${case_file}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -d "${install_prefix}" ]]; then
    echo "TileXR installation prefix not found: ${install_prefix}" >&2
    exit 1
fi

export TILEXR_INSTALL_PREFIX="${install_prefix}"
export LD_LIBRARY_PATH="${install_prefix}/lib64:${install_prefix}/lib:${LD_LIBRARY_PATH:-}"
if [[ "${mode}" != "benchmark" ]]; then
    export HCCL_NPU_SOCKET_PORT_RANGE="${hccl_npu_socket_port_range}"
fi

echo "MoonEP mode: ${mode}"
echo "Case: ${case_id}"
echo "Ranks: ${rank_size} logical, ${physical_device_count} physical"
echo "Devices: ${ASCEND_RT_VISIBLE_DEVICES}"
if [[ "${mode}" != "benchmark" ]]; then
    echo "HCCL NPU socket ports: ${HCCL_NPU_SOCKET_PORT_RANGE}"
fi
echo "Results: ${output_dir}"
echo "Tensor snapshots: ${dump_stage_tensors}"
if [[ "${dump_stage_tensors}" == "true" ]]; then
    echo "Tensor preview elements: ${tensor_preview_elements}"
fi
echo "Flowcharts: ${generate_flowcharts}"

cd "${TILEXR_HOME}"
launcher_args=(
    --mode "${mode}"
    --cases "${case_file}"
    --case-ids "${case_id}"
    --world-size "${rank_size}"
    --physical-device-count "${physical_device_count}"
    --ranks-per-device "${ranks_per_device}"
    --install-prefix "${install_prefix}"
    --output-dir "${output_dir}"
    --timeout-sec "${timeout_sec}"
)
if [[ "${dump_stage_tensors}" == "true" ]]; then
    launcher_args+=("--dump-stage-tensors")
    launcher_args+=("--tensor-preview-elements" "${tensor_preview_elements}")
fi
echo "ASCEND_PROCESS_LOG_PATH: ${ASCEND_PROCESS_LOG_PATH}"
python -m tools.moonep.launcher "${launcher_args[@]}"

if [[ ! -f "${summary_file}" ]]; then
    echo "Summary file not found: ${summary_file}" >&2
    exit 1
fi

if [[ "${generate_flowcharts}" == "true" ]]; then
    flowchart_dir="${output_dir}/${case_id}/flowcharts"
    python -m tools.moonep.flowcharts \
        --case-dir "${output_dir}/${case_id}" \
        --world-size "${rank_size}" \
        --output-dir "${flowchart_dir}"
    expected_flowchart_prefixes=(
        1_planning
        2_dispatch
        3_prefetch-weight
        4_expert-forward
        5_combine
        6_reduce-grad
    )
    for prefix in "${expected_flowchart_prefixes[@]}"; do
        basename="${prefix}-${rank_size}rank-detailed-lr"
        mmd_file="${flowchart_dir}/${basename}.mmd"
        svg_file="${flowchart_dir}/${basename}.svg"
        png_file="${flowchart_dir}/${basename}.png"
        if [[ ! -s "${mmd_file}" ]]; then
            echo "Generated Mermaid source is missing or empty: ${mmd_file}" >&2
            exit 1
        fi
        "${mmdc_bin}" \
            "${mmdc_config_flag}" "${browser_config}" \
            --input "${mmd_file}" \
            --output "${svg_file}" \
            "${mmdc_background_flag}" transparent
        read -r svg_width svg_height < <(
            python - "${svg_file}" <<'PY'
import math
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
view_box = root.attrib["viewBox"].split()
if len(view_box) != 4:
    raise ValueError(f"invalid SVG viewBox: {root.attrib['viewBox']}")
# Mermaid CLI renders inside the browser's default 8px body margin on each edge.
print(
    math.ceil(float(view_box[2])) + 16,
    math.ceil(float(view_box[3])) + 16,
)
PY
        )
        "${mmdc_bin}" \
            "${mmdc_config_flag}" "${browser_config}" \
            --input "${mmd_file}" \
            --output "${png_file}" \
            "${mmdc_background_flag}" white \
            --width "${svg_width}" \
            --height "${svg_height}" \
            --scale 2
        if [[ ! -s "${svg_file}" || ! -s "${png_file}" ]]; then
            echo "Flowchart render is missing or empty: ${basename}" >&2
            exit 1
        fi
    done
    echo "Flowcharts: ${flowchart_dir}"
    echo "Flowchart files: 6 .mmd, 6 .svg, 6 .png"
fi

echo "MoonEP ${mode} passed. Summary: ${summary_file}"
python -m json.tool "${summary_file}"

if [[ "${dump_stage_tensors}" == "true" ]]; then
    tensor_dump_root="${output_dir}/${case_id}"
    preview_file="${tensor_dump_root}/rank_0/tensor_dumps/preview.log"
    if [[ ! -f "${preview_file}" ]]; then
        echo "Tensor preview file not found: ${preview_file}" >&2
        exit 1
    fi
    snapshot_count="$(find "${tensor_dump_root}" -path '*/tensor_dumps/*' -type f -name '*.pt' | wc -l | tr -d ' ')"
    manifest_count="$(find "${tensor_dump_root}" -path '*/tensor_dumps/*' -type f -name '*.json' | wc -l | tr -d ' ')"
    readable_count="$(find "${tensor_dump_root}" -path '*/tensor_dumps/*' -type f -name '*.txt' | wc -l | tr -d ' ')"
    echo "Tensor snapshots: ${tensor_dump_root}"
    echo "Snapshot files: ${snapshot_count} .pt, ${manifest_count} .json"
    echo "Readable files: ${readable_count} .txt"
    echo "Rank 0 preview: ${preview_file}"
fi
