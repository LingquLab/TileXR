from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "scripts" / "run_moonep.sh").read_text(encoding="utf-8")


def test_script_exposes_tensor_dump_controls() -> None:
    assert "-d|--dump-stage-tensors" in SCRIPT
    assert "--no-dump-stage-tensors" in SCRIPT
    assert "-p|--tensor-preview-elements" in SCRIPT
    assert "TILEXR_MOONEP_TENSOR_PREVIEW_ELEMENTS" in SCRIPT


def test_reference_and_correctness_default_to_tensor_dumps() -> None:
    assert 'if [[ -z "${dump_stage_tensors}" ]]' in SCRIPT
    assert 'if [[ "${mode}" == "benchmark" ]]' in SCRIPT
    assert 'dump_stage_tensors="false"' in SCRIPT
    assert 'dump_stage_tensors="true"' in SCRIPT


def test_script_forwards_dump_arguments_and_reports_artifacts() -> None:
    assert 'launcher_args+=("--dump-stage-tensors")' in SCRIPT
    assert 'launcher_args+=("--tensor-preview-elements" "${tensor_preview_elements}")' in SCRIPT
    assert 'echo "Tensor snapshots: ${tensor_dump_root}"' in SCRIPT
    assert 'echo "Snapshot files: ${snapshot_count} .pt, ${manifest_count} .json"' in SCRIPT


def test_script_accepts_visible_devices_and_defaults_from_device_zero() -> None:
    assert "-v|--visible-devices" in SCRIPT
    assert 'visible_device_spec="$2"' in SCRIPT
    assert 'if [[ -n "${visible_device_spec}" ]]' in SCRIPT
    assert 'for ((device = 0; device < physical_device_count; device++))' in SCRIPT
    assert 'export ASCEND_RT_VISIBLE_DEVICES="$(IFS=,; echo "${visible_devices[*]}")"' in SCRIPT
    assert "first_device=" not in SCRIPT


def test_script_owns_and_validates_hccl_npu_port_range() -> None:
    assert "--hccl-npu-socket-port-range" in SCRIPT
    assert 'hccl_npu_socket_port_range="47000-47100"' in SCRIPT
    assert 'hccl_npu_socket_port_range="$2"' in SCRIPT
    assert 'export HCCL_NPU_SOCKET_PORT_RANGE="${hccl_npu_socket_port_range}"' in SCRIPT
    assert 'echo "HCCL NPU socket ports: ${HCCL_NPU_SOCKET_PORT_RANGE}"' in SCRIPT


def test_script_prints_final_plog_path_before_python_launch() -> None:
    log_path_report = SCRIPT.index(
        'echo "ASCEND_PROCESS_LOG_PATH: ${ASCEND_PROCESS_LOG_PATH}"'
    )
    launcher_call = SCRIPT.index("python -m tools.moonep.launcher")
    assert log_path_report < launcher_call


def test_script_disables_tilexr_udma_only_for_single_rank() -> None:
    single_rank_override = '''if [[ "${rank_size}" == "1" ]]; then
    export TILEXR_ENABLE_UDMA=0
fi'''
    assert single_rank_override in SCRIPT
    assert SCRIPT.count("export TILEXR_ENABLE_UDMA=") == 1
    assert SCRIPT.index(single_rank_override) < SCRIPT.index(
        "python -m tools.moonep.launcher"
    )


def test_script_selects_manual_small_through_case_id_only() -> None:
    assert "-c|--case-id" in SCRIPT
    assert 'case_id="$2"' in SCRIPT
    assert 'case_id="planning-no-dedup"' in SCRIPT
    assert "--manual-small" not in SCRIPT
    assert '--case-ids "${case_id}"' in SCRIPT
    assert 'summary_file="${output_dir}/${case_id}/summary.json"' in SCRIPT
    assert 'tensor_dump_root="${output_dir}/${case_id}"' in SCRIPT
    assert 'echo "Readable files: ${readable_count} .txt"' in SCRIPT


def test_usage_lists_every_available_case_id_with_a_description() -> None:
    case_path = ROOT / "tools" / "moonep" / "cases" / "correctness.json"
    raw_cases = json.loads(case_path.read_text(encoding="utf-8"))
    case_ids = [item["id"] for item in raw_cases]
    usage_cases = SCRIPT.split("Available case IDs:", 1)[1].split(
        "\n\nEnvironment:", 1
    )[0]
    for case_id in case_ids:
        assert re.search(rf"^  {re.escape(case_id)}\s+\S.+$", usage_cases, re.MULTILINE)


def test_script_exposes_opt_in_flowchart_export() -> None:
    assert "--generate-flowcharts" in SCRIPT
    assert 'generate_flowcharts="false"' in SCRIPT
    assert 'generate_flowcharts="true"' in SCRIPT
    assert 'flowchart_dir="${output_dir}/${case_id}/flowcharts"' in SCRIPT


def test_flowcharts_require_reference_data_and_skip_benchmark_timing() -> None:
    assert 'if [[ "${mode}" == "benchmark" && "${generate_flowcharts}" == "true" ]]' in SCRIPT
    assert "--generate-flowcharts cannot be used in benchmark mode" in SCRIPT
    assert 'if [[ "${generate_flowcharts}" == "true" && "${dump_stage_tensors}" != "true" ]]' in SCRIPT
    assert "--generate-flowcharts requires stage tensor snapshots" in SCRIPT


def test_script_checks_mermaid_before_launch_and_renders_all_formats() -> None:
    dependency_check = SCRIPT.index('command -v mmdc')
    launcher_call = SCRIPT.index('python -m tools.moonep.launcher')
    conda_activation = SCRIPT.index('conda activate')
    assert conda_activation < dependency_check
    assert dependency_check < launcher_call
    assert 'mmdc_bin="$(command -v mmdc)"' in SCRIPT
    assert '"${mmdc_bin}" --help' in SCRIPT
    assert "--puppeteerConfigFile" in SCRIPT
    assert "--playwright-config-file" in SCRIPT
    assert 'browser_config="${TILEXR_HOME}/tools/moonep/mermaid-browser-config.json"' in SCRIPT
    assert 'mermaid_smoke_dir="$(mktemp -d' in SCRIPT
    assert "flowchart LR" in SCRIPT
    smoke_render = SCRIPT.index('mermaid-smoke.svg')
    assert dependency_check < smoke_render < launcher_call
    assert "python -m playwright install chromium" in SCRIPT
    assert SCRIPT.count('"${mmdc_bin}" \\') == 3
    assert 'python -m tools.moonep.flowcharts' in SCRIPT
    assert '"${mmdc_config_flag}" "${browser_config}"' in SCRIPT
    assert 'read -r svg_width svg_height' in SCRIPT
    assert 'root.attrib["viewBox"]' in SCRIPT
    assert "math.ceil(float(view_box[2])) + 16" in SCRIPT
    assert "math.ceil(float(view_box[3])) + 16" in SCRIPT
    assert '--width "${svg_width}"' in SCRIPT
    assert '--height "${svg_height}"' in SCRIPT
    assert '--scale 2' in SCRIPT
    assert 'expected_flowchart_prefixes=(' in SCRIPT
    for prefix in (
        "1_planning",
        "2_dispatch",
        "3_prefetch-weight",
        "4_expert-forward",
        "5_combine",
        "6_reduce-grad",
    ):
        assert prefix in SCRIPT
