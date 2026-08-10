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


def test_script_reports_oversubscribed_reference_collective() -> None:
    assert 'if [[ "${mode}" == "correctness" ]] || (( ranks_per_device == 2 )); then' in SCRIPT
    assert 'echo "Reference collective: Gloo (CPU staging)"' in SCRIPT
    assert 'echo "Reference collective: HCCL"' in SCRIPT


def test_script_discovers_the_shared_anaconda_installation() -> None:
    system_miniconda = "/home/miniconda3/etc/profile.d/conda.sh"
    system_anaconda = "/home/anaconda3/etc/profile.d/conda.sh"
    home_miniconda = '"${HOME}/miniconda3/etc/profile.d/conda.sh"'

    assert system_anaconda in SCRIPT
    assert SCRIPT.index(system_miniconda) < SCRIPT.index(system_anaconda)
    assert SCRIPT.index(system_anaconda) < SCRIPT.index(home_miniconda)


def test_default_output_directory_is_under_project_run_moonep() -> None:
    assert (
        'output_dir="${TILEXR_MOONEP_OUTPUT_DIR:-${TILEXR_HOME}/run/moonep/'
        'tilexr-moonep-${mode}-${rank_size}r-$(date +%Y%m%d-%H%M%S)-$$}"'
        in SCRIPT
    )
    assert "/tmp/tilexr-moonep-${mode}" not in SCRIPT


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
    for number, case_id in enumerate(case_ids, start=1):
        assert re.search(
            rf"^  {number}\s+{re.escape(case_id)}\s+\S.+$",
            usage_cases,
            re.MULTILINE,
        )


def test_usage_describes_only_fixed_padding_non_duplicate_cases() -> None:
    usage_cases = SCRIPT.split("Available case IDs:", 1)[1].split(
        "\n\nEnvironment:", 1
    )[0]
    descriptions = [line for line in usage_cases.splitlines() if line.strip()]
    assert descriptions
    assert all("P=1" in line for line in descriptions)
    assert "dedup-3" not in usage_cases
    assert "duplicate group" not in usage_cases
    assert "P=4" not in usage_cases


def test_usage_describes_rank_size_and_rank_per_device_for_every_case() -> None:
    usage_cases = SCRIPT.split("Available case IDs:", 1)[1].split(
        "\n\nEnvironment:", 1
    )[0]
    descriptions = [line for line in usage_cases.splitlines() if line.strip()]
    assert len(descriptions) == 14
    assert all("rank_size=" in line for line in descriptions)
    assert all("rank_per_dev=" in line for line in descriptions)


def test_usage_describes_eight_and_sixteen_rank_cases() -> None:
    assert re.search(
        r"^  8\s+planning-8rank-topk-8\s+8-rank .+P=1\)$",
        SCRIPT,
        re.MULTILINE,
    )
    assert re.search(
        r"^  9\s+planning-16rank-topk-16\s+16-rank, 8 NPUs x 2 ranks .+P=1\)$",
        SCRIPT,
        re.MULTILINE,
    )
    assert re.search(
        r"^  10\s+planning-8rank-single-route\s+8-rank single route .+P=1\)$",
        SCRIPT,
        re.MULTILINE,
    )
    assert re.search(
        r"^  11\s+planning-16rank-single-route\s+16-rank, 8 NPUs x 2 ranks single route .+P=1\)$",
        SCRIPT,
        re.MULTILINE,
    )
    assert re.search(
        r"^  12\s+planning-64rank-single-route\s+64-rank, 8 nodes x 8 NPUs single route .+P=1\)$",
        SCRIPT,
        re.MULTILINE,
    )
    assert re.search(
        r"^  13\s+planning-128rank-single-route\s+128-rank, 16 nodes x 8 NPUs single route .+P=1\)$",
        SCRIPT,
        re.MULTILINE,
    )
    assert re.search(
        r"^  14\s+planning-16rank-16card-single-route\s+16-rank, 2 nodes x 8 NPUs single route .+rank_size=16, rank_per_dev=1.+P=1\)$",
        SCRIPT,
        re.MULTILINE,
    )


def test_script_exposes_managed_multinode_three_mode_launch() -> None:
    for option in ("--node-count", "--node-rank", "--master-addr", "--master-port"):
        assert option in SCRIPT
    assert 'if (( node_count > 1 )); then' in SCRIPT
    assert 'python -m tools.moonep.distributed_node' in SCRIPT
    assert 'distributed_args+=("--dump-stage-tensors")' in SCRIPT
    assert '--mode "${mode}"' in SCRIPT
    assert '--benchmark-kind flow' in SCRIPT
    assert "multi-node runs require --mode reference" not in SCRIPT
    assert "multi-node runs require exactly one rank per visible NPU" in SCRIPT


def test_script_resolves_numeric_case_before_constructing_artifact_paths() -> None:
    numeric_check = SCRIPT.index('if [[ "${case_id}" =~ ^[0-9]+$ ]]')
    resolver = SCRIPT.index("select_cases(load_cases(sys.argv[1]), sys.argv[2])")
    canonical_assignment = SCRIPT.index('case_id="${resolved_case_id}"')
    summary_path = SCRIPT.index('summary_file="${output_dir}/${case_id}/summary.json"')
    assert numeric_check < resolver < canonical_assignment < summary_path


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
