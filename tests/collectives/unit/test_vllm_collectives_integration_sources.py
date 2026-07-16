#!/usr/bin/env python3
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read_rel(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def assert_exists(path: str) -> None:
    full_path = ROOT / path
    assert full_path.exists(), f"missing required integration file: {path}"


def test_vllm_ascend_shim_files_exist() -> None:
    for path in [
        "integrations/vllm_ascend/tilexr_collectives/__init__.py",
        "integrations/vllm_ascend/tilexr_collectives/runtime.py",
        "integrations/vllm_ascend/tilexr_collectives/torch_collectives.py",
        "integrations/vllm_ascend/tilexr_collectives/vllm_patch.py",
        "integrations/vllm_ascend/smoke_collectives.py",
        "integrations/vllm_ascend/run_tilexr_collectives_smoke.sh",
        "integrations/vllm_ascend/sitecustomize.py",
        "tests/collectives/deploy_and_run_vllm_remote.sh",
    ]:
        assert_exists(path)


def test_runtime_uses_tilexr_c_abi_and_not_hccl() -> None:
    runtime_source = read_rel("integrations/vllm_ascend/tilexr_collectives/runtime.py")
    required_tokens = [
        "TileXRCommInitRankLocal",
        "TileXRCommDestroy",
        "TileXRAllGather",
        "TileXRAllToAll",
        "TILEXR_DATA_TYPE_INT32",
        "TILEXR_DATA_TYPE_FP16",
    ]
    for token in required_tokens:
        assert token in runtime_source
    assert "hccl" not in runtime_source.lower()


def test_runtime_exposes_phase3_collective_c_api() -> None:
    runtime_source = read_rel("integrations/vllm_ascend/tilexr_collectives/runtime.py")
    for token in [
        "TileXRAllReduce",
        "TileXRReduceScatter",
        "TileXRBroadcast",
        "TILEXR_REDUCE_SUM",
        "def all_reduce(",
        "def reduce_scatter(",
        "def broadcast(",
    ]:
        assert token in runtime_source


def test_torch_helpers_require_contiguous_npu_tensors() -> None:
    source = read_rel("integrations/vllm_ascend/tilexr_collectives/torch_collectives.py")
    for token in [
        "def all_gather",
        "def all_to_all",
        "tensor.is_contiguous()",
        "tensor.device.type != \"npu\"",
        "torch.npu.current_stream()",
        "torch.npu.set_device(device_index)",
    ]:
        assert token in source
    assert "torch.uint8" not in source
    assert "TILEXR_DATA_TYPE_UINT8" not in source


def test_torch_helpers_expose_vllm_compatible_collectives() -> None:
    source = read_rel("integrations/vllm_ascend/tilexr_collectives/torch_collectives.py")
    for token in [
        "def _move_dim_to_front(",
        "def _restore_dim_from_front(",
        "def all_gather(tensor, rank: int, world_size: int, install_prefix: str, dim: int = -1",
        "def all_reduce(tensor, rank: int, world_size: int, install_prefix: str",
        "def reduce_scatter(tensor, rank: int, world_size: int, install_prefix: str, dim: int = -1",
        "def broadcast(tensor, rank: int, world_size: int, install_prefix: str, root: int = 0",
    ]:
        assert token in source


def test_vllm_adapter_is_opt_in_and_import_lightweight() -> None:
    source = read_rel("integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py")
    for token in [
        "VLLM_ASCEND_TILEXR_COLLECTIVES",
        "class TileXRVllmCollectivesAdapter",
        "def enabled(",
        "def all_reduce(",
        "def all_gather(",
        "def reduce_scatter(",
        "def all_to_all(",
        "def broadcast(",
        "root=src",
        "def should_fallback(",
    ]:
        assert token in source
    assert "import vllm" not in source
    assert "from vllm" not in source


def test_vllm_adapter_normalizes_all_to_all_dims_before_fallback() -> None:
    source = read_rel("integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py")
    assert "_same_normalized_dim(" in source
    assert "if not self._same_normalized_dim(input_, scatter_dim, gather_dim):" in source
    assert "if scatter_dim != gather_dim:" not in source


def test_vllm_patch_hooks_npu_communicator_methods() -> None:
    source = read_rel("integrations/vllm_ascend/tilexr_collectives/vllm_patch.py")
    for token in [
        "def patch_npu_communicator(",
        "VLLM_ASCEND_TILEXR_COLLECTIVES",
        "TILEXR_VLLM_TRACE",
        "NPUCommunicator",
        "_tilexr_collectives_adapter",
        "_tilexr_collectives_route_counts",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "broadcast",
        "return original(self",
    ]:
        assert token in source


def test_sitecustomize_auto_patches_only_when_enabled() -> None:
    source = read_rel("integrations/vllm_ascend/sitecustomize.py")
    for token in [
        "VLLM_ASCEND_TILEXR_COLLECTIVES",
        "patch_npu_communicator",
        "sitecustomize",
    ]:
        assert token in source
    assert "raise" not in source


def test_remote_script_is_isolated_and_logs_environment() -> None:
    source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in [
        "TILEXR_VLLM_REMOTE",
        "TILEXR_VLLM_REMOTE_BASE",
        "TILEXR_VLLM_REMOTE_ASCEND_HOME_PATH",
        "TILEXR_VLLM_REMOTE_ASCEND_DRIVER_PATH",
        "TILEXR_VLLM_REMOTE_CMAKE_CCE_COMPILER",
        "rsync -a --delete",
        "sync_local_submodule",
        "--exclude='.worktrees'",
        "npu-smi info",
        "run_tilexr_collectives_smoke.sh",
    ]:
        assert token in source
    forbidden_tokens = [
        "TILEXR_VLLM_REMOTE:-",
        "TILEXR_VLLM_REMOTE_BASE:-",
        "REMOTE_BASE=/",
        "REMOTE_BASE=\"/",
        "submodule update --init --recursive",
        ">> ~/.bashrc",
        "pip install --user",
        "apt-get install",
        "yum install",
        "rm -rf /usr/local/Ascend",
    ]
    for token in forbidden_tokens:
        assert token not in source
    assert "3rdparty/hcomm" not in source


def test_remote_script_supports_selected_python_environment() -> None:
    source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in [
        "TILEXR_VLLM_REMOTE_PYTHON",
        "TILEXR_VLLM_REMOTE_CONDA_ENV",
        "TILEXR_VLLM_REMOTE_CONDA_SH",
        "TILEXR_VLLM_REMOTE_PYTHONPATH",
        "select_remote_python",
        "selected_python=",
        "remote_pythonpath",
        "dump_selected_python_environment",
        "run_selected_python_preflight",
        "torch.npu.current_stream()",
        "npu_stream",
        "TILEXR_VLLM_SMOKE_PYTHON=\"\\${selected_python}\"",
        "TILEXR_VLLM_SMOKE_PYTHONPATH=\"\\${remote_pythonpath}\"",
    ]:
        assert token in source
    assert "python3 -m pip show torch" not in source
    assert "python3 -m pip show torch-npu" not in source


def test_remote_script_can_pass_consistent_ssh_options_to_ssh_and_rsync() -> None:
    source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in [
        "TILEXR_VLLM_REMOTE_SSH_OPTS",
        "ssh_args=(",
        "rsync_ssh_cmd=(",
        "ssh \"${ssh_args[@]}\"",
        "rsync -e \"${rsync_ssh_cmd[*]}\"",
    ]:
        assert token in source


def test_remote_script_can_probe_vllm_source_trees_without_crashing() -> None:
    source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in [
        "TILEXR_VLLM_REMOTE_VLLM_SOURCE",
        "TILEXR_VLLM_REMOTE_VLLM_ASCEND_SOURCE",
        "TILEXR_VLLM_REMOTE_DUMMY_MODEL",
        "TILEXR_VLLM_REMOTE_VLLM_PLUGINS",
        "TILEXR_VLLM_REMOTE_NNAL_ATB_SET_ENV",
        "build_vllm_probe_pythonpath",
        "run_vllm_import_probe",
        "subprocess.run",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "vllm.distributed.device_communicators.base_device_communicator",
        "vllm_ascend.distributed.device_communicators.npu_communicator",
        "probe_vllm_environment \"pre-cann\"",
        "probe_vllm_environment \"post-cann\"",
        "local probe_label=\"\\${1:?probe label required}\"",
        "pythonpath_entries+=(\"\\${remote_pythonpath}\")",
        "probe_vllm_communicator_patch",
        "TileXR vllm NPUCommunicator patch probe",
        "TileXR vllm NPUCommunicator route counts",
        "patch_npu_communicator(\"install\")",
        "probe_vllm_dummy_inference",
        "load_format=\"dummy\"",
        "skip_tokenizer_init=True",
        "PASS TileXR vllm dummy inference probe",
        "source \"\\${remote_nnal_atb_set_env}\"",
    ]:
        assert token in source
    assert "local probe_label=\"${1:?probe label required}\"" not in source


def test_smoke_launcher_supports_python_override() -> None:
    source = read_rel("integrations/vllm_ascend/run_tilexr_collectives_smoke.sh")
    for token in [
        "PYTHON_BIN=\"${TILEXR_VLLM_SMOKE_PYTHON:-python3}\"",
        "command -v \"${PYTHON_BIN}\"",
        "import sys",
        "ERROR: Python command failed interpreter preflight",
        "TileXR vllm collectives smoke",
        "\"${PYTHON_BIN}\" \"${SCRIPT_DIR}/smoke_collectives.py\"",
    ]:
        assert token in source
    assert "python3 \"${SCRIPT_DIR}/smoke_collectives.py\"" not in source


def test_smoke_launcher_accepts_preflight_sentinel_with_plugin_stdout(tmp_path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  if [[ "${2:-}" == *"tilexr_python_preflight_ok"* ]]; then
    echo "INFO plugin stdout before sentinel"
    echo "tilexr_python_preflight_ok"
    exit 0
  fi
  if [[ "${2:-}" == *"sys.executable"* ]]; then
    printf '%s\\n' "$0"
    exit 0
  fi
fi
if [[ "${1:-}" == *"smoke_collectives.py" ]]; then
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    launcher = ROOT / "integrations/vllm_ascend/run_tilexr_collectives_smoke.sh"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "2",
            "16",
            "0",
            str(tmp_path / "install"),
            "allgather",
            "int32",
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "TILEXR_VLLM_SMOKE_PYTHON": str(fake_python),
            "TILEXR_VLLM_SMOKE_TIMEOUT_SEC": "5",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS TileXR vllm collectives smoke" in result.stdout


def test_phase3_smoke_covers_core_collectives() -> None:
    smoke_source = read_rel("integrations/vllm_ascend/smoke_collectives.py")
    launcher_source = read_rel("integrations/vllm_ascend/run_tilexr_collectives_smoke.sh")
    remote_source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in ["allreduce", "reducescatter", "broadcast"]:
        assert token in smoke_source
        assert token in launcher_source
        assert token in remote_source
    for token in ["probe_vllm_environment", "vllm_ascend", "VLLM_ASCEND_TILEXR_COLLECTIVES"]:
        assert token in remote_source


def test_phase3_docs_describe_feature_flag_and_boundaries() -> None:
    readme = read_rel("tests/collectives/README.md")
    for token in [
        "VLLM_ASCEND_TILEXR_COLLECTIVES=1",
        "TILEXR_VLLM_REMOTE_VLLM_SOURCE",
        "TILEXR_VLLM_REMOTE_VLLM_ASCEND_SOURCE",
        "TILEXR_VLLM_REMOTE_DUMMY_MODEL",
        "TILEXR_VLLM_REMOTE_NNAL_ATB_SET_ENV",
        "TILEXR_VLLM_REMOTE_PYTHONPATH",
        "tilexr-vllm29",
        "torch==2.9.0",
        "torch-npu==2.9.0",
        "triton-ascend==3.2.0",
        "vllm_ascend.distributed.device_communicators.npu_communicator",
        "NPUCommunicator patch probe",
        "TileXR vllm NPUCommunicator route counts",
        "TileXR vllm dummy inference probe",
        "NPUCommunicator patched",
        "load_format=\"dummy\"",
        "libatb.so",
        "torchvision",
        "rc=0",
        "allreduce",
        "reducescatter",
        "broadcast",
        "fallback",
        "libhccl.so",
        "zmq",
        "vllm-ascend inference",
    ]:
        assert token in readme


def main() -> None:
    test_vllm_ascend_shim_files_exist()
    test_runtime_uses_tilexr_c_abi_and_not_hccl()
    test_runtime_exposes_phase3_collective_c_api()
    test_torch_helpers_require_contiguous_npu_tensors()
    test_torch_helpers_expose_vllm_compatible_collectives()
    test_vllm_adapter_is_opt_in_and_import_lightweight()
    test_vllm_adapter_normalizes_all_to_all_dims_before_fallback()
    test_vllm_patch_hooks_npu_communicator_methods()
    test_sitecustomize_auto_patches_only_when_enabled()
    test_remote_script_is_isolated_and_logs_environment()
    test_remote_script_supports_selected_python_environment()
    test_remote_script_can_probe_vllm_source_trees_without_crashing()
    test_smoke_launcher_supports_python_override()
    test_phase3_smoke_covers_core_collectives()
    test_phase3_docs_describe_feature_flag_and_boundaries()
    print("PASS vllm collectives integration source checks")


if __name__ == "__main__":
    main()
