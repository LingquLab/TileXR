#!/usr/bin/env python3
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
        "integrations/vllm_ascend/smoke_collectives.py",
        "integrations/vllm_ascend/run_tilexr_collectives_smoke.sh",
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
        "def should_fallback(",
    ]:
        assert token in source
    assert "import vllm" not in source
    assert "from vllm" not in source


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


def test_remote_script_supports_selected_python_environment() -> None:
    source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in [
        "TILEXR_VLLM_REMOTE_PYTHON",
        "TILEXR_VLLM_REMOTE_CONDA_ENV",
        "TILEXR_VLLM_REMOTE_CONDA_SH",
        "select_remote_python",
        "selected_python=",
        "dump_selected_python_environment",
        "run_selected_python_preflight",
        "torch.npu.current_stream()",
        "npu_stream",
        "TILEXR_VLLM_SMOKE_PYTHON=\"\\${selected_python}\"",
    ]:
        assert token in source
    assert "python3 -m pip show torch" not in source
    assert "python3 -m pip show torch-npu" not in source


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


def main() -> None:
    test_vllm_ascend_shim_files_exist()
    test_runtime_uses_tilexr_c_abi_and_not_hccl()
    test_runtime_exposes_phase3_collective_c_api()
    test_torch_helpers_require_contiguous_npu_tensors()
    test_torch_helpers_expose_vllm_compatible_collectives()
    test_vllm_adapter_is_opt_in_and_import_lightweight()
    test_remote_script_is_isolated_and_logs_environment()
    test_remote_script_supports_selected_python_environment()
    test_smoke_launcher_supports_python_override()
    print("PASS vllm collectives integration source checks")


if __name__ == "__main__":
    main()
