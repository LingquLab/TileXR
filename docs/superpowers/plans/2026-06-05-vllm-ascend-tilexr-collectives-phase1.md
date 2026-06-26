# vllm-ascend TileXR Collectives Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible `ssh blue` validation path proving that vllm-ascend Python can call TileXR `AllGather` on torch-npu NPU tensors through a minimal shim.

**Architecture:** Keep Phase 1 outside the vllm-ascend inference path. Add a small Python `ctypes` shim that loads installed TileXR libraries, calls the TileXR C ABI with torch-npu tensor pointers, and is exercised by a multi-rank smoke runner. Add a remote deployment script that syncs the current TileXR commit to an isolated scratch directory on `blue`, builds collectives, runs standalone collectives checks, then runs the shim smoke.

**Tech Stack:** Bash, CMake, CTest, Python 3, `ctypes`, torch-npu, TileXR C ABI, ACL/CANN runtime.

---

## Scope Check

This plan implements Phase 1 from the approved spec. Phase 2, the opt-in vllm-ascend inference-path backend, depends on Phase 1 results about CANN version, torch-npu stream handles, dtype mapping, and rank lifecycle, so it should receive its own plan after Phase 1 passes on `blue`.

## File Structure

- Create `integrations/vllm_ascend/tilexr_collectives/__init__.py`
  - Package exports for the Phase 1 shim.
- Create `integrations/vllm_ascend/tilexr_collectives/runtime.py`
  - `ctypes` wrapper for `libtile-comm.so` and `libtilexr-collectives.so`, communicator lifecycle, dtype constants, and TileXR error handling.
- Create `integrations/vllm_ascend/tilexr_collectives/torch_collectives.py`
  - torch-npu tensor helpers for `all_gather` and `all_to_all`.
- Create `integrations/vllm_ascend/smoke_collectives.py`
  - Per-rank Python smoke test that allocates NPU tensors, calls the shim, and validates results.
- Create `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
  - Multi-process launcher for the smoke test.
- Create `tests/collectives/deploy_and_run_vllm_remote.sh`
  - Remote deployment and validation entrypoint for `ssh blue`.
- Create `tests/collectives/unit/test_vllm_collectives_integration_sources.py`
  - Source-level regression checks for the shim and deployment scripts.
- Modify `tests/collectives/CMakeLists.txt`
  - Register the new Python source-level test with CTest.
- Modify `tests/collectives/README.md`
  - Document the Phase 1 remote validation command.

## Task 1: Add Source-Level Tests for the Integration Files

**Files:**
- Create: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`
- Modify: `tests/collectives/CMakeLists.txt`

- [ ] **Step 1: Write the failing source test**

Create `tests/collectives/unit/test_vllm_collectives_integration_sources.py`:

```python
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


def test_torch_helpers_require_contiguous_npu_tensors() -> None:
    source = read_rel("integrations/vllm_ascend/tilexr_collectives/torch_collectives.py")
    for token in [
        "def all_gather",
        "def all_to_all",
        "tensor.is_contiguous()",
        "tensor.device.type != \"npu\"",
        "torch.npu.current_stream()",
    ]:
        assert token in source


def test_remote_script_is_isolated_and_logs_environment() -> None:
    source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in [
        "TILEXR_VLLM_REMOTE",
        "TILEXR_VLLM_REMOTE_BASE",
        "rsync -a --delete",
        "--exclude='.worktrees'",
        "npu-smi info",
        "python3 -m pip show torch",
        "run_tilexr_collectives_smoke.sh",
    ]:
        assert token in source
    forbidden_tokens = [
        ">> ~/.bashrc",
        "pip install --user",
        "apt-get install",
        "yum install",
        "rm -rf /usr/local/Ascend",
    ]
    for token in forbidden_tokens:
        assert token not in source


def main() -> None:
    test_vllm_ascend_shim_files_exist()
    test_runtime_uses_tilexr_c_abi_and_not_hccl()
    test_torch_helpers_require_contiguous_npu_tensors()
    test_remote_script_is_isolated_and_logs_environment()
    print("PASS vllm collectives integration source checks")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the source test and verify it fails**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: fails with an assertion mentioning a missing integration file.

- [ ] **Step 3: Register the source test in CMake**

Modify the `if(Python3_Interpreter_FOUND)` block in `tests/collectives/CMakeLists.txt` so it includes both Python tests:

```cmake
if(Python3_Interpreter_FOUND)
    add_test(
        NAME test_collective_profile_report
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_collective_profile_report.py
    )
    add_test(
        NAME test_vllm_collectives_integration_sources
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_vllm_collectives_integration_sources.py
    )
endif()
```

- [ ] **Step 4: Commit**

```bash
git add tests/collectives/unit/test_vllm_collectives_integration_sources.py tests/collectives/CMakeLists.txt
git commit -m "test: add vllm collectives integration source checks"
```

## Task 2: Add the TileXR Python `ctypes` Runtime

**Files:**
- Create: `integrations/vllm_ascend/tilexr_collectives/__init__.py`
- Create: `integrations/vllm_ascend/tilexr_collectives/runtime.py`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Create package exports**

Create `integrations/vllm_ascend/tilexr_collectives/__init__.py`:

```python
from .runtime import (
    TILEXR_DATA_TYPE_BFP16,
    TILEXR_DATA_TYPE_FP16,
    TILEXR_DATA_TYPE_FP32,
    TILEXR_DATA_TYPE_INT8,
    TILEXR_DATA_TYPE_INT32,
    TILEXR_DATA_TYPE_INT64,
    TILEXR_DATA_TYPE_UINT8,
    TileXRCollectivesError,
    TileXRCollectivesRuntime,
)
from .torch_collectives import all_gather, all_to_all

__all__ = [
    "TILEXR_DATA_TYPE_BFP16",
    "TILEXR_DATA_TYPE_FP16",
    "TILEXR_DATA_TYPE_FP32",
    "TILEXR_DATA_TYPE_INT8",
    "TILEXR_DATA_TYPE_INT32",
    "TILEXR_DATA_TYPE_INT64",
    "TILEXR_DATA_TYPE_UINT8",
    "TileXRCollectivesError",
    "TileXRCollectivesRuntime",
    "all_gather",
    "all_to_all",
]
```

- [ ] **Step 2: Create the `ctypes` runtime wrapper**

Create `integrations/vllm_ascend/tilexr_collectives/runtime.py`:

```python
from __future__ import annotations

import ctypes
import os
from pathlib import Path


TILEXR_SUCCESS = 0
TILEXR_DATA_TYPE_INT8 = 0
TILEXR_DATA_TYPE_INT32 = 2
TILEXR_DATA_TYPE_FP16 = 3
TILEXR_DATA_TYPE_FP32 = 4
TILEXR_DATA_TYPE_INT64 = 5
TILEXR_DATA_TYPE_UINT8 = 7
TILEXR_DATA_TYPE_BFP16 = 11


class TileXRCollectivesError(RuntimeError):
    def __init__(self, operation: str, ret: int, detail: str):
        super().__init__(f"{operation} failed with TileXR ret={ret}: {detail}")
        self.operation = operation
        self.ret = ret
        self.detail = detail


def _resolve_install_prefix(install_prefix: str | os.PathLike[str] | None) -> Path:
    if install_prefix is not None:
        return Path(install_prefix).resolve()
    env_prefix = os.environ.get("TILEXR_INSTALL_PREFIX")
    if env_prefix:
        return Path(env_prefix).resolve()
    return Path.cwd().resolve() / "install"


def _resolve_library(name: str, explicit_env: str, install_prefix: Path) -> str:
    explicit = os.environ.get(explicit_env)
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{explicit_env} points to missing library: {path}")
        return str(path)

    candidates = [
        install_prefix / "lib" / name,
        install_prefix / "lib64" / name,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find {name}; checked {candidate_text}")


def _void_p(value: int | ctypes.c_void_p | None) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        return value
    if value is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(int(value))


class TileXRCollectivesRuntime:
    def __init__(
        self,
        rank: int,
        world_size: int,
        install_prefix: str | os.PathLike[str] | None = None,
    ):
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.install_prefix = _resolve_install_prefix(install_prefix)
        self._comm = ctypes.c_void_p()
        self._closed = False

        comm_lib_path = _resolve_library("libtile-comm.so", "TILEXR_COMM_LIB", self.install_prefix)
        collectives_lib_path = _resolve_library(
            "libtilexr-collectives.so",
            "TILEXR_COLLECTIVES_LIB",
            self.install_prefix,
        )
        self._comm_lib = ctypes.CDLL(comm_lib_path, mode=ctypes.RTLD_GLOBAL)
        self._collectives_lib = ctypes.CDLL(collectives_lib_path, mode=ctypes.RTLD_GLOBAL)
        self._configure_symbols()
        ret = self._comm_lib.TileXRCommInitRankLocal(
            ctypes.c_int(self.world_size),
            ctypes.c_int(self.rank),
            ctypes.byref(self._comm),
        )
        self._check("TileXRCommInitRankLocal", ret, f"rank={self.rank} world_size={self.world_size}")

    def _configure_symbols(self) -> None:
        self._comm_lib.TileXRCommInitRankLocal.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._comm_lib.TileXRCommInitRankLocal.restype = ctypes.c_int
        self._comm_lib.TileXRCommDestroy.argtypes = [ctypes.c_void_p]
        self._comm_lib.TileXRCommDestroy.restype = ctypes.c_int

        self._collectives_lib.TileXRAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._collectives_lib.TileXRAllGather.restype = ctypes.c_int
        self._collectives_lib.TileXRAllToAll.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._collectives_lib.TileXRAllToAll.restype = ctypes.c_int

    def _check(self, operation: str, ret: int, detail: str) -> None:
        if int(ret) != TILEXR_SUCCESS:
            raise TileXRCollectivesError(operation, int(ret), detail)

    @property
    def comm_ptr(self) -> int:
        if not self._comm.value:
            raise TileXRCollectivesError("comm_ptr", -1, "TileXR communicator is not initialized")
        return int(self._comm.value)

    def all_gather(
        self,
        send_ptr: int,
        recv_ptr: int,
        send_count: int,
        tilexr_dtype: int,
        stream_ptr: int | None,
    ) -> None:
        ret = self._collectives_lib.TileXRAllGather(
            _void_p(send_ptr),
            _void_p(recv_ptr),
            ctypes.c_int64(int(send_count)),
            ctypes.c_int(int(tilexr_dtype)),
            self._comm,
            _void_p(stream_ptr),
        )
        self._check("TileXRAllGather", ret, f"rank={self.rank} count={send_count} dtype={tilexr_dtype}")

    def all_to_all(
        self,
        send_ptr: int,
        recv_ptr: int,
        send_count_per_peer: int,
        tilexr_dtype: int,
        stream_ptr: int | None,
    ) -> None:
        ret = self._collectives_lib.TileXRAllToAll(
            _void_p(send_ptr),
            _void_p(recv_ptr),
            ctypes.c_int64(int(send_count_per_peer)),
            ctypes.c_int(int(tilexr_dtype)),
            self._comm,
            _void_p(stream_ptr),
        )
        self._check(
            "TileXRAllToAll",
            ret,
            f"rank={self.rank} count_per_peer={send_count_per_peer} dtype={tilexr_dtype}",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._comm.value:
            ret = self._comm_lib.TileXRCommDestroy(self._comm)
            self._comm = ctypes.c_void_p()
            self._check("TileXRCommDestroy", ret, f"rank={self.rank}")

    def __enter__(self) -> "TileXRCollectivesRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
```

- [ ] **Step 3: Run the source test and verify the runtime assertions pass only after torch helpers exist**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: fails because `torch_collectives.py`, `smoke_collectives.py`, `run_tilexr_collectives_smoke.sh`, and `deploy_and_run_vllm_remote.sh` are still missing.

- [ ] **Step 4: Commit**

```bash
git add integrations/vllm_ascend/tilexr_collectives/__init__.py integrations/vllm_ascend/tilexr_collectives/runtime.py
git commit -m "feat: add tilexr collectives ctypes runtime"
```

## Task 3: Add torch-npu Tensor Helpers

**Files:**
- Create: `integrations/vllm_ascend/tilexr_collectives/torch_collectives.py`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Create torch tensor helpers**

Create `integrations/vllm_ascend/tilexr_collectives/torch_collectives.py`:

```python
from __future__ import annotations

from functools import lru_cache

from .runtime import (
    TILEXR_DATA_TYPE_BFP16,
    TILEXR_DATA_TYPE_FP16,
    TILEXR_DATA_TYPE_FP32,
    TILEXR_DATA_TYPE_INT8,
    TILEXR_DATA_TYPE_INT32,
    TILEXR_DATA_TYPE_INT64,
    TileXRCollectivesRuntime,
)


def _torch():
    import torch

    return torch


def _torch_dtype_to_tilexr(dtype) -> int:
    torch = _torch()
    mapping = {
        torch.int8: TILEXR_DATA_TYPE_INT8,
        torch.int32: TILEXR_DATA_TYPE_INT32,
        torch.int64: TILEXR_DATA_TYPE_INT64,
        torch.float16: TILEXR_DATA_TYPE_FP16,
        torch.float32: TILEXR_DATA_TYPE_FP32,
        torch.bfloat16: TILEXR_DATA_TYPE_BFP16,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TileXR collective dtype: {dtype}")
    return mapping[dtype]


def _current_stream_ptr(device_index: int) -> int:
    torch = _torch()
    torch.npu.set_device(device_index)
    stream = torch.npu.current_stream()
    stream_value = getattr(stream, "npu_stream", None)
    if stream_value is None:
        stream_value = getattr(stream, "stream", None)
    if stream_value is None:
        raise RuntimeError("torch.npu.current_stream() does not expose npu_stream or stream")
    return int(stream_value)


def _validate_npu_contiguous(tensor, name: str) -> None:
    if tensor.device.type != "npu":
        raise ValueError(f"{name} must be on an NPU device, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _npu_device_index(tensor) -> int:
    if tensor.device.index is None:
        return int(_torch().npu.current_device())
    return int(tensor.device.index)


def _bind_npu_device(device_index: int) -> None:
    torch = _torch()
    torch.npu.set_device(device_index)


@lru_cache(maxsize=32)
def _cached_runtime(rank: int, world_size: int, install_prefix: str, device_index: int) -> TileXRCollectivesRuntime:
    _bind_npu_device(device_index)
    runtime = TileXRCollectivesRuntime(rank=rank, world_size=world_size, install_prefix=install_prefix)
    runtime.device_index = int(device_index)
    return runtime


def get_runtime(rank: int, world_size: int, install_prefix: str, device_index: int | None = None) -> TileXRCollectivesRuntime:
    if device_index is None:
        device_index = int(_torch().npu.current_device())
    return _cached_runtime(int(rank), int(world_size), str(install_prefix), int(device_index))


def all_gather(tensor, rank: int, world_size: int, install_prefix: str, runtime=None):
    torch = _torch()
    _validate_npu_contiguous(tensor, "tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must contain at least one element")
    device_index = _npu_device_index(tensor)
    _bind_npu_device(device_index)
    rt = runtime or get_runtime(rank, world_size, install_prefix, device_index=device_index)
    output = torch.empty((world_size * tensor.numel(),), dtype=tensor.dtype, device=tensor.device)
    rt.all_gather(
        send_ptr=tensor.data_ptr(),
        recv_ptr=output.data_ptr(),
        send_count=tensor.numel(),
        tilexr_dtype=_torch_dtype_to_tilexr(tensor.dtype),
        stream_ptr=_current_stream_ptr(device_index),
    )
    return output.view(world_size, *tensor.shape)


def all_to_all(tensor, rank: int, world_size: int, install_prefix: str, runtime=None):
    torch = _torch()
    _validate_npu_contiguous(tensor, "tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must contain at least one element")
    if tensor.numel() % world_size != 0:
        raise ValueError(f"tensor.numel()={tensor.numel()} must be divisible by world_size={world_size}")
    device_index = _npu_device_index(tensor)
    _bind_npu_device(device_index)
    rt = runtime or get_runtime(rank, world_size, install_prefix, device_index=device_index)
    output = torch.empty_like(tensor)
    rt.all_to_all(
        send_ptr=tensor.data_ptr(),
        recv_ptr=output.data_ptr(),
        send_count_per_peer=tensor.numel() // world_size,
        tilexr_dtype=_torch_dtype_to_tilexr(tensor.dtype),
        stream_ptr=_current_stream_ptr(device_index),
    )
    return output
```

- [ ] **Step 2: Run the source test and verify remaining missing files**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: fails because the smoke script, launcher script, and remote deployment script are still missing.

- [ ] **Step 3: Commit**

```bash
git add integrations/vllm_ascend/tilexr_collectives/torch_collectives.py
git commit -m "feat: add torch npu tilexr collective helpers"
```

## Task 4: Add the Multi-Rank Python Smoke Test

**Files:**
- Create: `integrations/vllm_ascend/smoke_collectives.py`
- Create: `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Create the per-rank smoke test**

Create `integrations/vllm_ascend/smoke_collectives.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401

from tilexr_collectives import all_gather, all_to_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TileXR vllm-ascend collectives smoke test")
    parser.add_argument("--rank-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--first-npu", type=int, default=0)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--dtype", choices=["int32", "fp16"], default="int32")
    parser.add_argument("--op", choices=["allgather", "alltoall"], default="allgather")
    parser.add_argument("--install-prefix", required=True)
    return parser.parse_args()


def dtype_from_name(name: str):
    if name == "int32":
        return torch.int32
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype name: {name}")


def make_values(rank_size: int, rank: int, count: int, dtype):
    base = rank * 1000
    values = torch.arange(base, base + count, dtype=torch.int32, device=f"npu:{torch.npu.current_device()}")
    if dtype is torch.float16:
        values = values.to(torch.float16)
    return values


def check_all_gather(result, rank_size: int, count: int, dtype) -> None:
    cpu = result.cpu()
    for src in range(rank_size):
        expected = torch.arange(src * 1000, src * 1000 + count, dtype=torch.int32)
        if dtype is torch.float16:
            expected = expected.to(torch.float16)
        actual = cpu[src].reshape(-1)
        if not torch.equal(actual, expected):
            raise AssertionError(f"AllGather mismatch for src={src}: actual={actual} expected={expected}")


def make_all_to_all_values(rank_size: int, rank: int, count: int, dtype):
    host = []
    for dst in range(rank_size):
        for idx in range(count):
            host.append(rank * 100000 + dst * 1000 + idx)
    values = torch.tensor(host, dtype=torch.int32, device=f"npu:{torch.npu.current_device()}")
    if dtype is torch.float16:
        values = values.to(torch.float16)
    return values


def check_all_to_all(result, rank_size: int, rank: int, count: int, dtype) -> None:
    cpu = result.cpu().reshape(-1)
    expected_values = []
    for src in range(rank_size):
        for idx in range(count):
            expected_values.append(src * 100000 + rank * 1000 + idx)
    expected = torch.tensor(expected_values, dtype=torch.int32)
    if dtype is torch.float16:
        expected = expected.to(torch.float16)
    if not torch.equal(cpu, expected):
        raise AssertionError(f"AllToAll mismatch on rank={rank}: actual={cpu} expected={expected}")


def main() -> None:
    args = parse_args()
    if args.rank_size <= 0:
        raise ValueError("--rank-size must be positive")
    if args.rank < 0 or args.rank >= args.rank_size:
        raise ValueError("--rank must be in [0, rank_size)")
    if args.count <= 0:
        raise ValueError("--count must be positive")

    device_id = args.first_npu + args.rank
    torch.npu.set_device(device_id)
    dtype = dtype_from_name(args.dtype)

    if args.op == "allgather":
        send = make_values(args.rank_size, args.rank, args.count, dtype).contiguous()
        result = all_gather(send, rank=args.rank, world_size=args.rank_size, install_prefix=args.install_prefix)
        torch.npu.synchronize()
        check_all_gather(result, args.rank_size, args.count, dtype)
    else:
        send = make_all_to_all_values(args.rank_size, args.rank, args.count, dtype).contiguous()
        result = all_to_all(send, rank=args.rank, world_size=args.rank_size, install_prefix=args.install_prefix)
        torch.npu.synchronize()
        check_all_to_all(result, args.rank_size, args.rank, args.count, dtype)

    print(f"PASS rank={args.rank} op={args.op} dtype={args.dtype} count={args.count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the multi-rank launcher**

Create `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`:

```bash
#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_SIZE="${1:-2}"
COUNT="${2:-16}"
FIRST_NPU="${3:-0}"
INSTALL_PREFIX="${4:?usage: $0 rank_size count first_npu install_prefix op dtype}"
OP="${5:-allgather}"
DTYPE="${6:-int32}"
TIMEOUT_SEC="${TILEXR_VLLM_SMOKE_TIMEOUT_SEC:-600}"

if [[ ! "${RANK_SIZE}" =~ ^[0-9]+$ || "${RANK_SIZE}" -le 0 ]]; then
  echo "ERROR: rank_size must be a positive integer" >&2
  exit 2
fi
if [[ ! "${COUNT}" =~ ^[0-9]+$ || "${COUNT}" -le 0 ]]; then
  echo "ERROR: count must be a positive integer" >&2
  exit 2
fi
if [[ "${OP}" != "allgather" && "${OP}" != "alltoall" ]]; then
  echo "ERROR: op must be allgather or alltoall" >&2
  exit 2
fi
if [[ "${DTYPE}" != "int32" && "${DTYPE}" != "fp16" ]]; then
  echo "ERROR: dtype must be int32 or fp16" >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TILEXR_INSTALL_PREFIX="${INSTALL_PREFIX}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"

pids=()
tail_logs() {
  local rank log
  for ((rank = 0; rank < RANK_SIZE; rank++)); do
    log="tilexr_vllm_collectives_${OP}_${DTYPE}_rank${rank}.log"
    if [[ -f "${log}" ]]; then
      echo "===== ${log} =====" >&2
      tail -n 80 "${log}" >&2
    fi
  done
}

kill_remaining_children() {
  local pid
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 1
  for pid in "${pids[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

for ((rank = 0; rank < RANK_SIZE; rank++)); do
  log="tilexr_vllm_collectives_${OP}_${DTYPE}_rank${rank}.log"
  python3 "${SCRIPT_DIR}/smoke_collectives.py" \
    --rank-size "${RANK_SIZE}" \
    --rank "${rank}" \
    --first-npu "${FIRST_NPU}" \
    --count "${COUNT}" \
    --dtype "${DTYPE}" \
    --op "${OP}" \
    --install-prefix "${INSTALL_PREFIX}" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

sleep "${TIMEOUT_SEC}" >/dev/null 2>&1 &
watchdog_pid="$!"

trap 'echo "ERROR: interrupted; killing remaining ranks" >&2; kill "${watchdog_pid}" 2>/dev/null || true; wait "${watchdog_pid}" 2>/dev/null || true; kill_remaining_children; tail_logs; exit 130' INT TERM

completed_count=0
while (( completed_count < RANK_SIZE )); do
  if wait -n; then
    if ! kill -0 "${watchdog_pid}" 2>/dev/null; then
      echo "ERROR: timed out after ${TIMEOUT_SEC}s" >&2
      kill_remaining_children
      tail_logs
      trap - INT TERM
      exit 124
    fi
    completed_count=$((completed_count + 1))
  else
    rc="$?"
    echo "ERROR: rank process exited with status ${rc}" >&2
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
    kill_remaining_children
    tail_logs
    trap - INT TERM
    exit 1
  fi
done

kill "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true
trap - INT TERM
echo "PASS TileXR vllm collectives smoke rank_size=${RANK_SIZE} op=${OP} dtype=${DTYPE}"
```

- [ ] **Step 3: Make the launcher executable**

Run:

```bash
chmod +x integrations/vllm_ascend/run_tilexr_collectives_smoke.sh
```

- [ ] **Step 4: Run the source test and verify only remote script is missing**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: fails because `tests/collectives/deploy_and_run_vllm_remote.sh` is still missing.

- [ ] **Step 5: Commit**

```bash
git add integrations/vllm_ascend/smoke_collectives.py integrations/vllm_ascend/run_tilexr_collectives_smoke.sh
git commit -m "feat: add vllm collectives smoke runner"
```

## Task 5: Add the `ssh blue` Remote Validation Script

**Files:**
- Create: `tests/collectives/deploy_and_run_vllm_remote.sh`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Create the remote deployment script**

Create `tests/collectives/deploy_and_run_vllm_remote.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILEXR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE="${TILEXR_VLLM_REMOTE:?set TILEXR_VLLM_REMOTE to the SSH target for remote vllm collectives validation}"
REMOTE_BASE="${TILEXR_VLLM_REMOTE_BASE:?set TILEXR_VLLM_REMOTE_BASE to a scratch directory on the remote host}"
REMOTE_REPO="${REMOTE_BASE}/TileXR"
REMOTE_LOG="${REMOTE_BASE}/logs/deploy.log"
REMOTE_ASCEND_HOME_PATH="${TILEXR_VLLM_REMOTE_ASCEND_HOME_PATH:-}"
REMOTE_ASCEND_DRIVER_PATH="${TILEXR_VLLM_REMOTE_ASCEND_DRIVER_PATH:-}"
REMOTE_CMAKE_CCE_COMPILER="${TILEXR_VLLM_REMOTE_CMAKE_CCE_COMPILER:-}"

branch="$(git -C "${TILEXR_ROOT}" rev-parse --abbrev-ref HEAD)"
commit="$(git -C "${TILEXR_ROOT}" rev-parse HEAD)"
staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/tilexr_vllm_collectives.XXXXXX")"
trap 'rm -rf "${staging_dir}"' EXIT
staging_repo="${staging_dir}/TileXR"

echo "Deploying TileXR vllm collectives validation"
echo "  remote: ${REMOTE}"
echo "  remote base: ${REMOTE_BASE}"
echo "  branch: ${branch}"
echo "  commit: ${commit}"

git clone --no-hardlinks --no-checkout "${TILEXR_ROOT}" "${staging_repo}"
git -C "${staging_repo}" checkout --detach "${commit}"

sync_local_submodule() {
  local rel_path="$1"
  local src="${TILEXR_ROOT}/${rel_path}"
  local dst="${staging_repo}/${rel_path}"
  if [[ ! -d "${src}" ]]; then
    echo "ERROR: required local submodule is missing: ${rel_path}" >&2
    echo "Initialize local submodules before running this script." >&2
    exit 1
  fi
  mkdir -p "${dst}"
  rsync -a --delete --exclude='.git' "${src}/" "${dst}/"
}

sync_local_submodule "3rdparty/hcomm"
sync_local_submodule "3rdparty/ops-transformer"
sync_local_submodule "3rdparty/spdlog"

remote_prepare=$(cat <<EOF
set -euo pipefail
remote_base=$(printf '%q' "${REMOTE_BASE}")
remote_repo=$(printf '%q' "${REMOTE_REPO}")
case "\${remote_repo}" in
  "\${remote_base}"/TileXR)
    rm -rf -- "\${remote_repo}"
    mkdir -p -- "\${remote_repo}"
    mkdir -p -- "\${remote_base}/logs" "\${remote_base}/artifacts"
    ;;
  *)
    echo "Refusing to clean unexpected remote repo: \${remote_repo}" >&2
    exit 2
    ;;
esac
EOF
)

ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_prepare}")"

rsync -a --delete \
  --exclude='.worktrees' \
  --exclude='build' \
  --exclude='build_*' \
  --exclude='build-*' \
  --exclude='install' \
  --exclude='run' \
  --exclude='env/temp' \
  "${staging_repo}/" "${REMOTE}:${REMOTE_REPO}/"

remote_script=$(cat <<EOF
set -euo pipefail
remote_ascend_home_path=$(printf '%q' "${REMOTE_ASCEND_HOME_PATH}")
remote_ascend_driver_path=$(printf '%q' "${REMOTE_ASCEND_DRIVER_PATH}")
remote_cmake_cce_compiler=$(printf '%q' "${REMOTE_CMAKE_CCE_COMPILER}")
cd $(printf '%q' "${REMOTE_REPO}")
{
  echo "Remote branch source: ${branch}"
  echo "Remote commit source: ${commit}"
  echo "Remote host: \$(hostname)"
  echo "Remote user: \$(whoami)"
  command -v npu-smi || true
  npu-smi info || true
  python3 --version || true
  cmake --version 2>/dev/null | sed -n '1p' || true
  gcc --version 2>/dev/null | sed -n '1p' || true
  g++ --version 2>/dev/null | sed -n '1p' || true
  python3 -m pip show torch || true
  python3 -m pip show torch-npu || true
  python3 -m pip show vllm || true
  python3 -m pip show vllm-ascend || true
  git submodule status --recursive || true
  if [[ -n "\${remote_ascend_home_path}" ]]; then
    export ASCEND_HOME_PATH="\${remote_ascend_home_path}"
    if [[ -f "\${ASCEND_HOME_PATH}/set_env.sh" ]]; then
      set +u
      source "\${ASCEND_HOME_PATH}/set_env.sh"
      set -u
    fi
  fi
  if [[ -n "\${remote_ascend_driver_path}" ]]; then
    export ASCEND_DRIVER_PATH="\${remote_ascend_driver_path}"
  fi
  if [[ -n "\${remote_cmake_cce_compiler}" ]]; then
    export CMAKE_CCE_COMPILER="\${remote_cmake_cce_compiler}"
  fi
  set +u
  source scripts/common_env.sh
  set -u
  cmake_args=(
    -DCMAKE_INSTALL_PREFIX=install
    -DTILEXR_BUILD_COLLECTIVES=ON
    -DTILEXR_BUILD_TESTS=ON
    -DBUILD_TESTING=ON
  )
  if [[ -n "\${remote_cmake_cce_compiler}" ]]; then
    cmake_args+=("-DCMAKE_CCE_COMPILER=\${remote_cmake_cce_compiler}")
  fi
  if [[ -n "\${remote_ascend_driver_path}" ]]; then
    cmake_args+=("-DASCEND_DRIVER_PATH=\${remote_ascend_driver_path}")
  fi
  cmake -S . -B build-vllm-collectives \
    "\${cmake_args[@]}"
  cmake --build build-vllm-collectives --target install test_tilexr_collectives_correctness tilexr_collective_perf -j"\$(nproc)"
  ctest --test-dir build-vllm-collectives --output-on-failure
  (
    cd tests/collectives
    TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=600 ./run_collectives_correctness.sh 2 16 0 ../../build-vllm-collectives/tests/collectives allgather
  )
  (
    cd integrations/vllm_ascend
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather int32
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather fp16
  )
} 2>&1 | tee $(printf '%q' "${REMOTE_LOG}")
EOF
)

ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_script}")"

echo "Remote validation log: ${REMOTE}:${REMOTE_LOG}"
```

- [ ] **Step 2: Make the script executable**

Run:

```bash
chmod +x tests/collectives/deploy_and_run_vllm_remote.sh
```

- [ ] **Step 3: Run the source test and verify it passes**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: exits with status 0.

- [ ] **Step 4: Commit**

```bash
git add tests/collectives/deploy_and_run_vllm_remote.sh
git commit -m "feat: add remote vllm collectives validation"
```

## Task 6: Document the Phase 1 Command

**Files:**
- Modify: `tests/collectives/README.md`

- [ ] **Step 1: Add a vllm-ascend smoke section**

Append this section to `tests/collectives/README.md`:

````markdown
## vllm-ascend Shim Smoke on `blue`

The Phase 1 vllm-ascend integration smoke keeps all remote state under a scratch directory and does not modify
the remote system Python or shell startup files.

```bash
TILEXR_VLLM_REMOTE=blue \
TILEXR_VLLM_REMOTE_BASE=/path/to/remote/tilexr_vllm_collectives_$(date +%Y%m%d_%H%M%S) \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

The script syncs the current TileXR commit and initialized local submodules, dumps the NPU/CANN/Python environment,
builds `tile-comm` and `tilexr-collectives`, runs the standalone 2-rank AllGather correctness check, and runs
the Python torch-npu shim AllGather smoke for `int32` and `fp16`.

Expected success lines include:

```text
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=fp16
```

If Python packages such as `torch`, `torch-npu`, `vllm`, or `vllm-ascend` are missing on the remote host, the
environment dump records that fact before the shim smoke runs. Install those packages inside the selected scratch
environment before using the shim for vllm-ascend inference-path work.
````

- [ ] **Step 2: Commit**

```bash
git add tests/collectives/README.md
git commit -m "docs: document vllm collectives smoke"
```

## Task 7: Local Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run the source-level Python test**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: exits with status 0.

- [ ] **Step 2: Configure collectives tests**

Run:

```bash
bash -lc 'source scripts/common_env.sh && cmake -S . -B build-vllm-plan-check -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DBUILD_TESTING=ON'
```

Expected in a fully configured CANN environment: CMake configuration succeeds.

Expected in the current local environment without CANN headers: CMake may configure but later build fails on missing CANN headers. Record the missing header error and continue to remote verification.

- [ ] **Step 3: Run CTest source checks if configuration succeeds**

Run:

```bash
ctest --test-dir build-vllm-plan-check -R 'test_vllm_collectives_integration_sources|test_tilexr_collectives_tools_sources' --output-on-failure
```

Expected: selected source checks pass when the build directory exists.

- [ ] **Step 4: Commit any verification-only fixes**

If verification required changes, commit them:

```bash
git add integrations/vllm_ascend tests/collectives
git commit -m "fix: stabilize vllm collectives source checks"
```

If no changes were needed, do not create a commit.

## Task 8: Remote Verification on `ssh blue`

**Files:**
- Verify only.

- [ ] **Step 1: Run the remote validation script**

Run:

```bash
TILEXR_VLLM_REMOTE=blue \
TILEXR_VLLM_REMOTE_BASE=/path/to/remote/tilexr_vllm_collectives_$(date +%Y%m%d_%H%M%S) \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

Expected on a complete remote environment: the script prints a `Remote validation log:` line and both shim smoke commands print `PASS`.

Expected on the currently observed `blue` environment: the environment dump shows missing `torch`, `torch-npu`, `vllm`, and `vllm-ascend`. If the TileXR build passes but shim import fails due missing Python packages, record the failure as an environment gap, not a TileXR collectives failure.

- [ ] **Step 2: Preserve the remote log path**

Copy the final `Remote validation log:` line into the task notes. The path format is:

```text
blue:/path/to/remote/tilexr_vllm_collectives_YYYYMMDD_HHMMSS/logs/deploy.log
```

- [ ] **Step 3: Commit remote-script fixes only if needed**

If the remote run exposes script bugs, fix them and commit:

```bash
git add tests/collectives/deploy_and_run_vllm_remote.sh integrations/vllm_ascend
git commit -m "fix: harden remote vllm collectives validation"
```

If the only failure is missing remote Python packages, do not change code.
