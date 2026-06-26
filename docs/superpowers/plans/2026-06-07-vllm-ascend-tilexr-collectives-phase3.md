# vllm-ascend TileXR Collectives Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect TileXR collectives to the vllm/vllm-ascend group collective path from the rewritten `main` branch with an opt-in adapter, expanded Python collective coverage, remote validation, documentation, and a PR.

**Architecture:** Keep TileXR host/kernel code unchanged on `main`; extend the existing Python `ctypes` shim to cover the standalone collectives C API and add a vllm-style adapter that can be called from vllm-ascend `NPUCommunicator`. The TileXR path is selected only with `VLLM_ASCEND_TILEXR_COLLECTIVES=1`; unsupported shapes, operations, or missing libraries return to the existing vllm-ascend/HCCL path.

**Tech Stack:** Bash, Python 3, `ctypes`, pytest/source guards, torch-npu, vllm `DeviceCommunicatorBase` semantics, TileXR C ABI, CMake/CTest, `ssh blue`.

---

## File Structure

- Modify `integrations/vllm_ascend/tilexr_collectives/runtime.py`
  - Configure `TileXRAllReduce`, `TileXRReduceScatter`, and `TileXRBroadcast` symbols.
  - Add runtime methods for all-reduce SUM, reduce-scatter SUM, and broadcast.
- Modify `integrations/vllm_ascend/tilexr_collectives/torch_collectives.py`
  - Add tensor-shape helpers for vllm-compatible `dim` handling.
  - Add `all_reduce`, `reduce_scatter`, and `broadcast`.
  - Update `all_gather` and `all_to_all` signatures for vllm-compatible dimensions.
- Modify `integrations/vllm_ascend/tilexr_collectives/__init__.py`
  - Export the new helpers and reduce-op constant.
- Create `integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py`
  - Opt-in adapter mirroring vllm device communicator methods.
- Modify `integrations/vllm_ascend/smoke_collectives.py`
  - Add deterministic checks for all-reduce, reduce-scatter, and broadcast.
- Modify `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
  - Allow `allreduce`, `reducescatter`, and `broadcast` smoke operations.
- Modify `tests/collectives/deploy_and_run_vllm_remote.sh`
  - Run expanded shim smoke on `blue`.
  - Add explicit vllm/vllm-ascend environment probe.
- Modify `tests/collectives/unit/test_vllm_collectives_integration_sources.py`
  - Source-level guards for the expanded runtime, adapter, launcher, and docs.
- Modify `tests/collectives/README.md`
  - Document Phase 3 feature flag, coverage, fallback boundaries, and remote validation.

## Task 1: Guard Phase 3 Runtime and Adapter Surfaces

**Files:**
- Modify: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Write failing source guards**

Add this test to `tests/collectives/unit/test_vllm_collectives_integration_sources.py`:

```python
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
```

Add this test:

```python
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
```

Add this test:

```python
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
```

Update `main()` in the same file to call the three new tests.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: failure mentioning missing Phase 3 runtime tokens and missing `vllm_adapter.py`.

- [ ] **Step 3: Commit tests**

```bash
git add tests/collectives/unit/test_vllm_collectives_integration_sources.py
git commit -m "test: guard vllm phase3 collectives surfaces"
```

## Task 2: Extend the ctypes Runtime

**Files:**
- Modify: `integrations/vllm_ascend/tilexr_collectives/runtime.py`
- Modify: `integrations/vllm_ascend/tilexr_collectives/__init__.py`

- [ ] **Step 1: Configure new symbols**

In `runtime.py`, add:

```python
TILEXR_REDUCE_SUM = 0
```

Inside `_configure_symbols`, add:

```python
self._collectives_lib.TileXRAllReduce.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
self._collectives_lib.TileXRAllReduce.restype = ctypes.c_int
self._collectives_lib.TileXRReduceScatter.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
self._collectives_lib.TileXRReduceScatter.restype = ctypes.c_int
self._collectives_lib.TileXRBroadcast.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
self._collectives_lib.TileXRBroadcast.restype = ctypes.c_int
```

- [ ] **Step 2: Add runtime methods**

Add to `TileXRCollectivesRuntime`:

```python
def all_reduce(
    self,
    send_ptr: int,
    recv_ptr: int,
    count: int,
    tilexr_dtype: int,
    stream_ptr: int | None,
) -> None:
    ret = self._collectives_lib.TileXRAllReduce(
        _void_p(send_ptr),
        _void_p(recv_ptr),
        ctypes.c_int64(int(count)),
        ctypes.c_int(int(tilexr_dtype)),
        ctypes.c_int(TILEXR_REDUCE_SUM),
        self._comm,
        _void_p(stream_ptr),
    )
    self._check("TileXRAllReduce", ret, f"rank={self.rank} count={count} dtype={tilexr_dtype}")
```

Add:

```python
def reduce_scatter(
    self,
    send_ptr: int,
    recv_ptr: int,
    recv_count: int,
    tilexr_dtype: int,
    stream_ptr: int | None,
) -> None:
    ret = self._collectives_lib.TileXRReduceScatter(
        _void_p(send_ptr),
        _void_p(recv_ptr),
        ctypes.c_int64(int(recv_count)),
        ctypes.c_int(int(tilexr_dtype)),
        ctypes.c_int(TILEXR_REDUCE_SUM),
        self._comm,
        _void_p(stream_ptr),
    )
    self._check("TileXRReduceScatter", ret, f"rank={self.rank} recv_count={recv_count} dtype={tilexr_dtype}")
```

Add:

```python
def broadcast(
    self,
    buf_ptr: int,
    count: int,
    tilexr_dtype: int,
    root: int,
    stream_ptr: int | None,
) -> None:
    ret = self._collectives_lib.TileXRBroadcast(
        _void_p(buf_ptr),
        ctypes.c_int64(int(count)),
        ctypes.c_int(int(tilexr_dtype)),
        ctypes.c_int(int(root)),
        self._comm,
        _void_p(stream_ptr),
    )
    self._check("TileXRBroadcast", ret, f"rank={self.rank} count={count} dtype={tilexr_dtype} root={root}")
```

- [ ] **Step 3: Export reduce-op constant**

In `__init__.py`, import and add `TILEXR_REDUCE_SUM` to `__all__`.

- [ ] **Step 4: Verify GREEN for runtime guards**

Run:

```bash
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m py_compile integrations/vllm_ascend/tilexr_collectives/runtime.py integrations/vllm_ascend/tilexr_collectives/__init__.py
```

Expected: adapter guard still fails until Task 4, but runtime tokens are present and py_compile passes.

- [ ] **Step 5: Commit runtime expansion**

```bash
git add integrations/vllm_ascend/tilexr_collectives/runtime.py integrations/vllm_ascend/tilexr_collectives/__init__.py
git commit -m "feat: expose phase3 tilexr collective runtime"
```

## Task 3: Add vllm-Compatible Torch Helpers

**Files:**
- Modify: `integrations/vllm_ascend/tilexr_collectives/torch_collectives.py`
- Modify: `integrations/vllm_ascend/tilexr_collectives/__init__.py`

- [ ] **Step 1: Add dimension helpers**

In `torch_collectives.py`, add:

```python
def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"invalid dim={dim} for tensor with ndim={ndim}")
    return dim


def _move_dim_to_front(tensor, dim: int):
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous(), dim
    return tensor.movedim(dim, 0).contiguous(), dim


def _restore_dim_from_front(tensor, dim: int):
    if dim == 0:
        return tensor.contiguous()
    return tensor.movedim(0, dim).contiguous()
```

- [ ] **Step 2: Update `all_gather`**

Change the signature to include `dim: int = -1`. Use `_move_dim_to_front`, call runtime with the moved tensor, reshape to `(world_size, *moved.shape)`, flatten the first two axes, then restore the original dimension:

```python
def all_gather(tensor, rank: int, world_size: int, install_prefix: str, dim: int = -1, runtime=None):
    torch = _torch()
    _validate_npu_contiguous(tensor, "tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must contain at least one element")
    moved, normalized_dim = _move_dim_to_front(tensor, dim)
    device_index = _npu_device_index(moved)
    _bind_npu_device(device_index)
    rt = _select_runtime(runtime, rank, world_size, install_prefix, device_index)
    output = torch.empty((world_size * moved.numel(),), dtype=moved.dtype, device=moved.device)
    rt.all_gather(
        send_ptr=moved.data_ptr(),
        recv_ptr=output.data_ptr(),
        send_count=moved.numel(),
        tilexr_dtype=_torch_dtype_to_tilexr(moved.dtype),
        stream_ptr=_current_stream_ptr(device_index),
    )
    gathered = output.view(world_size, *moved.shape).reshape(world_size * moved.shape[0], *moved.shape[1:])
    return _restore_dim_from_front(gathered, normalized_dim)
```

- [ ] **Step 3: Add `all_reduce`**

```python
def all_reduce(tensor, rank: int, world_size: int, install_prefix: str, runtime=None):
    torch = _torch()
    _validate_npu_contiguous(tensor, "tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must contain at least one element")
    device_index = _npu_device_index(tensor)
    _bind_npu_device(device_index)
    rt = _select_runtime(runtime, rank, world_size, install_prefix, device_index)
    output = torch.empty_like(tensor)
    rt.all_reduce(
        send_ptr=tensor.data_ptr(),
        recv_ptr=output.data_ptr(),
        count=tensor.numel(),
        tilexr_dtype=_torch_dtype_to_tilexr(tensor.dtype),
        stream_ptr=_current_stream_ptr(device_index),
    )
    return output
```

- [ ] **Step 4: Add `reduce_scatter`**

```python
def reduce_scatter(tensor, rank: int, world_size: int, install_prefix: str, dim: int = -1, runtime=None):
    torch = _torch()
    _validate_npu_contiguous(tensor, "tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must contain at least one element")
    moved, normalized_dim = _move_dim_to_front(tensor, dim)
    if moved.shape[0] % world_size != 0:
        raise ValueError(f"tensor.shape[{normalized_dim}]={tensor.shape[normalized_dim]} must be divisible by world_size={world_size}")
    device_index = _npu_device_index(moved)
    _bind_npu_device(device_index)
    rt = _select_runtime(runtime, rank, world_size, install_prefix, device_index)
    output_shape = (moved.shape[0] // world_size, *moved.shape[1:])
    output = torch.empty(output_shape, dtype=moved.dtype, device=moved.device)
    rt.reduce_scatter(
        send_ptr=moved.data_ptr(),
        recv_ptr=output.data_ptr(),
        recv_count=output.numel(),
        tilexr_dtype=_torch_dtype_to_tilexr(moved.dtype),
        stream_ptr=_current_stream_ptr(device_index),
    )
    return _restore_dim_from_front(output, normalized_dim)
```

- [ ] **Step 5: Add `broadcast`**

```python
def broadcast(tensor, rank: int, world_size: int, install_prefix: str, root: int = 0, runtime=None):
    _validate_npu_contiguous(tensor, "tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must contain at least one element")
    if root < 0 or root >= world_size:
        raise ValueError(f"root must be in [0, {world_size}), got {root}")
    device_index = _npu_device_index(tensor)
    _bind_npu_device(device_index)
    rt = _select_runtime(runtime, rank, world_size, install_prefix, device_index)
    output = tensor.contiguous()
    rt.broadcast(
        buf_ptr=output.data_ptr(),
        count=output.numel(),
        tilexr_dtype=_torch_dtype_to_tilexr(output.dtype),
        root=root,
        stream_ptr=_current_stream_ptr(device_index),
    )
    return output
```

- [ ] **Step 6: Export helper functions**

In `__init__.py`, import and add `all_reduce`, `reduce_scatter`, and `broadcast` to `__all__`.

- [ ] **Step 7: Verify helper guards**

Run:

```bash
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m py_compile integrations/vllm_ascend/tilexr_collectives/torch_collectives.py integrations/vllm_ascend/tilexr_collectives/__init__.py
```

Expected: helper and runtime guards pass; adapter guard still fails until Task 4.

- [ ] **Step 8: Commit torch helpers**

```bash
git add integrations/vllm_ascend/tilexr_collectives/torch_collectives.py integrations/vllm_ascend/tilexr_collectives/__init__.py
git commit -m "feat: add vllm compatible tilexr torch collectives"
```

## Task 4: Add the Opt-In vllm Adapter

**Files:**
- Create: `integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py`
- Modify: `integrations/vllm_ascend/tilexr_collectives/__init__.py`

- [ ] **Step 1: Create adapter**

Create `vllm_adapter.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass

from .torch_collectives import all_gather, all_reduce, all_to_all, reduce_scatter


def _flag_enabled(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _flag_enabled(os.environ.get("VLLM_ASCEND_TILEXR_COLLECTIVES"))


@dataclass
class TileXRVllmCollectivesAdapter:
    rank: int
    world_size: int
    install_prefix: str

    def should_fallback(self, tensor=None, *, scatter_sizes=None, gather_sizes=None) -> bool:
        if not enabled():
            return True
        if tensor is not None and getattr(tensor, "device", None).type != "npu":
            return True
        return scatter_sizes is not None or gather_sizes is not None

    def all_reduce(self, input_):
        if self.should_fallback(input_):
            return None
        return all_reduce(input_, self.rank, self.world_size, self.install_prefix)

    def all_gather(self, input_, dim: int = -1):
        if self.should_fallback(input_):
            return None
        return all_gather(input_, self.rank, self.world_size, self.install_prefix, dim=dim)

    def reduce_scatter(self, input_, dim: int = -1):
        if self.should_fallback(input_):
            return None
        return reduce_scatter(input_, self.rank, self.world_size, self.install_prefix, dim=dim)

    def all_to_all(
        self,
        input_,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ):
        if self.should_fallback(input_, scatter_sizes=scatter_sizes, gather_sizes=gather_sizes):
            return None
        if scatter_dim != gather_dim:
            return None
        return all_to_all(input_, self.rank, self.world_size, self.install_prefix, scatter_dim=scatter_dim, gather_dim=gather_dim)
```

- [ ] **Step 2: Export adapter**

In `__init__.py`, import `TileXRVllmCollectivesAdapter` and `enabled` as `tilexr_vllm_collectives_enabled`, and add both to `__all__`.

- [ ] **Step 3: Verify all source guards GREEN**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m py_compile integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py integrations/vllm_ascend/tilexr_collectives/__init__.py
```

Expected: all source guards pass.

- [ ] **Step 4: Commit adapter**

```bash
git add integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py integrations/vllm_ascend/tilexr_collectives/__init__.py
git commit -m "feat: add opt-in vllm tilexr collectives adapter"
```

## Task 5: Expand Smoke Tests and Remote Validation

**Files:**
- Modify: `integrations/vllm_ascend/smoke_collectives.py`
- Modify: `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
- Modify: `tests/collectives/deploy_and_run_vllm_remote.sh`
- Modify: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Add source guards for new smoke ops**

Add to `test_vllm_collectives_integration_sources.py`:

```python
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
```

Update `main()` to call it.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: failure because smoke/deploy scripts do not yet contain the new ops and probe.

- [ ] **Step 3: Extend smoke parser and dispatch**

In `smoke_collectives.py`:

- Change `parser.add_argument("--op", choices=[...])` to include `allreduce`, `reducescatter`, and `broadcast`.
- Import `all_reduce`, `reduce_scatter`, and `broadcast`.
- Add deterministic value/check helpers:

```python
def check_all_reduce(result, rank_size: int, count: int, dtype) -> None:
    cpu = result.cpu().reshape(-1)
    expected = sum(torch.arange(src * 1000, src * 1000 + count, dtype=torch.int32) for src in range(rank_size))
    if dtype is torch.float16:
        expected = expected.to(torch.float16)
    if not torch.equal(cpu, expected):
        raise AssertionError(f"AllReduce mismatch: actual={cpu} expected={expected}")
```

```python
def make_reduce_scatter_values(rank_size: int, rank: int, count: int, dtype):
    host = []
    for dst in range(rank_size):
        for idx in range(count):
            host.append(rank * 100000 + dst * 1000 + idx)
    values = torch.tensor(host, dtype=torch.int32, device=f"npu:{torch.npu.current_device()}")
    if dtype is torch.float16:
        values = values.to(torch.float16)
    return values
```

```python
def check_reduce_scatter(result, rank_size: int, rank: int, count: int, dtype) -> None:
    cpu = result.cpu().reshape(-1)
    expected = torch.tensor(
        [sum(src * 100000 + rank * 1000 + idx for src in range(rank_size)) for idx in range(count)],
        dtype=torch.int32,
    )
    if dtype is torch.float16:
        expected = expected.to(torch.float16)
    if not torch.equal(cpu, expected):
        raise AssertionError(f"ReduceScatter mismatch on rank={rank}: actual={cpu} expected={expected}")
```

```python
def check_broadcast(result, root: int, count: int, dtype) -> None:
    cpu = result.cpu().reshape(-1)
    expected = torch.arange(root * 1000, root * 1000 + count, dtype=torch.int32)
    if dtype is torch.float16:
        expected = expected.to(torch.float16)
    if not torch.equal(cpu, expected):
        raise AssertionError(f"Broadcast mismatch: actual={cpu} expected={expected}")
```

Dispatch based on `args.op`:

```python
elif args.op == "allreduce":
    send = make_values(args.rank_size, args.rank, args.count, dtype).contiguous()
    result = all_reduce(send, rank=args.rank, world_size=args.rank_size, install_prefix=args.install_prefix)
    torch.npu.synchronize()
    check_all_reduce(result, args.rank_size, args.count, dtype)
elif args.op == "reducescatter":
    send = make_reduce_scatter_values(args.rank_size, args.rank, args.count, dtype).contiguous()
    result = reduce_scatter(send, rank=args.rank, world_size=args.rank_size, install_prefix=args.install_prefix, dim=0)
    torch.npu.synchronize()
    check_reduce_scatter(result, args.rank_size, args.rank, args.count, dtype)
elif args.op == "broadcast":
    send = make_values(args.rank_size, args.rank if args.rank == 0 else args.rank, args.count, dtype).contiguous()
    result = broadcast(send, rank=args.rank, world_size=args.rank_size, install_prefix=args.install_prefix, root=0)
    torch.npu.synchronize()
    check_broadcast(result, 0, args.count, dtype)
```

- [ ] **Step 4: Extend launcher op validation**

In `run_tilexr_collectives_smoke.sh`, replace the op validation with:

```bash
case "${OP}" in
  allgather|alltoall|allreduce|reducescatter|broadcast) ;;
  *)
    echo "ERROR: op must be allgather, alltoall, allreduce, reducescatter, or broadcast" >&2
    exit 2
    ;;
esac
```

- [ ] **Step 5: Extend remote script**

In `deploy_and_run_vllm_remote.sh`, add a remote function:

```bash
probe_vllm_environment() {
  "${selected_python}" - <<'PY'
import importlib.util
for mod in ["vllm", "vllm_ascend"]:
    spec = importlib.util.find_spec(mod)
    print(f"{mod}: {spec.origin if spec else 'MISSING'}")
PY
}
```

Call it after Python environment dump.

Add smoke calls for:

```bash
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install allgather int32
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install allgather fp16
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install allreduce int32
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install allreduce fp16
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install reducescatter int32
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install reducescatter fp16
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install broadcast int32
TILEXR_VLLM_SMOKE_PYTHON="${selected_python}" integrations/vllm_ascend/run_tilexr_collectives_smoke.sh 2 16 0 install broadcast fp16
```

Set `VLLM_ASCEND_TILEXR_COLLECTIVES=1` in the remote smoke environment.

- [ ] **Step 6: Verify local checks**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
bash -n integrations/vllm_ascend/run_tilexr_collectives_smoke.sh tests/collectives/deploy_and_run_vllm_remote.sh
python3 -m py_compile integrations/vllm_ascend/smoke_collectives.py
```

Expected: pass.

- [ ] **Step 7: Commit smoke expansion**

```bash
git add integrations/vllm_ascend/smoke_collectives.py integrations/vllm_ascend/run_tilexr_collectives_smoke.sh tests/collectives/deploy_and_run_vllm_remote.sh tests/collectives/unit/test_vllm_collectives_integration_sources.py
git commit -m "test: expand vllm tilexr collectives smoke"
```

## Task 6: Update Documentation

**Files:**
- Modify: `tests/collectives/README.md`
- Optionally create: `integrations/vllm_ascend/README.md`

- [ ] **Step 1: Add documentation guard**

Add to `test_vllm_collectives_integration_sources.py`:

```python
def test_phase3_docs_describe_feature_flag_and_boundaries() -> None:
    readme = read_rel("tests/collectives/README.md")
    for token in [
        "VLLM_ASCEND_TILEXR_COLLECTIVES=1",
        "allreduce",
        "reducescatter",
        "broadcast",
        "fallback",
        "vllm-ascend inference",
    ]:
        assert token in readme
```

Update `main()` to call it.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: docs guard fails until README is updated.

- [ ] **Step 3: Update README**

In `tests/collectives/README.md`, update the vllm section to describe:

- Phase 3 uses `VLLM_ASCEND_TILEXR_COLLECTIVES=1`.
- Current TileXR Python shim validates allgather, allreduce, reducescatter, broadcast, and topology-gated alltoall.
- Uneven `all_gatherv` / `reduce_scatterv` and unsupported shapes fall back in the vllm adapter.
- Shim smoke is not a substitute for final vllm-ascend inference validation.

- [ ] **Step 4: Verify docs guard**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: pass.

- [ ] **Step 5: Commit docs**

```bash
git add tests/collectives/README.md tests/collectives/unit/test_vllm_collectives_integration_sources.py
git commit -m "docs: document vllm tilexr phase3 validation"
```

## Task 7: Local and Remote Verification

**Files:**
- No source changes expected unless verification exposes bugs.

- [ ] **Step 1: Run local verification**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
bash -n integrations/vllm_ascend/run_tilexr_collectives_smoke.sh tests/collectives/deploy_and_run_vllm_remote.sh
python3 -m py_compile integrations/vllm_ascend/smoke_collectives.py integrations/vllm_ascend/tilexr_collectives/runtime.py integrations/vllm_ascend/tilexr_collectives/torch_collectives.py integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py integrations/vllm_ascend/tilexr_collectives/__init__.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: Run CMake registration smoke if local CANN is sufficient**

Run:

```bash
cmake -S . -B build-vllm-phase3-local -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DBUILD_TESTING=ON
ctest --test-dir build-vllm-phase3-local -R test_vllm_collectives_integration_sources --output-on-failure
```

Expected: CTest sees and passes the source guard. If local CANN/toolchain is unavailable, record the exact configure failure and rely on `blue` for full build.

- [ ] **Step 3: Run remote blue validation**

Run:

```bash
TILEXR_VLLM_REMOTE=blue \
TILEXR_VLLM_REMOTE_BASE=/home/<remote-user>/tilexr_vllm_phase3_$(date +%Y%m%d_%H%M%S) \
TILEXR_VLLM_REMOTE_CONDA_ENV=tt4 \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

Expected:

- remote build/install succeeds;
- CTest passes;
- allgather/allreduce/reducescatter/broadcast int32/fp16 PASS;
- vllm/vllm-ascend probe output is present;
- remote log path is printed.

- [ ] **Step 4: Investigate failures systematically**

If any command fails, use `superpowers:systematic-debugging` before modifying code. Fix with TDD where possible and commit each fix.

## Task 8: Final Review, PR, and Completion Audit

**Files:**
- No source changes expected unless review exposes bugs.

- [ ] **Step 1: Request code review**

Use the code-review workflow with:

- Base: `origin/main`
- Head: current branch
- Focus: main-history migration, TileXR Python ABI coverage, opt-in adapter fallback, remote validation.

- [ ] **Step 2: Run final verification**

Repeat local verification after any review fixes:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py
bash -n integrations/vllm_ascend/run_tilexr_collectives_smoke.sh tests/collectives/deploy_and_run_vllm_remote.sh
python3 -m py_compile integrations/vllm_ascend/smoke_collectives.py integrations/vllm_ascend/tilexr_collectives/runtime.py integrations/vllm_ascend/tilexr_collectives/torch_collectives.py integrations/vllm_ascend/tilexr_collectives/vllm_adapter.py integrations/vllm_ascend/tilexr_collectives/__init__.py
git diff --check
```

- [ ] **Step 3: Push branch and create PR**

Run:

```bash
git status --short --branch
git push -u origin codex/vllm-collectives-phase3
gh pr create --base main --head codex/vllm-collectives-phase3 --title "Integrate TileXR collectives with vllm-ascend" --body-file /tmp/tilexr-vllm-phase3-pr.md
```

The PR body must include:

- Summary of migration from rewritten `main`;
- Feature flag and fallback behavior;
- Local verification commands;
- Remote `blue` validation result and log path;
- Remaining environment or inference-path caveats, if any.

- [ ] **Step 4: Completion audit**

Before marking the goal complete, verify:

- branch is based on `origin/main`;
- PR exists and targets `main`;
- docs are updated;
- local checks pass;
- remote `blue` checks pass;
- vllm/vllm-ascend inference validation has either passed or the PR explicitly includes a resolved environment setup and a concrete passing substitute accepted by the user.
