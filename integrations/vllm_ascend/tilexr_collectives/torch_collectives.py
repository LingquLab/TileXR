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


def _npu_device_index(tensor) -> int:
    if tensor.device.index is None:
        return int(_torch().npu.current_device())
    return int(tensor.device.index)


def _bind_npu_device(device_index: int) -> None:
    torch = _torch()
    torch.npu.set_device(device_index)


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


def _select_runtime(runtime, rank: int, world_size: int, install_prefix: str, device_index: int):
    if runtime is None:
        return get_runtime(rank, world_size, install_prefix, device_index=device_index)
    runtime_device_index = getattr(runtime, "device_index", None)
    if runtime_device_index is None:
        raise ValueError("runtime must expose device_index when passed explicitly")
    if int(runtime_device_index) != int(device_index):
        raise ValueError(f"runtime device_index={runtime_device_index} does not match tensor device_index={device_index}")
    return runtime


def all_gather(tensor, rank: int, world_size: int, install_prefix: str, runtime=None):
    torch = _torch()
    _validate_npu_contiguous(tensor, "tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must contain at least one element")
    device_index = _npu_device_index(tensor)
    _bind_npu_device(device_index)
    rt = _select_runtime(runtime, rank, world_size, install_prefix, device_index)
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
    rt = _select_runtime(runtime, rank, world_size, install_prefix, device_index)
    output = torch.empty_like(tensor)
    rt.all_to_all(
        send_ptr=tensor.data_ptr(),
        recv_ptr=output.data_ptr(),
        send_count_per_peer=tensor.numel() // world_size,
        tilexr_dtype=_torch_dtype_to_tilexr(tensor.dtype),
        stream_ptr=_current_stream_ptr(device_index),
    )
    return output
