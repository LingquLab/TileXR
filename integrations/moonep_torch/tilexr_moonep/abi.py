from __future__ import annotations

import ctypes
from enum import IntEnum, IntFlag


TILEXR_SUCCESS = 0
TILEXR_MOONEP_ABI_VERSION = 1
TILEXR_MOONEP_MAX_TENSOR_RANK = 4
TILEXR_MOONEP_FLAG_NONE = 0


class TileXRMoonEPDType(IntEnum):
    INT32 = 2
    FLOAT16 = 3
    FLOAT32 = 4
    BFLOAT16 = 11


class TileXRMoonEPStage(IntFlag):
    PLANNING = 1 << 0
    DISPATCH = 1 << 1
    PREFETCH_WEIGHT = 1 << 2
    COMBINE = 1 << 3
    REDUCE_GRAD = 1 << 4


class TileXRMoonEPTensorV1(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_uint32),
        ("abiVersion", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("elementCount", ctypes.c_uint64),
        ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_int64 * TILEXR_MOONEP_MAX_TENSOR_RANK),
    ]


class TileXRMoonEPPlanV1(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_uint32),
        ("abiVersion", ctypes.c_uint32),
        ("s", ctypes.c_int64),
        ("k", ctypes.c_int64),
        ("e", ctypes.c_int64),
        ("b", ctypes.c_int64),
        ("rank", ctypes.c_int64),
        ("world", ctypes.c_int64),
        ("dispatchedCapacity", ctypes.c_int64),
        ("dst", ctypes.c_void_p),
        ("cu", ctypes.c_void_p),
        ("expertsToCopy", ctypes.c_void_p),
        ("remoteStats", ctypes.c_void_p),
        ("status", ctypes.c_void_p),
    ]


class TileXRMoonEPPlanningArgsV1(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_uint32),
        ("abiVersion", ctypes.c_uint32),
        ("comm", ctypes.c_void_p),
        ("topkExperts", ctypes.POINTER(TileXRMoonEPTensorV1)),
        ("tokensPerExpert", ctypes.POINTER(TileXRMoonEPTensorV1)),
        ("workspace", ctypes.c_void_p),
        ("workspaceBytes", ctypes.c_uint64),
        ("plan", ctypes.POINTER(TileXRMoonEPPlanV1)),
        ("waitIterations", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
    ]


class _TileXRMoonEPStageArgsV1(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_uint32),
        ("abiVersion", ctypes.c_uint32),
        ("comm", ctypes.c_void_p),
        ("plan", ctypes.POINTER(TileXRMoonEPPlanV1)),
        ("input", ctypes.POINTER(TileXRMoonEPTensorV1)),
        ("output", ctypes.POINTER(TileXRMoonEPTensorV1)),
        ("flags", ctypes.c_uint64),
    ]


class TileXRMoonEPDispatchArgsV1(_TileXRMoonEPStageArgsV1):
    pass


class TileXRMoonEPPrefetchWeightArgsV1(_TileXRMoonEPStageArgsV1):
    pass


class TileXRMoonEPCombineArgsV1(_TileXRMoonEPStageArgsV1):
    pass


class TileXRMoonEPReduceGradArgsV1(_TileXRMoonEPStageArgsV1):
    pass


def initialize_struct(value: ctypes.Structure) -> ctypes.Structure:
    value.structSize = ctypes.sizeof(type(value))
    value.abiVersion = TILEXR_MOONEP_ABI_VERSION
    return value


def void_p(value: int | ctypes.c_void_p | None) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        return value
    return ctypes.c_void_p(0 if value is None else int(value))


def dtype_code(dtype) -> int:
    name = str(dtype).lower()
    if "bfloat16" in name:
        return int(TileXRMoonEPDType.BFLOAT16)
    if "float16" in name or name.endswith(".half"):
        return int(TileXRMoonEPDType.FLOAT16)
    if "float32" in name or name.endswith(".float"):
        return int(TileXRMoonEPDType.FLOAT32)
    if "int32" in name:
        return int(TileXRMoonEPDType.INT32)
    raise TypeError(f"unsupported MoonEP dtype: {dtype}")


def tensor_nbytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def tensor_ptr(tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(tensor.data_ptr()))


def make_tensor_v1(tensor) -> TileXRMoonEPTensorV1:
    shape = tuple(int(value) for value in tensor.shape)
    if not shape or len(shape) > TILEXR_MOONEP_MAX_TENSOR_RANK:
        raise ValueError(
            f"MoonEP native tensors must have rank in [1, {TILEXR_MOONEP_MAX_TENSOR_RANK}]"
        )
    value = initialize_struct(TileXRMoonEPTensorV1())
    value.data = tensor_ptr(tensor)
    value.elementCount = int(tensor.numel())
    value.dtype = dtype_code(tensor.dtype)
    value.rank = len(shape)
    for index, dimension in enumerate(shape):
        value.shape[index] = dimension
    return value
