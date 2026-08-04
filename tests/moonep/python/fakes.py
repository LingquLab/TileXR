from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = ROOT / 'integrations' / 'moonep_torch'
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))

from tilexr_moonep.runtime import NativeCapabilities


@dataclass(frozen=True)
class FakeDevice:
    type: str = "npu"
    index: int | None = 0

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FakeTensor:
    _next_ptr = 0x1000

    def __init__(
        self,
        shape,
        dtype,
        device="npu:0",
        *,
        contiguous=True,
        storage_offset=0,
    ):
        self.shape = tuple(int(value) for value in shape)
        self.dtype = dtype
        if isinstance(device, FakeDevice):
            self.device = device
        else:
            kind, _, index = str(device).partition(":")
            self.device = FakeDevice(kind, int(index) if index else None)
        self._contiguous = bool(contiguous)
        self._storage_offset = int(storage_offset)
        self._item = 0
        self._ptr = FakeTensor._next_ptr
        FakeTensor._next_ptr += max(64, self.numel() * self.element_size())

    def is_contiguous(self):
        return self._contiguous

    def storage_offset(self):
        return self._storage_offset

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return {
            "uint8": 1,
            "float16": 2,
            "bfloat16": 2,
            "int32": 4,
            "float32": 4,
            "int64": 8,
        }[self.dtype]

    def data_ptr(self):
        return self._ptr

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return FakeTensor(shape, self.dtype, self.device)

    def to(self, *, dtype):
        return FakeTensor(self.shape, dtype, self.device)

    def item(self):
        return self._item

    def __mul__(self, other):
        return FakeTensor(self.shape, self.dtype, self.device)

    __rmul__ = __mul__

    def contiguous(self):
        return self


class FakeStream:
    def __init__(self, value=0xCAFE):
        self.npu_stream = int(value)


class FakeNpu:
    def __init__(self, current_device=0):
        self._current_device = int(current_device)
        self._stream = FakeStream()
        self.synchronize_calls = []

    def current_device(self):
        return self._current_device

    def current_stream(self):
        return self._stream

    def synchronize(self, device=None):
        self.synchronize_calls.append(device)


class FakeTorch:
    uint8 = "uint8"
    int32 = "int32"
    int64 = "int64"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self, current_device=0):
        self.npu = FakeNpu(current_device)

    @staticmethod
    def empty(shape, *, dtype, device):
        return FakeTensor(shape, dtype, device)


class FakeRuntime:
    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size
        self.calls = []
        self.closed = False
        self.capabilities = NativeCapabilities(
            abi_version=1,
            stage_mask=1,
            stub_mask=30,
        )

    def planning_workspace_size(self, context):
        self.calls.append(("planning_workspace_size", None))
        return 64

    def planning(self, context, topk, tpe, plan, stream, wait_iterations):
        self.calls.append(("planning", plan, stream, wait_iterations))

    def dispatch(
        self,
        context,
        plan,
        hidden,
        output,
        stream,
        route_weights=None,
        output_route_weights=None,
    ):
        self.calls.append(("dispatch", plan, stream))

    def prefetch_weight(self, context, plan, projections, outputs, stream):
        self.calls.append(("prefetch_weight", plan, stream))

    def combine(
        self,
        context,
        plan,
        hidden,
        output,
        stream,
        route_weights=None,
        output_route_weights=None,
    ):
        self.calls.append(("combine", plan, stream))

    def reduce_grad(self, context, plan, gradients, stream):
        self.calls.append(("reduce_grad", plan, stream))

    def close(self):
        self.closed = True
        self.calls.append(("close", None))
