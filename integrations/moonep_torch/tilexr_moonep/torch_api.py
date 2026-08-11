from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .runtime import TileXRMoonEPRuntime


_PREFETCH_WEIGHT_STATUS_SUCCESS = 4000
_UDMA_REGISTRATION_ALIGNMENT = 2 * 1024 * 1024


def _torch():
    import torch

    return torch


def _shape(value) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _device_index(torch_module, tensor) -> int:
    index = tensor.device.index
    if index is None:
        index = torch_module.npu.current_device()
    return int(index)


def _current_stream_ptr(torch_module, device_index: int) -> int:
    current_device = int(torch_module.npu.current_device())
    if current_device != int(device_index):
        raise ValueError(
            f"current NPU device is {current_device}, expected npu:{device_index}; "
            "the caller must select the device before invoking TileXR MoonEP"
        )
    stream = torch_module.npu.current_stream()
    value = getattr(stream, "npu_stream", None)
    if value is None:
        value = getattr(stream, "stream", None)
    if value is None:
        raise RuntimeError("torch.npu.current_stream() does not expose npu_stream or stream")
    return int(value)


def _validate_tensor(
    torch_module,
    tensor,
    name: str,
    *,
    dtype=None,
    shape: tuple[int, ...] | None = None,
    device_index: int,
    allow_empty: bool = False,
    allow_storage_offset: bool = False,
) -> None:
    if tensor is None:
        raise ValueError(f"{name} is required")
    if tensor.device.type != "npu":
        raise ValueError(f"{name} must be on an NPU device, got {tensor.device}")
    actual_device = _device_index(torch_module, tensor)
    if actual_device != int(device_index):
        raise ValueError(f"{name} must be on npu:{device_index}, got {tensor.device}")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if shape is not None and _shape(tensor) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {_shape(tensor)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not allow_storage_offset and int(tensor.storage_offset()) != 0:
        raise ValueError(f"{name} must have storage_offset 0")
    if not allow_empty and int(tensor.numel()) <= 0:
        raise ValueError(f"{name} must contain at least one element")


@dataclass
class ProjectionBuffers:
    gate: Any
    up: Any
    down: Any
    gate_reduce: Any | None = None
    up_reduce: Any | None = None
    down_reduce: Any | None = None
    backing: Any | None = None
    udma_handle: int | None = None

    @classmethod
    def from_local_weights(
        cls,
        context,
        gate,
        up,
        down,
        *,
        slot_fill_value: float = 0.0,
        torch_module=None,
    ) -> "ProjectionBuffers":
        torch_module = torch_module or _torch()
        sources = (gate, up, down)
        alignment = 64
        offsets = []
        cursor_bytes = 0
        for name, source in zip(("gate", "up", "down"), sources):
            _validate_tensor(
                torch_module,
                source,
                f"local_weights.{name}",
                dtype=context.dtype,
                device_index=context.device_index,
            )
            source_shape = _shape(source)
            if (len(source_shape) < 2 or len(source_shape) > 4 or
                    source_shape[0] != context.experts_per_rank):
                raise ValueError(
                    f"local_weights.{name} must have shape [B,...] with B="
                    f"{context.experts_per_rank}, got {source_shape}"
                )
            row_bytes = int(source.numel()) * int(source.element_size()) // source_shape[0]
            if row_bytes <= 0 or row_bytes % alignment:
                raise ValueError(
                    f"local_weights.{name} expert row bytes must be 64-byte aligned, "
                    f"got {row_bytes}"
                )
            cursor_bytes = (cursor_bytes + alignment - 1) // alignment * alignment
            offsets.append(cursor_bytes)
            cursor_bytes += 2 * int(source.numel()) * int(source.element_size())

        element_bytes = int(gate.element_size())
        if any(source.dtype != gate.dtype or int(source.element_size()) != element_bytes
               for source in sources):
            raise TypeError("gate, up, and down local weights must use the same dtype")
        registered_bytes = (
            (cursor_bytes + _UDMA_REGISTRATION_ALIGNMENT - 1) //
            _UDMA_REGISTRATION_ALIGNMENT * _UDMA_REGISTRATION_ALIGNMENT
        )
        allocation_elements = (
            registered_bytes + _UDMA_REGISTRATION_ALIGNMENT - 1 + element_bytes - 1
        ) // element_bytes
        allocation = torch_module.empty(
            (allocation_elements,), dtype=gate.dtype,
            device=f"npu:{context.device_index}"
        )
        allocation_ptr = int(allocation.data_ptr())
        offset_bytes = (-allocation_ptr) % _UDMA_REGISTRATION_ALIGNMENT
        if offset_bytes % element_bytes:
            raise RuntimeError("projection registration offset is not element-aligned")
        backing = allocation.narrow(
            0, offset_bytes // element_bytes, registered_bytes // element_bytes
        )
        if (int(backing.data_ptr()) % _UDMA_REGISTRATION_ALIGNMENT or
                int(backing.numel()) * element_bytes != registered_bytes):
            raise RuntimeError("projection backing is not 2-MiB registration-aligned")

        views = []
        for source, offset_bytes in zip(sources, offsets):
            view_shape = (2 * context.experts_per_rank, *_shape(source)[1:])
            elements = 2 * int(source.numel())
            view = backing.narrow(0, offset_bytes // element_bytes, elements).reshape(
                view_shape
            )
            view.narrow(0, 0, context.experts_per_rank).copy_(source)
            view.narrow(
                0, context.experts_per_rank, context.experts_per_rank
            ).fill_(slot_fill_value)
            views.append(view)
        return cls(gate=views[0], up=views[1], down=views[2], backing=backing)


@dataclass(frozen=True, slots=True)
class MoonEPPlan:
    dst: Any
    experts_to_copy: Any
    zero_fill_ranges: Any
    remote_stats: Any
    dup_groups: Any
    dup_loffs: Any
    dup_counts: Any
    status: Any
    reduce_grad_status: Any
    workspace: Any
    n: int
    tokens_per_rank: int
    topk: int
    expert_count: int
    rank_size: int
    prefetch_slots: int
    nv_s: int
    token_padding: int
    epoch: int
    backend: str
    runtime: Any
    dst_local_offset: int = 0

    @property
    def dispatched_capacity(self) -> int:
        return int(self.nv_s)


@dataclass
class DispatchResult:
    hidden: Any
    route_weights: Any | None
    plan: MoonEPPlan


@dataclass
class CombineResult:
    hidden: Any
    route_weights: Any | None


@dataclass
class ForwardState:
    plan: MoonEPPlan
    projections: ProjectionBuffers
    dispatched_route_weights: Any | None = None
    apply_route_weights: bool = True
    expert_cache: Any = None


@dataclass
class ForwardResult:
    hidden: Any
    route_weights: Any | None
    state: ForwardState


@dataclass
class BackwardResult:
    grad_hidden: Any
    plan: MoonEPPlan


@dataclass
class TileXRMoonEPContext:
    runtime: Any
    global_rank: int
    global_world_size: int
    node_rank: int
    node_count: int
    local_rank: int
    local_world_size: int
    planner_group_rank: int
    planner_group_size: int
    lane_group_rank: int
    lane_group_size: int
    device_index: int
    tokens_per_rank: int
    hidden_size: int
    topk: int
    expert_count: int
    dtype: Any
    token_padding: int = 1
    prefetch_slots: int | None = None
    _dispatch_workspace_owner: Any = field(init=False, default=None, repr=False)
    _dispatch_workspace_ptr: int = field(init=False, default=0, repr=False)
    _dispatch_workspace_bytes: int = field(init=False, default=0, repr=False)
    _dispatch_workspace_alignment: int = field(init=False, default=0, repr=False)
    _dispatch_workspace_handle: int | None = field(init=False, default=None, repr=False)
    _buffer_owner: Any = field(init=False, default=None, repr=False)
    _bound_stream_ptr: int | None = field(init=False, default=None, repr=False)
    _poisoned: bool = field(init=False, default=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.planner_group_size <= 0 or self.planner_group_size > 128:
            raise ValueError(
                f"planner_group_size must be in [1, 128], got {self.planner_group_size}"
            )
        if self.planner_group_rank < 0 or self.planner_group_rank >= self.planner_group_size:
            raise ValueError("planner_group_rank is outside the planner group")
        if (self.tokens_per_rank <= 0 or self.hidden_size <= 0 or self.topk <= 0 or
                self.topk > 32):
            raise ValueError(
                "tokens_per_rank and hidden_size must be positive; topk must be in [1, 32]"
            )
        if self.expert_count <= 0 or self.expert_count % self.planner_group_size != 0:
            raise ValueError(
                "expert_count must be positive and divisible by planner_group_size"
            )
        if self.token_padding <= 0:
            raise ValueError("token_padding must be positive")
        if self.prefetch_slots is None:
            self.prefetch_slots = self.experts_per_rank
        if self.prefetch_slots <= 0 or self.prefetch_slots > self.experts_per_rank:
            raise ValueError("prefetch_slots must be in [1, E/R]")
        if self.nv_s > 2**63 - 1:
            raise OverflowError("NvS exceeds signed int64")
        if self.planner_group_size * self.nv_s > 2**31:
            raise OverflowError(
                "signed int32 dst encoding requires "
                "planner_group_size * NvS <= INT32_MAX + 1"
            )
        runtime_rank = getattr(self.runtime, "rank", self.planner_group_rank)
        runtime_world = getattr(self.runtime, "world_size", self.planner_group_size)
        if int(runtime_rank) != self.planner_group_rank or int(runtime_world) != self.planner_group_size:
            raise ValueError("runtime rank metadata does not match the planner group")

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.planner_group_size

    @property
    def dispatched_capacity(self) -> int:
        return self.nv_s

    @property
    def route_count(self) -> int:
        return self.tokens_per_rank * self.topk

    @property
    def nv_s(self) -> int:
        return self.route_count + (self.token_padding - 1) * 2 * self.experts_per_rank

    @classmethod
    def from_env(
        cls,
        *,
        tokens_per_rank: int,
        hidden_size: int,
        topk: int,
        expert_count: int,
        dtype,
        token_padding: int = 1,
        prefetch_slots: int | None = None,
        runtime=None,
        install_prefix=None,
        torch_module=None,
    ) -> "TileXRMoonEPContext":
        torch_module = torch_module or _torch()
        global_rank = int(os.environ.get("RANK", "0"))
        global_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank)))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(global_world_size)))
        planner_group_rank = int(os.environ.get("TILEXR_PLANNER_GROUP_RANK", str(global_rank)))
        planner_group_size = int(
            os.environ.get("TILEXR_PLANNER_GROUP_SIZE", str(global_world_size))
        )
        device_index = int(torch_module.npu.current_device())
        if local_rank != device_index:
            raise ValueError(
                f"LOCAL_RANK={local_rank} does not match current NPU device {device_index}"
            )
        if runtime is None:
            runtime = TileXRMoonEPRuntime(
                rank=planner_group_rank,
                world_size=planner_group_size,
                install_prefix=install_prefix,
            )
        return cls(
            runtime=runtime,
            global_rank=global_rank,
            global_world_size=global_world_size,
            node_rank=int(os.environ.get("NODE_RANK", "0")),
            node_count=int(os.environ.get("NODE_COUNT", "1")),
            local_rank=local_rank,
            local_world_size=local_world_size,
            planner_group_rank=planner_group_rank,
            planner_group_size=planner_group_size,
            lane_group_rank=int(os.environ.get("TILEXR_LANE_GROUP_RANK", "0")),
            lane_group_size=int(os.environ.get("TILEXR_LANE_GROUP_SIZE", "1")),
            device_index=device_index,
            tokens_per_rank=int(tokens_per_rank),
            hidden_size=int(hidden_size),
            topk=int(topk),
            expert_count=int(expert_count),
            dtype=dtype,
            token_padding=int(token_padding),
            prefetch_slots=(None if prefetch_slots is None else int(prefetch_slots)),
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._buffer_owner is not None:
            raise RuntimeError("close the owning MoonEP Buffer before closing its context")
        self.runtime.unregister_dispatch_workspace(self._dispatch_workspace_handle)
        self._dispatch_workspace_handle = None
        self._dispatch_workspace_ptr = 0
        self._dispatch_workspace_bytes = 0
        self._dispatch_workspace_owner = None
        self.runtime.close()
        self._closed = True

    def attach_buffer(self, owner: Any) -> None:
        if self._closed or self._poisoned:
            raise RuntimeError("MoonEP context is not reusable")
        if self._buffer_owner is not None and self._buffer_owner is not owner:
            raise RuntimeError(
                "one MoonEP communicator/workspace can be owned by only one Buffer"
            )
        self._buffer_owner = owner

    def detach_buffer(self, owner: Any) -> None:
        if self._buffer_owner is owner:
            self._buffer_owner = None
            self._bound_stream_ptr = None

    def bind_stream(self, owner: Any, stream_ptr: int) -> None:
        if self._poisoned:
            raise RuntimeError("MoonEP context is poisoned and cannot launch more work")
        if self._buffer_owner is not owner:
            raise RuntimeError("Buffer does not own the MoonEP context workspace")
        if self._bound_stream_ptr is None:
            self._bound_stream_ptr = int(stream_ptr)
        elif self._bound_stream_ptr != int(stream_ptr):
            raise RuntimeError(
                "MoonEP communicator/workspace is bound to one ordered NPU stream; "
                "quiesce before changing streams"
            )

    def release_stream(self, owner: Any) -> None:
        if self._buffer_owner is owner:
            self._bound_stream_ptr = None

    def ensure_dispatch_workspace(self, torch_module) -> None:
        if self._dispatch_workspace_owner is not None:
            return
        workspace_bytes, alignment = self.runtime.dispatch_workspace_size(self)
        if workspace_bytes <= 0 or alignment <= 0 or workspace_bytes % alignment != 0:
            raise RuntimeError(
                "native Dispatch returned invalid workspace contract "
                f"bytes={workspace_bytes} alignment={alignment}"
            )
        raw = torch_module.empty(
            (workspace_bytes + alignment - 1,),
            dtype=torch_module.uint8,
            device=f"npu:{self.device_index}",
        )
        raw.zero_()
        raw_ptr = int(raw.data_ptr())
        aligned_ptr = ((raw_ptr + alignment - 1) // alignment) * alignment
        if aligned_ptr + workspace_bytes > raw_ptr + int(raw.numel()):
            raise RuntimeError("aligned Dispatch workspace exceeds its raw allocation")
        handle = self.runtime.register_dispatch_workspace(aligned_ptr, workspace_bytes)
        self._dispatch_workspace_owner = raw
        self._dispatch_workspace_ptr = aligned_ptr
        self._dispatch_workspace_bytes = workspace_bytes
        self._dispatch_workspace_alignment = alignment
        self._dispatch_workspace_handle = handle

    def activate_dispatch_workspace(self) -> None:
        if self._dispatch_workspace_owner is None:
            raise RuntimeError("Dispatch workspace is not initialized")
        self._dispatch_workspace_handle = self.runtime.register_dispatch_workspace(
            self._dispatch_workspace_ptr, self._dispatch_workspace_bytes
        )

    @property
    def dispatch_workspace(self) -> tuple[int, int]:
        if self._dispatch_workspace_owner is None:
            raise RuntimeError("Dispatch workspace is not initialized")
        return self._dispatch_workspace_ptr, self._dispatch_workspace_bytes

    def mark_poisoned(self) -> None:
        self._poisoned = True


class TileXRMoonEPBuffer:
    """Torch facade for an explicitly orchestrated MoonEP forward/backward flow."""

    def __init__(
        self,
        context: TileXRMoonEPContext,
        *,
        wait_iterations: int = 1_000_000,
        requested_udma_chunk_bytes: int = 0,
        torch_module=None,
    ):
        if wait_iterations <= 0:
            raise ValueError(f"wait_iterations must be positive, got {wait_iterations}")
        self.context = context
        self.runtime = context.runtime
        self.wait_iterations = int(wait_iterations)
        self.requested_udma_chunk_bytes = int(requested_udma_chunk_bytes)
        if self.requested_udma_chunk_bytes < 0:
            raise ValueError("requested_udma_chunk_bytes must be non-negative")
        self._torch = torch_module or _torch()
        self._closed = False
        self._epoch = 0
        self._bound_stream_ptr: int | None = None
        self._pending_refs: list[tuple[Any, ...]] = []
        self._pending_plans: list[MoonEPPlan] = []
        self._pending_reduce_plans: list[MoonEPPlan] = []
        self._pending_statuses: dict[int, int] = {}
        self._plans_needing_dedup: dict[int, MoonEPPlan] = {}
        self._registered_projections: ProjectionBuffers | None = None
        self._reduce_grad_owner_token = object()
        self._reduce_grad_token_held = False
        self._reduce_grad_inflight = False
        self._reduce_grad_workspace = None
        self._reduce_grad_workspace_allocation = None
        self._reduce_grad_signature = None
        self._reduce_grad_info = None
        self._quiesced = True
        if context.dtype not in (self._torch.bfloat16, self._torch.float16):
            raise TypeError(
                f"MoonEP hidden dtype must be bfloat16 or float16, got {context.dtype}"
            )
        self.context.attach_buffer(self)
        try:
            self.context.ensure_dispatch_workspace(self._torch)
            self._stream_ptr()
        except Exception:
            self.context.detach_buffer(self)
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TileXRMoonEPBuffer is closed")
        if self.context._poisoned:
            raise RuntimeError("TileXRMoonEPBuffer is poisoned")

    def _stream_ptr(self) -> int:
        stream_ptr = _current_stream_ptr(self._torch, self.context.device_index)
        if self._bound_stream_ptr is None:
            self._bound_stream_ptr = stream_ptr
        elif stream_ptr != self._bound_stream_ptr:
            raise RuntimeError(
                "TileXRMoonEPBuffer is bound to one NPU stream; create a separate "
                "buffer or synchronize before using another stream"
            )
        self.context.bind_stream(self, stream_ptr)
        return stream_ptr

    def _synchronize_device(self) -> None:
        try:
            self._torch.npu.synchronize(self.context.device_index)
        except TypeError:
            self._torch.npu.synchronize()

    def _trace_stage(self, stage: str) -> None:
        if os.environ.get("TILEXR_MOONEP_TRACE_STAGES", "0") == "1":
            print(
                f"[TileXR MoonEP rank {self.context.global_rank}] {stage}",
                flush=True,
            )

    def _retain(self, plan: MoonEPPlan, *values: Any) -> None:
        self._quiesced = False
        self._pending_refs.append((plan, *values))
        if all(existing is not plan for existing in self._pending_plans):
            self._pending_plans.append(plan)

    def _expect_status(self, plan: MoonEPPlan, status: int) -> None:
        self._pending_statuses[id(plan)] = int(status)

    def _empty(self, shape: tuple[int, ...], dtype):
        return self._torch.empty(
            shape,
            dtype=dtype,
            device=f"npu:{self.context.device_index}",
        )

    def _zeros(self, shape: tuple[int, ...], dtype):
        return self._torch.zeros(
            shape,
            dtype=dtype,
            device=f"npu:{self.context.device_index}",
        )

    def _aligned_workspace(self, size_bytes: int, alignment: int):
        if size_bytes <= 0 or alignment <= 0:
            raise ValueError("ReduceGrad workspace size and alignment must be positive")
        allocation = self._zeros((size_bytes + alignment - 1,), self._torch.uint8)
        offset = (-int(allocation.data_ptr())) % alignment
        workspace = allocation.narrow(0, offset, size_bytes)
        if int(workspace.data_ptr()) % alignment != 0:
            raise RuntimeError("failed to align ReduceGrad UDMA workspace")
        return workspace, allocation

    def _record_event(self):
        stream = self._torch.npu.current_stream()
        if hasattr(stream, "record_event"):
            return stream.record_event()
        event_type = getattr(self._torch.npu, "Event", None)
        if event_type is None:
            raise RuntimeError("torch.npu does not expose event recording")
        event = event_type()
        event.record(stream)
        return event

    def _validate_plan(self, plan: MoonEPPlan) -> None:
        if not isinstance(plan, MoonEPPlan):
            raise TypeError("plan must be a MoonEPPlan")
        if plan.runtime is not self.runtime:
            raise ValueError("plan belongs to another TileXR MoonEP runtime")
        c = self.context
        dims = (
            plan.tokens_per_rank,
            plan.topk,
            plan.expert_count,
            plan.rank_size,
            plan.prefetch_slots,
            plan.nv_s,
            plan.token_padding,
        )
        expected_dims = (
            c.tokens_per_rank,
            c.topk,
            c.expert_count,
            c.planner_group_size,
            c.prefetch_slots,
            c.nv_s,
            c.token_padding,
        )
        if dims != expected_dims:
            raise ValueError("plan dimensions do not match the buffer context")
        for name, tensor, shape in (
            ("plan.dst", plan.dst, (c.route_count,)),
            ("plan.experts_to_copy", plan.experts_to_copy,
                (c.planner_group_size, c.prefetch_slots)),
            ("plan.zero_fill_ranges", plan.zero_fill_ranges,
                (c.expert_count + c.prefetch_slots, 2)),
            ("plan.remote_stats", plan.remote_stats, (2,)),
            ("plan.dup_groups", plan.dup_groups, (c.nv_s, 3)),
            ("plan.dup_loffs", plan.dup_loffs, (c.nv_s,)),
            ("plan.dup_counts", plan.dup_counts, (2,)),
            ("plan.status", plan.status, (1,)),
            ("plan.reduce_grad_status", plan.reduce_grad_status, (1,)),
        ):
            _validate_tensor(
                self._torch,
                tensor,
                name,
                dtype=self._torch.int32,
                shape=shape,
                device_index=c.device_index,
            )

    def planning(self, topk_experts, tokens_per_expert) -> tuple[MoonEPPlan, Any]:
        self._require_open()
        c = self.context
        _validate_tensor(
            self._torch,
            topk_experts,
            "topk_experts",
            dtype=self._torch.int32,
            shape=(c.tokens_per_rank, c.topk),
            device_index=c.device_index,
        )
        _validate_tensor(
            self._torch,
            tokens_per_expert,
            "tokens_per_expert",
            dtype=self._torch.int32,
            shape=(c.expert_count,),
            device_index=c.device_index,
        )
        self._trace_stage("planning_workspace_begin")
        workspace_bytes = int(self.runtime.planning_workspace_size(c))
        self._trace_stage("planning_workspace_end")
        if workspace_bytes <= 0:
            raise RuntimeError(f"native planner returned invalid workspace size {workspace_bytes}")
        dst_local_offset = int(self.runtime.planning_dst_local_offset(c))
        dst_local_bytes = int(c.nv_s) * 4
        if (
            dst_local_offset < 0
            or dst_local_offset % 4 != 0
            or dst_local_offset + dst_local_bytes > workspace_bytes
        ):
            raise RuntimeError(
                "native Planner returned invalid dstLocal workspace range "
                f"offset={dst_local_offset} bytes={dst_local_bytes} "
                f"workspace={workspace_bytes}"
            )
        self._epoch += 1
        cu_seqlens = self._empty(
            (c.expert_count + c.prefetch_slots,), self._torch.int32
        )
        plan = MoonEPPlan(
            dst=self._empty((c.route_count,), self._torch.int32),
            experts_to_copy=self._empty(
                (c.planner_group_size, c.prefetch_slots), self._torch.int32
            ),
            zero_fill_ranges=self._empty(
                (c.expert_count + c.prefetch_slots, 2), self._torch.int32
            ),
            remote_stats=self._empty((2,), self._torch.int32),
            dup_groups=self._empty((c.nv_s, 3), self._torch.int32),
            dup_loffs=self._empty((c.nv_s,), self._torch.int32),
            dup_counts=self._empty((2,), self._torch.int32),
            status=self._empty((1,), self._torch.int32),
            reduce_grad_status=self._zeros((1,), self._torch.int32),
            workspace=self._empty((workspace_bytes,), self._torch.uint8),
            n=c.route_count,
            tokens_per_rank=c.tokens_per_rank,
            topk=c.topk,
            expert_count=c.expert_count,
            rank_size=c.planner_group_size,
            prefetch_slots=c.prefetch_slots,
            nv_s=c.nv_s,
            token_padding=c.token_padding,
            epoch=self._epoch,
            backend="native",
            runtime=self.runtime,
            dst_local_offset=dst_local_offset,
        )
        self._trace_stage("planning_kernel_launch_begin")
        self.runtime.planning(
            c,
            topk_experts,
            tokens_per_expert,
            plan,
            cu_seqlens,
            self._stream_ptr(),
            self.wait_iterations,
        )
        self._trace_stage("planning_kernel_launch_end")
        self._plans_needing_dedup[id(plan)] = plan
        self._retain(plan, topk_experts, tokens_per_expert, cu_seqlens)
        self._expect_status(plan, 0)
        return plan, cu_seqlens

    def dispatch(
        self,
        hidden_sh,
        route_weights_sk=None,
        topk_experts_sk=None,
        tokens_per_expert=None,
        plan: MoonEPPlan | None = None,
        async_finish: bool = False,
        *,
        inter_rank_sync: bool = True,
        zero_copy: bool = False,
    ):
        self._require_open()
        if zero_copy:
            raise NotImplementedError("TileXR MoonEP does not support zero_copy=True")
        if not bool(inter_rank_sync):
            raise NotImplementedError(
                "TileXR MoonEP does not support inter_rank_sync=False; "
                "peer protocol synchronization is required"
            )
        inline_plan = plan is None
        if inline_plan:
            if topk_experts_sk is None or tokens_per_expert is None:
                raise ValueError(
                    "topk_experts_sk and tokens_per_expert are required when plan is None"
                )
            plan, cu_seqlens = self.planning(topk_experts_sk, tokens_per_expert)
        else:
            self._validate_plan(plan)
            cu_seqlens = None
        build_dedup = self._plans_needing_dedup.get(id(plan)) is plan
        c = self.context
        _validate_tensor(
            self._torch,
            hidden_sh,
            "hidden_sh",
            dtype=c.dtype,
            shape=(c.tokens_per_rank, c.hidden_size),
            device_index=c.device_index,
        )
        if route_weights_sk is not None:
            _validate_tensor(
                self._torch,
                route_weights_sk,
                "route_weights_sk",
                dtype=self._torch.float32,
                shape=(c.tokens_per_rank, c.topk),
                device_index=c.device_index,
            )
        hidden_nvsh = self._empty((c.nv_s, c.hidden_size), c.dtype)
        route_weights_nvs = (
            self._zeros((c.nv_s,), self._torch.float32)
            if route_weights_sk is not None
            else None
        )
        stream_ptr = self._stream_ptr()
        self._trace_stage("planning_sync_begin")
        self._synchronize_device()
        self._trace_stage("planning_sync_end")
        if build_dedup:
            planner_status = int(plan.status.item())
            self._trace_stage(f"planning_status={planner_status}")
            if os.environ.get("TILEXR_MOONEP_TRACE_STAGES", "0") == "1":
                copy_count = int((plan.experts_to_copy >= 0).sum().item())
                self._trace_stage(f"planning_experts_to_copy_nonnegative={copy_count}")
            if planner_status != 0:
                raise RuntimeError(f"MoonEP Planner device status is {planner_status}")
        self.context.activate_dispatch_workspace()
        self._retain(
            plan, hidden_sh, hidden_nvsh, route_weights_sk, route_weights_nvs, cu_seqlens
        )
        self._trace_stage("dispatch_kernel_launch_begin")
        self.runtime.dispatch(
            c,
            plan,
            hidden_sh,
            hidden_nvsh,
            stream_ptr,
            route_weights_sk,
            route_weights_nvs,
            build_dedup=False,
            inter_rank_sync=bool(inter_rank_sync),
            registered_workspace=self.context.dispatch_workspace[0],
            registered_workspace_bytes=self.context.dispatch_workspace[1],
        )
        self._trace_stage("dispatch_kernel_launch_end")
        if build_dedup:
            self._plans_needing_dedup.pop(id(plan), None)
        self._expect_status(plan, 0)
        result = (hidden_nvsh, route_weights_nvs, cu_seqlens, plan)
        if async_finish:
            return (*result, self._record_event())
        return result

    def _validate_projection_buffers(self, buffers: ProjectionBuffers, *, reduce: bool) -> None:
        if not isinstance(buffers, ProjectionBuffers):
            raise TypeError("projection buffers must be ProjectionBuffers")
        c = self.context
        dtype = self._torch.float32 if reduce else c.dtype
        for name in ("gate", "up", "down"):
            tensor = getattr(buffers, name)
            _validate_tensor(
                self._torch,
                tensor,
                f"projection.{name}",
                dtype=dtype,
                device_index=c.device_index,
                allow_storage_offset=not reduce,
            )
            tensor_shape = _shape(tensor)
            if len(tensor_shape) < 2 or len(tensor_shape) > 4:
                raise ValueError(f"projection.{name} rank must be in [2, 4]")
            if any(value <= 0 for value in tensor_shape[1:]):
                raise ValueError(f"projection.{name} must be non-empty")
            expected_rows = (
                c.expert_count + c.prefetch_slots
                if reduce else 2 * c.experts_per_rank
            )
            if tensor_shape[0] != expected_rows:
                raise ValueError(
                    f"projection.{name} first dimension must be {expected_rows}"
                )
            row_bytes = int(tensor.numel()) * int(tensor.element_size()) // tensor_shape[0]
            if not reduce and (row_bytes <= 0 or row_bytes % 64):
                raise ValueError(f"projection.{name} expert rows must be 64-byte aligned")
            if not reduce and int(tensor.data_ptr()) % 64:
                raise ValueError(f"projection.{name} must be 64-byte aligned")
        if not reduce:
            return
        legacy_buffers = [
            getattr(buffers, name) for name in ("gate_reduce", "up_reduce", "down_reduce")
        ]
        if all(value is None for value in legacy_buffers):
            return
        if any(value is None for value in legacy_buffers):
            raise ValueError("legacy ReduceGrad buffers must be all provided or all omitted")
        for full_name, reduce_name in (
            ("gate", "gate_reduce"),
            ("up", "up_reduce"),
            ("down", "down_reduce"),
        ):
            full = getattr(buffers, full_name)
            reduced = getattr(buffers, reduce_name)
            expected = (
                c.planner_group_size,
                c.prefetch_slots,
                *_shape(full)[1:],
            )
            _validate_tensor(
                self._torch,
                reduced,
                f"projection.{reduce_name}",
                dtype=self._torch.float32,
                shape=expected,
                device_index=c.device_index,
            )

    def register_projection_buffers(self, projections: ProjectionBuffers) -> None:
        self._require_open()
        self._validate_projection_buffers(projections, reduce=False)
        if projections.backing is None:
            raise ValueError("projection buffers require one flat backing allocation")
        if self._registered_projections is projections and projections.udma_handle is not None:
            return
        if self._registered_projections is not None:
            raise RuntimeError("only one projection allocation can be registered per buffer")
        self._synchronize_device()
        projections.udma_handle = self.runtime.udma_register(projections.backing)
        self._registered_projections = projections

    def prefetch_weight(
        self,
        plan: MoonEPPlan,
        projections: ProjectionBuffers,
        async_finish: bool = False,
    ):
        self._require_open()
        self._validate_plan(plan)
        self._validate_projection_buffers(projections, reduce=False)
        if self._registered_projections is not projections or projections.udma_handle is None:
            raise RuntimeError("projection buffers must be registered before PrefetchWeight")
        self._synchronize_device()
        projections.udma_handle = self.runtime.udma_register(projections.backing)
        self._retain(plan, projections)
        self.runtime.prefetch_weight(
            self.context, plan, projections, self._stream_ptr()
        )
        self._expect_status(plan, _PREFETCH_WEIGHT_STATUS_SUCCESS)
        return self._record_event() if async_finish else projections

    def combine(
        self,
        plan: MoonEPPlan | None = None,
        hidden_nvsh=None,
        route_weights_nvs=None,
        async_finish: bool = False,
        inter_rank_sync: bool = True,
        *,
        zero_copy: bool = False,
        phase_barrier: Callable[[str], None] | None = None,
    ):
        self._require_open()
        if zero_copy:
            raise NotImplementedError("TileXR MoonEP does not support zero_copy=True")
        if not bool(inter_rank_sync):
            raise NotImplementedError(
                "TileXR MoonEP does not support inter_rank_sync=False; "
                "peer protocol synchronization is required"
            )
        self._validate_plan(plan)
        c = self.context
        _validate_tensor(
            self._torch,
            hidden_nvsh,
            "hidden_nvsh",
            dtype=c.dtype,
            shape=(c.nv_s, c.hidden_size),
            device_index=c.device_index,
        )
        if route_weights_nvs is not None:
            _validate_tensor(
                self._torch,
                route_weights_nvs,
                "route_weights_nvs",
                dtype=self._torch.float32,
                shape=(c.nv_s,),
                device_index=c.device_index,
            )
        del phase_barrier
        self._trace_stage("combine_workspace_activation_begin")
        self._synchronize_device()
        self.context.activate_dispatch_workspace()
        self._trace_stage("combine_workspace_activation_end")
        hidden_sh = self._empty((c.tokens_per_rank, c.hidden_size), c.dtype)
        route_weights_sk = (
            self._empty((c.tokens_per_rank, c.topk), self._torch.float32)
            if route_weights_nvs is not None
            else None
        )
        self._retain(plan, hidden_nvsh, hidden_sh, route_weights_nvs, route_weights_sk)
        self._trace_stage("combine_v2_launch_begin")
        self.runtime.combine(
            c,
            plan,
            hidden_nvsh,
            hidden_sh,
            self._stream_ptr(),
            route_weights_nvs,
            route_weights_sk,
            inter_rank_sync=bool(inter_rank_sync),
            flags=0,
            registered_workspace=self.context.dispatch_workspace[0],
            registered_workspace_bytes=self.context.dispatch_workspace[1],
        )
        self._trace_stage("combine_v2_launch_end")
        event = self._record_event() if async_finish else None
        return hidden_sh, route_weights_sk, event

    def prepare_reduce_grad(
        self,
        plan: MoonEPPlan,
        gradients: ProjectionBuffers,
        *,
        _owner_token: object | None = None,
    ):
        self._require_open()
        self._validate_plan(plan)
        self._validate_projection_buffers(gradients, reduce=True)
        shape_signature = (
            tuple(_shape(getattr(gradients, name)) for name in ("gate", "up", "down")),
            self.requested_udma_chunk_bytes,
        )
        if self._reduce_grad_signature == shape_signature:
            return self._reduce_grad_info

        info = self.runtime.reduce_grad_workspace_info(
            self.context,
            plan,
            gradients,
            requested_udma_chunk_bytes=self.requested_udma_chunk_bytes,
        )
        if self._reduce_grad_signature is not None:
            if not self._quiesced:
                self.synchronize()
            self.runtime.unregister_reduce_grad_workspace(owner_token=_owner_token)
            self._reduce_grad_workspace = None
            self._reduce_grad_workspace_allocation = None
            self._reduce_grad_signature = None
            self._reduce_grad_info = None

        if info.uses_udma:
            if not self._quiesced:
                self.synchronize()
            workspace, allocation = self._aligned_workspace(
                info.workspace_bytes, info.workspace_alignment
            )
            self.quiesce()
            self.runtime.register_reduce_grad_workspace(
                workspace, info.workspace_bytes, owner_token=_owner_token
            )
            self._reduce_grad_workspace = workspace
            self._reduce_grad_workspace_allocation = allocation
        else:
            self._reduce_grad_workspace = None
            self._reduce_grad_workspace_allocation = None
        self._reduce_grad_signature = shape_signature
        self._reduce_grad_info = info
        return info

    @property
    def reduce_grad_info(self):
        return self._reduce_grad_info

    def reduce_grad(
        self,
        plan: MoonEPPlan | None = None,
        async_finish: bool = False,
        full_gate_grad=None,
        full_up_grad=None,
        full_down_grad=None,
        gate_reduce_buffer=None,
        up_reduce_buffer=None,
        down_reduce_buffer=None,
    ):
        self._require_open()
        self._validate_plan(plan)
        gradients = ProjectionBuffers(
            full_gate_grad,
            full_up_grad,
            full_down_grad,
            gate_reduce_buffer,
            up_reduce_buffer,
            down_reduce_buffer,
        )
        self._validate_projection_buffers(gradients, reduce=True)
        legacy_names = ("gate_reduce", "up_reduce", "down_reduce")
        legacy_buffers = [getattr(gradients, name) for name in legacy_names]
        if all(value is not None for value in legacy_buffers):
            begin = self.context.expert_count
            end = begin + self.context.prefetch_slots
            for full_name, reduce_name in zip(("gate", "up", "down"), legacy_names):
                getattr(gradients, full_name)[begin:end].copy_(
                    getattr(gradients, reduce_name)[self.context.planner_group_rank]
                )

        self.runtime._acquire_reduce_grad(self._reduce_grad_owner_token)
        self._reduce_grad_token_held = True
        retained = False
        try:
            self.prepare_reduce_grad(
                plan, gradients, _owner_token=self._reduce_grad_owner_token
            )
            if self._reduce_grad_info is not None and self._reduce_grad_info.uses_udma:
                self._synchronize_device()
                self.runtime.register_reduce_grad_workspace(
                    self._reduce_grad_workspace,
                    self._reduce_grad_info.workspace_bytes,
                    owner_token=self._reduce_grad_owner_token,
                )
            self._retain(plan, gradients, self._reduce_grad_workspace)
            retained = True
            if all(existing is not plan for existing in self._pending_reduce_plans):
                self._pending_reduce_plans.append(plan)
            self.runtime.reduce_grad(
                self.context,
                plan,
                gradients,
                self._reduce_grad_workspace,
                self._stream_ptr(),
                self.wait_iterations,
                requested_udma_chunk_bytes=self.requested_udma_chunk_bytes,
            )
            self._reduce_grad_inflight = True
            if all(value is not None for value in legacy_buffers):
                for value in legacy_buffers:
                    value[self.context.planner_group_rank].zero_()
            return self._record_event() if async_finish else None
        except Exception:
            if retained:
                self._quiesced = False
            if not self._reduce_grad_inflight and self._reduce_grad_token_held:
                self.runtime._release_reduce_grad(self._reduce_grad_owner_token)
                self._reduce_grad_token_held = False
            raise

    @staticmethod
    def _unwrap_expert_result(value) -> tuple[Any, Any]:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("expert callback tuple must be (tensor, cache)")
            return value
        return value, None

    def forward(
        self,
        *,
        hidden,
        route_weights,
        topk_experts,
        tokens_per_expert,
        projections: ProjectionBuffers,
        expert_forward: Callable[[Any, MoonEPPlan, ProjectionBuffers], Any],
        apply_route_weights: bool = True,
    ) -> ForwardResult:
        hidden_nvsh, route_weights_nvs, _, plan = self.dispatch(
            hidden,
            route_weights,
            topk_experts,
            tokens_per_expert,
        )
        self.prefetch_weight(plan, projections)
        expert_output, expert_cache = self._unwrap_expert_result(
            expert_forward(hidden_nvsh, plan, projections)
        )
        if apply_route_weights and route_weights_nvs is not None:
            expert_output = expert_output * route_weights_nvs.reshape(
                self.context.nv_s, 1
            )
        hidden_sh, route_weights_sk, _ = self.combine(
            plan, expert_output, route_weights_nvs
        )
        return ForwardResult(
            hidden=hidden_sh,
            route_weights=route_weights_sk,
            state=ForwardState(
                plan=plan,
                projections=projections,
                dispatched_route_weights=route_weights_nvs,
                apply_route_weights=bool(apply_route_weights),
                expert_cache=expert_cache,
            ),
        )

    def backward(
        self,
        *,
        grad_output,
        state: ForwardState,
        expert_backward: Callable[[Any, ForwardState], Any],
        gradients: ProjectionBuffers,
    ) -> BackwardResult:
        self._validate_plan(state.plan)
        dispatched_grad, _, _, _ = self.dispatch(grad_output, plan=state.plan)
        if state.apply_route_weights and state.dispatched_route_weights is not None:
            dispatched_grad = dispatched_grad * state.dispatched_route_weights.reshape(
                self.context.nv_s, 1
            ).to(dtype=dispatched_grad.dtype)
        grad_expert_result = expert_backward(dispatched_grad, state)
        if isinstance(grad_expert_result, tuple):
            if len(grad_expert_result) != 2:
                raise ValueError("expert backward tuple must be (grad_hidden, gradients)")
            grad_expert_hidden, gradients = grad_expert_result
        else:
            grad_expert_hidden = grad_expert_result
        grad_hidden, _, _ = self.combine(state.plan, grad_expert_hidden)
        self.reduce_grad(
            state.plan,
            full_gate_grad=gradients.gate,
            full_up_grad=gradients.up,
            full_down_grad=gradients.down,
            gate_reduce_buffer=gradients.gate_reduce,
            up_reduce_buffer=gradients.up_reduce,
            down_reduce_buffer=gradients.down_reduce,
        )
        return BackwardResult(grad_hidden=grad_hidden, plan=state.plan)

    def quiesce(self) -> None:
        """Wait until this process has no active work on the bound NPU device."""
        self._require_open()
        try:
            self._torch.npu.synchronize(self.context.device_index)
        except TypeError:
            if int(self._torch.npu.current_device()) != self.context.device_index:
                raise RuntimeError(
                    "torch.npu.synchronize() cannot target the buffer device in this "
                    "Torch-NPU version"
                )
            self._torch.npu.synchronize()
        self._quiesced = True

    def check_pending_status(self) -> None:
        """Release pending references after a successful quiesce boundary."""
        self._require_open()
        if not self._quiesced:
            raise RuntimeError("check_pending_status requires a successful quiesce")
        try:
            statuses = []
            for plan in self._pending_plans:
                expected = self._pending_statuses.get(id(plan))
                if expected is None:
                    continue
                statuses.append(
                    (
                        plan.epoch,
                        int(plan.status.item()),
                        expected,
                    )
                )
            for plan in self._pending_reduce_plans:
                statuses.append((plan.epoch, int(plan.reduce_grad_status.item()), 0))
            self._pending_refs.clear()
            self._pending_plans.clear()
            self._pending_reduce_plans.clear()
            self._pending_statuses.clear()
            self._bound_stream_ptr = None
            self.context.release_stream(self)
            failed = [
                (epoch, status, expected)
                for epoch, status, expected in statuses
                if status != expected
            ]
            if failed:
                if any(status in (2005, 2006, 2007) for _, status, _ in failed):
                    self.context.mark_poisoned()
                details = ", ".join(
                    f"epoch {epoch}: actual {status}, expected {expected}"
                    for epoch, status, expected in failed
                )
                raise RuntimeError(f"MoonEP device status failures: {details}")
        finally:
            if self._reduce_grad_inflight:
                self.runtime._release_reduce_grad(self._reduce_grad_owner_token)
                self._reduce_grad_inflight = False
                self._reduce_grad_token_held = False

    def synchronize(self) -> None:
        self.quiesce()
        self.check_pending_status()

    def close(self) -> None:
        if self._closed:
            return
        self.quiesce()
        sync_error = None
        try:
            self.check_pending_status()
        except Exception as exc:
            sync_error = exc
        try:
            try:
                self.runtime.unregister_reduce_grad_workspace()
            finally:
                try:
                    if self._registered_projections is not None:
                        projections = self._registered_projections
                        if projections.udma_handle is not None:
                            self.runtime.udma_unregister(projections.udma_handle)
                            projections.udma_handle = None
                        self._registered_projections = None
                finally:
                    self.context.detach_buffer(self)
                    self.context.close()
        finally:
            self._closed = True
            self._pending_refs.clear()
            self._pending_plans.clear()
            self._pending_reduce_plans.clear()
            self._pending_statuses.clear()
            self._plans_needing_dedup.clear()
            self._registered_projections = None
            self._reduce_grad_workspace = None
            self._reduce_grad_workspace_allocation = None
        if sync_error is not None:
            raise sync_error

    def __enter__(self) -> "TileXRMoonEPBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
