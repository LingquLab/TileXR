from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from .runtime import TileXRMoonEPRuntime


_DISPATCH_STATUS_SUCCESS = 2000
_COMBINE_STATUS_SUCCESS = 3000
_PREFETCH_WEIGHT_STATUS_SUCCESS = 4000
_REDUCE_GRAD_STATUS_SUCCESS = 5000


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
    if int(tensor.storage_offset()) != 0:
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
            raise ValueError("prefetch_slots must be in [1, experts_per_rank]")
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
        self.runtime.close()


class TileXRMoonEPBuffer:
    """Torch facade for an explicitly orchestrated MoonEP forward/backward flow."""

    def __init__(
        self,
        context: TileXRMoonEPContext,
        *,
        wait_iterations: int = 1_000_000,
        torch_module=None,
    ):
        if wait_iterations <= 0:
            raise ValueError(f"wait_iterations must be positive, got {wait_iterations}")
        self.context = context
        self.runtime = context.runtime
        self.wait_iterations = int(wait_iterations)
        self._torch = torch_module or _torch()
        self._closed = False
        self._epoch = 0
        self._bound_stream_ptr: int | None = None
        self._pending_refs: list[tuple[Any, ...]] = []
        self._pending_plans: list[MoonEPPlan] = []
        self._pending_statuses: dict[int, int] = {}
        self._plans_needing_dedup: dict[int, MoonEPPlan] = {}
        self._quiesced = True
        if context.dtype != self._torch.bfloat16:
            raise TypeError(f"MoonEP hidden dtype must be bfloat16, got {context.dtype}")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TileXRMoonEPBuffer is closed")

    def _stream_ptr(self) -> int:
        stream_ptr = _current_stream_ptr(self._torch, self.context.device_index)
        if self._bound_stream_ptr is None:
            self._bound_stream_ptr = stream_ptr
        elif stream_ptr != self._bound_stream_ptr:
            raise RuntimeError(
                "TileXRMoonEPBuffer is bound to one NPU stream; create a separate "
                "buffer or synchronize before using another stream"
            )
        return stream_ptr

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
        workspace_bytes = int(self.runtime.planning_workspace_size(c))
        if workspace_bytes <= 0:
            raise RuntimeError(f"native planner returned invalid workspace size {workspace_bytes}")
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
        )
        self.runtime.planning(
            c,
            topk_experts,
            tokens_per_expert,
            plan,
            cu_seqlens,
            self._stream_ptr(),
            self.wait_iterations,
        )
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
            self._empty((c.nv_s,), self._torch.float32)
            if route_weights_sk is not None
            else None
        )
        self._retain(
            plan, hidden_sh, hidden_nvsh, route_weights_sk, route_weights_nvs, cu_seqlens
        )
        self.runtime.dispatch(
            c,
            plan,
            hidden_sh,
            hidden_nvsh,
            self._stream_ptr(),
            route_weights_sk,
            route_weights_nvs,
            build_dedup=build_dedup,
            inter_rank_sync=bool(inter_rank_sync),
        )
        if build_dedup:
            self._plans_needing_dedup.pop(id(plan), None)
        self._expect_status(plan, _DISPATCH_STATUS_SUCCESS)
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
            )
            tensor_shape = _shape(tensor)
            if len(tensor_shape) != 3 or tensor_shape[1] <= 0 or tensor_shape[2] <= 0:
                raise ValueError(f"projection.{name} must be a non-empty rank-3 tensor")
            if tensor_shape[0] != c.expert_count + c.prefetch_slots:
                raise ValueError(
                    f"projection.{name} first dimension must be E+B="
                    f"{c.expert_count + c.prefetch_slots}"
                )
        if not reduce:
            return
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

    def prefetch_weight(
        self,
        plan: MoonEPPlan | None = None,
        async_finish: bool = False,
        *,
        full_gate_weight=None,
        full_up_weight=None,
        full_down_weight=None,
    ):
        self._require_open()
        self._validate_plan(plan)
        projections = ProjectionBuffers(
            full_gate_weight, full_up_weight, full_down_weight
        )
        self._validate_projection_buffers(projections, reduce=False)
        self._retain(plan, projections)
        self.runtime.prefetch_weight(
            self.context, plan, projections, self._stream_ptr()
        )
        self._expect_status(plan, _PREFETCH_WEIGHT_STATUS_SUCCESS)
        return self._record_event() if async_finish else None

    def combine(
        self,
        plan: MoonEPPlan | None = None,
        hidden_nvsh=None,
        route_weights_nvs=None,
        async_finish: bool = False,
        inter_rank_sync: bool = True,
        *,
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
        hidden_sh = self._empty((c.tokens_per_rank, c.hidden_size), c.dtype)
        route_weights_sk = (
            self._empty((c.tokens_per_rank, c.topk), self._torch.float32)
            if route_weights_nvs is not None
            else None
        )
        self._retain(plan, hidden_nvsh, hidden_sh, route_weights_nvs, route_weights_sk)
        self.runtime.combine(
            c,
            plan,
            hidden_nvsh,
            hidden_sh,
            self._stream_ptr(),
            route_weights_nvs,
            route_weights_sk,
            inter_rank_sync=bool(inter_rank_sync),
        )
        self._expect_status(plan, _COMBINE_STATUS_SUCCESS)
        event = self._record_event() if async_finish else None
        return hidden_sh, route_weights_sk, event

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
        self._retain(plan, gradients)
        self.runtime.reduce_grad(self.context, plan, gradients, self._stream_ptr())
        self._expect_status(plan, _REDUCE_GRAD_STATUS_SUCCESS)
        return self._record_event() if async_finish else None

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
        self.prefetch_weight(
            plan,
            full_gate_weight=projections.gate,
            full_up_weight=projections.up,
            full_down_weight=projections.down,
        )
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
        statuses = []
        for plan in self._pending_plans:
            statuses.append(
                (
                    plan.epoch,
                    int(plan.status.item()),
                    self._pending_statuses.get(id(plan), 0),
                )
            )
        self._pending_refs.clear()
        self._pending_plans.clear()
        self._pending_statuses.clear()
        self._bound_stream_ptr = None
        failed = [
            (epoch, status, expected)
            for epoch, status, expected in statuses
            if status != expected
        ]
        if failed:
            details = ", ".join(
                f"epoch {epoch}: actual {status}, expected {expected}"
                for epoch, status, expected in failed
            )
            raise RuntimeError(f"MoonEP device status failures: {details}")

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
            self.context.close()
        finally:
            self._closed = True
            self._pending_refs.clear()
            self._pending_plans.clear()
            self._pending_statuses.clear()
            self._plans_needing_dedup.clear()
        if sync_error is not None:
            raise sync_error

    def __enter__(self) -> "TileXRMoonEPBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
