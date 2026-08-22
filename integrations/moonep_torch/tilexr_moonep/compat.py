from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

from .torch_api import (
    MoonEPPlan as NativeMoonEPPlan,
    TileXRMoonEPBuffer,
    TileXRMoonEPContext,
)


def _torch():
    import torch

    return torch


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AssertionError(f"{name} must be a positive int, got {value}")
    return int(value)


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


class _CompletionEvent:
    def __init__(self, event, on_wait):
        self._event = event
        self._on_wait = on_wait

    def _complete_once(self) -> None:
        if self._on_wait is not None:
            callback = self._on_wait
            self._on_wait = None
            callback()

    def wait(self, *args, **kwargs):
        result = self._event.wait(*args, **kwargs)
        self._complete_once()
        return result

    def synchronize(self):
        result = self._event.synchronize()
        self._complete_once()
        return result

    def __getattr__(self, name):
        return getattr(self._event, name)


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    dst: Any
    experts_to_copy: Any
    zero_fill_ranges: Any
    remote_stats: Any
    N: int
    R: int
    E: int
    B: int
    NvS: int
    K: int
    dup_groups: Any
    dup_loffs: Any
    dup_counts: Any
    _native_plan: NativeMoonEPPlan | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def _from_native(cls, native: NativeMoonEPPlan) -> "MoonEPCommPlan":
        if not isinstance(native, NativeMoonEPPlan):
            raise TypeError("native plan must be a tilexr_moonep MoonEPPlan")
        value = cls(
            dst=native.dst,
            experts_to_copy=native.experts_to_copy,
            zero_fill_ranges=native.zero_fill_ranges,
            remote_stats=native.remote_stats,
            N=int(native.n),
            R=int(native.rank_size),
            E=int(native.expert_count),
            B=int(native.prefetch_slots),
            NvS=int(native.nv_s),
            K=int(native.topk),
            dup_groups=native.dup_groups,
            dup_loffs=native.dup_loffs,
            dup_counts=native.dup_counts,
        )
        object.__setattr__(value, "_native_plan", native)
        return value

    def _require_native(self) -> NativeMoonEPPlan:
        if self._native_plan is None:
            raise ValueError("MoonEPCommPlan was not created by this TileXR Buffer")
        return self._native_plan

    def clone(self) -> "MoonEPCommPlan":
        native = self._require_native()
        cloned_tensors = {
            name: getattr(self, name).clone()
            for name in (
                "dst",
                "experts_to_copy",
                "zero_fill_ranges",
                "remote_stats",
                "dup_groups",
                "dup_loffs",
                "dup_counts",
            )
        }
        cloned_native = replace(
            native,
            **cloned_tensors,
            status=native.status.clone(),
            reduce_grad_status=native.reduce_grad_status.clone(),
            workspace=native.workspace.clone(),
        )
        return type(self)._from_native(cloned_native)


class Buffer:
    """MoonEP-compatible communication facade backed by TileXR.

    ExpertForward remains caller-owned, matching upstream MoonEP. TileXR's optional
    Torch-NPU GMM helper is intentionally not exposed as a Buffer method.
    """

    def __init__(
        self,
        S: int,
        H: int,
        K: int,
        E: int,
        num_ep_ranks: int,
        num_sms: int | None = None,
        token_padding: int = 128,
        B: int | None = None,
        group: Any = None,
        comm_stream_priority: int = -1,
        enable_pdl: bool = True,
        explicitly_destroy: bool = False,
    ):
        self.S = _require_positive_int("S", S)
        self.H = _require_positive_int("H", H)
        self.K = _require_positive_int("K", K)
        self.E = _require_positive_int("E", E)
        self.R = _require_positive_int("num_ep_ranks", num_ep_ranks)
        if self.E % self.R:
            raise AssertionError(f"E ({self.E}) must be divisible by R ({self.R})")
        self.num_sms = 32 if num_sms is None else _require_positive_int("num_sms", num_sms)
        self.token_padding = _require_positive_int("token_padding", token_padding)
        self.B = self.E // self.R if B is None else _require_positive_int("B", B)
        if self.B > self.E // self.R:
            raise AssertionError(f"B must be in [1, E/R], got {self.B}")
        if isinstance(comm_stream_priority, bool) or not isinstance(comm_stream_priority, int):
            raise AssertionError(
                "comm_stream_priority must be an int, got "
                f"{type(comm_stream_priority).__name__}"
            )
        if not isinstance(enable_pdl, bool):
            raise AssertionError(
                f"enable_pdl must be a bool, got {type(enable_pdl).__name__}"
            )
        if not isinstance(explicitly_destroy, bool):
            raise AssertionError(
                "explicitly_destroy must be a bool, got "
                f"{type(explicitly_destroy).__name__}"
            )

        self._torch = _torch()
        self.group = group
        self._group_rank = None
        self._group_world_size = None
        distributed = getattr(self._torch, "distributed", None)
        if distributed is not None:
            if not distributed.is_initialized():
                raise AssertionError(
                    "torch.distributed must be initialized before Buffer construction"
                )
            self._group_rank = int(distributed.get_rank(group=group))
            self._group_world_size = int(distributed.get_world_size(group=group))
            if self._group_world_size != self.R:
                raise AssertionError(
                    f"num_ep_ranks ({self.R}) must equal group world size "
                    f"({self._group_world_size})"
                )
        self.comm_stream_priority = int(comm_stream_priority)
        self.enable_pdl = bool(enable_pdl)
        self.explicitly_destroy = bool(explicitly_destroy)
        self._destroyed = True
        self._zero_copy_aliases = None
        self._context = TileXRMoonEPContext.from_env(
            tokens_per_rank=self.S,
            hidden_size=self.H,
            topk=self.K,
            expert_count=self.E,
            dtype=self._torch.bfloat16,
            token_padding=self.token_padding,
            prefetch_slots=self.B,
            torch_module=self._torch,
        )
        if (
            self._group_rank is not None
            and (
                self._context.planner_group_rank != self._group_rank
                or self._context.planner_group_size != self._group_world_size
            )
        ):
            self._context.close()
            raise AssertionError(
                "TileXR planner rank/world environment does not match the process group"
            )
        try:
            self._native_buffer = TileXRMoonEPBuffer(
                self._context,
                wait_iterations=_env_positive_int(
                    "TILEXR_MOONEP_PLANNER_WAIT_ITERATIONS", 1_000_000
                ),
                torch_module=self._torch,
            )
        except Exception:
            self._context.close()
            raise
        self._destroyed = False

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def _require_open(self) -> None:
        if self._destroyed:
            raise AssertionError("MoonEP Buffer has been destroyed")

    def _require_ctx(self) -> dict[str, Any]:
        self._require_open()
        aliases = self._zero_copy_aliases
        return {
            "S": self.S,
            "H": self.H,
            "K": self.K,
            "E": self.E,
            "R": self.R,
            "B": self.B,
            "N": self.S * self.K,
            "NvS": self._context.nv_s,
            "hidden_buf_local": None if aliases is None else aliases[1],
            "weights_buf_local": None if aliases is None else aliases[2],
        }

    def dispatch(
        self,
        hidden_sh,
        route_weights_sk=None,
        topk_experts_sk=None,
        tokens_per_expert=None,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        *,
        inter_rank_sync: bool = True,
        zero_copy: bool = False,
    ):
        self._require_open()
        self._zero_copy_aliases = None
        native_plan = None
        if plan is not None:
            if not isinstance(plan, MoonEPCommPlan):
                raise TypeError("plan must be a MoonEPCommPlan")
            native_plan = plan._require_native()

        result = self._native_buffer.dispatch(
            hidden_sh,
            route_weights_sk,
            topk_experts_sk,
            tokens_per_expert,
            plan=native_plan,
            async_finish=bool(async_finish),
            inter_rank_sync=bool(inter_rank_sync),
            # TileXR writes directly into the returned output allocation, so the
            # compatibility facade does not request the native facade's rejected
            # boundary-copy mode.
            zero_copy=False,
        )
        if async_finish:
            hidden_nvsh, route_weights_nvs, cu_seqlens, returned_native, event = result
        else:
            hidden_nvsh, route_weights_nvs, cu_seqlens, returned_native = result

        if plan is None:
            public_plan = MoonEPCommPlan._from_native(returned_native)
        else:
            if returned_native is not native_plan:
                raise RuntimeError("native Dispatch did not echo the saved plan")
            public_plan = plan
        if zero_copy:
            self._zero_copy_aliases = (
                returned_native,
                hidden_nvsh,
                route_weights_nvs,
            )

        public_result = (
            hidden_nvsh,
            route_weights_nvs,
            cu_seqlens,
            public_plan,
        )
        if async_finish:
            return (*public_result, event)
        return public_result

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

    def _validate_full_weight(self, tensor, name: str) -> tuple[int, ...]:
        if tensor is None:
            raise AssertionError("prefetch_weight tensors must be provided together")
        shape = tuple(int(value) for value in tensor.shape)
        expected_rows = self.E + self.B
        if (
            tensor.dtype != self._torch.bfloat16
            or not tensor.is_contiguous()
            or len(shape) != 3
            or shape[0] != expected_rows
        ):
            raise AssertionError(
                f"{name} must be contiguous BF16 [E+B, H, H'], got {shape}"
            )
        if tensor.device.type != "npu":
            raise AssertionError(f"{name} must be on an NPU device")
        device_index = tensor.device.index
        if device_index is None:
            device_index = self._torch.npu.current_device()
        if int(device_index) != int(self._context.device_index):
            raise AssertionError(
                f"{name} must be on npu:{self._context.device_index}"
            )
        if int(tensor.storage_offset()) != 0:
            raise AssertionError(f"{name} must have storage_offset 0")
        return shape

    def prefetch_weight(
        self,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        *,
        full_gate_weight=None,
        full_up_weight=None,
        full_down_weight=None,
    ):
        self._require_open()
        if not isinstance(plan, MoonEPCommPlan):
            raise AssertionError("Buffer.prefetch_weight: plan is required")
        plan._require_native()
        full_weights = (full_gate_weight, full_up_weight, full_down_weight)
        tuple(
            self._validate_full_weight(value, name)
            for value, name in zip(
                full_weights,
                ("full_gate_weight", "full_up_weight", "full_down_weight"),
            )
        )
        experts = plan.experts_to_copy[
            self._context.planner_group_rank
        ].to(dtype=self._torch.int64)
        valid = experts >= 0
        safe_experts = experts.clamp_min(0)
        mask_shape = (self.B, 1, 1)
        for full in full_weights:
            slots = full.narrow(0, self.E, self.B)
            selected = full.narrow(0, 0, self.E).index_select(0, safe_experts)
            slots.copy_(
                self._torch.where(valid.reshape(mask_shape), selected, slots)
            )
        return self._record_event() if async_finish else None

    def _host_phase_barrier(self, phase: str) -> None:
        del phase
        distributed = getattr(self._torch, "distributed", None)
        if distributed is None or not distributed.is_initialized():
            raise RuntimeError(
                "cross-node Combine requires an initialized torch.distributed group"
            )
        distributed.barrier(group=self.group)

    def combine(
        self,
        plan: MoonEPCommPlan | None = None,
        hidden_nvsh=None,
        route_weights_nvs=None,
        async_finish: bool = False,
        inter_rank_sync: bool = True,
        *,
        zero_copy: bool = False,
    ):
        self._require_open()
        if not isinstance(plan, MoonEPCommPlan):
            raise AssertionError("Buffer.combine: plan is required")
        native_plan = plan._require_native()
        if hidden_nvsh is None:
            raise AssertionError("Buffer.combine: hidden_nvsh is required")

        if zero_copy:
            aliases = self._zero_copy_aliases
            if (
                aliases is None
                or aliases[0] is not native_plan
                or int(hidden_nvsh.data_ptr()) != int(aliases[1].data_ptr())
            ):
                raise AssertionError(
                    "combine(zero_copy=True): hidden_nvsh must alias the latest "
                    "zero-copy Dispatch output"
                )
            if route_weights_nvs is not None and (
                aliases[2] is None
                or int(route_weights_nvs.data_ptr()) != int(aliases[2].data_ptr())
            ):
                raise AssertionError(
                    "combine(zero_copy=True): route_weights_nvs must alias the latest "
                    "zero-copy Dispatch weights"
                )

        phase_barrier = (
            self._host_phase_barrier if int(self._context.node_count) > 1 else None
        )
        try:
            return self._native_buffer.combine(
                plan=native_plan,
                hidden_nvsh=hidden_nvsh,
                route_weights_nvs=route_weights_nvs,
                async_finish=bool(async_finish),
                inter_rank_sync=bool(inter_rank_sync),
                zero_copy=False,
                phase_barrier=phase_barrier,
            )
        finally:
            self._zero_copy_aliases = None

    def reduce_grad(
        self,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        full_gate_grad=None,
        full_up_grad=None,
        full_down_grad=None,
        gate_reduce_buffer=None,
        up_reduce_buffer=None,
        down_reduce_buffer=None,
    ):
        self._require_open()
        tensors = (
            full_gate_grad,
            full_up_grad,
            full_down_grad,
            gate_reduce_buffer,
            up_reduce_buffer,
            down_reduce_buffer,
        )
        if any(value is None for value in tensors):
            raise AssertionError("reduce_grad tensors must be provided together")
        if not isinstance(plan, MoonEPCommPlan):
            raise AssertionError("Buffer.reduce_grad: plan is required")
        native_plan = plan._require_native()
        event = self._native_buffer.reduce_grad(
            plan=native_plan,
            async_finish=bool(async_finish),
            full_gate_grad=full_gate_grad,
            full_up_grad=full_up_grad,
            full_down_grad=full_down_grad,
            gate_reduce_buffer=gate_reduce_buffer,
            up_reduce_buffer=up_reduce_buffer,
            down_reduce_buffer=down_reduce_buffer,
        )
        if async_finish:
            return _CompletionEvent(
                event,
                self._native_buffer.synchronize,
            )
        self._native_buffer.synchronize()
        return None

    def destroy(self) -> None:
        if self._destroyed:
            return
        try:
            self._native_buffer.close()
        finally:
            self._zero_copy_aliases = None
            self._destroyed = True

    def __enter__(self) -> "Buffer":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()

    def __del__(self):
        if getattr(self, "_destroyed", True):
            return
        if getattr(self, "explicitly_destroy", False):
            warnings.warn(
                "MoonEP Buffer was not destroyed explicitly; resources may leak.",
                ResourceWarning,
            )
            return
        try:
            self.destroy()
        except Exception as exc:
            warnings.warn(
                f"MoonEP Buffer destructor failed to destroy resources: {exc!r}",
                ResourceWarning,
            )
