from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol, runtime_checkable


CORRECTNESS_STAGES = (
    "planning",
    "dispatch",
    "prefetch_weight",
    "expert_forward",
    "combine",
    "reduce_grad",
)


class ContractError(ValueError):
    """A backend value does not satisfy the normalized MoonEP contract."""


class BackendUnavailableError(RuntimeError):
    """A requested correctness backend is not installed or is incomplete."""


@dataclass(frozen=True, slots=True)
class MoonEPDimensions:
    rank: int
    world_size: int
    tokens_per_rank: int
    topk: int
    expert_count: int
    prefetch_slots: int
    token_padding: int
    hidden_size: int
    intermediate_size: int

    def __post_init__(self) -> None:
        positive = (
            "world_size",
            "tokens_per_rank",
            "topk",
            "expert_count",
            "prefetch_slots",
            "token_padding",
            "hidden_size",
            "intermediate_size",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ContractError(f"{name} must be positive")
        if not 0 <= int(self.rank) < int(self.world_size):
            raise ContractError("rank must be in [0, world_size)")
        if self.expert_count % self.world_size != 0:
            raise ContractError("expert_count must be divisible by world_size")
        if self.topk > self.expert_count:
            raise ContractError("topk cannot exceed expert_count")
        if self.topk > 32:
            raise ContractError("topk cannot exceed upstream kmask limit 32")
        if self.prefetch_slots > self.experts_per_rank:
            raise ContractError("prefetch_slots cannot exceed experts_per_rank")
        if self.world_size * self.nvsh > 2**31 - 1:
            raise ContractError("encoded destination range exceeds signed int32")

    @property
    def route_count(self) -> int:
        return self.tokens_per_rank * self.topk

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    @property
    def vm_group_count(self) -> int:
        return self.expert_count + self.prefetch_slots

    @property
    def nvsh(self) -> int:
        return self.route_count + (
            (self.token_padding - 1) * 2 * self.experts_per_rank
        )


@dataclass(slots=True)
class DedupPlan:
    groups: Any
    loffs: Any
    counts: Any

    def clone(self) -> "DedupPlan":
        return DedupPlan(self.groups.clone(), self.loffs.clone(), self.counts.clone())


@dataclass(slots=True)
class MoonEPPlan:
    dimensions: MoonEPDimensions
    dst: Any
    cu_seqlens: Any
    experts_to_copy: Any
    zero_fill_ranges: Any
    remote_stats: Any
    dedup: DedupPlan | None = None

    def clone(self) -> "MoonEPPlan":
        return MoonEPPlan(
            dimensions=self.dimensions,
            dst=self.dst.clone(),
            cu_seqlens=self.cu_seqlens.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dedup=None if self.dedup is None else self.dedup.clone(),
        )


@dataclass(slots=True)
class ProjectionTensors:
    gate: Any
    up: Any
    down: Any

    def clone(self) -> "ProjectionTensors":
        return ProjectionTensors(self.gate.clone(), self.up.clone(), self.down.clone())

    def items(self):
        return (("gate", self.gate), ("up", self.up), ("down", self.down))


@dataclass(slots=True)
class PlanningResult:
    plan: MoonEPPlan


@dataclass(slots=True)
class DispatchResult:
    hidden: Any
    route_weights: Any | None
    plan: MoonEPPlan


@dataclass(slots=True)
class PrefetchResult:
    projections: ProjectionTensors


@dataclass(slots=True)
class ExpertForwardResult:
    hidden: Any


@dataclass(slots=True)
class CombineResult:
    hidden: Any
    route_weights: Any | None


@dataclass(slots=True)
class ReduceGradResult:
    full_grads: ProjectionTensors
    reduce_buffers: ProjectionTensors


@dataclass(slots=True)
class CanonicalMoonEPCase:
    case_id: str
    dimensions: MoonEPDimensions
    topk_experts: Any
    tokens_per_expert: Any
    hidden: Any
    route_weights: Any
    projections: ProjectionTensors
    full_grads: ProjectionTensors
    reduce_buffers: ProjectionTensors

    def clone(self) -> "CanonicalMoonEPCase":
        return CanonicalMoonEPCase(
            case_id=self.case_id,
            dimensions=self.dimensions,
            topk_experts=self.topk_experts.clone(),
            tokens_per_expert=self.tokens_per_expert.clone(),
            hidden=self.hidden.clone(),
            route_weights=self.route_weights.clone(),
            projections=self.projections.clone(),
            full_grads=self.full_grads.clone(),
            reduce_buffers=self.reduce_buffers.clone(),
        )


@runtime_checkable
class MoonEPBackend(Protocol):
    name: str
    dimensions: MoonEPDimensions

    def planning(self, topk_experts, tokens_per_expert) -> PlanningResult: ...

    def dispatch(
        self, plan: MoonEPPlan, hidden_sh, route_weights_sk=None
    ) -> DispatchResult: ...

    def prefetch_weight(
        self, plan: MoonEPPlan, projections: ProjectionTensors
    ) -> PrefetchResult: ...

    def combine(
        self, plan: MoonEPPlan, expert_output_nvsh, route_weights_nvs=None
    ) -> CombineResult: ...

    def reduce_grad(
        self,
        plan: MoonEPPlan,
        full_grads: ProjectionTensors,
        reduce_buffers: ProjectionTensors,
    ) -> ReduceGradResult: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


def clone_value(value):
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return type(value)(
            **{field.name: clone_value(getattr(value, field.name)) for field in fields(value)}
        )
    raise TypeError(f"cannot clone value of type {type(value).__name__}")


def validate_tensor(
    tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype,
    device_type: str | None = None,
) -> None:
    if tensor is None:
        raise ContractError(f"{name} is required")
    actual_shape = tuple(int(value) for value in tensor.shape)
    if actual_shape != tuple(shape):
        raise ContractError(f"{name} must have shape {shape}, got {actual_shape}")
    if tensor.dtype != dtype:
        raise ContractError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ContractError(f"{name} must be contiguous")
    if int(tensor.storage_offset()) != 0:
        raise ContractError(f"{name} must have storage_offset 0")
    if device_type is not None and tensor.device.type != device_type:
        raise ContractError(
            f"{name} must be on {device_type}, got {tensor.device.type}"
        )


def validate_plan(plan: MoonEPPlan, torch_module, *, require_dedup: bool) -> None:
    if not isinstance(plan, MoonEPPlan):
        raise ContractError("plan must be a MoonEPPlan")
    d = plan.dimensions
    validate_tensor(plan.dst, "plan.dst", shape=(d.route_count,), dtype=torch_module.int32)
    validate_tensor(
        plan.cu_seqlens,
        "plan.cu_seqlens",
        shape=(d.vm_group_count,),
        dtype=torch_module.int32,
    )
    validate_tensor(
        plan.experts_to_copy,
        "plan.experts_to_copy",
        shape=(d.world_size, d.prefetch_slots),
        dtype=torch_module.int32,
    )
    validate_tensor(
        plan.zero_fill_ranges,
        "plan.zero_fill_ranges",
        shape=(d.vm_group_count, 2),
        dtype=torch_module.int32,
    )
    validate_tensor(
        plan.remote_stats,
        "plan.remote_stats",
        shape=(2,),
        dtype=torch_module.int32,
    )
    if require_dedup and plan.dedup is None:
        raise ContractError("plan dedup metadata is not ready")
    if plan.dedup is not None:
        validate_tensor(
            plan.dedup.groups,
            "plan.dedup.groups",
            shape=(d.nvsh, 3),
            dtype=torch_module.int32,
        )
        validate_tensor(
            plan.dedup.loffs,
            "plan.dedup.loffs",
            shape=(d.nvsh,),
            dtype=torch_module.int32,
        )
        validate_tensor(
            plan.dedup.counts,
            "plan.dedup.counts",
            shape=(2,),
            dtype=torch_module.int32,
        )


def validate_projections(
    projections: ProjectionTensors,
    dimensions: MoonEPDimensions,
    torch_module,
    *,
    dtype,
    reduce_buffers: bool = False,
) -> None:
    if not isinstance(projections, ProjectionTensors):
        raise ContractError("projections must be ProjectionTensors")
    prefix = (
        (dimensions.world_size, dimensions.prefetch_slots)
        if reduce_buffers
        else (dimensions.expert_count + dimensions.prefetch_slots,)
    )
    shapes = {
        "gate": prefix + (dimensions.hidden_size, dimensions.intermediate_size),
        "up": prefix + (dimensions.hidden_size, dimensions.intermediate_size),
        "down": prefix + (dimensions.intermediate_size, dimensions.hidden_size),
    }
    for name, tensor in projections.items():
        validate_tensor(tensor, name, shape=shapes[name], dtype=dtype)
