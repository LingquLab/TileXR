from __future__ import annotations

from dataclasses import dataclass
from tilexr_moonep import (
    MoonEPPlan as NativeMoonEPPlan,
    ProjectionBuffers,
    TileXRMoonEPBuffer,
    TileXRMoonEPContext,
)

from .contracts import (
    BackendUnavailableError,
    CombineResult,
    ContractError,
    DedupPlan,
    DispatchResult,
    MoonEPDimensions,
    MoonEPPlan,
    PlanningResult,
    PrefetchResult,
    ProjectionTensors,
    ReduceGradResult,
    validate_plan,
    validate_projections,
    validate_tensor,
)


_STAGES = (
    "planning",
    "dispatch",
    "prefetch_weight",
    "combine",
    "reduce_grad",
)


@dataclass
class _NativePlanEntry:
    normalized: MoonEPPlan
    native: NativeMoonEPPlan
    cu_seqlens: object
    dedup_ready: bool = False


class TileXRMoonEPBackend:
    """Normalized correctness backend backed by the native TileXR Torch facade."""

    name = "tilexr_native"

    def __init__(
        self,
        *,
        torch_module,
        dimensions: MoonEPDimensions,
        buffer: TileXRMoonEPBuffer,
    ) -> None:
        self.torch = torch_module
        self.dimensions = dimensions
        self.buffer = buffer
        self._native_plans: dict[int, _NativePlanEntry] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TileXRMoonEPBackend is closed")

    def _plan_entry(
        self, plan: MoonEPPlan, *, require_dedup: bool
    ) -> _NativePlanEntry:
        self._require_open()
        entry = self._native_plans.get(id(plan))
        if entry is None or entry.normalized is not plan:
            raise ContractError("plan is not owned by this TileXR backend")
        if plan.dimensions != self.dimensions:
            raise ContractError("plan dimensions do not match the TileXR backend")
        validate_plan(plan, self.torch, require_dedup=require_dedup)
        native = entry.native
        core = (
            ("dst", native.dst),
            ("cu_seqlens", entry.cu_seqlens),
            ("experts_to_copy", native.experts_to_copy),
            ("zero_fill_ranges", native.zero_fill_ranges),
            ("remote_stats", native.remote_stats),
        )
        for name, expected in core:
            if getattr(plan, name) is not expected:
                raise ContractError(f"plan.{name} no longer references native storage")
        if entry.dedup_ready:
            dedup = plan.dedup
            if (
                dedup is None
                or dedup.groups is not native.dup_groups
                or dedup.loffs is not native.dup_loffs
                or dedup.counts is not native.dup_counts
            ):
                raise ContractError("plan dedup no longer references native storage")
        elif plan.dedup is not None:
            raise ContractError("plan dedup metadata is not ready")
        if require_dedup and not entry.dedup_ready:
            raise ContractError("plan dedup metadata is not ready")
        return entry

    def planning(self, topk_experts, tokens_per_expert) -> PlanningResult:
        self._require_open()
        native, cu_seqlens = self.buffer.planning(topk_experts, tokens_per_expert)
        plan = MoonEPPlan(
            dimensions=self.dimensions,
            dst=native.dst,
            cu_seqlens=cu_seqlens,
            experts_to_copy=native.experts_to_copy,
            zero_fill_ranges=native.zero_fill_ranges,
            remote_stats=native.remote_stats,
            dedup=None,
        )
        self._native_plans[id(plan)] = _NativePlanEntry(plan, native, cu_seqlens)
        return PlanningResult(plan)

    def dispatch(
        self, plan: MoonEPPlan, hidden_sh, route_weights_sk=None
    ) -> DispatchResult:
        entry = self._plan_entry(plan, require_dedup=False)
        d = self.dimensions
        validate_tensor(
            hidden_sh,
            "hidden_sh",
            shape=(d.tokens_per_rank, d.hidden_size),
            dtype=self.torch.bfloat16,
        )
        if route_weights_sk is not None:
            validate_tensor(
                route_weights_sk,
                "route_weights_sk",
                shape=(d.tokens_per_rank, d.topk),
                dtype=self.torch.float32,
            )
        hidden, route_weights, _, returned_native = self.buffer.dispatch(
            hidden_sh,
            route_weights_sk,
            plan=entry.native,
        )
        if returned_native is not entry.native:
            raise ContractError("native Dispatch returned a different Plan")
        if not entry.dedup_ready:
            plan.dedup = DedupPlan(
                groups=entry.native.dup_groups,
                loffs=entry.native.dup_loffs,
                counts=entry.native.dup_counts,
            )
            entry.dedup_ready = True
        return DispatchResult(hidden, route_weights, plan)

    def prefetch_weight(
        self, plan: MoonEPPlan, projections: ProjectionTensors
    ) -> PrefetchResult:
        entry = self._plan_entry(plan, require_dedup=False)
        validate_projections(
            projections, self.dimensions, self.torch, dtype=self.torch.bfloat16
        )
        self.buffer.prefetch_weight(
            entry.native,
            full_gate_weight=projections.gate,
            full_up_weight=projections.up,
            full_down_weight=projections.down,
        )
        return PrefetchResult(projections)

    def combine(
        self, plan: MoonEPPlan, expert_output_nvsh, route_weights_nvs=None
    ) -> CombineResult:
        entry = self._plan_entry(plan, require_dedup=True)
        d = self.dimensions
        validate_tensor(
            expert_output_nvsh,
            "expert_output_nvsh",
            shape=(d.nvsh, d.hidden_size),
            dtype=self.torch.bfloat16,
        )
        if route_weights_nvs is not None:
            validate_tensor(
                route_weights_nvs,
                "route_weights_nvs",
                shape=(d.nvsh,),
                dtype=self.torch.float32,
            )
        hidden, route_weights, _ = self.buffer.combine(
            entry.native,
            expert_output_nvsh,
            route_weights_nvs,
        )
        return CombineResult(hidden, route_weights)

    def reduce_grad(
        self,
        plan: MoonEPPlan,
        full_grads: ProjectionTensors,
        reduce_buffers: ProjectionTensors,
    ) -> ReduceGradResult:
        entry = self._plan_entry(plan, require_dedup=True)
        validate_projections(
            full_grads, self.dimensions, self.torch, dtype=self.torch.float32
        )
        validate_projections(
            reduce_buffers,
            self.dimensions,
            self.torch,
            dtype=self.torch.float32,
            reduce_buffers=True,
        )
        self.buffer.reduce_grad(
            entry.native,
            full_gate_grad=full_grads.gate,
            full_up_grad=full_grads.up,
            full_down_grad=full_grads.down,
            gate_reduce_buffer=reduce_buffers.gate,
            up_reduce_buffer=reduce_buffers.up,
            down_reduce_buffer=reduce_buffers.down,
        )
        return ReduceGradResult(full_grads, reduce_buffers)

    def synchronize(self) -> None:
        self._require_open()
        self.buffer.synchronize()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.buffer.close()
        finally:
            self._closed = True
            self._native_plans.clear()


def _validate_context(
    torch_module, context: TileXRMoonEPContext, dimensions: MoonEPDimensions
) -> None:
    if (
        int(context.node_count) != 1
        or int(context.node_rank) != 0
        or int(context.local_world_size) != int(context.global_world_size)
    ):
        raise BackendUnavailableError(
            "TileXR correctness adapter currently supports same-node runs only"
        )

    expected = {
        "global_rank": dimensions.rank,
        "global_world_size": dimensions.world_size,
        "planner_group_rank": dimensions.rank,
        "planner_group_size": dimensions.world_size,
        "tokens_per_rank": dimensions.tokens_per_rank,
        "topk": dimensions.topk,
        "expert_count": dimensions.expert_count,
        "prefetch_slots": dimensions.prefetch_slots,
        "token_padding": dimensions.token_padding,
        "hidden_size": dimensions.hidden_size,
        "nv_s": dimensions.nvsh,
    }
    for name, value in expected.items():
        actual = int(getattr(context, name))
        if actual != int(value):
            raise BackendUnavailableError(
                f"TileXR context {name}={actual} does not match correctness "
                f"dimensions {int(value)}"
            )
    if context.dtype != torch_module.bfloat16:
        raise BackendUnavailableError(
            f"TileXR context dtype must be {torch_module.bfloat16}, got {context.dtype}"
        )
    current_device = int(torch_module.npu.current_device())
    if int(context.device_index) != current_device or int(context.local_rank) != current_device:
        raise BackendUnavailableError(
            "TileXR context device/local rank does not match the current NPU device"
        )

    capabilities = getattr(context.runtime, "capabilities", None)
    if capabilities is None:
        raise BackendUnavailableError("TileXR runtime does not expose MoonEP capabilities")
    if int(getattr(capabilities, "abi_version", 0)) != 1:
        raise BackendUnavailableError(
            f"TileXR MoonEP ABI V1 is required, got {capabilities.abi_version}"
        )
    for stage in _STAGES:
        implementation = capabilities.implementation(stage)
        if implementation != "native":
            raise BackendUnavailableError(
                f"TileXR MoonEP {stage} capability is {implementation}, expected native"
            )
    if not bool(capabilities.transport_correctness_valid):
        raise BackendUnavailableError(
            "TileXR MoonEP transport_correctness_valid capability is false"
        )


def _construction_error(
    error: Exception, cleanup_error: Exception | None
) -> Exception:
    if isinstance(error, BackendUnavailableError) and cleanup_error is None:
        return error
    message = f"TileXR correctness backend is unavailable: {type(error).__name__}: {error}"
    if cleanup_error is not None:
        message += (
            f"; context cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        )
    return BackendUnavailableError(message)


def create_backend(*, torch_module, dimensions, case, args) -> TileXRMoonEPBackend:
    del case
    context = None
    try:
        context = TileXRMoonEPContext.from_env(
            tokens_per_rank=dimensions.tokens_per_rank,
            hidden_size=dimensions.hidden_size,
            topk=dimensions.topk,
            expert_count=dimensions.expert_count,
            dtype=torch_module.bfloat16,
            token_padding=dimensions.token_padding,
            prefetch_slots=dimensions.prefetch_slots,
            install_prefix=getattr(args, "install_prefix", None),
            torch_module=torch_module,
        )
        _validate_context(torch_module, context, dimensions)
        buffer = TileXRMoonEPBuffer(
            context,
            wait_iterations=int(getattr(args, "wait_iterations", 1_000_000)),
            torch_module=torch_module,
        )
        return TileXRMoonEPBackend(
            torch_module=torch_module,
            dimensions=dimensions,
            buffer=buffer,
        )
    except Exception as error:
        cleanup_error = None
        if context is not None:
            try:
                context.close()
            except Exception as exc:
                cleanup_error = exc
        wrapped = _construction_error(error, cleanup_error)
        if wrapped is error:
            raise
        raise wrapped from error
