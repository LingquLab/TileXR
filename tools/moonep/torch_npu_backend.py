from __future__ import annotations

from typing import Any

from .contracts import (
    BackendUnavailableError,
    ContractError,
    DedupPlan,
    DispatchResult,
    MoonEPDimensions,
    MoonEPPlan,
    PlanningResult,
    ProjectionTensors,
    PrefetchResult,
    CombineResult,
    ReduceGradResult,
    validate_plan,
    validate_projections,
    validate_tensor,
)


class TorchDistributedCollective:
    """Small HCCL adapter used only by the correctness reference backend."""

    def __init__(self, torch_module, dimensions: MoonEPDimensions, group=None):
        self.torch = torch_module
        self.dimensions = dimensions
        self.group = group

    def all_gather(self, tensor):
        d = self.dimensions
        if d.world_size == 1:
            return tensor.unsqueeze(0)
        dist = self.torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise BackendUnavailableError(
                "multi-rank Torch MoonEP reference requires an initialized HCCL process group"
            )
        if dist.get_world_size(self.group) != d.world_size:
            raise ContractError("HCCL world size does not match MoonEP dimensions")
        if dist.get_rank(self.group) != d.rank:
            raise ContractError("HCCL rank does not match MoonEP dimensions")
        backend = str(dist.get_backend(self.group)).lower()
        if tensor.device.type == "npu" and backend != "hccl":
            raise BackendUnavailableError(
                f"NPU MoonEP reference requires HCCL, got process-group backend {backend}"
            )
        gathered = [self.torch.empty_like(tensor) for _ in range(d.world_size)]
        dist.all_gather(gathered, tensor, group=self.group)
        return self.torch.stack(gathered, dim=0)

    def all_agree(self, passed: bool, *, device) -> bool:
        d = self.dimensions
        if d.world_size == 1:
            return bool(passed)
        value = self.torch.tensor([int(passed)], dtype=self.torch.int32, device=device)
        self.torch.distributed.all_reduce(
            value,
            op=self.torch.distributed.ReduceOp.MIN,
            group=self.group,
        )
        return bool(int(value.item()))


class TorchNpuMoonEPBackend:
    """Baseline V1 MoonEP semantics expressed with ordinary Torch operations."""

    name = "torch_npu_reference_v1"

    def __init__(
        self,
        torch_module,
        dimensions: MoonEPDimensions,
        *,
        collective: Any | None = None,
    ) -> None:
        self.torch = torch_module
        self.dimensions = dimensions
        self.collective = collective or TorchDistributedCollective(
            torch_module, dimensions
        )

    def synchronize(self) -> None:
        npu = getattr(self.torch, "npu", None)
        if npu is not None:
            npu.synchronize()

    def close(self) -> None:
        return None

    def planning(self, topk_experts, tokens_per_expert) -> PlanningResult:
        torch = self.torch
        d = self.dimensions
        validate_tensor(
            topk_experts,
            "topk_experts",
            shape=(d.tokens_per_rank, d.topk),
            dtype=torch.int32,
        )
        validate_tensor(
            tokens_per_expert,
            "tokens_per_expert",
            shape=(d.expert_count,),
            dtype=torch.int32,
        )
        if bool(((topk_experts < 0) | (topk_experts >= d.expert_count)).any().item()):
            raise ContractError("topk_experts contains an out-of-range expert id")
        expected_tpe = torch.bincount(
            topk_experts.reshape(-1).to(dtype=torch.int64), minlength=d.expert_count
        ).to(dtype=torch.int32)
        if not torch.equal(tokens_per_expert, expected_tpe):
            raise ContractError("tokens_per_expert does not match topk_experts")

        all_tpe = self.collective.all_gather(tokens_per_expert)
        if tuple(all_tpe.shape) != (d.world_size, d.expert_count):
            raise ContractError("all_gather(tokens_per_expert) returned an invalid shape")
        return PlanningResult(self._build_plan(topk_experts, all_tpe))

    def _build_plan(self, topk_experts, all_tpe) -> MoonEPPlan:
        torch = self.torch
        d = self.dimensions
        device = topk_experts.device
        R, E, B, epn = (
            d.world_size,
            d.expert_count,
            d.prefetch_slots,
            d.experts_per_rank,
        )

        tpe_cumsum = all_tpe.cumsum(dim=0, dtype=torch.int32)
        global_counts = tpe_cumsum[-1]
        balance = global_counts.reshape(R, epn).sum(dim=1) - d.route_count
        alloc = torch.zeros((E, R), dtype=torch.int32, device=device)
        expert_ids = torch.arange(E, dtype=torch.int64, device=device)
        alloc[expert_ids, torch.div(expert_ids, epn, rounding_mode="floor")] = global_counts
        moves = torch.zeros((R, R), dtype=torch.int32, device=device)

        while True:
            source = int(balance.argmax().item())
            destination = int(balance.argmin().item())
            if int(balance[source].item()) <= 0:
                break
            move = -int(balance[destination].item())
            moves[source, destination] = move
            balance[source] -= move
            balance[destination] = 0

        for owner in range(R):
            begin = owner * epn
            remaining = global_counts[begin : begin + epn].clone()
            quotas = moves[owner].clone()
            while True:
                destination = int(quotas.argmax().item())
                quota = int(quotas[destination].item())
                if quota <= 0:
                    break
                local_expert = int(remaining.argmax().item())
                available = int(remaining[local_expert].item())
                take = min(available, quota)
                expert = begin + local_expert
                alloc[expert, destination] += take
                alloc[expert, owner] -= take
                remaining[local_expert] -= take
                quotas[destination] -= take

        if not torch.equal(alloc.sum(dim=1), global_counts):
            raise RuntimeError("Planning allocation lost expert tokens")
        if bool((alloc.sum(dim=0) > d.route_count).any().item()):
            raise RuntimeError("Planning allocation exceeds rank capacity")

        expert_offsets = torch.zeros((R, E), dtype=torch.int32, device=device)
        all_cu = torch.zeros((R, E + B), dtype=torch.int32, device=device)
        experts_to_copy = torch.full(
            (R, B), -1, dtype=torch.int32, device=device
        )
        remote_stats = torch.zeros((R, 2), dtype=torch.int32, device=device)
        zero_fill = torch.zeros((R, E + B, 2), dtype=torch.int32, device=device)

        for destination in range(R):
            local_begin = destination * epn
            local_end = local_begin + epn
            remote = [
                expert
                for expert in range(E)
                if not local_begin <= expert < local_end
                and int(alloc[expert, destination].item()) > 0
            ]
            remote.sort(
                key=lambda expert: (int(alloc[expert, destination].item()), expert),
                reverse=True,
            )
            remote_stats[destination, 0] = len(remote)
            selected = set(remote[:B])
            for slot, expert in enumerate(remote[:B]):
                experts_to_copy[destination, slot] = expert
                remote_stats[expert // epn, 1] += 1

            start = 0
            for group in range(E + B):
                count = 0
                expert = -1
                if group < E:
                    if group not in selected:
                        expert = group
                        count = int(alloc[expert, destination].item())
                else:
                    slot = group - E
                    expert = int(experts_to_copy[destination, slot].item())
                    if expert >= 0:
                        count = int(alloc[expert, destination].item())
                end = start + count
                padded = (
                    ((count + d.token_padding - 1) // d.token_padding)
                    * d.token_padding
                    if count > 0
                    else 0
                )
                padded_end = start + padded
                all_cu[destination, group] = padded_end
                if count > 0:
                    expert_offsets[destination, expert] = start
                    if padded > count:
                        zero_fill[destination, group, 0] = end
                        zero_fill[destination, group, 1] = padded - count
                start = padded_end
            if start > d.nvsh:
                raise RuntimeError("Planning padded layout exceeds NvS")

        alloc_cumsum = alloc.cumsum(dim=1)
        local_counts = torch.zeros(E, dtype=torch.int32, device=device)
        dst = torch.empty(d.route_count, dtype=torch.int32, device=device)
        flat_topk = topk_experts.reshape(-1)
        for route_index in range(d.route_count):
            expert = int(flat_topk[route_index].item())
            local_occurrence = int(local_counts[expert].item())
            local_counts[expert] += 1
            source_prefix = (
                0 if d.rank == 0 else int(tpe_cumsum[d.rank - 1, expert].item())
            )
            global_occurrence = source_prefix + local_occurrence
            destination = int(
                torch.searchsorted(
                    alloc_cumsum[expert],
                    torch.tensor(global_occurrence, dtype=torch.int32, device=device),
                    right=True,
                ).item()
            )
            if destination >= R:
                raise RuntimeError("Planning could not find a destination rank")
            previous = (
                0
                if destination == 0
                else int(alloc_cumsum[expert, destination - 1].item())
            )
            local_offset = (
                int(expert_offsets[destination, expert].item())
                + global_occurrence
                - previous
            )
            dst[route_index] = destination * d.nvsh + local_offset

        for token in range(d.tokens_per_rank):
            seen_destinations: set[int] = set()
            for topk_index in range(d.topk):
                route_index = token * d.topk + topk_index
                raw = int(dst[route_index].item())
                destination = raw // d.nvsh
                if destination in seen_destinations:
                    dst[route_index] = -raw - 1
                else:
                    seen_destinations.add(destination)

        return MoonEPPlan(
            dimensions=d,
            dst=dst,
            cu_seqlens=all_cu[d.rank].clone(),
            experts_to_copy=experts_to_copy.contiguous(),
            zero_fill_ranges=zero_fill[d.rank].clone(),
            remote_stats=remote_stats[d.rank].clone(),
            dedup=None,
        )

    def dispatch(
        self, plan: MoonEPPlan, hidden_sh, route_weights_sk=None
    ) -> DispatchResult:
        torch = self.torch
        d = self.dimensions
        self._require_plan_dimensions(plan)
        validate_plan(plan, torch, require_dedup=False)
        validate_tensor(
            hidden_sh,
            "hidden_sh",
            shape=(d.tokens_per_rank, d.hidden_size),
            dtype=torch.bfloat16,
        )
        if route_weights_sk is not None:
            validate_tensor(
                route_weights_sk,
                "route_weights_sk",
                shape=(d.tokens_per_rank, d.topk),
                dtype=torch.float32,
            )

        all_dst = self.collective.all_gather(plan.dst)
        all_hidden = self.collective.all_gather(hidden_sh)
        all_weights = (
            None
            if route_weights_sk is None
            else self.collective.all_gather(route_weights_sk)
        )
        output = torch.zeros(
            (d.nvsh, d.hidden_size), dtype=torch.bfloat16, device=hidden_sh.device
        )
        output_weights = (
            None
            if route_weights_sk is None
            else torch.zeros(d.nvsh, dtype=torch.float32, device=hidden_sh.device)
        )
        groups = torch.zeros((d.nvsh, 3), dtype=torch.int32, device=hidden_sh.device)
        loffs = torch.zeros(d.nvsh, dtype=torch.int32, device=hidden_sh.device)
        group_count = 0
        loff_count = 0

        for source in range(d.world_size):
            for token in range(d.tokens_per_rank):
                by_destination: dict[int, list[tuple[int, int]]] = {}
                for topk_index in range(d.topk):
                    route_index = token * d.topk + topk_index
                    encoded = int(all_dst[source, route_index].item())
                    raw = -encoded - 1 if encoded < 0 else encoded
                    destination = raw // d.nvsh
                    local_offset = raw % d.nvsh
                    by_destination.setdefault(destination, []).append(
                        (local_offset, topk_index)
                    )
                    if destination == d.rank:
                        output[local_offset].copy_(all_hidden[source, token])
                        if output_weights is not None:
                            output_weights[local_offset] = all_weights[
                                source, token, topk_index
                            ]
                local_routes = by_destination.get(d.rank, [])
                if len(local_routes) > 1:
                    duplicate_count = len(local_routes) - 1
                    if group_count >= d.nvsh or loff_count + duplicate_count > d.nvsh:
                        raise RuntimeError("Dispatch dedup metadata exceeds NvS")
                    groups[group_count, 0] = local_routes[0][0]
                    groups[group_count, 1] = loff_count
                    groups[group_count, 2] = duplicate_count
                    for duplicate, (local_offset, _) in enumerate(local_routes[1:]):
                        loffs[loff_count + duplicate] = local_offset
                    group_count += 1
                    loff_count += duplicate_count

        post_plan = plan.clone()
        post_plan.dedup = DedupPlan(
            groups=groups,
            loffs=loffs,
            counts=torch.tensor(
                [group_count, loff_count],
                dtype=torch.int32,
                device=hidden_sh.device,
            ),
        )
        return DispatchResult(output, output_weights, post_plan)

    def prefetch_weight(
        self, plan: MoonEPPlan, projections: ProjectionTensors
    ) -> PrefetchResult:
        torch = self.torch
        d = self.dimensions
        self._require_plan_dimensions(plan)
        validate_plan(plan, torch, require_dedup=False)
        validate_projections(projections, d, torch, dtype=torch.bfloat16)
        for _, tensor in projections.items():
            all_sources = self.collective.all_gather(tensor[: d.expert_count].contiguous())
            for slot in range(d.prefetch_slots):
                expert = int(plan.experts_to_copy[d.rank, slot].item())
                if expert < 0:
                    continue
                owner = expert // d.experts_per_rank
                tensor[d.expert_count + slot].copy_(all_sources[owner, expert])
        return PrefetchResult(projections)

    def combine(
        self, plan: MoonEPPlan, expert_output_nvsh, route_weights_nvs=None
    ) -> CombineResult:
        torch = self.torch
        d = self.dimensions
        self._require_plan_dimensions(plan)
        validate_plan(plan, torch, require_dedup=True)
        validate_tensor(
            expert_output_nvsh,
            "expert_output_nvsh",
            shape=(d.nvsh, d.hidden_size),
            dtype=torch.bfloat16,
        )
        if route_weights_nvs is not None:
            validate_tensor(
                route_weights_nvs,
                "route_weights_nvs",
                shape=(d.nvsh,),
                dtype=torch.float32,
            )
        all_hidden = self.collective.all_gather(expert_output_nvsh)
        all_weights = (
            None
            if route_weights_nvs is None
            else self.collective.all_gather(route_weights_nvs)
        )
        output = torch.empty(
            (d.tokens_per_rank, d.hidden_size),
            dtype=torch.bfloat16,
            device=expert_output_nvsh.device,
        )
        output_weights = (
            None
            if route_weights_nvs is None
            else torch.empty(
                (d.tokens_per_rank, d.topk),
                dtype=torch.float32,
                device=expert_output_nvsh.device,
            )
        )
        for token in range(d.tokens_per_rank):
            accumulated = torch.zeros(
                d.hidden_size,
                dtype=torch.float32,
                device=expert_output_nvsh.device,
            )
            for topk_index in range(d.topk):
                route_index = token * d.topk + topk_index
                encoded = int(plan.dst[route_index].item())
                raw = -encoded - 1 if encoded < 0 else encoded
                destination = raw // d.nvsh
                local_offset = raw % d.nvsh
                accumulated.add_(all_hidden[destination, local_offset].float())
                if output_weights is not None:
                    output_weights[token, topk_index] = all_weights[
                        destination, local_offset
                    ]
            output[token].copy_(accumulated.to(dtype=torch.bfloat16))
        return CombineResult(output, output_weights)

    def reduce_grad(
        self,
        plan: MoonEPPlan,
        full_grads: ProjectionTensors,
        reduce_buffers: ProjectionTensors,
    ) -> ReduceGradResult:
        torch = self.torch
        d = self.dimensions
        self._require_plan_dimensions(plan)
        validate_plan(plan, torch, require_dedup=True)
        validate_projections(full_grads, d, torch, dtype=torch.float32)
        validate_projections(
            reduce_buffers,
            d,
            torch,
            dtype=torch.float32,
            reduce_buffers=True,
        )
        owner_begin = d.rank * d.experts_per_rank
        owner_end = owner_begin + d.experts_per_rank
        for name, full_tensor in full_grads.items():
            reduce_tensor = getattr(reduce_buffers, name)
            gathered = self.collective.all_gather(
                reduce_tensor[d.rank].contiguous()
            )
            for source in range(d.world_size):
                for slot in range(d.prefetch_slots):
                    expert = int(plan.experts_to_copy[source, slot].item())
                    if owner_begin <= expert < owner_end:
                        full_tensor[expert].add_(gathered[source, slot])
            for slot in range(d.prefetch_slots):
                if int(plan.experts_to_copy[d.rank, slot].item()) >= 0:
                    reduce_tensor[d.rank, slot].zero_()
        return ReduceGradResult(full_grads, reduce_buffers)

    def _require_plan_dimensions(self, plan: MoonEPPlan) -> None:
        if not isinstance(plan, MoonEPPlan) or plan.dimensions != self.dimensions:
            raise ContractError("plan dimensions do not match the backend")
