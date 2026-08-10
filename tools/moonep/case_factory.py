from __future__ import annotations

from .contracts import CanonicalMoonEPCase, MoonEPDimensions, ProjectionTensors


def build_topk_experts(torch_module, dimensions: MoonEPDimensions, pattern: str):
    d = dimensions
    route = torch_module.arange(d.route_count, dtype=torch_module.int64)
    if pattern == "balanced":
        values = (route + d.rank * d.route_count).remainder(d.expert_count)
    elif pattern == "skewed":
        values = (route.remainder(max(1, d.topk))).remainder(d.expert_count)
    elif pattern == "unique_destinations":
        if d.topk > d.world_size:
            raise ValueError(
                "unique_destinations requires topk to be no greater than world_size"
            )
        token = torch_module.div(route, d.topk, rounding_mode="floor")
        slot = route.remainder(d.topk)
        owner = (d.rank + token + slot).remainder(d.world_size)
        local_expert = (d.rank + token).remainder(d.experts_per_rank)
        values = owner * d.experts_per_rank + local_expert
    elif pattern == "duplicate_destinations":
        owner = (d.rank + 1) % d.world_size
        values = owner * d.experts_per_rank + route.remainder(d.experts_per_rank)
    elif pattern == "imbalanced_duplicates":
        token = torch_module.div(route, d.topk, rounding_mode="floor")
        owner = torch_module.zeros_like(route)
        if d.world_size > 1 and d.rank == 0:
            owner = (token == 0).to(dtype=route.dtype)
        values = owner * d.experts_per_rank + route.remainder(d.experts_per_rank)
    else:
        raise ValueError(f"unknown MoonEP routing pattern: {pattern}")
    return values.to(dtype=torch_module.int32).reshape(d.tokens_per_rank, d.topk)


def _pattern_tensor(torch_module, shape, *, base: int, dtype, device):
    values = torch_module.arange(
        _numel(shape), dtype=torch_module.int64, device=device
    ).remainder(23)
    values = values + int(base)
    return values.reshape(shape).to(dtype=dtype)


def _numel(shape) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _projection_set(
    torch_module,
    dimensions: MoonEPDimensions,
    *,
    dtype,
    device,
    prefix: tuple[int, ...],
    base: int,
) -> ProjectionTensors:
    d = dimensions
    return ProjectionTensors(
        gate=_pattern_tensor(
            torch_module,
            prefix + (d.hidden_size, d.intermediate_size),
            base=base,
            dtype=dtype,
            device=device,
        ),
        up=_pattern_tensor(
            torch_module,
            prefix + (d.hidden_size, d.intermediate_size),
            base=base + 32,
            dtype=dtype,
            device=device,
        ),
        down=_pattern_tensor(
            torch_module,
            prefix + (d.intermediate_size, d.hidden_size),
            base=base + 64,
            dtype=dtype,
            device=device,
        ),
    )


def make_correctness_case(
    torch_module,
    dimensions: MoonEPDimensions,
    *,
    case_id: str = "correctness",
    routing_pattern: str = "balanced",
    device="cpu",
) -> CanonicalMoonEPCase:
    d = dimensions
    topk = build_topk_experts(torch_module, d, routing_pattern).to(device=device)
    tokens_per_expert = torch_module.bincount(
        topk.reshape(-1).to(dtype=torch_module.int64), minlength=d.expert_count
    ).to(dtype=torch_module.int32)
    hidden = _pattern_tensor(
        torch_module,
        (d.tokens_per_rank, d.hidden_size),
        base=1 + d.rank * 32,
        dtype=torch_module.bfloat16,
        device=device,
    )
    route_weights = _pattern_tensor(
        torch_module,
        (d.tokens_per_rank, d.topk),
        base=1 + d.rank * 8,
        dtype=torch_module.float32,
        device=device,
    )
    projections = _projection_set(
        torch_module,
        d,
        dtype=torch_module.bfloat16,
        device=device,
        prefix=(d.expert_count + d.prefetch_slots,),
        base=1 + d.rank * 128,
    )
    for _, tensor in projections.items():
        tensor[d.expert_count :].fill_(-99)
    full_grads = _projection_set(
        torch_module,
        d,
        dtype=torch_module.float32,
        device=device,
        prefix=(d.expert_count + d.prefetch_slots,),
        base=1,
    )
    reduce_buffers = _projection_set(
        torch_module,
        d,
        dtype=torch_module.float32,
        device=device,
        prefix=(d.world_size, d.prefetch_slots),
        base=1 + d.rank * 16,
    )
    return CanonicalMoonEPCase(
        case_id=case_id,
        dimensions=d,
        topk_experts=topk,
        tokens_per_expert=tokens_per_expert,
        hidden=hidden,
        route_weights=route_weights,
        projections=projections,
        full_grads=full_grads,
        reduce_buffers=reduce_buffers,
    )
