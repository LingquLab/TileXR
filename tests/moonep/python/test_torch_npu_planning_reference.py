from __future__ import annotations

import pytest
import torch

from tools.moonep.case_factory import make_correctness_case
from tools.moonep.contracts import ContractError, MoonEPDimensions, validate_plan
from tools.moonep.planner_reference import build_reference_plan
from tools.moonep.torch_npu_backend import TorchNpuMoonEPBackend


class StaticCollective:
    def __init__(self, gathered):
        self.gathered = gathered

    def all_gather(self, tensor):
        return self.gathered.to(device=tensor.device)


def dimensions(rank: int) -> MoonEPDimensions:
    return MoonEPDimensions(
        rank=rank,
        world_size=2,
        tokens_per_rank=4,
        topk=2,
        expert_count=4,
        prefetch_slots=2,
        token_padding=4,
        hidden_size=4,
        intermediate_size=3,
    )


@pytest.mark.parametrize(
    "pattern",
    ["balanced", "skewed", "duplicate_destinations", "imbalanced_duplicates"],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_planning_matches_repository_reference(pattern: str, rank: int) -> None:
    cases = [
        make_correctness_case(torch, dimensions(source), routing_pattern=pattern)
        for source in range(2)
    ]
    all_topk = torch.stack([case.topk_experts for case in cases])
    all_tpe = torch.stack([case.tokens_per_expert for case in cases])
    actual_plans = [
        TorchNpuMoonEPBackend(
            torch, dimensions(source), collective=StaticCollective(all_tpe)
        ).planning(cases[source].topk_experts, cases[source].tokens_per_expert).plan
        for source in range(2)
    ]
    actual = actual_plans[rank]
    validate_plan(actual, torch, require_dedup=False)

    d = dimensions(rank)
    expected = build_reference_plan(
        rank=rank,
        rank_size=d.world_size,
        tokens_per_rank=d.tokens_per_rank,
        topk=d.topk,
        expert_count=d.expert_count,
        prefetch_slots=d.prefetch_slots,
        token_padding=d.token_padding,
        all_topk=tuple(int(value) for value in all_topk.reshape(-1).tolist()),
    )
    assert tuple(actual.dst.tolist()) == expected.dst
    assert tuple(actual.cu_seqlens.tolist()) == expected.cu_seqlens
    assert tuple(actual.experts_to_copy.reshape(-1).tolist()) == expected.experts_to_copy
    assert tuple(actual.remote_stats.tolist()) == expected.remote_stats
    assert tuple(actual.zero_fill_ranges.reshape(-1).tolist()) == expected.zero_fill_ranges


def test_planning_rejects_inconsistent_tokens_per_expert() -> None:
    case = make_correctness_case(torch, dimensions(0))
    bad = case.tokens_per_expert.clone()
    bad[0] += 1
    backend = TorchNpuMoonEPBackend(
        torch,
        dimensions(0),
        collective=StaticCollective(torch.stack([bad, bad])),
    )
    with pytest.raises(ContractError, match="does not match"):
        backend.planning(case.topk_experts, bad)


def test_planning_zero_fill_covers_e2e_physical_tail() -> None:
    R, S, K, E = 8, 256, 4, 32
    experts_per_rank = E // R
    all_topk = tuple(
        ((rank + token + route) % R) * experts_per_rank
        + (token * K + route) % experts_per_rank
        for rank in range(R)
        for token in range(S)
        for route in range(K)
    )

    plan = build_reference_plan(
        rank=0,
        rank_size=R,
        tokens_per_rank=S,
        topk=K,
        expert_count=E,
        prefetch_slots=2,
        token_padding=128,
        all_topk=all_topk,
    )

    assert plan.dispatched_capacity == 2040
    assert plan.cu_seqlens[-1] == 1024
    assert plan.zero_fill_ranges[-2:] == (1024, 1016)


def test_two_rank_skewed_case_rebalances_owner_load_and_prefetches_remote_expert() -> None:
    small = [
        MoonEPDimensions(rank, 2, 2, 2, 4, 2, 1, 2, 2)
        for rank in range(2)
    ]
    cases = [
        make_correctness_case(torch, item, routing_pattern="skewed")
        for item in small
    ]
    all_tpe = torch.stack([case.tokens_per_expert for case in cases])
    assert torch.equal(
        all_tpe,
        torch.tensor([[2, 2, 0, 0], [2, 2, 0, 0]], dtype=torch.int32),
    )
    global_tpe = all_tpe.sum(dim=0)
    initial_owner_loads = torch.tensor(
        [global_tpe[:2].sum(), global_tpe[2:].sum()], dtype=torch.int32
    )
    assert torch.equal(initial_owner_loads, torch.tensor([8, 0], dtype=torch.int32))

    plans = [
        TorchNpuMoonEPBackend(
            torch, small[rank], collective=StaticCollective(all_tpe)
        ).planning(cases[rank].topk_experts, cases[rank].tokens_per_expert).plan
        for rank in range(2)
    ]

    assert [int(plan.cu_seqlens[-1].item()) for plan in plans] == [4, 4]
    assert torch.equal(
        plans[0].experts_to_copy,
        torch.tensor([[-1, -1], [0, -1]], dtype=torch.int32),
    )
    assert torch.equal(plans[0].remote_stats, torch.tensor([0, 1], dtype=torch.int32))
    assert torch.equal(plans[1].remote_stats, torch.tensor([1, 0], dtype=torch.int32))
    assert torch.equal(plans[0].dst, torch.tensor([4, 0, 5, 1], dtype=torch.int32))
    assert torch.equal(plans[1].dst, torch.tensor([6, 2, 7, 3], dtype=torch.int32))
