from __future__ import annotations

import torch

from tools.moonep.case_factory import make_correctness_case
from tools.moonep.contracts import MoonEPDimensions
from tools.moonep.torch_npu_backend import TorchNpuMoonEPBackend


class StaticCollective:
    def __init__(self, gathered):
        self.gathered = gathered

    def all_gather(self, tensor):
        return self.gathered.to(device=tensor.device)


class SequenceCollective:
    def __init__(self, *gathered):
        self.gathered = list(gathered)

    def all_gather(self, tensor):
        value = self.gathered.pop(0).to(device=tensor.device)
        assert tuple(value.shape[1:]) == tuple(tensor.shape)
        return value


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


def setup(pattern="skewed"):
    cases = [
        make_correctness_case(torch, dimensions(rank), routing_pattern=pattern)
        for rank in range(2)
    ]
    all_tpe = torch.stack([case.tokens_per_expert for case in cases])
    plans = [
        TorchNpuMoonEPBackend(
            torch, dimensions(rank), collective=StaticCollective(all_tpe)
        ).planning(case.topk_experts, case.tokens_per_expert).plan
        for rank, case in enumerate(cases)
    ]
    return cases, plans


def dispatch_all(cases, plans):
    all_dst = torch.stack([plan.dst for plan in plans])
    all_hidden = torch.stack([case.hidden for case in cases])
    all_weights = torch.stack([case.route_weights for case in cases])
    return [
        TorchNpuMoonEPBackend(
            torch,
            dimensions(rank),
            collective=SequenceCollective(all_dst, all_hidden, all_weights),
        ).dispatch(plans[rank], cases[rank].hidden, cases[rank].route_weights)
        for rank in range(2)
    ]


def test_two_rank_imbalanced_case_dispatches_a_three_duplicate_group() -> None:
    small = [
        MoonEPDimensions(rank, 2, 2, 4, 8, 4, 1, 2, 2)
        for rank in range(2)
    ]
    cases = [
        make_correctness_case(
            torch, item, routing_pattern="imbalanced_duplicates"
        )
        for item in small
    ]
    all_tpe = torch.stack([case.tokens_per_expert for case in cases])
    global_tpe = all_tpe.sum(dim=0)
    assert torch.equal(
        global_tpe.reshape(2, 4).sum(dim=1),
        torch.tensor([12, 4], dtype=torch.int32),
    )
    plans = [
        TorchNpuMoonEPBackend(
            torch, small[rank], collective=StaticCollective(all_tpe)
        ).planning(cases[rank].topk_experts, cases[rank].tokens_per_expert).plan
        for rank in range(2)
    ]
    assert torch.equal(
        plans[0].dst,
        torch.tensor([8, 9, 10, 11, 12, 0, 2, 5], dtype=torch.int32),
    )
    assert torch.equal(
        plans[1].dst,
        torch.tensor([13, 1, 3, 6, 14, 15, 4, 7], dtype=torch.int32),
    )
    assert [int(plan.cu_seqlens[-1].item()) for plan in plans] == [8, 8]
    assert torch.equal(
        plans[0].experts_to_copy,
        torch.tensor(
            [[-1, -1, -1, -1], [0, 1, -1, -1]], dtype=torch.int32
        ),
    )

    all_dst = torch.stack([plan.dst for plan in plans])
    all_hidden = torch.stack([case.hidden for case in cases])
    all_weights = torch.stack([case.route_weights for case in cases])
    dispatched = [
        TorchNpuMoonEPBackend(
            torch,
            small[rank],
            collective=SequenceCollective(all_dst, all_hidden, all_weights),
        ).dispatch(plans[rank], cases[rank].hidden, cases[rank].route_weights)
        for rank in range(2)
    ]
    expected_groups = (
        torch.tensor(
            [
                [0, 0, 2],
                [1, 2, 2],
                [4, 4, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.int32,
        ),
        torch.tensor(
            [
                [0, 0, 3],
                [6, 3, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )
    expected_loffs = (
        torch.tensor([2, 5, 3, 6, 7, 0, 0, 0], dtype=torch.int32),
        torch.tensor([1, 2, 3, 7, 0, 0, 0, 0], dtype=torch.int32),
    )
    expected_counts = (
        torch.tensor([3, 5], dtype=torch.int32),
        torch.tensor([2, 4], dtype=torch.int32),
    )
    for rank, result in enumerate(dispatched):
        assert result.plan.dedup is not None
        assert torch.equal(result.plan.dedup.groups, expected_groups[rank])
        assert torch.equal(result.plan.dedup.loffs, expected_loffs[rank])
        assert torch.equal(result.plan.dedup.counts, expected_counts[rank])
    assert int(dispatched[1].plan.dedup.groups[0, 2].item()) == 3


def test_dispatch_scatter_padding_weights_and_dedup() -> None:
    cases, plans = setup("duplicate_destinations")
    dispatched = dispatch_all(cases, plans)
    for destination, result in enumerate(dispatched):
        assert result.plan.dedup is not None
        for source in range(2):
            for token in range(dimensions(source).tokens_per_rank):
                for topk_index in range(dimensions(source).topk):
                    route = token * dimensions(source).topk + topk_index
                    encoded = int(plans[source].dst[route])
                    raw = -encoded - 1 if encoded < 0 else encoded
                    if raw // dimensions(source).nvsh != destination:
                        continue
                    loff = raw % dimensions(source).nvsh
                    assert torch.equal(result.hidden[loff], cases[source].hidden[token])
                    assert result.route_weights[loff] == cases[source].route_weights[
                        token, topk_index
                    ]
        for start, count in result.plan.zero_fill_ranges.tolist():
            if count:
                assert torch.count_nonzero(result.hidden[start : start + count]) == 0
                assert torch.count_nonzero(
                    result.route_weights[start : start + count]
                ) == 0


def test_prefetch_copies_live_slots_and_preserves_unused_slots() -> None:
    cases, plans = setup("skewed")
    for rank in range(2):
        projections = cases[rank].projections.clone()
        before = projections.clone()
        gathered = []
        for name, _ in projections.items():
            gathered.append(
                torch.stack(
                    [getattr(case.projections, name)[: dimensions(rank).expert_count] for case in cases]
                )
            )
        result = TorchNpuMoonEPBackend(
            torch,
            dimensions(rank),
            collective=SequenceCollective(*gathered),
        ).prefetch_weight(plans[rank], projections)
        for name, tensor in result.projections.items():
            assert torch.equal(
                tensor[: dimensions(rank).expert_count],
                getattr(before, name)[: dimensions(rank).expert_count],
            )
            for slot in range(dimensions(rank).prefetch_slots):
                expert = int(plans[rank].experts_to_copy[rank, slot])
                row = tensor[dimensions(rank).expert_count + slot]
                if expert < 0:
                    assert torch.all(row == -99)
                else:
                    owner = expert // dimensions(rank).experts_per_rank
                    assert torch.equal(row, getattr(cases[owner].projections, name)[expert])


def test_combine_gathers_all_routes_and_route_weights() -> None:
    cases, plans = setup("skewed")
    dispatched = dispatch_all(cases, plans)
    outputs = [
        (
            torch.arange(
                dimensions(rank).nvsh * dimensions(rank).hidden_size,
                dtype=torch.float32,
            ).reshape(dimensions(rank).nvsh, dimensions(rank).hidden_size)
            + rank * 100
        ).to(torch.bfloat16)
        for rank in range(2)
    ]
    all_outputs = torch.stack(outputs)
    all_route_weights = torch.stack([item.route_weights for item in dispatched])
    for rank in range(2):
        backend = TorchNpuMoonEPBackend(
            torch,
            dimensions(rank),
            collective=SequenceCollective(all_outputs, all_route_weights),
        )
        result = backend.combine(
            dispatched[rank].plan,
            outputs[rank],
            dispatched[rank].route_weights,
        )
        expected = torch.zeros_like(result.hidden, dtype=torch.float32)
        for token in range(dimensions(rank).tokens_per_rank):
            for topk_index in range(dimensions(rank).topk):
                route = token * dimensions(rank).topk + topk_index
                encoded = int(plans[rank].dst[route])
                raw = -encoded - 1 if encoded < 0 else encoded
                expected[token] += all_outputs[
                    raw // dimensions(rank).nvsh, raw % dimensions(rank).nvsh
                ].float()
        assert torch.equal(result.hidden, expected.to(torch.bfloat16))
        assert torch.equal(result.route_weights, cases[rank].route_weights)


def test_reduce_grad_updates_owned_rows_and_clears_only_live_slots() -> None:
    cases, plans = setup("skewed")
    dispatched = dispatch_all(cases, plans)
    for rank in range(2):
        full_grads = cases[rank].full_grads.clone()
        buffers = cases[rank].reduce_buffers.clone()
        before_grads = full_grads.clone()
        before_buffers = buffers.clone()
        gathered = []
        for name, _ in full_grads.items():
            gathered.append(
                torch.stack(
                    [getattr(cases[source].reduce_buffers, name)[source] for source in range(2)]
                )
            )
        result = TorchNpuMoonEPBackend(
            torch,
            dimensions(rank),
            collective=SequenceCollective(*gathered),
        ).reduce_grad(dispatched[rank].plan, full_grads, buffers)
        begin = rank * dimensions(rank).experts_per_rank
        end = begin + dimensions(rank).experts_per_rank
        for name, tensor in result.full_grads.items():
            expected = getattr(before_grads, name).clone()
            values = gathered[("gate", "up", "down").index(name)]
            for source in range(2):
                for slot in range(dimensions(rank).prefetch_slots):
                    expert = int(plans[rank].experts_to_copy[source, slot])
                    if begin <= expert < end:
                        expected[expert] += values[source, slot]
            assert torch.equal(tensor, expected)
        for name, tensor in result.reduce_buffers.items():
            original = getattr(before_buffers, name)
            for source in range(2):
                for slot in range(dimensions(rank).prefetch_slots):
                    expert = int(plans[rank].experts_to_copy[source, slot])
                    if source == rank and expert >= 0:
                        assert torch.count_nonzero(tensor[source, slot]) == 0
                    else:
                        assert torch.equal(tensor[source, slot], original[source, slot])
