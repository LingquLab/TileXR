from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.moonep.case_factory import make_correctness_case
from tools.moonep.config import load_cases
from tools.moonep.contracts import MoonEPDimensions, validate_projections


def dimensions(rank: int = 0) -> MoonEPDimensions:
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


def test_case_is_reproducible_and_counts_routes() -> None:
    first = make_correctness_case(torch, dimensions(), routing_pattern="balanced")
    second = make_correctness_case(torch, dimensions(), routing_pattern="balanced")
    assert torch.equal(first.topk_experts, second.topk_experts)
    assert torch.equal(first.hidden, second.hidden)
    assert torch.equal(first.tokens_per_expert, second.tokens_per_expert)
    assert int(first.tokens_per_expert.sum().item()) == dimensions().route_count
    expected = torch.bincount(
        first.topk_experts.reshape(-1).to(torch.int64), minlength=4
    ).to(torch.int32)
    assert torch.equal(first.tokens_per_expert, expected)


def test_case_has_upstream_projection_shapes_and_unused_slot_sentinel() -> None:
    case = make_correctness_case(torch, dimensions())
    assert not hasattr(case, "expert_output")
    validate_projections(case.projections, dimensions(), torch, dtype=torch.bfloat16)
    validate_projections(case.full_grads, dimensions(), torch, dtype=torch.float32)
    validate_projections(
        case.reduce_buffers,
        dimensions(),
        torch,
        dtype=torch.float32,
        reduce_buffers=True,
    )
    for _, tensor in case.projections.items():
        assert torch.all(tensor[dimensions().expert_count :] == -99)


def test_duplicate_destination_pattern_stays_inside_one_owner_group() -> None:
    case = make_correctness_case(
        torch, dimensions(), routing_pattern="duplicate_destinations"
    )
    owners = torch.div(
        case.topk_experts, dimensions().experts_per_rank, rounding_mode="floor"
    )
    assert torch.all(owners == owners[:, :1])


def test_case_clone_does_not_alias_inputs() -> None:
    case = make_correctness_case(torch, dimensions())
    cloned = case.clone()
    cloned.hidden.zero_()
    cloned.projections.gate.zero_()
    assert torch.count_nonzero(case.hidden).item() > 0
    assert torch.count_nonzero(case.projections.gate).item() > 0


def test_expert_weights_are_rank_distinguishable() -> None:
    rank_zero = make_correctness_case(torch, dimensions(0))
    rank_one = make_correctness_case(torch, dimensions(1))
    assert not torch.equal(
        rank_zero.projections.gate[: dimensions().expert_count],
        rank_one.projections.gate[: dimensions().expert_count],
    )


def test_manual_small_case_keeps_all_stage_tensors_human_auditable() -> None:
    root = Path(__file__).resolve().parents[3]
    case_path = root / "tools" / "moonep" / "cases" / "correctness.json"
    raw_cases = json.loads(case_path.read_text(encoding="utf-8"))
    raw_manual = next(case for case in raw_cases if case["id"] == "manual-small")
    assert "P" not in raw_manual
    assert "token_padding" not in raw_manual
    cases = load_cases(case_path)
    manual = next(case for case in cases if case.case_id == "manual-small")
    assert (
        manual.tokens_per_rank,
        manual.topk,
        manual.expert_count,
        manual.hidden_size,
        manual.intermediate_size,
        manual.prefetch_slots,
        manual.token_padding,
    ) == (2, 2, 4, 2, 2, 1, 1)
    dimensions = MoonEPDimensions(0, 4, 2, 2, 4, 1, 1, 2, 2)
    assert dimensions.nvsh == 4


def test_manual_two_rank_case_starts_with_all_routes_on_rank_zero_experts() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")
    case = next(item for item in cases if item.case_id == "manual-2rank-imbalanced")
    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.hidden_size,
        case.intermediate_size,
        case.prefetch_slots,
        case.token_padding,
        case.routing_pattern,
    ) == (2, 2, 4, 2, 2, 1, 1, "skewed")
    dimensions = MoonEPDimensions(0, 2, 2, 2, 4, 1, 1, 2, 2)
    generated = make_correctness_case(
        torch, dimensions, routing_pattern=case.routing_pattern
    )
    assert torch.equal(
        generated.tokens_per_expert,
        torch.tensor([2, 2, 0, 0], dtype=torch.int32),
    )


def test_manual_two_rank_case_combines_imbalance_and_three_duplicates() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")
    case = next(item for item in cases if item.case_id == "manual-2rank-dedup-3")
    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.hidden_size,
        case.intermediate_size,
        case.prefetch_slots,
        case.token_padding,
        case.routing_pattern,
    ) == (2, 4, 8, 2, 2, 2, 1, "imbalanced_duplicates")

    expected = (
        torch.tensor([[4, 5, 6, 7], [0, 1, 2, 3]], dtype=torch.int32),
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32),
    )
    for rank in range(2):
        dimensions = MoonEPDimensions(rank, 2, 2, 4, 8, 2, 1, 2, 2)
        generated = make_correctness_case(
            torch, dimensions, routing_pattern=case.routing_pattern
        )
        assert torch.equal(generated.topk_experts, expected[rank])

    all_tpe = torch.stack(
        [
            make_correctness_case(
                torch,
                MoonEPDimensions(rank, 2, 2, 4, 8, 2, 1, 2, 2),
                routing_pattern=case.routing_pattern,
            ).tokens_per_expert
            for rank in range(2)
        ]
    )
    global_tpe = all_tpe.sum(dim=0)
    assert torch.equal(global_tpe, torch.tensor([3, 3, 3, 3, 1, 1, 1, 1]))
    assert torch.equal(
        global_tpe.reshape(2, 4).sum(dim=1),
        torch.tensor([12, 4]),
    )
