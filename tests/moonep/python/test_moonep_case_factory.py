from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.moonep.case_factory import make_correctness_case
from tools.moonep.config import load_cases, select_cases
from tools.moonep.contracts import MoonEPDimensions, validate_projections
from tools.moonep.correctness import _has_duplicate_destinations
from tools.moonep.torch_npu_backend import TorchNpuMoonEPBackend


RUNNER_WORLD_SIZES = {
    "manual-small": 1,
    "manual-2rank-imbalanced": 2,
    "manual-2rank-topk-2": 2,
    "planning-small": 2,
    "planning-no-dedup": 4,
    "planning-4rank-topk-4": 4,
    "skewed-no-dup": 4,
    "planning-8rank-topk-8": 8,
    "planning-16rank-topk-16": 16,
    "planning-8rank-single-route": 8,
    "planning-16rank-single-route": 16,
    "planning-16rank-16card-single-route": 16,
    "planning-64rank-single-route": 64,
    "planning-128rank-single-route": 128,
    "dispatch-8rank-4k-ep8-grouped-urma": 8,
    "flow-8rank-4k-ep8-grouped-urma-plan-reuse": 8,
    "model-flow-8rank-4k-ep8-mindspeed": 8,
    "model-flow-16rank-4k-ep16-mindspeed": 16,
    "model-flow-8rank-8k-k16-ep8-mindspeed": 8,
    "model-flow-16rank-8k-k16-ep16-mindspeed": 16,
}

DISPATCH_ONLY_REPRO_CASE = "dispatch-8rank-4k-ep8-grouped-urma"
PRODUCTION_SCALE_REPRO_CASES = {
    DISPATCH_ONLY_REPRO_CASE,
    "flow-8rank-4k-ep8-grouped-urma-plan-reuse",
    "model-flow-8rank-4k-ep8-mindspeed",
    "model-flow-16rank-4k-ep16-mindspeed",
    "model-flow-8rank-8k-k16-ep8-mindspeed",
    "model-flow-16rank-8k-k16-ep16-mindspeed",
}


class StaticCollective:
    def __init__(self, gathered):
        self.gathered = gathered

    def all_gather(self, tensor):
        return self.gathered.to(device=tensor.device)


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


def test_unique_destination_pattern_rejects_more_routes_than_ranks() -> None:
    with pytest.raises(ValueError, match="topk.*world_size"):
        make_correctness_case(
            torch,
            MoonEPDimensions(
                rank=0,
                world_size=2,
                tokens_per_rank=2,
                topk=3,
                expert_count=6,
                prefetch_slots=3,
                token_padding=1,
                hidden_size=2,
                intermediate_size=2,
            ),
            routing_pattern="unique_destinations",
        )


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


def test_runner_case_matrix_uses_fixed_padding_and_unique_token_destinations() -> None:
    root = Path(__file__).resolve().parents[3]
    case_path = root / "tools" / "moonep" / "cases" / "correctness.json"
    raw_cases = json.loads(case_path.read_text(encoding="utf-8"))
    assert all(case.get("P") == 1 for case in raw_cases)
    assert all(
        case["routing"] not in {"duplicate_destinations", "imbalanced_duplicates"}
        for case in raw_cases
    )

    cases = load_cases(case_path)
    assert all(case.route_distribution == "rank_shifted_uniform" for case in cases)
    assert {case.case_id for case in cases} == set(RUNNER_WORLD_SIZES)

    for case in cases:
        world_size = RUNNER_WORLD_SIZES[case.case_id]
        assert case.token_padding == 1
        assert case.prefetch_slots == case.expert_count // world_size
        if case.case_id in PRODUCTION_SCALE_REPRO_CASES:
            continue
        assert (case.hidden_size, case.intermediate_size) == (8, 4)
        assert case.hidden_size * case.intermediate_size * 2 % 64 == 0
        for rank in range(world_size):
            dimensions = MoonEPDimensions(
                rank,
                world_size,
                case.tokens_per_rank,
                case.topk,
                case.expert_count,
                case.prefetch_slots,
                case.token_padding,
                case.hidden_size,
                case.intermediate_size,
            )
            generated = make_correctness_case(
                torch, dimensions, routing_pattern=case.routing_pattern
            )
            owners = torch.div(
                generated.topk_experts,
                dimensions.experts_per_rank,
                rounding_mode="floor",
            )
            for token_owners in owners.tolist():
                assert len(set(token_owners)) == case.topk


def test_runner_plans_do_not_assign_one_token_to_the_same_rank_twice() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    for case in cases:
        if case.case_id in PRODUCTION_SCALE_REPRO_CASES:
            continue
        world_size = RUNNER_WORLD_SIZES[case.case_id]
        dimensions_by_rank = [
            MoonEPDimensions(
                rank,
                world_size,
                case.tokens_per_rank,
                case.topk,
                case.expert_count,
                case.prefetch_slots,
                case.token_padding,
                case.hidden_size,
                case.intermediate_size,
            )
            for rank in range(world_size)
        ]
        generated = [
            make_correctness_case(
                torch, dimensions, routing_pattern=case.routing_pattern
            )
            for dimensions in dimensions_by_rank
        ]
        all_tpe = torch.stack([item.tokens_per_expert for item in generated])
        for dimensions, item in zip(dimensions_by_rank, generated):
            plan = TorchNpuMoonEPBackend(
                torch, dimensions, collective=StaticCollective(all_tpe)
            ).planning(item.topk_experts, item.tokens_per_expert).plan
            assert not _has_duplicate_destinations(torch, plan), case.case_id


def test_runner_case_matrix_varies_core_dimensions_and_retains_topk_above_one() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    for name in (
        "tokens_per_rank",
        "topk",
        "expert_count",
        "prefetch_slots",
    ):
        assert len({getattr(case, name) for case in cases}) >= 2
    assert {case.topk for case in cases} >= {1, 2, 4, 8, 16}


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("planning-8rank-topk-8", (4, 8, 32, 4, 1)),
        ("planning-16rank-topk-16", (2, 16, 32, 2, 1)),
    ],
)
def test_large_rank_cases_keep_bounded_routes_and_required_capacity(
    case_id: str, expected: tuple[int, ...]
) -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")
    case = next(item for item in cases if item.case_id == case_id)

    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.prefetch_slots,
        case.token_padding,
    ) == expected
    assert case.tokens_per_rank * case.topk == 32
    assert case.routing_pattern == "unique_destinations"


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("planning-8rank-single-route", (8, 1, 16, 2, 1)),
        ("planning-16rank-single-route", (8, 1, 16, 1, 1)),
        ("planning-16rank-16card-single-route", (8, 1, 16, 1, 1)),
        ("planning-64rank-single-route", (8, 1, 64, 1, 1)),
        ("planning-128rank-single-route", (8, 1, 128, 1, 1)),
    ],
)
def test_large_rank_single_route_cases_isolate_topk_from_rank_scaling(
    case_id: str, expected: tuple[int, ...]
) -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")
    case = next(item for item in cases if item.case_id == case_id)

    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.prefetch_slots,
        case.token_padding,
    ) == expected
    assert case.routing_pattern == "balanced"


def test_large_rank_single_route_cases_share_core_dimensions() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")
    selected = {
        case.case_id: case
        for case in cases
        if case.case_id
        in {"planning-8rank-single-route", "planning-16rank-single-route"}
    }
    eight = selected["planning-8rank-single-route"]
    sixteen = selected["planning-16rank-single-route"]

    assert (
        eight.tokens_per_rank,
        eight.topk,
        eight.expert_count,
        eight.hidden_size,
        eight.intermediate_size,
        eight.token_padding,
        eight.routing_pattern,
    ) == (
        sixteen.tokens_per_rank,
        sixteen.topk,
        sixteen.expert_count,
        sixteen.hidden_size,
        sixteen.intermediate_size,
        sixteen.token_padding,
        sixteen.routing_pattern,
    )
    assert eight.prefetch_slots == eight.expert_count // 8
    assert sixteen.prefetch_slots == sixteen.expert_count // 16


def test_runner_cases_support_one_based_numeric_selection() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    for number, case in enumerate(cases, start=1):
        assert select_cases(cases, str(number)) == [case]
    assert select_cases(cases, "1,16") == [cases[0], cases[15]]
    for invalid in ("0", "21"):
        with pytest.raises(ValueError, match=r"case number must be in \[1, 20\]"):
            select_cases(cases, invalid)


def test_case_14_is_the_two_node_one_rank_per_device_case() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    assert len(cases) == 20
    case = cases[13]
    assert case.case_id == "planning-16rank-16card-single-route"
    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.hidden_size,
        case.intermediate_size,
        case.prefetch_slots,
        case.token_padding,
        case.routing_pattern,
        case.route_distribution,
    ) == (8, 1, 16, 8, 4, 1, 1, "balanced", "rank_shifted_uniform")


def test_case_15_matches_the_4k_ep8_grouped_urma_dispatch_repro() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    case = cases[14]
    assert case.case_id == "dispatch-8rank-4k-ep8-grouped-urma"
    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.hidden_size,
        case.intermediate_size,
        case.prefetch_slots,
        case.token_padding,
        case.routing_pattern,
        case.route_distribution,
        case.warmup,
        case.iterations,
    ) == (
        4096,
        8,
        32,
        7168,
        2048,
        4,
        1,
        "unique_destinations",
        "rank_shifted_uniform",
        0,
        8,
    )


def test_case_16_matches_the_4k_ep8_grouped_urma_plan_reuse_repro() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    case = cases[15]
    assert case.case_id == "flow-8rank-4k-ep8-grouped-urma-plan-reuse"
    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.hidden_size,
        case.intermediate_size,
        case.prefetch_slots,
        case.token_padding,
        case.routing_pattern,
        case.route_distribution,
        case.warmup,
        case.iterations,
    ) == (
        4096,
        8,
        32,
        7168,
        2048,
        4,
        1,
        "unique_destinations",
        "rank_shifted_uniform",
        0,
        8,
    )


def test_case_17_matches_the_mindspeed_model_iteration() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    case = cases[16]
    assert case.case_id == "model-flow-8rank-4k-ep8-mindspeed"
    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.hidden_size,
        case.intermediate_size,
        case.prefetch_slots,
        case.token_padding,
        case.routing_pattern,
        case.warmup,
        case.iterations,
    ) == (
        4096,
        8,
        32,
        7168,
        2048,
        4,
        1,
        "model_replay",
        5,
        20,
    )


@pytest.mark.parametrize(
    ("case_number", "case_id", "expected"),
    [
        (
            18,
            "model-flow-16rank-4k-ep16-mindspeed",
            (4096, 8, 32, 7168, 2048, 2),
        ),
        (
            19,
            "model-flow-8rank-8k-k16-ep8-mindspeed",
            (8192, 16, 32, 3584, 2048, 4),
        ),
        (
            20,
            "model-flow-16rank-8k-k16-ep16-mindspeed",
            (8192, 16, 32, 3584, 2048, 2),
        ),
    ],
)
def test_model_flow_scale_cases(case_number: int, case_id: str, expected) -> None:
    root = Path(__file__).resolve().parents[3]
    cases = load_cases(root / "tools" / "moonep" / "cases" / "correctness.json")

    case = cases[case_number - 1]
    assert case.case_id == case_id
    assert (
        case.tokens_per_rank,
        case.topk,
        case.expert_count,
        case.hidden_size,
        case.intermediate_size,
        case.prefetch_slots,
    ) == expected
    assert (case.token_padding, case.routing_pattern, case.warmup, case.iterations) == (
        1,
        "model_replay",
        5,
        20,
    )
