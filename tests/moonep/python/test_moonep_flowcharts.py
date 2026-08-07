from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.moonep.flowcharts import FLOWCHART_STAGES, generate_mermaid_sources


def _write_snapshot(
    case_dir: Path,
    *,
    rank: int,
    stage: str,
    direction: str,
    value,
    role: str = "reference",
) -> None:
    target = case_dir / f"rank_{rank}" / "tensor_dumps" / stage / role
    target.mkdir(parents=True, exist_ok=True)
    torch.save(value, target / f"{direction}.pt")


def _stage_snapshots(rank: int) -> dict[str, tuple[object, object]]:
    plan = {
        "dimensions": {
            "rank": rank,
            "world_size": 2,
            "tokens_per_rank": 2,
            "topk": 2,
            "expert_count": 4,
            "prefetch_slots": 1,
            "token_padding": 1,
            "hidden_size": 2,
            "intermediate_size": 2,
        },
        "dst": torch.tensor([rank * 4, -(rank * 4 + 2)], dtype=torch.int32),
        "cu_seqlens": torch.tensor([1, 2, 2, 2, 2], dtype=torch.int32),
        "experts_to_copy": torch.tensor([[-1], [0]], dtype=torch.int32),
        "zero_fill_ranges": torch.zeros((5, 2), dtype=torch.int32),
        "remote_stats": torch.tensor([rank, 1 - rank], dtype=torch.int32),
    }
    dispatched_plan = {
        **plan,
        "dedup": {
            "groups": torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.int32),
            "loffs": torch.tensor([1, 0], dtype=torch.int32),
            "counts": torch.tensor([1, 1], dtype=torch.int32),
        },
    }
    hidden = torch.tensor([[rank + 1, rank + 2], [rank + 3, rank + 4]])
    projections = {
        name: torch.arange(20, dtype=torch.bfloat16).reshape(5, 2, 2) + rank
        for name in ("gate", "up", "down")
    }
    full_grads = {
        name: torch.arange(20, dtype=torch.float32).reshape(5, 2, 2) + rank
        for name in ("gate", "up", "down")
    }
    reduce_buffers = {
        name: torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2) + rank
        for name in ("gate", "up", "down")
    }
    return {
        "planning": (
            {
                "topk_experts": torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
                "tokens_per_expert": torch.tensor([2, 2, 0, 0], dtype=torch.int32),
            },
            {"plan": plan},
        ),
        "dispatch": (
            {
                "plan": plan,
                "hidden": hidden.to(torch.bfloat16),
                "route_weights": torch.arange(4, dtype=torch.float32) + rank,
            },
            {
                "plan": dispatched_plan,
                "hidden": hidden.to(torch.bfloat16),
                "route_weights": torch.arange(2, dtype=torch.float32) + rank,
            },
        ),
        "prefetch_weight": (
            {"plan": dispatched_plan, "projections": projections},
            {"projections": projections},
        ),
        "expert_forward": (
            {
                "hidden": hidden.to(torch.bfloat16),
                "cu_seqlens": plan["cu_seqlens"],
                "projections": projections,
                "route_weights": torch.arange(2, dtype=torch.float32) + rank,
            },
            {
                "hidden": torch.tensor(
                    [[1585446912.0, 1602224128.0], [3.0, 4.0]],
                    dtype=torch.bfloat16,
                )
            },
        ),
        "combine": (
            {
                "plan": dispatched_plan,
                "expert_output": (hidden * 2).to(torch.bfloat16),
                "route_weights": torch.arange(2, dtype=torch.float32) + rank,
            },
            {
                "hidden": hidden.to(torch.bfloat16),
                "route_weights": torch.arange(4, dtype=torch.float32) + rank,
            },
        ),
        "reduce_grad": (
            {
                "plan": dispatched_plan,
                "full_grads": full_grads,
                "reduce_buffers": reduce_buffers,
            },
            {"full_grads": full_grads, "reduce_buffers": reduce_buffers},
        ),
    }


def _write_complete_case(case_dir: Path) -> None:
    for rank in range(2):
        for stage, (stage_input, stage_output) in _stage_snapshots(rank).items():
            _write_snapshot(
                case_dir,
                rank=rank,
                stage=stage,
                direction="input",
                value=stage_input,
            )
            _write_snapshot(
                case_dir,
                rank=rank,
                stage=stage,
                direction="output",
                value=stage_output,
            )
            _write_snapshot(
                case_dir,
                rank=rank,
                stage=stage,
                direction="input",
                value={"candidate_only": torch.tensor([999999])},
                role="candidate",
            )


def test_generates_numbered_lr_sources_from_reference_snapshots(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    output_dir = case_dir / "flowcharts"
    _write_complete_case(case_dir)

    paths = generate_mermaid_sources(
        torch,
        case_dir=case_dir,
        world_size=2,
        output_dir=output_dir,
    )

    expected_names = [f"{prefix}-2rank-detailed-lr.mmd" for prefix, _, _ in FLOWCHART_STAGES]
    assert [path.name for path in paths] == expected_names
    assert all(path.is_file() for path in paths)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "flowchart LR" in source
        assert '"fontFamily":"Microsoft YaHei, Segoe UI, sans-serif"' in source
        assert '"diagramPadding":40' in source
        assert "Rank 0 输入" in source
        assert "Rank 1 输出" in source
        assert "999999" not in source
    expert_source = (output_dir / expected_names[3]).read_text(encoding="utf-8")
    assert "1585446912.0" in expert_source
    assert "1.585447e+09" not in expert_source


def test_missing_reference_snapshot_fails_with_exact_path(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_complete_case(case_dir)
    missing = (
        case_dir
        / "rank_1"
        / "tensor_dumps"
        / "combine"
        / "reference"
        / "output.pt"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=r"rank_1.*combine.*output\.pt"):
        generate_mermaid_sources(
            torch,
            case_dir=case_dir,
            world_size=2,
            output_dir=case_dir / "flowcharts",
        )
