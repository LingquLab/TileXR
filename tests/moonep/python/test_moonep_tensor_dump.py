from __future__ import annotations

import json

import torch

from tools.moonep.contracts import MoonEPDimensions, MoonEPPlan, PlanningResult
from tools.moonep.tensor_dump import StageTensorDumper


def _plan() -> MoonEPPlan:
    dimensions = MoonEPDimensions(0, 1, 2, 1, 2, 2, 2, 3, 4)
    return MoonEPPlan(
        dimensions=dimensions,
        dst=torch.tensor([3, -2], dtype=torch.int32),
        cu_seqlens=torch.tensor([1, 2], dtype=torch.int32),
        experts_to_copy=torch.tensor([[0, -1]], dtype=torch.int32),
        zero_fill_ranges=torch.tensor([[1, 1]], dtype=torch.int32),
        remote_stats=torch.tensor([[2, 1]], dtype=torch.int32),
    )


def test_dump_saves_complete_cpu_snapshot_and_bounded_manifest_preview(tmp_path) -> None:
    previews = []
    dumper = StageTensorDumper(
        torch,
        tmp_path,
        preview_elements=3,
        preview_sink=previews.append,
    )
    inputs = {
        "topk_experts": torch.arange(8, dtype=torch.int32).reshape(4, 2),
        "tokens_per_expert": torch.tensor([2, 2, 2, 2], dtype=torch.int32),
    }

    dumper.dump(
        "planning",
        "reference",
        "torch_npu_reference_v1",
        "input",
        inputs,
    )

    snapshot_path = tmp_path / "planning" / "reference" / "input.pt"
    manifest_path = tmp_path / "planning" / "reference" / "input.json"
    readable_path = tmp_path / "planning" / "reference" / "input.txt"
    snapshot = torch.load(snapshot_path, weights_only=False)
    assert torch.equal(snapshot["topk_experts"], inputs["topk_experts"].cpu())
    assert snapshot["topk_experts"].device.type == "cpu"
    assert torch.equal(snapshot["tokens_per_expert"], inputs["tokens_per_expert"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend_role"] == "reference"
    assert manifest["backend_name"] == "torch_npu_reference_v1"
    assert manifest["snapshot"] == "input.pt"
    assert manifest["human_readable"] == "input.txt"
    tensor = next(item for item in manifest["tensors"] if item["path"] == "topk_experts")
    assert tensor == {
        "path": "topk_experts",
        "shape": [4, 2],
        "dtype": "torch.int32",
        "device": "cpu",
        "numel": 8,
        "preview": [0, 1, 2],
        "preview_truncated": True,
    }
    assert previews == [
        "[tensor-dump] planning/reference/input topk_experts "
        "shape=[4, 2] dtype=torch.int32 preview=[0, 1, 2]"
        ,
        "[tensor-dump] planning/reference/input tokens_per_expert "
        "shape=[4] dtype=torch.int32 preview=[2, 2, 2]",
    ]
    readable = readable_path.read_text(encoding="utf-8")
    assert "stage: planning" in readable
    assert "[tensor] topk_experts" in readable
    assert "shape: [4, 2]" in readable
    assert "dtype: torch.int32" in readable
    assert "device: cpu" in readable
    assert "values:\n[[0, 1], [2, 3], [4, 5], [6, 7]]" in readable
    assert "..." not in readable


def test_dump_recurses_through_stage_dataclasses_and_preserves_scalar_metadata(tmp_path) -> None:
    dumper = StageTensorDumper(torch, tmp_path, preview_elements=2, preview_sink=None)
    dumper.dump(
        "planning",
        "candidate",
        "tilexr",
        "output",
        PlanningResult(_plan()),
    )

    snapshot = torch.load(
        tmp_path / "planning" / "candidate" / "output.pt",
        weights_only=False,
    )
    assert snapshot["__type__"] == "PlanningResult"
    assert snapshot["plan"]["__type__"] == "MoonEPPlan"
    assert snapshot["plan"]["dimensions"]["hidden_size"] == 3
    assert torch.equal(snapshot["plan"]["dst"], torch.tensor([3, -2], dtype=torch.int32))

    manifest = json.loads(
        (tmp_path / "planning" / "candidate" / "output.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["values"]["plan.dimensions.hidden_size"] == 3
    assert {item["path"] for item in manifest["tensors"]} >= {
        "plan.dst",
        "plan.cu_seqlens",
        "plan.experts_to_copy",
    }
    readable = (
        tmp_path / "planning" / "candidate" / "output.txt"
    ).read_text(encoding="utf-8")
    assert "[value] plan.dimensions.hidden_size = 3" in readable
    assert "[tensor] plan.dst" in readable
    assert "values:\n[3, -2]" in readable


def test_reference_and_candidate_snapshots_are_separate_and_preview_log_is_complete(tmp_path) -> None:
    dumper = StageTensorDumper(torch, tmp_path, preview_elements=1)
    dumper.dump("dispatch", "reference", "ref", "output", {"hidden": torch.tensor([1.0])})
    dumper.dump("dispatch", "candidate", "dut", "output", {"hidden": torch.tensor([2.0])})

    reference = torch.load(
        tmp_path / "dispatch" / "reference" / "output.pt", weights_only=False
    )
    candidate = torch.load(
        tmp_path / "dispatch" / "candidate" / "output.pt", weights_only=False
    )
    assert reference["hidden"].item() == 1.0
    assert candidate["hidden"].item() == 2.0
    preview_log = (tmp_path / "preview.log").read_text(encoding="utf-8")
    assert "dispatch/reference/output" in preview_log
    assert "dispatch/candidate/output" in preview_log
