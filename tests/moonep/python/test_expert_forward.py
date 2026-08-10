from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import tools.moonep.expert_forward as expert_forward
from tools.moonep.contracts import (
    BackendUnavailableError,
    ContractError,
    MoonEPDimensions,
    MoonEPPlan,
    ProjectionTensors,
)
from tools.moonep.expert_forward import run_expert_forward


class RecordingTorchNpu:
    def __init__(self) -> None:
        self.calls = []

    def npu_grouped_matmul(self, **kwargs):
        self.calls.append(("grouped_matmul", kwargs))
        inputs = kwargs["x"][0]
        weights = kwargs["weight"][0]
        ends = [int(value) for value in kwargs["group_list"].tolist()]
        chunks = []
        begin = 0
        for group, end in enumerate(ends):
            chunks.append(inputs[begin:end].float().matmul(weights[group].float()))
            begin = end
        output = torch.cat(chunks, dim=0).to(dtype=kwargs["output_dtype"])
        return [output]

    def npu_swiglu(self, value):
        self.calls.append(("swiglu", value))
        gate, up = value.chunk(2, dim=-1)
        return (functional.silu(gate.float()) * up.float()).to(dtype=value.dtype)


def dimensions() -> MoonEPDimensions:
    return MoonEPDimensions(
        rank=0,
        world_size=1,
        tokens_per_rank=2,
        topk=1,
        expert_count=2,
        prefetch_slots=2,
        token_padding=2,
        hidden_size=2,
        intermediate_size=2,
    )


def plan() -> MoonEPPlan:
    d = dimensions()
    return MoonEPPlan(
        dimensions=d,
        dst=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens=torch.tensor([1, 2, 2, 3], dtype=torch.int32),
        experts_to_copy=torch.full((1, 2), -1, dtype=torch.int32),
        zero_fill_ranges=torch.zeros((d.vm_group_count, 2), dtype=torch.int32),
        remote_stats=torch.zeros((2,), dtype=torch.int32),
    )


def projections() -> ProjectionTensors:
    gate = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0]],
            [[4.0, 0.0], [0.0, 4.0]],
        ],
        dtype=torch.bfloat16,
    )
    up = gate + 1
    down = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.bfloat16,
    )
    return ProjectionTensors(gate=gate, up=up, down=down)


def test_expert_forward_uses_only_torch_npu_grouped_pipeline() -> None:
    d = dimensions()
    normalized_plan = plan()
    weights = projections()
    hidden = torch.arange(d.nvsh * d.hidden_size, dtype=torch.float32).reshape(
        d.nvsh, d.hidden_size
    ).to(dtype=torch.bfloat16)
    route_weights = torch.linspace(0.25, 1.0, d.nvsh, dtype=torch.float32)
    ops = RecordingTorchNpu()

    result = run_expert_forward(
        torch,
        hidden,
        normalized_plan.cu_seqlens,
        weights,
        route_weights,
        torch_npu_module=ops,
    )

    assert [call[0] for call in ops.calls] == [
        "grouped_matmul",
        "swiglu",
        "grouped_matmul",
    ]
    first = ops.calls[0][1]
    second = ops.calls[2][1]
    assert first["x"][0].shape == (3, d.hidden_size)
    assert first["x"][0].data_ptr() != hidden.data_ptr()
    assert first["weight"][0].shape == (
        d.vm_group_count,
        d.hidden_size,
        2 * d.intermediate_size,
    )
    assert torch.equal(
        first["weight"][0], torch.cat((weights.gate, weights.up), dim=-1)
    )
    assert first["group_list"].dtype == torch.int64
    assert torch.equal(
        first["group_list"], normalized_plan.cu_seqlens.to(dtype=torch.int64)
    )
    assert first["split_item"] == 3
    assert first["group_list_type"] == 0
    assert first["group_type"] == 0
    assert first["output_dtype"] == torch.bfloat16
    assert second["weight"][0].data_ptr() != weights.down.data_ptr()
    assert torch.equal(second["weight"][0], weights.down)
    assert result.hidden.shape == (d.nvsh, d.hidden_size)
    assert result.hidden.dtype == torch.bfloat16
    assert torch.count_nonzero(result.hidden[3:]).item() == 0

    unweighted = ops.calls[2][1]["x"][0]
    expected_prefix = RecordingTorchNpu().npu_grouped_matmul(
        x=[unweighted],
        weight=[weights.down],
        split_item=3,
        group_list_type=0,
        group_type=0,
        group_list=normalized_plan.cu_seqlens.to(dtype=torch.int64),
        output_dtype=torch.bfloat16,
    )[0]
    expected_prefix = expected_prefix * route_weights[:3].to(torch.bfloat16).reshape(3, 1)
    assert torch.equal(result.hidden[:3], expected_prefix)


def test_expert_forward_does_not_mutate_inputs() -> None:
    d = dimensions()
    normalized_plan = plan()
    weights = projections()
    hidden = torch.ones((d.nvsh, d.hidden_size), dtype=torch.bfloat16)
    route_weights = torch.ones((d.nvsh,), dtype=torch.float32)
    snapshots = (
        hidden.clone(),
        route_weights.clone(),
        weights.gate.clone(),
        weights.up.clone(),
        weights.down.clone(),
        normalized_plan.cu_seqlens.clone(),
    )

    run_expert_forward(
        torch,
        hidden,
        normalized_plan.cu_seqlens,
        weights,
        route_weights,
        torch_npu_module=RecordingTorchNpu(),
    )

    actual = (
        hidden,
        route_weights,
        weights.gate,
        weights.up,
        weights.down,
        normalized_plan.cu_seqlens,
    )
    assert all(torch.equal(value, before) for value, before in zip(actual, snapshots))


def test_expert_forward_accepts_projection_views_into_shared_backing() -> None:
    d = dimensions()
    original = projections()

    def offset_view(value):
        backing = torch.empty(value.numel() + 1, dtype=value.dtype)
        view = backing[1:].view_as(value)
        view.copy_(value)
        assert view.storage_offset() == 1
        return view

    weights = ProjectionTensors(
        gate=offset_view(original.gate),
        up=offset_view(original.up),
        down=offset_view(original.down),
    )
    result = run_expert_forward(
        torch,
        torch.ones((d.nvsh, d.hidden_size), dtype=torch.bfloat16),
        plan().cu_seqlens,
        weights,
        torch.ones((d.nvsh,), dtype=torch.float32),
        torch_npu_module=RecordingTorchNpu(),
    )

    assert result.hidden.shape == (d.nvsh, d.hidden_size)


def test_expert_forward_requires_torch_npu_without_fallback(monkeypatch) -> None:
    d = dimensions()

    def missing(_name):
        raise ImportError("torch_npu unavailable")

    monkeypatch.setattr(expert_forward.importlib, "import_module", missing)
    with pytest.raises(BackendUnavailableError, match="requires the torch_npu package"):
        run_expert_forward(
            torch,
            torch.ones((d.nvsh, d.hidden_size), dtype=torch.bfloat16),
            plan().cu_seqlens,
            projections(),
            torch.ones((d.nvsh,), dtype=torch.float32),
        )


def test_expert_forward_rejects_negative_group_boundary() -> None:
    d = dimensions()
    group_list = plan().cu_seqlens
    group_list[0] = -1
    ops = RecordingTorchNpu()

    with pytest.raises(ContractError, match="must be non-negative"):
        run_expert_forward(
            torch,
            torch.ones((d.nvsh, d.hidden_size), dtype=torch.bfloat16),
            group_list,
            projections(),
            torch.ones((d.nvsh,), dtype=torch.float32),
            torch_npu_module=ops,
        )

    assert ops.calls == []
