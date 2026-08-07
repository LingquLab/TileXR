from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tools.moonep.contracts import (
    ContractError,
    MoonEPDimensions,
    ProjectionTensors,
    clone_value,
    validate_projections,
    validate_tensor,
)


def dimensions() -> MoonEPDimensions:
    return MoonEPDimensions(
        rank=0,
        world_size=2,
        tokens_per_rank=4,
        topk=2,
        expert_count=4,
        prefetch_slots=2,
        token_padding=4,
        hidden_size=4,
        intermediate_size=3,
    )


def test_dimensions_derive_upstream_capacity() -> None:
    d = dimensions()
    assert d.route_count == 8
    assert d.experts_per_rank == 2
    assert d.vm_group_count == 6
    assert d.nvsh == 20


def test_dimensions_reject_invalid_prefetch_slots() -> None:
    with pytest.raises(ContractError, match="prefetch_slots"):
        replace(dimensions(), prefetch_slots=3)


def test_validate_tensor_names_shape_and_dtype_errors() -> None:
    value = torch.zeros((2, 3), dtype=torch.float32)
    validate_tensor(value, "hidden", shape=(2, 3), dtype=torch.float32)
    with pytest.raises(ContractError, match=r"hidden must have shape"):
        validate_tensor(value, "hidden", shape=(3, 2), dtype=torch.float32)
    with pytest.raises(ContractError, match=r"hidden must have dtype"):
        validate_tensor(value, "hidden", shape=(2, 3), dtype=torch.bfloat16)


def test_projection_clone_has_independent_storage() -> None:
    tensors = ProjectionTensors(
        gate=torch.ones((6, 4, 3), dtype=torch.bfloat16),
        up=torch.ones((6, 4, 3), dtype=torch.bfloat16),
        down=torch.ones((6, 3, 4), dtype=torch.bfloat16),
    )
    cloned = clone_value(tensors)
    cloned.gate.zero_()
    assert torch.count_nonzero(tensors.gate).item() == tensors.gate.numel()
    assert cloned.gate.data_ptr() != tensors.gate.data_ptr()
    validate_projections(cloned, dimensions(), torch, dtype=torch.bfloat16)


def test_projection_validation_checks_reduce_shape() -> None:
    d = dimensions()
    buffers = ProjectionTensors(
        gate=torch.zeros((2, 2, 4, 3)),
        up=torch.zeros((2, 2, 4, 3)),
        down=torch.zeros((2, 2, 3, 4)),
    )
    validate_projections(
        buffers, d, torch, dtype=torch.float32, reduce_buffers=True
    )
    buffers.down = torch.zeros((2, 2, 4, 3))
    with pytest.raises(ContractError, match=r"down must have shape"):
        validate_projections(
            buffers, d, torch, dtype=torch.float32, reduce_buffers=True
        )


def test_active_python_does_not_depend_on_analysis_checkout() -> None:
    root = Path(__file__).resolve().parents[3]
    forbidden = ("3rd" + "party/moonep", "3rd" + "party\\moonep")
    violations = []
    for relative in ("tools/moonep", "integrations/moonep_torch", "tests/moonep/python"):
        for path in (root / relative).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in forbidden):
                violations.append(path.relative_to(root).as_posix())
    assert violations == []
