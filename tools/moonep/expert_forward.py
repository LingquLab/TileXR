from __future__ import annotations

import importlib

from .contracts import (
    BackendUnavailableError,
    ContractError,
    ExpertForwardResult,
    ProjectionTensors,
    validate_tensor,
)


def _load_torch_npu(torch_npu_module):
    if torch_npu_module is None:
        try:
            torch_npu_module = importlib.import_module("torch_npu")
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(
                "Torch-NPU Expert Forward requires the torch_npu package"
            ) from exc
    for name in ("npu_grouped_matmul", "npu_swiglu"):
        if not callable(getattr(torch_npu_module, name, None)):
            raise BackendUnavailableError(
                f"Torch-NPU Expert Forward requires torch_npu.{name}"
            )
    return torch_npu_module


def _single_output(value, name: str):
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise ContractError(f"torch_npu.{name} must return one tensor")
    return value[0]


def run_expert_forward(
    torch_module,
    hidden_nvsh,
    group_list,
    projections: ProjectionTensors,
    route_weights_nvs,
    *,
    torch_npu_module=None,
) -> ExpertForwardResult:
    if not isinstance(projections, ProjectionTensors) and not all(
        hasattr(projections, name) for name in ("gate", "up", "down")
    ):
        raise ContractError("projections must provide gate, up, and down tensors")

    group_count, hidden_size, intermediate_size = (
        int(value) for value in projections.gate.shape
    )
    nvsh = int(hidden_nvsh.shape[0])
    validate_tensor(
        hidden_nvsh,
        "expert_forward.hidden",
        shape=(nvsh, hidden_size),
        dtype=torch_module.bfloat16,
    )
    validate_tensor(
        group_list,
        "expert_forward.group_list",
        shape=(group_count,),
        dtype=torch_module.int32,
    )
    validate_tensor(
        projections.gate,
        "expert_forward.gate",
        shape=(group_count, hidden_size, intermediate_size),
        dtype=torch_module.bfloat16,
        allow_storage_offset=True,
    )
    validate_tensor(
        projections.up,
        "expert_forward.up",
        shape=(group_count, hidden_size, intermediate_size),
        dtype=torch_module.bfloat16,
        allow_storage_offset=True,
    )
    validate_tensor(
        projections.down,
        "expert_forward.down",
        shape=(group_count, intermediate_size, hidden_size),
        dtype=torch_module.bfloat16,
        allow_storage_offset=True,
    )
    validate_tensor(
        route_weights_nvs,
        "expert_forward.route_weights",
        shape=(nvsh,),
        dtype=torch_module.float32,
    )
    if bool((group_list < 0).any().item()):
        raise ContractError("expert_forward.group_list must be non-negative")
    if bool((group_list[1:] < group_list[:-1]).any().item()):
        raise ContractError("expert_forward.group_list must be monotonic")
    valid_rows = int(group_list[-1].item())
    if valid_rows < 0 or valid_rows > nvsh:
        raise ContractError("expert_forward.group_list exceeds hidden capacity")

    output = torch_module.zeros(
        (nvsh, hidden_size), dtype=torch_module.bfloat16, device=hidden_nvsh.device
    )
    if valid_rows == 0:
        return ExpertForwardResult(output)

    torch_npu_module = _load_torch_npu(torch_npu_module)
    gate = projections.gate.clone()
    up = projections.up.clone()
    down = projections.down.clone()
    packed_gate_up = torch_module.cat((gate, up), dim=-1).contiguous()
    gmm_group_list = group_list.to(dtype=torch_module.int64).contiguous()
    grouped_args = {
        "split_item": 3,
        "group_list_type": 0,
        "group_type": 0,
        "group_list": gmm_group_list,
        "output_dtype": torch_module.bfloat16,
    }
    gate_up = _single_output(
        torch_npu_module.npu_grouped_matmul(
            x=[hidden_nvsh[:valid_rows].clone()],
            weight=[packed_gate_up],
            **grouped_args,
        ),
        "npu_grouped_matmul",
    )
    validate_tensor(
        gate_up,
        "expert_forward.gmm1_output",
        shape=(valid_rows, 2 * intermediate_size),
        dtype=torch_module.bfloat16,
    )
    activated = torch_npu_module.npu_swiglu(gate_up)
    validate_tensor(
        activated,
        "expert_forward.swiglu_output",
        shape=(valid_rows, intermediate_size),
        dtype=torch_module.bfloat16,
    )
    expert_output = _single_output(
        torch_npu_module.npu_grouped_matmul(
            x=[activated], weight=[down], **grouped_args
        ),
        "npu_grouped_matmul",
    )
    validate_tensor(
        expert_output,
        "expert_forward.gmm2_output",
        shape=(valid_rows, hidden_size),
        dtype=torch_module.bfloat16,
    )
    weighted = expert_output * route_weights_nvs[:valid_rows].to(
        dtype=torch_module.bfloat16
    ).reshape(valid_rows, 1)
    output[:valid_rows].copy_(weighted)
    return ExpertForwardResult(output)
