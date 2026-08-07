from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from tools.moonep.case_factory import make_correctness_case
from tools.moonep.contracts import (
    BackendUnavailableError,
    ContractError,
    MoonEPDimensions,
)
from tools.moonep.correctness import CorrectnessError, CorrectnessRunner, STAGES
from tools.moonep.torch_npu_backend import TorchNpuMoonEPBackend


class CpuTorchNpu:
    def __init__(self, *, fault_gmm_call: int | None = None) -> None:
        self.fault_gmm_call = fault_gmm_call
        self.gmm_calls = 0

    def npu_grouped_matmul(self, **kwargs):
        self.gmm_calls += 1
        inputs = kwargs["x"][0]
        weights = kwargs["weight"][0]
        ends = [int(value) for value in kwargs["group_list"].tolist()]
        chunks = []
        begin = 0
        for group, end in enumerate(ends):
            chunks.append(inputs[begin:end].float().matmul(weights[group].float()))
            begin = end
        output = torch.cat(chunks, dim=0).to(dtype=kwargs["output_dtype"])
        if self.gmm_calls == self.fault_gmm_call:
            output.zero_()
        return [output]

    @staticmethod
    def npu_swiglu(value):
        gate, up = value.chunk(2, dim=-1)
        return (functional.silu(gate.float()) * up.float()).to(dtype=value.dtype)


def dimensions() -> MoonEPDimensions:
    return MoonEPDimensions(
        rank=0,
        world_size=1,
        tokens_per_rank=4,
        topk=2,
        expert_count=4,
        prefetch_slots=4,
        token_padding=4,
        hidden_size=4,
        intermediate_size=3,
    )


class FaultBackend:
    def __init__(self, backend, fault_stage: str):
        self.backend = backend
        self.name = f"fault_{fault_stage}"
        self.dimensions = backend.dimensions
        self.fault_stage = fault_stage
        self.calls = []

    def synchronize(self):
        self.backend.synchronize()

    def close(self):
        self.backend.close()

    def planning(self, *args):
        self.calls.append("planning")
        result = self.backend.planning(*args)
        if self.fault_stage == "planning":
            result.plan.dst[0] += 1
        return result

    def dispatch(self, *args):
        self.calls.append("dispatch")
        result = self.backend.dispatch(*args)
        if self.fault_stage == "dispatch":
            result.hidden[0, 0] += 1
        return result

    def prefetch_weight(self, *args):
        self.calls.append("prefetch_weight")
        result = self.backend.prefetch_weight(*args)
        if self.fault_stage == "prefetch_weight":
            result.projections.gate[0, 0, 0] += 1
        return result

    def combine(self, *args):
        self.calls.append("combine")
        result = self.backend.combine(*args)
        if self.fault_stage == "combine":
            result.hidden.zero_()
        return result

    def reduce_grad(self, *args):
        self.calls.append("reduce_grad")
        result = self.backend.reduce_grad(*args)
        if self.fault_stage == "reduce_grad":
            result.full_grads.gate[self.dimensions.expert_count, 0, 0] += 1
        return result


BACKEND_STAGES = tuple(stage for stage in STAGES if stage != "expert_forward")


def test_reference_and_identical_candidate_pass_all_stage_gates(tmp_path) -> None:
    case = make_correctness_case(torch, dimensions())
    reference = TorchNpuMoonEPBackend(torch, dimensions())
    candidate = TorchNpuMoonEPBackend(torch, dimensions())
    report = CorrectnessRunner(
        torch,
        reference,
        candidate,
        artifact_dir=tmp_path,
        expert_ops=CpuTorchNpu(),
    ).run_differential(case)
    assert report.passed
    assert [stage["stage"] for stage in report.stages] == list(STAGES)
    assert len(list(tmp_path.glob("*.json"))) == len(STAGES)


def test_candidate_without_dedup_support_rejects_duplicate_destinations_before_dispatch() -> None:
    case = make_correctness_case(torch, dimensions())
    reference = TorchNpuMoonEPBackend(torch, dimensions())
    candidate = FaultBackend(TorchNpuMoonEPBackend(torch, dimensions()), "none")
    candidate.supports_duplicate_destinations = False

    with pytest.raises(CorrectnessError) as captured:
        CorrectnessRunner(
            torch, reference, candidate, expert_ops=CpuTorchNpu()
        ).run_differential(case)

    assert captured.value.artifact["stage"] == "dispatch"
    assert "does not support duplicate destination ranks" in str(captured.value)
    assert candidate.calls == ["planning"]


def test_candidate_without_dedup_support_accepts_single_route_case() -> None:
    single_route = MoonEPDimensions(0, 1, 4, 1, 4, 4, 4, 4, 3)
    case = make_correctness_case(torch, single_route)
    reference = TorchNpuMoonEPBackend(torch, single_route)
    candidate = FaultBackend(TorchNpuMoonEPBackend(torch, single_route), "none")
    candidate.supports_duplicate_destinations = False

    report = CorrectnessRunner(
        torch, reference, candidate, expert_ops=CpuTorchNpu()
    ).run_differential(case)

    assert report.passed


def test_manual_small_reference_passes_and_writes_complete_readable_boundaries(
    tmp_path,
) -> None:
    small = MoonEPDimensions(0, 1, 2, 2, 4, 4, 1, 2, 2)
    case = make_correctness_case(torch, small, routing_pattern="skewed")
    dump_dir = tmp_path / "tensor_dumps"
    report = CorrectnessRunner(
        torch,
        TorchNpuMoonEPBackend(torch, small),
        tensor_dump_dir=dump_dir,
        preview_elements=2,
        preview_sink=lambda _line: None,
        expert_ops=CpuTorchNpu(),
    ).run_reference(case)

    assert report.passed
    assert len(list(dump_dir.glob("*/reference/*.txt"))) == len(STAGES) * 2


def test_runner_dumps_each_stage_input_and_output_for_both_backends(tmp_path) -> None:
    case = make_correctness_case(torch, dimensions())
    reference = TorchNpuMoonEPBackend(torch, dimensions())
    candidate = TorchNpuMoonEPBackend(torch, dimensions())
    dump_dir = tmp_path / "tensor_dumps"

    report = CorrectnessRunner(
        torch,
        reference,
        candidate,
        tensor_dump_dir=dump_dir,
        preview_elements=2,
        preview_sink=lambda _line: None,
        expert_ops=CpuTorchNpu(),
    ).run_differential(case)

    assert report.passed
    for stage in STAGES:
        for role in ("reference", "candidate"):
            for direction in ("input", "output"):
                assert (dump_dir / stage / role / f"{direction}.pt").is_file()
                assert (dump_dir / stage / role / f"{direction}.json").is_file()


@pytest.mark.parametrize("fault_stage", BACKEND_STAGES)
def test_fault_stops_at_owning_stage(fault_stage: str) -> None:
    case = make_correctness_case(torch, dimensions())
    reference = TorchNpuMoonEPBackend(torch, dimensions())
    candidate = FaultBackend(TorchNpuMoonEPBackend(torch, dimensions()), fault_stage)
    with pytest.raises(CorrectnessError) as captured:
        CorrectnessRunner(
            torch, reference, candidate, expert_ops=CpuTorchNpu()
        ).run_differential(case)
    assert captured.value.artifact["stage"] == fault_stage
    assert candidate.calls[-1] == fault_stage
    assert len(candidate.calls) == BACKEND_STAGES.index(fault_stage) + 1
    if fault_stage in ("dispatch", "combine", "reduce_grad"):
        assert "max_ulp_error" in captured.value.artifact


def test_expert_forward_fault_stops_before_combine() -> None:
    case = make_correctness_case(torch, dimensions())
    reference = TorchNpuMoonEPBackend(torch, dimensions())
    candidate = FaultBackend(TorchNpuMoonEPBackend(torch, dimensions()), "none")
    with pytest.raises(CorrectnessError) as captured:
        CorrectnessRunner(
            torch,
            reference,
            candidate,
            expert_ops=CpuTorchNpu(fault_gmm_call=4),
        ).run_differential(case)
    assert captured.value.artifact["stage"] == "expert_forward"
    assert candidate.calls == ["planning", "dispatch", "prefetch_weight"]


def test_differential_mode_requires_candidate() -> None:
    case = make_correctness_case(torch, dimensions())
    runner = CorrectnessRunner(
        torch,
        TorchNpuMoonEPBackend(torch, dimensions()),
        expert_ops=CpuTorchNpu(),
    )
    with pytest.raises(BackendUnavailableError, match="adapter is intentionally deferred"):
        runner.run_differential(case)


def test_plan_semantics_reject_negative_cu_seqlens() -> None:
    case = make_correctness_case(torch, dimensions())
    backend = TorchNpuMoonEPBackend(torch, dimensions())
    normalized_plan = backend.planning(
        case.topk_experts, case.tokens_per_expert
    ).plan
    normalized_plan.cu_seqlens[0] = -1
    runner = CorrectnessRunner(torch, backend, expert_ops=CpuTorchNpu())

    with pytest.raises(ContractError, match="must be non-negative"):
        runner._validate_plan_semantics(normalized_plan)
