from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import tools.moonep.benchmark as benchmark
from tools.moonep.benchmark import load_candidate_backend
from tools.moonep.config import BenchmarkCase
from tools.moonep.contracts import (
    BackendUnavailableError,
    MoonEPDimensions,
    ProjectionTensors,
)
from tools.moonep.launcher import _process_command, build_parser as build_launcher_parser
from tools.moonep.report import aggregate_correctness_artifacts, write_json
from tools.moonep.torch_npu_backend import TorchNpuMoonEPBackend


def dimensions() -> MoonEPDimensions:
    return MoonEPDimensions(0, 1, 4, 2, 4, 4, 4, 4, 3)


class IterationTorchNpu:
    def __init__(self) -> None:
        self.gmm_calls = 0

    def npu_grouped_matmul(self, **kwargs):
        self.gmm_calls += 1
        inputs = kwargs["x"][0]
        weights = kwargs["weight"][0]
        chunks = []
        begin = 0
        for group, end in enumerate(kwargs["group_list"].tolist()):
            chunks.append(inputs[begin:end].float().matmul(weights[group].float()))
            begin = end
        return [torch.cat(chunks).to(kwargs["output_dtype"])]

    @staticmethod
    def npu_swiglu(value):
        gate, up = value.chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(value.dtype)


class IterationBuffer:
    def __init__(self) -> None:
        self.context = SimpleNamespace(
            nv_s=4,
            prefetch_slots=1,
            planner_group_rank=0,
            experts_per_rank=1,
            expert_count=1,
        )
        self.hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.bfloat16)
        self.route_weights = torch.tensor([0.5, 1.0, 0.0, 0.0], dtype=torch.float32)
        self.group_list = torch.tensor([1, 2], dtype=torch.int32)
        self.plan = object()
        self.combine_inputs = []
        self.dispatch_calls = 0

    def dispatch(self, *_args, **_kwargs):
        self.dispatch_calls += 1
        if self.dispatch_calls == 1:
            return self.hidden, self.route_weights, self.group_list, self.plan
        return torch.ones_like(self.hidden), None, self.group_list, self.plan

    @staticmethod
    def prefetch_weight(*_args, **_kwargs):
        return None

    def combine(self, _plan, hidden, route_weights=None):
        self.combine_inputs.append(hidden.clone())
        return hidden, route_weights, None

    @staticmethod
    def reduce_grad(*_args, **_kwargs):
        return None

    @staticmethod
    def synchronize():
        return None


def test_timed_iteration_executes_torch_npu_expert_forward() -> None:
    weights = ProjectionTensors(
        gate=torch.ones((2, 2, 1), dtype=torch.bfloat16),
        up=torch.full((2, 2, 1), 2, dtype=torch.bfloat16),
        down=torch.ones((2, 1, 2), dtype=torch.bfloat16),
    )
    gradients = SimpleNamespace(
        gate=torch.ones_like(weights.gate, dtype=torch.float32),
        up=torch.ones_like(weights.up, dtype=torch.float32),
        down=torch.ones_like(weights.down, dtype=torch.float32),
        gate_reduce=torch.zeros((1,), dtype=torch.float32),
        up_reduce=torch.zeros((1,), dtype=torch.float32),
        down_reduce=torch.zeros((1,), dtype=torch.float32),
    )
    inputs = {
        "hidden": torch.ones((2, 2), dtype=torch.bfloat16),
        "route_weights": torch.ones((2, 1), dtype=torch.float32),
        "topk_experts": torch.zeros((2, 1), dtype=torch.int32),
        "tokens_per_expert": torch.ones((2,), dtype=torch.int32),
        "projections": weights,
        "grad_output": torch.ones((2, 2), dtype=torch.bfloat16),
        "gradients": gradients,
    }
    buffer = IterationBuffer()
    ops = IterationTorchNpu()

    benchmark.execute_iteration(
        buffer,
        inputs,
        torch_module=torch,
        torch_npu_module=ops,
    )

    assert ops.gmm_calls == 2
    assert torch.count_nonzero(buffer.combine_inputs[0][2:]).item() == 0


def test_case_aliases_add_reference_dimensions_without_changing_old_defaults() -> None:
    old = BenchmarkCase("old", 4, 2, 4, 8)
    assert old.intermediate_size is None
    assert old.prefetch_slots is None
    assert old.token_padding == 1
    case = BenchmarkCase.from_mapping(
        {
            "id": "new",
            "S": 4,
            "K": 2,
            "E": 4,
            "H": 8,
            "Hf": 16,
            "B": 2,
            "P": 4,
            "routing": "skewed",
        }
    )
    assert case.intermediate_size == 16
    assert case.prefetch_slots == 2
    assert case.token_padding == 4
    assert case.routing_pattern == "skewed"


@pytest.mark.parametrize(("world_size", "expected_slots"), [(1, 8), (4, 2)])
def test_correctness_dimensions_derive_required_prefetch_slots(
    world_size: int, expected_slots: int
) -> None:
    case = BenchmarkCase.from_mapping(
        {
            "id": "planning-small",
            "S": 8,
            "K": 2,
            "E": 8,
            "H": 8,
            "Hf": 4,
            "B": 2,
            "P": 4,
            "routing": "balanced",
        }
    )

    resolved = benchmark._correctness_dimensions(case, rank=0, world_size=world_size)

    assert resolved.prefetch_slots == expected_slots
    assert resolved.prefetch_slots == resolved.experts_per_rank


def test_launcher_forwards_mode_candidate_and_reference_overrides(tmp_path) -> None:
    args = build_launcher_parser().parse_args(
        [
            "--cases",
            str(tmp_path / "cases.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--mode",
            "correctness",
            "--candidate-backend",
            "candidate:create",
            "--intermediate-size",
            "16",
            "--prefetch-slots",
            "2",
            "--token-padding",
            "4",
            "--routing-pattern",
            "skewed",
            "--dump-stage-tensors",
            "--tensor-preview-elements",
            "5",
        ]
    )
    command = _process_command(args)
    assert command[command.index("--mode") + 1] == "correctness"
    assert command[command.index("--candidate-backend") + 1] == "candidate:create"
    assert command[command.index("--intermediate-size") + 1] == "16"
    assert command[command.index("--prefetch-slots") + 1] == "2"
    assert command[command.index("--token-padding") + 1] == "4"
    assert "--dump-stage-tensors" in command
    assert command[command.index("--tensor-preview-elements") + 1] == "5"


def test_candidate_factory_is_protocol_checked(monkeypatch) -> None:
    module = ModuleType("test_candidate_backend")

    def create_backend(**kwargs):
        return TorchNpuMoonEPBackend(kwargs["torch_module"], kwargs["dimensions"])

    module.create_backend = create_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = load_candidate_backend(
        "test_candidate_backend:create_backend",
        torch_module=torch,
        dimensions=dimensions(),
        case=BenchmarkCase("case", 4, 2, 4, 4),
        args=SimpleNamespace(),
    )
    assert backend.dimensions == dimensions()


class BackendWithoutClose:
    name = "missing_close"

    def __init__(self):
        self.dimensions = dimensions()

    def planning(self, *args):
        return None

    def dispatch(self, *args):
        return None

    def prefetch_weight(self, *args):
        return None

    def combine(self, *args):
        return None

    def reduce_grad(self, *args):
        return None

    def synchronize(self):
        return None


def test_candidate_factory_requires_close(monkeypatch) -> None:
    module = ModuleType("test_candidate_without_close")
    module.create_backend = lambda **_kwargs: BackendWithoutClose()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(BackendUnavailableError, match="did not return MoonEPBackend"):
        load_candidate_backend(
            "test_candidate_without_close:create_backend",
            torch_module=torch,
            dimensions=dimensions(),
            case=BenchmarkCase("case", 4, 2, 4, 4),
            args=SimpleNamespace(),
        )


class RecordingBackend:
    def __init__(self, name, events, *, close_error=None):
        self.name = name
        self.dimensions = dimensions()
        self.events = events
        self.close_error = close_error

    def planning(self, *args):
        return None

    def dispatch(self, *args):
        return None

    def prefetch_weight(self, *args):
        return None

    def combine(self, *args):
        return None

    def reduce_grad(self, *args):
        return None

    def synchronize(self):
        return None

    def close(self):
        self.events.append(f"close:{self.name}")
        if self.close_error is not None:
            raise self.close_error


def _correctness_args(tmp_path):
    return SimpleNamespace(
        mode="correctness",
        candidate_backend="candidate:create",
        output_dir=str(tmp_path),
        install_prefix=None,
        wait_iterations=1234,
    )


def _patch_correctness_case(monkeypatch, reference, candidate, runner):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(benchmark, "make_correctness_case", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        benchmark, "TorchNpuMoonEPBackend", lambda *_args, **_kwargs: reference
    )
    monkeypatch.setattr(
        benchmark, "load_candidate_backend", lambda *_args, **_kwargs: candidate
    )
    monkeypatch.setattr(
        benchmark,
        "CorrectnessRunner",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_reference=runner,
            run_differential=runner,
        ),
    )


def test_correctness_case_closes_candidate_then_reference(monkeypatch, tmp_path) -> None:
    events = []
    reference = RecordingBackend("reference", events)
    candidate = RecordingBackend("candidate", events)
    report = SimpleNamespace(as_dict=lambda: {"stages": []})
    _patch_correctness_case(
        monkeypatch,
        reference,
        candidate,
        lambda *_args, **_kwargs: report,
    )

    benchmark.run_correctness_case(
        torch,
        BenchmarkCase("case", 4, 2, 4, 4),
        _correctness_args(tmp_path),
        tmp_path,
    )

    assert events == ["close:candidate", "close:reference"]
    result = json.loads((tmp_path / "case" / "rank_0" / "result.json").read_text())
    assert result["status"] == "passed"


@pytest.mark.parametrize(
    ("candidate_spec", "expected_spec"),
    (
        (None, "tools.moonep.tilexr_backend:create_backend"),
        ("external.backend:create", "external.backend:create"),
    ),
)
def test_correctness_case_resolves_builtin_candidate_with_explicit_override(
    monkeypatch, tmp_path, candidate_spec, expected_spec
) -> None:
    events = []
    reference = RecordingBackend("reference", events)
    candidate = RecordingBackend("candidate", events)
    report = SimpleNamespace(as_dict=lambda: {"stages": []})
    _patch_correctness_case(
        monkeypatch,
        reference,
        candidate,
        lambda *_args, **_kwargs: report,
    )
    loaded = []

    def load(spec, **_kwargs):
        loaded.append(spec)
        return candidate

    monkeypatch.setattr(benchmark, "load_candidate_backend", load)
    args = _correctness_args(tmp_path)
    args.candidate_backend = candidate_spec
    benchmark.run_correctness_case(
        torch,
        BenchmarkCase("case", 4, 2, 4, 4),
        args,
        tmp_path,
    )

    assert loaded == [expected_spec]


def test_correctness_case_preserves_stage_error_and_records_cleanup(
    monkeypatch, tmp_path
) -> None:
    events = []
    reference = RecordingBackend("reference", events)
    candidate = RecordingBackend(
        "candidate", events, close_error=RuntimeError("candidate close failed")
    )

    def fail_stage(*_args, **_kwargs):
        raise ValueError("stage failed")

    _patch_correctness_case(monkeypatch, reference, candidate, fail_stage)
    with pytest.raises(ValueError, match="stage failed"):
        benchmark.run_correctness_case(
            torch,
            BenchmarkCase("case", 4, 2, 4, 4),
            _correctness_args(tmp_path),
            tmp_path,
        )

    assert events == ["close:candidate", "close:reference"]
    result = json.loads((tmp_path / "case" / "rank_0" / "result.json").read_text())
    assert result["failure_reason"] == "ValueError: stage failed"
    assert result["cleanup_errors"] == [
        "candidate: RuntimeError: candidate close failed"
    ]


def test_correctness_case_close_failure_changes_pass_to_failure(
    monkeypatch, tmp_path
) -> None:
    events = []
    reference = RecordingBackend("reference", events)
    candidate = RecordingBackend(
        "candidate", events, close_error=RuntimeError("candidate close failed")
    )
    report = SimpleNamespace(as_dict=lambda: {"stages": []})
    _patch_correctness_case(
        monkeypatch,
        reference,
        candidate,
        lambda *_args, **_kwargs: report,
    )

    with pytest.raises(RuntimeError, match="candidate close failed"):
        benchmark.run_correctness_case(
            torch,
            BenchmarkCase("case", 4, 2, 4, 4),
            _correctness_args(tmp_path),
            tmp_path,
        )

    result = json.loads((tmp_path / "case" / "rank_0" / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["cleanup_errors"] == [
        "candidate: RuntimeError: candidate close failed"
    ]


def test_correctness_factory_failure_writes_result_and_closes_reference(
    monkeypatch, tmp_path
) -> None:
    events = []
    reference = RecordingBackend("reference", events)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(benchmark, "make_correctness_case", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        benchmark, "TorchNpuMoonEPBackend", lambda *_args, **_kwargs: reference
    )

    def fail_factory(*_args, **_kwargs):
        raise BackendUnavailableError("capability missing")

    monkeypatch.setattr(benchmark, "load_candidate_backend", fail_factory)
    with pytest.raises(BackendUnavailableError, match="capability missing"):
        benchmark.run_correctness_case(
            torch,
            BenchmarkCase("case", 4, 2, 4, 4),
            _correctness_args(tmp_path),
            tmp_path,
        )

    assert events == ["close:reference"]
    result = json.loads((tmp_path / "case" / "rank_0" / "result.json").read_text())
    assert result["failure_reason"] == "BackendUnavailableError: capability missing"


def test_correctness_aggregation_is_explicitly_untimed(tmp_path) -> None:
    case_dir = tmp_path / "case"
    stages = [
        {"stage": name, "passed": True}
        for name in (
            "planning",
            "dispatch",
            "prefetch_weight",
            "expert_forward",
            "combine",
            "reduce_grad",
        )
    ]
    for rank in range(2):
        write_json(
            case_dir / f"rank_{rank}" / "result.json",
            {
                "status": "passed",
                "mode": "reference",
                "rank": rank,
                "case": {"case_id": "case"},
                "validation": {
                    "reference_backend": "torch_npu_reference_v1",
                    "candidate_backend": None,
                    "stages": stages,
                },
            },
        )
        for stage in stages:
            write_json(
                case_dir
                / f"rank_{rank}"
                / "stages"
                / f"case.{stage['stage']}.json",
                stage,
            )
    summary = aggregate_correctness_artifacts(case_dir, world_size=2)
    assert summary["status"] == "passed"
    assert summary["performance_valid"] is False
    assert [stage["stage"] for stage in summary["stages"]] == [
        stage["stage"] for stage in stages
    ]
    assert json.loads((case_dir / "summary.json").read_text())["performance_valid"] is False


def test_correctness_aggregation_rejects_missing_expert_forward(tmp_path) -> None:
    case_dir = tmp_path / "case"
    stale_stages = [
        {"stage": name, "passed": True}
        for name in ("planning", "dispatch", "prefetch_weight", "combine", "reduce_grad")
    ]
    for rank in range(2):
        write_json(
            case_dir / f"rank_{rank}" / "result.json",
            {
                "status": "passed",
                "mode": "reference",
                "rank": rank,
                "case": {"case_id": "case"},
                "validation": {
                    "reference_backend": "torch_npu_reference_v1",
                    "candidate_backend": None,
                    "stages": stale_stages,
                },
            },
        )

    with pytest.raises(ValueError, match="ordered correctness stages"):
        aggregate_correctness_artifacts(case_dir, world_size=2)


def test_correctness_aggregation_requires_stage_artifacts(tmp_path) -> None:
    case_dir = tmp_path / "case"
    stages = [
        {"stage": name, "passed": True}
        for name in (
            "planning",
            "dispatch",
            "prefetch_weight",
            "expert_forward",
            "combine",
            "reduce_grad",
        )
    ]
    write_json(
        case_dir / "rank_0" / "result.json",
        {
            "status": "passed",
            "mode": "reference",
            "rank": 0,
            "case": {"case_id": "case"},
            "validation": {
                "reference_backend": "torch_npu_reference_v1",
                "candidate_backend": None,
                "stages": stages,
            },
        },
    )

    with pytest.raises(FileNotFoundError, match="missing stage artifact"):
        aggregate_correctness_artifacts(case_dir, world_size=1)
