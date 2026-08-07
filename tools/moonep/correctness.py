from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    BackendUnavailableError,
    CanonicalMoonEPCase,
    CombineResult,
    ContractError,
    CORRECTNESS_STAGES,
    DispatchResult,
    ExpertForwardResult,
    MoonEPBackend,
    MoonEPPlan,
    PlanningResult,
    PrefetchResult,
    ProjectionTensors,
    ReduceGradResult,
    validate_plan,
    validate_projections,
    validate_tensor,
)
from .expert_forward import run_expert_forward
from .report import write_json
from .tensor_dump import PreviewSink, StageTensorDumper


STAGES = CORRECTNESS_STAGES

BF16_RTOL = 1.6e-2
BF16_ATOL = 1.6e-2


class CorrectnessError(AssertionError):
    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        message = (
            f"{artifact['stage']} correctness failed for {artifact.get('tensor', 'stage')}: "
            f"{artifact['message']}"
        )
        super().__init__(message)


@dataclass(slots=True)
class CorrectnessReport:
    case_id: str
    mode: str
    reference_backend: str
    candidate_backend: str | None
    stages: list[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return len(self.stages) == len(STAGES) and all(
            bool(stage["passed"]) for stage in self.stages
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "mode": self.mode,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "passed": self.passed,
            "stages": self.stages,
        }


def _first_mismatch(torch_module, actual, expected):
    mismatch = actual != expected
    if not bool(mismatch.any().item()):
        return None
    index = tuple(int(item) for item in mismatch.nonzero()[0].tolist())
    return index, actual[index].item(), expected[index].item()


def _max_ulp_error(torch_module, actual, expected) -> int | None:
    dtype_bits = {
        torch_module.bfloat16: (torch_module.int16, 16),
        torch_module.float16: (torch_module.int16, 16),
        torch_module.float32: (torch_module.int32, 32),
    }
    if actual.dtype not in dtype_bits:
        return None
    integer_dtype, width = dtype_bits[actual.dtype]
    mask = (1 << width) - 1
    sign = 1 << (width - 1)

    def ordered(value):
        bits = value.detach().cpu().contiguous().view(integer_dtype).to(torch_module.int64)
        bits = bits.bitwise_and(mask)
        return torch_module.where(
            bits.bitwise_and(sign) != 0,
            mask - bits,
            bits + sign,
        )

    distance = (ordered(actual) - ordered(expected)).abs()
    return int(distance.max().item()) if distance.numel() else 0


def require_tensor_equal(
    torch_module,
    stage: str,
    name: str,
    actual,
    expected,
    *,
    valid_prefix: int | None = None,
) -> None:
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        raise CorrectnessError(
            {
                "stage": stage,
                "tensor": name,
                "message": (
                    f"contract mismatch actual={tuple(actual.shape)}/{actual.dtype}, "
                    f"expected={tuple(expected.shape)}/{expected.dtype}"
                ),
            }
        )
    compared_actual = actual if valid_prefix is None else actual[:valid_prefix]
    compared_expected = expected if valid_prefix is None else expected[:valid_prefix]
    mismatch = _first_mismatch(torch_module, compared_actual, compared_expected)
    if mismatch is None:
        return
    index, actual_value, expected_value = mismatch
    if valid_prefix is not None:
        index = (index[0],) + index[1:]
    left = compared_actual.float()
    right = compared_expected.float()
    abs_error = (left - right).abs()
    denominator = right.abs().clamp_min(1.0e-30)
    artifact = {
            "stage": stage,
            "tensor": name,
            "message": "tensor values differ",
            "first_mismatch_index": list(index),
            "actual": actual_value,
            "expected": expected_value,
            "max_absolute_error": float(abs_error.max().item()),
            "max_relative_error": float((abs_error / denominator).max().item()),
            "valid_prefix": valid_prefix,
        }
    max_ulp = _max_ulp_error(torch_module, compared_actual, compared_expected)
    if max_ulp is not None:
        artifact["max_ulp_error"] = max_ulp
    raise CorrectnessError(artifact)


def require_tensor_close(
    torch_module,
    stage: str,
    name: str,
    actual,
    expected,
    *,
    rtol: float = BF16_RTOL,
    atol: float = BF16_ATOL,
) -> None:
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        raise CorrectnessError(
            {
                "stage": stage,
                "tensor": name,
                "message": (
                    f"contract mismatch actual={tuple(actual.shape)}/{actual.dtype}, "
                    f"expected={tuple(expected.shape)}/{expected.dtype}"
                ),
            }
        )
    left = actual.float()
    right = expected.float()
    close = torch_module.isclose(left, right, rtol=rtol, atol=atol)
    if bool(close.all().item()):
        return
    index = tuple(int(item) for item in (~close).nonzero()[0].tolist())
    abs_error = (left - right).abs()
    denominator = right.abs().clamp_min(1.0e-30)
    artifact = {
        "stage": stage,
        "tensor": name,
        "message": "tensor values differ beyond tolerance",
        "first_mismatch_index": list(index),
        "actual": actual[index].item(),
        "expected": expected[index].item(),
        "max_absolute_error": float(abs_error.max().item()),
        "max_relative_error": float((abs_error / denominator).max().item()),
        "rtol": rtol,
        "atol": atol,
    }
    max_ulp = _max_ulp_error(torch_module, actual, expected)
    if max_ulp is not None:
        artifact["max_ulp_error"] = max_ulp
    raise CorrectnessError(artifact)


def _plan_core_tensors(plan: MoonEPPlan):
    return (
        ("dst", plan.dst),
        ("cu_seqlens", plan.cu_seqlens),
        ("experts_to_copy", plan.experts_to_copy),
        ("zero_fill_ranges", plan.zero_fill_ranges),
        ("remote_stats", plan.remote_stats),
    )


def _dedup_semantics(plan: MoonEPPlan) -> set[tuple[int, tuple[int, ...]]]:
    if plan.dedup is None:
        raise ContractError("dedup metadata is required")
    group_count = int(plan.dedup.counts[0].item())
    loff_count = int(plan.dedup.counts[1].item())
    if group_count < 0 or loff_count < 0:
        raise ContractError("dedup counts must be non-negative")
    result = set()
    for group_index in range(group_count):
        primary, start, count = (
            int(value) for value in plan.dedup.groups[group_index].tolist()
        )
        if start < 0 or count <= 0 or start + count > loff_count:
            raise ContractError("dedup group references an invalid loff range")
        duplicates = tuple(
            sorted(int(value) for value in plan.dedup.loffs[start : start + count].tolist())
        )
        result.add((primary, duplicates))
    return result


def _has_duplicate_destinations(torch_module, plan: MoonEPPlan) -> bool:
    d = plan.dimensions
    if d.topk < 2:
        return False
    encoded = plan.dst.reshape(d.tokens_per_rank, d.topk)
    raw = torch_module.where(encoded < 0, -encoded - 1, encoded)
    destinations = torch_module.div(raw, d.nvsh, rounding_mode="floor")
    ordered = torch_module.sort(destinations, dim=1).values
    return bool((ordered[:, 1:] == ordered[:, :-1]).any().item())


def _assert_unchanged(torch_module, stage: str, name: str, value, snapshot) -> None:
    require_tensor_equal(torch_module, stage, f"input.{name}", value, snapshot)


class CorrectnessRunner:
    def __init__(
        self,
        torch_module,
        reference_backend: MoonEPBackend,
        candidate_backend: MoonEPBackend | None = None,
        *,
        artifact_dir: str | Path | None = None,
        tensor_dump_dir: str | Path | None = None,
        preview_elements: int = 8,
        preview_sink: PreviewSink | None = None,
        expert_ops=None,
    ) -> None:
        self.torch = torch_module
        self.reference = reference_backend
        self.candidate = candidate_backend
        self.artifact_dir = None if artifact_dir is None else Path(artifact_dir)
        self.tensor_dumper = (
            None
            if tensor_dump_dir is None
            else StageTensorDumper(
                torch_module,
                tensor_dump_dir,
                preview_elements=preview_elements,
                preview_sink=preview_sink,
            )
        )
        self.expert_ops = expert_ops
        if candidate_backend is not None and (
            candidate_backend.dimensions != reference_backend.dimensions
        ):
            raise ContractError("reference and candidate dimensions differ")

    def run_reference(self, case: CanonicalMoonEPCase) -> CorrectnessReport:
        return self._run(case, differential=False)

    def run_differential(self, case: CanonicalMoonEPCase) -> CorrectnessReport:
        if self.candidate is None:
            raise BackendUnavailableError(
                "correctness mode requires a conforming candidate backend; "
                "the TileXR adapter is intentionally deferred"
            )
        return self._run(case, differential=True)

    def _run(self, case: CanonicalMoonEPCase, *, differential: bool) -> CorrectnessReport:
        if case.dimensions != self.reference.dimensions:
            raise ContractError("case dimensions do not match the reference backend")
        backends = [self.reference]
        if differential:
            assert self.candidate is not None
            backends.append(self.candidate)
        states: dict[int, dict[str, Any]] = {id(backend): {} for backend in backends}
        reports: list[dict[str, Any]] = []
        for stage in STAGES:
            try:
                self._run_stage(stage, case, backends, states, differential=differential)
                self._rank_gate(True, stage, case.hidden.device)
                artifact = {"stage": stage, "passed": True}
                if stage == "dispatch":
                    dispatch = states[id(self.reference)]["dispatch"]
                    valid_prefix = int(dispatch.plan.cu_seqlens[-1].item())
                    artifact["defined_prefix"] = valid_prefix
                    artifact["undefined_tail"] = [
                        valid_prefix,
                        dispatch.plan.dimensions.nvsh,
                    ]
                reports.append(artifact)
                self._write_artifact(case.case_id, artifact)
            except Exception as exc:
                try:
                    self._rank_gate(False, stage, case.hidden.device)
                except Exception:
                    pass
                artifact = (
                    exc.artifact
                    if isinstance(exc, CorrectnessError)
                    else {
                        "stage": stage,
                        "passed": False,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                artifact["passed"] = False
                self._write_artifact(case.case_id, artifact)
                if isinstance(exc, CorrectnessError):
                    raise
                raise CorrectnessError(artifact) from exc
        return CorrectnessReport(
            case_id=case.case_id,
            mode="correctness" if differential else "reference",
            reference_backend=self.reference.name,
            candidate_backend=None if self.candidate is None else self.candidate.name,
            stages=reports,
        )

    def _run_stage(self, stage, case, backends, states, *, differential):
        if stage == "planning":
            self._planning(case, backends, states)
        elif stage == "dispatch":
            self._dispatch(case, backends, states)
        elif stage == "prefetch_weight":
            self._prefetch(case, backends, states)
        elif stage == "expert_forward":
            self._expert_forward(case, backends, states)
        elif stage == "combine":
            self._combine(case, backends, states)
        elif stage == "reduce_grad":
            self._reduce(case, backends, states)
        else:
            raise AssertionError(stage)
        if differential:
            self._compare_stage(stage, backends, states)

    def _planning(self, case, backends, states):
        torch = self.torch
        d = case.dimensions
        expected_tpe = torch.bincount(
            case.topk_experts.reshape(-1).to(torch.int64), minlength=d.expert_count
        ).to(torch.int32)
        require_tensor_equal(
            torch, "planning", "input.tokens_per_expert", case.tokens_per_expert, expected_tpe
        )
        for backend in backends:
            topk = case.topk_experts.clone()
            tpe = case.tokens_per_expert.clone()
            topk_before, tpe_before = topk.clone(), tpe.clone()
            self._dump(
                "planning",
                backend,
                "input",
                {"topk_experts": topk, "tokens_per_expert": tpe},
            )
            result = backend.planning(topk, tpe)
            backend.synchronize()
            if not isinstance(result, PlanningResult):
                raise ContractError("planning must return PlanningResult")
            validate_plan(result.plan, torch, require_dedup=False)
            self._validate_plan_semantics(result.plan)
            _assert_unchanged(torch, "planning", "topk_experts", topk, topk_before)
            _assert_unchanged(torch, "planning", "tokens_per_expert", tpe, tpe_before)
            self._dump("planning", backend, "output", result)
            states[id(backend)]["plan"] = result.plan

    def _dispatch(self, case, backends, states):
        torch = self.torch
        d = case.dimensions
        if len(backends) > 1 and not getattr(
            backends[1], "supports_duplicate_destinations", True
        ):
            reference_plan = states[id(backends[0])]["plan"]
            if _has_duplicate_destinations(torch, reference_plan):
                raise BackendUnavailableError(
                    f"{backends[1].name} current URMA Dispatch does not support "
                    "duplicate destination ranks for one token"
                )
        for backend in backends:
            plan = states[id(backend)]["plan"]
            core_before = [(name, tensor.clone()) for name, tensor in _plan_core_tensors(plan)]
            hidden = case.hidden.clone()
            weights = case.route_weights.clone()
            hidden_before, weights_before = hidden.clone(), weights.clone()
            self._dump(
                "dispatch",
                backend,
                "input",
                {"plan": plan, "hidden": hidden, "route_weights": weights},
            )
            result = backend.dispatch(plan, hidden, weights)
            backend.synchronize()
            if not isinstance(result, DispatchResult):
                raise ContractError("dispatch must return DispatchResult")
            validate_plan(result.plan, torch, require_dedup=True)
            validate_tensor(
                result.hidden,
                "dispatch.hidden",
                shape=(d.nvsh, d.hidden_size),
                dtype=torch.bfloat16,
            )
            validate_tensor(
                result.route_weights,
                "dispatch.route_weights",
                shape=(d.nvsh,),
                dtype=torch.float32,
            )
            _assert_unchanged(torch, "dispatch", "hidden", hidden, hidden_before)
            _assert_unchanged(torch, "dispatch", "route_weights", weights, weights_before)
            for name, before in core_before:
                _assert_unchanged(
                    torch, "dispatch", f"plan.{name}", getattr(plan, name), before
                )
            self._validate_dispatch_padding(result)
            _dedup_semantics(result.plan)
            self._dump("dispatch", backend, "output", result)
            states[id(backend)]["dispatch"] = result

    def _prefetch(self, case, backends, states):
        torch = self.torch
        d = case.dimensions
        for backend in backends:
            projections = case.projections.clone()
            before = projections.clone()
            plan = states[id(backend)]["dispatch"].plan
            self._dump(
                "prefetch_weight",
                backend,
                "input",
                {"plan": plan, "projections": projections},
            )
            result = backend.prefetch_weight(plan, projections)
            backend.synchronize()
            if not isinstance(result, PrefetchResult):
                raise ContractError("prefetch_weight must return PrefetchResult")
            validate_projections(result.projections, d, torch, dtype=torch.bfloat16)
            for name, tensor in result.projections.items():
                original = getattr(before, name)
                require_tensor_equal(
                    torch,
                    "prefetch_weight",
                    f"input.{name}.source_rows",
                    tensor[: d.expert_count],
                    original[: d.expert_count],
                )
                for slot in range(d.prefetch_slots):
                    if int(plan.experts_to_copy[d.rank, slot].item()) < 0:
                        require_tensor_equal(
                            torch,
                            "prefetch_weight",
                            f"input.{name}.unused_slot_{slot}",
                            tensor[d.expert_count + slot],
                            original[d.expert_count + slot],
                        )
            self._dump("prefetch_weight", backend, "output", result)
            states[id(backend)]["prefetch"] = result

    def _expert_forward(self, case, backends, states):
        torch = self.torch
        d = case.dimensions
        for backend in backends:
            dispatch = states[id(backend)]["dispatch"]
            projections = states[id(backend)]["prefetch"].projections
            hidden_before = dispatch.hidden.clone()
            weights_before = dispatch.route_weights.clone()
            projections_before = projections.clone()
            group_list_before = dispatch.plan.cu_seqlens.clone()
            self._dump(
                "expert_forward",
                backend,
                "input",
                {
                    "hidden": dispatch.hidden,
                    "cu_seqlens": dispatch.plan.cu_seqlens,
                    "projections": projections,
                    "route_weights": dispatch.route_weights,
                },
            )
            result = run_expert_forward(
                torch,
                dispatch.hidden,
                dispatch.plan.cu_seqlens,
                projections,
                dispatch.route_weights,
                torch_npu_module=self.expert_ops,
            )
            backend.synchronize()
            if not isinstance(result, ExpertForwardResult):
                raise ContractError("expert_forward must return ExpertForwardResult")
            validate_tensor(
                result.hidden,
                "expert_forward.hidden",
                shape=(d.nvsh, d.hidden_size),
                dtype=torch.bfloat16,
            )
            _assert_unchanged(
                torch, "expert_forward", "hidden", dispatch.hidden, hidden_before
            )
            _assert_unchanged(
                torch,
                "expert_forward",
                "route_weights",
                dispatch.route_weights,
                weights_before,
            )
            _assert_unchanged(
                torch,
                "expert_forward",
                "plan.cu_seqlens",
                dispatch.plan.cu_seqlens,
                group_list_before,
            )
            for name, before in projections_before.items():
                _assert_unchanged(
                    torch,
                    "expert_forward",
                    f"projections.{name}",
                    getattr(projections, name),
                    before,
                )
            self._dump("expert_forward", backend, "output", result)
            states[id(backend)]["expert_forward"] = result

    def _combine(self, case, backends, states):
        torch = self.torch
        d = case.dimensions
        for backend in backends:
            hidden = states[id(backend)]["expert_forward"].hidden.clone()
            weights = states[id(backend)]["dispatch"].route_weights.clone()
            hidden_before, weights_before = hidden.clone(), weights.clone()
            plan = states[id(backend)]["dispatch"].plan
            self._dump(
                "combine",
                backend,
                "input",
                {"plan": plan, "expert_output": hidden, "route_weights": weights},
            )
            result = backend.combine(
                plan, hidden, weights
            )
            backend.synchronize()
            if not isinstance(result, CombineResult):
                raise ContractError("combine must return CombineResult")
            validate_tensor(
                result.hidden,
                "combine.hidden",
                shape=(d.tokens_per_rank, d.hidden_size),
                dtype=torch.bfloat16,
            )
            validate_tensor(
                result.route_weights,
                "combine.route_weights",
                shape=(d.tokens_per_rank, d.topk),
                dtype=torch.float32,
            )
            _assert_unchanged(torch, "combine", "expert_output", hidden, hidden_before)
            _assert_unchanged(
                torch, "combine", "route_weights_nvs", weights, weights_before
            )
            self._dump("combine", backend, "output", result)
            states[id(backend)]["combine"] = result

    def _reduce(self, case, backends, states):
        torch = self.torch
        d = case.dimensions
        begin = d.rank * d.experts_per_rank
        end = begin + d.experts_per_rank
        for backend in backends:
            full = case.full_grads.clone()
            buffers = case.reduce_buffers.clone()
            full_before, buffers_before = full.clone(), buffers.clone()
            plan = states[id(backend)]["dispatch"].plan
            self._dump(
                "reduce_grad",
                backend,
                "input",
                {"plan": plan, "full_grads": full, "reduce_buffers": buffers},
            )
            result = backend.reduce_grad(plan, full, buffers)
            backend.synchronize()
            if not isinstance(result, ReduceGradResult):
                raise ContractError("reduce_grad must return ReduceGradResult")
            validate_projections(result.full_grads, d, torch, dtype=torch.float32)
            validate_projections(
                result.reduce_buffers,
                d,
                torch,
                dtype=torch.float32,
                reduce_buffers=True,
            )
            for name, tensor in result.full_grads.items():
                original = getattr(full_before, name)
                require_tensor_equal(
                    torch, "reduce_grad", f"input.{name}.nonowner_before", tensor[:begin], original[:begin]
                )
                require_tensor_equal(
                    torch, "reduce_grad", f"input.{name}.nonowner_after", tensor[end:], original[end:]
                )
            for name, tensor in result.reduce_buffers.items():
                original = getattr(buffers_before, name)
                for source in range(d.world_size):
                    for slot in range(d.prefetch_slots):
                        expert = int(plan.experts_to_copy[source, slot].item())
                        if source == d.rank and expert >= 0:
                            if bool((tensor[source, slot] != 0).any().item()):
                                raise CorrectnessError(
                                    {
                                        "stage": "reduce_grad",
                                        "tensor": f"{name}.live_slot_{source}_{slot}",
                                        "message": "consumed reduction slot was not cleared",
                                    }
                                )
                        else:
                            preservation = (
                                "unused_slot" if expert < 0 else "nonlocal_slot"
                            )
                            require_tensor_equal(
                                torch,
                                "reduce_grad",
                                f"input.{name}.{preservation}_{source}_{slot}",
                                tensor[source, slot],
                                original[source, slot],
                            )
            self._dump("reduce_grad", backend, "output", result)
            states[id(backend)]["reduce"] = result

    def _compare_stage(self, stage, backends, states):
        torch = self.torch
        reference, candidate = backends
        left = states[id(reference)]
        right = states[id(candidate)]
        if stage == "planning":
            for name, tensor in _plan_core_tensors(left["plan"]):
                require_tensor_equal(
                    torch, stage, name, getattr(right["plan"], name), tensor
                )
        elif stage == "dispatch":
            for name, tensor in _plan_core_tensors(left["dispatch"].plan):
                require_tensor_equal(
                    torch,
                    stage,
                    f"plan.{name}",
                    getattr(right["dispatch"].plan, name),
                    tensor,
                )
            prefix = int(left["dispatch"].plan.cu_seqlens[-1].item())
            require_tensor_equal(
                torch,
                stage,
                "hidden",
                right["dispatch"].hidden,
                left["dispatch"].hidden,
                valid_prefix=prefix,
            )
            require_tensor_equal(
                torch,
                stage,
                "route_weights",
                right["dispatch"].route_weights,
                left["dispatch"].route_weights,
                valid_prefix=prefix,
            )
            if _dedup_semantics(left["dispatch"].plan) != _dedup_semantics(
                right["dispatch"].plan
            ):
                raise CorrectnessError(
                    {
                        "stage": stage,
                        "tensor": "plan.dedup",
                        "message": "semantic dedup groups differ",
                    }
                )
        elif stage == "prefetch_weight":
            self._compare_projections(
                stage, right["prefetch"].projections, left["prefetch"].projections
            )
        elif stage == "expert_forward":
            require_tensor_close(
                torch,
                stage,
                "hidden",
                right["expert_forward"].hidden,
                left["expert_forward"].hidden,
            )
        elif stage == "combine":
            require_tensor_close(
                torch, stage, "hidden", right["combine"].hidden, left["combine"].hidden
            )
            require_tensor_equal(
                torch,
                stage,
                "route_weights",
                right["combine"].route_weights,
                left["combine"].route_weights,
            )
        elif stage == "reduce_grad":
            self._compare_projections(
                stage, right["reduce"].full_grads, left["reduce"].full_grads, prefix="full_grads"
            )
            self._compare_projections(
                stage,
                right["reduce"].reduce_buffers,
                left["reduce"].reduce_buffers,
                prefix="reduce_buffers",
            )

    def _compare_projections(self, stage, actual, expected, *, prefix="projections"):
        for name, tensor in expected.items():
            require_tensor_equal(
                self.torch, stage, f"{prefix}.{name}", getattr(actual, name), tensor
            )

    def _validate_plan_semantics(self, plan):
        d = plan.dimensions
        if bool((plan.cu_seqlens < 0).any().item()):
            raise ContractError("cu_seqlens must be non-negative")
        if bool((plan.cu_seqlens[1:] < plan.cu_seqlens[:-1]).any().item()):
            raise ContractError("cu_seqlens must be monotonic")
        if int(plan.cu_seqlens[-1].item()) > d.nvsh:
            raise ContractError("cu_seqlens exceeds NvS")
        raw = self.torch.where(plan.dst < 0, -plan.dst - 1, plan.dst)
        if bool(((raw < 0) | (raw >= d.world_size * d.nvsh)).any().item()):
            raise ContractError("dst contains an out-of-range destination")
        copies = plan.experts_to_copy
        if bool(((copies < -1) | (copies >= d.expert_count)).any().item()):
            raise ContractError("experts_to_copy contains an invalid expert")

    def _validate_dispatch_padding(self, result):
        for start, count in result.plan.zero_fill_ranges.tolist():
            if count <= 0:
                continue
            if bool((result.hidden[start : start + count] != 0).any().item()):
                raise ContractError("dispatch hidden padding is not zero")
            if result.route_weights is not None and bool(
                (result.route_weights[start : start + count] != 0).any().item()
            ):
                raise ContractError("dispatch route-weight padding is not zero")

    def _rank_gate(self, passed: bool, stage: str, device) -> None:
        collective = getattr(self.reference, "collective", None)
        if collective is None or not hasattr(collective, "all_agree"):
            if self.reference.dimensions.world_size != 1:
                raise BackendUnavailableError(
                    f"{stage} requires rank-wide agreement support"
                )
            return
        if not collective.all_agree(passed, device=device):
            raise CorrectnessError(
                {
                    "stage": stage,
                    "tensor": "rank_agreement",
                    "message": "one or more ranks failed the stage",
                }
            )

    def _write_artifact(self, case_id: str, artifact: dict[str, Any]) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / f"{case_id}.{artifact['stage']}.json"
        write_json(path, artifact)

    def _dump(self, stage: str, backend, direction: str, value: Any) -> None:
        if self.tensor_dumper is None:
            return
        role = "reference" if backend is self.reference else "candidate"
        self.tensor_dumper.dump(stage, role, backend.name, direction, value)
