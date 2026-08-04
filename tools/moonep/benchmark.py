from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

from .config import apply_overrides, build_case_parser, load_cases, select_cases
from .planner_reference import build_reference_plan, deterministic_all_topk
from .rendezvous import (
    completion_barrier_from_env,
    hold_for_managed_abort,
    signal_managed_abort,
)
from .report import write_json, write_jsonl


STAGE_ORDER = (
    "planning",
    "dispatch_forward",
    "prefetch_weight",
    "expert_forward",
    "combine_forward",
    "dispatch_backward",
    "expert_backward",
    "combine_backward",
    "reduce_grad",
)


def _hold_unsafe_teardown(
    *,
    output_root: Path,
    rank_dir: Path,
    result: dict[str, object],
    rank: int,
    reason: str,
    world_size: int,
) -> None:
    try:
        try:
            write_json(rank_dir / "result.json", result)
        except Exception:
            pass
        try:
            signal_managed_abort(output_root, rank, reason)
        except Exception:
            pass
    finally:
        if world_size > 1:
            hold_for_managed_abort()
    raise RuntimeError(reason)


class DeviceEventTimer:
    def __init__(self, torch_module):
        self._torch = torch_module
        self._events: dict[str, tuple[object, object]] = {}
        self._e2e_start = self._event()
        self._e2e_end = self._event()

    def _event(self):
        return self._torch.npu.Event(enable_timing=True)

    def start(self) -> None:
        self._e2e_start.record()

    def call(self, name: str, callback: Callable[[], object]):
        start = self._event()
        end = self._event()
        start.record()
        value = callback()
        end.record()
        self._events[name] = (start, end)
        return value

    def finish(self) -> dict[str, float]:
        self._e2e_end.record()
        self._torch.npu.synchronize()
        timings = {
            name: float(start.elapsed_time(end)) * 1000.0
            for name, (start, end) in self._events.items()
        }
        timings["end_to_end"] = float(
            self._e2e_start.elapsed_time(self._e2e_end)
        ) * 1000.0
        return timings


def _timed_call(timer: DeviceEventTimer | None, name: str, callback):
    return callback() if timer is None else timer.call(name, callback)


def _oversubscribed_planning_barrier(buffer, case_id: str, epoch: int) -> None:
    if os.environ.get("TILEXR_OVERSUBSCRIBED", "0") != "1":
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    buffer.synchronize()
    decision = completion_barrier_from_env(
        rank,
        world_size,
        case_id=f"{case_id}.planning.{epoch}",
        quiesced=True,
        passed=True,
    )
    if not decision.release or decision.abort:
        raise RuntimeError("oversubscribed Planning launch rendezvous failed")


def execute_iteration(buffer, inputs: dict[str, object], timer: DeviceEventTimer | None = None):
    if timer is not None:
        timer.start()
    plan = _timed_call(
        timer,
        "planning",
        lambda: buffer.planning(inputs["topk_experts"], inputs["tokens_per_expert"]),
    )
    dispatched = _timed_call(
        timer,
        "dispatch_forward",
        lambda: buffer.dispatch(inputs["hidden"], plan, inputs["route_weights"]),
    )
    prefetched = _timed_call(
        timer,
        "prefetch_weight",
        lambda: buffer.prefetch_weight(plan, inputs["projections"]),
    )

    def expert_forward():
        scale = prefetched.gate[-buffer.context.experts_per_rank :].mean()
        output = dispatched.hidden * scale.to(dtype=dispatched.hidden.dtype)
        if dispatched.route_weights is not None:
            route_weights = dispatched.route_weights.to(dtype=output.dtype)
            output = output * route_weights.reshape(buffer.context.dispatched_capacity, 1)
        return output

    expert_output = _timed_call(timer, "expert_forward", expert_forward)
    forward = _timed_call(
        timer,
        "combine_forward",
        lambda: buffer.combine(expert_output, plan, dispatched.route_weights),
    )
    dispatched_grad = _timed_call(
        timer,
        "dispatch_backward",
        lambda: buffer.dispatch(inputs["grad_output"], plan),
    )

    def expert_backward():
        scale = prefetched.gate[-buffer.context.experts_per_rank :].mean()
        output = dispatched_grad.hidden * scale.to(dtype=dispatched_grad.hidden.dtype)
        if dispatched.route_weights is not None:
            route_weights = dispatched.route_weights.to(dtype=output.dtype)
            output = output * route_weights.reshape(
                buffer.context.dispatched_capacity, 1
            )
        return output

    grad_expert_hidden = _timed_call(timer, "expert_backward", expert_backward)
    backward = _timed_call(
        timer,
        "combine_backward",
        lambda: buffer.combine(grad_expert_hidden, plan),
    )
    _timed_call(
        timer,
        "reduce_grad",
        lambda: buffer.reduce_grad(plan, inputs["gradients"]),
    )
    timings = None if timer is None else timer.finish()
    buffer.synchronize()
    return plan, forward, backward, timings


def _torch_dtype(torch_module, name: str):
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
    }[name]


def make_inputs(torch_module, case, context):
    from tilexr_moonep import ProjectionBuffers

    device = f"npu:{context.device_index}"
    torch_module.manual_seed(case.seed + context.global_rank)
    if hasattr(torch_module.npu, "manual_seed_all"):
        torch_module.npu.manual_seed_all(case.seed + context.global_rank)
    dtype = _torch_dtype(torch_module, case.dtype)
    route_ids = torch_module.arange(
        case.tokens_per_rank * case.topk,
        dtype=torch_module.int32,
        device=device,
    )
    route_ids = (
        route_ids + context.planner_group_rank * case.topk
    ).remainder(case.expert_count).reshape(case.tokens_per_rank, case.topk)
    tokens_per_expert = torch_module.bincount(
        route_ids.reshape(-1).to(dtype=torch_module.int64),
        minlength=case.expert_count,
    ).to(dtype=torch_module.int32)
    hidden = torch_module.randn(
        (case.tokens_per_rank, case.hidden_size), dtype=dtype, device=device
    )
    route_weights = torch_module.full(
        (case.tokens_per_rank, case.topk),
        1.0 / case.topk,
        dtype=torch_module.float32,
        device=device,
    )
    rows = case.expert_count + context.experts_per_rank
    projections = ProjectionBuffers(
        gate=torch_module.full((rows, case.hidden_size), 0.5, dtype=dtype, device=device),
        up=torch_module.full((rows, case.hidden_size), 0.25, dtype=dtype, device=device),
        down=torch_module.full((rows, case.hidden_size), 0.75, dtype=dtype, device=device),
    )
    gradients = ProjectionBuffers(
        gate=torch_module.ones((rows, case.hidden_size), dtype=torch_module.float32, device=device),
        up=torch_module.ones((rows, case.hidden_size), dtype=torch_module.float32, device=device),
        down=torch_module.ones((rows, case.hidden_size), dtype=torch_module.float32, device=device),
        gate_reduce=torch_module.zeros(
            (rows, case.hidden_size),
            dtype=torch_module.float32,
            device=device,
        ),
        up_reduce=torch_module.zeros(
            (rows, case.hidden_size),
            dtype=torch_module.float32,
            device=device,
        ),
        down_reduce=torch_module.zeros(
            (rows, case.hidden_size),
            dtype=torch_module.float32,
            device=device,
        ),
    )
    return {
        "topk_experts": route_ids,
        "tokens_per_expert": tokens_per_expert,
        "hidden": hidden,
        "route_weights": route_weights,
        "projections": projections,
        "grad_output": torch_module.ones_like(hidden),
        "gradients": gradients,
    }


def _checksum(tensor) -> float:
    value = float(tensor.float().sum().item())
    if not math.isfinite(value):
        raise RuntimeError("non-finite deterministic checksum")
    return value


def _flatten(values) -> list[int]:
    if isinstance(values, (list, tuple)):
        result = []
        for value in values:
            result.extend(_flatten(value))
        return result
    return [int(values)]


def _tensor_values(tensor) -> list[int]:
    return _flatten(tensor.cpu().tolist())


def _require_exact(name: str, actual: list[int], expected) -> None:
    expected_values = [int(value) for value in expected]
    if actual == expected_values:
        return
    mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(actual, expected_values))
            if left != right
        ),
        min(len(actual), len(expected_values)),
    )
    actual_value = actual[mismatch] if mismatch < len(actual) else "<missing>"
    expected_value = (
        expected_values[mismatch] if mismatch < len(expected_values) else "<missing>"
    )
    raise RuntimeError(
        f"Planner {name} mismatch at flat index {mismatch}: "
        f"actual={actual_value}, expected={expected_value}"
    )


def validate_plan(plan, context) -> dict[str, object]:
    status = int(plan.status.item())
    if status != 0:
        raise RuntimeError(f"Planner device status is {status}")
    reference = build_reference_plan(
        rank=context.planner_group_rank,
        rank_size=context.planner_group_size,
        tokens_per_rank=context.tokens_per_rank,
        topk=context.topk,
        expert_count=context.expert_count,
        all_topk=deterministic_all_topk(
            context.planner_group_size,
            context.tokens_per_rank,
            context.topk,
            context.expert_count,
        ),
    )
    _require_exact("dst", _tensor_values(plan.dst), reference.dst)
    _require_exact(
        "cu_seqlens", _tensor_values(plan.cu_seqlens), reference.cu_seqlens
    )
    _require_exact(
        "experts_to_copy",
        _tensor_values(plan.experts_to_copy),
        reference.experts_to_copy,
    )
    _require_exact(
        "remote_stats", _tensor_values(plan.remote_stats), reference.remote_stats
    )
    return {
        "passed": True,
        "mode": "planner_cpu_oracle",
        "checks": {
            "status_zero": True,
            "dst_exact": True,
            "cu_seqlens_exact": True,
            "experts_to_copy_exact": True,
            "remote_stats_exact": True,
        },
    }


def validate_stub_flow(torch_module, inputs, forward, backward, context) -> dict[str, bool]:
    scale = inputs["projections"].gate[-context.experts_per_rank :].mean()
    local_weights = inputs["route_weights"].reshape(-1)[: context.tokens_per_rank]
    local_weights = local_weights.reshape(context.tokens_per_rank, 1)
    expected_forward = inputs["hidden"] * scale.to(dtype=inputs["hidden"].dtype)
    expected_forward = expected_forward * local_weights.to(dtype=expected_forward.dtype)
    expected_backward = inputs["grad_output"] * scale.to(dtype=inputs["grad_output"].dtype)
    expected_backward = expected_backward * local_weights.to(dtype=expected_backward.dtype)
    checks = {
        "forward_hidden_exact": bool(torch_module.equal(forward.hidden, expected_forward)),
        "forward_weights_exact": bool(
            torch_module.equal(forward.route_weights, inputs["route_weights"])
        ),
        "backward_hidden_exact": bool(torch_module.equal(backward.hidden, expected_backward)),
        "gate_reduce_copy_exact": bool(
            torch_module.equal(inputs["gradients"].gate_reduce, inputs["gradients"].gate)
        ),
        "up_reduce_copy_exact": bool(
            torch_module.equal(inputs["gradients"].up_reduce, inputs["gradients"].up)
        ),
        "down_reduce_copy_exact": bool(
            torch_module.equal(inputs["gradients"].down_reduce, inputs["gradients"].down)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"native stub flow mismatch: {', '.join(failed)}")
    return checks


def _version_or_unknown(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_metadata(torch_module, root: Path) -> dict[str, object]:
    try:
        soc = str(torch_module.npu.get_device_name())
    except Exception:
        soc = "unknown"
    return {
        "git_sha": _git_sha(root),
        "python": platform.python_version(),
        "torch": getattr(torch_module, "__version__", "unknown"),
        "torch_npu": _version_or_unknown("torch-npu"),
        "cann_version": os.environ.get(
            "ASCEND_TOOLKIT_VERSION", os.environ.get("CANN_VERSION", "unknown")
        ),
        "cann_home": os.environ.get("ASCEND_HOME_PATH", "unknown"),
        "driver_version": os.environ.get("TILEXR_DRIVER_VERSION", "unknown"),
        "soc": soc,
    }


def topology_metadata(context) -> dict[str, object]:
    physical = int(os.environ.get("TILEXR_PHYSICAL_DEVICE_COUNT", "1"))
    ranks_per_device = int(os.environ.get("TILEXR_RANKS_PER_DEVICE", "1"))
    default_block_dim = 32 if ranks_per_device == 2 else 64
    planner_block_dim = int(
        os.environ.get("TILEXR_MOONEP_PLANNER_BLOCK_DIM", str(default_block_dim))
    )
    if planner_block_dim < context.planner_group_size or planner_block_dim > 64:
        raise ValueError(
            "effective TILEXR_MOONEP_PLANNER_BLOCK_DIM must satisfy "
            f"planner_group_size={context.planner_group_size} <= blockDim <= 64"
        )
    return {
        "global_rank": context.global_rank,
        "global_world_size": context.global_world_size,
        "node_rank": context.node_rank,
        "node_count": context.node_count,
        "local_rank": context.local_rank,
        "local_world_size": context.local_world_size,
        "planner_group_rank": context.planner_group_rank,
        "planner_group_size": context.planner_group_size,
        "lane_group_rank": context.lane_group_rank,
        "lane_group_size": context.lane_group_size,
        "physical_device_count": physical,
        "ranks_per_device": ranks_per_device,
        "oversubscribed": os.environ.get("TILEXR_OVERSUBSCRIBED", "0") == "1",
        "planner_block_dim": planner_block_dim,
        "planner_block_dim_source": os.environ.get(
            "TILEXR_MOONEP_PLANNER_BLOCK_DIM_SOURCE",
            "default_oversubscribed" if ranks_per_device == 2 else "default_native",
        ),
        "peer_memory_cross_node": context.node_count > 1,
        "cross_node_validated": os.environ.get(
            "TILEXR_MOONEP_CROSS_NODE_VALIDATED", "0"
        ) == "1",
    }


def run_case(torch_module, case, args, root: Path) -> None:
    from tilexr_moonep import TileXRMoonEPBuffer, TileXRMoonEPContext

    dtype = _torch_dtype(torch_module, case.dtype)
    output_root = Path(args.output_dir).resolve()
    global_rank = int(os.environ.get("RANK", "0"))
    rank_dir = (output_root / case.case_id / f"rank_{global_rank}").resolve()
    if output_root not in rank_dir.parents:
        raise ValueError(f"case artifact path escapes output root: {rank_dir}")
    result = {
        "schema_version": 1,
        "status": "failed",
        "failure_reason": None,
        "case": case.as_dict(),
        "rank": global_rank,
        "topology": None,
        "capabilities": None,
        "environment": environment_metadata(torch_module, root),
        "validation": {"passed": False, "mode": "not_run"},
        "stage_order": list(STAGE_ORDER),
    }
    context = None
    buffer = None
    failure = None

    def record_failure(step: str, exc: Exception) -> None:
        nonlocal failure
        reason = f"{step} failed: {type(exc).__name__}: {exc}"
        if result["failure_reason"]:
            result["failure_reason"] += f"; {reason}"
        else:
            result["status"] = "failed"
            result["failure_reason"] = reason
        if failure is None:
            failure = (exc, exc.__traceback__)

    try:
        context = TileXRMoonEPContext.from_env(
            tokens_per_rank=case.tokens_per_rank,
            hidden_size=case.hidden_size,
            topk=case.topk,
            expert_count=case.expert_count,
            dtype=dtype,
            install_prefix=args.install_prefix,
            torch_module=torch_module,
        )
        buffer = TileXRMoonEPBuffer(
            context, wait_iterations=args.wait_iterations, torch_module=torch_module
        )
        capabilities = context.runtime.capabilities.as_dict()
        result["rank"] = context.global_rank
        result["capabilities"] = capabilities
        result["topology"] = topology_metadata(context)
        inputs = make_inputs(torch_module, case, context)
        planning_epoch = 0

        def coordinated_iteration(timer=None):
            nonlocal planning_epoch
            _oversubscribed_planning_barrier(buffer, case.case_id, planning_epoch)
            planning_epoch += 1
            return execute_iteration(buffer, inputs, timer)

        if int(inputs["tokens_per_expert"].sum().item()) != context.dispatched_capacity:
            raise RuntimeError("tokens_per_expert does not sum to S*K")
        validation = {"passed": True, "mode": "disabled"}
        if case.correctness:
            first = coordinated_iteration()
            validation = validate_plan(first[0], context)
            implementations = capabilities["implementations"]
            stub_stages = ("dispatch", "prefetch_weight", "combine", "reduce_grad")
            if all(implementations[stage] == "stub" for stage in stub_stages):
                validation["stub_flow_checks"] = validate_stub_flow(
                    torch_module, inputs, first[1], first[2], context
                )
                validation["mode"] = "planner_cpu_oracle_and_stub_flow"
            first_checksum = _checksum(first[1].hidden) + _checksum(first[2].hidden)
            second = coordinated_iteration()
            second_checksum = _checksum(second[1].hidden) + _checksum(second[2].hidden)
            if first_checksum != second_checksum:
                raise RuntimeError(
                    f"stub flow is not deterministic: {first_checksum} != {second_checksum}"
                )
            validation["checksum"] = first_checksum
            validation["deterministic"] = True
        warmup_plan = None
        for _ in range(case.warmup):
            warmup_plan = coordinated_iteration()[0]
        if warmup_plan is not None and int(warmup_plan.status.item()) != 0:
            raise RuntimeError(
                f"Planner device status is {int(warmup_plan.status.item())} during warmup"
            )
        samples = []
        for iteration in range(case.iterations):
            timer = DeviceEventTimer(torch_module)
            plan, forward, backward, timings = coordinated_iteration(timer)
            if int(plan.status.item()) != 0:
                raise RuntimeError(f"Planner device status is {int(plan.status.item())}")
            samples.append(
                {
                    "iteration": iteration,
                    "timings_us": timings,
                    "checksums": {
                        "forward": _checksum(forward.hidden),
                        "backward": _checksum(backward.hidden),
                    },
                }
            )
        write_jsonl(rank_dir / "samples.jsonl", samples)
        result["status"] = "passed"
        result["validation"] = validation
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        failure = (exc, exc.__traceback__)
    finally:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        quiesced = buffer is None
        if buffer is not None:
            try:
                buffer.quiesce()
                quiesced = True
                buffer.check_pending_status()
            except Exception as sync_exc:
                record_failure("synchronize", sync_exc)
        local_passed = failure is None and result["status"] == "passed"
        try:
            decision = completion_barrier_from_env(
                global_rank,
                world_size,
                case_id=case.case_id,
                quiesced=quiesced,
                passed=local_passed,
            )
        except Exception as barrier_exc:
            record_failure("completion rendezvous", barrier_exc)
            _hold_unsafe_teardown(
                output_root=output_root,
                rank_dir=rank_dir,
                result=result,
                rank=global_rank,
                reason=str(barrier_exc),
                world_size=world_size,
            )
        if not decision.release:
            unsafe = RuntimeError("one or more ranks could not prove local NPU quiescence")
            record_failure("completion rendezvous", unsafe)
            _hold_unsafe_teardown(
                output_root=output_root,
                rank_dir=rank_dir,
                result=result,
                rank=global_rank,
                reason=str(unsafe),
                world_size=world_size,
            )
        if decision.abort and local_passed:
            record_failure(
                "completion rendezvous",
                RuntimeError("another rank failed the benchmark case"),
            )
        owner = buffer if buffer is not None else context
        if owner is not None:
            try:
                owner.close()
            except Exception as close_exc:
                record_failure("close", close_exc)
        write_json(rank_dir / "result.json", result)
    if failure is not None:
        raise failure[0].with_traceback(failure[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TileXR MoonEP complete-flow benchmark")
    build_case_parser(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--install-prefix", default=None)
    parser.add_argument("--wait-iterations", type=int, default=1_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wait_iterations <= 0:
        raise ValueError("wait_iterations must be positive")
    if (
        int(os.environ.get("WORLD_SIZE", "1")) > 1
        and os.environ.get("TILEXR_MOONEP_MANAGED_LAUNCH") != "1"
    ):
        raise RuntimeError("multi-rank MoonEP workers require the managed launcher")
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    cases = select_cases(load_cases(args.cases), args.case_ids)
    cases = [apply_overrides(case, args) for case in cases]
    root = Path(__file__).resolve().parents[2]
    if int(os.environ.get("RANK", "0")) == 0:
        write_json(
            Path(args.output_dir) / "metadata.json",
            {
                "schema_version": 1,
                "cases": [case.as_dict() for case in cases],
                "environment": environment_metadata(torch, root),
            },
        )
    for case in cases:
        run_case(torch, case, args, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
