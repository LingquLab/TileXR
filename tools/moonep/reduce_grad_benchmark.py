from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


MIN_RANK_COUNT = 4
INT32_MAX = (1 << 31) - 1
UDMA_REGISTRATION_ALIGNMENT = 2 * 1024 * 1024
NATIVE_BASELINE_COMMIT = "a49538a45e5c5bdc82aa6ae02548f99e72ec67eb"
PROJECTION_NAMES = ("gate", "up", "down")
PATTERNS = ("empty", "sparse", "mixed", "heavy", "full", "balanced-full")


@dataclass(frozen=True)
class ReduceGradDimensions:
    ranks: int
    experts_per_rank: int
    slots: int
    hidden: int
    projection: int

    def __post_init__(self) -> None:
        if self.ranks < MIN_RANK_COUNT:
            raise ValueError(f"ReduceGrad requires at least {MIN_RANK_COUNT} ranks")
        if min(self.experts_per_rank, self.slots, self.hidden, self.projection) <= 0:
            raise ValueError("ReduceGrad dimensions must be positive")
        if self.experts > INT32_MAX or self.slots > INT32_MAX:
            raise ValueError("experts and slots must fit in int32")
        if self.ranks * self.slots > INT32_MAX:
            raise ValueError("rank-by-slot contributor indices must fit in int32")

    @property
    def experts(self) -> int:
        return self.ranks * self.experts_per_rank

    @property
    def row_elements(self) -> int:
        return self.hidden * self.projection

    @property
    def row_bytes(self) -> int:
        return self.row_elements * 4


def build_experts_to_copy(
    ranks: int,
    slots: int,
    experts_per_rank: int,
    pattern: str,
) -> list[list[int]]:
    if ranks < MIN_RANK_COUNT:
        raise ValueError(f"ReduceGrad requires at least {MIN_RANK_COUNT} ranks")
    if slots <= 0 or experts_per_rank <= 0:
        raise ValueError("invalid ReduceGrad plan dimensions")
    if ranks * experts_per_rank > INT32_MAX or slots > INT32_MAX:
        raise ValueError("ReduceGrad expert and slot counts must fit in int32")
    if ranks * slots > INT32_MAX:
        raise ValueError("rank-by-slot contributor indices must fit in int32")
    if pattern not in PATTERNS:
        raise ValueError(f"unknown ReduceGrad pattern {pattern!r}")

    plan = [[-1 for _ in range(slots)] for _ in range(ranks)]
    if pattern == "empty":
        return plan
    if pattern == "balanced-full":
        for source in range(ranks):
            for slot in range(slots):
                owner = (source + slot) % ranks
                plan[source][slot] = owner * experts_per_rank + slot % experts_per_rank
        return plan

    counts_by_pattern = {
        "sparse": (1,),
        "mixed": (3, 0, 2, 1),
        "heavy": (3, 3, 2, 3, 1, 0, 2, 3),
        "full": (3,),
    }
    counts = counts_by_pattern[pattern]
    for slot in range(min(slots, experts_per_rank)):
        count = min(int(counts[slot % len(counts)]), ranks - 1)
        for source in range(1, count + 1):
            plan[source][slot] = slot
    return plan


def flatten_plan(plan: Sequence[Sequence[int]]) -> list[int]:
    return [int(value) for row in plan for value in row]


def plan_statistics(plan: Sequence[Sequence[int]]) -> dict[str, Any]:
    flattened = flatten_plan(plan)
    live = sum(value >= 0 for value in flattened)
    counts: dict[int, int] = {}
    for expert in flattened:
        if expert >= 0:
            counts[expert] = counts.get(expert, 0) + 1
    total = len(flattened)
    return {
        "entries": total,
        "live_entries": live,
        "density": (float(live) / total if total else 0.0),
        "active_experts": len(counts),
        "max_contributors_per_expert": max(counts.values(), default=0),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def cross_rank_statistics(samples_by_rank: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not samples_by_rank or not samples_by_rank[0]:
        raise ValueError("cross-rank statistics require samples")
    iterations = len(samples_by_rank[0])
    if any(len(samples) != iterations for samples in samples_by_rank):
        raise ValueError("all ranks must report the same number of samples")
    maxima = [
        max(float(samples[iteration]) for samples in samples_by_rank)
        for iteration in range(iterations)
    ]
    return {
        "cross_rank_max_us": maxima,
        "p50_us": percentile(maxima, 0.50),
        "p99_us": percentile(maxima, 0.99),
        "min_us": min(maxima),
        "max_us": max(maxima),
        "mean_us": sum(maxima) / len(maxima),
    }


def source_value(source_rank: int, projection: int, slot: int) -> float:
    value = (
        (projection + 1) * 0.5
        + (source_rank + 1) * 0.03125
        + (slot + 1) * 0.0009765625
    )
    return struct.unpack("f", struct.pack("f", value))[0]


def fp32_add(lhs: float, rhs: float) -> float:
    return struct.unpack("f", struct.pack("f", float(lhs) + float(rhs)))[0]


def expected_expert_value(
    plan: Sequence[Sequence[int]], global_expert: int, projection: int
) -> float:
    value = 0.0
    for source, row in enumerate(plan):
        for slot, expert in enumerate(row):
            if int(expert) == global_expert:
                value = fp32_add(value, source_value(source, projection, slot))
    return value


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _command_output(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command), check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return output if output else None


def environment_metadata(torch_module, torch_npu_module) -> dict[str, Any]:
    npu_smi = _command_output(("npu-smi", "info"))
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch_module.__version__),
        "torch_npu": str(torch_npu_module.__version__),
        "cann_home": os.environ.get("ASCEND_HOME_PATH"),
        "cann_opp_path": os.environ.get("ASCEND_OPP_PATH"),
        "driver_summary": None if npu_smi is None else npu_smi.splitlines()[:4],
    }


def _current_stream_ptr(torch_module, device_index: int) -> int:
    if int(torch_module.npu.current_device()) != int(device_index):
        raise RuntimeError(f"current NPU device is not npu:{device_index}")
    stream = torch_module.npu.current_stream()
    value = getattr(stream, "npu_stream", None)
    if value is None:
        value = getattr(stream, "stream", None)
    if value is None:
        raise RuntimeError("torch.npu.current_stream() exposes no native stream pointer")
    return int(value)


def _aligned_workspace(torch_module, device: str, size_bytes: int, alignment: int):
    if size_bytes <= 0 or alignment <= 0:
        raise ValueError("workspace size and alignment must be positive")
    allocation = torch_module.empty(
        (size_bytes + alignment - 1,), dtype=torch_module.uint8, device=device
    )
    offset = (-int(allocation.data_ptr())) % alignment
    workspace = allocation.narrow(0, offset, size_bytes)
    if int(workspace.data_ptr()) % alignment:
        raise RuntimeError("failed to align ReduceGrad UDMA workspace")
    return workspace, allocation


def _aligned_float32_tensor(torch_module, device: str, shape: Sequence[int]):
    elements = math.prod(int(value) for value in shape)
    alignment_elements = UDMA_REGISTRATION_ALIGNMENT // 4
    allocation = torch_module.empty(
        (elements + alignment_elements - 1,),
        dtype=torch_module.float32,
        device=device,
    )
    offset_bytes = (-int(allocation.data_ptr())) % UDMA_REGISTRATION_ALIGNMENT
    if offset_bytes % 4:
        raise RuntimeError("float32 allocation cannot satisfy UDMA alignment")
    tensor = allocation.narrow(0, offset_bytes // 4, elements).reshape(tuple(shape))
    if int(tensor.data_ptr()) % UDMA_REGISTRATION_ALIGNMENT:
        raise RuntimeError("failed to align UDMA float32 tensor")
    return tensor, allocation


def gradient_source_regions(full_gradients, expert_count: int, slots: int):
    sources = [
        gradient.narrow(0, expert_count, slots) for gradient in full_gradients
    ]
    return sources, list(full_gradients)


class TileXRReduceGradRunner:
    def __init__(self, torch_module, dimensions: ReduceGradDimensions, plan, args):
        from tilexr_moonep import (
            MoonEPPlan,
            ProjectionBuffers,
            TileXRMoonEPRuntime,
        )

        self.torch = torch_module
        self.dimensions = dimensions
        self.plan_host = plan
        self.rank = int(os.environ["RANK"])
        self.device_index = int(os.environ["LOCAL_RANK"])
        self.device = f"npu:{self.device_index}"
        if args.num_sms is not None:
            os.environ["TILEXR_MOONEP_REDUCE_GRAD_BLOCK_DIM"] = str(args.num_sms)
        if args.qp_route_spec is not None:
            os.environ["TILEXR_UDMA_QP_ROUTE_SPEC"] = args.qp_route_spec
        self.runtime = TileXRMoonEPRuntime(
            self.rank,
            dimensions.ranks,
            install_prefix=args.install_prefix,
        )
        self.prepared = None
        self.closed = False
        plan_tensor = torch_module.tensor(
            plan, dtype=torch_module.int32, device=self.device
        )
        self.plan = MoonEPPlan(
            dst=torch_module.zeros((1,), dtype=torch_module.int32, device=self.device),
            experts_to_copy=plan_tensor,
            zero_fill_ranges=torch_module.zeros(
                (dimensions.experts + dimensions.slots, 2),
                dtype=torch_module.int32,
                device=self.device,
            ),
            remote_stats=torch_module.zeros(
                (2,), dtype=torch_module.int32, device=self.device
            ),
            dup_groups=torch_module.zeros(
                (1, 3), dtype=torch_module.int32, device=self.device
            ),
            dup_loffs=torch_module.zeros(
                (1,), dtype=torch_module.int32, device=self.device
            ),
            dup_counts=torch_module.zeros(
                (2,), dtype=torch_module.int32, device=self.device
            ),
            status=torch_module.zeros(
                (1,), dtype=torch_module.int32, device=self.device
            ),
            reduce_grad_status=torch_module.zeros(
                (1,), dtype=torch_module.int32, device=self.device
            ),
            workspace=torch_module.empty(
                (1,), dtype=torch_module.uint8, device=self.device
            ),
            n=1,
            tokens_per_rank=1,
            topk=1,
            expert_count=dimensions.experts,
            rank_size=dimensions.ranks,
            prefetch_slots=dimensions.slots,
            nv_s=1,
            token_padding=1,
            epoch=1,
            backend="tilexr",
            runtime=self.runtime,
        )
        rows = dimensions.experts + dimensions.slots
        shapes = (
            (rows, dimensions.hidden, dimensions.projection),
            (rows, dimensions.hidden, dimensions.projection),
            (rows, dimensions.projection, dimensions.hidden),
        )
        gradient_pairs = [
            _aligned_float32_tensor(torch_module, self.device, shape)
            for shape in shapes
        ]
        self.full_gradients = [pair[0] for pair in gradient_pairs]
        self.gradient_allocations = [pair[1] for pair in gradient_pairs]
        for gradient in self.full_gradients:
            gradient.zero_()
        self.gradients = ProjectionBuffers(*self.full_gradients)
        self.sources, self.source_registrations = gradient_source_regions(
            self.full_gradients, dimensions.experts, dimensions.slots
        )
        self.source_strategy = "gradient-backing-interior-slice"
        self.reset_sources()
        torch_module.npu.synchronize()
        prepare_start = time.perf_counter()
        self.info = self.runtime.reduce_grad_workspace_info(
            None,
            self.plan,
            self.gradients,
            requested_udma_chunk_bytes=args.chunk_bytes,
        )
        self.workspace, self.workspace_allocation = _aligned_workspace(
            torch_module,
            self.device,
            self.info.workspace_bytes,
            self.info.workspace_alignment,
        )
        self.prepared = self.runtime.prepare_reduce_grad(
            None,
            self.plan,
            self.gradients,
            self.sources,
            self.source_registrations,
            self.workspace,
            requested_udma_chunk_bytes=args.chunk_bytes,
        )
        torch_module.npu.synchronize()
        self.prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
        self.wait_iterations = args.wait_iterations

    def reset_sources(self) -> None:
        for projection, source in enumerate(self.sources):
            for slot in range(self.dimensions.slots):
                source[slot].fill_(source_value(self.rank, projection, slot))

    def launch(self) -> None:
        self.runtime.reduce_grad(
            None,
            self.plan,
            self.gradients,
            self.sources,
            self.source_registrations,
            self.prepared,
            _current_stream_ptr(self.torch, self.device_index),
            self.wait_iterations,
        )

    def synchronize(self) -> None:
        self.torch.npu.synchronize(self.device_index)
        status = int(self.plan.reduce_grad_status.item())
        if status != 0:
            raise RuntimeError(f"TileXR ReduceGrad device status is {status}, expected 0")

    def validate(self) -> dict[str, Any]:
        d = self.dimensions
        owner_begin = self.rank * d.experts_per_rank
        owner_end = owner_begin + d.experts_per_rank
        for projection, gradient in enumerate(self.full_gradients):
            actual = gradient[owner_begin:owner_end]
            expected = self.torch.empty_like(gradient[owner_begin:owner_end])
            for local_expert in range(d.experts_per_rank):
                expected[local_expert].fill_(expected_expert_value(
                    self.plan_host, owner_begin + local_expert, projection
                ))
            if not self.torch.equal(actual, expected):
                actual_flat = actual.reshape(-1)
                expected_flat = expected.reshape(-1)
                mismatch = actual_flat != expected_flat
                mismatch_index = int(mismatch.to(self.torch.int32).argmax().item())
                staging_offset = int(self.info.lane_state_bytes)
                staging_bytes = bytes(
                    int(value) for value in
                    self.workspace.narrow(0, staging_offset, 4).cpu().tolist()
                )
                staging_first = struct.unpack("f", staging_bytes)[0]
                raise RuntimeError(
                    f"TileXR projection {projection} owner rows differ from ordered "
                    f"FP32 reference at flat index {mismatch_index}: "
                    f"actual={float(actual_flat[mismatch_index].item())}, "
                    f"expected={float(expected_flat[mismatch_index].item())}, "
                    f"gate_lane0_bank0_staging={staging_first}"
                )
            if owner_begin and bool(self.torch.count_nonzero(gradient[:owner_begin]).item()):
                raise RuntimeError(f"TileXR projection {projection} modified lower non-owner rows")
            if owner_end < d.experts and bool(
                self.torch.count_nonzero(gradient[owner_end:d.experts]).item()
            ):
                raise RuntimeError(f"TileXR projection {projection} modified upper non-owner rows")
            for slot in range(d.slots):
                live = int(self.plan_host[self.rank][slot]) >= 0
                if live:
                    if bool(self.torch.count_nonzero(self.sources[projection][slot]).item()):
                        raise RuntimeError(
                            f"TileXR projection {projection} live source slot {slot} was not cleared"
                        )
                else:
                    expected_source = source_value(self.rank, projection, slot)
                    if not bool(self.torch.all(
                        self.sources[projection][slot] == expected_source
                    ).item()):
                        raise RuntimeError(
                            f"TileXR projection {projection} unused source slot {slot} changed"
                        )
        return {"passed": True, "mode": "exact_ordered_fp32"}

    def layout_metadata(self) -> dict[str, Any]:
        metadata = self.info.as_dict()
        metadata["source_strategy"] = self.source_strategy
        metadata["source_alignment"] = UDMA_REGISTRATION_ALIGNMENT
        metadata["configured_qp_route_spec"] = os.environ.get(
            "TILEXR_UDMA_QP_ROUTE_SPEC"
        )
        return metadata

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.torch.npu.synchronize(self.device_index)
            self.runtime.destroy_reduce_grad(self.prepared)
            self.prepared = None
        finally:
            self.runtime.close()
            self.closed = True


class NativeReduceGradRunner:
    def __init__(self, torch_module, dimensions: ReduceGradDimensions, plan, args):
        native_root = Path(args.native_root).resolve()
        if str(native_root) not in sys.path:
            sys.path.insert(0, str(native_root))
        from ascend_moonep.buffer_c import (
            create_sym_dist_tensor,
            shmem_finalize,
            shmem_init,
        )
        from ascend_moonep.launchers import launch_grad_reduce

        self.torch = torch_module
        self.dimensions = dimensions
        self.plan_host = plan
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = f"npu:{self.local_rank}"
        self._launch_grad_reduce = launch_grad_reduce
        self._shmem_finalize = shmem_finalize
        self.num_sms = (
            int(args.num_sms) if args.num_sms is not None else
            int(os.environ.get("TILEXR_MOONEP_NATIVE_REDUCE_GRAD_BLOCK_DIM", "32"))
        )
        projection_bytes = (dimensions.experts_per_rank + dimensions.slots) * (
            dimensions.row_bytes
        )
        heap_bytes = _align_up(3 * projection_bytes + 256 * 1024 * 1024, 2 * 1024 * 1024)
        prepare_start = time.perf_counter()
        shmem_init(group=None, mem_size=heap_bytes)
        shapes = (
            (dimensions.hidden, dimensions.projection),
            (dimensions.hidden, dimensions.projection),
            (dimensions.projection, dimensions.hidden),
        )
        self.outputs = [
            create_sym_dist_tensor(
                (dimensions.experts_per_rank, *shape),
                torch_module.float32,
                self.local_rank,
            )
            for shape in shapes
        ]
        self.sources = [
            create_sym_dist_tensor(
                (dimensions.slots, *shape),
                torch_module.float32,
                self.local_rank,
            )
            for shape in shapes
        ]
        for output in self.outputs:
            output.zero_()
        self.plan = torch_module.tensor(
            plan, dtype=torch_module.int32, device=self.device
        )
        self.reset_sources()
        torch_module.npu.synchronize()
        self.prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

    def reset_sources(self) -> None:
        for projection, source in enumerate(self.sources):
            for slot in range(self.dimensions.slots):
                source[slot].fill_(source_value(self.rank, projection, slot))

    def launch(self) -> None:
        for output, source in zip(self.outputs, self.sources):
            self._launch_grad_reduce(
                output,
                source,
                self.plan,
                rank=self.rank,
                num_sms=self.num_sms,
                meta_buf=None,
                meta_stride=0,
                barrier_off=0,
                grid_sync_bar=None,
            )

    def synchronize(self) -> None:
        self.torch.npu.synchronize()

    def validate(self) -> dict[str, Any]:
        d = self.dimensions
        owner_begin = self.rank * d.experts_per_rank
        for projection, output in enumerate(self.outputs):
            expected = self.torch.empty_like(output)
            for local_expert in range(d.experts_per_rank):
                expected[local_expert].fill_(expected_expert_value(
                    self.plan_host, owner_begin + local_expert, projection
                ))
            if not self.torch.equal(output, expected):
                raise RuntimeError(
                    f"native projection {projection} owner rows differ from ordered FP32 reference"
                )
            for slot in range(d.slots):
                live = int(self.plan_host[self.rank][slot]) >= 0
                if live:
                    if bool(self.torch.count_nonzero(self.sources[projection][slot]).item()):
                        raise RuntimeError(
                            f"native projection {projection} live source slot {slot} was not cleared"
                        )
                else:
                    expected_source = source_value(self.rank, projection, slot)
                    if not bool(self.torch.all(
                        self.sources[projection][slot] == expected_source
                    ).item()):
                        raise RuntimeError(
                            f"native projection {projection} unused source slot {slot} changed"
                        )
        return {"passed": True, "mode": "exact_ordered_fp32"}

    def layout_metadata(self) -> dict[str, Any]:
        return {
            "block_dim": self.num_sms,
            "launch_count": 3,
            "transport": "cann-shmem-mte-get",
        }

    def close(self) -> None:
        self._shmem_finalize()


def _time_iteration(torch_module, distributed, runner) -> float:
    runner.reset_sources()
    torch_module.npu.synchronize()
    distributed.barrier()
    start = torch_module.npu.Event(enable_timing=True)
    end = torch_module.npu.Event(enable_timing=True)
    start.record()
    runner.launch()
    end.record()
    end.synchronize()
    elapsed_us = float(start.elapsed_time(end)) * 1000.0
    runner.synchronize()
    return elapsed_us


def _run(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    import torch_npu

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dimensions = ReduceGradDimensions(
        ranks=world_size,
        experts_per_rank=args.experts_per_rank,
        slots=args.slots,
        hidden=args.hidden,
        projection=args.projection,
    )
    torch.npu.set_device(local_rank)
    init_kwargs = {
        "backend": args.coordination_backend,
        "timeout": datetime.timedelta(seconds=args.coordination_timeout_seconds),
    }
    if args.coordination_backend == "hccl":
        init_kwargs["device_id"] = torch.device(f"npu:{local_rank}")
    dist.init_process_group(**init_kwargs)
    dist.barrier()
    plan = build_experts_to_copy(
        world_size, dimensions.slots, dimensions.experts_per_rank, args.pattern
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "backend": args.backend,
        "rank": rank,
        "world_size": world_size,
        "coordination_backend": args.coordination_backend,
        "case": {
            **asdict(dimensions),
            "experts": dimensions.experts,
            "pattern": args.pattern,
            "num_sms": args.num_sms,
            "qp_route_spec": args.qp_route_spec,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "plan": plan_statistics(plan),
        "timing_boundary": {
            "clock": "NPU device events on current stream",
            "tilexr": "one fused TileXRMoonEpReduceGradV2 launch",
            "native": "three sequential launch_grad_reduce launches",
            "excluded": [
                "allocation",
                "MR registration/import",
                "prepared-profile creation",
                "source reset",
                "cross-rank pre-launch alignment",
                "post-launch status synchronization",
            ],
        },
    }
    failure = None
    try:
        setup_start = time.perf_counter()
        if args.backend == "tilexr":
            runner = TileXRReduceGradRunner(torch, dimensions, plan, args)
        else:
            runner = NativeReduceGradRunner(torch, dimensions, plan, args)
        result["setup_ms"] = (time.perf_counter() - setup_start) * 1000.0
        result["prepare_ms"] = runner.prepare_ms
        result["layout"] = runner.layout_metadata()

        correctness = {"passed": None, "mode": "disabled"}
        if args.correctness:
            required = 3 * (dimensions.experts + dimensions.slots) * dimensions.row_bytes
            if required > args.correctness_max_bytes:
                raise ValueError(
                    f"exact correctness would inspect {required} bytes, above "
                    f"--correctness-max-bytes={args.correctness_max_bytes}"
                )
            runner.reset_sources()
            torch.npu.synchronize()
            dist.barrier()
            runner.launch()
            runner.synchronize()
            correctness = runner.validate()
            dist.barrier()
        result["correctness"] = correctness

        for _ in range(args.warmup):
            _time_iteration(torch, dist, runner)
        samples = [
            _time_iteration(torch, dist, runner) for _ in range(args.iterations)
        ]
        gathered: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, samples)
        result["local_samples_us"] = samples
        result["status"] = "passed"
        if rank == 0:
            result["samples_by_rank_us"] = gathered
            result["statistics"] = cross_rank_statistics(gathered)
            result["commits"] = {
                "tilexr_base": args.tilexr_commit,
                "tilexr_source_snapshot_sha256": args.source_snapshot_sha256,
                "native_moonep": NATIVE_BASELINE_COMMIT,
            }
            result["environment"] = environment_metadata(torch, torch_npu)
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        failure = (exc, exc.__traceback__)
    finally:
        try:
            torch.npu.synchronize()
            dist.barrier()
            if runner is not None:
                runner.close()
            dist.barrier()
        except Exception as cleanup_error:
            result["cleanup_error"] = (
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
            if failure is None:
                failure = (cleanup_error, cleanup_error.__traceback__)
                result["status"] = "failed"
        _write_json(output_dir / f"rank_{rank}.json", result)
        if rank == 0:
            _write_json(output_dir / "summary.json", result)
        dist.destroy_process_group()
    if failure is not None:
        raise failure[0].with_traceback(failure[1])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated TileXR versus native MoonEP ReduceGrad benchmark"
    )
    parser.add_argument("--backend", choices=("tilexr", "native"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--install-prefix", default=None)
    parser.add_argument("--native-root", default="reference/ascend-moonep-dev")
    parser.add_argument("--experts-per-rank", type=int, default=8)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--projection", type=int, default=32)
    parser.add_argument("--pattern", choices=PATTERNS, default="mixed")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument(
        "--qp-route-spec",
        default=os.environ.get("TILEXR_UDMA_QP_ROUTE_SPEC"),
    )
    parser.add_argument("--wait-iterations", type=int, default=1_000_000)
    parser.add_argument(
        "--coordination-backend", choices=("gloo", "hccl"), default="gloo"
    )
    parser.add_argument("--coordination-timeout-seconds", type=int, default=120)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--correctness-max-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--tilexr-commit", default=os.environ.get("TILEXR_GIT_COMMIT"))
    parser.add_argument(
        "--source-snapshot-sha256",
        default=os.environ.get("TILEXR_SOURCE_SNAPSHOT_SHA256"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if args.chunk_bytes <= 0 or args.wait_iterations <= 0:
        raise ValueError("chunk-bytes and wait-iterations must be positive")
    if args.num_sms is not None and args.num_sms <= 0:
        raise ValueError("num-sms must be positive")
    if args.correctness_max_bytes <= 0:
        raise ValueError("correctness-max-bytes must be positive")
    if args.coordination_timeout_seconds <= 0:
        raise ValueError("coordination-timeout-seconds must be positive")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
