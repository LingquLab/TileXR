from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECTION_NAMES = ("gate", "up", "down")
TILEXR_SUCCESS_STATUS = 4000
MIN_VALID_EVENT_US = 1.0


def projection_shapes(args):
    return (
        (args.hidden, args.projection),
        (args.hidden, args.projection),
        (args.projection, args.hidden),
    )


def percentile(values, quantile):
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def statistics(samples_by_rank):
    if not samples_by_rank or not samples_by_rank[0]:
        raise ValueError("cross-rank statistics require samples")
    iterations = len(samples_by_rank[0])
    if any(len(samples) != iterations for samples in samples_by_rank):
        raise ValueError("all ranks must report the same number of samples")
    maxima = [
        max(float(rank_samples[index]) for rank_samples in samples_by_rank)
        for index in range(iterations)
    ]
    valid = [value for value in maxima if value > MIN_VALID_EVENT_US]
    if not valid:
        raise ValueError("all cross-rank NPU event samples are invalid")
    return {
        "cross_rank_max_us": maxima,
        "valid_cross_rank_max_us": valid,
        "invalid_event_sample_count": len(maxima) - len(valid),
        "minimum_valid_event_us_exclusive": MIN_VALID_EVENT_US,
        "p50_us": percentile(valid, 0.50),
        "p99_us": percentile(valid, 0.99),
        "min_us": min(valid),
        "max_us": max(valid),
        "mean_us": sum(valid) / len(valid),
    }


def constant_value(owner, local_expert, projection):
    return float(owner * 16 + local_expert + projection * 0.25)


def ring_plan(ranks, experts_per_rank, slots):
    if slots > experts_per_rank:
        raise ValueError("slots must not exceed experts_per_rank")
    return [
        [((rank - 1) % ranks) * experts_per_rank + slot for slot in range(slots)]
        for rank in range(ranks)
    ]


def current_stream_ptr(torch_module):
    stream = torch_module.npu.current_stream()
    value = getattr(stream, "npu_stream", getattr(stream, "stream", None))
    if value is None:
        raise RuntimeError("current NPU stream exposes no native pointer")
    return int(value)


def command_output(command, cwd=None):
    try:
        completed = subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout.strip() or completed.stderr.strip()) or None


class TileXRRunner:
    def __init__(self, torch_module, args, rank, ranks, device, plan_host):
        from tilexr_moonep import MoonEPPlan, ProjectionBuffers, TileXRMoonEPRuntime

        self.torch = torch_module
        self.rank = rank
        self.ranks = ranks
        self.device = device
        self.args = args
        self.shapes = projection_shapes(args)
        os.environ["TILEXR_UDMA_QP_ROUTE_SPEC"] = args.qp_route_spec
        if args.num_sms is not None:
            os.environ["TILEXR_MOONEP_PREFETCH_BLOCK_DIM"] = str(args.num_sms)
        self.runtime = TileXRMoonEPRuntime(
            rank, ranks, install_prefix=args.install_prefix
        )
        self.context = SimpleNamespace(
            planner_group_size=ranks,
            expert_count=ranks * args.experts_per_rank,
            prefetch_slots=args.slots,
            nv_s=1,
            topk=1,
        )
        plan_tensor = torch_module.tensor(
            plan_host, dtype=torch_module.int32, device=device
        )
        self.plan = MoonEPPlan(
            dst=torch_module.zeros((1,), dtype=torch_module.int32, device=device),
            experts_to_copy=plan_tensor,
            zero_fill_ranges=torch_module.zeros(
                (self.context.expert_count + args.slots, 2),
                dtype=torch_module.int32,
                device=device,
            ),
            remote_stats=torch_module.tensor(
                [args.slots, args.slots], dtype=torch_module.int32, device=device
            ),
            dup_groups=torch_module.zeros(
                (1, 3), dtype=torch_module.int32, device=device
            ),
            dup_loffs=torch_module.zeros((1,), dtype=torch_module.int32, device=device),
            dup_counts=torch_module.zeros((2,), dtype=torch_module.int32, device=device),
            status=torch_module.zeros((1,), dtype=torch_module.int32, device=device),
            reduce_grad_status=torch_module.zeros(
                (1,), dtype=torch_module.int32, device=device
            ),
            workspace=torch_module.empty((1,), dtype=torch_module.uint8, device=device),
            n=1,
            tokens_per_rank=1,
            topk=1,
            expert_count=self.context.expert_count,
            rank_size=ranks,
            prefetch_slots=args.slots,
            nv_s=1,
            token_padding=1,
            epoch=1,
            backend="tilexr",
            runtime=self.runtime,
        )
        local_weights = []
        for projection, shape in enumerate(self.shapes):
            tensor = torch_module.empty(
                (args.experts_per_rank, *shape),
                dtype=torch_module.bfloat16,
                device=device,
            )
            for expert in range(args.experts_per_rank):
                tensor[expert].fill_(constant_value(rank, expert, projection))
            local_weights.append(tensor)
        self.projections = ProjectionBuffers.from_local_weights(
            SimpleNamespace(
                experts_per_rank=args.experts_per_rank,
                dtype=torch_module.bfloat16,
                device_index=int(os.environ["LOCAL_RANK"]),
            ),
            *local_weights,
            slot_fill_value=-7.0,
            torch_module=torch_module,
        )
        del local_weights
        torch_module.npu.synchronize()
        self.handle = self.runtime.udma_register(self.projections.backing)
        self.stream_ptr = current_stream_ptr(torch_module)

    def launch(self):
        self.runtime.prefetch_weight(
            self.context, self.plan, self.projections, self.stream_ptr
        )

    def synchronize(self):
        self.torch.npu.synchronize()
        actual = int(self.plan.status.item())
        if actual != TILEXR_SUCCESS_STATUS:
            raise RuntimeError(
                f"TileXR PrefetchWeight status {actual}, expected {TILEXR_SUCCESS_STATUS}"
            )

    def validate(self, plan_host):
        owner = (self.rank - 1) % self.ranks
        checks = {}
        for projection, name in enumerate(PROJECTION_NAMES):
            tensor = getattr(self.projections, name)
            for slot, expert in enumerate(plan_host[self.rank]):
                local_expert = int(expert) % self.args.experts_per_rank
                expected = constant_value(owner, local_expert, projection)
                actual = tensor[self.args.experts_per_rank + slot]
                passed = bool(self.torch.all(actual == expected).item())
                checks[f"{name}_{slot}"] = passed
                if not passed:
                    raise RuntimeError(
                        f"TileXR {name} slot {slot} differs from expert {expert}"
                    )
        return checks

    def layout(self):
        return {
            "launch_count": 1,
            "transport": "registered-memory UDMA GET",
            "qp_count": self.runtime.udma_qp_count,
            "block_dim_override": self.args.num_sms,
        }

    def close(self):
        self.torch.npu.synchronize()
        self.runtime.udma_unregister(self.handle)
        self.runtime.close()


class NativeRunner:
    def __init__(self, torch_module, dist, args, rank, ranks, device, plan_host):
        native_root = Path(args.native_root).resolve()
        sys.path.insert(0, str(native_root))
        from ascend_moonep import ShmemRuntime, launch_prefetch
        from ascend_moonep.buffer_c import (
            create_sym_tensor_from_ptr,
            create_vmm_physical,
            map_to_sym_ptr,
            reset_symmetric_descriptors,
        )

        self.torch = torch_module
        self.dist = dist
        self.args = args
        self.shapes = projection_shapes(args)
        self.rank = rank
        self.ranks = ranks
        self.device = device
        self.launch_prefetch = launch_prefetch
        self.ShmemRuntime = ShmemRuntime
        self.reset_symmetric_descriptors = reset_symmetric_descriptors
        self.sources = []
        for projection, shape in enumerate(self.shapes):
            source = create_vmm_physical(
                (args.experts_per_rank + args.slots, *shape),
                torch_module.bfloat16,
                int(os.environ["LOCAL_RANK"]),
            )[0]
            for expert in range(args.experts_per_rank):
                source[expert].fill_(constant_value(rank, expert, projection))
            source[args.experts_per_rank :].fill_(-7.0)
            self.sources.append(source)
        torch_module.npu.synchronize()
        dist.barrier()
        ShmemRuntime.init_with_buffer(group=None)
        self.plan = torch_module.tensor(
            plan_host[rank], dtype=torch_module.int32, device=device
        ).contiguous()
        self.remote_experts = []
        self.slots = []
        for projection, (source, shape) in enumerate(
            zip(self.sources, self.shapes)
        ):
            sym_base = map_to_sym_ptr(source.data_ptr(), f"projection_{projection}")
            row_bytes = math.prod(shape) * 2
            self.remote_experts.append(
                create_sym_tensor_from_ptr(
                    sym_base,
                    (ranks * args.experts_per_rank, *shape),
                    torch_module.bfloat16,
                    device,
                )
            )
            self.slots.append(
                create_sym_tensor_from_ptr(
                    sym_base + args.experts_per_rank * row_bytes,
                    (args.slots, *shape),
                    torch_module.bfloat16,
                    device,
                )
            )

    def launch(self):
        num_sms = self.args.num_sms or 32
        for remote_expert, slots in zip(self.remote_experts, self.slots):
            self.launch_prefetch(remote_expert, slots, self.plan, num_sms=num_sms)

    def synchronize(self):
        self.torch.npu.synchronize()

    def validate(self, plan_host):
        owner = (self.rank - 1) % self.ranks
        checks = {}
        for projection, (name, tensor) in enumerate(
            zip(PROJECTION_NAMES, self.slots)
        ):
            for slot, expert in enumerate(plan_host[self.rank]):
                local_expert = int(expert) % self.args.experts_per_rank
                expected = constant_value(owner, local_expert, projection)
                passed = bool(self.torch.all(tensor[slot] == expected).item())
                checks[f"{name}_{slot}"] = passed
                if not passed:
                    raise RuntimeError(
                        f"native {name} slot {slot} differs from expert {expert}"
                    )
        return checks

    def layout(self):
        return {
            "launch_count": 3,
            "transport": "CANN SHMEM MTE remote-to-UB-to-GM GET",
            "block_dim_requested": self.args.num_sms or 32,
        }

    def close(self):
        self.torch.npu.synchronize()
        self.dist.barrier()
        if self.ShmemRuntime.is_initialized():
            self.ShmemRuntime.finalize()
        for source in self.sources:
            allocation = getattr(source, "_vmm_allocation", None)
            if allocation is not None:
                allocation.destroy()
        self.reset_symmetric_descriptors()


def time_once(torch_module, dist, runner):
    torch_module.npu.synchronize()
    dist.barrier()
    start = torch_module.npu.Event(enable_timing=True)
    end = torch_module.npu.Event(enable_timing=True)
    start.record()
    runner.launch()
    end.record()
    end.synchronize()
    elapsed_us = float(start.elapsed_time(end)) * 1000.0
    runner.synchronize()
    return elapsed_us


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tilexr", "native"), required=True)
    parser.add_argument("--install-prefix")
    parser.add_argument("--native-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--projection", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-sms", type=int)
    parser.add_argument(
        "--qp-route-spec", default="port_count:6,port_count:2"
    )
    parser.add_argument("--tilexr-commit", default="unknown")
    parser.add_argument("--tilexr-source-sha256", default="unknown")
    parser.add_argument("--native-commit", default="unknown")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.experts_per_rank,
        args.slots,
        args.hidden,
        args.projection,
        args.iterations,
        args.repeats,
    ) <= 0 or args.warmup < 0:
        raise ValueError("invalid dimensions or iteration count")
    if args.backend == "native" and not args.native_root:
        raise ValueError("--native-root is required for the native backend")

    import torch
    import torch.distributed as dist
    import torch_npu

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    ranks = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    coordination_backend = "gloo" if args.backend == "tilexr" else "hccl"
    init_kwargs = {"backend": coordination_backend}
    if coordination_backend == "hccl":
        init_kwargs["device_id"] = torch.device(f"npu:{local_rank}")
    dist.init_process_group(**init_kwargs)
    device = f"npu:{local_rank}"
    plan_host = ring_plan(ranks, args.experts_per_rank, args.slots)
    runner = None
    result = {
        "schema_version": 1,
        "status": "failed",
        "backend": args.backend,
        "rank": rank,
        "world_size": ranks,
        "coordination_backend": coordination_backend,
        "case": {
            "experts_per_rank": args.experts_per_rank,
            "expert_count": ranks * args.experts_per_rank,
            "slots": args.slots,
            "projection_shapes": projection_shapes(args),
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "plan": plan_host,
        "timing_boundary": {
            "clock": "NPU events on the current stream",
            "tilexr": "one fused gate/up/down PrefetchWeight launch",
            "native": "three sequential launch_prefetch calls",
            "excluded": [
                "allocation",
                "MR or SHMEM registration",
                "correctness validation",
                "pre-iteration cross-rank barrier",
                "post-launch status reads",
            ],
        },
    }
    failure = None
    try:
        setup_start = time.perf_counter()
        if args.backend == "tilexr":
            runner = TileXRRunner(torch, args, rank, ranks, device, plan_host)
        else:
            runner = NativeRunner(torch, dist, args, rank, ranks, device, plan_host)
        result["setup_ms"] = (time.perf_counter() - setup_start) * 1000.0
        result["layout"] = runner.layout()
        dist.barrier()
        runner.launch()
        runner.synchronize()
        result["correctness"] = {
            "passed": True,
            "checks": runner.validate(plan_host),
        }
        all_repeats = []
        gathered_repeats = []
        for repeat in range(args.repeats):
            for _ in range(args.warmup):
                time_once(torch, dist, runner)
            local_samples = [
                time_once(torch, dist, runner) for _ in range(args.iterations)
            ]
            gathered = [None for _ in range(ranks)]
            dist.all_gather_object(gathered, local_samples)
            all_repeats.append(local_samples)
            if rank == 0:
                gathered_repeats.append(gathered)
        result["local_samples_us"] = all_repeats
        result["status"] = "passed"
        if rank == 0:
            projection_row_bytes = [
                math.prod(shape) * 2 for shape in projection_shapes(args)
            ]
            bytes_per_rank = args.slots * sum(projection_row_bytes)
            per_repeat = [statistics(samples) for samples in gathered_repeats]
            all_cross_rank = [
                value
                for repeat in per_repeat
                for value in repeat["cross_rank_max_us"]
            ]
            all_valid_cross_rank = [
                value
                for repeat in per_repeat
                for value in repeat["valid_cross_rank_max_us"]
            ]
            aggregate = {
                "cross_rank_max_us": all_cross_rank,
                "valid_cross_rank_max_us": all_valid_cross_rank,
                "invalid_event_sample_count":
                    len(all_cross_rank) - len(all_valid_cross_rank),
                "minimum_valid_event_us_exclusive": MIN_VALID_EVENT_US,
                "p50_us": percentile(all_valid_cross_rank, 0.50),
                "p99_us": percentile(all_valid_cross_rank, 0.99),
                "min_us": min(all_valid_cross_rank),
                "max_us": max(all_valid_cross_rank),
                "mean_us": sum(all_valid_cross_rank) / len(all_valid_cross_rank),
            }
            aggregate["effective_GBps_at_p50"] = (
                bytes_per_rank / aggregate["p50_us"] / 1000.0
            )
            result["bytes"] = {
                "per_projection_row": projection_row_bytes,
                "per_slot": sum(projection_row_bytes),
                "per_rank_per_iteration": bytes_per_rank,
            }
            result["samples_by_rank_us"] = gathered_repeats
            result["statistics_by_repeat"] = per_repeat
            result["statistics"] = aggregate
            result["commits"] = {
                "tilexr": args.tilexr_commit,
                "tilexr_source_snapshot_sha256": args.tilexr_source_sha256,
                "native_moonep": args.native_commit,
            }
            result["environment"] = {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "torch": str(torch.__version__),
                "torch_npu": str(torch_npu.__version__),
                "cann_home": os.environ.get("ASCEND_HOME_PATH"),
                "soc": str(torch.npu.get_device_name()),
                "npu_smi": command_output(("npu-smi", "info")),
            }
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        failure = (exc, exc.__traceback__)
    finally:
        try:
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
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"rank_{rank}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if rank == 0:
            (output_dir / "summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        dist.destroy_process_group()
    if failure is not None:
        raise failure[0].with_traceback(failure[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
