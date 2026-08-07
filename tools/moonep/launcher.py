from __future__ import annotations

import argparse
import copy
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from .config import apply_overrides, build_case_parser, load_cases, select_cases
from .rendezvous import offset_host_port
from .report import aggregate_rank_artifacts, write_json


PLANNER_BLOCK_DIM_ENV = "TILEXR_MOONEP_PLANNER_BLOCK_DIM"
UDMA_QP_ROUTE_SPEC_ENV = "TILEXR_UDMA_QP_ROUTE_SPEC"


def resolve_topology(
    *,
    physical_device_count: int,
    ranks_per_device: int,
    world_size: int | None,
    planner_block_dim: int | None,
    environment: Mapping[str, str],
) -> dict[str, object]:
    if physical_device_count <= 0:
        raise ValueError("physical_device_count must be positive")
    if ranks_per_device not in (1, 2):
        raise ValueError("ranks_per_device must be 1 or 2")
    capacity = physical_device_count * ranks_per_device
    logical_world_size = capacity if world_size is None else int(world_size)
    if logical_world_size <= 0 or logical_world_size > capacity:
        raise ValueError(f"world_size must be in [1, {capacity}], got {logical_world_size}")

    source = "default_native"
    effective = 64
    if planner_block_dim is not None:
        effective = int(planner_block_dim)
        source = "cli"
    elif environment.get(PLANNER_BLOCK_DIM_ENV):
        effective = int(environment[PLANNER_BLOCK_DIM_ENV])
        source = "environment"
    elif ranks_per_device == 2:
        effective = 32
        source = "default_oversubscribed"
    if effective < logical_world_size or effective > 64:
        raise ValueError(
            f"effective {PLANNER_BLOCK_DIM_ENV}={effective} must satisfy "
            f"world_size={logical_world_size} <= blockDim <= 64"
        )
    return {
        "logical_world_size": logical_world_size,
        "physical_device_count": physical_device_count,
        "ranks_per_device": ranks_per_device,
        "oversubscribed": logical_world_size > physical_device_count,
        "planner_block_dim": effective,
        "planner_block_dim_source": source,
    }


def rank_to_device(
    rank: int,
    physical_device_count: int,
    device_ids: tuple[int, ...] | None = None,
) -> int:
    if rank < 0 or physical_device_count <= 0:
        raise ValueError("rank must be non-negative and physical_device_count positive")
    if device_ids is not None:
        if len(device_ids) != physical_device_count:
            raise ValueError("device id count must match physical_device_count")
        return device_ids[rank % physical_device_count]
    return rank % physical_device_count


def _parse_device_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    try:
        device_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("device ids must be comma-separated non-negative integers") from exc
    if not device_ids or any(item < 0 for item in device_ids):
        raise ValueError("device ids must be comma-separated non-negative integers")
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("device ids must be unique")
    return device_ids


def _unused_local_comm_id() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"127.0.0.1:{port}"


def _append_case_overrides(command: list[str], args: argparse.Namespace) -> None:
    names = {
        "tokens_per_rank": "--tokens-per-rank",
        "topk": "--topk",
        "expert_count": "--expert-count",
        "hidden_size": "--hidden-size",
        "expert_shape": "--expert-shape",
        "dtype": "--dtype",
        "route_pattern": "--route-pattern",
        "seed": "--seed",
        "warmup": "--warmup",
        "iterations": "--iterations",
    }
    for name, flag in names.items():
        value = getattr(args, name)
        if value is not None:
            if name == "expert_shape":
                value = "x".join(str(dimension) for dimension in value)
            command.extend((flag, str(value)))
    if args.correctness is True:
        command.append("--correctness")
    elif args.correctness is False:
        command.append("--no-correctness")


def _prefetch_route_spec(worker_count: int) -> str:
    if worker_count not in (1, 2, 4, 8):
        raise ValueError("prefetch worker count must be chosen from 1,2,4,8")
    external_routes = {
        1: ("port_count:6",),
        2: ("topology", "port_count:6"),
        4: ("port_count:6", "port_count:6", "port_count:6", "port_count:2"),
        8: (
            "port_count:6",
            "port_count:6",
            "port_count:6",
            "port_count:2",
            "port_count:6",
            "port_count:6",
            "port_count:6",
            "port_count:2",
        ),
    }
    return ",".join(external_routes[worker_count])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch TileXR MoonEP logical ranks")
    build_case_parser(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--install-prefix", default=None)
    parser.add_argument("--physical-device-count", type=int, default=8)
    parser.add_argument(
        "--device-ids",
        default=None,
        help="ordered comma-separated physical device ids; overrides the device count",
    )
    parser.add_argument("--ranks-per-device", type=int, choices=(1, 2), default=1)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--planner-block-dim", type=int, default=None)
    parser.add_argument(
        "--prefetch-workers",
        default=None,
        help="comma-separated UDMA QP/AIV candidates chosen from 1,2,4,8",
    )
    parser.add_argument("--comm-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait-iterations", type=int, default=1_000_000)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    return parser


def _process_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "-m",
        "tools.moonep.benchmark",
        "--cases",
        str(Path(args.cases).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--wait-iterations",
        str(args.wait_iterations),
    ]
    if args.case_ids:
        command.extend(("--case-ids", args.case_ids))
    if args.install_prefix:
        command.extend(("--install-prefix", str(Path(args.install_prefix).resolve())))
    _append_case_overrides(command, args)
    return command


def _run_once(args: argparse.Namespace, prefetch_workers: int | None):
    if args.wait_iterations <= 0 or args.timeout_sec <= 0:
        raise ValueError("wait_iterations and timeout_sec must be positive")
    device_ids = _parse_device_ids(args.device_ids)
    physical_device_count = (
        len(device_ids) if device_ids is not None else args.physical_device_count
    )
    topology = resolve_topology(
        physical_device_count=physical_device_count,
        ranks_per_device=args.ranks_per_device,
        world_size=args.world_size,
        planner_block_dim=args.planner_block_dim,
        environment=os.environ,
    )
    topology["device_ids"] = (
        list(device_ids) if device_ids is not None else list(range(physical_device_count))
    )
    world_size = int(topology["logical_world_size"])
    cases = [
        apply_overrides(case, args)
        for case in select_cases(load_cases(args.cases), args.case_ids)
    ]
    if prefetch_workers is not None:
        too_small = [
            case.case_id
            for case in cases
            if case.expert_count % world_size != 0
            or case.expert_count // world_size < prefetch_workers
        ]
        if too_small:
            raise ValueError(
                f"prefetch worker count {prefetch_workers} exceeds B=E/R for cases: "
                + ", ".join(too_small)
            )
    comm_id = args.comm_id or _unused_local_comm_id()
    barrier_addr = os.environ.get("TILEXR_MOONEP_BARRIER_ADDR") or offset_host_port(
        comm_id
    )
    barrier_timeout_sec = args.timeout_sec + 60.0
    launch_id = secrets.token_hex(16)
    launch_secret = secrets.token_hex(32)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "launcher": topology,
        "comm_id": comm_id,
        "barrier_addr": barrier_addr,
        "barrier_timeout_sec": barrier_timeout_sec,
        "launch_id": launch_id,
        "command": _process_command(args),
        "prefetch_workers": prefetch_workers,
        "udma_qp_route_spec": (
            _prefetch_route_spec(prefetch_workers)
            if prefetch_workers is not None
            else os.environ.get(UDMA_QP_ROUTE_SPEC_ENV)
        ),
        "case_comm_ids": {
            case.case_id: offset_host_port(comm_id, offset=index)
            for index, case in enumerate(cases)
        },
        "case_barrier_addrs": {
            case.case_id: offset_host_port(barrier_addr, offset=index)
            for index, case in enumerate(cases)
        },
    }
    write_json(output_dir / "launcher_metadata.json", metadata)

    root = Path(__file__).resolve().parents[2]
    integration = root / "integrations" / "moonep_torch"
    base_env = os.environ.copy()
    pythonpath = [str(root), str(integration)]
    if base_env.get("PYTHONPATH"):
        pythonpath.append(base_env["PYTHONPATH"])
    base_env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    base_env["WORLD_SIZE"] = str(world_size)
    base_env["LOCAL_WORLD_SIZE"] = str(world_size)
    base_env["NODE_RANK"] = "0"
    base_env["NODE_COUNT"] = "1"
    base_env["TILEXR_COMM_ID"] = comm_id
    base_env["TILEXR_MOONEP_BARRIER_ADDR"] = barrier_addr
    base_env["TILEXR_MOONEP_BARRIER_TIMEOUT_SEC"] = str(barrier_timeout_sec)
    base_env["TILEXR_MOONEP_LAUNCH_ID"] = launch_id
    base_env["TILEXR_MOONEP_LAUNCH_SECRET"] = launch_secret
    base_env["TILEXR_MOONEP_MANAGED_LAUNCH"] = "1"
    base_env["TILEXR_PHYSICAL_DEVICE_COUNT"] = str(topology["physical_device_count"])
    base_env["TILEXR_DEVICE_IDS"] = ",".join(
        str(item) for item in topology["device_ids"]
    )
    base_env["TILEXR_RANKS_PER_DEVICE"] = str(topology["ranks_per_device"])
    base_env["TILEXR_OVERSUBSCRIBED"] = "1" if topology["oversubscribed"] else "0"
    base_env["TILEXR_MOONEP_PLANNER_BLOCK_DIM_SOURCE"] = str(
        topology["planner_block_dim_source"]
    )
    base_env[PLANNER_BLOCK_DIM_ENV] = str(topology["planner_block_dim"])
    if prefetch_workers is not None:
        base_env[UDMA_QP_ROUTE_SPEC_ENV] = _prefetch_route_spec(prefetch_workers)
        base_env["TILEXR_MOONEP_PREFETCH_BLOCK_DIM"] = str(prefetch_workers)

    command = _process_command(args)
    processes: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for rank in range(world_size):
            env = base_env.copy()
            device = rank_to_device(rank, physical_device_count, device_ids)
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(device)
            env["TILEXR_PLANNER_GROUP_RANK"] = str(rank)
            env["TILEXR_PLANNER_GROUP_SIZE"] = str(world_size)
            log_path = output_dir / f"launcher_rank_{rank}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((rank, process, log_handle))
        deadline = time.monotonic() + args.timeout_sec
        pending = {rank for rank, _, _ in processes}
        failures = []
        abort_prefix = f".tilexr_abort_{launch_id}_"
        while pending:
            abort_files = list(output_dir.glob(f"{abort_prefix}*"))
            if abort_files:
                raise RuntimeError(
                    "MoonEP rank could not prove local NPU quiescence: "
                    + ", ".join(path.name for path in abort_files)
                )
            for rank, process, _ in processes:
                if rank not in pending:
                    continue
                ret = process.poll()
                if ret is None:
                    continue
                pending.remove(rank)
                if ret != 0:
                    failures.append((rank, ret))
            if failures:
                raise RuntimeError(f"MoonEP ranks failed: {failures}")
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MoonEP launcher exceeded {args.timeout_sec}s")
            time.sleep(0.05)
    finally:
        for _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, log_handle in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log_handle.close()

    summaries = {}
    for case in cases:
        summaries[case.case_id] = aggregate_rank_artifacts(
            output_dir / case.case_id, world_size=world_size
        )
    return summaries


def _parse_prefetch_workers(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    workers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not workers or any(item not in (1, 2, 4, 8) for item in workers):
        raise ValueError("prefetch worker candidates must be chosen from 1,2,4,8")
    if len(workers) != len(set(workers)):
        raise ValueError("prefetch worker candidates must be unique")
    return workers


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = _parse_prefetch_workers(args.prefetch_workers)
    if len(candidates) <= 1:
        _run_once(args, candidates[0] if candidates else None)
        return 0

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate_index, workers in enumerate(candidates):
        candidate_args = copy.copy(args)
        candidate_args.output_dir = str(output_root / f"prefetch_workers_{workers}")
        if args.comm_id:
            candidate_args.comm_id = offset_host_port(
                args.comm_id, offset=candidate_index * 257
            )
        summaries = _run_once(candidate_args, workers)
        p50_values = [
            float(summary["metrics_us"]["prefetch_weight"]["p50"])
            for summary in summaries.values()
            if summary["prefetch_weight_performance_valid"]
        ]
        if not p50_values:
            raise RuntimeError(
                f"prefetch worker {workers} produced no valid data-plane samples"
            )
        results.append(
            {
                "workers": workers,
                "score_prefetch_p50_us": sum(p50_values) / len(p50_values),
                "cases": summaries,
            }
        )
    best = min(results, key=lambda item: item["score_prefetch_p50_us"])
    write_json(
        output_root / "prefetch_sweep_summary.json",
        {
            "schema_version": 1,
            "metric": "mean_case_cross_rank_prefetch_weight_p50_us",
            "recommended_worker_count": best["workers"],
            "candidates": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
