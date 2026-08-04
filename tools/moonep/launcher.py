from __future__ import annotations

import argparse
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


def rank_to_device(rank: int, physical_device_count: int) -> int:
    if rank < 0 or physical_device_count <= 0:
        raise ValueError("rank must be non-negative and physical_device_count positive")
    return rank % physical_device_count


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
        "dtype": "--dtype",
        "seed": "--seed",
        "warmup": "--warmup",
        "iterations": "--iterations",
    }
    for name, flag in names.items():
        value = getattr(args, name)
        if value is not None:
            command.extend((flag, str(value)))
    if args.correctness is True:
        command.append("--correctness")
    elif args.correctness is False:
        command.append("--no-correctness")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch TileXR MoonEP logical ranks")
    build_case_parser(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--install-prefix", default=None)
    parser.add_argument("--physical-device-count", type=int, default=8)
    parser.add_argument("--ranks-per-device", type=int, choices=(1, 2), default=1)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--planner-block-dim", type=int, default=None)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wait_iterations <= 0 or args.timeout_sec <= 0:
        raise ValueError("wait_iterations and timeout_sec must be positive")
    topology = resolve_topology(
        physical_device_count=args.physical_device_count,
        ranks_per_device=args.ranks_per_device,
        world_size=args.world_size,
        planner_block_dim=args.planner_block_dim,
        environment=os.environ,
    )
    world_size = int(topology["logical_world_size"])
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
    base_env["TILEXR_RANKS_PER_DEVICE"] = str(topology["ranks_per_device"])
    base_env["TILEXR_OVERSUBSCRIBED"] = "1" if topology["oversubscribed"] else "0"
    base_env["TILEXR_MOONEP_PLANNER_BLOCK_DIM_SOURCE"] = str(
        topology["planner_block_dim_source"]
    )
    base_env[PLANNER_BLOCK_DIM_ENV] = str(topology["planner_block_dim"])

    command = _process_command(args)
    processes: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for rank in range(world_size):
            env = base_env.copy()
            device = rank_to_device(rank, args.physical_device_count)
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

    cases = select_cases(load_cases(args.cases), args.case_ids)
    for case in cases:
        case = apply_overrides(case, args)
        aggregate_rank_artifacts(output_dir / case.case_id, world_size=world_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
