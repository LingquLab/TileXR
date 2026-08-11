from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import build_case_parser, validate_iteration_overrides
from .launcher import (
    DISPATCH_AIV_CORE_COUNT_ENV,
    PLANNER_BLOCK_DIM_ENV,
    _process_command,
    rank_to_device,
)
from .report import write_json


def resolve_node_topology(*, node_count: int, node_rank: int,
    local_world_size: int, physical_device_count: int,
    planner_block_dim: int, dispatch_aiv_core_count: int = 64,
    mode: str = "benchmark") -> dict[str, int]:
    if mode not in ("benchmark", "reference", "correctness"):
        raise ValueError("mode must be benchmark, reference, or correctness")
    if node_count <= 0 or node_rank < 0 or node_rank >= node_count:
        raise ValueError("node_rank must be in [0, node_count)")
    if local_world_size <= 0 or physical_device_count <= 0:
        raise ValueError("local_world_size and physical_device_count must be positive")
    if local_world_size > physical_device_count:
        raise ValueError("distributed runs require at most one rank per device")
    world_size = node_count * local_world_size
    if planner_block_dim < 1 or planner_block_dim > 64:
        raise ValueError("planner_block_dim must satisfy 1 <= blockDim <= 64")
    if dispatch_aiv_core_count < 1 or dispatch_aiv_core_count > 64:
        raise ValueError("dispatch_aiv_core_count must satisfy 1 <= coreCount <= 64")
    return {
        "node_count": node_count,
        "node_rank": node_rank,
        "local_world_size": local_world_size,
        "physical_device_count": physical_device_count,
        "world_size": world_size,
        "first_global_rank": node_rank * local_world_size,
        "planner_block_dim": planner_block_dim,
        "dispatch_aiv_core_count": dispatch_aiv_core_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch one TileXR MoonEP multi-node worker set")
    build_case_parser(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--install-prefix", required=True)
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--local-world-size", type=int, default=8)
    parser.add_argument("--physical-device-count", type=int, default=8)
    parser.add_argument("--planner-block-dim", type=int, default=64)
    parser.add_argument("--dispatch-aiv-core-count", type=int, default=64)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait-iterations", type=int, default=1_000_000)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument(
        "--mode", choices=("benchmark", "reference", "correctness"),
        default="benchmark")
    parser.add_argument("--candidate-backend", default=None, metavar="MODULE:FACTORY")
    parser.add_argument("--dump-stage-tensors", action="store_true")
    parser.add_argument("--tensor-preview-elements", type=int, default=8)
    parser.add_argument("--benchmark-kind", choices=("flow", "dispatch_hot_loop"),
        default="dispatch_hot_loop")
    parser.add_argument("--dispatch-modes", nargs="+",
        choices=("hidden", "weight", "pair"), default=("hidden",))
    return parser


def _require_shared_environment(
    environment: dict[str, str], *, mode: str = "benchmark"
) -> None:
    native_required = (
        "TILEXR_COMM_ID",
        "TILEXR_MOONEP_BARRIER_ADDR",
        "TILEXR_MOONEP_LAUNCH_ID",
        "TILEXR_MOONEP_LAUNCH_SECRET",
    )
    reference_required = (
        "MASTER_ADDR", "MASTER_PORT", "TILEXR_MOONEP_LAUNCH_ID")
    if mode == "benchmark":
        required = native_required
    elif mode == "reference":
        required = reference_required
    elif mode == "correctness":
        required = native_required + reference_required
    else:
        raise ValueError("mode must be benchmark, reference, or correctness")
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ValueError("missing shared multi-node environment: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_iteration_overrides(args.warmup, args.iterations)
    if args.wait_iterations <= 0 or args.timeout_sec <= 0:
        raise ValueError("wait_iterations and timeout_sec must be positive")
    if args.tensor_preview_elements <= 0:
        raise ValueError("tensor_preview_elements must be positive")
    if args.mode in ("reference", "correctness"):
        args.benchmark_kind = "flow"
    topology = resolve_node_topology(
        node_count=args.node_count,
        node_rank=args.node_rank,
        local_world_size=args.local_world_size,
        physical_device_count=args.physical_device_count,
        planner_block_dim=args.planner_block_dim,
        dispatch_aiv_core_count=args.dispatch_aiv_core_count,
        mode=args.mode,
    )
    world_size = topology["world_size"]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]
    integration = root / "integrations" / "moonep_torch"
    base_env = os.environ.copy()
    _require_shared_environment(base_env, mode=args.mode)
    selected_case_ids = [
        value.strip() for value in (args.case_ids or "").split(",") if value.strip()
    ]
    pythonpath = [str(root), str(integration)]
    if base_env.get("PYTHONPATH"):
        pythonpath.append(base_env["PYTHONPATH"])
    base_env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    base_env.update({
        "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(topology["local_world_size"]),
        "NODE_RANK": str(topology["node_rank"]),
        "NODE_COUNT": str(topology["node_count"]),
        "TILEXR_MOONEP_MANAGED_LAUNCH": "1",
        "TILEXR_PHYSICAL_DEVICE_COUNT": str(topology["physical_device_count"]),
        "TILEXR_RANKS_PER_DEVICE": "1",
        "TILEXR_OVERSUBSCRIBED": "0",
        "TILEXR_MOONEP_PLANNER_BLOCK_DIM_SOURCE": "distributed_cli",
        PLANNER_BLOCK_DIM_ENV: str(topology["planner_block_dim"]),
        "TILEXR_MOONEP_DISPATCH_AIV_CORE_COUNT_SOURCE": "distributed_cli",
        DISPATCH_AIV_CORE_COUNT_ENV: str(topology["dispatch_aiv_core_count"]),
    })
    if args.mode in ("benchmark", "correctness"):
        base_env.setdefault(
            "TILEXR_UDMA_QP_ROUTE_SPEC", "port_count:6,port_count:2"
        )
    command = _process_command(args)
    write_json(output_dir / f"node_{args.node_rank}_metadata.json", {
        "schema_version": 1,
        "topology": topology,
        "mode": args.mode,
        "comm_id": base_env.get("TILEXR_COMM_ID"),
        "barrier_addr": base_env.get("TILEXR_MOONEP_BARRIER_ADDR"),
        "master_addr": base_env.get("MASTER_ADDR"),
        "master_port": base_env.get("MASTER_PORT"),
        "launch_id": base_env["TILEXR_MOONEP_LAUNCH_ID"],
        "case_ids": selected_case_ids,
        "command": command,
    })

    processes: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for local_rank in range(topology["local_world_size"]):
            global_rank = topology["first_global_rank"] + local_rank
            env = base_env.copy()
            env["RANK"] = str(global_rank)
            env["LOCAL_RANK"] = str(rank_to_device(
                local_rank, topology["physical_device_count"]))
            env["TILEXR_PLANNER_GROUP_RANK"] = str(global_rank)
            env["TILEXR_PLANNER_GROUP_SIZE"] = str(world_size)
            log_path = output_dir / f"launcher_rank_{global_rank}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=str(root), env=env,
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
            processes.append((global_rank, process, log_handle))

        deadline = time.monotonic() + args.timeout_sec
        pending = {rank for rank, _, _ in processes}
        while pending:
            failures = []
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
                raise TimeoutError(
                    f"MoonEP node {args.node_rank} exceeded {args.timeout_sec}s")
            time.sleep(0.05)
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, log_handle in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log_handle.close()

    write_json(output_dir / f"node_{args.node_rank}_complete.json", {
        "schema_version": 1,
        "status": "passed",
        "mode": args.mode,
        "launch_id": base_env["TILEXR_MOONEP_LAUNCH_ID"],
        "case_ids": selected_case_ids,
        "topology": topology,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
