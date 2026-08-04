from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def aggregate_rank_artifacts(
    case_dir: str | Path,
    *,
    world_size: int,
) -> dict[str, object]:
    root = Path(case_dir)
    rank_results = []
    rank_samples = []
    for rank in range(world_size):
        rank_dir = root / f"rank_{rank}"
        with (rank_dir / "result.json").open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        samples = read_jsonl(rank_dir / "samples.jsonl")
        if result.get("status") != "passed":
            raise RuntimeError(f"rank {rank} failed: {result.get('failure_reason', 'unknown')}")
        if int(result.get("rank", rank)) != rank:
            raise ValueError(f"rank_{rank} contains result metadata for another rank")
        rank_results.append(result)
        rank_samples.append(samples)
    counts = {len(samples) for samples in rank_samples}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise ValueError(f"rank sample counts do not match: {sorted(counts)}")

    reference = rank_results[0]
    topology_keys = (
        "global_world_size",
        "node_count",
        "local_world_size",
        "planner_group_size",
        "lane_group_size",
        "physical_device_count",
        "ranks_per_device",
        "oversubscribed",
        "planner_block_dim",
        "planner_block_dim_source",
        "peer_memory_cross_node",
        "cross_node_validated",
    )
    reference_topology = {
        key: reference["topology"].get(key) for key in topology_keys
    }
    for rank, result in enumerate(rank_results[1:], start=1):
        if result["case"] != reference["case"]:
            raise ValueError(f"rank {rank} case metadata differs from rank 0")
        if result["capabilities"] != reference["capabilities"]:
            raise ValueError(f"rank {rank} capability metadata differs from rank 0")
        topology = {key: result["topology"].get(key) for key in topology_keys}
        if topology != reference_topology:
            raise ValueError(f"rank {rank} topology metadata differs from rank 0")

    iteration_count = next(iter(counts))
    metric_names = set(rank_samples[0][0]["timings_us"])
    maxima: list[dict[str, object]] = []
    for iteration in range(iteration_count):
        for rank, samples in enumerate(rank_samples):
            if int(samples[iteration]["iteration"]) != iteration:
                raise ValueError(f"rank {rank} sample iteration ordering is invalid")
            if set(samples[iteration]["timings_us"]) != metric_names:
                raise ValueError("timing metric names differ across ranks")
        maxima.append(
            {
                "iteration": iteration,
                "timings_us": {
                    name: max(
                        float(rank_samples[rank][iteration]["timings_us"][name])
                        for rank in range(world_size)
                    )
                    for name in sorted(metric_names)
                },
            }
        )

    metrics = {
        name: _metric_summary(
            [float(sample["timings_us"][name]) for sample in maxima]
        )
        for name in sorted(metric_names)
    }
    first = reference
    capabilities = first["capabilities"]
    oversubscribed = bool(first["topology"]["oversubscribed"])
    peer_memory_cross_node = bool(
        first["topology"].get("peer_memory_cross_node", False)
    )
    cross_node_validated = bool(
        first["topology"].get("cross_node_validated", False)
    )
    transport_performance_valid = bool(
        capabilities["transport_performance_valid"]
        and not oversubscribed
        and (not peer_memory_cross_node or cross_node_validated)
    )
    if oversubscribed:
        performance_scope = "oversubscribed_functional_only"
    elif peer_memory_cross_node and not cross_node_validated:
        performance_scope = "cross_node_functional_unvalidated"
    elif transport_performance_valid:
        performance_scope = "transport"
    else:
        performance_scope = "stub_contract_only"
    summary = {
        "schema_version": 1,
        "status": "passed",
        "case": first["case"],
        "logical_world_size": world_size,
        "physical_device_count": first["topology"]["physical_device_count"],
        "ranks_per_device": first["topology"]["ranks_per_device"],
        "oversubscribed": oversubscribed,
        "planner_block_dim": first["topology"].get("planner_block_dim"),
        "planner_block_dim_source": first["topology"].get("planner_block_dim_source"),
        "capabilities": capabilities,
        "transport_performance_valid": transport_performance_valid,
        "performance_scope": performance_scope,
        "validation": {
            "passed": all(bool(item["validation"]["passed"]) for item in rank_results),
            "mode": (
                "full" if transport_performance_valid else performance_scope
            ),
        },
        "cross_rank_max_samples": maxima,
        "metrics_us": metrics,
    }
    tokens_per_rank = int(first["case"]["tokens_per_rank"])
    throughput = []
    for sample in maxima:
        duration = float(sample["timings_us"]["end_to_end"])
        throughput.append(
            tokens_per_rank * world_size * 1_000_000.0 / duration
            if duration > 0.0
            else 0.0
        )
    summary["tokens_per_second"] = _metric_summary(throughput)
    write_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "min", "max", "mean", "p50", "p90", "p99"),
        )
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"metric": name, **values})
    return summary
