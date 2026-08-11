from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping

from .contracts import CORRECTNESS_STAGES


FLOW_STAGE_ORDER = (
    "planning",
    "dispatch",
    "prefetch_weight",
    "expert",
    "combine",
    "reduce_grad",
)
COMMUNICATION_STAGE_ORDER = (
    "dispatch",
    "prefetch_weight",
    "combine",
    "reduce_grad",
)
FLOW_STAGE_COMPONENTS = {
    "planning": ("planning",),
    "dispatch": ("dispatch_forward", "dispatch_backward"),
    "prefetch_weight": ("prefetch_weight",),
    "expert": ("expert_forward", "expert_backward"),
    "combine": ("combine_forward", "combine_backward"),
    "reduce_grad": ("reduce_grad",),
}
FLOW_STAGE_LABELS = {
    "planning": "Planning",
    "dispatch": "Dispatch",
    "prefetch_weight": "PrefetchWeight",
    "expert": "Expert",
    "combine": "Combine",
    "reduce_grad": "ReduceGrad",
}


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
        "p95": percentile(values, 0.95),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def _flow_stage_duration(timings: Mapping[str, object], stage: str) -> float:
    return sum(float(timings[name]) for name in FLOW_STAGE_COMPONENTS[stage])


def _aggregate_flow_stages(
    rank_samples: list[list[dict[str, object]]],
    *,
    world_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    required_metrics = {
        metric for components in FLOW_STAGE_COMPONENTS.values() for metric in components
    }
    has_stage_data = any(
        "planning" in sample["timings_us"] or "algorithm_bytes" in sample
        for samples in rank_samples
        for sample in samples
    )
    if not has_stage_data:
        return None

    communication_stages = set(COMMUNICATION_STAGE_ORDER)
    cross_rank_samples: list[dict[str, object]] = []
    for iteration in range(len(rank_samples[0])):
        per_rank = []
        for rank in range(world_size):
            sample = rank_samples[rank][iteration]
            timings = sample["timings_us"]
            missing_metrics = required_metrics - set(timings)
            if missing_metrics:
                raise ValueError(
                    f"rank {rank} iteration {iteration} is missing flow timing metrics: "
                    f"{sorted(missing_metrics)}"
                )
            algorithm_bytes = sample.get("algorithm_bytes")
            if not isinstance(algorithm_bytes, dict):
                raise ValueError(
                    f"rank {rank} iteration {iteration} is missing algorithm bytes"
                )
            missing_bytes = communication_stages - set(algorithm_bytes)
            if missing_bytes:
                raise ValueError(
                    f"rank {rank} iteration {iteration} is missing algorithm bytes for: "
                    f"{sorted(missing_bytes)}"
                )
            normalized_bytes = {}
            for stage in COMMUNICATION_STAGE_ORDER:
                value = algorithm_bytes[stage]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"rank {rank} iteration {iteration} has invalid {stage} "
                        "algorithm bytes"
                    )
                normalized_bytes[stage] = value
            per_rank.append(
                {
                    "durations": {
                        stage: _flow_stage_duration(timings, stage)
                        for stage in FLOW_STAGE_ORDER
                    },
                    "algorithm_bytes": normalized_bytes,
                }
            )

        stages = {}
        for stage in FLOW_STAGE_ORDER:
            critical_rank = max(
                range(world_size),
                key=lambda rank: per_rank[rank]["durations"][stage],
            )
            duration = float(per_rank[critical_rank]["durations"][stage])
            algorithm_bytes = (
                int(per_rank[critical_rank]["algorithm_bytes"][stage])
                if stage in communication_stages
                else None
            )
            bandwidth = (
                algorithm_bytes / duration / 1000.0
                if algorithm_bytes is not None and duration > 0.0
                else None
            )
            stages[stage] = {
                "critical_rank": critical_rank,
                "duration_us": duration,
                "algorithm_bytes": algorithm_bytes,
                "algorithm_bandwidth_GBps": bandwidth,
            }
        cross_rank_samples.append({"iteration": iteration, "stages": stages})

    stage_performance = {}
    for stage in FLOW_STAGE_ORDER:
        samples = [item["stages"][stage] for item in cross_rank_samples]
        byte_values = [
            float(sample["algorithm_bytes"])
            for sample in samples
            if sample["algorithm_bytes"] is not None
        ]
        bandwidth_values = [
            float(sample["algorithm_bandwidth_GBps"])
            for sample in samples
            if sample["algorithm_bandwidth_GBps"] is not None
        ]
        stage_performance[stage] = {
            "timings_us": _metric_summary(
                [float(sample["duration_us"]) for sample in samples]
            ),
            "algorithm_bytes": _metric_summary(byte_values) if byte_values else None,
            "algorithm_bandwidth_GBps": (
                _metric_summary(bandwidth_values)
                if len(bandwidth_values) == len(samples) and bandwidth_values
                else None
            ),
        }
    return cross_rank_samples, stage_performance


def _format_number(value: float, *, precision: int) -> str:
    return f"{value:.{precision}f}"


def _format_metadata_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _legacy_stage_execution(summary: Mapping[str, object]) -> dict[str, object]:
    capabilities = summary.get("capabilities")
    implementations = (
        capabilities.get("implementations", {})
        if isinstance(capabilities, dict)
        else {}
    )
    environment = summary.get("environment")
    torch_npu_version = (
        environment.get("torch_npu", "unknown")
        if isinstance(environment, dict)
        else "unknown"
    )
    metadata = {}
    for stage in FLOW_STAGE_ORDER:
        if stage == "expert":
            metadata[stage] = {
                "native": False,
                "implementation": "torch_npu",
                "kernel_version": (
                    f"torch_npu {torch_npu_version} (legacy artifact)"
                ),
            }
            continue
        implementation = str(implementations.get(stage, "unknown"))
        metadata[stage] = {
            "native": implementation == "native",
            "implementation": implementation,
            "kernel_version": "Unknown (legacy artifact)",
        }
    return metadata


def _stage_execution(summary: Mapping[str, object]) -> Mapping[str, object]:
    metadata = summary.get("stage_execution")
    if metadata is None:
        return _legacy_stage_execution(summary)
    if not isinstance(metadata, dict) or set(metadata) != set(FLOW_STAGE_ORDER):
        raise ValueError("summary has invalid stage execution metadata")
    for stage in FLOW_STAGE_ORDER:
        values = metadata[stage]
        if (
            not isinstance(values, dict)
            or not isinstance(values.get("native"), bool)
            or not isinstance(values.get("kernel_version"), str)
        ):
            raise ValueError(f"summary has invalid {stage} execution metadata")
    return metadata


def _format_benchmark_inputs(summary: Mapping[str, object]) -> str:
    case = summary.get("case")
    if not isinstance(case, dict):
        raise ValueError("summary does not contain benchmark case metadata")
    environment = summary.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    benchmark_config = summary.get("benchmark_config")
    if not isinstance(benchmark_config, dict):
        benchmark_config = {}

    def case_value(name: str) -> str:
        return _format_metadata_value(case.get(name))

    return "\n".join(
        (
            "MoonEP benchmark inputs",
            f"Case: {case_value('case_id')}  "
            f"mode={_format_metadata_value(summary.get('mode'))}",
            "Dimensions: "
            f"S={case_value('tokens_per_rank')}  K={case_value('topk')}  "
            f"E={case_value('expert_count')}  H={case_value('hidden_size')}  "
            f"Hf={case_value('intermediate_size')}  "
            f"B={case_value('prefetch_slots')}  P={case_value('token_padding')}",
            "Input: "
            f"dtype={case_value('dtype')}  seed={case_value('seed')}  "
            f"routing={case_value('routing_pattern')}  "
            f"route_distribution={case_value('route_distribution')}  "
            f"correctness={case_value('correctness')}",
            "Run: "
            f"warmup={case_value('warmup')}  iterations={case_value('iterations')}  "
            f"wait_iterations="
            f"{_format_metadata_value(benchmark_config.get('wait_iterations'))}  "
            f"logical_ranks={_format_metadata_value(summary.get('logical_world_size'))}  "
            f"nodes={_format_metadata_value(summary.get('node_count'))}",
            "Topology: "
            f"physical_devices={_format_metadata_value(summary.get('physical_device_count'))}  "
            f"ranks_per_device={_format_metadata_value(summary.get('ranks_per_device'))}  "
            f"oversubscribed={_format_metadata_value(summary.get('oversubscribed'))}  "
            f"visible_devices={_format_metadata_value(summary.get('visible_devices'))}",
            "Kernel config: "
            f"planner_block_dim={_format_metadata_value(summary.get('planner_block_dim'))}  "
            f"dispatch_aiv_core_count="
            f"{_format_metadata_value(summary.get('dispatch_aiv_core_count'))}  "
            f"udma_qp_route_spec="
            f"{_format_metadata_value(summary.get('udma_qp_route_spec'))}",
            "Build: "
            f"source_git_sha={_format_metadata_value(environment.get('git_sha'))}  "
            f"source_dirty={_format_metadata_value(environment.get('git_dirty'))}  "
            f"torch_npu={_format_metadata_value(environment.get('torch_npu'))}",
        )
    )


def format_stage_performance(summary: Mapping[str, object]) -> str:
    performance = summary.get("stage_performance")
    if not isinstance(performance, dict):
        raise ValueError("summary does not contain six-stage performance data")
    if set(performance) != set(FLOW_STAGE_ORDER):
        raise ValueError("summary has invalid six-stage performance keys")
    execution = _stage_execution(summary)

    headers = (
        "Stage",
        "Native",
        "Kernel/API version",
        "Mean us",
        "P50 us",
        "P95 us",
        "Algorithm bytes",
        "AlgBW GB/s",
    )
    rows = []
    for stage in FLOW_STAGE_ORDER:
        values = performance[stage]
        timings = values["timings_us"]
        bytes_summary = values["algorithm_bytes"]
        bandwidth_summary = values["algorithm_bandwidth_GBps"]
        timing_cells = (
            (
                _format_number(float(timings["mean"]), precision=3),
                _format_number(float(timings["p50"]), precision=3),
                _format_number(float(timings["p95"]), precision=3),
            )
            if timings is not None
            else ("N/A", "N/A", "N/A")
        )
        rows.append(
            (
                FLOW_STAGE_LABELS[stage],
                "Yes" if execution[stage]["native"] else "No",
                str(execution[stage]["kernel_version"]),
                *timing_cells,
                (
                    _format_number(float(bytes_summary["mean"]), precision=0)
                    if bytes_summary is not None
                    else "N/A"
                ),
                (
                    _format_number(float(bandwidth_summary["mean"]), precision=6)
                    if bandwidth_summary is not None
                    else "N/A"
                ),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[index]) if index < 3 else value.rjust(widths[index])
            for index, value in enumerate(row)
        )

    return "\n".join(
        (
            _format_benchmark_inputs(summary),
            "",
            "MoonEP six-stage performance (global critical rank per iteration)",
            format_row(headers),
            format_row(tuple("-" * width for width in widths)),
            *(format_row(row) for row in rows),
        )
    )


def _write_stage_csv(
    path: Path,
    stage_performance: Mapping[str, object],
    stage_execution: Mapping[str, object] | None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "native",
                "kernel_version",
                "mean_us",
                "p50_us",
                "p95_us",
                "algorithm_bytes",
                "algorithm_bandwidth_GBps",
            ),
        )
        writer.writeheader()
        execution = stage_execution or {}
        for stage in FLOW_STAGE_ORDER:
            values = stage_performance[stage]
            timings = values["timings_us"]
            byte_values = values["algorithm_bytes"]
            bandwidth_values = values["algorithm_bandwidth_GBps"]
            writer.writerow(
                {
                    "stage": FLOW_STAGE_LABELS[stage],
                    "native": (
                        execution.get(stage, {}).get("native", "")
                        if isinstance(execution.get(stage), dict)
                        else ""
                    ),
                    "kernel_version": (
                        execution.get(stage, {}).get("kernel_version", "")
                        if isinstance(execution.get(stage), dict)
                        else ""
                    ),
                    "mean_us": "" if timings is None else timings["mean"],
                    "p50_us": "" if timings is None else timings["p50"],
                    "p95_us": "" if timings is None else timings["p95"],
                    "algorithm_bytes": (
                        "" if byte_values is None else byte_values["mean"]
                    ),
                    "algorithm_bandwidth_GBps": (
                        "" if bandwidth_values is None else bandwidth_values["mean"]
                    ),
                }
            )


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
    if len(counts) != 1 or not counts:
        raise ValueError(f"rank sample counts do not match: {sorted(counts)}")

    reference = rank_results[0]
    reference_stage_execution = reference.get("stage_execution")
    reference_environment = reference.get("environment")
    reference_benchmark_config = reference.get("benchmark_config")
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
        "dispatch_aiv_core_count",
        "dispatch_aiv_core_count_source",
        "udma_qp_route_spec",
        "peer_memory_cross_node",
        "cross_node_validated",
    )
    reference_topology = {
        key: reference["topology"].get(key) for key in topology_keys
    }
    for rank, result in enumerate(rank_results[1:], start=1):
        if result.get("benchmark_kind", "flow") != reference.get(
            "benchmark_kind", "flow"
        ):
            raise ValueError(f"rank {rank} benchmark kind differs from rank 0")
        if result.get("dispatch_modes") != reference.get("dispatch_modes"):
            raise ValueError(f"rank {rank} Dispatch modes differ from rank 0")
        if result["case"] != reference["case"]:
            raise ValueError(f"rank {rank} case metadata differs from rank 0")
        if result["capabilities"] != reference["capabilities"]:
            raise ValueError(f"rank {rank} capability metadata differs from rank 0")
        if result.get("stage_execution") != reference_stage_execution:
            raise ValueError(f"rank {rank} stage execution metadata differs from rank 0")
        if result.get("benchmark_config") != reference_benchmark_config:
            raise ValueError(f"rank {rank} benchmark config differs from rank 0")
        for name in ("git_sha", "git_dirty", "torch_npu"):
            reference_value = (
                reference_environment.get(name)
                if isinstance(reference_environment, dict)
                else None
            )
            result_environment = result.get("environment")
            result_value = (
                result_environment.get(name)
                if isinstance(result_environment, dict)
                else None
            )
            if result_value != reference_value:
                raise ValueError(
                    f"rank {rank} environment {name} differs from rank 0"
                )
        topology = {key: result["topology"].get(key) for key in topology_keys}
        if topology != reference_topology:
            raise ValueError(f"rank {rank} topology metadata differs from rank 0")

    iteration_count = next(iter(counts))
    if iteration_count == 0:
        case = reference.get("case")
        if not isinstance(case, dict) or int(case.get("iterations", -1)) != 0:
            raise ValueError("rank samples are empty but case iterations is not zero")
        metric_names = set()
    else:
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
    dispatch_only = first.get("benchmark_kind") == "dispatch_hot_loop"
    dispatch_modes = tuple(first.get(
        "dispatch_modes", ("hidden", "weight", "pair")))
    if dispatch_only:
        if not dispatch_modes or len(set(dispatch_modes)) != len(dispatch_modes):
            raise ValueError("Dispatch modes must be non-empty and contain no duplicates")
        required_dispatch_metrics = {
            f"{mode}_{kind}" for mode in dispatch_modes for kind in ("host", "kernel")
        }
        missing = required_dispatch_metrics - metric_names
        if missing:
            raise ValueError(
                f"dispatch hot-loop samples are missing timing metrics: {sorted(missing)}"
            )
    flow_stages = (
        None
        if dispatch_only
        else (
            (
                [],
                {
                    stage: {
                        "timings_us": None,
                        "algorithm_bytes": None,
                        "algorithm_bandwidth_GBps": None,
                    }
                    for stage in FLOW_STAGE_ORDER
                },
            )
            if iteration_count == 0
            else _aggregate_flow_stages(rank_samples, world_size=world_size)
        )
    )
    transport_correctness_valid = bool(
        capabilities.get("transport_correctness_valid", False)
    )
    oversubscribed = bool(first["topology"]["oversubscribed"])
    peer_memory_cross_node = bool(
        first["topology"].get("peer_memory_cross_node", False)
    )
    cross_node_validated = bool(
        first["topology"].get("cross_node_validated", False)
    )
    native_mask_valid = (
        (int(capabilities["stage_mask"]) & 3) == 3
        and (int(capabilities["stub_mask"]) & 3) == 0
    ) if dispatch_only else bool(capabilities["transport_performance_valid"])
    transport_performance_valid = bool(
        iteration_count > 0
        and native_mask_valid
        and not oversubscribed
        and (not peer_memory_cross_node or cross_node_validated)
    )
    if iteration_count == 0:
        performance_scope = "warmup_only"
    elif oversubscribed:
        performance_scope = "oversubscribed_functional_only"
    elif peer_memory_cross_node and not cross_node_validated:
        performance_scope = "cross_node_functional_unvalidated"
    elif transport_performance_valid:
        performance_scope = "dispatch_native" if dispatch_only else "transport"
    elif transport_correctness_valid:
        performance_scope = "native_correctness_only"
    else:
        performance_scope = "stub_contract_only"
    validation_mode = (
        "planner_and_dispatch_bit_exact" if dispatch_only else
        ("full" if transport_performance_valid else performance_scope)
    )
    summary = {
        "schema_version": 1,
        "benchmark_kind": first.get("benchmark_kind", "flow"),
        "status": "passed",
        "mode": first.get("mode", "benchmark"),
        "case": first["case"],
        "dispatch_modes": list(dispatch_modes) if dispatch_only else None,
        "logical_world_size": world_size,
        "node_count": first["topology"].get("node_count"),
        "visible_devices": first["topology"].get("visible_devices"),
        "physical_device_count": first["topology"]["physical_device_count"],
        "ranks_per_device": first["topology"]["ranks_per_device"],
        "oversubscribed": oversubscribed,
        "planner_block_dim": first["topology"].get("planner_block_dim"),
        "planner_block_dim_source": first["topology"].get("planner_block_dim_source"),
        "dispatch_aiv_core_count": first["topology"].get("dispatch_aiv_core_count"),
        "dispatch_aiv_core_count_source": first["topology"].get(
            "dispatch_aiv_core_count_source"),
        "udma_qp_route_spec": first["topology"].get("udma_qp_route_spec"),
        "capabilities": capabilities,
        "environment": reference_environment,
        "benchmark_config": reference_benchmark_config,
        "transport_correctness_valid": transport_correctness_valid,
        "transport_performance_valid": transport_performance_valid,
        "performance_scope": performance_scope,
        "validation": {
            "passed": all(bool(item["validation"]["passed"]) for item in rank_results),
            "mode": validation_mode,
        },
        "cross_rank_max_samples": maxima,
        "metrics_us": metrics,
    }
    if flow_stages is not None:
        stage_samples, stage_performance = flow_stages
        summary["cross_rank_stage_samples"] = stage_samples
        summary["stage_performance"] = stage_performance
        if reference_stage_execution is not None:
            summary["stage_execution"] = reference_stage_execution
    tokens_per_rank = int(first["case"]["tokens_per_rank"])
    throughput_metrics = ({mode: f"{mode}_kernel" for mode in dispatch_modes}
        if dispatch_only else {"end_to_end": "end_to_end"})
    throughput_by_mode = {}
    for mode, metric_name in throughput_metrics.items():
        throughput = []
        for sample in maxima:
            duration = float(sample["timings_us"][metric_name])
            throughput.append(
                tokens_per_rank * world_size * 1_000_000.0 / duration
                if duration > 0.0
                else 0.0
            )
        throughput_by_mode[mode] = _metric_summary(throughput) if throughput else None
    summary["tokens_per_second"] = (
        throughput_by_mode["pair" if "pair" in dispatch_modes else dispatch_modes[0]]
        if dispatch_only else
        throughput_by_mode["end_to_end"]
    )
    if dispatch_only:
        summary["tokens_per_second_by_mode"] = throughput_by_mode
    write_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "min", "max", "mean", "p50", "p95", "p90", "p99"),
        )
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"metric": name, **values})
    if flow_stages is not None:
        _write_stage_csv(
            root / "stage_summary.csv",
            summary["stage_performance"],
            summary.get("stage_execution"),
        )
    return summary


def aggregate_distributed_artifacts(
    output_dir: str | Path,
    *,
    case_ids: Iterable[str],
    node_count: int,
    world_size: int,
    mode: str,
) -> dict[str, dict[str, object]]:
    if node_count <= 1:
        raise ValueError("distributed aggregation requires node_count greater than one")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if mode not in ("benchmark", "reference", "correctness"):
        raise ValueError("mode must be benchmark, reference, or correctness")
    normalized_case_ids = tuple(case_ids)
    if not normalized_case_ids:
        raise ValueError("at least one case id is required")
    if len(set(normalized_case_ids)) != len(normalized_case_ids):
        raise ValueError("case ids must be unique")
    root = Path(output_dir).resolve()
    covered_ranks = set()
    rank_owners = {}
    expected_launch_id = None
    for node_rank in range(node_count):
        marker_path = root / f"node_{node_rank}_complete.json"
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if marker.get("status") != "passed":
            raise RuntimeError(f"node {node_rank} did not complete successfully")
        if marker.get("mode") != mode:
            raise ValueError(f"node {node_rank} completion marker has invalid mode")
        launch_id = marker.get("launch_id")
        if not isinstance(launch_id, str) or not launch_id:
            raise ValueError(f"node {node_rank} completion marker has no launch id")
        if expected_launch_id is None:
            expected_launch_id = launch_id
        elif launch_id != expected_launch_id:
            raise ValueError("node completion markers have different launch ids")
        marker_case_ids = marker.get("case_ids")
        if (
            not isinstance(marker_case_ids, list)
            or any(not isinstance(value, str) for value in marker_case_ids)
            or set(marker_case_ids) != set(normalized_case_ids)
        ):
            raise ValueError(f"node {node_rank} completion marker has invalid case ids")
        topology = marker.get("topology")
        if not isinstance(topology, dict):
            raise ValueError(f"node {node_rank} completion marker has no topology")
        expected = {
            "node_count": node_count,
            "node_rank": node_rank,
            "world_size": world_size,
        }
        for name, value in expected.items():
            if int(topology.get(name, -1)) != value:
                raise ValueError(
                    f"node {node_rank} completion marker has invalid {name}"
                )
        first_rank = int(topology.get("first_global_rank", -1))
        local_world_size = int(topology.get("local_world_size", -1))
        if first_rank < 0 or local_world_size <= 0:
            raise ValueError(f"node {node_rank} completion marker has invalid rank range")
        node_ranks = set(range(first_rank, first_rank + local_world_size))
        if covered_ranks.intersection(node_ranks):
            raise ValueError("node completion rank ranges overlap")
        covered_ranks.update(node_ranks)
        for rank in node_ranks:
            rank_owners[rank] = (node_rank, first_rank, local_world_size)
    if covered_ranks != set(range(world_size)):
        raise ValueError("node completion rank ranges do not cover world_size")

    summaries = {}
    for case_id in normalized_case_ids:
        case_dir = (root / case_id).resolve()
        if root not in case_dir.parents:
            raise ValueError(f"case artifact path escapes output root: {case_dir}")
        for rank in range(world_size):
            result_path = case_dir / f"rank_{rank}" / "result.json"
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            if result.get("launch_id") != expected_launch_id:
                raise ValueError(f"rank {rank} result has invalid launch id")
            if result.get("mode") != mode:
                raise ValueError(f"rank {rank} result has invalid mode")
            if result.get("case", {}).get("case_id") != case_id:
                raise ValueError(f"rank {rank} result has invalid case id")
            if mode == "benchmark":
                topology = result.get("topology")
                if not isinstance(topology, dict):
                    raise ValueError(f"rank {rank} result has no topology")
                owner, first_rank, local_world_size = rank_owners[rank]
                expected = {
                    "global_rank": rank,
                    "global_world_size": world_size,
                    "node_rank": owner,
                    "local_rank": rank - first_rank,
                    "local_world_size": local_world_size,
                }
                for name, value in expected.items():
                    if int(topology.get(name, -1)) != value:
                        raise ValueError(f"rank {rank} result has invalid {name}")
        summaries[case_id] = (
            aggregate_rank_artifacts(case_dir, world_size=world_size)
            if mode == "benchmark"
            else aggregate_correctness_artifacts(case_dir, world_size=world_size)
        )
    return summaries


def aggregate_correctness_artifacts(
    case_dir: str | Path,
    *,
    world_size: int,
) -> dict[str, object]:
    root = Path(case_dir)
    results = []
    for rank in range(world_size):
        path = root / f"rank_{rank}" / "result.json"
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") != "passed":
            raise RuntimeError(
                f"rank {rank} failed: {result.get('failure_reason', 'unknown')}"
            )
        if int(result.get("rank", rank)) != rank:
            raise ValueError(f"rank_{rank} contains result metadata for another rank")
        validation = result.get("validation")
        if not isinstance(validation, dict) or not isinstance(
            validation.get("stages"), list
        ):
            raise ValueError(f"rank {rank} has invalid correctness validation metadata")
        stage_names = tuple(stage.get("stage") for stage in validation["stages"])
        if stage_names != CORRECTNESS_STAGES:
            raise ValueError(
                f"rank {rank} must contain ordered correctness stages "
                f"{CORRECTNESS_STAGES}, got {stage_names}"
            )
        case_id = result.get("case", {}).get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"rank {rank} has invalid correctness case id")
        for stage in validation["stages"]:
            stage_path = (
                root
                / f"rank_{rank}"
                / "stages"
                / f"{case_id}.{stage['stage']}.json"
            )
            if not stage_path.is_file():
                raise FileNotFoundError(
                    f"rank {rank} is missing stage artifact {stage_path.name}"
                )
            with stage_path.open("r", encoding="utf-8") as handle:
                artifact = json.load(handle)
            if artifact.get("stage") != stage["stage"] or bool(
                artifact.get("passed")
            ) != bool(stage.get("passed")):
                raise ValueError(
                    f"rank {rank} stage artifact {stage_path.name} differs from result"
                )
        results.append(result)
    reference = results[0]
    for rank, result in enumerate(results[1:], start=1):
        if result["case"] != reference["case"]:
            raise ValueError(f"rank {rank} case metadata differs from rank 0")
        if result["mode"] != reference["mode"]:
            raise ValueError(f"rank {rank} mode differs from rank 0")
    stages = [
        {
            "stage": stage_name,
            "passed": all(
                bool(result["validation"]["stages"][stage_index]["passed"])
                for result in results
            ),
        }
        for stage_index, stage_name in enumerate(CORRECTNESS_STAGES)
    ]
    summary = {
        "schema_version": 1,
        "status": "passed" if all(stage["passed"] for stage in stages) else "failed",
        "mode": reference["mode"],
        "case": reference["case"],
        "logical_world_size": world_size,
        "performance_valid": False,
        "reference_backend": reference["validation"]["reference_backend"],
        "candidate_backend": reference["validation"].get("candidate_backend"),
        "stages": stages,
    }
    write_json(root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate and print MoonEP reports")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--summary")
    action.add_argument("--aggregate-output-dir")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--node-count", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument(
        "--mode", choices=("benchmark", "reference", "correctness"),
        default="benchmark",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summary is not None:
        with Path(args.summary).open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        print(format_stage_performance(summary))
        return 0
    if not args.case_ids or args.node_count is None or args.world_size is None:
        raise ValueError(
            "distributed aggregation requires --case-id, --node-count, and --world-size"
        )
    aggregate_distributed_artifacts(
        args.aggregate_output_dir,
        case_ids=args.case_ids,
        node_count=args.node_count,
        world_size=args.world_size,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
