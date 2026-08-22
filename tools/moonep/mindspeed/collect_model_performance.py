from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Mapping

from tools.moonep.model_flow import model_operator_order, parse_tilexr_kernel_rows
from tools.moonep.model_replay_cache import ReplayShape


_BACKWARD_STAGES = {"dispatch_backward", "combine_backward", "reduce_grad"}
_RANK_COMPONENT = re.compile(r"^rank[_-]?(\d+)$")
_NODE_COMPONENT = re.compile(r"^node[_-]?(\d+)$")
_DEVICE_COLUMNS = ("Device_id", "Device ID", "Device", "device_id")
_KERNEL_VERSIONS = {
    "planning": "tilexr_ep_plan_kernel",
    "dispatch_forward": "tilexr_moonep_dispatch_urma_kernel",
    "dispatch_backward": "tilexr_moonep_dispatch_urma_kernel",
    "prefetch_weight": "tilexr_moonep_prefetch_weight_kernel",
    "combine_forward": "tilexr_moonep_combine_v2_kernel",
    "combine_backward": "tilexr_moonep_combine_v2_kernel",
    "reduce_grad": "tilexr_moonep_reduce_grad_kernel",
}


def operator_algorithm_bytes(
    stage: str,
    route_record: Mapping[str, object],
    shape: ReplayShape,
) -> int | None:
    route_count = shape.tokens_per_rank * shape.topk
    hidden_bytes = route_count * shape.hidden_size * 2
    route_weight_bytes = route_count * 4
    if stage == "planning":
        return None
    if stage in ("dispatch_forward", "combine_backward"):
        return hidden_bytes + route_weight_bytes
    if stage in ("dispatch_backward", "combine_forward"):
        return hidden_bytes
    remote_stats = route_record.get("remote_stats")
    if (
        not isinstance(remote_stats, list)
        or len(remote_stats) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in remote_stats
        )
    ):
        raise ValueError("model performance route record has invalid remote_stats")
    if stage == "prefetch_weight":
        projection_row_bytes = 6 * shape.hidden_size * shape.ffn_hidden_size + 64
        return remote_stats[0] * projection_row_bytes
    if stage == "reduce_grad":
        gradient_row_bytes = 6 * shape.hidden_size * shape.ffn_hidden_size + 32
        return remote_stats[1] * gradient_row_bytes
    raise ValueError(f"unknown model stage: {stage}")


def _path_rank(path: Path) -> int | None:
    for component in reversed(path.parts):
        match = _RANK_COMPONENT.fullmatch(component)
        if match:
            return int(match.group(1))
    return None


def _node_rank(path: Path) -> int | None:
    for component in reversed(path.parts):
        match = _NODE_COMPONENT.fullmatch(component)
        if match:
            return int(match.group(1))
    return None


def _device_id(rows: list[dict[str, str]]) -> int | None:
    if not rows:
        return None
    for column in _DEVICE_COLUMNS:
        raw = rows[0].get(column)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid profiler device ID: {raw!r}") from exc
    return None


def _rank_for_profile(
    path: Path,
    rows: list[dict[str, str]],
    *,
    devices_per_node: int,
) -> int:
    rank = _path_rank(path)
    if rank is not None:
        return rank
    node = _node_rank(path)
    device = _device_id(rows)
    if node is None or device is None:
        raise ValueError(f"cannot infer global rank for profiler CSV: {path}")
    return node * devices_per_node + device


def _route_call_for_operator(operator: Mapping[str, object]) -> int:
    stage = str(operator["stage"])
    occurrence = int(operator["stage_occurrence"])
    return occurrence + 5 if stage in _BACKWARD_STAGES else occurrence


def collect_model_performance(
    profile_root: str | Path,
    route_replay: Mapping[str, object],
    shape: ReplayShape,
    *,
    profiler_enabled: bool,
    backend: str,
    iterations: int,
) -> dict[str, object]:
    root = Path(profile_root)
    csv_paths = sorted(root.rglob("kernel_details.csv")) if profiler_enabled else []
    event_paths = (
        [] if profiler_enabled else sorted(root.rglob("rank*_performance.json"))
    )
    artifact_paths = csv_paths if profiler_enabled else event_paths
    if not artifact_paths:
        expected = "kernel_details.csv" if profiler_enabled else "rank*_performance.json"
        raise FileNotFoundError(f"no {expected} below {root}")
    node_ids = {
        node
        for path in artifact_paths
        for node in [_node_rank(path)]
        if node is not None
    }
    node_count = max(node_ids) + 1 if node_ids else 1
    if shape.world_size % node_count != 0:
        raise ValueError("profile node count does not divide replay world size")
    devices_per_node = shape.world_size // node_count
    replay_ranks = route_replay.get("ranks")
    if not isinstance(replay_ranks, Mapping):
        raise ValueError("route replay ranks are missing")

    ranks: dict[str, object] = {}
    timing_source = (
        "torch_npu_profiler_kernel_details" if profiler_enabled else "torch_npu_event"
    )
    for path in artifact_paths:
        if profiler_enabled:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rank = _rank_for_profile(
                path,
                rows,
                devices_per_node=devices_per_node,
            )
            parsed = parse_tilexr_kernel_rows(rows, iterations=iterations)
        else:
            with path.open("r", encoding="utf-8") as handle:
                event_payload = json.load(handle)
            if (
                not isinstance(event_payload, dict)
                or event_payload.get("schema_version") != 1
                or event_payload.get("timing_source") != "torch_npu_event"
            ):
                raise ValueError(f"invalid lightweight performance capture: {path}")
            rank = int(event_payload.get("rank", -1))
            captured = event_payload.get("operators")
            order = model_operator_order()
            if not isinstance(captured, list) or len(captured) != len(order):
                raise ValueError(f"invalid lightweight operator count: {path}")
            parsed = []
            for sequence, (item, expected) in enumerate(zip(captured, order)):
                if (
                    not isinstance(item, dict)
                    or item.get("sequence") != sequence
                    or item.get("stage") != expected["stage"]
                ):
                    raise ValueError(f"invalid lightweight operator order: {path}")
                parsed.append(
                    {
                        **expected,
                        "kernel_name": item.get(
                            "kernel_version", _KERNEL_VERSIONS[str(expected["stage"])]
                        ),
                        "kernel_launches_per_call": int(
                            item.get("kernel_launches_per_call", 1)
                        ),
                        "values_us": [float(item["latency_us"])],
                    }
                )
        if rank < 0 or rank >= shape.world_size or str(rank) in ranks:
            raise ValueError(f"duplicate or out-of-range profiler rank {rank}: {path}")
        route_records = replay_ranks.get(str(rank))
        if not isinstance(route_records, list) or len(route_records) != shape.forward_calls:
            raise ValueError(f"route replay rank {rank} has invalid call records")
        operators = []
        for sequence, operator in enumerate(parsed):
            if int(operator.get("sequence", -1)) != sequence:
                raise ValueError(f"profile rank {rank} has invalid operator order")
            values = operator.get("values_us")
            expected_iterations = iterations if profiler_enabled else 1
            if not isinstance(values, list) or len(values) != expected_iterations:
                raise ValueError(f"profile rank {rank} operator {sequence} has invalid timing")
            latency = statistics.fmean(float(value) for value in values)
            if latency <= 0.0:
                raise ValueError(f"profile rank {rank} operator {sequence} latency is not positive")
            call = _route_call_for_operator(operator)
            byte_count = operator_algorithm_bytes(
                str(operator["stage"]), route_records[call], shape
            )
            operators.append(
                {
                    "sequence": sequence,
                    "stage": operator["stage"],
                    "phase": operator["phase"],
                    "layer": operator["layer"],
                    "backend": backend,
                    "kernel_version": str(operator["kernel_name"]),
                    "kernel_launches_per_call": int(
                        operator["kernel_launches_per_call"]
                    ),
                    "algorithm_bytes": byte_count,
                    "latency_us": latency,
                    "algorithm_bandwidth_GBps": (
                        None if byte_count is None else byte_count / latency / 1000.0
                    ),
                }
            )
        ranks[str(rank)] = {
            "timing_artifact": str(path.resolve()),
            "operators": operators,
        }

    expected = {str(rank) for rank in range(shape.world_size)}
    if set(ranks) != expected:
        missing = sorted(expected - set(ranks), key=int)
        raise ValueError(f"model profiler is missing ranks: {', '.join(missing)}")
    return {
        "schema_version": 1,
        "world_size": shape.world_size,
        "profiler_enabled": bool(profiler_enabled),
        "timing_source": timing_source,
        "operator_order": model_operator_order(),
        "ranks": ranks,
    }
