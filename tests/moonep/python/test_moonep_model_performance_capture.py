from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.moonep.mindspeed.collect_model_performance import (
    collect_model_performance,
    operator_algorithm_bytes,
)
from tools.moonep.model_flow import model_operator_order
from tools.moonep.model_replay_cache import ReplayShape, validate_model_performance


def _shape() -> ReplayShape:
    return ReplayShape(
        tokens_per_rank=2,
        topk=2,
        hidden_size=128,
        ep_size=2,
        world_size=2,
        expert_count=4,
        ffn_hidden_size=32,
    )


def _identity() -> dict[str, object]:
    return {"shape": _shape().as_dict(), "operator_order": model_operator_order()}


def _route_replay() -> dict[str, object]:
    records = [
        {
            "call": call,
            "remote_stats": [call + 1, call + 2],
            "experts_to_copy": [[0, 1], [2, 3]],
        }
        for call in range(10)
    ]
    return {
        "schema_version": 1,
        "dimensions": {
            "world_size": 2,
            "tokens_per_rank": 2,
            "topk": 2,
            "expert_count": 4,
            "forward_calls": 10,
        },
        "ranks": {"0": records, "1": records},
    }


def _iteration_rows(offset: float) -> list[dict[str, object]]:
    rows = []

    def add(name: str, count: int, base: float) -> None:
        for index in range(count):
            rows.append(
                {
                    "Name": name,
                    "Duration(us)": offset + base + index,
                    "Device_id": int(offset // 1000),
                }
            )

    add("tilexr_ep_plan_kernel", 10, 100.0)
    add("tilexr_moonep_dispatch_urma_kernel", 15, 200.0)
    add("tilexr_moonep_prefetch_weight_kernel", 10, 300.0)
    add("tilexr_moonep_combine_v2_kernel", 20, 400.0)
    add("tilexr_moonep_reduce_grad_kernel", 5, 500.0)
    return rows


def _write_kernel_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Name", "Duration(us)", "Device_id"))
        writer.writeheader()
        writer.writerows(rows)


def test_operator_bytes_use_shape_and_captured_remote_counts() -> None:
    shape = _shape()
    record = _route_replay()["ranks"]["0"][2]

    assert operator_algorithm_bytes("planning", record, shape) is None
    assert operator_algorithm_bytes("dispatch_forward", record, shape) == 1040
    assert operator_algorithm_bytes("dispatch_backward", record, shape) == 1024
    assert operator_algorithm_bytes("combine_forward", record, shape) == 1024
    assert operator_algorithm_bytes("combine_backward", record, shape) == 1040
    assert operator_algorithm_bytes("prefetch_weight", record, shape) == 73920
    assert operator_algorithm_bytes("reduce_grad", record, shape) == 98432


def test_collect_model_performance_emits_all_rank_55_operator_contract(
    tmp_path: Path,
) -> None:
    _write_kernel_csv(
        tmp_path / "node_0" / "rank_0" / "kernel_details.csv",
        _iteration_rows(0.0),
    )
    _write_kernel_csv(
        tmp_path / "node_0" / "rank_1" / "kernel_details.csv",
        _iteration_rows(1000.0),
    )

    performance = collect_model_performance(
        tmp_path,
        _route_replay(),
        _shape(),
        profiler_enabled=True,
        backend="tilexr",
        iterations=1,
    )

    validate_model_performance(performance, _identity())
    assert performance["world_size"] == 2
    assert performance["profiler_enabled"] is True
    assert len(performance["ranks"]["0"]["operators"]) == 55
    first = performance["ranks"]["0"]["operators"][0]
    assert first["stage"] == "planning"
    assert first["latency_us"] == 100.0
    assert first["algorithm_bytes"] is None
    dispatch = performance["ranks"]["0"]["operators"][1]
    assert dispatch["stage"] == "dispatch_forward"
    assert dispatch["algorithm_bytes"] == 1040
    assert dispatch["algorithm_bandwidth_GBps"] == 1040 / 200.0 / 1000.0


def test_collect_model_performance_accepts_lightweight_event_capture(
    tmp_path: Path,
) -> None:
    order = model_operator_order()
    for rank in range(2):
        path = tmp_path / f"rank{rank}_performance.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "capture_id": "performance-1",
                    "rank": rank,
                    "timing_source": "torch_npu_event",
                    "operators": [
                        {
                            "sequence": sequence,
                            "source_operator": sequence + 330,
                            "stage": item["stage"],
                            "latency_us": 10.0 + sequence + rank,
                        }
                        for sequence, item in enumerate(order)
                    ],
                }
            ),
            encoding="utf-8",
        )

    performance = collect_model_performance(
        tmp_path,
        _route_replay(),
        _shape(),
        profiler_enabled=False,
        backend="tilexr",
        iterations=1,
    )

    validate_model_performance(performance, _identity())
    assert performance["profiler_enabled"] is False
    assert performance["timing_source"] == "torch_npu_event"
    assert performance["ranks"]["1"]["operators"][0]["latency_us"] == 11.0
