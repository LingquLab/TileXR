from __future__ import annotations

from tools.moonep.model_flow import model_operator_order
from tools.moonep.model_replay_compare import format_model_replay_comparison
from tools.moonep.report import MODEL_STAGE_ORDER


def _model_performance() -> dict[str, object]:
    order = model_operator_order()
    return {
        "schema_version": 1,
        "world_size": 2,
        "profiler_enabled": False,
        "timing_source": "torch_npu_event",
        "ranks": {
            str(rank): {
                "operators": [
                    {
                        "sequence": sequence,
                        "phase": item["phase"],
                        "layer": item["layer"],
                        "stage": item["stage"],
                        "backend": "tilexr",
                        "kernel_version": f"model_{item['stage']}",
                        "algorithm_bytes": None if item["stage"] == "planning" else 1024,
                        "latency_us": 100.0 + sequence + rank,
                        "algorithm_bandwidth_GBps": (
                            None if item["stage"] == "planning" else 0.01 + rank
                        ),
                    }
                    for sequence, item in enumerate(order)
                ]
            }
            for rank in range(2)
        },
    }


def _replay_summary() -> dict[str, object]:
    stage_counts = {
        stage: sum(1 for item in model_operator_order() if item["stage"] == stage)
        for stage in MODEL_STAGE_ORDER
    }
    return {
        "benchmark_kind": "model_flow",
        "stage_execution": {
            "planning": {"native": True},
            "dispatch": {"native": True},
            "prefetch_weight": {"native": True},
            "combine": {"native": True},
            "reduce_grad": {"native": True},
        },
        "model_operator_performance": [
            {
                **item,
                "kernel_name": f"replay_{item['stage']}",
                "latency_us": {"median": 110.0 + sequence, "min": 109.0, "max": 111.0},
                "algorithm_bytes": (
                    None
                    if item["stage"] == "planning"
                    else {"median": 1024.0, "min": 1024.0, "max": 1024.0}
                ),
                "algorithm_bandwidth_GBps": (
                    None
                    if item["stage"] == "planning"
                    else {"median": 0.02, "min": 0.01, "max": 0.03}
                ),
            }
            for sequence, item in enumerate(model_operator_order())
        ],
        "model_stage_performance": {
            stage: {
                "calls_per_rank": stage_counts[stage],
                "kernel_name": f"replay_{stage}",
                "latency_us": {"median": 120.0, "min": 110.0, "max": 130.0},
                "algorithm_bytes": (
                    None
                    if stage == "planning"
                    else {"median": 1024.0, "min": 1024.0, "max": 1024.0}
                ),
                "algorithm_bandwidth_GBps": (
                    None
                    if stage == "planning"
                    else {"median": 0.02, "min": 0.01, "max": 0.03}
                ),
            }
            for stage in MODEL_STAGE_ORDER
        },
    }


def test_comparison_prints_mandatory_fields_in_model_order() -> None:
    output = format_model_replay_comparison(
        _model_performance(), _replay_summary()
    )

    assert "MoonEP model vs cascade replay performance" in output
    assert "MoonEP model vs cascade replay stage summary" in output
    assert output.index("stage summary") < output.index("replay performance")
    for field in (
        "Stage",
        "Stage metadata",
        "Stage performance",
        "Backend M/R",
        "Native",
        "Kernel version M/R (short)",
        "Kernel legend",
        "Algorithm bytes med[min,max]",
        "Latency med[min,max]",
        "Bandwidth med[min,max]",
    ):
        assert field in output
    assert "100.5 us" in output
    assert "1.0 KiB" in output
    assert "GB/s" in output
    assert ".000" not in output
    assert ".000000" not in output
    assert output.index("initial_forward") < output.index("recompute_backward")
    assert len([line for line in output.splitlines() if line.startswith("1 ")]) == 1


def test_stage_summary_only_omits_operator_detail() -> None:
    output = format_model_replay_comparison(
        _model_performance(), _replay_summary(), stage_summary_only=True
    )

    assert "MoonEP model vs cascade replay stage summary" in output
    assert "MoonEP model vs cascade replay performance" not in output
    assert "Dispatch forward" in output
    assert "initial_forward" not in output


def test_stage_summary_can_color_latency_delta() -> None:
    output = format_model_replay_comparison(
        _model_performance(),
        _replay_summary(),
        stage_summary_only=True,
        use_color=True,
    )

    assert "\x1b[32m" in output
    assert "\x1b[31m" in output
    assert "-1.2%" in output
