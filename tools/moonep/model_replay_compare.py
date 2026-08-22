from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Mapping

from .model_flow import model_operator_order
from .report import MODEL_STAGE_LABELS, MODEL_STAGE_ORDER


_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_ANSI_GREEN = "\x1b[32m"
_ANSI_YELLOW = "\x1b[33m"
_ANSI_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"


def _distribution(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("comparison metrics must be non-empty and finite")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _human_latency(value: float) -> str:
    if value < 1000.0:
        return f"{value:.1f} us"
    if value < 1_000_000.0:
        return f"{value / 1000.0:.1f} ms"
    return f"{value / 1_000_000.0:.1f} s"


def _human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    normalized = float(value)
    unit = units[0]
    for unit in units:
        if normalized < 1024.0 or unit == units[-1]:
            break
        normalized /= 1024.0
    return f"{normalized:.1f} {unit}"


def _triplet(
    value: Mapping[str, object] | None,
    formatter: Callable[[float], str],
) -> str:
    if value is None:
        return "N/A"
    return " / ".join(
        formatter(float(value[name])) for name in ("median", "min", "max")
    )


def _bandwidth(value: float) -> str:
    return f"{value:.1f} GB/s"


def _split_value_unit(value: str) -> tuple[str, str]:
    pieces = value.rsplit(" ", 1)
    if len(pieces) == 2:
        return pieces[0], pieces[1]
    return value, ""


def _compact_triplet(
    value: Mapping[str, object] | None,
    formatter: Callable[[float], str],
) -> str:
    if value is None:
        return "N/A"
    median = formatter(float(value["median"]))
    minimum = formatter(float(value["min"]))
    maximum = formatter(float(value["max"]))
    median_value, median_unit = _split_value_unit(median)
    minimum_value, minimum_unit = _split_value_unit(minimum)
    maximum_value, maximum_unit = _split_value_unit(maximum)
    if median_unit and median_unit == minimum_unit == maximum_unit:
        return f"{median_value} {median_unit} [{minimum_value}, {maximum_value}]"
    return f"{median} [{minimum}, {maximum}]"


def _kernel_short(name: object) -> str:
    value = str(name)
    for prefix in (
        "tilexr_moonep_",
        "tilexr_ep_",
        "tilexr_",
        "model_",
        "replay_",
    ):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if value.endswith("_kernel"):
        value = value[: -len("_kernel")]
    if value == "plan":
        return "plan"
    return value


def _visible_len(value: str) -> int:
    return len(_ANSI_PATTERN.sub("", value))


def _pad(value: str, width: int) -> str:
    return value + " " * max(0, width - _visible_len(value))


def _format_delta(delta: float, *, use_color: bool) -> str:
    text = f"{delta:+.1f}%"
    if not use_color:
        return text
    magnitude = abs(delta)
    if magnitude <= 5.0:
        color = _ANSI_GREEN
    elif magnitude <= 10.0:
        color = _ANSI_YELLOW
    else:
        color = _ANSI_RED
    return f"{color}{text}{_ANSI_RESET}"


def _rank_operator_lists(performance: Mapping[str, object]) -> list[list[object]]:
    ranks = performance.get("ranks")
    if not isinstance(ranks, Mapping) or not ranks:
        raise ValueError("model performance ranks are missing")
    expected_order = model_operator_order()
    rank_operators = []
    for rank in sorted(ranks, key=int):
        record = ranks[rank]
        operators = record.get("operators") if isinstance(record, Mapping) else None
        if not isinstance(operators, list) or len(operators) != len(expected_order):
            raise ValueError(f"model rank {rank} has an invalid operator count")
        for sequence, (operator, expected) in enumerate(zip(operators, expected_order)):
            if (
                not isinstance(operator, Mapping)
                or operator.get("sequence") != sequence
                or operator.get("stage") != expected["stage"]
            ):
                raise ValueError(f"model operator {sequence} metadata is inconsistent")
        rank_operators.append(operators)
    return rank_operators


def _model_operators(performance: Mapping[str, object]) -> list[dict[str, object]]:
    rank_operators = _rank_operator_lists(performance)
    expected_order = model_operator_order()
    result = []
    for sequence, expected in enumerate(expected_order):
        operators = [values[sequence] for values in rank_operators]
        backends = {str(operator["backend"]) for operator in operators}
        kernels = {str(operator["kernel_version"]) for operator in operators}
        if len(backends) != 1 or len(kernels) != 1:
            raise ValueError(f"model operator {sequence} backend or kernel differs by rank")
        latency = _distribution([float(operator["latency_us"]) for operator in operators])
        bytes_values = [operator.get("algorithm_bytes") for operator in operators]
        bandwidth_values = [
            operator.get("algorithm_bandwidth_GBps") for operator in operators
        ]
        if all(value is None for value in bytes_values):
            byte_distribution = None
        elif any(value is None for value in bytes_values):
            raise ValueError(f"model operator {sequence} mixes byte values and N/A")
        else:
            byte_distribution = _distribution([float(value) for value in bytes_values])
        if all(value is None for value in bandwidth_values):
            bandwidth_distribution = None
        elif any(value is None for value in bandwidth_values):
            raise ValueError(f"model operator {sequence} mixes bandwidth and N/A")
        else:
            bandwidth_distribution = _distribution(
                [float(value) for value in bandwidth_values]
            )
        result.append(
            {
                **expected,
                "backend": next(iter(backends)),
                "kernel_version": next(iter(kernels)),
                "latency_us": latency,
                "algorithm_bytes": byte_distribution,
                "algorithm_bandwidth_GBps": bandwidth_distribution,
            }
        )
    return result


def _model_stage_performance(
    performance: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    rank_operators = _rank_operator_lists(performance)
    result: dict[str, dict[str, object]] = {}
    for stage in MODEL_STAGE_ORDER:
        rank_latency_values: list[float] = []
        rank_byte_values: list[float] = []
        rank_bandwidth_values: list[float] = []
        backends = set()
        kernels = set()
        calls_per_rank = None
        for operators in rank_operators:
            stage_operators = [
                operator for operator in operators if operator.get("stage") == stage
            ]
            if not stage_operators:
                raise ValueError(f"model stage {stage} is missing")
            if calls_per_rank is None:
                calls_per_rank = len(stage_operators)
            elif calls_per_rank != len(stage_operators):
                raise ValueError(f"model stage {stage} call count differs by rank")
            backends.update(str(operator["backend"]) for operator in stage_operators)
            kernels.update(str(operator["kernel_version"]) for operator in stage_operators)
            latency_mean = statistics.fmean(
                float(operator["latency_us"]) for operator in stage_operators
            )
            rank_latency_values.append(latency_mean)
            byte_values = [
                operator.get("algorithm_bytes") for operator in stage_operators
            ]
            if all(value is None for value in byte_values):
                continue
            if any(value is None for value in byte_values):
                raise ValueError(f"model stage {stage} mixes byte values and N/A")
            byte_mean = statistics.fmean(float(value) for value in byte_values)
            rank_byte_values.append(byte_mean)
            rank_bandwidth_values.append(
                byte_mean / latency_mean / 1000.0 if latency_mean > 0.0 else 0.0
            )
        if len(backends) != 1 or len(kernels) != 1 or calls_per_rank is None:
            raise ValueError(f"model stage {stage} backend or kernel differs by rank")
        result[stage] = {
            "calls_per_rank": calls_per_rank,
            "backend": next(iter(backends)),
            "kernel_version": next(iter(kernels)),
            "latency_us": _distribution(rank_latency_values),
            "algorithm_bytes": (
                _distribution(rank_byte_values) if rank_byte_values else None
            ),
            "algorithm_bandwidth_GBps": (
                _distribution(rank_bandwidth_values)
                if rank_bandwidth_values
                else None
            ),
        }
    return result


def _replay_stage_performance(
    replay_summary: Mapping[str, object],
) -> Mapping[str, object]:
    stages = replay_summary.get("model_stage_performance")
    if not isinstance(stages, Mapping) or set(stages) != set(MODEL_STAGE_ORDER):
        raise ValueError("replay summary has an invalid model stage summary")
    return stages


def _stage_execution_key(stage: str) -> str:
    if stage.startswith("dispatch_"):
        return "dispatch"
    if stage.startswith("combine_"):
        return "combine"
    return stage


def _native_label(value: object) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "N/A"


def _case_value(case: Mapping[str, object], key: str) -> object:
    value = case.get(key)
    return "N/A" if value is None else value


def _format_comparison_inputs(replay_summary: Mapping[str, object]) -> str:
    case = replay_summary.get("case")
    if not isinstance(case, Mapping):
        return "MoonEP model replay comparison inputs\nCase: N/A"
    return "\n".join(
        (
            "MoonEP model replay comparison inputs",
            f"Case: {_case_value(case, 'case_id')}  "
            f"mode={replay_summary.get('mode', 'N/A')}",
            "Dimensions: "
            f"S={_case_value(case, 'tokens_per_rank')}  "
            f"K={_case_value(case, 'topk')}  "
            f"E={_case_value(case, 'expert_count')}  "
            f"H={_case_value(case, 'hidden_size')}  "
            f"Hf={_case_value(case, 'intermediate_size')}  "
            f"B={_case_value(case, 'prefetch_slots')}  "
            f"P={_case_value(case, 'token_padding')}",
            "Run: "
            f"warmup={_case_value(case, 'warmup')}  "
            f"iterations={_case_value(case, 'iterations')}  "
            f"logical_ranks={replay_summary.get('logical_world_size', 'N/A')}  "
            f"nodes={replay_summary.get('node_count', 'N/A')}",
        )
    )


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(_visible_len(headers[index]), *(_visible_len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    return "\n".join(
        "  ".join(_pad(value, widths[index]) for index, value in enumerate(row))
        for row in (
            headers,
            tuple("-" * width for width in widths),
            *rows,
        )
    )


def _format_kernel_legend(
    entries: list[tuple[str, str, str]],
) -> str:
    lines = []
    seen = set()
    for short_pair, model_full, replay_full in entries:
        key = (short_pair, model_full, replay_full)
        if key in seen:
            continue
        seen.add(key)
        if model_full == replay_full:
            lines.append(f"  {short_pair}: {model_full}")
        else:
            lines.append(f"  {short_pair}: model={model_full}; replay={replay_full}")
    if not lines:
        return ""
    return "\n".join(
        (
            "Kernel legend: short names remove tilexr/model/replay prefixes and _kernel suffix",
            *lines,
        )
    )


def format_model_replay_stage_comparison(
    model_performance: Mapping[str, object],
    replay_summary: Mapping[str, object],
    *,
    use_color: bool = False,
) -> str:
    if replay_summary.get("benchmark_kind") != "model_flow":
        raise ValueError("replay summary is not a model-flow report")
    model = _model_stage_performance(model_performance)
    replay = _replay_stage_performance(replay_summary)
    execution = replay_summary.get("stage_execution")
    metadata_rows = []
    performance_rows = []
    kernel_legend_entries = []
    for stage in MODEL_STAGE_ORDER:
        model_item = model[stage]
        replay_item = replay[stage]
        if not isinstance(replay_item, Mapping):
            raise ValueError(f"replay stage {stage} metadata is inconsistent")
        model_latency = model_item["latency_us"]
        replay_latency = replay_item["latency_us"]
        delta = (
            (float(replay_latency["median"]) - float(model_latency["median"]))
            / float(model_latency["median"])
            * 100.0
        )
        native = "N/A"
        if isinstance(execution, Mapping):
            record = execution.get(_stage_execution_key(stage))
            if isinstance(record, Mapping):
                native = _native_label(record.get("native"))
        model_kernel = str(model_item["kernel_version"])
        replay_kernel = str(replay_item["kernel_name"])
        kernel_pair = f"{_kernel_short(model_kernel)} / {_kernel_short(replay_kernel)}"
        metadata_rows.append(
            (
                MODEL_STAGE_LABELS[stage],
                f"{model_item['backend']} / tilexr",
                native,
                kernel_pair,
            )
        )
        kernel_legend_entries.append((kernel_pair, model_kernel, replay_kernel))
        performance_rows.extend(
            (
                (
                    MODEL_STAGE_LABELS[stage],
                    "model",
                    _compact_triplet(model_item["algorithm_bytes"], _human_bytes),
                    _compact_triplet(model_latency, _human_latency),
                    "",
                    _compact_triplet(
                        model_item["algorithm_bandwidth_GBps"], _bandwidth
                    ),
                ),
                (
                    MODEL_STAGE_LABELS[stage],
                    "replay",
                    _compact_triplet(replay_item["algorithm_bytes"], _human_bytes),
                    _compact_triplet(replay_latency, _human_latency),
                    _format_delta(delta, use_color=use_color),
                    _compact_triplet(
                        replay_item["algorithm_bandwidth_GBps"], _bandwidth
                    ),
                ),
            )
        )
    metadata_headers = (
        "Stage",
        "Backend M/R",
        "Native",
        "Kernel version M/R (short)",
    )
    performance_headers = (
        "Stage",
        "Side",
        "Algorithm bytes med[min,max]",
        "Latency med[min,max]",
        "Δ Lat",
        "Bandwidth med[min,max]",
    )
    return "\n".join(
        (
            _format_comparison_inputs(replay_summary),
            "",
            "MoonEP model vs cascade replay stage summary",
            "Statistics: rank means aggregated as median / min / max; replay delta uses medians",
            "",
            "Stage metadata",
            _format_table(metadata_headers, metadata_rows),
            _format_kernel_legend(kernel_legend_entries),
            "",
            "Stage performance",
            _format_table(performance_headers, performance_rows),
        )
    )


def format_model_replay_operator_comparison(
    model_performance: Mapping[str, object],
    replay_summary: Mapping[str, object],
) -> str:
    if replay_summary.get("benchmark_kind") != "model_flow":
        raise ValueError("replay summary is not a model-flow report")
    model = _model_operators(model_performance)
    replay = replay_summary.get("model_operator_performance")
    if not isinstance(replay, list) or len(replay) != len(model):
        raise ValueError("replay summary has an invalid operator count")
    rows = []
    for sequence, (model_item, replay_item) in enumerate(zip(model, replay)):
        if (
            not isinstance(replay_item, Mapping)
            or replay_item.get("sequence") != sequence
            or replay_item.get("stage") != model_item["stage"]
        ):
            raise ValueError(f"replay operator {sequence} metadata is inconsistent")
        model_latency = model_item["latency_us"]
        replay_latency = replay_item["latency_us"]
        delta = (
            (float(replay_latency["median"]) - float(model_latency["median"]))
            / float(model_latency["median"])
            * 100.0
        )
        rows.append(
            (
                str(sequence + 1),
                str(model_item["phase"]),
                str(model_item["layer"]),
                str(model_item["stage"]),
                f"{model_item['backend']} / tilexr",
                f"{model_item['kernel_version']} / {replay_item['kernel_name']}",
                f"{_triplet(model_item['algorithm_bytes'], _human_bytes)} / "
                f"{_triplet(replay_item['algorithm_bytes'], _human_bytes)}",
                f"{_triplet(model_latency, _human_latency)} / "
                f"{_triplet(replay_latency, _human_latency)}",
                f"{delta:+.1f}%",
                f"{_triplet(model_item['algorithm_bandwidth_GBps'], _bandwidth)} / "
                f"{_triplet(replay_item['algorithm_bandwidth_GBps'], _bandwidth)}",
            )
        )
    headers = (
        "Seq",
        "Phase",
        "Layer",
        "Stage",
        "Backend (model / replay)",
        "Kernel version (model / replay)",
        "Algorithm bytes (model / replay median / min / max)",
        "Latency median / min / max (model / replay)",
        "Replay latency delta",
        "Bandwidth median / min / max (model / replay)",
    )
    return "\n".join(
        (
            "MoonEP model vs cascade replay performance",
            "Statistics: median / min / max across ranks; replay delta uses medians",
            _format_table(headers, rows),
        )
    )


def format_model_replay_comparison(
    model_performance: Mapping[str, object],
    replay_summary: Mapping[str, object],
    *,
    stage_summary_only: bool = False,
    use_color: bool = False,
) -> str:
    stage_summary = format_model_replay_stage_comparison(
        model_performance,
        replay_summary,
        use_color=use_color,
    )
    if stage_summary_only:
        return stage_summary
    return "\n\n".join(
        (
            stage_summary,
            format_model_replay_operator_comparison(model_performance, replay_summary),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare model and replay performance")
    parser.add_argument("--model", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument(
        "--stage-summary-only",
        action="store_true",
        help="print only the aggregated stage comparison table",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colorize latency delta values (default: auto)",
    )
    args = parser.parse_args(argv)
    with Path(args.model).open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    with Path(args.replay).open("r", encoding="utf-8") as handle:
        replay = json.load(handle)
    use_color = args.color == "always" or (
        args.color == "auto"
        and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )
    print(
        format_model_replay_comparison(
            model,
            replay,
            stage_summary_only=args.stage_summary_only,
            use_color=use_color,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
