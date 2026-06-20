import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import SimulationResult, dataclass_to_plain


SUMMARY_FIELDS = (
    "algorithm",
    "collective",
    "rank_count",
    "message_bytes",
    "latency_us",
    "algbw_gbps",
    "busbw_gbps",
    "bottleneck_resource",
    "data_source",
    "validation_ok",
)


def write_result(result: SimulationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dataclass_to_plain(result)
    (out_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def write_results(results: Sequence[SimulationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclass_to_plain(result) for result in results]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary(results: Sequence[SimulationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_FIELDS)
        for result in results:
            writer.writerow(
                [
                    result.algorithm,
                    result.collective,
                    result.rank_count,
                    result.message_bytes,
                    f"{result.latency_us:.6f}",
                    f"{result.algbw_gbps:.6f}",
                    f"{result.busbw_gbps:.6f}",
                    result.bottleneck_resource,
                    result.data_source,
                    str(result.validation.ok).lower(),
                ]
            )


def write_html_report(results: Sequence[SimulationResult], path: Path) -> None:
    write_html_report_from_plain([dataclass_to_plain(result) for result in results], path)


def write_html_report_from_plain(result_data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = _as_result_items(result_data)
    path.write_text(_render_html(items), encoding="utf-8")


def _as_result_items(result_data: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(result_data, Mapping):
        return (result_data,)
    if isinstance(result_data, list):
        return tuple(item for item in result_data if isinstance(item, Mapping))
    raise ValueError("result data must be a result object or a list of result objects")


def _render_html(results: Sequence[Mapping[str, Any]]) -> str:
    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <title>TileXR Collective Simulation Report</title>\n"
        "  <style>\n"
        "    body { font-family: Arial, sans-serif; margin: 24px; color: #222; }\n"
        "    table { border-collapse: collapse; width: 100%; margin: 12px 0 28px; }\n"
        "    th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }\n"
        "    th { background: #f2f4f7; }\n"
        "    .muted { color: #666; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>TileXR Collective Simulation Report</h1>\n"
        "  <h2>Algorithm Selection</h2>\n"
        f"{_render_algorithm_table(results)}\n"
        "  <h2>Congestion Drilldown</h2>\n"
        f"{_render_congestion_table(results)}\n"
        "  <h2>Rank Timeline</h2>\n"
        f"{_render_timeline_table(results)}\n"
        "</body>\n"
        "</html>\n"
    )


def _render_algorithm_table(results: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for result in results:
        validation = result.get("validation", {})
        rows.append(
            "    <tr>"
            f"<td>{_escape(result.get('algorithm'))}</td>"
            f"<td>{_escape(result.get('collective'))}</td>"
            f"<td>{_escape(result.get('rank_count'))}</td>"
            f"<td>{_escape(result.get('message_bytes'))}</td>"
            f"<td>{_float_cell(result.get('latency_us'))}</td>"
            f"<td>{_float_cell(result.get('algbw_gbps'))}</td>"
            f"<td>{_float_cell(result.get('busbw_gbps'))}</td>"
            f"<td>{_escape(result.get('bottleneck_resource'))}</td>"
            f"<td>{_escape(result.get('data_source'))}</td>"
            f"<td>{_escape(validation.get('ok') if isinstance(validation, Mapping) else '')}</td>"
            "</tr>"
        )
    return (
        "  <table>\n"
        "    <thead><tr><th>algorithm</th><th>collective</th><th>ranks</th><th>bytes</th>"
        "<th>latency_us</th><th>algbw_gbps</th><th>busbw_gbps</th><th>bottleneck</th>"
        "<th>source</th><th>validation</th></tr></thead>\n"
        f"    <tbody>\n{chr(10).join(rows)}\n    </tbody>\n"
        "  </table>"
    )


def _render_congestion_table(results: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for result in results:
        for event in _events(result):
            rows.append(
                "    <tr>"
                f"<td>{_escape(result.get('algorithm'))}</td>"
                f"<td>{_escape(result.get('message_bytes'))}</td>"
                f"<td>{_escape(event.get('op_id'))}</td>"
                f"<td>{_escape(event.get('op_type'))}</td>"
                f"<td>{_escape(event.get('rank'))}</td>"
                f"<td>{_escape(event.get('message_bytes'))}</td>"
                f"<td>{_float_cell(event.get('transfer_us'))}</td>"
                f"<td>{_float_cell(event.get('effective_gbps'))}</td>"
                f"<td>{_escape(event.get('bottleneck_resource'))}</td>"
                f"<td>{_escape(event.get('data_source'))}</td>"
                "</tr>"
            )
    if not rows:
        rows.append("    <tr><td colspan=\"10\" class=\"muted\">No simulated events</td></tr>")
    return (
        "  <table>\n"
        "    <thead><tr><th>algorithm</th><th>bytes</th><th>op</th><th>type</th><th>rank</th>"
        "<th>message_bytes</th><th>transfer_us</th><th>effective_gbps</th><th>bottleneck</th>"
        "<th>source</th></tr></thead>\n"
        f"    <tbody>\n{chr(10).join(rows)}\n    </tbody>\n"
        "  </table>"
    )


def _render_timeline_table(results: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for result in results:
        for event in _events(result):
            rows.append(
                "    <tr>"
                f"<td>{_escape(result.get('algorithm'))}</td>"
                f"<td>{_escape(result.get('message_bytes'))}</td>"
                f"<td>{_escape(event.get('rank'))}</td>"
                f"<td>{_escape(event.get('op_id'))}</td>"
                f"<td>{_float_cell(event.get('start_us'))}</td>"
                f"<td>{_float_cell(event.get('end_us'))}</td>"
                f"<td>{_float_cell(event.get('wait_us'))}</td>"
                f"<td>{_escape(', '.join(str(resource) for resource in event.get('resources', [])))}</td>"
                "</tr>"
            )
    if not rows:
        rows.append("    <tr><td colspan=\"8\" class=\"muted\">No simulated events</td></tr>")
    return (
        "  <table>\n"
        "    <thead><tr><th>algorithm</th><th>bytes</th><th>rank</th><th>op</th><th>start_us</th>"
        "<th>end_us</th><th>wait_us</th><th>resources</th></tr></thead>\n"
        f"    <tbody>\n{chr(10).join(rows)}\n    </tbody>\n"
        "  </table>"
    )


def _events(result: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    events = result.get("events", [])
    if not isinstance(events, list):
        return ()
    return tuple(event for event in events if isinstance(event, Mapping))


def _float_cell(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return _escape(value)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))
