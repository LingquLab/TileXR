import csv
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

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

PERFETTO_UI_URL = "https://ui.perfetto.dev/"


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
    write_report_bundle_from_plain(result_data, path)


def write_report_bundle(results: Sequence[SimulationResult], out_dir: Path) -> None:
    write_report_bundle_from_plain([dataclass_to_plain(result) for result in results], out_dir)


def write_report_bundle_from_plain(result_data: Any, out: Path) -> None:
    items = _as_result_items(result_data)
    index_path = _index_path(out)
    rank_dir = index_path.parent / "rank_reports"
    profile_dir = index_path.parent / "profiles"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _reset_generated_dir(profile_dir)
    _remove_generated_dir(index_path.parent / "rank_traces")
    _reset_generated_dir(rank_dir)
    profiles = _write_profile_traces(items, profile_dir)
    index_path.write_text(
        _render_index_html(items, rank_report_prefix="rank_reports", profiles=profiles),
        encoding="utf-8",
    )
    back_href = f"../{index_path.name}"
    for rank in _rank_ids(items):
        rank_path = rank_dir / f"rank_{rank:03d}.html"
        rank_path.write_text(
            _render_rank_html(items, rank, back_href, profiles),
            encoding="utf-8",
        )


def _as_result_items(result_data: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(result_data, Mapping):
        return (result_data,)
    if isinstance(result_data, list):
        return tuple(item for item in result_data if isinstance(item, Mapping))
    raise ValueError("result data must be a result object or a list of result objects")


def _reset_generated_dir(path: Path) -> None:
    _remove_generated_dir(path)
    path.mkdir(parents=True, exist_ok=True)


def _remove_generated_dir(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def _index_path(out: Path) -> Path:
    if out.suffix == ".html":
        return out
    return out / "report.html"


def _render_index_html(
    results: Sequence[Mapping[str, Any]],
    rank_report_prefix: str = "",
    profiles: Sequence[Mapping[str, Any]] = (),
) -> str:
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
        "    .toolbar { display: flex; align-items: center; gap: 8px; margin: 12px 0 24px; }\n"
        "    select, button { font: inherit; padding: 5px 8px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>TileXR Collective Simulation Report</h1>\n"
        "  <p class=\"muted\">Contention model: deterministic reservation batches. "
        "Same-start operations fair-share matching resources, then reserve them until completion; "
        "staggered dynamic contention is not modeled.</p>\n"
        "  <h2>Algorithm Selection</h2>\n"
        f"{_render_algorithm_table(results)}\n"
        "  <h2>Congestion Drilldown</h2>\n"
        f"{_render_bottleneck_summary(results)}\n"
        "  <h2>Rank Timeline</h2>\n"
        f"{_render_rank_navigation(results, rank_report_prefix)}\n"
        "  <h2>Perfetto Profiles</h2>\n"
        f"{_render_profile_table(profiles)}\n"
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


def _render_bottleneck_summary(results: Sequence[Mapping[str, Any]]) -> str:
    resources = {}
    for result in results:
        for event in _events(result):
            resource = str(event.get("bottleneck_resource") or "")
            if not resource:
                continue
            stats = resources.setdefault(resource, {"count": 0, "transfer_us": 0.0, "wait_us": 0.0})
            stats["count"] += 1
            stats["transfer_us"] += _float_value(event.get("transfer_us"))
            stats["wait_us"] += _float_value(event.get("wait_us"))
    rows = []
    for resource, stats in sorted(
        resources.items(),
        key=lambda item: (item[1]["wait_us"], item[1]["transfer_us"], item[1]["count"]),
        reverse=True,
    )[:50]:
        rows.append(
            "    <tr>"
            f"<td>{_escape(resource)}</td>"
            f"<td>{_escape(stats['count'])}</td>"
            f"<td>{_float_cell(stats['wait_us'])}</td>"
            f"<td>{_float_cell(stats['transfer_us'])}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("    <tr><td colspan=\"4\" class=\"muted\">No simulated events</td></tr>")
    return (
        "  <table>\n"
        "    <caption>Bottleneck Summary</caption>\n"
        "    <thead><tr><th>resource</th><th>events</th><th>total_wait_us</th>"
        "<th>total_transfer_us</th></tr></thead>\n"
        f"    <tbody>\n{chr(10).join(rows)}\n    </tbody>\n"
        "  </table>"
    )


def _render_profile_table(profiles: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for profile in profiles:
        rows.append(
            "    <tr>"
            f"<td>{_escape(profile.get('label'))}</td>"
            f"<td>{_escape(profile.get('collective'))}</td>"
            f"<td>{_escape(profile.get('rank_count'))}</td>"
            f"<td>{_escape(profile.get('message_bytes'))}</td>"
            f"<td>{_escape(profile.get('send_events'))}</td>"
            f"<td><a href=\"{_escape(profile.get('href'))}\">{_escape(profile.get('filename'))}</a></td>"
            "</tr>"
        )
    if not rows:
        rows.append("    <tr><td colspan=\"6\" class=\"muted\">No profiles available</td></tr>")
    return (
        "  <table>\n"
        "    <thead><tr><th>profile</th><th>collective</th><th>ranks</th><th>bytes</th>"
        "<th>send_events</th><th>trace</th></tr></thead>\n"
        f"    <tbody>\n{chr(10).join(rows)}\n    </tbody>\n"
        "  </table>"
    )


def _render_rank_navigation(results: Sequence[Mapping[str, Any]], rank_report_prefix: str) -> str:
    ranks = _rank_ids(results)
    if not ranks:
        return "  <p class=\"muted\">No ranks available</p>"
    options = []
    links = []
    for rank in ranks:
        href = _rank_href(rank, rank_report_prefix)
        options.append(f"      <option value=\"{_escape(href)}\">rank/card {rank}</option>")
        links.append(f"      <a href=\"{_escape(href)}\">rank/card {rank}</a>")
    return (
        "  <div class=\"toolbar\">\n"
        "    <label for=\"rankSelect\">Rank/Card</label>\n"
        "    <select id=\"rankSelect\">\n"
        f"{chr(10).join(options)}\n"
        "    </select>\n"
        "    <button type=\"button\" onclick=\"openSelectedRank()\">Open</button>\n"
        "  </div>\n"
        "  <p class=\"muted\">Per-rank files link to per-test Perfetto profiles and show send ownership for that rank/card.</p>\n"
        "  <div class=\"rank-links\">\n"
        f"{chr(10).join(links)}\n"
        "  </div>\n"
        "  <script>\n"
        "    function openSelectedRank() {\n"
        "      var select = document.getElementById('rankSelect');\n"
        "      if (select && select.value) window.location.href = select.value;\n"
        "    }\n"
        "  </script>"
    )


def _render_rank_html(
    results: Sequence[Mapping[str, Any]],
    rank: int,
    back_href: str,
    profiles: Sequence[Mapping[str, Any]],
) -> str:
    title = f"TileXR Rank {rank} Perfetto Trace"
    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        f"  <title>{_escape(title)}</title>\n"
        "  <style>\n"
        "    body { font-family: Arial, sans-serif; margin: 24px; color: #222; }\n"
        "    table { border-collapse: collapse; width: 100%; margin: 12px 0 28px; }\n"
        "    th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }\n"
        "    th { background: #f2f4f7; }\n"
        "    .muted { color: #666; }\n"
        "    .links { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 24px; }\n"
        "    .links a { border: 1px solid #d0d7de; color: #0969da; padding: 7px 10px; text-decoration: none; }\n"
        "    .links a:hover { background: #f6f8fa; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>Rank/Card {rank} Perfetto Trace</h1>\n"
        f"  <p><a href=\"{_escape(back_href)}\">Back to report</a></p>\n"
        "  <p class=\"muted\">Each Perfetto trace is one collective test only. "
        "Open a profile in Perfetto UI and select a slice to inspect its event arguments.</p>\n"
        "  <div class=\"links\">\n"
        f"    <a href=\"{_escape(PERFETTO_UI_URL)}\" target=\"_blank\" rel=\"noopener\">Open Perfetto UI</a>\n"
        "  </div>\n"
        "  <h2>Rank/Card Profiles</h2>\n"
        f"{_render_rank_trace_summary(results, rank, profiles)}\n"
        "</body>\n"
        "</html>\n"
    )


def _render_rank_trace_summary(
    results: Sequence[Mapping[str, Any]],
    rank: int,
    profiles: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for result, profile in zip(results, profiles):
        send_events = _rank_send_events(result, rank)
        rows.append(
            "    <tr>"
            f"<td>{_escape(_result_label(result))}</td>"
            f"<td>{_escape(result.get('collective'))}</td>"
            f"<td>{_escape(result.get('rank_count'))}</td>"
            f"<td>{_escape(result.get('message_bytes'))}</td>"
            f"<td>{_escape(len(send_events))}</td>"
            f"<td>{_float_cell(result.get('latency_us'))}</td>"
            f"<td>{_escape(result.get('bottleneck_resource'))}</td>"
            f"<td><a href=\"{_escape(profile.get('rank_href'))}\">{_escape(profile.get('filename'))}</a></td>"
            "</tr>"
        )
    if not rows:
        rows.append("    <tr><td colspan=\"8\" class=\"muted\">No results</td></tr>")
    return (
        "  <table>\n"
        "    <thead><tr><th>result</th><th>collective</th><th>ranks</th><th>bytes</th>"
        "<th>rank_send_events</th><th>latency_us</th><th>bottleneck</th><th>profile</th></tr></thead>\n"
        f"    <tbody>\n{chr(10).join(rows)}\n    </tbody>\n"
        "  </table>"
    )


def _write_profile_traces(
    results: Sequence[Mapping[str, Any]],
    profile_dir: Path,
) -> Sequence[Mapping[str, Any]]:
    profiles = []
    for index, result in enumerate(results):
        filename = _profile_filename(index, result)
        trace_path = profile_dir / filename
        trace_path.write_text(json.dumps(_profile_trace_payload(result), indent=2) + "\n", encoding="utf-8")
        profiles.append(
            {
                "index": index,
                "filename": filename,
                "href": f"profiles/{filename}",
                "rank_href": f"../profiles/{filename}",
                "label": _result_label(result),
                "collective": result.get("collective"),
                "rank_count": result.get("rank_count"),
                "message_bytes": result.get("message_bytes"),
                "send_events": len(_send_events(result)),
            }
        )
    return tuple(profiles)


def _profile_trace_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    trace_events = []
    process_ids = {}
    lane_ids = {}
    next_tid_by_pid = {}

    def lane_id(process_name: str, thread_name: str) -> Tuple[int, int]:
        if process_name not in process_ids:
            pid = len(process_ids) + 1
            process_ids[process_name] = pid
            next_tid_by_pid[pid] = 1
            trace_events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "args": {"name": process_name},
                }
            )
        pid = process_ids[process_name]
        key = (pid, thread_name)
        if key not in lane_ids:
            tid = next_tid_by_pid[pid]
            next_tid_by_pid[pid] = tid + 1
            lane_ids[key] = tid
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": thread_name},
                }
            )
        return pid, lane_ids[key]

    for event in _send_events(result):
        rank = _event_rank(event)
        if rank is None:
            continue
        process_name, thread_name, dst_rank = _trace_lane(event, rank)
        pid, tid = lane_id(process_name, thread_name)
        start_us = _float_value(event.get("start_us"))
        end_us = _float_value(event.get("end_us"))
        resources = tuple(str(resource) for resource in event.get("resources", []))
        trace_events.append(
            {
                "name": str(event.get("op_id") or "send"),
                "cat": "send",
                "ph": "X",
                "ts": start_us,
                "dur": max(0.0, end_us - start_us),
                "pid": pid,
                "tid": tid,
                "args": {
                    "algorithm": result.get("algorithm"),
                    "collective": result.get("collective"),
                    "rank_count": result.get("rank_count"),
                    "owner_rank": rank,
                    "destination_rank": dst_rank,
                    "message_bytes": result.get("message_bytes"),
                    "event_message_bytes": event.get("message_bytes"),
                    "start_us": start_us,
                    "end_us": end_us,
                    "wait_us": _float_value(event.get("wait_us")),
                    "transfer_us": _float_value(event.get("transfer_us")),
                    "effective_gbps": _float_value(event.get("effective_gbps")),
                    "bottleneck_resource": event.get("bottleneck_resource"),
                    "resources": ", ".join(resources),
                },
            }
        )
    return {"traceEvents": trace_events, "displayTimeUnit": "us"}


def _send_events(result: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    return tuple(
        sorted(
            (
                event
                for event in _events(result)
                if event.get("op_type") == "send" and _event_rank(event) is not None
            ),
            key=lambda event: (
                _event_rank(event),
                _float_value(event.get("start_us")),
                _float_value(event.get("end_us")),
                str(event.get("op_id")),
            ),
        )
    )


def _rank_send_events(result: Mapping[str, Any], rank: int) -> Sequence[Mapping[str, Any]]:
    return tuple(event for event in _send_events(result) if _event_rank(event) == rank)


def _trace_lane(event: Mapping[str, Any], rank: int) -> Tuple[str, str, Optional[int]]:
    resources = tuple(str(resource) for resource in event.get("resources", []))
    dst_rank = _send_destination(event)
    thread_name = f"send to rank{dst_rank}" if dst_rank is not None else "send"
    p2p_resource = next((resource for resource in resources if resource.startswith("p2p:")), None)
    if p2p_resource:
        group_name = p2p_resource
    elif any(resource.startswith("uplink:") or resource.startswith("clos:") for resource in resources):
        group_name = f"clos-link:rank{rank}"
    elif resources:
        group_name = resources[0]
    else:
        group_name = f"send:rank{rank}"
    return group_name, thread_name, dst_rank


def _send_destination(event: Mapping[str, Any]) -> Optional[int]:
    for resource in event.get("resources", []):
        text = str(resource)
        if text.startswith("uplink:dst:"):
            return _int_or_none(text.rsplit(":", 1)[-1])
        p2p_match = re.search(r"->(\d+)$", text)
        if p2p_match:
            return int(p2p_match.group(1))
    op_match = re.search(r"_to_r(\d+)(?:\D|$)", str(event.get("op_id") or ""))
    if op_match:
        return int(op_match.group(1))
    return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_rank(event: Mapping[str, Any]) -> Optional[int]:
    return _int_or_none(event.get("rank"))


def _events(result: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    events = result.get("events", [])
    if not isinstance(events, list):
        return ()
    return tuple(event for event in events if isinstance(event, Mapping))


def _rank_ids(results: Sequence[Mapping[str, Any]]) -> Sequence[int]:
    max_rank = -1
    for result in results:
        try:
            max_rank = max(max_rank, int(result.get("rank_count", 0)) - 1)
        except (TypeError, ValueError):
            pass
        for event in _events(result):
            try:
                max_rank = max(max_rank, int(event.get("rank")))
            except (TypeError, ValueError):
                pass
    return tuple(range(max_rank + 1))


def _rank_href(rank: int, prefix: str) -> str:
    filename = f"rank_{rank:03d}.html"
    if not prefix:
        return filename
    return f"{prefix}/{filename}"


def _profile_filename(index: int, result: Mapping[str, Any]) -> str:
    algorithm = _slug(result.get("algorithm") or "profile")
    rank_count = _int_or_none(result.get("rank_count"))
    message_bytes = _int_or_none(result.get("message_bytes"))
    rank_part = f"{rank_count}p" if rank_count is not None else "unknownp"
    byte_part = f"{message_bytes}b" if message_bytes is not None else "unknownb"
    return f"profile_{index:03d}_{algorithm}_{rank_part}_{byte_part}.trace.json"


def _result_label(result: Mapping[str, Any]) -> str:
    return (
        f"{result.get('algorithm', '')} "
        f"{result.get('rank_count', '')}P "
        f"{result.get('message_bytes', '')}B"
    ).strip()


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "profile"


def _float_cell(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return _escape(value)


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))
