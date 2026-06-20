import csv
import html
import json
import re
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
    write_report_bundle_from_plain(result_data, path)


def write_report_bundle(results: Sequence[SimulationResult], out_dir: Path) -> None:
    write_report_bundle_from_plain([dataclass_to_plain(result) for result in results], out_dir)


def write_report_bundle_from_plain(result_data: Any, out: Path) -> None:
    items = _as_result_items(result_data)
    index_path = _index_path(out)
    rank_dir = index_path.parent / "rank_reports"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(_render_index_html(items, rank_report_prefix="rank_reports"), encoding="utf-8")
    back_href = f"../{index_path.name}"
    for rank in _rank_ids(items):
        rank_path = rank_dir / f"rank_{rank:03d}.html"
        rank_path.write_text(_render_rank_html(items, rank, back_href), encoding="utf-8")


def _as_result_items(result_data: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(result_data, Mapping):
        return (result_data,)
    if isinstance(result_data, list):
        return tuple(item for item in result_data if isinstance(item, Mapping))
    raise ValueError("result data must be a result object or a list of result objects")


def _index_path(out: Path) -> Path:
    if out.suffix == ".html":
        return out
    return out / "report.html"


def _render_index_html(results: Sequence[Mapping[str, Any]], rank_report_prefix: str = "") -> str:
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
        "  <p class=\"muted\">Per-rank files contain the detailed congestion drilldown and timeline.</p>\n"
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


def _render_rank_html(results: Sequence[Mapping[str, Any]], rank: int, back_href: str) -> str:
    payload = _rank_payload(results, rank)
    title = f"TileXR Rank {rank} Timeline"
    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        f"  <title>{_escape(title)}</title>\n"
        "  <style>\n"
        "    body { font-family: Arial, sans-serif; margin: 24px; color: #222; }\n"
        "    .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 12px 0; }\n"
        "    select, input, label { font: inherit; }\n"
        "    #timelineWrap { overflow-x: auto; border: 1px solid #d0d7de; margin: 12px 0 20px; }\n"
        "    #timelineSvg { min-width: 900px; display: block; background: #fff; }\n"
        "    table { border-collapse: collapse; width: 100%; margin: 12px 0 28px; }\n"
        "    th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }\n"
        "    th { background: #f2f4f7; }\n"
        "    .muted { color: #666; }\n"
        "    .metric { display: inline-block; margin-right: 16px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>Rank/Card {rank} Timeline</h1>\n"
        f"  <p><a href=\"{_escape(back_href)}\">Back to report</a></p>\n"
        "  <div class=\"toolbar\">\n"
        "    <label for=\"resultSelect\">Result</label>\n"
        "    <select id=\"resultSelect\"></select>\n"
        "    <label for=\"zoomRange\">Zoom</label>\n"
        "    <input id=\"zoomRange\" type=\"range\" min=\"1\" max=\"8\" value=\"2\">\n"
        "    <label><input type=\"checkbox\" class=\"opFilter\" value=\"copy\" checked> copy</label>\n"
        "    <label><input type=\"checkbox\" class=\"opFilter\" value=\"send\" checked> send</label>\n"
        "    <label><input type=\"checkbox\" class=\"opFilter\" value=\"recv\" checked> recv</label>\n"
        "    <label><input type=\"checkbox\" class=\"opFilter\" value=\"put\" checked> put</label>\n"
        "    <label><input type=\"checkbox\" class=\"opFilter\" value=\"get\" checked> get</label>\n"
        "  </div>\n"
        "  <div id=\"summary\"></div>\n"
        "  <div id=\"timelineWrap\"><svg id=\"timelineSvg\" role=\"img\"></svg></div>\n"
        "  <table>\n"
        "    <thead><tr><th>op</th><th>type</th><th>owner_rank</th><th>start_us</th>"
        "<th>end_us</th><th>wait_us</th><th>transfer_us</th><th>effective_gbps</th>"
        "<th>bottleneck</th><th>resources</th></tr></thead>\n"
        "    <tbody id=\"eventRows\"></tbody>\n"
        "  </table>\n"
        f"  <script id=\"timeline-data\" type=\"application/json\">{_json_script(payload)}</script>\n"
        "  <script>\n"
        "    const payload = JSON.parse(document.getElementById('timeline-data').textContent);\n"
        "    const resultSelect = document.getElementById('resultSelect');\n"
        "    const zoomRange = document.getElementById('zoomRange');\n"
        "    const timelineSvg = document.getElementById('timelineSvg');\n"
        "    const eventRows = document.getElementById('eventRows');\n"
        "    const summary = document.getElementById('summary');\n"
        "    const colors = { copy: '#5b8def', send: '#d95f02', recv: '#1b9e77', put: '#7570b3', get: '#e7298a' };\n"
        "    function fmt(value) { return Number(value || 0).toFixed(3); }\n"
        "    function esc(value) { return String(value == null ? '' : value).replace(/[&<>\\\"]/g, function(ch) { return {'&':'&amp;','<':'&lt;','>':'&gt;','\\\"':'&quot;'}[ch]; }); }\n"
        "    function checkedOps() { return new Set(Array.from(document.querySelectorAll('.opFilter')).filter(function(item) { return item.checked; }).map(function(item) { return item.value; })); }\n"
        "    function selectedResult() { return payload.results[Number(resultSelect.value || 0)] || {events: []}; }\n"
        "    function draw() {\n"
        "      const result = selectedResult();\n"
        "      const ops = checkedOps();\n"
        "      const events = result.events.filter(function(event) { return ops.has(event.op_type); });\n"
        "      const maxEnd = Math.max(1, ...events.map(function(event) { return Number(event.end_us || 0); }));\n"
        "      const zoom = Number(zoomRange.value || 1);\n"
        "      const width = 900 * zoom;\n"
        "      const rowHeight = 24;\n"
        "      const height = Math.max(80, 36 + events.length * rowHeight);\n"
        "      timelineSvg.setAttribute('width', String(width));\n"
        "      timelineSvg.setAttribute('height', String(height));\n"
        "      timelineSvg.innerHTML = '';\n"
        "      const ns = 'http://www.w3.org/2000/svg';\n"
        "      function svgEl(name) { return document.createElementNS(ns, name); }\n"
        "      const axis = svgEl('line'); axis.setAttribute('x1', '90'); axis.setAttribute('x2', String(width - 20)); axis.setAttribute('y1', '22'); axis.setAttribute('y2', '22'); axis.setAttribute('stroke', '#8c959f'); timelineSvg.appendChild(axis);\n"
        "      events.forEach(function(event, index) {\n"
        "        const y = 36 + index * rowHeight;\n"
        "        const x = 90 + Number(event.start_us || 0) / maxEnd * (width - 130);\n"
        "        const w = Math.max(2, (Number(event.end_us || 0) - Number(event.start_us || 0)) / maxEnd * (width - 130));\n"
        "        const label = svgEl('text'); label.setAttribute('x', '8'); label.setAttribute('y', String(y + 12)); label.setAttribute('font-size', '11'); label.textContent = event.op_type + ' r' + event.rank; timelineSvg.appendChild(label);\n"
        "        const rect = svgEl('rect'); rect.setAttribute('x', String(x)); rect.setAttribute('y', String(y)); rect.setAttribute('width', String(w)); rect.setAttribute('height', '14'); rect.setAttribute('fill', colors[event.op_type] || '#586069'); rect.appendChild(svgEl('title')).textContent = event.op_id + ' ' + fmt(event.start_us) + '-' + fmt(event.end_us) + ' us'; timelineSvg.appendChild(rect);\n"
        "      });\n"
        "      summary.innerHTML = '<span class=\"metric\">events: ' + events.length + '</span><span class=\"metric\">latency_us: ' + fmt(result.latency_us) + '</span><span class=\"metric\">message_bytes: ' + esc(result.message_bytes) + '</span>';\n"
        "      eventRows.innerHTML = events.map(function(event) { return '<tr><td>' + esc(event.op_id) + '</td><td>' + esc(event.op_type) + '</td><td>' + esc(event.rank) + '</td><td>' + fmt(event.start_us) + '</td><td>' + fmt(event.end_us) + '</td><td>' + fmt(event.wait_us) + '</td><td>' + fmt(event.transfer_us) + '</td><td>' + fmt(event.effective_gbps) + '</td><td>' + esc(event.bottleneck_resource) + '</td><td>' + esc((event.resources || []).join(', ')) + '</td></tr>'; }).join('') || '<tr><td colspan=\"10\" class=\"muted\">No events for this filter</td></tr>';\n"
        "    }\n"
        "    payload.results.forEach(function(result, index) { const option = document.createElement('option'); option.value = String(index); option.textContent = result.label; resultSelect.appendChild(option); });\n"
        "    resultSelect.addEventListener('change', draw);\n"
        "    zoomRange.addEventListener('input', draw);\n"
        "    document.querySelectorAll('.opFilter').forEach(function(item) { item.addEventListener('change', draw); });\n"
        "    draw();\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )


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


def _rank_payload(results: Sequence[Mapping[str, Any]], rank: int) -> Mapping[str, Any]:
    payload_results = []
    for result in results:
        events = [event for event in _events(result) if _event_involves_rank(event, rank)]
        payload_results.append(
            {
                "label": _result_label(result),
                "algorithm": result.get("algorithm"),
                "collective": result.get("collective"),
                "rank_count": result.get("rank_count"),
                "message_bytes": result.get("message_bytes"),
                "latency_us": result.get("latency_us"),
                "algbw_gbps": result.get("algbw_gbps"),
                "busbw_gbps": result.get("busbw_gbps"),
                "bottleneck_resource": result.get("bottleneck_resource"),
                "events": sorted(events, key=lambda event: (_float_value(event.get("start_us")), _float_value(event.get("end_us")), str(event.get("op_id")))),
            }
        )
    return {"rank": rank, "results": payload_results}


def _result_label(result: Mapping[str, Any]) -> str:
    return (
        f"{result.get('algorithm', '')} "
        f"{result.get('rank_count', '')}P "
        f"{result.get('message_bytes', '')}B"
    ).strip()


def _event_involves_rank(event: Mapping[str, Any], rank: int) -> bool:
    try:
        if int(event.get("rank")) == rank:
            return True
    except (TypeError, ValueError):
        pass
    return any(_resource_involves_rank(str(resource), rank) for resource in event.get("resources", []))


def _resource_involves_rank(resource: str, rank: int) -> bool:
    if resource in {f"sdma:rank{rank}", f"uplink:src:{rank}", f"uplink:dst:{rank}"}:
        return True
    match = re.search(r":(\d+)->(\d+)$", resource)
    if match:
        return rank in {int(match.group(1)), int(match.group(2))}
    return False


def _json_script(value: Any) -> str:
    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


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
