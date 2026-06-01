#!/usr/bin/env python3

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path


TRACE_SCHEMA = "tilexr_perf_trace_report.v1"
RUN_SCHEMA = "tilexr_perf_trace_run.v1"
RANK_LAUNCH_RE = re.compile(r"rank([0-9]+)[/\\]launch([0-9]+)[/\\]trace\.json$")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate TileXR collective profile traces")
    parser.add_argument("profile_dir")
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--profile-sample-every", type=int, default=1)
    parser.add_argument("--emit-ai-prompt", action="store_true")
    return parser.parse_args()


def relpath(path, root):
    return path.relative_to(root).as_posix()


def parse_rank_launch(path, root):
    match = RANK_LAUNCH_RE.search(relpath(path, root))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def load_traces(root):
    traces = []
    diagnostics = []

    for path in sorted(root.glob("rank*/launch*/trace.json")):
        parsed = parse_rank_launch(path, root)
        if parsed is None:
            diagnostics.append(f"ignored unrecognized trace path {relpath(path, root)}")
            continue

        rank, launch_id = parsed
        source = relpath(path, root)
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            diagnostics.append(f"invalid json in {source}: {exc}")
            continue

        if not isinstance(trace, dict):
            diagnostics.append(f"invalid top-level trace type in {source}: expected object")
            continue

        if trace.get("schema") != TRACE_SCHEMA:
            diagnostics.append(f"invalid schema in {source}")
            continue

        trace = sanitize_trace(trace, source, diagnostics)
        traces.append({"rank": rank, "launch_id": launch_id, "path": path, "trace": trace})

    return traces, diagnostics


def sanitize_trace(trace, source, diagnostics):
    trace = dict(trace)
    stats = trace.get("stats", [])
    if not isinstance(stats, list):
        diagnostics.append(f"invalid stats in {source}: expected list")
        trace["stats"] = []
        return trace

    valid_stats = []
    for index, stat in enumerate(stats):
        if not isinstance(stat, dict):
            diagnostics.append(f"invalid stat entry in {source} stats[{index}]: expected object")
            continue
        valid_stats.append(stat)
    trace["stats"] = valid_stats
    return trace


def group_key(entry):
    trace = entry["trace"]
    return (
        trace.get("op_type"),
        trace.get("op_name", "Unknown"),
        trace.get("rank_size"),
        trace.get("message_bytes"),
        trace.get("stage_count"),
        trace.get("cycle_to_us_divisor"),
    )


def expected_launch_ids(measured_iters, sample_every):
    if measured_iters <= 0 or sample_every <= 0:
        return []
    return [launch_id for launch_id in range(measured_iters) if launch_id % sample_every == 0]


def expected_group_launch_ids(observed_launch_ids, measured_iters, sample_every):
    sampled_offsets = expected_launch_ids(measured_iters, sample_every)
    if not observed_launch_ids or not sampled_offsets:
        return observed_launch_ids

    first_launch = min(observed_launch_ids)
    launch_base = (first_launch // measured_iters) * measured_iters
    return [launch_base + offset for offset in sampled_offsets]


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cycles_to_us(cycles, divisor):
    if divisor == 0:
        return 0.0
    return float(cycles) / float(divisor)


def normalized_bars(entries, root):
    bars = []
    entries_by_rank_launch = defaultdict(list)
    for entry in entries:
        entries_by_rank_launch[(entry["rank"], entry["launch_id"])].append(entry)

    for (rank, launch_id), rank_entries in sorted(entries_by_rank_launch.items()):
        all_stats = []
        for entry in rank_entries:
            all_stats.extend(entry["trace"].get("stats", []))

        nonzero_starts = [
            as_int(stat.get("first_start_cycle"))
            for stat in all_stats
            if as_int(stat.get("first_start_cycle")) > 0
        ]
        zero_cycle = min(nonzero_starts) if nonzero_starts else 0

        for entry in rank_entries:
            divisor = as_int(entry["trace"].get("cycle_to_us_divisor"))
            source = relpath(entry["path"], root)
            drilldown = relpath(entry["path"].with_name("report.html"), root)
            for stat in entry["trace"].get("stats", []):
                count = as_int(stat.get("count"))
                first = as_int(stat.get("first_start_cycle"))
                last = as_int(stat.get("last_end_cycle"))
                if count == 0 and first == 0 and last == 0:
                    continue

                start_cycles = max(0, first - zero_cycle)
                end_cycles = max(start_cycles, last - zero_cycle)
                duration_cycles = max(0, last - first)
                core = as_int(stat.get("core"))
                stat_rank = as_int(stat.get("rank"), rank)
                raw_cycles = as_int(stat.get("raw_cycles"))

                bars.append({
                    "launch_id": launch_id,
                    "rank": stat_rank,
                    "core": core,
                    "stage": str(stat.get("stage", "unknown")),
                    "stage_id": as_int(stat.get("stage_id")),
                    "start_cycles": start_cycles,
                    "end_cycles": end_cycles,
                    "duration_cycles": duration_cycles,
                    "start_us": cycles_to_us(start_cycles, divisor),
                    "end_us": cycles_to_us(end_cycles, divisor),
                    "duration_us": cycles_to_us(duration_cycles, divisor),
                    "sum_us": as_float(stat.get("sum_us"), cycles_to_us(raw_cycles, divisor)),
                    "raw_cycles": raw_cycles,
                    "count": count,
                    "max_cycles": as_int(stat.get("max_cycles")),
                    "source": source,
                    "drilldown": drilldown,
                    "lane": f"rank{stat_rank}/core{core}",
                })

    return sorted(bars, key=lambda bar: (
        bar["launch_id"],
        bar["rank"],
        bar["core"],
        bar["start_cycles"],
        bar["stage_id"],
        bar["stage"],
    ))


def summarize_group(group):
    stage_totals = defaultdict(float)
    launch_kernel = defaultdict(float)
    rank_core_max = {"rank": 0, "core": 0, "stage": "", "duration_us": 0.0, "max_cycles": 0, "max_us": 0.0}

    for bar in group["bars"]:
        stage_totals[bar["stage"]] += bar["sum_us"]
        if bar["stage"] == "kernel_total":
            launch_kernel[bar["launch_id"]] = max(launch_kernel[bar["launch_id"]], bar["duration_us"])
        max_us = cycles_to_us(bar["max_cycles"], group["cycle_to_us_divisor"])
        if max_us > rank_core_max["max_us"]:
            rank_core_max = {
                "rank": bar["rank"],
                "core": bar["core"],
                "stage": bar["stage"],
                "duration_us": bar["duration_us"],
                "max_cycles": bar["max_cycles"],
                "max_us": max_us,
            }

    top_stage = max(stage_totals.items(), key=lambda item: item[1]) if stage_totals else ("none", 0.0)
    slowest_launch = max(launch_kernel.items(), key=lambda item: item[1]) if launch_kernel else (None, 0.0)

    return {
        "top_stage": {"stage": top_stage[0], "sum_us": top_stage[1]},
        "slowest_launch": {"launch_id": slowest_launch[0], "kernel_us": slowest_launch[1]},
        "rank_core_max": rank_core_max,
    }


def build_index(root, args):
    traces, diagnostics = load_traces(root)
    grouped = defaultdict(list)
    for entry in traces:
        grouped[group_key(entry)].append(entry)

    add_incompatible_group_diagnostics(grouped, root, diagnostics)
    groups = []

    def sort_key(item):
        key = item[0]
        return (
            key[1] or "",
            as_int(key[2]),
            as_int(key[3]),
            as_int(key[4]),
            as_int(key[5]),
            as_int(key[0]),
        )

    for key, entries in sorted(grouped.items(), key=sort_key):
        op_type, op_name, rank_size, message_bytes, stage_count, divisor = key
        rank_ids = sorted({entry["rank"] for entry in entries})
        launch_ids = sorted({entry["launch_id"] for entry in entries})
        effective_rank_size = as_int(rank_size, len(rank_ids))
        expected_ranks = list(range(effective_rank_size))
        expected_launches = expected_group_launch_ids(launch_ids, args.iters, args.profile_sample_every)

        for launch_id in expected_launches:
            for rank in expected_ranks:
                if not any(entry["rank"] == rank and entry["launch_id"] == launch_id for entry in entries):
                    diagnostics.append(f"missing trace for rank{rank} launch{launch_id}")

        group = {
            "op_type": op_type,
            "op_name": op_name,
            "rank_size": effective_rank_size,
            "message_bytes": as_int(message_bytes),
            "stage_count": as_int(stage_count),
            "cycle_to_us_divisor": as_int(divisor),
            "rank_ids": rank_ids,
            "launch_ids": launch_ids,
            "sources": [
                relpath(entry["path"], root)
                for entry in sorted(entries, key=lambda item: (item["launch_id"], item["rank"], relpath(item["path"], root)))
            ],
            "bars": normalized_bars(entries, root),
        }
        group["summary"] = summarize_group(group)
        groups.append(group)

    return {
        "schema": RUN_SCHEMA,
        "profile_dir": str(root),
        "warmup_iters": args.warmup_iters,
        "measured_iters": args.iters,
        "profile_sample_every": args.profile_sample_every,
        "groups": groups,
        "diagnostics": diagnostics,
    }


def add_incompatible_group_diagnostics(grouped, root, diagnostics):
    keys_by_launch = defaultdict(set)
    for key, entries in grouped.items():
        for entry in entries:
            keys_by_launch[entry["launch_id"]].add(key)

    for launch_id, keys in sorted(keys_by_launch.items()):
        if len(keys) <= 1:
            continue
        group_details = [
            describe_group(key, grouped[key], root)
            for key in sorted(keys, key=lambda item: (
                str(item[1]),
                as_int(item[2]),
                as_int(item[3]),
                as_int(item[4]),
                as_int(item[5]),
                as_int(item[0]),
            ))
        ]
        diagnostics.append(
            f"incompatible trace groups detected for launch{launch_id}: " + "; ".join(group_details)
        )


def describe_group(key, entries, root):
    op_type, op_name, rank_size, message_bytes, stage_count, divisor = key
    sources = ", ".join(
        relpath(entry["path"], root)
        for entry in sorted(entries, key=lambda item: (item["launch_id"], item["rank"], relpath(item["path"], root)))
    )
    return (
        f"op_type={op_type} op_name={op_name} rank_size={rank_size} "
        f"message_bytes={message_bytes} stage_count={stage_count} "
        f"cycle_to_us_divisor={divisor} sources=[{sources}]"
    )


def format_launches(launch_ids):
    if not launch_ids:
        return "none"
    return ", ".join(f"launch{launch_id}" for launch_id in launch_ids)


def format_slowest_launch(slowest):
    if slowest["launch_id"] is None:
        return "unavailable"
    return f"launch{slowest['launch_id']} at {slowest['kernel_us']:.3f} us"


def render_analysis(index):
    lines = [
        "# TileXR Collective Profile Run Analysis",
        "",
        f"- Warmup iterations: {index['warmup_iters']}",
        f"- Measured iterations: {index['measured_iters']}",
        f"- Profile sample every: {index['profile_sample_every']}",
        "- Note: cross-NPU raw cycle offsets are not assumed to be synchronized.",
        "",
    ]

    if index["diagnostics"]:
        lines.append("## Diagnostics")
        lines.extend(f"- {item}" for item in index["diagnostics"])
        lines.append("")

    if not index["groups"]:
        lines.append("No valid trace groups were found.")
        lines.append("")
        return "\n".join(lines)

    for group_index, group in enumerate(index["groups"]):
        summary = group["summary"]
        slowest = summary["slowest_launch"]
        rank_core = summary["rank_core_max"]
        stage_totals = defaultdict(float)
        for bar in group["bars"]:
            stage_totals[bar["stage"]] += bar["sum_us"]
        stage_summary = ", ".join(
            f"{stage}={total:.3f} us"
            for stage, total in sorted(stage_totals.items(), key=lambda item: (-item[1], item[0]))
        )
        lines.append(f"## Group {group_index}: {group['op_name']} message_bytes={group['message_bytes']}")
        lines.append(f"- Launches: {format_launches(group['launch_ids'])}")
        lines.append(f"- Slowest launch: {format_slowest_launch(slowest)}")
        lines.append(f"- Top stage: {summary['top_stage']['stage']} at {summary['top_stage']['sum_us']:.3f} us")
        lines.append(f"- Stage totals: {stage_summary}")
        lines.append(
            f"- Rank/core max: rank{rank_core['rank']} core{rank_core['core']} "
            f"{rank_core['stage']} max {rank_core['max_us']:.3f} us"
        )
        lines.append("")

    return "\n".join(lines)


def render_ai_prompt(index):
    lines = [
        "# TileXR collective profiling run",
        "",
        "Analyze this TileXR collective profiling run and suggest bottleneck investigation steps.",
        "",
        f"warmup iterations: {index['warmup_iters']}",
        f"measured iterations: {index['measured_iters']}",
        f"profile sample every: {index['profile_sample_every']}",
        "cross-NPU raw cycle offsets are not assumed to be synchronized",
        "",
    ]

    for group in index["groups"]:
        summary = group["summary"]
        slowest = summary["slowest_launch"]
        lines.append(f"- {group['op_name']} bytes={group['message_bytes']} launches={group['launch_ids']}")
        if slowest["launch_id"] is None:
            lines.append("  slowest_launch=unavailable")
        else:
            lines.append(f"  slowest_launch=launch{slowest['launch_id']} kernel_us={slowest['kernel_us']:.3f}")
        lines.append(f"  top_stage={summary['top_stage']['stage']} sum_us={summary['top_stage']['sum_us']:.3f}")

    if index["diagnostics"]:
        lines.append("")
        lines.append("diagnostics:")
        lines.extend(f"- {item}" for item in index["diagnostics"])

    return "\n".join(lines) + "\n"


def json_for_script(value):
    return (
        json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_html(index):
    data = json_for_script(index)
    summary_items = []
    fallback_rows = []

    for group in index["groups"]:
        summary = group["summary"]
        slowest = summary["slowest_launch"]
        rank_core = summary["rank_core_max"]
        summary_items.append(
            "<li>"
            f"{html.escape(str(group['op_name']))} bytes={group['message_bytes']}: "
            f"Slowest launch {html.escape(format_slowest_launch(slowest))}; "
            f"Top stage {html.escape(str(summary['top_stage']['stage']))} {summary['top_stage']['sum_us']:.3f} us; "
            f"Rank/core max rank{rank_core['rank']} core{rank_core['core']} "
            f"{html.escape(str(rank_core['stage']))} max {rank_core['max_us']:.3f} us"
            "</li>"
        )

        for bar in group["bars"]:
            fallback_rows.append(
                "<tr>"
                f"<td>launch{bar['launch_id']}</td>"
                f"<td>rank{bar['rank']}/core{bar['core']}</td>"
                f"<td>{html.escape(str(bar['stage']))}</td>"
                f"<td>{bar['duration_us']:.3f}</td>"
                f"<td>{bar['sum_us']:.3f}</td>"
                f"<td><a href=\"{html.escape(bar['drilldown'])}\">open launch report</a></td>"
                "</tr>"
            )

    if not summary_items:
        summary_items.append("<li>No valid trace groups were found.</li>")

    diagnostics = "".join(f"<li>{html.escape(item)}</li>" for item in index["diagnostics"])
    summary_html = "".join(summary_items) + diagnostics
    fallback_html = "".join(fallback_rows) or "<tr><td colspan=\"6\">No trace bars available.</td></tr>"

    stage_filter_html = "\n".join(
        f"<label><input type=\"checkbox\" checked onchange=\"toggleStage('{name}', this.checked)\"> {name}</label>"
        for name in ("wait", "copy", "sync", "total")
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TileXR Collective Profile Run Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#172033;background:#f8fafc}}
button{{margin-right:8px;padding:6px 10px;border:1px solid #94a3b8;background:#fff;border-radius:4px;cursor:pointer}}
.panel{{background:#fff;border:1px solid #cbd5e1;border-radius:6px;padding:16px;margin:16px 0}}
.timeline-wrap{{overflow:auto;border:1px solid #cbd5e1;background:#fff;height:520px;position:relative}}
.timeline{{position:relative;height:100%;min-width:100%;transform-origin:0 0}}
.bar{{position:absolute;height:16px;border-radius:3px;color:#0f172a;font-size:11px;overflow:hidden;white-space:nowrap;border:1px solid rgba(15,23,42,.25);text-decoration:none}}
.stage-total{{background:#cbd5e1}} .stage-wait{{background:#fca5a5}} .stage-copy{{background:#93c5fd}} .stage-sync{{background:#a7f3d0}}
.launch-line{{position:absolute;top:0;bottom:0;border-left:1px dashed #94a3b8;color:#475569;font-size:12px;padding-left:4px;text-decoration:none}}
table{{border-collapse:collapse;background:#fff;width:100%}} td,th{{border:1px solid #cbd5e1;padding:6px 8px}} th{{background:#e2e8f0;text-align:left}}
</style>
</head>
<body>
<h1>TileXR Collective Profile Run Report</h1>
<section class="panel">
<h2>Bottleneck First</h2>
<p>Warmup iterations: {index['warmup_iters']}; measured iterations: {index['measured_iters']}; sample every: {index['profile_sample_every']}.</p>
<ul>{summary_html}</ul>
</section>
<section class="panel">
<h2>Chronological Timeline</h2>
<p>Each rank/launch lane is normalized independently; cross-NPU raw cycle offsets are not assumed to be synchronized.</p>
<div>
<button onclick="zoomBy(1.25)">Zoom In</button>
<button onclick="zoomBy(0.8)">Zoom Out</button>
<button onclick="fitTimeline()">Fit</button>
<button onclick="toggleLaneMode()">Fold Cores</button>
<label>Launch <select id="launchFilter" onchange="renderTimeline()"><option value="all">all</option></select></label>
{stage_filter_html}
</div>
<div class="timeline-wrap" id="wrap"><div class="timeline" id="timeline"></div></div>
</section>
<section class="panel">
<h2>Fallback Table</h2>
<table><thead><tr><th>Launch</th><th>Lane</th><th>Stage</th><th>Duration us</th><th>Sum us</th><th>Drilldown</th></tr></thead><tbody>{fallback_html}</tbody></table>
</section>
<script>
const traceIndex = {data};
let scale = 1;
let foldCores = false;
const stageVisibility = {{wait: true, copy: true, sync: true, total: true}};
function category(stage) {{
  if (stage.includes('wait') || stage.includes('poll')) return 'wait';
  if (stage.includes('copy') || stage.includes('ipc')) return 'copy';
  if (stage.includes('sync') || stage.includes('barrier')) return 'sync';
  return 'total';
}}
function selectedLaunchId() {{
  const filter = document.getElementById('launchFilter');
  return filter ? filter.value : 'all';
}}
function visibleLaunches() {{
  const selectedLaunch = selectedLaunchId();
  const launches = [];
  for (const group of traceIndex.groups) {{
    for (const launchId of group.launch_ids) {{
      if (selectedLaunch !== 'all' && selectedLaunch !== String(launchId)) continue;
      const launchBars = group.bars.filter(bar => bar.launch_id === launchId);
      launches.push({{group, launchId, launchBars}});
    }}
  }}
  return launches;
}}
function timelineWidthAt(nextScale) {{
  const wrap = document.getElementById('wrap');
  const launchGap = 80;
  let xBase = 0;
  for (const launch of visibleLaunches()) {{
    const maxEnd = Math.max(1, ...launch.launchBars.map(bar => bar.end_us));
    xBase += Math.max(220, maxEnd * nextScale * 4) + launchGap;
  }}
  return Math.max(wrap.clientWidth, xBase);
}}
function launchDrilldown(launchBars) {{
  const preferred = launchBars.find(bar => bar.stage === 'kernel_total') || launchBars[0];
  return preferred ? preferred.drilldown : '#';
}}
function renderTimeline() {{
  const root = document.getElementById('timeline');
  const wrap = document.getElementById('wrap');
  root.innerHTML = '';
  refreshLaunchFilter();
  const laneHeight = 28;
  const launchGap = 80;
  let xBase = 0;
  const lanes = new Map();
  let laneCount = 0;
  for (const launch of visibleLaunches()) {{
    const launchId = launch.launchId;
    const launchBars = launch.launchBars;
    const maxEnd = Math.max(1, ...launchBars.map(bar => bar.end_us));
    const width = Math.max(220, maxEnd * scale * 4);
    const line = document.createElement('a');
    line.href = launchDrilldown(launchBars);
    line.className = 'launch-line';
    line.style.left = `${{xBase}}px`;
    line.textContent = `launch${{launchId}}`;
    root.appendChild(line);
    for (const bar of launchBars) {{
      const lane = foldCores ? `rank${{bar.rank}}` : `rank${{bar.rank}}/core${{bar.core}}`;
      if (!lanes.has(lane)) lanes.set(lane, laneCount++);
      const cat = category(bar.stage);
      const item = document.createElement('a');
      item.href = bar.drilldown;
      item.className = `bar stage-${{cat}}`;
      item.dataset.stageCategory = cat;
      item.style.display = stageVisibility[cat] ? '' : 'none';
      item.style.left = `${{xBase + bar.start_us * scale * 4}}px`;
      item.style.top = `${{24 + lanes.get(lane) * laneHeight}}px`;
      item.style.width = `${{Math.max(2, (bar.end_us - bar.start_us) * scale * 4)}}px`;
      item.title = `launch${{bar.launch_id}} ${{lane}} ${{bar.stage}} start=${{bar.start_us.toFixed(3)}}us end=${{bar.end_us.toFixed(3)}}us duration=${{bar.duration_us.toFixed(3)}}us sum=${{bar.sum_us.toFixed(3)}}us max_cycles=${{bar.max_cycles}} count=${{bar.count}} source=${{bar.source}}`;
      item.textContent = `${{lane}} ${{bar.stage}}`;
      root.appendChild(item);
    }}
    xBase += width + launchGap;
  }}
  root.style.width = `${{Math.max(wrap.clientWidth, xBase)}}px`;
  root.style.height = `${{Math.max(520, 60 + laneCount * laneHeight)}}px`;
}}
function zoomBy(factor) {{ scale = Math.max(0.1, Math.min(20, scale * factor)); renderTimeline(); }}
function fitTimeline() {{
  const wrap = document.getElementById('wrap');
  const target = Math.max(320, wrap.clientWidth - 16);
  let low = 0.1;
  let high = 20;
  if (timelineWidthAt(low) > target) {{
    scale = low;
  }} else {{
    for (let i = 0; i < 24; i++) {{
      const mid = (low + high) / 2;
      if (timelineWidthAt(mid) <= target) low = mid; else high = mid;
    }}
    scale = low;
  }}
  renderTimeline();
  wrap.scrollLeft = 0;
}}
function toggleLaneMode() {{ foldCores = !foldCores; renderTimeline(); }}
function refreshLaunchFilter() {{
  const select = document.getElementById('launchFilter');
  const current = select.value || 'all';
  const launches = new Set();
  for (const group of traceIndex.groups) for (const launchId of group.launch_ids) launches.add(String(launchId));
  select.innerHTML = '<option value="all">all</option>' + Array.from(launches).sort((a, b) => Number(a) - Number(b)).map(id => `<option value="${{id}}">launch${{id}}</option>`).join('');
  select.value = launches.has(current) ? current : 'all';
}}
function toggleStage(categoryName, visible) {{
  stageVisibility[categoryName] = visible;
  for (const item of document.querySelectorAll(`[data-stage-category="${{categoryName}}"]`)) {{
    item.style.display = visible ? '' : 'none';
  }}
}}
renderTimeline();
</script>
</body>
</html>
"""


def write_outputs(root, index, emit_ai_prompt):
    (root / "trace_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "analysis.md").write_text(render_analysis(index), encoding="utf-8")
    (root / "report.html").write_text(render_html(index), encoding="utf-8")

    prompt_path = root / "ai_prompt.md"
    if emit_ai_prompt:
        prompt_path.write_text(render_ai_prompt(index), encoding="utf-8")
    elif prompt_path.exists():
        prompt_path.unlink()


def main():
    args = parse_args()
    root = Path(args.profile_dir)
    root.mkdir(parents=True, exist_ok=True)
    index = build_index(root, args)
    write_outputs(root, index, args.emit_ai_prompt)
    print(f"wrote TileXR collective profile report to {root / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
