# Collective Profile Timeline Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a run-level TileXR collective profiling report that aggregates existing per-rank/per-launch traces into a chronological, zoomable, multi-iteration timeline.

**Architecture:** Keep the current C++ per-launch profiler unchanged. Add a self-contained Python report helper that reads `rank*/launch*/trace.json`, writes root-level `trace_index.json`, `analysis.md`, optional `ai_prompt.md`, and static `report.html`, then invoke it from `run_collective_perf.sh` only after all rank processes exit successfully.

**Tech Stack:** Python 3 standard library, Bash, CMake/CTest, existing TileXR collectives test structure.

---

## File Structure

- Create `tests/collectives/tilexr_collective_profile_report.py`
  - Self-contained aggregation helper.
  - Parses profile roots, validates trace compatibility, normalizes bars per launch/rank, writes JSON/Markdown/HTML.
- Create `tests/collectives/unit/test_collective_profile_report.py`
  - Python unittest fixtures that generate fake `rank*/launch*/trace.json` directories and execute the helper.
- Modify `tests/collectives/run_collective_perf.sh`
  - Parse profiling-related forwarded args.
  - Invoke the helper after the wait loop succeeds and before `exit 0`.
- Modify `tests/collectives/CMakeLists.txt`
  - Discover Python 3.
  - Add the Python unittest to CTest when Python 3 is available.
  - Install the helper beside `run_collective_perf.sh`.
- Modify `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`
  - Source-check helper presence, runner post-run aggregation wiring, CMake test wiring, and README documentation.
- Modify `tests/collectives/README.md`
  - Document root-level aggregation artifacts and manual helper use.

## Task 1: Add Failing Aggregation Helper Tests

**Files:**
- Create: `tests/collectives/unit/test_collective_profile_report.py`
- Test command: `python3 tests/collectives/unit/test_collective_profile_report.py`

- [ ] **Step 1: Write the failing Python unittest**

Create `tests/collectives/unit/test_collective_profile_report.py` with this content:

```python
#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COLLECTIVES_DIR = Path(__file__).resolve().parents[1]
HELPER = COLLECTIVES_DIR / "tilexr_collective_profile_report.py"


def make_stat(rank, core, stage, stage_id, first, duration, count=1, raw_cycles=None, max_cycles=None):
    if raw_cycles is None:
        raw_cycles = duration
    if max_cycles is None:
        max_cycles = duration
    return {
        "rank": rank,
        "core": core,
        "stage": stage,
        "stage_id": stage_id,
        "count": count,
        "raw_cycles": raw_cycles,
        "min_cycles": raw_cycles // count if count else 0,
        "max_cycles": max_cycles,
        "first_start_cycle": first,
        "last_end_cycle": first + duration,
        "aux0": 0,
        "aux1": 0,
        "sum_us": raw_cycles / 50.0,
    }


def write_trace(root, rank, launch, message_bytes=1024, schema="tilexr_perf_trace_report.v1", rank_size=2):
    launch_dir = root / f"rank{rank}" / f"launch{launch}"
    launch_dir.mkdir(parents=True)
    base = 1000000 + rank * 100000 + launch * 10000
    stats = [
        make_stat(rank, 0, "kernel_total", 0, base, 1000 + launch * 100 + rank * 10),
        make_stat(rank, 0, "flag_poll_wait", 4, base + 100, 320 + rank * 20, count=8, raw_cycles=160 + rank * 10, max_cycles=44),
        make_stat(rank, 0, "peer_ipc_to_output", 5, base + 450, 420 + launch * 30, count=2, raw_cycles=400 + launch * 20, max_cycles=260),
        make_stat(rank, 1, "kernel_total", 0, base + 25, 900 + launch * 80 + rank * 12),
    ]
    trace = {
        "schema": schema,
        "op_type": 3,
        "op_name": "TileXRAllGather",
        "rank_size": rank_size,
        "max_core_count": 2,
        "block_dim": 2,
        "stage_count": 7,
        "cycle_to_us_divisor": 50,
        "message_bytes": message_bytes,
        "stats": stats,
    }
    (launch_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    (launch_dir / "report.html").write_text("<html>single launch</html>\n", encoding="utf-8")


def run_helper(root, *args):
    command = [sys.executable, str(HELPER), str(root), *args]
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class CollectiveProfileReportTest(unittest.TestCase):
    def test_writes_chronological_multi_rank_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for launch in (0, 1):
                write_trace(root, 0, launch)
                write_trace(root, 1, launch)

            result = run_helper(root, "--warmup-iters", "5", "--iters", "2", "--profile-sample-every", "1", "--emit-ai-prompt")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["schema"], "tilexr_perf_trace_run.v1")
            self.assertEqual(index["warmup_iters"], 5)
            self.assertEqual(index["measured_iters"], 2)
            self.assertEqual(index["profile_sample_every"], 1)
            self.assertEqual(index["groups"][0]["launch_ids"], [0, 1])
            self.assertEqual(index["groups"][0]["rank_size"], 2)
            self.assertFalse(index["diagnostics"])

            bars = index["groups"][0]["bars"]
            first_kernel = next(bar for bar in bars if bar["launch_id"] == 0 and bar["rank"] == 0 and bar["core"] == 0 and bar["stage"] == "kernel_total")
            self.assertEqual(first_kernel["start_cycles"], 0)
            self.assertEqual(first_kernel["end_cycles"], 1000)
            self.assertEqual(first_kernel["source"], "rank0/launch0/trace.json")

            html = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("Bottleneck First", html)
            self.assertIn("Slowest launch", html)
            self.assertIn("Top stage", html)
            self.assertIn("Chronological Timeline", html)
            self.assertIn("Zoom In", html)
            self.assertIn("Zoom Out", html)
            self.assertIn("Fit", html)
            self.assertIn("Fold Cores", html)
            self.assertIn("launchFilter", html)
            self.assertIn("launch0", html)
            self.assertIn("rank0/core0", html)
            self.assertIn("rank0/launch0/report.html", html)

            analysis = (root / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("Slowest launch", analysis)
            self.assertIn("flag_poll_wait", analysis)
            self.assertIn("cross-NPU raw cycle offsets", analysis)

            prompt = (root / "ai_prompt.md").read_text(encoding="utf-8")
            self.assertIn("TileXR collective profiling run", prompt)
            self.assertIn("warmup iterations: 5", prompt)

    def test_preserves_sparse_launch_ids_and_reports_missing_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0)
            write_trace(root, 1, 0)
            write_trace(root, 0, 2)

            result = run_helper(root, "--warmup-iters", "3", "--iters", "4", "--profile-sample-every", "2")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["groups"][0]["launch_ids"], [0, 2])
            joined = "\n".join(index["diagnostics"])
            self.assertIn("missing trace for rank1 launch2", joined)
            self.assertNotIn("launch1", joined)

    def test_keeps_incompatible_message_bytes_out_of_valid_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0, message_bytes=1024)
            write_trace(root, 1, 0, message_bytes=2048)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index["groups"]), 2)
            joined = "\n".join(index["diagnostics"])
            self.assertIn("incompatible trace groups detected", joined)

    def test_invalid_schema_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0, schema="bad.schema")

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertFalse(index["groups"])
            self.assertIn("invalid schema in rank0/launch0/trace.json", "\n".join(index["diagnostics"]))

    def test_single_rank_single_launch_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0, rank_size=1)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["groups"][0]["rank_size"], 1)
            self.assertEqual(index["groups"][0]["launch_ids"], [0])
            self.assertFalse(index["diagnostics"])
            self.assertIn("Chronological Timeline", (root / "report.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 tests/collectives/unit/test_collective_profile_report.py
```

Expected: FAIL because `tests/collectives/tilexr_collective_profile_report.py` does not exist. The failure should include a Python error similar to `can't open file`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/collectives/unit/test_collective_profile_report.py
git commit -m "test: add collective profile aggregation fixtures"
```

## Task 2: Implement the Aggregation Helper

**Files:**
- Create: `tests/collectives/tilexr_collective_profile_report.py`
- Test: `tests/collectives/unit/test_collective_profile_report.py`

- [ ] **Step 1: Create the executable helper**

Create `tests/collectives/tilexr_collective_profile_report.py` with this implementation:

```python
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
    if not match:
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
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            diagnostics.append(f"invalid json in {relpath(path, root)}: {exc}")
            continue
        if trace.get("schema") != TRACE_SCHEMA:
            diagnostics.append(f"invalid schema in {relpath(path, root)}")
            continue
        traces.append({"rank": rank, "launch_id": launch_id, "path": path, "trace": trace})
    return traces, diagnostics


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


def expected_launch_ids(iters, sample_every):
    if iters <= 0 or sample_every <= 0:
        return []
    return [launch_id for launch_id in range(iters) if launch_id % sample_every == 0]


def cycles_to_us(cycles, divisor):
    return 0.0 if divisor == 0 else float(cycles) / float(divisor)


def normalized_bars(entries, root):
    bars = []
    entries_by_rank_launch = defaultdict(list)
    for entry in entries:
        entries_by_rank_launch[(entry["rank"], entry["launch_id"])].append(entry)

    for (rank, launch_id), rank_entries in sorted(entries_by_rank_launch.items()):
        stats = []
        for entry in rank_entries:
            stats.extend(entry["trace"].get("stats", []))
        starts = [
            int(stat.get("first_start_cycle", 0))
            for stat in stats
            if int(stat.get("first_start_cycle", 0)) > 0
        ]
        zero = min(starts) if starts else 0
        for entry in rank_entries:
            divisor = int(entry["trace"].get("cycle_to_us_divisor", 0))
            for stat in entry["trace"].get("stats", []):
                count = int(stat.get("count", 0))
                first = int(stat.get("first_start_cycle", 0))
                last = int(stat.get("last_end_cycle", 0))
                if count == 0 and first == 0 and last == 0:
                    continue
                start_cycles = max(0, first - zero)
                end_cycles = max(start_cycles, last - zero)
                duration_cycles = max(0, last - first)
                bars.append({
                    "launch_id": launch_id,
                    "rank": int(stat.get("rank", rank)),
                    "core": int(stat.get("core", 0)),
                    "stage": str(stat.get("stage", "unknown")),
                    "stage_id": int(stat.get("stage_id", 0)),
                    "start_cycles": start_cycles,
                    "end_cycles": end_cycles,
                    "duration_cycles": duration_cycles,
                    "start_us": cycles_to_us(start_cycles, divisor),
                    "end_us": cycles_to_us(end_cycles, divisor),
                    "duration_us": cycles_to_us(duration_cycles, divisor),
                    "sum_us": float(stat.get("sum_us", cycles_to_us(int(stat.get("raw_cycles", 0)), divisor))),
                    "raw_cycles": int(stat.get("raw_cycles", 0)),
                    "count": count,
                    "max_cycles": int(stat.get("max_cycles", 0)),
                    "source": relpath(entry["path"], root),
                    "drilldown": relpath(entry["path"].with_name("report.html"), root),
                    "lane": f"rank{rank}/core{int(stat.get('core', 0))}",
                })
    return bars


def summarize_group(group):
    bars = group["bars"]
    stage_totals = defaultdict(float)
    launch_kernel = defaultdict(float)
    rank_core_max = {"rank": 0, "core": 0, "stage": "", "duration_us": 0.0}
    for bar in bars:
        stage_totals[bar["stage"]] += bar["sum_us"]
        if bar["stage"] == "kernel_total":
            launch_kernel[bar["launch_id"]] = max(launch_kernel[bar["launch_id"]], bar["duration_us"])
        if bar["duration_us"] > rank_core_max["duration_us"]:
            rank_core_max = {
                "rank": bar["rank"],
                "core": bar["core"],
                "stage": bar["stage"],
                "duration_us": bar["duration_us"],
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
    if len(grouped) > 1:
        diagnostics.append("incompatible trace groups detected")

    expected = expected_launch_ids(args.iters, args.profile_sample_every)
    groups = []
    for key, entries in sorted(grouped.items(), key=lambda item: (item[0][3] or 0, item[0][0] or 0)):
        op_type, op_name, rank_size, message_bytes, stage_count, divisor = key
        rank_ids = sorted({entry["rank"] for entry in entries})
        launch_ids = sorted({entry["launch_id"] for entry in entries})
        if rank_size is None:
            rank_size = len(rank_ids)
        expected_ranks = list(range(int(rank_size)))
        for launch_id in expected or launch_ids:
            for rank in expected_ranks:
                if not any(entry["rank"] == rank and entry["launch_id"] == launch_id for entry in entries):
                    diagnostics.append(f"missing trace for rank{rank} launch{launch_id}")
        group = {
            "op_type": op_type,
            "op_name": op_name,
            "rank_size": int(rank_size),
            "message_bytes": int(message_bytes),
            "stage_count": int(stage_count),
            "cycle_to_us_divisor": int(divisor),
            "rank_ids": rank_ids,
            "launch_ids": launch_ids,
            "sources": [relpath(entry["path"], root) for entry in sorted(entries, key=lambda item: (item["launch_id"], item["rank"]))],
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


def render_analysis(index):
    lines = ["# TileXR Collective Profile Run Analysis", ""]
    lines.append(f"- Warmup iterations: {index['warmup_iters']}")
    lines.append(f"- Measured iterations: {index['measured_iters']}")
    lines.append(f"- Profile sample every: {index['profile_sample_every']}")
    lines.append("- Note: cross-NPU raw cycle offsets are not assumed to be synchronized.")
    lines.append("")
    if index["diagnostics"]:
        lines.append("## Diagnostics")
        lines.extend(f"- {item}" for item in index["diagnostics"])
        lines.append("")
    for group_index, group in enumerate(index["groups"]):
        summary = group["summary"]
        lines.append(f"## Group {group_index}: {group['op_name']} message_bytes={group['message_bytes']}")
        lines.append(f"- Launches: {', '.join('launch' + str(item) for item in group['launch_ids'])}")
        lines.append(f"- Slowest launch: launch{summary['slowest_launch']['launch_id']} at {summary['slowest_launch']['kernel_us']:.3f} us")
        lines.append(f"- Top stage: {summary['top_stage']['stage']} at {summary['top_stage']['sum_us']:.3f} us")
        lines.append(
            f"- Rank/core max: rank{summary['rank_core_max']['rank']} core{summary['rank_core_max']['core']} "
            f"{summary['rank_core_max']['stage']} {summary['rank_core_max']['duration_us']:.3f} us"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_ai_prompt(index):
    lines = ["# TileXR collective profiling run", ""]
    lines.append("Analyze this TileXR collective profiling run and suggest bottleneck investigation steps.")
    lines.append("")
    lines.append(f"warmup iterations: {index['warmup_iters']}")
    lines.append(f"measured iterations: {index['measured_iters']}")
    lines.append(f"profile sample every: {index['profile_sample_every']}")
    lines.append("cross-NPU raw cycle offsets are not assumed to be synchronized")
    lines.append("")
    for group in index["groups"]:
        summary = group["summary"]
        lines.append(f"- {group['op_name']} bytes={group['message_bytes']} launches={group['launch_ids']}")
        lines.append(f"  slowest_launch=launch{summary['slowest_launch']['launch_id']} kernel_us={summary['slowest_launch']['kernel_us']:.3f}")
        lines.append(f"  top_stage={summary['top_stage']['stage']} sum_us={summary['top_stage']['sum_us']:.3f}")
    if index["diagnostics"]:
        lines.append("")
        lines.append("diagnostics:")
        lines.extend(f"- {item}" for item in index["diagnostics"])
    return "\n".join(lines) + "\n"


def render_html(index):
    data = json.dumps(index, separators=(",", ":"))
    fallback_rows = []
    summary_items = []
    for group in index["groups"]:
        summary = group["summary"]
        summary_items.append(
            "<li>"
            f"{html.escape(group['op_name'])} bytes={group['message_bytes']}: "
            f"Slowest launch launch{summary['slowest_launch']['launch_id']} "
            f"{summary['slowest_launch']['kernel_us']:.3f} us; "
            f"Top stage {html.escape(summary['top_stage']['stage'])} "
            f"{summary['top_stage']['sum_us']:.3f} us; "
            f"Rank/core max rank{summary['rank_core_max']['rank']} core{summary['rank_core_max']['core']} "
            f"{html.escape(summary['rank_core_max']['stage'])} "
            f"{summary['rank_core_max']['duration_us']:.3f} us"
            "</li>"
        )
        for bar in group["bars"]:
            fallback_rows.append(
                "<tr>"
                f"<td>launch{bar['launch_id']}</td>"
                f"<td>rank{bar['rank']}/core{bar['core']}</td>"
                f"<td>{html.escape(bar['stage'])}</td>"
                f"<td>{bar['duration_us']:.3f}</td>"
                f"<td>{bar['sum_us']:.3f}</td>"
                f"<td><a href=\"{html.escape(bar['drilldown'])}\">open launch report</a></td>"
                "</tr>"
            )
    diagnostics = "".join(f"<li>{html.escape(item)}</li>" for item in index["diagnostics"])
    summary_html = "".join(summary_items) + diagnostics
    return f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\">
<title>TileXR Collective Profile Run Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#172033;background:#f8fafc}}
button{{margin-right:8px;padding:6px 10px;border:1px solid #94a3b8;background:#fff;border-radius:4px;cursor:pointer}}
.panel{{background:#fff;border:1px solid #cbd5e1;border-radius:6px;padding:16px;margin:16px 0}}
.timeline-wrap{{overflow:auto;border:1px solid #cbd5e1;background:#fff;height:520px;position:relative}}
.timeline{{position:relative;height:100%;min-width:960px;transform-origin:0 0}}
.bar{{position:absolute;height:16px;border-radius:3px;color:#0f172a;font-size:11px;overflow:hidden;white-space:nowrap;border:1px solid rgba(15,23,42,.25)}}
.stage-total{{background:#cbd5e1}} .stage-wait{{background:#fca5a5}} .stage-copy{{background:#93c5fd}} .stage-sync{{background:#a7f3d0}}
.launch-line{{position:absolute;top:0;bottom:0;border-left:1px dashed #94a3b8;color:#475569;font-size:12px;padding-left:4px}}
table{{border-collapse:collapse;background:#fff}} td,th{{border:1px solid #cbd5e1;padding:6px 8px}} th{{background:#e2e8f0;text-align:left}}
</style>
</head>
<body>
<h1>TileXR Collective Profile Run Report</h1>
<section class=\"panel\">
<h2>Bottleneck First</h2>
<p>Warmup iterations: {index['warmup_iters']}; measured iterations: {index['measured_iters']}; sample every: {index['profile_sample_every']}.</p>
<ul>{summary_html}</ul>
</section>
<section class=\"panel\">
<h2>Chronological Timeline</h2>
<div>
<button onclick=\"zoomBy(1.25)\">Zoom In</button>
<button onclick=\"zoomBy(0.8)\">Zoom Out</button>
<button onclick=\"fitTimeline()\">Fit</button>
<button onclick=\"toggleLaneMode()\">Fold Cores</button>
<label>Launch <select id=\"launchFilter\" onchange=\"renderTimeline()\"><option value=\"all\">all</option></select></label>
<label><input type=\"checkbox\" checked onchange=\"toggleStage('wait', this.checked)\"> wait</label>
<label><input type=\"checkbox\" checked onchange=\"toggleStage('copy', this.checked)\"> copy</label>
<label><input type=\"checkbox\" checked onchange=\"toggleStage('sync', this.checked)\"> sync</label>
<label><input type=\"checkbox\" checked onchange=\"toggleStage('total', this.checked)\"> total</label>
</div>
<div class=\"timeline-wrap\" id=\"wrap\"><div class=\"timeline\" id=\"timeline\"></div></div>
</section>
<section class=\"panel\">
<h2>Fallback Table</h2>
<table><thead><tr><th>Launch</th><th>Lane</th><th>Stage</th><th>Duration us</th><th>Sum us</th><th>Drilldown</th></tr></thead><tbody>{''.join(fallback_rows)}</tbody></table>
</section>
<script>
const traceIndex = {data};
let scale = 1;
let foldCores = false;
function category(stage) {{
  if (stage.includes('wait')) return 'wait';
  if (stage.includes('copy') || stage.includes('ipc')) return 'copy';
  if (stage.includes('sync') || stage.includes('barrier')) return 'sync';
  return 'total';
}}
function renderTimeline() {{
  const root = document.getElementById('timeline');
  root.innerHTML = '';
  refreshLaunchFilter();
  const selectedLaunch = document.getElementById('launchFilter').value;
  const laneHeight = 28;
  const launchGap = 80;
  let xBase = 0;
  const lanes = new Map();
  let laneCount = 0;
  for (const group of traceIndex.groups) {{
    for (const launchId of group.launch_ids) {{
      if (selectedLaunch !== 'all' && selectedLaunch !== String(launchId)) continue;
      const launchBars = group.bars.filter(bar => bar.launch_id === launchId);
      const maxEnd = Math.max(1, ...launchBars.map(bar => bar.end_us));
      const width = Math.max(220, maxEnd * scale * 4);
      const line = document.createElement('div');
      line.className = 'launch-line';
      line.style.left = `${{xBase}}px`;
      line.textContent = `launch${{launchId}}`;
      root.appendChild(line);
      for (const bar of launchBars) {{
        const lane = foldCores ? `rank${{bar.rank}}` : `rank${{bar.rank}}/core${{bar.core}}`;
        if (!lanes.has(lane)) lanes.set(lane, laneCount++);
        const item = document.createElement('a');
        item.href = bar.drilldown;
        const cat = category(bar.stage);
        item.className = `bar stage-${{cat}}`;
        item.dataset.stageCategory = cat;
        item.style.left = `${{xBase + bar.start_us * scale * 4}}px`;
        item.style.top = `${{24 + lanes.get(lane) * laneHeight}}px`;
        item.style.width = `${{Math.max(2, (bar.end_us - bar.start_us) * scale * 4)}}px`;
        item.title = `launch${{bar.launch_id}} ${{lane}} ${{bar.stage}} duration=${{bar.duration_us.toFixed(3)}}us sum=${{bar.sum_us.toFixed(3)}}us count=${{bar.count}} source=${{bar.source}}`;
        item.textContent = `${{lane}} ${{bar.stage}}`;
        root.appendChild(item);
      }}
      xBase += width + launchGap;
    }}
  }}
  root.style.width = `${{Math.max(960, xBase)}}px`;
  root.style.height = `${{Math.max(520, 60 + laneCount * laneHeight)}}px`;
}}
function zoomBy(factor) {{ scale = Math.max(0.1, Math.min(20, scale * factor)); renderTimeline(); }}
function fitTimeline() {{ scale = 1; renderTimeline(); document.getElementById('wrap').scrollLeft = 0; }}
function toggleLaneMode() {{ foldCores = !foldCores; renderTimeline(); }}
function refreshLaunchFilter() {{
  const select = document.getElementById('launchFilter');
  const current = select.value || 'all';
  const launches = new Set();
  for (const group of traceIndex.groups) for (const launchId of group.launch_ids) launches.add(String(launchId));
  select.innerHTML = '<option value=\"all\">all</option>' + Array.from(launches).sort((a, b) => Number(a) - Number(b)).map(id => `<option value=\"${{id}}\">launch${{id}}</option>`).join('');
  select.value = launches.has(current) ? current : 'all';
}}
function toggleStage(categoryName, visible) {{
  for (const item of document.querySelectorAll(`[data-stage-category=\"${{categoryName}}\"]`)) {{
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
    prompt = root / "ai_prompt.md"
    if emit_ai_prompt:
        prompt.write_text(render_ai_prompt(index), encoding="utf-8")
    elif prompt.exists():
        prompt.unlink()


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
```

- [ ] **Step 2: Make the helper executable**

Run:

```bash
chmod +x tests/collectives/tilexr_collective_profile_report.py
```

- [ ] **Step 3: Run the helper tests**

Run:

```bash
python3 tests/collectives/unit/test_collective_profile_report.py
```

Expected: PASS. The output ends with `OK`.

- [ ] **Step 4: Inspect generated report once**

Run:

```bash
tmpdir="$(mktemp -d)"
python3 - <<'PY' "${tmpdir}"
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for rank in (0, 1):
    launch_dir = root / f"rank{rank}" / "launch0"
    launch_dir.mkdir(parents=True)
    trace = {
        "schema": "tilexr_perf_trace_report.v1",
        "op_type": 3,
        "op_name": "TileXRAllGather",
        "rank_size": 2,
        "max_core_count": 1,
        "block_dim": 1,
        "stage_count": 7,
        "cycle_to_us_divisor": 50,
        "message_bytes": 1024,
        "stats": [
            {"rank": rank, "core": 0, "stage": "kernel_total", "stage_id": 0, "count": 1, "raw_cycles": 1000, "min_cycles": 1000, "max_cycles": 1000, "first_start_cycle": 100000 + rank * 1000, "last_end_cycle": 101000 + rank * 1000, "aux0": 0, "aux1": 0, "sum_us": 20.0}
        ],
    }
    (launch_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (launch_dir / "report.html").write_text("single launch", encoding="utf-8")
PY
python3 tests/collectives/tilexr_collective_profile_report.py "${tmpdir}" --warmup-iters 5 --iters 1 --profile-sample-every 1 --emit-ai-prompt
test -f "${tmpdir}/report.html"
test -f "${tmpdir}/trace_index.json"
test -f "${tmpdir}/analysis.md"
test -f "${tmpdir}/ai_prompt.md"
```

Expected: all `test -f` commands exit with status `0`.

- [ ] **Step 5: Commit the helper**

```bash
git add tests/collectives/tilexr_collective_profile_report.py tests/collectives/unit/test_collective_profile_report.py
git commit -m "feat: add collective profile run report helper"
```

## Task 3: Wire the Helper into CMake and Source Checks

**Files:**
- Modify: `tests/collectives/CMakeLists.txt`
- Modify: `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`
- Test: `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`

- [ ] **Step 1: Add failing CMake source checks**

In `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`, add these checks inside `TestCMakeWiring()`,
after the existing script checks:

```cpp
    CheckContains(path, text, "find_package(Python3 COMPONENTS Interpreter)");
    CheckContains(path, text, "test_collective_profile_report");
    CheckContains(path, text, "tilexr_collective_profile_report.py");
```

- [ ] **Step 2: Run the source check to verify it fails**

Run:

```bash
cmake --build build-profile --target test_tilexr_collectives_tools_sources -j"$(nproc)"
./build-profile/tests/collectives/test_tilexr_collectives_tools_sources
```

Expected: FAIL with missing strings such as `test_collective_profile_report`.

- [ ] **Step 3: Modify CMake wiring**

In `tests/collectives/CMakeLists.txt`, add this after `enable_testing()`:

```cmake
find_package(Python3 COMPONENTS Interpreter)
```

After the existing `add_test(NAME test_prepare_host_launch_context ...)` block, add:

```cmake
if(Python3_Interpreter_FOUND)
    add_test(
        NAME test_collective_profile_report
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_collective_profile_report.py
    )
endif()
```

In the `install(PROGRAMS ...)` list, add the helper between the runner scripts:

```cmake
install(PROGRAMS
    run_collectives_correctness.sh
    run_collective_perf.sh
    tilexr_collective_profile_report.py
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)
```

- [ ] **Step 4: Run CMake configure and helper test**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
ctest --test-dir build-profile -R test_collective_profile_report --output-on-failure
```

Expected: CMake config succeeds. The CTest run reports `100% tests passed`.

- [ ] **Step 5: Commit CMake and source check updates**

```bash
git add tests/collectives/CMakeLists.txt tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp
git commit -m "test: wire collective profile aggregation tests"
```

## Task 4: Invoke Aggregation After Successful Multi-Rank Runs

**Files:**
- Modify: `tests/collectives/run_collective_perf.sh`
- Test: `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`

- [ ] **Step 1: Add failing runner source checks**

In `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`, add these checks inside
`TestLauncherScripts()`, after the existing perf runner checks:

```cpp
    CheckContains(perfPath, perf, "parse_profile_args");
    CheckContains(perfPath, perf, "write_profile_report_if_enabled");
    CheckContains(perfPath, perf, "tilexr_collective_profile_report.py");
    CheckContains(perfPath, perf, "--warmup-iters");
    CheckContains(perfPath, perf, "--profile-sample-every");
```

- [ ] **Step 2: Run the source check to verify it fails**

Run:

```bash
cmake --build build-profile --target test_tilexr_collectives_tools_sources -j"$(nproc)"
./build-profile/tests/collectives/test_tilexr_collectives_tools_sources
```

Expected: FAIL with missing strings such as `parse_profile_args` and `write_profile_report_if_enabled`.

- [ ] **Step 3: Add profile arg parsing helpers**

In `tests/collectives/run_collective_perf.sh`, after `timeout_sec=...`, add:

```bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile_enabled="0"
profile_dir=""
profile_ai_prompt="0"
profile_sample_every="1"
warmup_iters="5"
measured_iters="20"

parse_profile_args() {
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --profile)
        if (( i + 1 < ${#args[@]} )); then profile_enabled="${args[$((i + 1))]}"; fi
        ;;
      --profile-dir)
        if (( i + 1 < ${#args[@]} )); then profile_dir="${args[$((i + 1))]}"; fi
        ;;
      --profile-ai-prompt)
        if (( i + 1 < ${#args[@]} )); then profile_ai_prompt="${args[$((i + 1))]}"; fi
        ;;
      --profile-sample-every)
        if (( i + 1 < ${#args[@]} )); then profile_sample_every="${args[$((i + 1))]}"; fi
        ;;
      --warmup-iters)
        if (( i + 1 < ${#args[@]} )); then warmup_iters="${args[$((i + 1))]}"; fi
        ;;
      --iters)
        if (( i + 1 < ${#args[@]} )); then measured_iters="${args[$((i + 1))]}"; fi
        ;;
    esac
  done
}

write_profile_report_if_enabled() {
  if [[ "${profile_enabled}" != "1" && "${profile_enabled}" != "true" ]]; then
    return 0
  fi
  local root="${profile_dir:-run/prof/collectives}"
  local helper="${script_dir}/tilexr_collective_profile_report.py"
  local python_cmd=""
  if command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
  elif command -v python >/dev/null 2>&1; then
    python_cmd="python"
  fi
  if [[ -z "${python_cmd}" ]]; then
    echo "WARNING: python3 not found; skipping aggregate profile report" >&2
    return 0
  fi
  if [[ ! -f "${helper}" ]]; then
    echo "WARNING: ${helper} not found; skipping aggregate profile report" >&2
    return 0
  fi
  local prompt_args=()
  if [[ "${profile_ai_prompt}" == "1" || "${profile_ai_prompt}" == "true" ]]; then
    prompt_args+=(--emit-ai-prompt)
  fi
  "${python_cmd}" "${helper}" "${root}" \
    --warmup-iters "${warmup_iters}" \
    --iters "${measured_iters}" \
    --profile-sample-every "${profile_sample_every}" \
    "${prompt_args[@]}"
}

parse_profile_args "$@"
```

- [ ] **Step 4: Call the helper only after successful waits**

Near the end of `tests/collectives/run_collective_perf.sh`, after:

```bash
trap - INT TERM
```

and before:

```bash
exit 0
```

add:

```bash
if ! write_profile_report_if_enabled; then
  echo "ERROR: aggregate profile report generation failed" >&2
  exit 1
fi
```

- [ ] **Step 5: Run source check**

Run:

```bash
cmake --build build-profile --target test_tilexr_collectives_tools_sources -j"$(nproc)"
./build-profile/tests/collectives/test_tilexr_collectives_tools_sources
```

Expected: PASS and prints `TileXR collectives tools source checks passed`.

- [ ] **Step 6: Smoke-test the runner helper path without NPU work**

Run:

```bash
tmpdir="$(mktemp -d)"
mkdir -p "${tmpdir}/rank0/launch0" "${tmpdir}/rank1/launch0"
python3 - <<'PY' "${tmpdir}"
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for rank in (0, 1):
    d = root / f"rank{rank}" / "launch0"
    trace = {
        "schema": "tilexr_perf_trace_report.v1",
        "op_type": 3,
        "op_name": "TileXRAllGather",
        "rank_size": 2,
        "max_core_count": 1,
        "block_dim": 1,
        "stage_count": 7,
        "cycle_to_us_divisor": 50,
        "message_bytes": 1024,
        "stats": [
            {"rank": rank, "core": 0, "stage": "kernel_total", "stage_id": 0, "count": 1, "raw_cycles": 1000, "min_cycles": 1000, "max_cycles": 1000, "first_start_cycle": 100000, "last_end_cycle": 101000, "aux0": 0, "aux1": 0, "sum_us": 20.0}
        ],
    }
    (d / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (d / "report.html").write_text("single launch", encoding="utf-8")
PY
python3 tests/collectives/tilexr_collective_profile_report.py "${tmpdir}" --warmup-iters 5 --iters 1 --profile-sample-every 1
grep -q "Chronological Timeline" "${tmpdir}/report.html"
```

Expected: `grep` exits with status `0`.

- [ ] **Step 7: Commit runner integration**

```bash
git add tests/collectives/run_collective_perf.sh tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp
git commit -m "feat: aggregate collective profile reports after runs"
```

## Task 5: Document Run-Level Reports

**Files:**
- Modify: `tests/collectives/README.md`
- Modify: `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`

- [ ] **Step 1: Add failing README source checks**

In `TestReadmeDocumentsManualRuns()` in `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`, add:

```cpp
    CheckContains(path, text, "trace_index.json");
    CheckContains(path, text, "root-level report.html");
    CheckContains(path, text, "tilexr_collective_profile_report.py");
    CheckContains(path, text, "zoom");
    CheckContains(path, text, "chronological timeline");
```

- [ ] **Step 2: Run the source check to verify it fails**

Run:

```bash
cmake --build build-profile --target test_tilexr_collectives_tools_sources -j"$(nproc)"
./build-profile/tests/collectives/test_tilexr_collectives_tools_sources
```

Expected: FAIL with missing README strings such as `trace_index.json`.

- [ ] **Step 3: Update README profiling section**

In `tests/collectives/README.md`, replace the paragraph that begins with `Each sampled launch writes` with:

````markdown
Each sampled measured launch writes `trace.json`, `summary.csv`, `analysis.md`, `report.html`, and `ai_prompt.md`
when prompt export is enabled. `--profile-dir` is a root directory; each rank writes sampled launches under
`run/prof/collectives/rank<N>/launch<M>/` in the example above.

After all rank processes finish successfully, `run_collective_perf.sh` also writes a root-level report:

```text
run/prof/collectives/report.html
run/prof/collectives/trace_index.json
run/prof/collectives/analysis.md
run/prof/collectives/ai_prompt.md
```

The root-level `report.html` keeps the bottleneck-first summary and adds a zoomable chronological timeline across
sampled measured iterations. Warmup execution is controlled by the existing `--warmup-iters` option and is reported
as metadata; warmup launches are not profiled by this report path. The per-launch `rank<N>/launch<M>/report.html`
files remain available for drilldown.

To regenerate the aggregate report from an existing profile directory:

```bash
python3 tilexr_collective_profile_report.py run/prof/collectives \
  --warmup-iters 5 --iters 20 --profile-sample-every 1 --emit-ai-prompt
```
````

- [ ] **Step 4: Run source check**

Run:

```bash
cmake --build build-profile --target test_tilexr_collectives_tools_sources -j"$(nproc)"
./build-profile/tests/collectives/test_tilexr_collectives_tools_sources
```

Expected: PASS.

- [ ] **Step 5: Commit docs**

```bash
git add tests/collectives/README.md tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp
git commit -m "docs: describe collective profile timeline reports"
```

## Task 6: Full Local Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run Python helper tests directly**

```bash
python3 tests/collectives/unit/test_collective_profile_report.py
```

Expected: PASS and output ends with `OK`.

- [ ] **Step 2: Run collectives CTest subset**

```bash
source scripts/common_env.sh
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
cmake --build build-profile --target test_tilexr_collectives_tools_sources -j"$(nproc)"
ctest --test-dir build-profile -R "test_collective_profile_report|test_tilexr_collectives_tools_sources" --output-on-failure
```

Expected: CTest reports `100% tests passed`.

- [ ] **Step 3: Run full profile build tests if build-profile is healthy**

```bash
ctest --test-dir build-profile --output-on-failure
```

Expected: all available host-side tests pass.

- [ ] **Step 4: Check git diff**

```bash
git status --short
git log --oneline -5
```

Expected: only intentional uncommitted files remain, if any. Recent commits include the task commits from this plan.

## Task 7: Hardware Verification on Ascend NPUs

**Files:**
- Verify only.

- [ ] **Step 1: Build profile target**

```bash
source scripts/common_env.sh
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
cmake --build build-profile --target tilexr_collective_perf -j"$(nproc)"
```

Expected: build succeeds.

- [ ] **Step 2: Run 2-rank AllGather with multiple measured iterations**

```bash
cd tests/collectives
./run_collective_perf.sh 2 0 ../../build-profile/tests/collectives \
  --op allgather --min-bytes 67108864 --max-bytes 67108864 \
  --iters 4 --warmup-iters 2 --datatype int32 --check 1 \
  --profile 1 --profile-dir run/prof/collectives2_timeline --profile-ai-prompt 1
```

Expected:

- Command exits with status `0`.
- `run/prof/collectives2_timeline/report.html` exists.
- `run/prof/collectives2_timeline/trace_index.json` exists.
- `trace_index.json` contains launch ids `0`, `1`, `2`, and `3`.
- Root `report.html` contains `Chronological Timeline`.

- [ ] **Step 3: Run 4-rank AllGather with multiple measured iterations**

```bash
cd tests/collectives
./run_collective_perf.sh 4 0 ../../build-profile/tests/collectives \
  --op allgather --min-bytes 67108864 --max-bytes 67108864 \
  --iters 4 --warmup-iters 2 --datatype int32 --check 1 \
  --profile 1 --profile-dir run/prof/collectives4_timeline --profile-ai-prompt 1
```

Expected:

- Command exits with status `0`.
- `run/prof/collectives4_timeline/report.html` exists.
- `run/prof/collectives4_timeline/trace_index.json` exists.
- Timeline contains lanes for ranks `0`, `1`, `2`, and `3`.
- Existing per-launch reports still exist under `rank<N>/launch<M>/report.html`.

- [ ] **Step 4: Compare overhead**

Run one non-profiled and one profiled 4-rank command with the same message size and iterations. Compare `avg(us)`.

Expected: no extra kernel-side overhead beyond existing profiling, because the new aggregation runs after rank
processes finish.
