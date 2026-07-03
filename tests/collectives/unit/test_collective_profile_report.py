#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COLLECTIVES_DIR = Path(__file__).resolve().parents[1]
HELPER = COLLECTIVES_DIR / "tilexr_collective_profile_report.py"
SINGLE_HOST_KERNEL_STAGES = [
    ("kernel_total", 0),
    ("chunk_total", 1),
    ("post_sync", 2),
    ("local_input_to_ipc", 3),
    ("flag_poll_wait", 4),
    ("peer_ipc_to_output", 5),
    ("chunk_barrier", 6),
]


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


def write_trace(root, rank, launch, message_bytes=1024, schema="tilexr_perf_trace_report.v1", rank_size=2,
                block_dim=2, max_core_count=2):
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
        "max_core_count": max_core_count,
        "block_dim": block_dim,
        "stage_count": 7,
        "cycle_to_us_divisor": 50,
        "message_bytes": message_bytes,
        "stats": stats,
    }
    (launch_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    (launch_dir / "report.html").write_text("<html>single launch</html>\n", encoding="utf-8")


def write_full_stage_trace(root, rank, launch, message_bytes=4096, rank_size=2, block_dim=2, max_core_count=2):
    launch_dir = root / f"rank{rank}" / f"launch{launch}"
    launch_dir.mkdir(parents=True)
    base = 2000000 + rank * 100000 + launch * 10000
    stats = []
    for index, (stage, stage_id) in enumerate(SINGLE_HOST_KERNEL_STAGES):
        first = base + index * 100
        duration = 900 - index * 50 + rank * 10 + launch * 20
        stats.append(make_stat(rank, 0, stage, stage_id, first, duration, count=1, max_cycles=duration))
    trace = {
        "schema": "tilexr_perf_trace_report.v1",
        "op_type": 11,
        "op_name": "TileXRProfileProbe",
        "rank_size": rank_size,
        "max_core_count": max_core_count,
        "block_dim": block_dim,
        "stage_count": len(SINGLE_HOST_KERNEL_STAGES),
        "cycle_to_us_divisor": 50,
        "message_bytes": message_bytes,
        "stats": stats,
    }
    (launch_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    (launch_dir / "report.html").write_text("<html>single launch</html>\n", encoding="utf-8")


def write_host_info(root, rank, host, ip):
    rank_dir = root / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    (rank_dir / "host_info.json").write_text(
        json.dumps(
            {
                "schema": "tilexr_collective_profile_host.v1",
                "rank": rank,
                "host": host,
                "ip": ip,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_custom_trace(root, rank, launch, trace):
    launch_dir = root / f"rank{rank}" / f"launch{launch}"
    launch_dir.mkdir(parents=True)
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

    def test_writes_perfetto_trace_for_ui_perfetto_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0)
            write_trace(root, 1, 0)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            trace_json = json.loads((root / "trace.json").read_text(encoding="utf-8"))
            self.assertIn("traceEvents", trace_json)

            perfetto = json.loads((root / "perfetto_trace.json").read_text(encoding="utf-8"))
            self.assertIn("traceEvents", perfetto)
            self.assertIn({"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "rank0"}}, perfetto["traceEvents"])
            self.assertIn({"name": "thread_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name": "rank0/core0"}}, perfetto["traceEvents"])

            kernel = next(
                event
                for event in perfetto["traceEvents"]
                if event.get("ph") == "X" and event.get("name") == "launch0/rank0/kernel_total" and
                event.get("pid") == 0 and event.get("tid") == 0
            )
            self.assertEqual(kernel["cat"], "TileXRAllGather")
            self.assertEqual(kernel["ts"], 0.0)
            self.assertEqual(kernel["dur"], 20.0)
            self.assertEqual(kernel["args"]["stage"], "kernel_total")
            self.assertEqual(kernel["args"]["launch_id"], 0)
            self.assertEqual(kernel["args"]["source"], "rank0/launch0/trace.json")
            self.assertEqual(kernel["args"]["message_bytes"], 1024)
            self.assertEqual(kernel["args"]["rank_size"], 2)

    def test_rank_level_summary_highlights_slowest_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for launch in (0, 1):
                write_trace(root, 0, launch)
                write_trace(root, 1, launch)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "2", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            summary = index["groups"][0]["summary"]
            self.assertEqual(summary["slowest_rank"]["rank"], 1)
            self.assertEqual(summary["slowest_rank"]["slowest_launch_id"], 1)
            self.assertAlmostEqual(summary["slowest_rank"]["avg_kernel_us"], 21.2)
            self.assertAlmostEqual(summary["slowest_rank"]["max_kernel_us"], 22.2)

            rank_kernel = {item["rank"]: item for item in summary["rank_kernel"]}
            self.assertEqual(rank_kernel[0]["launch_count"], 2)
            self.assertAlmostEqual(rank_kernel[0]["avg_kernel_us"], 21.0)
            self.assertAlmostEqual(rank_kernel[0]["max_kernel_us"], 22.0)
            self.assertEqual(rank_kernel[1]["launch_count"], 2)
            self.assertAlmostEqual(rank_kernel[1]["avg_kernel_us"], 21.2)
            self.assertAlmostEqual(rank_kernel[1]["max_kernel_us"], 22.2)

            analysis = (root / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("Slowest rank: rank1 avg 21.200 us max 22.200 us at launch1", analysis)
            self.assertIn("Rank kernel totals: rank1 avg=21.200 us max=22.200 us; rank0 avg=21.000 us max=22.000 us", analysis)

            html = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("Rank-Level Summary", html)
            self.assertIn("rank1", html)
            self.assertIn("21.200", html)
            self.assertIn("22.200", html)

    def test_multihost_report_preserves_single_host_kernel_stage_granularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for launch in (0, 1):
                write_full_stage_trace(root, 0, launch)
                write_full_stage_trace(root, 1, launch)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "2", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            expected_stage_names = [stage for stage, _ in SINGLE_HOST_KERNEL_STAGES]
            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            group = index["groups"][0]
            self.assertEqual(group["op_name"], "TileXRProfileProbe")
            self.assertEqual(group["stage_count"], len(expected_stage_names))
            self.assertFalse(index["diagnostics"])

            stages_by_rank_launch = {}
            for bar in group["bars"]:
                stages_by_rank_launch.setdefault((bar["rank"], bar["launch_id"]), set()).add(bar["stage"])

            for launch in (0, 1):
                for rank in (0, 1):
                    self.assertEqual(
                        sorted(stages_by_rank_launch[(rank, launch)]),
                        sorted(expected_stage_names),
                    )

            html = (root / "report.html").read_text(encoding="utf-8")
            perfetto = json.loads((root / "perfetto_trace.json").read_text(encoding="utf-8"))
            perfetto_names = {
                event.get("name")
                for event in perfetto["traceEvents"]
                if event.get("ph") == "X"
            }
            for stage in expected_stage_names:
                self.assertIn(stage, html)
                self.assertIn(f"launch0/rank0/{stage}", perfetto_names)
                self.assertIn(f"launch1/rank1/{stage}", perfetto_names)

    def test_host_metadata_is_included_in_html_analysis_and_perfetto(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_host_info(root, 0, "host62", "141.62.24.62")
            write_host_info(root, 1, "host70", "141.62.24.70")
            write_trace(root, 0, 0)
            write_trace(root, 1, 0)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["hosts"]["0"]["host"], "host62")
            self.assertEqual(index["hosts"]["0"]["ip"], "141.62.24.62")
            self.assertEqual(index["hosts"]["1"]["host"], "host70")
            self.assertEqual(index["hosts"]["1"]["ip"], "141.62.24.70")
            rank0_bar = next(bar for bar in index["groups"][0]["bars"] if bar["rank"] == 0)
            rank1_bar = next(bar for bar in index["groups"][0]["bars"] if bar["rank"] == 1)
            self.assertEqual(rank0_bar["host"], "host62")
            self.assertEqual(rank1_bar["host_ip"], "141.62.24.70")

            analysis = (root / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("Hosts: rank0@host62(141.62.24.62), rank1@host70(141.62.24.70)", analysis)
            self.assertIn("Slowest rank: rank1@host70", analysis)

            html = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("Host", html)
            self.assertIn("rank1@host70", html)
            self.assertIn("141.62.24.70", html)

            perfetto = json.loads((root / "perfetto_trace.json").read_text(encoding="utf-8"))
            self.assertIn(
                {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "rank1@host70"}},
                perfetto["traceEvents"],
            )
            kernel = next(
                event
                for event in perfetto["traceEvents"]
                if event.get("ph") == "X" and event.get("name") == "launch0/rank1@host70/kernel_total"
            )
            self.assertEqual(kernel["args"]["host"], "host70")
            self.assertEqual(kernel["args"]["host_ip"], "141.62.24.70")

    def test_perfetto_trace_offsets_launches_and_adds_alignment_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for launch in (0, 1):
                write_trace(root, 0, launch)
                write_trace(root, 1, launch)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "2", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            perfetto = json.loads((root / "perfetto_trace.json").read_text(encoding="utf-8"))
            launch0_kernel = next(
                event for event in perfetto["traceEvents"]
                if event.get("ph") == "X" and event.get("name") == "launch0/rank0/kernel_total"
            )
            launch1_kernel = next(
                event for event in perfetto["traceEvents"]
                if event.get("ph") == "X" and event.get("name") == "launch1/rank0/kernel_total"
            )
            self.assertGreater(launch1_kernel["ts"], launch0_kernel["ts"])
            self.assertEqual(launch1_kernel["args"]["normalized_ts"], 0.0)
            self.assertEqual(launch1_kernel["args"]["launch_offset_us"], 70.2)

            marker_thread = {"name": "thread_name", "ph": "M", "pid": 0, "tid": 1000000, "args": {"name": "rank0/launch_windows"}}
            self.assertIn(marker_thread, perfetto["traceEvents"])
            window = next(
                event for event in perfetto["traceEvents"]
                if event.get("ph") == "X" and event.get("name") == "launch1/rank0/window"
            )
            self.assertEqual(window["cat"], "launch_window")
            self.assertEqual(window["ts"], 70.2)
            self.assertEqual(window["args"]["rank"], 0)
            self.assertEqual(window["args"]["launch_id"], 1)

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

    def test_multi_size_global_launch_ids_do_not_report_false_missing_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for launch in (0, 1):
                write_trace(root, 0, launch, message_bytes=1024)
                write_trace(root, 1, launch, message_bytes=1024)
            for launch in (2, 3):
                write_trace(root, 0, launch, message_bytes=2048)
                write_trace(root, 1, launch, message_bytes=2048)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "2", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            launch_groups = [group["launch_ids"] for group in index["groups"]]
            self.assertEqual(launch_groups, [[0, 1], [2, 3]])
            self.assertFalse(index["diagnostics"])

    def test_sampled_multi_size_launch_ids_use_global_sampling_modulo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for launch in (0, 2):
                write_trace(root, 0, launch, message_bytes=1024)
                write_trace(root, 1, launch, message_bytes=1024)
            write_trace(root, 0, 4, message_bytes=2048)
            write_trace(root, 1, 4, message_bytes=2048)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "3", "--profile-sample-every", "2")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            launch_groups = [group["launch_ids"] for group in index["groups"]]
            self.assertEqual(launch_groups, [[0, 2], [4]])
            self.assertFalse(index["diagnostics"])

    def test_launch_labels_drill_down_and_fit_uses_wrapper_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0, rank_size=1)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            html = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("function launchDrilldown(launchBars)", html)
            self.assertIn("line.href = launchDrilldown(launchBars)", html)
            self.assertIn("function timelineWidthAt(nextScale)", html)
            self.assertIn("wrap.clientWidth", html)
            self.assertIn("return xBase;", html)

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
            self.assertIn("message_bytes=1024", joined)
            self.assertIn("rank0/launch0/trace.json", joined)
            self.assertIn("message_bytes=2048", joined)
            self.assertIn("rank1/launch0/trace.json", joined)

    def test_keeps_incompatible_kernel_dimensions_out_of_valid_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0, block_dim=4, max_core_count=4)
            write_trace(root, 1, 0, block_dim=2, max_core_count=2)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index["groups"]), 2)
            joined = "\n".join(index["diagnostics"])
            self.assertIn("incompatible trace groups detected", joined)
            self.assertIn("max_core_count=4 block_dim=4", joined)
            self.assertIn("rank0/launch0/trace.json", joined)
            self.assertIn("max_core_count=2 block_dim=2", joined)
            self.assertIn("rank1/launch0/trace.json", joined)

    def test_incomplete_kernel_trace_is_reported_and_visible_in_perfetto(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = {
                "schema": "tilexr_perf_trace_report.v1",
                "incomplete": True,
                "incomplete_reason": "aclrtSynchronizeEvent stop failed ret=507035",
                "op_type": 3,
                "op_name": "TileXRAllGather",
                "rank_size": 2,
                "max_core_count": 4,
                "block_dim": 4,
                "stage_count": 7,
                "cycle_to_us_divisor": 50,
                "message_bytes": 4096,
                "stats": [],
            }
            write_custom_trace(root, 0, 0, trace)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["groups"][0]["block_dim"], 4)
            self.assertEqual(index["groups"][0]["max_core_count"], 4)
            self.assertTrue(index["groups"][0]["trace_statuses"][0]["incomplete"])
            self.assertIn("aclrtSynchronizeEvent stop failed", index["groups"][0]["trace_statuses"][0]["reason"])
            joined = "\n".join(index["diagnostics"])
            self.assertIn("incomplete trace in rank0/launch0/trace.json", joined)
            self.assertIn("missing trace for rank1 launch0", joined)

            analysis = (root / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("incomplete trace in rank0/launch0/trace.json", analysis)
            html = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("aclrtSynchronizeEvent stop failed ret=507035", html)

            perfetto = json.loads((root / "perfetto_trace.json").read_text(encoding="utf-8"))
            event = next(
                event for event in perfetto["traceEvents"]
                if event.get("ph") == "X" and event.get("name") == "launch0/rank0/incomplete_trace"
            )
            self.assertEqual(event["cat"], "trace_status")
            self.assertEqual(event["args"]["reason"], "aclrtSynchronizeEvent stop failed ret=507035")

    def test_rank_core_max_uses_stage_max_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launch_dir = root / "rank0" / "launch0"
            launch_dir.mkdir(parents=True)
            trace = {
                "schema": "tilexr_perf_trace_report.v1",
                "op_type": 3,
                "op_name": "TileXRAllGather",
                "rank_size": 1,
                "max_core_count": 2,
                "block_dim": 2,
                "stage_count": 7,
                "cycle_to_us_divisor": 50,
                "message_bytes": 1024,
                "stats": [
                    make_stat(0, 0, "long_total", 1, 1000, 1000, max_cycles=200),
                    make_stat(0, 1, "short_peak", 2, 2000, 100, max_cycles=900),
                ],
            }
            (launch_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            summary = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))["groups"][0]["summary"]
            self.assertEqual(summary["rank_core_max"]["core"], 1)
            self.assertEqual(summary["rank_core_max"]["stage"], "short_peak")
            self.assertEqual(summary["rank_core_max"]["max_us"], 18.0)

    def test_count_zero_slots_are_ignored_even_when_cycles_are_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = {
                "schema": "tilexr_perf_trace_report.v1",
                "op_type": 3,
                "op_name": "TileXRAllGather",
                "rank_size": 2,
                "max_core_count": 2,
                "block_dim": 2,
                "stage_count": 7,
                "cycle_to_us_divisor": 50,
                "message_bytes": 1024,
                "stats": [
                    make_stat(1, 0, "kernel_total", 0, 1000, 500, count=1, max_cycles=500),
                    {
                        "rank": 0,
                        "core": 0,
                        "stage": "kernel_total",
                        "stage_id": 0,
                        "count": 0,
                        "raw_cycles": 500000,
                        "min_cycles": 500000,
                        "max_cycles": 500000,
                        "first_start_cycle": 1000,
                        "last_end_cycle": 501000,
                        "aux0": 0,
                        "aux1": 0,
                        "sum_us": 10000.0,
                    },
                ],
            }
            write_custom_trace(root, 1, 0, trace)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            bars = index["groups"][0]["bars"]
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0]["rank"], 1)
            self.assertEqual(index["groups"][0]["summary"]["slowest_rank"]["rank"], 1)

            perfetto = json.loads((root / "perfetto_trace.json").read_text(encoding="utf-8"))
            event_names = [event.get("name") for event in perfetto["traceEvents"] if event.get("ph") == "X"]
            self.assertIn("launch0/rank1/kernel_total", event_names)
            self.assertNotIn("launch0/rank0/kernel_total", event_names)

    def test_missing_kernel_total_renders_unavailable_slowest_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launch_dir = root / "rank0" / "launch0"
            launch_dir.mkdir(parents=True)
            trace = {
                "schema": "tilexr_perf_trace_report.v1",
                "op_type": 3,
                "op_name": "TileXRAllGather",
                "rank_size": 1,
                "max_core_count": 1,
                "block_dim": 1,
                "stage_count": 7,
                "cycle_to_us_divisor": 50,
                "message_bytes": 1024,
                "stats": [make_stat(0, 0, "flag_poll_wait", 4, 1000, 200, max_cycles=50)],
            }
            (launch_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            analysis = (root / "analysis.md").read_text(encoding="utf-8")
            report = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("Slowest launch: unavailable", analysis)
            self.assertNotIn("launchNone", analysis)
            self.assertNotIn("launchNone", report)

    def test_invalid_schema_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0, schema="bad.schema")

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertFalse(index["groups"])
            self.assertIn("invalid schema in rank0/launch0/trace.json", "\n".join(index["diagnostics"]))

    def test_non_object_json_top_level_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launch_dir = root / "rank0" / "launch0"
            launch_dir.mkdir(parents=True)
            (launch_dir / "trace.json").write_text("[]", encoding="utf-8")

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            self.assertFalse(index["groups"])
            self.assertIn("invalid top-level trace type in rank0/launch0/trace.json", "\n".join(index["diagnostics"]))

    def test_malformed_stats_are_reported_while_valid_stats_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = {
                "schema": "tilexr_perf_trace_report.v1",
                "op_type": 3,
                "op_name": "TileXRAllGather",
                "rank_size": 1,
                "max_core_count": 1,
                "block_dim": 1,
                "stage_count": 7,
                "cycle_to_us_divisor": 50,
                "message_bytes": 1024,
                "stats": {},
            }
            write_custom_trace(root, 0, 0, trace)
            trace["stats"] = ["bad", make_stat(0, 0, "kernel_total", 0, 1000, 500, max_cycles=900)]
            write_custom_trace(root, 0, 1, trace)

            result = run_helper(root, "--warmup-iters", "0", "--iters", "2", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            index = json.loads((root / "trace_index.json").read_text(encoding="utf-8"))
            bars = index["groups"][0]["bars"]
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0]["stage"], "kernel_total")
            joined = "\n".join(index["diagnostics"])
            self.assertIn("invalid stats in rank0/launch0/trace.json", joined)
            self.assertIn("invalid stat entry in rank0/launch1/trace.json stats[0]", joined)

    def test_trace_controlled_fields_are_safe_inside_report_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_trace(root, 0, 0, rank_size=1)
            trace_path = root / "rank0" / "launch0" / "trace.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace["op_name"] = "TileXRAllGather</script><script>alert(1)</script>"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            result = run_helper(root, "--warmup-iters", "0", "--iters", "1", "--profile-sample-every", "1")
            self.assertEqual(result.returncode, 0, result.stderr)

            html = (root / "report.html").read_text(encoding="utf-8")
            script_body = html.split("const traceIndex = ", 1)[1].split(";\nlet scale", 1)[0]
            self.assertNotIn("</script>", script_body.lower())
            self.assertNotIn("<script>", script_body.lower())
            self.assertIn("\\u003c/script\\u003e", script_body)

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
