#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "moonep" / "combine_v2_trace.py"
METRICS = (
    "selection_load_us",
    "selection_select_us",
    "self_route_decode_us",
    "self_copy_us",
    "remote_route_decode_us",
    "remote_descriptor_us",
    "remote_wqe_build_us",
    "remote_submit_us",
)


def write_log(path, missing_profile=None, experts=32, correctness="passed",
              fullmesh_fields=True, invalid_fullmesh_order=False,
              final_only=False, legacy_grant=True):
    elapsed = (8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0)
    lines = []
    for iteration in ((1,) if final_only else (0, 1)):
        for rank, elapsed_ms in enumerate(elapsed):
            lines.append(
                "COMBINE_V2_SAMPLE bs=8192 iteration={} rank={} "
                "elapsed_ms={:.6f}{}".format(
                    iteration, rank, elapsed_ms + iteration,
                    " timing_source=kernel_profile" if final_only else ""))
            for core in range(8):
                if missing_profile == (iteration, rank, core):
                    continue
                base = 1000000 + rank * 100000 + core * 1000 + iteration * 10000
                points = [base]
                for index in range(1, 26):
                    points.append(points[-1] + index * 10)
                fields = [
                    "COMBINE_V2_PROFILE",
                    "bs=8192",
                    "iteration={}".format(iteration),
                    "rank={}".format(rank),
                    "core={}".format(core),
                    "cycles_per_us=1000",
                ]
                if fullmesh_fields:
                    if core == 0:
                        peer = (rank + 1) % 8
                        successor = rank if rank == 0 else (rank + 2) % 8
                        fullmesh_points = [base + value for value in
                                           (220, 230, 240, 250, 260)]
                        if not legacy_grant:
                            fullmesh_points[3:] = (0, 0)
                        if invalid_fullmesh_order:
                            fullmesh_points[2] = base + 245
                            fullmesh_points[3] = base + 240
                        fields.extend((
                            "profile_version=5",
                            "transport=fullmesh",
                            "fm_step=0",
                            "fm_peer={}".format(peer),
                            "fm_successor={}".format(successor),
                            "fm_logical_qp={}".format(32 + peer),
                        ))
                    else:
                        fullmesh_points = [0] * 5
                        fields.extend((
                            "profile_version=5",
                            "transport=none",
                            "fm_step=-1",
                            "fm_peer=-1",
                            "fm_successor=-1",
                            "fm_logical_qp=-1",
                        ))
                    fields.extend(
                        "{}={}".format(name, value)
                        for name, value in zip((
                            "fm_wqe_build_end",
                            "fm_submit_end",
                            "fm_cq_success",
                            "clos_grant_submit",
                            "clos_grant_cq_success",
                        ), fullmesh_points))
                fields.extend("t{}={}".format(index, value)
                              for index, value in enumerate(points))
                fields.extend("{}={:.3f}".format(name, core + 0.5)
                              for name in METRICS)
                lines.append(" ".join(fields))
    for rank in range(8):
        lines.append(
            "COMBINE_V2_RANK_PERF bs=8192 rank={} avg_ms=1.0 correctness={}".format(
                rank, correctness))
    lines.append(
        "COMBINE_V2_PERF bs=8192 k=16 h=3584 experts={} dtype=bf16 "
        "ranks=8 iterations=2 avg_ms=4.0 avg_alg_bw_GBps=1.0 "
        "max_ms=9.0 max_alg_bw_GBps=1.0 reduce=disabled correctness={}".format(
            experts, correctness))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CombineV2TraceTest(unittest.TestCase):
    def run_tool(self, log_path, output_path):
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                str(log_path),
                "--output", str(output_path),
                "--bs", "8192",
                "--world-size", "8",
                "--experts", "32",
                "--reduce", "disabled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_split_tool(self, log_path, output_dir):
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                str(log_path),
                "--split-output-dir", str(output_dir),
                "--prefix", "combine_v2_test",
                "--host", "test_host",
                "--bs", "8192",
                "--world-size", "8",
                "--experts", "32",
                "--reduce", "disabled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_exports_last_iteration_and_middle_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path)

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            trace = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["otherData"]["iteration"], 1)
            self.assertEqual(trace["otherData"]["reduce"], "disabled")
            selected = trace["otherData"]["selected_ranks"]
            self.assertEqual(
                [(row["role"], row["rank"], row["sorted_position"])
                 for row in selected],
                [("fastest", 1, 1), ("p50", 7, 4), ("slowest", 0, 8)],
            )
            self.assertTrue(all(row["critical_core"] == 0 for row in selected))
            self.assertTrue(all(row["profile_total_us"] > 0 for row in selected))
            self.assertTrue(all("step0_wait" in row["critical_core_stages_us"]
                                for row in selected))
            self.assertTrue(all("step1_send" not in
                                row["critical_core_stages_us"]
                                for row in selected))
            self.assertTrue(all("remote_submit_us" in
                                row["critical_core_metrics_us"]
                                for row in selected))
            totals = [event for event in trace["traceEvents"]
                      if event.get("name") == "combine_v2_no_reduce"]
            self.assertEqual(len(totals), 24)
            complete = [event for event in trace["traceEvents"]
                        if event.get("ph") == "X"]
            self.assertEqual(len(complete), 288)
            self.assertTrue(all(event["dur"] > 0 for event in totals))
            self.assertTrue(all(event["args"]["reduce"] == "disabled"
                                for event in totals))
            stages = {event.get("name") for event in trace["traceEvents"]}
            self.assertNotIn("step7_wait", stages)
            self.assertIn("inbound_wait", stages)
            self.assertIn("finalize_no_reduce", stages)
            self.assertIn("full_sync_submit", stages)
            fullmesh_events = [event for event in trace["traceEvents"]
                               if event.get("cat") in ("fullmesh", "grant")]
            self.assertEqual(len(fullmesh_events), 15)
            self.assertEqual(
                [event["name"] for event in fullmesh_events[:5]],
                [
                    "fullmesh_wqe_build_end",
                    "fullmesh_submit_end",
                    "fullmesh_cq_success",
                    "clos_grant_submit",
                    "clos_grant_cq_success",
                ],
            )
            self.assertTrue(all(
                event["args"]["data_transport"] == "fullmesh"
                for event in fullmesh_events))

            summary_path = root / "trace_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_ranks"], selected)

    def test_exports_trace_when_correctness_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path, correctness="failed")

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            trace = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["otherData"]["correctness"], "failed")
            self.assertTrue(all(
                row["correctness"] == "failed"
                for row in trace["otherData"]["selected_ranks"]))

    def test_exports_final_only_kernel_profile_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path, final_only=True)

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            trace = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["otherData"]["iteration"], 1)
            self.assertEqual(
                trace["otherData"]["timing_source"], "kernel_profile")
            self.assertIn(
                "slowest-core kernel profile",
                trace["otherData"]["selection_method"])
            totals = [event for event in trace["traceEvents"]
                      if event.get("name") == "combine_v2_no_reduce"]
            self.assertTrue(all(
                event["args"]["timing_source"] == "kernel_profile" and
                "kernel_profile_elapsed_ms" in event["args"] and
                "acl_event_elapsed_ms" not in event["args"]
                for event in totals))

            split_dir = root / "traces"
            split_result = self.run_split_tool(log_path, split_dir)
            self.assertEqual(
                split_result.returncode, 0, split_result.stderr)
            edge_paths = sorted(split_dir.glob("*_edge_trace.json"))
            self.assertEqual(len(edge_paths), 3)
            for edge_path in edge_paths:
                edge = json.loads(edge_path.read_text(encoding="utf-8"))
                metadata = edge["metadata"]
                self.assertEqual(
                    metadata["timing_source"], "kernel_profile")
                self.assertIn("kernel_profile_elapsed_ms", metadata)
                self.assertNotIn("acl_event_elapsed_ms", metadata)

    def test_rejects_incomplete_final_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path, missing_profile=(1, 7, 7))

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 2)
            self.assertIn("missing profile cores [7]", result.stderr)
            self.assertFalse(output_path.exists())

    def test_rejects_shape_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path, experts=64)

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 2)
            self.assertIn("experts=64 (expected 32)", result.stderr)
            self.assertFalse(output_path.exists())

    def test_accepts_legacy_profile_without_fullmesh_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path, fullmesh_fields=False)

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            trace = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(any(
                event.get("cat") in ("fullmesh", "grant")
                for event in trace["traceEvents"]))

    def test_rejects_invalid_fullmesh_event_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path, invalid_fullmesh_order=True)

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid Fullmesh event order", result.stderr)
            self.assertFalse(output_path.exists())

    def test_accepts_fullmesh_profile_without_legacy_grant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "combine.log"
            output_path = root / "trace.json"
            write_log(log_path, legacy_grant=False)

            result = self.run_tool(log_path, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            trace = json.loads(output_path.read_text(encoding="utf-8"))
            fullmesh_events = [
                event for event in trace["traceEvents"]
                if event.get("cat") in ("fullmesh", "grant")
            ]
            self.assertEqual(len(fullmesh_events), 9)
            self.assertFalse(any(
                event.get("cat") == "grant" for event in fullmesh_events))
            self.assertTrue(all(
                event["args"]["grant_transport"] == "none"
                for event in fullmesh_events))


if __name__ == "__main__":
    unittest.main()
