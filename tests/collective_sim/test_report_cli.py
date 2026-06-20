import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.collective_sim.test_semantics import two_rank_allgather
from tests.collective_sim.test_simulator import calibration, cross_server_send, topology
from tilexr_collective_sim.report import write_html_report, write_report_bundle, write_result, write_summary
from tilexr_collective_sim.simulator import simulate_algorithm


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tools" / "collective_sim" / "examples" / "allgather_1d_clos"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")}


class ReportCliTest(unittest.TestCase):
    def test_report_writes_json_csv_and_html_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = simulate_algorithm(two_rank_allgather(), topology(2), calibration(), 1024)

            write_result(result, out)
            write_summary([result], out / "summary.csv")
            write_html_report([result], out / "report.html")

            data = json.loads((out / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "tilexr_collective_sim_result.v1")
            summary = (out / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("algorithm,collective,rank_count,message_bytes,latency_us", summary)
            html = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("Algorithm Selection", html)
            self.assertIn("Congestion Drilldown", html)
            self.assertIn("Rank Timeline", html)
            self.assertIn("estimate", html)

    def test_report_bundle_writes_light_index_and_rank_timelines(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = simulate_algorithm(two_rank_allgather(), topology(2), calibration(), 1024)

            write_report_bundle([result], out)

            main = (out / "report.html").read_text(encoding="utf-8")
            rank0_path = out / "rank_reports" / "rank_000.html"
            rank1_path = out / "rank_reports" / "rank_001.html"
            rank0_trace_path = out / "rank_traces" / "rank_000.trace.json"
            rank0 = rank0_path.read_text(encoding="utf-8")
            rank0_trace = json.loads(rank0_trace_path.read_text(encoding="utf-8"))["traceEvents"]

            self.assertTrue(rank0_path.exists())
            self.assertTrue(rank1_path.exists())
            self.assertTrue(rank0_trace_path.exists())
            self.assertIn("rankSelect", main)
            self.assertIn("rank_reports/rank_000.html", main)
            self.assertIn("Bottleneck Summary", main)
            self.assertNotIn("send_r0_to_r1", main)
            self.assertIn("https://ui.perfetto.dev/", rank0)
            self.assertIn("../rank_traces/rank_000.trace.json", rank0)
            self.assertNotIn("timelineSvg", rank0)
            slice_events = [event for event in rank0_trace if event.get("ph") == "X"]
            self.assertEqual([event["name"] for event in slice_events], ["send_r0_to_r1"])
            self.assertNotIn("send_r1_to_r0", {event["name"] for event in slice_events})
            self.assertTrue(all(event.get("cat") == "send" for event in slice_events))

    def test_perfetto_trace_groups_clos_sends_by_rank_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = simulate_algorithm(cross_server_send(128), topology(128), calibration(), 1024)

            write_report_bundle([result], out)

            trace_events = json.loads((out / "rank_traces" / "rank_000.trace.json").read_text(encoding="utf-8"))["traceEvents"]
            process_names = {
                event["args"]["name"]
                for event in trace_events
                if event.get("ph") == "M" and event.get("name") == "process_name"
            }
            thread_names = {
                event["args"]["name"]
                for event in trace_events
                if event.get("ph") == "M" and event.get("name") == "thread_name"
            }
            slice_events = [event for event in trace_events if event.get("ph") == "X"]

            self.assertTrue(any("clos-link:rank0" in name for name in process_names))
            self.assertFalse(any("uplink:src:0" in name or name.startswith("clos:") for name in process_names))
            self.assertIn("send to rank8", thread_names)
            self.assertEqual([event["name"] for event in slice_events], ["send_r0_to_r8"])

    def test_cli_validate_and_run_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"

            validate = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tilexr_collective_sim.cli",
                    "validate",
                    str(EXAMPLE / "algorithm.json"),
                    "--topology",
                    str(EXAMPLE / "topology_64p.yaml"),
                ],
                cwd=str(ROOT),
                env=ENV,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr)
            self.assertIn("validation ok", validate.stdout)

            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tilexr_collective_sim.cli",
                    "run",
                    str(EXAMPLE / "case.yaml"),
                    "--out",
                    str(out),
                ],
                cwd=str(ROOT),
                env=ENV,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((out / "result.json").exists())
            self.assertTrue((out / "results.json").exists())
            self.assertTrue((out / "summary.csv").exists())
            self.assertTrue((out / "report.html").exists())
            self.assertTrue((out / "rank_reports" / "rank_000.html").exists())

    def test_cli_report_regenerates_html_from_results_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            regenerated = Path(tmp) / "regenerated.html"

            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tilexr_collective_sim.cli",
                    "run",
                    str(EXAMPLE / "case.yaml"),
                    "--out",
                    str(out),
                ],
                cwd=str(EXAMPLE),
                env=ENV,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(run.returncode, 0, run.stderr)

            report = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tilexr_collective_sim.cli",
                    "report",
                    str(out / "results.json"),
                    "--out",
                    str(regenerated),
                ],
                cwd=str(ROOT),
                env=ENV,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertIn("Algorithm Selection", regenerated.read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / "rank_reports" / "rank_000.html").exists())


if __name__ == "__main__":
    unittest.main()
