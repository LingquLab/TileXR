import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.collective_sim.test_semantics import two_rank_allgather
from tests.collective_sim.test_simulator import calibration, topology
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
            rank0 = rank0_path.read_text(encoding="utf-8")

            self.assertTrue(rank0_path.exists())
            self.assertTrue(rank1_path.exists())
            self.assertIn("rankSelect", main)
            self.assertIn("rank_reports/rank_000.html", main)
            self.assertIn("Bottleneck Summary", main)
            self.assertNotIn("send_r0_to_r1", main)
            self.assertIn("timelineSvg", rank0)
            self.assertIn("zoomRange", rank0)
            self.assertIn("resultSelect", rank0)
            self.assertIn("timeline-data", rank0)
            self.assertIn("eventDetail", rank0)
            self.assertIn("selectEvent", rank0)
            self.assertIn("data-event-id", rank0)
            self.assertIn("lanes", rank0)
            self.assertIn("p2p:s0:1-", rank0)
            self.assertIn("send_r1_to_r0", rank0)

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
