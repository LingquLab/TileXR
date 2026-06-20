import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tilexr_collective_sim.dsl import allgather_direct_algorithm
from tilexr_collective_sim.io import load_algorithm, load_calibration, load_case, load_topology
from tilexr_collective_sim.semantics import validate_allgather_semantics


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tools" / "collective_sim" / "examples" / "allgather_1d_clos"


class DslExampleTest(unittest.TestCase):
    def test_direct_allgather_uses_comm_buffers_and_copy_stages(self):
        data = allgather_direct_algorithm(rank_count=4, message_bytes=1024)
        roles = {buffer["role"] for buffer in data["buffers"]}
        op_types = {op["type"] for op in data["ops"]}
        self.assertIn("comm_buffer", roles)
        self.assertIn("user_input", roles)
        self.assertIn("user_output", roles)
        self.assertIn("copy", op_types)
        self.assertIn("send", op_types)

    def test_generated_algorithm_passes_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithm.json"
            path.write_text(json.dumps(allgather_direct_algorithm(4, 1024), indent=2), encoding="utf-8")
            report = validate_allgather_semantics(load_algorithm(path))
            self.assertTrue(report.ok, report.issues)

    def test_checked_in_example_files_load(self):
        algorithm = load_algorithm(EXAMPLE / "algorithm.json")
        self.assertTrue(validate_allgather_semantics(algorithm).ok)
        self.assertEqual(load_topology(EXAMPLE / "topology_64p.yaml").rank_count, 64)
        self.assertEqual(load_topology(EXAMPLE / "topology_128p_2to1.yaml").rank_count, 128)
        self.assertIn("sdma_800g", load_calibration(EXAMPLE / "calibration.yaml").curves)
        self.assertEqual(load_case(EXAMPLE / "case.yaml").message_bytes, (1024, 1048576))

    def test_generator_rewrites_algorithm_file(self):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE / "generate_allgather.py"), "--rank-count", "4", "--message-bytes", "1024"],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("algorithm.json", result.stdout)


if __name__ == "__main__":
    unittest.main()
