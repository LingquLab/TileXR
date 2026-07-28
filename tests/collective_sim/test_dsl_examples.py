import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tilexr_collective_sim.dsl import allgather_direct_algorithm
from tilexr_collective_sim.io import load_algorithm, load_calibration, load_case, load_sweep, load_topology
from tilexr_collective_sim.model import TopologySpec
from tilexr_collective_sim.semantics import validate_allgather_semantics
from tilexr_collective_sim.simulator import simulate_algorithm


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

    def test_generated_algorithm_transfer_events_use_runtime_message_bytes(self):
        algorithm_data = allgather_direct_algorithm(rank_count=4, message_bytes=1024)
        calibration = load_calibration(EXAMPLE / "calibration.yaml")
        topology = TopologySpec(
            "one_d_clos",
            4,
            8,
            "p2p_50g",
            "uplink_300g",
            64,
            128,
            "2:1",
            "destination_pool_fair_share",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithm.json"
            path.write_text(json.dumps(algorithm_data, indent=2), encoding="utf-8")
            algorithm = load_algorithm(path)

        result_1k = simulate_algorithm(algorithm, topology, calibration, 1024, validate=True)
        result_1m = simulate_algorithm(algorithm, topology, calibration, 1048576, validate=True)

        self.assertTrue(result_1k.validation.ok, result_1k.validation.issues)
        self.assertTrue(result_1m.validation.ok, result_1m.validation.issues)
        self.assertEqual({event.message_bytes for event in result_1k.events}, {1024})
        self.assertEqual({event.message_bytes for event in result_1m.events}, {1048576})

    def test_checked_in_example_files_load(self):
        algorithm = load_algorithm(EXAMPLE / "algorithm.json")
        algorithm_128p = load_algorithm(EXAMPLE / "algorithm_128p.json")
        self.assertTrue(validate_allgather_semantics(algorithm).ok)
        self.assertTrue(validate_allgather_semantics(algorithm_128p).ok)
        self.assertEqual(algorithm.rank_count, 64)
        self.assertEqual(algorithm_128p.rank_count, 128)
        self.assertEqual(load_topology(EXAMPLE / "topology_64p.yaml").rank_count, 64)
        self.assertEqual(load_topology(EXAMPLE / "topology_128p_2to1.yaml").rank_count, 128)
        self.assertIn("sdma_800g", load_calibration(EXAMPLE / "calibration.yaml").curves)
        self.assertEqual(load_case(EXAMPLE / "case.yaml").message_bytes, (1024, 1048576))

    def test_checked_in_case_matches_topology_and_simulates(self):
        case = load_case(EXAMPLE / "case.yaml")
        algorithm = load_algorithm(EXAMPLE / case.algorithm)
        topology = load_topology(EXAMPLE / case.topology)
        calibration = load_calibration(EXAMPLE / case.calibration)

        self.assertEqual(algorithm.rank_count, topology.rank_count)
        result = simulate_algorithm(algorithm, topology, calibration, case.message_bytes[0], validate=case.validate)
        self.assertTrue(result.validation.ok, result.validation.issues)
        self.assertGreater(len(result.events), 0)

    def test_checked_in_sweep_pairs_match_rank_count_and_simulate(self):
        data = load_sweep(EXAMPLE / "sweep.yaml")
        calibration = load_calibration(EXAMPLE / data["calibration"])
        message_bytes = min(int(value) for value in data["message_bytes"])
        observed_pairs = set()

        for case in data["cases"]:
            algorithm_name = case["algorithm"]
            topology_name = case["topology"]
            algorithm = load_algorithm(EXAMPLE / algorithm_name)
            topology = load_topology(EXAMPLE / topology_name)
            with self.subTest(algorithm=algorithm_name, topology=topology_name):
                self.assertEqual(algorithm.rank_count, topology.rank_count)
                observed_pairs.add((algorithm.rank_count, topology.rank_count))
                result = simulate_algorithm(
                    algorithm,
                    topology,
                    calibration,
                    message_bytes,
                    validate=bool(data.get("validate", True)),
                )
                self.assertTrue(result.validation.ok, result.validation.issues)
                self.assertGreater(len(result.events), 0)

        self.assertIn((64, 64), observed_pairs)
        self.assertIn((128, 128), observed_pairs)

    def test_generator_rewrites_temp_algorithm_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied_example = Path(tmp) / "allgather_1d_clos"
            shutil.copytree(EXAMPLE, copied_example)
            result = subprocess.run(
                [
                    sys.executable,
                    str(copied_example / "generate_allgather.py"),
                    "--rank-count",
                    "4",
                    "--message-bytes",
                    "1024",
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("algorithm.json", result.stdout)
            self.assertEqual(load_algorithm(copied_example / "algorithm.json").rank_count, 4)
            self.assertEqual(load_algorithm(EXAMPLE / "algorithm.json").rank_count, 64)


if __name__ == "__main__":
    unittest.main()
