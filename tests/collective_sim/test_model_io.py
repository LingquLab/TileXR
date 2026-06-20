import json
import tempfile
import unittest
from pathlib import Path

from tilexr_collective_sim.io import load_algorithm, load_calibration, load_case, load_topology
from tilexr_collective_sim.model import COMM_BUFFER_ROLES, TRANSFER_OPS


class ModelIoTest(unittest.TestCase):
    def test_load_algorithm_with_buffer_roles_and_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithm.json"
            path.write_text(json.dumps({
                "name": "direct_allgather",
                "collective": "allgather",
                "rank_count": 2,
                "buffers": [
                    {"id": "r0_in", "rank": 0, "role": "user_input", "chunks": ["rank0.chunk0"]},
                    {"id": "r0_comm", "rank": 0, "role": "comm_buffer", "chunks": []},
                    {"id": "r1_comm", "rank": 1, "role": "comm_buffer", "chunks": []}
                ],
                "ops": [
                    {"id": "copy_r0", "type": "copy", "rank": 0, "bytes": 1024,
                     "src_buffer": "r0_in", "dst_buffer": "r0_comm", "mode": "sdma"},
                    {"id": "send_r0_to_r1", "type": "send", "rank": 0, "bytes": 1024,
                     "src_rank": 0, "dst_rank": 1, "src_buffer": "r0_comm",
                     "dst_buffer": "r1_comm", "deps": ["copy_r0"], "mode": "datacopy"}
                ],
                "metadata": {"chunk_bytes": 1024}
            }), encoding="utf-8")

            algorithm = load_algorithm(path)

            self.assertEqual(algorithm.name, "direct_allgather")
            self.assertEqual(algorithm.collective, "allgather")
            self.assertEqual(algorithm.rank_count, 2)
            self.assertEqual(algorithm.buffers["r0_comm"].role, "comm_buffer")
            self.assertEqual(algorithm.ops["send_r0_to_r1"].deps, ("copy_r0",))
            self.assertIn("comm_buffer", COMM_BUFFER_ROLES)
            self.assertIn("send", TRANSFER_OPS)

    def test_load_topology_calibration_and_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "topology.yaml").write_text(json.dumps({
                "topology": {
                    "type": "one_d_clos",
                    "rank_count": 128,
                    "ranks_per_server": 8,
                    "intra_server": {"kind": "full_mesh", "bandwidth_curve": "p2p_50g"},
                    "uplink": {"bandwidth_curve": "uplink_300g"},
                    "clos": {
                        "mode": "simplified_pool",
                        "non_blocking_until_ranks": 64,
                        "oversubscription": {"enabled_from_ranks": 128, "ratio": "2:1"},
                        "congestion": {"model": "destination_pool_fair_share"}
                    }
                }
            }), encoding="utf-8")
            (root / "calibration.yaml").write_text(json.dumps({
                "calibration": {
                    "curves": {
                        "p2p_50g": {"kind": "table", "points": [{"bytes": 1024, "gbps": 5}, {"bytes": 67108864, "gbps": 50}], "startup_latency_us": 2.0},
                        "uplink_300g": {"kind": "table", "points": [{"bytes": 1024, "gbps": 10}, {"bytes": 67108864, "gbps": 300}], "startup_latency_us": 3.0},
                        "sdma_800g": {"kind": "table", "points": [{"bytes": 1024, "gbps": 40}, {"bytes": 67108864, "gbps": 800}], "startup_latency_us": 1.0}
                    }
                }
            }), encoding="utf-8")
            (root / "case.yaml").write_text(json.dumps({
                "case": {
                    "algorithm": "algorithm.json",
                    "topology": "topology.yaml",
                    "calibration": "calibration.yaml",
                    "message_bytes": [1024, 4096],
                    "validate": True
                }
            }), encoding="utf-8")

            topology = load_topology(root / "topology.yaml")
            calibration = load_calibration(root / "calibration.yaml")
            case = load_case(root / "case.yaml")

            self.assertEqual(topology.rank_count, 128)
            self.assertEqual(topology.ranks_per_server, 8)
            self.assertEqual(topology.oversubscription_ratio, "2:1")
            self.assertIn("sdma_800g", calibration.curves)
            self.assertEqual(case.message_bytes, (1024, 4096))


if __name__ == "__main__":
    unittest.main()
