#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().with_name("run_udma_alltoall_perf.py")
SPEC = importlib.util.spec_from_file_location("run_udma_alltoall_perf", SCRIPT)
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)


class UDMAAllToAllPerfScriptTest(unittest.TestCase):
    def test_parse_size_to_bytes(self):
        self.assertEqual(perf.parse_size_to_bytes("4096"), 4096)
        self.assertEqual(perf.parse_size_to_bytes("4M"), 4 * 1024 * 1024)
        self.assertEqual(perf.parse_size_to_bytes("8MiB"), 8 * 1024 * 1024)
        self.assertEqual(perf.parse_size_to_bytes("1g"), 1024 * 1024 * 1024)

    def test_elements_per_peer_requires_int32_alignment(self):
        self.assertEqual(perf.bytes_to_int32_elements(8 * 1024 * 1024), 2 * 1024 * 1024)
        with self.assertRaises(ValueError):
            perf.bytes_to_int32_elements(1025)

    def test_select_hosts_for_world_size(self):
        hosts = ["h0", "h1", "h2"]
        self.assertEqual(perf.select_hosts(hosts, 8, 8), ["h0"])
        self.assertEqual(perf.select_hosts(hosts, 16, 8), ["h0", "h1"])
        with self.assertRaises(ValueError):
            perf.select_hosts(hosts, 20, 8)
        with self.assertRaises(ValueError):
            perf.select_hosts(hosts, 32, 8)

    def test_parse_demo_metrics(self):
        log = """
        [rank 0] alltoall udma-bigdata 100 iters(total=100 pass/iter) total=128.6929 ms perIter=1286.929 us payload=2.68435e+08 bytes bw=208.586 GB/s
        """
        metric = perf.parse_demo_metric(log)
        self.assertEqual(metric.rank, 0)
        self.assertEqual(metric.iters, 100)
        self.assertAlmostEqual(metric.total_ms, 128.6929)
        self.assertAlmostEqual(metric.per_iter_us, 1286.929)
        self.assertAlmostEqual(metric.payload_bytes, 2.68435e8)
        self.assertAlmostEqual(metric.bandwidth_gbps, 208.586)

    def test_summarize_metrics(self):
        metrics = [
            perf.RankMetric(rank=0, iters=10, total_ms=10.0, per_iter_us=1000.0, payload_bytes=1024.0, bandwidth_gbps=1.0),
            perf.RankMetric(rank=1, iters=10, total_ms=20.0, per_iter_us=2000.0, payload_bytes=1024.0, bandwidth_gbps=2.0),
            perf.RankMetric(rank=2, iters=10, total_ms=30.0, per_iter_us=3000.0, payload_bytes=1024.0, bandwidth_gbps=3.0),
        ]
        summary = perf.summarize_metrics(metrics)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min_us"], 1000.0)
        self.assertEqual(summary["p50_us"], 2000.0)
        self.assertEqual(summary["max_us"], 3000.0)
        self.assertEqual(summary["mean_us"], 2000.0)

    def test_infers_external_tool_env_from_cann_env(self):
        tool_env = perf.infer_tool_env("/home/user/tilexr/env/cann/cann/set_env.sh")
        self.assertEqual(tool_env, "/home/user/tilexr/env")
        prelude = perf.remote_env_prelude("/home/user/tilexr/env/cann/cann/set_env.sh", tool_env)
        self.assertIn("/home/user/tilexr/env/util/cmake/bin", prelude)
        self.assertIn("source /home/user/tilexr/env/cann/cann/set_env.sh", prelude)


if __name__ == "__main__":
    unittest.main()
