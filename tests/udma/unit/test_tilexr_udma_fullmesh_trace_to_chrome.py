#!/usr/bin/env python3
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "demo" / "tilexr_udma_fullmesh_trace_to_chrome.py"
SPEC = importlib.util.spec_from_file_location("tilexr_udma_fullmesh_trace_to_chrome", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullmeshTraceConverterTest(unittest.TestCase):
    def make_trace(self, path, *, rank=3, version=1, iteration_count=1, pass_count=1):
        data = bytearray(MODULE.TRACE_BYTES)
        struct.pack_into(
            MODULE.HEADER_FORMAT,
            data,
            0,
            MODULE.TRACE_MAGIC,
            version,
            rank,
            iteration_count,
            pass_count,
            MODULE.MAX_CORES,
            16,
            MODULE.PHASE_COUNT,
            1000,
            MODULE.TRACE_BYTES,
            MODULE.HEADER_BYTES,
            MODULE.TASK_BASE_OFFSET,
        )
        struct.pack_into("<QQ", data, MODULE.kernel_span_offset(0, 16, 0), 1000, 2000)
        struct.pack_into("<QQ", data, MODULE.kernel_span_offset(0, 34, 0), 1000, 2100)
        struct.pack_into("<QQ", data, MODULE.task_span_offset(0, 16, 0, 8, 5, pass_count, 16), 1100, 1200)
        struct.pack_into("<QQ", data, MODULE.task_span_offset(0, 34, 0, 8, 13, pass_count, 16), 1300, 1400)
        path.write_bytes(data)

    def test_converts_core_peer_phase_and_raw_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tilexr_fullmesh_trace_rank_3.bin"
            self.make_trace(path)

            trace = MODULE.build_chrome_trace([MODULE.read_rank_trace(path)])
            complete = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
            data_put = next(event for event in complete if event["name"] == "data-put")
            ack = next(event for event in complete if event["name"] == "ACK")

            self.assertEqual(data_put["tid"], 16)
            self.assertEqual(data_put["args"]["peer"], 8)
            self.assertEqual(data_put["args"]["beginCycle"], 1100)
            self.assertEqual(data_put["dur"], 0.1)
            self.assertEqual(ack["tid"], 34)
            self.assertEqual(ack["args"]["phase"], 13)
            self.assertNotIn("displayTimeUnit", trace)
            json.loads(json.dumps(trace))

    def test_normalizes_each_rank_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "rank3.bin"
            second = Path(directory) / "rank4.bin"
            self.make_trace(first, rank=3)
            self.make_trace(second, rank=4)
            second_data = bytearray(second.read_bytes())
            struct.pack_into("<QQ", second_data, MODULE.kernel_span_offset(0, 16, 0), 5000, 6000)
            struct.pack_into("<QQ", second_data, MODULE.kernel_span_offset(0, 34, 0), 5000, 6100)
            struct.pack_into("<QQ", second_data, MODULE.task_span_offset(0, 16, 0, 8, 5, 1, 16), 5100, 5200)
            second.write_bytes(second_data)

            trace = MODULE.build_chrome_trace([
                MODULE.read_rank_trace(first), MODULE.read_rank_trace(second)])
            data_put = [event for event in trace["traceEvents"] if event.get("name") == "data-put"]
            self.assertEqual([event["ts"] for event in data_put], [0.1, 0.1])

    def test_places_iterations_sequentially(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank3.bin"
            self.make_trace(path, iteration_count=2)
            data = bytearray(path.read_bytes())
            struct.pack_into("<QQ", data, MODULE.kernel_span_offset(1, 16, 0), 5000, 6000)
            struct.pack_into("<QQ", data, MODULE.kernel_span_offset(1, 34, 0), 5000, 6100)
            struct.pack_into(
                "<QQ", data, MODULE.task_span_offset(1, 16, 0, 8, 5, 1, 16), 5100, 5200)
            path.write_bytes(data)

            trace = MODULE.build_chrome_trace([MODULE.read_rank_trace(path)])
            kernels = [
                event for event in trace["traceEvents"]
                if event.get("name") == "kernel" and event["tid"] == 16
            ]
            self.assertGreater(kernels[1]["ts"], kernels[0]["ts"] + kernels[0]["dur"])

    def test_rejects_unknown_version_truncation_and_capacity_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.bin"
            self.make_trace(path, version=99)
            with self.assertRaisesRegex(ValueError, "version"):
                MODULE.read_rank_trace(path)

            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "size"):
                MODULE.read_rank_trace(path)

            self.make_trace(path, iteration_count=50, pass_count=4)
            with self.assertRaisesRegex(ValueError, "capacity"):
                MODULE.read_rank_trace(path)

    def test_rejects_half_written_span(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.bin"
            self.make_trace(path)
            data = bytearray(path.read_bytes())
            offset = MODULE.task_span_offset(0, 16, 0, 8, 5, 1, 16)
            struct.pack_into("<QQ", data, offset, 0, 1200)
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE.build_chrome_trace([MODULE.read_rank_trace(path)])

    def test_main_streams_json_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rank3.bin"
            output = Path(directory) / "trace.json"
            self.make_trace(source)

            with mock.patch.object(MODULE.json, "dumps", side_effect=AssertionError("no bulk JSON")):
                with mock.patch("sys.argv", ["converter", str(source), "--output", str(output)]):
                    MODULE.main()

            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(parsed["traceEvents"])


if __name__ == "__main__":
    unittest.main()
