#!/usr/bin/env python3
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "demo"
    / "tilexr_udma_alltoall_group_trace_to_chrome.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tilexr_udma_alltoall_group_trace_to_chrome", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GroupTraceConverterTest(unittest.TestCase):
    def test_assigns_each_core_disjoint_cache_lines(self):
        self.assertEqual(MODULE.kernel_span_offset(0, 1) % 128, 0)
        first = MODULE.task_span_offset(0, 0, 0, 0, 0, 1, 1)
        second = MODULE.task_span_offset(0, 1, 0, 0, 0, 1, 1)
        self.assertEqual(first % 128, 0)
        self.assertEqual(second % 128, 0)
        self.assertGreaterEqual(second - first, 128)

    def make_trace(
        self, path, *, rank=0, magic=None, iteration_count=1,
        group_count=1, pass_count=1, core_count=64,
    ):
        data = bytearray(MODULE.TRACE_BYTES)
        struct.pack_into(
            MODULE.HEADER_FORMAT,
            data,
            0,
            MODULE.TRACE_MAGIC if magic is None else magic,
            MODULE.TRACE_VERSION,
            rank,
            iteration_count,
            group_count,
            pass_count,
            core_count,
            MODULE.PHASE_COUNT,
            1000,
            MODULE.TRACE_BYTES,
            MODULE.HEADER_BYTES,
            MODULE.TASK_BASE_OFFSET,
        )
        for core in (0, 16):
            struct.pack_into(
                "<QQ", data, MODULE.kernel_span_offset(0, core), 1000, 3000)

        spans = (
            (16, 0, 0, 0, 1100, 1200, rank, MODULE.NO_QP),
            (0, 0, 0, 1, 1200, 1300, 1, 3),
            (0, 0, 0, 2, 1300, 1400, 1, 3),
            (16, 0, 0, 3, 1400, 1500, 1, MODULE.NO_QP),
            (16, 0, 0, 4, 1500, 1600, 1, MODULE.NO_QP),
        )
        for core, group, pass_index, phase, begin, end, peer, qp in spans:
            struct.pack_into(
                MODULE.TASK_FORMAT,
                data,
                MODULE.task_span_offset(
                    0, core, group, pass_index, phase, group_count, pass_count),
                begin,
                end,
                peer,
                qp,
            )
        path.write_bytes(data)

    def test_converts_all_grouped_pipeline_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tilexr_group_trace_rank_0.bin"
            self.make_trace(path)

            trace = MODULE.build_chrome_trace([MODULE.read_rank_trace(path)])
            complete = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
            self.assertEqual(
                {event["name"] for event in complete},
                {
                    "kernel", "self-copy", "send-put-signal", "send-quiet",
                    "receive-wait", "receive-copy",
                },
            )
            send = next(event for event in complete if event["name"] == "send-put-signal")
            self.assertEqual(send["args"]["peer"], 1)
            self.assertEqual(send["args"]["qp"], 3)
            self.assertEqual(send["args"]["lane"], 0)
            self.assertEqual(trace["otherData"]["displayTimeUnit"], "ns")
            json.loads(json.dumps(trace))

    def test_reads_suffixed_stage_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tilexr_group_trace_primary_rank_0.bin"
            self.make_trace(path)

            rank_trace = MODULE.read_rank_trace(path)
            trace = MODULE.build_chrome_trace([rank_trace])

            self.assertEqual(rank_trace["path"], str(path))
            self.assertIn(str(path), trace["otherData"]["sources"])

    def test_normalizes_ranks_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "rank0.bin"
            second = Path(directory) / "rank1.bin"
            self.make_trace(first, rank=0)
            self.make_trace(second, rank=1)
            second_data = bytearray(second.read_bytes())
            for core in (0, 16):
                struct.pack_into(
                    "<QQ", second_data, MODULE.kernel_span_offset(0, core), 5000, 7000)
            offset = MODULE.task_span_offset(0, 0, 0, 0, 1, 1, 1)
            struct.pack_into(MODULE.TASK_FORMAT, second_data, offset, 5200, 5300, 2, 3)
            second.write_bytes(second_data)

            trace = MODULE.build_chrome_trace([
                MODULE.read_rank_trace(first), MODULE.read_rank_trace(second)])
            sends = [
                event for event in trace["traceEvents"]
                if event.get("name") == "send-put-signal"
            ]
            self.assertEqual([event["ts"] for event in sends], [0.2, 0.2])

    def test_rejects_bad_magic_size_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.bin"
            self.make_trace(path, magic=0)
            with self.assertRaisesRegex(ValueError, "magic"):
                MODULE.read_rank_trace(path)

            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "size"):
                MODULE.read_rank_trace(path)

            self.make_trace(path, core_count=63)
            with self.assertRaisesRegex(ValueError, "core/phase"):
                MODULE.read_rank_trace(path)

            self.make_trace(path, iteration_count=50, group_count=8, pass_count=64)
            with self.assertRaisesRegex(ValueError, "capacity"):
                MODULE.read_rank_trace(path)

    def test_rejects_half_written_span(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.bin"
            self.make_trace(path)
            data = bytearray(path.read_bytes())
            offset = MODULE.task_span_offset(0, 0, 0, 0, 1, 1, 1)
            struct.pack_into(MODULE.TASK_FORMAT, data, offset, 0, 1300, 1, 3)
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE.build_chrome_trace([MODULE.read_rank_trace(path)])

    def test_main_writes_json_without_dumps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rank0.bin"
            output = Path(directory) / "trace.json"
            self.make_trace(source)

            with mock.patch.object(MODULE.json, "dumps", side_effect=AssertionError("no bulk JSON")):
                with mock.patch("sys.argv", ["converter", str(source), "--output", str(output)]):
                    MODULE.main()

            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["traceEvents"])


if __name__ == "__main__":
    unittest.main()
