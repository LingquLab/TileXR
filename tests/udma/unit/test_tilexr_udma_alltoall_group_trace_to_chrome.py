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
        version=None, phase_count=None,
    ):
        version = MODULE.TRACE_VERSION if version is None else version
        phase_count = (
            MODULE.CURRENT_PHASE_COUNT if phase_count is None else phase_count)
        task_format = (
            MODULE.TASK_FORMAT if version == MODULE.TRACE_VERSION
            else MODULE.LEGACY_TASK_FORMAT)
        task_bytes = struct.calcsize(task_format)
        header = struct.pack(
            MODULE.HEADER_FORMAT,
            MODULE.TRACE_MAGIC if magic is None else magic,
            version,
            rank,
            iteration_count,
            group_count,
            pass_count,
            core_count,
            phase_count,
            1000,
            MODULE.TRACE_BYTES,
            MODULE.HEADER_BYTES,
            MODULE.TASK_BASE_OFFSET,
        )
        spans = (
            (32, 0, 0, 0, 1100, 1200, rank, MODULE.NO_QP),
            (0, 0, 0, 1, 1200, 1300, 1, 3),
            (16, 0, 0, 1, 1250, 1350, 2, 4),
            (0, 0, 0, 2, 1300, 1400, 1, 3),
            (32, 0, 0, 3, 1400, 1500, 1, MODULE.NO_QP),
            (32, 0, 0, 4, 1500, 1600, 1, MODULE.NO_QP),
            (0, 0, 0, 5, 1050, 1100, 1, MODULE.NO_QP),
            (32, 0, 0, 6, 1600, 1650, 1, MODULE.NO_QP),
            (32, 0, 0, 7, 1650, 1700, 1, MODULE.NO_QP),
            (32, 0, 0, 8, 1605, 1615, 1, MODULE.NO_QP),
            (32, 0, 0, 9, 1615, 1625, 1, MODULE.NO_QP),
            (32, 0, 0, 10, 1625, 1635, 1, MODULE.NO_QP),
            (32, 0, 0, 11, 1635, 1645, 1, MODULE.NO_QP),
        )
        with path.open("wb") as stream:
            stream.truncate(MODULE.TRACE_BYTES)
            stream.seek(0)
            stream.write(header)
            for core in (0, 16, 32):
                stream.seek(MODULE.kernel_span_offset(0, core))
                stream.write(struct.pack("<QQ", 1000, 3000))
            for core, group, pass_index, phase, begin, end, peer, qp in spans:
                if phase >= phase_count:
                    continue
                stream.seek(MODULE.task_span_offset(
                    0, core, group, pass_index, phase, group_count, pass_count,
                    phase_count, task_bytes))
                values = [begin, end, peer, qp]
                if version == MODULE.TRACE_VERSION:
                    values.extend([0, 62, 0, 64, 33])
                stream.write(struct.pack(task_format, *values))

    def write_at(self, path, offset, payload):
        with path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(payload)

    def test_uses_128mib_files_and_reads_only_populated_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.bin"
            self.make_trace(path)

            rank_trace = MODULE.read_rank_trace(path)

            self.assertEqual(MODULE.TRACE_BYTES, 128 * 1024 * 1024)
            self.assertEqual(path.stat().st_size, MODULE.TRACE_BYTES)
            self.assertEqual(
                len(rank_trace["data"]), MODULE.layout_bytes(1, 1, 1))

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
                    "receive-wait", "receive-copy", "credit-wait",
                    "sdma-submit", "sdma-wait",
                    "sdma-prepare", "sdma-cache-clean", "sdma-dsb",
                    "sdma-doorbell",
                },
            )
            send = next(event for event in complete if event["name"] == "send-put-signal")
            self.assertEqual(send["args"]["peer"], 1)
            self.assertEqual(send["args"]["qp"], 3)
            self.assertEqual(send["args"]["lane"], 0)
            second_send = next(
                event for event in complete
                if event["name"] == "send-put-signal" and event["tid"] == 16)
            self.assertEqual(second_send["args"]["role"], "send")
            self.assertEqual(second_send["args"]["lane"], 0)
            self.assertEqual(trace["otherData"]["displayTimeUnit"], "ns")
            submit = next(event for event in complete if event["name"] == "sdma-submit")
            self.assertEqual(submit["args"]["sdmaTail"], 62)
            self.assertEqual(submit["args"]["sdmaNewTail"], 0)
            self.assertEqual(submit["args"]["sdmaDepth"], 64)
            self.assertTrue(submit["args"]["sdmaWrapped"])
            json.loads(json.dumps(trace))

    def test_labels_32_send_and_32_receive_cores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tilexr_group_trace_rank_0.bin"
            self.make_trace(path)

            trace = MODULE.build_chrome_trace([MODULE.read_rank_trace(path)])
            complete = [
                event for event in trace["traceEvents"] if event.get("ph") == "X"]
            thread_names = {
                event["tid"]: event["args"]["name"]
                for event in trace["traceEvents"]
                if event.get("name") == "thread_name"
            }
            self.assertEqual(thread_names[0], "core00 send")
            self.assertEqual(thread_names[2], "core02 send")
            self.assertEqual(thread_names[16], "core16 send")
            self.assertEqual(thread_names[31], "core31 send")
            self.assertEqual(thread_names[32], "core32 receive")
            self.assertEqual(thread_names[63], "core63 receive")
            receive = next(event for event in complete if event["name"] == "receive-copy")
            self.assertEqual(receive["args"]["lane"], 0)
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
            for core in (0, 16, 32):
                self.write_at(
                    second, MODULE.kernel_span_offset(0, core),
                    struct.pack("<QQ", 5000, 7000))
            offset = MODULE.task_span_offset(0, 0, 0, 0, 1, 1, 1)
            self.write_at(
                second, offset,
                struct.pack(MODULE.TASK_FORMAT, 5200, 5300, 2, 3,
                            0, 0, 0, 0, 0))

            trace = MODULE.build_chrome_trace([
                MODULE.read_rank_trace(first), MODULE.read_rank_trace(second)])
            sends = [
                event for event in trace["traceEvents"]
                if event.get("name") == "send-put-signal" and event["tid"] == 0
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
            offset = MODULE.task_span_offset(0, 0, 0, 0, 1, 1, 1)
            self.write_at(
                path, offset,
                struct.pack(MODULE.TASK_FORMAT, 0, 1300, 1, 3,
                            0, 0, 0, 0, 0))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE.build_chrome_trace([MODULE.read_rank_trace(path)])

    def test_reads_legacy_five_phase_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.bin"
            self.make_trace(
                path, version=MODULE.LEGACY_TRACE_VERSION,
                phase_count=MODULE.PHASE_COUNT)

            rank_trace = MODULE.read_rank_trace(path)
            trace = MODULE.build_chrome_trace([rank_trace])

            self.assertEqual(rank_trace["header"]["phase_count"], 5)
            names = {
                event["name"] for event in trace["traceEvents"]
                if event.get("ph") == "X"
            }
            self.assertNotIn("credit-wait", names)

    def test_reads_version_two_six_phase_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version2.bin"
            self.make_trace(path, version=2, phase_count=MODULE.TRACE_V2_PHASE_COUNT)
            rank_trace = MODULE.read_rank_trace(path)
            self.assertEqual(rank_trace["header"]["phase_count"], 6)
            names = {
                event["name"] for event in MODULE.build_chrome_trace([rank_trace])["traceEvents"]
                if event.get("ph") == "X"
            }
            self.assertIn("credit-wait", names)
            self.assertNotIn("sdma-submit", names)

    def test_reads_version_three_eight_phase_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version3.bin"
            self.make_trace(path, version=3, phase_count=MODULE.TRACE_V3_PHASE_COUNT)
            rank_trace = MODULE.read_rank_trace(path)
            self.assertEqual(rank_trace["header"]["phase_count"], 8)
            names = {
                event["name"] for event in MODULE.build_chrome_trace([rank_trace])["traceEvents"]
                if event.get("ph") == "X"
            }
            self.assertIn("sdma-submit", names)
            self.assertNotIn("sdma-prepare", names)

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
