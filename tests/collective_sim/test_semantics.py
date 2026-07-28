import unittest

from tilexr_collective_sim.model import AlgorithmSpec, BufferSpec, OpSpec
from tilexr_collective_sim.semantics import validate_allgather_semantics


def two_rank_allgather(include_remote_copy=True):
    buffers = {
        "r0_in": BufferSpec("r0_in", 0, "user_input", ("rank0.chunk0",)),
        "r1_in": BufferSpec("r1_in", 1, "user_input", ("rank1.chunk0",)),
        "r0_c0": BufferSpec("r0_c0", 0, "comm_buffer", ()),
        "r0_c1": BufferSpec("r0_c1", 0, "comm_buffer", ()),
        "r1_c0": BufferSpec("r1_c0", 1, "comm_buffer", ()),
        "r1_c1": BufferSpec("r1_c1", 1, "comm_buffer", ()),
        "r0_out": BufferSpec("r0_out", 0, "user_output", ()),
        "r1_out": BufferSpec("r1_out", 1, "user_output", ()),
    }
    ops = {
        "copy_r0_in": OpSpec("copy_r0_in", "copy", 0, 1024, "r0_in", "r0_c0", mode="sdma"),
        "copy_r1_in": OpSpec("copy_r1_in", "copy", 1, 1024, "r1_in", "r1_c1", mode="sdma"),
        "send_r0_to_r1": OpSpec("send_r0_to_r1", "send", 0, 1024, "r0_c0", "r1_c0", 0, 1, ("copy_r0_in",), "datacopy"),
        "send_r1_to_r0": OpSpec("send_r1_to_r0", "send", 1, 1024, "r1_c1", "r0_c1", 1, 0, ("copy_r1_in",), "datacopy"),
        "out_r0_local": OpSpec("out_r0_local", "copy", 0, 1024, "r0_c0", "r0_out", deps=("copy_r0_in",), mode="sdma"),
        "out_r1_local": OpSpec("out_r1_local", "copy", 1, 1024, "r1_c1", "r1_out", deps=("copy_r1_in",), mode="sdma"),
    }
    if include_remote_copy:
        ops.update({
            "out_r0_remote": OpSpec("out_r0_remote", "copy", 0, 1024, "r0_c1", "r0_out", deps=("send_r1_to_r0",), mode="sdma"),
            "out_r1_remote": OpSpec("out_r1_remote", "copy", 1, 1024, "r1_c0", "r1_out", deps=("send_r0_to_r1",), mode="sdma"),
        })
    return AlgorithmSpec("two_rank", "allgather", 2, buffers, ops, {})


class SemanticsTest(unittest.TestCase):
    def test_allgather_symbolic_execution_passes(self):
        report = validate_allgather_semantics(two_rank_allgather())
        self.assertTrue(report.ok, report.issues)

    def test_missing_remote_chunk_fails(self):
        report = validate_allgather_semantics(two_rank_allgather(include_remote_copy=False))
        self.assertFalse(report.ok)
        self.assertIn("allgather_missing_chunk", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
