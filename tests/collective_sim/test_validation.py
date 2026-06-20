import unittest

from tilexr_collective_sim.model import AlgorithmSpec, BufferSpec, OpSpec, TopologySpec
from tilexr_collective_sim.validation import topological_ops, validate_static


def topology(rank_count=2):
    return TopologySpec(
        type="one_d_clos",
        rank_count=rank_count,
        ranks_per_server=8,
        p2p_curve="p2p_50g",
        uplink_curve="uplink_300g",
        non_blocking_until_ranks=64,
        oversubscription_enabled_from_ranks=128,
        oversubscription_ratio="2:1",
        congestion_model="destination_pool_fair_share",
    )


def valid_algorithm():
    buffers = {
        "r0_in": BufferSpec("r0_in", 0, "user_input", ("rank0.chunk0",)),
        "r0_comm": BufferSpec("r0_comm", 0, "comm_buffer", ()),
        "r1_comm": BufferSpec("r1_comm", 1, "comm_buffer", ()),
    }
    ops = {
        "copy_r0": OpSpec("copy_r0", "copy", 0, 1024, "r0_in", "r0_comm", mode="sdma"),
        "send_r0": OpSpec("send_r0", "send", 0, 1024, "r0_comm", "r1_comm", 0, 1, ("copy_r0",), "datacopy"),
    }
    return AlgorithmSpec("valid", "allgather", 2, buffers, ops, {})


class StaticValidationTest(unittest.TestCase):
    def test_valid_dag_passes_and_topological_order_respects_deps(self):
        report = validate_static(valid_algorithm(), topology())
        self.assertTrue(report.ok, report.issues)
        order = [op.id for op in topological_ops(valid_algorithm())]
        self.assertEqual(order, ["copy_r0", "send_r0"])

    def test_direct_user_to_user_send_fails(self):
        algorithm = valid_algorithm()
        buffers = dict(algorithm.buffers)
        buffers["r1_out"] = BufferSpec("r1_out", 1, "user_output", ())
        ops = {
            "bad_send": OpSpec("bad_send", "send", 0, 1024, "r0_in", "r1_out", 0, 1, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad", "allgather", 2, buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("comm_buffer_endpoint_required", {issue.code for issue in report.issues})

    def test_transfer_from_user_input_to_comm_buffer_fails(self):
        algorithm = valid_algorithm()
        ops = {
            "bad_send": OpSpec("bad_send", "send", 0, 1024, "r0_in", "r1_comm", 0, 1, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad", "allgather", 2, algorithm.buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("comm_buffer_endpoint_required", {issue.code for issue in report.issues})

    def test_transfer_from_comm_buffer_to_user_output_fails(self):
        algorithm = valid_algorithm()
        buffers = dict(algorithm.buffers)
        buffers["r1_out"] = BufferSpec("r1_out", 1, "user_output", ())
        ops = {
            "bad_recv": OpSpec("bad_recv", "recv", 1, 1024, "r0_comm", "r1_out", 0, 1, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad", "allgather", 2, buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("comm_buffer_endpoint_required", {issue.code for issue in report.issues})

    def test_datacopy_without_comm_endpoint_fails(self):
        algorithm = valid_algorithm()
        buffers = dict(algorithm.buffers)
        buffers["r0_tmp"] = BufferSpec("r0_tmp", 0, "user_output", ())
        ops = {
            "bad_copy": OpSpec("bad_copy", "copy", 0, 1024, "r0_in", "r0_tmp", mode="datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad_copy", "allgather", 2, buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("comm_buffer_endpoint_required", {issue.code for issue in report.issues})

    def test_transfer_endpoint_rank_out_of_range_fails(self):
        algorithm = valid_algorithm()
        ops = {
            "bad_send": OpSpec("bad_send", "send", 0, 1024, "r0_comm", "r1_comm", 0, 2, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad_rank", "allgather", 2, algorithm.buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("invalid_transfer_rank", {issue.code for issue in report.issues})

    def test_missing_transfer_endpoint_rank_fails(self):
        algorithm = valid_algorithm()
        ops = {
            "bad_send": OpSpec("bad_send", "send", 0, 1024, "r0_comm", "r1_comm", 0, None, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad_rank", "allgather", 2, algorithm.buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("missing_transfer_rank", {issue.code for issue in report.issues})

    def test_non_integer_transfer_endpoint_rank_fails_validation(self):
        algorithm = valid_algorithm()
        ops = {
            "bad_send": OpSpec("bad_send", "send", 0, 1024, "r0_comm", "r1_comm", "zero", 1, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad_rank", "allgather", 2, algorithm.buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("invalid_transfer_rank", {issue.code for issue in report.issues})

    def test_transfer_endpoint_rank_must_match_buffer_rank(self):
        algorithm = valid_algorithm()
        ops = {
            "bad_send": OpSpec("bad_send", "send", 0, 1024, "r0_comm", "r1_comm", 1, 1, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad_rank", "allgather", 2, algorithm.buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("transfer_rank_buffer_mismatch", {issue.code for issue in report.issues})

    def test_missing_dep_and_cycle_fail(self):
        algorithm = valid_algorithm()
        ops = {
            "a": OpSpec("a", "wait", 0, deps=("b",)),
            "b": OpSpec("b", "wait", 0, deps=("a",)),
            "c": OpSpec("c", "wait", 0, deps=("missing",)),
        }
        report = validate_static(AlgorithmSpec("cycle", "allgather", 2, algorithm.buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("missing_dependency", {issue.code for issue in report.issues})
        self.assertIn("cycle_detected", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
