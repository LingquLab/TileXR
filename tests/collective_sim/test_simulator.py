import unittest

from tests.collective_sim.test_semantics import two_rank_allgather
from tilexr_collective_sim.model import (
    AlgorithmSpec,
    BandwidthCurve,
    BufferSpec,
    CalibrationSpec,
    CurvePoint,
    OpSpec,
    TopologySpec,
)
from tilexr_collective_sim.simulator import simulate_algorithm


def calibration():
    return CalibrationSpec(curves={
        "p2p_50g": BandwidthCurve("p2p_50g", (CurvePoint(1024, 5), CurvePoint(67108864, 50)), 2.0),
        "uplink_300g": BandwidthCurve("uplink_300g", (CurvePoint(1024, 10), CurvePoint(67108864, 300)), 3.0),
        "sdma_800g": BandwidthCurve("sdma_800g", (CurvePoint(1024, 40), CurvePoint(67108864, 800)), 1.0),
    })


def topology(rank_count=2):
    return TopologySpec("one_d_clos", rank_count, 8, "p2p_50g", "uplink_300g", 64, 128, "2:1", "destination_pool_fair_share")


def same_resource_sends():
    buffers = {
        "r0_c0": BufferSpec("r0_c0", 0, "comm_buffer", ()),
        "r0_c1": BufferSpec("r0_c1", 0, "comm_buffer", ()),
        "r1_c0": BufferSpec("r1_c0", 1, "comm_buffer", ()),
        "r1_c1": BufferSpec("r1_c1", 1, "comm_buffer", ()),
    }
    ops = {
        "send_a": OpSpec("send_a", "send", 0, 1024, "r0_c0", "r1_c0", 0, 1, (), "datacopy"),
        "send_b": OpSpec("send_b", "send", 0, 1024, "r0_c1", "r1_c1", 0, 1, (), "datacopy"),
    }
    return AlgorithmSpec("same_resource_sends", "custom", 2, buffers, ops, {})


def cross_server_send(rank_count=128):
    buffers = {
        "r0_c": BufferSpec("r0_c", 0, "comm_buffer", ()),
        "r8_c": BufferSpec("r8_c", 8, "comm_buffer", ()),
    }
    ops = {
        "send_r0_to_r8": OpSpec("send_r0_to_r8", "send", 0, 1024, "r0_c", "r8_c", 0, 8, (), "datacopy"),
    }
    return AlgorithmSpec("cross_server_send", "custom", rank_count, buffers, ops, {})


class SimulatorTest(unittest.TestCase):
    def test_simulates_valid_allgather_and_records_copy_events(self):
        result = simulate_algorithm(two_rank_allgather(), topology(2), calibration(), 1024, validate=True)
        self.assertTrue(result.validation.ok, result.validation.issues)
        self.assertGreater(result.latency_us, 0.0)
        event_ids = {event.op_id for event in result.events}
        self.assertIn("copy_r0_in", event_ids)
        self.assertIn("send_r0_to_r1", event_ids)
        copy_event = next(event for event in result.events if event.op_id == "copy_r0_in")
        self.assertEqual(copy_event.resources, ("sdma:rank0",))
        self.assertEqual(copy_event.data_source, "estimate")

    def test_validation_gate_blocks_bad_algorithm(self):
        algorithm = two_rank_allgather()
        ops = dict(algorithm.ops)
        ops["bad"] = OpSpec("bad", "send", 0, 1024, "r0_in", "r1_out", 0, 1, (), "datacopy")
        bad = algorithm.__class__(algorithm.name, algorithm.collective, algorithm.rank_count, algorithm.buffers, ops, algorithm.metadata)
        result = simulate_algorithm(bad, topology(2), calibration(), 1024, validate=True)
        self.assertFalse(result.validation.ok)
        self.assertEqual(result.events, ())

    def test_same_ready_time_flows_share_resource(self):
        result = simulate_algorithm(same_resource_sends(), topology(2), calibration(), 1024, validate=True)
        self.assertTrue(result.validation.ok, result.validation.issues)
        send_events = sorted((event for event in result.events if event.op_type == "send"), key=lambda event: event.op_id)
        self.assertEqual(len(send_events), 2)
        self.assertEqual(send_events[0].resources, send_events[1].resources)
        self.assertEqual(send_events[0].start_us, send_events[1].start_us)
        self.assertEqual(send_events[0].end_us, send_events[1].end_us)
        self.assertTrue(all(event.effective_gbps <= 2.5 for event in send_events))

    def test_allgather_validation_uses_topology_rank_count(self):
        result = simulate_algorithm(two_rank_allgather(), topology(3), calibration(), 1024, validate=True)
        self.assertFalse(result.validation.ok)
        self.assertIn("rank_count_mismatch", {issue.code for issue in result.validation.issues})
        self.assertEqual(result.events, ())

    def test_oversubscribed_cross_server_send_reports_clos_bottleneck(self):
        result = simulate_algorithm(cross_server_send(128), topology(128), calibration(), 1024, validate=True)
        self.assertTrue(result.validation.ok, result.validation.issues)
        event = next(event for event in result.events if any(resource.startswith("clos:") for resource in event.resources))
        clos_resource = next(resource for resource in event.resources if resource.startswith("clos:"))
        self.assertEqual(event.bottleneck_resource, clos_resource)
        self.assertNotEqual(event.bottleneck_resource, event.resources[0])


if __name__ == "__main__":
    unittest.main()
