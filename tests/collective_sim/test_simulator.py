import unittest

from tests.collective_sim.test_semantics import two_rank_allgather
from tilexr_collective_sim.model import BandwidthCurve, CalibrationSpec, CurvePoint, OpSpec, TopologySpec
from tilexr_collective_sim.simulator import simulate_algorithm


def calibration():
    return CalibrationSpec(curves={
        "p2p_50g": BandwidthCurve("p2p_50g", (CurvePoint(1024, 5), CurvePoint(67108864, 50)), 2.0),
        "uplink_300g": BandwidthCurve("uplink_300g", (CurvePoint(1024, 10), CurvePoint(67108864, 300)), 3.0),
        "sdma_800g": BandwidthCurve("sdma_800g", (CurvePoint(1024, 40), CurvePoint(67108864, 800)), 1.0),
    })


def topology(rank_count=2):
    return TopologySpec("one_d_clos", rank_count, 8, "p2p_50g", "uplink_300g", 64, 128, "2:1", "destination_pool_fair_share")


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
        result = simulate_algorithm(two_rank_allgather(), topology(2), calibration(), 1024, validate=True)
        send_events = [event for event in result.events if event.op_type == "send"]
        self.assertEqual(len(send_events), 2)
        self.assertTrue(all(event.effective_gbps <= 5.0 for event in send_events))


if __name__ == "__main__":
    unittest.main()
