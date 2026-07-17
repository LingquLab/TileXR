import unittest

from tilexr_collective_sim.model import BandwidthCurve, CurvePoint, TopologySpec
from tilexr_collective_sim.topology import interpolate_curve, oversubscription_scale, resource_ids_for_transfer, transfer_duration_us


def curve():
    return BandwidthCurve("test", (CurvePoint(1024, 10.0), CurvePoint(4096, 40.0)), 2.0)


def topology(rank_count):
    return TopologySpec("one_d_clos", rank_count, 8, "p2p_50g", "uplink_300g", 64, 128, "2:1", "destination_pool_fair_share")


class TopologyTest(unittest.TestCase):
    def test_curve_interpolation_and_duration(self):
        self.assertEqual(interpolate_curve(curve(), 1024), 10.0)
        self.assertEqual(interpolate_curve(curve(), 4096), 40.0)
        self.assertAlmostEqual(interpolate_curve(curve(), 2560), 25.0)
        duration = transfer_duration_us(curve(), 1024, active_flows=1)
        self.assertGreater(duration, 2.0)

    def test_sdma_active_flow_scaling(self):
        duration_one = transfer_duration_us(curve(), 4096, active_flows=1)
        duration_two = transfer_duration_us(curve(), 4096, active_flows=2)
        self.assertGreater(duration_two, duration_one)

    def test_resource_selection_same_server_and_cross_server(self):
        self.assertEqual(resource_ids_for_transfer(topology(64), 0, 7), ("p2p:s0:0->7",))
        self.assertEqual(
            resource_ids_for_transfer(topology(64), 0, 8),
            ("uplink:src:0", "clos:nonblocking", "uplink:dst:8"),
        )
        self.assertIn("clos:dst_server:1:oversub_2to1", resource_ids_for_transfer(topology(128), 0, 8))

    def test_cross_server_below_oversubscription_threshold_is_not_oversubscribed(self):
        resources = resource_ids_for_transfer(topology(96), 0, 8)
        self.assertEqual(resources, ("uplink:src:0", "clos:limited", "uplink:dst:8"))
        self.assertNotIn("clos:nonblocking", resources)
        self.assertNotIn("clos:dst_server:1:oversub_2to1", resources)

    def test_oversubscription_scale(self):
        self.assertEqual(oversubscription_scale(topology(64)), 1.0)
        self.assertEqual(oversubscription_scale(topology(128)), 0.5)


if __name__ == "__main__":
    unittest.main()
