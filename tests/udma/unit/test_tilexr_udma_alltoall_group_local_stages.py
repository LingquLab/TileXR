import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]


class GroupedAllToAllLocalStageSourceTest(unittest.TestCase):
    def test_local_route_is_split_into_send_and_copy_stages(self):
        route = (ROOT / "tests/udma/demo/tilexr_udma_alltoall_group_route.h").read_text()
        kernel = (ROOT / "tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp").read_text()
        host = (ROOT / "tests/udma/demo/tilexr_udma_demo.cpp").read_text()

        self.assertIn("kLocalSend", route)
        self.assertIn("kLocalCopy", route)
        self.assertIn("kRemoteSend", route)
        self.assertIn("kAllSend", route)
        self.assertIn("kRemoteWait", route)
        self.assertIn("kRemoteCopy", route)
        self.assertIn("kNoCopy", route)
        self.assertIn("AllToAllGroupStageRunsSendDevice(routeStage)", kernel)
        self.assertIn("AllToAllGroupStageRunsReceiveDevice(routeStage)", kernel)
        self.assertIn("AllToAllGroupReceivePeerInRouteStageDevice", kernel)
        self.assertIn("AllToAllGroupStageRunsCopyDevice(routeStage)", kernel)
        self.assertIn("AllToAllGroupStageWaitsForSignalDevice(routeStage)", kernel)
        self.assertIn(
            '"local-send", "local-copy", "remote-send", "all-send", '
            '"remote-wait", "remote-copy", "no-copy", "primary", "secondary", '
            '"combined"',
            " ".join(host.split()),
        )
        self.assertIn("constexpr size_t kRouteStageCount = 10U", host)
        stage_batch = host[host.index("auto runStageBatch") : host.index(
            "for (size_t stageIndex", host.index("auto runStageBatch"))]
        self.assertNotIn("invocationId = 0U", stage_batch)
        self.assertNotIn("grouped stage iteration", stage_batch)
        self.assertIn("aclrtSynchronizeStream grouped stage warmup", stage_batch)
        self.assertIn("aclrtSynchronizeStream grouped stage measured", stage_batch)


if __name__ == "__main__":
    unittest.main()
