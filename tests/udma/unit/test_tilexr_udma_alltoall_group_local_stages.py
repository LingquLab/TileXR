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
        self.assertIn("AllToAllGroupStageRunsSendDevice(routeStage)", kernel)
        self.assertIn("AllToAllGroupStageRunsCopyDevice(routeStage)", kernel)
        self.assertIn("AllToAllGroupStageWaitsForSignalDevice(routeStage)", kernel)
        self.assertIn(
            '"local-send", "local-copy", "primary", "secondary"', host
        )


if __name__ == "__main__":
    unittest.main()
