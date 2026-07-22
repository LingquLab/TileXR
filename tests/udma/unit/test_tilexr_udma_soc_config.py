import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class UDMASocConfigTest(unittest.TestCase):
    def test_ascend950_variants_use_dav3510(self):
        cmake = (ROOT / "tests" / "udma" / "CMakeLists.txt").read_text(
            encoding="utf-8")

        self.assertIn('MATCHES "^Ascend950(PR|DT)?$"', cmake)
        self.assertIn('set(TILEXR_UDMA_NPU_ARCH "dav-3510")', cmake)
        self.assertIn('set(TILEXR_UDMA_CATLASS_ARCH "3510")', cmake)

    def test_unknown_soc_is_rejected(self):
        cmake = (ROOT / "tests" / "udma" / "CMakeLists.txt").read_text(
            encoding="utf-8")

        self.assertIn('Unsupported TILEXR_UDMA_DEMO_SOC_TYPE', cmake)


if __name__ == "__main__":
    unittest.main()
