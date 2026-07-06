#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_GUARD = REPO_ROOT / "tests" / "comm" / "unit" / "test_tilexr_source_guards.cpp"
BUILD_ARTIFACT_GUARD = REPO_ROOT / "tests" / "ccu" / "check_tile_comm_no_hcomm_deps.sh"
DIRECT_SMOKE_PROBE = REPO_ROOT / "tests" / "ccu" / "ccu_tilexr_direct_smoke_probe.cpp"
DIRECT_SMOKE_RUNNER = REPO_ROOT / "tests" / "ccu" / "run_tilexr_ccu_direct_smoke.sh"
ROOT_CMAKE = REPO_ROOT / "CMakeLists.txt"
CCU_CMAKE = REPO_ROOT / "tests" / "ccu" / "CMakeLists.txt"


class TileXRCcuSourceGuardCoverageTest(unittest.TestCase):
    def test_source_guard_covers_hccl_include_and_path_variants(self):
        source = SOURCE_GUARD.read_text(encoding="utf-8").replace('\\"', '"')

        for needle in [
            "TestRootCMakeHcclIncludesAreNotTileCommSurface",
            "#include <hccl/",
            "#include \"hccl/",
            "#include <hccl.h>",
            "#include \"hccl.h\"",
            "pkg_inc/hccl",
            "include/hccl",
            "${ARCH}-linux/include/hccl",
            "${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl",
        ]:
            with self.subTest(needle=needle):
                self.assertIn(needle, source)

    def test_root_cmake_does_not_expose_hccl_include_globally(self):
        cmake = ROOT_CMAKE.read_text(encoding="utf-8")

        self.assertNotIn("${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl/", cmake)
        self.assertNotIn("${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl", cmake)

    def test_ccu_cmake_does_not_define_private_probe_targets(self):
        cmake = CCU_CMAKE.read_text(encoding="utf-8")

        for needle in [
            "TILEXR_BUILD_PRIVATE_CCU_PROBES",
            "ccu_context_probe",
            "ccu_barrier_kernel_probe",
            "ccu_taskinfo_probe",
            "include/hccl",
            "pkg_inc/hcomm",
            "hcomm",
            "hccl",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, cmake)

    def test_build_artifact_guard_scans_private_hcomm_hccl_and_ccu_surfaces(self):
        script = BUILD_ARTIFACT_GUARD.read_text(encoding="utf-8")

        for needle in [
            "readelf -d",
            "ldd",
            "nm -D",
            "strings -a",
            "libhcomm\\.so",
            "libhccl_v2\\.so",
            "libhccl_fwk\\.so",
            "libmc2_client\\.so",
            "HcclGetCcuTaskInfo",
            "HcclChannelAcquire",
            "HcommChannelNotify",
            "RT_RES_TYPE_CCU_XN",
        ]:
            with self.subTest(needle=needle):
                self.assertIn(needle, script)

    def test_direct_smoke_probe_keeps_private_producer_dependencies_out(self):
        source = DIRECT_SMOKE_PROBE.read_text(encoding="utf-8")

        for needle in [
            "#include <hcomm/",
            "#include \"hcomm/",
            "#include <hccl/",
            "#include \"hccl/",
            "#include <hccl.h>",
            "#include \"hccl.h\"",
            "libhcomm",
            "libhccl_v2",
            "libhccl_fwk",
            "libmc2_client",
            "HcclGetCcuTaskInfo",
            "HcomGetCcuTaskInfo",
            "CcuResBatchAllocator",
            "CcuResRepository",
            "runtime/kernel.h",
            "rtCCULaunch",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, source)

    def test_direct_smoke_runner_keeps_private_link_dependencies_out(self):
        source = DIRECT_SMOKE_RUNNER.read_text(encoding="utf-8")

        for needle in [
            "-lhcomm",
            "-lhccl",
            "-lhccl_v2",
            "-lhccl_fwk",
            "-lmc2_client",
            "libhcomm",
            "libhccl_v2",
            "libhccl_fwk",
            "libmc2_client",
            "pkg_inc/hcomm",
            "pkg_inc/hccl",
            "include/hccl",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, source)


if __name__ == "__main__":
    unittest.main()
