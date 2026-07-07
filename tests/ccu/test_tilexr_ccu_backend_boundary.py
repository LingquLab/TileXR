#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMM_HEADER = REPO_ROOT / "src" / "comm" / "tilexr_comm.h"
BACKEND_HEADER = REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_backend.h"
BACKEND_SOURCE = REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_backend.cpp"


class TileXRCcuBackendBoundaryTest(unittest.TestCase):
    def test_backend_files_exist(self):
        self.assertTrue(BACKEND_HEADER.exists())
        self.assertTrue(BACKEND_SOURCE.exists())

    def test_tilexr_comm_header_owns_only_opaque_backend(self):
        header = COMM_HEADER.read_text(encoding="utf-8")
        self.assertIn("class TileXRCcuBackend;", header)
        self.assertIn("std::unique_ptr<TileXRCcuBackend> ccuBackend_", header)
        for needle in [
            "tilexr_ccu_direct_orchestrator.h",
            "tilexr_ccu_direct_runtime.h",
            "tilexr_ccu_lower_layer_plan_builder.h",
            "TileXRCcuDirectRuntime",
            "directCcuBasicInfo_",
            "directCcuLowerLayerPlan_",
            "directCcuVerifiedEndpointRoutes_",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, header)

    def test_backend_header_exposes_facade_not_public_c_api(self):
        header = BACKEND_HEADER.read_text(encoding="utf-8")
        self.assertIn("class TileXRCcuBackend", header)
        self.assertIn("struct TileXRCcuBackendOptions", header)
        self.assertIn("TileXRSockExchange *exchange", header)
        self.assertIn("PrepareCollective", header)
        self.assertIn("SubmitCollective", header)
        for needle in [
            "TileXRDirectCcuPreparedTasksPtr",
            "TileXRCommPrepareDirectCcu",
            "TileXRDirectCcuSubmitPrepared",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, header)

    def test_backend_source_owns_restored_direct_ccu_runtime_glue(self):
        source = BACKEND_SOURCE.read_text(encoding="utf-8")
        for needle in [
            "#include \"ccu/tilexr_ccu_direct_runtime.h\"",
            "#include \"ccu/tilexr_ccu_repository.h\"",
            "TileXRCcuDirectRuntime",
            "PrepareDirectCcuInstallAttempt",
            "PrepareDirectCcuLowerLayerPlanCallback",
            "TileXRCcuRunDirectInstallAttempt",
            "TileXRCcuMakeRepositoryDeviceMemoryOps",
            "DirectCcuThreadAllGather",
            "g_directCcuAllGatherStates",
        ]:
            with self.subTest(needle=needle):
                self.assertIn(needle, source)
        for fake_ready in [
            "options_ = options;\n    initialized_ = true;\n    return TILEXR_SUCCESS;",
            "plan->ready = true;\n    return TILEXR_SUCCESS;",
            "return plan.ready ? TILEXR_SUCCESS",
        ]:
            with self.subTest(fake_ready=fake_ready):
                self.assertNotIn(fake_ready, source)


if __name__ == "__main__":
    unittest.main()
