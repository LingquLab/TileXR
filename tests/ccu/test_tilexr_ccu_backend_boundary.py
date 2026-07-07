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
        self.assertIn("PrepareCollective", header)
        self.assertIn("SubmitCollective", header)
        for needle in [
            "TileXRDirectCcuPreparedTasksPtr",
            "TileXRCommPrepareDirectCcu",
            "TileXRDirectCcuSubmitPrepared",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, header)


if __name__ == "__main__":
    unittest.main()
