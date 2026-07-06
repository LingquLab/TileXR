#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_HEADER = REPO_ROOT / "src" / "include" / "tilexr_api.h"
COMM_WRAP = REPO_ROOT / "src" / "comm" / "comm_wrap.cpp"
SMOKE_PROBE = REPO_ROOT / "tests" / "ccu" / "ccu_tilexr_direct_smoke_probe.cpp"


class TileXRCcuPublicCommApiTest(unittest.TestCase):
    def test_public_header_declares_opaque_direct_ccu_prepare_surface(self):
        header = PUBLIC_HEADER.read_text(encoding="utf-8")

        self.assertIn("typedef void *TileXRDirectCcuPreparedTasksPtr;", header)
        self.assertIn("#define TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES 2048", header)
        self.assertIn("int TileXRCommInitRankDirectCcuWithDomain(", header)
        self.assertIn("TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_WINDOW_MISSION", header)
        self.assertIn("TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_WINDOW_FULL_REPOSITORY", header)
        self.assertIn("TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_DATA_LEN_INSTRUCTION_BYTES", header)
        self.assertIn("TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_DATA_LEN_DESCRIPTOR_BYTES", header)
        self.assertIn("TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_ACL", header)
        self.assertIn("TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_ACL_MODULE3", header)
        self.assertIn("TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_RT_HBM", header)
        self.assertIn("TILEXR_DIRECT_CCU_INSTALL_ORDER_REPOSITORY_FIRST", header)
        self.assertIn("TILEXR_DIRECT_CCU_INSTALL_ORDER_LOWER_LAYER_FIRST", header)
        self.assertIn("#define TILEXR_DIRECT_CCU_SQE_ARGS_LEN 13U", header)
        self.assertIn("typedef struct TileXRDirectCcuPrepareOptions", header)
        self.assertIn("uint32_t sqeArgCount;", header)
        self.assertIn("uint16_t missionInstructionStartId;", header)
        self.assertIn("uint16_t gsaStartId;", header)
        self.assertIn("uint32_t repositoryInstallWindow;", header)
        self.assertIn("uint32_t repositoryInstallDataLenMode;", header)
        self.assertIn("uint32_t repositoryMemoryAllocMode;", header)
        self.assertIn("uint32_t installOrder;", header)
        self.assertIn("typedef struct TileXRDirectCcuPrepareReport", header)
        self.assertIn("typedef struct TileXRDirectCcuSubmitReport", header)
        self.assertIn("typedef struct TileXRDirectCcuMemoryCopyPrepareOptions", header)
        self.assertIn("typedef struct TileXRDirectCcuTaskInfo", header)
        self.assertIn("int TileXRCommPrepareDirectCcu(", header)
        self.assertIn("int TileXRCommPrepareDirectCcuMemoryCopy(", header)
        self.assertIn("int TileXRDirectCcuGetPreparedTask(", header)
        self.assertIn("int TileXRDirectCcuSubmitPrepared(", header)
        self.assertIn("int TileXRDirectCcuDestroyPrepared(", header)

    def test_public_header_does_not_expose_private_ccu_cpp_types(self):
        header = PUBLIC_HEADER.read_text(encoding="utf-8")

        for needle in [
            "tilexr_ccu_direct_orchestrator.h",
            "tilexr_ccu_runtime.h",
            "TileXRCcuDirectInstallOptions",
            "TileXRCcuDirectInstallAttempt",
            "TileXRCcuTask",
            "std::vector",
            "std::string message",
            "runtime/kernel.h",
            "rtCCULaunch",
            "rtCcuTaskInfo_t",
            "hcomm",
            "hccl",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, header)

    def test_wrapper_bridges_public_api_to_private_prepare_without_hcomm(self):
        wrapper = COMM_WRAP.read_text(encoding="utf-8")

        self.assertIn("TileXRCommInitRankDirectCcuWithDomain", wrapper)
        self.assertIn("InitDirectCcuOnly", wrapper)
        self.assertIn("TileXRCommPrepareDirectCcu", wrapper)
        self.assertIn("TileXRCommPrepareDirectCcuMemoryCopy", wrapper)
        self.assertIn("PrepareDirectCcuInstallAttempt", wrapper)
        self.assertIn("PrepareDirectCcuMemoryCopyInstallAttempt", wrapper)
        self.assertIn("installDiagnosticReady", wrapper)
        self.assertIn("TileXRDirectCcuSubmitPrepared", wrapper)
        self.assertIn("TileXRCcuSubmitPreparedTasks", wrapper)
        self.assertIn("TileXRDirectCcuDestroyPrepared", wrapper)
        self.assertIn("TILEXR_CCU_DIRECT_BARRIER_MODE", wrapper)
        self.assertIn("TileXRCcuBarrierMode::SyncCke", wrapper)
        self.assertIn("TileXRCcuBarrierMode::SyncCkeSetWait", wrapper)
        self.assertIn("TileXRCcuBarrierMode::SyncCkePostOnly", wrapper)
        self.assertIn("TileXRCcuBarrierMode::LocalCke", wrapper)
        self.assertIn("TileXRCcuBarrierMode::LocalCkePostOnly", wrapper)
        self.assertIn("TileXRCcuBarrierMode::SyncXnPostOnly", wrapper)
        self.assertIn("TileXRCcuBarrierMode::SyncXnLoadPostOnly", wrapper)
        self.assertIn("repositoryInstallOptions.window", wrapper)
        self.assertIn("options.sqeArgCount = publicOptions.sqeArgCount", wrapper)
        self.assertIn("options.missionInstructionStartId = publicOptions.missionInstructionStartId", wrapper)
        self.assertIn("options.gsaStartId = publicOptions.gsaStartId", wrapper)
        self.assertIn("TileXRCcuRepositoryInstallWindow::FullRepository", wrapper)
        self.assertIn("TileXRCcuRepositoryInstallWindow::Mission", wrapper)
        self.assertIn("repositoryInstallOptions.dataLenMode", wrapper)
        self.assertIn("TileXRCcuRepositoryInstallDataLenMode::DescriptorBytes", wrapper)
        self.assertIn("TileXRCcuRepositoryInstallDataLenMode::InstructionBytes", wrapper)
        self.assertIn("RepositoryMemoryAllocModeFromPublic", wrapper)
        self.assertIn("options.repositoryMemoryAllocMode", wrapper)
        self.assertIn("publicOptions.repositoryMemoryAllocMode", wrapper)
        self.assertIn("TileXRCcuRepositoryMemoryAllocMode::AclModule3", wrapper)
        self.assertIn("TileXRCcuRepositoryMemoryAllocMode::RtHbm", wrapper)
        self.assertIn("TileXRCcuRepositoryMemoryAllocMode::Acl", wrapper)
        self.assertIn("TileXRCcuInstallOrder::InstallLowerLayerFirst", wrapper)
        self.assertIn("TileXRCcuInstallOrder::RepositoryFirst", wrapper)
        self.assertNotIn("options.sqeArgCount = TILEXR_DIRECT_CCU_SQE_ARGS_LEN", wrapper)
        for needle in [
            "libhcomm",
            "libhccl_v2",
            "HcclGetCcuTaskInfo",
            "HcomGetCcuTaskInfo",
            "HcclChannelAcquire",
            "rtCCULaunch",
            "runtime/kernel.h",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, wrapper)

    def test_direct_ccu_only_init_skips_peer_ipc_but_keeps_socket_exchange(self):
        comm_header = (REPO_ROOT / "src" / "comm" / "tilexr_comm.h").read_text(encoding="utf-8")
        comm_source = (REPO_ROOT / "src" / "comm" / "tilexr_comm.cpp").read_text(encoding="utf-8")

        self.assertIn("InitDirectCcuOnly", comm_header)
        init_body = comm_source[
            comm_source.index("int TileXRComm::InitDirectCcuOnly"):
            comm_source.index("int TileXRComm::InitThread")
        ]
        self.assertIn("TileXRSockExchange", init_body)
        self.assertIn("GetDev()", init_body)
        self.assertIn("InitCommon()", init_body)
        self.assertIn("InitDirectCcuRuntime()", init_body)
        self.assertIn("inited_ = true", init_body)
        self.assertNotIn("InitCommMem()", init_body)
        self.assertNotIn("OpenIpcMem", init_body)
        self.assertNotIn("InitUDMA()", init_body)
        self.assertNotIn("SyncCommArgs()", init_body)

    def test_prepared_handle_releases_repository_resources_on_destroy(self):
        wrapper = COMM_WRAP.read_text(encoding="utf-8")
        direct_header = (REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_direct_orchestrator.h").read_text(
            encoding="utf-8")
        direct_source = (REPO_ROOT / "src" / "comm" / "ccu" / "tilexr_ccu_direct_orchestrator.cpp").read_text(
            encoding="utf-8")

        self.assertIn("TileXRCcuReleaseDirectInstallAttemptResources", direct_header)
        self.assertIn("TileXRCcuReleaseRepositoryInstallReceipt", direct_source)
        self.assertIn("~TileXRDirectCcuPreparedTasks()", wrapper)
        self.assertIn("TileXRCcuReleaseDirectInstallAttemptResources(attempt)", wrapper)
        self.assertIn("delete PreparedHandle(prepared)", wrapper)

    def test_smoke_probe_uses_public_direct_ccu_api_only(self):
        source = SMOKE_PROBE.read_text(encoding="utf-8")

        self.assertIn("TileXRCommPrepareDirectCcu", source)
        self.assertIn("TileXRDirectCcuGetPreparedTask", source)
        self.assertIn("TileXRDirectCcuSubmitPrepared", source)
        self.assertIn("TileXRDirectCcuDestroyPrepared", source)
        self.assertNotIn("#include \"ccu/tilexr_ccu_direct_orchestrator.h\"", source)
        self.assertNotIn("#include \"tilexr_comm.h\"", source)
        self.assertNotIn("static_cast<TileXR::TileXRComm*>", source)
        self.assertNotIn("TileXRCcuDirectInstallAttempt", source)
        self.assertNotIn("TileXRCcuSubmitPreparedTasks", source)


if __name__ == "__main__":
    unittest.main()
