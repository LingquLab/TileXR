#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "demo/tilexr_udma_alltoall_layout.h"

#ifndef TILEXR_SOURCE_ROOT
#define TILEXR_SOURCE_ROOT "."
#endif

namespace {

int g_failures = 0;

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) == std::string::npos) { \
            std::cerr << "CHECK_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_NOT_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) != std::string::npos) { \
            std::cerr << "CHECK_NOT_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
            ++g_failures; \
        } \
    } while (0)

std::string ReadFile(const std::string& path)
{
    std::ifstream in(path.c_str());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void TestAllToAllInputPattern()
{
    constexpr int rank = 2;
    constexpr int rankSize = 4;
    constexpr int32_t elementsPerPeer = 3;
    std::vector<int32_t> input(static_cast<size_t>(rankSize) * elementsPerPeer, -1);

    TileXR::Demo::FillAllToAllInput(input, rank, rankSize, elementsPerPeer);

    for (int dstRank = 0; dstRank < rankSize; ++dstRank) {
        int32_t expected = TileXR::Demo::AllToAllValue(rank, dstRank);
        for (int32_t elem = 0; elem < elementsPerPeer; ++elem) {
            CHECK_EQ(input[static_cast<size_t>(dstRank) * elementsPerPeer + elem], expected);
        }
    }
}

void TestAllToAllOutputValidation()
{
    constexpr int rank = 1;
    constexpr int rankSize = 3;
    constexpr int32_t elementsPerPeer = 2;
    std::vector<int32_t> output(static_cast<size_t>(rankSize) * elementsPerPeer, -1);

    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        std::fill(output.begin() + static_cast<size_t>(srcRank) * elementsPerPeer,
                  output.begin() + static_cast<size_t>(srcRank + 1) * elementsPerPeer,
                  TileXR::Demo::AllToAllValue(srcRank, rank));
    }

    CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(output, rank, rankSize, elementsPerPeer), true);
    output[static_cast<size_t>(2) * elementsPerPeer + 1] = 123;
    CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(output, rank, rankSize, elementsPerPeer), false);
}

void TestBuildAllToAllOutput()
{
    constexpr int rankSize = 3;
    constexpr int32_t elementsPerPeer = 2;
    std::vector<int32_t> allInputs(static_cast<size_t>(rankSize) * rankSize * elementsPerPeer, -1);

    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        std::vector<int32_t> oneInput(static_cast<size_t>(rankSize) * elementsPerPeer, -1);
        TileXR::Demo::FillAllToAllInput(oneInput, srcRank, rankSize, elementsPerPeer);
        std::copy(oneInput.begin(), oneInput.end(),
                  allInputs.begin() + static_cast<size_t>(srcRank) * rankSize * elementsPerPeer);
    }

    std::vector<int32_t> output(static_cast<size_t>(rankSize) * elementsPerPeer, -1);
    TileXR::Demo::BuildAllToAllOutputFromInputs(allInputs, 2, rankSize, elementsPerPeer, output);

    CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(output, 2, rankSize, elementsPerPeer), true);
}

void TestAllToAllMaxRank256With64MiBPerRank()
{
    constexpr int rankSize = 256;
    constexpr size_t perRankBytes = 64ULL * 1024ULL * 1024ULL;
    constexpr int32_t elementsPerPeer =
        static_cast<int32_t>(perRankBytes / (sizeof(int32_t) * rankSize));
    CHECK_EQ(elementsPerPeer, 65536);

    std::vector<int32_t> buffer(static_cast<size_t>(rankSize) * elementsPerPeer, -1);
    const int sampleRanks[] = {0, 1, 127, 255};
    for (int rank : sampleRanks) {
        TileXR::Demo::FillAllToAllInput(buffer, rank, rankSize, elementsPerPeer);
        CHECK_EQ(buffer[0], TileXR::Demo::AllToAllValue(rank, 0));
        CHECK_EQ(buffer[static_cast<size_t>(rankSize - 1) * elementsPerPeer],
                 TileXR::Demo::AllToAllValue(rank, rankSize - 1));
        CHECK_EQ(buffer[static_cast<size_t>(rankSize) * elementsPerPeer - 1],
                 TileXR::Demo::AllToAllValue(rank, rankSize - 1));

        for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
            std::fill(buffer.begin() + static_cast<size_t>(srcRank) * elementsPerPeer,
                      buffer.begin() + static_cast<size_t>(srcRank + 1) * elementsPerPeer,
                      TileXR::Demo::AllToAllValue(srcRank, rank));
        }
        CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(buffer, rank, rankSize, elementsPerPeer), true);
        buffer[static_cast<size_t>(rankSize) * elementsPerPeer - 1] = -1;
        CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(buffer, rank, rankSize, elementsPerPeer), false);
    }
}

void TestAllToAllBigDataPlan()
{
    constexpr int rankSize = 8;
    constexpr int32_t elementsPerPeer = 16 * 1024 * 1024; // 64 MiB per peer for int32_t.
    const auto plan = TileXR::Demo::PlanAllToAllBigDataUdma(rankSize, elementsPerPeer);

    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMaxRegisteredBytes, 64ULL * 1024ULL * 1024ULL);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataControlSlotBytes, 64ULL);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataCoresPerPeer, 5U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataLocalCopyShards, 2U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataPingPongSlots, 2U);
    CHECK_EQ(plan.registeredBytes <= TileXR::Demo::kAllToAllBigDataMaxRegisteredBytes, true);
    const size_t controlGroupBytes =
        static_cast<size_t>(TileXR::Demo::kAllToAllBigDataPingPongSlots) *
        static_cast<size_t>(rankSize) *
        static_cast<size_t>(TileXR::Demo::kAllToAllBigDataLocalCopyShards) *
        TileXR::Demo::kAllToAllBigDataControlSlotBytes;
    CHECK_EQ(plan.controlBytes, controlGroupBytes);
    CHECK_EQ(plan.signalBytes, 3ULL * controlGroupBytes);
    CHECK_EQ(plan.copyDoneOffset, plan.dataBytes);
    CHECK_EQ(plan.recvCopyDoneOffset, plan.copyDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.readySignalOffset, plan.recvCopyDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.ackSignalOffset, plan.readySignalOffset + controlGroupBytes);
    CHECK_EQ(plan.registeredBytes, plan.dataBytes + plan.controlBytes + plan.signalBytes);
    CHECK_EQ(plan.dataBytes,
             static_cast<size_t>(rankSize - 1) * TileXR::Demo::kAllToAllBigDataPingPongSlots * 2ULL *
             plan.chunkBytesPerPeer);
    CHECK_EQ(plan.chunkElements > 0, true);
    CHECK_EQ(plan.chunkBytesPerPeer, static_cast<size_t>(plan.chunkElements) * sizeof(int32_t));
    CHECK_EQ(plan.passCount > 1, true);
}

void TestAllToAllBigDataBlockDim()
{
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(0), 1U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(1), 5U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(8), 40U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(64), 320U);
}

void TestDemoDebugLayoutSource()
{
    const std::string demo =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    const std::string kernel =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo_kernel.cpp");

    CHECK_CONTAINS(demo, "kDebugUdmaStatusBase + TileXR::TILEXR_MAX_RANK_SIZE");
    CHECK_CONTAINS(demo, "kDebugIpcGather + 1");
    CHECK_CONTAINS(demo, "kDebugReadySeenBase + TileXR::TILEXR_MAX_RANK_SIZE");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + TileXR::TILEXR_MAX_RANK_SIZE");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer");
}

void TestAllToAllDataAsFlagSource()
{
    const std::string demo =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    const std::string kernel =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo_kernel.cpp");

    CHECK_CONTAINS(demo, "useAllToAllDataAsFlagIpc");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_USE_UDMA");
    CHECK_CONTAINS(demo, "skip TileXRUDMARegister for alltoall data-as-flag IPC path");
    CHECK_CONTAINS(demo, "forceAllToAllIpcFallback");
    CHECK_CONTAINS(demo, "strictAllToAllUdma");
    CHECK_CONTAINS(demo, "ERROR: strict alltoall UDMA registration failed");
    CHECK_CONTAINS(demo, "ERROR: strict alltoall UDMA CQ incomplete");
    CHECK_CONTAINS(demo, "TileXRUDMARegister failed; use alltoall data-as-flag IPC fallback");
    CHECK_CONTAINS(demo, "allToAllIpcFallbackLabel");
    CHECK_CONTAINS(kernel, "#include \"tilexr_data_as_flag.h\"");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET");
    CHECK_CONTAINS(kernel, "DataAsFlagBlockCountForPayloadBytes");
    CHECK_CONTAINS(kernel, "DataAsFlagInit");
    CHECK_CONTAINS(kernel, "DataAsFlagSend");
    CHECK_CONTAINS(kernel, "DataAsFlagCheckAndRecv");
}

void TestAllToAllChunkedUdmaSource()
{
    const std::string demo =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    const std::string layout =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_alltoall_layout.h");
    const std::string kernel =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo_kernel.cpp");

    CHECK_CONTAINS(layout, "struct AllToAllChunkPlan");
    CHECK_CONTAINS(layout, "PlanAllToAllUdmaChunks");
    CHECK_CONTAINS(layout, "kAllToAllUdmaMaxRegisteredBytes");
    CHECK_CONTAINS(demo, "PlanAllToAllUdmaChunks");
    CHECK_CONTAINS(demo, "alltoall UDMA chunk plan");
    CHECK_CONTAINS(demo, "skip IPC staging capacity guard for strict UDMA alltoall");
    CHECK_CONTAINS(demo, "AllToAllPlainIpcStagingBytes");
    CHECK_CONTAINS(demo, "alltoall plain IPC staging bytes=");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_REPEAT");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_SYNC_AT_END");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_PLAIN_IPC");
    CHECK_CONTAINS(demo, "skip TileXRUDMARegister for alltoall plain IPC path");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_FUSED_IPC");
    CHECK_CONTAINS(demo, "skip TileXRUDMARegister for alltoall fused IPC path");
    CHECK_CONTAINS(demo, "alltoall fused IPC: single kernel send+flag+recv");
    CHECK_CONTAINS(demo, "alltoall use ");
    CHECK_CONTAINS(demo, "plain IPC fallback");
    CHECK_CONTAINS(demo, "launch all-to-all kernel repeat=");
    CHECK_CONTAINS(demo, "for (uint32_t pass = 0; pass < chunkPlan.passCount; ++pass)");
    CHECK_CONTAINS(demo, "registered output chunk");
    CHECK_CONTAINS(kernel, "tilexr_all_to_all_plain_ipc_scatter_kernel");
    CHECK_CONTAINS(kernel, "tilexr_all_to_all_plain_ipc_gather_kernel");
    CHECK_CONTAINS(kernel, "tilexr_all_to_all_fused_ipc_kernel");
    CHECK_CONTAINS(kernel, "launch_tilexr_all_to_all_fused_ipc");
    CHECK_CONTAINS(kernel, "inputElementOffset");
    CHECK_CONTAINS(kernel, "chunkElements");
}

void TestAllToAllBigDataSource()
{
    const std::string demo =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    const std::string kernel =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo_kernel.cpp");
    const std::string udma =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/include/tilexr_udma.h");

    CHECK_CONTAINS(demo, "testType == 7");
    CHECK_CONTAINS(demo, "PlanAllToAllBigDataUdma");
    CHECK_CONTAINS(demo, "bigdata alltoall registered dataBytes=");
    CHECK_CONTAINS(demo, "launch_tilexr_udma_all_to_all_bigdata");
    CHECK_CONTAINS(demo, "for (int iter = 0; iter < allToAllRepeat; ++iter)");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_BIGDATA_PROFILE_STAGE");
    CHECK_CONTAINS(demo, "const uint64_t kernelLoopBase = static_cast<uint64_t>(iter)");
    CHECK_CONTAINS(demo, "bigDataPlan.passCount, 1, kernelLoopBase");
    CHECK_CONTAINS(demo, "bigDataPlan.copyDoneOffset");
    CHECK_CONTAINS(demo, "bigDataPlan.recvCopyDoneOffset");
    CHECK_CONTAINS(demo, "static_cast<uint8_t*>(registeredMemory) + bigDataPlan.copyDoneOffset");
    CHECK_CONTAINS(demo, "bigDataPlan.controlBytes + bigDataPlan.signalBytes");
    CHECK_CONTAINS(demo, "static_cast<uint32_t>(bigDataProfileStage)");
    CHECK_CONTAINS(demo, "alltoall udma-bigdata");
    CHECK_CONTAINS(demo, "ERROR: bigdata alltoall UDMA registration failed");
    CHECK_CONTAINS(kernel, "tilexr_udma_all_to_all_bigdata_kernel");
    CHECK_CONTAINS(kernel, "launch_tilexr_udma_all_to_all_bigdata");
    CHECK_CONTAINS(kernel, "uint64_t kernelLoopBase");
    CHECK_CONTAINS(kernel, "uint32_t profileStage");
    CHECK_CONTAINS(kernel, "if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_PREPARE)");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_PROFILE_STAGE_SEND_COPY");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_PROFILE_STAGE_DATA_PUT");
    CHECK_CONTAINS(kernel, "uint64_t copyDoneOffset");
    CHECK_CONTAINS(kernel, "BigDataGlobalPassIndex");
    CHECK_CONTAINS(kernel, "const uint64_t globalPass = BigDataGlobalPassIndex");
    CHECK_CONTAINS(kernel, "BigDataPassToken(globalPass)");
    CHECK_CONTAINS(kernel, "BigDataPingPongSlot(globalPass)");
    CHECK_CONTAINS(kernel, "BigDataKernelExitBarrier");
    CHECK_CONTAINS(kernel, "copyDoneOffset");
    CHECK_CONTAINS(kernel, "readySignalOffset");
    CHECK_CONTAINS(kernel, "ackSignalOffset");
    CHECK_CONTAINS(kernel, "BigDataWaitTokenMte");
    CHECK_CONTAINS(kernel, "BigDataLoadTokenMte");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_SIGNAL_MAX_POLLS");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER = 5U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_LOCAL_COPY_SHARDS = 2U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS = 2U");
    CHECK_CONTAINS(kernel, "BigDataCopyPeerWorker");
    CHECK_CONTAINS(kernel, "BigDataSendPeerWorker");
    CHECK_CONTAINS(kernel, "BigDataRecvPeerWorker");
    CHECK_CONTAINS(kernel, "BigDataCopyShardRange");
    CHECK_CONTAINS(kernel, "copyShard");
    CHECK_CONTAINS(kernel, "recvShard");
    CHECK_CONTAINS(kernel, "recvCopyDoneOffset");
    CHECK_CONTAINS(kernel, "const int32_t peer = blockIdx / static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER)");
    CHECK_CONTAINS(kernel, "const int32_t role = blockIdx % static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER)");
    CHECK_CONTAINS(kernel, "BigDataPingPongSlot");
    CHECK_CONTAINS(kernel, "BigDataNetworkPeerIndex");
    CHECK_CONTAINS(kernel, "BigDataStoreTokenMte");
    CHECK_CONTAINS(kernel, "BigDataIpcAckOffset");
    CHECK_CONTAINS(kernel, "BigDataLocalIpcAckSlot");
    CHECK_CONTAINS(kernel, "BigDataRemoteIpcAckSlot");
    CHECK_CONTAINS(kernel, "rankSize > 1 ? rankSize - 1 : 1");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, copyDoneOffset, slot, rankSize, peer, copyShard)");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, recvCopyDoneOffset, slot, rankSize, peer, recvShard)");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, readySignalOffset, slot, rankSize, peer, 0U)");
    CHECK_CONTAINS(kernel, "remoteDataOffset =");
    CHECK_CONTAINS(kernel, "BigDataNetworkPeerIndex(rank, peer)");
    CHECK_CONTAINS(kernel, "BigDataSlot(udmaMem, recvDataOffset, slot, networkPeerCount");
    CHECK_CONTAINS(kernel, "UDMAPutSignalNbi<int32_t>");
    CHECK_CONTAINS(kernel, "localSrc,");
    CHECK_CONTAINS(kernel, "remoteDataOffset, chunkBytes, remoteReadyOffset, token");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_PINGPONG_BYTES");
    CHECK_CONTAINS(kernel, "relayLocal[bufferId * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES]");
    CHECK_CONTAINS(kernel, "globalPass - static_cast<uint64_t>(TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS)");
    CHECK_CONTAINS(kernel, "recvSlotInt[0]");
    CHECK_CONTAINS(kernel, "BigDataStoreTokenMte(remoteAck, token, relayLocal)");
    CHECK_CONTAINS(kernel, "BigDataRemoteIpcAckSlot(args, peer, rank, slot, rankSize)");
    CHECK_CONTAINS(kernel, "ackSignal");
    CHECK_NOT_CONTAINS(kernel, "UDMAPutNbi<uint64_t>");
    CHECK_CONTAINS(udma, "if (length == 0)");
    CHECK_CONTAINS(udma, "reinterpret_cast<uint64_t>(addr) + length - 1");
}

} // namespace

int main()
{
    TestAllToAllInputPattern();
    TestAllToAllOutputValidation();
    TestBuildAllToAllOutput();
    TestAllToAllMaxRank256With64MiBPerRank();
    TestAllToAllBigDataPlan();
    TestAllToAllBigDataBlockDim();
    TestDemoDebugLayoutSource();
    TestAllToAllDataAsFlagSource();
    TestAllToAllChunkedUdmaSource();
    TestAllToAllBigDataSource();
    if (g_failures != 0) {
        std::cerr << g_failures << " all-to-all layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA all-to-all layout checks passed" << std::endl;
    return 0;
}
