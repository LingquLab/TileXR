#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "demo/tilexr_udma_alltoall_layout.h"
#include "demo/tilexr_udma_fullmesh_trace.h"

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

std::string SliceBetween(const std::string& text, const std::string& begin, const std::string& end)
{
    const size_t beginPos = text.find(begin);
    if (beginPos == std::string::npos) {
        return "";
    }
    const size_t endPos = text.find(end, beginPos + begin.size());
    if (endPos == std::string::npos) {
        return text.substr(beginPos);
    }
    return text.substr(beginPos, endPos - beginPos);
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

void TestFullmeshTraceLayout()
{
    using namespace TileXR::Demo;
    CHECK_EQ(kFullmeshTraceMagic, 0x464d5452U);
    CHECK_EQ(kFullmeshTraceVersion, 1U);
    CHECK_EQ(kFullmeshTraceBytes, 8ULL * 1024ULL * 1024ULL);
    CHECK_EQ(kFullmeshTraceHeaderBytes, 4096ULL);
    CHECK_EQ(kFullmeshTraceMaxIterations, 50U);
    CHECK_EQ(kFullmeshTraceMaxCores, 35U);
    CHECK_EQ(kFullmeshTraceKernelRegions, 2U);
    CHECK_EQ(kFullmeshTracePhaseCount, 14U);
    CHECK_EQ(sizeof(FullmeshTraceSpan), 16U);
    CHECK_EQ(FullmeshTraceLayoutBytes(50U, 1U, 16U) <= kFullmeshTraceBytes, true);
    CHECK_EQ(FullmeshTraceLayoutBytes(50U, 4U, 16U) > kFullmeshTraceBytes, true);
    CHECK_EQ(FullmeshTraceLayoutFits(50U, 1U, 16U), true);
    CHECK_EQ(FullmeshTraceLayoutFits(50U, 4U, 16U), false);
    CHECK_EQ(FullmeshTraceLayoutFits(51U, 1U, 16U), false);
    const size_t first = FullmeshTraceTaskSpanOffset(0U, 0U, 0U, 0U, 0U, 1U, 16U);
    const size_t nextPhase = FullmeshTraceTaskSpanOffset(0U, 0U, 0U, 0U, 1U, 1U, 16U);
    const size_t nextPeer = FullmeshTraceTaskSpanOffset(0U, 0U, 0U, 1U, 0U, 1U, 16U);
    CHECK_EQ(nextPhase - first, sizeof(FullmeshTraceSpan));
    CHECK_EQ(nextPeer - first, kFullmeshTracePhaseCount * sizeof(FullmeshTraceSpan));
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

    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMaxRegisteredBytes, 128ULL * 1024ULL * 1024ULL);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeRegisteredBytes, 1024ULL * 1024ULL * 1024ULL);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodePeerSlotBytes, 16ULL * 1024ULL * 1024ULL);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataControlSlotBytes, 128ULL);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataCoresPerPeer, 5U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataSingleNodeShards, 2U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataLocalCopyShards, 2U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeCopyCores, 16U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeRecvCores, 16U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeControlShards, 32U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeRemoteSendPrimaryCore, 16U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeRemoteSendSecondaryCore, 17U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeLocalSendCore, 18U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeRecvCoreBase, 19U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataMultiNodeBlockDim, 35U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataRemotePutOnlyBlockDim, 64U);
    CHECK_EQ(TileXR::Demo::kAllToAllBigDataPingPongSlots, 2U);
    CHECK_EQ(plan.registeredBytes <= TileXR::Demo::kAllToAllBigDataMaxRegisteredBytes, true);
    const size_t controlGroupBytes =
        static_cast<size_t>(TileXR::Demo::kAllToAllBigDataPingPongSlots) *
        static_cast<size_t>(rankSize) *
        static_cast<size_t>(TileXR::Demo::AllToAllBigDataShardCount(rankSize)) *
        TileXR::Demo::kAllToAllBigDataControlSlotBytes;
    CHECK_EQ(plan.controlBytes, controlGroupBytes);
    CHECK_EQ(plan.signalBytes, 3ULL * controlGroupBytes);
    CHECK_EQ(plan.copyDoneOffset, plan.dataBytes);
    CHECK_EQ(plan.recvCopyDoneOffset, plan.copyDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.remoteSendDoneOffset, 0ULL);
    CHECK_EQ(plan.readySignalOffset, plan.recvCopyDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.ackSignalOffset, plan.readySignalOffset + controlGroupBytes);
    CHECK_EQ(plan.registeredBytes, TileXR::Demo::kAllToAllBigDataMaxRegisteredBytes);
    CHECK_EQ(plan.dataBytes + plan.controlBytes + plan.signalBytes <= plan.registeredBytes, true);
    CHECK_EQ(plan.dataBytes,
             static_cast<size_t>(rankSize - 1) * TileXR::Demo::kAllToAllBigDataPingPongSlots * 2ULL *
             plan.chunkBytesPerPeer);
    CHECK_EQ(plan.chunkElements > 0, true);
    CHECK_EQ(plan.chunkBytesPerPeer, static_cast<size_t>(plan.chunkElements) * sizeof(int32_t));
    CHECK_EQ(plan.passCount >= 1, true);
}

void TestAllToAllBigDataMultiNodePlanUses16ShardsAndRemoteSendDone()
{
    constexpr int rankSize = 16;
    constexpr int32_t elementsPerPeer = 2 * 1024 * 1024;
    const auto plan = TileXR::Demo::PlanAllToAllBigDataUdma(rankSize, elementsPerPeer);
    const size_t controlGroupBytes =
        static_cast<size_t>(plan.passCount) *
        static_cast<size_t>(rankSize) *
        static_cast<size_t>(TileXR::Demo::kAllToAllBigDataMultiNodeControlShards) *
        TileXR::Demo::kAllToAllBigDataControlSlotBytes;

    CHECK_EQ(TileXR::Demo::AllToAllBigDataShardCount(rankSize), 32U);
    CHECK_EQ(plan.registeredBytes, TileXR::Demo::kAllToAllBigDataMultiNodeRegisteredBytes);
    CHECK_EQ(plan.passCount, 1U);
    CHECK_EQ(plan.chunkBytesPerPeer, static_cast<size_t>(elementsPerPeer) * sizeof(int32_t));
    CHECK_EQ(plan.controlBytes, controlGroupBytes);
    CHECK_EQ(plan.signalBytes, 4ULL * controlGroupBytes);
    CHECK_EQ(plan.copyDoneOffset, plan.dataBytes);
    CHECK_EQ(plan.recvCopyDoneOffset, plan.copyDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.remoteSendDoneOffset, plan.recvCopyDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.readySignalOffset, plan.remoteSendDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.ackSignalOffset, plan.readySignalOffset + controlGroupBytes);
    CHECK_EQ(plan.dataBytes + plan.controlBytes + plan.signalBytes <= plan.registeredBytes, true);
    CHECK_EQ(plan.dataBytes,
             static_cast<size_t>(rankSize - 1) * static_cast<size_t>(plan.passCount) * 2ULL *
             static_cast<size_t>(elementsPerPeer) * sizeof(int32_t));
}

void TestAllToAllBigDataMultiNodeSmallPayloadUsesPayloadSlot()
{
    constexpr int rankSize = 64;
    constexpr int32_t elementsPerPeer = 1024 * 1024; // 4 MiB per peer for int32_t.
    const auto plan = TileXR::Demo::PlanAllToAllBigDataUdma(rankSize, elementsPerPeer);
    const size_t expectedChunkBytes = static_cast<size_t>(elementsPerPeer) * sizeof(int32_t);

    CHECK_EQ(plan.registeredBytes, TileXR::Demo::kAllToAllBigDataMultiNodeRegisteredBytes);
    CHECK_EQ(plan.passCount, 1U);
    CHECK_EQ(plan.chunkBytesPerPeer, expectedChunkBytes);
    CHECK_EQ(plan.dataBytes + plan.controlBytes + plan.signalBytes <= plan.registeredBytes, true);
    CHECK_EQ(plan.dataBytes,
             static_cast<size_t>(rankSize - 1) * static_cast<size_t>(plan.passCount) * 2ULL *
             expectedChunkBytes);
}

void TestAllToAllBigDataForce35CorePlanFor8P()
{
    constexpr int rankSize = 8;
    constexpr int32_t elementsPerPeer = 2 * 1024 * 1024;
    const auto plan = TileXR::Demo::PlanAllToAllBigDataUdma(rankSize, elementsPerPeer, true);
    const size_t controlGroupBytes =
        static_cast<size_t>(plan.passCount) *
        static_cast<size_t>(rankSize) *
        static_cast<size_t>(TileXR::Demo::kAllToAllBigDataMultiNodeControlShards) *
        TileXR::Demo::kAllToAllBigDataControlSlotBytes;

    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsMultiNode(rankSize), false);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataUse35Core(rankSize, true), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataShardCount(rankSize, true), 32U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(rankSize, true), 35U);
    CHECK_EQ(plan.registeredBytes, TileXR::Demo::kAllToAllBigDataMultiNodeRegisteredBytes);
    CHECK_EQ(plan.chunkBytesPerPeer, static_cast<size_t>(elementsPerPeer) * sizeof(int32_t));
    CHECK_EQ(plan.controlBytes, controlGroupBytes);
    CHECK_EQ(plan.signalBytes, 4ULL * controlGroupBytes);
    CHECK_EQ(plan.remoteSendDoneOffset, plan.recvCopyDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.readySignalOffset, plan.remoteSendDoneOffset + controlGroupBytes);
    CHECK_EQ(plan.ackSignalOffset, plan.readySignalOffset + controlGroupBytes);
    CHECK_EQ(plan.dataBytes + plan.controlBytes + plan.signalBytes <= plan.registeredBytes, true);
}

void TestAllToAllBigDataBlockDim()
{
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(0), 1U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(1), 5U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(8), 40U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(8, true), 35U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(16), 35U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(32), 35U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(64), 35U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(16, false, true), 64U);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataBlockDim(64, false, true), 64U);
}

void TestAllToAllBigDataMultiNodeTopology()
{
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsMultiNode(8), false);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsMultiNode(16), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsMultiNode(2, 1), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsMultiNode(4, 2), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataValidTopology(8), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataValidTopology(16), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataValidTopology(24), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataValidTopology(10), false);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataValidTopology(2, 1), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataValidTopology(4, 2), true);

    CHECK_EQ(TileXR::Demo::AllToAllBigDataLocalNodeBegin(0), 0);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataLocalNodeEnd(0), 8);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataLocalNodeBegin(10), 8);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataLocalNodeEnd(10), 16);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataLocalNodeBegin(2, 2), 2);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataLocalNodeEnd(2, 2), 4);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsLocalPeer(10, 8), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsLocalPeer(10, 15), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsLocalPeer(10, 7), false);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsLocalPeer(0, 1, 1), false);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsLocalPeer(2, 3, 2), true);
    CHECK_EQ(TileXR::Demo::AllToAllBigDataIsLocalPeer(2, 1, 2), false);
}

void TestAllToAllBigDataRemotePeerQueue()
{
    std::vector<int32_t> peers = TileXR::Demo::AllToAllBigDataRemotePeers(3, 16);
    std::vector<int32_t> expected {11, 12, 13, 14, 15, 8, 9, 10};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataRemotePeers(10, 16);
    expected = {2, 3, 4, 5, 6, 7, 0, 1};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataRemotePeers(5, 24);
    expected = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataRemotePeers(5, 8);
    CHECK_EQ(peers.empty(), true);

    peers = TileXR::Demo::AllToAllBigDataRemotePeers(0, 2, 1);
    expected = {1};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataRemotePeers(1, 4, 2);
    expected = {3, 2};
    CHECK_EQ(peers == expected, true);
}

void TestAllToAllBigDataMergedPeerQueue()
{
    std::vector<int32_t> peers = TileXR::Demo::AllToAllBigDataLocalPeers(3);
    std::vector<int32_t> expected {4, 5, 6, 7, 0, 1, 2};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataMergedPeerTasks(3, 16);
    expected = {11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataMergedPeerTasks(5, 32);
    expected = {13, 14, 15, 6, 16, 17, 18, 7, 19, 20, 21, 0,
                22, 23, 24, 1, 25, 26, 27, 2, 28, 29, 30,
                3, 31, 8, 9, 4, 10, 11, 12};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataMergedPeerTasks(5, 8);
    CHECK_EQ(peers.empty(), true);

    peers = TileXR::Demo::AllToAllBigDataMergedPeerTasks(0, 2, 1);
    expected = {1};
    CHECK_EQ(peers == expected, true);

    peers = TileXR::Demo::AllToAllBigDataMergedPeerTasks(1, 4, 2);
    expected = {3, 0, 2};
    CHECK_EQ(peers == expected, true);
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
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_DEBUG_READY_SEEN_BASE + TileXR::TILEXR_MAX_RANK_SIZE");
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
    CHECK_CONTAINS(demo, "TILEXR_DEMO_BIGDATA_FORCE_35CORE");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_BARRIER_HOST");
    CHECK_CONTAINS(demo, "host = value.substr(0, colon)");
    CHECK_CONTAINS(demo, "addr.sin_addr.s_addr = htonl(INADDR_ANY)");
    CHECK_CONTAINS(demo, "ConnectBarrierServer(endpoint.host, endpoint.port)");
    CHECK_CONTAINS(demo, "const uint64_t kernelLoopBase = static_cast<uint64_t>(iter)");
    CHECK_CONTAINS(demo, "bigDataPlan.passCount, 1, kernelLoopBase");
    CHECK_CONTAINS(demo, "bigDataPlan.copyDoneOffset");
    CHECK_CONTAINS(demo, "bigDataPlan.recvCopyDoneOffset");
    CHECK_CONTAINS(demo, "bigDataPlan.remoteSendDoneOffset");
    CHECK_CONTAINS(demo, "bigDataRanksPerNode");
    CHECK_CONTAINS(demo, "AllToAllBigDataValidTopology(rankSize, bigDataRanksPerNode)");
    CHECK_CONTAINS(demo, "ERROR: bigdata alltoall multi-node requires rankSize multiple of ranksPerNode");
    CHECK_CONTAINS(demo, "bigdata multinode mode=");
    CHECK_CONTAINS(demo, "shards=");
    CHECK_CONTAINS(demo, "remoteSendDoneOffset=");
    CHECK_CONTAINS(demo, "force35Core=");
    CHECK_CONTAINS(demo, "TileXR::Demo::AllToAllBigDataShardCount(");
    CHECK_CONTAINS(demo, "const uint32_t bigDataBlockDim = TileXR::Demo::AllToAllBigDataBlockDim(");
    CHECK_CONTAINS(demo, "rankSize, forceBigData35Core, bigDataRemotePutOnly, bigDataRanksPerNode");
    CHECK_CONTAINS(demo, "bigDataBlockDim, stream, commArgsDev");
    CHECK_CONTAINS(demo, "static_cast<uint8_t*>(registeredMemory) + bigDataPlan.copyDoneOffset");
    CHECK_CONTAINS(demo, "bigDataPlan.controlBytes + bigDataPlan.signalBytes");
    CHECK_CONTAINS(demo, "static_cast<uint32_t>(bigDataProfileStage)");
    CHECK_CONTAINS(demo, "static_cast<uint32_t>(bigDataRanksPerNode) << 8U");
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
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES = 128ULL");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER = 5U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_SINGLE_NODE_SHARDS = 2U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_LOCAL_COPY_SHARDS");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS = 2U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_PEER_SLOT_BYTES = 16ULL * 1024ULL * 1024ULL");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_RANKS_PER_NODE = 8");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_CONTROL_SHARDS = 32U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_BLOCK_DIM =");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_PUT_ONLY_BLOCK_DIM = 64U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_PRIMARY_CORE = 16U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_SECONDARY_CORE = 17U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_LOCAL_SEND_CORE = 18U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORE_BASE = 19U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_PRIMARY = 2U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_SECONDARY = 3U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_LOCAL_COPY_READY = 4U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_WAIT_CORE = 20U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_LOCAL_FANOUT_SHARD_BASE = 5U");
    CHECK_CONTAINS(kernel, "BigDataNormalizeRanksPerNode");
    CHECK_CONTAINS(kernel, "force35CoreFlag >> 8U");
    CHECK_CONTAINS(kernel, "BigDataIsMultiNode(rankSize, ranksPerNode)");
    CHECK_CONTAINS(kernel, "BigDataValidTopology(rankSize, ranksPerNode)");
    CHECK_CONTAINS(kernel, "BigDataUse35Core(rankSize, force35Core, ranksPerNode)");
    CHECK_CONTAINS(kernel, "BigDataShardCount(rankSize, force35Core, ranksPerNode)");
    CHECK_CONTAINS(kernel, "BigDataTaskCount(rankSize, force35Core, ranksPerNode)");
    CHECK_CONTAINS(kernel, "BigDataNodeCount(rankSize, localRanks)");
    CHECK_CONTAINS(kernel, "BigDataRemotePeerAt(rank, rankSize, remoteIndex, localRanks)");
    CHECK_CONTAINS(kernel, "BigDataRemotePeerForwardAt(rank, rankSize, remoteIndex, ranksPerNode)");
    CHECK_CONTAINS(kernel, "BigDataLocalPeerAt");
    CHECK_CONTAINS(kernel, "BigDataMergedPeerTaskAt(");
    CHECK_CONTAINS(kernel, "BigDataRunSelfCopyShard(rank, rankSize");
    CHECK_CONTAINS(kernel, "BigDataRemoteSendSegmentWorker");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlySendWorker");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlySendTaskCount");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlySendTaskRemoteIndex");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlySendTaskSegment");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlyCheckIndex");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlyCheckWorker");
    CHECK_CONTAINS(kernel, "BigDataRemoteIpcAckSlot");
    CHECK_CONTAINS(kernel, "BigDataLocalIpcAckSlot");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlyPublishAck");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlyPublishReady");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlyWaitPeerAck");
    CHECK_CONTAINS(kernel, "BigDataStoreInt32Mte");
    CHECK_CONTAINS(kernel, "BigDataRemoteSendSegmentRange");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_FRAMEWORK");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_LOOP");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_PEER");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_QP");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_SEGMENT");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_ADDRESS");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_POST");
    CHECK_CONTAINS(kernel, "TILEXR_BIGDATA_REMOTE_PUT_STAGE_ACK");
    CHECK_CONTAINS(kernel, "BigDataSelectWeightedQp(");
    CHECK_CONTAINS(kernel, "BigDataSelectDistinctWeightedQp(");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlySegmentQp(");
    CHECK_CONTAINS(kernel, "(void)segmentId");
    CHECK_CONTAINS(kernel, "return BigDataSelectWeightedQp(args, peer, true)");
    CHECK_NOT_CONTAINS(kernel, "return 1U;");
    CHECK_NOT_CONTAINS(kernel, "return segmentId == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT ? 0U : 1U");
    CHECK_CONTAINS(kernel, "BigDataWaitCopyDoneRange");
    CHECK_CONTAINS(kernel, "BigDataPublishCopyReadyRange");
    CHECK_CONTAINS(kernel, "BigDataWaitCopyReady");
    CHECK_CONTAINS(kernel, "BigDataPublishReadySignal");
    CHECK_CONTAINS(kernel, "remoteSendDoneOffset");
    CHECK_CONTAINS(kernel, "!BigDataIsMultiNode(rankSize, ranksPerNode)");
    CHECK_CONTAINS(kernel, "const uint32_t activeBlockDim = remotePutOnly ?");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_PUT_ONLY_BLOCK_DIM");
    CHECK_CONTAINS(kernel, "if (blockIdx >= static_cast<int32_t>(activeBlockDim))");
    CHECK_CONTAINS(kernel, "const bool isCopyCore =");
    CHECK_CONTAINS(kernel, "const bool isRemoteSendPrimaryCore =");
    CHECK_CONTAINS(kernel, "const bool isRemoteSendSecondaryCore =");
    CHECK_CONTAINS(kernel, "const bool isLocalSendCore =");
    CHECK_CONTAINS(kernel, "const bool isRecvCore =");
    CHECK_CONTAINS(kernel, "if (isCopyCore)");
    CHECK_CONTAINS(kernel, "if (remotePutOnly)");
    CHECK_CONTAINS(kernel, "BigDataRemotePutOnlyCheckWorker(");
    CHECK_CONTAINS(kernel, "const int32_t sendTaskCount = BigDataRemotePutOnlySendTaskCount(remoteTaskCount)");
    CHECK_CONTAINS(kernel, "sendTask += static_cast<int32_t>(activeBlockDim)");
    CHECK_CONTAINS(kernel, "const int32_t remoteIndex = BigDataRemotePutOnlySendTaskRemoteIndex(sendTask, remoteTaskCount)");
    CHECK_CONTAINS(kernel, "const uint32_t segmentId = BigDataRemotePutOnlySendTaskSegment(sendTask, remoteTaskCount)");
    CHECK_CONTAINS(kernel, "remotePutOnly && profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_FRAMEWORK");
    CHECK_CONTAINS(kernel, "profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_LOOP");
    CHECK_CONTAINS(kernel, "profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_PEER");
    CHECK_CONTAINS(kernel, "profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_QP");
    CHECK_CONTAINS(kernel, "profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_SEGMENT");
    CHECK_CONTAINS(kernel, "profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_ADDRESS");
    CHECK_CONTAINS(kernel, "BigDataKernelExitBarrier()");
    CHECK_CONTAINS(kernel, "if (!isLocalPeer && isRemoteSendPrimaryCore)");
    CHECK_CONTAINS(kernel, "if (!isLocalPeer && isRemoteSendSecondaryCore)");
    CHECK_CONTAINS(kernel, "if (isLocalPeer && isLocalSendCore)");
    CHECK_CONTAINS(kernel, "if (isRecvCore)");
    CHECK_CONTAINS(kernel, "BigDataCopyPeerWorker");
    CHECK_CONTAINS(kernel, "BigDataSendPeerWorker");
    CHECK_CONTAINS(kernel, "BigDataRecvPeerWorker");
    CHECK_CONTAINS(kernel, "BigDataCopyShardRange");
    CHECK_CONTAINS(kernel, "copyShard");
    CHECK_CONTAINS(kernel, "recvShard");
    CHECK_CONTAINS(kernel, "recvCopyDoneOffset");
    CHECK_CONTAINS(kernel, "copyShardBegin = 0U");
    CHECK_CONTAINS(kernel, "copyShardEnd = TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END");
    CHECK_CONTAINS(kernel, "copyShardBegin = TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END");
    CHECK_CONTAINS(kernel, "copyShardEnd = dataShardCount");
    CHECK_CONTAINS(kernel, "copyShard == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_AGGREGATOR");
    CHECK_CONTAINS(kernel, "BigDataDataSlot(globalPass, pass, use35Core)");
    CHECK_CONTAINS(kernel, "BigDataNetworkPeerIndex");
    CHECK_CONTAINS(kernel, "BigDataStoreTokenMte");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, copyReadyOffset, slot, rankSize, shardCount, peer, readyShard)");
    CHECK_CONTAINS(kernel, "BigDataIpcAckOffset");
    CHECK_CONTAINS(kernel, "BigDataLocalIpcAckSlot");
    CHECK_CONTAINS(kernel, "BigDataRemoteIpcAckSlot");
    CHECK_CONTAINS(kernel, "BigDataRemoteRegisteredControlSlot");
    CHECK_CONTAINS(kernel, "rankSize > 1 ? rankSize - 1 : 1");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, copyDoneOffset, slot, rankSize, shardCount, peer, copyShard)");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, recvCopyDoneOffset, slot, rankSize, shardCount, peer, recvShard)");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, readySignalOffset, slot, rankSize, shardCount, peer, 0U)");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_SOURCE_SHARD");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_LOCAL_FANOUT_SHARD_BASE + recvShard");
    CHECK_CONTAINS(kernel, "remoteDataOffset =");
    CHECK_CONTAINS(kernel, "BigDataNetworkPeerIndex(rank, peer)");
    CHECK_CONTAINS(kernel, "BigDataSlot(udmaMem, recvDataOffset, slot, networkPeerCount");
    CHECK_CONTAINS(kernel, "UDMAPutSignalNbi<int32_t>");
    CHECK_CONTAINS(kernel, "UDMAPutNbiOnQp<int32_t>");
    CHECK_CONTAINS(kernel, "UDMAPutSignalNbiOnQp<uint64_t>");
    CHECK_CONTAINS(kernel, "UDMAPutNbi<int32_t>");
    CHECK_CONTAINS(kernel, "localSrc,");
    CHECK_CONTAINS(kernel, "remoteDataOffset, chunkBytes, remoteReadyOffset, token");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_PINGPONG_BYTES");
    CHECK_CONTAINS(kernel, "relayLocal[bufferId * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES]");
    CHECK_CONTAINS(kernel, "if (!use35Core && profileStage > TILEXR_BIGDATA_PROFILE_STAGE_ACK_PUT");
    CHECK_CONTAINS(kernel, "if (use35Core && profileStage == TILEXR_BIGDATA_PROFILE_STAGE_ACK_PUT");
    CHECK_CONTAINS(kernel, "BigDataControlSlot(udmaMem, ackSignalOffset, slot, rankSize, shardCount, peer, 0U)");
    CHECK_CONTAINS(kernel, "use35Core ? passCount : TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS");
    CHECK_CONTAINS(kernel, "recvSlotInt[0]");
    CHECK_NOT_CONTAINS(kernel, "BigDataPublishAckSignalUdma");
    CHECK_CONTAINS(kernel, "BigDataRemoteIpcAckSlot(args, peer, rank, slot, rankSize)");
    CHECK_CONTAINS(kernel, "ackSignal");
    CHECK_CONTAINS(kernel, "fillLocal.SetValue(0, value)");
    CHECK_CONTAINS(udma, "if (length == 0)");
    CHECK_CONTAINS(udma, "reinterpret_cast<uint64_t>(addr) + length - 1");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_BIGDATA_REMOTE_PUT_ONLY");
    CHECK_CONTAINS(demo, "skip result validation for bigdata remote-put-only profile");
    CHECK_CONTAINS(demo, "TILEXR_UDMA_FULLMESH_TRACE");
    CHECK_CONTAINS(demo, "TILEXR_UDMA_FULLMESH_TRACE_DIR");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_BIGDATA_ISOLATED_TASK");
    CHECK_CONTAINS(demo, "bigDataIsolatedTask < 0 || bigDataIsolatedTask > 11");
    CHECK_CONTAINS(demo, "bigDataProfilePartial = bigDataProfilePartial || bigDataIsolatedTask != 0");
    CHECK_CONTAINS(demo, "FullmeshTraceLayoutFits(");
    CHECK_CONTAINS(demo, "aclrtMalloc(&fullmeshTraceDevice");
    CHECK_CONTAINS(demo, "TileXR::Demo::kFullmeshTraceBytes, \"fullmesh trace\"");
    CHECK_CONTAINS(demo, "tilexr_fullmesh_trace_rank_");
    CHECK_CONTAINS(demo, "fullmeshTraceIteration");
    CHECK_CONTAINS(demo, "reinterpret_cast<GM_ADDR>(fullmeshTraceDevice)");

    const std::string fullMeshReady = SliceBetween(
        kernel, "BigDataPublishReadySignal", "BigDataRemoteSendSegmentWorker");
    const std::string localSend = SliceBetween(
        kernel, "BigDataSendPeerWorker", "BigDataRecvPeerWorker");
    const std::string fullMeshSend = SliceBetween(
        kernel, "BigDataRemoteSendSegmentWorker", "BigDataRemotePutOnlySendWorker");
    const std::string fullMeshRecv = SliceBetween(
        kernel, "BigDataRecvPeerWorker", "BigDataWaitCopyDoneRange");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END = 16U");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_AGGREGATOR = 15U");
    CHECK_CONTAINS(kernel, "if (shard == shardCount)");
    CHECK_CONTAINS(kernel, "return totalElements * static_cast<uint32_t>(sizeof(int32_t))");
    CHECK_CONTAINS(fullMeshReady, "UDMAPutSignalNbiOnQp<uint64_t>");
    CHECK_CONTAINS(fullMeshReady, "UDMAQuietStatusOnQp(args, peer, qpIdx)");
    CHECK_CONTAINS(fullMeshSend, "BigDataSelectWeightedQp(");
    CHECK_CONTAINS(fullMeshSend, "UDMAPutNbiOnQp<int32_t>");
    CHECK_CONTAINS(fullMeshSend, "UDMAQuietStatusOnQp(args, peer, qpIdx)");
    CHECK_NOT_CONTAINS(fullMeshSend, "TileXR::UDMAPutNbi<int32_t>(args, peer");
    CHECK_CONTAINS(fullMeshRecv, "BigDataRemoteRegisteredControlSlot(");
    CHECK_CONTAINS(fullMeshRecv, "BigDataStoreTokenMte(remoteAck, token, relayLocal)");
    CHECK_NOT_CONTAINS(fullMeshRecv, "BigDataPublishAckSignalUdma(");
    CHECK_CONTAINS(kernel, "tilexr_udma_fullmesh_trace.h");
    CHECK_CONTAINS(kernel, "BigDataFullmeshTraceRecordKernelSpan");
    CHECK_CONTAINS(kernel, "BigDataFullmeshTraceRecordTaskSpan");
    CHECK_CONTAINS(kernel, "if (trace == nullptr");
    CHECK_CONTAINS(kernel, "AscendC::GetSystemCycle()");
    CHECK_CONTAINS(kernel, "auto fullmeshTrace = (!remotePutOnly && use35Core &&");
    CHECK_CONTAINS(kernel, "uint32_t isolatedTask");
    CHECK_CONTAINS(kernel, "BigDataRunIsolatedTask(");
    CHECK_CONTAINS(kernel, "kFullmeshTracePhaseSelfCopy");
    CHECK_CONTAINS(kernel, "kFullmeshTracePhasePeerCopy");
    CHECK_CONTAINS(kernel, "kFullmeshTracePhasePublishCopyReady");
    CHECK_CONTAINS(fullMeshSend, "kFullmeshTracePhaseWaitCopyReady");
    CHECK_CONTAINS(fullMeshSend, "kFullmeshTracePhaseDataPut");
    CHECK_CONTAINS(fullMeshSend, "kFullmeshTracePhaseQuiet");
    CHECK_CONTAINS(fullMeshSend, "kFullmeshTracePhaseSegmentDone");
    CHECK_CONTAINS(fullMeshSend, "kFullmeshTracePhasePublishReady");
    CHECK_CONTAINS(localSend, "kFullmeshTracePhaseWaitCopyReady");
    CHECK_CONTAINS(localSend, "kFullmeshTracePhaseDataPut");
    CHECK_CONTAINS(localSend, "kFullmeshTracePhaseQuiet");
    CHECK_CONTAINS(fullMeshRecv, "kFullmeshTracePhaseWaitReady");
    CHECK_CONTAINS(fullMeshRecv, "kFullmeshTracePhaseOutputCopy");
    CHECK_CONTAINS(fullMeshRecv, "kFullmeshTracePhasePublishRecvDone");
    CHECK_CONTAINS(fullMeshRecv, "kFullmeshTracePhaseWaitRecvDone");
    CHECK_CONTAINS(fullMeshRecv, "kFullmeshTracePhaseAck");

    const std::string isolatedTasks = SliceBetween(
        kernel, "BigDataRunIsolatedTask", "BigDataRunRoleForPeer");
    for (uint32_t task = 1U; task <= 11U; ++task) {
        CHECK_CONTAINS(isolatedTasks, "TILEXR_BIGDATA_ISOLATED_TASK_" + std::to_string(task));
    }
    CHECK_NOT_CONTAINS(isolatedTasks, "BigDataWaitTokenMte(");
    CHECK_NOT_CONTAINS(isolatedTasks, "BigDataWaitCopyReady(");
    CHECK_CONTAINS(isolatedTasks, "UDMAPutNbiOnQp<int32_t>");
    CHECK_CONTAINS(isolatedTasks, "UDMAQuietStatusOnQp(args, peer, qpIdx)");
    CHECK_CONTAINS(isolatedTasks, "UDMAPutSignalNbi<int32_t>");
    CHECK_CONTAINS(isolatedTasks, "BigDataRemoteRegisteredControlSlot(");

    const std::string remotePutOnlySend = SliceBetween(
        kernel, "BigDataRemotePutOnlySendWorker", "BigDataRemotePutOnlyCheckIndex");
    const std::string remotePutOnlyCheck = SliceBetween(
        kernel, "BigDataRemotePutOnlyCheckWorker", "BigDataRunSelfCopyShard");
    CHECK_CONTAINS(remotePutOnlySend, "UDMAPutNbiOnQp<int32_t>");
    CHECK_CONTAINS(remotePutOnlySend, "BigDataRemoteIpcReadySlot");
    CHECK_CONTAINS(remotePutOnlySend, "BigDataRemotePutOnlyPublishReady");
    CHECK_CONTAINS(remotePutOnlySend, "UDMAQuietStatusOnQp");
    CHECK_NOT_CONTAINS(remotePutOnlySend, "UDMAPutNbiOnQp<uint64_t>");
    CHECK_NOT_CONTAINS(remotePutOnlySend, "BigDataStoreInt32Mte");
    const std::string remotePutOnlyCheckIndex = SliceBetween(
        kernel, "BigDataRemotePutOnlyCheckIndex", "BigDataRemotePutOnlyCheckWorker");
    CHECK_CONTAINS(remotePutOnlyCheckIndex, "return blockIdx");
    CHECK_NOT_CONTAINS(remotePutOnlyCheckIndex, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_PRIMARY_CORE");
    CHECK_NOT_CONTAINS(remotePutOnlyCheckIndex, "TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_SECONDARY_CORE");
    CHECK_CONTAINS(remotePutOnlyCheck, "observed");
    CHECK_CONTAINS(remotePutOnlyCheck, "BigDataLocalIpcReadySlot");
    CHECK_CONTAINS(remotePutOnlyCheck, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT");
    CHECK_CONTAINS(remotePutOnlyCheck, "TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_SEGMENT");
    CHECK_CONTAINS(remotePutOnlyCheck, "TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS");
    CHECK_NOT_CONTAINS(remotePutOnlyCheck, "BigDataRemotePutOnlyPublishAck");
    CHECK_NOT_CONTAINS(remotePutOnlyCheck, "BigDataRemotePutOnlyWaitPeerAck");
    CHECK_NOT_CONTAINS(remotePutOnlyCheck, "BigDataRemoteIpcAckSlot");
    CHECK_NOT_CONTAINS(remotePutOnlyCheck, "BigDataLocalIpcAckSlot");
    CHECK_NOT_CONTAINS(remotePutOnlyCheck, "UDMAQuietStatusOnQp");
}

} // namespace

int main()
{
    TestFullmeshTraceLayout();
    TestAllToAllInputPattern();
    TestAllToAllOutputValidation();
    TestBuildAllToAllOutput();
    TestAllToAllMaxRank256With64MiBPerRank();
    TestAllToAllBigDataPlan();
    TestAllToAllBigDataMultiNodePlanUses16ShardsAndRemoteSendDone();
    TestAllToAllBigDataMultiNodeSmallPayloadUsesPayloadSlot();
    TestAllToAllBigDataForce35CorePlanFor8P();
    TestAllToAllBigDataBlockDim();
    TestAllToAllBigDataMultiNodeTopology();
    TestAllToAllBigDataRemotePeerQueue();
    TestAllToAllBigDataMergedPeerQueue();
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
