#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "demo/tilexr_udma_alltoall_group_layout.h"
#include "demo/tilexr_udma_alltoall_group_route.h"
#include "demo/tilexr_udma_alltoall_group_trace.h"

namespace {

#ifndef TILEXR_SOURCE_ROOT
#define TILEXR_SOURCE_ROOT "."
#endif

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

void CheckSchedule(int rankSize, uint32_t groupWidth = TileXR::Demo::kAllToAllGroupWidth)
{
    for (int rank = 0; rank < rankSize; ++rank) {
        std::set<int> peers;
        for (uint32_t group = 0;
             group < TileXR::Demo::AllToAllGroupCount(rankSize, groupWidth); ++group) {
            for (uint32_t lane = 0; lane < groupWidth; ++lane) {
                const int peer = TileXR::Demo::AllToAllGroupPeer(
                    rank, rankSize, group, lane, groupWidth);
                if (peer < 0) {
                    continue;
                }
                CHECK_EQ(peer == rank, false);
                CHECK_EQ(peers.insert(peer).second, true);
                bool symmetric = false;
                for (uint32_t remoteLane = 0; remoteLane < groupWidth; ++remoteLane) {
                    symmetric = symmetric ||
                        TileXR::Demo::AllToAllGroupPeer(
                            peer, rankSize, group, remoteLane, groupWidth) == rank;
                }
                CHECK_EQ(symmetric, true);
            }
        }
        CHECK_EQ(peers.size(), static_cast<size_t>(rankSize - 1));
    }
}

void TestSchedules()
{
    for (int rankSize : {8, 16, 24, 32, 40, 64, 128, 256, 512, 1024}) {
        CheckSchedule(rankSize);
        CheckSchedule(rankSize, TileXR::Demo::kAllToAllGroupExperimentalWidth);
    }
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(8), 1U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(16), 1U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(24), 2U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(64), 4U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(128), 8U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(1024), 64U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(7), 0U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(18), 0U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(1032), 0U);

    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 0), 1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 7), 8);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 8), 63);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 15), 56);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 3, 7), 32);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 3, 15), -1);

    constexpr uint32_t width = TileXR::Demo::kAllToAllGroupExperimentalWidth;
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(16, width), 4U);
    const int expected[4][4] = {
        {1, 2, 15, 14},
        {3, 4, 13, 12},
        {5, 6, 11, 10},
        {7, 8, 9, -1},
    };
    for (uint32_t group = 0U; group < 4U; ++group) {
        for (uint32_t lane = 0U; lane < width; ++lane) {
            CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(
                0, 16, group, lane, width), expected[group][lane]);
        }
    }
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(16, 8U), 0U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 16, 0, 4, width), -1);
}

void TestPlan()
{
    constexpr int rankSize = 16;
    constexpr int32_t elementsPerPeer = 2 * 1024 * 1024;
    const auto plan = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer);

    CHECK_EQ(plan.valid, true);
    CHECK_EQ(plan.groupCount, 1U);
    CHECK_EQ(plan.passCount, 1U);
    CHECK_EQ(plan.chunkElements, elementsPerPeer);
    CHECK_EQ(plan.bytesPerPeer, 8ULL * 1024ULL * 1024ULL);
    CHECK_EQ(plan.payloadPlaneBytes, 128ULL * 1024ULL * 1024ULL);
    CHECK_EQ(plan.payloadOffset[0], 0ULL);
    CHECK_EQ(plan.payloadOffset[1] >= plan.payloadOffset[0] + plan.payloadPlaneBytes, true);
    CHECK_EQ(TileXR::Demo::kAllToAllGroupRouteSignalStride, 512U);
    CHECK_EQ(TileXR::Demo::kAllToAllGroupSignalSlotBytes, 1024U);
    CHECK_EQ(plan.signalPlaneBytes, static_cast<size_t>(rankSize) * 1024ULL);
    CHECK_EQ(TileXR::Demo::AllToAllGroupSignalByteOffset(3U, 0U), 3072ULL);
    CHECK_EQ(TileXR::Demo::AllToAllGroupSignalByteOffset(3U, 1U), 3584ULL);
    CHECK_EQ(plan.signalOffset[0] >= plan.payloadOffset[1] + plan.payloadPlaneBytes, true);
    CHECK_EQ(plan.signalOffset[1] >= plan.signalOffset[0] + plan.signalPlaneBytes, true);
    CHECK_EQ(plan.creditPlaneBytes, 0ULL);
    CHECK_EQ(plan.creditOffset[0], 0ULL);
    CHECK_EQ(plan.creditOffset[1], 0ULL);
    CHECK_EQ(plan.controlOffset >= plan.signalOffset[1] + plan.signalPlaneBytes, true);
    CHECK_EQ(plan.signalSourceOffset,
        plan.controlOffset + TileXR::Demo::kAllToAllGroupErrorBytes);
    CHECK_EQ(plan.signalSourceBytes,
        TileXR::Demo::kAllToAllGroupSignalSourceBytes);
    CHECK_EQ(plan.signalSourceOffset + plan.signalSourceBytes <=
        plan.controlOffset + plan.controlBytes, true);
    const auto legacyAlign = [](size_t value) {
        return (value + TileXR::Demo::kAllToAllGroupAlignment - 1U) &
            ~(TileXR::Demo::kAllToAllGroupAlignment - 1U);
    };
    const size_t legacyPayloadOffset1 = legacyAlign(plan.payloadPlaneBytes);
    const size_t legacySignalOffset0 = legacyAlign(
        legacyPayloadOffset1 + plan.payloadPlaneBytes);
    const size_t legacySignalOffset1 = legacyAlign(
        legacySignalOffset0 + plan.signalPlaneBytes);
    const size_t legacyControlOffset = legacyAlign(
        legacySignalOffset1 + plan.signalPlaneBytes);
    const size_t legacyRegisteredBytes = legacyAlign(
        legacyControlOffset + TileXR::Demo::kAllToAllGroupBaseControlBytes);
    CHECK_EQ(plan.payloadOffset[1], legacyPayloadOffset1);
    CHECK_EQ(plan.signalOffset[0], legacySignalOffset0);
    CHECK_EQ(plan.signalOffset[1], legacySignalOffset1);
    CHECK_EQ(plan.controlOffset, legacyControlOffset);
    CHECK_EQ(plan.registeredBytes, legacyRegisteredBytes);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCreditByteOffset(3U),
        3ULL * TileXR::Demo::kAllToAllGroupCreditStride);
    CHECK_EQ(plan.registeredBytes <= TileXR::Demo::kAllToAllGroupMaxRegisteredBytes, true);
    CHECK_EQ(TileXR::Demo::kAllToAllGroupMaxPayloadBytes, 16ULL << 30);

    const auto ingressPlan = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer,
        TileXR::Demo::kAllToAllGroupWidth, 1U);
    CHECK_EQ(ingressPlan.valid, true);
    CHECK_EQ(ingressPlan.creditPlaneBytes,
        static_cast<size_t>(rankSize) * TileXR::Demo::kAllToAllGroupCreditStride);
    CHECK_EQ(ingressPlan.creditOffset[0], 0ULL);
    CHECK_EQ(ingressPlan.creditOffset[1],
        TileXR::Demo::kAllToAllGroupCreditSlotBytes);
    CHECK_EQ(ingressPlan.registeredBytes, plan.registeredBytes);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer,
        TileXR::Demo::kAllToAllGroupExperimentalWidth, 1U).valid, false);

    const auto chunked = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer / 4);
    CHECK_EQ(chunked.valid, true);
    CHECK_EQ(chunked.passCount, 4U);
    CHECK_EQ(chunked.payloadPlaneBytes, plan.payloadPlaneBytes);

    const auto widthFour = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer,
        TileXR::Demo::kAllToAllGroupExperimentalWidth);
    CHECK_EQ(widthFour.valid, true);
    CHECK_EQ(widthFour.groupWidth, 4U);
    CHECK_EQ(widthFour.groupCount, 4U);

    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(7, 1024, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(18, 1024, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(1032, 1024, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(16, 0, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(16, elementsPerPeer, 0).valid, false);

    constexpr int32_t thirtyTwoMiBElements = 8 * 1024 * 1024;
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(
        rankSize, thirtyTwoMiBElements, thirtyTwoMiBElements).valid, true);

    constexpr int32_t oneGiBPerRankElementsPerPeer = 16 * 1024 * 1024;
    const auto oneGiBPerRank = TileXR::Demo::PlanAllToAllGroup(
        rankSize, oneGiBPerRankElementsPerPeer, oneGiBPerRankElementsPerPeer);
    CHECK_EQ(oneGiBPerRank.valid, true);
    CHECK_EQ(oneGiBPerRank.passCount, 1U);
    CHECK_EQ(oneGiBPerRank.payloadPlaneBytes, 1ULL << 30);
    CHECK_EQ(oneGiBPerRank.registeredBytes > (2ULL << 30), true);
    CHECK_EQ(oneGiBPerRank.registeredBytes <=
        TileXR::Demo::kAllToAllGroupMaxRegisteredBytes, true);

    constexpr size_t largeRankBytes[] = {
        2ULL << 30, 4ULL << 30, 8ULL << 30, 16ULL << 30};
    for (const size_t rankBytes : largeRankBytes) {
        const int32_t largeElementsPerPeer = static_cast<int32_t>(
            rankBytes / (static_cast<size_t>(rankSize) * sizeof(int32_t)));
        const auto largePlan = TileXR::Demo::PlanAllToAllGroup(
            rankSize, largeElementsPerPeer, largeElementsPerPeer);
        CHECK_EQ(largePlan.valid, true);
        CHECK_EQ(largePlan.passCount, 1U);
        CHECK_EQ(largePlan.payloadPlaneBytes, rankBytes);
        CHECK_EQ(largePlan.registeredBytes <=
            TileXR::Demo::kAllToAllGroupMaxRegisteredBytes, true);
    }

    constexpr size_t tooLargeRankBytes = 32ULL << 30;
    const int32_t tooLargeElementsPerPeer = static_cast<int32_t>(
        tooLargeRankBytes / (static_cast<size_t>(rankSize) * sizeof(int32_t)));
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(
        rankSize, tooLargeElementsPerPeer,
        tooLargeElementsPerPeer).valid, false);
}

void TestChannelPolicy()
{
    using TileXR::Demo::AllToAllGroupChannelMode;
    constexpr size_t threshold = 150ULL * 1024ULL * 1024ULL;
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidChannelMode(0U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidChannelMode(1U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidChannelMode(2U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidChannelMode(3U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupUseMultiChannel(
        threshold, AllToAllGroupChannelMode::kAuto), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupUseMultiChannel(
        threshold + sizeof(int32_t), AllToAllGroupChannelMode::kAuto), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupUseMultiChannel(
        threshold * 2U, AllToAllGroupChannelMode::kSingle), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupUseMultiChannel(
        sizeof(int32_t), AllToAllGroupChannelMode::kMulti), true);

    CHECK_EQ(TileXR::Demo::kAllToAllGroupSendWorkerCount, 32U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidWidth(4U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidWidth(16U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidWidth(8U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(1U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(2U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(4U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(8U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(16U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(32U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(64U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(0U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(3U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidQuietBatch(65U), false);
}

void TestScalePlanAndTraceCapacity()
{
    constexpr int rankSize = 1024;
    constexpr int32_t elementsPerPeer = 32768;
    const auto plan = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer);

    CHECK_EQ(plan.valid, true);
    CHECK_EQ(plan.groupCount, 64U);
    CHECK_EQ(plan.passCount, 1U);
    CHECK_EQ(plan.payloadPlaneBytes, 128ULL * 1024ULL * 1024ULL);
    CHECK_EQ(plan.registeredBytes <= TileXR::Demo::kAllToAllGroupMaxRegisteredBytes, true);

    CHECK_EQ(TileXR::Demo::kAllToAllGroupTraceBytes,
        128ULL * 1024ULL * 1024ULL);
    CHECK_EQ(TileXR::Demo::AllToAllGroupTraceLayoutFits(50U, 64U, 3U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupTraceLayoutFits(50U, 64U, 4U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupTraceLayoutFits(50U, 64U, 5U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupTraceLayoutFits(50U, 64U, 6U), false);
}

void TestTokens()
{
    const uint64_t token48 = TileXR::Demo::AllToAllGroupToken(48U, 0U, 0U);
    const uint64_t token49 = TileXR::Demo::AllToAllGroupToken(49U, 0U, 0U);
    CHECK_EQ(token48 != 0U, true);
    CHECK_EQ(token49 > token48, true);
    CHECK_EQ((token48 >> 31U) & 1ULL, 0ULL);
    CHECK_EQ((token49 >> 31U) & 1ULL, 1ULL);
    CHECK_EQ(TileXR::Demo::AllToAllGroupToken(49U, 1U, 0U) > token49, true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupToken(49U, 1U, 1U) >
        TileXR::Demo::AllToAllGroupToken(49U, 1U, 0U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCreditToken(49U, 1U),
        TileXR::Demo::AllToAllGroupToken(49U, 1U, 0U));
    CHECK_EQ(TileXR::Demo::AllToAllGroupCreditToken(50U, 1U) >
        TileXR::Demo::AllToAllGroupCreditToken(49U, 1U), true);
}

void TestIngressCreditPolicy()
{
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidIngressWindow(0U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidIngressWindow(1U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidIngressWindow(2U), false);

    constexpr int rankSize = 256;
    constexpr uint32_t groupCount = 16U;
    for (int rank = 0; rank < rankSize; ++rank) {
        for (uint32_t group = 0U; group < groupCount; ++group) {
            for (uint32_t lane = 0U;
                 lane < TileXR::Demo::kAllToAllGroupWidth; ++lane) {
                const int32_t nextPeer = TileXR::Demo::AllToAllGroupNextCreditPeer(
                    rank, rankSize, group, lane);
                const int32_t expected = group + 1U < groupCount ?
                    TileXR::Demo::AllToAllGroupPeer(
                        rank, rankSize, group + 1U, lane) : -1;
                CHECK_EQ(nextPeer, expected);
                if (nextPeer >= 0) {
                    uint32_t senderLane = lane ^
                        TileXR::Demo::kAllToAllGroupHalfWidth;
                    if (TileXR::Demo::AllToAllGroupPeer(
                            nextPeer, rankSize, group + 1U,
                            senderLane) < 0) {
                        senderLane = lane;
                    }
                    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(
                        nextPeer, rankSize, group + 1U, senderLane), rank);
                }
            }
        }
    }

    CHECK_EQ(TileXR::Demo::AllToAllGroupNextCreditPeer(0, 64, 2U, 0U), 25);
    CHECK_EQ(TileXR::Demo::AllToAllGroupNextCreditPeer(0, 64, 3U, 0U), -1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupNextCreditPeer(0, 64, 2U, 15U), -1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCreditOwner(0U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCreditOwner(15U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCreditOwner(16U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCreditOwner(31U), false);
    for (uint32_t lane = 0U;
         lane < TileXR::Demo::kAllToAllGroupSendCoreCount; ++lane) {
        const uint32_t primaryWorker = lane;
        const uint32_t secondaryWorker =
            lane + TileXR::Demo::kAllToAllGroupSendCoreCount;
        CHECK_EQ(primaryWorker % TileXR::Demo::kAllToAllGroupSendCoreCount,
            secondaryWorker % TileXR::Demo::kAllToAllGroupSendCoreCount);
        const int32_t primaryPeer = TileXR::Demo::AllToAllGroupPeer(
            0, rankSize, 3U,
            primaryWorker % TileXR::Demo::kAllToAllGroupSendCoreCount);
        const int32_t secondaryPeer = TileXR::Demo::AllToAllGroupPeer(
            0, rankSize, 3U,
            secondaryWorker % TileXR::Demo::kAllToAllGroupSendCoreCount);
        CHECK_EQ(primaryPeer, secondaryPeer);
        CHECK_EQ(TileXR::Demo::AllToAllGroupCreditByteOffset(primaryPeer),
            TileXR::Demo::AllToAllGroupCreditByteOffset(secondaryPeer));
    }
}

void TestDualRoutePeerPolicy()
{
    using TileXR::Demo::AllToAllGroupIsCrossNode;
    CHECK_EQ(AllToAllGroupIsCrossNode(0, 7), false);
    CHECK_EQ(AllToAllGroupIsCrossNode(0, 8), true);
    CHECK_EQ(AllToAllGroupIsCrossNode(15, 8), false);
    CHECK_EQ(AllToAllGroupIsCrossNode(15, 0), true);
}

void TestDualRouteQpWeights()
{
    const uint32_t weighted[] = {6U, 6U, 6U, 6U, 2U, 2U, 2U, 2U};
    const auto split = TileXR::Demo::AllToAllGroupSelectRouteQps(weighted, 8U);
    CHECK_EQ(split.primaryQp, 0U);
    CHECK_EQ(split.secondaryQp, 4U);

    const uint32_t threeWeights[] = {2U, 6U, 4U, 6U};
    const auto distinct = TileXR::Demo::AllToAllGroupSelectRouteQps(threeWeights, 4U);
    CHECK_EQ(distinct.primaryQp, 1U);
    CHECK_EQ(distinct.secondaryQp, 2U);

    const uint32_t equal[] = {3U, 3U, 3U, 3U};
    const auto fallback = TileXR::Demo::AllToAllGroupSelectRouteQps(equal, 4U);
    CHECK_EQ(fallback.primaryQp, 0U);
    CHECK_EQ(fallback.secondaryQp, 0U);

    const auto empty = TileXR::Demo::AllToAllGroupSelectRouteQps(nullptr, 0U);
    CHECK_EQ(empty.primaryQp, 0U);
    CHECK_EQ(empty.secondaryQp, 0U);

    const auto automatic = TileXR::Demo::AllToAllGroupSplitByRoute(
        10U, 6U, 2U, TileXR::Demo::kAllToAllGroupAutoPrimaryParts);
    CHECK_EQ(automatic.primaryElements, 7U);
    CHECK_EQ(automatic.secondaryElements, 3U);
    const auto tiny = TileXR::Demo::AllToAllGroupSplitByRoute(
        1U, 6U, 2U, TileXR::Demo::kAllToAllGroupAutoPrimaryParts);
    CHECK_EQ(tiny.primaryElements, 0U);
    CHECK_EQ(tiny.secondaryElements, 1U);
    const auto overrideSplit = TileXR::Demo::AllToAllGroupSplitByRoute(
        10U, 6U, 2U, 5U);
    CHECK_EQ(overrideSplit.primaryElements, 6U);
    CHECK_EQ(overrideSplit.secondaryElements, 4U);
    const auto noSecondary = TileXR::Demo::AllToAllGroupSplitByRoute(
        10U, 6U, 0U, 5U);
    CHECK_EQ(noSecondary.primaryElements, 10U);
    CHECK_EQ(noSecondary.secondaryElements, 0U);

    const auto firstPrimary = TileXR::Demo::AllToAllGroupRouteSliceForPass(
        10U, 0U, 5U, automatic.primaryElements, 0U);
    const auto firstSecondary = TileXR::Demo::AllToAllGroupRouteSliceForPass(
        10U, 0U, 5U, automatic.primaryElements, 1U);
    const auto secondPrimary = TileXR::Demo::AllToAllGroupRouteSliceForPass(
        10U, 5U, 5U, automatic.primaryElements, 0U);
    const auto secondSecondary = TileXR::Demo::AllToAllGroupRouteSliceForPass(
        10U, 5U, 5U, automatic.primaryElements, 1U);
    CHECK_EQ(firstPrimary.elementOffset, 0U);
    CHECK_EQ(firstPrimary.elements, 5U);
    CHECK_EQ(firstSecondary.elements, 0U);
    CHECK_EQ(secondPrimary.elementOffset, 5U);
    CHECK_EQ(secondPrimary.elements, 2U);
    CHECK_EQ(secondSecondary.elementOffset, 7U);
    CHECK_EQ(secondSecondary.elements, 3U);
}

void TestRouteStages()
{
    using TileXR::Demo::AllToAllGroupRouteStage;
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidRouteStage(0U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidRouteStage(9U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidRouteStage(10U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidRouteStage(11U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsSend(
        AllToAllGroupRouteStage::kLocalSend), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsCopy(
        AllToAllGroupRouteStage::kLocalSend), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsSend(
        AllToAllGroupRouteStage::kLocalCopy), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsCopy(
        AllToAllGroupRouteStage::kLocalCopy), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageWaitsForSignal(
        AllToAllGroupRouteStage::kLocalCopy), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsSend(
        AllToAllGroupRouteStage::kRemoteSend), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsCopy(
        AllToAllGroupRouteStage::kRemoteSend), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsReceive(
        AllToAllGroupRouteStage::kAllSend), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsSend(
        AllToAllGroupRouteStage::kAllSend), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsReceive(
        AllToAllGroupRouteStage::kRemoteWait), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageWaitsForSignal(
        AllToAllGroupRouteStage::kRemoteWait), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsCopy(
        AllToAllGroupRouteStage::kRemoteWait), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageWaitsForSignal(
        AllToAllGroupRouteStage::kRemoteCopy), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsCopy(
        AllToAllGroupRouteStage::kRemoteCopy), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsSend(
        AllToAllGroupRouteStage::kNoCopy), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsReceive(
        AllToAllGroupRouteStage::kNoCopy), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageWaitsForSignal(
        AllToAllGroupRouteStage::kNoCopy), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupStageRunsCopy(
        AllToAllGroupRouteStage::kNoCopy), false);

    for (int rankSize : {8, 16, 128}) {
        for (int rank = 0; rank < rankSize; ++rank) {
            int local = 0;
            int primary = 0;
            int secondary = 0;
            for (int peer = 0; peer < rankSize; ++peer) {
                if (peer == rank) {
                    continue;
                }
                const bool inLocal = TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kLocal);
                const bool inPrimary = TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kPrimary);
                const bool inSecondary = TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kSecondary);
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kLocalSend), inLocal);
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kLocalCopy), inLocal);
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kRemoteSend),
                    inPrimary || inSecondary);
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kAllSend), true);
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kRemoteWait),
                    inPrimary || inSecondary);
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kRemoteCopy),
                    inPrimary || inSecondary);
                CHECK_EQ(TileXR::Demo::AllToAllGroupReceivePeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kRemoteCopy), true);
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kNoCopy), true);
                CHECK_EQ(static_cast<int>(inLocal) + static_cast<int>(inPrimary), 1);
                CHECK_EQ(inSecondary, inPrimary);
                local += inLocal ? 1 : 0;
                primary += inPrimary ? 1 : 0;
                secondary += inSecondary ? 1 : 0;
                CHECK_EQ(TileXR::Demo::AllToAllGroupPeerInRouteStage(
                    rank, peer, AllToAllGroupRouteStage::kCombined), true);
            }
            CHECK_EQ(local, 7);
            CHECK_EQ(primary, rankSize - 8);
            CHECK_EQ(secondary, rankSize - 8);
        }
    }
}

void TestCopyoutWorkerPolicy()
{
    CHECK_EQ(TileXR::Demo::kAllToAllGroupBlockDim, 64U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(1U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(8U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(16U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(32U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(48U), true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(4U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupValidCopyoutWorkers(12U), false);
    CHECK_EQ(TileXR::Demo::AllToAllGroupBlockDim(32U, 32U), 64U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupBlockDim(32U, 1U), 33U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupBlockDim(16U, 48U), 0U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupBlockDim(32U, 48U), 0U);

    std::set<int32_t> lanes;
    for (uint32_t worker = 0U; worker < 8U; ++worker) {
        for (uint32_t assignment = 0U; assignment < 2U; ++assignment) {
            lanes.insert(TileXR::Demo::AllToAllGroupCopyoutLane(
                worker, assignment, 8U));
        }
    }
    CHECK_EQ(lanes.size(), 16U);
    CHECK_EQ(*lanes.begin(), 0);
    CHECK_EQ(*lanes.rbegin(), 15);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(7U, 1U, 16U), -1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(15U, 0U, 16U), 15);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(8U, 0U, 8U), -1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(0U, 0U, 32U), 0);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(15U, 0U, 32U), 15);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(16U, 0U, 32U), 0);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(31U, 0U, 32U), 15);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(16U, 1U, 32U), -1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(0U, 0U, 48U), 0);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(16U, 0U, 48U), 0);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(32U, 0U, 48U), 0);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(47U, 0U, 48U), 15);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(32U, 1U, 48U), -1);

    lanes.clear();
    for (uint32_t assignment = 0U; assignment < 16U; ++assignment) {
        lanes.insert(TileXR::Demo::AllToAllGroupCopyoutLane(
            0U, assignment, 1U));
    }
    CHECK_EQ(lanes.size(), 16U);
    CHECK_EQ(*lanes.begin(), 0);
    CHECK_EQ(*lanes.rbegin(), 15);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(0U, 16U, 1U), -1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCopyoutLane(1U, 0U, 1U), -1);
}

void TestKernelStructure()
{
    const std::string kernel = ReadFile(
        std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp");
    const std::string launcher = ReadFile(
        std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_alltoall_group_launcher.cpp");
    CHECK_CONTAINS(kernel, "tilexr_udma_all_to_all_group_kernel");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_SEND_CORES");
    CHECK_CONTAINS(kernel, "#include \"tilexr_udma_alltoall_group_route.h\"");
    CHECK_CONTAINS(kernel, "AllToAllGroupSelectRouteQps");
    CHECK_CONTAINS(kernel, "AllToAllGroupSplitByRouteDevice");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_ROUTE_SIGNAL_STRIDE");
    CHECK_CONTAINS(kernel, "AllToAllGroupWaitRouteTokensMte");
    CHECK_CONTAINS(kernel, "secondaryQp");
    CHECK_CONTAINS(kernel, "selectedQp");
    CHECK_CONTAINS(kernel, "copyoutWorkers");
    CHECK_CONTAINS(kernel, "constexpr uint32_t copyoutWorkers = 1U");
    CHECK_CONTAINS(kernel, "uint32_t multiChannel, uint32_t primaryRouteParts");
    CHECK_CONTAINS(kernel, "AllToAllGroupPeerInRouteStageDevice");
    CHECK_CONTAINS(kernel,
        "if (!AllToAllGroupPeerInRouteStageDevice(rank, peer, routeStage))");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL");
    CHECK_CONTAINS(kernel, "AllToAllGroupCopyoutLaneDevice");
    CHECK_CONTAINS(kernel, "AllToAllGroupRemoteAssistDevice");
    CHECK_CONTAINS(kernel, "copySliceCount");
    CHECK_CONTAINS(kernel, "copySliceIndex");
    CHECK_CONTAINS(kernel, "#include \"tilexr_sdma.h\"");
    CHECK_CONTAINS(kernel, "AllToAllGroupCopySdma");
    CHECK_CONTAINS(kernel,
        "args, relayDst, relaySrc, copyBytes, worker, sdmaEvent");
    CHECK_CONTAINS(kernel, "event == 0ULL");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_SDMA_FALLBACK");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_SDMA_FAILED");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_STAGE_SDMA");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_SEND_WORKERS + copyoutWorkers");
    CHECK_CONTAINS(kernel,
        "const uint32_t traceCore = copyoutWorkers < TILEXR_ALLTOALL_GROUP_SEND_CORES");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_SEND_WORKERS + lane : blockIdx");
    CHECK_CONTAINS(kernel, "UDMAPutNbiOnQpWithFlag<int32_t>");
    CHECK_CONTAINS(kernel, "UDMAPutNbiOnQpWithFlag<uint64_t>");
    CHECK_NOT_CONTAINS(kernel, "UDMAPutSignalNbiOnQp<int32_t>");
    CHECK_CONTAINS(kernel, "AllToAllGroupSignalSourceSlot(quietState)");
    CHECK_CONTAINS(kernel,
        "AllToAllGroupSignalSourceSlot(\n    const AllToAllGroupQuietState<false>&)");
    CHECK_CONTAINS(kernel, "return state.pendingCount;");
    CHECK_CONTAINS(kernel, "AllToAllGroupPendingQuiet");
    CHECK_CONTAINS(kernel, "AllToAllGroupFlushQuiet");
    CHECK_CONTAINS(kernel,
        "UDMAQuietStatusOnQp(args, request.peer, request.qpIdx)");
    CHECK_CONTAINS(kernel, "state.pendingCount != quietBatch");
    CHECK_CONTAINS(kernel, "template <bool BatchQuiet, bool IngressCredit>");
    CHECK_CONTAINS(kernel, "struct AllToAllGroupQuietState<true>");
    CHECK_CONTAINS(kernel, "AllToAllGroupQuietState<BatchQuiet> quietState");
    CHECK_CONTAINS(kernel, "AllToAllGroupCompleteQuiet(");
    CHECK_CONTAINS(kernel, "AllToAllGroupFinishQuiet(");
    CHECK_CONTAINS(kernel,
        "UDMAQuietStatusOnQp(args, peer, selectedQp)");
    CHECK_CONTAINS(kernel, "tilexr_udma_all_to_all_group_batch_kernel");
    CHECK_CONTAINS(kernel, "tilexr_udma_all_to_all_group_credit_kernel");
    CHECK_CONTAINS(kernel, "tilexr_udma_all_to_all_group_batch_credit_kernel");
    CHECK_CONTAINS(kernel, "AllToAllGroupKernelImpl<false, false>");
    CHECK_CONTAINS(kernel, "AllToAllGroupKernelImpl<true, false>");
    CHECK_CONTAINS(kernel, "AllToAllGroupKernelImpl<false, true>");
    CHECK_CONTAINS(kernel, "AllToAllGroupKernelImpl<true, true>");
    CHECK_CONTAINS(kernel, "if constexpr (IngressCredit)");
    CHECK_CONTAINS(kernel, "uint32_t groupWidth, uint32_t quietBatch");
    CHECK_CONTAINS(kernel, "uint64_t creditOffset0, uint64_t creditOffset1");
    CHECK_CONTAINS(kernel, "uint32_t ingressWindow");
    CHECK_CONTAINS(kernel, "AllToAllGroupPublishNextCredit");
    CHECK_CONTAINS(kernel, "AllToAllGroupCreditOwnerDevice(worker)");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_STAGE_CREDIT_WAIT");
    CHECK_CONTAINS(kernel, "kAllToAllGroupTraceCreditWait");
    CHECK_CONTAINS(kernel, "TileXR::TILEXR_UDMA_SQE_FLAG_COMPLETION");
    const size_t publishCreditBegin = kernel.find(
        "__aicore__ inline void AllToAllGroupPublishNextCredit");
    const size_t publishCreditEnd = kernel.find(
        "__aicore__ inline void AllToAllGroupRecordError", publishCreditBegin);
    const std::string publishCredit = publishCreditBegin == std::string::npos ?
        std::string() : kernel.substr(publishCreditBegin,
            publishCreditEnd == std::string::npos ? std::string::npos :
                publishCreditEnd - publishCreditBegin);
    CHECK_CONTAINS(publishCredit, "args->creditMems[nextPeer]");
    CHECK_CONTAINS(publishCredit, "TILEXR_ALLTOALL_GROUP_CREDIT_WORDS");
    CHECK_CONTAINS(publishCredit, "AscendC::HardEvent::S_MTE3");
    CHECK_CONTAINS(publishCredit, "AscendC::HardEvent::MTE3_S");
    CHECK_CONTAINS(publishCredit, "AscendC::DataCopy(");
    CHECK_NOT_CONTAINS(publishCredit, "*remoteCredit = creditToken");
    CHECK_NOT_CONTAINS(publishCredit, "UDMAPutNbiOnQpWithFlag<uint64_t>");
    CHECK_NOT_CONTAINS(publishCredit, "UDMAQuiet");
    CHECK_CONTAINS(kernel, "AllToAllGroupWaitTokenMte");
    CHECK_CONTAINS(kernel, "AllToAllGroupWaitCreditMte");
    CHECK_CONTAINS(kernel, "AllToAllGroupStageRunsSendDevice(routeStage)");
    CHECK_CONTAINS(kernel, "AllToAllGroupStageRunsReceiveDevice(routeStage)");
    CHECK_CONTAINS(kernel, "AllToAllGroupReceivePeerInRouteStageDevice");
    CHECK_CONTAINS(kernel, "AllToAllGroupStageRunsCopyDevice(routeStage)");
    CHECK_CONTAINS(kernel, "AllToAllGroupStageWaitsForSignalDevice(routeStage)");
    CHECK_CONTAINS(kernel, "observed >= expectedToken");
    CHECK_NOT_CONTAINS(kernel, "<<<");
    CHECK_CONTAINS(launcher, "rtDevBinaryRegister");
    CHECK_CONTAINS(launcher, "rtFunctionRegister");
    CHECK_CONTAINS(launcher, "rtKernelLaunchWithFlagV2");
    CHECK_CONTAINS(launcher, "GroupedAllToAllKernelArgs");
    CHECK_CONTAINS(launcher, "sizeof(GroupedAllToAllKernelArgs) == 128U");
    CHECK_CONTAINS(launcher, "GroupedAllToAllCreditKernelArgs");
    CHECK_CONTAINS(launcher, "sizeof(GroupedAllToAllCreditKernelArgs) == 152U");
    CHECK_CONTAINS(launcher, "TILEXR_GROUPED_ALLTOALL_BATCH_KERNEL_NAME");
    CHECK_CONTAINS(launcher, "TILEXR_GROUPED_ALLTOALL_CREDIT_KERNEL_NAME");
    CHECK_CONTAINS(launcher, "TILEXR_GROUPED_ALLTOALL_BATCH_CREDIT_KERNEL_NAME");
    CHECK_CONTAINS(launcher, "const bool useCredit = ingressWindow != 0U");
    CHECK_CONTAINS(launcher, "const bool useBatch = quietBatch != 1U");
    CHECK_CONTAINS(launcher, "cfgInfo.schemMode = RT_SCHEM_MODE_NORMAL");
    CHECK_NOT_CONTAINS(launcher, "<<<");
    CHECK_NOT_CONTAINS(kernel, "UDMAPutSignalNbi<int32_t>");
    CHECK_NOT_CONTAINS(kernel, "SyncAll");
    CHECK_NOT_CONTAINS(kernel, "elementsPerPeer) * lane /");
}

void TestHostStructure()
{
    const std::string demo = ReadFile(
        std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    CHECK_CONTAINS(demo, "#include \"tilexr_udma_alltoall_group_layout.h\"");
    CHECK_CONTAINS(demo, "testType == 8");
    CHECK_CONTAINS(demo, "RunGroupedAllToAll");
    CHECK_CONTAINS(demo, "PlanAllToAllGroup");
    CHECK_CONTAINS(demo, "launch_tilexr_udma_all_to_all_group");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_USE_SECONDARY_ROUTE");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_CHANNEL_MODE");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_WIDTH");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_QUIET_BATCH");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_INGRESS_WINDOW");
    CHECK_CONTAINS(demo, "TILEXR_ENABLE_CREDIT_IPC=1");
    CHECK_CONTAINS(demo, "*commArgsHost, commArgsDev");
    CHECK_CONTAINS(demo, "grouped ingress credit currently requires single pass");
    CHECK_CONTAINS(demo, "grouped ingress credit currently requires groupWidth=16");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_PRIMARY_ROUTE_PARTS");
    CHECK_CONTAINS(demo, "kAllToAllGroupSendWorkerCount");
    CHECK_CONTAINS(demo, "grouped alltoall registeredBytes=");
    CHECK_CONTAINS(demo, "grouped alltoall warmup=");
    const size_t begin = demo.find("bool RunGroupedAllToAll(");
    const size_t end = demo.find("void Cleanup(", begin);
    const std::string grouped = begin == std::string::npos ? std::string() :
        demo.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    CHECK_CONTAINS(grouped,
        "\"local-send\", \"local-copy\", \"remote-send\", \"all-send\", "
        "\"remote-wait\"");
    CHECK_CONTAINS(grouped,
        "\"remote-copy\", \"no-copy\", \"primary\", \"secondary\", \"combined\"");
    CHECK_CONTAINS(grouped, "constexpr size_t kRouteStageCount = 10U");
    CHECK_CONTAINS(grouped, "AllToAllGroupRouteStage::kCombined");
    const size_t stageBatchBegin = grouped.find("auto runStageBatch");
    const size_t stageBatchEnd = grouped.find("for (size_t stageIndex", stageBatchBegin);
    const std::string stageBatch = stageBatchBegin == std::string::npos ? std::string() :
        grouped.substr(stageBatchBegin,
            stageBatchEnd == std::string::npos ? std::string::npos :
                stageBatchEnd - stageBatchBegin);
    CHECK_NOT_CONTAINS(stageBatch, "invocationId = 0U");
    CHECK_NOT_CONTAINS(stageBatch, "grouped stage iteration");
    CHECK_CONTAINS(stageBatch, "aclrtSynchronizeStream grouped stage warmup");
    CHECK_CONTAINS(stageBatch, "aclrtSynchronizeStream grouped stage measured");
    CHECK_CONTAINS(grouped, "aclrtEventElapsedTime");
    CHECK_CONTAINS(grouped, "DemoBarrierAll(rank, rankSize, barrierStep)");
    CHECK_CONTAINS(grouped, "auto runStageBatch");
    CHECK_NOT_CONTAINS(grouped, "\" warmup=\"");
    CHECK_CONTAINS(grouped, "\" complete\"");
    CHECK_CONTAINS(grouped,
        "DemoBarrierAll(rank, rankSize, \"grouped route stages ready\")");
    CHECK_CONTAINS(grouped,
        "DemoBarrierAll(rank, rankSize, \"grouped measured ready\")");
    const size_t warmupSync = grouped.find(
        "aclrtSynchronizeStream grouped warmup");
    const size_t measuredBarrier = grouped.find(
        "DemoBarrierAll(rank, rankSize, \"grouped measured ready\")");
    const size_t measuredBegin = grouped.find(
        "const auto begin = std::chrono::steady_clock::now()");
    CHECK_EQ(warmupSync < measuredBarrier && measuredBarrier < measuredBegin, true);
    CHECK_CONTAINS(demo, "\"/tilexr_group_trace_\" + stageName + \"_rank_\"");
}

} // namespace

int main()
{
    TestSchedules();
    TestPlan();
    TestChannelPolicy();
    TestScalePlanAndTraceCapacity();
    TestTokens();
    TestIngressCreditPolicy();
    TestDualRoutePeerPolicy();
    TestDualRouteQpWeights();
    TestRouteStages();
    TestCopyoutWorkerPolicy();
    TestKernelStructure();
    TestHostStructure();
    if (g_failures != 0) {
        std::cerr << "TileXR grouped all-to-all layout checks failed: " << g_failures << std::endl;
        return 1;
    }
    std::cout << "TileXR grouped all-to-all layout checks passed" << std::endl;
    return 0;
}
