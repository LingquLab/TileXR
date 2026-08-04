/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_GROUP_LAYOUT_H
#define TILEXR_UDMA_ALLTOALL_GROUP_LAYOUT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace TileXR {
namespace Demo {

constexpr int32_t kAllToAllGroupMinRankSize = 8;
constexpr int32_t kAllToAllGroupMaxRankSize = 1024;
constexpr uint32_t kAllToAllGroupWidth = 16U;
constexpr uint32_t kAllToAllGroupHalfWidth = 8U;
constexpr uint32_t kAllToAllGroupExperimentalWidth = 4U;
constexpr uint32_t kAllToAllGroupMaxQuietBatch = 64U;
constexpr uint32_t kAllToAllGroupMaxIngressWindow = 1U;
constexpr uint32_t kAllToAllGroupPingPongSlots = 2U;
constexpr uint32_t kAllToAllGroupRouteSignalStride = 512U;
constexpr uint32_t kAllToAllGroupSignalSlotBytes = 1024U;
constexpr size_t kAllToAllGroupCreditStride = 512U;
constexpr size_t kAllToAllGroupCreditSlotBytes =
    static_cast<size_t>(kAllToAllGroupMaxRankSize) * kAllToAllGroupCreditStride;
constexpr uint32_t kAllToAllGroupSendCoreCount = 16U;
constexpr uint32_t kAllToAllGroupSendWorkerCount = 32U;
constexpr uint32_t kAllToAllGroupMaxGroupCount = 64U;
constexpr uint32_t kAllToAllGroupErrorWordsPerCore = 12U;
constexpr uint32_t kAllToAllGroupErrorCoreCount = 64U;
constexpr size_t kAllToAllGroupErrorBytes =
    static_cast<size_t>(kAllToAllGroupErrorWordsPerCore) *
    kAllToAllGroupErrorCoreCount * sizeof(uint32_t);
constexpr size_t kAllToAllGroupSignalSourceSlots =
    static_cast<size_t>(kAllToAllGroupSendWorkerCount) *
    kAllToAllGroupMaxQuietBatch;
constexpr size_t kAllToAllGroupSignalSourceBytes =
    kAllToAllGroupSignalSourceSlots * sizeof(uint64_t);
constexpr uint32_t kAllToAllGroupBlockDim = 64U;
constexpr size_t kAllToAllGroupMultiChannelThresholdBytes =
    150ULL * 1024ULL * 1024ULL;
constexpr size_t kAllToAllGroupAlignment = 512U;
constexpr size_t kAllToAllGroupBaseControlBytes =
    kAllToAllGroupErrorBytes + kAllToAllGroupSignalSourceBytes;
constexpr size_t kAllToAllGroupMaxPayloadBytes = 16ULL << 30;
constexpr size_t kAllToAllGroupMaxRegisteredBytes =
    2U * kAllToAllGroupMaxPayloadBytes +
    2U * static_cast<size_t>(kAllToAllGroupMaxRankSize) *
        kAllToAllGroupSignalSlotBytes +
    kAllToAllGroupBaseControlBytes;

struct AllToAllGroupPlan {
    bool valid = false;
    uint32_t groupWidth = kAllToAllGroupWidth;
    uint32_t groupCount = 0;
    uint32_t passCount = 0;
    int32_t chunkElements = 0;
    size_t bytesPerPeer = 0;
    size_t payloadPlaneBytes = 0;
    size_t payloadOffset[kAllToAllGroupPingPongSlots] = {0, 0};
    size_t signalPlaneBytes = 0;
    size_t signalOffset[kAllToAllGroupPingPongSlots] = {0, 0};
    size_t creditPlaneBytes = 0;
    size_t creditOffset[kAllToAllGroupPingPongSlots] = {0, 0};
    size_t controlOffset = 0;
    size_t controlBytes = kAllToAllGroupBaseControlBytes;
    size_t signalSourceOffset = 0;
    size_t signalSourceBytes = kAllToAllGroupSignalSourceBytes;
    size_t registeredBytes = 0;
};

inline bool AllToAllGroupValidWidth(uint32_t groupWidth)
{
    return groupWidth == kAllToAllGroupWidth ||
        groupWidth == kAllToAllGroupExperimentalWidth;
}

inline bool AllToAllGroupValidQuietBatch(uint32_t quietBatch)
{
    return quietBatch != 0U && quietBatch <= kAllToAllGroupMaxQuietBatch &&
        (quietBatch & (quietBatch - 1U)) == 0U;
}

inline bool AllToAllGroupValidIngressWindow(uint32_t ingressWindow)
{
    return ingressWindow <= kAllToAllGroupMaxIngressWindow;
}

enum class AllToAllGroupChannelMode : uint32_t {
    kAuto = 0U,
    kSingle = 1U,
    kMulti = 2U,
};

inline bool AllToAllGroupValidChannelMode(uint32_t mode)
{
    return mode <= static_cast<uint32_t>(AllToAllGroupChannelMode::kMulti);
}

inline bool AllToAllGroupUseMultiChannel(
    size_t perRankBytes, AllToAllGroupChannelMode mode)
{
    if (mode == AllToAllGroupChannelMode::kSingle) {
        return false;
    }
    if (mode == AllToAllGroupChannelMode::kMulti) {
        return true;
    }
    return perRankBytes > kAllToAllGroupMultiChannelThresholdBytes;
}

inline size_t AllToAllGroupSignalByteOffset(uint32_t sourceRank, uint32_t route)
{
    return static_cast<size_t>(sourceRank) * kAllToAllGroupSignalSlotBytes +
        static_cast<size_t>(route) * kAllToAllGroupRouteSignalStride;
}

inline size_t AllToAllGroupCreditByteOffset(uint32_t destinationRank)
{
    return static_cast<size_t>(destinationRank) * kAllToAllGroupCreditStride;
}

inline bool AllToAllGroupValidRankSize(int rankSize)
{
    return rankSize >= kAllToAllGroupMinRankSize &&
        rankSize <= kAllToAllGroupMaxRankSize && rankSize % 8 == 0;
}

inline bool AllToAllGroupValidCopyoutWorkers(uint32_t workers)
{
    return workers == 1U || workers == 8U || workers == 16U ||
        workers == 32U || workers == 48U;
}

inline uint32_t AllToAllGroupBlockDim(uint32_t sendWorkers, uint32_t copyoutWorkers)
{
    if (sendWorkers != kAllToAllGroupSendWorkerCount ||
        !AllToAllGroupValidCopyoutWorkers(copyoutWorkers) ||
        sendWorkers + copyoutWorkers > kAllToAllGroupBlockDim) {
        return 0U;
    }
    return sendWorkers + copyoutWorkers;
}

inline int32_t AllToAllGroupCopyoutLane(
    uint32_t worker, uint32_t assignment, uint32_t workers)
{
    if (!AllToAllGroupValidCopyoutWorkers(workers) || worker >= workers) {
        return -1;
    }
    if (workers >= 32U) {
        return assignment == 0U ?
            static_cast<int32_t>(worker % kAllToAllGroupWidth) : -1;
    }
    const uint32_t lane = worker + assignment * workers;
    return lane < kAllToAllGroupWidth ? static_cast<int32_t>(lane) : -1;
}

inline uint32_t AllToAllGroupCount(
    int rankSize, uint32_t groupWidth = kAllToAllGroupWidth)
{
    if (!AllToAllGroupValidRankSize(rankSize) ||
        !AllToAllGroupValidWidth(groupWidth)) {
        return 0U;
    }
    return static_cast<uint32_t>(
        (rankSize - 1 + static_cast<int32_t>(groupWidth) - 1) /
        static_cast<int32_t>(groupWidth));
}

inline int32_t AllToAllGroupPeer(
    int rank, int rankSize, uint32_t group, uint32_t lane,
    uint32_t groupWidth = kAllToAllGroupWidth)
{
    if (!AllToAllGroupValidRankSize(rankSize) || rank < 0 || rank >= rankSize ||
        !AllToAllGroupValidWidth(groupWidth) || lane >= groupWidth ||
        group >= AllToAllGroupCount(rankSize, groupWidth)) {
        return -1;
    }
    const uint32_t halfWidth = groupWidth / 2U;
    const uint32_t index = lane < halfWidth ? lane : lane - halfWidth;
    const int32_t distance = static_cast<int32_t>(
        group * halfWidth + index + 1U);
    const int32_t diameter = rankSize / 2;
    if (distance > diameter ||
        (lane >= halfWidth && distance == diameter)) {
        return -1;
    }
    return lane < halfWidth ?
        (rank + distance) % rankSize :
        (rank - distance + rankSize) % rankSize;
}

inline int32_t AllToAllGroupNextCreditPeer(
    int rank, int rankSize, uint32_t completedGroup, uint32_t lane,
    uint32_t groupWidth = kAllToAllGroupWidth)
{
    const uint32_t groupCount = AllToAllGroupCount(rankSize, groupWidth);
    if (groupCount == 0U || completedGroup + 1U >= groupCount) {
        return -1;
    }
    return AllToAllGroupPeer(
        rank, rankSize, completedGroup + 1U, lane, groupWidth);
}

inline int32_t AllToAllGroupTerminalCreditPeer(
    int rank, int rankSize, uint32_t lane,
    uint32_t groupWidth = kAllToAllGroupWidth)
{
    return AllToAllGroupPeer(rank, rankSize, 0U, lane, groupWidth);
}

inline bool AllToAllGroupCreditOwner(uint32_t copyoutWorker)
{
    return copyoutWorker < kAllToAllGroupSendCoreCount;
}

inline uint64_t AllToAllGroupToken(
    uint32_t invocationId, uint32_t group, uint32_t pass)
{
    const uint64_t invocation = static_cast<uint64_t>(invocationId) + 1ULL;
    const uint64_t slot = static_cast<uint64_t>(invocationId & 1U);
    return (invocation << 32U) | (slot << 31U) |
        (static_cast<uint64_t>(group) << 16U) |
        (static_cast<uint64_t>(pass) + 1ULL);
}

inline uint64_t AllToAllGroupCreditToken(
    uint32_t invocationId, uint32_t group)
{
    return AllToAllGroupToken(invocationId, group, 0U);
}

inline uint64_t AllToAllGroupTerminalCreditToken(
    uint32_t invocationId, uint32_t groupCount)
{
    return AllToAllGroupToken(invocationId, groupCount, 0U);
}

inline bool AllToAllGroupCheckedAdd(size_t lhs, size_t rhs, size_t& result)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

inline bool AllToAllGroupCheckedMul(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool AllToAllGroupAlignUp(size_t value, size_t& result)
{
    size_t expanded = 0;
    if (!AllToAllGroupCheckedAdd(value, kAllToAllGroupAlignment - 1U, expanded)) {
        return false;
    }
    result = expanded & ~(kAllToAllGroupAlignment - 1U);
    return true;
}

inline AllToAllGroupPlan PlanAllToAllGroup(
    int rankSize, int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t groupWidth = kAllToAllGroupWidth,
    uint32_t ingressWindow = 0U)
{
    AllToAllGroupPlan plan {};
    if (!AllToAllGroupValidRankSize(rankSize) ||
        !AllToAllGroupValidWidth(groupWidth) ||
        !AllToAllGroupValidIngressWindow(ingressWindow) ||
        (ingressWindow != 0U && groupWidth != kAllToAllGroupWidth) ||
        elementsPerPeer <= 0 || chunkElements <= 0) {
        return plan;
    }

    plan.groupWidth = groupWidth;
    plan.groupCount = AllToAllGroupCount(rankSize, groupWidth);
    plan.chunkElements = std::min(elementsPerPeer, chunkElements);
    plan.passCount = static_cast<uint32_t>(
        (static_cast<uint64_t>(elementsPerPeer) + static_cast<uint64_t>(plan.chunkElements) - 1ULL) /
        static_cast<uint64_t>(plan.chunkElements));
    if (plan.groupCount == 0U || plan.groupCount > 0x7FFFU ||
        plan.passCount == 0U || plan.passCount > 0xFFFFU) {
        return AllToAllGroupPlan {};
    }

    if (!AllToAllGroupCheckedMul(
            static_cast<size_t>(elementsPerPeer), sizeof(int32_t), plan.bytesPerPeer) ||
        !AllToAllGroupCheckedMul(
            static_cast<size_t>(rankSize), plan.bytesPerPeer, plan.payloadPlaneBytes) ||
        !AllToAllGroupCheckedMul(
            static_cast<size_t>(rankSize), kAllToAllGroupSignalSlotBytes, plan.signalPlaneBytes)) {
        return AllToAllGroupPlan {};
    }
    if (ingressWindow != 0U) {
        plan.creditPlaneBytes = static_cast<size_t>(rankSize) *
            kAllToAllGroupCreditStride;
        plan.creditOffset[0] = 0U;
        plan.creditOffset[1] = kAllToAllGroupCreditSlotBytes;
    }
    if (plan.payloadPlaneBytes > kAllToAllGroupMaxPayloadBytes) {
        return AllToAllGroupPlan {};
    }

    plan.payloadOffset[0] = 0U;
    if (!AllToAllGroupAlignUp(plan.payloadPlaneBytes, plan.payloadOffset[1])) {
        return AllToAllGroupPlan {};
    }

    size_t cursor = 0;
    if (!AllToAllGroupCheckedAdd(plan.payloadOffset[1], plan.payloadPlaneBytes, cursor) ||
        !AllToAllGroupAlignUp(cursor, plan.signalOffset[0]) ||
        !AllToAllGroupCheckedAdd(plan.signalOffset[0], plan.signalPlaneBytes, cursor) ||
        !AllToAllGroupAlignUp(cursor, plan.signalOffset[1]) ||
        !AllToAllGroupCheckedAdd(plan.signalOffset[1], plan.signalPlaneBytes, cursor)) {
        return AllToAllGroupPlan {};
    }
    if (!AllToAllGroupAlignUp(cursor, plan.controlOffset) ||
        !AllToAllGroupCheckedAdd(plan.controlOffset, plan.controlBytes, cursor) ||
        !AllToAllGroupAlignUp(cursor, plan.registeredBytes)) {
        return AllToAllGroupPlan {};
    }
    plan.signalSourceOffset = plan.controlOffset + kAllToAllGroupErrorBytes;
    if (plan.registeredBytes > kAllToAllGroupMaxRegisteredBytes) {
        return AllToAllGroupPlan {};
    }
    plan.valid = true;
    return plan;
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_GROUP_LAYOUT_H
