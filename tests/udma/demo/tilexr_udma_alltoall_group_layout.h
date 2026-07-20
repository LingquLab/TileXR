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
constexpr int32_t kAllToAllGroupMaxRankSize = 128;
constexpr uint32_t kAllToAllGroupWidth = 16U;
constexpr uint32_t kAllToAllGroupHalfWidth = 8U;
constexpr uint32_t kAllToAllGroupPingPongSlots = 2U;
constexpr uint32_t kAllToAllGroupSignalSlotBytes = 128U;
constexpr uint32_t kAllToAllGroupSendCoreCount = 16U;
constexpr uint32_t kAllToAllGroupBlockDim = 32U;
constexpr size_t kAllToAllGroupAlignment = 512U;
constexpr size_t kAllToAllGroupControlBytes = 4096U;
constexpr size_t kAllToAllGroupMaxRegisteredBytes = 1ULL << 30;

struct AllToAllGroupPlan {
    bool valid = false;
    uint32_t groupCount = 0;
    uint32_t passCount = 0;
    int32_t chunkElements = 0;
    size_t bytesPerPeer = 0;
    size_t payloadPlaneBytes = 0;
    size_t payloadOffset[kAllToAllGroupPingPongSlots] = {0, 0};
    size_t signalPlaneBytes = 0;
    size_t signalOffset[kAllToAllGroupPingPongSlots] = {0, 0};
    size_t controlOffset = 0;
    size_t controlBytes = kAllToAllGroupControlBytes;
    size_t registeredBytes = 0;
};

inline bool AllToAllGroupValidRankSize(int rankSize)
{
    return rankSize >= kAllToAllGroupMinRankSize &&
        rankSize <= kAllToAllGroupMaxRankSize && rankSize % 8 == 0;
}

inline bool AllToAllGroupValidCopyoutWorkers(uint32_t workers)
{
    return workers == 8U || workers == 16U;
}

inline uint32_t AllToAllGroupBlockDim(uint32_t workers)
{
    return AllToAllGroupValidCopyoutWorkers(workers) ?
        kAllToAllGroupSendCoreCount + workers : 0U;
}

inline int32_t AllToAllGroupCopyoutLane(
    uint32_t worker, uint32_t assignment, uint32_t workers)
{
    if (!AllToAllGroupValidCopyoutWorkers(workers) || worker >= workers) {
        return -1;
    }
    const uint32_t lane = worker + assignment * workers;
    return lane < kAllToAllGroupWidth ? static_cast<int32_t>(lane) : -1;
}

inline uint32_t AllToAllGroupCount(int rankSize)
{
    if (!AllToAllGroupValidRankSize(rankSize)) {
        return 0U;
    }
    return static_cast<uint32_t>((rankSize - 1 + static_cast<int32_t>(kAllToAllGroupWidth) - 1) /
        static_cast<int32_t>(kAllToAllGroupWidth));
}

inline int32_t AllToAllGroupPeer(
    int rank, int rankSize, uint32_t group, uint32_t lane)
{
    if (!AllToAllGroupValidRankSize(rankSize) || rank < 0 || rank >= rankSize ||
        lane >= kAllToAllGroupWidth || group >= AllToAllGroupCount(rankSize)) {
        return -1;
    }
    const uint32_t index = lane < kAllToAllGroupHalfWidth ?
        lane : lane - kAllToAllGroupHalfWidth;
    const int32_t distance = static_cast<int32_t>(
        group * kAllToAllGroupHalfWidth + index + 1U);
    const int32_t diameter = rankSize / 2;
    if (distance > diameter ||
        (lane >= kAllToAllGroupHalfWidth && distance == diameter)) {
        return -1;
    }
    return lane < kAllToAllGroupHalfWidth ?
        (rank + distance) % rankSize :
        (rank - distance + rankSize) % rankSize;
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
    int rankSize, int32_t elementsPerPeer, int32_t chunkElements)
{
    AllToAllGroupPlan plan {};
    if (!AllToAllGroupValidRankSize(rankSize) || elementsPerPeer <= 0 || chunkElements <= 0) {
        return plan;
    }

    plan.groupCount = AllToAllGroupCount(rankSize);
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

    plan.payloadOffset[0] = 0U;
    if (!AllToAllGroupAlignUp(plan.payloadPlaneBytes, plan.payloadOffset[1])) {
        return AllToAllGroupPlan {};
    }

    size_t cursor = 0;
    if (!AllToAllGroupCheckedAdd(plan.payloadOffset[1], plan.payloadPlaneBytes, cursor) ||
        !AllToAllGroupAlignUp(cursor, plan.signalOffset[0]) ||
        !AllToAllGroupCheckedAdd(plan.signalOffset[0], plan.signalPlaneBytes, cursor) ||
        !AllToAllGroupAlignUp(cursor, plan.signalOffset[1]) ||
        !AllToAllGroupCheckedAdd(plan.signalOffset[1], plan.signalPlaneBytes, cursor) ||
        !AllToAllGroupAlignUp(cursor, plan.controlOffset) ||
        !AllToAllGroupCheckedAdd(plan.controlOffset, plan.controlBytes, cursor) ||
        !AllToAllGroupAlignUp(cursor, plan.registeredBytes)) {
        return AllToAllGroupPlan {};
    }
    if (plan.registeredBytes > kAllToAllGroupMaxRegisteredBytes) {
        return AllToAllGroupPlan {};
    }
    plan.valid = true;
    return plan;
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_GROUP_LAYOUT_H
