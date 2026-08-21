#ifndef TILEXR_MOONEP_DISPATCH_SCHEDULE_H
#define TILEXR_MOONEP_DISPATCH_SCHEDULE_H

#include <cstdint>

#include "dispatch_common.h"

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_MOONEP_SCHEDULE_INLINE __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_MOONEP_SCHEDULE_INLINE inline
#endif

namespace TileXRMoonEp {

constexpr int64_t kDispatchSplitRankSize = 128;
constexpr int64_t kDispatchSplitHalfRankSize = 64;

TILEXR_MOONEP_SCHEDULE_INLINE void DispatchContiguousRange(
    uint64_t itemCount, uint32_t partCount, uint32_t part,
    uint64_t &start, uint64_t &end)
{
    start = 0U;
    end = 0U;
    if (partCount == 0U || part >= partCount) {
        return;
    }
    const uint64_t itemsPerPart = itemCount / static_cast<uint64_t>(partCount);
    const uint64_t extraItems = itemCount % static_cast<uint64_t>(partCount);
    const uint64_t prefixExtra = static_cast<uint64_t>(part) < extraItems ?
        static_cast<uint64_t>(part) : extraItems;
    start = static_cast<uint64_t>(part) * itemsPerPart + prefixExtra;
    end = start + itemsPerPart +
        (static_cast<uint64_t>(part) < extraItems ? 1U : 0U);
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchGroupCount(int64_t rankSize)
{
    return rankSize <= 0 ? 0 : rankSize / kDispatchRankGroupSize +
        (rankSize % kDispatchRankGroupSize == 0 ? 0 : 1);
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchTargetGroup(
    int64_t rank, int64_t rankSize, int64_t phase)
{
    const int64_t groupCount = DispatchGroupCount(rankSize);
    if (rank < 0 || rank >= rankSize || phase < 0 || phase >= groupCount) {
        return -1;
    }
    return (rank / kDispatchRankGroupSize + phase) % groupCount;
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchPeerForCore(
    int64_t rank, int64_t rankSize, int64_t phase, uint32_t core,
    uint32_t coreCount, uint32_t workIndex)
{
    if (coreCount == 0U || coreCount > kDispatchAivCoreCount ||
        core >= coreCount) {
        return -1;
    }
    const uint64_t peerOffset = static_cast<uint64_t>(core) +
        static_cast<uint64_t>(workIndex) * coreCount;
    if (peerOffset >= static_cast<uint64_t>(kDispatchRankGroupSize)) {
        return -1;
    }
    const int64_t targetGroup = DispatchTargetGroup(rank, rankSize, phase);
    if (targetGroup < 0) {
        return -1;
    }
    const int64_t peer = targetGroup * kDispatchRankGroupSize +
        static_cast<int64_t>(peerOffset);
    return peer < rankSize ? peer : -1;
}

TILEXR_MOONEP_SCHEDULE_INLINE uint32_t DispatchGroupedGroupCount(
    int64_t rankSize, uint32_t groupWidth)
{
    if (rankSize <= 1 || !DispatchGroupWidthValid(groupWidth)) {
        return 0U;
    }
    const uint64_t remotePeerCount = static_cast<uint64_t>(rankSize - 1);
    return static_cast<uint32_t>(remotePeerCount / groupWidth +
        (remotePeerCount % groupWidth == 0U ? 0U : 1U));
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchSplit128RingPeer(
    int64_t halfBase, int64_t center, int64_t offset)
{
    const int64_t index =
        (center + offset + kDispatchSplitHalfRankSize) %
        kDispatchSplitHalfRankSize;
    return halfBase + index;
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchGroupedPeerSplit128(
    int64_t rank, uint32_t group, uint32_t lane)
{
    constexpr uint32_t splitGroupWidth = 16U;
    constexpr uint32_t halfGroupWidth = splitGroupWidth / 2U;
    constexpr uint32_t splitGroupCount = 8U;
    if (rank < 0 || rank >= kDispatchSplitRankSize ||
        group >= splitGroupCount || lane >= splitGroupWidth) {
        return -1;
    }

    const int64_t localHalfBase = rank < kDispatchSplitHalfRankSize ?
        0 : kDispatchSplitHalfRankSize;
    const int64_t remoteHalfBase =
        kDispatchSplitHalfRankSize - localHalfBase;
    const int64_t center = rank - localHalfBase;

    if (lane < halfGroupWidth) {
        const uint32_t ordinal = group * halfGroupWidth + lane;
        if (ordinal >= static_cast<uint32_t>(
                kDispatchSplitHalfRankSize - 1)) {
            return -1;
        }
        const int64_t distance = static_cast<int64_t>(ordinal / 2U + 1U);
        const int64_t offset = ordinal % 2U == 0U ? distance : -distance;
        return DispatchSplit128RingPeer(localHalfBase, center, offset);
    }

    const uint32_t remoteLane = lane - halfGroupWidth;
    int64_t offset = 0;
    if (group == 0U) {
        if (remoteLane == 1U) {
            offset = kDispatchSplitHalfRankSize / 2;
        } else if (remoteLane >= 2U) {
            const uint32_t ordinal = remoteLane - 2U;
            const int64_t distance = static_cast<int64_t>(ordinal / 2U + 1U);
            offset = ordinal % 2U == 0U ? distance : -distance;
        }
    } else {
        const int64_t distance = static_cast<int64_t>(
            group * (halfGroupWidth / 2U) + remoteLane / 2U);
        offset = remoteLane % 2U == 0U ? distance : -distance;
    }
    return DispatchSplit128RingPeer(remoteHalfBase, center, offset);
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchGroupedPeer(
    int64_t rank, int64_t rankSize, uint32_t group, uint32_t lane,
    uint32_t groupWidth)
{
    const uint32_t groupCount = DispatchGroupedGroupCount(rankSize, groupWidth);
    if (rank < 0 || rank >= rankSize || group >= groupCount ||
        lane >= groupWidth) {
        return -1;
    }
    if (rankSize == kDispatchSplitRankSize && groupWidth == 16U) {
        return DispatchGroupedPeerSplit128(rank, group, lane);
    }
    const uint32_t halfWidth = groupWidth / 2U;
    const uint32_t index = lane < halfWidth ? lane : lane - halfWidth;
    const uint64_t distance = static_cast<uint64_t>(group) * halfWidth +
        index + 1U;
    const uint64_t diameter = static_cast<uint64_t>(rankSize) / 2U;
    if (distance > diameter ||
        (lane >= halfWidth && distance == diameter && rankSize % 2 == 0)) {
        return -1;
    }
    if (lane < halfWidth) {
        return (rank + static_cast<int64_t>(distance)) % rankSize;
    }
    return (rank - static_cast<int64_t>(distance) + rankSize) % rankSize;
}

TILEXR_MOONEP_SCHEDULE_INLINE uint64_t DispatchGroupedAssignmentCount(
    int64_t rankSize, uint32_t groupWidth)
{
    return static_cast<uint64_t>(DispatchGroupedGroupCount(rankSize, groupWidth)) *
        groupWidth;
}

TILEXR_MOONEP_SCHEDULE_INLINE bool DispatchSameServer(
    int64_t rank, int64_t peer, int64_t localRankSize)
{
    return rank >= 0 && peer >= 0 && localRankSize > 0 &&
        rank / localRankSize == peer / localRankSize;
}

TILEXR_MOONEP_SCHEDULE_INLINE uint32_t DispatchFullmeshSlot(
    int64_t peer, int64_t localRankSize)
{
    return peer < 0 || localRankSize <= 0 ? UINT32_MAX :
        static_cast<uint32_t>(peer % localRankSize);
}

TILEXR_MOONEP_SCHEDULE_INLINE uint32_t DispatchCompletionLaneCount(
    int64_t rank, int64_t peer, int64_t localRankSize)
{
    return DispatchSameServer(rank, peer, localRankSize) ? 1U :
        2U;
}

TILEXR_MOONEP_SCHEDULE_INLINE uint32_t DispatchGroupedPeerWorkCount(
    int64_t rankSize, uint32_t groupWidth, uint32_t coreCount)
{
    if (coreCount == 0U || coreCount > kDispatchAivCoreCount) {
        return 0U;
    }
    const uint64_t assignments = DispatchGroupedAssignmentCount(
        rankSize, groupWidth);
    if (assignments == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>(assignments / coreCount +
        (assignments % coreCount == 0U ? 0U : 1U));
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchGroupedPeerForCore(
    int64_t rank, int64_t rankSize, uint32_t groupWidth, uint32_t core,
    uint32_t coreCount, uint32_t workIndex, uint32_t &group, uint32_t &lane)
{
    group = UINT32_MAX;
    lane = UINT32_MAX;
    if (coreCount == 0U || coreCount > kDispatchAivCoreCount ||
        core >= coreCount) {
        return -1;
    }
    const uint64_t assignment = static_cast<uint64_t>(core) +
        static_cast<uint64_t>(workIndex) * coreCount;
    if (assignment >= DispatchGroupedAssignmentCount(rankSize, groupWidth)) {
        return -1;
    }
    group = static_cast<uint32_t>(assignment / groupWidth);
    lane = static_cast<uint32_t>(assignment % groupWidth);
    return DispatchGroupedPeer(rank, rankSize, group, lane, groupWidth);
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchGroupedNextCreditPeer(
    int64_t rank, int64_t rankSize, uint32_t completedGroup, uint32_t lane,
    uint32_t groupWidth)
{
    if (completedGroup == UINT32_MAX) {
        return -1;
    }
    return DispatchGroupedPeer(
        rank, rankSize, completedGroup + 1U, lane, groupWidth);
}

TILEXR_MOONEP_SCHEDULE_INLINE uint32_t DispatchPeerWorkCount(uint32_t coreCount)
{
    if (coreCount == 0U || coreCount > kDispatchAivCoreCount) {
        return 0U;
    }
    const uint32_t groupSize = static_cast<uint32_t>(kDispatchRankGroupSize);
    return groupSize / coreCount + (groupSize % coreCount == 0U ? 0U : 1U);
}

TILEXR_MOONEP_SCHEDULE_INLINE int64_t DispatchPeerForCore(
    int64_t rank, int64_t rankSize, int64_t phase, uint32_t core)
{
    return DispatchPeerForCore(rank, rankSize, phase, core,
        kDispatchAivCoreCount, 0U);
}

} // namespace TileXRMoonEp

#undef TILEXR_MOONEP_SCHEDULE_INLINE

#endif // TILEXR_MOONEP_DISPATCH_SCHEDULE_H
