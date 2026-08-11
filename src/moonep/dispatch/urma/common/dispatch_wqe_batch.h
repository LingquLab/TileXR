#ifndef TILEXR_MOONEP_DISPATCH_WQE_BATCH_H
#define TILEXR_MOONEP_DISPATCH_WQE_BATCH_H

#include <cstdint>

#include "dispatch_common.h"

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_MOONEP_WQE_BATCH_INLINE __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_MOONEP_WQE_BATCH_INLINE inline
#endif

namespace TileXRMoonEp {

constexpr uint32_t kDispatchWqeBatchCapacity = 128U;
constexpr uint32_t kDispatchLogicalWqeBatchCapacity =
    2U * kDispatchWqeBatchCapacity;
constexpr uint32_t kDispatchQpCount = 2U;
constexpr uint32_t kDispatchQpSplitPeriod = 4U;
constexpr uint32_t kDispatchSqPollReserve = 10U;
constexpr uint32_t kDispatchSharedQpCoreCount = 16U;
constexpr uint32_t kDispatchSharedQpCount =
    kDispatchQpCount * kDispatchSharedQpCoreCount;

TILEXR_MOONEP_WQE_BATCH_INLINE bool DispatchQpCountSupported(
    uint32_t availableQpCount, bool sharedQp = false)
{
    return sharedQp ? availableQpCount == kDispatchSharedQpCount :
        availableQpCount >= kDispatchQpCount;
}

TILEXR_MOONEP_WQE_BATCH_INLINE uint32_t DispatchPeerCoreCount(
    uint32_t launchedCoreCount, bool sharedQp)
{
    if (launchedCoreCount == 0U ||
        launchedCoreCount > kDispatchAivCoreCount) {
        return 0U;
    }
    return sharedQp && launchedCoreCount > kDispatchSharedQpCoreCount ?
        kDispatchSharedQpCoreCount : launchedCoreCount;
}

TILEXR_MOONEP_WQE_BATCH_INLINE uint32_t DispatchPhysicalQpIndex(
    uint32_t logicalQpIdx, uint32_t coreIdx, bool sharedQp)
{
    if (logicalQpIdx >= kDispatchQpCount) {
        return UINT32_MAX;
    }
    if (!sharedQp) {
        return logicalQpIdx;
    }
    if (coreIdx >= kDispatchSharedQpCoreCount) {
        return UINT32_MAX;
    }
    return logicalQpIdx * kDispatchSharedQpCoreCount + coreIdx;
}

TILEXR_MOONEP_WQE_BATCH_INLINE uint32_t DispatchWqeBatchCount(
    uint64_t remainingWqes, uint32_t sqHead, uint32_t sqEntryCount)
{
    if (remainingWqes == 0U || sqEntryCount == 0U) {
        return 0U;
    }
    const uint32_t ringRemaining = sqEntryCount - sqHead % sqEntryCount;
    uint64_t count = remainingWqes <
        static_cast<uint64_t>(kDispatchWqeBatchCapacity) ? remainingWqes :
        static_cast<uint64_t>(kDispatchWqeBatchCapacity);
    if (count > static_cast<uint64_t>(ringRemaining)) {
        count = ringRemaining;
    }
    return static_cast<uint32_t>(count);
}

TILEXR_MOONEP_WQE_BATCH_INLINE uint32_t DispatchCqePollBatchCount(
    uint32_t cqTail, uint32_t cqEntryCount, uint32_t batchCapacity)
{
    if (cqEntryCount == 0U || batchCapacity == 0U) {
        return 0U;
    }
    const uint32_t ringRemaining = cqEntryCount - cqTail % cqEntryCount;
    return ringRemaining < batchCapacity ? ringRemaining : batchCapacity;
}

TILEXR_MOONEP_WQE_BATCH_INLINE bool DispatchCqeOwnerReady(
    uint32_t cqTail, uint32_t cqEntryCount, uint32_t owner)
{
    if (cqEntryCount == 0U) {
        return false;
    }
    const uint32_t producerOwner = (cqTail / cqEntryCount) & 1U;
    return (owner & 1U) != producerOwner;
}

TILEXR_MOONEP_WQE_BATCH_INLINE bool DispatchCompletedSqTail(
    uint32_t sqTail, uint32_t outstanding, uint32_t cqeEntryIdx,
    uint32_t sqEntryCount, uint32_t &completedSqTail)
{
    if (sqEntryCount == 0U || outstanding == 0U ||
        outstanding >= sqEntryCount) {
        return false;
    }
    const uint32_t completedRingTail =
        (cqeEntryIdx % sqEntryCount + 1U) % sqEntryCount;
    const uint32_t currentRingTail = sqTail % sqEntryCount;
    const uint32_t advance =
        (completedRingTail + sqEntryCount - currentRingTail) % sqEntryCount;
    if (advance == 0U || advance > outstanding) {
        return false;
    }
    completedSqTail = sqTail + advance;
    return true;
}

TILEXR_MOONEP_WQE_BATCH_INLINE bool DispatchSqTailIsFurther(
    uint32_t sqTail, uint32_t candidateSqTail, uint32_t completedSqTail)
{
    return static_cast<uint32_t>(candidateSqTail - sqTail) >
        static_cast<uint32_t>(completedSqTail - sqTail);
}

TILEXR_MOONEP_WQE_BATCH_INLINE uint32_t DispatchQp1RouteCount(
    uint32_t routeCount, uint32_t sequencePhase)
{
    const uint32_t firstQp1 = (3U - (sequencePhase & 3U)) & 3U;
    if (routeCount <= firstQp1) {
        return 0U;
    }
    return 1U + (routeCount - 1U - firstQp1) / kDispatchQpSplitPeriod;
}

TILEXR_MOONEP_WQE_BATCH_INLINE uint32_t DispatchQpRouteCount(
    uint32_t routeCount, uint32_t sequencePhase, uint32_t qpIdx)
{
    const uint32_t qp1Count = DispatchQp1RouteCount(routeCount, sequencePhase);
    if (qpIdx == 0U) {
        return routeCount - qp1Count;
    }
    return qpIdx == 1U ? qp1Count : 0U;
}

TILEXR_MOONEP_WQE_BATCH_INLINE uint32_t DispatchQpSelectedIndex(
    uint32_t qpRouteIndex, uint32_t sequencePhase, uint32_t qpIdx)
{
    const uint32_t firstQp1 = (3U - (sequencePhase & 3U)) & 3U;
    if (qpIdx == 1U) {
        return firstQp1 + qpRouteIndex * kDispatchQpSplitPeriod;
    }
    if (qpIdx != 0U) {
        return UINT32_MAX;
    }
    if (qpRouteIndex < firstQp1) {
        return qpRouteIndex;
    }
    const uint32_t afterPrefix = qpRouteIndex - firstQp1;
    return firstQp1 + 1U + afterPrefix / 3U * kDispatchQpSplitPeriod +
        afterPrefix % 3U;
}

TILEXR_MOONEP_WQE_BATCH_INLINE bool DispatchPeerWqesFitSq(
    uint64_t routeCount, uint32_t sqEntryCount,
    uint32_t reserve = kDispatchSqPollReserve)
{
    if (routeCount > UINT32_MAX || sqEntryCount <= reserve) {
        return false;
    }
    const uint32_t count = static_cast<uint32_t>(routeCount);
    const uint32_t available = sqEntryCount - reserve;
    for (uint32_t qpIdx = 0U; qpIdx < kDispatchQpCount; ++qpIdx) {
        const uint32_t qpPayloadWqes = DispatchQpRouteCount(
            count, 0U, qpIdx);
        if (qpPayloadWqes >= available || qpPayloadWqes + 1U > available) {
            return false;
        }
    }
    return true;
}

TILEXR_MOONEP_WQE_BATCH_INLINE bool DispatchBatchNeedsCompletion(
    bool finalBatchForPeer)
{
    return finalBatchForPeer;
}

} // namespace TileXRMoonEp

#undef TILEXR_MOONEP_WQE_BATCH_INLINE

#endif // TILEXR_MOONEP_DISPATCH_WQE_BATCH_H
