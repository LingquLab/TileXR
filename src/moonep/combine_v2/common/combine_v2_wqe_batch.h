#ifndef TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_H
#define TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_H

#include <cstdint>

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE inline
#endif

namespace TileXRMoonEp {

constexpr uint32_t kMoonEpCombineV2PayloadBatchRows = 128U;
constexpr uint32_t kMoonEpCombineV2MaxSelectorThreads = 128U;
constexpr uint32_t kMoonEpCombineV2SelectorThreads =
    kMoonEpCombineV2PayloadBatchRows;
constexpr uint32_t kMoonEpCombineV2BuilderThreads =
    kMoonEpCombineV2PayloadBatchRows;
constexpr uint32_t kMoonEpCombineV2MaxSelectedPayloadWqes =
    2U * kMoonEpCombineV2PayloadBatchRows;
constexpr uint32_t kMoonEpCombineV2WqeBatchCapacity =
    kMoonEpCombineV2PayloadBatchRows;
constexpr uint32_t kMoonEpCombineV2BatchQpCount = 2U;
constexpr uint32_t kMoonEpCombineV2QpSplitPeriod = 4U;
constexpr uint32_t kMoonEpCombineV2SelfRelayHalfBytes = 64U * 1024U;
constexpr uint32_t kMoonEpCombineV2SelfMaxBatchRows = 8U;
constexpr uint32_t kMoonEpCombineV2SelfUbAlignment = 32U;

static_assert(kMoonEpCombineV2PayloadBatchRows > 0U &&
    kMoonEpCombineV2PayloadBatchRows <=
        kMoonEpCombineV2MaxSelectorThreads,
    "Combine V2 payload batch rows must be in [1, 128]");
static_assert(kMoonEpCombineV2SelectorThreads ==
        kMoonEpCombineV2BuilderThreads,
    "Combine V2 selector and builder widths must match");

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint64_t
MoonEpCombineV2SelfAlignedRowBytes(uint64_t rowBytes)
{
    if (rowBytes == 0U || rowBytes >
        UINT64_MAX - (kMoonEpCombineV2SelfUbAlignment - 1U)) {
        return 0U;
    }
    return (rowBytes + kMoonEpCombineV2SelfUbAlignment - 1U) /
        kMoonEpCombineV2SelfUbAlignment *
        kMoonEpCombineV2SelfUbAlignment;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t
MoonEpCombineV2SelfRowsPerBatch(uint64_t rowBytes)
{
    const uint64_t alignedRowBytes =
        MoonEpCombineV2SelfAlignedRowBytes(rowBytes);
    if (alignedRowBytes == 0U ||
        alignedRowBytes > kMoonEpCombineV2SelfRelayHalfBytes) {
        return 0U;
    }
    const uint32_t rows = static_cast<uint32_t>(
        kMoonEpCombineV2SelfRelayHalfBytes / alignedRowBytes);
    return rows < kMoonEpCombineV2SelfMaxBatchRows ?
        rows : kMoonEpCombineV2SelfMaxBatchRows;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint64_t
MoonEpCombineV2SelfTileCount(uint64_t rowBytes)
{
    if (rowBytes == 0U) {
        return 0U;
    }
    return rowBytes / kMoonEpCombineV2SelfRelayHalfBytes +
        (rowBytes % kMoonEpCombineV2SelfRelayHalfBytes == 0U ? 0U : 1U);
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t
MoonEpCombineV2SelfTileBytes(uint64_t rowBytes, uint64_t tileIndex)
{
    if (tileIndex >= MoonEpCombineV2SelfTileCount(rowBytes)) {
        return 0U;
    }
    const uint64_t tileOffset =
        tileIndex * kMoonEpCombineV2SelfRelayHalfBytes;
    const uint64_t remaining = rowBytes - tileOffset;
    return static_cast<uint32_t>(
        remaining < kMoonEpCombineV2SelfRelayHalfBytes ? remaining :
        kMoonEpCombineV2SelfRelayHalfBytes);
}

enum MoonEpCombineV2SingleCqeResult : uint32_t {
    MOONEP_COMBINE_V2_SINGLE_CQE_NO_COMPLETION = 0U,
    MOONEP_COMBINE_V2_SINGLE_CQE_PROGRESS = 1U,
    MOONEP_COMBINE_V2_SINGLE_CQE_TARGET_REACHED = 2U,
    MOONEP_COMBINE_V2_SINGLE_CQE_ERROR = 3U,
    MOONEP_COMBINE_V2_SINGLE_CQE_INVALID_STATE = 4U,
};

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t MoonEpCombineV2WqeBatchCount(
    uint64_t remainingWqes, uint32_t sqHead, uint32_t sqEntryCount)
{
    if (remainingWqes == 0U || sqEntryCount == 0U) {
        return 0U;
    }
    const uint32_t ringRemaining =
        sqEntryCount - sqHead % sqEntryCount;
    uint64_t batchCount = remainingWqes <
        static_cast<uint64_t>(kMoonEpCombineV2WqeBatchCapacity) ?
        remainingWqes :
        static_cast<uint64_t>(kMoonEpCombineV2WqeBatchCapacity);
    if (batchCount > static_cast<uint64_t>(ringRemaining)) {
        batchCount = ringRemaining;
    }
    return static_cast<uint32_t>(batchCount);
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t
MoonEpCombineV2SelectorFirstIndex(uint32_t chunkStart, uint32_t threadId)
{
    return chunkStart + threadId;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t
MoonEpCombineV2SelectorResumeIndex(uint32_t lastScannedIndex)
{
    return lastScannedIndex + kMoonEpCombineV2SelectorThreads;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE bool
MoonEpCombineV2SelectorIndexInChunk(uint32_t absoluteIndex,
    uint32_t chunkStart, uint32_t chunkElements)
{
    return absoluteIndex >= chunkStart &&
        static_cast<uint64_t>(absoluteIndex) <
            static_cast<uint64_t>(chunkStart) + chunkElements;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t MoonEpCombineV2CqePollBatchCount(
    uint32_t cqTail, uint32_t cqEntryCount, uint32_t batchCapacity)
{
    if (cqEntryCount == 0U || batchCapacity == 0U) {
        return 0U;
    }
    const uint32_t ringRemaining = cqEntryCount - cqTail % cqEntryCount;
    return ringRemaining < batchCapacity ? ringRemaining : batchCapacity;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE bool MoonEpCombineV2CqeOwnerReady(
    uint32_t cqTail, uint32_t cqEntryCount, uint32_t owner)
{
    if (cqEntryCount == 0U) {
        return false;
    }
    const uint32_t producerOwner = (cqTail / cqEntryCount) & 1U;
    return (owner & 1U) != producerOwner;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE bool MoonEpCombineV2CompletedSqTail(
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

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t MoonEpCombineV2AdvanceSingleCqe(
    uint32_t cqTail, uint32_t cqEntryCount, uint32_t owner,
    uint32_t status, uint32_t subStatus, uint32_t sqTail,
    uint32_t submittedSqTail, uint32_t cqeEntryIdx,
    uint32_t sqEntryCount, uint32_t &completedSqTail,
    uint32_t &detail)
{
    completedSqTail = sqTail;
    detail = 0U;
    if (cqEntryCount == 0U || sqEntryCount == 0U) {
        return MOONEP_COMBINE_V2_SINGLE_CQE_INVALID_STATE;
    }
    if (!MoonEpCombineV2CqeOwnerReady(
            cqTail, cqEntryCount, owner)) {
        return MOONEP_COMBINE_V2_SINGLE_CQE_NO_COMPLETION;
    }
    if (status != 0U || subStatus != 0U) {
        detail = ((status & 0xFFU) << 8U) | (subStatus & 0xFFU);
        return MOONEP_COMBINE_V2_SINGLE_CQE_ERROR;
    }
    const uint32_t outstanding = submittedSqTail - sqTail;
    if (!MoonEpCombineV2CompletedSqTail(
            sqTail, outstanding, cqeEntryIdx,
            sqEntryCount, completedSqTail)) {
        return MOONEP_COMBINE_V2_SINGLE_CQE_INVALID_STATE;
    }
    return completedSqTail == submittedSqTail ?
        MOONEP_COMBINE_V2_SINGLE_CQE_TARGET_REACHED :
        MOONEP_COMBINE_V2_SINGLE_CQE_PROGRESS;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t MoonEpCombineV2Qp1TokenCount(
    uint32_t tokenCount, uint32_t sequencePhase)
{
    const uint32_t firstQp1 =
        (3U - (sequencePhase & 3U)) & 3U;
    if (tokenCount <= firstQp1) {
        return 0U;
    }
    return 1U + (tokenCount - 1U - firstQp1) /
        kMoonEpCombineV2QpSplitPeriod;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t MoonEpCombineV2QpTokenCount(
    uint32_t tokenCount, uint32_t sequencePhase, uint32_t qpIdx)
{
    const uint32_t qp1Count = MoonEpCombineV2Qp1TokenCount(
        tokenCount, sequencePhase);
    if (qpIdx == 0U) {
        return tokenCount - qp1Count;
    }
    return qpIdx == 1U ? qp1Count : 0U;
}

TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE uint32_t MoonEpCombineV2QpSelectedIndex(
    uint32_t qpTokenIndex, uint32_t sequencePhase, uint32_t qpIdx)
{
    const uint32_t firstQp1 =
        (3U - (sequencePhase & 3U)) & 3U;
    if (qpIdx == 1U) {
        return firstQp1 + qpTokenIndex * kMoonEpCombineV2QpSplitPeriod;
    }
    if (qpIdx != 0U) {
        return UINT32_MAX;
    }
    if (qpTokenIndex < firstQp1) {
        return qpTokenIndex;
    }
    const uint32_t afterPrefix = qpTokenIndex - firstQp1;
    return firstQp1 + 1U + afterPrefix / 3U *
        kMoonEpCombineV2QpSplitPeriod + afterPrefix % 3U;
}

} // namespace TileXRMoonEp

#undef TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_INLINE

#endif // TILEXR_MOONEP_COMBINE_V2_WQE_BATCH_H
