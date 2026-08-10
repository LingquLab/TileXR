#ifndef TILEXR_MOONEP_COMBINE_V2_SCHEDULE_H
#define TILEXR_MOONEP_COMBINE_V2_SCHEDULE_H

#include <cstdint>
#include <limits>

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_MOONEP_COMBINE_V2_INLINE \
    __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_MOONEP_COMBINE_V2_INLINE inline constexpr
#endif

namespace TileXRMoonEp {

constexpr uint32_t kMoonEpCombineV2RankCount = 128U;
constexpr uint32_t kMoonEpCombineV2GroupSize = 8U;
constexpr uint32_t kMoonEpCombineV2GroupCount = 16U;
constexpr uint32_t kMoonEpCombineV2GroupsPerHalf = 8U;
constexpr uint32_t kMoonEpCombineV2StepCount = 8U;
constexpr uint32_t kMoonEpCombineV2GrantStepCount =
    kMoonEpCombineV2StepCount - 1U;
constexpr uint32_t kMoonEpCombineV2CoreCount = 16U;
constexpr uint32_t kMoonEpCombineV2LaneCount = 2U;
constexpr uint32_t kMoonEpCombineV2QpCount = 32U;
constexpr uint32_t kMoonEpCombineV2MaxSourcesPerCore =
    kMoonEpCombineV2RankCount / kMoonEpCombineV2CoreCount;
constexpr uint32_t kMoonEpCombineV2EpochCount = 2U;
constexpr uint32_t kMoonEpCombineV2LogicalBatchRows = 128U;
constexpr uint32_t kMoonEpCombineV2SixPortRows = 96U;
constexpr uint32_t kMoonEpCombineV2TwoPortRows = 32U;
constexpr uint32_t kMoonEpCombineV2SelectionChunkRows = 16384U;
constexpr uint32_t kMoonEpCombineV2MaxOutstanding = 16384U;
constexpr int64_t kMoonEpCombineV2SmallBs = 8;
constexpr int64_t kMoonEpCombineV2SmallSlots = 128;
constexpr int64_t kMoonEpCombineV2TargetBs = 8192;
constexpr int64_t kMoonEpCombineV2TargetH = 3584;
constexpr int64_t kMoonEpCombineV2TargetTopK = 16;
constexpr int64_t kMoonEpCombineV2TargetSlots = 131072;
constexpr uint64_t kMoonEpCombineV2TokenStrideBytes = 64U;
constexpr uint64_t kMoonEpCombineV2GrantSlotBytes = 512U;
constexpr uint64_t kMoonEpCombineV2GrantReceiveOffsetBytes = 0U;
constexpr uint64_t kMoonEpCombineV2GrantSourceOffsetBytes = 64U;
constexpr uint32_t kMoonEpCombineV2FailureMarker = 0x47505632U;
constexpr uint64_t kMoonEpCombineV2MaxMagic =
    std::numeric_limits<uint64_t>::max() >> 3U;

enum MoonEpCombineV2Lane : uint32_t {
    MOONEP_COMBINE_V2_SIX_PORT = 0U,
    MOONEP_COMBINE_V2_TWO_PORT = 1U,
};

enum MoonEpCombineV2FailureStatus : uint32_t {
    MOONEP_COMBINE_V2_SUCCESS = 0U,
    MOONEP_COMBINE_V2_INVALID_CONFIG = 1U,
    MOONEP_COMBINE_V2_POISONED = 2U,
    MOONEP_COMBINE_V2_OUTSTANDING_LIMIT = 3U,
    MOONEP_COMBINE_V2_CQ_TIMEOUT = 4U,
    MOONEP_COMBINE_V2_CQ_ERROR = 5U,
    MOONEP_COMBINE_V2_GRANT_TIMEOUT = 6U,
    MOONEP_COMBINE_V2_DONE_TIMEOUT = 7U,
    MOONEP_COMBINE_V2_BAD_DESTINATION = 8U,
};

struct alignas(64) MoonEpCombineV2FailureRecord {
    uint64_t magic;
    uint32_t status;
    uint32_t rank;
    uint32_t core;
    uint32_t step;
    uint32_t peer;
    uint32_t lane;
    uint32_t qp;
    uint32_t cqStatus;
    uint64_t expected;
    uint64_t observed;
    uint32_t poison;
    uint32_t marker;
};

struct MoonEpCombineV2LaneCounts {
    uint32_t sixPort;
    uint32_t twoPort;
};

struct MoonEpCombineV2RingSegments {
    uint32_t first;
    uint32_t second;
};

static_assert(sizeof(MoonEpCombineV2FailureRecord) == 64U,
    "MoonEP Combine V2 failure record ABI changed");
static_assert(kMoonEpCombineV2GrantSourceOffsetBytes + sizeof(uint64_t) <=
        kMoonEpCombineV2GrantSlotBytes,
    "MoonEP Combine V2 Grant source exceeds its slot");

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2RankSizeSupported(uint32_t rankSize)
{
    return (rankSize >= 2U && rankSize <= kMoonEpCombineV2GroupSize) ||
        rankSize == 16U || rankSize == 32U || rankSize == 64U ||
        rankSize == kMoonEpCombineV2RankCount;
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2RankValid(uint32_t rank, uint32_t rankSize)
{
    return MoonEpCombineV2RankSizeSupported(rankSize) && rank < rankSize;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2ActiveCoreCount(uint32_t rankSize)
{
    return rankSize <= kMoonEpCombineV2GroupSize ? rankSize :
        kMoonEpCombineV2CoreCount;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2StepCount(uint32_t rankSize)
{
    return rankSize <= kMoonEpCombineV2GroupSize ? 1U :
        rankSize / kMoonEpCombineV2CoreCount;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2GroupsPerHalf(uint32_t rankSize)
{
    return rankSize / kMoonEpCombineV2CoreCount;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2LocalRankSize(uint32_t rankSize)
{
    return rankSize <= kMoonEpCombineV2GroupSize ? rankSize :
        kMoonEpCombineV2GroupSize;
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2CoreValid(uint32_t core, uint32_t rankSize)
{
    return MoonEpCombineV2RankSizeSupported(rankSize) &&
        core < MoonEpCombineV2ActiveCoreCount(rankSize);
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2StepValid(uint32_t step, uint32_t rankSize)
{
    return MoonEpCombineV2RankSizeSupported(rankSize) &&
        step < MoonEpCombineV2StepCount(rankSize);
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2MagicValid(uint64_t magic)
{
    return magic != 0U && magic <= kMoonEpCombineV2MaxMagic;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2Epoch(uint64_t magic)
{
    return static_cast<uint32_t>(magic & 1U);
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2Qp(uint32_t core, uint32_t lane)
{
    return lane == MOONEP_COMBINE_V2_SIX_PORT ? core :
        kMoonEpCombineV2CoreCount + core;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2LaneForPosition(uint32_t position)
{
    return (position & 3U) == 3U ? MOONEP_COMBINE_V2_TWO_PORT :
        MOONEP_COMBINE_V2_SIX_PORT;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2ControlWqesPerLane(
    uint32_t step, uint32_t stepCount, bool finalBatch)
{
    return !finalBatch ? 0U :
        (step + 1U < stepCount ? 2U : 1U);
}

TILEXR_MOONEP_COMBINE_V2_INLINE MoonEpCombineV2LaneCounts
MoonEpCombineV2BatchLaneCounts(uint32_t logicalRows,
    uint32_t sequencePhase, uint32_t step, uint32_t stepCount,
    bool finalBatch)
{
    const uint32_t firstTwoPort =
        (3U - (sequencePhase & 3U)) & 3U;
    const uint32_t twoPort = logicalRows <= firstTwoPort ? 0U :
        1U + (logicalRows - 1U - firstTwoPort) / 4U;
    const uint32_t control =
        MoonEpCombineV2ControlWqesPerLane(step, stepCount, finalBatch);
    return MoonEpCombineV2LaneCounts {
        logicalRows - twoPort + control, twoPort + control};
}

TILEXR_MOONEP_COMBINE_V2_INLINE MoonEpCombineV2RingSegments
MoonEpCombineV2SplitRingCopy(uint32_t absoluteHead,
    uint32_t count, uint32_t ringEntries)
{
    const uint32_t ringIndex = ringEntries == 0U ? 0U :
        absoluteHead % ringEntries;
    const uint32_t untilEnd = ringEntries - ringIndex;
    const uint32_t first = count < untilEnd ? count : untilEnd;
    return ringEntries == 0U ? MoonEpCombineV2RingSegments {0U, 0U} :
        MoonEpCombineV2RingSegments {first, count - first};
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2NextCqTarget(
    uint32_t cqTail, bool finalBatch)
{
    return finalBatch ? cqTail + 1U : cqTail;
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2CqTargetReached(
    uint32_t cqTail, uint32_t cqTarget)
{
    return cqTail == cqTarget;
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2ShapeValid(
    int64_t bs, int64_t h, int64_t topK, int64_t nvS)
{
    return h == kMoonEpCombineV2TargetH &&
        topK == kMoonEpCombineV2TargetTopK &&
        ((bs == kMoonEpCombineV2SmallBs &&
            nvS == kMoonEpCombineV2SmallSlots) ||
        (bs == kMoonEpCombineV2TargetBs &&
            nvS == kMoonEpCombineV2TargetSlots));
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2DestinationValid(
    int32_t encodedDestination, uint64_t slots, uint32_t rankSize)
{
    return MoonEpCombineV2RankSizeSupported(rankSize) &&
        encodedDestination >= 0 &&
        (slots == static_cast<uint64_t>(
            kMoonEpCombineV2SmallSlots) ||
        slots == static_cast<uint64_t>(
            kMoonEpCombineV2TargetSlots)) &&
        static_cast<uint64_t>(encodedDestination) <
            static_cast<uint64_t>(rankSize) * slots;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2Peer(
    uint32_t sourceRank, uint32_t step, uint32_t core, uint32_t rankSize)
{
    if (rankSize <= kMoonEpCombineV2GroupSize) {
        return core;
    }
    const uint32_t groupsPerHalf = MoonEpCombineV2GroupsPerHalf(rankSize);
    const uint32_t sourceGroup =
        (sourceRank / kMoonEpCombineV2GroupSize) %
        groupsPerHalf;
    const uint32_t targetHalf = core / kMoonEpCombineV2GroupSize;
    const uint32_t targetOffset = core % kMoonEpCombineV2GroupSize;
    const uint32_t distance = (step + 1U) % groupsPerHalf;
    const uint32_t targetIndex = core < kMoonEpCombineV2GroupSize ?
        (sourceGroup + distance) % groupsPerHalf :
        (sourceGroup + groupsPerHalf - distance) % groupsPerHalf;
    return (targetIndex + targetHalf * groupsPerHalf) *
        kMoonEpCombineV2GroupSize + targetOffset;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2Successor(
    uint32_t sourceRank, uint32_t core, uint32_t rankSize)
{
    if (rankSize <= kMoonEpCombineV2GroupSize) {
        return sourceRank;
    }
    const uint32_t halfRankCount = rankSize / 2U;
    const uint32_t groupsPerHalf = MoonEpCombineV2GroupsPerHalf(rankSize);
    const uint32_t halfBase = sourceRank / halfRankCount * halfRankCount;
    const uint32_t groupInHalf = (sourceRank % halfRankCount) /
        kMoonEpCombineV2GroupSize;
    const uint32_t successorGroup = core < kMoonEpCombineV2GroupSize ?
        (groupInHalf + groupsPerHalf - 1U) % groupsPerHalf :
        (groupInHalf + 1U) % groupsPerHalf;
    return halfBase + successorGroup * kMoonEpCombineV2GroupSize +
        sourceRank % kMoonEpCombineV2GroupSize;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2ReceiveStep(
    uint32_t destinationRank, uint32_t sourceRank, uint32_t rankSize)
{
    if (rankSize <= kMoonEpCombineV2GroupSize) {
        return 0U;
    }
    const uint32_t groupsPerHalf = MoonEpCombineV2GroupsPerHalf(rankSize);
    const uint32_t halfRankCount = rankSize / 2U;
    const uint32_t destinationIndex =
        (destinationRank / kMoonEpCombineV2GroupSize) %
        groupsPerHalf;
    const uint32_t sourceIndex =
        (sourceRank / kMoonEpCombineV2GroupSize) %
        groupsPerHalf;
    const uint32_t delta = destinationRank < halfRankCount ?
        (destinationIndex + groupsPerHalf - sourceIndex) % groupsPerHalf :
        (sourceIndex + groupsPerHalf - destinationIndex) % groupsPerHalf;
    const uint32_t distance = delta == 0U ? groupsPerHalf : delta;
    return distance - 1U;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint32_t
MoonEpCombineV2SourceForCore(
    uint32_t core, uint32_t sourceIndex, uint32_t rankSize)
{
    return core + sourceIndex * MoonEpCombineV2ActiveCoreCount(rankSize);
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint64_t
MoonEpCombineV2Token(
    uint64_t magic, uint32_t step)
{
    return (magic << 3U) | static_cast<uint64_t>(step);
}

TILEXR_MOONEP_COMBINE_V2_INLINE bool
MoonEpCombineV2TokenMatches(
    uint64_t token, uint64_t magic, uint32_t step, uint32_t rankSize)
{
    return MoonEpCombineV2MagicValid(magic) &&
        MoonEpCombineV2StepValid(step, rankSize) &&
        token == MoonEpCombineV2Token(magic, step);
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint64_t
MoonEpCombineV2DoneIndex(
    uint32_t epoch, uint32_t sourceRank, uint32_t lane)
{
    return (static_cast<uint64_t>(epoch) * kMoonEpCombineV2RankCount +
        sourceRank) * kMoonEpCombineV2LaneCount + lane;
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint64_t
MoonEpCombineV2GrantIndex(
    uint32_t epoch, uint32_t core, uint32_t lane, uint32_t step)
{
    return (((static_cast<uint64_t>(epoch) * kMoonEpCombineV2CoreCount +
        core) * kMoonEpCombineV2LaneCount + lane) *
        kMoonEpCombineV2GrantStepCount) + (step - 1U);
}

TILEXR_MOONEP_COMBINE_V2_INLINE uint64_t
MoonEpCombineV2FailureIndex(
    uint32_t epoch, uint32_t core)
{
    return static_cast<uint64_t>(epoch) * kMoonEpCombineV2CoreCount + core;
}

} // namespace TileXRMoonEp

#undef TILEXR_MOONEP_COMBINE_V2_INLINE

#endif // TILEXR_MOONEP_COMBINE_V2_SCHEDULE_H
