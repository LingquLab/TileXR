#ifndef TILEXR_EP_PLANNER_COMMON_EP_PLAN_TYPES_H
#define TILEXR_EP_PLANNER_COMMON_EP_PLAN_TYPES_H

#include <cstdint>

namespace TileXREp {
namespace Plan {

constexpr int32_t kPlanAbiVersion = 1;
constexpr int32_t kPlanStatusWords = 8;
constexpr int32_t kPlanCardsPerServer = 8;
constexpr int32_t kPlanCardsPerCabinet = 64;
constexpr int32_t kPlanCrossCandidateCount = 3;
constexpr uint64_t kPlanAffinityCacheValid = 1ULL;
constexpr uint64_t kPlanWorkspaceAlignment = 64;
constexpr uint64_t kPlanHeaderStrideBytes = 128;
constexpr uint64_t kPlanStatusStrideBytes = 64;
constexpr uint64_t kPlanBarrierSlotBytes = 64;
constexpr uint64_t kPlanMaxUdmaTransferBytes = 0xFFFFFFFFULL;

struct PlanCallHeader {
    int32_t abiVersion;
    int32_t headerBytes;
    int32_t rankSize;
    int32_t reserved0;
    int64_t s;
    int64_t k;
    int64_t expertNum;
    int64_t prefetchSlots;
    int64_t rankTokenCapacity;
    int64_t nvS;
    int64_t tokenPadding;
    int64_t tokenRouteLimitPerPair;
    int32_t cardsPerServer;
    int32_t cardsPerCabinet;
    int32_t crossCandidateCount;
    int32_t reserved1;
    uint64_t epoch;
    uint64_t topologyHash;
};

struct TokenSegmentMove {
    int32_t expertId;
    int32_t srcRank;
    int32_t dstRank;
    int32_t srcExpertBegin;
    int32_t dstExpertBegin;
    int32_t tokenCount;
};

struct PlanEpochState {
    uint64_t requestedEpoch;
    uint64_t committedEpoch;
    uint64_t topologyHash;
    uint64_t reserved;
};

} // namespace Plan
} // namespace TileXREp

#endif // TILEXR_EP_PLANNER_COMMON_EP_PLAN_TYPES_H
