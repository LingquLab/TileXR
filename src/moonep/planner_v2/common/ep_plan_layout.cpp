#include "ep_plan_layout.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "tilexr_types.h"

namespace TileXREp {
namespace Plan {
namespace {

constexpr int64_t kPlanMaxLogicalRankSize = 512;

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t *out)
{
    if (alignment == 0 || out == nullptr) {
        return false;
    }
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        *out = value;
        return true;
    }
    return CheckedAdd(value, alignment - remainder, out);
}

bool AddRegion(uint64_t bytes, uint64_t *cursor, PlanRegion *region)
{
    uint64_t offset = 0;
    uint64_t end = 0;
    if (cursor == nullptr || region == nullptr || !AlignUp(*cursor, kPlanWorkspaceAlignment, &offset) ||
        !CheckedAdd(offset, bytes, &end)) {
        return false;
    }
    region->offset = offset;
    region->bytes = bytes;
    *cursor = end;
    return true;
}

bool CountBytes(uint64_t count, uint64_t elementBytes, uint64_t *bytes)
{
    return CheckedMul(count, elementBytes, bytes);
}

bool ToUnsigned(int64_t value, uint64_t *out)
{
    if (value <= 0 || out == nullptr) {
        return false;
    }
    *out = static_cast<uint64_t>(value);
    return true;
}

bool FinishLayout(uint64_t cursor, uint64_t *totalBytes)
{
    uint64_t aligned = 0;
    if (!AlignUp(cursor, kPlanWorkspaceAlignment, &aligned) ||
        aligned > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *totalBytes = aligned;
    return true;
}

} // namespace

int BuildPlanWorkspaceLayout(int64_t rankSize, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig &config, PlanWorkspaceLayout *layout)
{
    if (layout == nullptr || rankSize <= 0 || rankSize > kPlanMaxLogicalRankSize || s <= 0 || topK <= 0 ||
        topK > 32 || expertNum <= 0 || expertNum % rankSize != 0 ||
        static_cast<uint64_t>(expertNum) * sizeof(int32_t) > kPlanMaxUdmaTransferBytes ||
        config.prefetchSlots <= 0 || config.prefetchSlots > INT32_MAX || config.tokenPadding <= 0 ||
        config.cardsPerServer != kPlanCardsPerServer || config.cardsPerCabinet != kPlanCardsPerCabinet ||
        config.crossCandidateCount != kPlanCrossCandidateCount || config.rankTokenCapacity <= 0 || config.nvS <= 0 ||
        config.tokenRouteLimitPerPair < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    uint64_t r = 0;
    uint64_t tokenCount = 0;
    uint64_t e = 0;
    uint64_t b = 0;
    uint64_t nvS = 0;
    uint64_t sk = 0;
    uint64_t rankSlots = 0;
    if (!ToUnsigned(rankSize, &r) || !ToUnsigned(expertNum, &e) || !ToUnsigned(config.prefetchSlots, &b) ||
        !ToUnsigned(config.nvS, &nvS) || !CheckedMul(static_cast<uint64_t>(s), static_cast<uint64_t>(topK), &sk) ||
        sk > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        config.rankTokenCapacity != static_cast<int64_t>(sk) || config.nvS < config.rankTokenCapacity ||
        config.tokenRouteLimitPerPair > config.rankTokenCapacity || !CheckedMul(r, nvS, &rankSlots) ||
        rankSlots > static_cast<uint64_t>(INT32_MAX)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    tokenCount = sk;

    uint64_t re = 0;
    uint64_t rb = 0;
    uint64_t rr = 0;
    uint64_t statusCount = 0;
    if (!CheckedMul(r, e, &re) || !CheckedMul(r, b, &rb) || !CheckedMul(r, r, &rr) ||
        !CheckedMul(r, static_cast<uint64_t>(kPlanStatusWords), &statusCount)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    PlanWorkspaceLayout result {};
    uint64_t cursor = 0;
    uint64_t bytes = 0;
#define TILEXR_PLAN_ADD_LOCAL(REGION, COUNT, TYPE) \
    do { \
        if (!CountBytes((COUNT), sizeof(TYPE), &bytes) || !AddRegion(bytes, &cursor, &result.local.REGION)) { \
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL; \
        } \
    } while (0)

    TILEXR_PLAN_ADD_LOCAL(expertCount, e, int32_t);
    TILEXR_PLAN_ADD_LOCAL(rankLoad, r, int32_t);
    TILEXR_PLAN_ADD_LOCAL(remainingTpe, re, int32_t);
    TILEXR_PLAN_ADD_LOCAL(alloc, re, int32_t);
    TILEXR_PLAN_ADD_LOCAL(remoteExpertSet, rb, int32_t);
    TILEXR_PLAN_ADD_LOCAL(srcExpertCursor, re, int32_t);
    TILEXR_PLAN_ADD_LOCAL(dstExpertCursor, re, int32_t);
    TILEXR_PLAN_ADD_LOCAL(expertPhysicalBase, re, int32_t);
    TILEXR_PLAN_ADD_LOCAL(localExpertOrdinal, tokenCount, int32_t);
    TILEXR_PLAN_ADD_LOCAL(tokenSegments, tokenCount, TokenSegmentMove);
    if (config.tokenRouteLimitPerPair != 0) {
        TILEXR_PLAN_ADD_LOCAL(routedPairTokens, rr, int32_t);
    }
    TILEXR_PLAN_ADD_LOCAL(scratchStatus, statusCount + r * 8, int32_t);
#undef TILEXR_PLAN_ADD_LOCAL
    if (!FinishLayout(cursor, &result.local.totalBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    cursor = 0;
#define TILEXR_PLAN_ADD_META(REGION, COUNT, TYPE) \
    do { \
        if (!CountBytes((COUNT), sizeof(TYPE), &bytes) || \
            !AddRegion(bytes, &cursor, &result.registeredMeta.REGION)) { \
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL; \
        } \
    } while (0)

    TILEXR_PLAN_ADD_META(planCallHeaders, r * kPlanHeaderStrideBytes, uint8_t);
    TILEXR_PLAN_ADD_META(tpe, re, int32_t);
    TILEXR_PLAN_ADD_META(globalRankIds, r, int32_t);
    TILEXR_PLAN_ADD_META(epochState, 1, PlanEpochState);
    TILEXR_PLAN_ADD_META(affinityOrder, rr, int32_t);
    TILEXR_PLAN_ADD_META(localStatusByRank, r * kPlanStatusStrideBytes, uint8_t);
    TILEXR_PLAN_ADD_META(barrierFlags, r * 3 * kPlanBarrierSlotBytes, uint8_t);
#undef TILEXR_PLAN_ADD_META
    if (!FinishLayout(cursor, &result.registeredMeta.totalBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    *layout = result;
    return TileXR::TILEXR_SUCCESS;
}

int ValidatePlanWorkspaceBytes(const PlanWorkspaceLayout &layout, uint64_t localWorkspaceBytes,
    uint64_t registeredMetaBytes)
{
    if (localWorkspaceBytes < layout.local.totalBytes || registeredMetaBytes < layout.registeredMeta.totalBytes) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace Plan
} // namespace TileXREp
int TileXRMoeEpPlanV2GetWorkspaceSize(int64_t rankSize, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig *config, uint64_t *localWorkspaceBytes, uint64_t *registeredMetaBytes)
{
    if (config == nullptr || localWorkspaceBytes == nullptr || registeredMetaBytes == nullptr ||
        reinterpret_cast<uintptr_t>(config) % alignof(TileXRMoonEPPlanConfig) != 0 ||
        reinterpret_cast<uintptr_t>(localWorkspaceBytes) % alignof(uint64_t) != 0 ||
        reinterpret_cast<uintptr_t>(registeredMetaBytes) % alignof(uint64_t) != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    TileXREp::Plan::PlanWorkspaceLayout layout {};
    const int ret = TileXREp::Plan::BuildPlanWorkspaceLayout(rankSize, s, topK, expertNum, *config, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    *localWorkspaceBytes = layout.local.totalBytes;
    *registeredMetaBytes = layout.registeredMeta.totalBytes;
    return TileXR::TILEXR_SUCCESS;
}
