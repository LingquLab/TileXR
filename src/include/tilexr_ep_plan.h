#ifndef TILEXR_EP_PLAN_H
#define TILEXR_EP_PLAN_H

#ifdef __cplusplus

#include <cstdint>

#include "acl/acl_base.h"
#include "tilexr_api.h"

struct TileXRMoonEPPlanConfig {
    int64_t prefetchSlots;
    int64_t rankTokenCapacity;
    int64_t nvS;
    int64_t tokenPadding;
    int64_t tokenRouteLimitPerPair;
    int32_t cardsPerServer;
    int32_t cardsPerCabinet;
    int32_t crossCandidateCount;
    int32_t reserved;
};


constexpr uint32_t TILEXR_MOONEP_PLAN_METADATA_V2_ABI_VERSION = 2;
constexpr uint64_t TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID = UINT64_MAX;

// ABI-stable downstream contract. Counts are element counts, not byte sizes.
// remoteExperts is destination-oriented [r,b]. expertTargets is owner-oriented
// [e/r, ceil(r/64)] for the current rank. Dispatch writes tokenRemap[nvS]
// using the global-token helpers; Combine consumes it. Planner does not fill it.
struct TileXRMoonEPPlanMetadataV2 {
    uint32_t structSize;
    uint32_t abiVersion;
    int32_t *dst;
    uint64_t dstCount;
    int32_t *cuSeqlens;
    uint64_t cuSeqlensCount;
    int32_t *remoteExperts;
    uint64_t remoteExpertsCount;
    uint64_t *expertTargets;
    uint64_t expertTargetsCount;
    int32_t *remoteStats;
    uint64_t remoteStatsCount;
    int32_t *status;
    uint64_t statusCount;
    int32_t *dupGroups;
    uint64_t dupGroupsCount;
    int32_t *dupLoffs;
    uint64_t dupLoffsCount;
    int32_t *dupCounts;
    uint64_t dupCountsCount;
    int64_t s;
    int64_t k;
    int64_t r;
    int64_t e;
    int64_t b;
    int64_t nvS;
    uint64_t epoch;
};

struct TileXRMoonEPPlanDesc {
    int32_t *dst;
    int32_t *cuSeqlens;
    int32_t *expertsToCopy;
    int32_t *remoteStats;
    int32_t *dupGroups;
    int32_t *dupLoffs;
    int32_t *dupCounts;
    int32_t *status;
    int64_t s;
    int64_t k;
    int64_t r;
    int64_t e;
    int64_t b;
    int64_t cap;
    int64_t nvS;
    int64_t tokenPadding;
    uint64_t epoch;
};

enum TileXRMoonEPPlanStatus : int32_t {
    PLAN_OK = 0,
    PLAN_PARTIAL_NO_FEASIBLE_PAIR = 1,
    PLAN_PARTIAL_PREFETCH_SLOT_EXHAUSTED = 2,
    PLAN_PARTIAL_TOKEN_SUPPLY = 3,
    PLAN_ERROR_CONFIG_MISMATCH = 4,
    PLAN_ERROR_TPE_MISMATCH = 5,
    PLAN_ERROR_LAYOUT_EXCEEDS_NVS = 6,
    PLAN_ERROR_MOVE_RECORD_OVERFLOW = 7,
    PLAN_ERROR_INTERNAL_INVARIANT = 8,
};

// Public C++ helpers shared by Planner, Dispatch, and Combine. encodedDst uses
// rawDst = dstRank * nvS + recvSlot; negative values are duplicate routes and
// decode through rawDst = ~encodedDst.
namespace TileXREp {
namespace Plan {

struct MoonEPRouteTarget {
    int32_t rawDst;
    int32_t dstRank;
    int32_t recvSlot;
    int32_t sendHidden;
    int32_t writeRouteWeight;
};

struct MoonEPRouteDescriptor {
    int32_t srcRank;
    int32_t tokenId;
    int32_t topKId;
    uint64_t globalTokenId;
    int32_t rawDst;
    int32_t dstRank;
    int32_t recvSlot;
    int32_t sendHidden;
    int32_t writeRouteWeight;
};

TileXRMoonEPPlanStatus EncodeMoonEPGlobalTokenId(int32_t srcRank, int32_t tokenId,
    int32_t topKId, int64_t rankSize, int64_t s, int64_t topK, uint64_t *globalTokenId);
TileXRMoonEPPlanStatus DecodeMoonEPGlobalTokenId(uint64_t globalTokenId, int64_t rankSize,
    int64_t s, int64_t topK, int32_t *srcRank, int32_t *tokenId, int32_t *topKId);
TileXRMoonEPPlanStatus DecodeMoonEPDst(
    int32_t encoded, int64_t nvS, int64_t rankSize, MoonEPRouteTarget *target);
TileXRMoonEPPlanStatus BuildMoonEPRouteDescriptor(int32_t srcRank, int32_t tokenId,
    int32_t topKId, int32_t encodedDst, int64_t rankSize, int64_t s, int64_t topK,
    int64_t nvS, MoonEPRouteDescriptor *descriptor);

} // namespace Plan
} // namespace TileXREp

extern "C" {

int TileXRMoeEpPlanV2GetWorkspaceSize(int64_t rankSize, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig *config, uint64_t *localWorkspaceBytes, uint64_t *registeredMetaBytes);

int TileXRMoeEpPlanV2(const int32_t *topkExperts, const int32_t *tokensPerExpert,
    const int32_t *globalRankIds, TileXRCommPtr comm, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig *config, TileXRMoonEPPlanDesc *plan, void *localWorkspace,
    uint64_t localWorkspaceBytes, void *registeredMetaWorkspace, uint64_t registeredMetaBytes,
    aclrtStream stream);

int TileXRMoeEpPlanV2WithMetadata(const int32_t *topkExperts, const int32_t *tokensPerExpert,
    const int32_t *globalRankIds, TileXRCommPtr comm, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig *config, TileXRMoonEPPlanMetadataV2 *metadata,
    void *localWorkspace, uint64_t localWorkspaceBytes, void *registeredMetaWorkspace,
    uint64_t registeredMetaBytes, aclrtStream stream);

} // extern "C"
#endif // __cplusplus
#endif // TILEXR_EP_PLAN_H
