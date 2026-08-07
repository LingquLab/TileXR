#ifndef TILEXR_EP_PLANNER_COMMON_EP_PLAN_DOWNSTREAM_H
#define TILEXR_EP_PLANNER_COMMON_EP_PLAN_DOWNSTREAM_H

#include <cstdint>

#include "tilexr_ep_plan.h"

namespace TileXREp {
namespace Plan {

struct MoonEPReceivedRoute {
    int32_t srcRank;
    int32_t tokenId;
    int32_t topKId;
    int32_t recvSlot;
    int32_t isPrimary;
};

TileXRMoonEPPlanStatus BuildMoonEPExpertTargets(const int32_t *remoteExperts,
    int64_t rankSize, int64_t expertNum, int64_t prefetchSlots, int32_t ownerRank,
    uint64_t *expertTargets, uint64_t expertTargetsCount);

TileXRMoonEPPlanStatus BuildMoonEPDuplicateMetadata(const MoonEPReceivedRoute *records,
    int64_t recordCount, int64_t rankSize, int64_t s, int64_t topK, int64_t nvS,
    int32_t *dupGroups, int32_t *dupLoffs, int32_t *dupCounts);

} // namespace Plan
} // namespace TileXREp

#endif // TILEXR_EP_PLANNER_COMMON_EP_PLAN_DOWNSTREAM_H
