#ifndef TILEXR_EP_PLANNER_COMMON_EP_PLAN_LAYOUT_H
#define TILEXR_EP_PLANNER_COMMON_EP_PLAN_LAYOUT_H

#include <cstdint>

#include "ep_plan_types.h"
#include "tilexr_ep_plan.h"

namespace TileXREp {
namespace Plan {

struct PlanRegion {
    uint64_t offset;
    uint64_t bytes;
};

struct PlanLocalWorkspaceLayout {
    PlanRegion expertCount;
    PlanRegion rankLoad;
    PlanRegion remainingTpe;
    PlanRegion alloc;
    PlanRegion remoteExpertSet;
    PlanRegion srcExpertCursor;
    PlanRegion dstExpertCursor;
    PlanRegion expertPhysicalBase;
    PlanRegion localExpertOrdinal;
    PlanRegion tokenSegments;
    PlanRegion routedPairTokens;
    PlanRegion scratchStatus;
    uint64_t totalBytes;
};

struct PlanRegisteredMetaLayout {
    PlanRegion planCallHeaders;
    PlanRegion tpe;
    PlanRegion globalRankIds;
    PlanRegion epochState;
    PlanRegion affinityOrder;
    PlanRegion localStatusByRank;
    PlanRegion barrierFlags;
    uint64_t totalBytes;
};

struct PlanWorkspaceLayout {
    PlanLocalWorkspaceLayout local;
    PlanRegisteredMetaLayout registeredMeta;
};

int BuildPlanWorkspaceLayout(int64_t rankSize, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig &config, PlanWorkspaceLayout *layout);

int ValidatePlanWorkspaceBytes(const PlanWorkspaceLayout &layout, uint64_t localWorkspaceBytes,
    uint64_t registeredMetaBytes);

} // namespace Plan
} // namespace TileXREp

#endif // TILEXR_EP_PLANNER_COMMON_EP_PLAN_LAYOUT_H
