#ifndef TILEXR_EP_PLANNER_COMMON_EP_PLAN_ALGORITHM_H
#define TILEXR_EP_PLANNER_COMMON_EP_PLAN_ALGORITHM_H

#include <climits>
#include <cstdint>
#include "ep_plan_types.h"
#include "tilexr_ep_plan.h"

#ifndef TILEXR_PLAN_ADDR
#define TILEXR_PLAN_ADDR
#endif

#ifndef TILEXR_PLAN_FN
#define TILEXR_PLAN_FN
#endif

namespace TileXREp { namespace Plan {

struct PlanAlgorithmInput {
    int32_t rank;
    int32_t rankSize;
    int64_t s;
    int64_t topK;
    int64_t expertNum;
    TileXRMoonEPPlanConfig config;
    const TILEXR_PLAN_ADDR int32_t *topkExperts;
    const TILEXR_PLAN_ADDR int32_t *tokensPerExpert;
    const TILEXR_PLAN_ADDR int32_t *globalRankIds;
};

struct PlanAlgorithmOutput {
    TILEXR_PLAN_ADDR int32_t *dst;
    TILEXR_PLAN_ADDR int32_t *cuSeqlens;
    TILEXR_PLAN_ADDR int32_t *expertsToCopy;
    TILEXR_PLAN_ADDR int32_t *remoteExperts;
    TILEXR_PLAN_ADDR uint64_t *expertTargets;
    TILEXR_PLAN_ADDR int32_t *remoteStats;
    TILEXR_PLAN_ADDR int32_t *status;
};

struct PlanAlgorithmWorkspace {
    TILEXR_PLAN_ADDR int32_t *expertCount;
    TILEXR_PLAN_ADDR int32_t *rankLoad;
    TILEXR_PLAN_ADDR int32_t *remainingTpe;
    TILEXR_PLAN_ADDR int32_t *alloc;
    TILEXR_PLAN_ADDR int32_t *remoteExpertSet;
    TILEXR_PLAN_ADDR int32_t *srcExpertCursor;
    TILEXR_PLAN_ADDR int32_t *dstExpertCursor;
    TILEXR_PLAN_ADDR int32_t *expertPhysicalBase;
    TILEXR_PLAN_ADDR int32_t *localExpertOrdinal;
    TILEXR_PLAN_ADDR TokenSegmentMove *tokenSegments;
    int32_t tokenSegmentCapacity;
    TILEXR_PLAN_ADDR int32_t *routedPairTokens;
    TILEXR_PLAN_ADDR int32_t *scratch;
    int32_t scratchCount;
    TILEXR_PLAN_ADDR int32_t *affinityOrder;
    bool affinityOrderValid;
};

TILEXR_PLAN_FN TileXRMoonEPPlanStatus RunPlanAlgorithm(const PlanAlgorithmInput &input,
    const PlanAlgorithmOutput &output, PlanAlgorithmWorkspace &workspace);

} }
#endif
