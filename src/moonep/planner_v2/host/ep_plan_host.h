#ifndef TILEXR_EP_PLANNER_HOST_EP_PLAN_HOST_H
#define TILEXR_EP_PLANNER_HOST_EP_PLAN_HOST_H

#include <cstdint>

#include "ep_plan_layout.h"
#include "tilexr_api.h"
#include "tilexr_ep_plan.h"

namespace TileXREp {
namespace Plan {

struct PlanHostArguments {
    const int32_t *topkExperts;
    const int32_t *tokensPerExpert;
    const int32_t *globalRankIds;
    int64_t s;
    int64_t topK;
    int64_t expertNum;
    const TileXRMoonEPPlanConfig *config;
    const TileXRMoonEPPlanDesc *plan;
    int32_t *remoteExperts;
    uint64_t *expertTargets;
    void *localWorkspace;
    uint64_t localWorkspaceBytes;
    void *registeredMetaWorkspace;
    uint64_t registeredMetaBytes;
    uint64_t waitIterations;
    aclrtStream stream;
};

struct PlanRuntimeMetadata {
    const TileXR::CommArgs *hostCommArgs;
    GM_ADDR deviceCommArgs;
};

struct PlanHostContext {
    PlanWorkspaceLayout layout;
    PlanCallHeader callHeader;
    int32_t rank;
    GM_ADDR deviceCommArgs;
};

int BuildPlanCallHeader(int64_t rankSize, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig &config, uint64_t epoch, uint64_t topologyHash, PlanCallHeader *header);

bool PlanCallHeadersMatch(const PlanCallHeader &lhs, const PlanCallHeader &rhs);

// Checks the registered-Meta commit and immutable Plan identity. The caller is
// still responsible for guaranteeing that Top-K routes and tokensPerExpert
// contents have not changed since the committed Plan was produced.
bool IsCommittedPlanReusable(const PlanCallHeader &committedHeader,
    const PlanCallHeader &requestedHeader, const PlanEpochState &epochState);

int ValidatePlanHostArguments(
    const PlanHostArguments &arguments, const PlanRuntimeMetadata &runtime, PlanHostContext *context);
int PreparePlanLaunchContext(
    TileXRCommPtr comm, const PlanHostArguments &arguments, PlanHostContext *context);

} // namespace Plan
} // namespace TileXREp

#endif // TILEXR_EP_PLANNER_HOST_EP_PLAN_HOST_H
