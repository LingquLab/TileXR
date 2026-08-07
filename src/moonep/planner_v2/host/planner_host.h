#ifndef TILEXR_MOONEP_PLANNER_HOST_H
#define TILEXR_MOONEP_PLANNER_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "planner_layout.h"
#include "tilexr_api.h"

namespace TileXRMoonEp {

struct PlannerParams {
    const int32_t *topkExpertIds = nullptr;
    const int32_t *tokensPerExpert = nullptr;
    TileXRCommPtr comm = nullptr;
    int64_t s = 0;
    int64_t k = 0;
    int64_t expertCount = 0;
    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    int32_t *dst = nullptr;
    int32_t *cuSeqlens = nullptr;
    int32_t *expertsToCopy = nullptr;
    int32_t *remoteStats = nullptr;
    int32_t *plannerStatus = nullptr;
    uint64_t waitIterations = 0;
    aclrtStream stream = nullptr;
};

struct PlannerLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    PlannerLayout layout {};
};

int TileXRMoonEpPrepareLayout(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, PlannerLayout *layout);

int TileXRMoonEpValidateParams(const PlannerParams &params, const TileXR::CommArgs &commArgs,
    const PlannerLayout &layout);

int TileXRMoonEpPrepareLaunchContext(const PlannerParams &params, PlannerLaunchContext *context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PLANNER_HOST_H
