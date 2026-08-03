#include "tilexr_moonep_planner.h"

#include "planner_host.h"
#include "planner_launch.h"
#include "tilexr_types.h"

int TileXRMoonEpPlannerGetWorkspaceSizeV2(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, uint64_t *workspaceBytes, int64_t *dispatchedCapacity)
{
    if (workspaceBytes == nullptr || dispatchedCapacity == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *workspaceBytes = 0;
    *dispatchedCapacity = 0;

    TileXRMoonEp::PlannerLayout layout {};
    const int ret = TileXRMoonEp::TileXRMoonEpPrepareLayout(
        comm, s, k, expertCount, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    *workspaceBytes = layout.workspaceBytes;
    *dispatchedCapacity = layout.dispatchedCapacity;
    return TileXR::TILEXR_SUCCESS;
}

int TileXRMoonEpPlannerV2(const int32_t *topkExpertIds, const int32_t *tokensPerExpert,
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount,
    void *workspace, uint64_t workspaceBytes, int32_t *dst, int32_t *cuSeqlens,
    int32_t *expertsToCopy, int32_t *remoteStats, int32_t *plannerStatus,
    uint64_t waitIterations, aclrtStream stream)
{
    TileXRMoonEp::PlannerParams params {};
    params.topkExpertIds = topkExpertIds;
    params.tokensPerExpert = tokensPerExpert;
    params.comm = comm;
    params.s = s;
    params.k = k;
    params.expertCount = expertCount;
    params.workspace = workspace;
    params.workspaceBytes = workspaceBytes;
    params.dst = dst;
    params.cuSeqlens = cuSeqlens;
    params.expertsToCopy = expertsToCopy;
    params.remoteStats = remoteStats;
    params.plannerStatus = plannerStatus;
    params.waitIterations = waitIterations;
    params.stream = stream;

    TileXRMoonEp::PlannerLaunchContext context {};
    int ret = TileXRMoonEp::TileXRMoonEpPrepareLaunchContext(params, &context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXRMoonEp::TileXRMoonEpLaunchKernel(params, context);
}
