#include "tilexr_moonep_planner.h"

#include "planner_host.h"
#include "planner_launch.h"
#include "tilexr_types.h"

int TileXRMoonEpPlannerGetWorkspaceSizeV3(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, int64_t b, int64_t tokenPadding,
    uint64_t *workspaceBytes, int64_t *nvS)
{
    if (workspaceBytes == nullptr || nvS == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *workspaceBytes = 0;
    *nvS = 0;

    TileXRMoonEpV3::PlannerLayout layout {};
    const int ret = TileXRMoonEpV3::TileXRMoonEpPrepareLayout(
        comm, s, k, expertCount, b, tokenPadding, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    *workspaceBytes = layout.workspaceBytes;
    *nvS = layout.nvS;
    return TileXR::TILEXR_SUCCESS;
}

int TileXRMoonEpPlannerGetDstLocalOffsetV3(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, int64_t b, int64_t tokenPadding,
    uint64_t *dstLocalOffset)
{
    if (dstLocalOffset == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *dstLocalOffset = 0;

    TileXRMoonEpV3::PlannerLayout layout {};
    const int ret = TileXRMoonEpV3::TileXRMoonEpPrepareLayout(
        comm, s, k, expertCount, b, tokenPadding, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    *dstLocalOffset = layout.dstLocalOffset;
    return TileXR::TILEXR_SUCCESS;
}

int TileXRMoonEpPlannerV3(const int32_t *topkExpertIds, const int32_t *tokensPerExpert,
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount,
    int64_t b, int64_t tokenPadding, void *workspace, uint64_t workspaceBytes,
    int32_t *dst, int32_t *cuSeqlens, int32_t *expertsToCopy,
    int32_t *zeroFillRanges, int32_t *remoteStats, int32_t *dupCounts,
    int32_t *plannerStatus, uint64_t waitIterations, aclrtStream stream)
{
    TileXRMoonEpV3::PlannerParams params {};
    params.topkExpertIds = topkExpertIds;
    params.tokensPerExpert = tokensPerExpert;
    params.comm = comm;
    params.s = s;
    params.k = k;
    params.expertCount = expertCount;
    params.b = b;
    params.tokenPadding = tokenPadding;
    params.workspace = workspace;
    params.workspaceBytes = workspaceBytes;
    params.dst = dst;
    params.cuSeqlens = cuSeqlens;
    params.expertsToCopy = expertsToCopy;
    params.zeroFillRanges = zeroFillRanges;
    params.remoteStats = remoteStats;
    params.dupCounts = dupCounts;
    params.plannerStatus = plannerStatus;
    params.waitIterations = waitIterations;
    params.stream = stream;

    TileXRMoonEpV3::PlannerLaunchContext context {};
    int ret = TileXRMoonEpV3::TileXRMoonEpPrepareLaunchContext(params, &context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXRMoonEpV3::TileXRMoonEpLaunchKernel(params, context);
}
