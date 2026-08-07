#include "tilexr_moonep_planner.h"

#include "planner_host.h"
#include "planner_launch.h"
#include "tilexr_types.h"

int TileXRMoonEpPlannerGetWorkspaceSizeV2(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, uint64_t *workspaceBytes, int64_t *dispatchedCapacity)
{
    TileXR::CommArgs *commArgs = nullptr;
    if (TileXRGetCommArgsHost(comm, commArgs) != TileXR::TILEXR_SUCCESS ||
        commArgs == nullptr || commArgs->rankSize <= 0 || expertCount % commArgs->rankSize != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXRMoonEpPlannerGetWorkspaceSizeV3(comm, s, k, expertCount,
        expertCount / commArgs->rankSize, 1, workspaceBytes, dispatchedCapacity);
}

int TileXRMoonEpPlannerGetWorkspaceSizeV3(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, int64_t b, int64_t tokenPadding,
    uint64_t *workspaceBytes, int64_t *nvS)
{
    if (workspaceBytes == nullptr || nvS == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *workspaceBytes = 0;
    *nvS = 0;

    TileXRMoonEp::PlannerLayout layout {};
    const int ret = TileXRMoonEp::TileXRMoonEpPrepareLayout(
        comm, s, k, expertCount, b, tokenPadding, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    *workspaceBytes = layout.workspaceBytes;
    *nvS = layout.nvS;
    return TileXR::TILEXR_SUCCESS;
}

int TileXRMoonEpPlannerV2(const int32_t *topkExpertIds, const int32_t *tokensPerExpert,
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount,
    void *workspace, uint64_t workspaceBytes, int32_t *dst, int32_t *cuSeqlens,
    int32_t *expertsToCopy, int32_t *remoteStats, int32_t *plannerStatus,
    uint64_t waitIterations, aclrtStream stream)
{
    TileXR::CommArgs *commArgs = nullptr;
    if (TileXRGetCommArgsHost(comm, commArgs) != TileXR::TILEXR_SUCCESS ||
        commArgs == nullptr || commArgs->rankSize <= 0 || expertCount % commArgs->rankSize != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    TileXRMoonEp::PlannerLayout layout {};
    const int layoutRet = TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(
        commArgs->rankSize, s, k, expertCount, expertCount / commArgs->rankSize, 1, &layout);
    if (layoutRet != TileXR::TILEXR_SUCCESS || workspace == nullptr ||
        workspaceBytes < layout.workspaceBytes) {
        return layoutRet == TileXR::TILEXR_SUCCESS ?
            TileXR::TILEXR_ERROR_PARA_CHECK_FAIL : layoutRet;
    }
    auto *base = static_cast<unsigned char *>(workspace);
    return TileXRMoonEpPlannerV3(topkExpertIds, tokensPerExpert, comm, s, k,
        expertCount, expertCount / commArgs->rankSize, 1, workspace, workspaceBytes,
        dst, cuSeqlens, expertsToCopy,
        reinterpret_cast<int32_t *>(base + layout.compatZeroFillOffset), remoteStats,
        reinterpret_cast<int32_t *>(base + layout.compatDupCountsOffset), plannerStatus,
        waitIterations, stream);
}

int TileXRMoonEpPlannerV3(const int32_t *topkExpertIds, const int32_t *tokensPerExpert,
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount,
    int64_t b, int64_t tokenPadding, void *workspace, uint64_t workspaceBytes,
    int32_t *dst, int32_t *cuSeqlens, int32_t *expertsToCopy,
    int32_t *zeroFillRanges, int32_t *remoteStats, int32_t *dupCounts,
    int32_t *plannerStatus, uint64_t waitIterations, aclrtStream stream)
{
    TileXRMoonEp::PlannerParams params {};
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

    TileXRMoonEp::PlannerLaunchContext context {};
    int ret = TileXRMoonEp::TileXRMoonEpPrepareLaunchContext(params, &context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXRMoonEp::TileXRMoonEpLaunchKernel(params, context);
}
