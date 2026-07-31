#include "tilexr_ep.h"

#include <limits>

#include "ep_urma_combine.h"
#include "ep_urma_combine_host.h"

int TileXRMoeEpCombineUrmaGetWorkspaceSize(int64_t rankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t selfSendCapacity, int64_t *workspaceBytes)
{
    if (workspaceBytes == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    TileXREp::EpUrmaCombineWorkspaceConfig layout {};
    const int ret = TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(
        rankSize, bs, h, topK, selfSendCapacity, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    const int64_t alignment = TileXREp::kEpUrmaCombineWorkspaceAlignment;
    const int64_t remainder = layout.requiredBytes % alignment;
    if (remainder != 0 && layout.requiredBytes >
        std::numeric_limits<int64_t>::max() - (alignment - remainder)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *workspaceBytes = remainder == 0 ? layout.requiredBytes : layout.requiredBytes + alignment - remainder;
    return TileXR::TILEXR_SUCCESS;
}

int TileXRMoeEpCombineUrmaGetProfileSize(int64_t rankSize, int64_t *profileBytes)
{
    return TileXREp::TileXREpGetUrmaCombineProfileSize(rankSize, profileBytes);
}

namespace {

int CombineUrmaImpl(void *expertOut, int32_t *assistInfoForCombine, float *topKWeights,
    TileXRCommPtr comm, int64_t selfSendCnt, int64_t bs, int64_t h, int64_t topK, void *yOut,
    void *workspace, int64_t workspaceBytes, void *perfTrace, int64_t perfTraceBytes,
    TileXR::TileXRDataType dtype, aclrtStream stream)
{
    TileXREp::EpUrmaCombineParams params {};
    params.expertOut = expertOut;
    params.assistInfoForCombine = assistInfoForCombine;
    params.topKWeights = topKWeights;
    params.comm = comm;
    params.selfSendCnt = selfSendCnt;
    params.bs = bs;
    params.h = h;
    params.topK = topK;
    params.yOut = yOut;
    params.workspace = workspace;
    params.workspaceBytes = workspaceBytes;
    params.perfTrace = perfTrace;
    params.perfTraceBytes = perfTraceBytes;
    params.dtype = dtype;
    params.stream = stream;

    int ret = TileXREp::TileXREpValidateBasicUrmaCombineParams(params);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    TileXREp::EpUrmaCombineLaunchContext context {};
    ret = TileXREp::TileXREpPrepareUrmaCombineLaunchContext(params, &context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXREp::TileXREpLaunchUrmaCombineKernel(params, context);
}

} // namespace

int TileXRMoeEpCombineUrma(void *expertOut, int32_t *assistInfoForCombine, float *topKWeights,
    TileXRCommPtr comm, int64_t selfSendCnt, int64_t bs, int64_t h, int64_t topK, void *yOut,
    void *workspace, int64_t workspaceBytes, TileXR::TileXRDataType dtype, aclrtStream stream)
{
    return CombineUrmaImpl(expertOut, assistInfoForCombine, topKWeights, comm, selfSendCnt, bs, h, topK,
        yOut, workspace, workspaceBytes, nullptr, 0, dtype, stream);
}

int TileXRMoeEpCombineUrmaProfile(void *expertOut, int32_t *assistInfoForCombine, float *topKWeights,
    TileXRCommPtr comm, int64_t selfSendCnt, int64_t bs, int64_t h, int64_t topK, void *yOut,
    void *workspace, int64_t workspaceBytes, void *perfTrace, int64_t perfTraceBytes,
    TileXR::TileXRDataType dtype, aclrtStream stream)
{
#if !defined(TILEXR_EP_ENABLE_PROFILING)
    (void)expertOut;
    (void)assistInfoForCombine;
    (void)topKWeights;
    (void)comm;
    (void)selfSendCnt;
    (void)bs;
    (void)h;
    (void)topK;
    (void)yOut;
    (void)workspace;
    (void)workspaceBytes;
    (void)perfTrace;
    (void)perfTraceBytes;
    (void)dtype;
    (void)stream;
    return TileXR::TILEXR_ERROR_NOT_SUPPORT;
#else
    if (perfTrace == nullptr || perfTraceBytes <= 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return CombineUrmaImpl(expertOut, assistInfoForCombine, topKWeights, comm, selfSendCnt, bs, h, topK,
        yOut, workspace, workspaceBytes, perfTrace, perfTraceBytes, dtype, stream);
#endif
}
