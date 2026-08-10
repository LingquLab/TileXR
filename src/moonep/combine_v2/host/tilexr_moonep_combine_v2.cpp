#include "combine_v2_host.h"

extern "C" int TileXRMoonEpCombineGetWorkspaceSizeV2(
    int64_t bs, int64_t h, int64_t topK, int64_t nvS, uint32_t dtype,
    uint64_t *workspaceBytes, uint64_t *profileOffset,
    uint64_t *outputEpoch0Offset, uint64_t *outputEpoch1Offset)
{
    if (workspaceBytes == nullptr || profileOffset == nullptr ||
        outputEpoch0Offset == nullptr || outputEpoch1Offset == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *workspaceBytes = 0;
    *profileOffset = 0;
    *outputEpoch0Offset = 0;
    *outputEpoch1Offset = 0;

    TileXRMoonEp::CombineV2Layout layout {};
    const int ret = TileXRMoonEp::TileXRMoonEpBuildCombineV2Layout(
        bs, h, topK, nvS, dtype, &layout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    *workspaceBytes = layout.totalBytes;
    *profileOffset = layout.profileOffset;
    *outputEpoch0Offset = layout.scratchOffset[0];
    *outputEpoch1Offset = layout.scratchOffset[1];
    return TILEXR_MOONEP_SUCCESS;
}

extern "C" int TileXRMoonEpCombineV2(void *registeredWorkspace,
    const int32_t *dstLocal, TileXRCommPtr comm, int64_t bs, int64_t h,
    int64_t topK, int64_t nvS, uint32_t aivCoreNum,
    uint64_t *activeOutputOffset,
    uint32_t dtype, aclrtStream stream)
{
    TileXRMoonEp::CombineV2Params params {};
    params.registeredWorkspace = registeredWorkspace;
    params.dstLocal = dstLocal;
    params.comm = comm;
    params.bs = bs;
    params.h = h;
    params.topK = topK;
    params.nvS = nvS;
    params.aivCoreNum = aivCoreNum;
    params.activeOutputOffset = activeOutputOffset;
    params.dtype = dtype;
    params.stream = stream;
    return TileXRMoonEp::TileXRMoonEpRunCombineV2(params);
}
