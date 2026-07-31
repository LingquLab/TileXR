#ifndef TILEXR_EP_HOST_EP_URMA_COMBINE_HOST_H
#define TILEXR_EP_HOST_EP_URMA_COMBINE_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "ep_urma_combine_layout.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

namespace TileXREp {

struct EpUrmaCombineParams {
    void *expertOut = nullptr;
    int32_t *assistInfoForCombine = nullptr;
    float *topKWeights = nullptr;
    TileXRCommPtr comm = nullptr;
    int64_t selfSendCnt = 0;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t topK = 0;
    void *yOut = nullptr;
    void *workspace = nullptr;
    int64_t workspaceBytes = 0;
    void *perfTrace = nullptr;
    int64_t perfTraceBytes = 0;
    void *strictKernelCycles = nullptr;
    int64_t strictKernelCyclesBytes = 0;
    TileXR::TileXRDataType dtype = TileXR::TILEXR_DATA_TYPE_RESERVED;
    aclrtStream stream = nullptr;
};

struct EpUrmaCombineLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    EpUrmaCombineWorkspaceConfig workspace {};
};

int TileXREpGetUrmaCombineProfileSize(int64_t rankSize, int64_t *profileBytes);
int TileXREpValidateBasicUrmaCombineParams(const EpUrmaCombineParams &params);
int TileXREpPrepareUrmaCombineLaunchContext(
    const EpUrmaCombineParams &params, EpUrmaCombineLaunchContext *context);
int TileXREpLaunchPreparedUrmaCombineKernel(
    const EpUrmaCombineParams &params, const EpUrmaCombineLaunchContext &context, int64_t magic,
    bool runStartGate = true);
int TileXREpLaunchUrmaCombineKernel(
    const EpUrmaCombineParams &params, const EpUrmaCombineLaunchContext &context);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_URMA_COMBINE_HOST_H
