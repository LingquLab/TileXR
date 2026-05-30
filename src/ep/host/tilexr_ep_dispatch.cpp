#include "tilexr_ep.h"

#include "ep_dispatch_host.h"
#include "ep_kernel_launch.h"

int TileXRMoeEpDispatch(void *x, int32_t *expertIds, TileXRCommPtr comm,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
    void *expandXOut, int64_t *expertTokenNumsOut, int32_t *epRecvCountsOut,
    int32_t *assistInfoForCombineOut, TileXR::TileXRDataType dtype, aclrtStream stream)
{
    TileXREp::EpDispatchParams params {};
    params.x = x;
    params.expertIds = expertIds;
    params.comm = comm;
    params.bs = bs;
    params.h = h;
    params.topK = topK;
    params.moeExpertNum = moeExpertNum;
    params.expandXOut = expandXOut;
    params.expertTokenNumsOut = expertTokenNumsOut;
    params.epRecvCountsOut = epRecvCountsOut;
    params.assistInfoForCombineOut = assistInfoForCombineOut;
    params.dtype = dtype;
    params.stream = stream;

    int ret = TileXREp::TileXREpValidateBasicDispatchParams(params);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXREp::EpHostLaunchContext context {};
    ret = TileXREp::TileXREpPrepareLaunchContext(params, &context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    return TileXREp::TileXREpLaunchDispatchKernel(params, context);
}
