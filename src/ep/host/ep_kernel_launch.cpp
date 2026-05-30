#include "ep_kernel_launch.h"

#include "tilexr_api.h"
#include "tilexr_types.h"

extern void launch_tilexr_ep_dispatch_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs, GM_ADDR x,
    GM_ADDR expertIds, GM_ADDR expandXOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
    GM_ADDR assistInfoForCombineOut, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes,
    int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot,
    int64_t slotBytes, int64_t totalBytes, int64_t magic);

namespace TileXREp {

int TileXREpLaunchDispatchKernel(const EpDispatchParams &params, const EpHostLaunchContext &context)
{
    int64_t magic = 0;
    const int magicRet = TileXRCommNextMagic(params.comm, &magic);
    if (magicRet != TileXR::TILEXR_SUCCESS) {
        return magicRet;
    }

    constexpr uint32_t kMvpBlockDim = 1;
    launch_tilexr_ep_dispatch_kernel(kMvpBlockDim, params.stream, context.devArgs, static_cast<GM_ADDR>(params.x),
        reinterpret_cast<GM_ADDR>(params.expertIds), static_cast<GM_ADDR>(params.expandXOut),
        reinterpret_cast<GM_ADDR>(params.expertTokenNumsOut), reinterpret_cast<GM_ADDR>(params.epRecvCountsOut),
        reinterpret_cast<GM_ADDR>(params.assistInfoForCombineOut), params.bs, params.h, params.topK,
        params.moeExpertNum, context.window.dtypeBytes, context.window.maxRoutesPerSrc, context.window.rowBytes,
        context.window.payloadBytesPerSlot, context.window.assistBytesPerSlot, context.window.slotBytes,
        context.window.totalBytes, magic);
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
