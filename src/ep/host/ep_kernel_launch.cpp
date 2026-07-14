#include "ep_kernel_launch.h"

#include <cstdint>

#include "ep_window.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

extern void launch_tilexr_ep_dispatch_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs, GM_ADDR x,
    GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut,
    GM_ADDR epRecvCountsOut, GM_ADDR tpRecvCountsOut, GM_ADDR assistInfoForCombineOut, GM_ADDR workspace, int64_t bs, int64_t h,
    int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes, int64_t maxRoutesPerSrc, int64_t rowBytes,
    int64_t payloadRowBytes,
    int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes,
    int64_t expertTokenNumsType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode,
    int64_t tpWorldSize, int64_t tpRankId, int64_t magic);

extern void launch_tilexr_ep_dispatch_cross_node_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs, GM_ADDR x,
    GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut,
    GM_ADDR epRecvCountsOut, GM_ADDR tpRecvCountsOut, GM_ADDR assistInfoForCombineOut, GM_ADDR workspace, int64_t bs, int64_t h,
    int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes, int64_t maxRoutesPerSrc, int64_t rowBytes,
    int64_t payloadRowBytes,
    int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes,
    int64_t expertTokenNumsType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode,
    int64_t tpWorldSize, int64_t tpRankId, int64_t magic);

extern void launch_tilexr_ep_combine_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs, GM_ADDR expertOut,
    GM_ADDR assistInfoForCombine, GM_ADDR epRecvCounts, GM_ADDR yOut, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, int64_t dtype, int64_t dtypeBytes, int64_t maxRoutesPerSrc, int64_t rowBytes,
    int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes, int64_t magic);

extern void launch_tilexr_ep_combine_cross_node_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs,
    GM_ADDR expertOut, GM_ADDR assistInfoForCombine, GM_ADDR epRecvCounts, GM_ADDR yOut, GM_ADDR workspace,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum, int64_t dtype, int64_t dtypeBytes,
    int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot,
    int64_t slotBytes, int64_t totalBytes, int64_t magic);

namespace TileXREp {

namespace {

bool TileXREpUsesCrossNodeKernel(const EpHostLaunchContext &context)
{
    return context.hostArgs != nullptr && context.hostArgs->localRankSize > 0 &&
        context.hostArgs->localRankSize < context.hostArgs->rankSize;
}

} // namespace

int TileXREpLaunchDispatchKernel(const EpDispatchParams &params, const EpHostLaunchContext &context)
{
    int64_t magic = 0;
    const int magicRet = TileXRCommNextMagic(params.comm, &magic);
    if (magicRet != TileXR::TILEXR_SUCCESS) {
        return magicRet;
    }

    constexpr uint32_t kMvpBlockDim = 1;
    if (TileXREpUsesCrossNodeKernel(context)) {
        launch_tilexr_ep_dispatch_cross_node_kernel(kMvpBlockDim, params.stream, context.devArgs,
            static_cast<GM_ADDR>(params.x), reinterpret_cast<GM_ADDR>(params.expertIds),
            static_cast<GM_ADDR>(params.scales), reinterpret_cast<GM_ADDR>(params.xActiveMask),
            static_cast<GM_ADDR>(params.expandXOut), static_cast<GM_ADDR>(params.dynamicScalesOut),
            reinterpret_cast<GM_ADDR>(params.expertTokenNumsOut), reinterpret_cast<GM_ADDR>(params.epRecvCountsOut),
            reinterpret_cast<GM_ADDR>(params.tpRecvCountsOut),
            reinterpret_cast<GM_ADDR>(params.assistInfoForCombineOut), static_cast<GM_ADDR>(params.workspace),
            params.bs, params.h, params.topK, params.moeExpertNum, context.window.dtypeBytes,
            context.window.maxRoutesPerSrc, context.window.rowBytes, context.window.payloadRowBytes,
            context.window.payloadBytesPerSlot,
            context.window.assistBytesPerSlot, context.window.slotBytes, context.window.totalBytes,
            params.expertTokenNumsType, params.sharedExpertNum, params.sharedExpertRankNum, params.quantMode,
            params.tpWorldSize, params.tpRankId, magic);
    } else {
        launch_tilexr_ep_dispatch_kernel(kMvpBlockDim, params.stream, context.devArgs, static_cast<GM_ADDR>(params.x),
            reinterpret_cast<GM_ADDR>(params.expertIds), static_cast<GM_ADDR>(params.scales),
            reinterpret_cast<GM_ADDR>(params.xActiveMask), static_cast<GM_ADDR>(params.expandXOut),
            static_cast<GM_ADDR>(params.dynamicScalesOut), reinterpret_cast<GM_ADDR>(params.expertTokenNumsOut),
            reinterpret_cast<GM_ADDR>(params.epRecvCountsOut),
            reinterpret_cast<GM_ADDR>(params.tpRecvCountsOut), reinterpret_cast<GM_ADDR>(params.assistInfoForCombineOut),
            static_cast<GM_ADDR>(params.workspace), params.bs, params.h, params.topK, params.moeExpertNum, context.window.dtypeBytes,
            context.window.maxRoutesPerSrc, context.window.rowBytes, context.window.payloadRowBytes,
            context.window.payloadBytesPerSlot, context.window.assistBytesPerSlot, context.window.slotBytes,
            context.window.totalBytes, params.expertTokenNumsType, params.sharedExpertNum, params.sharedExpertRankNum,
            params.quantMode, params.tpWorldSize, params.tpRankId, magic);
    }
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpLaunchCombineKernel(const EpCombineParams &params, const EpHostLaunchContext &context)
{
    int64_t magic = 0;
    const int magicRet = TileXRCommNextMagic(params.comm, &magic);
    if (magicRet != TileXR::TILEXR_SUCCESS) {
        return magicRet;
    }

    constexpr uint32_t kMvpBlockDim = 1;
    if (TileXREpUsesCrossNodeKernel(context)) {
        launch_tilexr_ep_combine_cross_node_kernel(kMvpBlockDim, params.stream, context.devArgs,
            static_cast<GM_ADDR>(params.expertOut), reinterpret_cast<GM_ADDR>(params.assistInfoForCombine),
            reinterpret_cast<GM_ADDR>(params.epRecvCounts), static_cast<GM_ADDR>(params.yOut),
            static_cast<GM_ADDR>(params.workspace), params.bs, params.h, params.topK, params.moeExpertNum,
            static_cast<int64_t>(params.dtype), context.window.dtypeBytes, context.window.maxRoutesPerSrc,
            context.window.rowBytes, context.window.payloadBytesPerSlot, context.window.assistBytesPerSlot,
            context.window.slotBytes, context.window.totalBytes, magic);
    } else {
        launch_tilexr_ep_combine_kernel(kMvpBlockDim, params.stream, context.devArgs,
            static_cast<GM_ADDR>(params.expertOut), reinterpret_cast<GM_ADDR>(params.assistInfoForCombine),
            reinterpret_cast<GM_ADDR>(params.epRecvCounts), static_cast<GM_ADDR>(params.yOut),
            params.bs, params.h, params.topK, params.moeExpertNum, static_cast<int64_t>(params.dtype),
            context.window.dtypeBytes, context.window.maxRoutesPerSrc, context.window.rowBytes,
            context.window.payloadBytesPerSlot, context.window.assistBytesPerSlot, context.window.slotBytes,
            context.window.totalBytes, magic);
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
