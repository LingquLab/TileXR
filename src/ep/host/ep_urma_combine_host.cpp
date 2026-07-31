#include "ep_urma_combine_host.h"

#include <cstdint>
#include <limits>


#include "ep_urma_combine.h"
#include "tilexr_perf_trace.h"
#include "tilexr_udma_reg.h"

extern void launch_tilexr_ep_urma_combine_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs,
    GM_ADDR expertOut, GM_ADDR assistInfoForCombine, GM_ADDR topKWeights, GM_ADDR yOut, GM_ADDR workspace,
    int64_t selfSendCnt, int64_t bs, int64_t h, int64_t topK, int64_t workspaceBytes, int64_t magic,
    int64_t commBytes, int64_t blockCount, int64_t routeStride, int64_t rxWindowBytes, int64_t rxWindowOffset0,
    int64_t rxWindowOffset1, int64_t roundDoneOffset0, int64_t roundDoneOffset1, int64_t rxLaneDoneOffset,
    int64_t senderDoneOffset, int64_t roundPublishOffset, int64_t roundCreditOffset,
    int64_t startGateOffset, int64_t runStartGate,
    int64_t errorStatusOffset, int64_t txReadyOffset, int64_t txDataOffset, GM_ADDR perfTrace, int64_t perfTraceBytes,
    GM_ADDR strictKernelCycles);

namespace TileXREp {


int TileXREpGetUrmaCombineProfileSize(int64_t rankSize, int64_t *profileBytes)
{
    if (profileBytes == nullptr || rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    constexpr int64_t kStatsPerRank = kEpUrmaCombineAivCount * kEpUrmaCombinePerfStageCount;
    constexpr int64_t kStatsBytesPerRank =
        kStatsPerRank * static_cast<int64_t>(sizeof(TileXR::TileXRPerfCoreStageStats));
    if (rankSize > (std::numeric_limits<int64_t>::max() -
        static_cast<int64_t>(TileXR::TILEXR_PERF_TRACE_STATS_OFFSET)) / kStatsBytesPerRank) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *profileBytes = static_cast<int64_t>(TileXR::TILEXR_PERF_TRACE_STATS_OFFSET) +
        rankSize * kStatsBytesPerRank;
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpValidateBasicUrmaCombineParams(const EpUrmaCombineParams &params)
{
    if ((params.selfSendCnt > 0 && (params.expertOut == nullptr || params.assistInfoForCombine == nullptr)) ||
        params.topKWeights == nullptr || params.comm == nullptr || params.yOut == nullptr ||
        params.workspace == nullptr || params.stream == nullptr || params.selfSendCnt < 0 || params.bs <= 0 ||
        params.h <= 0 || params.h > kEpUrmaCombineMaxHidden || params.topK <= 0 ||
        params.topK > kEpUrmaCombineMaxTopK || params.workspaceBytes <= 0 || params.perfTraceBytes < 0 ||
        ((params.perfTrace == nullptr) != (params.perfTraceBytes == 0)) ||
        params.strictKernelCyclesBytes < 0 ||
        ((params.strictKernelCycles == nullptr) != (params.strictKernelCyclesBytes == 0)) ||
        params.dtype != TileXR::TILEXR_DATA_TYPE_FP16) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if ((reinterpret_cast<uintptr_t>(params.workspace) % kEpUrmaCombineWorkspaceAlignment) != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (params.perfTrace != nullptr && (reinterpret_cast<uintptr_t>(params.perfTrace) % 32) != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    constexpr int64_t kStrictKernelCyclesBytes =
        kEpUrmaCombineAivCount * static_cast<int64_t>(sizeof(uint64_t));
    if (params.strictKernelCycles != nullptr && (params.perfTrace != nullptr ||
        params.strictKernelCyclesBytes < kStrictKernelCyclesBytes ||
        (reinterpret_cast<uintptr_t>(params.strictKernelCycles) % 32) != 0)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpPrepareUrmaCombineLaunchContext(
    const EpUrmaCombineParams &params, EpUrmaCombineLaunchContext *context)
{
    if (context == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *context = EpUrmaCombineLaunchContext {};

    int ret = TileXREpValidateBasicUrmaCombineParams(params);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    ret = TileXRGetCommArgsHost(params.comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS || context->hostArgs == nullptr) {
        *context = EpUrmaCombineLaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ? TileXR::TILEXR_ERROR_NOT_INITIALIZED : ret;
    }
    ret = TileXRGetCommArgsDev(params.comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS || context->devArgs == nullptr) {
        *context = EpUrmaCombineLaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ? TileXR::TILEXR_ERROR_NOT_INITIALIZED : ret;
    }

    const TileXR::CommArgs &args = *context->hostArgs;
    if (args.rankSize <= 0 || args.rankSize > TileXR::TILEXR_MAX_RANK_SIZE || args.rank < 0 ||
        args.rank >= args.rankSize || (args.rankSize > 1 &&
            ((args.extraFlag & TileXR::ExtraFlag::UDMA) == 0 || args.udmaInfoPtr == nullptr ||
                args.udmaRegistryPtr == nullptr))) {
        *context = EpUrmaCombineLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    if (params.perfTrace != nullptr) {
        int64_t requiredProfileBytes = 0;
        ret = TileXREpGetUrmaCombineProfileSize(args.rankSize, &requiredProfileBytes);
        if (ret != TileXR::TILEXR_SUCCESS || params.perfTraceBytes < requiredProfileBytes) {
            *context = EpUrmaCombineLaunchContext {};
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
    }

    ret = TileXREpBuildUrmaCombineWorkspaceConfig(
        args.rankSize, params.bs, params.h, params.topK, params.selfSendCnt, &context->workspace);
    if (ret != TileXR::TILEXR_SUCCESS || context->workspace.requiredBytes > params.workspaceBytes) {
        *context = EpUrmaCombineLaunchContext {};
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    if (args.rankSize > 1) {
        const TileXR::TileXRUDMARegistry *registry = nullptr;
        ret = TileXRGetUDMARegistryHost(params.comm, &registry);
        if (ret != TileXR::TILEXR_SUCCESS || !TileXR::UDMARegistryValid(registry, args.rankSize)) {
            *context = EpUrmaCombineLaunchContext {};
            return ret == TileXR::TILEXR_SUCCESS ? TileXR::TILEXR_ERROR_NOT_INITIALIZED : ret;
        }
        if (registry->regions[args.rank].base != static_cast<GM_ADDR>(params.workspace) ||
            !TileXR::UDMARegionContains(registry, args.rank, 0,
                static_cast<uint64_t>(context->workspace.requiredBytes))) {
            *context = EpUrmaCombineLaunchContext {};
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        for (int32_t peer = 0; peer < args.rankSize; ++peer) {
            if (!TileXR::UDMARegionContains(
                    registry, peer, 0, static_cast<uint64_t>(context->workspace.fixedBytes))) {
                *context = EpUrmaCombineLaunchContext {};
                return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
            }
        }
    }
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpLaunchPreparedUrmaCombineKernel(
    const EpUrmaCombineParams &params, const EpUrmaCombineLaunchContext &context, int64_t magic,
    bool runStartGate)
{
    launch_tilexr_ep_urma_combine_kernel(static_cast<uint32_t>(kEpUrmaCombineAivCount),
        params.stream, context.devArgs,
        static_cast<GM_ADDR>(params.expertOut), reinterpret_cast<GM_ADDR>(params.assistInfoForCombine),
        reinterpret_cast<GM_ADDR>(params.topKWeights), static_cast<GM_ADDR>(params.yOut),
        static_cast<GM_ADDR>(params.workspace), params.selfSendCnt, params.bs, params.h, params.topK,
        params.workspaceBytes, magic, context.workspace.commBytes, context.workspace.blockCount,
        context.workspace.routeStride, context.workspace.rxWindowBytes, context.workspace.rxWindowOffsets[0],
        context.workspace.rxWindowOffsets[1], context.workspace.roundDoneOffsets[0],
        context.workspace.roundDoneOffsets[1], context.workspace.rxLaneDoneOffset,
        context.workspace.senderDoneOffset, context.workspace.roundPublishOffset,
        context.workspace.roundCreditOffset,
        context.workspace.startGateOffset, runStartGate ? 1 : 0, context.workspace.errorStatusOffset,
        context.workspace.txReadyOffset, context.workspace.txDataOffset,
        static_cast<GM_ADDR>(params.perfTrace), params.perfTraceBytes,
        static_cast<GM_ADDR>(params.strictKernelCycles));
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpLaunchUrmaCombineKernel(
    const EpUrmaCombineParams &params, const EpUrmaCombineLaunchContext &context)
{
    int64_t magic = 0;
    const int ret = TileXRCommNextMagic(params.comm, &magic);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXREpLaunchPreparedUrmaCombineKernel(params, context, magic, true);
}

} // namespace TileXREp
