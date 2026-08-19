#include "combine_v2_host.h"

#include "acl/acl_rt.h"
#include "combine_v2_launch.h"
#include "tilexr_types.h"
#include "tilexr_udma_reg.h"

namespace TileXRMoonEp {
namespace {

bool CombineV2RankSizeSupported(int rankSize)
{
    return rankSize >= 0 && MoonEpCombineV2RankSizeSupported(
        static_cast<uint32_t>(rankSize));
}

int CombineV2ExpectedLocalRankSize(int rankSize)
{
    return rankSize >= 0 ? static_cast<int>(MoonEpCombineV2LocalRankSize(
        static_cast<uint32_t>(rankSize))) : 0;
}

int ValidateAivCoreCount(uint32_t aivCoreNum)
{
    if (aivCoreNum != kMoonEpCombineV2CoreCount) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    int32_t deviceId = 0;
    if (aclrtGetDevice(&deviceId) != ACL_SUCCESS) {
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }
    int64_t vectorCoreCount = 0;
    if (aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId),
            ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreCount) != ACL_SUCCESS ||
        vectorCoreCount <= 0 ||
        static_cast<uint64_t>(vectorCoreCount) <
            kMoonEpCombineV2CoreCount) {
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }
    return TILEXR_MOONEP_SUCCESS;
}

int ValidateRegisteredWorkspace(const CombineV2Params &params,
    const CombineV2LaunchContext &context)
{
    const TileXR::TileXRUDMARegistry *registry = nullptr;
    const int ret = TileXRGetUDMARegistryHost(params.comm, &registry);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (!TileXR::UDMARegistryValid(registry, context.hostArgs->rankSize) ||
        registry->regions[context.hostArgs->rank].base !=
            static_cast<GM_ADDR>(params.registeredWorkspace)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    for (int rank = 0; rank < context.hostArgs->rankSize; ++rank) {
        if (!TileXR::UDMARegionContains(
                registry, rank, 0, context.layout.totalBytes)) {
            return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
        }
    }
    return TILEXR_MOONEP_SUCCESS;
}

int ValidateFullmeshCapability(const CombineV2Params &params,
    const TileXR::CommArgs &commArgs)
{
    if ((commArgs.extraFlag & TileXR::ExtraFlag::UDMA_FULLMESH) == 0U ||
        commArgs.udmaFullmeshPtr == nullptr ||
        commArgs.udmaRegistrationGeneration == 0U) {
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }
    TileXR::TileXRUDMAFullmeshHostView view {};
    const int ret = TileXRUDMAFullmeshQuery(params.comm, &view);
    if (ret != TileXR::TILEXR_SUCCESS ||
        !TileXR::UDMAFullmeshHostViewValid(view,
            static_cast<uint32_t>(commArgs.localRank),
            static_cast<uint32_t>(commArgs.localRankSize),
            commArgs.udmaRegistrationGeneration) ||
        view.viewDev != commArgs.udmaFullmeshPtr) {
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }
    return TILEXR_MOONEP_SUCCESS;
}

int ValidateWeightMemory(const CombineV2Params &params,
    CombineV2LaunchContext *context)
{
    const bool hasWeights = params.routeWeightsNvs != nullptr;
    if ((params.routeWeightsNvs == nullptr) !=
            (params.routeWeightsSk == nullptr)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if (!hasWeights) {
        return TILEXR_MOONEP_SUCCESS;
    }
    if (params.weightMemoryComm == nullptr ||
        params.weightMemoryComm == params.comm) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    TileXR::CommArgs *memoryArgs = nullptr;
    int ret = TileXRGetCommArgsHost(params.weightMemoryComm, memoryArgs);
    if (ret != TileXR::TILEXR_SUCCESS || memoryArgs == nullptr) {
        return ret == TileXR::TILEXR_SUCCESS ?
            TILEXR_MOONEP_ERROR_INTERNAL : ret;
    }
    const TileXR::CommArgs &hiddenArgs = *context->hostArgs;
    if (memoryArgs->rank != hiddenArgs.rank ||
        memoryArgs->rankSize != hiddenArgs.rankSize ||
        memoryArgs->localRank != hiddenArgs.localRank ||
        memoryArgs->localRankSize != hiddenArgs.localRankSize ||
        memoryArgs->commDomain == hiddenArgs.commDomain ||
        (memoryArgs->extraFlag & TileXR::ExtraFlag::MEMORY_ONLY) == 0U ||
        (memoryArgs->extraFlag & (TileXR::ExtraFlag::UDMA |
            TileXR::ExtraFlag::UDMA_SHARED_QP |
            TileXR::ExtraFlag::UDMA_FULLMESH |
            TileXR::ExtraFlag::SDMA)) != 0U ||
        memoryArgs->peerMemBytes <=
            static_cast<uint64_t>(TileXR::IPC_DATA_OFFSET)) {
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }
    for (int peer = 0; peer < memoryArgs->rankSize; ++peer) {
        if (memoryArgs->peerMems[peer] == nullptr) {
            return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
        }
    }

    ret = TileXRMoonEpBuildCombineV2WeightLayout(
        params.nvS, memoryArgs->rankSize, &context->weightLayout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    const uint64_t dataCapacity = memoryArgs->peerMemBytes -
        static_cast<uint64_t>(TileXR::IPC_DATA_OFFSET);
    if (context->weightLayout.totalBytes > dataCapacity) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t bs = static_cast<uint64_t>(params.bs);
    const uint64_t topK = static_cast<uint64_t>(params.topK);
    if (bs != 0U && topK > UINT64_MAX / bs) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    context->weightOutputElements = bs * topK;
    context->weightMemoryHostArgs = memoryArgs;
    ret = TileXRGetCommArgsDev(
        params.weightMemoryComm, context->weightMemoryDevArgs);
    if (ret != TileXR::TILEXR_SUCCESS ||
        context->weightMemoryDevArgs == nullptr) {
        return ret == TileXR::TILEXR_SUCCESS ?
            TILEXR_MOONEP_ERROR_INTERNAL : ret;
    }
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace

int TileXRMoonEpPrepareCombineV2Launch(
    const CombineV2Params &params, CombineV2LaunchContext *context)
{
    if (context == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *context = CombineV2LaunchContext {};
    if (params.registeredWorkspace == nullptr || params.dstLocal == nullptr ||
        params.comm == nullptr || params.activeOutputOffset == nullptr ||
        params.stream == nullptr ||
        params.aivCoreNum != kMoonEpCombineV2CoreCount) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if ((params.routeWeightsNvs == nullptr) !=
            (params.routeWeightsSk == nullptr)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    int ret = TileXRMoonEpBuildCombineV2Layout(params.bs, params.h,
        params.topK, params.nvS, params.dtype, &context->layout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    ret = TileXRGetCommArgsHost(params.comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS || context->hostArgs == nullptr) {
        *context = CombineV2LaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ?
            TILEXR_MOONEP_ERROR_INTERNAL : ret;
    }

    const TileXR::CommArgs &commArgs = *context->hostArgs;
    const int expectedLocalRankSize =
        CombineV2ExpectedLocalRankSize(commArgs.rankSize);
    if (!CombineV2RankSizeSupported(commArgs.rankSize) ||
        commArgs.rank < 0 || commArgs.rank >= commArgs.rankSize ||
        commArgs.localRank < 0 ||
        commArgs.localRankSize != expectedLocalRankSize ||
        commArgs.localRank >= commArgs.localRankSize ||
        commArgs.rank % commArgs.localRankSize != commArgs.localRank) {
        *context = CombineV2LaunchContext {};
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    for (int peer = 0; peer < commArgs.rankSize; ++peer) {
        if (commArgs.creditMems[peer] == nullptr) {
            *context = CombineV2LaunchContext {};
            return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
        }
    }
    if ((commArgs.extraFlag & TileXR::ExtraFlag::UDMA) == 0 ||
        (commArgs.extraFlag & TileXR::ExtraFlag::UDMA_SHARED_QP) == 0 ||
        commArgs.udmaInfoPtr == nullptr || commArgs.udmaRegistryPtr == nullptr) {
        *context = CombineV2LaunchContext {};
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }

    uint32_t qpCount = 0;
    ret = TileXRUDMAGetQpCount(params.comm, &qpCount);
    if (ret != TileXR::TILEXR_SUCCESS ||
        qpCount != kMoonEpCombineV2QpCount) {
        *context = CombineV2LaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ?
            TILEXR_MOONEP_ERROR_NOT_SUPPORTED : ret;
    }
    ret = ValidateFullmeshCapability(params, commArgs);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = CombineV2LaunchContext {};
        return ret;
    }
    ret = ValidateRegisteredWorkspace(params, *context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = CombineV2LaunchContext {};
        return ret;
    }
    ret = ValidateAivCoreCount(params.aivCoreNum);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = CombineV2LaunchContext {};
        return ret;
    }
    ret = ValidateWeightMemory(params, context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = CombineV2LaunchContext {};
        return ret;
    }
    ret = TileXRGetCommArgsDev(params.comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS || context->devArgs == nullptr) {
        *context = CombineV2LaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ?
            TILEXR_MOONEP_ERROR_INTERNAL : ret;
    }
    ret = TileXRCommNextMagic(params.comm, &context->magic);
    if (ret != TileXR::TILEXR_SUCCESS || context->magic <= 0 ||
        !MoonEpCombineV2MagicValid(static_cast<uint64_t>(context->magic))) {
        *context = CombineV2LaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ?
            TILEXR_MOONEP_ERROR_INTERNAL : ret;
    }
    return TILEXR_MOONEP_SUCCESS;
}

int TileXRMoonEpRunCombineV2(const CombineV2Params &params)
{
    CombineV2LaunchContext context {};
    const int ret = TileXRMoonEpPrepareCombineV2Launch(params, &context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    return TileXRMoonEpLaunchCombineV2Kernel(params, context);
}

} // namespace TileXRMoonEp
