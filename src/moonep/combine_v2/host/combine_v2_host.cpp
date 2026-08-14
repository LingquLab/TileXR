#include "combine_v2_host.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "acl/acl_rt.h"
#include "combine_v2_launch.h"
#include "tilexr_types.h"
#include "tilexr_udma_reg.h"

namespace TileXRMoonEp {
namespace {

constexpr const char *kCombineV2FullSyncEnv =
    "TILEXR_MOONEP_COMBINE_V2_FULL_SYNC";

int ResolveCombineV2FullSync(bool *enabled)
{
    if (enabled == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const char *value = std::getenv(kCombineV2FullSyncEnv);
    if (value == nullptr || value[0] == '\0') {
        *enabled = true;
        return TILEXR_MOONEP_SUCCESS;
    }
    const std::string text(value);
    if (text == "1" || text == "true" || text == "TRUE" ||
        text == "on" || text == "ON") {
        *enabled = true;
        return TILEXR_MOONEP_SUCCESS;
    }
    if (text == "0" || text == "false" || text == "FALSE" ||
        text == "off" || text == "OFF") {
        *enabled = false;
        return TILEXR_MOONEP_SUCCESS;
    }
    std::cerr << kCombineV2FullSyncEnv << " has invalid value '" << text
              << "'" << std::endl;
    return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
}

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
    int32_t deviceId = 0;
    if (aclrtGetDevice(&deviceId) != ACL_SUCCESS) {
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }
    int64_t vectorCoreCount = 0;
    if (aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId),
            ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreCount) != ACL_SUCCESS ||
        vectorCoreCount <= 0 ||
        static_cast<uint64_t>(vectorCoreCount) < aivCoreNum) {
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
        params.aivCoreNum < kMoonEpCombineV2CoreCount) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    int ret = TileXRMoonEpBuildCombineV2Layout(params.bs, params.h,
        params.topK, params.nvS, params.dtype, &context->layout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    bool fullSyncEnabled = false;
    if (params.reduceHidden) {
        ret = ResolveCombineV2FullSync(&fullSyncEnabled);
        if (ret != TILEXR_MOONEP_SUCCESS) {
            *context = CombineV2LaunchContext {};
            return ret;
        }
    }
    context->fullSync = params.reduceHidden && fullSyncEnabled;
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
