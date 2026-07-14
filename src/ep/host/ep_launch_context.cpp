#include "ep_dispatch_host.h"

#include <cstdint>

#include "ep_window.h"
#include "tilexr_udma_reg.h"
#include "tilexr_types.h"

namespace TileXREp {
namespace {

bool TileXREpUsesCrossNodeComm(const TileXR::CommArgs &commArgs)
{
    return commArgs.localRankSize > 0 && commArgs.localRankSize < commArgs.rankSize;
}

int64_t TileXREpDispatchWorkspaceBytes(const EpWindowConfig &window, int64_t tpWorldSize)
{
    const int64_t alignedTotal = TileXREpAlignUp(window.totalBytes, kEpWindowAlignmentBytes);
    const int64_t effectiveTpWorldSize = tpWorldSize == 0 ? 1 : tpWorldSize;
    if (alignedTotal == TileXR::TILEXR_INVALID_VALUE || effectiveTpWorldSize <= 0 ||
        effectiveTpWorldSize > INT64_MAX - 2) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    const int64_t factor = effectiveTpWorldSize + 2;
    if (alignedTotal != 0 && factor > INT64_MAX / alignedTotal) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return alignedTotal * factor;
}

int ValidateRegisteredWorkspace(
    TileXRCommPtr comm, const TileXR::CommArgs &commArgs, void *workspace, int64_t requiredBytes)
{
    if (!TileXREpUsesCrossNodeComm(commArgs)) {
        return TileXR::TILEXR_SUCCESS;
    }
    if (workspace == nullptr || requiredBytes <= 0) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    const TileXR::TileXRUDMARegistry *registry = nullptr;
    int ret = TileXRGetUDMARegistryHost(comm, &registry);
    if (ret != TileXR::TILEXR_SUCCESS || !TileXR::UDMARegistryValid(registry, commArgs.rankSize)) {
        return ret == TileXR::TILEXR_SUCCESS ? TileXR::TILEXR_ERROR_NOT_INITIALIZED : ret;
    }
    const uint64_t required = static_cast<uint64_t>(requiredBytes);
    for (int rank = 0; rank < commArgs.rankSize; ++rank) {
        if (!TileXR::UDMARegionContains(registry, rank, 0, required)) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
    }
    if (registry->regions[commArgs.rank].base != static_cast<GM_ADDR>(workspace)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace

int TileXREpPrepareLaunchContext(const EpDispatchParams &params, EpHostLaunchContext *context)
{
    if (context == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *context = EpHostLaunchContext {};

    int ret = TileXRGetCommArgsHost(params.comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (context->hostArgs == nullptr) {
        *context = EpHostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    ret = TileXRGetCommArgsDev(params.comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    if (context->devArgs == nullptr) {
        *context = EpHostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    ret = TileXREpValidateDispatchConfig(params, *context->hostArgs, &context->window);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    const int64_t dispatchWorkspaceBytes = TileXREpDispatchWorkspaceBytes(context->window, params.tpWorldSize);
    ret = ValidateRegisteredWorkspace(params.comm, *context->hostArgs, params.workspace, dispatchWorkspaceBytes);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpPrepareCombineLaunchContext(const EpCombineParams &params, EpHostLaunchContext *context)
{
    if (context == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *context = EpHostLaunchContext {};

    int ret = TileXRGetCommArgsHost(params.comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (context->hostArgs == nullptr) {
        *context = EpHostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    ret = TileXRGetCommArgsDev(params.comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    if (context->devArgs == nullptr) {
        *context = EpHostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    ret = TileXREpValidateCombineConfig(params, *context->hostArgs, &context->window);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    const int64_t combineWorkspaceBytes = TileXREpCombineDataAsFlagRequiredWorkspaceBytes(
        context->window.totalBytes, context->window.rankSize, context->window.slotBytes);
    ret = ValidateRegisteredWorkspace(params.comm, *context->hostArgs, params.workspace, combineWorkspaceBytes);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
