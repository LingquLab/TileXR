#include "ep_dispatch_host.h"

#include <cstdint>

#include "ep_transport_route.h"
#include "ep_window.h"
#include "tilexr_udma_reg.h"
#include "tilexr_types.h"

namespace TileXREp {
namespace {

int ValidateRegisteredWorkspace(
    TileXRCommPtr comm, const TileXR::CommArgs &commArgs, void *workspace, int64_t requiredBytes)
{
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

int ValidateMemoryWindow(int64_t requiredBytes)
{
    if (requiredBytes <= 0 || requiredBytes > TileXR::IPC_BUFF_MAX_SIZE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

int ValidatePeerMems(
    const TileXR::CommArgs &commArgs, TileXR::TileXRTransportKind transport)
{
    int beginRank = 0;
    int endRank = commArgs.rankSize;
    if (transport == TileXR::TileXRTransportKind::DIRECT_URMA) {
        if (commArgs.localRankSize <= 0) {
            return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
        }
        beginRank = (commArgs.rank / commArgs.localRankSize) * commArgs.localRankSize;
        endRank = beginRank + commArgs.localRankSize;
        if (endRank > commArgs.rankSize) {
            endRank = commArgs.rankSize;
        }
    }
    for (int rank = beginRank; rank < endRank; ++rank) {
        if (commArgs.peerMems[rank] == nullptr) {
            return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
        }
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

    ret = TileXREpResolveTransportForWorkspaceFromEnv(*context->hostArgs,
        static_cast<uint64_t>(context->window.totalBytes), params.workspace, &context->transport);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }

    ret = ValidatePeerMems(*context->hostArgs, context->transport);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }

    const int64_t dispatchWorkspaceBytes = TileXREpUdmaRequiredWorkspaceBytes(
        context->window.totalBytes, context->window.rankSize, context->window.slotBytes);
    ret = context->transport == TileXR::TileXRTransportKind::DIRECT_URMA ?
        ValidateRegisteredWorkspace(params.comm, *context->hostArgs, params.workspace, dispatchWorkspaceBytes) :
        ValidateMemoryWindow(context->window.totalBytes);
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

    ret = TileXREpResolveTransportForWorkspaceFromEnv(*context->hostArgs,
        static_cast<uint64_t>(context->window.totalBytes), params.workspace, &context->transport);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }

    ret = ValidatePeerMems(*context->hostArgs, context->transport);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }

    const int64_t combineWorkspaceBytes = TileXREpUdmaRequiredWorkspaceBytes(
        context->window.totalBytes, context->window.rankSize, context->window.slotBytes);
    ret = context->transport == TileXR::TileXRTransportKind::DIRECT_URMA ?
        ValidateRegisteredWorkspace(params.comm, *context->hostArgs, params.workspace, combineWorkspaceBytes) :
        ValidateMemoryWindow(context->window.totalBytes);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = EpHostLaunchContext {};
        return ret;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
