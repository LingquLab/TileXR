#include "prefetch_weight_host.h"

#include <cstdlib>

#include "tilexr_types.h"

namespace TileXRMoonEp {

int TileXRMoonEpPreparePrefetchWeightLaunch(
    const TileXRMoonEpPrefetchWeightArgsV1 *args, aclrtStream stream,
    PrefetchWeightParams *params, PrefetchWeightLaunchContext *context)
{
    if (params == nullptr || context == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *params = PrefetchWeightParams {};
    *context = PrefetchWeightLaunchContext {};
    if (args == nullptr || args->structSize < sizeof(*args) ||
        args->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || args->comm == nullptr ||
        args->plan == nullptr || args->gate == nullptr ||
        args->up == nullptr || args->down == nullptr ||
        args->flags != TILEXR_MOONEP_FLAG_NONE || stream == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    int ret = TileXRGetCommArgsHost(args->comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (context->hostArgs == nullptr || context->hostArgs->rankSize <= 0 ||
        context->hostArgs->rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        context->hostArgs->rank < 0 ||
        context->hostArgs->rank >= context->hostArgs->rankSize) {
        *context = PrefetchWeightLaunchContext {};
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if ((context->hostArgs->extraFlag & TileXR::ExtraFlag::TOPO_910A5) == 0) {
        *context = PrefetchWeightLaunchContext {};
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }

    const TileXR::TileXRUDMARegistry *registry = nullptr;
    uint32_t qpNum = 0;
    if (TileXRGetUDMARegistryHost(args->comm, &registry) != TileXR::TILEXR_SUCCESS ||
        TileXRUDMAGetQpCount(args->comm, &qpNum) != TileXR::TILEXR_SUCCESS ||
        registry == nullptr) {
        *context = PrefetchWeightLaunchContext {};
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }
    ret = TileXRMoonEpBuildPrefetchWeightLayout(*args, *context->hostArgs,
        *registry, qpNum, std::getenv("TILEXR_MOONEP_PREFETCH_BLOCK_DIM"),
        &context->layout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = PrefetchWeightLaunchContext {};
        return ret;
    }
    ret = TileXRGetCommArgsDev(args->comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (context->devArgs == nullptr) {
        *context = PrefetchWeightLaunchContext {};
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }

    params->comm = args->comm;
    params->expertsToCopy = static_cast<const int32_t *>(args->plan->expertsToCopy);
    params->gate = args->gate->data;
    params->up = args->up->data;
    params->down = args->down->data;
    params->status = static_cast<int32_t *>(args->plan->status);
    params->stream = stream;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
