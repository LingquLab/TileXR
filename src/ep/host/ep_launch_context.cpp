#include "ep_dispatch_host.h"

#include "tilexr_types.h"

namespace TileXREp {

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
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
