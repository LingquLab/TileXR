#include "dispatch_host.h"

#include "moonep_stage_host.h"

namespace TileXRMoonEp {

int TileXRMoonEpPrepareDispatchLaunch(const TileXRMoonEpDispatchArgsV1 *args,
    aclrtStream stream, DispatchParams *params, DispatchLaunchContext *context)
{
    if (params == nullptr || context == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *params = DispatchParams {};
    *context = DispatchLaunchContext {};
    if (args == nullptr || args->structSize < sizeof(*args) ||
        args->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || args->comm == nullptr ||
        args->plan == nullptr || args->hiddenSh == nullptr || args->hiddenNvsh == nullptr ||
        stream == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if ((args->flags & TILEXR_MOONEP_FLAG_ZERO_COPY) != 0) {
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }

    int ret = PrepareStageHost(args->comm, context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    ret = TileXRMoonEpBuildDispatchLayout(context->hostArgs->rank,
        context->hostArgs->rankSize, args->plan, args->hiddenSh, args->routeWeightsSk,
        args->hiddenNvsh, args->routeWeightsNvs, args->flags, &context->layout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = DispatchLaunchContext {};
        return ret;
    }

    ret = PrepareStageDevice(args->comm, context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }

    params->comm = args->comm;
    params->dst = static_cast<const int32_t *>(args->plan->dst);
    params->zeroFillRanges = static_cast<const int32_t *>(args->plan->zeroFillRanges);
    params->dupGroups = static_cast<int32_t *>(args->plan->dupGroups);
    params->dupLoffs = static_cast<int32_t *>(args->plan->dupLoffs);
    params->dupCounts = static_cast<int32_t *>(args->plan->dupCounts);
    params->hiddenSh = args->hiddenSh->data;
    params->routeWeightsSk = args->routeWeightsSk == nullptr ? nullptr :
        static_cast<const float *>(args->routeWeightsSk->data);
    params->hiddenNvsh = args->hiddenNvsh->data;
    params->routeWeightsNvs = args->routeWeightsNvs == nullptr ? nullptr :
        static_cast<float *>(args->routeWeightsNvs->data);
    params->status = static_cast<int32_t *>(args->plan->status);
    params->flags = args->flags;
    params->stream = stream;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
