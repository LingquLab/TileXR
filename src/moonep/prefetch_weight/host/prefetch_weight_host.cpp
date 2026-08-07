#include "prefetch_weight_host.h"

#include "moonep_stage_host.h"

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
        args->plan == nullptr || args->fullGateWeight == nullptr ||
        args->fullUpWeight == nullptr || args->fullDownWeight == nullptr ||
        args->flags != TILEXR_MOONEP_FLAG_NONE || stream == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    int ret = PrepareStageHost(args->comm, context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    ret = TileXRMoonEpBuildPrefetchWeightLayout(context->hostArgs->rank,
        context->hostArgs->rankSize, args->plan, args->fullGateWeight,
        args->fullUpWeight, args->fullDownWeight, args->flags, &context->layout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = PrefetchWeightLaunchContext {};
        return ret;
    }
    ret = PrepareStageDevice(args->comm, context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }

    params->comm = args->comm;
    params->expertsToCopy = static_cast<const int32_t *>(args->plan->expertsToCopy);
    params->fullGateWeight = args->fullGateWeight->data;
    params->fullUpWeight = args->fullUpWeight->data;
    params->fullDownWeight = args->fullDownWeight->data;
    params->status = static_cast<int32_t *>(args->plan->status);
    params->stream = stream;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
