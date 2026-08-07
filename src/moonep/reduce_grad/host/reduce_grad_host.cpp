#include "reduce_grad_host.h"

#include "moonep_stage_host.h"

namespace TileXRMoonEp {

int TileXRMoonEpPrepareReduceGradLaunch(const TileXRMoonEpReduceGradArgsV1 *args,
    aclrtStream stream, ReduceGradParams *params, ReduceGradLaunchContext *context)
{
    if (params == nullptr || context == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *params = ReduceGradParams {};
    *context = ReduceGradLaunchContext {};
    if (args == nullptr || args->structSize < sizeof(*args) ||
        args->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || args->comm == nullptr ||
        args->plan == nullptr || args->fullGateGrad == nullptr ||
        args->fullUpGrad == nullptr || args->fullDownGrad == nullptr ||
        args->gateReduceBuffer == nullptr || args->upReduceBuffer == nullptr ||
        args->downReduceBuffer == nullptr || args->flags != TILEXR_MOONEP_FLAG_NONE ||
        stream == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    int ret = PrepareStageHost(args->comm, context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    ret = TileXRMoonEpBuildReduceGradLayout(context->hostArgs->rank,
        context->hostArgs->rankSize, args->plan, args->fullGateGrad,
        args->fullUpGrad, args->fullDownGrad, args->gateReduceBuffer,
        args->upReduceBuffer, args->downReduceBuffer, args->flags, &context->layout);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        *context = ReduceGradLaunchContext {};
        return ret;
    }
    ret = PrepareStageDevice(args->comm, context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }

    params->comm = args->comm;
    params->expertsToCopy = static_cast<const int32_t *>(args->plan->expertsToCopy);
    params->fullGateGrad = args->fullGateGrad->data;
    params->fullUpGrad = args->fullUpGrad->data;
    params->fullDownGrad = args->fullDownGrad->data;
    params->gateReduceBuffer = args->gateReduceBuffer->data;
    params->upReduceBuffer = args->upReduceBuffer->data;
    params->downReduceBuffer = args->downReduceBuffer->data;
    params->status = static_cast<int32_t *>(args->plan->status);
    params->stream = stream;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
