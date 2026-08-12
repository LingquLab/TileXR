#include "prefetch_weight_host.h"

#include "prefetch_weight_launch.h"

namespace TileXRMoonEp {

int TileXRMoonEpRunPrefetchWeightV1(
    const TileXRMoonEpPrefetchWeightArgsV1 *args, aclrtStream stream)
{
    PrefetchWeightParams params {};
    PrefetchWeightLaunchContext context {};
    const int ret = TileXRMoonEpPreparePrefetchWeightLaunch(args, stream, &params, &context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    return TileXRMoonEpLaunchPrefetchWeightKernel(params, context);
}

} // namespace TileXRMoonEp
