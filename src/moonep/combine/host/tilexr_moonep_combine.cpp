#include "combine_host.h"

#include "combine_launch.h"

namespace TileXRMoonEp {

int TileXRMoonEpRunCombineV1(
    const TileXRMoonEpCombineArgsV1 *args, aclrtStream stream)
{
    CombineParams params {};
    CombineLaunchContext context {};
    const int ret = TileXRMoonEpPrepareCombineLaunch(args, stream, &params, &context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    return TileXRMoonEpLaunchCombineKernel(params, context);
}

} // namespace TileXRMoonEp
