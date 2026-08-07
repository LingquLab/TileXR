#include "dispatch_host.h"

#include "dispatch_launch.h"

namespace TileXRMoonEp {

int TileXRMoonEpRunDispatchV1(
    const TileXRMoonEpDispatchArgsV1 *args, aclrtStream stream)
{
    DispatchParams params {};
    DispatchLaunchContext context {};
    const int ret = TileXRMoonEpPrepareDispatchLaunch(args, stream, &params, &context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    return TileXRMoonEpLaunchDispatchKernel(params, context);
}

} // namespace TileXRMoonEp
