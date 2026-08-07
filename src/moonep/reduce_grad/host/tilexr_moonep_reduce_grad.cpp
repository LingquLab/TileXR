#include "reduce_grad_host.h"

#include "reduce_grad_launch.h"

namespace TileXRMoonEp {

int TileXRMoonEpRunReduceGradV1(
    const TileXRMoonEpReduceGradArgsV1 *args, aclrtStream stream)
{
    ReduceGradParams params {};
    ReduceGradLaunchContext context {};
    const int ret = TileXRMoonEpPrepareReduceGradLaunch(args, stream, &params, &context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    return TileXRMoonEpLaunchReduceGradKernel(params, context);
}

} // namespace TileXRMoonEp
