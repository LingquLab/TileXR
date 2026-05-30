#include "ep_kernel_launch.h"

#include "tilexr_types.h"

namespace TileXREp {

int TileXREpLaunchDispatchKernel(const EpDispatchParams &params, const EpHostLaunchContext &context)
{
    (void)params;
    (void)context;
    return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
}

} // namespace TileXREp
