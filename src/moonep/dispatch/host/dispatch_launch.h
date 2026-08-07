#ifndef TILEXR_MOONEP_DISPATCH_LAUNCH_H
#define TILEXR_MOONEP_DISPATCH_LAUNCH_H

#include "dispatch_host.h"

namespace TileXRMoonEp {

int TileXRMoonEpLaunchDispatchKernel(
    const DispatchParams &params, const DispatchLaunchContext &context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_DISPATCH_LAUNCH_H
