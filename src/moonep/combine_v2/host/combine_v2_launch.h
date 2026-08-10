#ifndef TILEXR_MOONEP_COMBINE_V2_LAUNCH_H
#define TILEXR_MOONEP_COMBINE_V2_LAUNCH_H

#include "combine_v2_host.h"

namespace TileXRMoonEp {

int TileXRMoonEpLaunchCombineV2Kernel(
    const CombineV2Params &params,
    const CombineV2LaunchContext &context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_V2_LAUNCH_H
