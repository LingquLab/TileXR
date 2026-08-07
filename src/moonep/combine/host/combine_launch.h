#ifndef TILEXR_MOONEP_COMBINE_LAUNCH_H
#define TILEXR_MOONEP_COMBINE_LAUNCH_H

#include "combine_host.h"

namespace TileXRMoonEp {

int TileXRMoonEpLaunchCombineKernel(
    const CombineParams &params, const CombineLaunchContext &context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_LAUNCH_H
