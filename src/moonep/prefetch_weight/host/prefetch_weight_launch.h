#ifndef TILEXR_MOONEP_PREFETCH_WEIGHT_LAUNCH_H
#define TILEXR_MOONEP_PREFETCH_WEIGHT_LAUNCH_H

#include "prefetch_weight_host.h"

namespace TileXRMoonEp {

int TileXRMoonEpLaunchPrefetchWeightKernel(
    const PrefetchWeightParams &params, const PrefetchWeightLaunchContext &context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_LAUNCH_H
