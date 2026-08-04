#ifndef TILEXR_MOONEP_PREFETCH_WEIGHT_LAUNCH_H
#define TILEXR_MOONEP_PREFETCH_WEIGHT_LAUNCH_H

#include "acl/acl_base.h"
#include "prefetch_weight_layout.h"

namespace TileXRMoonEp {

int LaunchPrefetchWeight(const PrefetchWeightLayout &layout, GM_ADDR commArgs,
    GM_ADDR expertsToCopy, GM_ADDR status, aclrtStream stream);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_LAUNCH_H
