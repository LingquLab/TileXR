#ifndef TILEXR_MOONEP_PREFETCH_WEIGHT_HOST_H
#define TILEXR_MOONEP_PREFETCH_WEIGHT_HOST_H

#include "acl/acl_base.h"
#include "prefetch_weight_layout.h"
#include "tilexr_api.h"

namespace TileXRMoonEp {

struct PrefetchWeightParams {
    TileXRCommPtr comm = nullptr;
    const int32_t *expertsToCopy = nullptr;
    void *fullGateWeight = nullptr;
    void *fullUpWeight = nullptr;
    void *fullDownWeight = nullptr;
    int32_t *status = nullptr;
    aclrtStream stream = nullptr;
};

struct PrefetchWeightLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    PrefetchWeightLayout layout {};
    uint64_t waitIterations = 0;
    int64_t magic = 0;
};

int TileXRMoonEpPreparePrefetchWeightLaunch(
    const TileXRMoonEpPrefetchWeightArgsV1 *args, aclrtStream stream,
    PrefetchWeightParams *params, PrefetchWeightLaunchContext *context);

int TileXRMoonEpRunPrefetchWeightV1(
    const TileXRMoonEpPrefetchWeightArgsV1 *args, aclrtStream stream);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_HOST_H
