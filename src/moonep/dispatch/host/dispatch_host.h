#ifndef TILEXR_MOONEP_DISPATCH_HOST_H
#define TILEXR_MOONEP_DISPATCH_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "dispatch_layout.h"
#include "tilexr_api.h"

namespace TileXRMoonEp {

struct DispatchParams {
    TileXRCommPtr comm = nullptr;
    const int32_t *dst = nullptr;
    const int32_t *zeroFillRanges = nullptr;
    int32_t *dupGroups = nullptr;
    int32_t *dupLoffs = nullptr;
    int32_t *dupCounts = nullptr;
    const void *hiddenSh = nullptr;
    const float *routeWeightsSk = nullptr;
    void *hiddenNvsh = nullptr;
    float *routeWeightsNvs = nullptr;
    int32_t *status = nullptr;
    uint64_t flags = 0;
    aclrtStream stream = nullptr;
};

struct DispatchLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    DispatchLayout layout {};
    uint64_t waitIterations = 0;
    int64_t magic = 0;
};

int TileXRMoonEpPrepareDispatchLaunch(const TileXRMoonEpDispatchArgsV1 *args,
    aclrtStream stream, DispatchParams *params, DispatchLaunchContext *context);

int TileXRMoonEpRunDispatchV1(
    const TileXRMoonEpDispatchArgsV1 *args, aclrtStream stream);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_DISPATCH_HOST_H
