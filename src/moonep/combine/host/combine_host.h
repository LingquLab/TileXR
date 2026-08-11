#ifndef TILEXR_MOONEP_COMBINE_HOST_H
#define TILEXR_MOONEP_COMBINE_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "combine_layout.h"
#include "tilexr_api.h"

namespace TileXRMoonEp {

struct CombineParams {
    TileXRCommPtr comm = nullptr;
    const int32_t *dstLocal = nullptr;
    const int32_t *dst = nullptr;
    const int32_t *dupGroups = nullptr;
    const int32_t *dupLoffs = nullptr;
    const int32_t *dupCounts = nullptr;
    const void *hiddenNvsh = nullptr;
    const float *routeWeightsNvs = nullptr;
    void *hiddenSh = nullptr;
    float *routeWeightsSk = nullptr;
    int32_t *status = nullptr;
    uint64_t flags = 0;
    aclrtStream stream = nullptr;
};

struct CombineLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    CombineLayout layout {};
    uint64_t waitIterations = 0;
    int64_t magic = 0;
};

int TileXRMoonEpPrepareCombineLaunch(const TileXRMoonEpCombineArgsV1 *args,
    aclrtStream stream, CombineParams *params, CombineLaunchContext *context);

int TileXRMoonEpRunCombineV1(
    const TileXRMoonEpCombineArgsV1 *args, aclrtStream stream);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_HOST_H
