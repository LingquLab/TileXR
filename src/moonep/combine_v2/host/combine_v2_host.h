#ifndef TILEXR_MOONEP_COMBINE_V2_HOST_H
#define TILEXR_MOONEP_COMBINE_V2_HOST_H

#include "combine_v2_layout.h"
#include "tilexr_api.h"
#include "tilexr_moonep_combine_v2.h"

namespace TileXRMoonEp {

struct CombineV2Params {
    void *registeredWorkspace = nullptr;
    const int32_t *dstLocal = nullptr;
    TileXRCommPtr comm = nullptr;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t topK = 0;
    int64_t nvS = 0;
    uint32_t aivCoreNum = 0;
    uint64_t *activeOutputOffset = nullptr;
    uint32_t dtype = 0;
    bool reduceHidden = false;
    aclrtStream stream = nullptr;
};

struct CombineV2LaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    CombineV2Layout layout {};
    int64_t magic = 0;
};

int TileXRMoonEpPrepareCombineV2Launch(
    const CombineV2Params &params, CombineV2LaunchContext *context);

int TileXRMoonEpRunCombineV2(const CombineV2Params &params);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_V2_HOST_H
