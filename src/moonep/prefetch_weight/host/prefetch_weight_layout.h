#ifndef TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H
#define TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H

#include <cstdint>

#include "tilexr_moonep.h"

namespace TileXRMoonEp {

struct PrefetchProjectionLayout {
    int64_t rowBytes = 0;
    int64_t chunkBytes = 0;
    int64_t chunkCount = 0;
};

struct PrefetchWeightLayout {
    int64_t rank = 0;
    int64_t world = 0;
    int64_t e = 0;
    int64_t b = 0;
    int64_t expertsPerRank = 0;
    int64_t blockDim = 0;
    int64_t iterationCount = 0;
    PrefetchProjectionLayout gate {};
    PrefetchProjectionLayout up {};
    PrefetchProjectionLayout down {};
};

int TileXRMoonEpBuildPrefetchWeightLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan, const TileXRMoonEpTensorV1 *fullGateWeight,
    const TileXRMoonEpTensorV1 *fullUpWeight,
    const TileXRMoonEpTensorV1 *fullDownWeight, uint64_t flags,
    PrefetchWeightLayout *layout);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H
