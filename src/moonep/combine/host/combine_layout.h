#ifndef TILEXR_MOONEP_COMBINE_LAYOUT_H
#define TILEXR_MOONEP_COMBINE_LAYOUT_H

#include <cstdint>

#include "tilexr_moonep.h"

namespace TileXRMoonEp {

struct CombineLayout {
    int64_t rank = 0;
    int64_t world = 0;
    int64_t s = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t nvS = 0;
    int64_t hiddenSize = 0;
    int64_t blockDim = 0;
    int64_t stepCount = 0;
    int64_t sourcesPerCore = 0;
    int64_t chunkCount = 0;
    uint64_t flags = 0;
    uint64_t hiddenRowBytes = 0;
    uint64_t hiddenChunkBytes = 0;
    uint64_t hiddenChunkStride = 0;
    uint64_t sourceHiddenOffset = 0;
    uint64_t hiddenPayloadBytes = 0;
    uint64_t receiveHiddenOffset = 0;
    uint64_t sourceWeightsOffset = 0;
    uint64_t receiveWeightsOffset = 0;
    uint64_t routeWeightsBytes = 0;
    uint64_t duplicateMaskOffset = 0;
    uint64_t duplicateMaskBytes = 0;
    uint64_t doneOffset = 0;
    uint64_t doneBytes = 0;
    uint64_t coreStatusOffset = 0;
    uint64_t coreStatusBytes = 0;
    uint64_t windowBytes = 0;
};

int TileXRMoonEpBuildCombineLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan, const TileXRMoonEpTensorV1 *hiddenNvsh,
    const TileXRMoonEpTensorV1 *routeWeightsNvs, TileXRMoonEpTensorV1 *hiddenSh,
    TileXRMoonEpTensorV1 *routeWeightsSk, uint64_t flags, CombineLayout *layout);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_LAYOUT_H
