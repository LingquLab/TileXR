#ifndef TILEXR_MOONEP_DISPATCH_LAYOUT_H
#define TILEXR_MOONEP_DISPATCH_LAYOUT_H

#include <cstdint>

#include "tilexr_moonep.h"

namespace TileXRMoonEp {

struct DispatchLayout {
    int64_t rank = 0;
    int64_t world = 0;
    int64_t s = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t nvS = 0;
    int64_t hiddenSize = 0;
    int64_t blockDim = 0;
    int64_t chunkCount = 0;
    uint64_t flags = 0;
    uint64_t hiddenRowBytes = 0;
    uint64_t hiddenChunkBytes = 0;
    uint64_t hiddenChunkStride = 0;
    uint64_t hiddenPayloadBytes = 0;
    uint64_t routeWeightsOffset = 0;
    uint64_t routeWeightsBytes = 0;
    uint64_t dedupParentsOffset = 0;
    uint64_t dedupParentsBytes = 0;
    uint64_t dedupGroupMapOffset = 0;
    uint64_t dedupGroupMapBytes = 0;
    uint64_t windowBytes = 0;
};

uint64_t TileXRMoonEpAlignDispatchBytes(uint64_t value);

int TileXRMoonEpBuildDispatchLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan, const TileXRMoonEpTensorV1 *hiddenSh,
    const TileXRMoonEpTensorV1 *routeWeightsSk, TileXRMoonEpTensorV1 *hiddenNvsh,
    TileXRMoonEpTensorV1 *routeWeightsNvs, uint64_t flags, DispatchLayout *layout);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_DISPATCH_LAYOUT_H
