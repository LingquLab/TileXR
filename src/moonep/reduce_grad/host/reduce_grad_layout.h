#ifndef TILEXR_MOONEP_REDUCE_GRAD_LAYOUT_H
#define TILEXR_MOONEP_REDUCE_GRAD_LAYOUT_H

#include <cstdint>

#include "tilexr_moonep.h"

namespace TileXRMoonEp {

struct ReduceGradProjectionLayout {
    int64_t rowBytes = 0;
    int64_t chunkBytes = 0;
    int64_t chunkStride = 0;
    int64_t chunkCount = 0;
    uint64_t payloadBytes = 0;
};

struct ReduceGradLayout {
    int64_t rank = 0;
    int64_t world = 0;
    int64_t e = 0;
    int64_t b = 0;
    int64_t expertsPerRank = 0;
    int64_t blockDim = 0;
    int64_t iterationCount = 0;
    ReduceGradProjectionLayout gate {};
    ReduceGradProjectionLayout up {};
    ReduceGradProjectionLayout down {};
};

int TileXRMoonEpBuildReduceGradLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *fullGateGrad,
    const TileXRMoonEpTensorV1 *fullUpGrad,
    const TileXRMoonEpTensorV1 *fullDownGrad,
    const TileXRMoonEpTensorV1 *gateReduceBuffer,
    const TileXRMoonEpTensorV1 *upReduceBuffer,
    const TileXRMoonEpTensorV1 *downReduceBuffer,
    uint64_t flags, ReduceGradLayout *layout);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_LAYOUT_H
