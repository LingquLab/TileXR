#ifndef TILEXR_MOONEP_REDUCE_GRAD_LAYOUT_H
#define TILEXR_MOONEP_REDUCE_GRAD_LAYOUT_H

#include <cstdint>

#include "reduce_grad_common.h"

namespace TileXRMoonEp {

uint64_t TileXRMoonEpReduceGradPeerWindowBytes();

int TileXRMoonEpBuildReduceGradLayout(int64_t rank, int64_t rankSize,
    int64_t expertCount, int64_t prefetchSlots,
    const uint64_t rowElements[kReduceGradProjectionCount],
    uint64_t peerWindowBytes, uint64_t requestedUdmaChunkBytes,
    ReduceGradLayout *out);

int TileXRMoonEpBuildReduceGradLayout(int64_t rank, int64_t rankSize,
    int64_t expertCount, const uint64_t rowElements[kReduceGradProjectionCount],
    uint64_t peerWindowBytes, uint64_t requestedUdmaChunkBytes,
    ReduceGradLayout *out);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_LAYOUT_H
