#ifndef TILEXR_MOONEP_COMBINE_V2_H
#define TILEXR_MOONEP_COMBINE_V2_H

#include <stdint.h>

#include "tilexr_moonep.h"

#ifdef __cplusplus
extern "C" {
#endif

int TileXRMoonEpCombineGetWorkspaceSizeV2(
    int64_t bs, int64_t h, int64_t topK, int64_t nvS, uint32_t dtype,
    uint64_t *workspaceBytes, uint64_t *profileOffset,
    uint64_t *outputEpoch0Offset, uint64_t *outputEpoch1Offset);

// Supports rank sizes 2 through 8, 16, 32, 64, and 128. For 2 through 8
// ranks, localRankSize must equal rankSize; larger configurations require
// localRankSize == 8. The shared-QP communicator must expose exactly 32 QPs.
// aivCoreNum is the Runtime launch block dimension and must be at least 16 and
// no greater than the vector-core count reported by the current device.
int TileXRMoonEpCombineV2(void *registeredWorkspace,
    const int32_t *dstLocal, TileXRCommPtr comm, int64_t bs, int64_t h,
    int64_t topK, int64_t nvS, uint32_t aivCoreNum,
    uint64_t *activeOutputOffset,
    uint32_t dtype, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // TILEXR_MOONEP_COMBINE_V2_H
