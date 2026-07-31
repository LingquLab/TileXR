#ifndef TILEXR_EP_HOST_EP_URMA_COMBINE_LAYOUT_H
#define TILEXR_EP_HOST_EP_URMA_COMBINE_LAYOUT_H

#include <cstdint>

namespace TileXREp {

struct EpUrmaCombineWorkspaceConfig {
    int64_t rankSize = 0;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t topK = 0;
    int64_t selfSendCnt = 0;
    int64_t quantDataBytes = 0;
    int64_t commBytes = 0;
    int64_t blockCount = 0;
    int64_t routeStride = 0;
    int64_t routeCount = 0;
    int64_t rxWindowBytes = 0;
    int64_t rxWindowOffsets[2] = {0, 0};
    int64_t roundDoneOffsets[2] = {0, 0};
    int64_t rxLaneDoneOffset = 0;
    int64_t senderDoneOffset = 0;
    int64_t roundPublishOffset = 0;
    int64_t roundCreditOffset = 0;
    int64_t startGateOffset = 0;
    int64_t errorStatusOffset = 0;
    int64_t fixedBytes = 0;
    int64_t txReadyOffset = 0;
    int64_t txDataOffset = 0;
    int64_t requiredBytes = 0;
};

int TileXREpBuildUrmaCombineWorkspaceConfig(int64_t rankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t selfSendCnt, EpUrmaCombineWorkspaceConfig *out);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_URMA_COMBINE_LAYOUT_H
