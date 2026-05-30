#ifndef TILEXR_EP_HOST_EP_LAYOUT_H
#define TILEXR_EP_HOST_EP_LAYOUT_H

#include <cstdint>

#include "tilexr_types.h"

namespace TileXREp {

struct EpWindowConfig {
    int64_t rankSize = 0;
    int64_t localExpertNum = 0;
    int64_t maxRoutesPerSrc = 0;
    int64_t topK = 0;
    int64_t hiddenSize = 0;
    int64_t dtypeBytes = 0;
    int64_t rowBytes = 0;
    int64_t payloadBytesPerSlot = 0;
    int64_t assistBytesPerSlot = 0;
    int64_t slotBytes = 0;
    int64_t totalBytes = 0;
};

int64_t TileXREpAlignUp(int64_t value, int64_t alignment);
int64_t TileXREpDataTypeSize(TileXR::TileXRDataType dtype);
bool TileXREpIsSupportedDataType(TileXR::TileXRDataType dtype);

int TileXREpBuildWindowConfig(int64_t rankSize, int64_t maxTokensPerSrc, int64_t hiddenSize, int64_t topK,
    int64_t expertNum, TileXR::TileXRDataType dtype, EpWindowConfig *config);

int TileXREpDstRank(int expertId, int localExpertNum);
int TileXREpLocalExpert(int expertId, int localExpertNum);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_LAYOUT_H
