#ifndef TILEXR_MOONEP_COMBINE_COMMON_H
#define TILEXR_MOONEP_COMBINE_COMMON_H

#include "moonep_peer_window.h"

namespace TileXRMoonEp {

constexpr int64_t kMoonEpCombineHiddenTileElements = 19648;
constexpr uint32_t kMoonEpCombineBfloatScratchBytes =
    kMoonEpCombineHiddenTileElements * sizeof(uint16_t);
constexpr uint32_t kMoonEpCombineFloatScratchBytes =
    kMoonEpCombineHiddenTileElements * sizeof(float);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_COMMON_H
