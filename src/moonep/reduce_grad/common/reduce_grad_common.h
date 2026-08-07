#ifndef TILEXR_MOONEP_REDUCE_GRAD_COMMON_H
#define TILEXR_MOONEP_REDUCE_GRAD_COMMON_H

#include "moonep_peer_window.h"

namespace TileXRMoonEp {

constexpr int64_t kMoonEpReduceGradTileElements = 24 * 1024;
constexpr uint32_t kMoonEpReduceGradTileBytes =
    kMoonEpReduceGradTileElements * sizeof(float);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_COMMON_H
