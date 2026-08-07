#ifndef TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H
#define TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H

#include <cstdint>

#include "comm_args.h"
#include "tilexr_moonep.h"
#include "tilexr_udma_reg.h"

namespace TileXRMoonEp {

constexpr uint32_t kPrefetchWeightMaxWorkers = 8;
constexpr uint32_t kPrefetchWeightAlignment = 64;

struct PrefetchWeightProjectionLayout {
    GM_ADDR localBase = nullptr;
    uint64_t registryOffset = 0;
    uint32_t rowBytes = 0;
};

struct PrefetchWeightLayout {
    PrefetchWeightProjectionLayout gate;
    PrefetchWeightProjectionLayout up;
    PrefetchWeightProjectionLayout down;
    int32_t rank = 0;
    int32_t rankSize = 0;
    int32_t expertsPerRank = 0;
    uint32_t qpNum = 0;
    uint32_t blockDim = 0;
};

int BuildPrefetchWeightLayout(const TileXRMoonEpPrefetchWeightArgsV1 &args,
    const TileXR::CommArgs &commArgs, const TileXR::TileXRUDMARegistry &registry,
    uint32_t qpNum, const char *blockDimOverride, PrefetchWeightLayout &layout);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H
