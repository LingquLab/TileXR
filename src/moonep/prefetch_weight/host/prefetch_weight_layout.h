#ifndef TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H
#define TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H

#include <cstdint>

#include "tilexr_moonep.h"
#include "tilexr_api.h"

namespace TileXRMoonEp {

constexpr uint32_t kPrefetchWeightMaxWorkers = 8;
constexpr uint32_t kPrefetchWeightAlignment = 64;
constexpr uint32_t kPrefetchWeightSharedQpCount = 32;
constexpr uint32_t kPrefetchWeightSecondClosQpBase = 16;

struct PrefetchWeightProjectionLayout {
    GM_ADDR localBase = nullptr;
    uint64_t registryOffset = 0;
    uint32_t rowBytes = 0;
};

struct PrefetchWeightLayout {
    PrefetchWeightProjectionLayout gate {};
    PrefetchWeightProjectionLayout up {};
    PrefetchWeightProjectionLayout down {};
    int64_t expertsPerRank = 0;
    int64_t prefetchSlots = 0;
    int32_t rank = 0;
    int32_t rankSize = 0;
    uint32_t qpNum = 0;
    uint32_t blockDim = 0;
    uint64_t physicalQpMap = 0;
};

int TileXRMoonEpBuildPrefetchWeightQpMap(uint32_t workers, uint32_t qpNum,
    bool sharedQps, uint64_t *physicalQpMap);

inline uint32_t TileXRMoonEpPrefetchWeightPhysicalQp(
    uint64_t physicalQpMap, uint32_t worker)
{
    return worker < kPrefetchWeightMaxWorkers ?
        static_cast<uint32_t>((physicalQpMap >> (worker * 8U)) & UINT64_C(0xFF)) :
        UINT32_MAX;
}

int TileXRMoonEpBuildPrefetchWeightLayout(
    const TileXRMoonEpPrefetchWeightArgsV1 &args,
    const TileXR::CommArgs &commArgs,
    const TileXR::TileXRUDMARegistry &registry, uint32_t qpNum,
    const char *blockDimOverride, PrefetchWeightLayout *layout);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_LAYOUT_H
