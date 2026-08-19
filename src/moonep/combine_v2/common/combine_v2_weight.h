#ifndef TILEXR_MOONEP_COMBINE_V2_WEIGHT_H
#define TILEXR_MOONEP_COMBINE_V2_WEIGHT_H

#include <cstdint>

namespace TileXRMoonEp {

constexpr uint64_t kMoonEpCombineV2WeightRecordBytes = 16U;
constexpr uint64_t kMoonEpCombineV2WeightDoneStrideBytes = 64U;

struct alignas(16) MoonEpCombineV2WeightRecord {
    float value;
    uint32_t reserved;
    uint64_t magic;
};

static_assert(sizeof(MoonEpCombineV2WeightRecord) ==
        kMoonEpCombineV2WeightRecordBytes,
    "Combine V2 weight record ABI changed");

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_V2_WEIGHT_H
