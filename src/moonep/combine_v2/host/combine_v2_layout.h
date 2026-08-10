#ifndef TILEXR_MOONEP_COMBINE_V2_LAYOUT_H
#define TILEXR_MOONEP_COMBINE_V2_LAYOUT_H

#include <cstdint>

#include "combine_v2_schedule.h"

namespace TileXRMoonEp {

constexpr uint64_t kCombineV2ScratchAlignmentBytes = 64;
constexpr uint64_t kCombineV2RegistrationAlignmentBytes = UINT64_C(2) * 1024 * 1024;

struct CombineV2Layout {
    uint64_t slots = 0;
    uint64_t rowBytes = 0;
    uint64_t expertBytes = 0;
    uint64_t profileOffset = 0;
    uint64_t profileBytes = 0;
    uint64_t scratchOffset[kMoonEpCombineV2EpochCount] = {};
    uint64_t scratchBytes = 0;
    uint64_t doneOffset = 0;
    uint64_t doneBytes = 0;
    uint64_t grantOffset = 0;
    uint64_t grantBytes = 0;
    uint64_t controlSourceOffset = 0;
    uint64_t controlSourceBytes = 0;
    uint64_t failureOffset = 0;
    uint64_t failureBytes = 0;
    uint64_t totalBytes = 0;
};

int TileXRMoonEpBuildCombineV2Layout(int64_t bs, int64_t h,
    int64_t topK, int64_t nvS, uint32_t dtype,
    CombineV2Layout *layout);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_V2_LAYOUT_H
