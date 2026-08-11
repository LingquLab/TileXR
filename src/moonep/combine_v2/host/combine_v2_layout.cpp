#include "combine_v2_layout.h"

#include <cstddef>
#include <limits>

#include "combine_v2_profile.h"
#include "tilexr_moonep.h"

namespace TileXRMoonEp {
namespace {

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (result == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool CheckedAlign(uint64_t value, uint64_t alignment, uint64_t *result)
{
    if (result == nullptr || alignment == 0 || value >
        std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
        return false;
    }
    *result = (value + alignment - 1) / alignment * alignment;
    return true;
}

} // namespace

int TileXRMoonEpBuildCombineV2Layout(int64_t bs, int64_t h,
    int64_t topK, int64_t nvS, uint32_t dtype,
    CombineV2Layout *layout)
{
    if (layout == nullptr || !MoonEpCombineV2ShapeValid(bs, h, topK, nvS) ||
        (dtype != TILEXR_MOONEP_DTYPE_BFLOAT16 &&
            dtype != TILEXR_MOONEP_DTYPE_FLOAT32) ||
        (dtype == TILEXR_MOONEP_DTYPE_FLOAT32 && h != 1)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    const uint64_t slots = static_cast<uint64_t>(nvS);
    const uint64_t elementBytes = dtype == TILEXR_MOONEP_DTYPE_BFLOAT16 ?
        sizeof(uint16_t) : sizeof(float);
    uint64_t rowBytes = 0;
    uint64_t expertBytes = 0;
    uint64_t profileOffset = 0;
    uint64_t profileBytes = 0;
    uint64_t scratch0Offset = 0;
    uint64_t scratch1Offset = 0;
    uint64_t requiredBytes = 0;
    uint64_t doneOffset = 0;
    uint64_t doneBytes = 0;
    uint64_t grantOffset = 0;
    uint64_t grantBytes = 0;
    uint64_t controlSourceOffset = 0;
    uint64_t controlSourceBytes = 0;
    uint64_t failureOffset = 0;
    uint64_t failureBytes = 0;
    uint64_t outputOffset = 0;
    uint64_t outputBytes = 0;
    uint64_t totalBytes = 0;

    if (!CheckedMultiply(static_cast<uint64_t>(h), elementBytes, &rowBytes) ||
        !CheckedMultiply(slots, rowBytes, &expertBytes) ||
        !CheckedAlign(expertBytes, kCombineV2ScratchAlignmentBytes, &profileOffset) ||
        !CheckedMultiply(kMoonEpCombineV2CoreCount,
            sizeof(MoonEpCombineV2ProfileRecord), &profileBytes) ||
        !CheckedAlign(profileBytes, kCombineV2ScratchAlignmentBytes, &profileBytes) ||
        !CheckedAdd(profileOffset, profileBytes, &scratch0Offset) ||
        !CheckedAdd(scratch0Offset, expertBytes, &scratch1Offset) ||
        !CheckedAdd(scratch1Offset, expertBytes, &requiredBytes) ||
        !CheckedAlign(requiredBytes, kMoonEpCombineV2TokenStrideBytes, &doneOffset) ||
        !CheckedMultiply(static_cast<uint64_t>(kMoonEpCombineV2EpochCount) *
                kMoonEpCombineV2RankCount * kMoonEpCombineV2LaneCount,
            kMoonEpCombineV2TokenStrideBytes, &doneBytes) ||
        !CheckedAdd(doneOffset, doneBytes, &grantOffset) ||
        !CheckedMultiply(static_cast<uint64_t>(kMoonEpCombineV2EpochCount) *
                kMoonEpCombineV2CoreCount * kMoonEpCombineV2LaneCount *
                kMoonEpCombineV2GrantStepCount,
            kMoonEpCombineV2GrantSlotBytes, &grantBytes) ||
        !CheckedAdd(grantOffset, grantBytes, &controlSourceOffset) ||
        !CheckedMultiply(static_cast<uint64_t>(kMoonEpCombineV2CoreCount) *
                kMoonEpCombineV2LaneCount,
            kMoonEpCombineV2TokenStrideBytes, &controlSourceBytes) ||
        !CheckedAdd(controlSourceOffset, controlSourceBytes, &failureOffset) ||
        !CheckedMultiply(static_cast<uint64_t>(kMoonEpCombineV2EpochCount) *
                kMoonEpCombineV2CoreCount,
            kMoonEpCombineV2TokenStrideBytes, &failureBytes) ||
        !CheckedAdd(failureOffset, failureBytes, &requiredBytes) ||
        !CheckedAlign(requiredBytes, kCombineV2ScratchAlignmentBytes,
            &outputOffset) ||
        !CheckedMultiply(static_cast<uint64_t>(bs), rowBytes, &outputBytes) ||
        !CheckedAdd(outputOffset, outputBytes, &requiredBytes) ||
        !CheckedAlign(requiredBytes, kCombineV2RegistrationAlignmentBytes, &totalBytes) ||
        totalBytes > std::numeric_limits<uint32_t>::max() ||
        totalBytes > std::numeric_limits<std::size_t>::max()) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    CombineV2Layout next {};
    next.slots = slots;
    next.rowBytes = rowBytes;
    next.expertBytes = expertBytes;
    next.profileOffset = profileOffset;
    next.profileBytes = profileBytes;
    next.scratchOffset[0] = scratch0Offset;
    next.scratchOffset[1] = scratch1Offset;
    next.scratchBytes = expertBytes;
    next.doneOffset = doneOffset;
    next.doneBytes = doneBytes;
    next.grantOffset = grantOffset;
    next.grantBytes = grantBytes;
    next.controlSourceOffset = controlSourceOffset;
    next.controlSourceBytes = controlSourceBytes;
    next.failureOffset = failureOffset;
    next.failureBytes = failureBytes;
    next.outputOffset = outputOffset;
    next.outputBytes = outputBytes;
    next.totalBytes = totalBytes;
    *layout = next;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
