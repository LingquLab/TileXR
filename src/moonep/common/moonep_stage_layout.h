#ifndef TILEXR_MOONEP_STAGE_LAYOUT_H
#define TILEXR_MOONEP_STAGE_LAYOUT_H

#include <cstdint>
#include <limits>

#include "tilexr_moonep.h"

namespace TileXRMoonEp {
namespace Layout {

inline bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

inline bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0 || value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

inline bool AppendRegion(uint64_t bytes, uint64_t alignment, uint64_t *cursor,
    uint64_t *offset)
{
    if (cursor == nullptr || offset == nullptr) {
        return false;
    }
    const uint64_t aligned = AlignUp(*cursor, alignment);
    uint64_t end = 0;
    if (aligned == std::numeric_limits<uint64_t>::max() ||
        !CheckedAdd(aligned, bytes, &end)) {
        return false;
    }
    *offset = aligned;
    *cursor = end;
    return true;
}

inline bool TensorHeaderValid(const TileXRMoonEpTensorV1 *tensor)
{
    if (tensor == nullptr || tensor->structSize < sizeof(*tensor) ||
        tensor->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || tensor->data == nullptr ||
        tensor->rank == 0 || tensor->rank > TILEXR_MOONEP_MAX_TENSOR_RANK ||
        tensor->elementCount == 0) {
        return false;
    }
    uint64_t elements = 1;
    for (uint32_t dim = 0; dim < TILEXR_MOONEP_MAX_TENSOR_RANK; ++dim) {
        if (dim < tensor->rank) {
            if (tensor->shape[dim] <= 0 ||
                !CheckedMul(elements, static_cast<uint64_t>(tensor->shape[dim]), &elements)) {
                return false;
            }
        } else if (tensor->shape[dim] != 0) {
            return false;
        }
    }
    return elements == tensor->elementCount;
}

inline bool PlanValid(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan, int64_t *s)
{
    if (s == nullptr || plan == nullptr || plan->structSize < sizeof(*plan) ||
        plan->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || commWorld <= 0 ||
        commWorld > 128 || commRank < 0 || commRank >= commWorld ||
        plan->n <= 0 || plan->r != commWorld || plan->e <= 0 || plan->b <= 0 ||
        plan->nvS < plan->n || plan->k <= 0 || plan->k > 32 ||
        plan->n % plan->k != 0 || plan->e % plan->r != 0 ||
        plan->b > plan->e / plan->r || plan->dst == nullptr ||
        plan->e > std::numeric_limits<int64_t>::max() - plan->b ||
        plan->expertsToCopy == nullptr || plan->zeroFillRanges == nullptr ||
        plan->remoteStats == nullptr || plan->dupGroups == nullptr ||
        plan->dupLoffs == nullptr || plan->dupCounts == nullptr || plan->status == nullptr) {
        return false;
    }
    uint64_t encodedCapacity = 0;
    if (!CheckedMul(static_cast<uint64_t>(plan->r),
            static_cast<uint64_t>(plan->nvS), &encodedCapacity) ||
        encodedCapacity > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1U) {
        return false;
    }
    *s = plan->n / plan->k;
    return *s > 0;
}

} // namespace Layout
} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_STAGE_LAYOUT_H
