#include "prefetch_weight_layout.h"

#include <limits>

#include "comm_args.h"
#include "moonep_peer_window.h"
#include "moonep_stage_layout.h"

namespace TileXRMoonEp {
namespace {

bool BuildProjection(const TileXRMoonEpTensorV1 *tensor, int64_t rows,
    PrefetchProjectionLayout *projection)
{
    if (projection == nullptr || !Layout::TensorHeaderValid(tensor) ||
        tensor->dtype != TILEXR_MOONEP_DTYPE_BFLOAT16 || tensor->rank != 3 ||
        tensor->shape[0] != rows) {
        return false;
    }
    uint64_t rowElements = 0;
    uint64_t rowBytes = 0;
    if (!Layout::CheckedMul(static_cast<uint64_t>(tensor->shape[1]),
            static_cast<uint64_t>(tensor->shape[2]), &rowElements) ||
        !Layout::CheckedMul(rowElements, sizeof(uint16_t), &rowBytes) || rowBytes == 0 ||
        rowBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    const uint64_t maxChunk = (static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) /
        kMoonEpStageAlignment) * kMoonEpStageAlignment;
    const uint64_t chunkBytes = rowBytes < maxChunk ? rowBytes : maxChunk;
    const uint64_t chunkCount = (rowBytes - 1) / chunkBytes + 1;
    if (chunkBytes == 0 || chunkCount > static_cast<uint64_t>(
            std::numeric_limits<int64_t>::max())) {
        return false;
    }
    projection->rowBytes = static_cast<int64_t>(rowBytes);
    projection->chunkBytes = static_cast<int64_t>(chunkBytes);
    projection->chunkCount = static_cast<int64_t>(chunkCount);
    return true;
}

} // namespace

int TileXRMoonEpBuildPrefetchWeightLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan, const TileXRMoonEpTensorV1 *fullGateWeight,
    const TileXRMoonEpTensorV1 *fullUpWeight,
    const TileXRMoonEpTensorV1 *fullDownWeight, uint64_t flags,
    PrefetchWeightLayout *layout)
{
    if (layout == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *layout = PrefetchWeightLayout {};
    int64_t s = 0;
    if (flags != TILEXR_MOONEP_FLAG_NONE ||
        !Layout::PlanValid(commRank, commWorld, plan, &s)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    PrefetchWeightLayout next {};
    if (!BuildProjection(fullGateWeight, plan->e + plan->b, &next.gate) ||
        !BuildProjection(fullUpWeight, plan->e + plan->b, &next.up) ||
        !BuildProjection(fullDownWeight, plan->e + plan->b, &next.down)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    uint64_t chunks = 0;
    if (!Layout::CheckedAdd(static_cast<uint64_t>(next.gate.chunkCount),
            static_cast<uint64_t>(next.up.chunkCount), &chunks) ||
        !Layout::CheckedAdd(chunks, static_cast<uint64_t>(next.down.chunkCount), &chunks) ||
        !Layout::CheckedMul(chunks, static_cast<uint64_t>(plan->b), &chunks) ||
        chunks > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - 500) / 3U) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    next.rank = commRank;
    next.world = commWorld;
    next.e = plan->e;
    next.b = plan->b;
    next.expertsPerRank = plan->e / plan->r;
    next.blockDim = kMoonEpStageAivBlockCount;
    next.iterationCount = static_cast<int64_t>(chunks);
    *layout = next;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
