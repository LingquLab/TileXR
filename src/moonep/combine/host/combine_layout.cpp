#include "combine_layout.h"

#include "comm_args.h"
#include "moonep_peer_window.h"
#include "moonep_stage_layout.h"

namespace TileXRMoonEp {
namespace {

bool HiddenValid(const TileXRMoonEpTensorV1 *input, TileXRMoonEpTensorV1 *output,
    int64_t s, int64_t nvS, uint64_t *rowBytes)
{
    return rowBytes != nullptr && Layout::TensorHeaderValid(input) &&
        Layout::TensorHeaderValid(output) && input->dtype == TILEXR_MOONEP_DTYPE_BFLOAT16 &&
        output->dtype == TILEXR_MOONEP_DTYPE_BFLOAT16 && input->rank == 2 &&
        output->rank == 2 && input->shape[0] == nvS && input->shape[1] > 0 &&
        output->shape[0] == s && output->shape[1] == input->shape[1] &&
        Layout::CheckedMul(static_cast<uint64_t>(input->shape[1]), sizeof(uint16_t), rowBytes);
}

bool WeightsValid(const TileXRMoonEpTensorV1 *input, TileXRMoonEpTensorV1 *output,
    int64_t s, int64_t k, int64_t nvS)
{
    if (input == nullptr || output == nullptr) {
        return input == nullptr && output == nullptr;
    }
    return Layout::TensorHeaderValid(input) && Layout::TensorHeaderValid(output) &&
        input->dtype == TILEXR_MOONEP_DTYPE_FLOAT32 &&
        output->dtype == TILEXR_MOONEP_DTYPE_FLOAT32 && input->rank == 1 &&
        input->shape[0] == nvS && output->rank == 2 && output->shape[0] == s &&
        output->shape[1] == k;
}

} // namespace

int TileXRMoonEpBuildCombineLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan, const TileXRMoonEpTensorV1 *hiddenNvsh,
    const TileXRMoonEpTensorV1 *routeWeightsNvs, TileXRMoonEpTensorV1 *hiddenSh,
    TileXRMoonEpTensorV1 *routeWeightsSk, uint64_t flags, CombineLayout *layout)
{
    if (layout == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *layout = CombineLayout {};
    const uint64_t splitFlags = TILEXR_MOONEP_FLAG_COMBINE_PUBLISH_ONLY |
        TILEXR_MOONEP_FLAG_COMBINE_CONSUME_ONLY;
    const uint64_t allowedFlags = TILEXR_MOONEP_FLAG_SKIP_INTER_RANK_SYNC | splitFlags;
    if ((flags & ~allowedFlags) != 0) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if ((flags & splitFlags) == splitFlags) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    int64_t s = 0;
    uint64_t hiddenRowBytes = 0;
    if (!Layout::PlanValid(commRank, commWorld, plan, &s) ||
        !HiddenValid(hiddenNvsh, hiddenSh, s, plan->nvS, &hiddenRowBytes) ||
        !WeightsValid(routeWeightsNvs, routeWeightsSk, s, plan->k, plan->nvS)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t nvS = static_cast<uint64_t>(plan->nvS);
    uint64_t routeWeightsBytes = 0;
    if (routeWeightsNvs != nullptr &&
        !Layout::CheckedMul(nvS, sizeof(float), &routeWeightsBytes)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if (routeWeightsBytes > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) -
        (kMoonEpStageAlignment - 1)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t available = static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) -
        routeWeightsBytes - (kMoonEpStageAlignment - 1);
    const uint64_t maxStride = (available / nvS / kMoonEpStageAlignment) *
        kMoonEpStageAlignment;
    if (maxStride == 0) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t hiddenChunkBytes = hiddenRowBytes < maxStride ? hiddenRowBytes : maxStride;
    const uint64_t hiddenChunkStride = Layout::AlignUp(hiddenChunkBytes, kMoonEpStageAlignment);
    uint64_t hiddenPayloadBytes = 0;
    uint64_t cursor = 0;
    CombineLayout next {};
    if (hiddenChunkStride == std::numeric_limits<uint64_t>::max() ||
        !Layout::CheckedMul(nvS, hiddenChunkStride, &hiddenPayloadBytes) ||
        !Layout::AppendRegion(hiddenPayloadBytes, kMoonEpStageAlignment, &cursor,
            &next.hiddenPayloadBytes) || next.hiddenPayloadBytes != 0 ||
        !Layout::AppendRegion(routeWeightsBytes, kMoonEpStageAlignment, &cursor,
            &next.routeWeightsOffset) ||
        cursor > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t chunkCount = (hiddenRowBytes - 1) / hiddenChunkBytes + 1;
    if ((flags & splitFlags) != 0 && chunkCount != 1) {
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }
    if (chunkCount > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() -
            kMoonEpCombineWindowDrainedStep) / 4U) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    next.rank = commRank;
    next.world = commWorld;
    next.s = s;
    next.k = plan->k;
    next.n = plan->n;
    next.nvS = plan->nvS;
    next.hiddenSize = hiddenNvsh->shape[1];
    next.blockDim = kMoonEpStageAivBlockCount;
    next.chunkCount = static_cast<int64_t>(chunkCount);
    next.flags = flags;
    next.hiddenRowBytes = hiddenRowBytes;
    next.hiddenChunkBytes = hiddenChunkBytes;
    next.hiddenChunkStride = hiddenChunkStride;
    next.hiddenPayloadBytes = hiddenPayloadBytes;
    next.routeWeightsBytes = routeWeightsBytes;
    next.windowBytes = cursor;
    *layout = next;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
