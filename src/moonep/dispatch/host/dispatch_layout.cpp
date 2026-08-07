#include "dispatch_layout.h"

#include "comm_args.h"
#include "moonep_peer_window.h"
#include "moonep_stage_layout.h"

namespace TileXRMoonEp {
namespace {

bool HiddenValid(const TileXRMoonEpTensorV1 *input, const TileXRMoonEpTensorV1 *output,
    int64_t s, int64_t nvS, uint64_t *rowBytes)
{
    return rowBytes != nullptr && Layout::TensorHeaderValid(input) &&
        Layout::TensorHeaderValid(output) && input->dtype == TILEXR_MOONEP_DTYPE_BFLOAT16 &&
        output->dtype == TILEXR_MOONEP_DTYPE_BFLOAT16 && input->rank == 2 &&
        output->rank == 2 && input->shape[0] == s && input->shape[1] > 0 &&
        output->shape[0] == nvS && output->shape[1] == input->shape[1] &&
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
        output->dtype == TILEXR_MOONEP_DTYPE_FLOAT32 && input->rank == 2 &&
        input->shape[0] == s && input->shape[1] == k && output->rank == 1 &&
        output->shape[0] == nvS;
}

} // namespace

uint64_t TileXRMoonEpAlignDispatchBytes(uint64_t value)
{
    return Layout::AlignUp(value, kMoonEpStageAlignment);
}

int TileXRMoonEpBuildDispatchLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan, const TileXRMoonEpTensorV1 *hiddenSh,
    const TileXRMoonEpTensorV1 *routeWeightsSk, TileXRMoonEpTensorV1 *hiddenNvsh,
    TileXRMoonEpTensorV1 *routeWeightsNvs, uint64_t flags, DispatchLayout *layout)
{
    if (layout == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *layout = DispatchLayout {};
    const uint64_t allowedFlags = TILEXR_MOONEP_FLAG_BUILD_DEDUP |
        TILEXR_MOONEP_FLAG_SKIP_INTER_RANK_SYNC;
    if ((flags & ~allowedFlags) != 0) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    int64_t s = 0;
    uint64_t hiddenRowBytes = 0;
    if (!Layout::PlanValid(commRank, commWorld, plan, &s) ||
        !HiddenValid(hiddenSh, hiddenNvsh, s, plan->nvS, &hiddenRowBytes) ||
        !WeightsValid(routeWeightsSk, routeWeightsNvs, s, plan->k, plan->nvS)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t nvS = static_cast<uint64_t>(plan->nvS);
    uint64_t fixedBytes = 0;
    uint64_t routeWeightsBytes = 0;
    uint64_t dedupBytes = 0;
    if (routeWeightsSk != nullptr && !Layout::CheckedMul(nvS, sizeof(float), &routeWeightsBytes)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if ((flags & TILEXR_MOONEP_FLAG_BUILD_DEDUP) != 0 &&
        !Layout::CheckedMul(nvS, sizeof(int32_t), &dedupBytes)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if (!Layout::CheckedAdd(routeWeightsBytes, dedupBytes, &fixedBytes) ||
        !Layout::CheckedAdd(fixedBytes, dedupBytes, &fixedBytes) ||
        fixedBytes > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) -
            3U * (kMoonEpStageAlignment - 1)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t available = static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) - fixedBytes -
        3U * (kMoonEpStageAlignment - 1);
    uint64_t maxStride = (available / nvS / kMoonEpStageAlignment) * kMoonEpStageAlignment;
    if (maxStride == 0) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t hiddenChunkBytes = hiddenRowBytes < maxStride ? hiddenRowBytes : maxStride;
    const uint64_t hiddenChunkStride = Layout::AlignUp(hiddenChunkBytes, kMoonEpStageAlignment);
    uint64_t hiddenPayloadBytes = 0;
    uint64_t cursor = 0;
    DispatchLayout next {};
    if (hiddenChunkStride == std::numeric_limits<uint64_t>::max() ||
        !Layout::CheckedMul(nvS, hiddenChunkStride, &hiddenPayloadBytes) ||
        !Layout::AppendRegion(hiddenPayloadBytes, kMoonEpStageAlignment, &cursor,
            &next.hiddenPayloadBytes) || next.hiddenPayloadBytes != 0 ||
        !Layout::AppendRegion(routeWeightsBytes, kMoonEpStageAlignment, &cursor,
            &next.routeWeightsOffset) ||
        !Layout::AppendRegion(dedupBytes, kMoonEpStageAlignment, &cursor,
            &next.dedupParentsOffset) ||
        !Layout::AppendRegion(dedupBytes, kMoonEpStageAlignment, &cursor,
            &next.dedupGroupMapOffset) ||
        cursor > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t chunkCount = (hiddenRowBytes - 1) / hiddenChunkBytes + 1;
    if (chunkCount > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() -
            kMoonEpDispatchWindowDrainedStep) / 4U) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    next.rank = commRank;
    next.world = commWorld;
    next.s = s;
    next.k = plan->k;
    next.n = plan->n;
    next.nvS = plan->nvS;
    next.hiddenSize = hiddenSh->shape[1];
    next.blockDim = kMoonEpStageAivBlockCount;
    next.chunkCount = static_cast<int64_t>(chunkCount);
    next.flags = flags;
    next.hiddenRowBytes = hiddenRowBytes;
    next.hiddenChunkBytes = hiddenChunkBytes;
    next.hiddenChunkStride = hiddenChunkStride;
    next.hiddenPayloadBytes = hiddenPayloadBytes;
    next.routeWeightsBytes = routeWeightsBytes;
    next.dedupParentsBytes = dedupBytes;
    next.dedupGroupMapBytes = dedupBytes;
    next.windowBytes = cursor;
    *layout = next;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
