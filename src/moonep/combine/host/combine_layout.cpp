#include "combine_layout.h"

#include <limits>

#include "moonep_combine_schedule.h"
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

bool BuildRegions(uint64_t nvS, uint64_t hiddenStride, uint64_t weightsBytes,
    uint64_t world, uint64_t activeCores, CombineLayout *layout)
{
    if (layout == nullptr || nvS == 0 || hiddenStride == 0) {
        return false;
    }
    CombineLayout next = *layout;
    uint64_t hiddenBytes = 0;
    uint64_t doneBytes = 0;
    uint64_t statusBytes = 0;
    uint64_t maskBytes = 0;
    uint64_t cursor = 0;
    if (!Layout::CheckedMul(nvS, hiddenStride, &hiddenBytes) ||
        !Layout::CheckedMul(nvS, sizeof(int32_t), &maskBytes) ||
        !Layout::CheckedMul(world, kMoonEpCombineV2TokenStrideBytes, &doneBytes) ||
        !Layout::CheckedMul(activeCores, kMoonEpCombineV2TokenStrideBytes, &statusBytes) ||
        !Layout::AppendRegion(hiddenBytes, kMoonEpStageAlignment, &cursor,
            &next.sourceHiddenOffset) ||
        !Layout::AppendRegion(hiddenBytes, kMoonEpStageAlignment, &cursor,
            &next.receiveHiddenOffset) ||
        !Layout::AppendRegion(weightsBytes, kMoonEpStageAlignment, &cursor,
            &next.sourceWeightsOffset) ||
        !Layout::AppendRegion(weightsBytes, kMoonEpStageAlignment, &cursor,
            &next.receiveWeightsOffset) ||
        !Layout::AppendRegion(maskBytes, kMoonEpStageAlignment, &cursor,
            &next.duplicateMaskOffset) ||
        !Layout::AppendRegion(doneBytes, kMoonEpCombineV2TokenStrideBytes, &cursor,
            &next.doneOffset) ||
        !Layout::AppendRegion(statusBytes, kMoonEpCombineV2TokenStrideBytes, &cursor,
            &next.coreStatusOffset) ||
        cursor > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
        return false;
    }
    next.hiddenPayloadBytes = hiddenBytes;
    next.routeWeightsBytes = weightsBytes;
    next.duplicateMaskBytes = maskBytes;
    next.doneBytes = doneBytes;
    next.coreStatusBytes = statusBytes;
    next.windowBytes = cursor;
    *layout = next;
    return true;
}

uint64_t MaxHiddenStride(uint64_t nvS, uint64_t weightsBytes,
    uint64_t world, uint64_t activeCores)
{
    const uint64_t capacity = static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE);
    const uint64_t maxUnits = capacity / kMoonEpStageAlignment / nvS / 2U;
    uint64_t low = 0;
    uint64_t high = maxUnits;
    while (low < high) {
        const uint64_t mid = low + (high - low + 1U) / 2U;
        CombineLayout candidate {};
        if (BuildRegions(nvS, mid * kMoonEpStageAlignment, weightsBytes,
                world, activeCores, &candidate)) {
            low = mid;
        } else {
            high = mid - 1U;
        }
    }
    return low * kMoonEpStageAlignment;
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
    int64_t s = 0;
    uint64_t hiddenRowBytes = 0;
    if (flags != TILEXR_MOONEP_FLAG_NONE ||
        !MoonEpCombineV2RankSizeSupported(static_cast<uint32_t>(commWorld)) ||
        !Layout::PlanValid(commRank, commWorld, plan, &s) ||
        !HiddenValid(hiddenNvsh, hiddenSh, s, plan->nvS, &hiddenRowBytes) ||
        hiddenRowBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        !WeightsValid(routeWeightsNvs, routeWeightsSk, s, plan->k, plan->nvS)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    const uint64_t world = static_cast<uint64_t>(commWorld);
    const uint64_t nvS = static_cast<uint64_t>(plan->nvS);
    const uint64_t activeCores = MoonEpCombineV2ActiveCoreCount(
        static_cast<uint32_t>(commWorld));
    uint64_t routeWeightsBytes = 0;
    if (routeWeightsNvs != nullptr &&
        !Layout::CheckedMul(nvS, sizeof(float), &routeWeightsBytes)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t maxStride = MaxHiddenStride(
        nvS, routeWeightsBytes, world, activeCores);
    if (maxStride == 0) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t hiddenChunkBytes = hiddenRowBytes < maxStride ? hiddenRowBytes : maxStride;
    const uint64_t hiddenChunkStride = Layout::AlignUp(hiddenChunkBytes, kMoonEpStageAlignment);
    if (hiddenChunkStride == std::numeric_limits<uint64_t>::max()) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    const uint64_t chunkCount = (hiddenRowBytes - 1U) / hiddenChunkBytes + 1U;
    if (chunkCount > 0xFFFFFFU) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    CombineLayout next {};
    if (!BuildRegions(nvS, hiddenChunkStride, routeWeightsBytes,
            world, activeCores, &next)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    next.rank = commRank;
    next.world = commWorld;
    next.s = s;
    next.k = plan->k;
    next.n = plan->n;
    next.nvS = plan->nvS;
    next.hiddenSize = hiddenNvsh->shape[1];
    next.blockDim = static_cast<int64_t>(activeCores);
    next.stepCount = static_cast<int64_t>(MoonEpCombineV2StepCount(
        static_cast<uint32_t>(commWorld)));
    next.sourcesPerCore = commWorld / next.blockDim;
    next.chunkCount = static_cast<int64_t>(chunkCount);
    next.flags = flags;
    next.hiddenRowBytes = hiddenRowBytes;
    next.hiddenChunkBytes = hiddenChunkBytes;
    next.hiddenChunkStride = hiddenChunkStride;
    *layout = next;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
