#include "reduce_grad_layout.h"

#include <limits>

#include "comm_args.h"
#include "moonep_peer_window.h"
#include "moonep_stage_layout.h"

namespace TileXRMoonEp {
namespace {

bool BuildProjection(const TileXRMoonEpTensorV1 *full,
    const TileXRMoonEpTensorV1 *buffer, int64_t world, int64_t e, int64_t b,
    ReduceGradProjectionLayout *projection)
{
    if (projection == nullptr || !Layout::TensorHeaderValid(full) ||
        !Layout::TensorHeaderValid(buffer) ||
        full->dtype != TILEXR_MOONEP_DTYPE_FLOAT32 ||
        buffer->dtype != TILEXR_MOONEP_DTYPE_FLOAT32 || full->rank != 3 ||
        buffer->rank != 4 || full->shape[0] != e + b || buffer->shape[0] != world ||
        buffer->shape[1] != b || buffer->shape[2] != full->shape[1] ||
        buffer->shape[3] != full->shape[2]) {
        return false;
    }
    uint64_t rowElements = 0;
    uint64_t rowBytes = 0;
    if (!Layout::CheckedMul(static_cast<uint64_t>(full->shape[1]),
            static_cast<uint64_t>(full->shape[2]), &rowElements) ||
        !Layout::CheckedMul(rowElements, sizeof(float), &rowBytes) || rowBytes == 0 ||
        rowBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    const uint64_t maxStride = (static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) /
        static_cast<uint64_t>(b) / kMoonEpStageAlignment) * kMoonEpStageAlignment;
    if (maxStride == 0) {
        return false;
    }
    const uint64_t chunkBytes = rowBytes < maxStride ? rowBytes : maxStride;
    const uint64_t chunkStride = Layout::AlignUp(chunkBytes, kMoonEpStageAlignment);
    uint64_t payloadBytes = 0;
    if (chunkStride == std::numeric_limits<uint64_t>::max() ||
        !Layout::CheckedMul(static_cast<uint64_t>(b), chunkStride, &payloadBytes) ||
        payloadBytes > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
        return false;
    }
    projection->rowBytes = static_cast<int64_t>(rowBytes);
    projection->chunkBytes = static_cast<int64_t>(chunkBytes);
    projection->chunkStride = static_cast<int64_t>(chunkStride);
    projection->chunkCount = static_cast<int64_t>((rowBytes - 1) / chunkBytes + 1);
    projection->payloadBytes = payloadBytes;
    return true;
}

} // namespace

int TileXRMoonEpBuildReduceGradLayout(int64_t commRank, int64_t commWorld,
    const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *fullGateGrad,
    const TileXRMoonEpTensorV1 *fullUpGrad,
    const TileXRMoonEpTensorV1 *fullDownGrad,
    const TileXRMoonEpTensorV1 *gateReduceBuffer,
    const TileXRMoonEpTensorV1 *upReduceBuffer,
    const TileXRMoonEpTensorV1 *downReduceBuffer,
    uint64_t flags, ReduceGradLayout *layout)
{
    if (layout == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *layout = ReduceGradLayout {};
    int64_t s = 0;
    if (flags != TILEXR_MOONEP_FLAG_NONE ||
        !Layout::PlanValid(commRank, commWorld, plan, &s)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    ReduceGradLayout next {};
    if (!BuildProjection(fullGateGrad, gateReduceBuffer, commWorld, plan->e, plan->b,
            &next.gate) ||
        !BuildProjection(fullUpGrad, upReduceBuffer, commWorld, plan->e, plan->b,
            &next.up) ||
        !BuildProjection(fullDownGrad, downReduceBuffer, commWorld, plan->e, plan->b,
            &next.down)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    uint64_t iterations = 0;
    if (!Layout::CheckedAdd(static_cast<uint64_t>(next.gate.chunkCount),
            static_cast<uint64_t>(next.up.chunkCount), &iterations) ||
        !Layout::CheckedAdd(iterations, static_cast<uint64_t>(next.down.chunkCount),
            &iterations) ||
        iterations > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - 600) / 3U) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    next.rank = commRank;
    next.world = commWorld;
    next.e = plan->e;
    next.b = plan->b;
    next.expertsPerRank = plan->e / plan->r;
    next.blockDim = kMoonEpStageAivBlockCount;
    next.iterationCount = static_cast<int64_t>(iterations);
    *layout = next;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
