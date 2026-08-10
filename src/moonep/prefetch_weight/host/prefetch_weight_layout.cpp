#include "prefetch_weight_layout.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <limits>

#include "moonep_stage_layout.h"

namespace TileXRMoonEp {
namespace {

bool SupportedWorkerCount(uint32_t value)
{
    return value == 1 || value == 2 || value == 4 ||
        value == kPrefetchWeightMaxWorkers;
}

bool ParseWorkerOverride(const char *value, uint32_t *workers)
{
    if (workers == nullptr) {
        return false;
    }
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX ||
        !SupportedWorkerCount(static_cast<uint32_t>(parsed))) {
        return false;
    }
    *workers = static_cast<uint32_t>(parsed);
    return true;
}

bool BuildProjection(const TileXRMoonEpTensorV1 *tensor, int64_t expertsPerRank,
    int32_t localRank,
    const TileXR::TileXRUDMARegistry &registry,
    PrefetchWeightProjectionLayout *projection, uint64_t *rangeEnd)
{
    if (projection == nullptr || rangeEnd == nullptr ||
        !Layout::TensorHeaderValid(tensor) ||
        (tensor->dtype != TILEXR_MOONEP_DTYPE_FLOAT16 &&
            tensor->dtype != TILEXR_MOONEP_DTYPE_BFLOAT16) ||
        tensor->rank < 2 || tensor->rank > TILEXR_MOONEP_MAX_TENSOR_RANK ||
        expertsPerRank <= 0 || tensor->shape[0] != 2 * expertsPerRank) {
        return false;
    }

    uint64_t rowBytes = 0;
    uint64_t tensorBytes = 0;
    if (!Layout::CheckedMul(tensor->elementCount, sizeof(uint16_t), &tensorBytes) ||
        tensorBytes % static_cast<uint64_t>(2 * expertsPerRank) != 0) {
        return false;
    }
    rowBytes = tensorBytes / static_cast<uint64_t>(2 * expertsPerRank);
    if (rowBytes == 0 || rowBytes > UINT32_MAX ||
        rowBytes % kPrefetchWeightAlignment != 0) {
        return false;
    }

    const uintptr_t regionBase =
        reinterpret_cast<uintptr_t>(registry.regions[localRank].base);
    const uintptr_t tensorBase = reinterpret_cast<uintptr_t>(tensor->data);
    if (regionBase == 0 || tensorBase < regionBase ||
        tensorBase % kPrefetchWeightAlignment != 0) {
        return false;
    }
    const uint64_t offset = static_cast<uint64_t>(tensorBase - regionBase);
    uint64_t sourceBytes = 0;
    if (offset % kPrefetchWeightAlignment != 0 ||
        !Layout::CheckedMul(static_cast<uint64_t>(expertsPerRank), rowBytes,
            &sourceBytes) ||
        !TileXR::UDMARegionContains(&registry, localRank, offset, tensorBytes)) {
        return false;
    }
    for (uint32_t peer = 0; peer < registry.rankSize; ++peer) {
        if (!TileXR::UDMARegionContains(&registry, static_cast<int>(peer),
                offset, sourceBytes)) {
            return false;
        }
    }
    if (offset > std::numeric_limits<uint64_t>::max() - tensorBytes) {
        return false;
    }

    projection->localBase = static_cast<GM_ADDR>(tensor->data);
    projection->registryOffset = offset;
    projection->rowBytes = static_cast<uint32_t>(rowBytes);
    *rangeEnd = offset + tensorBytes;
    return true;
}

bool RangesOverlap(uint64_t lhsBegin, uint64_t lhsEnd,
    uint64_t rhsBegin, uint64_t rhsEnd)
{
    return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

} // namespace

int TileXRMoonEpBuildPrefetchWeightLayout(
    const TileXRMoonEpPrefetchWeightArgsV1 &args,
    const TileXR::CommArgs &commArgs,
    const TileXR::TileXRUDMARegistry &registry, uint32_t qpNum,
    const char *blockDimOverride, PrefetchWeightLayout *layout)
{
    if (layout == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *layout = PrefetchWeightLayout {};
    int64_t s = 0;
    if (args.flags != TILEXR_MOONEP_FLAG_NONE ||
        !Layout::PlanValid(commArgs.rank, commArgs.rankSize, args.plan, &s) ||
        args.plan->e > INT32_MAX || args.plan->b > INT32_MAX ||
        args.plan->b > args.plan->e / args.plan->r ||
        (commArgs.extraFlag & TileXR::ExtraFlag::UDMA) == 0 ||
        commArgs.udmaInfoPtr == nullptr || commArgs.udmaRegistryPtr == nullptr ||
        !TileXR::UDMARegistryValid(&registry, commArgs.rankSize) ||
        commArgs.rank < 0 || commArgs.rank >= commArgs.rankSize ||
        !SupportedWorkerCount(qpNum) || qpNum > kPrefetchWeightMaxWorkers) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    PrefetchWeightLayout next {};
    const int64_t expertsPerRank = args.plan->e / args.plan->r;
    uint64_t gateEnd = 0;
    uint64_t upEnd = 0;
    uint64_t downEnd = 0;
    if (args.gate == nullptr || args.up == nullptr || args.down == nullptr ||
        args.gate->dtype != args.up->dtype || args.gate->dtype != args.down->dtype ||
        !BuildProjection(args.gate, expertsPerRank, commArgs.rank,
            registry, &next.gate, &gateEnd) ||
        !BuildProjection(args.up, expertsPerRank, commArgs.rank,
            registry, &next.up, &upEnd) ||
        !BuildProjection(args.down, expertsPerRank, commArgs.rank,
            registry, &next.down, &downEnd) ||
        RangesOverlap(next.gate.registryOffset, gateEnd,
            next.up.registryOffset, upEnd) ||
        RangesOverlap(next.gate.registryOffset, gateEnd,
            next.down.registryOffset, downEnd) ||
        RangesOverlap(next.up.registryOffset, upEnd,
            next.down.registryOffset, downEnd)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    const bool hasOverride = blockDimOverride != nullptr && blockDimOverride[0] != '\0';
    uint32_t workers = qpNum;
    if (!ParseWorkerOverride(blockDimOverride, &workers) || workers > qpNum ||
        (hasOverride && workers > static_cast<uint32_t>(args.plan->b))) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    while (workers > static_cast<uint32_t>(args.plan->b)) {
        workers >>= 1;
    }
    next.expertsPerRank = expertsPerRank;
    next.prefetchSlots = args.plan->b;
    next.rank = commArgs.rank;
    next.rankSize = commArgs.rankSize;
    next.qpNum = qpNum;
    next.blockDim = workers;
    *layout = next;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
