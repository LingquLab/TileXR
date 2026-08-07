#include "prefetch_weight_layout.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <limits>
#include <string>

namespace TileXRMoonEp {
namespace {

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool DTypeBytes(uint32_t dtype, uint64_t &bytes)
{
    if (dtype != TILEXR_MOONEP_DTYPE_FLOAT16 &&
        dtype != TILEXR_MOONEP_DTYPE_BFLOAT16) {
        return false;
    }
    bytes = 2;
    return true;
}

bool SupportedWorkerCount(uint32_t value)
{
    return value == 1 || value == 2 || value == 4 ||
        value == kPrefetchWeightMaxWorkers;
}

bool ParseWorkerOverride(const char *value, uint32_t &workers)
{
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
    workers = static_cast<uint32_t>(parsed);
    return true;
}

bool TensorByteCount(const TileXRMoonEpTensorV1 &tensor, uint64_t &bytes)
{
    if (tensor.structSize < sizeof(TileXRMoonEpTensorV1) ||
        tensor.abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || tensor.data == nullptr ||
        tensor.rank < 2 || tensor.rank > TILEXR_MOONEP_MAX_TENSOR_RANK ||
        tensor.elementCount == 0) {
        return false;
    }

    uint64_t dtypeBytes = 0;
    uint64_t elements = 1;
    if (!DTypeBytes(tensor.dtype, dtypeBytes)) {
        return false;
    }
    for (uint32_t dim = 0; dim < TILEXR_MOONEP_MAX_TENSOR_RANK; ++dim) {
        if (dim < tensor.rank) {
            if (tensor.shape[dim] <= 0 ||
                !CheckedMultiply(elements, static_cast<uint64_t>(tensor.shape[dim]), elements)) {
                return false;
            }
        } else if (tensor.shape[dim] != 0) {
            return false;
        }
    }
    return elements == tensor.elementCount &&
        CheckedMultiply(elements, dtypeBytes, bytes);
}

bool BuildProjection(const TileXRMoonEpTensorV1 &tensor, int32_t expertsPerRank,
    int32_t localRank, const TileXR::TileXRUDMARegistry &registry,
    PrefetchWeightProjectionLayout &projection, uint64_t &rangeEnd)
{
    if (expertsPerRank <= 0 || tensor.shape[0] != 2LL * expertsPerRank) {
        return false;
    }

    uint64_t tensorBytes = 0;
    if (!TensorByteCount(tensor, tensorBytes)) {
        return false;
    }
    const uint64_t rowCount = static_cast<uint64_t>(2LL * expertsPerRank);
    if (tensorBytes % rowCount != 0) {
        return false;
    }
    const uint64_t rowBytes = tensorBytes / rowCount;
    if (rowBytes == 0 || rowBytes > UINT32_MAX ||
        rowBytes % kPrefetchWeightAlignment != 0) {
        return false;
    }

    const uintptr_t regionBase = reinterpret_cast<uintptr_t>(registry.regions[localRank].base);
    const uintptr_t tensorBase = reinterpret_cast<uintptr_t>(tensor.data);
    if (regionBase == 0 || tensorBase < regionBase ||
        tensorBase % kPrefetchWeightAlignment != 0) {
        return false;
    }
    const uint64_t offset = static_cast<uint64_t>(tensorBase - regionBase);
    if (offset % kPrefetchWeightAlignment != 0 ||
        !TileXR::UDMARegionContains(&registry, localRank, offset, tensorBytes)) {
        return false;
    }

    uint64_t sourceBytes = 0;
    if (!CheckedMultiply(static_cast<uint64_t>(expertsPerRank), rowBytes, sourceBytes)) {
        return false;
    }
    for (uint32_t peer = 0; peer < registry.rankSize; ++peer) {
        if (!TileXR::UDMARegionContains(&registry, static_cast<int>(peer), offset, sourceBytes)) {
            return false;
        }
    }
    if (offset > std::numeric_limits<uint64_t>::max() - tensorBytes) {
        return false;
    }

    projection.localBase = static_cast<GM_ADDR>(tensor.data);
    projection.registryOffset = offset;
    projection.rowBytes = static_cast<uint32_t>(rowBytes);
    rangeEnd = offset + tensorBytes;
    return true;
}

bool RangesOverlap(uint64_t lhsBegin, uint64_t lhsEnd,
    uint64_t rhsBegin, uint64_t rhsEnd)
{
    return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

} // namespace

bool BuildPrefetchWeightRouteWeights(
    uint32_t qpNum, const char *routeSpec, uint64_t &packedWeights)
{
    if (!SupportedWorkerCount(qpNum) || qpNum > kPrefetchWeightMaxWorkers) {
        return false;
    }
    if (routeSpec == nullptr || routeSpec[0] == '\0') {
        packedWeights = 0;
        for (uint32_t qp = 0; qp < qpNum; ++qp) {
            packedWeights |= static_cast<uint64_t>(1) <<
                (qp * kPrefetchWeightRouteWeightBits);
        }
        return true;
    }

    const std::string spec(routeSpec);
    uint64_t result = 0;
    size_t begin = 0;
    uint32_t qp = 0;
    while (begin <= spec.size() && qp < qpNum) {
        const size_t comma = spec.find(',', begin);
        const size_t end = comma == std::string::npos ? spec.size() : comma;
        const std::string rule = spec.substr(begin, end - begin);
        uint32_t weight = 0;
        if (rule == "topology") {
            weight = 1;
        } else {
            constexpr char prefix[] = "port_count:";
            if (rule.compare(0, sizeof(prefix) - 1, prefix) != 0) {
                return false;
            }
            const std::string value = rule.substr(sizeof(prefix) - 1);
            if (value.empty()) {
                return false;
            }
            for (char ch : value) {
                if (ch < '0' || ch > '9') {
                    return false;
                }
                const uint32_t digit = static_cast<uint32_t>(ch - '0');
                if (weight > (kPrefetchWeightMaxRouteWeight - digit) / 10U) {
                    return false;
                }
                weight = weight * 10U + digit;
            }
            if (weight == 0) {
                return false;
            }
        }
        result |= static_cast<uint64_t>(weight) <<
            (qp * kPrefetchWeightRouteWeightBits);
        ++qp;
        if (comma == std::string::npos) {
            begin = spec.size() + 1;
            break;
        }
        begin = comma + 1;
    }
    if (qp != qpNum || begin <= spec.size()) {
        return false;
    }
    packedWeights = result;
    return true;
}

bool BuildPrefetchWeightSlice(uint32_t rowBytes, uint32_t worker,
    uint32_t workerCount, uint64_t packedWeights, PrefetchWeightSlice &slice)
{
    if (rowBytes == 0 || rowBytes % kPrefetchWeightAlignment != 0 ||
        workerCount == 0 || workerCount > kPrefetchWeightMaxWorkers ||
        worker >= workerCount) {
        return false;
    }
    uint32_t totalWeight = 0;
    uint32_t beginWeight = 0;
    uint32_t endWeight = 0;
    for (uint32_t index = 0; index < workerCount; ++index) {
        const uint32_t weight = static_cast<uint32_t>(
            (packedWeights >> (index * kPrefetchWeightRouteWeightBits)) &
            kPrefetchWeightMaxRouteWeight);
        if (weight == 0 || totalWeight > UINT32_MAX - weight) {
            return false;
        }
        if (index < worker) {
            beginWeight += weight;
        }
        totalWeight += weight;
        if (index <= worker) {
            endWeight += weight;
        }
    }
    const uint64_t beginNumerator = static_cast<uint64_t>(rowBytes) * beginWeight;
    const uint64_t endNumerator = static_cast<uint64_t>(rowBytes) * endWeight;
    const uint32_t begin = static_cast<uint32_t>(
        (beginNumerator / totalWeight) / kPrefetchWeightAlignment *
        kPrefetchWeightAlignment);
    const uint32_t end = worker + 1 == workerCount ? rowBytes :
        static_cast<uint32_t>((endNumerator / totalWeight) /
            kPrefetchWeightAlignment * kPrefetchWeightAlignment);
    if (begin > end || end > rowBytes) {
        return false;
    }
    slice.offset = begin;
    slice.bytes = end - begin;
    return true;
}

int BuildPrefetchWeightLayout(const TileXRMoonEpPrefetchWeightArgsV1 &args,
    const TileXR::CommArgs &commArgs, const TileXR::TileXRUDMARegistry &registry,
    uint32_t qpNum, const char *blockDimOverride, const char *routeSpec,
    PrefetchWeightLayout &layout)
{
    if (args.structSize < sizeof(TileXRMoonEpPrefetchWeightArgsV1) ||
        args.abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || args.comm == nullptr ||
        args.plan == nullptr || args.gate == nullptr || args.up == nullptr ||
        args.down == nullptr || args.flags != TILEXR_MOONEP_FLAG_NONE ||
        args.plan->b <= 0 || args.plan->b > INT32_MAX ||
        reinterpret_cast<uintptr_t>(args.plan->expertsToCopy) % alignof(int32_t) != 0 ||
        reinterpret_cast<uintptr_t>(args.plan->status) % alignof(uint32_t) != 0 ||
        args.plan->rank != commArgs.rank || args.plan->world != commArgs.rankSize ||
        (commArgs.extraFlag & TileXR::ExtraFlag::UDMA) == 0 ||
        commArgs.udmaInfoPtr == nullptr || commArgs.udmaRegistryPtr == nullptr ||
        !TileXR::UDMARegistryValid(&registry, commArgs.rankSize) ||
        commArgs.rank < 0 || commArgs.rank >= commArgs.rankSize ||
        !SupportedWorkerCount(qpNum) || qpNum > kPrefetchWeightMaxWorkers) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if (args.gate->dtype != args.up->dtype || args.gate->dtype != args.down->dtype) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    PrefetchWeightLayout result {};
    result.rank = commArgs.rank;
    result.rankSize = commArgs.rankSize;
    result.expertsPerRank = static_cast<int32_t>(args.plan->b);
    result.qpNum = qpNum;
    if (!BuildPrefetchWeightRouteWeights(qpNum, routeSpec, result.routeWeights)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    uint64_t gateEnd = 0;
    uint64_t upEnd = 0;
    uint64_t downEnd = 0;
    if (!BuildProjection(*args.gate, result.expertsPerRank, result.rank, registry,
            result.gate, gateEnd) ||
        !BuildProjection(*args.up, result.expertsPerRank, result.rank, registry,
            result.up, upEnd) ||
        !BuildProjection(*args.down, result.expertsPerRank, result.rank, registry,
            result.down, downEnd) ||
        RangesOverlap(result.gate.registryOffset, gateEnd,
            result.up.registryOffset, upEnd) ||
        RangesOverlap(result.gate.registryOffset, gateEnd,
            result.down.registryOffset, downEnd) ||
        RangesOverlap(result.up.registryOffset, upEnd,
            result.down.registryOffset, downEnd)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    const bool hasOverride = blockDimOverride != nullptr && blockDimOverride[0] != '\0';
    uint32_t workers = qpNum;
    if (!ParseWorkerOverride(blockDimOverride, workers) || workers > qpNum ||
        (hasOverride && workers > static_cast<uint32_t>(result.expertsPerRank))) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    while (workers > static_cast<uint32_t>(result.expertsPerRank)) {
        workers >>= 1;
    }
    result.blockDim = workers;
    layout = result;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp
