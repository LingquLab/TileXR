#include "planner_layout.h"

#include <cerrno>
#include <cstdlib>
#include <limits>

#include "comm_args.h"
#include "planner_common.h"
#include "tilexr_types.h"

namespace TileXRMoonEp {
namespace {

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool AppendRegion(uint64_t elementCount, uint64_t elementBytes, uint64_t *cursor, uint64_t *offset)
{
    if (cursor == nullptr || offset == nullptr) {
        return false;
    }
    const uint64_t aligned = TileXRMoonEpAlignUp(*cursor, kPlannerWorkspaceAlignment);
    if (aligned == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    uint64_t bytes = 0;
    uint64_t end = 0;
    if (!CheckedMul(elementCount, elementBytes, &bytes) || !CheckedAdd(aligned, bytes, &end)) {
        return false;
    }
    *offset = aligned;
    *cursor = end;
    return true;
}

bool PositiveAndFitsInt32(int64_t value)
{
    return value > 0 && value <= std::numeric_limits<int32_t>::max();
}

bool ResolveBlockDim(int64_t rankSize, int64_t *blockDim)
{
    if (blockDim == nullptr) {
        return false;
    }
    const char *value = std::getenv("TILEXR_MOONEP_PLANNER_BLOCK_DIM");
    int64_t selected = kPlannerAivBlockCount;
    if (value != nullptr && value[0] != '\0') {
        errno = 0;
        char *end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (errno != 0 || end == value || end == nullptr || *end != '\0') {
            return false;
        }
        selected = static_cast<int64_t>(parsed);
    }
    if (selected <= 0 || selected > kPlannerAivBlockCount || selected < rankSize) {
        return false;
    }
    *blockDim = selected;
    return true;
}

} // namespace

uint64_t TileXRMoonEpAlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0 || value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

int TileXRMoonEpBuildPlannerLayout(int64_t rankSize, int64_t s, int64_t k,
    int64_t expertCount, PlannerLayout *out)
{
    if (out == nullptr || rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        !PositiveAndFitsInt32(s) || k <= 0 || k > kPlannerMaxTopK ||
        !PositiveAndFitsInt32(expertCount) || expertCount > kPlannerMaxExpertCount ||
        expertCount % rankSize != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t blockDim = 0;
    if (!ResolveBlockDim(rankSize, &blockDim)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const uint64_t rankSizeU = static_cast<uint64_t>(rankSize);
    const uint64_t sU = static_cast<uint64_t>(s);
    const uint64_t kU = static_cast<uint64_t>(k);
    const uint64_t expertCountU = static_cast<uint64_t>(expertCount);
    const uint64_t expertsPerRankU = expertCountU / rankSizeU;

    uint64_t routeCount = 0;
    uint64_t encodedCapacity = 0;
    if (!CheckedMul(sU, kU, &routeCount) || routeCount == 0 ||
        !CheckedMul(rankSizeU, routeCount, &encodedCapacity) ||
        routeCount > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        encodedCapacity > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1U) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    uint64_t rankExpertElements = 0;
    uint64_t histogramElements = 0;
    uint64_t rankSquareElements = 0;
    if (!CheckedMul(rankSizeU, expertCountU, &rankExpertElements) ||
        !CheckedMul(static_cast<uint64_t>(kPlannerAivBlockCount), expertCountU, &histogramElements) ||
        !CheckedMul(rankSizeU, rankSizeU, &rankSquareElements)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    PlannerLayout next {};
    uint64_t cursor = 0;
    if (!AppendRegion(rankExpertElements, sizeof(int32_t), &cursor, &next.tpePrefixOffset) ||
        !AppendRegion(histogramElements, sizeof(int32_t), &cursor, &next.blockHistogramOffset) ||
        !AppendRegion(rankExpertElements, sizeof(int32_t), &cursor, &next.allocPrefixOffset) ||
        !AppendRegion(rankExpertElements, sizeof(int32_t), &cursor, &next.expertOffsetsOffset) ||
        !AppendRegion(rankSquareElements, sizeof(int32_t), &cursor, &next.zOffset) ||
        !AppendRegion(rankSizeU, sizeof(int32_t), &cursor, &next.groupTotalsOffset)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const uint64_t workspaceBytes = TileXRMoonEpAlignUp(cursor, kPlannerWorkspaceAlignment);
    uint64_t peerPublicationBytes = 0;
    if (workspaceBytes == std::numeric_limits<uint64_t>::max() ||
        !CheckedMul(expertCountU, sizeof(int32_t), &peerPublicationBytes) ||
        peerPublicationBytes > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    next.rankSize = rankSize;
    next.s = s;
    next.k = k;
    next.expertCount = expertCount;
    next.expertsPerRank = static_cast<int64_t>(expertsPerRankU);
    next.routeCount = static_cast<int64_t>(routeCount);
    next.dispatchedCapacity = static_cast<int64_t>(routeCount);
    next.blockDim = blockDim;
    next.workspaceBytes = workspaceBytes;
    next.peerPublicationBytes = peerPublicationBytes;
    *out = next;
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXRMoonEp
