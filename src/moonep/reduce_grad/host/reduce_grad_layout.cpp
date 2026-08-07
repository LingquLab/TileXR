#include "reduce_grad_layout.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <limits>

#include "comm_args.h"
#include "tilexr_moonep.h"
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

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? 0 : value / alignment * alignment;
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t *out)
{
    if (out == nullptr || alignment == 0) {
        return false;
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? (*out = value, true) :
        CheckedAdd(value, alignment - remainder, out);
}

bool DivideRoundUp(uint64_t value, uint64_t divisor, uint64_t *out)
{
    if (out == nullptr || divisor == 0) {
        return false;
    }
    *out = value / divisor + (value % divisor != 0 ? 1U : 0U);
    return true;
}

bool ResolveBlockDim(int64_t rankSize, bool, int64_t *blockDim,
    int64_t *controlBlockCount)
{
    if (blockDim == nullptr || controlBlockCount == nullptr) {
        return false;
    }
    const char *value = std::getenv("TILEXR_MOONEP_REDUCE_GRAD_BLOCK_DIM");
    int64_t selected = kReduceGradMaxAivBlockCount;
    if (value != nullptr && value[0] != '\0') {
        errno = 0;
        char *end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (errno != 0 || end == value || end == nullptr || *end != '\0') {
            return false;
        }
        selected = static_cast<int64_t>(parsed);
    }
    if (selected <= 0 || selected > kReduceGradMaxAivBlockCount) {
        return false;
    }

    int64_t controls = 0;
    if (rankSize > 1) {
        controls = rankSize - 1;
        if (selected <= controls) {
            return false;
        }
    }
    *blockDim = selected;
    *controlBlockCount = controls;
    return true;
}

} // namespace

uint64_t TileXRMoonEpReduceGradPeerWindowBytes()
{
    return TileXR::IPC_BUFF_MAX_SIZE > 0 ?
        static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) : 0;
}

int TileXRMoonEpBuildReduceGradLayout(int64_t rank, int64_t rankSize,
    int64_t expertCount, const uint64_t rowElements[kReduceGradProjectionCount],
    uint64_t peerWindowBytes, uint64_t requestedUdmaChunkBytes,
    ReduceGradLayout *out)
{
    if (out == nullptr || rowElements == nullptr || rankSize <= 0 ||
        rankSize > TileXR::TILEXR_MAX_RANK_SIZE || rank < 0 || rank >= rankSize ||
        expertCount <= 0 || expertCount % rankSize != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    ReduceGradLayout next {};
    next.rank = rank;
    next.rankSize = rankSize;
    next.expertCount = expertCount;
    next.expertsPerRank = expertCount / rankSize;
    next.peerWindowBytes = peerWindowBytes;

    bool usesPeer = false;
    bool usesUdma = false;
    uint64_t maxUdmaRowBytes = 0;
    for (uint32_t q = 0; q < kReduceGradProjectionCount; ++q) {
        uint64_t bytes = 0;
        if (rowElements[q] == 0 || !CheckedMul(rowElements[q], sizeof(float), &bytes)) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        next.rowElements[q] = rowElements[q];
        next.rowBytes[q] = bytes;
        if (rankSize == 1 || bytes <= kReduceGradUdmaThresholdBytes) {
            next.transports[q] = TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER;
            usesPeer = true;
        } else {
            next.transports[q] = TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA;
            usesUdma = true;
            if (bytes > maxUdmaRowBytes) {
                maxUdmaRowBytes = bytes;
            }
        }
    }

    if (!ResolveBlockDim(rankSize, usesUdma, &next.blockDim, &next.controlBlockCount)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    if (usesPeer) {
        if (peerWindowBytes <= kReduceGradStateWindowBytes) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        next.peerRecordBaseOffset = kReduceGradStateWindowBytes;
        next.peerHalfBytes = AlignDown(
            (peerWindowBytes - kReduceGradStateWindowBytes) / 2,
            kReduceGradDataAsFlagRecordBytes);
        const uint64_t incomingSlots = static_cast<uint64_t>(expertCount);
        next.peerSlotStrideBytes = AlignDown(next.peerHalfBytes / incomingSlots,
            kReduceGradDataAsFlagRecordBytes);
        if (next.peerSlotStrideBytes < kReduceGradDataAsFlagRecordBytes ||
            !CheckedMul(next.peerSlotStrideBytes / kReduceGradDataAsFlagRecordBytes,
                kReduceGradDataAsFlagPayloadBytes, &next.peerChunkPayloadBytes) ||
            next.peerChunkPayloadBytes == 0) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        for (uint32_t q = 0; q < kReduceGradProjectionCount; ++q) {
            if (next.transports[q] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
                !DivideRoundUp(next.rowBytes[q], next.peerChunkPayloadBytes,
                    &next.peerChunkCounts[q])) {
                return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
            }
        }
    }

    if (usesUdma) {
        uint64_t desiredChunk = requestedUdmaChunkBytes == 0 ?
            kReduceGradDefaultUdmaChunkBytes : requestedUdmaChunkBytes;
        if (desiredChunk < kReduceGradUdmaThresholdBytes || desiredChunk > UINT32_MAX) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        if (desiredChunk > maxUdmaRowBytes) {
            desiredChunk = maxUdmaRowBytes;
        }
        if (!AlignUp(desiredChunk, kReduceGradUdmaAlignment, &next.udmaChunkBytes) ||
            next.udmaChunkBytes > UINT32_MAX) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }

        uint64_t stateBytes = 0;
        uint64_t stageCount = 0;
        uint64_t payloadBytes = 0;
        uint64_t cursor = 0;
        if (!CheckedMul(static_cast<uint64_t>(rankSize), kReduceGradUdmaPeerStateBytes,
                &stateBytes) ||
            !AlignUp(stateBytes, kReduceGradUdmaAlignment, &cursor) ||
            !CheckedMul(static_cast<uint64_t>(rankSize), 2, &stageCount) ||
            !CheckedMul(stageCount, next.udmaChunkBytes, &payloadBytes)) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        next.udmaStateOffset = 0;
        next.udmaOutboundOffset = cursor;
        if (!CheckedAdd(cursor, payloadBytes, &cursor) ||
            !AlignUp(cursor, kReduceGradUdmaAlignment, &cursor)) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        next.udmaInboundOffset = cursor;
        if (!CheckedAdd(cursor, payloadBytes, &cursor) ||
            !AlignUp(cursor, kReduceGradUdmaWorkspaceAlignment, &next.workspaceBytes)) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        for (uint32_t q = 0; q < kReduceGradProjectionCount; ++q) {
            if (next.transports[q] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA &&
                !DivideRoundUp(next.rowBytes[q], next.udmaChunkBytes,
                    &next.udmaChunkCounts[q])) {
                return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
            }
        }
        uint64_t sequenceCount = 0;
        for (uint32_t q = 0; q < kReduceGradProjectionCount; ++q) {
            uint64_t projectionSequences = 0;
            if (!CheckedMul(static_cast<uint64_t>(next.expertsPerRank),
                    next.udmaChunkCounts[q], &projectionSequences) ||
                !CheckedAdd(sequenceCount, projectionSequences, &sequenceCount)) {
                return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
            }
        }
        if (sequenceCount > UINT32_MAX) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
    }

    *out = next;
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXRMoonEp
