#include "ep_urma_combine_layout.h"

#include <limits>

#include "comm_args.h"
#include "ep_urma_combine.h"
#include "tilexr_types.h"

namespace TileXREp {
namespace {

bool AddInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool MulInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0 ||
        (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool AlignUpInt64(int64_t value, int64_t alignment, int64_t *out)
{
    if (out == nullptr || value < 0 || alignment <= 0) {
        return false;
    }
    const int64_t remainder = value % alignment;
    return remainder == 0 ? (*out = value, true) : AddInt64(value, alignment - remainder, out);
}

bool AddAlignedRegion(int64_t *offset, int64_t bytes, int64_t alignment, int64_t *regionOffset)
{
    int64_t aligned = 0;
    if (offset == nullptr || regionOffset == nullptr || bytes < 0 ||
        !AlignUpInt64(*offset, alignment, &aligned) || !AddInt64(aligned, bytes, offset)) {
        return false;
    }
    *regionOffset = aligned;
    return true;
}

} // namespace

int TileXREpBuildUrmaCombineWorkspaceConfig(int64_t rankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t selfSendCnt, EpUrmaCombineWorkspaceConfig *out)
{
    if (out == nullptr || rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE || bs <= 0 || h <= 0 ||
        h > kEpUrmaCombineMaxHidden || topK <= 0 || topK > kEpUrmaCombineMaxTopK || selfSendCnt < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    EpUrmaCombineWorkspaceConfig next {};
    next.rankSize = rankSize;
    next.bs = bs;
    next.h = h;
    next.topK = topK;
    next.selfSendCnt = selfSendCnt;

    if (!AlignUpInt64(h, kEpUrmaCombineQuantHeaderBytes, &next.quantDataBytes) ||
        !AddInt64(kEpUrmaCombineQuantHeaderBytes, next.quantDataBytes, &next.commBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.blockCount = (next.commBytes + kEpUrmaCombinePayloadBytes - 1) / kEpUrmaCombinePayloadBytes;
    if (next.blockCount <= 0 || next.blockCount > kEpUrmaCombineMaxBlocksPerRoute ||
        !MulInt64(next.blockCount, kEpUrmaCombineDataBlockBytes, &next.routeStride) ||
        !MulInt64(bs, topK, &next.routeCount) ||
        !MulInt64(next.routeCount, next.routeStride, &next.rxWindowBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t offset = kEpUrmaCombineHeaderBytes;
    int64_t roundDoneBytes = 0;
    int64_t rxLaneDoneBytes = 0;
    int64_t senderDoneBytes = 0;
    int64_t txReadyBytes = 0;
    int64_t txDataBytes = 0;
    if (!MulInt64(rankSize, kEpUrmaCombineCacheLineBytes, &roundDoneBytes) ||
        !MulInt64(kEpUrmaCombinePackLaneCount, kEpUrmaCombineCacheLineBytes, &rxLaneDoneBytes) ||
        !MulInt64(kEpUrmaCombineSendLaneCount, kEpUrmaCombineCacheLineBytes, &senderDoneBytes) ||
        !MulInt64(selfSendCnt, kEpUrmaCombineCacheLineBytes, &txReadyBytes) ||
        !MulInt64(selfSendCnt, next.routeStride, &txDataBytes) ||
        !AddAlignedRegion(&offset, next.rxWindowBytes, kEpUrmaCombineDataBlockBytes, &next.rxWindowOffsets[0]) ||
        !AddAlignedRegion(&offset, next.rxWindowBytes, kEpUrmaCombineDataBlockBytes, &next.rxWindowOffsets[1]) ||
        !AddAlignedRegion(&offset, roundDoneBytes, kEpUrmaCombineCacheLineBytes, &next.roundDoneOffsets[0]) ||
        !AddAlignedRegion(&offset, roundDoneBytes, kEpUrmaCombineCacheLineBytes, &next.roundDoneOffsets[1]) ||
        !AddAlignedRegion(&offset, rxLaneDoneBytes, kEpUrmaCombineCacheLineBytes, &next.rxLaneDoneOffset) ||
        !AddAlignedRegion(&offset, senderDoneBytes, kEpUrmaCombineCacheLineBytes, &next.senderDoneOffset) ||
        !AddAlignedRegion(&offset, kEpUrmaCombineCacheLineBytes, kEpUrmaCombineCacheLineBytes,
            &next.roundPublishOffset)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (kEpUrmaCombineDeferredRoundCredit &&
        !AddAlignedRegion(&offset, kEpUrmaCombineCacheLineBytes, kEpUrmaCombineCacheLineBytes,
            &next.roundCreditOffset)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (kEpUrmaCombineStartGate &&
        !AddAlignedRegion(&offset, roundDoneBytes, kEpUrmaCombineCacheLineBytes, &next.startGateOffset)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!AddAlignedRegion(&offset, kEpUrmaCombineCacheLineBytes, kEpUrmaCombineCacheLineBytes,
            &next.errorStatusOffset)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    next.fixedBytes = offset;
    if (!AddAlignedRegion(&offset, txReadyBytes, kEpUrmaCombineCacheLineBytes, &next.txReadyOffset) ||
        !AddAlignedRegion(&offset, txDataBytes, kEpUrmaCombineDataBlockBytes, &next.txDataOffset)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.requiredBytes = offset;
    *out = next;
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXREp
