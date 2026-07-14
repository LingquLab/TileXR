#include "ep_layout.h"

#include <limits>

#include "comm_args.h"
#include "ep_window.h"
#include "tilexr_udma_types.h"

namespace TileXREp {
namespace {

bool MulInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool AddInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

constexpr int64_t kDataAsFlagPayloadBytes = 480;
constexpr int64_t kDataAsFlagBlockBytes = 512;
constexpr int64_t kDataAsFlagPingPongBuffers = 2;

bool DataAsFlagSlotBytesChecked(int64_t slotBytes, int64_t *out)
{
    if (out == nullptr || slotBytes <= 0) {
        return false;
    }
    int64_t blocks = 0;
    if (!AddInt64(slotBytes, kDataAsFlagPayloadBytes - 1, &blocks)) {
        return false;
    }
    blocks /= kDataAsFlagPayloadBytes;
    return MulInt64(blocks, kDataAsFlagBlockBytes, out);
}

bool IsPositive(int64_t value)
{
    return value > 0;
}

} // namespace

int64_t TileXREpAlignUp(int64_t value, int64_t alignment)
{
    if (value < 0 || alignment <= 0) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    const int64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    int64_t aligned = 0;
    if (!AddInt64(value, alignment - remainder, &aligned)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return aligned;
}

int64_t TileXREpDataTypeSize(TileXR::TileXRDataType dtype)
{
    if (dtype == TileXR::TILEXR_DATA_TYPE_INT8) {
        return 1;
    }
    if (dtype == TileXR::TILEXR_DATA_TYPE_FP16 || dtype == TileXR::TILEXR_DATA_TYPE_BFP16) {
        return 2;
    }
    return TileXR::TILEXR_INVALID_VALUE;
}

bool TileXREpIsSupportedDataType(TileXR::TileXRDataType dtype)
{
    return TileXREpDataTypeSize(dtype) > 0;
}

int64_t TileXREpUdmaOperationBytes(int64_t totalBytes, int64_t rankSize, int64_t slotBytes)
{
    if (totalBytes <= 0 || rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE || slotBytes <= 0) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    int64_t alignedTotal = TileXREpAlignUp(totalBytes, kEpWindowAlignmentBytes);
    if (alignedTotal == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    int64_t readyBytes = 0;
    int64_t doubleTotal = 0;
    int64_t readyOffset = 0;
    if (!MulInt64(static_cast<int64_t>(rankSize), static_cast<int64_t>(sizeof(uint64_t)), &readyBytes) ||
        !MulInt64(alignedTotal, 2, &doubleTotal) ||
        !AddInt64(doubleTotal, readyBytes, &readyOffset)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    int64_t relaySlotsOffset = TileXREpAlignUp(readyOffset, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    if (relaySlotsOffset == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    int64_t rankSquare = 0;
    int64_t relayBytes = 0;
    int64_t relayReadyBase = 0;
    if (!MulInt64(rankSize, rankSize, &rankSquare) ||
        !MulInt64(rankSquare, slotBytes, &relayBytes) ||
        !AddInt64(relaySlotsOffset, relayBytes, &relayReadyBase)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    relayReadyBase = TileXREpAlignUp(relayReadyBase, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    if (relayReadyBase == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    int64_t relayReadyEnd = 0;
    if (!AddInt64(relayReadyBase, readyBytes, &relayReadyEnd)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return TileXREpAlignUp(relayReadyEnd, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
}

int64_t TileXREpUdmaRequiredWorkspaceBytes(int64_t totalBytes, int64_t rankSize, int64_t slotBytes)
{
    int64_t operationBytes = TileXREpUdmaOperationBytes(totalBytes, rankSize, slotBytes);
    if (operationBytes == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    int64_t twoOperations = 0;
    int64_t withStatus = 0;
    if (!MulInt64(operationBytes, 2, &twoOperations) ||
        !AddInt64(twoOperations, static_cast<int64_t>(sizeof(uint64_t)), &withStatus)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return TileXREpAlignUp(withStatus, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
}

int64_t TileXREpDataAsFlagSlotBytes(int64_t slotBytes)
{
    int64_t encoded = 0;
    return DataAsFlagSlotBytesChecked(slotBytes, &encoded) ? encoded : TileXR::TILEXR_INVALID_VALUE;
}

int64_t TileXREpDataAsFlagTotalBytes(int64_t rankSize, int64_t slotBytes)
{
    if (rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    const int64_t encodedSlot = TileXREpDataAsFlagSlotBytes(slotBytes);
    int64_t total = 0;
    if (encodedSlot == TileXR::TILEXR_INVALID_VALUE || !MulInt64(rankSize, encodedSlot, &total)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return total;
}

int64_t TileXREpDataAsFlagPingPongTotalBytes(int64_t rankSize, int64_t slotBytes)
{
    const int64_t total = TileXREpDataAsFlagTotalBytes(rankSize, slotBytes);
    int64_t pingPongTotal = 0;
    if (total == TileXR::TILEXR_INVALID_VALUE ||
        !MulInt64(total, kDataAsFlagPingPongBuffers, &pingPongTotal)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return pingPongTotal;
}

int64_t TileXREpCombineDataAsFlagRecvWindowOffset(int64_t totalBytes, int64_t rankSize, int64_t slotBytes)
{
    return TileXREpUdmaOperationBytes(totalBytes, rankSize, slotBytes);
}

int64_t TileXREpCombineDataAsFlagSendWindowOffset(int64_t totalBytes, int64_t rankSize, int64_t slotBytes)
{
    const int64_t recvOffset = TileXREpCombineDataAsFlagRecvWindowOffset(totalBytes, rankSize, slotBytes);
    const int64_t alignedTotal = TileXREpAlignUp(totalBytes, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    int64_t sendOffset = 0;
    if (recvOffset == TileXR::TILEXR_INVALID_VALUE || alignedTotal == TileXR::TILEXR_INVALID_VALUE ||
        !AddInt64(recvOffset, alignedTotal, &sendOffset)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return sendOffset;
}

int64_t TileXREpCombineDataAsFlagSendRoundOffset(
    int64_t totalBytes, int64_t rankSize, int64_t slotBytes, int64_t roundIndex)
{
    const int64_t sendOffset = TileXREpCombineDataAsFlagSendWindowOffset(totalBytes, rankSize, slotBytes);
    const int64_t dafTotal = TileXREpDataAsFlagTotalBytes(rankSize, slotBytes);
    const int64_t alignedDafTotal = TileXREpAlignUp(dafTotal, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    int64_t roundBytes = 0;
    int64_t roundOffset = 0;
    if ((roundIndex != 0 && roundIndex != 1) || sendOffset == TileXR::TILEXR_INVALID_VALUE ||
        alignedDafTotal == TileXR::TILEXR_INVALID_VALUE || !MulInt64(roundIndex, alignedDafTotal, &roundBytes) ||
        !AddInt64(sendOffset, roundBytes, &roundOffset)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return roundOffset;
}

int64_t TileXREpCombineDataAsFlagRemoteRecvWindowOffset(int64_t totalBytes, int64_t rankSize, int64_t slotBytes)
{
    const int64_t sendOffset = TileXREpCombineDataAsFlagSendWindowOffset(totalBytes, rankSize, slotBytes);
    const int64_t dafTotal = TileXREpDataAsFlagTotalBytes(rankSize, slotBytes);
    const int64_t alignedDafTotal = TileXREpAlignUp(dafTotal, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    int64_t pingPongDafTotal = 0;
    int64_t remoteRecvOffset = 0;
    if (sendOffset == TileXR::TILEXR_INVALID_VALUE || alignedDafTotal == TileXR::TILEXR_INVALID_VALUE ||
        !MulInt64(kDataAsFlagPingPongBuffers, alignedDafTotal, &pingPongDafTotal) ||
        !AddInt64(sendOffset, pingPongDafTotal, &remoteRecvOffset)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return remoteRecvOffset;
}

int64_t TileXREpCombineDataAsFlagRemoteRecvRoundOffset(
    int64_t totalBytes, int64_t rankSize, int64_t slotBytes, int64_t roundIndex)
{
    const int64_t remoteRecvOffset =
        TileXREpCombineDataAsFlagRemoteRecvWindowOffset(totalBytes, rankSize, slotBytes);
    const int64_t dafTotal = TileXREpDataAsFlagTotalBytes(rankSize, slotBytes);
    const int64_t alignedDafTotal = TileXREpAlignUp(dafTotal, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    int64_t roundBytes = 0;
    int64_t roundOffset = 0;
    if ((roundIndex != 0 && roundIndex != 1) || remoteRecvOffset == TileXR::TILEXR_INVALID_VALUE ||
        alignedDafTotal == TileXR::TILEXR_INVALID_VALUE || !MulInt64(roundIndex, alignedDafTotal, &roundBytes) ||
        !AddInt64(remoteRecvOffset, roundBytes, &roundOffset)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return roundOffset;
}

int64_t TileXREpCombineDataAsFlagStatusOffset(int64_t totalBytes, int64_t rankSize, int64_t slotBytes)
{
    const int64_t remoteRecvOffset =
        TileXREpCombineDataAsFlagRemoteRecvWindowOffset(totalBytes, rankSize, slotBytes);
    const int64_t dafTotal = TileXREpDataAsFlagTotalBytes(rankSize, slotBytes);
    const int64_t alignedDafTotal = TileXREpAlignUp(dafTotal, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    int64_t pingPongDafTotal = 0;
    int64_t statusOffset = 0;
    if (remoteRecvOffset == TileXR::TILEXR_INVALID_VALUE || alignedDafTotal == TileXR::TILEXR_INVALID_VALUE ||
        !MulInt64(kDataAsFlagPingPongBuffers, alignedDafTotal, &pingPongDafTotal) ||
        !AddInt64(remoteRecvOffset, pingPongDafTotal, &statusOffset)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return statusOffset;
}

int64_t TileXREpCombineDataAsFlagRequiredWorkspaceBytes(int64_t totalBytes, int64_t rankSize, int64_t slotBytes)
{
    const int64_t statusOffset = TileXREpCombineDataAsFlagStatusOffset(totalBytes, rankSize, slotBytes);
    int64_t withStatus = 0;
    if (statusOffset == TileXR::TILEXR_INVALID_VALUE ||
        !AddInt64(statusOffset, static_cast<int64_t>(sizeof(uint64_t)), &withStatus)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return TileXREpAlignUp(withStatus, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
}

int TileXREpBuildWindowConfig(int64_t rankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, TileXR::TileXRDataType dtype, EpWindowConfig *out)
{
    return TileXREpBuildDispatchWindowConfig(rankSize, bs, h, topK, moeExpertNum, 0, 0, dtype, out);
}

int TileXREpBuildDispatchWindowConfig(int64_t rankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum, TileXR::TileXRDataType dtype,
    EpWindowConfig *out)
{
    return TileXREpBuildDispatchWindowConfigForExpertRanks(rankSize, rankSize, bs, h, topK, moeExpertNum,
        sharedExpertNum, sharedExpertRankNum, dtype, out);
}

int TileXREpBuildDispatchWindowConfigForExpertRanks(int64_t rankSize, int64_t expertRankSize, int64_t bs,
    int64_t h, int64_t topK, int64_t moeExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    TileXR::TileXRDataType dtype, EpWindowConfig *out)
{
    return TileXREpBuildDispatchWindowConfigForExpertRanks(rankSize, expertRankSize, bs, h, topK, moeExpertNum,
        sharedExpertNum, sharedExpertRankNum, dtype, 0, out);
}

int TileXREpBuildDispatchWindowConfigForExpertRanks(int64_t rankSize, int64_t expertRankSize, int64_t bs,
    int64_t h, int64_t topK, int64_t moeExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    TileXR::TileXRDataType dtype, int64_t payloadExtraBytes, EpWindowConfig *out)
{
    const int64_t moeRankNum = expertRankSize - sharedExpertRankNum;
    if (out == nullptr || !IsPositive(rankSize) || rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        !IsPositive(expertRankSize) || expertRankSize > rankSize || !IsPositive(bs) || !IsPositive(h) ||
        !IsPositive(topK) || !IsPositive(moeExpertNum) || sharedExpertNum < 0 || sharedExpertRankNum < 0 ||
        sharedExpertRankNum > expertRankSize || !IsPositive(moeRankNum) ||
        moeExpertNum % moeRankNum != 0 || payloadExtraBytes < 0 || !TileXREpIsSupportedDataType(dtype)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    EpWindowConfig next {};
    next.rankSize = rankSize;
    next.bs = bs;
    next.h = h;
    next.topK = topK;
    next.moeExpertNum = moeExpertNum;
    next.localExpertNum = sharedExpertRankNum > 0 ? moeExpertNum / moeRankNum : moeExpertNum / expertRankSize;
    next.dtypeBytes = TileXREpDataTypeSize(dtype);

    int64_t routesPerToken = 0;
    if (!AddInt64(topK, sharedExpertNum, &routesPerToken) || !MulInt64(bs, routesPerToken, &next.maxRoutesPerSrc)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    if (!MulInt64(h, next.dtypeBytes, &next.rowBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.payloadRowBytes = next.rowBytes;

    int64_t payloadBytes = 0;
    int64_t scaleBytes = 0;
    if (!MulInt64(next.maxRoutesPerSrc, next.payloadRowBytes, &payloadBytes) ||
        !MulInt64(next.maxRoutesPerSrc, payloadExtraBytes, &scaleBytes) ||
        !AddInt64(payloadBytes, scaleBytes, &payloadBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.payloadBytesPerSlot = TileXREpAlignUp(payloadBytes, kEpWindowAlignmentBytes);
    if (next.payloadBytesPerSlot == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t assistTupleBytes = 0;
    int64_t assistBytes = 0;
    if (!MulInt64(kEpAssistTupleInts, static_cast<int64_t>(sizeof(int32_t)), &assistTupleBytes) ||
        !MulInt64(next.maxRoutesPerSrc, assistTupleBytes, &assistBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.assistBytesPerSlot = TileXREpAlignUp(assistBytes, kEpWindowAlignmentBytes);
    if (next.assistBytesPerSlot == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t slotPayloadEnd = 0;
    if (!AddInt64(kEpSrcSlotHeaderBytes, next.payloadBytesPerSlot, &slotPayloadEnd) ||
        !AddInt64(slotPayloadEnd, next.assistBytesPerSlot, &next.slotBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.slotBytes = TileXREpAlignUp(next.slotBytes, kEpWindowAlignmentBytes);
    if (next.slotBytes == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t slotsBytes = 0;
    if (!MulInt64(rankSize, next.slotBytes, &slotsBytes) ||
        !AddInt64(kEpWindowHeaderBytes, slotsBytes, &next.totalBytes) ||
        next.totalBytes > TileXR::IPC_BUFF_MAX_SIZE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    *out = next;
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpDstRank(int32_t expertId, int64_t localExpertNum)
{
    if (expertId < 0 || localExpertNum <= 0) {
        return static_cast<int>(TileXR::TILEXR_INVALID_VALUE);
    }
    return expertId / localExpertNum;
}

int TileXREpLocalExpert(int32_t expertId, int64_t localExpertNum)
{
    if (expertId < 0 || localExpertNum <= 0) {
        return static_cast<int>(TileXR::TILEXR_INVALID_VALUE);
    }
    return expertId % localExpertNum;
}

} // namespace TileXREp
