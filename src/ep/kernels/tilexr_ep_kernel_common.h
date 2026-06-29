#pragma once

#include "comm_args.h"
#include "ep_window.h"
#include "kernel_operator.h"
#include "tilexr_sync.h"
#include "tilexr_udma.h"

namespace {

constexpr uint32_t kEpUbBytes = 64 * 1024;
constexpr uint32_t kEpSyncUbBytes = 4 * 1024;
constexpr uint32_t kEpCopyTileBytes = kEpUbBytes - kEpSyncUbBytes;
constexpr uint32_t kEpScalarUbBytes = 64;
constexpr uint32_t kEpScalarUbOffset = kEpSyncUbBytes - kEpScalarUbBytes;
constexpr int64_t kTileXrDataTypeFp16 = 3;

__aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    if (alignment <= 0) {
        return value;
    }
    const int64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

__aicore__ inline int64_t SlotOffset(int64_t srcRank, int64_t slotBytes)
{
    return TileXREp::kEpWindowHeaderBytes + srcRank * slotBytes;
}

__aicore__ inline int64_t PayloadOffset(int64_t srcRank, int64_t slotBytes)
{
    return SlotOffset(srcRank, slotBytes) + TileXREp::kEpSrcSlotHeaderBytes;
}

__aicore__ inline int64_t AssistOffset(int64_t srcRank, int64_t slotBytes, int64_t payloadBytesPerSlot)
{
    return PayloadOffset(srcRank, slotBytes) + payloadBytesPerSlot;
}

__aicore__ inline int64_t UDMAReadyOffset(int64_t totalBytes, int32_t rank)
{
    return AlignUp(totalBytes, TileXREp::kEpWindowAlignmentBytes) * 2 +
        static_cast<int64_t>(rank) * static_cast<int64_t>(sizeof(uint64_t));
}

__aicore__ inline int64_t UDMARecvWindowOffset(int64_t totalBytes)
{
    return AlignUp(totalBytes, TileXREp::kEpWindowAlignmentBytes);
}

__aicore__ inline int64_t UDMARelaySlotsOffset(int64_t totalBytes, int32_t rankSize)
{
    const int64_t readyBytes = UDMAReadyOffset(totalBytes, rankSize);
    return AlignUp(readyBytes, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
}

__aicore__ inline int64_t UDMARelaySlotOffset(
    int64_t totalBytes, int32_t rankSize, int32_t srcRank, int32_t dstRank, int64_t slotBytes)
{
    const int64_t slotIndex = static_cast<int64_t>(srcRank) * static_cast<int64_t>(rankSize) + dstRank;
    return UDMARelaySlotsOffset(totalBytes, rankSize) + slotIndex * slotBytes;
}

__aicore__ inline int64_t UDMARelayReadyBaseOffset(int64_t totalBytes, int32_t rankSize, int64_t slotBytes)
{
    const int64_t relayBytes = static_cast<int64_t>(rankSize) * static_cast<int64_t>(rankSize) * slotBytes;
    return AlignUp(UDMARelaySlotsOffset(totalBytes, rankSize) + relayBytes, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
}

__aicore__ inline int64_t UDMARelayReadyOffset(
    int64_t totalBytes, int32_t rankSize, int64_t slotBytes, int32_t srcRank)
{
    return UDMARelayReadyBaseOffset(totalBytes, rankSize, slotBytes) +
        static_cast<int64_t>(srcRank) * static_cast<int64_t>(sizeof(uint64_t));
}

__aicore__ inline int64_t UDMAOperationBytes(int64_t totalBytes, int32_t rankSize, int64_t slotBytes)
{
    const int64_t relayReadyBytes = static_cast<int64_t>(rankSize) * static_cast<int64_t>(sizeof(uint64_t));
    return AlignUp(UDMARelayReadyBaseOffset(totalBytes, rankSize, slotBytes) + relayReadyBytes,
        TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
}

__aicore__ inline int64_t UDMASecondOperationOffset(int64_t totalBytes, int32_t rankSize, int64_t slotBytes)
{
    return UDMAOperationBytes(totalBytes, rankSize, slotBytes);
}

__aicore__ inline void CopyBytesGmToGm(
    GM_ADDR dstGM, GM_ADDR srcGM, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf, int64_t bytes)
{
    if (dstGM == nullptr || srcGM == nullptr || bytes <= 0) {
        return;
    }

    AscendC::LocalTensor<uint8_t> local =
        tBuf.GetWithOffset<uint8_t>(kEpCopyTileBytes, kEpSyncUbBytes);
    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(srcGM), bytes);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dstGM), bytes);

    for (int64_t copied = 0; copied < bytes; copied += kEpCopyTileBytes) {
        int64_t tileBytes = bytes - copied;
        if (tileBytes > static_cast<int64_t>(kEpCopyTileBytes)) {
            tileBytes = kEpCopyTileBytes;
        }

        AscendC::DataCopyPadParams padParams {false, 0, 0, 0};
        AscendC::DataCopyParams copyParams {1, static_cast<uint16_t>(tileBytes), 0, 0};
        AscendC::DataCopyPad(local, src[copied], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[copied], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TileXREpInvalidateLocalCacheLines(GM_ADDR localGM, int64_t bytes)
{
    if (localGM == nullptr || bytes <= 0) {
        return;
    }
    __gm__ uint8_t *start = reinterpret_cast<__gm__ uint8_t *>(
        reinterpret_cast<uint64_t>(localGM) / TileXR::TILEXR_UDMA_CACHE_LINE_SIZE *
        TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    __gm__ uint8_t *end = reinterpret_cast<__gm__ uint8_t *>(
        (reinterpret_cast<uint64_t>(localGM) + static_cast<uint64_t>(bytes) - 1) /
        TileXR::TILEXR_UDMA_CACHE_LINE_SIZE * TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint64_t offset = 0; offset <= static_cast<uint64_t>(end - start);
        offset += TileXR::TILEXR_UDMA_CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[offset]);
        __asm__ __volatile__("");
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline int32_t LoadInt32FromGm(GM_ADDR srcGM, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int32_t> local =
        tBuf.GetWithOffset<int32_t>(kEpScalarUbBytes / sizeof(int32_t), kEpScalarUbOffset);
    AscendC::GlobalTensor<int32_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(srcGM), 1);

    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return local.GetValue(0);
}

__aicore__ inline uint64_t LoadUint64FromGm(GM_ADDR srcGM, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<uint64_t> local =
        tBuf.GetWithOffset<uint64_t>(kEpScalarUbBytes / sizeof(uint64_t), kEpScalarUbOffset);
    AscendC::GlobalTensor<uint64_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(srcGM), 1);

    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint64_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return local.GetValue(0);
}

__aicore__ inline void StoreInt64WordsToGm(
    GM_ADDR dstGM, AscendC::LocalTensor<int64_t> &local, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf,
    uint32_t bytes)
{
    (void)tBuf;
    AscendC::GlobalTensor<int64_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(dstGM), bytes / sizeof(int64_t));

    AscendC::DataCopyExtParams copyParams {1, bytes, 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(dst, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline int64_t PackInt32Pair(uint32_t low, int32_t high)
{
    const uint64_t packed = static_cast<uint64_t>(low) | (static_cast<uint64_t>(static_cast<uint32_t>(high)) << 32);
    return static_cast<int64_t>(packed);
}

__aicore__ inline void StoreWindowHeader(GM_ADDR headerGM, int32_t rankSize, int64_t maxRoutesPerSrc,
    int64_t rowBytes, int64_t slotBytes, int64_t totalBytes, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int64_t> local =
        tBuf.GetWithOffset<int64_t>(kEpScalarUbBytes / sizeof(int64_t), kEpScalarUbOffset);
    local.SetValue(0, PackInt32Pair(TileXREp::kEpWindowMagic, rankSize));
    local.SetValue(1, maxRoutesPerSrc);
    local.SetValue(2, rowBytes);
    local.SetValue(3, slotBytes);
    local.SetValue(4, totalBytes);
    local.SetValue(5, 0);
    local.SetValue(6, 0);
    local.SetValue(7, 0);
    StoreInt64WordsToGm(headerGM, local, tBuf, TileXREp::kEpWindowHeaderBytes);
}

__aicore__ inline void ClearLocalWindow(
    GM_ADDR localWindow, int32_t rankSize, int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t slotBytes,
    int64_t totalBytes, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    StoreWindowHeader(localWindow, rankSize, maxRoutesPerSrc, rowBytes, slotBytes, totalBytes, tBuf);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline TileXREp::EpAssistTuple LoadAssistTupleFromGm(
    GM_ADDR srcGM, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int32_t> local =
        tBuf.GetWithOffset<int32_t>(TileXREp::kEpAssistTupleInts, kEpScalarUbOffset);
    AscendC::GlobalTensor<int32_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(srcGM), TileXREp::kEpAssistTupleInts);

    AscendC::DataCopyExtParams copyParams {
        1, static_cast<uint32_t>(sizeof(TileXREp::EpAssistTuple)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    TileXREp::EpAssistTuple tuple;
    tuple.srcRank = local.GetValue(0);
    tuple.tokenId = local.GetValue(1);
    tuple.topKId = local.GetValue(2);
    tuple.expertId = local.GetValue(3);
    return tuple;
}

__aicore__ inline void StoreSlotHeader(GM_ADDR slotGM, int32_t count, int32_t srcRank, int64_t payloadBytes,
    int64_t assistBytes, int64_t magic, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int64_t> local =
        tBuf.GetWithOffset<int64_t>(kEpScalarUbBytes / sizeof(int64_t), kEpScalarUbOffset);
    local.SetValue(0, PackInt32Pair(static_cast<uint32_t>(count), srcRank));
    local.SetValue(1, payloadBytes);
    local.SetValue(2, assistBytes);
    local.SetValue(3, magic);
    local.SetValue(4, 0);
    local.SetValue(5, 0);
    local.SetValue(6, 0);
    local.SetValue(7, 0);
    StoreInt64WordsToGm(slotGM, local, tBuf, TileXREp::kEpSrcSlotHeaderBytes);
}

__aicore__ inline void WriteAssist(
    GM_ADDR assistGM, int64_t index, int32_t srcRank, int32_t tokenId, int32_t topKId, int32_t expertId,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int32_t> local =
        tBuf.GetWithOffset<int32_t>(kEpScalarUbBytes / sizeof(int32_t), kEpScalarUbOffset);
    local.SetValue(0, srcRank);
    local.SetValue(1, tokenId);
    local.SetValue(2, topKId);
    local.SetValue(3, expertId);

    AscendC::GlobalTensor<int32_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(assistGM) + index * TileXREp::kEpAssistTupleInts,
        TileXREp::kEpAssistTupleInts);
    AscendC::DataCopyExtParams copyParams {
        1, static_cast<uint32_t>(sizeof(TileXREp::EpAssistTuple)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(dst, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline bool IsValidShape(int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes,
    int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot,
    int64_t slotBytes, int64_t totalBytes, int32_t rankSize)
{
    if (bs <= 0 || h <= 0 || topK <= 0 || moeExpertNum <= 0 || dtypeBytes <= 0 || maxRoutesPerSrc <= 0 ||
        rowBytes <= 0 || payloadBytesPerSlot <= 0 || assistBytesPerSlot <= 0 || slotBytes <= 0 || totalBytes <= 0 ||
        moeExpertNum % rankSize != 0) {
        return false;
    }
    const int64_t expectedRoutes = bs * topK;
    const int64_t expectedRowBytes = h * dtypeBytes;
    const int64_t expectedPayload = AlignUp(expectedRoutes * rowBytes, TileXREp::kEpWindowAlignmentBytes);
    const int64_t expectedAssist = AlignUp(expectedRoutes * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)),
        TileXREp::kEpWindowAlignmentBytes);
    const int64_t expectedSlot =
        AlignUp(TileXREp::kEpSrcSlotHeaderBytes + expectedPayload + expectedAssist, TileXREp::kEpWindowAlignmentBytes);
    const int64_t expectedTotal = TileXREp::kEpWindowHeaderBytes + static_cast<int64_t>(rankSize) * expectedSlot;
    return maxRoutesPerSrc == expectedRoutes && rowBytes == expectedRowBytes &&
        payloadBytesPerSlot == expectedPayload && assistBytesPerSlot == expectedAssist && slotBytes == expectedSlot &&
        totalBytes == expectedTotal && totalBytes <= TileXR::IPC_BUFF_MAX_SIZE;
}

__aicore__ inline uint64_t TileXREpReadyValue(int64_t magic, int32_t step)
{
    return (static_cast<uint64_t>(magic) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(step));
}

__aicore__ inline bool TileXREpSameNode(int32_t lhsRank, int32_t rhsRank, int32_t localRankSize)
{
    return localRankSize > 0 && lhsRank / localRankSize == rhsRank / localRankSize;
}

__aicore__ inline int32_t TileXREpNodeStart(int32_t rank, int32_t localRankSize)
{
    return localRankSize > 0 ? (rank / localRankSize) * localRankSize : rank;
}

__aicore__ inline int32_t TileXREpNodeEnd(int32_t rank, int32_t localRankSize, int32_t rankSize)
{
    int32_t end = TileXREpNodeStart(rank, localRankSize) + localRankSize;
    return end > rankSize ? rankSize : end;
}

__aicore__ inline void TileXREpStoreLocalReadyValue(GM_ADDR localWindow, int64_t totalBytes, int32_t rank,
    uint64_t ready)
{
    GM_ADDR readyAddr = localWindow + UDMAReadyOffset(totalBytes, rank);
    auto readyGM = reinterpret_cast<__gm__ uint64_t *>(readyAddr);
    readyGM[0] = ready;
    AscendC::PipeBarrier<PIPE_ALL>();
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(readyAddr), sizeof(uint64_t));
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TileXREpStoreRelayReadyValue(GM_ADDR relayBase, int64_t totalBytes, int32_t rankSize,
    int64_t slotBytes, int32_t srcRank, uint64_t ready)
{
    GM_ADDR readyAddr = relayBase + UDMARelayReadyOffset(totalBytes, rankSize, slotBytes, srcRank);
    auto readyGM = reinterpret_cast<__gm__ uint64_t *>(readyAddr);
    readyGM[0] = ready;
    AscendC::PipeBarrier<PIPE_ALL>();
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(readyAddr), sizeof(uint64_t));
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TileXREpWaitRelayReadyValue(GM_ADDR relayBase, int64_t totalBytes, int32_t rankSize,
    int64_t slotBytes, int32_t srcRank, uint64_t ready, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    GM_ADDR readyAddr = relayBase + UDMARelayReadyOffset(totalBytes, rankSize, slotBytes, srcRank);
    int64_t retries = 0;
    while (true) {
        TileXREpInvalidateLocalCacheLines(readyAddr, static_cast<int64_t>(sizeof(uint64_t)));
        if (LoadUint64FromGm(readyAddr, tBuf) == ready) {
            break;
        }
        ++retries;
        if (retries >= static_cast<int64_t>(TileXR::TILEXR_UDMA_MAX_RETRY_TIMES)) {
            AscendC::printf("tilexr_ep_relay_ready timeout src %d\n", srcRank);
            break;
        }
    }
}

__aicore__ inline void TileXREpNotifyRemoteUdmaReadySeparate(
    const __gm__ TileXR::CommArgs *args, GM_ADDR localWindow, int32_t rank, int32_t rankSize, int32_t localRankSize,
    int64_t totalBytes, int64_t magic, int64_t slotBytes, int32_t step, int64_t remoteRecvWindowOffset)
{
    const uint64_t ready = TileXREpReadyValue(magic, step);
    const int64_t remoteWindowBaseOffset = remoteRecvWindowOffset - UDMARecvWindowOffset(totalBytes);
    const int64_t localReadyOffset = UDMAReadyOffset(totalBytes, rank);
    const uint64_t remoteReadyOffset = static_cast<uint64_t>(remoteWindowBaseOffset + localReadyOffset);
    TileXREpStoreLocalReadyValue(localWindow, totalBytes, rank, ready);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank || TileXREpSameNode(peer, rank, localRankSize)) {
            continue;
        }
        TileXR::UDMAPutNbi<uint8_t>(args, peer,
            reinterpret_cast<__gm__ uint8_t *>(localWindow + SlotOffset(peer, slotBytes)),
            static_cast<uint64_t>(remoteRecvWindowOffset + SlotOffset(rank, slotBytes)),
            static_cast<uint32_t>(slotBytes));
        TileXR::UDMAQuiet(args, peer);
        TileXR::UDMAPutNbi<uint8_t>(args, peer, reinterpret_cast<__gm__ uint8_t *>(localWindow + localReadyOffset),
            remoteReadyOffset, static_cast<uint32_t>(sizeof(uint64_t)));
        TileXR::UDMAQuiet(args, peer);
    }
}

__aicore__ inline void TileXREpWaitRemoteUdmaReady(GM_ADDR localWindow, int32_t rank, int32_t rankSize,
    int32_t localRankSize, int64_t totalBytes, int64_t magic, int32_t step,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    const uint64_t ready = TileXREpReadyValue(magic, step);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank || TileXREpSameNode(peer, rank, localRankSize)) {
            continue;
        }
        GM_ADDR readyAddr = localWindow + UDMAReadyOffset(totalBytes, peer);
        int64_t retries = 0;
        while (true) {
            TileXREpInvalidateLocalCacheLines(readyAddr, static_cast<int64_t>(sizeof(uint64_t)));
            if (LoadUint64FromGm(readyAddr, tBuf) == ready) {
                break;
            }
            ++retries;
            if (retries >= static_cast<int64_t>(TileXR::TILEXR_UDMA_MAX_RETRY_TIMES)) {
                AscendC::printf("tilexr_ep_remote_ready timeout rank %d peer %d step %d\n", rank, peer, step);
                break;
            }
        }
    }
}

__aicore__ inline void TileXREpFlushDispatchSlotHeaders(
    GM_ADDR localWindow, int32_t rankSize, int64_t slotBytes, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        (void)LoadInt32FromGm(localWindow + SlotOffset(dstRank, slotBytes), tBuf);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TileXREpFlushUdmaSourceWindow(GM_ADDR localWindow, int64_t totalBytes)
{
    if (localWindow == nullptr || totalBytes <= 0) {
        return;
    }
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(localWindow),
        static_cast<uint64_t>(totalBytes));
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace
