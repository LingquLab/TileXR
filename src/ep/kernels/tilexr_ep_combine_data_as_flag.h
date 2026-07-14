#ifndef TILEXR_EP_COMBINE_DATA_AS_FLAG_H
#define TILEXR_EP_COMBINE_DATA_AS_FLAG_H

#include "tilexr_data_as_flag.h"

namespace {

constexpr int32_t kCombineDafRecvNotReady = 0;
constexpr int32_t kCombineDafRecvDone = 1;
constexpr int32_t kCombineDafRecvError = 2;
constexpr int64_t kCombineDafPayloadCacheSafeBytes =
    static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES) -
    static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES % TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
static_assert(TileXR::DATA_AS_FLAG_ALIGN_BYTES % sizeof(uint64_t) == 0U,
    "Combine DataAsFlag payload copy chunks must stay 8-byte aligned");
static_assert(kCombineDafPayloadCacheSafeBytes % static_cast<int64_t>(sizeof(uint64_t)) == 0,
    "Combine DataAsFlag cache-safe prefix must end on an 8-byte boundary");

__aicore__ inline GM_ADDR TileXREpDataAsFlagPayloadAddr(GM_ADDR dafSlotGM, int64_t ordinaryOffset)
{
    const int64_t block = ordinaryOffset / static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES);
    const int64_t inBlock = ordinaryOffset % static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES);
    return dafSlotGM + block * static_cast<int64_t>(TileXR::DATA_AS_FLAG_BLOCK_BYTES) + inBlock;
}

__aicore__ inline void TileXREpCleanDataAsFlagRange(GM_ADDR addr, int64_t bytes)
{
    if (addr == nullptr || bytes <= 0) {
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(addr), static_cast<uint64_t>(bytes));
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool TileXREpDataAsFlagPayloadSharesFlagLine(int64_t ordinaryOffset)
{
    const int64_t inBlock = ordinaryOffset % static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES);
    return inBlock >= kCombineDafPayloadCacheSafeBytes;
}

__aicore__ inline void TileXREpCopyDataAsFlagPayloadTailBypass(
    GM_ADDR dstGM, GM_ADDR srcGM, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf, int64_t bytes)
{
    (void)tBuf;
    if (dstGM == nullptr || srcGM == nullptr || bytes <= 0) {
        return;
    }
    // The DAF payload tail shares the ready-flag cache line, so it uses uint64_t ld_dev/st_dev
    // instead of DataCopy after cache maintenance. The static_asserts above keep this
    // tail bypass from silently dropping sub-word tails if the DataAsFlag alignment changes.
    __gm__ uint64_t *src = reinterpret_cast<__gm__ uint64_t *>(srcGM);
    __gm__ uint64_t *dst = reinterpret_cast<__gm__ uint64_t *>(dstGM);
    const int64_t words = bytes / static_cast<int64_t>(sizeof(uint64_t));
    for (int64_t idx = 0; idx < words; ++idx) {
        const uint64_t value = ld_dev(src + idx, 0);
        st_dev(value, dst + idx, 0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TileXREpCopyBytesGmToGmExt(
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

        AscendC::DataCopyExtParams copyParams {1U, static_cast<uint32_t>(tileBytes), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0, 0, 0};
        AscendC::DataCopyPad(local, src[copied], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[copied], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TileXREpCopyGmToDataAsFlagPayload(GM_ADDR dafSlotGM, int64_t ordinaryOffset,
    GM_ADDR srcGM, int64_t bytes, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    int64_t copied = 0;
    while (copied < bytes) {
        const int64_t logical = ordinaryOffset + copied;
        const int64_t inBlock = logical % static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES);
        int64_t chunk = static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES) - inBlock;
        if (chunk > bytes - copied) {
            chunk = bytes - copied;
        }
        GM_ADDR dstAddr = TileXREpDataAsFlagPayloadAddr(dafSlotGM, logical);
        CopyBytesGmToGm(dstAddr, srcGM + copied, tBuf, chunk);
        TileXREpCleanDataAsFlagRange(dstAddr, chunk);
        copied += chunk;
    }
}

__aicore__ inline void TileXREpStoreBytesToDataAsFlagPayload(GM_ADDR dafSlotGM, int64_t ordinaryOffset,
    AscendC::LocalTensor<uint8_t> &local, int64_t bytes,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    (void)tBuf;
    int64_t copied = 0;
    while (copied < bytes) {
        const int64_t logical = ordinaryOffset + copied;
        const int64_t inBlock = logical % static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES);
        int64_t chunk = static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES) - inBlock;
        if (chunk > bytes - copied) {
            chunk = bytes - copied;
        }
        GM_ADDR dstAddr = TileXREpDataAsFlagPayloadAddr(dafSlotGM, logical);
        AscendC::GlobalTensor<uint8_t> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dstAddr), chunk);
        AscendC::DataCopyExtParams copyParams {1U, static_cast<uint32_t>(chunk), 0U, 0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst, local[copied], copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        TileXREpCleanDataAsFlagRange(dstAddr, chunk);
        copied += chunk;
    }
}

__aicore__ inline void TileXREpStoreDataAsFlagSlotHeader(GM_ADDR dafSlotGM, int32_t count, int32_t srcRank,
    int64_t payloadBytes, int64_t assistBytes, int64_t magic,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int64_t> words =
        tBuf.GetWithOffset<int64_t>(kEpScalarUbBytes / sizeof(int64_t), kEpScalarUbOffset);
    words.SetValue(0, PackInt32Pair(static_cast<uint32_t>(count), srcRank));
    words.SetValue(1, payloadBytes);
    words.SetValue(2, assistBytes);
    words.SetValue(3, magic);
    words.SetValue(4, 0);
    words.SetValue(5, 0);
    words.SetValue(6, 0);
    words.SetValue(7, 0);
    AscendC::LocalTensor<uint8_t> bytes = words.template ReinterpretCast<uint8_t>();
    TileXREpStoreBytesToDataAsFlagPayload(dafSlotGM, 0, bytes, TileXREp::kEpSrcSlotHeaderBytes, tBuf);
}

__aicore__ inline void TileXREpWriteDataAsFlagAssist(GM_ADDR dafSlotGM, int64_t assistOffset, int64_t index,
    int32_t srcRank, int32_t tokenId, int32_t topKId, int32_t expertId,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int32_t> tuple =
        tBuf.GetWithOffset<int32_t>(kEpScalarUbBytes / sizeof(int32_t), kEpScalarUbOffset);
    tuple.SetValue(0, srcRank);
    tuple.SetValue(1, tokenId);
    tuple.SetValue(2, topKId);
    tuple.SetValue(3, expertId);
    AscendC::LocalTensor<uint8_t> bytes = tuple.template ReinterpretCast<uint8_t>();
    TileXREpStoreBytesToDataAsFlagPayload(dafSlotGM,
        assistOffset + index * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), bytes,
        static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), tBuf);
}

__aicore__ inline void TileXREpStoreDataAsFlagTag(GM_ADDR addr, uint64_t tag,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int64_t> local =
        tBuf.GetWithOffset<int64_t>(kEpScalarUbBytes / sizeof(int64_t), kEpScalarUbOffset);
    const int64_t signedTag = static_cast<int64_t>(tag);
    local.SetValue(0, signedTag);
    local.SetValue(1, signedTag);
    local.SetValue(2, signedTag);
    local.SetValue(3, signedTag);
    StoreInt64WordsToGm(addr, local, tBuf, static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_BYTES));
}

__aicore__ inline bool TileXREpDataAsFlagBlockMatchesExpected(GM_ADDR flagAddr, uint64_t expectedReadyTag,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    (void)tBuf;
    const int64_t words = static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_BYTES) /
        static_cast<int64_t>(sizeof(uint64_t));
    __gm__ uint64_t *flagWords = reinterpret_cast<__gm__ uint64_t *>(flagAddr);
    for (int64_t idx = 0; idx < words; ++idx) {
        if (ld_dev(flagWords + idx, 0) != expectedReadyTag) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void TileXREpInvalidateDataAsFlagPayload(GM_ADDR dafSlotGM, int64_t slotBytes)
{
    const int64_t payloadBytes = static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES);
    const int64_t encodedStepBytes = payloadBytes + static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_BYTES);
    for (int64_t logical = 0; logical < slotBytes; logical += payloadBytes) {
        int64_t invalidateBytes = slotBytes - logical;
        if (invalidateBytes > kCombineDafPayloadCacheSafeBytes) {
            invalidateBytes = kCombineDafPayloadCacheSafeBytes;
        }
        if (invalidateBytes <= 0) {
            continue;
        }
        const int64_t dafOffset = logical / payloadBytes * encodedStepBytes;
        TileXREpInvalidateLocalCacheLines(dafSlotGM + dafOffset, invalidateBytes);
    }
}

__aicore__ inline void TileXREpCopyDataAsFlagPayloadToGm(GM_ADDR dafSlotGM, int64_t slotBytes, GM_ADDR recvSlotGM,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    TileXREpInvalidateDataAsFlagPayload(dafSlotGM, slotBytes);
    int64_t copied = 0;
    while (copied < slotBytes) {
        const int64_t inBlock = copied % static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES);
        int64_t bytes = static_cast<int64_t>(TileXR::DATA_AS_FLAG_ALIGN_BYTES);
        const int64_t blockRemaining = static_cast<int64_t>(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES) - inBlock;
        if (bytes > blockRemaining) {
            bytes = blockRemaining;
        }
        if (bytes > slotBytes - copied) {
            bytes = slotBytes - copied;
        }
        GM_ADDR srcAddr = TileXREpDataAsFlagPayloadAddr(dafSlotGM, copied);
        if (TileXREpDataAsFlagPayloadSharesFlagLine(copied)) {
            TileXREpCopyDataAsFlagPayloadTailBypass(recvSlotGM + copied, srcAddr, tBuf, bytes);
        } else {
            TileXREpCopyBytesGmToGmExt(recvSlotGM + copied, srcAddr, tBuf, bytes);
        }
        copied += bytes;
    }
}

__aicore__ inline bool TileXREpDataAsFlagFlagsMatchExpected(GM_ADDR dafSlotGM, int64_t slotBytes,
    uint64_t expectedReadyTag, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    // Do not use TileXREpInvalidateLocalCacheLines here: it clean+invalidates and can write stale
    // local flags over UDMA-updated ready tags. ld_dev is the receive-side scalar poll path.
    const int64_t blocks = CombineDataAsFlagSlotBytes(slotBytes) /
        static_cast<int64_t>(TileXR::DATA_AS_FLAG_BLOCK_BYTES);
    for (int64_t block = 0; block < blocks; ++block) {
        GM_ADDR flag = dafSlotGM + block * static_cast<int64_t>(TileXR::DATA_AS_FLAG_BLOCK_BYTES) +
            static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES);
        if (!TileXREpDataAsFlagBlockMatchesExpected(flag, expectedReadyTag, tBuf)) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool TileXREpCombineDataAsFlagCheckExpected(GM_ADDR dafSlotGM, int64_t slotBytes,
    uint64_t expectedReadyTag, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    return TileXREpDataAsFlagFlagsMatchExpected(dafSlotGM, slotBytes, expectedReadyTag, tBuf);
}

__aicore__ inline bool TileXREpRevalidateDataAsFlagAfterRecv(GM_ADDR dafSlotGM, int64_t slotBytes,
    uint64_t expectedReadyTag, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    return TileXREpDataAsFlagFlagsMatchExpected(dafSlotGM, slotBytes, expectedReadyTag, tBuf);
}

__aicore__ inline void TileXREpClearDataAsFlagFlags(GM_ADDR dafSlotGM, int64_t slotBytes,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    const int64_t blocks = CombineDataAsFlagSlotBytes(slotBytes) /
        static_cast<int64_t>(TileXR::DATA_AS_FLAG_BLOCK_BYTES);
    for (int64_t block = 0; block < blocks; ++block) {
        GM_ADDR flag = dafSlotGM + block * static_cast<int64_t>(TileXR::DATA_AS_FLAG_BLOCK_BYTES) +
            static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES);
        TileXREpStoreDataAsFlagTag(flag, 0U, tBuf);
        TileXREpCleanDataAsFlagRange(flag, static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_BYTES));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TileXREpFinalizeDataAsFlagFlags(GM_ADDR dafSlotGM, int64_t slotBytes,
    uint64_t expectedReadyTag, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    const int64_t blocks = CombineDataAsFlagSlotBytes(slotBytes) /
        static_cast<int64_t>(TileXR::DATA_AS_FLAG_BLOCK_BYTES);
    for (int64_t block = 0; block < blocks; ++block) {
        GM_ADDR flag = dafSlotGM + block * static_cast<int64_t>(TileXR::DATA_AS_FLAG_BLOCK_BYTES) +
            static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES);
        TileXREpStoreDataAsFlagTag(flag, expectedReadyTag, tBuf);
        TileXREpCleanDataAsFlagRange(flag, static_cast<int64_t>(TileXR::DATA_AS_FLAG_FLAG_BYTES));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline int32_t TileXREpCombineDataAsFlagRecvIfReady(GM_ADDR dafSlotGM, int64_t slotBytes,
    uint64_t expectedReadyTag, GM_ADDR recvSlotGM, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    if (!TileXREpCombineDataAsFlagCheckExpected(dafSlotGM, slotBytes, expectedReadyTag, tBuf)) {
        return kCombineDafRecvNotReady;
    }
    TileXREpCopyDataAsFlagPayloadToGm(dafSlotGM, slotBytes, recvSlotGM, tBuf);
    if (!TileXREpRevalidateDataAsFlagAfterRecv(dafSlotGM, slotBytes, expectedReadyTag, tBuf)) {
        return kCombineDafRecvError;
    }
    TileXREpClearDataAsFlagFlags(dafSlotGM, slotBytes, tBuf);
    return kCombineDafRecvDone;
}

} // namespace

#endif // TILEXR_EP_COMBINE_DATA_AS_FLAG_H
