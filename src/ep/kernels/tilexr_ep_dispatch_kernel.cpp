#include "comm_args.h"
#include "ep_window.h"
#include "kernel_operator.h"
#include "tilexr_sync.h"

namespace {

constexpr uint32_t kEpUbBytes = 64*1024;
constexpr uint32_t kEpSyncUbBytes = 4*1024;
constexpr uint32_t kEpCopyTileBytes = kEpUbBytes - kEpSyncUbBytes;
constexpr uint32_t kEpScalarUbBytes = 32;
constexpr uint32_t kEpScalarUbOffset = kEpSyncUbBytes - kEpScalarUbBytes;

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

__aicore__ inline void ClearLocalWindow(
    GM_ADDR localWindow, int32_t rankSize, int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t slotBytes,
    int64_t totalBytes)
{
    auto header = reinterpret_cast<__gm__ TileXREp::EpWindowHeader *>(localWindow);
    header->magic = TileXREp::kEpWindowMagic;
    header->rankSize = rankSize;
    header->maxRoutesPerSrc = maxRoutesPerSrc;
    header->rowBytes = rowBytes;
    header->slotBytes = slotBytes;
    header->totalBytes = totalBytes;
    header->reserved0 = 0;
    header->reserved1 = 0;
    header->reserved2 = 0;

    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        auto slot = reinterpret_cast<__gm__ TileXREp::EpSrcSlotHeader *>(localWindow + SlotOffset(srcRank, slotBytes));
        slot->count = 0;
        slot->srcRank = srcRank;
        slot->payloadBytes = 0;
        slot->assistBytes = 0;
        slot->reserved0 = 0;
        slot->reserved1 = 0;
        slot->reserved2 = 0;
        slot->reserved3 = 0;
        slot->reserved4 = 0;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void WriteAssist(
    GM_ADDR assistGM, int64_t index, int32_t srcRank, int32_t tokenId, int32_t topKId, int32_t expertId)
{
    auto assist = reinterpret_cast<__gm__ TileXREp::EpAssistTuple *>(assistGM);
    assist[index].srcRank = srcRank;
    assist[index].tokenId = tokenId;
    assist[index].topKId = topKId;
    assist[index].expertId = expertId;
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
    const int64_t expectedPayload = AlignUp(expectedRoutes * expectedRowBytes, TileXREp::kEpWindowAlignmentBytes);
    const int64_t expectedAssist = AlignUp(expectedRoutes * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)),
        TileXREp::kEpWindowAlignmentBytes);
    const int64_t expectedSlot =
        AlignUp(TileXREp::kEpSrcSlotHeaderBytes + expectedPayload + expectedAssist, TileXREp::kEpWindowAlignmentBytes);
    const int64_t expectedTotal = TileXREp::kEpWindowHeaderBytes + static_cast<int64_t>(rankSize) * expectedSlot;
    return maxRoutesPerSrc == expectedRoutes && rowBytes == expectedRowBytes &&
        payloadBytesPerSlot == expectedPayload && assistBytesPerSlot == expectedAssist && slotBytes == expectedSlot &&
        totalBytes == expectedTotal && totalBytes <= TileXR::IPC_BUFF_MAX_SIZE;
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_ep_dispatch_kernel(GM_ADDR commArgsGM, GM_ADDR xGM, GM_ADDR expertIdsGM,
    GM_ADDR expandXOutGM, GM_ADDR expertTokenNumsOutGM, GM_ADDR epRecvCountsOutGM, GM_ADDR assistInfoForCombineOutGM,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes, int64_t maxRoutesPerSrc,
    int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes,
    int64_t magic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        if (commArgsGM == nullptr || xGM == nullptr || expertIdsGM == nullptr || expandXOutGM == nullptr ||
            expertTokenNumsOutGM == nullptr || epRecvCountsOutGM == nullptr || assistInfoForCombineOutGM == nullptr) {
            return;
        }

        auto args = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
        const int32_t rank = args->rank;
        const int32_t rankSize = args->rankSize;
        if (rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE || rank < 0 || rank >= rankSize ||
            !IsValidShape(bs, h, topK, moeExpertNum, dtypeBytes, maxRoutesPerSrc, rowBytes, payloadBytesPerSlot,
                assistBytesPerSlot, slotBytes, totalBytes, rankSize)) {
            return;
        }

        GM_ADDR shareAddrs[TileXR::TILEXR_MAX_RANK_SIZE];
        AscendC::GlobalTensor<GM_ADDR> peerMems;
        peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            shareAddrs[peer] = peerMems.GetValue(peer);
            if (shareAddrs[peer] == nullptr) {
                return;
            }
        }

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, kEpUbBytes);
        SyncCollectives sync;
        sync.Init(rank, rankSize, shareAddrs, tBuf);

        GM_ADDR localWindow = shareAddrs[rank] + TileXR::IPC_DATA_OFFSET;
        ClearLocalWindow(localWindow, rankSize, maxRoutesPerSrc, rowBytes, slotBytes, totalBytes);
        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepWindowCleared);
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepWindowCleared, peer);
        }

        int64_t dstCounts[TileXR::TILEXR_MAX_RANK_SIZE];
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            dstCounts[peer] = 0;
        }

        const int64_t localExpertNum = moeExpertNum / rankSize;
        for (int64_t token = 0; token < bs; ++token) {
            for (int64_t topKId = 0; topKId < topK; ++topKId) {
                const int64_t route = token * topK + topKId;
                const int32_t expertId = LoadInt32FromGm(expertIdsGM + route * static_cast<int64_t>(sizeof(int32_t)),
                    tBuf);
                if (expertId < 0 || static_cast<int64_t>(expertId) >= moeExpertNum) {
                    continue;
                }
                const int64_t dstRank = static_cast<int64_t>(expertId) / localExpertNum;
                if (dstRank < 0 || dstRank >= rankSize || dstCounts[dstRank] >= maxRoutesPerSrc) {
                    continue;
                }

                const int64_t dstIndex = dstCounts[dstRank];
                GM_ADDR dstWindow = shareAddrs[dstRank] + TileXR::IPC_DATA_OFFSET;
                CopyBytesGmToGm(dstWindow + PayloadOffset(rank, slotBytes) + dstIndex * rowBytes,
                    xGM + token * rowBytes, tBuf, rowBytes);
                WriteAssist(dstWindow + AssistOffset(rank, slotBytes, payloadBytesPerSlot), dstIndex, rank,
                    static_cast<int32_t>(token), static_cast<int32_t>(topKId), expertId);
                ++dstCounts[dstRank];
            }
        }

        for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
            GM_ADDR dstWindow = shareAddrs[dstRank] + TileXR::IPC_DATA_OFFSET;
            auto slot =
                reinterpret_cast<__gm__ TileXREp::EpSrcSlotHeader *>(dstWindow + SlotOffset(rank, slotBytes));
            slot->count = static_cast<int32_t>(dstCounts[dstRank]);
            slot->srcRank = rank;
            slot->payloadBytes = dstCounts[dstRank] * rowBytes;
            slot->assistBytes = dstCounts[dstRank] * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple));
            slot->reserved0 = 0;
            slot->reserved1 = 0;
            slot->reserved2 = 0;
            slot->reserved3 = 0;
            slot->reserved4 = 0;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepDispatchReady);
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepDispatchReady, peer);
        }

        auto expertTokenNumsOut = reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOutGM);
        auto epRecvCountsOut = reinterpret_cast<__gm__ int32_t *>(epRecvCountsOutGM);
        const int64_t localExpertNum64 = localExpertNum;
        for (int64_t localExpert = 0; localExpert < localExpertNum64; ++localExpert) {
            expertTokenNumsOut[localExpert] = 0;
        }

        int64_t outRecord = 0;
        auto localAssistBase = reinterpret_cast<__gm__ TileXREp::EpAssistTuple *>(assistInfoForCombineOutGM);
        for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
            auto slot =
                reinterpret_cast<__gm__ TileXREp::EpSrcSlotHeader *>(localWindow + SlotOffset(srcRank, slotBytes));
            const int64_t count = slot->count;
            epRecvCountsOut[srcRank] = static_cast<int32_t>(count);
            if (count <= 0 || count > maxRoutesPerSrc) {
                continue;
            }

            GM_ADDR payloadBase = localWindow + PayloadOffset(srcRank, slotBytes);
            GM_ADDR assistBase = localWindow + AssistOffset(srcRank, slotBytes, payloadBytesPerSlot);
            for (int64_t item = 0; item < count; ++item) {
                CopyBytesGmToGm(expandXOutGM + outRecord * rowBytes, payloadBase + item * rowBytes, tBuf, rowBytes);
                const TileXREp::EpAssistTuple tuple = LoadAssistTupleFromGm(
                    assistBase + item * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), tBuf);
                localAssistBase[outRecord].srcRank = tuple.srcRank;
                localAssistBase[outRecord].tokenId = tuple.tokenId;
                localAssistBase[outRecord].topKId = tuple.topKId;
                localAssistBase[outRecord].expertId = tuple.expertId;
                const int64_t localExpert = static_cast<int64_t>(tuple.expertId) % localExpertNum64;
                if (localExpert >= 0 && localExpert < localExpertNum64) {
                    expertTokenNumsOut[localExpert] = expertTokenNumsOut[localExpert] + 1;
                }
                ++outRecord;
            }
        }
    }
}

void launch_tilexr_ep_dispatch_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR expandXOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR assistInfoForCombineOut,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum, int64_t dtypeBytes, int64_t maxRoutesPerSrc,
    int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes,
    int64_t magic)
{
    tilexr_ep_dispatch_kernel<<<blockDim, nullptr, stream>>>(commArgs, x, expertIds, expandXOut, expertTokenNumsOut,
        epRecvCountsOut, assistInfoForCombineOut, bs, h, topK, moeExpertNum, dtypeBytes, maxRoutesPerSrc, rowBytes,
        payloadBytesPerSlot, assistBytesPerSlot, slotBytes, totalBytes, magic);
}
