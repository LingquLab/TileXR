#include "tilexr_ep_kernel_common.h"
#include "tilexr_ep_combine_data_as_flag.h"
#include "tilexr_ep_combine_udma_wqe_simt.h"

namespace {

template <typename T>
__aicore__ inline void ZeroRows(GM_ADDR yOutGM, int64_t bs, int64_t h)
{
    auto yOut = reinterpret_cast<__gm__ T *>(yOutGM);
    const int64_t elements = bs * h;
    for (int64_t index = 0; index < elements; ++index) {
        yOut[index] = static_cast<T>(0.0f);
    }
}

__aicore__ inline float TileXREpCombineHalfBitsToFloat(uint16_t bits)
{
    union {
        uint16_t u;
        float16_t h;
    } value;
    value.u = bits;
    return static_cast<float>(value.h);
}

__aicore__ inline void AccumulateRowFp16(
    GM_ADDR yOutGM, GM_ADDR srcGM, int64_t token, int64_t h, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    (void)tBuf;
    auto yOut = reinterpret_cast<__gm__ float16_t *>(yOutGM);
    auto src = reinterpret_cast<__gm__ uint16_t *>(srcGM);
    const int64_t yOffset = token * h;
    for (int64_t elem = 0; elem < h; ++elem) {
        const float16_t yValue = yOut[yOffset + elem];
        const float srcValue = TileXREpCombineHalfBitsToFloat(ld_dev(src + elem, 0));
        yOut[yOffset + elem] =
            static_cast<float16_t>(static_cast<float>(yValue) + srcValue);
    }
}

__aicore__ inline void ScatterCombineRows(GM_ADDR localWindow, GM_ADDR expertOutGM, GM_ADDR assistInfoForCombineGM,
    GM_ADDR epRecvCountsGM, int32_t rankSize, int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t slotBytes,
    int64_t payloadBytesPerSlot, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    int64_t inRecord = 0;
    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        int64_t count = LoadInt32FromGm(epRecvCountsGM + srcRank * static_cast<int64_t>(sizeof(int32_t)), tBuf);
        if (count < 0 || count > maxRoutesPerSrc) {
            count = 0;
        }

        GM_ADDR payloadBase = localWindow + PayloadOffset(srcRank, slotBytes);
        GM_ADDR assistBase = localWindow + AssistOffset(srcRank, slotBytes, payloadBytesPerSlot);
        for (int64_t item = 0; item < count; ++item) {
            CopyBytesGmToGm(payloadBase + item * rowBytes, expertOutGM + inRecord * rowBytes, tBuf, rowBytes);
            const TileXREp::EpAssistTuple tuple = LoadAssistTupleFromGm(
                assistInfoForCombineGM + inRecord * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), tBuf);
            WriteAssist(assistBase, item, srcRank, tuple.tokenId, tuple.topKId, tuple.expertId, tBuf);
            ++inRecord;
        }
        StoreSlotHeader(localWindow + SlotOffset(srcRank, slotBytes), static_cast<int32_t>(count), srcRank,
            count * rowBytes, count * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), 0, tBuf);
    }
}

__aicore__ inline void ScatterCombineRowsDataAsFlag(GM_ADDR localIpcWindow, GM_ADDR dafSendWindow,
    GM_ADDR expertOutGM, GM_ADDR assistInfoForCombineGM, GM_ADDR epRecvCountsGM, int32_t rank, int32_t rankSize,
    int32_t localRankSize, int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t slotBytes,
    int64_t payloadBytesPerSlot, int64_t magic, uint64_t expectedReadyTag,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf, GM_ADDR *shareAddrs)
{
    (void)localIpcWindow;
    int64_t inRecord = 0;
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        int64_t count = LoadInt32FromGm(epRecvCountsGM + dstRank * static_cast<int64_t>(sizeof(int32_t)), tBuf);
        if (count < 0 || count > maxRoutesPerSrc) {
            count = 0;
        }

        const bool sameNode = TileXREpSameNode(dstRank, rank, localRankSize);
        GM_ADDR ordinarySlot = sameNode ? shareAddrs[dstRank] + TileXR::IPC_DATA_OFFSET + SlotOffset(rank, slotBytes)
                                        : nullptr;
        GM_ADDR dafSlot = sameNode ? nullptr : dafSendWindow + CombineDataAsFlagSlotOffset(dstRank, slotBytes);
        if (!sameNode) {
            TileXREpClearDataAsFlagFlags(dafSlot, slotBytes, tBuf);
        }

        const int64_t payloadLogicalOffset = TileXREp::kEpSrcSlotHeaderBytes;
        const int64_t assistLogicalOffset = TileXREp::kEpSrcSlotHeaderBytes + payloadBytesPerSlot;
        for (int64_t item = 0; item < count; ++item) {
            GM_ADDR srcPayload = expertOutGM + inRecord * rowBytes;
            const TileXREp::EpAssistTuple tuple = LoadAssistTupleFromGm(
                assistInfoForCombineGM + inRecord * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), tBuf);
            if (sameNode) {
                CopyBytesGmToGm(ordinarySlot + payloadLogicalOffset + item * rowBytes, srcPayload, tBuf, rowBytes);
                WriteAssist(ordinarySlot + assistLogicalOffset, item, dstRank, tuple.tokenId, tuple.topKId,
                    tuple.expertId, tBuf);
            } else {
                TileXREpCopyGmToDataAsFlagPayload(
                    dafSlot, payloadLogicalOffset + item * rowBytes, srcPayload, rowBytes, tBuf);
                TileXREpWriteDataAsFlagAssist(
                    dafSlot, assistLogicalOffset, item, dstRank, tuple.tokenId, tuple.topKId, tuple.expertId, tBuf);
            }
            ++inRecord;
        }

        if (sameNode) {
            StoreSlotHeader(ordinarySlot, static_cast<int32_t>(count), dstRank, count * rowBytes,
                count * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), magic, tBuf);
        } else {
            TileXREpStoreDataAsFlagSlotHeader(dafSlot, static_cast<int32_t>(count), dstRank, count * rowBytes,
                count * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), magic, tBuf);
            TileXREpFinalizeDataAsFlagFlags(dafSlot, slotBytes, expectedReadyTag, tBuf);
        }
    }
}

__aicore__ inline void TileXREpFlushCrossNodeSameNodeIpcSlots(GM_ADDR *shareAddrs, int32_t rank,
    int32_t rankSize, int32_t localRankSize, int64_t slotBytes,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    const int32_t localNodeStart = TileXREpNodeStart(rank, localRankSize);
    const int32_t localNodeEnd = TileXREpNodeEnd(rank, localRankSize, rankSize);
    for (int32_t dstRank = localNodeStart; dstRank < localNodeEnd; ++dstRank) {
        GM_ADDR targetWindow = shareAddrs[dstRank] + TileXR::IPC_DATA_OFFSET;
        (void)LoadInt32FromGm(targetWindow + SlotOffset(rank, slotBytes), tBuf);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void DrainCombineRows(GM_ADDR *shareAddrs, GM_ADDR yOutGM, int32_t rank, int32_t rankSize,
    int64_t bs, int64_t h, int64_t topK, int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t slotBytes,
    int64_t payloadBytesPerSlot, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    ZeroRows<T>(yOutGM, bs, h);
    for (int32_t expertRank = 0; expertRank < rankSize; ++expertRank) {
        GM_ADDR sourceWindow = shareAddrs[expertRank] + TileXR::IPC_DATA_OFFSET;
        const int64_t count = LoadInt32FromGm(sourceWindow + SlotOffset(rank, slotBytes), tBuf);
        if (count <= 0 || count > maxRoutesPerSrc) {
            continue;
        }

        GM_ADDR payloadBase = sourceWindow + PayloadOffset(rank, slotBytes);
        GM_ADDR assistBase = sourceWindow + AssistOffset(rank, slotBytes, payloadBytesPerSlot);
        for (int64_t item = 0; item < count; ++item) {
            const TileXREp::EpAssistTuple tuple = LoadAssistTupleFromGm(
                assistBase + item * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), tBuf);
            if (tuple.srcRank != rank || tuple.tokenId < 0 || static_cast<int64_t>(tuple.tokenId) >= bs ||
                tuple.topKId < 0 || static_cast<int64_t>(tuple.topKId) >= topK) {
                continue;
            }
            AccumulateRowFp16(yOutGM, payloadBase + item * rowBytes, tuple.tokenId, h, tBuf);
        }
    }
}

__aicore__ inline void TileXREpDrainCrossNodeCombineRows(GM_ADDR localIpcWindow, GM_ADDR recvWindow, GM_ADDR yOutGM,
    int32_t rank, int32_t rankSize, int32_t localRankSize, int64_t bs, int64_t h, int64_t topK,
    int64_t maxRoutesPerSrc, int64_t rowBytes, int64_t slotBytes, int64_t payloadBytesPerSlot,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    ZeroRows<float16_t>(yOutGM, bs, h);
    for (int32_t expertRank = 0; expertRank < rankSize; ++expertRank) {
        const bool sameNode = TileXREpSameNode(expertRank, rank, localRankSize);
        GM_ADDR sourceWindow = sameNode ? localIpcWindow : recvWindow;
        if (sameNode && expertRank != rank) {
            TileXREpInvalidateLocalCacheLines(sourceWindow + SlotOffset(expertRank, slotBytes), slotBytes);
        }
        const int64_t count = LoadInt32FromGm(sourceWindow + SlotOffset(expertRank, slotBytes), tBuf);
        if (count <= 0 || count > maxRoutesPerSrc) {
            continue;
        }
        GM_ADDR payloadBase = sourceWindow + PayloadOffset(expertRank, slotBytes);
        GM_ADDR assistBase = sourceWindow + AssistOffset(expertRank, slotBytes, payloadBytesPerSlot);
        for (int64_t item = 0; item < count; ++item) {
            const TileXREp::EpAssistTuple tuple = LoadAssistTupleFromGm(
                assistBase + item * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), tBuf);
            if (tuple.srcRank != rank || tuple.tokenId < 0 || static_cast<int64_t>(tuple.tokenId) >= bs ||
                tuple.topKId < 0 || static_cast<int64_t>(tuple.topKId) >= topK) {
                continue;
            }
            AccumulateRowFp16(yOutGM, payloadBase + item * rowBytes, tuple.tokenId, h, tBuf);
        }
    }
}

__aicore__ inline void TileXREpAbortCrossNodeCombineLocal(SyncCollectives &sync, GM_ADDR workspaceGM,
    int64_t totalBytes, int32_t rankSize, int64_t slotBytes, int64_t magic)
{
    TileXREpStoreCombineDataAsFlagStatusValue(
        workspaceGM, totalBytes, rankSize, slotBytes, TileXREp::kEpStatusRemoteReadyTimeout);
    sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineRelayReady);
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_ep_combine_kernel(GM_ADDR commArgsGM, GM_ADDR expertOutGM,
    GM_ADDR assistInfoForCombineGM, GM_ADDR epRecvCountsGM, GM_ADDR yOutGM, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, int64_t dtype, int64_t dtypeBytes, int64_t maxRoutesPerSrc, int64_t rowBytes,
    int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes, int64_t magic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        if (commArgsGM == nullptr || expertOutGM == nullptr || assistInfoForCombineGM == nullptr ||
            epRecvCountsGM == nullptr || yOutGM == nullptr) {
            return;
        }

        auto args = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
        const int32_t rank = args->rank;
        const int32_t rankSize = args->rankSize;
        if (rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE || rank < 0 || rank >= rankSize ||
            !IsValidShape(bs, h, topK, moeExpertNum, dtypeBytes, maxRoutesPerSrc, rowBytes, payloadBytesPerSlot,
                assistBytesPerSlot, slotBytes, totalBytes, rankSize) || dtype != kTileXrDataTypeFp16) {
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
        ClearLocalWindow(localWindow, rankSize, maxRoutesPerSrc, rowBytes, slotBytes, totalBytes, tBuf);
        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineWindowCleared);
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineWindowCleared, peer);
        }

        ScatterCombineRows(localWindow, expertOutGM, assistInfoForCombineGM, epRecvCountsGM, rankSize,
            maxRoutesPerSrc, rowBytes, slotBytes, payloadBytesPerSlot, tBuf);
        TileXREpFlushDispatchSlotHeaders(localWindow, rankSize, slotBytes, tBuf);
        AscendC::PipeBarrier<PIPE_ALL>();

        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineReady);
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineReady, peer);
        }

        DrainCombineRows<float16_t>(shareAddrs, yOutGM, rank, rankSize, bs, h, topK, maxRoutesPerSrc, rowBytes,
            slotBytes, payloadBytesPerSlot, tBuf);
    }
}

extern "C" __global__ __aicore__ void tilexr_ep_combine_cross_node_kernel(GM_ADDR commArgsGM, GM_ADDR expertOutGM,
    GM_ADDR assistInfoForCombineGM, GM_ADDR epRecvCountsGM, GM_ADDR yOutGM, GM_ADDR workspaceGM, int64_t bs,
    int64_t h, int64_t topK, int64_t moeExpertNum, int64_t dtype, int64_t dtypeBytes, int64_t maxRoutesPerSrc,
    int64_t rowBytes, int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes,
    int64_t totalBytes, int64_t magic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        if (commArgsGM == nullptr || expertOutGM == nullptr || assistInfoForCombineGM == nullptr ||
            epRecvCountsGM == nullptr || yOutGM == nullptr || workspaceGM == nullptr) {
            return;
        }

        auto args = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
        const int32_t rank = args->rank;
        const int32_t rankSize = args->rankSize;
        const int32_t localRankSize = args->localRankSize;
        if (rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE || rank < 0 || rank >= rankSize ||
            localRankSize <= 0 || localRankSize > rankSize || !TileXR::UDMARegistryEnabled(args) ||
            !IsValidShape(bs, h, topK, moeExpertNum, dtypeBytes, maxRoutesPerSrc, rowBytes, payloadBytesPerSlot,
                assistBytesPerSlot, slotBytes, totalBytes, rankSize) || dtype != kTileXrDataTypeFp16) {
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

        GM_ADDR localIpcWindow = shareAddrs[rank] + TileXR::IPC_DATA_OFFSET;
        const int32_t roundIndex = TileXREpDataAsFlagRoundIndex(magic);
        const uint64_t expectedReadyTag = TileXREpDataAsFlagReadyTag(magic);
        GM_ADDR recvWindow = workspaceGM + CombineDataAsFlagRecvWindowOffset(totalBytes, rankSize, slotBytes);
        GM_ADDR dafSendWindow =
            workspaceGM + CombineDataAsFlagSendRoundOffset(totalBytes, rankSize, slotBytes, roundIndex);
        GM_ADDR dafRecvWindow =
            workspaceGM + CombineDataAsFlagRemoteRecvRoundOffset(totalBytes, rankSize, slotBytes, roundIndex);
        TileXREpStoreCombineDataAsFlagStatusValue(
            workspaceGM, totalBytes, rankSize, slotBytes, TileXREp::kEpStatusOk);
        TileXREpInvalidateLocalCacheLines(localIpcWindow, totalBytes);
        ClearLocalWindow(localIpcWindow, rankSize, maxRoutesPerSrc, rowBytes, slotBytes, totalBytes, tBuf);
        ClearLocalWindow(recvWindow, rankSize, maxRoutesPerSrc, rowBytes, slotBytes, totalBytes, tBuf);

        const int32_t localNodeStart = TileXREpNodeStart(rank, localRankSize);
        const int32_t localNodeEnd = TileXREpNodeEnd(rank, localRankSize, rankSize);
        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineWindowCleared);
        for (int32_t peer = localNodeStart; peer < localNodeEnd; ++peer) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineWindowCleared, peer);
        }

        ScatterCombineRowsDataAsFlag(localIpcWindow, dafSendWindow, expertOutGM, assistInfoForCombineGM,
            epRecvCountsGM, rank, rankSize, localRankSize, maxRoutesPerSrc, rowBytes, slotBytes,
            payloadBytesPerSlot, magic, expectedReadyTag, tBuf, shareAddrs);
        TileXREpFlushCrossNodeSameNodeIpcSlots(shareAddrs, rank, rankSize, localRankSize, slotBytes, tBuf);

        sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineReady);
        for (int32_t peer = localNodeStart; peer < localNodeEnd; ++peer) {
            sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineReady, peer);
        }

        const bool postOk = TileXREpPostCombineDataAsFlagWqes(
            args, dafSendWindow, rank, rankSize, localRankSize, totalBytes, slotBytes, roundIndex, tBuf);
        if (!postOk) {
            TileXREpAbortCrossNodeCombineLocal(sync, workspaceGM, totalBytes, rankSize, slotBytes, magic);
            return;
        }

        int32_t recvDone[TileXR::TILEXR_MAX_RANK_SIZE];
        int32_t remainingPeers = 0;
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            const bool waitPeer = peer != rank && !TileXREpSameNode(peer, rank, localRankSize);
            recvDone[peer] = waitPeer ? 0 : 1;
            if (waitPeer) {
                ++remainingPeers;
            }
        }

        int64_t retries = 0;
        while (remainingPeers > 0 && retries < static_cast<int64_t>(TileXR::TILEXR_UDMA_MAX_RETRY_TIMES)) {
            bool madeProgress = false;
            for (int32_t peer = 0; peer < rankSize; ++peer) {
                if (recvDone[peer] != 0) {
                    continue;
                }
                GM_ADDR srcDafSlot = dafRecvWindow + CombineDataAsFlagSlotOffset(peer, slotBytes);
                GM_ADDR recvSlot = recvWindow + SlotOffset(peer, slotBytes);
                const int32_t recvStatus = TileXREpCombineDataAsFlagRecvIfReady(
                    srcDafSlot, slotBytes, expectedReadyTag, recvSlot, tBuf);
                if (recvStatus == kCombineDafRecvError) {
                    TileXREpAbortCrossNodeCombineLocal(sync, workspaceGM, totalBytes, rankSize, slotBytes, magic);
                    return;
                }
                if (recvStatus == kCombineDafRecvDone) {
                    recvDone[peer] = 1;
                    --remainingPeers;
                    madeProgress = true;
                }
            }
            if (madeProgress) {
                retries = 0;
            } else {
                ++retries;
            }
        }
        if (remainingPeers > 0) {
            TileXREpAbortCrossNodeCombineLocal(sync, workspaceGM, totalBytes, rankSize, slotBytes, magic);
            return;
        }

        if (localRankSize > 1) {
            sync.SetInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineRelayReady);
            for (int32_t peer = localNodeStart; peer < localNodeEnd; ++peer) {
                sync.WaitRankInnerFlag(static_cast<int32_t>(magic), TileXREp::kEpStepCombineRelayReady, peer);
            }
        }
        TileXREpDrainCrossNodeCombineRows(localIpcWindow, recvWindow, yOutGM, rank, rankSize, localRankSize, bs, h,
            topK, maxRoutesPerSrc, rowBytes, slotBytes, payloadBytesPerSlot, tBuf);
    }
}

void launch_tilexr_ep_combine_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs, GM_ADDR expertOut,
    GM_ADDR assistInfoForCombine, GM_ADDR epRecvCounts, GM_ADDR yOut, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, int64_t dtype, int64_t dtypeBytes, int64_t maxRoutesPerSrc, int64_t rowBytes,
    int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes, int64_t magic)
{
    tilexr_ep_combine_kernel<<<blockDim, nullptr, stream>>>(commArgs, expertOut, assistInfoForCombine, epRecvCounts,
        yOut, bs, h, topK, moeExpertNum, dtype, dtypeBytes, maxRoutesPerSrc, rowBytes, payloadBytesPerSlot,
        assistBytesPerSlot, slotBytes, totalBytes, magic);
}

void launch_tilexr_ep_combine_cross_node_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs, GM_ADDR expertOut,
    GM_ADDR assistInfoForCombine, GM_ADDR epRecvCounts, GM_ADDR yOut, GM_ADDR workspace, int64_t bs, int64_t h,
    int64_t topK, int64_t moeExpertNum, int64_t dtype, int64_t dtypeBytes, int64_t maxRoutesPerSrc, int64_t rowBytes,
    int64_t payloadBytesPerSlot, int64_t assistBytesPerSlot, int64_t slotBytes, int64_t totalBytes, int64_t magic)
{
    tilexr_ep_combine_cross_node_kernel<<<blockDim, nullptr, stream>>>(commArgs, expertOut, assistInfoForCombine,
        epRecvCounts, yOut, workspace, bs, h, topK, moeExpertNum, dtype, dtypeBytes, maxRoutesPerSrc, rowBytes,
        payloadBytesPerSlot, assistBytesPerSlot, slotBytes, totalBytes, magic);
}
