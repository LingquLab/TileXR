#ifndef TILEXR_EP_COMBINE_UDMA_WQE_SIMT_H
#define TILEXR_EP_COMBINE_UDMA_WQE_SIMT_H

#include "simt_api/asc_simt.h"

namespace {

constexpr uint32_t kCombineDafChunkBytes = 64U * 1024U;
constexpr uint32_t kCombineDafQpIdx = 0U;
constexpr uint32_t kCombineDafSimtThreads = 128U;

#if !defined(TILEXR_COMBINE_DAF_WQE_MODE_API) && !defined(TILEXR_COMBINE_DAF_WQE_MODE_SCALAR_RAW) && \
    !defined(TILEXR_COMBINE_DAF_WQE_MODE_SIMT_RAW)
#define TILEXR_COMBINE_DAF_WQE_MODE_SIMT_RAW 1
#endif

#if (defined(TILEXR_COMBINE_DAF_WQE_MODE_API) + defined(TILEXR_COMBINE_DAF_WQE_MODE_SCALAR_RAW) + \
    defined(TILEXR_COMBINE_DAF_WQE_MODE_SIMT_RAW)) != 1
#error "Define exactly one TILEXR_COMBINE_DAF_WQE_MODE_* macro"
#endif

static_assert(kCombineDafChunkBytes % TileXR::DATA_AS_FLAG_BLOCK_BYTES == 0U,
    "Combine DataAsFlag chunk size must stay aligned to DataAsFlag blocks");

struct CombineDafWqeTask {
    uint64_t localAddr;
    uint64_t remoteAddr;
    uint64_t remoteOffset;
    uint64_t wqeAddr;
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    uint32_t byteCount;
    uint32_t peer;
    uint32_t headIndex;
    uint32_t wqeSize;
    uint32_t tokenEn;
    uint32_t rmtJettyType;
    uint32_t targetHint;
    uint32_t tpId;
    uint32_t rmtJettyOrSegId;
    uint32_t rmtTokenValue;
    uint32_t reserved;
};

struct CombineDafDoorbellTask {
    uint32_t peer;
    uint32_t finalHead;
    uint32_t finalWqeCnt;
    uint32_t reserved;
    uint64_t dbAddr;
    uint64_t headAddr;
    uint64_t wqeCntAddr;
};

__aicore__ inline bool TileXREpBuildCombineDataAsFlagWqeTasks(const __gm__ TileXR::CommArgs *args,
    GM_ADDR dafSendWindow, int32_t rank, int32_t rankSize, int32_t localRankSize, int64_t totalBytes,
    int64_t slotBytes, int32_t roundIndex, __ubuf__ CombineDafWqeTask *tasks, uint32_t maxTasks,
    __ubuf__ uint32_t *taskCountOut)
{
    if (args == nullptr || tasks == nullptr || taskCountOut == nullptr) {
        return false;
    }

    __gm__ TileXR::TileXRUDMARegistry *registry = TileXR::GetUDMARegistry(args);
    if (registry == nullptr) {
        return false;
    }

    const uint64_t dafSlotBytes = static_cast<uint64_t>(CombineDataAsFlagSlotBytes(slotBytes));
    const uint64_t remoteRecvBase = static_cast<uint64_t>(
        CombineDataAsFlagRemoteRecvRoundOffset(totalBytes, rankSize, slotBytes, roundIndex));
    uint32_t taskCount = 0U;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank || TileXREpSameNode(peer, rank, localRankSize)) {
            continue;
        }

        for (uint64_t offset = 0; offset < dafSlotBytes; offset += kCombineDafChunkBytes) {
            const uint64_t remaining = dafSlotBytes - offset;
            const uint32_t bytes = remaining < static_cast<uint64_t>(kCombineDafChunkBytes) ?
                static_cast<uint32_t>(remaining) : kCombineDafChunkBytes;
            const uint64_t remoteOffset = remoteRecvBase + static_cast<uint64_t>(rank) * dafSlotBytes + offset;
            if (taskCount >= maxTasks ||
                !TileXR::UDMARegisteredRangeValid(registry, peer, remoteOffset, static_cast<uint64_t>(bytes))) {
                return false;
            }

            tasks[taskCount].localAddr = reinterpret_cast<uint64_t>(
                dafSendWindow + CombineDataAsFlagSlotOffset(peer, slotBytes) + static_cast<int64_t>(offset));
            tasks[taskCount].remoteAddr = reinterpret_cast<uint64_t>(
                TileXR::UDMARegisteredRemoteAddr(registry, peer, remoteOffset));
            tasks[taskCount].remoteOffset = remoteOffset;
            tasks[taskCount].wqeAddr = 0U;
            tasks[taskCount].rmtEidL = 0U;
            tasks[taskCount].rmtEidH = 0U;
            tasks[taskCount].byteCount = bytes;
            tasks[taskCount].peer = static_cast<uint32_t>(peer);
            tasks[taskCount].headIndex = 0U;
            tasks[taskCount].wqeSize = 0U;
            tasks[taskCount].tokenEn = 0U;
            tasks[taskCount].rmtJettyType = 0U;
            tasks[taskCount].targetHint = 0U;
            tasks[taskCount].tpId = 0U;
            tasks[taskCount].rmtJettyOrSegId = 0U;
            tasks[taskCount].rmtTokenValue = 0U;
            tasks[taskCount].reserved = 0U;
            ++taskCount;
        }
    }

    taskCountOut[0] = taskCount;
    return true;
}

__aicore__ inline bool TileXREpPlanCombineDataAsFlagPeerWqeReservations(const __gm__ TileXR::CommArgs *args,
    __ubuf__ CombineDafWqeTask *tasks, uint32_t taskCount, __ubuf__ CombineDafDoorbellTask *doorbells,
    uint32_t maxDoorbells, __ubuf__ uint32_t *doorbellCountOut)
{
    if (args == nullptr || tasks == nullptr || doorbells == nullptr || doorbellCountOut == nullptr) {
        return false;
    }

    __gm__ TileXR::UDMAInfo *udmaInfo = TileXR::GetUDMAInfo(args);
    __gm__ TileXR::TileXRUDMARegistry *registry = TileXR::GetUDMARegistry(args);
    if (udmaInfo == nullptr || registry == nullptr) {
        return false;
    }

    doorbellCountOut[0] = 0U;
    const uint32_t wqeBbCnt = TileXR::UDMAWqeBBCnt(TileXR::UDMAOpcode::WRITE);
    for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
        const uint32_t peer = tasks[taskIdx].peer;
        __gm__ TileXR::UDMAWQCtx *wq = TileXR::UDMAGetWQCtx(udmaInfo, peer, kCombineDafQpIdx);
        __gm__ TileXR::UDMACQCtx *scq = TileXR::UDMAGetSCQCtx(udmaInfo, peer, kCombineDafQpIdx);
        __gm__ TileXR::UDMAMemInfo *remoteMem = TileXR::UDMAGetRemoteMemInfo(udmaInfo, peer);
        if (wq == nullptr || scq == nullptr || remoteMem == nullptr || wq->bufAddr == 0 || wq->headAddr == 0 ||
            wq->wqeCntAddr == 0 || wq->dbAddr == 0 || scq->bufAddr == 0 || scq->tailAddr == 0 ||
            remoteMem->eidAddr == 0) {
            return false;
        }

        const bool queueContextReady = wq->tailAddr != 0 && scq->dbAddr != 0;
        if (!queueContextReady) {
            return false;
        }

        uint32_t doorbellIdx = doorbellCountOut[0];
        for (uint32_t idx = 0; idx < doorbellCountOut[0]; ++idx) {
            if (doorbells[idx].headAddr == wq->headAddr && doorbells[idx].dbAddr == wq->dbAddr &&
                doorbells[idx].wqeCntAddr == wq->wqeCntAddr) {
                doorbellIdx = idx;
                break;
            }
        }
        if (doorbellIdx == doorbellCountOut[0]) {
            if (doorbellIdx >= maxDoorbells) {
                return false;
            }
            doorbells[doorbellIdx].peer = peer;
            doorbells[doorbellIdx].finalHead = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0);
            doorbells[doorbellIdx].finalWqeCnt = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->wqeCntAddr), 0);
            doorbells[doorbellIdx].reserved = 0U;
            doorbells[doorbellIdx].dbAddr = wq->dbAddr;
            doorbells[doorbellIdx].headAddr = wq->headAddr;
            doorbells[doorbellIdx].wqeCntAddr = wq->wqeCntAddr;
            doorbellCountOut[0] = doorbellIdx + 1U;
        }

        uint32_t curHead = doorbells[doorbellIdx].finalHead;
        uint32_t wqeCnt = doorbells[doorbellIdx].finalWqeCnt;
        const uint32_t wqeSize = 1U << wq->baseBkShift;
        __gm__ uint64_t *rmtEid = reinterpret_cast<__gm__ uint64_t *>(remoteMem->eidAddr);
        TileXR::UDMAPollCQWhenSQOverflow(udmaInfo, wq, wqeCnt, peer, kCombineDafQpIdx);
        tasks[taskIdx].headIndex = curHead;
        tasks[taskIdx].remoteAddr = reinterpret_cast<uint64_t>(
            TileXR::UDMARegisteredRemoteAddr(registry, static_cast<int>(peer), tasks[taskIdx].remoteOffset));
        tasks[taskIdx].wqeAddr = wq->bufAddr +
            static_cast<uint64_t>(wqeSize) * (curHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT);
        tasks[taskIdx].wqeSize = wqeSize;
        tasks[taskIdx].tokenEn = remoteMem->tokenValueValid ? 1U : 0U;
        tasks[taskIdx].rmtJettyType = remoteMem->rmtJettyType;
        tasks[taskIdx].targetHint = remoteMem->targetHint;
        tasks[taskIdx].tpId = remoteMem->tpn;
        tasks[taskIdx].rmtJettyOrSegId = remoteMem->tid;
        tasks[taskIdx].rmtTokenValue = remoteMem->rmtTokenValue;
        tasks[taskIdx].rmtEidL = rmtEid[0];
        tasks[taskIdx].rmtEidH = rmtEid[1];
        doorbells[doorbellIdx].finalHead = curHead + wqeBbCnt;
        doorbells[doorbellIdx].finalWqeCnt = wqeCnt + 1U;
    }

    return true;
}

__aicore__ inline void TileXREpCommitCombineDataAsFlagPeerWqeReservations(
    __ubuf__ CombineDafDoorbellTask *doorbells, uint32_t doorbellCount)
{
    if (doorbells == nullptr) {
        return;
    }

    for (uint32_t idx = 0; idx < doorbellCount; ++idx) {
        st_dev(doorbells[idx].finalHead, reinterpret_cast<__gm__ uint32_t *>(doorbells[idx].dbAddr), 0);
        st_dev(doorbells[idx].finalHead, reinterpret_cast<__gm__ uint32_t *>(doorbells[idx].headAddr), 0);
        st_dev(doorbells[idx].finalWqeCnt, reinterpret_cast<__gm__ uint32_t *>(doorbells[idx].wqeCntAddr), 0);
    }
}

__aicore__ inline bool TileXREpReserveCombineDataAsFlagPeerWqes(const __gm__ TileXR::CommArgs *args,
    __ubuf__ CombineDafWqeTask *tasks, uint32_t taskCount, __ubuf__ CombineDafDoorbellTask *doorbells,
    uint32_t maxDoorbells, __ubuf__ uint32_t *doorbellCountOut)
{
    if (!TileXREpPlanCombineDataAsFlagPeerWqeReservations(args, tasks, taskCount, doorbells, maxDoorbells,
            doorbellCountOut)) {
        return false;
    }
    return true;
}

__simt_callee__ inline void TileXREpCombineSimtWriteWqeRaw(
    __gm__ uint8_t *wqeAddr, const CombineDafWqeTask &task)
{
    constexpr uint32_t kWriteOpcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
    const uint32_t owner = (task.headIndex & TileXR::TILEXR_UDMA_SQ_BB_COUNT) == 0U ? 1U : 0U;
    __gm__ uint32_t *wqeWords = reinterpret_cast<__gm__ uint32_t *>(wqeAddr);

    wqeWords[0] = (0b00100010U << 16) |
        ((task.tokenEn & 0x1U) << 28) |
        ((task.rmtJettyType & 0x3U) << 29) |
        (owner << 31);
    wqeWords[1] = (task.targetHint & 0xFFU) | (kWriteOpcode << 8);
    wqeWords[2] = (task.tpId & 0xFFFFFFU) | (1U << 24);
    wqeWords[3] = task.rmtJettyOrSegId & 0xFFFFFU;

    __gm__ uint64_t *wqeWords64 = reinterpret_cast<__gm__ uint64_t *>(wqeAddr);
    wqeWords64[2] = task.rmtEidL;
    wqeWords64[3] = task.rmtEidH;

    wqeWords[8] = task.rmtTokenValue;
    wqeWords[9] = 0U;
    wqeWords[10] = static_cast<uint32_t>(task.remoteAddr & 0xFFFFFFFFULL);
    wqeWords[11] = static_cast<uint32_t>((task.remoteAddr >> 32) & 0xFFFFFFFFULL);
    wqeWords[12] = task.byteCount;
    wqeWords[13] = 0U;
    wqeWords[14] = static_cast<uint32_t>(task.localAddr & 0xFFFFFFFFULL);
    wqeWords[15] = static_cast<uint32_t>((task.localAddr >> 32) & 0xFFFFFFFFULL);
}

__aicore__ inline bool TileXREpPostCombineDataAsFlagScalarRawSequential(
    const __gm__ TileXR::CommArgs *args, __ubuf__ CombineDafWqeTask *tasks, uint32_t taskCount)
{
    if (args == nullptr || tasks == nullptr) {
        return false;
    }

    __gm__ TileXR::UDMAInfo *udmaInfo = TileXR::GetUDMAInfo(args);
    if (udmaInfo == nullptr) {
        return false;
    }

    for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
        const uint32_t peer = tasks[taskIdx].peer;
        __gm__ TileXR::UDMAWQCtx *wq = TileXR::UDMAGetWQCtx(udmaInfo, peer, kCombineDafQpIdx);
        __gm__ TileXR::UDMAMemInfo *remoteMem = TileXR::UDMAGetRemoteMemInfo(udmaInfo, peer);
        if (wq == nullptr || remoteMem == nullptr || wq->bufAddr == 0 || wq->headAddr == 0 ||
            wq->wqeCntAddr == 0 || wq->dbAddr == 0 || remoteMem->eidAddr == 0 || tasks[taskIdx].remoteAddr == 0) {
            return false;
        }

        uint32_t curHead = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0);
        uint32_t wqeCnt = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->wqeCntAddr), 0);
        TileXR::UDMAPollCQWhenSQOverflow(udmaInfo, wq, wqeCnt, peer, kCombineDafQpIdx);

        const uint32_t wqeSize = 1U << wq->baseBkShift;
        __gm__ uint8_t *wqeAddr = reinterpret_cast<__gm__ uint8_t *>(
            wq->bufAddr + static_cast<uint64_t>(wqeSize) * (curHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT));
        __gm__ TileXR::UDMASqeCtx *sqeCtx = reinterpret_cast<__gm__ TileXR::UDMASqeCtx *>(wqeAddr);
        TileXR::UDMAFillSqeCtx(sqeCtx, reinterpret_cast<__gm__ uint8_t *>(tasks[taskIdx].remoteAddr),
            remoteMem, curHead, TileXR::UDMAOpcode::WRITE, nullptr);
        __gm__ TileXR::UDMASgeCtx *sgeCtx = reinterpret_cast<__gm__ TileXR::UDMASgeCtx *>(
            TileXR::UDMAGetSgeCtxAddr(wqeAddr, TileXR::UDMAOpcode::WRITE));
        TileXR::UDMAFillSgeCtx(
            sgeCtx, tasks[taskIdx].byteCount, reinterpret_cast<__gm__ uint8_t *>(tasks[taskIdx].localAddr));
        TileXR::UDMACleanCacheLines(
            wqeAddr, static_cast<uint64_t>(wqeSize) * TileXR::UDMAWqeBBCnt(TileXR::UDMAOpcode::WRITE));
        AscendC::PipeBarrier<PIPE_ALL>();

        const uint32_t nextHead = curHead + TileXR::UDMAWqeBBCnt(TileXR::UDMAOpcode::WRITE);
        st_dev(nextHead, reinterpret_cast<__gm__ uint32_t *>(wq->dbAddr), 0);
        st_dev(nextHead, reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0);
        st_dev(wqeCnt + 1U, reinterpret_cast<__gm__ uint32_t *>(wq->wqeCntAddr), 0);

        if (taskIdx + 1U == taskCount || tasks[taskIdx + 1U].peer != peer) {
            TileXR::UDMAQuiet(args, static_cast<int>(peer));
        }
    }
    return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kCombineDafSimtThreads) inline void TileXREpCombineSimtWriteWqeVf(
    __ubuf__ CombineDafWqeTask *tasks, uint32_t taskCount)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    for (uint32_t taskIdx = tid; taskIdx < taskCount; taskIdx += stride) {
        CombineDafWqeTask task;
        task.localAddr = tasks[taskIdx].localAddr;
        task.remoteAddr = tasks[taskIdx].remoteAddr;
        task.remoteOffset = tasks[taskIdx].remoteOffset;
        task.wqeAddr = tasks[taskIdx].wqeAddr;
        task.rmtEidL = tasks[taskIdx].rmtEidL;
        task.rmtEidH = tasks[taskIdx].rmtEidH;
        task.byteCount = tasks[taskIdx].byteCount;
        task.peer = tasks[taskIdx].peer;
        task.headIndex = tasks[taskIdx].headIndex;
        task.wqeSize = tasks[taskIdx].wqeSize;
        task.tokenEn = tasks[taskIdx].tokenEn;
        task.rmtJettyType = tasks[taskIdx].rmtJettyType;
        task.targetHint = tasks[taskIdx].targetHint;
        task.tpId = tasks[taskIdx].tpId;
        task.rmtJettyOrSegId = tasks[taskIdx].rmtJettyOrSegId;
        task.rmtTokenValue = tasks[taskIdx].rmtTokenValue;
        task.reserved = tasks[taskIdx].reserved;
        __gm__ uint8_t *wqeAddr = reinterpret_cast<__gm__ uint8_t *>(task.wqeAddr);
        TileXREpCombineSimtWriteWqeRaw(wqeAddr, task);
    }

    asc_syncthreads();
}

__aicore__ inline bool TileXREpPostCombineDataAsFlagWqes(const __gm__ TileXR::CommArgs *args,
    GM_ADDR dafSendWindow, int32_t rank, int32_t rankSize, int32_t localRankSize, int64_t totalBytes,
    int64_t slotBytes, int32_t roundIndex, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    constexpr uint32_t kMaxDoorbells = TileXR::TILEXR_MAX_RANK_SIZE;
    constexpr uint32_t kMaxTaskBytes = kEpCopyTileBytes -
        static_cast<uint32_t>(sizeof(CombineDafDoorbellTask) * kMaxDoorbells) - sizeof(uint32_t) * 2U;
    constexpr uint32_t kMaxTasks = kMaxTaskBytes / sizeof(CombineDafWqeTask);
    static_assert(kMaxTasks > 0U, "Combine DataAsFlag SIMT helper needs task storage");

    AscendC::LocalTensor<uint8_t> raw = tBuf.GetWithOffset<uint8_t>(kEpCopyTileBytes, kEpSyncUbBytes);
    __ubuf__ uint8_t *rawAddr = reinterpret_cast<__ubuf__ uint8_t *>(raw.GetPhyAddr());
    __ubuf__ CombineDafWqeTask *tasks = reinterpret_cast<__ubuf__ CombineDafWqeTask *>(rawAddr);
    __ubuf__ CombineDafDoorbellTask *doorbells = reinterpret_cast<__ubuf__ CombineDafDoorbellTask *>(
        rawAddr + sizeof(CombineDafWqeTask) * kMaxTasks);
    __ubuf__ uint32_t *counts = reinterpret_cast<__ubuf__ uint32_t *>(
        rawAddr + sizeof(CombineDafWqeTask) * kMaxTasks + sizeof(CombineDafDoorbellTask) * kMaxDoorbells);
    __ubuf__ uint32_t *taskCount = counts;
    __ubuf__ uint32_t *doorbellCount = counts + 1;
    taskCount[0] = 0U;
    doorbellCount[0] = 0U;
    if (!TileXREpBuildCombineDataAsFlagWqeTasks(args, dafSendWindow, rank, rankSize, localRankSize, totalBytes,
            slotBytes, roundIndex, tasks, kMaxTasks, taskCount)) {
        return false;
    }
    if (taskCount[0] == 0U) {
        return true;
    }
#if defined(TILEXR_COMBINE_DAF_WQE_MODE_API)
    {
        for (uint32_t idx = 0; idx < taskCount[0]; ++idx) {
            TileXR::UDMAPutNbi<uint8_t>(args, static_cast<int>(tasks[idx].peer),
                reinterpret_cast<__gm__ uint8_t *>(tasks[idx].localAddr), tasks[idx].remoteOffset,
                tasks[idx].byteCount);
            if (idx + 1U == taskCount[0] || tasks[idx + 1U].peer != tasks[idx].peer) {
                TileXR::UDMAQuiet(args, static_cast<int>(tasks[idx].peer));
            }
        }
        return true;
    }
#elif defined(TILEXR_COMBINE_DAF_WQE_MODE_SCALAR_RAW)
    {
        return TileXREpPostCombineDataAsFlagScalarRawSequential(args, tasks, taskCount[0]);
    }
#elif defined(TILEXR_COMBINE_DAF_WQE_MODE_SIMT_RAW)
    if (!TileXREpReserveCombineDataAsFlagPeerWqes(args, tasks, taskCount[0], doorbells, kMaxDoorbells,
            doorbellCount)) {
        return false;
    }

    asc_vf_call<TileXREpCombineSimtWriteWqeVf>(dim3 {kCombineDafSimtThreads, 1, 1}, tasks, taskCount[0]);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t idx = 0; idx < taskCount[0]; ++idx) {
        TileXR::UDMACleanCacheLines(
            reinterpret_cast<__gm__ uint8_t *>(tasks[idx].wqeAddr), static_cast<uint64_t>(tasks[idx].wqeSize));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    TileXREpCommitCombineDataAsFlagPeerWqeReservations(doorbells, doorbellCount[0]);
    for (uint32_t idx = 0; idx < doorbellCount[0]; ++idx) {
        TileXR::UDMAQuiet(args, static_cast<int>(doorbells[idx].peer));
    }
    return true;
#endif
}

} // namespace

#endif // TILEXR_EP_COMBINE_UDMA_WQE_SIMT_H
