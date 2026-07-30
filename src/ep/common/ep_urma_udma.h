#ifndef TILEXR_EP_COMMON_EP_URMA_UDMA_H
#define TILEXR_EP_COMMON_EP_URMA_UDMA_H

#include "tilexr_udma.h"

namespace TileXREp {

// S22 gives each Send AIV an independent QP. Keep the optimized WQE/CQE path
// private to the operator instead of changing TileXR's shared UDMA helpers.
__aicore__ inline __gm__ TileXR::UDMAMemInfo *EpUrmaGetRemoteMemInfo(
    __gm__ TileXR::UDMAInfo *info, uint32_t rank, uint32_t qpIdx)
{
    return reinterpret_cast<__gm__ TileXR::UDMAMemInfo *>(
        info->memPtr + sizeof(TileXR::UDMAMemInfo) * (rank * info->qpNum + qpIdx));
}

__aicore__ inline uint32_t EpUrmaPollCQ(
    __gm__ TileXR::UDMAInfo *info, uint32_t rank, uint32_t qpIdx, uint32_t target)
{
    if (target == 0) {
        return 0;
    }
    __gm__ TileXR::UDMACQCtx *cq = TileXR::UDMAGetSCQCtx(info, rank, qpIdx);
    __gm__ TileXR::UDMAWQCtx *sq = TileXR::UDMAGetWQCtx(info, rank, qpIdx);
    const uint64_t cqBase = cq->bufAddr;
    const uint32_t cqeSize = 1U << cq->baseBkShift;
    uint32_t tail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(cq->tailAddr), 0);
    while (tail != target) {
        __gm__ uint32_t *cqeWords = reinterpret_cast<__gm__ uint32_t *>(
            cqBase + cqeSize * (tail & (TileXR::TILEXR_UDMA_CQ_DEPTH - 1)));
        const bool validOwner = ((tail / TileXR::TILEXR_UDMA_CQ_DEPTH) & 1U) != 0;
        uint32_t word0 = 0;
        uint32_t retries = 0;
        while (retries < TileXR::TILEXR_UDMA_MAX_RETRY_TIMES) {
            TileXR::UDMACleanCacheLines(
                reinterpret_cast<__gm__ uint8_t *>(cqeWords), sizeof(TileXR::UDMACqeCtx));
            __asm__ __volatile__("" ::: "memory");
            word0 = cqeWords[0];
            __asm__ __volatile__("" ::: "memory");
            if ((validOwner ^ ((word0 & (1U << 2U)) != 0)) != 0) {
                break;
            }
            ++retries;
        }
        if (retries >= TileXR::TILEXR_UDMA_MAX_RETRY_TIMES) {
            return 0xFFU;
        }
        const uint8_t status = static_cast<uint8_t>((word0 >> 24U) & 0xFFU);
        const uint8_t subStatus = static_cast<uint8_t>((word0 >> 16U) & 0xFFU);
        if (status != 0 || subStatus != 0) {
            return (static_cast<uint32_t>(status) << 8U) | subStatus;
        }
        ++tail;
    }
    TileXR::UDMAPollCQUpdateInfo(tail, cq, sq);
    return 0;
}

__aicore__ inline void EpUrmaFillWriteWqe(
    __gm__ uint8_t *wqe, __gm__ uint8_t *remoteAddr,
    __gm__ TileXR::UDMAMemInfo *remote, uint32_t head, uint32_t depth,
    uint64_t bytes, __gm__ uint8_t *localAddr)
{
    __gm__ uint32_t *sqe = reinterpret_cast<__gm__ uint32_t *>(wqe);
    const uint32_t owner = (head & depth) == 0U ? 1U : 0U;
    sqe[0] = (head % depth) | (0b00100010U << 16U) |
        ((remote->tokenValueValid ? 1U : 0U) << 28U) |
        ((remote->rmtJettyType & 0x3U) << 29U) | (owner << 31U);
    sqe[1] = (static_cast<uint32_t>(remote->targetHint) & 0xFFU) |
        (static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE) << 8U);
    sqe[2] = (remote->tpn & 0xFFFFFFU) | (1U << 24U);
    sqe[3] = remote->tid & 0xFFFFFU;
    __gm__ uint32_t *eid = reinterpret_cast<__gm__ uint32_t *>(remote->eidAddr);
    sqe[4] = eid[0];
    sqe[5] = eid[1];
    sqe[6] = eid[2];
    sqe[7] = eid[3];
    sqe[8] = remote->rmtTokenValue;
    sqe[9] = 0;
    const uint64_t remoteValue = reinterpret_cast<uint64_t>(remoteAddr);
    sqe[10] = static_cast<uint32_t>(remoteValue);
    sqe[11] = static_cast<uint32_t>(remoteValue >> 32U);

    __gm__ uint32_t *sge = sqe + sizeof(TileXR::UDMASqeCtx) / sizeof(uint32_t);
    const uint64_t localValue = reinterpret_cast<uint64_t>(localAddr);
    sge[0] = static_cast<uint32_t>(bytes);
    sge[1] = 0;
    sge[2] = static_cast<uint32_t>(localValue);
    sge[3] = static_cast<uint32_t>(localValue >> 32U);
}

__aicore__ inline void EpUrmaPostWrite(
    __gm__ TileXR::UDMAInfo *info, __gm__ uint8_t *remoteAddr,
    __gm__ uint8_t *localAddr, uint32_t rank, uint32_t qpIdx, uint64_t bytes)
{
    __gm__ TileXR::UDMAWQCtx *sq = TileXR::UDMAGetWQCtx(info, rank, qpIdx);
    const uint32_t wqeSize = 1U << sq->baseBkShift;
    uint32_t head = ld_dev(reinterpret_cast<__gm__ uint32_t *>(sq->headAddr), 0);
    uint32_t count = ld_dev(reinterpret_cast<__gm__ uint32_t *>(sq->wqeCntAddr), 0);
    const uint32_t tail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(sq->tailAddr), 0);
    if ((count + 10U) % TileXR::TILEXR_UDMA_SQ_BB_COUNT ==
        tail % TileXR::TILEXR_UDMA_SQ_BB_COUNT) {
        const uint32_t target = tail + TileXR::TILEXR_UDMA_NUM_CQE_PER_POLL > count ?
            count : tail + TileXR::TILEXR_UDMA_NUM_CQE_PER_POLL;
        (void)EpUrmaPollCQ(info, rank, qpIdx, target);
    }

    __gm__ uint8_t *wqe = reinterpret_cast<__gm__ uint8_t *>(
        sq->bufAddr + wqeSize * (head % TileXR::TILEXR_UDMA_SQ_BB_COUNT));
    EpUrmaFillWriteWqe(wqe, remoteAddr, EpUrmaGetRemoteMemInfo(info, rank, qpIdx),
        head, sq->depth, bytes, localAddr);
    AscendC::PipeBarrier<PIPE_ALL>();
    TileXR::UDMACleanCacheLines(wqe, wqeSize);
    AscendC::PipeBarrier<PIPE_ALL>();
    ++head;
    st_dev(head, reinterpret_cast<__gm__ uint32_t *>(sq->dbAddr), 0);
    st_dev(head, reinterpret_cast<__gm__ uint32_t *>(sq->headAddr), 0);
    st_dev(++count, reinterpret_cast<__gm__ uint32_t *>(sq->wqeCntAddr), 0);
}

template <typename T>
__aicore__ inline void EpUrmaUDMAPutNbi(
    const __gm__ TileXR::CommArgs *args, int targetRank, const __gm__ T *localSrc,
    uint64_t byteOffset, uint32_t byteCount, uint32_t qpIdx = 0)
{
    if (!TileXR::UDMARegistryEnabled(args)) {
        return;
    }
    __gm__ TileXR::UDMAInfo *info = TileXR::GetUDMAInfo(args);
    if (qpIdx >= info->qpNum) {
        return;
    }
    __gm__ TileXR::TileXRUDMARegistry *registry = TileXR::GetUDMARegistry(args);
    if (!TileXR::UDMARegisteredRangeValid(registry, targetRank, byteOffset, byteCount)) {
        return;
    }
    EpUrmaPostWrite(info, TileXR::UDMARegisteredRemoteAddr(registry, targetRank, byteOffset),
        reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ T *>(localSrc)),
        static_cast<uint32_t>(targetRank), qpIdx, byteCount);
}

__aicore__ inline uint32_t EpUrmaUDMAQuiet(
    const __gm__ TileXR::CommArgs *args, int targetRank, uint32_t qpIdx = 0)
{
    if (!TileXR::UDMAEnabled(args)) {
        return 0;
    }
    __gm__ TileXR::UDMAInfo *info = TileXR::GetUDMAInfo(args);
    if (qpIdx >= info->qpNum) {
        return 0;
    }
    __gm__ TileXR::UDMAWQCtx *sq =
        TileXR::UDMAGetWQCtx(info, static_cast<uint32_t>(targetRank), qpIdx);
    const uint32_t count = ld_dev(reinterpret_cast<__gm__ uint32_t *>(sq->wqeCntAddr), 0);
    return EpUrmaPollCQ(info, static_cast<uint32_t>(targetRank), qpIdx, count);
}

} // namespace TileXREp

#endif // TILEXR_EP_COMMON_EP_URMA_UDMA_H
