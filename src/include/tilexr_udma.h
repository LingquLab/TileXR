/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_H
#define TILEXR_UDMA_H

#include "kernel_operator.h"
#include "comm_args.h"
#include "tilexr_udma_reg.h"
#include "tilexr_udma_types.h"

namespace TileXR {

constexpr uint32_t TILEXR_UDMA_SQE_FLAG_COMPLETION = 0x20U;
constexpr uint32_t TILEXR_UDMA_SQE_FLAG_STRONG_ORDER = 0x02U;
constexpr uint32_t TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION =
    TILEXR_UDMA_SQE_FLAG_COMPLETION | TILEXR_UDMA_SQE_FLAG_STRONG_ORDER;

constexpr uint32_t TILEXR_UDMA_STATUS_SUCCESS = 0U;
constexpr uint32_t TILEXR_UDMA_STATUS_CQ_TIMEOUT = 0xFFU;
constexpr uint32_t TILEXR_UDMA_STATUS_SQ_FULL = 0xFFFFFFFEU;
constexpr uint32_t TILEXR_UDMA_STATUS_INVALID = 0xFFFFFFFFU;
constexpr uint32_t TILEXR_UDMA_DEVICE_MAX_QP_COUNT = 8U;

/**
 * @file tilexr_udma.h
 * @brief Device-side UDMA wrapper for TileXR-registered memory.
 *
 * Host side registers ordinary device memory with TileXRUDMARegister. Device
 * kernels then use byte offsets into that registered region for PUT/GET/SIGNAL.
 * This header is self-contained for the device path and intentionally avoids
 * symmetric-memory APIs.
 */

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510)) || defined(TILEXR_UDMA_FORCE_ENABLE)
constexpr bool TILEXR_UDMA_ARCH_SUPPORTED = true;
#else
constexpr bool TILEXR_UDMA_ARCH_SUPPORTED = false;
#endif

struct UDMASignalParams {
    __gm__ uint64_t* sigAddr;
    uint64_t signal;
};

__aicore__ inline bool UDMAEnabled(const __gm__ CommArgs* args)
{
    return args != nullptr && ((args->extraFlag & ExtraFlag::UDMA) != 0) && args->udmaInfoPtr != nullptr;
}

__aicore__ inline bool UDMARegistryEnabled(const __gm__ CommArgs* args)
{
    return UDMAEnabled(args) && args->udmaRegistryPtr != nullptr;
}

__aicore__ inline uint32_t UDMAQpCount(const __gm__ CommArgs* args)
{
    if (!UDMAEnabled(args)) {
        return 0U;
    }
    const uint32_t qpNum = reinterpret_cast<__gm__ UDMAInfo*>(args->udmaInfoPtr)->qpNum;
    return qpNum > 0U && qpNum <= TILEXR_UDMA_DEVICE_MAX_QP_COUNT ? qpNum : 0U;
}

__aicore__ inline bool UDMAQpValid(const __gm__ CommArgs* args, uint32_t qpIdx)
{
    return qpIdx < UDMAQpCount(args);
}

__aicore__ inline bool UDMARankValid(const __gm__ CommArgs* args, int peer)
{
    return args != nullptr && args->rankSize > 0 && args->rankSize <= TILEXR_MAX_RANK_SIZE &&
        peer >= 0 && peer < args->rankSize && peer != args->rank;
}

__aicore__ inline bool UDMAQueueOperationValid(
    const __gm__ CommArgs* args, int peer, uint32_t qpIdx)
{
    return UDMARankValid(args, peer) && UDMAQpValid(args, qpIdx);
}

__aicore__ inline __gm__ UDMAInfo* GetUDMAInfo(const __gm__ CommArgs* args)
{
    return reinterpret_cast<__gm__ UDMAInfo*>(args->udmaInfoPtr);
}

__aicore__ inline __gm__ TileXRUDMARegistry* GetUDMARegistry(const __gm__ CommArgs* args)
{
    return reinterpret_cast<__gm__ TileXRUDMARegistry*>(args->udmaRegistryPtr);
}

__aicore__ inline bool UDMARegisteredRangeValid(
    const __gm__ TileXRUDMARegistry* registry, int targetRank, uint64_t byteOffset, uint64_t byteCount)
{
    if (registry == nullptr || registry->magic != TILEXR_UDMA_REGISTRY_MAGIC ||
        registry->version != TILEXR_UDMA_REGISTRY_VERSION || registry->regionCount == 0 ||
        registry->regionCount > TILEXR_UDMA_MAX_REGIONS || registry->rankSize == 0 ||
        registry->rankSize > TILEXR_MAX_RANK_SIZE || targetRank < 0 ||
        static_cast<uint32_t>(targetRank) >= registry->rankSize) {
        return false;
    }
    const auto& region = registry->regions[targetRank];
    if (region.base == nullptr || byteOffset > region.bytes) {
        return false;
    }
    return byteCount <= region.bytes - byteOffset;
}

__aicore__ inline __gm__ uint8_t* UDMARegisteredRemoteAddr(
    const __gm__ TileXRUDMARegistry* registry, int targetRank, uint64_t byteOffset)
{
    return reinterpret_cast<__gm__ uint8_t*>(registry->regions[targetRank].base + byteOffset);
}

__aicore__ inline void UDMACleanCacheLines(__gm__ uint8_t* addr, uint64_t length)
{
    if (addr == nullptr || length == 0) {
        return;
    }
    __gm__ uint8_t* start = reinterpret_cast<__gm__ uint8_t*>(
        reinterpret_cast<uint64_t>(addr) / TILEXR_UDMA_CACHE_LINE_SIZE * TILEXR_UDMA_CACHE_LINE_SIZE);
    __gm__ uint8_t* end = reinterpret_cast<__gm__ uint8_t*>(
        (reinterpret_cast<uint64_t>(addr) + length - 1) / TILEXR_UDMA_CACHE_LINE_SIZE *
        TILEXR_UDMA_CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint64_t i = 0; i <= static_cast<uint64_t>(end - start); i += TILEXR_UDMA_CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[i]);
        __asm__ __volatile__("");
    }
}

__aicore__ inline __gm__ UDMAWQCtx* UDMAGetWQCtx(__gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx)
{
    const uint32_t qpNum = udmaInfo == nullptr ? 0U : udmaInfo->qpNum;
    if (qpNum == 0U || qpNum > TILEXR_UDMA_DEVICE_MAX_QP_COUNT || qpIdx >= qpNum) {
        return nullptr;
    }
    const uint64_t entry = static_cast<uint64_t>(pe) * qpNum + qpIdx;
    return reinterpret_cast<__gm__ UDMAWQCtx*>(udmaInfo->sqPtr + entry * sizeof(UDMAWQCtx));
}

__aicore__ inline __gm__ UDMACQCtx* UDMAGetSCQCtx(__gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx)
{
    const uint32_t qpNum = udmaInfo == nullptr ? 0U : udmaInfo->qpNum;
    if (qpNum == 0U || qpNum > TILEXR_UDMA_DEVICE_MAX_QP_COUNT || qpIdx >= qpNum) {
        return nullptr;
    }
    const uint64_t entry = static_cast<uint64_t>(pe) * qpNum + qpIdx;
    return reinterpret_cast<__gm__ UDMACQCtx*>(udmaInfo->scqPtr + entry * sizeof(UDMACQCtx));
}

__aicore__ inline __gm__ UDMAMemInfo* UDMAGetRemoteMemInfo(
    __gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx)
{
    const uint32_t qpNum = udmaInfo == nullptr ? 0U : udmaInfo->qpNum;
    if (qpNum == 0U || qpNum > TILEXR_UDMA_DEVICE_MAX_QP_COUNT || qpIdx >= qpNum) {
        return nullptr;
    }
    const uint64_t entry = static_cast<uint64_t>(pe) * qpNum + qpIdx;
    return reinterpret_cast<__gm__ UDMAMemInfo*>(
        udmaInfo->memPtr + sizeof(UDMAMemInfo) * entry);
}

__aicore__ inline __gm__ UDMAMemInfo* UDMAGetRemoteMemInfo(__gm__ UDMAInfo* udmaInfo, uint32_t pe)
{
    return UDMAGetRemoteMemInfo(udmaInfo, pe, 0U);
}

__aicore__ inline uint32_t UDMAWqeBBCnt(UDMAOpcode opcode)
{
    return opcode == UDMAOpcode::WRITE_WITH_NOTIFY ? 2U : 1U;
}

__aicore__ inline void UDMAPollCQUpdateInfo(
    uint32_t cqTail, uint32_t sqTail, __gm__ UDMACQCtx* cqCtxEntry, __gm__ UDMAWQCtx* wqCtxEntry)
{
    st_dev(cqTail, reinterpret_cast<__gm__ uint32_t*>(cqCtxEntry->tailAddr), 0);
    st_dev(sqTail, reinterpret_cast<__gm__ uint32_t*>(wqCtxEntry->tailAddr), 0);
    st_dev(static_cast<uint32_t>(cqTail & 0xFFFFFFU),
        reinterpret_cast<__gm__ uint32_t*>(cqCtxEntry->dbAddr), 0);
}

__aicore__ inline uint32_t UDMAPollCQ(__gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx, uint32_t idx)
{
    if (udmaInfo == nullptr || udmaInfo->qpNum == 0U ||
        udmaInfo->qpNum > TILEXR_UDMA_DEVICE_MAX_QP_COUNT || qpIdx >= udmaInfo->qpNum || udmaInfo->sqPtr == 0U ||
        udmaInfo->scqPtr == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    __gm__ UDMACQCtx* cqCtxEntry = UDMAGetSCQCtx(udmaInfo, pe, qpIdx);
    __gm__ UDMAWQCtx* wqCtxEntry = UDMAGetWQCtx(udmaInfo, pe, qpIdx);
    if (wqCtxEntry->bufAddr == 0U || wqCtxEntry->headAddr == 0U ||
        wqCtxEntry->tailAddr == 0U || wqCtxEntry->depth != TILEXR_UDMA_SQ_BB_COUNT ||
        wqCtxEntry->baseBkShift >= 32U ||
        (1U << wqCtxEntry->baseBkShift) < sizeof(UDMASqeCtx) + sizeof(UDMASgeCtx) ||
        cqCtxEntry->bufAddr == 0U || cqCtxEntry->tailAddr == 0U || cqCtxEntry->dbAddr == 0U ||
        cqCtxEntry->depth != TILEXR_UDMA_CQ_DEPTH || cqCtxEntry->baseBkShift >= 32U ||
        (1U << cqCtxEntry->baseBkShift) < sizeof(UDMACqeCtx)) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    uint64_t cqBaseAddr = cqCtxEntry->bufAddr;
    uint32_t cqeSize = 1U << cqCtxEntry->baseBkShift;
    uint32_t cqTail = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cqCtxEntry->tailAddr), 0);
    uint32_t sqTail = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtxEntry->tailAddr), 0);
    const uint32_t completionCount = idx - cqTail;
    if (completionCount > cqCtxEntry->depth) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    for (uint32_t completed = 0U; completed < completionCount; ++completed) {
        __gm__ UDMACqeCtx* cqeAddr = reinterpret_cast<__gm__ UDMACqeCtx*>(
            cqBaseAddr + cqeSize * (cqTail & (TILEXR_UDMA_CQ_DEPTH - 1U)));
        bool validOwner = ((cqTail / TILEXR_UDMA_CQ_DEPTH) & 1U) != 0U;
        uint32_t times = 0;
        while ((validOwner ^ (cqeAddr->owner != 0)) == 0 && times < TILEXR_UDMA_MAX_RETRY_TIMES) {
            UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(cqeAddr), sizeof(UDMACqeCtx));
            ++times;
        }
        if (times >= TILEXR_UDMA_MAX_RETRY_TIMES) {
            return TILEXR_UDMA_STATUS_CQ_TIMEOUT;
        }
        uint8_t status = cqeAddr->status & 0xFF;
        uint8_t subStatus = cqeAddr->substatus & 0xFF;
        if (status != 0 || subStatus != 0) {
            return (static_cast<uint32_t>(status) << 8) | subStatus;
        }

        const uint32_t sqHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtxEntry->headAddr), 0);
        const uint32_t sqOutstanding = sqHead - sqTail;
        if (sqOutstanding > wqCtxEntry->depth || cqeAddr->entryIdx >= wqCtxEntry->depth) {
            return TILEXR_UDMA_STATUS_INVALID;
        }
        const uint32_t sqTailIdx = sqTail % wqCtxEntry->depth;
        const uint32_t entryDistance =
            (static_cast<uint32_t>(cqeAddr->entryIdx) + wqCtxEntry->depth - sqTailIdx) % wqCtxEntry->depth;
        if (entryDistance >= sqOutstanding) {
            return TILEXR_UDMA_STATUS_INVALID;
        }
        const uint32_t absoluteEntry = sqTail + entryDistance;
        __gm__ UDMASqeCtx* completedSqe = reinterpret_cast<__gm__ UDMASqeCtx*>(
            wqCtxEntry->bufAddr +
            (1U << wqCtxEntry->baseBkShift) * (absoluteEntry % wqCtxEntry->depth));
        const UDMAOpcode opcode = static_cast<UDMAOpcode>(completedSqe->opcode);
        if (opcode != UDMAOpcode::WRITE && opcode != UDMAOpcode::READ &&
            opcode != UDMAOpcode::WRITE_WITH_NOTIFY) {
            return TILEXR_UDMA_STATUS_INVALID;
        }
        const uint32_t reclaimedBb = entryDistance + UDMAWqeBBCnt(opcode);
        if (reclaimedBb > sqOutstanding) {
            return TILEXR_UDMA_STATUS_INVALID;
        }
        sqTail += reclaimedBb;
        ++cqTail;
        UDMAPollCQUpdateInfo(cqTail, sqTail, cqCtxEntry, wqCtxEntry);
    }
    return TILEXR_UDMA_STATUS_SUCCESS;
}

__aicore__ inline __gm__ uint8_t* UDMAGetSgeCtxAddr(__gm__ uint8_t* wqeAddr, UDMAOpcode opcode)
{
    if (opcode == UDMAOpcode::WRITE_WITH_NOTIFY) {
        return wqeAddr + sizeof(UDMASqeCtx) + sizeof(UDMANotifyCtx);
    }
    return wqeAddr + sizeof(UDMASqeCtx);
}

__aicore__ inline void UDMAFillNotifyData(
    __gm__ UDMASqeCtx* sqeCtx, uint32_t tid, uint32_t tokenValue, const UDMASignalParams* params)
{
    if (params == nullptr) {
        return;
    }
    __gm__ UDMANotifyCtx* notifyCtx =
        reinterpret_cast<__gm__ UDMANotifyCtx*>(reinterpret_cast<__gm__ uint8_t*>(sqeCtx) + sizeof(UDMASqeCtx));
    notifyCtx->notifyTokenId = tid & 0xFFFFF;
    notifyCtx->rsv = 0;
    notifyCtx->notifyTokenValue = tokenValue;
    notifyCtx->notifyAddrL = reinterpret_cast<uint64_t>(params->sigAddr) & 0xFFFFFFFF;
    notifyCtx->notifyAddrH = (reinterpret_cast<uint64_t>(params->sigAddr) >> 32) & 0xFFFFFFFF;
    notifyCtx->notifyDataL = params->signal & 0xFFFFFFFF;
    notifyCtx->notifyDataH = (params->signal >> 32) & 0xFFFFFFFF;
    notifyCtx->rsv2[0] = 0;
    notifyCtx->rsv2[1] = 0;
}

__aicore__ inline __gm__ uint8_t* UDMAGetSqLogicalAddr(
    __gm__ UDMAWQCtx* qpCtxEntry, uint32_t curHead, uint32_t logicalByteOffset)
{
    const uint64_t wqeSize = 1ULL << qpCtxEntry->baseBkShift;
    const uint64_t ringBytes = wqeSize * qpCtxEntry->depth;
    const uint64_t start = wqeSize * (curHead % qpCtxEntry->depth);
    return reinterpret_cast<__gm__ uint8_t*>(
        qpCtxEntry->bufAddr + (start + logicalByteOffset) % ringBytes);
}

__aicore__ inline void UDMAFillWrappedNotifyData(
    __gm__ UDMAWQCtx* qpCtxEntry, uint32_t curHead, uint32_t tid,
    uint32_t tokenValue, const UDMASignalParams* params)
{
    if (params == nullptr) {
        return;
    }
    const uint32_t baseOffset = sizeof(UDMASqeCtx);
    const uint64_t signalAddr = reinterpret_cast<uint64_t>(params->sigAddr);
    st_dev(tid & 0xFFFFFU, reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset)), 0);
    st_dev(tokenValue, reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset + 4U)), 0);
    st_dev(static_cast<uint32_t>(signalAddr), reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset + 8U)), 0);
    st_dev(static_cast<uint32_t>(signalAddr >> 32U), reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset + 12U)), 0);
    st_dev(static_cast<uint32_t>(params->signal), reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset + 16U)), 0);
    st_dev(static_cast<uint32_t>(params->signal >> 32U), reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset + 20U)), 0);
    st_dev(0U, reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset + 24U)), 0);
    st_dev(0U, reinterpret_cast<__gm__ uint32_t*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, baseOffset + 28U)), 0);
}

__aicore__ inline void UDMAFillSqeCtx(
    __gm__ UDMASqeCtx* sqeCtx, __gm__ uint8_t* remoteAddr, __gm__ UDMAMemInfo* remoteMemInfo,
    uint32_t curHead, UDMAOpcode opcode, uint32_t sqeFlag)
{
    sqeCtx->sqeBbIdx = curHead % TILEXR_UDMA_SQ_BB_COUNT;
    sqeCtx->opcode = static_cast<uint32_t>(opcode);
    sqeCtx->flag = sqeFlag;
    sqeCtx->rsv0 = 0;
    sqeCtx->nf = 0;
    sqeCtx->tokenEn = remoteMemInfo->tokenValueValid;
    sqeCtx->rmtJettyType = remoteMemInfo->rmtJettyType;
    sqeCtx->owner = ((curHead / TILEXR_UDMA_SQ_BB_COUNT) & 1U) == 0U ? 1U : 0U;
    sqeCtx->targetHint = remoteMemInfo->targetHint;
    sqeCtx->inlineMsgLen = 0;
    sqeCtx->rsv1 = 0;
    sqeCtx->tpId = remoteMemInfo->tpn;
    sqeCtx->sgeNum = 1;
    sqeCtx->rmtJettyOrSegId = remoteMemInfo->tid;
    sqeCtx->rsv2 = 0;
    sqeCtx->rmtTokenValue = remoteMemInfo->rmtTokenValue;
    sqeCtx->udfType = 0;
    sqeCtx->reduceDataType = 0;
    sqeCtx->reduceOpcode = 0;
    sqeCtx->rsv3 = 0;
    uint64_t remoteAddrValue = reinterpret_cast<uint64_t>(remoteAddr);
    sqeCtx->rmtAddrLOrTokenId = remoteAddrValue & 0xFFFFFFFF;
    sqeCtx->rmtAddrHOrTokenValue = (remoteAddrValue >> 32) & 0xFFFFFFFF;
    __gm__ uint64_t* rmtEid = reinterpret_cast<__gm__ uint64_t*>(remoteMemInfo->eidAddr);
    sqeCtx->rmtEidL = rmtEid[0];
    sqeCtx->rmtEidH = rmtEid[1];
}

__aicore__ inline void UDMAFillSgeCtx(
    __gm__ UDMASgeCtx* sgeCtx, uint64_t messageLen, __gm__ uint8_t* localAddr,
    uint32_t localTokenId)
{
    sgeCtx->len = messageLen;
    sgeCtx->tokenId = localTokenId;
    sgeCtx->va = reinterpret_cast<uint64_t>(localAddr);
}

__aicore__ inline void UDMARingDoorbell(uint32_t curHead, __gm__ UDMAWQCtx* qpCtxEntry)
{
    st_dev(curHead, reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->dbAddr), 0);
}

__aicore__ inline void UDMACleanWrappedWqe(
    __gm__ UDMAWQCtx* qpCtxEntry, uint32_t curHead, uint32_t wqeBbCnt)
{
    const uint32_t headIndex = curHead % qpCtxEntry->depth;
    const uint32_t firstBb = wqeBbCnt < qpCtxEntry->depth - headIndex
        ? wqeBbCnt : qpCtxEntry->depth - headIndex;
    const uint64_t wqeSize = 1ULL << qpCtxEntry->baseBkShift;
    UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(
        qpCtxEntry->bufAddr + wqeSize * headIndex), wqeSize * firstBb);
    if (firstBb != wqeBbCnt) {
        UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(qpCtxEntry->bufAddr),
            wqeSize * (wqeBbCnt - firstBb));
    }
}

__aicore__ inline uint32_t UDMAValidatePostSend(
    __gm__ UDMAInfo* udmaInfo, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t qpIdx, uint64_t messageLen, UDMAOpcode opcode, uint32_t sqeFlag,
    __gm__ UDMAWQCtx* qpCtxEntry)
{
    if (udmaInfo == nullptr || qpIdx >= udmaInfo->qpNum || udmaInfo->sqPtr == 0U ||
        udmaInfo->memPtr == 0U || remoteAddr == nullptr ||
        (localAddr == nullptr && messageLen != 0U) || messageLen > 0xFFFFFFFFULL ||
        (sqeFlag & TILEXR_UDMA_SQE_FLAG_COMPLETION) == 0U ||
        (opcode != UDMAOpcode::WRITE && opcode != UDMAOpcode::READ &&
         opcode != UDMAOpcode::WRITE_WITH_NOTIFY) ||
        qpCtxEntry == nullptr || qpCtxEntry->bufAddr == 0U || qpCtxEntry->headAddr == 0U ||
        qpCtxEntry->tailAddr == 0U || qpCtxEntry->wqeCntAddr == 0U || qpCtxEntry->dbAddr == 0U ||
        qpCtxEntry->depth != TILEXR_UDMA_SQ_BB_COUNT || qpCtxEntry->baseBkShift >= 32U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }

    const uint32_t wqeSize = 1U << qpCtxEntry->baseBkShift;
    const uint32_t requiredBb = UDMAWqeBBCnt(opcode);
    const uint32_t requiredBytes = sizeof(UDMASqeCtx) + sizeof(UDMASgeCtx) +
        (opcode == UDMAOpcode::WRITE_WITH_NOTIFY ? sizeof(UDMANotifyCtx) : 0U);
    if (wqeSize < sizeof(UDMASqeCtx) + sizeof(UDMASgeCtx) ||
        requiredBb > qpCtxEntry->depth ||
        static_cast<uint64_t>(wqeSize) * requiredBb < requiredBytes) {
        return TILEXR_UDMA_STATUS_INVALID;
    }

    const uint32_t head = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->headAddr), 0);
    const uint32_t tail = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->tailAddr), 0);
    const uint32_t outstanding = head - tail;
    if (outstanding > qpCtxEntry->depth) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    if (requiredBb > qpCtxEntry->depth - outstanding) {
        return TILEXR_UDMA_STATUS_SQ_FULL;
    }
    return TILEXR_UDMA_STATUS_SUCCESS;
}

__aicore__ inline uint32_t UDMAPostSend(
    __gm__ UDMAInfo* udmaInfo, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen, UDMAOpcode opcode,
    const UDMASignalParams* signalParams, uint32_t sqeFlag, bool ringDoorbell)
{
    if (udmaInfo == nullptr || udmaInfo->qpNum == 0U ||
        udmaInfo->qpNum > TILEXR_UDMA_DEVICE_MAX_QP_COUNT || qpIdx >= udmaInfo->qpNum || udmaInfo->sqPtr == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    __gm__ UDMAWQCtx* qpCtxEntry = UDMAGetWQCtx(udmaInfo, pe, qpIdx);
    uint32_t validation = UDMAValidatePostSend(
        udmaInfo, remoteAddr, localAddr, qpIdx, messageLen, opcode, sqeFlag, qpCtxEntry);
    if (validation == TILEXR_UDMA_STATUS_SQ_FULL && ringDoorbell) {
        const uint32_t submittedHead =
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->headAddr), 0);
        UDMARingDoorbell(submittedHead, qpCtxEntry);
        const uint32_t submittedWqeCount =
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->wqeCntAddr), 0);
        const uint32_t reclaimStatus = UDMAPollCQ(udmaInfo, pe, qpIdx, submittedWqeCount);
        if (reclaimStatus != TILEXR_UDMA_STATUS_SUCCESS) {
            return reclaimStatus;
        }
        validation = UDMAValidatePostSend(
            udmaInfo, remoteAddr, localAddr, qpIdx, messageLen, opcode, sqeFlag, qpCtxEntry);
    }
    if (validation != TILEXR_UDMA_STATUS_SUCCESS) {
        return validation;
    }

    __gm__ UDMAMemInfo* remoteMemInfo = UDMAGetRemoteMemInfo(udmaInfo, pe, qpIdx);
    if (remoteMemInfo->eidAddr == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    const uint32_t wqeSize = 1U << qpCtxEntry->baseBkShift;
    uint32_t curHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->headAddr), 0);
    uint32_t wqeCnt = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->wqeCntAddr), 0);
    __gm__ uint8_t* wqeAddr =
        reinterpret_cast<__gm__ uint8_t*>(qpCtxEntry->bufAddr + wqeSize * (curHead % TILEXR_UDMA_SQ_BB_COUNT));
    __gm__ UDMASqeCtx* sqeCtx = reinterpret_cast<__gm__ UDMASqeCtx*>(wqeAddr);
    UDMAFillSqeCtx(sqeCtx, remoteAddr, remoteMemInfo, curHead, opcode, sqeFlag);

    if (opcode == UDMAOpcode::WRITE_WITH_NOTIFY) {
        UDMAFillWrappedNotifyData(qpCtxEntry, curHead, remoteMemInfo->tid,
            remoteMemInfo->rmtTokenValue, signalParams);
    }

    const uint32_t sgeOffset = sizeof(UDMASqeCtx) +
        (opcode == UDMAOpcode::WRITE_WITH_NOTIFY ? sizeof(UDMANotifyCtx) : 0U);
    __gm__ UDMASgeCtx* sgeCtx = reinterpret_cast<__gm__ UDMASgeCtx*>(
        UDMAGetSqLogicalAddr(qpCtxEntry, curHead, sgeOffset));
    UDMAFillSgeCtx(sgeCtx, messageLen, localAddr, qpCtxEntry->localTokenId);
    const uint32_t wqeBbCnt = UDMAWqeBBCnt(opcode);
    UDMACleanWrappedWqe(qpCtxEntry, curHead, wqeBbCnt);
    curHead += wqeBbCnt;
    ++wqeCnt;
    st_dev(curHead, reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->headAddr), 0);
    st_dev(wqeCnt, reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->wqeCntAddr), 0);
    if (ringDoorbell) {
        UDMARingDoorbell(curHead, qpCtxEntry);
    }
    return TILEXR_UDMA_STATUS_SUCCESS;
}

__aicore__ inline uint32_t UDMAWrite(
    const __gm__ CommArgs* args, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    uint32_t sqeFlag = TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION, bool ringDoorbell = true)
{
    if (TILEXR_UDMA_ARCH_SUPPORTED && UDMAQueueOperationValid(
            args, static_cast<int>(pe), qpIdx)) {
        return UDMAPostSend(GetUDMAInfo(args), remoteAddr, localAddr, pe, qpIdx, messageLen,
            UDMAOpcode::WRITE, nullptr, sqeFlag, ringDoorbell);
    }
    return TILEXR_UDMA_STATUS_INVALID;
}

__aicore__ inline uint32_t UDMARead(
    const __gm__ CommArgs* args, __gm__ uint8_t* localAddr, __gm__ uint8_t* remoteAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen)
{
    if (TILEXR_UDMA_ARCH_SUPPORTED && UDMAQueueOperationValid(
            args, static_cast<int>(pe), qpIdx)) {
        return UDMAPostSend(GetUDMAInfo(args), remoteAddr, localAddr, pe, qpIdx, messageLen,
            UDMAOpcode::READ, nullptr, TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION, true);
    }
    return TILEXR_UDMA_STATUS_INVALID;
}

__aicore__ inline uint32_t UDMAWriteNotify(
    const __gm__ CommArgs* args, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen, const UDMASignalParams* signalParams)
{
    if (TILEXR_UDMA_ARCH_SUPPORTED && UDMAQueueOperationValid(
            args, static_cast<int>(pe), qpIdx)) {
        if (signalParams == nullptr || signalParams->sigAddr == nullptr) {
            return TILEXR_UDMA_STATUS_INVALID;
        }
        return UDMAPostSend(GetUDMAInfo(args), remoteAddr, localAddr, pe, qpIdx, messageLen,
            UDMAOpcode::WRITE_WITH_NOTIFY, signalParams,
            TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION, true);
    }
    return TILEXR_UDMA_STATUS_INVALID;
}

__aicore__ inline bool UDMARegisteredOperationValid(
    const __gm__ CommArgs* args, int peer, uint32_t qpIdx, uint64_t byteOffset, uint64_t byteCount)
{
    if (!UDMARegistryEnabled(args) || !UDMARankValid(args, peer) || !UDMAQpValid(args, qpIdx)) {
        return false;
    }
    const __gm__ TileXRUDMARegistry* registry = GetUDMARegistry(args);
    return registry->rankSize == static_cast<uint32_t>(args->rankSize) &&
        UDMARegisteredRangeValid(registry, peer, byteOffset, byteCount);
}

template <typename T>
__aicore__ inline uint32_t UDMAPutNbiOnQpWithFlagDeferred(
    const __gm__ CommArgs* args, int targetRank, uint32_t qpIdx,
    const __gm__ T* localSrc, uint64_t byteOffset, uint32_t byteCount, uint32_t sqeFlag)
{
    if (!UDMARegisteredOperationValid(args, targetRank, qpIdx, byteOffset, byteCount) ||
        (localSrc == nullptr && byteCount != 0U) ||
        (sqeFlag & TILEXR_UDMA_SQE_FLAG_COMPLETION) == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }

    auto registry = GetUDMARegistry(args);
    auto remoteAddr = UDMARegisteredRemoteAddr(registry, targetRank, byteOffset);
    return UDMAWrite(args, remoteAddr,
        reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(localSrc)),
        static_cast<uint32_t>(targetRank), qpIdx, byteCount, sqeFlag, false);
}

template <typename T>
__aicore__ inline uint32_t UDMAPutNbiOnQpWithFlag(
    const __gm__ CommArgs* args, int targetRank, uint32_t qpIdx,
    const __gm__ T* localSrc, uint64_t byteOffset, uint32_t byteCount, uint32_t sqeFlag)
{
    if (!UDMARegisteredOperationValid(args, targetRank, qpIdx, byteOffset, byteCount) ||
        (localSrc == nullptr && byteCount != 0U) ||
        (sqeFlag & TILEXR_UDMA_SQE_FLAG_COMPLETION) == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }

    auto registry = GetUDMARegistry(args);
    auto remoteAddr = UDMARegisteredRemoteAddr(registry, targetRank, byteOffset);
    return UDMAWrite(args, remoteAddr,
        reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(localSrc)),
        static_cast<uint32_t>(targetRank), qpIdx, byteCount, sqeFlag, true);
}

template <typename T>
__aicore__ inline uint32_t UDMAPutNbiOnQp(
    const __gm__ CommArgs* args, int targetRank, uint32_t qpIdx,
    const __gm__ T* localSrc, uint64_t byteOffset, uint32_t byteCount)
{
    return UDMAPutNbiOnQpWithFlag<T>(args, targetRank, qpIdx, localSrc,
        byteOffset, byteCount, TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
}

template <typename T>
__aicore__ inline void UDMAPutNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc,
    uint64_t byteOffset, uint32_t byteCount)
{
    (void)UDMAPutNbiOnQp<T>(args, targetRank, 0U, localSrc, byteOffset, byteCount);
}

template <typename T>
__aicore__ inline void UDMAPutRegisteredNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc, uint64_t byteOffset, uint32_t byteCount)
{
    UDMAPutNbi<T>(args, targetRank, localSrc, byteOffset, byteCount);
}

template <typename T>
__aicore__ inline uint32_t UDMAGetNbiOnQp(
    const __gm__ CommArgs* args, int sourceRank, uint32_t qpIdx,
    __gm__ T* localDst, uint64_t byteOffset, uint32_t byteCount)
{
    if (!UDMARegisteredOperationValid(args, sourceRank, qpIdx, byteOffset, byteCount) ||
        (localDst == nullptr && byteCount != 0U)) {
        return TILEXR_UDMA_STATUS_INVALID;
    }

    auto registry = GetUDMARegistry(args);
    auto remoteAddr = UDMARegisteredRemoteAddr(registry, sourceRank, byteOffset);
    return UDMARead(args, reinterpret_cast<__gm__ uint8_t*>(localDst), remoteAddr,
        static_cast<uint32_t>(sourceRank), qpIdx, byteCount);
}

template <typename T>
__aicore__ inline void UDMAGetNbi(
    const __gm__ CommArgs* args, int sourceRank, __gm__ T* localDst,
    uint64_t byteOffset, uint32_t byteCount)
{
    (void)UDMAGetNbiOnQp<T>(args, sourceRank, 0U, localDst, byteOffset, byteCount);
}

template <typename T>
__aicore__ inline void UDMAGetRegisteredNbi(
    const __gm__ CommArgs* args, int sourceRank, __gm__ T* localDst, uint64_t byteOffset, uint32_t byteCount)
{
    UDMAGetNbi<T>(args, sourceRank, localDst, byteOffset, byteCount);
}

template <typename T>
__aicore__ inline void UDMAPutSignalNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc, uint64_t byteOffset,
    uint32_t byteCount, uint64_t signalByteOffset, uint64_t signal)
{
    if (!UDMARegisteredOperationValid(args, targetRank, 0U, byteOffset, byteCount) ||
        !UDMARegisteredOperationValid(args, targetRank, 0U, signalByteOffset, sizeof(uint64_t)) ||
        (localSrc == nullptr && byteCount != 0U)) {
        return;
    }

    auto registry = GetUDMARegistry(args);
    UDMASignalParams signalParams = {};
    signalParams.sigAddr = reinterpret_cast<__gm__ uint64_t*>(
        UDMARegisteredRemoteAddr(registry, targetRank, signalByteOffset));
    signalParams.signal = signal;
    auto remoteAddr = UDMARegisteredRemoteAddr(registry, targetRank, byteOffset);
    (void)UDMAWriteNotify(args, remoteAddr,
        reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(localSrc)),
        static_cast<uint32_t>(targetRank), 0U, byteCount, &signalParams);
}

template <typename T>
__aicore__ inline void UDMAPutRegisteredSignalNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc, uint64_t byteOffset,
    uint32_t byteCount, uint64_t signalByteOffset, uint64_t signal)
{
    UDMAPutSignalNbi<T>(args, targetRank, localSrc, byteOffset, byteCount, signalByteOffset, signal);
}

__aicore__ inline uint32_t UDMAFlushQpDoorbell(
    const __gm__ CommArgs* args, int targetRank, uint32_t qpIdx)
{
    if (!TILEXR_UDMA_ARCH_SUPPORTED) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    if (!UDMAQueueOperationValid(args, targetRank, qpIdx)) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    __gm__ UDMAInfo* udmaInfo = GetUDMAInfo(args);
    if (udmaInfo->sqPtr == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    __gm__ UDMAWQCtx* qpCtxEntry =
        UDMAGetWQCtx(udmaInfo, static_cast<uint32_t>(targetRank), qpIdx);
    if (qpCtxEntry->headAddr == 0U || qpCtxEntry->tailAddr == 0U || qpCtxEntry->dbAddr == 0U ||
        qpCtxEntry->depth != TILEXR_UDMA_SQ_BB_COUNT) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    const uint32_t head = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->headAddr), 0);
    const uint32_t tail = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->tailAddr), 0);
    if (head - tail > qpCtxEntry->depth) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    UDMARingDoorbell(head, qpCtxEntry);
    return TILEXR_UDMA_STATUS_SUCCESS;
}

__aicore__ inline uint32_t UDMAQuietStatusOnQp(
    const __gm__ CommArgs* args, int targetRank, uint32_t qpIdx)
{
    if (!TILEXR_UDMA_ARCH_SUPPORTED) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    if (!UDMAQueueOperationValid(args, targetRank, qpIdx)) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    __gm__ UDMAInfo* udmaInfo = GetUDMAInfo(args);
    if (udmaInfo->sqPtr == 0U || udmaInfo->scqPtr == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    __gm__ UDMAWQCtx* qpCtxEntry =
        UDMAGetWQCtx(udmaInfo, static_cast<uint32_t>(targetRank), qpIdx);
    if (qpCtxEntry->wqeCntAddr == 0U) {
        return TILEXR_UDMA_STATUS_INVALID;
    }
    uint32_t wqeCnt = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->wqeCntAddr), 0);
    return UDMAPollCQ(udmaInfo, static_cast<uint32_t>(targetRank), qpIdx, wqeCnt);
}

__aicore__ inline uint32_t UDMAQuietStatus(const __gm__ CommArgs* args, int targetRank)
{
    return UDMAQuietStatusOnQp(args, targetRank, 0U);
}

__aicore__ inline void UDMAQuiet(const __gm__ CommArgs* args, int targetRank)
{
    (void)UDMAQuietStatus(args, targetRank);
}

} // namespace TileXR

#endif // TILEXR_UDMA_H
