/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_H
#define TILEXR_UDMA_H

#include "kernel_operator.h"
#include "comm_args.h"
#include "tilexr_udma_reg.h"

namespace TileXR {

/**
 * @file tilexr_udma.h
 * @brief Device-side UDMA wrapper for TileXR-registered memory.
 *
 * Host side registers ordinary device memory with TileXRUDMARegister. Device
 * kernels then use byte offsets into that registered region for PUT/GET/SIGNAL.
 * This header is self-contained for the device path and intentionally avoids
 * symmetric-memory APIs.
 */

constexpr uint32_t TILEXR_UDMA_CQ_DEPTH = 16384;
constexpr uint32_t TILEXR_UDMA_SQ_DEPTH = 4096;
constexpr uint32_t TILEXR_UDMA_MAX_SQE_BB_NUM = 4;
constexpr uint32_t TILEXR_UDMA_SQ_BB_COUNT = TILEXR_UDMA_SQ_DEPTH * TILEXR_UDMA_MAX_SQE_BB_NUM;
constexpr uint32_t TILEXR_UDMA_NUM_CQE_PER_POLL = 100;
constexpr uint32_t TILEXR_UDMA_CACHE_LINE_SIZE = 64;
constexpr uint32_t TILEXR_UDMA_MAX_RETRY_TIMES = 1000000;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr bool TILEXR_UDMA_ARCH_SUPPORTED = true;
#else
constexpr bool TILEXR_UDMA_ARCH_SUPPORTED = false;
#endif

enum class UDMAOpcode : uint32_t {
    WRITE = 3,
    WRITE_WITH_NOTIFY = 5,
    READ = 6,
};

enum class UDMADBMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB = 1,
};

struct UDMAMemInfo {
    bool tokenValueValid;
    uint32_t rmtJettyType : 2;
    uint8_t targetHint;
    uint32_t tpn;
    uint32_t tid;
    uint32_t rmtTokenValue;
    uint32_t len;
    uint64_t addr;
    uint64_t eidAddr;
};

struct UDMAWQCtx {
    uint32_t wqn;
    uint64_t bufAddr;
    uint32_t baseBkShift;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    UDMADBMode dbMode;
    uint64_t dbAddr;
    uint32_t sl;
    uint64_t wqeCntAddr;
    uint64_t amoAddr;
};

struct UDMACQCtx {
    uint32_t cqn;
    uint64_t bufAddr;
    uint32_t baseBkShift;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    UDMADBMode dbMode;
    uint64_t dbAddr;
};

struct UDMAInfo {
    uint32_t qpNum;
    uint64_t sqPtr;
    uint64_t rqPtr;
    uint64_t scqPtr;
    uint64_t rcqPtr;
    uint64_t memPtr;
};

struct UDMASqeCtx {
    uint32_t sqeBbIdx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t tokenEn : 1;
    uint32_t rmtJettyType : 2;
    uint32_t owner : 1;
    uint32_t targetHint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inlineMsgLen : 10;
    uint32_t tpId : 24;
    uint32_t sgeNum : 8;
    uint32_t rmtJettyOrSegId : 20;
    uint32_t rsv2 : 12;
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    uint32_t rmtTokenValue;
    uint32_t udfType : 8;
    uint32_t reduceDataType : 4;
    uint32_t reduceOpcode : 4;
    uint32_t rsv3 : 16;
    uint32_t rmtAddrLOrTokenId;
    uint32_t rmtAddrHOrTokenValue;
};

struct UDMASgeCtx {
    uint32_t len;
    uint32_t tokenId;
    uint64_t va;
};

struct UDMANotifyCtx {
    uint32_t notifyTokenId : 20;
    uint32_t rsv : 12;
    uint32_t notifyTokenValue;
    uint32_t notifyAddrL;
    uint32_t notifyAddrH;
    uint32_t notifyDataL;
    uint32_t notifyDataH;
    uint32_t rsv2[2];
};

struct UDMACqeCtx {
    uint32_t sR : 1;
    uint32_t isJetty : 1;
    uint32_t owner : 1;
    uint32_t inlineEn : 1;
    uint32_t opcode : 3;
    uint32_t fd : 1;
    uint32_t rsv : 8;
    uint32_t substatus : 8;
    uint32_t status : 8;
    uint32_t entryIdx : 16;
    uint32_t localNumL : 16;
    uint32_t localNumH : 4;
    uint32_t rmtIdx : 20;
    uint32_t rsv1 : 8;
    uint32_t tpn : 24;
    uint32_t rsv2 : 8;
    uint32_t byteCnt;
    uint32_t userDataL;
    uint32_t userDataH;
    uint32_t rmtEid[4];
    uint32_t dataL;
    uint32_t dataH;
    uint32_t inlineData[3];
};

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
    __gm__ uint8_t* start = reinterpret_cast<__gm__ uint8_t*>(
        reinterpret_cast<uint64_t>(addr) / TILEXR_UDMA_CACHE_LINE_SIZE * TILEXR_UDMA_CACHE_LINE_SIZE);
    __gm__ uint8_t* end = reinterpret_cast<__gm__ uint8_t*>(
        (reinterpret_cast<uint64_t>(addr) + length) / TILEXR_UDMA_CACHE_LINE_SIZE * TILEXR_UDMA_CACHE_LINE_SIZE);
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
    uint32_t qpNum = udmaInfo->qpNum;
    return reinterpret_cast<__gm__ UDMAWQCtx*>(udmaInfo->sqPtr + (pe * qpNum + qpIdx) * sizeof(UDMAWQCtx));
}

__aicore__ inline __gm__ UDMACQCtx* UDMAGetSCQCtx(__gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx)
{
    uint32_t qpNum = udmaInfo->qpNum;
    return reinterpret_cast<__gm__ UDMACQCtx*>(udmaInfo->scqPtr + (pe * qpNum + qpIdx) * sizeof(UDMACQCtx));
}

__aicore__ inline __gm__ UDMAMemInfo* UDMAGetRemoteMemInfo(__gm__ UDMAInfo* udmaInfo, uint32_t pe)
{
    return reinterpret_cast<__gm__ UDMAMemInfo*>(udmaInfo->memPtr + sizeof(UDMAMemInfo) * pe);
}

__aicore__ inline void UDMAPollCQUpdateInfo(
    uint32_t curTail, __gm__ UDMACQCtx* cqCtxEntry, __gm__ UDMAWQCtx* wqCtxEntry)
{
    st_dev(static_cast<uint32_t>(curTail & 0xFFFFFF), reinterpret_cast<__gm__ uint32_t*>(cqCtxEntry->dbAddr), 0);
    st_dev(curTail, reinterpret_cast<__gm__ uint32_t*>(wqCtxEntry->tailAddr), 0);
}

__aicore__ inline uint32_t UDMAPollCQ(__gm__ UDMAInfo* udmaInfo, uint32_t pe, uint32_t qpIdx, uint32_t idx)
{
    if (idx == 0) {
        return 0;
    }
    __gm__ UDMACQCtx* cqCtxEntry = UDMAGetSCQCtx(udmaInfo, pe, qpIdx);
    __gm__ UDMAWQCtx* wqCtxEntry = UDMAGetWQCtx(udmaInfo, pe, qpIdx);
    uint64_t cqBaseAddr = cqCtxEntry->bufAddr;
    uint32_t cqeSize = 1U << cqCtxEntry->baseBkShift;
    uint32_t curTail = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cqCtxEntry->tailAddr), 0);
    while (curTail != idx) {
        __gm__ UDMACqeCtx* cqeAddr = reinterpret_cast<__gm__ UDMACqeCtx*>(
            cqBaseAddr + cqeSize * (curTail & (TILEXR_UDMA_CQ_DEPTH - 1)));
        bool validOwner = ((curTail / TILEXR_UDMA_CQ_DEPTH) & 1) != 0;
        uint32_t times = 0;
        while ((validOwner ^ (cqeAddr->owner != 0)) == 0 && times < TILEXR_UDMA_MAX_RETRY_TIMES) {
            UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(cqeAddr), sizeof(UDMACqeCtx));
            ++times;
        }
        if (times >= TILEXR_UDMA_MAX_RETRY_TIMES) {
            return 0xFF;
        }
        uint8_t status = cqeAddr->status & 0xFF;
        uint8_t subStatus = cqeAddr->substatus & 0xFF;
        if (status != 0 || subStatus != 0) {
            return (static_cast<uint32_t>(status) << 8) | subStatus;
        }
        ++curTail;
    }
    st_dev(curTail, reinterpret_cast<__gm__ uint32_t*>(cqCtxEntry->tailAddr), 0);
    UDMAPollCQUpdateInfo(curTail, cqCtxEntry, wqCtxEntry);
    return 0;
}

__aicore__ inline uint32_t UDMAWqeBBCnt(UDMAOpcode opcode)
{
    return opcode == UDMAOpcode::WRITE_WITH_NOTIFY ? 2U : 1U;
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
    notifyCtx->notifyTokenValue = tokenValue;
    notifyCtx->notifyAddrL = reinterpret_cast<uint64_t>(params->sigAddr) & 0xFFFFFFFF;
    notifyCtx->notifyAddrH = (reinterpret_cast<uint64_t>(params->sigAddr) >> 32) & 0xFFFFFFFF;
    notifyCtx->notifyDataL = params->signal & 0xFFFFFFFF;
    notifyCtx->notifyDataH = (params->signal >> 32) & 0xFFFFFFFF;
}

__aicore__ inline void UDMAFillSqeCtx(
    __gm__ UDMASqeCtx* sqeCtx, __gm__ uint8_t* remoteAddr, __gm__ UDMAMemInfo* remoteMemInfo,
    uint32_t curHead, UDMAOpcode opcode, const UDMASignalParams* signalParams)
{
    sqeCtx->opcode = static_cast<uint32_t>(opcode);
    sqeCtx->flag = 0b00100010;
    sqeCtx->nf = 0;
    sqeCtx->tokenEn = remoteMemInfo->tokenValueValid;
    sqeCtx->rmtJettyType = remoteMemInfo->rmtJettyType;
    sqeCtx->owner = (curHead & TILEXR_UDMA_SQ_BB_COUNT) == 0 ? 1 : 0;
    sqeCtx->targetHint = remoteMemInfo->targetHint;
    sqeCtx->inlineMsgLen = 0;
    sqeCtx->tpId = remoteMemInfo->tpn;
    sqeCtx->sgeNum = 1;
    sqeCtx->rmtJettyOrSegId = remoteMemInfo->tid;
    sqeCtx->rmtTokenValue = remoteMemInfo->rmtTokenValue;
    sqeCtx->udfType = 0;
    sqeCtx->reduceDataType = 0;
    sqeCtx->reduceOpcode = 0;
    uint64_t remoteAddrValue = reinterpret_cast<uint64_t>(remoteAddr);
    sqeCtx->rmtAddrLOrTokenId = remoteAddrValue & 0xFFFFFFFF;
    sqeCtx->rmtAddrHOrTokenValue = (remoteAddrValue >> 32) & 0xFFFFFFFF;
    __gm__ uint64_t* rmtEid = reinterpret_cast<__gm__ uint64_t*>(remoteMemInfo->eidAddr);
    sqeCtx->rmtEidL = rmtEid[0];
    sqeCtx->rmtEidH = rmtEid[1];
    UDMAFillNotifyData(sqeCtx, remoteMemInfo->tid, remoteMemInfo->rmtTokenValue, signalParams);
}

__aicore__ inline void UDMAFillSgeCtx(
    __gm__ UDMASgeCtx* sgeCtx, uint64_t messageLen, __gm__ uint8_t* localAddr)
{
    sgeCtx->len = messageLen;
    sgeCtx->tokenId = 0;
    sgeCtx->va = reinterpret_cast<uint64_t>(localAddr);
}

__aicore__ inline void UDMAPollCQWhenSQOverflow(
    __gm__ UDMAInfo* udmaInfo, __gm__ UDMAWQCtx* qpCtxEntry, uint32_t wqeCnt, uint32_t pe, uint32_t qpIdx)
{
    constexpr uint32_t pollCQThreshold = 10;
    uint32_t curTail = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->tailAddr), 0);
    if ((wqeCnt + pollCQThreshold) % TILEXR_UDMA_SQ_BB_COUNT == curTail % TILEXR_UDMA_SQ_BB_COUNT) {
        uint32_t idx = (curTail + TILEXR_UDMA_NUM_CQE_PER_POLL) > wqeCnt ?
            wqeCnt : curTail + TILEXR_UDMA_NUM_CQE_PER_POLL;
        (void)UDMAPollCQ(udmaInfo, pe, qpIdx, idx);
    }
}

__aicore__ inline void UDMAPostSendUpdateInfo(uint32_t curHead, __gm__ UDMAWQCtx* qpCtxEntry)
{
    st_dev(curHead, reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->dbAddr), 0);
    st_dev(curHead, reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->headAddr), 0);
}

__aicore__ inline void UDMAPostSend(
    __gm__ UDMAInfo* udmaInfo, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen, UDMAOpcode opcode, const UDMASignalParams* signalParams)
{
    __gm__ UDMAWQCtx* qpCtxEntry = UDMAGetWQCtx(udmaInfo, pe, qpIdx);
    uint32_t wqeSize = 1U << qpCtxEntry->baseBkShift;
    uint32_t curHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->headAddr), 0);
    uint32_t wqeCnt = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->wqeCntAddr), 0);
    UDMAPollCQWhenSQOverflow(udmaInfo, qpCtxEntry, wqeCnt, pe, qpIdx);

    __gm__ UDMAMemInfo* remoteMemInfo = UDMAGetRemoteMemInfo(udmaInfo, pe);
    __gm__ uint8_t* wqeAddr =
        reinterpret_cast<__gm__ uint8_t*>(qpCtxEntry->bufAddr + wqeSize * (curHead % TILEXR_UDMA_SQ_BB_COUNT));
    __gm__ UDMASqeCtx* sqeCtx = reinterpret_cast<__gm__ UDMASqeCtx*>(wqeAddr);
    UDMAFillSqeCtx(sqeCtx, remoteAddr, remoteMemInfo, curHead, opcode, signalParams);

    __gm__ UDMASgeCtx* sgeCtx = reinterpret_cast<__gm__ UDMASgeCtx*>(UDMAGetSgeCtxAddr(wqeAddr, opcode));
    UDMAFillSgeCtx(sgeCtx, messageLen, localAddr);
    uint32_t wqeBbCnt = UDMAWqeBBCnt(opcode);
    UDMACleanCacheLines(wqeAddr, wqeSize * wqeBbCnt);
    curHead += wqeBbCnt;
    UDMAPostSendUpdateInfo(curHead, qpCtxEntry);
    ++wqeCnt;
    st_dev(wqeCnt, reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->wqeCntAddr), 0);
}

__aicore__ inline void UDMAWrite(
    const __gm__ CommArgs* args, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen)
{
    if constexpr (TILEXR_UDMA_ARCH_SUPPORTED) {
        UDMAPostSend(GetUDMAInfo(args), remoteAddr, localAddr, pe, qpIdx, messageLen, UDMAOpcode::WRITE, nullptr);
    }
}

__aicore__ inline void UDMARead(
    const __gm__ CommArgs* args, __gm__ uint8_t* localAddr, __gm__ uint8_t* remoteAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen)
{
    if constexpr (TILEXR_UDMA_ARCH_SUPPORTED) {
        UDMAPostSend(GetUDMAInfo(args), remoteAddr, localAddr, pe, qpIdx, messageLen, UDMAOpcode::READ, nullptr);
    }
}

__aicore__ inline void UDMAWriteNotify(
    const __gm__ CommArgs* args, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen, const UDMASignalParams* signalParams)
{
    if constexpr (TILEXR_UDMA_ARCH_SUPPORTED) {
        UDMAPostSend(GetUDMAInfo(args), remoteAddr, localAddr, pe, qpIdx, messageLen,
                     UDMAOpcode::WRITE_WITH_NOTIFY, signalParams);
    }
}

template <typename T>
__aicore__ inline void UDMAPutNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc, uint64_t byteOffset, uint32_t byteCount)
{
    if (!UDMARegistryEnabled(args)) return;

    auto registry = GetUDMARegistry(args);
    if (!UDMARegisteredRangeValid(registry, targetRank, byteOffset, byteCount)) return;

    auto remoteAddr = UDMARegisteredRemoteAddr(registry, targetRank, byteOffset);
    UDMAWrite(args, remoteAddr, reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(localSrc)),
              targetRank, 0, byteCount);
}

template <typename T>
__aicore__ inline void UDMAPutRegisteredNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc, uint64_t byteOffset, uint32_t byteCount)
{
    UDMAPutNbi<T>(args, targetRank, localSrc, byteOffset, byteCount);
}

template <typename T>
__aicore__ inline void UDMAGetNbi(
    const __gm__ CommArgs* args, int sourceRank, __gm__ T* localDst, uint64_t byteOffset, uint32_t byteCount)
{
    if (!UDMARegistryEnabled(args)) return;

    auto registry = GetUDMARegistry(args);
    if (!UDMARegisteredRangeValid(registry, sourceRank, byteOffset, byteCount)) return;

    auto remoteAddr = UDMARegisteredRemoteAddr(registry, sourceRank, byteOffset);
    UDMARead(args, reinterpret_cast<__gm__ uint8_t*>(localDst), remoteAddr, sourceRank, 0, byteCount);
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
    if (!UDMARegistryEnabled(args)) return;

    auto registry = GetUDMARegistry(args);
    if (!UDMARegisteredRangeValid(registry, targetRank, byteOffset, byteCount) ||
        !UDMARegisteredRangeValid(registry, targetRank, signalByteOffset, sizeof(uint64_t))) {
        return;
    }

    UDMASignalParams signalParams = {};
    signalParams.sigAddr = reinterpret_cast<__gm__ uint64_t*>(
        UDMARegisteredRemoteAddr(registry, targetRank, signalByteOffset));
    signalParams.signal = signal;
    auto remoteAddr = UDMARegisteredRemoteAddr(registry, targetRank, byteOffset);
    UDMAWriteNotify(args, remoteAddr, reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(localSrc)),
                    targetRank, 0, byteCount, &signalParams);
}

template <typename T>
__aicore__ inline void UDMAPutRegisteredSignalNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc, uint64_t byteOffset,
    uint32_t byteCount, uint64_t signalByteOffset, uint64_t signal)
{
    UDMAPutSignalNbi<T>(args, targetRank, localSrc, byteOffset, byteCount, signalByteOffset, signal);
}

__aicore__ inline void UDMAQuiet(const __gm__ CommArgs* args, int targetRank)
{
    if (!UDMAEnabled(args)) return;
    __gm__ UDMAInfo* udmaInfo = GetUDMAInfo(args);
    __gm__ UDMAWQCtx* qpCtxEntry = UDMAGetWQCtx(udmaInfo, targetRank, 0);
    uint32_t wqeCnt = ld_dev(reinterpret_cast<__gm__ uint32_t*>(qpCtxEntry->wqeCntAddr), 0);
    (void)UDMAPollCQ(udmaInfo, targetRank, 0, wqeCnt);
}

} // namespace TileXR

#endif // TILEXR_UDMA_H
