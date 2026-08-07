/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <cstdint>

#include "comm_args.h"
#include "../common/dispatch_common.h"
#include "../common/dispatch_credit.h"
#include "../common/dispatch_profile.h"
#include "../common/dispatch_schedule.h"
#include "../common/dispatch_wqe_batch.h"
#include "kernel_operator.h"
#include "tilexr_udma.h"

namespace {

constexpr uint32_t kUbAlignBytes = 32U;
constexpr uint32_t kFullUbBytes = 190U * 1024U;
constexpr uint32_t kRelayUbBytes = 4U * 1024U;
constexpr uint32_t kDiagnosticUbBytes = 4U * 1024U;
constexpr uint32_t kRouteTileElements = 1024U;
constexpr uint32_t kRouteTileBytes = kRouteTileElements * sizeof(int32_t);
constexpr uint32_t kVectorCompareMinElements = 256U / sizeof(int32_t);
constexpr uint32_t kMaxPipelineBufferNum = 8U;
constexpr uint32_t kOutputCopyBufferNum = 2U;
constexpr uint64_t kDeferredReclaimWqes =
    TileXR::TILEXR_UDMA_SQ_BB_COUNT / 4U;
constexpr uint32_t kDispatchUdmaWqeBytes = 64U;
constexpr uint32_t kDispatchWqeBuildThreads = 64U;
constexpr uint32_t kDispatchSqPollReserve = 10U;
constexpr uint32_t kDispatchPreparedPeerCapacity = 16U;
constexpr uint32_t kDispatchCqePollBatchCapacity =
    kRelayUbBytes / static_cast<uint32_t>(sizeof(TileXR::UDMACqeCtx));

constexpr uint32_t kDispatchWqeBatchBytes =
    TileXRMoonEp::kDispatchWqeBatchCapacity * kDispatchUdmaWqeBytes;

struct alignas(32) DispatchWqeBatchContext {
    uint64_t localSourceBase;
    uint64_t remoteScratchBase;
    uint64_t rowBytes;
    uint64_t routeCountMask;
    uint64_t signalLocalAddr;
    uint64_t signalRemoteAddr;
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    uint32_t batchHead;
    uint32_t batchOutputOffset;
    uint32_t tokenCount;
    uint32_t appendSignal;
    uint32_t topKMagic;
    uint32_t topKShift;
    uint32_t hiddenMode;
    uint32_t tokenEn;
    uint32_t rmtJettyType;
    uint32_t targetHint;
    uint32_t tpId;
    uint32_t rmtJettyOrSegId;
    uint32_t rmtTokenValue;
    uint32_t selectedStart;
    uint32_t qpSelection;
};

constexpr uint32_t kDispatchWqeBatchContextBytes =
    static_cast<uint32_t>(sizeof(DispatchWqeBatchContext));
constexpr uint32_t kDispatchWqeBatchContextOffset = kDispatchWqeBatchBytes;
constexpr uint32_t kDispatchUdmaIssueUbBytes =
    kDispatchWqeBatchBytes + kDispatchWqeBatchContextBytes;

static_assert(sizeof(TileXR::UDMASqeCtx) + sizeof(TileXR::UDMASgeCtx) ==
    kDispatchUdmaWqeBytes, "MoonEP Dispatch WRITE WQE must occupy one basic block");
static_assert(sizeof(TileXR::UDMACqeCtx) == 64U,
    "MoonEP Dispatch UDMA CQE must occupy one cache line");
static_assert(kDispatchWqeBatchBytes == 8192U,
    "MoonEP Dispatch WQE batch must occupy 8 KiB of UB");
static_assert(sizeof(DispatchWqeBatchContext) == 128U,
    "MoonEP Dispatch WQE batch context ABI changed");
static_assert(kDispatchUdmaIssueUbBytes % kUbAlignBytes == 0U,
    "MoonEP Dispatch issue UB must be 32-byte aligned");
static_assert(kDispatchCqePollBatchCapacity > 0U,
    "MoonEP Dispatch CQ poll relay must hold at least one CQE");

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
__simt_vf__ __aicore__ LAUNCH_BOUND(kDispatchWqeBuildThreads)
inline void DispatchBuildWriteWqeBatchVf(__ubuf__ uint8_t *wqeBytes,
    __ubuf__ const int16_t *selectedRouteIndices,
    __ubuf__ const int32_t *dstValues,
    __ubuf__ const DispatchWqeBatchContext *context)
{
    const uint32_t taskCount = context->tokenCount + context->appendSignal;
    for (uint32_t task = static_cast<uint32_t>(threadIdx.x);
        task < taskCount; task += kDispatchWqeBuildThreads) {
        const uint32_t outputIndex = context->batchOutputOffset + task;
        __ubuf__ uint8_t *wqe = wqeBytes +
            outputIndex * kDispatchUdmaWqeBytes;
        __ubuf__ uint32_t *words =
            reinterpret_cast<__ubuf__ uint32_t *>(wqe);
        for (uint32_t word = 0U;
            word < kDispatchUdmaWqeBytes / sizeof(uint32_t); ++word) {
            words[word] = 0U;
        }

        uint64_t localAddr = context->signalLocalAddr;
        uint64_t remoteAddr = context->signalRemoteAddr;
        uint32_t sqeFlag =
            TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION;
        if (task < context->tokenCount) {
            const uint32_t qpIdx = context->qpSelection >> 2U;
            const uint32_t sequencePhase = context->qpSelection & 3U;
            const uint32_t selectedIndex =
                TileXRMoonEp::DispatchQpSelectedIndex(
                    context->selectedStart + task, sequencePhase, qpIdx);
            const uint32_t routeId = static_cast<uint16_t>(
                selectedRouteIndices[selectedIndex]);
            const uint64_t targetSlot = static_cast<uint64_t>(
                static_cast<uint32_t>(dstValues[routeId])) &
                context->routeCountMask;
            const uint32_t sourceRow = context->hiddenMode != 0U ?
                AscendC::Simt::UintDiv(routeId, context->topKMagic,
                    context->topKShift) : routeId;
            localAddr = context->localSourceBase +
                static_cast<uint64_t>(sourceRow) * context->rowBytes;
            remoteAddr = context->remoteScratchBase +
                targetSlot * context->rowBytes;
            sqeFlag = 0U;
        }

        __ubuf__ TileXR::UDMASqeCtx *sqe =
            reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
        sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
        sqe->flag = sqeFlag;
        sqe->nf = 0U;
        sqe->tokenEn = context->tokenEn;
        sqe->rmtJettyType = context->rmtJettyType;
        const uint32_t wqeHead = context->batchHead + outputIndex;
        sqe->sqeBbIdx = static_cast<uint16_t>(
            wqeHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT);
        sqe->owner =
            (wqeHead & TileXR::TILEXR_UDMA_SQ_BB_COUNT) == 0U ? 1U : 0U;
        sqe->targetHint = context->targetHint;
        sqe->inlineMsgLen = 0U;
        sqe->tpId = context->tpId;
        sqe->sgeNum = 1U;
        sqe->rmtJettyOrSegId = context->rmtJettyOrSegId;
        sqe->rmtTokenValue = context->rmtTokenValue;
        sqe->rmtEidL = context->rmtEidL;
        sqe->rmtEidH = context->rmtEidH;
        sqe->rmtAddrLOrTokenId = remoteAddr & 0xFFFFFFFFU;
        sqe->rmtAddrHOrTokenValue =
            (remoteAddr >> 32U) & 0xFFFFFFFFU;

        __ubuf__ TileXR::UDMASgeCtx *sge =
            reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
                wqe + sizeof(TileXR::UDMASqeCtx));
        sge->len = task < context->tokenCount ?
            static_cast<uint32_t>(context->rowBytes) :
            static_cast<uint32_t>(sizeof(uint64_t));
        sge->tokenId = 0U;
        sge->va = localAddr;
    }
}
#endif

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    const AscendC::TEventID eventId = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

__aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

__aicore__ inline uint64_t MultiplyU32ToU64(uint32_t lhs, uint32_t rhs)
{
    constexpr uint32_t kHalfMask = UINT16_MAX;
    const uint32_t lhsLow = lhs & kHalfMask;
    const uint32_t lhsHigh = lhs >> 16U;
    const uint32_t rhsLow = rhs & kHalfMask;
    const uint32_t rhsHigh = rhs >> 16U;
    return static_cast<uint64_t>(lhsLow * rhsLow) +
        (static_cast<uint64_t>(lhsLow * rhsHigh) << 16U) +
        (static_cast<uint64_t>(lhsHigh * rhsLow) << 16U) +
        (static_cast<uint64_t>(lhsHigh * rhsHigh) << 32U);
}

__aicore__ inline uint32_t CalcPipelineBufferNum(
    uint64_t fixedBytes, uint32_t perBufferBytes)
{
    if (fixedBytes >= kFullUbBytes || perBufferBytes == 0U) {
        return 0U;
    }
    const uint64_t count = (kFullUbBytes - fixedBytes) / perBufferBytes;
    return static_cast<uint32_t>(count < kMaxPipelineBufferNum ? count :
        kMaxPipelineBufferNum);
}

__aicore__ inline bool ShouldFlushPartialDoorbell(
    uint64_t issuedWqes, uint64_t totalWqes)
{
    if (issuedWqes == 0U || issuedWqes >= totalWqes) {
        return false;
    }
    return issuedWqes == CeilDiv(totalWqes, 4U) ||
        issuedWqes == CeilDiv(totalWqes, 2U) ||
        issuedWqes == totalWqes - totalWqes / 4U;
}

__aicore__ inline void ReclaimDeferredSegment(const __gm__ TileXR::CommArgs *args,
    int32_t peer, uint32_t qpIdx, uint64_t issuedPeerWqes, uint32_t phase,
    uint32_t &dfxFlags, uint32_t &firstQuietStatus,
    uint32_t &firstQuietPhase)
{
    if (issuedPeerWqes == 0U || issuedPeerWqes % kDeferredReclaimWqes != 0U) {
        return;
    }
    TileXR::UDMAFlushQpDoorbell(args, peer, qpIdx);
    const uint32_t quietStatus = TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
    if (quietStatus != 0U && firstQuietStatus == 0U) {
        firstQuietStatus = quietStatus;
        firstQuietPhase = phase;
        dfxFlags |= TileXRMoonEp::kDispatchDfxQuietError;
    }
}

struct DispatchWqeBatchInitContext {
    const __gm__ TileXR::CommArgs *args;
    __gm__ TileXR::UDMAInfo *udmaInfo;
    __gm__ TileXR::TileXRUDMARegistry *registry;
    uint64_t remoteScratchOffset;
    uint64_t scratchBytes;
    uint64_t remoteFlagBase;
};

struct DispatchWqeBatchState {
    const __gm__ TileXR::CommArgs *args;
    __gm__ TileXR::UDMAWQCtx *qpCtxEntry;
    __gm__ TileXR::UDMACQCtx *cqCtxEntry;
    __gm__ uint8_t *remoteScratchBase;
    __gm__ uint8_t *remoteSignalAddr;
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    int32_t targetRank;
    uint32_t qpIdx;
    uint32_t head;
    uint32_t wqeCount;
    uint32_t tail;
    uint32_t cqTail;
    uint32_t outstanding;
    uint32_t batchCount;
    uint32_t batchLimit;
    uint32_t tokenEn;
    uint32_t rmtJettyType;
    uint32_t targetHint;
    uint32_t tpId;
    uint32_t rmtJettyOrSegId;
    uint32_t rmtTokenValue;
    uint32_t stagedDoorbellHead;
    uint32_t finalWqeCount;
    uint32_t doorbellPending;
    uint32_t finalStaged;
    uint32_t doorbellRung;
};

struct DispatchPreparedPeer {
    int32_t targetRank;
    uint32_t issuePhase;
    bool initialized;
    DispatchWqeBatchState qpState[TileXRMoonEp::kDispatchQpCount];
};

__aicore__ inline bool InitDispatchWqeBatchInitContext(
    const __gm__ TileXR::CommArgs *args, uint64_t remoteScratchOffset,
    uint64_t scratchBytes, uint64_t remoteFlagBase,
    DispatchWqeBatchInitContext &context)
{
    if (!TileXR::UDMARegistryEnabled(args)) {
        return false;
    }
    __gm__ TileXR::UDMAInfo *udmaInfo = TileXR::GetUDMAInfo(args);
    if (udmaInfo == nullptr ||
        udmaInfo->qpNum != TileXRMoonEp::kDispatchQpCount) {
        return false;
    }
    __gm__ TileXR::TileXRUDMARegistry *registry = TileXR::GetUDMARegistry(args);
    if (registry == nullptr) {
        return false;
    }
    context.args = args;
    context.udmaInfo = udmaInfo;
    context.registry = registry;
    context.remoteScratchOffset = remoteScratchOffset;
    context.scratchBytes = scratchBytes;
    context.remoteFlagBase = remoteFlagBase;
    return true;
}

__aicore__ inline bool InitDispatchWqeBatchState(
    const DispatchWqeBatchInitContext &context, int32_t targetRank,
    uint32_t qpIdx, __gm__ uint8_t *remoteScratchBase,
    __gm__ uint8_t *remoteSignalAddr, DispatchWqeBatchState &state)
{
    if (qpIdx >= TileXRMoonEp::kDispatchQpCount) {
        return false;
    }

    __gm__ TileXR::UDMAWQCtx *qpCtxEntry = TileXR::UDMAGetWQCtx(
        context.udmaInfo, static_cast<uint32_t>(targetRank), qpIdx);
    __gm__ TileXR::UDMACQCtx *cqCtxEntry = TileXR::UDMAGetSCQCtx(
        context.udmaInfo, static_cast<uint32_t>(targetRank), qpIdx);
    if (qpCtxEntry == nullptr || qpCtxEntry->baseBkShift >= 32U ||
        (1U << qpCtxEntry->baseBkShift) != kDispatchUdmaWqeBytes ||
        qpCtxEntry->depth != TileXR::TILEXR_UDMA_SQ_BB_COUNT ||
        cqCtxEntry == nullptr || cqCtxEntry->baseBkShift >= 32U ||
        (1U << cqCtxEntry->baseBkShift) != sizeof(TileXR::UDMACqeCtx) ||
        cqCtxEntry->depth != TileXR::TILEXR_UDMA_CQ_DEPTH) {
        return false;
    }
    __gm__ TileXR::UDMAMemInfo *remoteMemInfo = TileXR::UDMAGetRemoteMemInfo(
        context.udmaInfo, static_cast<uint32_t>(targetRank), qpIdx);
    if (remoteMemInfo == nullptr || remoteMemInfo->eidAddr == 0U) {
        return false;
    }
    __gm__ uint64_t *remoteEid = reinterpret_cast<__gm__ uint64_t *>(
        remoteMemInfo->eidAddr);

    state.args = context.args;
    state.qpCtxEntry = qpCtxEntry;
    state.cqCtxEntry = cqCtxEntry;
    state.remoteScratchBase = remoteScratchBase;
    state.remoteSignalAddr = remoteSignalAddr;
    state.rmtEidL = remoteEid[0];
    state.rmtEidH = remoteEid[1];
    state.targetRank = targetRank;
    state.qpIdx = qpIdx;
    state.head = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        qpCtxEntry->headAddr), 0);
    state.wqeCount = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        qpCtxEntry->wqeCntAddr), 0);
    state.tail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        qpCtxEntry->tailAddr), 0);
    state.cqTail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        cqCtxEntry->tailAddr), 0);
    state.outstanding = state.wqeCount - state.tail;
    state.batchCount = 0U;
    state.batchLimit = TileXRMoonEp::DispatchWqeBatchCount(
        UINT64_MAX, state.head, TileXR::TILEXR_UDMA_SQ_BB_COUNT);
    state.tokenEn = remoteMemInfo->tokenValueValid;
    state.rmtJettyType = remoteMemInfo->rmtJettyType;
    state.targetHint = remoteMemInfo->targetHint;
    state.tpId = remoteMemInfo->tpn;
    state.rmtJettyOrSegId = remoteMemInfo->tid;
    state.rmtTokenValue = remoteMemInfo->rmtTokenValue;
    state.stagedDoorbellHead = state.head;
    state.finalWqeCount = state.wqeCount;
    state.doorbellPending = 0U;
    state.finalStaged = 0U;
    state.doorbellRung = 0U;
    return state.batchLimit != 0U &&
        state.outstanding < TileXR::TILEXR_UDMA_SQ_BB_COUNT;
}

__aicore__ inline bool InitDispatchPreparedPeer(
    const DispatchWqeBatchInitContext &context, int32_t targetRank,
    uint32_t issuePhase, DispatchPreparedPeer &preparedPeer)
{
    const uint64_t remoteFlagBytes =
        TileXRMoonEp::kDispatchQpCount * sizeof(uint64_t);
    preparedPeer.targetRank = targetRank;
    preparedPeer.issuePhase = issuePhase;
    preparedPeer.initialized = false;
    if (!TileXR::UDMARegisteredRangeValid(context.registry, targetRank,
            context.remoteScratchOffset, context.scratchBytes) ||
        !TileXR::UDMARegisteredRangeValid(context.registry, targetRank,
            context.remoteFlagBase, remoteFlagBytes)) {
        return false;
    }
    __gm__ uint8_t *remoteScratchBase = TileXR::UDMARegisteredRemoteAddr(
        context.registry, targetRank, context.remoteScratchOffset);
    __gm__ uint8_t *remoteFlagBase = TileXR::UDMARegisteredRemoteAddr(
        context.registry, targetRank, context.remoteFlagBase);
    for (uint32_t qpIdx = 0U;
        qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
        if (!InitDispatchWqeBatchState(context, targetRank, qpIdx,
                remoteScratchBase,
                remoteFlagBase + qpIdx * sizeof(uint64_t),
                preparedPeer.qpState[qpIdx])) {
            return false;
        }
    }
    preparedPeer.initialized = true;
    return true;
}

__aicore__ inline void PrepareDispatchRemotePeerBatch(
    const DispatchWqeBatchInitContext *context, int32_t rank,
    int32_t rankSize, int64_t groupCount, uint32_t peerWorkCount,
    uint32_t blockIdx, uint32_t blockNum, uint64_t &peerCursor,
    DispatchPreparedPeer *peerBatch, uint32_t &peerCount,
    uint64_t &visitedPeerCount)
{
    peerCount = 0U;
    const uint64_t totalPeerAssignments =
        static_cast<uint64_t>(groupCount) * peerWorkCount;
    while (peerCursor < totalPeerAssignments &&
        peerCount < kDispatchPreparedPeerCapacity) {
        const uint64_t issuePhase = peerCursor / peerWorkCount;
        const uint32_t peerWork = static_cast<uint32_t>(
            peerCursor % peerWorkCount);
        ++peerCursor;
        const int64_t peer = TileXRMoonEp::DispatchPeerForCore(
            rank, rankSize, static_cast<int64_t>(issuePhase), blockIdx,
            blockNum, peerWork);
        if (peer < 0) {
            continue;
        }
        ++visitedPeerCount;
        if (peer == rank) {
            continue;
        }
        DispatchPreparedPeer &preparedPeer = peerBatch[peerCount];
        preparedPeer.targetRank = static_cast<int32_t>(peer);
        preparedPeer.issuePhase = static_cast<uint32_t>(issuePhase);
        preparedPeer.initialized = context != nullptr &&
            InitDispatchPreparedPeer(*context, static_cast<int32_t>(peer),
                static_cast<uint32_t>(issuePhase), preparedPeer);
        ++peerCount;
    }
}

__aicore__ inline uint32_t SelectDispatchPeerRoutes(
    AscendC::LocalTensor<uint8_t> compareMaskLocal,
    AscendC::LocalTensor<int32_t> routeRankLocal,
    AscendC::LocalTensor<int16_t> routeIndexLocal,
    AscendC::LocalTensor<int16_t> selectedRouteIndexLocal,
    int32_t peer, uint32_t routeCount)
{
    AscendC::Compares(compareMaskLocal, routeRankLocal, peer,
        AscendC::CMPMODE::EQ, routeCount);
    AscendC::PipeBarrier<PIPE_V>();
    uint64_t selectedCount = 0U;
    AscendC::GatherMask(selectedRouteIndexLocal, routeIndexLocal,
        compareMaskLocal.ReinterpretCast<uint16_t>(), true, routeCount,
        {1U, 1U, 0U, 0U}, selectedCount);
    SyncFunc<AscendC::HardEvent::V_S>();
    return static_cast<uint32_t>(selectedCount);
}

__aicore__ inline bool BuildDispatchWriteWqeBatch(
    AscendC::LocalTensor<uint8_t> issueLocal,
    AscendC::LocalTensor<int16_t> selectedRouteIndices,
    AscendC::LocalTensor<int32_t> dstValues,
    DispatchWqeBatchState &state, uint64_t localSourceBase,
    uint64_t rowBytes, uint64_t routeCountMask, uint32_t topKMagic,
    uint32_t topKShift, bool hiddenMode, uint32_t selectedStart, uint32_t tokenCount,
    bool appendSignal, uint64_t signalLocalAddr,
    uint32_t sequencePhase)
{
    __ubuf__ uint8_t *issueAddr = reinterpret_cast<__ubuf__ uint8_t *>(
        issueLocal.GetPhyAddr());
    __ubuf__ DispatchWqeBatchContext *context =
        reinterpret_cast<__ubuf__ DispatchWqeBatchContext *>(
            issueAddr + kDispatchWqeBatchContextOffset);
    context->localSourceBase = localSourceBase;
    context->remoteScratchBase =
        reinterpret_cast<uint64_t>(state.remoteScratchBase);
    context->rowBytes = rowBytes;
    context->routeCountMask = routeCountMask;
    context->signalLocalAddr = signalLocalAddr;
    context->signalRemoteAddr =
        reinterpret_cast<uint64_t>(state.remoteSignalAddr);
    context->rmtEidL = state.rmtEidL;
    context->rmtEidH = state.rmtEidH;
    context->batchHead = state.head;
    context->batchOutputOffset = state.batchCount;
    context->tokenCount = tokenCount;
    context->appendSignal = appendSignal ? 1U : 0U;
    context->topKMagic = topKMagic;
    context->topKShift = topKShift;
    context->hiddenMode = hiddenMode ? 1U : 0U;
    context->tokenEn = state.tokenEn;
    context->rmtJettyType = state.rmtJettyType;
    context->targetHint = state.targetHint;
    context->tpId = state.tpId;
    context->rmtJettyOrSegId = state.rmtJettyOrSegId;
    context->rmtTokenValue = state.rmtTokenValue;
    context->selectedStart = selectedStart;
    context->qpSelection =
        (state.qpIdx << 2U) | (sequencePhase & 3U);

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Simt::VF_CALL<DispatchBuildWriteWqeBatchVf>(
        AscendC::Simt::Dim3{kDispatchWqeBuildThreads, 1U, 1U},
        issueAddr,
        reinterpret_cast<__ubuf__ const int16_t *>(
            selectedRouteIndices.GetPhyAddr()),
        reinterpret_cast<__ubuf__ const int32_t *>(
            dstValues.GetPhyAddr()),
        reinterpret_cast<__ubuf__ const DispatchWqeBatchContext *>(context));
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
#else
    return false;
#endif
}

__aicore__ inline uint32_t DispatchPollCqBatch(
    DispatchWqeBatchState &state, AscendC::LocalTensor<uint8_t> cqeLocal)
{
    const uint32_t pollCount = TileXRMoonEp::DispatchCqePollBatchCount(
        state.cqTail, TileXR::TILEXR_UDMA_CQ_DEPTH,
        kDispatchCqePollBatchCapacity);
    if (pollCount == 0U) {
        return 0xFFU;
    }

    const uint32_t cqeBytes =
        pollCount * static_cast<uint32_t>(sizeof(TileXR::UDMACqeCtx));
    __gm__ uint8_t *cqeAddr = reinterpret_cast<__gm__ uint8_t *>(
        state.cqCtxEntry->bufAddr + sizeof(TileXR::UDMACqeCtx) *
            (state.cqTail % TileXR::TILEXR_UDMA_CQ_DEPTH));
    AscendC::GlobalTensor<uint8_t> cqeGlobal;
    cqeGlobal.SetGlobalBuffer(cqeAddr, cqeBytes);
    const AscendC::DataCopyExtParams copyParams {
        1U, cqeBytes, 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<uint8_t> padParams {
        false, 0U, 0U, 0U};

    for (uint32_t retry = 0U;
        retry < TileXR::TILEXR_UDMA_MAX_RETRY_TIMES; ++retry) {
        TileXR::UDMACleanCacheLines(cqeAddr, cqeBytes);
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        AscendC::DataCopyPad(cqeLocal, cqeGlobal, copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        __ubuf__ TileXR::UDMACqeCtx *cqeRecords =
            reinterpret_cast<__ubuf__ TileXR::UDMACqeCtx *>(
                cqeLocal.GetPhyAddr());
        uint32_t validCount = 0U;
        uint32_t completedSqTail = state.tail;
        for (; validCount < pollCount; ++validCount) {
            const uint32_t absoluteCqTail = state.cqTail + validCount;
            __ubuf__ const TileXR::UDMACqeCtx *cqe =
                cqeRecords + validCount;
            if (!TileXRMoonEp::DispatchCqeOwnerReady(
                    absoluteCqTail, TileXR::TILEXR_UDMA_CQ_DEPTH,
                    cqe->owner)) {
                break;
            }
            const uint32_t status = static_cast<uint32_t>(cqe->status);
            const uint32_t subStatus = static_cast<uint32_t>(cqe->substatus);
            if (status != 0U || subStatus != 0U) {
                return (status << 8U) | subStatus;
            }
            uint32_t candidateSqTail = state.tail;
            if (!TileXRMoonEp::DispatchCompletedSqTail(
                    state.tail, state.outstanding,
                    static_cast<uint32_t>(cqe->entryIdx),
                    TileXR::TILEXR_UDMA_SQ_BB_COUNT, candidateSqTail)) {
                return 0xFEU;
            }
            if (TileXRMoonEp::DispatchSqTailIsFurther(
                    state.tail, candidateSqTail, completedSqTail)) {
                completedSqTail = candidateSqTail;
            }
        }
        if (validCount == 0U) {
            continue;
        }

        const uint32_t newCqTail = state.cqTail + validCount;
        st_dev(newCqTail, reinterpret_cast<__gm__ uint32_t *>(
            state.cqCtxEntry->tailAddr), 0);
        st_dev(newCqTail & 0xFFFFFFU,
            reinterpret_cast<__gm__ uint32_t *>(
                state.cqCtxEntry->dbAddr), 0);
        st_dev(completedSqTail, reinterpret_cast<__gm__ uint32_t *>(
            state.qpCtxEntry->tailAddr), 0);
        state.cqTail = newCqTail;
        state.tail = completedSqTail;
        state.outstanding = state.wqeCount - state.tail;
        return 0U;
    }
    return 0xFFU;
}

__aicore__ inline bool DispatchEnsureSqBatchCapacity(
    DispatchWqeBatchState &state, uint32_t batchCount,
    AscendC::LocalTensor<uint8_t> cqeLocal, uint32_t phase,
    uint32_t &dfxFlags, uint32_t &firstQuietStatus,
    uint32_t &firstQuietPhase)
{
    while (static_cast<uint64_t>(state.outstanding) + batchCount >
        TileXR::TILEXR_UDMA_SQ_BB_COUNT - kDispatchSqPollReserve) {
        const uint32_t pollStatus = DispatchPollCqBatch(state, cqeLocal);
        if (pollStatus != 0U) {
            if (firstQuietStatus == 0U) {
                firstQuietStatus = pollStatus;
                firstQuietPhase = phase;
            }
            dfxFlags |= TileXRMoonEp::kDispatchDfxQuietError;
            return false;
        }
    }
    return true;
}

__aicore__ inline bool SubmitDispatchWqeBatch(DispatchWqeBatchState &state,
    AscendC::LocalTensor<uint8_t> issueLocal,
    AscendC::LocalTensor<uint8_t> cqeLocal, uint32_t phase,
    uint32_t &dfxFlags, uint32_t &firstQuietStatus,
    uint32_t &firstQuietPhase)
{
    const uint32_t batchCount = state.batchCount;
    if (batchCount == 0U) {
        return true;
    }
    if (batchCount > state.batchLimit ||
        !DispatchEnsureSqBatchCapacity(state, batchCount, cqeLocal,
            phase, dfxFlags, firstQuietStatus, firstQuietPhase)) {
        return false;
    }

    __ubuf__ uint8_t *issueAddr = reinterpret_cast<__ubuf__ uint8_t *>(
        issueLocal.GetPhyAddr());
    __ubuf__ TileXR::UDMASqeCtx *lastSqe =
        reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(
            issueAddr + (batchCount - 1U) * kDispatchUdmaWqeBytes);
    lastSqe->flag = static_cast<uint32_t>(lastSqe->flag) |
        TileXR::TILEXR_UDMA_SQE_FLAG_COMPLETION;

    __gm__ uint8_t *wqeAddr = reinterpret_cast<__gm__ uint8_t *>(
        state.qpCtxEntry->bufAddr + kDispatchUdmaWqeBytes *
            (state.head % TileXR::TILEXR_UDMA_SQ_BB_COUNT));
    const uint32_t batchBytes = batchCount * kDispatchUdmaWqeBytes;
    AscendC::GlobalTensor<uint8_t> wqeGlobal;
    wqeGlobal.SetGlobalBuffer(wqeAddr, batchBytes);
    const AscendC::DataCopyExtParams copyParams {
        1U, batchBytes, 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(wqeGlobal, issueLocal, copyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    const uint32_t batchEndHead = state.head + batchCount;
    const uint32_t batchEndWqeCount = state.wqeCount + batchCount;
    st_dev(batchEndHead, reinterpret_cast<__gm__ uint32_t *>(
        state.qpCtxEntry->headAddr), 0);
    st_dev(batchEndWqeCount, reinterpret_cast<__gm__ uint32_t *>(
        state.qpCtxEntry->wqeCntAddr), 0);
    st_dev(batchEndHead, reinterpret_cast<__gm__ uint32_t *>(
        state.qpCtxEntry->dbAddr), 0);

    state.head = batchEndHead;
    state.wqeCount = batchEndWqeCount;
    state.outstanding += batchCount;
    state.batchCount = 0U;
    state.batchLimit = TileXRMoonEp::DispatchWqeBatchCount(
        UINT64_MAX, state.head, TileXR::TILEXR_UDMA_SQ_BB_COUNT);
    return state.batchLimit != 0U;
}

__aicore__ inline bool AppendDispatchWqes(DispatchWqeBatchState &state,
    AscendC::LocalTensor<uint8_t> issueLocal,
    AscendC::LocalTensor<uint8_t> cqeLocal,
    AscendC::LocalTensor<int16_t> selectedRouteIndices,
    AscendC::LocalTensor<int32_t> dstValues, uint64_t localSourceBase,
    uint64_t rowBytes, uint64_t routeCountMask, uint32_t topKMagic,
    uint32_t topKShift, bool hiddenMode, uint32_t selectedRouteCount, bool appendSignal,
    uint64_t signalLocalAddr, uint32_t phase, uint32_t &dfxFlags,
    uint32_t &firstQuietStatus, uint32_t &firstQuietPhase,
    uint32_t sequencePhase)
{
    uint32_t selectedStart = 0U;
    bool signalPending = appendSignal;
    while (selectedStart < selectedRouteCount || signalPending) {
        if (state.batchCount == state.batchLimit &&
            !SubmitDispatchWqeBatch(state, issueLocal, cqeLocal,
                phase, dfxFlags,
                firstQuietStatus, firstQuietPhase)) {
            return false;
        }
        const uint32_t available = state.batchLimit - state.batchCount;
        const uint32_t selectedRemaining = selectedRouteCount - selectedStart;
        const bool appendSignalNow = signalPending &&
            static_cast<uint64_t>(selectedRemaining) + 1U <= available;
        const uint32_t tokenCapacity = available -
            (appendSignalNow ? 1U : 0U);
        const uint32_t tokenCount = selectedRemaining < tokenCapacity ?
            selectedRemaining : tokenCapacity;
        if (!BuildDispatchWriteWqeBatch(issueLocal,
                selectedRouteIndices, dstValues, state, localSourceBase,
                rowBytes, routeCountMask, topKMagic, topKShift, hiddenMode,
                selectedStart, tokenCount,
                appendSignalNow, signalLocalAddr, sequencePhase)) {
            return false;
        }
        state.batchCount += tokenCount + (appendSignalNow ? 1U : 0U);
        selectedStart += tokenCount;
        if (appendSignalNow) {
            signalPending = false;
        }
        if (state.batchCount == state.batchLimit &&
            !SubmitDispatchWqeBatch(state, issueLocal, cqeLocal,
                phase, dfxFlags,
                firstQuietStatus, firstQuietPhase)) {
            return false;
        }
    }
    return !appendSignal || SubmitDispatchWqeBatch(state, issueLocal,
        cqeLocal, phase, dfxFlags, firstQuietStatus, firstQuietPhase);
}

__aicore__ inline bool DispatchDrainSqToExpected(
    DispatchWqeBatchState &state, uint32_t expectedSqTail,
    AscendC::LocalTensor<uint8_t> cqeLocal, uint64_t timeoutTicks,
    uint32_t phase, uint32_t &dfxFlags, uint32_t &firstQuietStatus,
    uint32_t &firstQuietPhase, uint32_t &timeoutPeer,
    uint32_t &timeoutPhase, uint64_t &timeoutObserved)
{
    const uint32_t pending = expectedSqTail - state.tail;
    if (pending == 0U) {
        return true;
    }
    if (pending >= TileXR::TILEXR_UDMA_SQ_BB_COUNT) {
        dfxFlags |= TileXRMoonEp::kDispatchDfxCqError;
        if (firstQuietStatus == 0U) {
            firstQuietStatus = 0xFFFFFFFCU;
            firstQuietPhase = phase;
        }
        return false;
    }

    const uint64_t waitStart = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (state.tail != expectedSqTail) {
        const uint32_t pollStatus = DispatchPollCqBatch(state, cqeLocal);
        if (pollStatus != 0U && pollStatus != 0xFFU) {
            dfxFlags |= TileXRMoonEp::kDispatchDfxCqError;
            if (firstQuietStatus == 0U) {
                firstQuietStatus = pollStatus;
                firstQuietPhase = phase;
            }
            return false;
        }
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - waitStart >=
            timeoutTicks) {
            dfxFlags |= TileXRMoonEp::kDispatchDfxCqError;
            if (timeoutPeer == UINT32_MAX) {
                timeoutPeer = static_cast<uint32_t>(state.targetRank);
                timeoutPhase = phase;
                timeoutObserved = state.tail;
            }
            return false;
        }
    }
    return true;
}

__aicore__ inline bool DispatchDrainHistoricalCq(
    DispatchPreparedPeer &peer, AscendC::LocalTensor<uint8_t> cqeLocal,
    uint64_t timeoutTicks, uint32_t &dfxFlags,
    uint32_t &firstQuietStatus, uint32_t &firstQuietPhase,
    uint32_t &timeoutPeer, uint32_t &timeoutPhase,
    uint64_t &timeoutObserved)
{
    for (uint32_t qpIdx = 0U; qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
        DispatchWqeBatchState &state = peer.qpState[qpIdx];
        if (!DispatchDrainSqToExpected(state, state.wqeCount, cqeLocal,
                timeoutTicks, peer.issuePhase, dfxFlags, firstQuietStatus,
                firstQuietPhase, timeoutPeer, timeoutPhase, timeoutObserved)) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool DispatchBuildGroupedQpBatch(
    DispatchWqeBatchState &state, AscendC::LocalTensor<uint8_t> issueLocal,
    AscendC::LocalTensor<int16_t> selectedRouteIndices,
    AscendC::LocalTensor<int32_t> dstValues, uint64_t localSourceBase,
    uint64_t rowBytes, uint64_t routeCountMask, uint32_t topKMagic,
    uint32_t topKShift, bool hiddenMode, uint32_t selectedRouteCount,
    uint32_t &selectedStart, bool &signalPending, uint64_t signalLocalAddr,
    uint32_t sequencePhase, bool &finalBatch)
{
    finalBatch = false;
    state.batchCount = 0U;
    if (selectedStart >= selectedRouteCount && !signalPending) {
        return true;
    }
    const uint32_t available = state.batchLimit;
    if (available == 0U) {
        return false;
    }
    const uint32_t selectedRemaining = selectedRouteCount - selectedStart;
    const bool appendSignalNow = signalPending &&
        static_cast<uint64_t>(selectedRemaining) + 1U <= available;
    const uint32_t tokenCapacity = available - (appendSignalNow ? 1U : 0U);
    const uint32_t tokenCount = selectedRemaining < tokenCapacity ?
        selectedRemaining : tokenCapacity;
    if (tokenCount == 0U && !appendSignalNow) {
        return false;
    }
    if (!BuildDispatchWriteWqeBatch(issueLocal, selectedRouteIndices,
            dstValues, state, localSourceBase, rowBytes, routeCountMask,
            topKMagic, topKShift, hiddenMode, selectedStart, tokenCount,
            appendSignalNow, signalLocalAddr, sequencePhase)) {
        return false;
    }
    state.batchCount = tokenCount + (appendSignalNow ? 1U : 0U);
    selectedStart += tokenCount;
    if (appendSignalNow) {
        signalPending = false;
        finalBatch = true;
    }
    return true;
}

__aicore__ inline bool StageDispatchQpBatch(
    DispatchWqeBatchState &state, AscendC::LocalTensor<uint8_t> issueLocal,
    AscendC::LocalTensor<uint8_t> cqeLocal, bool finalBatch,
    uint32_t phase, uint32_t &dfxFlags, uint32_t &firstQuietStatus,
    uint32_t &firstQuietPhase)
{
    const uint32_t batchCount = state.batchCount;
    if (batchCount == 0U || state.doorbellPending != 0U ||
        batchCount > state.batchLimit ||
        !DispatchEnsureSqBatchCapacity(state, batchCount, cqeLocal,
            phase, dfxFlags, firstQuietStatus, firstQuietPhase)) {
        return false;
    }

    __gm__ uint8_t *wqeAddr = reinterpret_cast<__gm__ uint8_t *>(
        state.qpCtxEntry->bufAddr + kDispatchUdmaWqeBytes *
            (state.head % TileXR::TILEXR_UDMA_SQ_BB_COUNT));
    const uint32_t batchBytes = batchCount * kDispatchUdmaWqeBytes;
    AscendC::GlobalTensor<uint8_t> wqeGlobal;
    wqeGlobal.SetGlobalBuffer(wqeAddr, batchBytes);
    const AscendC::DataCopyExtParams copyParams {
        1U, batchBytes, 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(wqeGlobal, issueLocal, copyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    state.head += batchCount;
    state.wqeCount += batchCount;
    state.outstanding += batchCount;
    state.stagedDoorbellHead = state.head;
    state.doorbellPending = 1U;
    if (finalBatch) {
        state.finalWqeCount = state.wqeCount;
        state.finalStaged = 1U;
    }
    st_dev(state.head, reinterpret_cast<__gm__ uint32_t *>(
        state.qpCtxEntry->headAddr), 0);
    st_dev(state.wqeCount, reinterpret_cast<__gm__ uint32_t *>(
        state.qpCtxEntry->wqeCntAddr), 0);
    state.batchCount = 0U;
    state.batchLimit = TileXRMoonEp::DispatchWqeBatchCount(
        UINT64_MAX, state.head, TileXR::TILEXR_UDMA_SQ_BB_COUNT);
    return state.batchLimit != 0U;
}

__aicore__ inline void RingDispatchPeerDoorbells(DispatchPreparedPeer &peer)
{
    for (uint32_t qpIdx = 0U; qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
        DispatchWqeBatchState &state = peer.qpState[qpIdx];
        if (state.doorbellPending == 0U) {
            continue;
        }
        st_dev(state.stagedDoorbellHead, reinterpret_cast<__gm__ uint32_t *>(
            state.qpCtxEntry->dbAddr), 0);
        state.doorbellPending = 0U;
        state.doorbellRung = 1U;
    }
}

__aicore__ inline bool DispatchDrainPeerFinalCq(
    DispatchPreparedPeer &peer, AscendC::LocalTensor<uint8_t> cqeLocal,
    uint64_t timeoutTicks, uint32_t &dfxFlags,
    uint32_t &firstQuietStatus, uint32_t &firstQuietPhase,
    uint32_t &timeoutPeer, uint32_t &timeoutPhase,
    uint64_t &timeoutObserved)
{
    for (uint32_t qpIdx = 0U; qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
        DispatchWqeBatchState &state = peer.qpState[qpIdx];
        if (state.finalStaged == 0U || state.doorbellRung == 0U ||
            !DispatchDrainSqToExpected(state, state.finalWqeCount, cqeLocal,
                timeoutTicks, peer.issuePhase, dfxFlags, firstQuietStatus,
                firstQuietPhase, timeoutPeer, timeoutPhase, timeoutObserved)) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool DecodeSendDst(int32_t encoded, uint64_t destinationCapacity,
    int32_t rankSize, int32_t &targetRank, uint64_t &targetSlot)
{
    int64_t rank = -1;
    int64_t slot = -1;
    const bool valid = TileXRMoonEp::DispatchDecodeDestination(encoded,
        static_cast<int64_t>(destinationCapacity), rankSize, &rank, &slot);
    targetRank = static_cast<int32_t>(rank);
    targetSlot = static_cast<uint64_t>(slot);
    return valid;
}

__aicore__ inline int32_t DecodeRawDst(int32_t encoded)
{
    return encoded;
}

__aicore__ inline void RecordInvalidRoute(uint64_t routeId, int32_t rawDst,
    uint32_t &dfxFlags, uint32_t &firstInvalidRouteId,
    int32_t &firstInvalidRawDst)
{
    dfxFlags |= TileXRMoonEp::kDispatchDfxInvalidRoute;
    if (firstInvalidRouteId == UINT32_MAX) {
        firstInvalidRouteId = static_cast<uint32_t>(routeId);
        firstInvalidRawDst = rawDst;
    }
}

__aicore__ inline void CopyBytesGmToGm(__gm__ uint8_t *dst,
    const __gm__ uint8_t *src, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    for (uint32_t offset = 0U; offset < bytes; offset += kRelayUbBytes) {
        const uint32_t remaining = bytes - offset;
        const uint32_t tileBytes = remaining < kRelayUbBytes ? remaining : kRelayUbBytes;
        AscendC::GlobalTensor<uint8_t> srcGlobal;
        srcGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t *>(src) + offset);
        const AscendC::DataCopyExtParams copyIn {1U, tileBytes, 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0U};
        AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        AscendC::GlobalTensor<uint8_t> dstGlobal;
        dstGlobal.SetGlobalBuffer(dst + offset);
        const AscendC::DataCopyExtParams copyOut {1U, tileBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CopyContiguousBytesGmToGmPipelined(
    __gm__ uint8_t *dst, const __gm__ uint8_t *src, uint64_t bytes,
    uint32_t tileBytes,
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> &outputCopyQueue)
{
    if (bytes == 0U || tileBytes == 0U) {
        return;
    }

    AscendC::GlobalTensor<uint8_t> srcGlobal;
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    srcGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t *>(src), bytes);
    dstGlobal.SetGlobalBuffer(dst, bytes);

    const AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0U};
    uint64_t pendingOffset = 0U;
    uint32_t pendingBytes = static_cast<uint32_t>(
        bytes < tileBytes ? bytes : tileBytes);
    AscendC::LocalTensor<uint8_t> firstLocal =
        outputCopyQueue.AllocTensor<uint8_t>();
    const AscendC::DataCopyExtParams firstCopyIn {
        1U, pendingBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(firstLocal, srcGlobal, firstCopyIn, padIn);
    outputCopyQueue.EnQue(firstLocal);

    uint64_t offset = pendingBytes;
    while (offset < bytes) {
        AscendC::LocalTensor<uint8_t> pendingLocal =
            outputCopyQueue.DeQue<uint8_t>();
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

        const uint64_t remaining = bytes - offset;
        const uint32_t currentBytes = static_cast<uint32_t>(
            remaining < tileBytes ? remaining : tileBytes);
        AscendC::LocalTensor<uint8_t> currentLocal =
            outputCopyQueue.AllocTensor<uint8_t>();
        const AscendC::DataCopyExtParams currentCopyIn {
            1U, currentBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(currentLocal, srcGlobal[offset], currentCopyIn, padIn);

        const AscendC::DataCopyExtParams copyOut {
            1U, pendingBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstGlobal[pendingOffset], pendingLocal, copyOut);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        outputCopyQueue.FreeTensor(pendingLocal);

        outputCopyQueue.EnQue(currentLocal);
        pendingOffset = offset;
        pendingBytes = currentBytes;
        offset += currentBytes;
    }

    AscendC::LocalTensor<uint8_t> pendingLocal =
        outputCopyQueue.DeQue<uint8_t>();
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    const AscendC::DataCopyExtParams copyOut {
        1U, pendingBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(dstGlobal[pendingOffset], pendingLocal, copyOut);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    outputCopyQueue.FreeTensor(pendingLocal);
}

__aicore__ inline uint64_t LoadCompletionFlag(__gm__ uint64_t *flag,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint8_t> flagGlobal;
    flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(flag));
    const AscendC::DataCopyExtParams copyIn {
        1U, static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0U};
    AscendC::DataCopyPad(relayLocal, flagGlobal, copyIn, padIn);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return relayLocal.ReinterpretCast<uint64_t>().GetValue(0);
}

__aicore__ inline bool WaitCompletionFlag(__gm__ uint64_t *flag,
    uint64_t expected, uint64_t waitStart, uint64_t timeoutTicks,
    uint64_t &observed, AscendC::LocalTensor<uint8_t> relayLocal)
{
    observed = LoadCompletionFlag(flag, relayLocal);
    while (observed < expected) {
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - waitStart >=
            timeoutTicks) {
            return false;
        }
        observed = LoadCompletionFlag(flag, relayLocal);
    }
    return true;
}

__aicore__ inline uint64_t LoadDispatchCredit(
    __gm__ uint64_t *credit, AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint64_t> creditGlobal;
    creditGlobal.SetGlobalBuffer(credit,
        TileXRMoonEp::kDispatchCreditStrideBytes / sizeof(uint64_t));
    auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();
    AscendC::DataCopy(creditLocal, creditGlobal,
        TileXRMoonEp::kDispatchCreditStrideBytes / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    return creditLocal.GetValue(0);
}

__aicore__ inline bool WaitDispatchPeerCredit(
    const __gm__ TileXR::CommArgs *args, int32_t rank, int32_t peer,
    uint32_t group, int64_t magic, uint64_t timeoutTicks,
    AscendC::LocalTensor<uint8_t> relayLocal, uint64_t &observed)
{
    if (!TileXRMoonEp::DispatchCreditRequired(group)) {
        observed = 0U;
        return true;
    }
    uint64_t expected = 0U;
    const uint64_t planeOffset = TileXRMoonEp::DispatchCreditPlaneOffset(magic);
    const uint64_t entryOffset = TileXRMoonEp::DispatchCreditEntryOffset(
        static_cast<uint32_t>(peer));
    if (args == nullptr || rank < 0 || peer < 0 ||
        args->creditMems[rank] == nullptr || planeOffset == UINT64_MAX ||
        entryOffset == UINT64_MAX ||
        !TileXRMoonEp::DispatchCreditToken(magic, group, expected)) {
        observed = 0U;
        return false;
    }
    auto credit = reinterpret_cast<__gm__ uint64_t *>(
        args->creditMems[rank] + planeOffset + entryOffset);
    const uint64_t waitStart = static_cast<uint64_t>(AscendC::GetSystemCycle());
    observed = LoadDispatchCredit(credit, relayLocal);
    while (!TileXRMoonEp::DispatchCreditReady(observed, expected)) {
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - waitStart >=
            timeoutTicks) {
            return false;
        }
        observed = LoadDispatchCredit(credit, relayLocal);
    }
    return true;
}

__aicore__ inline bool PublishDispatchNextCredit(
    const __gm__ TileXR::CommArgs *args, int32_t rank, int32_t rankSize,
    uint32_t group, uint32_t lane, uint32_t groupWidth, int64_t magic,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    const int64_t nextPeer = TileXRMoonEp::DispatchGroupedNextCreditPeer(
        rank, rankSize, group, lane, groupWidth);
    if (nextPeer < 0) {
        return true;
    }
    uint64_t token = 0U;
    const uint64_t planeOffset = TileXRMoonEp::DispatchCreditPlaneOffset(magic);
    const uint64_t entryOffset = TileXRMoonEp::DispatchCreditEntryOffset(
        static_cast<uint32_t>(rank));
    if (args == nullptr || args->creditMems[nextPeer] == nullptr ||
        planeOffset == UINT64_MAX || entryOffset == UINT64_MAX ||
        !TileXRMoonEp::DispatchCreditToken(magic, group + 1U, token)) {
        return false;
    }

    auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();
    constexpr uint32_t creditWords =
        TileXRMoonEp::kDispatchCreditStrideBytes / sizeof(uint64_t);
    for (uint32_t word = 0U; word < creditWords; ++word) {
        creditLocal.SetValue(word, word == 0U ? token : 0U);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::GlobalTensor<uint64_t> remoteCreditGlobal;
    remoteCreditGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint64_t *>(
            args->creditMems[nextPeer] + planeOffset + entryOffset),
        creditWords);
    AscendC::DataCopy(remoteCreditGlobal, creditLocal, creditWords);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    return true;
}

__aicore__ inline bool WaitDispatchIncomingPeerAndPublishCredit(
    const __gm__ TileXR::CommArgs *args, __gm__ uint64_t *receiveFlags,
    int32_t rank, int32_t rankSize, int32_t incomingPeer,
    uint32_t group, uint32_t lane, uint32_t groupWidth, int64_t magic,
    bool publishCredit, uint64_t timeoutTicks,
    AscendC::LocalTensor<uint8_t> relayLocal, uint32_t &dfxFlags,
    uint32_t &timeoutPeer, uint32_t &timeoutPhase,
    uint64_t &timeoutObserved)
{
    const uint64_t waitStart = static_cast<uint64_t>(AscendC::GetSystemCycle());
    const uint64_t peerFlagBase = static_cast<uint64_t>(incomingPeer) *
        TileXRMoonEp::kDispatchQpCount;
    for (uint32_t qpIdx = 0U; qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
        uint64_t observed = 0U;
        if (!WaitCompletionFlag(receiveFlags + peerFlagBase + qpIdx,
                static_cast<uint64_t>(magic), waitStart, timeoutTicks,
                observed, relayLocal)) {
            dfxFlags |= TileXRMoonEp::kDispatchDfxCompletionTimeout;
            if (timeoutPeer == UINT32_MAX) {
                timeoutPeer = static_cast<uint32_t>(incomingPeer);
                timeoutPhase = group;
                timeoutObserved = observed;
            }
            return false;
        }
    }
    if (publishCredit && !PublishDispatchNextCredit(args, rank, rankSize,
            group, lane, groupWidth, magic, relayLocal)) {
        dfxFlags |= TileXRMoonEp::kDispatchDfxInvalidConfig;
        return false;
    }
    return true;
}

__aicore__ inline uint32_t LoadRouteTile(
    AscendC::GlobalTensor<int32_t> dstGlobal, uint64_t tileStart,
    uint64_t routeCount, AscendC::LocalTensor<int32_t> routeTileLocal)
{
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    const uint64_t remaining = routeCount - tileStart;
    const uint32_t tileElements = static_cast<uint32_t>(
        remaining < kRouteTileElements ? remaining : kRouteTileElements);
    const AscendC::DataCopyExtParams copyIn {
        1U, tileElements * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<int32_t> padIn {false, 0U, 0U, 0U};
    AscendC::DataCopyPad(routeTileLocal, dstGlobal[tileStart], copyIn, padIn);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    return tileElements;
}

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_DFX)
__aicore__ inline void WriteDfxRecord(__gm__ uint8_t *dfx,
    uint32_t payloadMode, uint32_t rank, uint32_t core, uint32_t flags,
    uint32_t firstInvalidRouteId, int32_t firstInvalidRawDst,
    uint32_t firstQuietStatus, uint32_t firstQuietPhase,
    uint32_t timeoutPeer, uint32_t timeoutPhase, uint64_t expectedRouteCount,
    uint64_t processedRouteCount, uint64_t magic, uint64_t timeoutExpectedMagic,
    uint64_t timeoutObservedFlag, AscendC::LocalTensor<uint8_t> diagnosticLocal)
{
    __ubuf__ TileXRMoonEp::DispatchDfxRecord *record =
        reinterpret_cast<__ubuf__ TileXRMoonEp::DispatchDfxRecord *>(
            diagnosticLocal.GetPhyAddr());
    record->marker = TileXRMoonEp::kDispatchDfxMarker;
    record->version = TileXRMoonEp::kDispatchDiagnosticVersion;
    record->recordBytes = sizeof(TileXRMoonEp::DispatchDfxRecord);
    record->payloadMode = payloadMode;
    record->rank = rank;
    record->core = core;
    record->flags = flags;
    record->firstInvalidRouteId = firstInvalidRouteId;
    record->firstInvalidRawDst = firstInvalidRawDst;
    record->firstQuietStatus = firstQuietStatus;
    record->firstQuietPhase = firstQuietPhase;
    record->timeoutPeer = timeoutPeer;
    record->timeoutPhase = timeoutPhase;
    record->reserved0 = 0U;
    record->expectedRouteCount = expectedRouteCount;
    record->processedRouteCount = processedRouteCount;
    record->magic = magic;
    record->timeoutExpectedMagic = timeoutExpectedMagic;
    record->timeoutObservedFlag = timeoutObservedFlag;
    for (uint32_t index = 0U; index < 4U; ++index) {
        record->reserved[index] = 0U;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    AscendC::GlobalTensor<uint8_t> dfxGlobal;
    dfxGlobal.SetGlobalBuffer(dfx + static_cast<uint64_t>(core) *
        sizeof(TileXRMoonEp::DispatchDfxRecord));
    const AscendC::DataCopyExtParams copyOut {
        1U, static_cast<uint32_t>(sizeof(TileXRMoonEp::DispatchDfxRecord)),
        0U, 0U, 0U};
    AscendC::DataCopyPad(dfxGlobal, diagnosticLocal, copyOut);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}
#endif

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
__aicore__ inline void WriteProfileRecord(__gm__ uint8_t *profile,
    uint32_t payloadMode, uint32_t rank, uint32_t core, uint32_t blockDim,
    uint32_t flags, uint64_t magic, uint32_t scratchIndex,
    uint32_t groupCount, uint32_t selectMode, uint32_t fallbackReason,
    uint64_t scannedRouteCount, uint64_t matchedRouteCount,
    uint64_t selectedRouteCount, uint64_t processedRouteCount,
    uint64_t issuedPutCount, uint64_t issuedPutBytes,
    uint64_t visitedPeerCount, uint64_t completionFlagCount,
    uint64_t kernelCycles, uint64_t stagingCycles, uint64_t putIssueCycles,
    uint64_t flagWaitCycles, uint64_t outputCopyCycles, uint64_t quietCycles,
    uint64_t stagingEndOffset, uint64_t issueStartOffset,
    uint64_t remoteIssueEndOffset, uint64_t issueEndOffset,
    uint64_t flagWaitStartOffset, uint64_t flagWaitEndOffset,
    uint64_t dfxWriteEndOffset, uint64_t syncAllEndOffset,
    uint64_t outputStartOffset, uint64_t outputEndOffset,
    uint64_t quietEndOffset,
    AscendC::LocalTensor<uint8_t> diagnosticLocal)
{
    __ubuf__ TileXRMoonEp::DispatchProfileRecord *record =
        reinterpret_cast<__ubuf__ TileXRMoonEp::DispatchProfileRecord *>(
            diagnosticLocal.GetPhyAddr());
    record->marker = TileXRMoonEp::kDispatchProfileMarker;
    record->version = TileXRMoonEp::kDispatchDiagnosticVersion;
    record->recordBytes = sizeof(TileXRMoonEp::DispatchProfileRecord);
    record->payloadMode = payloadMode;
    record->rank = rank;
    record->core = core;
    record->blockDim = blockDim;
    record->flags = flags;
    record->selectMode = selectMode;
    record->vectorFallbackReason = fallbackReason;
    record->scratchIndex = scratchIndex;
    record->groupCount = groupCount;
    record->magic = magic;
    record->scannedRouteCount = scannedRouteCount;
    record->matchedRouteCount = matchedRouteCount;
    record->selectedRouteCount = selectedRouteCount;
    record->processedRouteCount = processedRouteCount;
    record->issuedPutCount = issuedPutCount;
    record->issuedPutBytes = issuedPutBytes;
    record->visitedPeerCount = visitedPeerCount;
    record->completionFlagCount = completionFlagCount;
    record->kernelCycles = kernelCycles;
    record->stagingCycles = stagingCycles;
    record->putIssueCycles = putIssueCycles;
    record->flagWaitCycles = flagWaitCycles;
    record->outputCopyCycles = outputCopyCycles;
    record->quietCycles = quietCycles;
    record->reserved[TileXRMoonEp::kDispatchTimelineStagingEnd] = stagingEndOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineIssueStart] = issueStartOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineRemoteIssueEnd] =
        remoteIssueEndOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineIssueEnd] = issueEndOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineFlagWaitStart] = flagWaitStartOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineFlagWaitEnd] = flagWaitEndOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineDfxWriteEnd] = dfxWriteEndOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineSyncAllEnd] = syncAllEndOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineOutputStart] = outputStartOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineOutputEnd] = outputEndOffset;
    record->reserved[TileXRMoonEp::kDispatchTimelineQuietEnd] = quietEndOffset;
    for (uint32_t index = TileXRMoonEp::kDispatchTimelinePointCount;
        index < 11U; ++index) {
        record->reserved[index] = 0U;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    AscendC::GlobalTensor<uint8_t> profileGlobal;
    profileGlobal.SetGlobalBuffer(profile + static_cast<uint64_t>(core) *
        sizeof(TileXRMoonEp::DispatchProfileRecord));
    const AscendC::DataCopyExtParams copyOut {
        1U, static_cast<uint32_t>(sizeof(TileXRMoonEp::DispatchProfileRecord)),
        0U, 0U, 0U};
    AscendC::DataCopyPad(profileGlobal, diagnosticLocal, copyOut);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}
#endif

__aicore__ inline int32_t StatusFromDfxFlags(uint32_t flags)
{
    if ((flags & TileXRMoonEp::kDispatchDfxCreditTimeout) != 0U) {
        return TileXRMoonEp::kDispatchStatusCreditTimeout;
    }
    if ((flags & TileXRMoonEp::kDispatchDfxCqError) != 0U) {
        return TileXRMoonEp::kDispatchStatusCqError;
    }
    if ((flags & TileXRMoonEp::kDispatchDfxCompletionTimeout) != 0U) {
        return TileXRMoonEp::kDispatchStatusCompletionTimeout;
    }
    if ((flags & TileXRMoonEp::kDispatchDfxInvalidConfig) != 0U) {
        return TileXRMoonEp::kDispatchStatusInvalidConfig;
    }
    if ((flags & TileXRMoonEp::kDispatchDfxInvalidRoute) != 0U) {
        return TileXRMoonEp::kDispatchStatusInvalidRoute;
    }
    if ((flags & TileXRMoonEp::kDispatchDfxRouteCountMismatch) != 0U) {
        return TileXRMoonEp::kDispatchStatusRouteCountMismatch;
    }
    if ((flags & TileXRMoonEp::kDispatchDfxUpstreamPlannerError) != 0U) {
        return TileXRMoonEp::kDispatchStatusUpstreamPlanner;
    }
    if ((flags & TileXRMoonEp::kDispatchDfxQuietError) != 0U) {
        return TileXRMoonEp::kDispatchStatusQuietError;
    }
    return TileXRMoonEp::kDispatchStatusSuccess;
}

__aicore__ inline void DispatchPublishFirstStatus(
    __gm__ int32_t *planStatus, int32_t status)
{
    if (planStatus == nullptr || status == TileXRMoonEp::kDispatchStatusSuccess) {
        return;
    }
    AscendC::AtomicCas(
        reinterpret_cast<__gm__ uint32_t *>(planStatus),
        static_cast<uint32_t>(TileXRMoonEp::kDispatchStatusSuccess),
        static_cast<uint32_t>(status));
}

__aicore__ inline void ClearDispatchBytes(__gm__ uint8_t *dst,
    uint64_t bytes, AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::LocalTensor<uint16_t> zeros = relayLocal.ReinterpretCast<uint16_t>();
    AscendC::Duplicate(zeros, static_cast<uint16_t>(0),
        static_cast<int32_t>(kRelayUbBytes / sizeof(uint16_t)));
    SyncFunc<AscendC::HardEvent::V_MTE3>();

    AscendC::GlobalTensor<uint8_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(dst, bytes);
    for (uint64_t offset = 0U; offset < bytes; offset += kRelayUbBytes) {
        const uint64_t remaining = bytes - offset;
        const uint32_t tileBytes = static_cast<uint32_t>(
            remaining < kRelayUbBytes ? remaining : kRelayUbBytes);
        const AscendC::DataCopyExtParams copyOut {
            1U, tileBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstGlobal[offset], relayLocal, copyOut);
    }
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

__aicore__ inline bool ClearDispatchZeroFillRanges(
    __gm__ uint8_t *scratch, const __gm__ int32_t *zeroFillRanges,
    uint64_t zeroFillRangeCount, uint64_t destinationCapacity,
    uint64_t rowBytes, uint64_t blockIdx, uint64_t blockNum,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    for (uint64_t range = blockIdx; range < zeroFillRangeCount;
         range += blockNum) {
        const int32_t startValue = zeroFillRanges[range * 2U];
        const int32_t countValue = zeroFillRanges[range * 2U + 1U];
        if (startValue < 0 || countValue < 0) {
            return false;
        }
        const uint64_t start = static_cast<uint64_t>(startValue);
        const uint64_t count = static_cast<uint64_t>(countValue);
        if (start > destinationCapacity ||
            count > destinationCapacity - start) {
            return false;
        }
        if (count != 0U) {
            ClearDispatchBytes(scratch + start * rowBytes,
                count * rowBytes, relayLocal);
        }
    }
    return true;
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_moonep_dispatch_urma_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR dstGM,
    GM_ADDR zeroFillRangesGM, GM_ADDR workspaceGM, GM_ADDR outputGM,
    GM_ADDR planStatusGM, uint64_t profileOffset,
    uint64_t scratchOffset, uint64_t completionFlagsOffset,
    uint64_t signalOffset, uint64_t dfxOffset, uint64_t kernelStatusOffset,
    int64_t s, int64_t k, int64_t h, int64_t routeCountArg,
    int64_t destinationCapacityArg, int64_t zeroFillRangeCountArg,
    uint64_t rowBytes,
    uint64_t payloadMode, int64_t magic,
    uint64_t completionTimeoutTicks, uint64_t peerMode,
    uint64_t groupWidthArg)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
        auto input = reinterpret_cast<__gm__ uint8_t *>(inputGM);
        auto dst = reinterpret_cast<__gm__ int32_t *>(dstGM);
        auto zeroFillRanges = reinterpret_cast<__gm__ int32_t *>(zeroFillRangesGM);
        auto workspace = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        auto output = reinterpret_cast<__gm__ uint8_t *>(outputGM);
        auto planStatus = reinterpret_cast<__gm__ int32_t *>(planStatusGM);
        if (args == nullptr || input == nullptr || dst == nullptr ||
            zeroFillRanges == nullptr || workspace == nullptr || output == nullptr ||
            planStatus == nullptr || s <= 0 || k <= 0 || h <= 0 ||
            routeCountArg <= 0 || destinationCapacityArg < routeCountArg ||
            zeroFillRangeCountArg <= 0 || zeroFillRangeCountArg > UINT32_MAX ||
            rowBytes == 0U || rowBytes > UINT32_MAX || magic <= 0 ||
            completionTimeoutTicks == 0U ||
            peerMode > UINT32_MAX || groupWidthArg > UINT32_MAX ||
            !TileXRMoonEp::DispatchPayloadModeValid(static_cast<uint32_t>(payloadMode)) ||
            !TileXRMoonEp::DispatchPeerModeValid(static_cast<uint32_t>(peerMode)) ||
            !TileXRMoonEp::DispatchGroupWidthValid(
                static_cast<uint32_t>(groupWidthArg))) {
            return;
        }

        const int32_t rank = args->rank;
        const int32_t rankSize = args->rankSize;
        const bool localOnly = rankSize == 1;
        const bool groupedPeerMode = TileXRMoonEp::DispatchPeerModeUsesGroups(
            static_cast<uint32_t>(peerMode));
        const bool creditPeerMode = TileXRMoonEp::DispatchPeerModeUsesCredit(
            static_cast<uint32_t>(peerMode));
        const uint32_t groupWidth = static_cast<uint32_t>(groupWidthArg);
        auto udmaInfo = localOnly ? nullptr : TileXR::GetUDMAInfo(args);
        if (rankSize <= 0 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE || rank < 0 ||
            rank >= rankSize || (!localOnly && (!TileXR::UDMARegistryEnabled(args) ||
                udmaInfo == nullptr || TileXR::UDMAQpCount(args) <
                    TileXRMoonEp::kDispatchQpCount))) {
            return;
        }

        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        const uint64_t routeCount = static_cast<uint64_t>(routeCountArg);
        const uint64_t destinationCapacity = static_cast<uint64_t>(
            destinationCapacityArg);
        const uint64_t zeroFillRangeCount = static_cast<uint64_t>(
            zeroFillRangeCountArg);
        if (blockNum == 0U || blockNum > TileXRMoonEp::kDispatchAivCoreCount ||
            routeCount > UINT32_MAX || destinationCapacity > UINT32_MAX ||
            routeCount / static_cast<uint64_t>(k) != static_cast<uint64_t>(s) ||
            routeCount % static_cast<uint64_t>(k) != 0U) {
            return;
        }

        const bool hiddenMode = payloadMode == static_cast<uint64_t>(
            TileXRMoonEp::DispatchPayloadMode::Hidden);
        uint32_t topKMagic = 0U;
        uint32_t topKShift = 0U;
        AscendC::GetUintDivMagicAndShift(topKMagic, topKShift,
            static_cast<uint32_t>(k));
        const uint64_t sourceRows = hiddenMode ? static_cast<uint64_t>(s) : routeCount;
        const uint64_t scratchSlotBytes = MultiplyU32ToU64(
            static_cast<uint32_t>(destinationCapacity),
            static_cast<uint32_t>(rowBytes));
        const uint64_t expectedFlag = static_cast<uint64_t>(magic);
        const uint64_t scratchIndex = expectedFlag %
            TileXRMoonEp::kDispatchScratchBufferCount;
        const int64_t groupCount = groupedPeerMode ?
            static_cast<int64_t>(TileXRMoonEp::DispatchGroupedGroupCount(
                rankSize, groupWidth)) :
            TileXRMoonEp::DispatchGroupCount(rankSize);
        const uint32_t peerWorkCount = groupedPeerMode ?
            TileXRMoonEp::DispatchGroupedPeerWorkCount(rankSize, groupWidth,
                static_cast<uint32_t>(blockNum)) :
            TileXRMoonEp::DispatchPeerWorkCount(static_cast<uint32_t>(blockNum));
        if ((!localOnly && (groupCount <= 0 || peerWorkCount == 0U)) ||
            (!groupedPeerMode && groupCount > 8)) {
            return;
        }

        uint32_t fallbackReason = TileXRMoonEp::kDispatchVectorFallbackNone;
        bool useVectorSlotSelect =
            (destinationCapacity & (destinationCapacity - 1U)) == 0U;
        if (!useVectorSlotSelect) {
            fallbackReason = TileXRMoonEp::kDispatchVectorFallbackNotPowerOfTwo;
        } else if (routeCount < kVectorCompareMinElements) {
            useVectorSlotSelect = false;
            fallbackReason = TileXRMoonEp::kDispatchVectorFallbackApiGranularity;
        } else if (routeCount > static_cast<uint64_t>(INT16_MAX) + 1U ||
            routeCount > UINT32_MAX / sizeof(int32_t)) {
            useVectorSlotSelect = false;
            fallbackReason = TileXRMoonEp::kDispatchVectorFallbackIndexRange;
        }

        uint32_t routePlanUbBytes = kRouteTileBytes;
        uint32_t routeRankUbBytes = 0U;
        uint32_t routeIndexUbBytes = 0U;
        uint32_t compareMaskUbBytes = 0U;
        if (useVectorSlotSelect) {
            routePlanUbBytes = static_cast<uint32_t>(AlignUp(
                routeCount * sizeof(int32_t), kUbAlignBytes));
            routeRankUbBytes = routePlanUbBytes;
            routeIndexUbBytes = static_cast<uint32_t>(AlignUp(
                routeCount * sizeof(int16_t), kUbAlignBytes));
            compareMaskUbBytes = static_cast<uint32_t>(AlignUp(
                CeilDiv(routeCount, 8U), kUbAlignBytes));
            const uint64_t fixedBytes = static_cast<uint64_t>(routePlanUbBytes) +
                routeRankUbBytes + 2ULL * routeIndexUbBytes + compareMaskUbBytes +
                kDiagnosticUbBytes +
                2ULL * kDispatchUdmaIssueUbBytes;
            if (CalcPipelineBufferNum(fixedBytes, kRelayUbBytes) == 0U) {
                useVectorSlotSelect = false;
                fallbackReason = TileXRMoonEp::kDispatchVectorFallbackUbBudget;
                routePlanUbBytes = kRouteTileBytes;
                routeRankUbBytes = 0U;
                routeIndexUbBytes = 0U;
                compareMaskUbBytes = 0U;
            }
        }
        if (static_cast<uint64_t>(routePlanUbBytes) + kRelayUbBytes +
            kDiagnosticUbBytes >= kFullUbBytes) {
            return;
        }
        if (groupedPeerMode && !localOnly && !useVectorSlotSelect) {
            DispatchPublishFirstStatus(planStatus,
                TileXRMoonEp::kDispatchStatusInvalidConfig);
            return;
        }

        const uint64_t outputCopyFixedUbBytes =
            kRelayUbBytes + kDiagnosticUbBytes;
        if (outputCopyFixedUbBytes >= kFullUbBytes) {
            return;
        }
        const uint64_t outputCopyBytesPerBuffer =
            (kFullUbBytes - outputCopyFixedUbBytes) / kOutputCopyBufferNum;
        const uint32_t outputCopyTileBytes = static_cast<uint32_t>(
            outputCopyBytesPerBuffer / kUbAlignBytes * kUbAlignBytes);

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> routePlanBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> relayBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> udmaIssueQp0Buf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> udmaIssueQp1Buf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> routeRankBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> routeIndexBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> selectedRouteIndexBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> compareMaskBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> diagnosticBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> outputRelayBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> outputDiagnosticBuf;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> outputCopyQueue;

        const uint32_t routeRankAllocBytes = useVectorSlotSelect ?
            routeRankUbBytes : kUbAlignBytes;
        const uint32_t routeIndexAllocBytes = useVectorSlotSelect ?
            routeIndexUbBytes : kUbAlignBytes;
        const uint32_t compareMaskAllocBytes = useVectorSlotSelect ?
            compareMaskUbBytes : kUbAlignBytes;
        pipe.Reset();
        pipe.InitBuffer(routePlanBuf, routePlanUbBytes);
        pipe.InitBuffer(relayBuf, kRelayUbBytes);
        pipe.InitBuffer(udmaIssueQp0Buf, kDispatchUdmaIssueUbBytes);
        pipe.InitBuffer(udmaIssueQp1Buf, kDispatchUdmaIssueUbBytes);
        pipe.InitBuffer(routeRankBuf, routeRankAllocBytes);
        pipe.InitBuffer(routeIndexBuf, routeIndexAllocBytes);
        pipe.InitBuffer(selectedRouteIndexBuf, routeIndexAllocBytes);
        pipe.InitBuffer(compareMaskBuf, compareMaskAllocBytes);
        pipe.InitBuffer(diagnosticBuf, kDiagnosticUbBytes);

        AscendC::LocalTensor<int32_t> routePlanLocal = routePlanBuf.Get<int32_t>();
        AscendC::LocalTensor<uint8_t> relayLocal = relayBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> udmaIssueQp0Local =
            udmaIssueQp0Buf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> udmaIssueQp1Local =
            udmaIssueQp1Buf.Get<uint8_t>();
        AscendC::LocalTensor<int32_t> routeRankLocal = routeRankBuf.Get<int32_t>();
        AscendC::LocalTensor<int16_t> routeIndexLocal = routeIndexBuf.Get<int16_t>();
        AscendC::LocalTensor<int16_t> selectedRouteIndexLocal =
            selectedRouteIndexBuf.Get<int16_t>();
        AscendC::LocalTensor<uint8_t> compareMaskLocal = compareMaskBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> diagnosticLocal = diagnosticBuf.Get<uint8_t>();

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t kernelStartCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        const uint64_t stagingStartCycle = kernelStartCycle;
#endif
        for (uint64_t sourceRow = blockIdx; sourceRow < sourceRows; sourceRow += blockNum) {
            CopyBytesGmToGm(workspace + sourceRow * rowBytes,
                input + sourceRow * rowBytes, static_cast<uint32_t>(rowBytes), relayLocal);
        }
        AscendC::SyncAll<true>();
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t stagingEndCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        const uint64_t stagingCycles = stagingEndCycle - stagingStartCycle;
#endif

        const int32_t upstreamStatus = *planStatus;
        auto currentScratch = workspace + scratchOffset + scratchIndex * scratchSlotBytes;
        auto receiveFlags = reinterpret_cast<__gm__ uint64_t *>(
            workspace + completionFlagsOffset);
        auto signalSource = reinterpret_cast<__gm__ uint64_t *>(
            workspace + signalOffset + blockIdx * TileXRMoonEp::kDispatchSignalStrideBytes);
        if (!localOnly) {
            for (uint32_t qpIdx = 0U;
                qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
                signalSource[qpIdx] = expectedFlag;
            }
            TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(signalSource),
                TileXRMoonEp::kDispatchQpCount * sizeof(uint64_t));
        }

        AscendC::GlobalTensor<int32_t> dstGlobal;
        dstGlobal.SetGlobalBuffer(dst, routeCount);
        uint32_t dfxFlags = upstreamStatus == TileXRMoonEp::kDispatchStatusSuccess ? 0U :
            TileXRMoonEp::kDispatchDfxUpstreamPlannerError;
        uint32_t firstInvalidRouteId = UINT32_MAX;
        int32_t firstInvalidRawDst = 0;
        uint32_t firstQuietStatus = 0U;
        uint32_t firstQuietPhase = 0U;
        uint32_t timeoutPeer = UINT32_MAX;
        uint32_t timeoutPhase = UINT32_MAX;
        uint64_t timeoutObservedFlag = 0U;
        uint64_t scannedRouteCount = 0U;
        uint64_t matchedRouteCount = 0U;
        uint64_t selectedRouteCount = 0U;
        uint64_t processedRouteCount = 0U;
        uint64_t issuedPutCount = 0U;
        uint64_t issuedPutBytes = 0U;
        uint64_t visitedPeerCount = 0U;
        uint64_t completionFlagCount = 0U;
        bool requiresFinalQuiet = !useVectorSlotSelect;
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        uint64_t putIssueCycles = 0U;
        uint64_t flagWaitCycles = 0U;
        uint64_t outputCopyCycles = 0U;
        uint64_t quietCycles = 0U;
#endif

        if (useVectorSlotSelect && upstreamStatus == TileXRMoonEp::kDispatchStatusSuccess) {
            const uint32_t routePlanDataBytes = static_cast<uint32_t>(
                routeCount * sizeof(int32_t));
            const AscendC::DataCopyExtParams copyIn {1U, routePlanDataBytes, 0U, 0U, 0U};
            const AscendC::DataCopyPadExtParams<int32_t> padIn {false, 0U, 0U, 0U};
            AscendC::DataCopyPad(routePlanLocal, dstGlobal, copyIn, padIn);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            uint32_t routeShift = 0U;
            for (uint64_t value = destinationCapacity; value > 1U; value >>= 1U) {
                ++routeShift;
            }
            AscendC::ShiftRight(routeRankLocal, routePlanLocal,
                static_cast<int32_t>(routeShift), static_cast<int32_t>(routeCount));
            AscendC::CreateVecIndex(routeIndexLocal, static_cast<int16_t>(0),
                static_cast<uint32_t>(routeCount));
            AscendC::PipeBarrier<PIPE_V>();
        }

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t issueWindowStartCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
        if (useVectorSlotSelect && groupedPeerMode) {
            const uint64_t remoteScratchOffset = scratchOffset +
                scratchIndex * scratchSlotBytes;
            const uint64_t remoteFlagBase = completionFlagsOffset +
                static_cast<uint64_t>(rank) *
                    TileXRMoonEp::kDispatchQpCount * sizeof(uint64_t);
            DispatchWqeBatchInitContext initContext {};
            const bool initContextValid = localOnly ||
                InitDispatchWqeBatchInitContext(args, remoteScratchOffset,
                    scratchSlotBytes, remoteFlagBase, initContext);
            DispatchPreparedPeer previousPeer {};
            bool previousPeerValid = false;

            for (uint32_t peerWork = 0U; peerWork < peerWorkCount; ++peerWork) {
                uint32_t group = UINT32_MAX;
                uint32_t lane = UINT32_MAX;
                const int64_t peerValue = TileXRMoonEp::DispatchGroupedPeerForCore(
                    rank, rankSize, groupWidth, static_cast<uint32_t>(blockIdx),
                    static_cast<uint32_t>(blockNum), peerWork, group, lane);
                if (peerValue < 0) {
                    continue;
                }
                ++visitedPeerCount;
                if (peerValue == rank || upstreamStatus !=
                        TileXRMoonEp::kDispatchStatusSuccess) {
                    continue;
                }
                const int32_t peer = static_cast<int32_t>(peerValue);
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
                const uint64_t putIssueStartCycle =
                    static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
                DispatchPreparedPeer preparedPeer {};
                if (!initContextValid || !InitDispatchPreparedPeer(initContext,
                        peer, group, preparedPeer) ||
                    !DispatchDrainHistoricalCq(preparedPeer, relayLocal,
                        completionTimeoutTicks, dfxFlags, firstQuietStatus,
                        firstQuietPhase, timeoutPeer, timeoutPhase,
                        timeoutObservedFlag)) {
                    dfxFlags |= TileXRMoonEp::kDispatchDfxCqError;
                    break;
                }

                const uint32_t selectedCount = SelectDispatchPeerRoutes(
                    compareMaskLocal, routeRankLocal, routeIndexLocal,
                    selectedRouteIndexLocal, peer,
                    static_cast<uint32_t>(routeCount));
                scannedRouteCount += routeCount;
                matchedRouteCount += selectedCount;
                selectedRouteCount += selectedCount;

                uint32_t qpSelectedCount[TileXRMoonEp::kDispatchQpCount] = {};
                uint32_t qpSelectedStart[TileXRMoonEp::kDispatchQpCount] = {};
                bool qpSignalPending[TileXRMoonEp::kDispatchQpCount] = {true, true};
                for (uint32_t qpIdx = 0U;
                    qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
                    qpSelectedCount[qpIdx] = TileXRMoonEp::DispatchQpRouteCount(
                        selectedCount, 0U, qpIdx);
                }

                bool firstLogicalBatch = true;
                bool sendOk = true;
                while (sendOk && (qpSelectedStart[0] < qpSelectedCount[0] ||
                        qpSelectedStart[1] < qpSelectedCount[1] ||
                        qpSignalPending[0] || qpSignalPending[1])) {
                    bool finalBatch[TileXRMoonEp::kDispatchQpCount] = {};
                    for (uint32_t qpIdx = 0U;
                        sendOk && qpIdx < TileXRMoonEp::kDispatchQpCount;
                        ++qpIdx) {
                        DispatchWqeBatchState &qpState =
                            preparedPeer.qpState[qpIdx];
                        AscendC::LocalTensor<uint8_t> issueLocal = qpIdx == 0U ?
                            udmaIssueQp0Local : udmaIssueQp1Local;
                        sendOk = DispatchBuildGroupedQpBatch(qpState,
                            issueLocal, selectedRouteIndexLocal, routePlanLocal,
                            reinterpret_cast<uint64_t>(workspace), rowBytes,
                            destinationCapacity - 1U, topKMagic, topKShift, hiddenMode,
                            qpSelectedCount[qpIdx], qpSelectedStart[qpIdx],
                            qpSignalPending[qpIdx], reinterpret_cast<uint64_t>(
                                signalSource + qpIdx), 0U, finalBatch[qpIdx]);
                    }
                    for (uint32_t qpIdx = 0U;
                        sendOk && qpIdx < TileXRMoonEp::kDispatchQpCount;
                        ++qpIdx) {
                        if (preparedPeer.qpState[qpIdx].batchCount == 0U) {
                            continue;
                        }
                        sendOk = StageDispatchQpBatch(
                            preparedPeer.qpState[qpIdx],
                            qpIdx == 0U ? udmaIssueQp0Local : udmaIssueQp1Local,
                            relayLocal, finalBatch[qpIdx], group, dfxFlags,
                            firstQuietStatus, firstQuietPhase);
                    }
                    if (sendOk && firstLogicalBatch && previousPeerValid) {
                        sendOk = DispatchDrainPeerFinalCq(previousPeer,
                            relayLocal, completionTimeoutTicks, dfxFlags,
                            firstQuietStatus, firstQuietPhase, timeoutPeer,
                            timeoutPhase, timeoutObservedFlag);
                        previousPeerValid = false;
                    }
                    if (sendOk && firstLogicalBatch && creditPeerMode &&
                        TileXRMoonEp::DispatchCreditRequired(group)) {
                        uint64_t observedCredit = 0U;
                        if (!WaitDispatchPeerCredit(args, rank, peer, group,
                                magic, completionTimeoutTicks, relayLocal,
                                observedCredit)) {
                            dfxFlags |= TileXRMoonEp::kDispatchDfxCreditTimeout;
                            if (timeoutPeer == UINT32_MAX) {
                                timeoutPeer = static_cast<uint32_t>(peer);
                                timeoutPhase = group;
                                timeoutObservedFlag = observedCredit;
                            }
                            sendOk = false;
                        }
                    }
                    if (sendOk) {
                        RingDispatchPeerDoorbells(preparedPeer);
                    }
                    firstLogicalBatch = false;
                }

                if (!sendOk || preparedPeer.qpState[0].finalStaged == 0U ||
                    preparedPeer.qpState[1].finalStaged == 0U) {
                    if ((dfxFlags & (TileXRMoonEp::kDispatchDfxCreditTimeout |
                            TileXRMoonEp::kDispatchDfxCqError)) == 0U) {
                        dfxFlags |= TileXRMoonEp::kDispatchDfxInvalidConfig;
                    }
                    break;
                }

                issuedPutCount += selectedCount;
                issuedPutBytes += static_cast<uint64_t>(selectedCount) * rowBytes;
                processedRouteCount += selectedCount;
                completionFlagCount += TileXRMoonEp::kDispatchQpCount;
                previousPeer = preparedPeer;
                previousPeerValid = true;

                if (!WaitDispatchIncomingPeerAndPublishCredit(args,
                        receiveFlags, rank, rankSize, peer, group, lane,
                        groupWidth, magic, creditPeerMode,
                        completionTimeoutTicks, relayLocal, dfxFlags,
                        timeoutPeer, timeoutPhase, timeoutObservedFlag)) {
                    break;
                }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
                putIssueCycles +=
                    static_cast<uint64_t>(AscendC::GetSystemCycle()) -
                    putIssueStartCycle;
#endif
            }
            if (previousPeerValid && !DispatchDrainPeerFinalCq(previousPeer,
                    relayLocal, completionTimeoutTicks, dfxFlags,
                    firstQuietStatus, firstQuietPhase, timeoutPeer,
                    timeoutPhase, timeoutObservedFlag)) {
                dfxFlags |= TileXRMoonEp::kDispatchDfxCqError;
            }
        } else if (useVectorSlotSelect) {
            const uint64_t remoteScratchOffset = scratchOffset +
                scratchIndex * scratchSlotBytes;
            const uint64_t remoteFlagBase = completionFlagsOffset +
                static_cast<uint64_t>(rank) *
                    TileXRMoonEp::kDispatchQpCount * sizeof(uint64_t);
            DispatchWqeBatchInitContext initContext {};
            const bool initContextValid = InitDispatchWqeBatchInitContext(
                args, remoteScratchOffset, scratchSlotBytes,
                remoteFlagBase, initContext);
            DispatchPreparedPeer peerBatch[kDispatchPreparedPeerCapacity] {};
            uint64_t peerCursor = 0U;
            const uint64_t totalPeerAssignments =
                static_cast<uint64_t>(groupCount) * peerWorkCount;
            while (peerCursor < totalPeerAssignments) {
                uint32_t peerCount = 0U;
                PrepareDispatchRemotePeerBatch(
                    initContextValid ? &initContext : nullptr, rank, rankSize,
                    groupCount, peerWorkCount, static_cast<uint32_t>(blockIdx),
                    static_cast<uint32_t>(blockNum), peerCursor, peerBatch,
                    peerCount, visitedPeerCount);
                for (uint32_t peerIndex = 0U;
                    peerIndex < peerCount; ++peerIndex) {
                    DispatchPreparedPeer &preparedPeer = peerBatch[peerIndex];
                    const int32_t peer = preparedPeer.targetRank;
                    const uint32_t issuePhase = preparedPeer.issuePhase;
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
                    const uint64_t putIssueStartCycle =
                        static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
                    uint32_t selectedCount = 0U;
                    if (upstreamStatus == TileXRMoonEp::kDispatchStatusSuccess) {
                        selectedCount = SelectDispatchPeerRoutes(
                            compareMaskLocal, routeRankLocal, routeIndexLocal,
                            selectedRouteIndexLocal, peer,
                            static_cast<uint32_t>(routeCount));
                        scannedRouteCount += routeCount;
                        matchedRouteCount += selectedCount;
                        selectedRouteCount += selectedCount;
                    }
                    const uint64_t peerWqeCount = selectedCount;
                    uint64_t issuedPeerWqeCount = 0U;
                    bool signalSubmitted = false;
                    if (upstreamStatus == TileXRMoonEp::kDispatchStatusSuccess) {
                        const bool batchInitialized =
                            preparedPeer.initialized;
                        if (batchInitialized) {
                            bool batchOk = true;
                            constexpr uint32_t sequencePhase = 0U;
                            for (uint32_t qpIdx = 0U;
                                batchOk &&
                                    qpIdx < TileXRMoonEp::kDispatchQpCount;
                                ++qpIdx) {
                                const uint32_t qpRouteCount =
                                    TileXRMoonEp::DispatchQpRouteCount(
                                        selectedCount, sequencePhase, qpIdx);
                                batchOk = AppendDispatchWqes(
                                    preparedPeer.qpState[qpIdx],
                                    qpIdx == 0U ? udmaIssueQp0Local :
                                        udmaIssueQp1Local,
                                    relayLocal,
                                    selectedRouteIndexLocal, routePlanLocal,
                                    reinterpret_cast<uint64_t>(workspace),
                                    rowBytes, destinationCapacity - 1U,
                                    topKMagic, topKShift, hiddenMode,
                                    qpRouteCount, true,
                                    reinterpret_cast<uint64_t>(
                                        signalSource + qpIdx),
                                    issuePhase, dfxFlags, firstQuietStatus,
                                    firstQuietPhase, sequencePhase);
                            }
                            if (batchOk) {
                                issuedPutCount += selectedCount;
                                issuedPutBytes += selectedCount * rowBytes;
                                processedRouteCount += selectedCount;
                                signalSubmitted = true;
                            } else {
                                dfxFlags |= TileXRMoonEp::kDispatchDfxQuietError;
                                if (firstQuietStatus == 0U) {
                                    firstQuietStatus = 0xFFFFFFFDU;
                                    firstQuietPhase = issuePhase;
                                }
                            }
                        } else {
                            for (uint32_t selected = 0U;
                                selected < selectedCount; ++selected) {
                                const uint64_t routeId = static_cast<uint16_t>(
                                    selectedRouteIndexLocal.GetValue(selected));
                                if (routeId >= routeCount) {
                                    RecordInvalidRoute(routeId, 0, dfxFlags,
                                        firstInvalidRouteId,
                                        firstInvalidRawDst);
                                    continue;
                                }
                                const int32_t rawDst =
                                    routePlanLocal.GetValue(routeId);
                                int32_t targetRank = -1;
                                uint64_t targetSlot = 0U;
                                if (!DecodeSendDst(rawDst, destinationCapacity, rankSize,
                                        targetRank, targetSlot) ||
                                    targetRank != peer) {
                                    RecordInvalidRoute(routeId, rawDst, dfxFlags,
                                        firstInvalidRouteId,
                                        firstInvalidRawDst);
                                    continue;
                                }
                                const uint64_t sourceRow = hiddenMode ?
                                    routeId / static_cast<uint64_t>(k) : routeId;
                                TileXR::UDMAPutNbiOnQpWithFlagDeferred<uint8_t>(
                                    args, udmaIssueQp0Local, targetRank, 0U,
                                    workspace + sourceRow * rowBytes,
                                    remoteScratchOffset + targetSlot * rowBytes,
                                    static_cast<uint32_t>(rowBytes),
                                    TileXR::TILEXR_UDMA_SQE_FLAG_COMPLETION);
                                ++issuedPeerWqeCount;
                                ++issuedPutCount;
                                issuedPutBytes += rowBytes;
                                ++processedRouteCount;
                                ReclaimDeferredSegment(args, targetRank, 0U,
                                    issuedPeerWqeCount, issuePhase, dfxFlags,
                                    firstQuietStatus, firstQuietPhase);
                                if (ShouldFlushPartialDoorbell(
                                        issuedPeerWqeCount, peerWqeCount)) {
                                    TileXR::UDMAFlushQpDoorbell(
                                        args, targetRank, 0U);
                                }
                            }
                        }
                    }
                    if (!signalSubmitted) {
                        requiresFinalQuiet = true;
                        for (uint32_t qpIdx = 0U;
                            qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
                            TileXR::UDMAPutNbiOnQpWithFlagDeferred<uint64_t>(
                                args, qpIdx == 0U ? udmaIssueQp0Local :
                                    udmaIssueQp1Local,
                                peer, qpIdx, signalSource + qpIdx,
                                remoteFlagBase + qpIdx * sizeof(uint64_t),
                                static_cast<uint32_t>(sizeof(uint64_t)),
                                TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
                            TileXR::UDMAFlushQpDoorbell(args, peer, qpIdx);
                        }
                    }
                    completionFlagCount += TileXRMoonEp::kDispatchQpCount;
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
                    putIssueCycles +=
                        static_cast<uint64_t>(AscendC::GetSystemCycle()) -
                        putIssueStartCycle;
#endif
                }
            }

        } else {
            for (int64_t issuePhase = 0; issuePhase < groupCount; ++issuePhase) {
                for (uint32_t peerWork = 0U;
                    peerWork < peerWorkCount; ++peerWork) {
                    const int64_t peer = TileXRMoonEp::DispatchPeerForCore(
                        rank, rankSize, issuePhase,
                        static_cast<uint32_t>(blockIdx),
                        static_cast<uint32_t>(blockNum), peerWork);
                    if (peer < 0) {
                        continue;
                    }
                    ++visitedPeerCount;
                    if (peer == rank) {
                        continue;
                    }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
                    const uint64_t putIssueStartCycle =
                        static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
                    uint64_t peerWqeCount = 0U;
                    uint64_t issuedPeerWqeCount = 0U;
                    if (upstreamStatus ==
                        TileXRMoonEp::kDispatchStatusSuccess) {
                        for (uint64_t tileStart = 0U;
                            tileStart < routeCount;
                            tileStart += kRouteTileElements) {
                            const uint32_t tileElements = LoadRouteTile(
                                dstGlobal, tileStart, routeCount,
                                routePlanLocal);
                            scannedRouteCount += tileElements;
                            for (uint32_t tileRoute = 0U;
                                tileRoute < tileElements; ++tileRoute) {
                                const uint64_t routeId = tileStart + tileRoute;
                                const int32_t encoded =
                                    routePlanLocal.GetValue(tileRoute);
                                int32_t targetRank = -1;
                                uint64_t targetSlot = 0U;
                                if (!DecodeSendDst(encoded, destinationCapacity,
                                        rankSize, targetRank, targetSlot)) {
                                    RecordInvalidRoute(routeId,
                                        DecodeRawDst(encoded), dfxFlags,
                                        firstInvalidRouteId,
                                        firstInvalidRawDst);
                                    continue;
                                }
                                if (targetRank == peer) {
                                    ++peerWqeCount;
                                    ++matchedRouteCount;
                                }
                            }
                        }
                        selectedRouteCount += peerWqeCount;
                        for (uint64_t tileStart = 0U;
                            tileStart < routeCount;
                            tileStart += kRouteTileElements) {
                            const uint32_t tileElements = LoadRouteTile(
                                dstGlobal, tileStart, routeCount,
                                routePlanLocal);
                            scannedRouteCount += tileElements;
                            for (uint32_t tileRoute = 0U;
                                tileRoute < tileElements; ++tileRoute) {
                                const uint64_t routeId = tileStart + tileRoute;
                                const int32_t encoded =
                                    routePlanLocal.GetValue(tileRoute);
                                int32_t targetRank = -1;
                                uint64_t targetSlot = 0U;
                                if (!DecodeSendDst(encoded, destinationCapacity,
                                        rankSize, targetRank, targetSlot) ||
                                    targetRank != peer) {
                                    continue;
                                }
                                const uint64_t sourceRow = hiddenMode ?
                                    routeId / static_cast<uint64_t>(k) :
                                    routeId;
                                TileXR::UDMAPutNbiOnQpWithFlagDeferred<uint8_t>(
                                    args, udmaIssueQp0Local, targetRank, 0U,
                                    workspace + sourceRow * rowBytes,
                                    scratchOffset +
                                        scratchIndex * scratchSlotBytes +
                                        targetSlot * rowBytes,
                                    static_cast<uint32_t>(rowBytes),
                                    TileXR::TILEXR_UDMA_SQE_FLAG_COMPLETION);
                                ++issuedPeerWqeCount;
                                ++issuedPutCount;
                                issuedPutBytes += rowBytes;
                                ++processedRouteCount;
                                ReclaimDeferredSegment(args, targetRank, 0U,
                                    issuedPeerWqeCount,
                                    static_cast<uint32_t>(issuePhase),
                                    dfxFlags, firstQuietStatus,
                                    firstQuietPhase);
                                if (ShouldFlushPartialDoorbell(
                                        issuedPeerWqeCount, peerWqeCount)) {
                                    TileXR::UDMAFlushQpDoorbell(
                                        args, targetRank, 0U);
                                }
                            }
                        }
                    }

                    if (peer != rank) {
                        const uint64_t remoteFlagBase = completionFlagsOffset +
                            static_cast<uint64_t>(rank) *
                                TileXRMoonEp::kDispatchQpCount *
                                sizeof(uint64_t);
                        for (uint32_t qpIdx = 0U;
                            qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
                            TileXR::UDMAPutNbiOnQpWithFlagDeferred<uint64_t>(
                                args, qpIdx == 0U ? udmaIssueQp0Local :
                                    udmaIssueQp1Local,
                                static_cast<int32_t>(peer), qpIdx,
                                signalSource + qpIdx,
                                remoteFlagBase + qpIdx * sizeof(uint64_t),
                                static_cast<uint32_t>(sizeof(uint64_t)),
                                TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
                            TileXR::UDMAFlushQpDoorbell(
                                args, static_cast<int32_t>(peer), qpIdx);
                        }
                        completionFlagCount +=
                            TileXRMoonEp::kDispatchQpCount;
                    }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
                    putIssueCycles +=
                        static_cast<uint64_t>(AscendC::GetSystemCycle()) -
                        putIssueStartCycle;
#endif
                }
            }
        }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t remoteIssueEndCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
        if (upstreamStatus == TileXRMoonEp::kDispatchStatusSuccess) {
            if (useVectorSlotSelect) {
                const uint32_t localRouteCount = SelectDispatchPeerRoutes(
                    compareMaskLocal, routeRankLocal, routeIndexLocal,
                    selectedRouteIndexLocal, rank,
                    static_cast<uint32_t>(routeCount));
                scannedRouteCount += routeCount;
                uint64_t localStart = 0U;
                uint64_t localEnd = 0U;
                TileXRMoonEp::DispatchContiguousRange(localRouteCount,
                    static_cast<uint32_t>(blockNum),
                    static_cast<uint32_t>(blockIdx), localStart, localEnd);
                const uint64_t localCount = localEnd - localStart;
                matchedRouteCount += localCount;
                selectedRouteCount += localCount;
                for (uint64_t selected = localStart;
                    selected < localEnd; ++selected) {
                    const uint64_t routeId = static_cast<uint16_t>(
                        selectedRouteIndexLocal.GetValue(selected));
                    const int32_t rawDst = routePlanLocal.GetValue(routeId);
                    int32_t targetRank = -1;
                    uint64_t targetSlot = 0U;
                    if (!DecodeSendDst(rawDst, destinationCapacity, rankSize,
                            targetRank, targetSlot) || targetRank != rank) {
                        RecordInvalidRoute(routeId, rawDst, dfxFlags,
                            firstInvalidRouteId, firstInvalidRawDst);
                        continue;
                    }
                    const uint64_t sourceRow = hiddenMode ?
                        routeId / static_cast<uint64_t>(k) : routeId;
                    CopyBytesGmToGm(currentScratch + targetSlot * rowBytes,
                        workspace + sourceRow * rowBytes,
                        static_cast<uint32_t>(rowBytes), relayLocal);
                    ++processedRouteCount;
                }
            } else {
                uint64_t localStart = 0U;
                uint64_t localEnd = 0U;
                TileXRMoonEp::DispatchContiguousRange(routeCount,
                    static_cast<uint32_t>(blockNum),
                    static_cast<uint32_t>(blockIdx), localStart, localEnd);
                for (uint64_t tileStart = localStart;
                    tileStart < localEnd; tileStart += kRouteTileElements) {
                    const uint32_t tileElements = LoadRouteTile(
                        dstGlobal, tileStart, localEnd, routePlanLocal);
                    scannedRouteCount += tileElements;
                    for (uint32_t tileRoute = 0U;
                        tileRoute < tileElements; ++tileRoute) {
                        const uint64_t routeId = tileStart + tileRoute;
                        const int32_t rawDst =
                            routePlanLocal.GetValue(tileRoute);
                        int32_t targetRank = -1;
                        uint64_t targetSlot = 0U;
                        if (!DecodeSendDst(rawDst, destinationCapacity, rankSize,
                                targetRank, targetSlot)) {
                            RecordInvalidRoute(routeId, rawDst, dfxFlags,
                                firstInvalidRouteId, firstInvalidRawDst);
                            continue;
                        }
                        if (targetRank != rank) {
                            continue;
                        }
                        ++matchedRouteCount;
                        ++selectedRouteCount;
                        const uint64_t sourceRow = hiddenMode ?
                            routeId / static_cast<uint64_t>(k) : routeId;
                        CopyBytesGmToGm(
                            currentScratch + targetSlot * rowBytes,
                            workspace + sourceRow * rowBytes,
                            static_cast<uint32_t>(rowBytes), relayLocal);
                        ++processedRouteCount;
                    }
                }
            }
        }

        // Finish every send-side consumer before reusing the send UB layout.
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe.Reset();
        pipe.InitBuffer(outputRelayBuf, kRelayUbBytes);
        pipe.InitBuffer(outputDiagnosticBuf, kDiagnosticUbBytes);
        if (outputCopyTileBytes != 0U) {
            pipe.InitBuffer(outputCopyQueue,
                static_cast<uint8_t>(kOutputCopyBufferNum),
                outputCopyTileBytes);
        }
        relayLocal = outputRelayBuf.Get<uint8_t>();
        diagnosticLocal = outputDiagnosticBuf.Get<uint8_t>();

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t issueWindowEndCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
        const uint64_t flagWaitStartCycle = issueWindowEndCycle;
#else
        const uint64_t flagWaitStartCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
        if (!groupedPeerMode) {
            for (int64_t completionPhase = 0; completionPhase < groupCount;
                ++completionPhase) {
                for (uint32_t peerWork = 0U;
                    peerWork < peerWorkCount; ++peerWork) {
                    const int64_t peer = TileXRMoonEp::DispatchPeerForCore(
                        rank, rankSize, completionPhase,
                        static_cast<uint32_t>(blockIdx),
                        static_cast<uint32_t>(blockNum), peerWork);
                    if (peer >= 0 && peer != rank) {
                        const uint64_t peerFlagBase = static_cast<uint64_t>(peer) *
                            TileXRMoonEp::kDispatchQpCount;
                        for (uint32_t qpIdx = 0U;
                            qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
                            uint64_t observed = 0U;
                            if (!WaitCompletionFlag(
                                    receiveFlags + peerFlagBase + qpIdx,
                                    expectedFlag, flagWaitStartCycle,
                                    completionTimeoutTicks, observed,
                                    relayLocal)) {
                                dfxFlags |= TileXRMoonEp::
                                    kDispatchDfxCompletionTimeout;
                                if (timeoutPeer == UINT32_MAX) {
                                    timeoutPeer = static_cast<uint32_t>(peer);
                                    timeoutPhase = static_cast<uint32_t>(
                                        completionPhase);
                                    timeoutObservedFlag = observed;
                                }
                            }
                        }
                    }
                }
            }
        }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t flagWaitEndCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        if (!groupedPeerMode) {
            flagWaitCycles = flagWaitEndCycle - flagWaitStartCycle;
        }
#endif
        if (selectedRouteCount != processedRouteCount) {
            dfxFlags |= TileXRMoonEp::kDispatchDfxRouteCountMismatch;
        }
        if (upstreamStatus == TileXRMoonEp::kDispatchStatusSuccess &&
            !ClearDispatchZeroFillRanges(currentScratch, zeroFillRanges,
                zeroFillRangeCount, destinationCapacity, rowBytes,
                blockIdx, blockNum, relayLocal)) {
            dfxFlags |= TileXRMoonEp::kDispatchDfxInvalidConfig;
        }
        const int32_t localExecutionStatus = StatusFromDfxFlags(dfxFlags);
        DispatchPublishFirstStatus(planStatus, localExecutionStatus);
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_DFX)
        WriteDfxRecord(workspace + dfxOffset, static_cast<uint32_t>(payloadMode),
            static_cast<uint32_t>(rank), static_cast<uint32_t>(blockIdx), dfxFlags,
            firstInvalidRouteId, firstInvalidRawDst, firstQuietStatus, firstQuietPhase,
            timeoutPeer, timeoutPhase, routeCount, processedRouteCount, expectedFlag,
            expectedFlag, timeoutObservedFlag, diagnosticLocal);
#endif
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t dfxWriteEndCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif

        AscendC::SyncAll<true>();
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t syncAllEndCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_DFX)
        uint32_t globalDfxFlags = 0U;
        auto allDfx = reinterpret_cast<__gm__ TileXRMoonEp::DispatchDfxRecord *>(
            workspace + dfxOffset);
        for (uint32_t core = 0U; core < static_cast<uint32_t>(blockNum); ++core) {
            globalDfxFlags |= allDfx[core].flags;
        }
        const int32_t executionStatus = StatusFromDfxFlags(globalDfxFlags);
#else
        const int32_t executionStatus = *planStatus;
#endif
        auto kernelStatus = reinterpret_cast<__gm__ TileXRMoonEp::DispatchKernelStatus *>(
            workspace + kernelStatusOffset);
        if (blockIdx == 0U) {
            kernelStatus->marker = TileXRMoonEp::kDispatchKernelStatusMarker;
            kernelStatus->version = TileXRMoonEp::kDispatchDiagnosticVersion;
            kernelStatus->recordBytes = sizeof(TileXRMoonEp::DispatchKernelStatus);
            kernelStatus->status = executionStatus;
            kernelStatus->payloadMode = static_cast<uint32_t>(payloadMode);
            kernelStatus->magic = expectedFlag;
            for (uint32_t index = 0U; index < 5U; ++index) {
                kernelStatus->reserved[index] = 0U;
            }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_DFX)
            kernelStatus->reserved[0] |=
                TileXRMoonEp::kDispatchKernelStatusFeatureDfxEnabled;
#endif
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
            kernelStatus->reserved[0] |=
                TileXRMoonEp::kDispatchKernelStatusFeatureProfilingEnabled;
#endif
        }

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t outputCopyStartCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif
        if (executionStatus == TileXRMoonEp::kDispatchStatusSuccess) {
            uint64_t outputStartSlot = 0U;
            uint64_t outputEndSlot = 0U;
            TileXRMoonEp::DispatchContiguousRange(destinationCapacity,
                static_cast<uint32_t>(blockNum),
                static_cast<uint32_t>(blockIdx),
                outputStartSlot, outputEndSlot);
            const uint64_t outputSlotCount = outputEndSlot - outputStartSlot;
            if (outputCopyTileBytes != 0U) {
                CopyContiguousBytesGmToGmPipelined(
                    output + outputStartSlot * rowBytes,
                    currentScratch + outputStartSlot * rowBytes,
                    outputSlotCount * rowBytes, outputCopyTileBytes,
                    outputCopyQueue);
            } else {
                for (uint64_t targetSlot = outputStartSlot;
                    targetSlot < outputEndSlot; ++targetSlot) {
                    CopyBytesGmToGm(output + targetSlot * rowBytes,
                        currentScratch + targetSlot * rowBytes,
                        static_cast<uint32_t>(rowBytes), relayLocal);
                }
            }
        }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t outputCopyEndCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        outputCopyCycles = outputCopyEndCycle - outputCopyStartCycle;
        const uint64_t quietStartCycle =
            static_cast<uint64_t>(AscendC::GetSystemCycle());
#endif

        if (requiresFinalQuiet) {
            for (int64_t completionPhase = 0;
                completionPhase < groupCount; ++completionPhase) {
                for (uint32_t peerWork = 0U;
                    peerWork < peerWorkCount; ++peerWork) {
                    const int64_t peer = TileXRMoonEp::DispatchPeerForCore(
                        rank, rankSize, completionPhase,
                        static_cast<uint32_t>(blockIdx),
                        static_cast<uint32_t>(blockNum), peerWork);
                    if (peer < 0 || peer == rank) {
                        continue;
                    }
                    for (uint32_t qpIdx = 0U;
                        qpIdx < TileXRMoonEp::kDispatchQpCount; ++qpIdx) {
                        const uint32_t quietStatus = TileXR::UDMAQuietStatusOnQp(
                            args, static_cast<int32_t>(peer), qpIdx);
                        if (quietStatus != 0U && firstQuietStatus == 0U) {
                            firstQuietStatus = quietStatus;
                            firstQuietPhase =
                                static_cast<uint32_t>(completionPhase);
                            dfxFlags |= TileXRMoonEp::kDispatchDfxQuietError;
                            DispatchPublishFirstStatus(planStatus,
                                TileXRMoonEp::kDispatchStatusQuietError);
                        }
                    }
                }
            }
        }
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t quietEndCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        quietCycles = quietEndCycle - quietStartCycle;
#endif
#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_DFX)
        WriteDfxRecord(workspace + dfxOffset, static_cast<uint32_t>(payloadMode),
            static_cast<uint32_t>(rank), static_cast<uint32_t>(blockIdx), dfxFlags,
            firstInvalidRouteId, firstInvalidRawDst, firstQuietStatus, firstQuietPhase,
            timeoutPeer, timeoutPhase, routeCount, processedRouteCount, expectedFlag,
            expectedFlag, timeoutObservedFlag, diagnosticLocal);
#endif

#if defined(TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING)
        const uint64_t kernelCycles = static_cast<uint64_t>(AscendC::GetSystemCycle()) -
            kernelStartCycle;
        WriteProfileRecord(workspace + profileOffset, static_cast<uint32_t>(payloadMode),
            static_cast<uint32_t>(rank), static_cast<uint32_t>(blockIdx),
            static_cast<uint32_t>(blockNum), dfxFlags, expectedFlag,
            static_cast<uint32_t>(scratchIndex), static_cast<uint32_t>(groupCount),
            useVectorSlotSelect ? TileXRMoonEp::kDispatchSelectVector :
                TileXRMoonEp::kDispatchSelectScalarTiled,
            fallbackReason, scannedRouteCount, matchedRouteCount, selectedRouteCount,
            processedRouteCount, issuedPutCount, issuedPutBytes, visitedPeerCount,
            completionFlagCount, kernelCycles, stagingCycles, putIssueCycles,
            flagWaitCycles, outputCopyCycles, quietCycles,
            stagingEndCycle - kernelStartCycle,
            issueWindowStartCycle - kernelStartCycle,
            remoteIssueEndCycle - kernelStartCycle,
            issueWindowEndCycle - kernelStartCycle,
            flagWaitStartCycle - kernelStartCycle,
            flagWaitEndCycle - kernelStartCycle,
            dfxWriteEndCycle - kernelStartCycle,
            syncAllEndCycle - kernelStartCycle,
            outputCopyStartCycle - kernelStartCycle,
            outputCopyEndCycle - kernelStartCycle,
            quietEndCycle - kernelStartCycle, diagnosticLocal);
#endif
    }
}
