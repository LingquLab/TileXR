/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_MOONEP_COMBINE_V2_KERNEL_H
#define TILEXR_MOONEP_COMBINE_V2_KERNEL_H

#include <cstdint>

#include "comm_args.h"
#include "combine_v2_profile.h"
#include "kernel_operator.h"
#include "combine_v2_schedule.h"
#include "combine_v2_wqe_batch.h"
#include "tilexr_udma.h"
#include "tilexr_udma_fullmesh.h"

namespace {

using namespace AscendC;

constexpr uint32_t kUbAlignBytes = 32U;
constexpr uint32_t kFullUbBytes = 216U * 1024U;
constexpr uint32_t kWqeBytes = 64U;
constexpr uint32_t kControlWqesPerLane = 2U;
constexpr uint32_t kSixPortPayloadCapacity = 96U;
constexpr uint32_t kTwoPortPayloadCapacity = 32U;
constexpr uint32_t kSixPortIssueCapacity = 98U;
constexpr uint32_t kTwoPortIssueCapacity = 34U;
constexpr uint32_t kTotalIssueCapacity =
    kSixPortIssueCapacity + kTwoPortIssueCapacity;
constexpr uint32_t kFullmeshIssueCapacity =
    TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows + 1U;
constexpr uint32_t kSixPortIssueBytes = kSixPortIssueCapacity * kWqeBytes;
constexpr uint32_t kTwoPortIssueBytes = kTwoPortIssueCapacity * kWqeBytes;
constexpr uint32_t kTotalIssueBytes = kTotalIssueCapacity * kWqeBytes;
constexpr uint32_t kDstSlotBytes =
    TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows * sizeof(int32_t);
constexpr uint32_t kSelectionIndexBytes =
    TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows * sizeof(int16_t);
constexpr uint32_t kSelectionMaskBytes =
    ((TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows + 7U) / 8U +
        kUbAlignBytes - 1U) / kUbAlignBytes * kUbAlignBytes;
constexpr uint32_t kSelectionMaskElements =
    kSelectionMaskBytes / sizeof(uint16_t);
constexpr uint32_t kWqeContextBytes = 256U;
constexpr uint32_t kSendBufferBytes = kDstSlotBytes + kTotalIssueBytes +
    2U * kSelectionIndexBytes + 2U * kSelectionMaskBytes + kWqeContextBytes +
    2U * TileXRMoonEp::kMoonEpCombineV2SelfRelayHalfBytes;
constexpr uint32_t kReduceMaxInputBuffers = 8U;
constexpr uint32_t kReduceOutputBuffers = 2U;
constexpr uint32_t kReduceMaxRowBytes = 14U * 1024U;
constexpr uint32_t kReduceMaxRowElements =
    kReduceMaxRowBytes / sizeof(bfloat16_t);
constexpr uint64_t kOperationTimeoutCycles = 10000000000ULL;
constexpr uint32_t kPollNoCompletion = UINT32_MAX;
constexpr uint32_t kPollInvalidState = UINT32_MAX - 1U;
// Trusted-input benchmark mode compiles defensive validation and diagnostics
// out of the transfer hot path.
constexpr bool kEnableSafetyChecks = false;
constexpr bool kEnableFullSync = true;
constexpr uint32_t kCollectiveInitStage = 15U;
constexpr uint32_t kCollectiveValidationStage = 14U;
constexpr uint32_t kCollectiveDoneStage = 13U;
constexpr uint32_t kCollectiveReduceStage = 12U;
constexpr TileXRMoonEp::MoonEpCombineV2ScheduleMode
    kCombineV2ScheduleMode =
        TileXRMoonEp::MOONEP_COMBINE_V2_BIDIRECTIONAL_RING;

static_assert(sizeof(TileXR::UDMASqeCtx) + sizeof(TileXR::UDMASgeCtx) ==
    kWqeBytes, "Combine V2 WRITE WQE must occupy one basic block");
static_assert(sizeof(TileXR::UDMACqeCtx) == 64U,
    "Combine V2 CQE must occupy one cache line");
static_assert(TileXRMoonEp::kMoonEpCombineV2ProfileStepCount ==
        TileXRMoonEp::kMoonEpCombineV2StepCount,
    "Combine V2 profile must cover every schedule step");
static_assert(TileXRMoonEp::kMoonEpCombineV2MaxSelectedPayloadWqes % 4U == 0U,
    "Combine V2 maximum payload count must preserve the 3:1 QP split");
static_assert(TileXRMoonEp::kMoonEpCombineV2MaxSelectedPayloadWqes * 3U / 4U ==
        kSixPortPayloadCapacity,
    "Combine V2 six-port payload capacity must preserve the 3:1 QP split");
static_assert(TileXRMoonEp::kMoonEpCombineV2MaxSelectedPayloadWqes / 4U ==
        kTwoPortPayloadCapacity,
    "Combine V2 two-port payload capacity must preserve the 3:1 QP split");
static_assert(kSixPortPayloadCapacity + kControlWqesPerLane ==
        kSixPortIssueCapacity,
    "Combine V2 six-port issue buffer cannot hold payload plus controls");
static_assert(kTwoPortPayloadCapacity + kControlWqesPerLane ==
        kTwoPortIssueCapacity,
    "Combine V2 two-port issue buffer cannot hold payload plus controls");
static_assert(kTotalIssueCapacity == 132U,
    "Combine V2 issue buffer must contain 128 payload and four controls");
static_assert(kFullmeshIssueCapacity <= kTotalIssueCapacity,
    "Combine V2 issue buffer cannot hold Fullmesh payload plus done");
static_assert(kSelectionMaskBytes == 1024U,
    "Combine V2 selection mask size changed unexpectedly");
static_assert(kSendBufferBytes == 207360U,
    "Combine V2 send buffer layout must occupy 207360 bytes");
static_assert(kFullUbBytes - kSendBufferBytes == 13824U,
    "Combine V2 send buffer layout must leave 13824 bytes of UB");

template <HardEvent event> __aicore__ inline void SyncFunc()
{
    const TEventID eventId = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

__aicore__ inline bool TimedOut(uint64_t startCycles)
{
    return static_cast<uint64_t>(GetSystemCycle()) - startCycles >=
        kOperationTimeoutCycles;
}

struct MoonEpCombineV2RemoteFields {
    uint64_t remoteRowBase;
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    uint32_t tokenEn;
    uint32_t rmtJettyType;
    uint32_t targetHint;
    uint32_t tpId;
    uint32_t rmtJettyOrSegId;
    uint32_t rmtTokenValue;
};

struct alignas(32) MoonEpCombineV2OperatorFields {
    uint64_t rowBytes;
    uint64_t reserved[3];
};

struct alignas(32) MoonEpCombineV2PeerFields {
    uint64_t rowBytes;
    uint64_t reserved[3];
    MoonEpCombineV2RemoteFields remote[
        TileXRMoonEp::kMoonEpCombineV2LaneCount];
};

struct alignas(32) MoonEpCombineV2BuildContext {
    uint64_t localRowBase;
    uint64_t rowBytes;
    uint64_t remoteRowBase[TileXRMoonEp::kMoonEpCombineV2LaneCount];
    uint64_t peerBase;
    uint32_t chunkStart;
    uint32_t batchOffset;
    uint32_t batchCount;
    uint32_t sequencePhase;
    uint32_t head[TileXRMoonEp::kMoonEpCombineV2LaneCount];
};

struct alignas(32) MoonEpCombineV2FullSyncBuildContext {
    uint64_t localSignalAddr;
    uint32_t head;
    uint32_t count;
    MoonEpCombineV2RemoteFields remote[
        TileXRMoonEp::kMoonEpCombineV2FullSyncMaxPeersPerCore];
};

static_assert(sizeof(MoonEpCombineV2OperatorFields) <= kWqeContextBytes,
    "Combine V2 operator fields exceed their UB allocation");
static_assert(sizeof(MoonEpCombineV2PeerFields) <= kWqeContextBytes,
    "Combine V2 peer fields exceed their UB allocation");
static_assert(sizeof(MoonEpCombineV2BuildContext) <= kWqeContextBytes,
    "Combine V2 build context exceeds its UB allocation");
static_assert(sizeof(MoonEpCombineV2FullSyncBuildContext) <= kDstSlotBytes,
    "Combine V2 full-sync build context exceeds its UB allocation");

struct MoonEpCombineV2LaneState {
    __gm__ TileXR::UDMAWQCtx *sq;
    __gm__ TileXR::UDMACQCtx *cq;
    uint32_t qp;
    uint32_t head;
    uint32_t tail;
    uint32_t completionCount;
    uint32_t cqTail;
    uint32_t cqTarget;
    uint32_t submittedHead;
};

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
__simt_vf__ __aicore__ LAUNCH_BOUND(
    TileXRMoonEp::kMoonEpCombineV2BuilderThreads)
inline void MoonEpCombineV2PrefillOperatorWqesVf(
    __ubuf__ uint8_t *sixPortWqes, __ubuf__ uint8_t *twoPortWqes,
    __ubuf__ const MoonEpCombineV2OperatorFields *fields)
{
    constexpr uint32_t totalCapacity =
        kSixPortPayloadCapacity + kTwoPortPayloadCapacity;
    for (uint32_t task = static_cast<uint32_t>(threadIdx.x);
        task < totalCapacity;
        task += TileXRMoonEp::kMoonEpCombineV2BuilderThreads) {
        __ubuf__ uint8_t *wqe = task < kSixPortPayloadCapacity ?
            sixPortWqes + task * kWqeBytes :
            twoPortWqes + (task - kSixPortPayloadCapacity) * kWqeBytes;
        __ubuf__ uint32_t *words =
            reinterpret_cast<__ubuf__ uint32_t *>(wqe);
        for (uint32_t word = 0U; word < kWqeBytes / sizeof(uint32_t);
            ++word) {
            words[word] = 0U;
        }
        __ubuf__ TileXR::UDMASqeCtx *sqe =
            reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
        sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
        sqe->flag = 0U;
        sqe->nf = 0U;
        sqe->inlineMsgLen = 0U;
        sqe->sgeNum = 1U;
        __ubuf__ TileXR::UDMASgeCtx *sge =
            reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
                wqe + sizeof(TileXR::UDMASqeCtx));
        sge->len = static_cast<uint32_t>(fields->rowBytes);
        sge->tokenId = 0U;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(
    TileXRMoonEp::kMoonEpCombineV2BuilderThreads)
inline void MoonEpCombineV2PrefillPeerWqesVf(
    __ubuf__ uint8_t *sixPortWqes, __ubuf__ uint8_t *twoPortWqes,
    __ubuf__ const MoonEpCombineV2PeerFields *fields)
{
    constexpr uint32_t totalCapacity =
        kSixPortPayloadCapacity + kTwoPortPayloadCapacity;
    for (uint32_t task = static_cast<uint32_t>(threadIdx.x);
        task < totalCapacity;
        task += TileXRMoonEp::kMoonEpCombineV2BuilderThreads) {
        const uint32_t lane = task < kSixPortPayloadCapacity ?
            TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT :
            TileXRMoonEp::MOONEP_COMBINE_V2_TWO_PORT;
        const uint32_t laneIndex = task < kSixPortPayloadCapacity ? task :
            task - kSixPortPayloadCapacity;
        __ubuf__ uint8_t *wqe =
            (lane == TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT ?
                sixPortWqes : twoPortWqes) + laneIndex * kWqeBytes;
        __ubuf__ uint32_t *words =
            reinterpret_cast<__ubuf__ uint32_t *>(wqe);
        for (uint32_t word = 0U; word < kWqeBytes / sizeof(uint32_t);
            ++word) {
            words[word] = 0U;
        }
        __ubuf__ TileXR::UDMASqeCtx *sqe =
            reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
        sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
        sqe->flag = 0U;
        sqe->nf = 0U;
        sqe->tokenEn = fields->remote[lane].tokenEn;
        sqe->rmtJettyType = fields->remote[lane].rmtJettyType;
        sqe->targetHint = fields->remote[lane].targetHint;
        sqe->inlineMsgLen = 0U;
        sqe->tpId = fields->remote[lane].tpId;
        sqe->sgeNum = 1U;
        sqe->rmtJettyOrSegId = fields->remote[lane].rmtJettyOrSegId;
        sqe->rmtTokenValue = fields->remote[lane].rmtTokenValue;
        sqe->rmtEidL = fields->remote[lane].rmtEidL;
        sqe->rmtEidH = fields->remote[lane].rmtEidH;
        __ubuf__ TileXR::UDMASgeCtx *sge =
            reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
                wqe + sizeof(TileXR::UDMASqeCtx));
        sge->len = static_cast<uint32_t>(fields->rowBytes);
        sge->tokenId = 0U;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(
    TileXRMoonEp::kMoonEpCombineV2BuilderThreads)
inline void MoonEpCombineV2PrefillFullmeshWqesVf(
    __ubuf__ uint8_t *wqes,
    __ubuf__ const MoonEpCombineV2PeerFields *fields)
{
    for (uint32_t task = static_cast<uint32_t>(threadIdx.x);
        task < TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows;
        task += TileXRMoonEp::kMoonEpCombineV2BuilderThreads) {
        __ubuf__ uint8_t *wqe = wqes + task * kWqeBytes;
        __ubuf__ uint32_t *words =
            reinterpret_cast<__ubuf__ uint32_t *>(wqe);
        for (uint32_t word = 0U; word < kWqeBytes / sizeof(uint32_t);
            ++word) {
            words[word] = 0U;
        }
        __ubuf__ const MoonEpCombineV2RemoteFields *remote =
            &fields->remote[0];
        __ubuf__ TileXR::UDMASqeCtx *sqe =
            reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
        sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
        sqe->flag = 0U;
        sqe->nf = 0U;
        sqe->tokenEn = remote->tokenEn;
        sqe->rmtJettyType = remote->rmtJettyType;
        sqe->targetHint = remote->targetHint;
        sqe->inlineMsgLen = 0U;
        sqe->tpId = remote->tpId;
        sqe->sgeNum = 1U;
        sqe->rmtJettyOrSegId = remote->rmtJettyOrSegId;
        sqe->rmtTokenValue = remote->rmtTokenValue;
        sqe->rmtEidL = remote->rmtEidL;
        sqe->rmtEidH = remote->rmtEidH;
        __ubuf__ TileXR::UDMASgeCtx *sge =
            reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
                wqe + sizeof(TileXR::UDMASqeCtx));
        sge->len = static_cast<uint32_t>(fields->rowBytes);
        sge->tokenId = 0U;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(
    TileXRMoonEp::kMoonEpCombineV2BuilderThreads)
inline void MoonEpCombineV2BuildPayloadWqesVf(
    __ubuf__ uint8_t *sixPortWqes, __ubuf__ uint8_t *twoPortWqes,
    __ubuf__ const int32_t *dstSlots,
    __ubuf__ const int16_t *selectedIndices,
    __ubuf__ const MoonEpCombineV2BuildContext *context)
{
    const uint32_t task = static_cast<uint32_t>(threadIdx.x);
    if (task < context->batchCount) {
        const uint32_t densePosition = context->batchOffset + task;
        const uint32_t relativeIndex = static_cast<uint16_t>(
            selectedIndices[densePosition]);
        const uint32_t sourceSlotIndex = context->chunkStart + relativeIndex;
        const uint32_t targetSlot = static_cast<uint32_t>(
            static_cast<uint64_t>(dstSlots[relativeIndex]) -
                context->peerBase);
        const uint32_t position = context->sequencePhase + task;
        const uint32_t lane =
            TileXRMoonEp::MoonEpCombineV2LaneForPosition(position);
        const uint32_t firstTwoPort =
            (3U - (context->sequencePhase & 3U)) & 3U;
        const uint32_t twoBefore = task <= firstTwoPort ? 0U :
            1U + (task - 1U - firstTwoPort) / 4U;
        const uint32_t laneIndex =
            lane == TileXRMoonEp::MOONEP_COMBINE_V2_TWO_PORT ?
            twoBefore : task - twoBefore;
        __ubuf__ uint8_t *wqe =
            (lane == TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT ?
                sixPortWqes : twoPortWqes) + laneIndex * kWqeBytes;
        const uint64_t localAddr = context->localRowBase +
            static_cast<uint64_t>(sourceSlotIndex) *
                context->rowBytes;
        const uint64_t remoteAddr = context->remoteRowBase[lane] +
            static_cast<uint64_t>(targetSlot) *
                context->rowBytes;
        const uint32_t absoluteHead = context->head[lane] + laneIndex;

        __ubuf__ TileXR::UDMASqeCtx *sqe =
            reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
        sqe->sqeBbIdx = static_cast<uint16_t>(
            absoluteHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT);
        sqe->owner =
            (absoluteHead & TileXR::TILEXR_UDMA_SQ_BB_COUNT) == 0U ? 1U : 0U;
        sqe->rmtAddrLOrTokenId = remoteAddr & 0xFFFFFFFFU;
        sqe->rmtAddrHOrTokenValue = (remoteAddr >> 32U) & 0xFFFFFFFFU;

        __ubuf__ TileXR::UDMASgeCtx *sge =
            reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
                wqe + sizeof(TileXR::UDMASqeCtx));
        sge->va = localAddr;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(
    TileXRMoonEp::kMoonEpCombineV2BuilderThreads)
inline void MoonEpCombineV2BuildFullmeshPayloadWqesVf(
    __ubuf__ uint8_t *wqes, __ubuf__ const int32_t *dstSlots,
    __ubuf__ const int16_t *selectedIndices,
    __ubuf__ const MoonEpCombineV2BuildContext *context)
{
    const uint32_t task = static_cast<uint32_t>(threadIdx.x);
    if (task >= context->batchCount) {
        return;
    }
    const uint32_t densePosition = context->batchOffset + task;
    const uint32_t relativeIndex = static_cast<uint16_t>(
        selectedIndices[densePosition]);
    const uint32_t sourceSlotIndex = context->chunkStart + relativeIndex;
    const uint32_t targetSlot = static_cast<uint32_t>(
        static_cast<uint64_t>(dstSlots[relativeIndex]) - context->peerBase);
    __ubuf__ uint8_t *wqe = wqes + task * kWqeBytes;
    const uint64_t localAddr = context->localRowBase +
        static_cast<uint64_t>(sourceSlotIndex) * context->rowBytes;
    const uint64_t remoteAddr = context->remoteRowBase[0] +
        static_cast<uint64_t>(targetSlot) * context->rowBytes;
    const uint32_t absoluteHead = context->head[0] + task;

    __ubuf__ TileXR::UDMASqeCtx *sqe =
        reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
    sqe->sqeBbIdx = static_cast<uint16_t>(
        absoluteHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT);
    sqe->owner =
        (absoluteHead & TileXR::TILEXR_UDMA_SQ_BB_COUNT) == 0U ? 1U : 0U;
    sqe->rmtAddrLOrTokenId = remoteAddr & 0xFFFFFFFFU;
    sqe->rmtAddrHOrTokenValue = (remoteAddr >> 32U) & 0xFFFFFFFFU;
    __ubuf__ TileXR::UDMASgeCtx *sge =
        reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
            wqe + sizeof(TileXR::UDMASqeCtx));
    sge->va = localAddr;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(
    TileXRMoonEp::kMoonEpCombineV2BuilderThreads)
inline void MoonEpCombineV2BuildFullSyncWqesVf(
    __ubuf__ uint8_t *wqes,
    __ubuf__ const MoonEpCombineV2FullSyncBuildContext *context)
{
    const uint32_t task = static_cast<uint32_t>(threadIdx.x);
    if (task >= context->count) {
        return;
    }
    __ubuf__ uint8_t *wqe = wqes + task * kWqeBytes;
    __ubuf__ uint32_t *words =
        reinterpret_cast<__ubuf__ uint32_t *>(wqe);
    for (uint32_t word = 0U; word < kWqeBytes / sizeof(uint32_t); ++word) {
        words[word] = 0U;
    }

    const uint32_t absoluteHead = context->head + task;
    __ubuf__ const MoonEpCombineV2RemoteFields *remote =
        &context->remote[task];
    __ubuf__ TileXR::UDMASqeCtx *sqe =
        reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
    sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
    sqe->flag = task + 1U == context->count ?
        TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION : 0U;
    sqe->nf = 0U;
    sqe->tokenEn = remote->tokenEn;
    sqe->rmtJettyType = remote->rmtJettyType;
    sqe->sqeBbIdx = static_cast<uint16_t>(
        absoluteHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT);
    sqe->owner =
        (absoluteHead & TileXR::TILEXR_UDMA_SQ_BB_COUNT) == 0U ? 1U : 0U;
    sqe->targetHint = remote->targetHint;
    sqe->inlineMsgLen = 0U;
    sqe->tpId = remote->tpId;
    sqe->sgeNum = 1U;
    sqe->rmtJettyOrSegId = remote->rmtJettyOrSegId;
    sqe->rmtTokenValue = remote->rmtTokenValue;
    sqe->rmtEidL = remote->rmtEidL;
    sqe->rmtEidH = remote->rmtEidH;
    sqe->rmtAddrLOrTokenId = remote->remoteRowBase & 0xFFFFFFFFU;
    sqe->rmtAddrHOrTokenValue =
        (remote->remoteRowBase >> 32U) & 0xFFFFFFFFU;

    __ubuf__ TileXR::UDMASgeCtx *sge =
        reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
            wqe + sizeof(TileXR::UDMASqeCtx));
    sge->len = TileXRMoonEp::kMoonEpCombineV2FullSyncSignalBytes;
    sge->tokenId = 0U;
    sge->va = context->localSignalAddr;
}
#endif

class MoonEpCombineV2 {
public:
    __aicore__ inline void Init(GM_ADDR commArgsGM,
        GM_ADDR registeredWorkspaceGM, GM_ADDR dstLocalGM,
        uint64_t profileOffset, uint64_t scratchEpoch0Offset,
        uint64_t scratchEpoch1Offset, uint64_t doneOffset,
        uint64_t grantOffset, uint64_t controlSourceOffset,
        uint64_t failureOffset, uint64_t fullSyncReceiveOffset,
        uint64_t fullSyncSourceOffset, uint64_t fullSyncBarrierOffset,
        uint64_t outputOffset, int64_t bs,
        int64_t h, int64_t topK, int64_t nvS, uint64_t rowBytes,
        bool reduceHidden, int64_t magic, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBuffers();
    __aicore__ inline void PrefillOperatorWqes();
    __aicore__ inline bool PrefillPeerWqes(uint32_t peer);
    __aicore__ inline bool PrefillFullmeshWqes(uint32_t peer);
    __aicore__ inline void SetFailure(uint32_t status, uint32_t step,
        uint32_t peer, uint32_t lane, uint32_t cqStatus = 0U,
        uint64_t expected = 0U, uint64_t observed = 0U,
        uint32_t qp = UINT32_MAX);
    __aicore__ inline void PublishFailureRecord(uint32_t stageId);
    __aicore__ inline bool CheckPoison();
    __aicore__ inline bool ValidateDestinations();
    __aicore__ inline void LoadSelectionChunk(uint64_t chunkStart,
        uint32_t chunkElements);
    __aicore__ inline uint32_t SelectPeerIndices(uint32_t peer,
        uint32_t chunkElements);
    __aicore__ inline bool InitLaneStates();
    __aicore__ inline bool InitFullmeshState(uint32_t peer);
    __aicore__ inline bool ResolveRemoteFields(uint32_t targetRank,
        uint32_t lane, uint64_t remoteBaseOffset,
        __ubuf__ MoonEpCombineV2RemoteFields *fields);
    __aicore__ inline bool ResolveFullmeshRemoteFields(uint32_t targetRank,
        uint64_t remoteBaseOffset,
        __ubuf__ MoonEpCombineV2RemoteFields *fields);
    __aicore__ inline uint32_t PollCqOnce(
        MoonEpCombineV2LaneState &state);
    __aicore__ inline bool WaitStepCqs(uint32_t step);
    __aicore__ inline bool WaitFullmeshCq(uint32_t step, uint32_t peer);
    __aicore__ inline bool WaitStepGrant(uint32_t step);
    __aicore__ inline bool BuildPayloadWqes(uint32_t chunkStart,
        uint32_t batchOffset, uint32_t batchCount, uint32_t sequenceBase,
        uint64_t peerBase);
    __aicore__ inline bool BuildFullmeshPayloadWqes(uint32_t chunkStart,
        uint32_t batchOffset, uint32_t batchCount, uint64_t peerBase);
    __aicore__ inline bool AppendControlWqe(LocalTensor<uint8_t> issue,
        uint32_t outputIndex, MoonEpCombineV2LaneState &state,
        __gm__ TileXR::UDMAInfo *info,
        uint32_t targetRank, uint64_t remoteOffset,
        __gm__ uint64_t *localSource, uint32_t flag);
    __aicore__ inline void PublishLocalGrant(uint32_t step, uint32_t lane);
    __aicore__ inline void PublishSelfDone(uint32_t step);
    __aicore__ inline void CopyIssueToSq(LocalTensor<uint8_t> issue,
        MoonEpCombineV2LaneState &state, uint32_t count);
    __aicore__ inline bool SubmitPair(uint32_t peer, uint32_t step,
        uint32_t chunkStart, uint32_t batchOffset, uint32_t batchCount,
        uint32_t sequenceBase, uint64_t peerBase, bool finalBatch);
    __aicore__ inline bool SendRemoteStep(uint32_t peer, uint32_t step);
    __aicore__ inline bool SubmitFullmeshBatch(uint32_t peer, uint32_t step,
        uint32_t chunkStart, uint32_t batchOffset, uint32_t batchCount,
        uint64_t peerBase, bool finalBatch);
    __aicore__ inline bool SendFullmeshStep(uint32_t peer, uint32_t step);
    __aicore__ inline void CopySelfRowsIn(uint32_t selectedStart,
        uint32_t selectedCount, uint32_t chunkStart,
        uint32_t localRowStride,
        LocalTensor<uint8_t> relay);
    __aicore__ inline void CopySelfRowsOut(uint32_t selectedStart,
        uint32_t selectedCount, uint64_t peerBase,
        uint32_t localRowStride,
        LocalTensor<uint8_t> relay);
    __aicore__ inline void CopySelfTileIn(uint32_t selectedIndex,
        uint32_t chunkStart, uint64_t rowOffset, uint32_t tileBytes,
        LocalTensor<uint8_t> relay);
    __aicore__ inline void CopySelfTileOut(uint32_t selectedIndex,
        uint64_t peerBase, uint64_t rowOffset, uint32_t tileBytes,
        LocalTensor<uint8_t> relay);
    __aicore__ inline bool CopySelfSelectedIndices(uint32_t selectedCount,
        uint32_t chunkStart, uint64_t peerBase);
    __aicore__ inline bool SubmitSelfGrant(uint32_t step);
    __aicore__ inline bool SendSelfStep(uint32_t peer, uint32_t step);
    __aicore__ inline bool WaitInboundDone();
    __aicore__ inline bool ResolveFullSyncRemoteFields(uint32_t targetRank,
        uint64_t remoteOffset, __ubuf__ MoonEpCombineV2RemoteFields *fields);
    __aicore__ inline void PrepareFullSyncSignal(
        uint32_t boundaryId, uint32_t generation);
    __aicore__ inline bool BuildFullSyncWqes(
        uint32_t generation, uint32_t *count);
    __aicore__ inline bool PublishFullSyncBatch(uint32_t count);
    __aicore__ inline bool WaitFullSyncCq();
    __aicore__ inline bool WaitFullSyncSources(
        uint32_t boundaryId, uint32_t generation,
        uint32_t timeoutStatus);
    __aicore__ inline bool FullSyncSignalMatches(
        __gm__ TileXRMoonEp::MoonEpCombineV2FullSyncSignal *signal,
        uint32_t boundaryId, uint32_t sourceRank, uint32_t sourceCore,
        uint64_t *observedMagic);
    __aicore__ inline bool BeginCollectiveStage(uint32_t stageId);
    __aicore__ inline bool EndCollectiveStage(
        uint32_t stageId, bool localSucceeded);
    __aicore__ inline bool RunGlobalBarrier(uint32_t boundaryId,
        uint32_t timeoutStatus, bool localSucceeded = true);
    __aicore__ inline void InitReduceBuffers(uint32_t inputBufferNum,
        uint32_t inputSlotBytes, uint32_t floatRowBytes,
        uint32_t outputSlotBytes);
    __aicore__ inline void CopyReduceInput(uint64_t workOrdinal,
        int64_t tokenBegin, uint64_t rowCount);
    __aicore__ inline bool ReduceHidden();
    __aicore__ inline uint64_t LoadToken(__gm__ uint64_t *token);
    __aicore__ inline void RecordProfilePoint(uint32_t index);
    __aicore__ inline uint64_t BeginProfileMetric();
    __aicore__ inline void EndProfileMetric(uint32_t index,
        uint64_t startCycles);
    __aicore__ inline void WriteProfile();

    bool valid_{false};
    bool collectiveReady_{false};
    bool activeWorker_{false};
    TPipe *pipe_{nullptr};
    __gm__ TileXR::CommArgs *args_{nullptr};
    __gm__ TileXR::TileXRUDMARegistry *registry_{nullptr};
    __gm__ TileXR::TileXRUDMAFullmeshDeviceView *fullmeshView_{nullptr};
    __gm__ TileXR::UDMAInfo *fullmeshInfo_{nullptr};
    __gm__ uint8_t *workspace_{nullptr};
    __gm__ int32_t *dstGlobalAddr_{nullptr};
    __gm__ uint8_t *scratch_{nullptr};
    __gm__ uint8_t *doneBase_{nullptr};
    __gm__ uint8_t *grantBase_{nullptr};
    __gm__ uint8_t *controlSourceBase_{nullptr};
    __gm__ uint8_t *failureBase_{nullptr};
    __gm__ uint8_t *fullSyncReceiveBase_{nullptr};
    __gm__ uint8_t *fullSyncSourceBase_{nullptr};
    __gm__ uint8_t *fullSyncBarrierBase_{nullptr};
    uint64_t scratchOffset_{0U};
    uint64_t profileOffset_{0U};
    uint64_t doneOffset_{0U};
    uint64_t grantOffset_{0U};
    uint64_t failureOffset_{0U};
    uint64_t fullSyncReceiveOffset_{0U};
    uint64_t outputOffset_{0U};
    uint64_t slots_{0U};
    uint64_t rowBytes_{0U};
    int64_t bs_{0};
    int64_t h_{0};
    int64_t topK_{0};
    uint64_t magic_{0U};
    uint64_t operationStartCycles_{0U};
    uint64_t fullSyncStartCycles_{0U};
    uint32_t epoch_{0U};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    uint32_t core_{0U};
    uint32_t activeCoreCount_{0U};
    uint32_t launchCoreCount_{0U};
    uint32_t stepCount_{0U};
    uint32_t sourcesPerCore_{0U};
    uint32_t issuedRows_{0U};
    uint32_t selectedPeerRows_{0U};
    uint64_t remoteRowBase_[TileXRMoonEp::kMoonEpCombineV2LaneCount] {};
    uint32_t failureStatus_{TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS};
    uint32_t failureStep_{UINT32_MAX};
    uint32_t failurePeer_{UINT32_MAX};
    uint32_t failureLane_{UINT32_MAX};
    uint32_t failureCqStatus_{0U};
    uint32_t failureQp_{UINT32_MAX};
    uint64_t failureExpected_{0U};
    uint64_t failureObserved_{0U};
    bool selectionBuffersUsed_{false};
    bool reduceHidden_{false};
    MoonEpCombineV2LaneState lane_[
        TileXRMoonEp::kMoonEpCombineV2LaneCount] {};
    MoonEpCombineV2LaneState fullmeshLane_ {};
    uint32_t fullmeshPeer_{TileXRMoonEp::kMoonEpCombineV2InvalidPeer};
    uint32_t fullmeshProfileRoute_{0U};

    TBuf<QuePosition::VECCALC> dstSlotBuf_;
    TBuf<QuePosition::VECCALC> slotIndexBuf_;
    TBuf<QuePosition::VECCALC> selectedIndexBuf_;
    TBuf<QuePosition::VECCALC> lowerMaskBuf_;
    TBuf<QuePosition::VECCALC> upperMaskBuf_;
    TBuf<QuePosition::VECCALC> wqeIssueBuf_;
    TBuf<QuePosition::VECCALC> wqeContextBuf_;
    TQue<QuePosition::VECIN, 1> selfCopyQueue_;
    TQue<QuePosition::VECIN, kReduceMaxInputBuffers> reduceInputQueue_;
    TQue<QuePosition::VECOUT, 1> reduceOutputQueue_;
    TBuf<QuePosition::VECCALC> reduceRowBuf_;
    TBuf<QuePosition::VECCALC> reduceAccumulatorBuf_;
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    int64_t profileTimePoint_[
        TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCapacity];
    uint64_t profileMetric_[
        TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount];
#endif
};

__aicore__ inline void MoonEpCombineV2::Init(
    GM_ADDR commArgsGM, GM_ADDR registeredWorkspaceGM, GM_ADDR dstLocalGM,
    uint64_t profileOffset, uint64_t scratchEpoch0Offset,
    uint64_t scratchEpoch1Offset, uint64_t doneOffset,
    uint64_t grantOffset, uint64_t controlSourceOffset,
    uint64_t failureOffset, uint64_t fullSyncReceiveOffset,
    uint64_t fullSyncSourceOffset, uint64_t fullSyncBarrierOffset,
    uint64_t outputOffset, int64_t bs,
    int64_t h, int64_t topK, int64_t nvS, uint64_t rowBytes,
    bool reduceHidden, int64_t magic, TPipe *pipe)
{
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    const uint64_t profileStartCycles =
        static_cast<uint64_t>(GetSystemCycle());
    for (uint32_t index = 0U;
        index < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCapacity;
        ++index) {
        profileTimePoint_[index] = 0;
    }
    for (uint32_t index = 0U;
        index < TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount;
        ++index) {
        profileMetric_[index] = 0U;
    }
    profileTimePoint_[TileXRMoonEp::MOONEP_COMBINE_V2_TIME_INIT_BEGIN] =
        static_cast<int64_t>(profileStartCycles);
#endif
    pipe_ = pipe;
    args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
    workspace_ = reinterpret_cast<__gm__ uint8_t *>(registeredWorkspaceGM);
    dstGlobalAddr_ = reinterpret_cast<__gm__ int32_t *>(dstLocalGM);
    core_ = static_cast<uint32_t>(GetBlockIdx());
    launchCoreCount_ = static_cast<uint32_t>(GetBlockNum());
    operationStartCycles_ = static_cast<uint64_t>(GetSystemCycle());
    collectiveReady_ = pipe_ != nullptr && args_ != nullptr &&
        workspace_ != nullptr && dstGlobalAddr_ != nullptr &&
        launchCoreCount_ == TileXRMoonEp::kMoonEpCombineV2CoreCount &&
        core_ < launchCoreCount_;
    if (!collectiveReady_) {
        return;
    }
    profileOffset_ = profileOffset;
    failureOffset_ = failureOffset;
    failureBase_ = workspace_ + failureOffset;
    fullSyncBarrierBase_ = workspace_ + fullSyncBarrierOffset;
    magic_ = magic > 0 ? static_cast<uint64_t>(magic) : 0U;
    epoch_ = TileXRMoonEp::MoonEpCombineV2Epoch(magic_);
    rank_ = args_->rank >= 0 ? static_cast<uint32_t>(args_->rank) : 0U;
    rankSize_ = args_->rankSize > 0 ?
        static_cast<uint32_t>(args_->rankSize) : 0U;
    activeCoreCount_ = TileXRMoonEp::MoonEpCombineV2RankSizeSupported(
            rankSize_) ?
        TileXRMoonEp::MoonEpCombineV2ActiveCoreCount(rankSize_) : 0U;
    activeWorker_ = core_ < activeCoreCount_;
    __gm__ TileXR::UDMAInfo *closInfo = TileXR::GetUDMAInfo(args_);
    fullmeshView_ = reinterpret_cast<__gm__
        TileXR::TileXRUDMAFullmeshDeviceView *>(args_->udmaFullmeshPtr);
    if (!TileXR::UDMARegistryEnabled(args_) ||
        (args_->extraFlag &
            TileXR::ExtraFlag::UDMA_SHARED_QP) == 0U ||
        (args_->extraFlag & TileXR::ExtraFlag::UDMA_FULLMESH) == 0U ||
        closInfo == nullptr || fullmeshView_ == nullptr ||
        !TileXRMoonEp::MoonEpCombineV2RankSizeSupported(
            static_cast<uint32_t>(args_->rankSize)) ||
        args_->rank < 0 || args_->rank >= args_->rankSize ||
        args_->localRank < 0 ||
        args_->localRankSize != static_cast<int>(
            TileXRMoonEp::MoonEpCombineV2LocalRankSize(
                static_cast<uint32_t>(args_->rankSize))) ||
        args_->localRank >= args_->localRankSize ||
        args_->rank % args_->localRankSize != args_->localRank ||
        !TileXRMoonEp::MoonEpCombineV2ShapeValid(
            bs, h, topK, nvS) || magic <= 0 ||
        rowBytes == 0U || rowBytes > UINT32_MAX ||
        (reduceHidden && rowBytes !=
            static_cast<uint64_t>(h) * sizeof(bfloat16_t)) ||
        closInfo->qpNum != TileXRMoonEp::kMoonEpCombineV2QpCount ||
        fullmeshView_->magic != TileXR::TILEXR_UDMA_FULLMESH_MAGIC ||
        fullmeshView_->version != TileXR::TILEXR_UDMA_FULLMESH_VERSION ||
        fullmeshView_->slotCount !=
            TileXRMoonEp::kMoonEpCombineV2FullmeshSlotCount ||
        fullmeshView_->localRank != static_cast<uint32_t>(args_->localRank) ||
        fullmeshView_->connectedCount + 1U !=
            static_cast<uint32_t>(args_->localRankSize) ||
        fullmeshView_->validPeerMask != TileXR::UDMAFullmeshExpectedPeerMask(
            static_cast<uint32_t>(args_->localRank),
            static_cast<uint32_t>(args_->localRankSize)) ||
        fullmeshView_->registrationReady == 0U ||
        fullmeshView_->registrationGeneration == 0U ||
        fullmeshView_->registrationGeneration !=
            args_->udmaRegistrationGeneration ||
        fullmeshView_->infoPtr == 0U) {
        return;
    }
    fullmeshInfo_ = reinterpret_cast<__gm__ TileXR::UDMAInfo *>(
        fullmeshView_->infoPtr);
    if (fullmeshInfo_ == nullptr || fullmeshInfo_->qpNum !=
        TileXRMoonEp::kMoonEpCombineV2FullmeshSlotCount) {
        return;
    }
    if (!TileXRMoonEp::MoonEpCombineV2MagicValid(magic_)) {
        return;
    }
    stepCount_ = TileXRMoonEp::MoonEpCombineV2StepCount(rankSize_);
    sourcesPerCore_ = rankSize_ / activeCoreCount_;
    slots_ = static_cast<uint64_t>(nvS);
    rowBytes_ = rowBytes;
    bs_ = bs;
    h_ = h;
    topK_ = topK;
    reduceHidden_ = reduceHidden;
    scratchOffset_ = epoch_ == 0U ? scratchEpoch0Offset :
        scratchEpoch1Offset;
    profileOffset_ = profileOffset;
    doneOffset_ = doneOffset;
    grantOffset_ = grantOffset;
    failureOffset_ = failureOffset;
    fullSyncReceiveOffset_ = fullSyncReceiveOffset;
    outputOffset_ = outputOffset;
    scratch_ = workspace_ + scratchOffset_;
    doneBase_ = workspace_ + doneOffset;
    grantBase_ = workspace_ + grantOffset;
    controlSourceBase_ = workspace_ + controlSourceOffset +
        static_cast<uint64_t>(core_) *
            TileXRMoonEp::kMoonEpCombineV2LaneCount *
            TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes;
    failureBase_ = workspace_ + failureOffset;
    fullSyncReceiveBase_ = workspace_ + fullSyncReceiveOffset;
    fullSyncSourceBase_ = workspace_ + fullSyncSourceOffset;
    valid_ = true;
    const bool laneStatesReady = !activeWorker_ || InitLaneStates();
    if (!laneStatesReady) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            UINT32_MAX, UINT32_MAX, UINT32_MAX);
        valid_ = false;
    }
    (void)laneStatesReady;
    RecordProfilePoint(TileXRMoonEp::MOONEP_COMBINE_V2_TIME_INIT_END);
}

__aicore__ inline void MoonEpCombineV2::InitBuffers()
{
    static_assert(kSendBufferBytes <= kFullUbBytes,
        "Combine V2 send buffers exceed 216 KiB UB");
    pipe_->Reset();
    pipe_->InitBuffer(dstSlotBuf_, kDstSlotBytes);
    pipe_->InitBuffer(slotIndexBuf_, kSelectionIndexBytes);
    pipe_->InitBuffer(selectedIndexBuf_, kSelectionIndexBytes);
    pipe_->InitBuffer(lowerMaskBuf_, kSelectionMaskBytes);
    pipe_->InitBuffer(upperMaskBuf_, kSelectionMaskBytes);
    pipe_->InitBuffer(wqeIssueBuf_, kTotalIssueBytes);
    pipe_->InitBuffer(wqeContextBuf_, kWqeContextBytes);
    pipe_->InitBuffer(selfCopyQueue_, 2U,
        TileXRMoonEp::kMoonEpCombineV2SelfRelayHalfBytes);
    CreateVecIndex(slotIndexBuf_.Get<int16_t>(), static_cast<int16_t>(0),
        TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MoonEpCombineV2::PrefillOperatorWqes()
{
    __ubuf__ MoonEpCombineV2OperatorFields *fields = reinterpret_cast<__ubuf__
        MoonEpCombineV2OperatorFields *>(
            wqeContextBuf_.Get<uint8_t>().GetPhyAddr());
    fields->rowBytes = rowBytes_;
    LocalTensor<uint8_t> issue = wqeIssueBuf_.Get<uint8_t>();
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    PipeBarrier<PIPE_ALL>();
    Simt::VF_CALL<MoonEpCombineV2PrefillOperatorWqesVf>(
        Simt::Dim3{TileXRMoonEp::kMoonEpCombineV2BuilderThreads, 1U, 1U},
        reinterpret_cast<__ubuf__ uint8_t *>(
            issue.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint8_t *>(
            issue[kSixPortIssueBytes].GetPhyAddr()), fields);
    PipeBarrier<PIPE_ALL>();
#endif
}

__aicore__ inline bool MoonEpCombineV2::PrefillPeerWqes(uint32_t peer)
{
    __ubuf__ MoonEpCombineV2PeerFields *fields = reinterpret_cast<__ubuf__
        MoonEpCombineV2PeerFields *>(
            wqeContextBuf_.Get<uint8_t>().GetPhyAddr());
    fields->rowBytes = rowBytes_;
    for (uint32_t lane = 0U;
        lane < TileXRMoonEp::kMoonEpCombineV2LaneCount; ++lane) {
        const bool ready = ResolveRemoteFields(
            peer, lane, scratchOffset_, &fields->remote[lane]);
        if (kEnableSafetyChecks && !ready) {
            return false;
        }
        (void)ready;
        remoteRowBase_[lane] = fields->remote[lane].remoteRowBase;
    }
    LocalTensor<uint8_t> issue = wqeIssueBuf_.Get<uint8_t>();
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    PipeBarrier<PIPE_ALL>();
    Simt::VF_CALL<MoonEpCombineV2PrefillPeerWqesVf>(
        Simt::Dim3{TileXRMoonEp::kMoonEpCombineV2BuilderThreads, 1U, 1U},
        reinterpret_cast<__ubuf__ uint8_t *>(
            issue.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint8_t *>(
            issue[kSixPortIssueBytes].GetPhyAddr()), fields);
    PipeBarrier<PIPE_ALL>();
#endif
    return true;
}

__aicore__ inline bool MoonEpCombineV2::PrefillFullmeshWqes(uint32_t peer)
{
    __ubuf__ MoonEpCombineV2PeerFields *fields = reinterpret_cast<__ubuf__
        MoonEpCombineV2PeerFields *>(
            wqeContextBuf_.Get<uint8_t>().GetPhyAddr());
    fields->rowBytes = rowBytes_;
    if (!ResolveFullmeshRemoteFields(peer, scratchOffset_,
            &fields->remote[0])) {
        return false;
    }
    remoteRowBase_[0] = fields->remote[0].remoteRowBase;
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    PipeBarrier<PIPE_ALL>();
    Simt::VF_CALL<MoonEpCombineV2PrefillFullmeshWqesVf>(
        Simt::Dim3{TileXRMoonEp::kMoonEpCombineV2BuilderThreads, 1U, 1U},
        reinterpret_cast<__ubuf__ uint8_t *>(
            wqeIssueBuf_.Get<uint8_t>().GetPhyAddr()), fields);
    PipeBarrier<PIPE_ALL>();
#endif
    return true;
}

__aicore__ inline void MoonEpCombineV2::SetFailure(
    uint32_t status, uint32_t step, uint32_t peer, uint32_t lane,
    uint32_t cqStatus, uint64_t expected, uint64_t observed, uint32_t qp)
{
    if (failureStatus_ != TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS) {
        return;
    }
    failureStatus_ = status;
    failureStep_ = step;
    failurePeer_ = peer;
    failureLane_ = lane;
    failureCqStatus_ = cqStatus;
    failureQp_ = qp;
    failureExpected_ = expected;
    failureObserved_ = observed;
}

__aicore__ inline void MoonEpCombineV2::PublishFailureRecord(uint32_t stageId)
{
    const uint64_t index = TileXRMoonEp::MoonEpCombineV2FailureIndex(
        epoch_, core_);
    __gm__ TileXRMoonEp::MoonEpCombineV2FailureRecord *record =
        reinterpret_cast<__gm__
            TileXRMoonEp::MoonEpCombineV2FailureRecord *>(failureBase_ +
                index * TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
    record->magic = magic_;
    record->status = failureStatus_;
    record->rank = rank_;
    record->core = core_;
    record->step = failureStep_ == UINT32_MAX ? stageId : failureStep_;
    record->peer = failurePeer_;
    record->lane = failureLane_;
    record->qp = failureQp_ != UINT32_MAX ? failureQp_ :
        (failureLane_ < TileXRMoonEp::kMoonEpCombineV2LaneCount ?
            TileXRMoonEp::MoonEpCombineV2Qp(core_, failureLane_) :
            UINT32_MAX);
    record->cqStatus = failureCqStatus_;
    record->expected = failureExpected_;
    record->observed = failureObserved_;
    record->poison = 1U;
    record->marker = TileXRMoonEp::kMoonEpCombineV2FailureMarker |
        (epoch_ & 1U);
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(record),
        sizeof(*record));
}

__aicore__ inline bool MoonEpCombineV2::CheckPoison()
{
    for (uint32_t epoch = 0U;
        epoch < TileXRMoonEp::kMoonEpCombineV2EpochCount; ++epoch) {
        for (uint32_t core = 0U;
            core < activeCoreCount_; ++core) {
            const uint64_t index =
                TileXRMoonEp::MoonEpCombineV2FailureIndex(epoch, core);
            __gm__ TileXRMoonEp::MoonEpCombineV2FailureRecord *record = reinterpret_cast<__gm__
                TileXRMoonEp::MoonEpCombineV2FailureRecord *>(failureBase_ +
                    index * TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
            TileXR::UDMACleanCacheLines(
                reinterpret_cast<__gm__ uint8_t *>(record), sizeof(*record));
            if ((record->marker & ~1U) ==
                    TileXRMoonEp::kMoonEpCombineV2FailureMarker &&
                record->poison != 0U) {
                SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_POISONED,
                    UINT32_MAX, UINT32_MAX, UINT32_MAX, 0U,
                    record->magic, record->status);
                return false;
            }
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::InitLaneStates()
{
    __gm__ TileXR::UDMAInfo *info = TileXR::GetUDMAInfo(args_);
    if (info == nullptr || info->sqPtr == 0U || info->scqPtr == 0U ||
        info->memPtr == 0U) {
        return false;
    }
    const uint32_t scheduledFirstPeer = TileXRMoonEp::MoonEpCombineV2Peer(
        rank_, 0U, core_, rankSize_, kCombineV2ScheduleMode);
    const uint32_t firstPeer = scheduledFirstPeer ==
            TileXRMoonEp::kMoonEpCombineV2InvalidPeer ?
        rank_ : scheduledFirstPeer;
    for (uint32_t lane = 0U;
        lane < TileXRMoonEp::kMoonEpCombineV2LaneCount; ++lane) {
        MoonEpCombineV2LaneState &state = lane_[lane];
        state.qp = TileXRMoonEp::MoonEpCombineV2Qp(core_, lane);
        state.sq = TileXR::UDMAGetWQCtx(info, firstPeer, state.qp);
        state.cq = TileXR::UDMAGetSCQCtx(info, firstPeer, state.qp);
        if (state.sq == nullptr || state.cq == nullptr ||
            state.sq->baseBkShift >= 32U ||
            (1U << state.sq->baseBkShift) != kWqeBytes ||
            state.sq->depth != TileXR::TILEXR_UDMA_SQ_BB_COUNT ||
            state.sq->bufAddr == 0U || state.sq->headAddr == 0U ||
            state.sq->tailAddr == 0U || state.sq->wqeCntAddr == 0U ||
            state.sq->dbAddr == 0U ||
            state.cq->baseBkShift >= 32U ||
            (1U << state.cq->baseBkShift) !=
                sizeof(TileXR::UDMACqeCtx) ||
            state.cq->depth != TileXR::TILEXR_UDMA_CQ_DEPTH ||
            state.cq->bufAddr == 0U || state.cq->tailAddr == 0U ||
            state.cq->dbAddr == 0U) {
            return false;
        }
        state.head = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
            state.sq->headAddr), 0);
        state.tail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
            state.sq->tailAddr), 0);
        state.completionCount = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
            state.sq->wqeCntAddr), 0);
        state.cqTail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
            state.cq->tailAddr), 0);
        state.cqTarget = state.cqTail;
        state.submittedHead = state.head;
        if (state.head != state.tail) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::InitFullmeshState(uint32_t peer)
{
    if (peer == rank_ ||
        !TileXRMoonEp::MoonEpCombineV2SameServer(
            rank_, peer, static_cast<uint32_t>(args_->localRankSize)) ||
        fullmeshInfo_ == nullptr || fullmeshInfo_->sqPtr == 0U ||
        fullmeshInfo_->scqPtr == 0U || fullmeshInfo_->memPtr == 0U) {
        return false;
    }
    if (fullmeshPeer_ == peer) {
        return fullmeshLane_.sq != nullptr && fullmeshLane_.cq != nullptr;
    }
    if (fullmeshPeer_ != TileXRMoonEp::kMoonEpCombineV2InvalidPeer &&
        (fullmeshLane_.head != fullmeshLane_.tail ||
        fullmeshLane_.cqTail != fullmeshLane_.cqTarget)) {
        return false;
    }
    const uint32_t slot = TileXRMoonEp::MoonEpCombineV2LocalSlot(
        peer, static_cast<uint32_t>(args_->localRankSize));
    if (slot >= TileXRMoonEp::kMoonEpCombineV2FullmeshSlotCount ||
        (fullmeshView_->validPeerMask & (1U << slot)) == 0U) {
        return false;
    }
    MoonEpCombineV2LaneState state {};
    state.qp = slot;
    state.sq = TileXR::UDMAGetWQCtx(fullmeshInfo_, peer, slot);
    state.cq = TileXR::UDMAGetSCQCtx(fullmeshInfo_, peer, slot);
    if (state.sq == nullptr || state.cq == nullptr ||
        state.sq->baseBkShift >= 32U ||
        (1U << state.sq->baseBkShift) != kWqeBytes ||
        state.sq->depth != TileXR::TILEXR_UDMA_SQ_BB_COUNT ||
        state.sq->bufAddr == 0U || state.sq->headAddr == 0U ||
        state.sq->tailAddr == 0U || state.sq->wqeCntAddr == 0U ||
        state.sq->dbAddr == 0U || state.cq->baseBkShift >= 32U ||
        (1U << state.cq->baseBkShift) != sizeof(TileXR::UDMACqeCtx) ||
        state.cq->depth != TileXR::TILEXR_UDMA_CQ_DEPTH ||
        state.cq->bufAddr == 0U || state.cq->tailAddr == 0U ||
        state.cq->dbAddr == 0U) {
        return false;
    }
    state.head = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        state.sq->headAddr), 0);
    state.tail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        state.sq->tailAddr), 0);
    state.completionCount = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        state.sq->wqeCntAddr), 0);
    state.cqTail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(
        state.cq->tailAddr), 0);
    state.cqTarget = state.cqTail;
    state.submittedHead = state.head;
    if (state.head != state.tail) {
        return false;
    }
    fullmeshLane_ = state;
    fullmeshPeer_ = peer;
    return true;
}

__aicore__ inline bool MoonEpCombineV2::ResolveRemoteFields(
    uint32_t targetRank, uint32_t lane, uint64_t remoteBaseOffset,
    __ubuf__ MoonEpCombineV2RemoteFields *fields)
{
    if (kEnableSafetyChecks &&
        (targetRank >= rankSize_ ||
        lane >= TileXRMoonEp::kMoonEpCombineV2LaneCount ||
        fields == nullptr ||
        !TileXR::UDMARegisteredRangeValid(registry_, targetRank,
            remoteBaseOffset, rowBytes_ * slots_))) {
        return false;
    }
    const uint32_t qp = TileXRMoonEp::MoonEpCombineV2Qp(core_, lane);
    __gm__ TileXR::UDMAMemInfo *mem = TileXR::UDMAGetRemoteMemInfo(
        TileXR::GetUDMAInfo(args_), targetRank, qp);
    const uint64_t rangeBytes = rowBytes_ * slots_;
    const uint64_t registeredBase = reinterpret_cast<uint64_t>(
        registry_->regions[targetRank].base);
    if (kEnableSafetyChecks &&
        (mem->eidAddr == 0U || !mem->tokenValueValid || mem->addr == 0U ||
        mem->addr != registeredBase || remoteBaseOffset > mem->len ||
        rangeBytes > static_cast<uint64_t>(mem->len) - remoteBaseOffset)) {
        return false;
    }
    __gm__ uint64_t *eid = reinterpret_cast<__gm__ uint64_t *>(mem->eidAddr);
    fields->remoteRowBase = reinterpret_cast<uint64_t>(
        TileXR::UDMARegisteredRemoteAddr(registry_, targetRank,
            remoteBaseOffset));
    fields->rmtEidL = eid[0];
    fields->rmtEidH = eid[1];
    fields->tokenEn = mem->tokenValueValid;
    fields->rmtJettyType = mem->rmtJettyType;
    fields->targetHint = mem->targetHint;
    fields->tpId = mem->tpn;
    fields->rmtJettyOrSegId = mem->tid;
    fields->rmtTokenValue = mem->rmtTokenValue;
    return true;
}

__aicore__ inline bool MoonEpCombineV2::ResolveFullmeshRemoteFields(
    uint32_t targetRank, uint64_t remoteBaseOffset,
    __ubuf__ MoonEpCombineV2RemoteFields *fields)
{
    const uint32_t localRankSize = static_cast<uint32_t>(
        args_->localRankSize);
    const uint32_t slot = TileXRMoonEp::MoonEpCombineV2LocalSlot(
        targetRank, localRankSize);
    const uint64_t rangeBytes = rowBytes_ * slots_;
    if (targetRank >= rankSize_ || targetRank == rank_ ||
        !TileXRMoonEp::MoonEpCombineV2SameServer(
            rank_, targetRank, localRankSize) ||
        slot >= TileXRMoonEp::kMoonEpCombineV2FullmeshSlotCount ||
        fields == nullptr ||
        !TileXR::UDMARegisteredRangeValid(registry_, targetRank,
            remoteBaseOffset, rangeBytes)) {
        return false;
    }
    __gm__ TileXR::UDMAMemInfo *mem = TileXR::UDMAGetRemoteMemInfo(
        fullmeshInfo_, targetRank, slot);
    const uint64_t registeredBase = reinterpret_cast<uint64_t>(
        registry_->regions[targetRank].base);
    if (mem == nullptr || mem->eidAddr == 0U || !mem->tokenValueValid ||
        mem->addr == 0U || mem->addr != registeredBase ||
        remoteBaseOffset > mem->len ||
        rangeBytes > static_cast<uint64_t>(mem->len) - remoteBaseOffset) {
        return false;
    }
    __gm__ uint64_t *eid = reinterpret_cast<__gm__ uint64_t *>(mem->eidAddr);
    fields->remoteRowBase = reinterpret_cast<uint64_t>(
        TileXR::UDMARegisteredRemoteAddr(
            registry_, targetRank, remoteBaseOffset));
    fields->rmtEidL = eid[0];
    fields->rmtEidH = eid[1];
    fields->tokenEn = mem->tokenValueValid;
    fields->rmtJettyType = mem->rmtJettyType;
    fields->targetHint = mem->targetHint;
    fields->tpId = mem->tpn;
    fields->rmtJettyOrSegId = mem->tid;
    fields->rmtTokenValue = mem->rmtTokenValue;
    return true;
}

__aicore__ inline bool MoonEpCombineV2::ValidateDestinations()
{
    const uint32_t rowsPerCore = static_cast<uint32_t>(CeilDiv(
        slots_, activeCoreCount_));
    const uint32_t firstSlot = core_ * rowsPerCore;
    const uint32_t rowCount = firstSlot >= slots_ ? 0U :
        static_cast<uint32_t>(slots_ - firstSlot < rowsPerCore ?
            slots_ - firstSlot : rowsPerCore);
    if (rowCount == 0U) {
        return true;
    }
    for (uint32_t validated = 0U; validated < rowCount;
        validated += TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows) {
        const uint32_t chunkElements = rowCount - validated <
                TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows ?
            rowCount - validated :
            TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows;
        GlobalTensor<int32_t> dstGlobal;
        dstGlobal.SetGlobalBuffer(dstGlobalAddr_, slots_);
        LocalTensor<int32_t> dst = dstSlotBuf_.Get<int32_t>();
        const DataCopyExtParams params {
            1U, chunkElements * static_cast<uint32_t>(sizeof(int32_t)),
            0U, 0U, 0U};
        const DataCopyPadExtParams<int32_t> pad {false, 0U, 0U, 0U};
        if (selectionBuffersUsed_) {
            SyncFunc<HardEvent::S_MTE2>();
        }
        DataCopyPad(dst, dstGlobal[firstSlot + validated], params, pad);
        SyncFunc<HardEvent::MTE2_S>();
        selectionBuffersUsed_ = true;
        for (uint32_t localSlot = 0U; localSlot < chunkElements;
            ++localSlot) {
            const int32_t encoded = dst.GetValue(localSlot);
            if (!TileXRMoonEp::MoonEpCombineV2DestinationValid(
                    encoded, slots_, rankSize_)) {
                const uint32_t peer = encoded < 0 ? UINT32_MAX :
                    static_cast<uint32_t>(
                        static_cast<uint64_t>(encoded) / slots_);
                SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_BAD_DESTINATION,
                    UINT32_MAX, peer, UINT32_MAX,
                    firstSlot + validated + localSlot,
                    static_cast<uint64_t>(rankSize_) * slots_,
                    static_cast<uint32_t>(encoded));
                return false;
            }
        }
    }
    return true;
}

__aicore__ inline void MoonEpCombineV2::LoadSelectionChunk(
    uint64_t chunkStart, uint32_t chunkElements)
{
    if (selectionBuffersUsed_) {
        SyncFunc<HardEvent::S_MTE2>();
    }
    GlobalTensor<int32_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(dstGlobalAddr_, slots_);
    LocalTensor<int32_t> dst = dstSlotBuf_.Get<int32_t>();
    const DataCopyExtParams params {
        1U, chunkElements * static_cast<uint32_t>(sizeof(int32_t)),
        0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> pad {false, 0U, 0U, 0U};
    DataCopyPad(dst, dstGlobal[chunkStart], params, pad);
    SyncFunc<HardEvent::MTE2_V>();
    selectionBuffersUsed_ = true;
}

__aicore__ inline uint32_t MoonEpCombineV2::SelectPeerIndices(
    uint32_t peer, uint32_t chunkElements)
{
    const uint64_t peerBase = static_cast<uint64_t>(peer) * slots_;
    const uint64_t peerEnd = peerBase + slots_;
    if (peerBase > static_cast<uint64_t>(INT32_MAX)) {
        return 0U;
    }
    LocalTensor<int32_t> dstSlots = dstSlotBuf_.Get<int32_t>();
    LocalTensor<uint8_t> lowerMask = lowerMaskBuf_.Get<uint8_t>();
    Compares(lowerMask, dstSlots, static_cast<int32_t>(peerBase),
        CMPMODE::GE, chunkElements);
    if (peerEnd <= static_cast<uint64_t>(INT32_MAX)) {
        LocalTensor<uint8_t> upperMask = upperMaskBuf_.Get<uint8_t>();
        Compares(upperMask, dstSlots, static_cast<int32_t>(peerEnd),
            CMPMODE::LT, chunkElements);
        PipeBarrier<PIPE_V>();
        And(lowerMask.ReinterpretCast<uint16_t>(),
            lowerMask.ReinterpretCast<uint16_t>(),
            upperMask.ReinterpretCast<uint16_t>(),
            static_cast<int32_t>(kSelectionMaskElements));
    }
    PipeBarrier<PIPE_V>();
    uint64_t selectedCount = 0U;
    GatherMask(selectedIndexBuf_.Get<int16_t>(),
        slotIndexBuf_.Get<int16_t>(), lowerMask.ReinterpretCast<uint16_t>(),
        true, chunkElements, {1U, 1U, 0U, 0U}, selectedCount);
    SyncFunc<HardEvent::V_S>();
    return static_cast<uint32_t>(selectedCount);
}

__aicore__ inline uint64_t MoonEpCombineV2::LoadToken(
    __gm__ uint64_t *token)
{
    TileXR::UDMACleanCacheLines(
        reinterpret_cast<__gm__ uint8_t *>(token), sizeof(uint64_t));
    return *token;
}

__aicore__ inline uint32_t MoonEpCombineV2::PollCqOnce(
    MoonEpCombineV2LaneState &state)
{
    if (TileXRMoonEp::MoonEpCombineV2CqTargetReached(
            state.cqTail, state.cqTarget)) {
        return 0U;
    }
    __gm__ TileXR::UDMACqeCtx *cqe = reinterpret_cast<__gm__ TileXR::UDMACqeCtx *>(
        state.cq->bufAddr + sizeof(TileXR::UDMACqeCtx) *
            (state.cqTail % TileXR::TILEXR_UDMA_CQ_DEPTH));
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(cqe),
        sizeof(TileXR::UDMACqeCtx));
    uint32_t completedTail = state.tail;
    uint32_t detail = 0U;
    const uint32_t result = TileXRMoonEp::MoonEpCombineV2AdvanceSingleCqe(
        state.cqTail, TileXR::TILEXR_UDMA_CQ_DEPTH,
        static_cast<uint32_t>(cqe->owner),
        static_cast<uint32_t>(cqe->status),
        static_cast<uint32_t>(cqe->substatus), state.tail,
        state.submittedHead, static_cast<uint32_t>(cqe->entryIdx),
        TileXR::TILEXR_UDMA_SQ_BB_COUNT, completedTail, detail);
    if (result ==
        TileXRMoonEp::MOONEP_COMBINE_V2_SINGLE_CQE_NO_COMPLETION) {
        return kPollNoCompletion;
    }
    if (result == TileXRMoonEp::MOONEP_COMBINE_V2_SINGLE_CQE_ERROR) {
        return detail == 0U ? kPollInvalidState : detail;
    }
    if (result ==
        TileXRMoonEp::MOONEP_COMBINE_V2_SINGLE_CQE_INVALID_STATE) {
        return kPollInvalidState;
    }
    ++state.cqTail;
    state.tail = completedTail;
    st_dev(state.cqTail, reinterpret_cast<__gm__ uint32_t *>(
        state.cq->tailAddr), 0);
    st_dev(state.cqTail & 0xFFFFFFU,
        reinterpret_cast<__gm__ uint32_t *>(state.cq->dbAddr), 0);
    st_dev(state.tail, reinterpret_cast<__gm__ uint32_t *>(
        state.sq->tailAddr), 0);
    return TileXRMoonEp::MoonEpCombineV2CqTargetReached(
        state.cqTail, state.cqTarget) ? 0U : kPollNoCompletion;
}

__aicore__ inline bool MoonEpCombineV2::WaitStepCqs(uint32_t step)
{
    bool ready[2] = {
        TileXRMoonEp::MoonEpCombineV2CqTargetReached(
            lane_[0].cqTail, lane_[0].cqTarget),
        TileXRMoonEp::MoonEpCombineV2CqTargetReached(
            lane_[1].cqTail, lane_[1].cqTarget)};
    uint32_t cursor = 0U;
    while (!(ready[0] && ready[1])) {
        for (uint32_t offset = 0U; offset < 2U; ++offset) {
            const uint32_t lane = (cursor + offset) & 1U;
            if (ready[lane]) {
                continue;
            }
            const uint32_t status = PollCqOnce(lane_[lane]);
            if (status == 0U) {
                ready[lane] = true;
            } else if (status != kPollNoCompletion) {
                SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
                    step, UINT32_MAX, lane, status);
                return false;
            }
        }
        cursor ^= 1U;
        if (TimedOut(operationStartCycles_)) {
            for (uint32_t lane = 0U; lane < 2U; ++lane) {
                if (!ready[lane]) {
                    SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_TIMEOUT,
                        step, UINT32_MAX, lane);
                    return false;
                }
            }
        }
    }
    for (uint32_t lane = 0U;
        lane < TileXRMoonEp::kMoonEpCombineV2LaneCount; ++lane) {
        MoonEpCombineV2LaneState &state = lane_[lane];
        if (state.cqTail != state.cqTarget ||
            state.tail != state.submittedHead || state.head != state.tail) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
                step, UINT32_MAX, lane, kPollInvalidState,
                state.submittedHead, state.tail, state.qp);
            return false;
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::WaitFullmeshCq(
    uint32_t step, uint32_t peer)
{
    const uint32_t logicalQp =
        TileXRMoonEp::MoonEpCombineV2FullmeshLogicalQp(
            peer, static_cast<uint32_t>(args_->localRankSize));
    while (!TileXRMoonEp::MoonEpCombineV2CqTargetReached(
        fullmeshLane_.cqTail, fullmeshLane_.cqTarget)) {
        const uint32_t status = PollCqOnce(fullmeshLane_);
        if (status != 0U && status != kPollNoCompletion) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
                step, peer, TileXRMoonEp::MOONEP_COMBINE_V2_FULLMESH,
                status, fullmeshLane_.cqTarget, fullmeshLane_.cqTail,
                logicalQp);
            return false;
        }
        if (TimedOut(operationStartCycles_)) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_TIMEOUT,
                step, peer, TileXRMoonEp::MOONEP_COMBINE_V2_FULLMESH,
                0U, fullmeshLane_.cqTarget, fullmeshLane_.cqTail,
                logicalQp);
            return false;
        }
    }
    if (fullmeshLane_.tail != fullmeshLane_.submittedHead ||
        fullmeshLane_.head != fullmeshLane_.tail) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
            step, peer, TileXRMoonEp::MOONEP_COMBINE_V2_FULLMESH,
            kPollInvalidState, fullmeshLane_.submittedHead,
            fullmeshLane_.tail, logicalQp);
        return false;
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::WaitStepGrant(uint32_t step)
{
    bool ready[2] = {false, false};
    uint64_t observed[2] = {0U, 0U};
    const uint64_t expected = TileXRMoonEp::MoonEpCombineV2GrantToken(
        magic_, step, stepCount_);
    uint32_t cursor = 0U;
    while (!(ready[0] && ready[1])) {
        for (uint32_t offset = 0U; offset < 2U; ++offset) {
            const uint32_t lane = (cursor + offset) & 1U;
            if (ready[lane]) {
                continue;
            }
            const uint64_t index = TileXRMoonEp::MoonEpCombineV2GrantIndex(
                epoch_, core_, lane, step);
            __gm__ uint64_t *token = reinterpret_cast<__gm__ uint64_t *>(
                grantBase_ + index *
                    TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                    TileXRMoonEp::kMoonEpCombineV2GrantReceiveOffsetBytes);
            observed[lane] = LoadToken(token);
            ready[lane] = observed[lane] == expected;
        }
        cursor ^= 1U;
        if (TimedOut(operationStartCycles_)) {
            for (uint32_t lane = 0U; lane < 2U; ++lane) {
                if (!ready[lane]) {
                    SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_GRANT_TIMEOUT,
                        step, UINT32_MAX, lane, 0U, expected,
                        observed[lane]);
                    return false;
                }
            }
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::BuildPayloadWqes(
    uint32_t chunkStart, uint32_t batchOffset, uint32_t batchCount,
    uint32_t sequenceBase, uint64_t peerBase)
{
    LocalTensor<uint8_t> contextTensor = wqeContextBuf_.Get<uint8_t>();
    __ubuf__ MoonEpCombineV2BuildContext *context = reinterpret_cast<__ubuf__
        MoonEpCombineV2BuildContext *>(contextTensor.GetPhyAddr());
    context->localRowBase = reinterpret_cast<uint64_t>(workspace_);
    context->rowBytes = rowBytes_;
    context->peerBase = peerBase;
    context->chunkStart = chunkStart;
    context->batchOffset = batchOffset;
    context->batchCount = batchCount;
    context->sequencePhase = sequenceBase & 3U;
    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        context->head[lane] = lane_[lane].head;
        context->remoteRowBase[lane] = remoteRowBase_[lane];
    }
    LocalTensor<uint8_t> issue = wqeIssueBuf_.Get<uint8_t>();
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    PipeBarrier<PIPE_ALL>();
    Simt::VF_CALL<MoonEpCombineV2BuildPayloadWqesVf>(
        Simt::Dim3{TileXRMoonEp::kMoonEpCombineV2BuilderThreads, 1U, 1U},
        reinterpret_cast<__ubuf__ uint8_t *>(
            issue.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint8_t *>(
            issue[kSixPortIssueBytes].GetPhyAddr()),
        reinterpret_cast<__ubuf__ const int32_t *>(
            dstSlotBuf_.Get<int32_t>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ const int16_t *>(
            selectedIndexBuf_.Get<int16_t>().GetPhyAddr()), context);
    PipeBarrier<PIPE_ALL>();
#endif
    return true;
}

__aicore__ inline bool MoonEpCombineV2::BuildFullmeshPayloadWqes(
    uint32_t chunkStart, uint32_t batchOffset, uint32_t batchCount,
    uint64_t peerBase)
{
    __ubuf__ MoonEpCombineV2BuildContext *context = reinterpret_cast<__ubuf__
        MoonEpCombineV2BuildContext *>(
            wqeContextBuf_.Get<uint8_t>().GetPhyAddr());
    context->localRowBase = reinterpret_cast<uint64_t>(workspace_);
    context->rowBytes = rowBytes_;
    context->remoteRowBase[0] = remoteRowBase_[0];
    context->peerBase = peerBase;
    context->chunkStart = chunkStart;
    context->batchOffset = batchOffset;
    context->batchCount = batchCount;
    context->head[0] = fullmeshLane_.head;
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    PipeBarrier<PIPE_ALL>();
    Simt::VF_CALL<MoonEpCombineV2BuildFullmeshPayloadWqesVf>(
        Simt::Dim3{TileXRMoonEp::kMoonEpCombineV2BuilderThreads, 1U, 1U},
        reinterpret_cast<__ubuf__ uint8_t *>(
            wqeIssueBuf_.Get<uint8_t>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ const int32_t *>(
            dstSlotBuf_.Get<int32_t>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ const int16_t *>(
            selectedIndexBuf_.Get<int16_t>().GetPhyAddr()), context);
    PipeBarrier<PIPE_ALL>();
#endif
    return true;
}

__aicore__ inline bool MoonEpCombineV2::AppendControlWqe(
    LocalTensor<uint8_t> issue, uint32_t outputIndex,
    MoonEpCombineV2LaneState &state, __gm__ TileXR::UDMAInfo *info,
    uint32_t targetRank,
    uint64_t remoteOffset, __gm__ uint64_t *localSource, uint32_t flag)
{
    if (info == nullptr || outputIndex >= kTotalIssueCapacity ||
        !TileXR::UDMARegisteredRangeValid(registry_, targetRank,
            remoteOffset, sizeof(uint64_t))) {
        return false;
    }
    __gm__ TileXR::UDMAMemInfo *mem = TileXR::UDMAGetRemoteMemInfo(
        info, targetRank, state.qp);
    const uint64_t registeredBase = reinterpret_cast<uint64_t>(
        registry_->regions[targetRank].base);
    if (localSource == nullptr || mem == nullptr || mem->eidAddr == 0U ||
        !mem->tokenValueValid || mem->addr == 0U ||
        mem->addr != registeredBase || remoteOffset > mem->len ||
        sizeof(uint64_t) > static_cast<uint64_t>(mem->len) - remoteOffset) {
        return false;
    }
    __gm__ uint64_t *eid = reinterpret_cast<__gm__ uint64_t *>(mem->eidAddr);
    __ubuf__ uint8_t *wqe = reinterpret_cast<__ubuf__ uint8_t *>(
        issue.GetPhyAddr()) + outputIndex * kWqeBytes;
    __ubuf__ uint32_t *words = reinterpret_cast<__ubuf__ uint32_t *>(wqe);
    for (uint32_t word = 0U; word < kWqeBytes / sizeof(uint32_t); ++word) {
        words[word] = 0U;
    }
    const uint32_t absoluteHead = state.head + outputIndex;
    __ubuf__ TileXR::UDMASqeCtx *sqe =
        reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
    sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
    sqe->flag = flag;
    sqe->nf = 0U;
    sqe->tokenEn = mem->tokenValueValid;
    sqe->rmtJettyType = mem->rmtJettyType;
    sqe->sqeBbIdx = static_cast<uint16_t>(
        absoluteHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT);
    sqe->owner =
        (absoluteHead & TileXR::TILEXR_UDMA_SQ_BB_COUNT) == 0U ? 1U : 0U;
    sqe->targetHint = mem->targetHint;
    sqe->inlineMsgLen = 0U;
    sqe->tpId = mem->tpn;
    sqe->sgeNum = 1U;
    sqe->rmtJettyOrSegId = mem->tid;
    sqe->rmtTokenValue = mem->rmtTokenValue;
    sqe->rmtEidL = eid[0];
    sqe->rmtEidH = eid[1];
    const uint64_t remoteAddr = reinterpret_cast<uint64_t>(
        TileXR::UDMARegisteredRemoteAddr(registry_, targetRank, remoteOffset));
    sqe->rmtAddrLOrTokenId = remoteAddr & 0xFFFFFFFFU;
    sqe->rmtAddrHOrTokenValue = (remoteAddr >> 32U) & 0xFFFFFFFFU;
    __ubuf__ TileXR::UDMASgeCtx *sge =
        reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
            wqe + sizeof(TileXR::UDMASqeCtx));
    sge->len = sizeof(uint64_t);
    sge->tokenId = 0U;
    sge->va = reinterpret_cast<uint64_t>(localSource);
    return true;
}

__aicore__ inline void MoonEpCombineV2::PublishLocalGrant(
    uint32_t step, uint32_t lane)
{
    const uint64_t grantIndex = TileXRMoonEp::MoonEpCombineV2GrantIndex(
        epoch_, core_, lane, step);
    __gm__ uint64_t *grant = reinterpret_cast<__gm__ uint64_t *>(
        grantBase_ + grantIndex *
            TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
            TileXRMoonEp::kMoonEpCombineV2GrantReceiveOffsetBytes);
    *grant = TileXRMoonEp::MoonEpCombineV2GrantToken(
        magic_, step, stepCount_);
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(grant),
        TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
}

__aicore__ inline void MoonEpCombineV2::PublishSelfDone(uint32_t step)
{
    for (uint32_t lane = 0U;
        lane < TileXRMoonEp::kMoonEpCombineV2LaneCount; ++lane) {
        const uint64_t index = TileXRMoonEp::MoonEpCombineV2DoneIndex(
            epoch_, rank_, lane);
        __gm__ uint64_t *done = reinterpret_cast<__gm__ uint64_t *>(
            doneBase_ + index *
                TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
        *done = TileXRMoonEp::MoonEpCombineV2Token(magic_, step);
        TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(done),
            TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
    }
}

__aicore__ inline void MoonEpCombineV2::CopyIssueToSq(
    LocalTensor<uint8_t> issue, MoonEpCombineV2LaneState &state,
    uint32_t count)
{
    const TileXRMoonEp::MoonEpCombineV2RingSegments segments =
        TileXRMoonEp::MoonEpCombineV2SplitRingCopy(
            state.head, count, TileXR::TILEXR_UDMA_SQ_BB_COUNT);
    const uint32_t ringIndex =
        state.head % TileXR::TILEXR_UDMA_SQ_BB_COUNT;
    const uint32_t firstCount = segments.first;
    GlobalTensor<uint8_t> firstDst;
    firstDst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
        state.sq->bufAddr + static_cast<uint64_t>(ringIndex) * kWqeBytes));
    const DataCopyExtParams firstParams {
        1U, firstCount * kWqeBytes, 0U, 0U, 0U};
    DataCopyPad(firstDst, issue, firstParams);
    const uint32_t secondCount = segments.second;
    if (secondCount != 0U) {
        GlobalTensor<uint8_t> secondDst;
        secondDst.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t *>(state.sq->bufAddr));
        const DataCopyExtParams secondParams {
            1U, secondCount * kWqeBytes, 0U, 0U, 0U};
        DataCopyPad(secondDst, issue[firstCount * kWqeBytes], secondParams);
    }
}

__aicore__ inline bool MoonEpCombineV2::SubmitPair(
    uint32_t peer, uint32_t step, uint32_t chunkStart,
    uint32_t batchOffset, uint32_t batchCount, uint32_t sequenceBase,
    uint64_t peerBase, bool finalBatch)
{
    const uint64_t buildStart = BeginProfileMetric();
    const bool payloadReady = BuildPayloadWqes(chunkStart, batchOffset,
        batchCount, sequenceBase, peerBase);
    EndProfileMetric(TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_REMOTE_WQE_BUILD,
        buildStart);
    if (kEnableSafetyChecks && !payloadReady) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            step, peer, UINT32_MAX);
        return false;
    }
    (void)payloadReady;
    const uint64_t submitStart = BeginProfileMetric();
    const uint32_t phase = sequenceBase & 3U;
    const TileXRMoonEp::MoonEpCombineV2LaneCounts batchCounts =
        TileXRMoonEp::MoonEpCombineV2BatchLaneCounts(
            batchCount, phase, step, stepCount_, false);
    uint32_t count[2] = {batchCounts.sixPort, batchCounts.twoPort};
    LocalTensor<uint8_t> allIssue = wqeIssueBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> laneIssue[2] = {
        allIssue,
        allIssue[kSixPortIssueBytes]
    };
    if (finalBatch) {
        const uint32_t successor =
            TileXRMoonEp::MoonEpCombineV2Successor(
                rank_, step, core_, rankSize_, kCombineV2ScheduleMode);
        for (uint32_t lane = 0U; lane < 2U; ++lane) {
            LocalTensor<uint8_t> issue = laneIssue[lane];
            __gm__ uint64_t *doneSource = reinterpret_cast<__gm__ uint64_t *>(
                controlSourceBase_ + static_cast<uint64_t>(lane) *
                    TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
            if (successor == rank_) {
                PublishLocalGrant(step, lane);
            } else {
                const uint64_t grantIndex =
                    TileXRMoonEp::MoonEpCombineV2GrantIndex(
                        epoch_, core_, lane, step);
                __gm__ uint64_t *grantSource = reinterpret_cast<__gm__ uint64_t *>(
                    grantBase_ + grantIndex *
                        TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                        TileXRMoonEp::kMoonEpCombineV2GrantSourceOffsetBytes);
                *grantSource = TileXRMoonEp::MoonEpCombineV2GrantToken(
                    magic_, step, stepCount_);
                TileXR::UDMACleanCacheLines(
                    reinterpret_cast<__gm__ uint8_t *>(grantSource),
                    TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
                const bool grantAppended = AppendControlWqe(
                    issue, count[lane], lane_[lane],
                    TileXR::GetUDMAInfo(args_), successor,
                    grantOffset_ + grantIndex *
                        TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                        TileXRMoonEp::kMoonEpCombineV2GrantReceiveOffsetBytes,
                    grantSource,
                    TileXR::TILEXR_UDMA_SQE_FLAG_STRONG_ORDER);
                if (!grantAppended) {
                    SetFailure(
                        TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                        step, successor, lane);
                    EndProfileMetric(
                        TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_REMOTE_SUBMIT,
                        submitStart);
                    return false;
                }
                (void)grantAppended;
                ++count[lane];
            }
            *doneSource = TileXRMoonEp::MoonEpCombineV2Token(magic_, step);
            TileXR::UDMACleanCacheLines(
                reinterpret_cast<__gm__ uint8_t *>(doneSource),
                TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
            const uint64_t doneIndex =
                TileXRMoonEp::MoonEpCombineV2DoneIndex(
                    epoch_, rank_, lane);
            const bool doneAppended = AppendControlWqe(
                issue, count[lane], lane_[lane],
                TileXR::GetUDMAInfo(args_), peer,
                doneOffset_ + doneIndex *
                        TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes,
                doneSource,
                TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
            if (!doneAppended) {
                SetFailure(
                    TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                    step, peer, lane);
                EndProfileMetric(
                    TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_REMOTE_SUBMIT,
                    submitStart);
                return false;
            }
            (void)doneAppended;
            ++count[lane];
        }
    }

    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        const uint32_t outstanding = lane_[lane].head - lane_[lane].tail;
        if (kEnableSafetyChecks &&
            (static_cast<uint64_t>(outstanding) + count[lane] >=
                TileXRMoonEp::kMoonEpCombineV2MaxOutstanding)) {
            SetFailure(
                TileXRMoonEp::MOONEP_COMBINE_V2_OUTSTANDING_LIMIT,
                step, peer, lane, outstanding + count[lane]);
            EndProfileMetric(
                TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_REMOTE_SUBMIT,
                submitStart);
            return false;
        }
    }

    SyncFunc<HardEvent::S_MTE3>();
    if (count[0] != 0U) {
        CopyIssueToSq(laneIssue[0], lane_[0], count[0]);
    }
    if (count[1] != 0U) {
        CopyIssueToSq(laneIssue[1], lane_[1], count[1]);
    }
    SyncFunc<HardEvent::MTE3_S>();
    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        lane_[lane].head += count[lane];
        lane_[lane].completionCount +=
            TileXRMoonEp::MoonEpCombineV2CompletionCount(finalBatch);
        st_dev(lane_[lane].head, reinterpret_cast<__gm__ uint32_t *>(
            lane_[lane].sq->headAddr), 0);
        st_dev(lane_[lane].completionCount,
            reinterpret_cast<__gm__ uint32_t *>(
            lane_[lane].sq->wqeCntAddr), 0);
    }
    if (count[0] != 0U) {
        st_dev(lane_[0].head, reinterpret_cast<__gm__ uint32_t *>(
            lane_[0].sq->dbAddr), 0);
    }
    if (count[1] != 0U) {
        st_dev(lane_[1].head, reinterpret_cast<__gm__ uint32_t *>(
            lane_[1].sq->dbAddr), 0);
    }
    if (finalBatch) {
        for (uint32_t lane = 0U; lane < 2U; ++lane) {
            lane_[lane].submittedHead = lane_[lane].head;
            lane_[lane].cqTarget =
                TileXRMoonEp::MoonEpCombineV2NextCqTarget(
                    lane_[lane].cqTail, true);
        }
    }
    issuedRows_ += batchCount;
    EndProfileMetric(TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_REMOTE_SUBMIT,
        submitStart);
    return true;
}

__aicore__ inline bool MoonEpCombineV2::SendRemoteStep(
    uint32_t peer, uint32_t step)
{
    issuedRows_ = 0U;
    selectedPeerRows_ = 0U;
    const uint64_t peerBase = static_cast<uint64_t>(peer) * slots_;
    const bool prefilled = PrefillPeerWqes(peer);
    if (kEnableSafetyChecks && !prefilled) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            step, peer, UINT32_MAX);
        return false;
    }
    (void)prefilled;
    for (uint64_t chunkStart = 0U; chunkStart < slots_;
        chunkStart += TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows) {
        const uint32_t chunkElements = static_cast<uint32_t>(
            slots_ - chunkStart <
                TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows ?
            slots_ - chunkStart :
            TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows);
        uint64_t profileStart = BeginProfileMetric();
        LoadSelectionChunk(chunkStart, chunkElements);
        EndProfileMetric(
            TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_SELECTION_LOAD,
            profileStart);
        profileStart = BeginProfileMetric();
        const uint32_t selectedCount = SelectPeerIndices(peer, chunkElements);
        EndProfileMetric(
            TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_SELECTION_SELECT,
            profileStart);
        const bool lastChunk = chunkStart + chunkElements == slots_;
        if (selectedCount == 0U && lastChunk) {
            const bool submitted = SubmitPair(peer, step,
                static_cast<uint32_t>(chunkStart), 0U, 0U, issuedRows_,
                peerBase, true);
            if (kEnableSafetyChecks && !submitted) {
                return false;
            }
            (void)submitted;
        }
        for (uint32_t batchOffset = 0U; batchOffset < selectedCount;
            batchOffset += TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows) {
            const uint32_t remaining = selectedCount - batchOffset;
            const uint32_t batchCount = remaining <
                    TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows ?
                remaining : TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows;
            const bool finalBatch = lastChunk &&
                batchOffset + batchCount == selectedCount;
            const bool submitted = SubmitPair(peer, step,
                static_cast<uint32_t>(chunkStart), batchOffset, batchCount,
                issuedRows_, peerBase, finalBatch);
            if (kEnableSafetyChecks && !submitted) {
                return false;
            }
            (void)submitted;
            selectedPeerRows_ += batchCount;
            if (kEnableSafetyChecks && selectedPeerRows_ >
                TileXRMoonEp::kMoonEpCombineV2MaxOutstanding) {
                SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_OUTSTANDING_LIMIT,
                    step, peer, UINT32_MAX, selectedPeerRows_);
                return false;
            }
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::SubmitFullmeshBatch(
    uint32_t peer, uint32_t step, uint32_t chunkStart,
    uint32_t batchOffset, uint32_t batchCount, uint64_t peerBase,
    bool finalBatch)
{
    if (batchCount > TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows ||
        !BuildFullmeshPayloadWqes(
            chunkStart, batchOffset, batchCount, peerBase)) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            step, peer, TileXRMoonEp::MOONEP_COMBINE_V2_FULLMESH, 0U,
            0U, 0U, TileXRMoonEp::MoonEpCombineV2FullmeshLogicalQp(
                peer, static_cast<uint32_t>(args_->localRankSize)));
        return false;
    }
    uint32_t count = batchCount;
    LocalTensor<uint8_t> issue = wqeIssueBuf_.Get<uint8_t>();
    if (finalBatch) {
        __gm__ uint64_t *doneSource = reinterpret_cast<__gm__ uint64_t *>(
            controlSourceBase_);
        *doneSource = TileXRMoonEp::MoonEpCombineV2Token(magic_, step);
        TileXR::UDMACleanCacheLines(
            reinterpret_cast<__gm__ uint8_t *>(doneSource),
            TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
        const uint64_t doneIndex = TileXRMoonEp::MoonEpCombineV2DoneIndex(
            epoch_, rank_, TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT);
        if (!AppendControlWqe(issue, count, fullmeshLane_, fullmeshInfo_,
                peer, doneOffset_ + doneIndex *
                    TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes,
                doneSource,
                TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION)) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                step, peer, TileXRMoonEp::MOONEP_COMBINE_V2_FULLMESH,
                0U, 0U, 0U,
                TileXRMoonEp::MoonEpCombineV2FullmeshLogicalQp(
                    peer, static_cast<uint32_t>(args_->localRankSize)));
            return false;
        }
        ++count;
        RecordProfilePoint(TileXRMoonEp::
            MOONEP_COMBINE_V2_DIAG_FULLMESH_WQE_BUILD_END);
    }
    const uint32_t outstanding =
        fullmeshLane_.head - fullmeshLane_.tail;
    const bool completionBatch = finalBatch ||
        (count != 0U && static_cast<uint64_t>(outstanding) + count >=
            TileXRMoonEp::kMoonEpCombineV2MaxOutstanding -
                TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows);
    if (completionBatch && !finalBatch) {
        __ubuf__ TileXR::UDMASqeCtx *last = reinterpret_cast<__ubuf__
            TileXR::UDMASqeCtx *>(issue[(count - 1U) * kWqeBytes].GetPhyAddr());
        last->flag = TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION;
    }
    if (count == 0U || static_cast<uint64_t>(outstanding) + count >=
        TileXRMoonEp::kMoonEpCombineV2MaxOutstanding) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_OUTSTANDING_LIMIT,
            step, peer, TileXRMoonEp::MOONEP_COMBINE_V2_FULLMESH,
            outstanding + count, 0U, 0U,
            TileXRMoonEp::MoonEpCombineV2FullmeshLogicalQp(
                peer, static_cast<uint32_t>(args_->localRankSize)));
        return false;
    }

    SyncFunc<HardEvent::S_MTE3>();
    CopyIssueToSq(issue, fullmeshLane_, count);
    SyncFunc<HardEvent::MTE3_S>();
    fullmeshLane_.head += count;
    if (completionBatch) {
        ++fullmeshLane_.completionCount;
        fullmeshLane_.submittedHead = fullmeshLane_.head;
        fullmeshLane_.cqTarget =
            TileXRMoonEp::MoonEpCombineV2NextCqTarget(
                fullmeshLane_.cqTail, true);
    }
    st_dev(fullmeshLane_.head, reinterpret_cast<__gm__ uint32_t *>(
        fullmeshLane_.sq->headAddr), 0);
    st_dev(fullmeshLane_.completionCount,
        reinterpret_cast<__gm__ uint32_t *>(
            fullmeshLane_.sq->wqeCntAddr), 0);
    st_dev(fullmeshLane_.head, reinterpret_cast<__gm__ uint32_t *>(
        fullmeshLane_.sq->dbAddr), 0);
    if (finalBatch) {
        RecordProfilePoint(TileXRMoonEp::
            MOONEP_COMBINE_V2_DIAG_FULLMESH_SUBMIT_END);
    }
    issuedRows_ += batchCount;
    if (completionBatch && !finalBatch &&
        !WaitFullmeshCq(step, peer)) {
        return false;
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::SendFullmeshStep(
    uint32_t peer, uint32_t step)
{
    issuedRows_ = 0U;
    selectedPeerRows_ = 0U;
    if (!InitFullmeshState(peer) || !PrefillFullmeshWqes(peer)) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            step, peer, TileXRMoonEp::MOONEP_COMBINE_V2_FULLMESH,
            0U, 0U, 0U,
            TileXRMoonEp::MoonEpCombineV2FullmeshLogicalQp(
                peer, static_cast<uint32_t>(args_->localRankSize)));
        return false;
    }
    const uint64_t peerBase = static_cast<uint64_t>(peer) * slots_;
    for (uint64_t chunkStart = 0U; chunkStart < slots_;
        chunkStart += TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows) {
        const uint32_t chunkElements = static_cast<uint32_t>(
            slots_ - chunkStart <
                    TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows ?
                slots_ - chunkStart :
                TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows);
        LoadSelectionChunk(chunkStart, chunkElements);
        const uint32_t selectedCount = SelectPeerIndices(peer, chunkElements);
        const bool lastChunk = chunkStart + chunkElements == slots_;
        if (selectedCount == 0U && lastChunk &&
            !SubmitFullmeshBatch(peer, step,
                static_cast<uint32_t>(chunkStart), 0U, 0U,
                peerBase, true)) {
            return false;
        }
        for (uint32_t batchOffset = 0U; batchOffset < selectedCount;
            batchOffset += TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows) {
            const uint32_t remaining = selectedCount - batchOffset;
            const uint32_t batchCount = remaining <
                    TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows ?
                remaining : TileXRMoonEp::kMoonEpCombineV2PayloadBatchRows;
            const bool finalBatch = lastChunk &&
                batchOffset + batchCount == selectedCount;
            if (!SubmitFullmeshBatch(peer, step,
                    static_cast<uint32_t>(chunkStart), batchOffset,
                    batchCount, peerBase, finalBatch)) {
                return false;
            }
            selectedPeerRows_ += batchCount;
        }
    }
    return true;
}

__aicore__ inline void MoonEpCombineV2::CopySelfRowsIn(
    uint32_t selectedStart, uint32_t selectedCount, uint32_t chunkStart,
    uint32_t localRowStride, LocalTensor<uint8_t> relay)
{
    LocalTensor<int16_t> selectedIndices =
        selectedIndexBuf_.Get<int16_t>();
    const uint32_t copyBytes = static_cast<uint32_t>(rowBytes_);
    const DataCopyExtParams copyIn {1U, copyBytes, 0U, 0U, 0U};
    const DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0U};
    for (uint32_t row = 0U; row < selectedCount; ++row) {
        const uint32_t relativeIndex = static_cast<uint16_t>(
            selectedIndices.GetValue(selectedStart + row));
        const uint32_t sourceSlotIndex = chunkStart + relativeIndex;
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(workspace_ +
            static_cast<uint64_t>(sourceSlotIndex) * rowBytes_);
        DataCopyPad(relay[row * localRowStride], source, copyIn, padIn);
    }
}

__aicore__ inline void MoonEpCombineV2::CopySelfRowsOut(
    uint32_t selectedStart, uint32_t selectedCount, uint64_t peerBase,
    uint32_t localRowStride, LocalTensor<uint8_t> relay)
{
    LocalTensor<int16_t> selectedIndices =
        selectedIndexBuf_.Get<int16_t>();
    LocalTensor<int32_t> dstSlots = dstSlotBuf_.Get<int32_t>();
    const uint32_t copyBytes = static_cast<uint32_t>(rowBytes_);
    const DataCopyExtParams copyOut {1U, copyBytes, 0U, 0U, 0U};
    for (uint32_t row = 0U; row < selectedCount; ++row) {
        const uint32_t relativeIndex = static_cast<uint16_t>(
            selectedIndices.GetValue(selectedStart + row));
        const uint64_t encoded = static_cast<uint64_t>(
            dstSlots.GetValue(relativeIndex));
        const uint32_t targetSlot = static_cast<uint32_t>(
            encoded - peerBase);
        GlobalTensor<uint8_t> target;
        target.SetGlobalBuffer(scratch_ +
            static_cast<uint64_t>(targetSlot) * rowBytes_);
        DataCopyPad(target, relay[row * localRowStride], copyOut);
    }
}

__aicore__ inline void MoonEpCombineV2::CopySelfTileIn(
    uint32_t selectedIndex, uint32_t chunkStart, uint64_t rowOffset,
    uint32_t tileBytes, LocalTensor<uint8_t> relay)
{
    const uint32_t relativeIndex = static_cast<uint16_t>(
        selectedIndexBuf_.Get<int16_t>().GetValue(selectedIndex));
    const uint32_t sourceSlotIndex = chunkStart + relativeIndex;
    GlobalTensor<uint8_t> source;
    source.SetGlobalBuffer(workspace_ +
        static_cast<uint64_t>(sourceSlotIndex) * rowBytes_ + rowOffset);
    const DataCopyExtParams copyIn {1U, tileBytes, 0U, 0U, 0U};
    const DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0U};
    DataCopyPad(relay, source, copyIn, padIn);
}

__aicore__ inline void MoonEpCombineV2::CopySelfTileOut(
    uint32_t selectedIndex, uint64_t peerBase, uint64_t rowOffset,
    uint32_t tileBytes, LocalTensor<uint8_t> relay)
{
    const uint32_t relativeIndex = static_cast<uint16_t>(
        selectedIndexBuf_.Get<int16_t>().GetValue(selectedIndex));
    const uint64_t encoded = static_cast<uint64_t>(
        dstSlotBuf_.Get<int32_t>().GetValue(relativeIndex));
    const uint32_t targetSlot = static_cast<uint32_t>(encoded - peerBase);
    GlobalTensor<uint8_t> target;
    target.SetGlobalBuffer(scratch_ +
        static_cast<uint64_t>(targetSlot) * rowBytes_ + rowOffset);
    const DataCopyExtParams copyOut {1U, tileBytes, 0U, 0U, 0U};
    DataCopyPad(target, relay, copyOut);
}

__aicore__ inline bool MoonEpCombineV2::CopySelfSelectedIndices(
    uint32_t selectedCount, uint32_t chunkStart, uint64_t peerBase)
{
    if (selectedCount == 0U) {
        return true;
    }
    const uint64_t localRowStride =
        TileXRMoonEp::MoonEpCombineV2SelfAlignedRowBytes(rowBytes_);
    if (kEnableSafetyChecks && (localRowStride == 0U ||
        selectedCount >
            TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows)) {
        return false;
    }

    const uint32_t rowsPerBatch =
        TileXRMoonEp::MoonEpCombineV2SelfRowsPerBatch(rowBytes_);
    if (rowsPerBatch != 0U) {
        const uint32_t rowStride = static_cast<uint32_t>(localRowStride);
        uint32_t pendingStart = 0U;
        uint32_t pendingRows = selectedCount < rowsPerBatch ?
            selectedCount : rowsPerBatch;
        LocalTensor<uint8_t> first = selfCopyQueue_.AllocTensor<uint8_t>();
        CopySelfRowsIn(pendingStart, pendingRows, chunkStart,
            rowStride, first);
        selfCopyQueue_.EnQue(first);

        uint32_t nextStart = pendingRows;
        while (nextStart < selectedCount) {
            LocalTensor<uint8_t> pending =
                selfCopyQueue_.DeQue<uint8_t>();
            SyncFunc<HardEvent::MTE2_MTE3>();

            const uint32_t remaining = selectedCount - nextStart;
            const uint32_t currentRows = remaining < rowsPerBatch ?
                remaining : rowsPerBatch;
            LocalTensor<uint8_t> current =
                selfCopyQueue_.AllocTensor<uint8_t>();
            CopySelfRowsIn(nextStart, currentRows, chunkStart,
                rowStride, current);
            CopySelfRowsOut(pendingStart, pendingRows, peerBase,
                rowStride, pending);
            SyncFunc<HardEvent::MTE3_S>();
            selfCopyQueue_.FreeTensor(pending);
            selfCopyQueue_.EnQue(current);

            pendingStart = nextStart;
            pendingRows = currentRows;
            nextStart += currentRows;
        }

        LocalTensor<uint8_t> pending = selfCopyQueue_.DeQue<uint8_t>();
        SyncFunc<HardEvent::MTE2_MTE3>();
        CopySelfRowsOut(pendingStart, pendingRows, peerBase,
            rowStride, pending);
        SyncFunc<HardEvent::MTE3_S>();
        selfCopyQueue_.FreeTensor(pending);
        return true;
    }

    const uint64_t tileCapacity =
        TileXRMoonEp::kMoonEpCombineV2SelfRelayHalfBytes;
    const uint32_t tilesPerRow = static_cast<uint32_t>(
        TileXRMoonEp::MoonEpCombineV2SelfTileCount(rowBytes_));
    const uint64_t totalTiles =
        static_cast<uint64_t>(selectedCount) * tilesPerRow;
    uint32_t pendingRoute = 0U;
    uint64_t pendingOffset = 0U;
    uint32_t pendingBytes =
        TileXRMoonEp::MoonEpCombineV2SelfTileBytes(rowBytes_, 0U);
    LocalTensor<uint8_t> first = selfCopyQueue_.AllocTensor<uint8_t>();
    CopySelfTileIn(pendingRoute, chunkStart, pendingOffset,
        pendingBytes, first);
    selfCopyQueue_.EnQue(first);

    for (uint64_t task = 1U; task < totalTiles; ++task) {
        LocalTensor<uint8_t> pending = selfCopyQueue_.DeQue<uint8_t>();
        SyncFunc<HardEvent::MTE2_MTE3>();

        const uint32_t currentRoute = static_cast<uint32_t>(task / tilesPerRow);
        const uint32_t tileInRow = static_cast<uint32_t>(task % tilesPerRow);
        const uint64_t currentOffset =
            static_cast<uint64_t>(tileInRow) * tileCapacity;
        const uint32_t currentBytes =
            TileXRMoonEp::MoonEpCombineV2SelfTileBytes(
                rowBytes_, tileInRow);
        LocalTensor<uint8_t> current = selfCopyQueue_.AllocTensor<uint8_t>();
        CopySelfTileIn(currentRoute, chunkStart, currentOffset,
            currentBytes, current);
        CopySelfTileOut(pendingRoute, peerBase, pendingOffset,
            pendingBytes, pending);
        SyncFunc<HardEvent::MTE3_S>();
        selfCopyQueue_.FreeTensor(pending);
        selfCopyQueue_.EnQue(current);

        pendingRoute = currentRoute;
        pendingOffset = currentOffset;
        pendingBytes = currentBytes;
    }

    LocalTensor<uint8_t> pending = selfCopyQueue_.DeQue<uint8_t>();
    SyncFunc<HardEvent::MTE2_MTE3>();
    CopySelfTileOut(pendingRoute, peerBase, pendingOffset,
        pendingBytes, pending);
    SyncFunc<HardEvent::MTE3_S>();
    selfCopyQueue_.FreeTensor(pending);
    return true;
}

__aicore__ inline bool MoonEpCombineV2::SubmitSelfGrant(uint32_t step)
{
    const uint32_t successor = TileXRMoonEp::MoonEpCombineV2Successor(
        rank_, step, core_, rankSize_, kCombineV2ScheduleMode);
    if (successor == rank_) {
        for (uint32_t lane = 0U; lane < 2U; ++lane) {
            PublishLocalGrant(step, lane);
        }
        return true;
    }
    LocalTensor<uint8_t> allIssue = wqeIssueBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> laneIssue[2] = {
        allIssue,
        allIssue[kSixPortIssueBytes]
    };
    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        const uint64_t grantIndex = TileXRMoonEp::MoonEpCombineV2GrantIndex(
            epoch_, core_, lane, step);
        __gm__ uint64_t *grantSource = reinterpret_cast<__gm__ uint64_t *>(
            grantBase_ + grantIndex *
                TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                TileXRMoonEp::kMoonEpCombineV2GrantSourceOffsetBytes);
        *grantSource = TileXRMoonEp::MoonEpCombineV2GrantToken(
            magic_, step, stepCount_);
        TileXR::UDMACleanCacheLines(
            reinterpret_cast<__gm__ uint8_t *>(grantSource),
            TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
        const bool appended = AppendControlWqe(
            laneIssue[lane], 0U, lane_[lane],
            TileXR::GetUDMAInfo(args_), successor,
            grantOffset_ + grantIndex *
                TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                TileXRMoonEp::kMoonEpCombineV2GrantReceiveOffsetBytes,
            grantSource,
            TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
        if (!appended) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                step, successor, lane);
            return false;
        }
        (void)appended;
        const uint32_t outstanding = lane_[lane].head - lane_[lane].tail;
        if (kEnableSafetyChecks &&
            static_cast<uint64_t>(outstanding) + 1U >=
                TileXRMoonEp::kMoonEpCombineV2MaxOutstanding) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_OUTSTANDING_LIMIT,
                step, successor, lane, outstanding + 1U);
            return false;
        }
    }

    SyncFunc<HardEvent::S_MTE3>();
    CopyIssueToSq(laneIssue[0], lane_[0], 1U);
    CopyIssueToSq(laneIssue[1], lane_[1], 1U);
    SyncFunc<HardEvent::MTE3_S>();
    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        ++lane_[lane].head;
        ++lane_[lane].completionCount;
        lane_[lane].submittedHead = lane_[lane].head;
        lane_[lane].cqTarget = TileXRMoonEp::MoonEpCombineV2NextCqTarget(
            lane_[lane].cqTail, true);
        st_dev(lane_[lane].head, reinterpret_cast<__gm__ uint32_t *>(
            lane_[lane].sq->headAddr), 0);
        st_dev(lane_[lane].completionCount,
            reinterpret_cast<__gm__ uint32_t *>(
                lane_[lane].sq->wqeCntAddr), 0);
        st_dev(lane_[lane].head, reinterpret_cast<__gm__ uint32_t *>(
            lane_[lane].sq->dbAddr), 0);
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::SendSelfStep(
    uint32_t peer, uint32_t step)
{
    if (kEnableSafetyChecks && peer != rank_) {
        return false;
    }
    const uint64_t peerBase = static_cast<uint64_t>(peer) * slots_;
    for (uint64_t chunkStart = 0U; chunkStart < slots_;
        chunkStart += TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows) {
        const uint32_t chunkElements = static_cast<uint32_t>(
            slots_ - chunkStart <
                TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows ?
            slots_ - chunkStart :
            TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows);
        uint64_t profileStart = BeginProfileMetric();
        LoadSelectionChunk(chunkStart, chunkElements);
        EndProfileMetric(
            TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_SELECTION_LOAD,
            profileStart);
        profileStart = BeginProfileMetric();
        const uint32_t selectedCount =
            SelectPeerIndices(peer, chunkElements);
        EndProfileMetric(
            TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_SELECTION_SELECT,
            profileStart);
        profileStart = BeginProfileMetric();
        const bool copied = CopySelfSelectedIndices(selectedCount,
            static_cast<uint32_t>(chunkStart), peerBase);
        EndProfileMetric(
            TileXRMoonEp::MOONEP_COMBINE_V2_METRIC_SELF_COPY,
            profileStart);
        if (kEnableSafetyChecks && !copied) {
            return false;
        }
        (void)copied;
    }
    PublishSelfDone(step);
    return SubmitSelfGrant(step);
}

__aicore__ inline bool MoonEpCombineV2::WaitInboundDone()
{
    bool ready[TileXRMoonEp::kMoonEpCombineV2RankCount]
        [TileXRMoonEp::kMoonEpCombineV2LaneCount] = {};
    uint64_t observed[TileXRMoonEp::kMoonEpCombineV2RankCount]
        [TileXRMoonEp::kMoonEpCombineV2LaneCount] = {};
    uint32_t remaining = 0U;
    for (uint32_t sourceIndex = 0U;
        sourceIndex < rankSize_; ++sourceIndex) {
        for (uint32_t lane = 0U;
            lane < TileXRMoonEp::kMoonEpCombineV2LaneCount; ++lane) {
            ready[sourceIndex][lane] =
                !TileXRMoonEp::MoonEpCombineV2DoneLaneRequired(
                    sourceIndex, rank_, lane,
                    static_cast<uint32_t>(args_->localRankSize));
            if (!ready[sourceIndex][lane]) {
                ++remaining;
            }
        }
    }
    const uint32_t conditionCount = rankSize_ *
        TileXRMoonEp::kMoonEpCombineV2LaneCount;
    uint32_t cursor = 0U;
    while (remaining != 0U) {
        for (uint32_t offset = 0U; offset < conditionCount; ++offset) {
            const uint32_t condition =
                (cursor + offset) % conditionCount;
            const uint32_t sourceIndex = condition /
                TileXRMoonEp::kMoonEpCombineV2LaneCount;
            const uint32_t lane = condition &
                (TileXRMoonEp::kMoonEpCombineV2LaneCount - 1U);
            if (ready[sourceIndex][lane]) {
                continue;
            }
            const uint32_t source = sourceIndex;
            const uint32_t step =
                TileXRMoonEp::MoonEpCombineV2ReceiveStep(
                    rank_, source, rankSize_, kCombineV2ScheduleMode);
            const uint64_t expected =
                TileXRMoonEp::MoonEpCombineV2Token(magic_, step);
            const uint64_t index = TileXRMoonEp::MoonEpCombineV2DoneIndex(
                epoch_, source, lane);
            __gm__ uint64_t *token = reinterpret_cast<__gm__ uint64_t *>(
                doneBase_ + index *
                    TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
            observed[sourceIndex][lane] = LoadToken(token);
            if (observed[sourceIndex][lane] == expected) {
                ready[sourceIndex][lane] = true;
                --remaining;
            }
        }
        cursor = (cursor + 1U) % conditionCount;
        if (TimedOut(operationStartCycles_)) {
            for (uint32_t condition = 0U;
                condition < conditionCount; ++condition) {
                const uint32_t sourceIndex = condition /
                    TileXRMoonEp::kMoonEpCombineV2LaneCount;
                const uint32_t lane = condition &
                    (TileXRMoonEp::kMoonEpCombineV2LaneCount - 1U);
                if (ready[sourceIndex][lane]) {
                    continue;
                }
                const uint32_t source = sourceIndex;
                const uint32_t step =
                    TileXRMoonEp::MoonEpCombineV2ReceiveStep(
                        rank_, source, rankSize_, kCombineV2ScheduleMode);
                SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_DONE_TIMEOUT,
                    step, source, lane, 0U,
                    TileXRMoonEp::MoonEpCombineV2Token(magic_, step),
                    observed[sourceIndex][lane]);
                return false;
            }
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::ResolveFullSyncRemoteFields(
    uint32_t targetRank, uint64_t remoteOffset,
    __ubuf__ MoonEpCombineV2RemoteFields *fields)
{
    if (targetRank >= rankSize_ || fields == nullptr ||
        !TileXR::UDMARegisteredRangeValid(registry_, targetRank,
            remoteOffset,
            TileXRMoonEp::kMoonEpCombineV2FullSyncSignalBytes)) {
        return false;
    }
    const uint32_t qp = TileXRMoonEp::MoonEpCombineV2Qp(
        core_, TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT);
    __gm__ TileXR::UDMAMemInfo *mem = TileXR::UDMAGetRemoteMemInfo(
        TileXR::GetUDMAInfo(args_), targetRank, qp);
    const uint64_t registeredBase = reinterpret_cast<uint64_t>(
        registry_->regions[targetRank].base);
    if (mem == nullptr || mem->eidAddr == 0U || !mem->tokenValueValid ||
        mem->addr == 0U || mem->addr != registeredBase ||
        remoteOffset > mem->len ||
        TileXRMoonEp::kMoonEpCombineV2FullSyncSignalBytes >
            static_cast<uint64_t>(mem->len) - remoteOffset) {
        return false;
    }
    __gm__ uint64_t *eid = reinterpret_cast<__gm__ uint64_t *>(mem->eidAddr);
    fields->remoteRowBase = reinterpret_cast<uint64_t>(
        TileXR::UDMARegisteredRemoteAddr(
            registry_, targetRank, remoteOffset));
    fields->rmtEidL = eid[0];
    fields->rmtEidH = eid[1];
    fields->tokenEn = mem->tokenValueValid;
    fields->rmtJettyType = mem->rmtJettyType;
    fields->targetHint = mem->targetHint;
    fields->tpId = mem->tpn;
    fields->rmtJettyOrSegId = mem->tid;
    fields->rmtTokenValue = mem->rmtTokenValue;
    return true;
}

__aicore__ inline void MoonEpCombineV2::PrepareFullSyncSignal(
    uint32_t boundaryId, uint32_t generation)
{
    const uint64_t index = TileXRMoonEp::MoonEpCombineV2FullSyncCoreIndex(
        epoch_, generation, core_);
    __gm__ TileXRMoonEp::MoonEpCombineV2FullSyncSignal *signal =
        reinterpret_cast<__gm__
            TileXRMoonEp::MoonEpCombineV2FullSyncSignal *>(
                fullSyncSourceBase_ + index *
                    TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    signal->magic = magic_;
    signal->marker = TileXRMoonEp::kMoonEpCombineV2FullSyncMarker;
    signal->boundaryId = boundaryId;
    signal->sourceRank = rank_;
    signal->sourceCore = core_;
    signal->rankSize = rankSize_;
    signal->reserved0 = 0U;
    signal->guard = TileXRMoonEp::MoonEpCombineV2FullSyncGuard(
        magic_, boundaryId, rank_, core_, rankSize_);
    for (uint32_t index = 0U; index < 3U; ++index) {
        signal->reserved[index] = 0U;
    }
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(signal),
        TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
}

__aicore__ inline bool MoonEpCombineV2::BuildFullSyncWqes(
    uint32_t generation, uint32_t *count)
{
    if (count == nullptr) {
        return false;
    }
    *count = 0U;
    __ubuf__ MoonEpCombineV2FullSyncBuildContext *context =
        reinterpret_cast<__ubuf__ MoonEpCombineV2FullSyncBuildContext *>(
            dstSlotBuf_.Get<uint8_t>().GetPhyAddr());
    const uint64_t sourceIndex =
        TileXRMoonEp::MoonEpCombineV2FullSyncCoreIndex(
            epoch_, generation, core_);
    context->localSignalAddr = reinterpret_cast<uint64_t>(
        fullSyncSourceBase_ + sourceIndex *
            TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    context->head = lane_[TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT].head;
    for (uint32_t step = 0U; step < stepCount_; ++step) {
        const uint32_t peer = TileXRMoonEp::MoonEpCombineV2Peer(
            rank_, step, core_, rankSize_, kCombineV2ScheduleMode);
        if (peer == rank_ ||
            peer == TileXRMoonEp::kMoonEpCombineV2InvalidPeer) {
            continue;
        }
        if (*count >=
            TileXRMoonEp::kMoonEpCombineV2FullSyncMaxPeersPerCore) {
            return false;
        }
        const uint64_t receiveIndex =
            TileXRMoonEp::MoonEpCombineV2FullSyncReceiveIndex(
                epoch_, generation, rank_);
        const uint64_t remoteOffset = fullSyncReceiveOffset_ + receiveIndex *
            TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes;
        if (!ResolveFullSyncRemoteFields(
                peer, remoteOffset, &context->remote[*count])) {
            return false;
        }
        ++(*count);
    }
    context->count = *count;
    if (*count == 0U) {
        return true;
    }
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    PipeBarrier<PIPE_ALL>();
    LocalTensor<uint8_t> issue = wqeIssueBuf_.Get<uint8_t>();
    Simt::VF_CALL<MoonEpCombineV2BuildFullSyncWqesVf>(
        Simt::Dim3{
            TileXRMoonEp::kMoonEpCombineV2BuilderThreads,
            1U, 1U},
        reinterpret_cast<__ubuf__ uint8_t *>(issue.GetPhyAddr()), context);
    PipeBarrier<PIPE_ALL>();
#endif
    return true;
}

__aicore__ inline bool MoonEpCombineV2::PublishFullSyncBatch(uint32_t count)
{
    if (count == 0U) {
        return true;
    }
    MoonEpCombineV2LaneState &state =
        lane_[TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT];
    const uint32_t outstanding = state.head - state.tail;
    if (static_cast<uint64_t>(outstanding) + count >=
        TileXRMoonEp::kMoonEpCombineV2MaxOutstanding) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_OUTSTANDING_LIMIT,
            UINT32_MAX, UINT32_MAX,
            TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT, outstanding + count);
        return false;
    }
    SyncFunc<HardEvent::S_MTE3>();
    CopyIssueToSq(wqeIssueBuf_.Get<uint8_t>(), state, count);
    SyncFunc<HardEvent::MTE3_S>();
    state.head += count;
    ++state.completionCount;
    state.submittedHead = state.head;
    state.cqTarget = TileXRMoonEp::MoonEpCombineV2NextCqTarget(
        state.cqTail, true);
    st_dev(state.head, reinterpret_cast<__gm__ uint32_t *>(
        state.sq->headAddr), 0);
    st_dev(state.completionCount, reinterpret_cast<__gm__ uint32_t *>(
        state.sq->wqeCntAddr), 0);
    st_dev(state.head, reinterpret_cast<__gm__ uint32_t *>(
        state.sq->dbAddr), 0);
    return true;
}

__aicore__ inline bool MoonEpCombineV2::WaitFullSyncCq()
{
    MoonEpCombineV2LaneState &state =
        lane_[TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT];
    while (!TileXRMoonEp::MoonEpCombineV2CqTargetReached(
        state.cqTail, state.cqTarget)) {
        const uint32_t status = PollCqOnce(state);
        if (status != 0U && status != kPollNoCompletion) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
                UINT32_MAX, UINT32_MAX,
                TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT, status,
                state.cqTarget, state.cqTail,
                TileXRMoonEp::MoonEpCombineV2Qp(core_,
                    TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT));
            return false;
        }
        if (TimedOut(fullSyncStartCycles_)) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_TIMEOUT,
                UINT32_MAX, UINT32_MAX,
                TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT, 0U,
                state.cqTarget, state.cqTail,
                TileXRMoonEp::MoonEpCombineV2Qp(core_,
                    TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT));
            return false;
        }
    }
    if (state.cqTail != state.cqTarget ||
        state.tail != state.submittedHead || state.head != state.tail) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
            UINT32_MAX, UINT32_MAX,
            TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT,
            kPollInvalidState, state.submittedHead, state.tail,
            TileXRMoonEp::MoonEpCombineV2Qp(core_,
                TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT));
        return false;
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::FullSyncSignalMatches(
    __gm__ TileXRMoonEp::MoonEpCombineV2FullSyncSignal *signal,
    uint32_t boundaryId, uint32_t sourceRank, uint32_t sourceCore,
    uint64_t *observedMagic)
{
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(signal),
        TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    if (observedMagic != nullptr) {
        *observedMagic = signal->magic;
    }
    return signal->magic == magic_ &&
        signal->marker == TileXRMoonEp::kMoonEpCombineV2FullSyncMarker &&
        signal->boundaryId == boundaryId &&
        signal->sourceRank == sourceRank &&
        signal->sourceCore == sourceCore && signal->rankSize == rankSize_ &&
        signal->guard == TileXRMoonEp::MoonEpCombineV2FullSyncGuard(
            magic_, boundaryId, sourceRank, sourceCore, rankSize_);
}

__aicore__ inline bool MoonEpCombineV2::WaitFullSyncSources(
    uint32_t boundaryId, uint32_t generation, uint32_t timeoutStatus)
{
    bool ready[TileXRMoonEp::kMoonEpCombineV2FullSyncMaxPeersPerCore] = {};
    uint64_t observed[
        TileXRMoonEp::kMoonEpCombineV2FullSyncMaxPeersPerCore] = {};
    uint32_t peer[TileXRMoonEp::kMoonEpCombineV2FullSyncMaxPeersPerCore] = {};
    uint32_t sourceCore[
        TileXRMoonEp::kMoonEpCombineV2FullSyncMaxPeersPerCore] = {};
    uint32_t remaining = 0U;
    for (uint32_t step = 0U; step < stepCount_; ++step) {
        const uint32_t remote = TileXRMoonEp::MoonEpCombineV2Peer(
            rank_, step, core_, rankSize_, kCombineV2ScheduleMode);
        if (remote == rank_ ||
            remote == TileXRMoonEp::kMoonEpCombineV2InvalidPeer) {
            continue;
        }
        peer[remaining] = remote;
        sourceCore[remaining] = TileXRMoonEp::MoonEpCombineV2SenderCore(
            remote, rank_, rankSize_, kCombineV2ScheduleMode);
        ++remaining;
    }
    const uint32_t peerCount = remaining;
    uint32_t cursor = 0U;
    while (remaining != 0U) {
        for (uint32_t offset = 0U; offset < peerCount; ++offset) {
            const uint32_t peerIndex = (cursor + offset) % peerCount;
            if (ready[peerIndex]) {
                continue;
            }
            const uint64_t index =
                TileXRMoonEp::MoonEpCombineV2FullSyncReceiveIndex(
                    epoch_, generation, peer[peerIndex]);
            __gm__ TileXRMoonEp::MoonEpCombineV2FullSyncSignal *signal =
                reinterpret_cast<__gm__
                    TileXRMoonEp::MoonEpCombineV2FullSyncSignal *>(
                        fullSyncReceiveBase_ + index *
                            TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
            if (sourceCore[peerIndex] !=
                    TileXRMoonEp::kMoonEpCombineV2InvalidPeer &&
                FullSyncSignalMatches(signal, boundaryId, peer[peerIndex],
                    sourceCore[peerIndex], &observed[peerIndex])) {
                ready[peerIndex] = true;
                --remaining;
            }
        }
        cursor = (cursor + 1U) % peerCount;
        if (TimedOut(fullSyncStartCycles_)) {
            for (uint32_t peerIndex = 0U;
                peerIndex < peerCount; ++peerIndex) {
                if (ready[peerIndex]) {
                    continue;
                }
                SetFailure(timeoutStatus, boundaryId, peer[peerIndex],
                    TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT, 0U,
                    TileXRMoonEp::MoonEpCombineV2FullSyncGuard(
                        magic_, boundaryId, peer[peerIndex],
                        sourceCore[peerIndex], rankSize_),
                    observed[peerIndex]);
                return false;
            }
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::BeginCollectiveStage(
    uint32_t stageId)
{
    const uint64_t index = TileXRMoonEp::MoonEpCombineV2CollectiveStatusIndex(
        epoch_, stageId);
    __gm__ TileXRMoonEp::MoonEpCombineV2CollectiveStatus *status =
        reinterpret_cast<__gm__
            TileXRMoonEp::MoonEpCombineV2CollectiveStatus *>(
                fullSyncBarrierBase_ + index *
                    TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    if (core_ == 0U) {
        status->magic = magic_;
        status->guard = TileXRMoonEp::MoonEpCombineV2CollectiveStatusGuard(
            magic_, stageId);
        status->marker = TileXRMoonEp::kMoonEpCombineV2CollectiveStatusMarker;
        status->stageId = stageId;
        status->status = TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS;
        status->firstFailureCore = UINT32_MAX;
        for (uint32_t reserved = 0U; reserved < 4U; ++reserved) {
            status->reserved[reserved] = 0U;
        }
        TileXR::UDMACleanCacheLines(
            reinterpret_cast<__gm__ uint8_t *>(status),
            TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    }
    AscendC::SyncAll<true>();
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(status),
        TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    const bool ready = status->magic == magic_ &&
        status->marker == TileXRMoonEp::kMoonEpCombineV2CollectiveStatusMarker &&
        status->stageId == stageId &&
        status->guard == TileXRMoonEp::MoonEpCombineV2CollectiveStatusGuard(
            magic_, stageId);
    if (!ready) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_COLLECTIVE_STATUS_ERROR,
            stageId, rank_, UINT32_MAX, 0U,
            TileXRMoonEp::MoonEpCombineV2CollectiveStatusGuard(
                magic_, stageId), status->guard);
    }
    return ready;
}

__aicore__ inline bool MoonEpCombineV2::EndCollectiveStage(
    uint32_t stageId, bool localSucceeded)
{
    const uint64_t index = TileXRMoonEp::MoonEpCombineV2CollectiveStatusIndex(
        epoch_, stageId);
    __gm__ TileXRMoonEp::MoonEpCombineV2CollectiveStatus *status =
        reinterpret_cast<__gm__
            TileXRMoonEp::MoonEpCombineV2CollectiveStatus *>(
                fullSyncBarrierBase_ + index *
                    TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    if (!localSucceeded) {
        if (failureStatus_ == TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS) {
            SetFailure(
                TileXRMoonEp::MOONEP_COMBINE_V2_COLLECTIVE_STATUS_ERROR,
                stageId, rank_, UINT32_MAX);
        }
        PublishFailureRecord(stageId);
        (void)AscendC::AtomicCas(&status->firstFailureCore, UINT32_MAX, core_);
        (void)AscendC::AtomicCas(&status->status,
            static_cast<uint32_t>(TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS),
            failureStatus_);
    }
    AscendC::SyncAll<true>();
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(status),
        TileXRMoonEp::kMoonEpCombineV2FullSyncSlotBytes);
    const bool succeeded = status->magic == magic_ &&
        status->marker == TileXRMoonEp::kMoonEpCombineV2CollectiveStatusMarker &&
        status->stageId == stageId &&
        status->guard == TileXRMoonEp::MoonEpCombineV2CollectiveStatusGuard(
            magic_, stageId) &&
        status->status == TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS;
    AscendC::SyncAll<true>();
    return succeeded;
}

__aicore__ inline bool MoonEpCombineV2::RunGlobalBarrier(
    uint32_t boundaryId, uint32_t timeoutStatus, bool localSucceeded)
{
    const bool stageReady = BeginCollectiveStage(boundaryId);
    bool workerSucceeded = localSucceeded && stageReady;
    fullSyncStartCycles_ = static_cast<uint64_t>(GetSystemCycle());
    if (activeWorker_ && workerSucceeded) {
        const uint32_t generation =
            TileXRMoonEp::MoonEpCombineV2FullSyncGeneration(boundaryId);
        uint32_t count = 0U;
        workerSucceeded = BuildFullSyncWqes(generation, &count);
        if (!workerSucceeded) {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                boundaryId, UINT32_MAX,
                TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT);
        }
        if (workerSucceeded && count != 0U) {
            PrepareFullSyncSignal(boundaryId, generation);
        }
        if (workerSucceeded) {
            workerSucceeded = PublishFullSyncBatch(count);
        }
        if (workerSucceeded) {
            workerSucceeded = WaitFullSyncCq();
        }
        if (workerSucceeded) {
            workerSucceeded = WaitFullSyncSources(
                boundaryId, generation, timeoutStatus);
        }
    }
    return EndCollectiveStage(
        boundaryId, !activeWorker_ || workerSucceeded);
}

__aicore__ inline void MoonEpCombineV2::InitReduceBuffers(
    uint32_t inputBufferNum, uint32_t inputSlotBytes,
    uint32_t floatRowBytes, uint32_t outputSlotBytes)
{
    constexpr uint32_t maxFloatRowBytes =
        kReduceMaxRowElements * sizeof(float);
    static_assert(kReduceMaxInputBuffers * kReduceMaxRowBytes +
            2U * maxFloatRowBytes +
            kReduceOutputBuffers * kReduceMaxRowBytes <= kFullUbBytes,
        "Combine V2 multi-buffer reduction exceeds UB");
    pipe_->Reset();
    pipe_->InitBuffer(reduceInputQueue_, inputBufferNum, inputSlotBytes);
    pipe_->InitBuffer(reduceOutputQueue_, kReduceOutputBuffers,
        outputSlotBytes);
    pipe_->InitBuffer(reduceRowBuf_, floatRowBytes);
    pipe_->InitBuffer(reduceAccumulatorBuf_, floatRowBytes);
}

__aicore__ inline void MoonEpCombineV2::CopyReduceInput(
    uint64_t workOrdinal, int64_t tokenBegin, uint64_t rowCount)
{
    const uint64_t topK = static_cast<uint64_t>(topK_);
    const uint64_t groupOrdinal = workOrdinal / topK;
    const uint64_t route = workOrdinal % topK;
    const uint64_t tokenLocal = groupOrdinal / rowCount;
    const uint64_t hiddenRow = groupOrdinal % rowCount;
    const uint64_t token = static_cast<uint64_t>(tokenBegin) + tokenLocal;
    const uint64_t hiddenOffset = hiddenRow * kReduceMaxRowElements;
    const uint32_t rowElements = static_cast<uint32_t>(
        static_cast<uint64_t>(h_) - hiddenOffset <
                kReduceMaxRowElements ?
            static_cast<uint64_t>(h_) - hiddenOffset :
            kReduceMaxRowElements);

    LocalTensor<bfloat16_t> inputRow =
        reduceInputQueue_.AllocTensor<bfloat16_t>();
    GlobalTensor<bfloat16_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
        scratch_ + (token * topK + route) * rowBytes_) + hiddenOffset,
        rowElements);
    const DataCopyExtParams copyIn {
        1U, static_cast<uint32_t>(rowElements * sizeof(bfloat16_t)),
        0U, 0U, 0U};
    const DataCopyPadExtParams<bfloat16_t> pad {false, 0U, 0U, 0U};
    DataCopyPad(inputRow, input, copyIn, pad);
    reduceInputQueue_.EnQue(inputRow);
}

__aicore__ inline bool MoonEpCombineV2::ReduceHidden()
{
    if (!reduceHidden_) {
        return true;
    }
    const int64_t tokenBegin = bs_ * core_ / activeCoreCount_;
    const int64_t tokenEnd = bs_ * (core_ + 1U) / activeCoreCount_;
    if (tokenBegin >= tokenEnd) {
        return true;
    }
    const uint64_t rowCapacityElements = static_cast<uint64_t>(h_) <
            kReduceMaxRowElements ?
        static_cast<uint64_t>(h_) : kReduceMaxRowElements;
    const uint64_t rowCount = CeilDiv(
        static_cast<uint64_t>(h_), kReduceMaxRowElements);
    const uint32_t inputSlotBytes = static_cast<uint32_t>(AlignUp(
        rowCapacityElements * sizeof(bfloat16_t), kUbAlignBytes));
    const uint32_t floatRowBytes = static_cast<uint32_t>(AlignUp(
        rowCapacityElements * sizeof(float), kUbAlignBytes));
    const uint32_t fixedBytes = 2U * floatRowBytes +
        kReduceOutputBuffers * inputSlotBytes;
    if (fixedBytes >= kFullUbBytes) {
        return false;
    }
    const uint64_t totalRouteWorkItems =
        static_cast<uint64_t>(tokenEnd - tokenBegin) * rowCount *
        static_cast<uint64_t>(topK_);
    uint32_t inputBufferNum = static_cast<uint32_t>(
        (kFullUbBytes - fixedBytes) / inputSlotBytes);
    inputBufferNum = inputBufferNum < kReduceMaxInputBuffers ?
        inputBufferNum : kReduceMaxInputBuffers;
    inputBufferNum = totalRouteWorkItems < inputBufferNum ?
        static_cast<uint32_t>(totalRouteWorkItems) : inputBufferNum;
    if (inputBufferNum == 0U) {
        return false;
    }

    PipeBarrier<PIPE_ALL>();
    InitReduceBuffers(inputBufferNum, inputSlotBytes, floatRowBytes,
        inputSlotBytes);
    LocalTensor<float> row = reduceRowBuf_.Get<float>();
    LocalTensor<float> accumulator = reduceAccumulatorBuf_.Get<float>();

    uint64_t issueOrdinal = 0U;
    for (; issueOrdinal < inputBufferNum; ++issueOrdinal) {
        CopyReduceInput(issueOrdinal, tokenBegin, rowCount);
    }

    const uint64_t topK = static_cast<uint64_t>(topK_);
    for (uint64_t consumeOrdinal = 0U;
        consumeOrdinal < totalRouteWorkItems; ++consumeOrdinal) {
        const uint64_t groupOrdinal = consumeOrdinal / topK;
        const uint64_t route = consumeOrdinal % topK;
        const uint64_t tokenLocal = groupOrdinal / rowCount;
        const uint64_t hiddenRow = groupOrdinal % rowCount;
        const uint64_t hiddenOffset = hiddenRow * kReduceMaxRowElements;
        const uint32_t rowElements = static_cast<uint32_t>(
            static_cast<uint64_t>(h_) - hiddenOffset <
                    kReduceMaxRowElements ?
                static_cast<uint64_t>(h_) - hiddenOffset :
                kReduceMaxRowElements);

        LocalTensor<bfloat16_t> inputRow =
            reduceInputQueue_.DeQue<bfloat16_t>();
        if (route == 0U) {
            Duplicate(accumulator, 0.0f,
                static_cast<int32_t>(rowElements));
            PipeBarrier<PIPE_V>();
        }
        Cast(row, inputRow, RoundMode::CAST_NONE,
            static_cast<int32_t>(rowElements));
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, row,
            static_cast<int32_t>(rowElements));
        PipeBarrier<PIPE_V>();
        reduceInputQueue_.FreeTensor(inputRow);

        if (issueOrdinal < totalRouteWorkItems) {
            CopyReduceInput(issueOrdinal, tokenBegin, rowCount);
            ++issueOrdinal;
        }

        if (route + 1U != topK) {
            continue;
        }
        LocalTensor<bfloat16_t> outputRow =
            reduceOutputQueue_.AllocTensor<bfloat16_t>();
        Cast(outputRow, accumulator, RoundMode::CAST_RINT,
            static_cast<int32_t>(rowElements));
        reduceOutputQueue_.EnQue(outputRow);
        outputRow = reduceOutputQueue_.DeQue<bfloat16_t>();
        GlobalTensor<bfloat16_t> output;
        const uint64_t token =
            static_cast<uint64_t>(tokenBegin) + tokenLocal;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            workspace_ + outputOffset_) + token *
                static_cast<uint64_t>(h_) + hiddenOffset,
            rowElements);
        const DataCopyExtParams copyOut {
            1U, static_cast<uint32_t>(rowElements * sizeof(bfloat16_t)),
            0U, 0U, 0U};
        DataCopyPad(output, outputRow, copyOut);
        reduceOutputQueue_.FreeTensor(outputRow);
    }
    PipeBarrier<PIPE_ALL>();
    return true;
}

__aicore__ inline void MoonEpCombineV2::RecordProfilePoint(
    uint32_t index)
{
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    if (index < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCapacity) {
        profileTimePoint_[index] = static_cast<int64_t>(GetSystemCycle());
    }
#else
    (void)index;
    return;
#endif
}

__aicore__ inline uint64_t MoonEpCombineV2::BeginProfileMetric()
{
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    return static_cast<uint64_t>(GetSystemCycle());
#else
    return 0U;
#endif
}

__aicore__ inline void MoonEpCombineV2::EndProfileMetric(
    uint32_t index, uint64_t startCycles)
{
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    if (index < TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount) {
        profileMetric_[index] +=
            static_cast<uint64_t>(GetSystemCycle()) - startCycles;
    }
#else
    (void)index;
    (void)startCycles;
    return;
#endif
}

__aicore__ inline void MoonEpCombineV2::WriteProfile()
{
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    if (profileTimePoint_[
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FINAL_END] == 0) {
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FINAL_END);
    }
    int64_t lastPoint = profileTimePoint_[
        TileXRMoonEp::MOONEP_COMBINE_V2_TIME_INIT_BEGIN];
    for (uint32_t index = 1U;
        index < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount;
        ++index) {
        if (profileTimePoint_[index] == 0) {
            profileTimePoint_[index] = lastPoint;
        } else {
            lastPoint = profileTimePoint_[index];
        }
    }

    __gm__ TileXRMoonEp::MoonEpCombineV2ProfileRecord *record =
        reinterpret_cast<__gm__
            TileXRMoonEp::MoonEpCombineV2ProfileRecord *>(
                workspace_ + profileOffset_ +
            static_cast<uint64_t>(core_) *
                sizeof(TileXRMoonEp::MoonEpCombineV2ProfileRecord));
    record->marker = TileXRMoonEp::kMoonEpCombineV2ProfileMarker;
    record->version = TileXRMoonEp::kMoonEpCombineV2ProfileVersion;
    record->recordBytes =
        sizeof(TileXRMoonEp::MoonEpCombineV2ProfileRecord);
    record->rank = rank_;
    record->core = core_;
    record->blockDim = launchCoreCount_;
    record->timePointCount =
        TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount;
    record->metricCount = TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount;
    record->reserved = 0U;
    for (uint32_t index = 0U;
        index < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCapacity;
        ++index) {
        record->timePoint[index] = profileTimePoint_[index];
    }
    for (uint32_t index = 0U;
        index < TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount;
        ++index) {
        record->metric[index] = profileMetric_[index];
    }
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_FAILURE_STATUS] =
        static_cast<int64_t>(failureStatus_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_FAILURE_STEP] =
        static_cast<int64_t>(failureStep_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_FAILURE_PEER] =
        static_cast<int64_t>(failurePeer_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_FAILURE_LANE] =
        static_cast<int64_t>(failureLane_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_FAILURE_QP] =
        failureQp_ != UINT32_MAX ? static_cast<int64_t>(failureQp_) :
            (failureLane_ < TileXRMoonEp::kMoonEpCombineV2LaneCount ?
                static_cast<int64_t>(TileXRMoonEp::MoonEpCombineV2Qp(
                    core_, failureLane_)) : -1);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_CQ_STATUS] =
        static_cast<int64_t>(failureCqStatus_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_EXPECTED] =
        static_cast<int64_t>(failureExpected_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_OBSERVED] =
        static_cast<int64_t>(failureObserved_);
    record->reserved = fullmeshProfileRoute_;
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(record),
        sizeof(*record));
#else
    return;
#endif
}

__aicore__ inline void MoonEpCombineV2::Process()
{
    AscendC::SyncAll<true>();
    if (!collectiveReady_) {
        return;
    }

    bool stageReady = BeginCollectiveStage(kCollectiveInitStage);
    bool localSucceeded = stageReady && valid_;
    if (activeWorker_ && localSucceeded) {
        registry_ = TileXR::GetUDMARegistry(args_);
        localSucceeded = registry_ != nullptr;
        if (localSucceeded) {
            InitBuffers();
        } else {
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                UINT32_MAX, UINT32_MAX, UINT32_MAX);
        }
    }
    bool succeeded = EndCollectiveStage(kCollectiveInitStage,
        !activeWorker_ || localSucceeded);
    if (!succeeded) {
        WriteProfile();
        return;
    }

    stageReady = BeginCollectiveStage(kCollectiveValidationStage);
    localSucceeded = stageReady;
    if (activeWorker_ && localSucceeded && kEnableSafetyChecks) {
        localSucceeded = failureStatus_ ==
            TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS && CheckPoison();
        if (localSucceeded) {
            localSucceeded = ValidateDestinations();
        }
    }
    succeeded = EndCollectiveStage(kCollectiveValidationStage,
        !activeWorker_ || localSucceeded);
    RecordProfilePoint(TileXRMoonEp::MOONEP_COMBINE_V2_TIME_PREPARE_END);
    if (!succeeded) {
        WriteProfile();
        return;
    }

    if (kEnableFullSync) {
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_BEGIN);
        succeeded = RunGlobalBarrier(0U,
            TileXRMoonEp::MOONEP_COMBINE_V2_FULL_SYNC_TIMEOUT, succeeded);
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_SUBMIT_END);
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_RECEIVE_END);
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_BARRIER_END);
        if (!succeeded) {
            WriteProfile();
            return;
        }
    } else {
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_BEGIN);
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_SUBMIT_END);
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_RECEIVE_END);
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FULL_SYNC_BARRIER_END);
    }
    if (activeWorker_) {
        PrefillOperatorWqes();
    }
    for (uint32_t step = 0U; step < stepCount_; ++step) {
        localSucceeded = succeeded;
        bool fullmeshStep = false;
        if (activeWorker_ && localSucceeded) {
            const uint32_t peer = TileXRMoonEp::MoonEpCombineV2Peer(
                rank_, step, core_, rankSize_, kCombineV2ScheduleMode);
            const bool selfStep = peer == rank_ ||
                peer == TileXRMoonEp::kMoonEpCombineV2InvalidPeer;
            fullmeshStep = !selfStep &&
                TileXRMoonEp::MoonEpCombineV2SameServer(rank_, peer,
                    static_cast<uint32_t>(args_->localRankSize));
            if (selfStep) {
                localSucceeded = SendSelfStep(rank_, step);
            } else if (fullmeshStep) {
                const uint32_t successor =
                    TileXRMoonEp::MoonEpCombineV2Successor(
                        rank_, step, core_, rankSize_, kCombineV2ScheduleMode);
                fullmeshProfileRoute_ =
                    TileXRMoonEp::MoonEpCombineV2PackFullmeshProfileRoute(
                        step, peer, successor,
                        TileXRMoonEp::MoonEpCombineV2FullmeshLogicalQp(
                            peer, static_cast<uint32_t>(
                                args_->localRankSize)));
                localSucceeded = SendFullmeshStep(peer, step);
                if (localSucceeded) {
                    localSucceeded = WaitFullmeshCq(step, peer);
                }
                if (localSucceeded) {
                    RecordProfilePoint(TileXRMoonEp::
                        MOONEP_COMBINE_V2_DIAG_FULLMESH_CQ_SUCCESS);
                    localSucceeded = SubmitSelfGrant(step);
                }
                if (localSucceeded) {
                    RecordProfilePoint(TileXRMoonEp::
                        MOONEP_COMBINE_V2_DIAG_FULLMESH_GRANT_SUBMIT);
                }
            } else {
                localSucceeded = SendRemoteStep(peer, step);
            }
        }
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_STEP0_SEND_END +
            step * TileXRMoonEp::kMoonEpCombineV2ProfileStepPointStride);
        if (activeWorker_ && localSucceeded) {
            localSucceeded = WaitStepCqs(step);
        }
        if (activeWorker_ && localSucceeded && fullmeshStep) {
            RecordProfilePoint(TileXRMoonEp::
                MOONEP_COMBINE_V2_DIAG_FULLMESH_GRANT_CQ_SUCCESS);
        }
        if (activeWorker_ && localSucceeded) {
            localSucceeded = WaitStepGrant(step);
        }
        const uint32_t boundaryId =
            TileXRMoonEp::MoonEpCombineV2RoundBoundary(step, rankSize_);
        succeeded = RunGlobalBarrier(boundaryId,
            TileXRMoonEp::MOONEP_COMBINE_V2_STEP_BARRIER_TIMEOUT,
            localSucceeded);
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_STEP0_READY_END +
            step * TileXRMoonEp::kMoonEpCombineV2ProfileStepPointStride);
        if (!succeeded) {
            break;
        }
    }
    if (succeeded) {
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_STEP_LOOP_END);
    }
    stageReady = BeginCollectiveStage(kCollectiveDoneStage);
    localSucceeded = succeeded && stageReady;
    if (activeWorker_ && localSucceeded) {
        localSucceeded = WaitInboundDone();
    }
    succeeded = EndCollectiveStage(kCollectiveDoneStage,
        !activeWorker_ || localSucceeded);
    RecordProfilePoint(
        TileXRMoonEp::MOONEP_COMBINE_V2_TIME_INBOUND_DONE_END);

    stageReady = BeginCollectiveStage(kCollectiveReduceStage);
    localSucceeded = succeeded && stageReady;
    if (activeWorker_ && localSucceeded && !ReduceHidden()) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            UINT32_MAX, UINT32_MAX, UINT32_MAX);
        localSucceeded = false;
    }
    succeeded = EndCollectiveStage(kCollectiveReduceStage,
        !activeWorker_ || localSucceeded);
    (void)succeeded;
    RecordProfilePoint(TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FINAL_END);
    WriteProfile();
}

} // namespace

#endif // TILEXR_MOONEP_COMBINE_V2_KERNEL_H
