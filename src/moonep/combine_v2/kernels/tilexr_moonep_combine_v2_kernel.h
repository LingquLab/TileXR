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

namespace {

using namespace AscendC;

constexpr uint32_t kUbAlignBytes = 32U;
constexpr uint32_t kFullUbBytes = 216U * 1024U;
constexpr uint32_t kRelayBytes = 4U * 1024U;
constexpr uint32_t kWqeBytes = 64U;
constexpr uint32_t kSixPortFinalCapacity =
    TileXRMoonEp::kMoonEpCombineV2SixPortRows + 2U;
constexpr uint32_t kTwoPortFinalCapacity =
    TileXRMoonEp::kMoonEpCombineV2TwoPortRows + 2U;
constexpr uint32_t kSixPortIssueBytes = kSixPortFinalCapacity * kWqeBytes;
constexpr uint32_t kTwoPortIssueBytes = kTwoPortFinalCapacity * kWqeBytes;
constexpr uint32_t kDescriptorBytes =
    TileXRMoonEp::kMoonEpCombineV2LogicalBatchRows * 2U * sizeof(uint32_t);
constexpr uint32_t kBuildContextBytes = 256U;
constexpr uint32_t kWqeBuildThreads = 64U;
constexpr uint32_t kReduceRouteBatch = 4U;
constexpr uint32_t kReduceTileElements = 4096U;
constexpr uint64_t kOperationTimeoutCycles = 10000000000ULL;
constexpr uint32_t kPollNoCompletion = UINT32_MAX;
constexpr uint32_t kPollInvalidState = UINT32_MAX - 1U;
// Trusted-input benchmark mode compiles defensive validation and diagnostics
// out of the transfer hot path.
constexpr bool kEnableSafetyChecks = true;

static_assert(sizeof(TileXR::UDMASqeCtx) + sizeof(TileXR::UDMASgeCtx) ==
    kWqeBytes, "Combine V2 WRITE WQE must occupy one basic block");
static_assert(sizeof(TileXR::UDMACqeCtx) == 64U,
    "Combine V2 CQE must occupy one cache line");
static_assert(TileXRMoonEp::kMoonEpCombineV2ProfileStepCount ==
        TileXRMoonEp::kMoonEpCombineV2StepCount,
    "Combine V2 profile must cover every schedule step");

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

struct MoonEpCombineV2Descriptor {
    uint32_t sourceSlot;
    uint32_t targetSlot;
};

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

struct alignas(32) MoonEpCombineV2BuildContext {
    uint64_t localRowBase;
    uint64_t rowBytes;
    uint32_t logicalCount;
    uint32_t sequencePhase;
    uint32_t head[TileXRMoonEp::kMoonEpCombineV2LaneCount];
    uint32_t laneCount[TileXRMoonEp::kMoonEpCombineV2LaneCount];
    MoonEpCombineV2RemoteFields remote[
        TileXRMoonEp::kMoonEpCombineV2LaneCount];
};

static_assert(sizeof(MoonEpCombineV2BuildContext) <= kBuildContextBytes,
    "Combine V2 build context exceeds its UB allocation");

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
__simt_vf__ __aicore__ LAUNCH_BOUND(kWqeBuildThreads)
inline void MoonEpCombineV2BuildPayloadVf(
    __ubuf__ uint8_t *sixPortWqes, __ubuf__ uint8_t *twoPortWqes,
    __ubuf__ const MoonEpCombineV2Descriptor *descriptors,
    __ubuf__ const MoonEpCombineV2BuildContext *context)
{
    for (uint32_t task = static_cast<uint32_t>(threadIdx.x);
        task < context->logicalCount; task += kWqeBuildThreads) {
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
        __ubuf__ uint32_t *words =
            reinterpret_cast<__ubuf__ uint32_t *>(wqe);
        for (uint32_t word = 0U; word < kWqeBytes / sizeof(uint32_t);
            ++word) {
            words[word] = 0U;
        }

        const uint64_t localAddr = context->localRowBase +
            static_cast<uint64_t>(descriptors[task].sourceSlot) *
                context->rowBytes;
        const uint64_t remoteAddr = context->remote[lane].remoteRowBase +
            static_cast<uint64_t>(descriptors[task].targetSlot) *
                context->rowBytes;
        const uint32_t absoluteHead = context->head[lane] + laneIndex;

        __ubuf__ TileXR::UDMASqeCtx *sqe =
            reinterpret_cast<__ubuf__ TileXR::UDMASqeCtx *>(wqe);
        sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);
        sqe->flag = 0U;
        sqe->nf = 0U;
        sqe->tokenEn = context->remote[lane].tokenEn;
        sqe->rmtJettyType = context->remote[lane].rmtJettyType;
        sqe->sqeBbIdx = static_cast<uint16_t>(
            absoluteHead % TileXR::TILEXR_UDMA_SQ_BB_COUNT);
        sqe->owner =
            (absoluteHead & TileXR::TILEXR_UDMA_SQ_BB_COUNT) == 0U ? 1U : 0U;
        sqe->targetHint = context->remote[lane].targetHint;
        sqe->inlineMsgLen = 0U;
        sqe->tpId = context->remote[lane].tpId;
        sqe->sgeNum = 1U;
        sqe->rmtJettyOrSegId = context->remote[lane].rmtJettyOrSegId;
        sqe->rmtTokenValue = context->remote[lane].rmtTokenValue;
        sqe->rmtEidL = context->remote[lane].rmtEidL;
        sqe->rmtEidH = context->remote[lane].rmtEidH;
        sqe->rmtAddrLOrTokenId = remoteAddr & 0xFFFFFFFFU;
        sqe->rmtAddrHOrTokenValue = (remoteAddr >> 32U) & 0xFFFFFFFFU;

        __ubuf__ TileXR::UDMASgeCtx *sge =
            reinterpret_cast<__ubuf__ TileXR::UDMASgeCtx *>(
                wqe + sizeof(TileXR::UDMASqeCtx));
        sge->len = static_cast<uint32_t>(context->rowBytes);
        sge->tokenId = 0U;
        sge->va = localAddr;
    }
}
#endif

__aicore__ inline void CopyBytesGmToGm(__gm__ uint8_t *dst,
    const __gm__ uint8_t *src, uint32_t bytes, LocalTensor<uint8_t> relay)
{
    for (uint32_t offset = 0U; offset < bytes; offset += kRelayBytes) {
        const uint32_t tileBytes = bytes - offset < kRelayBytes ?
            bytes - offset : kRelayBytes;
        GlobalTensor<uint8_t> srcGlobal;
        srcGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t *>(src) + offset);
        const DataCopyExtParams copyIn {1U, tileBytes, 0U, 0U, 0U};
        const DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0U};
        DataCopyPad(relay, srcGlobal, copyIn, padIn);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        GlobalTensor<uint8_t> dstGlobal;
        dstGlobal.SetGlobalBuffer(dst + offset);
        const DataCopyExtParams copyOut {1U, tileBytes, 0U, 0U, 0U};
        DataCopyPad(dstGlobal, relay, copyOut);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}

class MoonEpCombineV2 {
public:
    __aicore__ inline void Init(GM_ADDR commArgsGM,
        GM_ADDR registeredWorkspaceGM, GM_ADDR dstLocalGM,
        uint64_t profileOffset, uint64_t scratchEpoch0Offset,
        uint64_t scratchEpoch1Offset, uint64_t doneOffset,
        uint64_t grantOffset, uint64_t controlSourceOffset,
        uint64_t failureOffset, uint64_t outputOffset, int64_t bs,
        int64_t h, int64_t topK, int64_t nvS, uint64_t rowBytes,
        bool reduceHidden,
        int64_t magic, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBuffers();
    __aicore__ inline void SetFailure(uint32_t status, uint32_t step,
        uint32_t peer, uint32_t lane, uint32_t cqStatus = 0U,
        uint64_t expected = 0U, uint64_t observed = 0U);
    __aicore__ inline bool CheckPoison();
    __aicore__ inline bool ValidateDestinations();
    __aicore__ inline void LoadSelectionChunk(uint64_t chunkStart,
        uint32_t chunkElements);
    __aicore__ inline uint32_t SelectPeer(uint32_t peer,
        uint32_t chunkElements);
    __aicore__ inline bool InitLaneStates();
    __aicore__ inline bool ResolveRemoteFields(uint32_t targetRank,
        uint32_t lane, uint64_t remoteBaseOffset,
        __ubuf__ MoonEpCombineV2RemoteFields *fields);
    __aicore__ inline uint32_t PollCqOnce(
        MoonEpCombineV2LaneState &state);
    __aicore__ inline bool WaitAdmission(uint32_t step);
    __aicore__ inline bool WaitFinalCqs();
    __aicore__ inline bool AppendDescriptor(uint32_t sourceSlot,
        uint32_t targetSlot, uint32_t peer, uint32_t step);
    __aicore__ inline bool BuildPayloadWqes(uint32_t peer);
    __aicore__ inline bool AppendControlWqe(LocalTensor<uint8_t> issue,
        uint32_t outputIndex, MoonEpCombineV2LaneState &state,
        uint32_t targetRank, uint64_t remoteOffset,
        __gm__ uint64_t *localSource, uint32_t flag);
    __aicore__ inline void CopyIssueToSq(LocalTensor<uint8_t> issue,
        MoonEpCombineV2LaneState &state, uint32_t count);
    __aicore__ inline bool SubmitPair(uint32_t peer, uint32_t step,
        bool finalBatch);
    __aicore__ inline bool SendRemoteStep(uint32_t peer, uint32_t step);
    __aicore__ inline bool SendSelfStep(uint32_t peer);
    __aicore__ inline bool WaitInboundDone();
    __aicore__ inline void InitReduceBuffers();
    __aicore__ inline bool ReduceHidden();
    __aicore__ inline uint64_t LoadToken(__gm__ uint64_t *token);
    __aicore__ inline void PublishFailureAndConverge();
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    __aicore__ inline void RecordProfilePoint(uint32_t index);
    __aicore__ inline void WriteProfile();
#endif

    bool valid_{false};
    TPipe *pipe_{nullptr};
    __gm__ TileXR::CommArgs *args_{nullptr};
    __gm__ TileXR::TileXRUDMARegistry *registry_{nullptr};
    __gm__ uint8_t *workspace_{nullptr};
    __gm__ int32_t *dstGlobalAddr_{nullptr};
    __gm__ uint8_t *scratch_{nullptr};
    __gm__ uint8_t *doneBase_{nullptr};
    __gm__ uint8_t *grantBase_{nullptr};
    __gm__ uint8_t *controlSourceBase_{nullptr};
    __gm__ uint8_t *failureBase_{nullptr};
    uint64_t scratchOffset_{0U};
    uint64_t profileOffset_{0U};
    uint64_t doneOffset_{0U};
    uint64_t grantOffset_{0U};
    uint64_t failureOffset_{0U};
    uint64_t outputOffset_{0U};
    uint64_t slots_{0U};
    uint64_t rowBytes_{0U};
    int64_t bs_{0};
    int64_t h_{0};
    int64_t topK_{0};
    uint64_t magic_{0U};
    uint64_t operationStartCycles_{0U};
    uint32_t epoch_{0U};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    uint32_t core_{0U};
    uint32_t activeCoreCount_{0U};
    uint32_t stepCount_{0U};
    uint32_t sourcesPerCore_{0U};
    uint32_t descriptorCount_{0U};
    uint32_t issuedRows_{0U};
    uint32_t selectedPeerRows_{0U};
    uint32_t currentChunkElements_{0U};
    uint32_t failureStatus_{TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS};
    uint32_t failureStep_{UINT32_MAX};
    uint32_t failurePeer_{UINT32_MAX};
    uint32_t failureLane_{UINT32_MAX};
    uint32_t failureCqStatus_{0U};
    uint64_t failureExpected_{0U};
    uint64_t failureObserved_{0U};
    bool selectionBuffersUsed_{false};
    bool allCoresSucceeded_{false};
    bool reduceHidden_{false};
    uint32_t convergenceRound_{0U};
    MoonEpCombineV2LaneState lane_[
        TileXRMoonEp::kMoonEpCombineV2LaneCount] {};

    TBuf<QuePosition::VECCALC> dstBuf_;
    TBuf<QuePosition::VECCALC> dstRankBuf_;
    TBuf<QuePosition::VECCALC> slotIndexBuf_;
    TBuf<QuePosition::VECCALC> selectedIndexBuf_;
    TBuf<QuePosition::VECCALC> compareMaskBuf_;
    TBuf<QuePosition::VECCALC> relayBuf_;
    TBuf<QuePosition::VECCALC> descriptorBuf_;
    TBuf<QuePosition::VECCALC> sixPortIssueBuf_;
    TBuf<QuePosition::VECCALC> twoPortIssueBuf_;
    TBuf<QuePosition::VECCALC> buildContextBuf_;
    TQue<QuePosition::VECIN, 1> reduceInputQueue_;
    TQue<QuePosition::VECOUT, 1> reduceOutputQueue_;
    TBuf<QuePosition::VECCALC> reduceRowBuf_;
    TBuf<QuePosition::VECCALC> reduceAccumulatorBuf_;
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    int64_t profileTimePoint_[
        TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCapacity];
#endif
};

__aicore__ inline void MoonEpCombineV2::Init(
    GM_ADDR commArgsGM, GM_ADDR registeredWorkspaceGM, GM_ADDR dstLocalGM,
    uint64_t profileOffset, uint64_t scratchEpoch0Offset,
    uint64_t scratchEpoch1Offset, uint64_t doneOffset,
    uint64_t grantOffset, uint64_t controlSourceOffset,
    uint64_t failureOffset, uint64_t outputOffset, int64_t bs,
    int64_t h, int64_t topK, int64_t nvS, uint64_t rowBytes,
    bool reduceHidden,
    int64_t magic, TPipe *pipe)
{
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    const uint64_t profileStartCycles =
        static_cast<uint64_t>(GetSystemCycle());
    for (uint32_t index = 0U;
        index < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCapacity;
        ++index) {
        profileTimePoint_[index] = 0;
    }
    profileTimePoint_[TileXRMoonEp::MOONEP_COMBINE_V2_TIME_INIT_BEGIN] =
        static_cast<int64_t>(profileStartCycles);
#endif
    pipe_ = pipe;
    args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
    workspace_ = reinterpret_cast<__gm__ uint8_t *>(registeredWorkspaceGM);
    dstGlobalAddr_ = reinterpret_cast<__gm__ int32_t *>(dstLocalGM);
    operationStartCycles_ = kEnableSafetyChecks ?
        static_cast<uint64_t>(GetSystemCycle()) : 0U;
    if (kEnableSafetyChecks &&
        (pipe_ == nullptr || args_ == nullptr || workspace_ == nullptr ||
        dstGlobalAddr_ == nullptr || !TileXR::UDMARegistryEnabled(args_) ||
        (args_->extraFlag &
            TileXR::ExtraFlag::UDMA_SHARED_QP) == 0U ||
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
        GetBlockNum() < TileXRMoonEp::kMoonEpCombineV2CoreCount ||
        TileXR::GetUDMAInfo(args_)->qpNum !=
            TileXRMoonEp::kMoonEpCombineV2QpCount)) {
        return;
    }
    magic_ = static_cast<uint64_t>(magic);
    if (kEnableSafetyChecks &&
        !TileXRMoonEp::MoonEpCombineV2MagicValid(magic_)) {
        return;
    }
    rank_ = static_cast<uint32_t>(args_->rank);
    rankSize_ = static_cast<uint32_t>(args_->rankSize);
    core_ = static_cast<uint32_t>(GetBlockIdx());
    activeCoreCount_ =
        TileXRMoonEp::MoonEpCombineV2ActiveCoreCount(rankSize_);
    stepCount_ = TileXRMoonEp::MoonEpCombineV2StepCount(rankSize_);
    sourcesPerCore_ = rankSize_ / activeCoreCount_;
    epoch_ = TileXRMoonEp::MoonEpCombineV2Epoch(magic_);
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
    outputOffset_ = outputOffset;
    scratch_ = workspace_ + scratchOffset_;
    doneBase_ = workspace_ + doneOffset;
    grantBase_ = workspace_ + grantOffset;
    controlSourceBase_ = workspace_ + controlSourceOffset +
        static_cast<uint64_t>(core_) *
            TileXRMoonEp::kMoonEpCombineV2LaneCount *
            TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes;
    failureBase_ = workspace_ + failureOffset;
    valid_ = true;
    const bool laneStatesReady = InitLaneStates();
    if (kEnableSafetyChecks && !laneStatesReady) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            UINT32_MAX, UINT32_MAX, UINT32_MAX);
    }
    (void)laneStatesReady;
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    RecordProfilePoint(TileXRMoonEp::MOONEP_COMBINE_V2_TIME_INIT_END);
#endif
}

__aicore__ inline void MoonEpCombineV2::InitBuffers()
{
    constexpr uint32_t chunk =
        TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows;
    constexpr uint32_t dstBytes = chunk * sizeof(int32_t);
    constexpr uint32_t indexBytes = chunk * sizeof(int16_t);
    constexpr uint32_t maskBytes = (chunk / 8U + kUbAlignBytes - 1U) /
        kUbAlignBytes * kUbAlignBytes;
    constexpr uint32_t totalBytes = 2U * dstBytes + 2U * indexBytes +
        maskBytes + kRelayBytes + kDescriptorBytes + kSixPortIssueBytes +
        kTwoPortIssueBytes + kBuildContextBytes;
    static_assert(totalBytes <= kFullUbBytes,
        "Combine V2 send buffers exceed 216 KiB UB");
    pipe_->Reset();
    pipe_->InitBuffer(dstBuf_, dstBytes);
    pipe_->InitBuffer(dstRankBuf_, dstBytes);
    pipe_->InitBuffer(slotIndexBuf_, indexBytes);
    pipe_->InitBuffer(selectedIndexBuf_, indexBytes);
    pipe_->InitBuffer(compareMaskBuf_, maskBytes);
    pipe_->InitBuffer(relayBuf_, kRelayBytes);
    pipe_->InitBuffer(descriptorBuf_, kDescriptorBytes);
    pipe_->InitBuffer(sixPortIssueBuf_, kSixPortIssueBytes);
    pipe_->InitBuffer(twoPortIssueBuf_, kTwoPortIssueBytes);
    pipe_->InitBuffer(buildContextBuf_, kBuildContextBytes);
    LocalTensor<int16_t> slotIndex = slotIndexBuf_.Get<int16_t>();
    CreateVecIndex(slotIndex, static_cast<int16_t>(0), chunk);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MoonEpCombineV2::SetFailure(
    uint32_t status, uint32_t step, uint32_t peer, uint32_t lane,
    uint32_t cqStatus, uint64_t expected, uint64_t observed)
{
    if (failureStatus_ != TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS) {
        return;
    }
    failureStatus_ = status;
    failureStep_ = step;
    failurePeer_ = peer;
    failureLane_ = lane;
    failureCqStatus_ = cqStatus;
    failureExpected_ = expected;
    failureObserved_ = observed;
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
    if (kEnableSafetyChecks &&
        (info == nullptr || info->sqPtr == 0U || info->scqPtr == 0U ||
        info->memPtr == 0U)) {
        return false;
    }
    const uint32_t firstPeer = TileXRMoonEp::MoonEpCombineV2Peer(
        rank_, 0U, core_, rankSize_);
    for (uint32_t lane = 0U;
        lane < TileXRMoonEp::kMoonEpCombineV2LaneCount; ++lane) {
        MoonEpCombineV2LaneState &state = lane_[lane];
        state.qp = TileXRMoonEp::MoonEpCombineV2Qp(core_, lane);
        state.sq = TileXR::UDMAGetWQCtx(info, firstPeer, state.qp);
        state.cq = TileXR::UDMAGetSCQCtx(info, firstPeer, state.qp);
        if (kEnableSafetyChecks &&
            (state.sq == nullptr || state.cq == nullptr ||
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
            state.cq->dbAddr == 0U)) {
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
        if (kEnableSafetyChecks && state.head != state.tail) {
            return false;
        }
    }
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
    GlobalTensor<int32_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(dstGlobalAddr_, slots_);
    LocalTensor<int32_t> dst = dstBuf_.Get<int32_t>();
    const DataCopyExtParams params {
        1U, rowCount * static_cast<uint32_t>(sizeof(int32_t)),
        0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> pad {false, 0U, 0U, 0U};
    DataCopyPad(dst, dstGlobal[firstSlot], params, pad);
    SyncFunc<HardEvent::MTE2_S>();
    selectionBuffersUsed_ = true;
    for (uint32_t localSlot = 0U; localSlot < rowCount; ++localSlot) {
        const int32_t encoded = dst.GetValue(localSlot);
        if (!TileXRMoonEp::MoonEpCombineV2DestinationValid(
                encoded, slots_, rankSize_)) {
            const uint32_t peer = encoded < 0 ? UINT32_MAX :
                static_cast<uint32_t>(
                    static_cast<uint64_t>(encoded) / slots_);
            SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_BAD_DESTINATION,
                UINT32_MAX, peer, UINT32_MAX, firstSlot + localSlot,
                static_cast<uint64_t>(rankSize_) * slots_,
                static_cast<uint32_t>(encoded));
            return false;
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
    LocalTensor<int32_t> dst = dstBuf_.Get<int32_t>();
    const DataCopyExtParams params {
        1U, chunkElements * static_cast<uint32_t>(sizeof(int32_t)),
        0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> pad {false, 0U, 0U, 0U};
    DataCopyPad(dst, dstGlobal[chunkStart], params, pad);
    SyncFunc<HardEvent::MTE2_S>();
    if (selectionBuffersUsed_) {
        SyncFunc<HardEvent::S_V>();
    }
    LocalTensor<int32_t> dstRank = dstRankBuf_.Get<int32_t>();
    for (uint32_t index = 0U; index < chunkElements; ++index) {
        const int32_t encoded = dst.GetValue(index);
        dstRank.SetValue(index, encoded < 0 ? -1 : static_cast<int32_t>(
            static_cast<uint64_t>(encoded) / slots_));
    }
    SyncFunc<HardEvent::S_V>();
    PipeBarrier<PIPE_V>();
    currentChunkElements_ = chunkElements;
    selectionBuffersUsed_ = true;
}

__aicore__ inline uint32_t MoonEpCombineV2::SelectPeer(
    uint32_t peer, uint32_t chunkElements)
{
    LocalTensor<uint8_t> mask = compareMaskBuf_.Get<uint8_t>();
    Compares(mask, dstRankBuf_.Get<int32_t>(), static_cast<int32_t>(peer),
        CMPMODE::EQ, chunkElements);
    PipeBarrier<PIPE_V>();
    uint64_t selected = 0U;
    GatherMask(selectedIndexBuf_.Get<int16_t>(),
        slotIndexBuf_.Get<int16_t>(), mask.ReinterpretCast<uint16_t>(), true,
        chunkElements, {1U, 1U, 0U, 0U}, selected);
    SyncFunc<HardEvent::V_S>();
    return static_cast<uint32_t>(selected);
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
    if (kEnableSafetyChecks &&
        result == TileXRMoonEp::MOONEP_COMBINE_V2_SINGLE_CQE_ERROR) {
        return detail == 0U ? kPollInvalidState : detail;
    }
    if (kEnableSafetyChecks &&
        result == TileXRMoonEp::MOONEP_COMBINE_V2_SINGLE_CQE_INVALID_STATE) {
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

__aicore__ inline bool MoonEpCombineV2::WaitAdmission(uint32_t step)
{
    if (step == 0U) {
        return true;
    }
    bool cqReady[2] = {false, false};
    bool grantReady[2] = {false, false};
    uint64_t observed[2] = {0U, 0U};
    const uint64_t expected =
        TileXRMoonEp::MoonEpCombineV2Token(magic_, step);
    uint32_t cursor = 0U;
    while (!(cqReady[0] && cqReady[1] && grantReady[0] &&
        grantReady[1])) {
        for (uint32_t offset = 0U; offset < 4U; ++offset) {
            const uint32_t condition = (cursor + offset) & 3U;
            if (condition < 2U && !cqReady[condition]) {
                const uint32_t status = PollCqOnce(lane_[condition]);
                if (status == 0U) {
                    cqReady[condition] = true;
                } else if (kEnableSafetyChecks &&
                    status != kPollNoCompletion) {
                    SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
                        step, UINT32_MAX, condition, status);
                    return false;
                }
            } else if (condition >= 2U) {
                const uint32_t lane = condition - 2U;
                if (grantReady[lane]) {
                    continue;
                }
                const uint64_t index =
                    TileXRMoonEp::MoonEpCombineV2GrantIndex(
                        epoch_, core_, lane, step);
                __gm__ uint64_t *token = reinterpret_cast<__gm__ uint64_t *>(
                    grantBase_ + index *
                        TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                        TileXRMoonEp::kMoonEpCombineV2GrantReceiveOffsetBytes);
                observed[lane] = LoadToken(token);
                grantReady[lane] = observed[lane] == expected;
            }
        }
        cursor = (cursor + 1U) & 3U;
        if (kEnableSafetyChecks && TimedOut(operationStartCycles_)) {
            for (uint32_t lane = 0U; lane < 2U; ++lane) {
                if (!cqReady[lane]) {
                    SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_TIMEOUT,
                        step, UINT32_MAX, lane);
                    return false;
                }
                if (!grantReady[lane]) {
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

__aicore__ inline bool MoonEpCombineV2::BuildPayloadWqes(uint32_t peer)
{
    LocalTensor<uint8_t> contextTensor = buildContextBuf_.Get<uint8_t>();
    __ubuf__ MoonEpCombineV2BuildContext *context = reinterpret_cast<__ubuf__
        MoonEpCombineV2BuildContext *>(contextTensor.GetPhyAddr());
    context->localRowBase = reinterpret_cast<uint64_t>(workspace_);
    context->rowBytes = rowBytes_;
    context->logicalCount = descriptorCount_;
    context->sequencePhase = issuedRows_ & 3U;
    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        context->head[lane] = lane_[lane].head;
        context->laneCount[lane] = 0U;
        const bool remoteFieldsReady = ResolveRemoteFields(
            peer, lane, scratchOffset_, &context->remote[lane]);
        if (kEnableSafetyChecks && !remoteFieldsReady) {
            return false;
        }
        (void)remoteFieldsReady;
    }
    context->laneCount[1] = TileXRMoonEp::MoonEpCombineV2Qp1TokenCount(
        descriptorCount_, context->sequencePhase);
    context->laneCount[0] = descriptorCount_ - context->laneCount[1];
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    PipeBarrier<PIPE_ALL>();
    Simt::VF_CALL<MoonEpCombineV2BuildPayloadVf>(
        Simt::Dim3{kWqeBuildThreads, 1U, 1U},
        reinterpret_cast<__ubuf__ uint8_t *>(
            sixPortIssueBuf_.Get<uint8_t>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint8_t *>(
            twoPortIssueBuf_.Get<uint8_t>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ const MoonEpCombineV2Descriptor *>(
            descriptorBuf_.Get<uint8_t>().GetPhyAddr()), context);
    PipeBarrier<PIPE_ALL>();
#endif
    return true;
}

__aicore__ inline bool MoonEpCombineV2::AppendDescriptor(
    uint32_t sourceSlot, uint32_t targetSlot, uint32_t peer, uint32_t step)
{
    if (descriptorCount_ ==
        TileXRMoonEp::kMoonEpCombineV2LogicalBatchRows) {
        const bool batchSubmitted = SubmitPair(peer, step, false);
        if (kEnableSafetyChecks && !batchSubmitted) {
            return false;
        }
        (void)batchSubmitted;
    }
    __ubuf__ MoonEpCombineV2Descriptor *descriptors = reinterpret_cast<__ubuf__
        MoonEpCombineV2Descriptor *>(
            descriptorBuf_.Get<uint8_t>().GetPhyAddr());
    descriptors[descriptorCount_].sourceSlot = sourceSlot;
    descriptors[descriptorCount_].targetSlot = targetSlot;
    ++descriptorCount_;
    ++selectedPeerRows_;
    if (kEnableSafetyChecks && selectedPeerRows_ >
        TileXRMoonEp::kMoonEpCombineV2MaxOutstanding) {
        SetFailure(
            TileXRMoonEp::MOONEP_COMBINE_V2_OUTSTANDING_LIMIT,
            step, peer, UINT32_MAX, selectedPeerRows_);
        return false;
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::AppendControlWqe(
    LocalTensor<uint8_t> issue, uint32_t outputIndex,
    MoonEpCombineV2LaneState &state, uint32_t targetRank,
    uint64_t remoteOffset, __gm__ uint64_t *localSource, uint32_t flag)
{
    if (kEnableSafetyChecks &&
        !TileXR::UDMARegisteredRangeValid(registry_, targetRank,
            remoteOffset, sizeof(uint64_t))) {
        return false;
    }
    __gm__ TileXR::UDMAMemInfo *mem = TileXR::UDMAGetRemoteMemInfo(
        TileXR::GetUDMAInfo(args_), targetRank, state.qp);
    const uint64_t registeredBase = reinterpret_cast<uint64_t>(
        registry_->regions[targetRank].base);
    if (kEnableSafetyChecks &&
        (localSource == nullptr || mem->eidAddr == 0U ||
        !mem->tokenValueValid || mem->addr == 0U ||
        mem->addr != registeredBase || remoteOffset > mem->len ||
        sizeof(uint64_t) > static_cast<uint64_t>(mem->len) - remoteOffset)) {
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
    uint32_t peer, uint32_t step, bool finalBatch)
{
    const bool payloadReady = BuildPayloadWqes(peer);
    if (kEnableSafetyChecks && !payloadReady) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            step, peer, UINT32_MAX);
        return false;
    }
    (void)payloadReady;
    const uint32_t phase = issuedRows_ & 3U;
    const TileXRMoonEp::MoonEpCombineV2LaneCounts batchCounts =
        TileXRMoonEp::MoonEpCombineV2BatchLaneCounts(
            descriptorCount_, phase, step, stepCount_, false);
    uint32_t count[2] = {batchCounts.sixPort, batchCounts.twoPort};
    if (finalBatch) {
        const uint32_t successor =
            TileXRMoonEp::MoonEpCombineV2Successor(
                rank_, core_, rankSize_);
        for (uint32_t lane = 0U; lane < 2U; ++lane) {
            LocalTensor<uint8_t> issue = lane == 0U ?
                sixPortIssueBuf_.Get<uint8_t>() :
                twoPortIssueBuf_.Get<uint8_t>();
            __gm__ uint64_t *doneSource = reinterpret_cast<__gm__ uint64_t *>(
                controlSourceBase_ + static_cast<uint64_t>(lane) *
                    TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
            if (step + 1U < stepCount_) {
                const uint64_t grantIndex =
                    TileXRMoonEp::MoonEpCombineV2GrantIndex(
                        epoch_, core_, lane, step + 1U);
                __gm__ uint64_t *grantSource = reinterpret_cast<__gm__ uint64_t *>(
                    grantBase_ + grantIndex *
                        TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                        TileXRMoonEp::kMoonEpCombineV2GrantSourceOffsetBytes);
                *grantSource = TileXRMoonEp::MoonEpCombineV2Token(
                    magic_, step + 1U);
                TileXR::UDMACleanCacheLines(
                    reinterpret_cast<__gm__ uint8_t *>(grantSource),
                    TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
                const bool grantAppended = AppendControlWqe(
                    issue, count[lane], lane_[lane], successor,
                    grantOffset_ + grantIndex *
                        TileXRMoonEp::kMoonEpCombineV2GrantSlotBytes +
                        TileXRMoonEp::kMoonEpCombineV2GrantReceiveOffsetBytes,
                    grantSource,
                    TileXR::TILEXR_UDMA_SQE_FLAG_STRONG_ORDER);
                if (kEnableSafetyChecks && !grantAppended) {
                    SetFailure(
                        TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                        step, successor, lane);
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
                issue, count[lane], lane_[lane], peer,
                doneOffset_ + doneIndex *
                        TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes,
                doneSource,
                TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
            if (kEnableSafetyChecks && !doneAppended) {
                SetFailure(
                    TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
                    step, peer, lane);
                return false;
            }
            (void)doneAppended;
            ++count[lane];
        }
    }

    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        const uint32_t outstanding = lane_[lane].head - lane_[lane].tail;
        if (kEnableSafetyChecks &&
            (count[lane] == 0U ||
            static_cast<uint64_t>(outstanding) + count[lane] >=
                TileXRMoonEp::kMoonEpCombineV2MaxOutstanding)) {
            SetFailure(
                TileXRMoonEp::MOONEP_COMBINE_V2_OUTSTANDING_LIMIT,
                step, peer, lane, outstanding + count[lane]);
            return false;
        }
    }

    SyncFunc<HardEvent::S_MTE3>();
    CopyIssueToSq(sixPortIssueBuf_.Get<uint8_t>(), lane_[0], count[0]);
    CopyIssueToSq(twoPortIssueBuf_.Get<uint8_t>(), lane_[1], count[1]);
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
    st_dev(lane_[0].head, reinterpret_cast<__gm__ uint32_t *>(
        lane_[0].sq->dbAddr), 0);
    st_dev(lane_[1].head, reinterpret_cast<__gm__ uint32_t *>(
        lane_[1].sq->dbAddr), 0);
    if (finalBatch) {
        for (uint32_t lane = 0U; lane < 2U; ++lane) {
            lane_[lane].submittedHead = lane_[lane].head;
            lane_[lane].cqTarget =
                TileXRMoonEp::MoonEpCombineV2NextCqTarget(
                    lane_[lane].cqTail, true);
        }
    }
    issuedRows_ += descriptorCount_;
    descriptorCount_ = 0U;
    return true;
}

__aicore__ inline bool MoonEpCombineV2::SendRemoteStep(
    uint32_t peer, uint32_t step)
{
    descriptorCount_ = 0U;
    issuedRows_ = 0U;
    selectedPeerRows_ = 0U;
    LocalTensor<int16_t> selected = selectedIndexBuf_.Get<int16_t>();
    LocalTensor<int32_t> dst = dstBuf_.Get<int32_t>();
    for (uint64_t chunkStart = 0U; chunkStart < slots_;
        chunkStart += TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows) {
        const uint32_t chunkElements = static_cast<uint32_t>(
            slots_ - chunkStart <
                TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows ?
            slots_ - chunkStart :
            TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows);
        LoadSelectionChunk(chunkStart, chunkElements);
        const uint32_t selectedCount = SelectPeer(peer, chunkElements);
        for (uint32_t index = 0U; index < selectedCount; ++index) {
            const uint32_t sourceInChunk = static_cast<uint16_t>(
                selected.GetValue(index));
            if (kEnableSafetyChecks && sourceInChunk >= chunkElements) {
                SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_BAD_DESTINATION,
                    step, peer, UINT32_MAX);
                return false;
            }
            const uint32_t targetSlot = static_cast<uint32_t>(
                static_cast<uint64_t>(dst.GetValue(sourceInChunk)) % slots_);
            const bool descriptorAppended = AppendDescriptor(
                static_cast<uint32_t>(chunkStart) + sourceInChunk,
                targetSlot, peer, step);
            if (kEnableSafetyChecks && !descriptorAppended) {
                return false;
            }
            (void)descriptorAppended;
        }
    }
    return SubmitPair(peer, step, true);
}

__aicore__ inline bool MoonEpCombineV2::SendSelfStep(uint32_t peer)
{
    LocalTensor<int16_t> selected = selectedIndexBuf_.Get<int16_t>();
    LocalTensor<int32_t> dst = dstBuf_.Get<int32_t>();
    LocalTensor<uint8_t> relay = relayBuf_.Get<uint8_t>();
    for (uint64_t chunkStart = 0U; chunkStart < slots_;
        chunkStart += TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows) {
        const uint32_t chunkElements = static_cast<uint32_t>(
            slots_ - chunkStart <
                TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows ?
            slots_ - chunkStart :
            TileXRMoonEp::kMoonEpCombineV2SelectionChunkRows);
        LoadSelectionChunk(chunkStart, chunkElements);
        const uint32_t selectedCount = SelectPeer(peer, chunkElements);
        for (uint32_t index = 0U; index < selectedCount; ++index) {
            const uint32_t sourceInChunk = static_cast<uint16_t>(
                selected.GetValue(index));
            const uint32_t targetSlot = static_cast<uint32_t>(
                static_cast<uint64_t>(dst.GetValue(sourceInChunk)) % slots_);
            CopyBytesGmToGm(scratch_ +
                    static_cast<uint64_t>(targetSlot) * rowBytes_,
                workspace_ + (chunkStart + sourceInChunk) * rowBytes_,
                static_cast<uint32_t>(rowBytes_), relay);
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::WaitFinalCqs()
{
    bool ready[2] = {false, false};
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
            } else if (kEnableSafetyChecks &&
                status != kPollNoCompletion) {
                SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_CQ_ERROR,
                    stepCount_ - 1U, UINT32_MAX, lane, status);
                return false;
            }
        }
        cursor ^= 1U;
        if (kEnableSafetyChecks && TimedOut(operationStartCycles_)) {
            for (uint32_t lane = 0U; lane < 2U; ++lane) {
                if (!ready[lane]) {
                    SetFailure(
                        TileXRMoonEp::MOONEP_COMBINE_V2_CQ_TIMEOUT,
                        stepCount_ - 1U, UINT32_MAX, lane);
                    return false;
                }
            }
        }
    }
    return true;
}

__aicore__ inline bool MoonEpCombineV2::WaitInboundDone()
{
    bool ready[TileXRMoonEp::kMoonEpCombineV2MaxSourcesPerCore]
        [TileXRMoonEp::kMoonEpCombineV2LaneCount] = {};
    uint64_t observed[TileXRMoonEp::kMoonEpCombineV2MaxSourcesPerCore]
        [TileXRMoonEp::kMoonEpCombineV2LaneCount] = {};
    uint32_t remaining = 0U;
    for (uint32_t sourceIndex = 0U;
        sourceIndex < sourcesPerCore_; ++sourceIndex) {
        const uint32_t source = TileXRMoonEp::MoonEpCombineV2SourceForCore(
            core_, sourceIndex, rankSize_);
        for (uint32_t lane = 0U;
            lane < TileXRMoonEp::kMoonEpCombineV2LaneCount; ++lane) {
            ready[sourceIndex][lane] = source == rank_;
            if (!ready[sourceIndex][lane]) {
                ++remaining;
            }
        }
    }
    const uint32_t conditionCount = sourcesPerCore_ *
        TileXRMoonEp::kMoonEpCombineV2LaneCount;
    uint32_t cursor = 0U;
    while (remaining != 0U) {
        for (uint32_t offset = 0U; offset < conditionCount; ++offset) {
            const uint32_t condition =
                (cursor + offset) & (conditionCount - 1U);
            const uint32_t sourceIndex = condition /
                TileXRMoonEp::kMoonEpCombineV2LaneCount;
            const uint32_t lane = condition &
                (TileXRMoonEp::kMoonEpCombineV2LaneCount - 1U);
            if (ready[sourceIndex][lane]) {
                continue;
            }
            const uint32_t source =
                TileXRMoonEp::MoonEpCombineV2SourceForCore(
                    core_, sourceIndex, rankSize_);
            const uint32_t step =
                TileXRMoonEp::MoonEpCombineV2ReceiveStep(
                    rank_, source, rankSize_);
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
        cursor = (cursor + 1U) & (conditionCount - 1U);
        if (kEnableSafetyChecks && TimedOut(operationStartCycles_)) {
            for (uint32_t condition = 0U;
                condition < conditionCount; ++condition) {
                const uint32_t sourceIndex = condition /
                    TileXRMoonEp::kMoonEpCombineV2LaneCount;
                const uint32_t lane = condition &
                    (TileXRMoonEp::kMoonEpCombineV2LaneCount - 1U);
                if (ready[sourceIndex][lane]) {
                    continue;
                }
                const uint32_t source =
                    TileXRMoonEp::MoonEpCombineV2SourceForCore(
                        core_, sourceIndex, rankSize_);
                const uint32_t step =
                    TileXRMoonEp::MoonEpCombineV2ReceiveStep(
                        rank_, source, rankSize_);
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

__aicore__ inline void MoonEpCombineV2::InitReduceBuffers()
{
    constexpr uint32_t inputBytes = kReduceRouteBatch *
        kReduceTileElements * sizeof(bfloat16_t);
    constexpr uint32_t outputBytes =
        kReduceTileElements * sizeof(bfloat16_t);
    constexpr uint32_t floatBytes = kReduceTileElements * sizeof(float);
    static_assert(inputBytes + outputBytes + 2U * floatBytes <= kFullUbBytes,
        "Combine V2 reduction buffers exceed UB");
    pipe_->Reset();
    pipe_->InitBuffer(reduceInputQueue_, 1U, inputBytes);
    pipe_->InitBuffer(reduceOutputQueue_, 1U, outputBytes);
    pipe_->InitBuffer(reduceRowBuf_, floatBytes);
    pipe_->InitBuffer(reduceAccumulatorBuf_, floatBytes);
}

__aicore__ inline bool MoonEpCombineV2::ReduceHidden()
{
    if (!reduceHidden_) {
        return true;
    }
    InitReduceBuffers();
    const int64_t tokenBegin = bs_ * core_ / activeCoreCount_;
    const int64_t tokenEnd = bs_ * (core_ + 1U) / activeCoreCount_;
    LocalTensor<float> row = reduceRowBuf_.Get<float>();
    LocalTensor<float> accumulator = reduceAccumulatorBuf_.Get<float>();

    for (int64_t token = tokenBegin; token < tokenEnd; ++token) {
        for (int64_t hiddenOffset = 0; hiddenOffset < h_;
             hiddenOffset += kReduceTileElements) {
            const int64_t tileElements = h_ - hiddenOffset <
                    static_cast<int64_t>(kReduceTileElements) ?
                h_ - hiddenOffset : static_cast<int64_t>(kReduceTileElements);
            const uint32_t inputStrideElements =
                TileXRMoonEp::MoonEpCombineV2ReduceInputStrideElements(
                    static_cast<uint32_t>(tileElements));
            Duplicate(accumulator, 0.0f,
                static_cast<int32_t>(tileElements));
            PipeBarrier<PIPE_V>();
            for (int64_t topkBegin = 0; topkBegin < topK_;
                 topkBegin += kReduceRouteBatch) {
                const int64_t batchRoutes = topK_ - topkBegin <
                        static_cast<int64_t>(kReduceRouteBatch) ?
                    topK_ - topkBegin : static_cast<int64_t>(kReduceRouteBatch);
                LocalTensor<bfloat16_t> inputRows =
                    reduceInputQueue_.AllocTensor<bfloat16_t>();
                for (int64_t batchRoute = 0; batchRoute < batchRoutes;
                     ++batchRoute) {
                    const int64_t route = token * topK_ + topkBegin + batchRoute;
                    GlobalTensor<bfloat16_t> input;
                    input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                        scratch_ + static_cast<uint64_t>(route) * rowBytes_) +
                            hiddenOffset, tileElements);
                    const DataCopyExtParams copyIn {
                        1U, static_cast<uint32_t>(
                            tileElements * sizeof(bfloat16_t)), 0U, 0U, 0U};
                    const DataCopyPadExtParams<bfloat16_t> pad {
                        false, 0U, 0U, 0U};
                    DataCopyPad(inputRows[batchRoute * inputStrideElements],
                        input, copyIn, pad);
                }
                reduceInputQueue_.EnQue(inputRows);
                inputRows = reduceInputQueue_.DeQue<bfloat16_t>();

                for (int64_t batchRoute = 0; batchRoute < batchRoutes;
                     ++batchRoute) {
                    Cast(row, inputRows[batchRoute * inputStrideElements],
                        RoundMode::CAST_NONE,
                        static_cast<int32_t>(tileElements));
                    PipeBarrier<PIPE_V>();
                    Add(accumulator, accumulator, row,
                        static_cast<int32_t>(tileElements));
                    PipeBarrier<PIPE_V>();
                }
                reduceInputQueue_.FreeTensor(inputRows);
            }

            LocalTensor<bfloat16_t> outputRow =
                reduceOutputQueue_.AllocTensor<bfloat16_t>();
            Cast(outputRow, accumulator, RoundMode::CAST_RINT,
                static_cast<int32_t>(tileElements));
            reduceOutputQueue_.EnQue(outputRow);
            outputRow = reduceOutputQueue_.DeQue<bfloat16_t>();
            GlobalTensor<bfloat16_t> output;
            output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                workspace_ + outputOffset_) + token * h_ + hiddenOffset,
                tileElements);
            const DataCopyExtParams copyOut {
                1U, static_cast<uint32_t>(
                    tileElements * sizeof(bfloat16_t)), 0U, 0U, 0U};
            DataCopyPad(output, outputRow, copyOut);
            reduceOutputQueue_.FreeTensor(outputRow);
        }
    }
    PipeBarrier<PIPE_ALL>();
    return true;
}

__aicore__ inline void MoonEpCombineV2::PublishFailureAndConverge()
{
    const uint32_t convergenceMarker =
        TileXRMoonEp::kMoonEpCombineV2FailureMarker |
        ((convergenceRound_ + 1U) & 1U);
    const uint64_t index = TileXRMoonEp::MoonEpCombineV2FailureIndex(
        epoch_, core_);
    __gm__ TileXRMoonEp::MoonEpCombineV2FailureRecord *record = reinterpret_cast<__gm__
        TileXRMoonEp::MoonEpCombineV2FailureRecord *>(failureBase_ +
            index * TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
    record->magic = magic_;
    record->status = failureStatus_;
    record->rank = rank_;
    record->core = core_;
    record->step = failureStep_;
    record->peer = failurePeer_;
    record->lane = failureLane_;
    record->qp = failureLane_ < 2U ?
        TileXRMoonEp::MoonEpCombineV2Qp(core_, failureLane_) : UINT32_MAX;
    record->cqStatus = failureCqStatus_;
    record->expected = failureExpected_;
    record->observed = failureObserved_;
    record->poison = failureStatus_ ==
        TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS ? 0U : 1U;
    record->marker = convergenceMarker;
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(record),
        sizeof(*record));

    allCoresSucceeded_ = true;
    bool ready[TileXRMoonEp::kMoonEpCombineV2CoreCount] = {};
    uint32_t remaining = activeCoreCount_;
    uint32_t cursor = 0U;
    while (remaining != 0U) {
        for (uint32_t offset = 0U; offset < activeCoreCount_; ++offset) {
            const uint32_t core =
                (cursor + offset) % activeCoreCount_;
            if (ready[core]) {
                continue;
            }
            const uint64_t peerIndex =
                TileXRMoonEp::MoonEpCombineV2FailureIndex(epoch_, core);
            __gm__ TileXRMoonEp::MoonEpCombineV2FailureRecord *peerRecord = reinterpret_cast<__gm__
                TileXRMoonEp::MoonEpCombineV2FailureRecord *>(failureBase_ +
                    peerIndex *
                        TileXRMoonEp::kMoonEpCombineV2TokenStrideBytes);
            TileXR::UDMACleanCacheLines(
                reinterpret_cast<__gm__ uint8_t *>(peerRecord),
                sizeof(*peerRecord));
            if (peerRecord->marker != convergenceMarker ||
                peerRecord->magic != magic_) {
                continue;
            }
            ready[core] = true;
            --remaining;
            if (peerRecord->status !=
                TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS) {
                allCoresSucceeded_ = false;
            }
        }
        cursor = (cursor + 1U) % activeCoreCount_;
        if (TimedOut(operationStartCycles_)) {
            allCoresSucceeded_ = false;
            break;
        }
    }
    ++convergenceRound_;
}

#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
__aicore__ inline void MoonEpCombineV2::RecordProfilePoint(
    uint32_t index)
{
    if (index < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount) {
        profileTimePoint_[index] = static_cast<int64_t>(GetSystemCycle());
    }
}

__aicore__ inline void MoonEpCombineV2::WriteProfile()
{
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
    record->blockDim = activeCoreCount_;
    record->timePointCount =
        TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount;
    for (uint32_t index = 0U;
        index < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCapacity;
        ++index) {
        record->timePoint[index] = index <
                TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount ?
            profileTimePoint_[index] : 0;
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
        failureLane_ < 2U ? static_cast<int64_t>(
            TileXRMoonEp::MoonEpCombineV2Qp(core_, failureLane_)) : -1;
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_CQ_STATUS] =
        static_cast<int64_t>(failureCqStatus_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_EXPECTED] =
        static_cast<int64_t>(failureExpected_);
    record->timePoint[
        TileXRMoonEp::MOONEP_COMBINE_V2_DIAG_OBSERVED] =
        static_cast<int64_t>(failureObserved_);
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(record),
        sizeof(*record));
}
#endif

__aicore__ inline void MoonEpCombineV2::Process()
{
    if (kEnableSafetyChecks && !valid_) {
        return;
    }
    registry_ = TileXR::GetUDMARegistry(args_);
    InitBuffers();
    bool succeeded = true;
    if (kEnableSafetyChecks) {
        succeeded = failureStatus_ ==
            TileXRMoonEp::MOONEP_COMBINE_V2_SUCCESS && CheckPoison();
        if (succeeded) {
            succeeded = ValidateDestinations();
        }
        PublishFailureAndConverge();
        succeeded = succeeded && allCoresSucceeded_;
    }
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    RecordProfilePoint(TileXRMoonEp::MOONEP_COMBINE_V2_TIME_PREPARE_END);
#endif
    if (kEnableSafetyChecks && !succeeded) {
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
        WriteProfile();
#endif
        return;
    }
    for (uint32_t step = 0U;
        step < stepCount_ && succeeded;
        ++step) {
        succeeded = WaitAdmission(step);
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_STEP0_GRANT_END +
            step * TileXRMoonEp::kMoonEpCombineV2ProfileStepPointStride);
#endif
        if (!succeeded) {
            break;
        }
        const uint32_t peer = TileXRMoonEp::MoonEpCombineV2Peer(
            rank_, step, core_, rankSize_);
        succeeded = peer == rank_ ? SendSelfStep(peer) :
            SendRemoteStep(peer, step);
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_STEP0_SEND_END +
            step * TileXRMoonEp::kMoonEpCombineV2ProfileStepPointStride);
#endif
    }
    if (succeeded) {
        succeeded = WaitFinalCqs();
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FINAL_CQ_END);
#endif
    }
    if (succeeded) {
        succeeded = WaitInboundDone();
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
        RecordProfilePoint(
            TileXRMoonEp::MOONEP_COMBINE_V2_TIME_INBOUND_DONE_END);
#endif
    }
    if (kEnableSafetyChecks) {
        PublishFailureAndConverge();
        succeeded = succeeded && allCoresSucceeded_;
    }
    if (succeeded && !ReduceHidden()) {
        SetFailure(TileXRMoonEp::MOONEP_COMBINE_V2_INVALID_CONFIG,
            UINT32_MAX, UINT32_MAX, UINT32_MAX);
    }
#if defined(TILEXR_MOONEP_COMBINE_V2_ENABLE_PROFILING)
    RecordProfilePoint(TileXRMoonEp::MOONEP_COMBINE_V2_TIME_FINAL_END);
    WriteProfile();
#endif
}

} // namespace

#endif // TILEXR_MOONEP_COMBINE_V2_KERNEL_H
