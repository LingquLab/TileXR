/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_udma.h"
#include "tilexr_udma_alltoall_group_route.h"
#include "tilexr_udma_alltoall_group_trace.h"

namespace {

constexpr uint32_t TILEXR_ALLTOALL_GROUP_SEND_CORES = 16U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SEND_WORKERS = 32U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_BLOCK_DIM = 64U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_DEFAULT_WIDTH = 16U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_EXPERIMENTAL_WIDTH = 4U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_MAX_QUIET_BATCH = 64U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_SIGNAL_STRIDE = 512U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE = 1024U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_RELAY_BYTES = 64U * 1024U;
constexpr uint64_t TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES = 10000000000ULL;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ERROR_WORDS = 12U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ERROR_BYTES =
    TILEXR_ALLTOALL_GROUP_ERROR_WORDS * TILEXR_ALLTOALL_GROUP_BLOCK_DIM *
    sizeof(uint32_t);
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_CONFIG = 1U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_QUIET = 2U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_WAIT = 3U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_COMBINED = 0U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL = 1U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY = 2U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_SECONDARY = 3U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_SEND = 4U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_COPY = 5U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_SEND = 6U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_ALL_SEND = 7U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_WAIT = 8U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_COPY = 9U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_NO_COPY = 10U;

struct AllToAllGroupDeviceError {
    uint32_t valid;
    uint32_t stage;
    uint32_t group;
    uint32_t pass;
    int32_t peer;
    uint32_t qpIdx;
    uint32_t quietStatus;
    uint32_t reserved;
    uint64_t expectedToken;
    uint64_t observedToken;
};

__aicore__ inline int32_t AllToAllGroupDevicePeer(
    int32_t rank, int32_t rankSize, uint32_t group, uint32_t lane,
    uint32_t groupWidth)
{
    if (rankSize < 8 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        (rankSize & 7) != 0 || rank < 0 || rank >= rankSize ||
        (groupWidth != TILEXR_ALLTOALL_GROUP_DEFAULT_WIDTH &&
            groupWidth != TILEXR_ALLTOALL_GROUP_EXPERIMENTAL_WIDTH) ||
        lane >= groupWidth) {
        return -1;
    }
    const uint32_t halfWidth = groupWidth / 2U;
    const uint32_t index = lane < halfWidth ? lane : lane - halfWidth;
    const int32_t distance = static_cast<int32_t>(
        group * halfWidth + index + 1U);
    const int32_t diameter = rankSize / 2;
    if (distance > diameter ||
        (lane >= halfWidth && distance == diameter)) {
        return -1;
    }
    return lane < halfWidth ?
        (rank + distance) % rankSize :
        (rank - distance + rankSize) % rankSize;
}

__aicore__ inline uint64_t AllToAllGroupDeviceToken(
    uint32_t invocationId, uint32_t group, uint32_t pass)
{
    const uint64_t invocation = static_cast<uint64_t>(invocationId) + 1ULL;
    const uint64_t slot = static_cast<uint64_t>(invocationId & 1U);
    return (invocation << 32U) | (slot << 31U) |
        (static_cast<uint64_t>(group) << 16U) |
        (static_cast<uint64_t>(pass) + 1ULL);
}

__aicore__ inline bool AllToAllGroupPeerInRouteStageDevice(
    int32_t rank, int32_t peer, uint32_t routeStage)
{
    if (rank < 0 || peer < 0 || rank == peer ||
        routeStage > TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_NO_COPY) {
        return false;
    }
    if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_COMBINED ||
        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_ALL_SEND ||
        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_NO_COPY) {
        return true;
    }
    const bool crossNode =
        rank / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode) !=
        peer / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode);
    if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL ||
        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_SEND ||
        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_COPY) {
        return !crossNode;
    }
    if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_SEND ||
        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_WAIT ||
        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_COPY) {
        return crossNode;
    }
    return crossNode;
}

__aicore__ inline bool AllToAllGroupStageRunsSendDevice(uint32_t routeStage)
{
    return routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_COPY;
}

__aicore__ inline bool AllToAllGroupStageRunsReceiveDevice(uint32_t routeStage)
{
    return routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_SEND &&
        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_SEND &&
        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_ALL_SEND;
}

__aicore__ inline bool AllToAllGroupStageRunsCopyDevice(uint32_t routeStage)
{
    return AllToAllGroupStageRunsReceiveDevice(routeStage) &&
        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_WAIT &&
        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_NO_COPY;
}

__aicore__ inline bool AllToAllGroupStageWaitsForSignalDevice(uint32_t routeStage)
{
    return AllToAllGroupStageRunsReceiveDevice(routeStage) &&
        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_COPY &&
        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_COPY;
}

__aicore__ inline bool AllToAllGroupReceivePeerInRouteStageDevice(
    int32_t rank, int32_t peer, uint32_t routeStage)
{
    if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_COPY) {
        return rank >= 0 && peer >= 0 && rank != peer;
    }
    return AllToAllGroupPeerInRouteStageDevice(rank, peer, routeStage);
}

__aicore__ inline int32_t AllToAllGroupCopyoutLaneDevice(
    uint32_t worker, uint32_t assignment, uint32_t copyoutWorkers)
{
    if (copyoutWorkers >= 32U) {
        return assignment == 0U ?
            static_cast<int32_t>(worker % TILEXR_ALLTOALL_GROUP_SEND_CORES) : -1;
    }
    const uint32_t lane = worker + assignment * copyoutWorkers;
    return lane < TILEXR_ALLTOALL_GROUP_SEND_CORES ?
        static_cast<int32_t>(lane) : -1;
}

__aicore__ inline bool AllToAllGroupRemoteAssistDevice(
    uint32_t worker, uint32_t copyoutWorkers)
{
    return copyoutWorkers >= 32U && worker >= TILEXR_ALLTOALL_GROUP_SEND_CORES;
}

__aicore__ inline void AllToAllGroupSelectRouteQps(
    const __gm__ TileXR::CommArgs* args, int32_t peer,
    uint32_t& primaryQp, uint32_t& secondaryQp,
    uint32_t& primaryWeight, uint32_t& secondaryWeight)
{
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    const uint32_t qpCount = TileXR::UDMAGetLogicalQpNum(udmaInfo);
    primaryQp = 0U;
    primaryWeight = TileXR::UDMAGetQpWeight(udmaInfo, peer, 0U);
    for (uint32_t qp = 1U; qp < qpCount; ++qp) {
        const uint32_t weight = TileXR::UDMAGetQpWeight(udmaInfo, peer, qp);
        if (weight > primaryWeight) {
            primaryQp = qp;
            primaryWeight = weight;
        }
    }

    secondaryQp = primaryQp;
    secondaryWeight = 0U;
    for (uint32_t qp = 0U; qp < qpCount; ++qp) {
        const uint32_t weight = TileXR::UDMAGetQpWeight(udmaInfo, peer, qp);
        if (weight != 0U && weight < primaryWeight && weight > secondaryWeight) {
            secondaryQp = qp;
            secondaryWeight = weight;
        }
    }
}

__aicore__ inline void AllToAllGroupSplitByRouteDevice(
    uint32_t elements, uint32_t primaryWeight, uint32_t secondaryWeight,
    uint32_t primaryRouteParts, uint32_t& primaryElements,
    uint32_t& secondaryElements)
{
    if (secondaryWeight == 0U) {
        primaryElements = elements;
        secondaryElements = 0U;
        return;
    }
    uint64_t numerator = primaryWeight;
    uint64_t denominator = static_cast<uint64_t>(primaryWeight) + secondaryWeight;
    if (primaryRouteParts != TileXR::Demo::kAllToAllGroupAutoPrimaryParts) {
        numerator = primaryRouteParts > TileXR::Demo::kAllToAllGroupRouteParts ?
            TileXR::Demo::kAllToAllGroupRouteParts : primaryRouteParts;
        denominator = TileXR::Demo::kAllToAllGroupRouteParts;
    }
    primaryElements = denominator == 0U ? elements : static_cast<uint32_t>(
        static_cast<uint64_t>(elements) * numerator / denominator);
    secondaryElements = elements - primaryElements;
}

__aicore__ inline void AllToAllGroupRouteSliceForPassDevice(
    uint32_t totalElements, uint32_t passOffset, uint32_t passElements,
    uint32_t primaryElements, uint32_t route,
    uint32_t& elementOffset, uint32_t& elements)
{
    elementOffset = 0U;
    elements = 0U;
    if (route > 1U || passOffset >= totalElements || passElements == 0U) {
        return;
    }
    const uint32_t passEnd = passElements > totalElements - passOffset ?
        totalElements : passOffset + passElements;
    const uint32_t routeBegin = route == 0U ? 0U : primaryElements;
    const uint32_t routeEnd = route == 0U ? primaryElements : totalElements;
    const uint32_t begin = passOffset > routeBegin ? passOffset : routeBegin;
    const uint32_t end = passEnd < routeEnd ? passEnd : routeEnd;
    if (end > begin) {
        elementOffset = begin;
        elements = end - begin;
    }
}

__aicore__ inline bool AllToAllGroupRouteRunsInStage(uint32_t routeStage, uint32_t route)
{
    if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY) {
        return route == 0U;
    }
    if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_SECONDARY) {
        return route == 1U;
    }
    return true;
}

__aicore__ inline void AllToAllGroupCopyMte(
    __gm__ uint8_t* dst, const __gm__ uint8_t* src, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    for (uint32_t offset = 0U; offset < bytes; offset += TILEXR_ALLTOALL_GROUP_RELAY_BYTES) {
        const uint32_t tileBytes = bytes - offset < TILEXR_ALLTOALL_GROUP_RELAY_BYTES ?
            bytes - offset : TILEXR_ALLTOALL_GROUP_RELAY_BYTES;
        AscendC::GlobalTensor<uint8_t> srcGlobal;
        srcGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t*>(src) + offset);
        AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
        AscendC::DataCopyExtParams copyIn {1U, tileBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        AscendC::GlobalTensor<uint8_t> dstGlobal;
        dstGlobal.SetGlobalBuffer(dst + offset);
        AscendC::DataCopyExtParams copyOut {1U, tileBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline uint64_t AllToAllGroupLoadTokenMte(
    __gm__ uint64_t* signal, AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint8_t> signalGlobal;
    signalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(signal));
    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
    AscendC::DataCopyExtParams copyIn {
        1U, static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(relayLocal, signalGlobal, copyIn, padIn);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return relayLocal.ReinterpretCast<uint64_t>().GetValue(0);
}

__aicore__ inline bool AllToAllGroupWaitTokenMte(
    __gm__ uint64_t* signal, uint64_t expectedToken, uint64_t timeoutCycles,
    AscendC::LocalTensor<uint8_t> relayLocal, uint64_t& observed)
{
    const uint64_t begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
    observed = AllToAllGroupLoadTokenMte(signal, relayLocal);
    while (observed < expectedToken) {
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - begin >= timeoutCycles) {
            return false;
        }
        observed = AllToAllGroupLoadTokenMte(signal, relayLocal);
    }
    return observed >= expectedToken;
}

__aicore__ inline bool AllToAllGroupWaitRouteTokensMte(
    __gm__ uint64_t* primarySignal, __gm__ uint64_t* secondarySignal,
    bool waitPrimary, bool waitSecondary, uint64_t expectedToken,
    uint64_t timeoutCycles, AscendC::LocalTensor<uint8_t> relayLocal,
    uint64_t& observed)
{
    const uint64_t begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
    uint64_t primaryObserved = waitPrimary ? 0ULL : expectedToken;
    uint64_t secondaryObserved = waitSecondary ? 0ULL : expectedToken;
    while (primaryObserved < expectedToken || secondaryObserved < expectedToken) {
        if (waitPrimary && primaryObserved < expectedToken) {
            primaryObserved = AllToAllGroupLoadTokenMte(primarySignal, relayLocal);
        }
        if (waitSecondary && secondaryObserved < expectedToken) {
            secondaryObserved = AllToAllGroupLoadTokenMte(secondarySignal, relayLocal);
        }
        if (primaryObserved >= expectedToken && secondaryObserved >= expectedToken) {
            break;
        }
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - begin >= timeoutCycles) {
            observed = primaryObserved < expectedToken ? primaryObserved : secondaryObserved;
            return false;
        }
    }
    observed = primaryObserved < secondaryObserved ? primaryObserved : secondaryObserved;
    return true;
}

__aicore__ inline void AllToAllGroupRecordError(
    __gm__ int32_t* debug, uint32_t blockIdx, uint32_t stage,
    uint32_t group, uint32_t pass, int32_t peer, uint32_t qpIdx,
    uint32_t quietStatus, uint64_t expectedToken, uint64_t observedToken)
{
    if (debug == nullptr) {
        return;
    }
    auto record = reinterpret_cast<__gm__ AllToAllGroupDeviceError*>(
        debug + static_cast<uint64_t>(blockIdx) * TILEXR_ALLTOALL_GROUP_ERROR_WORDS);
    if (record->valid != 0U) {
        return;
    }
    record->stage = stage;
    record->group = group;
    record->pass = pass;
    record->peer = peer;
    record->qpIdx = qpIdx;
    record->quietStatus = quietStatus;
    record->expectedToken = expectedToken;
    record->observedToken = observedToken;
    record->valid = 1U;
}

__aicore__ inline uint64_t AllToAllGroupTraceCycle(__gm__ uint8_t* trace)
{
    return trace == nullptr ? 0ULL : static_cast<uint64_t>(AscendC::GetSystemCycle());
}

__aicore__ inline void AllToAllGroupTraceRecordKernel(
    __gm__ uint8_t* trace, uint32_t iteration, uint32_t core,
    uint64_t beginCycle, uint64_t endCycle)
{
    if (trace == nullptr || iteration >= TileXR::Demo::kAllToAllGroupTraceMaxIterations ||
        core >= TileXR::Demo::kAllToAllGroupTraceCoreCount) {
        return;
    }
    const uint64_t offset = TileXR::Demo::kAllToAllGroupTraceHeaderBytes +
        (static_cast<uint64_t>(iteration) * TileXR::Demo::kAllToAllGroupTraceCoreCount + core) *
        TileXR::Demo::kAllToAllGroupTraceCacheLineBytes;
    auto span = reinterpret_cast<__gm__ TileXR::Demo::AllToAllGroupTraceSpan*>(
        trace + offset);
    span->beginCycle = beginCycle;
    span->endCycle = endCycle;
}

__aicore__ inline void AllToAllGroupTraceRecordTask(
    __gm__ uint8_t* trace, uint32_t iteration, uint32_t core,
    uint32_t group, uint32_t pass, uint32_t phase,
    uint32_t groupCount, uint32_t passCount, int32_t peer, uint32_t qpIdx,
    uint64_t beginCycle, uint64_t endCycle)
{
    if (trace == nullptr || iteration >= TileXR::Demo::kAllToAllGroupTraceMaxIterations ||
        core >= TileXR::Demo::kAllToAllGroupTraceCoreCount || group >= groupCount ||
        pass >= passCount || phase >= TileXR::Demo::kAllToAllGroupTracePhaseCount) {
        return;
    }
    const uint64_t rawCoreBytes = static_cast<uint64_t>(groupCount) * passCount *
        TileXR::Demo::kAllToAllGroupTracePhaseCount *
        sizeof(TileXR::Demo::AllToAllGroupTraceTaskSpan);
    const uint64_t coreBytes =
        (rawCoreBytes + TileXR::Demo::kAllToAllGroupTraceCacheLineBytes - 1U) &
        ~(static_cast<uint64_t>(TileXR::Demo::kAllToAllGroupTraceCacheLineBytes) - 1ULL);
    const uint64_t coreIndex = static_cast<uint64_t>(iteration) *
        TileXR::Demo::kAllToAllGroupTraceCoreCount + core;
    const uint64_t taskIndex = ((static_cast<uint64_t>(group) * passCount + pass) *
        TileXR::Demo::kAllToAllGroupTracePhaseCount) + phase;
    const uint64_t taskBaseOffset = TileXR::Demo::kAllToAllGroupTraceHeaderBytes +
        static_cast<uint64_t>(TileXR::Demo::kAllToAllGroupTraceMaxIterations) *
        TileXR::Demo::kAllToAllGroupTraceCoreCount *
        TileXR::Demo::kAllToAllGroupTraceCacheLineBytes;
    const uint64_t offset = taskBaseOffset +
        coreIndex * coreBytes + taskIndex * sizeof(TileXR::Demo::AllToAllGroupTraceTaskSpan);
    auto span = reinterpret_cast<__gm__ TileXR::Demo::AllToAllGroupTraceTaskSpan*>(
        trace + offset);
    span->peer = peer;
    span->qpIdx = qpIdx;
    span->beginCycle = beginCycle;
    span->endCycle = endCycle;
}

struct AllToAllGroupPendingQuiet {
    int32_t peer;
    uint32_t qpIdx;
    uint32_t group;
    uint32_t pass;
    uint64_t expectedToken;
};

template <bool BatchQuiet>
struct AllToAllGroupQuietState {};

template <>
struct AllToAllGroupQuietState<true> {
    AllToAllGroupPendingQuiet pending[TILEXR_ALLTOALL_GROUP_MAX_QUIET_BATCH];
    uint32_t pendingCount = 0U;
};

__aicore__ inline uint32_t AllToAllGroupSignalSourceSlot(
    const AllToAllGroupQuietState<false>&)
{
    return 0U;
}

__aicore__ inline uint32_t AllToAllGroupSignalSourceSlot(
    const AllToAllGroupQuietState<true>& state)
{
    return state.pendingCount;
}

__aicore__ inline bool AllToAllGroupFlushQuiet(
    const __gm__ TileXR::CommArgs* args,
    AllToAllGroupPendingQuiet* pending, uint32_t& pendingCount,
    __gm__ int32_t* debug, uint32_t blockIdx,
    __gm__ uint8_t* trace, uint32_t traceIteration,
    uint32_t groupCount, uint32_t passCount)
{
    for (uint32_t index = 0U; index < pendingCount; ++index) {
        const auto& request = pending[index];
        const uint64_t quietBegin = AllToAllGroupTraceCycle(trace);
        const uint32_t quietStatus =
            TileXR::UDMAQuietStatusOnQp(args, request.peer, request.qpIdx);
        AllToAllGroupTraceRecordTask(
            trace, traceIteration, blockIdx, request.group, request.pass,
            TileXR::Demo::kAllToAllGroupTraceSendQuiet, groupCount, passCount,
            request.peer, request.qpIdx, quietBegin,
            AllToAllGroupTraceCycle(trace));
        if (quietStatus != 0U) {
            AllToAllGroupRecordError(debug, blockIdx,
                TILEXR_ALLTOALL_GROUP_STAGE_QUIET, request.group, request.pass,
                request.peer, request.qpIdx, quietStatus,
                request.expectedToken, 0ULL);
            return false;
        }
    }
    pendingCount = 0U;
    return true;
}

__aicore__ inline bool AllToAllGroupCompleteQuiet(
    const __gm__ TileXR::CommArgs* args,
    AllToAllGroupQuietState<false>&,
    uint32_t, int32_t peer, uint32_t selectedQp,
    uint32_t group, uint32_t pass, uint64_t expectedToken,
    __gm__ int32_t* debug, uint32_t blockIdx,
    __gm__ uint8_t* trace, uint32_t traceIteration,
    uint32_t groupCount, uint32_t passCount)
{
    const uint64_t quietBegin = AllToAllGroupTraceCycle(trace);
    const uint32_t quietStatus =
        TileXR::UDMAQuietStatusOnQp(args, peer, selectedQp);
    AllToAllGroupTraceRecordTask(
        trace, traceIteration, blockIdx, group, pass,
        TileXR::Demo::kAllToAllGroupTraceSendQuiet, groupCount, passCount,
        peer, selectedQp, quietBegin, AllToAllGroupTraceCycle(trace));
    if (quietStatus != 0U) {
        AllToAllGroupRecordError(debug, blockIdx,
            TILEXR_ALLTOALL_GROUP_STAGE_QUIET, group, pass, peer,
            selectedQp, quietStatus, expectedToken, 0ULL);
        return false;
    }
    return true;
}

__aicore__ inline bool AllToAllGroupCompleteQuiet(
    const __gm__ TileXR::CommArgs* args,
    AllToAllGroupQuietState<true>& state,
    uint32_t quietBatch, int32_t peer, uint32_t selectedQp,
    uint32_t group, uint32_t pass, uint64_t expectedToken,
    __gm__ int32_t* debug, uint32_t blockIdx,
    __gm__ uint8_t* trace, uint32_t traceIteration,
    uint32_t groupCount, uint32_t passCount)
{
    state.pending[state.pendingCount++] = {
        peer, selectedQp, group, pass, expectedToken};
    return state.pendingCount != quietBatch || AllToAllGroupFlushQuiet(
        args, state.pending, state.pendingCount, debug, blockIdx,
        trace, traceIteration, groupCount, passCount);
}

__aicore__ inline bool AllToAllGroupFinishQuiet(
    const __gm__ TileXR::CommArgs*, AllToAllGroupQuietState<false>&,
    __gm__ int32_t*, uint32_t, __gm__ uint8_t*, uint32_t, uint32_t, uint32_t)
{
    return true;
}

__aicore__ inline bool AllToAllGroupFinishQuiet(
    const __gm__ TileXR::CommArgs* args, AllToAllGroupQuietState<true>& state,
    __gm__ int32_t* debug, uint32_t blockIdx,
    __gm__ uint8_t* trace, uint32_t traceIteration,
    uint32_t groupCount, uint32_t passCount)
{
    return state.pendingCount == 0U || AllToAllGroupFlushQuiet(
        args, state.pending, state.pendingCount, debug, blockIdx,
        trace, traceIteration, groupCount, passCount);
}

} // namespace

template <bool BatchQuiet>
__aicore__ inline void AllToAllGroupKernelImpl(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t quietBatch)
{
    constexpr uint32_t copyoutWorkers = 32U;
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    auto groupTrace = blockIdx < TileXR::Demo::kAllToAllGroupTraceCoreCount ?
        reinterpret_cast<__gm__ uint8_t*>(groupTraceGM) : nullptr;
    const uint64_t kernelBegin = AllToAllGroupTraceCycle(groupTrace);
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto registeredMemory = reinterpret_cast<__gm__ uint8_t*>(registeredMemoryGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);
    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;

    if ((copyoutWorkers != 8U && copyoutWorkers != 16U &&
            copyoutWorkers != 32U && copyoutWorkers != 48U) ||
        routeStage > TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_NO_COPY ||
        multiChannel > 1U ||
        (groupWidth != TILEXR_ALLTOALL_GROUP_DEFAULT_WIDTH &&
            groupWidth != TILEXR_ALLTOALL_GROUP_EXPERIMENTAL_WIDTH) ||
        quietBatch == 0U ||
        quietBatch > TILEXR_ALLTOALL_GROUP_MAX_QUIET_BATCH ||
        (quietBatch & (quietBatch - 1U)) != 0U ||
        (primaryRouteParts > TileXR::Demo::kAllToAllGroupRouteParts &&
            primaryRouteParts != TileXR::Demo::kAllToAllGroupAutoPrimaryParts) ||
        TILEXR_ALLTOALL_GROUP_SEND_WORKERS + copyoutWorkers >
            TILEXR_ALLTOALL_GROUP_BLOCK_DIM ||
        blockIdx >= TILEXR_ALLTOALL_GROUP_SEND_WORKERS + copyoutWorkers ||
        !TileXR::UDMARegistryEnabled(args) || rankSize < 8 ||
        rankSize > TileXR::TILEXR_MAX_RANK_SIZE || (rankSize & 7) != 0 ||
        elementsPerPeer <= 0 || chunkElements <= 0 || passCount == 0U ||
        groupCount == 0U || groupCount != static_cast<uint32_t>(
            (rankSize - 1 + static_cast<int32_t>(groupWidth) - 1) /
            static_cast<int32_t>(groupWidth))) {
        AllToAllGroupRecordError(debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_CONFIG,
            0U, 0U, -1, 0U, 0U, 0ULL, 0ULL);
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
    pipe.InitBuffer(relayTBuf, TILEXR_ALLTOALL_GROUP_RELAY_BYTES);
    AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();

    const uint32_t slot = invocationId & 1U;
    const uint64_t payloadOffsets[2] = {payloadOffset0, payloadOffset1};
    const uint64_t signalOffsets[2] = {signalOffset0, signalOffset1};
    const uint64_t bytesPerPeer =
        static_cast<uint64_t>(elementsPerPeer) * sizeof(int32_t);

    if (blockIdx >= TILEXR_ALLTOALL_GROUP_SEND_WORKERS) {
        if (!AllToAllGroupStageRunsReceiveDevice(routeStage)) {
            AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                kernelBegin, AllToAllGroupTraceCycle(groupTrace));
            return;
        }
        const uint32_t worker = blockIdx - TILEXR_ALLTOALL_GROUP_SEND_WORKERS;
        const uint32_t selfCopyWorkers = copyoutWorkers >= 32U ? 16U : copyoutWorkers;
        const int32_t selfBegin = worker < selfCopyWorkers ? static_cast<int32_t>(
            static_cast<int64_t>(elementsPerPeer) * worker / selfCopyWorkers) : 0;
        const int32_t selfEnd = worker < selfCopyWorkers ? static_cast<int32_t>(
            static_cast<int64_t>(elementsPerPeer) * (worker + 1U) / selfCopyWorkers) : 0;
        if ((routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_COMBINED ||
                routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL ||
                routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_LOCAL_COPY ||
                routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_REMOTE_COPY) &&
            selfEnd > selfBegin) {
            const uint64_t selfCopyBegin = AllToAllGroupTraceCycle(groupTrace);
            auto selfSrc = reinterpret_cast<__gm__ uint8_t*>(
                input + static_cast<uint64_t>(rank) * elementsPerPeer + selfBegin);
            auto selfDst = reinterpret_cast<__gm__ uint8_t*>(
                output + static_cast<uint64_t>(rank) * elementsPerPeer + selfBegin);
            AllToAllGroupCopyMte(selfDst, selfSrc,
                static_cast<uint32_t>(selfEnd - selfBegin) * sizeof(int32_t), relayLocal);
            AllToAllGroupTraceRecordTask(
                groupTrace, traceIteration, blockIdx, 0U, 0U,
                TileXR::Demo::kAllToAllGroupTraceSelfCopy, groupCount, passCount,
                rank, TileXR::Demo::kAllToAllGroupTraceNoQp,
                selfCopyBegin, AllToAllGroupTraceCycle(groupTrace));
        }

        for (uint32_t group = 0U; group < groupCount; ++group) {
            for (uint32_t assignment = 0U; ; ++assignment) {
                const int32_t laneValue = AllToAllGroupCopyoutLaneDevice(
                    worker, assignment, copyoutWorkers);
                if (laneValue < 0) {
                    break;
                }
                const uint32_t lane = static_cast<uint32_t>(laneValue);
            const int32_t peer = AllToAllGroupDevicePeer(
                rank, rankSize, group, lane, groupWidth);
            if (peer < 0) {
                continue;
            }
            if (!AllToAllGroupReceivePeerInRouteStageDevice(rank, peer, routeStage)) {
                continue;
            }
            const bool crossNode =
                rank / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode) !=
                peer / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode);
            const bool remoteAssist =
                AllToAllGroupRemoteAssistDevice(worker, copyoutWorkers);
            if (remoteAssist && !crossNode) {
                continue;
            }
            uint32_t primaryQp = 0U;
            uint32_t secondaryQp = 0U;
            uint32_t primaryWeight = 0U;
            uint32_t secondaryWeight = 0U;
            AllToAllGroupSelectRouteQps(args, peer, primaryQp, secondaryQp,
                primaryWeight, secondaryWeight);
            if (multiChannel == 0U || !crossNode || secondaryQp == primaryQp) {
                secondaryWeight = 0U;
            }
            uint32_t primaryTotalElements = 0U;
            uint32_t secondaryTotalElements = 0U;
            AllToAllGroupSplitByRouteDevice(
                static_cast<uint32_t>(elementsPerPeer), primaryWeight,
                secondaryWeight, primaryRouteParts,
                primaryTotalElements, secondaryTotalElements);
            const uint32_t traceCore = copyoutWorkers == 8U ?
                TILEXR_ALLTOALL_GROUP_SEND_WORKERS + lane : blockIdx;
            for (uint32_t pass = 0U; pass < passCount; ++pass) {
                const int32_t chunkElementOffset = static_cast<int32_t>(pass) * chunkElements;
                const int32_t remaining = elementsPerPeer - chunkElementOffset;
                if (remaining <= 0) {
                    continue;
                }
                const int32_t currentElements = remaining < chunkElements ? remaining : chunkElements;
                const uint32_t copySliceCount =
                    copyoutWorkers >= 32U && crossNode ?
                    copyoutWorkers / TILEXR_ALLTOALL_GROUP_SEND_CORES : 1U;
                const uint32_t copySliceIndex = remoteAssist ?
                    worker / TILEXR_ALLTOALL_GROUP_SEND_CORES : 0U;
                uint32_t copyElementBegin = static_cast<uint32_t>(
                    static_cast<int64_t>(currentElements) * copySliceIndex / copySliceCount);
                uint32_t copyElementEnd = static_cast<uint32_t>(
                    static_cast<int64_t>(currentElements) * (copySliceIndex + 1U) /
                    copySliceCount);
                uint32_t primaryElementOffset = 0U;
                uint32_t primaryElements = 0U;
                uint32_t secondaryElementOffset = 0U;
                uint32_t secondaryElements = 0U;
                AllToAllGroupRouteSliceForPassDevice(
                    static_cast<uint32_t>(elementsPerPeer),
                    static_cast<uint32_t>(chunkElementOffset),
                    static_cast<uint32_t>(currentElements), primaryTotalElements, 0U,
                    primaryElementOffset, primaryElements);
                AllToAllGroupRouteSliceForPassDevice(
                    static_cast<uint32_t>(elementsPerPeer),
                    static_cast<uint32_t>(chunkElementOffset),
                    static_cast<uint32_t>(currentElements), primaryTotalElements, 1U,
                    secondaryElementOffset, secondaryElements);
                if ((routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY &&
                        primaryElements == 0U) ||
                    (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_SECONDARY &&
                        secondaryElements == 0U)) {
                    continue;
                }
                const uint64_t expectedToken =
                    AllToAllGroupDeviceToken(invocationId, group, pass);
                auto primarySignal = reinterpret_cast<__gm__ uint64_t*>(
                    registeredMemory + signalOffsets[slot] +
                    static_cast<uint64_t>(peer) * TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE);
                auto secondarySignal = reinterpret_cast<__gm__ uint64_t*>(
                    reinterpret_cast<__gm__ uint8_t*>(primarySignal) +
                    TILEXR_ALLTOALL_GROUP_ROUTE_SIGNAL_STRIDE);
                if (AllToAllGroupStageWaitsForSignalDevice(routeStage)) {
                    uint64_t observed = 0ULL;
                    const uint64_t waitBegin = AllToAllGroupTraceCycle(groupTrace);
                    const bool waitPrimary = primaryElements != 0U &&
                        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_SECONDARY;
                    const bool waitSecondary = secondaryElements != 0U &&
                        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY;
                    if (!AllToAllGroupWaitRouteTokensMte(
                            primarySignal, secondarySignal, waitPrimary, waitSecondary,
                            expectedToken, TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES,
                            relayLocal, observed)) {
                        AllToAllGroupTraceRecordTask(
                            groupTrace, traceIteration, traceCore, group, pass,
                            TileXR::Demo::kAllToAllGroupTraceReceiveWait, groupCount, passCount,
                            peer, TileXR::Demo::kAllToAllGroupTraceNoQp,
                            waitBegin, AllToAllGroupTraceCycle(groupTrace));
                        AllToAllGroupRecordError(debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_WAIT,
                            group, pass, peer, 0U, 0U, expectedToken, observed);
                        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
                        return;
                    }
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceReceiveWait, groupCount, passCount,
                        peer, TileXR::Demo::kAllToAllGroupTraceNoQp,
                        waitBegin, AllToAllGroupTraceCycle(groupTrace));
                }
                if (!AllToAllGroupStageRunsCopyDevice(routeStage)) {
                    continue;
                }
                if (routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY ||
                    routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_SECONDARY) {
                    const uint32_t routeOffset =
                        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY ?
                        primaryElementOffset : secondaryElementOffset;
                    const uint32_t routeElements =
                        routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY ?
                        primaryElements : secondaryElements;
                    const uint32_t routeBegin =
                        routeOffset - static_cast<uint32_t>(chunkElementOffset);
                    const uint32_t routeEnd = routeBegin + routeElements;
                    copyElementBegin = copyElementBegin > routeBegin ?
                        copyElementBegin : routeBegin;
                    copyElementEnd = copyElementEnd < routeEnd ? copyElementEnd : routeEnd;
                }
                if (copyElementEnd <= copyElementBegin) {
                    continue;
                }
                const uint32_t copyBytes =
                    (copyElementEnd - copyElementBegin) * sizeof(int32_t);
                auto relaySrc = registeredMemory + payloadOffsets[slot] +
                    static_cast<uint64_t>(peer) * bytesPerPeer +
                    static_cast<uint64_t>(chunkElementOffset + copyElementBegin) *
                    sizeof(int32_t);
                auto relayDst = reinterpret_cast<__gm__ uint8_t*>(
                    output + static_cast<uint64_t>(peer) * elementsPerPeer +
                    chunkElementOffset + copyElementBegin);
                const uint64_t receiveCopyBegin = AllToAllGroupTraceCycle(groupTrace);
                AllToAllGroupCopyMte(relayDst, relaySrc, copyBytes, relayLocal);
                AllToAllGroupTraceRecordTask(
                    groupTrace, traceIteration, traceCore, group, pass,
                    TileXR::Demo::kAllToAllGroupTraceReceiveCopy, groupCount, passCount,
                    peer, TileXR::Demo::kAllToAllGroupTraceNoQp,
                    receiveCopyBegin, AllToAllGroupTraceCycle(groupTrace));
            }
            }
        }
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        return;
    }

    if (!AllToAllGroupStageRunsSendDevice(routeStage)) {
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        return;
    }
    const uint32_t lane = blockIdx % TILEXR_ALLTOALL_GROUP_SEND_CORES;
    const uint32_t workerRoute = blockIdx / TILEXR_ALLTOALL_GROUP_SEND_CORES;
    AllToAllGroupQuietState<BatchQuiet> quietState;
    for (uint32_t group = 0U; group < groupCount; ++group) {
        const int32_t peer = AllToAllGroupDevicePeer(
            rank, rankSize, group, lane, groupWidth);
        if (peer < 0) {
            continue;
        }
        if (!AllToAllGroupPeerInRouteStageDevice(rank, peer, routeStage)) {
            continue;
        }
        uint32_t primaryQp = 0U;
        uint32_t secondaryQp = 0U;
        uint32_t primaryWeight = 0U;
        uint32_t secondaryWeight = 0U;
        AllToAllGroupSelectRouteQps(args, peer, primaryQp, secondaryQp,
            primaryWeight, secondaryWeight);
        const bool crossNode =
            rank / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode) !=
            peer / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode);
        if (multiChannel == 0U || !crossNode || secondaryQp == primaryQp) {
            secondaryWeight = 0U;
        }
        uint32_t primaryTotalElements = 0U;
        uint32_t secondaryTotalElements = 0U;
        AllToAllGroupSplitByRouteDevice(
            static_cast<uint32_t>(elementsPerPeer), primaryWeight,
            secondaryWeight, primaryRouteParts,
            primaryTotalElements, secondaryTotalElements);
        for (uint32_t pass = 0U; pass < passCount; ++pass) {
            const int32_t chunkElementOffset = static_cast<int32_t>(pass) * chunkElements;
            const int32_t remaining = elementsPerPeer - chunkElementOffset;
            if (remaining <= 0) {
                continue;
            }
            const int32_t currentElements = remaining < chunkElements ? remaining : chunkElements;
            uint32_t primaryElementOffset = 0U;
            uint32_t primaryElements = 0U;
            uint32_t secondaryElementOffset = 0U;
            uint32_t secondaryElements = 0U;
            AllToAllGroupRouteSliceForPassDevice(
                static_cast<uint32_t>(elementsPerPeer),
                static_cast<uint32_t>(chunkElementOffset),
                static_cast<uint32_t>(currentElements), primaryTotalElements, 0U,
                primaryElementOffset, primaryElements);
            AllToAllGroupRouteSliceForPassDevice(
                static_cast<uint32_t>(elementsPerPeer),
                static_cast<uint32_t>(chunkElementOffset),
                static_cast<uint32_t>(currentElements), primaryTotalElements, 1U,
                secondaryElementOffset, secondaryElements);
            const uint64_t expectedToken =
                AllToAllGroupDeviceToken(invocationId, group, pass);
            const uint32_t routeBegin = workerRoute;
            const uint32_t routeEnd = workerRoute + 1U;
            for (uint32_t route = routeBegin; route < routeEnd; ++route) {
                if (!AllToAllGroupRouteRunsInStage(routeStage, route)) {
                    continue;
                }
                const uint32_t segmentElements = route == 0U ?
                    primaryElements : secondaryElements;
                if (segmentElements == 0U) {
                    continue;
                }
                const uint32_t segmentElementOffset = route == 0U ?
                    primaryElementOffset : secondaryElementOffset;
                const uint32_t selectedQp = route == 0U ? primaryQp : secondaryQp;
                const uint64_t elementOffset = segmentElementOffset;
                auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer +
                    elementOffset;
                const uint64_t remotePayloadOffset = payloadOffsets[slot] +
                    static_cast<uint64_t>(rank) * bytesPerPeer +
                    elementOffset * sizeof(int32_t);
                const uint64_t remoteSignalOffset = signalOffsets[slot] +
                    static_cast<uint64_t>(rank) * TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE +
                    static_cast<uint64_t>(route) *
                        TILEXR_ALLTOALL_GROUP_ROUTE_SIGNAL_STRIDE;
                const uint64_t signalSourceIndex =
                    static_cast<uint64_t>(blockIdx) *
                    TILEXR_ALLTOALL_GROUP_MAX_QUIET_BATCH +
                    AllToAllGroupSignalSourceSlot(quietState);
                auto signalLocal = reinterpret_cast<__gm__ uint64_t*>(
                    reinterpret_cast<__gm__ uint8_t*>(debug) +
                    TILEXR_ALLTOALL_GROUP_ERROR_BYTES +
                    signalSourceIndex * sizeof(uint64_t));
                *signalLocal = expectedToken;
                TileXR::UDMACleanCacheLines(
                    reinterpret_cast<__gm__ uint8_t*>(signalLocal), sizeof(uint64_t));
                const uint64_t putBegin = AllToAllGroupTraceCycle(groupTrace);
                TileXR::UDMAPutNbiOnQpWithFlag<int32_t>(
                    args, peer, selectedQp, localSrc, remotePayloadOffset,
                    segmentElements * sizeof(int32_t),
                    TileXR::TILEXR_UDMA_SQE_FLAG_COMPLETION);
                TileXR::UDMAPutNbiOnQpWithFlag<uint64_t>(
                    args, peer, selectedQp, signalLocal, remoteSignalOffset,
                    sizeof(uint64_t),
                    TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
                AllToAllGroupTraceRecordTask(
                    groupTrace, traceIteration, blockIdx, group, pass,
                    TileXR::Demo::kAllToAllGroupTraceSendPutSignal, groupCount, passCount,
                    peer, selectedQp, putBegin, AllToAllGroupTraceCycle(groupTrace));
                if (!AllToAllGroupCompleteQuiet(
                        args, quietState, quietBatch, peer, selectedQp,
                        group, pass, expectedToken, debug, blockIdx,
                        groupTrace, traceIteration, groupCount, passCount)) {
                    AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                        kernelBegin, AllToAllGroupTraceCycle(groupTrace));
                    return;
                }
            }
        }
    }
    if (!AllToAllGroupFinishQuiet(
            args, quietState, debug, blockIdx, groupTrace, traceIteration,
            groupCount, passCount)) {
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        return;
    }
    AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
        kernelBegin, AllToAllGroupTraceCycle(groupTrace));
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_group_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t quietBatch)
{
    AllToAllGroupKernelImpl<false>(
        commArgsGM, inputGM, outputGM, registeredMemoryGM, debugGM, invocationId,
        elementsPerPeer, chunkElements, passCount, groupCount,
        payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
        groupTraceGM, traceIteration, routeStage, multiChannel, primaryRouteParts,
        groupWidth, quietBatch);
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_group_batch_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t quietBatch)
{
    AllToAllGroupKernelImpl<true>(
        commArgsGM, inputGM, outputGM, registeredMemoryGM, debugGM, invocationId,
        elementsPerPeer, chunkElements, passCount, groupCount,
        payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
        groupTraceGM, traceIteration, routeStage, multiChannel, primaryRouteParts,
        groupWidth, quietBatch);
}

void launch_tilexr_udma_all_to_all_group(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR registeredMemory, GM_ADDR debug, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    GM_ADDR groupTrace, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t quietBatch)
{
    if (quietBatch == 1U) {
        tilexr_udma_all_to_all_group_kernel<<<blockDim, nullptr, stream>>>(
            commArgs, input, output, registeredMemory, debug, invocationId,
            elementsPerPeer, chunkElements, passCount, groupCount,
            payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
            groupTrace, traceIteration, routeStage,
            multiChannel, primaryRouteParts, groupWidth, quietBatch);
    } else {
        tilexr_udma_all_to_all_group_batch_kernel<<<blockDim, nullptr, stream>>>(
            commArgs, input, output, registeredMemory, debug, invocationId,
            elementsPerPeer, chunkElements, passCount, groupCount,
            payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
            groupTrace, traceIteration, routeStage,
            multiChannel, primaryRouteParts, groupWidth, quietBatch);
    }
}
