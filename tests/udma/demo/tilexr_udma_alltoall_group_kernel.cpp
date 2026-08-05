/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_sdma.h"
#include "tilexr_udma.h"
#include "tilexr_udma_alltoall_group_route.h"
#include "tilexr_udma_alltoall_group_simt.h"
#include "tilexr_udma_alltoall_group_trace.h"

namespace {

constexpr uint32_t TILEXR_ALLTOALL_GROUP_SEND_CORES = 16U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SEND_WORKERS = 32U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SIMT_SEND_WORKERS = 1U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_BLOCK_DIM = 64U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_DEFAULT_WIDTH = 16U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_EXPERIMENTAL_WIDTH = 4U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_MAX_QUIET_BATCH = 64U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_MAX_GROUP_COUNT = 64U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_READY_BITMAP_WORDS =
    (TileXR::TILEXR_MAX_RANK_SIZE + 63U) / 64U;
static_assert(TILEXR_ALLTOALL_GROUP_READY_BITMAP_WORDS * 64U >=
    TileXR::TILEXR_MAX_RANK_SIZE, "ready bitmap must cover every peer task");
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ROUTE_SIGNAL_STRIDE = 512U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE = 1024U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_RELAY_BYTES = 64U * 1024U;
constexpr uint64_t TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES = 10000000000ULL;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ERROR_WORDS = 12U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ERROR_BYTES =
    TILEXR_ALLTOALL_GROUP_ERROR_WORDS * TILEXR_ALLTOALL_GROUP_BLOCK_DIM *
    sizeof(uint32_t);
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SIGNAL_SOURCE_BYTES =
    TILEXR_ALLTOALL_GROUP_SEND_WORKERS *
    TILEXR_ALLTOALL_GROUP_MAX_QUIET_BATCH * sizeof(uint64_t);
constexpr uint32_t TILEXR_ALLTOALL_GROUP_CREDIT_STRIDE = 512U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_CREDIT_WORDS =
    TILEXR_ALLTOALL_GROUP_CREDIT_STRIDE / sizeof(uint64_t);
constexpr uint32_t TILEXR_ALLTOALL_GROUP_IPC_SIGNAL_STRIDE = 512U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_IPC_SIGNAL_WORDS =
    TILEXR_ALLTOALL_GROUP_IPC_SIGNAL_STRIDE / sizeof(uint64_t);
static_assert(TILEXR_ALLTOALL_GROUP_CREDIT_STRIDE == TileXR::CREDIT_IPC_STRIDE,
    "grouped credit slot must match the runtime IPC layout");
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_CONFIG = 1U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_QUIET = 2U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_WAIT = 3U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_CREDIT_WAIT = 4U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_SDMA = 5U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_SDMA_PREWARM = 6U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SDMA_FALLBACK = 0U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SDMA_COMPLETE = 1U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SDMA_FAILED = 2U;
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

__aicore__ inline bool AllToAllGroupCoResidentPeerDevice(
    int32_t rank, int32_t peer, int32_t localRankSize, uint32_t npuCount)
{
    return npuCount != 0U && localRankSize > static_cast<int32_t>(npuCount) &&
        rank >= 0 && peer >= 0 && rank != peer &&
        rank / localRankSize == peer / localRankSize &&
        (rank % localRankSize) % static_cast<int32_t>(npuCount) ==
            (peer % localRankSize) % static_cast<int32_t>(npuCount);
}

__aicore__ inline uint64_t AllToAllGroupIpcSignalOffsetDevice(
    uint32_t slot, int32_t rankSize, int32_t sourceRank)
{
    return (static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
        static_cast<uint64_t>(sourceRank)) *
        TILEXR_ALLTOALL_GROUP_IPC_SIGNAL_STRIDE;
}

__aicore__ inline uint64_t AllToAllGroupIpcPayloadOffsetDevice(
    uint32_t slot, int32_t rankSize, uint64_t bytesPerPeer,
    int32_t sourceRank)
{
    return static_cast<uint64_t>(TileXR::IPC_DATA_OFFSET) +
        (static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
            static_cast<uint64_t>(sourceRank)) * bytesPerPeer;
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

__aicore__ inline uint64_t AllToAllGroupDeviceTerminalCreditToken(
    uint32_t invocationId, uint32_t groupCount)
{
    return AllToAllGroupDeviceToken(invocationId, groupCount, 0U);
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

__aicore__ inline bool AllToAllGroupCreditOwnerDevice(uint32_t worker)
{
    return worker < TILEXR_ALLTOALL_GROUP_SEND_CORES;
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

__aicore__ inline void AllToAllGroupPublishIpcToken(
    __gm__ uint8_t* remoteBase, uint64_t signalOffset, uint64_t token,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    auto tokenLocal = relayLocal.ReinterpretCast<uint64_t>();
    tokenLocal.SetValue(0, token);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint64_t> remoteSignal;
    remoteSignal.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint64_t*>(remoteBase + signalOffset),
        TILEXR_ALLTOALL_GROUP_IPC_SIGNAL_WORDS);
    AscendC::DataCopy(
        remoteSignal, tokenLocal, TILEXR_ALLTOALL_GROUP_IPC_SIGNAL_WORDS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void AllToAllGroupSendIpcLoopback(
    const __gm__ TileXR::CommArgs* args, __gm__ int32_t* input,
    int32_t peer, uint32_t slot, uint64_t bytesPerPeer,
    int32_t chunkElementOffset, int32_t currentElements,
    uint64_t expectedToken, AscendC::LocalTensor<uint8_t> relayLocal)
{
    auto remoteBase = args->peerMems[peer];
    auto src = reinterpret_cast<__gm__ uint8_t*>(
        input + static_cast<uint64_t>(peer) *
            (bytesPerPeer / sizeof(int32_t)) + chunkElementOffset);
    auto dst = remoteBase + AllToAllGroupIpcPayloadOffsetDevice(
        slot, args->rankSize, bytesPerPeer, args->rank) +
        static_cast<uint64_t>(chunkElementOffset) * sizeof(int32_t);
    AllToAllGroupCopyMte(
        dst, src, static_cast<uint32_t>(currentElements) * sizeof(int32_t),
        relayLocal);
    AllToAllGroupPublishIpcToken(
        remoteBase,
        AllToAllGroupIpcSignalOffsetDevice(
            slot, args->rankSize, args->rank),
        expectedToken, relayLocal);
}

__aicore__ inline uint32_t AllToAllGroupCopySdmaSubmit(
    const __gm__ TileXR::CommArgs* args, __gm__ uint8_t* dst,
    __gm__ uint8_t* src, uint32_t bytes, uint32_t channel,
    uint64_t& event, TileXR::SDMASubmitTrace& submitTrace)
{
    event = 0ULL;
    if (!TileXR::SDMAEnabled(args)) {
        return TILEXR_ALLTOALL_GROUP_SDMA_FALLBACK;
    }
    event = TileXR::SDMACopyNbi(
        args, dst, src, bytes, channel, &submitTrace);
    if (event == 0ULL) {
        return TILEXR_ALLTOALL_GROUP_SDMA_FALLBACK;
    }
    return TILEXR_ALLTOALL_GROUP_SDMA_COMPLETE;
}

__aicore__ inline uint32_t AllToAllGroupCopySdmaWait(
    const __gm__ TileXR::CommArgs* args, uint64_t event, uint32_t channel)
{
    return TileXR::SDMAWait(args, event, channel) ?
        TILEXR_ALLTOALL_GROUP_SDMA_COMPLETE :
        TILEXR_ALLTOALL_GROUP_SDMA_FAILED;
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

__aicore__ inline uint64_t AllToAllGroupLoadCreditMte(
    __gm__ uint64_t* credit, AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint64_t> creditGlobal;
    creditGlobal.SetGlobalBuffer(
        credit, TILEXR_ALLTOALL_GROUP_CREDIT_WORDS);
    auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();
    AscendC::DataCopy(
        creditLocal, creditGlobal, TILEXR_ALLTOALL_GROUP_CREDIT_WORDS);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return creditLocal.GetValue(0);
}

__aicore__ inline bool AllToAllGroupWaitCreditMte(
    __gm__ uint64_t* credit, uint64_t expectedToken, uint64_t timeoutCycles,
    AscendC::LocalTensor<uint8_t> relayLocal, uint64_t& observed)
{
    const uint64_t begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
    observed = AllToAllGroupLoadCreditMte(credit, relayLocal);
    while (observed < expectedToken) {
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - begin >= timeoutCycles) {
            return false;
        }
        observed = AllToAllGroupLoadCreditMte(credit, relayLocal);
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

__aicore__ inline bool AllToAllGroupRouteTokensReadyMte(
    __gm__ uint64_t* primarySignal, __gm__ uint64_t* secondarySignal,
    bool waitPrimary, bool waitSecondary, uint64_t expectedToken,
    AscendC::LocalTensor<uint8_t> relayLocal, uint64_t& observed)
{
    const uint64_t primaryObserved = waitPrimary ?
        AllToAllGroupLoadTokenMte(primarySignal, relayLocal) : expectedToken;
    const uint64_t secondaryObserved = waitSecondary ?
        AllToAllGroupLoadTokenMte(secondarySignal, relayLocal) : expectedToken;
    observed = primaryObserved < secondaryObserved ?
        primaryObserved : secondaryObserved;
    return observed >= expectedToken;
}

__aicore__ inline void AllToAllGroupRecordError(
    __gm__ int32_t* debug, uint32_t blockIdx, uint32_t stage,
    uint32_t group, uint32_t pass, int32_t peer, uint32_t qpIdx,
    uint32_t quietStatus, uint64_t expectedToken, uint64_t observedToken);

__aicore__ inline int32_t AllToAllGroupNextCreditPeerDevice(
    int32_t rank, int32_t rankSize, uint32_t completedGroup,
    uint32_t lane, uint32_t groupCount, uint32_t groupWidth)
{
    if (completedGroup + 1U >= groupCount) {
        return -1;
    }
    return AllToAllGroupDevicePeer(
        rank, rankSize, completedGroup + 1U, lane, groupWidth);
}

__aicore__ inline void AllToAllGroupPublishCredit(
    const __gm__ TileXR::CommArgs* args,
    int32_t rank, int32_t peer, uint64_t creditToken, uint64_t creditOffset,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    auto remoteCredit = reinterpret_cast<__gm__ uint64_t*>(
        args->creditMems[peer] + creditOffset +
        static_cast<uint64_t>(rank) * TILEXR_ALLTOALL_GROUP_CREDIT_STRIDE);
    auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();
    creditLocal.SetValue(0, creditToken);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint64_t> remoteCreditGlobal;
    remoteCreditGlobal.SetGlobalBuffer(
        remoteCredit, TILEXR_ALLTOALL_GROUP_CREDIT_WORDS);
    AscendC::DataCopy(
        remoteCreditGlobal, creditLocal,
        TILEXR_ALLTOALL_GROUP_CREDIT_WORDS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void AllToAllGroupPublishNextCredit(
    const __gm__ TileXR::CommArgs* args,
    int32_t rank, int32_t rankSize, uint32_t invocationId,
    uint32_t completedGroup, uint32_t lane, uint32_t groupCount,
    uint32_t groupWidth, uint64_t creditOffset,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    const int32_t nextPeer = AllToAllGroupNextCreditPeerDevice(
        rank, rankSize, completedGroup, lane, groupCount, groupWidth);
    if (nextPeer < 0) {
        return;
    }
    AllToAllGroupPublishCredit(
        args, rank, nextPeer,
        AllToAllGroupDeviceToken(invocationId, completedGroup + 1U, 0U),
        creditOffset, relayLocal);
}

__aicore__ inline void AllToAllGroupPublishTerminalCredits(
    const __gm__ TileXR::CommArgs* args,
    int32_t rank, int32_t rankSize, uint32_t invocationId,
    uint32_t worker, uint32_t copyoutWorkers, uint32_t groupCount,
    uint32_t groupWidth, uint64_t creditOffset,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    if (!AllToAllGroupCreditOwnerDevice(worker)) {
        return;
    }
    const uint64_t terminalToken =
        AllToAllGroupDeviceTerminalCreditToken(invocationId, groupCount);
    for (uint32_t assignment = 0U; ; ++assignment) {
        const int32_t laneValue = AllToAllGroupCopyoutLaneDevice(
            worker, assignment, copyoutWorkers);
        if (laneValue < 0) {
            return;
        }
        const int32_t firstPeer = AllToAllGroupDevicePeer(
            rank, rankSize, 0U, static_cast<uint32_t>(laneValue), groupWidth);
        if (firstPeer >= 0) {
            AllToAllGroupPublishCredit(
                args, rank, firstPeer, terminalToken, creditOffset, relayLocal);
        }
    }
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
    uint64_t beginCycle, uint64_t endCycle,
    uint32_t sdmaHead = 0U, uint32_t sdmaTail = 0U,
    uint32_t sdmaNewTail = 0U, uint32_t sdmaDepth = 0U,
    uint32_t sdmaGeneration = 0U)
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
    span->sdmaHead = sdmaHead;
    span->sdmaTail = sdmaTail;
    span->sdmaNewTail = sdmaNewTail;
    span->sdmaDepth = sdmaDepth;
    span->sdmaGeneration = sdmaGeneration;
    span->beginCycle = beginCycle;
    span->endCycle = endCycle;
}

__aicore__ inline void AllToAllGroupTraceRecordSimtTask(
    __gm__ uint8_t* trace, uint32_t iteration,
    __ubuf__ TileXR::Demo::AllToAllGroupSimtBatch* batch, uint32_t task,
    uint32_t phase, uint32_t groupCount, uint32_t passCount,
    uint64_t beginCycle, uint64_t endCycle, uint32_t pollCount = 0U)
{
    const uint32_t worker = batch->worker[task];
    const uint32_t traceCore = worker == 0U ? 0U : 32U + worker;
    AllToAllGroupTraceRecordTask(
        trace, iteration, traceCore,
        batch->group[task], batch->pass[task], phase, groupCount, passCount,
        static_cast<int32_t>(batch->peer[task]), batch->qpIdx[task],
        beginCycle, endCycle, worker, worker / 16U, batch->byteCount[task], pollCount);
}

__aicore__ inline bool AllToAllGroupWaitTerminalCredit(
    const __gm__ TileXR::CommArgs* args,
    int32_t rank, int32_t rankSize, uint32_t invocationId,
    uint32_t lane, uint32_t groupCount, uint32_t passCount,
    uint32_t groupWidth, uint64_t creditOffset,
    AscendC::LocalTensor<uint8_t> relayLocal,
    __gm__ int32_t* debug, uint32_t blockIdx,
    __gm__ uint8_t* groupTrace, uint32_t traceIteration)
{
    const int32_t firstPeer = AllToAllGroupDevicePeer(
        rank, rankSize, 0U, lane, groupWidth);
    if (firstPeer < 0) {
        return true;
    }
    const uint64_t expectedCredit =
        AllToAllGroupDeviceTerminalCreditToken(invocationId, groupCount);
    auto creditSignal = reinterpret_cast<__gm__ uint64_t*>(
        args->creditMems[rank] + creditOffset +
        static_cast<uint64_t>(firstPeer) * TILEXR_ALLTOALL_GROUP_CREDIT_STRIDE);
    const uint64_t creditWaitBegin = AllToAllGroupTraceCycle(groupTrace);
    uint64_t observedCredit = 0ULL;
    const bool ok = AllToAllGroupWaitCreditMte(
        creditSignal, expectedCredit, TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES,
        relayLocal, observedCredit);
    AllToAllGroupTraceRecordTask(
        groupTrace, traceIteration, blockIdx, 0U, 0U,
        TileXR::Demo::kAllToAllGroupTraceCreditWait,
        groupCount, passCount, firstPeer,
        TileXR::Demo::kAllToAllGroupTraceNoQp,
        creditWaitBegin, AllToAllGroupTraceCycle(groupTrace));
    if (!ok) {
        AllToAllGroupRecordError(
            debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_CREDIT_WAIT,
            groupCount, 0U, firstPeer, 0U, 0U,
            expectedCredit, observedCredit);
    }
    return ok;
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

__aicore__ inline bool AllToAllGroupFlushSimt(
    const __gm__ TileXR::CommArgs* args,
    __ubuf__ TileXR::Demo::AllToAllGroupSimtBatch* batch,
    uint32_t& taskCount, uint32_t& queueCount,
    __gm__ int32_t* debug, uint32_t blockIdx,
    __gm__ uint8_t* trace, uint32_t traceIteration,
    uint32_t groupCount, uint32_t passCount)
{
    if (taskCount == 0U) {
        queueCount = 0U;
        return true;
    }
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    const uint64_t postBegin = AllToAllGroupTraceCycle(trace);
    TileXR::UDMACleanCacheLines(
        reinterpret_cast<__gm__ uint8_t*>(debug) +
            TILEXR_ALLTOALL_GROUP_ERROR_BYTES,
        static_cast<uint64_t>(taskCount) * sizeof(uint64_t));
    TileXR::Demo::AllToAllGroupSimtPost(batch, queueCount, udmaInfo);
    for (uint32_t task = 0U; task < taskCount; ++task) {
        if (batch->active[task] == 0U) {
            continue;
        }
        auto wq = TileXR::UDMAGetWQCtx(
            udmaInfo, batch->peer[task], batch->qpIdx[task]);
        const uint32_t wqeSize = 1U << wq->baseBkShift;
        auto wqeAddr = reinterpret_cast<__gm__ uint8_t*>(
            wq->bufAddr + static_cast<uint64_t>(wqeSize) *
                (batch->reservedHead[task] % wq->depth));
        TileXR::UDMACleanCacheLines(
            wqeAddr, wqeSize * TileXR::Demo::kAllToAllGroupSimtWqesPerTask);
    }
    for (uint32_t queue = 0U; queue < queueCount; ++queue) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        auto wq = TileXR::UDMAGetWQCtx(
            udmaInfo, batch->queuePeer[queue], batch->queueQpIdx[queue]);
        st_dev(batch->queueHead[queue],
            reinterpret_cast<__gm__ uint32_t*>(wq->dbAddr), 0);
        st_dev(batch->queueHead[queue],
            reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        st_dev(batch->queueExpectedCount[queue],
            reinterpret_cast<__gm__ uint32_t*>(wq->wqeCntAddr), 0);
    }
    const uint64_t postEnd = AllToAllGroupTraceCycle(trace);
    for (uint32_t task = 0U; task < taskCount; ++task) {
        if (batch->active[task] != 0U) {
            AllToAllGroupTraceRecordSimtTask(
                trace, traceIteration, batch, task,
                TileXR::Demo::kAllToAllGroupTraceSendPutSignal,
                groupCount, passCount, postBegin, postEnd);
        }
    }
    const uint64_t quietBegin = AllToAllGroupTraceCycle(trace);
    TileXR::Demo::AllToAllGroupSimtQuiet(batch, queueCount, udmaInfo);
    const uint64_t quietEnd = AllToAllGroupTraceCycle(trace);
    for (uint32_t queue = 0U; queue < queueCount; ++queue) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        const uint32_t taskBegin = batch->queueTaskBegin[queue];
        const uint32_t queueTasks = batch->queueTaskCount[queue];
        for (uint32_t offset = 0U; offset < queueTasks; ++offset) {
            const uint32_t task = taskBegin + offset;
            AllToAllGroupTraceRecordSimtTask(
                trace, traceIteration, batch, task,
                TileXR::Demo::kAllToAllGroupTraceSendQuiet,
                groupCount, passCount, quietBegin, quietEnd,
                batch->queuePollCount[queue]);
        }
        if (batch->queueQuietStatus[queue] != 0U) {
            const uint32_t task = taskBegin;
            AllToAllGroupRecordError(
                debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_QUIET,
                batch->group[task], batch->pass[task],
                static_cast<int32_t>(batch->peer[task]), batch->qpIdx[task],
                batch->queueQuietStatus[queue], batch->signal[task], 0ULL);
            return false;
        }
        auto wq = TileXR::UDMAGetWQCtx(
            udmaInfo, batch->queuePeer[queue], batch->queueQpIdx[queue]);
        auto cq = TileXR::UDMAGetSCQCtx(
            udmaInfo, batch->queuePeer[queue], batch->queueQpIdx[queue]);
        TileXR::UDMAPollCQUpdateInfo(batch->queueCompletedTail[queue], cq, wq);
    }
    taskCount = 0U;
    queueCount = 0U;
    return true;
}

__aicore__ inline bool AllToAllGroupFlushSplitSimt(
    const __gm__ TileXR::CommArgs* args,
    __ubuf__ TileXR::Demo::AllToAllGroupSimtBatch* batch,
    uint32_t& taskCount, uint32_t& queueCount,
    __gm__ int32_t* debug, uint32_t blockIdx,
    __gm__ uint8_t* trace, uint32_t traceIteration,
    uint32_t groupCount, uint32_t passCount,
    uint32_t postPhase, bool quietAfterPost, bool resetBatch,
    uint64_t aggregatePostBegin)
{
    if (taskCount == 0U) {
        queueCount = 0U;
        return true;
    }
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    if (postPhase != TileXR::Demo::kAllToAllGroupSimtPostPayload) {
        TileXR::UDMACleanCacheLines(
            reinterpret_cast<__gm__ uint8_t*>(debug) +
                TILEXR_ALLTOALL_GROUP_ERROR_BYTES,
            static_cast<uint64_t>(taskCount) * sizeof(uint64_t));
    }
    if (postPhase == TileXR::Demo::kAllToAllGroupSimtPostPayload) {
        TileXR::Demo::AllToAllGroupSimtPostPayload(batch, queueCount, udmaInfo);
    } else {
        TileXR::Demo::AllToAllGroupSimtPostSignal(batch, queueCount, udmaInfo);
    }
    for (uint32_t task = 0U; task < taskCount; ++task) {
        if (batch->active[task] == 0U) {
            continue;
        }
        auto wq = TileXR::UDMAGetWQCtx(
            udmaInfo, batch->peer[task], batch->qpIdx[task]);
        const uint32_t wqeSize = 1U << wq->baseBkShift;
        auto wqeAddr = reinterpret_cast<__gm__ uint8_t*>(
            wq->bufAddr + static_cast<uint64_t>(wqeSize) *
                (batch->reservedHead[task] % wq->depth));
        TileXR::UDMACleanCacheLines(
            wqeAddr, wqeSize *
                TileXR::Demo::AllToAllGroupSimtWqesForPhase(postPhase));
    }
    for (uint32_t queue = 0U; queue < queueCount; ++queue) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        auto wq = TileXR::UDMAGetWQCtx(
            udmaInfo, batch->queuePeer[queue], batch->queueQpIdx[queue]);
        st_dev(batch->queueHead[queue],
            reinterpret_cast<__gm__ uint32_t*>(wq->dbAddr), 0);
        st_dev(batch->queueHead[queue],
            reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        st_dev(batch->queueExpectedCount[queue],
            reinterpret_cast<__gm__ uint32_t*>(wq->wqeCntAddr), 0);
    }
    const uint64_t postEnd = AllToAllGroupTraceCycle(trace);
    if (!quietAfterPost) {
        return true;
    }
    for (uint32_t task = 0U; task < taskCount; ++task) {
        if (batch->active[task] != 0U) {
            AllToAllGroupTraceRecordSimtTask(
                trace, traceIteration, batch, task,
                TileXR::Demo::kAllToAllGroupTraceSendPutSignal,
                groupCount, passCount, aggregatePostBegin, postEnd);
        }
    }
    const uint64_t quietBegin = AllToAllGroupTraceCycle(trace);
    TileXR::Demo::AllToAllGroupSimtQuiet(batch, queueCount, udmaInfo);
    const uint64_t quietEnd = AllToAllGroupTraceCycle(trace);
    for (uint32_t queue = 0U; queue < queueCount; ++queue) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        const uint32_t taskBegin = batch->queueTaskBegin[queue];
        const uint32_t queueTasks = batch->queueTaskCount[queue];
        for (uint32_t offset = 0U; offset < queueTasks; ++offset) {
            const uint32_t task = taskBegin + offset;
            AllToAllGroupTraceRecordSimtTask(
                trace, traceIteration, batch, task,
                TileXR::Demo::kAllToAllGroupTraceSendQuiet,
                groupCount, passCount, quietBegin, quietEnd,
                batch->queuePollCount[queue]);
        }
        if (batch->queueQuietStatus[queue] != 0U) {
            const uint32_t task = taskBegin;
            AllToAllGroupRecordError(
                debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_QUIET,
                batch->group[task], batch->pass[task],
                static_cast<int32_t>(batch->peer[task]), batch->qpIdx[task],
                batch->queueQuietStatus[queue], batch->signal[task], 0ULL);
            return false;
        }
        auto wq = TileXR::UDMAGetWQCtx(
            udmaInfo, batch->queuePeer[queue], batch->queueQpIdx[queue]);
        auto cq = TileXR::UDMAGetSCQCtx(
            udmaInfo, batch->queuePeer[queue], batch->queueQpIdx[queue]);
        TileXR::UDMAPollCQUpdateInfo(batch->queueCompletedTail[queue], cq, wq);
    }
    if (resetBatch) {
        taskCount = 0U;
        queueCount = 0U;
    }
    return true;
}

template <bool IngressCredit>
__aicore__ inline bool AllToAllGroupRunSimtSend(
    const __gm__ TileXR::CommArgs* args, __gm__ int32_t* input,
    __gm__ TileXR::TileXRUDMARegistry* registry, __gm__ int32_t* debug,
    __ubuf__ TileXR::Demo::AllToAllGroupSimtBatch* batch,
    uint32_t blockIdx, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    const uint64_t* payloadOffsets, const uint64_t* signalOffsets,
    const uint64_t* creditOffsets,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t quietBatch, uint32_t npuCount,
    AscendC::LocalTensor<uint8_t> relayLocal,
    __gm__ uint8_t* trace, uint32_t traceIteration)
{
    const uint32_t slot = invocationId & 1U;
    uint32_t taskCount = 0U;
    uint32_t queueCount = 0U;
    const uint32_t batchLimit =
        quietBatch < TileXR::Demo::kAllToAllGroupSimtMaxTasks ?
        quietBatch : TileXR::Demo::kAllToAllGroupSimtMaxTasks;

    for (uint32_t group = 0U; group < groupCount; ++group) {
        if constexpr (IngressCredit) {
            if (group != 0U) {
                const uint64_t waitBegin = AllToAllGroupTraceCycle(trace);
                const uint64_t expectedCredit =
                    AllToAllGroupDeviceToken(invocationId, group, 0U);
                int32_t failedPeer = -1;
                uint64_t observedCredit = expectedCredit;
                auto localCreditBase = args->creditMems[args->rank];
                for (uint32_t lane = 0U; lane < groupWidth; ++lane) {
                    const int32_t peer = AllToAllGroupDevicePeer(
                        args->rank, args->rankSize, group, lane, groupWidth);
                    if (peer < 0) {
                        continue;
                    }
                    auto credit = reinterpret_cast<__gm__ uint64_t*>(
                        localCreditBase + creditOffsets[slot] +
                        static_cast<uint64_t>(peer) *
                            TILEXR_ALLTOALL_GROUP_CREDIT_STRIDE);
                    if (!AllToAllGroupWaitCreditMte(
                            credit, expectedCredit,
                            TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES,
                            relayLocal, observedCredit)) {
                        failedPeer = peer;
                        break;
                    }
                }
                AllToAllGroupTraceRecordTask(
                    trace, traceIteration, blockIdx, group, 0U,
                    TileXR::Demo::kAllToAllGroupTraceCreditWait,
                    groupCount, passCount, failedPeer,
                    TileXR::Demo::kAllToAllGroupTraceNoQp,
                    waitBegin, AllToAllGroupTraceCycle(trace));
                if (failedPeer >= 0) {
                    AllToAllGroupRecordError(
                        debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_CREDIT_WAIT,
                        group, 0U, failedPeer, 0U, 0U,
                        expectedCredit, observedCredit);
                    return false;
                }
            }
        }

        for (uint32_t pass = 0U; pass < passCount; ++pass) {
            const int32_t chunkElementOffset =
                static_cast<int32_t>(pass) * chunkElements;
            const int32_t remaining = elementsPerPeer - chunkElementOffset;
            if (remaining <= 0) {
                continue;
            }
            const int32_t currentElements =
                remaining < chunkElements ? remaining : chunkElements;
            for (uint32_t lane = 0U; lane < groupWidth; ++lane) {
                const int32_t peer = AllToAllGroupDevicePeer(
                    args->rank, args->rankSize, group, lane, groupWidth);
                if (!AllToAllGroupCoResidentPeerDevice(
                        args->rank, peer, args->localRankSize, npuCount) ||
                    !AllToAllGroupPeerInRouteStageDevice(
                        args->rank, peer, routeStage)) {
                    continue;
                }
                const uint64_t ipcSendBegin = AllToAllGroupTraceCycle(trace);
                AllToAllGroupSendIpcLoopback(
                    args, input, peer, slot,
                    static_cast<uint64_t>(elementsPerPeer) * sizeof(int32_t),
                    chunkElementOffset, currentElements,
                    AllToAllGroupDeviceToken(invocationId, group, pass),
                    relayLocal);
                AllToAllGroupTraceRecordTask(
                    trace, traceIteration, blockIdx, group, pass,
                    TileXR::Demo::kAllToAllGroupTraceSendPutSignal,
                    groupCount, passCount, peer,
                    TileXR::Demo::kAllToAllGroupTraceNoQp,
                    ipcSendBegin, AllToAllGroupTraceCycle(trace));
            }
            for (uint32_t workerBegin = 0U;
                 workerBegin < TILEXR_ALLTOALL_GROUP_SEND_WORKERS;
                 workerBegin += batchLimit) {
                const uint32_t workerCount =
                    TILEXR_ALLTOALL_GROUP_SEND_WORKERS - workerBegin < batchLimit ?
                    TILEXR_ALLTOALL_GROUP_SEND_WORKERS - workerBegin : batchLimit;
                const uint64_t tokenBase = reinterpret_cast<uint64_t>(
                    reinterpret_cast<__gm__ uint8_t*>(debug) +
                    TILEXR_ALLTOALL_GROUP_ERROR_BYTES);
                TileXR::Demo::AllToAllGroupSimtBuild(
                    batch, workerBegin, workerCount, args, registry, input,
                    tokenBase, invocationId, group, pass, elementsPerPeer,
                    chunkElementOffset, currentElements, payloadOffsets[slot],
                    signalOffsets[slot], routeStage, multiChannel,
                    primaryRouteParts, groupWidth, npuCount);
                for (uint32_t task = 0U; task < workerCount; ++task) {
                    if (batch->configStatus[task] != 0U) {
                        AllToAllGroupRecordError(
                            debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_CONFIG,
                            group, pass, static_cast<int32_t>(batch->peer[task]),
                            batch->qpIdx[task], batch->configStatus[task],
                            batch->configRegionBytes[task],
                            batch->configOffset[task]);
                        return false;
                    }
                }
                taskCount = workerCount;
                queueCount = workerCount;
                const uint64_t aggregatePostBegin =
                    AllToAllGroupTraceCycle(trace);
                if (registry->regionCount > 1U) {
                    if (!AllToAllGroupFlushSplitSimt(
                            args, batch, taskCount, queueCount, debug, blockIdx,
                            trace, traceIteration, groupCount, passCount,
                            TileXR::Demo::kAllToAllGroupSimtPostPayload,
                            false, false, aggregatePostBegin) ||
                        !AllToAllGroupFlushSplitSimt(
                            args, batch, taskCount, queueCount, debug, blockIdx,
                            trace, traceIteration, groupCount, passCount,
                            TileXR::Demo::kAllToAllGroupSimtPostSignal,
                            true, true, aggregatePostBegin)) {
                        return false;
                    }
                } else if (!AllToAllGroupFlushSimt(
                               args, batch, taskCount, queueCount, debug,
                               blockIdx, trace, traceIteration, groupCount,
                               passCount)) {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

template <bool IngressCredit>
__aicore__ inline void AllToAllGroupTerminalBarrier()
{
    if constexpr (IngressCredit) {
        // Every block must reach the terminal barrier exactly once, including
        // error paths, otherwise a peer block remains blocked in SyncAll.
        AscendC::SyncAll();
    }
}

template <bool BatchQuiet, bool IngressCredit>
__aicore__ inline void AllToAllGroupKernelImpl(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    uint64_t creditOffset0, uint64_t creditOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t simtMode, uint32_t groupWidth, uint32_t quietBatch,
    uint32_t prewarmSq, uint32_t npuCount)
{
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    auto groupTrace = blockIdx < TileXR::Demo::kAllToAllGroupTraceCoreCount ?
        reinterpret_cast<__gm__ uint8_t*>(groupTraceGM) : nullptr;
    const uint64_t kernelBegin = AllToAllGroupTraceCycle(groupTrace);
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    const uint32_t copyoutWorkers = TileXR::SDMAEnabled(args) ? 1U : 32U;
    const uint32_t sendWorkers = simtMode != 0U ?
        TILEXR_ALLTOALL_GROUP_SIMT_SEND_WORKERS :
        TILEXR_ALLTOALL_GROUP_SEND_WORKERS;
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto registeredMemory = reinterpret_cast<__gm__ uint8_t*>(registeredMemoryGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);
    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;

    if ((copyoutWorkers != 1U && copyoutWorkers != 8U && copyoutWorkers != 16U &&
            copyoutWorkers != 32U && copyoutWorkers != 48U) ||
        routeStage > TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_NO_COPY ||
        multiChannel > 1U || simtMode > 1U ||
        npuCount == 0U ||
        (args->localRankSize > static_cast<int32_t>(npuCount) && simtMode == 0U) ||
        (groupWidth != TILEXR_ALLTOALL_GROUP_DEFAULT_WIDTH &&
            groupWidth != TILEXR_ALLTOALL_GROUP_EXPERIMENTAL_WIDTH) ||
        quietBatch == 0U ||
        quietBatch > TILEXR_ALLTOALL_GROUP_MAX_QUIET_BATCH ||
        (quietBatch & (quietBatch - 1U)) != 0U ||
        prewarmSq > 1U ||
        (prewarmSq != 0U && copyoutWorkers != 1U) ||
        (primaryRouteParts > TileXR::Demo::kAllToAllGroupRouteParts &&
            primaryRouteParts != TileXR::Demo::kAllToAllGroupAutoPrimaryParts) ||
        sendWorkers + copyoutWorkers >
            TILEXR_ALLTOALL_GROUP_BLOCK_DIM ||
        blockIdx >= sendWorkers + copyoutWorkers ||
        !TileXR::UDMARegistryEnabled(args) || rankSize < 8 ||
        rankSize > TileXR::TILEXR_MAX_RANK_SIZE || (rankSize & 7) != 0 ||
        elementsPerPeer <= 0 || chunkElements <= 0 || passCount == 0U ||
        groupCount == 0U || groupCount != static_cast<uint32_t>(
            (rankSize - 1 + static_cast<int32_t>(groupWidth) - 1) /
            static_cast<int32_t>(groupWidth)) ||
        (IngressCredit &&
            (debug == nullptr ||
             groupWidth != TILEXR_ALLTOALL_GROUP_DEFAULT_WIDTH ||
             passCount != 1U ||
             routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_COMBINED))) {
        AllToAllGroupRecordError(debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_CONFIG,
            0U, 0U, -1, 0U, 0U, 0ULL, 0ULL);
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        AllToAllGroupTerminalBarrier<IngressCredit>();
        return;
    }

    if constexpr (IngressCredit) {
        auto priorError = reinterpret_cast<__gm__ AllToAllGroupDeviceError*>(
            debug + static_cast<uint64_t>(blockIdx) *
                TILEXR_ALLTOALL_GROUP_ERROR_WORDS);
        if (priorError->valid != 0U) {
            AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                kernelBegin, AllToAllGroupTraceCycle(groupTrace));
            AllToAllGroupTerminalBarrier<IngressCredit>();
            return;
        }
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
    pipe.InitBuffer(relayTBuf, TILEXR_ALLTOALL_GROUP_RELAY_BYTES);
    AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();

    const uint32_t slot = invocationId & 1U;
    const uint64_t payloadOffsets[2] = {payloadOffset0, payloadOffset1};
    const uint64_t signalOffsets[2] = {signalOffset0, signalOffset1};
    const uint64_t creditOffsets[2] = {creditOffset0, creditOffset1};
    const uint64_t bytesPerPeer =
        static_cast<uint64_t>(elementsPerPeer) * sizeof(int32_t);

    if (blockIdx >= sendWorkers) {
        if (!AllToAllGroupStageRunsReceiveDevice(routeStage)) {
            AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                kernelBegin, AllToAllGroupTraceCycle(groupTrace));
            AllToAllGroupTerminalBarrier<IngressCredit>();
            return;
        }
        const uint32_t worker = blockIdx - sendWorkers;
        if (prewarmSq != 0U && worker == 0U &&
            !TileXR::SDMAPrewarmSqPages(args, groupWidth, relayLocal)) {
            AllToAllGroupRecordError(
                debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_SDMA_PREWARM,
                0U, 0U, -1, 0U, 0U, groupWidth, 0ULL);
            AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                kernelBegin, AllToAllGroupTraceCycle(groupTrace));
            AllToAllGroupTerminalBarrier<IngressCredit>();
            return;
        }
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

        const bool readyDrivenCopyout = copyoutWorkers == 1U &&
            TileXR::SDMAEnabled(args) &&
            routeStage == TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_COMBINED;
        const uint32_t schedulerPassCount = readyDrivenCopyout ? passCount : 1U;
        for (uint32_t schedulerPass = 0U;
             schedulerPass < schedulerPassCount; ++schedulerPass) {
            uint64_t completedTasks[TILEXR_ALLTOALL_GROUP_READY_BITMAP_WORDS];
            for (uint32_t word = 0U;
                 word < TILEXR_ALLTOALL_GROUP_READY_BITMAP_WORDS; ++word) {
                completedTasks[word] = 0ULL;
            }
            uint64_t noProgressBegin =
                static_cast<uint64_t>(AscendC::GetSystemCycle());
            bool schedulerDone = false;
            while (!schedulerDone) {
                bool pendingTask = false;
                bool madeProgress = false;
                uint32_t lastPendingGroup = 0U;
                uint32_t lastPendingPass = schedulerPass;
                int32_t lastPendingPeer = -1;
                uint64_t lastExpectedToken = 0ULL;
                uint64_t lastObservedToken = 0ULL;
        for (uint32_t group = 0U; group < groupCount; ++group) {
            for (uint32_t assignment = 0U; ; ++assignment) {
                const int32_t laneValue = AllToAllGroupCopyoutLaneDevice(
                    worker, assignment, copyoutWorkers);
                if (laneValue < 0) {
                    break;
                }
                const uint32_t lane = static_cast<uint32_t>(laneValue);
            const uint32_t taskIndex = group * groupWidth + lane;
            const uint32_t taskWord = taskIndex / 64U;
            const uint64_t taskMask = 1ULL << (taskIndex % 64U);
            if (readyDrivenCopyout && (completedTasks[taskWord] & taskMask) != 0ULL) {
                continue;
            }
            const int32_t peer = AllToAllGroupDevicePeer(
                rank, rankSize, group, lane, groupWidth);
            if (peer < 0) {
                continue;
            }
            if (!AllToAllGroupReceivePeerInRouteStageDevice(rank, peer, routeStage)) {
                continue;
            }
            const bool coResident = AllToAllGroupCoResidentPeerDevice(
                rank, peer, args->localRankSize, npuCount);
            const bool crossNode =
                rank / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode) !=
                peer / static_cast<int32_t>(TileXR::Demo::kAllToAllGroupRanksPerNode);
            const bool remoteAssist =
                AllToAllGroupRemoteAssistDevice(worker, copyoutWorkers);
            if (coResident && remoteAssist) {
                continue;
            }
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
            if (coResident) {
                primaryTotalElements = static_cast<uint32_t>(elementsPerPeer);
                secondaryTotalElements = 0U;
            }
            const uint32_t traceCore = copyoutWorkers < TILEXR_ALLTOALL_GROUP_SEND_CORES ?
                sendWorkers + lane : blockIdx;
            const uint32_t passBegin = readyDrivenCopyout ? schedulerPass : 0U;
            const uint32_t passEnd = readyDrivenCopyout ? schedulerPass + 1U : passCount;
            for (uint32_t pass = passBegin; pass < passEnd; ++pass) {
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
                if (readyDrivenCopyout) {
                    pendingTask = true;
                }
                const uint64_t expectedToken =
                    AllToAllGroupDeviceToken(invocationId, group, pass);
                auto primarySignal = reinterpret_cast<__gm__ uint64_t*>(
                    registeredMemory + signalOffsets[slot] +
                    static_cast<uint64_t>(peer) * TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE);
                auto secondarySignal = reinterpret_cast<__gm__ uint64_t*>(
                    reinterpret_cast<__gm__ uint8_t*>(primarySignal) +
                    TILEXR_ALLTOALL_GROUP_ROUTE_SIGNAL_STRIDE);
                auto ipcSignal = reinterpret_cast<__gm__ uint64_t*>(
                    args->peerMems[rank] +
                    AllToAllGroupIpcSignalOffsetDevice(slot, rankSize, peer));
                if (AllToAllGroupStageWaitsForSignalDevice(routeStage)) {
                    uint64_t observed = 0ULL;
                    const uint64_t waitBegin = AllToAllGroupTraceCycle(groupTrace);
                    const bool waitPrimary = primaryElements != 0U &&
                        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_SECONDARY;
                    const bool waitSecondary = secondaryElements != 0U &&
                        routeStage != TILEXR_ALLTOALL_GROUP_ROUTE_STAGE_PRIMARY;
                    const bool ready = coResident ?
                        (readyDrivenCopyout ?
                            AllToAllGroupLoadTokenMte(ipcSignal, relayLocal) >=
                                expectedToken :
                            AllToAllGroupWaitTokenMte(
                                ipcSignal, expectedToken,
                                TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES,
                                relayLocal, observed)) :
                        (readyDrivenCopyout ?
                            AllToAllGroupRouteTokensReadyMte(
                                primarySignal, secondarySignal, waitPrimary, waitSecondary,
                                expectedToken, relayLocal, observed) :
                            AllToAllGroupWaitRouteTokensMte(
                                primarySignal, secondarySignal, waitPrimary, waitSecondary,
                                expectedToken, TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES,
                                relayLocal, observed));
                    if (coResident && readyDrivenCopyout) {
                        observed = ready ? expectedToken : 0ULL;
                    }
                    if (!ready && readyDrivenCopyout) {
                        lastPendingGroup = group;
                        lastPendingPass = pass;
                        lastPendingPeer = peer;
                        lastExpectedToken = expectedToken;
                        lastObservedToken = observed;
                        continue;
                    }
                    if (!ready) {
                        AllToAllGroupTraceRecordTask(
                            groupTrace, traceIteration, traceCore, group, pass,
                            TileXR::Demo::kAllToAllGroupTraceReceiveWait, groupCount, passCount,
                            peer, TileXR::Demo::kAllToAllGroupTraceNoQp,
                            waitBegin, AllToAllGroupTraceCycle(groupTrace));
                        AllToAllGroupRecordError(debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_WAIT,
                            group, pass, peer, 0U, 0U, expectedToken, observed);
                        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
                        AllToAllGroupTerminalBarrier<IngressCredit>();
                        return;
                    }
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceReceiveWait, groupCount, passCount,
                        peer, TileXR::Demo::kAllToAllGroupTraceNoQp,
                        waitBegin, AllToAllGroupTraceCycle(groupTrace));
                }
                if constexpr (IngressCredit) {
                    if (AllToAllGroupCreditOwnerDevice(worker) &&
                        pass + 1U == passCount) {
                        AllToAllGroupPublishNextCredit(
                            args, rank, rankSize, invocationId, group, lane,
                            groupCount, groupWidth, creditOffsets[slot], relayLocal);
                    }
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
                auto relaySrc = coResident ?
                    args->peerMems[rank] +
                        AllToAllGroupIpcPayloadOffsetDevice(
                            slot, rankSize, bytesPerPeer, peer) +
                        static_cast<uint64_t>(
                            chunkElementOffset + copyElementBegin) * sizeof(int32_t) :
                    registeredMemory + payloadOffsets[slot] +
                        static_cast<uint64_t>(peer) * bytesPerPeer +
                        static_cast<uint64_t>(
                            chunkElementOffset + copyElementBegin) * sizeof(int32_t);
                auto relayDst = reinterpret_cast<__gm__ uint8_t*>(
                    output + static_cast<uint64_t>(peer) * elementsPerPeer +
                    chunkElementOffset + copyElementBegin);
                const uint64_t receiveCopyBegin = AllToAllGroupTraceCycle(groupTrace);
                uint64_t sdmaEvent = 0ULL;
                TileXR::SDMASubmitTrace sdmaTrace {};
                const uint64_t sdmaSubmitBegin = AllToAllGroupTraceCycle(groupTrace);
                const uint32_t sdmaSubmitStatus = AllToAllGroupCopySdmaSubmit(
                    args, relayDst, relaySrc, copyBytes, lane, sdmaEvent, sdmaTrace);
                const uint64_t sdmaSubmitEnd = AllToAllGroupTraceCycle(groupTrace);
                if (sdmaSubmitStatus != TILEXR_ALLTOALL_GROUP_SDMA_FALLBACK) {
                    const uint64_t sdmaWaitBegin = AllToAllGroupTraceCycle(groupTrace);
                    const uint32_t sdmaStatus = AllToAllGroupCopySdmaWait(
                        args, sdmaEvent, lane);
                    const uint64_t sdmaWaitEnd = AllToAllGroupTraceCycle(groupTrace);
                    const uint64_t receiveCopyEnd = AllToAllGroupTraceCycle(groupTrace);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceReceiveCopy,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        receiveCopyBegin, receiveCopyEnd);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceSdmaSubmit,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        sdmaSubmitBegin, sdmaSubmitEnd,
                        sdmaTrace.head, sdmaTrace.tail, sdmaTrace.newTail,
                        sdmaTrace.depth, sdmaTrace.generation);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceSdmaWait,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        sdmaWaitBegin, sdmaWaitEnd);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceSdmaPrepare,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        sdmaTrace.prepareBegin, sdmaTrace.prepareEnd,
                        sdmaTrace.head, sdmaTrace.tail, sdmaTrace.newTail,
                        sdmaTrace.depth, sdmaTrace.generation);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceSdmaCacheClean,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        sdmaTrace.cacheCleanBegin, sdmaTrace.cacheCleanEnd,
                        sdmaTrace.head, sdmaTrace.tail, sdmaTrace.newTail,
                        sdmaTrace.depth, sdmaTrace.generation);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceSdmaDsb,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        sdmaTrace.dsbBegin, sdmaTrace.dsbEnd,
                        sdmaTrace.head, sdmaTrace.tail, sdmaTrace.newTail,
                        sdmaTrace.depth, sdmaTrace.generation);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceSdmaDoorbell,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        sdmaTrace.doorbellBegin, sdmaTrace.doorbellEnd,
                        sdmaTrace.head, sdmaTrace.tail, sdmaTrace.newTail,
                        sdmaTrace.depth, sdmaTrace.generation);
                    if (sdmaStatus == TILEXR_ALLTOALL_GROUP_SDMA_FAILED) {
                        AllToAllGroupRecordError(
                            debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_SDMA,
                            group, pass, peer, worker, 0U,
                            static_cast<uint64_t>(worker), sdmaEvent);
                        AllToAllGroupTraceRecordKernel(
                            groupTrace, traceIteration, blockIdx, kernelBegin,
                            AllToAllGroupTraceCycle(groupTrace));
                        AllToAllGroupTerminalBarrier<IngressCredit>();
                        return;
                    }
                } else {
                    AllToAllGroupCopyMte(relayDst, relaySrc, copyBytes, relayLocal);
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, traceCore, group, pass,
                        TileXR::Demo::kAllToAllGroupTraceReceiveCopy,
                        groupCount, passCount, peer,
                        TileXR::Demo::kAllToAllGroupTraceNoQp,
                        receiveCopyBegin, AllToAllGroupTraceCycle(groupTrace));
                }
                if (readyDrivenCopyout) {
                    completedTasks[taskWord] |= taskMask;
                    madeProgress = true;
                }
            }
            }
        }
                if (!readyDrivenCopyout || !pendingTask) {
                    schedulerDone = true;
                    continue;
                }
                if (madeProgress) {
                    noProgressBegin =
                        static_cast<uint64_t>(AscendC::GetSystemCycle());
                    continue;
                }
                if (static_cast<uint64_t>(AscendC::GetSystemCycle()) -
                        noProgressBegin >= TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES) {
                    AllToAllGroupRecordError(
                        debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_WAIT,
                        lastPendingGroup, lastPendingPass, lastPendingPeer,
                        0U, 0U, lastExpectedToken, lastObservedToken);
                    AllToAllGroupTraceRecordKernel(
                        groupTrace, traceIteration, blockIdx, kernelBegin,
                        AllToAllGroupTraceCycle(groupTrace));
                    AllToAllGroupTerminalBarrier<IngressCredit>();
                    return;
                }
            }
        }
        if constexpr (IngressCredit) {
            AllToAllGroupTerminalBarrier<IngressCredit>();
            AllToAllGroupPublishTerminalCredits(
                args, rank, rankSize, invocationId, worker, copyoutWorkers,
                groupCount, groupWidth, creditOffsets[slot], relayLocal);
        }
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        return;
    }

    if (!AllToAllGroupStageRunsSendDevice(routeStage)) {
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        AllToAllGroupTerminalBarrier<IngressCredit>();
        return;
    }
    if (simtMode != 0U) {
        auto registry = TileXR::GetUDMARegistry(args);
        auto simtBatch = reinterpret_cast<__ubuf__
            TileXR::Demo::AllToAllGroupSimtBatch*>(relayLocal.GetPhyAddr());
        if (!AllToAllGroupRunSimtSend<IngressCredit>(
                args, input, registry, debug, simtBatch, blockIdx,
                invocationId, elementsPerPeer, chunkElements,
                passCount, groupCount, payloadOffsets, signalOffsets,
                creditOffsets, routeStage, multiChannel, primaryRouteParts,
                groupWidth, quietBatch, npuCount, relayLocal,
                groupTrace, traceIteration)) {
            AllToAllGroupTraceRecordKernel(
                groupTrace, traceIteration, blockIdx, kernelBegin,
                AllToAllGroupTraceCycle(groupTrace));
            AllToAllGroupTerminalBarrier<IngressCredit>();
            return;
        }
        if constexpr (IngressCredit) {
            AllToAllGroupTerminalBarrier<IngressCredit>();
            for (uint32_t lane = 0U; lane < groupWidth; ++lane) {
                if (!AllToAllGroupWaitTerminalCredit(
                        args, rank, rankSize, invocationId, lane,
                        groupCount, passCount, groupWidth,
                        creditOffsets[slot], relayLocal, debug, blockIdx,
                        groupTrace, traceIteration)) {
                    AllToAllGroupTraceRecordKernel(
                        groupTrace, traceIteration, blockIdx, kernelBegin,
                        AllToAllGroupTraceCycle(groupTrace));
                    return;
                }
            }
        }
        AllToAllGroupTraceRecordKernel(
            groupTrace, traceIteration, blockIdx, kernelBegin,
            AllToAllGroupTraceCycle(groupTrace));
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
        if (AllToAllGroupCoResidentPeerDevice(
                rank, peer, args->localRankSize, npuCount)) {
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
        if constexpr (IngressCredit) {
            const uint32_t routeElements = workerRoute == 0U ?
                primaryTotalElements : secondaryTotalElements;
            if (group != 0U) {
                const uint64_t expectedCredit =
                    AllToAllGroupDeviceToken(invocationId, group, 0U);
                const uint64_t creditWaitBegin =
                    AllToAllGroupTraceCycle(groupTrace);
                if (routeElements != 0U) {
                    auto creditSignal = reinterpret_cast<__gm__ uint64_t*>(
                        args->creditMems[rank] + creditOffsets[slot] +
                        static_cast<uint64_t>(peer) *
                            TILEXR_ALLTOALL_GROUP_CREDIT_STRIDE);
                    uint64_t observedCredit = 0ULL;
                    if (!AllToAllGroupWaitCreditMte(
                            creditSignal, expectedCredit,
                            TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES,
                            relayLocal, observedCredit)) {
                        AllToAllGroupRecordError(debug, blockIdx,
                            TILEXR_ALLTOALL_GROUP_STAGE_CREDIT_WAIT, group, 0U,
                            peer, workerRoute, 0U, expectedCredit,
                            observedCredit);
                        AllToAllGroupTraceRecordTask(
                            groupTrace, traceIteration, blockIdx, group, 0U,
                            TileXR::Demo::kAllToAllGroupTraceCreditWait,
                            groupCount, passCount, peer,
                            TileXR::Demo::kAllToAllGroupTraceNoQp,
                            creditWaitBegin,
                            AllToAllGroupTraceCycle(groupTrace));
                        AllToAllGroupTraceRecordKernel(
                            groupTrace, traceIteration, blockIdx, kernelBegin,
                            AllToAllGroupTraceCycle(groupTrace));
                        AllToAllGroupTerminalBarrier<IngressCredit>();
                        return;
                    }
                }
                AllToAllGroupTraceRecordTask(
                    groupTrace, traceIteration, blockIdx, group, 0U,
                    TileXR::Demo::kAllToAllGroupTraceCreditWait,
                    groupCount, passCount, peer,
                    TileXR::Demo::kAllToAllGroupTraceNoQp,
                    creditWaitBegin, AllToAllGroupTraceCycle(groupTrace));
            }
            if (routeElements == 0U) {
                continue;
            }
        }
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
                    AllToAllGroupTerminalBarrier<IngressCredit>();
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
        AllToAllGroupTerminalBarrier<IngressCredit>();
        return;
    }
    if constexpr (IngressCredit) {
        AllToAllGroupTerminalBarrier<IngressCredit>();
        if (workerRoute == 0U && !AllToAllGroupWaitTerminalCredit(
                args, rank, rankSize, invocationId, lane, groupCount, passCount,
                groupWidth, creditOffsets[slot], relayLocal,
                debug, blockIdx, groupTrace, traceIteration)) {
            AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                kernelBegin, AllToAllGroupTraceCycle(groupTrace));
            return;
        }
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
    uint32_t simtMode, uint32_t groupWidth, uint32_t quietBatch,
    uint32_t prewarmSq, uint32_t npuCount)
{
    AllToAllGroupKernelImpl<false, false>(
        commArgsGM, inputGM, outputGM, registeredMemoryGM, debugGM, invocationId,
        elementsPerPeer, chunkElements, passCount, groupCount,
        payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
        0ULL, 0ULL,
        groupTraceGM, traceIteration, routeStage, multiChannel, primaryRouteParts,
        simtMode, groupWidth, quietBatch, prewarmSq, npuCount);
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
    uint32_t simtMode, uint32_t groupWidth, uint32_t quietBatch,
    uint32_t prewarmSq, uint32_t npuCount)
{
    AllToAllGroupKernelImpl<true, false>(
        commArgsGM, inputGM, outputGM, registeredMemoryGM, debugGM, invocationId,
        elementsPerPeer, chunkElements, passCount, groupCount,
        payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
        0ULL, 0ULL,
        groupTraceGM, traceIteration, routeStage, multiChannel, primaryRouteParts,
        simtMode, groupWidth, quietBatch, prewarmSq, npuCount);
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_group_credit_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    uint64_t creditOffset0, uint64_t creditOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t simtMode, uint32_t groupWidth, uint32_t quietBatch,
    uint32_t ingressWindow, uint32_t prewarmSq, uint32_t npuCount)
{
    AllToAllGroupKernelImpl<false, true>(
        commArgsGM, inputGM, outputGM, registeredMemoryGM, debugGM, invocationId,
        elementsPerPeer, chunkElements, passCount, groupCount,
        payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
        creditOffset0, creditOffset1,
        groupTraceGM, traceIteration, routeStage, multiChannel, primaryRouteParts,
        simtMode, groupWidth, quietBatch, prewarmSq, npuCount);
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_group_batch_credit_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    uint64_t creditOffset0, uint64_t creditOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t simtMode, uint32_t groupWidth, uint32_t quietBatch,
    uint32_t ingressWindow, uint32_t prewarmSq, uint32_t npuCount)
{
    AllToAllGroupKernelImpl<true, true>(
        commArgsGM, inputGM, outputGM, registeredMemoryGM, debugGM, invocationId,
        elementsPerPeer, chunkElements, passCount, groupCount,
        payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
        creditOffset0, creditOffset1,
        groupTraceGM, traceIteration, routeStage, multiChannel, primaryRouteParts,
        simtMode, groupWidth, quietBatch, prewarmSq, npuCount);
}

#include "tilexr_udma_alltoall_group_launcher.cpp"
