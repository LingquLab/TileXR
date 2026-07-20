/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_udma.h"
#include "tilexr_udma_alltoall_group_trace.h"

namespace {

constexpr uint32_t TILEXR_ALLTOALL_GROUP_SEND_CORES = 16U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_BLOCK_DIM = 32U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_HALF_WIDTH = 8U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE = 128U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_RELAY_BYTES = 64U * 1024U;
constexpr uint64_t TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES = 10000000000ULL;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_ERROR_WORDS = 12U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_CONFIG = 1U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_QUIET = 2U;
constexpr uint32_t TILEXR_ALLTOALL_GROUP_STAGE_WAIT = 3U;

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
    int32_t rank, int32_t rankSize, uint32_t group, uint32_t lane)
{
    if (rankSize < 8 || rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        (rankSize & 7) != 0 || rank < 0 || rank >= rankSize || lane >= 16U) {
        return -1;
    }
    const uint32_t index = lane < TILEXR_ALLTOALL_GROUP_HALF_WIDTH ?
        lane : lane - TILEXR_ALLTOALL_GROUP_HALF_WIDTH;
    const int32_t distance = static_cast<int32_t>(
        group * TILEXR_ALLTOALL_GROUP_HALF_WIDTH + index + 1U);
    const int32_t diameter = rankSize / 2;
    if (distance > diameter ||
        (lane >= TILEXR_ALLTOALL_GROUP_HALF_WIDTH && distance == diameter)) {
        return -1;
    }
    return lane < TILEXR_ALLTOALL_GROUP_HALF_WIDTH ?
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

__aicore__ inline uint32_t AllToAllGroupSelectMaxWeightQp(
    const __gm__ TileXR::CommArgs* args, int32_t peer)
{
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    const uint32_t qpCount = udmaInfo->qpNum == 0U ? 1U : udmaInfo->qpNum;
    uint32_t selected = 0U;
    uint32_t selectedWeight = TileXR::UDMAGetQpWeight(udmaInfo, peer, 0U);
    for (uint32_t qpIdx = 1U; qpIdx < qpCount; ++qpIdx) {
        const uint32_t weight = TileXR::UDMAGetQpWeight(udmaInfo, peer, qpIdx);
        if (weight > selectedWeight) {
            selected = qpIdx;
            selectedWeight = weight;
        }
    }
    return selected;
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
        sizeof(TileXR::Demo::AllToAllGroupTraceSpan);
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
    const uint64_t index =
        (((((static_cast<uint64_t>(iteration) * TileXR::Demo::kAllToAllGroupTraceCoreCount + core) *
        groupCount + group) * passCount + pass) *
        TileXR::Demo::kAllToAllGroupTracePhaseCount) + phase);
    const uint64_t taskBaseOffset = TileXR::Demo::kAllToAllGroupTraceHeaderBytes +
        static_cast<uint64_t>(TileXR::Demo::kAllToAllGroupTraceMaxIterations) *
        TileXR::Demo::kAllToAllGroupTraceCoreCount *
        sizeof(TileXR::Demo::AllToAllGroupTraceSpan);
    const uint64_t offset = taskBaseOffset +
        index * sizeof(TileXR::Demo::AllToAllGroupTraceTaskSpan);
    auto span = reinterpret_cast<__gm__ TileXR::Demo::AllToAllGroupTraceTaskSpan*>(
        trace + offset);
    span->peer = peer;
    span->qpIdx = qpIdx;
    span->beginCycle = beginCycle;
    span->endCycle = endCycle;
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_group_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration)
{
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

    if (blockIdx >= TILEXR_ALLTOALL_GROUP_BLOCK_DIM ||
        !TileXR::UDMARegistryEnabled(args) || rankSize < 8 ||
        rankSize > TileXR::TILEXR_MAX_RANK_SIZE || (rankSize & 7) != 0 ||
        elementsPerPeer <= 0 || chunkElements <= 0 || passCount == 0U || groupCount == 0U) {
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

    if (blockIdx >= TILEXR_ALLTOALL_GROUP_SEND_CORES) {
        const uint32_t lane = blockIdx - TILEXR_ALLTOALL_GROUP_SEND_CORES;
        const int32_t selfBegin = static_cast<int32_t>(
            static_cast<int64_t>(elementsPerPeer) * lane /
            TILEXR_ALLTOALL_GROUP_SEND_CORES);
        const int32_t selfEnd = static_cast<int32_t>(
            static_cast<int64_t>(elementsPerPeer) * (lane + 1U) /
            TILEXR_ALLTOALL_GROUP_SEND_CORES);
        if (selfEnd > selfBegin) {
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
            const int32_t peer = AllToAllGroupDevicePeer(rank, rankSize, group, lane);
            if (peer < 0) {
                continue;
            }
            for (uint32_t pass = 0U; pass < passCount; ++pass) {
                const int32_t chunkElementOffset = static_cast<int32_t>(pass) * chunkElements;
                const int32_t remaining = elementsPerPeer - chunkElementOffset;
                if (remaining <= 0) {
                    continue;
                }
                const int32_t currentElements = remaining < chunkElements ? remaining : chunkElements;
                const uint32_t chunkBytes =
                    static_cast<uint32_t>(currentElements) * sizeof(int32_t);
                const uint64_t expectedToken =
                    AllToAllGroupDeviceToken(invocationId, group, pass);
                auto signal = reinterpret_cast<__gm__ uint64_t*>(
                    registeredMemory + signalOffsets[slot] +
                    static_cast<uint64_t>(peer) * TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE);
                uint64_t observed = 0ULL;
                const uint64_t waitBegin = AllToAllGroupTraceCycle(groupTrace);
                if (!AllToAllGroupWaitTokenMte(signal, expectedToken,
                        TILEXR_ALLTOALL_GROUP_WAIT_TIMEOUT_CYCLES, relayLocal, observed)) {
                    AllToAllGroupTraceRecordTask(
                        groupTrace, traceIteration, blockIdx, group, pass,
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
                    groupTrace, traceIteration, blockIdx, group, pass,
                    TileXR::Demo::kAllToAllGroupTraceReceiveWait, groupCount, passCount,
                    peer, TileXR::Demo::kAllToAllGroupTraceNoQp,
                    waitBegin, AllToAllGroupTraceCycle(groupTrace));
                auto relaySrc = registeredMemory + payloadOffsets[slot] +
                    static_cast<uint64_t>(peer) * bytesPerPeer +
                    static_cast<uint64_t>(chunkElementOffset) * sizeof(int32_t);
                auto relayDst = reinterpret_cast<__gm__ uint8_t*>(
                    output + static_cast<uint64_t>(peer) * elementsPerPeer + chunkElementOffset);
                const uint64_t receiveCopyBegin = AllToAllGroupTraceCycle(groupTrace);
                AllToAllGroupCopyMte(relayDst, relaySrc, chunkBytes, relayLocal);
                AllToAllGroupTraceRecordTask(
                    groupTrace, traceIteration, blockIdx, group, pass,
                    TileXR::Demo::kAllToAllGroupTraceReceiveCopy, groupCount, passCount,
                    peer, TileXR::Demo::kAllToAllGroupTraceNoQp,
                    receiveCopyBegin, AllToAllGroupTraceCycle(groupTrace));
            }
        }
        AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
            kernelBegin, AllToAllGroupTraceCycle(groupTrace));
        return;
    }

    const uint32_t lane = blockIdx;
    for (uint32_t group = 0U; group < groupCount; ++group) {
        const int32_t peer = AllToAllGroupDevicePeer(rank, rankSize, group, lane);
        if (peer < 0) {
            continue;
        }
        const uint32_t qpIdx = AllToAllGroupSelectMaxWeightQp(args, peer);
        for (uint32_t pass = 0U; pass < passCount; ++pass) {
            const int32_t chunkElementOffset = static_cast<int32_t>(pass) * chunkElements;
            const int32_t remaining = elementsPerPeer - chunkElementOffset;
            if (remaining <= 0) {
                continue;
            }
            const int32_t currentElements = remaining < chunkElements ? remaining : chunkElements;
            const uint32_t chunkBytes =
                static_cast<uint32_t>(currentElements) * sizeof(int32_t);
            const uint64_t chunkByteOffset =
                static_cast<uint64_t>(chunkElementOffset) * sizeof(int32_t);
            const uint64_t expectedToken =
                AllToAllGroupDeviceToken(invocationId, group, pass);
            auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer +
                chunkElementOffset;
            const uint64_t remotePayloadOffset = payloadOffsets[slot] +
                static_cast<uint64_t>(rank) * bytesPerPeer + chunkByteOffset;
            const uint64_t remoteSignalOffset = signalOffsets[slot] +
                static_cast<uint64_t>(rank) * TILEXR_ALLTOALL_GROUP_SIGNAL_STRIDE;
            const uint64_t putBegin = AllToAllGroupTraceCycle(groupTrace);
            TileXR::UDMAPutSignalNbiOnQp<int32_t>(
                args, peer, qpIdx, localSrc, remotePayloadOffset, chunkBytes,
                remoteSignalOffset, expectedToken);
            AllToAllGroupTraceRecordTask(
                groupTrace, traceIteration, blockIdx, group, pass,
                TileXR::Demo::kAllToAllGroupTraceSendPutSignal, groupCount, passCount,
                peer, qpIdx, putBegin, AllToAllGroupTraceCycle(groupTrace));
            const uint64_t quietBegin = AllToAllGroupTraceCycle(groupTrace);
            const uint32_t quietStatus = TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
            AllToAllGroupTraceRecordTask(
                groupTrace, traceIteration, blockIdx, group, pass,
                TileXR::Demo::kAllToAllGroupTraceSendQuiet, groupCount, passCount,
                peer, qpIdx, quietBegin, AllToAllGroupTraceCycle(groupTrace));
            if (quietStatus != 0U) {
                AllToAllGroupRecordError(debug, blockIdx, TILEXR_ALLTOALL_GROUP_STAGE_QUIET,
                    group, pass, peer, qpIdx, quietStatus, expectedToken, 0ULL);
                AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
                    kernelBegin, AllToAllGroupTraceCycle(groupTrace));
                return;
            }
        }
    }
    AllToAllGroupTraceRecordKernel(groupTrace, traceIteration, blockIdx,
        kernelBegin, AllToAllGroupTraceCycle(groupTrace));
}

void launch_tilexr_udma_all_to_all_group(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR registeredMemory, GM_ADDR debug, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    GM_ADDR groupTrace, uint32_t traceIteration)
{
    tilexr_udma_all_to_all_group_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, registeredMemory, debug, invocationId,
        elementsPerPeer, chunkElements, passCount, groupCount,
        payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
        groupTrace, traceIteration);
}
