/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_data_as_flag.h"
#include "tilexr_udma.h"

constexpr int32_t TILEXR_UDMA_DEMO_MAGIC = 0x5444554d; // "TDUM"
constexpr uint64_t TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET = TileXR::IPC_DATA_OFFSET;
constexpr uint64_t TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET = TileXR::IPC_DATA_OFFSET;
constexpr uint32_t TILEXR_UDMA_DEMO_DATA_AS_FLAG_UB_BYTES = 64 * 1024;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE = 6;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_RANGE_VALID_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 16;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_WQE_BEFORE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 32;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_WQE_AFTER_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 48;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_LOCAL_TOKEN_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 64;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_REMOTE_BASE_LOW_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 80;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_MEM_ADDR_LOW_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 96;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_TPN_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 112;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_SEND_SAMPLE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 128;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_RECV_SAMPLE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 144;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_RECV_SLOT_SAMPLE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 160;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_REMOTE_DATA_OFFSET_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 176;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_REMOTE_READY_OFFSET_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 192;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_READY_SEEN_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 208;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER = TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER = TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SUM = TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER + 1;
constexpr uint64_t TILEXR_UDMA_DEMO_SIGNAL_MAX_POLLS = 100000000ULL;
constexpr uint64_t TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES = TileXR::TILEXR_UDMA_CACHE_LINE_SIZE;
constexpr int32_t TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS = -1001;
constexpr int32_t TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS = -1002;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_COPY_CORE_COUNT = 32U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES = 64 * 1024;

namespace {

__aicore__ inline uint64_t AllToAllPayloadBytes(int32_t elementsPerPeer)
{
    return static_cast<uint64_t>(elementsPerPeer) * sizeof(int32_t);
}

__aicore__ inline uint64_t AllToAllDataAsFlagSegmentBytes(uint64_t payloadBytes)
{
    return static_cast<uint64_t>(TileXR::DataAsFlagBlockCountForPayloadBytes(payloadBytes)) *
        TileXR::DATA_AS_FLAG_BLOCK_BYTES;
}

__aicore__ inline uint64_t WaitUdmaSignalToken(__gm__ uint64_t* signal, uint64_t token)
{
    uint64_t observed = ld_dev(signal, 0);
    uint64_t polls = 0;
    while (observed < token && polls < TILEXR_UDMA_DEMO_SIGNAL_MAX_POLLS) {
        observed = ld_dev(signal, 0);
        ++polls;
    }
    return observed;
}

__aicore__ inline __gm__ uint64_t* ControlSlot(__gm__ uint8_t* base, uint64_t offset, int32_t peer)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        base + offset + static_cast<uint64_t>(peer) * TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES);
}

__aicore__ inline uint32_t BigDataCopyCoreCount(int32_t blockNum)
{
    uint32_t cores = static_cast<uint32_t>(blockNum);
    if (cores > TILEXR_UDMA_DEMO_BIGDATA_COPY_CORE_COUNT) {
        cores = TILEXR_UDMA_DEMO_BIGDATA_COPY_CORE_COUNT;
    }
    return cores == 0 ? 1U : cores;
}

__aicore__ inline void BigDataCopyOneRelay(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint8_t> srcGlobal;
    srcGlobal.SetGlobalBuffer(src);
    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
    AscendC::DataCopyExtParams copyIn {1U, bytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint8_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(dst);
    AscendC::DataCopyExtParams copyOut {1U, bytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline void BigDataSendCopyShard(
    int32_t blockIdx, int32_t blockNum, int32_t rank, int32_t rankSize,
    __gm__ int32_t* input, __gm__ int32_t* output, __gm__ uint8_t* udmaMem,
    int32_t elementsPerPeer, int32_t chunkOffset, uint32_t chunkBytes,
    uint64_t sendDataOffset, uint64_t chunkBytesPerPeer,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint32_t copyCores = BigDataCopyCoreCount(blockNum);
    if (static_cast<uint32_t>(blockIdx) >= copyCores || chunkBytes == 0 || rankSize <= 0) {
        return;
    }
    const uint32_t chunksPerPeer =
        (chunkBytes + TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES - 1U) /
        TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
    const uint64_t totalUnits = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(chunksPerPeer);
    for (uint64_t unit = static_cast<uint64_t>(blockIdx); unit < totalUnits; unit += copyCores) {
        const int32_t peer = static_cast<int32_t>(unit / static_cast<uint64_t>(chunksPerPeer));
        const uint32_t chunkInPeer = static_cast<uint32_t>(unit % static_cast<uint64_t>(chunksPerPeer));
        const uint32_t offset = chunkInPeer * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
        const uint32_t bytes =
            (chunkBytes - offset < TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES) ?
            (chunkBytes - offset) : TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;

        auto src = reinterpret_cast<__gm__ uint8_t*>(
            input + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset) + offset;
        __gm__ uint8_t* dst = nullptr;
        if (peer == rank) {
            dst = reinterpret_cast<__gm__ uint8_t*>(
                output + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset) + offset;
        } else {
            dst = udmaMem + sendDataOffset + static_cast<uint64_t>(peer) * chunkBytesPerPeer + offset;
        }
        BigDataCopyOneRelay(dst, src, bytes, relayLocal);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void BigDataRelayCopyShard(
    int32_t blockIdx, int32_t blockNum, int32_t rank, int32_t rankSize,
    __gm__ int32_t* output, __gm__ uint8_t* udmaMem,
    int32_t elementsPerPeer, int32_t chunkOffset, uint32_t chunkBytes,
    uint64_t recvDataOffset, uint64_t chunkBytesPerPeer,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint32_t copyCores = BigDataCopyCoreCount(blockNum);
    if (static_cast<uint32_t>(blockIdx) >= copyCores || chunkBytes == 0 || rankSize <= 1) {
        return;
    }
    const uint32_t chunksPerPeer =
        (chunkBytes + TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES - 1U) /
        TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
    const uint64_t totalUnits = static_cast<uint64_t>(rankSize - 1) * static_cast<uint64_t>(chunksPerPeer);
    for (uint64_t unit = static_cast<uint64_t>(blockIdx); unit < totalUnits; unit += copyCores) {
        const int32_t logicalPeer = static_cast<int32_t>(unit / static_cast<uint64_t>(chunksPerPeer));
        const int32_t peer = logicalPeer < rank ? logicalPeer : logicalPeer + 1;
        const uint32_t chunkInPeer = static_cast<uint32_t>(unit % static_cast<uint64_t>(chunksPerPeer));
        const uint32_t offset = chunkInPeer * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
        const uint32_t bytes =
            (chunkBytes - offset < TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES) ?
            (chunkBytes - offset) : TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;

        auto src = udmaMem + recvDataOffset + static_cast<uint64_t>(peer) * chunkBytesPerPeer + offset;
        auto dst = reinterpret_cast<__gm__ uint8_t*>(
            output + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset) + offset;
        BigDataCopyOneRelay(dst, src, bytes, relayLocal);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_udma_all_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerRank;
    }
    if (!enabled) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)));
        TileXR::UDMAQuiet(args, peer);
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_put_signal_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR signalGM, GM_ADDR debugGM,
    int32_t elementsPerRank, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto signals = reinterpret_cast<__gm__ uint64_t*>(signalGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerRank;
    }
    if (!enabled) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    uint64_t signalBaseOffset = static_cast<uint64_t>(rankSize) * elementsPerRank * sizeof(int32_t);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutSignalNbi<int32_t>(args, peer, localSrc,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)),
            signalBaseOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t), signal);
        TileXR::UDMAQuiet(args, peer);
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset, int32_t chunkElements)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    // Multi-core: one block per peer. Block b handles peer b:
    //   - peer == rank -> local self-copy via DataCopyPad
    //   - peer != rank -> UDMA PUT to that peer
    // Block 0 writes the shared debug header; per-peer slots are written by
    // their owning block only, so no cross-block debug races.
    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(outputByteOffset);
    }
    if (!enabled) {
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t payloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t bytes = static_cast<uint32_t>(payloadBytes);

    // This block's assigned peer. When host launches rankSize blocks, block b
    // handles peer b. If fewer blocks are launched, peers are round-robined
    // and each block may handle more than one peer (still correct, just less
    // parallel); stride == blockNum keeps peer slots disjoint across blocks.
    for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
        if (peer == rank) {
            // Self-copy: local DataCopyPad, no network.
            auto selfSrc = input + static_cast<uint64_t>(rank) * elementsPerPeer + inputElementOffset;
            auto selfDst = output + static_cast<uint64_t>(rank) * effectiveChunkElements;
            constexpr uint32_t SELF_COPY_UB_BYTES = 64 * 1024;
            AscendC::TPipe pipe;
            AscendC::TBuf<AscendC::QuePosition::VECCALC> selfCopyTBuf;
            pipe.InitBuffer(selfCopyTBuf, SELF_COPY_UB_BYTES);
            AscendC::LocalTensor<uint8_t> selfCopyLocal = selfCopyTBuf.Get<uint8_t>();

            auto selfSrcBytes = reinterpret_cast<__gm__ uint8_t*>(selfSrc);
            auto selfDstBytes = reinterpret_cast<__gm__ uint8_t*>(selfDst);
            for (uint32_t offset = 0; offset < bytes; offset += SELF_COPY_UB_BYTES) {
                uint32_t copyBytes = (bytes - offset < SELF_COPY_UB_BYTES)
                    ? (bytes - offset) : SELF_COPY_UB_BYTES;

                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(selfSrcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, copyBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(selfCopyLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(selfDstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, copyBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, selfCopyLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }

        // UDMA PUT to peer.
        auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer + inputElementOffset;
        uint64_t remoteOffset = outputByteOffset +
            static_cast<uint64_t>(rank) * payloadBytes;
        auto registry = TileXR::GetUDMARegistry(args);
        auto udmaInfo = TileXR::GetUDMAInfo(args);
        auto wqCtx = TileXR::UDMAGetWQCtx(udmaInfo, peer, 0);
        auto remoteMemInfo = TileXR::UDMAGetRemoteMemInfo(udmaInfo, peer);
        bool rangeValid = TileXR::UDMARegisteredRangeValid(registry, peer, remoteOffset, bytes);
        uint32_t wqeBefore = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_RANGE_VALID_BASE + peer] = rangeValid ? 1 : 0;
            debug[TILEXR_UDMA_DEMO_DEBUG_WQE_BEFORE_BASE + peer] = static_cast<int32_t>(wqeBefore);
            debug[TILEXR_UDMA_DEMO_DEBUG_LOCAL_TOKEN_BASE + peer] = static_cast<int32_t>(wqCtx->localTokenId);
            debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_BASE_LOW_BASE + peer] =
                static_cast<int32_t>(reinterpret_cast<uint64_t>(registry->regions[peer].base) & 0xFFFFFFFFU);
            debug[TILEXR_UDMA_DEMO_DEBUG_MEM_ADDR_LOW_BASE + peer] =
                static_cast<int32_t>(remoteMemInfo->addr & 0xFFFFFFFFU);
            debug[TILEXR_UDMA_DEMO_DEBUG_TPN_BASE + peer] = static_cast<int32_t>(remoteMemInfo->tpn);
        }
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteOffset, bytes);
        uint32_t wqeAfter = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_WQE_AFTER_BASE + peer] = static_cast<int32_t>(wqeAfter);
        }
        uint32_t status = TileXR::UDMAQuietStatus(args, peer);
        if (debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    const uint64_t payloadBytes = AllToAllPayloadBytes(elementsPerPeer);
    const uint64_t segmentBytes = AllToAllDataAsFlagSegmentBytes(payloadBytes);
    auto inputBytes = reinterpret_cast<__gm__ uint8_t*>(input);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
    pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_DATA_AS_FLAG_UB_BYTES);
    AscendC::LocalTensor<uint8_t> scratch = tBuf.Get<uint8_t>();
    if (TileXR::DataAsFlagInit(scratch) == 0U) {
        if (debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = -1;
        }
        return;
    }

    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto localSrc = inputBytes + static_cast<uint64_t>(dstRank) * payloadBytes;
        auto remoteDst = reinterpret_cast<__gm__ uint8_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET +
            static_cast<uint64_t>(rank) * segmentBytes);
        (void)TileXR::DataAsFlagSend(remoteDst, localSrc, payloadBytes, scratch);
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    const uint64_t payloadBytes = AllToAllPayloadBytes(elementsPerPeer);
    const uint64_t segmentBytes = AllToAllDataAsFlagSegmentBytes(payloadBytes);
    auto outputBytes = reinterpret_cast<__gm__ uint8_t*>(output);
    auto localBase = reinterpret_cast<__gm__ uint8_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
    pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_DATA_AS_FLAG_UB_BYTES);
    AscendC::LocalTensor<uint8_t> scratch = tBuf.Get<uint8_t>();

    for (int32_t srcRank = 0; srcRank < args->rankSize; ++srcRank) {
        auto localSrc = localBase + static_cast<uint64_t>(srcRank) * segmentBytes;
        auto localDst = outputBytes + static_cast<uint64_t>(srcRank) * payloadBytes;
        if (!TileXR::DataAsFlagCheckAndRecv(localSrc, payloadBytes, localDst, scratch)) {
            if (debug != nullptr) {
                debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = -1;
            }
            return;
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_plain_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + static_cast<uint64_t>(rank) * rankSize * elementsPerPeer;
        auto localSrc = input + static_cast<uint64_t>(dstRank) * elementsPerPeer;
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            remoteDst[i] = localSrc[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 2;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_plain_ipc_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rankSize = args->rankSize;
    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        auto localSrc = localBase + static_cast<uint64_t>(srcRank) * rankSize * elementsPerPeer;
        auto localDst = output + static_cast<uint64_t>(srcRank) * elementsPerPeer;
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            localDst[i] = localSrc[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 2;
    }
}

// Fused single-kernel alltoall: send + flag-sync + receive in one kernel launch.
// Each rank writes data to peers, sets a per-peer flag, polls peer flags, then reads.
// Flag layout: peerMems[peer] + srcRank * sizeof(uint64_t), within the 2MB flag region.
// Round parameter enables multi-round pipelining without host barriers.
extern "C" __global__ __aicore__ void tilexr_all_to_all_fused_ipc_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, int32_t round)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    const int64_t flagValue = static_cast<int64_t>(round);

    // UB: 64KB for DMA data relay + 64B for flag polling
    constexpr uint32_t DATA_UB_BYTES = 64 * 1024;
    constexpr uint32_t CHUNK_BYTES = DATA_UB_BYTES;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dataTBuf;
    pipe.InitBuffer(dataTBuf, DATA_UB_BYTES);
    AscendC::LocalTensor<uint8_t> dataLocal = dataTBuf.Get<uint8_t>();

    // Flag layout in peer's flag region (before IPC_DATA_OFFSET):
    //   [0 .. rankSize-1]                           = ready flags (data written)
    //   [rankSize .. 2*rankSize-1]                  = consumed ACK flags (data read)
    // Each flag is int64_t, indexed by srcRank.

    // Phase 1: Write data to all peers' IPC staging area via DMA (DataCopyPad)
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + static_cast<uint64_t>(rank) * rankSize * elementsPerPeer;
        auto localSrc = input + static_cast<uint64_t>(dstRank) * elementsPerPeer;

        auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
        auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(remoteDst);
        uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

        for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
            uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
            uint32_t alignedBytes = bytes & ~31U;

            if (alignedBytes > 0) {
                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(srcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(dstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            for (uint32_t i = alignedBytes; i < bytes; ++i) {
                dstBytes[offset + i] = srcBytes[offset + i];
            }
        }
    }

    // Phase 2: Send "data ready" flag to each peer via scalar store.
    // Scalar store to P2P address after MTE3_S barrier ensures all DMA
    // data writes are submitted before the flag.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        if (dstRank == rank) {
            continue;
        }
        auto flagAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[dstRank] + static_cast<uint64_t>(rank) * sizeof(int64_t));
        *flagAddr = flagValue;
    }

    // Phase 3: Poll all peers' "data ready" flags (read from own flag region).
    // Wait until each peer has written their data to our staging area.
    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        if (srcRank == rank) {
            continue;
        }
        auto flagAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[rank] + static_cast<uint64_t>(srcRank) * sizeof(int64_t));
        int64_t observed = 0;
        do {
            observed = *flagAddr;
        } while (observed < flagValue);
    }

    // Phase 4: Read data from local IPC staging area into output via DMA.
    // By now all peers' data has arrived (they set their ready flags).
    {
        auto localBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
            auto localSrc = localBase + static_cast<uint64_t>(srcRank) * rankSize * elementsPerPeer;
            auto localDst = output + static_cast<uint64_t>(srcRank) * elementsPerPeer;

            auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
            auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(localDst);
            uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

            for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
                uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
                uint32_t alignedBytes = bytes & ~31U;

                if (alignedBytes > 0) {
                    AscendC::GlobalTensor<uint8_t> srcGlobal;
                    srcGlobal.SetGlobalBuffer(srcBytes + offset);
                    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                    AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                    AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                    AscendC::GlobalTensor<uint8_t> dstGlobal;
                    dstGlobal.SetGlobalBuffer(dstBytes + offset);
                    AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                    AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                for (uint32_t i = alignedBytes; i < bytes; ++i) {
                    dstBytes[offset + i] = srcBytes[offset + i];
                }
            }
        }
    }

    // Phase 5: Send "data consumed" ACK to each peer.
    // Tells peers that we have read their data; they can safely overwrite
    // the staging area in the next round.
    // ACK flag offset: rankSize * sizeof(int64_t) + rank * sizeof(int64_t)
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        if (dstRank == rank) {
            continue;
        }
        auto ackAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[dstRank] +
            static_cast<uint64_t>(rankSize + rank) * sizeof(int64_t));
        *ackAddr = flagValue;
    }

    // Phase 6: Poll all peers' "data consumed" ACKs.
    // Wait until all peers confirm they have read our data.
    // Only then can we proceed to the next round (which will overwrite).
    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        if (srcRank == rank) {
            continue;
        }
        auto ackAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[rank] +
            static_cast<uint64_t>(rankSize + srcRank) * sizeof(int64_t));
        int64_t observed = 0;
        do {
            observed = *ackAddr;
        } while (observed < flagValue);
    }

    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 3;
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 3;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_reduce_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + rank * elementsPerRank;
        for (int32_t i = 0; i < elementsPerRank; ++i) {
            remoteDst[i] = input[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_reduce_ipc_sum_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
    for (int32_t i = 0; i < elementsPerRank; ++i) {
        int32_t sum = 0;
        for (int32_t srcRank = 0; srcRank < args->rankSize; ++srcRank) {
            sum += localBase[srcRank * elementsPerRank + i];
        }
        output[i] = sum;
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SUM] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_registered_smoke_kernel(
    GM_ADDR commArgsGM, GM_ADDR localGM, GM_ADDR debugGM, uint32_t bytes, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto local = reinterpret_cast<__gm__ uint8_t*>(localGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    bool enabled = TileXR::UDMARegistryEnabled(args);
    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = enabled ? 1 : 0;
        debug[2] = static_cast<int32_t>(bytes);
    }
    if (!enabled) {
        return;
    }

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutRegisteredNbi<uint8_t>(args, peer, local, 0, bytes);
        TileXR::UDMAGetRegisteredNbi<uint8_t>(args, peer, local, 0, bytes);
        TileXR::UDMAPutRegisteredSignalNbi<uint8_t>(args, peer, local, 0, bytes, 0, signal);
        TileXR::UDMAQuiet(args, peer);
    }
}

// DMA-based scatter: write data to all peers' IPC staging area via DataCopyPad.
// Split from the fused kernel to allow host-side sync between scatter and gather,
// which guarantees P2P write visibility without fragile in-kernel flag polling.
extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_scatter_dma_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;

    constexpr uint32_t DATA_UB_BYTES = 64 * 1024;
    constexpr uint32_t CHUNK_BYTES = DATA_UB_BYTES;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dataTBuf;
    pipe.InitBuffer(dataTBuf, DATA_UB_BYTES);
    AscendC::LocalTensor<uint8_t> dataLocal = dataTBuf.Get<uint8_t>();

    // Write data to all peers' IPC staging area via DMA
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + static_cast<uint64_t>(rank) * rankSize * elementsPerPeer;
        auto localSrc = input + static_cast<uint64_t>(dstRank) * elementsPerPeer;

        auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
        auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(remoteDst);
        uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

        for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
            uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
            uint32_t alignedBytes = bytes & ~31U;

            if (alignedBytes > 0) {
                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(srcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(dstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            for (uint32_t i = alignedBytes; i < bytes; ++i) {
                dstBytes[offset + i] = srcBytes[offset + i];
            }
        }
    }

    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 4;
    }
}

// DMA-based gather: read data from local IPC staging area into output via DataCopyPad.
extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_gather_dma_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;

    constexpr uint32_t DATA_UB_BYTES = 64 * 1024;
    constexpr uint32_t CHUNK_BYTES = DATA_UB_BYTES;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dataTBuf;
    pipe.InitBuffer(dataTBuf, DATA_UB_BYTES);
    AscendC::LocalTensor<uint8_t> dataLocal = dataTBuf.Get<uint8_t>();

    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);

    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        auto localSrc = localBase + static_cast<uint64_t>(srcRank) * rankSize * elementsPerPeer;
        auto localDst = output + static_cast<uint64_t>(srcRank) * elementsPerPeer;

        auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
        auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(localDst);
        uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

        for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
            uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
            uint32_t alignedBytes = bytes & ~31U;

            if (alignedBytes > 0) {
                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(srcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(dstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            for (uint32_t i = alignedBytes; i < bytes; ++i) {
                dstBytes[offset + i] = srcBytes[offset + i];
            }
        }
    }

    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 4;
    }
}

void launch_tilexr_all_to_all_ipc_scatter_dma(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_scatter_dma_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_ipc_gather_dma(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_gather_dma_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerPeer);
}

void launch_tilexr_udma_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_udma_all_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, debug, elementsPerRank);
}

void launch_tilexr_udma_put_signal(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signal)
{
    tilexr_udma_put_signal_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, signals, debug, elementsPerRank, signal);
}

void launch_tilexr_udma_all_to_all(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements)
{
    tilexr_udma_all_to_all_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, outputByteOffset, inputElementOffset, chunkElements);
}

void launch_tilexr_all_to_all_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_ipc_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_plain_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_plain_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_plain_ipc_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_plain_ipc_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_fused_ipc(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, int32_t round)
{
    tilexr_all_to_all_fused_ipc_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, round);
}

void launch_tilexr_all_reduce_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_all_reduce_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerRank);
}

void launch_tilexr_all_reduce_ipc_sum(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_all_reduce_ipc_sum_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerRank);
}

void launch_tilexr_udma_registered_smoke(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR local, GM_ADDR debug, uint32_t bytes, uint64_t signal)
{
    tilexr_udma_registered_smoke_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, local, debug, bytes, signal);
}

// ---------------------------------------------------------------------------
// Latency micro-kernels (testType 4 = P2P-only, testType 5 = DataCopy-only).
// These mirror the two halves of tilexr_udma_all_to_all_kernel so the P2P
// communication latency and the local DataCopy latency can be measured in
// isolation. Each block handles exactly one peer (block b -> peer b); for
// the P2P kernel the self-peer (peer == rank) is skipped so that block does
// no network work, and for the DataCopy kernel every block performs a local
// GM->UB->GM copy of the same payload size as one alltoall peer slice.
// ---------------------------------------------------------------------------

extern "C" __global__ __aicore__ void tilexr_udma_p2p_latency_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(outputByteOffset);
    }
    if (!enabled) {
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t payloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t bytes = static_cast<uint32_t>(payloadBytes);

    // One block per peer, skip self (peer == rank): no local copy here.
    for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
        if (peer == rank) {
            continue;
        }
        auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer + inputElementOffset;
        uint64_t remoteOffset = outputByteOffset +
            static_cast<uint64_t>(rank) * payloadBytes;
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteOffset, bytes);
        uint32_t status = TileXR::UDMAQuietStatus(args, peer);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_datacopy_latency_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, int32_t chunkElements)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;

    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = 1; // always "enabled" path for datacopy
        debug[4] = elementsPerPeer;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t payloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t bytes = static_cast<uint32_t>(payloadBytes);

    // Each block performs the same GM->UB->GM self-copy that the alltoall
    // kernel uses for the self-peer slice. block b copies rank b's slice.
    for (int32_t whichRank = blockIdx; whichRank < rankSize; whichRank += blockNum) {
        auto selfSrc = input + static_cast<uint64_t>(whichRank) * elementsPerPeer;
        auto selfDst = output + static_cast<uint64_t>(whichRank) * effectiveChunkElements;
        constexpr uint32_t SELF_COPY_UB_BYTES = 64 * 1024;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> selfCopyTBuf;
        pipe.InitBuffer(selfCopyTBuf, SELF_COPY_UB_BYTES);
        AscendC::LocalTensor<uint8_t> selfCopyLocal = selfCopyTBuf.Get<uint8_t>();

        auto selfSrcBytes = reinterpret_cast<__gm__ uint8_t*>(selfSrc);
        auto selfDstBytes = reinterpret_cast<__gm__ uint8_t*>(selfDst);
        for (uint32_t offset = 0; offset < bytes; offset += SELF_COPY_UB_BYTES) {
            uint32_t copyBytes = (bytes - offset < SELF_COPY_UB_BYTES)
                ? (bytes - offset) : SELF_COPY_UB_BYTES;

            AscendC::GlobalTensor<uint8_t> srcGlobal;
            srcGlobal.SetGlobalBuffer(selfSrcBytes + offset);
            AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
            AscendC::DataCopyExtParams copyIn {1U, copyBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(selfCopyLocal, srcGlobal, copyIn, padIn);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            AscendC::GlobalTensor<uint8_t> dstGlobal;
            dstGlobal.SetGlobalBuffer(selfDstBytes + offset);
            AscendC::DataCopyExtParams copyOut {1U, copyBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(dstGlobal, selfCopyLocal, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void launch_tilexr_udma_p2p_latency(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements)
{
    tilexr_udma_p2p_latency_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, outputByteOffset, inputElementOffset, chunkElements);
}

void launch_tilexr_datacopy_latency(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, int32_t chunkElements)
{
    tilexr_datacopy_latency_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, chunkElements);
}


// ---------------------------------------------------------------------------
// Fused alltoall kernel -- single launch, loop×pass inside, no IPC fallback.
// input  : full-size GM (NOT registered), H2D primed once on host.
// output : full-size GM (NOT registered), filled inside kernel.
// udmaMem: registered chunk-sized relay buffer (rankSize*chunkElements*4B),
//          reused every pass (overwrite) -- fits the 32MB URMA single-shot
//          registration limit regardless of total data size.
// signals: registered region of rankSize uint64 slots.
//
// Per loop iter L, per pass p (chunkOffset = p*chunkElements):
//   1. SEND : block b (peer b!=rank) UDMAPutSignalNbi input[b-slice][offset]
//             -> peer b's udmaMem[rank slot] + signal (L*passCount+p+1).
//      SELF: DataCopyPad input[rank-slice][offset] -> output[rank-slice][offset].
//   2. QUIET: UDMAQuietStatus per owned peer (local WQE completion).
//   3. WAIT : spin until each peer wrote this pass's signal into our
//             signals[peer] (peer's data landed in our udmaMem[peer slot]).
//   4. RELAY: DataCopyPad udmaMem[peer slot] -> output[peer-slice][offset].
//   5. SyncAll; next pass overwrites udmaMem.
// Signal tokens are globally monotonic (loop*passCount+pass+1) so no
// re-zeroing is needed across loops/passes.
// ---------------------------------------------------------------------------

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_fused_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR udmaMemGM,
    GM_ADDR signalGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t udmaMemByteOffset, uint64_t signalByteOffsetBase,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto udmaMem = reinterpret_cast<__gm__ int32_t*>(udmaMemGM);
    auto signals = reinterpret_cast<__gm__ uint64_t*>(signalGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(udmaMemByteOffset);
    }
    if (!enabled) {
        for (uint32_t l = 0; l < loopCount; ++l) { AscendC::SyncAll(); }
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t peerPayloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t peerBytes = static_cast<uint32_t>(peerPayloadBytes);
    constexpr uint32_t RELAY_UB_BYTES = 64 * 1024;

    for (uint32_t loop = 0; loop < loopCount; ++loop) {
        for (uint32_t pass = 0; pass < passCount; ++pass) {
            const int32_t chunkOffset = static_cast<int32_t>(pass) * effectiveChunkElements;
            const int32_t chunkElem = (elementsPerPeer - chunkOffset < effectiveChunkElements)
                ? (elementsPerPeer - chunkOffset) : effectiveChunkElements;
            const uint32_t chunkBytes = static_cast<uint32_t>(
                static_cast<uint64_t>(chunkElem) * sizeof(int32_t));
            const uint64_t expectedSignal =
                static_cast<uint64_t>(loop) * static_cast<uint64_t>(passCount) +
                static_cast<uint64_t>(pass) + 1;

            // ---- 1. SEND + SELF ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) {
                    // self: input -> output directly (no P2P, no relay).
                    auto selfSrc = input + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset;
                    auto selfDst = output + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset;
                    AscendC::TPipe pipe;
                    AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
                    pipe.InitBuffer(relayTBuf, RELAY_UB_BYTES);
                    AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();
                    auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(selfSrc);
                    auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(selfDst);
                    for (uint32_t off = 0; off < chunkBytes; off += RELAY_UB_BYTES) {
                        uint32_t cb = (chunkBytes - off < RELAY_UB_BYTES) ? (chunkBytes - off) : RELAY_UB_BYTES;
                        AscendC::GlobalTensor<uint8_t> srcGlobal; srcGlobal.SetGlobalBuffer(srcBytes + off);
                        AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                        AscendC::DataCopyExtParams copyIn {1U, cb, 0U, 0U, 0U};
                        AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::GlobalTensor<uint8_t> dstGlobal; dstGlobal.SetGlobalBuffer(dstBytes + off);
                        AscendC::DataCopyExtParams copyOut {1U, cb, 0U, 0U, 0U};
                        AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    continue;
                }
                // P2P: read input[peer slice][offset] -> peer's udmaMem[rank slot] + signal.
                auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset;
                uint64_t remoteOffset = udmaMemByteOffset +
                    static_cast<uint64_t>(rank) * peerPayloadBytes;
                uint64_t remoteSignalOffset = signalByteOffsetBase +
                    static_cast<uint64_t>(rank) * sizeof(uint64_t);
                TileXR::UDMAPutSignalNbi<int32_t>(args, peer, localSrc,
                    remoteOffset, chunkBytes, remoteSignalOffset, expectedSignal);
            }

            // ---- 2. QUIET ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) continue;
                uint32_t status = TileXR::UDMAQuietStatus(args, peer);
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
                }
            }

            // ---- 3. WAIT : each block spins on its owned peer's signal slot. ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) continue;
                __gm__ uint64_t* slot = signals + static_cast<uint64_t>(peer);
                while (*slot != expectedSignal) {
                }
            }
            AscendC::SyncAll();

            // ---- 4. RELAY : udmaMem[peer slot] -> output[peer slice][offset]. ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) continue;
                auto relaySrc = udmaMem + static_cast<uint64_t>(peer) * effectiveChunkElements;
                auto relayDst = output + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset;
                AscendC::TPipe pipe;
                AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
                pipe.InitBuffer(relayTBuf, RELAY_UB_BYTES);
                AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();
                auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(relaySrc);
                auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(relayDst);
                for (uint32_t off = 0; off < chunkBytes; off += RELAY_UB_BYTES) {
                    uint32_t cb = (chunkBytes - off < RELAY_UB_BYTES) ? (chunkBytes - off) : RELAY_UB_BYTES;
                    AscendC::GlobalTensor<uint8_t> srcGlobal; srcGlobal.SetGlobalBuffer(srcBytes + off);
                    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                    AscendC::DataCopyExtParams copyIn {1U, cb, 0U, 0U, 0U};
                    AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::GlobalTensor<uint8_t> dstGlobal; dstGlobal.SetGlobalBuffer(dstBytes + off);
                    AscendC::DataCopyExtParams copyOut {1U, cb, 0U, 0U, 0U};
                    AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::SyncAll();
        }
    }
    AscendC::SyncAll();
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_bigdata_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR udmaMemGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t dataOffset, uint64_t readyPayloadOffset, uint64_t ackPayloadOffset,
    uint64_t readySignalOffset, uint64_t ackSignalOffset,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount, uint64_t tokenBase,
    uint32_t profileStage)
{
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_PREPARE = 0;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_SEND_COPY = 1;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_SEND_SYNC = 2;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_DATA_PUT = 3;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_WAIT_READY = 4;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_RELAY_COPY = 5;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_ACK_PUT = 6;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_WAIT_ACK = 7;
    constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_FULL = 8;
    if (profileStage > TILEXR_BIGDATA_PROFILE_STAGE_FULL) {
        profileStage = TILEXR_BIGDATA_PROFILE_STAGE_FULL;
    }

    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto udmaMem = reinterpret_cast<__gm__ uint8_t*>(udmaMemGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);
    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = chunkElements;
    }
    if (!enabled) {
        for (uint32_t l = 0; l < loopCount; ++l) {
            AscendC::SyncAll();
        }
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t chunkBytesPerPeer = static_cast<uint64_t>(effectiveChunkElements) * sizeof(int32_t);
    const uint64_t sendDataOffset = dataOffset;
    const uint64_t recvDataOffset = dataOffset + static_cast<uint64_t>(rankSize) * chunkBytesPerPeer;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
    pipe.InitBuffer(relayTBuf, TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES);
    AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();

    for (uint32_t loop = 0; loop < loopCount; ++loop) {
        for (uint32_t pass = 0; pass < passCount; ++pass) {
            const int32_t chunkOffset = static_cast<int32_t>(pass) * effectiveChunkElements;
            const int32_t remaining = elementsPerPeer - chunkOffset;
            const int32_t chunkElem = remaining < effectiveChunkElements ? remaining : effectiveChunkElements;
            if (chunkElem <= 0) {
                continue;
            }
            const uint32_t chunkBytes = static_cast<uint32_t>(static_cast<uint64_t>(chunkElem) * sizeof(int32_t));
            const uint64_t token =
                tokenBase + static_cast<uint64_t>(loop) * static_cast<uint64_t>(passCount) +
                static_cast<uint64_t>(pass) + 1ULL;

            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_PREPARE) {
                continue;
            }

            BigDataSendCopyShard(blockIdx, blockNum, rank, rankSize, input, output, udmaMem,
                elementsPerPeer, chunkOffset, chunkBytes, sendDataOffset, chunkBytesPerPeer, relayLocal);
            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_SEND_COPY) {
                continue;
            }
            AscendC::SyncAll();
            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_SEND_SYNC) {
                continue;
            }

            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) {
                    continue;
                }
                auto sendSlot = udmaMem + sendDataOffset + static_cast<uint64_t>(peer) * chunkBytesPerPeer;
                auto localSrc = reinterpret_cast<__gm__ int32_t*>(sendSlot);
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_SEND_SAMPLE_BASE + peer] = localSrc[0];
                }
                const uint64_t remoteDataOffset =
                    recvDataOffset + static_cast<uint64_t>(rank) * chunkBytesPerPeer;
                const uint64_t remoteReadyOffset =
                    readySignalOffset + static_cast<uint64_t>(rank) * TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES;
                auto registry = TileXR::GetUDMARegistry(args);
                auto udmaInfo = TileXR::GetUDMAInfo(args);
                auto wqCtx = TileXR::UDMAGetWQCtx(udmaInfo, peer, 0);
                auto remoteMemInfo = TileXR::UDMAGetRemoteMemInfo(udmaInfo, peer);
                bool rangeValid = TileXR::UDMARegisteredRangeValid(registry, peer, remoteDataOffset, chunkBytes);
                uint32_t wqeBefore = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_RANGE_VALID_BASE + peer] = rangeValid ? 1 : 0;
                    debug[TILEXR_UDMA_DEMO_DEBUG_WQE_BEFORE_BASE + peer] = static_cast<int32_t>(wqeBefore);
                    debug[TILEXR_UDMA_DEMO_DEBUG_LOCAL_TOKEN_BASE + peer] = static_cast<int32_t>(wqCtx->localTokenId);
                    debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_BASE_LOW_BASE + peer] =
                        static_cast<int32_t>(reinterpret_cast<uint64_t>(registry->regions[peer].base) & 0xFFFFFFFFU);
                    debug[TILEXR_UDMA_DEMO_DEBUG_MEM_ADDR_LOW_BASE + peer] =
                        static_cast<int32_t>(remoteMemInfo->addr & 0xFFFFFFFFU);
                    debug[TILEXR_UDMA_DEMO_DEBUG_TPN_BASE + peer] = static_cast<int32_t>(remoteMemInfo->tpn);
                    debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_DATA_OFFSET_BASE + peer] =
                        static_cast<int32_t>(remoteDataOffset);
                    debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_READY_OFFSET_BASE + peer] =
                        static_cast<int32_t>(remoteReadyOffset);
                }
                TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteDataOffset, chunkBytes);
                uint32_t status = TileXR::UDMAQuietStatus(args, peer);
                auto readyPayload = ControlSlot(udmaMem, readyPayloadOffset, peer);
                *readyPayload = token;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(readyPayload), sizeof(uint64_t));
                TileXR::UDMAPutNbi<uint64_t>(args, peer, readyPayload, remoteReadyOffset, sizeof(uint64_t));
                uint32_t readyStatus = TileXR::UDMAQuietStatus(args, peer);
                if (status == 0) {
                    status = readyStatus;
                }
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                    uint32_t wqeAfter = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
                    debug[TILEXR_UDMA_DEMO_DEBUG_WQE_AFTER_BASE + peer] = static_cast<int32_t>(wqeAfter);
                    debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
                }
            }
            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_DATA_PUT) {
                continue;
            }

            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) {
                    continue;
                }
                uint64_t observed = WaitUdmaSignalToken(ControlSlot(udmaMem, readySignalOffset, peer), token);
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_READY_SEEN_BASE + peer] =
                        static_cast<int32_t>(observed);
                    if (observed < token) {
                        debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                            TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS;
                    }
                }
            }
            AscendC::SyncAll();
            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_WAIT_READY) {
                continue;
            }

            if (blockIdx < rankSize && blockIdx != rank && debug != nullptr && loop == 0 && pass == 0 && blockIdx < 16) {
                auto recvSlot = udmaMem + recvDataOffset + static_cast<uint64_t>(blockIdx) * chunkBytesPerPeer;
                auto recvSlotInt = reinterpret_cast<__gm__ int32_t*>(recvSlot);
                debug[TILEXR_UDMA_DEMO_DEBUG_RECV_SLOT_SAMPLE_BASE + blockIdx] = recvSlotInt[0];
            }
            BigDataRelayCopyShard(blockIdx, blockNum, rank, rankSize, output, udmaMem,
                elementsPerPeer, chunkOffset, chunkBytes, recvDataOffset, chunkBytesPerPeer, relayLocal);
            AscendC::SyncAll();
            if (blockIdx < rankSize && blockIdx != rank && debug != nullptr && loop == 0 && pass == 0 && blockIdx < 16) {
                auto relayDst = output + static_cast<uint64_t>(blockIdx) * elementsPerPeer + chunkOffset;
                debug[TILEXR_UDMA_DEMO_DEBUG_RECV_SAMPLE_BASE + blockIdx] = relayDst[0];
            }
            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_RELAY_COPY) {
                continue;
            }

            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) {
                    continue;
                }
                auto ackPayload = ControlSlot(udmaMem, ackPayloadOffset, peer);
                *ackPayload = token;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ackPayload), sizeof(uint64_t));
                const uint64_t remoteAckOffset =
                    ackSignalOffset + static_cast<uint64_t>(rank) * TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES;
                TileXR::UDMAPutNbi<uint64_t>(args, peer, ackPayload, remoteAckOffset, sizeof(uint64_t));
                (void)TileXR::UDMAQuietStatus(args, peer);
            }
            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_ACK_PUT) {
                continue;
            }

            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) {
                    continue;
                }
                uint64_t observed = WaitUdmaSignalToken(ControlSlot(udmaMem, ackSignalOffset, peer), token);
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16 && observed < token) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                        TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS;
                }
            }
            AscendC::SyncAll();
            if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_WAIT_ACK) {
                continue;
            }
        }
    }
    AscendC::SyncAll();
}

void launch_tilexr_udma_all_to_all_bigdata(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR udmaMem, GM_ADDR debug, int32_t elementsPerPeer,
    uint64_t dataOffset, uint64_t readyPayloadOffset, uint64_t ackPayloadOffset,
    uint64_t readySignalOffset, uint64_t ackSignalOffset,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount, uint64_t tokenBase,
    uint32_t profileStage)
{
    tilexr_udma_all_to_all_bigdata_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, udmaMem, debug, elementsPerPeer,
        dataOffset, readyPayloadOffset, ackPayloadOffset, readySignalOffset, ackSignalOffset,
        chunkElements, passCount, loopCount, tokenBase, profileStage);
}

void launch_tilexr_udma_all_to_all_fused(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR udmaMem, GM_ADDR signal, GM_ADDR debug, int32_t elementsPerPeer,
    uint64_t udmaMemByteOffset, uint64_t signalByteOffsetBase,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount)
{
    tilexr_udma_all_to_all_fused_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, udmaMem, signal, debug, elementsPerPeer,
        udmaMemByteOffset, signalByteOffsetBase, chunkElements, passCount, loopCount);
}
