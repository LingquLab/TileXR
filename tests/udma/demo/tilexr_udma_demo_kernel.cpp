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
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER = TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER = TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SUM = TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER + 1;

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
