/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_data_as_flag.h"
#include "tilexr_udma.h"

constexpr int32_t TILEXR_UDMA_DEMO_MAGIC = 0x5444554d; // "TDUM"
constexpr uint32_t TILEXR_UDMA_DEMO_P2P_UB_BYTES = 64 * 1024;
constexpr uint32_t TILEXR_UDMA_DEMO_P2P_SYNC_UB_BYTES = 4 * 1024;
constexpr uint32_t TILEXR_UDMA_DEMO_P2P_COPY_TILE_BYTES =
    TILEXR_UDMA_DEMO_P2P_UB_BYTES - TILEXR_UDMA_DEMO_P2P_SYNC_UB_BYTES;
constexpr uint32_t TILEXR_UDMA_DEMO_P2P_MAX_DEBUG_BLOCKS = 16;
constexpr uint32_t TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES = 32;
constexpr uint32_t TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_VALUE = 0xace00001u;

__aicore__ inline uint32_t TileXRUdmaDemoCeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

__aicore__ inline void TileXRUdmaDemoBlockSlice(
    uint32_t total, uint32_t blockNum, uint32_t blockIdx, uint32_t& offset, uint32_t& bytes)
{
    uint32_t perBlock = TileXRUdmaDemoCeilDiv(total, blockNum);
    perBlock = ((perBlock + TileXR::BLOCK_UNIT_BYTE - 1) / TileXR::BLOCK_UNIT_BYTE) * TileXR::BLOCK_UNIT_BYTE;
    offset = blockIdx * perBlock;
    if (offset >= total) {
        offset = total;
        bytes = 0;
        return;
    }
    bytes = total - offset;
    if (bytes > perBlock) {
        bytes = perBlock;
    }
}

__aicore__ inline void TileXRUdmaDemoWqeSlice(
    uint32_t total, uint32_t wqeCount, uint32_t wqeIdx, uint32_t& offset, uint32_t& bytes)
{
    uint32_t perWqe = TileXRUdmaDemoCeilDiv(total, wqeCount);
    perWqe = ((perWqe + TileXR::BLOCK_UNIT_BYTE - 1) / TileXR::BLOCK_UNIT_BYTE) * TileXR::BLOCK_UNIT_BYTE;
    offset = wqeIdx * perWqe;
    if (wqeCount == 0 || offset >= total) {
        offset = total;
        bytes = 0;
        return;
    }
    bytes = total - offset;
    if (bytes > perWqe) {
        bytes = perWqe;
    }
}

__aicore__ inline bool TileXRUdmaDemoResolvePeer(
    int32_t rank, int32_t srcRank, int32_t dstRank, int32_t traffic, int32_t& peer)
{
    const bool bidir = traffic == 1;
    if (bidir) {
        if (rank == srcRank) {
            peer = dstRank;
            return true;
        }
        if (rank == dstRank) {
            peer = srcRank;
            return true;
        }
        peer = -1;
        return false;
    }
    peer = dstRank;
    return rank == srcRank;
}

__aicore__ inline bool TileXRUdmaDemoResolveDataAsFlagRole(
    int32_t rank, int32_t srcRank, int32_t dstRank, int32_t traffic,
    bool& isSender, bool& isReceiver, int32_t& peer)
{
    const bool bidir = traffic == 1;
    isSender = false;
    isReceiver = false;
    peer = -1;
    if (bidir) {
        if (rank == srcRank) {
            isSender = true;
            isReceiver = true;
            peer = dstRank;
            return true;
        }
        if (rank == dstRank) {
            isSender = true;
            isReceiver = true;
            peer = srcRank;
            return true;
        }
        return false;
    }
    if (rank == srcRank) {
        isSender = true;
        peer = dstRank;
        return true;
    }
    if (rank == dstRank) {
        isReceiver = true;
        peer = srcRank;
        return true;
    }
    return false;
}

__aicore__ inline uint32_t TileXRUdmaDemoFoldDebugStatus(
    __gm__ uint32_t* debug, uint32_t blockIdx, uint32_t status)
{
    if (debug != nullptr && blockIdx < TILEXR_UDMA_DEMO_P2P_MAX_DEBUG_BLOCKS) {
        debug[8 + blockIdx] = status;
    }
    return status;
}

__aicore__ inline uint32_t TileXRUdmaDemoMemoryVisibleAckOffset(uint32_t bytes)
{
    return ((bytes + TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES - 1U) /
               TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES) *
        TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES;
}

__aicore__ inline uint32_t TileXRUdmaDemoMemoryVisibleAckValue(
    uint32_t pattern, uint32_t bytes, uint32_t blockIdx, uint32_t token)
{
    return TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_VALUE ^ pattern ^ bytes ^ blockIdx ^ (token << 16);
}

__aicore__ inline void TileXRUdmaDemoWriteMemoryVisibleAck(
    GM_ADDR ackGM, AscendC::TBuf<AscendC::QuePosition::VECCALC>& tBuf, uint32_t ackValue)
{
    AscendC::LocalTensor<uint32_t> local =
        tBuf.GetWithOffset<uint32_t>(
            TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES, TILEXR_UDMA_DEMO_P2P_SYNC_UB_BYTES);
    AscendC::Duplicate<uint32_t>(local, ackValue, TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES / sizeof(uint32_t));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint32_t> ackGlobal;
    ackGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ackGM),
        TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES / sizeof(uint32_t));
    AscendC::DataCopyPad(
        ackGlobal, local, AscendC::DataCopyParams {1, TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline bool TileXRUdmaDemoCheckMemoryVisibleAck(
    GM_ADDR ackGM, AscendC::TBuf<AscendC::QuePosition::VECCALC>& tBuf, uint32_t expectedValue)
{
    AscendC::LocalTensor<uint32_t> local =
        tBuf.GetWithOffset<uint32_t>(
            TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES, TILEXR_UDMA_DEMO_P2P_SYNC_UB_BYTES);
    AscendC::GlobalTensor<uint32_t> ackGlobal;
    ackGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ackGM),
        TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES / sizeof(uint32_t));
    AscendC::DataCopyPad(
        local, ackGlobal, AscendC::DataCopyParams {1, TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES, 0, 0},
        AscendC::DataCopyPadParams {false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return local.GetValue(0) == expectedValue;
}

__aicore__ inline void TileXRUdmaDemoDataAsFlagSlice(
    uint32_t bytes, uint32_t blockNum, uint32_t blockIdx,
    uint32_t& payloadOffset, uint32_t& sliceBytes, uint32_t& dataAsFlagOffset)
{
    uint32_t totalBlocks = TileXR::DataAsFlagBlockCountForPayloadBytes(bytes);
    uint32_t perBlock = TileXRUdmaDemoCeilDiv(totalBlocks, blockNum);
    uint32_t startBlock = blockIdx * perBlock;
    if (blockNum == 0 || perBlock == 0 || startBlock >= totalBlocks) {
        payloadOffset = bytes;
        sliceBytes = 0;
        dataAsFlagOffset = totalBlocks * TileXR::DATA_AS_FLAG_BLOCK_BYTES;
        return;
    }
    uint32_t blockCount = totalBlocks - startBlock;
    if (blockCount > perBlock) {
        blockCount = perBlock;
    }
    payloadOffset = startBlock * TileXR::DATA_AS_FLAG_PAYLOAD_BYTES;
    uint32_t maxPayloadBytes = blockCount * TileXR::DATA_AS_FLAG_PAYLOAD_BYTES;
    sliceBytes = bytes - payloadOffset;
    if (sliceBytes > maxPayloadBytes) {
        sliceBytes = maxPayloadBytes;
    }
    dataAsFlagOffset = startBlock * TileXR::DATA_AS_FLAG_BLOCK_BYTES;
}

__aicore__ inline void TileXRUdmaDemoCopyBytesGmToGm(
    GM_ADDR dstGM, GM_ADDR srcGM, AscendC::TBuf<AscendC::QuePosition::VECCALC>& tBuf, uint32_t bytes)
{
    if (dstGM == nullptr || srcGM == nullptr || bytes == 0) {
        return;
    }

    AscendC::LocalTensor<uint8_t> local =
        tBuf.GetWithOffset<uint8_t>(TILEXR_UDMA_DEMO_P2P_COPY_TILE_BYTES, TILEXR_UDMA_DEMO_P2P_SYNC_UB_BYTES);
    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcGM), bytes);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstGM), bytes);

    for (uint32_t copied = 0; copied < bytes; copied += TILEXR_UDMA_DEMO_P2P_COPY_TILE_BYTES) {
        uint32_t tileBytes = bytes - copied;
        if (tileBytes > TILEXR_UDMA_DEMO_P2P_COPY_TILE_BYTES) {
            tileBytes = TILEXR_UDMA_DEMO_P2P_COPY_TILE_BYTES;
        }
        AscendC::DataCopyPadParams padParams {false, 0, 0, 0};
        AscendC::DataCopyParams copyParams {1, static_cast<uint16_t>(tileBytes), 0, 0};
        AscendC::DataCopyPad(local, src[copied], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[copied], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

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

extern "C" __global__ __aicore__ void tilexr_udma_p2p_perf_kernel(
    GM_ADDR commArgsGM, GM_ADDR srcGM, GM_ADDR debugGM,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
    auto debug = reinterpret_cast<__gm__ uint32_t*>(debugGM);

    int32_t rank = args->rank;
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();
    bool enabled = TileXR::UDMARegistryEnabled(args);
    uint32_t qpNum = enabled ? TileXR::GetUDMAInfo(args)->qpNum : 0;
    if (debug != nullptr && blockIdx == 0) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = enabled ? 1 : 0;
        debug[3] = bytes;
        debug[4] = pattern;
        debug[5] = 0xffffffffu;
        debug[6] = blockNum;
        debug[7] = qpNum;
    }

    int32_t peer = -1;
    if (!enabled || blockNum == 0 || qpNum == 0 ||
        !TileXRUdmaDemoResolvePeer(rank, srcRank, dstRank, traffic, peer)) {
        return;
    }
    uint32_t jettyCount = blockNum < qpNum ? blockNum : qpNum;
    if (blockIdx >= jettyCount) {
        TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 0);
        return;
    }

    uint32_t offset = 0;
    uint32_t sliceBytes = 0;
    TileXRUdmaDemoWqeSlice(bytes, jettyCount, blockIdx, offset, sliceBytes);
    if (debug != nullptr && blockIdx < 8) {
        debug[16 + blockIdx] = offset;
        debug[24 + blockIdx] = sliceBytes;
    }
    if (sliceBytes == 0) {
        TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 0);
        return;
    }
    TileXR::UDMAPutNbiQp<uint8_t>(args, peer, blockIdx, src + offset, dstByteOffset + offset, sliceBytes);
    uint32_t status = TileXR::UDMAQuietStatusQp(args, peer, blockIdx);
    TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, status);
    if (debug != nullptr && blockIdx == 0) {
        debug[5] = status;
        debug[15] = jettyCount;
    }
}

extern "C" __global__ __aicore__ void tilexr_memory_p2p_perf_kernel(
    GM_ADDR commArgsGM, GM_ADDR srcGM, GM_ADDR debugGM,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
        auto debug = reinterpret_cast<__gm__ uint32_t*>(debugGM);

        int32_t rank = args->rank;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (debug != nullptr && blockIdx == 0) {
            debug[0] = TILEXR_UDMA_DEMO_MAGIC;
            debug[1] = rank;
            debug[2] = 1;
            debug[3] = bytes;
            debug[4] = pattern;
            debug[5] = 0xffffffffu;
        }

        int32_t peer = -1;
        if (blockNum == 0 || !TileXRUdmaDemoResolvePeer(rank, srcRank, dstRank, traffic, peer) ||
            peer < 0 || peer >= args->rankSize) {
            return;
        }

        AscendC::GlobalTensor<GM_ADDR> peerMems;
        peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
        GM_ADDR dstBase = peerMems.GetValue(peer);
        if (dstBase == nullptr) {
            uint32_t status = TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 2);
            if (debug != nullptr && blockIdx == 0) {
                debug[5] = status;
            }
            return;
        }

        uint32_t offset = 0;
        uint32_t sliceBytes = 0;
        TileXRUdmaDemoBlockSlice(bytes, blockNum, blockIdx, offset, sliceBytes);
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_P2P_UB_BYTES);
        TileXRUdmaDemoCopyBytesGmToGm(
            dstBase + dstByteOffset + offset, srcGM + offset, tBuf, sliceBytes);
        TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 0);
        if (debug != nullptr && blockIdx == 0) {
            debug[5] = 0;
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_memory_visible_ack_p2p_perf_kernel(
    GM_ADDR commArgsGM, GM_ADDR srcGM, GM_ADDR debugGM,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic, uint32_t token)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
        auto debug = reinterpret_cast<__gm__ uint32_t*>(debugGM);

        int32_t rank = args->rank;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (debug != nullptr && blockIdx == 0) {
            debug[0] = TILEXR_UDMA_DEMO_MAGIC;
            debug[1] = rank;
            debug[2] = 1;
            debug[3] = bytes;
            debug[4] = pattern;
            debug[5] = 0xffffffffu;
        }

        bool isSender = false;
        bool isReceiver = false;
        int32_t peer = -1;
        if (blockNum == 0 ||
            !TileXRUdmaDemoResolveDataAsFlagRole(rank, srcRank, dstRank, traffic, isSender, isReceiver, peer) ||
            peer < 0 || peer >= args->rankSize) {
            return;
        }

        AscendC::GlobalTensor<GM_ADDR> peerMems;
        peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
        GM_ADDR peerBase = peerMems.GetValue(peer);
        GM_ADDR localBase = peerMems.GetValue(rank);
        if (peerBase == nullptr || localBase == nullptr) {
            uint32_t status = TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 2);
            if (debug != nullptr && blockIdx == 0) {
                debug[5] = status;
            }
            return;
        }

        uint32_t offset = 0;
        uint32_t sliceBytes = 0;
        TileXRUdmaDemoBlockSlice(bytes, blockNum, blockIdx, offset, sliceBytes);
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_P2P_UB_BYTES);

        uint32_t status = 0;
        const uint32_t ackOffset = TileXRUdmaDemoMemoryVisibleAckOffset(bytes) +
            blockIdx * TILEXR_UDMA_DEMO_MEMORY_VISIBLE_ACK_BYTES;
        const uint32_t ackValue = TileXRUdmaDemoMemoryVisibleAckValue(pattern, bytes, blockIdx, token);
        if (isSender) {
            TileXRUdmaDemoCopyBytesGmToGm(
                peerBase + dstByteOffset + offset, srcGM + offset, tBuf, sliceBytes);
            TileXRUdmaDemoWriteMemoryVisibleAck(peerBase + dstByteOffset + ackOffset, tBuf, ackValue);
        }
        if (isReceiver && status == 0) {
            GM_ADDR ackAddr = localBase + dstByteOffset + ackOffset;
            while (!TileXRUdmaDemoCheckMemoryVisibleAck(ackAddr, tBuf, ackValue)) {
            }
        }

        TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, status);
        if (debug != nullptr && blockIdx == 0) {
            debug[5] = status;
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_data_as_flag_p2p_perf_kernel(
    GM_ADDR commArgsGM, GM_ADDR srcGM, GM_ADDR dstGM, GM_ADDR debugGM,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern, int32_t traffic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
        auto debug = reinterpret_cast<__gm__ uint32_t*>(debugGM);
        auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
        auto dst = reinterpret_cast<__gm__ uint8_t*>(dstGM);

        int32_t rank = args->rank;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (debug != nullptr && blockIdx == 0) {
            debug[0] = TILEXR_UDMA_DEMO_MAGIC;
            debug[1] = rank;
            debug[2] = 1;
            debug[3] = bytes;
            debug[4] = pattern;
            debug[5] = 0xffffffffu;
        }

        bool isSender = false;
        bool isReceiver = false;
        int32_t peer = -1;
        if (blockNum == 0 ||
            !TileXRUdmaDemoResolveDataAsFlagRole(rank, srcRank, dstRank, traffic, isSender, isReceiver, peer) ||
            peer < 0 || peer >= args->rankSize) {
            return;
        }

        AscendC::GlobalTensor<GM_ADDR> peerMems;
        peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
        GM_ADDR peerBase = peerMems.GetValue(peer);
        GM_ADDR localBase = peerMems.GetValue(rank);
        if (peerBase == nullptr || localBase == nullptr || dst == nullptr) {
            uint32_t status = TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 2);
            if (debug != nullptr && blockIdx == 0) {
                debug[5] = status;
            }
            return;
        }

        uint32_t payloadOffset = 0;
        uint32_t sliceBytes = 0;
        uint32_t dataAsFlagOffset = 0;
        TileXRUdmaDemoDataAsFlagSlice(bytes, blockNum, blockIdx, payloadOffset, sliceBytes, dataAsFlagOffset);
        if (sliceBytes == 0) {
            TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, 0);
            return;
        }

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_P2P_UB_BYTES);
        AscendC::LocalTensor<uint8_t> scratch = tBuf.Get<uint8_t>();

        uint32_t status = 0;
        if (isSender) {
            uint32_t initBlocks = TileXR::DataAsFlagInit(scratch);
            uint32_t sentBlocks = TileXR::DataAsFlagSend(
                reinterpret_cast<__gm__ uint8_t*>(peerBase + dstByteOffset + dataAsFlagOffset),
                src + payloadOffset, sliceBytes, scratch);
            if (initBlocks == 0 || sentBlocks == 0) {
                status = 3;
            }
        }
        if (isReceiver && status == 0) {
            bool received = TileXR::DataAsFlagCheckAndRecv(
                reinterpret_cast<__gm__ uint8_t*>(localBase + dstByteOffset + dataAsFlagOffset),
                sliceBytes, dst + payloadOffset, scratch);
            if (!received) {
                status = 4;
            }
        }
        TileXRUdmaDemoFoldDebugStatus(debug, blockIdx, status);
        if (debug != nullptr && blockIdx == 0) {
            debug[5] = status;
        }
    }
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

void launch_tilexr_udma_registered_smoke(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR local, GM_ADDR debug, uint32_t bytes, uint64_t signal)
{
    tilexr_udma_registered_smoke_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, local, debug, bytes, signal);
}

void launch_tilexr_udma_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic)
{
    tilexr_udma_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, debug, srcRank, dstRank, dstByteOffset, bytes, pattern, traffic);
}

void launch_tilexr_memory_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic)
{
    tilexr_memory_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, debug, srcRank, dstRank, dstByteOffset, bytes, pattern, traffic);
}

void launch_tilexr_memory_visible_ack_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic,
    uint32_t token)
{
    tilexr_memory_visible_ack_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, debug, srcRank, dstRank, dstByteOffset, bytes, pattern, traffic, token);
}

void launch_tilexr_data_as_flag_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR dst, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic)
{
    tilexr_data_as_flag_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, dst, debug, srcRank, dstRank, dstByteOffset, bytes, pattern, traffic);
}
