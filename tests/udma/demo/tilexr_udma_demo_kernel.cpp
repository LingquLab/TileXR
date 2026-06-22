/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_udma.h"

constexpr int32_t TILEXR_UDMA_DEMO_MAGIC = 0x5444554d; // "TDUM"
constexpr uint32_t TILEXR_UDMA_DEMO_P2P_UB_BYTES = 64 * 1024;
constexpr uint32_t TILEXR_UDMA_DEMO_P2P_SYNC_UB_BYTES = 4 * 1024;
constexpr uint32_t TILEXR_UDMA_DEMO_P2P_COPY_TILE_BYTES =
    TILEXR_UDMA_DEMO_P2P_UB_BYTES - TILEXR_UDMA_DEMO_P2P_SYNC_UB_BYTES;

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
    uint32_t bytes, uint32_t pattern)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
    auto debug = reinterpret_cast<__gm__ uint32_t*>(debugGM);

    int32_t rank = args->rank;
    bool enabled = TileXR::UDMARegistryEnabled(args);
    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = enabled ? 1 : 0;
        debug[3] = bytes;
        debug[4] = pattern;
        debug[5] = 0xffffffffu;
    }
    if (!enabled || rank != srcRank) {
        return;
    }

    TileXR::UDMAPutNbi<uint8_t>(args, dstRank, src, dstByteOffset, bytes);
    uint32_t status = TileXR::UDMAQuietStatus(args, dstRank);
    if (debug != nullptr) {
        debug[5] = status;
    }
}

extern "C" __global__ __aicore__ void tilexr_memory_p2p_perf_kernel(
    GM_ADDR commArgsGM, GM_ADDR srcGM, GM_ADDR debugGM,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern)
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

        if (rank != srcRank || blockNum == 0 || dstRank < 0 || dstRank >= args->rankSize) {
            return;
        }

        AscendC::GlobalTensor<GM_ADDR> peerMems;
        peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
        GM_ADDR dstBase = peerMems.GetValue(dstRank);
        if (dstBase == nullptr) {
            if (debug != nullptr && blockIdx == 0) {
                debug[5] = 2;
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
        if (debug != nullptr && blockIdx == 0) {
            debug[5] = 0;
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
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern)
{
    tilexr_udma_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, debug, srcRank, dstRank, dstByteOffset, bytes, pattern);
}

void launch_tilexr_memory_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern)
{
    tilexr_memory_p2p_perf_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, src, debug, srcRank, dstRank, dstByteOffset, bytes, pattern);
}
