/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "comm_args.h"
#include "kernel_operator.h"
#include "tilexr_sync.h"

namespace {
constexpr int32_t TILEXR_MEMORY_DEMO_MAGIC = 0x544d454d; // "TMEM"
constexpr int32_t TILEXR_MEMORY_DEMO_STEP_READY = 1;
constexpr uint32_t TILEXR_MEMORY_DEMO_UB_BYTES = 64 * 1024;
constexpr uint32_t TILEXR_MEMORY_DEMO_SYNC_UB_BYTES = 4 * 1024;

template <typename T>
__aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    return CeilDiv<T>(value, alignment) * alignment;
}

template <typename T>
__aicore__ inline void GetBlockSlice(
    int64_t total, int64_t blockNum, int64_t blockIdx, int64_t& offset, int64_t& count)
{
    int64_t perBlock = CeilDiv<T>(total, blockNum);
    int64_t minAligned = TileXR::BLOCK_UNIT_BYTE / static_cast<int64_t>(sizeof(T));
    if (perBlock < minAligned) {
        perBlock = minAligned;
    }
    perBlock = AlignUp<T>(perBlock, minAligned);

    offset = blockIdx * perBlock;
    if (offset >= total) {
        offset = total;
        count = 0;
        return;
    }
    count = perBlock;
    if (offset + count > total) {
        count = total - offset;
    }
}

template <typename T>
__aicore__ inline void CopyGmToGm(
    AscendC::GlobalTensor<T>& dst, AscendC::GlobalTensor<T>& src, AscendC::TBuf<AscendC::QuePosition::VECCALC>& tBuf,
    int64_t count)
{
    if (count <= 0) {
        return;
    }

    constexpr int64_t kTileElements = (TILEXR_MEMORY_DEMO_UB_BYTES - TILEXR_MEMORY_DEMO_SYNC_UB_BYTES) / sizeof(T);
    AscendC::LocalTensor<T> local = tBuf.GetWithOffset<T>(kTileElements, TILEXR_MEMORY_DEMO_SYNC_UB_BYTES);
    for (int64_t copied = 0; copied < count; copied += kTileElements) {
        int64_t tile = count - copied;
        if (tile > kTileElements) {
            tile = kTileElements;
        }
        AscendC::DataCopy(local, src[copied], static_cast<uint32_t>(tile));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dst[copied], local, static_cast<uint32_t>(tile));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
} // namespace

extern "C" __global__ __aicore__ void tilexr_memory_all_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
        auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
        auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
        auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

        int32_t rank = args->rank;
        int32_t rankSize = args->rankSize;
        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t blockNum = AscendC::GetBlockNum();
        if (debug != nullptr && blockIdx == 0) {
            debug[0] = TILEXR_MEMORY_DEMO_MAGIC;
            debug[1] = rank;
            debug[2] = rankSize;
            debug[3] = elementsPerRank;
            debug[4] = blockNum;
        }
        if (elementsPerRank <= 0 || rankSize <= 0 || rank < 0 || rank >= rankSize) {
            return;
        }

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, TILEXR_MEMORY_DEMO_UB_BYTES);

        GM_ADDR shareAddrs[TileXR::TILEXR_MAX_RANK_SIZE];
        AscendC::GlobalTensor<GM_ADDR> peerMems;
        peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
        for (int32_t peer = 0; peer < rankSize; ++peer) {
            shareAddrs[peer] = peerMems.GetValue(peer);
        }

        int64_t localOffset = 0;
        int64_t localCount = 0;
        GetBlockSlice<int32_t>(elementsPerRank, blockNum, blockIdx, localOffset, localCount);

        AscendC::GlobalTensor<int32_t> inputTensor;
        AscendC::GlobalTensor<int32_t> shareTensor;
        inputTensor.SetGlobalBuffer(input + localOffset, localCount);
        shareTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shareAddrs[rank] + TileXR::IPC_DATA_OFFSET) +
                                        localOffset,
            localCount);
        CopyGmToGm<int32_t>(shareTensor, inputTensor, tBuf, localCount);

        SyncCollectives sync;
        sync.Init(rank, rankSize, shareAddrs, tBuf);
        sync.SetInnerFlag(TILEXR_MEMORY_DEMO_MAGIC, TILEXR_MEMORY_DEMO_STEP_READY);

        int32_t blocksPerRank = blockNum / rankSize;
        if (blocksPerRank <= 0) {
            blocksPerRank = 1;
        }
        int32_t activeBlocks = blocksPerRank * rankSize;
        if (blockIdx >= activeBlocks) {
            return;
        }

        int32_t sourceRank = blockIdx / blocksPerRank;
        int32_t sourceBlockIdx = blockIdx - sourceRank * blocksPerRank;
        int64_t remoteOffset = 0;
        int64_t remoteCount = 0;
        GetBlockSlice<int32_t>(elementsPerRank, blocksPerRank, sourceBlockIdx, remoteOffset, remoteCount);

        sync.WaitRankInnerFlag(TILEXR_MEMORY_DEMO_MAGIC, TILEXR_MEMORY_DEMO_STEP_READY, sourceRank);

        AscendC::GlobalTensor<int32_t> remoteTensor;
        AscendC::GlobalTensor<int32_t> outputTensor;
        remoteTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shareAddrs[sourceRank] + TileXR::IPC_DATA_OFFSET) +
                                         remoteOffset,
            remoteCount);
        outputTensor.SetGlobalBuffer(output + static_cast<int64_t>(sourceRank) * elementsPerRank + remoteOffset,
            remoteCount);
        CopyGmToGm<int32_t>(outputTensor, remoteTensor, tBuf, remoteCount);
    }
}

void launch_tilexr_memory_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output, GM_ADDR debug,
    int32_t elementsPerRank)
{
    tilexr_memory_all_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerRank);
}
