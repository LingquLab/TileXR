/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "comm_args.h"
#include "kernel_operator.h"

namespace {
constexpr int32_t TILEXR_MEMORY_DEMO_MAGIC = 0x544d454d; // "TMEM"
constexpr int32_t TILEXR_MEMORY_DEMO_STEP_PUSH = 1;
constexpr int32_t TILEXR_MEMORY_DEMO_STEP_COLLECT = 2;
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
        const uint32_t tileBytes = static_cast<uint32_t>(tile * static_cast<int64_t>(sizeof(T)));
        AscendC::DataCopyParams copyParams {1, static_cast<uint16_t>(tileBytes), 0, 0};
        AscendC::DataCopyPadParams padParams {false, 0, 0, 0};
        AscendC::DataCopyPad(local, src[copied], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[copied], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool LoadMemoryDemoArgs(
    GM_ADDR commArgsGM, int32_t elementsPerRank, int32_t& rank, int32_t& rankSize, GM_ADDR* shareAddrs)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    rank = args->rank;
    rankSize = args->rankSize;
    if (elementsPerRank <= 0 || rankSize <= 0 || rank < 0 || rank >= rankSize ||
        rankSize > TileXR::TILEXR_MAX_RANK_SIZE) {
        return false;
    }

    AscendC::GlobalTensor<GM_ADDR> peerMems;
    peerMems.SetGlobalBuffer(&(args->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        shareAddrs[peer] = peerMems.GetValue(peer);
        if (shareAddrs[peer] == nullptr) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void WriteMemoryDemoDebug(
    __gm__ int32_t* debug, int32_t rank, int32_t rankSize, int32_t elementsPerRank, int32_t blockNum,
    int32_t step)
{
    if (debug == nullptr || AscendC::GetBlockIdx() != 0) {
        return;
    }
    debug[0] = TILEXR_MEMORY_DEMO_MAGIC;
    debug[1] = rank;
    debug[2] = rankSize;
    debug[3] = elementsPerRank;
    debug[4] = blockNum;
    debug[5] = step;
}
} // namespace

extern "C" __global__ __aicore__ void tilexr_memory_push_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
        auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t blockNum = AscendC::GetBlockNum();
        int32_t rank = 0;
        int32_t rankSize = 0;
        GM_ADDR shareAddrs[TileXR::TILEXR_MAX_RANK_SIZE];
        if (!LoadMemoryDemoArgs(commArgsGM, elementsPerRank, rank, rankSize, shareAddrs)) {
            return;
        }
        WriteMemoryDemoDebug(debug, rank, rankSize, elementsPerRank, blockNum, TILEXR_MEMORY_DEMO_STEP_PUSH);

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, TILEXR_MEMORY_DEMO_UB_BYTES);

        int64_t localOffset = 0;
        int64_t localCount = 0;
        GetBlockSlice<int32_t>(elementsPerRank, blockNum, blockIdx, localOffset, localCount);

        AscendC::GlobalTensor<int32_t> inputTensor;
        inputTensor.SetGlobalBuffer(input + localOffset, localCount);
        for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
            AscendC::GlobalTensor<int32_t> shareTensor;
            shareTensor.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(shareAddrs[dstRank] + TileXR::IPC_DATA_OFFSET) +
                    static_cast<int64_t>(rank) * elementsPerRank + localOffset,
                localCount);
            CopyGmToGm<int32_t>(shareTensor, inputTensor, tBuf, localCount);
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_memory_collect_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    if constexpr (g_coreType == AscendC::AIV) {
        auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
        auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t blockNum = AscendC::GetBlockNum();
        int32_t rank = 0;
        int32_t rankSize = 0;
        GM_ADDR shareAddrs[TileXR::TILEXR_MAX_RANK_SIZE];
        if (!LoadMemoryDemoArgs(commArgsGM, elementsPerRank, rank, rankSize, shareAddrs)) {
            return;
        }
        WriteMemoryDemoDebug(debug, rank, rankSize, elementsPerRank, blockNum, TILEXR_MEMORY_DEMO_STEP_COLLECT);

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, TILEXR_MEMORY_DEMO_UB_BYTES);

        const int64_t totalElements = static_cast<int64_t>(rankSize) * elementsPerRank;
        int64_t localOffset = 0;
        int64_t localCount = 0;
        GetBlockSlice<int32_t>(totalElements, blockNum, blockIdx, localOffset, localCount);
        if (localCount <= 0) {
            return;
        }

        AscendC::GlobalTensor<int32_t> shareTensor;
        AscendC::GlobalTensor<int32_t> outputTensor;
        shareTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(shareAddrs[rank] + TileXR::IPC_DATA_OFFSET) + localOffset, localCount);
        outputTensor.SetGlobalBuffer(output + localOffset, localCount);
        CopyGmToGm<int32_t>(outputTensor, shareTensor, tBuf, localCount);
    }
}

void launch_tilexr_memory_push(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output, GM_ADDR debug,
    int32_t elementsPerRank)
{
    tilexr_memory_push_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerRank);
}

void launch_tilexr_memory_collect(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output, GM_ADDR debug,
    int32_t elementsPerRank)
{
    tilexr_memory_collect_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerRank);
}
