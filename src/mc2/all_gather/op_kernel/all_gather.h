/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_add.h
 * \brief
 */
#ifndef ALL_GATHER_H
#define ALL_GATHER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_tiling.h"
#include "all_gather_tiling_key.h"
// #include "../../../../../common/include/kernel/tilexr_sync.h"
#include "tilexr_sync.h"
#include "comm_args.h"

constexpr int32_t ALLGATHER_BUFFER_NUM = 1;
constexpr int SERVER_DIM = 8;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr static uint32_t RDMA_DATA_SIZE = 100U * 1024U * 1024U;
constexpr static uint32_t IPC_DATA_OFFSET = 2U * 1024U * 1024U;
constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr static uint32_t UB_SIZE = 128U * 1024U;
constexpr static uint32_t RDMA_PER_RANK_SIZE = 4U * 1024U * 1024U;
constexpr static uint32_t STATE_SPACE_SIZE = 1024U * 1024U;
constexpr static uint32_t STATE_OFFSET = 512U;
constexpr int UB_ALIGN_SIZE = 32;
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t MEM_DMA_UNIT_SIZE = MEM_DMA_UNIT_INT_NUM * sizeof(int64_t);
constexpr int64_t STEP1 = 1;  // 算法步骤1

namespace AscendC {

template <typename T>
class AllGather {
public:
    __aicore__ inline AllGather(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR gatherGM, 
                                GM_ADDR workspaceGM, AllGatherTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GetBlockDataCount(
        const int64_t dataLen, const int64_t useBlockNum, int64_t& blockDataOffset, int64_t& blockDataCount);
    __aicore__ inline void CopyGM2GM(GlobalTensor<T> outputGM, GlobalTensor<T> inputGM, int copyNum);

private:

    AllGatherTilingData *tilingData_;

    TPipe *tPipe_;

    TQue<QuePosition::VECIN, ALLGATHER_BUFFER_NUM> inputQueue;
    TBuf<QuePosition::VECCALC> tBuf;

    GlobalTensor<T> inputAGM;
    GlobalTensor<T> gatherOutGM;
    GlobalTensor<T> shareGm;

    GlobalTensor<uint32_t> bufferIdGlobal_;
    GlobalTensor<uint32_t> outBufferIdGlobal_;
    GlobalTensor<uint64_t> magicGlobal_;
    GlobalTensor<uint32_t> statusSpaceGlobal_;

    int64_t blockElemNum_ = 0;
    int64_t tileNum_ = 0;
    uint32_t addTileElemNum_ = 0;

    int rankId;
    int rankSize;
    int32_t magic;

    GM_ADDR shareAddrs[SERVER_DIM];
    uint32_t dataSpaceSize_{0};
    uint32_t halfWinSize_{0};
    uint64_t magicValue{0};
    uint32_t bufferId_{0};


    int64_t baseOffsetSize;  // 共享数据区起点的偏移（Bytes）
    // step1数据切片
    int64_t offsetFromInput;  // 从input拷贝数据的地址偏移
    int64_t offsetToShare;  // 拷贝至share[rank]数据的地址偏移
    int64_t countToShare;  // 拷贝至share[rank]数据的个数
    // step2数据切片
    int64_t useCoreNumToOutput;  // 搬运数据至output阶段使用的core数
    int64_t blockNumPerRank;  // 单个rank负责搬运数据的core数量
    int64_t blockRank;  // 当前core负责搬运数据的rank
    int64_t offsetFromShare;;  // 从share[blockRank]拷贝数据的地址偏移
    int64_t offsetToOutput;  // 拷贝至output数据的地址偏移
    int64_t countToOutput;  // 拷贝至output数据的个数
    int globalRank;
    int globalRankSize;
    int localRankSize;

    int64_t blockNum;  // 总aicore数
    int blockIdx;


    SyncCollectives sync;

    GM_ADDR commArgs;
};

template <typename T>
__aicore__ inline void AllGather<T>::Init(GM_ADDR aGM, GM_ADDR gatherGM, 
                                          GM_ADDR workspaceGM, AllGatherTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    blockElemNum_ = tilingData->blockElemNum;
    tileNum_ = tilingData->tileNum;
    blockIdx = AscendC::GetBlockIdx();
    addTileElemNum_ = UB_SIZE  / 2 / sizeof(T);
    magic = 12345;
    baseOffsetSize = IPC_DATA_OFFSET;
    commArgs = (GM_ADDR) (tilingData_->commDataPtr);



    // 初始化hccl对象
    // hccl_.InitV2(contextGM, tilingData);
    // hccl_.SetCcTilingV2(offsetof(AllGatherTilingData, mc2CcTiling));

    rankId = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs)->localRank;
    rankSize = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs)->localRankSize;

    
    tPipe_->InitBuffer(inputQueue, ALLGATHER_BUFFER_NUM, addTileElemNum_ * sizeof(half));
    tPipe_->InitBuffer(tBuf, UB_SIZE/2);

    GlobalTensor<GM_ADDR> peerMemsAddrGm;
    peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs))->peerMems[0], TileXR::TILEXR_MAX_RANK_SIZE);
    for (int i = 0; i < rankSize; ++i) {
        shareAddrs[i] = peerMemsAddrGm.GetValue(i) +
                        (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
    }

    blockNum = AscendC::GetBlockNum();
    // 计算step1数据分片，input-->share，所有core参与搬运
    GetBlockDataCount(tilingData->gatherTileElemNum, blockNum, offsetFromInput, countToShare);
    offsetToShare = offsetFromInput;
    inputAGM.SetGlobalBuffer((__gm__ T*)aGM + offsetFromInput, countToShare); // 非多轮切分AllGather场景，每张卡参与Gather的数据大小为{240，256}

    // 计算step2数据分片，share-->output，当前core负责一个rank的部分或全部数据搬运
    blockNumPerRank = blockNum / rankSize;  // 均分core至每个rank，多余的core不使用
    useCoreNumToOutput = blockNumPerRank * rankSize;

    if (blockIdx >= useCoreNumToOutput) {
        // 不用的核直接退出
        return;
    }
    
    GetBlockDataCount(tilingData->gatherTileElemNum, blockNumPerRank, offsetFromShare, countToOutput);
    blockRank = blockIdx / blockNumPerRank;

    offsetToOutput = blockRank * tilingData->gatherTileElemNum + offsetFromShare;

    // 当前block的output分片
    
    gatherOutGM.SetGlobalBuffer((__gm__ T*)gatherGM + offsetToOutput, countToOutput);
    sync.Init(rankId, rankSize, shareAddrs, tBuf);
}

template <typename T>
__aicore__ inline void AllGather<T>::GetBlockDataCount(
        const int64_t dataLen, const int64_t useBlockNum, int64_t& blockDataOffset, int64_t& blockDataCount)
{
    // 向上整除获取每个core切分的数据个数
    blockDataCount = (dataLen + useBlockNum - 1) / useBlockNum;
    // 设置每个core数据下限
    blockDataCount = blockDataCount > MEM_DMA_UNIT_SIZE / sizeof(half) ?
                        blockDataCount : MEM_DMA_UNIT_SIZE / sizeof(half);
    // 极小数据量情况，core分配到数据下限，后面若干个core数据量为0
    blockDataOffset = blockIdx % useBlockNum * blockDataCount;  // 使用当前block在useBlock里的相对index计算偏移
    if (blockDataOffset >= dataLen) {
        blockDataOffset = dataLen;
        blockDataCount = 0;
        return;
    }
    // 非整除情况，最后一个core数据量为剩余数据量
    if (blockDataOffset + blockDataCount > dataLen) {
        blockDataCount = dataLen - blockDataOffset;
    }
}

template <typename T>
__aicore__ inline void AllGather<T>::CopyGM2GM(GlobalTensor<T> outputGM, GlobalTensor<T> inputGM, int copyNum)
{
    int copyTimes = copyNum / addTileElemNum_;
    int remaining =  copyNum % addTileElemNum_;
    if (remaining > 0) {
        copyTimes++;
    }
    for (int j = 0 ; j < copyTimes; j++) { // copyTimes
        int offset = j*addTileElemNum_;
        int copyLength;
        if (remaining > 0 && (j == (copyTimes -1))){
            copyLength = copyNum -  offset;
        } else {
            copyLength = addTileElemNum_;
        }
        AscendC::LocalTensor<T> reduceLocalIn = inputQueue.AllocTensor<T>();
        AscendC::DataCopy(reduceLocalIn, inputGM[j*addTileElemNum_], copyLength); //copyLength
        inputQueue.EnQue(reduceLocalIn);
        AscendC::LocalTensor<T> reduceLocalOut = inputQueue.DeQue<T>();
        PipeBarrier<PIPE_ALL>(); //PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(outputGM[j*addTileElemNum_], reduceLocalOut, copyLength); //copyLength
        inputQueue.FreeTensor(reduceLocalOut);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void AllGather<T>::Process()
{
    if constexpr (g_coreType == AscendC::AIV)
    {
        // step1：拷贝input至共享内存
        shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[rankId] + baseOffsetSize) + offsetToShare, countToShare);
        if (countToShare > 0) {
            CopyGM2GM(shareGm, inputAGM, countToShare);
        }

        // 卡内同步，确保数据已拷贝至共享内存
        sync.SetInnerFlag(magic, STEP1);  // 当前rank当前核的数据搬运已完成
        sync.WaitRankInnerFlag(magic, STEP1, blockRank);  // 等待目标rank的数据全部搬运完成
        // step2：拷贝共享内存至output
        // if (rankId == 1 && blockIdx == 0) {
        //     AscendC::DumpTensor(shareGm[0], 6541000 + rankId * 10 + blockIdx, 64);
        // }
        if (blockIdx >= useCoreNumToOutput) {
            // 不用的核直接退出
            return;
        }
        shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[blockRank] + baseOffsetSize) + offsetFromShare,
            countToOutput);
        if (countToOutput > 0) {
            CopyGM2GM(gatherOutGM, shareGm, countToOutput);
        }
    }
}
}
#endif