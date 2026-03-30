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
 * \file all_gather.h
 * \brief
 */
#ifndef ALL_GATHER_H
#define ALL_GATHER_H

#include <hccl/hccl_types.h>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_tiling.h"
#include "all_gather_tiling_key.h"
#include "tilexr_sync.h"
#include "comm_args.h"
#include "moe_distribute_base.h"

constexpr int32_t ALLGATHER_BUFFER_NUM = 1;
constexpr int SERVER_DIM = 8;
constexpr int64_t PING_PONG_SIZE = 2;
const uint32_t HCCL_AIV_STREAM_NUM = 32;
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
constexpr int64_t STEP2 = 2;  // 算法步骤2

namespace AscendC {

template <typename T>
class AllGather {
public:
    __aicore__ inline AllGather(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR gatherGM, 
                                GM_ADDR workspaceGM, GM_ADDR contextGM, AllGatherTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void MemCopySQE(GM_ADDR srcAddr, GM_ADDR dstAddr, uint32_t length, uint32_t queueIdx);
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value);
    __aicore__ inline int32_t SplitMagicWithValue(int64_t value);

private:
    __aicore__ inline void GetBlockDataCount(
        const int64_t dataLen, const int64_t useBlockNum, int64_t& blockDataOffset, int64_t& blockDataCount, int blockId);
    __aicore__ inline void CopyGM2GM(GlobalTensor<T> outputGM, GlobalTensor<T> inputGM, int copyNum);

private:

    AllGatherTilingData *tilingData_;

    TPipe *tPipe_;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

    TQue<QuePosition::VECIN, ALLGATHER_BUFFER_NUM> inputQueue;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<TPosition::VECCALC> tBufSqe_;
    LocalTensor<uint8_t> sqeUb_;
    TBuf<TPosition::VECCALC> tBufTail_;
    LocalTensor<uint32_t> tailUb_;

    GlobalTensor<T> inputAGM;
    GlobalTensor<T> gatherOutGM;
    GlobalTensor<T> shareGm;

    GlobalTensor<uint32_t> bufferIdGlobal_;
    GlobalTensor<uint32_t> outBufferIdGlobal_;
    GlobalTensor<uint64_t> magicGlobal_;
    GlobalTensor<uint64_t> magic2Global_;
    GlobalTensor<uint32_t> statusSpaceGlobal_;

    int64_t blockElemNum_ = 0;
    int64_t tileNum_ = 0;
    uint32_t addTileElemNum_ = 0;

    int rankId;
    int rankSize;
    int32_t magic;

    GM_ADDR shareAddrs[SERVER_DIM];

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
    GM_ADDR aGM_;
    GM_ADDR gatherGM_;
    __gm__ HcclA2CombineOpParam *winContext_;
    __gm__ TileXrContext *tileXrContext;
};

template <typename T>
__aicore__ inline int64_t AllGather<T>::MergeMagicWithValue(int32_t magic, int32_t value)
{
    // magic作为高位，eventID作为低位，组成一个value值用于比较
    return (static_cast<int64_t>(magic) << MAGIC_OFFSET) | static_cast<int64_t>(value);
}

template <typename T>
__aicore__ inline int32_t AllGather<T>::SplitMagicWithValue(int64_t value)
{
    // magic作为高位，eventID作为低位，组成一个value值用于比较
    return (static_cast<int32_t>((value) >> MAGIC_OFFSET));
}

template <typename T>
__aicore__ inline void AllGather<T>::Init(GM_ADDR aGM, GM_ADDR gatherGM, 
                                          GM_ADDR workspaceGM, GM_ADDR contextGM, AllGatherTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    blockElemNum_ = tilingData->blockElemNum;
    tileNum_ = tilingData->tileNum;
    blockIdx = AscendC::GetBlockIdx();
    addTileElemNum_ = UB_SIZE  / 2 / sizeof(T);
    baseOffsetSize = IPC_DATA_OFFSET;
    commArgs = (GM_ADDR) (tilingData_->commDataPtr);

    // 初始化hccl对象
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(AllGatherTilingData, mc2CcTiling));

    rankId = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs)->localRank;
    rankSize = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs)->localRankSize;

    winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM;
    if (blockIdx == 0) {
		tileXrContext = (__gm__ TileXrContext *)winContext_->tileXrContext;

		if (!tileXrContext) return;

		//reset status
		for (uint32_t i = 0; i < tileXrContext->streamInfo.streamNum; i++) {
			tileXrContext->streamInfo.statusList[i] = 0;
		}
	}
    AscendC::SyncAll();
    
    tPipe_->InitBuffer(inputQueue, ALLGATHER_BUFFER_NUM, addTileElemNum_ * sizeof(half));
    tPipe_->InitBuffer(tBuf, UB_SIZE/2);

    magicGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(hccl_.GetWindowsInAddr(rankId)));
    magic2Global_.SetGlobalBuffer((__gm__ uint64_t *)(hccl_.GetWindowsInAddr(rankId) + 8));
    magic = SplitMagicWithValue(magicGlobal_.GetValue(0)) + 1;

    // 初始化 ub sqe
    tPipe_->InitBuffer(tBufSqe_, HCCL_AIV_STREAM_NUM * sizeof(TileXrStarsMemcpyAsyncSqe));
    sqeUb_ = tBufSqe_.Get<uint8_t>(0);
    GlobalTensor<uint8_t> sqeDefault;
    if (blockIdx == 0) {
        sqeDefault.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(&tileXrContext->streamInfo.starsMemcpyAsyncSqeDefault));
        DataCopy(sqeUb_, sqeDefault, HCCL_AIV_STREAM_NUM * sizeof(TileXrStarsMemcpyAsyncSqe));
    }
    tPipe_->InitBuffer(tBufTail_, HCCL_AIV_STREAM_NUM * sizeof(uint32_t) * 32);
    tailUb_ = tBufTail_.Get<uint32_t>(0);
    if (blockIdx == 0) {
        for (uint32_t i = 0; i < HCCL_AIV_STREAM_NUM; i++) {
            tailUb_.SetValue(i * 32, tileXrContext->streamInfo.tailList[i]);
        }
    }

    GlobalTensor<GM_ADDR> peerMemsAddrGm;
    peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs))->peerMems[0], TileXR::TILEXR_MAX_RANK_SIZE);
    for (int i = 0; i < rankSize; ++i) {
        shareAddrs[i] = peerMemsAddrGm.GetValue(i) +
                        (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
    }

    blockNum = AscendC::GetBlockNum();
    aGM_ = aGM;
    gatherGM_ = gatherGM;

    magicGlobal_.SetValue(0, MergeMagicWithValue(magic, STEP1));
    magic2Global_.SetValue(0, MergeMagicWithValue(magic, STEP2));
    AscendC::SyncAll();
    sync.Init(rankId, rankSize, shareAddrs, tBuf);
}

template <typename T>
__aicore__ inline void AllGather<T>::MemCopySQE(GM_ADDR srcAddr, GM_ADDR dstAddr, uint32_t length, uint32_t queueIdx)
{
    uint32_t tailIdx = 32 * queueIdx;
    // 组 sqe
    __gm__ TileXrStarsMemcpyAsyncSqe * const sqe =
                    (__gm__ TileXrStarsMemcpyAsyncSqe * const)tileXrContext->streamInfo.sqBaseList[queueIdx];
    uint16_t const streamId = (uint16_t const)tileXrContext->streamInfo.idList[queueIdx];
    uint16_t taskId = (uint16_t const)tileXrContext->streamInfo.taskIdList[queueIdx];
    __gm__ uint32_t * const headGm = (__gm__ uint32_t * const)tileXrContext->streamInfo.headList + queueIdx;
    __gm__ uint32_t * const tailGm = (__gm__ uint32_t * const)tileXrContext->streamInfo.tailList + queueIdx;
    uint32_t tail = tailUb_.GetValue(tailIdx);
    uint32_t const depth = tileXrContext->streamInfo.depthList[queueIdx];

    __ubuf__ TileXrStarsMemcpyAsyncSqe* sqeUb = (__ubuf__ TileXrStarsMemcpyAsyncSqe*)sqeUb_.GetPhyAddr(queueIdx * sizeof(TileXrStarsMemcpyAsyncSqe));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    sqeUb->length = length;
    *(__ubuf__ uint64_t*)(&sqeUb->src_addr_low) = (uint64_t)(srcAddr);
    *(__ubuf__ uint64_t*)(&sqeUb->dst_addr_low) = (uint64_t)(dstAddr);
    sqeUb->header.rtStreamId = streamId;
    sqeUb->header.taskId = taskId++;
    tileXrContext->streamInfo.depthList[queueIdx] = taskId + 1;
    tailUb_.SetValue(tailIdx, tailUb_.GetValue(tailIdx) + 1);

    // datacopy sqe
    AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    copy_ubuf_to_gm_align_b8(sqe + tail, sqeUb, 0, 1, sizeof(TileXrStarsMemcpyAsyncSqe), 0, 0, 0, 0);
    PipeBarrier<PIPE_MTE3>();

    // 敲DB
    copy_ubuf_to_gm_align_b8(tailGm, (__ubuf__ uint32_t *)tailUb_.GetPhyAddr(tailIdx), 0, 1, sizeof(uint32_t), 0, 0, 0, 0);
    PipeBarrier<PIPE_MTE3>();
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void AllGather<T>::GetBlockDataCount(
        const int64_t dataLen, const int64_t useBlockNum, int64_t& blockDataOffset, int64_t& blockDataCount, int blockId)
{
    // 向上整除获取每个core切分的数据个数
    blockDataCount = (dataLen + useBlockNum - 1) / useBlockNum;
    // 设置每个core数据下限
    blockDataCount = blockDataCount > MEM_DMA_UNIT_SIZE / sizeof(half) ?
                        blockDataCount : MEM_DMA_UNIT_SIZE / sizeof(half);
    // 极小数据量情况，core分配到数据下限，后面若干个core数据量为0
    blockDataOffset = blockId % useBlockNum * blockDataCount;  // 使用当前block在useBlock里的相对index计算偏移
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
        if (blockIdx == 0) {
            // step1：拷贝input至共享内存

            for (int queueIdx = 0; queueIdx < HCCL_AIV_STREAM_NUM; queueIdx++) {
                // 计算step1数据分片，input-->share，所有通道参与搬运
                GetBlockDataCount(tilingData_->gatherTileElemNum, HCCL_AIV_STREAM_NUM, offsetFromInput, countToShare, queueIdx);
                offsetToShare = offsetFromInput;
                inputAGM.SetGlobalBuffer((__gm__ T*)aGM_ + offsetFromInput, countToShare); // 非多轮切分AllGather场景，每张卡参与Gather的数据大小为{240，256}
                shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[rankId] + baseOffsetSize) + offsetToShare, countToShare);
                if (countToShare > 0) {
                    MemCopySQE((GM_ADDR)inputAGM.GetPhyAddr(), (GM_ADDR)shareGm.GetPhyAddr(), countToShare*sizeof(T), queueIdx);
                }
                __gm__ int64_t* basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rankId]) + queueIdx * FLAG_UNIT_INT_NUM;
                MemCopySQE((GM_ADDR)magicGlobal_.GetPhyAddr(), (GM_ADDR)basicSyncAddr, 8, queueIdx);
            }

            for (int queueIdx = 0; queueIdx < HCCL_AIV_STREAM_NUM; queueIdx++) {
                blockNumPerRank = HCCL_AIV_STREAM_NUM / rankSize;  // 均分core至每个rank，多余的core不使用
                useCoreNumToOutput = blockNumPerRank * rankSize;

                if (queueIdx >= useCoreNumToOutput) {
                    // 不用的通道直接退出
                    break;
                }
                
                // 计算step2数据分片，share-->output，当前core负责一个rank的部分或全部数据搬运
                GetBlockDataCount(tilingData_->gatherTileElemNum, blockNumPerRank, offsetFromShare, countToOutput, queueIdx);
                blockRank = queueIdx / blockNumPerRank;
                // 卡内同步，确保数据已拷贝至共享内存
                sync.WaitRankInnerFlag(magic, STEP1, blockRank, HCCL_AIV_STREAM_NUM);  // 等待目标rank的数据全部搬运完成

                offsetToOutput = blockRank * tilingData_->gatherTileElemNum + offsetFromShare;

                // 当前block的output分片
                
                gatherOutGM.SetGlobalBuffer((__gm__ T*)gatherGM_ + offsetToOutput, countToOutput);
                shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[blockRank] + baseOffsetSize) + offsetFromShare,
                    countToOutput);
                
                if (countToOutput > 0) {
                    // step2：拷贝共享内存至output
                    MemCopySQE((GM_ADDR)shareGm.GetPhyAddr(), (GM_ADDR)gatherOutGM.GetPhyAddr(), countToOutput*sizeof(T), queueIdx);
                    __gm__ int64_t* basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rankId]) + queueIdx * FLAG_UNIT_INT_NUM;
                    MemCopySQE((GM_ADDR)magic2Global_.GetPhyAddr(), (GM_ADDR)basicSyncAddr, 8, queueIdx);
                }

            }

            for (int queueIdx = 0; queueIdx < HCCL_AIV_STREAM_NUM; queueIdx++) {
                blockNumPerRank = HCCL_AIV_STREAM_NUM / rankSize;  // 均分core至每个rank，多余的core不使用
                useCoreNumToOutput = blockNumPerRank * rankSize;

                if (queueIdx >= useCoreNumToOutput) {
                    // 不用的通道直接退出
                    break;
                }
                GetBlockDataCount(tilingData_->gatherTileElemNum, blockNumPerRank, offsetFromShare, countToOutput, queueIdx);
                if (countToOutput > 0) {
                    sync.WaitRankInnerOneFlag(magic, STEP2, rankId, queueIdx);
                }
            }
        }
    }
}
}
#endif