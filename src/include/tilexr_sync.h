/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_SYNC_H
#define TILEXR_SYNC_H

#include "comm_args.h"

using namespace AscendC;

// 同步标志位占用长度
constexpr int64_t FLAG_UNIT_INT_NUM = 4;
// 每个同步单位占用内存大小（Bytes）
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);
// magic作为比较值时高位偏移量
constexpr int64_t MAGIC_OFFSET = 32;

// 多轮循环复用ipcBuff时，magic初始左移位数
constexpr int64_t MAGIC_ORIGIN_OFFSET = 10;
constexpr int64_t MAGIC_MASK = ~((1LL << MAGIC_OFFSET) - 1);

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs, TBuf<QuePosition::VECCALC> &tBuf)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        // 单个标志段长度
        segmentCount = GetBlockNum() * FLAG_UNIT_INT_NUM;
        // 初始化当前核对应的卡内/卡间同步地址
        localSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]);
        basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + segmentCount + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        this->tBuf = tBuf;
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(localSyncAddr + eventID * FLAG_UNIT_INT_NUM, v);
    }

    /**
     * @brief 设置指定卡的指定eventID的flag，设置的值为 magic 和 value 组合而成的值。
     * @param magic 算子批次，最终会组合到要set的flag的数值中高32位去
     * @param value 具体的最终要set的flag的数值中低32位的值
     * @param eventID 实际上从物理地址来看，是以共享内存首地址起往后的偏移量（要进行缩放，不是偏移量绝对值）。
     * @param rank 这个rank是在CommArgs结构体内peerMems数组内对应的rankId，并非global或local的id。
     *              （91093场景local不适用，910B多机场景global不适用。）
     */
    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline int32_t CalEventIdByMulBlockNum(int32_t blockMultiplier, int32_t targetCoreId)
    {
        return (blockMultiplier * blockNum) + targetCoreId;
    }

    /**
     * @brief 等待指定卡的指定eventID的flag变为 magic 和 value 组合而成的值。
     * @param magic 算子批次，最终会组合到要wait的flag的数值中高32位去
     * @param value 具体的最终要wait的flag的数值中低32位的值
     * @param eventID 实际上从物理地址来看，是以共享内存首地址起往后的偏移量。
     * @param rank 这个rank是在CommArgs结构体内peerMems数组内对应的rankId，并非global或local的id。
     *              （91093场景local不适用，910B多机场景global不适用。）
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    /**
     * @brief 相比起WaitSyncFlag函数，额外允许 远端Flag > 期望要check的FlagValue的值 通过校验。
     */
    __aicore__ inline void WaitSyncGreaterFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v, false);
    }

    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[this->rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    __aicore__ inline void WaitSyncGreaterFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[this->rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v, false);
    }

    /**
     * @brief 等待指定卡的指定eventID往后的flagNum个flag变为 magic 和 value 组合而成的值。<br>
     *          注：[eventID, eventID + flagNum)
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank, int64_t flagNum)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum, v);
    }

    __aicore__ inline void WaitSyncGreaterFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank,
                                               int64_t flagNum)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum, v, false);
    }

    // 设置单个卡内同步标志（内存A）
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(basicSyncAddr, value);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag((__gm__ int64_t*)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    // 等待单个卡内同步标志（内存A）
    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }

    // 等待整个rank内所有卡内同步标志（内存A）
    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // 等待整个rank内所有卡内同步标志（内存A）
    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]), flagNum, value);
    }

    // 等待整个rank内单个同步标志（内存A）
    __aicore__ inline void WaitRankInnerOneFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankOneFlag((__gm__ int64_t*)(shareAddrs[waitRank]), flagNum, value);
    }

    // 检验整个rank内所有卡内同步标志（内存A）
    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        return CheckOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // 设置单个卡间同步标志（内存B）
    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(blockOuterSyncAddr, value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(setRank, setBlock);
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(flagAddr, value);
    }

    // 等待单个卡间同步标志（内存B）
    __aicore__ inline void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(waitRank, waitBlock);
        WaitOneRankPartFlag(flagAddr, 1, value);
    }

    // 等待整个rank内所有卡间同步标志（内存B）
    __aicore__ inline void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(rank, 0);
        WaitOneRankPartFlag(flagAddr, blockNum, value);
    }

    // 等待所有rank从startBlock开始的flagNum个卡间同步标志（内存B）
    __aicore__ inline void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // 错峰读取rank标志，防止多核并发拷贝影响性能
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            WaitOneRankPartFlag(flagAddr, flagNum, value);
        }
    }

    // 检验所有rank从startBlock开始的flagNum个卡间同步标志（内存B）
    __aicore__ inline bool CheckAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock,
        int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // 错峰读取rank标志，防止多核并发拷贝影响性能
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            if (!CheckOneRankPartFlag(flagAddr, flagNum, value)) {
                return false;
            }
        }
        return true;
    }

    // 等待所有rank的所有卡间同步标志，全rank同步（内存B）
    __aicore__ inline void WaitAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        WaitAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // 检验所有rank的所有卡间同步标志，全rank同步（内存B）
    __aicore__ inline bool CheckAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        return CheckAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // 低级接口，设置同步标志
    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);

        // 将global同步标识拷贝至local
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // 等待SetValue完成
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // 等待UB->GM完成
    }

    // 低级接口，等待同步标志
    __aicore__ inline void WaitFlag(__gm__ int64_t* waitAddr, int64_t waitValue)
    {
        WaitOneRankPartFlag(waitAddr, 1, waitValue);
    }

    // 读取一个标志位，返回立即数
    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(1, 0);
        // 将global拷贝至local
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // 等待GM->UB

        int64_t res = localWait.GetValue(0);
        return res;
    }

    // 获取单个卡内多个连续的同步标志
    __aicore__ inline void WaitOneRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank,
                                                    int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(waitRank, startBlock);
        WaitOneRankPartFlag(flagAddr, flagNum, value);
    }

    // 获取单个卡内同步标志（内存A）
    __aicore__ inline int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM);
    }

    __aicore__ inline int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + segmentCount + waitBlock * FLAG_UNIT_INT_NUM);
    }

private:
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        // magic作为高位，eventID作为低位，组成一个value值用于比较
        return (static_cast<int64_t>(magic) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }

    /**
     * @brief 等待一个rank内部分同步标志
     * @param int64_t waitAddr  等待的首个标志位的地址（含）
     * @param int64_t flagNum   等待的标志位个数
     * @param int64_t checkValue    checkValue
     * @param bool mustEqual   用于当远端flagValue大于等于当前checkValue时，控制进一步判断逻辑。<br>
     *                      true表示相等，即MAGIC_MASK掩码部分必须严格相等；false表示可以接受远端的掩码部分大于等于checkValue的掩码部分。
     * @return
     */
    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue,
                                               bool mustEqual = true)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        int64_t checkedFlagNum = 0;
        do {
            int64_t remainToCheck = flagNum - checkedFlagNum;
            // 将global同步标识拷贝至local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM], remainToCheck * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // 等待GM->UB

            // 检验同步标识是否为checkValue
            isSync = true;
            for (auto i = 0; i < remainToCheck; ++i) {
                // 当有core未达到checkValue的阶段时，继续等待
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((mustEqual && (v < checkValue || ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK)))) ||
                    ((!mustEqual) && (v < checkValue || ((v & MAGIC_MASK) < (checkValue & MAGIC_MASK))))) {
                    isSync = false;
                    break;
                }
                checkedFlagNum++;
            }
        } while (!isSync);
    }

    /**
     * @brief 等待一个rank内单个同步标志
     * @param int64_t waitAddr  等待的首个标志位的地址（含）
     * @param int64_t flagNum   等待的标志位
     * @param int64_t checkValue    checkValue
     * @param bool mustEqual   用于当远端flagValue大于等于当前checkValue时，控制进一步判断逻辑。<br>
     *                      true表示相等，即MAGIC_MASK掩码部分必须严格相等；false表示可以接受远端的掩码部分大于等于checkValue的掩码部分。
     * @return
     */
    __aicore__ inline void WaitOneRankOneFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue,
                                               bool mustEqual = true)
    {
        __gm__ int64_t* basicSyncAddr = waitAddr + flagNum * FLAG_UNIT_INT_NUM;
        GlobalTensor<uint64_t> globalWait;
        globalWait.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(basicSyncAddr));
        LocalTensor<uint64_t> localWait = tBuf.GetWithOffset<uint64_t>(FLAG_UNIT_INT_NUM, 0);
        while (true) {
            DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
            if (localWait.GetValue(0) == checkValue) {
                break;
            }
        }
    }

    // 等待一个rank内所有同步标志
    __aicore__ inline void WaitOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        WaitOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    // 检验一个rank内部分同步标志，仅拷贝一次
    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        // 将global同步标识拷贝至local
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // 等待GM->UB
        // 检验同步标识是否为checkValue
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            // 当有core未达到checkValue的阶段时，继续等待
            int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
            if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                isSync = false;
                break;
            }
        }
        return isSync;
    }

    // 检验一个rank内所有同步标志，仅拷贝一次
    __aicore__ inline bool CheckOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        return CheckOneRankPartFlag(waitAddr, blockNum, checkValue);
    }
    int rank;
    int rankSize;
    int blockIdx;
    int blockNum;
    GM_ADDR *shareAddrs;
    int64_t segmentCount;  // 一组同步标志段的长度（int64_t类型计数）
    __gm__ int64_t* localSyncAddr;
    __gm__ int64_t* basicSyncAddr;  // 当前block卡内同步标志地址
    __gm__ int64_t* blockOuterSyncAddr;  // 当前block卡间同步标志地址
    TBuf<QuePosition::VECCALC> tBuf;
};

#endif // TILEXR_SYNC_H