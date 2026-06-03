#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t ALIGNED_LEN = 256U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t SPLIT_BLOCK_SIZE = 512U;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_FLAG_SIZE = 32U;
constexpr uint32_t SPLIT_BLOCK_FLAG_COUNT = SPLIT_BLOCK_FLAG_SIZE / sizeof(float);
constexpr uint32_t AXIS_H = 5120U;
constexpr uint32_t AXIS_K = 6U;
constexpr uint32_t TOTAL_EXPERTS = 128U;
constexpr uint32_t SHARED_EXPERT_NUM = 0U;
constexpr uint32_t FLAG_RCV_COUNT = AXIS_K + SHARED_EXPERT_NUM;
constexpr uint32_t RANK_HIST_SIZE = 128U;

#ifndef SIM_LOOP_ONLY
#define SIM_LOOP_ONLY 1
#endif

__aicore__ inline uint32_t CeilDivLocal(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}

__aicore__ inline uint32_t CeilLocal(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}

__aicore__ inline uint32_t Ceil32Local(uint32_t x)
{
    return (x + 31U) >> 5U;
}

__aicore__ inline uint32_t Align2Local(uint32_t x)
{
    return (x + 1U) & (~1U);
}

__aicore__ inline uint32_t Align256Local(uint32_t x)
{
    return (x + 255U) & (~255U);
}

template<AscendC::HardEvent event>
__aicore__ inline void SyncFuncLocal()
{
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <bool FinalVersion, uint32_t BatchSize, uint32_t EpWorldSize, bool LoopOnly>
class MoeEpCardCompare {
public:
    __aicore__ inline void Init(GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR output, GM_ADDR rankStats,
        GM_ADDR workspace)
    {
        hFloatAlign32Size_ = CeilDivLocal(AXIS_H * sizeof(float), UB_ALIGN) * UB_ALIGN;
        hFloatAlign256Size_ = CeilDivLocal(AXIS_H * sizeof(float), ALIGNED_LEN) * ALIGNED_LEN;
        hExpandXTypeSize_ = AXIS_H * sizeof(half);
        hExpandXAlign32Size_ = CeilDivLocal(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
        uint32_t scaleNum = Align2Local(Ceil32Local(AXIS_H));
        tokenScaleCnt_ = Align256Local(AXIS_H) / sizeof(half) + scaleNum;
        commDataBytes_ = tokenScaleCnt_ * sizeof(half);
        blockCntPerToken_ = CeilDivLocal(commDataBytes_, SPLIT_BLOCK_DATA_SIZE);
        hAlignWinSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
        moeExpertPerRankNum_ = TOTAL_EXPERTS / EpWorldSize;

        uint32_t packedDataBytes = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE;
        uint32_t rawPackedDataBytes = CeilDivLocal(hExpandXTypeSize_, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_DATA_SIZE;
        uint32_t moeQueueBytes = packedDataBytes > rawPackedDataBytes ? packedDataBytes : rawPackedDataBytes;
        moeQueueBytes = moeQueueBytes > hExpandXAlign32Size_ ? moeQueueBytes : hExpandXAlign32Size_;

        epWindowGM_ = workspace;
        expertIdsGM_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
        expertScalesGM_.SetGlobalBuffer((__gm__ float*)expertScales);
        outputGM_.SetGlobalBuffer((__gm__ float*)output);
        rankStatsGM_.SetGlobalBuffer((__gm__ uint32_t*)rankStats);

        pipe_.InitBuffer(rowTmpFloatBuf_, hFloatAlign32Size_);
        pipe_.InitBuffer(mulBuf_, hFloatAlign256Size_);
        pipe_.InitBuffer(sumFloatBuf_, hFloatAlign32Size_);
        pipe_.InitBuffer(moeSumQueue_, 1, moeQueueBytes);
        pipe_.InitBuffer(expertScalesBuf_,
            CeilDivLocal(BatchSize * AXIS_K * sizeof(float), UB_ALIGN) * UB_ALIGN);

        const uint32_t checkFlagCount = FLAG_RCV_COUNT * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
        const uint32_t checkCompareCount = CeilLocal(checkFlagCount, 64U) * 64U;
        pipe_.InitBuffer(packedCheckFlagBuf_, CeilLocal(checkCompareCount * sizeof(float), UB_ALIGN) * UB_ALIGN);
        pipe_.InitBuffer(packedCheckCompareBuf_, CeilLocal(checkCompareCount * sizeof(uint8_t), ALIGNED_LEN) * ALIGNED_LEN);

        rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
        mulBufLocal_ = mulBuf_.Get<float>();
        sumFloatBufLocal_ = sumFloatBuf_.Get<float>();
        expertScalesLocal_ = expertScalesBuf_.Get<float>();

        DataCopyPad(expertScalesLocal_, expertScalesGM_,
            DataCopyExtParams{1U, static_cast<uint32_t>(BatchSize * AXIS_K * sizeof(float)), 0U, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0U});
        SyncFuncLocal<AscendC::HardEvent::MTE2_S>();
        Duplicate(sumFloatBufLocal_, 0.0f, AXIS_H);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CheckPackedFlagRangeArrive(GM_ADDR flagBaseAddr, uint16_t blockCount,
        uint32_t flagFloatNum, uint32_t srcStrideBytes)
    {
        if (flagFloatNum == 0U) {
            return true;
        }
        const uint32_t compareCount = CeilLocal(flagFloatNum, 64U) * 64U;
        const uint32_t compResultU64Num = CeilLocal(flagFloatNum, 64U);
        LocalTensor<float> flagTensor = packedCheckFlagBuf_.Get<float>();
        LocalTensor<uint8_t> compResultTensor = packedCheckCompareBuf_.Get<uint8_t>();
        LocalTensor<uint64_t> compResultU64Tensor = packedCheckCompareBuf_.Get<uint64_t>();
        Duplicate<float>(flagTensor, float(0), compareCount);
        PipeBarrier<PIPE_V>();
        GlobalTensor<float> dataFlagGlobal;
        dataFlagGlobal.SetGlobalBuffer((__gm__ float*)(flagBaseAddr));
        DataCopyPad(flagTensor, dataFlagGlobal,
            DataCopyExtParams{blockCount, SPLIT_BLOCK_FLAG_SIZE, srcStrideBytes, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0U});
        SyncFuncLocal<AscendC::HardEvent::MTE2_V>();
        CompareScalar(compResultTensor, flagTensor, float(1), AscendC::CMPMODE::EQ, compareCount);
        SyncFuncLocal<AscendC::HardEvent::V_S>();

        uint32_t arriveFlagNum = 0U;
        for (uint32_t i = 0U; i < compResultU64Num; ++i) {
            uint64_t flagCompMask = compResultU64Tensor.GetValue(i);
            int64_t firstInvalidIdx = ScalarGetSFFValue<0>(flagCompMask);
            if (firstInvalidIdx == -1) {
                arriveFlagNum += 64U;
            } else {
                arriveFlagNum += static_cast<uint32_t>(firstInvalidIdx);
                break;
            }
        }
        if (arriveFlagNum > flagFloatNum) {
            arriveFlagNum = flagFloatNum;
        }
        return arriveFlagNum == flagFloatNum;
    }

    __aicore__ inline bool CheckPackedTokenArrive(GM_ADDR rankGM)
    {
        if (blockCntPerToken_ == 0U) {
            return true;
        }
        const uint32_t flagFloatNum = blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
        return CheckPackedFlagRangeArrive(rankGM + SPLIT_BLOCK_DATA_SIZE,
            static_cast<uint16_t>(blockCntPerToken_), flagFloatNum, SPLIT_BLOCK_DATA_SIZE);
    }

    __aicore__ inline bool CheckPackedTokenArriveAllSlots(uint32_t tokenIndex)
    {
        const uint32_t flagFloatNum = FLAG_RCV_COUNT * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
        GM_ADDR tokenBaseAddr = (__gm__ uint8_t*)(epWindowGM_) + tokenIndex * FLAG_RCV_COUNT * hAlignWinSize_;
        return CheckPackedFlagRangeArrive(tokenBaseAddr + SPLIT_BLOCK_DATA_SIZE,
            static_cast<uint16_t>(FLAG_RCV_COUNT * blockCntPerToken_), flagFloatNum, SPLIT_BLOCK_DATA_SIZE);
    }

    __aicore__ inline bool CheckPackedTokenArriveBatch(uint32_t tokenIndex)
    {
        if constexpr (FinalVersion) {
            if (blockCntPerToken_ == 0U) {
                return true;
            }
            const uint32_t flagFloatNum = FLAG_RCV_COUNT * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
            const uint32_t compareCount = CeilLocal(flagFloatNum, 64U) * 64U;
            const uint32_t compResultU64Num = CeilLocal(flagFloatNum, 64U);
            LocalTensor<float> flagTensor = packedCheckFlagBuf_.Get<float>();
            LocalTensor<uint8_t> compResultTensor = packedCheckCompareBuf_.Get<uint8_t>();
            LocalTensor<uint64_t> compResultU64Tensor = packedCheckCompareBuf_.Get<uint64_t>();
            Duplicate<float>(flagTensor, float(0), compareCount);
            PipeBarrier<PIPE_V>();

            GM_ADDR tokenBaseAddr = (__gm__ uint8_t*)(epWindowGM_) + tokenIndex * FLAG_RCV_COUNT * hAlignWinSize_;
            GlobalTensor<float> tokenFlagGlobal;
            tokenFlagGlobal.SetGlobalBuffer((__gm__ float*)(tokenBaseAddr + SPLIT_BLOCK_DATA_SIZE));
            DataCopyPad(flagTensor, tokenFlagGlobal,
                DataCopyExtParams{static_cast<uint16_t>(FLAG_RCV_COUNT * blockCntPerToken_),
                    SPLIT_BLOCK_FLAG_SIZE, SPLIT_BLOCK_DATA_SIZE, 0U, 0U},
                DataCopyPadExtParams<float>{false, 0U, 0U, 0U});
            SyncFuncLocal<AscendC::HardEvent::MTE2_V>();
            CompareScalar(compResultTensor, flagTensor, float(1), AscendC::CMPMODE::EQ, compareCount);
            SyncFuncLocal<AscendC::HardEvent::V_S>();

            uint32_t arriveFlagNum = 0U;
            for (uint32_t i = 0U; i < compResultU64Num; ++i) {
                uint64_t flagCompMask = compResultU64Tensor.GetValue(i);
                int64_t firstInvalidIdx = ScalarGetSFFValue<0>(flagCompMask);
                if (firstInvalidIdx == -1) {
                    arriveFlagNum += 64U;
                } else {
                    arriveFlagNum += static_cast<uint32_t>(firstInvalidIdx);
                    break;
                }
            }
            if (arriveFlagNum > flagFloatNum) {
                arriveFlagNum = flagFloatNum;
            }
            return arriveFlagNum == flagFloatNum;
        } else {
            return CheckPackedTokenArriveAllSlots(tokenIndex);
        }
    }

    __aicore__ inline bool WaitDispatch(uint32_t tokenIndex)
    {
        return CheckPackedTokenArriveBatch(tokenIndex);
    }

    __aicore__ inline void ProcessMoeExpert(uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal)
    {
        const DataCopyExtParams xScaleCopyParams{
            static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE, SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
        const DataCopyPadExtParams<half> copyPadExtParams{false, 0U, 0U, 0U};

        GM_ADDR wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
        rowTmpGlobal_.SetGlobalBuffer((__gm__ half*)wAddr);
        LocalTensor<half> tmpUb = moeSumQueue_.AllocTensor<half>();
        DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
        moeSumQueue_.EnQue(tmpUb);
        tmpUb = moeSumQueue_.DeQue<half>();

        LocalTensor<fp8_e5m2_t> fp8LocalTensor = tmpUb.template ReinterpretCast<fp8_e5m2_t>();
        Cast(rowTmpFloatLocal_, fp8LocalTensor, AscendC::RoundMode::CAST_NONE, AXIS_H);
        PipeBarrier<PIPE_V>();
        Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, AXIS_H);
        PipeBarrier<PIPE_V>();
        Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, AXIS_H);

        if constexpr (!FinalVersion) {
            PipeBarrier<PIPE_ALL>();
        }
        moeSumQueue_.FreeTensor<half>(tmpUb);
    }

    __aicore__ inline void ProcessMoeExpertsLoop(uint32_t tokenIndex, uint32_t tokenIndexOffset, uint32_t& index)
    {
        (void)tokenIndex;
        for (uint32_t topkId = 0U; topkId < AXIS_K; topkId++) {
            float scaleVal = expertScalesLocal_.GetValue(index);
            ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
            index++;
        }
    }

    __aicore__ inline void ProcessAllTokens()
    {
        uint32_t index = 0U;
        for (uint32_t token = 0U; token < BatchSize; ++token) {
            if constexpr (!LoopOnly) {
                if (!WaitDispatch(token)) {
                    continue;
                }
            }
            Duplicate(sumFloatBufLocal_, 0.0f, AXIS_H);
            PipeBarrier<PIPE_V>();
            ProcessMoeExpertsLoop(token, token * FLAG_RCV_COUNT, index);
        }
    }

    __aicore__ inline void Store()
    {
        SyncFuncLocal<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(outputGM_, sumFloatBufLocal_,
            DataCopyExtParams{1U, static_cast<uint32_t>(AXIS_H * sizeof(float)), 0U, 0U, 0U});
    }

    __aicore__ inline void StoreRankStats()
    {
        for (uint32_t i = 0U; i < RANK_HIST_SIZE; ++i) {
            rankStatsGM_.SetValue(i, 0U);
        }
        rankStatsGM_.SetValue(RANK_HIST_SIZE + 0U, EpWorldSize);
        rankStatsGM_.SetValue(RANK_HIST_SIZE + 1U, TOTAL_EXPERTS);
        rankStatsGM_.SetValue(RANK_HIST_SIZE + 2U, moeExpertPerRankNum_);
        rankStatsGM_.SetValue(RANK_HIST_SIZE + 3U, BatchSize);
        rankStatsGM_.SetValue(RANK_HIST_SIZE + 4U, AXIS_K);
        rankStatsGM_.SetValue(RANK_HIST_SIZE + 5U, blockCntPerToken_);
        rankStatsGM_.SetValue(RANK_HIST_SIZE + 6U, hAlignWinSize_);

        for (uint32_t token = 0U; token < BatchSize; ++token) {
            for (uint32_t topkId = 0U; topkId < AXIS_K; ++topkId) {
                uint32_t expertId = static_cast<uint32_t>(expertIdsGM_.GetValue(token * AXIS_K + topkId));
                uint32_t fromRank = expertId / moeExpertPerRankNum_;
                uint32_t oldCount = rankStatsGM_.GetValue(fromRank);
                rankStatsGM_.SetValue(fromRank, oldCount + 1U);
            }
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> packedCheckFlagBuf_;
    TBuf<> packedCheckCompareBuf_;
    GlobalTensor<half> rowTmpGlobal_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<float> outputGM_;
    GlobalTensor<uint32_t> rankStatsGM_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;
    LocalTensor<float> expertScalesLocal_;
    GM_ADDR epWindowGM_{nullptr};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t commDataBytes_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t moeExpertPerRankNum_{0};
};

extern "C" __global__ __aicore__ void KERNEL_NAME(
    GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR output, GM_ADDR rankStats, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)tiling;
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
        MoeEpCardCompare<(USE_FINAL != 0), CASE_BS, CASE_EP, (SIM_LOOP_ONLY != 0)> op;
        op.Init(expertIds, expertScales, output, rankStats, workspace);
        op.StoreRankStats();
        op.ProcessAllTokens();
        op.Store();
    }
}
