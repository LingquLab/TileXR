#ifndef TILEXR_EP_KERNELS_TILEXR_EP_COMBINE_MEMORY_KERNEL_H
#define TILEXR_EP_KERNELS_TILEXR_EP_COMBINE_MEMORY_KERNEL_H

#include <cstdint>

#include "comm_args.h"
#include "kernel_operator.h"
#include "tilexr_data_as_flag.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "tilexr_ep_mxfp8_quant.h"
#endif
#include "tilexr_types.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

namespace Mc2Kernel {

using namespace AscendC;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t STATE_WINDOW_BYTES = 1024U * 1024U;
constexpr uint32_t ASSIST_INFO_WIDTH = 4U;
constexpr int64_t ACTIVE_MASK_NONE = 0;
constexpr int64_t ACTIVE_MASK_TOKEN = 1;
constexpr int64_t MXFP8_E5M2_COMM_QUANT = 3;
constexpr int64_t MXFP8_E4M3_COMM_QUANT = 4;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    AscendC::TEventID eventId = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

__aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

class TileXRCombineMemoryContext {
public:
    __aicore__ inline void Init(GM_ADDR commArgsGM)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
    }

    __aicore__ inline uint32_t Rank() const
    {
        return static_cast<uint32_t>(args_->rank);
    }

    __aicore__ inline uint32_t RankSize() const
    {
        return static_cast<uint32_t>(args_->rankSize);
    }

    __aicore__ inline GM_ADDR PeerDataBase(uint32_t rank) const
    {
        return args_->peerMems[rank] + TileXR::IPC_DATA_OFFSET;
    }

private:
    __gm__ TileXR::CommArgs *args_{nullptr};
};

template <typename XType>
class MoeDistributeCombineV2A5Mte {
public:
    __aicore__ inline void Init(GM_ADDR commArgsGM, GM_ADDR expertOutGM, GM_ADDR assistInfoGM,
        GM_ADDR sendCountsGM, GM_ADDR expertScalesGM, GM_ADDR xActiveMaskGM, GM_ADDR sharedExpertXGM,
        GM_ADDR yOutGM, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
        int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
        int64_t activeMaskType, int64_t quantMode, int64_t magic, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReadSendCount();
    __aicore__ inline void SplitSendRows();
    __aicore__ inline void InitSendBuffers();
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void SendOneRow(uint32_t row, LocalTensor<int32_t> assistLocal);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    template <typename Fp8Type>
    __aicore__ inline void QuantMxfp8(LocalTensor<uint8_t> quantLocal, LocalTensor<XType> inputLocal);
#endif
    __aicore__ inline void InitReceiveBuffers();
    __aicore__ inline bool TokenActive(uint32_t tokenIndex);
    __aicore__ inline bool CheckPackedTokenArrive(uint32_t tokenIndex);
    __aicore__ inline void CopyAndAccumulateSlot(uint32_t tokenIndex, uint32_t slotIndex, float scale,
        LocalTensor<float> sumLocal, LocalTensor<float> rowFloat, LocalTensor<float> mulLocal);
    __aicore__ inline void AddSharedExpertX(uint32_t tokenIndex, LocalTensor<float> sumLocal,
        LocalTensor<float> rowFloat);
    __aicore__ inline void ClearTokenFlags(uint32_t tokenIndex, LocalTensor<float> clearLocal);
    __aicore__ inline void WriteOutput(uint32_t tokenIndex, LocalTensor<float> sumLocal);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline GM_ADDR CombineWindowBase(uint32_t rank) const;

    TPipe *pipe_{nullptr};
    TileXRCombineMemoryContext context_;
    uint32_t coreIdx_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t moeExpertNum_{0};
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t sendCountNum_{0};
    uint32_t selfSendCnt_{0};
    uint32_t startSendRow_{0};
    uint32_t sendRowCount_{0};
    uint32_t dataState_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t packedRowBytes_{0};
    uint32_t compactPayloadBytes_{0};
    uint32_t commDataBytes_{0};
    uint32_t rowBytes_{0};
    uint32_t rowAlignBytes_{0};
    uint32_t inputAlignBytes_{0};
    uint32_t sendInputBytes_{0};
    uint32_t receiveInputBytes_{0};
    uint32_t floatRowAlignBytes_{0};
    uint32_t quantPayloadBytes_{0};
    uint32_t quantScaleCount_{0};
    uint32_t quantComputeCount_{0};
    uint32_t quantComputeScaleCount_{0};
    uint32_t quantWorkBytes_{0};
    uint32_t slotCount_{0};
    uint64_t totalWinSize_{0};
    uint64_t halfWinSize_{0};
    int64_t activeMaskType_{ACTIVE_MASK_NONE};
    int64_t quantMode_{0};
    bool useMxfp8_{false};
    bool hasExpertScales_{false};
    bool hasSharedExpertX_{false};
    GlobalTensor<XType> expertOutGM_;
    GlobalTensor<int32_t> assistInfoGM_;
    GlobalTensor<int32_t> sendCountsGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> yOutGM_;

    TBuf<> metaBuf_;
    TBuf<> assistBuf_;
    TQue<QuePosition::VECIN, 1> sendInputQueue_;
    TQue<QuePosition::VECOUT, 1> sendOutputQueue_;
    TBuf<> quantResultBuf_;
    TBuf<> quantWorkBuf_;
    TBuf<> packedCheckFlagBuf_;
    TBuf<> packedCheckCompareBuf_;
    TQue<QuePosition::VECIN, 1> packedInputQueue_;
    TBuf<> rowFloatBuf_;
    TBuf<> mulFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> dequantScaleBuf_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<> expertScaleBuf_;
    TBuf<> activeMaskBuf_;
    TBuf<> clearFlagBuf_;
    TBuf<> tokenStatusBuf_;
};

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::Init(GM_ADDR commArgsGM, GM_ADDR expertOutGM,
    GM_ADDR assistInfoGM, GM_ADDR sendCountsGM, GM_ADDR expertScalesGM, GM_ADDR xActiveMaskGM,
    GM_ADDR sharedExpertXGM, GM_ADDR yOutGM, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t activeMaskType, int64_t quantMode, int64_t magic, TPipe *pipe)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    pipe_ = pipe;
    context_.Init(commArgsGM);
    coreIdx_ = GetBlockIdx();
    aivNum_ = GetBlockNum();
    epRankId_ = context_.Rank();
    epWorldSize_ = context_.RankSize();
    axisBS_ = static_cast<uint32_t>(bs);
    axisH_ = static_cast<uint32_t>(h);
    axisK_ = static_cast<uint32_t>(topK);
    moeExpertNum_ = static_cast<uint32_t>(moeExpertNum);
    sharedExpertNum_ = static_cast<uint32_t>(sharedExpertNum);
    sharedExpertRankNum_ = static_cast<uint32_t>(sharedExpertRankNum);
    activeMaskType_ = activeMaskType;
    quantMode_ = quantMode;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    useMxfp8_ = quantMode_ == MXFP8_E5M2_COMM_QUANT || quantMode_ == MXFP8_E4M3_COMM_QUANT;
#else
    useMxfp8_ = false;
#endif
    slotCount_ = axisK_ + sharedExpertNum_;
    const uint32_t moeRankNum = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeRankNum;
    sendCountNum_ = epRankId_ < sharedExpertRankNum_ ? epWorldSize_ : epWorldSize_ * moeExpertNumPerRank_;

    rowBytes_ = axisH_ * sizeof(XType);
    rowAlignBytes_ = static_cast<uint32_t>(AlignUp(rowBytes_, UB_ALIGN));
    floatRowAlignBytes_ = static_cast<uint32_t>(
        AlignUp(axisH_ * sizeof(float), useMxfp8_ ? 512U : UB_ALIGN));
    if (useMxfp8_) {
        quantPayloadBytes_ = static_cast<uint32_t>(AlignUp(axisH_, 256U));
        quantScaleCount_ = static_cast<uint32_t>(AlignUp(CeilDiv(axisH_, 32U), 2U));
        quantComputeCount_ = quantPayloadBytes_;
        quantComputeScaleCount_ = quantComputeCount_ / 32U;
        inputAlignBytes_ = quantComputeCount_ * sizeof(XType);
        commDataBytes_ = quantPayloadBytes_ + quantScaleCount_;
        quantWorkBytes_ = static_cast<uint32_t>(AlignUp(
            AlignUp(quantComputeScaleCount_, 32U) * sizeof(float) +
                quantComputeScaleCount_ * sizeof(uint16_t), UB_ALIGN));
    } else {
        inputAlignBytes_ = rowAlignBytes_;
        commDataBytes_ = rowBytes_;
    }
    blockCntPerToken_ = static_cast<uint32_t>(
        CeilDiv(commDataBytes_, TileXR::DATA_AS_FLAG_PAYLOAD_BYTES));
    packedRowBytes_ = blockCntPerToken_ * TileXR::DATA_AS_FLAG_BLOCK_BYTES;
    compactPayloadBytes_ = blockCntPerToken_ * TileXR::DATA_AS_FLAG_PAYLOAD_BYTES;
    sendInputBytes_ = useMxfp8_ ? inputAlignBytes_ :
        (inputAlignBytes_ > compactPayloadBytes_ ? inputAlignBytes_ : compactPayloadBytes_);
    receiveInputBytes_ = rowAlignBytes_ > compactPayloadBytes_ ?
        rowAlignBytes_ : compactPayloadBytes_;

    const uint64_t workspaceStatusNum = static_cast<uint64_t>(epWorldSize_) * moeExpertNumPerRank_;
    const uint64_t workspaceBytes = AlignUp(
        static_cast<uint64_t>(aivNum_) * workspaceStatusNum * sizeof(int32_t), UB_ALIGN);
    totalWinSize_ = TileXR::IPC_BUFF_MAX_SIZE - STATE_WINDOW_BYTES - workspaceBytes;
    halfWinSize_ = totalWinSize_ / 2U;

    expertOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(expertOutGM));
    assistInfoGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(assistInfoGM));
    sendCountsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsGM));
    expertScalesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expertScalesGM));
    xActiveMaskGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(xActiveMaskGM));
    sharedExpertXGM_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(sharedExpertXGM));
    yOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(yOutGM));
    hasExpertScales_ = expertScalesGM != nullptr;
    hasSharedExpertX_ = sharedExpertXGM != nullptr;
    dataState_ = static_cast<uint32_t>(magic) & 1U;

    pipe_->InitBuffer(metaBuf_, 2U * UB_ALIGN);
    ReadSendCount();
    SplitSendRows();
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::ReadSendCount()
{
    LocalTensor<int32_t> countLocal = metaBuf_.GetWithOffset<int32_t>(UB_ALIGN / sizeof(int32_t), UB_ALIGN);
    const DataCopyExtParams countParams {1U, sizeof(int32_t), 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> padParams {false, 0U, 0U, 0U};
    DataCopyPad(countLocal, sendCountsGM_[sendCountNum_ - 1U], countParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    const int32_t count = countLocal.GetValue(0);
    selfSendCnt_ = count > 0 ? static_cast<uint32_t>(count) : 0U;
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::SplitSendRows()
{
    sendRowCount_ = selfSendCnt_ / aivNum_;
    const uint32_t remainder = selfSendCnt_ % aivNum_;
    startSendRow_ = sendRowCount_ * coreIdx_;
    if (coreIdx_ < remainder) {
        ++sendRowCount_;
        startSendRow_ += coreIdx_;
    } else {
        startSendRow_ += remainder;
    }
}

template <typename XType>
__aicore__ inline GM_ADDR MoeDistributeCombineV2A5Mte<XType>::CombineWindowBase(uint32_t rank) const
{
    return context_.PeerDataBase(rank) + STATE_WINDOW_BYTES + dataState_ * halfWinSize_;
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::InitSendBuffers()
{
    pipe_->Reset();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    pipe_->InitBuffer(assistBuf_, UB_ALIGN);
    pipe_->InitBuffer(sendInputQueue_, 1, sendInputBytes_);
    pipe_->InitBuffer(sendOutputQueue_, 1, packedRowBytes_);
    if (useMxfp8_) {
        pipe_->InitBuffer(quantResultBuf_, compactPayloadBytes_);
        pipe_->InitBuffer(quantWorkBuf_, quantWorkBytes_);
    }
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <typename XType>
template <typename Fp8Type>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::QuantMxfp8(
    LocalTensor<uint8_t> quantLocal, LocalTensor<XType> inputLocal)
{
    __ubuf__ XType *srcAddr = reinterpret_cast<__ubuf__ XType *>(inputLocal.GetPhyAddr());
    __ubuf__ uint8_t *workAddr = reinterpret_cast<__ubuf__ uint8_t *>(
        quantWorkBuf_.Get<uint8_t>().GetPhyAddr());
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(workAddr);
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        workAddr + AlignUp(quantComputeScaleCount_, 32U) * sizeof(float));
    __ubuf__ int8_t *outAddr = reinterpret_cast<__ubuf__ int8_t *>(quantLocal.GetPhyAddr());
    __ubuf__ uint16_t *scaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        quantLocal[quantPayloadBytes_].GetPhyAddr());

    TileXRMxfp8Quant::ComputeMaxExp(srcAddr, maxExpAddr, quantComputeCount_);
    TileXRMxfp8Quant::ComputeScale<Fp8Type>(
        maxExpAddr, scaleAddr, halfScaleAddr, quantComputeScaleCount_);
    TileXRMxfp8Quant::ComputeFp8Data<XType, Fp8Type,
        RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
        srcAddr, halfScaleAddr, outAddr, quantComputeCount_);
}
#endif

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::SendOneRow(uint32_t row,
    LocalTensor<int32_t> assistLocal)
{
    const DataCopyExtParams assistParams {1U, ASSIST_INFO_WIDTH * sizeof(int32_t), 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> assistPad {false, 0U, 0U, 0U};
    DataCopyPad(assistLocal, assistInfoGM_[row * ASSIST_INFO_WIDTH], assistParams, assistPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    const uint32_t toRankId = static_cast<uint32_t>(assistLocal.GetValue(0));
    const uint32_t tokenId = static_cast<uint32_t>(assistLocal.GetValue(1));
    const uint32_t topkId = static_cast<uint32_t>(assistLocal.GetValue(2));
    if (toRankId >= epWorldSize_ || tokenId >= axisBS_ || topkId >= slotCount_) {
        return;
    }
    const uint64_t slot = static_cast<uint64_t>(tokenId) * slotCount_ + topkId;
    GM_ADDR dst = CombineWindowBase(toRankId) + slot * packedRowBytes_;

    LocalTensor<XType> inputLocal = sendInputQueue_.AllocTensor<XType>();
    const DataCopyExtParams inputParams {1U, rowBytes_, 0U, 0U, 0U};
    const DataCopyPadExtParams<XType> inputPad {useMxfp8_, 0U, 0U, 0U};
    Duplicate<uint32_t>(inputLocal.template ReinterpretCast<uint32_t>(), 0U,
        sendInputBytes_ / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(inputLocal, expertOutGM_[static_cast<uint64_t>(row) * axisH_], inputParams, inputPad);
    sendInputQueue_.EnQue(inputLocal);
    inputLocal = sendInputQueue_.DeQue<XType>();

    LocalTensor<XType> outputLocal = sendOutputQueue_.AllocTensor<XType>();
    LocalTensor<float> sourceFloat = inputLocal.template ReinterpretCast<float>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if (useMxfp8_) {
        LocalTensor<uint8_t> quantLocal = quantResultBuf_.Get<uint8_t>();
        Duplicate<uint32_t>(quantLocal.ReinterpretCast<uint32_t>(), 0U,
            compactPayloadBytes_ / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        if (quantMode_ == MXFP8_E5M2_COMM_QUANT) {
            QuantMxfp8<fp8_e5m2_t>(quantLocal, inputLocal);
        } else {
            QuantMxfp8<fp8_e4m3fn_t>(quantLocal, inputLocal);
        }
        PipeBarrier<PIPE_V>();
        sourceFloat = quantLocal.template ReinterpretCast<float>();
    }
#endif
    LocalTensor<float> packedFloat = outputLocal.template ReinterpretCast<float>();
    Duplicate(packedFloat, TileXR::DATA_AS_FLAG_READY_VALUE, packedRowBytes_ / sizeof(float));
    PipeBarrier<PIPE_V>();
    Copy(packedFloat, sourceFloat, 64U, static_cast<uint8_t>(blockCntPerToken_), {1, 1, 16, 15});
    Copy(packedFloat[64], sourceFloat[64], 56U, static_cast<uint8_t>(blockCntPerToken_), {1, 1, 16, 15});
    sendOutputQueue_.EnQue(outputLocal);
    outputLocal = sendOutputQueue_.DeQue<XType>();

    GlobalTensor<float> dstPackedGlobal;
    dstPackedGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
    DataCopy(dstPackedGlobal, packedFloat, packedRowBytes_ / sizeof(float));
    sendOutputQueue_.FreeTensor(outputLocal);
    sendInputQueue_.FreeTensor(inputLocal);
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::ExpertAlltoAllDispatchCopyAdd()
{
    if (sendRowCount_ == 0U) {
        return;
    }
    LocalTensor<int32_t> assistLocal = assistBuf_.Get<int32_t>();
    uint32_t permStride = 1U;
    if (sendRowCount_ > 2U) {
        permStride = sendRowCount_ / 2U + 1U;
        if (((sendRowCount_ & 1U) == 0U) && ((permStride & 1U) == 0U)) {
            ++permStride;
        }
    }
    const uint32_t rankOffset = (epRankId_ * sendRowCount_) / epWorldSize_;
    uint32_t permIdx = rankOffset % sendRowCount_;
    for (uint32_t loop = 0U; loop < sendRowCount_; ++loop) {
        SendOneRow(startSendRow_ + permIdx, assistLocal);
        permIdx += permStride;
        if (permIdx >= sendRowCount_) {
            permIdx -= sendRowCount_;
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::InitReceiveBuffers()
{
    pipe_->Reset();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    const uint32_t totalBlocks = slotCount_ * blockCntPerToken_;
    const uint32_t flagFloatCount = totalBlocks * TileXR::DATA_AS_FLAG_FLAG_FLOATS;
    const uint32_t compareCount = static_cast<uint32_t>(AlignUp(flagFloatCount, 64U));
    pipe_->InitBuffer(packedCheckFlagBuf_, compareCount * sizeof(float));
    pipe_->InitBuffer(packedCheckCompareBuf_, AlignUp(compareCount * sizeof(uint8_t), 256U));
    pipe_->InitBuffer(packedInputQueue_, 1, receiveInputBytes_);
    pipe_->InitBuffer(rowFloatBuf_, floatRowAlignBytes_);
    pipe_->InitBuffer(mulFloatBuf_, floatRowAlignBytes_);
    pipe_->InitBuffer(sumFloatBuf_, floatRowAlignBytes_);
    if (useMxfp8_) {
        pipe_->InitBuffer(dequantScaleBuf_,
            AlignUp(quantScaleCount_, 128U) * sizeof(float) * 2U);
    }
    pipe_->InitBuffer(outputQueue_, 1, rowAlignBytes_);
    pipe_->InitBuffer(expertScaleBuf_, AlignUp(axisK_ * sizeof(float), UB_ALIGN));
    pipe_->InitBuffer(activeMaskBuf_, AlignUp(axisBS_ * sizeof(bool), UB_ALIGN));
    pipe_->InitBuffer(clearFlagBuf_, blockCntPerToken_ * TileXR::DATA_AS_FLAG_FLAG_BYTES);
    const uint32_t maxTokenCountPerCore = static_cast<uint32_t>(CeilDiv(axisBS_, aivNum_));
    pipe_->InitBuffer(tokenStatusBuf_, AlignUp(maxTokenCountPerCore * sizeof(int32_t), UB_ALIGN));
    if (activeMaskType_ == ACTIVE_MASK_TOKEN) {
        LocalTensor<bool> maskLocal = activeMaskBuf_.Get<bool>();
        const DataCopyExtParams maskParams {
            1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
        const DataCopyPadExtParams<bool> maskPad {false, 0U, 0U, 0U};
        DataCopyPad(maskLocal, xActiveMaskGM_, maskParams, maskPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
    LocalTensor<float> clearLocal = clearFlagBuf_.Get<float>();
    Duplicate(clearLocal, 0.0f,
        blockCntPerToken_ * TileXR::DATA_AS_FLAG_FLAG_BYTES / sizeof(float));
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <typename XType>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<XType>::TokenActive(uint32_t tokenIndex)
{
    if (activeMaskType_ != ACTIVE_MASK_TOKEN) {
        return true;
    }
    return activeMaskBuf_.Get<bool>().GetValue(tokenIndex);
}

template <typename XType>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<XType>::CheckPackedTokenArrive(uint32_t tokenIndex)
{
    const uint32_t totalBlocks = slotCount_ * blockCntPerToken_;
    const uint32_t flagFloatCount = totalBlocks * TileXR::DATA_AS_FLAG_FLAG_FLOATS;
    const uint32_t compareCount = static_cast<uint32_t>(AlignUp(flagFloatCount, 64U));
    const uint32_t compareU64Count = static_cast<uint32_t>(CeilDiv(flagFloatCount, 64U));
    GM_ADDR tokenBase = CombineWindowBase(epRankId_) +
        static_cast<uint64_t>(tokenIndex) * slotCount_ * packedRowBytes_;

    LocalTensor<float> flagLocal = packedCheckFlagBuf_.Get<float>();
    LocalTensor<uint8_t> compareLocal = packedCheckCompareBuf_.Get<uint8_t>();
    LocalTensor<uint64_t> compareU64 = packedCheckCompareBuf_.Get<uint64_t>();
    Duplicate(flagLocal, 0.0f, compareCount);
    PipeBarrier<PIPE_V>();

    GlobalTensor<float> flagGlobal;
    flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        tokenBase + TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES));
    const DataCopyExtParams flagParams {static_cast<uint16_t>(totalBlocks),
        TileXR::DATA_AS_FLAG_FLAG_BYTES, TileXR::DATA_AS_FLAG_PAYLOAD_BYTES, 0U, 0U};
    const DataCopyPadExtParams<float> flagPad {false, 0U, 0U, 0U};
    DataCopyPad(flagLocal, flagGlobal, flagParams, flagPad);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    CompareScalar(compareLocal, flagLocal, TileXR::DATA_AS_FLAG_READY_VALUE,
        AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t arrived = 0U;
    for (uint32_t index = 0U; index < compareU64Count; ++index) {
        const uint64_t mask = compareU64.GetValue(index);
        const int64_t firstInvalid = ScalarGetSFFValue<0>(mask);
        if (firstInvalid == -1) {
            arrived += 64U;
        } else {
            arrived += static_cast<uint32_t>(firstInvalid);
            break;
        }
    }
    return (arrived > flagFloatCount ? flagFloatCount : arrived) == flagFloatCount;
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::CopyAndAccumulateSlot(uint32_t tokenIndex,
    uint32_t slotIndex, float scale, LocalTensor<float> sumLocal, LocalTensor<float> rowFloat,
    LocalTensor<float> mulLocal)
{
    GM_ADDR slotAddr = CombineWindowBase(epRankId_) +
        (static_cast<uint64_t>(tokenIndex) * slotCount_ + slotIndex) * packedRowBytes_;
    GlobalTensor<XType> packedGlobal;
    packedGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(slotAddr));
    LocalTensor<XType> packedLocal = packedInputQueue_.AllocTensor<XType>();
    const DataCopyExtParams copyParams {static_cast<uint16_t>(blockCntPerToken_),
        TileXR::DATA_AS_FLAG_PAYLOAD_BYTES, TileXR::DATA_AS_FLAG_FLAG_BYTES, 0U, 0U};
    const DataCopyPadExtParams<XType> copyPad {false, 0U, 0U, 0U};
    DataCopyPad(packedLocal, packedGlobal, copyParams, copyPad);
    packedInputQueue_.EnQue(packedLocal);
    packedLocal = packedInputQueue_.DeQue<XType>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if (useMxfp8_) {
        __ubuf__ uint8_t *tokenAddr = reinterpret_cast<__ubuf__ uint8_t *>(packedLocal.GetPhyAddr());
        __ubuf__ fp8_e8m0_t *scaleAddr = reinterpret_cast<__ubuf__ fp8_e8m0_t *>(
            tokenAddr + quantPayloadBytes_);
        __ubuf__ float *scaleWorkAddr = reinterpret_cast<__ubuf__ float *>(
            dequantScaleBuf_.Get<float>().GetPhyAddr());
        __ubuf__ float *sumAddr = reinterpret_cast<__ubuf__ float *>(sumLocal.GetPhyAddr());
        if (quantMode_ == MXFP8_E5M2_COMM_QUANT) {
            TileXRMxfp8Quant::DequantizeAndAccumulate<fp8_e5m2_t>(
                tokenAddr, scaleAddr, scaleWorkAddr, sumAddr,
                axisH_, quantScaleCount_, scale);
        } else {
            TileXRMxfp8Quant::DequantizeAndAccumulate<fp8_e4m3fn_t>(
                tokenAddr, scaleAddr, scaleWorkAddr, sumAddr,
                axisH_, quantScaleCount_, scale);
        }
        packedInputQueue_.FreeTensor(packedLocal);
        return;
    }
#endif
    Cast(rowFloat, packedLocal, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    if (scale == 1.0f) {
        Add(sumLocal, sumLocal, rowFloat, axisH_);
    } else {
        Muls(mulLocal, rowFloat, scale, axisH_);
        PipeBarrier<PIPE_V>();
        Add(sumLocal, sumLocal, mulLocal, axisH_);
    }
    PipeBarrier<PIPE_V>();
    packedInputQueue_.FreeTensor(packedLocal);
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::AddSharedExpertX(uint32_t tokenIndex,
    LocalTensor<float> sumLocal, LocalTensor<float> rowFloat)
{
    GlobalTensor<XType> sharedGlobal = sharedExpertXGM_;
    LocalTensor<XType> packedLocal = packedInputQueue_.AllocTensor<XType>();
    const DataCopyExtParams copyParams {1U, rowBytes_, 0U, 0U, 0U};
    const DataCopyPadExtParams<XType> copyPad {false, 0U, 0U, 0U};
    DataCopyPad(packedLocal, sharedGlobal[tokenIndex * axisH_], copyParams, copyPad);
    packedInputQueue_.EnQue(packedLocal);
    packedLocal = packedInputQueue_.DeQue<XType>();
    Cast(rowFloat, packedLocal, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    Add(sumLocal, sumLocal, rowFloat, axisH_);
    PipeBarrier<PIPE_V>();
    packedInputQueue_.FreeTensor(packedLocal);
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::ClearTokenFlags(uint32_t tokenIndex,
    LocalTensor<float> clearLocal)
{
    const DataCopyExtParams clearParams {static_cast<uint16_t>(blockCntPerToken_),
        TileXR::DATA_AS_FLAG_FLAG_BYTES, 0U, TileXR::DATA_AS_FLAG_PAYLOAD_BYTES, 0U};
    for (uint32_t slot = 0U; slot < slotCount_; ++slot) {
        GM_ADDR slotAddr = CombineWindowBase(epRankId_) +
            (static_cast<uint64_t>(tokenIndex) * slotCount_ + slot) * packedRowBytes_;
        GlobalTensor<float> flagGlobal;
        flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            slotAddr + TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES));
        DataCopyPad(flagGlobal, clearLocal, clearParams);
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::WriteOutput(uint32_t tokenIndex,
    LocalTensor<float> sumLocal)
{
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> outputLocal = outputQueue_.AllocTensor<XType>();
    Cast(outputLocal, sumLocal, RoundMode::CAST_RINT, axisH_);
    outputQueue_.EnQue(outputLocal);
    outputLocal = outputQueue_.DeQue<XType>();
    const DataCopyExtParams outputParams {1U, rowBytes_, 0U, 0U, 0U};
    DataCopyPad(yOutGM_[tokenIndex * axisH_], outputLocal, outputParams);
    outputQueue_.FreeTensor(outputLocal);
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::LocalWindowCopy()
{
    LocalTensor<float> rowFloat = rowFloatBuf_.Get<float>();
    LocalTensor<float> mulLocal = mulFloatBuf_.Get<float>();
    LocalTensor<float> sumLocal = sumFloatBuf_.Get<float>();
    LocalTensor<float> scalesLocal = expertScaleBuf_.Get<float>();
    LocalTensor<float> clearLocal = clearFlagBuf_.Get<float>();
    const DataCopyExtParams scaleParams {
        1U, static_cast<uint32_t>(axisK_ * sizeof(float)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> scalePad {false, 0U, 0U, 0U};

    uint32_t tokenCount = axisBS_ / aivNum_;
    const uint32_t remainder = axisBS_ % aivNum_;
    uint32_t beginIndex = tokenCount * coreIdx_;
    if (coreIdx_ < remainder) {
        ++tokenCount;
        beginIndex += coreIdx_;
    } else {
        beginIndex += remainder;
    }
    if (tokenCount == 0U) {
        return;
    }
    const uint32_t endIndex = beginIndex + tokenCount;
    LocalTensor<int32_t> tokenStatus = tokenStatusBuf_.Get<int32_t>();
    Duplicate(tokenStatus, 0, tokenCount);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t completed = 0U;
    while (completed != tokenCount) {
        for (uint32_t tokenIndex = beginIndex; tokenIndex < endIndex; ++tokenIndex) {
            const uint32_t localIndex = tokenIndex - beginIndex;
            if (tokenStatus.GetValue(localIndex) == 1) {
                continue;
            }
            if (!TokenActive(tokenIndex)) {
                Duplicate(sumLocal, 0.0f, axisH_);
                WriteOutput(tokenIndex, sumLocal);
            } else {
                if (!CheckPackedTokenArrive(tokenIndex)) {
                    continue;
                }
                Duplicate(sumLocal, 0.0f, axisH_);
                if (hasExpertScales_) {
                    DataCopyPad(scalesLocal, expertScalesGM_[tokenIndex * axisK_], scaleParams, scalePad);
                    SyncFunc<AscendC::HardEvent::MTE2_S>();
                }
                for (uint32_t topk = 0U; topk < axisK_; ++topk) {
                    const float scale = hasExpertScales_ ? scalesLocal.GetValue(topk) : 1.0f;
                    CopyAndAccumulateSlot(tokenIndex, topk, scale, sumLocal, rowFloat, mulLocal);
                }
                for (uint32_t shared = 0U; shared < sharedExpertNum_; ++shared) {
                    CopyAndAccumulateSlot(tokenIndex, axisK_ + shared, 1.0f, sumLocal, rowFloat, mulLocal);
                }
                if (hasSharedExpertX_) {
                    AddSharedExpertX(tokenIndex, sumLocal, rowFloat);
                }
                ClearTokenFlags(tokenIndex, clearLocal);
                WriteOutput(tokenIndex, sumLocal);
            }
            tokenStatus.SetValue(localIndex, 1);
            ++completed;
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType>
__aicore__ inline void MoeDistributeCombineV2A5Mte<XType>::Process()
{
    if ASCEND_IS_AIV {
        InitSendBuffers();
        ExpertAlltoAllDispatchCopyAdd();
        PipeBarrier<PIPE_ALL>();
        InitReceiveBuffers();
        LocalWindowCopy();
    }
}

template <typename XType>
__aicore__ inline void RunCombine(GM_ADDR commArgsGM, GM_ADDR expertOutGM, GM_ADDR assistInfoGM,
    GM_ADDR sendCountsGM, GM_ADDR expertScalesGM, GM_ADDR xActiveMaskGM, GM_ADDR sharedExpertXGM,
    GM_ADDR yOutGM, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t activeMaskType, int64_t quantMode, int64_t magic)
{
    TPipe pipe;
    MoeDistributeCombineV2A5Mte<XType> op;
    op.Init(commArgsGM, expertOutGM, assistInfoGM, sendCountsGM, expertScalesGM, xActiveMaskGM,
        sharedExpertXGM, yOutGM, bs, h, topK, moeExpertNum, sharedExpertNum, sharedExpertRankNum,
        globalBs, activeMaskType, quantMode, magic, &pipe);
    op.Process();
}

} // namespace Mc2Kernel

#endif // TILEXR_EP_KERNELS_TILEXR_EP_COMBINE_MEMORY_KERNEL_H
