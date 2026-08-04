#ifndef TILEXR_EP_KERNELS_TILEXR_EP_DISPATCH_MEMORY_KERNEL_H
#define TILEXR_EP_KERNELS_TILEXR_EP_DISPATCH_MEMORY_KERNEL_H

#include <cstdint>

#include "adv_api/reduce/sum.h"
#include "comm_args.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "tilexr_ep_mxfp8_quant.h"
#endif
#include "tilexr_types.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

namespace Mc2Kernel {
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL;
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL;
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL;
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint64_t SPLIT_BLOCK_COUNT = 128UL;
constexpr int32_t FULL_MESH_MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256 / sizeof(int32_t);
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_DATA_COUNT = 120U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 8U;
constexpr uint32_t RUNPOS_CALCUMSUM = 2U;
constexpr uint32_t RUNPOS_CUMSUMFLAG = 3U;
constexpr uint32_t RUNPOS_ARRIVECNT = 4U;
constexpr uint8_t VALID_EVENT_FLAG_NUM = 8U;
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U;
constexpr uint8_t STATUS_COUNT_INDEX = UB_ALIGN_DATA_COUNT - 2U;
constexpr uint8_t STATUS_FLAG_INDEX = UB_ALIGN_DATA_COUNT - 1U;
constexpr uint32_t STATUS_COUNT_PATTERN = 1U << STATUS_COUNT_INDEX;
constexpr uint32_t STATUS_FLAG_PATTERN = 1U << STATUS_FLAG_INDEX;
constexpr uint64_t STATUS_FLAG_DUPLICATE_MASK = 0x8080808080808080ULL;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint8_t BUFFER_NUM = 2U;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint8_t COMBINE_IN_DATA_SIZE = 2U;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t EXPAND_IDX_INFO = 3U;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL;
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint32_t ZERONE_STATE_POS = 0U;
constexpr uint32_t OPOSITION_POS = 1U;
constexpr uint32_t TILING_EPRANKID_POS = 2U;
constexpr uint32_t MOE_NUM_POS = 3U;
constexpr uint32_t TILING_WORLDSIZE_POS = 4U;
constexpr uint32_t GLOBALBS_POS = 5U;
constexpr int64_t ACTIVE_MASK_NONE = 0;
constexpr int64_t ACTIVE_MASK_TOKEN = 1;
constexpr int64_t ACTIVE_MASK_EXPERT = 2;
constexpr int64_t MX_QUANT = 4;

using namespace AscendC;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    AscendC::TEventID eventId = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

class TileXRMemoryContext {
public:
    __aicore__ inline void Init(GM_ADDR commArgsGM)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
        rank_ = args_->rank;
        rankSize_ = args_->rankSize;
    }

    __aicore__ inline uint32_t GetEpRankId()
    {
        return static_cast<uint32_t>(rank_);
    }

    __aicore__ inline uint32_t GetEpWorldSize()
    {
        return static_cast<uint32_t>(rankSize_);
    }

    __aicore__ inline GM_ADDR GetStatusDataSpaceGm()
    {
        return GetPeerStateBase(rank_);
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(int32_t rankId, int32_t)
    {
        return GetPeerStateBase(rankId) + A5_MTE_STATE_WIN_SIZE;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(int32_t rankId, int32_t)
    {
        return GetPeerStateBase(rankId);
    }

    __aicore__ inline GM_ADDR GetPeerMemBase(int32_t rankId)
    {
        return args_->peerMems[rankId];
    }

private:
    __aicore__ inline GM_ADDR GetPeerStateBase(int32_t rankId)
    {
        return args_->peerMems[rankId] + TileXR::IPC_DATA_OFFSET;
    }

    __gm__ TileXR::CommArgs *args_{nullptr};
    int32_t rank_{0};
    int32_t rankSize_{0};
};



template <typename XType>
class MoeDistributeDispatchV2FullMesh {
public:
    __aicore__ inline MoeDistributeDispatchV2FullMesh() {}
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR xActiveMask,
        GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
        GM_ADDR assistInfoForCombineOut, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
        int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
        int64_t expertTokenNumsType, int64_t activeMaskType, int64_t quantMode,
        int64_t expandXOutDtype, int64_t magic, TPipe *pipe);
    __aicore__ inline void Run();

private:
    __aicore__ inline void ExpIdsCopyAndMaskCal();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void SetDataStatus();
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void SetTilingDataAndCal();
    __aicore__ inline uint32_t InitWinState(GlobalTensor<uint32_t> selfDataStatusGMTensor,
    uint32_t epRankId, uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t globalBs, uint32_t dataStateSeed,
    TBuf<> dataStateBuf);
    __aicore__ inline void SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf);
    __aicore__ inline void ExpertActiveMaskInit();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalcSendTokenBufNum(TBuf<>& outBuf);
    __aicore__ inline void AllToAllDispatchA5(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf);
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void CalAndSendCntByRank();
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor,
         LocalTensor<uint32_t> &gatherTmpTensor, LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId);

    __aicore__ inline void RunPosRecord(const uint32_t runPos);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlag(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyInAndOut(LocalTensor<int32_t> xOutInt32Tensor,
        GM_ADDR wAddr, uint32_t index, uint32_t srcExpertId, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void FillTriple(LocalTensor<uint8_t> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void TokenToExpert(GlobalTensor<uint8_t> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                        uint32_t srcTokenIndex, uint32_t toExpertIndex);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void TokenToExpertInQuant(GlobalTensor<uint8_t> dstWinGMTensor,
        TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex, uint32_t toExpertIndex);
    template <typename Fp8Type>
    __aicore__ inline void QuantMxfp8(LocalTensor<uint8_t> &outLocal, LocalTensor<XType> &inLocal);
#endif
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindAddrByRankId(rankId, epRankIdOriginal_) + winDataSizeOffset_;
    }


    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindStateAddrByRankId(rankId, epRankIdOriginal_) + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<int32_t> statusTensor_;
    LocalTensor<int32_t> waitStatusTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<uint8_t> outTensor_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<uint8_t> flagCompResultU8_;
    LocalTensor<uint64_t> flagCompResultLtU64_;
    LocalTensor<uint32_t> flagRecvGatherMask_;
    LocalTensor<float> cleanUpTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<uint8_t> xTmpTensor_;
    LocalTensor<uint8_t> quantTensor_;
    LocalTensor<float> quantWorkTensor_;

    LocalTensor<float> flagReduceWorkTensor_;
    LocalTensor<float> flagRecvTensor_;

    TBuf<> statusBuf_;
    TBuf<> recvStatusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> calBeginBuf_;
    TBuf<> calEndBuf_;
    GM_ADDR expandXOutGM_;
    GM_ADDR assistInfoForCombineOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGM_;


    uint32_t syncFlagId_{0};
    uint8_t sendTokenBufNum_{0};
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};

    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0};
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t rankNumPerSharedExpert_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hAlignSize_{0};
    uint32_t inputTensorBytes_{0};
    uint32_t packedPayloadBytes_{0};
    uint32_t quantComputeCount_{0};
    uint32_t quantComputeScaleCount_{0};
    uint32_t startId_;
    uint32_t endId_;
    uint32_t sendNum_;
    uint32_t statusCntAlign_;
    uint32_t dataState_{0};
    uint32_t dataStateSeed_{0};
    uint32_t tBufRealSize_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    uint64_t totalWinSize_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t axisHCommu_{0};
    uint32_t hCommuSize_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t quantWorkBytes_{0};
    uint32_t quantTensorBytes_{0};
    bool useMxfp8_{false};
    bool useMxfp8E4M3_{false};

    uint32_t expertIdsBufSize_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedAllToAll_{0};
    uint32_t maxSize_{0};
    uint32_t expertIdsSize_{0};
    uint32_t globalBS_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};
    uint64_t totalUbSize_{0};
    TileXRMemoryContext ctx_;

    DataCopyParams hCopyParams_;
    DataCopyParams dataStateParams_{1U, sizeof(uint32_t), 0U, 0U};
};

template <typename XType>
__aicore__ inline uint32_t MoeDistributeDispatchV2FullMesh<XType>::InitWinState(GlobalTensor<uint32_t> selfDataStatusGMTensor,
    uint32_t epRankId, uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t globalBs, uint32_t dataStateSeed,
    TBuf<> dataStateBuf)
{
    LocalTensor<uint64_t> dataStateLocalTensor64 = dataStateBuf.Get<uint64_t>();
    LocalTensor<uint32_t> dataStateLocalTensor = dataStateBuf.Get<uint32_t>();
    DataCopy(dataStateLocalTensor, selfDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t dataState = dataStateSeed;
    dataStateLocalTensor.SetValue(OPOSITION_POS, 1);
    dataStateLocalTensor.SetValue(TILING_EPRANKID_POS, epRankId);
    dataStateLocalTensor.SetValue(MOE_NUM_POS, moeExpertNum);
    dataStateLocalTensor.SetValue(TILING_WORLDSIZE_POS, epWorldSize);
    dataStateLocalTensor.SetValue(GLOBALBS_POS, globalBs);
    uint32_t opCnt = dataStateLocalTensor64.GetValue(OP_CNT_POSUL);
    dataStateLocalTensor64.SetValue(OP_CNT_POSUL, opCnt + 1);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfDataStatusGMTensor, dataStateLocalTensor, UB_ALIGN / sizeof(uint32_t));
    return dataState;
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::SetTilingDataAndCal()
{
    copyInAxisH_ = axisH_;
    copyOutAxisH_ = axisH_;
    isShareExpertRankFlag_ = epRankId_ < static_cast<int32_t>(sharedExpertRankNum_);
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    expertIdsCnt_ = axisBS_ * axisK_;
    hOutSize_ = copyOutAxisH_ * (useMxfp8_ ? sizeof(uint8_t) : sizeof(XType));
    if (useMxfp8_) {
        scaleOutBytes_ = ((Ceil(axisH_, 32U) + 1U) / 2U) * 2U;
        quantComputeCount_ = Ceil(axisH_, 256U) * 256U;
        quantComputeScaleCount_ = quantComputeCount_ / 32U;
        hAlignSize_ = quantComputeCount_ * sizeof(XType);
        const uint32_t quantPayloadBytes = Ceil(hOutSize_, 256U) * 256U + scaleOutBytes_;
        tokenQuantAlign_ = Ceil(quantPayloadBytes, UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
        quantWorkBytes_ = Ceil(
            Ceil(quantComputeScaleCount_, 32U) * 32U * sizeof(float) +
                quantComputeScaleCount_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN;
    } else {
        hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
        tokenQuantAlign_ = hAlignSize_ / sizeof(int32_t);
    }
    hOutSizeAlign_ = tokenQuantAlign_ * sizeof(int32_t) + UB_ALIGN;
    blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    packedPayloadBytes_ = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE;
    inputTensorBytes_ = useMxfp8_ ? hAlignSize_ : packedPayloadBytes_;
    quantTensorBytes_ = packedPayloadBytes_;
    hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    axisHCommu_ = hCommuSize_;
    expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_;
    rscvStatusNum_ = isShareExpertRankFlag_ ? epWorldSize_ : epWorldSize_ * moeExpertNumPerRank_;
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    statusCntAlign_ = Ceil(totalExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    aivUsedCumSum_ = totalExpertNum_ / 16;
    aivUsedCumSum_ = aivUsedCumSum_ == 0 ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = aivUsedCumSum_ >= aivNum_ / 2 ? aivNum_ / 2 : aivUsedCumSum_;
    aivUsedCumSum_ = aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    aivUsedCumSum_ = aivUsedCumSum_ >= rscvStatusNum_ ? rscvStatusNum_ : aivUsedCumSum_;
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = aivUsedAllToAll_ * sharedExpertNum_ / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_;

    expertIdsSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t expertMaskRowBytes = Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    uint32_t expertMaskBytes = axisBS_ * expertMaskRowBytes * sizeof(half);
    maxSize_ = expertIdsSize_ > expertMaskBytes ? expertIdsSize_ : expertMaskBytes;
    if (useMxfp8_ && maxSize_ < quantWorkBytes_) {
        maxSize_ = quantWorkBytes_;
    }
    totalUbSize_ = FULL_MESH_MAX_UB_SIZE;
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::SetDataStatus()
{
    statusDataSpaceGM_ = ctx_.GetStatusDataSpaceGm();
    selfDataStatusGMTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t *>(statusDataSpaceGM_ + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);

    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankId_, epWorldSize_, moeExpertNum_, globalBS_,
        dataStateSeed_, dataStateBuf);
    uint64_t hSizeAlignCombine =
        Ceil(axisH_ * COMBINE_IN_DATA_SIZE, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_SIZE;
    winDataSizeOffset_ = dataState_ * (totalWinSize_ / BUFFER_NUM) +
        axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::Init(GM_ADDR commArgs, GM_ADDR x,
    GM_ADDR expertIds, GM_ADDR xActiveMask, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut,
    GM_ADDR sendCountsOut, GM_ADDR assistInfoForCombineOut, int64_t bs, int64_t h, int64_t topK,
    int64_t moeExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t expertTokenNumsType, int64_t activeMaskType, int64_t quantMode,
    int64_t expandXOutDtype, int64_t magic, TPipe *pipe)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tpipe_ = pipe;
    tpipe_->InitBuffer(calBeginBuf_, UB_ALIGN);
    aivId_ = GetBlockIdx();
    ctx_.Init(commArgs);

    axisBS_ = static_cast<uint32_t>(bs);
    axisH_ = static_cast<uint32_t>(h);
    axisK_ = static_cast<uint32_t>(topK);
    moeExpertNum_ = static_cast<uint32_t>(moeExpertNum);
    sharedExpertNum_ = static_cast<uint32_t>(sharedExpertNum);
    sharedExpertRankNum_ = static_cast<uint32_t>(sharedExpertRankNum);
    globalBS_ = static_cast<uint32_t>(globalBs);
    expertTokenNumsType_ = static_cast<uint32_t>(expertTokenNumsType);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    useMxfp8_ = quantMode == MX_QUANT;
#else
    useMxfp8_ = false;
#endif
    useMxfp8E4M3_ = expandXOutDtype == TileXR::TILEXR_DATA_TYPE_FP8E4M3;
    dataStateSeed_ = static_cast<uint32_t>(magic) & 1U;
    epRankId_ = static_cast<int32_t>(ctx_.GetEpRankId());
    epRankIdOriginal_ = epRankId_;
    epWorldSize_ = ctx_.GetEpWorldSize();
    epWorldSizeOriginal_ = epWorldSize_;
    aivNum_ = GetBlockNum();
    axisMaxBS_ = globalBS_ / epWorldSize_;
    isTokenMaskFlag_ = activeMaskType == ACTIVE_MASK_TOKEN;
    isExpertMaskFlag_ = activeMaskType == ACTIVE_MASK_EXPERT;

    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(x));
    xActiveMaskGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(xActiveMask));
    expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertIds));
    expertTokenNumsOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOut));
    dynamicScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dynamicScalesOut));

    SetTilingDataAndCal();
    uint64_t workspaceStatusNum = static_cast<uint64_t>(epWorldSize_) * moeExpertNumPerRank_;
    uint64_t workspaceBytes = Ceil(aivNum_ * workspaceStatusNum * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    totalWinSize_ = TileXR::IPC_BUFF_MAX_SIZE - A5_MTE_STATE_WIN_SIZE - workspaceBytes;
    SetDataStatus();

    expandXOutGM_ = expandXOut;
    assistInfoForCombineOutGM_ = assistInfoForCombineOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCntWorkspaceGM_ = ctx_.GetStatusDataSpaceGm() + TileXR::IPC_BUFF_MAX_SIZE - workspaceBytes;
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    windowInstatusFp32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(statusSpaceGM_));
    selfRankWinInGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(statusDataSpaceGM_));
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
    hCopyParams_ = {1U, static_cast<uint32_t>(copyInAxisH_ * sizeof(XType)), 0U, 0U};
    dataStateParams_ = {1U, sizeof(uint32_t), 0U, 0U};
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::FillTriple(
    LocalTensor<uint8_t> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;
    xOutTint32(tokenQuantAlign_ + 2) = k;
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <typename XType>
template <typename Fp8Type>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::QuantMxfp8(
    LocalTensor<uint8_t> &outLocal, LocalTensor<XType> &inLocal)
{
    __ubuf__ XType *srcAddr = reinterpret_cast<__ubuf__ XType *>(inLocal.GetPhyAddr());
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(quantWorkTensor_.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        quantWorkTensor_[Ceil(quantComputeScaleCount_, 32U) * 32U].GetPhyAddr());
    __ubuf__ int8_t *outAddr = reinterpret_cast<__ubuf__ int8_t *>(outLocal.GetPhyAddr());
    __ubuf__ uint16_t *scaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        outLocal[Ceil(axisH_, 256U) * 256U].GetPhyAddr());

    TileXRMxfp8Quant::ComputeMaxExp(srcAddr, maxExpAddr, quantComputeCount_);
    TileXRMxfp8Quant::ComputeScale<Fp8Type>(
        maxExpAddr, scaleAddr, halfScaleAddr, quantComputeScaleCount_);
    TileXRMxfp8Quant::ComputeFp8Data<XType, Fp8Type, RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
        srcAddr, halfScaleAddr, outAddr, quantComputeCount_);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::TokenToExpertInQuant(
    GlobalTensor<uint8_t> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
    uint32_t srcTokenIndex, uint32_t toExpertIndex)
{
    DataCopyPadParams copyPadParams{true, 0U, 0U, 0U};
    LocalTensor<XType> xInTensor = inQueue.AllocTensor<XType>();
    Duplicate<uint32_t>(xInTensor.template ReinterpretCast<uint32_t>(), 0U,
        hAlignSize_ / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XType>();
    if (useMxfp8E4M3_) {
        QuantMxfp8<fp8_e4m3fn_t>(quantTensor_, xInTensor);
    } else {
        QuantMxfp8<fp8_e5m2_t>(quantTensor_, xInTensor);
    }
    inQueue.FreeTensor<XType>(xInTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
    FillTriple(quantTensor_, srcTokenIndex, toExpertIndex);
    SyncFunc<AscendC::HardEvent::S_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    LocalTensor<int32_t> quantTensorInt32 = quantTensor_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 =
        outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_].template ReinterpretCast<int32_t>();
    Copy(outTensorInt32, quantTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    Copy(outTensorInt32[64], quantTensorInt32[64], uint64_t(56),
        uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    DataCopy(dstWinGMTensor, outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_], axisHCommu_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    ++syncFlagId_;
}
#endif

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::TokenToExpert(
    GlobalTensor<uint8_t> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
    uint32_t srcTokenIndex, uint32_t toExpertIndex)
{
    DataCopyPadParams copyPadParams{false, 0U, 0U, 0U};
    LocalTensor<XType> xInTensor = inQueue.AllocTensor<XType>();
    Duplicate<uint32_t>(xInTensor.template ReinterpretCast<uint32_t>(), 0U,
        inputTensorBytes_ / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XType>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    LocalTensor<uint8_t> xInTensorBytes = xInTensor.template ReinterpretCast<uint8_t>();
    FillTriple(xInTensorBytes, srcTokenIndex, toExpertIndex);
    SyncFunc<AscendC::HardEvent::S_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    LocalTensor<int32_t> xInTensorInt32 = xInTensorBytes.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 =
        (outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_]).template ReinterpretCast<int32_t>();

    Copy(outTensorInt32, xInTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});

    Copy(outTensorInt32[64], xInTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    inQueue.FreeTensor<XType>(xInTensor);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    DataCopy(dstWinGMTensor, outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_], axisHCommu_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    syncFlagId_ ++;
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront)

{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else if (aivId_ >= aivUsedAllToAll_) {
        newAivId = aivId_ - aivUsedAllToAll_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_;
    }
    startTokenId = sendTokenNum * newAivId;
    if (newAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{

    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * sendTokenBufNum_ / sizeof(float));
    PipeBarrier<PIPE_V>();

    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_;
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {return;}

    GlobalTensor<uint8_t> dstWinGMTensor;
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;
    syncFlagId_ = 0;
    for (int i = 0; i < sendTokenBufNum_; i ++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t sendTokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        dstWinGMTensor.SetGlobalBuffer((__gm__ uint8_t *)(uint64_t(GetWindAddrByRankId(toRankId))
            + expertPerSizeOnWin_ * epRankId_ + sendTokenIndex * hCommuSize_));
        uint32_t srcTokenIndex = sendTokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = validBsIndexTensor_.GetValue(sendTokenIndex);
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (useMxfp8_) {
            TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, axisK_ + toSharedExpertIndex);
        } else {
#endif
            TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, axisK_ + toSharedExpertIndex);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        }
#endif
    }
    for (int i = 0; i < sendTokenBufNum_; i ++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue,
    TBuf<> expertMaskBuf, TBuf<> outBuf)
{

    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * sendTokenBufNum_ / sizeof(float));
    uint32_t validTokenNum = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    GlobalTensor<uint8_t> dstWinGMTensor;

    int32_t dstTokenIdx = 0;
    syncFlagId_ = 0;

    for (int i = 0; i < sendTokenBufNum_; i ++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }

    for (int32_t index = aivId_; index < validTokenNum; index += moeUsedAivNum_) {
        int32_t tokenId = index / axisK_;
        int32_t topKId = index % axisK_;
        int32_t expertId = validExpertIdsTensor_(index);
        if (expertId >= moeExpertNum_ || expertId < 0)
            continue;
        int32_t toRankId = expertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        CalTokenSendExpertCnt(expertId, index, dstTokenIdx);

        dstWinGMTensor.SetGlobalBuffer((__gm__ uint8_t *)(uint64_t(GetWindAddrByRankId(toRankId))
            + expertPerSizeOnWin_ * ((epRankId_ + toRankId) % epWorldSize_ * moeExpertNumPerRank_
            + expertId % moeExpertNumPerRank_) + dstTokenIdx * hCommuSize_));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (useMxfp8_) {
            TokenToExpertInQuant(dstWinGMTensor, inQueue, tokenId, topKId);
        } else {
#endif
            TokenToExpert(dstWinGMTensor, inQueue, tokenId, topKId);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        }
#endif
    }

    for (int i = 0; i < sendTokenBufNum_; i ++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CalcSendTokenBufNum(TBuf<>& outBuf)
{
    tpipe_->InitBuffer(calEndBuf_, UB_ALIGN);
    uint64_t beiginUbAddr = (calBeginBuf_.Get<uint8_t>()).GetPhyAddr();
    uint64_t endUbAddr = (calEndBuf_.Get<uint8_t>()).GetPhyAddr();
    uint64_t remainUbSize = totalUbSize_ - (endUbAddr - beiginUbAddr + UB_ALIGN);

    sendTokenBufNum_ = remainUbSize / hCommuSize_;
    if (sendTokenBufNum_ > VALID_EVENT_FLAG_NUM)
        sendTokenBufNum_ = VALID_EVENT_FLAG_NUM;
    if (sendTokenBufNum_ == 0) { return; }


    tpipe_->InitBuffer(outBuf, hCommuSize_ * sendTokenBufNum_);
    outTensor_ = outBuf.Get<uint8_t>();
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::AllToAllDispatch()
{
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<> outBuf, expertMaskBuf, inQueueCleanBuf, calTempBuf;
    TBuf<> quantBuf, quantWorkBuf, quantAuxBuf;
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    AscendC::TBufPool<AscendC::TPosition::VECIN> tbufPool0, tbufPool1;
    tpipe_->InitBufPool(tbufPool0, BUFFER_NUM * inputTensorBytes_);
    tpipe_->InitBufPool(tbufPool1, BUFFER_NUM * hAlignSize_, tbufPool0);
    tbufPool0.InitBuffer(inQueue, BUFFER_NUM, inputTensorBytes_);
    tbufPool1.InitBuffer(inQueueCleanBuf, BUFFER_NUM * hAlignSize_);
    if (useMxfp8_) {
        LocalTensor<uint8_t> cleanTensor = inQueueCleanBuf.Get<uint8_t>();
        Duplicate<uint32_t>(cleanTensor.ReinterpretCast<uint32_t>(), 0U,
            BUFFER_NUM * hAlignSize_ / sizeof(uint32_t));
    }

    uint32_t calTokenIdxBuffSize = Ceil(axisBS_ * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    bool needMaskCalFlag = isTokenMaskFlag_ || isExpertMaskFlag_;
    if (needMaskCalFlag) {
        tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);
    }
    if (useMxfp8_) {
        tpipe_->InitBuffer(quantBuf, quantTensorBytes_);
        tpipe_->InitBuffer(quantWorkBuf, maxSize_);
        tpipe_->InitBuffer(quantAuxBuf, maxSize_);
        quantTensor_ = quantBuf.Get<uint8_t>();
        Duplicate<uint32_t>(quantTensor_.ReinterpretCast<uint32_t>(), 0U,
            quantTensorBytes_ / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        quantWorkTensor_ = quantWorkBuf.Get<float>();
        dstExpBuf_ = quantWorkBuf;
        subExpBuf_ = quantAuxBuf;
    } else if (needMaskCalFlag) {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_);
        tpipe_->InitBuffer(subExpBuf_, maxSize_);
    } else {
        tpipe_->InitBuffer(dstExpBuf_, calTokenIdxBuffSize);
        tpipe_->InitBuffer(subExpBuf_, calTokenIdxBuffSize);
    }
    tpipe_->InitBuffer(calTempBuf, calTokenIdxBuffSize);
    workLocalTensor_ = calTempBuf.Get<float>();
    ExpIdsCopyAndMaskCal();
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    AllToAllDispatchA5(inQueue, expertMaskBuf, outBuf);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::AllToAllDispatchA5(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf)
{
    CalcSendTokenBufNum(outBuf);
    if ((aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
        SendToSharedExpert(inQueue, outBuf);
    } else {
        SendToMoeExpert(inQueue, expertMaskBuf, outBuf);
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    if (calCnt < axisK_) {
        curExpertCnt = 0;
        return;
    }
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CalAndSendCntByRank()
{

    uint32_t startRankId, endRankId, sendRankNum;
    uint32_t startExpertId, endExpertId, sendExpertNum;
    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    uint64_t mask[2] = {STATUS_FLAG_DUPLICATE_MASK, 0};
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1, 8);
    SyncFunc<AscendC::HardEvent::V_S>();

    GlobalTensor<int32_t> rankGMTensor;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;

    for (uint32_t dstRankId = newAivId; dstRankId < epWorldSize_; dstRankId += aivUsedCumSum_) {

        if (dstRankId >= sharedExpertRankNum_) {
            startExpertId = (dstRankId - sharedExpertRankNum_) * moeExpertNumPerRank_;
            endExpertId = startExpertId +  moeExpertNumPerRank_;
            for (uint32_t curMoeExpertId = startExpertId; curMoeExpertId < endExpertId; ++curMoeExpertId) {
                int32_t curExpertCnt = 0;
                int32_t cntPosIndex =
                    (curMoeExpertId + sharedExpertRankNum_) * UB_ALIGN_DATA_COUNT + STATUS_COUNT_INDEX;

                if (sendToMoeExpTokenCnt_ > 0) {
                    CalTokenSendExpertCnt(curMoeExpertId, maskCnt, curExpertCnt);
                }
                statusTensor_.SetValue(cntPosIndex, curExpertCnt);
            }
        } else {
            int32_t curExpertCnt = 0;
            int32_t cntPosIndex = dstRankId * UB_ALIGN_DATA_COUNT + STATUS_COUNT_INDEX;

            if (activeMaskBsCnt_ > 0) {
                if (dstRankId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                    curExpertCnt = activeMaskBsCnt_;
                }
            }
            statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        }
    }
    if (newAivId < epWorldSize_)
        SyncFunc<AscendC::HardEvent::S_MTE3>();
    for (uint32_t dstRankId = newAivId; dstRankId < epWorldSize_; dstRankId += aivUsedCumSum_) {
        uint32_t offset = STATE_OFFSET * epRankId_;
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        if (dstRankId >= sharedExpertRankNum_) {
            uint32_t startStatusIdx =
                (dstRankId - sharedExpertRankNum_) * moeExpertNumPerRank_ + sharedExpertRankNum_;
            DataCopyParams cntCopyParams = {uint16_t(moeExpertNumPerRank_), 1U, 0U, uint16_t(epWorldSize_ - 1)};
            DataCopy<int32_t>(rankGMTensor, statusTensor_[startStatusIdx * UB_ALIGN_DATA_COUNT], cntCopyParams);
        } else {
            DataCopy<int32_t>(rankGMTensor, statusTensor_[dstRankId * UB_ALIGN_DATA_COUNT], UB_ALIGN_DATA_COUNT);
        }
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::BufferInit()
{
    uint32_t waitStatusBufSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize = (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize);
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);
    uint32_t statusBufSize = rscvStatusNum_ * UB_ALIGN;
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(recvStatusBuf_, statusBufSize);
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {STATUS_FLAG_DUPLICATE_MASK, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::GatherSumRecvCnt(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, STATUS_COUNT_PATTERN);
    uint32_t mask = UB_ALIGN_DATA_COUNT;
    SyncFunc<AscendC::HardEvent::S_V>();



    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
        {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();


    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN)
        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);


    uint32_t newAivId = aivId_ - aivUsedAllToAll_;

    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);

    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor, sumIntriParams);
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId)

{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1), 0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();


    while (true) {
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }


    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }

    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();

    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);

    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = UB_ALIGN_DATA_COUNT;
    gatherTmpTensor.SetValue(0, STATUS_FLAG_PATTERN);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    uint64_t gatheredFlagCount = 0;
    uint32_t flagSumInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams flagSumParams{1, flagSumInner, recStatusNumPerCore_};

    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)], intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        gatheredFlagCount = 0;
        GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
            {1, static_cast<uint16_t>(recStatusNumPerCore_), 1, 0}, gatheredFlagCount);
        PipeBarrier<PIPE_V>();
        Sum(statusSumOutTensor, gatherMaskOutTensor, flagSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    RunPosRecord(RUNPOS_CALCUMSUM);

    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CalRecvAndSetFlag()
{

    LocalTensor<int32_t> outCountLocal;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    GetCumSum(outCountLocal, newAivId);

    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    waitStatusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = waitStatusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + STATUS_COUNT_INDEX);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = (__gm__ uint8_t*)(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer((__gm__ int32_t*)wAddr);
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);

    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    uint8_t repeatTimes = Ceil(aivNum_, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();

    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_MTE3>();

    DataCopy(selfRankWinInGMTensor_[(CUMSUM_FLAG_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor, sumIntriParams);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CalCumSum()
{

    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);
    tpipe_->InitBuffer(subExpBuf_, maxSize_);
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(statusBuf_, statusCntAlign_ * UB_ALIGN);
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    statusTensor_ = statusBuf_.Get<int32_t>();
    ExpIdsCopyAndMaskCal();
    CalAndSendCntByRank();
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    BufferInit();
    WaitDispatch();
    CalRecvAndSetFlag();

}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::WaitCumSumFlag()
{

    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (CUMSUM_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    while (true) {
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    RunPosRecord(RUNPOS_CUMSUMFLAG);

    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum)
{

    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    if (aivId_ == 0) {
        uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
        int64_t lastVal = 0;
        uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
        LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
        for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
            if (expertTokenNumsType_ == 0) {
                expertTokenNumsLocalTensor(localExpertIdx) = int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ +
                epWorldSize_ - 1));
            } else {
                expertTokenNumsLocalTensor(localExpertIdx) = int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ +
                epWorldSize_ - 1)) - lastVal;
                lastVal = int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ + epWorldSize_ -1));
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)),
            0U, 0U, 0U};
        DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
    }

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startId_; index < endId_; index++) {
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            validNum += 1;
        }
    }
}

template <typename XType>
__aicore__ inline uint32_t MoeDistributeDispatchV2FullMesh<XType>::CheckDataArriveWithFlag(uint32_t srcExpDataIdx,
    int32_t beginIdx, int32_t copyCnt)
{
    uint32_t flagNum = blockCntPerToken_ * uint32_t(copyCnt);
    DataCopyParams expFlagCopyParams{static_cast<uint16_t>(flagNum), 1U,
        static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE / UB_ALIGN), 0U};
    GlobalTensor<float> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ +
        beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE;
    dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
    DataCopy(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<float> flagSumOutTensor = scalarBuf_.Get<float>();
    ReduceSum(flagSumOutTensor, flagRecvTensor_, flagReduceWorkTensor_, 1U, flagNum, 1U);
    SyncFunc<AscendC::HardEvent::V_S>();
    return flagSumOutTensor.GetValue(0) == 1.0f * flagNum ? static_cast<uint32_t>(copyCnt) : 0U;
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CopyInAndOut(
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t srcExpertId,
    uint32_t dstPosition, uint32_t arriveCount)
{
    GlobalTensor<uint8_t> dataFlagGlobal, expandXOutGlobal;
    GlobalTensor<int32_t> assistGlobal;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(wAddr));
    expandXOutGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t *>(expandXOutGM_) + dstPosition * hOutSize_);
    assistGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(assistInfoForCombineOutGM_) + dstPosition * 4);

    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount),
        static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint16_t>(UB_ALIGN), 0};
    DataCopyExtParams tokenCopyParams{static_cast<uint16_t>(arriveCount), hOutSize_,
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams scalesCopyParams{static_cast<uint16_t>(arriveCount), scaleOutBytes_,
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(xTmpTensor_,
        dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_],
        srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    if (useMxfp8_) {
        LocalTensor<uint8_t> scalesLocal = xTmpTensor_[Ceil(axisH_, 256U) * 256U];
        DataCopyPad(dynamicScalesOutGMTensor_[dstPosition * scaleOutBytes_], scalesLocal, scalesCopyParams);
    }
    DataCopyPad(expandXOutGlobal, xTmpTensor_, tokenCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    uint32_t tokenStride = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t);
    int32_t moeExpertId = static_cast<int32_t>(sharedExpertNum_) +
        (epRankId_ - static_cast<int32_t>(sharedExpertRankNum_)) *
            static_cast<int32_t>(moeExpertNumPerRank_) +
        static_cast<int32_t>(srcExpertId / epWorldSize_);
    for (uint32_t token = 0; token < arriveCount; ++token) {
        uint32_t srcOffset = token * tokenStride + tokenQuantAlign_;
        uint32_t dstOffset = token * 4;
        int32_t topKId = xOutInt32Tensor.GetValue(srcOffset + 2);
        xOutInt32Tensor.SetValue(dstOffset, xOutInt32Tensor.GetValue(srcOffset));
        xOutInt32Tensor.SetValue(dstOffset + 1, xOutInt32Tensor.GetValue(srcOffset + 1));
        xOutInt32Tensor.SetValue(dstOffset + 2, topKId);
        xOutInt32Tensor.SetValue(dstOffset + 3,
            isShareExpertRankFlag_ ? topKId - static_cast<int32_t>(axisK_) : moeExpertId);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams assistCopyParams{1U,
        static_cast<uint32_t>(arriveCount * 4 * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(assistGlobal, xOutInt32Tensor, assistCopyParams);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / hCommuSize_;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpertId = 0;
    uint32_t dstPosition = 0;
    uint32_t arriveCount = 0;
    uint32_t copyCnt = 0;
    uint32_t srcDataBlockIdx = 0;
    uint32_t flagNum = blockCntPerToken_ * maxCopyTokenCnt;
    uint32_t reduceWorkSize = Ceil(flagNum * sizeof(float), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    uint32_t flagRecvSize = flagNum * UB_ALIGN;
    GlobalTensor<float> cleanGlobal;
    flagReduceWorkTensor_ = tBuf.GetWithOffset<float>(reduceWorkSize / sizeof(float), 0);
    flagRecvTensor_ = tBuf.GetWithOffset<float>(flagRecvSize / sizeof(float), reduceWorkSize);
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    DataCopyParams cleanUpParams = {uint16_t(blockCntPerToken_), 1U, 0U, SPLIT_BLOCK_DATA_SIZE / UB_ALIGN};
    while (true) {
        if (expertLeftNumTensor_(index) == 0) {
            index = (index + 1) % validNum;
            continue;
        }
        srcExpertId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index);
        srcDataBlockIdx = srcExpertId % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
        if (!isShareExpertRankFlag_) {
            srcDataBlockIdx = (srcExpertId + epRankId_) % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
        }
        arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        if (arriveCount == copyCnt) {
            dstPosition = srcExpertId != 0 ? sendCntTensor_(srcExpertId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
            CopyInAndOut(xOutInt32Tensor, wAddr, index, srcExpertId, dstPosition, arriveCount);

            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            PipeBarrier<PIPE_ALL>();
            if (expertLeftNumTensor_(index) == 0) {
                cleanGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
                for (uint32_t i = 0; i < expertFinishNumTensor_(index); i++){
                    uint32_t flagIndex = i * SPLIT_BLOCK_COUNT * blockCntPerToken_ + SPLIT_BLOCK_DATA_COUNT;
                    DataCopy(cleanGlobal[flagIndex], cleanUpTensor_, cleanUpParams);
                }
                finishNum++;
            }
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::RunPosRecord(const uint32_t runPos)
{
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, runPos);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams_);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::LocalWindowCopy()
{

    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);

    WaitCumSumFlag();
    if (sendNum_ == 0) {

        return;
    }

    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf, cleanUpBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);
    tpipe_->InitBuffer(cleanUpBuf, blockCntPerToken_ * UB_ALIGN);
    tBufRealSize_ = FULL_MESH_MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
        (expInfoSize * 3) - BUFFER_NUM * UB_ALIGN - blockCntPerToken_ * UB_ALIGN;
    tpipe_->InitBuffer(tBuf, tBufRealSize_);
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    if (validNum == 0) {
        return;
    }
    flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
    flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    cleanUpTensor_ = cleanUpBuf.Get<float>();
    xTmpTensor_ = tBuf.Get<uint8_t>();
    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    Duplicate<float>(cleanUpTensor_, float(0), blockCntPerToken_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::V_S>();
    flagRecvGatherMask_.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_V>();
    WaitAndFormatOutput(tBuf, validNum);
    RunPosRecord(RUNPOS_ARRIVECNT);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::TokenActiveMaskCal()
{

    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    uint32_t axisBsAlignSize = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = gatherMaskTBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CalValidBSCnt(LocalTensor<bool> maskStrideTensor)
{
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    LocalTensor<half> tempTensor = validExpertIndexBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = expertIdsBuf_.Get<half>();
    LocalTensor<half> tokenTargetTensor = validBsIndexTBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = gatherMaskTBuf_.Get<uint32_t>();
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    SumParams axisBsSumParams{1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize);
    PipeBarrier<PIPE_V>();
    Sum(tokenTargetTensor, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::CalValidExpIdx(LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);

    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    LocalTensor<half> tempTensor = subExpBuf_.Get<half>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = expertIdsBuf_.Get<int32_t>();

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();

    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, SIZE_ALIGN_256) * SIZE_ALIGN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::ExpertActiveMaskInit()
{
    uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
    uint32_t validBufferSize = expertIdsSize_ > xActivateMaskSize ? expertIdsSize_ : xActivateMaskSize;
    tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
    validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::ExpertActiveMaskCal()
{

    LocalTensor<bool> maskStrideTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskStrideParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalValidBSCnt(maskStrideTensor);

    LocalTensor<bool> maskInputTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalValidExpIdx(maskInputTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::ExpIdsCopyAndMaskCal()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    if (isExpertMaskFlag_) {
        ExpertActiveMaskInit();
    }

    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    Duplicate<int32_t>(validExpertIdsTensor_, -1, int32_t(expertIdsBufSize_ / sizeof(int32_t)));

    if (isExpertMaskFlag_) {
        LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
        LocalTensor<float> tmpExpertIdsTensorFloat = subExpBuf_.Get<float>();
        LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.ReinterpretCast<uint8_t>();
        DataCopyExtParams expertIdsMaskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> expertIdsMaskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsMaskParams, expertIdsMaskCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        LocalTensor<float> validExpertIdsFloat = validExpertIdsTensor_.ReinterpretCast<float>();
        Select(validExpertIdsFloat, gatherMaskTensorInt8, tmpExpertIdsTensorFloat, static_cast<float>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, expertIdsCnt_);
        SyncFunc<AscendC::HardEvent::V_S>();
    } else {
        uint32_t expertIdsMask = activeMaskBsCnt_ * axisK_;
        uint32_t expertIdsAlignCnt = Ceil(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
        uint32_t rightPadding = expertIdsAlignCnt - expertIdsMask;
        DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{true, 0U, uint8_t(rightPadding), -1};
        DataCopyExtParams expertIdsCntParams{1U, static_cast<uint32_t>(expertIdsMask * sizeof(uint32_t)), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }
}

template <typename XType>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<XType>::Run()
{
    if ASCEND_IS_AIV {
        if (aivId_ < aivUsedAllToAll_) {
            AllToAllDispatch();
        } else {
            CalCumSum();
        }

        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy();
    }
}

} // namespace Mc2Kernel

#endif // TILEXR_EP_KERNELS_TILEXR_EP_DISPATCH_MEMORY_KERNEL_H
