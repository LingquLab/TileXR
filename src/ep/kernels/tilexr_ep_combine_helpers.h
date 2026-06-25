#pragma once

__aicore__ inline TileXREp::EpAssistTuple TileXREpLoadAssistTuple(
    GM_ADDR srcGM, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int32_t> local =
        tBuf.GetWithOffset<int32_t>(TileXREp::kEpAssistTupleInts, kEpScalarUbOffset);
    AscendC::GlobalTensor<int32_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(srcGM), TileXREp::kEpAssistTupleInts);

    AscendC::DataCopyExtParams copyParams {
        1, static_cast<uint32_t>(sizeof(TileXREp::EpAssistTuple)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    TileXREp::EpAssistTuple tuple;
    tuple.srcRank = local.GetValue(0);
    tuple.tokenId = local.GetValue(1);
    tuple.topKId = local.GetValue(2);
    tuple.expertId = local.GetValue(3);
    return tuple;
}

__aicore__ inline int32_t TileXREpGetCombineTokenId(const TileXREp::EpAssistTuple &tuple)
{
    return tuple.tokenId;
}

__aicore__ inline int32_t TileXREpGetCombineTopKId(const TileXREp::EpAssistTuple &tuple)
{
    return tuple.topKId;
}

__aicore__ inline int32_t TileXREpWaitDispatchSlotReady(
    GM_ADDR slotGM, int32_t srcRank, int64_t magic, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    while (true) {
        const uint64_t packed = LoadUint64FromGm(slotGM, tBuf);
        const int32_t slotSrcRank = static_cast<int32_t>((packed >> 32) & 0xffffffffULL);
        const uint64_t slotMagic = LoadUint64FromGm(slotGM + 3 * static_cast<int64_t>(sizeof(uint64_t)), tBuf);
        if (slotSrcRank == srcRank && slotMagic == static_cast<uint64_t>(magic)) {
            return static_cast<int32_t>(packed & 0xffffffffULL);
        }
    }
}

__aicore__ inline void TileXREpClearExpertTokenNums(
    GM_ADDR expertTokenNumsOutGM, int64_t localExpertNum, int64_t expertTokenNumsType)
{
    (void)expertTokenNumsType;
    auto expertTokenNumsOut = reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOutGM);
    for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
        expertTokenNumsOut[localExpert] = 0;
    }
}

__aicore__ inline void TileXREpIncrementExpertTokenNum(
    GM_ADDR expertTokenNumsOutGM, int64_t localExpert, int64_t expertTokenNumsType)
{
    (void)expertTokenNumsType;
    auto expertTokenNumsOut = reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOutGM);
    expertTokenNumsOut[localExpert] = expertTokenNumsOut[localExpert] + 1;
}

__aicore__ inline void TileXREpFinalizeExpertTokenNums(
    GM_ADDR expertTokenNumsOutGM, int64_t localExpertNum, int64_t expertTokenNumsType)
{
    if (expertTokenNumsType != 0) {
        return;
    }

    auto expertTokenNumsOut = reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOutGM);
    int64_t running = 0;
    for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
        running += expertTokenNumsOut[localExpert];
        expertTokenNumsOut[localExpert] = running;
    }
}

__aicore__ inline int64_t TileXREpDrainSourceWindow(GM_ADDR sourceWindow, int32_t slotRank, int32_t srcRank,
    int64_t slotBytes, int64_t payloadBytesPerSlot, int64_t maxRoutesPerSrc, int64_t rowBytes,
    int64_t payloadRowBytes, int64_t quantMode, int64_t localExpertNum, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, GM_ADDR expandXOutGM, GM_ADDR dynamicScalesOutGM, GM_ADDR expertTokenNumsOutGM,
    int64_t expertTokenNumsType, int64_t magic, __gm__ int32_t *epRecvCountsOut, __gm__ int32_t *tpRecvCountsOut,
    __gm__ TileXREp::EpAssistTuple *localAssistBase, int64_t outRecord, GM_ADDR tpPublishWindow,
    int32_t tpPublishRank, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    const int64_t count = TileXREpWaitDispatchSlotReady(sourceWindow + SlotOffset(slotRank, slotBytes), srcRank,
        magic, tBuf);
    epRecvCountsOut[srcRank] = static_cast<int32_t>(count);
    if (tpRecvCountsOut != nullptr) {
        tpRecvCountsOut[srcRank] = static_cast<int32_t>(count);
    }
    if (count <= 0 || count > maxRoutesPerSrc) {
        if (tpPublishWindow != nullptr && tpPublishRank >= 0) {
            TileXREpStoreDispatchSlotHeader(tpPublishWindow + SlotOffset(srcRank, slotBytes), 0, srcRank, 0, 0,
                magic, tBuf);
        }
        return outRecord;
    }

    GM_ADDR payloadBase = sourceWindow + PayloadOffset(slotRank, slotBytes);
    GM_ADDR scaleBase = payloadBase + maxRoutesPerSrc * rowBytes;
    GM_ADDR assistBase = sourceWindow + AssistOffset(slotRank, slotBytes, payloadBytesPerSlot);
    for (int64_t item = 0; item < count; ++item) {
        CopyBytesGmToGm(expandXOutGM + outRecord * rowBytes, payloadBase + item * payloadRowBytes, tBuf, rowBytes);
        if (quantMode == kEpQuantModePerTokenDynamic && dynamicScalesOutGM != nullptr) {
            const float scale = LoadFloatFromGm(scaleBase + item * static_cast<int64_t>(sizeof(float)), tBuf);
            StoreFloatToGm(dynamicScalesOutGM + outRecord * static_cast<int64_t>(sizeof(float)), scale, tBuf);
        }
        const TileXREp::EpAssistTuple tuple = TileXREpLoadAssistTuple(
            assistBase + item * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), tBuf);
        localAssistBase[outRecord].srcRank = tuple.srcRank;
        localAssistBase[outRecord].tokenId = TileXREpGetCombineTokenId(tuple);
        localAssistBase[outRecord].topKId = TileXREpGetCombineTopKId(tuple);
        localAssistBase[outRecord].expertId = tuple.expertId;
        const int64_t localExpert =
            TileXREpRouteToLocalExpert(tuple.expertId, localExpertNum, sharedExpertNum, sharedExpertRankNum);
        if (localExpert >= 0 && localExpert < localExpertNum) {
            TileXREpIncrementExpertTokenNum(expertTokenNumsOutGM, localExpert, expertTokenNumsType);
        }
        if (tpPublishWindow != nullptr && tpPublishRank >= 0) {
            CopyBytesGmToGm(tpPublishWindow + PayloadOffset(srcRank, slotBytes) + item * payloadRowBytes,
                payloadBase + item * payloadRowBytes, tBuf, payloadRowBytes);
            if (quantMode == kEpQuantModePerTokenDynamic) {
                CopyBytesGmToGm(tpPublishWindow + PayloadOffset(srcRank, slotBytes) + maxRoutesPerSrc * rowBytes +
                    item * static_cast<int64_t>(sizeof(float)), scaleBase + item * static_cast<int64_t>(sizeof(float)),
                    tBuf, static_cast<int64_t>(sizeof(float)));
            }
            TileXREpStoreAssistTuple(tpPublishWindow + AssistOffset(srcRank, slotBytes, payloadBytesPerSlot),
                item, tuple.srcRank, tuple.tokenId, tuple.topKId, tuple.expertId, tBuf);
        }
        ++outRecord;
    }
    if (tpPublishWindow != nullptr && tpPublishRank >= 0) {
        TileXREpStoreDispatchSlotHeader(tpPublishWindow + SlotOffset(srcRank, slotBytes),
            static_cast<int32_t>(count), srcRank,
            count * payloadRowBytes + (quantMode == kEpQuantModePerTokenDynamic ?
                maxRoutesPerSrc * static_cast<int64_t>(sizeof(float)) : 0),
            count * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)), magic, tBuf);
    }
    return outRecord;
}
