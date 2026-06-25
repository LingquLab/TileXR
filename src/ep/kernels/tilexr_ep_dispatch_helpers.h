#pragma once

__aicore__ inline bool TileXREpIsTokenActive(GM_ADDR xActiveMaskGM, int64_t token,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    if (xActiveMaskGM == nullptr) {
        return true;
    }

    AscendC::LocalTensor<uint8_t> local =
        tBuf.GetWithOffset<uint8_t>(kEpScalarUbBytes / sizeof(uint8_t), kEpScalarUbOffset);
    AscendC::GlobalTensor<uint8_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(xActiveMaskGM) + token, 1);

    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(sizeof(uint8_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return local.GetValue(0) != 0;
}

__aicore__ inline void TileXREpStoreDispatchSlotHeader(GM_ADDR slotGM, int32_t count, int32_t srcRank,
    int64_t payloadBytes, int64_t assistBytes, int64_t magic,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int64_t> local =
        tBuf.GetWithOffset<int64_t>(kEpScalarUbBytes / sizeof(int64_t), kEpScalarUbOffset);
    local.SetValue(0, PackInt32Pair(static_cast<uint32_t>(count), srcRank));
    local.SetValue(1, payloadBytes);
    local.SetValue(2, assistBytes);
    local.SetValue(3, magic);
    local.SetValue(4, 0);
    local.SetValue(5, 0);
    local.SetValue(6, 0);
    local.SetValue(7, 0);
    StoreInt64WordsToGm(slotGM, local, tBuf, TileXREp::kEpSrcSlotHeaderBytes);
}

__aicore__ inline void TileXREpStoreAssistTuple(GM_ADDR assistGM, int64_t index, int32_t srcRank, int32_t tokenId,
    int32_t topKId, int32_t expertId, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    AscendC::LocalTensor<int32_t> local =
        tBuf.GetWithOffset<int32_t>(kEpScalarUbBytes / sizeof(int32_t), kEpScalarUbOffset);
    local.SetValue(0, srcRank);
    local.SetValue(1, tokenId);
    local.SetValue(2, topKId);
    local.SetValue(3, expertId);

    AscendC::GlobalTensor<int32_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(assistGM) + index * TileXREp::kEpAssistTupleInts,
        TileXREp::kEpAssistTupleInts);
    AscendC::DataCopyExtParams copyParams {
        1, static_cast<uint32_t>(sizeof(TileXREp::EpAssistTuple)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(dst, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

}

__aicore__ inline int64_t TileXREpMoeRankCount(int32_t rankSize, int64_t sharedExpertRankNum)
{
    return static_cast<int64_t>(rankSize) - sharedExpertRankNum;
}

__aicore__ inline int64_t TileXREpLocalExpertCount(
    int64_t moeExpertNum, int32_t rankSize, int64_t sharedExpertRankNum)
{
    const int64_t moeRankNum = TileXREpMoeRankCount(rankSize, sharedExpertRankNum);
    if (moeExpertNum <= 0 || moeRankNum <= 0) {
        return -1;
    }
    return moeExpertNum / moeRankNum;
}

__aicore__ inline int64_t TileXREpRouteToDstRank(
    int32_t expertId, int64_t localExpertNum, int32_t rankSize, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum)
{
    if (expertId < 0 || localExpertNum <= 0 || sharedExpertNum < 0 || sharedExpertRankNum < 0) {
        return -1;
    }
    if (static_cast<int64_t>(expertId) < sharedExpertNum) {
        return expertId < sharedExpertRankNum ? expertId : -1;
    }
    const int64_t moeExpertId = static_cast<int64_t>(expertId) - sharedExpertNum;
    const int64_t dstRank = sharedExpertRankNum + moeExpertId / localExpertNum;
    return dstRank < rankSize ? dstRank : -1;
}

__aicore__ inline int64_t TileXREpRouteToLocalExpert(
    int32_t expertId, int64_t localExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum)
{
    if (expertId < 0 || localExpertNum <= 0 || sharedExpertNum < 0 || sharedExpertRankNum < 0) {
        return -1;
    }
    if (static_cast<int64_t>(expertId) < sharedExpertNum) {
        return expertId < sharedExpertRankNum ? expertId : -1;
    }
    return (static_cast<int64_t>(expertId) - sharedExpertNum) % localExpertNum;
}

__aicore__ inline void TileXREpCopyRoutePayload(GM_ADDR localWindow, int64_t dstRank, int64_t dstIndex,
    int64_t slotBytes, GM_ADDR xGM, int64_t token, int64_t inputRowBytes, int64_t rowBytes,
    int64_t payloadRowBytes,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    CopyBytesGmToGm(localWindow + PayloadOffset(dstRank, slotBytes) + dstIndex * payloadRowBytes,
        xGM + token * inputRowBytes, tBuf, rowBytes);
}

__aicore__ inline void TileXREpCopyStaticQuantRoutePayload(GM_ADDR localWindow, int64_t dstRank, int64_t dstIndex,
    int64_t slotBytes, GM_ADDR xGM, GM_ADDR scalesGM, int64_t token, int64_t h, int64_t inputRowBytes,
    int64_t payloadRowBytes, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    const float scale = LoadFloatFromGm(scalesGM, tBuf);
    GM_ADDR dstBase = localWindow + PayloadOffset(dstRank, slotBytes) + dstIndex * payloadRowBytes;
    GM_ADDR srcBase = xGM + token * inputRowBytes;
    AscendC::LocalTensor<int8_t> local =
        tBuf.GetWithOffset<int8_t>(kEpCopyTileBytes, kEpSyncUbBytes);
    for (int64_t elem = 0; elem < h; ++elem) {
        const uint16_t fp16Bits = LoadUint16FromGm(srcBase + elem * static_cast<int64_t>(sizeof(uint16_t)), tBuf);
        const float value = TileXREpHalfBitsToFloat(fp16Bits) * scale;
        local.SetValue(static_cast<int32_t>(elem), TileXREpClampInt8(value));
    }
    AscendC::GlobalTensor<int8_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(dstBase), h);
    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(h * sizeof(int8_t)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(dst, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline float TileXREpPerTokenDynamicScale(GM_ADDR xGM, int64_t token, int64_t h,
    int64_t inputRowBytes, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    GM_ADDR srcBase = xGM + token * inputRowBytes;
    float maxAbs = 0.0f;
    for (int64_t elem = 0; elem < h; ++elem) {
        const uint16_t fp16Bits = LoadUint16FromGm(srcBase + elem * static_cast<int64_t>(sizeof(uint16_t)), tBuf);
        const float absValue = TileXREpAbsFloat(TileXREpHalfBitsToFloat(fp16Bits));
        if (absValue > maxAbs) {
            maxAbs = absValue;
        }
    }
    return maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
}

__aicore__ inline void TileXREpCopyPerTokenDynamicQuantRoutePayload(GM_ADDR localWindow, int64_t dstRank,
    int64_t dstIndex, int64_t slotBytes, int64_t maxRoutesPerSrc, GM_ADDR xGM, int64_t token, int64_t h,
    int64_t inputRowBytes, int64_t rowBytes, AscendC::TBuf<AscendC::QuePosition::VECCALC> &tBuf)
{
    GM_ADDR dstBase = localWindow + PayloadOffset(dstRank, slotBytes) + dstIndex * rowBytes;
    GM_ADDR scaleBase = localWindow + PayloadOffset(dstRank, slotBytes) + maxRoutesPerSrc * rowBytes;
    GM_ADDR srcBase = xGM + token * inputRowBytes;
    float maxAbs = 0.0f;
    AscendC::LocalTensor<float> values =
        tBuf.GetWithOffset<float>(static_cast<uint32_t>(h), kEpSyncUbBytes);
    for (int64_t elem = 0; elem < h; ++elem) {
        const uint16_t fp16Bits = LoadUint16FromGm(srcBase + elem * static_cast<int64_t>(sizeof(uint16_t)), tBuf);
        const float value = TileXREpHalfBitsToFloat(fp16Bits);
        const float absValue = TileXREpAbsFloat(value);
        if (absValue > maxAbs) {
            maxAbs = absValue;
        }
        values.SetValue(static_cast<int32_t>(elem), value);
    }
    const float dequantScale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
    const float quantScale = dequantScale > 0.0f ? 1.0f / dequantScale : 1.0f;
    AscendC::LocalTensor<int8_t> local =
        tBuf.GetWithOffset<int8_t>(kEpCopyTileBytes - kEpSyncUbBytes, kEpSyncUbBytes * 2);
    for (int64_t elem = 0; elem < h; ++elem) {
        const float value = values.GetValue(static_cast<int32_t>(elem)) * quantScale;
        local.SetValue(static_cast<int32_t>(elem), TileXREpClampInt8(value));
    }

    AscendC::GlobalTensor<int8_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(dstBase), h);
    AscendC::DataCopyExtParams copyParams {1, static_cast<uint32_t>(h * sizeof(int8_t)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(dst, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

    StoreFloatToGm(scaleBase + dstIndex * static_cast<int64_t>(sizeof(float)), dequantScale, tBuf);
}
