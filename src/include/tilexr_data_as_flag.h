/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_DATA_AS_FLAG_H
#define TILEXR_DATA_AS_FLAG_H

#include <cstdint>

#include "comm_args.h"

#if TILEXR_ASCENDC_AICORE_COMPILE
#include "adv_api/reduce/sum.h"
#define TILEXR_DATA_AS_FLAG_INLINE __aicore__ inline
#else
#define TILEXR_DATA_AS_FLAG_INLINE inline
#endif

namespace TileXR {

constexpr uint32_t DATA_AS_FLAG_BLOCK_BYTES = 512;
constexpr uint32_t DATA_AS_FLAG_PAYLOAD_BYTES = 480;
constexpr uint32_t DATA_AS_FLAG_FLAG_BYTES = 32;
constexpr uint32_t DATA_AS_FLAG_FLAG_OFFSET_BYTES = 480;
constexpr uint32_t DATA_AS_FLAG_ALIGN_BYTES = 32;
constexpr uint32_t DATA_AS_FLAG_FLOAT_BYTES = sizeof(float);
constexpr uint32_t DATA_AS_FLAG_FLAG_FLOATS = DATA_AS_FLAG_FLAG_BYTES / DATA_AS_FLAG_FLOAT_BYTES;
constexpr uint32_t DATA_AS_FLAG_SUM_RESULT_BYTES = DATA_AS_FLAG_ALIGN_BYTES;
constexpr uint64_t DATA_AS_FLAG_COMMIT_BIT = 0x80000000ULL;
constexpr float DATA_AS_FLAG_READY_VALUE = 1.0f;

static_assert(DATA_AS_FLAG_PAYLOAD_BYTES + DATA_AS_FLAG_FLAG_BYTES == DATA_AS_FLAG_BLOCK_BYTES,
    "DataAsFlag block layout must be 480B payload plus 32B flag");
static_assert(DATA_AS_FLAG_FLAG_OFFSET_BYTES == DATA_AS_FLAG_PAYLOAD_BYTES,
    "DataAsFlag flag must start immediately after payload");

TILEXR_DATA_AS_FLAG_INLINE uint32_t DataAsFlagBlockCountForPayloadBytes(uint64_t dataBytes)
{
    if (dataBytes == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>(
        (dataBytes + DATA_AS_FLAG_PAYLOAD_BYTES - 1U) / DATA_AS_FLAG_PAYLOAD_BYTES);
}

TILEXR_DATA_AS_FLAG_INLINE uint64_t DataAsFlagAlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0U) {
        return value;
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0U ? value : value + alignment - remainder;
}

TILEXR_DATA_AS_FLAG_INLINE uint64_t DataAsFlagEpoch(int32_t magic, int32_t step)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(magic)) << 32) |
        static_cast<uint32_t>(step);
}

TILEXR_DATA_AS_FLAG_INLINE bool DataAsFlagEpochReady(uint64_t observed, uint64_t expected)
{
    constexpr uint64_t kMagicMask = 0xffffffff00000000ULL;
    return (observed & kMagicMask) == (expected & kMagicMask) && observed >= expected;
}

TILEXR_DATA_AS_FLAG_INLINE uint64_t DataAsFlagCommitEpoch(uint64_t epoch)
{
    return epoch | DATA_AS_FLAG_COMMIT_BIT;
}

#if TILEXR_ASCENDC_AICORE_COMPILE

__aicore__ inline uint32_t DataAsFlagScratchBytes(const AscendC::LocalTensor<uint8_t>& scratch)
{
    return static_cast<uint32_t>(scratch.GetSize());
}

__aicore__ inline uint32_t DataAsFlagAlignedFloatCount(uint32_t valueCount)
{
    return static_cast<uint32_t>(
        DataAsFlagAlignUp(static_cast<uint64_t>(valueCount) * sizeof(float), DATA_AS_FLAG_ALIGN_BYTES) /
        sizeof(float));
}

__aicore__ inline uint32_t DataAsFlagSumWorkspaceBytes(uint32_t valueCount)
{
    if (valueCount == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>(
        DataAsFlagAlignUp(static_cast<uint64_t>(DataAsFlagAlignedFloatCount(valueCount)) * sizeof(float),
            DATA_AS_FLAG_ALIGN_BYTES));
}

__aicore__ inline uint32_t DataAsFlagFlagBytes(uint32_t blockCount)
{
    return blockCount * DATA_AS_FLAG_FLAG_BYTES;
}

__aicore__ inline uint32_t DataAsFlagSumInputValueCount(uint32_t blockCount)
{
    return blockCount * DATA_AS_FLAG_FLAG_FLOATS;
}

__aicore__ inline uint32_t DataAsFlagSumBytes(uint32_t blockCount)
{
    return DATA_AS_FLAG_SUM_RESULT_BYTES +
        DataAsFlagSumWorkspaceBytes(DataAsFlagSumInputValueCount(blockCount));
}

__aicore__ inline bool DataAsFlagSplitCheckScratch(
    AscendC::LocalTensor<uint8_t>& scratch,
    uint32_t blockCount,
    AscendC::LocalTensor<float>& flagLocal,
    AscendC::LocalTensor<float>& sumOut,
    AscendC::LocalTensor<uint8_t>& sharedTmpBuffer)
{
    const uint32_t scratchBytes = DataAsFlagScratchBytes(scratch);
    const uint32_t flagBytes = DataAsFlagFlagBytes(blockCount);
    const uint32_t sumBytes = DataAsFlagSumBytes(blockCount);
    if (blockCount == 0U || scratchBytes < flagBytes + sumBytes) {
        return false;
    }

    flagLocal = scratch.template ReinterpretCast<float>();
    sumOut = scratch[flagBytes].template ReinterpretCast<float>();
    sharedTmpBuffer = scratch[flagBytes + DATA_AS_FLAG_SUM_RESULT_BYTES];
    return true;
}

__aicore__ inline uint32_t DataAsFlagMaxCheckBlocks(uint32_t scratchBytes)
{
    uint32_t blocks = scratchBytes / DATA_AS_FLAG_FLAG_BYTES;
    while (blocks > 0U) {
        const uint32_t requiredBytes = DataAsFlagFlagBytes(blocks) + DataAsFlagSumBytes(blocks);
        if (scratchBytes >= requiredBytes) {
            return blocks;
        }
        --blocks;
    }
    return 0U;
}

__aicore__ inline uint32_t DataAsFlagMaxRecvBlocks(uint32_t scratchBytes)
{
    uint32_t blocks = scratchBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
    while (blocks > 0U) {
        const uint64_t requiredBytes =
            static_cast<uint64_t>(DataAsFlagFlagBytes(blocks)) +
            static_cast<uint64_t>(DataAsFlagSumBytes(blocks)) +
            static_cast<uint64_t>(blocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        if (static_cast<uint64_t>(scratchBytes) >= requiredBytes) {
            return blocks;
        }
        --blocks;
    }
    return 0U;
}

__aicore__ inline uint32_t DataAsFlagMaxEpochOrderedSendBlocks(uint32_t scratchBytes)
{
    return scratchBytes / DATA_AS_FLAG_BLOCK_BYTES;
}

__aicore__ inline uint32_t DataAsFlagMaxEpochOrderedRecvBlocks(uint32_t scratchBytes)
{
    return DataAsFlagMaxEpochOrderedSendBlocks(scratchBytes);
}

__aicore__ inline uint32_t DataAsFlagInit(AscendC::LocalTensor<uint8_t>& sendScratch)
{
    const uint32_t sendBlocks = DataAsFlagScratchBytes(sendScratch) / DATA_AS_FLAG_BLOCK_BYTES;
    if (sendBlocks == 0U) {
        return 0U;
    }

    AscendC::LocalTensor<float> sendFloat = sendScratch.template ReinterpretCast<float>();
    AscendC::Duplicate<float>(
        sendFloat,
        DATA_AS_FLAG_READY_VALUE,
        sendBlocks * DATA_AS_FLAG_BLOCK_BYTES / sizeof(float));
    AscendC::PipeBarrier<PIPE_ALL>();
    return sendBlocks;
}

__aicore__ inline void DataAsFlagCopyPayloadToScratch(
    AscendC::LocalTensor<uint8_t>& sendScratch,
    const __gm__ uint8_t* srcGM,
    uint64_t srcOffset,
    uint32_t fullBlocks,
    uint32_t tailBytes)
{
    AscendC::GlobalTensor<uint8_t> srcGlobal;
    srcGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t*>(srcGM + srcOffset));
    AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0U, 0U, 0U};

    if (fullBlocks > 0U) {
        AscendC::DataCopyExtParams fullParams {
            static_cast<uint16_t>(fullBlocks),
            DATA_AS_FLAG_PAYLOAD_BYTES,
            0U,
            DATA_AS_FLAG_FLAG_BYTES / DATA_AS_FLAG_ALIGN_BYTES,
            0U};
        AscendC::DataCopyPad(sendScratch, srcGlobal, fullParams, padParams);
    }
    if (tailBytes > 0U) {
        AscendC::DataCopyExtParams tailParams {1U, tailBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(
            sendScratch[static_cast<uint64_t>(fullBlocks) * DATA_AS_FLAG_BLOCK_BYTES],
            srcGlobal[static_cast<uint64_t>(fullBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES],
            tailParams,
            padParams);
    }
}

__aicore__ inline void DataAsFlagCopyScratchToDataAsFlagGM(
    __gm__ uint8_t* dstDataAsFlagGM,
    uint32_t dstBlockOffset,
    AscendC::LocalTensor<uint8_t>& sendScratch,
    uint32_t batchBlocks)
{
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(
        dstDataAsFlagGM + static_cast<uint64_t>(dstBlockOffset) * DATA_AS_FLAG_BLOCK_BYTES);
    AscendC::DataCopyExtParams outParams {
        1U,
        batchBlocks * DATA_AS_FLAG_BLOCK_BYTES,
        0U,
        0U,
        0U};
    AscendC::DataCopyPad(dstGlobal, sendScratch, outParams);
}

__aicore__ inline uint32_t DataAsFlagSend(
    __gm__ uint8_t* dstDataAsFlagGM,
    const __gm__ uint8_t* srcGM,
    uint64_t dataBytes,
    AscendC::LocalTensor<uint8_t>& sendScratch)
{
    if (dstDataAsFlagGM == nullptr || srcGM == nullptr || dataBytes == 0U) {
        return 0U;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t sendBlockCapacity = DataAsFlagScratchBytes(sendScratch) / DATA_AS_FLAG_BLOCK_BYTES;
    if (sendBlockCapacity == 0U) {
        return 0U;
    }

    uint32_t sentBlocks = 0U;
    uint64_t sentBytes = 0U;
    while (sentBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - sentBlocks;
        const uint32_t batchBlocks =
            remainingBlocks < sendBlockCapacity ? remainingBlocks : sendBlockCapacity;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint64_t remainingBytes = dataBytes - sentBytes;
        const uint32_t batchPayloadBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        const uint32_t fullBlocks = batchPayloadBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t tailBytes = batchPayloadBytes % DATA_AS_FLAG_PAYLOAD_BYTES;

        DataAsFlagCopyPayloadToScratch(sendScratch, srcGM, sentBytes, fullBlocks, tailBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataAsFlagCopyScratchToDataAsFlagGM(dstDataAsFlagGM, sentBlocks, sendScratch, batchBlocks);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        sentBlocks += batchBlocks;
        sentBytes += batchPayloadBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return totalBlocks;
}

__aicore__ inline void DataAsFlagFillEpochFlags(
    AscendC::LocalTensor<uint8_t>& flagScratch,
    uint32_t blockCount,
    uint64_t epoch)
{
    AscendC::LocalTensor<int64_t> flagWords = flagScratch.template ReinterpretCast<int64_t>();
    const uint32_t words = blockCount * DATA_AS_FLAG_FLAG_BYTES / sizeof(int64_t);
    const int64_t signedEpoch = static_cast<int64_t>(epoch);
    for (uint32_t i = 0; i < words; ++i) {
        flagWords.SetValue(i, signedEpoch);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
}

__aicore__ inline void DataAsFlagWriteBatchCommitFlag(
    __gm__ uint8_t* dstDataAsFlagGM,
    uint32_t dstBlockOffset,
    AscendC::LocalTensor<uint8_t>& scratch,
    uint32_t batchBlocks,
    uint64_t commitEpoch)
{
    if (batchBlocks == 0U) {
        return;
    }
    DataAsFlagFillEpochFlags(scratch, 1U, commitEpoch);
    AscendC::GlobalTensor<int64_t> flagGlobal;
    const uint32_t lastBlock = dstBlockOffset + batchBlocks - 1U;
    flagGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(
            dstDataAsFlagGM + static_cast<uint64_t>(lastBlock) * DATA_AS_FLAG_BLOCK_BYTES +
            DATA_AS_FLAG_FLAG_OFFSET_BYTES));
    AscendC::LocalTensor<int64_t> flagWords = scratch.template ReinterpretCast<int64_t>();
    constexpr uint32_t kFlagWords = DATA_AS_FLAG_FLAG_BYTES / sizeof(int64_t);
    AscendC::DataCopy(flagGlobal, flagWords, kFlagWords);
}

__aicore__ inline uint32_t DataAsFlagSendEpochOrdered(
    __gm__ uint8_t* dstDataAsFlagGM,
    const __gm__ uint8_t* srcGM,
    uint64_t dataBytes,
    uint64_t epoch,
    AscendC::LocalTensor<uint8_t>& scratch)
{
    if (dstDataAsFlagGM == nullptr || srcGM == nullptr || dataBytes == 0U) {
        return 0U;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxEpochOrderedSendBlocks(DataAsFlagScratchBytes(scratch));
    if (batchCapacity == 0U) {
        return 0U;
    }

    uint32_t sentBlocks = 0U;
    uint64_t sentBytes = 0U;
    while (sentBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - sentBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint64_t remainingBytes = dataBytes - sentBytes;
        const uint32_t batchPayloadBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        const uint32_t fullBlocks = batchPayloadBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t tailBytes = batchPayloadBytes % DATA_AS_FLAG_PAYLOAD_BYTES;

        AscendC::LocalTensor<uint8_t> payloadScratch = scratch;

        AscendC::Duplicate<uint8_t>(payloadScratch, 0U, batchBlocks * DATA_AS_FLAG_BLOCK_BYTES);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataAsFlagCopyPayloadToScratch(payloadScratch, srcGM, sentBytes, fullBlocks, tailBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataAsFlagCopyScratchToDataAsFlagGM(dstDataAsFlagGM, sentBlocks, payloadScratch, batchBlocks);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

        DataAsFlagWriteBatchCommitFlag(dstDataAsFlagGM, sentBlocks, scratch, batchBlocks, DataAsFlagCommitEpoch(epoch));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

        sentBlocks += batchBlocks;
        sentBytes += batchPayloadBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return totalBlocks;
}

__aicore__ inline bool DataAsFlagCheckBatch(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockOffset,
    uint32_t batchBlocks,
    AscendC::LocalTensor<uint8_t>& recvScratch)
{
    AscendC::LocalTensor<float> flagLocal;
    AscendC::LocalTensor<float> sumOut;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
    if (!DataAsFlagSplitCheckScratch(recvScratch, batchBlocks, flagLocal, sumOut, sharedTmpBuffer)) {
        return false;
    }

    AscendC::GlobalTensor<float> flagGlobal;
    flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
        const_cast<__gm__ uint8_t*>(
            dataAsFlagGM + static_cast<uint64_t>(blockOffset) * DATA_AS_FLAG_BLOCK_BYTES +
            DATA_AS_FLAG_FLAG_OFFSET_BYTES)));
    AscendC::DataCopyExtParams flagParams {
        static_cast<uint16_t>(batchBlocks),
        sizeof(float),
        DATA_AS_FLAG_BLOCK_BYTES - sizeof(float),
        0U,
        0U};
    AscendC::DataCopyPadExtParams<float> flagPadParams {
        true,
        0U,
        static_cast<uint8_t>(DATA_AS_FLAG_FLAG_FLOATS - 1U),
        0.0f};
    AscendC::DataCopyPad(flagLocal, flagGlobal, flagParams, flagPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    const uint32_t valueCount = DataAsFlagSumInputValueCount(batchBlocks);
    const uint32_t inner = DataAsFlagAlignedFloatCount(valueCount);
    AscendC::SumParams sumParams {1U, inner, valueCount};
    AscendC::Sum<float>(sumOut, flagLocal, sharedTmpBuffer, sumParams);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return sumOut.GetValue(0) == static_cast<float>(batchBlocks);
}

__aicore__ inline uint64_t DataAsFlagLoadEpochFlag(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockIndex,
    AscendC::LocalTensor<uint8_t>& scratch)
{
    AscendC::GlobalTensor<int64_t> flagGlobal;
    flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(
        const_cast<__gm__ uint8_t*>(
            dataAsFlagGM + static_cast<uint64_t>(blockIndex) * DATA_AS_FLAG_BLOCK_BYTES +
            DATA_AS_FLAG_FLAG_OFFSET_BYTES)));
    AscendC::LocalTensor<int64_t> flagLocal = scratch.template ReinterpretCast<int64_t>();
    AscendC::DataCopy(flagLocal, flagGlobal, DATA_AS_FLAG_FLAG_BYTES / sizeof(int64_t));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return static_cast<uint64_t>(flagLocal.GetValue(0));
}

__aicore__ inline bool DataAsFlagCheckBatchEpochStrict(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockOffset,
    uint32_t batchBlocks,
    uint64_t commitEpoch,
    AscendC::LocalTensor<uint8_t>& scratch)
{
    if (batchBlocks == 0U) {
        return false;
    }
    const uint32_t lastBlock = blockOffset + batchBlocks - 1U;
    if (!DataAsFlagEpochReady(DataAsFlagLoadEpochFlag(dataAsFlagGM, lastBlock, scratch), commitEpoch)) {
        return false;
    }
    return true;
}

__aicore__ inline bool DataAsFlagCheck(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    AscendC::LocalTensor<uint8_t>& recvScratch)
{
    if (dataAsFlagGM == nullptr) {
        return false;
    }
    if (dataBytes == 0U) {
        return true;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxCheckBlocks(DataAsFlagScratchBytes(recvScratch));
    if (batchCapacity == 0U) {
        return false;
    }

    uint32_t checkedBlocks = 0U;
    while (checkedBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - checkedBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        if (!DataAsFlagCheckBatch(dataAsFlagGM, checkedBlocks, batchBlocks, recvScratch)) {
            return false;
        }
        checkedBlocks += batchBlocks;
    }
    return true;
}

__aicore__ inline void DataAsFlagCopyBatchToRecvGM(
    const __gm__ uint8_t* dataAsFlagGM,
    uint32_t blockOffset,
    uint64_t recvOffset,
    uint32_t batchBytes,
    __gm__ uint8_t* recvGM,
    AscendC::LocalTensor<uint8_t>& copyLocal)
{
    const uint32_t fullBlocks = batchBytes / DATA_AS_FLAG_PAYLOAD_BYTES;
    const uint32_t tailBytes = batchBytes % DATA_AS_FLAG_PAYLOAD_BYTES;
    AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0U, 0U, 0U};

    AscendC::GlobalTensor<uint8_t> dataAsFlagGlobal;
    dataAsFlagGlobal.SetGlobalBuffer(const_cast<__gm__ uint8_t*>(
        dataAsFlagGM + static_cast<uint64_t>(blockOffset) * DATA_AS_FLAG_BLOCK_BYTES));
    if (fullBlocks > 0U) {
        AscendC::DataCopyExtParams payloadInParams {
            static_cast<uint16_t>(fullBlocks),
            DATA_AS_FLAG_PAYLOAD_BYTES,
            DATA_AS_FLAG_FLAG_BYTES,
            0U,
            0U};
        AscendC::DataCopyPad(copyLocal, dataAsFlagGlobal, payloadInParams, padParams);
    }
    if (tailBytes > 0U) {
        AscendC::DataCopyExtParams tailInParams {1U, tailBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(
            copyLocal[static_cast<uint64_t>(fullBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES],
            dataAsFlagGlobal[static_cast<uint64_t>(fullBlocks) * DATA_AS_FLAG_BLOCK_BYTES],
            tailInParams,
            padParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint8_t> recvGlobal;
    recvGlobal.SetGlobalBuffer(recvGM + recvOffset);
    AscendC::DataCopyExtParams payloadOutParams {1U, batchBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(recvGlobal, copyLocal, payloadOutParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline bool DataAsFlagCheckAndRecvEpochOrdered(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    __gm__ uint8_t* recvGM,
    uint64_t epoch,
    AscendC::LocalTensor<uint8_t>& recvScratch,
    bool strict)
{
    if (dataAsFlagGM == nullptr || recvGM == nullptr) {
        return false;
    }
    if (dataBytes == 0U) {
        return true;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxEpochOrderedRecvBlocks(DataAsFlagScratchBytes(recvScratch));
    if (batchCapacity == 0U) {
        return false;
    }

    uint32_t processedBlocks = 0U;
    uint64_t processedBytes = 0U;
    const uint64_t commitEpoch = DataAsFlagCommitEpoch(epoch);
    while (processedBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - processedBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        const uint32_t lastBlock = processedBlocks + batchBlocks - 1U;
        while (!DataAsFlagEpochReady(DataAsFlagLoadEpochFlag(dataAsFlagGM, lastBlock, recvScratch), commitEpoch)) {
        }
        if (strict &&
            !DataAsFlagCheckBatchEpochStrict(dataAsFlagGM, processedBlocks, batchBlocks, commitEpoch, recvScratch)) {
            return false;
        }

        const uint64_t remainingBytes = dataBytes - processedBytes;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t batchBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        DataAsFlagCopyBatchToRecvGM(
            dataAsFlagGM, processedBlocks, processedBytes, batchBytes, recvGM, recvScratch);
        processedBlocks += batchBlocks;
        processedBytes += batchBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
}

__aicore__ inline bool DataAsFlagCheckAndRecv(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    __gm__ uint8_t* recvGM,
    AscendC::LocalTensor<uint8_t>& recvScratch)
{
    if (dataAsFlagGM == nullptr || recvGM == nullptr) {
        return false;
    }
    if (dataBytes == 0U) {
        return true;
    }

    const uint32_t totalBlocks = DataAsFlagBlockCountForPayloadBytes(dataBytes);
    const uint32_t batchCapacity = DataAsFlagMaxRecvBlocks(DataAsFlagScratchBytes(recvScratch));
    if (batchCapacity == 0U) {
        return false;
    }

    uint32_t processedBlocks = 0U;
    uint64_t processedBytes = 0U;
    while (processedBlocks < totalBlocks) {
        const uint32_t remainingBlocks = totalBlocks - processedBlocks;
        const uint32_t batchBlocks = remainingBlocks < batchCapacity ? remainingBlocks : batchCapacity;
        while (!DataAsFlagCheckBatch(dataAsFlagGM, processedBlocks, batchBlocks, recvScratch)) {
        }

        const uint64_t remainingBytes = dataBytes - processedBytes;
        const uint64_t maxBatchBytes = static_cast<uint64_t>(batchBlocks) * DATA_AS_FLAG_PAYLOAD_BYTES;
        const uint32_t batchBytes = static_cast<uint32_t>(
            remainingBytes < maxBatchBytes ? remainingBytes : maxBatchBytes);
        const uint32_t flagBytes = DataAsFlagFlagBytes(batchBlocks);
        const uint32_t sumBytes = DataAsFlagSumBytes(batchBlocks);
        AscendC::LocalTensor<uint8_t> copyLocal = recvScratch[flagBytes + sumBytes];
        DataAsFlagCopyBatchToRecvGM(
            dataAsFlagGM, processedBlocks, processedBytes, batchBytes, recvGM, copyLocal);
        processedBlocks += batchBlocks;
        processedBytes += batchBytes;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return true;
}

#endif // TILEXR_ASCENDC_AICORE_COMPILE

} // namespace TileXR

#undef TILEXR_DATA_AS_FLAG_INLINE

#endif // TILEXR_DATA_AS_FLAG_H
