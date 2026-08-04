/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_A5_H
#define TILEXR_SDMA_A5_H

#include "kernel_operator.h"
#include "tilexr_sdma_a5_types.h"

namespace TileXR {
namespace detail {

__aicore__ inline void A5SdmaCleanCacheLine(__gm__ uint8_t* address)
{
    AscendC::GlobalTensor<uint8_t> line;
    line.SetGlobalBuffer(address);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(line);
    __asm__ __volatile__("");
}

__aicore__ inline void A5SdmaZeroSqe(__gm__ A5SdmaSqe* sqe)
{
    __gm__ uint32_t* words = reinterpret_cast<__gm__ uint32_t*>(sqe);
    for (uint32_t index = 0U; index < TILEXR_SDMA_A5_SQE_BYTES / sizeof(uint32_t); ++index) {
        words[index] = 0U;
    }
}

__aicore__ inline void A5SdmaBuildSqe(__gm__ A5SdmaSqe* sqe,
                                      uint32_t streamId,
                                      uint32_t taskId,
                                      uint64_t source,
                                      uint64_t destination,
                                      uint32_t bytes)
{
    A5SdmaZeroSqe(sqe);
    sqe->typeLockUnlock = static_cast<uint8_t>(TILEXR_SDMA_A5_SQE_TYPE);
    sqe->rtStreamId = static_cast<uint16_t>(streamId);
    sqe->taskId = static_cast<uint16_t>(taskId);
    sqe->kernelCredit = static_cast<uint8_t>(TILEXR_SDMA_A5_KERNEL_CREDIT);
    sqe->transferFlags = (1U << 8U) | (1U << 9U) | (1U << 10U) | (1U << 11U);
    sqe->mpamFlags = static_cast<uint8_t>(TILEXR_SDMA_A5_QOS << 3U);
    sqe->srcAddressLow = static_cast<uint32_t>(source & 0xFFFFFFFFULL);
    sqe->srcAddressHigh = static_cast<uint32_t>(source >> 32U);
    sqe->dstAddressLow = static_cast<uint32_t>(destination & 0xFFFFFFFFULL);
    sqe->dstAddressHigh = static_cast<uint32_t>(destination >> 32U);
    sqe->length = bytes;
}

__aicore__ inline void A5SdmaRingDoorbell(uint64_t address, uint32_t tail)
{
    __ubuf__ uint32_t value[8];
    value[0] = tail;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(reinterpret_cast<__gm__ uint32_t*>(address), value,
                             0, 1, sizeof(uint32_t), 0, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
}

__aicore__ inline uint32_t A5SdmaReadCompletion(__gm__ A5SdmaCompletionLine* completion)
{
    return AscendC::ReadGmByPassDCache<uint32_t>(&completion->generation);
}

__aicore__ inline bool A5SdmaWorkspaceValid(const __gm__ A5SdmaWorkspace* workspace)
{
    return workspace != nullptr &&
        workspace->header.magic == TILEXR_SDMA_A5_WORKSPACE_MAGIC &&
        workspace->header.abiVersion == TILEXR_SDMA_A5_ABI_VERSION &&
        workspace->header.backendKind == TILEXR_SDMA_A5_BACKEND_KIND &&
        workspace->header.channelCount == TILEXR_SDMA_A5_CHANNEL_COUNT &&
        workspace->header.sqeSize == TILEXR_SDMA_A5_SQE_BYTES &&
        workspace->header.channelStride == sizeof(A5SdmaChannel) &&
        workspace->header.workspaceSize == sizeof(A5SdmaWorkspace);
}

__aicore__ inline uint64_t A5SdmaCopyStridedNbi(
    __gm__ uint8_t* workspaceAddress,
    __gm__ uint8_t* destination,
    __gm__ uint8_t* source,
    uint64_t bytes,
    uint32_t copyCount,
    uint64_t destinationStrideBytes,
    uint64_t sourceStrideBytes,
    uint32_t channelIndex)
{
    __gm__ A5SdmaWorkspace* workspace =
        reinterpret_cast<__gm__ A5SdmaWorkspace*>(workspaceAddress);
    if (!A5SdmaWorkspaceValid(workspace) || destination == nullptr || source == nullptr ||
        !A5SdmaTransferLengthValid(bytes) || !A5SdmaChannelValid(channelIndex) ||
        copyCount == 0U || copyCount == 0xFFFFFFFFU) {
        return 0ULL;
    }
    const uint64_t destinationAddress = reinterpret_cast<uint64_t>(destination);
    const uint64_t sourceAddress = reinterpret_cast<uint64_t>(source);
    if (copyCount > 1U &&
        (!A5SdmaStridedRangeValid(
             destinationAddress, bytes, copyCount, destinationStrideBytes) ||
         !A5SdmaStridedRangeValid(
             sourceAddress, bytes, copyCount, sourceStrideBytes))) {
        return 0ULL;
    }

    __gm__ A5SdmaChannel* channel = &workspace->channels[channelIndex];
    const uint32_t requiredEntries = copyCount + 1U;
    if (channel->sqBase == 0U || channel->rtsqAddress == 0U ||
        channel->completionPayloadAddress == 0U || channel->completionRecordAddress == 0U ||
        channel->rtsqLength < sizeof(uint32_t) ||
        !A5SdmaQueueHasEntriesCapacity(
            channel->head, channel->tail, channel->depth, requiredEntries) ||
        channel->streamId > 0xFFFFU) {
        return 0ULL;
    }
    if (AscendC::AtomicCas<uint32_t>(&channel->outstanding, 0U, 1U) != 0U) {
        return 0ULL;
    }

    const uint32_t generation = A5SdmaNextGeneration(channel->generation);
    const uint32_t firstIndex = channel->tail;
    __gm__ A5SdmaCompletionLine* payload = reinterpret_cast<__gm__ A5SdmaCompletionLine*>(
        channel->completionPayloadAddress);
    payload->generation = generation;

    __gm__ A5SdmaSqe* sqBase = reinterpret_cast<__gm__ A5SdmaSqe*>(channel->sqBase);
    for (uint32_t copy = 0U; copy < copyCount; ++copy) {
        const uint32_t sqeIndex = A5SdmaAdvanceTailBy(firstIndex, copy, channel->depth);
        const uint64_t sourceOffset = static_cast<uint64_t>(copy) * sourceStrideBytes;
        const uint64_t destinationOffset =
            static_cast<uint64_t>(copy) * destinationStrideBytes;
        A5SdmaBuildSqe(
            sqBase + sqeIndex, channel->streamId,
            A5SdmaAdvanceTaskIdBy(channel->taskId, copy),
            sourceAddress + sourceOffset, destinationAddress + destinationOffset,
            static_cast<uint32_t>(bytes));
    }
    const uint32_t completionIndex =
        A5SdmaAdvanceTailBy(firstIndex, copyCount, channel->depth);
    A5SdmaBuildSqe(
        sqBase + completionIndex, channel->streamId,
        A5SdmaAdvanceTaskIdBy(channel->taskId, copyCount),
        channel->completionPayloadAddress,
        channel->completionRecordAddress,
        TILEXR_SDMA_A5_COMPLETION_BYTES);
    const uint32_t newTail =
        A5SdmaAdvanceTailBy(firstIndex, requiredEntries, channel->depth);

    channel->generation = generation;
    channel->tail = newTail;
    channel->taskId = A5SdmaAdvanceTaskIdBy(channel->taskId, requiredEntries);
    pipe_barrier(PIPE_ALL);
    A5SdmaCleanCacheLine(reinterpret_cast<__gm__ uint8_t*>(payload));
    for (uint32_t entry = 0U; entry < requiredEntries; ++entry) {
        const uint32_t sqeIndex = A5SdmaAdvanceTailBy(firstIndex, entry, channel->depth);
        A5SdmaCleanCacheLine(reinterpret_cast<__gm__ uint8_t*>(sqBase + sqeIndex));
    }
    A5SdmaCleanCacheLine(reinterpret_cast<__gm__ uint8_t*>(channel));
    A5SdmaCleanCacheLine(reinterpret_cast<__gm__ uint8_t*>(channel) + 64U);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    A5SdmaRingDoorbell(channel->rtsqAddress, newTail);
    return A5SdmaEncodeEvent(channelIndex, generation);
}

__aicore__ inline uint64_t A5SdmaCopyNbi(__gm__ uint8_t* workspaceAddress,
                                         __gm__ uint8_t* destination,
                                         __gm__ uint8_t* source,
                                         uint64_t bytes,
                                         uint32_t channelIndex)
{
    return A5SdmaCopyStridedNbi(
        workspaceAddress, destination, source, bytes, 1U, 0U, 0U, channelIndex);
}

__aicore__ inline bool A5SdmaWaitEvent(__gm__ uint8_t* workspaceAddress,
                                       uint64_t event,
                                       uint32_t expectedChannel)
{
    uint32_t channelIndex = 0U;
    uint32_t generation = 0U;
    if (!A5SdmaDecodeEvent(event, channelIndex, generation) || channelIndex != expectedChannel) {
        return false;
    }
    __gm__ A5SdmaWorkspace* workspace =
        reinterpret_cast<__gm__ A5SdmaWorkspace*>(workspaceAddress);
    if (!A5SdmaWorkspaceValid(workspace)) {
        return false;
    }
    __gm__ A5SdmaChannel* channel = &workspace->channels[channelIndex];
    if (channel->generation != generation ||
        AscendC::AtomicCas<uint32_t>(&channel->outstanding, 1U, 1U) != 1U ||
        channel->completionRecordAddress == 0U) {
        return false;
    }
    __gm__ A5SdmaCompletionLine* completion = reinterpret_cast<__gm__ A5SdmaCompletionLine*>(
        channel->completionRecordAddress);
    while (A5SdmaReadCompletion(completion) != generation) {
    }
    dsb(DSB_DDR);
    if (channel->generation != generation) {
        return false;
    }
    channel->head = channel->tail;
    A5SdmaCleanCacheLine(reinterpret_cast<__gm__ uint8_t*>(channel));
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (channel->generation != generation ||
        AscendC::AtomicCas<uint32_t>(&channel->outstanding, 1U, 0U) != 1U) {
        return false;
    }
    return true;
}

} // namespace detail
} // namespace TileXR

#endif // TILEXR_SDMA_A5_H
