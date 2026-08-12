/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_udma.h"

namespace {

constexpr uint64_t kProbeMagic = UINT64_C(0x5458505250524f42); // TXPRPROB
constexpr uint32_t kModeProfileTransfer = 0U;
constexpr uint32_t kModeProfileConsume = 1U;
constexpr uint32_t kModeLegacyGet = 2U;
constexpr uint32_t kConsumeTileBytes = 48U * 1024U;

__aicore__ inline void StoreStatus(__gm__ uint64_t* status, uint32_t postStatus,
    uint32_t frontier, uint32_t flushStatus, uint32_t quietStatus,
    uint32_t qpIdx, uint32_t localRegion, uint32_t remoteRegion,
    uint32_t batchCount, uint64_t batchBytes, uint32_t mode)
{
    if (status == nullptr) {
        return;
    }
    status[0] = kProbeMagic;
    status[1] = postStatus;
    status[2] = frontier;
    status[3] = flushStatus;
    status[4] = quietStatus;
    status[5] = qpIdx;
    status[6] = localRegion;
    status[7] = remoteRegion;
    status[8] = batchCount;
    status[9] = batchBytes;
    status[10] = mode;
}

__aicore__ inline uint32_t ConsumeStaging(
    AscendC::TPipe& pipe, const __gm__ TileXR::CommArgs* args,
    const __gm__ TileXR::TileXRUDMAProfileRegistry* registry,
    uint32_t localRegion, __gm__ uint8_t* consumer, uint64_t bytes)
{
    if (!TileXR::UDMAProfileRegisteredRangeValid(
            registry, args->rank, localRegion, 0U, bytes) ||
        consumer == nullptr || bytes == 0U || (bytes % sizeof(uint32_t)) != 0U) {
        return TileXR::TILEXR_UDMA_STATUS_INVALID;
    }

    __gm__ uint8_t* staging = TileXR::UDMAProfileRegisteredAddr(
        registry, args->rank, localRegion, 0U);
    if (staging == nullptr) {
        return TileXR::TILEXR_UDMA_STATUS_INVALID;
    }

    AscendC::TBuf<AscendC::QuePosition::VECCALC> copyBuf;
    pipe.InitBuffer(copyBuf, kConsumeTileBytes);
    AscendC::LocalTensor<uint32_t> local = copyBuf.Get<uint32_t>();
    AscendC::GlobalTensor<uint32_t> source;
    AscendC::GlobalTensor<uint32_t> destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(staging), bytes / sizeof(uint32_t));
    destination.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t*>(consumer), bytes / sizeof(uint32_t));

    const uint32_t tileElements = kConsumeTileBytes / sizeof(uint32_t);
    const uint64_t totalElements = bytes / sizeof(uint32_t);
    for (uint64_t offset = 0U; offset < totalElements; offset += tileElements) {
        const uint64_t remaining = totalElements - offset;
        const uint32_t elements = remaining < tileElements ?
            static_cast<uint32_t>(remaining) : tileElements;
        AscendC::DataCopy(local, source[offset], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopy(destination[offset], local, elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    return TileXR::TILEXR_UDMA_STATUS_SUCCESS;
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_udma_profile_probe_kernel(
    GM_ADDR commArgsGM, GM_ADDR profileInfoGM, GM_ADDR profileRegistryGM,
    GM_ADDR legacyGM, GM_ADDR consumerGM, GM_ADDR statusGM,
    int32_t peer, uint32_t qpIdx, uint32_t localRegion, uint32_t remoteRegion,
    uint64_t transferBytes, uint32_t batchCount, uint32_t mode)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto profileInfo = reinterpret_cast<__gm__ TileXR::UDMAInfo*>(profileInfoGM);
    auto profileRegistry =
        reinterpret_cast<__gm__ TileXR::TileXRUDMAProfileRegistry*>(profileRegistryGM);
    auto legacy = reinterpret_cast<__gm__ uint8_t*>(legacyGM);
    auto consumer = reinterpret_cast<__gm__ uint8_t*>(consumerGM);
    auto status = reinterpret_cast<__gm__ uint64_t*>(statusGM);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wqeBuf;
    pipe.InitBuffer(wqeBuf, TileXR::TILEXR_UDMA_WQE_SCRATCH_BYTES);
    auto wqeScratch = wqeBuf.Get<uint8_t>();

    uint32_t postStatus = TileXR::TILEXR_UDMA_STATUS_INVALID;
    uint32_t frontier = 0U;
    uint32_t flushStatus = TileXR::TILEXR_UDMA_STATUS_INVALID;
    uint32_t quietStatus = TileXR::TILEXR_UDMA_STATUS_INVALID;
    uint64_t batchBytes = 0U;

    if (args == nullptr || peer < 0 || peer >= args->rankSize ||
        peer == args->rank || transferBytes == 0U) {
        StoreStatus(status, postStatus, frontier, flushStatus, quietStatus,
            qpIdx, localRegion, remoteRegion, batchCount, batchBytes, mode);
        return;
    }

    if (mode == kModeLegacyGet) {
        if (legacy != nullptr && transferBytes <= UINT32_MAX) {
            postStatus = TileXR::UDMAGetNbiOnQp<uint8_t>(args, wqeScratch,
                peer, 0U, legacy, 0U, static_cast<uint32_t>(transferBytes));
            if (postStatus == TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                quietStatus = TileXR::UDMAQuietStatusOnQp(args, peer, 0U);
            }
        }
        StoreStatus(status, postStatus, frontier, flushStatus, quietStatus,
            0U, 0U, 0U, 1U, transferBytes, mode);
        return;
    }

    if (batchCount != 0U) {
        postStatus = TileXR::TILEXR_UDMA_STATUS_SUCCESS;
        uint64_t offset = 0U;
        for (uint32_t batch = 0U; batch < batchCount; ++batch) {
            const uint32_t ret = TileXR::UDMAProfileGetNbiOnQpDeferred(
                args, profileInfo, profileRegistry, wqeScratch, peer, qpIdx,
                localRegion, offset, remoteRegion, offset, transferBytes);
            if (ret != TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                postStatus = ret;
                break;
            }
            if (transferBytes > UINT64_MAX - offset) {
                postStatus = TileXR::TILEXR_UDMA_STATUS_INVALID;
                break;
            }
            offset += transferBytes;
        }
        batchBytes = offset;
    }

    if (postStatus == TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
        frontier = TileXR::UDMAProfileCompletionFrontier(
            args, profileInfo, profileRegistry, peer, qpIdx);
        flushStatus = TileXR::UDMAProfileFlushQpDoorbell(
            args, profileInfo, profileRegistry, peer, qpIdx);
    }
    if (flushStatus == TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
        quietStatus = TileXR::UDMAProfileQuietStatusOnQpUntil(
            args, profileInfo, profileRegistry, peer, qpIdx, frontier);
    }
    if (quietStatus == TileXR::TILEXR_UDMA_STATUS_SUCCESS &&
        mode == kModeProfileConsume) {
        quietStatus = ConsumeStaging(
            pipe, args, profileRegistry, localRegion, consumer, batchBytes);
    }

    StoreStatus(status, postStatus, frontier, flushStatus, quietStatus,
        qpIdx, localRegion, remoteRegion, batchCount, batchBytes, mode);
}
