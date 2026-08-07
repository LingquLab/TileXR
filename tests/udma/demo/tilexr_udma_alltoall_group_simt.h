/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_GROUP_SIMT_H
#define TILEXR_UDMA_ALLTOALL_GROUP_SIMT_H

#include "kernel_operator.h"
#include "ascendc/basic_api/interface/simt_api/common_functions.h"
#include "ascendc/basic_api/interface/simt_api/device_atomic_functions.h"
#include "ascendc/basic_api/interface/simt_api/device_sync_functions.h"
#include "tilexr_udma.h"

namespace TileXR {
namespace Demo {

constexpr uint32_t kAllToAllGroupSimtThreads = 32U;
constexpr uint32_t kAllToAllGroupSimtMaxTasks = 32U;
constexpr uint32_t kAllToAllGroupSimtPendingMaxTasks = 64U;
constexpr uint32_t kAllToAllGroupSimtMaxQueues =
    kAllToAllGroupSimtPendingMaxTasks;
constexpr uint32_t kAllToAllGroupSimtWqesPerTask = 2U;
constexpr uint32_t kAllToAllGroupSimtPostCombined = 0U;
constexpr uint32_t kAllToAllGroupSimtPostPayload = 1U;
constexpr uint32_t kAllToAllGroupSimtPostSignal = 2U;
constexpr uint32_t kAllToAllGroupSimtCreditScratchBytes = 512U;

struct AllToAllGroupSimtBatch {
    uint32_t sendCoreCount;
    uint32_t active[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t configStatus[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t configOffset[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t configRegionBytes[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t localAddr[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t remoteAddr[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t signalLocalAddr[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t signalAddr[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t signal[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t byteCount[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t peer[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t qpIdx[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t payloadRegionIndex[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t signalRegionIndex[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t worker[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t group[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t pass[kAllToAllGroupSimtPendingMaxTasks];
    uint64_t postBegin[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t reservedHead[kAllToAllGroupSimtPendingMaxTasks];
    uint32_t queuePeer[kAllToAllGroupSimtMaxQueues];
    uint32_t queueQpIdx[kAllToAllGroupSimtMaxQueues];
    uint32_t queueTaskBegin[kAllToAllGroupSimtMaxQueues];
    uint32_t queueTaskCount[kAllToAllGroupSimtMaxQueues];
    uint32_t queueHead[kAllToAllGroupSimtMaxQueues];
    uint32_t queueExpectedCount[kAllToAllGroupSimtMaxQueues];
    uint32_t queueCompletedTail[kAllToAllGroupSimtMaxQueues];
    uint32_t queueQuietStatus[kAllToAllGroupSimtMaxQueues];
    uint32_t queuePollCount[kAllToAllGroupSimtMaxQueues];
};

constexpr uint32_t kAllToAllGroupSimtBatchStorageBytes =
    (sizeof(AllToAllGroupSimtBatch) +
        kAllToAllGroupSimtCreditScratchBytes - 1U) /
        kAllToAllGroupSimtCreditScratchBytes *
        kAllToAllGroupSimtCreditScratchBytes;

static_assert(kAllToAllGroupSimtBatchStorageBytes +
        kAllToAllGroupSimtCreditScratchBytes <= 64U * 1024U,
    "grouped AllToAll SIMT batch and credit scratch must fit in relay UB");

__aicore__ inline uint32_t AllToAllGroupSimtWqesForPhase(uint32_t phase)
{
    return phase == kAllToAllGroupSimtPostCombined ?
        kAllToAllGroupSimtWqesPerTask : 1U;
}

__simt_callee__ inline bool AllToAllGroupSimtResolveRegisteredOffset(
    const __gm__ TileXRUDMARegistry* registry, int32_t peer,
    uint64_t byteOffset, UDMARegionLocation& location)
{
    uint64_t cursor = 0U;
    for (uint32_t regionIndex = 0U; regionIndex < registry->regionCount;
         ++regionIndex) {
        const auto& region = registry->regions[peer][regionIndex];
        if (byteOffset >= cursor && byteOffset - cursor < region.bytes) {
            location.regionIndex = regionIndex;
            location.regionOffset = byteOffset - cursor;
            location.bytesAvailable = region.bytes - location.regionOffset;
            location.addr = reinterpret_cast<__gm__ uint8_t*>(
                region.base + location.regionOffset);
            return true;
        }
        cursor += region.bytes;
    }
    return false;
}

__simt_callee__ inline __gm__ UDMAMemInfo* AllToAllGroupSimtRemoteMemInfo(
    __gm__ UDMAInfo* udmaInfo, uint32_t peer, uint32_t qpIdx,
    uint32_t regionIndex)
{
    const uint32_t regionCount =
        udmaInfo->regionCount == 0U ? 1U : udmaInfo->regionCount;
    const uint64_t index =
        (static_cast<uint64_t>(peer) * udmaInfo->qpNum + qpIdx) *
            regionCount + regionIndex;
    return reinterpret_cast<__gm__ UDMAMemInfo*>(
        udmaInfo->memPtr + index * sizeof(UDMAMemInfo));
}

__simt_callee__ inline bool AllToAllGroupSimtPeerInRouteStage(
    int32_t rank, int32_t peer, uint32_t routeStage)
{
    if (rank < 0 || peer < 0 || rank == peer || routeStage > 10U) {
        return false;
    }
    if (routeStage == 0U || routeStage == 7U || routeStage == 10U) {
        return true;
    }
    const bool crossNode = rank / static_cast<int32_t>(kAllToAllGroupRanksPerNode) !=
        peer / static_cast<int32_t>(kAllToAllGroupRanksPerNode);
    if (routeStage == 1U || routeStage == 4U || routeStage == 5U) {
        return !crossNode;
    }
    return crossNode;
}

__simt_callee__ inline int32_t AllToAllGroupSimtPeer(
    int32_t rank, int32_t rankSize, uint32_t group, uint32_t lane,
    uint32_t groupWidth)
{
    if (rankSize < 8 || rankSize > TILEXR_MAX_RANK_SIZE ||
        (rankSize & 7) != 0 || rank < 0 || rank >= rankSize ||
        (groupWidth != 16U && groupWidth != 4U) || lane >= groupWidth) {
        return -1;
    }
    const uint32_t halfWidth = groupWidth / 2U;
    const uint32_t index = lane < halfWidth ? lane : lane - halfWidth;
    const int32_t distance = static_cast<int32_t>(group * halfWidth + index + 1U);
    const int32_t diameter = rankSize / 2;
    if (distance > diameter || (lane >= halfWidth && distance == diameter)) {
        return -1;
    }
    return lane < halfWidth ? (rank + distance) % rankSize :
        (rank - distance + rankSize) % rankSize;
}

__simt_callee__ inline uint32_t AllToAllGroupSimtQpWeight(
    __gm__ UDMAInfo* udmaInfo, uint32_t peer, uint32_t qp)
{
    if (udmaInfo->qpWeightPtr == 0U) {
        return 1U;
    }
    auto weights = reinterpret_cast<__gm__ uint32_t*>(udmaInfo->qpWeightPtr);
    const uint32_t weight = weights[peer * udmaInfo->qpNum + qp];
    return weight == 0U ? 1U : weight;
}

__simt_callee__ inline bool AllToAllGroupSimtCoResidentPeer(
    int32_t rank, int32_t peer, int32_t localRankSize, uint32_t npuCount)
{
    return npuCount != 0U && localRankSize > static_cast<int32_t>(npuCount) &&
        rank >= 0 && peer >= 0 && rank != peer &&
        rank / localRankSize == peer / localRankSize &&
        (rank % localRankSize) % static_cast<int32_t>(npuCount) ==
            (peer % localRankSize) % static_cast<int32_t>(npuCount);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kAllToAllGroupSimtThreads)
inline void AllToAllGroupSimtBuildVf(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t taskBase,
    uint32_t workerCount,
    const __gm__ CommArgs* args,
    __gm__ TileXRUDMARegistry* registry, __gm__ int32_t* input,
    uint64_t tokenBase, uint32_t invocationId, uint32_t group, uint32_t pass,
    int32_t elementsPerPeer, int32_t chunkElementOffset,
    int32_t currentElements, uint64_t payloadOffset, uint64_t signalOffset,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t npuCount)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (thread < workerCount) {
        const uint32_t slot = taskBase + thread;
        batch->active[slot] = 0U;
        batch->configStatus[slot] = 0U;
        const uint32_t worker = batch->worker[slot];
        batch->queuePollCount[slot] = 0U;
        const uint32_t lane = worker % 16U;
        const uint32_t route = worker / 16U;
        const int32_t rank = args->rank;
        const int32_t peer = AllToAllGroupSimtPeer(
            rank, args->rankSize, group, lane, groupWidth);
        const bool routeRuns = (routeStage == 2U) ? route == 0U :
            ((routeStage == 3U) ? route == 1U : true);
        if (peer >= 0 && routeRuns &&
            !AllToAllGroupSimtCoResidentPeer(
                rank, peer, args->localRankSize, npuCount) &&
            AllToAllGroupSimtPeerInRouteStage(rank, peer, routeStage)) {
            auto udmaInfo = reinterpret_cast<__gm__ UDMAInfo*>(args->udmaInfoPtr);
            const uint32_t qpCount = udmaInfo->qpNum == 0U ? 1U : udmaInfo->qpNum;
            uint32_t primaryQp = 0U;
            uint32_t primaryWeight = AllToAllGroupSimtQpWeight(udmaInfo, peer, 0U);
            for (uint32_t qp = 1U; qp < qpCount; ++qp) {
                const uint32_t weight = AllToAllGroupSimtQpWeight(udmaInfo, peer, qp);
                if (weight > primaryWeight) {
                    primaryQp = qp;
                    primaryWeight = weight;
                }
            }
            uint32_t secondaryQp = primaryQp;
            uint32_t secondaryWeight = 0U;
            for (uint32_t qp = 0U; qp < qpCount; ++qp) {
                const uint32_t weight = AllToAllGroupSimtQpWeight(udmaInfo, peer, qp);
                if (weight != 0U && weight < primaryWeight && weight > secondaryWeight) {
                    secondaryQp = qp;
                    secondaryWeight = weight;
                }
            }
            const bool crossNode = rank / static_cast<int32_t>(kAllToAllGroupRanksPerNode) !=
                peer / static_cast<int32_t>(kAllToAllGroupRanksPerNode);
            if (multiChannel == 0U || !crossNode || secondaryQp == primaryQp) {
                secondaryWeight = 0U;
            }
            uint64_t numerator = primaryWeight;
            uint64_t denominator = static_cast<uint64_t>(primaryWeight) + secondaryWeight;
            if (secondaryWeight == 0U) {
                numerator = 1U;
                denominator = 1U;
            } else if (primaryRouteParts != kAllToAllGroupAutoPrimaryParts) {
                numerator = primaryRouteParts > kAllToAllGroupRouteParts ?
                    kAllToAllGroupRouteParts : primaryRouteParts;
                denominator = kAllToAllGroupRouteParts;
            }
            const uint32_t primaryElements = denominator == 0U ?
                static_cast<uint32_t>(elementsPerPeer) : static_cast<uint32_t>(
                    static_cast<uint64_t>(elementsPerPeer) * numerator / denominator);
            const uint32_t passBegin = static_cast<uint32_t>(chunkElementOffset);
            const uint32_t passEndCandidate = passBegin + static_cast<uint32_t>(currentElements);
            const uint32_t passEnd = passEndCandidate < static_cast<uint32_t>(elementsPerPeer) ?
                passEndCandidate : static_cast<uint32_t>(elementsPerPeer);
            const uint32_t routeBegin = route == 0U ? 0U : primaryElements;
            const uint32_t routeEnd = route == 0U ? primaryElements :
                static_cast<uint32_t>(elementsPerPeer);
            const uint32_t segmentBegin = passBegin > routeBegin ? passBegin : routeBegin;
            const uint32_t segmentEnd = passEnd < routeEnd ? passEnd : routeEnd;
            if (segmentEnd > segmentBegin) {
                const uint32_t selectedQp = route == 0U ? primaryQp : secondaryQp;
                const uint64_t bytesPerPeer =
                    static_cast<uint64_t>(elementsPerPeer) * sizeof(int32_t);
                const uint64_t remotePayloadOffset = payloadOffset +
                    static_cast<uint64_t>(rank) * bytesPerPeer +
                    static_cast<uint64_t>(segmentBegin) * sizeof(int32_t);
                const uint64_t remoteSignalOffset = signalOffset +
                    static_cast<uint64_t>(rank) * 1024U +
                    static_cast<uint64_t>(route) * 512U;
                const uint32_t byteCount = (segmentEnd - segmentBegin) * sizeof(int32_t);
                const bool registryValid = registry->magic == TILEXR_UDMA_REGISTRY_MAGIC &&
                    registry->version == TILEXR_UDMA_REGISTRY_VERSION &&
                    registry->regionCount != 0U &&
                    registry->regionCount <= TILEXR_UDMA_MAX_REGIONS &&
                    registry->rankSize <= TILEXR_MAX_RANK_SIZE &&
                    static_cast<uint32_t>(peer) < registry->rankSize;
                UDMARegionLocation payloadLocation {};
                UDMARegionLocation signalLocation {};
                const bool payloadValid = registryValid &&
                    AllToAllGroupSimtResolveRegisteredOffset(
                        registry, peer, remotePayloadOffset, payloadLocation) &&
                    byteCount <= payloadLocation.bytesAvailable;
                const bool signalValid = registryValid &&
                    AllToAllGroupSimtResolveRegisteredOffset(
                        registry, peer, remoteSignalOffset, signalLocation) &&
                    sizeof(uint64_t) <= signalLocation.bytesAvailable;
                batch->peer[slot] = static_cast<uint32_t>(peer);
                batch->qpIdx[slot] = selectedQp;
                if (!payloadValid || !signalValid) {
                    batch->configStatus[slot] =
                        (payloadValid ? 0U : 1U) | (signalValid ? 0U : 2U);
                    batch->configOffset[slot] =
                        payloadValid ? remoteSignalOffset : remotePayloadOffset;
                    batch->configRegionBytes[slot] = payloadValid ?
                        signalLocation.bytesAvailable : payloadLocation.bytesAvailable;
                } else {
                    const uint64_t signal =
                        (static_cast<uint64_t>(invocationId) + 1ULL) << 32U |
                        static_cast<uint64_t>(invocationId & 1U) << 31U |
                        static_cast<uint64_t>(group) << 16U |
                        (static_cast<uint64_t>(pass) + 1ULL);
                    auto signalLocal = reinterpret_cast<__gm__ uint64_t*>(
                        tokenBase + static_cast<uint64_t>(slot) * sizeof(uint64_t));
                    *signalLocal = signal;
                    batch->localAddr[slot] = reinterpret_cast<uint64_t>(input +
                        static_cast<uint64_t>(peer) * elementsPerPeer + segmentBegin);
                    batch->remoteAddr[slot] = reinterpret_cast<uint64_t>(
                        payloadLocation.addr);
                    batch->signalLocalAddr[slot] = reinterpret_cast<uint64_t>(signalLocal);
                    batch->signalAddr[slot] = reinterpret_cast<uint64_t>(
                        signalLocation.addr);
                    batch->signal[slot] = signal;
                    batch->byteCount[slot] = byteCount;
                    batch->payloadRegionIndex[slot] = payloadLocation.regionIndex;
                    batch->signalRegionIndex[slot] = signalLocation.regionIndex;
                    batch->group[slot] = group;
                    batch->pass[slot] = pass;
                    batch->queuePeer[slot] = static_cast<uint32_t>(peer);
                    batch->queueQpIdx[slot] = selectedQp;
                    batch->queueTaskBegin[slot] = slot;
                    batch->queueTaskCount[slot] = 1U;
                    batch->active[slot] = 1U;
                }
            }
        }
    }
    asc_syncthreads();
}

__simt_callee__ inline void AllToAllGroupSimtWriteWqe(
    __gm__ UDMAWQCtx* wq, __gm__ UDMAMemInfo* remoteMem,
    uint32_t head, uint64_t localAddr, uint64_t remoteAddr,
    uint32_t byteCount, uint32_t sqeFlag)
{
    const uint32_t depth = wq->depth;
    const uint32_t wqeSize = 1U << wq->baseBkShift;
    auto words = reinterpret_cast<__gm__ uint32_t*>(
        wq->bufAddr + static_cast<uint64_t>(wqeSize) * (head % depth));
    const uint32_t owner = (head & depth) == 0U ? 1U : 0U;
    const uint32_t tokenEnable = remoteMem->tokenValueValid ? 1U : 0U;
    const uint64_t eidAddr = remoteMem->eidAddr;
    auto eid = reinterpret_cast<__gm__ uint32_t*>(eidAddr);

    words[0] = (head % depth) |
        ((sqeFlag & 0xFFU) << 16U) |
        (tokenEnable << 28U) |
        ((remoteMem->rmtJettyType & 0x3U) << 29U) |
        (owner << 31U);
    words[1] = static_cast<uint32_t>(remoteMem->targetHint) |
        (static_cast<uint32_t>(UDMAOpcode::WRITE) << 8U);
    words[2] = (remoteMem->tpn & 0xFFFFFFU) | (1U << 24U);
    words[3] = remoteMem->tid & 0xFFFFFU;
    words[4] = eid[0];
    words[5] = eid[1];
    words[6] = eid[2];
    words[7] = eid[3];
    words[8] = remoteMem->rmtTokenValue;
    words[9] = 0U;
    words[10] = static_cast<uint32_t>(remoteAddr);
    words[11] = static_cast<uint32_t>(remoteAddr >> 32U);

    words[12] = byteCount;
    words[13] = wq->localTokenId;
    words[14] = static_cast<uint32_t>(localAddr);
    words[15] = static_cast<uint32_t>(localAddr >> 32U);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kAllToAllGroupSimtThreads)
inline void AllToAllGroupSimtPostVf(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueBegin,
    uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t queue = queueBegin + thread;
         queue < queueBegin + queueCount;
         queue += kAllToAllGroupSimtThreads) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        const uint32_t peer = batch->queuePeer[queue];
        const uint32_t qpIdx = batch->queueQpIdx[queue];
        const uint32_t taskBegin = batch->queueTaskBegin[queue];
        const uint32_t taskCount = batch->queueTaskCount[queue];
        const uint32_t queueIndex = peer * udmaInfo->qpNum + qpIdx;
        __gm__ UDMAWQCtx* wq = reinterpret_cast<__gm__ UDMAWQCtx*>(
            udmaInfo->sqPtr + static_cast<uint64_t>(queueIndex) * sizeof(UDMAWQCtx));
        const uint32_t firstHead = asc_atomic_add(
            reinterpret_cast<__gm__ uint32_t*>(wq->headAddr),
            taskCount * kAllToAllGroupSimtWqesPerTask);
        const uint32_t wqeCount = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(
                reinterpret_cast<__gm__ uint32_t*>(wq->wqeCntAddr));
        batch->queueHead[queue] = firstHead +
            taskCount * kAllToAllGroupSimtWqesPerTask;
        batch->queueExpectedCount[queue] = wqeCount +
            taskCount * kAllToAllGroupSimtWqesPerTask;
        for (uint32_t offset = 0U; offset < taskCount; ++offset) {
            const uint32_t task = taskBegin + offset;
            const uint32_t head = firstHead +
                offset * kAllToAllGroupSimtWqesPerTask;
            batch->reservedHead[task] = head;
            __gm__ UDMAMemInfo* payloadRemoteMem =
                AllToAllGroupSimtRemoteMemInfo(
                    udmaInfo, peer, qpIdx, batch->payloadRegionIndex[task]);
            __gm__ UDMAMemInfo* signalRemoteMem =
                AllToAllGroupSimtRemoteMemInfo(
                    udmaInfo, peer, qpIdx, batch->signalRegionIndex[task]);
            AllToAllGroupSimtWriteWqe(
                wq, payloadRemoteMem, head, batch->localAddr[task],
                batch->remoteAddr[task], batch->byteCount[task],
                TILEXR_UDMA_SQE_FLAG_COMPLETION);
            AllToAllGroupSimtWriteWqe(
                wq, signalRemoteMem, head + 1U,
                batch->signalLocalAddr[task], batch->signalAddr[task],
                sizeof(uint64_t), TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
        }
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kAllToAllGroupSimtThreads)
inline void AllToAllGroupSimtPostPayloadVf(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueBegin,
    uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t queue = queueBegin + thread;
         queue < queueBegin + queueCount;
         queue += kAllToAllGroupSimtThreads) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        const uint32_t peer = batch->queuePeer[queue];
        const uint32_t qpIdx = batch->queueQpIdx[queue];
        const uint32_t taskBegin = batch->queueTaskBegin[queue];
        const uint32_t taskCount = batch->queueTaskCount[queue];
        const uint32_t queueIndex = peer * udmaInfo->qpNum + qpIdx;
        __gm__ UDMAWQCtx* wq = reinterpret_cast<__gm__ UDMAWQCtx*>(
            udmaInfo->sqPtr + static_cast<uint64_t>(queueIndex) * sizeof(UDMAWQCtx));
        const uint32_t firstHead = asc_atomic_add(
            reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), taskCount);
        const uint32_t wqeCount = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(
                reinterpret_cast<__gm__ uint32_t*>(wq->wqeCntAddr));
        batch->queueHead[queue] = firstHead + taskCount;
        batch->queueExpectedCount[queue] = wqeCount + taskCount;
        for (uint32_t offset = 0U; offset < taskCount; ++offset) {
            const uint32_t task = taskBegin + offset;
            const uint32_t head = firstHead + offset;
            batch->reservedHead[task] = head;
            __gm__ UDMAMemInfo* payloadRemoteMem =
                AllToAllGroupSimtRemoteMemInfo(
                    udmaInfo, peer, qpIdx, batch->payloadRegionIndex[task]);
            AllToAllGroupSimtWriteWqe(
                wq, payloadRemoteMem, head, batch->localAddr[task],
                batch->remoteAddr[task], batch->byteCount[task],
                TILEXR_UDMA_SQE_FLAG_COMPLETION);
        }
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kAllToAllGroupSimtThreads)
inline void AllToAllGroupSimtPostSignalVf(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueBegin,
    uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t queue = queueBegin + thread;
         queue < queueBegin + queueCount;
         queue += kAllToAllGroupSimtThreads) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        const uint32_t peer = batch->queuePeer[queue];
        const uint32_t qpIdx = batch->queueQpIdx[queue];
        const uint32_t taskBegin = batch->queueTaskBegin[queue];
        const uint32_t taskCount = batch->queueTaskCount[queue];
        const uint32_t queueIndex = peer * udmaInfo->qpNum + qpIdx;
        __gm__ UDMAWQCtx* wq = reinterpret_cast<__gm__ UDMAWQCtx*>(
            udmaInfo->sqPtr + static_cast<uint64_t>(queueIndex) * sizeof(UDMAWQCtx));
        const uint32_t firstHead = asc_atomic_add(
            reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), taskCount);
        const uint32_t wqeCount = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(
                reinterpret_cast<__gm__ uint32_t*>(wq->wqeCntAddr));
        batch->queueHead[queue] = firstHead + taskCount;
        batch->queueExpectedCount[queue] = wqeCount + taskCount;
        for (uint32_t offset = 0U; offset < taskCount; ++offset) {
            const uint32_t task = taskBegin + offset;
            const uint32_t head = firstHead + offset;
            batch->reservedHead[task] = head;
            __gm__ UDMAMemInfo* signalRemoteMem =
                AllToAllGroupSimtRemoteMemInfo(
                    udmaInfo, peer, qpIdx, batch->signalRegionIndex[task]);
            AllToAllGroupSimtWriteWqe(
                wq, signalRemoteMem, head,
                batch->signalLocalAddr[task], batch->signalAddr[task],
                sizeof(uint64_t), TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
        }
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kAllToAllGroupSimtThreads)
inline void AllToAllGroupSimtQuietVf(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t queue = thread; queue < queueCount;
         queue += kAllToAllGroupSimtThreads) {
        if (batch->active[queue] == 0U) {
            continue;
        }
        const uint32_t peer = batch->queuePeer[queue];
        const uint32_t qpIdx = batch->queueQpIdx[queue];
        const uint32_t queueIndex = peer * udmaInfo->qpNum + qpIdx;
        __gm__ UDMAWQCtx* wq = reinterpret_cast<__gm__ UDMAWQCtx*>(
            udmaInfo->sqPtr + static_cast<uint64_t>(queueIndex) * sizeof(UDMAWQCtx));
        __gm__ UDMACQCtx* cq = reinterpret_cast<__gm__ UDMACQCtx*>(
            udmaInfo->scqPtr + static_cast<uint64_t>(queueIndex) * sizeof(UDMACQCtx));
        uint32_t curTail = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(
                reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr));
        const uint32_t expected = batch->queueExpectedCount[queue];
        uint32_t quietStatus = 0U;
        uint32_t pollCount = 0U;
        const uint32_t cqeSize = 1U << cq->baseBkShift;

        while (curTail != expected && quietStatus == 0U) {
            auto cqeWordAddr = reinterpret_cast<__gm__ uint32_t*>(
                cq->bufAddr + static_cast<uint64_t>(cqeSize) *
                    (curTail & (TILEXR_UDMA_CQ_DEPTH - 1U)));
            const bool validOwner =
                ((curTail / TILEXR_UDMA_CQ_DEPTH) & 1U) != 0U;
            uint32_t cqeWord = 0U;
            uint32_t retry = 0U;
            for (; retry < TILEXR_UDMA_MAX_RETRY_TIMES; ++retry) {
                cqeWord = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                    L1CacheType::NON_CACHEABLE>(cqeWordAddr);
                const bool owner = ((cqeWord >> 2U) & 1U) != 0U;
                if ((validOwner ^ owner) != 0U) {
                    break;
                }
            }
            pollCount += retry < TILEXR_UDMA_MAX_RETRY_TIMES ? retry + 1U : retry;
            if (retry == TILEXR_UDMA_MAX_RETRY_TIMES) {
                quietStatus = 0xFFU;
                break;
            }
            const uint32_t status = (cqeWord >> 24U) & 0xFFU;
            const uint32_t subStatus = (cqeWord >> 16U) & 0xFFU;
            if (status != 0U || subStatus != 0U) {
                quietStatus = (status << 8U) | subStatus;
                break;
            }
            ++curTail;
        }
        batch->queueCompletedTail[queue] = curTail;
        batch->queueQuietStatus[queue] = quietStatus;
        batch->queuePollCount[queue] = pollCount;
    }
    asc_syncthreads();
}

__aicore__ inline void AllToAllGroupSimtPost(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueBegin,
    uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    asc_vf_call<AllToAllGroupSimtPostVf>(
        dim3{kAllToAllGroupSimtThreads, 1U, 1U},
        batch, queueBegin, queueCount, udmaInfo);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AllToAllGroupSimtPostPayload(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueBegin,
    uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    asc_vf_call<AllToAllGroupSimtPostPayloadVf>(
        dim3{kAllToAllGroupSimtThreads, 1U, 1U},
        batch, queueBegin, queueCount, udmaInfo);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AllToAllGroupSimtPostSignal(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueBegin,
    uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    asc_vf_call<AllToAllGroupSimtPostSignalVf>(
        dim3{kAllToAllGroupSimtThreads, 1U, 1U},
        batch, queueBegin, queueCount, udmaInfo);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AllToAllGroupSimtBuildPrepared(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t taskBase,
    uint32_t workerCount,
    const __gm__ CommArgs* args,
    __gm__ TileXRUDMARegistry* registry, __gm__ int32_t* input,
    uint64_t tokenBase, uint32_t invocationId, uint32_t group, uint32_t pass,
    int32_t elementsPerPeer, int32_t chunkElementOffset,
    int32_t currentElements, uint64_t payloadOffset, uint64_t signalOffset,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t npuCount)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    asc_vf_call<AllToAllGroupSimtBuildVf>(
        dim3{kAllToAllGroupSimtThreads, 1U, 1U}, batch,
        taskBase, workerCount, args, registry, input, tokenBase,
        invocationId, group, pass, elementsPerPeer, chunkElementOffset,
        currentElements, payloadOffset, signalOffset, routeStage,
        multiChannel, primaryRouteParts, groupWidth, npuCount);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AllToAllGroupSimtQuiet(
    __ubuf__ AllToAllGroupSimtBatch* batch, uint32_t queueCount,
    __gm__ UDMAInfo* udmaInfo)
{
    asc_vf_call<AllToAllGroupSimtQuietVf>(
        dim3{kAllToAllGroupSimtThreads, 1U, 1U},
        batch, queueCount, udmaInfo);
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_GROUP_SIMT_H
