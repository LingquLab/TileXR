/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_udma.h"

constexpr int32_t TILEXR_UDMA_DEMO_MAGIC = 0x5444554d; // "TDUM"
constexpr uint32_t TILEXR_UDMA_DEMO_MAX_QPS = 8U;
constexpr uint32_t TILEXR_UDMA_DEMO_QP_COUNT_WORD = 5U;
constexpr uint32_t TILEXR_UDMA_DEMO_QP_STATUS_BASE_WORD = 6U;
constexpr uint32_t TILEXR_UDMA_DEMO_LOCAL_TOKEN_WORD = 14U;
constexpr uint32_t TILEXR_UDMA_DEMO_SGE_TOKEN_WORD = 15U;
constexpr uint32_t TILEXR_UDMA_DEMO_QP_EQUAL_MASK_BASE_WORD = 9U;
constexpr uint32_t TILEXR_UDMA_DEMO_POST_VALIDATION_WORD = 11U;

__aicore__ inline uint32_t TileXRUDMAPostValidationMask(
    const __gm__ TileXR::CommArgs* args, int32_t peer, uint32_t qpIdx,
    const __gm__ int32_t* localSrc, uint64_t byteOffset, uint32_t byteCount)
{
    uint32_t mask = 0U;
    mask |= !TileXR::UDMARegistryEnabled(args) ? 1U << 0U : 0U;
    mask |= !TileXR::UDMARankValid(args, peer) ? 1U << 1U : 0U;
    mask |= !TileXR::UDMAQpValid(args, qpIdx) ? 1U << 2U : 0U;
    const __gm__ TileXR::TileXRUDMARegistry* registry = TileXR::GetUDMARegistry(args);
    mask |= registry->rankSize != static_cast<uint32_t>(args->rankSize) ? 1U << 3U : 0U;
    mask |= !TileXR::UDMARegisteredRangeValid(
        registry, peer, byteOffset, byteCount) ? 1U << 4U : 0U;
    mask |= localSrc == nullptr && byteCount != 0U ? 1U << 5U : 0U;
    __gm__ uint8_t* remoteAddr = TileXR::UDMARegisteredRemoteAddr(
        registry, peer, byteOffset);
    mask |= remoteAddr == nullptr ? 1U << 6U : 0U;
    __gm__ TileXR::UDMAInfo* udmaInfo = TileXR::GetUDMAInfo(args);
    mask |= udmaInfo == nullptr || udmaInfo->sqPtr == 0U || udmaInfo->memPtr == 0U
        ? 1U << 7U : 0U;
    __gm__ TileXR::UDMAWQCtx* qpCtx = TileXR::UDMAGetWQCtx(
        udmaInfo, static_cast<uint32_t>(peer), qpIdx);
    mask |= qpCtx->bufAddr == 0U ? 1U << 8U : 0U;
    mask |= qpCtx->headAddr == 0U ? 1U << 9U : 0U;
    mask |= qpCtx->tailAddr == 0U ? 1U << 10U : 0U;
    mask |= qpCtx->wqeCntAddr == 0U ? 1U << 11U : 0U;
    mask |= qpCtx->dbAddr == 0U ? 1U << 12U : 0U;
    mask |= qpCtx->depth != TileXR::TILEXR_UDMA_SQ_BB_COUNT ? 1U << 13U : 0U;
    mask |= qpCtx->baseBkShift >= 32U ? 1U << 14U : 0U;
    if (qpCtx->baseBkShift < 32U) {
        mask |= (1U << qpCtx->baseBkShift) <
            sizeof(TileXR::UDMASqeCtx) + sizeof(TileXR::UDMASgeCtx) ? 1U << 15U : 0U;
    }
    __gm__ TileXR::UDMAMemInfo* remoteMem = TileXR::UDMAGetRemoteMemInfo(
        udmaInfo, static_cast<uint32_t>(peer), qpIdx);
    mask |= remoteMem->eidAddr == 0U ? 1U << 16U : 0U;
    const uint32_t head = ld_dev(
        reinterpret_cast<__gm__ uint32_t*>(qpCtx->headAddr), 0);
    const uint32_t tail = ld_dev(
        reinterpret_cast<__gm__ uint32_t*>(qpCtx->tailAddr), 0);
    mask |= head - tail > qpCtx->depth ? 1U << 17U : 0U;
    return mask;
}

__aicore__ inline uint32_t TileXRUDMAQueueStateEqualMask(
    __gm__ TileXR::UDMAInfo* udmaInfo, uint32_t peer, uint32_t lhsQp, uint32_t rhsQp)
{
    __gm__ TileXR::UDMAWQCtx* lhsSq = TileXR::UDMAGetWQCtx(udmaInfo, peer, lhsQp);
    __gm__ TileXR::UDMAWQCtx* rhsSq = TileXR::UDMAGetWQCtx(udmaInfo, peer, rhsQp);
    __gm__ TileXR::UDMACQCtx* lhsCq = TileXR::UDMAGetSCQCtx(udmaInfo, peer, lhsQp);
    __gm__ TileXR::UDMACQCtx* rhsCq = TileXR::UDMAGetSCQCtx(udmaInfo, peer, rhsQp);
    uint32_t mask = 0U;
    mask |= lhsSq->bufAddr == rhsSq->bufAddr ? 1U << 0U : 0U;
    mask |= lhsSq->headAddr == rhsSq->headAddr ? 1U << 1U : 0U;
    mask |= lhsSq->tailAddr == rhsSq->tailAddr ? 1U << 2U : 0U;
    mask |= lhsSq->wqeCntAddr == rhsSq->wqeCntAddr ? 1U << 3U : 0U;
    mask |= lhsSq->dbAddr == rhsSq->dbAddr ? 1U << 4U : 0U;
    mask |= lhsCq->bufAddr == rhsCq->bufAddr ? 1U << 5U : 0U;
    mask |= lhsCq->headAddr == rhsCq->headAddr ? 1U << 6U : 0U;
    mask |= lhsCq->tailAddr == rhsCq->tailAddr ? 1U << 7U : 0U;
    mask |= lhsCq->dbAddr == rhsCq->dbAddr ? 1U << 8U : 0U;
    return mask;
}

__aicore__ inline bool TileXRUDMAMultiQpAllGather(
    const __gm__ TileXR::CommArgs* args, __gm__ int32_t* data, __gm__ int32_t* debug,
    int32_t elementsPerRank)
{
    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;
    const bool enabled = TileXR::UDMARegistryEnabled(args);
    const uint32_t qpCount = TileXR::UDMAQpCount(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerRank;
        debug[TILEXR_UDMA_DEMO_QP_COUNT_WORD] = static_cast<int32_t>(qpCount);
    }
    if (!enabled || qpCount == 0U || qpCount > TILEXR_UDMA_DEMO_MAX_QPS) {
        return false;
    }

    uint32_t qpStatus[TILEXR_UDMA_DEMO_MAX_QPS] = {};
    __gm__ TileXR::UDMAInfo* udmaInfo = TileXR::GetUDMAInfo(args);
    const uint64_t elementsPerQp = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(elementsPerRank);
    const uint32_t segmentBytes = static_cast<uint32_t>(elementsPerRank) * sizeof(int32_t);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }

        for (uint32_t qpIdx = 1U; qpIdx < qpCount; ++qpIdx) {
            for (uint32_t previousQp = 0U; previousQp < qpIdx; ++previousQp) {
                const uint32_t equalMask = TileXRUDMAQueueStateEqualMask(
                    udmaInfo, static_cast<uint32_t>(peer), previousQp, qpIdx);
                if (debug != nullptr && previousQp == 0U &&
                    peer == (rank == 0 ? 1 : 0)) {
                    debug[TILEXR_UDMA_DEMO_QP_EQUAL_MASK_BASE_WORD + qpIdx - 1U] =
                        static_cast<int32_t>(equalMask);
                }
                if (equalMask != 0U) {
                    qpStatus[previousQp] = TileXR::TILEXR_UDMA_STATUS_INVALID;
                    qpStatus[qpIdx] = TileXR::TILEXR_UDMA_STATUS_INVALID;
                }
            }
        }
        for (uint32_t qpIdx = 0U; qpIdx < qpCount; ++qpIdx) {
            if (qpStatus[qpIdx] != TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                continue;
            }
            const uint64_t elementOffset =
                qpIdx * elementsPerQp + static_cast<uint64_t>(rank) * elementsPerRank;
            if (debug != nullptr && qpIdx == 0U && peer == (rank == 0 ? 1 : 0)) {
                debug[TILEXR_UDMA_DEMO_POST_VALIDATION_WORD] = static_cast<int32_t>(
                    TileXRUDMAPostValidationMask(args, peer, qpIdx, data + elementOffset,
                        elementOffset * sizeof(int32_t), segmentBytes));
            }
            qpStatus[qpIdx] = TileXR::UDMAPutNbiOnQpWithFlagDeferred<int32_t>(
                args, peer, qpIdx, data + elementOffset, elementOffset * sizeof(int32_t), segmentBytes,
                TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
            if (debug != nullptr && qpIdx == 0U) {
                __gm__ TileXR::UDMAWQCtx* qpCtx =
                    TileXR::UDMAGetWQCtx(udmaInfo, static_cast<uint32_t>(peer), qpIdx);
                const uint32_t head = ld_dev(
                    reinterpret_cast<__gm__ uint32_t*>(qpCtx->headAddr), 0);
                __gm__ TileXR::UDMASgeCtx* sge = reinterpret_cast<__gm__ TileXR::UDMASgeCtx*>(
                    TileXR::UDMAGetSqLogicalAddr(
                        qpCtx, head - 1U, sizeof(TileXR::UDMASqeCtx)));
                debug[TILEXR_UDMA_DEMO_LOCAL_TOKEN_WORD] =
                    static_cast<int32_t>(qpCtx->localTokenId);
                debug[TILEXR_UDMA_DEMO_SGE_TOKEN_WORD] =
                    static_cast<int32_t>(sge->tokenId);
            }
        }
        for (uint32_t qpIdx = 0U; qpIdx < qpCount; ++qpIdx) {
            if (qpStatus[qpIdx] == TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                qpStatus[qpIdx] = TileXR::UDMAFlushQpDoorbell(args, peer, qpIdx);
            }
        }
        for (uint32_t qpIdx = 0U; qpIdx < qpCount; ++qpIdx) {
            if (qpStatus[qpIdx] == TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                qpStatus[qpIdx] = TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
            }
        }
    }

    bool allQpsSucceeded = true;
    for (uint32_t qpIdx = 0U; qpIdx < qpCount; ++qpIdx) {
        allQpsSucceeded = qpStatus[qpIdx] == TileXR::TILEXR_UDMA_STATUS_SUCCESS && allQpsSucceeded;
        if (debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_QP_STATUS_BASE_WORD + qpIdx] = static_cast<int32_t>(qpStatus[qpIdx]);
        }
    }
    return allQpsSucceeded;
}

extern "C" __global__ __aicore__ void tilexr_udma_all_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    (void)TileXRUDMAMultiQpAllGather(args, data, debug, elementsPerRank);
}

extern "C" __global__ __aicore__ void tilexr_udma_put_signal_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR signalGM, GM_ADDR debugGM,
    int32_t elementsPerRank, uint64_t signalByteOffset, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);
    (void)signalGM;

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    if (!TileXRUDMAMultiQpAllGather(args, data, debug, elementsPerRank)) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutSignalNbi<int32_t>(args, peer, localSrc,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)),
            signalByteOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t), signal);
        const uint32_t quietStatus = TileXR::UDMAQuietStatusOnQp(args, peer, 0U);
        if (debug != nullptr && debug[TILEXR_UDMA_DEMO_QP_STATUS_BASE_WORD] == 0 &&
            quietStatus != TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
            debug[TILEXR_UDMA_DEMO_QP_STATUS_BASE_WORD] = static_cast<int32_t>(quietStatus);
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_slot_signal_get_probe_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR signalGM, GM_ADDR debugGM,
    int32_t elementsPerRank, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto signals = reinterpret_cast<__gm__ uint64_t*>(signalGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerRank;
        debug[5] = 2;
    }
    if (!enabled) {
        return;
    }

    uint64_t signalBaseOffset = static_cast<uint64_t>(rankSize) * elementsPerRank * sizeof(int32_t);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutSignalNbi<int32_t>(args, peer, data + rank * elementsPerRank,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t), sizeof(int32_t),
            signalBaseOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t), signal);
        TileXR::UDMAQuiet(args, peer);
    }

    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        while (signals[peer] != signal) {
        }
        TileXR::UDMAGetNbi<int32_t>(args, peer, data + peer * elementsPerRank,
            peer * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)));
        TileXR::UDMAQuiet(args, peer);
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_registered_smoke_kernel(
    GM_ADDR commArgsGM, GM_ADDR localGM, GM_ADDR debugGM, uint32_t bytes, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto local = reinterpret_cast<__gm__ uint8_t*>(localGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    bool enabled = TileXR::UDMARegistryEnabled(args);
    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = enabled ? 1 : 0;
        debug[2] = static_cast<int32_t>(bytes);
    }
    if (!enabled) {
        return;
    }

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutRegisteredNbi<uint8_t>(args, peer, local, 0, bytes);
        TileXR::UDMAGetRegisteredNbi<uint8_t>(args, peer, local, 0, bytes);
        TileXR::UDMAPutRegisteredSignalNbi<uint8_t>(args, peer, local, 0, bytes, 0, signal);
        TileXR::UDMAQuiet(args, peer);
    }
}

void launch_tilexr_udma_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_udma_all_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, debug, elementsPerRank);
}

void launch_tilexr_udma_put_signal(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signalByteOffset, uint64_t signal)
{
    tilexr_udma_put_signal_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, signals, debug, elementsPerRank, signalByteOffset, signal);
}

void launch_tilexr_udma_slot_signal_get_probe(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signal)
{
    tilexr_udma_slot_signal_get_probe_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, signals, debug, elementsPerRank, signal);
}

void launch_tilexr_udma_registered_smoke(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR local, GM_ADDR debug, uint32_t bytes, uint64_t signal)
{
    tilexr_udma_registered_smoke_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, local, debug, bytes, signal);
}
