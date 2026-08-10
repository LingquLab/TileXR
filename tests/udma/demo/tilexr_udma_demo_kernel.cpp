/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_udma.h"

constexpr int32_t TILEXR_UDMA_DEMO_MAGIC = 0x5444554d; // "TDUM"
constexpr uint32_t TILEXR_UDMA_DEMO_MAX_QPS = TileXR::TILEXR_UDMA_DEVICE_MAX_QP_COUNT;
constexpr uint32_t TILEXR_UDMA_DEMO_QP_COUNT_WORD = 5U;
constexpr uint32_t TILEXR_UDMA_DEMO_QP_STATUS_BASE_WORD = 6U;

__aicore__ inline bool TileXRUDMAMultiQpAllGather(
    const __gm__ TileXR::CommArgs* args, const AscendC::LocalTensor<uint8_t>& wqeScratch,
    __gm__ int32_t* data, __gm__ int32_t* debug,
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
    const uint64_t elementsPerQp = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(elementsPerRank);
    const uint32_t segmentBytes = static_cast<uint32_t>(elementsPerRank) * sizeof(int32_t);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }

        for (uint32_t qpIdx = 0U; qpIdx < qpCount; ++qpIdx) {
            if (qpStatus[qpIdx] != TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                continue;
            }
            const uint64_t elementOffset =
                qpIdx * elementsPerQp + static_cast<uint64_t>(rank) * elementsPerRank;
            qpStatus[qpIdx] = TileXR::UDMAPutNbiOnQpWithFlagDeferred<int32_t>(
                args, wqeScratch, peer, qpIdx, data + elementOffset,
                elementOffset * sizeof(int32_t), segmentBytes,
                TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
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
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wqeBuf;
    pipe.InitBuffer(wqeBuf, TileXR::TILEXR_UDMA_WQE_SCRATCH_BYTES);

    (void)TileXRUDMAMultiQpAllGather(args, wqeBuf.Get<uint8_t>(), data, debug, elementsPerRank);
}

extern "C" __global__ __aicore__ void tilexr_udma_put_signal_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR signalGM, GM_ADDR debugGM,
    int32_t elementsPerRank, uint64_t signalByteOffset, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);
    (void)signalGM;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wqeBuf;
    pipe.InitBuffer(wqeBuf, TileXR::TILEXR_UDMA_WQE_SCRATCH_BYTES);
    auto wqeScratch = wqeBuf.Get<uint8_t>();

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    if (!TileXRUDMAMultiQpAllGather(args, wqeScratch, data, debug, elementsPerRank)) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutSignalNbi<int32_t>(args, wqeScratch, peer, localSrc,
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
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wqeBuf;
    pipe.InitBuffer(wqeBuf, TileXR::TILEXR_UDMA_WQE_SCRATCH_BYTES);
    auto wqeScratch = wqeBuf.Get<uint8_t>();

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
        TileXR::UDMAPutSignalNbi<int32_t>(args, wqeScratch, peer,
            data + rank * elementsPerRank,
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
        TileXR::UDMAGetNbi<int32_t>(args, wqeScratch, peer, data + peer * elementsPerRank,
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
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wqeBuf;
    pipe.InitBuffer(wqeBuf, TileXR::TILEXR_UDMA_WQE_SCRATCH_BYTES);
    auto wqeScratch = wqeBuf.Get<uint8_t>();

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
        TileXR::UDMAPutRegisteredNbi<uint8_t>(args, wqeScratch, peer, local, 0, bytes);
        TileXR::UDMAGetRegisteredNbi<uint8_t>(args, wqeScratch, peer, local, 0, bytes);
        TileXR::UDMAPutRegisteredSignalNbi<uint8_t>(
            args, wqeScratch, peer, local, 0, bytes, 0, signal);
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
