/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_udma.h"

constexpr int32_t TILEXR_UDMA_DEMO_MAGIC = 0x5444554d; // "TDUM"
constexpr uint64_t TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET = TileXR::IPC_DATA_OFFSET;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE = 6;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER = TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER = TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SUM = TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER + 1;

extern "C" __global__ __aicore__ void tilexr_udma_all_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
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
    }
    if (!enabled) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)));
        TileXR::UDMAQuiet(args, peer);
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_put_signal_kernel(
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
    }
    if (!enabled) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    uint64_t signalBaseOffset = static_cast<uint64_t>(rankSize) * elementsPerRank * sizeof(int32_t);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutSignalNbi<int32_t>(args, peer, localSrc,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)),
            signalBaseOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t), signal);
        TileXR::UDMAQuiet(args, peer);
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t outputByteOffset)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(outputByteOffset);
    }
    if (!enabled) {
        return;
    }

    auto selfSrc = input + rank * elementsPerPeer;
    auto selfDst = output + rank * elementsPerPeer;
    for (int32_t i = 0; i < elementsPerPeer; ++i) {
        selfDst[i] = selfSrc[i];
    }

    uint32_t bytes = static_cast<uint32_t>(elementsPerPeer * sizeof(int32_t));
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        auto localSrc = input + peer * elementsPerPeer;
        uint64_t remoteOffset = outputByteOffset +
            static_cast<uint64_t>(rank) * elementsPerPeer * sizeof(int32_t);
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteOffset, bytes);
        uint32_t status = TileXR::UDMAQuietStatus(args, peer);
        if (debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto localSrc = input + dstRank * elementsPerPeer;
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + rank * elementsPerPeer;
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            remoteDst[i] = localSrc[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
    for (int32_t srcRank = 0; srcRank < args->rankSize; ++srcRank) {
        auto localSrc = localBase + srcRank * elementsPerPeer;
        auto localDst = output + srcRank * elementsPerPeer;
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            localDst[i] = localSrc[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_reduce_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + rank * elementsPerRank;
        for (int32_t i = 0; i < elementsPerRank; ++i) {
            remoteDst[i] = input[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_reduce_ipc_sum_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
    for (int32_t i = 0; i < elementsPerRank; ++i) {
        int32_t sum = 0;
        for (int32_t srcRank = 0; srcRank < args->rankSize; ++srcRank) {
            sum += localBase[srcRank * elementsPerRank + i];
        }
        output[i] = sum;
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SUM] = 1;
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
    int32_t elementsPerRank, uint64_t signal)
{
    tilexr_udma_put_signal_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, signals, debug, elementsPerRank, signal);
}

void launch_tilexr_udma_all_to_all(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset)
{
    tilexr_udma_all_to_all_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, outputByteOffset);
}

void launch_tilexr_all_to_all_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_ipc_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerPeer);
}

void launch_tilexr_all_reduce_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_all_reduce_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerRank);
}

void launch_tilexr_all_reduce_ipc_sum(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_all_reduce_ipc_sum_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerRank);
}

void launch_tilexr_udma_registered_smoke(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR local, GM_ADDR debug, uint32_t bytes, uint64_t signal)
{
    tilexr_udma_registered_smoke_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, local, debug, bytes, signal);
}
