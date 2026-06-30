/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_LAYOUT_H
#define TILEXR_UDMA_ALLTOALL_LAYOUT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace TileXR {
namespace Demo {

constexpr int32_t kAllToAllBaseValue = 100000;
constexpr size_t kAllToAllUdmaMaxRegisteredBytes = 512ULL * 1024ULL * 1024ULL;

struct AllToAllChunkPlan {
    uint32_t passCount = 1;
    int32_t chunkElements = 0;
    size_t chunkBytesPerRank = 0;
    size_t registeredBytes = 0;
};

constexpr size_t kAllToAllBigDataMaxRegisteredBytes = 128ULL * 1024ULL * 1024ULL;
constexpr size_t kAllToAllBigDataMultiNodeRegisteredBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr size_t kAllToAllBigDataMultiNodePeerSlotBytes = 16ULL * 1024ULL * 1024ULL;
constexpr size_t kAllToAllBigDataControlSlotBytes = 128ULL;
constexpr uint32_t kAllToAllBigDataCoresPerPeer = 5U;
constexpr uint32_t kAllToAllBigDataSingleNodeShards = 2U;
constexpr uint32_t kAllToAllBigDataLocalCopyShards = kAllToAllBigDataSingleNodeShards;
constexpr uint32_t kAllToAllBigDataPingPongSlots = 2U;
constexpr int32_t kAllToAllBigDataRanksPerNode = 8;
constexpr uint32_t kAllToAllBigDataMultiNodeCopyCores = 16U;
constexpr uint32_t kAllToAllBigDataMultiNodeRecvCores = 16U;
constexpr uint32_t kAllToAllBigDataMultiNodeControlShards = 32U;
constexpr uint32_t kAllToAllBigDataMultiNodeRemoteSendPrimaryCore = 16U;
constexpr uint32_t kAllToAllBigDataMultiNodeRemoteSendSecondaryCore = 17U;
constexpr uint32_t kAllToAllBigDataMultiNodeLocalSendCore = 18U;
constexpr uint32_t kAllToAllBigDataMultiNodeRecvCoreBase = 19U;
constexpr uint32_t kAllToAllBigDataMultiNodeBlockDim =
    kAllToAllBigDataMultiNodeRecvCoreBase + kAllToAllBigDataMultiNodeRecvCores;
constexpr uint32_t kAllToAllBigDataRemotePutOnlyBlockDim = 64U;

struct AllToAllBigDataPlan {
    uint32_t passCount = 1;
    int32_t chunkElements = 0;
    size_t chunkBytesPerPeer = 0;
    size_t dataBytes = 0;
    size_t copyDoneOffset = 0;
    size_t recvCopyDoneOffset = 0;
    size_t remoteSendDoneOffset = 0;
    size_t readySignalOffset = 0;
    size_t ackSignalOffset = 0;
    size_t controlBytes = 0;
    size_t signalBytes = 0;
    size_t registeredBytes = 0;
};

inline int32_t AllToAllValue(int srcRank, int dstRank)
{
    return kAllToAllBaseValue + srcRank * 1000 + dstRank;
}

inline int32_t AllToAllBigDataNormalizeRanksPerNode(int rankSize, int32_t ranksPerNode)
{
    (void)rankSize;
    if (ranksPerNode <= 0) {
        return kAllToAllBigDataRanksPerNode;
    }
    return ranksPerNode;
}

inline bool AllToAllBigDataIsMultiNode(
    int rankSize, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    return rankSize > AllToAllBigDataNormalizeRanksPerNode(rankSize, ranksPerNode);
}

inline bool AllToAllBigDataUse35Core(
    int rankSize, bool force35Core = false, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rankSize, ranksPerNode);
    return AllToAllBigDataIsMultiNode(rankSize, localRanks) ||
        (force35Core && rankSize == localRanks);
}

inline uint32_t AllToAllBigDataShardCount(
    int rankSize, bool force35Core = false, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    return AllToAllBigDataUse35Core(rankSize, force35Core, ranksPerNode) ?
        kAllToAllBigDataMultiNodeControlShards : kAllToAllBigDataSingleNodeShards;
}

inline AllToAllChunkPlan PlanAllToAllUdmaChunks(int rankSize, int32_t elementsPerPeer)
{
    AllToAllChunkPlan plan {};
    if (rankSize <= 0 || elementsPerPeer <= 0) {
        return plan;
    }

    const size_t totalBytesPerRank = static_cast<size_t>(rankSize) * static_cast<size_t>(elementsPerPeer) *
        sizeof(int32_t);
    const size_t maxChunkBytesPerRank =
        std::max<size_t>(sizeof(int32_t) * static_cast<size_t>(rankSize), kAllToAllUdmaMaxRegisteredBytes / 2);
    size_t chunkElements = maxChunkBytesPerRank / (static_cast<size_t>(rankSize) * sizeof(int32_t));
    if (chunkElements == 0) {
        chunkElements = 1;
    }
    if (chunkElements > static_cast<size_t>(elementsPerPeer)) {
        chunkElements = static_cast<size_t>(elementsPerPeer);
    }

    plan.chunkElements = static_cast<int32_t>(chunkElements);
    plan.chunkBytesPerRank = static_cast<size_t>(rankSize) * chunkElements * sizeof(int32_t);
    plan.registeredBytes = plan.chunkBytesPerRank * 2;
    plan.passCount = static_cast<uint32_t>(
        (static_cast<size_t>(elementsPerPeer) + chunkElements - 1) / chunkElements);
    if (plan.chunkBytesPerRank > totalBytesPerRank) {
        plan.chunkBytesPerRank = totalBytesPerRank;
    }
    return plan;
}

inline AllToAllBigDataPlan PlanAllToAllBigDataUdma(
    int rankSize, int32_t elementsPerPeer, bool force35Core = false,
    int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    AllToAllBigDataPlan plan {};
    if (rankSize <= 0 || elementsPerPeer <= 0) {
        return plan;
    }

    const bool use35Core = AllToAllBigDataUse35Core(rankSize, force35Core, ranksPerNode);
    plan.registeredBytes = use35Core ? kAllToAllBigDataMultiNodeRegisteredBytes :
        kAllToAllBigDataMaxRegisteredBytes;
    const uint32_t shardCount = AllToAllBigDataShardCount(rankSize, force35Core, ranksPerNode);
    const size_t maxChunkElements = use35Core ?
        kAllToAllBigDataMultiNodePeerSlotBytes / sizeof(int32_t) :
        static_cast<size_t>(elementsPerPeer);
    size_t chunkElements = static_cast<size_t>(elementsPerPeer);
    if (use35Core && chunkElements > maxChunkElements) {
        chunkElements = maxChunkElements;
    }
    if (chunkElements == 0) {
        chunkElements = 1;
    }
    plan.passCount = static_cast<uint32_t>(
        (static_cast<size_t>(elementsPerPeer) + chunkElements - 1) / chunkElements);
    const size_t slotCountForControls = use35Core ?
        static_cast<size_t>(plan.passCount) : static_cast<size_t>(kAllToAllBigDataPingPongSlots);
    const size_t controlGroupBytes =
        slotCountForControls * static_cast<size_t>(rankSize) *
        static_cast<size_t>(shardCount) * kAllToAllBigDataControlSlotBytes;
    plan.controlBytes = controlGroupBytes;
    plan.signalBytes = use35Core ? 4ULL * controlGroupBytes : 3ULL * controlGroupBytes;
    if (plan.controlBytes + plan.signalBytes >= plan.registeredBytes) {
        return plan;
    }

    const size_t networkPeerCount = static_cast<size_t>(rankSize > 1 ? rankSize - 1 : 1);
    const size_t dataSlotCount = networkPeerCount *
        (use35Core ? static_cast<size_t>(plan.passCount) :
            static_cast<size_t>(kAllToAllBigDataPingPongSlots)) * 2ULL;
    plan.dataBytes = plan.registeredBytes - plan.controlBytes - plan.signalBytes;
    if (use35Core) {
        plan.chunkElements = static_cast<int32_t>(chunkElements);
        plan.chunkBytesPerPeer = static_cast<size_t>(plan.chunkElements) * sizeof(int32_t);
    } else {
        plan.chunkElements = static_cast<int32_t>(
            plan.dataBytes / (dataSlotCount * sizeof(int32_t)));
        if (plan.chunkElements <= 0) {
            plan.chunkElements = 1;
        }
        if (plan.chunkElements > elementsPerPeer) {
            plan.chunkElements = elementsPerPeer;
        }
        plan.chunkBytesPerPeer = static_cast<size_t>(plan.chunkElements) * sizeof(int32_t);
    }

    plan.dataBytes = dataSlotCount * plan.chunkBytesPerPeer;
    if (plan.dataBytes + plan.controlBytes + plan.signalBytes > plan.registeredBytes) {
        return AllToAllBigDataPlan {};
    }
    plan.copyDoneOffset = plan.dataBytes;
    plan.recvCopyDoneOffset = plan.copyDoneOffset + controlGroupBytes;
    if (use35Core) {
        plan.remoteSendDoneOffset = plan.recvCopyDoneOffset + controlGroupBytes;
        plan.readySignalOffset = plan.remoteSendDoneOffset + controlGroupBytes;
        plan.signalBytes = 4ULL * controlGroupBytes;
    } else {
        plan.remoteSendDoneOffset = 0ULL;
        plan.readySignalOffset = plan.recvCopyDoneOffset + controlGroupBytes;
        plan.signalBytes = 3ULL * controlGroupBytes;
    }
    plan.ackSignalOffset = plan.readySignalOffset + controlGroupBytes;
    plan.controlBytes = controlGroupBytes;
    return plan;
}

inline bool AllToAllBigDataValidTopology(
    int rankSize, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    if (rankSize <= 0) {
        return false;
    }
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rankSize, ranksPerNode);
    if (localRanks <= 0) {
        return false;
    }
    if (!AllToAllBigDataIsMultiNode(rankSize, localRanks)) {
        return true;
    }
    return rankSize % localRanks == 0;
}

inline int32_t AllToAllBigDataLocalNodeBegin(
    int rank, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rank + 1, ranksPerNode);
    return (rank / localRanks) * localRanks;
}

inline int32_t AllToAllBigDataLocalNodeEnd(
    int rank, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    return AllToAllBigDataLocalNodeBegin(rank, ranksPerNode) +
        AllToAllBigDataNormalizeRanksPerNode(rank + 1, ranksPerNode);
}

inline int32_t AllToAllBigDataNodeCount(
    int rankSize, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    if (!AllToAllBigDataValidTopology(rankSize, ranksPerNode)) {
        return 0;
    }
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rankSize, ranksPerNode);
    if (!AllToAllBigDataIsMultiNode(rankSize, localRanks)) {
        return 1;
    }
    return rankSize / localRanks;
}

inline bool AllToAllBigDataIsLocalPeer(
    int rank, int peer, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(std::max(rank, peer) + 1, ranksPerNode);
    const int32_t begin = AllToAllBigDataLocalNodeBegin(rank, localRanks);
    return peer >= begin && peer < begin + localRanks;
}

inline int32_t AllToAllBigDataLocalPeerForWorker(
    int rank, uint32_t workerGroup, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rank + 1, ranksPerNode);
    if (workerGroup >= static_cast<uint32_t>(localRanks)) {
        return -1;
    }
    return AllToAllBigDataLocalNodeBegin(rank, localRanks) + static_cast<int32_t>(workerGroup);
}

inline std::vector<int32_t> AllToAllBigDataRemotePeers(
    int rank, int rankSize, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    std::vector<int32_t> peers;
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rankSize, ranksPerNode);
    if (!AllToAllBigDataIsMultiNode(rankSize, localRanks) ||
        !AllToAllBigDataValidTopology(rankSize, localRanks)) {
        return peers;
    }
    peers.reserve(static_cast<size_t>(rankSize - localRanks));
    for (int32_t step = 0; step < rankSize &&
         peers.size() < static_cast<size_t>(rankSize - localRanks); ++step) {
        const int32_t peer = (rank + localRanks + step) % rankSize;
        if (!AllToAllBigDataIsLocalPeer(rank, peer, localRanks)) {
            peers.push_back(peer);
        }
    }
    return peers;
}

inline std::vector<int32_t> AllToAllBigDataLocalPeers(
    int rank, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    std::vector<int32_t> peers;
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rank + 1, ranksPerNode);
    peers.reserve(static_cast<size_t>(std::max<int32_t>(localRanks - 1, 0)));
    const int32_t begin = AllToAllBigDataLocalNodeBegin(rank, localRanks);
    const int32_t local = rank - begin;
    for (int32_t step = 1; step < localRanks; ++step) {
        peers.push_back(begin + ((local + step) % localRanks));
    }
    return peers;
}

inline std::vector<int32_t> AllToAllBigDataMergedPeerTasks(
    int rank, int rankSize, int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    std::vector<int32_t> tasks;
    const int32_t localRanks = AllToAllBigDataNormalizeRanksPerNode(rankSize, ranksPerNode);
    if (!AllToAllBigDataIsMultiNode(rankSize, localRanks) ||
        !AllToAllBigDataValidTopology(rankSize, localRanks)) {
        return tasks;
    }
    const std::vector<int32_t> remotePeers = AllToAllBigDataRemotePeers(rank, rankSize, localRanks);
    const std::vector<int32_t> localPeers = AllToAllBigDataLocalPeers(rank, localRanks);
    const int32_t remoteBurst = AllToAllBigDataNodeCount(rankSize, localRanks) - 1;
    size_t remoteIndex = 0;
    size_t localIndex = 0;
    tasks.reserve(remotePeers.size() + localPeers.size());
    while (remoteIndex < remotePeers.size() || localIndex < localPeers.size()) {
        for (int32_t i = 0; i < remoteBurst && remoteIndex < remotePeers.size(); ++i) {
            tasks.push_back(remotePeers[remoteIndex++]);
        }
        if (localIndex < localPeers.size()) {
            tasks.push_back(localPeers[localIndex++]);
        }
    }
    return tasks;
}

inline uint32_t AllToAllBigDataBlockDim(
    int rankSize, bool force35Core = false, bool remotePutOnly = false,
    int32_t ranksPerNode = kAllToAllBigDataRanksPerNode)
{
    if (rankSize <= 0) {
        return 1U;
    }
    if (AllToAllBigDataUse35Core(rankSize, force35Core, ranksPerNode)) {
        if (remotePutOnly) {
            return kAllToAllBigDataRemotePutOnlyBlockDim;
        }
        return kAllToAllBigDataMultiNodeBlockDim;
    }
    return static_cast<uint32_t>(rankSize) * kAllToAllBigDataCoresPerPeer;
}

inline void FillAllToAllInput(
    std::vector<int32_t>& input, int rank, int rankSize, int32_t elementsPerPeer)
{
    for (int dstRank = 0; dstRank < rankSize; ++dstRank) {
        std::fill(input.begin() + static_cast<size_t>(dstRank) * elementsPerPeer,
                  input.begin() + static_cast<size_t>(dstRank + 1) * elementsPerPeer,
                  AllToAllValue(rank, dstRank));
    }
}

inline bool ValidateAllToAllOutput(
    const std::vector<int32_t>& output, int rank, int rankSize, int32_t elementsPerPeer)
{
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        const int32_t expected = AllToAllValue(srcRank, rank);
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            const size_t offset = static_cast<size_t>(srcRank) * elementsPerPeer + i;
            if (output[offset] != expected) {
                return false;
            }
        }
    }
    return true;
}

inline void BuildAllToAllOutputFromInputs(
    const std::vector<int32_t>& allInputs, int rank, int rankSize, int32_t elementsPerPeer,
    std::vector<int32_t>& output)
{
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        const size_t srcBase = static_cast<size_t>(srcRank) * rankSize * elementsPerPeer +
            static_cast<size_t>(rank) * elementsPerPeer;
        const size_t dstBase = static_cast<size_t>(srcRank) * elementsPerPeer;
        std::copy(allInputs.begin() + srcBase,
                  allInputs.begin() + srcBase + elementsPerPeer,
                  output.begin() + dstBase);
    }
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_LAYOUT_H
