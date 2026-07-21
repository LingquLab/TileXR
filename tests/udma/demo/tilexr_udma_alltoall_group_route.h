/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_GROUP_ROUTE_H
#define TILEXR_UDMA_ALLTOALL_GROUP_ROUTE_H

#include <cstdint>

namespace TileXR {
namespace Demo {

constexpr uint32_t kAllToAllGroupRanksPerNode = 8U;
constexpr uint32_t kAllToAllGroupPrimaryPeersPerNode = 6U;

enum class AllToAllGroupRouteStage : uint32_t {
    kCombined = 0U,
    kLocal = 1U,
    kPrimary = 2U,
    kSecondary = 3U,
    kLocalSend = 4U,
    kLocalCopy = 5U,
    kRemoteSend = 6U,
    kAllSend = 7U,
    kRemoteWait = 8U,
    kRemoteCopy = 9U,
    kNoCopy = 10U,
};

inline bool AllToAllGroupValidRouteStage(uint32_t value)
{
    return value <= static_cast<uint32_t>(AllToAllGroupRouteStage::kNoCopy);
}

inline bool AllToAllGroupStageRunsSend(AllToAllGroupRouteStage stage)
{
    return stage != AllToAllGroupRouteStage::kLocalCopy;
}

inline bool AllToAllGroupStageRunsReceive(AllToAllGroupRouteStage stage)
{
    return stage != AllToAllGroupRouteStage::kLocalSend &&
        stage != AllToAllGroupRouteStage::kRemoteSend &&
        stage != AllToAllGroupRouteStage::kAllSend;
}

inline bool AllToAllGroupStageRunsCopy(AllToAllGroupRouteStage stage)
{
    return AllToAllGroupStageRunsReceive(stage) &&
        stage != AllToAllGroupRouteStage::kRemoteWait &&
        stage != AllToAllGroupRouteStage::kNoCopy;
}

inline bool AllToAllGroupStageWaitsForSignal(AllToAllGroupRouteStage stage)
{
    return AllToAllGroupStageRunsReceive(stage) &&
        stage != AllToAllGroupRouteStage::kLocalCopy &&
        stage != AllToAllGroupRouteStage::kRemoteCopy;
}

struct AllToAllGroupRouteQps {
    uint32_t primaryQp = 0U;
    uint32_t secondaryQp = 0U;
};

inline bool AllToAllGroupIsCrossNode(int rank, int peer)
{
    return rank >= 0 && peer >= 0 &&
        rank / static_cast<int>(kAllToAllGroupRanksPerNode) !=
        peer / static_cast<int>(kAllToAllGroupRanksPerNode);
}

inline bool AllToAllGroupUseSecondaryRoute(int rank, int peer)
{
    if (!AllToAllGroupIsCrossNode(rank, peer)) {
        return false;
    }
    const uint32_t sourceLocal =
        static_cast<uint32_t>(rank) % kAllToAllGroupRanksPerNode;
    const uint32_t targetLocal =
        static_cast<uint32_t>(peer) % kAllToAllGroupRanksPerNode;
    return (sourceLocal + targetLocal) % kAllToAllGroupRanksPerNode >=
        kAllToAllGroupPrimaryPeersPerNode;
}

inline bool AllToAllGroupUseSecondaryRoute(
    int rank, int peer, bool useSecondaryRoute)
{
    return useSecondaryRoute && AllToAllGroupUseSecondaryRoute(rank, peer);
}

inline bool AllToAllGroupPeerInRouteStage(
    int rank, int peer, AllToAllGroupRouteStage stage)
{
    if (rank < 0 || peer < 0 || rank == peer) {
        return false;
    }
    const bool crossNode = AllToAllGroupIsCrossNode(rank, peer);
    switch (stage) {
        case AllToAllGroupRouteStage::kCombined:
        case AllToAllGroupRouteStage::kAllSend:
        case AllToAllGroupRouteStage::kNoCopy:
            return true;
        case AllToAllGroupRouteStage::kLocal:
        case AllToAllGroupRouteStage::kLocalSend:
        case AllToAllGroupRouteStage::kLocalCopy:
            return !crossNode;
        case AllToAllGroupRouteStage::kRemoteSend:
        case AllToAllGroupRouteStage::kRemoteWait:
        case AllToAllGroupRouteStage::kRemoteCopy:
            return crossNode;
        case AllToAllGroupRouteStage::kPrimary:
            return crossNode && !AllToAllGroupUseSecondaryRoute(rank, peer);
        case AllToAllGroupRouteStage::kSecondary:
            return crossNode && AllToAllGroupUseSecondaryRoute(rank, peer);
    }
    return false;
}

inline bool AllToAllGroupReceivePeerInRouteStage(
    int rank, int peer, AllToAllGroupRouteStage stage)
{
    if (stage == AllToAllGroupRouteStage::kRemoteCopy) {
        return rank >= 0 && peer >= 0 && rank != peer;
    }
    return AllToAllGroupPeerInRouteStage(rank, peer, stage);
}

inline AllToAllGroupRouteQps AllToAllGroupSelectRouteQps(
    const uint32_t* weights, uint32_t qpCount)
{
    AllToAllGroupRouteQps result {};
    if (weights == nullptr || qpCount == 0U) {
        return result;
    }

    uint32_t primaryWeight = weights[0];
    for (uint32_t qp = 1U; qp < qpCount; ++qp) {
        if (weights[qp] > primaryWeight) {
            result.primaryQp = qp;
            primaryWeight = weights[qp];
        }
    }

    uint32_t secondaryWeight = 0U;
    for (uint32_t qp = 0U; qp < qpCount; ++qp) {
        const uint32_t weight = weights[qp];
        if (weight != 0U && weight < primaryWeight && weight > secondaryWeight) {
            result.secondaryQp = qp;
            secondaryWeight = weight;
        }
    }
    if (secondaryWeight == 0U) {
        result.secondaryQp = result.primaryQp;
    }
    return result;
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_GROUP_ROUTE_H
