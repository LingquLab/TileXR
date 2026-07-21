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
};

inline bool AllToAllGroupValidRouteStage(uint32_t value)
{
    return value <= static_cast<uint32_t>(AllToAllGroupRouteStage::kLocalCopy);
}

inline bool AllToAllGroupStageRunsSend(AllToAllGroupRouteStage stage)
{
    return stage != AllToAllGroupRouteStage::kLocalCopy;
}

inline bool AllToAllGroupStageRunsCopy(AllToAllGroupRouteStage stage)
{
    return stage != AllToAllGroupRouteStage::kLocalSend;
}

inline bool AllToAllGroupStageWaitsForSignal(AllToAllGroupRouteStage stage)
{
    return AllToAllGroupStageRunsCopy(stage) &&
        stage != AllToAllGroupRouteStage::kLocalCopy;
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

inline bool AllToAllGroupPeerInRouteStage(
    int rank, int peer, AllToAllGroupRouteStage stage)
{
    if (rank < 0 || peer < 0 || rank == peer) {
        return false;
    }
    const bool crossNode = AllToAllGroupIsCrossNode(rank, peer);
    switch (stage) {
        case AllToAllGroupRouteStage::kCombined:
            return true;
        case AllToAllGroupRouteStage::kLocal:
        case AllToAllGroupRouteStage::kLocalSend:
        case AllToAllGroupRouteStage::kLocalCopy:
            return !crossNode;
        case AllToAllGroupRouteStage::kPrimary:
            return crossNode && !AllToAllGroupUseSecondaryRoute(rank, peer);
        case AllToAllGroupRouteStage::kSecondary:
            return crossNode && AllToAllGroupUseSecondaryRoute(rank, peer);
    }
    return false;
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
