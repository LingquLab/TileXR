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
