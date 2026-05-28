/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_REG_H
#define TILEXR_UDMA_REG_H

#include <cstddef>
#include <cstdint>

#include "comm_args.h"

namespace TileXR {

constexpr uint32_t TILEXR_UDMA_REGISTRY_MAGIC = 0x54585255U; // TXRU
constexpr uint32_t TILEXR_UDMA_REGISTRY_VERSION = 1U;
constexpr uint32_t TILEXR_UDMA_MAX_REGIONS = 1U;

struct TileXRUDMARegionDesc {
    GM_ADDR base = nullptr;
    uint64_t bytes = 0;
};

struct TileXRUDMARegistry {
    uint32_t magic = TILEXR_UDMA_REGISTRY_MAGIC;
    uint32_t version = TILEXR_UDMA_REGISTRY_VERSION;
    uint32_t rankSize = 0;
    uint32_t regionCount = 0;
    TileXRUDMARegionDesc regions[TILEXR_MAX_RANK_SIZE] = {};
};

inline bool UDMARegistryValid(const TileXRUDMARegistry *registry, int expectedRankSize)
{
    return registry != nullptr &&
           registry->magic == TILEXR_UDMA_REGISTRY_MAGIC &&
           registry->version == TILEXR_UDMA_REGISTRY_VERSION &&
           expectedRankSize >= 0 &&
           registry->rankSize == static_cast<uint32_t>(expectedRankSize) &&
           registry->rankSize <= TILEXR_MAX_RANK_SIZE &&
           registry->regionCount > 0 &&
           registry->regionCount <= TILEXR_UDMA_MAX_REGIONS;
}

inline bool UDMARegionContains(const TileXRUDMARegistry *registry, int rank, uint64_t byteOffset, uint64_t byteCount)
{
    if (!UDMARegistryValid(registry, static_cast<int>(registry == nullptr ? 0 : registry->rankSize))) {
        return false;
    }
    if (rank < 0 || static_cast<uint32_t>(rank) >= registry->rankSize) {
        return false;
    }
    const auto &region = registry->regions[rank];
    if (region.base == nullptr || byteOffset > region.bytes) {
        return false;
    }
    return byteCount <= region.bytes - byteOffset;
}

inline GM_ADDR UDMARemoteAddr(const TileXRUDMARegistry *registry, int rank, uint64_t byteOffset)
{
    if (registry == nullptr || rank < 0 || static_cast<uint32_t>(rank) >= registry->rankSize) {
        return nullptr;
    }
    return registry->regions[rank].base + byteOffset;
}

} // namespace TileXR

#endif // TILEXR_UDMA_REG_H
