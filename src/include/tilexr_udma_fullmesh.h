/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_FULLMESH_H
#define TILEXR_UDMA_FULLMESH_H

#include <cstdint>

#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

namespace TileXR {

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_UDMA_FULLMESH_INLINE \
    __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_UDMA_FULLMESH_INLINE inline
#endif

constexpr uint32_t TILEXR_UDMA_FULLMESH_MAGIC = 0x5458464DU; // TXFM
constexpr uint32_t TILEXR_UDMA_FULLMESH_VERSION = 1U;
constexpr uint32_t TILEXR_UDMA_FULLMESH_SLOT_COUNT = 8U;

struct TileXRUDMAFullmeshDeviceView {
    uint32_t magic = TILEXR_UDMA_FULLMESH_MAGIC;
    uint32_t version = TILEXR_UDMA_FULLMESH_VERSION;
    uint32_t slotCount = TILEXR_UDMA_FULLMESH_SLOT_COUNT;
    uint32_t connectedCount = 0U;
    uint32_t localRank = 0U;
    uint32_t validPeerMask = 0U;
    uint32_t registrationReady = 0U;
    uint32_t reserved = 0U;
    uint64_t registrationGeneration = 0U;
    uint64_t infoPtr = 0U;
};

struct TileXRUDMAFullmeshHostView {
    uint32_t version = TILEXR_UDMA_FULLMESH_VERSION;
    uint32_t slotCount = 0U;
    uint32_t connectedCount = 0U;
    uint32_t localRank = 0U;
    uint32_t validPeerMask = 0U;
    uint32_t registrationReady = 0U;
    uint64_t registrationGeneration = 0U;
    GM_ADDR infoDev = nullptr;
    GM_ADDR viewDev = nullptr;
};

TILEXR_UDMA_FULLMESH_INLINE uint32_t UDMAFullmeshExpectedPeerMask(
    uint32_t localRank, uint32_t localRankSize)
{
    if (localRankSize == 0U ||
        localRankSize > TILEXR_UDMA_FULLMESH_SLOT_COUNT ||
        localRank >= localRankSize) {
        return 0U;
    }
    const uint32_t activeMask = (1U << localRankSize) - 1U;
    return activeMask & ~(1U << localRank);
}

TILEXR_UDMA_FULLMESH_INLINE bool UDMAFullmeshHostViewValid(
    const TileXRUDMAFullmeshHostView& view, uint32_t localRank,
    uint32_t localRankSize, uint64_t registrationGeneration)
{
    return view.version == TILEXR_UDMA_FULLMESH_VERSION &&
        view.slotCount == TILEXR_UDMA_FULLMESH_SLOT_COUNT &&
        view.localRank == localRank &&
        view.connectedCount + 1U == localRankSize &&
        view.validPeerMask ==
            UDMAFullmeshExpectedPeerMask(localRank, localRankSize) &&
        view.registrationReady != 0U &&
        view.registrationGeneration != 0U &&
        view.registrationGeneration == registrationGeneration &&
        view.infoDev != nullptr && view.viewDev != nullptr;
}

} // namespace TileXR

#undef TILEXR_UDMA_FULLMESH_INLINE

#endif // TILEXR_UDMA_FULLMESH_H
