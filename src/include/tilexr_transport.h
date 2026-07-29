/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_TRANSPORT_H
#define TILEXR_TRANSPORT_H

#include <cstdint>

#include "comm_args.h"

namespace TileXR {

#if TILEXR_ASCENDC_AICORE_COMPILE
#define TILEXR_TRANSPORT_INLINE __aicore__ inline
#define TILEXR_TRANSPORT_GM __gm__
#else
#define TILEXR_TRANSPORT_INLINE inline
#define TILEXR_TRANSPORT_GM
#endif

enum class TileXRTransportKind : uint8_t {
    MEMORY = 0,
    DIRECT_URMA = 1,
};

constexpr uint64_t TILEXR_AUTO_SAME_NODE_DIRECT_URMA_THRESHOLD_BYTES = 4ULL * 1024ULL * 1024ULL;
constexpr uint64_t TILEXR_AUTO_CROSS_NODE_DIRECT_URMA_THRESHOLD_BYTES = 128ULL * 1024ULL;
constexpr uint64_t TILEXR_AUTO_DIRECT_URMA_THRESHOLD_BYTES =
    TILEXR_AUTO_SAME_NODE_DIRECT_URMA_THRESHOLD_BYTES;

TILEXR_TRANSPORT_INLINE bool TileXRDirectUrmaCapable(const TILEXR_TRANSPORT_GM CommArgs* args)
{
    return args != nullptr &&
           ((args->extraFlag & ExtraFlag::UDMA) != 0) &&
           args->udmaInfoPtr != nullptr;
}

TILEXR_TRANSPORT_INLINE bool TileXRDirectUrmaAvailable(const TILEXR_TRANSPORT_GM CommArgs* args)
{
    return TileXRDirectUrmaCapable(args) && args->udmaRegistryPtr != nullptr;
}

TILEXR_TRANSPORT_INLINE bool TileXRDirectUrmaPeerRoutable(
    const TILEXR_TRANSPORT_GM CommArgs* args, int targetRank)
{
    if (!TileXRDirectUrmaCapable(args) || args->rankSize <= 1 || args->rank < 0 ||
        args->rank >= args->rankSize || targetRank < 0 || targetRank >= args->rankSize ||
        targetRank == args->rank || args->localRankSize <= 0) {
        return false;
    }
    if (args->localRankSize >= args->rankSize) {
        return true;
    }
    return targetRank / args->localRankSize != args->rank / args->localRankSize;
}

TILEXR_TRANSPORT_INLINE bool TileXRCommSpansNodes(const TILEXR_TRANSPORT_GM CommArgs* args)
{
    return args != nullptr && args->localRankSize > 0 && args->localRankSize < args->rankSize;
}

TILEXR_TRANSPORT_INLINE TileXRTransportKind TileXRSelectAutoTransport(
    const TILEXR_TRANSPORT_GM CommArgs* args, uint64_t bytes)
{
    if (bytes == 0) {
        return TileXRTransportKind::MEMORY;
    }
    if (!TileXRDirectUrmaAvailable(args)) {
        return TileXRTransportKind::MEMORY;
    }
    const uint64_t threshold = TileXRCommSpansNodes(args) ?
        TILEXR_AUTO_CROSS_NODE_DIRECT_URMA_THRESHOLD_BYTES :
        TILEXR_AUTO_SAME_NODE_DIRECT_URMA_THRESHOLD_BYTES;
    if (bytes >= threshold) {
        return TileXRTransportKind::DIRECT_URMA;
    }
    return TileXRTransportKind::MEMORY;
}

#undef TILEXR_TRANSPORT_INLINE
#undef TILEXR_TRANSPORT_GM

} // namespace TileXR

#endif // TILEXR_TRANSPORT_H
