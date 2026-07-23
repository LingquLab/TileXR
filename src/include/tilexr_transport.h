/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_TRANSPORT_H
#define TILEXR_TRANSPORT_H

#include <cstdint>

#include "comm_args.h"
#if TILEXR_ASCENDC_AICORE_COMPILE
#include "tilexr_udma.h"
#endif

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

constexpr uint64_t TILEXR_AUTO_DIRECT_URMA_THRESHOLD_BYTES = 4ULL * 1024ULL * 1024ULL;

TILEXR_TRANSPORT_INLINE bool TileXRDirectUrmaAvailable(const TILEXR_TRANSPORT_GM CommArgs* args)
{
    return args != nullptr &&
           ((args->extraFlag & ExtraFlag::UDMA) != 0) &&
           args->udmaInfoPtr != nullptr &&
           args->udmaRegistryPtr != nullptr;
}

TILEXR_TRANSPORT_INLINE TileXRTransportKind TileXRSelectAutoTransport(
    const TILEXR_TRANSPORT_GM CommArgs* args, uint64_t bytes)
{
    if (!TileXRDirectUrmaAvailable(args)) {
        return TileXRTransportKind::MEMORY;
    }
    if (bytes >= TILEXR_AUTO_DIRECT_URMA_THRESHOLD_BYTES) {
        return TileXRTransportKind::DIRECT_URMA;
    }
    return TileXRTransportKind::MEMORY;
}

template <typename T>
TILEXR_TRANSPORT_INLINE TileXRTransportKind TileXRPutAutoNbi(
    const TILEXR_TRANSPORT_GM CommArgs* args,
    int targetRank,
    const TILEXR_TRANSPORT_GM T* localSrc,
    uint64_t remoteOffset,
    uint32_t bytes)
{
    TileXRTransportKind route = TileXRSelectAutoTransport(args, bytes);
#if TILEXR_ASCENDC_AICORE_COMPILE
    if (route == TileXRTransportKind::DIRECT_URMA) {
        UDMAPutNbi<T>(args, targetRank, localSrc, remoteOffset, bytes);
    }
#else
    (void)targetRank;
    (void)localSrc;
    (void)remoteOffset;
#endif
    return route;
}

template <typename T>
TILEXR_TRANSPORT_INLINE TileXRTransportKind TileXRGetAutoNbi(
    const TILEXR_TRANSPORT_GM CommArgs* args,
    int sourceRank,
    TILEXR_TRANSPORT_GM T* localDst,
    uint64_t remoteOffset,
    uint32_t bytes)
{
    TileXRTransportKind route = TileXRSelectAutoTransport(args, bytes);
#if TILEXR_ASCENDC_AICORE_COMPILE
    if (route == TileXRTransportKind::DIRECT_URMA) {
        UDMAGetNbi<T>(args, sourceRank, localDst, remoteOffset, bytes);
    }
#else
    (void)sourceRank;
    (void)localDst;
    (void)remoteOffset;
#endif
    return route;
}

#undef TILEXR_TRANSPORT_INLINE
#undef TILEXR_TRANSPORT_GM

} // namespace TileXR

#endif // TILEXR_TRANSPORT_H
