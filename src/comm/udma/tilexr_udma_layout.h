/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_LAYOUT_H
#define TILEXR_UDMA_LAYOUT_H

#include <cstdint>
#include <map>
#include <vector>

#include "tilexr_udma_types.h"

namespace TileXR {

constexpr int TILEXR_UDMA_LAYOUT_SUCCESS = 0;
constexpr int TILEXR_UDMA_LAYOUT_INVALID = -3;

int BuildUDMAInfoImage(
    uintptr_t deviceBase,
    uint32_t qpNum,
    const std::vector<UDMAWQCtx>& sq,
    const std::vector<UDMAWQCtx>& rq,
    const std::vector<UDMACQCtx>& scq,
    const std::vector<UDMACQCtx>& rcq,
    const std::vector<UDMAMemInfo>& mem,
    UDMAInfo& info,
    std::vector<uint8_t>& bytes);

int BuildUDMAInfoImage(
    uintptr_t deviceBase,
    uint32_t qpNum,
    const std::vector<UDMAWQCtx>& sq,
    const std::vector<UDMAWQCtx>& rq,
    const std::vector<UDMACQCtx>& scq,
    const std::vector<UDMACQCtx>& rcq,
    const std::vector<UDMAMemInfo>& mem,
    const std::vector<uint32_t>& qpWeights,
    UDMAInfo& info,
    std::vector<uint8_t>& bytes);

std::vector<uint32_t> BuildUDMAMultiRouteQpToEid(
    const std::vector<uint32_t>& routeEids,
    uint32_t qpsPerRoute);

std::vector<uint32_t> BuildUDMAMultiRouteQpWeights(
    const std::vector<uint32_t>& routeEids,
    const std::map<uint32_t, uint32_t>& routeWeights,
    uint32_t qpsPerRoute);

std::vector<uint32_t> SelectExplicitUDMARouteEids(
    const char* routeList,
    const std::vector<uint32_t>& candidateEids);

} // namespace TileXR

#endif // TILEXR_UDMA_LAYOUT_H
