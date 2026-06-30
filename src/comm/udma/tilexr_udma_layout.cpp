/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_layout.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace TileXR {
template <typename T>
void CopyVector(std::vector<uint8_t>& dst, size_t offset, const std::vector<T>& src)
{
    if (!src.empty()) {
        std::memcpy(dst.data() + offset, src.data(), src.size() * sizeof(T));
    }
}

int BuildUDMAInfoImage(
    uintptr_t deviceBase,
    uint32_t qpNum,
    const std::vector<UDMAWQCtx>& sq,
    const std::vector<UDMAWQCtx>& rq,
    const std::vector<UDMACQCtx>& scq,
    const std::vector<UDMACQCtx>& rcq,
    const std::vector<UDMAMemInfo>& mem,
    UDMAInfo& info,
    std::vector<uint8_t>& bytes)
{
    return BuildUDMAInfoImage(
        deviceBase, qpNum, sq, rq, scq, rcq, mem, std::vector<uint32_t>(mem.size(), 1), info, bytes);
}

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
    std::vector<uint8_t>& bytes)
{
    if (qpNum == 0 || sq.empty() || sq.size() % qpNum != 0 ||
        rq.size() != sq.size() || scq.size() != sq.size() ||
        rcq.size() != sq.size() || mem.size() != sq.size() ||
        qpWeights.size() != sq.size()) {
        return TILEXR_UDMA_LAYOUT_INVALID;
    }

    const size_t sqOffset = sizeof(UDMAInfo);
    const size_t rqOffset = sqOffset + sq.size() * sizeof(UDMAWQCtx);
    const size_t scqOffset = rqOffset + rq.size() * sizeof(UDMAWQCtx);
    const size_t rcqOffset = scqOffset + scq.size() * sizeof(UDMACQCtx);
    const size_t memOffset = rcqOffset + rcq.size() * sizeof(UDMACQCtx);
    const size_t qpWeightOffset = memOffset + mem.size() * sizeof(UDMAMemInfo);
    const size_t totalBytes = qpWeightOffset + qpWeights.size() * sizeof(uint32_t);

    info = {};
    info.qpNum = qpNum;
    info.sqPtr = deviceBase + sqOffset;
    info.rqPtr = deviceBase + rqOffset;
    info.scqPtr = deviceBase + scqOffset;
    info.rcqPtr = deviceBase + rcqOffset;
    info.memPtr = deviceBase + memOffset;
    info.qpWeightPtr = deviceBase + qpWeightOffset;

    bytes.assign(totalBytes, 0);
    std::memcpy(bytes.data(), &info, sizeof(info));
    CopyVector(bytes, sqOffset, sq);
    CopyVector(bytes, rqOffset, rq);
    CopyVector(bytes, scqOffset, scq);
    CopyVector(bytes, rcqOffset, rcq);
    CopyVector(bytes, memOffset, mem);
    CopyVector(bytes, qpWeightOffset, qpWeights);
    return TILEXR_UDMA_LAYOUT_SUCCESS;
}

std::vector<uint32_t> BuildUDMAMultiRouteQpToEid(
    const std::vector<uint32_t>& routeEids,
    uint32_t qpsPerRoute)
{
    std::vector<uint32_t> qpToEid;
    if (routeEids.empty() || qpsPerRoute == 0) {
        return qpToEid;
    }
    qpToEid.reserve(routeEids.size() * qpsPerRoute);
    for (uint32_t eid : routeEids) {
        for (uint32_t qp = 0; qp < qpsPerRoute; ++qp) {
            qpToEid.push_back(eid);
        }
    }
    return qpToEid;
}

std::vector<uint32_t> BuildUDMAMultiRouteQpWeights(
    const std::vector<uint32_t>& routeEids,
    const std::map<uint32_t, uint32_t>& routeWeights,
    uint32_t qpsPerRoute)
{
    std::vector<uint32_t> qpWeights;
    if (routeEids.empty() || qpsPerRoute == 0) {
        return qpWeights;
    }
    qpWeights.reserve(routeEids.size() * qpsPerRoute);
    for (uint32_t eid : routeEids) {
        uint32_t weight = 1;
        const auto weightIt = routeWeights.find(eid);
        if (weightIt != routeWeights.end() && weightIt->second != 0) {
            weight = weightIt->second;
        }
        for (uint32_t qp = 0; qp < qpsPerRoute; ++qp) {
            qpWeights.push_back(weight);
        }
    }
    return qpWeights;
}

std::vector<uint32_t> SelectExplicitUDMARouteEids(
    const char* routeList,
    const std::vector<uint32_t>& candidateEids)
{
    std::vector<uint32_t> selected;
    if (routeList == nullptr || routeList[0] == '\0' || candidateEids.empty()) {
        return selected;
    }

    const char* cursor = routeList;
    while (*cursor != '\0') {
        char* end = nullptr;
        unsigned long parsed = std::strtoul(cursor, &end, 0);
        if (end != cursor && parsed <= UINT32_MAX) {
            const uint32_t eid = static_cast<uint32_t>(parsed);
            if (std::find(candidateEids.begin(), candidateEids.end(), eid) != candidateEids.end() &&
                std::find(selected.begin(), selected.end(), eid) == selected.end()) {
                selected.push_back(eid);
            }
            cursor = end;
        }
        while (*cursor != '\0' && *cursor != ',') {
            ++cursor;
        }
        if (*cursor == ',') {
            ++cursor;
        }
    }
    return selected;
}

std::vector<uint32_t> SelectUDMARoutesForPeer(
    bool peerIsRemoteNode,
    const std::vector<uint32_t>& topoRoutes,
    const std::vector<uint32_t>& aggregateRoutes)
{
    if (peerIsRemoteNode && !aggregateRoutes.empty()) {
        return aggregateRoutes;
    }
    if (!topoRoutes.empty()) {
        return topoRoutes;
    }
    return aggregateRoutes;
}

} // namespace TileXR
