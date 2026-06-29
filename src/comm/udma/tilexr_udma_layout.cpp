/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_layout.h"

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
    if (qpNum == 0 || sq.empty() || sq.size() % qpNum != 0 ||
        rq.size() != sq.size() || scq.size() != sq.size() ||
        rcq.size() != sq.size() || mem.size() != sq.size()) {
        return TILEXR_UDMA_LAYOUT_INVALID;
    }

    const size_t sqOffset = sizeof(UDMAInfo);
    const size_t rqOffset = sqOffset + sq.size() * sizeof(UDMAWQCtx);
    const size_t scqOffset = rqOffset + rq.size() * sizeof(UDMAWQCtx);
    const size_t rcqOffset = scqOffset + scq.size() * sizeof(UDMACQCtx);
    const size_t memOffset = rcqOffset + rcq.size() * sizeof(UDMACQCtx);
    const size_t totalBytes = memOffset + mem.size() * sizeof(UDMAMemInfo);

    info = {};
    info.qpNum = qpNum;
    info.sqPtr = deviceBase + sqOffset;
    info.rqPtr = deviceBase + rqOffset;
    info.scqPtr = deviceBase + scqOffset;
    info.rcqPtr = deviceBase + rcqOffset;
    info.memPtr = deviceBase + memOffset;

    bytes.assign(totalBytes, 0);
    std::memcpy(bytes.data(), &info, sizeof(info));
    CopyVector(bytes, sqOffset, sq);
    CopyVector(bytes, rqOffset, rq);
    CopyVector(bytes, scqOffset, scq);
    CopyVector(bytes, rcqOffset, rcq);
    CopyVector(bytes, memOffset, mem);
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

} // namespace TileXR
