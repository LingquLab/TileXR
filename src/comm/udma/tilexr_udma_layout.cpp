/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_layout.h"

#include <cstring>

namespace TileXR {
namespace {

constexpr uint32_t TILEXR_UDMA_QP_NUM = 1;

template <typename T>
void CopyVector(std::vector<uint8_t>& dst, size_t offset, const std::vector<T>& src)
{
    if (!src.empty()) {
        std::memcpy(dst.data() + offset, src.data(), src.size() * sizeof(T));
    }
}

} // namespace

int BuildUDMAInfoImage(
    uintptr_t deviceBase,
    const std::vector<UDMAWQCtx>& sq,
    const std::vector<UDMAWQCtx>& rq,
    const std::vector<UDMACQCtx>& scq,
    const std::vector<UDMACQCtx>& rcq,
    const std::vector<UDMAMemInfo>& mem,
    UDMAInfo& info,
    std::vector<uint8_t>& bytes)
{
    const size_t rankCount = sq.size();
    if (rankCount == 0 || rq.size() != rankCount || scq.size() != rankCount ||
        rcq.size() != rankCount || mem.size() != rankCount) {
        return TILEXR_UDMA_LAYOUT_INVALID;
    }

    const size_t sqOffset = sizeof(UDMAInfo);
    const size_t rqOffset = sqOffset + sq.size() * sizeof(UDMAWQCtx);
    const size_t scqOffset = rqOffset + rq.size() * sizeof(UDMAWQCtx);
    const size_t rcqOffset = scqOffset + scq.size() * sizeof(UDMACQCtx);
    const size_t memOffset = rcqOffset + rcq.size() * sizeof(UDMACQCtx);
    const size_t totalBytes = memOffset + mem.size() * sizeof(UDMAMemInfo);

    info = {};
    info.qpNum = TILEXR_UDMA_QP_NUM;
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

} // namespace TileXR
