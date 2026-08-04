/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_layout.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace TileXR {
namespace {

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
    uint32_t qpNum,
    const std::vector<UDMAWQCtx>& sq,
    const std::vector<UDMAWQCtx>& rq,
    const std::vector<UDMACQCtx>& scq,
    const std::vector<UDMACQCtx>& rcq,
    const std::vector<UDMAMemInfo>& mem,
    UDMAInfo& info,
    std::vector<uint8_t>& bytes)
{
    const size_t entryCount = sq.size();
    if (!IsSupportedUDMAQpNum(qpNum) || entryCount == 0 || entryCount % qpNum != 0 ||
        rq.size() != entryCount || scq.size() != entryCount ||
        rcq.size() != entryCount || mem.size() != entryCount) {
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

bool IsSupportedUDMAQpNum(uint32_t qpNum)
{
    return qpNum == 1 || qpNum == 2 || qpNum == 4 || qpNum == TILEXR_UDMA_MAX_QP_NUM;
}

int ParseUDMAQpNum(const char* value, uint32_t& qpNum)
{
    if (value == nullptr || value[0] == '\0') {
        qpNum = TILEXR_UDMA_DEFAULT_QP_NUM;
        return TILEXR_UDMA_LAYOUT_SUCCESS;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX ||
        !IsSupportedUDMAQpNum(static_cast<uint32_t>(parsed))) {
        return TILEXR_UDMA_LAYOUT_INVALID;
    }
    qpNum = static_cast<uint32_t>(parsed);
    return TILEXR_UDMA_LAYOUT_SUCCESS;
}

} // namespace TileXR
