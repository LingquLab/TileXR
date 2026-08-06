/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_layout.h"

#include <cstring>
#include <limits>

namespace TileXR {
namespace {

bool CheckedMultiply(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(size_t lhs, size_t rhs, size_t& result)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedArrayEnd(size_t offset, size_t count, size_t elementSize, size_t& end)
{
    size_t arrayBytes = 0;
    return CheckedMultiply(count, elementSize, arrayBytes) && CheckedAdd(offset, arrayBytes, end);
}

bool CheckedDeviceAddress(uintptr_t deviceBase, size_t offset, uint64_t& address)
{
    constexpr uint64_t maxAddress = std::numeric_limits<uint64_t>::max();
    if (deviceBase > static_cast<uintptr_t>(maxAddress) || offset > maxAddress) {
        return false;
    }
    const uint64_t base = static_cast<uint64_t>(deviceBase);
    const uint64_t deviceOffset = static_cast<uint64_t>(offset);
    if (deviceOffset > maxAddress - base) {
        return false;
    }
    address = base + deviceOffset;
    return true;
}

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
    if (qpNum == 0 || entryCount == 0 || entryCount % qpNum != 0 ||
        rq.size() != entryCount || scq.size() != entryCount ||
        rcq.size() != entryCount || mem.size() != entryCount) {
        return TILEXR_UDMA_LAYOUT_INVALID;
    }

    const size_t sqOffset = sizeof(UDMAInfo);
    size_t rqOffset = 0;
    size_t scqOffset = 0;
    size_t rcqOffset = 0;
    size_t memOffset = 0;
    size_t totalBytes = 0;
    if (!CheckedArrayEnd(sqOffset, entryCount, sizeof(UDMAWQCtx), rqOffset) ||
        !CheckedArrayEnd(rqOffset, entryCount, sizeof(UDMAWQCtx), scqOffset) ||
        !CheckedArrayEnd(scqOffset, entryCount, sizeof(UDMACQCtx), rcqOffset) ||
        !CheckedArrayEnd(rcqOffset, entryCount, sizeof(UDMACQCtx), memOffset) ||
        !CheckedArrayEnd(memOffset, entryCount, sizeof(UDMAMemInfo), totalBytes) ||
        totalBytes > bytes.max_size()) {
        return TILEXR_UDMA_LAYOUT_INVALID;
    }

    UDMAInfo nextInfo {};
    uint64_t imageEnd = 0;
    nextInfo.qpNum = qpNum;
    if (!CheckedDeviceAddress(deviceBase, sqOffset, nextInfo.sqPtr) ||
        !CheckedDeviceAddress(deviceBase, rqOffset, nextInfo.rqPtr) ||
        !CheckedDeviceAddress(deviceBase, scqOffset, nextInfo.scqPtr) ||
        !CheckedDeviceAddress(deviceBase, rcqOffset, nextInfo.rcqPtr) ||
        !CheckedDeviceAddress(deviceBase, memOffset, nextInfo.memPtr) ||
        !CheckedDeviceAddress(deviceBase, totalBytes, imageEnd)) {
        return TILEXR_UDMA_LAYOUT_INVALID;
    }

    std::vector<uint8_t> nextBytes(totalBytes, 0);
    std::memcpy(nextBytes.data(), &nextInfo, sizeof(nextInfo));
    CopyVector(nextBytes, sqOffset, sq);
    CopyVector(nextBytes, rqOffset, rq);
    CopyVector(nextBytes, scqOffset, scq);
    CopyVector(nextBytes, rcqOffset, rcq);
    CopyVector(nextBytes, memOffset, mem);

    info = nextInfo;
    bytes.swap(nextBytes);
    return TILEXR_UDMA_LAYOUT_SUCCESS;
}

} // namespace TileXR
