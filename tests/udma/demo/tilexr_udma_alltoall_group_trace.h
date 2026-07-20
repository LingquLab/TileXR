/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_GROUP_TRACE_H
#define TILEXR_UDMA_ALLTOALL_GROUP_TRACE_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace TileXR {
namespace Demo {

constexpr uint32_t kAllToAllGroupTraceMagic = 0x47545243U; // "GTRC"
constexpr uint32_t kAllToAllGroupTraceVersion = 1U;
constexpr size_t kAllToAllGroupTraceBytes = 8ULL * 1024ULL * 1024ULL;
constexpr size_t kAllToAllGroupTraceHeaderBytes = 4096ULL;
constexpr uint32_t kAllToAllGroupTraceMaxIterations = 50U;
constexpr uint32_t kAllToAllGroupTraceCoreCount = 32U;
constexpr uint32_t kAllToAllGroupTracePhaseCount = 5U;
constexpr uint32_t kAllToAllGroupTraceNoQp = 0xFFFFFFFFU;
constexpr uint64_t kAllToAllGroupTraceCyclesPerUs = 1000ULL;

enum AllToAllGroupTracePhase : uint32_t {
    kAllToAllGroupTraceSelfCopy = 0U,
    kAllToAllGroupTraceSendPutSignal = 1U,
    kAllToAllGroupTraceSendQuiet = 2U,
    kAllToAllGroupTraceReceiveWait = 3U,
    kAllToAllGroupTraceReceiveCopy = 4U,
};

struct AllToAllGroupTraceSpan {
    uint64_t beginCycle;
    uint64_t endCycle;
};

struct AllToAllGroupTraceTaskSpan {
    uint64_t beginCycle;
    uint64_t endCycle;
    int32_t peer;
    uint32_t qpIdx;
};

struct AllToAllGroupTraceHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t rank;
    uint32_t iterationCount;
    uint32_t groupCount;
    uint32_t passCount;
    uint32_t coreCount;
    uint32_t phaseCount;
    uint64_t cyclesPerUs;
    uint64_t traceBytes;
    uint64_t kernelSpanOffset;
    uint64_t taskSpanOffset;
};

constexpr size_t AllToAllGroupTraceKernelSpanOffset(uint32_t iteration, uint32_t core)
{
    return kAllToAllGroupTraceHeaderBytes +
        (static_cast<size_t>(iteration) * kAllToAllGroupTraceCoreCount + core) *
        sizeof(AllToAllGroupTraceSpan);
}

constexpr size_t AllToAllGroupTraceTaskSpanBaseOffset()
{
    return kAllToAllGroupTraceHeaderBytes +
        static_cast<size_t>(kAllToAllGroupTraceMaxIterations) *
        kAllToAllGroupTraceCoreCount * sizeof(AllToAllGroupTraceSpan);
}

inline bool AllToAllGroupTraceCheckedMultiply(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0U && rhs > std::numeric_limits<size_t>::max() / lhs) {
        result = std::numeric_limits<size_t>::max();
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline size_t AllToAllGroupTraceLayoutBytes(
    uint32_t iterationCount, uint32_t groupCount, uint32_t passCount)
{
    size_t count = iterationCount;
    size_t next = 0U;
    const size_t factors[] = {
        kAllToAllGroupTraceCoreCount, groupCount, passCount,
        kAllToAllGroupTracePhaseCount, sizeof(AllToAllGroupTraceTaskSpan)};
    for (size_t factor : factors) {
        if (!AllToAllGroupTraceCheckedMultiply(count, factor, next)) {
            return std::numeric_limits<size_t>::max();
        }
        count = next;
    }
    if (count > std::numeric_limits<size_t>::max() -
            AllToAllGroupTraceTaskSpanBaseOffset()) {
        return std::numeric_limits<size_t>::max();
    }
    return AllToAllGroupTraceTaskSpanBaseOffset() + count;
}

inline bool AllToAllGroupTraceLayoutFits(
    uint32_t iterationCount, uint32_t groupCount, uint32_t passCount)
{
    return iterationCount > 0U &&
        iterationCount <= kAllToAllGroupTraceMaxIterations &&
        groupCount > 0U && passCount > 0U &&
        AllToAllGroupTraceLayoutBytes(iterationCount, groupCount, passCount) <=
            kAllToAllGroupTraceBytes;
}

inline size_t AllToAllGroupTraceTaskSpanOffset(
    uint32_t iteration, uint32_t core, uint32_t group, uint32_t pass,
    uint32_t phase, uint32_t groupCount, uint32_t passCount)
{
    const size_t index =
        (((((static_cast<size_t>(iteration) * kAllToAllGroupTraceCoreCount + core) *
        groupCount + group) * passCount + pass) *
        kAllToAllGroupTracePhaseCount) + phase);
    return AllToAllGroupTraceTaskSpanBaseOffset() +
        index * sizeof(AllToAllGroupTraceTaskSpan);
}

static_assert(sizeof(AllToAllGroupTraceSpan) == 16U,
    "group trace kernel span must contain two uint64 timestamps");
static_assert(sizeof(AllToAllGroupTraceTaskSpan) == 24U,
    "group trace task span layout changed");
static_assert(sizeof(AllToAllGroupTraceHeader) <= kAllToAllGroupTraceHeaderBytes,
    "group trace header must fit its region");
static_assert(AllToAllGroupTraceTaskSpanBaseOffset() < kAllToAllGroupTraceBytes,
    "group trace kernel spans must fit in 8 MiB");

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_GROUP_TRACE_H
