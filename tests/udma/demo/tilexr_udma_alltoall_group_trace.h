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
constexpr uint32_t kAllToAllGroupTraceVersion = 5U;
constexpr size_t kAllToAllGroupTraceBytes = 128ULL * 1024ULL * 1024ULL;
constexpr size_t kAllToAllGroupTraceHeaderBytes = 4096ULL;
constexpr uint32_t kAllToAllGroupTraceMaxIterations = 50U;
constexpr uint32_t kAllToAllGroupTraceCoreCount = 64U;
constexpr uint32_t kAllToAllGroupTracePhaseCount = 12U;
constexpr uint32_t kAllToAllGroupTraceNoQp = 0xFFFFFFFFU;
constexpr uint64_t kAllToAllGroupTraceCyclesPerUs = 1000ULL;
constexpr size_t kAllToAllGroupTraceCacheLineBytes = 128U;

enum AllToAllGroupTracePhase : uint32_t {
    kAllToAllGroupTraceSelfCopy = 0U,
    kAllToAllGroupTraceSendPutSignal = 1U,
    kAllToAllGroupTraceSendQuiet = 2U,
    kAllToAllGroupTraceReceiveWait = 3U,
    kAllToAllGroupTraceReceiveCopy = 4U,
    kAllToAllGroupTraceCreditWait = 5U,
    kAllToAllGroupTraceSdmaSubmit = 6U,
    kAllToAllGroupTraceSdmaWait = 7U,
    kAllToAllGroupTraceSdmaPrepare = 8U,
    kAllToAllGroupTraceSdmaCacheClean = 9U,
    kAllToAllGroupTraceSdmaDsb = 10U,
    kAllToAllGroupTraceSdmaDoorbell = 11U,
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
    uint32_t sdmaHead;
    uint32_t sdmaTail;
    uint32_t sdmaNewTail;
    uint32_t sdmaDepth;
    uint32_t sdmaGeneration;
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
    uint32_t activeCoreCount;
    uint32_t sendWorkerCount;
    uint32_t simtThreadCount;
    uint32_t reserved;
};

constexpr uint32_t AllToAllGroupTraceSimtStorageCore(uint32_t worker)
{
    return worker == 0U ? 0U : 32U + worker;
}

constexpr size_t AllToAllGroupTraceKernelSpanOffset(uint32_t iteration, uint32_t core)
{
    return kAllToAllGroupTraceHeaderBytes +
        (static_cast<size_t>(iteration) * kAllToAllGroupTraceCoreCount + core) *
        kAllToAllGroupTraceCacheLineBytes;
}

constexpr size_t AllToAllGroupTraceTaskSpanBaseOffset()
{
    return kAllToAllGroupTraceHeaderBytes +
        static_cast<size_t>(kAllToAllGroupTraceMaxIterations) *
        kAllToAllGroupTraceCoreCount * kAllToAllGroupTraceCacheLineBytes;
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
    size_t coreBytes = groupCount;
    size_t next = 0U;
    const size_t factors[] = {
        passCount, kAllToAllGroupTracePhaseCount, sizeof(AllToAllGroupTraceTaskSpan)};
    for (size_t factor : factors) {
        if (!AllToAllGroupTraceCheckedMultiply(coreBytes, factor, next)) {
            return std::numeric_limits<size_t>::max();
        }
        coreBytes = next;
    }
    if (coreBytes > std::numeric_limits<size_t>::max() -
            (kAllToAllGroupTraceCacheLineBytes - 1U)) {
        return std::numeric_limits<size_t>::max();
    }
    coreBytes = (coreBytes + kAllToAllGroupTraceCacheLineBytes - 1U) &
        ~(kAllToAllGroupTraceCacheLineBytes - 1U);
    size_t count = iterationCount;
    if (!AllToAllGroupTraceCheckedMultiply(
            count, kAllToAllGroupTraceCoreCount, next) ||
        !AllToAllGroupTraceCheckedMultiply(next, coreBytes, count)) {
        return std::numeric_limits<size_t>::max();
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
    const size_t rawCoreBytes = static_cast<size_t>(groupCount) * passCount *
        kAllToAllGroupTracePhaseCount * sizeof(AllToAllGroupTraceTaskSpan);
    const size_t coreBytes = (rawCoreBytes + kAllToAllGroupTraceCacheLineBytes - 1U) &
        ~(kAllToAllGroupTraceCacheLineBytes - 1U);
    const size_t coreIndex = static_cast<size_t>(iteration) *
        kAllToAllGroupTraceCoreCount + core;
    const size_t taskIndex = (((static_cast<size_t>(group) * passCount + pass) *
        kAllToAllGroupTracePhaseCount) + phase);
    return AllToAllGroupTraceTaskSpanBaseOffset() +
        coreIndex * coreBytes + taskIndex * sizeof(AllToAllGroupTraceTaskSpan);
}

static_assert(sizeof(AllToAllGroupTraceSpan) == 16U,
    "group trace kernel span must contain two uint64 timestamps");
static_assert(sizeof(AllToAllGroupTraceTaskSpan) == 48U,
    "group trace task span layout changed");
static_assert(sizeof(AllToAllGroupTraceHeader) <= kAllToAllGroupTraceHeaderBytes,
    "group trace header must fit its region");
static_assert(AllToAllGroupTraceSimtStorageCore(0U) == 0U &&
    AllToAllGroupTraceSimtStorageCore(31U) == 63U,
    "SIMT trace workers must fit unused trace core slots");
static_assert(AllToAllGroupTraceTaskSpanBaseOffset() < kAllToAllGroupTraceBytes,
    "group trace kernel spans must fit in trace storage");

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_GROUP_TRACE_H
