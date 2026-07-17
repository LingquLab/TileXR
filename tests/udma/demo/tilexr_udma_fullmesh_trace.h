/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_FULLMESH_TRACE_H
#define TILEXR_UDMA_FULLMESH_TRACE_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace TileXR {
namespace Demo {

constexpr uint32_t kFullmeshTraceMagic = 0x464d5452U; // "FMTR"
constexpr uint32_t kFullmeshTraceVersion = 1U;
constexpr size_t kFullmeshTraceBytes = 8ULL * 1024ULL * 1024ULL;
constexpr size_t kFullmeshTraceHeaderBytes = 4096ULL;
constexpr uint32_t kFullmeshTraceMaxIterations = 50U;
constexpr uint32_t kFullmeshTraceMaxCores = 35U;
constexpr uint32_t kFullmeshTraceKernelRegions = 2U;
constexpr uint32_t kFullmeshTracePhaseCount = 14U;
constexpr uint64_t kFullmeshTraceCyclesPerUs = 1000ULL;

enum FullmeshTraceKernelRegion : uint32_t {
    kFullmeshTraceKernel = 0U,
    kFullmeshTraceWork = 1U,
};

enum FullmeshTracePhase : uint32_t {
    kFullmeshTracePhasePass = 0U,
    kFullmeshTracePhaseSelfCopy = 1U,
    kFullmeshTracePhasePeerCopy = 2U,
    kFullmeshTracePhasePublishCopyReady = 3U,
    kFullmeshTracePhaseWaitCopyReady = 4U,
    kFullmeshTracePhaseDataPut = 5U,
    kFullmeshTracePhaseQuiet = 6U,
    kFullmeshTracePhaseSegmentDone = 7U,
    kFullmeshTracePhasePublishReady = 8U,
    kFullmeshTracePhaseWaitReady = 9U,
    kFullmeshTracePhaseOutputCopy = 10U,
    kFullmeshTracePhasePublishRecvDone = 11U,
    kFullmeshTracePhaseWaitRecvDone = 12U,
    kFullmeshTracePhaseAck = 13U,
};

struct FullmeshTraceSpan {
    uint64_t beginCycle;
    uint64_t endCycle;
};

struct FullmeshTraceHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t rank;
    uint32_t iterationCount;
    uint32_t passCount;
    uint32_t coreCount;
    uint32_t rankSize;
    uint32_t phaseCount;
    uint64_t cyclesPerUs;
    uint64_t traceBytes;
    uint64_t kernelSpanOffset;
    uint64_t taskSpanOffset;
};

constexpr size_t FullmeshTraceKernelSpanCount()
{
    return static_cast<size_t>(kFullmeshTraceMaxIterations) *
        kFullmeshTraceMaxCores * kFullmeshTraceKernelRegions;
}

constexpr size_t FullmeshTraceKernelSpanOffset(
    uint32_t iteration, uint32_t core, uint32_t region)
{
    return kFullmeshTraceHeaderBytes +
        ((static_cast<size_t>(iteration) * kFullmeshTraceMaxCores + core) *
        kFullmeshTraceKernelRegions + region) * sizeof(FullmeshTraceSpan);
}

constexpr size_t FullmeshTraceTaskSpanBaseOffset()
{
    return kFullmeshTraceHeaderBytes +
        FullmeshTraceKernelSpanCount() * sizeof(FullmeshTraceSpan);
}

inline bool FullmeshTraceMultiply(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0U && rhs > std::numeric_limits<size_t>::max() / lhs) {
        result = std::numeric_limits<size_t>::max();
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline size_t FullmeshTraceLayoutBytes(
    uint32_t iterationCount, uint32_t passCount, uint32_t rankSize)
{
    size_t count = iterationCount;
    size_t next = 0U;
    const size_t factors[] = {
        kFullmeshTraceMaxCores, passCount, rankSize,
        kFullmeshTracePhaseCount, sizeof(FullmeshTraceSpan)};
    for (size_t factor : factors) {
        if (!FullmeshTraceMultiply(count, factor, next)) {
            return std::numeric_limits<size_t>::max();
        }
        count = next;
    }
    if (count > std::numeric_limits<size_t>::max() - FullmeshTraceTaskSpanBaseOffset()) {
        return std::numeric_limits<size_t>::max();
    }
    return FullmeshTraceTaskSpanBaseOffset() + count;
}

inline bool FullmeshTraceLayoutFits(
    uint32_t iterationCount, uint32_t passCount, uint32_t rankSize)
{
    return iterationCount > 0U && iterationCount <= kFullmeshTraceMaxIterations &&
        passCount > 0U && rankSize > 0U &&
        FullmeshTraceLayoutBytes(iterationCount, passCount, rankSize) <= kFullmeshTraceBytes;
}

inline size_t FullmeshTraceTaskSpanOffset(
    uint32_t iteration, uint32_t core, uint32_t pass, uint32_t peer,
    uint32_t phase, uint32_t passCount, uint32_t rankSize)
{
    const size_t index =
        ((((static_cast<size_t>(iteration) * kFullmeshTraceMaxCores + core) *
        passCount + pass) * rankSize + peer) * kFullmeshTracePhaseCount) + phase;
    return FullmeshTraceTaskSpanBaseOffset() + index * sizeof(FullmeshTraceSpan);
}

static_assert(sizeof(FullmeshTraceSpan) == 16U, "fullmesh trace span must contain two uint64 timestamps");
static_assert(sizeof(FullmeshTraceHeader) <= kFullmeshTraceHeaderBytes,
    "fullmesh trace header must fit its region");
static_assert(FullmeshTraceTaskSpanBaseOffset() < kFullmeshTraceBytes,
    "fullmesh trace kernel spans must fit in 8 MiB");

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_FULLMESH_TRACE_H
