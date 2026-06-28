/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_LAYOUT_H
#define TILEXR_UDMA_ALLTOALL_LAYOUT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace TileXR {
namespace Demo {

constexpr int32_t kAllToAllBaseValue = 100000;
constexpr size_t kAllToAllUdmaMaxRegisteredBytes = 128ULL * 1024ULL * 1024ULL;

struct AllToAllChunkPlan {
    uint32_t passCount = 1;
    int32_t chunkElements = 0;
    size_t chunkBytesPerRank = 0;
    size_t registeredBytes = 0;
};

constexpr size_t kAllToAllBigDataMaxRegisteredBytes = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kAllToAllBigDataControlSlotBytes = 64ULL;
constexpr uint32_t kAllToAllBigDataCopyCoreCount = 32U;

struct AllToAllBigDataPlan {
    uint32_t passCount = 1;
    int32_t chunkElements = 0;
    size_t chunkBytesPerPeer = 0;
    size_t dataBytes = 0;
    size_t readyPayloadOffset = 0;
    size_t ackPayloadOffset = 0;
    size_t readySignalOffset = 0;
    size_t ackSignalOffset = 0;
    size_t controlBytes = 0;
    size_t signalBytes = 0;
    size_t registeredBytes = 0;
};

inline int32_t AllToAllValue(int srcRank, int dstRank)
{
    return kAllToAllBaseValue + srcRank * 1000 + dstRank;
}

inline AllToAllChunkPlan PlanAllToAllUdmaChunks(int rankSize, int32_t elementsPerPeer)
{
    AllToAllChunkPlan plan {};
    if (rankSize <= 0 || elementsPerPeer <= 0) {
        return plan;
    }

    const size_t totalBytesPerRank = static_cast<size_t>(rankSize) * static_cast<size_t>(elementsPerPeer) *
        sizeof(int32_t);
    const size_t maxChunkBytesPerRank =
        std::max<size_t>(sizeof(int32_t) * static_cast<size_t>(rankSize), kAllToAllUdmaMaxRegisteredBytes / 2);
    size_t chunkElements = maxChunkBytesPerRank / (static_cast<size_t>(rankSize) * sizeof(int32_t));
    if (chunkElements == 0) {
        chunkElements = 1;
    }
    if (chunkElements > static_cast<size_t>(elementsPerPeer)) {
        chunkElements = static_cast<size_t>(elementsPerPeer);
    }

    plan.chunkElements = static_cast<int32_t>(chunkElements);
    plan.chunkBytesPerRank = static_cast<size_t>(rankSize) * chunkElements * sizeof(int32_t);
    plan.registeredBytes = plan.chunkBytesPerRank * 2;
    plan.passCount = static_cast<uint32_t>(
        (static_cast<size_t>(elementsPerPeer) + chunkElements - 1) / chunkElements);
    if (plan.chunkBytesPerRank > totalBytesPerRank) {
        plan.chunkBytesPerRank = totalBytesPerRank;
    }
    return plan;
}

inline AllToAllBigDataPlan PlanAllToAllBigDataUdma(int rankSize, int32_t elementsPerPeer)
{
    AllToAllBigDataPlan plan {};
    if (rankSize <= 0 || elementsPerPeer <= 0) {
        return plan;
    }

    plan.registeredBytes = kAllToAllBigDataMaxRegisteredBytes;
    plan.controlBytes = 2ULL * static_cast<size_t>(rankSize) * kAllToAllBigDataControlSlotBytes;
    plan.signalBytes = 2ULL * static_cast<size_t>(rankSize) * kAllToAllBigDataControlSlotBytes;
    if (plan.controlBytes + plan.signalBytes >= plan.registeredBytes) {
        return plan;
    }

    const size_t dataSlotCount = static_cast<size_t>(rankSize) * 2ULL;
    plan.dataBytes = plan.registeredBytes - plan.controlBytes - plan.signalBytes;
    plan.chunkElements = static_cast<int32_t>(
        plan.dataBytes / (dataSlotCount * sizeof(int32_t)));
    if (plan.chunkElements <= 0) {
        plan.chunkElements = 1;
    }
    if (plan.chunkElements > elementsPerPeer) {
        plan.chunkElements = elementsPerPeer;
    }

    plan.chunkBytesPerPeer = static_cast<size_t>(plan.chunkElements) * sizeof(int32_t);
    plan.dataBytes = dataSlotCount * plan.chunkBytesPerPeer;
    plan.readyPayloadOffset = plan.dataBytes;
    plan.ackPayloadOffset =
        plan.readyPayloadOffset + static_cast<size_t>(rankSize) * kAllToAllBigDataControlSlotBytes;
    plan.readySignalOffset =
        plan.ackPayloadOffset + static_cast<size_t>(rankSize) * kAllToAllBigDataControlSlotBytes;
    plan.ackSignalOffset =
        plan.readySignalOffset + static_cast<size_t>(rankSize) * kAllToAllBigDataControlSlotBytes;
    plan.controlBytes = 2ULL * static_cast<size_t>(rankSize) * kAllToAllBigDataControlSlotBytes;
    plan.signalBytes = 2ULL * static_cast<size_t>(rankSize) * kAllToAllBigDataControlSlotBytes;
    plan.registeredBytes = plan.dataBytes + plan.controlBytes + plan.signalBytes;
    plan.passCount = static_cast<uint32_t>(
        (static_cast<size_t>(elementsPerPeer) + static_cast<size_t>(plan.chunkElements) - 1) /
        static_cast<size_t>(plan.chunkElements));
    return plan;
}

inline uint32_t AllToAllBigDataBlockDim(int rankSize)
{
    if (rankSize <= 0) {
        return kAllToAllBigDataCopyCoreCount;
    }
    return std::max(static_cast<uint32_t>(rankSize), kAllToAllBigDataCopyCoreCount);
}

inline void FillAllToAllInput(
    std::vector<int32_t>& input, int rank, int rankSize, int32_t elementsPerPeer)
{
    for (int dstRank = 0; dstRank < rankSize; ++dstRank) {
        std::fill(input.begin() + static_cast<size_t>(dstRank) * elementsPerPeer,
                  input.begin() + static_cast<size_t>(dstRank + 1) * elementsPerPeer,
                  AllToAllValue(rank, dstRank));
    }
}

inline bool ValidateAllToAllOutput(
    const std::vector<int32_t>& output, int rank, int rankSize, int32_t elementsPerPeer)
{
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        const int32_t expected = AllToAllValue(srcRank, rank);
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            const size_t offset = static_cast<size_t>(srcRank) * elementsPerPeer + i;
            if (output[offset] != expected) {
                return false;
            }
        }
    }
    return true;
}

inline void BuildAllToAllOutputFromInputs(
    const std::vector<int32_t>& allInputs, int rank, int rankSize, int32_t elementsPerPeer,
    std::vector<int32_t>& output)
{
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        const size_t srcBase = static_cast<size_t>(srcRank) * rankSize * elementsPerPeer +
            static_cast<size_t>(rank) * elementsPerPeer;
        const size_t dstBase = static_cast<size_t>(srcRank) * elementsPerPeer;
        std::copy(allInputs.begin() + srcBase,
                  allInputs.begin() + srcBase + elementsPerPeer,
                  output.begin() + dstBase);
    }
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_LAYOUT_H
