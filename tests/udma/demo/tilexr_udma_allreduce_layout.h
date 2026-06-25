/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLREDUCE_LAYOUT_H
#define TILEXR_UDMA_ALLREDUCE_LAYOUT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace TileXR {
namespace Demo {

constexpr int32_t kAllReduceBaseValue = 1000;

inline int32_t AllReduceValue(int rank)
{
    return kAllReduceBaseValue + rank;
}

inline int32_t AllReduceExpectedSum(int rankSize)
{
    return rankSize * kAllReduceBaseValue + rankSize * (rankSize - 1) / 2;
}

inline void FillAllReduceInput(
    std::vector<int32_t>& input, int rank, int32_t elementsPerRank)
{
    std::fill(input.begin(), input.begin() + elementsPerRank, AllReduceValue(rank));
}

inline bool ValidateAllReduceOutput(
    const std::vector<int32_t>& output, int rankSize, int32_t elementsPerRank)
{
    const int32_t expected = AllReduceExpectedSum(rankSize);
    for (int32_t i = 0; i < elementsPerRank; ++i) {
        if (output[static_cast<size_t>(i)] != expected) {
            return false;
        }
    }
    return true;
}

inline void BuildAllReduceOutputFromInputs(
    const std::vector<int32_t>& allInputs, int rankSize, int32_t elementsPerRank,
    std::vector<int32_t>& output)
{
    std::fill(output.begin(), output.begin() + elementsPerRank, 0);
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        const size_t srcBase = static_cast<size_t>(srcRank) * elementsPerRank;
        for (int32_t i = 0; i < elementsPerRank; ++i) {
            output[static_cast<size_t>(i)] += allInputs[srcBase + i];
        }
    }
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLREDUCE_LAYOUT_H
