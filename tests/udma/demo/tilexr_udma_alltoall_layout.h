/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_LAYOUT_H
#define TILEXR_UDMA_ALLTOALL_LAYOUT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace TileXR {
namespace Demo {

constexpr int32_t kAllToAllBaseValue = 100000;

inline int32_t AllToAllValue(int srcRank, int dstRank)
{
    return kAllToAllBaseValue + srcRank * 1000 + dstRank;
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
