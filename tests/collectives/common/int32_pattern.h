/*
 * Copyright (c) 2024-2026 TileXR Project
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILEXR_TESTS_COLLECTIVES_INT32_PATTERN_H
#define TILEXR_TESTS_COLLECTIVES_INT32_PATTERN_H

#include <cstdint>
#include <limits>

namespace TileXRCollectivesTest {

inline int RankBitsForPattern(int rankSize)
{
    int bits = 0;
    int capacity = 1;
    while (capacity < rankSize && bits < 7) {
        capacity <<= 1;
        ++bits;
    }
    return bits == 0 ? 1 : bits;
}

inline int IndexBitsForPattern(int rankSize)
{
    const int rankBits = RankBitsForPattern(rankSize);
    return 32 - 2 * rankBits;
}

inline int64_t MaxCollisionFreeInt32Count(int rankSize)
{
    const int indexBits = IndexBitsForPattern(rankSize);
    if (indexBits <= 0 || indexBits >= 63) {
        return 0;
    }
    return static_cast<int64_t>(1ULL << indexBits);
}

inline bool CanUseCollisionFreeInt32Pattern(int rankSize, int64_t count)
{
    return rankSize > 0 && rankSize <= 128 && count > 0 && count <= MaxCollisionFreeInt32Count(rankSize);
}

inline int32_t ExpectedInt32Value(int rankSize, int srcRank, int dstRank, int64_t index)
{
    const int rankBits = RankBitsForPattern(rankSize);
    const int indexBits = 32 - 2 * rankBits;
    const uint32_t rankMask = (1U << rankBits) - 1U;
    const uint32_t indexMask = indexBits == 32 ? std::numeric_limits<uint32_t>::max() :
        ((1U << indexBits) - 1U);
    const uint32_t encoded =
        ((static_cast<uint32_t>(srcRank) & rankMask) << (rankBits + indexBits)) |
        ((static_cast<uint32_t>(dstRank) & rankMask) << indexBits) |
        (static_cast<uint32_t>(index) & indexMask);
    return static_cast<int32_t>(encoded);
}

inline int32_t ExpectedAllGatherValue(int rankSize, int srcRank, int64_t index)
{
    return ExpectedInt32Value(rankSize, srcRank, 0, index);
}

inline int32_t ExpectedAllToAllValue(int rankSize, int srcRank, int dstRank, int64_t index)
{
    return ExpectedInt32Value(rankSize, srcRank, dstRank, index);
}

} // namespace TileXRCollectivesTest

#endif // TILEXR_TESTS_COLLECTIVES_INT32_PATTERN_H
