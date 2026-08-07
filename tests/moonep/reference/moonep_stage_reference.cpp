#include "moonep_stage_reference.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace TileXRMoonEp {
namespace Reference {
namespace {

constexpr int kSuccess = 0;
constexpr int kInvalid = -1;

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool ShapeValid(const Shape &shape, bool requireHidden, uint64_t *routeCount,
    uint64_t *capacity)
{
    if (routeCount == nullptr || capacity == nullptr || shape.world <= 0 || shape.s <= 0 ||
        shape.k <= 0 || shape.k > 32 || (requireHidden && shape.h <= 0)) {
        return false;
    }
    uint64_t routes = 0;
    if (!CheckedMul(static_cast<uint64_t>(shape.s), static_cast<uint64_t>(shape.k), &routes) ||
        routes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    const uint64_t rows = shape.nvS > 0 ? static_cast<uint64_t>(shape.nvS) : routes;
    if (rows < routes || rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    *routeCount = routes;
    *capacity = rows;
    return true;
}

bool RankCountValid(size_t size, int64_t world)
{
    return world >= 0 && static_cast<uint64_t>(world) <=
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()) &&
        size == static_cast<size_t>(world);
}

bool FlatSize(uint64_t rows, uint64_t columns, size_t *size)
{
    uint64_t elements = 0;
    if (size == nullptr || !CheckedMul(rows, columns, &elements) ||
        elements > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *size = static_cast<size_t>(elements);
    return true;
}

bool DestinationsValid(const Shape &shape, uint64_t routeCount, const Destinations &dst)
{
    if (!RankCountValid(dst.size(), shape.world)) {
        return false;
    }
    for (const auto &rankDst : dst) {
        if (rankDst.size() != static_cast<size_t>(routeCount)) {
            return false;
        }
    }
    return true;
}

} // namespace

int ComputeNvS(int64_t s, int64_t k, int64_t e, int64_t world,
    int64_t tokenPadding, int64_t *nvS)
{
    if (nvS == nullptr || s <= 0 || k <= 0 || e <= 0 || world <= 0 ||
        tokenPadding <= 0 || e % world != 0) {
        return kInvalid;
    }
    uint64_t routes = 0;
    uint64_t padding = 0;
    uint64_t result = 0;
    if (!CheckedMul(static_cast<uint64_t>(s), static_cast<uint64_t>(k), &routes) ||
        !CheckedMul(static_cast<uint64_t>(tokenPadding - 1),
            static_cast<uint64_t>(2 * (e / world)), &padding) ||
        !CheckedAdd(routes, padding, &result) ||
        result > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return kInvalid;
    }
    *nvS = static_cast<int64_t>(result);
    return kSuccess;
}

int BuildPaddedLayout(const std::vector<int32_t> &counts, int64_t tokenPadding,
    int64_t nvS, std::vector<int32_t> *cuSeqlens, FillRanges *zeroFillRanges)
{
    if (counts.empty() || tokenPadding <= 0 || nvS <= 0 || cuSeqlens == nullptr ||
        zeroFillRanges == nullptr) {
        return kInvalid;
    }
    std::vector<int32_t> nextCu;
    FillRanges nextRanges;
    nextCu.reserve(counts.size());
    nextRanges.reserve(counts.size());
    uint64_t cursor = 0;
    for (int32_t count : counts) {
        if (count < 0) {
            return kInvalid;
        }
        uint64_t realEnd = 0;
        if (!CheckedAdd(cursor, static_cast<uint64_t>(count), &realEnd)) {
            return kInvalid;
        }
        uint64_t paddedEnd = realEnd;
        if (count > 0) {
            const uint64_t alignment = static_cast<uint64_t>(tokenPadding);
            const uint64_t remainder = realEnd % alignment;
            if (remainder != 0 && !CheckedAdd(realEnd, alignment - remainder, &paddedEnd)) {
                return kInvalid;
            }
        }
        if (paddedEnd > static_cast<uint64_t>(nvS) ||
            paddedEnd > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            return kInvalid;
        }
        nextCu.push_back(static_cast<int32_t>(paddedEnd));
        if (paddedEnd > realEnd) {
            nextRanges.push_back({{static_cast<int32_t>(realEnd),
                static_cast<int32_t>(paddedEnd - realEnd)}});
        } else {
            nextRanges.push_back({{0, 0}});
        }
        cursor = paddedEnd;
    }
    *cuSeqlens = std::move(nextCu);
    *zeroFillRanges = std::move(nextRanges);
    return kSuccess;
}

int BuildDuplicatePlans(const Shape &shape, const Destinations &dst,
    DuplicatePlans *plans)
{
    uint64_t routeCount = 0;
    uint64_t capacity = 0;
    if (plans == nullptr || !ShapeValid(shape, false, &routeCount, &capacity) ||
        !DestinationsValid(shape, routeCount, dst)) {
        return kInvalid;
    }
    DuplicatePlans next(static_cast<size_t>(shape.world));
    for (int64_t src = 0; src < shape.world; ++src) {
        for (int64_t token = 0; token < shape.s; ++token) {
            std::vector<int64_t> primaries(static_cast<size_t>(shape.world), -1);
            std::vector<std::vector<int32_t>> duplicates(static_cast<size_t>(shape.world));
            for (int64_t topk = 0; topk < shape.k; ++topk) {
                const int64_t routeIndex = token * shape.k + topk;
                DecodedRoute route {};
                if (DecodeRoute(dst[static_cast<size_t>(src)][static_cast<size_t>(routeIndex)],
                        static_cast<int64_t>(capacity), shape.world, &route) != kSuccess) {
                    return kInvalid;
                }
                if (route.duplicate) {
                    duplicates[static_cast<size_t>(route.rank)].push_back(
                        static_cast<int32_t>(route.offset));
                } else if (primaries[static_cast<size_t>(route.rank)] < 0) {
                    primaries[static_cast<size_t>(route.rank)] = route.offset;
                }
            }
            for (int64_t rank = 0; rank < shape.world; ++rank) {
                const auto &dups = duplicates[static_cast<size_t>(rank)];
                if (dups.empty()) {
                    continue;
                }
                if (primaries[static_cast<size_t>(rank)] < 0) {
                    return kInvalid;
                }
                DuplicatePlan &plan = next[static_cast<size_t>(rank)];
                if (plan.groups.size() >= capacity || plan.loffs.size() + dups.size() > capacity) {
                    return kInvalid;
                }
                plan.groups.push_back({{static_cast<int32_t>(primaries[static_cast<size_t>(rank)]),
                    static_cast<int32_t>(plan.loffs.size()), static_cast<int32_t>(dups.size())}});
                plan.loffs.insert(plan.loffs.end(), dups.begin(), dups.end());
            }
        }
    }
    for (DuplicatePlan &plan : next) {
        plan.counts[0] = static_cast<int32_t>(plan.groups.size());
        plan.counts[1] = static_cast<int32_t>(plan.loffs.size());
    }
    *plans = std::move(next);
    return kSuccess;
}

int DecodeRoute(int32_t encoded, int64_t capacity, int64_t world, DecodedRoute *route)
{
    if (route == nullptr || capacity <= 0 || world <= 0) {
        return kInvalid;
    }
    const int64_t encoded64 = static_cast<int64_t>(encoded);
    const int64_t raw = encoded64 >= 0 ? encoded64 : -encoded64 - 1;
    uint64_t limit = 0;
    if (!CheckedMul(static_cast<uint64_t>(capacity), static_cast<uint64_t>(world), &limit) ||
        static_cast<uint64_t>(raw) >= limit) {
        return kInvalid;
    }
    route->rank = raw / capacity;
    route->offset = raw % capacity;
    route->duplicate = encoded < 0;
    return kSuccess;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = UINT32_C(0x7fff) + ((bits >> 16) & 1U);
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int DispatchBfloat16(const Shape &shape, const Destinations &dst,
    const Bfloat16Ranks &input, Bfloat16Ranks *output)
{
    uint64_t routeCount = 0;
    uint64_t capacity = 0;
    size_t inputElements = 0;
    size_t outputElements = 0;
    if (output == nullptr || !ShapeValid(shape, true, &routeCount, &capacity) ||
        !DestinationsValid(shape, routeCount, dst) || !RankCountValid(input.size(), shape.world) ||
        !FlatSize(static_cast<uint64_t>(shape.s), static_cast<uint64_t>(shape.h), &inputElements) ||
        !FlatSize(capacity, static_cast<uint64_t>(shape.h), &outputElements)) {
        return kInvalid;
    }
    for (const auto &rankInput : input) {
        if (rankInput.size() != inputElements) {
            return kInvalid;
        }
    }

    output->assign(static_cast<size_t>(shape.world), std::vector<uint16_t>(outputElements, 0));
    for (int64_t srcRank = 0; srcRank < shape.world; ++srcRank) {
        for (int64_t routeIndex = 0; routeIndex < static_cast<int64_t>(routeCount); ++routeIndex) {
            DecodedRoute route {};
            if (DecodeRoute(dst[static_cast<size_t>(srcRank)][static_cast<size_t>(routeIndex)],
                    static_cast<int64_t>(capacity), shape.world, &route) != kSuccess) {
                return kInvalid;
            }
            const int64_t token = routeIndex / shape.k;
            for (int64_t column = 0; column < shape.h; ++column) {
                (*output)[static_cast<size_t>(route.rank)]
                    [static_cast<size_t>(route.offset * shape.h + column)] =
                    input[static_cast<size_t>(srcRank)]
                        [static_cast<size_t>(token * shape.h + column)];
            }
        }
    }
    return kSuccess;
}

int DispatchFloat32(const Shape &shape, const Destinations &dst,
    const Float32Ranks &input, Float32Ranks *output)
{
    uint64_t routeCount = 0;
    uint64_t capacity = 0;
    if (output == nullptr || !ShapeValid(shape, false, &routeCount, &capacity) ||
        !DestinationsValid(shape, routeCount, dst) || !RankCountValid(input.size(), shape.world)) {
        return kInvalid;
    }
    for (const auto &rankInput : input) {
        if (rankInput.size() != static_cast<size_t>(routeCount)) {
            return kInvalid;
        }
    }

    output->assign(static_cast<size_t>(shape.world),
        std::vector<float>(static_cast<size_t>(capacity), 0.0F));
    for (int64_t srcRank = 0; srcRank < shape.world; ++srcRank) {
        for (int64_t routeIndex = 0; routeIndex < static_cast<int64_t>(routeCount); ++routeIndex) {
            DecodedRoute route {};
            if (DecodeRoute(dst[static_cast<size_t>(srcRank)][static_cast<size_t>(routeIndex)],
                    static_cast<int64_t>(capacity), shape.world, &route) != kSuccess) {
                return kInvalid;
            }
            (*output)[static_cast<size_t>(route.rank)][static_cast<size_t>(route.offset)] =
                input[static_cast<size_t>(srcRank)][static_cast<size_t>(routeIndex)];
        }
    }
    return kSuccess;
}

int CombineBfloat16(const Shape &shape, const Destinations &dst,
    const Bfloat16Ranks &input, Bfloat16Ranks *output)
{
    uint64_t routeCount = 0;
    uint64_t capacity = 0;
    size_t inputElements = 0;
    size_t outputElements = 0;
    if (output == nullptr || !ShapeValid(shape, true, &routeCount, &capacity) ||
        !DestinationsValid(shape, routeCount, dst) || !RankCountValid(input.size(), shape.world) ||
        !FlatSize(capacity, static_cast<uint64_t>(shape.h), &inputElements) ||
        !FlatSize(static_cast<uint64_t>(shape.s), static_cast<uint64_t>(shape.h), &outputElements)) {
        return kInvalid;
    }
    for (const auto &rankInput : input) {
        if (rankInput.size() != inputElements) {
            return kInvalid;
        }
    }

    output->assign(static_cast<size_t>(shape.world), std::vector<uint16_t>(outputElements, 0));
    for (int64_t originRank = 0; originRank < shape.world; ++originRank) {
        for (int64_t token = 0; token < shape.s; ++token) {
            for (int64_t column = 0; column < shape.h; ++column) {
                float sum = 0.0F;
                for (int64_t topk = 0; topk < shape.k; ++topk) {
                    const int64_t routeIndex = token * shape.k + topk;
                    DecodedRoute route {};
                    if (DecodeRoute(dst[static_cast<size_t>(originRank)]
                            [static_cast<size_t>(routeIndex)], static_cast<int64_t>(capacity),
                            shape.world, &route) != kSuccess) {
                        return kInvalid;
                    }
                    sum += Bfloat16ToFloat(input[static_cast<size_t>(route.rank)]
                        [static_cast<size_t>(route.offset * shape.h + column)]);
                }
                (*output)[static_cast<size_t>(originRank)]
                    [static_cast<size_t>(token * shape.h + column)] = FloatToBfloat16(sum);
            }
        }
    }
    return kSuccess;
}

int CombineFloat32(const Shape &shape, const Destinations &dst,
    const Float32Ranks &input, Float32Ranks *output)
{
    uint64_t routeCount = 0;
    uint64_t capacity = 0;
    if (output == nullptr || !ShapeValid(shape, false, &routeCount, &capacity) ||
        !DestinationsValid(shape, routeCount, dst) || !RankCountValid(input.size(), shape.world)) {
        return kInvalid;
    }
    for (const auto &rankInput : input) {
        if (rankInput.size() != static_cast<size_t>(capacity)) {
            return kInvalid;
        }
    }

    output->assign(static_cast<size_t>(shape.world),
        std::vector<float>(static_cast<size_t>(routeCount), 0.0F));
    for (int64_t originRank = 0; originRank < shape.world; ++originRank) {
        for (int64_t routeIndex = 0; routeIndex < static_cast<int64_t>(routeCount); ++routeIndex) {
            DecodedRoute route {};
            if (DecodeRoute(dst[static_cast<size_t>(originRank)][static_cast<size_t>(routeIndex)],
                    static_cast<int64_t>(capacity), shape.world, &route) != kSuccess) {
                return kInvalid;
            }
            (*output)[static_cast<size_t>(originRank)][static_cast<size_t>(routeIndex)] =
                input[static_cast<size_t>(route.rank)][static_cast<size_t>(route.offset)];
        }
    }
    return kSuccess;
}

int PrefetchBfloat16(int64_t rank, int64_t world, int64_t e, int64_t b,
    int64_t rowElements, const std::vector<int32_t> &expertsToCopy,
    std::vector<uint16_t> *fullWeight)
{
    if (fullWeight == nullptr || rank < 0 || world <= 0 || rank >= world || e <= 0 ||
        e % world != 0 || b <= 0 || rowElements <= 0 ||
        expertsToCopy.size() != static_cast<size_t>(world * b)) {
        return kInvalid;
    }
    size_t expected = 0;
    if (!FlatSize(static_cast<uint64_t>(e + b), static_cast<uint64_t>(rowElements),
            &expected) || fullWeight->size() != expected) {
        return kInvalid;
    }
    for (int64_t slot = 0; slot < b; ++slot) {
        const int32_t expert = expertsToCopy[static_cast<size_t>(rank * b + slot)];
        if (expert == -1) {
            continue;
        }
        if (expert < 0 || expert >= e) {
            return kInvalid;
        }
        const size_t src = static_cast<size_t>(expert * rowElements);
        const size_t dst = static_cast<size_t>((e + slot) * rowElements);
        std::copy_n(fullWeight->begin() + static_cast<std::ptrdiff_t>(src),
            static_cast<size_t>(rowElements), fullWeight->begin() + static_cast<std::ptrdiff_t>(dst));
    }
    return kSuccess;
}

int ReduceGradFloat32(int64_t world, int64_t e, int64_t b, int64_t rowElements,
    const std::vector<int32_t> &expertsToCopy, Float32Ranks *fullGrads,
    Float32Ranks *localReduceSlots)
{
    if (fullGrads == nullptr || localReduceSlots == nullptr || world <= 0 || e <= 0 ||
        e % world != 0 || b <= 0 || rowElements <= 0 ||
        expertsToCopy.size() != static_cast<size_t>(world * b) ||
        !RankCountValid(fullGrads->size(), world) ||
        !RankCountValid(localReduceSlots->size(), world)) {
        return kInvalid;
    }
    size_t gradElements = 0;
    size_t slotElements = 0;
    if (!FlatSize(static_cast<uint64_t>(e + b), static_cast<uint64_t>(rowElements),
            &gradElements) ||
        !FlatSize(static_cast<uint64_t>(b), static_cast<uint64_t>(rowElements),
            &slotElements)) {
        return kInvalid;
    }
    for (int64_t rank = 0; rank < world; ++rank) {
        if ((*fullGrads)[static_cast<size_t>(rank)].size() != gradElements ||
            (*localReduceSlots)[static_cast<size_t>(rank)].size() != slotElements) {
            return kInvalid;
        }
    }
    const int64_t expertsPerRank = e / world;
    for (int64_t srcRank = 0; srcRank < world; ++srcRank) {
        for (int64_t slot = 0; slot < b; ++slot) {
            const size_t planIndex = static_cast<size_t>(srcRank * b + slot);
            const int32_t expert = expertsToCopy[planIndex];
            if (expert == -1) {
                continue;
            }
            if (expert < 0 || expert >= e) {
                return kInvalid;
            }
            const int64_t owner = expert / expertsPerRank;
            for (int64_t column = 0; column < rowElements; ++column) {
                (*fullGrads)[static_cast<size_t>(owner)]
                    [static_cast<size_t>(expert * rowElements + column)] +=
                    (*localReduceSlots)[static_cast<size_t>(srcRank)]
                        [static_cast<size_t>(slot * rowElements + column)];
            }
        }
    }
    for (int64_t srcRank = 0; srcRank < world; ++srcRank) {
        for (int64_t slot = 0; slot < b; ++slot) {
            if (expertsToCopy[static_cast<size_t>(srcRank * b + slot)] < 0) {
                continue;
            }
            std::fill_n((*localReduceSlots)[static_cast<size_t>(srcRank)].begin() +
                    static_cast<std::ptrdiff_t>(slot * rowElements),
                static_cast<size_t>(rowElements), 0.0F);
        }
    }
    return kSuccess;
}

} // namespace Reference
} // namespace TileXRMoonEp
