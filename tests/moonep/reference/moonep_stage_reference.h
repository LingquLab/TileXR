#ifndef TILEXR_MOONEP_STAGE_REFERENCE_H
#define TILEXR_MOONEP_STAGE_REFERENCE_H

#include <array>
#include <cstdint>
#include <vector>

namespace TileXRMoonEp {
namespace Reference {

struct Shape {
    int64_t world = 0;
    int64_t s = 0;
    int64_t k = 0;
    int64_t h = 0;
    int64_t e = 0;
    int64_t b = 0;
    int64_t nvS = 0;
    int64_t tokenPadding = 1;
};

struct DecodedRoute {
    int64_t rank = 0;
    int64_t offset = 0;
    bool duplicate = false;
};

using Destinations = std::vector<std::vector<int32_t>>;
using Bfloat16Ranks = std::vector<std::vector<uint16_t>>;
using Float32Ranks = std::vector<std::vector<float>>;
using FillRanges = std::vector<std::array<int32_t, 2>>;

struct DuplicatePlan {
    std::vector<std::array<int32_t, 3>> groups;
    std::vector<int32_t> loffs;
    std::array<int32_t, 2> counts {{0, 0}};
};

using DuplicatePlans = std::vector<DuplicatePlan>;

int ComputeNvS(int64_t s, int64_t k, int64_t e, int64_t world,
    int64_t tokenPadding, int64_t *nvS);
int BuildPaddedLayout(const std::vector<int32_t> &counts, int64_t tokenPadding,
    int64_t nvS, std::vector<int32_t> *cuSeqlens, FillRanges *zeroFillRanges);
int BuildDuplicatePlans(const Shape &shape, const Destinations &dst,
    DuplicatePlans *plans);

int DecodeRoute(int32_t encoded, int64_t capacity, int64_t world, DecodedRoute *route);

uint16_t FloatToBfloat16(float value);
float Bfloat16ToFloat(uint16_t value);

int DispatchBfloat16(const Shape &shape, const Destinations &dst,
    const Bfloat16Ranks &input, Bfloat16Ranks *output);
int DispatchFloat32(const Shape &shape, const Destinations &dst,
    const Float32Ranks &input, Float32Ranks *output);
int CombineBfloat16(const Shape &shape, const Destinations &dst,
    const Bfloat16Ranks &input, Bfloat16Ranks *output);
int CombineFloat32(const Shape &shape, const Destinations &dst,
    const Float32Ranks &input, Float32Ranks *output);
int PrefetchBfloat16(int64_t rank, int64_t world, int64_t e, int64_t b,
    int64_t rowElements, const std::vector<int32_t> &expertsToCopy,
    std::vector<uint16_t> *fullWeight);
int ReduceGradFloat32(int64_t world, int64_t e, int64_t b, int64_t rowElements,
    const std::vector<int32_t> &expertsToCopy, Float32Ranks *fullGrads,
    Float32Ranks *localReduceSlots);

} // namespace Reference
} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_STAGE_REFERENCE_H
