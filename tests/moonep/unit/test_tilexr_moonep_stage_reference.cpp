#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "moonep_stage_reference.h"

namespace {

using TileXRMoonEp::Reference::Bfloat16Ranks;
using TileXRMoonEp::Reference::Destinations;
using TileXRMoonEp::Reference::Float32Ranks;
using TileXRMoonEp::Reference::Shape;

int failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #expr << '\n'; \
            ++failures; \
        } \
    } while (0)

void CheckStatus(const char *name, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << name << ": expected " << expected << ", got " << actual << '\n';
        ++failures;
    }
}

Destinations BalancedDestinations(const Shape &shape)
{
    const int64_t capacity = shape.s * shape.k;
    Destinations dst(static_cast<size_t>(shape.world),
        std::vector<int32_t>(static_cast<size_t>(capacity)));
    for (int64_t src = 0; src < shape.world; ++src) {
        for (int64_t route = 0; route < capacity; ++route) {
            const int64_t global = src * capacity + route;
            const int64_t destRank = global % shape.world;
            const int64_t destOffset = global / shape.world;
            dst[static_cast<size_t>(src)][static_cast<size_t>(route)] =
                static_cast<int32_t>(destRank * capacity + destOffset);
        }
    }
    return dst;
}

Destinations DuplicateDestinations(const Shape &shape)
{
    const int64_t capacity = shape.s * shape.k;
    Destinations dst(static_cast<size_t>(shape.world),
        std::vector<int32_t>(static_cast<size_t>(capacity)));
    for (int64_t src = 0; src < shape.world; ++src) {
        for (int64_t token = 0; token < shape.s; ++token) {
            const int64_t globalToken = src * shape.s + token;
            const int64_t destRank = globalToken % shape.world;
            const int64_t destToken = globalToken / shape.world;
            for (int64_t topk = 0; topk < shape.k; ++topk) {
                const int64_t destOffset = destToken * shape.k + topk;
                const int32_t raw = static_cast<int32_t>(destRank * capacity + destOffset);
                dst[static_cast<size_t>(src)][static_cast<size_t>(token * shape.k + topk)] =
                    topk == 0 ? raw : -raw - 1;
            }
        }
    }
    return dst;
}

Bfloat16Ranks HiddenInput(const Shape &shape)
{
    Bfloat16Ranks input(static_cast<size_t>(shape.world));
    for (int64_t rank = 0; rank < shape.world; ++rank) {
        auto &rows = input[static_cast<size_t>(rank)];
        rows.resize(static_cast<size_t>(shape.s * shape.h));
        for (int64_t i = 0; i < shape.s * shape.h; ++i) {
            rows[static_cast<size_t>(i)] = TileXRMoonEp::Reference::FloatToBfloat16(
                static_cast<float>(rank * 100 + i + 1));
        }
    }
    return input;
}

Float32Ranks WeightInput(const Shape &shape)
{
    Float32Ranks input(static_cast<size_t>(shape.world));
    for (int64_t rank = 0; rank < shape.world; ++rank) {
        auto &weights = input[static_cast<size_t>(rank)];
        weights.resize(static_cast<size_t>(shape.s * shape.k));
        for (int64_t i = 0; i < shape.s * shape.k; ++i) {
            weights[static_cast<size_t>(i)] = static_cast<float>(rank * 1000 + i) + 0.25F;
        }
    }
    return input;
}

void TestDecode()
{
    TileXRMoonEp::Reference::DecodedRoute route {};
    CheckStatus("decode zero", TileXRMoonEp::Reference::DecodeRoute(0, 8, 2, &route), 0);
    CHECK_TRUE(route.rank == 0 && route.offset == 0 && !route.duplicate);
    CheckStatus("decode remote", TileXRMoonEp::Reference::DecodeRoute(10, 8, 2, &route), 0);
    CHECK_TRUE(route.rank == 1 && route.offset == 2 && !route.duplicate);
    CheckStatus("decode negative", TileXRMoonEp::Reference::DecodeRoute(-11, 8, 2, &route), 0);
    CHECK_TRUE(route.rank == 1 && route.offset == 2 && route.duplicate);
    CheckStatus("decode INT32_MIN", TileXRMoonEp::Reference::DecodeRoute(
        std::numeric_limits<int32_t>::min(), 268435456, 8, &route), 0);
    CHECK_TRUE(route.rank == 7 && route.offset == 268435455 && route.duplicate);
    CHECK_TRUE(TileXRMoonEp::Reference::DecodeRoute(16, 8, 2, &route) != 0);
}

void TestBalancedHiddenAndWeights()
{
    const Shape shape {2, 3, 2, 4};
    const Destinations dst = BalancedDestinations(shape);
    const Bfloat16Ranks hidden = HiddenInput(shape);
    Bfloat16Ranks dispatched;
    CheckStatus("hidden dispatch", TileXRMoonEp::Reference::DispatchBfloat16(
        shape, dst, hidden, &dispatched), 0);
    CHECK_TRUE(dispatched.size() == 2);
    CHECK_TRUE(dispatched[0].size() == static_cast<size_t>(shape.s * shape.k * shape.h));

    Bfloat16Ranks combined;
    CheckStatus("hidden combine", TileXRMoonEp::Reference::CombineBfloat16(
        shape, dst, dispatched, &combined), 0);
    for (int64_t rank = 0; rank < shape.world; ++rank) {
        for (int64_t i = 0; i < shape.s * shape.h; ++i) {
            const float original = TileXRMoonEp::Reference::Bfloat16ToFloat(
                hidden[static_cast<size_t>(rank)][static_cast<size_t>(i)]);
            const uint16_t expected = TileXRMoonEp::Reference::FloatToBfloat16(
                original * static_cast<float>(shape.k));
            CHECK_TRUE(combined[static_cast<size_t>(rank)][static_cast<size_t>(i)] == expected);
        }
    }

    const Float32Ranks weights = WeightInput(shape);
    Float32Ranks dispatchedWeights;
    Float32Ranks combinedWeights;
    CheckStatus("weight dispatch", TileXRMoonEp::Reference::DispatchFloat32(
        shape, dst, weights, &dispatchedWeights), 0);
    CheckStatus("weight combine", TileXRMoonEp::Reference::CombineFloat32(
        shape, dst, dispatchedWeights, &combinedWeights), 0);
    CHECK_TRUE(combinedWeights == weights);
}

void TestDuplicateHeavyAndUnusedRows()
{
    const Shape shape {8, 2, 32, 3};
    const Destinations dst = DuplicateDestinations(shape);
    const Bfloat16Ranks hidden = HiddenInput(shape);
    Bfloat16Ranks dispatched;
    CheckStatus("duplicate dispatch", TileXRMoonEp::Reference::DispatchBfloat16(
        shape, dst, hidden, &dispatched), 0);

    Bfloat16Ranks combinedFirst;
    Bfloat16Ranks combinedSecond;
    Bfloat16Ranks dispatchedSecond;
    CheckStatus("saved plan redispatch", TileXRMoonEp::Reference::DispatchBfloat16(
        shape, dst, hidden, &dispatchedSecond), 0);
    CHECK_TRUE(dispatched == dispatchedSecond);
    CheckStatus("duplicate combine", TileXRMoonEp::Reference::CombineBfloat16(
        shape, dst, dispatched, &combinedFirst), 0);
    CheckStatus("saved plan reuse", TileXRMoonEp::Reference::CombineBfloat16(
        shape, dst, dispatched, &combinedSecond), 0);
    CHECK_TRUE(combinedFirst == combinedSecond);

    const Float32Ranks weights = WeightInput(shape);
    Float32Ranks dispatchedWeights;
    Float32Ranks combinedWeights;
    CheckStatus("duplicate weight dispatch", TileXRMoonEp::Reference::DispatchFloat32(
        shape, dst, weights, &dispatchedWeights), 0);
    CheckStatus("duplicate weight combine", TileXRMoonEp::Reference::CombineFloat32(
        shape, dst, dispatchedWeights, &combinedWeights), 0);
    CHECK_TRUE(combinedWeights == weights);

    const Shape sparseShape {1, 2, 2, 3};
    const Destinations sparseDst {{0, -1, 2, -3}};
    const Bfloat16Ranks sparseHidden = HiddenInput(sparseShape);
    Bfloat16Ranks sparseDispatched;
    CheckStatus("sparse hidden dispatch", TileXRMoonEp::Reference::DispatchBfloat16(
        sparseShape, sparseDst, sparseHidden, &sparseDispatched), 0);
    for (int64_t row : {1, 3}) {
        for (int64_t column = 0; column < sparseShape.h; ++column) {
            CHECK_TRUE(sparseDispatched[0]
                [static_cast<size_t>(row * sparseShape.h + column)] == 0);
        }
    }

    const Shape skewedShape {2, 2, 2, 3};
    const Destinations skewedDst {{0, -1, 1, -2}, {2, -3, 3, -4}};
    const Bfloat16Ranks skewedHidden = HiddenInput(skewedShape);
    Bfloat16Ranks skewedDispatched;
    CheckStatus("skewed hidden dispatch", TileXRMoonEp::Reference::DispatchBfloat16(
        skewedShape, skewedDst, skewedHidden, &skewedDispatched), 0);
    for (uint16_t value : skewedDispatched[1]) {
        CHECK_TRUE(value == 0);
    }
}

void TestWorldAndKMatrix()
{
    const int64_t worlds[] = {1, 2, 8};
    const int64_t topks[] = {1, 2, 32};
    for (int64_t world : worlds) {
        for (int64_t k : topks) {
            const Shape shape {world, 2, k, 2};
            const Destinations dst = k == 32 ? DuplicateDestinations(shape) : BalancedDestinations(shape);
            const Float32Ranks weights = WeightInput(shape);
            Float32Ranks dispatched;
            Float32Ranks combined;
            CheckStatus("matrix dispatch", TileXRMoonEp::Reference::DispatchFloat32(
                shape, dst, weights, &dispatched), 0);
            CheckStatus("matrix combine", TileXRMoonEp::Reference::CombineFloat32(
                shape, dst, dispatched, &combined), 0);
            CHECK_TRUE(combined == weights);
        }
    }
}

void TestDeterministicExpertTransform()
{
    const Shape shape {2, 3, 1, 2};
    const Destinations dst = BalancedDestinations(shape);
    const Bfloat16Ranks hidden = HiddenInput(shape);
    Bfloat16Ranks dispatched;
    CheckStatus("transform dispatch", TileXRMoonEp::Reference::DispatchBfloat16(
        shape, dst, hidden, &dispatched), 0);
    for (int64_t rank = 0; rank < shape.world; ++rank) {
        for (uint16_t &bits : dispatched[static_cast<size_t>(rank)]) {
            const float value = TileXRMoonEp::Reference::Bfloat16ToFloat(bits);
            bits = TileXRMoonEp::Reference::FloatToBfloat16(
                value + static_cast<float>((rank + 1) * 10));
        }
    }
    Bfloat16Ranks combined;
    CheckStatus("transform combine", TileXRMoonEp::Reference::CombineBfloat16(
        shape, dst, dispatched, &combined), 0);
    CHECK_TRUE(combined.size() == static_cast<size_t>(shape.world));
    CHECK_TRUE(combined[0] != hidden[0]);
}

void TestPaddingAndDuplicatePlan()
{
    int64_t nvS = 0;
    CheckStatus("compute NvS", TileXRMoonEp::Reference::ComputeNvS(
        4, 2, 8, 2, 4, &nvS), 0);
    CHECK_TRUE(nvS == 32);
    CHECK_TRUE(TileXRMoonEp::Reference::ComputeNvS(
        4, 2, 7, 2, 4, &nvS) != 0);

    std::vector<int32_t> cu;
    TileXRMoonEp::Reference::FillRanges ranges;
    CheckStatus("padded layout", TileXRMoonEp::Reference::BuildPaddedLayout(
        {3, 0, 5, 1}, 4, 16, &cu, &ranges), 0);
    CHECK_TRUE(cu == std::vector<int32_t>({4, 4, 12, 16}));
    CHECK_TRUE(ranges.size() == 4);
    CHECK_TRUE(ranges[0][0] == 3 && ranges[0][1] == 1);
    CHECK_TRUE(ranges[1][0] == 0 && ranges[1][1] == 0);
    CHECK_TRUE(ranges[2][0] == 9 && ranges[2][1] == 3);
    CHECK_TRUE(ranges[3][0] == 13 && ranges[3][1] == 3);
    CheckStatus("padded layout physical tail", TileXRMoonEp::Reference::BuildPaddedLayout(
        {3, 0, 5, 1}, 4, 20, &cu, &ranges), 0);
    CHECK_TRUE(cu == std::vector<int32_t>({4, 4, 12, 16}));
    CHECK_TRUE(ranges[3][0] == 13 && ranges[3][1] == 7);
    CHECK_TRUE(TileXRMoonEp::Reference::BuildPaddedLayout(
        {3, 5}, 4, 8, &cu, &ranges) != 0);

    Shape shape {2, 2, 3, 4};
    shape.nvS = 8;
    const Destinations dst {
        {0, -2, -3, 3, 12, -14},
        {4, -6, 7, 8, -10, -11},
    };
    TileXRMoonEp::Reference::DuplicatePlans plans;
    CheckStatus("duplicate plans", TileXRMoonEp::Reference::BuildDuplicatePlans(
        shape, dst, &plans), 0);
    CHECK_TRUE(plans.size() == 2);
    CHECK_TRUE(plans[0].counts[0] == 2 && plans[0].counts[1] == 3);
    const std::array<int32_t, 3> firstGroup {{0, 0, 2}};
    CHECK_TRUE(plans[0].groups[0] == firstGroup);
    CHECK_TRUE(plans[0].loffs[0] == 1 && plans[0].loffs[1] == 2);
}

void TestPrefetchWeight()
{
    const int64_t world = 2;
    const int64_t e = 4;
    const int64_t b = 3;
    const int64_t row = 2;
    const std::vector<int32_t> experts {2, -1, 2, 1, 3, -1};
    std::vector<uint16_t> weight(static_cast<size_t>((e + b) * row),
        TileXRMoonEp::Reference::FloatToBfloat16(-7.0F));
    for (int64_t expert = 0; expert < e; ++expert) {
        for (int64_t column = 0; column < row; ++column) {
            weight[static_cast<size_t>(expert * row + column)] =
                TileXRMoonEp::Reference::FloatToBfloat16(
                    static_cast<float>(expert * 10 + column));
        }
    }
    const uint16_t untouched = weight[static_cast<size_t>((e + 1) * row)];
    CheckStatus("prefetch", TileXRMoonEp::Reference::PrefetchBfloat16(
        0, world, e, b, row, experts, &weight), 0);
    CHECK_TRUE(weight[static_cast<size_t>(e * row)] == weight[static_cast<size_t>(2 * row)]);
    CHECK_TRUE(weight[static_cast<size_t>((e + 2) * row + 1)] ==
        weight[static_cast<size_t>(2 * row + 1)]);
    CHECK_TRUE(weight[static_cast<size_t>((e + 1) * row)] == untouched);
}

void TestReduceGrad()
{
    const int64_t world = 2;
    const int64_t e = 4;
    const int64_t b = 3;
    const int64_t row = 2;
    const std::vector<int32_t> experts {2, -1, 2, 1, 3, -1};
    Float32Ranks grads(static_cast<size_t>(world),
        std::vector<float>(static_cast<size_t>((e + b) * row), 0.0F));
    Float32Ranks slots(static_cast<size_t>(world),
        std::vector<float>(static_cast<size_t>(b * row), 0.0F));
    for (int64_t rank = 0; rank < world; ++rank) {
        for (int64_t i = 0; i < e * row; ++i) {
            grads[static_cast<size_t>(rank)][static_cast<size_t>(i)] =
                static_cast<float>(rank * 1000 + i);
        }
        for (int64_t i = 0; i < b * row; ++i) {
            slots[static_cast<size_t>(rank)][static_cast<size_t>(i)] =
                static_cast<float>((rank + 1) * 100 + i);
        }
    }
    const float nonlocal = grads[0][static_cast<size_t>(2 * row)];
    const float unused = slots[0][static_cast<size_t>(row)];
    CheckStatus("reduce grad", TileXRMoonEp::Reference::ReduceGradFloat32(
        world, e, b, row, experts, &grads, &slots), 0);
    CHECK_TRUE(grads[1][static_cast<size_t>(2 * row)] ==
        static_cast<float>(1000 + 2 * row) + 100.0F + 104.0F);
    CHECK_TRUE(grads[0][static_cast<size_t>(1 * row)] ==
        static_cast<float>(1 * row) + 200.0F);
    CHECK_TRUE(grads[0][static_cast<size_t>(2 * row)] == nonlocal);
    CHECK_TRUE(slots[0][0] == 0.0F && slots[0][4] == 0.0F);
    CHECK_TRUE(slots[0][static_cast<size_t>(row)] == unused);
    CHECK_TRUE(slots[1][0] == 0.0F && slots[1][2] == 0.0F);
    CHECK_TRUE(slots[1][4] != 0.0F);
}

} // namespace

int main()
{
    TestDecode();
    TestBalancedHiddenAndWeights();
    TestDuplicateHeavyAndUnusedRows();
    TestWorldAndKMatrix();
    TestDeterministicExpertTransform();
    TestPaddingAndDuplicatePlan();
    TestPrefetchWeight();
    TestReduceGrad();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
