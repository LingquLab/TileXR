#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "tilexr_moonep.h"

namespace {

int g_failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    static_assert(sizeof(void *) == 8, "The TileXR MoonEP ABI targets 64-bit hosts");
    static_assert(std::is_standard_layout<TileXRMoonEpTensorV1>::value,
        "TensorV1 must be standard layout");
    static_assert(std::is_standard_layout<TileXRMoonEpPlanV1>::value,
        "PlanV1 must be standard layout");
    static_assert(sizeof(TileXRMoonEpTensorV1) == 64, "Unexpected TensorV1 size");
    static_assert(offsetof(TileXRMoonEpTensorV1, data) == 8, "Unexpected TensorV1 data offset");
    static_assert(offsetof(TileXRMoonEpTensorV1, elementCount) == 16,
        "Unexpected TensorV1 elementCount offset");
    static_assert(offsetof(TileXRMoonEpTensorV1, shape) == 32,
        "Unexpected TensorV1 shape offset");
    static_assert(sizeof(TileXRMoonEpPlanV1) == 120, "Unexpected PlanV1 size");
    static_assert(offsetof(TileXRMoonEpPlanV1, n) == 8, "Unexpected PlanV1 N offset");
    static_assert(offsetof(TileXRMoonEpPlanV1, nvS) == 40, "Unexpected PlanV1 NvS offset");
    static_assert(offsetof(TileXRMoonEpPlanV1, dst) == 56, "Unexpected PlanV1 dst offset");
    static_assert(offsetof(TileXRMoonEpPlanV1, dupGroups) == 88,
        "Unexpected PlanV1 duplicate groups offset");
    static_assert(offsetof(TileXRMoonEpPlanV1, status) == 112,
        "Unexpected PlanV1 status offset");
    static_assert(sizeof(TileXRMoonEpPlanningArgsV1) == 80,
        "Unexpected PlanningArgsV1 size");
    static_assert(offsetof(TileXRMoonEpPlanningArgsV1, cuSeqlens) == 48,
        "Unexpected PlanningArgsV1 cuSeqlens offset");
    static_assert(sizeof(TileXRMoonEpDispatchArgsV1) == 64,
        "Unexpected DispatchArgsV1 size");
    static_assert(offsetof(TileXRMoonEpDispatchArgsV1, hiddenSh) == 24,
        "Unexpected DispatchArgsV1 hidden input offset");
    static_assert(offsetof(TileXRMoonEpDispatchArgsV1, routeWeightsNvs) == 48,
        "Unexpected DispatchArgsV1 route-weight output offset");
    static_assert(sizeof(TileXRMoonEpPrefetchWeightArgsV1) == 56,
        "Unexpected PrefetchWeightArgsV1 size");
    static_assert(offsetof(TileXRMoonEpPrefetchWeightArgsV1, fullGateWeight) == 24,
        "Unexpected PrefetchWeightArgsV1 gate offset");
    static_assert(sizeof(TileXRMoonEpCombineArgsV1) == 64,
        "Unexpected CombineArgsV1 size");
    static_assert(offsetof(TileXRMoonEpCombineArgsV1, hiddenNvsh) == 24,
        "Unexpected CombineArgsV1 hidden input offset");
    static_assert(sizeof(TileXRMoonEpReduceGradArgsV1) == 80,
        "Unexpected ReduceGradArgsV1 size");
    static_assert(offsetof(TileXRMoonEpReduceGradArgsV1, gateReduceBuffer) == 48,
        "Unexpected ReduceGradArgsV1 reduce buffer offset");

    Check(TILEXR_MOONEP_ABI_VERSION_V1 == 1, "ABI version must be 1");
    Check((TILEXR_MOONEP_STAGE_PLANNING & TILEXR_MOONEP_STAGE_DISPATCH) == 0,
        "Stage capability bits must not overlap");
    Check(TILEXR_MOONEP_MAX_TENSOR_RANK == 4, "Tensor rank must remain fixed at four");
    Check(TILEXR_MOONEP_FLAG_BUILD_DEDUP != TILEXR_MOONEP_FLAG_SKIP_INTER_RANK_SYNC,
        "Dispatch flags must not overlap");
    Check((TILEXR_MOONEP_FLAG_ZERO_COPY & TILEXR_MOONEP_FLAG_BUILD_DEDUP) == 0,
        "Zero-copy and dedup flags must not overlap");
    return g_failures == 0 ? 0 : 1;
}
