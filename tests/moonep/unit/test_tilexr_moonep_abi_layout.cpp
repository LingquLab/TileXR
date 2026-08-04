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
    static_assert(sizeof(TileXRMoonEpPlanV1) == 104, "Unexpected PlanV1 size");
    static_assert(offsetof(TileXRMoonEpPlanV1, dispatchedCapacity) == 56,
        "Unexpected PlanV1 capacity offset");
    static_assert(offsetof(TileXRMoonEpPlanV1, dst) == 64, "Unexpected PlanV1 dst offset");
    static_assert(sizeof(TileXRMoonEpPlanningArgsV1) == 72,
        "Unexpected PlanningArgsV1 size");
    static_assert(sizeof(TileXRMoonEpDispatchArgsV1) == 48,
        "Unexpected DispatchArgsV1 size");
    static_assert(sizeof(TileXRMoonEpDispatchArgsV1) == sizeof(TileXRMoonEpPrefetchWeightArgsV1),
        "Stub argument layouts must match");
    static_assert(sizeof(TileXRMoonEpDispatchArgsV1) == sizeof(TileXRMoonEpCombineArgsV1),
        "Stub argument layouts must match");
    static_assert(sizeof(TileXRMoonEpDispatchArgsV1) == sizeof(TileXRMoonEpReduceGradArgsV1),
        "Stub argument layouts must match");

    Check(TILEXR_MOONEP_ABI_VERSION_V1 == 1, "ABI version must be 1");
    Check((TILEXR_MOONEP_STAGE_PLANNING & TILEXR_MOONEP_STAGE_DISPATCH) == 0,
        "Stage capability bits must not overlap");
    Check(TILEXR_MOONEP_MAX_TENSOR_RANK == 4, "Tensor rank must remain fixed at four");
    return g_failures == 0 ? 0 : 1;
}
