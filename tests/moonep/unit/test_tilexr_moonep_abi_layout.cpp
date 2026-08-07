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
    static_assert(std::is_standard_layout<TileXRMoonEpReduceGradWorkspaceQueryV2>::value,
        "ReduceGrad workspace query must be standard layout");
    static_assert(std::is_standard_layout<TileXRMoonEpReduceGradWorkspaceInfoV2>::value,
        "ReduceGrad workspace info must be standard layout");
    static_assert(std::is_standard_layout<TileXRMoonEpReduceGradArgsV2>::value,
        "ReduceGrad V2 args must be standard layout");
    static_assert(sizeof(TileXRMoonEpReduceGradWorkspaceQueryV2) == 64,
        "Unexpected ReduceGrad workspace query size");
    static_assert(sizeof(TileXRMoonEpReduceGradWorkspaceInfoV2) == 96,
        "Unexpected ReduceGrad workspace info size");
    static_assert(offsetof(TileXRMoonEpReduceGradWorkspaceInfoV2, rowBytes) == 56,
        "Unexpected ReduceGrad rowBytes offset");
    static_assert(offsetof(TileXRMoonEpReduceGradWorkspaceInfoV2, transports) == 80,
        "Unexpected ReduceGrad transports offset");
    static_assert(sizeof(TileXRMoonEpReduceGradArgsV2) == 96,
        "Unexpected ReduceGrad V2 args size");
    static_assert(offsetof(TileXRMoonEpReduceGradArgsV2, workspace) == 48,
        "Unexpected ReduceGrad workspace offset");
    static_assert(offsetof(TileXRMoonEpReduceGradArgsV2, status) == 64,
        "Unexpected ReduceGrad status offset");

    Check(TILEXR_MOONEP_ABI_VERSION_V1 == 1, "ABI version must be 1");
    Check(TILEXR_MOONEP_ABI_VERSION_V2 == 2, "V2 ABI version must be 2");
    Check(TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES == UINT64_C(1048576),
        "ReduceGrad UDMA threshold must be exactly 1 MiB");
    Check((TILEXR_MOONEP_STAGE_PLANNING & TILEXR_MOONEP_STAGE_DISPATCH) == 0,
        "Stage capability bits must not overlap");
    Check(TILEXR_MOONEP_MAX_TENSOR_RANK == 4, "Tensor rank must remain fixed at four");
    return g_failures == 0 ? 0 : 1;
}
