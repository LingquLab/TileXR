#include <cstdlib>
#include <iostream>

#include "planner_layout.h"
#include "tilexr_types.h"

namespace {

int failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++failures;
    }
}

void SetBlockDimOverride(const char *value)
{
#ifdef _WIN32
    _putenv_s("TILEXR_MOONEP_PLANNER_BLOCK_DIM", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("TILEXR_MOONEP_PLANNER_BLOCK_DIM");
    } else {
        setenv("TILEXR_MOONEP_PLANNER_BLOCK_DIM", value, 1);
    }
#endif
}

} // namespace

int main()
{
    TileXRMoonEpV3::PlannerLayout layout {};
    SetBlockDimOverride("64");
    Check(TileXRMoonEpV3::TileXRMoonEpBuildPlannerLayout(
        128, 8, 1, 128, 1, 1, &layout) == TileXR::TILEXR_SUCCESS,
        "128-rank Planner V3 layout rejected 64 AIV blocks");
    Check(layout.blockDim == 64, "128-rank Planner V3 blockDim mismatch");

    SetBlockDimOverride("65");
    Check(TileXRMoonEpV3::TileXRMoonEpBuildPlannerLayout(
        128, 8, 1, 128, 1, 1, &layout) != TileXR::TILEXR_SUCCESS,
        "Planner V3 accepted blockDim above the AIV limit");
    SetBlockDimOverride(nullptr);
    return failures == 0 ? 0 : 1;
}
