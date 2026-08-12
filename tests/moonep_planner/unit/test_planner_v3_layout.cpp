#include <cstdlib>
#include <iostream>

#include "planner_common.h"
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
    Check(layout.dstLocalOffset % TileXRMoonEpV3::kPlannerWorkspaceAlignment == 0,
        "Planner V3 dstLocal offset is not aligned");
    Check(layout.dstLocalOffset + static_cast<uint64_t>(layout.nvS) * sizeof(int32_t) <=
            layout.workspaceBytes,
        "Planner V3 dstLocal exceeds workspace");
    Check(layout.peerDstOffset >= 128U * sizeof(int32_t),
        "Planner V3 peer dst overlaps tokens-per-expert publication");
    Check(layout.peerPublicationBytes ==
            layout.peerDstOffset + static_cast<uint64_t>(layout.routeCount) * sizeof(int32_t),
        "Planner V3 peer publication size excludes dst routes");

    Check(TileXRMoonEpV3::TileXRMoonEpBuildPlannerLayout(
        8, 256, 4, 32, 2, 128, &layout) == TileXR::TILEXR_SUCCESS,
        "PR113 Planner V3 layout rejected");
    Check(layout.nvS == 2040, "PR113 Planner V3 NvS mismatch");
    Check(layout.workspaceBytes >=
            layout.dstLocalOffset + 2040U * sizeof(int32_t),
        "PR113 Planner V3 dstLocal allocation is incomplete");

    SetBlockDimOverride("65");
    Check(TileXRMoonEpV3::TileXRMoonEpBuildPlannerLayout(
        128, 8, 1, 128, 1, 1, &layout) != TileXR::TILEXR_SUCCESS,
        "Planner V3 accepted blockDim above the AIV limit");
    SetBlockDimOverride(nullptr);
    return failures == 0 ? 0 : 1;
}
