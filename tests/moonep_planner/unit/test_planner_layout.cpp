#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

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

void TestPrimary()
{
    SetBlockDimOverride(nullptr);
    TileXRMoonEp::PlannerLayout layout {};
    const int ret = TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(
        8, 8192, 16, 896, 112, 128, &layout);
    Check(ret == TileXR::TILEXR_SUCCESS, "primary layout failed");
    Check(layout.expertsPerRank == 112, "primary B mismatch");
    Check(layout.routeCount == 131072, "primary N mismatch");
    Check(layout.b == 112 && layout.tokenPadding == 128, "primary slot/padding mismatch");
    Check(layout.nvS == 159520, "primary NvS mismatch");
    Check(layout.blockDim == 64, "default blockDim mismatch");
    Check(layout.workspaceBytes < 400 * 1024, "primary workspace unexpectedly large");
    Check(layout.tpePrefixOffset % 32 == 0, "tpe offset unaligned");
    Check(layout.blockHistogramOffset % 32 == 0, "histogram offset unaligned");
    Check(layout.allocPrefixOffset % 32 == 0, "alloc offset unaligned");
    Check(layout.expertOffsetsOffset % 32 == 0, "expert offset unaligned");
    Check(layout.zOffset % 32 == 0, "z offset unaligned");
    Check(layout.groupTotalsOffset % 32 == 0, "group totals offset unaligned");
    Check(layout.workspaceBytes % 32 == 0, "workspace bytes unaligned");
    Check(layout.peerPublicationBytes == 896 * sizeof(int32_t), "peer publication bytes mismatch");
}

void TestBlockDimOverride()
{
    TileXRMoonEp::PlannerLayout layout {};
    SetBlockDimOverride("32");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(16, 64, 4, 64, 4, 1, &layout) ==
        TileXR::TILEXR_SUCCESS, "16-rank blockDim override rejected");
    Check(layout.blockDim == 32, "16-rank blockDim override mismatch");

    SetBlockDimOverride("15");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(16, 64, 4, 64, 4, 1, &layout) != 0,
        "blockDim below rankSize accepted");
    SetBlockDimOverride("65");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(16, 64, 4, 64, 4, 1, &layout) != 0,
        "blockDim above AIV count accepted");
    SetBlockDimOverride("32x");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(16, 64, 4, 64, 4, 1, &layout) != 0,
        "malformed blockDim accepted");
    SetBlockDimOverride(nullptr);
}

void TestEncodedCapacityBoundary()
{
    SetBlockDimOverride(nullptr);
    TileXRMoonEp::PlannerLayout layout {};
    constexpr int64_t kExactlyRepresentableRoutes = 33554432;
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(
        64, kExactlyRepresentableRoutes, 1, 64, 1, 1, &layout) == TileXR::TILEXR_SUCCESS,
        "R*N == INT32_MAX+1 rejected");
    Check(layout.nvS == kExactlyRepresentableRoutes,
        "boundary dispatched capacity mismatch");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(
        64, kExactlyRepresentableRoutes + 1, 1, 64, 1, 1, &layout) != 0,
        "R*N above INT32_MAX+1 accepted");
}

void TestRejectsInvalid()
{
    TileXRMoonEp::PlannerLayout layout {};
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(0, 1, 1, 1, 1, 1, &layout) != 0,
        "zero rank accepted");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, 1, 33, 896, 112, 1, &layout) != 0,
        "K > 32 accepted");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, 1, 1, 895, 1, 1, &layout) != 0,
        "non-divisible experts accepted");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, 1, 1, 8192, 1, 1, &layout) != 0,
        "expert count beyond UB contract accepted");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, std::numeric_limits<int32_t>::max(),
        32, 896, 112, 1, &layout) != 0, "overflow route count accepted");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, 64, 4, 64, 0, 1, &layout) != 0,
        "zero B accepted");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, 64, 4, 64, 9, 1, &layout) != 0,
        "B above E/R accepted");
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, 64, 4, 64, 8, 0, &layout) != 0,
        "zero tokenPadding accepted");
}

} // namespace

int main()
{
    TestPrimary();
    TestBlockDimOverride();
    TestEncodedCapacityBoundary();
    TestRejectsInvalid();
    SetBlockDimOverride(nullptr);
    return failures == 0 ? 0 : 1;
}
