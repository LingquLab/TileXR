#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "combine_v2_layout.h"
#include "combine_v2_profile.h"
#include "tilexr_moonep.h"

namespace {

int failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

void CheckStatus(int actual, int expected, const char *message)
{
    if (actual != expected) {
        std::cerr << message << ": expected " << expected
                  << ", got " << actual << '\n';
        ++failures;
    }
}

void TestTargetLayout()
{
    using namespace TileXRMoonEp;
    Check(sizeof(MoonEpCombineV2ProfileRecord) == 384U,
        "profile record size mismatch");
    Check(kMoonEpCombineV2ProfileTimePointCount == 22U,
        "profile point count mismatch");
    Check(kMoonEpCombineV2ProfileMetricCount == 8U,
        "profile metric count mismatch");

    CombineV2Layout layout {};
    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        8192, 3584, 16, 131072, TILEXR_MOONEP_DTYPE_BFLOAT16, &layout),
        TILEXR_MOONEP_SUCCESS, "target layout");
    Check(layout.slots == 131072U, "target slots mismatch");
    Check(layout.rowBytes == 7168U, "target row bytes mismatch");
    Check(layout.expertBytes == 939524096U, "target expert bytes mismatch");
    Check(layout.profileBytes ==
        kMoonEpCombineV2CoreCount * sizeof(MoonEpCombineV2ProfileRecord),
        "target profile bytes mismatch");
    Check(layout.scratchOffset[0] == layout.profileOffset + layout.profileBytes,
        "scratch epoch 0 offset mismatch");
    Check(layout.scratchOffset[1] == layout.scratchOffset[0] + layout.expertBytes,
        "scratch epoch 1 offset mismatch");
    Check(layout.doneBytes == 32768U, "done bytes mismatch");
    Check(layout.grantBytes == 262144U, "grant bytes mismatch");
    Check(layout.controlSourceOffset == 2818873344ULL,
        "control source offset mismatch");
    Check(layout.failureOffset == 2818875392ULL,
        "failure offset mismatch");
    Check(layout.controlSourceBytes == 2048U, "control source bytes mismatch");
    Check(layout.failureBytes == 2048U, "failure bytes mismatch");
    Check(layout.outputOffset == 2818877440ULL,
        "target output offset mismatch");
    Check(layout.outputOffset == layout.failureOffset + layout.failureBytes,
        "output offset mismatch");
    Check(layout.outputBytes == 58720256U, "output bytes mismatch");
    Check(layout.totalBytes == 2879389696ULL, "target total bytes mismatch");
}

void TestSmallAndInvalidLayouts()
{
    using namespace TileXRMoonEp;
    CombineV2Layout layout {};
    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        8, 3584, 16, 128, TILEXR_MOONEP_DTYPE_BFLOAT16, &layout),
        TILEXR_MOONEP_SUCCESS, "small layout");
    Check(layout.expertBytes == 917504U, "small expert bytes mismatch");
    Check(layout.totalBytes == 4194304U, "small total bytes mismatch");
    Check(layout.totalBytes % kCombineV2RegistrationAlignmentBytes == 0U,
        "workspace is not registration aligned");
    Check(layout.outputOffset >= layout.failureOffset + layout.failureBytes &&
        layout.outputOffset + layout.outputBytes <= layout.totalBytes,
        "small output overlaps control workspace");

    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        256, 1024, 4, 2040, TILEXR_MOONEP_DTYPE_BFLOAT16, &layout),
        TILEXR_MOONEP_SUCCESS, "PR113 hidden layout");
    Check(layout.rowBytes == 2048U && layout.slots == 2040U,
        "PR113 hidden layout mismatch");
    Check(layout.outputBytes == 524288U,
        "PR113 hidden output bytes mismatch");
    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        256, 1, 4, 2040, TILEXR_MOONEP_DTYPE_FLOAT32, &layout),
        TILEXR_MOONEP_SUCCESS, "PR113 route-weight layout");
    Check(layout.rowBytes == sizeof(float), "route-weight row size mismatch");
    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        8192, 3584, 16, 131071, TILEXR_MOONEP_DTYPE_BFLOAT16, &layout),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT, "undersized slots");
    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        8192, 3584, 16, 131072, TILEXR_MOONEP_DTYPE_FLOAT16, &layout),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT, "unsupported dtype");
    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        8, 3584, 16, 128, TILEXR_MOONEP_DTYPE_BFLOAT16, nullptr),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT, "null layout");
}

} // namespace

int main()
{
    TestTargetLayout();
    TestSmallAndInvalidLayouts();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
