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
    Check(sizeof(MoonEpCombineV2ProfileRecord) == 320U,
        "profile record size mismatch");
    Check(kMoonEpCombineV2ProfileTimePointCount == 22U,
        "profile point count mismatch");

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
    Check(layout.grantBytes == 229376U, "grant bytes mismatch");
    Check(layout.controlSourceBytes == 2048U, "control source bytes mismatch");
    Check(layout.failureBytes == 2048U, "failure bytes mismatch");
    Check(layout.totalBytes == 2820669440ULL, "target total bytes mismatch");
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

    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        16, 3584, 16, 256, TILEXR_MOONEP_DTYPE_BFLOAT16, &layout),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT, "unsupported shape");
    CheckStatus(TileXRMoonEpBuildCombineV2Layout(
        8192, 3584, 16, 131071, TILEXR_MOONEP_DTYPE_BFLOAT16, &layout),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT, "slot mismatch");
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
