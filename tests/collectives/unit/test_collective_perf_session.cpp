#include <cstdint>
#include <iostream>

#include "tilexr_collectives_perf.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

template <typename T>
void CheckEq(const char *label, T actual, T expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckTrue(const char *label, bool condition)
{
    if (!condition) {
        std::cerr << label << " failed" << std::endl;
        ++g_failures;
    }
}

void TestCollectivePerfSessionLifecycle()
{
    TileXRCollectivePerfSession session = nullptr;
    CheckEq("create rejects null config", TileXRCollectivePerfSessionCreate(nullptr, &session),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    TileXRCollectivePerfConfig config {};
    config.enabled = 1;
    config.outputDir = "/tmp/tilexr_perf_session_test";
    config.emitAiPrompt = 1;
    config.sampleEveryN = 1;

    CheckEq("create succeeds", TileXRCollectivePerfSessionCreate(&config, &session), TileXR::TILEXR_SUCCESS);
    CheckTrue("session created", session != nullptr);
    CheckEq("set active session succeeds", TileXRCollectivePerfSetActiveSession(session), TileXR::TILEXR_SUCCESS);
    CheckEq("clear active session succeeds", TileXRCollectivePerfSetActiveSession(nullptr), TileXR::TILEXR_SUCCESS);
    CheckEq("destroy succeeds", TileXRCollectivePerfSessionDestroy(session), TileXR::TILEXR_SUCCESS);
}

} // namespace

int main()
{
    TestCollectivePerfSessionLifecycle();
    return g_failures == 0 ? 0 : 1;
}
