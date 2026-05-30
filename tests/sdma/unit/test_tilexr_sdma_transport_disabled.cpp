#include <cstdlib>
#include <iostream>

#include "sdma/tilexr_sdma_transport.h"
#include "tilexr_sdma_types.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << static_cast<int>(lhsValue) << " vs " << static_cast<int>(rhsValue) << ")" \
                      << std::endl; \
            ++g_failures; \
        } \
    } while (0)

void TestEnvDisabledSkipsInitialization()
{
    unsetenv("TILEXR_ENABLE_SDMA");
    TileXR::TileXRSDMATransport transport;
    TileXR::TileXRSDMATransportOptions options {};
    options.devId = 0;
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), TileXR::SDMAInitStatus::DISABLED_BY_ENV);
}

void TestEnvZeroSkipsInitialization()
{
    setenv("TILEXR_ENABLE_SDMA", "0", 1);
    TileXR::TileXRSDMATransport transport;
    TileXR::TileXRSDMATransportOptions options {};
    options.devId = 0;
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), TileXR::SDMAInitStatus::DISABLED_BY_ENV);
    unsetenv("TILEXR_ENABLE_SDMA");
}

} // namespace

int main()
{
    TestEnvDisabledSkipsInitialization();
    TestEnvZeroSkipsInitialization();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA transport disabled checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA transport disabled checks passed" << std::endl;
    return 0;
}
