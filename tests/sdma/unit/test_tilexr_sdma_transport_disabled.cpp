#include <cstdlib>
#include <iostream>
#include <string>

#include "acl/acl_rt.h"
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

class EnvGuard {
public:
    EnvGuard()
    {
        const char* value = std::getenv("TILEXR_ENABLE_SDMA");
        if (value != nullptr) {
            hadValue_ = true;
            value_ = value;
        }
    }

    ~EnvGuard()
    {
        if (hadValue_) {
            setenv("TILEXR_ENABLE_SDMA", value_.c_str(), 1);
        } else {
            unsetenv("TILEXR_ENABLE_SDMA");
        }
    }

    EnvGuard(const EnvGuard&) = delete;
    EnvGuard& operator=(const EnvGuard&) = delete;

private:
    bool hadValue_ = false;
    std::string value_;
};

void TestEnvDisabledSkipsInitialization()
{
    EnvGuard env;
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
    EnvGuard env;
    setenv("TILEXR_ENABLE_SDMA", "0", 1);
    TileXR::TileXRSDMATransport transport;
    TileXR::TileXRSDMATransportOptions options {};
    options.devId = 0;
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), TileXR::SDMAInitStatus::DISABLED_BY_ENV);
}

TileXR::SDMAInitStatus ExpectedUnavailableStatus()
{
    return TileXR::detail::ClassifySDMABackend(aclrtGetSocName()) ==
            TileXR::detail::SDMABackendKind::A5_DIRECT
        ? TileXR::SDMAInitStatus::INIT_FAILED
        : TileXR::SDMAInitStatus::PTO_UNAVAILABLE;
}

void TestBackendClassification()
{
    using TileXR::detail::ClassifySDMABackend;
    using TileXR::detail::SDMABackendKind;
    CHECK_EQ(ClassifySDMABackend("Ascend950PR_9589"), SDMABackendKind::A5_DIRECT);
    CHECK_EQ(ClassifySDMABackend("Ascend910B3"), SDMABackendKind::PTO);
    CHECK_EQ(ClassifySDMABackend("Ascend910A"), SDMABackendKind::PTO);
    CHECK_EQ(ClassifySDMABackend("Ascend910_9391"), SDMABackendKind::PTO);
    CHECK_EQ(ClassifySDMABackend("Ascend310P"), SDMABackendKind::UNSUPPORTED);
    CHECK_EQ(ClassifySDMABackend(nullptr), SDMABackendKind::UNSUPPORTED);
}

void TestEnvOneReportsBackendUnavailable()
{
    EnvGuard env;
    setenv("TILEXR_ENABLE_SDMA", "1", 1);
    TileXR::TileXRSDMATransport transport;
    TileXR::TileXRSDMATransportOptions options {};
    options.devId = 0;
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), ExpectedUnavailableStatus());
}

void TestSameInstanceTransitionsResetState()
{
    EnvGuard env;
    TileXR::TileXRSDMATransport transport;
    TileXR::TileXRSDMATransportOptions options {};
    options.devId = 0;

    unsetenv("TILEXR_ENABLE_SDMA");
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), TileXR::SDMAInitStatus::DISABLED_BY_ENV);

    setenv("TILEXR_ENABLE_SDMA", "1", 1);
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), ExpectedUnavailableStatus());

    transport.Shutdown();
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), ExpectedUnavailableStatus());

    unsetenv("TILEXR_ENABLE_SDMA");
    CHECK_EQ(transport.Init(options), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!transport.IsAvailable());
    CHECK_TRUE(transport.GetWorkspaceDev() == nullptr);
    CHECK_EQ(transport.GetLastStatus(), TileXR::SDMAInitStatus::DISABLED_BY_ENV);
}

} // namespace

int main()
{
    TestEnvDisabledSkipsInitialization();
    TestEnvZeroSkipsInitialization();
    TestBackendClassification();
    TestEnvOneReportsBackendUnavailable();
    TestSameInstanceTransitionsResetState();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA transport disabled checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA transport disabled checks passed" << std::endl;
    return 0;
}
