#include <cstdlib>
#include <cstdint>
#include <iostream>

#include "ep_transport_route.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

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

TileXR::CommArgs MakeArgs(bool directUrmaAvailable)
{
    TileXR::CommArgs args {};
    args.rankSize = 2;
    args.localRankSize = 1;
    if (directUrmaAvailable) {
        args.extraFlag |= TileXR::ExtraFlag::UDMA;
        args.udmaInfoPtr = reinterpret_cast<GM_ADDR>(0x100000);
        args.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(0x200000);
    }
    return args;
}

TileXR::CommArgs MakeUdmaCapableArgs()
{
    TileXR::CommArgs args {};
    args.rankSize = 2;
    args.localRankSize = 1;
    args.extraFlag |= TileXR::ExtraFlag::UDMA;
    args.udmaInfoPtr = reinterpret_cast<GM_ADDR>(0x100000);
    return args;
}

void SetTransportMode(const char *value)
{
#ifdef _WIN32
    _putenv_s("TILEXR_TRANSPORT_MODE", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("TILEXR_TRANSPORT_MODE");
    } else {
        setenv("TILEXR_TRANSPORT_MODE", value, 1);
    }
#endif
}

void TestParseTransportMode()
{
    TileXREp::EpTransportMode mode = TileXREp::EpTransportMode::MEMORY;
    CHECK_EQ(TileXREp::TileXREpParseTransportMode(nullptr, &mode), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(mode, TileXREp::EpTransportMode::AUTO);

    CHECK_EQ(TileXREp::TileXREpParseTransportMode("", &mode), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(mode, TileXREp::EpTransportMode::AUTO);
    CHECK_EQ(TileXREp::TileXREpParseTransportMode("auto", &mode), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(mode, TileXREp::EpTransportMode::AUTO);
    CHECK_EQ(TileXREp::TileXREpParseTransportMode("memory", &mode), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(mode, TileXREp::EpTransportMode::MEMORY);
    CHECK_EQ(TileXREp::TileXREpParseTransportMode("direct_urma", &mode), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(mode, TileXREp::EpTransportMode::DIRECT_URMA);
    CHECK_EQ(TileXREp::TileXREpParseTransportMode("invalid", &mode), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CHECK_EQ(TileXREp::TileXREpParseTransportMode("auto", nullptr), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestResolveForcedTransport()
{
    const TileXR::CommArgs noUdmaArgs = MakeArgs(false);
    const TileXR::CommArgs udmaArgs = MakeArgs(true);
    TileXR::TileXRTransportKind transport = TileXR::TileXRTransportKind::DIRECT_URMA;

    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::MEMORY, noUdmaArgs, 4096,
        &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::MEMORY);

    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::DIRECT_URMA, noUdmaArgs, 4096,
        &transport), TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::DIRECT_URMA, udmaArgs, 4096,
        &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::DIRECT_URMA);
    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::AUTO, udmaArgs, 4096,
        &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::MEMORY);
    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::AUTO, udmaArgs,
        TileXR::TILEXR_AUTO_CROSS_NODE_DIRECT_URMA_THRESHOLD_BYTES, &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::DIRECT_URMA);
    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::AUTO, noUdmaArgs, 4096,
        &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::MEMORY);
    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::AUTO, udmaArgs, 4096,
        nullptr), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestSameNodeUsesFourMiBThreshold()
{
    TileXR::CommArgs args = MakeArgs(true);
    args.localRankSize = args.rankSize;
    TileXR::TileXRTransportKind transport = TileXR::TileXRTransportKind::DIRECT_URMA;

    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::AUTO, args,
        TileXR::TILEXR_AUTO_SAME_NODE_DIRECT_URMA_THRESHOLD_BYTES - 1, &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::MEMORY);
    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::AUTO, args,
        TileXR::TILEXR_AUTO_SAME_NODE_DIRECT_URMA_THRESHOLD_BYTES, &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::DIRECT_URMA);
    CHECK_EQ(TileXREp::TileXREpResolveTransport(TileXREp::EpTransportMode::DIRECT_URMA, args,
        8ULL * 1024ULL * 1024ULL, &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::DIRECT_URMA);
}

void TestResolveTransportFromEnvironment()
{
    const TileXR::CommArgs noUdmaArgs = MakeArgs(false);
    TileXR::TileXRTransportKind transport = TileXR::TileXRTransportKind::DIRECT_URMA;

    SetTransportMode(nullptr);
    CHECK_EQ(TileXREp::TileXREpResolveTransportFromEnv(noUdmaArgs, 8ULL * 1024ULL * 1024ULL,
        &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::MEMORY);

    SetTransportMode("auto");
    CHECK_EQ(TileXREp::TileXREpResolveTransportFromEnv(noUdmaArgs, 8ULL * 1024ULL * 1024ULL,
        &transport), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(transport, TileXR::TileXRTransportKind::MEMORY);

    SetTransportMode("direct_urma");
    CHECK_EQ(TileXREp::TileXREpResolveTransportFromEnv(noUdmaArgs, 8ULL * 1024ULL * 1024ULL,
        &transport), TileXR::TILEXR_ERROR_NOT_INITIALIZED);

    SetTransportMode("invalid");
    CHECK_EQ(TileXREp::TileXREpResolveTransportFromEnv(noUdmaArgs, 8ULL * 1024ULL * 1024ULL,
        &transport), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    SetTransportMode(nullptr);
}

void TestDemoRegistrationDecisionUsesCapabilityBeforeRegistry()
{
    const TileXR::CommArgs noUdmaArgs = MakeArgs(false);
    const TileXR::CommArgs capableArgs = MakeUdmaCapableArgs();

    CHECK_EQ(TileXREp::TileXREpShouldRegisterWorkspace(TileXREp::EpTransportMode::AUTO, capableArgs), true);
    CHECK_EQ(TileXREp::TileXREpShouldRegisterWorkspace(TileXREp::EpTransportMode::DIRECT_URMA, capableArgs), true);
    CHECK_EQ(TileXREp::TileXREpShouldRegisterWorkspace(TileXREp::EpTransportMode::MEMORY, capableArgs), false);
    CHECK_EQ(TileXREp::TileXREpShouldRegisterWorkspace(TileXREp::EpTransportMode::AUTO, noUdmaArgs), false);
}

} // namespace

int main()
{
    TestParseTransportMode();
    TestResolveForcedTransport();
    TestSameNodeUsesFourMiBThreshold();
    TestResolveTransportFromEnvironment();
    TestDemoRegistrationDecisionUsesCapabilityBeforeRegistry();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR EP transport route checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR EP transport route checks passed" << std::endl;
    return 0;
}
