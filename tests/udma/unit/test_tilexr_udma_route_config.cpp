#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "udma/tilexr_udma_config.h"

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
        const auto lhsValue = (lhs); \
        const auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs << std::endl; \
            ++g_failures; \
        } \
    } while (0)

bool ParseOk(const char* text, TileXR::UDMAQpConfig& config)
{
    std::string error;
    const auto status = TileXR::ParseUDMAQpRouteSpec(text, config, &error);
    if (status != TileXR::UDMAQpConfigParseStatus::SUCCESS) {
        std::cerr << "unexpected parse error for '" << (text == nullptr ? "<null>" : text)
                  << "': " << error << std::endl;
        ++g_failures;
        return false;
    }
    return true;
}

void ExpectInvalid(const char* text)
{
    TileXR::UDMAQpConfig config;
    std::string error;
    CHECK_EQ(TileXR::ParseUDMAQpRouteSpec(text, config, &error),
             TileXR::UDMAQpConfigParseStatus::INVALID);
    CHECK_TRUE(!error.empty());
    CHECK_TRUE(!config.explicitConfig);
    CHECK_TRUE(config.routes.empty());
}

void TestLegacyConfig()
{
    for (const char* text : {static_cast<const char*>(nullptr), "", "  \t\r\n "}) {
        TileXR::UDMAQpConfig config;
        if (!ParseOk(text, config)) {
            continue;
        }
        CHECK_TRUE(!config.explicitConfig);
        CHECK_TRUE(config.routes.empty());
        CHECK_EQ(TileXR::UDMAQpConfigQpCount(config), 1U);

        const auto wire = TileXR::BuildUDMAQpConfigWireDescriptor(
            config, TileXR::UDMAQpConfigParseStatus::SUCCESS);
        CHECK_EQ(wire.qpCount, 1U);
        CHECK_EQ(wire.routeRules[0].selectorKind, 0U);
        CHECK_TRUE(TileXR::ValidateUDMAQpConfigWireDescriptor(wire));
    }
}

void TestNormalizedRules()
{
    TileXR::UDMAQpConfig config;
    CHECK_TRUE(ParseOk(" topology ", config));
    CHECK_TRUE(config.explicitConfig);
    CHECK_EQ(config.routes.size(), static_cast<size_t>(1));
    CHECK_EQ(config.routes[0].selector, TileXR::UDMAQpRouteSelector::TOPOLOGY);
    CHECK_EQ(config.routes[0].value, 0U);

    CHECK_TRUE(ParseOk(" port_count:6 , port_count:2 ", config));
    CHECK_EQ(TileXR::UDMAQpConfigQpCount(config), 2U);
    CHECK_EQ(config.routes[0].selector, TileXR::UDMAQpRouteSelector::PORT_COUNT);
    CHECK_EQ(config.routes[0].value, 6U);
    CHECK_EQ(config.routes[1].value, 2U);

    CHECK_TRUE(ParseOk("port_count:6,port_count:6,port_count:2", config));
    CHECK_EQ(config.routes.size(), static_cast<size_t>(3));
    CHECK_EQ(config.routes[0].value, config.routes[1].value);

    CHECK_TRUE(ParseOk(
        "topology,port_count:1,port_count:2,port_count:3,"
        "port_count:4,port_count:5,port_count:6,port_count:7", config));
    CHECK_EQ(TileXR::UDMAQpConfigQpCount(config), 8U);
}

void TestInvalidRules()
{
    ExpectInvalid(",");
    ExpectInvalid(",topology");
    ExpectInvalid("topology,");
    ExpectInvalid("topology,,port_count:2");
    ExpectInvalid("unknown");
    ExpectInvalid("port_count");
    ExpectInvalid("port_count:");
    ExpectInvalid("port_count:0");
    ExpectInvalid("port_count:-1");
    ExpectInvalid("port_count:+1");
    ExpectInvalid("port_count: 2");
    ExpectInvalid("port_count:2x");
    ExpectInvalid("port_count:4294967296");
    ExpectInvalid(
        "topology,topology,topology,topology,topology,topology,topology,topology,topology");
}

void TestWireDescriptor()
{
    TileXR::UDMAQpConfig config;
    CHECK_TRUE(ParseOk("port_count:6,port_count:6,port_count:2", config));
    const auto wire = TileXR::BuildUDMAQpConfigWireDescriptor(
        config, TileXR::UDMAQpConfigParseStatus::SUCCESS);
    CHECK_TRUE(TileXR::ValidateUDMAQpConfigWireDescriptor(wire));
    CHECK_EQ(wire.qpCount, 3U);
    CHECK_EQ(wire.routeRules[0].selectorKind,
             static_cast<uint32_t>(TileXR::UDMAQpRouteSelector::PORT_COUNT));
    CHECK_EQ(wire.routeRules[1].selectorValue, 6U);
    CHECK_EQ(wire.routeRules[2].selectorValue, 2U);

    TileXR::UDMAQpConfig roundTrip;
    CHECK_TRUE(TileXR::UDMAQpConfigFromWireDescriptor(wire, roundTrip));
    CHECK_TRUE(roundTrip.explicitConfig);
    CHECK_EQ(roundTrip.routes.size(), config.routes.size());
    CHECK_EQ(roundTrip.routes[1].value, config.routes[1].value);

    auto mismatch = wire;
    mismatch.routeRules[1].selectorValue = 2;
    CHECK_TRUE(!TileXR::UDMAQpConfigWireDescriptorsEqual(wire, mismatch));
    CHECK_TRUE(TileXR::UDMAQpConfigWireDescriptorsEqual(wire, wire));

    auto invalid = wire;
    invalid.qpCount = 9;
    CHECK_TRUE(!TileXR::ValidateUDMAQpConfigWireDescriptor(invalid));
    invalid = wire;
    invalid.routeRules[7].selectorKind = 99;
    CHECK_TRUE(!TileXR::ValidateUDMAQpConfigWireDescriptor(invalid));

    const auto parseFailure = TileXR::BuildUDMAQpConfigWireDescriptor(
        config, TileXR::UDMAQpConfigParseStatus::INVALID);
    CHECK_EQ(parseFailure.qpCount, 0U);
    CHECK_TRUE(TileXR::ValidateUDMAQpConfigWireDescriptor(parseFailure));
    CHECK_TRUE(!TileXR::UDMAQpConfigFromWireDescriptor(parseFailure, roundTrip));
}

void TestEnvironmentConfig()
{
    CHECK_EQ(setenv(TileXR::TILEXR_UDMA_QP_ROUTE_SPEC_ENV, "port_count:6,topology", 1), 0);
    TileXR::UDMAQpConfig config;
    std::string error;
    CHECK_EQ(TileXR::LoadUDMAQpConfigFromEnv(config, &error),
             TileXR::UDMAQpConfigParseStatus::SUCCESS);
    CHECK_TRUE(config.explicitConfig);
    CHECK_EQ(config.routes.size(), static_cast<size_t>(2));
    CHECK_EQ(config.routes[0].value, 6U);
    CHECK_EQ(config.routes[1].selector, TileXR::UDMAQpRouteSelector::TOPOLOGY);
    CHECK_EQ(unsetenv(TileXR::TILEXR_UDMA_QP_ROUTE_SPEC_ENV), 0);
}

} // namespace

int main()
{
    TestLegacyConfig();
    TestNormalizedRules();
    TestInvalidRules();
    TestWireDescriptor();
    TestEnvironmentConfig();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA route configuration checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA route configuration checks passed" << std::endl;
    return 0;
}
