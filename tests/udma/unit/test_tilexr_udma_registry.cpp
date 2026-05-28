#include <cstdint>
#include <iostream>

#include "tilexr_udma_reg.h"

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
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

void TestRemoteAddressCalculation()
{
    TileXR::TileXRUDMARegistry registry = {};
    registry.magic = TileXR::TILEXR_UDMA_REGISTRY_MAGIC;
    registry.version = TileXR::TILEXR_UDMA_REGISTRY_VERSION;
    registry.regionCount = 1;
    registry.rankSize = 2;
    registry.regions[0].base = reinterpret_cast<GM_ADDR>(0x100000);
    registry.regions[0].bytes = 4096;
    registry.regions[1].base = reinterpret_cast<GM_ADDR>(0x200000);
    registry.regions[1].bytes = 2048;

    CHECK_TRUE(TileXR::UDMARegistryValid(&registry, 2));
    CHECK_TRUE(TileXR::UDMARegionContains(&registry, 0, 128, 256));
    CHECK_TRUE(TileXR::UDMARegionContains(&registry, 1, 1024, 1024));
    CHECK_TRUE(!TileXR::UDMARegionContains(&registry, 1, 1024, 1025));
    CHECK_TRUE(!TileXR::UDMARegionContains(&registry, 2, 0, 1));
    CHECK_EQ(reinterpret_cast<uintptr_t>(TileXR::UDMARemoteAddr(&registry, 1, 64)),
             static_cast<uintptr_t>(0x200040));
}

} // namespace

int main()
{
    TestRemoteAddressCalculation();
    if (g_failures != 0) {
        std::cerr << g_failures << " registry checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA registry checks passed" << std::endl;
    return 0;
}
