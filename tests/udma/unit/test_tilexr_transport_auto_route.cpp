#include <cstdint>
#include <iostream>

#include "tilexr_transport.h"

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
    TileXR::CommArgs args = {};
    if (directUrmaAvailable) {
        args.extraFlag |= TileXR::ExtraFlag::UDMA;
        args.udmaInfoPtr = reinterpret_cast<GM_ADDR>(0x100000);
        args.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(0x200000);
    }
    return args;
}

TileXR::CommArgs MakeArgsMissingInfo()
{
    TileXR::CommArgs args = {};
    args.extraFlag |= TileXR::ExtraFlag::UDMA;
    args.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(0x200000);
    return args;
}

TileXR::CommArgs MakeArgsMissingRegistry()
{
    TileXR::CommArgs args = {};
    args.extraFlag |= TileXR::ExtraFlag::UDMA;
    args.udmaInfoPtr = reinterpret_cast<GM_ADDR>(0x100000);
    return args;
}

void TestAutoUsesMemoryForNullArgs()
{
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(nullptr, 64ULL * 1024ULL * 1024ULL),
             TileXR::TileXRTransportKind::MEMORY);
}

void TestAutoUsesMemoryForZeroBytes()
{
    auto args = MakeArgs(true);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, 0),
             TileXR::TileXRTransportKind::MEMORY);
}

void TestAutoUsesMemoryBelowFourMiB()
{
    auto args = MakeArgs(true);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, TileXR::TILEXR_AUTO_DIRECT_URMA_THRESHOLD_BYTES - 1),
             TileXR::TileXRTransportKind::MEMORY);
}

void TestAutoUsesDirectUrmaAtFourMiB()
{
    auto args = MakeArgs(true);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, TileXR::TILEXR_AUTO_DIRECT_URMA_THRESHOLD_BYTES),
             TileXR::TileXRTransportKind::DIRECT_URMA);
}

void TestAutoUsesDirectUrmaAboveFourMiB()
{
    auto args = MakeArgs(true);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, 8ULL * 1024ULL * 1024ULL),
             TileXR::TileXRTransportKind::DIRECT_URMA);
}

void TestAutoFallsBackToMemoryWhenUrmaUnavailable()
{
    auto args = MakeArgs(false);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, 64ULL * 1024ULL * 1024ULL),
             TileXR::TileXRTransportKind::MEMORY);
}

void TestAutoFallsBackToMemoryWhenUrmaInfoMissing()
{
    auto args = MakeArgsMissingInfo();
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, 64ULL * 1024ULL * 1024ULL),
             TileXR::TileXRTransportKind::MEMORY);
}

void TestAutoFallsBackToMemoryWhenUrmaRegistryMissing()
{
    auto args = MakeArgsMissingRegistry();
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, 64ULL * 1024ULL * 1024ULL),
             TileXR::TileXRTransportKind::MEMORY);
}

void TestAutoPutWrapperReturnsSelectedRouteOnHost()
{
    auto args = MakeArgs(true);
    auto* localSrc = reinterpret_cast<const int32_t*>(0x300000);
    CHECK_EQ(TileXR::TileXRPutAutoNbi<int32_t>(&args, 1, localSrc, 0, 8ULL * 1024ULL * 1024ULL),
             TileXR::TileXRTransportKind::DIRECT_URMA);
}

void TestAutoGetWrapperReturnsSelectedRouteOnHost()
{
    auto args = MakeArgs(false);
    auto* localDst = reinterpret_cast<int32_t*>(0x400000);
    CHECK_EQ(TileXR::TileXRGetAutoNbi<int32_t>(&args, 1, localDst, 0, 8ULL * 1024ULL * 1024ULL),
             TileXR::TileXRTransportKind::MEMORY);
}

} // namespace

int main()
{
    TestAutoUsesMemoryForNullArgs();
    TestAutoUsesMemoryForZeroBytes();
    TestAutoUsesMemoryBelowFourMiB();
    TestAutoUsesDirectUrmaAtFourMiB();
    TestAutoUsesDirectUrmaAboveFourMiB();
    TestAutoFallsBackToMemoryWhenUrmaUnavailable();
    TestAutoFallsBackToMemoryWhenUrmaInfoMissing();
    TestAutoFallsBackToMemoryWhenUrmaRegistryMissing();
    TestAutoPutWrapperReturnsSelectedRouteOnHost();
    TestAutoGetWrapperReturnsSelectedRouteOnHost();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR transport auto route checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR transport auto route checks passed" << std::endl;
    return 0;
}
