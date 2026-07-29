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

TileXR::CommArgs MakeCrossNodeArgs(bool directUrmaAvailable)
{
    TileXR::CommArgs args = MakeArgs(directUrmaAvailable);
    args.rankSize = 2;
    args.localRankSize = 1;
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

void TestCrossNodeUsesMemoryBelow128KiB()
{
    auto args = MakeCrossNodeArgs(true);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(
        &args, TileXR::TILEXR_AUTO_CROSS_NODE_DIRECT_URMA_THRESHOLD_BYTES - 1),
        TileXR::TileXRTransportKind::MEMORY);
}

void TestCrossNodeUsesDirectUrmaAt128KiB()
{
    auto args = MakeCrossNodeArgs(true);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(
        &args, TileXR::TILEXR_AUTO_CROSS_NODE_DIRECT_URMA_THRESHOLD_BYTES),
        TileXR::TileXRTransportKind::DIRECT_URMA);
}

void TestAutoKeepsZeroBytesOnMemoryForCrossNode()
{
    auto args = MakeCrossNodeArgs(true);
    CHECK_EQ(TileXR::TileXRSelectAutoTransport(&args, 0),
             TileXR::TileXRTransportKind::MEMORY);
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

void TestDirectUrmaPeerRoutabilityMatchesAllocatedResources()
{
    auto args = MakeArgs(true);
    args.rank = 0;
    args.rankSize = 4;
    args.localRankSize = 2;
    CHECK_EQ(TileXR::TileXRDirectUrmaPeerRoutable(&args, 0), false);
    CHECK_EQ(TileXR::TileXRDirectUrmaPeerRoutable(&args, 1), false);
    CHECK_EQ(TileXR::TileXRDirectUrmaPeerRoutable(&args, 2), true);
    CHECK_EQ(TileXR::TileXRDirectUrmaPeerRoutable(&args, 3), true);

    args.rankSize = 2;
    args.localRankSize = 2;
    CHECK_EQ(TileXR::TileXRDirectUrmaPeerRoutable(&args, 1), true);

    args.localRankSize = 0;
    CHECK_EQ(TileXR::TileXRDirectUrmaPeerRoutable(&args, 1), false);

    args = MakeArgs(false);
    args.rank = 0;
    args.rankSize = 2;
    args.localRankSize = 1;
    CHECK_EQ(TileXR::TileXRDirectUrmaPeerRoutable(&args, 1), false);
}

} // namespace

int main()
{
    TestAutoUsesMemoryForNullArgs();
    TestAutoUsesMemoryForZeroBytes();
    TestAutoUsesMemoryBelowFourMiB();
    TestAutoUsesDirectUrmaAtFourMiB();
    TestAutoUsesDirectUrmaAboveFourMiB();
    TestCrossNodeUsesMemoryBelow128KiB();
    TestCrossNodeUsesDirectUrmaAt128KiB();
    TestAutoKeepsZeroBytesOnMemoryForCrossNode();
    TestAutoFallsBackToMemoryWhenUrmaUnavailable();
    TestAutoFallsBackToMemoryWhenUrmaInfoMissing();
    TestAutoFallsBackToMemoryWhenUrmaRegistryMissing();
    TestDirectUrmaPeerRoutabilityMatchesAllocatedResources();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR transport auto route checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR transport auto route checks passed" << std::endl;
    return 0;
}
