#include <iostream>

#include "tools/socket/tilexr_sock_exchange_layout.h"

namespace {

int g_failures = 0;

#define CHECK_EQ(lhs, rhs) \
    do { \
        const auto lhsValue = (lhs); \
        const auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " \
                      << lhsValue << " != " << rhsValue << std::endl; \
            ++g_failures; \
        } \
    } while (0)

void Test512RanksUseEightCabinetLeaders()
{
    for (int rank = 0; rank < 512; ++rank) {
        const auto layout = TileXR::BuildSockExchangeGroupLayout(rank, 512, 64);
        CHECK_EQ(layout.groupCount, 8);
        CHECK_EQ(layout.groupIndex, rank / 64);
        CHECK_EQ(layout.groupBegin, (rank / 64) * 64);
        CHECK_EQ(layout.groupEnd, (rank / 64 + 1) * 64);
        CHECK_EQ(layout.groupLeader, (rank / 64) * 64);
    }
}

void Test1024RanksUseSixteenCabinetLeaders()
{
    const auto first = TileXR::BuildSockExchangeGroupLayout(0, 1024, 64);
    const auto last = TileXR::BuildSockExchangeGroupLayout(1023, 1024, 64);
    CHECK_EQ(first.groupCount, 16);
    CHECK_EQ(first.groupLeader, 0);
    CHECK_EQ(last.groupCount, 16);
    CHECK_EQ(last.groupBegin, 960);
    CHECK_EQ(last.groupEnd, 1024);
    CHECK_EQ(last.groupLeader, 960);
}

void TestTailGroupIsClampedToRankSize()
{
    const auto layout = TileXR::BuildSockExchangeGroupLayout(129, 130, 64);
    CHECK_EQ(layout.groupCount, 3);
    CHECK_EQ(layout.groupIndex, 2);
    CHECK_EQ(layout.groupBegin, 128);
    CHECK_EQ(layout.groupEnd, 130);
    CHECK_EQ(layout.groupLeader, 128);
}

void TestInvalidInputsDisableLayout()
{
    CHECK_EQ(TileXR::BuildSockExchangeGroupLayout(-1, 512, 64).groupCount, 0);
    CHECK_EQ(TileXR::BuildSockExchangeGroupLayout(512, 512, 64).groupCount, 0);
    CHECK_EQ(TileXR::BuildSockExchangeGroupLayout(0, 512, 0).groupCount, 0);
}

} // namespace

int main()
{
    Test512RanksUseEightCabinetLeaders();
    Test1024RanksUseSixteenCabinetLeaders();
    TestTailGroupIsClampedToRankSize();
    TestInvalidInputsDisableLayout();
    if (g_failures != 0) {
        std::cerr << g_failures << " test(s) failed" << std::endl;
        return 1;
    }
    std::cout << "tilexr socket exchange layout tests passed" << std::endl;
    return 0;
}
