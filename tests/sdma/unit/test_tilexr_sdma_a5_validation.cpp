#include <cstdint>
#include <iostream>

#include "sdma/tilexr_sdma_a5_backend.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

TileXR::detail::A5BuiltinChannelInfo ValidChannel()
{
    TileXR::detail::A5BuiltinChannelInfo channel {};
    channel.sqHead = 1U;
    channel.sqTail = 2U;
    channel.sqBase = 0x1000U;
    channel.sqRegisterBase = 0x2000U;
    channel.sqDepth = 8U;
    channel.sqId = 3U;
    channel.cqId = 4U;
    channel.logicalCqId = 5U;
    channel.streamId = 6U;
    channel.deviceId = 7U;
    return channel;
}

TileXR::detail::A5HostChannelIdentity ValidIdentity()
{
    return {6U, 3U, 4U, 5U, 7U};
}

void TestCompleteAndExpectedPartialClassification()
{
    using namespace TileXR::detail;
    A5BuiltinChannelInfo channel = ValidChannel();
    const A5HostChannelIdentity identity = ValidIdentity();
    CHECK_TRUE(ClassifyA5QueryResult(0, 1U, 48U, channel, identity) ==
               A5QueryResultKind::COMPLETE);
    channel.sqRegisterBase = 0U;
    CHECK_TRUE(ClassifyA5QueryResult(TILEXR_SDMA_A5_EXPECTED_QUERY_STATUS,
                                    0U, 0U, channel, identity) ==
               A5QueryResultKind::EXPECTED_PARTIAL);
}

void TestPartialClassificationFailsClosed()
{
    using namespace TileXR::detail;
    A5BuiltinChannelInfo channel = ValidChannel();
    channel.sqRegisterBase = 0U;
    const A5HostChannelIdentity identity = ValidIdentity();
    CHECK_TRUE(ClassifyA5QueryResult(507019, 0U, 0U, channel, identity) ==
               A5QueryResultKind::INVALID);
    CHECK_TRUE(ClassifyA5QueryResult(TILEXR_SDMA_A5_EXPECTED_QUERY_STATUS,
                                    1U, 0U, channel, identity) ==
               A5QueryResultKind::INVALID);
    channel.sqTail = channel.sqDepth;
    CHECK_TRUE(ClassifyA5QueryResult(TILEXR_SDMA_A5_EXPECTED_QUERY_STATUS,
                                    0U, 0U, channel, identity) ==
               A5QueryResultKind::INVALID);
    channel = ValidChannel();
    channel.sqRegisterBase = 0U;
    channel.streamId += 1U;
    CHECK_TRUE(ClassifyA5QueryResult(TILEXR_SDMA_A5_EXPECTED_QUERY_STATUS,
                                    0U, 0U, channel, identity) ==
               A5QueryResultKind::INVALID);
    channel = ValidChannel();
    channel.sqHead = 0U;
    channel.sqTail = channel.sqDepth - 1U;
    CHECK_TRUE(ClassifyA5QueryResult(0, 1U, TILEXR_SDMA_A5_CHANNEL_COUNT,
                                    channel, identity) == A5QueryResultKind::INVALID);
}

void TestCompleteClassificationRequiresFinishedHeader()
{
    using namespace TileXR::detail;
    const A5BuiltinChannelInfo channel = ValidChannel();
    const A5HostChannelIdentity identity = ValidIdentity();
    CHECK_TRUE(ClassifyA5QueryResult(0, 0U, TILEXR_SDMA_A5_CHANNEL_COUNT,
                                    channel, identity) == A5QueryResultKind::INVALID);
    CHECK_TRUE(ClassifyA5QueryResult(0, 1U, TILEXR_SDMA_A5_CHANNEL_COUNT - 1U,
                                    channel, identity) == A5QueryResultKind::INVALID);
    CHECK_TRUE(ClassifyA5QueryResult(0, 1U, 0U, channel, identity) ==
               A5QueryResultKind::INVALID);
}

} // namespace

int main()
{
    TestCompleteAndExpectedPartialClassification();
    TestPartialClassificationFailsClosed();
    TestCompleteClassificationRequiresFinishedHeader();
    if (g_failures != 0) {
        std::cerr << g_failures << " A5 SDMA validation checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR A5 SDMA validation checks passed" << std::endl;
    return 0;
}
