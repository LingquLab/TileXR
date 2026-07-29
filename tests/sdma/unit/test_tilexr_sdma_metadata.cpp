#include <cstddef>
#include <cstdint>
#include <iostream>

#include "comm_args.h"
#include "tilexr_sdma_a5_types.h"
#include "tilexr_sdma_types.h"

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

void TestSdmaFlagDoesNotOverlapExistingFlags()
{
    constexpr uint32_t sdma = TileXR::ExtraFlag::SDMA;
    CHECK_EQ(sdma, static_cast<uint32_t>(1U << 11));
    CHECK_EQ(sdma & TileXR::ExtraFlag::UDMA, 0U);
    CHECK_EQ(sdma & TileXR::ExtraFlag::RDMA, 0U);
    CHECK_EQ(sdma & TileXR::ExtraFlag::TOPO_PCIE, 0U);
}

void TestCommArgsHasSdmaWorkspace()
{
    TileXR::CommArgs args {};
    CHECK_TRUE(args.sdmaWorkspacePtr == nullptr);
    CHECK_TRUE(offsetof(TileXR::CommArgs, sdmaWorkspacePtr) > offsetof(TileXR::CommArgs, udmaRegistryPtr));
}

void TestSdmaConstants()
{
    CHECK_EQ(TileXR::TILEXR_SDMA_DEFAULT_BLOCK_BYTES, static_cast<uint64_t>(1024 * 1024));
    CHECK_EQ(TileXR::TILEXR_SDMA_DEFAULT_QUEUE_NUM, 1U);
    CHECK_EQ(TileXR::TILEXR_SDMA_SCRATCH_BYTES, 256U);
}

void TestA5WorkspaceAbi()
{
    using namespace TileXR::detail;
    CHECK_EQ(TILEXR_SDMA_A5_CHANNEL_COUNT, 48U);
    CHECK_EQ(sizeof(A5SdmaWorkspaceHeader), 64U);
    CHECK_EQ(sizeof(A5SdmaChannel), 192U);
    CHECK_EQ(sizeof(A5SdmaCompletionLine), 64U);
    CHECK_EQ(sizeof(A5SdmaSqe), 64U);
    CHECK_EQ(offsetof(A5SdmaChannel, generation), 72U);
    CHECK_EQ(offsetof(A5SdmaChannel, outstanding), 128U);
    CHECK_EQ(offsetof(A5SdmaSqe, srcAddressLow), 32U);
    CHECK_EQ(offsetof(A5SdmaSqe, length), 48U);
    CHECK_EQ(offsetof(A5SdmaWorkspace, channels), 64U);
    CHECK_EQ(sizeof(A5SdmaWorkspace) % 64U, 0U);
}

void TestA5QueueAndTransferHelpers()
{
    using namespace TileXR::detail;
    CHECK_TRUE(!A5SdmaQueueStateValid(0U, 2U));
    CHECK_TRUE(A5SdmaQueueStateValid(2U, 3U));
    CHECK_TRUE(!A5SdmaQueueStateValid(3U, 3U));
    CHECK_EQ(A5SdmaAdvanceTail(0U, 3U), 2U);
    CHECK_EQ(A5SdmaAdvanceTail(2U, 3U), 1U);
    CHECK_EQ(A5SdmaQueueDistance(2U, 1U, 3U), 2U);
    CHECK_TRUE(A5SdmaQueueHasCapacity(0U, 0U, 3U));
    CHECK_TRUE(A5SdmaQueueHasCapacity(1U, 2U, 8U));
    CHECK_TRUE(!A5SdmaQueueHasCapacity(0U, 2U, 3U));
    CHECK_TRUE(!A5SdmaQueueHasCapacity(3U, 0U, 3U));
    CHECK_EQ(A5SdmaAdvanceTaskId(0xFFFFU), 1U);
    CHECK_TRUE(!A5SdmaTransferLengthValid(0U));
    CHECK_TRUE(A5SdmaTransferLengthValid(1U));
    CHECK_TRUE(A5SdmaTransferLengthValid(TILEXR_SDMA_A5_MAX_TRANSFER_BYTES));
    CHECK_TRUE(!A5SdmaTransferLengthValid(TILEXR_SDMA_A5_MAX_TRANSFER_BYTES + 1U));
    CHECK_EQ(A5SdmaNextGeneration(0U), 1U);
    CHECK_EQ(A5SdmaNextGeneration(1U), 2U);
    CHECK_EQ(A5SdmaNextGeneration(0xFFFFFFFFU), 1U);
}

void TestA5EventHelpers()
{
    using namespace TileXR::detail;
    uint32_t channel = 0U;
    uint32_t generation = 0U;
    const uint64_t event = A5SdmaEncodeEvent(47U, 0x12345678U);
    CHECK_TRUE(event != 0U);
    CHECK_TRUE(A5SdmaDecodeEvent(event, channel, generation));
    CHECK_EQ(channel, 47U);
    CHECK_EQ(generation, 0x12345678U);
    CHECK_EQ(A5SdmaEncodeEvent(48U, 1U), 0U);
    CHECK_EQ(A5SdmaEncodeEvent(0U, 0U), 0U);
    CHECK_TRUE(!A5SdmaDecodeEvent(0U, channel, generation));
    CHECK_TRUE(!A5SdmaDecodeEvent(event | TILEXR_SDMA_A5_EVENT_RESERVED_MASK,
                                 channel, generation));
    CHECK_TRUE(!A5SdmaDecodeEvent(event & ~TILEXR_SDMA_A5_EVENT_GENERATION_MASK,
                                 channel, generation));
}

} // namespace

int main()
{
    TestSdmaFlagDoesNotOverlapExistingFlags();
    TestCommArgsHasSdmaWorkspace();
    TestSdmaConstants();
    TestA5WorkspaceAbi();
    TestA5QueueAndTransferHelpers();
    TestA5EventHelpers();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA metadata checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA metadata checks passed" << std::endl;
    return 0;
}
