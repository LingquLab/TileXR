#include <cstddef>
#include <cstdint>
#include <iostream>

#include "comm_args.h"
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

} // namespace

int main()
{
    TestSdmaFlagDoesNotOverlapExistingFlags();
    TestCommArgsHasSdmaWorkspace();
    TestSdmaConstants();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA metadata checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA metadata checks passed" << std::endl;
    return 0;
}
