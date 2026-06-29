#include <cstdint>
#include <iostream>
#include <vector>

#include "udma/tilexr_udma_layout.h"

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

void TestHostLayoutUsesDeviceRelativePointers()
{
    std::vector<TileXR::UDMAWQCtx> sq(2);
    std::vector<TileXR::UDMAWQCtx> rq(2);
    std::vector<TileXR::UDMACQCtx> scq(2);
    std::vector<TileXR::UDMACQCtx> rcq(2);
    std::vector<TileXR::UDMAMemInfo> mem(2);

    sq[0].bufAddr = 0x1000;
    sq[1].bufAddr = 0x2000;
    scq[1].dbAddr = 0x3000;
    mem[1].addr = 0x4000;
    mem[1].tid = 7;
    mem[1].tpn = 9;

    constexpr uintptr_t deviceBase = 0x100000000ULL;
    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;
    const int ret = TileXR::BuildUDMAInfoImage(deviceBase, 1, sq, rq, scq, rcq, mem, info, bytes);

    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_SUCCESS);
    CHECK_EQ(info.qpNum, 1U);
    CHECK_TRUE(info.sqPtr >= deviceBase);
    CHECK_TRUE(info.rqPtr > info.sqPtr);
    CHECK_TRUE(info.scqPtr > info.rqPtr);
    CHECK_TRUE(info.rcqPtr > info.scqPtr);
    CHECK_TRUE(info.memPtr > info.rcqPtr);
    CHECK_EQ(bytes.size(),
             sizeof(TileXR::UDMAInfo) + 2 * sizeof(TileXR::UDMAWQCtx) +
                 2 * sizeof(TileXR::UDMAWQCtx) + 2 * sizeof(TileXR::UDMACQCtx) +
                 2 * sizeof(TileXR::UDMACQCtx) + 2 * sizeof(TileXR::UDMAMemInfo));

    const auto* imageInfo = reinterpret_cast<const TileXR::UDMAInfo*>(bytes.data());
    CHECK_EQ(imageInfo->sqPtr, info.sqPtr);
    const auto* imageSq = reinterpret_cast<const TileXR::UDMAWQCtx*>(
        bytes.data() + (info.sqPtr - deviceBase));
    const auto* imageMem = reinterpret_cast<const TileXR::UDMAMemInfo*>(
        bytes.data() + (info.memPtr - deviceBase));
    CHECK_EQ(imageSq[1].bufAddr, static_cast<uint64_t>(0x2000));
    CHECK_EQ(imageMem[1].addr, static_cast<uint64_t>(0x4000));
    CHECK_EQ(imageMem[1].tid, 7U);
    CHECK_EQ(imageMem[1].tpn, 9U);
}

void TestRejectsMismatchedArrays()
{
    std::vector<TileXR::UDMAWQCtx> sq(2);
    std::vector<TileXR::UDMAWQCtx> rq(1);
    std::vector<TileXR::UDMACQCtx> scq(2);
    std::vector<TileXR::UDMACQCtx> rcq(2);
    std::vector<TileXR::UDMAMemInfo> mem(2);

    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;
    const int ret = TileXR::BuildUDMAInfoImage(0x1000, 1, sq, rq, scq, rcq, mem, info, bytes);
    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_INVALID);
}

void TestHostLayoutSupportsMultipleQps()
{
    std::vector<TileXR::UDMAWQCtx> sq(4);
    std::vector<TileXR::UDMAWQCtx> rq(4);
    std::vector<TileXR::UDMACQCtx> scq(4);
    std::vector<TileXR::UDMACQCtx> rcq(4);
    std::vector<TileXR::UDMAMemInfo> mem(4);

    sq[3].bufAddr = 0x8000;
    mem[3].tpn = 17;

    constexpr uintptr_t deviceBase = 0x200000000ULL;
    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;
    const int ret = TileXR::BuildUDMAInfoImage(deviceBase, 2, sq, rq, scq, rcq, mem, info, bytes);

    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_SUCCESS);
    CHECK_EQ(info.qpNum, 2U);
    const auto* imageSq = reinterpret_cast<const TileXR::UDMAWQCtx*>(
        bytes.data() + (info.sqPtr - deviceBase));
    const auto* imageMem = reinterpret_cast<const TileXR::UDMAMemInfo*>(
        bytes.data() + (info.memPtr - deviceBase));
    CHECK_EQ(imageSq[3].bufAddr, static_cast<uint64_t>(0x8000));
    CHECK_EQ(imageMem[3].tpn, 17U);
}

void TestLargeTransfersUseSub256MiBChunks()
{
    constexpr uint64_t oneMiB = 1024ULL * 1024ULL;
    CHECK_EQ(TileXR::TILEXR_UDMA_MAX_WQE_BYTES, static_cast<uint32_t>(128ULL * oneMiB));
    CHECK_EQ(TileXR::UDMAWqeChunkCount(128ULL * oneMiB), 1U);
    CHECK_EQ(TileXR::UDMAWqeChunkCount(256ULL * oneMiB), 2U);
    CHECK_EQ(TileXR::UDMAWqeChunkCount(512ULL * oneMiB), 4U);
    CHECK_EQ(TileXR::UDMAWqeChunkCount(1024ULL * oneMiB), 8U);
}

void TestMultiRouteQpMappingRepeatsEachRoute()
{
    const std::vector<uint32_t> routeEids = {7, 8};
    const std::vector<uint32_t> qpToEid = TileXR::BuildUDMAMultiRouteQpToEid(routeEids, 2);

    CHECK_EQ(qpToEid.size(), static_cast<size_t>(4));
    CHECK_EQ(qpToEid[0], 7U);
    CHECK_EQ(qpToEid[1], 7U);
    CHECK_EQ(qpToEid[2], 8U);
    CHECK_EQ(qpToEid[3], 8U);
}

void TestMultiRouteQpMappingRejectsEmptyInputs()
{
    CHECK_TRUE(TileXR::BuildUDMAMultiRouteQpToEid({}, 1).empty());
    CHECK_TRUE(TileXR::BuildUDMAMultiRouteQpToEid({7}, 0).empty());
}

} // namespace

int main()
{
    TestHostLayoutUsesDeviceRelativePointers();
    TestRejectsMismatchedArrays();
    TestHostLayoutSupportsMultipleQps();
    TestLargeTransfersUseSub256MiBChunks();
    TestMultiRouteQpMappingRepeatsEachRoute();
    TestMultiRouteQpMappingRejectsEmptyInputs();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA transport layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA transport layout checks passed" << std::endl;
    return 0;
}
