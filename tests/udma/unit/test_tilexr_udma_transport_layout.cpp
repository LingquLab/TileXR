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

} // namespace

int main()
{
    TestHostLayoutUsesDeviceRelativePointers();
    TestRejectsMismatchedArrays();
    TestHostLayoutSupportsMultipleQps();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA transport layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA transport layout checks passed" << std::endl;
    return 0;
}
