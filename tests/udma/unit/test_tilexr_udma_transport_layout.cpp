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

void TestMultiQpLayoutUsesPeerMajorIndexing()
{
    constexpr uint32_t qpNum = 4;
    constexpr size_t rankCount = 2;
    const size_t entryCount = rankCount * qpNum;
    std::vector<TileXR::UDMAWQCtx> sq(entryCount);
    std::vector<TileXR::UDMAWQCtx> rq(entryCount);
    std::vector<TileXR::UDMACQCtx> scq(entryCount);
    std::vector<TileXR::UDMACQCtx> rcq(entryCount);
    std::vector<TileXR::UDMAMemInfo> mem(entryCount);
    const size_t peerOneQpThree = 1 * qpNum + 3;
    sq[peerOneQpThree].bufAddr = 0x1234000;
    scq[peerOneQpThree].tailAddr = 0x5678000;
    mem[peerOneQpThree].tpn = 77;

    constexpr uintptr_t deviceBase = 0x200000000ULL;
    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;
    const int ret = TileXR::BuildUDMAInfoImage(
        deviceBase, qpNum, sq, rq, scq, rcq, mem, info, bytes);

    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_SUCCESS);
    CHECK_EQ(info.qpNum, qpNum);
    const auto* imageSq = reinterpret_cast<const TileXR::UDMAWQCtx*>(
        bytes.data() + (info.sqPtr - deviceBase));
    const auto* imageScq = reinterpret_cast<const TileXR::UDMACQCtx*>(
        bytes.data() + (info.scqPtr - deviceBase));
    const auto* imageMem = reinterpret_cast<const TileXR::UDMAMemInfo*>(
        bytes.data() + (info.memPtr - deviceBase));
    CHECK_EQ(imageSq[peerOneQpThree].bufAddr, static_cast<uint64_t>(0x1234000));
    CHECK_EQ(imageScq[peerOneQpThree].tailAddr, static_cast<uint64_t>(0x5678000));
    CHECK_EQ(imageMem[peerOneQpThree].tpn, 77U);
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

void TestRejectsInvalidQpCountsAndEntryCounts()
{
    std::vector<TileXR::UDMAWQCtx> sq(6);
    std::vector<TileXR::UDMAWQCtx> rq(6);
    std::vector<TileXR::UDMACQCtx> scq(6);
    std::vector<TileXR::UDMACQCtx> rcq(6);
    std::vector<TileXR::UDMAMemInfo> mem(6);
    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;

    CHECK_EQ(TileXR::BuildUDMAInfoImage(0x1000, 3, sq, rq, scq, rcq, mem, info, bytes),
             TileXR::TILEXR_UDMA_LAYOUT_INVALID);
    CHECK_EQ(TileXR::BuildUDMAInfoImage(0x1000, 4, sq, rq, scq, rcq, mem, info, bytes),
             TileXR::TILEXR_UDMA_LAYOUT_INVALID);
    CHECK_TRUE(TileXR::IsSupportedUDMAQpNum(1));
    CHECK_TRUE(TileXR::IsSupportedUDMAQpNum(2));
    CHECK_TRUE(TileXR::IsSupportedUDMAQpNum(4));
    CHECK_TRUE(TileXR::IsSupportedUDMAQpNum(8));
    CHECK_TRUE(!TileXR::IsSupportedUDMAQpNum(3));
}

void TestParsesQpCountConfiguration()
{
    uint32_t qpNum = 0;
    CHECK_EQ(TileXR::ParseUDMAQpNum(nullptr, qpNum), TileXR::TILEXR_UDMA_LAYOUT_SUCCESS);
    CHECK_EQ(qpNum, TileXR::TILEXR_UDMA_DEFAULT_QP_NUM);
    CHECK_EQ(TileXR::ParseUDMAQpNum("8", qpNum), TileXR::TILEXR_UDMA_LAYOUT_SUCCESS);
    CHECK_EQ(qpNum, 8U);
    CHECK_EQ(TileXR::ParseUDMAQpNum("3", qpNum), TileXR::TILEXR_UDMA_LAYOUT_INVALID);
    CHECK_EQ(TileXR::ParseUDMAQpNum("4x", qpNum), TileXR::TILEXR_UDMA_LAYOUT_INVALID);
    CHECK_EQ(TileXR::ParseUDMAQpNum("-1", qpNum), TileXR::TILEXR_UDMA_LAYOUT_INVALID);
}

} // namespace

int main()
{
    TestHostLayoutUsesDeviceRelativePointers();
    TestMultiQpLayoutUsesPeerMajorIndexing();
    TestRejectsMismatchedArrays();
    TestRejectsInvalidQpCountsAndEntryCounts();
    TestParsesQpCountConfiguration();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA transport layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA transport layout checks passed" << std::endl;
    return 0;
}
