#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include "udma/tilexr_udma_layout.h"

namespace {

int g_failures = 0;

static_assert(
    std::is_standard_layout<TileXR::UDMAInfo>::value, "UDMAInfo must keep a stable binary layout");
static_assert(offsetof(TileXR::UDMAInfo, qpNum) == 0, "UDMAInfo::qpNum ABI changed");
static_assert(offsetof(TileXR::UDMAInfo, sqPtr) == 8, "UDMAInfo::sqPtr ABI changed");
static_assert(offsetof(TileXR::UDMAInfo, rqPtr) == 16, "UDMAInfo::rqPtr ABI changed");
static_assert(offsetof(TileXR::UDMAInfo, scqPtr) == 24, "UDMAInfo::scqPtr ABI changed");
static_assert(offsetof(TileXR::UDMAInfo, rcqPtr) == 32, "UDMAInfo::rcqPtr ABI changed");
static_assert(offsetof(TileXR::UDMAInfo, memPtr) == 40, "UDMAInfo::memPtr ABI changed");
static_assert(sizeof(TileXR::UDMAInfo) == 48, "UDMAInfo size ABI changed");
static_assert(sizeof(TileXR::UDMAWQCtx) == 80, "UDMAWQCtx size ABI changed");
static_assert(offsetof(TileXR::UDMAWQCtx, localTokenId) == 60,
    "UDMAWQCtx::localTokenId ABI changed");
static_assert(offsetof(TileXR::UDMAWQCtx, wqeCntAddr) == 64,
    "UDMAWQCtx::wqeCntAddr ABI changed");

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

size_t EntryIndex(size_t rank, uint32_t qpIdx, uint32_t qpNum)
{
    return rank * qpNum + qpIdx;
}

template <typename T>
T ReadImageEntry(const std::vector<uint8_t>& bytes, size_t offset, size_t index = 0)
{
    T value {};
    std::memcpy(&value, bytes.data() + offset + index * sizeof(T), sizeof(T));
    return value;
}

void TestRankMajorQpMinorLayout(uint32_t qpNum)
{
    constexpr size_t rankCount = 2;
    const size_t entryCount = rankCount * qpNum;
    std::vector<TileXR::UDMAWQCtx> sq(entryCount);
    std::vector<TileXR::UDMAWQCtx> rq(entryCount);
    std::vector<TileXR::UDMACQCtx> scq(entryCount);
    std::vector<TileXR::UDMACQCtx> rcq(entryCount);
    std::vector<TileXR::UDMAMemInfo> mem(entryCount);

    for (size_t rank = 0; rank < rankCount; ++rank) {
        for (uint32_t qpIdx = 0; qpIdx < qpNum; ++qpIdx) {
            const size_t entry = EntryIndex(rank, qpIdx, qpNum);
            sq[entry].bufAddr = 0x1000 + entry * 0x100;
            sq[entry].localTokenId = static_cast<uint32_t>(600 + entry);
            rq[entry].bufAddr = 0x2000 + entry * 0x100;
            rq[entry].localTokenId = static_cast<uint32_t>(700 + entry);
            scq[entry].dbAddr = 0x3000 + entry * 0x100;
            rcq[entry].dbAddr = 0x4000 + entry * 0x100;
            mem[entry].addr = 0x5000 + rank * 0x1000 + qpIdx * 0x100;
            mem[entry].tid = static_cast<uint32_t>(100 + entry);
            mem[entry].tpn = static_cast<uint32_t>(200 + entry);
            mem[entry].eidAddr = 0x9000 + rank * 0x1000;
        }
    }

    constexpr uintptr_t deviceBase = 0x100000000ULL;
    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;
    const int ret = TileXR::BuildUDMAInfoImage(
        deviceBase, qpNum, sq, rq, scq, rcq, mem, info, bytes);

    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_SUCCESS);
    CHECK_EQ(info.qpNum, qpNum);
    CHECK_EQ(info.sqPtr, deviceBase + sizeof(TileXR::UDMAInfo));
    CHECK_EQ(info.rqPtr, info.sqPtr + entryCount * sizeof(TileXR::UDMAWQCtx));
    CHECK_EQ(info.scqPtr, info.rqPtr + entryCount * sizeof(TileXR::UDMAWQCtx));
    CHECK_EQ(info.rcqPtr, info.scqPtr + entryCount * sizeof(TileXR::UDMACQCtx));
    CHECK_EQ(info.memPtr, info.rcqPtr + entryCount * sizeof(TileXR::UDMACQCtx));
    CHECK_EQ(bytes.size(),
             sizeof(TileXR::UDMAInfo) + entryCount * 2 * sizeof(TileXR::UDMAWQCtx) +
                 entryCount * 2 * sizeof(TileXR::UDMACQCtx) + entryCount * sizeof(TileXR::UDMAMemInfo));

    const auto imageInfo = ReadImageEntry<TileXR::UDMAInfo>(bytes, 0);
    CHECK_EQ(imageInfo.qpNum, qpNum);
    CHECK_EQ(imageInfo.memPtr, info.memPtr);
    for (size_t rank = 0; rank < rankCount; ++rank) {
        for (uint32_t qpIdx = 0; qpIdx < qpNum; ++qpIdx) {
            const size_t entry = EntryIndex(rank, qpIdx, qpNum);
            const auto imageSq = ReadImageEntry<TileXR::UDMAWQCtx>(
                bytes, static_cast<size_t>(info.sqPtr - deviceBase), entry);
            const auto imageRq = ReadImageEntry<TileXR::UDMAWQCtx>(
                bytes, static_cast<size_t>(info.rqPtr - deviceBase), entry);
            const auto imageScq = ReadImageEntry<TileXR::UDMACQCtx>(
                bytes, static_cast<size_t>(info.scqPtr - deviceBase), entry);
            const auto imageRcq = ReadImageEntry<TileXR::UDMACQCtx>(
                bytes, static_cast<size_t>(info.rcqPtr - deviceBase), entry);
            const auto imageMem = ReadImageEntry<TileXR::UDMAMemInfo>(
                bytes, static_cast<size_t>(info.memPtr - deviceBase), entry);
            CHECK_EQ(imageSq.bufAddr, sq[entry].bufAddr);
            CHECK_EQ(imageSq.localTokenId, sq[entry].localTokenId);
            CHECK_EQ(imageRq.bufAddr, rq[entry].bufAddr);
            CHECK_EQ(imageRq.localTokenId, rq[entry].localTokenId);
            CHECK_EQ(imageScq.dbAddr, scq[entry].dbAddr);
            CHECK_EQ(imageRcq.dbAddr, rcq[entry].dbAddr);
            CHECK_EQ(imageMem.addr, mem[entry].addr);
            CHECK_EQ(imageMem.tid, mem[entry].tid);
            CHECK_EQ(imageMem.tpn, mem[entry].tpn);
            CHECK_EQ(imageMem.eidAddr, mem[entry].eidAddr);
        }
    }

    if (qpNum > 1) {
        const size_t qp0 = EntryIndex(1, 0, qpNum);
        const size_t qp1 = EntryIndex(1, 1, qpNum);
        CHECK_EQ(mem[qp0].eidAddr, mem[qp1].eidAddr);
        CHECK_TRUE(mem[qp0].tpn != mem[qp1].tpn);
    }
}

int BuildWithEntryCount(size_t entryCount, uint32_t qpNum, uintptr_t deviceBase,
    TileXR::UDMAInfo& info, std::vector<uint8_t>& bytes, bool mismatchRq = false)
{
    std::vector<TileXR::UDMAWQCtx> sq(entryCount);
    std::vector<TileXR::UDMAWQCtx> rq(mismatchRq && entryCount != 0 ? entryCount - 1 : entryCount);
    std::vector<TileXR::UDMACQCtx> scq(entryCount);
    std::vector<TileXR::UDMACQCtx> rcq(entryCount);
    std::vector<TileXR::UDMAMemInfo> mem(entryCount);
    return TileXR::BuildUDMAInfoImage(
        deviceBase, qpNum, sq, rq, scq, rcq, mem, info, bytes);
}

void CheckInvalidBuild(size_t entryCount, uint32_t qpNum, uintptr_t deviceBase, bool mismatchRq = false)
{
    TileXR::UDMAInfo info {};
    info.qpNum = 99;
    info.sqPtr = 0x1234;
    std::vector<uint8_t> bytes {1, 2, 3};
    const int ret = BuildWithEntryCount(entryCount, qpNum, deviceBase, info, bytes, mismatchRq);
    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_INVALID);
    CHECK_EQ(info.qpNum, 99U);
    CHECK_EQ(info.sqPtr, static_cast<uint64_t>(0x1234));
    CHECK_EQ(bytes.size(), static_cast<size_t>(3));
    CHECK_EQ(bytes[0], static_cast<uint8_t>(1));
}

void TestRejectsInvalidLayouts()
{
    CheckInvalidBuild(2, 0, 0x1000);
    CheckInvalidBuild(0, 1, 0x1000);
    CheckInvalidBuild(5, 2, 0x1000);
    CheckInvalidBuild(2, 1, 0x1000, true);
}

void TestRejectsDeviceAddressOverflow()
{
    const uintptr_t overflowingBase =
        std::numeric_limits<uintptr_t>::max() - sizeof(TileXR::UDMAInfo);
    CheckInvalidBuild(1, 1, overflowingBase);
}

} // namespace

int main()
{
    TestRankMajorQpMinorLayout(1);
    TestRankMajorQpMinorLayout(2);
    TestRankMajorQpMinorLayout(3);
    TestRejectsInvalidLayouts();
    TestRejectsDeviceAddressOverflow();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA transport layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA transport layout checks passed" << std::endl;
    return 0;
}
