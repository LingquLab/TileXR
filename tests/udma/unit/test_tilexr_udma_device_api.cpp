
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "tilexr_udma.h"

namespace {

using namespace TileXR;

static_assert(sizeof(UDMASqeCtx) + sizeof(UDMASgeCtx) == 64U,
    "host model requires a one-BB WRITE/READ WQE");
static_assert(sizeof(UDMASqeCtx) + sizeof(UDMANotifyCtx) + sizeof(UDMASgeCtx) <= 128U,
    "host model requires a two-BB WRITE_WITH_NOTIFY WQE");
static_assert(sizeof(UDMACqeCtx) == 64U, "host model requires a 64-byte CQE");
static_assert(sizeof(UDMAWQCtx) == 80U, "local token must reuse the existing WQ ABI padding");
static_assert(offsetof(UDMAWQCtx, localTokenId) == 60U, "UDMAWQCtx local-token ABI changed");
static_assert(offsetof(UDMAWQCtx, wqeCntAddr) == 64U, "UDMAWQCtx counter ABI changed");

int gFailures = 0;

void Check(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++gFailures;
    }
}

template <typename T>
uint64_t AddressOf(T* ptr)
{
    return reinterpret_cast<uint64_t>(ptr);
}

struct Fixture {
    static constexpr uint32_t kRankSize = 2U;
    static constexpr uint32_t kQpNum = 2U;
    static constexpr uint32_t kEntryCount = kRankSize * kQpNum;
    static constexpr uint32_t kWqeBytes = 64U;
    static constexpr uint32_t kCqeBytes = 64U;
    static constexpr size_t kSqBytes = static_cast<size_t>(TILEXR_UDMA_SQ_BB_COUNT) * kWqeBytes;
    static constexpr size_t kSqGuardBytes = 64U;

    CommArgs args = {};
    UDMAInfo info = {};
    TileXRUDMARegistry registry = {};
    std::array<UDMAWQCtx, kEntryCount> wq = {};
    std::array<UDMACQCtx, kEntryCount> cq = {};
    std::array<UDMAMemInfo, kEntryCount> mem = {};
    std::array<std::array<uint64_t, 2>, kEntryCount> eid = {};
    std::array<std::vector<uint8_t>, kQpNum> sqBuffers;
    std::array<std::vector<uint8_t>, kQpNum> cqBuffers;
    std::array<uint32_t, kQpNum> sqHead = {};
    std::array<uint32_t, kQpNum> sqTail = {};
    std::array<uint32_t, kQpNum> wqeCount = {};
    std::array<uint32_t, kQpNum> sqDoorbell = {};
    std::array<uint32_t, kQpNum> cqTail = {};
    std::array<uint32_t, kQpNum> cqDoorbell = {};
    alignas(TILEXR_UDMA_WQE_SCRATCH_ALIGNMENT)
        std::array<uint8_t, TILEXR_UDMA_WQE_SCRATCH_BYTES> wqeScratch = {};
    std::array<uint8_t, 256> localRegion = {};
    std::array<uint8_t, 256> remoteRegion = {};

    Fixture()
    {
        args.rank = 0;
        args.rankSize = static_cast<int>(kRankSize);
        args.extraFlag = ExtraFlag::UDMA;
        args.udmaInfoPtr = reinterpret_cast<GM_ADDR>(&info);
        args.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(&registry);

        info.qpNum = kQpNum;
        info.sqPtr = AddressOf(wq.data());
        info.scqPtr = AddressOf(cq.data());
        info.memPtr = AddressOf(mem.data());

        registry.rankSize = kRankSize;
        registry.regionCount = 1U;
        registry.regions[0].base = localRegion.data();
        registry.regions[0].bytes = localRegion.size();
        registry.regions[1].base = remoteRegion.data();
        registry.regions[1].bytes = remoteRegion.size();

        for (uint32_t qp = 0U; qp < kQpNum; ++qp) {
            const uint32_t entry = kQpNum + qp;
            sqBuffers[qp].resize(kSqBytes + kSqGuardBytes);
            std::memset(sqBuffers[qp].data() + kSqBytes, 0xA5, kSqGuardBytes);
            cqBuffers[qp].resize(static_cast<size_t>(TILEXR_UDMA_CQ_DEPTH) * kCqeBytes);

            wq[entry].bufAddr = AddressOf(sqBuffers[qp].data());
            wq[entry].baseBkShift = 6U;
            wq[entry].depth = TILEXR_UDMA_SQ_BB_COUNT;
            wq[entry].headAddr = AddressOf(&sqHead[qp]);
            wq[entry].tailAddr = AddressOf(&sqTail[qp]);
            wq[entry].dbAddr = AddressOf(&sqDoorbell[qp]);
            wq[entry].localTokenId = 400U + qp;
            wq[entry].wqeCntAddr = AddressOf(&wqeCount[qp]);

            cq[entry].bufAddr = AddressOf(cqBuffers[qp].data());
            cq[entry].baseBkShift = 6U;
            cq[entry].depth = TILEXR_UDMA_CQ_DEPTH;
            cq[entry].tailAddr = AddressOf(&cqTail[qp]);
            cq[entry].dbAddr = AddressOf(&cqDoorbell[qp]);

            eid[entry][0] = 0x1000U + qp;
            eid[entry][1] = 0x2000U + qp;
            mem[entry].eidAddr = AddressOf(eid[entry].data());
            mem[entry].tpn = 100U + qp;
            mem[entry].tid = 200U + qp;
            mem[entry].rmtTokenValue = 300U + qp;
        }
    }

    UDMASqeCtx* Sqe(uint32_t qp, uint32_t bbIdx)
    {
        return reinterpret_cast<UDMASqeCtx*>(
            sqBuffers[qp].data() + static_cast<size_t>(bbIdx) * kWqeBytes);
    }

    UDMACqeCtx* Cqe(uint32_t qp, uint32_t cqeIdx)
    {
        return reinterpret_cast<UDMACqeCtx*>(
            cqBuffers[qp].data() + static_cast<size_t>(cqeIdx) * kCqeBytes);
    }

    AscendC::LocalTensor<uint8_t> Scratch()
    {
        return AscendC::LocalTensor<uint8_t>(wqeScratch.data(), wqeScratch.size());
    }

    bool SqGuardIntact(uint32_t qp) const
    {
        for (size_t i = kSqBytes; i < kSqBytes + kSqGuardBytes; ++i) {
            if (sqBuffers[qp][i] != 0xA5U) {
                return false;
            }
        }
        return true;
    }
};

void TestQpDiscoveryAndRemoteMemLayout()
{
    Fixture fixture;
    Check(UDMAQpCount(&fixture.args) == 2U, "enabled QP count");
    Check(UDMAQpValid(&fixture.args, 0U), "QP0 valid");
    Check(UDMAQpValid(&fixture.args, 1U), "QP1 valid");
    Check(!UDMAQpValid(&fixture.args, 2U), "out-of-range QP invalid");
    Check(UDMAGetRemoteMemInfo(&fixture.info, 1U, 0U) == &fixture.mem[2],
        "rank-major QP0 remote metadata");
    Check(UDMAGetRemoteMemInfo(&fixture.info, 1U, 1U) == &fixture.mem[3],
        "rank-major QP1 remote metadata");
    Check(UDMAGetRemoteMemInfo(&fixture.info, 1U) == &fixture.mem[2],
        "legacy remote metadata selects QP0");

    fixture.args.extraFlag = 0U;
    Check(UDMAQpCount(&fixture.args) == 0U, "disabled UDMA reports zero QPs");
    Check(!UDMAQpValid(&fixture.args, 0U), "disabled UDMA has no valid QP");

    fixture.args.extraFlag = ExtraFlag::UDMA;
    for (uint32_t invalidQpNum : {0U, 9U, UINT32_MAX}) {
        fixture.info.qpNum = invalidQpNum;
        Check(UDMAQpCount(&fixture.args) == 0U, "invalid device QP count is rejected");
        Check(!UDMAQpValid(&fixture.args, 0U), "invalid device QP count has no valid QP");
        Check(UDMAGetWQCtx(&fixture.info, 1U, 0U) == nullptr,
            "invalid device QP count cannot index SQ metadata");
        Check(UDMAGetRemoteMemInfo(&fixture.info, 1U, 0U) == nullptr,
            "invalid device QP count cannot index memory metadata");
    }
}

void TestValidationFailures()
{
    Fixture fixture;
    auto* local = fixture.localRegion.data();
    auto scratch = fixture.Scratch();

    fixture.args.extraFlag = 0U;
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, 1, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "disabled UDMA rejected");
    fixture.args.extraFlag = ExtraFlag::UDMA;

    GM_ADDR registry = fixture.args.udmaRegistryPtr;
    fixture.args.udmaRegistryPtr = nullptr;
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, 1, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "missing registry rejected");
    fixture.args.udmaRegistryPtr = registry;

    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, -1, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "negative rank rejected");
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, 0, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "self rank rejected");
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, 2, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "out-of-range rank rejected");
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, 1, 2U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "out-of-range QP rejected");
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, 1, 0U, local, 252U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "out-of-range registered memory rejected");
    Check(UDMAPutNbiOnQpWithFlag<uint8_t>(&fixture.args, scratch, 1, 0U, local, 0U, 8U,
        TILEXR_UDMA_SQE_FLAG_STRONG_ORDER) == TILEXR_UDMA_STATUS_INVALID,
        "flag without completion semantics rejected");
    Check(UDMAPutNbiOnQpWithFlag<uint8_t>(&fixture.args, scratch, 1, 0U, local, 0U, 8U,
        TILEXR_UDMA_SQE_FLAG_COMPLETION) == TILEXR_UDMA_STATUS_SUCCESS,
        "completion-only request is accepted for an in-order-completion QP");
    Check(UDMAFlushQpDoorbell(&fixture.args, 0, 0U) == TILEXR_UDMA_STATUS_INVALID,
        "doorbell flush rejects self rank");
    Check(UDMAQuietStatusOnQp(&fixture.args, 1, 2U) == TILEXR_UDMA_STATUS_INVALID,
        "quiet rejects out-of-range QP");

    fixture.sqHead[0] = TILEXR_UDMA_SQ_BB_COUNT + 1U;
    fixture.sqTail[0] = 0U;
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, scratch, 1, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "invalid SQ outstanding state rejected");
}

void TestInvalidScratchHasNoMutation()
{
    Fixture fixture;
    auto* local = fixture.localRegion.data();
    alignas(TILEXR_UDMA_WQE_SCRATCH_ALIGNMENT)
        std::array<uint8_t, TILEXR_UDMA_WQE_SCRATCH_BYTES + 1U> storage = {};
    const uint32_t headBefore = fixture.sqHead[0];

    AscendC::LocalTensor<uint8_t> empty;
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, empty, 1, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "null scratch is rejected");
    AscendC::LocalTensor<uint8_t> shortScratch(
        storage.data(), TILEXR_UDMA_WQE_SCRATCH_BYTES - 1U);
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, shortScratch, 1, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "short scratch is rejected");
    AscendC::LocalTensor<uint8_t> misalignedScratch(
        storage.data() + 1U, TILEXR_UDMA_WQE_SCRATCH_BYTES);
    Check(UDMAPutNbiOnQp<uint8_t>(&fixture.args, misalignedScratch, 1, 0U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_INVALID, "misaligned scratch is rejected");
    Check(fixture.sqHead[0] == headBefore && fixture.wqeCount[0] == 0U &&
        fixture.sqDoorbell[0] == 0U, "invalid scratch does not mutate queue state");
}

void TestSqFullHasNoMutation()
{
    Fixture fixture;
    auto* local = fixture.localRegion.data();
    auto scratch = fixture.Scratch();
    fixture.sqHead[1] = TILEXR_UDMA_SQ_BB_COUNT;
    fixture.sqTail[1] = 0U;
    fixture.wqeCount[1] = 9U;
    fixture.sqDoorbell[1] = 77U;
    std::memset(fixture.sqBuffers[1].data(), 0xA5, Fixture::kWqeBytes);
    std::array<uint8_t, Fixture::kWqeBytes> before = {};
    std::memcpy(before.data(), fixture.sqBuffers[1].data(), before.size());

    const uint32_t status = UDMAPutNbiOnQpWithFlagDeferred<uint8_t>(
        &fixture.args, scratch, 1, 1U, local, 0U, 8U,
        TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
    Check(status == TILEXR_UDMA_STATUS_SQ_FULL, "full SQ reports capacity failure");
    Check(fixture.sqHead[1] == TILEXR_UDMA_SQ_BB_COUNT, "full SQ preserves head");
    Check(fixture.sqTail[1] == 0U, "full SQ preserves tail");
    Check(fixture.wqeCount[1] == 9U, "full SQ preserves completion count");
    Check(fixture.sqDoorbell[1] == 77U, "full SQ preserves doorbell");
    Check(std::memcmp(before.data(), fixture.sqBuffers[1].data(), before.size()) == 0,
        "full SQ preserves WQE bytes");
}

void TestDeferredPutAndFlushAreQpSpecific()
{
    Fixture fixture;
    auto* local = fixture.localRegion.data();
    auto scratch = fixture.Scratch();
    fixture.sqDoorbell[1] = 55U;

    const uint32_t status = UDMAPutNbiOnQpWithFlagDeferred<uint8_t>(
        &fixture.args, scratch, 1, 1U, local, 8U, 8U,
        TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION);
    Check(status == TILEXR_UDMA_STATUS_SUCCESS, "deferred PUT succeeds");
    Check(fixture.sqHead[1] == 1U, "deferred PUT advances selected QP head");
    Check(fixture.wqeCount[1] == 1U, "deferred PUT advances selected QP completion count");
    Check(fixture.sqDoorbell[1] == 55U, "deferred PUT does not ring doorbell");
    Check(fixture.sqHead[0] == 0U && fixture.wqeCount[0] == 0U && fixture.sqDoorbell[0] == 0U,
        "deferred PUT does not mutate another QP");

    const UDMASqeCtx* sqe = fixture.Sqe(1U, 0U);
    Check(sqe->sqeBbIdx == 0U, "SQE records ring basic-block index");
    Check(sqe->flag == TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION,
        "SQE uses requested ordered-completion flag");
    Check(sqe->opcode == static_cast<uint32_t>(UDMAOpcode::WRITE), "deferred PUT emits WRITE");
    Check(sqe->tpId == fixture.mem[3].tpn, "deferred PUT uses QP-specific remote metadata");
    const UDMASgeCtx* sge = reinterpret_cast<const UDMASgeCtx*>(
        fixture.sqBuffers[1].data() + sizeof(UDMASqeCtx));
    Check(sge->tokenId == fixture.wq[3].localTokenId,
        "deferred PUT uses the selected QP local MR token");
    const uint64_t remoteAddr =
        static_cast<uint64_t>(sqe->rmtAddrLOrTokenId) |
        (static_cast<uint64_t>(sqe->rmtAddrHOrTokenValue) << 32U);
    Check(remoteAddr == AddressOf(fixture.remoteRegion.data() + 8U),
        "deferred PUT uses registered byte offset");

    Check(UDMAFlushQpDoorbell(&fixture.args, 1, 1U) == TILEXR_UDMA_STATUS_SUCCESS,
        "selected QP doorbell flush succeeds");
    Check(fixture.sqDoorbell[1] == fixture.sqHead[1], "flush rings current selected-QP head");
    Check(fixture.sqDoorbell[0] == 0U, "flush leaves another QP doorbell unchanged");
}

void TestImmediatePutReclaimsCompletedFullSq()
{
    Fixture fixture;
    auto* local = fixture.localRegion.data();
    auto scratch = fixture.Scratch();
    fixture.sqHead[0] = TILEXR_UDMA_SQ_BB_COUNT;
    fixture.sqTail[0] = 0U;
    fixture.wqeCount[0] = TILEXR_UDMA_SQ_BB_COUNT;
    for (uint32_t index = 0U; index < TILEXR_UDMA_SQ_BB_COUNT; ++index) {
        fixture.Sqe(0U, index)->opcode = static_cast<uint32_t>(UDMAOpcode::WRITE);
        fixture.Cqe(0U, index)->owner = 1U;
        fixture.Cqe(0U, index)->entryIdx = index;
    }

    UDMAPutNbi<uint8_t>(&fixture.args, scratch, 1, local, 0U, 8U);

    Check(fixture.sqTail[0] == TILEXR_UDMA_SQ_BB_COUNT,
        "legacy immediate PUT reclaims completed SQ entries when full");
    Check(fixture.cqTail[0] == TILEXR_UDMA_CQ_DEPTH,
        "legacy immediate PUT consumes completed CQ entries when full");
    Check(fixture.sqHead[0] == TILEXR_UDMA_SQ_BB_COUNT + 1U &&
        fixture.wqeCount[0] == TILEXR_UDMA_SQ_BB_COUNT + 1U,
        "legacy immediate PUT retries after reclaim");
    Check(fixture.sqDoorbell[0] == TILEXR_UDMA_SQ_BB_COUNT + 1U,
        "legacy immediate PUT rings the retried producer position");
    Check(fixture.Sqe(0U, 0U)->opcode == static_cast<uint32_t>(UDMAOpcode::WRITE),
        "legacy immediate PUT writes the reclaimed ring entry");
}

void TestGetAndLegacyQp0Wrappers()
{
    Fixture fixture;
    auto* local = fixture.localRegion.data();
    auto scratch = fixture.Scratch();

    Check(UDMAGetNbiOnQp<uint8_t>(&fixture.args, scratch, 1, 1U, local, 0U, 8U) ==
        TILEXR_UDMA_STATUS_SUCCESS, "QP-aware GET succeeds");
    Check(fixture.Sqe(1U, 0U)->opcode == static_cast<uint32_t>(UDMAOpcode::READ),
        "QP-aware GET emits READ on selected QP");
    Check(fixture.Sqe(1U, 0U)->flag == TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION,
        "GET uses ordered completion flag");

    UDMAPutNbi<uint8_t>(&fixture.args, scratch, 1, local, 0U, 8U);
    Check(fixture.sqHead[0] == 1U && fixture.sqDoorbell[0] == 1U,
        "legacy PUT selects QP0 and rings its doorbell");
    Check(fixture.Sqe(0U, 0U)->flag == TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION,
        "legacy PUT uses named ordered-completion flag");
}

void TestWriteNotifyWrapsWithinSqRing()
{
    Fixture fixture;
    constexpr uint32_t qp = 1U;
    auto* local = fixture.localRegion.data();
    auto scratch = fixture.Scratch();
    fixture.sqHead[qp] = TILEXR_UDMA_SQ_BB_COUNT - 1U;
    fixture.sqTail[qp] = TILEXR_UDMA_SQ_BB_COUNT - 1U;

    UDMASignalParams signal = {};
    signal.sigAddr = reinterpret_cast<uint64_t*>(fixture.remoteRegion.data() + 128U);
    signal.signal = 0x1122334455667788ULL;
    const uint32_t status = UDMAWriteNotify(&fixture.args, scratch,
        fixture.remoteRegion.data(), local, 1U, qp, 16U, &signal);

    Check(status == TILEXR_UDMA_STATUS_SUCCESS, "wrapped WRITE_WITH_NOTIFY succeeds");
    Check(fixture.sqHead[qp] == TILEXR_UDMA_SQ_BB_COUNT + 1U,
        "wrapped WRITE_WITH_NOTIFY advances by two BBs");
    Check(fixture.Sqe(qp, TILEXR_UDMA_SQ_BB_COUNT - 1U)->opcode ==
        static_cast<uint32_t>(UDMAOpcode::WRITE_WITH_NOTIFY),
        "wrapped WRITE_WITH_NOTIFY keeps SQE in the final BB");
    const uint32_t* wrappedHead = reinterpret_cast<const uint32_t*>(fixture.sqBuffers[qp].data());
    Check(wrappedHead[0] == 0x55667788U && wrappedHead[1] == 0x11223344U,
        "wrapped notify payload continues at ring head");
    const UDMASgeCtx* wrappedSge = reinterpret_cast<const UDMASgeCtx*>(
        fixture.sqBuffers[qp].data() + 16U);
    Check(wrappedSge->len == 16U && wrappedSge->va == AddressOf(local),
        "wrapped SGE follows notify data at ring head");
    Check(wrappedSge->tokenId == fixture.wq[Fixture::kQpNum + qp].localTokenId,
        "wrapped SGE uses the selected QP local MR token");
    Check(fixture.SqGuardIntact(qp), "wrapped WRITE_WITH_NOTIFY stays inside SQ allocation");

    fixture.Cqe(qp, 0U)->owner = 1U;
    fixture.Cqe(qp, 0U)->entryIdx = 0U;
    Check(UDMAQuietStatusOnQp(&fixture.args, 1, qp) == TILEXR_UDMA_STATUS_SUCCESS,
        "wrapped WRITE_WITH_NOTIFY completion succeeds");
    Check(fixture.sqTail[qp] == TILEXR_UDMA_SQ_BB_COUNT + 1U,
        "wrapped WRITE_WITH_NOTIFY completion reclaims both BBs");
}

void TestWriteNotifyCompletionUsesFinalBbIndex()
{
    Fixture fixture;
    constexpr uint32_t qp = 0U;
    auto scratch = fixture.Scratch();
    UDMASignalParams signal = {};
    signal.sigAddr = reinterpret_cast<uint64_t*>(fixture.remoteRegion.data() + 128U);
    signal.signal = 7U;

    Check(UDMAWriteNotify(&fixture.args, scratch, fixture.remoteRegion.data(),
        fixture.localRegion.data(), 1U, qp, 8U, &signal) == TILEXR_UDMA_STATUS_SUCCESS,
        "WRITE_WITH_NOTIFY posts a two-BB WQE");
    const UDMANotifyCtx* notify = reinterpret_cast<const UDMANotifyCtx*>(
        fixture.sqBuffers[qp].data() + sizeof(UDMASqeCtx));
    const uint64_t notifyAddr = static_cast<uint64_t>(notify->notifyAddrL) |
        (static_cast<uint64_t>(notify->notifyAddrH) << 32U);
    const uint64_t notifyData = static_cast<uint64_t>(notify->notifyDataL) |
        (static_cast<uint64_t>(notify->notifyDataH) << 32U);
    Check(notify->notifyTokenId == (fixture.mem[2].tid & 0xFFFFFU),
        "WRITE_WITH_NOTIFY encodes the remote notify token ID");
    Check(notify->notifyTokenValue == fixture.mem[2].rmtTokenValue,
        "WRITE_WITH_NOTIFY encodes the remote notify token value");
    Check(notifyAddr == AddressOf(signal.sigAddr),
        "WRITE_WITH_NOTIFY encodes the remote signal address");
    Check(notifyData == signal.signal,
        "WRITE_WITH_NOTIFY encodes the signal payload");
    bool paddingIsZero = true;
    for (size_t byte = sizeof(UDMASqeCtx) + sizeof(UDMANotifyCtx) + sizeof(UDMASgeCtx);
         byte < TILEXR_UDMA_WQE_SCRATCH_BYTES; ++byte) {
        paddingIsZero = paddingIsZero && fixture.sqBuffers[qp][byte] == 0U;
    }
    Check(paddingIsZero, "WRITE_WITH_NOTIFY clears the published padding bytes");
    fixture.Cqe(qp, 0U)->owner = 1U;
    fixture.Cqe(qp, 0U)->entryIdx = 1U;

    Check(UDMAQuietStatusOnQp(&fixture.args, 1, qp) == TILEXR_UDMA_STATUS_SUCCESS,
        "WRITE_WITH_NOTIFY completion accepts its final BB index");
    Check(fixture.sqTail[qp] == 2U,
        "WRITE_WITH_NOTIFY completion reclaims both basic blocks");
}

void TestLegacyQuietWithoutRegistry()
{
    Fixture fixture;
    fixture.args.udmaRegistryPtr = nullptr;
    fixture.sqHead[0] = 1U;
    fixture.sqTail[0] = 0U;
    fixture.wqeCount[0] = 1U;
    fixture.Sqe(0U, 0U)->opcode = static_cast<uint32_t>(UDMAOpcode::WRITE);
    fixture.Cqe(0U, 0U)->owner = 1U;
    fixture.Cqe(0U, 0U)->entryIdx = 0U;

    Check(UDMAQuietStatus(&fixture.args, 1) == TILEXR_UDMA_STATUS_SUCCESS,
        "legacy QP0 quiet does not require registered-memory metadata");
    Check(fixture.sqTail[0] == 1U && fixture.cqTail[0] == 1U,
        "legacy QP0 quiet reclaims queue state without a registry");
}

void TestCqReclaimAcrossSqAndCqWrap()
{
    Fixture fixture;
    constexpr uint32_t qp = 1U;
    fixture.sqTail[qp] = TILEXR_UDMA_SQ_BB_COUNT - 1U;
    fixture.sqHead[qp] = TILEXR_UDMA_SQ_BB_COUNT + 2U;
    fixture.cqTail[qp] = TILEXR_UDMA_CQ_DEPTH - 1U;
    fixture.wqeCount[qp] = TILEXR_UDMA_CQ_DEPTH + 1U;

    UDMASqeCtx* write = fixture.Sqe(qp, TILEXR_UDMA_SQ_BB_COUNT - 1U);
    write->opcode = static_cast<uint32_t>(UDMAOpcode::WRITE);
    write->sqeBbIdx = TILEXR_UDMA_SQ_BB_COUNT - 1U;
    UDMASqeCtx* notify = fixture.Sqe(qp, 0U);
    notify->opcode = static_cast<uint32_t>(UDMAOpcode::WRITE_WITH_NOTIFY);
    notify->sqeBbIdx = 0U;

    UDMACqeCtx* first = fixture.Cqe(qp, TILEXR_UDMA_CQ_DEPTH - 1U);
    first->owner = 1U;
    first->entryIdx = TILEXR_UDMA_SQ_BB_COUNT - 1U;
    UDMACqeCtx* second = fixture.Cqe(qp, 0U);
    second->owner = 0U;
    second->entryIdx = 1U;

    Check(UDMAQuietStatusOnQp(&fixture.args, 1, qp) == TILEXR_UDMA_STATUS_SUCCESS,
        "quiet reclaims valid wrapped CQEs");
    Check(fixture.sqTail[qp] == TILEXR_UDMA_SQ_BB_COUNT + 2U,
        "quiet reclaims one-BB and two-BB SQEs");
    Check(fixture.cqTail[qp] == TILEXR_UDMA_CQ_DEPTH + 1U,
        "quiet advances absolute CQ consumer across wrap");
    Check(fixture.cqDoorbell[qp] == TILEXR_UDMA_CQ_DEPTH + 1U,
        "quiet rings CQ doorbell with absolute consumer");
}

void TestInvalidCqeDoesNotReclaim()
{
    Fixture fixture;
    fixture.sqHead[0] = 1U;
    fixture.sqTail[0] = 0U;
    fixture.wqeCount[0] = 1U;
    fixture.Sqe(0U, 0U)->opcode = static_cast<uint32_t>(UDMAOpcode::WRITE);
    fixture.Cqe(0U, 0U)->owner = 1U;
    fixture.Cqe(0U, 0U)->entryIdx = 2U;
    fixture.cqDoorbell[0] = 99U;

    Check(UDMAQuietStatus(&fixture.args, 1) == TILEXR_UDMA_STATUS_INVALID,
        "QP0 quiet wrapper reports impossible reclaim");
    Check(fixture.sqTail[0] == 0U, "invalid CQE preserves SQ reclaimed tail");
    Check(fixture.cqTail[0] == 0U, "invalid CQE preserves CQ consumer");
    Check(fixture.cqDoorbell[0] == 99U, "invalid CQE preserves CQ doorbell");
    UDMAQuiet(&fixture.args, 1);
}

} // namespace

int main()
{
    TestQpDiscoveryAndRemoteMemLayout();
    TestValidationFailures();
    TestInvalidScratchHasNoMutation();
    TestSqFullHasNoMutation();
    TestDeferredPutAndFlushAreQpSpecific();
    TestImmediatePutReclaimsCompletedFullSq();
    TestGetAndLegacyQp0Wrappers();
    TestWriteNotifyWrapsWithinSqRing();
    TestWriteNotifyCompletionUsesFinalBbIndex();
    TestLegacyQuietWithoutRegistry();
    TestCqReclaimAcrossSqAndCqWrap();
    TestInvalidCqeDoesNotReclaim();

    if (gFailures != 0) {
        std::cerr << gFailures << " test(s) failed\n";
        return 1;
    }
    std::cout << "TileXR UDMA device API tests passed\n";
    return 0;
}
