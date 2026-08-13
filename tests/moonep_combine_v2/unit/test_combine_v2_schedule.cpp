#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "combine_v2_schedule.h"
#include "combine_v2_wqe_batch.h"

namespace {

int failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

std::vector<uint32_t> SelectPeerRoutesReference(
    const std::vector<int32_t> &dst, uint32_t chunkStart,
    uint32_t peer, uint32_t slots)
{
    using namespace TileXRMoonEp;
    std::array<uint32_t, kMoonEpCombineV2MaxSelectorThreads> cursor {};
    std::vector<uint32_t> selected;
    bool firstPass = true;
    uint32_t paused = 0U;
    do {
        const uint32_t batchBase = static_cast<uint32_t>(selected.size());
        paused = 0U;
        for (uint32_t thread = 0U;
            thread < kMoonEpCombineV2SelectorThreads; ++thread) {
            uint32_t index = firstPass ?
                MoonEpCombineV2SelectorFirstIndex(chunkStart, thread) :
                MoonEpCombineV2SelectorResumeIndex(cursor[thread]);
            uint32_t lastScanned = index;
            while (MoonEpCombineV2SelectorIndexInChunk(
                index, chunkStart, static_cast<uint32_t>(dst.size()))) {
                lastScanned = index;
                const int32_t encoded = dst[index - chunkStart];
                if (encoded >= 0 &&
                    static_cast<uint32_t>(encoded) / slots == peer) {
                    const uint32_t routeIndex =
                        static_cast<uint32_t>(selected.size()) - batchBase;
                    selected.push_back(index);
                    if (routeIndex >= kMoonEpCombineV2PayloadBatchRows) {
                        cursor[thread] = lastScanned;
                        const uint32_t next =
                            MoonEpCombineV2SelectorResumeIndex(index);
                        if (MoonEpCombineV2SelectorIndexInChunk(next,
                            chunkStart, static_cast<uint32_t>(dst.size()))) {
                            ++paused;
                        }
                        break;
                    }
                }
                index = MoonEpCombineV2SelectorResumeIndex(index);
            }
            cursor[thread] = lastScanned;
        }
        Check(selected.size() - batchBase <=
                kMoonEpCombineV2MaxSelectedPayloadWqes,
            "selector reference batch exceeded the route capacity");
        firstPass = false;
    } while (paused != 0U);
    return selected;
}

void TestSchedule()
{
    using namespace TileXRMoonEp;
    const uint32_t supported[] = {2U, 3U, 4U, 5U, 6U, 7U, 8U,
        16U, 32U, 64U, 128U};
    for (uint32_t rankSize = 1U;
        rankSize <= kMoonEpCombineV2RankCount; ++rankSize) {
        bool expected = false;
        for (uint32_t value : supported) {
            expected = expected || rankSize == value;
        }
        Check(MoonEpCombineV2RankSizeSupported(rankSize) == expected,
            "supported-rank predicate mismatch");
    }
    Check(!MoonEpCombineV2RankSizeSupported(0U),
        "zero ranks accepted");
    Check(!MoonEpCombineV2RankSizeSupported(1U),
        "one rank accepted");

    for (uint32_t rankSize : supported) {
        const uint32_t activeCores =
            MoonEpCombineV2ActiveCoreCount(rankSize);
        const uint32_t stepCount = MoonEpCombineV2StepCount(rankSize);
        Check(activeCores == (rankSize <= 8U ? rankSize : 16U),
            "active-core contract mismatch");
        Check(stepCount == (rankSize <= 8U ? 1U : rankSize / 16U),
            "step-count contract mismatch");
        Check(MoonEpCombineV2LocalRankSize(rankSize) ==
                (rankSize <= 8U ? rankSize : 8U),
            "local-rank contract mismatch");
        for (uint32_t source = 0U; source < rankSize; ++source) {
            std::array<bool, kMoonEpCombineV2RankCount> seen {};
            uint32_t selfCount = 0U;
            for (uint32_t step = 0U; step < stepCount; ++step) {
                for (uint32_t core = 0U; core < activeCores; ++core) {
                    const uint32_t peer = MoonEpCombineV2Peer(
                        source, step, core, rankSize);
                    Check(peer < rankSize, "peer out of range");
                    Check(!seen[peer], "peer repeated");
                    seen[peer] = true;
                    if (rankSize > 8U) {
                        Check((core < 8U && peer < rankSize / 2U) ||
                                (core >= 8U && peer >= rankSize / 2U),
                            "core crossed target half");
                    }
                    Check(MoonEpCombineV2ReceiveStep(
                            peer, source, rankSize) == step,
                        "receive step does not invert peer");
                    if (peer == source) {
                        ++selfCount;
                        Check(step + 1U == stepCount,
                            "self peer must be in the final step");
                    }
                    if (step + 1U < stepCount) {
                        const uint32_t successor = MoonEpCombineV2Successor(
                            source, core, rankSize);
                        Check(successor / (rankSize / 2U) ==
                                source / (rankSize / 2U),
                            "successor crossed source half");
                        Check(peer == MoonEpCombineV2Peer(successor,
                                step + 1U, core, rankSize),
                            "successor schedule mismatch");
                    }
                }
            }
            for (uint32_t peer = 0U; peer < rankSize; ++peer) {
                Check(seen[peer], "schedule does not cover every peer");
            }
            Check(selfCount == 1U, "schedule must visit self once");
        }
    }
}

void TestQpAndBatchContract()
{
    using namespace TileXRMoonEp;
    std::array<bool, kMoonEpCombineV2QpCount> seen {};
    for (uint32_t core = 0; core < kMoonEpCombineV2CoreCount; ++core) {
        for (uint32_t lane = 0; lane < kMoonEpCombineV2LaneCount; ++lane) {
            const uint32_t qp = MoonEpCombineV2Qp(core, lane);
            Check(qp < kMoonEpCombineV2QpCount, "QP out of range");
            Check(!seen[qp], "QP ownership is not exclusive");
            seen[qp] = true;
        }
    }

    uint32_t sixPortRows = 0;
    uint32_t twoPortRows = 0;
    for (uint32_t row = 0; row < kMoonEpCombineV2LogicalBatchRows; ++row) {
        if (MoonEpCombineV2LaneForPosition(row) ==
            MOONEP_COMBINE_V2_SIX_PORT) {
            ++sixPortRows;
        } else {
            ++twoPortRows;
        }
    }
    Check(sixPortRows == kMoonEpCombineV2SixPortRows,
        "six-port row count mismatch");
    Check(twoPortRows == kMoonEpCombineV2TwoPortRows,
        "two-port row count mismatch");

    const MoonEpCombineV2LaneCounts payload =
        MoonEpCombineV2BatchLaneCounts(
            128U, 0U, 0U, kMoonEpCombineV2StepCount, false);
    Check(payload.sixPort == 96U && payload.twoPort == 32U,
        "payload split mismatch");
    const MoonEpCombineV2LaneCounts finalBatch =
        MoonEpCombineV2BatchLaneCounts(
            128U, 0U, 0U, kMoonEpCombineV2StepCount, true);
    Check(finalBatch.sixPort == 98U && finalBatch.twoPort == 34U,
        "control WQE split mismatch");

    for (uint32_t phase = 0; phase < 4U; ++phase) {
        for (uint32_t rows = 0; rows <= 128U; ++rows) {
            const MoonEpCombineV2LaneCounts counts =
                MoonEpCombineV2BatchLaneCounts(
                    rows, phase, 0U, kMoonEpCombineV2StepCount, false);
            Check(counts.sixPort + counts.twoPort == rows,
                "batch split lost rows");
        }
    }

    const MoonEpCombineV2RingSegments wrapped =
        MoonEpCombineV2SplitRingCopy(16380U, 34U, 16384U);
    Check(wrapped.first == 4U && wrapped.second == 30U,
        "SQ wrap split mismatch");
    Check(MoonEpCombineV2NextCqTarget(UINT32_MAX, true) == 0U,
        "CQ target must wrap as uint32");
    Check(MoonEpCombineV2CompletionCount(false) == 0U,
        "payload-only batch must not publish a CQ completion");
    Check(MoonEpCombineV2CompletionCount(true) == 1U,
        "final batch must publish exactly one CQ completion per lane");
    Check(MoonEpCombineV2CqTargetReached(0U, 0U),
        "wrapped CQ target must be reachable");
}

void TestTokensAndShapes()
{
    using namespace TileXRMoonEp;
    Check(MoonEpCombineV2ShapeValid(kMoonEpCombineV2TargetBs,
        kMoonEpCombineV2TargetH, kMoonEpCombineV2TargetTopK,
        kMoonEpCombineV2TargetSlots), "target shape rejected");
    Check(MoonEpCombineV2ShapeValid(kMoonEpCombineV2SmallBs,
        kMoonEpCombineV2TargetH, kMoonEpCombineV2TargetTopK,
        kMoonEpCombineV2SmallSlots), "small shape rejected");
    Check(MoonEpCombineV2ShapeValid(256, 1024, 4, 2040),
        "PR113 shape rejected");
    Check(MoonEpCombineV2ReduceInputStrideElements(8U) == 16U,
        "short BF16 reduction rows must use a 32-byte UB stride");
    Check(MoonEpCombineV2ReduceInputStrideElements(4096U) == 4096U,
        "aligned BF16 reduction rows changed stride");
    Check(!MoonEpCombineV2ShapeValid(256, 1024, 4, 1023),
        "undersized NvS accepted");

    const uint64_t magic = 17U;
    const uint64_t token = MoonEpCombineV2Token(magic, 6U);
    Check(MoonEpCombineV2TokenMatches(token, magic, 6U, 128U),
        "token did not match");
    Check(!MoonEpCombineV2TokenMatches(token, magic + 1U, 6U, 128U),
        "stale magic accepted");
    Check(!MoonEpCombineV2TokenMatches(token, magic, 6U, 64U),
        "step outside runtime schedule accepted");
    Check(MoonEpCombineV2GrantIndex(1U, 0U, 0U, 0U) == 256U,
        "grant epoch layout mismatch");
    Check(MoonEpCombineV2DestinationValid(0,
        kMoonEpCombineV2SmallSlots, 128U), "valid destination rejected");
    Check(MoonEpCombineV2DestinationValid(-1,
        2040U, 8U), "padding sentinel rejected");
    Check(!MoonEpCombineV2DestinationValid(-2,
        2040U, 8U), "invalid negative destination accepted");
}

void TestWqeBatchHelpers()
{
    using namespace TileXRMoonEp;
    Check(kMoonEpCombineV2PayloadBatchRows == 128U,
        "payload batch tuning constant mismatch");
    Check(kMoonEpCombineV2SelectorThreads == 128U &&
            kMoonEpCombineV2BuilderThreads == 128U,
        "SIMT thread counts are not derived from payload batch rows");
    Check(kMoonEpCombineV2MaxSelectedPayloadWqes == 256U,
        "selector payload bound mismatch");
    Check(kMoonEpCombineV2WqeBatchCapacity == 128U,
        "WQE batch capacity mismatch");
    Check(kMoonEpCombineV2BatchQpCount == 2U,
        "WQE batch QP count mismatch");
    Check(kMoonEpCombineV2SelfRelayHalfBytes == 64U * 1024U,
        "Self relay half size mismatch");
    Check(MoonEpCombineV2SelfRowsPerBatch(7168U) == 8U,
        "H=3584 BF16 must use eight Self rows per batch");
    Check(MoonEpCombineV2SelfRowsPerBatch(8192U) == 8U,
        "8 KiB rows must use eight Self rows per batch");
    Check(MoonEpCombineV2SelfRowsPerBatch(14336U) == 4U,
        "14 KiB rows must use four Self rows per batch");
    Check(MoonEpCombineV2SelfRowsPerBatch(32768U) == 2U,
        "32 KiB rows must use two Self rows per batch");
    Check(MoonEpCombineV2SelfRowsPerBatch(65536U) == 1U,
        "64 KiB rows must use one Self row per batch");
    Check(MoonEpCombineV2SelfRowsPerBatch(65537U) == 0U,
        "oversized Self rows must select tiled copy mode");
    Check(MoonEpCombineV2SelfTileCount(65537U) == 2U,
        "oversized Self row tile count mismatch");
    Check(MoonEpCombineV2SelfTileBytes(65537U, 0U) == 65536U,
        "oversized Self row first tile mismatch");
    Check(MoonEpCombineV2SelfTileBytes(65537U, 1U) == 1U,
        "oversized Self row final partial tile mismatch");
    Check(MoonEpCombineV2SelfTileBytes(65537U, 2U) == 0U,
        "out-of-range Self row tile was not rejected");
    Check(MoonEpCombineV2SelfAlignedRowBytes(7169U) == 7200U,
        "Self relay row alignment mismatch");
    Check(MoonEpCombineV2SelfAlignedRowBytes(UINT64_MAX) == 0U,
        "Self relay alignment overflow was not rejected");
    Check(MoonEpCombineV2WqeBatchCount(0U, 0U, 16384U) == 0U,
        "empty WQE batch mismatch");
    Check(MoonEpCombineV2WqeBatchCount(129U, 0U, 16384U) == 128U,
        "WQE batch capacity not enforced");
    Check(MoonEpCombineV2WqeBatchCount(128U, 16380U, 16384U) == 4U,
        "WQE ring wrap not split");
    Check(MoonEpCombineV2SelectorFirstIndex(16384U, 127U) == 16511U,
        "selector first index mismatch");
    Check(MoonEpCombineV2SelectorResumeIndex(16511U) == 16639U,
        "selector resume index mismatch");
    Check(MoonEpCombineV2SelectorIndexInChunk(
            16511U, 16384U, 128U),
        "selector rejected the last full-chunk index");
    Check(!MoonEpCombineV2SelectorIndexInChunk(
            16512U, 16384U, 128U),
        "selector accepted an index after the chunk");
    Check(!MoonEpCombineV2SelectorIndexInChunk(
            130U, 128U, 2U),
        "selector accepted a tail-thread index after the chunk");
    Check(MoonEpCombineV2CqePollBatchCount(16380U, 16384U, 128U) == 4U,
        "CQ poll ring wrap not split");
    Check(MoonEpCombineV2CqeOwnerReady(0U, 16384U, 1U),
        "initial CQ owner not recognized");
    Check(!MoonEpCombineV2CqeOwnerReady(0U, 16384U, 0U),
        "stale CQ owner accepted");

    uint32_t completed = 0;
    uint32_t detail = 0;
    Check(MoonEpCombineV2AdvanceSingleCqe(0U, 16384U, 1U, 0U, 0U,
        0U, 2U, 1U, 16384U, completed, detail) ==
        MOONEP_COMBINE_V2_SINGLE_CQE_TARGET_REACHED && completed == 2U,
        "CQ completion did not reach the submitted SQ tail");
    Check(MoonEpCombineV2AdvanceSingleCqe(0U, 16384U, 1U, 3U, 4U,
        0U, 2U, 1U, 16384U, completed, detail) ==
        MOONEP_COMBINE_V2_SINGLE_CQE_ERROR && detail == 0x304U,
        "CQ error detail mismatch");

    for (uint32_t phase = 0; phase < 4U; ++phase) {
        for (uint32_t tokenCount = 0;
            tokenCount <= kMoonEpCombineV2MaxSelectedPayloadWqes;
            ++tokenCount) {
            std::array<bool, kMoonEpCombineV2MaxSelectedPayloadWqes> seen {};
            uint32_t selectedCount = 0;
            for (uint32_t qp = 0; qp < kMoonEpCombineV2BatchQpCount; ++qp) {
                const uint32_t qpTokens = MoonEpCombineV2QpTokenCount(
                    tokenCount, phase, qp);
                selectedCount += qpTokens;
                for (uint32_t index = 0; index < qpTokens; ++index) {
                    const uint32_t selected = MoonEpCombineV2QpSelectedIndex(
                        index, phase, qp);
                    Check(selected < tokenCount, "QP selection out of range");
                    Check(!seen[selected], "QP selection duplicated a token");
                    seen[selected] = true;
                }
            }
            Check(selectedCount == tokenCount,
                "QP split did not cover every token");
        }
    }

    for (uint32_t phase = 0; phase < 4U; ++phase) {
        const MoonEpCombineV2LaneCounts payload =
            MoonEpCombineV2BatchLaneCounts(
                kMoonEpCombineV2MaxSelectedPayloadWqes, phase,
                0U, kMoonEpCombineV2StepCount, false);
        Check(payload.sixPort == 192U && payload.twoPort == 64U,
            "maximum payload does not fit the 192/64 QP split");
        const MoonEpCombineV2LaneCounts finalBatch =
            MoonEpCombineV2BatchLaneCounts(
                kMoonEpCombineV2MaxSelectedPayloadWqes, phase,
                0U, kMoonEpCombineV2StepCount, true);
        Check(finalBatch.sixPort == 194U && finalBatch.twoPort == 66U,
            "maximum final batch does not fit the 194/66 issue split");
    }
}

uint32_t EffectivePeer(uint32_t peer, uint32_t source)
{
    return peer == TileXRMoonEp::kMoonEpCombineV2InvalidPeer ?
        source : peer;
}

void TestFullGrantRounds()
{
    using namespace TileXRMoonEp;
    Check(kMoonEpCombineV2GrantStepCount == kMoonEpCombineV2StepCount,
        "grant storage does not cover every data step");
    Check(MoonEpCombineV2GrantIndex(0U, 0U, 0U, 0U) == 0U,
        "first grant transition index mismatch");
    Check(MoonEpCombineV2GrantIndex(0U, 0U, 0U, 7U) == 7U,
        "terminal grant transition index mismatch");
    Check(MoonEpCombineV2GrantIndex(0U, 0U, 1U, 0U) == 8U,
        "grant lane stride mismatch");
    Check(MoonEpCombineV2GrantIndex(1U, 0U, 0U, 0U) == 256U,
        "grant epoch stride mismatch");
    Check(MoonEpCombineV2NextStep(6U, 8U) == 7U,
        "non-terminal next step mismatch");
    Check(MoonEpCombineV2NextStep(7U, 8U) == 0U,
        "terminal next step did not wrap");

    const uint64_t magic = 17U;
    Check(MoonEpCombineV2GrantToken(magic, 6U, 8U) ==
            MoonEpCombineV2Token(magic, 7U),
        "non-terminal grant token mismatch");
    Check(MoonEpCombineV2GrantToken(magic, 7U, 8U) ==
            MoonEpCombineV2Token(magic, 0U),
        "terminal grant token did not wrap to step zero");
    Check(MoonEpCombineV2ControlWqesPerLane(7U, 8U, true) == 2U,
        "final data step does not reserve grant plus done controls");

    const MoonEpCombineV2ScheduleMode modes[] = {
        MOONEP_COMBINE_V2_SINGLE_RING,
        MOONEP_COMBINE_V2_BIDIRECTIONAL_RING};
    const uint32_t supported[] = {2U, 3U, 4U, 5U, 6U, 7U, 8U,
        16U, 32U, 64U, 128U};
    for (MoonEpCombineV2ScheduleMode mode : modes) {
        for (uint32_t rankSize : supported) {
            const uint32_t activeCores =
                MoonEpCombineV2ActiveCoreCount(rankSize);
            const uint32_t stepCount = MoonEpCombineV2StepCount(rankSize);
            for (uint32_t source = 0U; source < rankSize; ++source) {
                for (uint32_t step = 0U; step < stepCount; ++step) {
                    const uint32_t nextStep =
                        MoonEpCombineV2NextStep(step, stepCount);
                    for (uint32_t core = 0U; core < activeCores; ++core) {
                        const uint32_t currentPeer = EffectivePeer(
                            MoonEpCombineV2Peer(
                                source, step, core, rankSize, mode),
                            source);
                        const uint32_t successor = MoonEpCombineV2Successor(
                            source, step, core, rankSize, mode);
                        Check(successor < rankSize,
                            "cyclic grant successor out of range");
                        const uint32_t nextPeer = EffectivePeer(
                            MoonEpCombineV2Peer(successor, nextStep, core,
                                rankSize, mode),
                            successor);
                        Check(currentPeer == nextPeer,
                            "cyclic grant successor invariant mismatch");
                    }
                }
            }
        }
    }
}

void TestBidirectionalScheduleAnchors()
{
    using namespace TileXRMoonEp;
    const MoonEpCombineV2ScheduleMode mode =
        MOONEP_COMBINE_V2_BIDIRECTIONAL_RING;

    const std::array<uint32_t, 8U> expected8 = {
        1U, 7U, 2U, 6U, 3U, 5U, 4U,
        kMoonEpCombineV2InvalidPeer};
    for (uint32_t core = 0U; core < expected8.size(); ++core) {
        Check(MoonEpCombineV2Peer(0U, 0U, core, 8U, mode) ==
                expected8[core],
            "8P bidirectional anchor mismatch");
    }

    const std::array<uint32_t, 16U> expected32Step0 = {
        1U, 15U, 2U, 14U, 3U, 13U, 4U, 12U,
        16U, 24U, 17U, 31U, 18U, 30U, 19U, 29U};
    const std::array<uint32_t, 16U> expected32Step1 = {
        5U, 11U, 6U, 10U, 7U, 9U, 8U,
        kMoonEpCombineV2InvalidPeer,
        20U, 28U, 21U, 27U, 22U, 26U, 23U, 25U};
    const std::array<uint32_t, 16U> expected64Step0 = {
        1U, 31U, 2U, 30U, 3U, 29U, 4U, 28U,
        32U, 48U, 33U, 63U, 34U, 62U, 35U, 61U};
    const std::array<uint32_t, 16U> expected64Step3 = {
        13U, 19U, 14U, 18U, 15U, 17U, 16U,
        kMoonEpCombineV2InvalidPeer,
        44U, 52U, 45U, 51U, 46U, 50U, 47U, 49U};
    const std::array<uint32_t, 16U> expected128Step0 = {
        1U, 63U, 2U, 62U, 3U, 61U, 4U, 60U,
        64U, 96U, 65U, 127U, 66U, 126U, 67U, 125U};
    const std::array<uint32_t, 16U> expected128Step7 = {
        29U, 35U, 30U, 34U, 31U, 33U, 32U,
        kMoonEpCombineV2InvalidPeer,
        92U, 100U, 93U, 99U, 94U, 98U, 95U, 97U};

    for (uint32_t core = 0U; core < 16U; ++core) {
        Check(MoonEpCombineV2Peer(0U, 0U, core, 32U, mode) ==
                expected32Step0[core],
            "32P bidirectional first-step anchor mismatch");
        Check(MoonEpCombineV2Peer(0U, 1U, core, 32U, mode) ==
                expected32Step1[core],
            "32P bidirectional final-step anchor mismatch");
        Check(MoonEpCombineV2Peer(0U, 0U, core, 64U, mode) ==
                expected64Step0[core],
            "64P bidirectional first-step anchor mismatch");
        Check(MoonEpCombineV2Peer(0U, 3U, core, 64U, mode) ==
                expected64Step3[core],
            "64P bidirectional final-step anchor mismatch");
        Check(MoonEpCombineV2Peer(0U, 0U, core, 128U, mode) ==
                expected128Step0[core],
            "128P bidirectional first-step anchor mismatch");
        Check(MoonEpCombineV2Peer(0U, 7U, core, 128U, mode) ==
                expected128Step7[core],
            "128P bidirectional final-step anchor mismatch");
    }

    const std::array<uint32_t, 16U> expectedRank32Step0 = {
        33U, 31U, 34U, 30U, 35U, 29U, 36U, 28U,
        96U, 64U, 97U, 95U, 98U, 94U, 99U, 93U};
    for (uint32_t core = 0U; core < 16U; ++core) {
        Check(MoonEpCombineV2Peer(32U, 0U, core, 128U, mode) ==
                expectedRank32Step0[core],
            "128P nonzero-center bidirectional anchor mismatch");
    }
}

void TestBidirectionalSchedule()
{
    using namespace TileXRMoonEp;
    const MoonEpCombineV2ScheduleMode single =
        MOONEP_COMBINE_V2_SINGLE_RING;
    const MoonEpCombineV2ScheduleMode bidirectional =
        MOONEP_COMBINE_V2_BIDIRECTIONAL_RING;
    const uint32_t supported[] = {2U, 3U, 4U, 5U, 6U, 7U, 8U,
        16U, 32U, 64U, 128U};

    for (uint32_t rankSize : supported) {
        const uint32_t activeCores = MoonEpCombineV2ActiveCoreCount(rankSize);
        const uint32_t stepCount = MoonEpCombineV2StepCount(rankSize);
        for (uint32_t source = 0U; source < rankSize; ++source) {
            std::array<bool, kMoonEpCombineV2RankCount> seen {};
            uint32_t invalidCount = 0U;
            uint32_t selfCount = 0U;
            for (uint32_t step = 0U; step < stepCount; ++step) {
                for (uint32_t core = 0U; core < activeCores; ++core) {
                    Check(MoonEpCombineV2Peer(
                            source, step, core, rankSize, single) ==
                            MoonEpCombineV2Peer(
                                source, step, core, rankSize),
                        "explicit single-ring mode changed the old peer order");
                    const uint32_t rawPeer = MoonEpCombineV2Peer(
                        source, step, core, rankSize, bidirectional);
                    if (rawPeer == kMoonEpCombineV2InvalidPeer) {
                        ++invalidCount;
                    } else {
                        Check(rawPeer < rankSize,
                            "bidirectional peer out of range");
                    }
                    const uint32_t peer = EffectivePeer(rawPeer, source);
                    Check(!seen[peer], "bidirectional peer repeated");
                    seen[peer] = true;
                    if (peer == source) {
                        ++selfCount;
                    }
                    Check(MoonEpCombineV2ReceiveStep(
                            peer, source, rankSize, bidirectional) == step,
                        "bidirectional receive step does not invert peer");
                    if (rankSize >= 16U) {
                        const uint32_t halfRankCount = rankSize / 2U;
                        Check(peer / halfRankCount == core / 8U,
                            "bidirectional core crossed target half");
                    }
                    if (step + 1U < stepCount) {
                        const uint32_t successor =
                            MoonEpCombineV2Successor(source, step, core,
                                rankSize, bidirectional);
                        Check(successor / (rankSize / 2U) ==
                                source / (rankSize / 2U),
                            "bidirectional successor crossed source half");
                        const uint32_t nextRawPeer = MoonEpCombineV2Peer(
                            successor, step + 1U, core, rankSize,
                            bidirectional);
                        Check(peer == EffectivePeer(nextRawPeer, successor),
                            "bidirectional successor schedule mismatch");
                    }
                }
            }
            for (uint32_t peer = 0U; peer < rankSize; ++peer) {
                Check(seen[peer],
                    "bidirectional schedule does not cover every peer");
            }
            const bool optimized = rankSize == 8U || rankSize >= 16U;
            Check(invalidCount == (optimized ? 1U : 0U),
                "bidirectional invalid-peer count mismatch");
            Check(selfCount == 1U,
                "bidirectional schedule must visit Self once");
        }
    }

    TestBidirectionalScheduleAnchors();
}

void TestSelectorReferenceModel()
{
    using namespace TileXRMoonEp;
    const uint32_t chunkStart = 32768U;
    const uint32_t slots = 16384U;

    std::vector<int32_t> noMatches(23U, -1);
    Check(SelectPeerRoutesReference(noMatches, chunkStart, 3U, slots).empty(),
        "selector reference produced a route for an empty tail chunk");

    std::vector<int32_t> concentrated(16384U, -1);
    std::vector<uint32_t> expected;
    for (uint32_t index = 7U; index < concentrated.size();
        index += kMoonEpCombineV2SelectorThreads) {
        concentrated[index] = static_cast<int32_t>(5U * slots + index);
        expected.push_back(chunkStart + index);
    }
    const std::vector<uint32_t> selected = SelectPeerRoutesReference(
        concentrated, chunkStart, 5U, slots);
    Check(selected == expected,
        "selector reference skipped or duplicated concentrated routes");
    Check(selected.size() == 128U,
        "concentrated selector case did not exercise one owning thread");

    std::vector<int32_t> allMatches(1024U);
    for (uint32_t index = 0U; index < allMatches.size(); ++index) {
        allMatches[index] = static_cast<int32_t>(2U * slots + index);
    }
    const std::vector<uint32_t> allSelected = SelectPeerRoutesReference(
        allMatches, chunkStart, 2U, slots);
    Check(allSelected.size() == allMatches.size(),
        "selector reference did not resume through every matching route");
    std::array<bool, 1024U> seen {};
    for (uint32_t absolute : allSelected) {
        const uint32_t relative = absolute - chunkStart;
        Check(relative < seen.size() && !seen[relative],
            "selector reference duplicated an all-match route");
        if (relative < seen.size()) {
            seen[relative] = true;
        }
    }
}

} // namespace

int main()
{
    TestSchedule();
    TestBidirectionalSchedule();
    TestFullGrantRounds();
    TestQpAndBatchContract();
    TestTokensAndShapes();
    TestWqeBatchHelpers();
    TestSelectorReferenceModel();
    return failures == 0 ? 0 : 1;
}
