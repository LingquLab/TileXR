#include <array>
#include <cstdint>
#include <iostream>

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
    Check(!MoonEpCombineV2ShapeValid(16,
        kMoonEpCombineV2TargetH, kMoonEpCombineV2TargetTopK, 256),
        "unsupported shape accepted");

    const uint64_t magic = 17U;
    const uint64_t token = MoonEpCombineV2Token(magic, 6U);
    Check(MoonEpCombineV2TokenMatches(token, magic, 6U, 128U),
        "token did not match");
    Check(!MoonEpCombineV2TokenMatches(token, magic + 1U, 6U, 128U),
        "stale magic accepted");
    Check(!MoonEpCombineV2TokenMatches(token, magic, 6U, 64U),
        "step outside runtime schedule accepted");
    Check(MoonEpCombineV2GrantIndex(1U, 0U, 0U, 1U) == 224U,
        "grant epoch layout mismatch");
    Check(MoonEpCombineV2DestinationValid(0,
        kMoonEpCombineV2SmallSlots, 128U), "valid destination rejected");
    Check(!MoonEpCombineV2DestinationValid(-1,
        kMoonEpCombineV2SmallSlots, 128U), "negative destination accepted");
}

void TestWqeBatchHelpers()
{
    using namespace TileXRMoonEp;
    Check(kMoonEpCombineV2WqeBatchCapacity == 128U,
        "WQE batch capacity mismatch");
    Check(kMoonEpCombineV2BatchQpCount == 2U,
        "WQE batch QP count mismatch");
    Check(MoonEpCombineV2WqeBatchCount(0U, 0U, 16384U) == 0U,
        "empty WQE batch mismatch");
    Check(MoonEpCombineV2WqeBatchCount(129U, 0U, 16384U) == 128U,
        "WQE batch capacity not enforced");
    Check(MoonEpCombineV2WqeBatchCount(128U, 16380U, 16384U) == 4U,
        "WQE ring wrap not split");
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
        for (uint32_t tokenCount = 0; tokenCount <= 128U; ++tokenCount) {
            std::array<bool, 128U> seen {};
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
}

} // namespace

int main()
{
    TestSchedule();
    TestQpAndBatchContract();
    TestTokensAndShapes();
    TestWqeBatchHelpers();
    return failures == 0 ? 0 : 1;
}
