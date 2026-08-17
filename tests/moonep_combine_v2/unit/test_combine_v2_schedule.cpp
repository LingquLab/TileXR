#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "combine_v2_schedule.h"
#include "combine_v2_wqe_batch.h"
#include "tilexr_udma_fullmesh.h"

namespace {

int failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

std::vector<uint32_t> SelectPeerIndicesReference(
    const std::vector<int32_t> &dst, uint32_t peer, uint32_t slots)
{
    std::vector<uint32_t> selected;
    const uint64_t peerBase = static_cast<uint64_t>(peer) * slots;
    const uint64_t peerEnd = peerBase + slots;
    for (uint32_t index = 0U; index < dst.size(); ++index) {
        const int32_t encoded = dst[index];
        if (encoded >= 0 &&
            static_cast<uint64_t>(encoded) >= peerBase &&
            static_cast<uint64_t>(encoded) < peerEnd) {
            selected.push_back(index);
        }
    }
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
    Check(kMoonEpCombineV2BuilderThreads == 128U,
        "SIMT builder thread count mismatch");
    Check(kMoonEpCombineV2MaxSelectedPayloadWqes == 128U,
        "payload submission bound mismatch");
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
        Check(payload.sixPort == 96U && payload.twoPort == 32U,
            "maximum payload does not fit the 96/32 QP split");
        const MoonEpCombineV2LaneCounts finalBatch =
            MoonEpCombineV2BatchLaneCounts(
                kMoonEpCombineV2MaxSelectedPayloadWqes, phase,
                0U, kMoonEpCombineV2StepCount, true);
        Check(finalBatch.sixPort == 98U && finalBatch.twoPort == 34U,
            "maximum final batch does not fit the 98/34 issue split");
    }
}

uint32_t EffectivePeer(uint32_t peer, uint32_t source)
{
    return peer == TileXRMoonEp::kMoonEpCombineV2InvalidPeer ?
        source : peer;
}

void TestServerPairSchedule()
{
    using namespace TileXRMoonEp;
    const MoonEpCombineV2ScheduleMode mode =
        MOONEP_COMBINE_V2_SERVER_PAIR_SAME_CROSS;
    const uint32_t rankSizes[] = {32U, 64U, 128U};

    Check(!MoonEpCombineV2ServerPairRankSize(16U) &&
            MoonEpCombineV2ServerPairRankSize(32U) &&
            MoonEpCombineV2ServerPairRankSize(64U) &&
            MoonEpCombineV2ServerPairRankSize(128U),
        "Server-Pair rank-size contract mismatch");
    Check(MoonEpCombineV2ServerPairPhaseStepCount(16U) == 0U &&
            MoonEpCombineV2ServerPairRoundCount(16U) == 0U &&
            MoonEpCombineV2ServerPairPhase(0U, 16U) ==
                kMoonEpCombineV2InvalidPeer &&
            MoonEpCombineV2ServerPairPhaseStep(0U, 16U) ==
                kMoonEpCombineV2InvalidPeer,
        "Server-Pair helpers accepted a legacy rank size");
    const MoonEpCombineV2ScheduleCoordinate invalid =
        MoonEpCombineV2ServerPairReceive(0U, 0U, 16U);
    Check(invalid.phase == kMoonEpCombineV2InvalidPeer &&
            invalid.phaseStep == kMoonEpCombineV2InvalidPeer &&
            invalid.round == kMoonEpCombineV2InvalidPeer &&
            invalid.core == kMoonEpCombineV2InvalidPeer,
        "Server-Pair receive accepted a legacy rank size");

    const uint32_t legacyRankSizes[] = {2U, 3U, 4U, 5U, 6U, 7U, 8U, 16U};
    for (uint32_t rankSize : legacyRankSizes) {
        for (uint32_t source = 0U; source < rankSize; ++source) {
            for (uint32_t step = 0U;
                step < MoonEpCombineV2StepCount(rankSize); ++step) {
                for (uint32_t core = 0U;
                    core < MoonEpCombineV2ActiveCoreCount(rankSize); ++core) {
                    Check(MoonEpCombineV2Peer(
                            source, step, core, rankSize, mode) ==
                            MoonEpCombineV2Peer(source, step, core, rankSize,
                                MOONEP_COMBINE_V2_BIDIRECTIONAL_RING),
                        "Server-Pair mode changed the 2P-16P Ring schedule");
                }
            }
        }
    }

    for (uint32_t rankSize : rankSizes) {
        const uint32_t phaseStepCount = rankSize / 32U;
        const uint32_t roundCount = rankSize / 16U;
        Check(MoonEpCombineV2ServerPairPhaseStepCount(rankSize) ==
                phaseStepCount &&
                MoonEpCombineV2ServerPairRoundCount(rankSize) == roundCount,
            "Server-Pair phase dimensions mismatch");
        for (uint32_t source = 0U; source < rankSize; ++source) {
            std::array<bool, kMoonEpCombineV2RankCount> seen {};
            for (uint32_t round = 0U; round < roundCount; ++round) {
                const uint32_t phase = round / phaseStepCount;
                const uint32_t phaseStep = round % phaseStepCount;
                const uint32_t expectedHalf = phase == 0U ?
                    source / (rankSize / 2U) :
                    1U - source / (rankSize / 2U);
                std::array<uint32_t, kMoonEpCombineV2GroupCount>
                    targetServerCounts {};
                for (uint32_t core = 0U;
                    core < kMoonEpCombineV2CoreCount; ++core) {
                    const uint32_t peer = MoonEpCombineV2ServerPairPeer(
                        source, round, core, rankSize);
                    Check(peer < rankSize, "Server-Pair peer out of range");
                    Check(!seen[peer], "Server-Pair peer repeated");
                    seen[peer] = true;
                    Check(peer / (rankSize / 2U) == expectedHalf,
                        "Server-Pair peer selected the wrong target half");
                    Check(peer % kMoonEpCombineV2GroupSize ==
                            core % kMoonEpCombineV2GroupSize,
                        "Server-Pair core changed the target local rank");
                    ++targetServerCounts[
                        peer / kMoonEpCombineV2GroupSize];
                    Check(MoonEpCombineV2Peer(
                            source, round, core, rankSize, mode) == peer,
                        "active Server-Pair mode does not use its peer helper");
                    const MoonEpCombineV2ScheduleCoordinate coordinate =
                        MoonEpCombineV2ServerPairReceive(
                            peer, source, rankSize);
                    Check(coordinate.phase == phase &&
                            coordinate.phaseStep == phaseStep &&
                            coordinate.round == round &&
                            coordinate.core == core,
                        "Server-Pair receive does not invert peer mapping");
                    Check(MoonEpCombineV2ReceiveStep(
                            peer, source, rankSize, mode) == round,
                        "active Server-Pair receive round mismatch");
                    Check(peer == source || MoonEpCombineV2SenderCore(
                            source, peer, rankSize, mode) == core,
                        "active Server-Pair sender core mismatch");
                }
                uint32_t targetServerCount = 0U;
                for (uint32_t count : targetServerCounts) {
                    if (count != 0U) {
                        ++targetServerCount;
                        Check(count == kMoonEpCombineV2GroupSize,
                            "Server-Pair round does not cover a full server");
                    }
                }
                Check(targetServerCount == 2U,
                    "Server-Pair round does not select exactly two servers");
            }
            for (uint32_t destination = 0U;
                destination < rankSize; ++destination) {
                Check(seen[destination],
                    "Server-Pair schedule does not cover every destination");
                const MoonEpCombineV2ScheduleCoordinate coordinate =
                    MoonEpCombineV2ServerPairReceive(
                        destination, source, rankSize);
                Check(coordinate.round < roundCount &&
                        coordinate.core < kMoonEpCombineV2CoreCount &&
                        MoonEpCombineV2ServerPairPeer(source,
                            coordinate.round, coordinate.core, rankSize) ==
                            destination,
                    "Server-Pair coordinate failed exhaustive inversion");
            }
        }

        for (uint32_t destination = 0U;
            destination < rankSize; ++destination) {
            for (uint32_t round = 0U; round < roundCount; ++round) {
                std::array<uint32_t, kMoonEpCombineV2GroupCount>
                    sourceServerCounts {};
                uint32_t sourceCount = 0U;
                for (uint32_t source = 0U; source < rankSize; ++source) {
                    const MoonEpCombineV2ScheduleCoordinate coordinate =
                        MoonEpCombineV2ServerPairReceive(
                            destination, source, rankSize);
                    if (coordinate.round == round) {
                        ++sourceCount;
                        ++sourceServerCounts[
                            source / kMoonEpCombineV2GroupSize];
                        const uint32_t expectedSourceHalf =
                            round / phaseStepCount == 0U ?
                                destination / (rankSize / 2U) :
                                1U - destination / (rankSize / 2U);
                        Check(source / (rankSize / 2U) == expectedSourceHalf,
                            "Server-Pair fan-in came from the wrong half");
                    }
                }
                uint32_t sourceServerCount = 0U;
                for (uint32_t count : sourceServerCounts) {
                    if (count != 0U) {
                        ++sourceServerCount;
                        Check(count == kMoonEpCombineV2GroupSize,
                            "Server-Pair fan-in omitted a source card");
                    }
                }
                Check(sourceCount == 2U * kMoonEpCombineV2GroupSize &&
                        sourceServerCount == 2U,
                    "Server-Pair fan-in is not two complete source servers");
            }
        }
    }

    const uint32_t expected128Servers[8U][2U] = {
        {1U, 2U}, {3U, 4U}, {5U, 6U}, {7U, 0U},
        {9U, 10U}, {11U, 12U}, {13U, 14U}, {15U, 8U}};
    for (uint32_t round = 0U; round < 8U; ++round) {
        for (uint32_t core = 0U; core < 16U; ++core) {
            const uint32_t expectedServer =
                expected128Servers[round][core / 8U];
            Check(MoonEpCombineV2ServerPairPeer(
                    0U, round, core, 128U) ==
                    expectedServer * 8U + core % 8U,
                "128P Server-Pair anchor mismatch");
        }
    }
}

void TestServerPairParitySchedule()
{
    using namespace TileXRMoonEp;
    const MoonEpCombineV2ScheduleMode mode =
        MOONEP_COMBINE_V2_SERVER_PAIR_PARITY;
    const uint32_t rankSizes[] = {32U, 64U, 128U};

    const uint32_t legacyRankSizes[] = {2U, 3U, 4U, 5U, 6U, 7U, 8U, 16U};
    for (uint32_t rankSize : legacyRankSizes) {
        for (uint32_t source = 0U; source < rankSize; ++source) {
            for (uint32_t step = 0U;
                step < MoonEpCombineV2StepCount(rankSize); ++step) {
                for (uint32_t core = 0U;
                    core < MoonEpCombineV2ActiveCoreCount(rankSize); ++core) {
                    Check(MoonEpCombineV2Peer(
                            source, step, core, rankSize, mode) ==
                            MoonEpCombineV2Peer(source, step, core, rankSize,
                                MOONEP_COMBINE_V2_BIDIRECTIONAL_RING),
                        "parity Server-Pair changed the 2P-16P Ring schedule");
                }
            }
        }
    }

    for (uint32_t rankSize : rankSizes) {
        const uint32_t phaseStepCount = rankSize / 32U;
        const uint32_t roundCount = rankSize / 16U;
        const uint32_t halfRankCount = rankSize / 2U;
        for (uint32_t source = 0U; source < rankSize; ++source) {
            std::array<bool, kMoonEpCombineV2RankCount> seen {};
            for (uint32_t round = 0U; round < roundCount; ++round) {
                const uint32_t phase = round / phaseStepCount;
                const uint32_t phaseStep = round % phaseStepCount;
                const uint32_t expectedHalf = phase == 0U ?
                    1U - (source & 1U) : source & 1U;
                std::array<uint32_t, kMoonEpCombineV2GroupCount>
                    targetServerCounts {};
                for (uint32_t core = 0U;
                    core < kMoonEpCombineV2CoreCount; ++core) {
                    const uint32_t peer = MoonEpCombineV2ServerPairPeer(
                        source, round, core, rankSize, mode);
                    Check(peer < rankSize,
                        "parity Server-Pair peer out of range");
                    Check(!seen[peer],
                        "parity Server-Pair peer repeated");
                    seen[peer] = true;
                    Check(peer / halfRankCount == expectedHalf,
                        "parity Server-Pair selected the wrong target half");
                    Check(peer % kMoonEpCombineV2GroupSize ==
                            core % kMoonEpCombineV2GroupSize,
                        "parity Server-Pair changed target local rank");
                    ++targetServerCounts[
                        peer / kMoonEpCombineV2GroupSize];
                    Check(MoonEpCombineV2Peer(
                            source, round, core, rankSize, mode) == peer,
                        "active parity mode bypasses its peer helper");
                    const MoonEpCombineV2ScheduleCoordinate coordinate =
                        MoonEpCombineV2ServerPairReceive(
                            peer, source, rankSize, mode);
                    Check(coordinate.phase == phase &&
                            coordinate.phaseStep == phaseStep &&
                            coordinate.round == round &&
                            coordinate.core == core,
                        "parity receive does not invert peer mapping");
                    Check(MoonEpCombineV2ReceiveStep(
                            peer, source, rankSize, mode) == round,
                        "active parity receive round mismatch");
                    Check(peer == source || MoonEpCombineV2SenderCore(
                            source, peer, rankSize, mode) == core,
                        "active parity sender core mismatch");
                }
                uint32_t targetServerCount = 0U;
                for (uint32_t count : targetServerCounts) {
                    if (count != 0U) {
                        ++targetServerCount;
                        Check(count == kMoonEpCombineV2GroupSize,
                            "parity round does not cover a full server");
                    }
                }
                Check(targetServerCount == 2U,
                    "parity round does not select exactly two servers");
            }
            for (uint32_t destination = 0U;
                destination < rankSize; ++destination) {
                Check(seen[destination],
                    "parity schedule does not cover every destination");
                const MoonEpCombineV2ScheduleCoordinate coordinate =
                    MoonEpCombineV2ServerPairReceive(
                        destination, source, rankSize, mode);
                Check(coordinate.round < roundCount &&
                        coordinate.core < kMoonEpCombineV2CoreCount &&
                        MoonEpCombineV2ServerPairPeer(source,
                            coordinate.round, coordinate.core, rankSize,
                            mode) == destination,
                    "parity coordinate failed exhaustive inversion");
            }
        }

        for (uint32_t destination = 0U;
            destination < rankSize; ++destination) {
            const uint32_t destinationHalf = destination / halfRankCount;
            for (uint32_t round = 0U; round < roundCount; ++round) {
                const uint32_t phase = round / phaseStepCount;
                const uint32_t expectedParity = phase == 0U ?
                    1U - destinationHalf : destinationHalf;
                std::array<uint32_t, kMoonEpCombineV2GroupCount>
                    sourceServerCounts {};
                uint32_t sourceCount = 0U;
                for (uint32_t source = 0U; source < rankSize; ++source) {
                    const MoonEpCombineV2ScheduleCoordinate coordinate =
                        MoonEpCombineV2ServerPairReceive(
                            destination, source, rankSize, mode);
                    if (coordinate.round != round) {
                        continue;
                    }
                    ++sourceCount;
                    ++sourceServerCounts[
                        source / kMoonEpCombineV2GroupSize];
                    Check((source & 1U) == expectedParity,
                        "parity fan-in came from the wrong card parity");
                }
                uint32_t sourceServerCount = 0U;
                uint32_t sameHalfServerCount = 0U;
                for (uint32_t server = 0U;
                    server < rankSize / kMoonEpCombineV2GroupSize; ++server) {
                    const uint32_t count = sourceServerCounts[server];
                    if (count == 0U) {
                        continue;
                    }
                    ++sourceServerCount;
                    sameHalfServerCount +=
                        server / (rankSize / 16U) == destinationHalf ? 1U : 0U;
                    Check(count == 4U,
                        "parity fan-in is not four same-parity cards");
                }
                Check(sourceCount == kMoonEpCombineV2CoreCount &&
                        sourceServerCount == 4U,
                    "parity fan-in is not four servers by four cards");
                Check(sameHalfServerCount == 2U,
                    "parity fan-in does not contain two same-half servers");
            }
        }
    }

    const uint32_t expected128Servers[8U][2U] = {
        {9U, 10U}, {11U, 12U}, {13U, 14U}, {15U, 8U},
        {1U, 2U}, {3U, 4U}, {5U, 6U}, {7U, 0U}};
    for (uint32_t round = 0U; round < 8U; ++round) {
        for (uint32_t core = 0U; core < 16U; ++core) {
            Check(MoonEpCombineV2ServerPairPeer(
                    0U, round, core, 128U, mode) ==
                    expected128Servers[round][core / 8U] * 8U + core % 8U,
                "128P parity Server-Pair anchor mismatch");
        }
    }
}

void TestServerGrantSchedule()
{
    using namespace TileXRMoonEp;
    const uint32_t rankSizes[] = {32U, 64U, 128U};
    Check(kMoonEpCombineV2ServerGrantSignalCount == 64U &&
            kMoonEpCombineV2ServerGrantSignalsPerCore == 4U,
        "Server-Grant signal or shard count mismatch");
    Check(kMoonEpCombineV2ServerGrantReceiveBytes == 65536U &&
            kMoonEpCombineV2ServerGrantSourceBytes == 65536U &&
            kMoonEpCombineV2ServerGrantWorkspaceBytes == 131072U,
        "Server-Grant workspace constants mismatch");
    Check(!MoonEpCombineV2ServerGrantEnabled(
            16U, MOONEP_COMBINE_V2_SERVER_PAIR_PARITY) &&
            !MoonEpCombineV2ServerGrantEnabled(
                32U, MOONEP_COMBINE_V2_SERVER_PAIR_SAME_CROSS) &&
            MoonEpCombineV2ServerGrantEnabled(
                32U, MOONEP_COMBINE_V2_SERVER_PAIR_PARITY),
        "Server-Grant activation contract mismatch");

    std::array<bool,
        kMoonEpCombineV2ServerGrantRoundCount *
            kMoonEpCombineV2ServerGrantSignalCount> epochSlots {};
    for (uint32_t flatRound = 0U;
        flatRound < kMoonEpCombineV2ServerGrantRoundCount; ++flatRound) {
        for (uint32_t ordinal = 0U;
            ordinal < kMoonEpCombineV2ServerGrantSignalCount; ++ordinal) {
            const uint64_t index = MoonEpCombineV2ServerGrantSignalIndex(
                0U, flatRound, ordinal);
            Check(index < epochSlots.size() && !epochSlots[index],
                "Server-Grant epoch slot overlaps");
            epochSlots[index] = true;
            Check(MoonEpCombineV2ServerGrantSignalIndex(
                    1U, flatRound, ordinal) ==
                    epochSlots.size() + index,
                "Server-Grant epochs do not occupy disjoint slots");
        }
    }
    for (bool seen : epochSlots) {
        Check(seen, "Server-Grant epoch layout has a hole");
    }

    std::array<bool, kMoonEpCombineV2ServerGrantSignalCount> shardSeen {};
    for (uint32_t core = 0U; core < kMoonEpCombineV2CoreCount; ++core) {
        uint32_t shardCount = 0U;
        for (uint32_t sourceCardLane = 0U;
            sourceCardLane < kMoonEpCombineV2ServerGrantSourceCardCount;
            ++sourceCardLane) {
            const uint32_t ordinal = MoonEpCombineV2ServerGrantOrdinal(
                sourceCardLane, core);
            Check(ordinal == sourceCardLane * kMoonEpCombineV2CoreCount + core,
                "Server-Grant core shard ordinal mismatch");
            Check(ordinal < shardSeen.size() && !shardSeen[ordinal],
                "Server-Grant core shards overlap");
            shardSeen[ordinal] = true;
            ++shardCount;
        }
        Check(shardCount == kMoonEpCombineV2ServerGrantSignalsPerCore,
            "Server-Grant core does not own exactly four signals");
    }
    for (bool seen : shardSeen) {
        Check(seen, "Server-Grant core shards do not cover all ordinals");
    }

    for (uint32_t rankSize : rankSizes) {
        const uint32_t phaseStepCount = rankSize / 32U;
        const uint32_t serverCount = rankSize / kMoonEpCombineV2GroupSize;
        for (uint32_t sourceServer = 0U;
            sourceServer < serverCount; ++sourceServer) {
            const uint32_t sourceRank =
                sourceServer * kMoonEpCombineV2GroupSize;
            const uint32_t targetServer =
                MoonEpCombineV2ServerGrantTargetServer(sourceRank, rankSize);
            Check(targetServer < serverCount &&
                    MoonEpCombineV2ServerGrantSourceServer(
                        targetServer * kMoonEpCombineV2GroupSize,
                        rankSize) == sourceServer,
                "Server-Grant source/target server inverse mismatch");
            Check(targetServer / phaseStepCount ==
                    sourceServer / phaseStepCount,
                "Server-Grant target escaped its quarter");
            for (uint32_t phaseStep = 0U;
                phaseStep < phaseStepCount; ++phaseStep) {
                const uint32_t currentPair =
                    (sourceServer % phaseStepCount + phaseStep) %
                        phaseStepCount;
                const uint32_t nextPhaseStep =
                    (phaseStep + 1U) % phaseStepCount;
                const uint32_t targetNextPair =
                    (targetServer % phaseStepCount + nextPhaseStep) %
                        phaseStepCount;
                Check(currentPair == targetNextPair,
                    "Server-Grant target violates next-step pair invariant");
            }
        }

        for (uint32_t targetRank = 0U;
            targetRank < rankSize; ++targetRank) {
            const uint32_t expectedSourceServer =
                MoonEpCombineV2ServerGrantSourceServer(targetRank, rankSize);
            const uint32_t targetCardLane =
                (targetRank % kMoonEpCombineV2GroupSize) / 2U;
            std::array<bool, kMoonEpCombineV2ServerGrantSignalCount>
                publishers {};
            for (uint32_t sourceCardLane = 0U;
                sourceCardLane < kMoonEpCombineV2ServerGrantSourceCardCount;
                ++sourceCardLane) {
                const uint32_t sourceRank =
                    MoonEpCombineV2ServerGrantSourceRank(
                        targetRank, sourceCardLane, rankSize);
                Check(sourceRank / kMoonEpCombineV2GroupSize ==
                        expectedSourceServer &&
                        (sourceRank & 1U) == (targetRank & 1U),
                    "Server-Grant publisher group mismatch");
                Check(MoonEpCombineV2ServerGrantTargetRank(
                        sourceRank, targetCardLane, rankSize) == targetRank,
                    "Server-Grant publisher/waiter inverse mismatch");
                Check(MoonEpCombineV2ServerGrantSourceCardLane(
                        sourceRank, rankSize) == sourceCardLane,
                    "Server-Grant source card lane mismatch");
                for (uint32_t sourceCore = 0U;
                    sourceCore < kMoonEpCombineV2CoreCount; ++sourceCore) {
                    const uint32_t ordinal = MoonEpCombineV2ServerGrantOrdinal(
                        sourceCardLane, sourceCore);
                    Check(ordinal < publishers.size() && !publishers[ordinal],
                        "Server-Grant target has a duplicate publisher");
                    publishers[ordinal] = true;
                }
            }
            for (bool published : publishers) {
                Check(published,
                    "Server-Grant target does not have 64 publishers");
            }

            const uint32_t directSourceCardLane =
                MoonEpCombineV2DirectGrantSourceCardLane(
                    targetRank, rankSize);
            const uint32_t directSourceRank =
                MoonEpCombineV2DirectGrantSourceRank(targetRank, rankSize);
            const uint32_t directTargetRank =
                MoonEpCombineV2DirectGrantTargetRank(
                    directSourceRank, rankSize);
            Check(directSourceCardLane == targetCardLane,
                "direct Grant does not select the target card lane");
            Check(directSourceRank < rankSize &&
                    directSourceRank % kMoonEpCombineV2GroupSize ==
                        targetRank % kMoonEpCombineV2GroupSize,
                "direct Grant source does not preserve local rank");
            Check(MoonEpCombineV2ServerGrantTargetRank(
                    directSourceRank, targetCardLane, rankSize) == targetRank,
                "direct Grant source/target mapping is not invertible");
            Check(directTargetRank == targetRank,
                "direct Grant does not have one target rank");
            Check((rankSize == 32U) ==
                    (directTargetRank == directSourceRank),
                "direct Grant local/remote target contract mismatch");
            Check(MoonEpCombineV2DirectGrantSourceRank(
                    directTargetRank, rankSize) == directSourceRank,
                "direct Grant target/source helper inverse mismatch");
            for (uint32_t core = 0U;
                core < kMoonEpCombineV2CoreCount; ++core) {
                Check(MoonEpCombineV2DirectGrantOrdinal(
                        targetRank, core, rankSize) ==
                        directSourceCardLane *
                            kMoonEpCombineV2CoreCount + core,
                    "direct Grant ordinal does not select one source/core");
            }
        }


        const auto mode = MOONEP_COMBINE_V2_SERVER_PAIR_PARITY;
        for (uint32_t targetRank = 0U;
            targetRank < rankSize; ++targetRank) {
            const uint32_t directSourceRank =
                MoonEpCombineV2DirectGrantSourceRank(targetRank, rankSize);
            for (uint32_t phase = 0U; phase < 2U; ++phase) {
                for (uint32_t phaseStep = 0U;
                    phaseStep + 1U < phaseStepCount; ++phaseStep) {
                    const uint32_t sourceRound =
                        phase * phaseStepCount + phaseStep;
                    const uint32_t targetRound = sourceRound + 1U;
                    for (uint32_t core = 0U;
                        core < kMoonEpCombineV2CoreCount; ++core) {
                        Check(MoonEpCombineV2ServerPairPeer(
                                directSourceRank, sourceRound, core,
                                rankSize, mode) ==
                                MoonEpCombineV2ServerPairPeer(
                                    targetRank, targetRound, core,
                                    rankSize, mode),
                            "direct Grant does not preserve the next-step peer");
                    }
                }
            }
        }
    }

    Check(MoonEpCombineV2DirectGrantTargetRank(0U, 16U) ==
            kMoonEpCombineV2InvalidPeer &&
            MoonEpCombineV2DirectGrantTargetRank(32U, 32U) ==
                kMoonEpCombineV2InvalidPeer,
        "direct Grant target accepts an unsupported size or invalid rank");

    const uint64_t guard = MoonEpCombineV2ServerGrantGuard(
        17U, 23U, 7U, 128U, 6U);
    Check(guard != MoonEpCombineV2ServerGrantGuard(
            19U, 23U, 7U, 128U, 6U) &&
            guard != MoonEpCombineV2ServerGrantGuard(
                17U, 22U, 7U, 128U, 6U) &&
            guard != MoonEpCombineV2ServerGrantGuard(
                17U, 23U, 6U, 128U, 6U) &&
            guard != MoonEpCombineV2ServerGrantGuard(
                17U, 23U, 7U, 64U, 6U) &&
            guard != MoonEpCombineV2ServerGrantGuard(
                17U, 23U, 7U, 128U, 4U),
        "Server-Grant guard omits a stale-rejection field");
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
    Check(MoonEpCombineV2LegacyGrantEnabled(2U),
        "2P must preserve Legacy Grant");
    Check(MoonEpCombineV2LegacyGrantEnabled(16U),
        "16P must preserve Legacy Grant");
    Check(!MoonEpCombineV2LegacyGrantEnabled(32U),
        "32P must disable Legacy Grant");
    Check(!MoonEpCombineV2LegacyGrantEnabled(64U),
        "64P must disable Legacy Grant");
    Check(!MoonEpCombineV2LegacyGrantEnabled(128U),
        "128P must disable Legacy Grant");
    Check(!MoonEpCombineV2LegacyGrantEnabled(1U),
        "unsupported rank sizes must not enable Legacy Grant");

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

void TestVectorSelectionReferenceModel()
{
    using namespace TileXRMoonEp;
    const uint32_t slots = 2040U;

    std::vector<int32_t> noMatches(23U, -1);
    Check(SelectPeerIndicesReference(noMatches, 3U, slots).empty(),
        "vector selection produced an index for an empty tail chunk");

    std::vector<int32_t> mixed(kMoonEpCombineV2SelectionChunkRows, -1);
    std::vector<uint32_t> expected;
    for (uint32_t index = 7U; index < mixed.size(); index += 37U) {
        mixed[index] = static_cast<int32_t>(5U * slots + index % slots);
        expected.push_back(index);
    }
    mixed[0] = static_cast<int32_t>(5U * slots);
    expected.insert(expected.begin(), 0U);
    mixed[1] = static_cast<int32_t>(6U * slots);
    const std::vector<uint32_t> selected = SelectPeerIndicesReference(
        mixed, 5U, slots);
    Check(selected == expected,
        "vector selection did not preserve compacted chunk indices");

    std::vector<int32_t> allMatches(1024U);
    for (uint32_t index = 0U; index < allMatches.size(); ++index) {
        allMatches[index] = static_cast<int32_t>(2U * slots + index);
    }
    const std::vector<uint32_t> allSelected = SelectPeerIndicesReference(
        allMatches, 2U, slots);
    Check(allSelected.size() == allMatches.size(),
        "vector selection did not retain every matching index");
    std::array<bool, 1024U> seen {};
    for (uint32_t relative : allSelected) {
        Check(relative < seen.size() && !seen[relative],
            "vector selection duplicated an all-match index");
        if (relative < seen.size()) {
            seen[relative] = true;
        }
    }
}

void TestFullSyncScheduleContract()
{
    using namespace TileXRMoonEp;
    const uint32_t supported[] = {2U, 3U, 4U, 5U, 6U, 7U, 8U,
        16U, 32U, 64U, 128U};
    const MoonEpCombineV2ScheduleMode mode =
        MOONEP_COMBINE_V2_SERVER_PAIR_SAME_CROSS;

    for (uint32_t rankSize : supported) {
        const uint32_t activeCores = MoonEpCombineV2ActiveCoreCount(rankSize);
        const uint32_t stepCount = MoonEpCombineV2StepCount(rankSize);
        for (uint32_t rank = 0U; rank < rankSize; ++rank) {
            std::array<bool, kMoonEpCombineV2RankCount> sent {};
            std::array<bool, kMoonEpCombineV2RankCount> waited {};
            uint32_t sendCount = 0U;
            uint32_t waitCount = 0U;
            for (uint32_t core = 0U; core < activeCores; ++core) {
                std::array<bool, kMoonEpCombineV2RankCount> corePeers {};
                uint32_t corePeerCount = 0U;
                for (uint32_t step = 0U; step < stepCount; ++step) {
                    const uint32_t peer = MoonEpCombineV2Peer(
                        rank, step, core, rankSize, mode);
                    if (peer == rank || peer == kMoonEpCombineV2InvalidPeer) {
                        continue;
                    }
                    Check(peer < rankSize && !sent[peer],
                        "full-sync send peer is invalid or repeated");
                    sent[peer] = true;
                    corePeers[peer] = true;
                    ++corePeerCount;
                    ++sendCount;
                }
                Check(corePeerCount <= kMoonEpCombineV2FullSyncMaxPeersPerCore,
                    "full-sync core peer count exceeds SIMT batch capacity");

                for (uint32_t peer = 0U; peer < rankSize; ++peer) {
                    if (!corePeers[peer]) {
                        continue;
                    }
                    Check(peer != rank && !waited[peer],
                        "full-sync wait source is invalid or repeated");
                    Check(MoonEpCombineV2SenderCore(
                            peer, rank, rankSize, mode) < activeCores,
                        "full-sync wait peer has no inverse sender core");
                    waited[peer] = true;
                    ++waitCount;
                }
            }
            Check(sendCount == rankSize - 1U,
                "full-sync send set does not cover every remote rank");
            Check(waitCount == rankSize - 1U,
                "full-sync wait set does not cover every remote rank");
            Check(!sent[rank] && !waited[rank],
                "full-sync send or wait set contains self");

            for (uint32_t peer = 0U; peer < rankSize; ++peer) {
                if (peer == rank) {
                    Check(MoonEpCombineV2SenderCore(
                            rank, peer, rankSize, mode) ==
                            kMoonEpCombineV2InvalidPeer,
                        "full-sync sender-core helper accepted self");
                    continue;
                }
                const uint32_t senderCore = MoonEpCombineV2SenderCore(
                    rank, peer, rankSize, mode);
                Check(senderCore < activeCores,
                    "full-sync sender-core helper returned invalid core");
                uint32_t matches = 0U;
                for (uint32_t step = 0U; step < stepCount; ++step) {
                    matches += MoonEpCombineV2Peer(rank, step, senderCore,
                        rankSize, mode) == peer ? 1U : 0U;
                }
                Check(matches == 1U,
                    "full-sync sender-core helper does not invert peer mapping");
            }
        }
    }

    Check(MoonEpCombineV2FullSyncReceiveIndex(0U, 0U, 17U) == 17U &&
        MoonEpCombineV2FullSyncReceiveIndex(0U, 1U, 17U) ==
            kMoonEpCombineV2RankCount + 17U &&
        MoonEpCombineV2FullSyncReceiveIndex(1U, 0U, 17U) ==
            2U * kMoonEpCombineV2RankCount + 17U,
        "full-sync receive epoch/generation index mismatch");
    Check(MoonEpCombineV2FullSyncCoreIndex(0U, 0U, 7U) == 7U &&
        MoonEpCombineV2FullSyncCoreIndex(0U, 1U, 7U) ==
            kMoonEpCombineV2CoreCount + 7U &&
        MoonEpCombineV2FullSyncCoreIndex(1U, 0U, 7U) ==
            2U * kMoonEpCombineV2CoreCount + 7U,
        "full-sync core epoch/generation index mismatch");
    Check(MoonEpCombineV2RoundBoundary(0U, 128U) == 1U &&
        MoonEpCombineV2RoundBoundary(3U, 128U) == 4U &&
        MoonEpCombineV2RoundBoundary(4U, 128U) == 6U &&
        MoonEpCombineV2RoundBoundary(7U, 128U) == 9U,
        "128-rank boundary numbering mismatch");
    Check(MoonEpCombineV2RoundBoundary(0U, 32U) == 1U &&
        MoonEpCombineV2RoundBoundary(1U, 32U) == 3U &&
        MoonEpCombineV2RoundBoundary(0U, 16U) == 1U,
        "boundary numbering does not reserve the phase slot");
    Check(MoonEpCombineV2PhaseBoundary(32U) == 2U &&
            MoonEpCombineV2PhaseBoundary(64U) == 3U &&
            MoonEpCombineV2PhaseBoundary(128U) == 5U &&
            MoonEpCombineV2PhaseBoundary(16U) ==
                kMoonEpCombineV2InvalidPeer,
        "phase boundary numbering mismatch");
    Check(MoonEpCombineV2PhaseBarrierOrdinal(128U) == 5U &&
            MoonEpCombineV2RoundBarrierOrdinal(3U, 128U) == 4U &&
            MoonEpCombineV2RoundBarrierOrdinal(4U, 128U) == 6U &&
            MoonEpCombineV2RoundBarrierOrdinal(0U, 16U) == 1U &&
            MoonEpCombineV2RoundBarrierOrdinal(8U, 128U) ==
                kMoonEpCombineV2InvalidPeer,
        "R4 barrier ordinal mismatch");
    Check(MoonEpCombineV2PhaseBarrierAfterRound(0U, 32U) &&
            MoonEpCombineV2PhaseBarrierAfterRound(1U, 64U) &&
            MoonEpCombineV2PhaseBarrierAfterRound(3U, 128U) &&
            !MoonEpCombineV2PhaseBarrierAfterRound(4U, 128U) &&
            !MoonEpCombineV2PhaseBarrierAfterRound(0U, 16U),
        "phase barrier insertion point mismatch");

    const uint32_t phaseRankSizes[] = {32U, 64U, 128U};
    for (uint32_t rankSize : phaseRankSizes) {
        std::vector<uint32_t> boundaries {0U};
        std::vector<uint32_t> ordinals {0U};
        const uint32_t stepCount = MoonEpCombineV2StepCount(rankSize);
        for (uint32_t round = 0U; round < stepCount; ++round) {
            boundaries.push_back(static_cast<uint32_t>(
                MoonEpCombineV2RoundBoundary(round, rankSize)));
            ordinals.push_back(
                MoonEpCombineV2RoundBarrierOrdinal(round, rankSize));
            if (MoonEpCombineV2PhaseBarrierAfterRound(round, rankSize)) {
                boundaries.push_back(MoonEpCombineV2PhaseBoundary(rankSize));
                ordinals.push_back(
                    MoonEpCombineV2PhaseBarrierOrdinal(rankSize));
            }
        }
        Check(boundaries.size() == stepCount + 2U &&
                ordinals.size() == boundaries.size(),
            "R4 barrier sequence size mismatch");
        for (uint32_t index = 0U; index < boundaries.size(); ++index) {
            Check(boundaries[index] == index && ordinals[index] == index,
                "R4 barrier identity and execution order diverged");
            if (index != 0U) {
                Check(MoonEpCombineV2FullSyncGeneration(ordinals[index]) !=
                        MoonEpCombineV2FullSyncGeneration(
                            ordinals[index - 1U]),
                    "adjacent R4 barriers reuse one generation");
            }
            if (index >= 2U) {
                Check(MoonEpCombineV2FullSyncGuard(
                            17U, boundaries[index], 23U, 7U, rankSize) !=
                        MoonEpCombineV2FullSyncGuard(
                            17U, boundaries[index - 2U], 23U, 7U, rankSize),
                    "R4 guard accepts a stale same-generation boundary");
            }
        }
    }
    const uint64_t boundaryGuard = MoonEpCombineV2FullSyncGuard(
        17U, 6U, 23U, 7U, 128U);
    Check(boundaryGuard != MoonEpCombineV2FullSyncGuard(
            17U, 4U, 23U, 7U, 128U),
        "full-sync guard accepts a stale same-generation boundary");
    Check(MoonEpCombineV2CollectiveStatusIndex(0U, 17U) == 1U &&
        MoonEpCombineV2CollectiveStatusIndex(1U, 17U) ==
            kMoonEpCombineV2CollectiveStatusSlotCount + 1U,
        "collective status slot reuse mismatch");
    Check(MoonEpCombineV2Epoch(1U) == 1U &&
        MoonEpCombineV2Epoch(2U) == 0U &&
        MoonEpCombineV2Epoch(3U) == 1U,
        "full-sync epoch sequence is not ping-pong");
}

void TestFullmeshRouting()
{
    using namespace TileXRMoonEp;
    const uint32_t rankSizes[] = {2U, 8U, 16U, 32U, 64U, 128U};
    for (uint32_t rankSize : rankSizes) {
        const uint32_t localRankSize = MoonEpCombineV2LocalRankSize(rankSize);
        for (uint32_t rank = 0U; rank < rankSize; ++rank) {
            const uint32_t localRank = rank % localRankSize;
            uint32_t peerMask = 0U;
            uint32_t localRemoteCount = 0U;
            for (uint32_t peer = 0U; peer < rankSize; ++peer) {
                const bool sameServer = rank / localRankSize ==
                    peer / localRankSize;
                Check(MoonEpCombineV2SameServer(
                        rank, peer, localRankSize) == sameServer,
                    "same-server classification mismatch");
                Check(MoonEpCombineV2LocalSlot(peer, localRankSize) ==
                        peer % localRankSize,
                    "Fullmesh local slot mismatch");
                Check(MoonEpCombineV2FullmeshLogicalQp(
                        peer, localRankSize) ==
                        kMoonEpCombineV2FullmeshLogicalQpBase +
                            peer % localRankSize,
                    "Fullmesh logical QP mismatch");
                const uint32_t expectedDone = peer == rank ? 0U :
                    (sameServer ? 1U : 2U);
                Check(MoonEpCombineV2ExpectedDoneCount(
                        peer, rank, localRankSize) == expectedDone,
                    "locality-aware done count mismatch");
                for (uint32_t lane = 0U;
                    lane < kMoonEpCombineV2LaneCount; ++lane) {
                    Check(MoonEpCombineV2DoneLaneRequired(
                            peer, rank, lane, localRankSize) ==
                            (lane < expectedDone),
                        "locality-aware done lane mismatch");
                }
                if (sameServer && peer != rank) {
                    peerMask |= 1U << (peer % localRankSize);
                    ++localRemoteCount;
                }
            }
            Check(localRemoteCount == localRankSize - 1U,
                "Fullmesh peer count mismatch");
            Check(peerMask == TileXR::UDMAFullmeshExpectedPeerMask(
                    localRank, localRankSize),
                "Fullmesh peer mask mismatch");
        }
    }
    Check(!MoonEpCombineV2SameServer(0U, 1U, 0U),
        "zero local-rank size accepted");
    Check(MoonEpCombineV2LocalSlot(1U, 9U) ==
            kMoonEpCombineV2InvalidPeer,
        "oversized Fullmesh domain accepted");
}

} // namespace

int main()
{
    TestSchedule();
    TestBidirectionalSchedule();
    TestServerPairSchedule();
    TestServerPairParitySchedule();
    TestServerGrantSchedule();
    TestFullGrantRounds();
    TestQpAndBatchContract();
    TestTokensAndShapes();
    TestWqeBatchHelpers();
    TestVectorSelectionReferenceModel();
    TestFullSyncScheduleContract();
    TestFullmeshRouting();
    return failures == 0 ? 0 : 1;
}
