#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "ep_plan_reference.h"

namespace {

using TileXREp::Plan::ReferenceExpertMove;
using TileXREp::Plan::ReferenceInput;
using TileXREp::Plan::ReferenceOutput;
using TileXREp::Plan::ReferenceRankPair;
using TileXREp::Plan::TokenSourceAssignment;

int g_failures = 0;

void Check(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void CheckEq(int64_t actual, int64_t expected, const std::string &message)
{
    if (actual != expected) {
        std::cerr << message << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

TileXRMoonEPPlanConfig MakeConfig(int64_t cap, int64_t nvS, int64_t b = 2, int64_t padding = 1,
    int64_t routeLimit = 0)
{
    TileXRMoonEPPlanConfig config {};
    config.prefetchSlots = b;
    config.rankTokenCapacity = cap;
    config.nvS = nvS;
    config.tokenPadding = padding;
    config.tokenRouteLimitPerPair = routeLimit;
    config.cardsPerServer = 8;
    config.cardsPerCabinet = 64;
    config.crossCandidateCount = 3;
    return config;
}

ReferenceInput MakeInput(int32_t rankSize, int64_t s, int64_t k, int64_t expertNum,
    const std::vector<int32_t> &globalRankIds, const std::vector<int32_t> &topkExperts,
    int64_t b = 2, int64_t padding = 1, int64_t nvS = -1, int64_t routeLimit = 0)
{
    ReferenceInput input {};
    input.rankSize = rankSize;
    input.s = s;
    input.topK = k;
    input.expertNum = expertNum;
    input.globalRankIds = globalRankIds;
    input.topkExperts = topkExperts;
    const int64_t cap = s * k;
    input.config = MakeConfig(cap, nvS < 0 ? cap : nvS, b, padding, routeLimit);
    input.tokensPerExpert.assign(static_cast<size_t>(rankSize * expertNum), 0);
    for (int32_t rank = 0; rank < rankSize; ++rank) {
        for (int64_t i = 0; i < cap; ++i) {
            const int32_t expert = topkExperts[static_cast<size_t>(rank * cap + i)];
            if (expert >= 0 && expert < expertNum) {
                ++input.tokensPerExpert[static_cast<size_t>(rank * expertNum + expert)];
            }
        }
    }
    return input;
}

ReferenceInput MakeHomeLoadInput(const std::vector<int32_t> &homeLoads,
    const std::vector<int32_t> &globalRankIds, int32_t cap, int64_t b = 2,
    int64_t padding = 1, int64_t nvS = -1, int64_t routeLimit = 0)
{
    const int32_t rankSize = static_cast<int32_t>(homeLoads.size());
    std::vector<int32_t> remaining = homeLoads;
    std::vector<int32_t> routes;
    routes.reserve(static_cast<size_t>(rankSize * cap));
    int32_t expert = 0;
    for (int32_t src = 0; src < rankSize; ++src) {
        for (int32_t i = 0; i < cap; ++i) {
            while (expert < rankSize && remaining[static_cast<size_t>(expert)] == 0) {
                ++expert;
            }
            Check(expert < rankSize, "home load generator exhausted experts");
            if (expert >= rankSize) {
                routes.push_back(0);
            } else {
                routes.push_back(expert);
                --remaining[static_cast<size_t>(expert)];
            }
        }
    }
    Check(std::accumulate(remaining.begin(), remaining.end(), int64_t {0}) == 0,
        "home load generator must consume every route");
    return MakeInput(rankSize, cap, 1, rankSize, globalRankIds, routes, b, padding, nvS, routeLimit);
}

void TestValidation()
{
    ReferenceInput valid = MakeInput(2, 2, 1, 2, {0, 8}, {0, 0, 1, 1}, 1);
    ReferenceOutput output;
    CheckEq(TileXREp::Plan::BuildReferencePlan(valid, &output), PLAN_OK,
        "valid reference plan must pass");
    Check(TileXREp::Plan::BuildReferencePlan(valid, nullptr) == PLAN_ERROR_CONFIG_MISMATCH,
        "null output must fail");

    ReferenceInput duplicateIds = valid;
    duplicateIds.globalRankIds = {0, 0};
    Check(TileXREp::Plan::BuildReferencePlan(duplicateIds, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "duplicate global rank ids must fail");

    ReferenceInput negativeGlobalRank = valid;
    negativeGlobalRank.globalRankIds[0] = -1;
    Check(TileXREp::Plan::BuildReferencePlan(negativeGlobalRank, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "negative global rank id must fail");

    ReferenceInput invalidExpert = valid;
    invalidExpert.topkExperts[0] = 2;
    Check(TileXREp::Plan::BuildReferencePlan(invalidExpert, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "out-of-range expert id must fail");

    ReferenceInput negativeExpert = valid;
    negativeExpert.topkExperts[0] = -1;
    Check(TileXREp::Plan::BuildReferencePlan(negativeExpert, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "negative expert id must fail");

    ReferenceInput negativeTpe = valid;
    negativeTpe.tokensPerExpert[0] = -1;
    Check(TileXREp::Plan::BuildReferencePlan(negativeTpe, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "negative tokensPerExpert must fail");

    ReferenceInput mismatchedTpe = valid;
    ++mismatchedTpe.tokensPerExpert[0];
    Check(TileXREp::Plan::BuildReferencePlan(mismatchedTpe, &output) == PLAN_ERROR_TPE_MISMATCH,
        "histogram mismatch must fail");

    ReferenceInput nonDivisible = valid;
    nonDivisible.expertNum = 3;
    nonDivisible.tokensPerExpert.resize(6, 0);
    Check(TileXREp::Plan::BuildReferencePlan(nonDivisible, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "E modulo R mismatch must fail");

    ReferenceInput invalidTopology = valid;
    invalidTopology.config.cardsPerServer = 4;
    Check(TileXREp::Plan::BuildReferencePlan(invalidTopology, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "invalid topology constants must fail");

    ReferenceInput invalidLimit = valid;
    invalidLimit.config.tokenRouteLimitPerPair = 3;
    Check(TileXREp::Plan::BuildReferencePlan(invalidLimit, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "route limit above CAP must fail");

    ReferenceInput extraSlots = valid;
    extraSlots.config.prefetchSlots = 3;
    Check(TileXREp::Plan::BuildReferencePlan(extraSlots, &output) == PLAN_OK,
        "B greater than E remains a valid explicit workspace shape");

    ReferenceInput zeroSlots = valid;
    zeroSlots.config.prefetchSlots = 0;
    Check(TileXREp::Plan::BuildReferencePlan(zeroSlots, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "B == 0 must fail");

    ReferenceInput largePadding = MakeInput(1, 1, 1, 1, {0}, {0}, 1,
        static_cast<int64_t>(INT32_MAX) + 1, 1);
    Check(TileXREp::Plan::BuildReferencePlan(largePadding, &output) == PLAN_ERROR_LAYOUT_EXCEEDS_NVS,
        "positive padding above INT32 remains valid config and must fail only when padded layout exceeds NvS");

    ReferenceInput negativeShape = valid;
    negativeShape.s = -1;
    Check(TileXREp::Plan::BuildReferencePlan(negativeShape, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "negative shape must fail");

    ReferenceInput shortInput = valid;
    shortInput.topkExperts.pop_back();
    Check(TileXREp::Plan::BuildReferencePlan(shortInput, &output) == PLAN_ERROR_CONFIG_MISMATCH,
        "short top-k input must fail");

    std::vector<int32_t> logicalRankIds(512);
    std::vector<int32_t> logicalRoutes(512);
    for (int32_t rank = 0; rank < 512; ++rank) {
        logicalRankIds[static_cast<size_t>(rank)] = rank;
        logicalRoutes[static_cast<size_t>(rank)] = rank;
    }
    ReferenceInput logical512 = MakeInput(512, 1, 1, 512,
        logicalRankIds, logicalRoutes, 1, 1, 1);
    Check(TileXREp::Plan::BuildReferencePlan(logical512, &output) == PLAN_OK,
        "logical 512-rank CPU reference must succeed");

    std::vector<int32_t> uniqueRoutes(32);
    std::iota(uniqueRoutes.begin(), uniqueRoutes.end(), 0);
    ReferenceInput maxLocalSegments = MakeInput(1, 1, 32, 32, {0}, uniqueRoutes, 1, 1, 32);
    Check(TileXREp::Plan::BuildReferencePlan(maxLocalSegments, &output) == PLAN_OK,
        "exact S*K local Token Segment bound must succeed");
    CheckEq(output.segments.size(), 32, "local Token Segment count must reach S*K bound");
    CheckEq(output.statusByRank[7], 32, "status local Token Segment count must report S*K bound");
}

void TestIntraServerScenarios()
{
    const int32_t cap = 100;
    std::vector<ReferenceRankPair> pairs;
    Check(TileXREp::Plan::BuildIntraServerPairs(
        {150, 140, 100, 100, 80, 70, 100, 100},
        {664, 665, 666, 667, 668, 669, 670, 671}, cap, &pairs),
        "standard intra-server pair build must succeed");
    CheckEq(pairs.size(), 2, "standard scenario pair count");
    if (pairs.size() == 2) {
        CheckEq(pairs[0].srcRank, 1, "140 sender must pair first");
        CheckEq(pairs[0].dstRank, 4, "80 receiver must pair first");
        CheckEq(pairs[0].requestedUnits, 20, "140->80 units");
        CheckEq(pairs[1].srcRank, 0, "150 sender must pair second");
        CheckEq(pairs[1].dstRank, 5, "70 receiver must pair second");
        CheckEq(pairs[1].requestedUnits, 30, "150->70 units");
    }

    pairs.clear();
    Check(TileXREp::Plan::BuildIntraServerPairs(
        {180, 55, 50, 45, 40, 35, 30, 25},
        {664, 665, 666, 667, 668, 669, 670, 671}, cap, &pairs),
        "middle-line scenario pair build must succeed");
    CheckEq(pairs.size(), 1, "middle-line scenario pair count");
    if (!pairs.empty()) {
        CheckEq(pairs[0].srcRank, 0, "180 must be sender");
        CheckEq(pairs[0].dstRank, 2, "180 must pair with 50, not 55");
        CheckEq(pairs[0].requestedUnits, 50, "180->50 units");
    }

    pairs.clear();
    Check(TileXREp::Plan::BuildIntraServerPairs({40, 30}, {0, 1}, cap, &pairs),
        "low-load guard call must succeed");
    Check(pairs.empty(), "low-load guard must produce no pairs");
    Check(TileXREp::Plan::BuildIntraServerPairs({130, 120}, {0, 1}, cap, &pairs),
        "all-overload guard call must succeed");
    Check(pairs.empty(), "all-overload guard must produce no pairs");
}

void TestStaticTopThree()
{
    const std::vector<int32_t> affinity = {3, 0, 1, 2, 4};
    std::vector<int32_t> candidates;
    Check(TileXREp::Plan::BuildStaticTopCandidates(3, {0, 1, 2, 4}, affinity, &candidates),
        "top-three build must succeed");
    Check(candidates == std::vector<int32_t>({0, 1, 2}),
        "static top-three must filter only sender-group membership");
}

void TestTokenSourceScenarios()
{
    std::vector<int32_t> remaining {50, 40, 100};
    std::vector<int32_t> routed(3, 0);
    std::vector<int32_t> cursors(3, 0);
    int32_t dstCursor = 0;
    TokenSourceAssignment assignment;
    Check(TileXREp::Plan::AssignTokenSources(7, 0, 30, 0, {0, 1, 2},
        &remaining, &routed, &cursors, &dstCursor, &assignment),
        "single-source token assignment must succeed");
    CheckEq(assignment.actualAssigned, 30, "single-source assigned count");
    CheckEq(assignment.segments.size(), 1, "single-source segment count");
    CheckEq(assignment.segments[0].srcRank, 0, "nearest source must be selected");

    remaining = {20, 40, 100};
    routed.assign(3, 0);
    cursors.assign(3, 0);
    dstCursor = 0;
    assignment = TokenSourceAssignment {};
    Check(TileXREp::Plan::AssignTokenSources(3, 2, 90, 0, {0, 1, 2},
        &remaining, &routed, &cursors, &dstCursor, &assignment),
        "multi-source token assignment must succeed");
    CheckEq(assignment.actualAssigned, 90, "multi-source assigned count");
    CheckEq(assignment.segments.size(), 3, "multi-source segment count");
    CheckEq(assignment.segments[0].tokenCount, 20, "first source amount");
    CheckEq(assignment.segments[1].tokenCount, 40, "second source amount");
    CheckEq(assignment.segments[2].tokenCount, 30, "third source amount");

    remaining = {15, 25, 30};
    routed.assign(3, 0);
    cursors.assign(3, 0);
    dstCursor = 0;
    assignment = TokenSourceAssignment {};
    Check(TileXREp::Plan::AssignTokenSources(9, 2, 120, 0, {0, 1, 2},
        &remaining, &routed, &cursors, &dstCursor, &assignment),
        "partial token assignment call must succeed");
    CheckEq(assignment.actualAssigned, 70, "partial assignment must clamp to supply");
    CheckEq(assignment.unmetDemand, 50, "partial assignment unmet demand");
    Check(assignment.supplyExhausted, "partial assignment must report supply exhaustion");

    remaining = {100, 100};
    routed = {4, 0};
    cursors.assign(2, 0);
    dstCursor = 0;
    assignment = TokenSourceAssignment {};
    Check(TileXREp::Plan::AssignTokenSources(1, 1, 10, 5, {0, 1},
        &remaining, &routed, &cursors, &dstCursor, &assignment),
        "route-limited token assignment call must succeed");
    CheckEq(assignment.actualAssigned, 6, "route limit must be cumulative per pair");
    CheckEq(assignment.unmetDemand, 4, "route limit unmet demand");
    Check(assignment.routeLimited, "route-limited assignment must report route blocking");
}

void TestInterServerOccupiedAndDynamicSkip()
{
    ReferenceOutput output;
    ReferenceInput occupied = MakeHomeLoadInput(
        {150, 140, 130, 60, 20}, {0, 16, 32, 8, 9}, 100, 2);
    Check(TileXREp::Plan::BuildReferencePlan(occupied, &output) == PLAN_OK,
        "occupied scenario must fully balance");
    CheckEq(output.rankLoad[0], 100, "occupied src0 final load");
    CheckEq(output.rankLoad[1], 100, "occupied src1 final load");
    CheckEq(output.rankLoad[2], 100, "occupied src2 final load");
    CheckEq(output.rankLoad[3], 100, "occupied dst1 final load");
    CheckEq(output.rankLoad[4], 100, "occupied dst2 final load");
    Check(output.expertMoves.size() >= 3, "occupied scenario must record inter moves");
    if (output.expertMoves.size() >= 2) {
        CheckEq(output.expertMoves[0].dstRank, 4, "lowest receiver must be handled first");
        CheckEq(output.expertMoves[0].srcHomeRank, 0, "first receiver must choose top1");
        CheckEq(output.expertMoves[1].dstRank, 3, "second receiver must be handled second");
        CheckEq(output.expertMoves[1].srcHomeRank, 1, "occupied top1 must force top2");
        CheckEq(output.expertMoves[0].round, output.expertMoves[1].round,
            "occupied moves must occur in the same round");
    }

    ReferenceInput dynamicSkip = MakeHomeLoadInput(
        {190, 130, 95, 65, 20}, {32, 16, 0, 9, 8}, 100, 3);
    Check(TileXREp::Plan::BuildReferencePlan(dynamicSkip, &output) == PLAN_OK,
        "dynamic-skip scenario must fully balance");
    Check(!output.expertMoves.empty(), "dynamic-skip scenario must record moves");
    if (!output.expertMoves.empty()) {
        CheckEq(output.expertMoves[0].dstRank, 4, "dynamic-skip lowest receiver");
        CheckEq(output.expertMoves[0].srcHomeRank, 1,
            "zero-overflow static top1 must be skipped in favor of top2");
    }
}

void TestFusedAllocationAndPartialStatus()
{
    ReferenceOutput output;
    ReferenceInput slotLimited = MakeHomeLoadInput({140, 140, 20}, {0, 16, 8}, 100, 1);
    Check(TileXREp::Plan::BuildReferencePlan(slotLimited, &output) ==
        PLAN_PARTIAL_PREFETCH_SLOT_EXHAUSTED,
        "distinct remote experts beyond B must report slot exhaustion");
    Check(output.remoteStats[2 * 2] <= 1, "remote expert count must never exceed B");

    ReferenceInput routeLimited = MakeHomeLoadInput({200, 0}, {0, 8}, 100, 1, 1, 100, 20);
    Check(TileXREp::Plan::BuildReferencePlan(routeLimited, &output) ==
        PLAN_PARTIAL_NO_FEASIBLE_PAIR,
        "pair route limit must prevent fabricated completion");
    Check(output.statusByRank[3] > 0, "partial plan must expose unmet token count");
}

void TestLayoutDstDedupAndDeterminism()
{
    ReferenceOutput first;
    ReferenceOutput second;
    ReferenceInput input = MakeInput(2, 2, 2, 2, {0, 8},
        {0, 0, 0, 0, 0, 0, 0, 0}, 1, 1, 4);
    Check(TileXREp::Plan::BuildReferencePlan(input, &first) == PLAN_OK,
        "dedup plan must succeed");
    Check(TileXREp::Plan::BuildReferencePlan(input, &second) == PLAN_OK,
        "repeated dedup plan must succeed");
    Check(first.dst == second.dst && first.cuSeqlens == second.cuSeqlens &&
        first.expertsToCopy == second.expertsToCopy && first.remoteStats == second.remoteStats,
        "reference outputs must be byte-for-byte deterministic");
    CheckEq(first.dst.size(), 8, "global dst size");
    for (int32_t rank = 0; rank < 2; ++rank) {
        for (int32_t token = 0; token < 2; ++token) {
            const size_t base = static_cast<size_t>(rank * 4 + token * 2);
            Check(first.dst[base] >= 0, "first route to a rank must be non-negative");
            Check(first.dst[base + 1] < 0, "duplicate route to same rank must be negative");
            Check((~first.dst[base + 1]) / 4 == first.dst[base] / 4,
                "negative dst must decode to the same destination rank");
        }
    }
    std::string invariantError;
    Check(TileXREp::Plan::CheckReferencePlanInvariants(input, first, &invariantError),
        std::string("dedup invariants failed: ") + invariantError);

    ReferenceInput padded = MakeHomeLoadInput({3, 3}, {0, 8}, 3, 1, 4, 8);
    Check(TileXREp::Plan::BuildReferencePlan(padded, &first) == PLAN_OK,
        "padded layout must succeed when NvS is sufficient");
    CheckEq(first.cuSeqlens[0], 4, "home group must be padded to four rows");

    padded.config.nvS = 3;
    Check(TileXREp::Plan::BuildReferencePlan(padded, &first) == PLAN_ERROR_LAYOUT_EXCEEDS_NVS,
        "padded layout beyond NvS must fail explicitly");
}


void TestCuSeqlensEncodePadding()
{
    ReferenceOutput output;
    ReferenceInput input = MakeHomeLoadInput({3, 3}, {0, 8}, 3, 1, 4, 8);
    Check(TileXREp::Plan::BuildReferencePlan(input, &output) == PLAN_OK,
        "padded layout scenario must build");
    CheckEq(output.cuSeqlens[0], 4,
        "cuSeqlens must include aligned padding without a zero-fill side output");

    output.cuSeqlens[0] = 3;
    std::string invariantError;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, output, &invariantError),
        "invariant checker must reject a cumulative end that omits required padding");
    Check(invariantError == "cuSeqlens does not match group payload",
        "padding violation must have a precise cuSeqlens diagnostic");
}

void TestInvariantCheckerRejectsPlanOkRankCapacityViolation()
{
    ReferenceOutput output;
    ReferenceInput input = MakeHomeLoadInput({2, 2}, {0, 1}, 2, 1, 1, 2);
    Check(TileXREp::Plan::BuildReferencePlan(input, &output) == PLAN_OK,
        "capacity invariant scenario must build a valid plan");

    // Preserve every expert's global token count while making rank 0 under-capacity
    // and rank 1 over-capacity. A PLAN_OK invariant check must reject this state.
    --output.alloc[0];
    ++output.alloc[2];
    --output.rankLoad[0];
    ++output.rankLoad[1];

    std::string invariantError;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, output, &invariantError),
        "PLAN_OK invariant checker must reject rank loads that differ from CAP");
    Check(invariantError == "PLAN_OK rank capacity violated",
        "capacity violation must have a precise invariant diagnostic");
}

void TestInvariantCheckerRejectsOutputContractCorruption()
{
    ReferenceInput input = MakeHomeLoadInput({2, 2}, {0, 1}, 2, 1, 1, 2);
    ReferenceOutput valid;
    Check(TileXREp::Plan::BuildReferencePlan(input, &valid) == PLAN_OK,
        "output contract invariant scenario must build a valid plan");

    ReferenceOutput missingExpertSlot = valid;
    missingExpertSlot.expertsToCopy.pop_back();
    std::string invariantError;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, missingExpertSlot, &invariantError),
        "invariant checker must reject expertsToCopy shape corruption");
    Check(invariantError == "expertsToCopy shape mismatch",
        "expertsToCopy shape violation must have a precise invariant diagnostic");

    ReferenceOutput shortStatus = valid;
    shortStatus.statusByRank.pop_back();
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, shortStatus, &invariantError),
        "invariant checker must reject statusByRank shape corruption");
    Check(invariantError == "statusByRank shape mismatch",
        "status shape violation must have a precise invariant diagnostic");

    ReferenceOutput splitStatus = valid;
    splitStatus.statusByRank[TileXREp::Plan::kPlanStatusWords] = PLAN_PARTIAL_NO_FEASIBLE_PAIR;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, splitStatus, &invariantError),
        "invariant checker must reject status[0] disagreement between ranks");
    Check(invariantError == "status[0] consensus mismatch",
        "status consensus violation must have a precise invariant diagnostic");

    ReferenceOutput wrongFinalStatus = valid;
    for (int32_t rank = 0; rank < input.rankSize; ++rank) {
        wrongFinalStatus.statusByRank[static_cast<size_t>(rank * TileXREp::Plan::kPlanStatusWords)] =
            PLAN_PARTIAL_NO_FEASIBLE_PAIR;
    }
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, wrongFinalStatus, &invariantError),
        "invariant checker must reject status[0] that differs from finalStatus");
    Check(invariantError == "status[0] does not match final status",
        "final status violation must have a precise invariant diagnostic");
}

void TestInvariantCheckerRejectsLayoutSemanticCorruption()
{
    ReferenceOutput migrated;
    ReferenceInput migrationInput = MakeHomeLoadInput({120, 80}, {0, 1}, 100, 1, 1, 100);
    Check(TileXREp::Plan::BuildReferencePlan(migrationInput, &migrated) == PLAN_OK,
        "layout semantic invariant scenario must build a migrated plan");

    ReferenceOutput wrongExpertCopy = migrated;
    wrongExpertCopy.expertsToCopy[1] = 1;
    std::string invariantError;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(
        migrationInput, wrongExpertCopy, &invariantError),
        "invariant checker must reject expertsToCopy content corruption");
    Check(invariantError == "expertsToCopy content mismatch",
        "expertsToCopy content violation must have a precise invariant diagnostic");

    ReferenceOutput wrongExpertTargetsShape = migrated;
    wrongExpertTargetsShape.expertTargets.pop_back();
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(
        migrationInput, wrongExpertTargetsShape, &invariantError),
        "invariant checker must reject expertTargets shape corruption");
    Check(invariantError == "expertTargets shape mismatch",
        "expertTargets shape violation must have a precise invariant diagnostic");

    ReferenceOutput wrongExpertTargets = migrated;
    wrongExpertTargets.expertTargets[0] ^= 1ULL;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(
        migrationInput, wrongExpertTargets, &invariantError),
        "invariant checker must reject expertTargets bitmap corruption");
    Check(invariantError == "expertTargets content mismatch",
        "expertTargets content violation must have a precise invariant diagnostic");

    ReferenceOutput wrongRemoteStats = migrated;
    ++wrongRemoteStats.remoteStats[0];
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(
        migrationInput, wrongRemoteStats, &invariantError),
        "invariant checker must reject remoteStats corruption");
    Check(invariantError == "remoteStats mismatch",
        "remoteStats violation must have a precise invariant diagnostic");

    ReferenceOutput padded;
    ReferenceInput paddingInput = MakeHomeLoadInput({3, 3}, {0, 8}, 3, 1, 4, 8);
    Check(TileXREp::Plan::BuildReferencePlan(paddingInput, &padded) == PLAN_OK,
        "padding semantic invariant scenario must build a padded plan");
    padded.cuSeqlens[0] = 3;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(paddingInput, padded, &invariantError),
        "invariant checker must reject a cumulative end that omits required padding");
    Check(invariantError == "cuSeqlens does not match group payload",
        "padding semantic violation must have a precise invariant diagnostic");
}

void TestInvariantCheckerRejectsTokenSegmentPartitionCorruption()
{
    ReferenceInput input = MakeHomeLoadInput({120, 80}, {0, 1}, 100, 1, 1, 100);
    ReferenceOutput valid;
    Check(TileXREp::Plan::BuildReferencePlan(input, &valid) == PLAN_OK,
        "token segment invariant scenario must build a valid plan");
    Check(!valid.segments.empty(), "token segment invariant scenario must produce segments");
    if (valid.segments.empty()) {
        return;
    }

    ReferenceOutput wrongDestination = valid;
    wrongDestination.segments[0].dstRank = 1 - wrongDestination.segments[0].dstRank;
    std::string invariantError;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, wrongDestination, &invariantError),
        "invariant checker must reject destination/expert segment conservation corruption");
    Check(invariantError == "destination/expert segment conservation failed",
        "destination segment violation must have a precise invariant diagnostic");

    ReferenceOutput wrongSourceRange = valid;
    ++wrongSourceRange.segments[0].srcExpertBegin;
    Check(!TileXREp::Plan::CheckReferencePlanInvariants(input, wrongSourceRange, &invariantError),
        "invariant checker must reject non-contiguous source/expert segment ranges");
    Check(invariantError == "source/expert segment ranges are not contiguous",
        "source segment range violation must have a precise invariant diagnostic");
}

void TestStatusContract()
{
    ReferenceOutput output;
    ReferenceInput input = MakeHomeLoadInput({120, 80}, {0, 1}, 100, 1, 1, 100);
    Check(TileXREp::Plan::BuildReferencePlan(input, &output) == PLAN_OK,
        "status contract scenario must balance");
    CheckEq(output.statusByRank.size(), 2 * TileXREp::Plan::kPlanStatusWords,
        "status shape must be [R,8]");
    CheckEq(output.statusByRank[0], PLAN_OK, "rank0 status code");
    CheckEq(output.statusByRank[1], 1, "one intra-server migration round");
    CheckEq(output.statusByRank[2], 0, "no inter-server migration round");
    CheckEq(output.statusByRank[3], 0, "balanced plan has no unmet tokens");
    CheckEq(output.statusByRank[4], 1, "maximum remote expert count");
    const size_t groupCount = static_cast<size_t>(input.expertNum + input.config.prefetchSlots);
    CheckEq(output.statusByRank[5], output.cuSeqlens[groupCount - 1],
        "rank0 status[5] must be local padded end");
    CheckEq(output.statusByRank[6], output.expertMoves.size(),
        "status[6] must be global expert move count");
    const int32_t rank0Segments = static_cast<int32_t>(std::count_if(output.segments.begin(),
        output.segments.end(), [](const TileXREp::Plan::TokenSegmentMove &move) {
            return move.srcRank == 0;
        }));
    CheckEq(output.statusByRank[7], rank0Segments,
        "rank0 status[7] must be local retained segment count");

    const size_t rank1Base = TileXREp::Plan::kPlanStatusWords;
    CheckEq(output.statusByRank[rank1Base + 1], 1, "rank1 sees the same intra round count");
    CheckEq(output.statusByRank[rank1Base + 2], 0, "rank1 sees no inter round");
    CheckEq(output.statusByRank[rank1Base + 5], output.cuSeqlens[2 * groupCount - 1],
        "rank1 status[5] must be local padded end");
    const int32_t rank1Segments = static_cast<int32_t>(std::count_if(output.segments.begin(),
        output.segments.end(), [](const TileXREp::Plan::TokenSegmentMove &move) {
            return move.srcRank == 1;
        }));
    CheckEq(output.statusByRank[rank1Base + 7], rank1Segments,
        "rank1 status[7] must be local retained segment count");
}

} // namespace

int main()
{
    TestValidation();
    TestIntraServerScenarios();
    TestStaticTopThree();
    TestTokenSourceScenarios();
    TestInterServerOccupiedAndDynamicSkip();
    TestFusedAllocationAndPartialStatus();
    TestLayoutDstDedupAndDeterminism();
    TestCuSeqlensEncodePadding();
    TestInvariantCheckerRejectsPlanOkRankCapacityViolation();
    TestInvariantCheckerRejectsOutputContractCorruption();
    TestInvariantCheckerRejectsLayoutSemanticCorruption();
    TestInvariantCheckerRejectsTokenSegmentPartitionCorruption();
    TestStatusContract();
    if (g_failures != 0) {
        std::cerr << g_failures << " Plan reference tests failed" << std::endl;
        return 1;
    }
    std::cout << "Plan reference tests passed" << std::endl;
    return 0;
}
