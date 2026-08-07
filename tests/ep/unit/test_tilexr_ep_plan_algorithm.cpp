#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "ep_plan_algorithm.h"
#include "ep_plan_reference.h"

namespace {

using TileXREp::Plan::PlanAlgorithmInput;
using TileXREp::Plan::PlanAlgorithmOutput;
using TileXREp::Plan::PlanAlgorithmWorkspace;
using TileXREp::Plan::ReferenceInput;
using TileXREp::Plan::ReferenceOutput;
using TileXREp::Plan::TokenSegmentMove;

int g_failures = 0;

void Check(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

TileXRMoonEPPlanConfig MakeConfig(int64_t cap, int64_t nvS, int64_t b,
    int64_t padding = 1, int64_t routeLimit = 0)
{
    TileXRMoonEPPlanConfig config {};
    config.prefetchSlots = b;
    config.rankTokenCapacity = cap;
    config.nvS = nvS;
    config.tokenPadding = padding;
    config.tokenRouteLimitPerPair = routeLimit;
    config.cardsPerServer = TileXREp::Plan::kPlanCardsPerServer;
    config.cardsPerCabinet = TileXREp::Plan::kPlanCardsPerCabinet;
    config.crossCandidateCount = TileXREp::Plan::kPlanCrossCandidateCount;
    return config;
}

ReferenceInput MakeInput(int32_t rankSize, int64_t s, int64_t k, int64_t expertNum,
    const std::vector<int32_t> &globalRankIds, const std::vector<int32_t> &routes,
    int64_t b, int64_t padding = 1, int64_t nvS = -1, int64_t routeLimit = 0)
{
    ReferenceInput input {};
    input.rankSize = rankSize;
    input.s = s;
    input.topK = k;
    input.expertNum = expertNum;
    input.globalRankIds = globalRankIds;
    input.topkExperts = routes;
    const int64_t cap = s * k;
    input.config = MakeConfig(cap, nvS < 0 ? cap : nvS, b, padding, routeLimit);
    input.tokensPerExpert.assign(static_cast<size_t>(rankSize * expertNum), 0);
    for (int32_t rank = 0; rank < rankSize; ++rank) {
        for (int32_t route = 0; route < cap; ++route) {
            const int32_t expert = routes[static_cast<size_t>(rank * cap + route)];
            ++input.tokensPerExpert[static_cast<size_t>(rank * expertNum + expert)];
        }
    }
    return input;
}

ReferenceInput MakeHomeLoadInput(const std::vector<int32_t> &homeLoads,
    const std::vector<int32_t> &globalRankIds, int32_t cap, int64_t b,
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
            routes.push_back(expert);
            --remaining[static_cast<size_t>(expert)];
        }
    }
    return MakeInput(rankSize, cap, 1, rankSize, globalRankIds, routes,
        b, padding, nvS, routeLimit);
}

template <class T>
void CheckVector(const std::vector<T> &actual, const std::vector<T> &expected,
    const std::string &name)
{
    if (actual != expected) {
        std::cerr << name << " mismatch" << std::endl;
        ++g_failures;
    }
}

void CompareScenario(const ReferenceInput &input, const std::string &name,
    bool aliasLegacyOutputToCurrentRemoteRow = false)
{
    ReferenceOutput reference;
    const TileXRMoonEPPlanStatus expectedStatus =
        TileXREp::Plan::BuildReferencePlan(input, &reference);
    Check(expectedStatus == reference.finalStatus, name + ": reference status mismatch");

    const int32_t rankSize = input.rankSize;
    const int32_t expertNum = static_cast<int32_t>(input.expertNum);
    const int32_t cap = static_cast<int32_t>(input.config.rankTokenCapacity);
    const int32_t b = static_cast<int32_t>(input.config.prefetchSlots);
    const int32_t groupCount = expertNum + b;

    for (int32_t rank = 0; rank < rankSize; ++rank) {
        std::vector<int32_t> expertCount(static_cast<size_t>(expertNum));
        std::vector<int32_t> rankLoad(static_cast<size_t>(rankSize));
        std::vector<int32_t> remainingTpe(static_cast<size_t>(rankSize * expertNum));
        std::vector<int32_t> alloc(static_cast<size_t>(rankSize * expertNum));
        std::vector<int32_t> remoteExpertSet(static_cast<size_t>(rankSize * b));
        std::vector<int32_t> srcExpertCursor(static_cast<size_t>(rankSize * expertNum));
        std::vector<int32_t> dstExpertCursor(static_cast<size_t>(rankSize * expertNum));
        std::vector<int32_t> expertPhysicalBase(static_cast<size_t>(rankSize * expertNum));
        std::vector<int32_t> localExpertOrdinal(static_cast<size_t>(cap));
        std::vector<TokenSegmentMove> tokenSegments(static_cast<size_t>(cap));
        std::vector<int32_t> routedPairTokens(static_cast<size_t>(rankSize * rankSize));
        std::vector<int32_t> scratch(static_cast<size_t>(rankSize * 16));
        std::vector<int32_t> affinityOrder(static_cast<size_t>(rankSize * rankSize));

        std::vector<int32_t> dst(static_cast<size_t>(cap));
        std::vector<int32_t> cuSeqlens(static_cast<size_t>(groupCount));
        const int32_t expertsCanary = 0x5a5a5a5a;
        std::vector<int32_t> expertsStorage(static_cast<size_t>(b + 2), expertsCanary);
        std::vector<int32_t> remoteExperts(static_cast<size_t>(rankSize * b), -7);
        int32_t *expertsToCopy = aliasLegacyOutputToCurrentRemoteRow
            ? remoteExperts.data() + static_cast<size_t>(rank * b)
            : expertsStorage.data() + 1;
        const int32_t targetWords = (rankSize + 63) / 64;
        std::vector<uint64_t> expertTargets(static_cast<size_t>((expertNum / rankSize) * targetWords), ~0ULL);
        std::vector<int32_t> remoteStats(2);
        std::vector<int32_t> status(TileXREp::Plan::kPlanStatusWords);

        PlanAlgorithmInput algorithmInput {};
        algorithmInput.rank = rank;
        algorithmInput.rankSize = rankSize;
        algorithmInput.s = input.s;
        algorithmInput.topK = input.topK;
        algorithmInput.expertNum = input.expertNum;
        algorithmInput.config = input.config;
        algorithmInput.topkExperts = input.topkExperts.data() + static_cast<size_t>(rank * cap);
        algorithmInput.tokensPerExpert = input.tokensPerExpert.data();
        algorithmInput.globalRankIds = input.globalRankIds.data();

        PlanAlgorithmOutput algorithmOutput {};
        algorithmOutput.dst = dst.data();
        algorithmOutput.cuSeqlens = cuSeqlens.data();
        algorithmOutput.expertsToCopy = expertsToCopy;
        algorithmOutput.remoteExperts = remoteExperts.data();
        algorithmOutput.expertTargets = expertTargets.data();
        algorithmOutput.remoteStats = remoteStats.data();
        algorithmOutput.status = status.data();

        PlanAlgorithmWorkspace workspace {};
        workspace.expertCount = expertCount.data();
        workspace.rankLoad = rankLoad.data();
        workspace.remainingTpe = remainingTpe.data();
        workspace.alloc = alloc.data();
        workspace.remoteExpertSet = remoteExpertSet.data();
        workspace.srcExpertCursor = srcExpertCursor.data();
        workspace.dstExpertCursor = dstExpertCursor.data();
        workspace.expertPhysicalBase = expertPhysicalBase.data();
        workspace.localExpertOrdinal = localExpertOrdinal.data();
        workspace.tokenSegments = tokenSegments.data();
        workspace.tokenSegmentCapacity = cap;
        workspace.routedPairTokens = routedPairTokens.data();
        workspace.scratch = scratch.data();
        workspace.scratchCount = static_cast<int32_t>(scratch.size());
        workspace.affinityOrder = affinityOrder.data();

        const TileXRMoonEPPlanStatus actualStatus =
            TileXREp::Plan::RunPlanAlgorithm(algorithmInput, algorithmOutput, workspace);
        Check(actualStatus == expectedStatus, name + ": algorithm status mismatch at rank " +
            std::to_string(rank));

        if (reference.dst.size() == static_cast<size_t>(rankSize * cap)) {
            CheckVector(dst, std::vector<int32_t>(reference.dst.begin() + rank * cap,
                reference.dst.begin() + (rank + 1) * cap), name + ": dst rank " + std::to_string(rank));
        }
        if (reference.cuSeqlens.size() == static_cast<size_t>(rankSize * groupCount)) {
            CheckVector(cuSeqlens, std::vector<int32_t>(reference.cuSeqlens.begin() + rank * groupCount,
                reference.cuSeqlens.begin() + (rank + 1) * groupCount),
                name + ": cuSeqlens rank " + std::to_string(rank));
        }
        if (reference.expertsToCopy.size() == static_cast<size_t>(rankSize * b)) {
            CheckVector(remoteExperts, reference.expertsToCopy,
                name + ": remoteExperts rank " + std::to_string(rank));
        }
        if (reference.expertTargets.size() == static_cast<size_t>(rankSize * (expertNum / rankSize) * targetWords)) {
            const auto begin = reference.expertTargets.begin() + rank * (expertNum / rankSize) * targetWords;
            const std::vector<uint64_t> expectedTargets(begin,
                begin + (expertNum / rankSize) * targetWords);
            if (expertTargets != expectedTargets) {
                Check(false, name + ": expertTargets rank " + std::to_string(rank));
            }
        }
        if (!aliasLegacyOutputToCurrentRemoteRow) {
            Check(expertsStorage.front() == expertsCanary && expertsStorage.back() == expertsCanary,
                name + ": expertsToCopy bounds corrupted at rank " + std::to_string(rank));
        }
        if (reference.expertsToCopy.size() == static_cast<size_t>(rankSize * b)) {
            CheckVector(std::vector<int32_t>(expertsToCopy, expertsToCopy + b),
                std::vector<int32_t>(reference.expertsToCopy.begin() + rank * b,
                    reference.expertsToCopy.begin() + (rank + 1) * b),
                name + ": expertsToCopy rank " + std::to_string(rank));
        }
        if (reference.remoteStats.size() == static_cast<size_t>(rankSize * 2)) {
            CheckVector(remoteStats, std::vector<int32_t>(reference.remoteStats.begin() + rank * 2,
                reference.remoteStats.begin() + (rank + 1) * 2),
                name + ": remoteStats rank " + std::to_string(rank));
        }
        CheckVector(status, std::vector<int32_t>(reference.statusByRank.begin() +
            rank * TileXREp::Plan::kPlanStatusWords, reference.statusByRank.begin() +
            (rank + 1) * TileXREp::Plan::kPlanStatusWords),
            name + ": status rank " + std::to_string(rank));
    }
}


uint32_t NextRandom(uint32_t *state)
{
    *state = *state * 1664525U + 1013904223U;
    return *state;
}

std::vector<int32_t> MakeTopology(int32_t rankSize, bool spreadAcrossServers)
{
    std::vector<int32_t> globalRankIds(static_cast<size_t>(rankSize));
    for (int32_t rank = 0; rank < rankSize; ++rank) {
        globalRankIds[static_cast<size_t>(rank)] = spreadAcrossServers
            ? (rank / 2) * TileXREp::Plan::kPlanCardsPerServer + rank % 2
            : rank;
    }
    return globalRankIds;
}

void TestNamedV2ScenarioParity()
{
    CompareScenario(MakeHomeLoadInput(
        {150, 140, 100, 100, 80, 70, 100, 100},
        {664, 665, 666, 667, 668, 669, 670, 671}, 100, 2),
        "v2-1-standard-symmetric-intra");

    CompareScenario(MakeHomeLoadInput(
        {180, 55, 50, 45, 40, 35, 30, 25,
         180, 160, 150, 140, 130, 130, 130, 120},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        100, 4, 1, 1600), "v2-2-middle-line-embedded");

    CompareScenario(MakeHomeLoadInput({100, 100}, {0, 1}, 100, 1),
        "v2-3-balanced-guard");
    CompareScenario(MakeHomeLoadInput({150, 140, 130, 60, 20},
        {0, 16, 32, 8, 9}, 100, 2), "v2-4-nearest-and-v2-5-occupied");
    CompareScenario(MakeHomeLoadInput({190, 130, 95, 65, 20},
        {32, 16, 0, 9, 8}, 100, 3), "v2-6-dynamic-top1-skip");

    CompareScenario(MakeHomeLoadInput({120, 80}, {0, 8}, 100, 1),
        "v2-7-single-token-source");
    CompareScenario(MakeHomeLoadInput({120, 80}, {0, 8}, 100, 1),
        "v2-metadata-current-rank-row-alias", true);
    CompareScenario(MakeHomeLoadInput({140, 140, 20}, {0, 16, 8}, 100, 2),
        "v2-8-multiple-token-sources");
    CompareScenario(MakeHomeLoadInput({200, 0}, {0, 8}, 100, 1, 1, 100, 20),
        "v2-9-partial-token-supply");

    CompareScenario(MakeHomeLoadInput({140, 140, 20}, {0, 16, 8}, 100, 1),
        "v2-prefetch-slot-hard-limit");
    CompareScenario(MakeInput(2, 2, 2, 2, {0, 8},
        {0, 0, 0, 0, 0, 0, 0, 0}, 1, 1, 4), "v2-dedup-negative-dst");
    CompareScenario(MakeHomeLoadInput({3, 3}, {0, 8}, 3, 1, 4, 8),
        "v2-padding-layout");
    CompareScenario(MakeInput(2, 1, 1, 2, {0, 8}, {0, 1}, 3, 1, 1),
        "v2-explicit-prefetch-slots-above-expert-count");
    CompareScenario(MakeInput(1, 1, 1, 1, {0}, {0}, 1,
        static_cast<int64_t>(INT32_MAX) + 1, 1),
        "v2-positive-padding-above-int32-layout-failure");
}

void TestDeterministicDifferentialSweep()
{
    uint32_t state = 0x54c9d17bU;
    const int32_t rankSizes[] = {1, 2, 4, 8};
    for (int32_t rankSize : rankSizes) {
        for (int32_t caseIndex = 0; caseIndex < 32; ++caseIndex) {
            const int64_t topKOptions[] = {1, 2, 4};
            const int64_t topK = topKOptions[NextRandom(&state) % 3U];
            const int64_t s = 1 + static_cast<int64_t>(NextRandom(&state) % 8U);
            const int64_t cap = s * topK;
            const int64_t expertsPerRank = 1 + static_cast<int64_t>(NextRandom(&state) % 2U);
            const int64_t expertNum = rankSize * expertsPerRank;
            const int64_t b = 1 + static_cast<int64_t>(NextRandom(&state) %
                static_cast<uint32_t>(expertsPerRank));
            const int64_t paddingOptions[] = {1, 2, 4};
            const int64_t padding = paddingOptions[NextRandom(&state) % 3U];
            const int64_t groupCount = expertNum + b;
            const int64_t nvS = rankSize * cap + groupCount * (padding - 1);
            const int64_t routeLimit = (caseIndex % 5 == 0)
                ? 1 + static_cast<int64_t>(NextRandom(&state) % static_cast<uint32_t>(cap))
                : 0;

            std::vector<int32_t> routes(static_cast<size_t>(rankSize * cap));
            for (size_t route = 0; route < routes.size(); ++route) {
                routes[route] = static_cast<int32_t>(NextRandom(&state) %
                    static_cast<uint32_t>(expertNum));
            }
            const std::vector<int32_t> globalRankIds = MakeTopology(
                rankSize, (caseIndex & 1) != 0);
            ReferenceInput input = MakeInput(rankSize, s, topK, expertNum,
                globalRankIds, routes, b, padding, nvS, routeLimit);

            std::ostringstream name;
            name << "differential-r" << rankSize << "-case" << caseIndex
                 << "-s" << s << "-k" << topK << "-e" << expertNum
                 << "-b" << b << "-pad" << padding << "-limit" << routeLimit;
            CompareScenario(input, name.str());
        }
    }
}

} // namespace

void TestZeroPrefetchSlotsRejected()
{
    int32_t topkExperts[1] = {0};
    int32_t tokensPerExpert[1] = {1};
    int32_t globalRankIds[1] = {0};
    int32_t dst[1] = {0};
    int32_t cuSeqlens[1] = {0};
    int32_t expertsToCopy[1] = {-1};
    int32_t remoteStats[2] = {0, 0};
    int32_t status[TileXREp::Plan::kPlanStatusWords] = {0};
    int32_t expertCount[1] = {0};
    int32_t rankLoad[1] = {0};
    int32_t remainingTpe[1] = {0};
    int32_t alloc[1] = {0};
    int32_t remoteExpertSet[1] = {-1};
    int32_t srcExpertCursor[1] = {0};
    int32_t dstExpertCursor[1] = {0};
    int32_t expertPhysicalBase[1] = {0};
    int32_t localExpertOrdinal[1] = {0};
    TokenSegmentMove tokenSegments[1] = {};
    int32_t routedPairTokens[1] = {0};
    int32_t scratch[16] = {0};
    int32_t affinityOrder[1] = {0};

    PlanAlgorithmInput input {};
    input.rank = 0;
    input.rankSize = 1;
    input.s = 1;
    input.topK = 1;
    input.expertNum = 1;
    input.config = MakeConfig(1, 1, 0);
    input.topkExperts = topkExperts;
    input.tokensPerExpert = tokensPerExpert;
    input.globalRankIds = globalRankIds;

    PlanAlgorithmOutput output {};
    output.dst = dst;
    output.cuSeqlens = cuSeqlens;
    output.expertsToCopy = expertsToCopy;
    output.remoteStats = remoteStats;
    output.status = status;

    PlanAlgorithmWorkspace workspace {};
    workspace.expertCount = expertCount;
    workspace.rankLoad = rankLoad;
    workspace.remainingTpe = remainingTpe;
    workspace.alloc = alloc;
    workspace.remoteExpertSet = remoteExpertSet;
    workspace.srcExpertCursor = srcExpertCursor;
    workspace.dstExpertCursor = dstExpertCursor;
    workspace.expertPhysicalBase = expertPhysicalBase;
    workspace.localExpertOrdinal = localExpertOrdinal;
    workspace.tokenSegments = tokenSegments;
    workspace.tokenSegmentCapacity = 1;
    workspace.routedPairTokens = routedPairTokens;
    workspace.scratch = scratch;
    workspace.scratchCount = 16;
    workspace.affinityOrder = affinityOrder;

    Check(TileXREp::Plan::RunPlanAlgorithm(input, output, workspace) ==
        PLAN_ERROR_CONFIG_MISMATCH, "raw algorithm must reject B == 0");
}

void TestMoveRecordOverflowIsReported()
{
    int32_t topkExperts[1] = {0};
    int32_t tokensPerExpert[1] = {1};
    int32_t globalRankIds[1] = {0};
    int32_t dst[1] = {0};
    int32_t cuSeqlens[2] = {0};
    int32_t expertsToCopy[1] = {-1};
    int32_t remoteStats[2] = {0};
    int32_t status[TileXREp::Plan::kPlanStatusWords] = {0};
    int32_t expertCount[1] = {0};
    int32_t rankLoad[1] = {0};
    int32_t remainingTpe[1] = {0};
    int32_t alloc[1] = {0};
    int32_t remoteExpertSet[1] = {-1};
    int32_t srcExpertCursor[1] = {0};
    int32_t dstExpertCursor[1] = {0};
    int32_t expertPhysicalBase[1] = {0};
    int32_t localExpertOrdinal[1] = {0};
    TokenSegmentMove tokenSegments[1] = {};
    int32_t scratch[16] = {0};
    int32_t affinityOrder[1] = {0};

    PlanAlgorithmInput input {};
    input.rank = 0;
    input.rankSize = 1;
    input.s = 1;
    input.topK = 1;
    input.expertNum = 1;
    input.config = MakeConfig(1, 1, 1);
    input.topkExperts = topkExperts;
    input.tokensPerExpert = tokensPerExpert;
    input.globalRankIds = globalRankIds;

    PlanAlgorithmOutput output {};
    output.dst = dst;
    output.cuSeqlens = cuSeqlens;
    output.expertsToCopy = expertsToCopy;
    output.remoteStats = remoteStats;
    output.status = status;

    PlanAlgorithmWorkspace workspace {};
    workspace.expertCount = expertCount;
    workspace.rankLoad = rankLoad;
    workspace.remainingTpe = remainingTpe;
    workspace.alloc = alloc;
    workspace.remoteExpertSet = remoteExpertSet;
    workspace.srcExpertCursor = srcExpertCursor;
    workspace.dstExpertCursor = dstExpertCursor;
    workspace.expertPhysicalBase = expertPhysicalBase;
    workspace.localExpertOrdinal = localExpertOrdinal;
    workspace.tokenSegments = tokenSegments;
    workspace.tokenSegmentCapacity = 0;
    workspace.routedPairTokens = nullptr;
    workspace.scratch = scratch;
    workspace.scratchCount = 16;
    workspace.affinityOrder = affinityOrder;

    Check(TileXREp::Plan::RunPlanAlgorithm(input, output, workspace) ==
        PLAN_ERROR_MOVE_RECORD_OVERFLOW,
        "token-segment capacity exhaustion must report move-record overflow");
    Check(status[0] == PLAN_ERROR_MOVE_RECORD_OVERFLOW,
        "move-record overflow must be published through status[0]");
}

void TestCachedAffinityOrderIsReused()
{
    int32_t topkExperts[1] = {0};
    int32_t tokensPerExpert[4] = {1, 0, 0, 1};
    int32_t globalRankIds[2] = {669, 1551};
    int32_t dst[1] = {0};
    int32_t cuSeqlens[3] = {0};
    int32_t expertsToCopy[2] = {-1, -1};
    int32_t remoteStats[2] = {0};
    int32_t status[TileXREp::Plan::kPlanStatusWords] = {0};
    int32_t expertCount[2] = {0};
    int32_t rankLoad[2] = {0};
    int32_t remainingTpe[4] = {0};
    int32_t alloc[4] = {0};
    int32_t remoteExpertSet[2] = {-1, -1};
    int32_t srcExpertCursor[4] = {0};
    int32_t dstExpertCursor[4] = {0};
    int32_t expertPhysicalBase[4] = {0};
    int32_t localExpertOrdinal[1] = {0};
    TokenSegmentMove tokenSegments[1] = {};
    int32_t scratch[32] = {0};
    int32_t affinityOrder[4] = {-7, -7, -7, -7};

    PlanAlgorithmInput input {};
    input.rank = 0;
    input.rankSize = 2;
    input.s = 1;
    input.topK = 1;
    input.expertNum = 2;
    input.config = MakeConfig(1, 1, 1);
    input.topkExperts = topkExperts;
    input.tokensPerExpert = tokensPerExpert;
    input.globalRankIds = globalRankIds;

    PlanAlgorithmOutput output {};
    output.dst = dst;
    output.cuSeqlens = cuSeqlens;
    output.expertsToCopy = expertsToCopy;
    output.remoteStats = remoteStats;
    output.status = status;

    PlanAlgorithmWorkspace workspace {};
    workspace.expertCount = expertCount;
    workspace.rankLoad = rankLoad;
    workspace.remainingTpe = remainingTpe;
    workspace.alloc = alloc;
    workspace.remoteExpertSet = remoteExpertSet;
    workspace.srcExpertCursor = srcExpertCursor;
    workspace.dstExpertCursor = dstExpertCursor;
    workspace.expertPhysicalBase = expertPhysicalBase;
    workspace.localExpertOrdinal = localExpertOrdinal;
    workspace.tokenSegments = tokenSegments;
    workspace.tokenSegmentCapacity = 1;
    workspace.routedPairTokens = nullptr;
    workspace.scratch = scratch;
    workspace.scratchCount = 32;
    workspace.affinityOrder = affinityOrder;
    workspace.affinityOrderValid = true;

    Check(TileXREp::Plan::RunPlanAlgorithm(input, output, workspace) == PLAN_OK,
        "cached affinity scenario must remain valid");
    for (int32_t value : affinityOrder) {
        Check(value == -7, "valid cached affinity order must not be rebuilt");
    }

    for (int32_t &value : affinityOrder) value = -9;
    workspace.affinityOrderValid = false;
    Check(TileXREp::Plan::RunPlanAlgorithm(input, output, workspace) == PLAN_OK,
        "invalid affinity cache scenario must remain valid");
    Check(workspace.affinityOrderValid, "rebuilt affinity cache must be marked valid");
    Check(affinityOrder[0] == 0 && affinityOrder[1] == 1 &&
        affinityOrder[2] == 1 && affinityOrder[3] == 0,
        "invalid affinity cache must be rebuilt in deterministic XOR order");
}
int main()
{
    TestZeroPrefetchSlotsRejected();
    TestMoveRecordOverflowIsReported();
    TestCachedAffinityOrderIsReused();
    TestNamedV2ScenarioParity();
    TestDeterministicDifferentialSweep();

    if (g_failures != 0) {
        std::cerr << g_failures << " Plan algorithm tests failed" << std::endl;
        return 1;
    }
    std::cout << "Plan algorithm tests passed" << std::endl;
    return 0;
}
