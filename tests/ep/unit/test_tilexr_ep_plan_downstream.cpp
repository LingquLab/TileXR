#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ep_plan_downstream.h"
#include "ep_plan_reference.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void TestDstDecode()
{
    TileXREp::Plan::MoonEPRouteTarget target {};
    Check(TileXREp::Plan::DecodeMoonEPDst(2 * 64 + 7, 64, 4, &target) == PLAN_OK,
        "positive dst must decode");
    Check(target.rawDst == 135 && target.dstRank == 2 && target.recvSlot == 7,
        "positive dst fields mismatch");
    Check(target.sendHidden == 1 && target.writeRouteWeight == 1,
        "positive dst must send hidden and weight");

    const int32_t negative = ~(2 * 64 + 9);
    Check(TileXREp::Plan::DecodeMoonEPDst(negative, 64, 4, &target) == PLAN_OK,
        "negative dst must decode");
    Check(target.rawDst == 137 && target.dstRank == 2 && target.recvSlot == 9,
        "negative dst fields mismatch");
    Check(target.sendHidden == 0 && target.writeRouteWeight == 1,
        "negative dst must skip hidden but preserve weight");

    Check(TileXREp::Plan::DecodeMoonEPDst(0, 0, 4, &target) == PLAN_ERROR_CONFIG_MISMATCH,
        "NvS == 0 must fail");
    Check(TileXREp::Plan::DecodeMoonEPDst(4 * 64, 64, 4, &target) == PLAN_ERROR_CONFIG_MISMATCH,
        "decoded rank outside communicator must fail");
    Check(TileXREp::Plan::DecodeMoonEPDst(0, 64, 4, nullptr) == PLAN_ERROR_CONFIG_MISMATCH,
        "null target must fail");
}

void TestDuplicateMetadata()
{
    using TileXREp::Plan::MoonEPReceivedRoute;
    const MoonEPReceivedRoute records[] = {
        {1, 0, 2, 5, 0},
        {0, 0, 2, 9, 0},
        {1, 0, 0, 3, 1},
        {0, 0, 0, 2, 1},
        {1, 0, 1, 4, 0},
        {0, 1, 0, 7, 1},
    };
    std::vector<int32_t> groups(16 * 3, 99);
    std::vector<int32_t> loffs(16, 99);
    int32_t counts[2] = {-1, -1};

    Check(TileXREp::Plan::BuildMoonEPDuplicateMetadata(records, 6, 2, 2, 3, 16,
        groups.data(), loffs.data(), counts) == PLAN_OK,
        "valid duplicate metadata must build");
    Check(counts[0] == 2 && counts[1] == 3,
        "duplicate counts must report groups and duplicate offsets");
    Check(groups[0] == 2 && groups[1] == 0 && groups[2] == 1,
        "source rank 0 token 0 group mismatch");
    Check(groups[3] == 3 && groups[4] == 1 && groups[5] == 2,
        "source rank 1 token 0 group mismatch");
    Check(loffs[0] == 9 && loffs[1] == 4 && loffs[2] == 5,
        "duplicate offsets must follow fixed top-k order");
    Check(std::all_of(groups.begin() + 6, groups.end(), [](int32_t value) { return value == -1; }),
        "unused duplicate groups must be initialized");
    Check(std::all_of(loffs.begin() + 3, loffs.end(), [](int32_t value) { return value == -1; }),
        "unused duplicate offsets must be initialized");
}

void TestReferencePlanFeedsDuplicateMetadata()
{
    TileXREp::Plan::ReferenceInput input {};
    input.rankSize = 2;
    input.s = 2;
    input.topK = 2;
    input.expertNum = 2;
    input.globalRankIds = {0, 8};
    input.topkExperts.assign(8, 0);
    input.tokensPerExpert = {4, 0, 4, 0};
    input.config.prefetchSlots = 1;
    input.config.rankTokenCapacity = 4;
    input.config.nvS = 4;
    input.config.tokenPadding = 1;
    input.config.tokenRouteLimitPerPair = 4;
    input.config.cardsPerServer = 8;
    input.config.cardsPerCabinet = 64;
    input.config.crossCandidateCount = 3;

    TileXREp::Plan::ReferenceOutput output;
    Check(TileXREp::Plan::BuildReferencePlan(input, &output) == PLAN_OK,
        "reference dedup plan must succeed");
    std::vector<std::vector<uint64_t> > tokenRemap(static_cast<size_t>(input.rankSize),
        std::vector<uint64_t>(static_cast<size_t>(input.config.nvS),
            TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID));
    for (int32_t srcRank = 0; srcRank < input.rankSize; ++srcRank) {
        for (int32_t token = 0; token < input.s; ++token) {
            for (int32_t topKId = 0; topKId < input.topK; ++topKId) {
                const size_t routeIndex = static_cast<size_t>(
                    (srcRank * input.s + token) * input.topK + topKId);
                TileXREp::Plan::MoonEPRouteDescriptor route {};
                Check(TileXREp::Plan::BuildMoonEPRouteDescriptor(srcRank, token, topKId,
                    output.dst[routeIndex], input.rankSize, input.s, input.topK,
                    input.config.nvS, &route) == PLAN_OK,
                    "all-rank Planner route must build a Dispatch descriptor");
                uint64_t &slot = tokenRemap[static_cast<size_t>(route.dstRank)]
                    [static_cast<size_t>(route.recvSlot)];
                Check(slot == TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID,
                    "Planner dst must not collide while Dispatch reconstructs tokenRemap");
                slot = route.globalTokenId;
            }
        }
    }
    for (int32_t dstRank = 0; dstRank < input.rankSize; ++dstRank) {
        for (uint64_t globalTokenId : tokenRemap[static_cast<size_t>(dstRank)]) {
            Check(globalTokenId != TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID,
                "balanced reference plan must populate every target slot");
            int32_t srcRank = -1, token = -1, topKId = -1;
            Check(TileXREp::Plan::DecodeMoonEPGlobalTokenId(globalTokenId,
                input.rankSize, input.s, input.topK, &srcRank, &token, &topKId) == PLAN_OK,
                "Combine must decode every reconstructed tokenRemap entry");
        }
    }
    const int32_t expertsPerRank = static_cast<int32_t>(input.expertNum / input.rankSize);
    const int32_t targetWords = (input.rankSize + 63) / 64;
    for (int32_t ownerRank = 0; ownerRank < input.rankSize; ++ownerRank) {
        std::vector<uint64_t> rebuilt(static_cast<size_t>(expertsPerRank * targetWords), 0);
        Check(TileXREp::Plan::BuildMoonEPExpertTargets(output.expertsToCopy.data(), input.rankSize,
            input.expertNum, input.config.prefetchSlots, ownerRank, rebuilt.data(), rebuilt.size()) == PLAN_OK,
            "expertTargets must be derivable from destination-oriented remoteExperts");
        const auto expectedBegin = output.expertTargets.begin() +
            ownerRank * expertsPerRank * targetWords;
        Check(std::equal(rebuilt.begin(), rebuilt.end(), expectedBegin),
            "remoteExperts and expertTargets must be bidirectionally consistent");
    }
    for (int32_t dstRank = 0; dstRank < input.rankSize; ++dstRank) {
        std::vector<TileXREp::Plan::MoonEPReceivedRoute> records;
        for (int32_t srcRank = 0; srcRank < input.rankSize; ++srcRank) {
            for (int32_t token = 0; token < input.s; ++token) {
                for (int32_t k = 0; k < input.topK; ++k) {
                    const size_t index = static_cast<size_t>(
                        (srcRank * input.s + token) * input.topK + k);
                    TileXREp::Plan::MoonEPRouteTarget target {};
                    Check(TileXREp::Plan::DecodeMoonEPDst(output.dst[index], input.config.nvS,
                        input.rankSize, &target) == PLAN_OK, "reference dst must decode");
                    if (target.dstRank == dstRank) {
                        records.push_back({srcRank, token, k, target.recvSlot, target.sendHidden});
                    }
                }
            }
        }
        std::vector<int32_t> groups(static_cast<size_t>(input.config.nvS * 3), -1);
        std::vector<int32_t> loffs(static_cast<size_t>(input.config.nvS), -1);
        int32_t counts[2] = {};
        Check(TileXREp::Plan::BuildMoonEPDuplicateMetadata(records.data(), records.size(),
            input.rankSize, input.s, input.topK, input.config.nvS,
            groups.data(), loffs.data(), counts) == PLAN_OK,
            "real Plan V2 dst must build downstream duplicate metadata");
        Check(counts[0] == 2 && counts[1] == 2,
            "each receiving rank must contain two real dedup groups");
    }
}
void TestDuplicateMetadataAllowsEmptyRank()
{
    std::vector<int32_t> groups(12, 7);
    std::vector<int32_t> loffs(4, 7);
    int32_t counts[2] = {7, 7};
    Check(TileXREp::Plan::BuildMoonEPDuplicateMetadata(nullptr, 0, 2, 2, 2, 4,
        groups.data(), loffs.data(), counts) == PLAN_OK,
        "rank with no received routes must produce empty duplicate metadata");
    Check(counts[0] == 0 && counts[1] == 0,
        "empty rank duplicate counts must be zero");
    Check(std::all_of(groups.begin(), groups.end(), [](int32_t value) { return value == -1; }),
        "empty rank duplicate groups must be initialized");
    Check(std::all_of(loffs.begin(), loffs.end(), [](int32_t value) { return value == -1; }),
        "empty rank duplicate offsets must be initialized");
}

void TestDuplicateMetadataRejectsInvalidGroups()
{
    using TileXREp::Plan::MoonEPReceivedRoute;
    int32_t groups[12] = {};
    int32_t loffs[4] = {};
    int32_t counts[2] = {};

    const MoonEPReceivedRoute noPrimary[] = {
        {0, 0, 0, 1, 0},
        {0, 0, 1, 2, 0},
    };
    Check(TileXREp::Plan::BuildMoonEPDuplicateMetadata(noPrimary, 2, 1, 1, 2, 4,
        groups, loffs, counts) == PLAN_ERROR_INTERNAL_INVARIANT,
        "duplicate group without a primary must fail");

    const MoonEPReceivedRoute twoPrimaries[] = {
        {0, 0, 0, 1, 1},
        {0, 0, 1, 2, 1},
    };
    Check(TileXREp::Plan::BuildMoonEPDuplicateMetadata(twoPrimaries, 2, 1, 1, 2, 4,
        groups, loffs, counts) == PLAN_ERROR_INTERNAL_INVARIANT,
        "same-rank routes without negative dedup encoding must fail");

    const MoonEPReceivedRoute badSlot[] = {{0, 0, 0, 4, 1}};
    Check(TileXREp::Plan::BuildMoonEPDuplicateMetadata(badSlot, 1, 1, 1, 1, 4,
        groups, loffs, counts) == PLAN_ERROR_CONFIG_MISMATCH,
        "recvSlot outside NvS must fail");
}


void TestGlobalTokenAndRouteDescriptor()
{
    uint64_t id = 0;
    Check(TileXREp::Plan::EncodeMoonEPGlobalTokenId(3, 7, 2, 8, 16, 4, &id) == PLAN_OK,
        "global token encode must succeed");
    Check(id == static_cast<uint64_t>((3 * 16 + 7) * 4 + 2), "global token encoding mismatch");
    int32_t rank = -1, token = -1, topk = -1;
    Check(TileXREp::Plan::DecodeMoonEPGlobalTokenId(id, 8, 16, 4, &rank, &token, &topk) == PLAN_OK,
        "global token decode must succeed");
    Check(rank == 3 && token == 7 && topk == 2, "global token round trip mismatch");
    Check(TileXREp::Plan::DecodeMoonEPGlobalTokenId(TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID,
        8, 16, 4, &rank, &token, &topk) == PLAN_ERROR_CONFIG_MISMATCH,
        "invalid remap sentinel must be rejected");
    Check(TileXREp::Plan::DecodeMoonEPGlobalTokenId(
        static_cast<uint64_t>(INT32_MAX) + 1, static_cast<int64_t>(INT32_MAX) + 2,
        1, 1, &rank, &token, &topk) == PLAN_ERROR_CONFIG_MISMATCH,
        "decoded source rank outside int32 must be rejected");
    Check(TileXREp::Plan::DecodeMoonEPGlobalTokenId(
        static_cast<uint64_t>(INT32_MAX) + 1, 1,
        static_cast<int64_t>(INT32_MAX) + 2, 1, &rank, &token, &topk) == PLAN_ERROR_CONFIG_MISMATCH,
        "decoded token id outside int32 must be rejected");
    Check(TileXREp::Plan::DecodeMoonEPGlobalTokenId(
        static_cast<uint64_t>(INT32_MAX) + 1, 1, 1,
        static_cast<int64_t>(INT32_MAX) + 2, &rank, &token, &topk) == PLAN_ERROR_CONFIG_MISMATCH,
        "decoded top-k id outside int32 must be rejected");
    TileXREp::Plan::MoonEPRouteDescriptor route {};
    const int32_t raw = 5 * 100 + 99;
    Check(TileXREp::Plan::BuildMoonEPRouteDescriptor(3, 7, 2, ~raw, 8, 16, 4, 100,
        &route) == PLAN_OK, "duplicate route descriptor must build");
    Check(route.globalTokenId == id && route.dstRank == 5 && route.recvSlot == 99 &&
        route.sendHidden == 0 && route.writeRouteWeight == 1, "route descriptor mismatch");
}

void TestExpertTargetsContract()
{
    const int32_t remoteExperts[8] = {2, -1, 0, -1, 3, -1, 1, -1};
    uint64_t targets[2] = {};
    Check(TileXREp::Plan::BuildMoonEPExpertTargets(remoteExperts, 4, 8, 2, 0,
        targets, 2) == PLAN_OK, "expert targets must build");
    Check(targets[0] == (1ULL << 1) && targets[1] == (1ULL << 3),
        "owner-oriented expert bitmap mismatch");
}

} // namespace

int main()
{
    TestGlobalTokenAndRouteDescriptor();
    TestExpertTargetsContract();
    TestDstDecode();
    TestDuplicateMetadata();
    TestDuplicateMetadataAllowsEmptyRank();
    TestDuplicateMetadataRejectsInvalidGroups();
    TestReferencePlanFeedsDuplicateMetadata();
    if (g_failures != 0) {
        std::cerr << g_failures << " MoonEP downstream contract tests failed" << std::endl;
        return 1;
    }
    std::cout << "MoonEP downstream contract tests passed" << std::endl;
    return 0;
}
