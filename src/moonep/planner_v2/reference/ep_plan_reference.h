#ifndef TILEXR_EP_PLANNER_REFERENCE_EP_PLAN_REFERENCE_H
#define TILEXR_EP_PLANNER_REFERENCE_EP_PLAN_REFERENCE_H

#include <cstdint>
#include <string>
#include <vector>

#include "ep_plan_types.h"
#include "tilexr_ep_plan.h"

namespace TileXREp {
namespace Plan {

struct ReferenceInput {
    int32_t rankSize = 0;
    int64_t s = 0;
    int64_t topK = 0;
    int64_t expertNum = 0;
    TileXRMoonEPPlanConfig config {};
    std::vector<int32_t> globalRankIds;
    std::vector<int32_t> topkExperts;
    std::vector<int32_t> tokensPerExpert;
};

struct ReferenceRankPair {
    int32_t srcRank = -1;
    int32_t dstRank = -1;
    int32_t requestedUnits = 0;
};

struct ReferenceExpertMove {
    int32_t expertId = -1;
    int32_t srcHomeRank = -1;
    int32_t dstRank = -1;
    int32_t requestedUnits = 0;
    int32_t actualAssigned = 0;
    int32_t round = 0;
};

struct TokenSourceAssignment {
    int32_t actualAssigned = 0;
    int32_t unmetDemand = 0;
    bool supplyExhausted = false;
    bool routeLimited = false;
    std::vector<TokenSegmentMove> segments;
};

struct ReferenceOutput {
    TileXRMoonEPPlanStatus finalStatus = PLAN_OK;
    std::vector<int32_t> dst;
    std::vector<int32_t> cuSeqlens;
    std::vector<int32_t> expertsToCopy; // compatibility alias: remoteExperts [R,B]
    std::vector<uint64_t> expertTargets; // [R,E/R,ceil(R/64)], owner-major
    std::vector<int32_t> remoteStats;
    std::vector<int32_t> statusByRank;
    std::vector<int32_t> rankLoad;
    std::vector<int32_t> alloc;
    std::vector<TokenSegmentMove> segments;
    std::vector<ReferenceExpertMove> expertMoves;
};

bool BuildIntraServerPairs(const std::vector<int32_t> &rankLoad,
    const std::vector<int32_t> &globalRankIds, int32_t cap,
    std::vector<ReferenceRankPair> *pairs);

bool BuildStaticTopCandidates(int32_t receiver, const std::vector<int32_t> &senderGroup,
    const std::vector<int32_t> &affinityOrder, std::vector<int32_t> *candidates);

bool AssignTokenSources(int32_t expertId, int32_t dstRank, int32_t requestedUnits,
    int32_t routeLimit, const std::vector<int32_t> &affinityOrder,
    std::vector<int32_t> *remainingBySrc, std::vector<int32_t> *routedBySrc,
    std::vector<int32_t> *srcCursor, int32_t *dstCursor,
    TokenSourceAssignment *assignment);

TileXRMoonEPPlanStatus BuildReferencePlan(const ReferenceInput &input,
    ReferenceOutput *output);

bool CheckReferencePlanInvariants(const ReferenceInput &input,
    const ReferenceOutput &output, std::string *error);

} // namespace Plan
} // namespace TileXREp

#endif // TILEXR_EP_PLANNER_REFERENCE_EP_PLAN_REFERENCE_H
