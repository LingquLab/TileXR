#include "ep_plan_reference.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace TileXREp {
namespace Plan {
namespace {

struct RankedLoad {
    int32_t rank;
    int32_t load;
    int32_t globalRankId;
};

struct PlannerState {
    const ReferenceInput &input;
    ReferenceOutput &output;
    int32_t rankSize;
    int32_t expertNum;
    int32_t expertsPerRank;
    int32_t cap;
    int32_t prefetchSlots;
    int32_t routeLimit;
    std::vector<std::vector<int32_t> > affinityOrder;
    std::vector<int32_t> remainingTpe;
    std::vector<int32_t> srcExpertCursor;
    std::vector<int32_t> dstExpertCursor;
    std::vector<int32_t> routedPairTokens;
    std::vector<std::set<int32_t> > remoteExpertSets;
    bool sawSlotBlock;
    bool sawSupplyBlock;
    bool sawRouteOrPairBlock;

    PlannerState(const ReferenceInput &in, ReferenceOutput &out)
        : input(in), output(out), rankSize(in.rankSize),
          expertNum(static_cast<int32_t>(in.expertNum)),
          expertsPerRank(static_cast<int32_t>(in.expertNum / in.rankSize)),
          cap(static_cast<int32_t>(in.config.rankTokenCapacity)),
          prefetchSlots(static_cast<int32_t>(in.config.prefetchSlots)),
          routeLimit(static_cast<int32_t>(in.config.tokenRouteLimitPerPair)),
          affinityOrder(static_cast<size_t>(in.rankSize)),
          remainingTpe(in.tokensPerExpert),
          srcExpertCursor(static_cast<size_t>(in.rankSize * in.expertNum), 0),
          dstExpertCursor(static_cast<size_t>(in.rankSize * in.expertNum), 0),
          routedPairTokens(static_cast<size_t>(in.rankSize * in.rankSize), 0),
          remoteExpertSets(static_cast<size_t>(in.rankSize)),
          sawSlotBlock(false), sawSupplyBlock(false), sawRouteOrPairBlock(false)
    {
    }

    size_t ReIndex(int32_t rank, int32_t expert) const
    {
        return static_cast<size_t>(rank * expertNum + expert);
    }

    size_t PairIndex(int32_t src, int32_t dst) const
    {
        return static_cast<size_t>(src * rankSize + dst);
    }
};

bool SetError(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
    return false;
}

bool CheckedProduct(int64_t a, int64_t b, int64_t *result)
{
    if (result == nullptr || a < 0 || b < 0) {
        return false;
    }
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

int32_t HomeRank(int32_t expert, int32_t expertsPerRank)
{
    return expert / expertsPerRank;
}

int32_t AlignUp(int32_t value, int64_t alignment, bool *ok)
{
    if (ok == nullptr || value < 0 || alignment <= 0) {
        if (ok != nullptr) {
            *ok = false;
        }
        return 0;
    }
    if (value > 0 && alignment > INT32_MAX) {
        *ok = false;
        return 0;
    }
    const int64_t aligned = ((static_cast<int64_t>(value) + alignment - 1) / alignment) * alignment;
    if (aligned > INT32_MAX) {
        *ok = false;
        return 0;
    }
    *ok = true;
    return static_cast<int32_t>(aligned);
}

TileXRMoonEPPlanStatus ValidateInput(const ReferenceInput &input)
{
    if (input.rankSize <= 0 || input.rankSize > 512 || input.s <= 0 || input.topK <= 0 ||
        input.topK > 32 || input.expertNum <= 0 || input.expertNum % input.rankSize != 0) {
        return PLAN_ERROR_CONFIG_MISMATCH;
    }
    int64_t cap = 0;
    if (!CheckedProduct(input.s, input.topK, &cap) || cap <= 0 || cap > INT32_MAX) {
        return PLAN_ERROR_CONFIG_MISMATCH;
    }
    const TileXRMoonEPPlanConfig &config = input.config;
    if (config.rankTokenCapacity != cap || config.prefetchSlots <= 0 ||
        config.prefetchSlots > INT32_MAX ||
        config.nvS < cap || config.nvS > INT32_MAX || config.tokenPadding <= 0 ||
        config.tokenRouteLimitPerPair < 0 ||
        config.tokenRouteLimitPerPair > cap || config.cardsPerServer != kPlanCardsPerServer ||
        config.cardsPerCabinet != kPlanCardsPerCabinet ||
        config.crossCandidateCount != kPlanCrossCandidateCount) {
        return PLAN_ERROR_CONFIG_MISMATCH;
    }
    int64_t routeCount = 0;
    int64_t tpeCount = 0;
    int64_t addressRange = 0;
    if (!CheckedProduct(input.rankSize, cap, &routeCount) ||
        !CheckedProduct(input.rankSize, input.expertNum, &tpeCount) ||
        !CheckedProduct(input.rankSize, config.nvS, &addressRange) ||
        addressRange > INT32_MAX ||
        input.globalRankIds.size() != static_cast<size_t>(input.rankSize) ||
        input.topkExperts.size() != static_cast<size_t>(routeCount) ||
        input.tokensPerExpert.size() != static_cast<size_t>(tpeCount)) {
        return PLAN_ERROR_CONFIG_MISMATCH;
    }
    std::set<int32_t> globalIds;
    for (int32_t rank = 0; rank < input.rankSize; ++rank) {
        const int32_t globalId = input.globalRankIds[static_cast<size_t>(rank)];
        if (globalId < 0 || !globalIds.insert(globalId).second) {
            return PLAN_ERROR_CONFIG_MISMATCH;
        }
    }
    std::vector<int32_t> histogram(static_cast<size_t>(input.expertNum), 0);
    for (int32_t rank = 0; rank < input.rankSize; ++rank) {
        std::fill(histogram.begin(), histogram.end(), 0);
        for (int32_t route = 0; route < static_cast<int32_t>(cap); ++route) {
            const int32_t expert = input.topkExperts[static_cast<size_t>(rank * cap + route)];
            if (expert < 0 || expert >= input.expertNum) {
                return PLAN_ERROR_CONFIG_MISMATCH;
            }
            ++histogram[static_cast<size_t>(expert)];
        }
        int64_t tpeSum = 0;
        for (int32_t expert = 0; expert < input.expertNum; ++expert) {
            const int32_t value = input.tokensPerExpert[static_cast<size_t>(rank * input.expertNum + expert)];
            if (value < 0) {
                return PLAN_ERROR_CONFIG_MISMATCH;
            }
            tpeSum += value;
            if (value != histogram[static_cast<size_t>(expert)]) {
                return PLAN_ERROR_TPE_MISMATCH;
            }
        }
        if (tpeSum != cap) {
            return PLAN_ERROR_TPE_MISMATCH;
        }
    }
    return PLAN_OK;
}

void BuildAffinityOrders(PlannerState *state)
{
    for (int32_t dst = 0; dst < state->rankSize; ++dst) {
        std::vector<int32_t> &order = state->affinityOrder[static_cast<size_t>(dst)];
        order.resize(static_cast<size_t>(state->rankSize));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [state, dst](int32_t lhs, int32_t rhs) {
            const uint32_t dstId = static_cast<uint32_t>(state->input.globalRankIds[static_cast<size_t>(dst)]);
            const uint32_t lhsId = static_cast<uint32_t>(state->input.globalRankIds[static_cast<size_t>(lhs)]);
            const uint32_t rhsId = static_cast<uint32_t>(state->input.globalRankIds[static_cast<size_t>(rhs)]);
            const uint32_t lhsDistance = lhsId ^ dstId;
            const uint32_t rhsDistance = rhsId ^ dstId;
            if (lhsDistance != rhsDistance) {
                return lhsDistance < rhsDistance;
            }
            if (lhsId != rhsId) {
                return lhsId < rhsId;
            }
            return lhs < rhs;
        });
    }
}

int32_t ApplyExpertMove(PlannerState *state, int32_t srcHomeRank, int32_t dstRank,
    int32_t requestedUnits, int32_t round)
{
    if (requestedUnits <= 0 || srcHomeRank == dstRank) {
        return 0;
    }
    const int32_t begin = srcHomeRank * state->expertsPerRank;
    const int32_t end = begin + state->expertsPerRank;
    std::vector<int32_t> experts;
    experts.reserve(static_cast<size_t>(state->expertsPerRank));
    for (int32_t expert = begin; expert < end; ++expert) {
        if (state->output.alloc[state->ReIndex(srcHomeRank, expert)] > 0) {
            experts.push_back(expert);
        }
    }
    std::sort(experts.begin(), experts.end(), [state, srcHomeRank](int32_t lhs, int32_t rhs) {
        const int32_t lhsAlloc = state->output.alloc[state->ReIndex(srcHomeRank, lhs)];
        const int32_t rhsAlloc = state->output.alloc[state->ReIndex(srcHomeRank, rhs)];
        if (lhsAlloc != rhsAlloc) {
            return lhsAlloc > rhsAlloc;
        }
        return lhs < rhs;
    });

    int32_t demand = requestedUnits;
    bool foundCandidate = false;
    for (size_t i = 0; i < experts.size() && demand > 0; ++i) {
        const int32_t expert = experts[i];
        const int32_t availableAlloc = state->output.alloc[state->ReIndex(srcHomeRank, expert)];
        if (availableAlloc <= 0) {
            continue;
        }
        const bool alreadyRemote = state->remoteExpertSets[static_cast<size_t>(dstRank)].count(expert) != 0;
        if (!alreadyRemote && static_cast<int32_t>(state->remoteExpertSets[static_cast<size_t>(dstRank)].size()) >=
            state->prefetchSlots) {
            state->sawSlotBlock = true;
            continue;
        }
        foundCandidate = true;
        const int32_t expertRequest = std::min(availableAlloc, demand);
        std::vector<int32_t> remaining(static_cast<size_t>(state->rankSize));
        std::vector<int32_t> routed(static_cast<size_t>(state->rankSize));
        std::vector<int32_t> cursors(static_cast<size_t>(state->rankSize));
        for (int32_t src = 0; src < state->rankSize; ++src) {
            remaining[static_cast<size_t>(src)] = state->remainingTpe[state->ReIndex(src, expert)];
            routed[static_cast<size_t>(src)] = state->routedPairTokens[state->PairIndex(src, dstRank)];
            cursors[static_cast<size_t>(src)] = state->srcExpertCursor[state->ReIndex(src, expert)];
        }
        int32_t dstCursor = state->dstExpertCursor[state->ReIndex(dstRank, expert)];
        TokenSourceAssignment assignment;
        if (!AssignTokenSources(expert, dstRank, expertRequest, state->routeLimit,
            state->affinityOrder[static_cast<size_t>(dstRank)], &remaining, &routed,
            &cursors, &dstCursor, &assignment)) {
            state->sawRouteOrPairBlock = true;
            continue;
        }
        for (int32_t src = 0; src < state->rankSize; ++src) {
            state->remainingTpe[state->ReIndex(src, expert)] = remaining[static_cast<size_t>(src)];
            state->routedPairTokens[state->PairIndex(src, dstRank)] = routed[static_cast<size_t>(src)];
            state->srcExpertCursor[state->ReIndex(src, expert)] = cursors[static_cast<size_t>(src)];
        }
        state->dstExpertCursor[state->ReIndex(dstRank, expert)] = dstCursor;
        if (assignment.actualAssigned > 0) {
            if (!alreadyRemote) {
                state->remoteExpertSets[static_cast<size_t>(dstRank)].insert(expert);
            }
            state->output.alloc[state->ReIndex(srcHomeRank, expert)] -= assignment.actualAssigned;
            state->output.alloc[state->ReIndex(dstRank, expert)] += assignment.actualAssigned;
            state->output.rankLoad[static_cast<size_t>(srcHomeRank)] -= assignment.actualAssigned;
            state->output.rankLoad[static_cast<size_t>(dstRank)] += assignment.actualAssigned;
            demand -= assignment.actualAssigned;
            state->output.segments.insert(state->output.segments.end(),
                assignment.segments.begin(), assignment.segments.end());
            ReferenceExpertMove move;
            move.expertId = expert;
            move.srcHomeRank = srcHomeRank;
            move.dstRank = dstRank;
            move.requestedUnits = expertRequest;
            move.actualAssigned = assignment.actualAssigned;
            move.round = round;
            state->output.expertMoves.push_back(move);
        }
        if (assignment.unmetDemand > 0) {
            if (assignment.routeLimited) {
                state->sawRouteOrPairBlock = true;
            } else if (assignment.supplyExhausted) {
                state->sawSupplyBlock = true;
            }
        }
    }
    if (demand > 0 && !foundCandidate) {
        state->sawRouteOrPairBlock = true;
    }
    return requestedUnits - demand;
}

void RunIntraServerStage(PlannerState *state, int32_t *roundCounter, int32_t *stageRounds)
{
    std::vector<int32_t> serverKeys;
    for (int32_t rank = 0; rank < state->rankSize; ++rank) {
        const int32_t key = state->input.globalRankIds[static_cast<size_t>(rank)] / kPlanCardsPerServer;
        if (std::find(serverKeys.begin(), serverKeys.end(), key) == serverKeys.end()) {
            serverKeys.push_back(key);
        }
    }
    std::sort(serverKeys.begin(), serverKeys.end());
    for (size_t keyIndex = 0; keyIndex < serverKeys.size(); ++keyIndex) {
        std::vector<int32_t> members;
        for (int32_t rank = 0; rank < state->rankSize; ++rank) {
            if (state->input.globalRankIds[static_cast<size_t>(rank)] / kPlanCardsPerServer == serverKeys[keyIndex]) {
                members.push_back(rank);
            }
        }
        while (true) {
            std::vector<int32_t> localLoads;
            std::vector<int32_t> localIds;
            for (size_t i = 0; i < members.size(); ++i) {
                localLoads.push_back(state->output.rankLoad[static_cast<size_t>(members[i])]);
                localIds.push_back(state->input.globalRankIds[static_cast<size_t>(members[i])]);
            }
            std::vector<ReferenceRankPair> localPairs;
            if (!BuildIntraServerPairs(localLoads, localIds, state->cap, &localPairs) || localPairs.empty()) {
                break;
            }
            int32_t actualRound = 0;
            for (size_t i = 0; i < localPairs.size(); ++i) {
                const int32_t src = members[static_cast<size_t>(localPairs[i].srcRank)];
                const int32_t dst = members[static_cast<size_t>(localPairs[i].dstRank)];
                actualRound += ApplyExpertMove(state, src, dst, localPairs[i].requestedUnits, *roundCounter);
            }
            ++(*roundCounter);
            if (actualRound == 0) {
                state->sawRouteOrPairBlock = true;
                break;
            }
            ++(*stageRounds);
        }
    }
}

void RunInterServerStage(PlannerState *state, int32_t *roundCounter, int32_t *stageRounds)
{
    while (true) {
        bool allBalanced = true;
        std::vector<RankedLoad> active;
        int32_t underloadedCount = 0;
        for (int32_t rank = 0; rank < state->rankSize; ++rank) {
            const int32_t load = state->output.rankLoad[static_cast<size_t>(rank)];
            if (load != state->cap) {
                allBalanced = false;
                RankedLoad item {rank, load, state->input.globalRankIds[static_cast<size_t>(rank)]};
                active.push_back(item);
                if (load < state->cap) {
                    ++underloadedCount;
                }
            }
        }
        if (allBalanced || active.size() < 2 || underloadedCount == 0 ||
            underloadedCount == static_cast<int32_t>(active.size())) {
            break;
        }
        std::sort(active.begin(), active.end(), [](const RankedLoad &lhs, const RankedLoad &rhs) {
            if (lhs.load != rhs.load) {
                return lhs.load > rhs.load;
            }
            if (lhs.globalRankId != rhs.globalRankId) {
                return lhs.globalRankId < rhs.globalRankId;
            }
            return lhs.rank < rhs.rank;
        });
        const int32_t receiverCount = std::min(underloadedCount,
            static_cast<int32_t>(active.size() / 2));
        if (receiverCount <= 0) {
            break;
        }
        std::vector<int32_t> senderGroup;
        std::vector<RankedLoad> receivers;
        const int32_t senderCount = static_cast<int32_t>(active.size()) - receiverCount;
        for (int32_t i = 0; i < senderCount; ++i) {
            senderGroup.push_back(active[static_cast<size_t>(i)].rank);
        }
        for (int32_t i = senderCount; i < static_cast<int32_t>(active.size()); ++i) {
            receivers.push_back(active[static_cast<size_t>(i)]);
        }
        std::sort(receivers.begin(), receivers.end(), [](const RankedLoad &lhs, const RankedLoad &rhs) {
            if (lhs.load != rhs.load) {
                return lhs.load < rhs.load;
            }
            if (lhs.globalRankId != rhs.globalRankId) {
                return lhs.globalRankId < rhs.globalRankId;
            }
            return lhs.rank < rhs.rank;
        });

        std::set<int32_t> occupied;
        int32_t actualRound = 0;
        for (size_t receiverIndex = 0; receiverIndex < receivers.size(); ++receiverIndex) {
            const int32_t dst = receivers[receiverIndex].rank;
            std::vector<int32_t> candidates;
            BuildStaticTopCandidates(dst, senderGroup,
                state->affinityOrder[static_cast<size_t>(dst)], &candidates);
            for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
                const int32_t src = candidates[candidateIndex];
                if (occupied.count(src) != 0) {
                    continue;
                }
                const int32_t overflow = state->output.rankLoad[static_cast<size_t>(src)] - state->cap;
                const int32_t deficit = state->cap - state->output.rankLoad[static_cast<size_t>(dst)];
                const int32_t requested = std::min(overflow, deficit);
                if (requested <= 0) {
                    continue;
                }
                const int32_t actual = ApplyExpertMove(state, src, dst, requested, *roundCounter);
                if (actual > 0) {
                    actualRound += actual;
                    occupied.insert(src);
                    break;
                }
            }
        }
        ++(*roundCounter);
        if (actualRound == 0) {
            state->sawRouteOrPairBlock = true;
            break;
        }
        ++(*stageRounds);
    }
}

void AppendRemainderSegments(PlannerState *state)
{
    for (int32_t expert = 0; expert < state->expertNum; ++expert) {
        const int32_t dst = HomeRank(expert, state->expertsPerRank);
        for (int32_t src = 0; src < state->rankSize; ++src) {
            const size_t index = state->ReIndex(src, expert);
            const int32_t count = state->remainingTpe[index];
            if (count <= 0) {
                continue;
            }
            TokenSegmentMove segment;
            segment.expertId = expert;
            segment.srcRank = src;
            segment.dstRank = dst;
            segment.srcExpertBegin = state->srcExpertCursor[index];
            segment.dstExpertBegin = state->dstExpertCursor[state->ReIndex(dst, expert)];
            segment.tokenCount = count;
            state->output.segments.push_back(segment);
            state->remainingTpe[index] = 0;
            state->srcExpertCursor[index] += count;
            state->dstExpertCursor[state->ReIndex(dst, expert)] += count;
        }
    }
}

TileXRMoonEPPlanStatus BuildLayouts(PlannerState *state,
    std::vector<int32_t> *expertPhysicalBase)
{
    const int64_t groupCount = static_cast<int64_t>(state->expertNum) + state->prefetchSlots;
    state->output.cuSeqlens.assign(static_cast<size_t>(state->rankSize * groupCount), 0);
    state->output.expertsToCopy.assign(
        static_cast<size_t>(static_cast<int64_t>(state->rankSize) * state->prefetchSlots), -1);
    state->output.remoteStats.assign(static_cast<size_t>(state->rankSize * 2), 0);
    const int32_t targetWords = (state->rankSize + 63) / 64;
    state->output.expertTargets.assign(static_cast<size_t>(state->rankSize * state->expertsPerRank * targetWords), 0);
    expertPhysicalBase->assign(static_cast<size_t>(state->rankSize * state->expertNum), 0);

    std::vector<std::vector<int32_t> > remoteExperts(static_cast<size_t>(state->rankSize));
    for (int32_t dst = 0; dst < state->rankSize; ++dst) {
        std::vector<int32_t> &remote = remoteExperts[static_cast<size_t>(dst)];
        for (int32_t expert = 0; expert < state->expertNum; ++expert) {
            if (HomeRank(expert, state->expertsPerRank) != dst &&
                state->output.alloc[state->ReIndex(dst, expert)] > 0) {
                remote.push_back(expert);
            }
        }
        std::sort(remote.begin(), remote.end(), [state, dst](int32_t lhs, int32_t rhs) {
            const int32_t lhsAlloc = state->output.alloc[state->ReIndex(dst, lhs)];
            const int32_t rhsAlloc = state->output.alloc[state->ReIndex(dst, rhs)];
            if (lhsAlloc != rhsAlloc) {
                return lhsAlloc > rhsAlloc;
            }
            return lhs < rhs;
        });
        if (static_cast<int32_t>(remote.size()) > state->prefetchSlots) {
            return PLAN_PARTIAL_PREFETCH_SLOT_EXHAUSTED;
        }
        state->output.remoteStats[static_cast<size_t>(dst * 2)] = static_cast<int32_t>(remote.size());
        for (size_t slot = 0; slot < remote.size(); ++slot) {
            state->output.expertsToCopy[static_cast<size_t>(dst * state->prefetchSlots + slot)] = remote[slot];
            const int32_t home = HomeRank(remote[slot], state->expertsPerRank);
            ++state->output.remoteStats[static_cast<size_t>(home * 2 + 1)];
            const int32_t localExpert = remote[slot] - home * state->expertsPerRank;
            state->output.expertTargets[static_cast<size_t>((home * state->expertsPerRank + localExpert) * targetWords + dst / 64)] |=
                1ULL << (dst % 64);
        }
    }

    for (int32_t dst = 0; dst < state->rankSize; ++dst) {
        std::vector<int32_t> slotOfExpert(static_cast<size_t>(state->expertNum), -1);
        const std::vector<int32_t> &remote = remoteExperts[static_cast<size_t>(dst)];
        for (size_t slot = 0; slot < remote.size(); ++slot) {
            slotOfExpert[static_cast<size_t>(remote[slot])] = static_cast<int32_t>(slot);
        }
        int32_t previousEnd = 0;
        for (int64_t group = 0; group < groupCount; ++group) {
            int32_t count = 0;
            if (group < state->expertNum) {
                if (slotOfExpert[static_cast<size_t>(group)] < 0) {
                    count = state->output.alloc[state->ReIndex(dst, group)];
                    (*expertPhysicalBase)[state->ReIndex(dst, group)] = previousEnd;
                }
            } else {
                const int64_t slot = group - state->expertNum;
                if (slot < static_cast<int64_t>(remote.size())) {
                    const int32_t expert = remote[static_cast<size_t>(slot)];
                    count = state->output.alloc[state->ReIndex(dst, expert)];
                    (*expertPhysicalBase)[state->ReIndex(dst, expert)] = previousEnd;
                }
            }
            bool alignOk = true;
            const int32_t padded = count == 0 ? 0 : AlignUp(count,
                state->input.config.tokenPadding, &alignOk);
            if (!alignOk || padded > state->input.config.nvS - previousEnd) {
                return PLAN_ERROR_LAYOUT_EXCEEDS_NVS;
            }
            const size_t groupIndex = static_cast<size_t>(dst * groupCount + group);
            previousEnd += padded;
            state->output.cuSeqlens[groupIndex] = previousEnd;
        }
    }
    return PLAN_OK;
}

TileXRMoonEPPlanStatus BuildDst(PlannerState *state,
    const std::vector<int32_t> &expertPhysicalBase)
{
    const int32_t routeCount = state->cap;
    state->output.dst.assign(static_cast<size_t>(state->rankSize * routeCount), 0);
    std::vector<std::vector<const TokenSegmentMove *> > segmentsBySourceExpert(
        static_cast<size_t>(state->rankSize * state->expertNum));
    for (size_t i = 0; i < state->output.segments.size(); ++i) {
        const TokenSegmentMove &segment = state->output.segments[i];
        segmentsBySourceExpert[state->ReIndex(segment.srcRank, segment.expertId)].push_back(&segment);
    }
    for (size_t i = 0; i < segmentsBySourceExpert.size(); ++i) {
        std::sort(segmentsBySourceExpert[i].begin(), segmentsBySourceExpert[i].end(),
            [](const TokenSegmentMove *lhs, const TokenSegmentMove *rhs) {
                return lhs->srcExpertBegin < rhs->srcExpertBegin;
            });
    }

    for (int32_t src = 0; src < state->rankSize; ++src) {
        std::vector<int32_t> ordinal(static_cast<size_t>(state->expertNum), 0);
        for (int64_t token = 0; token < state->input.s; ++token) {
            std::set<int32_t> seenRanks;
            for (int64_t k = 0; k < state->input.topK; ++k) {
                const int32_t localRoute = static_cast<int32_t>(token * state->input.topK + k);
                const size_t routeIndex = static_cast<size_t>(src * routeCount + localRoute);
                const int32_t expert = state->input.topkExperts[routeIndex];
                const int32_t currentOrdinal = ordinal[static_cast<size_t>(expert)]++;
                const std::vector<const TokenSegmentMove *> &candidates =
                    segmentsBySourceExpert[state->ReIndex(src, expert)];
                const TokenSegmentMove *matched = nullptr;
                for (size_t segmentIndex = 0; segmentIndex < candidates.size(); ++segmentIndex) {
                    const TokenSegmentMove *segment = candidates[segmentIndex];
                    if (currentOrdinal >= segment->srcExpertBegin &&
                        currentOrdinal < segment->srcExpertBegin + segment->tokenCount) {
                        matched = segment;
                        break;
                    }
                }
                if (matched == nullptr) {
                    return PLAN_ERROR_INTERNAL_INVARIANT;
                }
                const int32_t segmentOffset = currentOrdinal - matched->srcExpertBegin;
                const int64_t localOffset = static_cast<int64_t>(
                    expertPhysicalBase[state->ReIndex(matched->dstRank, expert)]) +
                    matched->dstExpertBegin + segmentOffset;
                const int64_t raw = static_cast<int64_t>(matched->dstRank) * state->input.config.nvS + localOffset;
                if (localOffset < 0 || localOffset >= state->input.config.nvS || raw < 0 || raw > INT32_MAX) {
                    return PLAN_ERROR_INTERNAL_INVARIANT;
                }
                const int32_t rawDst = static_cast<int32_t>(raw);
                state->output.dst[routeIndex] = seenRanks.insert(matched->dstRank).second ? rawDst : ~rawDst;
            }
        }
    }
    return PLAN_OK;
}

void FillStatus(PlannerState *state, TileXRMoonEPPlanStatus status,
    int32_t intraRounds, int32_t interRounds)
{
    state->output.finalStatus = status;
    int32_t deficit = 0;
    int32_t maxRemoteExperts = 0;
    for (int32_t rank = 0; rank < state->rankSize; ++rank) {
        const int32_t delta = state->output.rankLoad[static_cast<size_t>(rank)] - state->cap;
        if (delta < 0) {
            deficit -= delta;
        }
        if (!state->output.remoteStats.empty()) {
            maxRemoteExperts = std::max(maxRemoteExperts,
                state->output.remoteStats[static_cast<size_t>(rank * 2)]);
        }
    }
    const int64_t groupCount = static_cast<int64_t>(state->expertNum) + state->prefetchSlots;
    state->output.statusByRank.assign(static_cast<size_t>(state->rankSize * kPlanStatusWords), 0);
    for (int32_t rank = 0; rank < state->rankSize; ++rank) {
        int32_t localSegments = 0;
        for (size_t index = 0; index < state->output.segments.size(); ++index) {
            if (state->output.segments[index].srcRank == rank) {
                ++localSegments;
            }
        }
        const size_t base = static_cast<size_t>(rank * kPlanStatusWords);
        state->output.statusByRank[base] = static_cast<int32_t>(status);
        state->output.statusByRank[base + 1] = intraRounds;
        state->output.statusByRank[base + 2] = interRounds;
        state->output.statusByRank[base + 3] = deficit;
        state->output.statusByRank[base + 4] = maxRemoteExperts;
        state->output.statusByRank[base + 5] = state->output.cuSeqlens.empty() ? 0 :
            state->output.cuSeqlens[static_cast<size_t>((rank + 1) * groupCount - 1)];
        state->output.statusByRank[base + 6] = static_cast<int32_t>(state->output.expertMoves.size());
        state->output.statusByRank[base + 7] = localSegments;
    }
}

} // namespace

bool BuildIntraServerPairs(const std::vector<int32_t> &rankLoad,
    const std::vector<int32_t> &globalRankIds, int32_t cap,
    std::vector<ReferenceRankPair> *pairs)
{
    if (pairs == nullptr || cap <= 0 || rankLoad.size() != globalRankIds.size()) {
        return false;
    }
    pairs->clear();
    std::vector<RankedLoad> active;
    for (size_t rank = 0; rank < rankLoad.size(); ++rank) {
        if (rankLoad[rank] < 0 || globalRankIds[rank] < 0) {
            return false;
        }
        if (rankLoad[rank] != cap) {
            RankedLoad item {static_cast<int32_t>(rank), rankLoad[rank], globalRankIds[rank]};
            active.push_back(item);
        }
    }
    std::sort(active.begin(), active.end(), [](const RankedLoad &lhs, const RankedLoad &rhs) {
        if (lhs.load != rhs.load) {
            return lhs.load > rhs.load;
        }
        if (lhs.globalRankId != rhs.globalRankId) {
            return lhs.globalRankId < rhs.globalRankId;
        }
        return lhs.rank < rhs.rank;
    });
    const int32_t count = static_cast<int32_t>(active.size());
    if (count < 2 || active[0].load + active[1].load < cap ||
        active[static_cast<size_t>(count - 1)].load > cap) {
        return true;
    }
    int32_t middle = count / 2;
    while (middle > 0 &&
        active[static_cast<size_t>(middle - 1)].load + active[static_cast<size_t>(middle)].load <= cap) {
        --middle;
    }
    if (middle == 0) {
        return true;
    }
    std::vector<RankedLoad> senders;
    std::vector<RankedLoad> receivers;
    for (int32_t i = middle - 1; i >= 0; --i) {
        if (active[static_cast<size_t>(i)].load > cap) {
            senders.push_back(active[static_cast<size_t>(i)]);
        }
    }
    for (int32_t i = middle; i < count; ++i) {
        if (active[static_cast<size_t>(i)].load < cap) {
            receivers.push_back(active[static_cast<size_t>(i)]);
        }
    }
    const size_t pairCount = std::min(senders.size(), receivers.size());
    for (size_t i = 0; i < pairCount; ++i) {
        const int32_t requested = std::min(senders[i].load - cap, cap - receivers[i].load);
        if (requested > 0) {
            ReferenceRankPair pair;
            pair.srcRank = senders[i].rank;
            pair.dstRank = receivers[i].rank;
            pair.requestedUnits = requested;
            pairs->push_back(pair);
        }
    }
    return true;
}

bool BuildStaticTopCandidates(int32_t receiver, const std::vector<int32_t> &senderGroup,
    const std::vector<int32_t> &affinityOrder, std::vector<int32_t> *candidates)
{
    if (candidates == nullptr || receiver < 0) {
        return false;
    }
    candidates->clear();
    std::set<int32_t> senders(senderGroup.begin(), senderGroup.end());
    for (size_t i = 0; i < affinityOrder.size() &&
        candidates->size() < static_cast<size_t>(kPlanCrossCandidateCount); ++i) {
        const int32_t rank = affinityOrder[i];
        if (senders.count(rank) != 0) {
            candidates->push_back(rank);
        }
    }
    return true;
}

bool AssignTokenSources(int32_t expertId, int32_t dstRank, int32_t requestedUnits,
    int32_t routeLimit, const std::vector<int32_t> &affinityOrder,
    std::vector<int32_t> *remainingBySrc, std::vector<int32_t> *routedBySrc,
    std::vector<int32_t> *srcCursor, int32_t *dstCursor,
    TokenSourceAssignment *assignment)
{
    if (expertId < 0 || dstRank < 0 || requestedUnits < 0 || routeLimit < 0 ||
        remainingBySrc == nullptr || routedBySrc == nullptr || srcCursor == nullptr ||
        dstCursor == nullptr || assignment == nullptr ||
        remainingBySrc->size() != routedBySrc->size() ||
        remainingBySrc->size() != srcCursor->size()) {
        return false;
    }
    *assignment = TokenSourceAssignment {};
    int32_t demand = requestedUnits;
    for (size_t i = 0; i < affinityOrder.size() && demand > 0; ++i) {
        const int32_t src = affinityOrder[i];
        if (src < 0 || static_cast<size_t>(src) >= remainingBySrc->size()) {
            return false;
        }
        const int32_t available = (*remainingBySrc)[static_cast<size_t>(src)];
        if (available < 0 || (*routedBySrc)[static_cast<size_t>(src)] < 0 ||
            (*srcCursor)[static_cast<size_t>(src)] < 0 || *dstCursor < 0) {
            return false;
        }
        int64_t pairRemain = demand;
        if (routeLimit != 0) {
            pairRemain = static_cast<int64_t>(routeLimit) - (*routedBySrc)[static_cast<size_t>(src)];
        }
        const int32_t take = static_cast<int32_t>(std::min<int64_t>(
            std::min<int64_t>(available, demand), std::max<int64_t>(pairRemain, 0)));
        if (take <= 0) {
            continue;
        }
        TokenSegmentMove segment;
        segment.expertId = expertId;
        segment.srcRank = src;
        segment.dstRank = dstRank;
        segment.srcExpertBegin = (*srcCursor)[static_cast<size_t>(src)];
        segment.dstExpertBegin = *dstCursor;
        segment.tokenCount = take;
        assignment->segments.push_back(segment);
        (*remainingBySrc)[static_cast<size_t>(src)] -= take;
        (*srcCursor)[static_cast<size_t>(src)] += take;
        *dstCursor += take;
        if (routeLimit != 0) {
            (*routedBySrc)[static_cast<size_t>(src)] += take;
        }
        assignment->actualAssigned += take;
        demand -= take;
    }
    assignment->unmetDemand = demand;
    if (demand > 0) {
        int64_t supply = 0;
        bool routeBlocked = false;
        for (size_t src = 0; src < remainingBySrc->size(); ++src) {
            supply += (*remainingBySrc)[src];
            if (routeLimit != 0 && (*remainingBySrc)[src] > 0 &&
                (*routedBySrc)[src] >= routeLimit) {
                routeBlocked = true;
            }
        }
        assignment->routeLimited = routeBlocked;
        assignment->supplyExhausted = supply == 0;
    }
    return true;
}

TileXRMoonEPPlanStatus BuildReferencePlan(const ReferenceInput &input,
    ReferenceOutput *output)
{
    if (output == nullptr) {
        return PLAN_ERROR_CONFIG_MISMATCH;
    }
    *output = ReferenceOutput {};
    const TileXRMoonEPPlanStatus validation = ValidateInput(input);
    if (validation != PLAN_OK) {
        output->finalStatus = validation;
        return validation;
    }

    PlannerState state(input, *output);
    output->rankLoad.assign(static_cast<size_t>(state.rankSize), 0);
    output->alloc.assign(static_cast<size_t>(state.rankSize * state.expertNum), 0);
    for (int32_t expert = 0; expert < state.expertNum; ++expert) {
        int32_t count = 0;
        for (int32_t src = 0; src < state.rankSize; ++src) {
            count += input.tokensPerExpert[state.ReIndex(src, expert)];
        }
        const int32_t home = HomeRank(expert, state.expertsPerRank);
        output->alloc[state.ReIndex(home, expert)] = count;
        output->rankLoad[static_cast<size_t>(home)] += count;
    }
    BuildAffinityOrders(&state);
    int32_t roundCounter = 0;
    int32_t intraRounds = 0;
    int32_t interRounds = 0;
    RunIntraServerStage(&state, &roundCounter, &intraRounds);
    RunInterServerStage(&state, &roundCounter, &interRounds);
    AppendRemainderSegments(&state);

    bool allBalanced = true;
    for (int32_t rank = 0; rank < state.rankSize; ++rank) {
        if (output->rankLoad[static_cast<size_t>(rank)] != state.cap) {
            allBalanced = false;
            break;
        }
    }
    TileXRMoonEPPlanStatus balanceStatus = PLAN_OK;
    if (!allBalanced) {
        if (state.sawSlotBlock) {
            balanceStatus = PLAN_PARTIAL_PREFETCH_SLOT_EXHAUSTED;
        } else if (state.sawSupplyBlock) {
            balanceStatus = PLAN_PARTIAL_TOKEN_SUPPLY;
        } else {
            balanceStatus = PLAN_PARTIAL_NO_FEASIBLE_PAIR;
        }
    }

    std::vector<int32_t> expertPhysicalBase;
    TileXRMoonEPPlanStatus layoutStatus = BuildLayouts(&state, &expertPhysicalBase);
    TileXRMoonEPPlanStatus status = balanceStatus;
    if (allBalanced) {
        status = layoutStatus;
        if (status == PLAN_OK) {
            status = BuildDst(&state, expertPhysicalBase);
        }
    } else if (layoutStatus == PLAN_OK) {
        // A partial diagnostic plan may contain an overloaded Rank and therefore exceed NvS.
        // Preserve the algorithmic partial reason; generate dst only when the diagnostic
        // layout itself is representable.
        (void)BuildDst(&state, expertPhysicalBase);
    }
    FillStatus(&state, status, intraRounds, interRounds);
    return status;
}

bool CheckReferencePlanInvariants(const ReferenceInput &input,
    const ReferenceOutput &output, std::string *error)
{
    if (ValidateInput(input) != PLAN_OK) {
        return SetError(error, "input validation failed");
    }
    const int32_t rankSize = input.rankSize;
    const int32_t expertNum = static_cast<int32_t>(input.expertNum);
    const int32_t cap = static_cast<int32_t>(input.config.rankTokenCapacity);
    const int64_t groupCount = static_cast<int64_t>(expertNum) + input.config.prefetchSlots;
    if (output.expertsToCopy.size() !=
        static_cast<size_t>(static_cast<int64_t>(rankSize) * input.config.prefetchSlots)) {
        return SetError(error, "expertsToCopy shape mismatch");
    }
    const int32_t targetWords = (rankSize + 63) / 64;
    if (output.expertTargets.size() !=
        static_cast<size_t>(static_cast<int64_t>(expertNum) * targetWords)) {
        return SetError(error, "expertTargets shape mismatch");
    }
    if (output.statusByRank.size() !=
        static_cast<size_t>(static_cast<int64_t>(rankSize) * kPlanStatusWords)) {
        return SetError(error, "statusByRank shape mismatch");
    }
    const int32_t agreedStatus = output.statusByRank[0];
    for (int32_t rank = 1; rank < rankSize; ++rank) {
        if (output.statusByRank[static_cast<size_t>(rank * kPlanStatusWords)] != agreedStatus) {
            return SetError(error, "status[0] consensus mismatch");
        }
    }
    if (agreedStatus != static_cast<int32_t>(output.finalStatus)) {
        return SetError(error, "status[0] does not match final status");
    }
    if (output.alloc.size() != static_cast<size_t>(rankSize * expertNum) ||
        output.rankLoad.size() != static_cast<size_t>(rankSize) ||
        output.dst.size() != static_cast<size_t>(rankSize * cap) ||
        output.cuSeqlens.size() != static_cast<size_t>(static_cast<int64_t>(rankSize) * groupCount) ||
        output.remoteStats.size() != static_cast<size_t>(rankSize * 2)) {
        return SetError(error, "output shape mismatch");
    }
    for (int32_t expert = 0; expert < expertNum; ++expert) {
        int64_t expected = 0;
        int64_t actual = 0;
        for (int32_t rank = 0; rank < rankSize; ++rank) {
            expected += input.tokensPerExpert[static_cast<size_t>(rank * expertNum + expert)];
            const int32_t value = output.alloc[static_cast<size_t>(rank * expertNum + expert)];
            if (value < 0) {
                return SetError(error, "negative allocation");
            }
            actual += value;
        }
        if (actual != expected) {
            return SetError(error, "expert conservation failed");
        }
    }
    for (int32_t rank = 0; rank < rankSize; ++rank) {
        int64_t load = 0;
        for (int32_t expert = 0; expert < expertNum; ++expert) {
            load += output.alloc[static_cast<size_t>(rank * expertNum + expert)];
        }
        if (load != output.rankLoad[static_cast<size_t>(rank)]) {
            return SetError(error, "rank load does not match allocation");
        }
        if (output.finalStatus == PLAN_OK && load != cap) {
            return SetError(error, "PLAN_OK rank capacity violated");
        }
    }

    std::vector<std::vector<int32_t> > expectedRemoteExperts(static_cast<size_t>(rankSize));
    std::vector<int32_t> expectedRemoteStats(static_cast<size_t>(rankSize * 2), 0);
    std::vector<uint64_t> expectedExpertTargets(
        static_cast<size_t>(static_cast<int64_t>(expertNum) * targetWords), 0);
    const int32_t expertsPerRank = expertNum / rankSize;
    for (int32_t rank = 0; rank < rankSize; ++rank) {
        std::vector<int32_t> &remote = expectedRemoteExperts[static_cast<size_t>(rank)];
        for (int32_t expert = 0; expert < expertNum; ++expert) {
            if (HomeRank(expert, expertsPerRank) != rank &&
                output.alloc[static_cast<size_t>(rank * expertNum + expert)] > 0) {
                remote.push_back(expert);
            }
        }
        std::sort(remote.begin(), remote.end(), [&output, rank, expertNum](int32_t lhs, int32_t rhs) {
            const int32_t lhsAlloc = output.alloc[static_cast<size_t>(rank * expertNum + lhs)];
            const int32_t rhsAlloc = output.alloc[static_cast<size_t>(rank * expertNum + rhs)];
            return lhsAlloc != rhsAlloc ? lhsAlloc > rhsAlloc : lhs < rhs;
        });
        expectedRemoteStats[static_cast<size_t>(rank * 2)] = static_cast<int32_t>(remote.size());
        for (size_t slot = 0; slot < remote.size(); ++slot) {
            const int32_t expert = remote[slot];
            ++expectedRemoteStats[static_cast<size_t>(HomeRank(expert, expertsPerRank) * 2 + 1)];
            expectedExpertTargets[static_cast<size_t>(
                static_cast<int64_t>(expert) * targetWords + rank / 64)] |= 1ULL << (rank % 64);
        }
        if (output.finalStatus == PLAN_OK) {
            for (int64_t slot = 0; slot < input.config.prefetchSlots; ++slot) {
                const int32_t expected = slot < static_cast<int64_t>(remote.size()) ?
                    remote[static_cast<size_t>(slot)] : -1;
                if (output.expertsToCopy[static_cast<size_t>(
                    static_cast<int64_t>(rank) * input.config.prefetchSlots + slot)] != expected) {
                    return SetError(error, "expertsToCopy content mismatch");
                }
            }
        }
    }
    if (output.finalStatus == PLAN_OK && output.remoteStats != expectedRemoteStats) {
        return SetError(error, "remoteStats mismatch");
    }
    if (output.finalStatus == PLAN_OK && output.expertTargets != expectedExpertTargets) {
        return SetError(error, "expertTargets content mismatch");
    }

    const size_t rankExpertCount = static_cast<size_t>(rankSize * expertNum);
    std::vector<int64_t> segmentCount(rankExpertCount, 0);
    std::vector<int64_t> destinationSegmentCount(rankExpertCount, 0);
    std::vector<std::vector<std::pair<int32_t, int32_t> > > sourceRanges(rankExpertCount);
    std::vector<std::vector<std::pair<int32_t, int32_t> > > destinationRanges(rankExpertCount);
    for (size_t i = 0; i < output.segments.size(); ++i) {
        const TokenSegmentMove &segment = output.segments[i];
        if (segment.expertId < 0 || segment.expertId >= expertNum || segment.srcRank < 0 ||
            segment.srcRank >= rankSize || segment.dstRank < 0 || segment.dstRank >= rankSize ||
            segment.srcExpertBegin < 0 || segment.dstExpertBegin < 0 || segment.tokenCount <= 0) {
            return SetError(error, "invalid token segment");
        }
        const size_t sourceIndex = static_cast<size_t>(segment.srcRank * expertNum + segment.expertId);
        const size_t destinationIndex = static_cast<size_t>(segment.dstRank * expertNum + segment.expertId);
        segmentCount[sourceIndex] += segment.tokenCount;
        destinationSegmentCount[destinationIndex] += segment.tokenCount;
        sourceRanges[sourceIndex].push_back(std::make_pair(segment.srcExpertBegin, segment.tokenCount));
        destinationRanges[destinationIndex].push_back(std::make_pair(
            segment.dstExpertBegin, segment.tokenCount));
    }
    for (size_t i = 0; i < segmentCount.size(); ++i) {
        if (segmentCount[i] != input.tokensPerExpert[i]) {
            return SetError(error, "source/expert segment conservation failed");
        }
        if (destinationSegmentCount[i] != output.alloc[i]) {
            return SetError(error, "destination/expert segment conservation failed");
        }
        std::sort(sourceRanges[i].begin(), sourceRanges[i].end());
        int64_t sourceCursor = 0;
        for (size_t range = 0; range < sourceRanges[i].size(); ++range) {
            if (sourceRanges[i][range].first != sourceCursor) {
                return SetError(error, "source/expert segment ranges are not contiguous");
            }
            sourceCursor += sourceRanges[i][range].second;
        }
        std::sort(destinationRanges[i].begin(), destinationRanges[i].end());
        int64_t destinationCursor = 0;
        for (size_t range = 0; range < destinationRanges[i].size(); ++range) {
            if (destinationRanges[i][range].first != destinationCursor) {
                return SetError(error, "destination/expert segment ranges are not contiguous");
            }
            destinationCursor += destinationRanges[i][range].second;
        }
    }
    for (int32_t rank = 0; rank < rankSize; ++rank) {
        if (output.remoteStats[static_cast<size_t>(rank * 2)] > input.config.prefetchSlots) {
            return SetError(error, "remote expert count exceeds B");
        }
        std::vector<int32_t> slotOfExpert(static_cast<size_t>(expertNum), -1);
        const std::vector<int32_t> &remote = expectedRemoteExperts[static_cast<size_t>(rank)];
        for (size_t slot = 0; slot < remote.size(); ++slot) {
            slotOfExpert[static_cast<size_t>(remote[slot])] = static_cast<int32_t>(slot);
        }
        int32_t previous = 0;
        for (int32_t group = 0; group < groupCount; ++group) {
            const int32_t end = output.cuSeqlens[static_cast<size_t>(rank * groupCount + group)];
            if (end < previous || end > input.config.nvS) {
                return SetError(error, "cuSeqlens is not monotonic or exceeds NvS");
            }
            int32_t realCount = 0;
            if (group < expertNum) {
                if (slotOfExpert[static_cast<size_t>(group)] < 0) {
                    realCount = output.alloc[static_cast<size_t>(rank * expertNum + group)];
                }
            } else {
                const int64_t slot = static_cast<int64_t>(group) - expertNum;
                if (slot < static_cast<int64_t>(remote.size())) {
                    realCount = output.alloc[static_cast<size_t>(
                        rank * expertNum + remote[static_cast<size_t>(slot)])];
                }
            }
            bool alignOk = true;
            const int32_t padded = realCount == 0 ? 0 :
                AlignUp(realCount, input.config.tokenPadding, &alignOk);
            if (!alignOk || padded > input.config.nvS - previous) {
                return SetError(error, "group payload exceeds layout capacity");
            }
            if (output.finalStatus == PLAN_OK && end != previous + padded) {
                return SetError(error, "cuSeqlens does not match group payload");
            }
            previous = end;
        }
    }
    std::set<int32_t> rawSlots;
    for (int32_t src = 0; src < rankSize; ++src) {
        for (int64_t token = 0; token < input.s; ++token) {
            std::set<int32_t> positiveRanks;
            std::set<int32_t> seenRanks;
            for (int64_t k = 0; k < input.topK; ++k) {
                const size_t index = static_cast<size_t>(src * cap + token * input.topK + k);
                const int32_t encoded = output.dst[index];
                const int32_t raw = encoded >= 0 ? encoded : ~encoded;
                if (raw < 0 || raw >= rankSize * input.config.nvS) {
                    return SetError(error, "dst is not decodable");
                }
                if (!rawSlots.insert(raw).second) {
                    return SetError(error, "two routes share one destination slot");
                }
                const int32_t dstRank = static_cast<int32_t>(raw / input.config.nvS);
                const bool first = seenRanks.insert(dstRank).second;
                if (first && encoded < 0) {
                    return SetError(error, "first route to rank is negative");
                }
                if (!first && encoded >= 0) {
                    return SetError(error, "duplicate route to rank is non-negative");
                }
                if (encoded >= 0) {
                    positiveRanks.insert(dstRank);
                }
            }
            if (positiveRanks != seenRanks) {
                return SetError(error, "dedup positive route count mismatch");
            }
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

} // namespace Plan
} // namespace TileXREp
