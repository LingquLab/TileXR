#ifndef TILEXR_EP_PLANNER_COMMON_EP_PLAN_ALGORITHM_IMPL_H
#define TILEXR_EP_PLANNER_COMMON_EP_PLAN_ALGORITHM_IMPL_H

// Single source of truth for both the C++14 CPU oracle and the Ascend C kernel.
// The including translation unit supplies TILEXR_PLAN_FN; pointer address spaces
// are frozen by ep_plan_algorithm.h through TILEXR_PLAN_ADDR.

namespace TileXREp { namespace Plan { namespace {

TILEXR_PLAN_FN int32_t MinI32(int32_t a, int32_t b) { return a < b ? a : b; }
TILEXR_PLAN_FN int32_t MaxI32(int32_t a, int32_t b) { return a > b ? a : b; }
TILEXR_PLAN_FN int64_t ReIndex(int32_t r, int32_t e, int32_t expertNum)
{ return static_cast<int64_t>(r) * expertNum + e; }
TILEXR_PLAN_FN int64_t PairIndex(int32_t s, int32_t d, int32_t rankSize)
{ return static_cast<int64_t>(s) * rankSize + d; }
TILEXR_PLAN_FN int32_t HomeRank(int32_t e, int32_t expertsPerRank) { return e / expertsPerRank; }

TILEXR_PLAN_FN bool SortRanks(TILEXR_PLAN_ADDR int32_t *ranks, int32_t count,
    const TILEXR_PLAN_ADDR int32_t *rankLoad,
    const TILEXR_PLAN_ADDR int32_t *globalRankIds,
    int32_t rankSize, bool ascending)
{
    if (count < 0 || count > rankSize) return false;
    if (count > 0) {
        const int32_t first = ranks[0];
        if (first < 0 || first >= rankSize) return false;
    }
    for (int32_t i = 1; i < count; ++i) {
        const int32_t value = ranks[i];
        if (value < 0 || value >= rankSize) return false;
        int32_t j = i;
        while (j > 0) {
            const int32_t rhs = ranks[j - 1];
            if (rhs < 0 || rhs >= rankSize) return false;

            const int32_t valueLoad = rankLoad[value];
            const int32_t rhsLoad = rankLoad[rhs];
            bool better = false;
            if (valueLoad != rhsLoad) {
                better = ascending ? valueLoad < rhsLoad : valueLoad > rhsLoad;
            } else {
                const int32_t valueGlobal = globalRankIds[value];
                const int32_t rhsGlobal = globalRankIds[rhs];
                better = valueGlobal != rhsGlobal ? valueGlobal < rhsGlobal : value < rhs;
            }
            if (!better) break;
            ranks[j] = rhs;
            --j;
        }
        ranks[j] = value;
    }
    return true;
}

TILEXR_PLAN_FN TileXRMoonEPPlanStatus Validate(const PlanAlgorithmInput &in,
    const PlanAlgorithmOutput &out, const PlanAlgorithmWorkspace &ws)
{
    if (in.rankSize <= 0 || in.rankSize > 128 || in.rank < 0 || in.rank >= in.rankSize ||
        in.s <= 0 || in.topK <= 0 || in.topK > 32 || in.expertNum <= 0 ||
        in.expertNum > INT32_MAX || in.expertNum % in.rankSize != 0 ||
        in.topkExperts == nullptr || in.tokensPerExpert == nullptr || in.globalRankIds == nullptr ||
        out.dst == nullptr || out.cuSeqlens == nullptr || out.expertsToCopy == nullptr ||
        out.remoteStats == nullptr || out.status == nullptr ||
        ((out.remoteExperts == nullptr) != (out.expertTargets == nullptr)) ||
        ws.expertCount == nullptr || ws.rankLoad == nullptr || ws.remainingTpe == nullptr ||
        ws.alloc == nullptr || ws.srcExpertCursor == nullptr || ws.dstExpertCursor == nullptr ||
        ws.expertPhysicalBase == nullptr || ws.localExpertOrdinal == nullptr ||
        ws.tokenSegments == nullptr || ws.scratch == nullptr || ws.affinityOrder == nullptr ||
        ws.scratchCount < in.rankSize * 16 || in.s > INT32_MAX / in.topK) {
        return PLAN_ERROR_CONFIG_MISMATCH;
    }
    const int32_t cap = static_cast<int32_t>(in.s * in.topK);
    const TileXRMoonEPPlanConfig &c = in.config;
    if (c.rankTokenCapacity != cap || c.prefetchSlots <= 0 ||
        c.prefetchSlots > INT32_MAX || c.nvS < cap || c.nvS > INT32_MAX || c.tokenPadding <= 0 ||
        c.tokenRouteLimitPerPair < 0 ||
        c.tokenRouteLimitPerPair > cap || c.cardsPerServer != kPlanCardsPerServer ||
        c.cardsPerCabinet != kPlanCardsPerCabinet || c.crossCandidateCount != kPlanCrossCandidateCount ||
        static_cast<int64_t>(in.rankSize) * c.nvS > INT32_MAX || ws.tokenSegmentCapacity < 0 ||
        (c.prefetchSlots > 0 && ws.remoteExpertSet == nullptr) ||
        (c.tokenRouteLimitPerPair != 0 && ws.routedPairTokens == nullptr)) {
        return PLAN_ERROR_CONFIG_MISMATCH;
    }
    for (int32_t lhs = 0; lhs < in.rankSize; ++lhs) {
        if (in.globalRankIds[lhs] < 0) return PLAN_ERROR_CONFIG_MISMATCH;
        for (int32_t rhs = 0; rhs < lhs; ++rhs)
            if (in.globalRankIds[lhs] == in.globalRankIds[rhs]) return PLAN_ERROR_CONFIG_MISMATCH;
    }
    const int32_t expertNum = static_cast<int32_t>(in.expertNum);
    for (int32_t e = 0; e < expertNum; ++e) ws.expertCount[e] = 0;
    for (int32_t route = 0; route < cap; ++route) {
        const int32_t e = in.topkExperts[route];
        if (e < 0 || e >= expertNum) return PLAN_ERROR_CONFIG_MISMATCH;
        ++ws.expertCount[e];
    }
    for (int32_t src = 0; src < in.rankSize; ++src) {
        int64_t sum = 0;
        for (int32_t e = 0; e < expertNum; ++e) {
            const int32_t value = in.tokensPerExpert[ReIndex(src, e, expertNum)];
            if (value < 0) return PLAN_ERROR_CONFIG_MISMATCH;
            sum += value;
            if (src == in.rank && value != ws.expertCount[e]) return PLAN_ERROR_TPE_MISMATCH;
        }
        if (sum != cap) return PLAN_ERROR_TPE_MISMATCH;
    }
    return PLAN_OK;
}

struct State {
    const PlanAlgorithmInput &in;
    const PlanAlgorithmOutput &out;
    PlanAlgorithmWorkspace &ws;
    int32_t rankSize, expertNum, expertsPerRank, cap, b, routeLimit;
    int32_t localSegments, globalSegments, expertMoves, intraRounds, interRounds;
    bool slotBlock, supplyBlock, pairBlock;
    TileXRMoonEPPlanStatus fatal;
    TILEXR_PLAN_FN State(const PlanAlgorithmInput &input, const PlanAlgorithmOutput &output,
        PlanAlgorithmWorkspace &workspace)
        : in(input), out(output), ws(workspace), rankSize(input.rankSize),
          expertNum(static_cast<int32_t>(input.expertNum)),
          expertsPerRank(static_cast<int32_t>(input.expertNum / input.rankSize)),
          cap(static_cast<int32_t>(input.config.rankTokenCapacity)),
          b(static_cast<int32_t>(input.config.prefetchSlots)),
          routeLimit(static_cast<int32_t>(input.config.tokenRouteLimitPerPair)),
          localSegments(0), globalSegments(0), expertMoves(0), intraRounds(0), interRounds(0),
          slotBlock(false), supplyBlock(false), pairBlock(false), fatal(PLAN_OK) {}
};

TILEXR_PLAN_FN bool AppendSegment(State &s, int32_t expert, int32_t src, int32_t dst,
    int32_t srcBegin, int32_t dstBegin, int32_t count)
{
    ++s.globalSegments;
    if (src != s.in.rank) return true;
    if (s.localSegments >= s.ws.tokenSegmentCapacity) {
        s.fatal = PLAN_ERROR_MOVE_RECORD_OVERFLOW;
        return false;
    }
    TILEXR_PLAN_ADDR TokenSegmentMove *m = &s.ws.tokenSegments[s.localSegments++];
    m->expertId = expert; m->srcRank = src; m->dstRank = dst;
    m->srcExpertBegin = srcBegin; m->dstExpertBegin = dstBegin; m->tokenCount = count;
    return true;
}

TILEXR_PLAN_FN bool BetterAffinity(int32_t lhs, int32_t rhs, int32_t dst, const PlanAlgorithmInput &in)
{
    const uint32_t did = static_cast<uint32_t>(in.globalRankIds[dst]);
    const uint32_t lid = static_cast<uint32_t>(in.globalRankIds[lhs]);
    const uint32_t rid = static_cast<uint32_t>(in.globalRankIds[rhs]);
    const uint32_t ld = lid ^ did, rd = rid ^ did;
    if (ld != rd) return ld < rd;
    if (lid != rid) return lid < rid;
    return lhs < rhs;
}

TILEXR_PLAN_FN void BuildAffinity(State &s)
{
    for (int32_t dst = 0; dst < s.rankSize; ++dst) {
        TILEXR_PLAN_ADDR int32_t *row = s.ws.affinityOrder + static_cast<int64_t>(dst) * s.rankSize;
        for (int32_t r = 0; r < s.rankSize; ++r) row[r] = r;
        for (int32_t i = 1; i < s.rankSize; ++i) {
            const int32_t value = row[i]; int32_t j = i;
            while (j > 0 && BetterAffinity(value, row[j - 1], dst, s.in)) {
                row[j] = row[j - 1]; --j;
            }
            row[j] = value;
        }
    }
}

TILEXR_PLAN_FN int32_t FindRemote(const State &s, int32_t dst, int32_t expert)
{
    for (int32_t slot = 0; slot < s.ws.scratch[dst]; ++slot)
        if (s.ws.remoteExpertSet[static_cast<int64_t>(dst) * s.b + slot] == expert) return slot;
    return -1;
}

TILEXR_PLAN_FN int32_t ApplyExpertMove(State &s, int32_t home, int32_t dst, int32_t requested)
{
    if (requested <= 0 || home == dst || s.fatal != PLAN_OK) return 0;
    int32_t orderCount = 0;
    const int32_t begin = home * s.expertsPerRank, end = begin + s.expertsPerRank;
    for (int32_t e = begin; e < end; ++e)
        if (s.ws.alloc[ReIndex(home, e, s.expertNum)] > 0) s.ws.expertCount[orderCount++] = e;
    for (int32_t i = 1; i < orderCount; ++i) {
        const int32_t value = s.ws.expertCount[i];
        const int32_t valueAlloc = s.ws.alloc[ReIndex(home, value, s.expertNum)];
        int32_t j = i;
        while (j > 0) {
            const int32_t prior = s.ws.expertCount[j - 1];
            const int32_t priorAlloc = s.ws.alloc[ReIndex(home, prior, s.expertNum)];
            if (valueAlloc < priorAlloc || (valueAlloc == priorAlloc && value > prior)) break;
            s.ws.expertCount[j] = prior; --j;
        }
        s.ws.expertCount[j] = value;
    }
    int32_t demand = requested;
    bool found = false;
    for (int32_t oi = 0; oi < orderCount && demand > 0; ++oi) {
        const int32_t expert = s.ws.expertCount[oi];
        const int64_t homeIndex = ReIndex(home, expert, s.expertNum);
        const int32_t availableAlloc = s.ws.alloc[homeIndex];
        if (availableAlloc <= 0) continue;
        const bool alreadyRemote = FindRemote(s, dst, expert) >= 0;
        if (!alreadyRemote && s.ws.scratch[dst] >= s.b) { s.slotBlock = true; continue; }
        found = true;
        const int32_t expertRequest = MinI32(availableAlloc, demand);
        int32_t expertDemand = expertRequest, actual = 0;
        TILEXR_PLAN_ADDR int32_t *affinity = s.ws.affinityOrder + static_cast<int64_t>(dst) * s.rankSize;
        for (int32_t ai = 0; ai < s.rankSize && expertDemand > 0; ++ai) {
            const int32_t src = affinity[ai];
            const int64_t re = ReIndex(src, expert, s.expertNum);
            int32_t pairRemain = expertDemand;
            if (s.routeLimit != 0) {
                pairRemain = s.routeLimit - s.ws.routedPairTokens[PairIndex(src, dst, s.rankSize)];
                if (pairRemain < 0) pairRemain = 0;
            }
            const int32_t take = MinI32(MinI32(s.ws.remainingTpe[re], expertDemand), pairRemain);
            if (take <= 0) continue;
            const int64_t dstRe = ReIndex(dst, expert, s.expertNum);
            if (!AppendSegment(s, expert, src, dst, s.ws.srcExpertCursor[re],
                s.ws.dstExpertCursor[dstRe], take)) return 0;
            s.ws.remainingTpe[re] -= take; s.ws.srcExpertCursor[re] += take;
            s.ws.dstExpertCursor[dstRe] += take;
            if (s.routeLimit != 0) s.ws.routedPairTokens[PairIndex(src, dst, s.rankSize)] += take;
            actual += take; expertDemand -= take;
        }
        if (actual > 0) {
            if (!alreadyRemote) {
                const int32_t slot = s.ws.scratch[dst]++;
                s.ws.remoteExpertSet[static_cast<int64_t>(dst) * s.b + slot] = expert;
            }
            s.ws.alloc[homeIndex] -= actual;
            s.ws.alloc[ReIndex(dst, expert, s.expertNum)] += actual;
            s.ws.rankLoad[home] -= actual; s.ws.rankLoad[dst] += actual;
            demand -= actual; ++s.expertMoves;
        }
        if (expertDemand > 0) {
            int64_t supply = 0; bool routeBlocked = false;
            for (int32_t src = 0; src < s.rankSize; ++src) {
                const int64_t re = ReIndex(src, expert, s.expertNum);
                supply += s.ws.remainingTpe[re];
                if (s.routeLimit != 0 && s.ws.remainingTpe[re] > 0 &&
                    s.ws.routedPairTokens[PairIndex(src, dst, s.rankSize)] >= s.routeLimit)
                    routeBlocked = true;
            }
            if (routeBlocked) s.pairBlock = true;
            else if (supply == 0) s.supplyBlock = true;
        }
    }
    if (demand > 0 && !found) s.pairBlock = true;
    return requested - demand;
}
TILEXR_PLAN_FN void RunIntra(State &s)
{
    TILEXR_PLAN_ADDR int32_t *active = s.ws.scratch + s.rankSize;
    TILEXR_PLAN_ADDR int32_t *senders = s.ws.scratch + s.rankSize * 2;
    TILEXR_PLAN_ADDR int32_t *receivers = s.ws.scratch + s.rankSize * 3;
    int32_t previousKey = INT_MIN;
    while (true) {
        int32_t serverKey = INT_MAX;
        for (int32_t r = 0; r < s.rankSize; ++r) {
            const int32_t key = s.in.globalRankIds[r] / kPlanCardsPerServer;
            if (key > previousKey && key < serverKey) serverKey = key;
        }
        if (serverKey == INT_MAX) break;
        previousKey = serverKey;
        while (true) {
            int32_t activeCount = 0;
            for (int32_t r = 0; r < s.rankSize; ++r)
                if (s.in.globalRankIds[r] / kPlanCardsPerServer == serverKey && s.ws.rankLoad[r] != s.cap)
                    active[activeCount++] = r;
            if (!SortRanks(active, activeCount, s.ws.rankLoad, s.in.globalRankIds,
                s.rankSize, false)) {
                s.fatal = PLAN_ERROR_INTERNAL_INVARIANT;
                return;
            }
            if (activeCount < 2 || s.ws.rankLoad[active[0]] + s.ws.rankLoad[active[1]] < s.cap ||
                s.ws.rankLoad[active[activeCount - 1]] > s.cap) break;
            int32_t middle = activeCount / 2;
            while (middle > 0 && s.ws.rankLoad[active[middle - 1]] +
                s.ws.rankLoad[active[middle]] <= s.cap) --middle;
            if (middle == 0) break;
            int32_t senderCount = 0, receiverCount = 0;
            for (int32_t i = middle - 1; i >= 0; --i)
                if (s.ws.rankLoad[active[i]] > s.cap) senders[senderCount++] = active[i];
            for (int32_t i = middle; i < activeCount; ++i)
                if (s.ws.rankLoad[active[i]] < s.cap) receivers[receiverCount++] = active[i];
            const int32_t pairs = MinI32(senderCount, receiverCount);
            int32_t actualRound = 0;
            for (int32_t i = 0; i < pairs; ++i) {
                const int32_t requested = MinI32(s.ws.rankLoad[senders[i]] - s.cap,
                    s.cap - s.ws.rankLoad[receivers[i]]);
                if (requested > 0) actualRound += ApplyExpertMove(s, senders[i], receivers[i], requested);
                if (s.fatal != PLAN_OK) return;
            }
            if (actualRound == 0) { s.pairBlock = true; break; }
            ++s.intraRounds;
        }
    }
}

TILEXR_PLAN_FN bool RankInList(int32_t rank, const TILEXR_PLAN_ADDR int32_t *list, int32_t count)
{
    for (int32_t i = 0; i < count; ++i) if (list[i] == rank) return true;
    return false;
}

TILEXR_PLAN_FN void RunInter(State &s)
{
    TILEXR_PLAN_ADDR int32_t *active = s.ws.scratch + s.rankSize;
    TILEXR_PLAN_ADDR int32_t *senders = s.ws.scratch + s.rankSize * 2;
    TILEXR_PLAN_ADDR int32_t *receivers = s.ws.scratch + s.rankSize * 3;
    TILEXR_PLAN_ADDR int32_t *occupied = s.ws.scratch + s.rankSize * 4;
    while (true) {
        bool allBalanced = true;
        int32_t activeCount = 0, underloaded = 0;
        for (int32_t r = 0; r < s.rankSize; ++r) {
            if (s.ws.rankLoad[r] != s.cap) {
                allBalanced = false; active[activeCount++] = r;
                if (s.ws.rankLoad[r] < s.cap) ++underloaded;
            }
        }
        if (allBalanced || activeCount < 2 || underloaded == 0 || underloaded == activeCount) break;
        if (!SortRanks(active, activeCount, s.ws.rankLoad, s.in.globalRankIds,
            s.rankSize, false)) {
            s.fatal = PLAN_ERROR_INTERNAL_INVARIANT;
            return;
        }
        const int32_t receiverCount = MinI32(underloaded, activeCount / 2);
        if (receiverCount <= 0) break;
        const int32_t senderCount = activeCount - receiverCount;
        for (int32_t i = 0; i < senderCount; ++i) senders[i] = active[i];
        for (int32_t i = 0; i < receiverCount; ++i) receivers[i] = active[senderCount + i];
        if (!SortRanks(receivers, receiverCount, s.ws.rankLoad, s.in.globalRankIds,
            s.rankSize, true)) {
            s.fatal = PLAN_ERROR_INTERNAL_INVARIANT;
            return;
        }
        for (int32_t r = 0; r < s.rankSize; ++r) occupied[r] = 0;
        int32_t actualRound = 0;
        for (int32_t ri = 0; ri < receiverCount; ++ri) {
            const int32_t dst = receivers[ri];
            TILEXR_PLAN_ADDR int32_t *affinity = s.ws.affinityOrder + static_cast<int64_t>(dst) * s.rankSize;
            int32_t staticCandidates = 0;
            for (int32_t ai = 0; ai < s.rankSize && staticCandidates < kPlanCrossCandidateCount; ++ai) {
                const int32_t src = affinity[ai];
                if (!RankInList(src, senders, senderCount)) continue;
                ++staticCandidates;
                if (occupied[src]) continue;
                const int32_t requested = MinI32(s.ws.rankLoad[src] - s.cap,
                    s.cap - s.ws.rankLoad[dst]);
                if (requested <= 0) continue;
                const int32_t actual = ApplyExpertMove(s, src, dst, requested);
                if (s.fatal != PLAN_OK) return;
                if (actual > 0) { actualRound += actual; occupied[src] = 1; break; }
            }
        }
        if (actualRound == 0) { s.pairBlock = true; break; }
        ++s.interRounds;
    }
}

TILEXR_PLAN_FN void AppendRemainders(State &s)
{
    for (int32_t expert = 0; expert < s.expertNum; ++expert) {
        const int32_t dst = HomeRank(expert, s.expertsPerRank);
        for (int32_t src = 0; src < s.rankSize; ++src) {
            const int64_t re = ReIndex(src, expert, s.expertNum);
            const int32_t count = s.ws.remainingTpe[re];
            if (count <= 0) continue;
            const int64_t dstRe = ReIndex(dst, expert, s.expertNum);
            if (!AppendSegment(s, expert, src, dst, s.ws.srcExpertCursor[re],
                s.ws.dstExpertCursor[dstRe], count)) return;
            s.ws.remainingTpe[re] = 0; s.ws.srcExpertCursor[re] += count;
            s.ws.dstExpertCursor[dstRe] += count;
        }
    }
}

TILEXR_PLAN_FN bool BetterRemote(int32_t lhs, int32_t rhs, int32_t dst, const State &s)
{
    const int32_t la = s.ws.alloc[ReIndex(dst, lhs, s.expertNum)];
    const int32_t ra = s.ws.alloc[ReIndex(dst, rhs, s.expertNum)];
    if (la != ra) return la > ra;
    return lhs < rhs;
}

TILEXR_PLAN_FN int32_t AlignCount(int32_t value, int64_t alignment, bool &ok)
{
    if (value > 0 && alignment > INT32_MAX) { ok = false; return 0; }
    const int64_t aligned = ((static_cast<int64_t>(value) + alignment - 1) / alignment) * alignment;
    if (aligned > INT32_MAX) { ok = false; return 0; }
    return static_cast<int32_t>(aligned);
}

TILEXR_PLAN_FN TileXRMoonEPPlanStatus BuildLayouts(State &s, int32_t &maxRemote, int32_t &localEnd)
{
    const int64_t groups = static_cast<int64_t>(s.expertNum) + s.b;
    const int64_t remoteSlotCount = static_cast<int64_t>(s.rankSize) * s.b;
    for (int64_t i = 0; i < remoteSlotCount; ++i) s.ws.remoteExpertSet[i] = -1;
    for (int32_t slot = 0; slot < s.b; ++slot) s.out.expertsToCopy[slot] = -1;
    const int32_t targetWords = (s.rankSize + 63) / 64;
    if (s.out.remoteExperts != nullptr) {
        for (int64_t i = 0; i < remoteSlotCount; ++i) s.out.remoteExperts[i] = -1;
        for (int64_t i = 0; i < static_cast<int64_t>(s.expertsPerRank) * targetWords; ++i)
            s.out.expertTargets[i] = 0;
    }
    for (int64_t g = 0; g < groups; ++g) {
        s.out.cuSeqlens[g] = 0;
    }
    s.out.remoteStats[0] = 0; s.out.remoteStats[1] = 0;
    maxRemote = 0; localEnd = 0;
    for (int32_t i = 0; i < s.rankSize * s.expertNum; ++i) s.ws.expertPhysicalBase[i] = 0;

    for (int32_t dst = 0; dst < s.rankSize; ++dst) {
        int32_t remoteCount = 0;
        TILEXR_PLAN_ADDR int32_t *row = s.ws.remoteExpertSet + static_cast<int64_t>(dst) * s.b;
        for (int32_t expert = 0; expert < s.expertNum; ++expert) {
            if (HomeRank(expert, s.expertsPerRank) == dst ||
                s.ws.alloc[ReIndex(dst, expert, s.expertNum)] <= 0) continue;
            if (remoteCount >= s.b) return PLAN_PARTIAL_PREFETCH_SLOT_EXHAUSTED;
            int32_t pos = remoteCount;
            while (pos > 0 && BetterRemote(expert, row[pos - 1], dst, s)) {
                row[pos] = row[pos - 1]; --pos;
            }
            row[pos] = expert; ++remoteCount;
        }
        maxRemote = MaxI32(maxRemote, remoteCount);
        if (dst == s.in.rank) s.out.remoteStats[0] = remoteCount;
        for (int32_t slot = 0; slot < remoteCount; ++slot)
            if (HomeRank(row[slot], s.expertsPerRank) == s.in.rank) ++s.out.remoteStats[1];
        if (dst == s.in.rank)
            for (int32_t slot = 0; slot < s.b; ++slot) s.out.expertsToCopy[slot] = row[slot];
        if (s.out.remoteExperts != nullptr) {
            for (int32_t slot = 0; slot < s.b; ++slot) {
                const int32_t expert = row[slot];
                s.out.remoteExperts[static_cast<int64_t>(dst) * s.b + slot] = expert;
                if (expert >= 0 && HomeRank(expert, s.expertsPerRank) == s.in.rank) {
                    const int32_t localExpert = expert - s.in.rank * s.expertsPerRank;
                    s.out.expertTargets[static_cast<int64_t>(localExpert) * targetWords + dst / 64] |=
                        (1ULL << (dst % 64));
                }
            }
        }
    }

    for (int32_t dst = 0; dst < s.rankSize; ++dst) {
        int32_t previousEnd = 0;
        TILEXR_PLAN_ADDR int32_t *row = s.ws.remoteExpertSet + static_cast<int64_t>(dst) * s.b;
        for (int64_t group = 0; group < groups; ++group) {
            int32_t count = 0, physicalExpert = -1;
            if (group < s.expertNum) {
                bool remote = false;
                for (int32_t slot = 0; slot < s.b; ++slot) if (row[slot] == group) { remote = true; break; }
                if (!remote) { physicalExpert = group; count = s.ws.alloc[ReIndex(dst, group, s.expertNum)]; }
            } else {
                const int64_t slot = group - s.expertNum;
                if (slot < s.b && row[slot] >= 0) {
                    physicalExpert = row[slot]; count = s.ws.alloc[ReIndex(dst, physicalExpert, s.expertNum)];
                }
            }
            if (physicalExpert >= 0)
                s.ws.expertPhysicalBase[ReIndex(dst, physicalExpert, s.expertNum)] = previousEnd;
            bool ok = true;
            const int32_t padded = count == 0 ? 0 : AlignCount(count,
                s.in.config.tokenPadding, ok);
            if (!ok || padded > s.in.config.nvS - previousEnd) return PLAN_ERROR_LAYOUT_EXCEEDS_NVS;
            if (dst == s.in.rank) s.out.cuSeqlens[group] = previousEnd + padded;
            previousEnd += padded;
        }
        if (dst == s.in.rank) localEnd = previousEnd;
    }
    return PLAN_OK;
}
TILEXR_PLAN_FN TileXRMoonEPPlanStatus BuildDst(State &s)
{
    for (int32_t route = 0; route < s.cap; ++route) s.out.dst[route] = 0;
    for (int32_t expert = 0; expert < s.expertNum; ++expert) s.ws.expertCount[expert] = 0;
    for (int32_t route = 0; route < s.cap; ++route) {
        const int32_t expert = s.in.topkExperts[route];
        s.ws.localExpertOrdinal[route] = s.ws.expertCount[expert]++;
    }
    for (int32_t token = 0; token < s.in.s; ++token) {
        for (int32_t k = 0; k < s.in.topK; ++k) {
            const int32_t route = static_cast<int32_t>(token * s.in.topK + k);
            const int32_t expert = s.in.topkExperts[route];
            const int32_t ordinal = s.ws.localExpertOrdinal[route];
            const TILEXR_PLAN_ADDR TokenSegmentMove *matched = nullptr;
            for (int32_t i = 0; i < s.localSegments; ++i) {
                const TILEXR_PLAN_ADDR TokenSegmentMove *move = &s.ws.tokenSegments[i];
                if (move->expertId == expert && ordinal >= move->srcExpertBegin &&
                    ordinal < move->srcExpertBegin + move->tokenCount) { matched = move; break; }
            }
            if (matched == nullptr) return PLAN_ERROR_INTERNAL_INVARIANT;
            const int64_t localOffset = static_cast<int64_t>(s.ws.expertPhysicalBase[
                ReIndex(matched->dstRank, expert, s.expertNum)]) + matched->dstExpertBegin +
                (ordinal - matched->srcExpertBegin);
            const int64_t raw = static_cast<int64_t>(matched->dstRank) * s.in.config.nvS + localOffset;
            if (localOffset < 0 || localOffset >= s.in.config.nvS || raw < 0 || raw > INT32_MAX)
                return PLAN_ERROR_INTERNAL_INVARIANT;
            bool first = true;
            for (int32_t prior = 0; prior < k; ++prior) {
                const int32_t encoded = s.out.dst[static_cast<int32_t>(token * s.in.topK + prior)];
                const int32_t decoded = encoded >= 0 ? encoded : ~encoded;
                if (decoded / static_cast<int32_t>(s.in.config.nvS) == matched->dstRank) {
                    first = false; break;
                }
            }
            const int32_t rawDst = static_cast<int32_t>(raw);
            s.out.dst[route] = first ? rawDst : ~rawDst;
        }
    }
    return PLAN_OK;
}

TILEXR_PLAN_FN void FillStatus(State &s, TileXRMoonEPPlanStatus status, int32_t maxRemote, int32_t localEnd)
{
    int32_t deficit = 0;
    for (int32_t r = 0; r < s.rankSize; ++r)
        if (s.ws.rankLoad[r] < s.cap) deficit += s.cap - s.ws.rankLoad[r];
    s.out.status[0] = static_cast<int32_t>(status);
    s.out.status[1] = s.intraRounds;
    s.out.status[2] = s.interRounds;
    s.out.status[3] = deficit;
    s.out.status[4] = maxRemote;
    s.out.status[5] = localEnd;
    s.out.status[6] = s.expertMoves;
    s.out.status[7] = s.localSegments;
}

} // namespace

TILEXR_PLAN_FN TileXRMoonEPPlanStatus RunPlanAlgorithm(const PlanAlgorithmInput &in,
    const PlanAlgorithmOutput &out, PlanAlgorithmWorkspace &ws)
{
    if (out.status != nullptr)
        for (int32_t i = 0; i < kPlanStatusWords; ++i) out.status[i] = 0;
    const TileXRMoonEPPlanStatus validation = Validate(in, out, ws);
    if (validation != PLAN_OK) {
        if (out.status != nullptr) out.status[0] = static_cast<int32_t>(validation);
        return validation;
    }
    State s(in, out, ws);
    for (int32_t r = 0; r < s.rankSize; ++r) { ws.rankLoad[r] = 0; ws.scratch[r] = 0; }
    const int64_t reCount = static_cast<int64_t>(s.rankSize) * s.expertNum;
    for (int64_t i = 0; i < reCount; ++i) {
        ws.remainingTpe[i] = in.tokensPerExpert[i]; ws.alloc[i] = 0;
        ws.srcExpertCursor[i] = 0; ws.dstExpertCursor[i] = 0; ws.expertPhysicalBase[i] = 0;
    }
    const int64_t remoteSlotCount = static_cast<int64_t>(s.rankSize) * s.b;
    for (int64_t i = 0; i < remoteSlotCount; ++i) ws.remoteExpertSet[i] = -1;
    if (s.routeLimit != 0)
        for (int32_t i = 0; i < s.rankSize * s.rankSize; ++i) ws.routedPairTokens[i] = 0;
    for (int32_t expert = 0; expert < s.expertNum; ++expert) {
        int32_t count = 0;
        for (int32_t src = 0; src < s.rankSize; ++src)
            count += in.tokensPerExpert[ReIndex(src, expert, s.expertNum)];
        const int32_t home = HomeRank(expert, s.expertsPerRank);
        ws.alloc[ReIndex(home, expert, s.expertNum)] = count;
        ws.rankLoad[home] += count;
    }
    if (!ws.affinityOrderValid) {
        BuildAffinity(s);
        ws.affinityOrderValid = true;
    }
    RunIntra(s);
    if (s.fatal == PLAN_OK) RunInter(s);
    if (s.fatal == PLAN_OK) AppendRemainders(s);
    if (s.fatal != PLAN_OK) { FillStatus(s, s.fatal, 0, 0); return s.fatal; }

    bool balanced = true;
    for (int32_t r = 0; r < s.rankSize; ++r)
        if (ws.rankLoad[r] != s.cap) { balanced = false; break; }
    TileXRMoonEPPlanStatus balanceStatus = PLAN_OK;
    if (!balanced) {
        if (s.slotBlock) balanceStatus = PLAN_PARTIAL_PREFETCH_SLOT_EXHAUSTED;
        else if (s.supplyBlock) balanceStatus = PLAN_PARTIAL_TOKEN_SUPPLY;
        else balanceStatus = PLAN_PARTIAL_NO_FEASIBLE_PAIR;
    }
    int32_t maxRemote = 0, localEnd = 0;
    const TileXRMoonEPPlanStatus layoutStatus = BuildLayouts(s, maxRemote, localEnd);
    TileXRMoonEPPlanStatus status = balanceStatus;
    if (balanced) {
        status = layoutStatus;
        if (status == PLAN_OK) status = BuildDst(s);
    } else if (layoutStatus == PLAN_OK) {
        (void)BuildDst(s);
    }
    FillStatus(s, status, maxRemote, localEnd);
    return status;
}

} } // namespace TileXREp::Plan

#endif
