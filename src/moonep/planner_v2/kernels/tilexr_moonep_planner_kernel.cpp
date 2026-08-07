#include "kernel_operator.h"

#include "comm_args.h"
#include "ep_plan_types.h"
#define TILEXR_PLAN_MAILBOX_FN __aicore__ inline
#include "ep_plan_peer_mailbox.h"
#undef TILEXR_PLAN_MAILBOX_FN
#include "planner_common.h"
#include "tilexr_ep_plan.h"
#include "tilexr_sync.h"

#define TILEXR_PLAN_ADDR __gm__
#define TILEXR_PLAN_FN __aicore__ inline
#include "ep_plan_algorithm.h"
#include "ep_plan_algorithm_impl.h"
#undef TILEXR_PLAN_FN
#undef TILEXR_PLAN_ADDR

namespace {
using TileXREp::Plan::PlanAlgorithmInput;
using TileXREp::Plan::PlanAlgorithmOutput;
using TileXREp::Plan::PlanAlgorithmWorkspace;
using TileXREp::Plan::PlanCallHeader;
using TileXREp::Plan::PlanEpochState;
using TileXREp::Plan::PlanPeerMailboxLayout;
using TileXREp::Plan::TokenSegmentMove;

constexpr uint64_t kAlign = TileXREp::Plan::kPlanWorkspaceAlignment;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr int32_t kPhaseData = 1;
constexpr int32_t kPhaseStatus = 2;
constexpr int32_t kPhaseReady = 3;
constexpr int32_t kPhaseConsensus = 4;
constexpr uint64_t kBarrierSpinLimit = 1ULL << 24;
constexpr uint64_t kCacheLineBytes = 64;
constexpr uint64_t kPlannerBarrierStrideBytes = 512;
constexpr uint32_t kPlannerBarrierWords = kPlannerBarrierStrideBytes / sizeof(uint64_t);
constexpr uint64_t kPlannerBarrierBaseBytes =
    static_cast<uint64_t>(TileXRMoonEp::kPlannerReadyEventId) * SYNC_UNIT_SIZE;
constexpr uint64_t kPlannerBarrierPhaseCount = 4;
constexpr int32_t kBarrierDebugBuildTag = 0x5A17C0DE;

static_assert(kPlannerBarrierBaseBytes +
    kPlannerBarrierPhaseCount * TileXR::TILEXR_MAX_RANK_SIZE * kPlannerBarrierStrideBytes <=
    static_cast<uint64_t>(TileXR::IPC_DATA_OFFSET),
    "planner barrier slots must remain inside the IPC flag region");

struct BarrierDebug {
    int64_t mteValue;
    int64_t scalarValue;
    GM_ADDR peerAddress;
    uint64_t publishSelfAddress;
    uint64_t publishRemoteAddress;
    int32_t peer;
    int32_t phase;
};

__aicore__ inline uint64_t AlignUp(uint64_t value)
{
    return (value + kAlign - 1) & ~(kAlign - 1);
}

__aicore__ inline uint64_t AddRegion(uint64_t &cursor, uint64_t bytes)
{
    cursor = AlignUp(cursor);
    const uint64_t offset = cursor;
    cursor += bytes;
    return offset;
}

struct MetaOffsets {
    uint64_t headers;
    uint64_t tpe;
    uint64_t globalRankIds;
    uint64_t epochState;
    uint64_t affinityOrder;
    uint64_t localStatus;
    uint64_t barrierFlags;
};

struct LocalOffsets {
    uint64_t expertCount;
    uint64_t rankLoad;
    uint64_t remainingTpe;
    uint64_t alloc;
    uint64_t remoteExpertSet;
    uint64_t srcExpertCursor;
    uint64_t dstExpertCursor;
    uint64_t expertPhysicalBase;
    uint64_t localExpertOrdinal;
    uint64_t tokenSegments;
    uint64_t routedPairTokens;
    uint64_t scratch;
};

__aicore__ inline MetaOffsets BuildMetaOffsets(int64_t rankSize, int64_t expertNum)
{
    MetaOffsets offsets {};
    uint64_t cursor = 0;
    offsets.headers = AddRegion(cursor, static_cast<uint64_t>(rankSize) *
        TileXREp::Plan::kPlanHeaderStrideBytes);
    offsets.tpe = AddRegion(cursor, static_cast<uint64_t>(rankSize) *
        static_cast<uint64_t>(expertNum) * sizeof(int32_t));
    offsets.globalRankIds = AddRegion(cursor, static_cast<uint64_t>(rankSize) * sizeof(int32_t));
    offsets.epochState = AddRegion(cursor, sizeof(PlanEpochState));
    offsets.affinityOrder = AddRegion(cursor, static_cast<uint64_t>(rankSize) *
        static_cast<uint64_t>(rankSize) * sizeof(int32_t));
    offsets.localStatus = AddRegion(cursor, static_cast<uint64_t>(rankSize) *
        TileXREp::Plan::kPlanStatusStrideBytes);
    offsets.barrierFlags = AddRegion(cursor, static_cast<uint64_t>(rankSize) * 3ULL *
        TileXREp::Plan::kPlanBarrierSlotBytes);
    return offsets;
}

__aicore__ inline LocalOffsets BuildLocalOffsets(int64_t rankSize, int64_t s, int64_t topK,
    int64_t expertNum, int64_t prefetchSlots, int64_t tokenRouteLimitPerPair)
{
    LocalOffsets offsets {};
    const uint64_t r = static_cast<uint64_t>(rankSize);
    const uint64_t e = static_cast<uint64_t>(expertNum);
    const uint64_t cap = static_cast<uint64_t>(s) * static_cast<uint64_t>(topK);
    uint64_t cursor = 0;
    offsets.expertCount = AddRegion(cursor, e * sizeof(int32_t));
    offsets.rankLoad = AddRegion(cursor, r * sizeof(int32_t));
    offsets.remainingTpe = AddRegion(cursor, r * e * sizeof(int32_t));
    offsets.alloc = AddRegion(cursor, r * e * sizeof(int32_t));
    offsets.remoteExpertSet = AddRegion(cursor, r * static_cast<uint64_t>(prefetchSlots) * sizeof(int32_t));
    offsets.srcExpertCursor = AddRegion(cursor, r * e * sizeof(int32_t));
    offsets.dstExpertCursor = AddRegion(cursor, r * e * sizeof(int32_t));
    offsets.expertPhysicalBase = AddRegion(cursor, r * e * sizeof(int32_t));
    offsets.localExpertOrdinal = AddRegion(cursor, cap * sizeof(int32_t));
    offsets.tokenSegments = AddRegion(cursor, cap * sizeof(TokenSegmentMove));
    offsets.routedPairTokens = tokenRouteLimitPerPair == 0 ? 0 :
        AddRegion(cursor, r * r * sizeof(int32_t));
    offsets.scratch = AddRegion(cursor, r * 16ULL * sizeof(int32_t));
    return offsets;
}

__aicore__ inline uint64_t BuildTopologyHash(const __gm__ int32_t *globalRankIds, int64_t rankSize)
{
    uint64_t hash = kFnvOffset;
    for (int64_t rank = 0; rank < rankSize; ++rank) {
        hash ^= static_cast<uint64_t>(static_cast<uint32_t>(globalRankIds[rank]));
        hash *= kFnvPrime;
    }
    return hash;
}

__aicore__ inline void RefreshCacheLines(GM_ADDR address, uint64_t bytes)
{
    if (address == nullptr || bytes == 0) return;
    const uint64_t raw = reinterpret_cast<uint64_t>(address);
    const uint64_t begin = raw / kCacheLineBytes * kCacheLineBytes;
    const uint64_t end = (raw + bytes - 1) / kCacheLineBytes * kCacheLineBytes;
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(begin));
    for (uint64_t offset = 0; offset <= end - begin; offset += kCacheLineBytes) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(global[offset]);
        __asm__ __volatile__("");
    }
}

__aicore__ inline void CopyWords(__gm__ int32_t *dst, const __gm__ int32_t *src, int64_t count)
{
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(const_cast<__gm__ int32_t *>(src)),
        static_cast<uint64_t>(count) * sizeof(int32_t));
    for (int64_t index = 0; index < count; ++index) dst[index] = src[index];
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(dst), static_cast<uint64_t>(count) * sizeof(int32_t));
}

__aicore__ inline __gm__ PlanCallHeader *HeaderAt(
    GM_ADDR meta, const MetaOffsets &offsets, int64_t rank)
{
    return reinterpret_cast<__gm__ PlanCallHeader *>(meta + offsets.headers +
        static_cast<uint64_t>(rank) * TileXREp::Plan::kPlanHeaderStrideBytes);
}

__aicore__ inline GM_ADDR StatusRow(GM_ADDR meta, const MetaOffsets &offsets, int64_t rank)
{
    return meta + offsets.localStatus + static_cast<uint64_t>(rank) *
        TileXREp::Plan::kPlanStatusStrideBytes;
}

__aicore__ inline void FillHeader(__gm__ PlanCallHeader *header, int64_t rankSize,
    int64_t s, int64_t topK, int64_t expertNum, int64_t prefetchSlots,
    int64_t rankTokenCapacity, int64_t nvS, int64_t tokenPadding,
    int64_t tokenRouteLimitPerPair, int32_t cardsPerServer, int32_t cardsPerCabinet,
    int32_t crossCandidateCount, uint64_t epoch, uint64_t topologyHash)
{
    header->abiVersion = TileXREp::Plan::kPlanAbiVersion;
    header->headerBytes = sizeof(PlanCallHeader);
    header->rankSize = static_cast<int32_t>(rankSize);
    header->reserved0 = 0;
    header->s = s;
    header->k = topK;
    header->expertNum = expertNum;
    header->prefetchSlots = prefetchSlots;
    header->rankTokenCapacity = rankTokenCapacity;
    header->nvS = nvS;
    header->tokenPadding = tokenPadding;
    header->tokenRouteLimitPerPair = tokenRouteLimitPerPair;
    header->cardsPerServer = cardsPerServer;
    header->cardsPerCabinet = cardsPerCabinet;
    header->crossCandidateCount = crossCandidateCount;
    header->reserved1 = 0;
    header->epoch = epoch;
    header->topologyHash = topologyHash;
}

__aicore__ inline bool HeadersMatch(GM_ADDR meta, const MetaOffsets &offsets, int64_t rankSize)
{
    const __gm__ PlanCallHeader *expected = HeaderAt(meta, offsets, 0);
    for (int64_t rank = 1; rank < rankSize; ++rank) {
        const __gm__ PlanCallHeader *current = HeaderAt(meta, offsets, rank);
        if (current->abiVersion != expected->abiVersion || current->headerBytes != expected->headerBytes ||
            current->rankSize != expected->rankSize || current->s != expected->s || current->k != expected->k ||
            current->expertNum != expected->expertNum || current->prefetchSlots != expected->prefetchSlots ||
            current->rankTokenCapacity != expected->rankTokenCapacity || current->nvS != expected->nvS ||
            current->tokenPadding != expected->tokenPadding ||
            current->tokenRouteLimitPerPair != expected->tokenRouteLimitPerPair ||
            current->cardsPerServer != expected->cardsPerServer ||
            current->cardsPerCabinet != expected->cardsPerCabinet ||
            current->crossCandidateCount != expected->crossCandidateCount || current->epoch != expected->epoch ||
            current->topologyHash != expected->topologyHash) return false;
    }
    return true;
}

__aicore__ inline int32_t BarrierEventId(int32_t phase, int32_t sourceRank, int32_t rankSize)
{
    return TileXRMoonEp::kPlannerReadyEventId + (phase - 1) * rankSize + sourceRank;
}

__aicore__ inline __gm__ uint64_t *BarrierSlotAddress(GM_ADDR ownerPeerMem,
    int32_t phase, int32_t sourceRank, int32_t rankSize)
{
    const uint64_t slotIndex = static_cast<uint64_t>((phase - 1) * rankSize + sourceRank);
    return reinterpret_cast<__gm__ uint64_t *>(ownerPeerMem + kPlannerBarrierBaseBytes +
        slotIndex * kPlannerBarrierStrideBytes);
}

__aicore__ inline void ReloadPeerMems(GM_ADDR commArgsGM, GM_ADDR *peerMems, int32_t rankSize)
{
    auto commArgs = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
    AscendC::GlobalTensor<GM_ADDR> peerMemTable;
    peerMemTable.SetGlobalBuffer(&(commArgs->peerMems[0]), TileXR::TILEXR_MAX_RANK_SIZE);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        peerMems[peer] = peerMemTable.GetValue(peer);
    }
}

__aicore__ inline uint64_t LoadBarrierMte(__gm__ uint64_t *barrier,
    AscendC::LocalTensor<uint8_t> barrierRelay)
{
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(barrier), kPlannerBarrierStrideBytes);
    AscendC::GlobalTensor<uint64_t> barrierGlobal;
    barrierGlobal.SetGlobalBuffer(barrier, kPlannerBarrierWords);
    auto barrierLocal = barrierRelay.ReinterpretCast<uint64_t>();
    AscendC::DataCopy(barrierLocal, barrierGlobal, kPlannerBarrierWords);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return barrierLocal.GetValue(0);
}

__aicore__ inline void PublishBarrierFlags(GM_ADDR *peerMems, int32_t sourceRank,
    int32_t rankSize, int32_t magic, int32_t phase,
    AscendC::LocalTensor<uint8_t> barrierRelay, BarrierDebug *debug)
{
    const uint64_t barrierValue =
        (static_cast<uint64_t>(static_cast<uint32_t>(magic)) << MAGIC_OFFSET) |
        static_cast<uint64_t>(static_cast<uint32_t>(phase));
    auto barrierLocal = barrierRelay.ReinterpretCast<uint64_t>();
    barrierLocal.SetValue(0, barrierValue);
    const uint64_t slotIndex =
        static_cast<uint64_t>((phase - 1) * rankSize + sourceRank);
    const uint64_t barrierByteOffset =
        kPlannerBarrierBaseBytes + slotIndex * kPlannerBarrierStrideBytes;

    for (int32_t offset = 0; offset < rankSize; ++offset) {
        const int32_t targetRank = (sourceRank + offset) % rankSize;
        __gm__ uint64_t *remoteBarrier = reinterpret_cast<__gm__ uint64_t *>(
            peerMems[targetRank] + barrierByteOffset);
        if (debug != nullptr) {
            if (targetRank == sourceRank) {
                debug->publishSelfAddress = reinterpret_cast<uint64_t>(remoteBarrier);
            } else if (debug->publishRemoteAddress == 0) {
                debug->publishRemoteAddress = reinterpret_cast<uint64_t>(remoteBarrier);
            }
        }
        AscendC::GlobalTensor<uint64_t> remoteBarrierGlobal;
        remoteBarrierGlobal.SetGlobalBuffer(remoteBarrier, kPlannerBarrierWords);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopy(remoteBarrierGlobal, barrierLocal, kPlannerBarrierWords);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
}

__aicore__ inline bool WaitPhase(GM_ADDR localPeerAddress, int32_t sourceRank,
    int32_t rankSize, int32_t magic, int32_t phase, uint64_t waitIterations,
    AscendC::LocalTensor<uint8_t> barrierRelay, int64_t *lastMteValue,
    int64_t *lastScalarValue)
{
    __gm__ uint64_t *ready =
        BarrierSlotAddress(localPeerAddress, phase, sourceRank, rankSize);
    const uint64_t expected =
        (static_cast<uint64_t>(static_cast<uint32_t>(magic)) << MAGIC_OFFSET) |
        static_cast<uint64_t>(static_cast<uint32_t>(phase));
    uint64_t value = 0;
    for (uint64_t iteration = 0; iteration < waitIterations; ++iteration) {
        value = LoadBarrierMte(ready, barrierRelay);
        if ((value & static_cast<uint64_t>(MAGIC_MASK)) ==
                (expected & static_cast<uint64_t>(MAGIC_MASK)) &&
            value >= expected) {
            if (lastMteValue != nullptr) *lastMteValue = static_cast<int64_t>(value);
            if (lastScalarValue != nullptr) *lastScalarValue = static_cast<int64_t>(*ready);
            return true;
        }
    }
    if (lastMteValue != nullptr) *lastMteValue = static_cast<int64_t>(value);
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(ready), kPlannerBarrierStrideBytes);
    if (lastScalarValue != nullptr) *lastScalarValue = static_cast<int64_t>(*ready);
    return false;
}

__aicore__ inline bool CollectiveBarrier(GM_ADDR *peerMems, int32_t rank,
    int32_t rankSize, int32_t magic, int32_t phase, uint64_t waitIterations,
    AscendC::LocalTensor<uint8_t> barrierRelay, int32_t *timedOutPeer,
    BarrierDebug *debug)
{
    PublishBarrierFlags(peerMems, rank, rankSize, magic, phase, barrierRelay, debug);
    GM_ADDR localPeerMem = peerMems[rank];
    if (debug != nullptr) {
        __gm__ uint64_t *selfBarrier =
            BarrierSlotAddress(localPeerMem, phase, rank, rankSize);
        debug->mteValue = static_cast<int64_t>(LoadBarrierMte(selfBarrier, barrierRelay));
        debug->scalarValue = static_cast<int64_t>(*selfBarrier);
        debug->peerAddress = reinterpret_cast<GM_ADDR>(selfBarrier);
        debug->peer = rank;
        debug->phase = phase;
    }
    for (int32_t offset = 0; offset < rankSize; ++offset) {
        const int32_t sourceRank = (rank + offset) % rankSize;
        int64_t lastMteValue = 0;
        int64_t lastScalarValue = 0;
        if (!WaitPhase(localPeerMem, sourceRank, rankSize, magic, phase,
                waitIterations, barrierRelay, &lastMteValue, &lastScalarValue)) {
            if (timedOutPeer != nullptr) *timedOutPeer = sourceRank;
            if (debug != nullptr) {
                debug->mteValue = lastMteValue;
                debug->scalarValue = lastScalarValue;
                debug->peerAddress = reinterpret_cast<GM_ADDR>(
                    BarrierSlotAddress(localPeerMem, phase, sourceRank, rankSize));
                debug->peer = sourceRank;
                debug->phase = phase;
            }
            return false;
        }
    }
    return true;
}

__aicore__ inline GM_ADDR PeerMailboxRow(GM_ADDR ownerPeerMem,
    const PlanPeerMailboxLayout &layout, int64_t sourceRank)
{
    return ownerPeerMem + TileXR::IPC_DATA_OFFSET +
        TileXREp::Plan::PlanPeerMailboxRowOffset(layout, sourceRank);
}

__aicore__ inline void PushPeerMailboxRowMte(GM_ADDR remoteRow, GM_ADDR localRow,
    uint64_t rowBytes, AscendC::LocalTensor<uint8_t> relay)
{
    constexpr uint64_t transferBytes = TileXREp::Plan::kPlanPeerMailboxTransferBytes;
    constexpr uint32_t transferWords = transferBytes / sizeof(uint64_t);
    const uint64_t rowWords = rowBytes / sizeof(uint64_t);
    AscendC::GlobalTensor<uint64_t> localGlobal;
    AscendC::GlobalTensor<uint64_t> remoteGlobal;
    localGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(localRow), rowWords);
    remoteGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(remoteRow), rowWords);
    auto localRelay = relay.ReinterpretCast<uint64_t>();

    for (uint64_t wordOffset = 0; wordOffset < rowWords; wordOffset += transferWords) {
        AscendC::DataCopy(localRelay, localGlobal[wordOffset], transferWords);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopy(remoteGlobal[wordOffset], localRelay, transferWords);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void PushPeerMailboxToTargets(GM_ADDR *peerMems, int64_t sourceRank,
    int64_t rankSize, const PlanPeerMailboxLayout &layout,
    uint64_t bytes, AscendC::LocalTensor<uint8_t> relay)
{
    GM_ADDR localRow = PeerMailboxRow(peerMems[sourceRank], layout, sourceRank);
    RefreshCacheLines(localRow, bytes);
    for (int64_t offset = 1; offset < rankSize; ++offset) {
        const int64_t targetRank = (sourceRank + offset) % rankSize;
        GM_ADDR remoteRow = PeerMailboxRow(peerMems[targetRank], layout, sourceRank);
        PushPeerMailboxRowMte(remoteRow, localRow, bytes, relay);
    }
}

__aicore__ inline void PublishInputs(GM_ADDR *peerMems,
    const PlanPeerMailboxLayout &layout, const __gm__ int32_t *tokensPerExpert,
    const __gm__ int32_t *globalRankIds, int64_t rank, int64_t expertNum,
    int64_t rankSize, int64_t s, int64_t topK, int64_t prefetchSlots,
    int64_t rankTokenCapacity, int64_t nvS, int64_t tokenPadding,
    int64_t tokenRouteLimitPerPair, int32_t cardsPerServer, int32_t cardsPerCabinet,
    int32_t crossCandidateCount, uint64_t epoch, uint64_t topologyHash,
    AscendC::LocalTensor<uint8_t> relay)
{
    GM_ADDR row = PeerMailboxRow(peerMems[rank], layout, rank);
    auto header = reinterpret_cast<__gm__ PlanCallHeader *>(row + layout.header);
    FillHeader(header, rankSize, s, topK, expertNum, prefetchSlots, rankTokenCapacity,
        nvS, tokenPadding, tokenRouteLimitPerPair, cardsPerServer, cardsPerCabinet,
        crossCandidateCount, epoch, topologyHash);
    auto publishedTpe = reinterpret_cast<__gm__ int32_t *>(row + layout.tpe);
    for (int64_t expert = 0; expert < expertNum; ++expert) {
        publishedTpe[expert] = tokensPerExpert[expert];
    }
    *reinterpret_cast<__gm__ int32_t *>(row + layout.globalRankId) = globalRankIds[rank];
    RefreshCacheLines(row, layout.inputBytes);
    PushPeerMailboxToTargets(peerMems, rank, rankSize, layout, layout.inputBytes, relay);
}

__aicore__ inline void GatherInputs(GM_ADDR localOwnerPeerMem,
    const PlanPeerMailboxLayout &layout, GM_ADDR meta, const MetaOffsets &metaOffsets,
    int64_t rankSize, int64_t expertNum)
{
    auto gatheredTpe = reinterpret_cast<__gm__ int32_t *>(meta + metaOffsets.tpe);
    auto gatheredIds = reinterpret_cast<__gm__ int32_t *>(meta + metaOffsets.globalRankIds);
    for (int64_t sourceRank = 0; sourceRank < rankSize; ++sourceRank) {
        GM_ADDR row = PeerMailboxRow(localOwnerPeerMem, layout, sourceRank);
        CopyWords(reinterpret_cast<__gm__ int32_t *>(HeaderAt(meta, metaOffsets, sourceRank)),
            reinterpret_cast<__gm__ int32_t *>(row + layout.header),
            sizeof(PlanCallHeader) / sizeof(int32_t));
        CopyWords(gatheredTpe + sourceRank * expertNum,
            reinterpret_cast<__gm__ int32_t *>(row + layout.tpe), expertNum);
        CopyWords(gatheredIds + sourceRank,
            reinterpret_cast<__gm__ int32_t *>(row + layout.globalRankId), 1);
    }
}

__aicore__ inline void PublishStatus(GM_ADDR *peerMems,
    const PlanPeerMailboxLayout &layout, int64_t rank, int64_t rankSize,
    const __gm__ int32_t *status, AscendC::LocalTensor<uint8_t> relay)
{
    GM_ADDR row = PeerMailboxRow(peerMems[rank], layout, rank);
    CopyWords(reinterpret_cast<__gm__ int32_t *>(row + layout.status),
        status, TileXREp::Plan::kPlanStatusWords);
    PushPeerMailboxToTargets(peerMems, rank, rankSize, layout,
        TileXREp::Plan::kPlanPeerMailboxTransferBytes, relay);
}

__aicore__ inline void GatherStatuses(GM_ADDR localOwnerPeerMem,
    const PlanPeerMailboxLayout &layout, GM_ADDR meta, const MetaOffsets &metaOffsets,
    int64_t rankSize)
{
    for (int64_t sourceRank = 0; sourceRank < rankSize; ++sourceRank) {
        auto destination = reinterpret_cast<__gm__ int32_t *>(
            meta + metaOffsets.localStatus + static_cast<uint64_t>(sourceRank) *
            TileXREp::Plan::kPlanStatusStrideBytes);
        GM_ADDR row = PeerMailboxRow(localOwnerPeerMem, layout, sourceRank);
        auto source = reinterpret_cast<__gm__ int32_t *>(row + layout.status);
        CopyWords(destination, source, TileXREp::Plan::kPlanStatusWords);
    }
}

__aicore__ inline int32_t ReduceGlobalPlanStatus(GM_ADDR meta,
    const MetaOffsets &offsets, int64_t rankSize)
{
    int32_t result = PLAN_OK;
    for (int64_t rank = 0; rank < rankSize; ++rank) {
        auto row = reinterpret_cast<__gm__ int32_t *>(StatusRow(meta, offsets, rank));
        const int32_t candidate = row[0];
        if ((candidate >= PLAN_ERROR_CONFIG_MISMATCH &&
             (result < PLAN_ERROR_CONFIG_MISMATCH || candidate > result)) ||
            (candidate > result && result < PLAN_ERROR_CONFIG_MISMATCH)) result = candidate;
    }
    return result;
}

__aicore__ inline PlanAlgorithmWorkspace BindAlgorithmWorkspace(GM_ADDR localWorkspace,
    GM_ADDR meta, const MetaOffsets &metaOffsets, const LocalOffsets &localOffsets,
    int64_t rankSize, int64_t s, int64_t topK, int64_t tokenRouteLimitPerPair,
    bool affinityOrderValid)
{
    PlanAlgorithmWorkspace workspace {};
    workspace.expertCount = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.expertCount);
    workspace.rankLoad = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.rankLoad);
    workspace.remainingTpe = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.remainingTpe);
    workspace.alloc = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.alloc);
    workspace.remoteExpertSet = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.remoteExpertSet);
    workspace.srcExpertCursor = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.srcExpertCursor);
    workspace.dstExpertCursor = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.dstExpertCursor);
    workspace.expertPhysicalBase = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.expertPhysicalBase);
    workspace.localExpertOrdinal = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.localExpertOrdinal);
    workspace.tokenSegments = reinterpret_cast<__gm__ TokenSegmentMove *>(localWorkspace + localOffsets.tokenSegments);
    workspace.tokenSegmentCapacity = static_cast<int32_t>(s * topK);
    workspace.routedPairTokens = tokenRouteLimitPerPair == 0 ? nullptr :
        reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.routedPairTokens);
    workspace.scratch = reinterpret_cast<__gm__ int32_t *>(localWorkspace + localOffsets.scratch);
    workspace.scratchCount = static_cast<int32_t>(rankSize * 16);
    workspace.affinityOrder = reinterpret_cast<__gm__ int32_t *>(meta + metaOffsets.affinityOrder);
    workspace.affinityOrderValid = affinityOrderValid;
    return workspace;
}

__aicore__ inline void ClearStatus(__gm__ int32_t *status, int32_t value)
{
    for (int32_t word = 0; word < TileXREp::Plan::kPlanStatusWords; ++word) status[word] = 0;
    status[0] = value;
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(status),
        TileXREp::Plan::kPlanStatusWords * sizeof(int32_t));
}

__aicore__ inline void WriteBarrierDebugStatus(__gm__ int32_t *status, int32_t value,
    const BarrierDebug &debug)
{
    ClearStatus(status, value);
    status[1] = kBarrierDebugBuildTag;
    status[2] = debug.phase;
    status[3] = debug.peer;
    status[4] = static_cast<int32_t>(debug.publishSelfAddress);
    status[5] = static_cast<int32_t>(debug.publishSelfAddress >> 32);
    status[6] = static_cast<int32_t>(debug.publishRemoteAddress);
    status[7] = static_cast<int32_t>(debug.publishRemoteAddress >> 32);
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(status),
        TileXREp::Plan::kPlanStatusWords * sizeof(int32_t));
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_ep_plan_kernel(GM_ADDR commArgsGM,
    GM_ADDR topkExpertsGM, GM_ADDR tokensPerExpertGM, GM_ADDR globalRankIdsGM, GM_ADDR dstGM,
    GM_ADDR cuSeqlensGM, GM_ADDR expertsToCopyGM, GM_ADDR remoteExpertsGM,
    GM_ADDR expertTargetsGM, GM_ADDR remoteStatsGM, GM_ADDR statusGM,
    GM_ADDR localWorkspaceGM, GM_ADDR metaWorkspaceGM, int64_t rank, int64_t rankSize,
    int64_t s, int64_t topK, int64_t expertNum, int64_t prefetchSlots,
    int64_t rankTokenCapacity, int64_t nvS, int64_t tokenPadding, int64_t tokenRouteLimitPerPair,
    int32_t cardsPerServer, int32_t cardsPerCabinet, int32_t crossCandidateCount,
    uint64_t epoch, uint64_t waitIterations, int64_t magic)
{
    if (AscendC::GetBlockIdx() != 0) return;
    auto commArgs = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
    auto topkExperts = reinterpret_cast<__gm__ int32_t *>(topkExpertsGM);
    auto tokensPerExpert = reinterpret_cast<__gm__ int32_t *>(tokensPerExpertGM);
    auto globalRankIds = reinterpret_cast<__gm__ int32_t *>(globalRankIdsGM);
    auto status = reinterpret_cast<__gm__ int32_t *>(statusGM);
    const MetaOffsets metaOffsets = BuildMetaOffsets(rankSize, expertNum);
    const PlanPeerMailboxLayout peerMailbox =
        TileXREp::Plan::BuildPlanPeerMailboxLayout(rankSize, expertNum);
    const LocalOffsets localOffsets = BuildLocalOffsets(rankSize, s, topK, expertNum,
        prefetchSlots, tokenRouteLimitPerPair);

    GM_ADDR peerMems[TileXR::TILEXR_MAX_RANK_SIZE];
    ReloadPeerMems(commArgsGM, peerMems, static_cast<int32_t>(rankSize));

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> syncBuffer;
    pipe.InitBuffer(syncBuffer, TileXRMoonEp::kPlannerSyncUbBytes);
    AscendC::LocalTensor<uint8_t> barrierRelay = syncBuffer.Get<uint8_t>();

    const uint64_t topologyHash = BuildTopologyHash(globalRankIds, rankSize);
    auto epochState = reinterpret_cast<__gm__ PlanEpochState *>(metaWorkspaceGM + metaOffsets.epochState);
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(epochState), sizeof(PlanEpochState));
    const bool affinityOrderValid =
        (epochState->reserved & TileXREp::Plan::kPlanAffinityCacheValid) != 0 &&
        epochState->topologyHash == topologyHash;
    epochState->requestedEpoch = epoch;
    epochState->committedEpoch = 0;
    RefreshCacheLines(reinterpret_cast<GM_ADDR>(epochState), sizeof(PlanEpochState));

    PublishInputs(peerMems, peerMailbox, tokensPerExpert, globalRankIds, rank,
        expertNum, rankSize, s, topK, prefetchSlots, rankTokenCapacity, nvS, tokenPadding,
        tokenRouteLimitPerPair, cardsPerServer, cardsPerCabinet, crossCandidateCount,
        epoch, topologyHash, barrierRelay);

    int32_t timedOutPeer = -1;
    int32_t localStatus = PLAN_OK;
    BarrierDebug barrierDebug {};
    if (!CollectiveBarrier(peerMems, static_cast<int32_t>(rank),
            static_cast<int32_t>(rankSize), static_cast<int32_t>(magic), kPhaseData,
            waitIterations, barrierRelay, &timedOutPeer, &barrierDebug)) {
        localStatus = TileXRMoonEp::kPlannerStatusTimeoutBase + timedOutPeer;
    } else {
        GatherInputs(peerMems[rank], peerMailbox, metaWorkspaceGM, metaOffsets,
            rankSize, expertNum);
        if (!HeadersMatch(metaWorkspaceGM, metaOffsets, rankSize)) {
            localStatus = PLAN_ERROR_CONFIG_MISMATCH;
        }
    }

    if (localStatus == PLAN_OK) {
        PlanAlgorithmInput input {};
        input.rank = static_cast<int32_t>(rank);
        input.rankSize = static_cast<int32_t>(rankSize);
        input.s = s;
        input.topK = topK;
        input.expertNum = expertNum;
        input.config.prefetchSlots = prefetchSlots;
        input.config.rankTokenCapacity = rankTokenCapacity;
        input.config.nvS = nvS;
        input.config.tokenPadding = tokenPadding;
        input.config.tokenRouteLimitPerPair = tokenRouteLimitPerPair;
        input.config.cardsPerServer = cardsPerServer;
        input.config.cardsPerCabinet = cardsPerCabinet;
        input.config.crossCandidateCount = crossCandidateCount;
        input.config.reserved = 0;
        input.topkExperts = topkExperts;
        input.tokensPerExpert = reinterpret_cast<__gm__ int32_t *>(metaWorkspaceGM + metaOffsets.tpe);
        input.globalRankIds = reinterpret_cast<__gm__ int32_t *>(metaWorkspaceGM + metaOffsets.globalRankIds);

        PlanAlgorithmOutput output {};
        output.dst = reinterpret_cast<__gm__ int32_t *>(dstGM);
        output.cuSeqlens = reinterpret_cast<__gm__ int32_t *>(cuSeqlensGM);
        auto remoteExperts = reinterpret_cast<__gm__ int32_t *>(remoteExpertsGM);
        output.expertsToCopy = remoteExperts != nullptr
            ? remoteExperts + static_cast<int64_t>(rank) * prefetchSlots
            : reinterpret_cast<__gm__ int32_t *>(expertsToCopyGM);
        output.remoteExperts = remoteExperts;
        output.expertTargets = reinterpret_cast<__gm__ uint64_t *>(expertTargetsGM);
        output.remoteStats = reinterpret_cast<__gm__ int32_t *>(remoteStatsGM);
        output.status = status;
        PlanAlgorithmWorkspace workspace = BindAlgorithmWorkspace(localWorkspaceGM, metaWorkspaceGM,
            metaOffsets, localOffsets, rankSize, s, topK, tokenRouteLimitPerPair,
            affinityOrderValid);
        localStatus = static_cast<int32_t>(TileXREp::Plan::RunPlanAlgorithm(input, output, workspace));
        if (workspace.affinityOrderValid && !affinityOrderValid) {
            epochState->topologyHash = topologyHash;
            epochState->reserved = TileXREp::Plan::kPlanAffinityCacheValid;
            RefreshCacheLines(reinterpret_cast<GM_ADDR>(epochState), sizeof(PlanEpochState));
        }
    } else {
        WriteBarrierDebugStatus(status, localStatus, barrierDebug);
    }

    ReloadPeerMems(commArgsGM, peerMems, static_cast<int32_t>(rankSize));
    PublishStatus(peerMems, peerMailbox, rank, rankSize, status, barrierRelay);
    timedOutPeer = -1;
    BarrierDebug statusBarrierDebug {};
    if (CollectiveBarrier(peerMems, static_cast<int32_t>(rank),
            static_cast<int32_t>(rankSize), static_cast<int32_t>(magic), kPhaseStatus,
            waitIterations, barrierRelay, &timedOutPeer, &statusBarrierDebug)) {
        GatherStatuses(peerMems[rank], peerMailbox, metaWorkspaceGM, metaOffsets, rankSize);
        status[0] = ReduceGlobalPlanStatus(metaWorkspaceGM, metaOffsets, rankSize);
        RefreshCacheLines(statusGM, sizeof(int32_t));
    } else {
        WriteBarrierDebugStatus(status,
            TileXRMoonEp::kPlannerStatusTimeoutBase + timedOutPeer, statusBarrierDebug);
    }

    ReloadPeerMems(commArgsGM, peerMems, static_cast<int32_t>(rankSize));
    timedOutPeer = -1;
    BarrierDebug readyBarrierDebug {};
    if (!CollectiveBarrier(peerMems, static_cast<int32_t>(rank),
            static_cast<int32_t>(rankSize), static_cast<int32_t>(magic), kPhaseReady,
            waitIterations, barrierRelay, &timedOutPeer, &readyBarrierDebug)) {
        if (status[1] != kBarrierDebugBuildTag) {
            WriteBarrierDebugStatus(status,
                TileXRMoonEp::kPlannerStatusTimeoutBase + timedOutPeer, readyBarrierDebug);
        }
    }

    // Re-publish the ready outcome before reducing one final time. A rank that
    // timed out in the ready phase therefore prevents peers from committing a
    // success epoch based only on its earlier ready flag.
    ReloadPeerMems(commArgsGM, peerMems, static_cast<int32_t>(rankSize));
    PublishStatus(peerMems, peerMailbox, rank, rankSize, status, barrierRelay);
    timedOutPeer = -1;
    BarrierDebug consensusBarrierDebug {};
    if (CollectiveBarrier(peerMems, static_cast<int32_t>(rank),
            static_cast<int32_t>(rankSize), static_cast<int32_t>(magic), kPhaseConsensus,
            waitIterations, barrierRelay, &timedOutPeer, &consensusBarrierDebug)) {
        GatherStatuses(peerMems[rank], peerMailbox, metaWorkspaceGM, metaOffsets, rankSize);
        status[0] = ReduceGlobalPlanStatus(metaWorkspaceGM, metaOffsets, rankSize);
        RefreshCacheLines(statusGM, sizeof(int32_t));
    } else if (status[1] != kBarrierDebugBuildTag) {
        WriteBarrierDebugStatus(status,
            TileXRMoonEp::kPlannerStatusTimeoutBase + timedOutPeer, consensusBarrierDebug);
    }
    if (status[0] == PLAN_OK) {
        epochState->requestedEpoch = epoch;
        epochState->committedEpoch = epoch;
        RefreshCacheLines(reinterpret_cast<GM_ADDR>(epochState), sizeof(PlanEpochState));
    }
}
