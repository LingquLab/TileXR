#include "kernel_operator.h"

#include <cstdint>

#include "comm_args.h"
#include "planner_common.h"
#include "tilexr_sync.h"

namespace TileXRMoonEp {
namespace Kernel {

constexpr int64_t kArrayCapacity = kPlannerMaxExpertCount;
constexpr int64_t kArray0OffsetBytes = 0;
constexpr int64_t kArray1OffsetBytes = kArray0OffsetBytes + kArrayCapacity * sizeof(int32_t);
constexpr int64_t kArray2OffsetBytes = kArray1OffsetBytes + kArrayCapacity * sizeof(int32_t);
constexpr int64_t kRouteOffsetBytes = kArray2OffsetBytes + kArrayCapacity * sizeof(int32_t);
constexpr int64_t kDstOffsetBytes = kRouteOffsetBytes + kPlannerRouteTileInts * sizeof(int32_t);
constexpr int64_t kRequiredWorkBytes = kDstOffsetBytes + kPlannerRouteTileInts * sizeof(int32_t);
static_assert(kRequiredWorkBytes <= kPlannerWorkUbBytes, "planner UB layout exceeds work buffer");

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

__aicore__ inline void CopyGmToUb(AscendC::LocalTensor<int32_t> dst,
    const AscendC::GlobalTensor<int32_t> &src, int64_t count)
{
    if (count <= 0) {
        return;
    }
    AscendC::DataCopyExtParams params {
        1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(int32_t))), 0, 0, 0
    };
    AscendC::DataCopyPadExtParams<int32_t> pad {false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
}

__aicore__ inline void CopyUbToGm(AscendC::GlobalTensor<int32_t> dst,
    const AscendC::LocalTensor<int32_t> &src, int64_t count)
{
    if (count <= 0) {
        return;
    }
    AscendC::DataCopyExtParams params {
        1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(int32_t))), 0, 0, 0
    };
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(dst, src, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void WaitVectorForMte3()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
}

class PlannerKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR topkExpertIds,
        GM_ADDR tokensPerExpert, GM_ADDR workspace, GM_ADDR dst, GM_ADDR cuSeqlens,
        GM_ADDR expertsToCopy, GM_ADDR remoteStats, GM_ADDR plannerStatus,
        int64_t s, int64_t k, int64_t expertCount, int64_t expertsPerRank,
        int64_t routeCount, int64_t dispatchedCapacity, uint64_t waitIterations,
        uint64_t tpePrefixOffset, uint64_t blockHistogramOffset,
        uint64_t allocPrefixOffset, uint64_t expertOffsetsOffset, uint64_t zOffset,
        uint64_t groupTotalsOffset, int64_t magic)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
        rank_ = args_->rank;
        rankSize_ = args_->rankSize;
        s_ = s;
        k_ = k;
        expertCount_ = expertCount;
        expertsPerRank_ = expertsPerRank;
        routeCount_ = routeCount;
        dispatchedCapacity_ = dispatchedCapacity;
        waitIterations_ = waitIterations;
        magic_ = magic;
        const int64_t subBlockCount = static_cast<int64_t>(get_subblockdim());
        blockIdx_ = static_cast<int64_t>(get_block_idx()) * subBlockCount +
            static_cast<int64_t>(get_subblockid());
        blockCount_ = static_cast<int64_t>(get_block_num()) * subBlockCount;

        topkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(topkExpertIds), routeCount_);
        localTpeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokensPerExpert), expertCount_);
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dst), routeCount_);
        cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cuSeqlens),
            expertCount_ + expertsPerRank_);
        expertsToCopyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertsToCopy),
            rankSize_ * expertsPerRank_);
        remoteStatsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(remoteStats), 2);
        plannerStatusGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(plannerStatus), 1);

        GM_ADDR workBase = workspace;
        tpePrefixGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(workBase + tpePrefixOffset), rankSize_ * expertCount_);
        blockHistogramGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(workBase + blockHistogramOffset), blockCount_ * expertCount_);
        allocPrefixGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(workBase + allocPrefixOffset), rankSize_ * expertCount_);
        expertOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(workBase + expertOffsetsOffset), rankSize_ * expertCount_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workBase + zOffset), rankSize_ * rankSize_);
        groupTotalsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(workBase + groupTotalsOffset), rankSize_);

        pipe_.InitBuffer(syncBuf_, kPlannerSyncUbBytes);
        pipe_.InitBuffer(workBuf_, kPlannerWorkUbBytes);
        for (int32_t peer = 0; peer < rankSize_; ++peer) {
            shareAddrs_[peer] = args_->peerMems[peer];
        }
        sync_.Init(rank_, rankSize_, shareAddrs_, syncBuf_);
    }

    __aicore__ inline void Process()
    {
        InitializeStatus();
        AscendC::SyncAll<true>();

        PublishTpe();
        BuildLocalHistogram();
        AscendC::SyncAll<true>();

        CrossRankReady();
        AscendC::SyncAll<true>();
        if (!PlannerSucceeded()) {
            return;
        }

        GatherTpe();
        AscendC::SyncAll<true>();

        PrefixTpe();
        AscendC::SyncAll<true>();
        BuildBalanceQuotas();
        AscendC::SyncAll<true>();
        BuildAllocation();
        AscendC::SyncAll<true>();

        BuildExpertLayout();
        AscendC::SyncAll<true>();
        FinalizeRemoteStats();
        BuildPrefixes();
        AscendC::SyncAll<true>();

        BuildDst();
    }

private:
    __aicore__ inline AscendC::LocalTensor<int32_t> Array0()
    {
        return workBuf_.GetWithOffset<int32_t>(kArrayCapacity, kArray0OffsetBytes);
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> Array1()
    {
        return workBuf_.GetWithOffset<int32_t>(kArrayCapacity, kArray1OffsetBytes);
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> Array2()
    {
        return workBuf_.GetWithOffset<int32_t>(kArrayCapacity, kArray2OffsetBytes);
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> RouteTile()
    {
        return workBuf_.GetWithOffset<int32_t>(kPlannerRouteTileInts, kRouteOffsetBytes);
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> DstTile()
    {
        return workBuf_.GetWithOffset<int32_t>(kPlannerRouteTileInts, kDstOffsetBytes);
    }

    __aicore__ inline void TokenRange(int64_t *begin, int64_t *end) const
    {
        *begin = (s_ * blockIdx_) / blockCount_;
        *end = (s_ * (blockIdx_ + 1)) / blockCount_;
    }

    __aicore__ inline void InitializeStatus()
    {
        if (blockIdx_ != 0) {
            return;
        }
        AscendC::LocalTensor<int32_t> status = Array0();
        status.SetValue(0, kPlannerStatusSuccess);
        CopyUbToGm(plannerStatusGm_, status, 1);
    }

    __aicore__ inline bool PlannerSucceeded()
    {
        AscendC::LocalTensor<int32_t> status = Array0();
        CopyGmToUb(status, plannerStatusGm_, 1);
        return status.GetValue(0) == kPlannerStatusSuccess;
    }

    __aicore__ inline void PublishTpe()
    {
        if (blockIdx_ != 0) {
            return;
        }
        AscendC::GlobalTensor<int32_t> peerTpe;
        peerTpe.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            shareAddrs_[rank_] + TileXR::IPC_DATA_OFFSET), expertCount_);
        AscendC::LocalTensor<int32_t> local = Array0();
        CopyGmToUb(local, localTpeGm_, expertCount_);
        CopyUbToGm(peerTpe, local, expertCount_);
    }

    __aicore__ inline void BuildLocalHistogram()
    {
        AscendC::LocalTensor<int32_t> histogram = Array0();
        AscendC::LocalTensor<int32_t> routes = RouteTile();
        for (int64_t e = 0; e < expertCount_; ++e) {
            histogram.SetValue(e, 0);
        }

        int64_t tokenBegin = 0;
        int64_t tokenEnd = 0;
        TokenRange(&tokenBegin, &tokenEnd);
        const int64_t tokensPerTile = kPlannerRouteTileInts / k_;
        for (int64_t tileBegin = tokenBegin; tileBegin < tokenEnd; tileBegin += tokensPerTile) {
            const int64_t tileEnd = MinInt64(tileBegin + tokensPerTile, tokenEnd);
            const int64_t tileRoutes = (tileEnd - tileBegin) * k_;
            CopyGmToUb(routes, topkGm_[tileBegin * k_], tileRoutes);
            for (int64_t i = 0; i < tileRoutes; ++i) {
                const int32_t expert = routes.GetValue(i);
                histogram.SetValue(expert, histogram.GetValue(expert) + 1);
            }
        }
        CopyUbToGm(blockHistogramGm_[blockIdx_ * expertCount_], histogram, expertCount_);
    }

    __aicore__ inline bool WaitReadyFlag(int32_t peer)
    {
        AscendC::GlobalTensor<int64_t> readyGm;
        readyGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(shareAddrs_[peer]) +
                static_cast<int64_t>(kPlannerReadyEventId) * FLAG_UNIT_INT_NUM,
            FLAG_UNIT_INT_NUM);
        const int64_t expected =
            (static_cast<int64_t>(static_cast<int32_t>(magic_)) << MAGIC_OFFSET) |
            static_cast<int64_t>(kPlannerReadyStep);
        for (uint64_t iteration = 0; iteration < waitIterations_; ++iteration) {
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(readyGm);
            const int64_t value = readyGm.GetValue(0);
            if ((value & MAGIC_MASK) == (expected & MAGIC_MASK) && value >= expected) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void CrossRankReady()
    {
        if (blockIdx_ != 0) {
            return;
        }
        sync_.SetSyncFlag(static_cast<int32_t>(magic_), kPlannerReadyStep, kPlannerReadyEventId);
        for (int32_t peerOffset = 0; peerOffset < rankSize_; ++peerOffset) {
            const int32_t peer = (rank_ + peerOffset) % rankSize_;
            if (!WaitReadyFlag(peer)) {
                AscendC::LocalTensor<int32_t> status = Array0();
                status.SetValue(0, kPlannerStatusTimeoutBase + peer);
                CopyUbToGm(plannerStatusGm_, status, 1);
                return;
            }
        }
    }

    __aicore__ inline void GatherTpe()
    {
        if (blockIdx_ >= rankSize_) {
            return;
        }
        const int32_t sourceRank = static_cast<int32_t>(blockIdx_);
        AscendC::GlobalTensor<int32_t> peerTpe;
        peerTpe.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            shareAddrs_[sourceRank] + TileXR::IPC_DATA_OFFSET), expertCount_);
        AscendC::LocalTensor<int32_t> local = Array0();
        CopyGmToUb(local, peerTpe, expertCount_);
        CopyUbToGm(tpePrefixGm_[sourceRank * expertCount_], local, expertCount_);
    }

    __aicore__ inline void PrefixTpe()
    {
        if (blockIdx_ != 0) {
            return;
        }
        AscendC::LocalTensor<int32_t> cumulative = Array0();
        AscendC::LocalTensor<int32_t> row = Array1();
        AscendC::Duplicate(cumulative, static_cast<int32_t>(0), expertCount_);
        for (int32_t sourceRank = 0; sourceRank < rankSize_; ++sourceRank) {
            const int64_t rowOffset = static_cast<int64_t>(sourceRank) * expertCount_;
            CopyGmToUb(row, tpePrefixGm_[rowOffset], expertCount_);
            AscendC::Add(row, row, cumulative, static_cast<int32_t>(expertCount_));
            AscendC::Adds(cumulative, row, static_cast<int32_t>(0),
                static_cast<int32_t>(expertCount_));
            WaitVectorForMte3();
            CopyUbToGm(tpePrefixGm_[rowOffset], row, expertCount_);
        }
    }

    __aicore__ inline void BuildBalanceQuotas()
    {
        if (blockIdx_ != 0) {
            return;
        }
        AscendC::LocalTensor<int32_t> globalCounts = Array0();
        AscendC::LocalTensor<int32_t> balance = Array1();
        AscendC::LocalTensor<int32_t> quotas = RouteTile();
        CopyGmToUb(globalCounts,
            tpePrefixGm_[static_cast<int64_t>(rankSize_ - 1) * expertCount_], expertCount_);
        for (int32_t home = 0; home < rankSize_; ++home) {
            int32_t total = 0;
            const int64_t expertBegin = static_cast<int64_t>(home) * expertsPerRank_;
            for (int64_t localExpert = 0; localExpert < expertsPerRank_; ++localExpert) {
                total += globalCounts.GetValue(expertBegin + localExpert);
            }
            balance.SetValue(home, total - static_cast<int32_t>(routeCount_));
            for (int32_t dest = 0; dest < rankSize_; ++dest) {
                quotas.SetValue(static_cast<int64_t>(home) * rankSize_ + dest, 0);
            }
        }

        while (true) {
            int32_t surplus = -2147483647 - 1;
            int32_t deficit = 2147483647;
            int32_t surplusRank = 0;
            int32_t deficitRank = 0;
            for (int32_t r = 0; r < rankSize_; ++r) {
                const int32_t value = balance.GetValue(r);
                if (value > surplus) {
                    surplus = value;
                    surplusRank = r;
                }
                if (value < deficit) {
                    deficit = value;
                    deficitRank = r;
                }
            }
            if (surplus <= 0 || deficit >= 0) {
                break;
            }
            const int32_t move = -deficit;
            quotas.SetValue(static_cast<int64_t>(surplusRank) * rankSize_ + deficitRank, move);
            balance.SetValue(surplusRank, surplus - move);
            balance.SetValue(deficitRank, 0);
        }
        CopyUbToGm(zGm_, quotas, static_cast<int64_t>(rankSize_) * rankSize_);
    }

    __aicore__ inline void BuildAllocation()
    {
        if (blockIdx_ >= rankSize_) {
            return;
        }
        const int32_t owner = static_cast<int32_t>(blockIdx_);
        const int64_t expertBegin = static_cast<int64_t>(owner) * expertsPerRank_;
        AscendC::LocalTensor<int32_t> remaining = Array0();
        AscendC::LocalTensor<int32_t> quotas = Array1();
        AscendC::LocalTensor<int32_t> heapExperts = Array2();
        AscendC::LocalTensor<int32_t> ownerAllocation = RouteTile();

        CopyGmToUb(quotas, zGm_[static_cast<int64_t>(owner) * rankSize_], rankSize_);
        CopyGmToUb(remaining,
            tpePrefixGm_[static_cast<int64_t>(rankSize_ - 1) * expertCount_ + expertBegin],
            expertsPerRank_);
        AscendC::Duplicate(ownerAllocation, static_cast<int32_t>(0), expertCount_);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        for (int64_t localExpert = 0; localExpert < expertsPerRank_; ++localExpert) {
            heapExperts.SetValue(localExpert, static_cast<int32_t>(localExpert));
        }
        for (int64_t start = (expertsPerRank_ >> 1) - 1; start >= 0; --start) {
            const int32_t currentExpert = heapExperts.GetValue(start);
            const int32_t currentCount = remaining.GetValue(currentExpert);
            int64_t index = start;
            while (true) {
                const int64_t left = index * 2 + 1;
                if (left >= expertsPerRank_) {
                    break;
                }
                const int64_t right = left + 1;
                int64_t child = left;
                int32_t childExpert = heapExperts.GetValue(left);
                int32_t childCount = remaining.GetValue(childExpert);
                if (right < expertsPerRank_) {
                    const int32_t rightExpert = heapExperts.GetValue(right);
                    const int32_t rightCount = remaining.GetValue(rightExpert);
                    if (rightCount > childCount ||
                        (rightCount == childCount && rightExpert < childExpert)) {
                        child = right;
                        childExpert = rightExpert;
                        childCount = rightCount;
                    }
                }
                if (currentCount > childCount ||
                    (currentCount == childCount && currentExpert <= childExpert)) {
                    break;
                }
                heapExperts.SetValue(index, childExpert);
                index = child;
            }
            heapExperts.SetValue(index, currentExpert);
        }

        while (true) {
            int32_t maxQuota = 0;
            int32_t targetRank = 0;
            for (int32_t dest = 0; dest < rankSize_; ++dest) {
                const int32_t quota = quotas.GetValue(dest);
                if (quota > maxQuota) {
                    maxQuota = quota;
                    targetRank = dest;
                }
            }
            if (maxQuota <= 0) {
                break;
            }

            const int32_t selectedLocalExpert = heapExperts.GetValue(0);
            const int32_t maxRemaining = remaining.GetValue(selectedLocalExpert);
            if (maxRemaining <= 0) {
                break;
            }

            const int32_t take = maxRemaining < maxQuota ? maxRemaining : maxQuota;
            const int64_t targetIndex = static_cast<int64_t>(targetRank) * expertsPerRank_ +
                selectedLocalExpert;
            ownerAllocation.SetValue(targetIndex, ownerAllocation.GetValue(targetIndex) + take);
            const int32_t nextRemaining = maxRemaining - take;
            remaining.SetValue(selectedLocalExpert, nextRemaining);
            quotas.SetValue(targetRank, maxQuota - take);

            int64_t index = 0;
            while (true) {
                const int64_t left = index * 2 + 1;
                if (left >= expertsPerRank_) {
                    break;
                }
                const int64_t right = left + 1;
                int64_t child = left;
                int32_t childExpert = heapExperts.GetValue(left);
                int32_t childCount = remaining.GetValue(childExpert);
                if (right < expertsPerRank_) {
                    const int32_t rightExpert = heapExperts.GetValue(right);
                    const int32_t rightCount = remaining.GetValue(rightExpert);
                    if (rightCount > childCount ||
                        (rightCount == childCount && rightExpert < childExpert)) {
                        child = right;
                        childExpert = rightExpert;
                        childCount = rightCount;
                    }
                }
                if (nextRemaining > childCount ||
                    (nextRemaining == childCount && selectedLocalExpert <= childExpert)) {
                    break;
                }
                heapExperts.SetValue(index, childExpert);
                index = child;
            }
            heapExperts.SetValue(index, selectedLocalExpert);
        }

        for (int64_t localExpert = 0; localExpert < expertsPerRank_; ++localExpert) {
            const int64_t homeIndex = static_cast<int64_t>(owner) * expertsPerRank_ + localExpert;
            ownerAllocation.SetValue(
                homeIndex, ownerAllocation.GetValue(homeIndex) + remaining.GetValue(localExpert));
        }
        for (int32_t dest = 0; dest < rankSize_; ++dest) {
            const int64_t ownerRowOffset = static_cast<int64_t>(dest) * expertsPerRank_;
            for (int64_t localExpert = 0; localExpert < expertsPerRank_; ++localExpert) {
                remaining.SetValue(localExpert, ownerAllocation.GetValue(ownerRowOffset + localExpert));
            }
            CopyUbToGm(
                allocPrefixGm_[static_cast<int64_t>(dest) * expertCount_ + expertBegin],
                remaining, expertsPerRank_);
        }
    }

    __aicore__ inline void BuildExpertLayout()
    {
        if (blockIdx_ >= rankSize_) {
            return;
        }
        const int32_t dest = static_cast<int32_t>(blockIdx_);
        AscendC::LocalTensor<int32_t> counts = Array0();
        AscendC::LocalTensor<int32_t> selectedMask = Array1();
        AscendC::LocalTensor<int32_t> selectedExperts = Array2();
        AscendC::LocalTensor<int32_t> layout = RouteTile();
        AscendC::LocalTensor<int32_t> sortedExpertsScratch = DstTile();
        constexpr int64_t kMteAlignmentInts = 32 / sizeof(int32_t);
        const int64_t ownerCopiesOffset = AlignUp(expertCount_, kMteAlignmentInts);
        const int64_t cuSeqlensOffset = AlignUp(ownerCopiesOffset + rankSize_, kMteAlignmentInts);
        const int64_t groupCount = expertCount_ + expertsPerRank_;
        const int64_t statsOffset = AlignUp(cuSeqlensOffset + groupCount, kMteAlignmentInts);

        CopyGmToUb(counts, allocPrefixGm_[static_cast<int64_t>(dest) * expertCount_], expertCount_);
        const int64_t localExpertBegin = static_cast<int64_t>(dest) * expertsPerRank_;
        const int64_t localExpertEnd = localExpertBegin + expertsPerRank_;
        int32_t remoteCount = 0;
        int64_t heapSize = 0;
        for (int64_t slot = 0; slot < expertsPerRank_; ++slot) {
            sortedExpertsScratch.SetValue(slot, -1);
        }
        for (int64_t expert = 0; expert < expertCount_; ++expert) {
            layout.SetValue(expert, 0);
            if (expert >= localExpertBegin && expert < localExpertEnd) {
                continue;
            }
            const int32_t count = counts.GetValue(expert);
            if (count <= 0) {
                continue;
            }
            ++remoteCount;
            if (heapSize < expertsPerRank_) {
                int64_t index = heapSize++;
                selectedExperts.SetValue(index, static_cast<int32_t>(expert));
                selectedMask.SetValue(index, count);
                while (index > 0) {
                    const int64_t parent = (index - 1) >> 1;
                    const int32_t parentCount = selectedMask.GetValue(parent);
                    const int32_t parentExpert = selectedExperts.GetValue(parent);
                    if (count > parentCount || (count == parentCount && expert >= parentExpert)) {
                        break;
                    }
                    selectedMask.SetValue(index, parentCount);
                    selectedExperts.SetValue(index, parentExpert);
                    index = parent;
                }
                selectedMask.SetValue(index, count);
                selectedExperts.SetValue(index, static_cast<int32_t>(expert));
            } else {
                const int32_t rootCount = selectedMask.GetValue(0);
                const int32_t rootExpert = selectedExperts.GetValue(0);
                if (count < rootCount || (count == rootCount && expert <= rootExpert)) {
                    continue;
                }
                int64_t index = 0;
                while (true) {
                    const int64_t left = index * 2 + 1;
                    if (left >= heapSize) {
                        break;
                    }
                    const int64_t right = left + 1;
                    int64_t child = left;
                    int32_t childCount = selectedMask.GetValue(left);
                    int32_t childExpert = selectedExperts.GetValue(left);
                    if (right < heapSize) {
                        const int32_t rightCount = selectedMask.GetValue(right);
                        const int32_t rightExpert = selectedExperts.GetValue(right);
                        if (rightCount < childCount ||
                            (rightCount == childCount && rightExpert < childExpert)) {
                            child = right;
                            childCount = rightCount;
                            childExpert = rightExpert;
                        }
                    }
                    if (count < childCount || (count == childCount && expert <= childExpert)) {
                        break;
                    }
                    selectedMask.SetValue(index, childCount);
                    selectedExperts.SetValue(index, childExpert);
                    index = child;
                }
                selectedMask.SetValue(index, count);
                selectedExperts.SetValue(index, static_cast<int32_t>(expert));
            }
        }

        int64_t selectedCount = heapSize;
        while (heapSize > 0) {
            sortedExpertsScratch.SetValue(heapSize - 1, selectedExperts.GetValue(0));
            --heapSize;
            if (heapSize == 0) {
                break;
            }
            const int32_t lastCount = selectedMask.GetValue(heapSize);
            const int32_t lastExpert = selectedExperts.GetValue(heapSize);
            int64_t index = 0;
            while (true) {
                const int64_t left = index * 2 + 1;
                if (left >= heapSize) {
                    break;
                }
                const int64_t right = left + 1;
                int64_t child = left;
                int32_t childCount = selectedMask.GetValue(left);
                int32_t childExpert = selectedExperts.GetValue(left);
                if (right < heapSize) {
                    const int32_t rightCount = selectedMask.GetValue(right);
                    const int32_t rightExpert = selectedExperts.GetValue(right);
                    if (rightCount < childCount ||
                        (rightCount == childCount && rightExpert < childExpert)) {
                        child = right;
                        childCount = rightCount;
                        childExpert = rightExpert;
                    }
                }
                if (lastCount < childCount ||
                    (lastCount == childCount && lastExpert <= childExpert)) {
                    break;
                }
                selectedMask.SetValue(index, childCount);
                selectedExperts.SetValue(index, childExpert);
                index = child;
            }
            selectedMask.SetValue(index, lastCount);
            selectedExperts.SetValue(index, lastExpert);
        }

        for (int64_t expert = 0; expert < expertCount_; ++expert) {
            selectedMask.SetValue(expert, 0);
        }
        for (int32_t owner = 0; owner < rankSize_; ++owner) {
            layout.SetValue(ownerCopiesOffset + owner, 0);
        }
        for (int64_t slot = 0; slot < expertsPerRank_; ++slot) {
            const int32_t expert = slot < selectedCount ?
                sortedExpertsScratch.GetValue(slot) : -1;
            selectedExperts.SetValue(slot, expert);
            if (expert >= 0) {
                selectedMask.SetValue(expert, 1);
                const int32_t owner = expert / static_cast<int32_t>(expertsPerRank_);
                const int64_t ownerIndex = ownerCopiesOffset + owner;
                layout.SetValue(ownerIndex, layout.GetValue(ownerIndex) + 1);
            }
        }

        int32_t start = 0;
        for (int64_t group = 0; group < groupCount; ++group) {
            int32_t count = 0;
            int32_t expert = -1;
            if (group < expertCount_) {
                if (selectedMask.GetValue(group) == 0) {
                    expert = static_cast<int32_t>(group);
                    count = counts.GetValue(group);
                }
            } else {
                expert = selectedExperts.GetValue(group - expertCount_);
                if (expert >= 0) {
                    count = counts.GetValue(expert);
                }
            }
            if (count > 0) {
                layout.SetValue(expert, start);
            }
            const int32_t end = start + count;
            if (dest == rank_) {
                layout.SetValue(cuSeqlensOffset + group, end);
            }
            start = end;
        }
        layout.SetValue(statsOffset, remoteCount);

        CopyUbToGm(expertOffsetsGm_[static_cast<int64_t>(dest) * expertCount_],
            layout, expertCount_);
        CopyUbToGm(zGm_[static_cast<int64_t>(dest) * rankSize_],
            layout[ownerCopiesOffset], rankSize_);
        CopyUbToGm(expertsToCopyGm_[static_cast<int64_t>(dest) * expertsPerRank_],
            selectedExperts, expertsPerRank_);
        if (dest == rank_) {
            CopyUbToGm(cuSeqlensGm_, layout[cuSeqlensOffset], groupCount);
            CopyUbToGm(remoteStatsGm_, layout[statsOffset], 1);
        }
    }

    __aicore__ inline void FinalizeRemoteStats()
    {
        if (blockIdx_ != 0) {
            return;
        }
        AscendC::LocalTensor<int32_t> copyCounts = RouteTile();
        CopyGmToUb(copyCounts, zGm_, static_cast<int64_t>(rankSize_) * rankSize_);
        int32_t ownedCopies = 0;
        for (int32_t dest = 0; dest < rankSize_; ++dest) {
            ownedCopies += copyCounts.GetValue(static_cast<int64_t>(dest) * rankSize_ + rank_);
        }
        copyCounts.SetValue(0, ownedCopies);
        CopyUbToGm(remoteStatsGm_[1], copyCounts, 1);
    }

    __aicore__ inline void BuildPrefixes()
    {
        if (blockIdx_ != 0) {
            return;
        }

        AscendC::LocalTensor<int32_t> cumulative = Array0();
        AscendC::LocalTensor<int32_t> row = Array1();
        AscendC::LocalTensor<int32_t> next = Array2();
        AscendC::Duplicate(cumulative, static_cast<int32_t>(0), expertCount_);
        for (int32_t block = 0; block < blockCount_; ++block) {
            const int64_t rowOffset = static_cast<int64_t>(block) * expertCount_;
            CopyGmToUb(row, blockHistogramGm_[rowOffset], expertCount_);
            AscendC::Add(next, row, cumulative, static_cast<int32_t>(expertCount_));
            AscendC::Adds(row, cumulative, static_cast<int32_t>(0),
                static_cast<int32_t>(expertCount_));
            AscendC::Adds(cumulative, next, static_cast<int32_t>(0),
                static_cast<int32_t>(expertCount_));
            WaitVectorForMte3();
            CopyUbToGm(blockHistogramGm_[rowOffset], row, expertCount_);
        }

        AscendC::Duplicate(cumulative, static_cast<int32_t>(0), expertCount_);
        for (int32_t dest = 0; dest < rankSize_; ++dest) {
            const int64_t rowOffset = static_cast<int64_t>(dest) * expertCount_;
            CopyGmToUb(row, allocPrefixGm_[rowOffset], expertCount_);
            AscendC::Add(row, row, cumulative, static_cast<int32_t>(expertCount_));
            AscendC::Adds(cumulative, row, static_cast<int32_t>(0),
                static_cast<int32_t>(expertCount_));
            WaitVectorForMte3();
            CopyUbToGm(allocPrefixGm_[rowOffset], row, expertCount_);
        }
    }

    __aicore__ inline int32_t FindDestination(int32_t expert, int32_t globalRank, int32_t *previous) const
    {
        int32_t lo = 0;
        int32_t hi = rankSize_;
        while (lo < hi) {
            const int32_t mid = (lo + hi) >> 1;
            const int32_t cumulative = allocPrefixGm_.GetValue(
                static_cast<int64_t>(mid) * expertCount_ + expert);
            if (cumulative > globalRank) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        *previous = lo == 0 ? 0 : allocPrefixGm_.GetValue(
            static_cast<int64_t>(lo - 1) * expertCount_ + expert);
        return lo;
    }

    __aicore__ inline int32_t FindDestinationLocal(int32_t expert, int32_t globalRank,
        const AscendC::LocalTensor<int32_t> &allocation, int32_t *previous) const
    {
        if (rankSize_ == 8) {
            int32_t lo = 0;
            int32_t prior = 0;
            int32_t cumulative = allocation.GetValue(3 * expertCount_ + expert);
            if (cumulative <= globalRank) {
                lo = 4;
                prior = cumulative;
            }
            cumulative = allocation.GetValue(static_cast<int64_t>(lo + 1) * expertCount_ + expert);
            if (cumulative <= globalRank) {
                lo += 2;
                prior = cumulative;
            }
            cumulative = allocation.GetValue(static_cast<int64_t>(lo) * expertCount_ + expert);
            if (cumulative <= globalRank) {
                ++lo;
                prior = cumulative;
            }
            *previous = prior;
            return lo;
        }
        if (rankSize_ == 4) {
            int32_t lo = 0;
            int32_t prior = 0;
            int32_t cumulative = allocation.GetValue(expertCount_ + expert);
            if (cumulative <= globalRank) {
                lo = 2;
                prior = cumulative;
            }
            cumulative = allocation.GetValue(static_cast<int64_t>(lo) * expertCount_ + expert);
            if (cumulative <= globalRank) {
                ++lo;
                prior = cumulative;
            }
            *previous = prior;
            return lo;
        }
        int32_t lo = 0;
        int32_t hi = rankSize_;
        while (lo < hi) {
            const int32_t mid = (lo + hi) >> 1;
            const int32_t cumulative = allocation.GetValue(
                static_cast<int64_t>(mid) * expertCount_ + expert);
            if (cumulative > globalRank) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        *previous = lo == 0 ? 0 : allocation.GetValue(
            static_cast<int64_t>(lo - 1) * expertCount_ + expert);
        return lo;
    }

    __aicore__ inline void BuildDstPreloaded(int64_t sourcePrefixOffset)
    {
        AscendC::LocalTensor<int32_t> counters = Array0();
        AscendC::LocalTensor<int32_t> routes = Array1();
        AscendC::LocalTensor<int32_t> dst = Array2();
        AscendC::LocalTensor<int32_t> allocation = RouteTile();
        AscendC::LocalTensor<int32_t> offsets = DstTile();
        AscendC::LocalTensor<int32_t> sourcePrefixes = allocation[sourcePrefixOffset];
        const int64_t tableElements = static_cast<int64_t>(rankSize_) * expertCount_;

        CopyGmToUb(counters, blockHistogramGm_[blockIdx_ * expertCount_], expertCount_);
        CopyGmToUb(allocation, allocPrefixGm_, tableElements);
        CopyGmToUb(offsets, expertOffsetsGm_, tableElements);
        if (rank_ == 0) {
            AscendC::Duplicate(sourcePrefixes, static_cast<int32_t>(0), expertCount_);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        } else {
            CopyGmToUb(sourcePrefixes,
                tpePrefixGm_[static_cast<int64_t>(rank_ - 1) * expertCount_], expertCount_);
        }

        int64_t tokenBegin = 0;
        int64_t tokenEnd = 0;
        TokenRange(&tokenBegin, &tokenEnd);
        const int64_t tokensPerTile = kArrayCapacity / k_;
        for (int64_t tileBegin = tokenBegin; tileBegin < tokenEnd; tileBegin += tokensPerTile) {
            const int64_t tileEnd = MinInt64(tileBegin + tokensPerTile, tokenEnd);
            const int64_t tileTokenCount = tileEnd - tileBegin;
            const int64_t tileRoutes = tileTokenCount * k_;
            CopyGmToUb(routes, topkGm_[tileBegin * k_], tileRoutes);

            for (int64_t token = 0; token < tileTokenCount; ++token) {
                uint64_t seenLow = 0;
                uint64_t seenHigh = 0;
                for (int64_t topkIndex = 0; topkIndex < k_; ++topkIndex) {
                    const int64_t route = token * k_ + topkIndex;
                    const int32_t expert = routes.GetValue(route);
                    const int32_t localOccurrence = counters.GetValue(expert);
                    counters.SetValue(expert, localOccurrence + 1);
                    const int32_t globalRank = sourcePrefixes.GetValue(expert) + localOccurrence;
                    int32_t previous = 0;
                    const int32_t dest = FindDestinationLocal(expert, globalRank, allocation, &previous);
                    const int32_t base = offsets.GetValue(
                        static_cast<int64_t>(dest) * expertCount_ + expert);
                    const int32_t raw = static_cast<int32_t>(
                        static_cast<int64_t>(dest) * dispatchedCapacity_ + base + globalRank - previous);

                    bool duplicate = false;
                    if (dest < 64) {
                        const uint64_t bit = static_cast<uint64_t>(1) << dest;
                        duplicate = (seenLow & bit) != 0;
                        seenLow |= bit;
                    } else {
                        const uint64_t bit = static_cast<uint64_t>(1) << (dest - 64);
                        duplicate = (seenHigh & bit) != 0;
                        seenHigh |= bit;
                    }
                    dst.SetValue(route, duplicate ? -raw - 1 : raw);
                }
            }
            CopyUbToGm(dstGm_[tileBegin * k_], dst, tileRoutes);
        }
    }

    __aicore__ inline void BuildDstStreaming()
    {
        AscendC::LocalTensor<int32_t> counters = Array0();
        AscendC::LocalTensor<int32_t> routes = RouteTile();
        AscendC::LocalTensor<int32_t> dst = DstTile();
        CopyGmToUb(counters, blockHistogramGm_[blockIdx_ * expertCount_], expertCount_);

        int64_t tokenBegin = 0;
        int64_t tokenEnd = 0;
        TokenRange(&tokenBegin, &tokenEnd);
        const int64_t tokensPerTile = kPlannerRouteTileInts / k_;
        for (int64_t tileBegin = tokenBegin; tileBegin < tokenEnd; tileBegin += tokensPerTile) {
            const int64_t tileEnd = MinInt64(tileBegin + tokensPerTile, tokenEnd);
            const int64_t tileTokenCount = tileEnd - tileBegin;
            const int64_t tileRoutes = tileTokenCount * k_;
            CopyGmToUb(routes, topkGm_[tileBegin * k_], tileRoutes);

            for (int64_t token = 0; token < tileTokenCount; ++token) {
                uint64_t seenLow = 0;
                uint64_t seenHigh = 0;
                for (int64_t topkIndex = 0; topkIndex < k_; ++topkIndex) {
                    const int64_t route = token * k_ + topkIndex;
                    const int32_t expert = routes.GetValue(route);
                    const int32_t localOccurrence = counters.GetValue(expert);
                    counters.SetValue(expert, localOccurrence + 1);
                    const int32_t sourcePrefix = rank_ == 0 ? 0 : tpePrefixGm_.GetValue(
                        static_cast<int64_t>(rank_ - 1) * expertCount_ + expert);
                    const int32_t globalRank = sourcePrefix + localOccurrence;
                    int32_t previous = 0;
                    const int32_t dest = FindDestination(expert, globalRank, &previous);
                    const int32_t base = expertOffsetsGm_.GetValue(
                        static_cast<int64_t>(dest) * expertCount_ + expert);
                    const int32_t raw = static_cast<int32_t>(
                        static_cast<int64_t>(dest) * dispatchedCapacity_ + base + globalRank - previous);

                    bool duplicate = false;
                    if (dest < 64) {
                        const uint64_t bit = static_cast<uint64_t>(1) << dest;
                        duplicate = (seenLow & bit) != 0;
                        seenLow |= bit;
                    } else {
                        const uint64_t bit = static_cast<uint64_t>(1) << (dest - 64);
                        duplicate = (seenHigh & bit) != 0;
                        seenHigh |= bit;
                    }
                    dst.SetValue(route, duplicate ? -raw - 1 : raw);
                }
            }
            CopyUbToGm(dstGm_[tileBegin * k_], dst, tileRoutes);
        }
    }

    __aicore__ inline void BuildDst()
    {
        constexpr int64_t kMteAlignmentInts = 32 / sizeof(int32_t);
        const int64_t tableElements = static_cast<int64_t>(rankSize_) * expertCount_;
        const int64_t sourcePrefixOffset = AlignUp(tableElements, kMteAlignmentInts);
        if (sourcePrefixOffset + expertCount_ <= kPlannerRouteTileInts &&
            tableElements <= kPlannerRouteTileInts) {
            BuildDstPreloaded(sourcePrefixOffset);
        } else {
            BuildDstStreaming();
        }
    }

    __gm__ TileXR::CommArgs *args_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    int64_t s_ = 0;
    int64_t k_ = 0;
    int64_t expertCount_ = 0;
    int64_t expertsPerRank_ = 0;
    int64_t routeCount_ = 0;
    int64_t dispatchedCapacity_ = 0;
    uint64_t waitIterations_ = 0;
    int64_t magic_ = 0;
    int64_t blockIdx_ = 0;
    int64_t blockCount_ = 0;
    GM_ADDR shareAddrs_[TileXR::TILEXR_MAX_RANK_SIZE] = {};

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> syncBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;
    SyncCollectives sync_;
    AscendC::GlobalTensor<int32_t> topkGm_;
    AscendC::GlobalTensor<int32_t> localTpeGm_;
    AscendC::GlobalTensor<int32_t> dstGm_;
    AscendC::GlobalTensor<int32_t> cuSeqlensGm_;
    AscendC::GlobalTensor<int32_t> expertsToCopyGm_;
    AscendC::GlobalTensor<int32_t> remoteStatsGm_;
    AscendC::GlobalTensor<int32_t> plannerStatusGm_;
    AscendC::GlobalTensor<int32_t> tpePrefixGm_;
    AscendC::GlobalTensor<int32_t> blockHistogramGm_;
    AscendC::GlobalTensor<int32_t> allocPrefixGm_;
    AscendC::GlobalTensor<int32_t> expertOffsetsGm_;
    AscendC::GlobalTensor<int32_t> zGm_;
    AscendC::GlobalTensor<int32_t> groupTotalsGm_;
};

} // namespace Kernel
} // namespace TileXRMoonEp

extern "C" __global__ __aicore__ void tilexr_moonep_planner_kernel(GM_ADDR commArgs,
    GM_ADDR topkExpertIds, GM_ADDR tokensPerExpert, GM_ADDR workspace, GM_ADDR dst,
    GM_ADDR cuSeqlens, GM_ADDR expertsToCopy, GM_ADDR remoteStats, GM_ADDR plannerStatus,
    int64_t s, int64_t k, int64_t expertCount, int64_t expertsPerRank,
    int64_t routeCount, int64_t dispatchedCapacity, uint64_t waitIterations,
    uint64_t tpePrefixOffset, uint64_t blockHistogramOffset, uint64_t allocPrefixOffset,
    uint64_t expertOffsetsOffset, uint64_t zOffset, uint64_t groupTotalsOffset, int64_t magic)
{
    TileXRMoonEp::Kernel::PlannerKernel op;
    op.Init(commArgs, topkExpertIds, tokensPerExpert, workspace, dst, cuSeqlens,
        expertsToCopy, remoteStats, plannerStatus, s, k, expertCount, expertsPerRank,
        routeCount, dispatchedCapacity, waitIterations, tpePrefixOffset,
        blockHistogramOffset, allocPrefixOffset, expertOffsetsOffset, zOffset,
        groupTotalsOffset, magic);
    op.Process();
}
