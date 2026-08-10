#include "kernel_operator.h"

#include <cstdint>

#include "comm_args.h"
#include "prefetch_weight_common.h"
#include "tilexr_udma.h"

namespace TileXRMoonEp {
namespace Kernel {

constexpr uint32_t kMaxTrackedRankSize = 1024;
constexpr uint32_t kUsedPeerWordCount = kMaxTrackedRankSize / 64;

class PrefetchWeightKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertsToCopy,
        GM_ADDR gate, GM_ADDR up, GM_ADDR down, GM_ADDR status,
        uint64_t gateOffset, uint64_t upOffset, uint64_t downOffset,
        uint32_t gateRowBytes, uint32_t upRowBytes, uint32_t downRowBytes,
        int32_t rank, int32_t rankSize, int64_t expertsPerRank,
        int64_t prefetchSlots, uint32_t qpNum)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
        expertsToCopy_ = reinterpret_cast<__gm__ int32_t *>(expertsToCopy);
        projections_[0] = reinterpret_cast<__gm__ uint8_t *>(gate);
        projections_[1] = reinterpret_cast<__gm__ uint8_t *>(up);
        projections_[2] = reinterpret_cast<__gm__ uint8_t *>(down);
        offsets_[0] = gateOffset;
        offsets_[1] = upOffset;
        offsets_[2] = downOffset;
        rowBytes_[0] = gateRowBytes;
        rowBytes_[1] = upRowBytes;
        rowBytes_[2] = downRowBytes;
        status_ = reinterpret_cast<__gm__ uint32_t *>(status);
        expertsPerRank_ = expertsPerRank;
        prefetchSlots_ = prefetchSlots;
        rank_ = rank;
        rankSize_ = rankSize;
        qpNum_ = qpNum;
        const uint32_t subBlockCount = static_cast<uint32_t>(get_subblockdim());
        worker_ = static_cast<uint32_t>(get_block_idx()) * subBlockCount +
            static_cast<uint32_t>(get_subblockid());
        workerCount_ = static_cast<uint32_t>(get_block_num()) * subBlockCount;
        pipe_.InitBuffer(wqeBuf_, TileXR::TILEXR_UDMA_WQE_SCRATCH_BYTES);
    }

    __aicore__ inline void Process()
    {
        InitializeStatus();
        AscendC::SyncAll<true>();

        uint64_t usedPeers[kUsedPeerWordCount] = {};
        uint32_t workerStatus = ValidateRuntime();
        if (workerStatus == 0) {
            SubmitReads(usedPeers, workerStatus);
            CompleteReads(usedPeers, workerStatus);
        }
        if (workerStatus != 0) {
            (void)AscendC::AtomicCas(status_, static_cast<uint32_t>(0), workerStatus);
        }
        AscendC::SyncAll<true>();
        if (worker_ == 0) {
            (void)AscendC::AtomicCas(status_, static_cast<uint32_t>(0),
                kPrefetchWeightStatusSuccess);
        }
        AscendC::SyncAll<true>();
    }

private:
    __aicore__ inline void InitializeStatus()
    {
        if (worker_ == 0) {
            StoreStatus(0);
        }
    }

    __aicore__ inline void StoreStatus(uint32_t value)
    {
        status_[0] = value;
        AscendC::GlobalTensor<uint32_t> statusGm;
        statusGm.SetGlobalBuffer(status_, 1);
        AscendC::DataCacheCleanAndInvalid<uint32_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(statusGm);
    }

    __aicore__ inline uint32_t ValidateRuntime() const
    {
        if (args_ == nullptr || status_ == nullptr || expertsToCopy_ == nullptr ||
            projections_[0] == nullptr || projections_[1] == nullptr ||
            projections_[2] == nullptr || rowBytes_[0] == 0 || rowBytes_[1] == 0 ||
            rowBytes_[2] == 0 || expertsPerRank_ <= 0 || prefetchSlots_ <= 0 ||
            prefetchSlots_ > expertsPerRank_ ||
            rank_ < 0 || rank_ >= rankSize_ ||
            rankSize_ > static_cast<int32_t>(kMaxTrackedRankSize) ||
            workerCount_ == 0 || worker_ >= workerCount_ || worker_ >= qpNum_ ||
            !TileXR::UDMARegistryEnabled(args_) || args_->rank != rank_ ||
            args_->rankSize != rankSize_) {
            return kPrefetchWeightStatusInvalidRuntime;
        }
        __gm__ TileXR::UDMAInfo *info = TileXR::GetUDMAInfo(args_);
        return info->qpNum == qpNum_ ? 0 : kPrefetchWeightStatusInvalidRuntime;
    }

    __aicore__ inline void MarkPeer(
        uint64_t usedPeers[kUsedPeerWordCount], int32_t peer) const
    {
        usedPeers[static_cast<uint32_t>(peer) >> 6] |=
            static_cast<uint64_t>(1) << (static_cast<uint32_t>(peer) & 63U);
    }

    __aicore__ inline bool PeerUsed(
        const uint64_t usedPeers[kUsedPeerWordCount], int32_t peer) const
    {
        return (usedPeers[static_cast<uint32_t>(peer) >> 6] &
            (static_cast<uint64_t>(1) << (static_cast<uint32_t>(peer) & 63U))) != 0;
    }

    __aicore__ inline void SubmitReads(
        uint64_t usedPeers[kUsedPeerWordCount], uint32_t &workerStatus)
    {
        auto wqeScratch = wqeBuf_.Get<uint8_t>();
        const int64_t globalExpertCount = expertsPerRank_ * rankSize_;
        const int64_t planRow = static_cast<int64_t>(rank_) * prefetchSlots_;
        for (int64_t slot = static_cast<int64_t>(worker_); slot < prefetchSlots_;
             slot += static_cast<int64_t>(workerCount_)) {
            const int32_t expert = expertsToCopy_[planRow + slot];
            if (expert < 0) {
                continue;
            }
            if (static_cast<int64_t>(expert) >= globalExpertCount) {
                if (workerStatus == 0) {
                    workerStatus = kPrefetchWeightStatusInvalidExpert;
                }
                continue;
            }
            const int32_t owner = expert / static_cast<int32_t>(expertsPerRank_);
            if (owner == rank_) {
                if (workerStatus == 0) {
                    workerStatus = kPrefetchWeightStatusLocalExpert;
                }
                continue;
            }
            const int32_t localExpert =
                expert % static_cast<int32_t>(expertsPerRank_);
            MarkPeer(usedPeers, owner);
            for (uint32_t projection = 0; projection < 3; ++projection) {
                const uint64_t sourceOffset = offsets_[projection] +
                    static_cast<uint64_t>(localExpert) * rowBytes_[projection];
                __gm__ uint8_t *destination = projections_[projection] +
                    static_cast<uint64_t>(expertsPerRank_ + slot) *
                        rowBytes_[projection];
                const uint32_t submitStatus = TileXR::UDMAGetNbiOnQp<uint8_t>(
                    args_, wqeScratch, owner, worker_, destination, sourceOffset,
                    rowBytes_[projection]);
                if (submitStatus != TileXR::TILEXR_UDMA_STATUS_SUCCESS &&
                    workerStatus == 0) {
                    workerStatus = kPrefetchWeightStatusSubmitErrorBase +
                        (submitStatus & 0xFFU);
                }
            }
        }
    }

    __aicore__ inline void CompleteReads(
        const uint64_t usedPeers[kUsedPeerWordCount], uint32_t &workerStatus)
    {
        bool completedAny = false;
        for (int32_t peer = 0; peer < rankSize_; ++peer) {
            if (!PeerUsed(usedPeers, peer)) {
                continue;
            }
            completedAny = true;
            const uint32_t cqStatus = TileXR::UDMAQuietStatusOnQp(args_, peer, worker_);
            if (cqStatus != 0 && workerStatus == 0) {
                workerStatus = kPrefetchWeightStatusCqErrorBase + (cqStatus & 0xFFU);
            }
        }
        if (completedAny) {
            AscendC::GlobalTensor<uint64_t> cache;
            cache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(0), 1);
            AscendC::DataCacheCleanAndInvalid<uint64_t,
                AscendC::CacheLine::ENTIRE_DATA_CACHE,
                AscendC::DcciDst::CACHELINE_OUT>(cache);
        }
    }

    __gm__ TileXR::CommArgs *args_ = nullptr;
    __gm__ int32_t *expertsToCopy_ = nullptr;
    __gm__ uint8_t *projections_[3] = {nullptr, nullptr, nullptr};
    __gm__ uint32_t *status_ = nullptr;
    uint64_t offsets_[3] = {0, 0, 0};
    uint32_t rowBytes_[3] = {0, 0, 0};
    int64_t expertsPerRank_ = 0;
    int64_t prefetchSlots_ = 0;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    uint32_t qpNum_ = 0;
    uint32_t worker_ = 0;
    uint32_t workerCount_ = 0;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wqeBuf_;
};

} // namespace Kernel
} // namespace TileXRMoonEp

extern "C" __global__ __aicore__ void tilexr_moonep_prefetch_weight_kernel(
    GM_ADDR commArgs, GM_ADDR expertsToCopy, GM_ADDR gate, GM_ADDR up, GM_ADDR down,
    GM_ADDR status, uint64_t gateOffset, uint64_t upOffset, uint64_t downOffset,
    uint64_t gateRowBytes, uint64_t upRowBytes, uint64_t downRowBytes,
    int64_t rank, int64_t rankSize, int64_t expertsPerRank,
    int64_t prefetchSlots, uint64_t qpNum)
{
    TileXRMoonEp::Kernel::PrefetchWeightKernel op;
    op.Init(commArgs, expertsToCopy, gate, up, down, status,
        gateOffset, upOffset, downOffset, static_cast<uint32_t>(gateRowBytes),
        static_cast<uint32_t>(upRowBytes), static_cast<uint32_t>(downRowBytes),
        static_cast<int32_t>(rank), static_cast<int32_t>(rankSize),
        expertsPerRank, prefetchSlots, static_cast<uint32_t>(qpNum));
    op.Process();
}
