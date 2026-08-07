#include "kernel_operator.h"

#include <cstdint>

#include "comm_args.h"
#include "prefetch_weight_common.h"
#include "tilexr_sync.h"

namespace TileXRMoonEp {
namespace Kernel {

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void CopyBytesGmToGm(GM_ADDR dstAddr, GM_ADDR srcAddr,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &workBuf, int64_t bytes)
{
    AscendC::LocalTensor<uint8_t> local = workBuf.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(srcAddr), bytes);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dstAddr), bytes);
    for (int64_t copied = 0; copied < bytes; copied += kMoonEpWorkUbBytes) {
        const int64_t tileBytes = MinInt64(bytes - copied, kMoonEpWorkUbBytes);
        AscendC::DataCopyExtParams params {1, static_cast<uint32_t>(tileBytes), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> pad {false, 0, 0, 0};
        AscendC::DataCopyPad(local, src[copied], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[copied], local, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

class PrefetchWeightKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertsToCopy,
        GM_ADDR fullGateWeight, GM_ADDR fullUpWeight, GM_ADDR fullDownWeight,
        GM_ADDR status, int64_t e, int64_t b, int64_t expertsPerRank,
        int64_t gateRowBytes, int64_t gateChunkBytes, int64_t gateChunkCount,
        int64_t upRowBytes, int64_t upChunkBytes, int64_t upChunkCount,
        int64_t downRowBytes, int64_t downChunkBytes, int64_t downChunkCount,
        int64_t iterationCount, uint64_t waitIterations, int64_t magic)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
        expertsToCopyAddr_ = expertsToCopy;
        weights_[0] = fullGateWeight;
        weights_[1] = fullUpWeight;
        weights_[2] = fullDownWeight;
        rowBytes_[0] = gateRowBytes;
        rowBytes_[1] = upRowBytes;
        rowBytes_[2] = downRowBytes;
        chunkBytes_[0] = gateChunkBytes;
        chunkBytes_[1] = upChunkBytes;
        chunkBytes_[2] = downChunkBytes;
        chunkCounts_[0] = gateChunkCount;
        chunkCounts_[1] = upChunkCount;
        chunkCounts_[2] = downChunkCount;
        statusAddr_ = status;
        e_ = e;
        b_ = b;
        expertsPerRank_ = expertsPerRank;
        iterationCount_ = iterationCount;
        waitIterations_ = waitIterations;
        magic_ = magic;
        if (args_ == nullptr) {
            return;
        }
        rank_ = args_->rank;
        rankSize_ = args_->rankSize;
        for (int32_t peer = 0; peer < rankSize_; ++peer) {
            shareAddrs_[peer] = args_->peerMems[peer];
        }
        pipe_.InitBuffer(syncBuf_, kMoonEpSyncUbBytes);
        pipe_.InitBuffer(workBuf_, kMoonEpWorkUbBytes);
        sync_.Init(rank_, rankSize_, shareAddrs_, syncBuf_);
        initialized_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!Valid() || AscendC::GetBlockIdx() != 0) {
            return;
        }
        localWindow_ = shareAddrs_[rank_] + TileXR::IPC_DATA_OFFSET;
        int64_t iteration = 0;
        for (int32_t projection = 0; projection < 3; ++projection) {
            for (int64_t slot = 0; slot < b_; ++slot) {
                for (int64_t chunk = 0; chunk < chunkCounts_[projection]; ++chunk) {
                    const int64_t chunkOffset = chunk * chunkBytes_[projection];
                    const int64_t bytesThisChunk = MinInt64(
                        rowBytes_[projection] - chunkOffset, chunkBytes_[projection]);
                    if (!PublishOwners(projection, slot, chunkOffset, bytesThisChunk)) {
                        return;
                    }
                    PublishStep(kMoonEpPrefetchWeightReadyStep +
                        static_cast<int32_t>(iteration * 3));
                    if (!WaitAllPeers(kMoonEpPrefetchWeightReadyStep +
                            static_cast<int32_t>(iteration * 3))) {
                        return;
                    }
                    const int32_t expert = LoadExpert(rank_, slot);
                    if (expert >= 0) {
                        CopyBytesGmToGm(weights_[projection] +
                                (e_ + slot) * rowBytes_[projection] + chunkOffset,
                            localWindow_, workBuf_, bytesThisChunk);
                    }
                    PublishStep(kMoonEpPrefetchWeightDrainedStep +
                        static_cast<int32_t>(iteration * 3));
                    if (!WaitAllPeers(kMoonEpPrefetchWeightDrainedStep +
                            static_cast<int32_t>(iteration * 3))) {
                        return;
                    }
                    ++iteration;
                }
            }
        }
        if (iteration != iterationCount_) {
            Fail(kMoonEpPrefetchWeightStatusInvalidPlan);
            return;
        }
        StoreStatus(kMoonEpPrefetchWeightStatusSuccess);
    }

private:
    __aicore__ inline bool Valid() const
    {
        return initialized_ && expertsToCopyAddr_ != nullptr && statusAddr_ != nullptr &&
            weights_[0] != nullptr && weights_[1] != nullptr && weights_[2] != nullptr &&
            rank_ >= 0 && rank_ < rankSize_ && e_ > 0 && b_ > 0 &&
            expertsPerRank_ > 0 && e_ == expertsPerRank_ * rankSize_ &&
            rowBytes_[0] > 0 && rowBytes_[1] > 0 && rowBytes_[2] > 0 &&
            chunkBytes_[0] > 0 && chunkBytes_[1] > 0 && chunkBytes_[2] > 0 &&
            chunkCounts_[0] > 0 && chunkCounts_[1] > 0 && chunkCounts_[2] > 0 &&
            iterationCount_ > 0 && waitIterations_ > 0 && magic_ > 0;
    }

    __aicore__ inline int32_t LoadInt(GM_ADDR address)
    {
        AscendC::LocalTensor<int32_t> local = workBuf_.Get<int32_t>();
        AscendC::GlobalTensor<int32_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
        AscendC::DataCopyExtParams params {1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> pad {false, 0, 0, 0};
        AscendC::DataCopyPad(local, src, params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        return local.GetValue(0);
    }

    __aicore__ inline void StoreInt(GM_ADDR address, int32_t value)
    {
        AscendC::LocalTensor<int32_t> local = workBuf_.Get<int32_t>();
        AscendC::GlobalTensor<int32_t> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
        local.SetValue(0, value);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams params {1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(dst, local, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline int32_t LoadExpert(int64_t rank, int64_t slot)
    {
        return LoadInt(expertsToCopyAddr_ + (rank * b_ + slot) * sizeof(int32_t));
    }

    __aicore__ inline bool PublishOwners(int32_t projection, int64_t slot,
        int64_t chunkOffset, int64_t bytesThisChunk)
    {
        for (int32_t destination = 0; destination < rankSize_; ++destination) {
            const int32_t expert = LoadExpert(destination, slot);
            if (expert == -1) {
                continue;
            }
            if (expert < 0 || expert >= e_) {
                Fail(kMoonEpPrefetchWeightStatusInvalidPlan);
                return false;
            }
            if (expert / expertsPerRank_ == rank_) {
                CopyBytesGmToGm(shareAddrs_[destination] + TileXR::IPC_DATA_OFFSET,
                    weights_[projection] + static_cast<int64_t>(expert) *
                        rowBytes_[projection] + chunkOffset,
                    workBuf_, bytesThisChunk);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline void StoreStatus(int32_t status)
    {
        StoreInt(statusAddr_, status);
    }

    __aicore__ inline void PublishStep(int32_t step)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        sync_.SetInnerFlag(static_cast<int32_t>(magic_), step);
    }

    __aicore__ inline int32_t WaitPeerStep(int32_t peer, int32_t expectedStep)
    {
        AscendC::GlobalTensor<int64_t> flag;
        flag.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(shareAddrs_[peer]),
            FLAG_UNIT_INT_NUM);
        const int64_t expectedMagic =
            static_cast<int64_t>(static_cast<int32_t>(magic_)) << MAGIC_OFFSET;
        for (uint64_t iteration = 0; iteration < waitIterations_; ++iteration) {
            AscendC::DataCacheCleanAndInvalid<int64_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(flag);
            const int64_t value = flag.GetValue(0);
            if ((value & MAGIC_MASK) != (expectedMagic & MAGIC_MASK)) {
                continue;
            }
            const int32_t step = static_cast<int32_t>(value & ~MAGIC_MASK);
            if (step == expectedStep) {
                return 0;
            }
            if (step == kMoonEpPrefetchWeightFailedStep) {
                return 1;
            }
        }
        return 2;
    }

    __aicore__ inline bool WaitAllPeers(int32_t expectedStep)
    {
        for (int32_t offset = 0; offset < rankSize_; ++offset) {
            const int32_t peer = (rank_ + offset) % rankSize_;
            const int32_t result = WaitPeerStep(peer, expectedStep);
            if (result == 1) {
                Fail(kMoonEpPrefetchWeightStatusRemoteFailureBase + peer);
                return false;
            }
            if (result == 2) {
                Fail(kMoonEpPrefetchWeightStatusTimeoutBase + peer);
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void Fail(int32_t status)
    {
        StoreStatus(status);
        PublishStep(kMoonEpPrefetchWeightFailedStep);
    }

    __gm__ TileXR::CommArgs *args_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    int64_t e_ = 0;
    int64_t b_ = 0;
    int64_t expertsPerRank_ = 0;
    int64_t iterationCount_ = 0;
    uint64_t waitIterations_ = 0;
    int64_t magic_ = 0;
    bool initialized_ = false;
    GM_ADDR expertsToCopyAddr_ = nullptr;
    GM_ADDR weights_[3] = {};
    int64_t rowBytes_[3] = {};
    int64_t chunkBytes_[3] = {};
    int64_t chunkCounts_[3] = {};
    GM_ADDR statusAddr_ = nullptr;
    GM_ADDR localWindow_ = nullptr;
    GM_ADDR shareAddrs_[TileXR::TILEXR_MAX_RANK_SIZE] = {};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> syncBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;
    SyncCollectives sync_;
};

} // namespace Kernel
} // namespace TileXRMoonEp

extern "C" __global__ __aicore__ void tilexr_moonep_prefetch_weight_kernel(
    GM_ADDR commArgs, GM_ADDR expertsToCopy, GM_ADDR fullGateWeight,
    GM_ADDR fullUpWeight, GM_ADDR fullDownWeight, GM_ADDR status, int64_t e,
    int64_t b, int64_t expertsPerRank, int64_t gateRowBytes,
    int64_t gateChunkBytes, int64_t gateChunkCount, int64_t upRowBytes,
    int64_t upChunkBytes, int64_t upChunkCount, int64_t downRowBytes,
    int64_t downChunkBytes, int64_t downChunkCount, int64_t iterationCount,
    uint64_t waitIterations, int64_t magic)
{
    TileXRMoonEp::Kernel::PrefetchWeightKernel op;
    op.Init(commArgs, expertsToCopy, fullGateWeight, fullUpWeight, fullDownWeight,
        status, e, b, expertsPerRank, gateRowBytes, gateChunkBytes, gateChunkCount,
        upRowBytes, upChunkBytes, upChunkCount, downRowBytes, downChunkBytes,
        downChunkCount, iterationCount, waitIterations, magic);
    op.Process();
}
