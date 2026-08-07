#include "kernel_operator.h"

#include <cstdint>

#include "comm_args.h"
#include "combine_common.h"
#include "tilexr_sync.h"

namespace TileXRMoonEp {
namespace Kernel {

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void CopyBytesGmToGm(GM_ADDR dstAddr, GM_ADDR srcAddr,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &transferBuf, int64_t bytes)
{
    if (dstAddr == nullptr || srcAddr == nullptr || bytes <= 0) {
        return;
    }
    AscendC::LocalTensor<uint8_t> local = transferBuf.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(srcAddr), bytes);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dstAddr), bytes);

    for (int64_t copied = 0; copied < bytes; copied += kMoonEpCombineFloatScratchBytes) {
        const int64_t tileBytes = MinInt64(bytes - copied, kMoonEpCombineFloatScratchBytes);
        AscendC::DataCopyExtParams params {
            1, static_cast<uint32_t>(tileBytes), 0, 0, 0
        };
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

class CombineKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR dst, GM_ADDR dupGroups,
        GM_ADDR dupLoffs, GM_ADDR dupCounts, GM_ADDR hiddenNvsh,
        GM_ADDR routeWeightsNvs, GM_ADDR hiddenSh, GM_ADDR routeWeightsSk,
        GM_ADDR status, int64_t s, int64_t k, int64_t n, int64_t nvS,
        int64_t hiddenRowBytes, int64_t hiddenChunkBytes, int64_t hiddenChunkStride,
        int64_t chunkCount, uint64_t hiddenPayloadBytes, uint64_t routeWeightsOffset,
        uint64_t routeWeightsBytes, uint64_t waitIterations, uint64_t flags,
        int64_t magic)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
        dstAddr_ = dst;
        dupGroupsAddr_ = dupGroups;
        dupLoffsAddr_ = dupLoffs;
        dupCountsAddr_ = dupCounts;
        hiddenNvshAddr_ = hiddenNvsh;
        routeWeightsNvsAddr_ = routeWeightsNvs;
        hiddenShAddr_ = hiddenSh;
        routeWeightsSkAddr_ = routeWeightsSk;
        statusAddr_ = status;
        s_ = s;
        k_ = k;
        n_ = n;
        nvS_ = nvS;
        hiddenRowBytes_ = hiddenRowBytes;
        hiddenChunkBytes_ = hiddenChunkBytes;
        hiddenChunkStride_ = hiddenChunkStride;
        chunkCount_ = chunkCount;
        hiddenPayloadBytes_ = hiddenPayloadBytes;
        routeWeightsOffset_ = routeWeightsOffset;
        routeWeightsBytes_ = routeWeightsBytes;
        waitIterations_ = waitIterations;
        flags_ = flags;
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
        pipe_.InitBuffer(bfloatBuf_, kMoonEpCombineBfloatScratchBytes);
        pipe_.InitBuffer(routeBuf_, kMoonEpCombineFloatScratchBytes);
        pipe_.InitBuffer(accumulatorBuf_, kMoonEpCombineFloatScratchBytes);
        sync_.Init(rank_, rankSize_, shareAddrs_, syncBuf_);
        initialized_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!Valid() || AscendC::GetBlockIdx() != 0) {
            return;
        }
        localWindow_ = shareAddrs_[rank_] + TileXR::IPC_DATA_OFFSET;
        for (int64_t chunk = 0; chunk < chunkCount_; ++chunk) {
            const int64_t chunkOffset = chunk * hiddenChunkBytes_;
            const int64_t bytesThisChunk = MinInt64(
                hiddenRowBytes_ - chunkOffset, hiddenChunkBytes_);
            PublishLocalInput(chunkOffset, bytesThisChunk, chunk == 0);
            if (!PreReduceDuplicates(bytesThisChunk)) {
                return;
            }
            PublishStep(ChunkStep(kMoonEpCombineDataReadyStep, chunk));
            if (!WaitAllPeers(ChunkStep(kMoonEpCombineDataReadyStep, chunk))) {
                return;
            }

            if (!ReduceHiddenChunk(chunkOffset, bytesThisChunk)) {
                return;
            }
            if (chunk == 0 && routeWeightsBytes_ > 0 && !GatherWeights()) {
                return;
            }
            PublishStep(ChunkStep(kMoonEpCombineWindowDrainedStep, chunk));
            if (!WaitAllPeers(ChunkStep(kMoonEpCombineWindowDrainedStep, chunk))) {
                return;
            }
        }
        StoreStatus(kMoonEpCombineStatusSuccess);
    }

private:
    __aicore__ inline bool Valid() const
    {
        const uint64_t allowedFlags = kMoonEpFlagSkipInterRankSync;
        uint64_t windowBytes = hiddenPayloadBytes_;
        const uint64_t weightsEnd = routeWeightsOffset_ + routeWeightsBytes_;
        windowBytes = weightsEnd > windowBytes ? weightsEnd : windowBytes;
        return initialized_ && args_ != nullptr && dstAddr_ != nullptr &&
            dupGroupsAddr_ != nullptr && dupLoffsAddr_ != nullptr &&
            dupCountsAddr_ != nullptr && hiddenNvshAddr_ != nullptr &&
            hiddenShAddr_ != nullptr && statusAddr_ != nullptr && rank_ >= 0 &&
            rank_ < rankSize_ && s_ > 0 && k_ > 0 && k_ <= 32 && n_ == s_ * k_ &&
            nvS_ >= n_ && hiddenRowBytes_ > 0 && hiddenChunkBytes_ > 0 &&
            hiddenChunkStride_ >= hiddenChunkBytes_ &&
            hiddenChunkStride_ % static_cast<int64_t>(kMoonEpStageAlignment) == 0 &&
            chunkCount_ > 0 && hiddenPayloadBytes_ ==
                static_cast<uint64_t>(nvS_) * static_cast<uint64_t>(hiddenChunkStride_) &&
            windowBytes <= static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) &&
            waitIterations_ > 0 && magic_ > 0 && (flags_ & ~allowedFlags) == 0 &&
            ((routeWeightsBytes_ == 0 && routeWeightsNvsAddr_ == nullptr &&
                routeWeightsSkAddr_ == nullptr) ||
                (routeWeightsBytes_ == static_cast<uint64_t>(nvS_) * sizeof(float) &&
                    routeWeightsNvsAddr_ != nullptr && routeWeightsSkAddr_ != nullptr));
    }

    __aicore__ inline int32_t ChunkStep(int32_t base, int64_t chunk) const
    {
        return base + static_cast<int32_t>(chunk * 4);
    }

    __aicore__ inline int32_t LoadInt(GM_ADDR address)
    {
        AscendC::LocalTensor<int32_t> local = bfloatBuf_.Get<int32_t>();
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
        AscendC::LocalTensor<int32_t> local = bfloatBuf_.Get<int32_t>();
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

    __aicore__ inline int32_t LoadRoute(int64_t route)
    {
        return LoadInt(dstAddr_ + route * static_cast<int64_t>(sizeof(int32_t)));
    }

    __aicore__ inline bool DecodeRoute(int32_t encoded, int64_t &peer, int64_t &offset)
    {
        const int64_t encoded64 = static_cast<int64_t>(encoded);
        const int64_t raw = encoded64 >= 0 ? encoded64 : -encoded64 - 1;
        peer = raw / nvS_;
        offset = raw % nvS_;
        if (raw < 0 || peer < 0 || peer >= rankSize_ || offset < 0 || offset >= nvS_) {
            Fail(kMoonEpCombineStatusInvalidRoute);
            return false;
        }
        return true;
    }

    __aicore__ inline void PublishLocalInput(
        int64_t chunkOffset, int64_t bytesThisChunk, bool firstChunk)
    {
        for (int64_t row = 0; row < nvS_; ++row) {
            CopyBytesGmToGm(localWindow_ + row * hiddenChunkStride_,
                hiddenNvshAddr_ + row * hiddenRowBytes_ + chunkOffset,
                routeBuf_, bytesThisChunk);
        }
        if (firstChunk && routeWeightsBytes_ > 0) {
            CopyBytesGmToGm(localWindow_ + static_cast<int64_t>(routeWeightsOffset_),
                routeWeightsNvsAddr_, routeBuf_, static_cast<int64_t>(routeWeightsBytes_));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool PreReduceDuplicates(int64_t bytesThisChunk)
    {
        const int32_t groupCount = LoadInt(dupCountsAddr_);
        const int32_t duplicateCount = LoadInt(dupCountsAddr_ + sizeof(int32_t));
        if (groupCount < 0 || groupCount > nvS_ || duplicateCount < 0 ||
            duplicateCount > nvS_ || bytesThisChunk % sizeof(bfloat16_t) != 0) {
            Fail(kMoonEpCombineStatusInvalidRoute);
            return false;
        }

        AscendC::LocalTensor<bfloat16_t> bfloatScratch = bfloatBuf_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> routeScratch = routeBuf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        const int64_t elements = bytesThisChunk / sizeof(bfloat16_t);
        for (int32_t group = 0; group < groupCount; ++group) {
            const int64_t base = static_cast<int64_t>(group) * 3;
            const int32_t primary = LoadInt(dupGroupsAddr_ + base * sizeof(int32_t));
            const int32_t start = LoadInt(dupGroupsAddr_ + (base + 1) * sizeof(int32_t));
            const int32_t count = LoadInt(dupGroupsAddr_ + (base + 2) * sizeof(int32_t));
            if (primary < 0 || primary >= nvS_ || start < 0 || count <= 0 ||
                static_cast<int64_t>(start) + count > duplicateCount) {
                Fail(kMoonEpCombineStatusInvalidRoute);
                return false;
            }

            for (int64_t tileOffset = 0; tileOffset < elements;
                tileOffset += kMoonEpCombineHiddenTileElements) {
                const int64_t tileElements = MinInt64(
                    elements - tileOffset, kMoonEpCombineHiddenTileElements);
                AscendC::GlobalTensor<bfloat16_t> primaryGm;
                primaryGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                    localWindow_ + static_cast<int64_t>(primary) * hiddenChunkStride_) +
                    tileOffset, tileElements);
                AscendC::DataCopyExtParams params {
                    1, static_cast<uint32_t>(tileElements * sizeof(bfloat16_t)), 0, 0, 0
                };
                AscendC::DataCopyPadExtParams<bfloat16_t> pad {false, 0, 0, 0};
                AscendC::DataCopyPad(bfloatScratch, primaryGm, params, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(accumulator, bfloatScratch,
                    AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(tileElements));
                AscendC::PipeBarrier<PIPE_V>();

                for (int32_t index = 0; index < count; ++index) {
                    const int32_t duplicate = LoadInt(dupLoffsAddr_ +
                        static_cast<int64_t>(start + index) * sizeof(int32_t));
                    if (duplicate < 0 || duplicate >= nvS_) {
                        Fail(kMoonEpCombineStatusInvalidRoute);
                        return false;
                    }
                    AscendC::GlobalTensor<bfloat16_t> duplicateGm;
                    duplicateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                        localWindow_ + static_cast<int64_t>(duplicate) * hiddenChunkStride_) +
                        tileOffset, tileElements);
                    AscendC::DataCopyPad(bfloatScratch, duplicateGm, params, pad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(routeScratch, bfloatScratch,
                        AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(accumulator, accumulator, routeScratch,
                        static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                }

                AscendC::Cast(bfloatScratch, accumulator,
                    AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(tileElements));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(primaryGm, bfloatScratch, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline bool ReduceHiddenChunk(
        int64_t chunkOffset, int64_t bytesThisChunk)
    {
        if (bytesThisChunk % sizeof(bfloat16_t) != 0) {
            Fail(kMoonEpCombineStatusInvalidRoute);
            return false;
        }
        const int64_t elements = bytesThisChunk / sizeof(bfloat16_t);
        AscendC::LocalTensor<bfloat16_t> bfloatScratch = bfloatBuf_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> routeScratch = routeBuf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();

        for (int64_t token = 0; token < s_; ++token) {
            for (int64_t tileOffset = 0; tileOffset < elements;
                tileOffset += kMoonEpCombineHiddenTileElements) {
                const int64_t tileElements = MinInt64(
                    elements - tileOffset, kMoonEpCombineHiddenTileElements);
                AscendC::Duplicate(accumulator, 0.0f, static_cast<int32_t>(tileElements));
                AscendC::PipeBarrier<PIPE_V>();

                for (int64_t topk = 0; topk < k_; ++topk) {
                    const int64_t route = token * k_ + topk;
                    const int32_t encoded = LoadRoute(route);
                    if (encoded < 0) {
                        continue;
                    }
                    int64_t peer = 0;
                    int64_t offset = 0;
                    if (!DecodeRoute(encoded, peer, offset)) {
                        return false;
                    }
                    AscendC::GlobalTensor<bfloat16_t> remote;
                    remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                        shareAddrs_[peer] + TileXR::IPC_DATA_OFFSET +
                        offset * hiddenChunkStride_) + tileOffset, tileElements);
                    AscendC::DataCopyExtParams params {
                        1, static_cast<uint32_t>(tileElements * sizeof(bfloat16_t)), 0, 0, 0
                    };
                    AscendC::DataCopyPadExtParams<bfloat16_t> pad {false, 0, 0, 0};
                    AscendC::DataCopyPad(bfloatScratch, remote, params, pad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(routeScratch, bfloatScratch,
                        AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(accumulator, accumulator, routeScratch,
                        static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                }

                AscendC::Cast(bfloatScratch, accumulator,
                    AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(tileElements));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::GlobalTensor<bfloat16_t> output;
                output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                    hiddenShAddr_ + token * hiddenRowBytes_ + chunkOffset) + tileOffset,
                    tileElements);
                AscendC::DataCopyExtParams params {
                    1, static_cast<uint32_t>(tileElements * sizeof(bfloat16_t)), 0, 0, 0
                };
                AscendC::DataCopyPad(output, bfloatScratch, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline bool GatherWeights()
    {
        for (int64_t route = 0; route < n_; ++route) {
            const int32_t encoded = LoadRoute(route);
            int64_t peer = 0;
            int64_t offset = 0;
            if (!DecodeRoute(encoded, peer, offset)) {
                return false;
            }
            CopyBytesGmToGm(routeWeightsSkAddr_ + route * sizeof(float),
                shareAddrs_[peer] + TileXR::IPC_DATA_OFFSET +
                    static_cast<int64_t>(routeWeightsOffset_) + offset * sizeof(float),
                routeBuf_, sizeof(float));
        }
        return true;
    }

    __aicore__ inline void StoreStatus(int32_t value)
    {
        StoreInt(statusAddr_, value);
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
            if (step == kMoonEpCombineFailedStep) {
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
                Fail(kMoonEpCombineStatusRemoteFailureBase + peer);
                return false;
            }
            if (result == 2) {
                Fail(kMoonEpCombineStatusTimeoutBase + peer);
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void Fail(int32_t status)
    {
        StoreStatus(status);
        PublishStep(kMoonEpCombineFailedStep);
    }

    __gm__ TileXR::CommArgs *args_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    int64_t s_ = 0;
    int64_t k_ = 0;
    int64_t n_ = 0;
    int64_t nvS_ = 0;
    int64_t hiddenRowBytes_ = 0;
    int64_t hiddenChunkBytes_ = 0;
    int64_t hiddenChunkStride_ = 0;
    int64_t chunkCount_ = 0;
    uint64_t hiddenPayloadBytes_ = 0;
    uint64_t routeWeightsOffset_ = 0;
    uint64_t routeWeightsBytes_ = 0;
    uint64_t waitIterations_ = 0;
    uint64_t flags_ = 0;
    int64_t magic_ = 0;
    bool initialized_ = false;
    GM_ADDR dstAddr_ = nullptr;
    GM_ADDR dupGroupsAddr_ = nullptr;
    GM_ADDR dupLoffsAddr_ = nullptr;
    GM_ADDR dupCountsAddr_ = nullptr;
    GM_ADDR hiddenNvshAddr_ = nullptr;
    GM_ADDR routeWeightsNvsAddr_ = nullptr;
    GM_ADDR hiddenShAddr_ = nullptr;
    GM_ADDR routeWeightsSkAddr_ = nullptr;
    GM_ADDR statusAddr_ = nullptr;
    GM_ADDR localWindow_ = nullptr;
    GM_ADDR shareAddrs_[TileXR::TILEXR_MAX_RANK_SIZE] = {};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> syncBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bfloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> routeBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accumulatorBuf_;
    SyncCollectives sync_;
};

} // namespace Kernel
} // namespace TileXRMoonEp

extern "C" __global__ __aicore__ void tilexr_moonep_combine_kernel(GM_ADDR commArgs,
    GM_ADDR dst, GM_ADDR dupGroups, GM_ADDR dupLoffs, GM_ADDR dupCounts,
    GM_ADDR hiddenNvsh, GM_ADDR routeWeightsNvs, GM_ADDR hiddenSh,
    GM_ADDR routeWeightsSk, GM_ADDR status, int64_t s, int64_t k, int64_t n,
    int64_t nvS, int64_t hiddenRowBytes, int64_t hiddenChunkBytes,
    int64_t hiddenChunkStride, int64_t chunkCount, uint64_t hiddenPayloadBytes,
    uint64_t routeWeightsOffset, uint64_t routeWeightsBytes,
    uint64_t waitIterations, uint64_t flags, int64_t magic)
{
    TileXRMoonEp::Kernel::CombineKernel op;
    op.Init(commArgs, dst, dupGroups, dupLoffs, dupCounts, hiddenNvsh,
        routeWeightsNvs, hiddenSh, routeWeightsSk, status, s, k, n, nvS,
        hiddenRowBytes, hiddenChunkBytes, hiddenChunkStride, chunkCount,
        hiddenPayloadBytes, routeWeightsOffset, routeWeightsBytes,
        waitIterations, flags, magic);
    op.Process();
}
