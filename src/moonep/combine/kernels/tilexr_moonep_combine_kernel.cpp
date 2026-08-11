#include "kernel_operator.h"

#include <cstdint>

#include "comm_args.h"
#include "combine_common.h"
#include "moonep_combine_schedule.h"
#include "tilexr_sync.h"

namespace TileXRMoonEp {
namespace Kernel {

constexpr uint32_t kChunkReadyStepBase = 1U << 10U;
constexpr uint32_t kChunkDrainedStepBase = 1U << 25U;
constexpr uint32_t kMaxChunkCount = 0xFFFFFFU;

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
    for (int64_t copied = 0; copied < bytes;
         copied += kMoonEpCombineFloatScratchBytes) {
        const int64_t tileBytes = MinInt64(
            bytes - copied, kMoonEpCombineFloatScratchBytes);
        AscendC::DataCopyExtParams params {
            1, static_cast<uint32_t>(tileBytes), 0, 0, 0};
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
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR dstLocal, GM_ADDR dst,
        GM_ADDR dupGroups, GM_ADDR dupLoffs, GM_ADDR dupCounts,
        GM_ADDR hiddenNvsh, GM_ADDR routeWeightsNvs, GM_ADDR hiddenSh,
        GM_ADDR routeWeightsSk, GM_ADDR status, int64_t s, int64_t k,
        int64_t n, int64_t nvS, int64_t hiddenRowBytes,
        int64_t hiddenChunkBytes, int64_t hiddenChunkStride,
        int64_t chunkCount, uint64_t sourceHiddenOffset,
        uint64_t receiveHiddenOffset, uint64_t hiddenPayloadBytes,
        uint64_t sourceWeightsOffset, uint64_t receiveWeightsOffset,
        uint64_t routeWeightsBytes, uint64_t duplicateMaskOffset,
        uint64_t doneOffset, uint64_t coreStatusOffset,
        uint64_t windowBytes, uint64_t waitIterations, uint64_t flags,
        int64_t magic)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
        dstLocalAddr_ = dstLocal;
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
        sourceHiddenOffset_ = sourceHiddenOffset;
        receiveHiddenOffset_ = receiveHiddenOffset;
        hiddenPayloadBytes_ = hiddenPayloadBytes;
        sourceWeightsOffset_ = sourceWeightsOffset;
        receiveWeightsOffset_ = receiveWeightsOffset;
        routeWeightsBytes_ = routeWeightsBytes;
        duplicateMaskOffset_ = duplicateMaskOffset;
        doneOffset_ = doneOffset;
        coreStatusOffset_ = coreStatusOffset;
        windowBytes_ = windowBytes;
        waitIterations_ = waitIterations;
        flags_ = flags;
        magic_ = magic;
        if (args_ == nullptr) {
            return;
        }
        rank_ = args_->rank;
        rankSize_ = args_->rankSize;
        core_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (!MoonEpCombineV2RankSizeSupported(static_cast<uint32_t>(rankSize_))) {
            return;
        }
        activeCoreCount_ = MoonEpCombineV2ActiveCoreCount(
            static_cast<uint32_t>(rankSize_));
        stepCount_ = MoonEpCombineV2StepCount(static_cast<uint32_t>(rankSize_));
        sourcesPerCore_ = static_cast<uint32_t>(rankSize_) / activeCoreCount_;
        for (int32_t peer = 0; peer < rankSize_; ++peer) {
            shareAddrs_[peer] = args_->peerMems[peer];
        }
        localWindow_ = shareAddrs_[rank_] + TileXR::IPC_DATA_OFFSET;
        valid_ = ValidateConfiguration();
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        InitBuffers();
        sync_.Init(rank_, rankSize_, shareAddrs_, syncBuf_);
        StoreCoreStatus(0);
        if (core_ == 0U) {
            StoreInt(statusAddr_, 0);
        }
        AscendC::SyncAll<true>();

        BuildDuplicateMask();
        ValidateReverseRoutes();
        AscendC::SyncAll<true>();
        if (!CrossRankBarrier(kMoonEpCombineDataReadyStep)) {
            Finish();
            return;
        }

        for (uint32_t chunk = 0U;
             chunk < static_cast<uint32_t>(chunkCount_); ++chunk) {
            const int64_t chunkOffset =
                static_cast<int64_t>(chunk) * hiddenChunkBytes_;
            const int64_t bytesThisChunk = MinInt64(
                hiddenRowBytes_ - chunkOffset, hiddenChunkBytes_);
            PrepareChunk(chunkOffset, bytesThisChunk, chunk == 0U);
            AscendC::SyncAll<true>();
            PreReduceDuplicates(bytesThisChunk);
            AscendC::SyncAll<true>();
            if (!CrossRankBarrier(kChunkReadyStepBase + chunk)) {
                Finish();
                return;
            }

            for (uint32_t step = 0U; step < stepCount_; ++step) {
                const uint32_t peer = MoonEpCombineV2Peer(
                    static_cast<uint32_t>(rank_), step, core_,
                    static_cast<uint32_t>(rankSize_));
                PushPeerRows(peer, step, chunk, bytesThisChunk, chunk == 0U);
            }
            WaitInboundDone(chunk);
            AscendC::SyncAll<true>();
            if (FirstFailure() != 0) {
                CrossRankBarrier(kChunkDrainedStepBase + chunk);
                Finish();
                return;
            }
            ReduceHiddenChunk(chunkOffset, bytesThisChunk);
            if (chunk == 0U && routeWeightsBytes_ != 0U) {
                CopyReceivedWeights();
            }
            AscendC::SyncAll<true>();
            if (!CrossRankBarrier(kChunkDrainedStepBase + chunk)) {
                Finish();
                return;
            }
        }
        Finish();
    }

private:
    __aicore__ inline bool ValidateConfiguration() const
    {
        const uint64_t hiddenEnd = receiveHiddenOffset_ + hiddenPayloadBytes_;
        const uint64_t sourceWeightsEnd = sourceWeightsOffset_ + routeWeightsBytes_;
        const uint64_t receiveWeightsEnd = receiveWeightsOffset_ + routeWeightsBytes_;
        const uint64_t maskEnd = duplicateMaskOffset_ +
            static_cast<uint64_t>(nvS_) * sizeof(int32_t);
        const uint64_t doneEnd = doneOffset_ +
            static_cast<uint64_t>(rankSize_) * kMoonEpCombineV2TokenStrideBytes;
        const uint64_t statusEnd = coreStatusOffset_ +
            static_cast<uint64_t>(activeCoreCount_) *
                kMoonEpCombineV2TokenStrideBytes;
        return dstLocalAddr_ != nullptr && dstAddr_ != nullptr &&
            dupGroupsAddr_ != nullptr && dupLoffsAddr_ != nullptr &&
            dupCountsAddr_ != nullptr && hiddenNvshAddr_ != nullptr &&
            hiddenShAddr_ != nullptr && statusAddr_ != nullptr &&
            rank_ >= 0 && rank_ < rankSize_ && core_ < activeCoreCount_ &&
            AscendC::GetBlockNum() == activeCoreCount_ && s_ > 0 &&
            k_ > 0 && k_ <= 32 && n_ == s_ * k_ && nvS_ >= n_ &&
            hiddenRowBytes_ > 0 && hiddenChunkBytes_ > 0 &&
            hiddenChunkStride_ >= hiddenChunkBytes_ &&
            hiddenChunkStride_ % static_cast<int64_t>(kMoonEpStageAlignment) == 0 &&
            chunkCount_ > 0 && chunkCount_ <= static_cast<int64_t>(kMaxChunkCount) &&
            hiddenPayloadBytes_ == static_cast<uint64_t>(nvS_) *
                static_cast<uint64_t>(hiddenChunkStride_) &&
            sourceHiddenOffset_ == 0U && receiveHiddenOffset_ >= hiddenPayloadBytes_ &&
            hiddenEnd <= windowBytes_ && sourceWeightsEnd <= windowBytes_ &&
            receiveWeightsEnd <= windowBytes_ && maskEnd <= windowBytes_ &&
            doneEnd <= windowBytes_ && statusEnd <= windowBytes_ &&
            windowBytes_ <= static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) &&
            waitIterations_ > 0 && magic_ > 0 && magic_ <= INT32_MAX &&
            flags_ == 0U &&
            ((routeWeightsBytes_ == 0U && routeWeightsNvsAddr_ == nullptr &&
                routeWeightsSkAddr_ == nullptr) ||
                (routeWeightsBytes_ == static_cast<uint64_t>(nvS_) * sizeof(float) &&
                    routeWeightsNvsAddr_ != nullptr && routeWeightsSkAddr_ != nullptr));
    }

    __aicore__ inline void InitBuffers()
    {
        pipe_.InitBuffer(syncBuf_, kMoonEpSyncUbBytes);
        pipe_.InitBuffer(bfloatBuf_, kMoonEpCombineBfloatScratchBytes);
        pipe_.InitBuffer(routeBuf_, kMoonEpCombineFloatScratchBytes);
        pipe_.InitBuffer(accumulatorBuf_, kMoonEpCombineFloatScratchBytes);
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

    __aicore__ inline void ZeroBytes(GM_ADDR address, int64_t bytes)
    {
        AscendC::LocalTensor<uint32_t> local = routeBuf_.Get<uint32_t>();
        for (int64_t offset = 0; offset < bytes;
             offset += kMoonEpCombineFloatScratchBytes) {
            const int64_t tileBytes = MinInt64(
                bytes - offset, kMoonEpCombineFloatScratchBytes);
            const int32_t words = static_cast<int32_t>((tileBytes + 3) / 4);
            AscendC::Duplicate(local, static_cast<uint32_t>(0), words);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::GlobalTensor<uint8_t> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(address + offset),
                tileBytes);
            AscendC::DataCopyExtParams params {
                1, static_cast<uint32_t>(tileBytes), 0, 0, 0};
            AscendC::DataCopyPad(dst, local.ReinterpretCast<uint8_t>(), params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreCoreStatus(int32_t status)
    {
        failureStatus_ = status;
        StoreInt(localWindow_ + coreStatusOffset_ +
            static_cast<uint64_t>(core_) * kMoonEpCombineV2TokenStrideBytes,
            status);
    }

    __aicore__ inline void Fail(int32_t status)
    {
        if (failureStatus_ == 0) {
            StoreCoreStatus(status);
        }
    }

    __aicore__ inline int32_t FirstFailure()
    {
        for (uint32_t core = 0U; core < activeCoreCount_; ++core) {
            const int32_t status = LoadInt(localWindow_ + coreStatusOffset_ +
                static_cast<uint64_t>(core) * kMoonEpCombineV2TokenStrideBytes);
            if (status != 0) {
                return status;
            }
        }
        return 0;
    }

    __aicore__ inline bool DecodeReverseRoute(
        int32_t encoded, int32_t *peer, int64_t *target) const
    {
        if (encoded == -1) {
            return false;
        }
        if (encoded < 0) {
            return false;
        }
        *peer = static_cast<int32_t>(static_cast<int64_t>(encoded) / nvS_);
        *target = static_cast<int64_t>(encoded) % nvS_;
        return *peer >= 0 && *peer < rankSize_ && *target >= 0 && *target < n_;
    }

    __aicore__ inline void BuildDuplicateMask()
    {
        const int64_t rowBegin = nvS_ * core_ / activeCoreCount_;
        const int64_t rowEnd = nvS_ * (core_ + 1U) / activeCoreCount_;
        ZeroBytes(localWindow_ + duplicateMaskOffset_ +
            rowBegin * static_cast<int64_t>(sizeof(int32_t)),
            (rowEnd - rowBegin) * static_cast<int64_t>(sizeof(int32_t)));
        AscendC::SyncAll<true>();

        const int32_t groupCount = LoadInt(dupCountsAddr_);
        const int32_t duplicateCount = LoadInt(
            dupCountsAddr_ + sizeof(int32_t));
        if (groupCount < 0 || groupCount > nvS_ || duplicateCount < 0 ||
            duplicateCount > nvS_) {
            Fail(kMoonEpCombineStatusInvalidRoute);
            return;
        }
        const int32_t groupBegin = static_cast<int32_t>(
            static_cast<int64_t>(groupCount) * core_ / activeCoreCount_);
        const int32_t groupEnd = static_cast<int32_t>(
            static_cast<int64_t>(groupCount) * (core_ + 1U) / activeCoreCount_);
        for (int32_t group = groupBegin; group < groupEnd; ++group) {
            const int64_t base = static_cast<int64_t>(group) * 3;
            const int32_t primary = LoadInt(
                dupGroupsAddr_ + base * sizeof(int32_t));
            const int32_t start = LoadInt(
                dupGroupsAddr_ + (base + 1) * sizeof(int32_t));
            const int32_t count = LoadInt(
                dupGroupsAddr_ + (base + 2) * sizeof(int32_t));
            if (primary < 0 || primary >= nvS_ || start < 0 || count <= 0 ||
                static_cast<int64_t>(start) + count > duplicateCount) {
                Fail(kMoonEpCombineStatusInvalidRoute);
                return;
            }
            for (int32_t index = 0; index < count; ++index) {
                const int32_t duplicate = LoadInt(dupLoffsAddr_ +
                    static_cast<int64_t>(start + index) * sizeof(int32_t));
                if (duplicate < 0 || duplicate >= nvS_ || duplicate == primary) {
                    Fail(kMoonEpCombineStatusInvalidRoute);
                    return;
                }
                StoreInt(localWindow_ + duplicateMaskOffset_ +
                    static_cast<int64_t>(duplicate) * sizeof(int32_t), 1);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ValidateReverseRoutes()
    {
        const int64_t begin = nvS_ * core_ / activeCoreCount_;
        const int64_t end = nvS_ * (core_ + 1U) / activeCoreCount_;
        for (int64_t row = begin; row < end; ++row) {
            const int32_t encoded = LoadInt(
                dstLocalAddr_ + row * static_cast<int64_t>(sizeof(int32_t)));
            if (encoded == -1) {
                continue;
            }
            int32_t peer = 0;
            int64_t target = 0;
            if (!DecodeReverseRoute(encoded, &peer, &target)) {
                Fail(kMoonEpCombineStatusInvalidRoute);
                return;
            }
        }
    }

    __aicore__ inline int32_t WaitPeerStep(int32_t peer, uint32_t expectedStep)
    {
        AscendC::GlobalTensor<int64_t> flag;
        flag.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(shareAddrs_[peer]),
            FLAG_UNIT_INT_NUM);
        const int64_t expectedMagic = static_cast<int64_t>(
            static_cast<int32_t>(magic_)) << MAGIC_OFFSET;
        for (uint64_t iteration = 0; iteration < waitIterations_; ++iteration) {
            AscendC::DataCacheCleanAndInvalid<int64_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(flag);
            const int64_t value = flag.GetValue(0);
            if ((value & MAGIC_MASK) != (expectedMagic & MAGIC_MASK)) {
                continue;
            }
            const int32_t step = static_cast<int32_t>(value & ~MAGIC_MASK);
            if (step == static_cast<int32_t>(expectedStep)) {
                return 0;
            }
            if (step == kMoonEpCombineFailedStep) {
                return 1;
            }
        }
        return 2;
    }

    __aicore__ inline bool CrossRankBarrier(uint32_t step)
    {
        AscendC::SyncAll<true>();
        if (core_ == 0U) {
            const int32_t localFailure = FirstFailure();
            sync_.SetInnerFlag(static_cast<int32_t>(magic_),
                localFailure == 0 ? static_cast<int32_t>(step) :
                    kMoonEpCombineFailedStep);
            if (localFailure == 0) {
                for (int32_t peer = 0; peer < rankSize_; ++peer) {
                    const int32_t result = WaitPeerStep(peer, step);
                    if (result == 1) {
                        Fail(kMoonEpCombineStatusRemoteFailureBase + peer);
                        break;
                    }
                    if (result == 2) {
                        Fail(kMoonEpCombineStatusTimeoutBase + peer);
                        break;
                    }
                }
            }
        }
        AscendC::SyncAll<true>();
        return FirstFailure() == 0;
    }

    __aicore__ inline void PrepareChunk(
        int64_t chunkOffset, int64_t bytesThisChunk, bool firstChunk)
    {
        const int64_t sourceBegin = nvS_ * core_ / activeCoreCount_;
        const int64_t sourceEnd = nvS_ * (core_ + 1U) / activeCoreCount_;
        for (int64_t row = sourceBegin; row < sourceEnd; ++row) {
            CopyBytesGmToGm(localWindow_ + sourceHiddenOffset_ +
                    row * hiddenChunkStride_,
                hiddenNvshAddr_ + row * hiddenRowBytes_ + chunkOffset,
                routeBuf_, bytesThisChunk);
        }
        const int64_t receiveBegin = n_ * core_ / activeCoreCount_;
        const int64_t receiveEnd = n_ * (core_ + 1U) / activeCoreCount_;
        ZeroBytes(localWindow_ + receiveHiddenOffset_ +
                receiveBegin * hiddenChunkStride_,
            (receiveEnd - receiveBegin) * hiddenChunkStride_);
        if (firstChunk && routeWeightsBytes_ != 0U) {
            CopyBytesGmToGm(localWindow_ + sourceWeightsOffset_ +
                    sourceBegin * static_cast<int64_t>(sizeof(float)),
                routeWeightsNvsAddr_ +
                    sourceBegin * static_cast<int64_t>(sizeof(float)),
                routeBuf_, (sourceEnd - sourceBegin) * sizeof(float));
            ZeroBytes(localWindow_ + receiveWeightsOffset_ +
                    receiveBegin * static_cast<int64_t>(sizeof(float)),
                (receiveEnd - receiveBegin) * sizeof(float));
        }
    }

    __aicore__ inline void PreReduceDuplicates(int64_t bytesThisChunk)
    {
        const int32_t groupCount = LoadInt(dupCountsAddr_);
        const int32_t duplicateCount = LoadInt(
            dupCountsAddr_ + sizeof(int32_t));
        if (failureStatus_ != 0 || groupCount <= 0) {
            return;
        }
        AscendC::LocalTensor<bfloat16_t> bfloatScratch =
            bfloatBuf_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> routeScratch = routeBuf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        const int64_t elements = bytesThisChunk / sizeof(bfloat16_t);
        const int32_t groupBegin = static_cast<int32_t>(
            static_cast<int64_t>(groupCount) * core_ / activeCoreCount_);
        const int32_t groupEnd = static_cast<int32_t>(
            static_cast<int64_t>(groupCount) * (core_ + 1U) / activeCoreCount_);
        for (int32_t group = groupBegin; group < groupEnd; ++group) {
            const int64_t base = static_cast<int64_t>(group) * 3;
            const int32_t primary = LoadInt(
                dupGroupsAddr_ + base * sizeof(int32_t));
            const int32_t start = LoadInt(
                dupGroupsAddr_ + (base + 1) * sizeof(int32_t));
            const int32_t count = LoadInt(
                dupGroupsAddr_ + (base + 2) * sizeof(int32_t));
            if (start < 0 || count <= 0 ||
                static_cast<int64_t>(start) + count > duplicateCount) {
                Fail(kMoonEpCombineStatusInvalidRoute);
                return;
            }
            for (int64_t tileOffset = 0; tileOffset < elements;
                 tileOffset += kMoonEpCombineHiddenTileElements) {
                const int64_t tileElements = MinInt64(
                    elements - tileOffset, kMoonEpCombineHiddenTileElements);
                AscendC::GlobalTensor<bfloat16_t> primaryGm;
                primaryGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                    localWindow_ + sourceHiddenOffset_ +
                        static_cast<int64_t>(primary) * hiddenChunkStride_) +
                        tileOffset, tileElements);
                AscendC::DataCopyExtParams params {1,
                    static_cast<uint32_t>(tileElements * sizeof(bfloat16_t)),
                    0, 0, 0};
                AscendC::DataCopyPadExtParams<bfloat16_t> pad {false, 0, 0, 0};
                AscendC::DataCopyPad(bfloatScratch, primaryGm, params, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(accumulator, bfloatScratch,
                    AscendC::RoundMode::CAST_NONE,
                    static_cast<int32_t>(tileElements));
                AscendC::PipeBarrier<PIPE_V>();
                for (int32_t index = 0; index < count; ++index) {
                    const int32_t duplicate = LoadInt(dupLoffsAddr_ +
                        static_cast<int64_t>(start + index) * sizeof(int32_t));
                    AscendC::GlobalTensor<bfloat16_t> duplicateGm;
                    duplicateGm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ bfloat16_t *>(localWindow_ +
                            sourceHiddenOffset_ +
                            static_cast<int64_t>(duplicate) * hiddenChunkStride_) +
                            tileOffset, tileElements);
                    AscendC::DataCopyPad(bfloatScratch, duplicateGm, params, pad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(routeScratch, bfloatScratch,
                        AscendC::RoundMode::CAST_NONE,
                        static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(accumulator, accumulator, routeScratch,
                        static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Cast(bfloatScratch, accumulator,
                    AscendC::RoundMode::CAST_RINT,
                    static_cast<int32_t>(tileElements));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(primaryGm, bfloatScratch, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint64_t DoneToken(uint32_t chunk, uint32_t step) const
    {
        return (static_cast<uint64_t>(static_cast<uint32_t>(magic_)) << 32U) |
            (static_cast<uint64_t>(chunk) << 8U) | step;
    }

    __aicore__ inline void StoreToken(GM_ADDR address, uint64_t value)
    {
        AscendC::LocalTensor<uint64_t> local = bfloatBuf_.Get<uint64_t>();
        AscendC::GlobalTensor<uint64_t> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(address), 1);
        local.SetValue(0, value);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams params {1, sizeof(uint64_t), 0, 0, 0};
        AscendC::DataCopyPad(dst, local, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void PushPeerRows(uint32_t peer, uint32_t step,
        uint32_t chunk, int64_t bytesThisChunk, bool firstChunk)
    {
        GM_ADDR remoteWindow = shareAddrs_[peer] + TileXR::IPC_DATA_OFFSET;
        for (int64_t source = 0; source < nvS_; ++source) {
            const int32_t encoded = LoadInt(dstLocalAddr_ +
                source * static_cast<int64_t>(sizeof(int32_t)));
            if (encoded == -1) {
                continue;
            }
            int32_t targetPeer = 0;
            int64_t target = 0;
            if (!DecodeReverseRoute(encoded, &targetPeer, &target) ||
                targetPeer != static_cast<int32_t>(peer)) {
                continue;
            }
            const int32_t duplicate = LoadInt(localWindow_ + duplicateMaskOffset_ +
                source * static_cast<int64_t>(sizeof(int32_t)));
            if (duplicate == 0) {
                CopyBytesGmToGm(remoteWindow + receiveHiddenOffset_ +
                        target * hiddenChunkStride_,
                    localWindow_ + sourceHiddenOffset_ +
                        source * hiddenChunkStride_,
                    routeBuf_, bytesThisChunk);
            }
            if (firstChunk && routeWeightsBytes_ != 0U) {
                CopyBytesGmToGm(remoteWindow + receiveWeightsOffset_ +
                        target * static_cast<int64_t>(sizeof(float)),
                    localWindow_ + sourceWeightsOffset_ +
                        source * static_cast<int64_t>(sizeof(float)),
                    routeBuf_, sizeof(float));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreToken(remoteWindow + doneOffset_ +
            static_cast<uint64_t>(rank_) * kMoonEpCombineV2TokenStrideBytes,
            DoneToken(chunk, step));
    }

    __aicore__ inline uint64_t LoadToken(GM_ADDR address)
    {
        AscendC::GlobalTensor<uint64_t> token;
        token.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(address), 1);
        AscendC::DataCacheCleanAndInvalid<uint64_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(token);
        return token.GetValue(0);
    }

    __aicore__ inline bool WaitInboundDone(uint32_t chunk)
    {
        for (uint32_t sourceIndex = 0U;
             sourceIndex < sourcesPerCore_; ++sourceIndex) {
            const uint32_t source = MoonEpCombineV2SourceForCore(
                core_, sourceIndex, static_cast<uint32_t>(rankSize_));
            const uint32_t step = MoonEpCombineV2ReceiveStep(
                static_cast<uint32_t>(rank_), source,
                static_cast<uint32_t>(rankSize_));
            const uint64_t expected = DoneToken(chunk, step);
            GM_ADDR address = localWindow_ + doneOffset_ +
                static_cast<uint64_t>(source) *
                    kMoonEpCombineV2TokenStrideBytes;
            bool ready = false;
            for (uint64_t iteration = 0; iteration < waitIterations_; ++iteration) {
                if (LoadToken(address) == expected) {
                    ready = true;
                    break;
                }
            }
            if (!ready) {
                Fail(kMoonEpCombineStatusTimeoutBase +
                    static_cast<int32_t>(source));
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void ReduceHiddenChunk(
        int64_t chunkOffset, int64_t bytesThisChunk)
    {
        if (failureStatus_ != 0) {
            return;
        }
        const int64_t elements = bytesThisChunk / sizeof(bfloat16_t);
        AscendC::LocalTensor<bfloat16_t> bfloatScratch =
            bfloatBuf_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> routeScratch = routeBuf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        const int64_t tokenBegin = s_ * core_ / activeCoreCount_;
        const int64_t tokenEnd = s_ * (core_ + 1U) / activeCoreCount_;
        for (int64_t token = tokenBegin; token < tokenEnd; ++token) {
            for (int64_t tileOffset = 0; tileOffset < elements;
                 tileOffset += kMoonEpCombineHiddenTileElements) {
                const int64_t tileElements = MinInt64(
                    elements - tileOffset, kMoonEpCombineHiddenTileElements);
                AscendC::Duplicate(accumulator, 0.0f,
                    static_cast<int32_t>(tileElements));
                AscendC::PipeBarrier<PIPE_V>();
                for (int64_t topk = 0; topk < k_; ++topk) {
                    const int64_t route = token * k_ + topk;
                    AscendC::GlobalTensor<bfloat16_t> input;
                    input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                        localWindow_ + receiveHiddenOffset_ +
                            route * hiddenChunkStride_) + tileOffset,
                        tileElements);
                    AscendC::DataCopyExtParams params {1,
                        static_cast<uint32_t>(
                            tileElements * sizeof(bfloat16_t)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<bfloat16_t> pad {
                        false, 0, 0, 0};
                    AscendC::DataCopyPad(bfloatScratch, input, params, pad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(routeScratch, bfloatScratch,
                        AscendC::RoundMode::CAST_NONE,
                        static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(accumulator, accumulator, routeScratch,
                        static_cast<int32_t>(tileElements));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Cast(bfloatScratch, accumulator,
                    AscendC::RoundMode::CAST_RINT,
                    static_cast<int32_t>(tileElements));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::GlobalTensor<bfloat16_t> output;
                output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                    hiddenShAddr_ + token * hiddenRowBytes_ + chunkOffset) +
                        tileOffset, tileElements);
                AscendC::DataCopyExtParams params {1,
                    static_cast<uint32_t>(
                        tileElements * sizeof(bfloat16_t)), 0, 0, 0};
                AscendC::DataCopyPad(output, bfloatScratch, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyReceivedWeights()
    {
        const int64_t begin = n_ * core_ / activeCoreCount_;
        const int64_t end = n_ * (core_ + 1U) / activeCoreCount_;
        CopyBytesGmToGm(routeWeightsSkAddr_ +
                begin * static_cast<int64_t>(sizeof(float)),
            localWindow_ + receiveWeightsOffset_ +
                begin * static_cast<int64_t>(sizeof(float)),
            routeBuf_, (end - begin) * sizeof(float));
    }

    __aicore__ inline void Finish()
    {
        AscendC::SyncAll<true>();
        if (core_ == 0U) {
            StoreInt(statusAddr_, FirstFailure());
        }
        AscendC::SyncAll<true>();
    }

    __gm__ TileXR::CommArgs *args_ = nullptr;
    bool valid_ = false;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    uint32_t core_ = 0U;
    uint32_t activeCoreCount_ = 0U;
    uint32_t stepCount_ = 0U;
    uint32_t sourcesPerCore_ = 0U;
    int32_t failureStatus_ = 0;
    int64_t s_ = 0;
    int64_t k_ = 0;
    int64_t n_ = 0;
    int64_t nvS_ = 0;
    int64_t hiddenRowBytes_ = 0;
    int64_t hiddenChunkBytes_ = 0;
    int64_t hiddenChunkStride_ = 0;
    int64_t chunkCount_ = 0;
    uint64_t sourceHiddenOffset_ = 0U;
    uint64_t receiveHiddenOffset_ = 0U;
    uint64_t hiddenPayloadBytes_ = 0U;
    uint64_t sourceWeightsOffset_ = 0U;
    uint64_t receiveWeightsOffset_ = 0U;
    uint64_t routeWeightsBytes_ = 0U;
    uint64_t duplicateMaskOffset_ = 0U;
    uint64_t doneOffset_ = 0U;
    uint64_t coreStatusOffset_ = 0U;
    uint64_t windowBytes_ = 0U;
    uint64_t waitIterations_ = 0U;
    uint64_t flags_ = 0U;
    int64_t magic_ = 0;
    GM_ADDR dstLocalAddr_ = nullptr;
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

extern "C" __global__ __aicore__ void tilexr_moonep_combine_kernel(
    GM_ADDR commArgs, GM_ADDR dstLocal, GM_ADDR dst, GM_ADDR dupGroups,
    GM_ADDR dupLoffs, GM_ADDR dupCounts, GM_ADDR hiddenNvsh,
    GM_ADDR routeWeightsNvs, GM_ADDR hiddenSh, GM_ADDR routeWeightsSk,
    GM_ADDR status, int64_t s, int64_t k, int64_t n, int64_t nvS,
    int64_t hiddenRowBytes, int64_t hiddenChunkBytes,
    int64_t hiddenChunkStride, int64_t chunkCount,
    uint64_t sourceHiddenOffset, uint64_t receiveHiddenOffset,
    uint64_t hiddenPayloadBytes, uint64_t sourceWeightsOffset,
    uint64_t receiveWeightsOffset, uint64_t routeWeightsBytes,
    uint64_t duplicateMaskOffset, uint64_t doneOffset,
    uint64_t coreStatusOffset, uint64_t windowBytes,
    uint64_t waitIterations, uint64_t flags, int64_t magic)
{
    TileXRMoonEp::Kernel::CombineKernel op;
    op.Init(commArgs, dstLocal, dst, dupGroups, dupLoffs, dupCounts,
        hiddenNvsh, routeWeightsNvs, hiddenSh, routeWeightsSk, status,
        s, k, n, nvS, hiddenRowBytes, hiddenChunkBytes,
        hiddenChunkStride, chunkCount, sourceHiddenOffset,
        receiveHiddenOffset, hiddenPayloadBytes, sourceWeightsOffset,
        receiveWeightsOffset, routeWeightsBytes, duplicateMaskOffset,
        doneOffset, coreStatusOffset, windowBytes, waitIterations, flags,
        magic);
    op.Process();
}
