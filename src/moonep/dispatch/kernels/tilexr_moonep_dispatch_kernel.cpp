#include "kernel_operator.h"

#include <cstdint>

#include "comm_args.h"
#include "dispatch_common.h"
#include "tilexr_sync.h"

namespace TileXRMoonEp {
namespace Kernel {

constexpr int64_t kClearTileElements = 16 * 1024;

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void CopyBytesGmToGm(GM_ADDR dstAddr, GM_ADDR srcAddr,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &workBuf, int64_t bytes)
{
    if (dstAddr == nullptr || srcAddr == nullptr || bytes <= 0) {
        return;
    }
    AscendC::LocalTensor<uint8_t> local = workBuf.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(srcAddr), bytes);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dstAddr), bytes);

    for (int64_t copied = 0; copied < bytes; copied += kMoonEpWorkUbBytes) {
        const int64_t tileBytes = MinInt64(bytes - copied, kMoonEpWorkUbBytes);
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

class DispatchKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR dst, GM_ADDR zeroFillRanges,
        GM_ADDR dupGroups, GM_ADDR dupLoffs, GM_ADDR dupCounts, GM_ADDR hiddenSh,
        GM_ADDR routeWeightsSk, GM_ADDR hiddenNvsh, GM_ADDR routeWeightsNvs,
        GM_ADDR status, int64_t s, int64_t k, int64_t n, int64_t nvS,
        int64_t hiddenRowBytes, int64_t hiddenChunkBytes, int64_t hiddenChunkStride,
        int64_t chunkCount, uint64_t hiddenPayloadBytes, uint64_t routeWeightsOffset,
        uint64_t routeWeightsBytes, uint64_t dedupParentsOffset,
        uint64_t dedupParentsBytes, uint64_t dedupGroupMapOffset,
        uint64_t dedupGroupMapBytes, uint64_t waitIterations, uint64_t flags,
        int64_t magic)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
        dstAddr_ = dst;
        zeroFillRangesAddr_ = zeroFillRanges;
        dupGroupsAddr_ = dupGroups;
        dupLoffsAddr_ = dupLoffs;
        dupCountsAddr_ = dupCounts;
        hiddenShAddr_ = hiddenSh;
        routeWeightsSkAddr_ = routeWeightsSk;
        hiddenNvshAddr_ = hiddenNvsh;
        routeWeightsNvsAddr_ = routeWeightsNvs;
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
        dedupParentsOffset_ = dedupParentsOffset;
        dedupParentsBytes_ = dedupParentsBytes;
        dedupGroupMapOffset_ = dedupGroupMapOffset;
        dedupGroupMapBytes_ = dedupGroupMapBytes;
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
        for (int64_t chunk = 0; chunk < chunkCount_; ++chunk) {
            const int64_t chunkOffset = chunk * hiddenChunkBytes_;
            const int64_t bytesThisChunk = MinInt64(
                hiddenRowBytes_ - chunkOffset, hiddenChunkBytes_);
            ClearLocalWindow();
            PublishStep(ChunkStep(kMoonEpDispatchWindowClearedStep, chunk));
            if (!WaitAllPeers(ChunkStep(kMoonEpDispatchWindowClearedStep, chunk))) {
                return;
            }

            if (!ScatterRoutes(chunkOffset, bytesThisChunk, chunk == 0)) {
                return;
            }
            PublishStep(ChunkStep(kMoonEpDispatchDataReadyStep, chunk));
            if (!WaitAllPeers(ChunkStep(kMoonEpDispatchDataReadyStep, chunk))) {
                return;
            }

            if (chunk == 0 && BuildDedup() && !BuildDuplicatePlan()) {
                return;
            }
            if (!ExpandDuplicateRows(bytesThisChunk) ||
                !DrainHidden(chunkOffset, bytesThisChunk)) {
                return;
            }
            if (chunk == 0 && routeWeightsBytes_ > 0) {
                CopyBytesGmToGm(routeWeightsNvsAddr_,
                    localWindow_ + static_cast<int64_t>(routeWeightsOffset_), workBuf_,
                    static_cast<int64_t>(routeWeightsBytes_));
            }
            PublishStep(ChunkStep(kMoonEpDispatchWindowDrainedStep, chunk));
            if (!WaitAllPeers(ChunkStep(kMoonEpDispatchWindowDrainedStep, chunk))) {
                return;
            }
        }
        StoreStatus(kMoonEpDispatchStatusSuccess);
    }

private:
    __aicore__ inline bool Valid() const
    {
        const uint64_t allowedFlags = kMoonEpFlagBuildDedup |
            kMoonEpFlagSkipInterRankSync;
        uint64_t windowBytes = hiddenPayloadBytes_;
        const uint64_t weightsEnd = routeWeightsOffset_ + routeWeightsBytes_;
        const uint64_t parentsEnd = dedupParentsOffset_ + dedupParentsBytes_;
        const uint64_t groupMapEnd = dedupGroupMapOffset_ + dedupGroupMapBytes_;
        windowBytes = weightsEnd > windowBytes ? weightsEnd : windowBytes;
        windowBytes = parentsEnd > windowBytes ? parentsEnd : windowBytes;
        windowBytes = groupMapEnd > windowBytes ? groupMapEnd : windowBytes;
        return initialized_ && args_ != nullptr && dstAddr_ != nullptr &&
            zeroFillRangesAddr_ != nullptr && dupGroupsAddr_ != nullptr &&
            dupLoffsAddr_ != nullptr && dupCountsAddr_ != nullptr &&
            hiddenShAddr_ != nullptr && hiddenNvshAddr_ != nullptr && statusAddr_ != nullptr &&
            rank_ >= 0 && rank_ < rankSize_ && s_ > 0 && k_ > 0 && k_ <= 32 &&
            n_ == s_ * k_ && nvS_ >= n_ && hiddenRowBytes_ > 0 &&
            hiddenChunkBytes_ > 0 && hiddenChunkStride_ >= hiddenChunkBytes_ &&
            hiddenChunkStride_ % static_cast<int64_t>(kMoonEpStageAlignment) == 0 &&
            chunkCount_ > 0 && hiddenPayloadBytes_ ==
                static_cast<uint64_t>(nvS_) * static_cast<uint64_t>(hiddenChunkStride_) &&
            windowBytes <= static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) &&
            waitIterations_ > 0 && magic_ > 0 && (flags_ & ~allowedFlags) == 0 &&
            ((routeWeightsBytes_ == 0 && routeWeightsSkAddr_ == nullptr &&
                routeWeightsNvsAddr_ == nullptr) ||
                (routeWeightsBytes_ == static_cast<uint64_t>(nvS_) * sizeof(float) &&
                    routeWeightsSkAddr_ != nullptr && routeWeightsNvsAddr_ != nullptr)) &&
            ((!BuildDedup() && dedupParentsBytes_ == 0 && dedupGroupMapBytes_ == 0) ||
                (BuildDedup() && dedupParentsBytes_ ==
                    static_cast<uint64_t>(nvS_) * sizeof(int32_t) &&
                    dedupGroupMapBytes_ == dedupParentsBytes_));
    }

    __aicore__ inline bool BuildDedup() const
    {
        return (flags_ & kMoonEpFlagBuildDedup) != 0;
    }

    __aicore__ inline int32_t ChunkStep(int32_t base, int64_t chunk) const
    {
        return base + static_cast<int32_t>(chunk * 4);
    }

    __aicore__ inline void ClearLocalWindow()
    {
        uint64_t windowBytes = hiddenPayloadBytes_;
        const uint64_t weightsEnd = routeWeightsOffset_ + routeWeightsBytes_;
        const uint64_t parentsEnd = dedupParentsOffset_ + dedupParentsBytes_;
        const uint64_t groupMapEnd = dedupGroupMapOffset_ + dedupGroupMapBytes_;
        windowBytes = weightsEnd > windowBytes ? weightsEnd : windowBytes;
        windowBytes = parentsEnd > windowBytes ? parentsEnd : windowBytes;
        windowBytes = groupMapEnd > windowBytes ? groupMapEnd : windowBytes;

        AscendC::LocalTensor<uint16_t> zeros = workBuf_.Get<uint16_t>();
        AscendC::GlobalTensor<uint16_t> local;
        const int64_t elements = static_cast<int64_t>(windowBytes / sizeof(uint16_t));
        local.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(localWindow_), elements);
        for (int64_t cleared = 0; cleared < elements; cleared += kClearTileElements) {
            const int64_t tile = MinInt64(elements - cleared, kClearTileElements);
            AscendC::Duplicate(zeros, static_cast<uint16_t>(0), static_cast<int32_t>(tile));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams params {
                1, static_cast<uint32_t>(tile * sizeof(uint16_t)), 0, 0, 0
            };
            AscendC::DataCopyPad(local[cleared], zeros, params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
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
            Fail(kMoonEpDispatchStatusInvalidRoute);
            return false;
        }
        return true;
    }

    __aicore__ inline int64_t FindPrimaryOffset(int64_t route, int64_t destRank)
    {
        const int64_t tokenStart = (route / k_) * k_;
        for (int64_t candidate = tokenStart; candidate < route; ++candidate) {
            const int32_t encoded = LoadRoute(candidate);
            if (encoded < 0) {
                continue;
            }
            int64_t peer = 0;
            int64_t offset = 0;
            if (!DecodeRoute(encoded, peer, offset)) {
                return -1;
            }
            if (peer == destRank) {
                return offset;
            }
        }
        return -1;
    }

    __aicore__ inline bool ScatterRoutes(
        int64_t chunkOffset, int64_t bytesThisChunk, bool firstChunk)
    {
        for (int64_t route = 0; route < n_; ++route) {
            const int32_t encoded = LoadRoute(route);
            int64_t destRank = 0;
            int64_t destOffset = 0;
            if (!DecodeRoute(encoded, destRank, destOffset)) {
                return false;
            }
            GM_ADDR remoteWindow = shareAddrs_[destRank] + TileXR::IPC_DATA_OFFSET;
            if (encoded >= 0) {
                const int64_t token = route / k_;
                CopyBytesGmToGm(remoteWindow + destOffset * hiddenChunkStride_,
                    hiddenShAddr_ + token * hiddenRowBytes_ + chunkOffset,
                    workBuf_, bytesThisChunk);
            }
            if (firstChunk && routeWeightsBytes_ > 0) {
                CopyBytesGmToGm(remoteWindow + static_cast<int64_t>(routeWeightsOffset_) +
                        destOffset * static_cast<int64_t>(sizeof(float)),
                    routeWeightsSkAddr_ + route * static_cast<int64_t>(sizeof(float)),
                    workBuf_, sizeof(float));
            }
            if (firstChunk && BuildDedup() && encoded < 0) {
                const int64_t primary = FindPrimaryOffset(route, destRank);
                if (primary < 0 || primary >= nvS_) {
                    Fail(kMoonEpDispatchStatusInvalidRoute);
                    return false;
                }
                StoreInt(remoteWindow + static_cast<int64_t>(dedupParentsOffset_) +
                    destOffset * static_cast<int64_t>(sizeof(int32_t)),
                    static_cast<int32_t>(primary + 1));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline bool BuildDuplicatePlan()
    {
        int32_t groupCount = 0;
        int32_t duplicateCount = 0;
        StoreInt(dupCountsAddr_, 0);
        StoreInt(dupCountsAddr_ + sizeof(int32_t), 0);

        // First count each primary's duplicates. The loff scan interleaves groups,
        // so writing dupLoffs here would not produce the contiguous slices in Plan V1.
        for (int64_t loff = 0; loff < nvS_; ++loff) {
            const int32_t parent = LoadInt(localWindow_ +
                static_cast<int64_t>(dedupParentsOffset_) + loff * sizeof(int32_t));
            if (parent == 0) {
                continue;
            }
            const int64_t primary = static_cast<int64_t>(parent) - 1;
            if (primary < 0 || primary >= nvS_ || duplicateCount >= nvS_) {
                Fail(kMoonEpDispatchStatusInvalidRoute);
                return false;
            }
            GM_ADDR mapAddress = localWindow_ + static_cast<int64_t>(dedupGroupMapOffset_) +
                primary * sizeof(int32_t);
            int32_t groupMarker = LoadInt(mapAddress);
            int32_t group = groupMarker - 1;
            if (groupMarker == 0) {
                if (groupCount >= nvS_) {
                    Fail(kMoonEpDispatchStatusInvalidRoute);
                    return false;
                }
                group = groupCount++;
                StoreInt(dupGroupsAddr_ + (static_cast<int64_t>(group) * 3) * sizeof(int32_t),
                    static_cast<int32_t>(primary));
                StoreInt(dupGroupsAddr_ + (static_cast<int64_t>(group) * 3 + 1) *
                    sizeof(int32_t), 0);
                StoreInt(dupGroupsAddr_ + (static_cast<int64_t>(group) * 3 + 2) *
                    sizeof(int32_t), 0);
                StoreInt(mapAddress, group + 1);
            }
            GM_ADDR countAddress = dupGroupsAddr_ +
                (static_cast<int64_t>(group) * 3 + 2) * sizeof(int32_t);
            const int32_t count = LoadInt(countAddress);
            StoreInt(countAddress, count + 1);
            ++duplicateCount;
        }

        int32_t prefix = 0;
        for (int32_t group = 0; group < groupCount; ++group) {
            GM_ADDR startAddress = dupGroupsAddr_ +
                (static_cast<int64_t>(group) * 3 + 1) * sizeof(int32_t);
            GM_ADDR countAddress = dupGroupsAddr_ +
                (static_cast<int64_t>(group) * 3 + 2) * sizeof(int32_t);
            const int32_t count = LoadInt(countAddress);
            if (count <= 0 || prefix > nvS_ - count) {
                Fail(kMoonEpDispatchStatusInvalidRoute);
                return false;
            }
            StoreInt(startAddress, prefix);
            StoreInt(countAddress, 0);
            prefix += count;
        }
        if (prefix != duplicateCount) {
            Fail(kMoonEpDispatchStatusInvalidRoute);
            return false;
        }

        // Fill the prefix-partitioned slices, reusing each group's count as its cursor.
        for (int64_t loff = 0; loff < nvS_; ++loff) {
            const int32_t parent = LoadInt(localWindow_ +
                static_cast<int64_t>(dedupParentsOffset_) + loff * sizeof(int32_t));
            if (parent == 0) {
                continue;
            }
            const int64_t primary = static_cast<int64_t>(parent) - 1;
            const int32_t groupMarker = LoadInt(localWindow_ +
                static_cast<int64_t>(dedupGroupMapOffset_) + primary * sizeof(int32_t));
            const int32_t group = groupMarker - 1;
            if (group < 0 || group >= groupCount) {
                Fail(kMoonEpDispatchStatusInvalidRoute);
                return false;
            }
            GM_ADDR startAddress = dupGroupsAddr_ +
                (static_cast<int64_t>(group) * 3 + 1) * sizeof(int32_t);
            GM_ADDR countAddress = dupGroupsAddr_ +
                (static_cast<int64_t>(group) * 3 + 2) * sizeof(int32_t);
            const int32_t start = LoadInt(startAddress);
            const int32_t count = LoadInt(countAddress);
            const int64_t writeIndex = static_cast<int64_t>(start) + count;
            if (start < 0 || count < 0 || writeIndex >= duplicateCount) {
                Fail(kMoonEpDispatchStatusInvalidRoute);
                return false;
            }
            StoreInt(dupLoffsAddr_ + writeIndex * sizeof(int32_t),
                static_cast<int32_t>(loff));
            StoreInt(countAddress, count + 1);
        }
        StoreInt(dupCountsAddr_, groupCount);
        StoreInt(dupCountsAddr_ + sizeof(int32_t), duplicateCount);
        AscendC::PipeBarrier<PIPE_ALL>();
        return true;
    }

    __aicore__ inline bool ExpandDuplicateRows(int64_t bytesThisChunk)
    {
        const int32_t groupCount = LoadInt(dupCountsAddr_);
        const int32_t duplicateCount = LoadInt(dupCountsAddr_ + sizeof(int32_t));
        if (groupCount < 0 || groupCount > nvS_ || duplicateCount < 0 ||
            duplicateCount > nvS_) {
            Fail(kMoonEpDispatchStatusInvalidRoute);
            return false;
        }
        for (int32_t group = 0; group < groupCount; ++group) {
            const int64_t base = static_cast<int64_t>(group) * 3;
            const int32_t primary = LoadInt(dupGroupsAddr_ + base * sizeof(int32_t));
            const int32_t start = LoadInt(dupGroupsAddr_ + (base + 1) * sizeof(int32_t));
            const int32_t count = LoadInt(dupGroupsAddr_ + (base + 2) * sizeof(int32_t));
            if (primary < 0 || primary >= nvS_ || start < 0 || count <= 0 ||
                static_cast<int64_t>(start) + count > duplicateCount) {
                Fail(kMoonEpDispatchStatusInvalidRoute);
                return false;
            }
            for (int32_t index = 0; index < count; ++index) {
                const int32_t duplicate = LoadInt(dupLoffsAddr_ +
                    static_cast<int64_t>(start + index) * sizeof(int32_t));
                if (duplicate < 0 || duplicate >= nvS_) {
                    Fail(kMoonEpDispatchStatusInvalidRoute);
                    return false;
                }
                CopyBytesGmToGm(localWindow_ +
                        static_cast<int64_t>(duplicate) * hiddenChunkStride_,
                    localWindow_ + static_cast<int64_t>(primary) * hiddenChunkStride_,
                    workBuf_, bytesThisChunk);
            }
        }
        return true;
    }

    __aicore__ inline bool DrainHidden(int64_t chunkOffset, int64_t bytesThisChunk)
    {
        for (int64_t row = 0; row < nvS_; ++row) {
            CopyBytesGmToGm(hiddenNvshAddr_ + row * hiddenRowBytes_ + chunkOffset,
                localWindow_ + row * hiddenChunkStride_, workBuf_, bytesThisChunk);
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
            if (step == kMoonEpDispatchFailedStep) {
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
                Fail(kMoonEpDispatchStatusRemoteFailureBase + peer);
                return false;
            }
            if (result == 2) {
                Fail(kMoonEpDispatchStatusTimeoutBase + peer);
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void Fail(int32_t status)
    {
        StoreStatus(status);
        PublishStep(kMoonEpDispatchFailedStep);
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
    uint64_t dedupParentsOffset_ = 0;
    uint64_t dedupParentsBytes_ = 0;
    uint64_t dedupGroupMapOffset_ = 0;
    uint64_t dedupGroupMapBytes_ = 0;
    uint64_t waitIterations_ = 0;
    uint64_t flags_ = 0;
    int64_t magic_ = 0;
    bool initialized_ = false;
    GM_ADDR dstAddr_ = nullptr;
    GM_ADDR zeroFillRangesAddr_ = nullptr;
    GM_ADDR dupGroupsAddr_ = nullptr;
    GM_ADDR dupLoffsAddr_ = nullptr;
    GM_ADDR dupCountsAddr_ = nullptr;
    GM_ADDR hiddenShAddr_ = nullptr;
    GM_ADDR routeWeightsSkAddr_ = nullptr;
    GM_ADDR hiddenNvshAddr_ = nullptr;
    GM_ADDR routeWeightsNvsAddr_ = nullptr;
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

extern "C" __global__ __aicore__ void tilexr_moonep_dispatch_kernel(GM_ADDR commArgs,
    GM_ADDR dst, GM_ADDR zeroFillRanges, GM_ADDR dupGroups, GM_ADDR dupLoffs,
    GM_ADDR dupCounts, GM_ADDR hiddenSh, GM_ADDR routeWeightsSk, GM_ADDR hiddenNvsh,
    GM_ADDR routeWeightsNvs, GM_ADDR status, int64_t s, int64_t k, int64_t n,
    int64_t nvS, int64_t hiddenRowBytes, int64_t hiddenChunkBytes,
    int64_t hiddenChunkStride, int64_t chunkCount, uint64_t hiddenPayloadBytes,
    uint64_t routeWeightsOffset, uint64_t routeWeightsBytes,
    uint64_t dedupParentsOffset, uint64_t dedupParentsBytes,
    uint64_t dedupGroupMapOffset, uint64_t dedupGroupMapBytes,
    uint64_t waitIterations, uint64_t flags, int64_t magic)
{
    TileXRMoonEp::Kernel::DispatchKernel op;
    op.Init(commArgs, dst, zeroFillRanges, dupGroups, dupLoffs, dupCounts,
        hiddenSh, routeWeightsSk, hiddenNvsh, routeWeightsNvs, status, s, k, n, nvS,
        hiddenRowBytes, hiddenChunkBytes, hiddenChunkStride, chunkCount,
        hiddenPayloadBytes, routeWeightsOffset, routeWeightsBytes, dedupParentsOffset,
        dedupParentsBytes, dedupGroupMapOffset, dedupGroupMapBytes, waitIterations,
        flags, magic);
    op.Process();
}
