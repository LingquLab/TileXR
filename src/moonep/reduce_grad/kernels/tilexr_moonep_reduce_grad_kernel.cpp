#include "kernel_operator.h"
#include <cstdint>

#include "comm_args.h"
#include "reduce_grad_common.h"
#include "tilexr_data_as_flag.h"
#include "tilexr_udma.h"

namespace TileXRMoonEp {
namespace Kernel {

constexpr uint32_t kPeerBatchRecords = 64;
constexpr uint32_t kIoBufferBytes = 32 * 1024;
constexpr uint32_t kFlagBufferBytes = 32 * 1024;
constexpr uint32_t kClearBufferBytes = 4 * 1024;
constexpr uint32_t kUdmaPollBackoffMax = 256;

template <AscendC::HardEvent event>
__aicore__ inline void SyncEvent()
{
    const AscendC::TEventID eventId = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + (value % divisor == 0 ? 0 : 1);
}

class ReduceGradKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertsToCopy,
        GM_ADDR gate, GM_ADDR up, GM_ADDR down, GM_ADDR workspace, GM_ADDR status,
        int64_t rank, int64_t rankSize, int64_t expertCount,
        int64_t expertsPerRank, int64_t prefetchSlots, int64_t controlBlockCount,
        uint64_t gateRowElements, uint64_t upRowElements,
        uint64_t downRowElements, uint64_t gateRowBytes, uint64_t upRowBytes,
        uint64_t downRowBytes, uint32_t gateTransport, uint32_t upTransport,
        uint32_t downTransport, uint32_t udmaQpCount, uint64_t peerRecordBaseOffset,
        uint64_t peerHalfBytes, uint64_t peerSlotStrideBytes,
        uint64_t peerChunkPayloadBytes, uint64_t udmaStateOffset,
        uint64_t udmaOutboundOffset, uint64_t udmaInboundOffset,
        uint64_t udmaChunkBytes, uint64_t workspaceBytes,
        uint64_t waitIterations, int64_t magic)
    {
        args_ = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
        expertsToCopy_ = reinterpret_cast<__gm__ int32_t *>(expertsToCopy);
        gradients_[kReduceGradGate] = gate;
        gradients_[kReduceGradUp] = up;
        gradients_[kReduceGradDown] = down;
        workspace_ = workspace;
        status_ = reinterpret_cast<__gm__ int32_t *>(status);
        rank_ = rank;
        rankSize_ = rankSize;
        expertCount_ = expertCount;
        expertsPerRank_ = expertsPerRank;
        prefetchSlots_ = prefetchSlots;
        controlBlockCount_ = controlBlockCount;
        rowElements_[kReduceGradGate] = gateRowElements;
        rowElements_[kReduceGradUp] = upRowElements;
        rowElements_[kReduceGradDown] = downRowElements;
        rowBytes_[kReduceGradGate] = gateRowBytes;
        rowBytes_[kReduceGradUp] = upRowBytes;
        rowBytes_[kReduceGradDown] = downRowBytes;
        transports_[kReduceGradGate] = gateTransport;
        transports_[kReduceGradUp] = upTransport;
        transports_[kReduceGradDown] = downTransport;
        udmaQpCount_ = udmaQpCount;
        peerRecordBaseOffset_ = peerRecordBaseOffset;
        peerHalfBytes_ = peerHalfBytes;
        peerSlotStrideBytes_ = peerSlotStrideBytes;
        peerChunkPayloadBytes_ = peerChunkPayloadBytes;
        udmaStateOffset_ = udmaStateOffset;
        udmaOutboundOffset_ = udmaOutboundOffset;
        udmaInboundOffset_ = udmaInboundOffset;
        udmaChunkBytes_ = udmaChunkBytes;
        workspaceBytes_ = workspaceBytes;
        waitIterations_ = waitIterations;
        magic_ = magic;

        const int64_t subBlockCount = static_cast<int64_t>(get_subblockdim());
        blockIdx_ = static_cast<int64_t>(get_block_idx()) * subBlockCount +
            static_cast<int64_t>(get_subblockid());
        blockCount_ = static_cast<int64_t>(get_block_num()) * subBlockCount;
        receiverCount_ = blockCount_ - controlBlockCount_;

        pipe_.InitBuffer(ioBuf_, kIoBufferBytes);
        pipe_.InitBuffer(accumBuf_, kIoBufferBytes);
        pipe_.InitBuffer(flagBuf_, kFlagBufferBytes);
        pipe_.InitBuffer(clearBuf_, kClearBufferBytes);
        pipe_.InitBuffer(udmaWqeBuf_, TileXR::TILEXR_UDMA_WQE_SCRATCH_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (args_ == nullptr || expertsToCopy_ == nullptr || status_ == nullptr ||
            rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || expertCount_ <= 0 ||
            expertsPerRank_ <= 0 || prefetchSlots_ <= 0 ||
            prefetchSlots_ > expertsPerRank_ || controlBlockCount_ < 0 ||
            (rankSize_ > 1 && controlBlockCount_ == 0) || receiverCount_ <= 0 ||
            magic_ <= 0 ||
            (UsesUdma() && (udmaQpCount_ == 0 ||
                udmaQpCount_ > kReduceGradMaxUdmaQpCount ||
                TileXR::UDMAQpCount(args_) != udmaQpCount_))) {
            SetStatus(kReduceGradDeviceInvalidState);
            return;
        }
        if (blockIdx_ < controlBlockCount_) {
            RunSender();
        } else {
            RunReceiver();
        }
    }

private:
    __aicore__ inline void SetStatus(int32_t value)
    {
        status_[0] = value;
    }

    __aicore__ inline int32_t ExpertForSlot(int64_t sourceRank, int64_t slot) const
    {
        return expertsToCopy_[sourceRank * prefetchSlots_ + slot];
    }

    __aicore__ inline int64_t OwnerForExpert(int32_t expert) const
    {
        return expert < 0 ? -1 : static_cast<int64_t>(expert) / expertsPerRank_;
    }

    __aicore__ inline uint64_t ChunkBytes(uint32_t projection) const
    {
        return transports_[projection] == kReduceGradTransportPeer ?
            peerChunkPayloadBytes_ : udmaChunkBytes_;
    }

    __aicore__ inline uint64_t ChunkCount(uint32_t projection) const
    {
        return CeilDivU64(rowBytes_[projection], ChunkBytes(projection));
    }

    __aicore__ inline uint64_t Sequence(uint32_t projection, int64_t slot,
        uint64_t chunk) const
    {
        uint64_t ordinal = 1;
        for (uint32_t q = 0; q < projection; ++q) {
            if (transports_[q] == kReduceGradTransportUdma) {
                ordinal += static_cast<uint64_t>(prefetchSlots_) * ChunkCount(q);
            }
        }
        ordinal += static_cast<uint64_t>(slot) * ChunkCount(projection) + chunk;
        return (static_cast<uint64_t>(static_cast<uint32_t>(magic_)) << 32) |
            static_cast<uint32_t>(ordinal);
    }

    __aicore__ inline bool UsesUdma() const
    {
        for (uint32_t projection = 0; projection < kReduceGradProjectionCount; ++projection) {
            if (transports_[projection] == kReduceGradTransportUdma) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint32_t UDMAQpFor(uint32_t projection, int64_t slot,
        uint64_t chunk) const
    {
        const uint32_t ordinal = static_cast<uint32_t>(Sequence(projection, slot, chunk));
        return (ordinal - 1U) % udmaQpCount_;
    }

    __aicore__ inline int64_t ControlPeer(int64_t controlIndex) const
    {
        return controlIndex >= rank_ ? controlIndex + 1 : controlIndex;
    }

    __aicore__ inline GM_ADDR PeerRecord(int64_t owner, int64_t source,
        int64_t slot, uint64_t chunk) const
    {
        const uint64_t stage = chunk & 1U;
        const uint64_t slotIndex = static_cast<uint64_t>(source * prefetchSlots_ + slot);
        return args_->peerMems[owner] + TileXR::IPC_DATA_OFFSET +
            peerRecordBaseOffset_ + stage * peerHalfBytes_ +
            slotIndex * peerSlotStrideBytes_;
    }

    __aicore__ inline uint64_t UDMAPeerStateOffset(int64_t peer) const
    {
        return udmaStateOffset_ + static_cast<uint64_t>(peer) * kReduceGradUdmaPeerStateBytes;
    }

    __aicore__ inline uint64_t UDMAReadyOffset(int64_t source, uint64_t stage) const
    {
        return UDMAPeerStateOffset(source) + kReduceGradUdmaReadyOffset +
            stage * kReduceGradUdmaSignalStageStride;
    }

    __aicore__ inline uint64_t UDMACompletionOffset(int64_t source, uint64_t stage) const
    {
        return UDMAPeerStateOffset(source) + kReduceGradUdmaCompletionOffset +
            stage * kReduceGradUdmaSignalStageStride;
    }

    __aicore__ inline uint64_t UDMAPollScratchOffset(int64_t peer, uint64_t stage) const
    {
        return UDMAPeerStateOffset(peer) + kReduceGradUdmaPollScratchOffset +
            stage * kReduceGradUdmaSignalStageStride;
    }

    __aicore__ inline uint64_t UDMAOutboundOffset(int64_t target, uint64_t stage) const
    {
        return udmaOutboundOffset_ +
            (static_cast<uint64_t>(target) * 2 + stage) * udmaChunkBytes_;
    }

    __aicore__ inline uint64_t UDMAInboundOffset(int64_t source, uint64_t stage) const
    {
        return udmaInboundOffset_ +
            (static_cast<uint64_t>(source) * 2 + stage) * udmaChunkBytes_;
    }

    __aicore__ inline uint64_t LoadSignal(GM_ADDR address)
    {
        TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(address), sizeof(uint64_t));
        AscendC::PipeBarrier<PIPE_ALL>();
        return reinterpret_cast<__gm__ uint64_t *>(address)[0];
    }

    __aicore__ inline void UdmaPollBackoff(uint64_t attempt)
    {
        const uint32_t shift = static_cast<uint32_t>(attempt >> 4) > 8U ?
            8U : static_cast<uint32_t>(attempt >> 4);
        const uint32_t spins = 1U << shift;
        for (volatile uint32_t spin = 0; spin < spins && spin < kUdmaPollBackoffMax;
            ++spin) {
            __asm__ __volatile__("");
        }
    }

    __aicore__ inline void StoreSignal(GM_ADDR address, uint64_t value)
    {
        reinterpret_cast<__gm__ uint64_t *>(address)[0] = value;
        AscendC::PipeBarrier<PIPE_ALL>();
        TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(address), sizeof(uint64_t));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool WaitUdmaReady(int64_t source, uint64_t stage, uint64_t sequence)
    {
        GM_ADDR ready = workspace_ + UDMAReadyOffset(source, stage);
        for (uint64_t attempt = 0; attempt < waitIterations_; ++attempt) {
            if (LoadSignal(ready) == sequence) {
                return true;
            }
        }
        SetStatus(kReduceGradDevicePeerTimeout);
        return false;
    }

    __aicore__ inline bool InitializeUdmaCompletionTargets(
        int64_t target, uint32_t completionTargets[kReduceGradMaxUdmaQpCount])
    {
        if (!UsesUdma()) {
            return true;
        }
        __gm__ TileXR::UDMAInfo *udmaInfo = TileXR::GetUDMAInfo(args_);
        for (uint32_t qpIdx = 0; qpIdx < udmaQpCount_; ++qpIdx) {
            __gm__ TileXR::UDMAWQCtx *queue = TileXR::UDMAGetWQCtx(
                udmaInfo, static_cast<uint32_t>(target), qpIdx);
            if (queue == nullptr || queue->wqeCntAddr == 0U) {
                SetStatus(kReduceGradDeviceUdmaCqError);
                return false;
            }
            completionTargets[qpIdx] = ld_dev(
                reinterpret_cast<__gm__ uint32_t *>(queue->wqeCntAddr), 0);
        }
        return true;
    }

    __aicore__ inline bool WaitUdmaCompletion(int64_t target, uint32_t qpIdx,
        uint64_t stage, uint64_t sequence,
        uint32_t completionTargets[kReduceGradMaxUdmaQpCount])
    {
        const uint64_t remoteOffset = UDMACompletionOffset(rank_, stage);
        GM_ADDR localScratch = workspace_ + UDMAPollScratchOffset(target, stage);
        AscendC::LocalTensor<uint8_t> wqeScratch = udmaWqeBuf_.Get<uint8_t>();
        for (uint64_t attempt = 0; attempt < waitIterations_; ++attempt) {
            const uint32_t getStatus = TileXR::UDMAGetNbiOnQp<uint8_t>(args_, wqeScratch,
                static_cast<int>(target), qpIdx,
                reinterpret_cast<__gm__ uint8_t *>(localScratch), remoteOffset,
                static_cast<uint32_t>(sizeof(uint64_t)));
            if (getStatus != TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                SetStatus(kReduceGradDeviceUdmaCqError);
                return false;
            }
            const uint32_t completionTarget = ++completionTargets[qpIdx];
            if (TileXR::UDMAQuietStatusOnQpUntil(args_,
                    static_cast<int>(target), qpIdx, completionTarget) !=
                    TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
                SetStatus(kReduceGradDeviceUdmaCqError);
                return false;
            }
            if (LoadSignal(localScratch) == sequence) {
                return true;
            }
            UdmaPollBackoff(attempt);
        }
        SetStatus(kReduceGradDevicePeerTimeout);
        return false;
    }

    __aicore__ inline bool WaitPeerRecords(GM_ADDR recordBase, uint64_t payloadBytes,
        bool ready)
    {
        AscendC::LocalTensor<uint8_t> flags = flagBuf_.Get<uint8_t>();
        const uint32_t totalRecords = TileXR::DataAsFlagBlockCountForPayloadBytes(payloadBytes);
        const uint32_t capacity = TileXR::DataAsFlagMaxCheckBlocks(kFlagBufferBytes);
        if (totalRecords == 0 || capacity == 0) {
            SetStatus(kReduceGradDeviceInvalidState);
            return false;
        }
        uint32_t record = 0;
        while (record < totalRecords) {
            const uint32_t batch = MinU64(totalRecords - record, capacity);
            bool matched = false;
            for (uint64_t attempt = 0; attempt < waitIterations_; ++attempt) {
                matched = ready ? TileXR::DataAsFlagCheckBatch(
                    reinterpret_cast<__gm__ uint8_t *>(recordBase), record, batch, flags) :
                    TileXR::DataAsFlagCheckBatchCleared(
                        reinterpret_cast<__gm__ uint8_t *>(recordBase), record, batch, flags);
                if (matched) {
                    break;
                }
            }
            if (!matched) {
                SetStatus(kReduceGradDevicePeerTimeout);
                return false;
            }
            record += batch;
        }
        return true;
    }

    __aicore__ inline void CopyDense(GM_ADDR destination, GM_ADDR source, uint64_t bytes)
    {
        AscendC::LocalTensor<uint8_t> local = ioBuf_.Get<uint8_t>();
        uint64_t offset = 0;
        while (offset < bytes) {
            const uint32_t batch = static_cast<uint32_t>(MinU64(bytes - offset, kIoBufferBytes));
            AscendC::GlobalTensor<uint8_t> sourceGlobal;
            sourceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source + offset));
            AscendC::DataCopyExtParams inParams {1U, batch, 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<uint8_t> pad {false, 0U, 0U, 0U};
            AscendC::DataCopyPad(local, sourceGlobal, inParams, pad);
            SyncEvent<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::GlobalTensor<uint8_t> destinationGlobal;
            destinationGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(destination + offset));
            AscendC::DataCopyPad(destinationGlobal, local, inParams);
            SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
            offset += batch;
        }
    }

    __aicore__ inline void ZeroDense(GM_ADDR destination, uint64_t bytes)
    {
        AscendC::LocalTensor<float> local = ioBuf_.Get<float>();
        uint64_t offset = 0;
        while (offset < bytes) {
            const uint32_t batch = static_cast<uint32_t>(MinU64(bytes - offset, kIoBufferBytes));
            AscendC::Duplicate(local, 0.0f, batch / sizeof(float));
            SyncEvent<AscendC::HardEvent::V_MTE3>();
            AscendC::GlobalTensor<float> destinationGlobal;
            destinationGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(destination + offset));
            AscendC::DataCopyExtParams outParams {1U, batch, 0U, 0U, 0U};
            AscendC::DataCopyPad(destinationGlobal, local, outParams);
            SyncEvent<AscendC::HardEvent::MTE3_S>();
            offset += batch;
        }
    }

    __aicore__ inline void AddDense(GM_ADDR destination, GM_ADDR source, uint64_t bytes)
    {
        AscendC::LocalTensor<float> input = ioBuf_.Get<float>();
        AscendC::LocalTensor<float> output = accumBuf_.Get<float>();
        uint64_t offset = 0;
        while (offset < bytes) {
            const uint32_t batch = static_cast<uint32_t>(MinU64(bytes - offset, kIoBufferBytes));
            AscendC::GlobalTensor<float> inputGlobal;
            inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(source + offset));
            AscendC::GlobalTensor<float> outputGlobal;
            outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(destination + offset));
            AscendC::DataCopyExtParams inParams {1U, batch, 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<float> pad {false, 0U, 0U, 0U};
            AscendC::DataCopyPad(input, inputGlobal, inParams, pad);
            AscendC::DataCopyPad(output, outputGlobal, inParams, pad);
            SyncEvent<AscendC::HardEvent::MTE2_V>();
            AscendC::Add(output, output, input, batch / sizeof(float));
            SyncEvent<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(outputGlobal, output, inParams);
            SyncEvent<AscendC::HardEvent::MTE3_S>();
            offset += batch;
        }
    }

    __aicore__ inline bool AddPeerPacked(GM_ADDR destination, GM_ADDR recordBase,
        uint64_t payloadBytes)
    {
        if (!WaitPeerRecords(recordBase, payloadBytes, true)) {
            return false;
        }
        AscendC::LocalTensor<float> input = ioBuf_.Get<float>();
        AscendC::LocalTensor<float> output = accumBuf_.Get<float>();
        AscendC::LocalTensor<float> clear = clearBuf_.Get<float>();
        const uint32_t totalRecords = TileXR::DataAsFlagBlockCountForPayloadBytes(payloadBytes);
        uint32_t record = 0;
        while (record < totalRecords) {
            const uint32_t records = static_cast<uint32_t>(MinU64(
                totalRecords - record, kPeerBatchRecords));
            const uint64_t payloadOffset = static_cast<uint64_t>(record) *
                TileXR::DATA_AS_FLAG_PAYLOAD_BYTES;
            const uint32_t batchBytes = static_cast<uint32_t>(MinU64(
                payloadBytes - payloadOffset,
                static_cast<uint64_t>(records) * TileXR::DATA_AS_FLAG_PAYLOAD_BYTES));

            AscendC::GlobalTensor<float> packedGlobal;
            packedGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                recordBase + static_cast<uint64_t>(record) * TileXR::DATA_AS_FLAG_BLOCK_BYTES));
            AscendC::DataCopyExtParams packedParams {
                static_cast<uint16_t>(records), TileXR::DATA_AS_FLAG_PAYLOAD_BYTES,
                TileXR::DATA_AS_FLAG_FLAG_BYTES, 0U, 0U};
            AscendC::DataCopyPadExtParams<float> pad {false, 0U, 0U, 0U};
            AscendC::DataCopyPad(input, packedGlobal, packedParams, pad);

            AscendC::GlobalTensor<float> outputGlobal;
            outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                destination + payloadOffset));
            AscendC::DataCopyExtParams outputParams {1U, batchBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(output, outputGlobal, outputParams, pad);
            SyncEvent<AscendC::HardEvent::MTE2_V>();
            AscendC::Add(output, output, input, batchBytes / sizeof(float));
            SyncEvent<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(outputGlobal, output, outputParams);
            SyncEvent<AscendC::HardEvent::MTE3_S>();

            AscendC::Duplicate(clear, 0.0f,
                records * TileXR::DATA_AS_FLAG_FLAG_FLOATS);
            SyncEvent<AscendC::HardEvent::V_MTE3>();
            AscendC::GlobalTensor<float> flagsGlobal;
            flagsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                recordBase + static_cast<uint64_t>(record) * TileXR::DATA_AS_FLAG_BLOCK_BYTES +
                TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES));
            AscendC::DataCopyExtParams clearParams {
                static_cast<uint16_t>(records), TileXR::DATA_AS_FLAG_FLAG_BYTES,
                0U, TileXR::DATA_AS_FLAG_PAYLOAD_BYTES, 0U};
            AscendC::DataCopyPad(flagsGlobal, clear, clearParams);
            SyncEvent<AscendC::HardEvent::MTE3_S>();
            record += records;
        }
        return true;
    }

    __aicore__ inline bool SendPeerChunk(int64_t target, int64_t slot,
        uint32_t projection, uint64_t chunk, GM_ADDR source, uint64_t bytes)
    {
        GM_ADDR record = PeerRecord(target, rank_, slot, chunk);
        if (!WaitPeerRecords(record, bytes, false)) {
            return false;
        }
        AscendC::LocalTensor<uint8_t> local = ioBuf_.Get<uint8_t>();
        TileXR::DataAsFlagInit(local);
        return TileXR::DataAsFlagSend(reinterpret_cast<__gm__ uint8_t *>(record),
            reinterpret_cast<__gm__ uint8_t *>(source), bytes, local) != 0;
    }

    __aicore__ inline bool PostUdmaChunk(int64_t target, int64_t slot,
        uint32_t projection, uint64_t chunk, GM_ADDR source, uint64_t bytes,
        uint32_t completionTargets[kReduceGradMaxUdmaQpCount])
    {
        const uint64_t stage = chunk & 1U;
        const uint64_t outboundOffset = UDMAOutboundOffset(target, stage);
        GM_ADDR outbound = workspace_ + outboundOffset;
        CopyDense(outbound, source, bytes);
        TileXR::UDMACleanCacheLines(
            reinterpret_cast<__gm__ uint8_t *>(outbound), bytes);
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t sequence = Sequence(projection, slot, chunk);
        const uint32_t qpIdx = UDMAQpFor(projection, slot, chunk);
        AscendC::LocalTensor<uint8_t> wqeScratch = udmaWqeBuf_.Get<uint8_t>();
        const uint32_t putStatus = TileXR::UDMAPutRegisteredSignalNbiOnQp<uint8_t>(
            args_, wqeScratch, static_cast<int>(target), qpIdx,
            reinterpret_cast<__gm__ uint8_t *>(outbound),
            UDMAInboundOffset(rank_, stage), static_cast<uint32_t>(bytes),
            UDMAReadyOffset(rank_, stage), sequence);
        if (putStatus != TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
            SetStatus(kReduceGradDeviceUdmaCqError);
            return false;
        }
        const uint32_t completionTarget = ++completionTargets[qpIdx];
        if (TileXR::UDMAQuietStatusOnQpUntil(args_,
                static_cast<int>(target), qpIdx, completionTarget) !=
                TileXR::TILEXR_UDMA_STATUS_SUCCESS) {
            SetStatus(kReduceGradDeviceUdmaCqError);
            return false;
        }
        return true;
    }

    __aicore__ inline bool CompleteChunk(int64_t target, int64_t slot,
        uint32_t projection, uint64_t chunk, GM_ADDR source, uint64_t bytes,
        uint32_t completionTargets[kReduceGradMaxUdmaQpCount])
    {
        bool complete = false;
        if (transports_[projection] == kReduceGradTransportPeer) {
            complete = WaitPeerRecords(
                PeerRecord(target, rank_, slot, chunk), bytes, false);
        } else {
            const uint32_t qpIdx = UDMAQpFor(projection, slot, chunk);
            complete = WaitUdmaCompletion(
                target, qpIdx, chunk & 1U, Sequence(projection, slot, chunk),
                completionTargets);
        }
        if (!complete) {
            return false;
        }
        ZeroDense(source, bytes);
        return true;
    }

    __aicore__ inline bool RunSenderTo(int64_t target)
    {
        if (target < 0 || target >= rankSize_) {
            SetStatus(kReduceGradDeviceInvalidState);
            return false;
        }
        uint32_t completionTargets[kReduceGradMaxUdmaQpCount] = {};
        if (!InitializeUdmaCompletionTargets(target, completionTargets)) {
            return false;
        }
        for (uint32_t projection = 0; projection < kReduceGradProjectionCount; ++projection) {
            const uint64_t chunkBytes = ChunkBytes(projection);
            const uint64_t chunks = ChunkCount(projection);
            if (chunkBytes == 0 || chunks == 0) {
                SetStatus(kReduceGradDeviceInvalidState);
                return false;
            }
            for (int64_t slot = 0; slot < prefetchSlots_; ++slot) {
                const int32_t expert = ExpertForSlot(rank_, slot);
                if (expert < 0) {
                    continue;
                }
                if (expert >= expertCount_) {
                    SetStatus(kReduceGradDeviceInvalidState);
                    return false;
                }
                if (OwnerForExpert(expert) != target) {
                    continue;
                }
                GM_ADDR sourceRow = gradients_[projection] +
                    static_cast<uint64_t>(expertCount_ + slot) * rowBytes_[projection];
                for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
                    if (chunk >= 2) {
                        const uint64_t completedChunk = chunk - 2;
                        const uint64_t completedOffset = completedChunk * chunkBytes;
                        const uint64_t completedBytes = MinU64(
                            rowBytes_[projection] - completedOffset, chunkBytes);
                        if (!CompleteChunk(target, slot, projection, completedChunk,
                                sourceRow + completedOffset, completedBytes,
                                completionTargets)) {
                            return false;
                        }
                    }
                    const uint64_t offset = chunk * chunkBytes;
                    const uint64_t bytes = MinU64(rowBytes_[projection] - offset, chunkBytes);
                    const bool ok = transports_[projection] ==
                            kReduceGradTransportPeer ?
                        SendPeerChunk(target, slot, projection, chunk, sourceRow + offset, bytes) :
                        PostUdmaChunk(target, slot, projection, chunk,
                            sourceRow + offset, bytes, completionTargets);
                    if (!ok) {
                        return false;
                    }
                }
                const uint64_t drainBegin = chunks > 2 ? chunks - 2 : 0;
                for (uint64_t chunk = drainBegin; chunk < chunks; ++chunk) {
                    const uint64_t offset = chunk * chunkBytes;
                    const uint64_t bytes = MinU64(rowBytes_[projection] - offset, chunkBytes);
                    if (!CompleteChunk(target, slot, projection, chunk,
                            sourceRow + offset, bytes, completionTargets)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    __aicore__ inline void RunSender()
    {
        for (int64_t controlIndex = blockIdx_; controlIndex < rankSize_ - 1;
             controlIndex += controlBlockCount_) {
            if (!RunSenderTo(ControlPeer(controlIndex))) {
                return;
            }
        }
    }

    __aicore__ inline void RunReceiver()
    {
        const int64_t receiverIndex = blockIdx_ - controlBlockCount_;
        for (uint32_t projection = 0; projection < kReduceGradProjectionCount; ++projection) {
            const uint64_t chunkBytes = ChunkBytes(projection);
            const uint64_t chunks = ChunkCount(projection);
            for (int64_t source = 0; source < rankSize_; ++source) {
                for (int64_t slot = 0; slot < prefetchSlots_; ++slot) {
                    const int32_t expert = ExpertForSlot(source, slot);
                    if (expert < 0) {
                        continue;
                    }
                    if (expert >= expertCount_) {
                        SetStatus(kReduceGradDeviceInvalidState);
                        return;
                    }
                    if (OwnerForExpert(expert) != rank_) {
                        continue;
                    }
                    GM_ADDR outputRow = gradients_[projection] +
                        static_cast<uint64_t>(expert) * rowBytes_[projection];
                    const uint64_t localExpert = static_cast<uint64_t>(
                        expert - rank_ * expertsPerRank_);
                    for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
                        // A slot's two stage records are reused across every projection and
                        // chunk, so one receiver must consume them in sender order.
                        const uint64_t workIndex = localExpert;
                        if (static_cast<int64_t>(
                                workIndex % static_cast<uint64_t>(receiverCount_)) !=
                            receiverIndex) {
                            continue;
                        }
                        const uint64_t offset = chunk * chunkBytes;
                        const uint64_t bytes = MinU64(rowBytes_[projection] - offset, chunkBytes);
                        if (source == rank_) {
                            GM_ADDR sourceRow = gradients_[projection] +
                                static_cast<uint64_t>(expertCount_ + slot) * rowBytes_[projection];
                            AddDense(outputRow + offset, sourceRow + offset, bytes);
                            ZeroDense(sourceRow + offset, bytes);
                        } else if (transports_[projection] ==
                            kReduceGradTransportPeer) {
                            if (!AddPeerPacked(outputRow + offset,
                                    PeerRecord(rank_, source, slot, chunk), bytes)) {
                                return;
                            }
                        } else {
                            const uint64_t stage = chunk & 1U;
                            const uint64_t sequence = Sequence(projection, slot, chunk);
                            if (!WaitUdmaReady(source, stage, sequence)) {
                                return;
                            }
                            GM_ADDR inbound = workspace_ + UDMAInboundOffset(source, stage);
                            TileXR::UDMACleanCacheLines(
                                reinterpret_cast<__gm__ uint8_t *>(inbound), bytes);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AddDense(outputRow + offset, inbound, bytes);
                            StoreSignal(workspace_ + UDMACompletionOffset(source, stage), sequence);
                        }
                    }
                }
            }
        }
    }

    __gm__ TileXR::CommArgs *args_{nullptr};
    __gm__ int32_t *expertsToCopy_{nullptr};
    GM_ADDR gradients_[kReduceGradProjectionCount] = {};
    GM_ADDR workspace_{nullptr};
    __gm__ int32_t *status_{nullptr};
    int64_t rank_{0};
    int64_t rankSize_{0};
    int64_t expertCount_{0};
    int64_t expertsPerRank_{0};
    int64_t prefetchSlots_{0};
    int64_t controlBlockCount_{0};
    int64_t blockIdx_{0};
    int64_t blockCount_{0};
    int64_t receiverCount_{0};
    uint64_t rowElements_[kReduceGradProjectionCount] = {};
    uint64_t rowBytes_[kReduceGradProjectionCount] = {};
    uint32_t transports_[kReduceGradProjectionCount] = {};
    uint32_t udmaQpCount_{0};
    uint64_t peerRecordBaseOffset_{0};
    uint64_t peerHalfBytes_{0};
    uint64_t peerSlotStrideBytes_{0};
    uint64_t peerChunkPayloadBytes_{0};
    uint64_t udmaStateOffset_{0};
    uint64_t udmaOutboundOffset_{0};
    uint64_t udmaInboundOffset_{0};
    uint64_t udmaChunkBytes_{0};
    uint64_t workspaceBytes_{0};
    uint64_t waitIterations_{0};
    int64_t magic_{0};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> ioBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accumBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> flagBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> clearBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> udmaWqeBuf_;
};

} // namespace Kernel
} // namespace TileXRMoonEp

extern "C" __global__ __aicore__ void tilexr_moonep_reduce_grad_kernel(
    GM_ADDR commArgs, GM_ADDR expertsToCopy, GM_ADDR gate, GM_ADDR up, GM_ADDR down,
    GM_ADDR workspace, GM_ADDR status, int64_t rank, int64_t rankSize,
    int64_t expertCount, int64_t expertsPerRank, int64_t prefetchSlots,
    int64_t controlBlockCount,
    uint64_t gateRowElements, uint64_t upRowElements, uint64_t downRowElements,
    uint64_t gateRowBytes, uint64_t upRowBytes, uint64_t downRowBytes,
    uint32_t gateTransport, uint32_t upTransport, uint32_t downTransport,
    uint32_t udmaQpCount, uint64_t peerRecordBaseOffset, uint64_t peerHalfBytes,
    uint64_t peerSlotStrideBytes, uint64_t peerChunkPayloadBytes,
    uint64_t udmaStateOffset, uint64_t udmaOutboundOffset,
    uint64_t udmaInboundOffset, uint64_t udmaChunkBytes, uint64_t workspaceBytes,
    uint64_t waitIterations, int64_t magic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        TileXRMoonEp::Kernel::ReduceGradKernel op;
        op.Init(commArgs, expertsToCopy, gate, up, down, workspace, status, rank,
            rankSize, expertCount, expertsPerRank, prefetchSlots, controlBlockCount,
            gateRowElements, upRowElements, downRowElements, gateRowBytes,
            upRowBytes, downRowBytes, gateTransport, upTransport, downTransport,
            udmaQpCount,
            peerRecordBaseOffset, peerHalfBytes, peerSlotStrideBytes,
            peerChunkPayloadBytes, udmaStateOffset, udmaOutboundOffset,
            udmaInboundOffset, udmaChunkBytes, workspaceBytes, waitIterations, magic);
        op.Process();
    }
}
