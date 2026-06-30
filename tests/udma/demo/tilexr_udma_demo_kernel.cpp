/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "kernel_operator.h"
#include "tilexr_data_as_flag.h"
#include "tilexr_udma.h"

constexpr int32_t TILEXR_UDMA_DEMO_MAGIC = 0x5444554d; // "TDUM"
constexpr uint64_t TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET = TileXR::IPC_DATA_OFFSET;
constexpr uint64_t TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET = TileXR::IPC_DATA_OFFSET;
constexpr uint32_t TILEXR_UDMA_DEMO_DATA_AS_FLAG_UB_BYTES = 64 * 1024;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE = 6;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_RANGE_VALID_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 16;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_WQE_BEFORE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 32;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_WQE_AFTER_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 48;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_LOCAL_TOKEN_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 64;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_REMOTE_BASE_LOW_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 80;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_MEM_ADDR_LOW_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 96;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_TPN_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 112;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_SEND_SAMPLE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 128;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_RECV_SAMPLE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 144;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_RECV_SLOT_SAMPLE_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 160;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_REMOTE_DATA_OFFSET_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 176;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_REMOTE_READY_OFFSET_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 192;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_READY_SEEN_BASE =
    TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + 208;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ACK_SEEN_BASE =
    TILEXR_UDMA_DEMO_DEBUG_READY_SEEN_BASE + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER =
    TILEXR_UDMA_DEMO_DEBUG_ACK_SEEN_BASE + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER = TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER = TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER + 1;
constexpr int32_t TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SUM = TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER + 1;
constexpr uint64_t TILEXR_UDMA_DEMO_SIGNAL_MAX_POLLS = 100000000ULL;
constexpr uint64_t TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES = TileXR::TILEXR_UDMA_CACHE_LINE_SIZE;
constexpr uint64_t TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES = 128ULL;
constexpr int32_t TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS = -1001;
constexpr int32_t TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS = -1002;
constexpr int32_t TILEXR_UDMA_DEMO_COPY_TIMEOUT_STATUS = -1003;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES = 64 * 1024;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_PINGPONG_BYTES =
    TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES * 2U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS = 2U;
constexpr uint64_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_PEER_SLOT_BYTES = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_SINGLE_NODE_SHARDS = 2U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_LOCAL_COPY_SHARDS =
    TILEXR_UDMA_DEMO_BIGDATA_SINGLE_NODE_SHARDS;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER = 5U;
constexpr int32_t TILEXR_UDMA_DEMO_BIGDATA_RANKS_PER_NODE = 8;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES = 16U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORES = 16U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_CONTROL_SHARDS = 32U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_PRIMARY_CORE = 16U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_SECONDARY_CORE = 17U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_LOCAL_SEND_CORE = 18U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORE_BASE = 19U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_BLOCK_DIM =
    TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORE_BASE +
    TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORES;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_REMOTE_PUT_ONLY_BLOCK_DIM = 64U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT = 0U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_SEGMENT = 1U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_PRIMARY = 2U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_SECONDARY = 3U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_LOCAL_COPY_READY = 4U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END = 12U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_AGGREGATOR = 11U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_WAIT_CORE = 20U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_WAIT_SHARD =
    TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_WAIT_CORE -
    TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORE_BASE;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_SOURCE_SHARD = 0U;
constexpr uint32_t TILEXR_UDMA_DEMO_BIGDATA_LOCAL_FANOUT_SHARD_BASE = 5U;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_PREPARE = 0;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_SEND_COPY = 1;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_SEND_SYNC = 2;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_DATA_PUT = 3;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_WAIT_READY = 4;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_RELAY_COPY = 5;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_ACK_PUT = 6;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_WAIT_ACK = 7;
constexpr uint32_t TILEXR_BIGDATA_PROFILE_STAGE_FULL = 8;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_FRAMEWORK = 0;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_LOOP = 1;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_PEER = 2;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_QP = 3;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_SEGMENT = 4;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_ADDRESS = 5;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_POST = 6;
constexpr uint32_t TILEXR_BIGDATA_REMOTE_PUT_STAGE_ACK = 7;

namespace {

__aicore__ inline uint64_t AllToAllPayloadBytes(int32_t elementsPerPeer)
{
    return static_cast<uint64_t>(elementsPerPeer) * sizeof(int32_t);
}

__aicore__ inline uint64_t BigDataChunkBytesPerPeer(bool use35Core, int32_t effectiveChunkElements)
{
    const uint64_t chunkBytes = static_cast<uint64_t>(effectiveChunkElements) * sizeof(int32_t);
    if (!use35Core || chunkBytes < TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_PEER_SLOT_BYTES) {
        return chunkBytes;
    }
    return TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_PEER_SLOT_BYTES;
}

__aicore__ inline void TileXRUdmaDemoWeightedWqeSlice(
    __gm__ TileXR::UDMAInfo* udmaInfo, int32_t peer, uint32_t total,
    uint32_t wqeCount, uint32_t wqeIdx, uint32_t& offset, uint32_t& bytes)
{
    uint32_t weightSum = 0;
    uint32_t prefixWeight = 0;
    for (uint32_t i = 0; i < wqeCount; ++i) {
        uint32_t weight = TileXR::UDMAGetQpWeight(udmaInfo, peer, i);
        if (i < wqeIdx) {
            prefixWeight += weight;
        }
        weightSum += weight;
    }
    if (wqeCount == 0 || weightSum == 0 || wqeIdx >= wqeCount) {
        offset = total;
        bytes = 0;
        return;
    }

    uint64_t rawStart = static_cast<uint64_t>(total) * prefixWeight / weightSum;
    uint64_t rawEnd = static_cast<uint64_t>(total) *
        (prefixWeight + TileXR::UDMAGetQpWeight(udmaInfo, peer, wqeIdx)) / weightSum;
    uint32_t alignedStart = static_cast<uint32_t>(
        (rawStart / TileXR::BLOCK_UNIT_BYTE) * TileXR::BLOCK_UNIT_BYTE);
    uint32_t alignedEnd = wqeIdx + 1 == wqeCount ? total : static_cast<uint32_t>(
        ((rawEnd + TileXR::BLOCK_UNIT_BYTE - 1) / TileXR::BLOCK_UNIT_BYTE) * TileXR::BLOCK_UNIT_BYTE);
    if (alignedEnd > total) {
        alignedEnd = total;
    }
    if (alignedStart >= alignedEnd) {
        offset = total;
        bytes = 0;
        return;
    }
    offset = alignedStart;
    bytes = alignedEnd - alignedStart;
}

__aicore__ inline uint64_t AllToAllDataAsFlagSegmentBytes(uint64_t payloadBytes)
{
    return static_cast<uint64_t>(TileXR::DataAsFlagBlockCountForPayloadBytes(payloadBytes)) *
        TileXR::DATA_AS_FLAG_BLOCK_BYTES;
}

__aicore__ inline __gm__ uint64_t* ControlSlot(__gm__ uint8_t* base, uint64_t offset, int32_t peer)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        base + offset + static_cast<uint64_t>(peer) * TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES);
}

__aicore__ inline __gm__ uint64_t* BigDataControlSlot(
    __gm__ uint8_t* base, uint64_t offset, uint32_t slot, int32_t rankSize,
    uint32_t shardCount, int32_t peer, uint32_t shard)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        base + offset +
        ((static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
        static_cast<uint64_t>(peer)) *
        static_cast<uint64_t>(shardCount) +
        static_cast<uint64_t>(shard)) * TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES);
}

__aicore__ inline void BigDataCopyInTile(
    __gm__ uint8_t* src, uint32_t offset, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> local)
{
    AscendC::GlobalTensor<uint8_t> srcGlobal;
    srcGlobal.SetGlobalBuffer(src + offset);
    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
    AscendC::DataCopyExtParams copyIn {1U, bytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(local, srcGlobal, copyIn, padIn);
}

__aicore__ inline void BigDataCopyOutTile(
    __gm__ uint8_t* dst, uint32_t offset, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> local)
{
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(dst + offset);
    AscendC::DataCopyExtParams copyOut {1U, bytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(dstGlobal, local, copyOut);
}

__aicore__ inline void BigDataCopyOneRelay(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint8_t> srcGlobal;
    srcGlobal.SetGlobalBuffer(src);
    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
    AscendC::DataCopyExtParams copyIn {1U, bytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint8_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(dst);
    AscendC::DataCopyExtParams copyOut {1U, bytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline void BigDataStoreInt32Mte(
    __gm__ int32_t* dst, int32_t value, AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::LocalTensor<int32_t> fillLocal = relayLocal.ReinterpretCast<int32_t>();
    fillLocal.SetValue(0, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint8_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dst));
    AscendC::DataCopyExtParams copyOut {1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void BigDataWaitMte2ToMte3(uint32_t bufferId)
{
    if (bufferId == 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
    }
}

__aicore__ inline void BigDataSetMte2ToMte3(uint32_t bufferId)
{
    if (bufferId == 0U) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
    }
}

__aicore__ inline void BigDataWaitMte3ToMte2(uint32_t bufferId)
{
    if (bufferId == 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

__aicore__ inline void BigDataSetMte3ToMte2(uint32_t bufferId)
{
    if (bufferId == 0U) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

__aicore__ inline uint32_t BigDataTileBytes(uint32_t totalBytes, uint32_t offset)
{
    const uint32_t remain = totalBytes - offset;
    return remain < TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES ?
        remain : TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
}

__aicore__ inline bool BigDataCopyShardRange(
    uint32_t shard, uint32_t shardCount, uint32_t totalElements,
    uint32_t& shardOffsetBytes, uint32_t& shardBytes)
{
    if (shardCount == 0U || shard >= shardCount) {
        shardOffsetBytes = 0U;
        shardBytes = 0U;
        return false;
    }
    constexpr uint32_t alignElements = 32U / sizeof(int32_t);
    const uint64_t begin =
        static_cast<uint64_t>(totalElements) * static_cast<uint64_t>(shard) /
        static_cast<uint64_t>(shardCount);
    const uint64_t end =
        static_cast<uint64_t>(totalElements) * static_cast<uint64_t>(shard + 1U) /
        static_cast<uint64_t>(shardCount);
    const uint64_t alignedBegin = (shard == 0U) ? 0ULL :
        (begin / static_cast<uint64_t>(alignElements)) * static_cast<uint64_t>(alignElements);
    const uint64_t alignedEnd = (shard + 1U == shardCount) ?
        static_cast<uint64_t>(totalElements) :
        (end / static_cast<uint64_t>(alignElements)) * static_cast<uint64_t>(alignElements);
    shardOffsetBytes = static_cast<uint32_t>(alignedBegin * sizeof(int32_t));
    shardBytes = static_cast<uint32_t>((alignedEnd - alignedBegin) * sizeof(int32_t));
    return shardBytes > 0U;
}

__aicore__ inline uint32_t BigDataCopyShardStartBytes(
    uint32_t shard, uint32_t shardCount, uint32_t totalElements)
{
    if (shardCount == 0U || shard >= shardCount) {
        return 0U;
    }
    constexpr uint32_t alignElements = 32U / sizeof(int32_t);
    const uint64_t begin =
        static_cast<uint64_t>(totalElements) * static_cast<uint64_t>(shard) /
        static_cast<uint64_t>(shardCount);
    const uint64_t alignedBegin = (shard == 0U) ? 0ULL :
        (begin / static_cast<uint64_t>(alignElements)) * static_cast<uint64_t>(alignElements);
    return static_cast<uint32_t>(alignedBegin * sizeof(int32_t));
}

__aicore__ inline void BigDataCopyRangePingPong(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t bytes,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    if (bytes == 0U) {
        return;
    }
    const uint32_t tileCount =
        (bytes + TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES - 1U) /
        TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
    if (tileCount <= 2U) {
        for (uint32_t offset = 0; offset < bytes; offset += TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES) {
            const uint32_t tileBytes = BigDataTileBytes(bytes, offset);
            BigDataCopyOneRelay(dst + offset, src + offset, tileBytes, relayLocal);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }
    bool copyOutInFlight0 = false;
    bool copyOutInFlight1 = false;

    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t bufferId = tile & 1U;
        if (tile >= 2U) {
            BigDataWaitMte3ToMte2(bufferId);
            if (bufferId == 0U) {
                copyOutInFlight0 = false;
            } else {
                copyOutInFlight1 = false;
            }
        }

        const uint32_t offset = tile * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
        const uint32_t tileBytes = BigDataTileBytes(bytes, offset);
        AscendC::LocalTensor<uint8_t> local =
            relayLocal[bufferId * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES];
        BigDataCopyInTile(src, offset, tileBytes, local);
        BigDataSetMte2ToMte3(bufferId);

        if (tile > 0U) {
            const uint32_t prevBufferId = bufferId ^ 1U;
            const uint32_t prevOffset = offset - TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
            const uint32_t prevBytes = BigDataTileBytes(bytes, prevOffset);
            AscendC::LocalTensor<uint8_t> prevLocal =
                relayLocal[prevBufferId * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES];
            BigDataWaitMte2ToMte3(prevBufferId);
            BigDataCopyOutTile(dst, prevOffset, prevBytes, prevLocal);
            BigDataSetMte3ToMte2(prevBufferId);
            if (prevBufferId == 0U) {
                copyOutInFlight0 = true;
            } else {
                copyOutInFlight1 = true;
            }
        }
    }

    const uint32_t lastTile = tileCount - 1U;
    const uint32_t lastBufferId = lastTile & 1U;
    const uint32_t lastOffset = lastTile * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES;
    const uint32_t lastBytes = BigDataTileBytes(bytes, lastOffset);
    AscendC::LocalTensor<uint8_t> lastLocal =
        relayLocal[lastBufferId * TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_BYTES];
    BigDataWaitMte2ToMte3(lastBufferId);
    BigDataCopyOutTile(dst, lastOffset, lastBytes, lastLocal);
    BigDataSetMte3ToMte2(lastBufferId);
    if (lastBufferId == 0U) {
        copyOutInFlight0 = true;
    } else {
        copyOutInFlight1 = true;
    }

    if (copyOutInFlight0) {
        BigDataWaitMte3ToMte2(0U);
    }
    if (copyOutInFlight1) {
        BigDataWaitMte3ToMte2(1U);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool BigDataPassChunk(
    uint32_t pass, int32_t elementsPerPeer, int32_t effectiveChunkElements,
    int32_t& chunkOffset, uint32_t& chunkBytes)
{
    chunkOffset = static_cast<int32_t>(pass) * effectiveChunkElements;
    const int32_t remaining = elementsPerPeer - chunkOffset;
    const int32_t chunkElem = remaining < effectiveChunkElements ? remaining : effectiveChunkElements;
    if (chunkElem <= 0) {
        chunkBytes = 0U;
        return false;
    }
    chunkBytes = static_cast<uint32_t>(static_cast<uint64_t>(chunkElem) * sizeof(int32_t));
    return true;
}

__aicore__ inline uint64_t BigDataGlobalPassIndex(
    uint64_t kernelLoopBase, uint32_t passCount, uint32_t loop, uint32_t pass)
{
    return (kernelLoopBase + static_cast<uint64_t>(loop)) * static_cast<uint64_t>(passCount) +
        static_cast<uint64_t>(pass);
}

__aicore__ inline uint64_t BigDataPassToken(uint64_t globalPass)
{
    return globalPass + 1ULL;
}

__aicore__ inline uint32_t BigDataPingPongSlot(uint64_t globalPass)
{
    return static_cast<uint32_t>(globalPass &
        static_cast<uint64_t>(TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS - 1U));
}

__aicore__ inline uint32_t BigDataDataSlot(uint64_t globalPass, uint32_t pass, bool use35Core)
{
    return use35Core ? pass : BigDataPingPongSlot(globalPass);
}

__aicore__ inline void BigDataKernelExitBarrier()
{
    AscendC::SyncAll();
}

__aicore__ inline int32_t BigDataNormalizeRanksPerNode(int32_t ranksPerNode)
{
    return ranksPerNode > 0 ? ranksPerNode : TILEXR_UDMA_DEMO_BIGDATA_RANKS_PER_NODE;
}

__aicore__ inline bool BigDataIsMultiNode(int32_t rankSize, int32_t ranksPerNode)
{
    return rankSize > BigDataNormalizeRanksPerNode(ranksPerNode);
}

__aicore__ inline bool BigDataUse35Core(int32_t rankSize, bool force35Core, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    return BigDataIsMultiNode(rankSize, localRanks) ||
        (force35Core && rankSize == localRanks);
}

__aicore__ inline uint32_t BigDataShardCount(
    int32_t rankSize, bool force35Core = false, int32_t ranksPerNode = TILEXR_UDMA_DEMO_BIGDATA_RANKS_PER_NODE)
{
    return BigDataUse35Core(rankSize, force35Core, ranksPerNode) ?
        TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_CONTROL_SHARDS :
        TILEXR_UDMA_DEMO_BIGDATA_SINGLE_NODE_SHARDS;
}

__aicore__ inline bool BigDataValidTopology(int32_t rankSize, int32_t ranksPerNode)
{
    if (rankSize <= 0) {
        return false;
    }
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (localRanks <= 0) {
        return false;
    }
    if (!BigDataIsMultiNode(rankSize, localRanks)) {
        return true;
    }
    return rankSize % localRanks == 0;
}

__aicore__ inline int32_t BigDataLocalNodeBegin(int32_t rank, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    return (rank / localRanks) * localRanks;
}

__aicore__ inline int32_t BigDataNodeCount(int32_t rankSize, int32_t ranksPerNode)
{
    if (!BigDataValidTopology(rankSize, ranksPerNode)) {
        return 0;
    }
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (!BigDataIsMultiNode(rankSize, localRanks)) {
        return 1;
    }
    return rankSize / localRanks;
}

__aicore__ inline int32_t BigDataTaskCount(int32_t rankSize, bool force35Core, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (force35Core && rankSize == localRanks) {
        return localRanks - 1;
    }
    return rankSize - 1;
}

__aicore__ inline bool BigDataIsLocalPeer(int32_t rank, int32_t peer, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    const int32_t begin = BigDataLocalNodeBegin(rank, localRanks);
    return peer >= begin && peer < begin + localRanks;
}

__aicore__ inline int32_t BigDataRemotePeerAt(
    int32_t rank, int32_t rankSize, int32_t remoteIndex, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (!BigDataIsMultiNode(rankSize, localRanks) || !BigDataValidTopology(rankSize, localRanks) ||
        remoteIndex < 0) {
        return -1;
    }
    int32_t remoteCount = 0;
    for (int32_t step = 0; step < rankSize; ++step) {
        const int32_t peer =
            (rank + localRanks + step) % rankSize;
        if (!BigDataIsLocalPeer(rank, peer, localRanks)) {
            if (remoteCount == remoteIndex) {
                return peer;
            }
            ++remoteCount;
        }
    }
    return -1;
}

__aicore__ inline int32_t BigDataRemotePeerForwardAt(
    int32_t rank, int32_t rankSize, int32_t remoteIndex, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (!BigDataIsMultiNode(rankSize, localRanks) || !BigDataValidTopology(rankSize, localRanks) ||
        remoteIndex < 0) {
        return -1;
    }
    int32_t remoteCount = 0;
    for (int32_t step = 0; step < rankSize; ++step) {
        const int32_t peer =
            (rank + localRanks + step) % rankSize;
        if (!BigDataIsLocalPeer(rank, peer, localRanks)) {
            if (remoteCount == remoteIndex) {
                return peer;
            }
            ++remoteCount;
        }
    }
    return -1;
}

__aicore__ inline int32_t BigDataRemotePeerReverseAt(
    int32_t rank, int32_t rankSize, int32_t remoteIndex, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (!BigDataIsMultiNode(rankSize, localRanks) || !BigDataValidTopology(rankSize, localRanks) ||
        remoteIndex < 0) {
        return -1;
    }
    int32_t remoteCount = 0;
    for (int32_t step = 0; step < rankSize; ++step) {
        const int32_t peer =
            (rank - localRanks - step + rankSize * 2) % rankSize;
        if (!BigDataIsLocalPeer(rank, peer, localRanks)) {
            if (remoteCount == remoteIndex) {
                return peer;
            }
            ++remoteCount;
        }
    }
    return -1;
}

__aicore__ inline uint32_t BigDataSelectWeightedQp(
    const __gm__ TileXR::CommArgs* args, int32_t peer, bool selectMax)
{
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    const uint32_t qpCount = udmaInfo->qpNum == 0 ? 1U : udmaInfo->qpNum;
    uint32_t selected = 0U;
    uint32_t selectedWeight = TileXR::UDMAGetQpWeight(udmaInfo, peer, 0U);
    for (uint32_t qpIdx = 1U; qpIdx < qpCount; ++qpIdx) {
        const uint32_t weight = TileXR::UDMAGetQpWeight(udmaInfo, peer, qpIdx);
        if ((selectMax && weight > selectedWeight) || (!selectMax && weight < selectedWeight)) {
            selected = qpIdx;
            selectedWeight = weight;
        }
    }
    return selected;
}

__aicore__ inline uint32_t BigDataSelectDistinctWeightedQp(
    const __gm__ TileXR::CommArgs* args, int32_t peer, uint32_t avoidQp, bool selectMax)
{
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    const uint32_t qpCount = udmaInfo->qpNum == 0 ? 1U : udmaInfo->qpNum;
    if (qpCount <= 1U) {
        return 0U;
    }
    uint32_t selected = avoidQp == 0U ? 1U : 0U;
    uint32_t selectedWeight = TileXR::UDMAGetQpWeight(udmaInfo, peer, selected);
    for (uint32_t qpIdx = 0U; qpIdx < qpCount; ++qpIdx) {
        if (qpIdx == avoidQp) {
            continue;
        }
        const uint32_t weight = TileXR::UDMAGetQpWeight(udmaInfo, peer, qpIdx);
        if ((selectMax && weight > selectedWeight) || (!selectMax && weight < selectedWeight)) {
            selected = qpIdx;
            selectedWeight = weight;
        }
    }
    return selected;
}

__aicore__ inline uint32_t BigDataRemotePutOnlySegmentQp(
    const __gm__ TileXR::CommArgs* args, uint32_t segmentId)
{
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    const uint32_t qpCount = udmaInfo->qpNum == 0 ? 1U : udmaInfo->qpNum;
    if (qpCount <= 1U) {
        return 0U;
    }
    return segmentId == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT ? 0U : 1U;
}

__aicore__ inline int32_t BigDataLocalPeerAt(int32_t rank, int32_t localIndex, int32_t ranksPerNode)
{
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (localIndex < 0 || localIndex >= localRanks - 1) {
        return -1;
    }
    const int32_t begin = BigDataLocalNodeBegin(rank, localRanks);
    const int32_t local = rank - begin;
    return begin + ((local + localIndex + 1) % localRanks);
}

__aicore__ inline bool BigDataMergedPeerTaskAt(
    int32_t rank, int32_t rankSize, int32_t taskIndex, bool force35Core,
    int32_t ranksPerNode, int32_t& peer, bool& isLocalPeer)
{
    peer = -1;
    isLocalPeer = false;
    const int32_t localRanks = BigDataNormalizeRanksPerNode(ranksPerNode);
    if (force35Core && rankSize == localRanks) {
        peer = BigDataLocalPeerAt(rank, taskIndex, localRanks);
        isLocalPeer = true;
        return peer >= 0;
    }
    const int32_t nodeCount = BigDataNodeCount(rankSize, localRanks);
    if (nodeCount <= 1 || taskIndex < 0) {
        return false;
    }
    const int32_t remoteBurst = nodeCount - 1;
    const int32_t groupSize = remoteBurst + 1;
    const int32_t group = taskIndex / groupSize;
    const int32_t indexInGroup = taskIndex % groupSize;
    if (indexInGroup < remoteBurst) {
        const int32_t remoteIndex = group * remoteBurst + indexInGroup;
        peer = BigDataRemotePeerAt(rank, rankSize, remoteIndex, localRanks);
        isLocalPeer = false;
        return peer >= 0;
    }
    peer = BigDataLocalPeerAt(rank, group, localRanks);
    isLocalPeer = true;
    return peer >= 0;
}

__aicore__ inline int32_t BigDataNetworkPeerIndex(int32_t peer, int32_t rank)
{
    return peer < rank ? peer : peer - 1;
}

__aicore__ inline __gm__ uint8_t* BigDataSlot(
    __gm__ uint8_t* udmaMem, uint64_t baseOffset, uint32_t slot,
    int32_t networkPeerCount, int32_t peerIndex, uint64_t chunkBytesPerPeer)
{
    return udmaMem + baseOffset +
        (static_cast<uint64_t>(slot) * static_cast<uint64_t>(networkPeerCount) +
        static_cast<uint64_t>(peerIndex)) * chunkBytesPerPeer;
}

__aicore__ inline void BigDataStoreTokenMte(
    __gm__ uint64_t* slot, uint64_t token, AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::LocalTensor<uint64_t> tokenLocal = relayLocal.ReinterpretCast<uint64_t>();
    constexpr uint32_t controlSlotU64 =
        TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES / sizeof(uint64_t);
    for (uint32_t i = 0; i < controlSlotU64; ++i) {
        tokenLocal.SetValue(i, token);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint8_t> slotGlobal;
    slotGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slot));
    AscendC::DataCopyExtParams copyOut {
        1U, static_cast<uint32_t>(TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES), 0U, 0U, 0U};
    AscendC::DataCopyPad(slotGlobal, relayLocal, copyOut);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline uint64_t BigDataLoadTokenMte(
    __gm__ uint64_t* slot, AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint8_t> slotGlobal;
    slotGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slot));
    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
    AscendC::DataCopyExtParams copyIn {
        1U, static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(relayLocal, slotGlobal, copyIn, padIn);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return relayLocal.ReinterpretCast<uint64_t>().GetValue(0);
}

__aicore__ inline uint64_t BigDataWaitTokenMte(
    __gm__ uint64_t* slot, uint64_t token, AscendC::LocalTensor<uint8_t> relayLocal)
{
    uint64_t observed = BigDataLoadTokenMte(slot, relayLocal);
    uint64_t polls = 0;
    while (observed < token && polls < TILEXR_UDMA_DEMO_SIGNAL_MAX_POLLS) {
        observed = BigDataLoadTokenMte(slot, relayLocal);
        ++polls;
    }
    return observed;
}

__aicore__ inline __gm__ uint64_t* BigDataRemoteRegisteredControlSlot(
    const __gm__ TileXR::CommArgs* args, int32_t targetRank, uint64_t offset,
    uint32_t slot, int32_t rankSize, uint32_t shardCount, int32_t slotRank, uint32_t shard)
{
    auto registry = TileXR::GetUDMARegistry(args);
    const uint64_t remoteOffset =
        offset +
        ((static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
        static_cast<uint64_t>(slotRank)) *
        static_cast<uint64_t>(shardCount) +
        static_cast<uint64_t>(shard)) * TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES;
    if (!TileXR::UDMARegisteredRangeValid(registry, targetRank,
            remoteOffset, TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES)) {
        return nullptr;
    }
    return reinterpret_cast<__gm__ uint64_t*>(
        TileXR::UDMARegisteredRemoteAddr(registry, targetRank, remoteOffset));
}

__aicore__ inline uint64_t BigDataRegisteredControlOffset(
    uint64_t offset, uint32_t slot, int32_t rankSize, uint32_t shardCount,
    int32_t slotRank, uint32_t shard)
{
    return offset +
        ((static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
        static_cast<uint64_t>(slotRank)) *
        static_cast<uint64_t>(shardCount) +
        static_cast<uint64_t>(shard)) * TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES;
}

__aicore__ inline uint64_t BigDataIpcAckOffset(uint32_t slot, int32_t rankSize, int32_t slotRank)
{
    constexpr uint64_t maxAckBytes =
        static_cast<uint64_t>(TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS) *
        static_cast<uint64_t>(TileXR::TILEXR_MAX_RANK_SIZE) *
        TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES;
    const uint64_t baseOffset = static_cast<uint64_t>(TileXR::IPC_DATA_OFFSET) - maxAckBytes;
    return baseOffset +
        (static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
        static_cast<uint64_t>(slotRank)) * TILEXR_UDMA_DEMO_CONTROL_SLOT_BYTES;
}

__aicore__ inline __gm__ uint64_t* BigDataLocalIpcAckSlot(
    const __gm__ TileXR::CommArgs* args, int32_t rank, uint32_t slot, int32_t rankSize, int32_t peer)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        args->peerMems[rank] + BigDataIpcAckOffset(slot, rankSize, peer));
}

__aicore__ inline __gm__ uint64_t* BigDataRemoteIpcAckSlot(
    const __gm__ TileXR::CommArgs* args, int32_t targetRank, int32_t rank, uint32_t slot, int32_t rankSize)
{
    return reinterpret_cast<__gm__ uint64_t*>(
        args->peerMems[targetRank] + BigDataIpcAckOffset(slot, rankSize, rank));
}

__aicore__ inline __gm__ uint64_t* BigDataLocalIpcReadySlot(
    const __gm__ TileXR::CommArgs* args, int32_t rank, uint32_t slot, int32_t rankSize,
    int32_t peer, uint32_t segmentId)
{
    return BigDataLocalIpcAckSlot(args, rank, slot, rankSize, peer) + 1U + segmentId;
}

__aicore__ inline __gm__ uint64_t* BigDataRemoteIpcReadySlot(
    const __gm__ TileXR::CommArgs* args, int32_t targetRank, int32_t rank, uint32_t slot,
    int32_t rankSize, uint32_t segmentId)
{
    return BigDataRemoteIpcAckSlot(args, targetRank, rank, slot, rankSize) + 1U + segmentId;
}

__aicore__ inline uint32_t BigDataPublishAckSignalUdma(
    const __gm__ TileXR::CommArgs* args, int32_t peer, __gm__ uint8_t* udmaMem,
    uint64_t ackSignalOffset, uint32_t slot, int32_t rankSize, uint32_t shardCount,
    int32_t rank, uint64_t token, AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint64_t ackOffset =
        BigDataRegisteredControlOffset(ackSignalOffset, slot, rankSize, shardCount, rank, 0U);
    auto localAck = reinterpret_cast<__gm__ uint64_t*>(udmaMem + ackOffset);
    BigDataStoreTokenMte(localAck, token, relayLocal);
    TileXR::UDMAPutSignalNbi<uint64_t>(
        args, peer, localAck, ackOffset, sizeof(uint64_t), ackOffset, token);
    return TileXR::UDMAQuietStatus(args, peer);
}

__aicore__ inline void BigDataRemotePutOnlyPublishAck(
    __gm__ uint64_t* remoteAck, uint64_t token, AscendC::LocalTensor<uint8_t> relayLocal)
{
    BigDataStoreTokenMte(remoteAck, token, relayLocal);
}

__aicore__ inline void BigDataRemotePutOnlyPublishReady(
    __gm__ uint64_t* remoteReady, uint64_t token, AscendC::LocalTensor<uint8_t> relayLocal)
{
    BigDataStoreTokenMte(remoteReady, token, relayLocal);
}

__aicore__ inline uint64_t BigDataRemotePutOnlyWaitPeerAck(
    __gm__ uint64_t* peerAck, uint64_t token, AscendC::LocalTensor<uint8_t> relayLocal)
{
    return BigDataWaitTokenMte(peerAck, token, relayLocal);
}

__aicore__ inline void BigDataCopyPeerWorker(
    int32_t peer, int32_t rank, int32_t rankSize, const __gm__ TileXR::CommArgs* args,
    __gm__ int32_t* input, __gm__ int32_t* output,
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, int32_t elementsPerPeer, int32_t effectiveChunkElements,
    uint32_t passCount, uint32_t loop, uint32_t pass, uint64_t kernelLoopBase, uint32_t profileStage,
    bool use35Core, uint32_t copyShard, uint32_t shardCount, uint64_t sendDataOffset,
    uint64_t copyDoneOffset, uint64_t ackSignalOffset, uint64_t chunkBytesPerPeer,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint32_t dataShardCount = use35Core ?
        TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES : shardCount;
    if (peer < 0 || peer >= rankSize || shardCount == 0U || copyShard >= dataShardCount) {
        return;
    }
    if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_PREPARE) {
        return;
    }

    const int32_t networkPeerCount = rankSize > 1 ? rankSize - 1 : 1;
    int32_t chunkOffset = 0;
    uint32_t chunkBytes = 0U;
    if (!BigDataPassChunk(pass, elementsPerPeer, effectiveChunkElements, chunkOffset, chunkBytes)) {
        return;
    }
    uint32_t shardOffset = 0U;
    uint32_t shardBytes = 0U;
    const bool shardHasBytes =
        BigDataCopyShardRange(copyShard, dataShardCount, chunkBytes / sizeof(int32_t), shardOffset, shardBytes);
    const uint64_t globalPass = BigDataGlobalPassIndex(kernelLoopBase, passCount, loop, pass);
    const uint64_t token = BigDataPassToken(globalPass);
    const uint32_t slot = BigDataDataSlot(globalPass, pass, use35Core);

    if (peer == rank) {
        auto src = reinterpret_cast<__gm__ uint8_t*>(
            input + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset);
        auto dst = reinterpret_cast<__gm__ uint8_t*>(
            output + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset);
        if (shardHasBytes) {
            BigDataCopyRangePingPong(dst + shardOffset, src + shardOffset, shardBytes, relayLocal);
        }
        return;
    }

    if (!use35Core && profileStage > TILEXR_BIGDATA_PROFILE_STAGE_ACK_PUT &&
        globalPass >= TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS) {
        const uint64_t reuseToken = BigDataPassToken(
            globalPass - static_cast<uint64_t>(TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS));
        const uint64_t observed = BigDataWaitTokenMte(
            BigDataLocalIpcAckSlot(args, rank, slot, rankSize, peer),
            reuseToken, relayLocal);
        if (observed < reuseToken) {
            if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS;
            }
            return;
        }
    }
    if (use35Core && profileStage == TILEXR_BIGDATA_PROFILE_STAGE_ACK_PUT &&
        passCount > 0U && globalPass >= static_cast<uint64_t>(passCount) &&
        copyShard == 0U) {
        const uint64_t reuseToken = BigDataPassToken(globalPass - static_cast<uint64_t>(passCount));
        const uint64_t observed = BigDataLoadTokenMte(
            BigDataControlSlot(udmaMem, ackSignalOffset, slot, rankSize, shardCount, peer, 0U),
            relayLocal);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_ACK_SEEN_BASE + peer] =
                static_cast<int32_t>(observed);
            if (observed < reuseToken) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS;
            }
        }
    }

    auto src = reinterpret_cast<__gm__ uint8_t*>(
        input + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset);
    const int32_t peerIndex = BigDataNetworkPeerIndex(peer, rank);
    auto sendSlot = BigDataSlot(udmaMem, sendDataOffset, slot, networkPeerCount,
        peerIndex, chunkBytesPerPeer);
    if (shardHasBytes) {
        BigDataCopyRangePingPong(sendSlot + shardOffset, src + shardOffset, shardBytes, relayLocal);
    }
    BigDataStoreTokenMte(
        BigDataControlSlot(udmaMem, copyDoneOffset, slot, rankSize, shardCount, peer, copyShard),
        token, relayLocal);
}

__aicore__ inline bool BigDataWaitCopyReady(
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, uint64_t copyReadyOffset,
    uint32_t slot, int32_t rankSize, uint32_t shardCount, int32_t peer,
    uint32_t readyShard, uint64_t token, uint32_t loop, uint32_t pass,
    AscendC::LocalTensor<uint8_t> relayLocal);

__aicore__ inline void BigDataSendPeerWorker(
    int32_t peer, int32_t rank, int32_t rankSize, __gm__ TileXR::CommArgs* args,
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, int32_t elementsPerPeer,
    int32_t effectiveChunkElements, uint32_t passCount, uint32_t loop, uint32_t pass,
    uint64_t kernelLoopBase, uint32_t profileStage, uint32_t shardCount, uint64_t sendDataOffset,
    uint64_t recvDataOffset, uint64_t copyDoneOffset, uint64_t readySignalOffset,
    uint64_t chunkBytesPerPeer, bool use35Core, uint64_t copyReadyOffset,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    if (peer < 0 || peer >= rankSize || peer == rank || shardCount == 0U ||
        profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_SEND_COPY) {
        return;
    }

    const int32_t peerIndex = BigDataNetworkPeerIndex(peer, rank);
    const int32_t networkPeerCount = rankSize > 1 ? rankSize - 1 : 1;
    int32_t chunkOffset = 0;
    uint32_t chunkBytes = 0U;
    if (!BigDataPassChunk(pass, elementsPerPeer, effectiveChunkElements, chunkOffset, chunkBytes)) {
        return;
    }
    const uint64_t globalPass = BigDataGlobalPassIndex(kernelLoopBase, passCount, loop, pass);
    const uint64_t token = BigDataPassToken(globalPass);
    const uint32_t slot = BigDataDataSlot(globalPass, pass, use35Core);

    if (use35Core) {
        if (!BigDataWaitCopyReady(udmaMem, debug, copyReadyOffset, slot, rankSize, shardCount,
                peer, TILEXR_UDMA_DEMO_BIGDATA_LOCAL_COPY_READY, token, loop, pass, relayLocal)) {
            return;
        }
    } else {
        for (uint32_t copyShard = 0U; copyShard < shardCount; ++copyShard) {
            const uint64_t observed = BigDataWaitTokenMte(
                BigDataControlSlot(udmaMem, copyDoneOffset, slot, rankSize, shardCount, peer, copyShard),
                token, relayLocal);
            if (observed < token) {
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                        TILEXR_UDMA_DEMO_COPY_TIMEOUT_STATUS;
                }
                return;
            }
        }
    }
    if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_SEND_SYNC) {
        return;
    }

    auto sendSlot = BigDataSlot(udmaMem, sendDataOffset, slot, networkPeerCount,
        peerIndex, chunkBytesPerPeer);
    auto localSrc = reinterpret_cast<__gm__ int32_t*>(sendSlot);
    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
        debug[TILEXR_UDMA_DEMO_DEBUG_SEND_SAMPLE_BASE + peer] = localSrc[0];
    }
    const uint64_t remoteDataOffset =
        recvDataOffset +
        (static_cast<uint64_t>(slot) * static_cast<uint64_t>(networkPeerCount) +
        static_cast<uint64_t>(BigDataNetworkPeerIndex(rank, peer))) * chunkBytesPerPeer;
    const uint64_t remoteReadyOffset =
        readySignalOffset +
        ((static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
        static_cast<uint64_t>(rank)) *
        static_cast<uint64_t>(shardCount)) *
        TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES;
    auto registry = TileXR::GetUDMARegistry(args);
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    auto wqCtx = TileXR::UDMAGetWQCtx(udmaInfo, peer, 0);
    auto remoteMemInfo = TileXR::UDMAGetRemoteMemInfo(udmaInfo, peer, 0);
    bool rangeValid = TileXR::UDMARegisteredRangeValid(registry, peer, remoteDataOffset, chunkBytes) &&
        TileXR::UDMARegisteredRangeValid(registry, peer, remoteReadyOffset, sizeof(uint64_t));
    uint32_t wqeBefore = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
        debug[TILEXR_UDMA_DEMO_DEBUG_RANGE_VALID_BASE + peer] = rangeValid ? 1 : 0;
        debug[TILEXR_UDMA_DEMO_DEBUG_WQE_BEFORE_BASE + peer] = static_cast<int32_t>(wqeBefore);
        debug[TILEXR_UDMA_DEMO_DEBUG_LOCAL_TOKEN_BASE + peer] =
            static_cast<int32_t>(wqCtx->localTokenId);
        debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_BASE_LOW_BASE + peer] =
            static_cast<int32_t>(reinterpret_cast<uint64_t>(registry->regions[peer].base) &
                0xFFFFFFFFU);
        debug[TILEXR_UDMA_DEMO_DEBUG_MEM_ADDR_LOW_BASE + peer] =
            static_cast<int32_t>(remoteMemInfo->addr & 0xFFFFFFFFU);
        debug[TILEXR_UDMA_DEMO_DEBUG_TPN_BASE + peer] = static_cast<int32_t>(remoteMemInfo->tpn);
        debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_DATA_OFFSET_BASE + peer] =
            static_cast<int32_t>(remoteDataOffset);
        debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_READY_OFFSET_BASE + peer] =
            static_cast<int32_t>(remoteReadyOffset);
    }

    TileXR::UDMAPutSignalNbi<int32_t>(args, peer, localSrc,
        remoteDataOffset, chunkBytes, remoteReadyOffset, token);
    uint32_t status = TileXR::UDMAQuietStatus(args, peer);
    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
        uint32_t wqeAfter = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
        debug[TILEXR_UDMA_DEMO_DEBUG_WQE_AFTER_BASE + peer] = static_cast<int32_t>(wqeAfter);
        debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
    }
}

__aicore__ inline void BigDataRecvPeerWorker(
    int32_t peer, int32_t rank, int32_t rankSize, __gm__ TileXR::CommArgs* args,
    __gm__ int32_t* output, __gm__ uint8_t* udmaMem, __gm__ int32_t* debug,
    int32_t elementsPerPeer, int32_t effectiveChunkElements, uint32_t passCount,
    uint32_t loop, uint32_t pass, uint64_t kernelLoopBase, uint32_t profileStage,
    bool use35Core, uint32_t recvShard, uint32_t shardCount, uint64_t recvDataOffset,
    uint64_t recvCopyDoneOffset, uint64_t readySignalOffset, uint64_t ackSignalOffset,
    uint64_t chunkBytesPerPeer, AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint32_t dataShardCount = use35Core ?
        TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORES : shardCount;
    if (peer < 0 || peer >= rankSize || peer == rank ||
        shardCount == 0U || recvShard >= dataShardCount ||
        profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_DATA_PUT) {
        return;
    }
    (void)ackSignalOffset;

    const int32_t peerIndex = BigDataNetworkPeerIndex(peer, rank);
    const int32_t networkPeerCount = rankSize > 1 ? rankSize - 1 : 1;
    int32_t chunkOffset = 0;
    uint32_t chunkBytes = 0U;
    if (!BigDataPassChunk(pass, elementsPerPeer, effectiveChunkElements, chunkOffset, chunkBytes)) {
        return;
    }
    uint32_t shardOffset = 0U;
    uint32_t shardBytes = 0U;
    const bool shardHasBytes =
        BigDataCopyShardRange(recvShard, dataShardCount, chunkBytes / sizeof(int32_t), shardOffset, shardBytes);
    const uint64_t globalPass = BigDataGlobalPassIndex(kernelLoopBase, passCount, loop, pass);
    const uint64_t token = BigDataPassToken(globalPass);
    const uint32_t slot = BigDataDataSlot(globalPass, pass, use35Core);

    uint64_t observed = 0ULL;
    if (use35Core) {
        if (recvShard == TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_WAIT_SHARD) {
            observed = BigDataWaitTokenMte(
                BigDataControlSlot(udmaMem, readySignalOffset, slot, rankSize, shardCount,
                    peer, TILEXR_UDMA_DEMO_BIGDATA_RECV_READY_SOURCE_SHARD),
                token, relayLocal);
            if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                debug[TILEXR_UDMA_DEMO_DEBUG_READY_SEEN_BASE + peer] =
                    static_cast<int32_t>(observed);
                if (observed < token) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                        TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS;
                }
            }
            if (observed < token) {
                return;
            }
            for (uint32_t fanoutShard = 0U; fanoutShard < dataShardCount; ++fanoutShard) {
                BigDataStoreTokenMte(
                    BigDataControlSlot(udmaMem, readySignalOffset, slot, rankSize, shardCount,
                        peer, TILEXR_UDMA_DEMO_BIGDATA_LOCAL_FANOUT_SHARD_BASE + fanoutShard),
                    token, relayLocal);
            }
        }
        observed = BigDataWaitTokenMte(
            BigDataControlSlot(udmaMem, readySignalOffset, slot, rankSize, shardCount,
                peer, TILEXR_UDMA_DEMO_BIGDATA_LOCAL_FANOUT_SHARD_BASE + recvShard),
            token, relayLocal);
        if (observed < token) {
            if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS;
            }
            return;
        }
    } else {
        observed = BigDataWaitTokenMte(
            BigDataControlSlot(udmaMem, readySignalOffset, slot, rankSize, shardCount, peer, 0U),
            token, relayLocal);
        if (debug != nullptr && recvShard == 0U && loop == 0 && pass == 0 && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_READY_SEEN_BASE + peer] =
                static_cast<int32_t>(observed);
            if (observed < token) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS;
            }
        }
        if (observed < token) {
            return;
        }
    }
    if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_WAIT_READY) {
        return;
    }

    auto recvSlot = BigDataSlot(udmaMem, recvDataOffset, slot, networkPeerCount,
        peerIndex, chunkBytesPerPeer);
    if (debug != nullptr && recvShard == 0U && loop == 0 && pass == 0 && peer < 16) {
        auto recvSlotInt = reinterpret_cast<__gm__ int32_t*>(recvSlot);
        debug[TILEXR_UDMA_DEMO_DEBUG_RECV_SLOT_SAMPLE_BASE + peer] = recvSlotInt[0];
    }
    auto dst = reinterpret_cast<__gm__ uint8_t*>(
        output + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset);
    if (shardHasBytes) {
        BigDataCopyRangePingPong(dst + shardOffset, recvSlot + shardOffset, shardBytes, relayLocal);
    }
    if (debug != nullptr && recvShard == 0U && loop == 0 && pass == 0 && peer < 16) {
        auto relayDst = output + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset;
        debug[TILEXR_UDMA_DEMO_DEBUG_RECV_SAMPLE_BASE + peer] = relayDst[0];
    }
    if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_RELAY_COPY) {
        return;
    }

    BigDataStoreTokenMte(
        BigDataControlSlot(udmaMem, recvCopyDoneOffset, slot, rankSize, shardCount, peer, recvShard),
        token, relayLocal);
    if (recvShard + 1U != shardCount) {
        if (use35Core && recvShard + 1U == dataShardCount) {
            for (uint32_t shard = 0U; shard < dataShardCount; ++shard) {
                observed = BigDataWaitTokenMte(
                    BigDataControlSlot(udmaMem, recvCopyDoneOffset, slot, rankSize, shardCount, peer, shard),
                    token, relayLocal);
                if (observed < token) {
                    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                        debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                            TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS;
                    }
                    return;
                }
            }

            const uint32_t ackStatus = BigDataPublishAckSignalUdma(
                args, peer, udmaMem, ackSignalOffset, slot, rankSize, shardCount,
                rank, token, relayLocal);
            if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    static_cast<int32_t>(ackStatus);
            }
        }
        return;
    }
    for (uint32_t shard = 0U; shard < shardCount; ++shard) {
        observed = BigDataWaitTokenMte(
            BigDataControlSlot(udmaMem, recvCopyDoneOffset, slot, rankSize, shardCount, peer, shard),
            token, relayLocal);
        if (observed < token) {
            if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS;
            }
            return;
        }
    }

    __gm__ uint64_t* remoteAck = nullptr;
    if (use35Core) {
        const uint32_t ackStatus = BigDataPublishAckSignalUdma(
            args, peer, udmaMem, ackSignalOffset, slot, rankSize, shardCount,
            rank, token, relayLocal);
        if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                static_cast<int32_t>(ackStatus);
        }
        return;
    } else {
        remoteAck = BigDataRemoteIpcAckSlot(args, peer, rank, slot, rankSize);
    }
    if (remoteAck == nullptr) {
        if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS;
        }
        return;
    }
    BigDataStoreTokenMte(remoteAck, token, relayLocal);
}

__aicore__ inline bool BigDataWaitCopyDoneRange(
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, uint64_t copyDoneOffset,
    uint32_t slot, int32_t rankSize, uint32_t shardCount, int32_t peer,
    uint32_t copyShardBegin, uint32_t copyShardEnd, uint64_t token,
    uint32_t loop, uint32_t pass, AscendC::LocalTensor<uint8_t> relayLocal)
{
    if (copyShardBegin > copyShardEnd || copyShardEnd > shardCount) {
        return false;
    }
    for (uint32_t copyShard = copyShardBegin; copyShard < copyShardEnd; ++copyShard) {
        const uint64_t observed = BigDataWaitTokenMte(
            BigDataControlSlot(udmaMem, copyDoneOffset, slot, rankSize, shardCount, peer, copyShard),
            token, relayLocal);
        if (observed < token) {
            if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    TILEXR_UDMA_DEMO_COPY_TIMEOUT_STATUS;
            }
            return false;
        }
    }
    return true;
}

__aicore__ inline bool BigDataPublishCopyReadyRange(
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, uint64_t copyDoneOffset,
    uint64_t copyReadyOffset, uint32_t slot, int32_t rankSize, uint32_t shardCount,
    int32_t peer, uint32_t copyShardBegin, uint32_t copyShardEnd, uint32_t readyShard,
    uint64_t token, uint32_t loop, uint32_t pass, AscendC::LocalTensor<uint8_t> relayLocal)
{
    if (!BigDataWaitCopyDoneRange(udmaMem, debug, copyDoneOffset, slot, rankSize,
            shardCount, peer, copyShardBegin, copyShardEnd, token, loop, pass, relayLocal)) {
        return false;
    }
    BigDataStoreTokenMte(
        BigDataControlSlot(udmaMem, copyReadyOffset, slot, rankSize, shardCount, peer, readyShard),
        token, relayLocal);
    return true;
}

__aicore__ inline bool BigDataWaitCopyReady(
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, uint64_t copyReadyOffset,
    uint32_t slot, int32_t rankSize, uint32_t shardCount, int32_t peer,
    uint32_t readyShard, uint64_t token, uint32_t loop, uint32_t pass,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint64_t observed = BigDataWaitTokenMte(
        BigDataControlSlot(udmaMem, copyReadyOffset, slot, rankSize, shardCount, peer, readyShard),
        token, relayLocal);
    if (observed < token) {
        if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                TILEXR_UDMA_DEMO_COPY_TIMEOUT_STATUS;
        }
        return false;
    }
    return true;
}

__aicore__ inline bool BigDataRemoteSendSegmentRange(
    uint32_t segmentId, uint32_t dataShardCount, uint32_t chunkElements,
    uint32_t& copyShardBegin, uint32_t& copyShardEnd,
    uint32_t& segmentOffsetBytes, uint32_t& segmentBytes)
{
    if (dataShardCount == 0U) {
        return false;
    }
    const uint32_t chunkBytes = chunkElements * static_cast<uint32_t>(sizeof(int32_t));
    const uint32_t splitOffset = BigDataCopyShardStartBytes(
        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END, dataShardCount, chunkElements);
    if (segmentId == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT) {
        copyShardBegin = 0U;
        copyShardEnd = TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END;
        segmentOffsetBytes = 0U;
        segmentBytes = splitOffset;
    } else if (segmentId == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_SEGMENT) {
        copyShardBegin = TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END;
        copyShardEnd = dataShardCount;
        segmentOffsetBytes = splitOffset;
        segmentBytes = chunkBytes - splitOffset;
    } else {
        return false;
    }
    if (copyShardEnd > dataShardCount || segmentOffsetBytes > chunkBytes) {
        return false;
    }
    return true;
}

__aicore__ inline void BigDataPublishReadySignal(
    __gm__ TileXR::CommArgs* args, int32_t peer, __gm__ uint8_t* udmaMem,
    uint64_t readySignalOffset, uint32_t slot, int32_t rankSize, uint32_t shardCount,
    int32_t rank, uint64_t token)
{
    const uint64_t localReadyPayloadOffset =
        readySignalOffset +
        ((static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize) +
        static_cast<uint64_t>(rank)) * static_cast<uint64_t>(shardCount)) *
        TILEXR_UDMA_DEMO_BIGDATA_CONTROL_SLOT_BYTES;
    const uint64_t remoteReadyOffset = localReadyPayloadOffset;
    auto localSrc = reinterpret_cast<__gm__ uint64_t*>(udmaMem + localReadyPayloadOffset);
    TileXR::UDMAPutSignalNbi<uint64_t>(
        args, peer, localSrc, localReadyPayloadOffset, sizeof(uint64_t), remoteReadyOffset, token);
    (void)TileXR::UDMAQuietStatus(args, peer);
}

__aicore__ inline void BigDataRemoteSendSegmentWorker(
    int32_t peer, uint32_t segmentId, int32_t rank, int32_t rankSize, __gm__ TileXR::CommArgs* args,
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, int32_t elementsPerPeer,
    int32_t effectiveChunkElements, uint32_t passCount, uint32_t loop, uint32_t pass,
    uint64_t kernelLoopBase, uint32_t profileStage, uint32_t shardCount,
    uint64_t sendDataOffset, uint64_t recvDataOffset, uint64_t copyDoneOffset,
    uint64_t remoteSendDoneOffset, uint64_t readySignalOffset, uint64_t chunkBytesPerPeer,
    int32_t ranksPerNode, AscendC::LocalTensor<uint8_t> relayLocal)
{
    if (peer < 0 || peer >= rankSize || peer == rank || !BigDataIsMultiNode(rankSize, ranksPerNode) ||
        shardCount == 0U || profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_SEND_COPY) {
        return;
    }

    const int32_t peerIndex = BigDataNetworkPeerIndex(peer, rank);
    const int32_t networkPeerCount = rankSize > 1 ? rankSize - 1 : 1;
    int32_t chunkOffset = 0;
    uint32_t chunkBytes = 0U;
    if (!BigDataPassChunk(pass, elementsPerPeer, effectiveChunkElements, chunkOffset, chunkBytes)) {
        return;
    }
    (void)chunkOffset;
    (void)chunkBytes;
    const uint64_t globalPass = BigDataGlobalPassIndex(kernelLoopBase, passCount, loop, pass);
    const uint64_t token = BigDataPassToken(globalPass);
    const uint32_t slot = BigDataDataSlot(globalPass, pass, true);
    constexpr uint32_t dataShardCount = TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES;
    (void)copyDoneOffset;

    uint32_t copyShardBegin = 0U;
    uint32_t copyShardEnd = 0U;
    uint32_t segmentOffsetBytes = 0U;
    uint32_t segmentBytes = 0U;
    if (!BigDataRemoteSendSegmentRange(
            segmentId, dataShardCount, chunkBytes / sizeof(int32_t),
            copyShardBegin, copyShardEnd, segmentOffsetBytes, segmentBytes)) {
        return;
    }
    (void)copyShardBegin;
    (void)copyShardEnd;

    const uint32_t readyShard =
        (segmentId == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT) ?
        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_PRIMARY :
        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_SECONDARY;
    if (!BigDataWaitCopyReady(udmaMem, debug, remoteSendDoneOffset, slot, rankSize,
            shardCount, peer, readyShard, token, loop, pass, relayLocal)) {
        return;
    }
    if (profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_SEND_SYNC) {
        return;
    }

    auto sendSlot = BigDataSlot(udmaMem, sendDataOffset, slot, networkPeerCount,
        peerIndex, chunkBytesPerPeer);
    auto localSrc = reinterpret_cast<__gm__ int32_t*>(sendSlot + segmentOffsetBytes);
    const uint64_t remoteDataOffset =
        recvDataOffset +
        (static_cast<uint64_t>(slot) * static_cast<uint64_t>(networkPeerCount) +
        static_cast<uint64_t>(BigDataNetworkPeerIndex(rank, peer))) * chunkBytesPerPeer +
        static_cast<uint64_t>(segmentOffsetBytes);

    uint32_t status = 0U;
    if (segmentBytes > 0U) {
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteDataOffset, segmentBytes);
        status = TileXR::UDMAQuietStatus(args, peer);
    }
    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
        debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
    }
    BigDataStoreTokenMte(
        BigDataControlSlot(udmaMem, remoteSendDoneOffset, slot, rankSize, shardCount, peer, segmentId),
        token, relayLocal);

    if (segmentId != TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT ||
        profileStage <= TILEXR_BIGDATA_PROFILE_STAGE_DATA_PUT) {
        return;
    }
    for (uint32_t done = 0U; done <= TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_SEGMENT; ++done) {
        const uint64_t observed = BigDataWaitTokenMte(
            BigDataControlSlot(udmaMem, remoteSendDoneOffset, slot, rankSize, shardCount, peer, done),
            token, relayLocal);
        if (observed < token) {
            if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] =
                    TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS;
            }
            return;
        }
    }
    auto localReady = BigDataControlSlot(
        udmaMem, readySignalOffset, slot, rankSize, shardCount, rank, 0U);
    BigDataStoreTokenMte(localReady, token, relayLocal);
    BigDataPublishReadySignal(
        args, peer, udmaMem, readySignalOffset, slot, rankSize, shardCount, rank, token);
}

__aicore__ inline void BigDataRemotePutOnlySendWorker(
    int32_t peer, uint32_t segmentId, uint32_t qpIdx, int32_t rank, int32_t rankSize,
    __gm__ TileXR::CommArgs* args, __gm__ int32_t* input,
    __gm__ int32_t* debug, int32_t elementsPerPeer, int32_t effectiveChunkElements,
    uint32_t passCount, uint32_t loop, uint32_t pass, uint64_t kernelLoopBase,
    uint32_t profileStage, uint32_t shardCount, uint64_t recvDataOffset,
    uint64_t chunkBytesPerPeer, int32_t ranksPerNode, AscendC::LocalTensor<uint8_t> relayLocal)
{
    if (peer < 0 || peer >= rankSize || peer == rank || BigDataIsLocalPeer(rank, peer, ranksPerNode) ||
        !BigDataIsMultiNode(rankSize, ranksPerNode) || shardCount == 0U ||
        profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_LOOP) {
        return;
    }

    int32_t chunkOffset = 0;
    uint32_t chunkBytes = 0U;
    if (!BigDataPassChunk(pass, elementsPerPeer, effectiveChunkElements, chunkOffset, chunkBytes)) {
        return;
    }
    const uint64_t globalPass = BigDataGlobalPassIndex(kernelLoopBase, passCount, loop, pass);
    const uint32_t slot = BigDataDataSlot(globalPass, pass, true);
    constexpr uint32_t dataShardCount = TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES;

    uint32_t copyShardBegin = 0U;
    uint32_t copyShardEnd = 0U;
    uint32_t segmentOffsetBytes = 0U;
    uint32_t segmentBytes = 0U;
    if (!BigDataRemoteSendSegmentRange(
            segmentId, dataShardCount, chunkBytes / sizeof(int32_t),
            copyShardBegin, copyShardEnd, segmentOffsetBytes, segmentBytes)) {
        return;
    }
    (void)copyShardBegin;
    (void)copyShardEnd;
    if (profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_SEGMENT) {
        return;
    }

    auto localSrc = reinterpret_cast<__gm__ int32_t*>(
        reinterpret_cast<__gm__ uint8_t*>(
            input + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset) +
        segmentOffsetBytes);
    const uint64_t remoteDataOffset =
        recvDataOffset +
        (static_cast<uint64_t>(slot) * static_cast<uint64_t>(rankSize > 1 ? rankSize - 1 : 1) +
        static_cast<uint64_t>(BigDataNetworkPeerIndex(rank, peer))) * chunkBytesPerPeer +
        static_cast<uint64_t>(segmentOffsetBytes);
    if (profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_ADDRESS) {
        return;
    }

    uint32_t status = 0U;
    if (segmentBytes > 0U) {
        TileXR::UDMAPutNbiOnQp<int32_t>(args, peer, qpIdx, localSrc, remoteDataOffset, segmentBytes);
        status = TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
        BigDataRemotePutOnlyPublishReady(
            BigDataRemoteIpcReadySlot(args, peer, rank, slot, rankSize, segmentId),
            BigDataPassToken(globalPass), relayLocal);
    }
    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
        debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
    }
}

__aicore__ inline int32_t BigDataRemotePutOnlyCheckIndex(int32_t blockIdx)
{
    if (blockIdx < 0) {
        return -1;
    }
    return blockIdx;
}

__aicore__ inline int32_t BigDataRemotePutOnlySendTaskCount(int32_t remoteTaskCount)
{
    return remoteTaskCount > 0 ? remoteTaskCount * 2 : 0;
}

__aicore__ inline int32_t BigDataRemotePutOnlySendTaskRemoteIndex(
    int32_t sendTask, int32_t remoteTaskCount)
{
    if (sendTask < 0 || remoteTaskCount <= 0 ||
        sendTask >= BigDataRemotePutOnlySendTaskCount(remoteTaskCount)) {
        return -1;
    }
    return sendTask < remoteTaskCount ? sendTask : sendTask - remoteTaskCount;
}

__aicore__ inline uint32_t BigDataRemotePutOnlySendTaskSegment(
    int32_t sendTask, int32_t remoteTaskCount)
{
    return sendTask < remoteTaskCount ?
        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT :
        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_SEGMENT;
}

__aicore__ inline void BigDataRemotePutOnlyCheckWorker(
    int32_t remoteIndex, int32_t rank, int32_t rankSize, __gm__ TileXR::CommArgs* args,
    __gm__ uint8_t* udmaMem, __gm__ int32_t* debug, int32_t elementsPerPeer,
    int32_t effectiveChunkElements, uint32_t passCount, uint32_t loop, uint32_t pass,
    uint64_t kernelLoopBase, uint64_t recvDataOffset, uint64_t ackSignalOffset,
    uint64_t chunkBytesPerPeer, uint32_t shardCount, int32_t ranksPerNode,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    const int32_t peer = BigDataRemotePeerForwardAt(rank, rankSize, remoteIndex, ranksPerNode);
    if (peer < 0 || peer >= rankSize || peer == rank || BigDataIsLocalPeer(rank, peer, ranksPerNode) ||
        shardCount == 0U) {
        return;
    }
    int32_t chunkOffset = 0;
    uint32_t chunkBytes = 0U;
    if (!BigDataPassChunk(pass, elementsPerPeer, effectiveChunkElements, chunkOffset, chunkBytes) ||
        chunkBytes < sizeof(int32_t)) {
        return;
    }
    const uint64_t globalPass = BigDataGlobalPassIndex(kernelLoopBase, passCount, loop, pass);
    const uint64_t token = BigDataPassToken(globalPass);
    const uint32_t slot = BigDataDataSlot(globalPass, pass, true);
    (void)recvDataOffset;
    (void)chunkBytesPerPeer;
    (void)chunkBytes;

    uint32_t status = 0U;
    uint64_t observed = 0;
    constexpr uint32_t dataShardCount = TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES;
    for (uint32_t segmentId = TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT;
         segmentId <= TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_SEGMENT; ++segmentId) {
        uint32_t copyShardBegin = 0U;
        uint32_t copyShardEnd = 0U;
        uint32_t segmentOffsetBytes = 0U;
        uint32_t segmentBytes = 0U;
        if (!BigDataRemoteSendSegmentRange(
                segmentId, dataShardCount, chunkBytes / sizeof(int32_t),
                copyShardBegin, copyShardEnd, segmentOffsetBytes, segmentBytes) ||
            segmentBytes < sizeof(int32_t)) {
            continue;
        }
        (void)copyShardBegin;
        (void)copyShardEnd;
        (void)segmentOffsetBytes;
        (void)segmentBytes;
        observed = BigDataWaitTokenMte(
            BigDataLocalIpcReadySlot(args, rank, slot, rankSize, peer, segmentId),
            token, relayLocal);
        if (observed < token) {
            status = static_cast<uint32_t>(TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS);
            break;
        }
    }
    if (status == 0U) {
        auto remoteAck = BigDataRemoteIpcAckSlot(args, peer, rank, slot, rankSize);
        auto localAck = BigDataLocalIpcAckSlot(args, rank, slot, rankSize, peer);
        BigDataRemotePutOnlyPublishAck(remoteAck, token, relayLocal);
        const uint64_t ackObserved = BigDataRemotePutOnlyWaitPeerAck(
            localAck, token, relayLocal);
        if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_ACK_SEEN_BASE + peer] =
                static_cast<int32_t>(ackObserved);
        }
        if (ackObserved < token) {
            status = static_cast<uint32_t>(TILEXR_UDMA_DEMO_ACK_TIMEOUT_STATUS);
        }
    }
    if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
        debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
    }
}

__aicore__ inline void BigDataRunSelfCopyShard(
    int32_t rank, int32_t rankSize, __gm__ int32_t* input, __gm__ int32_t* output,
    int32_t elementsPerPeer, int32_t effectiveChunkElements, uint32_t pass,
    uint32_t copyShard, uint32_t shardCount, AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint32_t dataShardCount =
        (shardCount == TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_CONTROL_SHARDS) ?
        TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES : shardCount;
    if (rank < 0 || rank >= rankSize || shardCount == 0U || copyShard >= dataShardCount) {
        return;
    }
    int32_t chunkOffset = 0;
    uint32_t chunkBytes = 0U;
    if (!BigDataPassChunk(pass, elementsPerPeer, effectiveChunkElements, chunkOffset, chunkBytes)) {
        return;
    }
    uint32_t shardOffset = 0U;
    uint32_t shardBytes = 0U;
    if (!BigDataCopyShardRange(copyShard, dataShardCount, chunkBytes / sizeof(int32_t), shardOffset, shardBytes)) {
        return;
    }
    auto src = reinterpret_cast<__gm__ uint8_t*>(
        input + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset);
    auto dst = reinterpret_cast<__gm__ uint8_t*>(
        output + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset);
    BigDataCopyRangePingPong(dst + shardOffset, src + shardOffset, shardBytes, relayLocal);
}

__aicore__ inline void BigDataRunRoleForPeer(
    int32_t peer, int32_t role, int32_t rank, int32_t rankSize, __gm__ TileXR::CommArgs* args,
    __gm__ int32_t* input, __gm__ int32_t* output, __gm__ uint8_t* udmaMem,
    __gm__ int32_t* debug, int32_t elementsPerPeer, int32_t effectiveChunkElements,
    uint32_t passCount, uint32_t loop, uint32_t pass, uint64_t kernelLoopBase,
    uint32_t profileStage, bool use35Core, uint32_t shardCount, uint64_t sendDataOffset,
    uint64_t recvDataOffset, uint64_t copyDoneOffset, uint64_t recvCopyDoneOffset, uint64_t readySignalOffset,
    uint64_t ackSignalOffset, uint64_t chunkBytesPerPeer, AscendC::LocalTensor<uint8_t> relayLocal)
{
    const uint32_t dataShardCount = use35Core ?
        TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES : shardCount;
    if (role < static_cast<int32_t>(dataShardCount)) {
        BigDataCopyPeerWorker(peer, rank, rankSize, args, input, output, udmaMem, debug,
            elementsPerPeer, effectiveChunkElements, passCount, loop, pass, kernelLoopBase, profileStage,
            use35Core, static_cast<uint32_t>(role), shardCount, sendDataOffset, copyDoneOffset, ackSignalOffset,
            chunkBytesPerPeer, relayLocal);
        return;
    }
    if (role == static_cast<int32_t>(dataShardCount)) {
        BigDataSendPeerWorker(peer, rank, rankSize, args, udmaMem, debug,
            elementsPerPeer, effectiveChunkElements, passCount, loop, pass, kernelLoopBase, profileStage,
            shardCount, sendDataOffset, recvDataOffset, copyDoneOffset, readySignalOffset,
            chunkBytesPerPeer, use35Core, copyDoneOffset, relayLocal);
        return;
    }
    const uint32_t recvShard =
        static_cast<uint32_t>(role) - dataShardCount - 1U;
    BigDataRecvPeerWorker(peer, rank, rankSize, args, output, udmaMem, debug,
        elementsPerPeer, effectiveChunkElements, passCount, loop, pass, kernelLoopBase, profileStage,
        use35Core, recvShard, shardCount, recvDataOffset, recvCopyDoneOffset, readySignalOffset, ackSignalOffset,
        chunkBytesPerPeer, relayLocal);
}

} // namespace

extern "C" __global__ __aicore__ void tilexr_udma_all_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerRank;
    }
    if (!enabled) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)));
        TileXR::UDMAQuiet(args, peer);
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_put_signal_kernel(
    GM_ADDR commArgsGM, GM_ADDR dataGM, GM_ADDR signalGM, GM_ADDR debugGM,
    int32_t elementsPerRank, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto data = reinterpret_cast<__gm__ int32_t*>(dataGM);
    auto signals = reinterpret_cast<__gm__ uint64_t*>(signalGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerRank;
    }
    if (!enabled) {
        return;
    }

    auto localSrc = data + rank * elementsPerRank;
    uint64_t signalBaseOffset = static_cast<uint64_t>(rankSize) * elementsPerRank * sizeof(int32_t);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutSignalNbi<int32_t>(args, peer, localSrc,
            rank * static_cast<uint64_t>(elementsPerRank) * sizeof(int32_t),
            static_cast<uint32_t>(elementsPerRank * sizeof(int32_t)),
            signalBaseOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t), signal);
        TileXR::UDMAQuiet(args, peer);
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset, int32_t chunkElements)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    // Multi-core: one block per peer. Block b handles peer b:
    //   - peer == rank -> local self-copy via DataCopyPad
    //   - peer != rank -> UDMA PUT to that peer
    // Block 0 writes the shared debug header; per-peer slots are written by
    // their owning block only, so no cross-block debug races.
    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(outputByteOffset);
    }
    if (!enabled) {
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t payloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t bytes = static_cast<uint32_t>(payloadBytes);

    // This block's assigned peer. When host launches rankSize blocks, block b
    // handles peer b. If fewer blocks are launched, peers are round-robined
    // and each block may handle more than one peer (still correct, just less
    // parallel); stride == blockNum keeps peer slots disjoint across blocks.
    for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
        if (peer == rank) {
            // Self-copy: local DataCopyPad, no network.
            auto selfSrc = input + static_cast<uint64_t>(rank) * elementsPerPeer + inputElementOffset;
            auto selfDst = output + static_cast<uint64_t>(rank) * effectiveChunkElements;
            constexpr uint32_t SELF_COPY_UB_BYTES = 64 * 1024;
            AscendC::TPipe pipe;
            AscendC::TBuf<AscendC::QuePosition::VECCALC> selfCopyTBuf;
            pipe.InitBuffer(selfCopyTBuf, SELF_COPY_UB_BYTES);
            AscendC::LocalTensor<uint8_t> selfCopyLocal = selfCopyTBuf.Get<uint8_t>();

            auto selfSrcBytes = reinterpret_cast<__gm__ uint8_t*>(selfSrc);
            auto selfDstBytes = reinterpret_cast<__gm__ uint8_t*>(selfDst);
            for (uint32_t offset = 0; offset < bytes; offset += SELF_COPY_UB_BYTES) {
                uint32_t copyBytes = (bytes - offset < SELF_COPY_UB_BYTES)
                    ? (bytes - offset) : SELF_COPY_UB_BYTES;

                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(selfSrcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, copyBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(selfCopyLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(selfDstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, copyBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, selfCopyLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }

        // UDMA PUT to peer.
        auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer + inputElementOffset;
        uint64_t remoteOffset = outputByteOffset +
            static_cast<uint64_t>(rank) * payloadBytes;
        auto registry = TileXR::GetUDMARegistry(args);
        auto udmaInfo = TileXR::GetUDMAInfo(args);
        auto wqCtx = TileXR::UDMAGetWQCtx(udmaInfo, peer, 0);
        auto remoteMemInfo = TileXR::UDMAGetRemoteMemInfo(udmaInfo, peer, 0);
        bool rangeValid = TileXR::UDMARegisteredRangeValid(registry, peer, remoteOffset, bytes);
        uint32_t wqeBefore = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_RANGE_VALID_BASE + peer] = rangeValid ? 1 : 0;
            debug[TILEXR_UDMA_DEMO_DEBUG_WQE_BEFORE_BASE + peer] = static_cast<int32_t>(wqeBefore);
            debug[TILEXR_UDMA_DEMO_DEBUG_LOCAL_TOKEN_BASE + peer] = static_cast<int32_t>(wqCtx->localTokenId);
            debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_BASE_LOW_BASE + peer] =
                static_cast<int32_t>(reinterpret_cast<uint64_t>(registry->regions[peer].base) & 0xFFFFFFFFU);
            debug[TILEXR_UDMA_DEMO_DEBUG_MEM_ADDR_LOW_BASE + peer] =
                static_cast<int32_t>(remoteMemInfo->addr & 0xFFFFFFFFU);
            debug[TILEXR_UDMA_DEMO_DEBUG_TPN_BASE + peer] = static_cast<int32_t>(remoteMemInfo->tpn);
        }
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteOffset, bytes);
        uint32_t wqeAfter = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_WQE_AFTER_BASE + peer] = static_cast<int32_t>(wqeAfter);
        }
        uint32_t status = TileXR::UDMAQuietStatus(args, peer);
        if (debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    const uint64_t payloadBytes = AllToAllPayloadBytes(elementsPerPeer);
    const uint64_t segmentBytes = AllToAllDataAsFlagSegmentBytes(payloadBytes);
    auto inputBytes = reinterpret_cast<__gm__ uint8_t*>(input);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
    pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_DATA_AS_FLAG_UB_BYTES);
    AscendC::LocalTensor<uint8_t> scratch = tBuf.Get<uint8_t>();
    if (TileXR::DataAsFlagInit(scratch) == 0U) {
        if (debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = -1;
        }
        return;
    }

    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto localSrc = inputBytes + static_cast<uint64_t>(dstRank) * payloadBytes;
        auto remoteDst = reinterpret_cast<__gm__ uint8_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET +
            static_cast<uint64_t>(rank) * segmentBytes);
        (void)TileXR::DataAsFlagSend(remoteDst, localSrc, payloadBytes, scratch);
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    const uint64_t payloadBytes = AllToAllPayloadBytes(elementsPerPeer);
    const uint64_t segmentBytes = AllToAllDataAsFlagSegmentBytes(payloadBytes);
    auto outputBytes = reinterpret_cast<__gm__ uint8_t*>(output);
    auto localBase = reinterpret_cast<__gm__ uint8_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tBuf;
    pipe.InitBuffer(tBuf, TILEXR_UDMA_DEMO_DATA_AS_FLAG_UB_BYTES);
    AscendC::LocalTensor<uint8_t> scratch = tBuf.Get<uint8_t>();

    for (int32_t srcRank = 0; srcRank < args->rankSize; ++srcRank) {
        auto localSrc = localBase + static_cast<uint64_t>(srcRank) * segmentBytes;
        auto localDst = outputBytes + static_cast<uint64_t>(srcRank) * payloadBytes;
        if (!TileXR::DataAsFlagCheckAndRecv(localSrc, payloadBytes, localDst, scratch)) {
            if (debug != nullptr) {
                debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = -1;
            }
            return;
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_plain_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + static_cast<uint64_t>(rank) * rankSize * elementsPerPeer;
        auto localSrc = input + static_cast<uint64_t>(dstRank) * elementsPerPeer;
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            remoteDst[i] = localSrc[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 2;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_to_all_plain_ipc_gather_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rankSize = args->rankSize;
    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        auto localSrc = localBase + static_cast<uint64_t>(srcRank) * rankSize * elementsPerPeer;
        auto localDst = output + static_cast<uint64_t>(srcRank) * elementsPerPeer;
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            localDst[i] = localSrc[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 2;
    }
}

// Fused single-kernel alltoall: send + flag-sync + receive in one kernel launch.
// Each rank writes data to peers, sets a per-peer flag, polls peer flags, then reads.
// Flag layout: peerMems[peer] + srcRank * sizeof(uint64_t), within the 2MB flag region.
// Round parameter enables multi-round pipelining without host barriers.
extern "C" __global__ __aicore__ void tilexr_all_to_all_fused_ipc_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, int32_t round)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    const int64_t flagValue = static_cast<int64_t>(round);

    // UB: 64KB for DMA data relay + 64B for flag polling
    constexpr uint32_t DATA_UB_BYTES = 64 * 1024;
    constexpr uint32_t CHUNK_BYTES = DATA_UB_BYTES;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dataTBuf;
    pipe.InitBuffer(dataTBuf, DATA_UB_BYTES);
    AscendC::LocalTensor<uint8_t> dataLocal = dataTBuf.Get<uint8_t>();

    // Flag layout in peer's flag region (before IPC_DATA_OFFSET):
    //   [0 .. rankSize-1]                           = ready flags (data written)
    //   [rankSize .. 2*rankSize-1]                  = consumed ACK flags (data read)
    // Each flag is int64_t, indexed by srcRank.

    // Phase 1: Write data to all peers' IPC staging area via DMA (DataCopyPad)
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + static_cast<uint64_t>(rank) * rankSize * elementsPerPeer;
        auto localSrc = input + static_cast<uint64_t>(dstRank) * elementsPerPeer;

        auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
        auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(remoteDst);
        uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

        for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
            uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
            uint32_t alignedBytes = bytes & ~31U;

            if (alignedBytes > 0) {
                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(srcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(dstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            for (uint32_t i = alignedBytes; i < bytes; ++i) {
                dstBytes[offset + i] = srcBytes[offset + i];
            }
        }
    }

    // Phase 2: Send "data ready" flag to each peer via scalar store.
    // Scalar store to P2P address after MTE3_S barrier ensures all DMA
    // data writes are submitted before the flag.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        if (dstRank == rank) {
            continue;
        }
        auto flagAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[dstRank] + static_cast<uint64_t>(rank) * sizeof(int64_t));
        *flagAddr = flagValue;
    }

    // Phase 3: Poll all peers' "data ready" flags (read from own flag region).
    // Wait until each peer has written their data to our staging area.
    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        if (srcRank == rank) {
            continue;
        }
        auto flagAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[rank] + static_cast<uint64_t>(srcRank) * sizeof(int64_t));
        int64_t observed = 0;
        do {
            observed = *flagAddr;
        } while (observed < flagValue);
    }

    // Phase 4: Read data from local IPC staging area into output via DMA.
    // By now all peers' data has arrived (they set their ready flags).
    {
        auto localBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
            auto localSrc = localBase + static_cast<uint64_t>(srcRank) * rankSize * elementsPerPeer;
            auto localDst = output + static_cast<uint64_t>(srcRank) * elementsPerPeer;

            auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
            auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(localDst);
            uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

            for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
                uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
                uint32_t alignedBytes = bytes & ~31U;

                if (alignedBytes > 0) {
                    AscendC::GlobalTensor<uint8_t> srcGlobal;
                    srcGlobal.SetGlobalBuffer(srcBytes + offset);
                    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                    AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                    AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                    AscendC::GlobalTensor<uint8_t> dstGlobal;
                    dstGlobal.SetGlobalBuffer(dstBytes + offset);
                    AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                    AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                for (uint32_t i = alignedBytes; i < bytes; ++i) {
                    dstBytes[offset + i] = srcBytes[offset + i];
                }
            }
        }
    }

    // Phase 5: Send "data consumed" ACK to each peer.
    // Tells peers that we have read their data; they can safely overwrite
    // the staging area in the next round.
    // ACK flag offset: rankSize * sizeof(int64_t) + rank * sizeof(int64_t)
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        if (dstRank == rank) {
            continue;
        }
        auto ackAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[dstRank] +
            static_cast<uint64_t>(rankSize + rank) * sizeof(int64_t));
        *ackAddr = flagValue;
    }

    // Phase 6: Poll all peers' "data consumed" ACKs.
    // Wait until all peers confirm they have read our data.
    // Only then can we proceed to the next round (which will overwrite).
    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        if (srcRank == rank) {
            continue;
        }
        auto ackAddr = reinterpret_cast<__gm__ int64_t*>(
            args->peerMems[rank] +
            static_cast<uint64_t>(rankSize + srcRank) * sizeof(int64_t));
        int64_t observed = 0;
        do {
            observed = *ackAddr;
        } while (observed < flagValue);
    }

    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 3;
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 3;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_reduce_ipc_scatter_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + rank * elementsPerRank;
        for (int32_t i = 0; i < elementsPerRank; ++i) {
            remoteDst[i] = input[i];
        }
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SCATTER] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_all_reduce_ipc_sum_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerRank)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[args->rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
    for (int32_t i = 0; i < elementsPerRank; ++i) {
        int32_t sum = 0;
        for (int32_t srcRank = 0; srcRank < args->rankSize; ++srcRank) {
            sum += localBase[srcRank * elementsPerRank + i];
        }
        output[i] = sum;
    }
    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_ALLREDUCE_SUM] = 1;
    }
}

extern "C" __global__ __aicore__ void tilexr_udma_registered_smoke_kernel(
    GM_ADDR commArgsGM, GM_ADDR localGM, GM_ADDR debugGM, uint32_t bytes, uint64_t signal)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto local = reinterpret_cast<__gm__ uint8_t*>(localGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    bool enabled = TileXR::UDMARegistryEnabled(args);
    if (debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = enabled ? 1 : 0;
        debug[2] = static_cast<int32_t>(bytes);
    }
    if (!enabled) {
        return;
    }

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        TileXR::UDMAPutRegisteredNbi<uint8_t>(args, peer, local, 0, bytes);
        TileXR::UDMAGetRegisteredNbi<uint8_t>(args, peer, local, 0, bytes);
        TileXR::UDMAPutRegisteredSignalNbi<uint8_t>(args, peer, local, 0, bytes, 0, signal);
        TileXR::UDMAQuiet(args, peer);
    }
}

// DMA-based scatter: write data to all peers' IPC staging area via DataCopyPad.
// Split from the fused kernel to allow host-side sync between scatter and gather,
// which guarantees P2P write visibility without fragile in-kernel flag polling.
extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_scatter_dma_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;

    constexpr uint32_t DATA_UB_BYTES = 64 * 1024;
    constexpr uint32_t CHUNK_BYTES = DATA_UB_BYTES;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dataTBuf;
    pipe.InitBuffer(dataTBuf, DATA_UB_BYTES);
    AscendC::LocalTensor<uint8_t> dataLocal = dataTBuf.Get<uint8_t>();

    // Write data to all peers' IPC staging area via DMA
    for (int32_t dstRank = 0; dstRank < rankSize; ++dstRank) {
        auto remoteBase = reinterpret_cast<__gm__ int32_t*>(
            args->peerMems[dstRank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);
        auto remoteDst = remoteBase + static_cast<uint64_t>(rank) * rankSize * elementsPerPeer;
        auto localSrc = input + static_cast<uint64_t>(dstRank) * elementsPerPeer;

        auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
        auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(remoteDst);
        uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

        for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
            uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
            uint32_t alignedBytes = bytes & ~31U;

            if (alignedBytes > 0) {
                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(srcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(dstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            for (uint32_t i = alignedBytes; i < bytes; ++i) {
                dstBytes[offset + i] = srcBytes[offset + i];
            }
        }
    }

    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_SCATTER] = 4;
    }
}

// DMA-based gather: read data from local IPC staging area into output via DataCopyPad.
extern "C" __global__ __aicore__ void tilexr_all_to_all_ipc_gather_dma_kernel(
    GM_ADDR commArgsGM, GM_ADDR outputGM, GM_ADDR debugGM, int32_t elementsPerPeer)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;

    constexpr uint32_t DATA_UB_BYTES = 64 * 1024;
    constexpr uint32_t CHUNK_BYTES = DATA_UB_BYTES;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dataTBuf;
    pipe.InitBuffer(dataTBuf, DATA_UB_BYTES);
    AscendC::LocalTensor<uint8_t> dataLocal = dataTBuf.Get<uint8_t>();

    auto localBase = reinterpret_cast<__gm__ int32_t*>(
        args->peerMems[rank] + TILEXR_UDMA_DEMO_IPC_STAGING_OFFSET);

    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        auto localSrc = localBase + static_cast<uint64_t>(srcRank) * rankSize * elementsPerPeer;
        auto localDst = output + static_cast<uint64_t>(srcRank) * elementsPerPeer;

        auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(localSrc);
        auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(localDst);
        uint32_t payloadBytes = static_cast<uint32_t>(elementsPerPeer) * sizeof(int32_t);

        for (uint32_t offset = 0; offset < payloadBytes; offset += CHUNK_BYTES) {
            uint32_t bytes = (payloadBytes - offset < CHUNK_BYTES) ? (payloadBytes - offset) : CHUNK_BYTES;
            uint32_t alignedBytes = bytes & ~31U;

            if (alignedBytes > 0) {
                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(srcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dataLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(dstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, alignedBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, dataLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            for (uint32_t i = alignedBytes; i < bytes; ++i) {
                dstBytes[offset + i] = srcBytes[offset + i];
            }
        }
    }

    if (debug != nullptr) {
        debug[TILEXR_UDMA_DEMO_DEBUG_IPC_GATHER] = 4;
    }
}

void launch_tilexr_all_to_all_ipc_scatter_dma(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_scatter_dma_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_ipc_gather_dma(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_gather_dma_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerPeer);
}

void launch_tilexr_udma_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_udma_all_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, debug, elementsPerRank);
}

void launch_tilexr_udma_put_signal(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signal)
{
    tilexr_udma_put_signal_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, data, signals, debug, elementsPerRank, signal);
}

void launch_tilexr_udma_all_to_all(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements)
{
    tilexr_udma_all_to_all_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, outputByteOffset, inputElementOffset, chunkElements);
}

void launch_tilexr_all_to_all_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_ipc_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_ipc_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_plain_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_plain_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_plain_ipc_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer)
{
    tilexr_all_to_all_plain_ipc_gather_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerPeer);
}

void launch_tilexr_all_to_all_fused_ipc(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, int32_t round)
{
    tilexr_all_to_all_fused_ipc_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, round);
}

void launch_tilexr_all_reduce_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_all_reduce_ipc_scatter_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, debug, elementsPerRank);
}

void launch_tilexr_all_reduce_ipc_sum(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerRank)
{
    tilexr_all_reduce_ipc_sum_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, output, debug, elementsPerRank);
}

void launch_tilexr_udma_registered_smoke(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR local, GM_ADDR debug, uint32_t bytes, uint64_t signal)
{
    tilexr_udma_registered_smoke_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, local, debug, bytes, signal);
}

// ---------------------------------------------------------------------------
// Latency micro-kernels (testType 4 = P2P-only, testType 5 = DataCopy-only).
// These mirror the two halves of tilexr_udma_all_to_all_kernel so the P2P
// communication latency and the local DataCopy latency can be measured in
// isolation. Each block handles exactly one peer (block b -> peer b); for
// the P2P kernel the self-peer (peer == rank) is skipped so that block does
// no network work, and for the DataCopy kernel every block performs a local
// GM->UB->GM copy of the same payload size as one alltoall peer slice.
// ---------------------------------------------------------------------------

extern "C" __global__ __aicore__ void tilexr_udma_p2p_latency_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(outputByteOffset);
    }
    if (!enabled) {
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t payloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t bytes = static_cast<uint32_t>(payloadBytes);
    auto udmaInfo = TileXR::GetUDMAInfo(args);
    const uint32_t qpCount = udmaInfo->qpNum == 0 ? 1U : udmaInfo->qpNum;

    // One block per peer, skip self (peer == rank): no local copy here.
    for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
        if (peer == rank) {
            continue;
        }
        auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer + inputElementOffset;
        uint64_t remoteOffset = outputByteOffset +
            static_cast<uint64_t>(rank) * payloadBytes;
        uint32_t status = 0;
        for (uint32_t qpIdx = 0; qpIdx < qpCount; ++qpIdx) {
            uint32_t sliceOffset = 0;
            uint32_t sliceBytes = 0;
            TileXRUdmaDemoWeightedWqeSlice(udmaInfo, peer, bytes, qpCount, qpIdx, sliceOffset, sliceBytes);
            if (sliceBytes == 0) {
                continue;
            }
            auto sliceSrc = reinterpret_cast<__gm__ int32_t*>(
                reinterpret_cast<__gm__ uint8_t*>(localSrc) + sliceOffset);
            TileXR::UDMAPutNbiOnQp<int32_t>(
                args, peer, qpIdx, sliceSrc, remoteOffset + sliceOffset, sliceBytes);
        }
        for (uint32_t qpIdx = 0; qpIdx < qpCount; ++qpIdx) {
            uint32_t qpStatus = TileXR::UDMAQuietStatusOnQp(args, peer, qpIdx);
            if (qpStatus != 0 && status == 0) {
                status = qpStatus;
            }
        }
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
        }
    }
}

extern "C" __global__ __aicore__ void tilexr_datacopy_latency_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, int32_t chunkElements)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;

    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = 1; // always "enabled" path for datacopy
        debug[4] = elementsPerPeer;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t payloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t bytes = static_cast<uint32_t>(payloadBytes);

    // Each block performs the same GM->UB->GM self-copy that the alltoall
    // kernel uses for the self-peer slice. block b copies rank b's slice.
    for (int32_t whichRank = blockIdx; whichRank < rankSize; whichRank += blockNum) {
        auto selfSrc = input + static_cast<uint64_t>(whichRank) * elementsPerPeer;
        auto selfDst = output + static_cast<uint64_t>(whichRank) * effectiveChunkElements;
        constexpr uint32_t SELF_COPY_UB_BYTES = 64 * 1024;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> selfCopyTBuf;
        pipe.InitBuffer(selfCopyTBuf, SELF_COPY_UB_BYTES);
        AscendC::LocalTensor<uint8_t> selfCopyLocal = selfCopyTBuf.Get<uint8_t>();

        auto selfSrcBytes = reinterpret_cast<__gm__ uint8_t*>(selfSrc);
        auto selfDstBytes = reinterpret_cast<__gm__ uint8_t*>(selfDst);
        for (uint32_t offset = 0; offset < bytes; offset += SELF_COPY_UB_BYTES) {
            uint32_t copyBytes = (bytes - offset < SELF_COPY_UB_BYTES)
                ? (bytes - offset) : SELF_COPY_UB_BYTES;

            AscendC::GlobalTensor<uint8_t> srcGlobal;
            srcGlobal.SetGlobalBuffer(selfSrcBytes + offset);
            AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
            AscendC::DataCopyExtParams copyIn {1U, copyBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(selfCopyLocal, srcGlobal, copyIn, padIn);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            AscendC::GlobalTensor<uint8_t> dstGlobal;
            dstGlobal.SetGlobalBuffer(selfDstBytes + offset);
            AscendC::DataCopyExtParams copyOut {1U, copyBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(dstGlobal, selfCopyLocal, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void launch_tilexr_udma_p2p_latency(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements)
{
    tilexr_udma_p2p_latency_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, outputByteOffset, inputElementOffset, chunkElements);
}

void launch_tilexr_datacopy_latency(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, int32_t chunkElements)
{
    tilexr_datacopy_latency_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, chunkElements);
}


// ---------------------------------------------------------------------------
// Fused alltoall kernel -- single launch, loop×pass inside, no IPC fallback.
// input  : full-size GM (NOT registered), H2D primed once on host.
// output : full-size GM (NOT registered), filled inside kernel.
// udmaMem: registered chunk-sized relay buffer (rankSize*chunkElements*4B),
//          reused every pass (overwrite) -- fits the 32MB URMA single-shot
//          registration limit regardless of total data size.
// signals: registered region of rankSize uint64 slots.
//
// Per loop iter L, per pass p (chunkOffset = p*chunkElements):
//   1. SEND : block b (peer b!=rank) UDMAPutSignalNbi input[b-slice][offset]
//             -> peer b's udmaMem[rank slot] + signal (L*passCount+p+1).
//      SELF: DataCopyPad input[rank-slice][offset] -> output[rank-slice][offset].
//   2. QUIET: UDMAQuietStatus per owned peer (local WQE completion).
//   3. WAIT : spin until each peer wrote this pass's signal into our
//             signals[peer] (peer's data landed in our udmaMem[peer slot]).
//   4. RELAY: DataCopyPad udmaMem[peer slot] -> output[peer-slice][offset].
//   5. SyncAll; next pass overwrites udmaMem.
// Signal tokens are globally monotonic (loop*passCount+pass+1) so no
// re-zeroing is needed across loops/passes.
// ---------------------------------------------------------------------------

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_fused_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR udmaMemGM,
    GM_ADDR signalGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t udmaMemByteOffset, uint64_t signalByteOffsetBase,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto udmaMem = reinterpret_cast<__gm__ int32_t*>(udmaMemGM);
    auto signals = reinterpret_cast<__gm__ uint64_t*>(signalGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(udmaMemByteOffset);
    }
    if (!enabled) {
        for (uint32_t l = 0; l < loopCount; ++l) { AscendC::SyncAll(); }
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t peerPayloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t peerBytes = static_cast<uint32_t>(peerPayloadBytes);
    constexpr uint32_t RELAY_UB_BYTES = 64 * 1024;

    for (uint32_t loop = 0; loop < loopCount; ++loop) {
        for (uint32_t pass = 0; pass < passCount; ++pass) {
            const int32_t chunkOffset = static_cast<int32_t>(pass) * effectiveChunkElements;
            const int32_t chunkElem = (elementsPerPeer - chunkOffset < effectiveChunkElements)
                ? (elementsPerPeer - chunkOffset) : effectiveChunkElements;
            const uint32_t chunkBytes = static_cast<uint32_t>(
                static_cast<uint64_t>(chunkElem) * sizeof(int32_t));
            const uint64_t expectedSignal =
                static_cast<uint64_t>(loop) * static_cast<uint64_t>(passCount) +
                static_cast<uint64_t>(pass) + 1;

            // ---- 1. SEND + SELF ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) {
                    // self: input -> output directly (no P2P, no relay).
                    auto selfSrc = input + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset;
                    auto selfDst = output + static_cast<uint64_t>(rank) * elementsPerPeer + chunkOffset;
                    AscendC::TPipe pipe;
                    AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
                    pipe.InitBuffer(relayTBuf, RELAY_UB_BYTES);
                    AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();
                    auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(selfSrc);
                    auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(selfDst);
                    for (uint32_t off = 0; off < chunkBytes; off += RELAY_UB_BYTES) {
                        uint32_t cb = (chunkBytes - off < RELAY_UB_BYTES) ? (chunkBytes - off) : RELAY_UB_BYTES;
                        AscendC::GlobalTensor<uint8_t> srcGlobal; srcGlobal.SetGlobalBuffer(srcBytes + off);
                        AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                        AscendC::DataCopyExtParams copyIn {1U, cb, 0U, 0U, 0U};
                        AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::GlobalTensor<uint8_t> dstGlobal; dstGlobal.SetGlobalBuffer(dstBytes + off);
                        AscendC::DataCopyExtParams copyOut {1U, cb, 0U, 0U, 0U};
                        AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    continue;
                }
                // P2P: read input[peer slice][offset] -> peer's udmaMem[rank slot] + signal.
                auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset;
                uint64_t remoteOffset = udmaMemByteOffset +
                    static_cast<uint64_t>(rank) * peerPayloadBytes;
                uint64_t remoteSignalOffset = signalByteOffsetBase +
                    static_cast<uint64_t>(rank) * sizeof(uint64_t);
                TileXR::UDMAPutSignalNbi<int32_t>(args, peer, localSrc,
                    remoteOffset, chunkBytes, remoteSignalOffset, expectedSignal);
            }

            // ---- 2. QUIET ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) continue;
                uint32_t status = TileXR::UDMAQuietStatus(args, peer);
                if (debug != nullptr && loop == 0 && pass == 0 && peer < 16) {
                    debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
                }
            }

            // ---- 3. WAIT : each block spins on its owned peer's signal slot. ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) continue;
                __gm__ uint64_t* slot = signals + static_cast<uint64_t>(peer);
                while (*slot != expectedSignal) {
                }
            }
            AscendC::SyncAll();

            // ---- 4. RELAY : udmaMem[peer slot] -> output[peer slice][offset]. ----
            for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
                if (peer == rank) continue;
                auto relaySrc = udmaMem + static_cast<uint64_t>(peer) * effectiveChunkElements;
                auto relayDst = output + static_cast<uint64_t>(peer) * elementsPerPeer + chunkOffset;
                AscendC::TPipe pipe;
                AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
                pipe.InitBuffer(relayTBuf, RELAY_UB_BYTES);
                AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();
                auto srcBytes = reinterpret_cast<__gm__ uint8_t*>(relaySrc);
                auto dstBytes = reinterpret_cast<__gm__ uint8_t*>(relayDst);
                for (uint32_t off = 0; off < chunkBytes; off += RELAY_UB_BYTES) {
                    uint32_t cb = (chunkBytes - off < RELAY_UB_BYTES) ? (chunkBytes - off) : RELAY_UB_BYTES;
                    AscendC::GlobalTensor<uint8_t> srcGlobal; srcGlobal.SetGlobalBuffer(srcBytes + off);
                    AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                    AscendC::DataCopyExtParams copyIn {1U, cb, 0U, 0U, 0U};
                    AscendC::DataCopyPad(relayLocal, srcGlobal, copyIn, padIn);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::GlobalTensor<uint8_t> dstGlobal; dstGlobal.SetGlobalBuffer(dstBytes + off);
                    AscendC::DataCopyExtParams copyOut {1U, cb, 0U, 0U, 0U};
                    AscendC::DataCopyPad(dstGlobal, relayLocal, copyOut);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::SyncAll();
        }
    }
    AscendC::SyncAll();
}

extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_bigdata_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR udmaMemGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t dataOffset, uint64_t copyDoneOffset,
    uint64_t recvCopyDoneOffset, uint64_t remoteSendDoneOffset, uint64_t readySignalOffset,
    uint64_t ackSignalOffset, int32_t chunkElements, uint32_t passCount, uint32_t loopCount,
    uint64_t kernelLoopBase, uint32_t profileStage, uint32_t force35CoreFlag)
{
    if (profileStage > TILEXR_BIGDATA_PROFILE_STAGE_FULL) {
        profileStage = TILEXR_BIGDATA_PROFILE_STAGE_FULL;
    }
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto udmaMem = reinterpret_cast<__gm__ uint8_t*>(udmaMemGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);
    const int32_t blockIdx = AscendC::GetBlockIdx();
    const bool force35Core = (force35CoreFlag & 0x1U) != 0U;
    const bool remotePutOnly = (force35CoreFlag & 0x2U) != 0U;
    const int32_t ranksPerNode = BigDataNormalizeRanksPerNode(static_cast<int32_t>(force35CoreFlag >> 8U));

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = chunkElements;
    }
    if (!enabled) {
        return;
    }

    if (!BigDataValidTopology(rankSize, ranksPerNode)) {
        if (blockIdx == 0 && debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE] =
                TILEXR_UDMA_DEMO_READY_TIMEOUT_STATUS;
        }
        return;
    }
    const bool use35Core = BigDataUse35Core(rankSize, force35Core, ranksPerNode);
    const uint32_t shardCount = BigDataShardCount(rankSize, force35Core, ranksPerNode);
    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t chunkBytesPerPeer = BigDataChunkBytesPerPeer(use35Core, effectiveChunkElements);
    const uint64_t sendDataOffset = dataOffset;
    const uint64_t recvDataOffset =
        sendDataOffset +
        static_cast<uint64_t>(use35Core ? passCount : TILEXR_UDMA_DEMO_BIGDATA_PINGPONG_SLOTS) *
        static_cast<uint64_t>(rankSize > 1 ? rankSize - 1 : 1) * chunkBytesPerPeer;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> relayTBuf;
    pipe.InitBuffer(relayTBuf, TILEXR_UDMA_DEMO_BIGDATA_RELAY_UB_PINGPONG_BYTES);
    AscendC::LocalTensor<uint8_t> relayLocal = relayTBuf.Get<uint8_t>();
    if (!use35Core) {
        const int32_t workerGroup =
            blockIdx / static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER);
        const int32_t role =
            blockIdx % static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER);
        if (workerGroup >= rankSize || role < 0 ||
            role >= static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_CORES_PER_PEER)) {
            BigDataKernelExitBarrier();
            return;
        }
        for (uint32_t loop = 0; loop < loopCount; ++loop) {
            for (uint32_t pass = 0; pass < passCount; ++pass) {
                const int32_t peer = workerGroup;
                BigDataRunRoleForPeer(peer, role, rank, rankSize, args, input, output,
                    udmaMem, debug, elementsPerPeer, effectiveChunkElements, passCount,
                    loop, pass, kernelLoopBase, profileStage, use35Core, shardCount, sendDataOffset, recvDataOffset,
                    copyDoneOffset, recvCopyDoneOffset, readySignalOffset, ackSignalOffset,
                    chunkBytesPerPeer, relayLocal);
            }
        }
        BigDataKernelExitBarrier();
        return;
    }

    const uint32_t activeBlockDim = remotePutOnly ?
        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_PUT_ONLY_BLOCK_DIM :
        TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_BLOCK_DIM;
    if (blockIdx >= static_cast<int32_t>(activeBlockDim)) {
        BigDataKernelExitBarrier();
        return;
    }
    const bool isCopyCore =
        blockIdx < static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES);
    const bool isRemoteSendPrimaryCore =
        blockIdx == static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_PRIMARY_CORE);
    const bool isRemoteSendSecondaryCore =
        blockIdx == static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_REMOTE_SEND_SECONDARY_CORE);
    const bool isLocalSendCore =
        blockIdx == static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_LOCAL_SEND_CORE);
    const bool isRecvCore =
        blockIdx >= static_cast<int32_t>(TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORE_BASE) &&
        blockIdx < static_cast<int32_t>(activeBlockDim);
    const uint32_t copyShard = static_cast<uint32_t>(blockIdx);
    const uint32_t recvShard =
        static_cast<uint32_t>(blockIdx) - TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_RECV_CORE_BASE;
    const int32_t taskCount = BigDataTaskCount(rankSize, force35Core, ranksPerNode);
    const int32_t remoteTaskCount = rankSize - ranksPerNode;
    const int32_t sendTaskCount = BigDataRemotePutOnlySendTaskCount(remoteTaskCount);

    if (remotePutOnly && profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_FRAMEWORK) {
        BigDataKernelExitBarrier();
        return;
    }

    for (uint32_t loop = 0; loop < loopCount; ++loop) {
        for (uint32_t pass = 0; pass < passCount; ++pass) {
            if (remotePutOnly) {
                for (int32_t sendTask = blockIdx; sendTask < sendTaskCount;
                     sendTask += static_cast<int32_t>(activeBlockDim)) {
                    if (profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_LOOP) {
                        continue;
                    }
                    const int32_t remoteIndex = BigDataRemotePutOnlySendTaskRemoteIndex(sendTask, remoteTaskCount);
                    const int32_t peer = BigDataRemotePeerForwardAt(rank, rankSize, remoteIndex, ranksPerNode);
                    if (profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_PEER) {
                        continue;
                    }
                    const uint32_t segmentId = BigDataRemotePutOnlySendTaskSegment(sendTask, remoteTaskCount);
                    const uint32_t qpIdx = BigDataRemotePutOnlySegmentQp(args, segmentId);
                    if (profileStage <= TILEXR_BIGDATA_REMOTE_PUT_STAGE_QP) {
                        continue;
                    }
                    BigDataRemotePutOnlySendWorker(peer, segmentId, qpIdx,
                        rank, rankSize, args, input, debug,
                        elementsPerPeer, effectiveChunkElements, passCount, loop, pass,
                        kernelLoopBase, profileStage, shardCount, recvDataOffset,
                        chunkBytesPerPeer, ranksPerNode, relayLocal);
                }
                BigDataKernelExitBarrier();
                if (profileStage > TILEXR_BIGDATA_REMOTE_PUT_STAGE_POST) {
                    const int32_t remotePutOnlyCheckIndex = BigDataRemotePutOnlyCheckIndex(blockIdx);
                    if (remotePutOnlyCheckIndex >= 0 && remotePutOnlyCheckIndex < remoteTaskCount) {
                        BigDataRemotePutOnlyCheckWorker(
                            remotePutOnlyCheckIndex, rank, rankSize, args, udmaMem, debug,
                            elementsPerPeer, effectiveChunkElements, passCount, loop, pass,
                            kernelLoopBase, recvDataOffset, ackSignalOffset, chunkBytesPerPeer,
                            shardCount, ranksPerNode, relayLocal);
                    }
                }
                BigDataKernelExitBarrier();
                continue;
            }
            if (isCopyCore) {
                BigDataRunSelfCopyShard(rank, rankSize, input, output,
                    elementsPerPeer, effectiveChunkElements, pass,
                    copyShard, shardCount, relayLocal);
            }

            for (int32_t taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
                int32_t peer = -1;
                bool isLocalPeer = false;
                if (!BigDataMergedPeerTaskAt(
                        rank, rankSize, taskIndex, force35Core, ranksPerNode, peer, isLocalPeer)) {
                    continue;
                }

                if (isCopyCore) {
                    BigDataCopyPeerWorker(peer, rank, rankSize, args, input, output,
                        udmaMem, debug, elementsPerPeer, effectiveChunkElements,
                        passCount, loop, pass, kernelLoopBase, profileStage,
                        use35Core, copyShard, shardCount, sendDataOffset, copyDoneOffset, ackSignalOffset,
                        chunkBytesPerPeer, relayLocal);
                    if (profileStage > TILEXR_BIGDATA_PROFILE_STAGE_SEND_COPY) {
                        const uint64_t globalPass = BigDataGlobalPassIndex(kernelLoopBase, passCount, loop, pass);
                        const uint64_t token = BigDataPassToken(globalPass);
                        const uint32_t slot = BigDataDataSlot(globalPass, pass, true);
                        if (!isLocalPeer && copyShard == 0U) {
                            (void)BigDataPublishCopyReadyRange(
                                udmaMem, debug, copyDoneOffset, remoteSendDoneOffset,
                                slot, rankSize, shardCount, peer,
                                0U, TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END,
                                TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_PRIMARY,
                                token, loop, pass, relayLocal);
                        }
                        if (!isLocalPeer &&
                            copyShard == TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_AGGREGATOR) {
                            (void)BigDataPublishCopyReadyRange(
                                udmaMem, debug, copyDoneOffset, remoteSendDoneOffset,
                                slot, rankSize, shardCount, peer,
                                TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END,
                                TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES,
                                TILEXR_UDMA_DEMO_BIGDATA_REMOTE_COPY_READY_SECONDARY,
                                token, loop, pass, relayLocal);
                        }
                        if (isLocalPeer && copyShard == 0U) {
                            (void)BigDataPublishCopyReadyRange(
                                udmaMem, debug, copyDoneOffset, remoteSendDoneOffset,
                                slot, rankSize, shardCount, peer,
                                0U, TILEXR_UDMA_DEMO_BIGDATA_MULTINODE_COPY_CORES,
                                TILEXR_UDMA_DEMO_BIGDATA_LOCAL_COPY_READY,
                                token, loop, pass, relayLocal);
                        }
                    }
                }
                if (!isLocalPeer && isRemoteSendPrimaryCore) {
                    BigDataRemoteSendSegmentWorker(peer,
                        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SEGMENT,
                        rank, rankSize, args, udmaMem, debug,
                        elementsPerPeer, effectiveChunkElements, passCount, loop, pass,
                        kernelLoopBase, profileStage, shardCount, sendDataOffset, recvDataOffset,
                        copyDoneOffset, remoteSendDoneOffset, readySignalOffset,
                        chunkBytesPerPeer, ranksPerNode, relayLocal);
                }
                if (!isLocalPeer && isRemoteSendSecondaryCore) {
                    BigDataRemoteSendSegmentWorker(peer,
                        TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_SEGMENT,
                        rank, rankSize, args, udmaMem, debug,
                        elementsPerPeer, effectiveChunkElements, passCount, loop, pass,
                        kernelLoopBase, profileStage, shardCount, sendDataOffset, recvDataOffset,
                        copyDoneOffset, remoteSendDoneOffset, readySignalOffset,
                        chunkBytesPerPeer, ranksPerNode, relayLocal);
                }
                if (isLocalPeer && isLocalSendCore) {
                    BigDataSendPeerWorker(peer, rank, rankSize, args, udmaMem, debug,
                        elementsPerPeer, effectiveChunkElements, passCount, loop, pass,
                        kernelLoopBase, profileStage, shardCount, sendDataOffset, recvDataOffset,
                        copyDoneOffset, readySignalOffset, chunkBytesPerPeer, true,
                        remoteSendDoneOffset, relayLocal);
                }
                if (isRecvCore) {
                    BigDataRecvPeerWorker(peer, rank, rankSize, args, output, udmaMem, debug,
                        elementsPerPeer, effectiveChunkElements, passCount, loop, pass,
                        kernelLoopBase, profileStage, use35Core, recvShard, shardCount, recvDataOffset,
                        recvCopyDoneOffset, readySignalOffset, ackSignalOffset,
                        chunkBytesPerPeer, relayLocal);
                }
            }
        }
    }
    BigDataKernelExitBarrier();
}

void launch_tilexr_udma_all_to_all_bigdata(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR udmaMem, GM_ADDR debug, int32_t elementsPerPeer,
    uint64_t dataOffset, uint64_t copyDoneOffset,
    uint64_t recvCopyDoneOffset, uint64_t remoteSendDoneOffset, uint64_t readySignalOffset,
    uint64_t ackSignalOffset, int32_t chunkElements, uint32_t passCount, uint32_t loopCount,
    uint64_t kernelLoopBase, uint32_t profileStage, uint32_t force35Core)
{
    tilexr_udma_all_to_all_bigdata_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, udmaMem, debug, elementsPerPeer,
        dataOffset, copyDoneOffset, recvCopyDoneOffset, remoteSendDoneOffset, readySignalOffset,
        ackSignalOffset, chunkElements, passCount, loopCount, kernelLoopBase, profileStage, force35Core);
}

void launch_tilexr_udma_all_to_all_fused(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR udmaMem, GM_ADDR signal, GM_ADDR debug, int32_t elementsPerPeer,
    uint64_t udmaMemByteOffset, uint64_t signalByteOffsetBase,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount)
{
    tilexr_udma_all_to_all_fused_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, udmaMem, signal, debug, elementsPerPeer,
        udmaMemByteOffset, signalByteOffsetBase, chunkElements, passCount, loopCount);
}
