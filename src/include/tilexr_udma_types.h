/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_TYPES_H
#define TILEXR_UDMA_TYPES_H

#include <cstdint>

namespace TileXR {

constexpr uint32_t TILEXR_UDMA_CQ_DEPTH = 16384;
constexpr uint32_t TILEXR_UDMA_SQ_DEPTH = 4096;
constexpr uint32_t TILEXR_UDMA_MAX_SQE_BB_NUM = 4;
constexpr uint32_t TILEXR_UDMA_SQ_BB_COUNT = TILEXR_UDMA_SQ_DEPTH * TILEXR_UDMA_MAX_SQE_BB_NUM;
constexpr uint32_t TILEXR_UDMA_NUM_CQE_PER_POLL = 100;
constexpr uint32_t TILEXR_UDMA_CACHE_LINE_SIZE = 64;
constexpr uint32_t TILEXR_UDMA_MAX_RETRY_TIMES = 1000000;
constexpr uint32_t TILEXR_UDMA_MAX_WQE_BYTES = 128U * 1024U * 1024U;

constexpr uint32_t UDMAWqeChunkCount(uint64_t byteCount)
{
    return byteCount == 0 ? 0 :
        static_cast<uint32_t>((byteCount + TILEXR_UDMA_MAX_WQE_BYTES - 1U) / TILEXR_UDMA_MAX_WQE_BYTES);
}

enum class UDMAOpcode : uint32_t {
    WRITE = 3,
    WRITE_WITH_NOTIFY = 5,
    READ = 6,
};

enum class UDMADBMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB = 1,
};

struct UDMAMemInfo {
    bool tokenValueValid;
    uint32_t rmtJettyType : 2;
    uint8_t targetHint;
    uint32_t tpn;
    uint32_t tid;
    uint32_t rmtTokenValue;
    uint32_t len;
    uint64_t addr;
    uint64_t eidAddr;
};

struct UDMAWQCtx {
    uint32_t wqn;
    uint64_t bufAddr;
    uint32_t baseBkShift;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    UDMADBMode dbMode;
    uint64_t dbAddr;
    uint32_t sl;
    uint64_t wqeCntAddr;
    uint64_t amoAddr;
};

struct UDMACQCtx {
    uint32_t cqn;
    uint64_t bufAddr;
    uint32_t baseBkShift;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    UDMADBMode dbMode;
    uint64_t dbAddr;
};

struct UDMAInfo {
    uint32_t qpNum;
    uint64_t sqPtr;
    uint64_t rqPtr;
    uint64_t scqPtr;
    uint64_t rcqPtr;
    uint64_t memPtr;
    uint64_t qpWeightPtr;
};

struct UDMASqeCtx {
    uint32_t sqeBbIdx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t tokenEn : 1;
    uint32_t rmtJettyType : 2;
    uint32_t owner : 1;
    uint32_t targetHint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inlineMsgLen : 10;
    uint32_t tpId : 24;
    uint32_t sgeNum : 8;
    uint32_t rmtJettyOrSegId : 20;
    uint32_t rsv2 : 12;
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    uint32_t rmtTokenValue;
    uint32_t udfType : 8;
    uint32_t reduceDataType : 4;
    uint32_t reduceOpcode : 4;
    uint32_t rsv3 : 16;
    uint32_t rmtAddrLOrTokenId;
    uint32_t rmtAddrHOrTokenValue;
};

struct UDMASgeCtx {
    uint32_t len;
    uint32_t tokenId;
    uint64_t va;
};

struct UDMANotifyCtx {
    uint32_t notifyTokenId : 20;
    uint32_t rsv : 12;
    uint32_t notifyTokenValue;
    uint32_t notifyAddrL;
    uint32_t notifyAddrH;
    uint32_t notifyDataL;
    uint32_t notifyDataH;
    uint32_t rsv2[2];
};

struct UDMACqeCtx {
    uint32_t sR : 1;
    uint32_t isJetty : 1;
    uint32_t owner : 1;
    uint32_t inlineEn : 1;
    uint32_t opcode : 3;
    uint32_t fd : 1;
    uint32_t rsv : 8;
    uint32_t substatus : 8;
    uint32_t status : 8;
    uint32_t entryIdx : 16;
    uint32_t localNumL : 16;
    uint32_t localNumH : 4;
    uint32_t rmtIdx : 20;
    uint32_t rsv1 : 8;
    uint32_t tpn : 24;
    uint32_t rsv2 : 8;
    uint32_t byteCnt;
    uint32_t userDataL;
    uint32_t userDataH;
    uint32_t rmtEid[4];
    uint32_t dataL;
    uint32_t dataH;
    uint32_t inlineData[3];
};

} // namespace TileXR

#endif // TILEXR_UDMA_TYPES_H
