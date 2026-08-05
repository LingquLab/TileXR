/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ALLTOALL_GROUP_QUEUE_DIAG_H
#define TILEXR_UDMA_ALLTOALL_GROUP_QUEUE_DIAG_H

#include <cstdint>

namespace TileXR {
namespace Demo {

constexpr uint32_t kAllToAllGroupQueueDiagCoreCount = 64U;
constexpr uint32_t kAllToAllGroupQueueDiagSqeWords = 16U;
constexpr uint32_t kAllToAllGroupQueueDiagModeStandard = 1U;
constexpr uint32_t kAllToAllGroupQueueDiagModeSimt = 2U;

struct AllToAllGroupQueueDiagRecord {
    uint32_t valid;
    uint32_t frozen;
    uint32_t mode;
    uint32_t core;
    uint32_t invocation;
    uint32_t group;
    uint32_t pass;
    int32_t peer;
    uint32_t qp;
    uint32_t wqeCount;
    uint32_t headBefore;
    uint32_t reservedBegin;
    uint32_t reservedEnd;
    uint32_t headAfter;
    uint32_t wqeCntBefore;
    uint32_t expectedWqeCnt;
    uint32_t wqeCntAfter;
    uint32_t doorbellWritten;
    uint32_t tailAtQuietBegin;
    uint32_t expectedTail;
    uint32_t completedTail;
    uint32_t cqeSlot;
    uint32_t rawCqeWord;
    uint32_t pollCount;
    uint32_t quietStatus;
    uint32_t reserved;
    uint32_t sqNumber;
    uint32_t cqNumber;
    uint32_t sqDepth;
    uint32_t cqDepth;
    uint64_t sqBufAddr;
    uint64_t headAddr;
    uint64_t wqeCntAddr;
    uint64_t sqDbAddr;
    uint64_t cqBufAddr;
    uint64_t cqTailAddr;
    uint64_t cqDbAddr;
    uint64_t quietBeginCycle;
    uint32_t payloadSqe[kAllToAllGroupQueueDiagSqeWords];
    uint32_t signalSqe[kAllToAllGroupQueueDiagSqeWords];
};

static_assert(sizeof(AllToAllGroupQueueDiagRecord) == 312U,
    "grouped queue diagnostic record ABI changed");

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_ALLTOALL_GROUP_QUEUE_DIAG_H
