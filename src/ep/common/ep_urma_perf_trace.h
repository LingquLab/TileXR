#ifndef TILEXR_EP_COMMON_EP_URMA_PERF_TRACE_H
#define TILEXR_EP_COMMON_EP_URMA_PERF_TRACE_H

#include "kernel_operator.h"
#include "tilexr_perf_trace.h"

namespace TileXR {

constexpr uint32_t TILEXR_PERF_TRACE_LOCAL_MAX_STAGE_COUNT = 27;
constexpr uint32_t TILEXR_PERF_TRACE_LOCAL_STATS_UB_END = 195712;
constexpr uint32_t TILEXR_PERF_TRACE_LOCAL_STATS_UB_OFFSET =
    TILEXR_PERF_TRACE_LOCAL_STATS_UB_END -
    TILEXR_PERF_TRACE_LOCAL_MAX_STAGE_COUNT * sizeof(TileXRPerfCoreStageStats);
constexpr uint32_t TILEXR_PERF_TRACE_MIN_UB_BYTES = 192 * 1024;

static_assert(TILEXR_PERF_TRACE_LOCAL_STATS_UB_END <= TILEXR_PERF_TRACE_MIN_UB_BYTES,
    "local perf stats must fit the minimum supported AIV UB");
static_assert(TILEXR_PERF_TRACE_LOCAL_STATS_UB_OFFSET % 32 == 0,
    "local perf stats must remain 32-byte aligned");

#if defined(TILEXR_EP_ENABLE_PROFILING)

__attribute__((always_inline)) inline __aicore__ bool TileXRPerfTraceEnabled(GM_ADDR trace)
{
    return trace != nullptr;
}

__attribute__((always_inline)) inline __aicore__ uint64_t TileXRPerfCycleNow(GM_ADDR trace)
{
    return trace == nullptr ? 0 : static_cast<uint64_t>(AscendC::GetSystemCycle());
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalStatsInit(
    GM_ADDR trace, __ubuf__ TileXRPerfCoreStageStats *statsUB, uint32_t rank, uint32_t core,
    uint32_t stageCount)
{
    if (trace == nullptr || stageCount > TILEXR_PERF_TRACE_LOCAL_MAX_STAGE_COUNT) {
        return;
    }
    for (uint32_t stage = 0; stage < stageCount; ++stage) {
        statsUB[stage].rank = rank;
        statsUB[stage].core = core;
        statsUB[stage].stageId = stage;
        statsUB[stage].reserved = 0;
        statsUB[stage].count = 0;
        statsUB[stage].sumCycles = 0;
        statsUB[stage].minCycles = 0;
        statsUB[stage].maxCycles = 0;
        statsUB[stage].firstStartCycle = 0;
        statsUB[stage].lastEndCycle = 0;
        statsUB[stage].aux0 = 0;
        statsUB[stage].aux1 = 0;
        statsUB[stage].aux2 = 0;
        statsUB[stage].aux3 = 0;
    }
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalRecord(
    GM_ADDR trace, __ubuf__ TileXRPerfCoreStageStats *statsUB, uint32_t stageCount,
    uint32_t stage, uint64_t startCycle, uint64_t endCycle)
{
    if (trace == nullptr || stageCount > TILEXR_PERF_TRACE_LOCAL_MAX_STAGE_COUNT ||
        stage >= stageCount || endCycle < startCycle) {
        return;
    }
    __ubuf__ TileXRPerfCoreStageStats *stat = statsUB + stage;
    const uint64_t duration = endCycle - startCycle;
    if (stat->count == 0) {
        stat->minCycles = duration;
        stat->maxCycles = duration;
        stat->firstStartCycle = startCycle;
    } else {
        if (duration < stat->minCycles) {
            stat->minCycles = duration;
        }
        if (duration > stat->maxCycles) {
            stat->maxCycles = duration;
        }
        if (startCycle < stat->firstStartCycle) {
            stat->firstStartCycle = startCycle;
        }
    }
    stat->count += 1;
    stat->sumCycles += duration;
    if (endCycle > stat->lastEndCycle) {
        stat->lastEndCycle = endCycle;
    }
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalAddAux(
    GM_ADDR trace, __ubuf__ TileXRPerfCoreStageStats *statsUB, uint32_t stageCount,
    uint32_t stage, uint32_t auxIndex, uint64_t value)
{
    if (trace == nullptr || stageCount > TILEXR_PERF_TRACE_LOCAL_MAX_STAGE_COUNT || stage >= stageCount) {
        return;
    }
    __ubuf__ TileXRPerfCoreStageStats *stat = statsUB + stage;
    if (auxIndex == 0) {
        stat->aux0 += value;
    } else if (auxIndex == 1) {
        stat->aux1 += value;
    } else if (auxIndex == 2) {
        stat->aux2 += value;
    } else if (auxIndex == 3) {
        stat->aux3 += value;
    }
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalStatsFlush(
    GM_ADDR trace, uint32_t rank, uint32_t core, uint32_t maxCoreCount, uint32_t stageCount,
    __ubuf__ TileXRPerfCoreStageStats *statsUB)
{
    if (trace == nullptr || stageCount == 0 ||
        stageCount > TILEXR_PERF_TRACE_LOCAL_MAX_STAGE_COUNT) {
        return;
    }
    const size_t firstSlot = PerfTraceStatsOffset(rank, core, 0, maxCoreCount, stageCount);
    __gm__ uint8_t *dstAddr = reinterpret_cast<__gm__ uint8_t *>(
        reinterpret_cast<__gm__ TileXRPerfCoreStageStats *>(
            trace + TILEXR_PERF_TRACE_STATS_OFFSET) + firstSlot);
    AscendC::LocalTensor<uint8_t> local;
    AscendC::TBuffAddr localAddr;
    localAddr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    localAddr.bufferAddr = reinterpret_cast<uint64_t>(statsUB);
    local.SetAddr(localAddr);
    AscendC::GlobalTensor<uint8_t> dst;
    dst.SetGlobalBuffer(dstAddr);
    AscendC::DataCopyExtParams params {
        1, stageCount * static_cast<uint32_t>(sizeof(TileXRPerfCoreStageStats)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(dst, local, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    (void)statsUB;
}

#else

__attribute__((always_inline)) inline __aicore__ bool TileXRPerfTraceEnabled(GM_ADDR trace)
{
    (void)trace;
    return false;
}

__attribute__((always_inline)) inline __aicore__ uint64_t TileXRPerfCycleNow(GM_ADDR trace)
{
    (void)trace;
    return 0;
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalStatsInit(
    GM_ADDR, __ubuf__ TileXRPerfCoreStageStats *, uint32_t, uint32_t, uint32_t)
{
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalRecord(
    GM_ADDR, __ubuf__ TileXRPerfCoreStageStats *, uint32_t, uint32_t, uint64_t, uint64_t)
{
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalAddAux(
    GM_ADDR, __ubuf__ TileXRPerfCoreStageStats *, uint32_t, uint32_t, uint32_t, uint64_t)
{
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfLocalStatsFlush(
    GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, __ubuf__ TileXRPerfCoreStageStats *)
{
}

#endif

} // namespace TileXR

#endif // TILEXR_EP_COMMON_EP_URMA_PERF_TRACE_H
