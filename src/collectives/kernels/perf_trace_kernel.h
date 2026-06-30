#ifndef TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H
#define TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H

#include "comm_args.h"
#include "kernel_operator.h"
#include "datacopy_gm2gm.h"
#include "tilexr_perf_trace.h"

namespace TileXR {

struct TileXRPerfStageToken {
    uint64_t startCycle = 0;
};

constexpr int64_t TILEXR_PERF_TRACE_STATS_UB_OFFSET = 195616;

#if defined(TILEXR_COLLECTIVES_ENABLE_PROFILING)

__attribute__((always_inline)) inline __aicore__ bool TileXRPerfTraceEnabled(GM_ADDR trace)
{
    return trace != nullptr;
}

__attribute__((always_inline)) inline __aicore__ __gm__ TileXRPerfCoreStageStats *TileXRPerfStatsSlot(
    GM_ADDR trace, uint32_t rank, uint32_t core, uint32_t stage)
{
    if (trace == nullptr) {
        return nullptr;
    }

    const size_t slot = PerfTraceStatsOffset(rank, core, stage, GetBlockNum(), TILEXR_PERF_STAGE_COUNT);
    return reinterpret_cast<__gm__ TileXRPerfCoreStageStats *>(trace + TILEXR_PERF_TRACE_STATS_OFFSET) + slot;
}

__attribute__((always_inline)) inline __aicore__ TileXRPerfStageToken TileXRPerfStageBegin(
    GM_ADDR trace, PerfStageId stage, PerfBarrierPolicy policy)
{
    TileXRPerfStageToken token {};
    if (trace == nullptr) {
        return token;
    }
    if (policy == PerfBarrierPolicy::BARRIERED) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    (void)stage;
    token.startCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    return token;
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfAccumulateDuration(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, uint64_t startCycle, uint64_t endCycle,
    __ubuf__ TileXRPerfCoreStageStats *statsUB)
{
    if (endCycle < startCycle) {
        return;
    }

    const uint32_t stageId = static_cast<uint32_t>(stage);
    __gm__ TileXRPerfCoreStageStats *slot = TileXRPerfStatsSlot(trace, rank, core, stageId);
    if (slot == nullptr) {
        return;
    }

    CpGM2UB(reinterpret_cast<__ubuf__ uint8_t *>(statsUB),
        reinterpret_cast<__gm__ uint8_t *>(slot), sizeof(TileXRPerfCoreStageStats));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    const uint64_t duration = endCycle - startCycle;
    statsUB->rank = rank;
    statsUB->core = core;
    statsUB->stageId = stageId;
    statsUB->reserved = 0;
    if (statsUB->count == 0) {
        statsUB->minCycles = duration;
        statsUB->maxCycles = duration;
        statsUB->firstStartCycle = startCycle;
    } else {
        if (duration < statsUB->minCycles) {
            statsUB->minCycles = duration;
        }
        if (duration > statsUB->maxCycles) {
            statsUB->maxCycles = duration;
        }
        if (startCycle < statsUB->firstStartCycle) {
            statsUB->firstStartCycle = startCycle;
        }
    }
    statsUB->count += 1;
    statsUB->sumCycles += duration;
    if (endCycle > statsUB->lastEndCycle) {
        statsUB->lastEndCycle = endCycle;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    CpUB2GM(reinterpret_cast<__gm__ uint8_t *>(slot),
        reinterpret_cast<__ubuf__ uint8_t *>(statsUB), sizeof(TileXRPerfCoreStageStats));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfAccumulateDuration(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, uint64_t startCycle, uint64_t endCycle)
{
    TileXRPerfAccumulateDuration(
        trace, rank, core, stage, startCycle, endCycle,
        reinterpret_cast<__ubuf__ TileXRPerfCoreStageStats *>(TILEXR_PERF_TRACE_STATS_UB_OFFSET));
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfStageEnd(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, TileXRPerfStageToken token,
    PerfBarrierPolicy policy, __ubuf__ TileXRPerfCoreStageStats *statsUB)
{
    if (trace == nullptr) {
        return;
    }
    if (policy == PerfBarrierPolicy::BARRIERED || policy == PerfBarrierPolicy::END_BARRIER_ONLY) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    const uint64_t endCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    TileXRPerfAccumulateDuration(trace, rank, core, stage, token.startCycle, endCycle, statsUB);
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfStageEnd(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, TileXRPerfStageToken token,
    PerfBarrierPolicy policy)
{
    TileXRPerfStageEnd(
        trace, rank, core, stage, token, policy,
        reinterpret_cast<__ubuf__ TileXRPerfCoreStageStats *>(TILEXR_PERF_TRACE_STATS_UB_OFFSET));
}

#else

__attribute__((always_inline)) inline __aicore__ bool TileXRPerfTraceEnabled(GM_ADDR trace)
{
    (void)trace;
    return false;
}

__attribute__((always_inline)) inline __aicore__ __gm__ TileXRPerfCoreStageStats *TileXRPerfStatsSlot(
    GM_ADDR trace, uint32_t rank, uint32_t core, uint32_t stage)
{
    (void)trace;
    (void)rank;
    (void)core;
    (void)stage;
    return nullptr;
}

__attribute__((always_inline)) inline __aicore__ TileXRPerfStageToken TileXRPerfStageBegin(
    GM_ADDR trace, PerfStageId stage, PerfBarrierPolicy policy)
{
    (void)trace;
    (void)stage;
    (void)policy;
    return TileXRPerfStageToken {};
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfAccumulateDuration(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, uint64_t startCycle, uint64_t endCycle)
{
    (void)trace;
    (void)rank;
    (void)core;
    (void)stage;
    (void)startCycle;
    (void)endCycle;
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfAccumulateDuration(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, uint64_t startCycle, uint64_t endCycle,
    __ubuf__ TileXRPerfCoreStageStats *statsUB)
{
    (void)statsUB;
    TileXRPerfAccumulateDuration(trace, rank, core, stage, startCycle, endCycle);
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfStageEnd(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, TileXRPerfStageToken token,
    PerfBarrierPolicy policy)
{
    (void)trace;
    (void)rank;
    (void)core;
    (void)stage;
    (void)token;
    (void)policy;
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfStageEnd(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, TileXRPerfStageToken token,
    PerfBarrierPolicy policy, __ubuf__ TileXRPerfCoreStageStats *statsUB)
{
    (void)statsUB;
    TileXRPerfStageEnd(trace, rank, core, stage, token, policy);
}

#endif

} // namespace TileXR

#endif // TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H
