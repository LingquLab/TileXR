#ifndef TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H
#define TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H

#include "comm_args.h"
#include "kernel_operator.h"
#include "tilexr_perf_trace.h"

namespace TileXR {

struct TileXRPerfStageToken {
    uint64_t startCycle = 0;
};

#if defined(TILEXR_COLLECTIVES_ENABLE_PROFILING)

__attribute__((always_inline)) inline __aicore__ __gm__ TileXRPerfCoreStageStats *TileXRPerfStatsSlot(
    GM_ADDR trace, uint32_t rank, uint32_t core, uint32_t stage)
{
    if (trace == nullptr) {
        return nullptr;
    }

    __gm__ TileXRPerfTraceHeader *header = reinterpret_cast<__gm__ TileXRPerfTraceHeader *>(trace);
    const size_t slot = PerfTraceStatsOffset(rank, core, stage, header->maxCoreCount, header->stageCount);
    return reinterpret_cast<__gm__ TileXRPerfCoreStageStats *>(trace + header->statsOffset) + slot;
}

__attribute__((always_inline)) inline __aicore__ TileXRPerfStageToken TileXRPerfStageBegin(
    GM_ADDR trace, PerfStageId stage, PerfBarrierPolicy policy)
{
    (void)stage;
    TileXRPerfStageToken token {};
    if (trace == nullptr) {
        return token;
    }
    if (policy == PerfBarrierPolicy::BARRIERED) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    token.startCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    return token;
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfAccumulateDuration(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, uint64_t startCycle, uint64_t endCycle)
{
    if (endCycle < startCycle) {
        return;
    }

    const uint32_t stageId = static_cast<uint32_t>(stage);
    __gm__ TileXRPerfCoreStageStats *slot = TileXRPerfStatsSlot(trace, rank, core, stageId);
    if (slot == nullptr) {
        return;
    }

    const uint64_t duration = endCycle - startCycle;
    slot->rank = rank;
    slot->core = core;
    slot->stageId = stageId;
    if (slot->count == 0) {
        slot->minCycles = duration;
        slot->maxCycles = duration;
        slot->firstStartCycle = startCycle;
    } else {
        if (duration < slot->minCycles) {
            slot->minCycles = duration;
        }
        if (duration > slot->maxCycles) {
            slot->maxCycles = duration;
        }
        if (startCycle < slot->firstStartCycle) {
            slot->firstStartCycle = startCycle;
        }
    }
    slot->count += 1;
    slot->sumCycles += duration;
    if (endCycle > slot->lastEndCycle) {
        slot->lastEndCycle = endCycle;
    }
}

__attribute__((always_inline)) inline __aicore__ void TileXRPerfStageEnd(
    GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage, TileXRPerfStageToken token,
    PerfBarrierPolicy policy)
{
    if (trace == nullptr) {
        return;
    }
    if (policy == PerfBarrierPolicy::BARRIERED || policy == PerfBarrierPolicy::END_BARRIER_ONLY) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    const uint64_t endCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    TileXRPerfAccumulateDuration(trace, rank, core, stage, token.startCycle, endCycle);
}

#else

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

#endif

} // namespace TileXR

#endif // TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H
